import torch
import torch.distributed as dist
from megatron import get_args
from megatron import mpu
from megatron.checkpointing import load_dualencoder_checkpoint
from megatron.data.samplers import DistributedBatchSampler
from megatron.data.emdr2_index import detach, OpenRetreivalDataStore
from megatron.model.dualencoder_model import dualencoder_model_provider
from megatron.training import get_model
from megatron.mpu.initialize import get_data_parallel_group
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from collections import OrderedDict
from megatron.data.orqa_wiki_dataset_pretokenized import get_open_retrieval_wiki_dataset, get_open_retrieval_batch


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, eval=False, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn
        self.eval = eval
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        batch_size = len(batch_data)
        if batch_size == 0:
            raise StopIteration

        tensorized = OrderedDict()
        for d in batch_data:
            for k, v in d.items():
                tensorized.setdefault(k, []).append(v)
        assert len(tensorized) == 2

        tensorized['row_id'] = torch.LongTensor(tensorized['row_id'])

        input_encoding = self.bert_tokenizer.pad({'input_ids': tensorized['title_text_ids']},
                                                 padding='longest',
                                                 max_length=256,
                                                 pad_to_multiple_of=8,
                                                 return_attention_mask=True,
                                                 return_tensors='pt')
        assert input_encoding.input_ids.size(1) <= 256

        tensorized['context'] = input_encoding.input_ids
        tensorized['context_pad_mask'] = input_encoding.attention_mask
        tensorized['context_types'] = torch.LongTensor(input_encoding.input_ids.size()).fill_(0)

        mask = (tensorized['context'][:, None, :] >= 1) * (tensorized['context'][:, :, None] >= 1)
        # Inverting the mask
        tensorized['context_mask'] = ~mask

        return tensorized



def get_one_epoch_dataloader(dataset, batch_size=None):
    args = get_args()

    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    if batch_size is None:
        batch_size = args.batch_size
    global_batch_size = batch_size * world_size
    num_workers = args.num_workers

    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler,
                                            batch_size=global_batch_size,
                                            drop_last=False,
                                            rank=rank,
                                            world_size=world_size)

    return CustomDataLoader(dataset,
                            batch_sampler=batch_sampler,
                            num_workers=num_workers,
                            pin_memory=True)


class IndexBuilder(object):

    def __init__(self, call_load_attributes_func=True, custom_load_path=None, key_list=None):
        args = get_args()
        self.model = None
        self.dataloader = None
        self.evidence_embedder_obj = None
        self.dataset = get_open_retrieval_wiki_dataset()

        self.log_interval = args.indexer_log_interval
        self.batch_size = args.indexer_batch_size

        if call_load_attributes_func:
            self.load_attributes(custom_load_path=custom_load_path,
                                 key_list=key_list)
        self.is_main_builder = mpu.get_data_parallel_rank() == 0
        self.num_total_builders = mpu.get_data_parallel_world_size()

    def load_attributes(self, custom_load_path=None, key_list=None):
        args = get_args()
        only_context_model = True
        model = get_model(lambda: dualencoder_model_provider(only_context_model=only_context_model))

        if args.run_indexer is False and args.bert_load is None:
            self.model = load_dualencoder_checkpoint(model,
                                                     only_context_model=only_context_model,
                                                     custom_load_path=custom_load_path,
                                                     key_list=key_list)
        else:
            unwrapped_model = model
            while hasattr(unwrapped_model, 'module'):
                unwrapped_model = unwrapped_model.module
            unwrapped_model.init_state_dict_from_bert()
            self.model = model

        self.model.eval()
        self.dataloader = iter(get_one_epoch_dataloader(self.dataset,
                                                        self.batch_size))
        self.evidence_embedder_obj = OpenRetreivalDataStore(load_from_path=False)
        self.iteration = self.total_processed = 0

    def track_and_report_progress(self, batch_size):
        self.iteration += 1
        self.total_processed += batch_size * self.num_total_builders
        if self.is_main_builder and self.iteration % self.log_interval == 0:
            print('Batch {:10d} | Total {:10d}'.format(self.iteration, self.total_processed), flush=True)

    def build_and_save_index(self):
        unwrapped_model = self.model
        while not hasattr(unwrapped_model, 'embed_text'):
            unwrapped_model = unwrapped_model.module

        while True:
            try:
                # batch also has query_tokens and query_pad_data
                row_id, context_tokens, context_mask, context_types, \
                context_pad_mask = get_open_retrieval_batch(self.dataloader)
            except (StopIteration, IndexError):
                break

            assert context_mask.dtype == torch.bool
            context_logits = unwrapped_model.embed_text(unwrapped_model.context_model,
                                                        context_tokens,
                                                        context_mask,
                                                        context_types)
            context_logits = detach(context_logits)
            row_id = detach(row_id)

            self.evidence_embedder_obj.add_block_data(row_id, context_logits)
            self.track_and_report_progress(batch_size=len(row_id))

        # This process signals to finalize its shard and then synchronize with the other processes
        self.evidence_embedder_obj.save_shard()
        torch.distributed.barrier(get_data_parallel_group())
        del self.model

        # rank 0 process builds the final copy
        if self.is_main_builder:
            self.evidence_embedder_obj.merge_shards_and_save()
            # make sure that every single piece of data was embedded
            assert len(self.evidence_embedder_obj.embed_data) == len(self.dataset)
        self.evidence_embedder_obj.clear()

        # complete building the final copy
        torch.distributed.barrier(get_data_parallel_group())
