from collections import OrderedDict
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from megatron import get_args
from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.training import setup_model_and_optimizer
from megatron.training import train_step
from megatron.training import training_log
from megatron.utils import reduce_losses
from megatron import get_t0_tokenizer, get_msmarco_dev_reference
from megatron.indexer_emdr2 import IndexBuilder
from tasks.openqa.dense_retriever.evaluation.evaluate import OpenRetrievalEvaluator


def process_batch(batch):
    query_uid = batch['query_uid'].cuda()
    query_ids_bert = batch['query_ids_bert'].cuda()
    query_types = batch['query_types'].cuda()
    query_mask_bert = (batch['query_mask_bert'] < 0.5).cuda()
    query_ids_t5 = batch['query_ids_t5'].cuda()
    query_ids_t5_len = batch['query_ids_t5_len'].cuda()
    dec_ids = batch['dec_ids'].cuda()
    labels = batch['labels'].cuda()
    loss_mask = batch['loss_mask'].cuda()
    prefixed_query_ids_t0 = batch['prefixed_query_ids_t0'].cuda()
    prefixed_query_mask_t0 = batch['prefixed_query_mask_t0'].cuda()
    prefixed_query_ids_t0_len = batch['prefixed_query_ids_t0_len'].cuda()
    reference = batch['reference']

    return query_uid, query_ids_bert, query_types, query_mask_bert, \
           query_ids_t5, query_ids_t5_len, dec_ids, labels, loss_mask, \
           prefixed_query_ids_t0, prefixed_query_mask_t0, \
           prefixed_query_ids_t0_len, reference


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, eval=False, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn
        self.eval = eval
        self.t0_tokenizer = get_t0_tokenizer()
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        tensorized = OrderedDict()
        for d in batch_data:
            for k, v in d.items():
                tensorized.setdefault(k, []).append(v)
        assert len(tensorized) == 11

        tensorized['query_uid'] = torch.LongTensor(tensorized['query_uid'])
        tensorized['query_ids_bert'] = torch.LongTensor(tensorized['query_ids_bert'])
        tensorized['query_types'] = torch.LongTensor(tensorized['query_types'])
        tensorized['query_mask_bert'] = torch.LongTensor(tensorized['query_mask_bert'])
        tensorized['query_ids_t5'] = torch.LongTensor(tensorized['query_ids_t5'])
        tensorized['query_ids_t5_len'] = torch.LongTensor(tensorized['query_ids_t5_len'])
        tensorized['dec_ids'] = torch.LongTensor(tensorized['dec_ids'])
        tensorized['labels'] = torch.LongTensor(tensorized['labels'])
        tensorized['loss_mask'] = torch.FloatTensor(tensorized['loss_mask'])

        prefixed_query_ids_mask_t0 = self.t0_tokenizer(tensorized['prefixed_query_text'],
                                                       padding='longest',
                                                       max_length=128,
                                                       pad_to_multiple_of=8,
                                                       truncation=True,
                                                       return_tensors='pt')
        tensorized['prefixed_query_ids_t0'] = prefixed_query_ids_mask_t0.input_ids
        tensorized['prefixed_query_mask_t0'] = prefixed_query_ids_mask_t0.attention_mask
        tensorized['prefixed_query_ids_t0_len'] = torch.sum(prefixed_query_ids_mask_t0.attention_mask, dim=1)

        # The final key is the reference, which is already appended.
        return tensorized


def _cross_entropy_forward_step(batch, model):
    """Simple forward step with cross-entropy loss."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch

    query_uid, query_ids_bert, query_types, query_mask_bert, \
    query_ids_t5, query_ids_t5_len, dec_ids, labels, loss_mask, \
    prefixed_query_ids_t0, prefixed_query_mask_t0, \
    prefixed_query_ids_t0_len, reference = process_batch(batch_)
    assert torch.all(query_uid < 0), "query uid can't be positive"

    timers('batch generator').stop()

    # Forward model.
    topk_log_probs, gold_log_probs = model(query_uid,
                                           query_ids_bert,
                                           query_types,
                                           query_mask_bert,
                                           prefixed_query_ids_t0,
                                           prefixed_query_ids_t0_len)

    # Retriever loss
    retriever_loss = torch.FloatTensor([0]).cuda()
    if args.update_retriever:
        topk_log_probs = topk_log_probs.float()
        gold_log_probs = gold_log_probs.float()
        gold_log_probs_log_softmax = F.log_softmax(gold_log_probs, dim=1)
        loss_func = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        retriever_loss = loss_func(topk_log_probs, gold_log_probs_log_softmax)

    net_loss = retriever_loss
    reduced_loss = reduce_losses([retriever_loss])

    return net_loss, {'retriever_loss': reduced_loss[0]}


def accuracy_func_provider(single_dataset_provider, datapath):
    args = get_args()
    dataset = single_dataset_provider(datapath)
    drop_last = False

    dataloader = build_data_loader(dataset,
                                   args.eval_batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=drop_last,
                                   shuffle=False)
    dataloaders = (dataset.dataset_name, dataloader)

    def metrics_func(model, epoch):
        print_rank_0('calculating metrics ...')

    return metrics_func


def build_data_loader(dataset, batch_size, num_workers, drop_last, shuffle=True, rank0sampler=False):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""

    if rank0sampler:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        world_size = mpu.get_data_parallel_world_size()
        rank = mpu.get_data_parallel_rank()
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                  num_replicas=world_size,
                                                                  rank=rank,
                                                                  shuffle=shuffle)
    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = CustomDataLoader(dataset,
                                   batch_size=batch_size,
                                   sampler=sampler,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   drop_last=drop_last,
                                   pin_memory=True)
    return data_loader


def _build_infinite_size_dataloader(dataloader):
    """Build a looped dataloader with infinite size."""

    iterator = dataloader.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = dataloader.__iter__()


def _build_train_valid_dataloaders(train_dataset, valid_dataset):
    """Traing and validation dataloaders."""
    args = get_args()

    print_rank_0('building train and validation dataloaders ...')
    # Training dataset.
    train_dataloader = build_data_loader(train_dataset,
                                         args.batch_size,
                                         args.num_workers,
                                         not args.keep_last)

    # Set the training iterations.
    args.train_iters_per_epoch = len(train_dataloader)
    args.train_iters = args.epochs * args.train_iters_per_epoch

    # Validation dataset. For this dataset, we do not need to set up
    # shuffling so we can just use a simple infinite loop.
    valid_dataloader_ = build_data_loader(valid_dataset,
                                          args.batch_size,
                                          args.num_workers,
                                          not args.keep_last)

    valid_dataloader = _build_infinite_size_dataloader(valid_dataloader_)
    return train_dataloader, valid_dataloader


def get_retrieval_score(mips_index=None, iteration_num=-1):
    args = get_args()
    evaluator = OpenRetrievalEvaluator(custom_load_path=args.load,
                                       key_list=['retriever/biencoder_model'],
                                       load_evidence_dataset=False,
                                       use_faiss=False)
    query2passage_list = get_msmarco_dev_reference()

    if args.qa_file_dev is not None:
        evaluator.evaluate(args.qa_file_dev,
                           "DEV",
                           mips_index=mips_index,
                           query2passage_list=query2passage_list,
                           iteration_num=iteration_num)
        torch.distributed.barrier()

    if args.qa_file_test is not None:
        evaluator.evaluate(args.qa_file_test,
                           "TEST",
                           mips_index=mips_index,
                           query2passage_list=query2passage_list,
                           iteration_num=iteration_num)
        torch.distributed.barrier()

    del evaluator.model
    del evaluator
    torch.cuda.empty_cache()



def call_evidence_index_builder():
    args = get_args()
    index_builder = IndexBuilder(custom_load_path=args.load,
                                 key_list=['retriever/biencoder_model'])
    index_builder.build_and_save_index()
    del index_builder
    torch.cuda.empty_cache()


def _train(model, optimizer, lr_scheduler, forward_step, train_dataloader):
    """Train the model."""
    args = get_args()
    timers = get_timers()

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    losses_dict_sum = {}

    # Starting epoch and iteration
    start_epoch = args.iteration // args.train_iters_per_epoch
    start_iteration = args.iteration % args.train_iters_per_epoch
    iteration = args.iteration

    # For async updates
    last_reload_iteration = None
    last_eval_iteration = iteration

    if args.compute_fresh_evidence_embeddings:
        last_reload_iteration = iteration

    # Memory reporting flag.
    report_memory_flag = True

    # For each remaining epoch
    timers('interval time').start()
    for epoch in range(start_epoch, args.epochs):
        print_rank_0('working on epoch {} ...'.format(epoch + 1))

        # Set the data loader epoch to shuffle the index iterator.
        train_dataloader.sampler.set_epoch(args.seed + epoch)

        # For all the batches in the dataset.
        for iteration_, batch in enumerate(train_dataloader):

            # Ignore the iterations before starting value
            if iteration_ < start_iteration:
                continue
            # Set to zero so the next epoch does not skip any batches.
            start_iteration = 0

            # Evaluation
            if args.compute_fresh_evidence_embeddings and iteration >= last_reload_iteration + args.index_reload_interval:
                # Recompute evidence embeddings
                call_evidence_index_builder()
                print_rank_0("Training Group: Updating MIPS Index")
                # get model without FP16 and/or TorchDDP wrappers
                unwrapped_model = model
                while hasattr(unwrapped_model, 'module'):
                    unwrapped_model = unwrapped_model.module
                unwrapped_model.evidence_retriever.update_evidence_embedding()
                print_rank_0("Training Group: MIPS Index Updated")
                last_reload_iteration = iteration

            if iteration >= last_eval_iteration + args.eval_interval:
                unwrapped_model = model
                while hasattr(unwrapped_model, 'module'):
                    unwrapped_model = unwrapped_model.module
                get_retrieval_score(unwrapped_model.evidence_retriever.mips_index,
                                    iteration)
                last_eval_iteration = iteration

            # Train for one step.
            losses_dict, skipped_iter = train_step(forward_step, batch, model,
                                                   optimizer, lr_scheduler)
            iteration += 1

            # Logging.
            report_memory_flag = training_log(losses_dict, losses_dict_sum,
                                              optimizer.param_groups[0]['lr'],
                                              iteration, optimizer.loss_scale,
                                              report_memory_flag, skipped_iter)

            # Checkpointing
            if args.save and args.save_interval and \
                    iteration % args.save_interval == 0:
                save_checkpoint(iteration, model, optimizer, lr_scheduler)


            if args.exit_interval and iteration % args.exit_interval == 0:
                torch.distributed.barrier(mpu.get_data_parallel_group())
                rank = torch.distributed.get_rank()
                print_rank_0('rank: {} | exiting the program at iteration {}'.format(rank, iteration))
                sys.exit(0)

        # Checkpointing at the end of each epoch.
        # if args.save:
        #     save_checkpoint(iteration, model, optimizer, lr_scheduler)


def train(train_valid_datasets_provider, model_provider,
          forward_step=_cross_entropy_forward_step,
          end_of_epoch_callback_provider=None,
          end_of_training_callback_provider=None):

    args = get_args()
    timers = get_timers()

    # Train and validation data loaders.
    timers('train/valid/test dataset/dataloder').start()
    if args.epochs > 0:
        train_dataset, valid_dataset = train_valid_datasets_provider()
        train_dataloader, valid_dataloader = _build_train_valid_dataloaders(train_dataset,
                                                                            valid_dataset)
    timers('train/valid/test dataset/dataloder').stop()

    # Build model, optimizer and learning rate scheduler.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model and optimizer').stop()

    # If pretrained checkpoint is provided and we have not trained for
    # any iteration (i.e., iteration is zero), then load the pretrained
    # checkpoint.
    timers('pretrained checkpoint').start()
    if args.iteration == 0 and args.pretrained_checkpoint is not None:
        original_load = args.load
        args.load = args.pretrained_checkpoint
        _ = load_checkpoint(model, None, None)
        args.load = original_load
        # This is critical when only model is loaded. We should make sure
        # master parameters are also updated.
        if args.fp16:
            optimizer._model_params_to_master_params()
    timers('pretrained checkpoint').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['train/valid/test dataset/dataloder',
                'model and optimizer',
                'pretrained checkpoint'])
    print_rank_0('training ...')

    # Finetune the model.
    if args.epochs > 0 and args.upr_distillation_training:
        _train(model,
               optimizer,
               lr_scheduler,
               forward_step,
               train_dataloader)

    print_rank_0('done :-)')
