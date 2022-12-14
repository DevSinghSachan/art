import warnings
import math
import torch
import torch.nn.functional as F
from megatron import get_args, print_rank_0
from megatron.checkpointing import load_dualencoder_checkpoint
from megatron.module import MegatronModule
from megatron.model.dualencoder_model import dualencoder_model_provider
from megatron.mpu import get_mips_group, get_node_first_rank
from megatron import get_tokenizer
from megatron.tokenizer.tokenizer import vocab_size_with_padding
from megatron.data.art_index import OpenRetreivalDataStore, DistributedBruteForceIndex
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.mpu.initialize import get_data_parallel_group
from megatron import get_t0_model, get_t0_tokenizer
from transformers import BertTokenizer as HFBertTokenizer


class ARTModel(MegatronModule):
    def __init__(self, evidence_retriever=None):
        super(ARTModel, self).__init__()
        args = get_args()
        self.topk = args.topk_retrievals

        if args.topk_retrievals > 0:
            bert_tokenizer = get_tokenizer()
            bert_vocab_size = vocab_size_with_padding(bert_tokenizer.vocab_size,
                                                      args)
            print_rank_0('building Retriever for ART (Autoencoding-based Retriever Training) ...')
            self.retriever_model = dualencoder_model_provider(only_context_model=False,
                                                              only_query_model=False,
                                                              vocab_size=bert_vocab_size)
            self._retriever_model_key = 'retriever/biencoder_model'
            self.evidence_retriever = evidence_retriever

        # We have two tokenizers: (1) for BERT as the retriever models are trained using BERT.
        # and (2) for T0 as the pre-trained language model scorer uses T0 tokenization.
        self.hf_bert_tokenizer = HFBertTokenizer.from_pretrained("bert-large-uncased")
        self.t0_tokenizer = get_t0_tokenizer()

    def retriever_embedder(self, tokens, mask, types, embedder_type, disable_dropout=False):
        unwrapped_model = self.retriever_model
        while not hasattr(unwrapped_model, 'embed_text'):
            unwrapped_model = unwrapped_model.module

        if embedder_type == "query":
            if disable_dropout:
                unwrapped_model.query_model.eval()
            logits = unwrapped_model.embed_text(unwrapped_model.query_model,
                                                tokens,
                                                mask,
                                                types)
            return logits
        elif embedder_type == "context":
            if disable_dropout:
                unwrapped_model.context_model.eval()
            logits = unwrapped_model.embed_text(unwrapped_model.context_model,
                                                tokens,
                                                mask,
                                                types)
            return logits
        else:
            raise ValueError("Invalid embedder type.")

    def forward(self, query_uid, query_ids_bert, query_types, query_mask_bert,
                prefixed_query_ids_t0, prefixed_query_ids_t0_len):

        args = get_args()
        topk = self.topk
        bsize, max_seq_len = query_ids_bert.shape

        # assert bsize == 1, "for auto-encoder pre-training, we assume a local batch size of 1"
        assert args.initialize_t0_model_tokenizer_evidence, "for auto-encoder pre-training, we need to pass the argument --initialize-t0-model-and-tokenizer"

        # Compute "fresh" query logits
        query_logits = self.retriever_embedder(query_ids_bert,
                                               query_mask_bert,
                                               query_types,
                                               embedder_type="query",
                                               disable_dropout=args.disable_retriever_dropout)
        if args.no_query_embedder_training:
            query_logits = query_logits.detach()

        # Get top-K evidence data for the BERT tokenized query
        with torch.no_grad():
            topk_evidence_data, stale_topk_sim = self.evidence_retriever.get_topk(query_logits.clone().detach())

        with torch.no_grad():
            output = postprocess(query_uid,
                                 topk_evidence_data)
            all_title_context_ids_bert_tokenized, all_title_context_ids_for_t0 = output

        # Compute the language model score of the retrieved passages
        decoder_prefix_tensor = torch.repeat_interleave(prefixed_query_ids_t0, topk, dim=0)
        log_prob_list = []

        language_model = get_t0_model()
        language_model = language_model.cuda()

        for k in range(0, bsize * topk, topk):
            log_prob_list_one_question = []
            all_title_context_ids_one_question = all_title_context_ids_for_t0[k: k + topk]
            decoder_prefix_tensor_one_question = decoder_prefix_tensor[k: k + topk]
            prefixed_query_ids_t0_len_one_question = prefixed_query_ids_t0_len[k // topk]

            for i in range(0, topk, args.shard_size):
                all_title_context_ids_view = all_title_context_ids_one_question[i: i + args.shard_size]
                # pad the sequences
                input_encoding = self.t0_tokenizer.pad({'input_ids': all_title_context_ids_view},
                                                       padding='longest',
                                                       max_length=512,
                                                       pad_to_multiple_of=8,
                                                       return_attention_mask=True,
                                                       return_tensors='pt')
                assert input_encoding.input_ids.size(1) <= 512
                context_tensor, attention_mask = input_encoding.input_ids.cuda(), input_encoding.attention_mask.cuda()
                decoder_prefix_tensor_view = decoder_prefix_tensor_one_question[i: i + args.shard_size]

                with torch.no_grad():
                    lm_output = language_model(input_ids=context_tensor,
                                               attention_mask=attention_mask,
                                               labels=decoder_prefix_tensor_view,
                                               output_attentions=False,
                                               output_hidden_states=False)
                    lm_logits = lm_output.logits.float()
                    _, decoder_seq_length, vocab_size = lm_logits.shape

                    log_softmax = F.log_softmax(lm_logits, dim=-1)
                    gold_log_probs = log_softmax.gather(2, decoder_prefix_tensor_view.unsqueeze(2)).squeeze(2)

                    # this will work because the batch size is 1 and this implies all decoder labels have the same length
                    teacher_log_probs = torch.mean(gold_log_probs[:, :prefixed_query_ids_t0_len_one_question], dim=1)
                    log_prob_list_one_question.append(teacher_log_probs)

            log_prob_list_one_question = torch.cat(log_prob_list_one_question).unsqueeze(0)
            log_prob_list.append(log_prob_list_one_question)

        gold_log_probs = torch.cat(log_prob_list, dim=0)


        # Compute the retriever score
        input_encoding = self.hf_bert_tokenizer.pad({'input_ids': all_title_context_ids_bert_tokenized},
                                                    padding='longest',
                                                    max_length=512,
                                                    pad_to_multiple_of=8,
                                                    return_attention_mask=True,
                                                    return_tensors='pt')
        assert input_encoding.input_ids.size(1) <= 512

        all_title_context_ids = input_encoding.input_ids.cuda()
        all_context_types = torch.cuda.LongTensor(input_encoding.input_ids.size()).fill_(0)
        all_context_mask = (all_title_context_ids[:, None, :] >= 1) * (all_title_context_ids[:, :, None] >= 1)
        # Inverting the mask
        all_context_mask = ~all_context_mask

        # Compute "fresh" context logits
        all_context_logits = self.retriever_embedder(all_title_context_ids,
                                                     all_context_mask,
                                                     all_context_types,
                                                     embedder_type="context",
                                                     disable_dropout=args.disable_retriever_dropout)
        all_context_logits = all_context_logits.reshape(bsize, topk, -1)

        if args.no_context_embedder_training:
            all_context_logits = all_context_logits.detach()

        # [B, 1, dim]
        query_logits = query_logits.unsqueeze(1).float()
        all_context_logits = all_context_logits.float()

        # [B, 1, K]
        topk_sim_scores = torch.bmm(query_logits, all_context_logits.transpose(1, 2))

        if args.retriever_score_scaling:
            topk_sim_scores = topk_sim_scores / (args.inverse_temperature_multiplier * math.sqrt(args.hidden_size))

        # [B, 1, K]
        topk_log_probs = F.log_softmax(topk_sim_scores, dim=2)
        # B x 1 x K -> B x K
        topk_log_probs = topk_log_probs.squeeze(1)

        return topk_log_probs, gold_log_probs


    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads, add an extra key."""
        state_dict_ = dict()
        state_dict_[self._retriever_model_key] = self.retriever_model.state_dict_for_save_checkpoint(destination,
                                                                                                     prefix,
                                                                                                     keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dicts of each of the models"""
        self.retriever_model.load_state_dict(state_dict[self._retriever_model_key], strict)

    def init_state_dict_from_bert(self):
        """Initialize the state from pre-trained BERT model"""
        if self.retriever_model is not None:
            print_rank_0("Initializing retriever model from pretrained BERT")
            self.retriever_model.init_state_dict_from_bert()

    def init_state_dict_from_dualencoder(self):
        """Initialize the state from pre-trained DPR model and pre-trained T5 mode on iteration zero of pretraining"""
        args = get_args()

        if args.pretrained_dualencoder_load is None:
            assert args.bert_load is not None
            warnings.warn("Pretrained dual-encoder checkpoints are not found. Initializing from random weights")
            return

        print_rank_0("Initializing retriever model from pretrained Dual-Encoder checkpoint")
        load_dualencoder_checkpoint(self.retriever_model,
                                    custom_load_path=args.pretrained_dualencoder_load)


def postprocess(query_uid, topk_evidence_data):
    args = get_args()
    query_uid = query_uid.tolist()
    bert_tokenizer = get_tokenizer()
    t0_tokenizer = get_t0_tokenizer()

    verbalizer_head_ids = t0_tokenizer.encode(args.verbalizer_head,
                                              add_special_tokens=False)
    verbalizer_ids = t0_tokenizer.encode(args.verbalizer,
                                         add_special_tokens=False)
    all_title_context_ids_bert_tokenized = []
    all_title_context_ids_t0_tokenized = []
    MAX_SEQUENCE_LEN = 512

    for qid, (topkids, text_list, text_list_t0) in zip(query_uid, topk_evidence_data):
        k = 0
        for eid, (context_ids, title_ids), (context_ids_t0, title_ids_t0) in zip(topkids, text_list, text_list_t0):
            t0_context_title_ids = []
            # We should ignore the evidence from which query originates
            if qid != eid and k < args.topk_retrievals:
                k += 1

                bert_title_text_ids = [bert_tokenizer.cls] + title_ids + [bert_tokenizer.sep] + context_ids
                to_be_added_len = 1
                if len(bert_title_text_ids) + to_be_added_len >= MAX_SEQUENCE_LEN:
                    truncate_len = len(bert_title_text_ids) + to_be_added_len - MAX_SEQUENCE_LEN
                    bert_title_text_ids = bert_title_text_ids[: -truncate_len]
                bert_title_text_ids.extend([bert_tokenizer.sep])

                all_title_context_ids_bert_tokenized.append(bert_title_text_ids)

                # Original Input Style: Passage: <title> <passage> . Can you please write a question?
                t0_context_title_ids.extend(verbalizer_head_ids)
                t0_context_title_ids.extend(title_ids_t0)
                t0_context_title_ids.extend(context_ids_t0)

                # Truncating the sequence length if larger than 512
                to_be_added_len = len(verbalizer_ids) + 1
                if len(t0_context_title_ids) + to_be_added_len >= MAX_SEQUENCE_LEN:
                    truncate_len = len(t0_context_title_ids) + to_be_added_len - MAX_SEQUENCE_LEN
                    t0_context_title_ids = t0_context_title_ids[: -truncate_len]
                t0_context_title_ids.extend(verbalizer_ids)
                t0_context_title_ids.extend([t0_tokenizer.eos_token_id])

                all_title_context_ids_t0_tokenized.append(t0_context_title_ids)

    return all_title_context_ids_bert_tokenized, all_title_context_ids_t0_tokenized


def query_single_context_t5_format(query_ids, title_ids, context_ids, max_seq_length, sep_id, pad_id):
    enc_ids = []
    src_ids = query_ids + title_ids + [sep_id] + context_ids
    enc_ids.extend(src_ids)

    if len(enc_ids) > max_seq_length - 1:
        enc_ids = enc_ids[0: max_seq_length - 1]

    enc_ids.append(sep_id)

    padding_length = max_seq_length - len(enc_ids)
    if padding_length > 0:
        enc_ids.extend([pad_id] * padding_length)

    return enc_ids


class PreComputedEvidenceDocsRetriever(object):
    def __init__(self):
        args = get_args()
        self.topk = args.topk_retrievals
        self.embedding_size = args.hidden_size
        self.evidence_embedder_obj = None
        self.mips_index = None

        self.precomputed_index_wrapper()

        self.allow_trivial_doc = args.allow_trivial_doc
        if not args.allow_trivial_doc:
            self.topk = self.topk + 1

        self.local_rank = args.local_rank
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            self.device_count = 1
        else:
            # Note: This will work only for 1 node. May need to fix this?
            self.device_count = min(torch.cuda.device_count(), args.max_training_rank)

        self.passages_map_bert = make_indexed_dataset(args.indexed_evidence_bert_tokenized_data_path,
                                                      impl=args.data_impl,
                                                      skip_warmup=(not args.mmap_warmup))

        self.title_map_bert = make_indexed_dataset(args.indexed_title_bert_tokenized_data_path,
                                                   impl=args.data_impl,
                                                   skip_warmup=(not args.mmap_warmup))

        self.passages_map_t0 = make_indexed_dataset(args.indexed_evidence_t0_tokenized_data_path,
                                                    impl=args.data_impl,
                                                    skip_warmup=(not args.mmap_warmup))

        self.title_map_t0 = make_indexed_dataset(args.indexed_title_t0_tokenized_data_path,
                                                 impl=args.data_impl,
                                                 skip_warmup=(not args.mmap_warmup))

    def get_evidence_embedding(self, path):
        self.evidence_embedder_obj = OpenRetreivalDataStore(path,
                                                            load_from_path=True)

    def precomputed_index_wrapper(self):
        args = get_args()
        if get_node_first_rank() == torch.distributed.get_rank():
            self.get_evidence_embedding(args.embedding_path)
            assert self.evidence_embedder_obj is not None
            self.mips_index = DistributedBruteForceIndex(embed_size=self.embedding_size,
                                                         embed_data=self.evidence_embedder_obj)
        # Wait for the index to be initialized in all the GPUs
        torch.distributed.barrier(get_data_parallel_group())


    def update_evidence_embedding(self):
        """Reload index with new data loaded from disk. Should be performed after each indexer job completes."""
        if get_node_first_rank() == torch.distributed.get_rank():
            self.mips_index.update_index()

        # Wait for the MIPS index to be initialized in all the nodes
        torch.distributed.barrier(get_data_parallel_group())


    def get_topk(self, query_tensor):
        local_bsize = query_tensor.shape[0]
        input_ = torch.empty_like(query_tensor).copy_(query_tensor).detach_()
        tensor_list = [torch.empty_like(input_) for _ in range(self.device_count)]
        torch.distributed.all_gather(tensor_list, query_tensor, group=get_mips_group())

        if get_node_first_rank() == torch.distributed.get_rank():
            assert self.mips_index is not None, "MIPS Index is not initialized"
            all_query_tensor = torch.cat(tensor_list, dim=0).contiguous()
            distance, topkindex = self.mips_index.search_mips_index(all_query_tensor,
                                                                    top_k=self.topk,
                                                                    reconstruct=False)
        else:
            distance = torch.empty(self.device_count * local_bsize, self.topk, dtype=torch.float16).cuda()
            topkindex = torch.empty(self.device_count * local_bsize, self.topk, dtype=torch.int32).cuda()

        torch.distributed.broadcast(distance, src=get_node_first_rank(), group=get_mips_group())
        torch.distributed.broadcast(topkindex, src=get_node_first_rank(), group=get_mips_group())

        distance = torch.split(distance, local_bsize, dim=0)[self.local_rank]
        topkindex = torch.split(topkindex, local_bsize, dim=0)[self.local_rank]

        topk_data = []
        for topkarray in topkindex:
            # The idx contains passage text and title
            topkarray = topkarray.tolist()
            text_list_bert_tok = []
            text_list_t0_tok = []
            for idx in topkarray:
                doctext_ids_bert = self.passages_map_bert[idx - 1].tolist()
                title_ids_bert = self.title_map_bert[idx - 1].tolist()
                text_list_bert_tok.append((doctext_ids_bert, title_ids_bert))

                doctext_ids_t0 = self.passages_map_t0[idx - 1].tolist()
                title_ids_t0 = self.title_map_t0[idx - 1].tolist()
                text_list_t0_tok.append((doctext_ids_t0, title_ids_t0))
            topk_data.append((topkarray, text_list_bert_tok, text_list_t0_tok))

        return topk_data, distance
