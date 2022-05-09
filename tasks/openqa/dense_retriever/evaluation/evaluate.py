import os
import shutil
import json
import torch
from megatron import get_args, print_rank_0, get_tokenizer, mpu
from megatron.training import get_model
from megatron.model.dualencoder_model import dualencoder_model_provider
from megatron.checkpointing import load_dualencoder_checkpoint
from megatron.data.orqa_wiki_dataset import get_open_retrieval_wiki_dataset
from tasks.openqa.dense_retriever.evaluation.data import get_qa_dataset, get_one_epoch_qa_dataloader, process_qa_batch
from megatron.data.emdr2_index import OpenRetreivalDataStore, FaissMIPSIndex


class OpenRetrievalEvaluator(object):
    def __init__(self, custom_load_path=None, key_list=None,
                 load_evidence_dataset=True,
                 use_faiss=True):
        args = get_args()
        self.evidence_embedder_obj = None
        self.evidence_dataset = None
        self.mips_index = None

        # Load query encoder checkpoint
        only_query_model = True
        model = get_model(lambda: dualencoder_model_provider(only_query_model=only_query_model))
        self.model = load_dualencoder_checkpoint(model,
                                                 only_query_model=only_query_model,
                                                 custom_load_path=custom_load_path,
                                                 key_list=key_list)
        self.model.eval()

        if load_evidence_dataset:
            self.get_evidence_dataset()
        if use_faiss:
            self.faiss_wrapper()

        # Wait for the index to be initialized in all the nodes
        torch.distributed.barrier()

    def get_evidence_embedding(self):
        # This will load the embedding from the embedding path
        self.evidence_embedder_obj = OpenRetreivalDataStore(load_from_path=True)

    def get_evidence_dataset(self):
        self.evidence_dataset = get_open_retrieval_wiki_dataset()

    def faiss_wrapper(self):
        # Initialize FAISS wrapper on local rank = 0 as the evidence embeddings is distributed over all the GPUs in a node
        args = get_args()
        if args.local_rank == 0:
            self.get_evidence_embedding()
            assert self.evidence_embedder_obj is not None
            self.mips_index = FaissMIPSIndex(embed_size=args.hidden_size,
                                             embed_data=self.evidence_embedder_obj,
                                             use_gpu=args.faiss_use_gpu)

    def generate_query_vectors(self, eval_dataset):
        dataloader = iter(get_one_epoch_qa_dataloader(eval_dataset))
        tokenizer = get_tokenizer()
        query_vectors = []
        query_list = []
        question_id_list = []

        while True:
            try:
                batch = next(dataloader)
            except (StopIteration, IndexError):
                break

            # batch also has query_tokens and query_pad_data
            query_tokens, query_mask, query_types, \
            query_len, question_id = process_qa_batch(batch)

            unwrapped_model = self.model
            while not hasattr(unwrapped_model, 'embed_text'):
                unwrapped_model = unwrapped_model.module

            with torch.no_grad():
                query_logits = unwrapped_model.embed_text(unwrapped_model.query_model,
                                                          query_tokens,
                                                          query_mask,
                                                          query_types)

            for i in range(len(query_tokens)):
                query_list.append(tokenizer.decode(query_tokens[i].tolist()[:query_len[i]]))

            question_id_list.extend(question_id)
            query_vectors.extend(query_logits.split(1, dim=0))
            if len(query_vectors) % 100 == 0:
                print_rank_0('Encoded queries {}'.format(len(query_vectors) * mpu.get_data_parallel_world_size()))

        query_tensor = torch.cat(query_vectors, dim=0)
        return query_list, query_tensor, question_id_list

    def evaluate(self, qa_file, split, mips_index=None, query2passage_list=None, iteration_num=-1):
        args = get_args()
        eval_dataset = get_qa_dataset(qa_file, split)
        query_list, query_tensor, question_id_list = self.generate_query_vectors(eval_dataset)

        if mips_index is not None:
            mips_index_cls = mips_index
        else:
            mips_index_cls = self.mips_index

        local_rank = args.local_rank
        rank = torch.distributed.get_rank()
        device_count = torch.cuda.device_count()
        world_size = torch.distributed.get_world_size()

        if world_size == 1:
            num_nodes = 1
        else:
            num_nodes = world_size // device_count
        node_id = rank // device_count

        for node in range(num_nodes):
            start_rank = node * device_count
            if world_size == 1:
                end_rank = 1
            else:
                end_rank = (node + 1) * device_count
            ranks_list = list(range(start_rank, end_rank))
            node_group = torch.distributed.new_group(ranks=ranks_list)

            if node_id == node:
                device_start_rank = start_rank
                group = node_group

        input_ = torch.empty_like(query_tensor).copy_(query_tensor).detach_()
        all_query_tensor, allsizes = varsize_gather_nograd(input_, group)
        print_rank_0(all_query_tensor.shape)
        num_rows = len(all_query_tensor)

        if local_rank == 0 and mips_index_cls is not None:
            all_query_tensor = all_query_tensor.contiguous()
            all_distance, all_topkindex = [], []

            for i in range(0, len(all_query_tensor), args.shard_size):
                query_tensor_view = all_query_tensor[i: i + args.shard_size]

                distance, topkindex = mips_index_cls.search_mips_index(query_tensor_view,
                                                                       top_k=args.report_topk_accuracies[-1],
                                                                       reconstruct=False)
                if type(distance).__module__ == "numpy":
                    distance = torch.from_numpy(distance).half().cuda()
                    topkindex = torch.from_numpy(topkindex).int().cuda()

                all_distance.append(distance)
                all_topkindex.append(topkindex)

            distance = torch.cat(all_distance, dim=0)
            topkindex = torch.cat(all_topkindex, dim=0)

        if local_rank != 0:
            distance = torch.empty(len(all_query_tensor),
                                   args.report_topk_accuracies[-1],
                                   dtype=torch.float16).cuda()
            topkindex = torch.empty(len(all_query_tensor),
                                    args.report_topk_accuracies[-1],
                                    dtype=torch.int32).cuda()

        torch.distributed.broadcast(distance, src=device_start_rank, group=group)
        torch.distributed.broadcast(topkindex, src=device_start_rank, group=group)

        distance = torch.split(distance, allsizes, dim=0)[local_rank]
        topkindex = torch.split(topkindex, allsizes, dim=0)[local_rank]

        del all_query_tensor

        topk_sim_scores = distance #/ math.sqrt(args.hidden_size)

        qids_to_ranked_candidate_passages = {}

        for qid, topkarray in zip(question_id_list, topkindex):
            qid = int(qid)
            qids_to_ranked_candidate_passages[qid] = topkarray.tolist()

        if self.evidence_dataset is None:
            assert query2passage_list is not None
            qids_to_relevant_passageids = query2passage_list
        else:
            qids_to_relevant_passageids = self.evidence_dataset.query2passage_list


        metrics = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        mrr_at_10 = torch.FloatTensor([metrics['MRR @10']]).cuda()
        torch.distributed.all_reduce(mrr_at_10, torch.distributed.ReduceOp.SUM)
        mrr_at_10 = mrr_at_10 / num_rows

        print_str = "{} SET RESULTS\tstep: {}\tMRR@10: {}".format(split, iteration_num, mrr_at_10.item() * 100)
        print_rank_0(print_str)

        if args.save_topk_outputs_path is not None:
            data = ""
            str_template = "{} Q0 {} {} {} upr-distill\n"
            for i, (qid, candidate_list) in enumerate(qids_to_ranked_candidate_passages.items()):
                for rank, doc_id in enumerate(candidate_list):
                    score = 1.0 / int(rank + 1)
                    data += str_template.format(qid, doc_id, rank+1, score)

            temp_dir_name = os.path.join(args.save_topk_outputs_path,
                                         "_tmp_reranker_{}".format(os.getppid()))
            save_shard(data, temp_dir_name)
            del data
            torch.distributed.barrier()

            if mpu.get_data_parallel_rank() == 0:
                file_name = os.path.splitext(os.path.basename(qa_file))[0]
                all_data = merge_shards_and_save(args.save_topk_outputs_path, temp_dir_name, file_name)
                # make sure that every single piece of data was embedded
                assert len(all_data) == len(eval_dataset) * args.report_topk_accuracies[-1]
                del all_data

        torch.distributed.barrier()
        return


@torch.no_grad()
def varsize_gather_nograd(x, group=None):
    """gather tensors of different sizes along the first dimension"""

    #determine max size
    size = torch.tensor([x.shape[0]], device=x.device, dtype=torch.int)
    allsizes = [torch.zeros_like(size) for _ in range(mpu.get_data_parallel_world_size())]
    torch.distributed.all_gather(allsizes, size, group=group)
    max_size = max([size.cpu().max() for size in allsizes])

    padded = torch.empty(
                max_size,
                *x.shape[1:],
                dtype=x.dtype,
                device=x.device
            )
    padded[:x.shape[0]] = x
    output = [torch.zeros_like(padded) for _ in range(mpu.get_data_parallel_world_size())]
    torch.distributed.all_gather(output, padded, group=group)

    output = [tensor[:allsizes[k]] for k, tensor in enumerate(output)]
    output = torch.cat(output, dim=0)

    return output, allsizes


def save_shard(data, temp_dir_name):
    """
    Save the block data that was created this in this process
    """
    if not os.path.isdir(temp_dir_name):
        os.makedirs(temp_dir_name, exist_ok=True)

    outpath = os.path.join(temp_dir_name, "rank{}.txt".format(mpu.get_data_parallel_rank()))
    with open(outpath, "w") as writer:
        writer.write(data)


def merge_shards_and_save(output_dir_path, temp_dir_name, file_name):
    """Combine all the shards made using self.save_shard()"""
    shard_names = os.listdir(temp_dir_name)
    all_data = []

    for fname in os.listdir(temp_dir_name):
        shard_size = 0
        old_size = len(all_data)
        fpath = '{}/{}'.format(temp_dir_name, fname)
        with open(fpath, 'r') as f:
            for line in f:
                shard_size += 1
                all_data.append(line)

        assert len(all_data) == old_size + shard_size
        os.remove(fpath)

    # save the consolidated shards
    outpath = os.path.join(output_dir_path, "{}.txt".format(file_name))

    with open(outpath, 'w') as writer:
        for line in all_data:
            writer.write(line)

    print("Finished merging {} shards for a total of {} embeds".format(
        len(shard_names), len(all_data)), flush=True)

    shutil.rmtree(temp_dir_name, ignore_errors=True)

    return all_data


def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Compute MRR metric
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    MaxMRRRank = 10
    all_scores = {}
    MRR = 0
    qids_with_relevant_passages = 0
    ranking = []
    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0, MaxMRRRank):
                if candidate_pid[i] in target_pid:
                    MRR += 1 / (i + 1)
                    ranking.pop()
                    ranking.append(i + 1)
                    break
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

    # MRR = MRR / len(qids_to_relevant_passageids)
    all_scores['MRR @10'] = MRR
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    return all_scores
