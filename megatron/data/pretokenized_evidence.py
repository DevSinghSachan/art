from abc import ABC
from torch.utils.data import Dataset
from megatron import print_rank_0, get_args, get_tokenizer
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset


def get_pretokenized_evidence_dataset():
    args = get_args()
    tokenizer = get_tokenizer()

    dataset = EvidenceDatasetPreTokenized('2018 Wikipedia from DPR codebase',
                                          'evidence',
                                          tokenizer,
                                          args.seq_length_retriever)
    return dataset


def get_open_retrieval_batch(data_iterator):
    data = next(data_iterator)
    row_id = data['row_id'].cuda()
    context = data['context'].cuda()
    context_mask = data['context_mask'].cuda()
    context_types = data['context_types'].cuda()
    context_pad_mask = data['context_pad_mask'].cuda()
    return row_id, context, context_mask, context_types, context_pad_mask


class EvidenceDatasetPreTokenized(ABC, Dataset):
    def __init__(self, task_name, dataset_name, tokenizer, max_seq_length):
        # Store inputs.
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,
                                                             self.dataset_name))

        args = get_args()
        self.passages_map_bert = make_indexed_dataset(args.indexed_evidence_bert_tokenized_data_path,
                                                      impl=args.data_impl,
                                                      skip_warmup=(not args.mmap_warmup))
        self.title_map_bert = make_indexed_dataset(args.indexed_title_bert_tokenized_data_path,
                                                   impl=args.data_impl,
                                                   skip_warmup=(not args.mmap_warmup))
        print_rank_0('  >> total number of passages: {}'.format(len(self.passages_map_bert)))

    def __len__(self):
        return len(self.passages_map_bert)

    def __getitem__(self, idx):
        # These indexed datasets follow zero indexing
        text_ids = self.passages_map_bert[idx].tolist()
        title_ids = self.title_map_bert[idx].tolist()

        title_text_ids = [self.tokenizer.cls] + title_ids + [self.tokenizer.sep] + text_ids
        to_be_added_len = 1
        if len(title_text_ids) + to_be_added_len >= self.max_seq_length:
            truncate_len = len(title_text_ids) + to_be_added_len - self.max_seq_length
            title_text_ids = title_text_ids[: -truncate_len]

        title_text_ids.extend([self.tokenizer.sep])

        # idx + 1 is needed because in DPR Wikipedia passages the indexing starts from 1 and not 0.
        sample = {"row_id": idx + 1,
                  "title_text_ids": title_text_ids
                  }

        return sample
