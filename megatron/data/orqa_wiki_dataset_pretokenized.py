"""Wikipedia dataset from DPR code for ORQA."""

import csv
from abc import ABC
import numpy as np
from torch.utils.data import Dataset
from megatron import print_rank_0, get_args, get_tokenizer
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset


def get_open_retrieval_wiki_dataset():
    args = get_args()
    tokenizer = get_tokenizer()

    dataset = OpenRetrievalEvidenceDataset('2018 Wikipedia from DPR codebase',
                                           'evidence',
                                           args.evidence_data_path,
                                           tokenizer,
                                           args.seq_length_ret)
    return dataset


def get_open_retrieval_batch(data_iterator):
    data = next(data_iterator)
    row_id = data['row_id'].cuda()
    context = data['context'].cuda()
    context_mask = data['context_mask'].cuda()
    context_types = data['context_types'].cuda()
    context_pad_mask = data['context_pad_mask'].cuda()
    return row_id, context, context_mask, context_types, context_pad_mask



# noinspection DuplicatedCode
def build_tokens_types_paddings_from_ids(text_ids, max_seq_length,
                                         cls_id, sep_id, pad_id):
    """Build token types and paddings, trim if needed, and pad if needed."""
    enc_ids = []
    tokentypes_enc = []

    # [CLS].
    enc_ids.append(cls_id)
    tokentypes_enc.append(0)

    # A.
    len_src = len(text_ids)
    enc_ids.extend(text_ids)
    tokentypes_enc.extend([0] * len_src)

    # Cap the size.
    if len(enc_ids) > max_seq_length - 1:
        enc_ids = enc_ids[0: max_seq_length - 1]
        tokentypes_enc = tokentypes_enc[0: max_seq_length - 1]

    # [SEP].
    enc_ids.append(sep_id)
    tokentypes_enc.append(0)

    num_tokens_enc = len(enc_ids)
    # Padding.
    padding_length = max_seq_length - len(enc_ids)
    if padding_length > 0:
        enc_ids.extend([pad_id] * padding_length)
        tokentypes_enc.extend([pad_id] * padding_length)

    pad_mask = ([1] * num_tokens_enc) + ([0] * padding_length)
    pad_mask = np.array(pad_mask, dtype=np.int64)

    return enc_ids, tokentypes_enc, pad_mask


class OpenRetrievalEvidenceDataset(ABC, Dataset):
    """Open Retrieval Evidence dataset class."""

    def __init__(self, task_name, dataset_name, datapath, tokenizer, max_seq_length):
        # Store inputs.
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,
                                                             self.dataset_name))

        args = get_args()
        self.passages_map = make_indexed_dataset(args.indexed_evidence_bert_tokenized_data_path,
                                                 impl=args.data_impl,
                                                 skip_warmup=(not args.mmap_warmup))
        self.title_map = make_indexed_dataset(args.indexed_title_bert_tokenized_data_path,
                                              impl=args.data_impl,
                                              skip_warmup=(not args.mmap_warmup))
        # Process the files.
        # print_rank_0(datapath)
        # self.samples, self.id2text = process_samples_from_single_path(datapath)

        print_rank_0('  >> total number of passages: {}'.format(len(self.passages_map)))

    def __len__(self):
        return len(self.passages_map)

    def __getitem__(self, idx):
        # These indexed datasets follow zero indexing
        text_ids = self.passages_map[idx].tolist()
        title_ids = self.title_map[idx].tolist()

        title_text_ids = [self.tokenizer.cls] + title_ids + [self.tokenizer.sep] + text_ids
        to_be_added_len = 1
        if len(title_text_ids) + to_be_added_len >= 256:
            truncate_len = len(title_text_ids) + to_be_added_len - 256
            title_text_ids = title_text_ids[: -truncate_len]

        title_text_ids.extend([self.tokenizer.sep])

        # idx + 1 is needed because in DPR Wikipedia passages the indexing starts from 1 and not 0.
        sample = {"row_id": idx + 1,
                  "title_text_ids": title_text_ids
                  }

        return sample

    @staticmethod
    def process_samples_from_single_path(filename):
        args = get_args()
        if args.local_rank == 0:
            print(' > Processing {} ...'.format(filename))
        total = 0
        id2text = []

        with open(filename) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            next(reader, None)  # skip the headers

            # Fill some random text tuple in the first index
            id2text.append(("text", "title"))

            for row in reader:
                # file format: doc_id, doc_text, title
                doc_id = int(row[0])
                text = row[1]
                title = row[2]

                # doc_id is specified by the index of the list
                id2text.append((text, title))

                total += 1
                if total % 1000000 == 0:
                    if args.local_rank == 0:
                        print('  > processed {} rows so far ...'.format(total))

        if args.local_rank == 0:
            print(' >> processed {} samples.'.format(len(id2text)))

        return id2text
