# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wikipedia dataset from DPR code for ORQA."""

import csv
import random
import time
from abc import ABC

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron import print_rank_0, get_args, get_tokenizer, mpu
from megatron.data.mask_creation_utils import make_attention_mask
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


# def get_open_retrieval_batch(data_iterator):
#     data = next(data_iterator)
#     row_id = data['row_id'].long().cuda()
#     context = data['context'].long().cuda()
#
#     # TODO: make the context mask a binary one
#     context_mask = (data['context_mask'] < 0.5).cuda()
#     context_types = data['context_types'].long().cuda()
#     context_pad_mask = data['context_pad_mask'].long().cuda()
#
#     return row_id, context, context_mask, context_types, context_pad_mask


def get_open_retrieval_batch(data_iterator):
    data = next(data_iterator)
    row_id = data['row_id'].cuda()
    context = data['context'].cuda()
    context_mask = data['context_mask'].cuda()
    context_types = data['context_types'].cuda()
    context_pad_mask = data['context_pad_mask'].cuda()

    return row_id, context, context_mask, context_types, context_pad_mask


def build_tokens_types_paddings_from_text(row, tokenizer, max_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    title_ids = tokenizer.tokenize(row['title'])
    context_ids = tokenizer.tokenize(row['text'])

    # Appending the title of the context at front
    extended_context_ids = title_ids + [tokenizer.sep_id] + context_ids

    context_ids, context_types, context_pad_mask = build_tokens_types_paddings_from_ids(extended_context_ids,
                                                                                        max_seq_length,
                                                                                        tokenizer.cls,
                                                                                        tokenizer.sep,
                                                                                        tokenizer.pad)
    return context_ids, context_types, context_pad_mask


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


# def build_sample(row_id, context_ids, context_types, context_pad_mask):
#     """Convert to numpy and return a sample consumed by the batch producer."""
#
#     context_ids = np.array(context_ids, dtype=np.int64)
#     context_types = np.array(context_types, dtype=np.int64)
#     context_mask = make_attention_mask(context_ids, context_ids)
#
#     sample = ({
#         'row_id': row_id,
#         'context': context_ids,
#         'context_mask': context_mask,
#         'context_types': context_types,
#         'context_pad_mask': context_pad_mask
#     })
#     return sample


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
        self.passages_map = make_indexed_dataset(args.indexed_evidence_data_path,
                                                 impl=args.data_impl,
                                                 skip_warmup=(not args.mmap_warmup))
        self.title_map = make_indexed_dataset(args.indexed_title_data_path,
                                              impl=args.data_impl,
                                              skip_warmup=(not args.mmap_warmup))
        # Process the files.
        # print_rank_0(datapath)
        # self.samples, self.id2text = self.process_samples_from_single_path(datapath)

        print_rank_0('  >> total number of passages: {}'.format(len(self.passages_map)))

    def __len__(self):
        return len(self.passages_map)

    def __getitem__(self, idx):
        # These indexed datasets follow zero indexing
        text_ids = self.passages_map[idx].tolist()
        title_ids = self.title_map[idx].tolist()

        title_text_ids = [self.tokenizer.cls] + title_ids + [self.tokenizer.sep] + text_ids
        to_be_added_len = 1
        if len(title_text_ids) + to_be_added_len >= 512:
            truncate_len = len(title_text_ids) + to_be_added_len - 512
            title_text_ids = title_text_ids[: -truncate_len]

        title_text_ids.extend([self.tokenizer.sep])

        # idx + 1 is needed because in DPR Wikipedia passages the indexing starts from 1 and not 0.
        sample = {"row_id": idx + 1,
                  "title_text_ids": title_text_ids
                  }

        return sample

    @staticmethod
    def process_samples_from_single_path(filename):
        print_rank_0(' > Processing {} ...'.format(filename))
        total = 0

        rows = []
        id2text = {}

        with open(filename) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            next(reader, None)  # skip the headers
            for row in reader:
                # file format: doc_id, doc_text, title
                doc_id = int(row[0])
                text = row[1]
                title = row[2]

                rows.append({'doc_id': doc_id,
                             'text': text,
                             'title': title})

                assert doc_id not in id2text
                id2text[doc_id] = (text, title)

                total += 1
                if total % 100000 == 0:
                    print_rank_0('  > processed {} rows so far ...'.format(total))

        print_rank_0(' >> processed {} samples.'.format(len(rows)))

        return rows, id2text
