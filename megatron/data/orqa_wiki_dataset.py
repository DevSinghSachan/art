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

import re
import random
import time
from abc import ABC

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron import print_rank_0, get_args, get_tokenizer, mpu
from megatron.data.mask_creation_utils import make_attention_mask


def get_open_retrieval_wiki_dataset():
    args = get_args()
    tokenizer = get_tokenizer()

    dataset = OpenRetrievalEvidenceDataset('MSMARCO passage reference file',
                                           'query to passage ids',
                                           args.path_to_msmarco_dev_reference,
                                           tokenizer,
                                           args.seq_length_ret)
    return dataset


def get_open_retrieval_batch(data_iterator):
    data = next(data_iterator)
    row_id = data['row_id'].long().cuda()
    context = data['context'].long().cuda()

    # TODO: make the context mask a binary one
    context_mask = (data['context_mask'] < 0.5).cuda()
    context_types = data['context_types'].long().cuda()
    context_pad_mask = data['context_pad_mask'].long().cuda()

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


def build_sample(row_id, context_ids, context_types, context_pad_mask):
    """Convert to numpy and return a sample consumed by the batch producer."""

    context_ids = np.array(context_ids, dtype=np.int64)
    context_types = np.array(context_types, dtype=np.int64)
    context_mask = make_attention_mask(context_ids, context_ids)

    sample = ({
        'row_id': row_id,
        'context': context_ids,
        'context_mask': context_mask,
        'context_types': context_types,
        'context_pad_mask': context_pad_mask
    })
    return sample


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
        # Process the files.
        print_rank_0(datapath)
        self.query2passage_list = self.load_msmarco_dev_reference(datapath)

        # args = get_args()
        # if args.sample_rate < 1:  # subsample
        #     k = int(len(self.samples) * args.sample_rate)
        #     self.samples = random.sample(self.samples, k)
        #
        # print_rank_0('  >> total number of samples: {}'.format(len(self.samples)))
        self.samples = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        context_ids, context_types, context_pad_mask = build_tokens_types_paddings_from_text(row,
                                                                                             self.tokenizer,
                                                                                             self.max_seq_length)
        sample = build_sample(row['doc_id'],
                              context_ids,
                              context_types,
                              context_pad_mask)
        return sample

    @staticmethod
    def load_msmarco_dev_reference(datapath):
        """Load Reference reference relevant passages
        Args:path_to_reference (str): path to a file to load.
        Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
        """
        print_rank_0(' > Processing {} ...'.format(datapath))

        with open(datapath, 'r') as f:
            qids_to_relevant_passageids = load_reference_from_stream(f)
        return qids_to_relevant_passageids


def load_reference_from_stream(f):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    qids_to_relevant_passageids = {}
    for l in f:
        try:
            l = re.split('[\t\s]', l.strip())
            qid = int(l[0])
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            qids_to_relevant_passageids[qid].append(int(l[2]))
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids