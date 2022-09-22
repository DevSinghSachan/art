from abc import ABC
import csv
import numpy as np
from torch.utils.data import Dataset
from megatron import print_rank_0, get_args
from megatron.data.mask_creation_utils import make_attention_mask


def build_tokens_types_paddings_from_text(src_text, bert_tokenizer, max_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    src_text_ids = bert_tokenizer.tokenize(src_text)
    return build_tokens_types_paddings_from_ids(src_text_ids,
                                                max_seq_length,
                                                bert_tokenizer.cls,
                                                bert_tokenizer.sep,
                                                bert_tokenizer.pad)


def build_tokens_types_paddings_from_ids(src_ids, max_seq_length, cls_id, sep_id, pad_id):
    enc_ids = []
    tokentypes_enc = []

    # [CLS].
    enc_ids.append(cls_id)
    tokentypes_enc.append(0)

    # A.
    len_src = len(src_ids)
    enc_ids.extend(src_ids)
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

    return enc_ids, tokentypes_enc, num_tokens_enc


def build_sample(query_uid, token_ids, token_types, num_tokens, prefixed_query_text, reference):
    token_ids = np.array(token_ids, dtype=np.int64)
    token_types = np.array(token_types, dtype=np.int64)
    token_mask = make_attention_mask(token_ids, token_ids)

    sample = ({
        'query_uid': query_uid,
        'query_ids_bert': token_ids,
        'query_types': token_types,
        'query_mask_bert': token_mask,
        'prefixed_query_text': prefixed_query_text,
        'reference': reference
    })
    return sample


class OpenQADataset(ABC, Dataset):

    def __init__(self, task_name, dataset_name, datapaths,
                 bert_tokenizer,
                 max_seq_length):
        args = get_args()
        self.np_rng = np.random.RandomState(seed=args.seed)
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_length = max_seq_length
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,
                                                             self.dataset_name))
        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        print_rank_0(string)
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(self.process_samples_from_single_path(datapath))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        prefixed_query_text = "{} {}".format("Question:", raw_sample['question'])
        ques_tokens, tokentypes_enc, num_tokens_ques = build_tokens_types_paddings_from_text(raw_sample['question'],
                                                                                             self.bert_tokenizer,
                                                                                             self.max_seq_length)
        sample = build_sample(raw_sample['uid'],
                              ques_tokens,
                              tokentypes_enc,
                              num_tokens_ques,
                              prefixed_query_text,
                              raw_sample['answers'])
        return sample

    @staticmethod
    def process_samples_from_single_path(filename):
        print_rank_0(' > Processing {} ...'.format(filename))
        samples = []
        total = 0

        with open(filename, 'r') as ifile:
            reader = csv.reader(ifile, delimiter='\t')
            for row in reader:
                question = row[0]
                answers = eval(row[1])

                total += 1
                # We are keeping the uid as negative to avoid the conflict with evidence ID
                sample = {'uid': -1 * total, 'question': question, 'answers': answers}
                samples.append(sample)

                if total % 1000 == 0:
                    print_rank_0('  > processed {} so far ...'.format(total))

        print_rank_0(' >> processed {} samples.'.format(len(samples)))
        return samples
