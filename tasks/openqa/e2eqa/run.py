from megatron import get_args
from megatron import get_tokenizer, get_t0_tokenizer
from megatron import print_rank_0
from megatron.model import UPRModel, PreComputedEvidenceDocsRetriever
from tasks.openqa.e2eqa.train_e2eqa import accuracy_func_provider
from tasks.openqa.e2eqa.train_e2eqa import train


def open_retrieval_generative_qa(dataset_cls):

    def train_valid_datasets_provider():
        """Build train and validation dataset."""
        args = get_args()
        bert_tokenizer = get_tokenizer()
        t0_tokenizer = get_t0_tokenizer()

        train_dataset = dataset_cls("OPENQA_DATASET",
                                    "training",
                                    args.train_data,
                                    bert_tokenizer,
                                    t0_tokenizer,
                                    args.seq_length_ret,
                                    args.decoder_seq_length)

        valid_dataset = dataset_cls("OPENQA_DATASET",
                                    "validation",
                                    args.valid_data,
                                    bert_tokenizer,
                                    t0_tokenizer,
                                    args.seq_length_ret,
                                    args.decoder_seq_length)

        return train_dataset, valid_dataset

    def model_provider():
        """Build the model."""
        args = get_args()
        print_rank_0('building EMDR2 model for {} ...'.format(args.task))
        evidence_retriever = PreComputedEvidenceDocsRetriever()
        model = UPRModel(evidence_retriever)
        return model

    def single_dataset_provider(datapath):
        args = get_args()
        bert_tokenizer = get_tokenizer()
        t0_tokenizer = get_t0_tokenizer()

        name = datapath[0].split('/')[-1].split('.')[0]

        return dataset_cls("OPENQA_DATASET",
                           name,
                           datapath,
                           bert_tokenizer,
                           t0_tokenizer,
                           args.seq_length_ret,
                           args.decoder_seq_length)

    def distributed_metrics_func_provider(datapath):
        return accuracy_func_provider(single_dataset_provider, datapath)


    train(train_valid_datasets_provider,
          model_provider,
          end_of_epoch_callback_provider=distributed_metrics_func_provider,
          end_of_training_callback_provider=distributed_metrics_func_provider)


def main():
    args = get_args()

    if args.task == "OPENQA":
        from tasks.openqa.e2eqa.train_data_utils import OpenQADataset as dataset_cls
    else:
        raise NotImplementedError('ORQA task {} is not implemented.'.format(
            args.task))

    open_retrieval_generative_qa(dataset_cls)
