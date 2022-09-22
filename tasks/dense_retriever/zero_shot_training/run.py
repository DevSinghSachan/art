from megatron import get_args
from megatron import get_tokenizer as get_bert_tokenizer
from megatron import print_rank_0
from megatron.model import ARTModel, PreComputedEvidenceDocsRetriever
from tasks.dense_retriever.zero_shot_training.train import accuracy_func_provider
from tasks.dense_retriever.zero_shot_training.train import train


def zero_shot_retriever(dataset_cls):

    def train_dataset_provider():
        """Build train and validation dataset."""
        args = get_args()
        bert_tokenizer = get_bert_tokenizer()

        train_dataset = dataset_cls("OPENQA_DATASET",
                                    "training",
                                    args.train_data,
                                    bert_tokenizer,
                                    args.seq_length_retriever)
        return train_dataset

    def model_provider():
        """Build the model."""
        args = get_args()
        print_rank_0('building ART model for {} ...'.format(args.task))
        evidence_retriever = PreComputedEvidenceDocsRetriever()
        model = ARTModel(evidence_retriever)
        return model

    def single_dataset_provider(datapath):
        args = get_args()
        bert_tokenizer = get_bert_tokenizer()
        name = datapath[0].split('/')[-1].split('.')[0]

        return dataset_cls("OPENQA_DATASET",
                           name,
                           datapath,
                           bert_tokenizer,
                           args.seq_length_retriever)

    def distributed_metrics_func_provider(datapath):
        return accuracy_func_provider(single_dataset_provider, datapath)

    train(train_dataset_provider, model_provider)

    # These arguments can be useful for BEIR benchmark evaluation
    # end_of_epoch_callback_provider=distributed_metrics_func_provider,
    # end_of_training_callback_provider=distributed_metrics_func_provider)


def main():
    args = get_args()

    if args.task == "ZERO-SHOT-RETRIEVER":
        from tasks.dense_retriever.zero_shot_training.train_data_utils import OpenQADataset as dataset_cls
    else:
        raise NotImplementedError('Retrieval task {} is not implemented.'.format(args.task))

    zero_shot_retriever(dataset_cls)
