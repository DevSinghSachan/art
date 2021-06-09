import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron.global_vars import set_global_variables


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    group.add_argument('--task', type=str, required=True, help='Task name.')
    group.add_argument('--epochs', type=int, default=None,
                       help='Number of finetuning epochs. 0 results in evaluation only.')
    group.add_argument('--pretrained-checkpoint', type=str, default=None,
                       help='Pretrained checkpoint used for finetunning.')
    group.add_argument('--keep-last', action='store_true',
                       help='Keep the last batch (maybe incomplete) in the data loader')
    group.add_argument('--train-data', nargs='+', default=None,
                       help='Whitespace separated paths or corpora names for training.')
    group.add_argument('--eval-batch-size', type=int, default=None,
                       help='Eval Batch size per model instance (local batch size). Global batch size is local'
                             ' batch size times data parallel size.')
    return parser


if __name__ == '__main__':

    set_global_variables(extra_args_provider=get_tasks_args,
                         args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'},
                         ignore_unknown_args=False)
    args = get_args()
    initialize_megatron()

    if args.task == "SUPERVISED-RETRIEVER":
        from tasks.dense_retriever.supervised_training.run import main
    elif args.task == "ZERO-SHOT-RETRIEVER":
        from tasks.dense_retriever.zero_shot_training.run import main
    else:
        raise NotImplementedError('Task {} is not implemented.'.format(args.task))

    main()
