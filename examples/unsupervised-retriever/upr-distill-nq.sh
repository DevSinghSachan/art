
CONFIG="base"
BASE_DIR="/mnt/disks/project"
DATA_DIR="${BASE_DIR}/data/qas"
TRAIN_DATA="${DATA_DIR}/nq-train.csv"
VALID_DATA="${DATA_DIR}/nq-dev.csv"
TEST_DATA="${DATA_DIR}/nq-test.csv"

QA_FILE_DEV="${BASE_DIR}/data/dpr/retriever/qas/nq-dev.csv"
QA_FILE_TEST="${BASE_DIR}/data/dpr/retriever/qas/nq-test.csv"

EVIDENCE_DATA_PATH="${BASE_DIR}/data/dpr/wikipedia_split/psgs_w100.tsv"
TOPK=128

RETRIEVER_CHKPT_PATH="${BASE_DIR}/checkpoints/mss-emdr2-retriever-base-steps82k"
VOCAB_FILE="${BASE_DIR}/bert-vocab/bert-large-uncased-vocab.txt"

CHECKPOINT_PATH="${BASE_DIR}/checkpoints/temp-xyz-wctx-bf16-indexer"
#rm -rf ${CHECKPOINT_PATH}

mkdir -p ${BASE_DIR}"/embedding-path/upr-finetuning-embedding"

EMBEDDING_PATH="${BASE_DIR}/embedding-path/emdr2-finetuning-embedding/psgs_w100-retriever-nq-${CONFIG}-topk${TOPK}-epochs10-bsize16-indexer-upr-distill.pkl"
#rm ${EMBEDDING_PATH}

ORIGINAL_EMBEDDING_PATH="${BASE_DIR}/embedding-path/psgs_w100_univ-realm-retriever-base-steps82k_full-wikipedia_base.pkl"

# Copy the path
if [ -f ${EMBEDDING_PATH} ]; then
    echo "${EMBEDDING_PATH} exists. Not copying file"
else
    echo "Copying file"
    cp ${ORIGINAL_EMBEDDING_PATH} ${EMBEDDING_PATH}
fi

DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node 16 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"
#DISTRIBUTED_ARGS="-m pdb"

WORLD_SIZE=16

function config_base() {
    export CONFIG_ARGS="--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--kv-channels 64 \
--ffn-hidden-size 3072 \
--model-parallel-size 1"
}


if [ ${CONFIG} == "base" ]; then
    config_base
else
    echo "Invalid model configuration"
    exit 1
fi

OPTIONS=" \
          --train-data $TRAIN_DATA \
          --valid-data $VALID_DATA \
          --test-data $TEST_DATA \
          --qa-file-dev ${QA_FILE_DEV} \
          --qa-file-test ${QA_FILE_TEST} \
          --evidence-data-path ${EVIDENCE_DATA_PATH} \
          --indexed-evidence-bert-tokenized-data-path ${BASE_DIR}/data/evidence-wikipedia-indexed-mmap/bert/wikipedia-evidence_text_document \
          --indexed-title-bert-tokenized-data-path ${BASE_DIR}/data/evidence-wikipedia-indexed-mmap/bert/wikipedia-evidence_title_document \
          --indexed-evidence-t0-tokenized-data-path ${BASE_DIR}/data/evidence-wikipedia-indexed-mmap/t0/wikipedia-evidence-t0_text_document \
          --indexed-title-t0-tokenized-data-path ${BASE_DIR}/data/evidence-wikipedia-indexed-mmap/t0/wikipedia-evidence-t0_title_document \
          --save-interval 500 \
          --save ${CHECKPOINT_PATH} \
          --load ${CHECKPOINT_PATH} \
          --pretrained-dualencoder-load ${RETRIEVER_CHKPT_PATH} \
          --embedding-path ${EMBEDDING_PATH} \
          --log-interval 20 \
          --eval-interval 500 \
          --eval-iters 10 \
          --weight-decay 1.0e-1 \
          --seq-length 512 \
          --seq-length-ret 256 \
          --decoder-seq-length 32 \
          --max-position-embeddings 512 \
          --fp16 \
          --vocab-file $VOCAB_FILE \
          --model-parallel-size 1 \
          --num-workers 2 \
          --distributed-backend nccl \
          --checkpoint-activations \
          --task OPENQA \
          --tokenizer-type BertWordPieceLowerCase \
          --epochs 10 \
          --sample-rate 1.0 \
          --batch-size 1 \
          --eval-batch-size 1 \
          --lr 2e-5 \
          --warmup 0.01 \
          --DDP-impl local \
          --lr-decay-style linear \
          --max-training-rank ${WORLD_SIZE} \
          --faiss-use-gpu \
          --topk-retrievals ${TOPK} \
          --report-topk-accuracies 1 5 20 50 100 \
          --upr-distillation-training \
          --retriever-score-scaling \
          --update-retriever \
          --allow-trivial-doc \
          --shard-size 32 \
          --initialize-t0-model-tokenizer-evidence \
          --t0-model-in-bf16 \
          --index-reload-interval 500 \
          --compute-fresh-evidence-embeddings "


COMMAND="WORLD_SIZE=${WORLD_SIZE} python ${DISTRIBUTED_ARGS} tasks/run.py ${OPTIONS} ${CONFIG_ARGS}"
eval "${COMMAND}"
exit
