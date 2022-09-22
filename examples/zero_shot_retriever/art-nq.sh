
# CHANGE THIS PATH TO POINT TO DATA / CHECKPOINTS
BASE_DIR="/mnt/disks/project"

BERT_VOCAB_FILE="${BASE_DIR}/bert-vocab/bert-large-uncased-vocab.txt"
DATA_DIR="${BASE_DIR}/data"
TRAIN_DATA="${DATA_DIR}/qas/nq-train.csv"
QA_FILE_DEV="${DATA_DIR}/qas/nq-dev.csv"
QA_FILE_TEST="${DATA_DIR}/qas/nq-test.csv"
EVIDENCE_DATA_PATH="${DATA_DIR}/wikipedia-split/psgs_w100.tsv"

TOPK_DOCUMENTS=32
WORLD_SIZE=16
EPOCHS=10
RETRIEVER_CONFIG="base"

PRETRAINED_RETRIEVER_CHECKPOINT="${BASE_DIR}/checkpoints/mss-retriever-${RETRIEVER_CONFIG}"
PRETRAINED_RETRIEVER_EMBEDDING_PATH="${BASE_DIR}/embedding-path/psgs_w100-mss-retriever-full-wikipedia-${RETRIEVER_CONFIG}.pkl"

mkdir -p ${BASE_DIR}"/embedding-path/art-finetuning-embedding"
CHECKPOINT_PATH="${BASE_DIR}/checkpoints/nq-mss-${RETRIEVER_CONFIG}-init"
EMBEDDING_PATH="${BASE_DIR}/embedding-path/art-finetuning-embedding/psgs_w100-retriever-nq-${RETRIEVER_CONFIG}-topk${TOPK_DOCUMENTS}-epochs${EPOCHS}-bsize64-indexer.pkl"


# Copy the path
if [ -f ${EMBEDDING_PATH} ]; then
    echo "${EMBEDDING_PATH} exists. Not copying file"
else
    echo "Copying file"
    cp ${PRETRAINED_RETRIEVER_EMBEDDING_PATH} ${EMBEDDING_PATH}
fi

DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node ${WORLD_SIZE} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"

function config_base() {
    export CONFIG_ARGS="--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--kv-channels 64 \
--ffn-hidden-size 3072 \
--model-parallel-size 1"
}

function config_large() {
    export CONFIG_ARGS="--num-layers 24 \
--hidden-size 1024 \
--num-attention-heads 16 \
--kv-channels 64 \
--ffn-hidden-size 4096 \
--model-parallel-size 1"
}

if [ ${RETRIEVER_CONFIG} == "base" ]; then
    config_base
elif [ ${RETRIEVER_CONFIG} == "large" ]; then
    config_large
else
    echo "Invalid model configuration"
    exit 1
fi


OPTIONS=" \
          --train-data $TRAIN_DATA \
          --qa-file-dev ${QA_FILE_DEV} \
          --qa-file-test ${QA_FILE_TEST} \
          --evidence-data-path ${EVIDENCE_DATA_PATH} \
          --indexed-evidence-bert-tokenized-data-path ${DATA_DIR}/evidence-wikipedia-indexed-mmap/bert/wikipedia-evidence-bert_text_document \
          --indexed-title-bert-tokenized-data-path ${DATA_DIR}/evidence-wikipedia-indexed-mmap/bert/wikipedia-evidence-bert_title_document \
          --indexed-evidence-t0-tokenized-data-path ${DATA_DIR}/evidence-wikipedia-indexed-mmap/t0/wikipedia-evidence-t0_text_document \
          --indexed-title-t0-tokenized-data-path ${DATA_DIR}/evidence-wikipedia-indexed-mmap/t0/wikipedia-evidence-t0_title_document \
          --save-interval 500 \
          --save ${CHECKPOINT_PATH} \
          --load ${CHECKPOINT_PATH} \
          --pretrained-dualencoder-load ${PRETRAINED_RETRIEVER_CHECKPOINT} \
          --embedding-path ${EMBEDDING_PATH} \
          --vocab-file ${BERT_VOCAB_FILE} \
          --log-interval 20 \
          --eval-interval 500 \
          --weight-decay 1.0e-1 \
          --seq-length 512 \
          --seq-length-retriever 256 \
          --max-position-embeddings 512 \
          --fp16 \
          --num-workers 2 \
          --distributed-backend nccl \
          --checkpoint-activations \
          --task ZERO-SHOT-RETRIEVER \
          --tokenizer-type BertWordPieceLowerCase \
          --epochs ${EPOCHS} \
          --sample-rate 1.0 \
          --batch-size 4 \
          --eval-batch-size 1 \
          --lr 2e-5 \
          --warmup 0.01 \
          --DDP-impl local \
          --lr-decay-style linear \
          --max-training-rank ${WORLD_SIZE} \
          --topk-retrievals ${TOPK_DOCUMENTS} \
          --report-topk-accuracies 1 5 20 50 100 \
          --art-training \
          --retriever-score-scaling \
          --update-retriever \
          --allow-trivial-doc \
          --shard-size 16 \
          --initialize-t0-model-tokenizer-evidence \
          --t0-model-in-bf16 \
          --index-reload-interval 500 \
          --compute-fresh-evidence-embeddings "


COMMAND="WORLD_SIZE=${WORLD_SIZE} python ${DISTRIBUTED_ARGS} tasks/run.py ${OPTIONS} ${CONFIG_ARGS}"
eval "${COMMAND}"
exit
