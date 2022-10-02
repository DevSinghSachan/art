#!/bin/bash

CHECKPOINT_PATH=$1
WORLD_SIZE=16
BASE_DIR="/mnt/disks/project"
DATA_DIR="${BASE_DIR}/data"
EVIDENCE_DATA_PATH="${DATA_DIR}/wikipedia-split/psgs_w100.tsv"
VOCAB_FILE="${BASE_DIR}/bert-vocab/bert-large-uncased-vocab.txt"
RETRIEVER_CONFIG="base"

echo "$(basename "${CHECKPOINT_PATH}").pkl"
exit

EMBEDDING_PATH="${BASE_DIR}/embedding-path/art-finetuning-embedding/$(basename "${CHECKPOINT_PATH}").pkl"
QA_FILE_DEV="${DATA_DIR}/qas/nq-dev.csv"
QA_FILE_TEST="${DATA_DIR}/qas/nq-test.csv"

CREATE_EVIDENCE_INDEXES="true"
EVALUATE_RETRIEVER_RECALL="true"

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
        --batch-size 128 \
        --checkpoint-activations \
        --seq-length 512 \
        --seq-length-retriever 256 \
        --max-position-embeddings 512 \
        --load ${CHECKPOINT_PATH} \
        --evidence-data-path ${EVIDENCE_DATA_PATH} \
        --embedding-path ${EMBEDDING_PATH} \
        --indexer-log-interval 1000 \
        --indexer-batch-size 128 \
        --vocab-file ${VOCAB_FILE} \
        --num-workers 2 \
        --fp16 \
        --max-training-rank ${WORLD_SIZE}"


if [ ${CREATE_EVIDENCE_INDEXES} == "true" ];
then
    COMMAND="WORLD_SIZE=${WORLD_SIZE} python ${DISTRIBUTED_ARGS} create_doc_index.py ${OPTIONS} ${CONFIG_ARGS}"
    eval "${COMMAND}"
fi
set +x


OPTIONS="
    --checkpoint-activations \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --load ${CHECKPOINT_PATH} \
    --evidence-data-path ${DATA_DIR} \
    --embedding-path ${EMBEDDING_PATH} \
    --batch-size 16 \
    --seq-length-retriever 256 \
    --vocab-file ${VOCAB_FILE} \
    --qa-file-dev ${QA_FILE_DEV} \
    --qa-file-test ${QA_FILE_TEST} \
    --num-workers 2 \
    --faiss-use-gpu \
    --report-topk-accuracies 1 5 20 100 \
    --fp16 \
    --topk-retrievals 100 \
    --max-training-rank ${WORLD_SIZE}"


if [ ${EVALUATE_RETRIEVER_RECALL} == "true" ];
then
COMMAND="WORLD_SIZE=${WORLD_SIZE} python ${DISTRIBUTED_ARGS} evaluate_open_retrieval.py ${OPTIONS} ${CONFIG_ARGS}"
eval "${COMMAND}"
fi

set +x
