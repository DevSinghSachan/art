#!/bin/bash

BASE_DIR="/home/sachande"
DATA_DIR="${BASE_DIR}/data/msmarco-collection/msmarco-collection.tsv"
VOCAB_FILE="${BASE_DIR}/bert-vocab/bert-large-uncased-vocab.txt"

EMBEDDING_PATH="$BASE_DIR/embedding-path/upr-finetuning-embedding/msmarco-base-topk4-epochs10-bsize512-indexer-upr-distill-step5500.pkl"


CHECKPOINT_PATH="$BASE_DIR/checkpoints/msmarco-mss-base-init-bs512-topk4-epochs10-retriever"


DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"
NPROC=8
#DISTRIBUTED_ARGS="-m pdb"

QA_FILE_DEV="${BASE_DIR}/pyserini/collections/msmarco-passage/queries.dev.small.tsv"
MSMARCO_DEV_REFERENCE_PATH="${BASE_DIR}/pyserini/collections/msmarco-passage/qrels.dev.small.tsv"

CREATE_EVIDENCE_INDEXES="true"
EVALUATE_RETRIEVER_RECALL="true"

OPTIONS="--num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --batch-size 128 \
    --checkpoint-activations \
    --seq-length 512 \
    --seq-length-ret 256 \
    --max-position-embeddings 512 \
    --load ${CHECKPOINT_PATH} \
    --evidence-data-path ${DATA_DIR} \
    --embedding-path ${EMBEDDING_PATH} \
    --indexer-log-interval 1000 \
    --indexer-batch-size 128 \
    --vocab-file ${VOCAB_FILE} \
    --num-workers 2 \
    --fp16 \
    --indexed-evidence-bert-tokenized-data-path ${BASE_DIR}/data/evidence-msmarco-indexed-mmap/msmarco-evidence_text_document \
    --max-training-rank ${NPROC}"


if [ ${CREATE_EVIDENCE_INDEXES} == "true" ];
then
    COMMAND="WORLD_SIZE=8 python ${DISTRIBUTED_ARGS} create_doc_index.py ${OPTIONS}"
    eval "${COMMAND}"
fi
set +x


OPTIONS="--num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --checkpoint-activations \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --load ${CHECKPOINT_PATH} \
    --evidence-data-path ${DATA_DIR} \
    --embedding-path ${EMBEDDING_PATH} \
    --batch-size 16 \
    --seq-length-ret 256 \
    --vocab-file ${VOCAB_FILE} \
    --qa-file-dev ${QA_FILE_DEV} \
    --num-workers 2 \
    --faiss-use-gpu \
    --report-topk-accuracies 1 5 10 20 50 100 1000 \
    --fp16 \
    --topk-retrievals 1000 \
    --path-to-msmarco-dev-reference ${MSMARCO_DEV_REFERENCE_PATH} \
    --save-topk-outputs-path temp-topk-outputs-path \
    --max-training-rank ${NPROC}"


if [ ${EVALUATE_RETRIEVER_RECALL} == "true" ];
then
COMMAND="WORLD_SIZE=${NPROC} python ${DISTRIBUTED_ARGS} evaluate_open_retrieval.py ${OPTIONS}"
eval "${COMMAND}"
fi

set +x
