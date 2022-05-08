pip install transformers jsonlines sentencepiece spacy
export TRANSFORMERS_CACHE="${HOME}/transformers-cache"

CONFIG="base"
BASE_DIR="/home/sachande"
DATA_DIR="${BASE_DIR}/data/qas"
TRAIN_DATA="${DATA_DIR}/msmarco-train.csv"
# These are not really important, so putting some dummy values here
VALID_DATA="${DATA_DIR}/nq-dev.csv"
TEST_DATA="${DATA_DIR}/nq-test.csv"

QA_FILE_DEV="${BASE_DIR}/pyserini/collections/msmarco-passage/queries.dev.small.tsv"
MSMARCO_DEV_REFERENCE_PATH="${BASE_DIR}/pyserini/collections/msmarco-passage/qrels.dev.small.tsv"

EVIDENCE_DATA_PATH="${BASE_DIR}/data/msmarco-collection/msmarco-collection.tsv"
TOPK=32

RETRIEVER_CHKPT_PATH="${BASE_DIR}/checkpoints/mss-emdr2-retriever-base-steps82k"
VOCAB_FILE="${BASE_DIR}/bert-vocab/bert-large-uncased-vocab.txt"

CHECKPOINT_PATH="${BASE_DIR}/checkpoints/msmarco-mss-base-init-bs64-topk32-epochs10"
#rm -rf ${CHECKPOINT_PATH}

mkdir -p ${BASE_DIR}"/embedding-path/upr-finetuning-embedding"

EMBEDDING_PATH="${BASE_DIR}/embedding-path/upr-finetuning-embedding/msmarco-${CONFIG}-topk${TOPK}-epochs10-bsize64-indexer-upr-distill.pkl"
#rm ${EMBEDDING_PATH}

ORIGINAL_EMBEDDING_PATH="${BASE_DIR}/embedding-path/msmarco-mss-base-emdr2-steps82k.pkl"

# Copy the path
if [ -f ${EMBEDDING_PATH} ]; then
    echo "${EMBEDDING_PATH} exists. Not copying file"
else
    echo "Copying file"
    cp ${ORIGINAL_EMBEDDING_PATH} ${EMBEDDING_PATH}
fi

DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"
#DISTRIBUTED_ARGS="-m pdb"
WORLD_SIZE=8


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
          --evidence-data-path ${EVIDENCE_DATA_PATH} \
          --indexed-evidence-bert-tokenized-data-path ${BASE_DIR}/data/evidence-msmarco-indexed-mmap/msmarco-evidence_text_document \
          --indexed-evidence-t0-tokenized-data-path ${BASE_DIR}/data/evidence-msmarco-indexed-mmap/t0/msmarco-evidence-t0_text_document \
          --save-interval 1000 \
          --save ${CHECKPOINT_PATH} \
          --load ${CHECKPOINT_PATH} \
          --pretrained-dualencoder-load ${RETRIEVER_CHKPT_PATH} \
          --embedding-path ${EMBEDDING_PATH} \
          --log-interval 20 \
          --eval-interval 1000 \
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
          --batch-size 8 \
          --eval-batch-size 8 \
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
          --shard-size 4 \
          --initialize-t0-model-tokenizer-evidence \
          --t0-model-in-bf16 \
          --index-reload-interval 1000 \
          --path-to-msmarco-dev-reference ${MSMARCO_DEV_REFERENCE_PATH} \
          --compute-fresh-evidence-embeddings "


COMMAND="WORLD_SIZE=${WORLD_SIZE} python ${DISTRIBUTED_ARGS} tasks/run.py ${OPTIONS} ${CONFIG_ARGS}"
eval "${COMMAND}"
exit
