# Provide the path of MS MARCO trained retriever

RETRIEVER_CHECKPOINT=$1


for CQADUPSTACK_DATASET in android english gaming gis mathematica physics programmers stats tex unix webmasters wordpress
do
    echo ${CQADUPSTACK_DATASET}
    bash create_evidence_embeddings_and_evaluate_msmarco_beir_cqadupstack.sh ${RETRIEVER_CHECKPOINT} ${CQADUPSTACK_DATASET}
done
