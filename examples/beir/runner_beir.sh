# Provide the path of MS MARCO trained retriever

RETRIEVER_CHECKPOINT=$1

for BEIR_DATASET in scifact scidocs hotpotqa nfcorpus fiqa trec-covid webis-touche2020 hotpotqa quora dbpedia-entity fever climate-fever msmarco nq arguana
do
    echo ${BEIR_DATASET}
    bash examples/beir/embed_and_evaluate_beir.sh ${RETRIEVER_CHECKPOINT} ${BEIR_DATASET}
done
