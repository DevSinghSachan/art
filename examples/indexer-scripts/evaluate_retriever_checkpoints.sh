
DATA_DIR="/mnt/disks/project/art-open-source/checkpoints/"

# This will loop through checkpoints in the directory and call a script to create evidence embeddings

for DIR_PATH in "${DATA_DIR}"/*
do
    if [ -d ${DIR_PATH} ]
    then
        bash examples/indexer-scripts/create_evidence_embeddings_and_evaluate.sh ${DIR_PATH}
    fi
done
