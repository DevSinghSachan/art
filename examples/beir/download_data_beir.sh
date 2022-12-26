BASE_DIR=$1

echo "BASE_DIR=${BASE_DIR}"

# Create directories
mkdir -p "${BASE_DIR}/data"
mkdir -p "${BASE_DIR}/checkpoints"


echo "[Step 1/4] Download the ART retriever trained on MS MARCO questions"
wget https://www.dropbox.com/s/2huz1evykey5nno/msmarco-mss-base-init-bs512-topk4-epochs10.tar.gz
tar -xvzf msmarco-mss-base-init-bs512-topk4-epochs10.tar.gz --directory "${BASE_DIR}/checkpoints"
rm msmarco-mss-base-init-bs512-topk4-epochs10.tar.gz

echo "[Step 2/4] Download BEIR evidence files"
mkdir "${BASE_DIR}/data/wikipedia-split"
wget https://www.dropbox.com/s/ncsrjvn649yqwnr/evidence-beir.tar.gz
tar -xvzf evidence-beir.tar.gz --directory "${BASE_DIR}/data/"
rm evidence-beir.tar.gz


echo "[Step 3/4] Download pre-tokenized Wikipedia passage embeddings for BERT and T0 tokenizations"
wget https://www.dropbox.com/s/roq4ayxllc5xc99/evidence-beir-mmap.tar.gz
tar -xvzf evidence-beir-mmap.tar.gz --directory "${BASE_DIR}/data/"
rm evidence-beir-mmap.tar.gz


echo "[Step 4/4] Download BEIR evaluation datasets"
wget https://www.dropbox.com/s/rx2t8kbuk3zov8i/BEIR.tar.gz
tar -xvzf BEIR.tar.gz --directory "${BASE_DIR}/data"
rm BEIR.tar.gz

echo "Do pip install of required packages"
pip install pytrec_eval
