BASE_DIR=$1

echo "BASE_DIR=${BASE_DIR}"

# Create directories
mkdir "${BASE_DIR}/checkpoints"
mkdir "${BASE_DIR}/embedding-path"
mkdir "${BASE_DIR}/bert-vocab"
mkdir "${BASE_DIR}/data"

echo "[Step 1/6] Downloading  BERT-large-uncased vocabulary file"
wget https://www.dropbox.com/s/ttblv1uggd4cijt/bert-large-uncased-vocab.txt
mv bert-large-uncased-vocab.txt "${BASE_DIR}/bert-vocab"

echo "[Step 2/6] Download Wikipedia evidence file"
mkdir "${BASE_DIR}/data/wikipedia-split"
wget https://www.dropbox.com/s/bezryc9win2bha1/psgs_w100.tar.gz
tar -xvzf psgs_w100.tar.gz --directory "${BASE_DIR}/data/wikipedia-split"
rm psgs_w100.tar.gz

echo "[Step 3/6] Download the Masked Salient Span (MSS) pre-trained retriever checkpoint"
wget https://www.dropbox.com/s/069xj395ftxv4hz/mss-emdr2-retriever-base-steps82k.tar.gz
tar -xvzf mss-emdr2-retriever-base-steps82k.tar.gz --directory "${BASE_DIR}/checkpoints"
rm mss-emdr2-retriever-base-steps82k.tar.gz
mv "${BASE_DIR}/checkpoints/mss-emdr2-retriever-base-steps82k" "${BASE_DIR}/checkpoints/mss-retriever-base"

echo "[Step 4/6] Download evidence embedding pre-computed from the MSS pre-trained retriever"
wget https://www.dropbox.com/s/y7rg8u41yavje0y/psgs_w100_emdr2-retriever-base-steps82k_full-wikipedia_base.pkl
mv psgs_w100_emdr2-retriever-base-steps82k_full-wikipedia_base.pkl "${BASE_DIR}/embedding-path/psgs_w100-mss-retriever-full-wikipedia-base.pkl"

echo "[Step 5/6] Download pre-tokenized Wikipedia passage embeddings for BERT and T0 tokenizations"
mkdir "${BASE_DIR}/data/evidence-wikipedia-indexed-mmap"

wget https://www.dropbox.com/s/yxsne7qzz848pk4/indexed-evidence-bert-tokenized.tar.gz
tar -xvzf indexed-evidence-bert-tokenized.tar.gz --directory "${BASE_DIR}/data/evidence-wikipedia-indexed-mmap"
rm indexed-evidence-bert-tokenized.tar.gz

wget https://www.dropbox.com/s/4tvvll8qeso7fal/indexed-evidence-t0-tokenized.tar.gz
tar -xvzf indexed-evidence-t0-tokenized.tar.gz --directory "${BASE_DIR}/data/evidence-wikipedia-indexed-mmap"
rm indexed-evidence-t0-tokenized.tar.gz

echo "[Step 6/6] Download question-answer training and evaluation datasets"
wget https://www.dropbox.com/s/yj7hukwyl04hvs3/qas.tar.gz
tar -xvzf qas.tar.gz --directory "${BASE_DIR}/data"
rm qas.tar.gz


# echo "Do pip install of required packages"
#pip install transformers jsonlines sentencepiece spacy
