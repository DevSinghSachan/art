
# Details for MS MARCO experiments

The codebase in this branch is not very but should we working for MS MARCO experiments. 


## Data Download

* Download the `msmarco-train.csv` file [(url)](https://www.dropbox.com/s/jufz5g88w5v07qc/qas.tar.gz)

* Download the `msmarco-collection.tsv` file [(url)](https://www.dropbox.com/s/9542h4ocl8nw6bt/msmarco-collection.tar.gz). This is the evidence file.

* Download the MSS initialized precomputed evidence embedding for MSMARCO [(url)](https://www.dropbox.com/s/3yk8n4spik0e6x2/msmarco-mss-base-emdr2-steps82k.pkl).

* Download the MSS retriever weights and BERT-large vocab files as mentioned in the instructions in the `main` branch.

* Download both the BERT and T0 pre-tokenized MS MARCO evidence files [(url)](https://www.dropbox.com/s/2t4xaqcjcgs527g/evidence-msmarco-indexed-mmap.tar.gz).

* Download the QUERIES [(url)](https://www.dropbox.com/s/ihv8zch28i64v1u/queries.dev.small.tsv) and QRELS FILES [(url)](https://www.dropbox.com/s/1n9nh3tj37dkj3q/qrels.dev.small.tsv).

* Install `trec_eval` (https://github.com/usnistgov/trec_eval) in the same directory where the experiment runs.

* We provide an example script for training. Change the data paths and run the script as
```bash
bash examples/zero-shot-retriever/art-msmarco.sh
``` 

* The trained checkpoint can be evaluated as
```bash
bash examples/helper-scripts/create_msmarco_indexes_and_evaluate.sh
```

## Helpful Scripts

* Indexing MS MARCO evidence using BERT Tokenizer:
```bash
python tools/create_evidence_indexed_dataset.py --input /home/sachande/data/msmarco-collection/msmarco-collection.tsv --tsv-k
eys text --tokenizer-type BertWordPieceLowerCase --vocab-file /home/sachande/bert-vocab/bert-large-uncased-vocab.txt --output-prefix msmarco-evidence --workers 25
```

* Indexing MS MARCO evidence using T0 Tokenizer:
```bash
python tools/create_evidence_indexed_dataset_t0.py --input /home/sachande/data/msmarco-collection/msmarco-collection.tsv --tsv-keys text  --output-prefix msmarco-evidence-t0 --workers 25
```

* Get MS Marco train questions (MS MARCO passage data is downloaded from pyserini toolkit.)
```bash
python tools/get_msmarco_questions.py --train-questions-path /home/sachande/pyserini/collections/msmarco-passage/queries.train.tsv --qrels-train-path /home/sachande/pyserini/collections/msmarco-passage/qrels.train.tsv --output-path /home/sachande/pyserini/collections/msmarco-passage/msmarco-train.csv
```