import argparse
import csv

# original_evidence_path = "/home/sachande/pyserini/collections/msmarco-passage/collection.tsv"


def convert_collection(args):
    print('Converting collection...')
    with open(args.collection_path, encoding='utf-8') as reader, \
            open(args.output_path, 'w', encoding='utf-8') as writer:

        csvwriter = csv.writer(writer, delimiter="\t")
        csvwriter.writerow(["id", "text", "title"])

        for i, line in enumerate(reader):
            doc_id, doc_text = line.rstrip().split('\t')

            csvwriter.writerow([doc_id, doc_text, "dummy_title"])

            if i % 100000 == 0:
                print(f'Processed {i:,} docs and writing into file.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MSMARCO tsv passage collection into jsonl files for Anserini.')
    parser.add_argument('--collection-path', required=True, help='Path to MS MARCO tsv collection.')
    parser.add_argument('--output-path', required=True, help='Output Path.')

    args = parser.parse_args()

    convert_collection(args)
    print('Done!')
