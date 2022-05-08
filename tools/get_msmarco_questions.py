import argparse
import csv

def select_valid_questions(args):
    print('Selecting questions from training file whose annotations exist in qrel file...')

    qid_to_passage_id = {}
    with open(args.qrels_train_path) as reader:
        for line in reader:
            fields = line.rstrip().split('\t')
            qid_to_passage_id[int(fields[0])] = int(fields[2])

    with open(args.train_questions_path, encoding='utf-8') as reader, \
            open(args.output_path, 'w', encoding='utf-8') as writer:

        csvwriter = csv.writer(writer, delimiter="\t")

        for i, line in enumerate(reader):
            qid, question_text = line.rstrip().split('\t')

            if int(qid) in qid_to_passage_id:
                csvwriter.writerow([question_text, ["dummy_answer"]])

            if i % 100000 == 0:
                print(f'Processed {i:,} docs and writing into file.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MSMARCO tsv passage collection into jsonl files for Anserini.')
    parser.add_argument('--train-questions-path', required=True, help='Path to MS MARCO training questions.')
    parser.add_argument('--qrels-train-path', required=True, help='Path to MS MARCO qrel file for training questions.')
    parser.add_argument('--output-path', required=True, help='Output Path to save questions.')

    args = parser.parse_args()

    select_valid_questions(args)
    print('Done!')
