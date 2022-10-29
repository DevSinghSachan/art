import json
import csv
import argparse


def load_dataset(args):
    with open(args.input_path) as fp:
        data = json.load(fp)

    # # condition for interfacing with pyserineni BM25 outputs
    # if isinstance(data, dict):
    #     data =  list(data.values())

    with open(args.output_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for row in data:
            writer.writerow([row['question'], row['answers']])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path')
    parser.add_argument('--output-path')

    args = parser.parse_args()
