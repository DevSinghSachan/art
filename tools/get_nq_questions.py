import jsonlines
import csv

# v1.0-simplified_simplified-nq-train.jsonl can be downloaded from "https://ai.google.com/research/NaturalQuestions/download"
# under Simplified train set.
file_path = "/Users/dsachan/Downloads/v1.0-simplified_simplified-nq-train.jsonl"

output_file_path = "nq-train-307k.csv"

with jsonlines.open(file_path) as reader, open(output_file_path, 'w') as writer:
    csvwriter = csv.writer(writer, delimiter="\t")
    for line in reader:
        question_text = line["question_text"]
        csvwriter.writerow([question_text, ["dummy_answer"]])
