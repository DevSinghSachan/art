import jsonlines
import csv

salient_sentences_file_path = "/mnt/disks/project/data/salient-spans-training/filtered_sentences.jsonl"
output_file_path = "/mnt/disks/project/data/salient-spans-training/filtered_sentences.csv"

all_data = []
with jsonlines.open(salient_sentences_file_path) as reader, open(output_file_path, 'w') as writer:
    csvwriter = csv.writer(writer, delimiter="\t")
    for line in reader:
        sentence_text = line["sent_text"]
        if len(sentence_text.split()) <= 30:
            csvwriter.writerow([sentence_text, ["dummy_answer"]])
