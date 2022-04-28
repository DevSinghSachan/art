import jsonlines

salient_sentences_file_path = "/mnt/disks/project/data/salient_spans_training/filtered_sentences.jsonl"
output_file_path = "/mnt/disks/project/data/salient_spans_training/filtered_sentences.csv"

all_data = []
with jsonlines.open(salient_sentences_file_path) as reader, open(output_file_path, 'w') as writer:
    for line in reader:
        sentence_text = reader["sent_text"]
        writer.write("{}\t{}\n".format(sentence_text, "dummy_answer"))

