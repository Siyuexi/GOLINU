"""
input: finetune data from LANCE
output: data form of codex
train: 106381
java_code_valid: 13259
test: 12019
"""

import random

import pandas as pd
import re
import json

DELIMITER = " <###code###> "


def convert_LANCE_data_for_embedding(df, output_file_path):

    print("data num:", len(df))

    output_data = []

    for i, row in df.iterrows():

        sample_id = row["id"]
        input_code = row["input_code"]
        reformatted_input_code = row["reformatted_input_code"]
        output_logging_code_labels = row["logging_code_labels"]

        # meta data consists of reformatted code and logging code labels
        output_data.append({'prompt': input_code, 'metadata': str(sample_id) + DELIMITER + reformatted_input_code
                                                              + DELIMITER + output_logging_code_labels})

    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        for item in output_data:
            line = json.dumps(item)
            json_file.write(line+'\n')


def merge_train_validation_embeddings(train_embedding_file, valid_embedding_file, output_file_path):

    json_objects = []

    with open(train_embedding_file) as json_file1:
        for line in json_file1:
            json_objects.append(json.loads(line))

    with open(valid_embedding_file) as json_file2:
        for line in json_file2:
            json_objects.append(json.loads(line))

    with open(output_file_path, 'w') as outfile:
        for json_obj in json_objects:
            outfile.write(json.dumps(json_obj) + '\n')

    return


if __name__ == '__main__':

    df = pd.read_csv("reformatted_data/reformatted_all_train.tsv", sep='\t')
    output_file_path = "embeddings/input/LANCE_train_all_embedding_input.jsonl"
    convert_LANCE_data_for_embedding(df, output_file_path)
    
    df = pd.read_csv("reformatted_data/reformatted_all_test.tsv", sep='\t')
    output_file_path = "embeddings/input/LANCE_test_all_embedding_input.jsonl"
    convert_LANCE_data_for_embedding(df, output_file_path)

    train_embedding_file = "embeddings/output/LANCE_train_all_embedding_input_samples.0.jsonl"
    valid_embedding_file = "embeddings/output/LANCE_eval_all_embedding_input_samples.0.jsonl"
    output_file_path = "embeddings/output/LANCE_evalplustrain_all_embedding_input_samples.0.jsonl"
    merge_train_validation_embeddings(train_embedding_file, valid_embedding_file, output_file_path)


