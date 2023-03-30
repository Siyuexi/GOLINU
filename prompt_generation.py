"""
input: embedding file of train and test
output: prompt file of test file
"""


from tqdm import tqdm
import json
import torch
import numpy as np
import os
from torch.nn.functional import cosine_similarity
import re
from embedding_generation import DELIMITER


def retrieve(test, train, batch_size, topK, instruct, prompt_delimiter, completion_delimiter):
    train_contexts, train_completions, train_embeddings = zip(*train)
    test_contexts, test_completions, test_embeddings = zip(*test)
    train_embeddings = torch.from_numpy(np.asarray(train_embeddings, dtype=np.float)).float()
    test_embeddings = torch.from_numpy(np.asarray(test_embeddings, dtype=np.float)).float()
    prompts = []
    sims = []
    test_embeddings = test_embeddings.unsqueeze(1)
    for i in tqdm(range(0, len(train_contexts), batch_size)):
        tmp = cosine_similarity(test_embeddings, train_embeddings[i:i+batch_size], dim=-1)
        sims.append(tmp)
    sims = torch.cat(sims, 1)
    print(sims.shape)
    _, indices = sims.topk(topK, dim=1)
    print(indices.shape)
    for i, q in enumerate(test_contexts):
        top = indices[i]
        examples = []
        for j in top:
            examples.append((train_contexts[j], train_completions[j]))
        # reverse order
        examples.reverse()
        prompt = convert_to_prompt(examples, q, instruct, prompt_delimiter, completion_delimiter)
        prompts.append(prompt)
        # if len(test_completions) > 0:
        #     prompts.append(json.dumps({"context": prompt, "metadata": test_completions[i]}))
        # else:
        #     prompts.append(json.dumps({"context": prompt}))
    # send_telemetry(TelemetryMetric.DTE_INFERENCE_SAMPLES, len(prompts))
    return prompts


def convert_to_prompt(examples, query, instruct, prompt_delimiter, completion_delimiter):
    prompt_str = ""
    for item in examples:
        prompt_str += f"{prompt_delimiter}{item[0]}\n{completion_delimiter}{item[1]}\n\n"
    if instruct:
        instruct = instruct + '\n\n'
    else:
        instruct = ""
    prompt = f"{instruct}{prompt_str}{prompt_delimiter}{query}\n"
    return prompt


def read_embedding_json(file_path, top_num=None):

    embedding_list = []
    logging_label_list = []
    reformatted_code_list = []

    with open(file_path, 'r') as f:

        lines = f.readlines()

        if top_num is not None:
            lines = lines[:top_num]
        print("embedding data num:", len(lines))

        for line in lines:

            line = json.loads(line)

            embedding = line['data'][0]['embedding']
            embedding_list.append(embedding)

            metadata = line['metadata']
            sample_id, reformatted_code, logging_code_label = metadata.rsplit(DELIMITER.strip(), 2)
            sample_id = int(sample_id)

            # add context code
            # reformatted_code_lines = re.split(r"<line\d+>", reformatted_code)
            # logging_line_id = int(re.match(r"<line(\d+)>", logging_code_label).group(1))
            # above_cxt = reformatted_code_lines[logging_line_id - 1] if logging_line_id - 1 >= 0 else ""
            # below_cxt = reformatted_code_lines[logging_line_id + 1] if logging_line_id + 1 <= len(reformatted_code_list) else ""
            # logging_code_label = "<above_cxt> " + above_cxt + \
            #                      " <logging_start> " + logging_code_label + " <logging_end> " \
            #                      + below_cxt + " <below_cxt>"
            logging_code_label = "<START> " + logging_code_label + " <END>"

            reformatted_code_list.append(reformatted_code)
            logging_label_list.append(logging_code_label)

    return zip(reformatted_code_list, logging_label_list, embedding_list)


def get_retrived_prompts(ori_train_file, ori_test_file, dest_file):
    prompts, metadatas, embeddings = zip(*read_embedding_json(ori_test_file))
    train = read_embedding_json(file_path=ori_train_file)
    test = read_embedding_json(file_path=ori_test_file)
    retrieved_prompts = retrieve(test=test, train=train, batch_size=32, topK=5,
                                 instruct=r"select <line#> and insert log verbosity level (trace, debug, info, warn, error, fatal, all) and log message after <line#> \n",
                                 prompt_delimiter=' <prompt>: ', completion_delimiter=' <completion>: ')

    with open(dest_file, 'w') as f:
        for p, m, r in zip(prompts, metadatas, retrieved_prompts):

            line = {'prompt': r, 'completion': m, 'metadata': m}
            line = json.dumps(line)
            f.write(line+'\n')


if __name__ == '__main__':

    embedding_file_dir = "embeddings/output"
    embedding_train_file_name = "transformed_LANCE_train_all_embedding_input_samples.0.jsonl"
    embedding_test_file_name = "transformed_LANCE_test_all_embedding_input_samples.0.jsonl"
    dest_file_path = "final_prompt/final_prompt_all_test_all_train_instruct.jsonl"

    get_retrived_prompts(ori_train_file=os.path.join(embedding_file_dir, embedding_train_file_name),
                         ori_test_file=os.path.join(embedding_file_dir, embedding_test_file_name),
                         dest_file=dest_file_path)
