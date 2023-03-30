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
import os
import copy
from tqdm import tqdm
import multiprocessing

random.seed(123)

MULTI_PROCESS_STEP = 10000


def reformat_Java_code_by_Google_formatting(code, idx, temp_code_dir):

    java_code = "public class A {" + code + "}"  # construct Java class for google formatting tool
    java_code_file_path = os.path.join(temp_code_dir, "example" + str(idx) + ".java")
    with open(java_code_file_path, 'w', encoding='utf-8') as java_code_file:
        java_code_file.write(java_code)

    results = os.popen('java -jar google-java-format-1.15.0-all-deps.jar ' + java_code_file_path)
    context = results.read()
    if context == "":
        print(f"example-{idx} reformatting failed")
        return None
    else:
        lines = context.splitlines()
        results.close()
        return lines


def reformat_LANCE_java_code(df, temp_code_dir, output_file_path, random_sample_num=None, remove_diff_logging_only=True):

    print("reformatting started")

    if random_sample_num is not None:
        if isinstance(random_sample_num, int) and random_sample_num <= len(df):
            df = df.sample(frac=1)  # shuffle the data
        else:
            print("Invalid random sampling num")
    else:
        print("Disable sampling")

    processed_data = []

    correct_reformat_cnt = 0

    for idx, row in tqdm(df.iterrows()):

        input_code, output_code = df.loc[idx, 0], df.loc[idx, 1]

        if remove_diff_logging_only:  # only remove diff logging code for testing set

            input_java_code_lines = reformat_Java_code_by_Google_formatting(input_code, idx, temp_code_dir)
            output_java_code_lines = reformat_Java_code_by_Google_formatting(output_code, idx, temp_code_dir)

            if not (input_java_code_lines and output_java_code_lines):
                continue

            # identify all logging code for input and output snippet
            input_logging_code_labels = identify_logging_statement_code(input_java_code_lines)
            output_logging_code_labels = identify_logging_statement_code(output_java_code_lines)

            code_diff = set(output_logging_code_labels.keys()) - set(input_logging_code_labels.keys())
            if len(code_diff) != 1:
                print("no java_code_valid diff logging code")
                continue
            logging_code_label = list(code_diff)[0]
            logging_code_span = output_logging_code_labels[logging_code_label]
            logging_code_label = "<line" + str(logging_code_span[0]-1) + ">" + logging_code_label
            clean_output_java_code_lines = [x for n, x in enumerate(output_java_code_lines) if n not in range(logging_code_span[0], logging_code_span[1])]

            processed_java_code = ""
            for i, line in enumerate(clean_output_java_code_lines):
                line = line.strip()
                processed_java_code += line + " <line" + str(i) + ">" + " "

            processed_data.append([idx, input_code, processed_java_code, logging_code_label])

        else: # remove all identified logging code for output code

            output_java_code_lines = reformat_Java_code_by_Google_formatting(output_code, idx, temp_code_dir)

            if not output_java_code_lines:
                continue

            # only need to identify log code of output code
            output_logging_code_labels = identify_logging_statement_code(output_java_code_lines)

            if len(output_logging_code_labels) == 0:
                print("logging code not found")
                continue

            remove_line_id = []
            all_logging_code_labels = ""
            output_logging_code_labels = sorted(output_logging_code_labels.items(), key=lambda d: d[1][0])
            for logging_code_label, logging_code_span in output_logging_code_labels:
                # recalculate the line number because all logging code will be removed
                all_logging_code_labels += "<line" + str(logging_code_span[0] - (len(remove_line_id) + 1)) + ">" + logging_code_label
                remove_line_id.extend(list(range(logging_code_span[0], logging_code_span[1])))

            clean_output_java_code_lines = [x for n, x in enumerate(output_java_code_lines) if n not in remove_line_id]

            processed_java_code = ""
            for i, line in enumerate(clean_output_java_code_lines):
                line = line.strip()
                processed_java_code += line + " <line" + str(i) + ">" + " "

            processed_data.append([idx, input_code, processed_java_code, all_logging_code_labels])

        correct_reformat_cnt += 1

        if random_sample_num is not None and correct_reformat_cnt >= random_sample_num:
            break

    processed_df = pd.DataFrame(processed_data, columns=["id", "input_code",
                                                         "reformatted_input_code",
                                                         "logging_code_labels"])
    processed_df.to_csv(output_file_path, sep="\t", index=False)

    return


def identify_logging_statement_code_line(target_line, target_idx, lines):

    logging_code_regex = r'[.](off)?(fatal)?(error)?(warn)?(info)?(debug)?(trace)?(all)?[(]'

    logging_code_label = ""
    span = ()

    logging_code_match_res = re.search(logging_code_regex, target_line)

    if logging_code_match_res:

        # logging_code_label = "<line" + str(target_idx - 1) + "> " + target_line
        logging_code_label = target_line

        if target_line.endswith("("):  # consider cross lines logging statement code
            for j in range(target_idx + 1, len(lines)):
                logging_code_label += lines[j].strip()
                if lines[j].endswith(");"): span = (target_idx, j+1); break
        else:
            span = (target_idx, target_idx + 1)

    return logging_code_label, span


def identify_logging_statement_code(code_lines):

    '''
    :param code_lines: reformatted code lines
    :return: dict[logging_code]=logging_code_span
    '''

    logging_code_labels = {}

    for i, line in enumerate(code_lines):
        logging_code_label, logging_code_span = identify_logging_statement_code_line(line, i, code_lines)
        if logging_code_label:
            logging_code_labels[logging_code_label] = logging_code_span

    return logging_code_labels


def multi_process_reformat_LANCE_java_code(dataset_dir, dataset_name, temp_code_dir, sub_results_dir):

    df = pd.read_csv(os.path.join(dataset_dir, dataset_name), sep='\t', header=None)  # no header in raw LANCE dataset
    data_num = len(df)
    print("data num:", data_num)

    pool = multiprocessing.Pool(10)

    for start_idx in range(0, data_num, MULTI_PROCESS_STEP):
        sub_output_file_path = os.path.join(sub_results_dir, str(start_idx) + "_reformatted_" + dataset_name)

        end_idx = start_idx + MULTI_PROCESS_STEP
        sub_df = df.loc[start_idx:, :] if end_idx >= data_num else df.loc[start_idx:end_idx, :]
        sub_df_input = copy.deepcopy(sub_df)

        pool.apply_async(reformat_LANCE_java_code, args=(sub_df_input, temp_code_dir, sub_output_file_path, 10))

    pool.close()
    pool.join()

    res_df = []
    for sub_reformatted_data_name in os.listdir(sub_results_dir):
        sub_df = pd.read_csv(os.path.join(sub_results_dir, sub_reformatted_data_name), sep='\t')
        res_df.append(sub_df)
    res_df = pd.concat(res_df, axis=0)
    res_df.to_csv("reformatted_data/reformatted_train.tsv", sep="\t", index=False)


def extract_logging_snippet_LANCE_java_code(df, temp_code_dir, output_file_path,
                                            select_sample_ids=None,
                                            context_num=3):

    '''
    extract the context surrounding the logging code
    :param context_num: select context lines number above and below to the target logging code
    :return: extract data frame
    '''

    print("extract logging snippet started")

    logging_code_snippets_data = []

    if select_sample_ids is not None:
        df = df.loc[select_sample_ids, :]

    for idx, row in tqdm(df.iterrows()):

        input_code, output_code = df.loc[idx, 0], df.loc[idx, 1]

        output_java_code_lines = reformat_Java_code_by_Google_formatting(output_code, idx, temp_code_dir)

        if not output_java_code_lines:
            continue

        # only need to identify log code of output code
        output_logging_code_labels = identify_logging_statement_code(output_java_code_lines)

        if len(output_logging_code_labels) == 0:
            print("logging code not found")
            continue

        remove_line_ids = []
        logging_line_ids = []
        output_logging_code_labels = sorted(output_logging_code_labels.items(), key=lambda d: d[1][0])
        for logging_code_label, logging_code_span in output_logging_code_labels:
            # recalculate the line number because all logging code will be removed
            logging_line_id = logging_code_span[0] - (len(remove_line_ids) + 1)
            remove_line_ids.extend(list(range(logging_code_span[0], logging_code_span[1])))
            logging_line_ids.append(logging_line_id)

        clean_output_java_code_lines = [x for n, x in enumerate(output_java_code_lines) if n not in remove_line_ids]

        for k, logging_line_id in enumerate(logging_line_ids):
            logging_code_snippet_with_lineid = ""
            raw_logging_code_snippet = ""
            for i, line in enumerate(clean_output_java_code_lines):
                if (i >= max(logging_line_id - context_num, 0)) \
                        and (i <= min(logging_line_id + context_num, len(clean_output_java_code_lines))):
                    line = line.strip()
                    logging_code_snippet_with_lineid += line + " <line" + str(i) + ">" + " "
                    raw_logging_code_snippet += line + " "

            logging_code_snippets_data.append([idx, raw_logging_code_snippet, logging_code_snippet_with_lineid,
                                               "<line" + str(logging_line_id) + ">" + output_logging_code_labels[k][0]])

    processed_df = pd.DataFrame(logging_code_snippets_data, columns=["id", "input_code",
                                                         "reformatted_input_code",
                                                         "logging_code_labels"])
    processed_df.to_csv(output_file_path, sep="\t", index=False)

    return


if __name__ == '__main__':

    # prepare testing set
    dataset_dir = "data/"
    results_dir = "reformatted_data/"
    dataset_name = "test.tsv"
    reformatted_data_name = "reformatted_test.tsv"
    temp_code_dir = os.path.join(dataset_dir, "java_code_test/")  # for Java code file dump
    output_file_path = os.path.join(results_dir, "reformatted_" + dataset_name)
    
    df = pd.read_csv(os.path.join(dataset_dir, dataset_name), sep='\t', header=None)  # no header in raw LANCE dataset
    
    reformat_LANCE_java_code(df, temp_code_dir, output_file_path, random_sample_num=None, remove_diff_logging_only=True)


    # prepare training set
    dataset_dir = "data/"
    results_dir = "reformatted_data/"
    dataset_name = "train.tsv"
    reformatted_data_name = "reformatted_train.tsv"
    temp_code_dir = os.path.join(dataset_dir, "java_code_train/")  # for Java code file dump
    output_file_path = os.path.join(results_dir, "reformatted_" + dataset_name)

    df = pd.read_csv(os.path.join(dataset_dir, dataset_name), sep='\t', header=None)  # no header in raw LANCE dataset
    
    reformat_LANCE_java_code(df, temp_code_dir, output_file_path, random_sample_num=None, remove_diff_logging_only=False)
    














