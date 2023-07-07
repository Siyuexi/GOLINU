# README

## Dataset
 - Original dataset: https://github.com/antonio-mastropaolo/LANCE 
 - Filtered dataset: https://drive.google.com/drive/folders/1D12y-CIJTYLxMeSmGQjxEXjTEzQImgaH

The first linked dataset is the original dataset collected by LANCE, which includes a significant amount of data used exclusively for pretraining T5-small and does not contain any log statements. 

The second linked dataset is the dataset we re-organized from LANCE dataset after filtering code snippets without any logging statement. It consists of three subsets: `train.tsv`, `eval.tsv`, and `test.tsv`, with 101,405, 12,470, and 12,010 code snippets respectively. The first column of each `.tsv` file contains code snippets with logging statements removed, while the second column contains the complete logging statements.

## How to run

 - run dir_create.py to generate directories in workspace 
 - put train.tsv/test.tsv into data dir
 - run code_preprocessing.py to reformat LANCE code 
    - format code using google-java-format
    - add line numbers
    - identify logging code lines
 - run embedding_generation.py to create input file for CodeX embedding generation
 - get embedding file via CodeX
 - run prompt_generation.py using train/test embedding files to generate final prompt with demonstration examples
 - feed final_prompt into CodeX and get results
 - run evaluate.py to post-process the results and conduct evaluation
 
