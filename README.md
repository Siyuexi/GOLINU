# Prompt4UniLogging

## Dataset
 - https://github.com/antonio-mastropaolo/LANCE 
 - https://drive.google.com/drive/folders/1D12y-CIJTYLxMeSmGQjxEXjTEzQImgaH 

## How to run

 - run dir_create.py to generate directories in workspace 
 - put train.tsv/test.csv into data dir
 - run code_preprocessing.py to reformat LANCE code 
    - format code using google-java-format
    - add line numbers
    - identify logging code lines
 - run embedding_generation.py to create input file for CodeX embedding generation
 - get embedding file via CodeX
 - run prompt_generation.py using train/test embedding files to generate final prompt with demonstration examples
 - feed final_prompt into CodeX and get results
 - run evaluate.py to post-process the results and conduct evaluation
 