import os

if not os.path.exists("data/"):
    os.mkdir("data")

if not os.path.exists("data/java_code_train"):
    os.mkdir("data/java_code_train")

if not os.path.exists("data/java_code_test"):
    os.mkdir("data/java_code_test")


if not os.path.exists("embeddings/"):
    os.mkdir("embeddings/")

if not os.path.exists("embeddings/input/"):
    os.mkdir("embeddings/input/")

if not os.path.exists("embeddings/output/"):
    os.mkdir("embeddings/output/")

if not os.path.exists("final_prompt"):
    os.mkdir("final_prompt")

if not os.path.exists("reformatted_data"):
    os.mkdir("reformatted_data")

if not os.path.exists("reformatted_data/train"):
    os.mkdir("reformatted_data/train")

if not os.path.exists("reformatted_data/test"):
    os.mkdir("reformatted_data/test")

if not os.path.exists("results"):
    os.mkdir("results")
