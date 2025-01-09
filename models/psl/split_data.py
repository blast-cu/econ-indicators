import os
import shutil
from tqdm import tqdm

OUT_DIR = 'models/psl/data'
PREV_FINAL_DIR = 'models/psl/data_backup/final'
num_splits = 50


def main():

    # create new final subdirs, copy learn directory
    print("Copying learn directory...")
    for i in range(num_splits):
        try:
            os.makedirs(f"{OUT_DIR}/final{i}/", exist_ok=False)
            shutil.copytree(f"{PREV_FINAL_DIR}/learn/", f"{OUT_DIR}/final{i}/learn/")
        except FileExistsError:
            print(f"Directory {OUT_DIR}/final{i}/ already exists. Skipping.")
            continue

    # split eval files
    # has frame ann, split into 10
    print("Splitting eval files...")
    article_splits = {}  # article_id -> split_id
    with open(f"{PREV_FINAL_DIR}/eval/HasFrameAnn_obs.txt", 'r') as f:
        lines = f.readlines()
        num_lines = len(lines)
        split_size = num_lines // num_splits
        for i in range(num_splits):
            os.makedirs(f"{OUT_DIR}/final{i}/eval/", exist_ok=True)
            start_idx = i*split_size
            end_idx = (i+1)*split_size if i != num_splits-1 else num_lines
            with open(f"{OUT_DIR}/final{i}/eval/HasFrameAnn_obs.txt", 'w+') as out_f:
                for l in lines[start_idx:end_idx]:
                    out_f.write(l)
                    article_id = l.split('\t')[0]
                    article_splits[article_id] = i

    # loop over all eval files (except HasFrameAnn_obs.txt)
    for filename in tqdm(os.listdir(f"{PREV_FINAL_DIR}/eval/")):
        if filename == 'HasFrameAnn_obs.txt':
            continue
        with open(f"{PREV_FINAL_DIR}/eval/{filename}", 'r') as f:
            lines = f.readlines()

        split_num = 0
        out_f = open(f"{OUT_DIR}/final{split_num}/eval/{filename}", 'w+')
        for l in lines:
            # check if article_id is in this split
            article_id = l.split('\t')[0]
            if "_" in article_id:
                article_id = article_id.split("_")[0]
            if article_splits[article_id] != split_num:  # need to switch to new split
                split_num = article_splits[article_id]
                out_f.close()  # close current file
                os.makedirs(f"{OUT_DIR}/final{split_num}/eval/", exist_ok=True)
                out_f = open(f"{OUT_DIR}/final{split_num}/eval/{filename}", 'w+')  # open new file

            out_f.write(l)  # write line to file
        out_f.close()


if __name__ == "__main__":
    main()