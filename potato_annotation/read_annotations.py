import json
import os
from data_utils.dataset import qual_label_maps


def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    # print('Loaded {} records from {}'.format(len(data), input_path))
    return data

def get_potato_article_anns(): 
    ann_dict = {
        "frame": [],
        "econ_rate": [],
        "econ_change": []
    } # article_id, user_id, ann

    ann_output_dir = "potato_annotation/article_annotate/annotation_output/pilot/"
    dir_list = os.listdir(ann_output_dir)

    for annotator_id in dir_list:
        d = os.path.join(ann_output_dir, annotator_id)
        if os.path.isdir(d):
            f = os.path.join(d, "annotated_instances.jsonl")
            data = load_jsonl(f)
            for ann in data:
                article_id = ann["id"]
                ann = ann["label_annotations"]
                if list(ann["frame-macro"].keys())[0] == "Yes":
                    frame_val = "macro"
                else: 
                    frame_val = list(ann["frame"].keys())[0]
                ann_dict["frame"].append((article_id, annotator_id, frame_val.lower()))

                econ_rate_val = list(ann["Economic Conditions"].values())[0]
                econ_change_val = list(ann["Economic Conditions"].keys())[0]
                ann_dict["econ_rate"].append((article_id, annotator_id, econ_rate_val.lower()))
                ann_dict["econ_change"].append((article_id, annotator_id, econ_change_val.lower()))
    
    return ann_dict

def main():
    ann_dict = get_potato_article_anns()
    for k, v in ann_dict.items():
        print(k, v[:5])



if __name__ == "__main__":
    main()
