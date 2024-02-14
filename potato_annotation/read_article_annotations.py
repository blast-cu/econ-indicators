import json
import os
from data_utils.dataset import qual_label_maps

qual_predict_maps = {
    'frame': {
            1: 'business',
            2: 'industry',
            0: 'macro',
            3: 'government',
            4: 'other',
            5: 'personal'},
    'econ_rate': {
            0: 'good',
            1: 'poor',
            2: 'none',
            3: 'NA'},
    'econ_change': {
            0: 'better',
            1: 'worse',
            2: 'same',
            3: 'none',
            4: 'NA'}
}
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

    count = 0
    for annotator_id in dir_list:
        d = os.path.join(ann_output_dir, annotator_id)
        if os.path.isdir(d):
            # count += 1
            f = os.path.join(d, "annotated_instances.jsonl")
            data = load_jsonl(f)
            for ann in data:
                article_id = int(ann["id"])
                ann = ann["label_annotations"]
                if list(ann["relevant"].keys())[0] == "Yes":
                    frame_val = int(list(ann["frame"].values())[0])

                    if frame_val == 0:
                        frame_val = "macro"

                        label_id = int(list(ann["Economic Conditions"].values())[0])
                        econ_rate_val = qual_predict_maps["econ_rate"][label_id]

                        label_id = int(list(ann["Economic Conditions"].values())[0])
                        econ_change_val = qual_predict_maps["econ_change"][label_id]

                    else:
                        label_id = int(list(ann["frame"].values())[0])
                        frame_val = qual_predict_maps["frame"][label_id]
                        econ_change_val = "NA"
                        econ_rate_val = "NA"

                    ann_dict["frame"].append((article_id, annotator_id, frame_val))
                    ann_dict["econ_rate"].append((article_id, annotator_id, econ_rate_val))
                    ann_dict["econ_change"].append((article_id, annotator_id, econ_change_val))
        



                # if frame_val != "macro":
                #     if econ_change_val != "NA":
                #         print("Error: {}".format(annotator_id))
                #         # print(article_id, annotator_id, frame_val, econ_rate_val, econ_change_val)
                #         print(frame_val)
                #         print(econ_change_val)
                #         print()
    
    return ann_dict

def main():
    ann_dict = get_potato_article_anns()
    for k, v in ann_dict.items():
        print(k, v[:5])



if __name__ == "__main__":
    main()
