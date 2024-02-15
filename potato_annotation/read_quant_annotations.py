import json
import os
from data_utils.dataset import qual_label_maps

quant_predict_maps = {
    'type': {
            0: 'macro',
            2: 'industry',
            3: 'government',
            4: 'personal',
            1: 'business',
            5: 'other'},
    'macro_type': {
            0: 'jobs',
            7: 'retail',
            8: 'interest',
            5: 'prices',
            10: 'energy',
            4: 'wages',
            3: 'macro',
            1: 'market',
            9: 'currency',
            2: 'housing',
            6: 'confidence', # ???
            11: 'other',
            12: 'none'},
    'spin': {
            0: 'pos',
            1: 'neg',
            2: 'neutral',
            3: 'irrelevant'}
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

def get_potato_quant_anns(): 
    ann_list = [] # quant_id, user_id, type, macro_type, industry_type, gov_type, expenditure_type, revenue_type, spin

    ann_output_dir = "potato_annotation/quant_annotate/annotation_output/pilot"
    dir_list = os.listdir(ann_output_dir)

    count = 0
    for annotator_id in dir_list:
        # print(annotator_id)
        d = os.path.join(ann_output_dir, annotator_id)
        if os.path.isdir(d):
            count += 1
            f = os.path.join(d, "annotated_instances.jsonl")
            data = load_jsonl(f)
            for ann in data:
                quant_id = ann["id"]
                
                print(ann["displayed_text"])
                ann = ann["label_annotations"]
                print(ann["frame"])
                print(ann["macro_indicator"])
                print(ann["spin"])
                print()

                frame_val = int(list(ann["frame"].values())[0])

                if frame_val != 6:  # not relevant

                    if frame_val == 0: # macro
                        frame = "macro"
                        label_id = int(list(ann["macro_indicator"].values())[0])
                        macro_ind_val = quant_predict_maps["macro_type"][label_id]

                        label_id = int(list(ann["spin"].values())[0])
                        if label_id < 3:
                            spin_val = quant_predict_maps["spin"][label_id]
                        else:
                            spin_val = "none"
                            

                    else:
                        label_id = int(list(ann["frame"].values())[0])
                        frame_val = quant_predict_maps["type"][label_id]
                        macro_ind_val = "none"
                        spin_val = "none"

                    ann_dict = {}
                    ann_dict["quant_id"] = quant_id
                    ann_dict["user_id"] = annotator_id
                    ann_dict["type"] = frame_val
                    ann_dict["macro_type"] = macro_ind_val
                    ann_dict["spin"] = spin_val
                    ann_list.append(ann_dict)
                    
        



                # if frame_val != "macro":
                #     if econ_change_val != "NA":
                #         print("Error: {}".format(annotator_id))
                #         # print(article_id, annotator_id, frame_val, econ_rate_val, econ_change_val)
                #         print(frame_val)
                #         print(econ_change_val)
                #         print()
    
    return ann_list

def main():
    ann_list = get_potato_quant_anns()
    for v in ann_list:
        print(v)



if __name__ == "__main__":
    main()
