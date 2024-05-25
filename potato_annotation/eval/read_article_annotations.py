import json
import os
from data_utils.model_utils.dataset import qual_label_maps

qual_predict_maps = {
    'frame': {
            1: 'business',
            2: 'industry',
            0: 'macro',
            3: 'government',
            5: 'other',
            4: 'personal',
            6: 'NA'},
    'econ_rate': {
            0: 'good',
            1: 'poor',
            2: 'none',
            3: 'not macro',
            4: 'NA'},
    'econ_change': {
            0: 'better',
            1: 'worse',
            2: 'same',
            3: 'none',
            4: 'irrelevant',
            5: 'NA'}
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

def get_potato_article_anns(ann_output_dir, report_errors=False): 
    ann_dict = {
        "frame": [],
        "econ_rate": [],
        "econ_change": []
    }
    dir_list = os.listdir(ann_output_dir)

    annotator_stats = {}
    annotator_stats["user_id"] = []
    annotator_stats["total_anns"] = []
    annotator_stats["errors"] = []

    count = 0
    for annotator_id in dir_list:
        d = os.path.join(ann_output_dir, annotator_id)
        if os.path.isdir(d) and annotator_id != "reports":
            # count += 1
            annotator_stats["user_id"].append(annotator_id)
    
            ann_count = 0
            error_count = 0
            f = os.path.join(d, "annotated_instances.jsonl")
            data = load_jsonl(f)
            for ann in data: 

                article_id = ann["id"]
                ann = ann["label_annotations"]

                if 'frame' in ann:  # skip examples
                    ann_count += 1

                    article_id = int(article_id)

                    label_id = int(list(ann["frame"].values())[0])
                    frame_val = qual_predict_maps["frame"][label_id]

                    label_id = int(list(ann["Economic Conditions"].values())[0])
                    econ_rate_val = qual_predict_maps["econ_rate"][label_id]

                    label_id = int(list(ann["Economic Direction"].values())[0])
                    econ_change_val = qual_predict_maps["econ_change"][label_id]


                    if frame_val == "macro":
                        if econ_rate_val == "NA":
                            if report_errors:
                                print("Chose macro but econ_rate is NA: {}".format(annotator_id))
                            error_count += 1
                        if econ_change_val == "NA":
                            if report_errors:
                                print("Chose macro but econ_change is NA: {}".format(annotator_id))
                            error_count += 1
                        if econ_rate_val == "not macro":
                            if report_errors:
                                print("Chose macro but econ_rate is 'not macro': {}".format(annotator_id))
                            error_count += 1
                        if econ_change_val == "not macro":
                            if report_errors:
                                print("Chose macro but econ_change is 'not macro': {}".format(annotator_id))
                            error_count += 1

                    elif frame_val == "NA":

                        if econ_rate_val != "NA":
                            if report_errors:
                                print("Chose NA but econ_rate is not NA: {}".format(annotator_id))
                            error_count += 1
                            econ_rate_val = "NA"
                
                        if econ_change_val != "NA":
                            if report_errors:
                                print("Chose NA but econ_change is not NA: {}".format(annotator_id))
                            error_count += 1
                            econ_change_val = "NA"

                    else:
                        if econ_change_val != "not macro":
                            if report_errors:
                                print("Chose not macro but econ_change is not 'not macro': {}".format(annotator_id))
                            error_count += 1
                            econ_change_val = "not macro"
                        if econ_rate_val != "not macro":
                            if report_errors:
                                print("Chose not macro but econ_rate is not 'not macro': {}".format(annotator_id))
                            error_count += 1
                            econ_rate_val = "not macro"

                        econ_rate_val = 'NA'
                        econ_change_val = 'NA'
 

                    ann_dict["frame"].append((article_id, annotator_id, frame_val))
                    ann_dict["econ_rate"].append((article_id, annotator_id, econ_rate_val))
                    ann_dict["econ_change"].append((article_id, annotator_id, econ_change_val))
                    
        
            annotator_stats["total_anns"].append(ann_count)
            annotator_stats["errors"].append(error_count)


                # if frame_val != "macro":
                #     if econ_change_val != "NA":
                #         print("Error: {}".format(annotator_id))
                #         # print(article_id, annotator_id, frame_val, econ_rate_val, econ_change_val)
                #         print(frame_val)
                #         print(econ_change_val)
                #         print()

    return ann_dict, annotator_stats

def main():
    ann_dict, annotator_stats = get_potato_article_anns("potato_annotation/article_annotate/annotation_output/pilot_4_17")
    for k, v in annotator_stats.items():
        print(k, v)



if __name__ == "__main__":
    main()
