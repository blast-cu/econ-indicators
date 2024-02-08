
import pickle

def main():
    split_dir = "data/clean/"
    quant_dict = pickle.load(open(split_dir + "quant_dict", "rb"))

    for id, ann in quant_dict.items():
        if ann['type'] == 'macro':
            if ann['macro_type'] == 'none':
                print("type is macro and macro_type is none")
                print(ann['text'])
        else:
            if ann['type'] != '\x00':
                if ann['macro_type'] != 'none':
                    print("type is not macro and macro_type is not none")
                    print(ann)


if __name__ == "__main__":
    main()