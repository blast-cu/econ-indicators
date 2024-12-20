import json
import random
import re


def load_json(file_path):
    """
    Load a JSON file from the given file path.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def filter_frame_annotations(annotations, frame, econ_rate, econ_change=None):
    """
    Filter annotation dict by frame, econ_rate, and econ_change.
    """
    result = {}
    for ind, annotation in annotations.items():
        # print(annotation["frame"], annotation["econ_rate"], annotation["econ_change"])
        if (
            annotation["frame"] == frame
            and annotation["econ_rate"] == econ_rate
            # and annotation["econ_change"] == econ_change
        ):
            result[ind] = annotation

    return result


def filter_quant_annotations(annotations, q_type, spin, macro_type=None):
    """
    Filter annotation dict by q_type, spin, and macro_type.
    """
    result = {}
    for ind, annotation in annotations.items():
        if (
            annotation["type"] == q_type
            and annotation["spin"] == spin
            # and annotation["macro_type"] == macro_type
        ):
            result[ind] = annotation
    return result


def print_sample_annotations(annotations, n, key="text", alt_key=None):
    """
    Print n random annotations.
    """
    n = min(n, len(annotations))

    sample = random.sample(list(annotations.items()), n)
    for item in sample:
        print("\n")
        if alt_key:
            print(item[1][alt_key])
            print("\n")
        print(item[1][key])
        print("\n")


def main():
    """
    Main function.
    """
    file_path = "data/agreed_qual_dict.json"
    print(f"Loading annotations from {file_path}...")
    data = load_json(file_path)
    print(f"Loaded {len(data)} annotations.")
    # print(f"The first annotation is: {list(data.items())[0]}")

    frame = "macro"
    econ_rate = "none"

    filtered_data = filter_frame_annotations(data, frame, econ_rate)
    print_sample_annotations(filtered_data, 1)

    quant_file = "data/agreed_quant_dict.json"
    print(f"Loading quant annotations from {quant_file}...")
    quant_data = load_json(quant_file)
    print(f"Loaded {len(quant_data)} quant annotations.")

    q_type = "macro"
    spin = "neutral"

    filtered_quant_data = filter_quant_annotations(quant_data, q_type, spin)
    print_sample_annotations(
        filtered_quant_data, 10, key="excerpt", alt_key="indicator"
    )


if __name__ == "__main__":
    main()
