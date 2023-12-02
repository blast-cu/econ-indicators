import argparse
import os

qual_map = {
    'article_type': {
        'business': 0,
        'industry': 1,
        'macro': 2,
        'government': 3,
        'other': 4
    },
    'econ_rating': {
        'good': 0,
        'poor': 1,
        'none': 2
    },
    'econ_change': {
        'better': 0,
        'worse': 1,
        'same': 2,
        'none': 3
    },
}


quant_map = {
    'excerpt_type': {
        'business': 0,
        'industry': 1,
        'macro': 2,
        'government': 3,
        'other': 4
    },
    'spin': {
        'positive': 0,
        'negative': 1,
        'neutral': 2
    }
}

def generate_predicates(map, preamble):
    predicates = {}
    for ann in map.keys():
        predicate = preamble
        for split in ann.split('_'):
            predicate += split[0].upper() + split[1:]
        predicates[ann] = predicate
    return predicates


def write_map(out_dir, map):

    for ann in map.keys():
        with open(os.path.join(out_dir, ann + '_map.txt'), 'w') as f:
            curr_map = map[ann]
            for label, value in curr_map.items():
                f.write(str(value) + "\t" + label + '\n')
    return

def write_rules(out_dir):

    predicted_preds = {}
    predicted_preds['qual'] = generate_predicates(qual_map, 'predicted')
    predicted_preds['quant'] = generate_predicates(quant_map, 'predicted')

    val_preds = {}
    val_preds['qual'] = generate_predicates(qual_map, 'val')
    val_preds['quant'] = generate_predicates(quant_map, 'val')


    # generate rules connecting predictions and values
    # TODO: determine rule weights
    with open(os.path.join(out_dir, 'rules.txt'), 'w') as f:
        for ann in predicted_preds['qual'].keys(): # article level annotations
            f.write('1.0: ' + predicted_preds['qual'][ann] + '(A1) -> ' + val_preds['qual'][ann] + '(A1)\n')

        for ann in predicted_preds['quant'].keys(): # excerpt level annotations
            f.write('1.0: ' + predicted_preds['quant'][ann] + '(E1) -> ' + val_preds['quant'][ann] + '(E1)\n')

        

        



def main(args):

    out_dir = 'models/psl/data'
    os.makedirs(out_dir, exist_ok=True)

    write_map(out_dir, qual_map)
    write_map(out_dir, quant_map)

    write_rules(out_dir=out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=False)
    args = parser.parse_args()
    main(args)
