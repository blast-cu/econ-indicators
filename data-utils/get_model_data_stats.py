from inter_annotator_agreement import retrieve_anns, retrieve_quant_anns
import argparse
import sqlite3
from collections import Counter
import csv


def get_agreed_anns(ann_dict: dict):
    """takes a nested dictionary of annotations (list) and returns
    nested dictionary of final annotations (str) wrt full agreement """

    for id in ann_dict.keys(): 
        curr_ent = ann_dict[id]
        # TODO: for quant anns, check type before subtypes
        for type in curr_ent.keys(): 
            curr_t = curr_ent[type]
            result = '\0'

            if len(curr_t) > 0:
                anns = [a[1] for a in curr_t]
                c = Counter(anns).most_common()
                if len(c) == 1 or c[0][1] != c[1][1]:  # check for tie
                    result = c[0][0]
            ann_dict[id][type] = result

    return ann_dict


def print_agreed_anns_counts(ann_dict: dict):
    """takes dictionary output of get_agreed_anns func and prints
    agreed annotation stats"""

    labels = {}

    for id in ann_dict.keys():
        curr_ent = ann_dict[id]

        for type in curr_ent.keys():
            curr_t = curr_ent[type]
            if type not in labels:
                labels[type] = {}

            if curr_t != '\0': 
                if curr_t in labels[type]:
                    labels[type][curr_t] += 1
                else:
                    labels[type][curr_t] = 1

    for type in labels.keys():
        count = sum(labels[type].values())
        print(type + " - count: " + str(count))
        print(str(labels[type]) + '\n')
    
    return labels


def export_quants_to_csv(ann: dict, filename: str):
    """Takes as input the labels dictionary output by print_agreed_anns_counts
    and formats results into csv"""

    field_names = ['Type', 'Subtype', 'Nested Subtype', 'Annotations']
    row_list = []
    filename = "data_summary/" + filename

    def add_row(type, subtype, n_subtype, ann_ct):
        row = {}
        row['Type'] = type
        row['Subtype'] = subtype
        row['Nested Subtype'] = n_subtype
        row['Annotations'] = ann_ct
        row_list.append(row)

    macro_dict = ann['macro_type']
    for mt in macro_dict.keys():
        add_row('Macro', mt, 'NA', str(macro_dict[mt]))

    add_row('Macro', 'Total', 'NA', str(ann['type']['macro']))

    expen_dict = ann['expenditure_type']
    for et in expen_dict.keys():
        add_row('Government', 'Expenditures', et, str(expen_dict[et]))

    rev_dict = ann['revenue_type']
    for rt in rev_dict.keys():
        add_row('Government', 'Revenues', rt, str(rev_dict[rt]))

    add_row('Government', 'Debt and Deficit', 'NA', str(ann['gov_type']['deficit']))
    add_row('Government', 'Total', rt, ann['type']['government'])

    ind_dict = ann['industry_type']
    for it in ind_dict.keys():
        add_row('Industry', it, 'NA', str(ind_dict[it]))

    add_row('Industry', 'Total', 'NA', str(ann['type']['industry']))
    add_row('Business', 'NA', 'NA', str(ann['type']['business']))
    add_row('Personal', 'NA', 'NA', str(ann['type']['personal']))
    add_row('Other', 'NA', 'NA', str(ann['type']['other']))
    add_row('Total', 'NA', 'NA', str(sum(ann['type'].values())))

    with open(filename, 'w+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(row_list)


def main(args):

    con = sqlite3.connect(args.db)
    cur = con.cursor()
    ann = {}
    quant_ann = {}

    # TODO: retrieve_anns should be refactored to reduce db queries and
    # calls to retrieve_anns

    # Frame
    query = 'SELECT article_id, user_id, frame \
                FROM articleann \
                WHERE frame is NOT NULL and frame != "None";'
    res = cur.execute(query)
    retrieve_anns(ann, res, 'frame')

    # Econ rate
    query = 'SELECT article_id, user_id, econ_rate \
                FROM articleann \
                WHERE econ_rate is NOT NULL and econ_rate != "None";'
    res = cur.execute(query)
    retrieve_anns(ann, res, 'econ_rate')

    # Econ change
    query = 'SELECT article_id, user_id, econ_change \
                FROM articleann \
                WHERE econ_change is NOT NULL and econ_change != "None";'
    res = cur.execute(query)
    retrieve_anns(ann, res, 'econ_change')

    ann = get_agreed_anns(ann)
    print_agreed_anns_counts(ann)
   

    # Quantities
    query = 'SELECT quantity_id, user_id, type, macro_type, industry_type, \
                gov_type, expenditure_type, revenue_type, spin \
                FROM quantityann;'
    res = cur.execute(query)
    retrieve_quant_anns(quant_ann, res)

    quant_ann = get_agreed_anns(quant_ann)
    labels = print_agreed_anns_counts(quant_ann)
    export_quants_to_csv(labels, 'annotation_count.csv')

    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True)
    args = parser.parse_args()
    main(args)