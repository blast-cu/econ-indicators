import pickle
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

qual_dict = pickle.load(open('data/clean/qual_dict', 'rb'))
quant_dict = pickle.load(open('data/clean/quant_dict', 'rb'))

def get_data(x_axis, y_axis, x_ann, y_ann):

    vals = {}

    data = np.zeros((len(y_axis), len(x_axis)))
    for x_val in x_axis:
        print()
        print(x_val)
        for id in qual_dict.keys():
            if qual_dict[id][x_ann] == x_val:
                for y_val in y_axis:
                    quant_list = qual_dict[id]['quant_list']
                    for global_id in quant_list:
                        print(quant_dict[global_id][y_ann])
                        if quant_dict[global_id][y_ann] == y_val:
                            data[y_axis.index(y_val)][x_axis.index(x_val)] += 1
                            val = str(x_val) + '_' + str(y_val)

                            if val not in list(vals.keys()):
                                vals[val] = 0
                            vals[val] += 1
    for v in vals.keys():
        print(v, vals[v])

    return data


def main():
    # excerpts_dict = pickle.load(open('data/clean/quant_excerpts_dict', 'rb'))
    # print(len(excerpts_dict.keys()))
    # conn = sqlite3.connect('data/data.db')
    # publisher_count = {}
    # for global_id in list(excerpts_dict.keys()):
    #     article_id = global_id.split('_')[0]
        
    #     c = conn.cursor()
    #     c.execute(f'SELECT source FROM article WHERE id = {article_id}')

    #     source = c.fetchone()[0]
    #     if source not in publisher_count.keys():
    #         publisher_count[source] = 0
    #     publisher_count[source] += 1

    # conn.close()
    # print(list(publisher_count.keys()))

    # publishers = ['washingtonpost', 'huffpost', 'nytimes', 'wsj', 'foxnews', 'breitbart']

    # annotation_count = {}

    # conn = sqlite3.connect('data/data.db')
    # c = conn.cursor()
    # c.execute(f'SELECT quantity_id FROM quantityann')
    # # print(c.fetchall())
    # annotations = c.fetchall()
    # annotations = [a[0] for a in annotations]

    # ann_count = {}
    # for a in annotations:
    #     if a not in ann_count.keys():
    #         ann_count[a] = 0
    #     ann_count[a] += 1
    
    # cros_validated = [a for a in ann_count.keys() if ann_count[a] > 1]
    # print(len(cros_validated))


    # for id in annotations:
    #     c = conn.cursor()
    #     article_id = id.split('_')[0]
    #     c.execute(f'SELECT source FROM article WHERE id = {article_id}')
    #     source = c.fetchone()[0]
    #     if source != 'bbc':
    #         if source not in annotation_count.keys():
    #             annotation_count[source] = 0
    #         annotation_count[source] += 1
    # conn.close()
    
    # total = 0
    # for p in publishers:
    #     print(p, annotation_count[p])
    #     total += annotation_count[p]
    # print(total)

    
    # for p in publishers:

    #     annotation_count[p] = {}

    #     c = conn.cursor()
    #     c.execute(f'SELECT id FROM articleann WHERE source = "{p}"')
    #     for ann in c.fetchall():
    #         ann_id = ann[0]
    #         if ann_id not in annotation_count[p].keys():
    #             annotation_count[p][ann_id] = 0
    #         annotation_count[p][ann_id] += 1
    



    
    
    x_axis = ['none', 'same', 'worse', 'better'] # econ_change
    y_axis = [ 'pos', 'neg', 'neutral']  # spin
    data = get_data(x_axis, y_axis, 'econ_change', 'spin')

    

    fig, ax = plt.subplots()
    plt.imshow(data, cmap='Blues')
    ax.set_xticks(np.arange(len(x_axis)), labels= x_axis)
    ax.set_yticks(np.arange(len(y_axis)), labels= y_axis)
    ax.set_xlabel('Economic Direction')
    ax.set_ylabel('Indicator Polarity')

    for i in range(len(y_axis)):
        for j in range(len(x_axis)):
            text = ax.text(j, i, data[i, j], ha = "center", va = "center")
    plt.savefig('econ_change_spin.png', dpi=300)
    # plt.show()

    y_axis = ['macro', 'industry', 'government', 'personal', 'business', 'other']
    data = get_data(x_axis, y_axis, 'econ_change', 'type')

    fig, ax = plt.subplots()
    plt.imshow(data, cmap='Blues')
    ax.set_xticks(np.arange(len(x_axis)), labels= x_axis)
    ax.set_yticks(np.arange(len(y_axis)), labels= y_axis)
    ax.set_xlabel('Economic Direction')
    ax.set_ylabel('Quantity Type')

    for i in range(len(y_axis)):
        for j in range(len(x_axis)):
            text = ax.text(j, i, data[i, j], ha = "center", va = "center")
    plt.savefig('econ_change_type.png', dpi=300)

    y_axis = ['jobs', 'retail', 'interest', 'prices', 'energy', 'wages', 'macro', 'market', 'currency', 'housing', 'other', 'none']
    data = get_data(x_axis, y_axis, 'econ_change', 'macro_type')

    fig, ax = plt.subplots()
    plt.imshow(data, cmap='Blues')
    ax.set_xticks(np.arange(len(x_axis)), labels= x_axis,rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(y_axis)), labels= y_axis)
    ax.set_xlabel('Economic Direction')
    ax.set_ylabel('Macro Indicator')

    for i in range(len(y_axis)):
        for j in range(len(x_axis)):
            text = ax.text(j, i, data[i, j], ha = "center", va = "center")
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('econ_change_macro_type.png', dpi=300)

    x_axis = ['business', 'industry', 'macro', 'government', 'other']
    y_axis = [ 'pos', 'neg', 'neutral']  # spin
    data = get_data(x_axis, y_axis, 'frame', 'spin')

    fig, ax = plt.subplots()
    plt.imshow(data, cmap='Blues')
    # ax.set_xticks(np.arange(len(x_axis)), labels= x_axis,rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(len(x_axis)), labels= x_axis)
    ax.set_yticks(np.arange(len(y_axis)), labels= y_axis)
    # ax.set_xlabel('Article Type')
    # ax.set_ylabel('Indicator Polarity')

    for i in range(len(y_axis)):
        for j in range(len(x_axis)):
            text = ax.text(j, i, data[i, j], ha = "center", va = "center")
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('frame_spin.png', dpi=300)





if __name__ == "__main__":
    main()
