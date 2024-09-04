import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

qual_dict = pickle.load(open('data/clean/qual_dict', 'rb'))
quant_dict = pickle.load(open('data/clean/quant_dict', 'rb'))

def sort_bars(labels, counts):
    zipped = zip(labels, counts)
    sorted_zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
    labels, counts = zip(*sorted_zipped)
    return labels, counts

def main():

        # QUAL ANNS

    # frame plot
    frame_labels = ['business', 'industry', 'macro', 'government', 'other']
    frame_ann_counts = {}
    for id, ann_dict in qual_dict.items():
        if ann_dict['frame'] != '\x00':
            val = ann_dict['frame']
            if val not in frame_ann_counts.keys():
                frame_ann_counts[val] = 0
            frame_ann_counts[val] += 1

    frame_counts = []
    for label in frame_labels:
        frame_counts.append(frame_ann_counts[label])
    frame_labels = ['other', 'business', 'industry', 'macro', 'gov']

    # econ_rate plot
    econ_rate_labels = ['none', 'poor', 'good']
    econ_rate_ann_counts = {}
    for id, ann_dict in qual_dict.items():
        if ann_dict['econ_rate'] != '\x00':
            val = ann_dict['econ_rate']
            if val not in econ_rate_ann_counts.keys():
                econ_rate_ann_counts[val] = 0
            econ_rate_ann_counts[val] += 1

    # econ_change plot
    econ_rate_counts = []
    for label in econ_rate_labels:
        econ_rate_counts.append(econ_rate_ann_counts[label])

    econ_change_labels = ['none', 'same', 'worse', 'better']

    econ_change_ann_counts = {}
    for id, ann_dict in qual_dict.items():
        if ann_dict['econ_change'] != '\x00':
            val = ann_dict['econ_change']
            if val not in econ_change_ann_counts.keys():
                econ_change_ann_counts[val] = 0
            econ_change_ann_counts[val] += 1
    
    econ_change_counts = []
    for label in econ_change_labels:
        econ_change_counts.append(econ_change_ann_counts[label])

    # QUANT ANNS
    # type plot
    type_labels = [
        'macro', 'industry', 'government', 'personal', 'business', 'other'
    ]
    type_ann_counts = {}
    for id, ann_dict in quant_dict.items():
        if ann_dict['type'] != '\x00':
            val = ann_dict['type']
            if val not in type_ann_counts.keys():
                type_ann_counts[val] = 0
            type_ann_counts[val] += 1

    type_counts = []
    for label in type_labels:
        type_counts.append(type_ann_counts[label])
    type_labels = ['macro', 'industry', 'gov', 'personal', 'business', 'other']

    spin_labels = ['neutral', 'neg', 'pos']

    spin_ann_counts = {}
    for id, ann_dict in quant_dict.items():
        if ann_dict['spin'] != '\x00':
            val = ann_dict['spin']
            if val not in spin_ann_counts.keys():
                spin_ann_counts[val] = 0
            spin_ann_counts[val] += 1
    
    spin_counts = []
    for label in spin_labels:
        spin_counts.append(spin_ann_counts[label])
    spin_labels = ['neut', 'neg', 'pos']

    # macro_type_labels = ['none', 'other', 'housing', 'currency', 'market', 'macro', 'wages', 'energy', 'prices', 'interest', 'retail', 'jobs']
    macro_type_labels = ['other', 'housing', 'currency', 'market', 'macro', 'wages', 'energy', 'prices', 'interest', 'retail', 'jobs']
    
    macro_type_ann_counts = {}
    for id, ann_dict in quant_dict.items():
        if ann_dict['macro_type'] != '\x00':
            val = ann_dict['macro_type']
            if val not in macro_type_ann_counts.keys():
                macro_type_ann_counts[val] = 0
            macro_type_ann_counts[val] += 1
    
    macro_type_counts = []
    for label in macro_type_labels:
        macro_type_counts.append(macro_type_ann_counts[label])
    


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col', constrained_layout=True)
    fig.set_size_inches(10, 10)
    # plt.rc('font', size=20)

    # ax1.bar(frame_labels, frame_counts)
    frame_labels, frame_counts = sort_bars(frame_labels, frame_counts)
    econ_rate_labels, econ_rate_counts = sort_bars(econ_rate_labels, econ_rate_counts)
    econ_change_labels, econ_change_counts = sort_bars(econ_change_labels, econ_change_counts)

    ax1.barh(frame_labels, frame_counts)
    ax1.set_ylabel('Article Type', size=16)
    # ax1.get_yaxis().set_label_coords(-0.3, 0.5)
    ax2.barh(econ_rate_labels, econ_rate_counts)
    ax2.set_ylabel('Economic Cond.', size=16)
    # ax2.get_yaxis().set_label_coords(-0.3, 0.5)
    ax3.barh(econ_change_labels, econ_change_counts)
    ax3.set_ylabel('Economic Dir.', size=16)
    # ax3.get_yaxis().set_label_coords(-0.3, 0.5)

    fig.supxlabel('Annotation Counts', size=16)

    tick_size = 14
    ax1.tick_params(axis='both', which='major', labelsize=tick_size)
    ax2.tick_params(axis='both', which='major', labelsize=tick_size)
    ax3.tick_params(axis='both', which='major', labelsize=tick_size)

    fig.tight_layout()
    fig.align_ylabels()
    plt.savefig('data_utils/visualization/results/frame_dist_qual.png', dpi=300)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col', constrained_layout=True)
    fig.set_size_inches(10, 10)

    type_labels, type_counts = sort_bars(type_labels, type_counts)
    spin_labels, spin_counts = sort_bars(spin_labels, spin_counts)
    macro_type_labels, macro_type_counts = sort_bars(macro_type_labels, macro_type_counts)

    ax1.barh(type_labels, type_counts)
    ax1.set_ylabel('Quantity Type', size=16)
    # ax1.get_yaxis().set_label_coords(-0.35, 0.5)
    ax2.barh(spin_labels, spin_counts)
    ax2.set_ylabel('Indicator Polarity', size=16)
    # ax2.get_yaxis().set_label_coords(-0.35, 0.5)
    ax3.barh(macro_type_labels, macro_type_counts)
    ax3.set_ylabel('Macro Indicator', size=16)
    # ax3.get_yaxis().set_label_coords(-0.35, 0.5)
    

    # Show graphic
    tick_size = 14
    ax1.tick_params(axis='both', which='major', labelsize=tick_size)
    ax2.tick_params(axis='both', which='major', labelsize=tick_size)
    ax3.tick_params(axis='both', which='major', labelsize=tick_size)

    fig.supxlabel('Annotation Counts', size=16)
    fig.tight_layout()
    fig.align_ylabels()
    # plt.show()
    plt.savefig('data_utils/visualization/results/frame_dist_quant.png', dpi=300)

    # # QUAL ANNS

    # # frame plot
    # frame_labels = ['business', 'industry', 'macro', 'government', 'other']
    # frame_ann_counts = {}
    # for id, ann_dict in qual_dict.items():
    #     if ann_dict['frame'] != '\x00':
    #         val = ann_dict['frame']
    #         if val not in frame_ann_counts.keys():
    #             frame_ann_counts[val] = 0
    #         frame_ann_counts[val] += 1

    # frame_counts = []
    # for label in frame_labels:
    #     frame_counts.append(frame_ann_counts[label])
    # frame_labels = ['other', 'business', 'industry', 'macro', 'gov']

    # # econ_rate plot
    # econ_rate_labels = ['none', 'poor', 'good']
    # econ_rate_ann_counts = {}
    # for id, ann_dict in qual_dict.items():
    #     if ann_dict['econ_rate'] != '\x00':
    #         val = ann_dict['econ_rate']
    #         if val not in econ_rate_ann_counts.keys():
    #             econ_rate_ann_counts[val] = 0
    #         econ_rate_ann_counts[val] += 1

    # # econ_change plot
    # econ_rate_counts = []
    # for label in econ_rate_labels:
    #     econ_rate_counts.append(econ_rate_ann_counts[label])

    # econ_change_labels = ['none', 'same', 'worse', 'better']

    # econ_change_ann_counts = {}
    # for id, ann_dict in qual_dict.items():
    #     if ann_dict['econ_change'] != '\x00':
    #         val = ann_dict['econ_change']
    #         if val not in econ_change_ann_counts.keys():
    #             econ_change_ann_counts[val] = 0
    #         econ_change_ann_counts[val] += 1
    
    # econ_change_counts = []
    # for label in econ_change_labels:
    #     econ_change_counts.append(econ_change_ann_counts[label])

    # # QUANT ANNS
    # # type plot
    # type_labels = [
    #     'macro', 'industry', 'government', 'personal', 'business', 'other'
    # ]
    # type_ann_counts = {}
    # for id, ann_dict in quant_dict.items():
    #     if ann_dict['type'] != '\x00':
    #         val = ann_dict['type']
    #         if val not in type_ann_counts.keys():
    #             type_ann_counts[val] = 0
    #         type_ann_counts[val] += 1

    # type_counts = []
    # for label in type_labels:
    #     type_counts.append(type_ann_counts[label])
    # type_labels = ['macro', 'industry', 'gov', 'personal', 'business', 'other']

    # spin_labels = ['neutral', 'neg', 'pos']

    # spin_ann_counts = {}
    # for id, ann_dict in quant_dict.items():
    #     if ann_dict['spin'] != '\x00':
    #         val = ann_dict['spin']
    #         if val not in spin_ann_counts.keys():
    #             spin_ann_counts[val] = 0
    #         spin_ann_counts[val] += 1
    
    # spin_counts = []
    # for label in spin_labels:
    #     spin_counts.append(spin_ann_counts[label])
    # spin_labels = ['neut', 'neg', 'pos']

    # # macro_type_labels = ['none', 'other', 'housing', 'currency', 'market', 'macro', 'wages', 'energy', 'prices', 'interest', 'retail', 'jobs']
    # macro_type_labels = ['other', 'housing', 'currency', 'market', 'macro', 'wages', 'energy', 'prices', 'interest', 'retail', 'jobs']
    
    # macro_type_ann_counts = {}
    # for id, ann_dict in quant_dict.items():
    #     if ann_dict['macro_type'] != '\x00':
    #         val = ann_dict['macro_type']
    #         if val not in macro_type_ann_counts.keys():
    #             macro_type_ann_counts[val] = 0
    #         macro_type_ann_counts[val] += 1
    
    # macro_type_counts = []
    # for label in macro_type_labels:
    #     macro_type_counts.append(macro_type_ann_counts[label])
    


    # fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='col', constrained_layout=True)
    # fig.set_size_inches(10, 10)
    # # plt.rc('font', size=20)

    # # ax1.bar(frame_labels, frame_counts)
    # ax1.barh(frame_labels, frame_counts)
    # ax1.set_ylabel('Article Type', size=18)
    # ax1.get_yaxis().set_label_coords(-0.3, 0.5)
    # ax3.barh(econ_rate_labels, econ_rate_counts)
    # ax3.set_ylabel('Economic Conditions', size=18)
    # ax3.get_yaxis().set_label_coords(-0.3, 0.5)
    # ax5.barh(econ_change_labels, econ_change_counts)
    # ax5.set_ylabel('Economic Direction', size=18)
    # ax5.get_yaxis().set_label_coords(-0.3, 0.5)

    # ax2.barh(type_labels, type_counts)
    # ax2.set_ylabel('Quantity Type', size=18)
    # ax2.get_yaxis().set_label_coords(-0.35, 0.5)
    # ax4.barh(spin_labels, spin_counts)
    # ax4.set_ylabel('Indicator Polarity', size=18)
    # ax4.get_yaxis().set_label_coords(-0.35, 0.5)
    # ax6.barh(macro_type_labels, macro_type_counts)
    # ax6.set_ylabel('Macro Indicator', size=18)
    # ax6.get_yaxis().set_label_coords(-0.35, 0.5)
    

    # # Show graphic
    # # fig.tight_layout(pad=0.5)
    # fig.supxlabel('Annotation Counts', size=24)
    # # plt.show()
    # plt.savefig('data_utils/visualization/results/frame_dist.png', dpi=300)

if __name__ == "__main__":
    main()