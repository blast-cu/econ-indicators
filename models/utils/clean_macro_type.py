
import pickle
import os

DATA_DIR = 'data/clean/'

def main():
    
    quant_dict = pickle.load(open(os.path.join(DATA_DIR, 'quant_dict_clean'), 'rb'))

    # for k, v in quant_dict.items():
    #     if v['type'] == '\x00':
    #         print('found null')
    #     # print('type: ' + v['type'])
    #     # print('macro type: ' + v['macro_type'])
    #     # print()

    # for k, v in quant_dict.items():
    #     if v['type'] != 'macro' and v['type'] != '\x00':
    #         quant_dict[k]['macro_type'] = 'none'


    for k, v in quant_dict.items():
        # print("type: " + v['type'])
        # print(v['macro_type'])
        # print()

        print(v)

        print()
        


    # pickle.dump(quant_dict, open(os.path.join(DATA_DIR, 'quant_dict_clean'), 'wb'))




if __name__ == "__main__":
    main()

