from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
import data_utils.model_utils.dataset as d

"""
General utility functions for the roberta classifier model used by both qual and quant.
"""

def settings(args, type):
    """
    Function to retrieve the settings for the roberta classifier model.

    Args:
        args (object): The arguments object containing the model settings.
        type (str): The type of annotations for the classifier model (qual or quant).

    Returns:
        tuple: A tuple containing the output directory, model checkpoint, add noise flag, and best noise flag.
    """

    global MODEL_CHECKPOINT
    MODEL_CHECKPOINT = None

    global ADD_NOISE
    ADD_NOISE = False
    global BEST_NOISE
    BEST_NOISE = False

    global OUT_DIR

    if args.m == 'dapt':  # using domain-adapted model
        MODEL_CHECKPOINT = "data/masked/"
    elif args.m == 'base':  # using base model
        MODEL_CHECKPOINT = "roberta-base"
    elif args.m == 'large':
        MODEL_CHECKPOINT = "roberta-large"
    else:
        raise ValueError("Invalid setting: {}".format(args.m))

    # noisy annotations added to training set
    if args.n is not None:  
        ADD_NOISE = True
        if args.n == 'best':  # only use 'best' annotator's value for annotation
            BEST_NOISE = True
        elif args.n == 'all':  # use all annotator's values
            BEST_NOISE = False
        else:
            raise ValueError("Invalid setting: {}".format(args.n))

    setting_name = f'qual_roberta_{args.m}'
    if args.n is not None:
        setting_name += f'_noise_{args.n}'

    OUT_DIR = f"{d.ROBERTA_MODEL_DIR}/{type}_{setting_name}/"

    return OUT_DIR, MODEL_CHECKPOINT, ADD_NOISE, BEST_NOISE


def get_weights(y, annotation_map: dict):
    """
    Computes class weights for weighted loss.

    Parameters:
    - y: list or array of target values
    - annotation_map: dictionary mapping target values to class labels

    Returns:
    - class_weights: tensor of class weights, to be used in loss function
    """

    # compute class weights for weighted loss
    classes = list(set(list(annotation_map.values())))
    
    if len(classes) != len(set(y)):
        y = y + classes
    classes = np.array(classes)
    weight = \
        compute_class_weight(class_weight='balanced', classes=classes, y=y)

    weight = [float(w) for w in weight]
    class_weights = torch.tensor(weight).to('cuda')
    return class_weights


def check_done(val_f1_history: list, val_f1, patience, history_len):
    """
    Check if the model has stopped improving based on the validation loss history.

    Parameters:
    - val_loss_history (list): A list of previous validation losses.
    - val_loss (float): The current validation loss.
    - patience (float): The minimum difference between the current validation loss and the previous one to consider improvement.
    - history_len (int): The maximum length of the validation loss history.

    Returns:
    - improving (bool): True if the model is still improving, False otherwise.
    - val_loss_history (list): The updated validation loss history.
    """
    improving = True
    if len(val_f1_history) == history_len:
        val_f1_history.pop(0)  # remove at index 0

        if val_f1_history[0] == val_f1_history[1]:
            if val_f1_history[1] == val_f1:
                improving = False

    val_f1_history.append(val_f1)
    return improving, val_f1_history
