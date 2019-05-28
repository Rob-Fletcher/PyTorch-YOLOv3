import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import keras
import keras.backend as K
import torch
from torch import nn
import numpy as np
import tensorflow as tf
from models import *


def keras_to_pyt(km, pm):
    weight_dict = dict()
    i = 0
    for layer in km.layers:
        if type(layer) is keras.layers.convolutional.Conv2D:
            weight_dict[f'module_list.{i}.conv_{i}.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
           # weight_dict[f'conv_{i}.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.normalization.BatchNormalization:
            weight_dict[f'module_list.{i}.batch_norm_{i}.weight']      = layer.get_weights()[0]  # Gamma in keras
            weight_dict[f'module_list.{i}.batch_norm_{i}.bias']        = layer.get_weights()[1]  # Beta in keras
            weight_dict[f'module_list.{i}.batch_norm_{i}.running_mean']= layer.get_weights()[2]  # mean in keras
            weight_dict[f'module_list.{i}.batch_norm_{i}.running_var'] = layer.get_weights()[3]  # std in keras
        elif type(layer) is keras.layers.advanced_activations.LeakyReLU:
            i+=1

    pyt_state_dict = dict()
    for key in weight_dict.keys():
        pyt_state_dict[key] = torch.from_numpy(weight_dict[key])

    return pyt_state_dict

if __name__=='__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-k', '--keras', help='Keras model file .h5')
    argparser.add_argument('-p', '--pytorch', help='PyTorch weights file to write to.')
    args = argparser.parse_args()

    print(f"Loading keras model: {args.keras} ...")
    km = keras.models.load_model(args.keras)
    print("Done.")

    print(f"Loading PyTorch model: config/yolov3.cfg")
    pm = Darknet("config/yolov3.cfg", 416)
    print("Done.")
    ml = list(pm.children())[0]
    layer = list(ml.children())
    print(len(layer))


    raise Exception()

    state_dict = keras_to_pyt(km, pm)

    print(f"Saving the PyTorch state_dict to {args.pytorch}")
    torch.save(state_dict, args.pytorch)
    print("Done.")
