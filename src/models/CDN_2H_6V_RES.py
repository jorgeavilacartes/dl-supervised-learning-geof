""" 
Residual Network

Replica en 1D de Deep Residual Learning for Image Recognition: https://arxiv.org/pdf/1512.03385.pdf
original:  https://github.com/c1ph3rr/Deep-Residual-Learning-for-Image-Recognition/blob/master/Resnet50.py
"""
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, 
    Conv1D,#Conv2D, 
    Dense, 
    MaxPool1D, #MaxPool2D, 
    Flatten, 
    GlobalAveragePooling1D, #GlobalAveragePooling2D, 
    Add, 
    Activation, 
    BatchNormalization, 
    ZeroPadding1D, #ZeroPadding2D,
)

# Reference name of model
MODEL_NAME = str(Path(__file__).resolve().stem)

INPUTS = OrderedDict(
            input_2H6V_L=dict(
                len_input=4000,
                leads=["I","II","V1","V2","V3","V4","V5","V6"]
            )
        )

class ModelECG:
    '''
    Generate an instance of a keras model
    '''
    def __init__(self, n_output: int, output_layer: str,):
        self.inputs = INPUTS
        self.model_name = MODEL_NAME
        self.n_output = n_output
        self.output_layer = output_layer

    @staticmethod
    def identity_block(inp, filters, kernel_size, block, layer):
        
        f1, f2, f3 = filters
        
        conv_name = 'id_conv_b' + block + '_l' + layer
        batch_name = 'id_batch_b' + block + '_l' + layer
        
        x = Conv1D(filters=f1, kernel_size=1, padding='same', kernel_initializer='he_normal', name=conv_name + '_a')(inp)
        x = BatchNormalization(name=batch_name + '_a')(x)
        x = Activation('relu')(x)
        
        x = Conv1D(filters=f2, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name + '_b')(x)
        x = BatchNormalization(name=batch_name + '_b')(x)
        x = Activation('relu')(x)
        
        x = Conv1D(filters=f3, kernel_size=1, padding='same', kernel_initializer='he_normal', name=conv_name + '_c')(x)
        x = BatchNormalization(name=batch_name + '_c')(x)
        
        add = Add()([inp, x])
        x = Activation('relu')(add)
        
        return x

    @staticmethod
    def convolutional_block(inp, filters, kernel_size, block, layer, strides=2):
        
        f1, f2, f3 = filters
        
        conv_name = 'res_conv_b' + block + '_l' + layer
        batch_name = 'res_batch_b' + block + '_l' + layer
        
        y = Conv1D(filters=f1, kernel_size=1, padding='same', strides=strides, kernel_initializer='he_normal', name=conv_name + '_a',)(inp)
        y = BatchNormalization(name=batch_name + '_a')(y)
        y = Activation('relu')(y)
        
        y = Conv1D(filters=f2, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name + '_b')(y)
        y = BatchNormalization(name=batch_name + '_b')(y)
        y = Activation('relu')(y)
        
        y = Conv1D(filters=f3, kernel_size=1, padding='same', kernel_initializer='he_normal', name=conv_name + '_c')(y)
        y = BatchNormalization(name=batch_name + '_c')(y)
        
        shortcut = Conv1D(filters=f3, kernel_size=1, strides=strides, kernel_initializer='he_normal', name=conv_name + '_shortcut')(inp)
        shortcut = BatchNormalization(name=batch_name + '_shortcut')(shortcut)
        
        add = Add()([shortcut, y])
        y = Activation('relu')(add)
        
        return y

    # Load model
    def get_model(self, shape_inputs: Optional[List[Tuple]] = None,):
        f"""keras model for {self.model_name}

        Args:
            n_output (int): number of neurons in the last layer
            output_layer (str): activation function of last layer
            shape_inputs (list of tuples): Eg: For two inputs [(4800,1),(1000,4)]
        """    
        if shape_inputs is None:
            shape_inputs = [(value.get("len_input"), len(value.get("leads"))) for value in self.inputs.values()]
        
        # Inputs     
        inp = Input(shape=shape_inputs[0])

        padd = ZeroPadding1D(3)(inp)

        conv1 = Conv1D(64, 7, strides=2, padding='valid', name='conv1',)(padd)
        conv1 = BatchNormalization(name='batch2')(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = ZeroPadding1D(1)(conv1)
        conv1 = MaxPool1D(3,2)(conv1)

        conv2 = self.convolutional_block(conv1, [64,64,256], 3, '2', '1', strides=1)
        conv2 = self.identity_block(conv2, [64,64,256], 3, '2', '2')
        conv2 = self.identity_block(conv2, [64,64,256], 3, '2', '3')

        conv3 = self.convolutional_block(conv2, [128,128,512], 3, '3', '1')
        conv3 = self.identity_block(conv3, [128,128,512], 3, '3', '2')
        conv3 = self.identity_block(conv3, [128,128,512], 3, '3', '3')
        conv3 = self.identity_block(conv3, [128,128,512], 3, '3', '4')

        conv4 = self.convolutional_block(conv3, [256,256,1024], 3, '4', '1')
        conv4 = self.identity_block(conv4, [256,256,1024], 3, '4', '2')
        conv4 = self.identity_block(conv4, [256,256,1024], 3, '4', '3')
        conv4 = self.identity_block(conv4, [256,256,1024], 3, '4', '4')
        conv4 = self.identity_block(conv4, [256,256,1024], 3, '4', '5')
        conv4 = self.identity_block(conv4, [256,256,1024], 3, '4', '6')

        conv5 = self.convolutional_block(conv4, [512,512,2048], 3, '5', '1')
        conv5 = self.identity_block(conv5, [512,512,2048], 3, '5', '2')
        conv5 = self.identity_block(conv5, [512,512,2048], 3, '5', '3')

        avg_pool = GlobalAveragePooling1D()(conv5)
        output = Dense(self.n_output, activation=self.output_layer, dtype = tf.float32)(avg_pool)
        
        model = Model(inp, output)                                                                                                                 
        
        return model