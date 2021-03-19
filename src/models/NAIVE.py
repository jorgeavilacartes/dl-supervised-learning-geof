from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D, 
    Input, 
    Flatten, 
    Dense, 
    Dropout,
    concatenate
    )
from tensorflow.math import l2_normalize

# Reference name of model
MODEL_NAME = str(Path(__file__).resolve().stem)

# Default inputs
# Dictionary with {"Reference-name-of-input": {"len_input": <int>, "leads": <list with name of leads>}}
INPUTS = OrderedDict(
                target_size=250,
                channels=3
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

    # Load model
    def get_model(self,):
        f"""keras model for {self.model_name}

        Args:
            n_output (int): number of neurons in the last layer
            output_layer (str): activation function of last layer
            shape_inputs (list of tuples): Eg: For two inputs [(4800,1),(1000,4)]
        """    
    
        # Inputs     
        input_model = (
                    INPUTS.get("target_size"), 
                    INPUTS.get("target_size"), 
                    INPUTS.get("channels")
                    )
                    

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_model))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_output, activation=self.output_layer, dtype = tf.float32))
        return model
        # return Model(input_model, output)                                                                                                                 
        
