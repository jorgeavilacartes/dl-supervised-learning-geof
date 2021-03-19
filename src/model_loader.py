"""
@author: Jorge Avila, Rocuant Roberto
github.com/jorgeavilacartes, github.com/Nouvellie
"""
# Default imports
import re
import os  
import traceback
from abc import ABCMeta, abstractmethod # To force to implements some methods in a Base class

# Third party libraries
import numpy as np

# Local imports
from .file_loader import ImageLoader as FileLoader 

from .decoder_output import DecoderOutput  
from .preprocessing import (
    Pipeline, 
    register_in_pipeline,
)

from .models.inputs_model import InputsModelECG as InputsModel # TODO:Cambiar aca el loader de array -> input modelo
from .utils import fromJSON

from math import floor
from tensorflow import (
	convert_to_tensor,
	lite,
)
from tensorflow.keras.models import model_from_json
from pathlib import Path
from time import time


class BaseModelLoader(metaclass=ABCMeta):
    """Metaclass to define loaders for models"""
    def __init__(self, dir_model):
        self.dir_model=dir_model
        self.load_training_config()
        self.load_file_loader() # file to array
        self.load_preprocessing() # preprocess array
        self.load_input_model() # array as inputs for the model
        self.load_postprocessing() # get prediction based on classes
        self.preload_model()

    def load_training_config(self,):
        """Function to load training_config.json"""
        training_config_path = Path(f"{self.dir_model}/training_config.json")
        self.training_config = fromJSON(training_config_path)
        self.preprocessing_to_model = self.training_config.get("preprocessing_to_model", False) # not use ecg2input in generate_input_model()

    def load_file_loader(self,):
        """Function to load file as array"""
        INFO_LOADER = self.training_config.get("file_loader")#dict(scale=self.training_config.get("scale"))
        self.file_loader = FileLoader(**INFO_LOADER)

    def load_input_model(self,):
        #TODO: Esto depende del modelo, ya sea ECG, o RX, u otro
        
        # load dictionary with input configuration for the model
        name_inputs = [name for name in os.listdir(self.dir_model) if name.endswith("inputs.json")][0]
        inputs_model_path = Path(f"{self.dir_model}/{name_inputs}")

        #Input to initialize inputs model
        inputs_model = dict(
            model_name=self.training_config.get("model_name"), 
            order_leads=self.training_config.get("order_leads"), 
            order_batch=self.training_config.get("order_batch"),
            inputs_model_fromJSON=inputs_model_path
        )
        self.InputsModel = InputsModel(**inputs_model)
    
    def load_preprocessing(self,):
        """Function to apply preprocessing to an array"""
        preprocessing_path = Path(f"{self.dir_model}/preprocessing.json")
        self.preprocessing = Pipeline()
        self.preprocessing.fromJSON(preprocessing_path)

    def load_postprocessing(self,):
        """Function to apply postprocessing to the output of the model"""
        decoder_path = Path(f"{self.dir_model}/postprocessing.json")
        self.decoder = DecoderOutput()
        self.decoder.fromJSON(decoder_path)

    def generate_input_model(self, input):
        """From file->array->preprocessing->input for the model"""
        # Load file as array
        input = self.file_loader(input)#, only_signal=True, class_ecg=True) #FIXME class_ecg=True para probar NAb 
        
        # Preprocessing
        input = self.preprocessing(input) 

        # Transform as input for the model in inheritance
        if not self.preprocessing_to_model:
            input = self.InputsModel.ecg2input(input) # TODO: Esto depende del modelo ECG, RX, etc

        return input

    @abstractmethod # This method must be implemented
    def preload_model(self,):
        pass

    @abstractmethod # This method must be implemented in inheritance
    def predict(self,):
        pass

class ModelLoaderTFLITE(BaseModelLoader):
    """
    How to use
    >> # Instantiate model
    >> model = ModelLoaderTFLite("folder/to/my/model/files")
    >> pred = model("path/to/my/input_file", bool_confidence=True)
    """
    NUM_THREADS = 0

    def preload_model(self,):
        """Preload tflite model"""
        name_tflite = [name for name in os.listdir(self.dir_model) if name.endswith(".tflite")][0]
        model_path = Path(f"{self.dir_model}/{name_tflite}")
        
        if self.NUM_THREADS > 0:
            self.interpreter = lite.Interpreter(model_path=str(model_path), num_threads=self.NUM_THREADS)

        else:
            self.interpreter = lite.Interpreter(model_path=str(model_path))

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, input, confidence_bool=False,):
        # TODO: Must be implemented
        """Prediction of one input using tflite model

        Args:
            input ([type]): path to file
            confidence_bool (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """        
        try:
            # Pre-processing
            X = self.generate_input_model(input)

            # Predict
            for j,x in enumerate(X):
                input_X = convert_to_tensor(np.array(x), np.float32)
                self.interpreter.set_tensor(self.input_details[j]['index'], input_X)
            
            self.interpreter.invoke()
            
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])

            result = self.decoder.decode_output(prediction, include_confidence=confidence_bool)

            return result

        except Exception as e:
            full_traceback = re.sub(r"\n\s*", " || ", traceback.format_exc())
            print(full_traceback, e)

class ModelLoaderJSONHDF5(BaseModelLoader):
    
    def preload_model(self,):
        name_architecture = [name for name in os.listdir(self.dir_model) if name.startswith("architecture")][0]
        name_weights = [name for name in os.listdir(self.dir_model) if name.endswith(".hdf5")][0]
        print(name_architecture)
        print(name_weights)
        # Load architecture
        architecture_path = Path(f"{self.dir_model}/{name_architecture}")
        with open(str(architecture_path), 'r') as json_file:
            model = model_from_json(json_file.read())
        
        # Add weights
        weights_path = Path(f"{self.dir_model}/{name_weights}")
        model.load_weights(str(weights_path))
        self.model = model

    def predict(self, input, confidence_bool=False,):
        # TODO: Must be implemented
        """Prediction of one input_file using tflite model

        Args:
            input ([type]): path to file
            confidence_bool (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """        
        try:
            # Pre-processing
            X = self.generate_input_model(input)

            # Predict
            prediction = self.model.predict(X)
            
            # Decode Output
            result = self.decoder.decode_output(prediction, include_confidence=confidence_bool)

            return result

        except Exception as e:
            full_traceback = re.sub(r"\n\s*", " || ", traceback.format_exc())
            print(full_traceback, e)