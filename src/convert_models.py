"""
@author: Jorge Avila
github.com/jorgeavilacartes
"""
import os
import tensorflow as tf
from time import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


class TFLiteConverter:
    """
    Get a tflite version of the model from one of:
    - json (architecture) and hdf5 (weights)
    - h5 (weights and architecture)
    - keras (keras model)

    Raises:
        Exception: 'path_h5' must have extension
        Exception: 'path_hdf5' and 'path_json' must have extension"

    Returns:
        tflite: a model.tflite version of the provided model
    """
    VERSION = 1
    def from_keras(self, model, path_tflite=None):
        """Convert keras model to tflite

        Args:
            model: keras model
            path_tflite (str, optional): path to save the model as 'tflite'. Defaults to None.
        """       
        # convert to tflite and save
        self.asTFLite(model, path_tflite)

    def from_h5(self, path_h5: str, path_tflite=None):
        """Convert model saved as h5 to tflite

        Args:
            path_h5 (str): path to a model with 'h5' extension
            path_tflite (str, optional): path to save the model as 'tflite'. Defaults to None.

        Raises:
            Exception: If 'path_h5' or 'path_tflite' are not valid paths.
        """
        
        if self.__is_h5(path_h5):
            model = self.load_model(path_h5=path_h5)
            # convert to tflite
            self.asTFLite(model, path_tflite)
        else:
            raise Exception("input 'path_h5' must have extension")

    def from_json_hdf5(self, path_json: str, path_hdf5: str, path_tflite=None):
        """Convert model saved as json+hdf5 to tflite

        Args:
            path_json (str): path to architecture, with 'json' extension
            path_hdf5 (str): path to weights, with 'hdf5' extension
            path_tflite (str, optional): path to save the model as 'tflite'. Defaults to None.

        Raises:
            Exception: If either 'path_json', 'path_hdf5' or 'path_tflite' are not valid paths.
        """        
        if self.__is_hdf5(path_hdf5) and self.__is_json(path_json):
            model = self.load_model(path_json=path_json, path_hdf5=path_hdf5)
            # convert to tflite
            self.asTFLite(model, path_tflite)
        else:
            raise Exception("inputs 'path_hdf5' and 'path_json' must have extension")
        

    @staticmethod
    def __is_h5(path: str):
        if str(path).endswith(".h5"):
            return True
        return False

    @staticmethod
    def __is_hdf5(path: str):
        if str(path).endswith(".hdf5"):
            return True
        return False
    
    @staticmethod
    def __is_json(path: str):
        if str(path).endswith(".json"):
            return True
        return False
    
    @staticmethod
    def __is_tflite(path: str):
        if str(path).endswith(".tflite"):
            return True
        return False
    

    def asTFLite(self, model, path_tflite):
        """Save keras model as tflite.
        If 'path_tflite' is None, it will be set to 'model.tflite'
        """
        path_tflite = path_tflite if path_tflite else "model.tflite"
        if self.__is_tflite(path_tflite):
            print("Converting to tflite")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open(path_tflite, 'wb') as save_model: 
                save_model.write(tflite_model)
            print(f"Model saved at {path_tflite}")
        else:
            raise("Must provide a valid .tflite path to save the model")

    def load_model(self,*, path_h5=None, path_json=None, path_hdf5=None):
        if path_h5: 
            print("Loading model from h5")
            # Load model from h5
            model = tf.keras.models.load_model(path_h5)
        elif path_json and path_hdf5:
            print("Loading model from json and hdf5")
            # Load architecture from json
            with open(path_json, 'r') as fp:
                loaded_model_json = fp.read()
            
            # Load as keras model
            model = tf.keras.models.model_from_json(loaded_model_json)

            # Add weights from hdf5
            model.load_weights(str(path_hdf5))
        else: 
            raise("Must provide either a valid 'h5' or 'json' and 'hdf5' files")
        
        return model