import os
import glob
import importlib
from pathlib import Path
import numpy as np

OMMIT = {".ipynb_checkpoints","__pycache__","__init__"}
BASE_DIR = Path(__file__).resolve().parent
#___________________________

class ModelFactory:
    """
    Model factory for Keras default models: RGB input images
    """
    AVAILABLE_MODELS = [model[:-3] for model in os.listdir(BASE_DIR.joinpath('models')) if all([ommit not in model for ommit in OMMIT])]

    def __init__(self,):
        """Initialize ModelFactory"""        
        self.models_ = dict(
            CDN_1L4S=dict(
                module_name="CDN_1L4S",
            )
        )

    def get_model(self, class_names, 
                        model_name="CDN_1L4S", 
                        weights_path=None,
                        output_layer = 'softmax',
                        n_output = None
                        ):
        """get a keras model for ECG

        Args:
            class_names (list): list with all the classes involved
            model_name (str, optional): one of the available models. EG: "CDN_1L4S" 
            weights_path (str, optional): path to weights (Eg: weights_model.h5 / .hdf5 file). Defaults to None (random initialization).
            output_layer (str, optional): 'softmax' for binary classification with two outputs or multiclass classification (many outputs, one possible).
                                        'sigmoid' for binary classification with one output or multilabel classification (many outputs, many possibles).
                                        Defaults to 'softmax'.
            n_output (int, optional): If None, n_output is set to len(class_names), except in the next cases assumed as binary classification
                                
                                    [Binary Classification] two classes, one possible output
                                        When len(classes) == 2 
                                        If output_layer = 'sigmoid' then n_ouput is set to 1 
                                        If output_layer = 'softmax' then n_ouput is set to 2
                                    
                                    The other possible cases are: 

                                    [Multiclass Classification] many classes, one possible output
                                        output_layer = 'softmax' n_output = len(class_names) > 2

                                    [Multilabel Classification] many classes, many possible outputs
                                        output_layer = 'sigmoid' n_output = len(class_names) > 2

                                    Defaults to None.

        Returns:
            keras.model: keras model
        """                        
        assert isinstance(class_names,list), "'class_names' must be a list"
        assert output_layer in {'softmax','sigmoid'}, f"output_layer must be either 'sigmoid' or 'softmax'"              
        assert model_name in self.AVAILABLE_MODELS, f"'model_name' must be one of {self.AVAILABLE_MODELS}. Yours is {model_name}" # available keras models. See keras applications
        assert isinstance(n_output, int) or n_output is None, "'n_output' must be an integer"


        # Neurons in output layer
        # Special case: binary classification
        if n_output is None:
            if len(class_names) == 2:
                n_output = 1 if output_layer == 'sigmoid' else 2
            else:
                n_output = len(class_names)

        # Call class of model to load
        ModelECG = getattr(
            importlib.import_module(
                f"src.models.{model_name}"
            ),
            "ModelECG")

        model_ecg = ModelECG(n_output=n_output, output_layer=output_layer)
        print("***En ", str(Path(__file__).name))
        # Load architecture
        model = model_ecg.get_model()
    
        # Load weights to the model from file
        if weights_path is not None:
            print(f"\n **load model weights_path** : {weights_path}")
            model.load_weights(weights_path)

        print("\n**Model created**")        
        inputs_model = model_ecg.inputs
        return model, inputs_model
