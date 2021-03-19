import importlib
import numpy as np
from src.utils import fromJSON

class InputsModel:
    '''
    Generate inputs for the current model
    '''
    def __init__(self, model_name, inputs_model_fromJSON=None):
        self.model_name = model_name
        self.inputs_model_fromJSON = inputs_model_fromJSON
        self.get_inputs_model()

    def get_inputs_model(self,):
        # If inputs model are provided from a JSON
        if self.inputs_model_fromJSON is None:
            self.inputs = getattr(
            importlib.import_module(
                f"src.models.{self.model_name}"
            ),
            "INPUTS")
        else:
            # load inputs model from json
            self.inputs = fromJSON(self.inputs_model_fromJSON)

    # numpy ecg as input for the model
    def img2input(self, img):
        f"""get input for {self.model_name} for one image
        return each image with batch dimension (1, target_size, target_size, channel)
        Args:
            img (np.array): array image
        """
        return np.expand_dims(img, axis=0)
    
    def create_batch(self, list_img,):
        """Create batch
        list of numpy images to batches for the model"""
    
        list_inputs = [self.img2input(img) for img in list_img] # Cada imagen es (1, target_size, target_size, channel)
        #print([x.shape for x in list_inputs])
        return np.concatenate(list_inputs, axis=0)