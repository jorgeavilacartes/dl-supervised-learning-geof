from typing import Callable, Optional, List
import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    """Data Generator for keras from a list of paths to files"""
    VERSION=1
    def __init__(self, 
                list_paths,
                labels,
                file_loader: Callable, # from file to array
                batch_creator: Callable, # from list of arrays to batch for the specific model
                preprocessing: Optional[Callable] = None, 
                augmentation: Optional[Callable] = None,
                encoder_output: Optional[Callable] = None,
                shuffle=True,
                batch_size=32,
                ):
        self.list_paths = list_paths
        self.labels = labels
        self.batch_creator = batch_creator
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.file_loader = file_loader
        self.encoder_output = encoder_output
        self.shuffle=shuffle
        self.batch_size=batch_size

        # Initialize first batch
        self.on_epoch_end()

    def on_epoch_end(self,):
        """Updates indexes after each epoch (starting for the epoch '0'"""
        self.indexes = np.arange(len(self.list_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # shuffle indexes in place

    def __len__(self):
        # Must be implemented
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_paths) / self.batch_size))

    def __getitem__(self, index):
        # Must be implemented
        """To feed the model with data in training
        It generates one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of paths to ecg
        list_paths_temp = [self.list_paths[k] for k in indexes]

        # Generate data
        X, y = self.input_output_generation(list_paths_temp)
        return X, y

    def input_output_generation(self, list_paths_temp: List[str]):
        """Generates and augment data containing batch_size samples

        Args:
            list_path_temp (List[str]): sublist of list_path

        Returns:
            X : numpy.array
            y : numpy.array hot-encoding
        """        
        
        # load inputs
        list_inputs = [self.file_loader(path) for path in list_paths_temp]
        
        # Augmentation
        if callable(self.augmentation):
            # FIXME: Aumentacion de imgaug se usa sobre un batch. Decidir que hacer
            #list_inputs = [self.augmentation(input) for input in list_inputs] # Con clase Pipeline
            list_inputs = self.augmentation(list_inputs) # Con imgaug
        
        # Preprocessing
        if callable(self.preprocessing):
            list_inputs = [self.preprocessing(input) for input in list_inputs]

        # print([x.shape for x in list_inputs])
        
        # Create batch for the selected model
        X = self.batch_creator(list_inputs)
        print("BATCH", X[0].shape)
        # outputs for the model as batch -> [[0,1,0,...], [0,0,1,...],[1,0,0,...]]
        y = np.array([self.encoder_output(self.labels.get(path)) for path in list_paths_temp])
        
        return X, y