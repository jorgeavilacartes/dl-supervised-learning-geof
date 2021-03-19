# Default python libraries
import os
import json
import datetime
from pathlib import Path

# Third party
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# local imports
from src import (
    FileLoader,
    Preprocessing,
    Augmentation,
    EncoderOutput,
    DecoderOutput,
    DataGenerator,
    ModelFactory,
    TFLiteConverter,
    InputsModel,
)


from src.utils import (
    asJSON,
    fromJSON,
    set_logger,
    get_count_labels,
    import_batch_creator,
)


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
tf.keras.backend.clear_session()

# Mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

# Path to save train results
path_save_train_results = Path("train_results")
path_save_train_results.mkdir(parents=True, exist_ok=True)

# Create logger
logger = set_logger(path_log = "train_results/logs_training.log")

name_experiment = f"" # -> folder inside '/logs'
order_output_model = ["Perro", "Gato"] # None -> alphabetic order of classes from values of "data/encode_labels.json"

# Load Data
paths_train = fromJSON("data/list_train.json")
paths_val   = fromJSON("data/list_val.json")
labels = fromJSON("data/labels.json")
encode_labels = fromJSON("data/encode_labels.json")

# ! Numpy INFO !
# ecg will be loaded as numpy arrays with shape (12,5000) / (leads, signal)
# preprocessing and augmentation are applied to this shape configuration
# finally, the batch creator must to apply any transposition if needed to fit the model

# Info about Image
img_format =  "rgb"

# Model to use
model_name = 'NAIVE'
weights_path=None # None means that random weights will be used to initialize the net
output_layer = 'softmax' # activation function of last layer, 'sigmod' or 'softmax'
n_output = None # neurons in last layer. None -> default to len(class_names)

# Batches configuration: (batch_size, signal, lead)  
batch_size = 4
epochs = 3

# Optimizer
# optimizer=tf.keras.optimizers.Adam(
#     learning_rate=0.003,
#     beta_1=0.9,
#     beta_2=0.999,
#     epsilon=1e-07,
#     amsgrad=False,
#     name="Adam"
# )
optimizer=tf.keras.optimizers.Nadam(
    learning_rate=0.003, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=1e-07, 
    name="Nadam"
)
name_reference_optimizer = "nadam"

# Loss
loss = "binary_crossentropy"
name_reference_loss = "binary_crossentropy" #Name to use in training_config put in the dictionary (needed when using a custom loss)
bool_weighted_loss = True # True: use weighted loss using training set

# Metrics
metrics=["accuracy"]
name_reference_metrics = ["accuracy"]

# preprocessing inputs
target_size = 250

# All components of DataGenerator are:
# load ecg, preprocessing, augmentation, batch creation, encoder output 
inputs_file_loader = dict(
    format=img_format
)
file_loader = FileLoader(**inputs_file_loader)


# Generator of inputs for the model
inputs_model_ecg = dict(
    model_name=model_name,
)
inputs_model = InputsModel(**inputs_model_ecg)
batch_creator = inputs_model.create_batch

# Define preprocessing
preprocessing = Preprocessing(
    [
        ("rescale", dict(target_size=target_size)),
    ]
)

# Define augmentation
augmentation = None#Augmentation()

encoder_output = EncoderOutput(
    order_output_model=order_output_model, 
    encode_labels=encode_labels # dict to map labels in 'data/labels.json' to other classes
)

# shared configuration for both train and test data generators
config_generator = dict(
    labels=labels,
    file_loader=file_loader,
    batch_creator=batch_creator,
    preprocessing=preprocessing,
    augmentation=augmentation,
    encoder_output=encoder_output,
    batch_size=batch_size
)

# Instantiate DataGenerator for training set
train_generator = DataGenerator(
    list_paths=paths_train,
    **config_generator
)
logger.info("train_generator created.")                    

# Instantiate DataGenerator for validation set
val_generator = DataGenerator(
    list_paths=paths_val,
    shuffle=True,
    **config_generator,
)
logger.info("val_generator created.")

# Call model to use
model_factory = ModelFactory()
model, inputs_model = model_factory.get_model(
    class_names=order_output_model, 
    model_name=model_name,     
    weights_path=weights_path, 
    output_layer = output_layer,
    n_output=None
)
logger.info(f"Model created: {model_name}, see 'train_config.json' for details.")

# Compile model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
logger.info("Model compiled")

# Save architecture
model_json = model.to_json()
architecture = f"architecture-{model_name}-{output_layer}.json"
path_save_model = f"train_results/{architecture}"
with open(path_save_model, "w") as json_file:
    json_file.write(model_json)

## Callbacks
# ModelCheckpoint
weights_file = f'{path_save_train_results}/weights-{model_name}-' + 'epoch{epoch:03d}-val_acc{val_accuracy:.3f}.hdf5'
# Tensorboard
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
log_dir = os.path.join(
    "logs",
    f"{model_name}_{now}",
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=weights_file, save_best_only=True, save_weights_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001),
    tf.keras.callbacks.EarlyStopping(patience=7),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

if bool_weighted_loss:
    labels_train = [encode_labels.get(labels.get(path)[0]) for path in paths_train]
    class_weight = compute_class_weight('balanced', classes = order_output_model, y = labels_train)
    dict_class_weights = {j: weight for j,weight in enumerate(class_weight)}
    print(f"\n** class_weights {dict_class_weights}\n")
else:
    print("\n** class_weights None\n")
    class_weights = None


# Train model on dataset
logger.info(f"Begin training.")

# # Prueba sin generador
# X,y=train_generator.__getitem__(10)
# history_train = model.fit(
#     x=X,
#     y=y,
#     epochs=epochs,
#     validation_data=(X,y),#val_generator,
#     callbacks = callbacks,
#     #class_weight=class_weights
# )

history_train = model.fit(
    x=train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks = callbacks,
    class_weight=dict_class_weights
)
logger.info("End training.")

logger.info("Saving training info") 

# All weights
all_weights = os.listdir(path_save_train_results)
all_weights.sort(reverse=True) # From last to first

training_config = {
    "batch_size": batch_size,
    "target_size": target_size,
    "order_output_model": order_output_model,
    "epochs": epochs,
    "file_loader": inputs_file_loader,
    "list_preprocessing": preprocessing.pipeline,
    # "list_augmentation": augmentation.pipeline,
    "model_name": model_name,
    "weights_path": weights_path,
    "output_layer": output_layer,
    "n_output": n_output,
    "optimizer": name_reference_optimizer,
    "loss": name_reference_loss,
    "class_weights": list(class_weight) if bool_weighted_loss else None,
    "metrics": name_reference_metrics,
    "train_N_ecg": len(paths_train),
    "train_labels":  get_count_labels(paths_train, labels, encode_labels),
    "val_N_ecg": len(paths_val),
    "val_labels":  get_count_labels(paths_val, labels, encode_labels),
    "weights": [weight for weight in all_weights if weight.endswith("hdf5")][-1],
    "architecture": architecture
}

# Save training configuration
asJSON(training_config, "train_results/training_config.json", sort_keys=False)
logger.info("training info saved")

# Save input configuration of the model
asJSON(inputs_model, "train_results/{}_inputs.json".format(model_name), sort_keys=False)
logger.info("training info saved")

# Save preprocessing and augmentation
preprocessing.asJSON("train_results/preprocessing.json")
logger.info("preprocessing.json saved.")
# augmentation.asJSON("train_results/augmentation.json")
logger.info("augmentation.json saved.")

# TODO save postprocessing.json
postprocessing = {
    "order_output_model":order_output_model,
    "decode_output_by": "argmax"
}
asJSON(postprocessing,"train_results/postprocessing.json", sort_keys=False)
logger.info("postprocessing.json saved.")

# # Save tflite
# path_save = Path("train_results")
# name_weights = training_config.get("weights")[:-5]
# path_tflite = path_save.joinpath(f"{name_weights}.tflite")
# converter = TFLiteConverter()
# converter.from_json_hdf5(
#     path_json=path_save.joinpath(training_config.get("architecture")),
#     path_hdf5=path_save.joinpath(training_config.get("weights")),
#     path_tflite=path_tflite
# )