"""
Ejecutar desde el directorio ecg-supervised-learning
"""
import json
from pprint import pprint

import os
import json
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict

from src.utils import fromJSON, asJSON

from src.model_loader import (
    ModelLoaderTFLITE,
    ModelLoaderJSONHDF5,
)

from src.utils import (
    fromJSON,
    asJSON,
)

PATH_DATA = Path("data")
PATH_SAVE=Path("test_results")
PATH_SAVE.mkdir(parents=True, exist_ok=True)

# Evaluo ambos para verificar que se obtengan resultados similares
for subset  in ["val","test"]:
    print(subset)
    list_files = fromJSON( PATH_DATA.joinpath(f"list_{subset}.json") )
    labels = fromJSON( PATH_DATA.joinpath("labels.json") )

    # Cargar modelo a usar
    model = ModelLoaderJSONHDF5(dir_model="train_results") # Desde su formato json +  hdf5
    # model = ModelLoaderTFLITE(dir_model="model_training/train_results") # Desde su formato tflite
    # Realizar predicciones
    results = OrderedDict()
    for filename in tqdm(list_files):
        res_pred = model.predict(filename, confidence_bool=True)
        res = dict(
            ground_truth=labels.get(filename),
            prediction=res_pred.get("decoded_output"),
            prediction_model=res_pred.get("confidence_model")
        )
        results.update({filename: res})

    asJSON(data=results, path_save=PATH_SAVE.joinpath(f"predictions_{subset}.json"), sort_keys=False)