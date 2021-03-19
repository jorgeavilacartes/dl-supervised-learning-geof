"""
Ejecutar desde el directorio ecg-supervised-learning
"""
import json
from pprint import pprint

import os
import json
from pathlib import Path
from tqdm import tqdm
from src.utils import fromJSON, asJSON

from src.model_loader import (
    ModelLoaderTFLITE,
    ModelLoaderJSONHDF5,
)

from src import ECGLoader
from src.utils import (
    fromJSON,
    asJSON,
)

PATH_DATA = Path("data")

# Evaluo ambos para verificar que se obtengan resultados similares
for subset  in ["val","test"]:
    print(subset)
    list_files = fromJSON( PATH_DATA.joinpath(f"list_{subset}.json") )
    labels = fromJSON( PATH_DATA.joinpath("labels.json") )

    # Cargar modelo a usar
    model = ModelLoaderJSONHDF5(dir_model="train_results") # Desde su formato json +  hdf5
    # model = ModelLoaderTFLITE(dir_model="model_training/train_results") # Desde su formato json +  hdf5

    # Realizar predicciones
    results = []
    for filename in tqdm(list_files):
        res = dict(
            filename=str(filename),
            ground_truth=labels.get(filename),
            prediction=model.predict(filename, confidence_bool=True)
        )
        results.append(res)

    path_save=Path("test_results")
    path_save.mkdir(parents=True, exist_ok=True)

    asJSON(data=results, path_save=path_save.joinpath(f"predictions_{subset}.json"), sort_keys=False)