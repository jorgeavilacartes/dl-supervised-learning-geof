import json
from pprint import pprint

import glob
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

# Acceder a datos de diciembre

# Evaluo ambos para verificar que se obtengan resultados similares
subset  = "dec2020"
list_files = glob.glob("/mnt/atrys/ITMS/ECG/CL/Files/xml-files-dec2020/lib/temp/*.xml")
#labels = fromJSON( path_data.joinpath("labels.json") )

# Cargar modelo a usar
model = ModelLoaderJSONHDF5(dir_model="model_training/train_results") # Desde su formato json +  hdf5

# Realizar predicciones
results = []
for filename in tqdm(list_files):
    res = dict(
        filename=str(filename),
        # ground_truth=labels.get(filename),
        prediction=model.predict(filename, confidence_bool=True)
    )
    results.append(res)

path_save=Path("model_training/test_results")
path_save.mkdir(parents=True, exist_ok=True)

asJSON(data=results, path_save=path_save.joinpath(f"predictions_{subset}.json"), sort_keys=False)