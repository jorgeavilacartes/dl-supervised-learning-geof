# Deep Learning for Image Classification

Deep Learning Workflow for Image Classification with tensorflow and keras.

`src/` contains all the codes to train and test models. 

`tools/` contains two classes:
- `DatetimeOCR`: to get datetime from images
- `VideoInspector`: to get images from video

### Description of the code

To **add architectures**, create a new file following the `src/models/NAIVE.py` script:
- Define INPUTS at line 25
- Create an architecture inside the `get_model()` method

To **add preprocessing**, create a function inside `src/preprocessing.py` using `@register_in_pipeline decorator`. 
*For more details in how to use the Pipeline class read the examples [here](https://github.com/jorgeavilacartes/python-dl-tools)*