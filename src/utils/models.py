
import importlib
import torch.nn as nn

from src.utils.system import fetch_class_from_file, get_kwarg_dict
from src.constants import MODEL_DIR


def fetch_model(name):
    """
    Provide the filename of the model and locate it in the ENCODER_DIR

    Return (Model, train_func, kwarg_dict)
    """
    Model = fetch_class_from_file(
        MODEL_DIR, name.replace('.py', ''), nn.Module)

    module_path = str(MODEL_DIR).replace('/', '.') + '.' + name
    module = importlib.import_module(module_path)

    kwarg_dict = get_kwarg_dict(Model.__init__)
    return Model, module.train, kwarg_dict
