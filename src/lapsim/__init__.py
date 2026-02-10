from . import render, eval
from lapsim.models.ls1 import LS1
from lapsim.models.model_bridge import ModelBridge


# In future be smart about this to load diretly via a directory name
def get_model(model_name) -> ModelBridge:
    if model_name == "ls1":
        return LS1()
