import yaml
import importlib

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file) 
        return data
    
def get_class(model):
    model_class = globals()[model.split('.')[-1]]
    model = model_class()
    return model

def get_class_Scaler(scaler):
    module_name, class_name = scaler.rsplit(".", 1)
    print(module_name, class_name)
    
    module = importlib.import_module(module_name)
    print(module)

    model_class = getattr(module, class_name)
    print(model_class)
    return model_class() 