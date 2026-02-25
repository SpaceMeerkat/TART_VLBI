import json
import numpy as np

def to_json_safe(obj):
    """ 
    Prepare a general nested dictionary into a serialisable format
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    else:
        return obj

def save_to_json(data, path, filename):
    """
    Save data as a JSON file to disk
    """
    with open(f"{path}/{filename}", "w") as f:
        json.dump(data, f, indent=2)
    return "JSON saved to disk."

def load_from_json(path, filename):
    """
    Load JSON data from disk into python readable format
    """
    with open(f"{path}/{filename}", "r") as f:
        loaded = json.load(f)
    return loaded
    