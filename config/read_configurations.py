"""code to read the config file"""
import os

try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    print("YAML Module not found.")

"""Local terminal path"""
"""pycharm path"""
config_path_hydro_temp = "config/config_hydro_temp.yaml"
yaml = YAML(typ="safe")
path_hydro_temp = os.path.join(os.path.dirname(__file__), "config_hydro_temp.yaml")
stream_hydro_temp = open(path_hydro_temp, "r")
config_hydro_temp = yaml.load(stream_hydro_temp)

