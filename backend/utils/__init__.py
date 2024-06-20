from .yaml_parser import YMLParser
import os
from .logger import Logger

utils_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(utils_path,"../")
config_path = os.path.join(root_path,"config.yaml")
parsed_yaml = YMLParser(config_path)
variables = parsed_yaml.get_all()
set_variable = parsed_yaml.set
variable_names = parsed_yaml.available_keys()
run_mode = variables["run_mode"]
refreshToken = os.environ.get("REFRESH_TOKEN")

def relative_path(path):
    return root_path + path