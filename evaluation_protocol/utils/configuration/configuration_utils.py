from utils.configuration.yaml_include_loader import IncludeLoader
from omegaconf import OmegaConf
import yaml

def parse_config(config_path):
    """
    Initializes the configuration with contents from the specified file
    :param path: path to the configuration file in json format
    """
    with open(config_path, 'r') as f:
        yaml_object = yaml.load(f, IncludeLoader)

    # Loads the configuration file and converts it to a dictionary
    omegaconf_config = OmegaConf.create(yaml_object, flags={"allow_objects": True}) # Uses the experimental "allow_objects" flag to allow classes and functions to be stored directly in the configuration
    config = OmegaConf.to_container(omegaconf_config, resolve=True)

    return config