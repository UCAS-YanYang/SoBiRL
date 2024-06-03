import yaml
from dataclasses import dataclass, asdict

def load_config(config_filepath, args):
    with open(config_filepath, "r") as file:
        config = yaml.safe_load(file)

    for key, value in config.items():
        setattr(args, key, value)
    print("Load the args config:", config_filepath)
    print(args)
    return args