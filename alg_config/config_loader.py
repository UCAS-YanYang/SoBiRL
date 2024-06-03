import yaml
from dataclasses import dataclass, asdict

def load_config(config_filepath, args):
    # 从配置文件加载参数
    with open(config_filepath, "r") as file:
        config = yaml.safe_load(file)

    # 将配置文件中的参数应用于默认参数
    for key, value in config.items():
        setattr(args, key, value)
    print("Load the args config:", config_filepath)
    print(args)
    return args