import yaml

with open("config.yaml") as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)
