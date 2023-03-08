import yaml

with open("config.yaml", encoding='utf8') as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)


import collections.abc
import collections

collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.Sequence = collections.abc.Sequence
collections.MutableMapping = collections.abc.MutableMapping
