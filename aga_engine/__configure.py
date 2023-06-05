import os
import yaml
import re
from types import SimpleNamespace


CONFIG_PATH = './aga_engine/configs'
CONFIG_NOT_FOUND_ERROR = 'engine configure file not found'


class IterableSimpleNamespace(SimpleNamespace):
    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return '\n'.join(f'{k}={v}' for k, v in vars(self).items())

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)


def __load_yaml(yaml_path):
    with open(yaml_path, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
        return yaml.safe_load(s)


def load_config(mode, logger):
    config_path = os.path.join(CONFIG_PATH, mode + '.yaml')
    assert os.path.isfile(config_path), logger.error(CONFIG_NOT_FOUND_ERROR)
    
    config_dict = __load_yaml(config_path)
    for k, v in config_dict.items():
        if isinstance(v, str) and v.lower() == 'none':
            config_dict[k] = None
    # cfg_keys = cfg_dict.keys()
    config = IterableSimpleNamespace(**config_dict)
    return config


if __name__ == '__main__':
    import logging
    LOG_FORMAT = '[%(levelname)s] %(asctime)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    opt = load_config('debug', logging)
    print(opt.BACKBONE)