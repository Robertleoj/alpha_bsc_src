import json

class Config:

    def __init__(self):
        self.config = None
        self._cpp_conf = None

    def __getitem__(self, key):
        return self.config[key]

    def cpp_conf(self, key):
        return self._cpp_conf[key]

    def cpp_conf_has_key(self, key):
        return key in self._cpp_conf

    def initialize(self, fpath):
        with open(fpath, 'r') as f:
            self.config = json.load(f)

        with open("cpp_hyperparameters.json", 'r') as f:
            self._cpp_conf = json.load(f)


config = Config()

__all__ = ['config']