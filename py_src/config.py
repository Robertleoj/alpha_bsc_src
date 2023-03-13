import json

class Config:

    def __init__(self):
        self.config = None

    def __getitem__(self, key):
        return self.config[key]

    def initialize(self, fpath):
        with open(fpath, 'r') as f:
            self.config = json.load(f)


config = Config()

__all__ = ['config']