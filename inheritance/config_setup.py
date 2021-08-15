import json
import os

class Config(object):
    """docstring for Config."""
    def __init__(self,file):
        super(Config, self).__init__()

        config_path = os.path.abspath(__file__)
        config_file = config_path.replace("config_setup.py", file)

        with open(config_file) as json_data:
            self.__dict__ = json.load(json_data)
            json_data.close()

