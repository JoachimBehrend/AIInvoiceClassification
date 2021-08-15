import sys 
sys.path.append('..')
import config_setup
import os
import yaml
import logging.config
import logging

# |  Level    | Numeric Value  |      Function       |                               Used to                               |
# |-----------|----------------|---------------------|---------------------------------------------------------------------|
# | CRITICAL  |            50  | logging.critical()  | Show a serious error, the program may be unable to continue running |
# | ERROR     |            40  | logging.error()     | Show a more serious problem                                         |
# | WARNING   |            30  | logging.warning()   | Indicate something unexpected happened, or could happen             |
# | INFO      |            20  | logging.info()      | Confirm that things are working as expected                         |
# | DEBUG     |            10  | logging.debug()     | Diagnose problems, show detailed information                        |

cfg = config_setup.Config("../config.json")

def setup_logging(default_path=cfg.path_logger['logger'], root_path=cfg.path_logger['app'], default_level=logging.INFO, env_key='LOG_CFG'):
    """
    | **@author:** Prathyush SP
    | Logging Setup
    """
    full_path = os.path.join(root_path, default_path)

    value = os.getenv(env_key, None)
    if value:
        full_path = value
    if os.path.exists(full_path):
        with open(full_path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print(e)
                print('Error in Logging Configuration. Using default configs')
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print('Failed to load configuration file. Using default configs')
