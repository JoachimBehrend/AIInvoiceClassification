import sys 
sys.path.append('../inheritance/')
import pandas as pd
from input_data import * 


class Input_data_norm(Input_data):
    def __init__(self):
        super().__init__()
    