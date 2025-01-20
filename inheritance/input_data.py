import logger_setup  
import pandas as pd
import sys 
import openpyxl


class Input_data(object):
    def __init__(self):
        self.logger = logger_setup.logging.getLogger('input_data') 

    def read(self,path,sheet,column_names):
        wb = pd.read_excel(path,sheet_name=sheet,engine='openpyxl')  
        wb = wb.fillna(0) 
        try:
            wb.drop(columns=[""])
        except: 
            wb = wb
        
        # Checks that the desired columns are defined in the dataset
        for desired_col in column_names: 
            desired_col = str(desired_col)
            column_data = None
            try: 
                column_data = wb[desired_col]
            except: 
                self.logger.error("'{}' not found, path: '{}', sheet: {}".format(desired_col,path,sheet))
                sys.exit()
        return wb

    












