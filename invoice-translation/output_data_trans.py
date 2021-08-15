import sys 
sys.path.append('..')
import output_data 

class Output_data_trans(output_data.Output_data): 
    def __init__(self,path,name):
        super().__init__(path,name)