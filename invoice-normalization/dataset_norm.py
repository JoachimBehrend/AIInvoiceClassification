import sys 
sys.path.append('../inheritance/')
from dataset import *
import ticket_norm

class Dataset_norm(Dataset): 
    def __init__(self,wb,column_names):
        super().__init__(wb,column_names)
    
    def create_ticket_list(self):
        ticket_list = []
        for idx,elm in enumerate(self.wb.iloc):
            description_list = self.column_names[0]

            descriptions = []
            for d_name in description_list:
                descriptions.append(elm[d_name])

            ticket_obj = ticket_norm.Ticket(idx,descriptions,description_list)
            ticket_list.append(ticket_obj)
        self.ticket_list = ticket_list
        return self.ticket_list

