import sys 
sys.path.append('..')
from dataset import *
import ticket_alg
import re

class Dataset_alg(Dataset): 
    def __init__(self,wb,column_names):
        super().__init__(wb,column_names)
    
    def create_ticket_list(self,attributes):
        ticket_list = []
        for idx,elm in enumerate(self.wb.iloc):
            description = elm[self.column_names[0]]
            description = re.sub(';',' ',description).lower()
            ticket_obj = ticket_alg.Ticket(idx,description)
            for i,a in enumerate(attributes):
                setattr(ticket_obj,a,str(elm[self.column_names[i]]).lower()) 
            ticket_list.append(ticket_obj)
        self.ticket_list = ticket_list
        return self.ticket_list

    def create_class_list(self):
        classes = {}
        for c in self.wb[self.column_names[1]]: 
           classes[c.lower()] = "" 
        c = []
        for key in classes:
            c.append(key)
        return c