
import dataset
import ticket_trans
 
class Dataset_norm(dataset.Dataset): 
    def __init__(self,wb,column_names):
        super().__init__(wb,column_names)
    
    def create_ticket_list(self):
        ticket_list = []
        for idx,elm in enumerate(self.wb.iloc):
            description = elm[self.column_names[0]]
            state = elm[self.column_names[1]]
            ticket_obj = ticket_trans.Ticket(idx,description,state)
            ticket_list.append(ticket_obj)
        self.ticket_list = ticket_list
        return self.ticket_list
