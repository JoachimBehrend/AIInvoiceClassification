import sys 
sys.path.append('..')
import output_data
 
class Output_data_norm(output_data.Output_data): 
    def __init__(self,path,name):
        super().__init__(path,name)
    
    def Write_tickets(self,wb,tickets,start_row):
        row = start_row
        for ticket in tickets: 
            ticket_information = wb.iloc[ticket.index].values.tolist()
            a_cal = getattr(ticket,"descriptions_processed")
            for a in a_cal:
                ticket_information.append(a)
            for a in self.ticket_attributes:
                a_cal = getattr(ticket,a)
                ticket_information.append(a_cal)
            self.Write_row(ticket_information,row)
            row += 1
