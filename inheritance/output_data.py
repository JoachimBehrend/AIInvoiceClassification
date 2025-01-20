import os
import xlsxwriter as xw
import logger_setup


class Output_data:
    def __init__(self, path, name):
        self.output_path = path
        self.output_name = name
        self.ticket_attributes = None
        self.input_wb = None 
        self.logger = logger_setup.logging.getLogger('output_data') 
        
        if not(os.path.exists(self.output_path)):
            try: 
                os.makedirs(self.output_path)
            except OSError as e: 
                logger.error(e)
                raise

        # remove previous versions
        if os.path.exists(os.path.join(self.output_path, self.output_name)):  
            try: 
                os.remove(os.path.join(self.output_path, self.output_name))  
            except OSError as e:  
                self.logger.error(e)
                raise
        
        self.workbook = xw.Workbook(os.path.join(self.output_path, self.output_name))  
        self.sheet = self.workbook.add_worksheet('sheet_data')  

    
    def Set_input_data_information(self,wb):
        self.input_wb = wb
    
    def Write_row(self,row_info,row):
        col = 0
        for info in row_info:
            self.sheet.write(row,col,info)
            col += 1
    
    def Set_attributes(self,ticket_attributes):
        self.ticket_attributes = ticket_attributes
    
    def Write_header(self,wb):
        c = 0
        for col_info in wb.columns.values.tolist() + self.ticket_attributes:
            self.sheet.write(0,c,col_info)
            c += 1
    
    def Write_tickets(self,wb,tickets,start_row):
        row = start_row
        for ticket in tickets: 
            ticket_information = wb.iloc[ticket.index].values.tolist()
            for a in self.ticket_attributes:
                a_cal = getattr(ticket,a)
                ticket_information.append(a_cal)
            self.Write_row(ticket_information,row)
            row += 1

    def CloseWorkbook(self):
        self.workbook.close()

