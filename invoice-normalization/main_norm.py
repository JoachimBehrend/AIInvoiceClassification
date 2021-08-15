import sys 
sys.path.append('../inheritance/')
import input_data_norm 
import output_data_norm
import dataset_norm 
import ticket_norm
import application_list_handler
import config_setup


class main_norm(object):
    def __init__(self):
        cfg = config_setup.Config("../config.json")

        # Initialize 
        data_reader = input_data_norm.Input_data_norm()
        
        ticket_handler = ticket_norm.Ticket_handler_norm()
        wb_application_list = data_reader.read(cfg.path_input_data_norm['path'],cfg.path_input_data_norm['application_list_sheet'],[cfg.path_input_data_norm['application_col_input'],cfg.path_input_data_norm['application_col_output']])
        ticket_handler.application_list_handler = application_list_handler.Application_list_handler(wb_application_list,
            cfg.path_input_data_norm['application_col_input'],cfg.path_input_data_norm['application_col_output'])
        ticket_handler.Set_translator(cfg.language_check['expected_language'])
        
        # Normalize information
        description_columns = cfg.path_input_data_norm['description_column']
        wb_data = data_reader.read(cfg.path_input_data_norm['path'],cfg.path_input_data_norm['data_sheet'],cfg.path_input_data_norm['description_column'])
        dataset = dataset_norm.Dataset_norm(wb_data,[cfg.path_input_data_norm['description_column']])
        ticket_list = dataset.create_ticket_list()
        ticket_handler.Process_tickets(ticket_list)
        self.Create_output_norm(wb_data,ticket_list)

        # Write Output

    def Create_output_norm(self,wb,tickets):
        columns = tickets[0].names
        columns_processed = []
        for c in columns:
            columns_processed.append(c + "_processed")

        output = output_data_norm.Output_data_norm("./output","output_from_normalization.xlsx")
        attributes = ["description_processed","normalization_state","ticket_error","url_string","domain_string","ip_address_string","network_drive_string"]
        columns_processed = columns_processed + attributes
        output.Set_attributes(columns_processed)
        output.Write_header(wb)
        output.Set_attributes(attributes)
        output.Write_tickets(wb,tickets,1)
        output.CloseWorkbook()

        
main_norm()
