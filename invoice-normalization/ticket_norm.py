import re 
import application_list_handler
import enchant

import logging
logger = logging.getLogger(__name__)

class Ticket_handler_norm(object):
    def __init__(self):
        self.application_list_handler = None 
        self.dictionary = None 

    def Process_tickets(self,ticket_list):
        nr = 0
        for ticket in ticket_list:
            if nr % 1000 == 0:
                logger.info("Tickets normalized: {}".format(nr))
            ticket.description_processed = self.application_list_handler.Replace_applications(ticket.descriptions)
            desc_list = self.__process_description(ticket)
            ticket.right_language = self.__process_language(desc_list)

            description_processed = ""
            length = len(desc_list)
            for i in range(length):
                description_processed = description_processed + desc_list[i]
                if i < length - 1:
                    description_processed = description_processed + " ; "

            ticket.descriptions_processed = desc_list
            ticket.description_processed = description_processed

            if not(ticket.successfully_processed):
                ticket.normalization_state = "incorrect"
            elif not(ticket.right_language):
                ticket.normalization_state = "normalized - wrong language"
            else:
                ticket.normalization_state = "normalized"
            nr += 1


    def __process_language(self,texts):
        text = ""
        for t in texts:
            text = text + " " + str(t)

        correct_count_eng = 0
        words = str(text).split()
        amount_of_words = len(words)
        for w in words: 
            if self.dictionary.check(w):
                correct_count_eng += 1

        if amount_of_words == 0:
            return False 
        if correct_count_eng/amount_of_words < 0.6:
            return False 
        
        # Checks for chinese characters
        try:
            text.encode(encoding='utf-8').decode('ascii')  
        except UnicodeDecodeError:  
            return False 
        return True 


    def __process_description(self,ticket):
        list_processed_text = []
        for text in ticket.descriptions:
            proc_text = str(text)

            # Checking for empty string
            if proc_text == "":
                ticket.successfully_processed = False 
                ticket.ticket_error = "Description is empty"  
                return proc_text
            

            # Find and replace URLS, Domains, Ip-addresses and Network drives 
            regex_url = r"\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
            urls = re.findall(regex_url,proc_text)
            for i,url in enumerate(urls):
                ticket.url_string += url[0]
                ticket.url_string += ";"
            proc_text = re.sub(regex_url,"__url_string__",proc_text)

            regex_domain = r'\\\\[\w-]+[\.]+[\w\.-]+'
            domains = re.findall(regex_domain,proc_text)
            for i,domain in enumerate(domains):
                ticket.domain_string += domain
                ticket.domain_string += ";"

            proc_text = re.sub(regex_domain,"__domain_string__",proc_text)

            regex_ip_address = r'[0-9]+(?:\.[0-9]+){3}'
            ip_addresses = re.findall(regex_ip_address,proc_text)
            for i,ip_addr in enumerate(ip_addresses):
                ticket.ip_address_string += str(ip_addr)
                ticket.ip_address_string += ";"

            proc_text = re.sub(regex_ip_address,"__ip_address__",proc_text)

            regex_network_drive = r'([\w]+:\\\\([\w-]+[\.]?[\w\.-]+[\\]?)+)'
            network_drives = re.findall(regex_network_drive,proc_text)
            for i,net in enumerate(network_drives):
                ticket.network_drive_string += net[0]
                ticket.network_drive_string += ";"
            proc_text = re.sub(regex_network_drive,"__network_drive_string__",proc_text)
            
            # Split proc_text with our_pattern
            regex_pattern = r'__[\w\.-]+__'
            replacements = re.findall(regex_pattern, proc_text ) 

            proc_text_list = re.split(regex_pattern,proc_text)
            text_sum = ""
            for i,text in enumerate(proc_text_list):
                # Description text processing 
                text = re.sub( r'@', '', text) 
                text = re.sub( r'\|', '', text) 
                text = re.sub( r'\W|_', ' ', text) 
                text = re.sub( r'\ (\ )+', ' ', text)
                text = re.sub( r'-', ' ', text) 
                text = re.sub( r'\s(\s)*', ' ', text) 
                regex = r'\b[\d]+\b'
                text = re.sub(regex,' ',text) 
                text_sum = text_sum + text
                if i < len(replacements):
                    text_sum = text_sum + replacements[i]
            proc_text = text_sum.strip()
            list_processed_text.append(proc_text)
        
        if ticket.url_string[-1] == ';':
            ticket.url_string = ticket.url_string[:-1]
        if ticket.domain_string[-1] == ';':
            ticket.domain_string = ticket.domain_string[:-1]
        if ticket.ip_address_string[-1] == ';':
            ticket.ip_address_string = ticket.ip_address_string[:-1]
        if ticket.network_drive_string[-1] == ';':
            ticket.network_drive_string = ticket.network_drive_string[:-1]

        return list_processed_text


    def Set_translator(self,language):
        self.dictionary = enchant.Dict(language)


class Ticket(object):
    def __init__(self,index,descriptions,description_names):
        self.index = index
        self.names = description_names
        self.descriptions = descriptions
        self.descriptions_processed = None 
        self.description_processed = None 
        self.ticket_error = None
        self.successfully_processed = True 
        self.right_language = None 
        self.normalization_state = None 
        self.url_string = " "
        self.domain_string= " "
        self.ip_address_string = " "
        self.network_drive_string = " "
