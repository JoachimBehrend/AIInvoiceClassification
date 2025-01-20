import pandas 
import re
 
class Application_list_handler(object):
    def __init__(self,wb,input_col,output_col):
        self.application_list = []
        for elm in wb.iloc:
            application_name = elm[input_col]
            replacement = "__{}__".format(str(elm[output_col]))
            application = Application(application_name,replacement)
            self.application_list.append(application)

    def Replace_applications(self,txts):
        res = []
        for txt in txts:
            proc_text = str(txt).lower()
            # Check text for __patterns__
            proc_text = re.sub("__", ' ',proc_text)

            for i, application in enumerate(self.application_list):
                regex = "(\s+|\\b)(" + application.application_name + ")(\d+|\\b)"
                proc_text = re.sub(regex," " + application.replacement,proc_text.lower()) 

            res.append(proc_text)
        return res 

class Application(object):
    def __init__(self,name,replacement):
        self.application_name = str(name).lower()
        self.replacement = str(replacement).lower()
