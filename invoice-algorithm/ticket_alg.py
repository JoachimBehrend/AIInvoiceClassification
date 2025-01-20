import re 

import logging
logger = logging.getLogger(__name__)

 
class Ticket(object):
    def __init__(self,index,description):
        self.index = index
        self.description = description
        self.classification = None
        self.prediction = None 
        self.pred_naive_bayes = None
        self.pred_logistic_regression = None
