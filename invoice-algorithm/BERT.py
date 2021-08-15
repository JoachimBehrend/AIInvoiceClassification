import sys 
sys.path.append('..')
import config_setup
from simpletransformers.classification import MultiLabelClassificationModel
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import shutil
import numpy as np

global cfg, cfg_alg
cfg = config_setup.Config("../config.json")

if cfg.algorithm_bool['pretrained_model_eval_dataset2']:
    path = os.path.join("../skynet-algorithm/",cfg.model_paths['model_load_path'])
    cfg_alg = config_setup.Config(os.path.join(path,"model_args.json"))


class BERT(object):
    """docstring for BERT."""
    # TODO: Store trained classifier as file and load
    # TODO: Implement `sklearn.metrics.classification_report` instead of using `predict_proba`

    def __init__(self,classes):
        super(BERT, self).__init__()

        if cfg.algorithm_bool['pretrained_model_eval_dataset2']:
            self.classes = cfg_alg.labels_list
        else:  
            self.classes = classes

        self.classifier = None
        self.data = None
        print("Setting up model")
        if cfg.algorithm_bool['pretrained_model_eval_dataset2']:
            self.model = MultiLabelClassificationModel('roberta', cfg.model_paths['model_load_path'] ,use_cuda=True,num_labels=len(self.classes),
                args={'labels_list':self.classes,'train_batch_size':2, 'output_dir':"./output from model", 'save_steps':2000,'gradient_accumulation_steps':16, 'learning_rate': 3e-5, 'num_train_epochs': 1, 'max_seq_length': 512})
            print("Using exisiting pretrained model")
        else:
            self.model = MultiLabelClassificationModel('roberta', 'roberta-base' ,use_cuda=True,num_labels=len(self.classes),
                args={'labels_list':self.classes,'train_batch_size':2, 'output_dir':cfg.model_paths['model_save_path'],'overwrite_output_dir':True, 'save_steps':1000,'gradient_accumulation_steps':16, 'learning_rate': 3e-5, 'num_train_epochs': 2, 'max_seq_length': 512})
            print("Creating new model")

        print("Setting up model - Done")

        self.I = None
        self.X = None
        self.Y = None

        self.I_train = None
        self.I_test = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        self.Y_pred = None
        self.Y_proba = None

        self.df_train = None
        self.df_test = None

    def Split_dataset_to_train_test(self,should_shuffle, test_size, rand_state):
        self.I_train, self.I_test, self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.I, self.X, self.Y,
        shuffle=should_shuffle,test_size=test_size, random_state=rand_state)

    def Convert_training_data_to_model_format(self):
        labels_train = self.Create_labels(self.Y_train)
        train_data = {'index': self.I_train,'text': self.X_train,'class': self.Y_train,'labels':labels_train}
        self.df_train = pd.DataFrame(train_data, columns = ['index','text','class','labels'])

    def Convert_testing_data_to_model_format(self):
      test_data = {'index': self.I_test,'text': self.X_test}
      self.df_test = pd.DataFrame(test_data, columns = ['index','text'])

    def Dataset_to_train(self):
        self.I_train = self.I
        self.X_train = self.X
        self.Y_train = self.Y
        self.Convert_training_data_to_model_format()

    def Dataset_to_test(self):
        self.I_test = self.I
        self.X_test = self.X
        self.Y_test = self.Y
        self.Convert_testing_data_to_model_format()

    def Dataset_to_train_test(self,should_shuffle,test_size,rand_state):
        self.Split_dataset_to_train_test(should_shuffle,test_size,rand_state)
        self.Convert_training_data_to_model_format()

        labels_test = self.Create_labels(self.Y_test)
        test_data = {'index': self.I_test,'text': self.X_test,'class':self.Y_test,'labels':labels_test}
        self.df_test = pd.DataFrame(test_data, columns = ['index','text','class','labels'])

    def Tickets_to_arrays(self,arg):
        self.X = []
        self.Y = []
        self.I = []
        for ticket_obj in arg:
            self.X.append(ticket_obj.description)
            if ticket_obj.classification == None: 
                self.Y.append("")
            else:
                self.Y.append(ticket_obj.classification)
            self.I.append(ticket_obj.index)
            

    def Train(self):
        if os.path.exists("./outputs"):
            shutil.rmtree("./outputs")
        
        self.model.train_model(self.df_train)

    def Create_labels(self,class_array):
        labels_array = []
        for c in class_array:
            ticket_label = []
            for idx,c_valid in enumerate(self.classes):
                ticket_label.append(0)
                if str(str(c)) == str(c_valid):
                    ticket_label[-1] = 1
            labels_array.append(ticket_label)
        return labels_array

    def Predict(self):
        preds, outputs = self.model.predict(self.X_test)
        results = []
        for p in outputs:
            label = np.argmax(p,axis=0)
            classification = self.classes[label]
            results.append(classification)
        self.Y_pred = results
        self.Y_proba = outputs

        return preds, outputs, results

    def info_get_classes(self):
        return self.classes
