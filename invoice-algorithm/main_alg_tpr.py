import sys
sys.path.append('../inheritance/')
import BERT
import input_data_alg
import output_data_alg
import ticket_alg
import config_setup
import dataset_alg 
import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import collections

import matplotlib.pyplot as plt

import torch
import gc

class main_alg(object):
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda"
            print("GPU found. Model will be executed on GPU.")
        else:
            device = "cpu"
            print("GPU not fond. Model will be executed on CPU.")
        
        # Initialize 
        self.data_reader = input_data_alg.Input_data_alg()
        global cfg
        cfg = config_setup.Config("../config.json")

        if cfg.algorithm_bool['pretrained_model_eval_dataset2']:
            alg = BERT.BERT(None)
            wb_test, ticket_list_test = self.Load_evaluation_data()
            alg.Tickets_to_arrays(ticket_list_test)
            alg.Dataset_to_test()
            alg.Predict()

            self.output_xlsx(wb_test,alg,ticket_list_test)

        elif cfg.algorithm_bool['train_dataset1_eval_dataset2']:
            wb_train,ticket_list_train,classes = self.Load_training_data()
            ticket_list_train = ticket_list_train[:200]
            wb_test, ticket_list_test = self.Load_evaluation_data()
            alg = BERT.BERT(classes)
            alg.Tickets_to_arrays(ticket_list_train)
            alg.Dataset_to_train()
            alg.Train()

            alg.Tickets_to_arrays(ticket_list_test)
            alg.Dataset_to_test()
            alg.Predict()

            self.scores(alg)
            self.output_xlsx(wb_test,alg,ticket_list_test)

        elif cfg.algorithm_bool['train_dataset1_eval_dataset1']:
            wb_train,ticket_list_train,classes = self.Load_training_data()
            alg = BERT.BERT(classes)
            ticket_list_train = ticket_list_train[:200]
            alg.Tickets_to_arrays(ticket_list_train)
            alg.Dataset_to_train_test(False,0.25,42)
            test_labels = alg.Y_test
            class_freq = dict(collections.Counter(test_labels))
            print("true test-label frequency")
            print(class_freq)
            print("READY TO TRAIN - dataamount: " + str(len(ticket_list_train)))
            alg.Train()
            alg.Predict()
            self.scores(alg,classes)
            self.output_xlsx(wb_train,alg,ticket_list_train)


            ##self.plotProgress(a,LRAPs,"BERT predicitions - LRAP","LRAP")
            #self.plotProgress(a,f1_scores,"BERT predicitions - f1 score","f1 score")
            #self.plotProgress(a,accuracies,"BERT predicitions - accuracy","accuracy")
            ##self.plotProgress(a,losses,"BERT - loss","loss")
            
    
        elif cfg.algorithm_bool['train_dataset1_save_model']:
            wb_train,ticket_list_train,classes = self.Load_training_data()
            ticket_list_train = ticket_list_train[:200]
            alg = BERT.BERT(classes)
            alg.Tickets_to_arrays(ticket_list_train)
            alg.Dataset_to_train()
            alg.Train()
            print("Done training, model saved")

    def output_xlsx(self,wb,alg,ticket_list):
        output = output_data_alg.Output_data_alg("output/","output_from_algorithm.xlsx")
        output.Set_attributes(["prediction"])
        output.Write_header(wb)

        ticket_list_test = []
        for i in alg.I_test:
            ticket_list_test.append(ticket_list[i])

        for i,p in enumerate(alg.Y_pred):
            ticket_list_test[i].prediction = p
        
        output.Write_tickets(wb,ticket_list,1)
        output.CloseWorkbook()

    def scores(self,alg,classes):
        preds_freq = dict(collections.Counter(alg.Y_pred))
        print("predicted label frequency")
        print(preds_freq)

        f1 = f1_score(alg.Y_pred,alg.Y_test,average='macro')
        accuracy = accuracy_score(alg.Y_pred,alg.Y_test)
        print(f"accuracy: {accuracy}, f1: {f1}")
        
        self.tpr_fpr(alg.Y_test,alg.Y_pred,classes,"BERT")
        return accuracy, f1
        

    def Train_dataset1_Eval_dataset1(self):
        wb_train,ticket_list_train,classes = self.Load_training_data()
        alg = BERT.BERT(classes)
        alg.Tickets_to_arrays(ticket_list_train)
        alg.Dataset_to_train_test(True,0.25,42)
        print("HERE - READY TO TRAIN - dataamount: " + str(len(ticket_list_train)))
        alg.Train()
        ticket_list_val = []
        for i in alg.I_test:
            ticket_list_val.append(ticket_list_train[i])
        alg = self.Evaluate(alg,wb_train,ticket_list_val)
        classifications = alg.Y_test
        predictions = alg.Y_pred
        f1 = f1_score(predictions,classifications,average='macro')
        accuracy = accuracy_score(predictions,classifications)
        print(f"datasize: {len(ticket_list_train)}, accuracy: {accuracy}, f1: {f1}")
        
        return accuracy, f1
    
    def Train_dataset1_Eval_dataset12(self,datasize):
        wb_train,ticket_list_train,classes = self.Load_training_data()
        alg = BERT.BERT(classes)
        ticket_list_train = ticket_list_train[:datasize]
        alg.Tickets_to_arrays(ticket_list_train)
        alg.Dataset_to_train_test(False,0.25,42)
        print("HERE - READY TO TRAIN - dataamount: " + str(datasize))
        alg.Train()
        # for eval
        #indexes = [t.index for t in eval_ticket_list]
        #classifications = [t.classification for t in eval_ticket_list]
        #descriptions = [t.description for t in eval_ticket_list]
        #labels = alg.Create_labels(classifications)
        #eval_df = pd.DataFrame(zip(descriptions,labels))
        #eval_df.columns = ["text", "labels"]
        #results, model_outputs, wrong_predictions = alg.model.eval_model(eval_df)
        #predictions = []
        #for t in model_outputs:
        #    max_prop = max(t)
        #    idx = np.where(t == max_prop)[0][0]
        #    predictions.append(classes[idx])
        
        ticket_list_val = []
        for i in alg.I_test:
            ticket_list_val.append(ticket_list_train[i])

        alg = self.Evaluate(alg,wb_train,ticket_list_val)
        classifications = []
        for ticket in ticket_list_val:
            classifications.append(ticket.classification)
        predictions = alg.Y_pred
        f1 = f1_score(predictions,classifications,average='macro')
        accuracy = accuracy_score(predictions,classifications)
        print(f"datasize: {datasize}, accuracy: {accuracy}, f1: {f1}")
        #loss = alg.model.loss
        return accuracy, f1
    
    def Train_dataset1(self):
        wb_train,ticket_list_train,classes = self.Load_training_data()
        alg = BERT.BERT(classes)
        alg.Tickets_to_arrays(ticket_list_train)
        alg.Dataset_to_train()
        alg.Train()
        return alg

    def Pretrained_model_Eval_dataset2(self):
        wb_test,ticket_list_test = self.Load_evaluation_data()
        alg = BERT.BERT(None)
        self.Evaluate(alg,wb_test,ticket_list_test)



    def Load_training_data(self):
        wb_train = self.data_reader.read(cfg.path_input_data_alg['path'],cfg.path_input_data_alg['data_sheet'],[cfg.path_input_data_alg['description_column'],cfg.path_input_data_alg['classification_column']])
        dataset_train = dataset_alg.Dataset_alg(wb_train,[cfg.path_input_data_alg['description_column'],cfg.path_input_data_alg['classification_column']])
        ticket_list_train = dataset_train.create_ticket_list(["description","classification"])
        classes = dataset_train.create_class_list()
        return wb_train, ticket_list_train, classes

    def Load_evaluation_data(self):
        wb_test = self.data_reader.read(cfg.path_input_data_alg_test['path'],cfg.path_input_data_alg_test['data_sheet'],[cfg.path_input_data_alg['description_column']])
        dataset_test = dataset_alg.Dataset_alg(wb_test,[cfg.path_input_data_alg_test['description_column']])
        ticket_list_test = dataset_test.create_ticket_list(["description"])
        return wb_test, ticket_list_test

    def Evaluate(self,alg,wb,ticket_list):
        chunk_size = 40000
        ticket_chunks = [ticket_list[i:i + chunk_size] for i in range(0, len(ticket_list), chunk_size)]
        print("Amount of evaluation iterations - {}; Tickets per. iteration - {}".format(str(len(ticket_chunks)),str(chunk_size)))
        for i,ticket_c in enumerate(ticket_chunks):
            alg.Tickets_to_arrays(ticket_c)
            alg.Dataset_to_test()
            print("PREDICTING ITERATION - {} out of {}".format(str(i + 1),str(len(ticket_chunks))))
            alg.Predict()
            starting_row = i*chunk_size + 1
            if i == 0: 
                output = output_data_alg.Output_data_alg("output/","output_from_algorithm.xlsx")
                output.Set_attributes(["prediction"])
                output.Write_header(wb)
            ticket_list_output = []
            for ticket in ticket_c:
                ticket.prediction = alg.Y_pred[i]
            output.Write_tickets(wb,ticket_c,starting_row)
        output.CloseWorkbook()
        return alg

    def plotProgress(self,x,y,title, yaxis):
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        ax.plot(x,y,linestyle='solid')
        ax.set_xlabel('datasize')
        ax.set_ylabel(yaxis)
        ax.set_title(str(title) + " plot")
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.tight_layout()
        plt.savefig(str(title) + "_scatter" + ".png")
        print("Scatterplot - " + str(title) + " - done")
    
    def tpr_fpr(self,y_actual, y_hat, classes,title):
        lst = []
        for key in classes: 
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            amount = 0
            for i in range(len(y_hat)): 
                if y_hat[i] == key:
                    amount += 1
                    if y_actual[i]==y_hat[i]:
                        TP += 1
                    if y_actual[i]!=y_hat[i]:
                        FP += 1
                elif y_actual[i] == key:
                    FN += 1
                else: 
                    TN += 1
            
            TPR = TP/(TP+FN) if (TP+FN)>0 else 0
            FPR = FP/(FP+TN) if (FP+TN)>0 else 0
            precision = TP/(TP+FP) if (TP+FP)>0 else 0
            class_score = (key,amount,TPR,FPR,precision)
            lst.append(class_score)
        print(f"TPR, FPR for {title}")
        print(lst)
if __name__ == '__main__':
    main_alg()

