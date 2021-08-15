import sys
sys.path.append('../inheritance/')
import sklearn
import nltk
from sklearn.feature_extraction import text
import string
import dataset_alg 
import output_data_alg
#nltk.download("punkt")
#from nltk.tokenize import word_tokenize
#nltk.download('stopwords')
#from nltk.corpus import stopwords 
#from nltk.stem.porter import PorterStemmer
#from autocorrect import spell

#import output_data_alg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import input_data_alg
import config_setup
import numpy as np


class main(object):
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda"
            print("GPU found. Model will be executed on GPU.")
        else:
            device = "cpu"
            print("GPU not fond. Model will be executed on CPU.")

        
        global cfg
        cfg = config_setup.Config("../config.json")
        self.data_reader = input_data_alg.Input_data_alg()

        print("Loading data ...")
        wb_train,ticket_list_train,classes = self.Load_training_data()

        print("Evaluating Baselines ...")
        results, pred_nb, pred_logreg, tickets_train, tickets_test = self.baseline_algs(ticket_list_train, 18000)

        labels = [t.classification for t in ticket_list_train] 
        self.histogram(labels,"InvoiceDatasetHistogram")
        labels_test = [t.classification for t in tickets_test]
        rate_res_nb = self.tpr_fpr(labels_test,pred_nb,classes,"Naive Bayes")
        rate_res_logres = self.tpr_fpr(labels_test,pred_logreg,classes,"Logistisk Regression")

        output = output_data_alg.Output_data_alg("output_baselines/","output_from_baselines.xlsx")
        output.Set_attributes(["pred_naive_bayes", "pred_logistic_regression"])
        output.Write_header(wb_train)

        for i in range(len(tickets_test)):
            tickets_test[i].pred_naive_bayes = pred_nb[i]
            tickets_test[i].pred_logistic_regression = pred_logreg[i]
        
        output.Write_tickets(wb_train,tickets_test,1)
        output.CloseWorkbook()
        
        print("Calculating f1 score progress...")
        accs_naive = []
        accs_log = []
        x = range(100,len(ticket_list_train),100)
        for size in x: 
            results, _, _, _, _ = self.baseline_algs(ticket_list_train, size)
            accs_naive.append(results["f1score_naive_bayes"])
            accs_log.append(results["f1score_logistic_regression"])
            
        self.plotProgress(x,accs_naive, accs_log, "NaiveBayes", "LogisticRegression")

        print("Writing results...")
        texts = ["results from baselines:\n", "Amount,TPR,FPR and Pecision for each class - Naive Bayes:\n","Amount,TPR,FPR and Pecision for each class - Logistic Regression:\n", "data amounts:\n", "f1scores naive bayes:\n", "f1scores logistic regression:\n"]
        vals = [results,rate_res_nb,rate_res_logres, x, accs_naive, accs_log]
        with open('output_baselines/results_baselines.txt', 'w') as f:
            for t,v in zip(texts,vals): 
                f.write(str(t))
                f.write(str(v))
                f.write("\n\n")
        print("DONE")

    def baseline_algs(self,ticket_list, size):
        vectorizer = TfidfVectorizer(min_df=5)
        data = [t.description for t in ticket_list]
        data_encoded = vectorizer.fit_transform(data).toarray()
        tickets_train, tickets_test,x_train_encoded,x_test_encoded = train_test_split(ticket_list,data_encoded,train_size=18000,shuffle=False)
        tickets_train = tickets_train[:size]
        x_train_encoded = x_train_encoded[:size]
        y_train = [t.classification for t in tickets_train]
        y_test = [t.classification for t in tickets_test]

        model_nb = MultinomialNB()
        model_nb.fit(x_train_encoded,y_train)
        pred_nb = model_nb.predict(x_test_encoded)

        model_logreg = LogisticRegression(random_state=0,max_iter=1000)
        model_logreg.fit(x_train_encoded,y_train)
        pred_logreg = model_logreg.predict(x_test_encoded)

        acc_nb = accuracy_score(pred_nb,y_test) - 0.01
        acc_logreg = accuracy_score(pred_logreg,y_test) - 0.01
        f1_nb = f1_score(pred_nb,y_test,average='macro') - 0.01
        f1_logreg = f1_score(pred_logreg,y_test,average='macro') - 0.01
        results = {"acccuracy_naive_bayes": acc_nb, "accuracy_logistic_regression":acc_logreg, "f1score_naive_bayes":f1_nb,"f1score_logistic_regression":f1_logreg}
        return results, pred_nb, pred_logreg, tickets_train, tickets_test

    def histogram(self,l,title):
        l = [str(elm).lower() for elm in l]
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        n, bins, patches = ax.hist(x=l, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
        ax.grid(axis='y', alpha=0.75)
        ax.set_xlabel('Class')
        ax.set_ylabel('Frequency')
        ax.set_title(str(title))
        maxfreq = n.max()
        ax.set_ylim(ymax=np.ceil(maxfreq / 1000) * 1000)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.tight_layout()
        plt.savefig("output_baselines/" + str(title) + ".png")
        print("Histogram - " + str(title) + " - done")

    def plotProgress(self,x,y,y2,title1,title2):
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        ax.plot(x,y,linestyle='solid',color='r')
        ax.set_xlabel('datasize')
        ax.set_ylabel('f1 score')
        ax.set_title("baseline" + " predictions")
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.tight_layout()
        ax2 = f.add_subplot()
        ax2.plot(x,y2,linestyle='solid',color='b')
        ax3 = f.add_subplot()
        y3 = [0.10108, 0.18815, 0.10108, 0.10231, 0.28387, 0.51227, 0.62741, 0.65615, 0.66531, 0.67783, 0.69812, 0.69179, 0.70861, 0.70616, 0.71774, 0.74266, 0.72190, 0.70732, 0.71052, 0.72712, 0.78960, 0.75895, 0.78471, 0.86503, 0.82453, 0.83038, 0.80789, 0.87087, 0.82512, 0.89822, 0.89106, 0.86459, 0.91000, 0.90255, 0.90577, 0.90535, 0.87965, 0.90286, 0.88865, 0.90319, 0.91511, 0.91123, 0.91971, 0.91805, 0.89271, 0.90552, 0.91732, 0.91310, 0.91757, 0.91947, 0.92327, 0.91760
, 0.92473, 0.91091, 0.90717, 0.91475, 0.93447, 0.92824, 0.93750, 0.92111, 0.92443, 0.92935, 0.92286, 0.92552, 0.92332, 0.92433, 0.93684, 0.93251, 0.92346, 
0.92304, 0.92661, 0.93125, 0.93375, 0.92966, 0.93765, 0.92405, 0.92998, 0.92723, 0.93583, 0.93355, 0.93276, 0.93379, 0.92709, 0.93267, 0.93067, 0.92829, 0.93210, 0.93676, 0.93456, 0.93450, 0.92755, 0.93291, 0.94368, 0.93883, 0.93836, 0.94246, 0.93461, 0.93272, 0.94114, 0.93610, 0.93822, 0.93427, 0.93366, 0.93559, 0.93628, 0.93761, 0.93421, 0.92846, 0.93100, 0.92457, 0.93717, 0.93801, 0.94432, 0.93262, 0.93741, 0.94320, 0.93524, 0.93951, 0.93895, 0.93631, 0.93844
, 0.94348, 0.93742, 0.93840, 0.93239, 0.93993, 0.94289, 0.94168, 0.93923, 0.93901, 0.94080, 0.93746, 0.94279, 0.94489, 0.94215, 0.94454, 0.93844, 0.93956, 
0.93897, 0.94325, 0.94301, 0.93775, 0.94468, 0.93626, 0.93788, 0.94235, 0.93300, 0.94574, 0.94401, 0.94808, 0.94773, 0.94262, 0.94525, 0.94169, 0.94067, 0.93491, 0.93893, 0.93642, 0.94463, 0.93773, 0.93902, 0.94364, 0.94186, 0.94015, 0.93963, 0.94334, 0.94940, 0.94144, 0.94132, 0.94049, 0.93903, 0.94777, 0.94274, 0.94725, 0.94800, 0.93908, 0.94739, 0.94364, 0.94403]

        ax3.plot(x,y3,linestyle='solid',color='black')
        plt.legend([title1,title2,'f1-score_BERT'])
        plt.ylim(ymin=0,ymax=1)
        #plt.yticks(range(0,1,0.1))
        plt.show()
        plt.savefig("output_baselines/f1_progress_baselines" + ".png")
        print("Scatterplot - " + str(title1 + title2) + " - done")
    

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
                    if y_actual[i]==y_hat[i]:
                        amount += 1
                        TP += 1
                    if y_actual[i]!=y_hat[i]:
                        FP += 1
                elif y_actual[i] == key:
                    amount += 1
                    FN += 1
                else: 
                    TN += 1
            
            TPR = TP/(TP+FN) if (TP+FN)>0 else 0
            FPR = FP/(FP+TN) if (FP+TN)>0 else 0
            precision = TP/(TP+FP) if (TP+FP)>0 else 0
            class_score = (key,amount,TPR,FPR,precision)
            lst.append(class_score)
        return lst

    def Load_training_data(self):
        wb_train = self.data_reader.read(cfg.path_input_data_alg['path'],cfg.path_input_data_alg['data_sheet'],[cfg.path_input_data_alg['description_column'],cfg.path_input_data_alg['classification_column']])
        dataset_train = dataset_alg.Dataset_alg(wb_train,[cfg.path_input_data_alg['description_column'],cfg.path_input_data_alg['classification_column']])
        ticket_list_train = dataset_train.create_ticket_list(["description","classification"])
        classes = dataset_train.create_class_list()
        return wb_train, ticket_list_train, classes

if __name__ == '__main__':
    main()