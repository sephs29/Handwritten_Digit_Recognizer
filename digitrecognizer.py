#IMPORTS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.utils import np_utils
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd 
import os
import pickle
import cv2

from sorting_contours import sort_contours

class digitrecognizer():
    def get_mnist_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        #28*28 to 784
        num_pixels = X_train.shape[1] * X_train.shape[2]
        X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
        X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

        #keeping the y_train and y_test as digit labels(0to9)
        return (X_train,y_train,X_test,y_test)
    
    def model_list(self):
        models =[]
        models.append(('LogR',LogisticRegression(max_iter=200,solver='saga')))
        models.append(('GNB',GaussianNB()))
        models.append(('MNB',MultinomialNB()))
        models.append(('CNB',ComplementNB()))
        models.append(('KNN',KNeighborsClassifier()))
        models.append(('Dtree',DecisionTreeClassifier()))
        #models.append(('SVM',SVC(gamma='auto',kernel='linear')))
        models.append(('SGD',SGDClassifier()))
        #models.append(('GBoosting',GradientBoostingClassifier()))
        #models.append(('RF',RandomForestClassifier()))
        models.append(('NN_mlp',MLPClassifier(alpha=1e-5, hidden_layer_sizes=100 , random_state=0)))
        #print(models)

        return models

    def pickle_trained_models(self):
        models = self.model_list()
        (X_train,y_train,X_test,y_test) = self.get_mnist_data()

        df = pd.DataFrame(columns=['model','accuracy'])
        outputdir = 'Pickle files'
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        for name,model in models:
            #generating pickle files
            print('training started')
            model.fit(X_train,y_train)
            print('training done.')
            pickle_filename = str(name)+'.pickle'
            f = open(os.path.join(os.getcwd(),outputdir, pickle_filename), "wb")
            f.write(pickle.dumps(model))
            f.close()
            print('Pickle file',pickle_filename,' generated')
    
    def test_models(self):
        models = self.model_list()
        outputdir = 'Pickle files'
        (X_train,y_train,X_test,y_test) = self.get_mnist_data()

        for name,model in models:
            pickle_filename = str(name)+'.pickle'
            filename = os.path.join(os.getcwd(),outputdir, pickle_filename)
            model = pickle.load(open(filename,'rb'))
            y_predict = model.predict(X_test)
            accuracy_score = accuracy_score(y_test,y_predict)
            df = df.append({'model': name, 'accuracy': accuracy}, ignore_index=True)
        
        return df
    
    
    def get_digits(self,filename):#'mytestdata/testcase0.jpeg'
        #fetch image
        img = cv2.imread(filename,1)
        img =cv2.resize(img,None,fx=.15,fy=.15)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #preprocessing to extract boxes
        #thresholding
        img_bin = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.    THRESH_BINARY_INV,11,1)#3

        #extract boxes
        #kernel_length =int( np.array(img).shape[0]//12)#51
        kernel_length=50
        verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

        #vertical lines
        img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=1)
        verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=7)

        #horizontal lines
        img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=1)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=6)

        #boxes
        alpha,beta = 200,200
        img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0)
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)#5

        #contouring
        contours,_ = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #sort extracted boxes
        (contours, boundingBoxes) = sort_contours(contours,method= "left-to-right")



        #preprocessing to extract digits
        img_no= cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,    11,16)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        img_no = cv2.erode(~img_no, kernel, iterations=3)
        #inverse image
        img_no = 255-img_no

        #(n,784) array
        img_array_list = np.empty(shape = (1,784))

        #get digits
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            #print(x,y,w,h)
            if (90 > w >=70  and h >= 60):        
                new_img = img_no[y:y+h, x:x+w]

                #resize (28x28)
                new_img = cv2.resize(new_img, (28,28), interpolation = cv2.INTER_AREA)#28x28

                #reshape 2D to 1D for prediction
                #28*28 to 784
                num_pixels = new_img.shape[0] * new_img.shape[1]
                img_array = new_img.reshape((1,num_pixels)).astype('float32')
                img_array_list = np.append(img_array_list,np.array(img_array),axis=0)
        img_array_list = img_array_list[1:]       
        return (img,img_array_list)
    
    def predict_digits(self,filename,pickled_model):
        img,img_array_list = self.get_digits(filename)
        pickle_filename,outputdir =  pickled_model,'Pickle files'
        filename = os.path.join(os.getcwd(),outputdir, pickle_filename)
        model = pickle.load(open(filename,'rb'))
        prediction = model.predict(img_array_list)
        print(prediction)

        cv2.imshow('image',img)
        cv2.waitKey(0)

if __name__ == '__main__':

    filename = 'mytestdata/testcase0.jpeg'
    pickled_model = 'nn_mlp.pickle'
    obj = digitrecognizer()
    obj.predict_digits(filename,pickled_model)