import sys
import os
import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
from svgd import svgd_bayesnn
from classification import classification
import time
import preprocessing

if __name__ == "__main__":
    if len(sys.argv)>1:
        if len(sys.argv)<3:
            print('Requires arguments 0 or 3, but got {}'.format(len(sys.argv)))
            exit()

        train_dataset_path=sys.argv[1]
        test_dataset_path=sys.argv[2]

        prep=preprocessing.pre_processing()
    
        ## Transformation of training data and storing into local machine
        train_data, train_df_y = prep.training_preprocessing('Dataset')
        train_data.to_csv('preprocessed/train_data.csv', index=False)
        train_df_y.to_csv('preprocessed/labels.csv', index=False)
            
        ## Transformation of Evaluation data and storing into local machine
        ids, test_sub_df = prep.evaluation_preprocessing('Dataset')
        ids.to_csv('preprocessed/test_ids.csv', index=False)
        test_sub_df.to_csv('preprocessed/test_data.csv', index=False)
    
    if not os.path.exists('preprocessed/train_data.csv') or not os.path.exists('preprocessed/labels.csv'):
        print('either train data or labels file does not exist')
        exit()

    '''Loading Data for training'''
    X_input = np.array(pd.read_csv('preprocessed/train_data.csv'))
    y_input_raw = np.array(pd.read_csv('preprocessed/labels.csv'))
    y_input = y_input_raw.reshape((y_input_raw.shape[0],))
    print(X_input.shape, y_input.shape)
    
    '''Building the training and testing data set'''
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_input, y_input, test_size=0.20)
    print(X_train.shape, y_train.shape)

    '''Building, fitting and prediction using Standard Classification Algorithms'''
    classifiers = classification()
    classifiers.fit(X_train, y_train)
    
    classifiers.predict(X_test, y_test)
        
    ''' Training Bayesian neural network with SVGD '''
    start = time.time()
    
    batch_size, n_hidden, max_iter = 100, 50, 2000  # max_iter is a trade-off between running time and performance
    
    svgd = svgd_bayesnn(X_train, y_train, batch_size = batch_size, n_hidden = n_hidden, max_iter = max_iter)
    
    svgd_time = time.time() - start
    
    svgd_rmse, svgd_ll = svgd.evaluation(X_test, y_test)
    print('SVGD', svgd_rmse, svgd_ll, svgd_time)

    # ''' Predictions for test data '''
    # test_data=pd.read_csv('preprocessed/test_data.csv')
    # test_ids=pd.read_csv('preprocessed/test_ids.csv')
    # print(test_data.shape, test_ids.shape)

    # preds=svgd.prediction(np.array(test_data))
    # preds