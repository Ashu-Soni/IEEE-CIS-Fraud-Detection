import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.metrics as metrics
import missingno as msn
from sklearn import impute
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection

class pre_processing:
    def __init__(self):
        self.mean_imputer=impute.SimpleImputer(missing_values=np.nan, strategy='mean')
        self.mode_imputer=impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        self.median_imputer=impute.SimpleImputer(missing_values=np.nan, strategy='median')
        self.encoder=preprocessing.OrdinalEncoder()
        self.standardScaler=preprocessing.StandardScaler()
        self.drop_columns=[]
               
    def training_preprocessing(self, path):
        print('Preprocessing for training data\n')
        
        print('Loading training data...')
        train_transactions=pd.read_csv('{}/train_transaction.csv'.format(path))
        train_identity=pd.read_csv('{}/train_identity.csv'.format(path))
        print('Data loaded, success!')
        print('transactions data of shape {} and identity data of shape {}'.format(train_transactions.shape, 
                                                                                  train_identity.shape))
        
        print('\nMerging Transactions and Identity data')
        train_df=pd.merge(train_transactions, train_identity, on='TransactionID', how='left')
        print('Training data shape: ',train_df.shape)
        
        print('\nMissing Values Management')
        total=train_df.shape[0]
        cnt=0
        for col in train_df.columns:
            n=train_df[col].isna().sum()
            if n>0:
                cnt+=1
        print('Number of attributes having missing values(before): ',cnt)
        
        train_df_x=train_df.drop(axis='columns', labels=['isFraud'])
        train_df_y=train_df[['isFraud']]
        
        for col in train_df_x.columns:
            n=train_df_x[col].isna().sum()
            if n/total>0.50:
                self.drop_columns.append(col)
        
        train_df_removed=train_df_x.drop(axis='columns', labels=self.drop_columns)
        train_df_removed=train_df_removed.drop(axis='columns', labels=['TransactionID'])
        
        median_impute_columns=[]
        mode_impute_columns=[]
        for col in train_df_removed.columns:
            if train_df_removed[col].dtypes=='object':
                mode_impute_columns.append(col)
            else:
                median_impute_columns.append(col)
                
        median_df=train_df_removed[median_impute_columns]
        mode_df=train_df_removed[mode_impute_columns]
        mode_df=self.encoder.fit_transform(mode_df)
        
        median_df=pd.DataFrame(data=self.median_imputer.fit_transform(median_df))
        mode_df=pd.DataFrame(data=self.mode_imputer.fit_transform(mode_df))
        
        print('\nData Standardisation')
        median_df_norm=pd.DataFrame(data=self.standardScaler.fit_transform(median_df))
        
        train_data=pd.concat([median_df_norm, mode_df], axis=1)
        
        print('Number of attributes having missing values(after): ',cnt)
        
        print('\nPreprocessing for training data is completed!')
        
        return train_data, train_df_y
    
    def evaluation_preprocessing(self, path):
        print('\n####################################################')
        print('\nPreprocessing for Evaluation data\n')
        
        print('Loading evaluation data...')
        test_transactions=pd.read_csv('{}/test_transaction.csv'.format(path))
        test_identity=pd.read_csv('{}/test_identity.csv'.format(path))
        print('Data loaded, success!')
        print('transactions data of shape {} and identity data of shape {}'.format(test_transactions.shape, 
                                                                                  test_identity.shape))
        print('\nMerging Transactions and Identity data')
        test_df=pd.merge(test_transactions, test_identity, on='TransactionID', how='left')
        print('Evaluation data shape: ',test_df.shape)
        
        ids=test_df['TransactionID']
        test_df.drop(axis='columns', labels=['TransactionID'], inplace=True)
        
        test_drop_columns=[]
        for val in self.drop_columns:
            if val.split('_')[0]=='id':
                test_drop_columns.append('id-{}'.format(val.split('_')[1]))
            else:
                test_drop_columns.append(val)
        test_df_removed=test_df.drop(axis='columns', labels=test_drop_columns)
        
        median_test_columns=[]
        mode_test_columns=[]
        for col in test_df_removed.columns:
            if test_df_removed[col].dtype=='object':
                mode_test_columns.append(col)
            else:
                median_test_columns.append(col)
                
        median_test_df=test_df_removed[median_test_columns]
        mode_test_df=test_df_removed[mode_test_columns]
        mode_test_df=self.encoder.fit_transform(mode_test_df)
        
        median_test_df=pd.DataFrame(data=self.median_imputer.transform(median_test_df))
        mode_test_df=pd.DataFrame(data=self.mode_imputer.transform(mode_test_df))
        
        print('\nData Standardisation')
        median_test_df_norm=pd.DataFrame(data=self.standardScaler.transform(median_test_df))
        
        test_sub_df=pd.concat([median_test_df_norm, mode_test_df], axis=1)
        
        print('\nPreprocessing for evaluation data is completed!')
        
        return ids, test_sub_df

if __name__ == '__main__':
    ## Initialization of preprocessing class
    prep=pre_processing()
    
    ## Transformation of training data and storing into local machine
    train_data, train_df_y = prep.training_preprocessing('Dataset')
    train_data.to_csv('train_data.csv', index=False)
    train_df_y.to_csv('labels.csv', index=False)
     
    ## Transformation of Evaluation data and storing into local machine
    ids, test_sub_df = prep.evaluation_preprocessing('Dataset')
    ids.to_csv('test_ids.csv', index=False)
    test_sub_df.to_csv('test_data.csv', index=False)