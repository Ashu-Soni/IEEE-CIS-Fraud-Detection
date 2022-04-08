import numpy as np
import pandas as pd

import sklearn.linear_model as linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torchbnn as bnn

class classification:
    def __init__(self):
        self.logistic_regression = linear_model.LogisticRegression(max_iter=1000, verbose=2, class_weight='balanced')
        
        self.decision_tree=DecisionTreeClassifier(splitter="best", class_weight='balanced')
        
        self.random_forest=RandomForestClassifier(max_leaf_nodes=10000)
        
        self.ann = nn.Sequential(
            nn.Linear(218, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.bnn = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=1.0, in_features=218, out_features=512),
            bnn.BayesLinear(prior_mu=0, prior_sigma=1.0, in_features=512, out_features=256),
            bnn.BayesLinear(prior_mu=0, prior_sigma=1.0, in_features=256, out_features=128),
            bnn.BayesLinear(prior_mu=0, prior_sigma=1.0, in_features=128, out_features=32),
            bnn.BayesLinear(prior_mu=0, prior_sigma=1.0, in_features=32, out_features=1),
            nn.Sigmoid()
        )
        
    def fit(self, x_train, y_train, learning_rate=0.01):
        print('fitting model for Logistic Regression')
        self.logistic_regression.fit(x_train, y_train)
        
        print('fitting model for Decision Tree')
        self.decision_tree.fit(x_train, y_train)
        
        print('fitting model for Random Forest')
        self.random_forest.fit(x_train, y_train)
        
        print('fitting model for Artificial Neural Network')
        mse_loss_ann=nn.MSELoss()
        optimizer_ann=torch.optim.Adam(self.ann.parameters(), lr=0.01)
        kfold_nn=model_selection.KFold(shuffle=True)
        
        test_nn_accurcy=[]
        for epoch in range(50):
            print('\nEpoch {}'.format(epoch+1))
            cnt=1
            for train_ids, test_ids in kfold_nn.split(x_train):
                train_x_nn, test_x_nn = torch.from_numpy(np.array(x_train)[train_ids]),torch.from_numpy(np.array(x_train)[test_ids])
                train_y_nn, test_y_nn = torch.from_numpy(np.array(y_train)[train_ids]),torch.from_numpy(np.array(y_train)[test_ids])

                # Forward pass: Compute predicted y by passing x to the model
                y_pred=self.ann(train_x_nn.float())

                loss = mse_loss_ann(y_pred, train_y_nn.float())

                optimizer_ann.zero_grad()

                # perform a backward pass (backpropagation)
                loss.backward()

                # Update the parameters
                optimizer_ann.step()

                # Validation calculation
                test_out = self.ann(test_x_nn.float())
                loss_test = mse_loss_ann(test_out, test_y_nn.float())

                score_test=metrics.roc_auc_score(test_y_nn.numpy(), test_out.detach().numpy())

                print('\tsplit {}'.format(cnt))
                print('\t\t Train loss: {}\n\t\t Test Loss: {}\n\t\t Test Accuracy: {}'.format(
                    loss.item(), loss_test.item(), score_test
                ))
                cnt+=1
                
        print("fitting model for Bayesian Neural Network")
        mse_loss_bnn=nn.MSELoss()
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        kl_weight = 0.01
        optimizer_bnn = torch.optim.Adam(self.bnn.parameters(), lr=0.1)
        
        kfold_bnn=model_selection.KFold(shuffle=True)
        for epoch in range(50):
            print('\n')
            print('echo {}'.format(epoch+1))
            cnt=1
            for train_ids, test_ids in kfold_bnn.split(x_train):
                train_x_bnn, test_x_bnn = torch.from_numpy(np.array(x_train)[train_ids]),torch.from_numpy(np.array(x_train)[test_ids])
                train_y_bnn, test_y_bnn = torch.from_numpy(np.array(y_train)[train_ids]),torch.from_numpy(np.array(y_train)[test_ids])

                # Forward pass: Compute predicted y by passing x to the model
                y_pred=self.bnn(train_x_bnn.float())

                #  loss = cross_loss_bnn(y_pred, train_y_bnn.float())
                loss_bnn = mse_loss_bnn(y_pred, train_y_bnn.float())

                kl = kl_loss(self.bnn)
                cost_bnn = loss_bnn + kl_weight*kl

                optimizer_bnn.zero_grad()

                # perform a backward pass (backpropagation)
                cost_bnn.backward()

                # Update the parameters
                optimizer_bnn.step()

                # Validation calculation
                test_out = self.bnn(test_x_bnn.float())
                loss_test = mse_loss_bnn(test_out, test_y_bnn.float())

                _, preds_test = torch.max(test_out, 1)
                preds_test_np = np.squeeze(preds_test.numpy())

                score_test=metrics.roc_auc_score(test_y_bnn, test_out.detach().numpy())

                print('\tsplit {}'.format(cnt))
                print('\t\t Train loss: {}\n\t\t Test Loss: {}\n\t\t Test Accuracy: {}'.format(
                    loss.item(), loss_test.item(), score_test
                ))
                cnt+=1
                
    def predict(self, x_test, y_test):
        print('Prediction for test data using Logistic Regression...')
        y_pred_lr=self.logistic_regression.predict_proba(x_test)[:, -1]
        score_lr=metrics.roc_auc_score(y_test, y_pred_lr)
        print('Accuracy of Logistic Regression: ', score_lr)
        
        print('Prediction for test data using Decision Tree...')
        y_pred_dt=self.decision_tree.predict_proba(x_test)[:, -1]
        score_dt=metrics.roc_auc_score(y_test, y_pred_dt)
        print('Accuracy of Decision Tree: ', score_dt)
        
        print('Prediction for test data using Random Forest...')
        y_pred_rf=self.random_forest.predict_proba(x_test)[:, -1]
        score_rf=metrics.roc_auc_score(y_test, y_pred_rf)
        print('Accuracy of Random Forest: ', score_rf)

if __name__ == '__main__':
    X_input = np.array(pd.read_csv('preprocessed/train_data.csv'))
    y_input_raw = np.array(pd.read_csv('preprocessed/labels.csv'))
    y_input = y_input_raw.reshape((y_input_raw.shape[0],))
    print(X_input.shape, y_input.shape)
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_input, y_input, test_size=0.20)
    print(X_train.shape, y_train.shape)
    
    classifiers = classification()
    classifiers.fit(X_train, y_train)
    
    classifiers.predict(X_test, y_test)