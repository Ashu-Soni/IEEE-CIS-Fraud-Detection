import theano.tensor as T
import theano
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random
import time
import pandas as pd

class svgd_bayesnn:
    def __init__(self, X_train, y_train,  batch_size = 100, max_iter = 1000, M = 20, n_hidden = 50, a0 = 1, b0 = 0.1, master_stepsize = 1e-3, auto_corr = 0.9):
        self.n_hidden = n_hidden
        self.d = X_train.shape[1]   # number of data, dimension 
        self.M = M
        
        num_vars = self.d * n_hidden + n_hidden * 2 + 3  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden; b2 = 1; 2 variances
        self.theta = np.zeros([self.M, num_vars])  # particles, will be initialized later
        
        ## 10% of data for model validation purpose(i.e. used after training for the evaluation)
        size_dev = min(int(np.round(0.1 * X_train.shape[0])), 500)
        X_dev, y_dev = X_train[-size_dev:], y_train[-size_dev:]
        X_train, y_train = X_train[:-size_dev], y_train[:-size_dev]
              
        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)
        
        '''
            Defining the neural network here
        '''
        X = T.matrix('X') 
        y = T.vector('y') 
        
        w_1 = T.matrix('w_1') 
        b_1 = T.vector('b_1') 
        w_2 = T.vector('w_2') 
        b_2 = T.scalar('b_2') 
        
        N = T.scalar('N') 
        
        # variances related parameters
        log_gamma = T.scalar('log_gamma')   
        log_lambda = T.scalar('log_lambda')
        
        prediction = T.dot(T.nnet.relu(T.dot(X, w_1)+b_1), w_2) + b_2
        
        '''
            defining the log posterior distribution 
                - It is the part of driving force of the weight updation part
        '''
        log_lik_data = -0.5 * X.shape[0] * (T.log(2*np.pi) - log_gamma) - (T.exp(log_gamma)/2) * T.sum(T.power(prediction - y, 2))
        log_prior_data = (a0 - 1) * log_gamma - b0 * T.exp(log_gamma) + log_gamma
        log_prior_w = -0.5 * (num_vars-2) * (T.log(2*np.pi)-log_lambda) - (T.exp(log_lambda)/2)*((w_1**2).sum() + (w_2**2).sum() + (b_1**2).sum() + b_2**2)  \
                       + (a0-1) * log_lambda - b0 * T.exp(log_lambda) + log_lambda
        
        # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations
        log_posterior = (log_lik_data * N / X.shape[0] + log_prior_data + log_prior_w)
        dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda = T.grad(log_posterior, [w_1, b_1, w_2, b_2, log_gamma, log_lambda])
        
        # automatic gradient
        logp_gradient = theano.function(
             inputs = [X, y, w_1, b_1, w_2, b_2, log_gamma, log_lambda, N],
             outputs = [dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda]
        )
        
        # prediction function
        self.nn_predict = theano.function(inputs = [X, w_1, b_1, w_2, b_2], outputs = prediction)
        
        '''
            Training with SVGD
        '''
        N0 = X_train.shape[0]  # number of observations
        
        ''' initializing all particles '''
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.init_weights(a0, b0)
            ridx = np.random.choice(range(X_train.shape[0]), \
                                           np.min([X_train.shape[0], 1000]), replace = False)
            y_hat = self.nn_predict(X_train[ridx,:], w1, b1, w2, b2)
            loggamma = -np.log(np.mean(np.power(y_hat - y_train[ridx], 2)))
            self.theta[i,:] = self.pack_weights(w1, b1, w2, b2, loggamma, loglambda)

        grad_theta = np.zeros([self.M, num_vars])
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(max_iter):
            # sub-sampling
            batch = [ i % N0 for i in range(iter * batch_size, (iter + 1) * batch_size) ]
            for i in range(self.M):
                w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i,:])
                dw1, db1, dw2, db2, dloggamma, dloglambda = logp_gradient(X_train[batch,:], y_train[batch], w1, b1, w2, b2, loggamma, loglambda, N0)
                grad_theta[i,:] = self.pack_weights(dw1, db1, dw2, db2, dloggamma, dloglambda)
                
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(h=-1)  
            grad_theta = (np.matmul(kxy, grad_theta) + dxkxy) / self.M   # \Phi(x)
            
            # adagrad 
            if iter == 0:
                historical_grad = historical_grad + np.multiply(grad_theta, grad_theta)
            else:
                historical_grad = auto_corr * historical_grad + (1 - auto_corr) * np.multiply(grad_theta, grad_theta)
            adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
            self.theta = self.theta + master_stepsize * adj_grad 

        '''
            Model selection by using a development set
        '''
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
            pred_y_dev = self.nn_predict(X_dev, w1, b1, w2, b2) * self.std_y_train + self.mean_y_train

            # likelihood
            def f_log_lik(loggamma): return np.sum(  np.log(np.sqrt(np.exp(loggamma)) /np.sqrt(2*np.pi) * np.exp( -1 * (np.power(pred_y_dev - y_dev, 2) / 2) * np.exp(loggamma) )) )
            
            # The higher probability is better    
            lik1 = f_log_lik(loggamma)
            
            # one heuristic setting
            loggamma = -np.log(np.mean(np.power(pred_y_dev - y_dev, 2)))
            lik2 = f_log_lik(loggamma)
            if lik2 > lik1:
                self.theta[i,-2] = loggamma  # update loggamma
    
    '''
        Initialize all particles
    '''
    def init_weights(self, a0, b0):
        w1 = 1.0 / np.sqrt(self.d + 1) * np.random.randn(self.d, self.n_hidden)
        b1 = np.zeros((self.n_hidden,))
        w2 = 1.0 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden)
        b2 = 0.
        loggamma = np.log(np.random.gamma(a0, b0))
        loglambda = np.log(np.random.gamma(a0, b0))
        return (w1, b1, w2, b2, loggamma, loglambda)
    
    '''
        Calculate kernel matrix and its gradient: K, \nabla_x k
    ''' 
    def svgd_kernel(self, h = -1):
        sq_dist = pdist(self.theta)
        pairwise_dists = squareform(sq_dist)**2
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(self.theta.shape[0]+1))

        # compute the rbf kernel
        
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, self.theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(self.theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(self.theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)
    
    
    '''
        Pack all parameters in our model
    '''    
    def pack_weights(self, w1, b1, w2, b2, loggamma, loglambda):
        params = np.concatenate([w1.flatten(), b1, w2, [b2], [loggamma],[loglambda]])
        return params
    
    '''
        Unpack all parameters in our model
    '''
    def unpack_weights(self, z):
        w = z
        w1 = np.reshape(w[:self.d*self.n_hidden], [self.d, self.n_hidden])
        b1 = w[self.d*self.n_hidden:(self.d+1)*self.n_hidden]
    
        w = w[(self.d+1)*self.n_hidden:]
        w2, b2 = w[:self.n_hidden], w[-3] 
        
        # the last two parameters are log variance
        loggamma, loglambda= w[-2], w[-1]
        
        return (w1, b1, w2, b2, loggamma, loglambda)

    
    '''
        Evaluating testing rmse and log-likelihood
    '''
    def evaluation(self, X_test, y_test):        
        # average over the output
        pred_y_test = np.zeros([self.M, len(y_test)])
        prob = np.zeros([self.M, len(y_test)])
        
        '''
            Since we have M particles, we use a Bayesian view to calculate rmse and log-likelihood
        '''
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
            pred_y_test[i, :] = self.nn_predict(X_test, w1, b1, w2, b2) * self.std_y_train + self.mean_y_train
            prob[i, :] = np.sqrt(np.exp(loggamma)) /np.sqrt(2*np.pi) * np.exp( -1 * (np.power(pred_y_test[i, :] - y_test, 2) / 2) * np.exp(loggamma) )
        pred = np.mean(pred_y_test, axis=0)
        
        # evaluation
        svgd_rmse = np.sqrt(np.mean((pred - y_test)**2))
        svgd_ll = np.mean(np.log(np.mean(prob, axis = 0)))
        
        return (svgd_rmse, svgd_ll)

    def prediction(self, test):
        # average over the output
        pred_y_test = np.zeros([self.M, len(test)])
        prob = np.zeros([self.M, len(test)])
        
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
            pred_y_test[i, :] = self.nn_predict(test, w1, b1, w2, b2) * self.std_y_train + self.mean_y_train
        pred = np.mean(pred_y_test, axis=0)

        return pred