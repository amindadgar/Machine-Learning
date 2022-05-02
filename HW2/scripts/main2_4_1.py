## This is the Linear regression form of main3_4.py

from ast import arguments
from cmath import inf
from math import dist
from re import X
from tkinter import Y
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans

class dataset:
    
    def __init__(self, phi, beta) -> None:
        """
        the type of phi function would be applied here
        Note that the K-means function would be applied as requested in the question


        Parameters:
        ------------
        phi : integer
            0 -> part 1 of question 3 (the gaussian type kernel)
            1 -> for part 2 of question 3 (the norm value)
            default: 0 -> The gaussian type
        beta : float
            a hyperparameter for kernel function
        """
        assert phi == 0 or phi == 1, "Error: there are just two type of kernel function, zero and one can be entered"

        self.__phi = phi
        self.__beta = beta
    

    def load_datasets(self):
        """
        load training and test data
        the kernel function will be also applied

        Returns:
        --------
        df_trn : array_like
            the train set initialized and applied values with kernel function
        df_tst : array_like
            the test set initialized and applied values with kernel function
        Y_trn : array_like
            the labels of training set
        Y_tst : array_like
            the labels of test set
        """
        ## the columns of the dataset
        columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

        
        df_tst = pd.read_csv('hw2_data/housing_tst.txt', sep=' +', names=columns, index_col=False, engine='python')
        
        df_trn = pd.read_csv('hw2_data/housing_trn.txt', sep=' +', names=columns, index_col=False, engine='python')

        Y_trn = df_trn[['MEDV']]
        Y_tst = df_tst[['MEDV']]

        centers = self.__find_centers_Kmeans(df_trn[columns[:-1]], Y_trn)
        if self.__phi == 0:
            phi_x_trn = self.__phi_type1(df_trn[columns[:-1]], self.__beta, centers)
            phi_x_tst = self.__phi_type1(df_tst[columns[:-1]], self.__beta, centers)

            ## update dataset
            df_trn = phi_x_trn
            df_tst = phi_x_tst
        elif self.__phi == 1:
            phi_x_trn = self.__phi_type2(df_trn[columns[:-1]], self.__beta, centers)
            phi_x_tst = self.__phi_type2(df_tst[columns[:-1]], self.__beta, centers)

            ## update dataset
            df_trn = phi_x_trn
            df_tst = phi_x_tst

        return df_trn, df_tst, Y_trn, Y_tst

    def __phi_type1(self,X, beta, Mu):
        """
        The kernel function for question three written above
        
        Parameters:
        -----------
        X : pandas dataframe
            the input values
        beta : float
            the hyperparameter of the phi kernel function
        Mu : array_like
            the mu array corresponding to 10 clusters of data points
        
        Returns:
        --------
        phi_x : array_like
            Applied kernel function on inputs `X`
        """
        ## create an empty array
        phi_x = []
        for i in range(len(X)):
            ## find the distance of the data to each cluster centers
            ## and save it in distances array
            distances_arr = []
            for cluster_center in Mu:
                distance = X.iloc[i] - cluster_center

                data = (-0.5 * beta) * (distance @ distance.T)
                distances_arr.append(data)

            phi_x.append(np.exp(distances_arr))
            
        ## just converting to numpy array
        phi_x = np.array(phi_x)
        return phi_x

    def __phi_type2(self, X, a, Mu):
        """
        the second kernel function
        sigmoid type is

        Parameters:
        ------------
        X : pandas dataframe
            the input values
        a : float
            the hyperparameter of the phi kernel function
        Mu : array_like
            the mu array corresponding to 10 clusters of data points
        """
        ## create an empty array
        phi_x = []
        for i in range(len(X)):
            ## find the distance of the data to each cluster centers
            ## and save it in distances array
            distances_arr = []
            for cluster_center in Mu:
                distance = X.iloc[i] - cluster_center

                data = 1 / 1 + (distance @ distance.T)
                distances_arr.append(data)

            phi_x.append(np.exp(distances_arr))
            
        ## just converting to numpy array
        phi_x = np.array(phi_x)
        return phi_x
        
    
    def __find_centers_Kmeans(self, X, Y):
        """
        the K-means algorithm for kernel functions
        this function is used to find the centers of each data


        Parameters:
        ------------
        X : array_like
            the dataset inputs
        Y : array_like
            the labels of each input (output value)
        
        Returns:
        ---------
        centers : array_like
            the center of each data cluster
        """
        KmeansClassifier = KMeans(n_clusters=10)
        KmeansClassifier.fit(X=X,y= Y)
        
        centers = KmeansClassifier.cluster_centers_

        return centers

class LR:
    def __init__(self, X_train, Y_train, X_test, Y_test) -> None:
        """
        initialize the class 

        Parameters:
        -----------
        X_train, X_test : matrix_like
            The features vector for training and test sets
        Y_train, Y_test : array_like
            target output corresponding to each feature vector of training and test sets
        """
        ## transposing the feature vectors, because the must be a column vector
        self.X_train = X_train
        self.Y_train = Y_train
        
        self.X_test = X_test
        self.Y_test = Y_test

    def solve(self):
        """
        Linear regression solve function
        Using the old equation w = invers(A) * b

        Returns:
        --------
        w : array_like
            The vector of weights fitted on `X` features
        """
        X = self.X_train.T
        Y = self.Y_train.T

        A = X.dot(X.T)
        ## create b and preprocess it
        b = X.dot(Y.T)
        # b = np.sum(b, axis=1)

        w = np.linalg.inv(A).dot(b)

        ## Calculate the mean squared error

        Y_pred = X.T @ w
        mse = self.__MSE(Y ,Y_pred)
        print(f'  Mean Squared Error:\n {mse.values}')

        return w

    def predict(self, w):
        """
        Predict the Linear Regression or Linear Regression using fixed input weights
        

        Parameters:
        ------------
        w : array_like
            array of weights, shape must meet `(n, 1)`, column vector

        Returns:
        --------
        Y_pred : array_like 
            the prediction of test data, using weights
        """
        ## the linear regression
        Y_pred = self.X_test @ w
        
        mse = self.__MSE(Y_test.values ,Y_pred)
        
        return Y_pred, mse

    
    def __LRGD_discriminant_func(self, w, x):
        """
        the discriminant function for Linear Regression
        the sigmoid function is used

        Parameters:
        ------------
        w : array_like
            weights of the function
        x : array_like
            the input values

        Returns:
        ---------
        y : array_like
            the classified value corresponding to each x input
        """
        y = np.dot(w.T,x.T)
        return y


    def solve_LRGD(self, X, Y, initial_weights, max_iter = 50, normalize = True):
        """
        The linear regression function

        Parameters:
        ------------
        X : array_like
            training data
        Y : array_like
            outputs for training data

        initial_weights : array_like
            the initial weights for the data
        max_iter : positive integer
            the maximum iteration counts for learning algorithm
        normalize : bool
            if true, normalize the dataset

        Returns:
        ---------
        weights : array_like
            the learned weights
        mse : array_like
            the last iteration mean square error 
        """
        assert learning_rate_mode >=0 and learning_rate_mode <=1, f"Error: Learning_rate_mode must be between 0 and 1!, the entered is {learning_rate_mode}"

        if normalize:
            X = self.__normalize_df(X)
            Y = self.__normalize(Y)
        
        errors_arr = []
        ## retrieve the initial weights
        weights = initial_weights

        ## Start the learning phase
        for i in range(0, max_iter):

            ## adjusting learning rate
            if learning_rate_mode == 0:
                ## plus one is added to avoid division by zero
                learning_rate = (2/(i+1)) 
            elif learning_rate_mode == 1:
                learning_rate = 1 / np.sqrt(i+1)
            else:
                learning_rate = learning_rate_mode
        
            
            ## calculate the changes needed
            w_changes = 0
            for j in range(0, len(X)):
                ## the predicted value in each iteration
                pred = self.__LRGD_discriminant_func(weights, X[j])

                # if np.isnan(pred):
                    # print(i, '  ', j)
                    # raise "AFDASD"
                w_changes += np.matrix((Y[j].values - pred) * X[j]).T

            weights += np.multiply(learning_rate, w_changes)

            ## calculate the error each time
            Y_pred = self.__LRGD_discriminant_func(weights, X)
            error = self.__MSE(Y.values, Y_pred=Y_pred)

            errors_arr.append(error)

        return weights, errors_arr


    def solve_incremental(self, W, iter = 1000, normalize = True, partial = 0, learning_rate_mode = 0):
        """
        Incremental Learning for Linear Regression
        The method used is online gradient descent
        the learning rate can be `2/t` or `1/sqrt(t)`, `t` stands for iteration number

        Parameters:
        ------------
        X : matrix_like
            the features vectors for training 
        Y : array_like
            the target output for each feature vectors represented in `X`
        W : array_like
            the initial weights for online gradient descent
        iter : integer
            the number of iterations to learn
        normalize : Boolean
            if True normalize the dataset, default: True
        partial : integer
            save and return the mean square errors in every intervals defined in this variable
            default is 0 meaning return no partial results 
        learning_rate_mode : integer
            representing to go through which learning rate function, default is 0
            0 -> `2/t`
            1 -> `1 / sqrt(t)`
            between 0 and 1 -> static learning rate

        Returns:
        ---------
        W : array_like
            the learned weights for Linear regression 
        mse : array_like
            the mean squared error of train and test for each interval
        """
        assert learning_rate_mode >=0 and learning_rate_mode <=1, f"Error: Learning_rate_mode must be between 0 and 1!, the entered is {learning_rate_mode}"

        X_unnormalized = self.X_train
        Y_unnormalized = self.Y_train

        ## normalize the datasets
        if normalize:
            X = self.__normalize_df(X_unnormalized)
            Y = self.__normalize(Y_unnormalized)
        else:
            X = X_unnormalized
            Y = Y_unnormalized

        mse = []
        ## incremental learning
        ## change the weights for each data
        for i in range(iter):
            ## data index is different from the index
            ## so we calculate it everytime
            data_index = i % len(Y)

            ## the update term that is added to old weight
            pred = self.__LRGD_discriminant_func(W, X[data_index])
            
            update_term = Y.iloc[data_index].values - pred

            ## setting the learning rate
            if learning_rate_mode == 0:
                learning_rate = (2/(i+1)) 
            elif learning_rate_mode == 1:
                learning_rate = 1 / np.sqrt(i+1)
            else:
                learning_rate = learning_rate_mode


            update_term = learning_rate * update_term * X[data_index]
            update_term = np.array(update_term).reshape(10, 1)

            ## update the weights
            W += update_term
            Y_pred = self.__LRGD_discriminant_func(W, X)

            mse = self.__MSE(Y, Y_pred)

        return W, mse

    def __normalize_df(self, df):
        """
        normalize a pandas dataframe

        Parameters:
        ------------
        df : pandas dataframe
            the dataset that is going to be normalized
        
        Returns:
        ----------
        df : pandas dataframe
            the normalized dataframe
        """
        df_normalized = df.copy()

        ## for each columns normalize it
        for col in range(0, df_normalized.shape[1]):
            df_normalized[:, col] = (df[:, col] - df[:, col].mean() ) / df[:, col].std()

        return df_normalized

    def __normalize(self, series):
        """
        normalize a pandas series

        Parameters:
        ------------
        df : pandas dataframe
            the dataset that is going to be normalized
        
        Returns:
        ----------
        df : pandas dataframe
            the normalized dataframe
        """
        series_normalized = series.copy()
       
        series_normalized = (series - series.mean()) / np.std(series_normalized.values)
        return series_normalized

    def __MSE(self, Y ,Y_pred, note=''):
        """
        Calculate the Mean Square error and print the value

        Parameters:
        -----------
        Y_pred : array_like
            the predicted Y value
        note : string
            a string text, for additional notes to be printed

        Returns:
        --------
        mse : floating value
            the mean square error 
        """
        error = np.power(Y.T - Y_pred, 2) 
        error = error / len(error)
        mse = np.sum(error)

        if note != '':
            print(note)

        return mse

def get_learning_rate_argument(string_argument):
    """
    retreive the learning rate via the argument passed (For online gradient descent)

    Parameters:
    ------------
    string_argument : string
        the argument for learning rate as a string
        example: `learning_rate=0.5`

    Returns:
    ---------
    learning_rate_mode : integer
        the learning_rate selected for online gradient descent
    """

    ## check if the float number is available
    idx = string_argument.find('.')
    ## if a point was found then there is a float number for learning rate
    if idx != -1:
        learning_rate_mode = string_argument[idx -1 :]
    ## else there is no float number, so return the last character
    else:
        learning_rate_mode = string_argument[idx:]
        
    ## convert to float
    learning_rate_mode = float(learning_rate_mode)

    return learning_rate_mode
def get_OfflineGD_iterations(string_argument):
    """
    Select the offline gradient descent iterations

    Parameters:
    ------------
    string_argument : string
        the argument for iteration count as a string
        example: `offline_iterationGD=50`

    Returns:
    ---------
    iter : integer
        the iteration counts selected for offline gradient descent
    """
    args = string_argument.split('=')
    iter = args[1]
    ## convert the string iterations to integer
    iter = int(iter)

    return iter


if __name__ == '__main__':
    ##################################### Retreiving arguments #####################################

    arguments = sys.argv
    ## For the incremental_learning
    ## the learning rate can be chosen
    ## default would be '2/t' function, if nothing was passed
    ## example:  'learning_rate=1'
    if len(arguments) == 2:
        string_argument = arguments[1]
        learning_rate_mode = get_learning_rate_argument(string_argument)

    ## example:  'learning_rate=1 iterations=500'
    elif len(arguments) == 3:
        string_argument = arguments[1]
        learning_rate_mode = get_learning_rate_argument(string_argument)

        iterations = get_OfflineGD_iterations(arguments[2])
    ## if nothing was passed use the default `2/t` and iteration count of 50
    else:
        learning_rate_mode = 0   
        iterations = 50     

    DELIMITER_SIZE = 7

    ##################################### START THE PROGRAM #####################################
    print('Linear Regression on classification dataset!')

    print('----------' * DELIMITER_SIZE)
    print('----------' * DELIMITER_SIZE)

    print('Loading Training and Test sets')
    
    ds = dataset(phi=0, beta=0.00001)
    X_train, X_test, Y_train, Y_test = ds.load_datasets()

    print('datasets loaded!')
    print(f'Train set head:\n{X_train[:2]}')
    print(f'Test set head :\n{X_test[:2]}')

    print('----------' * DELIMITER_SIZE)
    print('----------' * DELIMITER_SIZE)

    ##################################### Closed Form Solution #####################################
    print('\033[1mClosed Form Training: \033[0m')

    # ## load Linear regression class
    lr = LR(X_train, Y_train, X_test, Y_test)
    print(' Train Error: ')
    w = lr.solve()
    ## Prediction of Test set
    print(' Test Error: ')
    Y_pred_test, Y_pred_test_mse = lr.predict(w)
    print(f'  {Y_pred_test_mse}')

    print(' Final Weights of the closed form training')
    print(w)
    
    print('----------' * DELIMITER_SIZE)
    print('----------' * DELIMITER_SIZE)

    ##################################### Gradient Descent: Linear Regression #####################################
    
    # print(f'\033[1mGradient descent For Linear Regression with T={iterations} iterations\033[0m')
    # initial_weights = np.ones((X_train.shape[1], 1))
    
    # print(' Train Errors: ')
    # weights, offline_GD_mse= lr.solve_LRGD(X_train, Y_train.T, initial_weights, max_iter=iterations, normalize=False)
    # print('  MSE:', offline_GD_mse[-1:])
    # print(' Test Error: ')
    # Y_pred_test_Offline, offline_GD_test_mse = lr.predict(weights)
    # print('  MSE:', offline_GD_mse[-1:])

    # print('\n Final wights of Offline Gradient Descent Learning')
    # print(weights)

    
    # print('----------' * DELIMITER_SIZE)
    # print('----------' * DELIMITER_SIZE)

    ##################################### Online Gradient Descent : Linear Regression #####################################
    print('\033[1mIncremental Learning (Online GD) \033[0m')
    ## the initial weights
    initial_weights = np.ones((X_train.shape[1], 1))

    print('  Training Error:')
    ## the IL stands for Incremental Learning
    w_IL, mse = lr.solve_incremental(initial_weights, partial = 50, learning_rate_mode=learning_rate_mode, iter=iterations)
    print(f'   MSE:{mse[-1:].values}')

    print('  Test Error:')
    ## prediction of the test set Incremental Learning
    Y_pred_test_IL, Y_pred_test_IL_mse = lr.predict(w_IL)
    print(f'   MSE:{Y_pred_test_IL_mse}')

    

    print('\n Final wights of Incremental Learning')
    print(w_IL)

    print('----------' * DELIMITER_SIZE)




