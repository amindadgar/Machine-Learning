## This is a more completed form of main3_3.py
## having the polynomial dataset

from ast import arguments
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

class dataset:
    
    def load_train(self):
        df = self.__dataset()
        ## divide the output
        X = df.drop(columns=['MEDV'])
        Y = df.MEDV

        return X, Y
    
    def load_test(self):
        df = self.__dataset(test = True)
        ## divide the output
        X = df.drop(columns=['MEDV'])
        Y = df.MEDV

        return X, Y

    def __dataset(self, test = False):
        """
        load train or the test dataset

        Parameters:
        -----------
        test : Boolean
            If true, return the test dataset, default False meaning returning train dataset

        Returns:
        --------
        df : pandas dataframe
            returning test or the train dataset
        """
        ## the columns of the dataset
        columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

        if test:
            df = pd.read_csv('hw1_data/housing/housing_test.txt', sep=' +', names=columns, index_col=False, engine='python')
        else:
            df = pd.read_csv('hw1_data/housing/housing_train.txt', sep=' +', names=columns, index_col=False, engine='python')

        df = self.__extendx(df)

        return df

    def __extendx(self, X):
        """
        extend the X dataset and return the linear and two degrees dataset

        Parameters:
        -----------
        X : matrix_like
            our dataset

        Returns:
        --------
        extended_X : matrix_like
            the dataset containing both X and X^2 data
            Note: extended_X length is double the X 
        """
        df = X.copy()
        ## the power 2
        df_2 = df.multiply(df)

        df = pd.concat([ df ,df_2], ignore_index= True)


        return df
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
        b = Y.multiply(X)
        b = np.sum(b, axis=1)
        
        w = np.linalg.inv(A).dot(b)

        ## Calculate the mean squared error
        Y_pred = X.T @ w
        mse = self.__MSE(Y ,Y_pred)
        print(f'Mean Squared Error: {mse}')

        return w

    def predict(self, w, verbose = True):
        """
        Predict the Linear Regression using fixed input weights

        Parameters:
        ------------
        w : array_like
            array of weights, shape must meet `(n, 1)`, column vector
        verbose : Boolean
            print the error value or not

        Returns:
        --------
        Y_pred : array_like 
            the prediction of test data, using weights
        """

        Y_pred = self.X_test @ w
        mse = self.__MSE(Y_test ,Y_pred)
        if verbose:
            print('Test')
            print(f'Mean Squared Error: {mse}')

        return Y_pred
    
    def solve_incremental(self, W, iter = 1000, normalize = True, partial = 0, learning_rate_mode = 0):
        """
        Incremental Learning for Linear Regression
        The method used is online gradient descent
        the learning rate is 2/t, and t stands for iteration number

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
            1 -> `2 / sqrt(t)`
            between 0 and 1 -> static learning rate

        Returns:
        ---------
        W : array_like
            the learned weights for linear regression 
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
            update_term = Y[data_index] - self.__Function(X.iloc[data_index], W)

            ## setting the learning rate
            if learning_rate_mode == 0:
                learning_rate = (2/(i+1)) 
            elif learning_rate_mode == 1:
                learning_rate = 2 / np.sqrt(i+1)
            else:
                learning_rate = learning_rate_mode


            update_term = np.multiply(learning_rate * update_term, X.iloc[data_index])

            ## update the weights
            W = np.add(W, update_term)

            ## calculate and save the errors every partial interval
            if (partial != 0) and (i % partial == 0):
                Y_pred = X @ W
                train_error = self.__MSE(Y, Y_pred)
                Y_test_pred = self.predict(W, verbose=False)
                test_error = self.__MSE(self.Y_test, Y_test_pred)
                
                ## add train and test error as a tuple
                tuple = [train_error, test_error]

                mse.append(tuple)

        
        Y_pred = X @ W
        error = self.__MSE(Y, Y_pred=Y_pred, note='Training')
        print(f'Mean Squared Error: {error}')

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
        cols = df_normalized.columns
        for col in cols:
            df_normalized[col] = (df[col] - df[col].mean() ) / df[col].std()

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
       
        series_normalized = (series - series.mean()) / series.std()
        return series_normalized


    def __Function(self, X, W):
        """
        The function for calculating the predicted output for `X`

        Parameters:
        -----------
        X : array_like
            the features vector  
        W : array_like
            the learned weights

        Returns:
        --------
        Y_pred : float
            The predicted value for the input weights and the feature vector
        """
        Y_pred = np.dot(W.T, X)
        
        return Y_pred

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

        error = (Y - Y_pred) ** 2 
        error = error / len(error)
        mse = np.sum(error)

        if note != '':
            print(note)

        return mse


if __name__ == '__main__':

    arguments = sys.argv

    ## For the incremental_learning
    ## the learning rate can be chosen
    ## default would be '2/t' function, if nothing was passed
    ## example:  'learning_rate=1'
    if len(arguments) == 2:
        string_argument = arguments[1]
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
    
    ## if nothing was passed use the default
    else:
        learning_rate_mode = 0        

    DELIMITER_SIZE = 7

    print('Linear Regression on housing dataset program!')
    print('----------' * DELIMITER_SIZE)
    print('----------' * DELIMITER_SIZE)

    print('Loading Training and Test sets')
    
    ds = dataset()
    X_train, Y_train = ds.load_train()
    X_test, Y_test = ds.load_test()

    print('datasets loaded!')
    print(f'Train set head:\n{X_train.head()}')
    print(f'Test set head :\n{X_test.head()}')

    print('----------' * DELIMITER_SIZE)
    print('----------' * DELIMITER_SIZE)
    print('Training: ')

    ## load linear regression class
    lr = LR(X_train, Y_train, X_test, Y_test)

    w = lr.solve()
    ## Prediction of Test set
    print('Testing using the last trained weights: ')
    Y_pred_test = lr.predict(w)

    print('----------' * DELIMITER_SIZE)

    print('Weights of the training')
    print(w)
    print('----------' * DELIMITER_SIZE)
    print('----------' * DELIMITER_SIZE)

    print('Incremental Learning')
    ## the initial weights
    initial_weights = np.zeros(len(X_train.columns))

    ## the IL stands for Incremental Learning
    w_IL, mse = lr.solve_incremental(initial_weights, partial = 50, learning_rate_mode=learning_rate_mode)


    ## prediction of the test set Incremental Learning
    Y_pred_test_IL = lr.predict(w_IL)
    print('----------' * DELIMITER_SIZE)
    
    print('Final wights of Incremental Learning')
    print(w)

    print('----------' * DELIMITER_SIZE)


    ## if mse array is not empty
    ## plot each loss of intervals 
    print("Partial Wights")
    if mse:
        print("Plotting Training and Test Loss")
        mse = np.array(mse)

        xticks = np.linspace(0, 1000, 20)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(xticks ,mse[:,0])
        axes[0].legend(['Training MSE'])

        axes[1].plot(xticks ,mse[:, 1])
        axes[1].legend(['Test MSE'])

        learning_rate_equation = ''
        if learning_rate_mode == 0:
            learning_rate_equation = 'Learning Rate: 2/t'
        elif learning_rate_mode == 1:
            learning_rate_equation = 'Learning Rate: 2/sqrt(t)'
        else:
            learning_rate_equation = f'Static Learning Rate: {learning_rate_mode}'


        fig.suptitle(f'Online Linear Regression with Extended dataset\n{learning_rate_equation}')

        file_name = learning_rate_equation.replace(' ', '_')
        file_name = file_name.replace('/', '-')
        plt.savefig(f'main3_4_{file_name}.jpg')
        plt.show()    



