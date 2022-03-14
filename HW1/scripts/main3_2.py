import pandas as pd
import numpy as np


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
        self.X_train = X_train.T
        self.Y_train = Y_train
        
        self.X_test = X_test.T
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
        X = self.X_train
        Y = self.Y_train

        A = X.dot(X.T)
        ## create b and preprocess it
        b = Y.multiply(X)
        b = np.sum(b, axis=1)
        
        w = np.linalg.inv(A).dot(b)

        ## Calculate the mean squared error
        Y_pred = X.T @ w
        self.__MSE(Y ,Y_pred)

        return w

    def predict(self, w):
        """
        Predict the Linear Regression using fixed input weights

        w : array_like
            array of weights, shape must meet `(n, 1)`, column vector

        Returns:
        --------
        Y_pred : array_like 
            the prediction of test data, using weights
        """

        Y_pred = self.X_test.T @ w
        self.__MSE(Y_test ,Y_pred)

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
        """

        error = (Y - Y_pred) ** 2 
        error = error / len(error)
        mse = np.sum(error)

        if note != '':
            print(note)
        print(f'Mean Squared Error: {mse}')


if __name__ == '__main__':
    DELIMITER_SIZE = 7

    print('Linear Regression on housing dataset program!')
    print('----------' * DELIMITER_SIZE)
    print('Loading Training and Test sets')
    
    ds = dataset()
    X_train, Y_train = ds.load_train()
    X_test, Y_test = ds.load_test()

    print('datasets loaded!')
    print(f'Train set head:\n{X_train.head()}')
    print(f'Test set head :\n{X_test.head()}')

    print('----------' * DELIMITER_SIZE)
    print('Training: ')

    ## load linear regression class
    lr = LR(X_train, Y_train, X_test, Y_test)

    w = lr.solve()
    ## Prediction of Test set
    print('Testing: ')
    Y_pred_test = lr.predict(w)

    print('----------' * DELIMITER_SIZE)

    print('Weights of the training')
    print(w)
    



