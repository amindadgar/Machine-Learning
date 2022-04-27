## This is the logistic regression form of main3_4.py

from ast import arguments
from cmath import inf
from tkinter import Y
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

class dataset:
    
    def load_train(self):
        df = self.__dataset()
        ## divide the output
        X = df.drop(columns=['label'])
        Y = df.label

        return X, Y
    
    def load_test(self):
        df = self.__dataset(test = True)
        ## divide the output
        X = df.drop(columns=['label'])
        Y = df.label

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
        cols = ['X', 'Y', 'label']

        if test:
            df = pd.read_csv('./hw2_data/classification_tst.txt',
                       delimiter=' +',
                      names=cols,
                      index_col=False,
                      engine='python')
        else:
            df = pd.read_csv('./hw2_data/classification_trn.txt',
                       delimiter=' +',
                      names=cols,
                      index_col=False,
                      engine='python')

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
        Logistic regression solve function
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
        print(f'  Mean Squared Error: {mse}')

        return w

    def predict(self, w, method=2):
        """
        Predict the Linear Regression or Logistic Regression using fixed input weights
        

        Parameters:
        ------------
        w : array_like
            array of weights, shape must meet `(n, 1)`, column vector
        method : integer
            `1` -> Choose Linear Regression Prediction
            `2` -> Choose Logistic Regression Prediction
            default is `1` meaning Linear Regression 

        Returns:
        --------
        Y_pred : array_like 
            the prediction of test data, using weights
        """
        ## the linear regression
        if method == 1:
            Y_pred = self.X_test @ w
        ## otherwise choose logistic regression
        else:
            Y_pred = self.__LRGD_discriminant_func(w, X_test.T)
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
            
        
        mse = self.__MSE(Y_test.values ,Y_pred)
        
        return Y_pred, mse

    
    def __LRGD_discriminant_func(self, w, x):
        """
        the discriminant function for Logistic Regression
        the exponential function is used

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
        y = 1 / (1 + np.exp(-w.T @ x))
        ## remove infinity values (caused by zero division)
        y = np.where(y == inf, 1, y)

        return y


    def solve_LRGD(self, X, Y, initial_weights, max_iter = 50):
        """
        The offline logistic regression function

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

        Returns:
        ---------
        weights : array_like
            the learned weights
        mse : array_like
            the last iteration mean square error 
        """
        assert learning_rate_mode >=0 and learning_rate_mode <=1, f"Error: Learning_rate_mode must be between 0 and 1!, the entered is {learning_rate_mode}"


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
                w_changes += np.matrix((Y[j] - pred) * X[j]).T
            
            weights += learning_rate * w_changes

            ## calculate the error each time
            Y_pred = self.__LRGD_discriminant_func(weights, X)
            error = self.__MSE(Y.values, Y_pred=Y_pred)
            # print(f'   Mean Squared Error {error}')
            errors_arr.append(error)

        return weights, errors_arr


    def solve_incremental(self, W, iter = 1000, normalize = True, partial = 0, learning_rate_mode = 0):
        """
        Incremental Learning for Logistic Regression
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
            the learned weights for Logistic regression 
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
            update_term = Y[data_index] - self.__LRGD_discriminant_func(W, X.iloc[data_index])

            ## setting the learning rate
            if learning_rate_mode == 0:
                learning_rate = (2/(i+1)) 
            elif learning_rate_mode == 1:
                learning_rate = 1 / np.sqrt(i+1)
            else:
                learning_rate = learning_rate_mode


            update_term = np.multiply(learning_rate * update_term, X.iloc[data_index])

            ## update the weights
            W = np.add(W, update_term)

            ## calculate and save the errors every partial interval
            # if (partial != 0) and (i % partial == 0):
            Y_pred = self.__LRGD_discriminant_func(W, X.T)
            train_error = self.__MSE(Y, Y_pred)
            Y_test_pred, _ = self.predict(W, method=2)
            # print(Y_test_pred)
            test_error = self.__MSE(self.Y_test, Y_test_pred)
                
            ## add train and test error as a tuple
            tuple = [train_error, test_error]

            mse.append(tuple)

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

class report:
    def __init__(self, X, Y, Weights, description) -> None:
        """
        initialize the `X` inputs and `Y` labels and `Weights`
        the `description` is used for the figure file 
        """
        self.X = X
        self.Y = Y
        self.Weights = Weights
        self.description = description

        df_scores = self.__ROC_df()
        self.__export_ROC_Curve(df_scores)

    def __ROC_df(self):
        """
        Find the confusion matrix for each threshold (ROC)

        Returns:
        ---------
        df_scores : pandas dataframe
            dataframe contains the confusion matrix for different thresholds
        """

        scores = []
        y = self.Y
        y_pred = self.__LRGD_discriminant_func(self.Weights, self.X.T)

        ## find the values for each threshold
        ## TP -> True Positive
        ## TN -> True Negative
        ## FP -> False Positive
        ## FN -> False Negative
        for threshold in np.linspace(0, 1, 50):
            TP = ((y_pred >= threshold) & (y == 1)).sum()
            TN = ((y_pred <= threshold) & (y == 0)).sum()
            FP = ((y_pred > threshold) & (y == 0)).sum()
            FN = ((y_pred < threshold) & (y == 0)).sum()

            scores.append([threshold, TP, TN, FP, FN])
        
        df_cols = ['threshold', 'TP', 'TN', 'FP', 'FN']
        df_scores = pd.DataFrame(scores, columns=df_cols)

        ## sensitivity and specificity
        ## True Positive rate = sensitivity
        df_scores['sens'] = df_scores.TP / (df_scores.TP + df_scores.FN)
        ## False Positive rate = 1 - specificity
        df_scores['1-spec'] = df_scores.FP / (df_scores.FP + df_scores.TN)
        
        # print(df_scores)

        return df_scores

    def __export_ROC_Curve(self, df_scores):
        """
        save the roc curve using the dataframe scores
        """

        plt.plot(df_scores['sens'], df_scores['1-spec'])
        plt.title(f'ROC Curve for {self.description}')
        plt.savefig(f'main_2_5_ROC_{self.description}.png')
        plt.close()


    def __LRGD_discriminant_func(self, w, x):
        """
        the discriminant function for Logistic Regression
        the exponential function is used

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
        y = 1 / (1 + np.exp(-w.T @ x))
        ## remove infinity values (caused by zero division)
        y = np.where(y == inf, 1, y)

        return y

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
    print('Logistic Regression on classification dataset!')

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

    ##################################### Closed Form Solution #####################################
    print('\033[1mClosed Form Training: \033[0m')

    ## load Logistic regression class
    lr = LR(X_train, Y_train, X_test, Y_test)
    print(' Train Error: ')
    w = lr.solve()
    ## Prediction of Test set
    print(' Test Error: ')
    Y_pred_test, Y_pred_test_mse = lr.predict(w, method=2)
    print(f'  {Y_pred_test_mse}')

    print(' Final Weights of the closed form training')
    print(w)
    
    print('----------' * DELIMITER_SIZE)
    print('----------' * DELIMITER_SIZE)

    ##################################### Gradient Descent: Logistic Regression #####################################
    
    print(f'\033[1mGradient descent For Logistic Regression with T={iterations} iterations\033[0m')
    initial_weights = np.ones((len(X_train.columns), 1))
    
    print(' Train Errors: ')
    weights, offline_GD_mse= lr.solve_LRGD(X_train.T, Y_train.T, initial_weights, max_iter=iterations)
    print('  ', offline_GD_mse[-1:])
    print(' Test Error: ')
    Y_pred_test_Offline, offline_GD_test_mse = lr.predict(weights, method=2)
    print('  ', offline_GD_mse[-1:])

    print('\n Final wights of Offline Gradient Descent Learning')
    print(weights)

    report(X_train, Y_train, weights, 'Gradient_descent_logisticRegression')
    
    print('----------' * DELIMITER_SIZE)
    print('----------' * DELIMITER_SIZE)

    ##################################### Online Gradient Descent : Logistic Regression #####################################
    print('\033[1mIncremental Learning (Online GD) \033[0m')
    ## the initial weights
    initial_weights = np.zeros(len(X_train.columns))

    print('  Training Error:')
    ## the IL stands for Incremental Learning
    w_IL, mse = lr.solve_incremental(initial_weights, partial = 50, learning_rate_mode=learning_rate_mode, iter=iterations)
    print(f'   {mse[-1:]}')

    print('  Test Error:')
    ## prediction of the test set Incremental Learning
    Y_pred_test_IL, Y_pred_test_IL_mse = lr.predict(w_IL, method=2)
    print(f'   {Y_pred_test_IL_mse}')
    report(X_train, Y_train, w_IL, 'Online_Gradient_descent_logisticRegression')

    

    print('\n Final wights of Incremental Learning')
    print(w)

    print('----------' * DELIMITER_SIZE)


    ## plot loss
    if mse:
        # print("Plotting Training and Test Loss")
        mse = np.array(mse)

        xticks = np.linspace(0, iterations, iterations)

        fig, axes = plt.subplots(1, 3, figsize=(21, 5))
        axes[0].plot(xticks ,mse[:,0])
        axes[0].legend(['Online GD Training MSE'])

        axes[1].plot(xticks ,mse[:, 1])
        axes[1].legend(['Online GD Test MSE'])

        axes[2].plot(xticks ,offline_GD_mse)
        axes[2].legend(['offline GD MSE'])

        learning_rate_equation = ''
        if learning_rate_mode == 0:
            learning_rate_equation = 'Learning Rate: 2/t'
        elif learning_rate_mode == 1:
            learning_rate_equation = 'Learning Rate: 1/sqrt(t)'
        else:
            learning_rate_equation = f'Static Learning Rate: {learning_rate_mode}'


        fig.suptitle(f'Online Logistic Regression with Extended dataset\n{learning_rate_equation}')

        file_name = learning_rate_equation.replace(' ', '_')
        file_name = file_name.replace('/', '-')
        plt.savefig(f'main2_5_{file_name}.jpg')
        plt.close()



