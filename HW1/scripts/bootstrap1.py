from ast import arguments
import pandas as pd
from numpy.random import randint
import sys

def bootstrap1(data):
    """
    Demonstrate one iteration of bootstraping method (it is a with replacement method)

    INPUT:
    -------
    data: a pandas dataframe, containing our data

    OUPUTS:
    --------
    train_data: pandas dataframe of a sample data
    test_data: pandas dataframe of sample data, the data that are not included in train_data
    """
    ## find the length of our data (how many data rows we have)
    data_length = len(data)
    
    ## the indexes to be chosen from original data 
    indexes = randint(data_length, size=data_length)
    
    ## create the training set
    train_data = df.iloc[indexes].copy()

    ## choose the test set, (The data that is omited from training set)
    test_data = pd.concat([data,train_data, train_data]).drop_duplicates(keep=False)

    return train_data, test_data


if __name__ == '__main__':

    arguments = sys.argv

    ## arguments[1] is the directory of our dataset
    df = pd.read_csv(arguments[1])
    
    bootstrap_train, bootstrap_test = bootstrap1(df)