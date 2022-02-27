from ast import arguments
import pandas as pd
import sys

def kfold_crossvalidation(data, k, m):
    """
    K-fold cross validation 
    Note: test data is equavalent the validation data in normal machine learning models (Because it is used to evaluate one model)

    INPUTS:
    --------
    data: pandas dataframe containing feature vectors as rows
    k: positive integer, the number of folds
    m: target output 
    train_split:  the fraction of data that is used to split for training, default is 0.7

    OUTPUTS:
    ---------
    training_data: multi-dimensinal array of training data, each index contains the dataset for K-fold number
    test_data: multi-dimensinal array of test data, each ```index+1``` contains the dataset for each K-fold number
    """
    ## get the length of data to split it
    dataframe_size = len(data)

    ## find the length of each split
    # training_size = int(dataframe_size * train_split)
    # test_size = dataframe_size - training_size

    ## empty arrays to save data into it
    training_data = []
    test_data = []

    ## find the split size
    split = int(dataframe_size / k)

    ## split the data into k-fold and add the folds into the arrays
    for i in range(k):
        start_idx = int(i*split)
        end_idx = int((i+1)*split)

        test = data.iloc[start_idx:end_idx].copy()
        ## add the label column to corresponding index
        test['label'] = m.iloc[start_idx: end_idx]
        
        
        ## choose other part of dataset as train
        train = pd.concat([data, test, test]).drop_duplicates(keep=False)
        train['label'] = m.iloc[train.index]
        
        training_data.append(train)
        test_data.append(test)

    return training_data, test_data


if __name__ == '__main__':
    arguments = sys.argv
    assert len(arguments) == 3, f"[ERROR] 2 argument must be added as dataset directory! input arguments count {len(arguments)}"
    

    ## arguments[1] is the directory of our dataset
    ## NOTE: label column name must be in the dataset
    df = pd.read_csv(str(arguments[1]))

    ## arguments[2] is the K value of K-fold
    K = int(arguments[2])

    ## the data without target value
    data = df.drop('label')
    train_set, test_set = kfold_crossvalidation(data, K, df.label )


