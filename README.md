# msbd5001indproject
This repository holds project files for the MSBD 5001 Kaggle in-class competition assignment

## Programming language used: 
Python 

## Packages required: 
**Pandas** - For operations on the provided input csv files, <br />
**Sklearn** - To create and run machine learning models for prediction, <br />
**Warnings** - To suppress warnings generated just so that it doesn't clutter the printed output statements <br />

## Steps to run the program:
* All the code can be found in ***main.py*** file <br />
* The code can be run on any machine that has Anaconda/Spyder installed on it <br />
* Create a folder on the machine where you plan to run the program and download the ***main.py*** file into this folder <br />
* Also place the given ***train.csv*** and ***test.csv*** files into the newly created folder <br />
* Run the program <br />
* Output is printed on iPython console in Spyder (screenshot of the result is attached at the bottom of this page) <br />
* A ***submission.csv*** file is created in the folder <br />

## Code

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 18:03:30 2018

@author: vikramphilar-mm
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")


####################################################################################
# Function Name -   ReadInputFiles
#
# Parameters -      None
#
# Output -          trainDf,
#                   yLabel,
#                   testDf
#
# Description -     This function reads the train.csv and test.csv files into 
#                   two dataframes, trainDf and testDf. It also reads the ylabels of
#                   training data into yLabel
####################################################################################
def ReadInputFiles():
    
    trainDf = pd.read_csv('train.csv', usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], index_col=0)
    testDf = pd.read_csv('test.csv', usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], index_col=0)
    yLabel = pd.read_csv('train.csv', usecols=[14])
    
    return (trainDf, yLabel, testDf)



####################################################################################
# Function Name -   FeatureEngg
#
# Parameters -      trainDf 
#
# Output -          labelEncodeObj, 
#                   ord0
#
# Description -     This function performs feature engineering on all the 
#                   features of the training data. The same methods are applied
#                   to the test data. The 'penalty', 'alpha' and 'n_jobs' are 
#                   taken as categorical data. These columns are label encoded
#                   and then one-hot encoded. 
####################################################################################
def FeatureEngg(labelEncodeObj, tempDf):

    tempDf['n_jobs'] = tempDf['n_jobs'].apply(lambda x:1000 if x == -1 else x)

    X = tempDf.iloc[:,:].values
    
    X[:, 0] = labelEncodeObj.fit_transform(X[:, 0])
    X[:, 2] = labelEncodeObj.fit_transform(X[:, 2])
    X[:, 5] = labelEncodeObj.fit_transform(X[:, 5])
    Z = pd.DataFrame(X)
    
    oneHotEncoderObj = OneHotEncoder(categorical_features=[0])  # 'penalty'
    temp0 = oneHotEncoderObj.fit_transform(Z).toarray()
    
    oneHotEncoderObj = OneHotEncoder(categorical_features=[2])  # 'alpha'
    temp2 = oneHotEncoderObj.fit_transform(temp0).toarray()

    oneHotEncoderObj = OneHotEncoder(categorical_features=[5])  # 'n_jobs'
    temp5 = oneHotEncoderObj.fit_transform(temp2).toarray()
    tempDf = pd.DataFrame(temp5)
    
    return (tempDf)



####################################################################################
# Function Name -   PrepTrainingData
#
# Parameters -      trainDf 
#
# Output -          labelEncodeObj, 
#                   ord0
#
# Description -     This function prepares the training data by encoding certain 
#                   columns. The feature engineering of training data is carried out 
#                   by the FeatureEngg() function
####################################################################################
def PrepTrainingData(trainDf):

    labelEncodeObj = LabelEncoder()    
    ord0 = FeatureEngg(labelEncodeObj, trainDf)
       
    return (labelEncodeObj, ord0)



####################################################################################
# Function Name -   PrepTestingData
#
# Parameters -      labelEncodeObj, 
#                   testDf 
#
# Output -          newTestDf
#
# Description -     This function prepares the testing data by encoding (similar to
#                   encoding training data) certain columns. The feature engineering 
#                   of testing data is carried out by the FeatureEngg() function
####################################################################################
def PrepTestingData(labelEncodeObj, testDf):

    newTestDf = FeatureEngg(labelEncodeObj, testDf)
    
    return (newTestDf)
    


####################################################################################
# Function Name -   CreateAndRunModel
#
# Parameters -      estimator,
#                   labelEncodeObj, 
#                   newTrainDf, 
#                   yLabel, 
#                   newTestDf
#
# Output -          X_train_minmax_data, 
#                   X_test_minmax_data, 
#                   y_train, 
#                   y_test, 
#                   predTest, 
#                   predictions
#
# Description -     This function creates and runs the model to train with 75% of data
#                   and then predict 'time' off the 25% test data set. Next, the 
#                   already encoded test.csv data held by newTestDf is used for prediction.
#                   The data is scaled using MinMaxScaler because the data consists
#                   of features that are of varying scales.
####################################################################################
def CreateAndRunModel(estimator, labelEncodeObj, newTrainDf, yLabel, newTestDf):
    
    X_train, X_test, y_train, y_test = train_test_split(newTrainDf, yLabel.values.ravel(), test_size=0.25, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_minmax_data = scaler.fit_transform(X_train)
    X_test_minmax_data = scaler.transform(X_test)
    
    estimator.fit(X_train_minmax_data, y_train)
    predTest = estimator.predict(X_test_minmax_data)
    
    test_minmax_data = scaler.transform(newTestDf)
    predictions = estimator.predict(test_minmax_data)
            
    return (X_train_minmax_data, X_test_minmax_data, y_train, y_test, predTest, predictions)



####################################################################################
# Function Name -   PrintResults
#
# Parameters -      name, 
#                   estimator,
#                   X_train_minmax_data, 
#                   X_test_minmax_data, 
#                   y_train, 
#                   y_test, 
#                   predTest
#
# Description -     This function prints out the training and testing accuracy
#                   along with the mean_squared_error
####################################################################################
def PrintResults(name, estimator, X_train_minmax_data, X_test_minmax_data, y_train, y_test, predTest):
    
    print ('"' + name + '" -->', 'Training Accuracy:', str(int(estimator.score(X_train_minmax_data, y_train)*100)) + str('%') 
     + '.', 'Testing Accuracy:', str(int(estimator.score(X_test_minmax_data, y_test)*100)) + str('%') 
     + '.', 'MSE:', "{:.2f}".format(mean_squared_error(y_test,predTest)) + '.')
    


####################################################################################
# Function Name -   CreateSubmissionFile
#
# Parameters -      predictions
#
# Description -     This function writes predictions into submission.csv file
####################################################################################
def CreateSubmissionFile(predictions):
    # Copy the results from the prediction into result.csv file
    with open('submission.csv', 'w') as f:
        f.write('Id,time\n')
        for idn, item in enumerate(abs(predictions)):
            f.write('%s,%s\n' % (idn, "{:0.2f}".format(*item.flatten())))



####################################################################################
# Function Name -   main
#
# Description -     This is the main function and the starting point of the program.
#                   The following steps are followed:
#
#                   (a) Read the given training and testing csv files, ReadInputFiles()
#               
#                   (b) Prepare training data from PrepTrainingData()
#
#                   (c) Prepare testing data from PrepTestingData()
#
#                   (d) Feature engineering is performed on both, training and 
#                       testing data by calling the FeatureEngg() function from 
#                       the PrepTrainingData() and PrepTestingData() functions
#
#                   (e) For a list of estimators, create and use the model for prediction
#                       from the CreateAndRunModel() function iteratively
#
#                   (f) For each of the estimators, print out the accuracy results
#                       on the training and testing data along with Mean Squared Error (MSE)
#
#                   (g) If MSE of an algorithm is better than the 
#                       previously calculated MSE, then create submission.csv file 
#                       and write predicted values in it
####################################################################################
def main():
    
    print ('STEP 1] Reading train.csv and test.csv input files...')
    trainDf, yLabel, testDf = ReadInputFiles()
    
    print ('\n')
    print ('STEP 2] Prepare training data...')
    labelEncodeObj, newTrainDf = PrepTrainingData(trainDf)
    
    print ('\n')
    print ('STEP 3] Prepare testing data...')
    newTestDf = PrepTestingData(labelEncodeObj, testDf)
    
    print ('\n')
    print ('STEP 4] Create model and predict...')

    random_state = 42
    
    ESTIMATORS = { 
        "Random Forest" : RandomForestRegressor(random_state=random_state),
        "MLP Regressor" : MLPRegressor(activation='relu', 
                                       solver='adam', 
                                       hidden_layer_sizes=(250,250,250), 
                                       max_iter=10000, 
                                       random_state=random_state, 
                                       alpha=11, 
                                       warm_start=True, 
                                       learning_rate_init=0.0001, 
                                       verbose=False),
        "Linear Regressor" : LinearRegression(),
        "Gradient Descent Regressor" : SGDRegressor(random_state=random_state)                                     
    }
    
    print ('\n')
    print ('STEP 5] Printing results and creating submission.csv for the model with the best MSE...')
    print ('\n')
    
    bestMSE = 100
    bestModel = None
    
    for name, estimator in ESTIMATORS.items():
        
        X_train_minmax_data, X_test_minmax_data, y_train, y_test, predTest, predictions = CreateAndRunModel(estimator, labelEncodeObj, newTrainDf, yLabel, newTestDf)
            
        PrintResults(name, estimator, X_train_minmax_data, X_test_minmax_data, y_train, y_test, predTest)

        MSE = mean_squared_error(y_test,predTest)

        if MSE < bestMSE:
            print ('MSE of "' + name + '" is better compared to "' + str(bestModel) + '".')
            print ('Overwrite submission.csv file...','\n')
            CreateSubmissionFile(predictions)
            bestMSE = MSE
            bestModel = name          
        else:          
            print ('MSE of "' + name + '" is NOT better compared to "' + str(bestModel),'\n')

    print ('\n')
    print ('FINAL RESULT:')
    print ('Iteration through list of estimators is complete.')
    print ('"' + str(bestModel) + '"', 'has the best MSE:', "{:.2f}".format(bestMSE))
    


if __name__ == "__main__":
    main()
```

## Output
![Result](https://github.com/vikramphilar/msbd5001indproject/blob/master/result_screenshot.png)
