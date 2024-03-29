#!/usr/bin/env python3

# Imports
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def rf(input1, prediction_feature, num_runs, print_predictions, print_importances):
    #Reading in features
    features = pd.read_csv(input1, index_col=False)
    features = features.dropna(axis="columns", how="any")

    '''
    #This prints the total number of rows and features (columns)
    print('The shape of our features is:', features.shape)
    # Descriptive statistics for each column
    print('numbers of columns: ', len(features.columns))
    with pd.option_context('display.max_columns', 40):
        print(features.describe(include='all'))
    '''


    # Remove the labels from the features
    labels = np.array(features[prediction_feature])
    # axis 1 refers to the columns

    #Mok Data
    '''
    features = features.drop('SampleID', axis = 1)
    features = features.drop('GenotypeID', axis = 1)
    features = features.drop('Clearance', axis = 1)# Saving feature names for later use
    '''
    #Zhu data
    features = features.drop('SampleID', axis = 1)
    features = features.drop('SampleID.Pf3k', axis = 1)
    features = features.drop('Sample.collection.time.24hr.', axis = 1)
    features = features.drop('PC90', axis = 1)
    features = features.drop('Parasites.clearance.time', axis = 1)# Saving feature names for later use
    

    #Convert qualitative data into arbitrary values for processing
    features = pd.get_dummies(features)

    #converting to numpy array for ML algorithm
    feature_list = list(features.columns)
    features = np.array(features)

    # print(feature_list)

    '''
    NOTE:
        Idea to run random forest 1000x and keep the one with the best accuracy.
        Also, potentially save the list data of important features and see what is the most important feature across all 100 runs and is it the most important for the largest accuracy run?

    '''

    maccuracy = 0
    mtest_labels = []
    mpredictions = []
    mfeature_importances = 0
    mnum = 0
    feature_importances_list = []
    for i in range(num_runs):
        print('--------------------')
        print(f'Run {i}:')
        # Using Skicit-learn to split data into training and testing sets
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = i)

        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)# Train the model on training data
        rf.fit(train_features, train_labels)

        # Use the forest's predict method on the test data
        predictions = rf.predict(test_features)# Calculate the absolute errors
        errors = abs(predictions - test_labels)# Print out the mean absolute error (mae)
        
        
        if(print_predictions == True):
            #Printing actual values and prediction values
            
            print("\tActual\tPrediction")
            for j in range(len(predictions)):
                print(f'\t{round(test_labels[j], 5)}\t{round(predictions[j], 5)}')
            
        
        #print(f'\tMean Absolute Error: {round(np.mean(errors), 2)}')
        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * (errors / test_labels)# Calculate and display accuracy
        accuracy = 100 - np.mean(mape)
        print(f'\tAccuracy: {round(accuracy, 2)}%')

        #getting numerical feature importances
        importances = list(rf.feature_importances_)# List of tuples with variable and importance
        #creating feature importances tuple from feature list and values of importances
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        #sorting by most importance first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

        if(print_importances == True):
            print('\tVariable\t\tImportance')
            for(variable, value) in feature_importances:
                print(f'\t{variable:20}\t{value}')

        feature_importances_list.append(feature_importances)

        if(accuracy > maccuracy):
            #setting most accurate run number
            mnum = i
            #setting max accuracy
            maccuracy = accuracy
            #setting most accurate test labels and predictions list
            mtest_labels = test_labels
            mpredictions = predictions
            #finding most important feature from most accurate run
            mfeature_importances = feature_importances


    print('--------------------')
    print(f'Max Accuracy of {i+1} Runs was Run {mnum}: {round(maccuracy,2)}%')

    '''
    print(f'Run {mnum} Predictions:')
    print("\tActual\tPrediction")
    for j in range(len(predictions)):
        print(f'\t{round(test_labels[j], 5)}\t{round(predictions[j], 5)}')
    '''

    '''
    print(f'Run {mnum} Importances:')
    print('\tVariable\t\tImportance')
    for(variable, value) in mfeature_importances:
        print(f'\t{variable:20}\t{value}')
    '''

    mvar_list = []
    mvar_list2 = []
    mvar_list3 = []
    for i in feature_importances_list:
        mvar_list.append(i[0][0])
        mvar_list2.append(i[1][0])
        mvar_list3.append(i[2][0])

    
    

    mvar = max(set(mvar_list), key = mvar_list.count) 
    mvar2 = max(set(mvar_list2), key = mvar_list2.count) 
    mvar3 = max(set(mvar_list3), key = mvar_list3.count) 
    print(f"'{mvar}' is the most important feature in {mvar_list.count(mvar)}/{len(mvar_list)} runs.")
    print(f"'{mvar2}' is the second most important feature in {mvar_list2.count(mvar2)}/{len(mvar_list2)} runs.")
    print(f"'{mvar3}' is the second most important feature in {mvar_list3.count(mvar3)}/{len(mvar_list3)} runs.")
    # Import tools needed for visualization
    # from sklearn.tree import export_graphviz
    # import pydot# Pull out one tree from the forest
    # tree = rf.estimators_[5]# Import tools needed for visualization
    # from sklearn.tree import export_graphviz
    # import pydot# Pull out one tree from the forest
    # tree = rf.estimators_[5]# Export the image to a dot file
    # export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)# Use dot file to create a graph                                                                       
    # (graph, ) = pydot.graph_from_dot_file('tree.dot')# Write graph to a png file
    # graph.write_png('tree.png')


    
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    





    # plt.style.use('fivethirtyeight')# list of x locations for plotting
    # x_values = list(range(len(importances)))# Make a bar chart
    # plt.bar(x_values, importances, orientation = 'vertical')# Tick labels for x axis
    # plt.xticks(x_values, feature_list, rotation='vertical')# Axis labels and title
    # plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')


    # Use datetime for creating date objects for plotting
    # import datetime# Dates of training values
    # months = features[:, feature_list.index('month')]
    # days = features[:, feature_list.index('day')]
    # years = features[:, feature_list.index('year')]# List and then convert to datetime object
    # dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
    # dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]# Dataframe with true values and dates
    # test_samples1 = features[:, feature_list.index('RNA')]
    # print(len(test_samples1))
    # true_data = pd.DataFrame(data = {'sample': test_samples1, 'actual': labels})# Dates of predictions
    # # months = test_features[:, feature_list.index('month')]
    # # days = test_features[:, feature_list.index('day')]
    # # years = test_features[:, feature_list.index('year')]# Column of dates
    # # test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]# Convert to datetime objects
    # # test_dates = [datetime.datetime.strptime(sample, '%Y-%m-%d') for date in test_dates]# Dataframe with predictions and dates
    # test_samples2 = test_features[:, feature_list.index('GenotypeID')]

    # predictions_data = pd.DataFrame(data = {'sample': test_samples2, 'prediction': predictions})# Plot the actual values
    # plt.plot(true_data['sample'], true_data['actual'], 'b-', label = 'actual')# Plot the predicted values
    # plt.plot(predictions_data['sample'], predictions_data['prediction'], 'ro', label = 'prediction')
    # plt.xticks(rotation = '60'); 
    # plt.legend()# Graph labels
    # plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values')

    
    fig, ax = plt.subplots()
    ax.plot(range(len(mtest_labels)), mtest_labels,label = "Actual Clearence")
    ax.plot(range(len(mpredictions)), mpredictions,label = "Predictions")
    #ax.scatter(range(len(test_labels)), test_labels,label = "Actual Clearence")
    #ax.scatter(range(len(predictions)), predictions,label = "Predictions")
    ax.set(xlabel='Patients', ylabel='Clearance', title='Zhu Data Clearence Predictions')
    ax.legend()



    #plt.plot(range(262), test_labels, 'b-', label = 'actual')# Plot the predicted values
    #plt.plot(range(262), predictions, 'ro', label = 'prediction')

    plt.show()
    



if __name__ == "__main__":

    #i1 = input("Please enter the path to the .csv file for analysis.\n")
    #i2 = input("Please enter the feature you would like to predict.\n")
    #i3 = input("Please enter the number of prediction runs you would like to conduct.\n")
    #i3 = int(i3)
    #i4 = input("Would you like to print the prediction values for each datum (y or n)?\n")
    '''
    if(i4 == 'y'):
        i4 = True
    else:
        i4 = False
    '''
    print(sys.argv)
    i1 = 'malaria_data/zhu_meta.csv'
    i2 = 'Parasites.clearance.time'
    i3 = 10
    i4 = False
    i5 = False
    rf(i1, i2, i3, i4, i5)