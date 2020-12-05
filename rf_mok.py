# Pandas is used for data manipulation
import pandas as pd# Read in data and display first 5 rows
# Use numpy to convert to arrays
import numpy as np# Labels are the values we want to predict

from sklearn.model_selection import train_test_split# Split the data into training and testing sets

features = pd.read_csv('mok_meta.csv', index_col=False)
# print(features.head(5))

print('The shape of our features is:', features.shape)

# Descriptive statistics for each column
# print(features.describe())


labels = np.array(features['Clearance'])# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('Clearance', axis = 1)# Saving feature names for later use

# One-hot encode the data using pandas get_dummies
# print(features.iloc[:,5:].head(5))

features = pd.get_dummies(features)# Display the first 5 rows of the last 12 columns

feature_list = list(features.columns)# Convert to numpy array
features = np.array(features)

# print(feature_list)

# Using Skicit-learn to split data into training and testing sets

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# # The baseline predictions are the historical averages
# baseline_preds = test_features[:, feature_list.index('average')]# Baseline errors, and display average baseline error
# baseline_errors = abs(baseline_preds - test_labels)print('Average baseline error: ', round(np.mean(baseline_errors), 2))Average baseline error:  5.06 degrees.

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)# Train the model on training data
rf.fit(train_features, train_labels)


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)# Calculate the absolute errors
errors = abs(predictions - test_labels)# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))


print("actual     prediction")
for i in range(len(predictions)):
    print(test_labels[i], predictions[i])

# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances] 

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


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



# Get numerical feature importances
# importances = list(rf.feature_importances_)# List of tuples with variable and importance
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]# Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = False)# Print out the feature and importances 
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 




# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt

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

plt.plot(range(262), test_labels, 'b-', label = 'actual')# Plot the predicted values
plt.plot(range(262), predictions, 'ro', label = 'prediction')

plt.show()



