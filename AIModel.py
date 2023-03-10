import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import tensorflow as tf

# Load the data
data = pd.read_csv('Acworth_GA_onehot.csv')

# Preprocess the data
X = data[['sqft', 'baths', 'garage', 'stories']]
y = data['Sold_price']

# One-hot encode the tags
X = pd.get_dummies(X, columns=['baths', 'garage', 'stories'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)

# Make predictions on new data
new_data = pd.DataFrame({'sqft': [1500], 'baths': [1], 'garage': [0], 'stories': [0]})
new_data = pd.get_dummies(new_data, columns=['baths', 'garage', 'stories'])
new_data = scaler.transform(new_data)
predictions = model.predict(new_data)
print(predictions)

'''
# Open the original csv file
with open('Acworth_GA_onehot.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # save the header row

    # Shuffle the data
    data = list(reader)
    random.shuffle(data)

    # Calculate the split point
    split_point = int(len(data) * 0.8)

    # Write the data to the train file
    with open('TrainingData.csv', 'w', newline='') as train_file:
        writer = csv.writer(train_file)
        writer.writerow(header)  # write the header row
        writer.writerows(data[:split_point])  # write the training data

    # Write the data to the test file
    with open('TestingData.csv', 'w', newline='') as test_file:
        writer = csv.writer(test_file)
        writer.writerow(header)  # write the header row
        writer.writerows(data[split_point:])  # write the testing data
'''

# Adds All Unique Tags to UniqueTags.csv file
'''
with open('Acworth_GA.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    
    # Loop through the rows of the file
    unique_tags = set()
    for row in reader:
        tags = row[-1][1:-1].split(', ')  # Extract tags from last column
        unique_tags.update(tags)  # Add tags to set of unique tags
    
    # Write out the unique tags to a new file
    with open('UniqueTag.csv', 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        for tag in sorted(unique_tags):
            writer.writerow([tag])
'''
# Load the CSV file into a pandas DataFrame
# df = pd.read_csv('UniqueTags.csv')

'''

# Preprocess the data
X = df.drop(['sold_price'], axis=1)  # drop the target variable
y = df['sold_price']  # set the target variable as the label

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mse',
              optimizer=tf.keras.optimizers.RMSprop(0.001),
              metrics=['mae', 'mse'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# Evaluate the model on the test set
loss, mae, mse = model.evaluate(X_test, y_test, verbose=2)
print("Testing set Mean Absolute Error: {:5.2f} Sold Price".format(mae))

# Make predictions on new data
new_data = pd.DataFrame({'property_id': [12345], 'is_new_construction': [True], 'is_for_rent': [False], 
                         'is_subdivision': [True], 'is_contingent': [False], 'is_price_reduced': [False],
                         'is_pending': [False], 'is_foreclosure': [False], 'is_plan': [False], 
                         'is_coming_soon': [False], 'is_new_listing': [True], 'year_built': [2000], 
                         'baths_3qtr': [2], 'sold_date': ['2022-12-01'], 'baths_full': [2], 
                         'name': ['My Property'], 'baths_half': [0], 'lot_sqft': [10000], 
                         'sqft': [2000], 'baths': [2], 'sub_type': ['single_family'], 
                         'baths_1qtr': [0], 'garage': [2], 'stories': [2], 'beds': [3], 
                         'type': ['residential'], 'list_date': ['2022-11-01T12:00:00Z'], 
                         'list_price': [500000], 'status': ['active'], 'postal_code': [30101], 
                         'state': ['Georgia'], 'state_code': ['GA'], 'city': ['Acworth'], 
                         'county': ['Cobb'], 'address_line': ['1234 Main St'], 
                         'latitude': [34.057236], 'longitude': [-84.678534], 
                         'tags': ['central_air', 'community_clubhouse', 'community_outdoor_space']})
new_predictions = model.predict(new_data)
print(new_predictions)
'''

'''
# read the data from the file
with open('Acworth_GA.csv') as file:
    reader = csv.DictReader(file)
    data = [row for row in reader]

# separate the tags and create a set of unique tags
tags = set()
for row in data:
    row_tags = row['tags'].strip('[]').replace("'", "").split(', ')
    row['tags'] = row_tags
    tags.update(row_tags)

# create a new CSV file with one-hot encoded tags
with open('Acworth_GA_onehot.csv', 'w', newline='') as file:
    fieldnames = list(data[0].keys()) + list(tags)
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        # create a dictionary with one-hot encoded tags
        tag_dict = {tag: int(tag in row['tags']) for tag in tags}
        # combine the original row data with the one-hot encoded tag dictionary
        new_row = {**row, **tag_dict}
        writer.writerow(new_row)
'''

