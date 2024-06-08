# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:45:16 2024

@author: Selvibala
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load datasets
train_data = pd.read_csv(r'C:\Users\Selvibala\Downloads\creditcardfrauddetection\fraudTrain.csv')
test_data = pd.read_csv(r'C:\Users\Selvibala\Downloads\creditcardfrauddetection\fraudTest.csv')

# Drop timestamp column
train_data.drop(columns=['unix_time'], inplace=True)
test_data.drop(columns=['unix_time'], inplace=True)

# Separate features and target variable
X_train = train_data.drop(columns=['category'])
y_train = train_data['category']
X_test = test_data.drop(columns=['category'])
y_test = test_data['category']

# Define numerical columns
numerical_columns = X_train.columns
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns)])

# Decision Trees
dt_classifier = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', DecisionTreeClassifier())])
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\nDecision Trees:")
print("Accuracy:", dt_accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_predictions))
print("Classification Report:")
print(classification_report(y_test, dt_predictions))