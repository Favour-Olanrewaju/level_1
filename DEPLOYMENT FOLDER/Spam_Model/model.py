# loading the libraries 
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# building the pipeline
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 

# for preprocessing the data
from sklearn.feature_extraction.text import CountVectorizer #to convert text data into numerical data

# models
from sklearn.ensemble import RandomForestClassifier 

# splitting dataset
from sklearn.model_selection import train_test_split

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report


# loading the dataset
spam_detection = pd.read_csv("C:\\Datasets\\spam.csv")
spam_detection.rename(columns={'label':'Target', 'text':'Text'}, inplace=True)
# spam_detection.head()

# checking for duplicates
# print(spam_detection.duplicated().value_counts())
# print(f'Shape: {spam_detection.shape}')

# removing duplicate values
spam_detection = spam_detection.drop_duplicates(keep='first')
# print(spam_detection.duplicated().value_counts())
# print(f'Shape: {spam_detection.shape}')

# description of the dataset
# spam_detection.info()

# checking for missing values 
# spam_detection[spam_detection.isnull()].any()

# spam_detection['Target'].value_counts()

# data visualization  using pie
plt.pie(spam_detection['Target'].value_counts(), labels=['ham', 'spam'], autopct='%0.2f')

# getting the target and vector
X = spam_detection[['Text']]
y = spam_detection['Target']
# print(X.shape)
# print(y.shape)

# preprocessing Target
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder() 
y = encoder.fit_transform(y)
# y

# 0-ham, 1-spam

# splitting into train and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, random_state=42)

# preprocessing Vector alongside with pipeline initialization 
cat_transformer = CountVectorizer()

# ColumnTransformer 
preprocessor = ColumnTransformer(transformers=[('cat', cat_transformer, 'Text')])

# pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor ), ('classifier', RandomForestClassifier(random_state=42))])
pipeline.fit(Xtrain, ytrain)



# #############################################
# make predictions
y_pred = pipeline.predict(Xtest)

print('prediction:', y_pred)
print('actual:', ytest)

# accuracy metrices
print("Accuracy:", accuracy_score(ytest, y_pred))
print("\nClassification Report:\n", classification_report(ytest, y_pred))
# pipeline



from sklearn.metrics import confusion_matrix
import seaborn as sns
mat = confusion_matrix(ytest, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value');
plt.ylabel('true value');

##############################################

# creating a pickle file
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
    
print('model saved as model.pkl')

