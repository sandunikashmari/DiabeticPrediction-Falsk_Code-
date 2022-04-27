import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pickle

diabetics_data = pd.read_csv('diabetes.csv')

print(diabetics_data.head())

print(diabetics_data.shape)

#Replace Zero Values with their Mean Values
diabetics_data['Glucose']=diabetics_data['Glucose'].replace(0,diabetics_data['Glucose'].mean())
diabetics_data['BloodPressure']=diabetics_data['BloodPressure'].replace(0,diabetics_data['BloodPressure'].mean())
diabetics_data['SkinThickness']=diabetics_data['SkinThickness'].replace(0,diabetics_data['SkinThickness'].mean())
diabetics_data['Insulin']=diabetics_data['Insulin'].replace(0,diabetics_data['Insulin'].mean())
diabetics_data['BMI']=diabetics_data['BMI'].replace(0,diabetics_data['BMI'].mean())

x = diabetics_data.drop(columns='Outcome', axis=1)
y = diabetics_data['Outcome']

#Feature Scaling
scalar = StandardScaler()
standarized_data = scalar.fit_transform(x)

x = standarized_data
y = diabetics_data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

#Instantiate the Model
classifier = svm.SVC(kernel='linear')

#Fit the Model
classifier.fit(x_train, y_train)

#Make the Pickle file of our Model
pickle.dump(classifier, open("model.pkl", "wb"))