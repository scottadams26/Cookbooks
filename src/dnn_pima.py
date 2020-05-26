################################################################################
# Example DNN For Diabetes Classification Using Pima Diabetes Data
################################################################################

#===========#
# Libraries #
#===========#

import os
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense


#=====================#
# Load and Split Data #
#=====================#

pima = pd.read_csv("data/pima_diabetes.csv")
features = pima.drop("Outcome", axis=1)
target = pima["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=0)


#========================#
# Feature Transformation #
#========================#

num_features = features.select_dtypes(include=['int64', 'float64']).columns
cat_features = features.select_dtypes(include=['object']).columns

num_steps = [
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
]

cat_steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
]

num_transformer = Pipeline(steps=num_steps)
cat_transformer = Pipeline(steps=cat_steps)

transformers = [
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
]

preprocessor = ColumnTransformer(transformers=transformers)
X_trainPrep = preprocessor.fit_transform(X_train)
X_testPrep = preprocessor.transform(X_test)


#=====#
# DNN #
#=====#

model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])

history = model.fit(
  X_trainPrep,
  y_train,
  validation_data = (X_testPrep, y_test),
  epochs=100,
  batch_size=10)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()


#=======================================#
# Predictions and Classification Report #
#=======================================#

predProbs = model.predict(X_testPrep)
rounded = [round(x[0]) for x in predictions]
predClass = model.predict_classes(X_testPrep)
report = classification_report(y_test, predClass, output_dict=True)

pd.DataFrame(report).transpose().to_csv(
  "output/pimaDnnReport_" + dt.datetime.today().strftime("%Y%m%d") + ".csv")


#============#
# Save Model #
#============#

model.save_weights(
  "models/pimaDnnWeights"+ dt.datetime.today().strftime("%Y%m%d") + ".h5")

with open(
  "models/pimaDnnArchitecture"
  + dt.datetime.today().strftime("%Y%m%d") + ".json",
  "w") as f:
    f.write(model.to_json())


#==============#
# Reload Model #
#==============#

# from keras.models import model_from_json

# with open("models/pimaDnnArchitecture20200526.json", "r") as f:
#   model = model_from_json(f.read())

# model.load_weights("models/pimaDnnWeights20200526.h5")
