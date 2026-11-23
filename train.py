#STEP1:Import libraries

import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScalar
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KneiborsClassifier
from sklearn.tree import Decision TreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm importSVC
from sklearn.metrics import(
f1_score,
accuracy_score,
classification_report,
confusion_matrix
)
#STEP2:load_dataset

data=load_breast_cancer()

X=pd.DataFrame(
data.data,
columns=data.feature_names
)
y=pd.Series(data.target,name="target")

print("Dataset shape:",X.shape)
print("\nFirst5rows:")
print(X.head())
print("\n Target values:")
print(y.value_counts())

#STEP3:Split dataset

X_train,X_test,y_train,Y_test=train_test_split(
  X,
  y,
  test_size=0.20,
  random_state=42,
  stratify=y
)
print("\n Training samples:",X_train.shape[0])
print("Testing sample:",X_test.shape[0])

#STEP4:Define classification algorithms
model={

  "Logistic Regression":LogisticRegression(
    max_iter=5000
  ),

  "KNN":KNeighborsClassifier(),

  "Decision Tree":Decision TreeClassifier(
    random_state=42
  ),

  "RandomForest":RandomForestClassifier(
    random-state=42
  ),

  "SVM":SVC()
}

#STEP5:Define hyperparameter grids

param_grids={

  "LogisticRegression":{
    "model__C":[0.1,1,10],
  },

  "KNN":{
    "model__n_neighbors":[3,5,7,9],
    "model__weights":["uniform","distance"]
  },

  "Decision Tree":{
    "model__max_depth":[None,3,5,10],
    "model__min_sample_split":[2,5,10]
  },
  
  "RandomForest":{
    "model
