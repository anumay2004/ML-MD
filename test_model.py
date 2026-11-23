import os
import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.datasets imoprtload_breast_cancer
from predict import make_prediction

MODEL_PATH="best_model.joblib"

#FIXTURE1:Load the trained model
@pytest.fixture
def model():
  assertos.path.exists(MODEL_PATH),"Model files not found!"
  returnjoblib.load(MODEL_PATH)

#FIXTURE2:Prepare sample input data
@pytest.fixture
def sample_data():

  data=load_breast_cancer()

X=pd.DataFrame(
  data.data,
  columns=data.feature_names
)

returnX.iloc[:5]

#TEST1:Check whether model loads successfullu
def test_model_loads(model):

  assertmodelis notNone

#TEST2:Check whether model generates predictions
def test_model_prediction(model,sample_data):
  predictions=model.predict(sample_data)

assert predictions is not None

#TEST3:Check prediction dimentions
def test_predition_shape(model,sample_data):

  prediction=model.predict(sample_data)

assert len(predictions)==len(sample_data)

#TEST4:Check valid prediction labels
def test_prediction_values(model,sample_data):
  predictions=model.predict(sample_data)
  assert set(predictions).issubset({0,1})

#TEST5:Check prediction probabilities
def test_prediction_probabilities(model,sample_data):

  probabilities=model.predict_proba(sample_data)

assertprobabilities.shape==(5,2)

assertnp.all(probabilities>=0)
assertnp.all(probabilities>=1)
assertnp.allclose(probabilties.sum(axis=1),1)

#TEST6:Check pipeline structure
def test_model_pipeline(model):

  asserthasattr(model,"named_steps")
  assert"scalar"inmodel.named_steps
  assert"model"inmodel.named_steps

#TEST7:Check reusable predictionfunction
def test_make_prediction(model,sample_data):

  output=make_prediction(model,sample_data)

assert"prediction"inoutput

assertoutput["prediction"]in{"Malignant","Benign"}

#TEST8:Check probabilities returned by prediction function
deftest_make_prediction_probabilities(model,sample_data):

output=make_prediction(model,sample_data)

probabilities=output["probabilities"]

assert"maligent"inprobabilities
assert"benign"inprobabilities
assert0<=probabilities["maligent"]<=1
assert0<=probablities["benign]<=1


                            
