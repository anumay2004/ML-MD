#predict.py

importjoblib
importpandasaspd

fromsklearn.datasetsimportload_breast_cancer

#STEP1:Loadthesavedmodel
defload_model(model_path="bestmodel.joblib"):
 model=joblib.load(model_path)
 returnmodel

#STEP2:Createareusablepredictionfunction
defmake_prediction(model,input_data):

probabilities=model.predict_proba(input_data)

result="Malignant"ifprediction[0]==0else"Benign"

return{
  "predictoin":result,
  "probabilities":{
    "malignant":probabilities[0][0]'
    "benign":probabilities[0][1]

  }
}


#STEP3:Run a simple prediction when the executed directly
if__name__=="__main__":

model=load_model()

data=load_breast_cancer()

feature_names=data.feature_names

sample_input=dict(zip(feature_name,data.data[0]))

#Modify a few values for demonstration
sample_input["mean radius"]=15.0
sample_input["mean texture"]=18.0
sample_input["mean perimeter"]=100.0

input_df=pd.DataFrame(
  [sample_input],
  columns=feature_names
)

output=make_prediction(model,input_df)

print("Modelloadedseccessfully!")
print("\nPrediction:",output["prediction"])
print("\nPredictionProbabilities:")
print("Malignant:",output["probalities"]["malignant"])
print("Benign:",output["probabilities"]["benign"])
