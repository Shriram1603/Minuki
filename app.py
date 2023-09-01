from flask import Flask,render_template,request
import pickle
import numpy as np
# from sklearn.preprocessing import LabelEncoder

app=Flask(__name__)

with open('label_encoder.pkl', 'rb') as le_file:
    encoder = pickle.load(le_file)

model=pickle.load(open('model.pkl','rb'))



@app.route("/",methods=['GET'])
def hello():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[x for x in request.form.values()]
    # final=[np.array(int_features)]
    # prediction=model.predict_proba(final)
    y=encoder.transform([int_features])
    prediction = model.predict_proba(y)


    decoded_prediction_encoded = encoder.inverse_transform(prediction)

# Convert the result to a more interpretable format
    decoded_prediction = decoded_prediction_encoded[0] 
    #crop_name = label_encoder.inverse_transform([predicted_class_index])[0]
    #prediction=encoder.transform(prediction)

    
    # Use the label_encoder to map the index back to the original crop name

    return render_template('index.html', prediction_text=decoded_prediction)