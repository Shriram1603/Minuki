import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


df=pd.read_csv('newds.csv')
x=df.iloc[ : , 1:]
y=df.iloc[ : , 0]
# print(y)
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
# print(X_encoded)
model = RandomForestClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(y_pred)

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(encoder, le_file)
# Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))