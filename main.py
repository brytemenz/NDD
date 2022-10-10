#Diabetes checker
#import libriairies

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#title
st.write("""
# NasCo Diabetes Detection
Detecting whether someone has diabetes using machine learning and python


""")

image= Image.open('DR.jpg')
st.image(image, caption = 'Diabetes detector', use_column_width=True)

#data
df=pd.read_csv('diabetes.csv')

#sub header
st.subheader('Data Information')
st.dataframe(df)
#stats
st.write(df.describe())
#chart
chart=st.bar_chart(df)

#x and y
X= df.iloc[:, 0:8].values
Y= df.iloc[:, -1].values

#I'm spliting the data by 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)

#user imput
def get_user_input():
    pregnancies=st.sidebar.slider('pregnancies', 0,17,3)
    glucose=st.sidebar.slider('glucose', 0,199,117)
    bloodPressure=st.sidebar.slider('bloodPressure', 0,122,72)
    skinThickness=st.sidebar.slider('skinThickness', 0,99,3)
    insulin=st.sidebar.slider('Insulin', 0.0,846.0,30.5)
    BMI=st.sidebar.slider('BMI',0.0,67.1,32.0)
    DPF=st.sidebar.slider('DPF', 0.078,2.42,0.3725)
    age=st.sidebar.slider('age', 21,81,29)

#dictionaries

    user_data = {'pregnancies': pregnancies,
           'glucose': glucose,
           'bloodPressure':bloodPressure,
           'skinThickness':skinThickness,
           'insulin':insulin,
           'BMI': BMI,
           'DPF':DPF,
           'age':age,
           }

    features= pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

st.subheader('User Input')
st.write(user_input)

#Training the model

RandomForestClassifier=RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test))*100 )+ '%')

#store models
predictions=RandomForestClassifier.predict(user_input)

#classification subheader

st.subheader('Classifications:')
st.write(predictions)

