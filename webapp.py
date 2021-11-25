from numpy.lib.type_check import imag
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from PIL import Image
import streamlit as st

st.write("""
# Diabates Detection 
# Detect if someone has Diabtes using machine learning!
""")

# open and display an image
image = Image.open('C:/Users/Vishrut Goyal/Desktop/webapp/diabetes/download.jfif')
st.image(image, caption="ML", use_column_width=True)

# Get the data
df = pd.read_csv('C:/Users/Vishrut Goyal/Desktop/webapp/diabetes/diabetes.csv')

#set a subheader
st.subheader('Data Information')

#show the data frame
st.dataframe(df)

# show statistics on the data
st.write(df.describe())

# show correlation value
st.write(df.corr())

#show data as chart
bar = st.bar_chart(df)

#split the data into independent 'x' and dependent 'y' variable
x= df.iloc[:, 0:8].values
y=df.iloc[:,-1].values

#split the data set into 75% training and 25% testing

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20, random_state=1)

#get the feature input from the user
def get_user_input():
    pregancies = st.sidebar.slider('pregnancies',0,17,3)
    glucose = st.sidebar.slider('glucose',0,199,117)
    blood_pressure = st.sidebar.slider('blood_pressure',0,122,72)
    skin_thickness = st.sidebar.slider('skin_thickness',0,99,23)
    insulin = st.sidebar.slider('insulin',0.0,846.0,30.5)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    DPF = st.sidebar.slider('DPF',0.078,2.42,0.375)
    age = st.sidebar.slider('age',21,100,30)

    #store a dictionary into a variable
    user_data = {'pregnancies': pregancies,
                 'glucose':glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
                 'BMI': BMI,
                 'DPF': DPF,
                 'age': age
                }

    #transfer the data into a data frame
    features = pd.DataFrame(user_data, index = [0])
    return features

#store the user input variable
user_input = get_user_input()

#set a subheader and display the user input
st.subheader('User Input:')
st.write(user_input)

#create and train the model for Randomforest
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train,y_train)

#create and train the model for Decisiontress
DecisionTreeClassifier = DecisionTreeClassifier()
DecisionTreeClassifier.fit(x_train,y_train)

#show the model metrices
st.subheader('Model Test Accuracy Score for RandomForest: ')
st.write( str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) *100)+'%')

#store the model prediction in a variable
prediction_randomforest = RandomForestClassifier.predict(user_input)

prediction_decisiontree = DecisionTreeClassifier.predict(user_input)

#set a subheader and display the classification
st.subheader('classification  from random forest: ')
if(prediction_randomforest==0):
    st.write("No from random forest")
else:
    st.write("Yes from random forest")

st.write(prediction_randomforest)

st.subheader('Model Test Accuracy Score for DecisionTree: ')
st.write( str(accuracy_score(y_test, DecisionTreeClassifier.predict(x_test)) *100)+'%')

st.subheader('classification  from decision tree: ')
if(prediction_decisiontree==0):
    st.write("No from decision tree")
else:
    st.write("Yes from decision tree")

st.write(prediction_decisiontree)