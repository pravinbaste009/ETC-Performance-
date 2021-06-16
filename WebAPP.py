# imports the libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from PIL import Image
import streamlit as st
st.write("""
### Developed By: Mr. Pravin Baste
### Guided By: Dr. Anagha Pathak & Dr. Sandesh Jadkar
""")
# open and show the image
image = Image.open('ETC.png')


st.image(image, caption='ETC Water Heater',use_column_width=True)
# get the data
df = pd.read_csv('demo.csv')


# set a subheader
st.header(' Data Information ')

#show the data as a table
st.dataframe(df)

#show the statistics on the data
st.write(df.describe())

# show the data as a chart
chart =st.bar_chart(df)

# split the data into independend 'X' and dependent 'Y'variable

X = df.iloc[ : , :-1].values
Y = df.iloc[ : , 6].values

# split the data set into 75% training data and 25% testing data
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2, random_state=0)

#Get the feature input from the user
def get_user_input():
    Initial_Temp = st.sidebar.slider('Initial_Temp @ 10 A.M',0.0,100.0,35.0)
    Final_Temp = st.sidebar.slider('Final_Temp @ 5 P.M', 0.0,100.0,60.0)
    Amb_Temp = st.sidebar.slider('Avg. Amb_Temp ', 0.0,50.0,30.0)
    Glob_Rad = st.sidebar.slider('Avg. Glob_Rad', 0.0, 1000.0,750.0)
    Collector_Area = st.sidebar.slider('Collector_Area', 0.0,5.0,1.5)
    Tank_Capacity = st.sidebar.slider('Tank_Capacity', 0,500,100)


    #store a dictionary into a variable
    user_data = {'Initial_Temp':Initial_Temp,
                 'Final_Temp':Final_Temp,
                 'Amb_Temp':Amb_Temp,
                 'Glob. Rad.':Glob_Rad,
                 'Collector_Area':Collector_Area,
                 'Tank_Capacity':Tank_Capacity
                 }

    # Transform the data into a data frame
    features = pd.DataFrame(user_data, index = [0])
    return features

#store the user input into a variable
user_input =get_user_input()

# Set a subheader and display the user input
st.subheader('User Input')
st.write(user_input)

#Create and train the model
model_fit = LinearRegression()
model_fit.fit(X_train,Y_train)

#show the models metrics
#st.subheader('Model Test Accuracy Score: ')
#st.write(str(accuracy_score(Y_test,model_fit.predict(X_test))* 100)+'%')

#Store the models predictins in a variable
prediction = model_fit.predict(user_input)

# Reads in saved model


# set a subheader and display the classification
st.subheader ('Predicted Performacne : ')
st.write(prediction)


# save the model
import pickle
pickle.dump(model_fit,open('performance.pkl','wb'))