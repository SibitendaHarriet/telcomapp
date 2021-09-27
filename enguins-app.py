import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Telecom Prediction App

This app predicts the **User integration into internet access **!

Data obtained from a month ago of aggregated data on xDR. 
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://github.com/SibitendaHarriet/PythonPackageStructure/blob/main/data/processed_telecom.csv)
""")


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        MSISDN_Number = st.sidebar.selectbox('MSISDN_Number')
        No_of_xDRsessions = st.sidebar.selectbox('No_of_xDRsessions')
        Session_Duration_s = st.sidebar.slider('Session_Duration_s')
        Total_MB = st.sidebar.slider('Total_MB')
        Social_Media_MB = st.sidebar.slider('Social_Media_MB')
        Google_MB = st.sidebar.slider('Google_MB')
        Email_MB = st.sidebar.selectbox('Email_MB')
        Youtube_MB = st.sidebar.selectbox('Youtube_MB')
        Netflix_MB = st.sidebar.slider('Netflix_MB')
        Gaming_MB = st.sidebar.slider('Gaming_MB')
        Other_MB = st.sidebar.slider('Other_MB')
        DecileRank = st.sidebar.slider('DecileRank',0,1,2,3,4)
        
        data = {'MSISDN_Number': MSISDN_Number,
                
                'No_of_xDRsessions': No_of_xDRsessions,
                'Session_Duration_s': Session_Duration_s,
                'Total_MB': Total_MB,
                'Social_Media_MB': Social_Media_MB,
                'Email_MB': Email_MB,
                'Youtube_MB': Youtube_MB,
                'Netflix_MB': Netflix_MB,
                'Gaming_MB': Gaming_MB,
                'Other_MB': Other_MB
                'DecileRank': DecileRank}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
enguins_raw = pd.read_csv('https://github.com/SibitendaHarriet/PythonPackageStructure/blob/main/data/processed_telecom.csv')
enguins = enguins_raw.drop(columns=['MSISDN_Number'], axis=1)
df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['MSISDN_Number','No_of_xDRsessions','Session_Duration_s','Total_MB','Social_Media_MB','Google_MB','Email_MB','Youtube_MB','Netflix_MB', 'Gaming_MB', 'Other_MB']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('enguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
penguins_species = np.array(['neutral', 'bad', 'fair', 'good', 'very_good'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
