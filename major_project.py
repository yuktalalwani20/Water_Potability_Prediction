import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import joblib
import base64


data = pd.read_csv('new_data.csv')
data = data.iloc[:,0:]
original_data = pd.read_csv('new_data.csv')
feature_list = {'ph':0,'Turbidity':0,'pH_Type':0,'Turbidity_Type':0}

classifier = joblib.load('algorithms.sav')
main_bg = "nature-3267579_1920.jpg"
main_bg_ext = "jpg"

side_bg = "nature-3267579_1920.jpg"
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Water Potability Prediction !!!!!!!')
st.subheader('Predict the water you drink is pure or not ??')
st.sidebar.header('Predict The Purity')
for j in feature_list.keys():
    feature_list[j] = st.sidebar.text_input(f'enter value for {j}')




if st.sidebar.button('Predict'):
    if 0 in list(feature_list.values()):
        st.sidebar.markdown(' # Please fill all the values')
    else:
        #if st.sidebar.button('RF'):
        pred = RandomForestclassifier.predict([list(feature_list.values())])
        if pred[0]==0:
            st.sidebar.markdown('# water is not so potable for drinking purpose')
        else:
            st.sidebar.markdown('# water is potable for drinking purpose')
    
    
st.image('water.jpg')
st.dataframe(data.head(200))
header = st.container()
body = st.container()
with header:
    col1,col2 = st.columns(2)
    plot_type = col1.selectbox('Plot the feature',['histogram','line plot','area chart'])
    feat = col1.selectbox('Which feature', ['ph','Turbidity'])
    if plot_type == 'histogram':
        col2.bar_chart(data[feat][:90])
        col1.header(f'{plot_type} of {feat} feature')
    if plot_type == 'line plot':
        col2.line_chart(data[feat][:90])
        col1.header(f'{plot_type} of {feat} feature')
    if plot_type == 'area chart':
        col2.area_chart(data[feat][:90])
        col1.header(f'{plot_type} of {feat} feature')
with body:
    col1, col2 = st.columns(2)
    col1.header('About Me')
    col1.markdown('''My name is Yukta Lalwani , I'm pursuing my Computer Science Degree and love to do Machine Learning stuff''')
    col2.header('About Project')
    col2.markdown("Are you researching for water potability check, Just fill the content of water and here you go")

        
