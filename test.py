import streamlit as st
import joblib as jb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


rf = jb.load('./smoke-detection-rf.joblib')
sc = jb.load('./standardscaler.joblib')

st.title('Prediksi Kebakaran')
st.header("Dataset")

df1 = pd.read_csv('smoke_detection.csv')
st.dataframe(df1)

st.header('Input Data')
temp = st.number_input('Suhu[C]', value=20.0)
hum = st.number_input('Kelembapan[%]', value=57.36)
tvoc = st.number_input('TVOC[ppb]', value=0)
eco2 = st.number_input('Kadar Oksigen[ppm]', value=400)
h2 = st.number_input('Hidrogen Sensor', value=12306)
eth = st.number_input('Ethanol Sensor', value=18520)
press = st.number_input('Tekanan[hPa]', value=939.735)

if st.button('Predict'):
    data_input = np.array([[temp, hum, tvoc, eco2, h2, eth, press]])
    predict = rf.predict(sc.transform(data_input))
    
    if predict == 0:
        st.write('Prediksi Alarm Kebakaran = :green[OFF]')
        blink_green = '<p style="color:green;font-size:20px;display:inline-block">&#128994;</p>'
        blinking_icons = ' '.join([blink_green] * 3 )  
        st.markdown(blinking_icons, unsafe_allow_html=True)
        time.sleep(2)  
        st.empty()    
    else:
        st.write('Prediksi Alarm Kebakaran = :red[ON]')
        blink_red = '<p style="color:red;font-size:20px;display:inline-block">&#128993;</p>'
        blinking_icons = ' '.join([blink_red] * 3 )  
        st.markdown(blinking_icons, unsafe_allow_html=True)
        time.sleep(2)  
        st.empty()  