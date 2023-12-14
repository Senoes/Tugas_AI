import streamlit as st
import joblib as jb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rf = jb.load('./smoke-detection-rf.joblib')
sc = jb.load('./standardscaler.joblib')

st.title('Prediksi Kebakaran')
st.header("Dataset")

df1 = pd.read_csv('smoke_detection.csv')
st.dataframe(df1)

st.header('Grafik')

st.write('Grafik Suhu[C]')
chart_Temperature = df1["Temperature(C)"].head(208)
st.bar_chart(chart_Temperature)

st.write('Grafik Kelembapan[%]')
chart_Humidity = df1["Humidity(%)"].head(208)
st.bar_chart(chart_Humidity)

st.write('Grafik TVOC[ppb]')
chart_TVOC = df1["TVOC(ppb)"].head(208)
st.line_chart(chart_TVOC)

st.write('Grafik eCO2[ppm]')
chart_eCO2 = df1["eCO2(ppm)"].head(208)
st.line_chart(chart_eCO2)

st.write('Grafik Raw Hidrogen')
chart_RawH2 = df1["Raw H2"].head(208)
st.bar_chart(chart_RawH2)

st.write('Grafik Raw Ethanol')
chart_Ethanol = df1["Raw Ethanol"].head(208)
st.bar_chart(chart_Ethanol)

st.write('Grafik Pressure(hPa)')
chart_Pressure = df1["Pressure(hPa)"].head(208)
st.bar_chart(chart_Pressure)

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
        st.empty()    
    else:
        st.write('Prediksi Alarm Kebakaran = :red[ON]')
        blink_red = '<p style="color:red;font-size:20px;display:inline-block">&#128993;</p>'
        blinking_icons = ' '.join([blink_red] * 3 )  
        st.markdown(blinking_icons, unsafe_allow_html=True)
        st.empty()  
