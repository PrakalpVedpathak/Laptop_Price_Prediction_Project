import pandas as pd
import streamlit as st
import pickle
import numpy as np

df = pd.read_pickle(open('df.pkl', 'rb'))
pipe = pd.read_pickle(open('pipe.pkl', 'rb'))

st.title("Laptop Predictor")

company = st.selectbox('Company', df['Company'].unique())

typename = st.selectbox('Type Name', df['TypeName'].unique())

Ram = st.selectbox('RAM', [2, 4, 6, 8, 12, 16, 24, 32, 64])

weight = st.number_input('Weight')

touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

screen_size = st.number_input('Screen Size')

ips = st.selectbox('IPS', ['No', 'Yes'])

resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                                                '2880x1800', '2560x1600','2560x1440', '2304x1440'])


cpu = st.selectbox('CPU', df['CPU Brand'].unique())

hdd = st.selectbox('HDD', [0, 32, 128, 258, 512, 1000, 1024, 2048])

ssd = st.selectbox('SSD', [0, 32, 128, 258, 500, 512, 1024, 2048])

gpu = st.selectbox('GPU', df['GPU Brand'].unique())

os = st.selectbox('Operating System', df['OS'].unique())

if st.button('Predict Price'):

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2) + (y_res**2))**0.5 / screen_size
    query = np.array([company, typename, Ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)))))
