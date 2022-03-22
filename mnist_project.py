from tkinter import CENTER
from ctypes import alignment
import  numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pickle

m = pickle.load(open('model.pkl','rb'))

st.header("MNIST CLASSIFICATION")

canvas_result = st_canvas(
    fill_color = "#FFE94B",
    stroke_width = 10,
    stroke_color = "#ffffff",
    background_color = "#FFE94B",
    height = 250,width = 250,
    drawing_mode = 'freedraw',
    key = "canvas",
)


if canvas_result.image_data is not None:
    
    img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img.astype('uint8'), (28,28))
    img = img.reshape(1, 784)


if st.button('Predict'):
    
    prediction = m.predict(img)
    
    st.write(prediction)