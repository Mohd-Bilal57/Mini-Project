import streamlit as st
import joblib
import sklearn
import numpy as np
from PIL import Image
from skimage.io import imread             
from skimage.transform import resize      


model = joblib.load('/content/Image_classification_model')
st.title('Image Classifier')
st.text("Upload the image")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img,caption='Uploaded Image')

  if st.button('Predict'):
    st.write('Result...')

    flat_data = []
    Categories = ['Banana', 'Grapes']

    img = np.array(img)
    img_resized = resize(img,(150,150,3)) # Data will be normalize to 0 to 1
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    
    pred = model.predict(flat_data)
    pred = Categories[pred[0]]
    st.write("Predicted Output: " + pred)