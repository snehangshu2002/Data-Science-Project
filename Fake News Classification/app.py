import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model


with open("Model/vectorizer_text.pkl", "rb") as f:
    vectorizer_text = pickle.load(f)

with open("Model/vectorizer_title.pkl", "rb") as f:
    vectorizer_title = pickle.load(f)

with open("Model/random_forest.pkl", "rb") as f:
    random_forest = pickle.load(f)


# Streamlit app header
st.header("Fake News Prediction")
st.subheader("Created by Snehangshu Bhuin")

# Input fields for user
title = st.text_input("News Title")
text = st.text_area("Description")

# Prediction button
if st.button("Predict"):
    # Transform the input text
    title_transformed = vectorizer_title.transform([title])
    text_transformed = vectorizer_text.transform([text])
    

    input_features = np.hstack((title_transformed.toarray(), text_transformed.toarray()))
    
    # Make prediction
    prediction = random_forest.predict(input_features)[0]
    
    # Display the prediction result
    if prediction == 1:
        st.success("The news is likely Real")
    else:
        st.error("The news is likely Fake")