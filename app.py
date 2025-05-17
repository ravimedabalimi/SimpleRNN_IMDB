import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import streamlit as st


model = load_model('./simple_rnn_imdb.h5')
# model.summary()

# Use the default parameters to keras.datasets.imdb.load_data
start_char = 1
oov_char = 2
index_from = 3

word_index = imdb.get_word_index()
inverted_word_index = dict(
    (i + 3, word) for (word, i) in word_index.items()
)

def encode_review(review):
    encd_review = [word_index.get(word, oov_char) for word in review]
    return encd_review

max_len = 500

def preprocess_review(review):
    #Encode text to onehot
    encd_review = encode_review(review)
    #padding seq
    seq = sequence.pad_sequences([encd_review] ,maxlen=max_len)

    return seq


#Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify as Positive or Negative")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    
    preprocessed_seq = preprocess_review(user_input)

    prediction = model.predict(preprocessed_seq)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Score: {prediction[0][0]}")

else:
    st.write("Please enter review")




