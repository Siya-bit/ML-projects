import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    # converting to lower case
    text = text.lower()

    # tokenization
    text = nltk.word_tokenize(text)

    # removing special characters
    removedSC = list()
    for i in text:
        if i.isalnum():
            removedSC.append(i)

    # updating the text after removed special characters
    text = removedSC[:]

    # removing stop words and punctuation characters
    removedSWPC = list()
    for i in text:
        # stopwords.words('english') is a function of 'nltk', returns list of english stop words
        # string.punctuation is a part of 'string' module, containing the ASCII punctuation characters
        if i not in stopwords.words('english') and i not in string.punctuation:
            removedSWPC.append(i)

    # updating the text after removed stop words and punctuation characters
    text = removedSWPC[:]

    # stemming the data using 'PorterStemmer' algorithm.
    # nltk module provides this class to use.
    ps = PorterStemmer()
    stemmed = list()
    for i in text:
        stemmed.append(ps.stem(i))
    text = stemmed[:]
    return " ".join(text)


tf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")

input_sns = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. Preprocess
    transformed_sns = transform_text(input_sns)
    # 2. vectorize
    vector_input = tf.transform([transformed_sns])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")