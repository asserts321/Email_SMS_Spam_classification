import streamlit as st

import pickle
import string
import nltk
nltk.data.path.append('nltk_data')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
ps = PorterStemmer()
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the SMS/Email")
if st.button('Predict'):

    #1Preprocess
    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
        text = y.copy()
        y.clear()
        for i in text:
            if i not in stopwords.words('english') and i not in  string.punctuation :
                y.append(i)

        text = y.copy()
        y.clear()

        for i in text:
            y.append(ps.stem(i))
        return " ".join(y)

    Transformed_sms = transform_text(input_sms)
    #2Vectorize
    vector_input = tfidf.transform([Transformed_sms])
    #3Predict
    result = model.predict(vector_input)[0]
    #4Display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')