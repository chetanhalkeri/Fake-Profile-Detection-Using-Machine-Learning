import streamlit as st
from xgboost import XGBClassifier
import pickle
import numpy as np

with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
st.title('Fake Profile Detection')

def main():
    column_names = ['profile pic', 'nums/length username', 'fullname words',
       'nums/length fullname', 'name', 'description length', 'external URL',
       'private', '#posts', '#followers', '#follows']

    inp = []
    for col in column_names:
        t = st.text_input(col)
        inp.append(t)

    if st.button('predict'):
        pred = model.predict(np.array([inp], dtype = 'object'))
        if pred == 0:
            st.success('Real Profile')
        else:
            st.error('Fake Profile')





    
if __name__ == '__main__':
    main()