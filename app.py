import streamlit as st
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import plotly.express as px

st.title('Transportni klassifikatsiya qiluvchi model')

file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif'])
if file:
    st.image(file)
    img = PILImage.create(file)
    model = load_learner('transport_model.pkl')

    pred, pred_id, probs = model.predict(img)

    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimoligi: {probs[pred_id]*100:.1f}%')

# plotting
fig = px.bar(x=probs*100, y=model.dls.vocab)
st.plotly_chart(fig)