import streamlit as st

st.title("MindLink AI")

st.write("MindLink prototype is running successfully 🚀")

name = st.text_input("Enter your name")

if st.button("Submit"):
    st.write("Hello", name)