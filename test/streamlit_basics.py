# python streamlit_basics.py
# streamlit run test/streamlit_basics.py

import streamlit as st

# basics element
st.title("Streamlit Intro")
st.write("This is where we'll learn streamlit")

# Getting user input
name = st.text_input("What's your name?")
if name:
    st.write(f"Welcome, {name}! Ready to build something amazing?")

# More interative elements
age = st.slider("Your age", 0, 100)
gender = st.radio("Gender", ("Male", "Female", "Other"))
interest = st.selectbox("Interested in", ("CV", "ML", "Both"))

if st.button("Create profile"):
    st.write(f"Age : {age}, Gender : {gender}, Interest : {interest}")

# Save user profile without session_state
# all_profile = []

# new_profile = {"Name" : name,
#                 "Age" : age, 
#                "Gender" : gender, 
#                "Interest" : interest}

# all_profile.append(new_profile)

# if st.button("Show all Profile"):
#     st.write(all_profile)

# reruns the whole page from top to bottom
# all the variables initialised again
# a.py --> a = 0 , b = 0
# a+=1, b+=1 
# print(a,b) --> python a.py : a = 1, b = 1
# python a.py : a = 1, b = 1

# because streamlit retails value of widget we see the previous value
# instead of python native variables, we should use
# stream session_state
# session_state is like a dictionary

# Solution

if "profiles" not in st.session_state:
    st.session_state.profiles = []

new_profile = {"Name" : name,
                "Age" : age, 
               "Gender" : gender, 
               "Interest" : interest}

if st.button("Save and Show profile"):
    st.session_state.profiles.append(new_profile)
    st.write(f"All profiles : ", st.session_state.profiles)

