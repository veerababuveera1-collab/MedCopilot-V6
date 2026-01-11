import streamlit as st
from users import USERS

st.set_page_config(page_title="MedCopilot Login", page_icon="🔐", layout="centered")

st.title("🔐 MedCopilot OS — Secure Login")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""

# ---------- Login Form ----------
if not st.session_state.logged_in:
    st.subheader("Sign In")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = USERS[username]["role"]
            st.success("Login successful!")
            st.switch_page("landing.py")
        else:
            st.error("Invalid username or password")

# ---------- Already Logged In ----------
else:
    st.success(f"Logged in as {st.session_state.username}")
    if st.button("Go to Dashboard"):
        st.switch_page("landing.py")
