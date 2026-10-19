import streamlit as st
import requests
import os

# Read the backend service network URL from environmental variables (default to local instance)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="SMS Spam Detector", layout="centered")
st.title("📬 SMS Spam Classification Platform")
st.write("This user application layout communicates directly with the decoupled inferencing API over the network.")

input_sms = st.text_area("Enter the SMS text message below:")

if st.button('Analyse Message'):
    if input_sms.strip():
        with st.spinner("Communicating with classification service..."):
            try:
                # Issue HTTP POST request payload to the isolated API backend layer
                response = requests.post(f"{BACKEND_URL}/predict", json={"text": input_sms}, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    is_spam = data.get("is_spam")

                    if is_spam:
                        st.error("🚨 This message is classified as SPAM.")
                    else:
                        st.success("✅ This message is classified as HAM (Safe).")
                else:
                    st.error(f"Backend API returned an error status code: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not locate the microservice engine. Verify that the backend container is running on port 8000.")
    else:
        st.warning("Please enter a valid message block first.")
