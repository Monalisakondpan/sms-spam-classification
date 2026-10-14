import streamlit as st
import pickle
import string
import nltk
import time
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# 1. Download necessary NLTK components quietly
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

# Initialize session state variables for logs
if 'history' not in st.session_state:
    st.session_state.history = []

# List of common high-risk spam keywords to highlight
SPAM_TRIGGER_KEYWORDS = [
    'urgent', 'prize', 'claim', 'award', 'won', 'free', 'cash', 'call', 'text',
    'reply', 'win', 'winner', 'selected', 'congratulations', 'gift', 'card',
    'click', 'link', 'subscribe', 'verify', 'account', 'bank', 'offer', 'guaranteed'
]


# 2. OPTIMIZED: Text Preprocessing Function (Handles symbols and alternative encodings safely)
def transform_text(text):
    text = str(text).lower()

    # Pre-clean string by replacing high-frequency currency symbols with text indicators
    text = text.replace('£', ' pound ').replace('\$', ' dollar ').replace('€', ' euro ')

    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        # Keep words that are alphanumeric OR are standard target links/codes
        if i.isalnum() or '.' in i or '-' in i:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# 3. Load Synchronized Vectorizer and Model Files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# 4. Streamlit App Interface Setup
st.set_page_config(page_title="AI Spam Detector Pro", page_icon="🛡️", layout="centered")

# Main Page Header Elements
st.title("🛡️ Email/SMS Spam Classifier")
st.subheader("Keep your inbox safe from fraudulent phishing links and scam messages.")

# Create UI Tabs to separate Single Scan from Batch Upload
tab1, tab2 = st.tabs(["🔤 Single Message Scan", "📁 Batch File Upload Processing"])

# ==================== TAB 1: SINGLE SCANNERS ====================
with tab1:
    def clear_text():
        st.session_state.sms_input_field = ""


    with st.form(key='spam_classifier_form', clear_on_submit=False):
        input_sms = st.text_area(
            "✍️ Paste the message you want to scan here:",
            height=150,
            placeholder="Example: Congratulations! You've been selected to receive a free gift card. Click the link to claim now...",
            key="sms_input_field"
        )

        if input_sms:
            words = len(input_sms.split())
            chars = len(input_sms)
            st.caption(f"📊 Live Metrics: {words} words | {chars} characters")

        col1, col2 = st.columns(2)
        with col1:
            predict_clicked = st.form_submit_button('Predict Message Category', use_container_width=True,
                                                    type="primary")
        with col2:
            clear_clicked = st.form_submit_button('Clear Text', use_container_width=True, on_click=clear_text)

    if predict_clicked:
        if input_sms.strip() == "":
            st.warning("⚠️ Please type or paste a message first before predicting!")
        else:
            with st.spinner("🤖 AI Engine is analyzing text vector tokens..."):
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.003)
                    progress_bar.progress(percent_complete + 1)
                progress_bar.empty()

            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])

            result = model.predict(vector_input)
            probabilities = model.predict_proba(vector_input)

            spam_confidence = probabilities[0][1] * 100
            ham_confidence = probabilities[0][0] * 100

            st.write("---")
            m_col1, m_col2 = st.columns(2)

            if result == 1:
                with m_col1:
                    st.metric(label="🔎 Prediction", value="SPAM", delta="High Risk", delta_color="inverse")
                with m_col2:
                    st.metric(label="🎯 Confidence Level", value=f"{spam_confidence:.2f}%")

                st.error(f"🚨 **Spam Detected!** This message matches known scam signatures.")
                st.info(
                    "⚠️ **Safety Tip:** Do not click any links, call listed numbers, or share personal details found in this message.")
                st.session_state.history.append({'text': input_sms, 'status': 'Spam'})

                found_triggers = [word for word in SPAM_TRIGGER_KEYWORDS if word in input_sms.lower()]
                if found_triggers:
                    st.markdown("🔍 **Detected Trigger Words:**")
                    st.warning(", ".join(found_triggers))
            else:
                with m_col1:
                    st.metric(label="🔎 Prediction", value="SAFE", delta="Clean", delta_color="normal")
                with m_col2:
                    st.metric(label="🎯 Confidence Level", value=f"{ham_confidence:.2f}%")

                st.success(f"✅ **Safe Message!** This text appears completely normal.")
                st.info("ℹ️ **Analysis:** This message does not contain typical spam keywords or patterns.")
                st.session_state.history.append({'text': input_sms, 'status': 'Not Spam'})

# ==================== TAB 2: BATCH PROCESSOR ====================
with tab2:
    st.markdown("### 📁 Upload a Dataset for Bulk Analysis")
    st.write(
        "Upload a standard spreadsheet `.csv` file containing text data. Make sure your file has a column containing the text messages.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

        columns = df_uploaded.columns.tolist()
        text_column = st.selectbox("Select the column name that holds the text messages:", columns)

        if st.button("Process Bulk Dataset", type="primary", use_container_width=True):
            with st.spinner("⚡ Running AI pipeline over bulk text arrays... Please hold..."):

                results_list = []
                for row_text in df_uploaded[text_column]:
                    processed = transform_text(row_text)
                    vectorized = tfidf.transform([processed])
                    pred = model.predict(vectorized)
                    results_list.append("Spam" if pred == 1 else "Safe")

                df_uploaded['AI_Prediction'] = results_list

                spam_count = results_list.count("Spam")
                safe_count = results_list.count("Safe")

                st.write("---")
                st.markdown("### 📊 Batch Evaluation Summary Metrics:")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Scanned rows", len(df_uploaded))
                c2.metric("Spam Flagged", spam_count, delta="High Risk" if spam_count > 0 else None,
                          delta_color="inverse")
                c3.metric("Safe Marked", safe_count)

                st.markdown("#### 👀 Data Preview (First 5 Rows with AI Predictions):")
                st.dataframe(df_uploaded.head(5), use_container_width=True)

                csv_download = df_uploaded.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Annotated Results Sheet (.CSV)",
                    data=csv_download,
                    file_name="spam_predictions_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# 6. Sidebar Construction
st.sidebar.title("🕒 Scan History Log")
if not st.session_state.history:
    st.sidebar.info("No messages scanned in this session yet.")
else:
    for idx, item in enumerate(reversed(st.session_state.history)):
        label_emoji = "🚨" if item['status'] == "Spam" else "✅"
        display_idx = len(st.session_state.history) - idx
        st.sidebar.markdown(f"**{label_emoji} Msg {display_idx}: {item['status']}**")
        st.sidebar.caption(f"\"{item['text'][:45]}...\"")
        st.sidebar.write("---")

# Footer Credit Line
st.write("")
st.write("")
st.markdown(
    """
    <div style='text-align: center; color: #888888; font-size: 0.85rem;'>
        ---<br>
        🛡️ <b>AI Spam Detector Pro</b> | Developed as an Advanced Machine Learning Project<br>
        Powered by Streamlit Framework & Scikit-Learn Engines
    </div>
    """,
    unsafe_allow_html=True
)
