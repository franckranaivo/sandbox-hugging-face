import streamlit as st
from transformers import pipeline

pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def main():
    st.title("Trustpilot Analysis with DistilBERT")

    input_text = st.text_area("Enter text for sentiment analysis:")

    if st.button("Analyze"):
        if input_text:
            result = pipeline(input_text)
            st.write("Sentiment:", result[0]['label'])
            if result[0]['label'] == 'POSITIVE':
                st.success("The sentiment is positive.")
            st.write("Confidence:", round(result[0]['score'], 4))
            if result[0]['label'] == 'NEGATIVE':
                st.error("The sentiment is negative.")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
