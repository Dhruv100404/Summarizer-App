# import streamlit as st
# from txtai.pipeline import Summary, Textractor
# from PyPDF2 import PdfReader

# st.set_page_config(layout="wide")

# @st.cache_resource
# def text_summary(text, maxlength=None):
#     #create summary instance
#     summary = Summary()
#     text = (text)
#     result = summary(text)
#     return result

# def extract_text_from_pdf(file_path):
#     # Open the PDF file using PyPDF2
#     with open(file_path, "rb") as f:
#         reader = PdfReader(f)
#         page = reader.pages[0]
#         text = page.extract_text()
#     return text

# choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])

# if choice == "Summarize Text":
#     st.subheader("Summarize Text using txtai")
#     input_text = st.text_area("Enter your text here")
#     if input_text is not None:
#         if st.button("Summarize Text"):
#             col1, col2 = st.columns([1,1])
#             with col1:
#                 st.markdown("**Your Input Text**")
#                 st.info(input_text)
#             with col2:
#                 st.markdown("**Summary Result**")
#                 result = text_summary(input_text)
#                 st.success(result)

# elif choice == "Summarize Document":
#     st.subheader("Summarize Document using txtai")
#     input_file = st.file_uploader("Upload your document here", type=['pdf'])
#     if input_file is not None:
#         if st.button("Summarize Document"):
#             with open("doc_file.pdf", "wb") as f:
#                 f.write(input_file.getbuffer())
#             col1, col2 = st.columns([1,1])
#             with col1:
#                 st.info("File uploaded successfully")
#                 extracted_text = extract_text_from_pdf("doc_file.pdf")
#                 st.markdown("**Extracted Text is Below:**")
#                 st.info(extracted_text)
#             with col2:
#                 st.markdown("**Summary Result**")
#                 text = extract_text_from_pdf("doc_file.pdf")
#                 doc_summary = text_summary(text)
#                 st.success(doc_summary)
                
# Use a pipeline as a high-level helper
import streamlit as st
from transformers import BertTokenizerFast, EncoderDecoderModel
import torch

# Load tokenizer and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization')
model = EncoderDecoderModel.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization').to(device)

# Function to generate summary with minimum length constraint
def generate_summary_with_min_length(text, min_length=100):
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Set minimum length constraint
    min_length = max(min_length, 0)
    if min_length > 0:
        output = model.generate(input_ids, attention_mask=attention_mask, min_length=min_length)
    else:
        output = model.generate(input_ids, attention_mask=attention_mask)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit app
st.title("Text Summarization with BERT")

text_input = st.text_area("Enter text to summarize")

min_length = st.number_input("Minimum length of the summary:", min_value=0, value=100)

if st.button("Summarize"):
    if text_input:
        summary = generate_summary_with_min_length(text_input, min_length=min_length)
        st.write(summary)
    else:
        st.error("Please enter some text to summarize.")


