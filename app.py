import streamlit as st
import pickle
import docx 
import PyPDF2
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Set page config at the top
st.set_page_config(page_title="DeepCV", page_icon="📝", layout="wide", initial_sidebar_state="expanded")

# Load pre-trained models
svc_model = pickle.load(open('clf.pkl', 'rb')) 
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  
le = pickle.load(open('encoder.pkl', 'rb')) 

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #e0eafc, #cfdef3); padding: 20px;}
    .sidebar .sidebar-content {background: #ffffff; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
    .stButton>button {
        background: linear-gradient(90deg, #007bff, #0056b3); 
        color: white; 
        border: none; 
        border-radius: 25px; 
        padding: 10px 20px; 
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0056b3, #003d80); 
        transform: scale(1.05);
    }
    .stTextArea textarea {
        border: 2px solid #007bff; 
        border-radius: 10px; 
        background-color: #f8f9fa;
    }
    .title {color: #1a3159; font-size: 40px; font-weight: 700; text-align: center; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);}
    .subheader {color: #2c5282; font-size: 24px; font-weight: 600; margin-top: 20px;}
    .result-box {background: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 6px 12px rgba(0,0,0,0.15); margin: 10px 0; border-left: 5px solid #007bff;}
    .percentage-circle {font-size: 28px; color: #28a745; text-align: center; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# Clean text function
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[{}]'.format(re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")), ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText.strip()

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Extract text from TXT
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

# Handle file upload
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text

# Predict category
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    vectorized_text = vectorized_text.toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]

# Calculate similarity
def calculate_similarity(resume_text, job_desc):
    cleaned_resume = cleanResume(resume_text)
    cleaned_job_desc = cleanResume(job_desc)
    resume_vec = tfidf.transform([cleaned_resume]).toarray()
    job_vec = tfidf.transform([cleaned_job_desc]).toarray()
    similarity = cosine_similarity(resume_vec, job_vec)[0][0]
    return round(similarity * 100, 2)

# Extract keywords and provide feedback
def get_feedback(resume_text, job_desc):
    cleaned_resume = cleanResume(resume_text).lower().split()
    cleaned_job_desc = cleanResume(job_desc).lower().split()
    
    # Common keywords
    common_keywords = set(cleaned_resume) & set(cleaned_job_desc)
    
    # Missing keywords from job description
    missing_keywords = set(cleaned_job_desc) - set(cleaned_resume)
    
    # Count frequencies for visualization
    resume_word_freq = Counter(cleaned_resume)
    job_word_freq = Counter(cleaned_job_desc)
    
    return common_keywords, missing_keywords, resume_word_freq, job_word_freq

def main():
    # Sidebar for inputs
    with st.sidebar:
        st.markdown('<p class="subheader">Input Section</p>', unsafe_allow_html=True)
        job_desc = st.text_area("📋 Job Description", height=200, placeholder="Paste the job description here...")
        uploaded_file = st.file_uploader("📄 Upload Resume", type=["pdf", "docx", "txt"], help="Upload PDF, DOCX, or TXT")
        analyze_button = st.button("Analyze Now", key="analyze")

    # Main content
    st.markdown('<p class="title">DeepCV</p>', unsafe_allow_html=True)
    st.markdown("Match your resume to any job with advanced insights! ✨", unsafe_allow_html=True)

    if analyze_button:
        if uploaded_file is not None and job_desc:
            with st.spinner("Analyzing your resume..."):
                time.sleep(1)  # Simulate processing
                try:
                    # Extract resume text
                    resume_text = handle_file_upload(uploaded_file)

                    # Results Section
                    st.markdown('<p class="subheader">Analysis Results</p>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Category Result
                        category = pred(resume_text)
                        st.markdown(f'<div class="result-box">Job Category: <b>{category}</b></div>', unsafe_allow_html=True)
                    
                    with col2:
                        # Similarity Result with Gauge Chart
                        match_percentage = calculate_similarity(resume_text, job_desc)
                        st.markdown(f'<div class="result-box">ATS Match: <span class="percentage-circle">{match_percentage}%</span></div>', unsafe_allow_html=True)
                        gauge_fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=match_percentage,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={'axis': {'range': [0, 100]},
                                   'bar': {'color': "#28a745"},
                                   'steps': [{'range': [0, 50], 'color': "#f8d7da"},
                                             {'range': [50, 75], 'color': "#fff3cd"},
                                             {'range': [75, 100], 'color': "#d4edda"}]},
                            title={'text': "Match Score"}
                        ))
                        st.plotly_chart(gauge_fig, use_container_width=True)

                    # Feedback Section
                    st.markdown('<p class="subheader">Improvement Feedback</p>', unsafe_allow_html=True)
                    common_keywords, missing_keywords, resume_freq, job_freq = get_feedback(resume_text, job_desc)
                    
                    # Missing Keywords
                    st.write("**Missing Keywords to Improve Your Resume:**")
                    st.write(", ".join(list(missing_keywords)[:10]) or "No significant keywords missing!")
                    
                    # Bar Chart for Keyword Comparison
                    top_job_keywords = dict(job_freq.most_common(10))
                    top_resume_keywords = {k: resume_freq.get(k, 0) for k in top_job_keywords.keys()}
                    
                    bar_fig = px.bar(
                        x=list(top_job_keywords.keys()),
                        y=[top_job_keywords[k] for k in top_job_keywords.keys()],
                        labels={'x': 'Keywords', 'y': 'Frequency in Job Description'},
                        title="Top Job Description Keywords",
                        color_discrete_sequence=['#007bff']
                    )
                    bar_fig.add_bar(
                        x=list(top_resume_keywords.keys()),
                        y=[top_resume_keywords[k] for k in top_resume_keywords.keys()],
                        name="Resume Frequency",
                        marker_color='#28a745'
                    )
                    bar_fig.update_layout(barmode='group', bargap=0.2)
                    st.plotly_chart(bar_fig, use_container_width=True)

                    # Optional Details
                    with st.expander("🔍 View Details"):
                        st.write("**Extracted Resume Text:**")
                        st.text_area("", resume_text, height=150)
                        st.write("**Cleaned Job Description:**")
                        st.text_area("", cleanResume(job_desc), height=150)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
        elif not uploaded_file:
            st.warning("Please upload a resume to proceed!")
        elif not job_desc:
            st.warning("Please enter a job description!")

if __name__ == "__main__":
    main()