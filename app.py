import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')

# Loading the models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Define a single function to clean resumes and text data
def clean_and_preprocess(text):
    # Clean resumes by removing URLs, mentions, and hashtags
    text = re.sub('http\S+\S', ' ', text)
    text = re.sub('@\S+', ' ', text)
    text = re.sub('#\S+', ' ', text)
    
    # Lowercase the text
    text = text.lower()
    
    # Remove special characters, numbers, and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # Join the cleaned tokens back into a single string
    cleaned_text = ' '.join(lemmatized_tokens)
    
    return cleaned_text

# Web-app
def main():
    st.title('Resume Screening App')
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])
    
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except:
            # If utf-8 decoding fails, try decoding with latin-1
            resume_text = resume_bytes.decode('latin-1')
    
        cleaned_resume = clean_and_preprocess(resume_text)
        cleaned_resume = tfidf.transform([cleaned_resume])
        
        prediction_id = clf.predict(cleaned_resume)[0]
        
        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)

# Python main
if __name__ == "__main__":
    main()