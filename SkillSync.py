import streamlit as st
import pymongo
import re
from fuzzywuzzy import fuzz, process
import spacy
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import sys 
import nltk

import pickle
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer

import uuid
import base64
from datetime import datetime
from bson.binary import Binary 
import io
# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["skillsync"]
users_collection = db["users"]
companies_collection = db["companies"]
classifier = pickle.load(open('classifier.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

label_mappings = {
    0: 'Advocate', 1: 'Arts', 2: 'Automation Testing', 3: 'Blockchain',
    4: 'Business Analyst', 5: 'Civil Engineer', 6: 'Data Science', 
    7: 'Database', 8: 'DevOps Engineer', 9: 'DotNet Developer', 
    10: 'ETL Developer', 11: 'Electrical Engineering', 12: 'HR', 
    13: 'Hadoop', 14: 'Health and fitness', 15: 'Java Developer', 
    16: 'Mechanical Engineer', 17: 'Network Security Engineer',
    18: 'Operations Manager', 19: 'PMO', 20: 'Python Developer', 
    21: 'SAP Developer', 22: 'Sales', 23: 'Testing', 24: 'Web Designing'
}
def cleanResume(resumetxt):
    cleanTxt = re.sub("http\S+\s", " ", resumetxt)
    cleanTxt = re.sub("RT|cc", "", cleanTxt)
    cleanTxt = re.sub("#\S+\s", "", cleanTxt)
    cleanTxt = re.sub("@\S+", "", cleanTxt)
    cleanTxt = re.sub("\s+", " ", cleanTxt)
    cleanTxt = re.sub(r"[^\x00-\x7f]", " ", cleanTxt)
    cleanTxt = re.sub("[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", cleanTxt)
    return cleanTxt
def categorize_resume():
    st.title("Resume Categorization")
    uploaded_file = st.file_uploader('Upload Resume', type=['pdf'])
    
    if uploaded_file is not None:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            resume_text = ""
            for page in pdf_reader.pages:
                resume_text += page.extract_text()
            resume_text = resume_text.encode('utf-8', 'ignore').decode('utf-8')
        except Exception as e:
            st.error(f"An error occurred while reading the PDF: {str(e)}")
            return
        
        cleanedResume = cleanResume(resume_text)
        features = tfidf.transform([cleanedResume])
        prediction_id = classifier.predict(features)[0]
        
        st.write("Prediction:")
        st.write(f"Category ID: {prediction_id}")
        st.write(f"Predicted Category: {label_mappings[prediction_id]}")
        
       


def validate_email(email):
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(pattern, email))

def signup():
    st.subheader("Create an Account")
    user_type = st.selectbox("Select user type", ["Job Seeker", "Company"])
    
    if user_type == "Job Seeker":
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        skills = st.text_input("Skills (comma-separated)")
        
        if st.button("Sign Up"):
            if not validate_email(email):
                st.error("Please enter a valid email address.")
            elif users_collection.find_one({"email": email}):
                st.error("Email already exists")
            else:
                user = {
                    "name": name,
                    "email": email,
                    "password": password,
                    "skills": [skill.strip() for skill in skills.split(",")],
                    "user_type": "job_seeker"
                }
                users_collection.insert_one(user)
                st.success("Account created successfully")
    
    else:  
        company_name = st.text_input("Company Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        industry = st.text_input("Industry")
        
        if st.button("Sign Up"):
            if not validate_email(email):
                st.error("Please enter a valid email address.")
            elif companies_collection.find_one({"email": email}):
                st.error("Email already exists")
            else:
                company = {
                    "company_name": company_name,
                    "email": email,
                    "password": password,
                    "industry": industry,
                    "user_type": "company"
                }
                companies_collection.insert_one(company)
                st.success("Company account created successfully")

def login():
    st.subheader("Login to Your Account")
    user_type = st.selectbox("Select user type", ["Job Seeker", "Company"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if not validate_email(email):
            st.error("Please enter a valid email address.")
        elif user_type == "Job Seeker":
            user = users_collection.find_one({"email": email})
        else:
            user = companies_collection.find_one({"email": email})
        
        if user and user['password'] == password:
            st.success("Logged in successfully")
            st.session_state.user = user
            st.session_state.logged_in = True
        else:
            st.error("Invalid email or password")

def company_job_postings():
    st.title("Manage Job Postings")
    
    # Display current job postings
    current_jobs = list(companies_collection.find({"email": st.session_state.user["email"]}, {"_id": 0}))
    if current_jobs:
        st.subheader(f"Current Job Postings:")
        for i, job in enumerate(current_jobs[0].get("job_titles", []), start=1):
            st.write(f"{i}. {job}")
    else:
        st.info("No job postings found.")

    # Form to add new job posting
    st.subheader("Add New Job Posting")
    job_title = st.text_input("Job Title")
    job_description = st.text_area("Job Description")
    required_skills = st.text_input("Required Skills (comma-separated)")

    if st.button("Post Job"):
        if not job_title or not job_description or not required_skills:
            st.error("Please fill out all fields.")
        else:
            try:
                companies_collection.update_one(
                    {"email": st.session_state.user["email"]},
                    {
                        "$addToSet": {
                            "job_titles": job_title,
                            "job_descriptions": job_description,
                            "required_skills": [skill.strip() for skill in required_skills.split(",")]
                        }
                    },
                    upsert=True
                )
                st.success("Job posted successfully!")
            except Exception as e:
                st.error(f"An error occurred while posting the job: {str(e)}")
def view_applicants():
    st.subheader("View Applicants")
    
    # Get the current company's job titles
    company_data = companies_collection.find_one({"email": st.session_state.user["email"]})
    job_titles = company_data.get("job_titles", [])
    
    if not job_titles:
        st.warning("You haven't posted any jobs yet.")
        return
    
    # Let the user select a job title
    selected_job = st.selectbox("Select a job title", job_titles)
    
    # Get applications for the selected job
    applications = company_data.get("applications", [])
    job_applications = [app for app in applications if app["job_title"] == selected_job]
    
    if not job_applications:
        st.info(f"No applications received for {selected_job}.")
        return
    
    # Display applicants and allow resume download
    st.write(f"Applications for {selected_job}:")
    for i, application in enumerate(job_applications, 1):
        applicant_name = application["applicant_name"]
        application_date = application["application_date"].strftime("%Y-%m-%d %H:%M:%S")
        
        st.write(f"{i}. {applicant_name} (Applied on: {application_date})")
        
        # Create a download button for the resume
        resume_data = application["resume"]
        resume_filename = application["resume_filename"]
        
        resume_download = io.BytesIO(resume_data)
        st.download_button(
            label=f"Download {applicant_name}'s Resume",
            data=resume_download,
            file_name=resume_filename,
            mime="application/pdf"
        )
        
        # Display the cover letter
        with st.expander("View Cover Letter"):
            st.write(application["cover_letter"])
        
        st.write("---")  # Separator between applicants
def job_application():
    st.subheader("Apply for a Job")
    
    applicant_name = st.session_state.user.get('name')
    st.write(f"Applicant: {applicant_name}")
    
    # Get all companies and their job titles
    companies = list(companies_collection.find({}, {"company_name": 1, "job_titles": 1, "_id": 0}))
    
    # Create a dictionary of companies and their job titles
    company_jobs = {company['company_name']: company.get('job_titles', []) for company in companies}
    
    # Let user select a company
    selected_company = st.selectbox("Select a company", list(company_jobs.keys()))
    
    # Let user select a job title from the selected company
    job_titles = company_jobs[selected_company]
    if job_titles:
        selected_job = st.selectbox("Select a job title", job_titles)
    else:
        st.warning("This company has no job postings.")
        return
    
    # Add a text area for the cover letter
    cover_letter = st.text_area("Cover Letter", "")
    
    # Add a file uploader for the resume
    uploaded_file = st.file_uploader("Upload your resume (PDF only)", type="pdf")
    
    if st.button("Submit Application"):
        if uploaded_file is not None:
            # Read the PDF file
            pdf_content = uploaded_file.read()
            
            # Create the application document
            application = {
                "applicant_name": applicant_name,
                "job_title": selected_job,
                "cover_letter": cover_letter,
                "resume": Binary(pdf_content),  # Store the PDF as Binary data
                "resume_filename": uploaded_file.name,
                "application_date": datetime.now()
            }
            
            # Update the company's document in the database
            result = companies_collection.update_one(
                {"company_name": selected_company},
                {"$push": {"applications": application}}
            )
            
            if result.modified_count > 0:
                st.success("Your application has been submitted successfully!")
            else:
                st.error("There was an error submitting your application. Please try again.")
        else:
            st.error("Please upload your resume before submitting the application.")
 

def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.experimental_rerun()

def search_jobs(query, num_results=5):
    # Retrieve job postings from MongoDB
    job_postings = list(companies_collection.find({}, {"_id": 0}))
    
    # Perform fuzzy search
    results = []
    for posting in job_postings:
        score = max(fuzz.token_sort_ratio(query.lower(), title.lower()) 
                    for title in posting.get("job_titles", []))
        
        if score >= 60:  # Adjust this threshold as needed
            title_index = next(i for i, t in enumerate(posting["job_titles"]) if fuzz.token_sort_ratio(query.lower(), t.lower()) == score)
            
            # Get the company name
            company_name = posting["company_name"]
            
            # Try to get the description
            try:
                description = posting["job_descriptions"][title_index]
            except IndexError:
                description = "No description available"
            
            results.append({
                "title": posting["job_titles"][title_index],
                "company_name": company_name,
                "description": description,
                "score": score
            })
    
    # Sort by score and then alphabetically
    sorted_results = sorted(results, key=lambda x: (-x['score'], x['title']))
    
    return sorted_results[:num_results]

def find_your_job():
    st.title("Find Your Job")
    st.markdown("### Discover opportunities that match your skills")

    query = st.text_input("Enter job title or keyword:", placeholder="e.g., Software Engineer")
    if query:
        num_results = st.slider("Number of results:", min_value=1, max_value=20, value=5)

        if st.button("Search"):
            with st.spinner("Searching for matching job titles..."):
                try:
                    results = search_jobs(query.lower(), num_results)
                    
                    if results:
                        st.subheader(f"Top {len(results)} matching job postings:")
                        for i, result in enumerate(results, start=1):
                            st.write(f"{i}. {result['title']} at {result['company_name']}")
                            st.write(f"Description: {result['description']}")
                            st.write("---")  # Separator between listings
                    else:
                        st.warning("No matching job postings found.")
                
                except Exception as e:
                    st.error(f"An error occurred during the search: {str(e)}")




def logged_in_page():
    # Move title and tagline up
    st.markdown("""
        <style>
            .stApp {
                padding-top: 0;
            }
            .stHeader {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and tagline
    st.title(f"SKILLSYNC")
    st.markdown("### Where the industry meets the minds")

    # Increase gap before welcome message
    st.markdown("&nbsp;&nbsp;")

    # Welcome message
    st.markdown(f"## Welcome, {st.session_state.user.get('name') or st.session_state.user.get('company_name')}!")

    # Sidebar menu
    user_type = st.session_state.user["user_type"]
    if user_type == "job_seeker":
        menu = ["Find Your Job", "Enhance Resume", "Apply for Job"]
    elif user_type == "company":
        menu = ["Manage Job Postings", "View Applicants", "Categorize Resume"]
    
    choice = st.sidebar.selectbox("Menu", menu)

    if user_type == "job_seeker":
        if choice == "Find Your Job":
            find_your_job()
        elif choice == "Enhance Resume":
            st.title("FEATURE YET TO BE ADDED")
        elif choice == "Apply for Job":
            job_application()
       
    elif user_type == "company":
        if choice == "Manage Job Postings":
            company_job_postings()
        elif choice == "View Applicants":
            view_applicants()
        elif choice == "Categorize Resume":
            categorize_resume()

    if st.sidebar.button("Logout"):
        logout()

def main():
    st.set_page_config(page_title="SkillSync - Login/Signup")
    custom_style = """
    <style>
    body {
        background: linear-gradient(to bottom, #e6f3ff, #b3d9ff);
    }
    </style>
    """

    # Apply the style
    st.markdown(custom_style, unsafe_allow_html=True)
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        logged_in_page()
    else:
        menu = ["Login", "Sign Up"]
        choice = st.sidebar.selectbox("Menu", menu)
        
        if choice == "Login":
            login()
        else:
            signup()

if __name__ == "__main__":
    main()