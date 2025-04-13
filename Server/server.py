'''import json
import pickle
import numpy as np
import pandas as pd
import PyPDF2
import os
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app,origins=["https://ey-1-0shs.onrender.com"])

# Initialize global variables
__skills = None
__data_columns = None
__model = None
__vectorizer = None  # To store the vectorizer for transforming input skills
__df = None  # DataFrame for course data


def get_suggestions(skills):
    """
    Function to predict the recommended courses based on input skills
    """
    if not __model or not __vectorizer or __df is None:
        raise ValueError("Model, vectorizer, or course data not loaded. Please load them first.")

    # Transform the input skills using the vectorizer
    skills_vector = __vectorizer.transform([skills])  # skills is a string of input skills

    # Calculate cosine similarities with the course skills vectors
    cosine_similarities = cosine_similarity(skills_vector, np.vstack(
        __df['Skills_Vector'].values))  # Use np.vstack to convert the list of arrays to a 2D array

    # Get the top 5 recommendations based on cosine similarity
    top_indexes = cosine_similarities[0].argsort()[-5:][::-1]

    # Retrieve Titles and URLs for the top courses
    recommended_courses = __df.iloc[top_indexes][['Title', 'URL']]

    return recommended_courses

def load_saved_skills():
    """
    Function to load the skills and courses from the JSON, pickle files, and CSV
    """
    global __skills
    global __data_columns
    global __model
    global __vectorizer
    global __df

    print("Loading saved skills and course data...")

    # Load skills from JSON file
    try:
        with open("./artifacts/skill_columns.json", 'r') as f:
            __data_columns = json.load(f)
            __skills = list(__data_columns.values())  # Get values (skills)

        print("Successfully loaded skills from skill_columns.json.")
    except Exception as e:
        print(f"Error loading skill columns JSON: {e}")

    # Load the model pickle file
    try:
        with open("./artifacts/model_pickle.pickle", 'rb') as f:
            __model = pickle.load(f)
        print(f"Successfully loaded the model from pickle. Model type: {type(__model)}")
    except Exception as e:
        print(f"Error loading model pickle file: {e}")

    # Load vectorizer pickle file
    try:
        with open("./artifacts/vectorizer_pickle.pickle", 'rb') as f:
            __vectorizer = pickle.load(f)
        print(f"Successfully loaded the vectorizer. Vectorizer type: {type(__vectorizer)}")
    except Exception as e:
        print(f"Error loading vectorizer pickle file: {e}")

    # Load course data from CSV
    try:
        __df = pd.read_csv("./artifacts/Online_courses.csv")
        print(f"Successfully loaded course data from Online_courses.csv.")

        # Clean the 'Skills' column: remove NaN and empty strings
        __df['Skills'] = __df['Skills'].fillna('')  # Replace NaNs with empty strings

        # Create the Skills_Vector column by transforming the 'Skills' column
        skills_vectors = __vectorizer.transform(__df['Skills'])

        # Convert to a dense matrix and store in the 'Skills_Vector' column
        __df['Skills_Vector'] = skills_vectors.toarray().tolist()  # Convert sparse matrix to dense list

        print(f"Successfully created the 'Skills_Vector' column.")
    except Exception as e:
        print(f"Error loading course data: {e}")

def get_skills():
    """
    Function to return the list of available skills.
    """
    global __skills
    if not __skills:
        raise ValueError("Skills data not loaded. Please load skills first.")
    return __skills

@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON input
    if not data or 'input' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    input_skills = data['input']
    try:
        # Get course suggestions based on the input skills
        suggestions = get_suggestions(input_skills)
        return jsonify(suggestions.to_dict(orient='records'))  # Return courses as a JSON list
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_skills', methods=['GET'])
def skills():
    try:
        skills = get_skills()
        return jsonify({'skills': skills})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load skills from Pickle file
with open("./artifacts/skills.pkl", "rb") as pkl_file:
    skills_columns = pickle.load(pkl_file)

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_matching_skills(text, skills_list):
    """Extract skills present in the given text."""
    text_lower = text.lower()
    matched_skills = {skill for skill in skills_list if skill.lower() in text_lower}
    return matched_skills

@app.route("/compare_skills", methods=["POST"])
def compare_skills():
    if "resume" not in request.files:
        return jsonify({"error": "No resume file provided"}), 400

    resume = request.files["resume"]
    job_description = request.form.get("job_description", "").strip()

    if resume.filename == "" or not job_description:
        return jsonify({"error": "Both resume and job description are required"}), 400

    # Save the resume file
    resume_path = os.path.join(app.config["UPLOAD_FOLDER"], resume.filename)
    resume.save(resume_path)

    # Extract text from resume
    resume_text = extract_text_from_pdf(resume_path)

    # Extract matching skills
    resume_skills = extract_matching_skills(resume_text, skills_columns)
    job_skills = extract_matching_skills(job_description, skills_columns)

    # Find missing skills
    matched_skills = resume_skills & job_skills
    missing_skills = job_skills - resume_skills

    

    response = {
        "message": "Resume processed successfully",
        "resume_filename": resume.filename,
        "job_description": job_description,
        "matched_skills": list(matched_skills),
        "missing_skills": list(missing_skills),
        
    }

    return jsonify(response), 200
load_saved_skills()
if __name__ == '__main__':
    # Load skills and course data from the JSON, pickle files, and CSV

    # Run the Flask application
    app.run(debug=True, port=5002)'''
import json
import pickle
import numpy as np
import pandas as pd
import PyPDF2
import os
from pathlib import Path
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://ey-1-0shs.onrender.com"])

# Initialize global variables
__skills = None
__data_columns = None
__model = None
__vectorizer = None
__df = None

# Determine base directory and artifact paths
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / 'artifacts'

def get_artifact_path(filename):
    """Get absolute path to artifact file"""
    return str(ARTIFACTS_DIR / filename)

def check_files_exist():
    """Verify all required files exist before loading"""
    required_files = [
        'skill_columns.json',
        'model_pickle.pickle',
        'vectorizer_pickle.pickle',
        'Online_courses.csv',
        'skills.pkl'
    ]
    
    missing = [f for f in required_files if not os.path.exists(get_artifact_path(f))]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

def load_saved_skills():
    """Load skills, models, and course data"""
    global __skills, __data_columns, __model, __vectorizer, __df
    
    print("Loading saved skills and course data...")
    
    try:
        # Load skills from JSON
        with open(get_artifact_path('skill_columns.json'), 'r') as f:
            __data_columns = json.load(f)
            __skills = list(__data_columns.values())
        print("Skills loaded successfully")

        # Load course data first
        __df = pd.read_csv(get_artifact_path('Online_courses.csv'))
        __df['Skills'] = __df['Skills'].fillna('')
        print("Course data loaded successfully")

        # Load vectorizer
        with open(get_artifact_path('vectorizer_pickle.pickle'), 'rb') as f:
            __vectorizer = pickle.load(f)
        
        # Process course vectors
        skills_vectors = __vectorizer.transform(__df['Skills'])
        __df['Skills_Vector'] = [x.tolist() for x in skills_vectors.toarray()]
        print("Vectorizer loaded and data processed")

        # Load model last
        with open(get_artifact_path('model_pickle.pickle'), 'rb') as f:
            __model = pickle.load(f)
        print("Model loaded successfully")

        if not all([__model, __vectorizer, __df is not None]):
            raise RuntimeError("Failed to load one or more components")

    except Exception as e:
        # Clean up if error occurs
        __skills = None
        __data_columns = None
        __model = None
        __vectorizer = None
        __df = None
        print(f"Error loading components: {str(e)}")
        raise

def get_suggestions(skills):
    """Get course recommendations based on skills"""
    if not all([__model, __vectorizer, __df is not None]):
        raise ValueError("Model components not loaded")
    
    skills_vector = __vectorizer.transform([skills])
    cosine_similarities = cosine_similarity(skills_vector, np.vstack(__df['Skills_Vector'].values))
    top_indexes = cosine_similarities[0].argsort()[-5:][::-1]
    return __df.iloc[top_indexes][['Title', 'URL']]

def get_skills():
    """Get available skills list"""
    if not __skills:
        raise ValueError("Skills data not loaded")
    return __skills

# Routes
@app.route('/')
def home():
    return "Upskill Hub API is running!"

@app.route('/health')
def health_check():
    status = {
        'model_loaded': __model is not None,
        'vectorizer_loaded': __vectorizer is not None,
        'data_loaded': __df is not None,
        'skills_loaded': __skills is not None
    }
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'input' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    try:
        suggestions = get_suggestions(data['input'])
        return jsonify(suggestions.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_skills', methods=['GET'])
def skills():
    try:
        return jsonify({'skills': get_skills()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# PDF Processing Routes
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

with open(get_artifact_path('skills.pkl'), "rb") as pkl_file:
    skills_columns = pickle.load(pkl_file)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_matching_skills(text, skills_list):
    """Match skills from text"""
    text_lower = text.lower()
    return {skill for skill in skills_list if skill.lower() in text_lower}

@app.route("/compare_skills", methods=["POST"])
def compare_skills():
    if "resume" not in request.files:
        return jsonify({"error": "No resume file provided"}), 400

    resume = request.files["resume"]
    job_description = request.form.get("job_description", "").strip()

    if not (resume.filename and job_description):
        return jsonify({"error": "Both resume and job description required"}), 400

    try:
        resume_path = os.path.join(app.config["UPLOAD_FOLDER"], resume.filename)
        resume.save(resume_path)
        resume_text = extract_text_from_pdf(resume_path)

        resume_skills = extract_matching_skills(resume_text, skills_columns)
        job_skills = extract_matching_skills(job_description, skills_columns)

        return jsonify({
            "matched_skills": list(resume_skills & job_skills),
            "missing_skills": list(job_skills - resume_skills)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Initialize application
try:
    check_files_exist()
    load_saved_skills()
    print("All components loaded successfully!")
except Exception as e:
    print(f"Failed to initialize: {str(e)}")
    raise

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5002)))
