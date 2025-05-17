import json
import pickle
import numpy as np
import pandas as pd
import PyPDF2
import os
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS  # Import CORS
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


app = Flask(__name__)
CORS(app,origins=["https://ey-1-0shs.onrender.com"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '..', 'Backend', '.env')
load_dotenv(dotenv_path=ENV_PATH, override=True)
api_key = os.getenv("GROQ_API_KEY") 
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables!")

client = Groq(api_key=api_key)



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
    current_dir = Path(__file__).parent  # Gets directory where this script lives
    data_path = current_dir / "artifacts" / "Online_Courses.csv"
    try:
        __df = pd.read_csv(data_path)
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

@app.route('/status', methods=['GET'])
def status():
    """Check if models and data are loaded."""
    status = {
        'skills_loaded': __skills is not None,
        'model_loaded': __model is not None,
        'vectorizer_loaded': __vectorizer is not None,
        'course_data_loaded': __df is not None and not __df.empty  # ← Correct check
    }
    return jsonify(status)
@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    try:
        data = request.get_json()
        topic = data.get('topic', '')
        difficulty = data.get('difficulty', 'medium')
        num_questions = data.get('num_questions', 5)

        if not topic:
            return jsonify({'error': 'Topic is required'}), 400

        prompt = f"""
        Generate a {difficulty} difficulty quiz about {topic} with {num_questions} MCQs.
        Respond with a JSON object with this format:

        {{
          "questions": [
            {{
              "question": "What is Python used for?",
              "options": {{
                "a": "Web development",
                "b": "Data analysis",
                "c": "Machine learning",
                "d": "All of the above"
              }},
              "answer": "d",
              "explanation": "Python is used in all of these areas."
            }}
          ]
        }}
        Respond with only valid JSON, without any markdown or extra text.
        """

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content.strip()

        # Remove markdown fences if any
        if content.startswith("```json"):
            content = content[len("```json"):].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        quiz_data = json.loads(content)

        # Return the actual questions list, not the full object
        return jsonify({'questions': quiz_data['questions']})

    except json.JSONDecodeError as e:
        return jsonify({'error': 'Failed to parse quiz questions', 'details': str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

load_saved_skills()
if __name__ == '__main__':
    # Load skills and course data from the JSON, pickle files, and CSV

    # Run the Flask application
    app.run(debug=True, port=5002)
