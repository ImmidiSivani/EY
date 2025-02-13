import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def save_skills_to_pickle():
    """
    Function to save the skills to a pickle file for later use
    """
    try:
        with open("./artifacts/skills_pickle.pickle", 'wb') as f:
            pickle.dump(__skills, f)
        print("Successfully saved skills to skills_pickle.pickle.")
    except Exception as e:
        print(f"Error saving skills to pickle: {e}")


def save_skills_to_json():
    """
    Function to save the skills to a JSON file (if needed to update it)
    """
    try:
        with open("./artifacts/skill_columns.json", 'w') as f:
            json.dump(__data_columns, f, indent=4)
        print("Successfully saved skills to skill_columns.json.")
    except Exception as e:
        print(f"Error saving skills to JSON: {e}")

def get_skills():
    """
    Function to return the list of available skills.
    """
    global __skills
    if not __skills:
        raise ValueError("Skills data not loaded. Please load skills first.")
    return __skills


if __name__ == '__main__':
    # Load skills and course data from the JSON, pickle files, and CSV
    load_saved_skills()

    # Example input skills (replace with actual skills for prediction)
    input_skills = " machine learning"

    # Get suggested courses for the input skills
    suggestions = get_suggestions(input_skills)
    print(f"Suggested courses for '{input_skills}':\n", suggestions)

    # Optionally, save the skills to pickle (if needed for persistence)
    save_skills_to_pickle()

    # If you want to update the JSON file, you can save it as well
    save_skills_to_json()
