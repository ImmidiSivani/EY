import json
import pickle

# Load skills from JSON
skills_json_path = "skill_columns.json"

with open(skills_json_path, "r", encoding="utf-8") as file:
    skills_data = json.load(file)
    skills_columns = list(skills_data.values())

# Save as Pickle file
with open("skills.pkl", "wb") as pkl_file:
    pickle.dump(skills_columns, pkl_file)

print("Pickle file saved as skills.pkl")
