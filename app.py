from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle

# Load the saved model and preprocessing artifacts
model = tf.keras.models.load_model('technician_recommendation_model_advanced.h5')
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('certifications_encoded_columns.pkl', 'rb') as f:
    certifications_encoded_columns = pickle.load(f)

# Load the original data
data = pd.read_csv('technicians.csv')
original_data = data.copy()

# Preprocess the data
data['skills'] = data['skills'].fillna('')
data['certifications'] = data['certifications'].fillna('')
skills_tfidf = tfidf.transform(data['skills']).toarray()
data['experience'] = data['experience'].fillna(0)
data['ratingsreceived'] = data['ratingsreceived'].fillna(0)
data[['experience', 'ratingsreceived']] = scaler.transform(data[['experience', 'ratingsreceived']])
certifications_encoded = pd.get_dummies(data['certifications']).reindex(columns=certifications_encoded_columns, fill_value=0)
X_exp = data['experience'].values.reshape(-1, 1)
X_rating = data['ratingsreceived'].values.reshape(-1, 1)
X_cert = certifications_encoded.values
X = np.hstack([skills_tfidf, X_exp, X_cert, X_rating])

def predict_best_technician(user_skill):
    # Preprocess the user input skill
    user_skill_tfidf = tfidf.transform([user_skill]).toarray()
    
    # Prepare the input data
    X_input = np.hstack([user_skill_tfidf, np.zeros((1, X.shape[1] - user_skill_tfidf.shape[1]))])
    
    # Predict scores for the user input skill
    predicted_score = model.predict(X_input).flatten()[0]
    
    # Combine with experience, certifications, and ratings
    best_match_score = -1
    best_technician_index = -1
    
    for idx in range(X.shape[0]):
        technician = data.iloc[idx]
        skill_match = user_skill.lower() in technician['skills'].lower()  # Ensure exact phrase matching
        if skill_match:
            combined_score = (predicted_score + 
                              technician['experience'] + 
                              technician['ratingsreceived'] + 
                              certifications_encoded.iloc[idx].sum())
            if combined_score > best_match_score:
                best_match_score = combined_score
                best_technician_index = idx
    
    if best_technician_index != -1:
        return original_data.iloc[best_technician_index]
    else:
        return "No matching technician found."

app = FastAPI()

class SkillInput(BaseModel):
    skill: str

@app.post("/recommend")
def recommend_technician(skill_input: SkillInput):
    user_skill = skill_input.skill
    recommended_technician = predict_best_technician(user_skill)
    return recommended_technician.to_dict() if isinstance(recommended_technician, pd.Series) else {"message": recommended_technician}
