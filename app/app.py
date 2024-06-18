from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Load models and preprocessing objects
content_based_model = tf.keras.models.load_model('content_based_filtering.h5')
collaborative_model = tf.keras.models.load_model('collaborative_filtering.h5')

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('scaler_experience.pkl', 'rb') as f:
    scaler_experience = pickle.load(f)
with open('scaler_ratings.pkl', 'rb') as f:
    scaler_ratings = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('user_id_map.pkl', 'rb') as f:
    user_id_map = pickle.load(f)
with open('technician_id_map.pkl', 'rb') as f:
    technician_id_map = pickle.load(f)

# Load technicians data
technicians_df = pd.read_csv('technicians.csv')
certifications_encoded_sparse = encoder.transform(technicians_df[['certifications']])
certifications_encoded = certifications_encoded_sparse.toarray()

# Define FastAPI app
app = FastAPI()

# Define request and response models
class UserSkillRequest(BaseModel):
    user_skill: str

class HybridRecommendationRequest(BaseModel):
    user_id: int
    user_skill: str

class TechnicianResponse(BaseModel):
    technician_id: int
    name: str
    skills: str
    certifications: str
    experience: float
    ratingsreceived: float
    phonenumber: str
    email: str
    address: str
    location: str

def content_based_filtering(user_skill):
    # Preprocess the user input skill
    user_skill_tfidf = tfidf.transform([user_skill]).toarray()
    
    # Prepare the input data
    X = np.hstack([
        tfidf.transform(technicians_df['skills']).toarray(),
        scaler_experience.transform(technicians_df[['experience']]),
        scaler_ratings.transform(technicians_df[['ratingsreceived']]),
        certifications_encoded
    ])
    
    if user_skill_tfidf.shape[1] < X.shape[1]:
        X_input = np.hstack([user_skill_tfidf, np.zeros((1, X.shape[1] - user_skill_tfidf.shape[1]))])
    else:
        X_input = user_skill_tfidf[:, :X.shape[1]]

    # Predict scores for the user input skill
    predicted_score = content_based_model.predict(X_input).flatten()[0]
    
    # Combine with experience, certifications, and ratings
    best_match_score = -1
    best_technician_index = -1
    
    for idx in range(X.shape[0]):
        technician = technicians_df.iloc[idx]
        skill_match = user_skill.lower() in technician['skills'].lower()  # Ensure exact phrase matching
        if skill_match:
            combined_score = (predicted_score + 
                              technician['experience'] + 
                              technician['ratingsreceived'] + 
                              certifications_encoded_sparse[idx].sum())
            if combined_score > best_match_score:
                best_match_score = combined_score
                best_technician_index = idx
    
    if best_technician_index != -1:
        return technicians_df.iloc[best_technician_index]
    else:
        return None

def collaborative_filtering(user_id):
    # Map the user ID to the corresponding index
    if user_id not in user_id_map:
        return None

    user_idx = user_id_map[user_id]
    num_technicians = len(technician_id_map)
    
    # Predict ratings for all technicians for the given user
    user_input = np.array([user_idx] * num_technicians)
    technician_input = np.arange(num_technicians)
    predicted_ratings = collaborative_model.predict([user_input, technician_input])
    
    # Get the highest-rated technician
    best_technician_idx = np.argmax(predicted_ratings)
    best_technician_id = list(technician_id_map.keys())[best_technician_idx]
    best_technician = technicians_df[technicians_df['technicianid'] == best_technician_id].iloc[0]
    
    return best_technician

def hybrid_recommendation(user_id, user_skill):
    # Get collaborative filtering recommendation
    collab_recommendation = collaborative_filtering(user_id)
    
    # Get content-based filtering recommendation
    content_recommendation = content_based_filtering(user_skill)
    
    if collab_recommendation is None and content_recommendation is None:
        raise HTTPException(status_code=404, detail="No matching technician found")

    # Extract and compute scores
    if collab_recommendation is not None:
        collab_score = (collab_recommendation['ratingsreceived'] + 
                        collab_recommendation['experience'] + 
                        certifications_encoded_sparse[collab_recommendation.name].sum())
    else:
        collab_score = -1
        
    if content_recommendation is not None:
        content_score = (content_recommendation['ratingsreceived'] + 
                         content_recommendation['experience'] + 
                         certifications_encoded_sparse[content_recommendation.name].sum())
    else:
        content_score = -1
    
    # Combine the scores (simple weighted average or other logic can be applied here)
    if collab_score >= content_score:
        return collab_recommendation
    else:
        return content_recommendation

@app.post("/content_based_recommend/", response_model=TechnicianResponse)
def content_based_recommend(request: UserSkillRequest):
    recommendation = content_based_filtering(request.user_skill)
    
    if recommendation is not None:
        return TechnicianResponse(
            technician_id=recommendation['technicianid'],
            name=recommendation['name'],
            skills=recommendation['skills'],
            certifications=recommendation['certifications'],
            experience=recommendation['experience'],
            ratingsreceived=recommendation['ratingsreceived'],
            phonenumber=recommendation['phonenumber'],
            email=recommendation['email'],
            address=recommendation['address'],
            location=recommendation['location']
        )
    else:
        raise HTTPException(status_code=404, detail="No matching technician found")

@app.post("/hybrid_recommend/", response_model=TechnicianResponse)
def hybrid_recommend(request: HybridRecommendationRequest):
    recommendation = hybrid_recommendation(request.user_id, request.user_skill)
    
    if recommendation is not None:
        return TechnicianResponse(
            technician_id=recommendation['technicianid'],
            name=recommendation['name'],
            skills=recommendation['skills'],
            certifications=recommendation['certifications'],
            experience=recommendation['experience'],
            ratingsreceived=recommendation['ratingsreceived'],
            phonenumber=recommendation['phonenumber'],
            email=recommendation['email'],
            address=recommendation['address'],
            location=recommendation['location']
        )
    else:
        raise HTTPException(status_code=404, detail="No matching technician found")