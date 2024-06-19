from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd

app = FastAPI()

# Load models and preprocessing objects
try:
    content_based_model = tf.keras.models.load_model('content_based_filtering.h5')
    collaborative_model = tf.keras.models.load_model('collaborative_filtering.h5')
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

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
try:
    technicians_df = pd.read_csv('technicians.csv')
except Exception as e:
    raise RuntimeError(f"Error loading technicians.csv: {e}")

ori_technician = technicians_df.copy()

# Preprocessing
tfidf_vectorizer = tfidf
skills_tfidf = tfidf_vectorizer.transform(technicians_df['skills']).toarray()

# Scale experience and ratings
technicians_df['experience'] = scaler_experience.transform(technicians_df[['experience']])
technicians_df['ratingsreceived'] = scaler_ratings.transform(technicians_df[['ratingsreceived']])

# Encode certifications
certifications_encoded_sparse = encoder.transform(technicians_df[['certifications']])
certifications_encoded = certifications_encoded_sparse.toarray()

X_exp = technicians_df['experience'].values.reshape(-1, 1)
X_rating = technicians_df['ratingsreceived'].values.reshape(-1, 1)
X_cert = certifications_encoded

# Combine features into a single array
X = np.hstack([skills_tfidf, X_exp, X_cert, X_rating])

# Pydantic models for request and response validation
class ContentRequest(BaseModel):
    user_skill: str

class CollaborativeRequest(BaseModel):
    user_id: int

class HybridRequest(BaseModel):
    user_id: int
    user_skill: str

@app.post("/recommend/content-based/")
def content_based_filtering(request: ContentRequest):
    user_skill = request.user_skill

    try:
        user_skill_tfidf = tfidf_vectorizer.transform([user_skill]).toarray()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing user skill: {e}")

    X_input = np.hstack([user_skill_tfidf, np.zeros((1, X.shape[1] - user_skill_tfidf.shape[1]))])

    try:
        predicted_score = content_based_model.predict(X_input).flatten()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting score: {e}")

    best_match_score = -1
    best_technician_index = -1

    for idx in range(X.shape[0]):
        technician = technicians_df.iloc[idx]
        skill_match = user_skill.lower() in technician['skills'].lower()
        if skill_match:
            combined_score = (predicted_score + technician['experience'] +
                              technician['ratingsreceived'] +
                              certifications_encoded_sparse[idx].sum())
            if combined_score > best_match_score:
                best_match_score = combined_score
                best_technician_index = idx

    if best_technician_index != -1:
        best_technician = ori_technician.iloc[best_technician_index]
        return {"message": best_technician}
    else:
        raise HTTPException(status_code=404, detail="No matching technician found.")

@app.post("/recommend/collaborative/")
def collaborative_filtering(request: CollaborativeRequest):
    user_id = request.user_id

    if user_id not in user_id_map:
        raise HTTPException(status_code=404, detail="User ID not found.")

    user_idx = user_id_map[user_id]

    try:
        user_input = np.array([user_idx] * len(technician_id_map))
        technician_input = np.arange(len(technician_id_map))
        predicted_ratings = collaborative_model.predict([user_input, technician_input])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting ratings: {e}")

    predictions = [(tech_id, pred) for tech_id, pred in zip(technician_id_map.keys(), predicted_ratings.flatten())]
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    top_recommendation = sorted_predictions[0]
    top_technician = ori_technician[technicians_df['technicianid'] == top_recommendation[0]].iloc[0]

    return {"message": top_technician}

@app.post("/recommend/hybrid/")
def hybrid_recommendation(request: HybridRequest):
    user_id = request.user_id
    user_skill = request.user_skill

    # Content-based part
    try:
        user_skill_tfidf = tfidf_vectorizer.transform([user_skill]).toarray()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing user skill: {e}")

    X_content_input = np.hstack([user_skill_tfidf, np.zeros((1, X.shape[1] - user_skill_tfidf.shape[1]))])

    try:
        predicted_content_score = content_based_model.predict(X_content_input).flatten()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting content-based score: {e}")

    # Collaborative part
    if user_id not in user_id_map:
        raise HTTPException(status_code=404, detail="User ID not found.")

    user_idx = user_id_map[user_id]

    try:
        user_input = np.array([user_idx] * len(technician_id_map))
        technician_input = np.arange(len(technician_id_map))
        predicted_ratings = collaborative_model.predict([user_input, technician_input])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting collaborative ratings: {e}")

    # Hybrid recommendation
    predictions = [(tech_id, (pred + predicted_content_score) / 2) for tech_id, pred in zip(technician_id_map.keys(), predicted_ratings.flatten())]
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    top_recommendation = sorted_predictions[0]
    top_technician = ori_technician[technicians_df['technicianid'] == top_recommendation[0]].iloc[0]

    return {"message": top_technician}