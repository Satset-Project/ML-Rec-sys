from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
import firebase_admin
from firebase_admin import credentials, firestore
import threading
import os

app = FastAPI()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Initialize the Firebase Admin SDK
cred = credentials.Certificate("servicesaccount.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# Load the saved model and preprocessing artifacts
try:
    model = tf.keras.models.load_model('technician_recommendation_model_advanced.h5', compile=False)
except TypeError as e:
    raise RuntimeError(f"Model deserialization error: {str(e)}")

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
    
def update_order(order_id, technician_email):
    technician_ref = db.collection('technicians').where('email', '==', technician_email).get()
    order_ref = db.collection('orders').document(order_id)
    for doc in technician_ref:
        doc_id = doc.id
        # Now you have the document ID
    order_ref.update({'technicianId': doc_id})
    order_ref.update({'status': 'Assigned'})
    print(f'Order {order_id} assigned to technician {doc_id}')

def on_snapshot(doc_snapshot, changes, read_time):
    for change in changes:
        if change.type.name == 'ADDED' or change.type.name == 'MODIFIED':
            doc = change.document.to_dict()
            if doc.get('status') == 'Pending' and doc.get('technicianId') is None:
                service_type = doc.get('serviceType')
                best_technician = predict_best_technician(service_type)
                if best_technician is not None:
                    technician_email = best_technician['email'] 
                    update_order(change.document.id, technician_email)

# Firestore listener thread
def start_firestore_listener():
    orders_ref = db.collection('orders')
    orders_ref.on_snapshot(on_snapshot)

listener_thread = threading.Thread(target=start_firestore_listener, daemon=True)
listener_thread.start()

class SkillInput(BaseModel):
    skill: str

def clean_nan_values(data):
    if isinstance(data, pd.Series):
        return data.replace({np.nan: None}).to_dict()
    if isinstance(data, dict):
        return {k: (None if pd.isna(v) else v) for k, v in data.items()}
    return data

@app.post("/recommend")
def recommend_technician(skill_input: SkillInput):
    try:
        user_skill = skill_input.skill.strip()
        if not user_skill:
            raise ValueError("Skill cannot be empty.")
        
        recommended_technician = predict_best_technician(user_skill)
        if isinstance(recommended_technician, pd.Series):
            cleaned_data = clean_nan_values(recommended_technician)
            return cleaned_data
        else:
            return {"message": recommended_technician}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.get("/recommendHello")
def recommend_hello():
    return {"message": "Hello World!"}
