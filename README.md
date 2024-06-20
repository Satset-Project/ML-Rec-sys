# **Bangkit Capstone Project - SatSet Machine Learning**

## **Project Overview**
Our SatSet app aims to connect users with skilled technicians for various services. To enhance the user experience for our SatSet app, we developed a recommendation system that matches users with technicians based on user preferences, technician skills, experience, and ratings. The recommendation system provides personalized suggestions by analyzing the attributes and preferences specified by the user. It takes into account the user's requirements and recommends technicians with relevant skills and high ratings. The model also leverages historical interaction data between users and technicians to improve its recommendations. It assumes that users who have exhibited similar behavior in the past will have comparable preferences in the future. This personalization optimizes the system's ability to cater to diverse user needs and preferences, significantly enhancing user engagement and satisfaction. By personalizing the recommendations, we optimize the system’s ability to cater to diverse user needs. As we continue to refine the model, we look forward to further improving user satisfaction and engagement.

#### **Datasets**
- **User Profiles:** Comprises 250 entries, each with 6 attributes including user ID, name, location, preferences, technician IDs, and ratings given.

- **Technician Profiles:** Consists of 200 entries, each with 10 attributes including technician ID, name, contact details, skills, experience, certifications, and ratings received.

## **Recommendation System Overview**
The recommendation system developed here is designed to suggest the best technicians for users based on their skills and preferences. This system leverages both content-based filtering and collaborative filtering methods, integrating them into a hybrid model to enhance recommendation accuracy. Content-based filtering focuses on the attributes of the items (technicians) to make recommendations. It uses the information provided about the technicians, such as their skills, experience, certifications, and ratings, to find the best matches for a user's skill requirements. Collaborative filtering recommends items based on the past interactions of users with items. In this context, it predicts the rating a user might give to a technician based on the ratings given by similar users. The hybrid model combines content-based and collaborative filtering approaches to leverage the strengths of both methods, providing more accurate and personalized recommendations.

## **Tools/Libraries**
- TensorFlow
- Keras
- Pandas
- NumPy
- scikit-learn
- FastAPI
- Pickle
- Keras Tuner
- TfidfVectorizer
- StandardScaler
- OneHotEncoder
- Pydantic
- Jupyter Notebook

## **Data Preprocessing**
1. The datasets are loaded and initial exploration is conducted to understand their structure and content.
2. Missing values in the 'skills' and 'certifications' columns are handled by filling them with empty strings and 'None,' respectively. Experience and ratings columns are also filled with zeros where necessary.
3. Irrelevant columns such as 'email,' 'phonenumber,' 'location,' and 'address' are dropped to simplify the dataset and focus on relevant features.
4. The 'skills' and 'certifications' columns are converted to lowercase to ensure uniformity.
5. TF-IDF Vectorization is applied to the 'skills' column to convert text data into numerical features. This helps in quantifying the importance of different skills.
6. Standard scaling is applied to 'experience' and 'ratingsreceived' columns to normalize the data and improve model performance.
7. One-hot encoding is used for the 'certifications' column to convert categorical data into a format suitable for machine learning models.
8. Extracted and transformed user ratings for technicians into a DataFrame.

## **Model Development**
1. For content-based, combined TF-IDF transformed skills, standardized experience, one-hot encoded certifications, and standardized ratings into a single feature array.
2. For collaborative filtering, mapped user and technician IDs to indices for embedding layers in the neural network model.

## **Model Training**
1. For content-based, we built a neural network using Keras with a sequential model. The architecture consists of dense layers with ReLU activation, dropout layers for regularization and final dense layer with linear activation. We divided our dataset into training and validation sets. We used the Adam optimizer and mean squared error (MSE) loss function. We trained the model for 50 epochs with a batch size of 32. 
2. For collaborative, we built a neural network using Keras included embedding layers for user and technician IDs, dense layers with ReLU activation, dropout layers for regularization, and a final dense layer with linear activation for rating prediction. We divided our dataset into training and test sets. We used Adam optimizer and mean_squared_error loss. We trained the model for 10 epochs with a batch size of 64 and validation data.

## **Model Evaluation**
1. For content-based, we evaluate the model performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE). Our content-based model achieved an MSE of 0.014 and MAE of 0.072 indicating relatively low prediction error and reasonable accuracy.
2. For collaborative, we evaluate the model performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE). Our collaborative model achieved an MSE of 0.553 and MAE of 0.622 indicating better performance and more accurate predictions.

## **Model Optimization**
1. We employed the Hyperband tuner from Keras Tuner to optimized the content-based model. Tuned hyperparameters like the number of units in dense layers, dropout rates, and learning rate. Prevent overfitting by stopping training when validation performance ceases to improve.

2. We employed the Hyperband tuner from Keras Tuner to optimized the collaborative model. Tuned hyperparameters like embedding size, number of dense layers, units per layer, dropout rates, and learning rate. The Test MAE decreased to 0.518 and the MSE decreased to 0.399 indicating the optimization process successfully enhanced the model's performance, resulting in better generalization and accuracy on the test set.

## **Model Deployment**
We deployed the recommendation systems models for technician recommendation using FastAPI. The application includes endpoints for content-based filtering, collaborative filtering, and hybrid recommendations. Each endpoint handles specific input, processes the data, makes predictions using the respective models, and returns the best matching technician.