from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
# Function to load the model
def load_model():
    model_path = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_path)
    return model

# Function to perform inference
def get_similar_response(question,embedding,model):

    # Encode the question using the model
    question_embedding = model.encode(question)
    
    # Calculate cosine similarity between question and responses
    cosine_sim = cosine_similarity([question_embedding], embedding).reshape(-1)
    # Find the index of the most similar response
    most_similar_index= np.argmax(cosine_sim)
    highest_cosine_similarity = cosine_sim.max()
    return most_similar_index, highest_cosine_similarity
