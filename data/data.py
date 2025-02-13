import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("travel.csv")
    return df

df = load_data()

# Load pre-trained models (assuming they are already saved)
scaler = joblib.load("scaler.pkl")  # StandardScaler for numerical features
kmeans = joblib.load("kmeans.pkl")  # Trained K-Means model

# Define clusters with top destinations
top_destinations = {
    0: ["Bali", "Paris", "London"],
    1: ["New York", "Tokyo", "Dubai"],
    2: ["Barcelona", "Sydney", "Bangkok"]
}

# Set up page style
page_style = '''
<style>
    .stApp {
        background-color: #e6f7ff;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #ff5733;
        background-color: #ffcccb;
        padding: 10px;
        border-radius: 10px;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #2c3e50;
        background-color: #d1e7dd;
        padding: 5px;
        border-radius: 10px;
    }
    .box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        color: #2c3e50;
        font-size: 18px;
    }
</style>
'''
st.markdown(page_style, unsafe_allow_html=True)

def main():
    st.markdown("<h1 class='title'>✈️ Travel Destination Recommender</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Enter your trip details to get personalized recommendations! 🌍</p>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='box'>", unsafe_allow_html=True)
        
        # User Inputs
        duration = st.number_input("⏳ Trip Duration (days)", min_value=1, max_value=30, value=5)
        budget = st.slider("💰 Estimated Budget ($)", min_value=100, max_value=30000, value=2000, step=100)
        travelers = st.number_input("👥 Number of Travelers", min_value=1, max_value=10, value=2)
        
        if st.button("🎯 Find My Travel Cluster & Recommendations", help="Click to get recommendations!"):  
            # Preprocess user input
            user_input = np.array([[duration, budget, travelers]])
            user_input_scaled = scaler.transform(user_input)
            
            # Predict cluster
            predicted_cluster = kmeans.predict(user_input_scaled)[0]
            
            # Recommend destinations
            recommended_places = top_destinations.get(predicted_cluster, ["No recommendations found"])
            
            st.success(f"✅ You belong to Travel Cluster {predicted_cluster}!")
            
            st.markdown("<h3 style='color: #ff5733; background-color: #ffeb99; padding: 5px; border-radius: 5px;'>🌍 Recommended Destinations:</h3>", unsafe_allow_html=True)
            
            for cluster, recommended_destinations in top_destinations.items():
                if cluster == predicted_cluster:
                    st.write(f"- {', '.join(recommended_destinations)}")
            
        st.markdown("</div>", unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()
