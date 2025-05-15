import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime


st.set_page_config(
    page_title="IPL 2025 Match Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)


if not os.path.exists('models'):
    os.makedirs('models')


for dir_path in ['data/raw', 'data/processed']:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_models():
    try:
        models = {}
        model_files = {
            'Logistic Regression': 'models/logistic_regression_model.pkl',
            'Random Forest': 'models/random_forest_model.pkl',
            'SVM': 'models/svm_model.pkl',
            'LightGBM': 'models/lightgbm_model.pkl'
        }
        
        for name, file_path in model_files.items():
            if os.path.exists(file_path):
                models[name] = joblib.load(file_path)
            else:
                st.warning(f"Model file not found: {file_path}")
        
        if not models:
            st.error("No models were loaded successfully. Please train the models first.")
            return None
            
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None


def predict_match_outcome(features, models):
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        try:
            
            pred = model.predict(features)
           
            prob = model.predict_proba(features)[0]
            
            predictions[name] = pred[0]
            probabilities[name] = prob
        except Exception as e:
            st.error(f"Error with {name} model: {str(e)}")
            predictions[name] = None
            probabilities[name] = None
    
    return predictions, probabilities


def main():
    
    st.sidebar.image("https://www.iplt20.com/assets/images/ipl-logo-new-old.png", width=200)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Predict Match", "Model Performance", "About"])
    
    models = load_models()
    
    
    if page == "Home":
        st.title("🏏 IPL 2025 Match Prediction")
        st.markdown("""
        ## Welcome to the IPL Match Predictor!
        
        This application uses machine learning to predict the outcomes of IPL 2025 matches.
        
        ### Features:
        - Predict match outcomes using multiple ML models
        - Compare model performance
        - Visualize prediction probabilities
        
        ### How to use:
        1. Navigate to the 'Predict Match' page
        2. Enter the match details
        3. Get predictions from different models
        
        ### Data Sources:
        - IPL Match Records (2008-2025)
        - Ball-by-Ball Data
        - Team and Player Statistics
        """)
        
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Matches Analyzed", "1000+")
        with col2:
            st.metric("Prediction Accuracy", "54%")
        with col3:
            st.metric("Models Used", "4")
    
   
    elif page == "Predict Match":
        st.title("Predict IPL Match Outcome")
        st.write("Enter match details to get predictions")
        
       
        with st.form("match_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                team1 = st.selectbox("Team 1", [
                    "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans", 
                    "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
                    "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bangalore",
                    "Sunrisers Hyderabad"
                ])
                
                venue = st.selectbox("Venue", [
                    "M.A. Chidambaram Stadium", "Arun Jaitley Stadium", "Narendra Modi Stadium",
                    "Eden Gardens", "BRSABV Ekana Cricket Stadium", "Wankhede Stadium",
                    "Punjab Cricket Association Stadium", "Sawai Mansingh Stadium",
                    "M. Chinnaswamy Stadium", "Rajiv Gandhi International Stadium"
                ])
                
                toss_winner = st.selectbox("Toss Winner", ["Team 1", "Team 2"])
                toss_decision = st.selectbox("Toss Decision", ["Bat", "Field"])
            
            with col2:
                team2 = st.selectbox("Team 2", [
                    "Mumbai Indians", "Royal Challengers Bangalore", "Kolkata Knight Riders",
                    "Chennai Super Kings", "Rajasthan Royals", "Delhi Capitals",
                    "Sunrisers Hyderabad", "Punjab Kings", "Gujarat Titans",
                    "Lucknow Super Giants"
                ])
                
                date = st.date_input("Match Date", datetime.now())
                
                team1_form = st.slider("Team 1 Recent Form (Last 5 matches won)", 0, 5, 2)
                team2_form = st.slider("Team 2 Recent Form (Last 5 matches won)", 0, 5, 2)
            
            submit_button = st.form_submit_button("Predict")
        
       
        if submit_button:
            if team1 == team2:
                st.error("Team 1 and Team 2 cannot be the same!")
            else:
                
               
                team_mappings = joblib.load('data/processed/team_mappings.pkl')
                venue_mappings = joblib.load('data/processed/venue_mappings.pkl')
                
                
                team1_encoded = team_mappings['teams'].get(team1, 0)
                team2_encoded = team_mappings['teams'].get(team2, 0)
                venue_encoded = venue_mappings['venues'].get(venue, 0)
                
               
                toss_winner_encoded = 1 if toss_winner == "Team 1" else 0
                
                
                team1_win_rate = team1_form / 5 
                team2_win_rate = team2_form / 5
                
               
                team1_toss_win_rate = 0.5
                team2_toss_win_rate = 0.5
                
              
                features = np.array([[
                    team1_encoded, 
                    team2_encoded,  
                    venue_encoded, 
                    toss_winner_encoded,  
                    team1_win_rate, 
                    team2_win_rate, 
                    team1_toss_win_rate, 
                    team2_toss_win_rate 
                ]])
                
               
                models = load_models()
                
                
                if models:
                    predictions, probabilities = predict_match_outcome(features, models)
                    
                   
                    st.subheader("Match Prediction Results")
                    
                   
                    model_accuracies = {
                        'Logistic Regression': 0.5251,
                        'Random Forest': 0.5205,
                        'SVM': 0.5434,
                        'LightGBM': 0.4795
                    }
                  
                    votes = {team1: 0, team2: 0}
                    for model, pred in predictions.items():
                        winner = team1 if pred == 1 else team2
                        votes[winner] += 1
                   
                    weighted_votes = {team1: 0, team2: 0}
                    for model, pred in predictions.items():
                        winner = team1 if pred == 1 else team2
                        weighted_votes[winner] += model_accuracies.get(model, 0)
                   
                    avg_probs = {team1: 0, team2: 0}
                    for model, prob in probabilities.items():
                        avg_probs[team1] += prob[1] if prob is not None else 0
                        avg_probs[team2] += prob[0] if prob is not None else 0
                    avg_probs[team1] /= len(probabilities)
                    avg_probs[team2] /= len(probabilities)
                  
                    confidence = max(avg_probs[team1], avg_probs[team2])
                    if 0.45 <= confidence <= 0.55:
                        final_prediction = "Too close to call"
                        confidence_label = "Low Confidence"
                    else:
                        final_prediction = team1 if avg_probs[team1] > avg_probs[team2] else team2
                        confidence_label = "High Confidence" if confidence > 0.65 else "Medium Confidence" if confidence > 0.55 else "Low Confidence"
                  
                    models_agree = max(votes.values())
                   
                  
                    st.markdown(f"### 🔮 Final Ensemble Prediction: **{final_prediction}**")
                    st.write(f"📊 Confidence Level: {confidence*100:.1f}% ({confidence_label})")
                    st.write(f"🧠 Models agree: {models_agree} out of {len(models)}")
                    st.write(f"Majority Voting: {votes}")
                    st.write(f"Weighted Voting: {weighted_votes}")
                    st.write(f"Average Win Probabilities: {avg_probs}")
                   
                    cols = st.columns(len(models))
                    for i, (model_name, prediction) in enumerate(predictions.items()):
                        with cols[i]:
                            winner = team1 if prediction == 1 else team2
                            st.metric(f"{model_name} Prediction", winner)
                           
                            prob = probabilities[model_name]
                            team1_prob = prob[1] * 100 if prob is not None and len(prob) > 1 else 0
                            team2_prob = prob[0] * 100 if prob is not None and len(prob) > 0 else 0
                           
                            fig, ax = plt.subplots(figsize=(4, 3))
                            ax.bar([team1, team2], [team1_prob, team2_prob], color=['#1f77b4', '#ff7f0e'])
                            ax.set_ylabel('Win Probability (%)')
                            ax.set_title(f'{model_name} Prediction')
                            st.pyplot(fig)
                else:
                    st.info("Models are not yet available. Please train the models first.")
                    
                   
                    st.subheader("Sample Prediction (Demo Only)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Winner", team1)
                        st.write(f"Confidence: 65%")
                    with col2:
                        
                        fig, ax = plt.subplots(figsize=(4, 3))
                        ax.bar([team1, team2], [65, 35], color=['#1f77b4', '#ff7f0e'])
                        ax.set_ylabel('Win Probability (%)')
                        ax.set_title('Sample Prediction')
                        st.pyplot(fig)
    
   
    elif page == "Model Performance":
        st.title("Model Performance Metrics")
        st.write("Compare the performance of different prediction models")
        
       
        metrics = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'SVM', 'LightGBM'],
            'Accuracy': [0.5251, 0.5205, 0.5434, 0.4795],
            'Precision': [0.5752, 0.5851, 0.5802, 0.5347],
            'Recall': [0.5372, 0.4545, 0.6281, 0.4463],
            'F1 Score': [0.5556, 0.5116, 0.6032, 0.4865]
        })
        
        
        st.dataframe(metrics.set_index('Model'))
        
        
        tab1, tab2 = st.tabs(["Accuracy Comparison", "All Metrics"])
        
        with tab1:
           
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Model', y='Accuracy', data=metrics, ax=ax)
            ax.set_title('Model Accuracy Comparison')
            ax.set_ylim(0.4, 0.6)  
            st.pyplot(fig)
        
        with tab2:
          
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
            
         
            metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            angles = np.linspace(0, 2*np.pi, len(metrics_cols), endpoint=False).tolist()
            angles += angles[:1]  
            
           
            for i, model in enumerate(metrics['Model']):
                values = metrics.loc[i, metrics_cols].tolist()
                values += values[:1]  
                ax.plot(angles, values, linewidth=2, label=model)
                ax.fill(angles, values, alpha=0.1)
            
           
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_cols)
            ax.set_ylim(0.4, 0.7)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            st.pyplot(fig)
    
   
    elif page == "About":
        st.title("About This Project")
        st.markdown("""
        ## IPL 2025 Match Prediction Application
        
        This application was developed to predict the outcomes of IPL 2025 matches using machine learning models.
        
        ### Models Used:
        - **Logistic Regression**: A baseline model for binary classification tasks
        - **Random Forest Classifier**: An ensemble method that builds multiple decision trees
        - **Support Vector Machine (SVM)**: Effective in high-dimensional spaces
        - **LightGBM**: A gradient boosting framework using tree-based learning algorithms
        
        ### Data Sources:
        - IPL 2025 Match Records
        - Ball-by-Ball Data
        - Historical Data (2008-2024)
        
        ### Implementation:
        - Data preprocessing and feature engineering
        - Model training and evaluation
        - Interactive Streamlit application
        """)


if __name__ == "__main__":
    main()