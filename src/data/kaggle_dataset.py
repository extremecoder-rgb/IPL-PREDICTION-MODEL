import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

class IPLDatasetManager:
    def __init__(self, raw_data_path='data/raw', processed_data_path='data/processed'):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.label_encoders = {}
        
       
        os.makedirs(raw_data_path, exist_ok=True)
        os.makedirs(processed_data_path, exist_ok=True)
    
    def download_dataset(self):
        """Download IPL dataset from Kaggle"""
        kaggle_dir = os.path.expanduser('~/.kaggle')
        kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
        project_kaggle_json = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'kaggle.json')

        if not os.path.exists(kaggle_json) and not os.path.exists(project_kaggle_json):
            print("\nKaggle API credentials not found. Please follow these steps:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Scroll to 'API' section and click 'Create New API Token'")
            print("3. Download the kaggle.json file")
            print(f"4. Create the directory: {kaggle_dir}")
            print(f"5. Move the downloaded kaggle.json to: {kaggle_json}")
            print("\nAfter setting up the credentials, run this script again.")
            return False

        try:
            import kaggle
            
            os.makedirs(self.raw_data_path, exist_ok=True)
            kaggle.api.dataset_download_files(
                'patrickb1912/ipl-complete-dataset-20082020',
                path=self.raw_data_path,
                unzip=True
            )
            print("Dataset downloaded successfully")
            return True
        except Exception as e:
            print(f"\nError downloading dataset: {e}")
            print("If you're seeing authentication errors, please ensure your kaggle.json is valid.")
            return False
    
    def load_and_preprocess_data(self):
        """Load and preprocess the IPL dataset"""
       
        matches_df = pd.read_csv(os.path.join(self.raw_data_path, 'matches.csv'))
        
       
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        
       
        categorical_columns = ['team1', 'team2', 'toss_winner', 'venue', 'winner']
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            matches_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(matches_df[col])
        
       
        team_stats = self._calculate_team_stats(matches_df)
        
        
        final_df = self._prepare_final_dataset(matches_df, team_stats)
        
       
        final_df.to_csv(os.path.join(self.processed_data_path, 'processed_matches.csv'), index=False)
         # Save team and venue mappings
        mappings = {
            'teams': dict(enumerate(self.label_encoders['team1'].classes_)),
            'venues': dict(enumerate(self.label_encoders['venue'].classes_))
        }
        joblib.dump(mappings, os.path.join(self.processed_data_path, 'team_mappings.pkl'))
        
       
        venue_mappings = {'venues': dict(enumerate(self.label_encoders['venue'].classes_))}
        joblib.dump(venue_mappings, os.path.join(self.processed_data_path, 'venue_mappings.pkl'))
        
        return final_df
    
    def _calculate_team_stats(self, df):
       
        team_stats = {}
        
        for team in df['team1'].unique():
           
            team_matches = df[(df['team1'] == team) | (df['team2'] == team)]
            wins = len(team_matches[team_matches['winner'] == team])
            total_matches = len(team_matches)
            win_rate = wins / total_matches if total_matches > 0 else 0
            
            
            toss_wins = len(team_matches[team_matches['toss_winner'] == team])
            toss_win_rate = toss_wins / total_matches if total_matches > 0 else 0
            
            team_stats[team] = {
                'win_rate': win_rate,
                'toss_win_rate': toss_win_rate,
                'total_matches': total_matches
            }
        
        return team_stats
    
    def _prepare_final_dataset(self, matches_df, team_stats):
        
        df = matches_df.copy()
        
      
        df['team1_win_rate'] = df['team1'].map(lambda x: team_stats[x]['win_rate'])
        df['team2_win_rate'] = df['team2'].map(lambda x: team_stats[x]['win_rate'])
        df['team1_toss_win_rate'] = df['team1'].map(lambda x: team_stats[x]['toss_win_rate'])
        df['team2_toss_win_rate'] = df['team2'].map(lambda x: team_stats[x]['toss_win_rate'])
        
       
        df['target'] = (df['winner'] == df['team1']).astype(int)
        
       
        feature_columns = [
            'team1_encoded', 'team2_encoded', 'venue_encoded',
            'toss_winner_encoded', 'team1_win_rate', 'team2_win_rate',
            'team1_toss_win_rate', 'team2_toss_win_rate',
            'target'
        ]
        
        return df[feature_columns]
    
    def get_team_mapping(self):
       
        return {
            'teams': dict(enumerate(self.label_encoders['team1'].classes_)),
            'venues': dict(enumerate(self.label_encoders['venue'].classes_))
        }