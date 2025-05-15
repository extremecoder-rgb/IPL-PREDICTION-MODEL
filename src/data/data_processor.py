import pandas as pd
import numpy as np
import os
from datetime import datetime
import re

class DataProcessor:
    def __init__(self, raw_data_path='data/raw', processed_data_path='data/processed'):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        
       
        os.makedirs(raw_data_path, exist_ok=True)
        os.makedirs(processed_data_path, exist_ok=True)
    
    def load_data(self, file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return None
    
    def clean_match_data(self, match_data):
        if match_data is None or match_data.empty:
            return None
        
       
        df = match_data.copy()
        df = df.drop_duplicates()
        
        
       
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        
       
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
       
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        
        if 'team1' in df.columns and 'team2' in df.columns:
            team_mapping = {
                'MI': 'Mumbai Indians',
                'CSK': 'Chennai Super Kings',
                'RCB': 'Royal Challengers Bangalore',
                'KKR': 'Kolkata Knight Riders',
                'RR': 'Rajasthan Royals',
                'DC': 'Delhi Capitals',
                'SRH': 'Sunrisers Hyderabad',
                'PBKS': 'Punjab Kings',
                'GT': 'Gujarat Titans',
                'LSG': 'Lucknow Super Giants'
            }
            
            
            for col in ['team1', 'team2', 'winner']:
                if col in df.columns:
                    df[col] = df[col].replace(team_mapping)
        
        return df
    
    def engineer_features(self, match_data, ball_by_ball_data=None, historical_data=None):
        if match_data is None or match_data.empty:
            return None
        
        
        df = match_data.copy()
        
      
        if 'venue' in df.columns and 'winner' in df.columns:
            venue_stats = df.groupby('venue').apply(
                lambda x: pd.Series({
                    'matches': len(x),
                    'home_wins': sum(x['winner'] == x['team1'])
                })
            )
            venue_stats['home_win_rate'] = venue_stats['home_wins'] / venue_stats['matches']
            df = df.merge(venue_stats[['home_win_rate']], left_on='venue', right_index=True, how='left')
        
       
        if 'team1' in df.columns and 'team2' in df.columns and 'date' in df.columns and 'winner' in df.columns:
           
            df = df.sort_values('date')
            
           
            teams = pd.unique(df[['team1', 'team2']].values.ravel('K'))
            
           
            team_form = {team: [] for team in teams}
            team_last_5 = {team: 0 for team in teams}
            
           
            for _, row in df.iterrows():
                team1, team2 = row['team1'], row['team2']
                winner = row['winner']
                
               
                team_form[team1].append(1 if winner == team1 else 0)
                team_last_5[team1] = sum(team_form[team1][-5:]) / min(5, len(team_form[team1]))
                
                
                team_form[team2].append(1 if winner == team2 else 0)
                team_last_5[team2] = sum(team_form[team2][-5:]) / min(5, len(team_form[team2]))
                
               
                df.loc[df.index == row.name, 'team1_form'] = team_last_5[team1]
                df.loc[df.index == row.name, 'team2_form'] = team_last_5[team2]
        
        
        if 'team1' in df.columns and 'team2' in df.columns and 'winner' in df.columns:
           
            h2h_records = {}
            
           
            for _, row in df.iterrows():
                team1, team2 = row['team1'], row['team2']
                winner = row['winner']
                
                
                teams_key = tuple(sorted([team1, team2]))
                
                if teams_key not in h2h_records:
                    h2h_records[teams_key] = {'matches': 0, 'wins': {team1: 0, team2: 0}}
                
                h2h_records[teams_key]['matches'] += 1
                if winner in [team1, team2]: 
                    h2h_records[teams_key]['wins'][winner] += 1
                
               
                matches = h2h_records[teams_key]['matches']
                team1_wins = h2h_records[teams_key]['wins'][team1]
                team2_wins = h2h_records[teams_key]['wins'][team2]
                
               
                df.loc[df.index == row.name, 'team1_h2h_win_rate'] = team1_wins / matches if matches > 0 else 0.5
                df.loc[df.index == row.name, 'team2_h2h_win_rate'] = team2_wins / matches if matches > 0 else 0.5
        
      
        if 'toss_winner' in df.columns and 'toss_decision' in df.columns:
           
            df['toss_winner_encoded'] = df.apply(
                lambda row: 1 if row['toss_winner'] == row['team1'] else 0, axis=1
            )
            
           
            df['toss_decision_encoded'] = df['toss_decision'].apply(
                lambda x: 1 if x.lower() == 'bat' else 0
            )
        
       
        if 'winner' in df.columns:
            df['team1_win'] = df.apply(
                lambda row: 1 if row['winner'] == row['team1'] else 0, axis=1
            )
        
        return df
    
    def preprocess_for_modeling(self, data):
        if data is None or data.empty:
            return None, None
        
       
        df = data.copy()
        
      
        features = [
            'toss_winner_encoded', 'toss_decision_encoded',
            'team1_form', 'team2_form',
            'team1_h2h_win_rate', 'team2_h2h_win_rate',
            'home_win_rate'
        ]
        
        
        features = [f for f in features if f in df.columns]
        
       
        categorical_cols = ['venue', 'team1', 'team2']
        for col in categorical_cols:
            if col in df.columns:
              
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
               
                features.extend(dummies.columns)
                
                df = pd.concat([df, dummies], axis=1)
        
       
        X = df[features]
        y = df['team1_win'] if 'team1_win' in df.columns else None
        
        return X, y
    
    def process_and_save(self, match_file, ball_by_ball_file=None, historical_file=None):
        
        match_data = self.load_data(os.path.join(self.raw_data_path, match_file))
        
        ball_by_ball_data = None
        if ball_by_ball_file:
            ball_by_ball_data = self.load_data(os.path.join(self.raw_data_path, ball_by_ball_file))
        
        historical_data = None
        if historical_file:
            historical_data = self.load_data(os.path.join(self.raw_data_path, historical_file))
        
        
        cleaned_match_data = self.clean_match_data(match_data)
        
       
        featured_data = self.engineer_features(cleaned_match_data, ball_by_ball_data, historical_data)
        
       
        X, y = self.preprocess_for_modeling(featured_data)
        
        
        if featured_data is not None:
            featured_data.to_csv(os.path.join(self.processed_data_path, 'processed_match_data.csv'), index=False)
        
        if X is not None and y is not None:
            X.to_csv(os.path.join(self.processed_data_path, 'X_features.csv'), index=False)
            y.to_csv(os.path.join(self.processed_data_path, 'y_target.csv'), index=False)
        
        return X, y


if __name__ == "__main__":
    processor = DataProcessor()
   