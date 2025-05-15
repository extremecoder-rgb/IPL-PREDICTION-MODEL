import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
   
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'LightGBM': LGBMClassifier(random_state=42)
        }
        
        self.param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['liblinear', 'saga']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 100]
            }
        }
        
       
        self.metrics = {}
    
    def load_data(self, X_path, y_path):
        try:
            X = pd.read_csv(X_path)
            y = pd.read_csv(y_path).iloc[:, 0]  
            return X, y
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def train_models(self, X, y, test_size=0.2, tune_hyperparams=False):
        if X is None or y is None:
            print("No data provided for training.")
            return None
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        trained_models = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if tune_hyperparams:
                
                grid_search = GridSearchCV(
                    model, self.param_grids[name], cv=5, scoring='accuracy', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
               
                best_model = grid_search.best_estimator_
                trained_models[name] = best_model
                
                print(f"Best parameters for {name}: {grid_search.best_params_}")
            else:
               
                model.fit(X_train, y_train)
                trained_models[name] = model
            
           
            y_pred = trained_models[name].predict(X_test)
            
          
            self.metrics[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            print(f"{name} - Accuracy: {self.metrics[name]['accuracy']:.4f}")
        
      
        self.X_test = X_test
        self.y_test = y_test
        
        return trained_models
    
    def save_models(self, models):
        if models is None:
            print("No models to save.")
            return
        
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        for name, model in models.items():
            model_filename = os.path.join(self.models_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
            try:
                joblib.dump(model, model_filename)
                print(f"Saved {name} model to {model_filename}")
            except Exception as e:
                print(f"Error saving {name} model: {e}")
    
    def load_saved_models(self):
        loaded_models = {}
        
        for name in self.models.keys():
            model_filename = os.path.join(self.models_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
            
            try:
                loaded_models[name] = joblib.load(model_filename)
                print(f"Loaded {name} model from {model_filename}")
            except Exception as e:
                print(f"Error loading {name} model: {e}")
        
        return loaded_models
    
    def evaluate_models(self, models=None, X=None, y=None):
        if models is None:
            models = self.load_saved_models()
        
        if X is None and hasattr(self, 'X_test'):
            X = self.X_test
        
        if y is None and hasattr(self, 'y_test'):
            y = self.y_test
        
        if models is None or X is None or y is None:
            print("Missing models or evaluation data.")
            return None
        
        metrics_data = []
        
        for name, model in models.items():
            y_pred = model.predict(X)
            
            metrics_dict = {
                'Model': name,
                'Accuracy': accuracy_score(y, y_pred),
                'Precision': precision_score(y, y_pred),
                'Recall': recall_score(y, y_pred),
                'F1 Score': f1_score(y, y_pred)
            }
            
            metrics_data.append(metrics_dict)
            
           
            cm = confusion_matrix(y, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Team 2 Win', 'Team 1 Win'],
                        yticklabels=['Team 2 Win', 'Team 1 Win'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {name}')
            plt.savefig(os.path.join(self.models_dir, f"{name.lower().replace(' ', '_')}_confusion_matrix.png"))
            plt.close()
        
        
        metrics_df = pd.DataFrame(metrics_data)
        
       
        metrics_df.to_csv(os.path.join(self.models_dir, 'model_metrics.csv'), index=False)
        
        return metrics_df
    
    def plot_feature_importance(self, models=None, feature_names=None):
        if models is None:
            models = self.load_saved_models()
        
        if models is None:
            print("No models available.")
            return
        
       
        for name, model in models.items():
            if name in ['Random Forest', 'LightGBM']:
                try:
                   
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        
                       
                        if feature_names is None:
                            feature_names = [f'Feature {i}' for i in range(len(importances))]
                        
                       
                        indices = np.argsort(importances)[::-1]
                        
                       
                        plt.figure(figsize=(10, 6))
                        plt.title(f'Feature Importance - {name}')
                        plt.bar(range(len(indices)), importances[indices], align='center')
                        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.models_dir, f"{name.lower().replace(' ', '_')}_feature_importance.png"))
                        plt.close()
                except Exception as e:
                    print(f"Error plotting feature importance for {name}: {e}")


if __name__ == "__main__":
    trainer = ModelTrainer()
   
