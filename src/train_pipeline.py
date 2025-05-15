import os
from data.kaggle_dataset import IPLDatasetManager
from models.model_trainer import ModelTrainer

def main():
    data_manager = IPLDatasetManager()
    print("Downloading IPL dataset...")
    if not data_manager.download_dataset():
        print("\nExiting: Dataset download failed. Please set up Kaggle credentials and try again.")
        return
    
    print("Processing dataset...")
    processed_data = data_manager.load_and_preprocess_data()
    
   
    mappings = data_manager.get_team_mapping()
    
   
    trainer = ModelTrainer()
    
  
    print("Training models...")
    X = processed_data.drop('target', axis=1)
    y = processed_data['target']
    trained_models = trainer.train_models(X, y, tune_hyperparams=True)
    
  
    print("Saving models...")
    trainer.save_models(trained_models)
    
    print("Training pipeline completed successfully!")
    print("\nModel Performance Metrics:")
    for model_name, metrics in trainer.metrics.items():
        print(f"\n{model_name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

if __name__ == '__main__':
    main()
