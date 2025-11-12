'''
Configuration file for Deepfake Detection System
Authors: Nischay Upadhya P, Supreeth Gutti, Kaushik Raju S, Nandeesha B
'''

import os

class Config:
    '''Configuration parameters for the deepfake detection system'''
    
    # Project paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, 'data')
    REAL_PATH = os.path.join(DATASET_PATH, 'real')
    FAKE_PATH = os.path.join(DATASET_PATH, 'fake')
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models')
    RESULTS_PATH = os.path.join(BASE_DIR, 'results')
    LOGS_PATH = os.path.join(BASE_DIR, 'logs')
    
    # Image parameters
    IMG_SIZE = 380
    CHANNELS = 3
    FRAMES_PER_VIDEO = 10  # Extract 10 frames from each video
    
    # Model parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # Train/Val/Test split
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Face detection
    FACE_DETECTION_CONFIDENCE = 0.9
    FACE_PADDING = 20
    
    @staticmethod
    def create_directories():
        '''Create necessary directories if they don't exist'''
        directories = [
            Config.DATASET_PATH,
            Config.REAL_PATH,
            Config.FAKE_PATH,
            Config.MODEL_SAVE_PATH,
            Config.RESULTS_PATH,
            Config.LOGS_PATH
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("All directories created successfully!")

if __name__ == "__main__":
    Config.create_directories()
    print("Configuration loaded successfully!")
