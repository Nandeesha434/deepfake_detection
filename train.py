'''
Training pipeline for deepfake detection model
'''

import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import json
from datetime import datetime

from config import Config
from models import get_model


class DeepfakeTrainer:
    '''Handles model training and evaluation'''
    
    def __init__(self, config, model_name='efficientnet'):
        self.config = config
        self.model_name = model_name
        self.model = None
        self.history = None
    
    def prepare_data(self, X, y):
        '''Split and prepare data for training'''
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_SEED, stratify=y
        )
        
        # Second split: train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.config.VAL_SIZE, 
            random_state=self.config.RANDOM_SEED, stratify=y_temp
        )
        
        print(f"\nDataset Split:")
        print(f"Training samples: {len(X_train)} (Real: {np.sum(y_train == 0)}, Fake: {np.sum(y_train == 1)})")
        print(f"Validation samples: {len(X_val)} (Real: {np.sum(y_val == 0)}, Fake: {np.sum(y_val == 1)})")
        print(f"Test samples: {len(X_test)} (Real: {np.sum(y_test == 0)}, Fake: {np.sum(y_test == 1)})")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_data_augmentation(self):
        '''Create data augmentation pipeline'''
        return ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )
    
    def compile_model(self, model):
        '''Compile model with optimizer and loss'''
        optimizer = keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def get_callbacks(self, log_dir=None):
        '''Define training callbacks'''
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(self.config.LOGS_PATH, f"{self.model_name}_{timestamp}")
        
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(self.config.MODEL_SAVE_PATH, f'best_{self.model_name}.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, use_augmentation=True):
        '''Train the model'''
        # Build and compile model
        print(f"\nBuilding {self.model_name} model...")
        self.model = get_model(self.model_name, self.config.IMG_SIZE)
        self.model = self.compile_model(self.model)
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Data augmentation
        if use_augmentation:
            print("\nUsing data augmentation...")
            datagen = self.create_data_augmentation()
            train_generator = datagen.flow(X_train, y_train, batch_size=self.config.BATCH_SIZE)
        else:
            train_generator = None
        
        # Training
        print(f"\nStarting training for {self.config.EPOCHS} epochs...")
        if use_augmentation:
            self.history = self.model.fit(
                train_generator,
                epochs=self.config.EPOCHS,
                validation_data=(X_val, y_val),
                callbacks=self.get_callbacks(),
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=self.config.BATCH_SIZE,
                epochs=self.config.EPOCHS,
                validation_data=(X_val, y_val),
                callbacks=self.get_callbacks(),
                verbose=1
            )
        
        print("\nTraining completed!")
        return self.history
    
    def save_model(self, filename=None):
        '''Save trained model'''
        if filename is None:
            filename = f'{self.model_name}_final.h5'
        
        filepath = os.path.join(self.config.MODEL_SAVE_PATH, filename)
        self.model.save(filepath)
        print(f"\nModel saved to: {filepath}")
        
        # Save training history
        history_path = os.path.join(self.config.MODEL_SAVE_PATH, f'{self.model_name}_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy types to Python types
            history_dict = {}
            for key, value in self.history.history.items():
                history_dict[key] = [float(v) for v in value]
            json.dump(history_dict, f, indent=4)
        print(f"Training history saved to: {history_path}")
    
    def load_model(self, filepath):
        '''Load pre-trained model'''
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")
        return self.model


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
