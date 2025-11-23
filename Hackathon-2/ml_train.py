"""
Crop Yield Prediction Model Training Script
Combines Satellite Imagery (NDVI) + Weather Data for ML Training
UN SDG 2: Zero Hunger
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import logging
import os
from typing import Tuple
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_training_data(csv_path: str) -> pd.DataFrame:
    """
    Load crop yield data from CSV
    Expected columns:
    - latitude, longitude, date, crop_type
    - temperature, precipitation, humidity, soil_moisture
    - ndvi, yield (target)
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples")
    return df

def generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate synthetic crop yield data for demonstration"""
    logger.info(f"Generating {n_samples} synthetic samples")
    
    np.random.seed(42)
    
    crop_types = ['maize', 'wheat', 'rice', 'soybean', 'potato']
    
    data = {
        'latitude': np.random.uniform(-60, 60, n_samples),
        'longitude': np.random.uniform(-180, 180, n_samples),
        'crop_type': np.random.choice(crop_types, n_samples),
        'temperature': np.random.uniform(10, 35, n_samples),
        'precipitation': np.random.uniform(0, 300, n_samples),
        'humidity': np.random.uniform(20, 95, n_samples),
        'soil_moisture': np.random.uniform(0.2, 0.8, n_samples),
        'ndvi': np.random.uniform(0.3, 0.9, n_samples),
        'elevation': np.random.uniform(0, 3000, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate yield based on features (with noise)
    df['yield'] = (
        1000
        + 2000 * df['ndvi']
        + 50 * df['temperature']
        + 5 * df['precipitation']
        + 40 * df['humidity']
        - 100 * (df['elevation'] / 100)
        + np.random.normal(0, 300, n_samples)
    )
    df['yield'] = df['yield'].clip(lower=500)  # Yield >= 500
    
    logger.info(f"Synthetic data generated with {len(df.columns)} features")
    logger.info(f"Target yield range: {df['yield'].min():.0f} - {df['yield'].max():.0f} kg/ha")
    
    return df

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Preprocess data: encode categorical, normalize numerical
    Returns: X, y, encoders dict
    """
    logger.info("Preprocessing data...")
    
    df = df.copy()
    
    # Handle missing values
    df = df.dropna()
    
    # Extract target
    y = df['yield'].values
    X = df.drop('yield', axis=1)
    
    # Categorical encoding
    encoders = {}
    categorical_cols = ['crop_type']
    
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
    
    # Drop non-numeric columns
    X = X.select_dtypes(include=[np.number]).values
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Feature stats - Mean: {X.mean(axis=0)[:5]}, Std: {X.std(axis=0)[:5]}")
    
    return X, y, encoders

def normalize_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize features using training data statistics"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# ============================================================================
# MACHINE LEARNING MODELS
# ============================================================================

class CropYieldModel:
    """Hybrid model combining Random Forest + Neural Network"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.rf_model = None
        self.nn_model = None
        self.scaler = StandardScaler()
        
    def build_neural_network(self) -> keras.Model:
        """
        Build neural network for yield prediction
        Architecture: Input → Dense(128) → Dropout → Dense(64) → Dense(32) → Output
        """
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            
            layers.Dense(16, activation='relu'),
            
            layers.Dense(1)  # Regression output
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        logger.info(f"Neural Network created with {model.count_params()} parameters")
        return model
    
    def build_random_forest(self) -> RandomForestRegressor:
        """Build Random Forest regressor"""
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, 
              epochs: int = 100):
        """Train both models"""
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train Neural Network
        logger.info("Training Neural Network...")
        self.nn_model = self.build_neural_network()
        
        history = self.nn_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-6
                )
            ],
            verbose=1
        )
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        self.rf_model = self.build_random_forest()
        self.rf_model.fit(X_train, y_train)
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction: weighted average of NN and RF"""
        X_scaled = self.scaler.transform(X)
        
        nn_pred = self.nn_model.predict(X_scaled, verbose=0).flatten()
        rf_pred = self.rf_model.predict(X)
        
        # Ensemble: 60% NN, 40% RF
        ensemble_pred = 0.6 * nn_pred + 0.4 * rf_pred
        
        return ensemble_pred
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
        }
        
        logger.info(f"Evaluation Metrics:")
        logger.info(f"  RMSE: {rmse:.2f} kg/ha")
        logger.info(f"  MAE:  {mae:.2f} kg/ha")
        logger.info(f"  R²:   {r2:.4f}")
        
        return metrics
    
    def save(self, model_dir: str = './models'):
        """Save models to disk"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save neural network
        self.nn_model.save(f'{model_dir}/nn_model.h5')
        
        # Save random forest and scaler
        with open(f'{model_dir}/rf_model.pkl', 'wb') as f:
            pickle.dump(self.rf_model, f)
        
        with open(f'{model_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Models saved to {model_dir}")
    
    def load(self, model_dir: str = './models'):
        """Load models from disk"""
        self.nn_model = keras.models.load_model(f'{model_dir}/nn_model.h5')
        
        with open(f'{model_dir}/rf_model.pkl', 'rb') as f:
            self.rf_model = pickle.load(f)
        
        with open(f'{model_dir}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"Models loaded from {model_dir}")

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    logger.info("=" * 70)
    logger.info("Crop Yield Prediction Model Training")
    logger.info("=" * 70)
    
    # 1. Load data
    logger.info("\n[1/6] Loading Data...")
    # df = load_training_data('crop_yield_data.csv')  # Use real data
    df = generate_synthetic_data(n_samples=5000)  # Or generate synthetic
    
    # 2. Preprocess
    logger.info("\n[2/6] Preprocessing Data...")
    X, y, encoders = preprocess_data(df)
    
    # 3. Train-test split
    logger.info("\n[3/6] Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Validation set size: {X_val.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    
    # 4. Build and train model
    logger.info("\n[4/6] Building & Training Model...")
    model = CropYieldModel(input_dim=X_train.shape[1])
    history = model.train(X_train, y_train, X_val, y_val, epochs=100)
    
    # 5. Evaluate
    logger.info("\n[5/6] Evaluating Model...")
    metrics = model.evaluate(X_test, y_test)
    
    # 6. Save
    logger.info("\n[6/6] Saving Model...")
    model.save('./models')
    
    # Visualization
    logger.info("\n[BONUS] Generating Visualizations...")
    plot_training_history(history)
    plot_predictions(model, X_test, y_test)
    
    logger.info("\n" + "=" * 70)
    logger.info("Training Complete! ✅")
    logger.info("=" * 70)
    
    return model, metrics

def plot_training_history(history):
    """Plot training and validation loss"""
    if history is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Model Loss During Training')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (kg/ha)')
    axes[1].set_title('Mean Absolute Error During Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    logger.info("Saved training history plot: training_history.png")

def plot_predictions(model, X_test: np.ndarray, y_test: np.ndarray):
    """Plot predicted vs actual yields"""
    predictions = model.predict(X_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(y_test, predictions, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Yield (kg/ha)')
    axes[0].set_ylabel('Predicted Yield (kg/ha)')
    axes[0].set_title('Predicted vs Actual Yield')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_test - predictions
    axes[1].hist(residuals, bins=50, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residual (kg/ha)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Residuals')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_analysis.png', dpi=150)
    logger.info("Saved predictions analysis: predictions_analysis.png")

if __name__ == '__main__':
    model, metrics = main()
