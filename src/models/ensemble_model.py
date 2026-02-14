"""Ensemble model combining XGBoost, LightGBM, and LSTM."""
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config


class EnsembleModel:
    """Ensemble of XGBoost, LightGBM, and LSTM for prediction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Models
        self.xgb_model = None
        self.lgb_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.accuracy_history = []
        self.last_accuracy = 0.0
        
    def load_or_initialize(self):
        """Load existing models or initialize new ones."""
        model_dir = Path(config.MODEL_PATH)
        
        try:
            # Try to load existing models
            if (model_dir / 'xgb_model.pkl').exists():
                with open(model_dir / 'xgb_model.pkl', 'rb') as f:
                    self.xgb_model = pickle.load(f)
                self.logger.info("Loaded existing XGBoost model")
            else:
                self.xgb_model = self._initialize_xgboost()
                self.logger.info("Initialized new XGBoost model")
            
            if (model_dir / 'lgb_model.pkl').exists():
                with open(model_dir / 'lgb_model.pkl', 'rb') as f:
                    self.lgb_model = pickle.load(f)
                self.logger.info("Loaded existing LightGBM model")
            else:
                self.lgb_model = self._initialize_lightgbm()
                self.logger.info("Initialized new LightGBM model")
            
            if (model_dir / 'lstm_model.h5').exists():
                self.lstm_model = keras.models.load_model(model_dir / 'lstm_model.h5')
                self.logger.info("Loaded existing LSTM model")
            else:
                self.lstm_model = self._initialize_lstm()
                self.logger.info("Initialized new LSTM model")
            
            if (model_dir / 'scaler.pkl').exists():
                with open(model_dir / 'scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
        
        except Exception as e:
            self.logger.error(f"Error loading models: {e}", exc_info=True)
            self.xgb_model = self._initialize_xgboost()
            self.lgb_model = self._initialize_lightgbm()
            self.lstm_model = self._initialize_lstm()
    
    def _initialize_xgboost(self) -> xgb.XGBClassifier:
        """Initialize XGBoost classifier."""
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    def _initialize_lightgbm(self) -> lgb.LGBMClassifier:
        """Initialize LightGBM classifier."""
        return lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    
    def _initialize_lstm(self, input_dim: int = 100) -> keras.Model:
        """Initialize LSTM network."""
        model = keras.Sequential([
            layers.Input(shape=(config.LSTM_SEQUENCE_LENGTH, input_dim)),
            layers.LSTM(config.LSTM_UNITS, return_sequences=True, dropout=config.LSTM_DROPOUT),
            layers.LSTM(config.LSTM_UNITS // 2, dropout=config.LSTM_DROPOUT),
            layers.Dense(64, activation='relu'),
            layers.Dropout(config.LSTM_DROPOUT),
            layers.Dense(32, activation='relu'),
            layers.Dense(3, activation='softmax')  # BUY, SELL, HOLD
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Generate ensemble prediction."""
        try:
            # Convert features to array
            feature_array = self._dict_to_array(features)
            feature_array_scaled = self.scaler.transform(feature_array.reshape(1, -1))
            
            # Get predictions from each model
            xgb_proba = self._predict_xgboost(feature_array_scaled)
            lgb_proba = self._predict_lightgbm(feature_array_scaled)
            lstm_proba = self._predict_lstm(feature_array_scaled)
            
            # Weighted ensemble
            ensemble_proba = (
                config.XGBOOST_WEIGHT * xgb_proba +
                config.LIGHTGBM_WEIGHT * lgb_proba +
                config.LSTM_WEIGHT * lstm_proba
            )
            
            # Get final prediction
            action_idx = np.argmax(ensemble_proba)
            actions = ['SELL', 'HOLD', 'BUY']
            action = actions[action_idx]
            confidence = ensemble_proba[action_idx]
            
            # Calculate expected return (simplified)
            if action == 'BUY':
                expected_return = confidence * 0.05  # Expect 5% return at 100% confidence
            elif action == 'SELL':
                expected_return = -confidence * 0.05
            else:
                expected_return = 0
            
            return {
                'action': action,
                'confidence': float(confidence),
                'expected_return': float(expected_return),
                'probabilities': {
                    'SELL': float(ensemble_proba[0]),
                    'HOLD': float(ensemble_proba[1]),
                    'BUY': float(ensemble_proba[2])
                },
                'model_agreement': self._calculate_agreement(xgb_proba, lgb_proba, lstm_proba)
            }
        
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}", exc_info=True)
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'expected_return': 0.0,
                'probabilities': {'SELL': 0.33, 'HOLD': 0.34, 'BUY': 0.33},
                'model_agreement': 0.0
            }
    
    def _predict_xgboost(self, features: np.ndarray) -> np.ndarray:
        """Get XGBoost prediction probabilities."""
        try:
            if self.xgb_model and hasattr(self.xgb_model, 'predict_proba'):
                return self.xgb_model.predict_proba(features)[0]
        except:
            pass
        return np.array([0.33, 0.34, 0.33])
    
    def _predict_lightgbm(self, features: np.ndarray) -> np.ndarray:
        """Get LightGBM prediction probabilities."""
        try:
            if self.lgb_model and hasattr(self.lgb_model, 'predict_proba'):
                return self.lgb_model.predict_proba(features)[0]
        except:
            pass
        return np.array([0.33, 0.34, 0.33])
    
    def _predict_lstm(self, features: np.ndarray) -> np.ndarray:
        """Get LSTM prediction probabilities."""
        try:
            if self.lstm_model:
                # Create sequence (replicate current features)
                sequence = np.tile(features, (config.LSTM_SEQUENCE_LENGTH, 1))
                sequence = sequence.reshape(1, config.LSTM_SEQUENCE_LENGTH, -1)
                return self.lstm_model.predict(sequence, verbose=0)[0]
        except:
            pass
        return np.array([0.33, 0.34, 0.33])
    
    async def retrain(self, training_data: pd.DataFrame) -> float:
        """Retrain all models with new data."""
        try:
            self.logger.info(f"Retraining models with {len(training_data)} samples...")
            
            # Prepare data
            X, y = self._prepare_training_data(training_data)
            
            if len(X) < 50:
                self.logger.warning("Insufficient data for meaningful retraining")
                return self.last_accuracy
            
            # Fit scaler
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train XGBoost
            self.xgb_model.fit(X_train, y_train)
            xgb_acc = accuracy_score(y_test, self.xgb_model.predict(X_test))
            
            # Train LightGBM
            self.lgb_model.fit(X_train, y_train)
            lgb_acc = accuracy_score(y_test, self.lgb_model.predict(X_test))
            
            # Train LSTM
            lstm_acc = self._train_lstm(X_train, y_train, X_test, y_test)
            
            # Calculate ensemble accuracy
            ensemble_acc = (xgb_acc + lgb_acc + lstm_acc) / 3
            
            self.last_accuracy = ensemble_acc
            self.accuracy_history.append({
                'timestamp': datetime.now(),
                'accuracy': ensemble_acc,
                'xgb_acc': xgb_acc,
                'lgb_acc': lgb_acc,
                'lstm_acc': lstm_acc,
                'samples': len(X_train)
            })
            
            self.logger.info(f"Retraining complete: {ensemble_acc:.2%} accuracy")
            self.logger.info(f"  XGBoost: {xgb_acc:.2%}, LightGBM: {lgb_acc:.2%}, LSTM: {lstm_acc:.2%}")
            
            return ensemble_acc
        
        except Exception as e:
            self.logger.error(f"Error during retraining: {e}", exc_info=True)
            return self.last_accuracy
    
    def _train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Train LSTM model."""
        try:
            # Reshape for LSTM (samples, timesteps, features)
            X_train_seq = np.array([np.tile(x, (config.LSTM_SEQUENCE_LENGTH, 1)) for x in X_train])
            X_test_seq = np.array([np.tile(x, (config.LSTM_SEQUENCE_LENGTH, 1)) for x in X_test])
            
            # Convert labels to categorical
            y_train_cat = keras.utils.to_categorical(y_train, num_classes=3)
            y_test_cat = keras.utils.to_categorical(y_test, num_classes=3)
            
            # Reinitialize LSTM with correct input dimension
            if X_train_seq.shape[-1] != self.lstm_model.input_shape[-1]:
                self.lstm_model = self._initialize_lstm(input_dim=X_train_seq.shape[-1])
            
            # Train
            self.lstm_model.fit(
                X_train_seq, y_train_cat,
                epochs=min(config.LSTM_EPOCHS, 20),  # Limit epochs for speed
                batch_size=config.LSTM_BATCH_SIZE,
                validation_split=0.2,
                verbose=0
            )
            
            # Evaluate
            y_pred = np.argmax(self.lstm_model.predict(X_test_seq, verbose=0), axis=1)
            return accuracy_score(y_test, y_pred)
        
        except Exception as e:
            self.logger.error(f"Error training LSTM: {e}")
            return 0.33
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from DataFrame."""
        # Extract features (all columns except 'label' and 'timestamp')
        feature_cols = [col for col in df.columns if col not in ['label', 'timestamp', 'outcome']]
        X = df[feature_cols].values
        
        # Extract labels
        if 'label' in df.columns:
            y = df['label'].values
        elif 'outcome' in df.columns:
            # Convert outcome to label (profit -> BUY/2, loss -> SELL/0, neutral -> HOLD/1)
            y = np.where(df['outcome'] > 0, 2, np.where(df['outcome'] < 0, 0, 1))
        else:
            # Generate random labels if none exist (for initial training)
            y = np.random.randint(0, 3, size=len(X))
        
        return X, y
    
    def _dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array."""
        # Remove metadata fields
        feature_dict = {k: v for k, v in features.items() if k != 'timestamp'}
        
        # Convert to sorted array
        sorted_keys = sorted(feature_dict.keys())
        return np.array([feature_dict[k] for k in sorted_keys])
    
    def _calculate_agreement(self, xgb_proba: np.ndarray, lgb_proba: np.ndarray,
                            lstm_proba: np.ndarray) -> float:
        """Calculate agreement between models."""
        # Get top prediction from each model
        xgb_pred = np.argmax(xgb_proba)
        lgb_pred = np.argmax(lgb_proba)
        lstm_pred = np.argmax(lstm_proba)
        
        # Count agreements
        agreements = sum([xgb_pred == lgb_pred, lgb_pred == lstm_pred, xgb_pred == lstm_pred])
        
        return agreements / 3.0
    
    def save(self):
        """Save all models to disk."""
        try:
            model_dir = Path(config.MODEL_PATH)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save XGBoost
            with open(model_dir / 'xgb_model.pkl', 'wb') as f:
                pickle.dump(self.xgb_model, f)
            
            # Save LightGBM
            with open(model_dir / 'lgb_model.pkl', 'wb') as f:
                pickle.dump(self.lgb_model, f)
            
            # Save LSTM
            self.lstm_model.save(model_dir / 'lstm_model.h5')
            
            # Save scaler
            with open(model_dir / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            self.logger.info("Models saved successfully")
        
        except Exception as e:
            self.logger.error(f"Error saving models: {e}", exc_info=True)
