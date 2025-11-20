"""
Anomaly detection using One-Class SVM model trained on normal embeddings
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import pickle
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Anomaly detector using One-Class SVM trained on normal video embeddings.
    Follows the academic approach: train on normal data, predict on test data.
    """
    
    def __init__(self, model_dir: str = "assets/models", baseline_dir: str = "assets/baselines"):
        """
        Initialize anomaly detector.
        
        Args:
            model_dir: Directory to store trained models
            baseline_dir: Directory to store baseline embeddings (for reference)
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.is_trained = False
    
    def train_one_class_model(
        self,
        normal_embeddings: List[np.ndarray],
        model_name: str = "default",
        nu: float = 0.1,
        gamma: str = "scale"
    ) -> str:
        """
        Train a One-Class SVM model on normal video embeddings.
        This is the core of Phase 1: train on MANY normal videos.
        
        Args:
            normal_embeddings: List of embedding arrays from normal videos
            model_name: Name for the trained model
            nu: An upper bound on the fraction of training errors (0.1 = 10% outliers expected)
            gamma: Kernel coefficient ('scale', 'auto', or float)
            
        Returns:
            Path to saved model file
        """
        # Convert to numpy array
        X = np.array(normal_embeddings)
        
        if len(X) == 0:
            raise ValueError("No embeddings provided for training")
        
        print(f"Training One-Class SVM on {len(X)} normal embeddings (dimension: {X.shape[1]})")
        
        # CRITICAL: Standardize features using StandardScaler
        # This ensures all features are on the same scale (mean=0, std=1)
        # The scaler MUST be saved and reused for test data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Scaled embeddings shape: {X_scaled.shape}")
        print(f"Scaled embeddings - Mean: {X_scaled.mean(axis=0)[:5]} (first 5), Std: {X_scaled.std(axis=0)[:5]} (first 5)")
        
        # Train One-Class SVM on scaled data
        self.model = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)
        self.model.fit(X_scaled)
        
        self.is_trained = True
        
        # Save model and scaler (BOTH are required for prediction)
        model_file = self.model_dir / f"{model_name}_model.pkl"
        scaler_file = self.model_dir / f"{model_name}_scaler.pkl"
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print(f"Model trained and saved: {model_file}")
            print(f"Scaler saved: {scaler_file}")
            print(f"Scaler fitted on {len(X)} samples with {X.shape[1]} features")
            
            # Verify scaler was saved correctly
            if self.scaler is None:
                raise ValueError("Scaler is None after saving - this should not happen!")
            
            return str(model_file)
        except Exception as e:
            print(f"Error saving model/scaler: {e}")
            raise
    
    def load_model(self, model_name: str = "default") -> bool:
        """
        Load a trained One-Class SVM model and its scaler.
        CRITICAL: Both model and scaler must be loaded for proper scaling.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if model and scaler loaded successfully, False otherwise
        """
        model_file = self.model_dir / f"{model_name}_model.pkl"
        scaler_file = self.model_dir / f"{model_name}_scaler.pkl"
        
        if not model_file.exists():
            print(f"Model file not found: {model_file}")
            return False
        
        if not scaler_file.exists():
            print(f"Scaler file not found: {scaler_file}. Model cannot be used without scaler.")
            return False
        
        try:
            # Load the model
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load the scaler (CRITICAL for proper feature scaling)
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Verify both are loaded
            if self.model is None:
                print("Error: Model object is None after loading")
                return False
            
            if self.scaler is None:
                print("Error: Scaler object is None after loading")
                return False
            
            self.is_trained = True
            print(f"Model and scaler loaded successfully: {model_name}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.scaler = None
            self.is_trained = False
            return False
    
    def save_baseline(
        self, 
        embeddings: List[np.ndarray], 
        baseline_name: str
    ) -> str:
        """
        Save normal baseline embeddings to file (for reference/visualization).
        
        Args:
            embeddings: List of embedding arrays
            baseline_name: Name for the baseline
            
        Returns:
            Path to saved baseline file
        """
        embeddings_array = np.array(embeddings)
        baseline_file = self.baseline_dir / f"{baseline_name}.npy"
        np.save(baseline_file, embeddings_array)
        return str(baseline_file)
    
    def load_baseline(self, baseline_name: str) -> Optional[np.ndarray]:
        """
        Load baseline embeddings from file (for reference).
        
        Args:
            baseline_name: Name of the baseline
            
        Returns:
            Numpy array of embeddings or None if not found
        """
        baseline_file = self.baseline_dir / f"{baseline_name}.npy"
        if baseline_file.exists():
            return np.load(baseline_file)
        return None
    
    def predict(self, test_embedding: np.ndarray) -> Tuple[bool, float]:
        """
        Predict if a test embedding is normal or anomalous using trained model.
        This is Phase 2: test new videos against trained model.
        
        Args:
            test_embedding: Single test embedding (1D array)
            
        Returns:
            Tuple of (is_anomaly, decision_score)
            - is_anomaly: True if anomalous, False if normal
            - decision_score: Distance from decision boundary (negative = anomaly, positive = normal)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_one_class_model() first.")
        
        if self.model is None:
            raise ValueError("Model object is None. Please reload the model.")
        
        if self.scaler is None:
            raise ValueError("Scaler object is None. The model was not properly saved/loaded. Please retrain.")
        
        # Reshape and scale using the SAME scaler that was used during training
        X = test_embedding.reshape(1, -1)
        
        # Verify embedding dimension matches scaler's expected dimension
        expected_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else len(self.scaler.mean_)
        actual_features = X.shape[1]
        
        if actual_features != expected_features:
            raise ValueError(
                f"Embedding dimension mismatch: Test embedding has {actual_features} features, "
                f"but scaler expects {expected_features} features. "
                f"Please ensure test embeddings are generated using the same VLM model."
            )
        
        # CRITICAL: Apply the same scaler transformation used during training
        # This ensures test embeddings are on the same scale as training embeddings
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            raise ValueError(f"Error scaling test embedding: {e}. Scaler may not be properly fitted.")
        
        # Predict: +1 = normal, -1 = anomaly
        prediction = self.model.predict(X_scaled)[0]
        is_anomaly = (prediction == -1)
        
        # Get decision function score (distance from boundary)
        decision_score = self.model.decision_function(X_scaled)[0]
        
        return (is_anomaly, float(decision_score))
    
    def detect_anomalies(
        self, 
        test_embeddings: List[np.ndarray]
    ) -> List[Tuple[bool, float]]:
        """
        Detect anomalies in test embeddings using trained One-Class model.
        CRITICAL: Test embeddings are scaled using the same scaler from training.
        
        Args:
            test_embeddings: List of test embeddings
            
        Returns:
            List of tuples (is_anomaly, decision_score)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_one_class_model() first.")
        
        if self.model is None:
            raise ValueError("Model object is None. Please reload the model.")
        
        if self.scaler is None:
            raise ValueError("Scaler object is None. The model was not properly saved/loaded. Please retrain.")
        
        # Verify scaler is fitted (has mean_ and scale_ attributes)
        if not hasattr(self.scaler, 'mean_') or not hasattr(self.scaler, 'scale_'):
            raise ValueError("Scaler is not properly fitted. Please retrain the model.")
        
        results = []
        for test_emb in test_embeddings:
            try:
                is_anomaly, score = self.predict(test_emb)
                results.append((is_anomaly, score))
            except Exception as e:
                # If prediction fails, log error but continue
                print(f"Error predicting on embedding: {e}")
                # Mark as anomaly if we can't predict
                results.append((True, -1.0))
        
        return results
    
    def compute_similarity(
        self, 
        test_embedding: np.ndarray, 
        baseline_embeddings: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute similarity score for compatibility with existing code.
        Uses decision function score from One-Class SVM.
        
        Args:
            test_embedding: Single test embedding (1D array)
            baseline_embeddings: Optional (kept for compatibility)
            
        Returns:
            Decision score (higher = more normal, lower = more anomalous)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_one_class_model() first.")
        
        _, score = self.predict(test_embedding)
        # Normalize score to 0-1 range for similarity-like interpretation
        # Higher score = more normal (similar to similarity)
        normalized_score = 1.0 / (1.0 + np.exp(-score))  # Sigmoid normalization
        return float(normalized_score)

