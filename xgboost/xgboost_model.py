import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class XGBoostModel:
    def __init__(self, num_class=3, eval_metric=['mlogloss'], early_stopping_rounds=10, savepath = 'xgboost/weight'):
        """
        Initialize the XGBoost model with default parameters.
        
        Args:
            num_class (int): Number of classes for classification.
            eval_metric (list): Evaluation metrics for training.
            early_stopping_rounds (int): Number of rounds for early stopping.
        """
        self.model = None
        self.num_class = num_class
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.params = {
            'objective': 'multi:softmax',
            'num_class': self.num_class,
            'max_depth': 7,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'lambda_': 1.5,
            'alpha': 1,
            'eval_metric': self.eval_metric,
            'early_stopping_rounds': self.early_stopping_rounds
        }
        self.savepath = savepath

    def compute_class_weights(self, y_train):
        """
        Calculate class weights based on class frequency.

        Args:
            y_train: Training labels.

        Returns:
            np.array: Sample weights for training.
        """
        class_counts = pd.Series(y_train).value_counts()
        total_samples = len(y_train)
        weights = {}
        for cls in class_counts.index:
            weights[cls] = total_samples / class_counts[cls]  # Inverse frequency
        # Limit maximum weight
        max_weight = 3
        for cls in weights:
            weights[cls] = min(weights[cls], max_weight)
        # Lower weight for hold class (2)
        weights[2] = 1.0 if 2 in weights else min(weights.values())
        sample_weights = np.array([weights[cls] for cls in y_train])
        return sample_weights

    def split_train_val(self, df, features, target, train_ratio=0.8):
        """
        Split data into training and validation sets.

        Args:
            df: DataFrame containing features and target.
            features: List of feature columns.
            target: Target column name.
            train_ratio: Ratio of training data.

        Returns:
            tuple: X_train, y_train, X_val, y_val
        """
        train_size = int(train_ratio * len(df))
        X_train = df[features][:train_size]
        y_train = df[target][:train_size]
        X_val = df[features][train_size:]
        y_val = df[target][train_size:]
        return X_train, y_train, X_val, y_val

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the XGBoost model with sample weights and validation set.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            xgb.XGBClassifier: Trained model.
        """
        sample_weights = self.compute_class_weights(y_train)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            verbose=True
        )
        model_path = os.path.join(self.savepath, 'xgboost_model.model')
        self.model.save_model(model_path)
        return self.model

    def evaluate(self, X_test, y_test, save_dir='trading_signals'):
        """
        Evaluate the model on test data and plot confusion matrix.

        Args:
            X_test: Test features.
            y_test: Test labels.
            save_dir: Directory to save confusion matrix plot.

        Returns:
            tuple: y_pred, y_confidence
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        y_prob = self.model.predict_proba(X_test)
        y_pred = np.argmax(y_prob, axis=1)
        y_confidence = np.max(y_prob, axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy trên tập test: {accuracy:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        self.plot_confusion_matrix(y_test, y_pred, save_dir)
        
        return y_pred, y_confidence

    def plot_confusion_matrix(self, y_true, y_pred, save_dir='trading_signals'):
        """
        Plot and save the confusion matrix.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            save_dir: Directory to save the plot.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Sell', 'Buy', 'Hold'], yticklabels=['Sell', 'Buy', 'Hold'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        filepath = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Đã lưu confusion matrix: {filepath}")
        plt.show()
        plt.close()