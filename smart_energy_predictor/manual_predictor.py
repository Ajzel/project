# Manual Energy Predictor Flask Application

from flask import Flask, render_template, jsonify, request
import json
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

app = Flask(__name__)

class ManualEnergyPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['temperature', 'humidity', 'light', 'motion']
        self.model_metrics = {}
    
    def train_with_user_data(self, data_file=None, data_dict=None):
        """Train model with user-provided data"""
        try:
            # Load data
            if data_file and os.path.exists(data_file):
                if data_file.endswith('.csv'):
                    data = pd.read_csv("Energy_consumption.csv")
                elif data_file.endswith(('.xlsx', '.xls')):
                    data = pd.read_excel(data_file)
                else:
                    raise ValueError("Unsupported file format")
            elif data_dict:
                data = pd.DataFrame(data_dict)
            else:
                # Generate sample data if no user data provided
                data = self._generate_sample_data()
            
            print(f"Training with {len(data)} records")
            print(f"Columns: {list(data.columns)}")
            
            # Prepare features and target
            feature_cols = [col for col in self.feature_names if col in data.columns]
            if not feature_cols:
                raise ValueError(f"No matching feature columns found. Expected: {self.feature_names}")
            
            X = data[feature_cols]
            
            # Target column (energy consumption)
            target_cols = ['energy_consumption', 'energy', 'power', 'consumption', 'kwh']
            target_col = None
            for col in target_cols:
                if col in data.columns:
                    target_col = col
                    break
            
            if target_col is None:
                raise ValueError(f"No target column found. Expected one of: {target_cols}")
            
            y = data[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
            
            self.model_metrics = {
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'feature_importance': dict(zip(feature_cols, self.model.feature_importances_)),
                'training_samples': len(data),
                'features_used': feature_cols,
                'target_column': target_col
            }
            
            self.feature_names = feature_cols
            self.is_trained = True
            
            return self.model_metrics
            
        except Exception as e:
            print(f"Training error: {e}")
            return {'error': str(e)}
    
    def predict_single(self, **kwargs):
        """Make a single prediction with given parameters"""
        if not self.is_trained:
            return {'error': 'Model not trained yet'}
        
        try:
            # Prepare input data
            input_values = []
            missing_features = []
            
            for feature in self.feature_names:
                if feature in kwargs:
                    input_values.append(float(kwargs[feature]))
                else:
                    missing_features.append(feature)
            
            if missing_features:
                return {'error': f'Missing required features: {missing_features}'}
            
            # Scale and predict
            input_array = np.array([input_values])
            input_scaled = self.scaler.transform(input_array)
            prediction = self.model.predict(input_scaled)[0]
            
            # Get prediction confidence (using feature importance)
            confidence = self._calculate_confidence(input_values)
            
            return {
                'prediction': max(0, prediction),  # Ensure non-negative
                'confidence': confidence,
                'input_features': dict(zip(self.feature_names, input_values))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def predict_batch(self, data_list):
        """Make predictions for multiple data points"""
        if not self.is_trained:
            return {'error': 'Model not trained yet'}
        
        try:
            predictions = []
            for data_point in data_list:
                result = self.predict_single(**data_point)
                predictions.append(result)
            
            return {'predictions': predictions}
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_confidence(self, input_values):
        """Calculate prediction confidence based on input ranges"""
        # Simple confidence calculation based on how typical the input values are
        confidence_scores = []
        
        # Define typical ranges for each feature
        typical_ranges = {
            'temperature': (15, 35),
            'humidity': (30, 80),
            'light': (0, 100),
            'motion': (0, 1)
        }
        
        for i, feature in enumerate(self.feature_names):
            value = input_values[i]
            min_val, max_val = typical_ranges.get(feature, (0, 100))
            
            if min_val <= value <= max_val:
                confidence_scores.append(1.0)
            else:
                # Reduce confidence for values outside typical range
                distance = min(abs(value - min_val), abs(value - max_val))
                confidence_scores.append(max(0.1, 1.0 - (distance / max_val)))
        
        return round(np.mean(confidence_scores) * 100, 1)
    
    def _generate_sample_data(self, num_samples=1000):
        """Generate sample training data"""
        np.random.seed(42)
        
        temperature = np.random.normal(25, 5, num_samples)
        humidity = np.random.normal(60, 15, num_samples)
        light = np.random.uniform(0, 100, num_samples)
        motion = np.random.binomial(1, 0.3, num_samples)
        
        # Calculate energy consumption
        energy = (
            temperature * 2.5 +
            humidity * 0.8 +
            (100 - light) * 1.2 +
            motion * 15 +
            np.random.normal(0, 5, num_samples)
        )
        energy = np.maximum(energy, 10)
        
        return pd.DataFrame({
            'temperature': temperature,
            'humidity': humidity,
            'light': light,
            'motion': motion,
            'energy_consumption': energy
        })
    
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_trained:
            return {'error': 'Model not trained yet'}
        
        return {
            'is_trained': self.is_trained,
            'features': self.feature_names,
            'metrics': self.model_metrics
        }
    
    def save_model(self, filepath='manual_energy_model.pkl'):
        """Save trained model"""
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'metrics': self.model_metrics
            }, filepath)
            return True
        return False
    
    def load_model(self, filepath='manual_energy_model.pkl'):
        """Load trained model"""
        try:
            saved_objects = joblib.load(filepath)
            self.model = saved_objects['model']
            self.scaler = saved_objects['scaler']
            self.feature_names = saved_objects.get('feature_names', self.feature_names)
            self.model_metrics = saved_objects.get('metrics', {})
            self.is_trained = True
            return True
        except:
            return False

# Initialize predictor
predictor = ManualEnergyPredictor()

# Database setup for storing predictions
def init_db():
    conn = sqlite3.connect('manual_predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            temperature REAL,
            humidity REAL,
            light REAL,
            motion INTEGER,
            prediction REAL,
            confidence REAL,
            user_notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(data):
    conn = sqlite3.connect('manual_predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions 
        (timestamp, temperature, humidity, light, motion, prediction, confidence, user_notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('timestamp', datetime.now().isoformat()),
        data.get('temperature'),
        data.get('humidity'),
        data.get('light'),
        data.get('motion'),
        data.get('prediction'),
        data.get('confidence'),
        data.get('notes', '')
    ))
    conn.commit()
    conn.close()

# Flask Routes
@app.route('/')
def index():
    return render_template('manual_predictor.html')

@app.route('/api/train_model', methods=['POST'])
def train_model():
    """Train model with user data"""
    try:
        data = request.get_json()
        
        if 'training_data' in data:
            # Train with provided data
            result = predictor.train_with_user_data(data_dict=data['training_data'])
        else:
            # Train with sample data
            result = predictor.train_with_user_data()
        
        if 'error' not in result:
            predictor.save_model()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """Make a single energy prediction"""
    try:
        data = request.get_json()
        
        # Extract prediction parameters
        prediction_params = {
            'temperature': data.get('temperature'),
            'humidity': data.get('humidity'),
            'light': data.get('light'),
            'motion': data.get('motion')
        }
        
        # Make prediction
        result = predictor.predict_single(**prediction_params)
        
        if 'error' not in result:
            # Save prediction to database
            save_data = {
                **prediction_params,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'notes': data.get('notes', '')
            }
            save_prediction(save_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict_batch', methods=['POST'])
def make_batch_prediction():
    """Make multiple predictions at once"""
    try:
        data = request.get_json()
        batch_data = data.get('batch_data', [])
        
        result = predictor.predict_batch(batch_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/model_info')
def get_model_info():
    """Get information about the current model"""
    return jsonify(predictor.get_model_info())

@app.route('/api/prediction_history')
def get_prediction_history():
    """Get history of manual predictions"""
    conn = sqlite3.connect('manual_predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM predictions 
        ORDER BY timestamp DESC 
        LIMIT 50
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row[0],
            'timestamp': row[1],
            'temperature': row[2],
            'humidity': row[3],
            'light': row[4],
            'motion': row[5],
            'prediction': row[6],
            'confidence': row[7],
            'notes': row[8]
        })
    
    return jsonify(history)

@app.route('/api/upload_training_data', methods=['POST'])
def upload_training_data():
    """Upload CSV file for model training"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)
        
        # Train model with uploaded data
        result = predictor.train_with_user_data(data_file=temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if 'error' not in result:
            predictor.save_model()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    init_db()
    
    # Try to load existing model, otherwise train with sample data
    if not predictor.load_model():
        print("No existing model found, training with sample data...")
        predictor.train_with_user_data()
        predictor.save_model()
        print("Model trained and saved!")
    else:
        print("Loaded existing model")
    
    app.run(debug=True, host='0.0.0.0', port=5000)