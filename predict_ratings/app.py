from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'csv'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store model and data
model_data = {
    'model': None,
    'scaler': None,
    'encoders': {},
    'feature_names': [],
    'results': None,
    'target_column': None
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df, target_column):
    """Preprocess the dataset"""
    # Handle missing values
    df = df.dropna()
    
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Encode categorical variables
    encoders = {}
    X_encoded = X.copy()
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # Convert all columns to numeric
    for col in X_encoded.columns:
        X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')
    
    # Remove any remaining NaN values after conversion
    X_encoded = X_encoded.fillna(0)
    
    return X_encoded, y, encoders

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple regression models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X_train.columns, abs(model.coef_)))
        else:
            feature_importance = {}
        
        results[name] = {
            'mse': round(mse, 4),
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'r2': round(r2, 4),
            'feature_importance': {k: round(v, 4) for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)}
        }
        
        trained_models[name] = model
    
    return results, trained_models

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        target_column = request.form.get('target_column', 'rating')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read CSV
            df = pd.read_csv(filepath)
            
            # Store target column
            model_data['target_column'] = target_column
            
            # Get data preview and stats
            data_preview = df.head(10).to_dict('records')
            
            # Get column info
            columns_info = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                missing = int(df[col].isnull().sum())
                unique = int(df[col].nunique())
                columns_info.append({
                    'name': col,
                    'type': col_type,
                    'missing': missing,
                    'unique': unique
                })
            
            data_stats = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns_info': columns_info,
                'missing_values': df.isnull().sum().to_dict()
            }
            
            # Add target column stats if it exists
            if target_column in df.columns:
                data_stats['target_stats'] = {
                    'mean': round(df[target_column].mean(), 2),
                    'std': round(df[target_column].std(), 2),
                    'min': round(df[target_column].min(), 2),
                    'max': round(df[target_column].max(), 2)
                }
            
            # Store data globally
            app.config['df'] = df
            
            return jsonify({
                'success': True,
                'preview': data_preview,
                'stats': data_stats,
                'columns': list(df.columns)
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload a CSV file.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train the ML model"""
    try:
        df = app.config.get('df')
        if df is None:
            return jsonify({'success': False, 'error': 'No data available. Upload a CSV first.'})
        
        target_column = model_data['target_column']
        if not target_column:
            return jsonify({'success': False, 'error': 'Target column not specified.'})
        
        # Preprocess data
        X, y, encoders = preprocess_data(df, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Train models
        results, models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Find best model based on R2 score
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        
        # Store model data
        model_data['model'] = models[best_model_name]
        model_data['scaler'] = scaler
        model_data['encoders'] = encoders
        model_data['feature_names'] = list(X.columns)
        model_data['results'] = results
        model_data['best_model'] = best_model_name
        
        return jsonify({
            'success': True,
            'results': results,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'best_model': best_model_name,
            'feature_names': list(X.columns)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction for new data"""
    try:
        if model_data['model'] is None:
            return jsonify({'success': False, 'error': 'Model not trained yet.'})
        
        data = request.json
        
        # Create dataframe from input
        input_data = pd.DataFrame([data])
        
        # Encode categorical variables
        for col, encoder in model_data['encoders'].items():
            if col in input_data.columns:
                try:
                    input_data[col] = encoder.transform(input_data[col])
                except ValueError:
                    # Handle unknown categories by using the most frequent class
                    input_data[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Convert all to numeric
        for col in input_data.columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
        
        input_data = input_data.fillna(0)
        
        # Ensure columns match training data
        for col in model_data['feature_names']:
            if col not in input_data.columns:
                input_data[col] = 0
        
        input_data = input_data[model_data['feature_names']]
        
        # Scale features
        input_scaled = model_data['scaler'].transform(input_data)
        
        # Make prediction
        prediction = model_data['model'].predict(input_scaled)[0]
        prediction = round(prediction, 2)
        
        return jsonify({
            'success': True,
            'predicted_value': prediction
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_categorical_values', methods=['POST'])
def get_categorical_values():
    """Get unique values for categorical columns"""
    try:
        df = app.config.get('df')
        if df is None:
            return jsonify({'success': False, 'error': 'No data available.'})
        
        column = request.json.get('column')
        if column not in df.columns:
            return jsonify({'success': False, 'error': f'Column {column} not found'})
        
        unique_values = df[column].dropna().unique().tolist()
        return jsonify({
            'success': True,
            'values': unique_values
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)