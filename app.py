from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS
import os
import json

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder
CORS(app)

# Category mapping dictionary - maintaining the order: age, workclass, education, marital-status, occupation, relationship, race, gender, hours-per-week, native-country
category_mappings = {
    # Education mappings
    'preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6, '12th': 7, 
    'hs-grad': 8, 'some-college': 9, 'assoc-voc': 10, 'assoc-acdm': 11, 'bachelors': 12, 'masters': 13, 
    'prof-school': 14, 'doctorate': 15,
    
    # Workclass mappings
    'federal-gov': 0, 'local-gov': 1, 'private': 2, 'self-emp-inc': 3, 'self-emp-not-inc': 4, 
    'state-gov': 5, 'without-pay': 6,
    
    # Marital status mappings
    'divorced': 0, 'married-af-spouse': 1, 'married-civ-spouse': 2, 'married-spouse-absent': 3, 
    'never-married': 4, 'separated': 5, 'widowed': 6,
    
    # Occupation mappings
    'adm-clerical': 0, 'armed-forces': 1, 'craft-repair': 2, 'exec-managerial': 3, 'farming-fishing': 4, 
    'handlers-cleaners': 5, 'machine-op-inspct': 6, 'other-service': 7, 'priv-house-serv': 8, 
    'prof-specialty': 9, 'protective-serv': 10, 'sales': 11, 'tech-support': 12, 'transport-moving': 13,
    
    # Relationship mappings
    'husband': 0, 'not-in-family': 1, 'other-relative': 2, 'own-child': 3, 'unmarried': 4, 'wife': 5,
    
    # Race mappings
    'amer-indian-eskimo': 0, 'asian-pac-islander': 1, 'black': 2, 'other': 3, 'white': 4,
    
    # Gender mappings
    'female': 0, 'male': 1,
    
    # Native country mappings
    'cambodia': 0, 'canada': 1, 'china': 2, 'columbia': 3, 'cuba': 4, 'dominican-republic': 5, 
    'ecuador': 6, 'el-salvador': 7, 'england': 8, 'france': 9, 'germany': 10, 'greece': 11, 
    'guatemala': 12, 'haiti': 13, 'holand-netherlands': 14, 'honduras': 15, 'hong': 16, 'hungary': 17, 
    'india': 18, 'iran': 19, 'ireland': 20, 'italy': 21, 'jamaica': 22, 'japan': 23, 'laos': 24, 
    'mexico': 25, 'nicaragua': 26, 'outlying-us(guam-usvi-etc)': 27, 'peru': 28, 'philippines': 29, 
    'poland': 30, 'portugal': 31, 'puerto-rico': 32, 'scotland': 33, 'south': 34, 'taiwan': 35, 
    'thailand': 36, 'trinadad&tobago': 37, 'united-states': 38, 'vietnam': 39, 'yugoslavia': 40
}

def predict_manual_input(input_features, model_path="xgb_best_model.pkl", threshold_path="xgb_best_threshold.txt"):
    """
    Predicts class based on manual input features using the saved model and threshold.
    
    Parameters:
    - input_features: list or array-like of feature values in the same order as training
    - model_path: str, path to the saved model
    - threshold_path: str, path to the saved threshold
    
    Returns:
    - predicted_class: int (0 or 1)
    - probability: float (probability of class 1)
    """
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Check if threshold file exists
        if not os.path.exists(threshold_path):
            raise FileNotFoundError(f"Threshold file not found: {threshold_path}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load threshold
        with open(threshold_path, "r") as f:
            threshold = float(f.read().strip())
        
        # Convert input to 2D array
        input_array = np.array(input_features).reshape(1, -1)
        
        # Predict probability
        prob = model.predict_proba(input_array)[0, 1]
        
        # Apply threshold
        prediction = int(prob >= threshold)
        
        # Convert numpy types to Python native types for JSON serialization
        prediction = int(prediction)
        prob = float(prob)
        
        print(f"Predicted Class: {prediction} (1=>50K, 0=<=50K)")
        print(f"Probability of >50K: {prob:.4f}")
        print(f"Used Threshold: {threshold:.4f}")
        
        return prediction, prob
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise e

def validate_form_data(form_data):
    """
    Validates the form data to ensure all required fields are present and valid.
    
    Parameters:
    - form_data: dict containing form data
    
    Returns:
    - bool: True if valid, raises ValueError if invalid
    """
    required_fields = [
        'age', 'workclass', 'education', 'maritalStatus', 'occupation', 
        'relationship', 'race', 'gender', 'hoursPerWeek', 'nativeCountry'
    ]
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in form_data or not form_data[field]:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate age
    try:
        age = int(form_data['age'])
        if age < 16 or age > 100:
            raise ValueError("Age must be between 16 and 100")
    except ValueError:
        raise ValueError("Age must be a valid number")
    
    # Validate hours per week
    try:
        hours = int(form_data['hoursPerWeek'])
        if hours < 1 or hours > 100:
            raise ValueError("Hours per week must be between 1 and 100")
    except ValueError:
        raise ValueError("Hours per week must be a valid number")
    
    return True

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get JSON data from request
        form_data = request.get_json()
        
        if not form_data:
            return jsonify({"error": "No data received"}), 400
        
        print(f"Received form data: {form_data}")
        
        # Validate form data
        try:
            validate_form_data(form_data)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        # Convert string values to lowercase for consistency
        processed_data = {}
        for key, value in form_data.items():
            if isinstance(value, str):
                processed_data[key] = value.lower().strip()
            else:
                processed_data[key] = value
        
        print(f"Processed data: {processed_data}")
        
        # Prepare numerical data in the correct order:
        # age, workclass, education, marital-status, occupation, relationship, race, gender, hours-per-week, native-country
        try:
            numerical_data = [
                int(processed_data['age']),                           # age (int)
                category_mappings[processed_data['workclass']],       # workclass
                category_mappings[processed_data['education']],       # education
                category_mappings[processed_data['maritalStatus']],   # marital-status
                category_mappings[processed_data['occupation']],      # occupation
                category_mappings[processed_data['relationship']],    # relationship
                category_mappings[processed_data['race']],            # race
                category_mappings[processed_data['gender']],          # gender
                int(processed_data['hoursPerWeek']),                 # hours-per-week (int)
                category_mappings[processed_data['nativeCountry']]    # native-country
            ]
        except KeyError as e:
            missing_key = str(e).strip("'")
            return jsonify({"error": f"Invalid selection for field: {missing_key}. Please check your selection."}), 400
        except ValueError as e:
            return jsonify({"error": f"Invalid numerical value: {str(e)}"}), 400
        
        print(f"Numerical data: {numerical_data}")
        
        # Make prediction
        try:
            prediction, probability = predict_manual_input(numerical_data)
            
            # Ensure all values are JSON serializable
            prediction = int(prediction)
            probability = float(probability)
            
            print(f"Final prediction: {prediction}, probability: {probability}")
            
        except FileNotFoundError as pred_error:
            print(f"Model file error: {str(pred_error)}")
            return jsonify({"error": "Model files not found. Please ensure the model files are in the correct location."}), 500
        except Exception as pred_error:
            print(f"Prediction error: {str(pred_error)}")
            return jsonify({"error": f"Model prediction failed: {str(pred_error)}"}), 500
        
        # Prepare result
        result = {
            'prediction': prediction,
            'salary': '>50K' if prediction == 1 else '<=50K',
            'confidence': round(probability * 100, 2)
        }
        
        print(f"Sending result: {result}")
        return jsonify(result), 200
        
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON data"}), 400
    except Exception as e:
        print(f"Unexpected error in predict route: {str(e)}")
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if model files exist
        model_exists = os.path.exists("xgb_best_model.pkl")
        threshold_exists = os.path.exists("xgb_best_threshold.txt")
        
        return jsonify({
            "status": "healthy",
            "model_file_exists": model_exists,
            "threshold_file_exists": threshold_exists,
            "ready_for_predictions": model_exists and threshold_exists
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/test', methods=['POST'])
def test_endpoint():
    """Test endpoint to debug form submission"""
    try:
        data = request.get_json()
        print(f"Test endpoint received: {data}")
        return jsonify({
            "status": "success",
            "received_data": data,
            "message": "Test endpoint working correctly"
        }), 200
    except Exception as e:
        print(f"Test endpoint error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Check if model files exist on startup
    model_exists = os.path.exists("xgb_best_model.pkl")
    threshold_exists = os.path.exists("xgb_best_threshold.txt")
    
    if not model_exists:
        print("WARNING: Model file 'xgb_best_model.pkl' not found!")
    if not threshold_exists:
        print("WARNING: Threshold file 'xgb_best_threshold.txt' not found!")
    
    if model_exists and threshold_exists:
        print("✅ All model files found. Server ready for predictions.")
    else:
        print("⚠️  Some model files are missing. Predictions may fail.")
    
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)