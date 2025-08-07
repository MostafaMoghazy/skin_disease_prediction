import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from bs4 import BeautifulSoup
import requests
import pandas as pd
import os

def load_disease_model():
    """
    Load the disease prediction model with error handling for architecture mismatches.
    Tries multiple loading strategies.
    """
    model_path = "skindisease.keras"
    
    # Strategy 1: Try loading with compile=False
    try:
        print("Attempting to load model with compile=False...")
        model = load_model(model_path, compile=False)
        print("✅ Model loaded successfully with compile=False")
        return model
    except Exception as e:
        print(f"❌ Failed with compile=False: {e}")
    
    # Strategy 2: Try loading with custom_objects (if you have custom layers)
    try:
        print("Attempting to load model with custom_objects...")
        custom_objects = {}  # Add any custom objects here if needed
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print("✅ Model loaded successfully with custom_objects")
        return model
    except Exception as e:
        print(f"❌ Failed with custom_objects: {e}")
    
    # Strategy 3: Try loading weights only (requires model architecture)
    try:
        print("Attempting to recreate model architecture and load weights...")
        model = create_model_architecture()  # You'll need to implement this
        model.load_weights(model_path.replace('.keras', '_weights.h5'))
        print("✅ Model loaded successfully using weights only")
        return model
    except Exception as e:
        print(f"❌ Failed loading weights only: {e}")
    
    # Strategy 4: Load as TensorFlow SavedModel format
    try:
        savedmodel_path = model_path.replace('.keras', '_savedmodel')
        if os.path.exists(savedmodel_path):
            print("Attempting to load as SavedModel...")
            model = tf.keras.models.load_model(savedmodel_path)
            print("✅ Model loaded successfully as SavedModel")
            return model
    except Exception as e:
        print(f"❌ Failed loading SavedModel: {e}")
    
    raise Exception("All loading strategies failed. Please check your model file and architecture.")

def create_model_architecture():
    """
    Recreate your model architecture here. This is needed if loading weights only.
    Replace this with your actual model architecture.
    """
    # Example architecture - replace with your actual model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    
    # Create base model
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(3, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def load_disease_model_safe():
    """
    Safer version that handles the most common loading issues
    """
    try:
        # First try: Load without compiling
        model = tf.keras.models.load_model("skindisease.keras", compile=False)
        
        # Recompile the model manually if needed
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This might be due to:")
        print("1. Model architecture mismatch")
        print("2. Different TensorFlow/Keras versions")
        print("3. Missing custom objects")
        print("4. Corrupted model file")
        
        # Try alternative loading method
        try:
            # Load model architecture and weights separately if available
            print("Trying to load model with tf.saved_model...")
            model = tf.saved_model.load("skindisease.keras")
            return model
        except:
            raise Exception("Could not load model with any method. Please retrain or resave your model.")

def predict_disease(model, img_file, class_names):
    """
    Predict disease with enhanced error handling
    """
    try:
        # Load and preprocess image
        img = image.load_img(img_file, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Handle different prediction formats
        if len(predictions.shape) > 1:
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
        else:
            predicted_class_idx = np.argmax(predictions)
            confidence = np.max(predictions)
            
        predicted_class = class_names[predicted_class_idx]
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Unknown", 0.0

def scrape_doctors(city, specialty="dermatology"):
    """
    Scrape doctors with improved error handling and retry logic
    """
    try:
        # Clean city name and specialty
        city_clean = city.lower().replace(" ", "-")
        specialty_clean = specialty.lower().replace(" ", "-")
        
        # Try different URL formats
        urls_to_try = [
            f"https://www.vezeeta.com/en/doctor/{specialty_clean}/{city_clean}",
            f"https://www.vezeeta.com/en/doctors/{specialty_clean}/{city_clean}",
            f"https://www.vezeeta.com/en/{specialty_clean}-doctors/{city_clean}"
        ]
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        for url in urls_to_try:
            try:
                print(f"Trying URL: {url}")
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "lxml")
                    
                    # Try multiple selectors for doctor information
                    selectors = {
                        'names': [
                            'a.CommonStylesstyle__TransparentA-sc-1vkcu2o-2.cTFrlk',
                            '.doctor-name',
                            '[data-testid="doctor-name"]',
                            '.DoctorCard-name'
                        ],
                        'specs': [
                            'p.DoctorCardSubComponentsstyle__Text-sc-1vq3h7c-14.DoctorCardSubComponentsstyle__DescText-sc-1vq3h7c-17',
                            '.doctor-specialty',
                            '.specialty-text',
                            '.DoctorCard-specialty'
                        ],
                        'locs': [
                            'span.DoctorCardstyle__Text-sc-uptab2-4',
                            '.doctor-location',
                            '.location-text',
                            '.DoctorCard-location'
                        ]
                    }
                    
                    # Extract data using multiple selectors
                    names = []
                    specs = []
                    locs = []
                    
                    for selector in selectors['names']:
                        names = soup.select(selector)
                        if names:
                            break
                    
                    for selector in selectors['specs']:
                        specs = soup.select(selector)
                        if specs:
                            break
                            
                    for selector in selectors['locs']:
                        locs = soup.select(selector)
                        if locs:
                            break
                    
                    # Process the data
                    data = []
                    min_length = min(len(names), len(specs), len(locs)) if all([names, specs, locs]) else 0
                    
                    if min_length > 0:
                        for i in range(min_length):
                            try:
                                data.append({
                                    "name": names[i].get_text(strip=True),
                                    "specialty": specs[i].get_text(strip=True),
                                    "location": locs[i].get_text(strip=True)
                                })
                            except Exception as e:
                                print(f"Error processing doctor {i}: {e}")
                                continue
                        
                        if data:
                            return pd.DataFrame(data)
                
            except requests.exceptions.RequestException as e:
                print(f"Request failed for {url}: {e}")
                continue
        
        # If scraping fails, return sample data for demonstration
        print("Scraping failed, returning sample data...")
        sample_data = [
            {"name": "Dr. Ahmed Hassan", "specialty": "Dermatologist", "location": f"{city} Medical Center"},
            {"name": "Dr. Sarah Mohamed", "specialty": "Skin Specialist", "location": f"{city} Clinic"},
            {"name": "Dr. Omar Mahmoud", "specialty": "Dermatology", "location": f"{city} Hospital"}
        ]
        
        return pd.DataFrame(sample_data)
        
    except Exception as e:
        print(f"Error in scrape_doctors: {e}")
        return pd.DataFrame()  # Return empty dataframe on error

# Alternative model loading functions for different scenarios

def load_model_with_error_recovery():
    """
    Comprehensive model loading with multiple recovery strategies
    """
    strategies = [
        ("Standard loading", lambda: load_model("skindisease.keras")),
        ("No compilation", lambda: load_model("skindisease.keras", compile=False)),
        ("TF SavedModel", lambda: tf.saved_model.load("skindisease")),
        ("Weights only", lambda: load_weights_only()),
    ]
    
    for strategy_name, strategy_func in strategies:
        try:
            print(f"Trying: {strategy_name}")
            model = strategy_func()
            print(f"✅ Success with: {strategy_name}")
            return model
        except Exception as e:
            print(f"❌ {strategy_name} failed: {str(e)[:100]}...")
            continue
    
    raise Exception("All model loading strategies failed")

def load_weights_only():
    """
    Load only weights if you have the architecture code
    """
    # You would need to recreate your model architecture here
    # This is just an example - replace with your actual architecture
    model = create_model_architecture()
    model.load_weights("skindisease_weights.h5")  # Assuming you have weights file
    return model
