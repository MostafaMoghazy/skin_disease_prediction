import streamlit as st
import sys
import traceback
from utils import load_disease_model_safe, predict_disease, scrape_doctors

# Configure Streamlit page
st.set_page_config(
    page_title="Disease Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #1f77b4;
    margin-bottom: 30px;
}
.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-weight: bold;
}
.doctor-card {
    border: 1px solid #ddd;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 10px;
    background-color: #f9f9f9;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🩺 AI-Powered Disease Prediction + Doctor Finder</h1>', unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# Model loading section
if not st.session_state.model_loaded:
    st.info("🔄 Loading AI model... This may take a moment.")
    
    try:
        with st.spinner("Loading disease prediction model..."):
            model = load_disease_model_safe()
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.success("✅ Model loaded successfully!")
            st.experimental_rerun()
    except Exception as e:
        st.session_state.error_message = str(e)
        st.error(f"❌ Failed to load model: {e}")
        
        # Show detailed error information
        with st.expander("🔧 Troubleshooting Information"):
            st.write("**Common causes and solutions:**")
            st.write("1. **Architecture Mismatch**: The model was saved with a different architecture")
            st.write("2. **Version Compatibility**: Different TensorFlow/Keras versions")
            st.write("3. **Missing Dependencies**: Some required packages might be missing")
            st.write("4. **Corrupted File**: The model file might be corrupted")
            
            st.write("\n**Recommended fixes:**")
            st.code("""
# Option 1: Retrain and save your model properly
model.save('skindisease.keras')

# Option 2: Save weights separately
model.save_weights('skindisease_weights.h5')

# Option 3: Use SavedModel format
tf.saved_model.save(model, 'skindisease_savedmodel')
            """)
            
            st.write("**Full Error Details:**")
            st.code(traceback.format_exc())

# Main application (only show if model is loaded)
if st.session_state.model_loaded and st.session_state.model is not None:
    
    # Define class names - update these to match your model
    class_names = ['Acne', 'Eczema', 'Psoriasis']  # Replace with your actual classes
    
    # Sidebar controls
    st.sidebar.header("📸 Step 1: Upload Your Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a medical image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the affected area"
    )
    
    st.sidebar.header("📍 Step 2: Enter Your Location")
    user_city = st.sidebar.text_input(
        "Where are you located?", 
        value="Cairo",
        help="Enter your city to find nearby doctors"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if uploaded_file:
            st.subheader("📋 Your Uploaded Image")
            st.image(uploaded_file, caption="Medical Image for Analysis", use_column_width=True)
            
            # Show image details
            st.write(f"**File name:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size} bytes")
            st.write(f"**File type:** {uploaded_file.type}")
        else:
            st.info("👆 Please upload an image using the sidebar")
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Prediction section
        if uploaded_file and user_city:
            
            # Prediction button
            if st.button("🔍 Analyze Image & Find Doctors", type="primary"):
                
                # Create tabs for results
                tab1, tab2 = st.tabs(["🧠 AI Analysis", "🏥 Doctor Recommendations"])
                
                with tab1:
                    with st.spinner("🔄 Analyzing image..."):
                        try:
                            prediction, confidence = predict_disease(
                                st.session_state.model, 
                                uploaded_file, 
                                class_names
                            )
                            
                            # Display prediction results
                            st.success(f"**Predicted Condition:** {prediction}")
                            
                            # Confidence meter
                            confidence_percent = confidence * 100
                            st.metric(
                                label="Confidence Level",
                                value=f"{confidence_percent:.1f}%"
                            )
                            
                            # Progress bar for confidence
                            st.progress(confidence)
                            
                            # Confidence interpretation
                            if confidence > 0.8:
                                st.success("🎯 High confidence prediction")
                            elif confidence > 0.6:
                                st.warning("⚠️ Moderate confidence - consider consulting a doctor")
                            else:
                                st.error("❌ Low confidence - definitely consult a medical professional")
                                
                            # Important disclaimer
                            st.warning("⚠️ **Medical Disclaimer**: This is an AI prediction tool and should not replace professional medical diagnosis. Always consult with a qualified healthcare provider.")
                            
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {e}")
                            st.write("Please try:")
                            st.write("- Using a different image format")
                            st.write("- Ensuring the image is clear and properly focused")
                            st.write("- Checking if the image shows the medical condition clearly")
                
                with tab2:
                    with st.spinner("🔄 Finding doctors in your area..."):
                        try:
                            doctor_df = scrape_doctors(user_city, specialty=prediction)
                            
                            if not doctor_df.empty:
                                st.success(f"Found {len(doctor_df)} doctors in {user_city}")
                                
                                # Display doctors
                                for idx, doctor in doctor_df.iterrows():
                                    st.markdown(f"""
                                    <div class="doctor-card">
                                        <h4 style="margin-bottom:5px; color:#1f77b4;">🧑‍⚕️ {doctor['name']}</h4>
                                        <p style="margin:5px 0;"><strong>Specialty:</strong> {doctor['specialty']}</p>
                                        <p style="margin:5px 0;"><strong>Location:</strong> 📍 {doctor['location']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning(f"❌ No doctors found in {user_city}")
                                st.info("💡 **Suggestions:**")
                                st.write("- Try a different city name")
                                st.write("- Use a major city nearby")
                                st.write("- Contact your local medical directory")
                                
                        except Exception as e:
                            st.error(f"❌ Error finding doctors: {e}")
                            st.info("You can manually search for dermatologists in your area using:")
                            st.write("- Google Maps")
                            st.write("- Local medical directories")
                            st.write("- Hospital websites")
        else:
            if not uploaded_file:
                st.info("📸 Please upload an image first")
            if not user_city:
                st.info("📍 Please enter your city")

    # Additional information section
    st.markdown("---")
    
    with st.expander("ℹ️ About This Tool"):
        st.write("""
        **AI Disease Prediction Tool** uses advanced machine learning to analyze medical images 
        and predict potential skin conditions. The tool can identify:
        
        - **Acne**: Common skin condition with pimples and inflammation
        - **Eczema**: Inflammatory skin condition causing dry, itchy patches
        - **Psoriasis**: Autoimmune condition causing scaly, red patches
        
        **How to use:**
        1. Upload a clear image of the affected area
        2. Enter your city for doctor recommendations
        3. Click 'Analyze Image & Find Doctors'
        4. Review the AI prediction and confidence level
        5. Consult with recommended doctors in your area
        
        **Important Notes:**
        - This tool is for educational purposes only
        - Always consult a qualified healthcare provider
        - AI predictions may not be 100% accurate
        - Multiple factors can affect prediction accuracy
        """)
        
    with st.expander("🔧 Technical Information"):
        st.write(f"""
        **Model Information:**
        - Model Type: Deep Learning CNN
        - Classes: {', '.join(class_names)}
        - Input Size: 224x224 pixels
        - Model Status: {'✅ Loaded' if st.session_state.model_loaded else '❌ Not Loaded'}
        
        **System Requirements:**
        - Python 3.7+
        - TensorFlow 2.x
        - Streamlit
        - PIL (Python Imaging Library)
        """)

else:
    # Show loading interface or error state
    if st.session_state.error_message:
        st.error("⚠️ Application cannot start due to model loading error.")
        
        if st.button("🔄 Retry Loading Model"):
            st.session_state.model_loaded = False
            st.session_state.error_message = None
            st.experimental_rerun()
    else:
        st.info("🔄 Initializing application...")
