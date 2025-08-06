import streamlit as st
from utils import load_disease_model, predict_disease, scrape_doctors

st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title("🩺 AI-Powered Disease Prediction + Doctor Finder")

model = load_disease_model()
class_names = ['Acne', 'Eczema', 'Psoriasis']  # Replace with your classes

# Sidebar: Upload image + city
st.sidebar.header("1️⃣ Upload Your Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

st.sidebar.header("2️⃣ Enter Your City")
user_city = st.sidebar.text_input("Where are you located?", value="Cairo")

# Start prediction
if uploaded_file and user_city:
    st.image(uploaded_file, caption="Your Image", use_column_width=True)

    if st.sidebar.button("🔍 Predict Disease & Find Doctors"):
        with st.spinner("Processing..."):
            prediction, confidence = predict_disease(model, uploaded_file, class_names)

            st.success(f"🧠 Predicted Disease: **{prediction}** ({confidence*100:.2f}%)")
            st.subheader("🏥 Recommended Doctors in Your Area")

            try:
                doctor_df = scrape_doctors(user_city, specialty=prediction)
                if doctor_df.empty:
                    st.warning("No doctors found. Try another city.")
                else:
                    for _, row in doctor_df.iterrows():
                        st.markdown(f"""
                        <div style="border:1px solid #ddd; padding:16px; border-radius:10px; margin-bottom:10px; background-color:#f9f9f9;">
                            <h4 style="margin-bottom:5px;">🧑‍⚕️ {row['name']}</h4>
                            <p style="margin:0;"><strong>Specialty:</strong> {row['specialty']}</p>
                            <p style="margin:0;"><strong>Location:</strong> 📍 {row['location']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
else:
    st.info("📸 Upload a medical image and enter your city to begin.")
