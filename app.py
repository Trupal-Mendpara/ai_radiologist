import streamlit as st
from models.cnn_model import predict
from models.llm_report_generator import generate_report
from utils.image_preprocessing import preprocess_image
from utils.load_css import local_css
from datetime import datetime
from PIL import Image
from utils.pdf_creator import generate_pdf_report
import tensorflow as tf

# --- App Config ---
st.set_page_config(page_title="AI Radiologist", layout="centered")

# --- Load CSS ---
local_css("assets/style.css")

st.markdown("<div class='title'>AI Radiologist👩🏻‍⚕️</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div class='h1'>📤 Upload the Chest X-ray</div>", unsafe_allow_html=True)

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload below", type=["jpg", "jpeg", "png"])

# --- Prediction and Report ---
if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray', use_container_width=True)

    # Preprocess Image
    image = preprocess_image(uploaded_file)

    # Run CNN Prediction
    disease, confidence = predict(image)
    if disease == "Invalid":
        st.error("❌ The uploaded image not a valid. Please upload a valid Chest X-ray.")
        st.stop()
    st.success(f"The given image shows that the person is suffering from **{disease}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")

    confirm = st.radio("Do you want to generate the radiology report?", ["No", "Yes"])
    if confirm == "Yes":        
        if "patient_submitted" not in st.session_state:
            st.session_state["patient_submitted"] = False
        # --- Patient Information ---
        with st.form("patient_form"):
            st.subheader("🧍 Patient Information")
            name = st.text_input("Name")
            age = st.slider("Age", 1, 120, 35)
            gender = st.radio("Gender", ["Male", "Female", "Other"])
            submitted = st.form_submit_button("Submit Info")

        if submitted:
            if not name.strip():
                st.error("❌ Patient name is required.")
            else:
                st.session_state["patient_submitted"] = True
                st.success("✅ Patient information submitted successfully.")

        # Prepare patient info for LLM
        current_time = datetime.now().strftime("%Y-%m-%d")
        patient_info = f"""
        Name: {name}
        Age: {age}
        Gender: {gender}
        Date: {current_time}
        """
        if st.button("📝 Generate Report"):
            if not st.session_state["patient_submitted"]:
                st.error("⚠️ Please submit patient information before generating the report.")
            else:    
                st.info("Generating the report... Please wait.")
                report = generate_report(disease, confidence, patient_info)
                st.subheader("📄 AI-Generated Report")
                st.write(report)
                st.success("Report generated successfully!")
                # Download Option
                pfd = generate_pdf_report(report)
                st.download_button(
                    label="📥 Download Report as pdf",
                    data=pfd,
                    file_name=f"{name}_radiology_report.pdf",
                    mime="application/pdf"
                )