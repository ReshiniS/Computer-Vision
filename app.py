import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from PIL import Image
from huggingface_hub import hf_hub_download
import pandas as pd
import base64
from datetime import datetime

# -----------------------------------------------------------
# Setup page
st.set_page_config(page_title="Food Waste Detection", layout="centered")
st.title("Food Waste Detection App")
st.write("Upload an image OR use your webcam to detect food waste items! 📸")

# -----------------------------------------------------------
# Load model from Hugging Face Hub
@st.cache_resource
def load_model():
    with st.spinner('Downloading model from HuggingFace...'):
        model_path = hf_hub_download(
            repo_id="Reshini/Food_Waste_Detection_YOLOv8",  # Replace with your repo
            filename="best (1).pt"
        )
    return YOLO(model_path)

model = load_model()

# -----------------------------------------------------------
# Prediction function
def predict_image(image_np):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        img = Image.fromarray(image_np)
        img.save(tmp_file.name)
        results = model.predict(source=tmp_file.name, save=False)
    return results

# -----------------------------------------------------------
# Upload or Webcam input
option = st.radio("Choose Input Method:", ('Upload Image', 'Use Webcam'))

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        image_np = np.array(image)

        with st.spinner('Detecting...'):
            results = predict_image(image_np)

        st.subheader("Detection Results")
        detected_image = results[0].plot()
        st.image(detected_image, channels="BGR", use_container_width=True)

        if results[0].boxes.cls.numel() > 0:
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            detected_classes = [model.names[cid] for cid in class_ids]
            st.success(f"Detected Items: {', '.join(set(detected_classes))}")

            st.subheader("📏 Estimated Portion Sizes (grams)")
            boxes = results[0].boxes.xywh.cpu().numpy()  # x, y, w, h format

            # Define realistic average weights per class
            scaling_factors = {
                "burger": 180,
                "green apple": 130,
                "red apple": 140,
                "french fries": 90,
            }

            # To store results for CSV
            log_data = []

            for i, box in enumerate(boxes):
                width = box[2]
                height = box[3]
                image_size = 640
                normalized_area = (width / image_size) * (height / image_size)

                label = model.names[class_ids[i]]
                scale = scaling_factors.get(label, 100)
                estimated_grams = round(min(normalized_area * scale, scale), 1)

                st.write(f"• **{label}** – Estimated: ~{estimated_grams}g")

                log_data.append({
                    "Item": label,
                    "Estimated_grams": estimated_grams
                })

            # Convert to CSV with timestamped filename
            if log_data:
                df = pd.DataFrame(log_data)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"detection_{timestamp}.csv"
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Results CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("No food waste items detected.")

elif option == 'Use Webcam':
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot access webcam")
    else:
        st.info("Webcam is running... Click 'Stop Webcam' to stop.")

    stop = st.button('Stop Webcam')

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with st.spinner('Detecting...'):
            results = predict_image(frame_rgb)

        annotated_frame = results[0].plot()
        stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        stop = st.button('Stop Webcam')

    cap.release()
    st.success("Webcam stopped. Thanks!")

# -----------------------------------------------------------
# Footer
st.markdown("---")
st.caption("Made with by Team 310 | Powered by YOLOv8 and Streamlit")
