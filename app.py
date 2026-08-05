import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load trained model
model = YOLO("best.pt")

# Pesticide recommendations
pesticides = {
    "APHIDS": "Imidacloprid or Neem Oil",
    "WHITEFLIES": "Acetamiprid or Buprofezin",
    "GRASSHOPPER": "Malathion Spray",
    "LADYBUG": "Beneficial Insect - No pesticide required"
}

st.set_page_config(
    page_title="SmartPest AI",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 SmartPest AI")
st.subheader("AI-Based Insect Detection and Pesticide Recommendation")

uploaded_file = st.file_uploader(
    "Upload an insect image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp.name)

    results = model.predict(temp.name)

    insect = results[0].names[results[0].probs.top1]
    confidence = float(results[0].probs.top1conf)

    st.success(f"Detected Insect: {insect}")

    st.write(f"Confidence: {confidence*100:.2f}%")

    if insect.upper() in pesticides:
        st.info(f"Recommended Solution: {pesticides[insect.upper()]}")
    else:
        st.warning("No recommendation available.")
