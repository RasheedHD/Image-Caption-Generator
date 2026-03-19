import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️")

st.title("Image Caption Generator")
st.write("Upload an image and generate a caption using a BLIP vision-language model.")

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    model.eval()
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(output[0], skip_special_tokens=True)

        st.subheader("Caption")
        st.write(caption)

        st.subheader("Model Info")
        st.write("Model: Salesforce/blip-image-captioning-base")
        st.write(f"Device: {device}")