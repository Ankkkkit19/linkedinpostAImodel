import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

# Title
st.title("📰 AI News Summarizer + Image Generator")

# Input box
news_text = st.text_area("Paste your news article here:")

# Load summarizer (only once for efficiency)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Load image generator (Stable Diffusion)
@st.cache_resource
def load_image_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")   # CPU mode, slow but works on normal PC
    return pipe

if st.button("Generate Summary & Image"):
    if news_text.strip() != "":
        summarizer = load_summarizer()
        summary = summarizer(news_text, max_length=120, min_length=30, do_sample=False)[0]['summary_text']

        st.subheader("📌 News Summary")
        st.write(summary)

        st.subheader("🖼 Generated Image")
        pipe = load_image_model()
        image = pipe(summary).images[0]
        st.image(image, caption="AI Generated Image", use_column_width=True)

    else:
        st.warning("⚠️ Please enter some news text first!")
