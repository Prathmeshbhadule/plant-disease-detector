import os
import streamlit as st

# Fix for Render port binding
os.environ["STREAMLIT_SERVER_PORT"] = os.environ.get("PORT", "8501")
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

import tensorflow as tf 
import numpy as np 
from PIL import Image 
import json 
from collections import defaultdict 
from datetime import datetime 
 
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide") 
 
IMG_SIZE = (160, 160) 
 
st.markdown(""" 
<style> 
    .stApp { 
        background-color: #d8d8d8; 
    } 
    .title-text { 
        font-family: Georgia, 'Times New Roman', serif; 
        font-size: 4.2rem; 
        font-weight: 700; 
        color: #14361f; 
        letter-spacing: 1px; 
        margin-bottom: 0; 
    } 
    .subtitle-text { 
        font-size: 1.25rem; 
        color: #2b2b2b; 
        margin-top: 6px; 
        margin-bottom: 1.8rem; 
    } 
 
    /* Uploader box */ 
    [data-testid="stFileUploaderDropzone"] { 
        background-color: #1c1c24; 
        border: 2px dashed #3a3a48; 
        border-radius: 10px; 
        padding: 1.2rem; 
    } 
    [data-testid="stFileUploaderDropzoneInstructions"] span, 
    [data-testid="stFileUploaderDropzoneInstructions"] small { 
        color: #f0f0f0 !important; 
        font-size: 1.05rem; 
    } 
    [data-testid="stBaseButton-secondary"] { 
        background-color: #ffffff !important; 
        color: #14361f !important; 
        font-weight: 600 !important; 
        border-radius: 6px !important; 
        border: none !important; 
    } 
 
    div.stButton > button { 
        background-color: #14361f; 
        color: white; 
        font-weight: 600; 
        letter-spacing: 2px; 
        border-radius: 6px; 
        padding: 0.6rem 2rem; 
        border: none; 
        width: 100%; 
    } 
    div.stButton > button:hover { 
        background-color: #1f5230; 
        color: white; 
    } 
 
    .result-text { 
        font-size: 1.3rem; 
        font-weight: 600; 
        color: #14361f; 
        margin-top: 1rem; 
    } 
 
    .diseases-heading, .history-heading { 
        color: #14361f; 
        font-weight: 700; 
        font-size: 1.4rem; 
        margin-bottom: 10px; 
    } 
 
    table.diseases { 
        width: 100%; 
        border-collapse: collapse; 
        font-size: 0.85rem; 
    } 
    table.diseases th { 
        background-color: #14361f; 
        color: #ffffff !important; 
        padding: 10px; 
        text-align: left; 
    } 
    table.diseases td { 
        padding: 10px; 
        border-bottom: 1px solid #bbb; 
        vertical-align: top; 
        color: #1a1a1a !important; 
        background-color: #eeeeee; 
    } 
    table.diseases tr:nth-child(even) td { 
        background-color: #e0e0e0; 
    } 
 
    .history-card { 
        background-color: #eeeeee; 
        border-radius: 8px; 
        padding: 10px; 
        margin-bottom: 10px; 
    } 
    .history-result { 
        font-weight: 600; 
        color: #14361f; 
    } 
    .history-time { 
        font-size: 0.8rem; 
        color: #555; 
    } 
</style> 
""", unsafe_allow_html=True) 
 
@st.cache_resource 
def load_model(): 
    model = tf.keras.models.load_model("models/crop_disease_model.keras") 
    with open("models/class_names.json") as f: 
        class_names = json.load(f) 
    return model, class_names 
 
model, class_names = load_model() 
 
def predict(image: Image.Image): 
    img = image.resize(IMG_SIZE) 
    img_array = tf.keras.utils.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    preds = model.predict(img_array, verbose=0)[0] 
    idx = np.argmax(preds) 
    return class_names[idx], float(preds[idx]) 
 
grouped = defaultdict(list) 
for name in class_names: 
    crop, disease = name.split("___") 
    grouped[crop.replace("_", " ")].append(disease.replace("_", " ")) 
 
if "history" not in st.session_state: 
    st.session_state.history = [] 
 
st.markdown('<p class="title-text">PLANT DISEASE DETECTOR</p>', unsafe_allow_html=True) 
st.markdown('<p class="subtitle-text">Share a picture of the plant and get immediate results!</p>', unsafe_allow_html=True) 
 
tab_detect, tab_history = st.tabs(["🔍 Detect", "🕘 History"]) 
 
with tab_detect: 
    left, right = st.columns([1, 1.3]) 
 
    with left: 
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed") 
        image = None 
        if uploaded_file is not None: 
            image = Image.open(uploaded_file).convert("RGB") 
            st.image(image, use_container_width=True) 
 
        analyze = st.button("ANALYZE") 
 
        if analyze and image is not None: 
            with st.spinner("Analyzing..."): 
                label, conf = predict(image) 
            st.markdown(f'<p class="result-text">Result = {label} ({conf*100:.1f}%)</p>', unsafe_allow_html=True) 
 
            st.session_state.history.insert(0, { 
                "image": image, 
                "label": label, 
                "confidence": conf, 
                "time": datetime.now().strftime("%d %b, %H:%M:%S") 
            }) 
        elif analyze and image is None: 
            st.warning("Please upload an image first.") 
 
    with right: 
        st.markdown('<p class="diseases-heading">Supported Diseases</p>', unsafe_allow_html=True) 
        rows = "" 
        for crop, diseases in grouped.items(): 
            rows += f"<tr><td><b>{crop}</b></td><td>{', '.join(diseases)}</td></tr>" 
        st.markdown(f'<table class="diseases"><tr><th>Name</th><th>Class Names</th></tr>{rows}</table>', unsafe_allow_html=True) 
 
with tab_history: 
    st.markdown('<p class="history-heading">Past Predictions</p>', unsafe_allow_html=True) 
 
    if not st.session_state.history: 
        st.info("No predictions yet. Analyze a leaf photo in the Detect tab to see history here.") 
    else: 
        if st.button("Clear history"): 
            st.session_state.history = [] 
            st.rerun() 
 
        for entry in st.session_state.history: 
            col_img, col_info = st.columns([1, 4]) 
            with col_img: 
                st.image(entry["image"], width=100) 
            with col_info: 
                st.markdown(f""" 
                <div class="history-card"> 
                    <div class="history-result">{entry['label']} — {entry['confidence']*100:.1f}%</div> 
                    <div class="history-time">{entry['time']}</div> 
                </div> 
                """, unsafe_allow_html=True)

st.write(" ")