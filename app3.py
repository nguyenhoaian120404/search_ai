import streamlit as st
from PIL import Image
from io import BytesIO
import requests
import numpy as np
import pandas as pd
import clip
import torch
from pathlib import Path
import os
import json

# Import suggestions từ file riêng
from suggestions_data import SUGGESTIONS_DB, ALL_KEYWORDS

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
DATA_DIR = Path(r"D:\NEU\ki7\Computer Vision\data")
IMAGE_DIR = Path(r"D:\NEU\ki7\Computer Vision\data_output\images")
FAV_FILE = "favorites.json" 


# ================= CÁC HÀM BỔ TRỢ AI & FILE =================

@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

@st.cache_data
def load_database():
    try:
        features = np.load(DATA_DIR / "features.npy")
        photo_ids_df = pd.read_csv(DATA_DIR / "photo_ids.csv")
        metadata_df = pd.read_csv(DATA_DIR / "photos_metadata.csv")
        return features, photo_ids_df, metadata_df
    except Exception as e:
        return None, None, None

def search_engine(query_vector, dataset_features, top_k=12):
    query_vector = query_vector / np.linalg.norm(query_vector)
    similarities = (dataset_features @ query_vector.T).squeeze()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return top_indices, similarities[top_indices]

def encode_text_query(text, model, device):
    text_token = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_token)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

def encode_image_query(image, preprocess, model, device):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

def encode_hybrid_query(image, text, preprocess, model, device, image_weight=0.5):
    """
    Kết hợp features từ cả ảnh và text
    image_weight: tỷ trọng của ảnh (0.0 - 1.0), text_weight = 1 - image_weight
    """
    # Encode image
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # Encode text
    text_token = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_token)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Kết hợp features với trọng số
    text_weight = 1.0 - image_weight
    combined_features = image_weight * image_features + text_weight * text_features
    
    # Normalize lại
    combined_features /= combined_features.norm(dim=-1, keepdim=True)
    
    return combined_features.cpu().numpy()

def get_image_path(photo_id):
    return IMAGE_DIR / f"{photo_id}.jpg"

def get_image_bytes(photo_id, metadata_df):
    path = get_image_path(photo_id)
    if path.exists():
        with open(path, "rb") as f:
            return f.read()
    else:
        meta = metadata_df[metadata_df['photo_id'] == photo_id].iloc[0]
        url = meta['photo_image_url']
        response = requests.get(url)
        return response.content

# ================= QUẢN LÝ YÊU THÍCH =================

def load_favorites():
    if os.path.exists(FAV_FILE):
        try:
            with open(FAV_FILE, "r") as f:
                return json.load(f)
        except: return []
    return []

def save_favorites(fav_list):
    try:
        os.makedirs(os.path.dirname(FAV_FILE), exist_ok=True)
        with open(FAV_FILE, "w") as f:
            json.dump(fav_list, f)
    except: pass

def toggle_favorite(photo_id):
    favs = list(st.session_state.favorites)
    if photo_id in favs:
        favs.remove(photo_id)
        st.toast("Removed from favorites", icon="💔")
    else:
        favs.append(photo_id)
        st.toast("Added to favorites!", icon="❤️")
    st.session_state.favorites = favs
    save_favorites(favs)

# ================= CALLBACKS =================

def on_suggestion_click(text):
    st.session_state.search_input_widget = text
    st.session_state.search_query = text
    st.session_state.trigger_search = True

def reset_to_home():
    st.session_state.search_query = ""
    st.session_state.search_input_widget = ""
    st.session_state.search_results = None
    st.session_state.trigger_search = False
    if 'home_samples' in st.session_state:
        del st.session_state.home_samples

# ================= GIAO DIỆN CHÍNH =================
def main():
    st.set_page_config(page_title="CLIP AI Search", layout="wide", initial_sidebar_state="expanded")

    # CSS Modern & Beautiful
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        /* Global Styles */
        * {
            font-family: 'Inter', sans-serif;
        }

        /* ===== MAIN CONTENT (WHITE BACKGROUND) ===== */
        .main {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 0;
        }

        /* ===== CONTENT CONTAINER ===== */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background: #ffffff;
            border-radius: 20px;
            margin: 1rem;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.12);
            border: 1px solid rgba(102, 126, 234, 0.08);
        }

        /* ===== SIDEBAR (Soft Purple) ===== */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem 1rem;
            box-shadow: 2px 0 20px rgba(102, 126, 234, 0.15);
        }

        [data-testid="stSidebar"] * {
            color: white !important;
        }

        /* ===== HEADERS ===== */
        h1 {
            color: #667eea;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        h2 {
            color: #5a67d8;
            font-weight: 600;
        }

        h3 {
            color: #667eea;
            font-weight: 600;
        }

        /* ===== TEXT ===== */
        p, span, div {
            color: #4a5568;
            font-size: 15px;
            line-height: 1.6;
        }

        /* ===== CAPTION ===== */
        .stCaption {
            color: #718096;
            font-size: 13px;
            font-weight: 500;
        }

        /* ===== PRIMARY BUTTON ===== */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            height: 3.2em;
            width: 100%;
            font-size: 16px;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.35);
            transition: all 0.3s ease;
        }

        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5);
            background: linear-gradient(135deg, #5a67d8 0%, #6b46a0 100%);
        }

        /* ===== NORMAL BUTTON ===== */
        .main div.stButton > button:not([kind="primary"]) {
            background: #f7fafc;
            color: #2d3748;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .main div.stButton > button:not([kind="primary"]):hover {
            background: #edf2f7;
            border-color: #cbd5e0;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }

        /* ===== INPUT ===== */
        .main input[type="text"], 
        .main textarea {
            background: #f7fafc;
            color: #2d3748;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.75rem;
            transition: all 0.3s ease;
        }

        .main input[type="text"]:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
            background: #ffffff;
        }

        /* ===== IMAGE CARD ===== */
        .element-container img {
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15);
            transition: all 0.3s ease;
            border: 1px solid rgba(102, 126, 234, 0.08);
        }

        .element-container img:hover {
            transform: scale(1.03);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.25);
            border-color: rgba(102, 126, 234, 0.2);
        }

        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background: transparent;
            border-bottom: 2px solid #e2e8f0;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 17px;
            font-weight: 600;
            color: #718096;
            padding: 1rem 2rem;
            border-radius: 8px 8px 0 0;
            transition: all 0.3s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #667eea;
            background: rgba(102, 126, 234, 0.05);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border-radius: 8px 8px 0 0;
            box-shadow: 0 -2px 10px rgba(102, 126, 234, 0.3);
        }

        /* ===== INFO BOX ===== */
        .stAlert {
            border-radius: 12px;
            border-left: 4px solid #667eea;
            background: linear-gradient(90deg, rgba(102, 126, 234, 0.08) 0%, rgba(255, 255, 255, 0.5) 100%);
            color: #2d3748;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
        }

        /* ===== DOWNLOAD BUTTON ===== */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 3px 12px rgba(245, 87, 108, 0.3);
        }

        .stDownloadButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 18px rgba(245, 87, 108, 0.45);
            background: linear-gradient(135deg, #e881f5 0%, #f34560 100%);
        }

        /* ===== SPINNER ===== */
        .stSpinner > div {
            border-color: #667eea !important;
        }

        /* ===== SLIDER ===== */
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: #667eea;
        }

        .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] > div {
            background-color: #cbd5e0;
        }

        /* ===== FILE UPLOADER ===== */
        [data-testid="stFileUploader"] {
            background: #f7fafc;
            border: 2px dashed #cbd5e0;
            border-radius: 12px;
            padding: 1rem;
            transition: all 0.3s ease;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.05);
        }

        /* ===== EXPANDER ===== */
        .streamlit-expanderHeader {
            background: rgba(102, 126, 234, 0.08);
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .streamlit-expanderHeader:hover {
            background: rgba(102, 126, 234, 0.15);
        }

        </style>
        """, unsafe_allow_html=True)

    # Initialize Session State
    if 'favorites' not in st.session_state: 
        st.session_state.favorites = load_favorites()
    if 'search_query' not in st.session_state: 
        st.session_state.search_query = ""
    if 'trigger_search' not in st.session_state: 
        st.session_state.trigger_search = False
    if 'search_results' not in st.session_state: 
        st.session_state.search_results = None

    # Load AI Model & Database
    model, preprocess, device = load_clip_model()
    features, photo_ids_df, metadata_df = load_database()

    if features is None: 
        st.error("Cannot load database. Please check your data files.")
        st.stop()

    # Main Tabs
    tab_search, tab_fav = st.tabs(["AI Image Search", "My Favorites"])

    # ==================== TAB 1: SEARCH ====================
    with tab_search:
        # Sidebar Controls
        st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>CLIP AI Search</h2>", unsafe_allow_html=True)
        
        if st.sidebar.button("Home", use_container_width=True):
            reset_to_home()
            st.rerun()
        
        st.sidebar.markdown("---")
        
        search_type = st.sidebar.radio(
            "Search Mode:", 
            ["Text Search", "Image Search", "Hybrid Search (Image + Text)"],
            label_visibility="collapsed"
        )

        query_image = None
        hybrid_text = ""
        image_weight = 0.5
        
        # TEXT SEARCH MODE
        if search_type == "Text Search":
            user_input = st.sidebar.text_input(
                "Search for images...", 
                key="search_input_widget", 
                placeholder="E.g., sunset over ocean, cute puppy..."
            )
            st.session_state.search_query = user_input
            
            # Smart Suggestions
            if user_input.strip() != "":
                matches = [kw for kw in ALL_KEYWORDS if user_input.lower() in kw.lower()][:6]
                if matches:
                    st.sidebar.markdown("<p class='category-header'>Quick Suggestions</p>", unsafe_allow_html=True)
                    for sug in matches:
                        st.sidebar.button(
                            f"🔍 {sug}", 
                            key=f"side_{sug}", 
                            on_click=on_suggestion_click, 
                            args=(sug,), 
                            use_container_width=True
                        )
            
            # Category Suggestions
            # st.sidebar.markdown("<p class='category-header'>Browse Categories</p>", unsafe_allow_html=True)
            # for category, items in list(SUGGESTIONS_DB.items())[:3]:
            #     with st.sidebar.expander(category):
            #         for item in items[:5]:
            #             st.button(
            #                 item, 
            #                 key=f"cat_{item}", 
            #                 on_click=on_suggestion_click, 
            #                 args=(item,),
            #                 use_container_width=True
            #             )
        
        # IMAGE SEARCH MODE
        elif search_type == "Image Search":
            st.sidebar.markdown("<p class='category-header'>📤 Upload Image</p>", unsafe_allow_html=True)
            file = st.sidebar.file_uploader(
                "Choose an image...", 
                type=['jpg', 'jpeg', 'png'],
                label_visibility="collapsed"
            )
            if file:
                query_image = Image.open(file).convert("RGB")
                st.sidebar.image(query_image, caption="Query Image", use_container_width=True)
        
        # HYBRID SEARCH MODE
        else:
            st.sidebar.markdown("<p class='category-header'>Hybrid Search</p>", unsafe_allow_html=True)
            st.sidebar.info("Upload an image, then describe what's different")
            
            # Upload image
            file = st.sidebar.file_uploader(
                "Upload base image", 
                type=['jpg', 'jpeg', 'png'],
                key="hybrid_image"
            )
            if file:
                query_image = Image.open(file).convert("RGB")
                st.sidebar.image(query_image, caption="Base Image", use_container_width=True)
            
            # Text description
            hybrid_text = st.sidebar.text_area(
                "Describe the difference:",
                placeholder="E.g., 'but red color', 'wearing sunglasses', 'at night', 'with snow'...",
                height=100,
                key="hybrid_text"
            )
            
            # Weight slider
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Balance:**")
            image_weight = st.sidebar.slider(
                "Image ← → Text",
                0.0, 1.0, 0.5, 0.1,
                help="Slide left for more text influence, right for more image influence",
                label_visibility="collapsed"
            )
            
            col1, col2 = st.sidebar.columns(2)
            col1.caption(f"Image: {image_weight*100:.0f}%")
            col2.caption(f"Text: {(1-image_weight)*100:.0f}%")

        # Search Settings
        st.sidebar.markdown("---")
        st.sidebar.markdown("<p class='category-header'>Search Settings</p>", unsafe_allow_html=True)
        top_k = st.sidebar.slider("Results to show:", 6, 60, 12, 6)
        
        # SEARCH BUTTON
        search_btn = st.sidebar.button("SEARCH NOW", type="primary", use_container_width=True)

        # Execute Search
        if search_btn or st.session_state.trigger_search:
            st.session_state.trigger_search = False
            current_q = st.session_state.search_query
            
            with st.spinner("AI is analyzing..."):
                q_vec = None
                query_display_name = ""
                
                # TEXT SEARCH
                if search_type == "Text Search" and current_q: 
                    q_vec = encode_text_query(current_q, model, device)
                    query_display_name = current_q
                
                # IMAGE SEARCH
                elif search_type == "Image Search" and query_image: 
                    q_vec = encode_image_query(query_image, preprocess, model, device)
                    query_display_name = "Image Search"
                
                # HYBRID SEARCH
                elif search_type == "Hybrid Search (Image + Text)" and query_image and hybrid_text.strip():
                    q_vec = encode_hybrid_query(query_image, hybrid_text, preprocess, model, device, image_weight)
                    query_display_name = f"Hybrid: Image + '{hybrid_text}'"
                
                elif search_type == "Hybrid Search (Image + Text)":
                    st.warning("Please upload an image AND enter text description for hybrid search")
                
                if q_vec is not None:
                    indices, scores = search_engine(q_vec, features, top_k=top_k)
                    st.session_state.search_results = {
                        'indices': indices, 
                        'scores': scores, 
                        'query_name': query_display_name
                    }
                    st.rerun()

        # Display Search Results
        if st.session_state.search_results:
            res = st.session_state.search_results
            st.markdown(f"<h2 style='text-align: center; margin-bottom: 2rem;'>Results for: <span style='color: #667eea; font-size: inherit; font-weight: 600;'>{res['query_name']}</span></h2>", unsafe_allow_html=True)
            
            cols = st.columns(4)
            for i, (idx, score) in enumerate(zip(res['indices'], res['scores'])):
                pid = photo_ids_df.iloc[idx]['photo_id']
                meta = metadata_df[metadata_df['photo_id'] == pid].iloc[0]
                img_path = get_image_path(pid)
                
                with cols[i % 4]:
                    # Display Image
                    if img_path.exists(): 
                        st.image(str(img_path), use_container_width=True)
                    else: 
                        st.image(meta['photo_image_url'] + "?w=600", use_container_width=True)
                    
                    # Action Buttons
                    c1, c2 = st.columns(2)
                    is_fav = pid in st.session_state.favorites
                    
                    with c1:
                        st.button(
                            "❤️" if is_fav else "🤍", 
                            key=f"btn_res_{pid}", 
                            on_click=toggle_favorite, 
                            args=(pid,), 
                            use_container_width=True,
                            help="Add to favorites"
                        )
                    
                    with c2:
                        img_data = get_image_bytes(pid, metadata_df)
                        st.download_button(
                            label="Download", 
                            data=img_data, 
                            file_name=f"{pid}.jpg", 
                            mime="image/jpeg", 
                            key=f"dl_res_{pid}", 
                            use_container_width=True,
                            help="Download image"
                        )
                    
                    st.caption(f"Match: {score:.2%} • {meta['photographer_username']}")
        
        # Home Screen
        else:
            st.markdown("<h1 style='text-align: center; margin-bottom: 1rem;'>AI-Powered Image Search</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-size: 18px; color: #718096; margin-bottom: 3rem;'>Discover millions of images using cutting-edge AI technology</p>", unsafe_allow_html=True)
            
            
            st.markdown("<h3 style='margin-top: 3rem; margin-bottom: 1.5rem;'>Explore Our Collection</h3>", unsafe_allow_html=True)
            
            if 'home_samples' not in st.session_state: 
                st.session_state.home_samples = photo_ids_df.sample(min(12, len(photo_ids_df)))
            
            samples = st.session_state.home_samples
            cols = st.columns(4)
            
            for i, (_, row) in enumerate(samples.iterrows()):
                pid = row['photo_id']
                path = get_image_path(pid)
                
                with cols[i % 4]:
                    if path.exists(): 
                        st.image(str(path), use_container_width=True)
                    else: 
                        st.image(row['photo_image_url'] + "?w=600", use_container_width=True)
                    
                    c1, c2 = st.columns(2)
                    fav = pid in st.session_state.favorites
                    
                    with c1:
                        st.button(
                            "❤️" if fav else "🤍", 
                            key=f"home_fav_{pid}", 
                            on_click=toggle_favorite, 
                            args=(pid,), 
                            use_container_width=True
                        )
                    
                    with c2:
                        img_data = get_image_bytes(pid, metadata_df)
                        st.download_button(
                            label="Download", 
                            data=img_data, 
                            file_name=f"{pid}.jpg", 
                            mime="image/jpeg", 
                            key=f"dl_home_{pid}", 
                            use_container_width=True
                        )

    # ==================== TAB 2: FAVORITES ====================
    with tab_fav:
        fav_ids = st.session_state.favorites
        
        if not fav_ids:
            st.markdown("<div style='text-align: center; padding: 4rem 0;'>", unsafe_allow_html=True)
            st.markdown("<h2>💔 Your favorites collection is empty</h2>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 18px; color: #718096;'>Start exploring and save your favorite images!</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='text-align: center; margin-bottom: 2rem;'>❤️ Your Collection ({len(fav_ids)} images)</h2>", unsafe_allow_html=True)
            
            fav_meta_df = metadata_df[metadata_df['photo_id'].isin(fav_ids)]
            cols = st.columns(4)
            
            for i, (_, row) in enumerate(fav_meta_df.iterrows()):
                pid = row['photo_id']
                path = get_image_path(pid)
                
                with cols[i % 4]:
                    if path.exists(): 
                        st.image(str(path), use_container_width=True)
                    else: 
                        st.image(row['photo_image_url'] + "?w=600", use_container_width=True)
                    
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.button(
                            "Remove", 
                            key=f"remove_fav_{pid}", 
                            on_click=toggle_favorite, 
                            args=(pid,), 
                            use_container_width=True
                        )
                    
                    with c2:
                        img_data = get_image_bytes(pid, metadata_df)
                        st.download_button(
                            label="Download", 
                            data=img_data, 
                            file_name=f"{pid}.jpg", 
                            mime="image/jpeg", 
                            key=f"dl_fav_{pid}", 
                            use_container_width=True
                        )
                    
                    st.caption(f"{row['photographer_username']}")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align: center; padding: 1rem;'>"
        "<p style='font-size: 12px; opacity: 0.7;'>Powered by OpenAI CLIP</p>"
        "<p style='font-size: 12px; opacity: 0.7;'>© 2024 AI Image Search</p>"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()