import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import clip
import torch
from pathlib import Path
import os
import json

# Import suggestions (Giữ nguyên logic)
try:
    from suggestions_data import SUGGESTIONS_DB, ALL_KEYWORDS
except ImportError:
    SUGGESTIONS_DB = {"General": ["dog", "cat", "nature", "city"]}
    ALL_KEYWORDS = ["dog", "cat", "nature", "city"]

# ================= CONFIGURATION =================
DATA_DIR = Path("data") 
FAV_FILE = "favorites.json" 

# ================= AI & DATA HELPERS (Optimized) =================

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
        
        # --- FAST LOOKUP DICTIONARIES ---
        url_map = pd.Series(metadata_df.photo_image_url.values, index=metadata_df.photo_id).to_dict()
        username_map = pd.Series(metadata_df.photographer_username.values, index=metadata_df.photo_id).to_dict()
        
        return features, photo_ids_df, metadata_df, url_map, username_map
    except Exception as e:
        return None, None, None, None, None

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
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    text_token = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_token)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    text_weight = 1.0 - image_weight
    combined_features = image_weight * image_features + text_weight * text_features
    combined_features /= combined_features.norm(dim=-1, keepdim=True)
    return combined_features.cpu().numpy()

# --- HELPER: GET URL FAST ---
def get_image_url(photo_id, url_map):
    url = url_map.get(photo_id)
    if url:
        return url + "?w=600" # Optimized size for display
    return "https://via.placeholder.com/600x400?text=Image+Not+Found"

def get_original_url(photo_id, url_map):
    url = url_map.get(photo_id)
    return url if url else "#"

# ================= FAVORITES MANAGEMENT =================
def load_favorites():
    if os.path.exists(FAV_FILE):
        try:
            with open(FAV_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_favorites(fav_list):
    try:
        with open(FAV_FILE, "w") as f: json.dump(fav_list, f)
    except: pass

def toggle_favorite(photo_id):
    favs = st.session_state.favorites
    if photo_id in favs:
        favs.remove(photo_id)
        st.toast("Removed from favorites", icon="💔")
    else:
        favs.append(photo_id)
        st.toast("Added to favorites!", icon="❤️")
    st.session_state.favorites = favs
    save_favorites(favs)

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

# ================= MAIN UI =================
def main():
    st.set_page_config(page_title="Visual.AI - Search Engine", layout="wide", initial_sidebar_state="expanded")

    # CSS Styles (Clean & English-ready)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
        * { font-family: 'Plus Jakarta Sans', sans-serif; }
        .main { background: #F8FAFC; }
        
        /* Sidebar */
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #4F46E5 0%, #7C3AED 100%); }
        [data-testid="stSidebar"] * { color: white !important; }
        
        /* Link Button (acts as Download) */
        a[data-testid="stLinkButton"] {
            display: inline-flex;
            justify-content: center;
            align-items: center;
            font-weight: 600;
            padding: 0.25rem 0.75rem;
            border-radius: 8px;
            min-height: 2.5rem;
            color: white;
            width: 100%;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            text-decoration: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: none;
            transition: transform 0.2s;
        }
        a[data-testid="stLinkButton"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        /* Custom Buttons */
        div.stButton.home-btn > button { background: rgba(255, 255, 255, 0.15) !important; border: 1px solid rgba(255, 255, 255, 0.3) !important; color: white !important; }
        .upload-container { background: rgba(255, 255, 255, 0.1); border: 2px dashed rgba(255, 255, 255, 0.4); border-radius: 12px; padding: 15px; text-align: center; }
        div.stButton.search-btn > button { background: #FFFFFF !important; color: #4F46E5 !important; border: none; font-weight: 800; text-transform: uppercase; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
        
        /* Image Cards */
        div[data-testid="stImage"] img { border-radius: 12px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08); }
        </style>
    """, unsafe_allow_html=True)

    # Init State
    if 'favorites' not in st.session_state: st.session_state.favorites = load_favorites()
    if 'search_query' not in st.session_state: st.session_state.search_query = ""
    if 'trigger_search' not in st.session_state: st.session_state.trigger_search = False
    if 'search_results' not in st.session_state: st.session_state.search_results = None

    # Load Data
    with st.spinner("🚀 Starting AI Engine..."):
        model, preprocess, device = load_clip_model()
        features, photo_ids_df, metadata_df, url_map, username_map = load_database()

    if features is None: 
        st.error("❌ Data not found. Please check your 'data' folder.")
        st.stop()

    tab_search, tab_fav = st.tabs(["🔍 Search Engine", "❤️ Favorites"])

    # --- TAB 1: SEARCH ---
    with tab_search:
        with st.sidebar:
            st.markdown("## 🎨 Visual.AI")
            st.markdown('<div class="stButton home-btn">', unsafe_allow_html=True)
            if st.button("🏠  Home / Reset", use_container_width=True):
                reset_to_home()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.divider()

            st.markdown("### Search Mode")
            search_type = st.radio("Mode", ["📝 Text Search", "🖼️ Image Search", "🧬 Hybrid Search"], label_visibility="collapsed")
            
            query_image = None
            hybrid_text = ""
            image_weight = 0.5
            
            st.markdown("---")

            # MODE: TEXT
            if search_type == "📝 Text Search":
                st.markdown("### Keywords")
                user_input = st.text_input("Keyword", key="search_input_widget", placeholder="e.g., A futuristic city at night...", label_visibility="collapsed")
                st.session_state.search_query = user_input
                
                if user_input.strip() != "":
                    matches = [kw for kw in ALL_KEYWORDS if user_input.lower() in kw.lower()][:4]
                    if matches:
                        st.caption("✨ Suggestions:")
                        cols = st.columns(2)
                        for i, sug in enumerate(matches):
                            if cols[i%2].button(sug, key=f"sug_{i}", use_container_width=True):
                                on_suggestion_click(sug)
            
            # MODE: IMAGE
            elif search_type == "🖼️ Image Search":
                st.markdown("### Upload Image")
                st.markdown('<div class="upload-container">', unsafe_allow_html=True)
                file = st.file_uploader("Upload", type=['jpg', 'png'], label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
                if file:
                    query_image = Image.open(file).convert("RGB")
                    st.image(query_image, caption="✅ Image Selected", use_container_width=True)

            # MODE: HYBRID
            else:
                st.markdown("### 1. Base Image")
                st.markdown('<div class="upload-container">', unsafe_allow_html=True)
                file = st.file_uploader("Base", type=['jpg', 'png'], key="hybrid_up", label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
                if file:
                    query_image = Image.open(file).convert("RGB")
                    st.image(query_image, width=150)
                
                st.markdown("### 2. Modification")
                hybrid_text = st.text_area("Diff", placeholder="e.g., ...but in winter / red color", height=68, label_visibility="collapsed")
                
                st.markdown("### 3. Balance")
                image_weight = st.slider("Image vs Text", 0.0, 1.0, 0.5, label_visibility="collapsed")
                c1, c2 = st.columns(2)
                c1.caption(f"Img: {int(image_weight*100)}%")
                c2.caption(f"Txt: {int((1-image_weight)*100)}%")

            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            st.markdown("### Results Count")
            top_k = st.slider("Top K", 4, 40, 12, label_visibility="collapsed")
            
            st.divider()
            st.markdown('<div class="stButton search-btn">', unsafe_allow_html=True)
            search_btn = st.button("🚀 SEARCH NOW", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Logic Search
        if search_btn or st.session_state.trigger_search:
            st.session_state.trigger_search = False
            current_q = st.session_state.search_query
            
            with st.spinner("🧠 AI is analyzing..."):
                q_vec = None
                query_name = ""
                
                if search_type == "📝 Text Search" and current_q: 
                    q_vec = encode_text_query(current_q, model, device)
                    query_name = f'"{current_q}"'
                
                elif search_type == "🖼️ Image Search" and query_image: 
                    q_vec = encode_image_query(query_image, preprocess, model, device)
                    query_name = "Uploaded Image"
                
                elif search_type == "🧬 Hybrid Search" and query_image and hybrid_text:
                    q_vec = encode_hybrid_query(query_image, hybrid_text, preprocess, model, device, image_weight)
                    query_name = f"Image + '{hybrid_text}'"
                
                else:
                    st.warning("Please provide all required inputs!")

                if q_vec is not None:
                    indices, scores = search_engine(q_vec, features, top_k=top_k)
                    st.session_state.search_results = {'indices': indices, 'scores': scores, 'query_name': query_name}
                    st.rerun()

        # Display Results
        if st.session_state.search_results:
            res = st.session_state.search_results
            st.markdown(f"### 🎯 Results for: {res['query_name']}")
            st.markdown("---")
            
            cols = st.columns(4)
            for i, (idx, score) in enumerate(zip(res['indices'], res['scores'])):
                pid = photo_ids_df.iloc[idx]['photo_id']
                
                # Fast Lookup
                image_url = get_image_url(pid, url_map)
                photographer = username_map.get(pid, "Unknown")
                original_url = get_original_url(pid, url_map)
                
                with cols[i % 4]:
                    st.image(image_url, use_container_width=True)
                    
                    c1, c2 = st.columns([1, 2])
                    is_fav = pid in st.session_state.favorites
                    with c1:
                        if st.button("❤️" if is_fav else "🤍", key=f"fav_{pid}"):
                            toggle_favorite(pid)
                            st.rerun()
                    with c2:
                        st.link_button("Download", original_url, use_container_width=True)
                    
                    st.caption(f"**{score:.0%}** • {photographer}")
        
        # Home Screen
        else:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("<h1 style='text-align: center; color: #1F2937;'>✨ Explore Infinite Images</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #6B7280; font-size: 1.1rem;'>Semantic search powered by OpenAI CLIP</p>", unsafe_allow_html=True)
            
            st.markdown("### 🌟 Today's Picks")
            if 'home_samples' not in st.session_state: 
                st.session_state.home_samples = photo_ids_df.sample(min(8, len(photo_ids_df)))['photo_id'].tolist()
            
            samples = st.session_state.home_samples
            cols = st.columns(4)
            for i, pid in enumerate(samples):
                image_url = get_image_url(pid, url_map)
                with cols[i % 4]:
                    st.image(image_url, use_container_width=True)

    # --- TAB 2: FAVORITES ---
    with tab_fav:
        fav_ids = st.session_state.favorites
        if not fav_ids:
            st.info("No favorites saved yet.")
        else:
            st.success(f"Saved {len(fav_ids)} images.")
            cols = st.columns(4)
            for i, pid in enumerate(fav_ids):
                image_url = get_image_url(pid, url_map)
                original_url = get_original_url(pid, url_map)
                
                with cols[i % 4]:
                    st.image(image_url, use_container_width=True)
                    c1, c2 = st.columns([1,2])
                    with c1:
                        if st.button("Remove", key=f"del_{pid}"):
                            toggle_favorite(pid)
                            st.rerun()
                    with c2:
                        st.link_button("Download", original_url, use_container_width=True)

if __name__ == "__main__":
    main()