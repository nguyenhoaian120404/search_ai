# app.py
import streamlit as st
import clip
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# --- CẤU HÌNH ---
# Lưu ý: Python string chứa backslash trên Windows nên dùng raw string (r"...")
DATA_PATH = Path("data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Visual Search AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Làm đẹp giao diện) ---
def local_css():
    st.markdown("""
        <style>
        /* Import Font Google */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Tiêu đề chính */
        .main-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .sub-title {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }

        /* Hiệu ứng ảnh kết quả */
        div[data-testid="stImage"] img {
            border-radius: 12px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        div[data-testid="stImage"] img:hover {
            transform: scale(1.03);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            cursor: pointer;
        }

        /* Style cho Button */
        div.stButton > button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-weight: 600;
        }
        div.stButton > button:hover {
            background-color: #FF2B2B;
            border-color: #FF2B2B;
        }
        
        /* Tab style */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- LOAD MODEL & DATA ---
@st.cache_resource
def load_model():
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    return model, preprocess

@st.cache_data
def load_data():
    try:
        features = np.load(DATA_PATH / "features.npy")
        photo_ids = pd.read_csv(DATA_PATH / "photo_ids.csv")
        metadata = pd.read_csv(DATA_PATH / "photos_metadata.csv")
        return features, photo_ids, metadata
    except FileNotFoundError:
        st.error(f"❌ Không tìm thấy dữ liệu tại: {DATA_PATH}. Hãy kiểm tra lại đường dẫn!")
        return None, None, None

# Load resources
with st.spinner("Đang khởi động AI Engine..."):
    model, preprocess = load_model()
    features, photo_ids_df, metadata = load_data()

# --- SIDEBAR (Cấu hình) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
    st.title("Settings")
    st.write("Cấu hình bộ tìm kiếm")
    
    top_k = st.slider("Số lượng kết quả:", min_value=1, max_value=20, value=8)
    
    st.divider()
    st.info(f"📊 Dataset: {len(photo_ids_df) if photo_ids_df is not None else 0} photos")
    st.info(f"🚀 Device: {DEVICE.upper()}")
    
    st.divider()
    st.markdown("Created by **Gemini Fan**")

# --- MAIN UI ---
st.markdown('<h1 class="main-title">Visual Search Engine</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Tìm kiếm hình ảnh thông minh với OpenAI CLIP</div>', unsafe_allow_html=True)

# --- LOGIC TÌM KIẾM ---
def search(query_features, dataset_features, k=5):
    similarity = (query_features @ dataset_features.T).squeeze(0)
    top_indices = similarity.argsort()[-k:][::-1]
    return top_indices, similarity[top_indices]

def display_results(indices, scores):
    st.markdown("### 🎯 Kết quả hàng đầu")
    
    # Hiển thị dạng lưới (Grid) thay vì cột đơn điệu
    cols = st.columns(4) # Chia làm 4 cột
    
    for i, idx in enumerate(indices):
        p_id = photo_ids_df.iloc[idx]['photo_id']
        info = metadata[metadata['photo_id'] == p_id].iloc[0]
        
        # Logic wrap cột (nếu i > 3 thì quay lại cột đầu)
        col = cols[i % 4]
        
        with col:
            # Container giúp gom nhóm phần tử UI
            with st.container():
                st.image(info['photo_image_url'] + "?w=400", use_container_width=True)
                
                # Metadata hiển thị gọn gàng
                st.markdown(f"**Score:** `{scores[i]:.4f}`")
                st.markdown(f"📸 [{info['photographer_first_name']}](https://unsplash.com/@{info['photographer_username']})")
                st.divider()

if features is not None:
    # Tạo Tab với icon đẹp
    tab1, tab2 = st.tabs(["📝 Text Search", "🖼️ Image Search"])

    # --- TAB 1: TEXT ---
    with tab1:
        st.write("Mô tả bức ảnh bạn đang nghĩ tới:")
        col_input, col_btn = st.columns([4, 1])
        
        with col_input:
            text_query = st.text_input("Ví dụ: A futuristic city at night", label_visibility="collapsed", placeholder="Nhập mô tả tiếng Anh...")
        
        with col_btn:
            search_btn = st.button("🔍 Tìm kiếm", key="btn_text")

        if search_btn and text_query:
            with st.spinner("AI đang đọc hiểu văn bản..."):
                text_tokenized = clip.tokenize([text_query]).to(DEVICE)
                with torch.no_grad():
                    query_feature = model.encode_text(text_tokenized)
                    query_feature /= query_feature.norm(dim=-1, keepdim=True)
                    query_feature = query_feature.cpu().numpy()
                
                indices, scores = search(query_feature, features, k=top_k)
                display_results(indices, scores)

    # --- TAB 2: IMAGE ---
    with tab2:
        col_up, col_preview = st.columns([1, 1])
        
        with col_up:
            uploaded_file = st.file_uploader("Upload ảnh để tìm tương tự", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                search_img_btn = st.button("🚀 Phân tích & Tìm kiếm", key="btn_img")

        with col_preview:
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Ảnh của bạn", width=250)
            else:
                st.info("👈 Hãy tải ảnh lên từ cột bên trái")

        if uploaded_file and search_img_btn:
            with st.spinner("AI đang nhìn ảnh của bạn..."):
                image_input = preprocess(image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    query_feature = model.encode_image(image_input)
                    query_feature /= query_feature.norm(dim=-1, keepdim=True)
                    query_feature = query_feature.cpu().numpy()
                
                indices, scores = search(query_feature, features, k=top_k)
                display_results(indices, scores)

else:
    st.warning("Dữ liệu chưa sẵn sàng. Hãy kiểm tra đường dẫn file.")