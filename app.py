# app.py
import streamlit as st
import clip
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# --- CẤU HÌNH ---
DATA_PATH = Path("data") # Thư mục chứa các file đã xuất ra
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="AI Image Search", layout="wide")
st.title("🔎 Hệ thống tìm kiếm ảnh đa phương thức (CLIP)")

# --- LOAD MODEL & DATA (Cache để không phải load lại mỗi lần click) ---
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
        st.error("Không tìm thấy file dữ liệu! Hãy chạy preprocess.py trước.")
        return None, None, None

model, preprocess = load_model()
features, photo_ids_df, metadata = load_data()

# --- HÀM TÌM KIẾM ---
def search(query_features, dataset_features, top_k=5):
    # Tính Cosine Similarity: (1, 512) x (N, 512).T -> (1, N)
    similarity = (query_features @ dataset_features.T).squeeze(0)
    
    # Lấy top K indices có điểm cao nhất
    top_indices = similarity.argsort()[-top_k:][::-1]
    return top_indices, similarity[top_indices]

def display_results(indices, scores):
    cols = st.columns(len(indices))
    for i, idx in enumerate(indices):
        # Lấy photo_id từ index
        p_id = photo_ids_df.iloc[idx]['photo_id']
        
        # Lấy thông tin metadata
        info = metadata[metadata['photo_id'] == p_id].iloc[0]
        
        with cols[i]:
            st.image(info['photo_image_url'] + "?w=400", use_container_width=True)
            st.caption(f"Score: {scores[i]:.4f}")
            st.markdown(f"**Photographer:** [{info['photographer_first_name']}](https://unsplash.com/@{info['photographer_username']})")

# --- GIAO DIỆN ---
if features is not None:
    tab1, tab2 = st.tabs(["📝 Text to Image", "🖼️ Image to Image"])

    # TAB 1: TÌM BẰNG TEXT
    with tab1:
        text_query = st.text_input("Nhập mô tả ảnh bạn muốn tìm (Tiếng Anh):", "A dog playing in the park")
        if st.button("Tìm kiếm", key="btn_text"):
            with st.spinner("Đang tìm..."):
                # Encode text
                text_tokenized = clip.tokenize([text_query]).to(DEVICE)
                with torch.no_grad():
                    query_feature = model.encode_text(text_tokenized)
                    query_feature /= query_feature.norm(dim=-1, keepdim=True)
                    query_feature = query_feature.cpu().numpy()
                
                # Search
                indices, scores = search(query_feature, features, top_k=5)
                display_results(indices, scores)

    # TAB 2: TÌM BẰNG ẢNH
    with tab2:
        uploaded_file = st.file_uploader("Tải lên một bức ảnh để tìm ảnh tương tự", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            # Hiển thị ảnh upload
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh gốc", width=300)
            
            if st.button("Tìm ảnh tương tự", key="btn_img"):
                with st.spinner("Đang phân tích..."):
                    # Encode image
                    image_input = preprocess(image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        query_feature = model.encode_image(image_input)
                        query_feature /= query_feature.norm(dim=-1, keepdim=True)
                        query_feature = query_feature.cpu().numpy()
                    
                    # Search
                    indices, scores = search(query_feature, features, top_k=5)
                    display_results(indices, scores)