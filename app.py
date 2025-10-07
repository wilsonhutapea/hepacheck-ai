import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import pickle
from pathlib import Path
from utils.eye_detection_module_v3 import multi_stage_detection

# ============================================================================
# LOAD KNN MODEL
# ============================================================================
@st.cache_resource
def load_knn_model():
    """Load KNN jaundice detection model"""
    model_path = Path(__file__).parent / "knn_jaundice_model_v5.pkl"
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading KNN model: {e}")
        return None

@st.cache_resource
def load_knn_urine_model():
    """Load KNN urine jaundice detection model"""
    model_path = Path(__file__).parent / "knn_urine_jaundice_model.pkl"
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading KNN Urine model: {e}")
        return None

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="Deteksi Jaundice",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PERTANYAAN UNTUK USER
# ============================================================================
QUESTIONS = [
    {
        "question": "Demam sebelum gejala hati muncul?",
        "options": ["Ya", "Tidak"],
        "scores": [2, 0]
    },
    {
        "question": "Nyeri otot (myalgia)?",
        "options": ["Sering/Sangat sering", "Jarang/Kadang", "Tidak pernah"],
        "scores": [2, 1, 0]
    },
    {
        "question": "Mual atau muntah?",
        "options": ["Sering/Sangat sering", "Jarang/Kadang", "Tidak pernah"],
        "scores": [2, 1, 0]
    },
    {
        "question": "Nyeri perut kanan atas?",
        "options": ["Nyeri berat/sangat parah", "Nyeri sedang", "Tidak/nyeri ringan"],
        "scores": [2, 1, 0]
    },
    {
        "question": "Feses pucat/abu-abu?",
        "options": ["Ya", "Tidak"],
        "scores": [2, 0]
    },
    {
        "question": "Lama gejala berlangsung:",
        "options": ["< 6 minggu", "6 minggu – 6 bulan", "> 6 bulan"],
        "scores": [0, 1, 3]
    },
    {
        "question": "Fatigue / lemas kronis?",
        "options": ["Sering/Sangat sering", "Jarang/Kadang", "Tidak"],
        "scores": [2, 1, 0]
    },
    {
        "question": "Gatal pada kulit?",
        "options": ["Sering/Sangat sering", "Jarang/Kadang", "Tidak"],
        "scores": [2, 1, 0]
    },
    {
        "question": "Perut membesar (ascites)?",
        "options": ["Ya", "Tidak"],
        "scores": [3, 0]
    },
    {
        "question": "Mudah memar/pendarahan?",
        "options": ["Ya", "Tidak"],
        "scores": [3, 0]
    },
    {
        "question": "Kaki bengkak (edema)?",
        "options": ["Ya", "Tidak"],
        "scores": [2, 0]
    }
]
MAX_SYMPTOM_SCORE = 25 # Sum of max scores: 2+2+2+2+2+3+2+2+3+3+2 = 25

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def convert_uploaded_image(uploaded_file):
    """Convert uploaded file to OpenCV format"""
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image_array = np.array(image)
    
    # Handle different image formats
    if len(image_array.shape) == 2:  # Grayscale
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    elif image_array.shape[2] == 4:  # RGBA
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    return image_array, image

def extract_sclera_rgb_from_detection(eye_image, eye_mask):
    """
    Extract median RGB values from detected sclera region
    
    Args:
        eye_image: numpy array (BGR format) - detected eye region
        eye_mask: binary mask of sclera region
    
    Returns:
        tuple: (R, G, B) median values or None if extraction fails
    """
    if eye_image is None or eye_mask is None:
        return None
    
    if np.sum(eye_mask) == 0:
        return None
    
    # Convert BGR to RGB
    eye_rgb = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
    
    # Extract sclera pixels
    sclera_pixels = eye_rgb[eye_mask > 0]
    
    if len(sclera_pixels) == 0:
        return None
    
    # Calculate median RGB
    median_rgb = np.median(sclera_pixels, axis=0)
    r, g, b = median_rgb
    
    return (int(r), int(g), int(b))

def classify_jaundice_knn(rgb_values, knn_model):
    """
    Classify jaundice using KNN model
    
    Args:
        rgb_values: tuple (R, G, B)
        knn_model: loaded KNN model
    
    Returns:
        dict with prediction results or None
    """
    if rgb_values is None or knn_model is None:
        return None
    
    try:
        # Prepare input for model
        input_data = np.array([list(rgb_values)])
        
        # Get prediction
        prediction = knn_model.predict(input_data)[0]
        
        # Get prediction probabilities
        probabilities = knn_model.predict_proba(input_data)[0]
        
        # Class 0 = Normal, Class 2 = Jaundice
        # The model was trained with classes 0 and 2
        classes = knn_model.classes_
        
        # Find confidence for the predicted class
        pred_idx = np.where(classes == prediction)[0][0]
        confidence = probabilities[pred_idx] * 100
        
        # Determine status
        is_jaundice = (prediction == 2)
        status = "Jaundice Terdeteksi" if is_jaundice else "Normal"
        
        return {
            'prediction': int(prediction),
            'is_jaundice': is_jaundice,
            'status': status,
            'confidence': confidence,
            'probabilities': probabilities,
            'classes': classes
        }
    
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None

def extract_urine_rgb(image):
    """
    Extract median RGB values from urine region in the image using HSV masking.
    This method is aligned with the dataset creation script.
    
    Args:
        image: numpy array (BGR format)
    
    Returns:
        tuple: ((R, G, B), mask) or (None, None) if processing fails
    """
    try:
        if image is None:
            return None, None
        
        # Convert to RGB (for final output) and HSV (for masking)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        h_img, w_img = image.shape[:2]
        
        # Strategy 1: Detect urine region using color and brightness masks
        hue_mask = cv2.inRange(h, 0, 45)
        saturation_mask = cv2.inRange(s, 40, 255)
        brightness_mask = cv2.inRange(v, 40, 245)
        
        temp_mask = cv2.bitwise_and(hue_mask, saturation_mask)
        urine_mask = cv2.bitwise_and(temp_mask, brightness_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        urine_mask = cv2.morphologyEx(urine_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        urine_mask = cv2.morphologyEx(urine_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        urine_pixels = image_rgb[urine_mask > 0]
        final_mask = urine_mask
        
        # Strategy 2: Fallback to central region if mask is insufficient
        if len(urine_pixels) < 500:
            y_start = int(h_img * 0.2)
            y_end = int(h_img * 0.8)
            x_start = int(w_img * 0.2)
            x_end = int(w_img * 0.8)
            
            center_region_rgb = image_rgb[y_start:y_end, x_start:x_end]
            
            center_hsv = cv2.cvtColor(center_region_rgb, cv2.COLOR_RGB2HSV)
            center_h, center_s, center_v = cv2.split(center_hsv)
            
            center_hue_mask = cv2.inRange(center_h, 0, 45)
            center_saturation_mask = cv2.inRange(center_s, 40, 255)
            center_brightness_mask = cv2.inRange(center_v, 40, 245)
            
            temp_mask = cv2.bitwise_and(center_hue_mask, center_saturation_mask)
            center_mask = cv2.bitwise_and(temp_mask, center_brightness_mask)
            
            urine_pixels = center_region_rgb.reshape(-1, 3)[center_mask.reshape(-1) > 0]
            
            final_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            final_mask[y_start:y_end, x_start:x_end] = center_mask

            if len(urine_pixels) < 100:
                urine_pixels = center_region_rgb.reshape(-1, 3)
                final_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                final_mask[y_start:y_end, x_start:x_end] = 255
        
        if len(urine_pixels) == 0:
            return None, None
        
        median_rgb = np.median(urine_pixels, axis=0)
        red, green, blue = median_rgb
        
        return (int(red), int(green), int(blue)), final_mask
    
    except Exception as e:
        st.error(f"Error during urine color extraction: {e}")
        return None, None

def classify_urine_jaundice_knn(rgb_values, knn_model):
    """
    Classify urine jaundice using KNN model
    
    Args:
        rgb_values: tuple (R, G, B)
        knn_model: loaded KNN model for urine
    
    Returns:
        dict with prediction results or None
    """
    if rgb_values is None or knn_model is None:
        return None
    
    try:
        input_data = np.array([list(rgb_values)])
        
        prediction = knn_model.predict(input_data)[0]
        probabilities = knn_model.predict_proba(input_data)[0]
        classes = knn_model.classes_
        
        pred_idx = np.where(classes == prediction)[0][0]
        confidence = probabilities[pred_idx] * 100
        
        # Class 2 = Jaundice, Class 0 = Normal
        is_jaundice = (prediction == 2)
        status = "Jaundice Terdeteksi" if is_jaundice else "Normal"
        
        return {
            'prediction': int(prediction),
            'is_jaundice': is_jaundice,
            'status': status,
            'confidence': confidence,
            'probabilities': probabilities,
            'classes': classes
        }
    except Exception as e:
        st.error(f"Error during urine classification: {e}")
        return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    if 'photos' not in st.session_state:
        st.session_state.photos = {}

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Initialize
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🔬 Sistem Deteksi Hepatitis</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Progress bar
    total_steps = 13  # 11 questions + 2 photos (eye + urine)
    progress = (st.session_state.step - 1) / total_steps
    st.progress(progress)
    st.caption(f"Progress: {st.session_state.step}/{total_steps}")
    
    # ========================================================================
    # BAGIAN 1: PERTANYAAN (Step 1-11)
    # ========================================================================
    if 1 <= st.session_state.step <= 11:
        st.header(f"📋 Pertanyaan {st.session_state.step} dari 11")
        
        question_data = QUESTIONS[st.session_state.step - 1]
        st.subheader(question_data["question"])
        
        st.write("")  # Spacing
        
        options = question_data["options"]
        scores = question_data["scores"]
        
        # Display options as buttons in a centered column
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            # Create a new inner column layout for the buttons
            num_options = len(options)
            if num_options > 0:
                cols = st.columns(num_options)
                for i, option in enumerate(options):
                    with cols[i]:
                        if st.button(option, width='stretch', key=f"q{st.session_state.step}_{i}"):
                            st.session_state.answers[f'q{st.session_state.step}'] = {
                                'answer': option,
                                'score': scores[i]
                            }
                            st.session_state.step += 1
                            st.rerun()

        # Show previous answers
        if st.session_state.step > 1:
            st.markdown("---")
            with st.expander("📊 Lihat Jawaban Sebelumnya"):
                for i in range(1, st.session_state.step):
                    answer_data = st.session_state.answers.get(f'q{i}')
                    if answer_data:
                        answer_text = answer_data['answer']
                        question_text = QUESTIONS[i-1]['question']
                        st.write(f"**Pertanyaan {i}:** {question_text} → **{answer_text}**")
    
    # ========================================================================
    # BAGIAN 2: UPLOAD FOTO MATA (Step 12)
    # ========================================================================
    elif st.session_state.step == 12:
        st.header("📸 Upload Foto Mata")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("""
            **Panduan Foto Mata:**
            - Gunakan pencahayaan yang baik dan merata
            - Foto bisa berupa full face atau close-up mata
            - Pastikan mata terlihat jelas (sclera/bagian putih mata)
            - Mata terbuka lebar
            - Hindari bayangan pada area mata
            - Format: JPG, JPEG, atau PNG
            """)
            
            uploaded_file = st.file_uploader(
                "Pilih foto mata (kiri atau kanan)",
                type=['jpg', 'jpeg', 'png'],
                key='eye_uploader'
            )
            
            if uploaded_file is not None:
                # Convert and store
                image_array, preview_image = convert_uploaded_image(uploaded_file)
                st.session_state.photos['eye'] = image_array
                
                # Preview
                st.success("✅ Foto mata berhasil diupload!")
                st.image(preview_image, caption="Preview Foto Mata", width='stretch')
                
                if st.button("➡️ Lanjut ke Upload Foto Urine", type="primary", width='stretch'):
                    st.session_state.step = 13
                    st.rerun()
        
        with col2:
            st.write("**Contoh Foto yang Baik:**")
            st.image("https://github.com/wilsonhutapea/hepacheck-ai/blob/main/assets/images/fadey%20contoh_compressed.jpg?raw=true", 
                    caption="Foto dengan pencahayaan baik", width=300)
    
    # ========================================================================
    # BAGIAN 3: UPLOAD FOTO URINE (Step 13)
    # ========================================================================
    elif st.session_state.step == 13:
        st.header("🧪 Upload Foto Urine")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("""
            **Panduan Foto Urine:**
            - Gunakan wadah transparan/bening (gelas/cup)
            - Foto dari samping dengan latar belakang putih
            - Pencahayaan natural/terang (hindari cahaya kuning)
            - Fokus pada warna urine
            - Jangan gunakan flash langsung
            - Format: JPG, JPEG, atau PNG
            """)
            
            uploaded_file = st.file_uploader(
                "Pilih foto urine",
                type=['jpg', 'jpeg', 'png'],
                key='urine_uploader'
            )
            
            if uploaded_file is not None:
                # Convert and store
                image_array, preview_image = convert_uploaded_image(uploaded_file)
                st.session_state.photos['urine'] = image_array
                
                # Preview
                st.success("✅ Foto urine berhasil diupload!")
                st.image(preview_image, caption="Preview Foto Urine", width='stretch')
                
                if st.button("🔬 Analisis Sekarang", type="primary", width='stretch'):
                    st.session_state.step = 14
                    st.rerun()
        
        with col2:
            st.write("**Contoh Foto yang Baik:**")
            st.image("https://github.com/wilsonhutapea/hepacheck-ai/blob/main/assets/images/Urine%20Normal.jpg?raw=true", 
                    caption="Urine dalam gelas bening", width=300)
    
    # ========================================================================
    # BAGIAN 4: PROSES & HASIL (Step 14)
    # ========================================================================
    elif st.session_state.step == 14:
        st.header("🔬 Hasil Analisis")
        
        # Load KNN models
        knn_model = load_knn_model()
        knn_urine_model = load_knn_urine_model()
        
        eye_image = st.session_state.photos.get('eye')
        urine_image = st.session_state.photos.get('urine')
        
        knn_result = None
        urine_result = None

        col1, col2 = st.columns(2)
        
        # ====================================================================
        # ANALISIS MATA MENGGUNAKAN KNN MODEL
        # ====================================================================
        with col1:
            st.subheader("👁️ Analisis Mata (KNN Model)")
            
            if eye_image is not None:
                with st.spinner("🔍 Step 1: Mendeteksi mata..."):
                    # Step 1: Eye Detection using eye_detection_module_v3
                    eye_result = multi_stage_detection(eye_image)
                
                if eye_result and eye_result.get('left_eye') is not None:
                    # Display detected eye
                    st.image(cv2.cvtColor(eye_result['left_eye'], cv2.COLOR_BGR2RGB), 
                            caption="✅ Mata Terdeteksi", width='stretch')
                    
                    with st.spinner("🔍 Step 2: Mengekstrak warna RGB sclera..."):
                        # Step 2: Extract RGB from sclera
                        rgb_values = extract_sclera_rgb_from_detection(
                            eye_result['left_eye'],
                            eye_result['left_mask']
                        )
                    
                    if rgb_values and knn_model:
                        r, g, b = rgb_values
                        
                        with st.spinner("🤖 Step 3: Klasifikasi dengan KNN Model..."):
                            # Step 3: Classify with KNN Model
                            knn_result = classify_jaundice_knn(rgb_values, knn_model)
                        
                        if knn_result:
                            # Display KNN prediction results
                            eye_jaundice = knn_result['is_jaundice']
                            eye_conf = knn_result['confidence']
                            
                            status_icon = "⚠️" if eye_jaundice else "✅"
                            st.metric("Status KNN", f"{status_icon} {knn_result['status']}")
                            st.metric("Confidence", f"{eye_conf:.1f}%")
                            st.metric("Prediksi Kelas", f"Class {knn_result['prediction']}")
                            
                            # Display masked sclera
                            if eye_result.get('left_mask') is not None:
                                eye_masked = eye_result['left_eye'].copy()
                                eye_masked[eye_result['left_mask'] == 0] = 0
                                with st.expander("👁️ Lihat Area Sclera"):
                                    st.image(cv2.cvtColor(eye_masked, cv2.COLOR_BGR2RGB), 
                                           caption="Sclera (Bagian Putih Mata)", width='stretch')
                            
                            # RGB details
                            with st.expander("🎨 Detail Warna Sclera (Input KNN)"):
                                st.write(f"**RGB (Median):** R={r}, G={g}, B={b}")
                                st.write(f"**Model:** K-Nearest Neighbors (k=3)")
                                st.write(f"**Classes:** {knn_result['classes']}")
                                
                                # Show probabilities
                                st.write("**Probabilitas:**")
                                for cls, prob in zip(knn_result['classes'], knn_result['probabilities']):
                                    cls_name = "Normal" if cls == 0 else "Jaundice"
                                    st.write(f"  - Class {cls} ({cls_name}): {prob*100:.1f}%")
                        else:
                            st.error("❌ Gagal melakukan klasifikasi KNN")
                            eye_jaundice, eye_conf = False, 0
                    else:
                        st.warning("⚠️ Tidak dapat mengekstrak RGB sclera atau model tidak tersedia")
                        eye_jaundice, eye_conf = False, 0
                else:
                    st.error("❌ Mata tidak terdeteksi pada foto")
                    eye_jaundice, eye_conf = False, 0
            else:
                st.error("❌ Foto mata tidak ditemukan")
                eye_jaundice, eye_conf = False, 0
        
        # ====================================================================
        # ANALISIS URINE
        # ====================================================================
        with col2:
            st.subheader("🧪 Analisis Urine")
            
            if urine_image is not None:
                with st.spinner("🔍 Menganalisis urine dengan KNN..."):
                    # Step 1: Extract RGB and mask
                    urine_rgb, urine_mask = extract_urine_rgb(urine_image)
                    
                    # Step 2: Classify with KNN
                    urine_result = classify_urine_jaundice_knn(urine_rgb, knn_urine_model)

                st.image(cv2.cvtColor(urine_image, cv2.COLOR_BGR2RGB), 
                        caption="Foto Urine", width='stretch')
                
                if urine_result:
                    urine_jaundice = urine_result['is_jaundice']
                    print(f"urine jaundice = {urine_jaundice}") # TODO HAPUS
                    urine_conf = urine_result['confidence']
                    
                    status_icon = "⚠️" if urine_jaundice else "✅"
                    st.metric("Status KNN", f"{status_icon} {urine_result['status']}")
                    st.metric("Confidence", f"{urine_conf:.1f}%")
                    st.metric("Prediksi Kelas", f"Class {urine_result['prediction']}")

                    # Add expander for masked urine
                    if urine_mask is not None:
                        urine_masked = urine_image.copy()
                        urine_masked[urine_mask == 0] = 0
                        with st.expander("🔬 Lihat Area Urine Terdeteksi"):
                            st.image(cv2.cvtColor(urine_masked, cv2.COLOR_BGR2RGB), 
                                   caption="Area Urine yang Dianalisis", width='stretch')

                    with st.expander("🎨 Detail Warna Urine (Input KNN)"):
                        r, g, b = urine_rgb
                        st.write(f"**RGB (Median):** R={r}, G={g}, B={b}")
                        st.write(f"**Model:** K-Nearest Neighbors")
                        st.write(f"**Classes:** {urine_result['classes']}")
                        
                        st.write("**Probabilitas:**")
                        for cls, prob in zip(urine_result['classes'], urine_result['probabilities']):
                            cls_name = "Normal" if cls == 0 else "Jaundice"
                            st.write(f"  - Class {cls} ({cls_name}): {prob*100:.1f}%")
                
                else:
                    st.error("❌ Gagal melakukan klasifikasi urine")
                    urine_jaundice, urine_conf = False, 0
            else:
                st.error("❌ Foto urine tidak ditemukan")
                urine_jaundice, urine_conf = False, 0
        
        # ====================================================================
        # KESIMPULAN AKHIR
        # ====================================================================
        st.markdown("---")
        st.header("📋 Kesimpulan Analisis")
        
        # Calculate symptom score
        symptom_score = sum(v['score'] for v in st.session_state.answers.values())
        
        # Calculate jaundice class score from models
        jaundice_class_score = 0
        if knn_result and knn_result.get('prediction') == 2:
            jaundice_class_score += 3
        if urine_result and urine_result.get('prediction') == 2:
            jaundice_class_score += 3

        # Combined detection
        visual_detection = eye_jaundice or urine_jaundice
        avg_confidence = (eye_conf + urine_conf) / 2
        
        # Calculate total risk score based on simple points division
        total_achieved_score = symptom_score + jaundice_class_score
        total_possible_score = MAX_SYMPTOM_SCORE + 6 # 25 symptoms + 6 from models
            
        # Calculate final risk score as a percentage
        if total_possible_score > 0:
            risk_score = (total_achieved_score / total_possible_score) * 100
        else:
            risk_score = 0
        
        # Determine risk level
        if risk_score >= 60:
            risk_level = "TINGGI"
            risk_color = "risk-high"
            risk_icon = "🔴"
        elif risk_score >= 35:
            risk_level = "SEDANG"
            risk_color = "risk-medium"
            risk_icon = "🟡"
        else:
            risk_level = "RENDAH"
            risk_color = "risk-low"
            risk_icon = "🟢"
        
        # Display risk assessment
        st.markdown(f'<div class="{risk_color}">', unsafe_allow_html=True)
        st.markdown(f"## {risk_icon} RISIKO {risk_level} HEPATITIS")
        st.markdown(f"**Skor Risiko Total:** {risk_score:.0f}%")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.write("")
        
        # Detailed breakdown
        col_detail1, col_detail2, col_detail3 = st.columns(3)
        
        with col_detail1:
            st.metric("Gejala Klinis & Model", f"{total_achieved_score}/{total_possible_score}", 
                     delta=f"{risk_score:.0f}% dari total skor" if total_achieved_score > 0 else "Tidak ada gejala")
        
        with col_detail2:
            eye_status = "⚠️ Abnormal" if eye_jaundice else "✅ Normal"
            st.metric("Status Mata", eye_status, 
                     delta=f"{eye_conf:.0f}% conf" if eye_jaundice else None)
        
        with col_detail3:
            urine_status = "⚠️ Abnormal" if urine_jaundice else "✅ Normal"
            st.metric("Status Urine", urine_status, 
                     delta=f"{urine_conf:.0f}% conf" if urine_jaundice else None)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Rekomendasi")
        
        if risk_level == "TINGGI":
            st.error("""
            **🚨 TINDAKAN SEGERA DIPERLUKAN:**
            
            1. **Segera konsultasi dengan dokter** dalam 24-48 jam
            2. Lakukan pemeriksaan laboratorium:
               - Tes fungsi hati (SGOT, SGPT, Alkaline Phosphatase)
               - Bilirubin total dan direk
               - Tes hepatitis (A, B, C)
            3. Hindari konsumsi alkohol
            4. Istirahat cukup
            5. Catat semua gejala yang dialami
            """)
        elif risk_level == "SEDANG":
            st.warning("""
            **⚠️ PEMERIKSAAN DISARANKAN:**
            
            1. **Jadwalkan pemeriksaan dokter** dalam 1-2 minggu
            2. Monitor gejala yang ada
            3. Perhatikan perubahan warna mata dan urine
            4. Jaga pola makan sehat
            5. Hindari konsumsi obat-obatan tanpa resep
            6. Istirahat cukup
            """)
        else:
            st.success("""
            **✅ RISIKO RENDAH - TETAP JAGA KESEHATAN:**
            
            1. Tidak ada indikasi Hepatitis yang signifikan
            2. Tetap monitor kesehatan secara berkala
            3. Jaga pola hidup sehat:
               - Diet seimbang
               - Olahraga teratur
               - Hindari alkohol berlebihan
               - Cukup istirahat
            4. Jika gejala muncul, segera konsultasi dokter
            """)
        
        # Detail gejala
        st.markdown("---")
        with st.expander("📊 Lihat Detail Semua Gejala"):
            for i in range(len(QUESTIONS)):
                question_index = i + 1
                answer_data = st.session_state.answers.get(f'q{question_index}')
                question_text = QUESTIONS[i]['question']

                if answer_data:
                    answer_text = answer_data['answer']
                    score = answer_data['score']
                    st.write(f"**{question_text}** → {answer_text} *(Skor: {score})*")
                else:
                    st.write(f"**{question_text}** → (Belum dijawab)")
        
        # Disclaimer
        st.markdown("---")
        st.info("""
        **⚠️ DISCLAIMER PENTING:**
        
        Hasil analisis ini **BUKAN** merupakan diagnosis medis final dan tidak menggantikan 
        pemeriksaan oleh tenaga medis profesional.
        
        Aplikasi ini hanya sebagai **alat bantu skrining awal** untuk membantu deteksi dini 
        potensi Hepatitis. Untuk diagnosis yang akurat dan penanganan yang tepat, 
        **WAJIB konsultasi dengan dokter** dan melakukan pemeriksaan laboratorium lengkap.
        
        Jangan menggunakan hasil ini sebagai dasar pengobatan mandiri.
        """)
        
        # Action buttons
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🔄 Analisis Baru", width='stretch', type="primary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col_btn2:
            if st.button("📄 Export Hasil (Coming Soon)", width='stretch', disabled=True):
                st.info("Fitur export akan segera hadir!")
        
        with col_btn3:
            if st.button("ℹ️ Info Lengkap", width='stretch'):
                st.session_state.show_info = True
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.header("ℹ️ Informasi Aplikasi")
        
        st.markdown("""
        ### 🔬 Tentang Sistem
        
        Aplikasi deteksi Hepatitis menggunakan:
        
        1. **11 Pertanyaan Gejala Klinis**
        2. **Analisis Foto Mata dengan KNN Model**
           - Deteksi mata (MediaPipe/Haar Cascade)
           - Ekstraksi RGB sclera
           - Klasifikasi K-Nearest Neighbors (k=3)
        3. **Analisis Foto Urine** (warna & brightness)
        
        ---
        
        ### 📊 Penilaian Risiko
        
        - **🔴 TINGGI (60-100):** Tindakan segera
        - **🟡 SEDANG (35-59):** Pemeriksaan disarankan
        - **🟢 RENDAH (0-34):** Monitor berkala
        
        **Komponen Skor:**
        - Gejala klinis: 40%
        - Analisis mata: 30%
        - Analisis urine: 30%
        
        ---
        
        ### 📸 Tips Foto
        
        **Mata:**
        - ✅ Pencahayaan terang merata
        - ✅ Mata terbuka lebar
        - ✅ Fokus pada sclera (putih mata)
        
        **Urine:**
        - ✅ Wadah transparan
        - ✅ Latar belakang putih
        - ✅ Cahaya natural
        - ❌ Jangan pakai flash
        
        ---
        
        ### ⚕️ Gejala Hepatitis
        
        **Utama:**
        - Kulit & mata kuning
        - Urin gelap
        - Feses pucat
        
        **Pendukung:**
        - Mual/muntah
        - Nyeri perut
        - Kelelahan
        - Gatal
        - Demam
        
        ---
        """)
        
        st.caption("© 2025 Hepacheck.ai")
        st.caption("Developed by Fadey Ezra & Diandra Almeira")
        
        # Progress indicator
        if st.session_state.step < 14:
            st.markdown("---")
            st.progress(progress)
            st.caption(f"Step {st.session_state.step} of {total_steps}")

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()