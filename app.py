import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.eye_detection import multi_stage_detection
from utils.jaundice_analysis import (
    analyze_sclera_color,
    detect_jaundice,
    get_risk_level,
    create_visualization_data
)
# hello
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
    "Demam sebelum gejala hati muncul?",
    "Nyeri otot (myalgia)?",
    "Mual atau muntah?",
    "Nyeri perut kanan atas?",
    "Feses pucat/abu-abu?",
    "Lama gejala berlangsung?",
    "Fatigue / lemas kronis?",
    "Gatal pada kulit?",
    "Perut membesar (ascites)?",
    "Mudah memar/pendarahan?",
    "Kaki bengkak?"
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def convert_uploaded_image(uploaded_file):
    """Convert uploaded file to OpenCV format"""
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    # Handle different image formats
    if len(image_array.shape) == 2:  # Grayscale
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    elif image_array.shape[2] == 4:  # RGBA
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    return image_array

def analyze_urine_color(image):
    """Analyze urine color for jaundice indicators"""
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Get average color from center region (avoid edges)
    h, w = image.shape[:2]
    center_region = hsv[h//4:3*h//4, w//4:3*w//4]
    
    avg_hsv = center_region.mean(axis=(0, 1))
    h_avg, s_avg, v_avg = avg_hsv
    
    # Get RGB average
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    center_rgb = rgb[h//4:3*h//4, w//4:3*w//4]
    avg_rgb = center_rgb.mean(axis=(0, 1))
    
    # Detect dark urine (indicator of jaundice)
    # Dark urine: low brightness, yellow-orange hue
    is_dark = v_avg < 150
    is_yellow_orange = (15 < h_avg < 45)
    high_saturation = s_avg > 80
    
    confidence = 0
    if is_dark:
        confidence += 40
    if is_yellow_orange:
        confidence += 35
    if high_saturation:
        confidence += 25
    
    is_abnormal = confidence > 50
    
    return {
        'rgb': avg_rgb,
        'hsv': avg_hsv,
        'is_abnormal': is_abnormal,
        'confidence': confidence,
        'brightness': v_avg
    }

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
    st.markdown('<p class="main-header">🔬 Sistem Deteksi Jaundice</p>', unsafe_allow_html=True)
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
        
        question = QUESTIONS[st.session_state.step - 1]
        st.subheader(question)
        
        st.write("")  # Spacing
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            pass  # Empty column for spacing
        
        with col2:
            col_yes, col_no = st.columns(2)
            
            with col_yes:
                if st.button("✅ Ya", use_container_width=True, type="primary", key=f"yes_{st.session_state.step}"):
                    st.session_state.answers[f'q{st.session_state.step}'] = 'Ya'
                    st.session_state.step += 1
                    st.rerun()
            
            with col_no:
                if st.button("❌ Tidak", use_container_width=True, key=f"no_{st.session_state.step}"):
                    st.session_state.answers[f'q{st.session_state.step}'] = 'Tidak'
                    st.session_state.step += 1
                    st.rerun()
        
        with col3:
            pass  # Empty column for spacing
        
        # Show previous answers
        if st.session_state.step > 1:
            with st.expander("📊 Lihat Jawaban Sebelumnya"):
                for i in range(1, st.session_state.step):
                    answer = st.session_state.answers.get(f'q{i}', 'N/A')
                    icon = "✅" if answer == "Ya" else "❌"
                    st.write(f"{icon} **Pertanyaan {i}:** {QUESTIONS[i-1]} - **{answer}**")
    
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
                image_array = convert_uploaded_image(uploaded_file)
                st.session_state.photos['eye'] = image_array
                
                # Preview
                st.success("✅ Foto mata berhasil diupload!")
                st.image(uploaded_file, caption="Preview Foto Mata", width='content')
                
                if st.button("➡️ Lanjut ke Upload Foto Urine", type="primary", use_container_width=True):
                    st.session_state.step = 13
                    st.rerun()
        
        with col2:
            st.write("**Contoh Foto yang Baik:**")
            st.image("https://via.placeholder.com/300x300.png?text=Contoh+Foto+Mata", 
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
                image_array = convert_uploaded_image(uploaded_file)
                st.session_state.photos['urine'] = image_array
                
                # Preview
                st.success("✅ Foto urine berhasil diupload!")
                st.image(uploaded_file, caption="Preview Foto Urine", width="content")
                
                if st.button("🔬 Analisis Sekarang", type="primary", use_container_width=True):
                    st.session_state.step = 14
                    st.rerun()
        
        with col2:
            st.write("**Contoh Foto yang Baik:**")
            st.image("https://via.placeholder.com/300x400.png?text=Contoh+Foto+Urine", 
                    caption="Urine dalam gelas bening", width=300)
    
    # ========================================================================
    # BAGIAN 4: PROSES & HASIL (Step 14)
    # ========================================================================
    elif st.session_state.step == 14:
        st.header("🔬 Hasil Analisis")
        
        eye_image = st.session_state.photos.get('eye')
        urine_image = st.session_state.photos.get('urine')
        
        col1, col2 = st.columns(2)
        
        # ====================================================================
        # ANALISIS MATA
        # ====================================================================
        with col1:
            st.subheader("👁️ Analisis Mata")
            
            if eye_image is not None:
                with st.spinner("🔍 Menganalisis mata..."):
                    eye_result = multi_stage_detection(eye_image)
                
                if eye_result and eye_result.get('left_eye') is not None:
                    # Display detected eye
                    st.image(cv2.cvtColor(eye_result['left_eye'], cv2.COLOR_BGR2RGB), 
                            caption="Detected Eye Region", width="content")
                    
                    # Analyze sclera color
                    eye_color = analyze_sclera_color(
                        eye_result['left_eye'], 
                        eye_result['left_mask']
                    )
                    
                    if eye_color:
                        eye_jaundice, eye_conf, eye_yi = detect_jaundice(eye_color)
                        
                        # Display metrics
                        st.metric("Status Sclera", "⚠️ Kekuningan" if eye_jaundice else "✅ Normal")
                        st.metric("Confidence", f"{eye_conf:.1f}%")
                        st.metric("Yellow Index", f"{eye_yi:.3f}")
                        
                        # Display masked sclera
                        if eye_result.get('left_mask') is not None:
                            eye_masked = eye_result['left_eye'].copy()
                            eye_masked[eye_result['left_mask'] == 0] = 0
                            with st.expander("👁️ Lihat Area Sclera"):
                                st.image(cv2.cvtColor(eye_masked, cv2.COLOR_BGR2RGB), 
                                       caption="Sclera (Bagian Putih Mata)", width="content")
                        
                        # Color details
                        with st.expander("🎨 Detail Warna Sclera"):
                            r, g, b = eye_color['rgb']
                            st.write(f"**RGB:** R={r:.1f}, G={g:.1f}, B={b:.1f}")
                            h, s, v = eye_color['hsv']
                            st.write(f"**HSV:** H={h:.1f}°, S={s:.1f}, V={v:.1f}")
                    else:
                        st.warning("⚠️ Tidak dapat menganalisis warna sclera")
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
                with st.spinner("🔍 Menganalisis urine..."):
                    urine_analysis = analyze_urine_color(urine_image)
                
                # Display urine image
                st.image(cv2.cvtColor(urine_image, cv2.COLOR_BGR2RGB), 
                        caption="Foto Urine", width="content")
                
                # Display metrics
                urine_status = "⚠️ Gelap/Abnormal" if urine_analysis['is_abnormal'] else "✅ Normal"
                st.metric("Status Warna", urine_status)
                st.metric("Confidence", f"{urine_analysis['confidence']:.1f}%")
                st.metric("Brightness", f"{urine_analysis['brightness']:.1f}")
                
                # Color details
                with st.expander("🎨 Detail Warna Urine"):
                    r, g, b = urine_analysis['rgb']
                    st.write(f"**RGB:** R={r:.1f}, G={g:.1f}, B={b:.1f}")
                    h, s, v = urine_analysis['hsv']
                    st.write(f"**HSV:** H={h:.1f}°, S={s:.1f}, V={v:.1f}")
                    st.write(f"**Brightness (V):** {v:.1f}")
                    
                    # Interpretation
                    st.write("---")
                    if urine_analysis['is_abnormal']:
                        st.warning("Urine tampak gelap, dapat mengindikasikan tingginya kadar bilirubin")
                    else:
                        st.success("Warna urine dalam kisaran normal")
                
                urine_jaundice = urine_analysis['is_abnormal']
                urine_conf = urine_analysis['confidence']
            else:
                st.error("❌ Foto urine tidak ditemukan")
                urine_jaundice, urine_conf = False, 0
        
        # ====================================================================
        # KESIMPULAN AKHIR
        # ====================================================================
        st.markdown("---")
        st.header("📋 Kesimpulan Analisis")
        
        # Calculate symptom score
        symptom_score = sum(1 for v in st.session_state.answers.values() if v == 'Ya')
        
        # Combined detection
        visual_detection = eye_jaundice or urine_jaundice
        avg_confidence = (eye_conf + urine_conf) / 2
        
        # Calculate total risk score
        risk_score = 0
        
        # Weight from symptoms (max 40 points)
        risk_score += (symptom_score / 11) * 40
        
        # Weight from eye (max 30 points)
        if eye_jaundice:
            risk_score += (eye_conf / 100) * 30
        
        # Weight from urine (max 30 points)
        if urine_jaundice:
            risk_score += (urine_conf / 100) * 30
        
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
        st.markdown(f"## {risk_icon} RISIKO {risk_level} JAUNDICE")
        st.markdown(f"**Skor Risiko Total:** {risk_score:.1f}/100")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.write("")
        
        # Detailed breakdown
        col_detail1, col_detail2, col_detail3 = st.columns(3)
        
        with col_detail1:
            st.metric("Gejala Klinis", f"{symptom_score}/11", 
                     delta=f"{(symptom_score/11)*100:.0f}% positif" if symptom_score > 0 else "Tidak ada")
        
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
            
            1. Tidak ada indikasi jaundice yang signifikan
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
            for i, (key, value) in enumerate(st.session_state.answers.items(), 1):
                icon = "✅" if value == "Ya" else "❌"
                st.write(f"{icon} **{QUESTIONS[i-1]}** → {value}")
        
        # Disclaimer
        st.markdown("---")
        st.info("""
        **⚠️ DISCLAIMER PENTING:**
        
        Hasil analisis ini **BUKAN** merupakan diagnosis medis final dan tidak menggantikan 
        pemeriksaan oleh tenaga medis profesional.
        
        Aplikasi ini hanya sebagai **alat bantu skrining awal** untuk membantu deteksi dini 
        potensi jaundice. Untuk diagnosis yang akurat dan penanganan yang tepat, 
        **WAJIB konsultasi dengan dokter** dan melakukan pemeriksaan laboratorium lengkap.
        
        Jangan menggunakan hasil ini sebagai dasar pengobatan mandiri.
        """)
        
        # Action buttons
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🔄 Analisis Baru", use_container_width=True, type="primary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col_btn2:
            if st.button("📄 Export Hasil (Coming Soon)", use_container_width=True, disabled=True):
                st.info("Fitur export akan segera hadir!")
        
        with col_btn3:
            if st.button("ℹ️ Info Lengkap", use_container_width=True):
                st.session_state.show_info = True
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.header("ℹ️ Informasi Aplikasi")
        
        st.markdown("""
        ### 🔬 Tentang Sistem
        
        Aplikasi deteksi jaundice menggunakan:
        
        1. **11 Pertanyaan Gejala Klinis**
        2. **Analisis Foto Mata** (warna sclera)
        3. **Analisis Foto Urine** (warna & brightness)
        
        ---
        
        ### 🎯 Metode Deteksi Mata
        
        - **MediaPipe Face Mesh** (full face)
        - **Haar Cascade** (zoomed face)
        - **Color Segmentation** (close-up)
        
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
        
        ### ⚕️ Gejala Jaundice
        
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
        
        st.caption("© 2024 Jaundice Detection System v2.0")
        st.caption("Developed with ❤️ using Streamlit")
        
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