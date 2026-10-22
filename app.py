
import streamlit as st
from plotly import express as px
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import datetime
import os
import html

# ==================================================
# 1. KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Anggriz | Fruit AI Studio",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. CSS CUTE AESTHETIC - SAGE GREEN
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&display=swap');

:root {
    --sage: #819779;
    --sage-dark: #52684F;
    --sage-light: #E7EDE2;
    --cream: #F8F7F0;
    --white: #FFFFFF;
    --text: #394738;
    --muted: #788374;
    --peach: #F5DCCB;
    --border: #E3E8DE;
}

/* Background */
.stApp {
    background: var(--cream);
    color: var(--text);
    font-family: 'Nunito', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at top left, #E8EDE2 0%, transparent 38%),
        var(--cream);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    visibility: hidden;
}

.block-container {
    max-width: 1250px !important;
    padding: 2rem 2rem 2.5rem !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #E7EDE2;
    border-right: 1px solid #D9E1D3;
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

[data-testid="stSidebar"] [data-testid="stImage"] img {
    border-radius: 16px;
}

[data-testid="stSidebar"] .stRadio label {
    font-weight: 700;
}

/* General text */
h1, h2, h3, p, label {
    font-family: 'Nunito', sans-serif !important;
}

h1, h2, h3 {
    color: var(--sage-dark) !important;
}

p {
    line-height: 1.75;
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #DCE7D5, #EFF1E5 65%, #F5E7D9);
    border: 1px solid #DCE4D6;
    border-radius: 30px;
    padding: 42px 38px;
    margin: 8px 0 24px;
    box-shadow: 0 10px 30px rgba(75, 99, 69, 0.08);
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.75);
    border: 1px solid #DCE4D6;
    border-radius: 50px;
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
    color: #52684F;
    margin-bottom: 16px;
}

.hero h1 {
    font-size: clamp(30px, 4vw, 48px) !important;
    line-height: 1.18;
    margin: 0 0 14px;
    letter-spacing: -1.2px;
    color: #435840 !important;
}

.hero p {
    font-size: 16px;
    color: #66755F;
    max-width: 650px;
    margin: 0;
}

.hero-art {
    font-size: 72px;
    text-align: center;
    padding: 15px 0;
    filter: drop-shadow(0 8px 8px rgba(70, 90, 60, 0.10));
}

/* Titles */
.section-title {
    color: var(--sage-dark);
    font-size: 21px;
    font-weight: 900;
    margin: 24px 0 14px;
}

.section-subtitle {
    color: var(--muted);
    font-size: 14px;
    margin-top: -8px;
    margin-bottom: 18px;
}

/* Cards */
.soft-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 5px 18px rgba(65, 84, 57, 0.045);
    margin-bottom: 16px;
}

.feature-card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 22px;
    height: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(65, 84, 57, 0.09);
}

.feature-icon {
    font-size: 30px;
    margin-bottom: 8px;
}

.feature-card h3 {
    font-size: 17px !important;
    margin: 5px 0;
}

.feature-card p {
    font-size: 13px;
    color: var(--muted);
    margin: 0;
}

/* Form inputs */
.stTextInput input,
.stTextArea textarea,
[data-baseweb="input"] input {
    border-radius: 12px !important;
    border-color: #DCE4D6 !important;
    background: #FFFFFF !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--sage) !important;
    box-shadow: 0 0 0 2px rgba(129,151,121,0.15) !important;
}

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 2px dashed #BACBB2;
    border-radius: 20px;
    padding: 16px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: var(--sage);
    color: #FFFFFF !important;
    border: none;
    border-radius: 14px;
    padding: 0.65rem 1.3rem;
    font-weight: 800;
    min-height: 44px;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: var(--sage-dark);
    color: #FFFFFF !important;
    border: none;
    box-shadow: 0 5px 12px rgba(82,104,79,0.18);
}

.stButton > button:active,
.stFormSubmitButton > button:active {
    transform: scale(0.98);
}

/* Metrics */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
    font-weight: 700;
}

[data-testid="stMetricValue"] {
    color: var(--sage-dark);
    font-weight: 900;
}

/* Progress */
.stProgress > div > div > div {
    background: var(--sage);
    border-radius: 10px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
}

/* Image */
[data-testid="stImage"] img {
    border-radius: 18px;
}

/* Feedback and status */
.stAlert {
    border-radius: 14px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 22px 10px;
    color: #778472;
    font-size: 13px;
    line-height: 1.8;
}

.pill {
    display: inline-block;
    background: #E7EDE2;
    color: #52684F;
    padding: 5px 11px;
    border-radius: 50px;
    font-size: 12px;
    font-weight: 800;
    margin: 3px;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2rem !important;
    }

    .hero {
        padding: 27px 22px;
        border-radius: 22px;
    }

    .hero-art {
        font-size: 52px;
    }

    .soft-card {
        padding: 18px;
    }
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. SESSION STATE
# ==================================================
defaults = {
    "page": "home",
    "user_name": "",
    "user_campus": "",
    "used_feature": False,
    "last_mode": None,
    "last_pred": None,
    "last_conf": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# 4. LOAD MODEL
# ==================================================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Anggriz_Laporan 4.pt")
    classifier = tf.keras.models.load_model(
        "model/anggriz_Laporan 2.h5"
    )
    return yolo_model, classifier


try:
    yolo_model, classifier = load_models()
except Exception as e:
    yolo_model, classifier = None, None
    st.error(
        "Model belum berhasil dimuat. "
        "Periksa lokasi file model dan dependensinya."
    )
    st.code(str(e))


# Sesuaikan dengan urutan kelas ketika model CNN dilatih.
class_names = ["Apel", "Tomat"]


# ==================================================
# 5. KOMPONEN TAMPILAN
# ==================================================
def hero_section(eyebrow, title, description, art="🍏 🍅"):
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            f"""
            <div class="hero">
                <div class="hero-tag">{eyebrow}</div>
                <h1>{title}</h1>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="hero">
                <div class="hero-art">{art}</div>
                <p style="text-align:center;font-weight:800;">
                    Fresh fruit, smart AI 🌿
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


def footer():
    st.markdown(
        """
        <div class="footer">
            <div>🌿 <b>ANGGRIZ FRUIT AI STUDIO</b></div>
            Made with care by Anggriz Hanaz Salsabil<br>
            Proyek UAS Praktikum Pemrograman Big Data
            <br>Universitas Syiah Kuala
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title, subtitle=None):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )

    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True
        )


# ==================================================
# 6. HALAMAN AWAL
# ==================================================
def halaman_awal():
    hero_section(
        "WELCOME TO YOUR LITTLE FRUIT LAB",
        "Kenali Buah, dengan Cara yang Lebih Cerdas.",
        "Dashboard AI untuk mendeteksi dan mengklasifikasikan "
        "buah apel dan tomat melalui teknologi YOLO dan CNN.",
        "🍎 🍅"
    )

    st.markdown(
        """
        <div style="text-align:center; margin: 12px 0 24px;">
            <span class="pill">🍃 Sage aesthetic</span>
            <span class="pill">🤖 YOLO detection</span>
            <span class="pill">🧠 CNN classification</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    section_title(
        "Kenalan dengan fitur kami ✨",
        "Pilih analisis yang kamu butuhkan."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔎</div>
                <h3>Deteksi Objek</h3>
                <p>
                    Temukan posisi apel dan tomat dalam gambar,
                    lengkap dengan bounding box dan confidence.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <h3>Klasifikasi Gambar</h3>
                <p>
                    Kenali kategori buah berdasarkan prediksi
                    model CNN dan probabilitas setiap kelas.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    left, center, right = st.columns([1, 2, 1])

    with center:
        if st.button(
            "Mulai Petualangan →",
            use_container_width=True,
            key="start_registration"
        ):
            st.session_state["page"] = "register"
            st.rerun()

    footer()


# ==================================================
# 7. HALAMAN REGISTRASI
# ==================================================
def halaman_registrasi():
    hero_section(
        "ONE LITTLE STEP",
        "Sebelum Mulai, Kenalan Dulu! 🌱",
        "Isi identitas singkatmu untuk masuk ke Fruit AI Studio.",
        "🌿"
    )

    _, center, _ = st.columns([1, 2, 1])

    with center:
        st.markdown(
            '<div class="soft-card">',
            unsafe_allow_html=True
        )

        with st.form("form_reg"):
            st.markdown("### 💌 Registrasi Pengguna")

            name = st.text_input(
                "Nama Lengkap",
                value=st.session_state["user_name"],
                placeholder="Contoh: Anggriz Hanaz"
            )

            campus = st.text_input(
                "Nama Kampus",
                value=st.session_state["user_campus"],
                placeholder="Contoh: Universitas Syiah Kuala"
            )

            submitted = st.form_submit_button(
                "Masuk ke Dashboard 🌷",
                use_container_width=True
            )

            if submitted:
                if not name.strip() or not campus.strip():
                    st.warning(
                        "Nama lengkap dan nama kampus wajib diisi."
                    )
                else:
                    st.session_state["user_name"] = name.strip()
                    st.session_state["user_campus"] = campus.strip()
                    st.session_state["page"] = "main"
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("← Kembali ke Halaman Awal"):
        st.session_state["page"] = "home"
        st.rerun()

    footer()


# ==================================================
# 8. HALAMAN DASHBOARD UTAMA
# ==================================================
def halaman_main():
    # Sidebar
    if os.path.exists("model/LOGO UUSK.jpg"):
        st.sidebar.image(
            "model/LOGO UUSK.jpg",
            use_container_width=True
        )

    st.sidebar.markdown("## 🍃 Fruit AI Studio")
    st.sidebar.caption("Your little fruit laboratory")

    menu = st.sidebar.radio(
        "🧭 Pilih Mode Analisis",
        ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Profil Pengguna")
    st.sidebar.write(f"**{st.session_state['user_name']}**")
    st.sidebar.caption(st.session_state["user_campus"])

    st.sidebar.info(
        "📸 Unggah gambar apel atau tomat untuk memulai analisis."
    )

    # Hero
    hero_section(
        "YOUR AI WORKSPACE",
        "Let's Explore Your Fruits! 🍎",
        "Unggah gambar buah dan biarkan AI membantu "
        "mengenali objek atau menentukan kategori buah.",
        "🍏"
    )

    mode_description = (
        "YOLO akan mendeteksi objek dan menunjukkan lokasinya."
        if menu == "Deteksi Objek (YOLO)"
        else "CNN akan memprediksi apakah gambar termasuk apel atau tomat."
    )

    section_title("Area Analisis 🍃", mode_description)

    # Upload berada di tengah
    _, upload_col, _ = st.columns([1, 2, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "📂 Pilih gambar buah",
            type=["jpg", "jpeg", "png"],
            key="fruit_upload",
            help="Format gambar: JPG, JPEG, atau PNG."
        )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="soft-card" style="text-align:center;">
                <div style="font-size:52px;">🧺</div>
                <h3>Siap untuk memulai?</h3>
                <p style="color:#788374;">
                    Unggah foto apel atau tomat menggunakan
                    kotak di atas untuk melihat hasil analisis AI.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        footer()
        if st.button("← Kembali ke Registrasi"):
            st.session_state["page"] = "register"
            st.rerun()
        return

    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Gambar tidak dapat dibaca. Silakan unggah gambar lain.")
        return

    # Tampilkan gambar sumber
    col1, col2 = st.columns(2, gap="large")

    with col1:
        section_title("🖼️ Gambar Asli")
        st.image(
            img,
            caption="Gambar yang kamu unggah",
            use_container_width=True
        )

    # ----------------------------------------------
    # DETEKSI YOLO
    # ----------------------------------------------
    if menu == "Deteksi Objek (YOLO)":
        if yolo_model is None:
            with col2:
                st.error("Model YOLO belum tersedia.")
        else:
            with st.spinner("🌿 AI sedang mencari objek..."):
                results = yolo_model(np.array(img), verbose=False)
                result = results[0]
                boxes = result.boxes

            with col2:
                section_title("🔎 Hasil Deteksi")

                if boxes is not None and len(boxes) > 0:
                    # Ultralytics plot menghasilkan citra BGR.
                    result_img = result.plot()
                    result_img = cv2.cvtColor(
                        result_img,
                        cv2.COLOR_BGR2RGB
                    )

                    st.image(
                        result_img,
                        caption="Hasil deteksi dengan bounding box",
                        use_container_width=True
                    )

                    st.success(
                        f"Berhasil mendeteksi {len(boxes)} objek! 🎉"
                    )

                    data = []
                    confidences = []

                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])

                        data.append({
                            "Label": result.names[cls],
                            "Confidence": f"{conf * 100:.2f}%",
                            "x1": round(xyxy[0]),
                            "y1": round(xyxy[1]),
                            "x2": round(xyxy[2]),
                            "y2": round(xyxy[3])
                        })

                        confidences.append(conf * 100)

                    df = pd.DataFrame(data)

                    st.session_state["last_mode"] = "detection"
                    st.session_state["last_pred"] = (
                        f"{len(boxes)} objek terdeteksi"
                    )
                    st.session_state["last_conf"] = float(
                        np.mean(confidences)
                    )
                    st.session_state["used_feature"] = True

                else:
                    st.warning(
                        "Belum ada objek yang terdeteksi. "
                        "Coba gambar dengan pencahayaan lebih baik."
                    )

                    df = pd.DataFrame()

            if not df.empty:
                section_title("📋 Rincian Deteksi")

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "💡 Confidence menunjukkan tingkat keyakinan "
                    "model terhadap setiap deteksi, bukan jaminan "
                    "bahwa prediksi selalu benar."
                )

    # ----------------------------------------------
    # KLASIFIKASI CNN
    # ----------------------------------------------
    else:
        if classifier is None:
            with col2:
                st.error("Model klasifikasi belum tersedia.")
        else:
            with st.spinner("🧠 CNN sedang mengenali buah..."):
                target_size = tuple(classifier.input_shape[1:3])

                img_resized = img.resize(
                    (target_size[1], target_size[0])
                )

                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction = classifier.predict(
                    img_array,
                    verbose=0
                )

                if prediction.ndim != 2 or prediction.shape[1] != len(class_names):
                    st.error(
                        f"Output model memiliki bentuk {prediction.shape}. "
                        "Pastikan jumlah kelas dan format output CNN sesuai."
                    )
                    return

                class_index = int(np.argmax(prediction[0]))
                confidence = float(prediction[0][class_index]) * 100

            with col2:
                section_title("🧠 Hasil Klasifikasi")

                st.markdown(
                    f"""
                    <div class="soft-card" style="text-align:center;">
                        <div style="font-size:48px;">
                            {"🍎" if class_names[class_index] == "Apel" else "🍅"}
                        </div>
                        <p style="color:#788374;font-weight:800;">
                            PREDIKSI KATEGORI
                        </p>
                        <h2 style="font-size:34px;">
                            {html.escape(class_names[class_index])}
                        </h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.metric(
                    "Confidence Prediksi",
                    f"{confidence:.2f}%"
                )

                st.progress(
                    max(0.0, min(1.0, confidence / 100))
                )

                prob_df = pd.DataFrame({
                    "Kategori": class_names,
                    "Probabilitas": [
                        f"{float(p) * 100:.2f}%"
                        for p in prediction[0]
                    ]
                })

                st.markdown("#### Probabilitas per Kelas")
                st.dataframe(
                    prob_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.session_state["last_mode"] = "classification"
                st.session_state["last_pred"] = class_names[class_index]
                st.session_state["last_conf"] = confidence
                st.session_state["used_feature"] = True

    # ----------------------------------------------
    # DONUT CHART
    # ----------------------------------------------
    if st.session_state["used_feature"]:
        val = st.session_state["last_conf"]

        if val is not None:
            section_title(
                "🌼 Tingkat Keyakinan Model",
                "Visualisasi perbandingan confidence dan sisa persentase."
            )

            benar = max(0.0, min(100.0, float(val)))
            sisa = 100.0 - benar

            pie_data = pd.DataFrame({
                "Hasil": ["Confidence", "Sisa Persentase"],
                "Persentase": [benar, sisa]
            })

            fig = px.pie(
                pie_data,
                values="Persentase",
                names="Hasil",
                hole=0.65,
                color="Hasil",
                color_discrete_map={
                    "Confidence": "#819779",
                    "Sisa Persentase": "#E5EADF"
                }
            )

            fig.update_traces(
                textinfo="label+percent",
                textfont_size=12,
                marker=dict(
                    line=dict(color="#F8F7F0", width=4)
                )
            )

            fig.update_layout(
                height=330,
                margin=dict(l=10, r=10, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="Nunito, sans-serif",
                    color="#52684F"
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                )
            )

            _, chart_col, _ = st.columns([1, 2, 1])

            with chart_col:
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

            st.caption(
                "Catatan: sisa persentase adalah 100% dikurangi "
                "confidence prediksi, bukan probabilitas kesalahan "
                "yang telah divalidasi."
            )

            _, feedback_col, _ = st.columns([1, 2, 1])

            with feedback_col:
                if st.button(
                    "💌 Berikan Feedback",
                    use_container_width=True
                ):
                    st.session_state["page"] = "feedback"
                    st.rerun()

    st.markdown("---")

    if st.button("← Kembali ke Registrasi"):
        st.session_state["page"] = "register"
        st.rerun()

    footer()


# ==================================================
# 9. HALAMAN FEEDBACK
# ==================================================
def halaman_feedback():
    hero_section(
        "YOUR VOICE MATTERS",
        "Ceritakan Pengalamanmu! 💌",
        "Masukanmu membantu kami membuat Fruit AI Studio "
        "menjadi lebih nyaman dan bermanfaat.",
        "🌷"
    )

    _, center, _ = st.columns([1, 2, 1])

    with center:
        with st.form("form_feedback"):
            st.markdown(
                '<div class="section-title">💚 Feedback Pengguna</div>',
                unsafe_allow_html=True
            )

            st.write(
                f"Terima kasih, **{st.session_state['user_name']}**!"
            )

            rating = st.slider(
                "Seberapa puas kamu dengan aplikasi ini?",
                min_value=1,
                max_value=5,
                value=5,
                help="1 = sangat tidak puas, 5 = sangat puas"
            )

            st.caption({
                1: "😕 Sangat tidak puas",
                2: "🙁 Kurang puas",
                3: "🙂 Cukup puas",
                4: "😊 Puas",
                5: "🥰 Sangat puas"
            }[rating])

            comment = st.text_area(
                "Komentar atau saran",
                height=130,
                placeholder="Tulis pengalaman atau saranmu di sini..."
            )

            submitted = st.form_submit_button(
                "Kirim Feedback 🌿",
                use_container_width=True
            )

            if submitted:
                now = datetime.datetime.now().isoformat(
                    timespec="seconds"
                )

                row = {
                    "timestamp": now,
                    "user_name": st.session_state["user_name"],
                    "user_campus": st.session_state["user_campus"],
                    "mode": st.session_state["last_mode"] or "",
                    "prediction": st.session_state["last_pred"] or "",
                    "confidence": (
                        st.session_state["last_conf"]
                        if st.session_state["last_conf"] is not None
                        else ""
                    ),
                    "rating": rating,
                    "comment": comment
                }

                df_row = pd.DataFrame([row])
                csv_path = "feedbacks.csv"

                try:
                    file_exists = (
                        os.path.exists(csv_path)
                        and os.path.getsize(csv_path) > 0
                    )

                    df_row.to_csv(
                        csv_path,
                        index=False,
                        mode="a" if file_exists else "w",
                        header=not file_exists,
                        encoding="utf-8"
                    )

                    st.session_state["used_feature"] = False
                    st.success(
                        "Feedback berhasil dikirim. "
                        "Terima kasih sudah berbagi! 🌷"
                    )

                except Exception as e:
                    st.error(f"Gagal menyimpan feedback: {e}")

    if st.button("← Kembali ke Dashboard"):
        st.session_state["page"] = "main"
        st.rerun()

    footer()


# ==================================================
# 10. ROUTING HALAMAN
# ==================================================
page = st.session_state["page"]

if page == "home":
    halaman_awal()
elif page == "register":
    halaman_registrasi()
elif page == "main":
    halaman_main()
elif page == "feedback":
    halaman_feedback()
else:
    st.session_state["page"] = "home"
    st.rerun()
