import streamlit as st
import pandas as pd
import pickle
import os
import base64
import numpy as np
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title=" AI Powered PCOS Detection",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .header-style {
        font-size: 36px;
        font-weight: bold;
        color: #0B1F3A;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: none;
        letter-spacing: 2px;
    }
    .subheader-style {
        font-size: 18px;
        color: #4ECDC4;
        margin: 20px 0px 10px 0px;
    }
    .info-box {
        background-color: rgba(255,255,255,0.10);
        padding: 18px;
        border-radius: 10px;
        margin: 10px 0px;
        border-left: 5px solid #00CFFF;
        color: #E0F7FF;
        backdrop-filter: blur(6px);
    }
    .info-box h3 { color: #00CFFF; }
    .howto-step {
        background: linear-gradient(135deg, rgba(10, 54, 104, 0.92) 0%, rgba(16, 92, 165, 0.85) 55%, rgba(7, 43, 88, 0.92) 100%);
        padding: 10px 12px;
        margin: 8px 0;
        border-left-width: 4px;
        border: 1px solid rgba(120, 188, 238, 0.45);
        box-shadow: 0 6px 14px rgba(0, 45, 95, 0.35);
    }
    .howto-step h3 {
        font-size: 18px;
        margin: 0 0 4px 0;
    }
    .howto-step p {
        font-size: 14px;
        margin: 0;
        line-height: 1.5;
    }
    .warning-box {
        background: linear-gradient(135deg, rgba(107, 26, 26, 0.95) 0%, rgba(74, 16, 16, 0.95) 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0px;
        border: 1px solid rgba(255, 130, 130, 0.55);
        border-left: 5px solid #FF6B6B;
        color: #FFF1F1;
        box-shadow: 0 8px 18px rgba(30, 8, 8, 0.45);
    }
    .success-box {
        background: linear-gradient(135deg, rgba(10, 73, 61, 0.95) 0%, rgba(8, 55, 46, 0.95) 100%);
        padding: 18px;
        border-radius: 10px;
        margin: 10px 0px;
        border: 1px solid rgba(125, 238, 214, 0.45);
        border-left: 5px solid #00DEB0;
        color: #EFFFF9;
        backdrop-filter: blur(6px);
        box-shadow: 0 8px 18px rgba(5, 30, 24, 0.45);
    }
    .warning-box h2, .success-box h2 {
        color: #FFFFFF;
        margin-bottom: 6px;
    }
    .warning-box p, .success-box p {
        color: #FFFFFF;
        margin-bottom: 0;
    }
    .success-box h3 { color: #00DEB0; }

    /* ── DNA background hero banner ── */
    .dna-hero {
        position: relative;
        width: 100%;
        min-height: 360px;
        border-radius: 18px;
        overflow: hidden;
        margin-bottom: 28px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background-image:
            linear-gradient(rgba(4, 38, 82, 0.55), rgba(4, 38, 82, 0.55)),
            radial-gradient(circle at 18% 35%, rgba(52, 174, 240, 0.45), transparent 35%),
            radial-gradient(circle at 75% 60%, rgba(35, 130, 210, 0.45), transparent 42%),
            linear-gradient(135deg, #06346a 0%, #0c4f95 35%, #0b3d82 65%, #0a2e65 100%);
        box-shadow: 0 8px 40px rgba(0,80,200,0.5);
    }
    .dna-hero::before {
        content: '';
        position: absolute;
        inset: 0;
        background:
            radial-gradient(ellipse at 20% 50%, rgba(0,150,255,0.18) 0%, transparent 60%),
            radial-gradient(ellipse at 80% 30%, rgba(0,210,200,0.12) 0%, transparent 50%),
            radial-gradient(ellipse at 60% 80%, rgba(0,80,255,0.10) 0%, transparent 40%);
        pointer-events: none;
    }
    /* Animated SVG DNA overlay */
    .dna-svg-wrap {
        position: absolute;
        inset: 0;
        opacity: 0.25;
        pointer-events: none;
    }
    .dna-hero-content {
        position: relative;
        z-index: 2;
        text-align: center;
        padding: 36px 30px 30px;
    }
    .dna-hero-title {
        font-size: 72px;
        font-weight: 900;
        color: #FFFFFF;
        letter-spacing: 3px;
        text-shadow: none;
        margin-bottom: 2px;
        font-family: Georgia, serif;
    }
    .dna-hero-tagline {
        font-size: 20px;
        color: #D9F2FF;
        margin-top: 10px;
        letter-spacing: 0.5px;
    }
    .dna-hero-subtitle {
        font-size: 33px;
        letter-spacing: 2px;
        color: #E7F5FF;
        margin-bottom: 4px;
        font-family: Georgia, serif;
    }
    .dna-hero-small {
        font-size: 28px;
        letter-spacing: 3px;
        color: #D4EAFB;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .dna-hero-desc { display: none; }
    /* dot grid overlay */
    .dna-dots {
        position: absolute;
        inset: 0;
        background-image: radial-gradient(circle, rgba(0,180,255,0.25) 1px, transparent 1px);
        background-size: 28px 28px;
        pointer-events: none;
        opacity: 0.4;
    }
    /* border frame like the reference image */
    .dna-frame {
        position: absolute;
        inset: 14px;
        border: 2px solid rgba(232, 245, 255, 0.8);
        border-radius: 10px;
        pointer-events: none;
    }
    .home-logo-wrap {
        display: flex;
        justify-content: center;
        margin-bottom: 14px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, #0b2f57 0%, #0a2545 100%);
        border: 1px solid rgba(120, 188, 238, 0.45);
        border-radius: 12px;
        padding: 14px 14px 8px 14px;
        margin: 14px 0;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.28);
    }
    .section-title {
        color: #FFFFFF;
        font-size: 20px;
        font-weight: 700;
        line-height: 1.2;
        margin: -14px -14px 12px -14px;
        padding: 10px 14px;
        border-radius: 10px 10px 0 0;
        background: linear-gradient(90deg, #0e4f95 0%, #1e6fc3 100%);
        border-bottom: 1px solid rgba(180, 220, 255, 0.45);
    }

    .page-hero {
        position: relative;
        width: 100%;
        min-height: 170px;
        border-radius: 16px;
        overflow: hidden;
        margin: 8px 0 22px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        background-image:
            linear-gradient(rgba(4, 38, 82, 0.60), rgba(4, 38, 82, 0.60)),
            radial-gradient(circle at 18% 35%, rgba(52, 174, 240, 0.35), transparent 35%),
            radial-gradient(circle at 75% 60%, rgba(35, 130, 210, 0.35), transparent 42%),
            linear-gradient(135deg, #06346a 0%, #0c4f95 35%, #0b3d82 65%, #0a2e65 100%);
        box-shadow: 0 8px 28px rgba(0,80,200,0.35);
    }
    .page-hero-content {
        position: relative;
        z-index: 2;
        text-align: center;
        padding: 28px 16px;
    }
    .page-hero-title {
        font-size: 48px;
        font-weight: 900;
        color: #FFFFFF;
        letter-spacing: 1px;
        margin: 0;
        font-family: Georgia, serif;
    }
    .page-hero-subtitle {
        color: #E9F6FF;
        font-size: 18px;
        margin-top: 8px;
        font-weight: 600;
    }

    .feature-panel {
        background: linear-gradient(145deg, rgba(8, 44, 82, 0.95) 0%, rgba(9, 36, 70, 0.95) 100%);
        border: 1px solid rgba(120, 188, 238, 0.45);
        border-left: 6px solid #00CFFF;
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 400px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.28);
    }
    .feature-title {
        color: #BFE9FF;
        font-size: 20px;
        font-weight: 700;
        margin: 0 0 8px 0;
    }
    .feature-item {
        color: #E9F8FF;
        margin: 8px 0;
        font-size: 20px;
        font-weight: 600;
        opacity: 0.85;
        animation: keywordPulse 1.8s ease-in-out infinite;
    }
    .feature-item:nth-child(2) { animation-delay: 0.2s; }
    .feature-item:nth-child(3) { animation-delay: 0.4s; }
    .feature-item:nth-child(4) { animation-delay: 0.6s; }
    .feature-item:nth-child(5) { animation-delay: 0.8s; }
    @keyframes keywordPulse {
        0% { opacity: 0.7; transform: translateX(0px); text-shadow: 0 0 0 rgba(0,207,255,0.0); }
        50% { opacity: 1; transform: translateX(3px); text-shadow: 0 0 10px rgba(0,207,255,0.55); }
        100% { opacity: 0.7; transform: translateX(0px); text-shadow: 0 0 0 rgba(0,207,255,0.0); }
    }
    .home-visual-wrap {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 0;
        margin: 8px 0 12px 0;
    }
    .home-banner-pane img {
        width: 480px;
        height: 300px;
        object-fit: cover;
        border-radius: 12px;
        border: 1px solid rgba(120, 188, 238, 0.45);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.28);
        display: block;
    }
    .home-logo-zone {
        position: relative;
        margin-left: -2px;
        width: 420px;
        min-height: 320px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .home-logo-semicircle {
        width: 210px;
        height: 210px;
        border-radius: 0 210px 210px 0;
        overflow: hidden;
        border: 3px solid #bfe9ff;
        border-left: 0;
        background: linear-gradient(135deg, #0e4f95 0%, #1e6fc3 100%);
        box-shadow: 0 8px 20px rgba(0, 30, 68, 0.35);
        position: relative;
        z-index: 2;
    }
    .home-logo-semicircle img {
        width: 210px;
        height: 210px;
        object-fit: cover;
        object-position: center;
        display: block;
    }
    .home-ladder-item {
        position: absolute;
        left: 200px;
        background: rgba(16, 88, 166, 0.94);
        color: #E9F8FF;
        border: 1px solid rgba(120, 188, 238, 0.45);
        border-left: 5px solid #00CFFF;
        border-radius: 10px;
        padding: 8px 12px;
        font-size: 15px;
        font-weight: 700;
        white-space: nowrap;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.25);
    }
    .home-ladder-1 { top: 24px; }
    .home-ladder-2 { top: 84px; left: 224px; }
    .home-ladder-3 { top: 146px; left: 248px; }
    .home-ladder-4 { top: 208px; left: 272px; }
    @media (max-width: 1100px) {
        .home-visual-wrap { flex-direction: column; align-items: stretch; gap: 12px; }
        .home-banner-pane img { width: 100%; height: auto; }
        .home-logo-zone { margin-left: 0; width: 100%; min-height: 260px; justify-content: flex-start; }
        .home-ladder-item { left: 220px; }
        .home-ladder-2 { left: 236px; }
        .home-ladder-3 { left: 252px; }
        .home-ladder-4 { left: 268px; }
    }
    .rotterdam-box {
        background: transparent;
        border: 1px solid rgba(95, 156, 235, 0.35);
        border-left: 6px solid #1f77b4;
        border-radius: 12px;
        padding: 22px 20px;
        margin: 10px 0 16px 0;
    }
    .rotterdam-main-title {
        color: #0A2342;
        font-size: 30px;
        font-weight: 800;
        margin: 0 0 6px 0;
    }
    .rotterdam-desc {
        color: #1B3A5B;
        font-size: 18px;
        margin: 0 0 18px 0;
    }
    .rotterdam-steps {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
    }
    .rotterdam-step {
        background: rgba(18, 58, 118, 0.45);
        border: 1px solid rgba(115, 180, 255, 0.35);
        border-radius: 10px;
        padding: 14px;
        min-height: 126px;
    }
    .rotterdam-step-num {
        color: #D40000;
        font-family: Georgia, serif;
        font-size: 30px;
        font-style: italic;
        line-height: 1;
        margin-bottom: 8px;
    }
    .rotterdam-step-text {
        color: #0F2D4E;
        font-size: 21px;
        font-weight: 700;
        line-height: 1.25;
    }
    </style>
""", unsafe_allow_html=True)

# Shared DNA hero SVG background HTML
DNA_HERO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 900 260" preserveAspectRatio="xMidYMid slice">
  <defs>
    <radialGradient id="glow1" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#00AAFF" stop-opacity="0.7"/>
      <stop offset="100%" stop-color="#003080" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <!-- left DNA helix -->
  <g stroke="#00AAFF" stroke-width="1.5" fill="none" opacity="0.7">
    <path d="M60,20 C100,60 140,100 100,140 C60,180 100,220 140,260" />
    <path d="M140,20 C100,60 60,100 100,140 C140,180 100,220 60,260" />
    <line x1="60" y1="20" x2="140" y2="20" stroke-opacity="0.4"/>
    <line x1="100" y1="55" x2="100" y2="55" stroke="#00FFDD" stroke-width="2"/>
    <line x1="72" y1="85" x2="128" y2="85" stroke-opacity="0.5"/>
    <line x1="60" y1="140" x2="140" y2="140" stroke-opacity="0.6"/>
    <line x1="72" y1="195" x2="128" y2="195" stroke-opacity="0.4"/>
    <circle cx="60" cy="20" r="4" fill="#00CFFF" fill-opacity="0.8" stroke="none"/>
    <circle cx="140" cy="20" r="4" fill="#00CFFF" fill-opacity="0.8" stroke="none"/>
    <circle cx="100" cy="140" r="5" fill="#00FFDD" fill-opacity="0.7" stroke="none"/>
    <circle cx="60" cy="260" r="4" fill="#00CFFF" fill-opacity="0.6" stroke="none"/>
    <circle cx="140" cy="260" r="4" fill="#00CFFF" fill-opacity="0.6" stroke="none"/>
  </g>
  <!-- right DNA helix -->
  <g stroke="#00DDFF" stroke-width="1.5" fill="none" opacity="0.6" transform="translate(700,0)">
    <path d="M60,20 C100,60 140,100 100,140 C60,180 100,220 140,260" />
    <path d="M140,20 C100,60 60,100 100,140 C140,180 100,220 60,260" />
    <line x1="60" y1="20" x2="140" y2="20" stroke-opacity="0.4"/>
    <line x1="72" y1="85" x2="128" y2="85" stroke-opacity="0.5"/>
    <line x1="60" y1="140" x2="140" y2="140" stroke-opacity="0.6"/>
    <line x1="72" y1="195" x2="128" y2="195" stroke-opacity="0.4"/>
    <circle cx="60" cy="20" r="4" fill="#00CFFF" fill-opacity="0.8" stroke="none"/>
    <circle cx="140" cy="20" r="4" fill="#00CFFF" fill-opacity="0.8" stroke="none"/>
    <circle cx="100" cy="140" r="5" fill="#00FFDD" fill-opacity="0.7" stroke="none"/>
  </g>
  <!-- glowing nodes network center -->
  <g opacity="0.35">
    <circle cx="450" cy="130" r="60" fill="url(#glow1)"/>
    <line x1="300" y1="80" x2="450" y2="130" stroke="#00AAFF" stroke-width="0.8"/>
    <line x1="600" y1="80" x2="450" y2="130" stroke="#00AAFF" stroke-width="0.8"/>
    <line x1="350" y1="200" x2="450" y2="130" stroke="#00AAFF" stroke-width="0.8"/>
    <line x1="550" y1="200" x2="450" y2="130" stroke="#00AAFF" stroke-width="0.8"/>
    <circle cx="300" cy="80" r="4" fill="#00CFFF"/>
    <circle cx="600" cy="80" r="4" fill="#00CFFF"/>
    <circle cx="350" cy="200" r="4" fill="#00CFFF"/>
    <circle cx="550" cy="200" r="4" fill="#00CFFF"/>
    <circle cx="450" cy="130" r="6" fill="#00FFDD"/>
  </g>
</svg>
"""

PCOS_LOGO_SVG = """
<svg width="170" height="120" viewBox="0 0 340 240" xmlns="http://www.w3.org/2000/svg" aria-label="PCOS logo">
  <g transform="translate(0,8)">
    <path d="M45 158 C80 90, 165 140, 245 120 C220 145, 188 162, 160 174 C125 189, 85 192, 45 178 Z" fill="#0b58c6"/>
    <path d="M194 136 C235 124, 264 106, 285 77 C288 95, 282 112, 268 125 C251 141, 228 149, 194 151 Z" fill="#0b58c6"/>
    <path d="M170 141 C198 132, 219 116, 236 95 C239 110, 234 125, 222 136 C208 148, 190 154, 170 155 Z" fill="#3a82da"/>
    <path d="M205 55
             C205 33, 224 20, 241 20
             C258 20, 272 31, 272 48
             C272 85, 233 100, 215 120
             C195 99, 160 84, 160 53
             C160 33, 176 20, 194 20
             C200 20, 205 22, 210 26
             C212 35, 212 44, 205 55 Z" fill="#e04372"/>
    <rect x="208" y="53" width="16" height="42" rx="3" fill="#ffffff"/>
    <rect x="195" y="66" width="42" height="16" rx="3" fill="#ffffff"/>
  </g>
</svg>
"""

# Sidebar configuration
st.sidebar.markdown("### PCOS Detection System")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "Select Analysis Mode:",
    ["Home", "Clinical Parameters Analysis", "About PCOS", "How to Use"]
)

# Helper function to load or create model
@st.cache_resource
def load_or_create_model():
    """Load trained XGBoost model; train one if missing."""
    model_path = "best_xgboost_model.pkl"

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            loaded_obj = pickle.load(f)
        if hasattr(loaded_obj, "predict_proba") and hasattr(loaded_obj, "get_booster"):
            return loaded_obj, True

    # Train and load XGBoost model if no valid model is found.
    try:
        from train_model import train_model
        train_model(data_path="PCOS_data_without_infertility.xlsx", output_path=model_path)
        with open(model_path, 'rb') as f:
            loaded_obj = pickle.load(f)
        if hasattr(loaded_obj, "predict_proba") and hasattr(loaded_obj, "get_booster"):
            return loaded_obj, True
        return None, False
    except Exception:
        return None, False

# Load model
model, model_loaded = load_or_create_model()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_BANNER_CANDIDATES = [
    "assets/home-banner.jpg",
    "assets/home-banner.jpeg",
    "assets/home-banner.png",
]


def get_home_banner_image_path():
    """Return first available custom home banner image path, else None."""
    for rel_path in HOME_BANNER_CANDIDATES:
        abs_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(abs_path):
            return abs_path
    return None


def get_image_data_uri(path):
    """Return image file as data URI."""
    with open(path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower().replace(".", "")
    mime = "jpeg" if ext == "jpg" else ext
    return f"data:image/{mime};base64,{encoded}"


def _count_connected_components(binary_mask):
    """Return connected component stats from a binary mask."""
    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for y in range(h):
        for x in range(w):
            if not binary_mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            min_y = max_y = y
            min_x = max_x = x

            while stack:
                cy, cx = stack.pop()
                area += 1
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)

                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if binary_mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            components.append(
                {
                    "area": area,
                    "width": max_x - min_x + 1,
                    "height": max_y - min_y + 1,
                }
            )
    return components


def estimate_follicle_count_from_image(uploaded_file):
    """Estimate follicle count from one ultrasound image using blob heuristics."""
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("L")
    image.thumbnail((512, 512))
    arr = np.array(image, dtype=np.float32)

    p2, p98 = np.percentile(arr, [2, 98])
    arr = np.clip((arr - p2) / max(1e-6, p98 - p2), 0.0, 1.0)

    h, w = arr.shape
    y0, y1 = int(h * 0.06), int(h * 0.94)
    x0, x1 = int(w * 0.06), int(w * 0.94)
    roi = arr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0

    threshold = np.percentile(roi, 30)
    binary = roi < threshold
    components = _count_connected_components(binary)

    count = 0
    for comp in components:
        area = comp["area"]
        width = comp["width"]
        height = comp["height"]
        aspect = min(width, height) / max(width, height)
        if 20 <= area <= 1200 and aspect >= 0.45 and 4 <= width <= 60 and 4 <= height <= 60:
            count += 1

    return int(max(0, min(30, count)))


def estimate_left_right_follicles(uploaded_files):
    """Estimate left/right follicle counts from uploaded ultrasound images."""
    if not uploaded_files:
        return None, None

    counts = []
    for file in uploaded_files[:2]:
        try:
            counts.append(estimate_follicle_count_from_image(file))
        except Exception:
            counts.append(0)

    if len(counts) == 1:
        return counts[0], counts[0]
    return counts[0], counts[1]


def get_home_logo_markup():
    """Return circular home logo markup if available, else fallback SVG."""
    logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        return (
            f'<img src="data:image/png;base64,{encoded}" alt="PCOS Logo" '
            'style="width:140px;height:140px;border-radius:50%;object-fit:cover;display:block;'
            'border:3px solid #bfe9ff;box-shadow:0 4px 14px rgba(0,0,0,0.35);" />'
        )
    return PCOS_LOGO_SVG


def render_page_hero(title, subtitle=""):
    subtitle_html = f'<div class="page-hero-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="dna-dots"></div>
            <div class="dna-frame"></div>
            <div class="dna-svg-wrap">{DNA_HERO_SVG}</div>
            <div class="page-hero-content">
                <div class="page-hero-title">{title}</div>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_metric_banner_image():
    """Create a banner inspired by the awareness reference image."""
    img = Image.new("RGB", (1100, 420), "#F5F8FA")
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle((10, 10, 1090, 410), radius=24, fill="#EAF2F3")
    draw.ellipse((40, 30, 460, 390), fill="#D9E8E7")
    draw.ellipse((700, 20, 1050, 370), fill="#E0ECEB")

    uterus_outline = "#f2a8b7"
    uterus_fill = "#f8ccd6"
    draw.ellipse((465, 95, 535, 155), fill=uterus_fill, outline=uterus_outline, width=6)
    draw.polygon([(405, 125), (465, 128), (470, 148), (390, 172)], fill=uterus_fill, outline=uterus_outline)
    draw.polygon([(535, 128), (595, 125), (610, 172), (530, 148)], fill=uterus_fill, outline=uterus_outline)
    draw.rounded_rectangle((475, 150, 525, 255), radius=20, fill=uterus_fill, outline=uterus_outline, width=5)
    draw.rounded_rectangle((488, 252, 512, 318), radius=10, fill=uterus_fill, outline=uterus_outline, width=5)
    draw.ellipse((356, 157, 395, 196), fill="#f6d5de", outline=uterus_outline, width=4)
    draw.ellipse((605, 157, 644, 196), fill="#f6d5de", outline=uterus_outline, width=4)

    draw.polygon([(0, 370), (0, 420), (300, 420), (210, 370)], fill="#ede3f7")
    draw.polygon([(800, 370), (1100, 370), (1100, 420), (720, 420)], fill="#ede3f7")
    return img

# HOME PAGE
if app_mode == "Home":
    st.markdown(f"""
    <div class="dna-hero">
        <div class="dna-dots"></div>
        <div class="dna-frame"></div>
        <div class="dna-svg-wrap">{DNA_HERO_SVG}</div>
        <div class="dna-hero-content">
            <div class="home-logo-wrap">{get_home_logo_markup()}</div>
            <div class="dna-hero-title">AI Powered PCOS Detection</div>
            <div class="dna-hero-tagline">Empowering every woman to take charge of her reproductive health.</div>
            <div class="dna-hero-desc">
                An intelligent screening system for Polycystic Ovary Syndrome using clinical parameters,
                machine learning, and evidence-based diagnostics — accessible to everyone.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="rotterdam-box">
            <div class="rotterdam-main-title">According to Rotterdam Criteria:</div>
            <p class="rotterdam-desc">PCOS diagnosis requires at least 2 of:</p>
            <div class="rotterdam-steps">
                <div class="rotterdam-step">
                    <div class="rotterdam-step-num">01</div>
                    <div class="rotterdam-step-text">Irregular ovulation</div>
                </div>
                <div class="rotterdam-step">
                    <div class="rotterdam-step-num">02</div>
                    <div class="rotterdam-step-text">Hyperandrogenism</div>
                </div>
                <div class="rotterdam-step">
                    <div class="rotterdam-step-num">03</div>
                    <div class="rotterdam-step-text">Polycystic ovaries on ultrasound.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    home_banner_path = get_home_banner_image_path()
    logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
    if home_banner_path and os.path.exists(logo_path):
        banner_uri = get_image_data_uri(home_banner_path)
        logo_uri = get_image_data_uri(logo_path)
        st.markdown(
            f"""
            <div class="home-visual-wrap">
                <div class="home-banner-pane">
                    <img src="{banner_uri}" alt="Home banner" />
                </div>
                <div class="home-logo-zone">
                    <div class="home-logo-semicircle">
                        <img src="{logo_uri}" alt="System logo" />
                    </div>
                    <div class="home-ladder-item home-ladder-1">01  Non-Invasive Screening</div>
                    <div class="home-ladder-item home-ladder-2">02  Fast AI Results</div>
                    <div class="home-ladder-item home-ladder-3">03  Privacy Protected</div>
                    <div class="home-ladder-item home-ladder-4">04  Evidence-Based Insights</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(
            "Required images not found for home visual layout. Add: "
            "assets/home-banner.jpg (or .jpeg/.png) and assets/logo.png"
        )
        if home_banner_path:
            st.image(home_banner_path, width=480)
        else:
            st.image(create_metric_banner_image(), width=480)
    
    st.markdown("---")
    st.markdown("### System Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", "87-92%", "Based on validation set")
    with col2:
        st.metric("Model Features", "43", "Used internally by AI model")
    with col3:
        st.metric("User Inputs", "541", "Patient records")
    
    st.markdown("---")
    st.info("Ready to analyze? Select 'Clinical Parameters Analysis' from the sidebar to get started!")


# CLINICAL PARAMETERS ANALYSIS
elif app_mode == "Clinical Parameters Analysis":
    render_page_hero(
        "<span style='color:#FFFFFF;'>Clinical Parameters Analysis</span>",
        "Enter patient details for AI-powered screening."
    )
    st.markdown("---")
    if not model_loaded:
        st.error("Model could not be loaded. Please ensure the training data is available.")
    else:
        st.markdown("### Enter Patient Clinical Data")
        st.markdown("Simple input form for non-doctors. Fill what you know and keep defaults for missing values.")
        with st.container(border=True):
            st.markdown('<div class="section-title">Physical</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age (years)", 15, 50, 28)
                height_cm = st.slider("Height (cm)", 140, 200, 165)
                weight_kg = st.slider("Weight (kg)", 40, 150, 65)
                bmi = weight_kg / (height_cm / 100) ** 2
                st.markdown(f"**Calculated BMI:** {bmi:.2f}")
            with col2:
                waist_inch = st.slider("Waist (inches)", 20, 50, 30)
                hip_inch = st.slider("Hip (inches)", 25, 55, 38)
                waist_hip_ratio = waist_inch / hip_inch
                st.markdown(f"**Calculated Waist:Hip Ratio:** {waist_hip_ratio:.2f}")
                pulse = st.slider("Pulse (bpm)", 40, 120, 75)

        with st.container(border=True):
            st.markdown('<div class="section-title">Hormone and Biochemical from blood test report</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                fsh = st.slider("FSH (mIU/mL)", 1.0, 15.0, 6.5)
                lh = st.slider("LH (mIU/mL)", 1.0, 25.0, 8.0)
                amh = st.slider("AMH (ng/mL)", 0.0, 15.0, 3.5)
            with col2:
                testo = st.slider("Testosterone (ng/mL)", 0.0, 1.5, 0.5)
                insulin = st.slider("Insulin (U/mL)", 0.0, 25.0, 5.0)
                rbs = st.slider("RBS (mg/dL)", 70, 200, 100)

        with st.container(border=True):
            st.markdown('<div class="section-title">Lifestyle</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                acne = st.selectbox("Acne", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                hair_growth = st.selectbox("Excess Hair Growth", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                skin_darkening = st.selectbox("Skin Darkening", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            with col2:
                pimples = st.selectbox("Pimples", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                fast_food = st.selectbox("Irregular Periods", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                reg_exercise = st.selectbox("Regular Exercise", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        with st.container(border=True):
            st.markdown('<div class="section-title">Upload Ultrasound Images</div>', unsafe_allow_html=True)
            uploaded_usg = st.file_uploader(
                " ",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            follicles_l, follicles_r = estimate_left_right_follicles(uploaded_usg) if uploaded_usg else (8, 8)
            if uploaded_usg:
                st.caption(f"{len(uploaded_usg)} image(s) uploaded")
                preview_cols = st.columns(min(3, len(uploaded_usg)))
                for idx, file in enumerate(uploaded_usg[:3]):
                    with preview_cols[idx]:
                        st.image(file, caption=f"Ultrasound {idx + 1}", use_container_width=True)
                st.markdown(f"**Estimated Follicle Count (Left Ovary):** {follicles_l}")
                st.markdown(f"**Estimated Follicle Count (Right Ovary):** {follicles_r}")
                if len(uploaded_usg) == 1:
                    st.caption("Only one image uploaded; same estimate used for both ovaries.")
            else:
                st.info("Ultrasound image is optional. Using default follicle counts for analysis if not uploaded.")
        if st.button("Analyze Patient", type="primary", use_container_width=True):
            patient_data = {
                'Age': age,
                'Height(Cm) ': height_cm,
                'Weight (Kg)': weight_kg,
                'BMI': bmi,
                'Waist(inch)': waist_inch,
                'Hip(inch)': hip_inch,
                'Waist:Hip Ratio': waist_hip_ratio,
                'FSH(mIU/mL)': fsh,
                'LH(mIU/mL)': lh,
                'Testosterone(ng/mL)': testo,
                'AMH(ng/mL)': amh,
                'Acne': acne,
                'Hair growth(Y/N)': hair_growth,
                'Skin darkening (Y/N)': skin_darkening,
                'Pimples(Y/N)': pimples,
                'Fast food (Y/N)': fast_food,
                'Reg.Exercise(Y/N)': reg_exercise,
                'Follicle No. (L)': follicles_l,
                'Follicle No. (R)': follicles_r,
                'Pulse': pulse,
                'Insulin(U/mL)': insulin,
                'RBS(mg/dL)': rbs,
            }
            input_df = pd.DataFrame([patient_data])
            model_features = (
                model.get_booster().feature_names
                if hasattr(model, "get_booster")
                else list(input_df.columns)
            )
            for feature in model_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            input_df = input_df[model_features]
            try:
                probability = model.predict_proba(input_df)[0]
                prediction = model.predict(input_df)[0]
                st.markdown("---")
                st.markdown("### Analysis Results")
                col1, col2 = st.columns(2)
                with col1:
                    if prediction == 1:
                        st.markdown(
                            '<div class="warning-box"><h2>High Risk: PCOS Likely Detected</h2>'
                            f'<p>Confidence: <strong>{probability[1] * 100:.1f}%</strong></p></div>',
                            unsafe_allow_html=True
                        )
                        st.markdown("### Possible Next Suggestions")
                        st.markdown("""
                        - Book a gynecologist visit for confirmation.
                        - Share ultrasound images and follicle counts during consultation.
                        - Track cycle dates, sleep, and physical activity for 4-8 weeks.
                        - Start small lifestyle steps: daily walk, balanced meals, less sugary snacks.
                        """)
                    else:
                        st.markdown(
                            '<div class="success-box"><h2>Low Risk: No PCOS Detected</h2>'
                            f'<p>Confidence: <strong>{probability[0] * 100:.1f}%</strong></p></div>',
                            unsafe_allow_html=True
                        )
                        st.markdown("### Possible Next Suggestions")
                        st.markdown("""
                        - Continue healthy routine and regular check-ups.
                        - If symptoms continue (irregular periods, acne, weight changes), consult a doctor.
                        - Keep symptom notes to discuss clearly in clinic visits.
                        """)
                with col2:
                    st.markdown("### Probability Distribution")
                    col_a, col_b = st.columns(2)
                    col_a.metric("No PCOS", f"{probability[0] * 100:.1f}%")
                    col_b.metric("PCOS", f"{probability[1] * 100:.1f}%")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    categories = ['No PCOS', 'PCOS']
                    colors = ['#51CF66', '#FF6B6B']
                    ax.bar(categories, probability, color=colors)
                    ax.set_ylabel('Probability')
                    ax.set_ylim([0, 1])
                    st.pyplot(fig)
                st.markdown("### Common Questions (Simple Answers)")
                total_follicles = follicles_l + follicles_r
                faq_options = [
                    "What is follicle count?",
                    "How much follicle count is found in my report?",
                    "Does this result mean I definitely have PCOS?",
                    "What should I do next?"
                ]
                selected_q = st.selectbox("Choose a question", faq_options)
                if selected_q == "What is follicle count?":
                    st.info("Follicle count means the number of small sacs (follicles) seen in ovaries on ultrasound.")
                elif selected_q == "How much follicle count is found in my report?":
                    st.info(
                        f"Left ovary: {follicles_l}, Right ovary: {follicles_r}, Total: {total_follicles}. "
                        "Your doctor uses this with symptoms and blood tests."
                    )
                elif selected_q == "Does this result mean I definitely have PCOS?":
                    st.info("No. This tool is for screening. A doctor confirms diagnosis after full clinical evaluation.")
                elif selected_q == "What should I do next?":
                    st.info("Take this report to a gynecologist, share symptoms and cycle history, and follow medical advice.")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# ABOUT PCOS
elif app_mode == "About PCOS":
    render_page_hero(
        "<span style='color:#FFFFFF;'>About PCOS</span>",
        "Understand symptoms, risks, diagnosis, and early care."
    )
    st.markdown("---")
    st.markdown("""
    ## Understanding PCOS

    Polycystic Ovary Syndrome is a complex hormonal condition affecting millions of women worldwide.
    Polycystic Ovary Syndrome (PCOS) is a common endocrine disorder that affects reproductive-aged women.
    It is characterized by hormonal imbalances, irregular menstrual cycles, and the presence of small cysts on the ovaries.
    """)
    about_pcos_image_candidates = [
        "assets/AboutPCOS.png",
        "assets/ABoutPCOS.png",
        "assets/aboutpcos.png",
    ]
    about_pcos_image_path = None
    for rel_path in about_pcos_image_candidates:
        abs_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(abs_path):
            about_pcos_image_path = abs_path
            break
    if about_pcos_image_path:
        st.image(about_pcos_image_path, width=700)


# HOW TO USE
elif app_mode == "How to Use":
    render_page_hero(
        "<span style='color:#FFFFFF;'>How To Use</span>",
        "Step-by-Step Guide | PCOS Screening System"
    )

    steps = [
        {
            "title": "Step 1: Gather Clinical Data",
            "text": (
                "Collect patient clinical history and all available reports before entering values."
            )
        },
        {
            "title": "Step 2: Fill the Parameters",
            "text": (
                "Enter physical, biochemical, and clinical parameters in the input form."
            )
        },
        {
            "title": "Step 3: Upload Ultrasound Images",
            "text": (
                "Upload ultrasound images and add follicle details where available."
            )
        },
        {
            "title": "Step 4: Run AI Analysis",
            "text": (
                "Click Analyze to run the AI model on the entered data."
            )
        },
        {
            "title": "Step 5: Review Results",
            "text": (
                "Review risk score, confidence, and suggested next steps."
            )
        },
    ]

    for s in steps:
        st.markdown(
            f"""
            <div class="info-box howto-step">
                <h3>{s['title']}</h3>
                <p>{s['text']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <p><strong>PCOS Detection System</strong> - AI for Social Good</p>
    <p>Built for Hackathon | Privacy Protected | Evidence-Based</p>
    <p><small>Disclaimer: This is a screening tool, not a replacement for professional medical diagnosis.</small></p>
</div>
""", unsafe_allow_html=True)
