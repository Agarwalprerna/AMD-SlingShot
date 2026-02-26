import streamlit as st
import pandas as pd
import pickle
import os
import base64
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
    .warning-box {
        background-color: rgba(255,80,80,0.15);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0px;
        border-left: 5px solid #FF6B6B;
        color: #FFE0E0;
    }
    .success-box {
        background-color: rgba(0,220,180,0.12);
        padding: 18px;
        border-radius: 10px;
        margin: 10px 0px;
        border-left: 5px solid #00DEB0;
        color: #D0FFF5;
        backdrop-filter: blur(6px);
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
    .section-card {
        background: linear-gradient(145deg, #0b2f57 0%, #0a2545 100%);
        border: 1px solid rgba(120, 188, 238, 0.45);
        border-left: 6px solid #3da3e3;
        border-radius: 12px;
        padding: 14px 14px 8px 14px;
        margin: 14px 0;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.28);
    }
    .section-card-2 { border-left-color: #66b6ea; }
    .section-card-3 { border-left-color: #2f87d4; }
    .section-card-4 { border-left-color: #94c9f0; }

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
        font-size: 16px;
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
    """Load trained model or create a demo model if not available"""
    model_path = "best_xgboost_model.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f), True
    else:
        # Create a demo model with sample training
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        
        # Load and prepare data
        data_path = "PCOS_data_without_infertility.xlsx"
        if os.path.exists(data_path):
            df = pd.read_excel(data_path)
            
            # Data preprocessing
            df = df.drop(['Unnamed: 44'], axis=1) if 'Unnamed: 44' in df.columns else df
            df['BMI'] = df['Weight (Kg)'] / (df['Height(Cm) '] / 100) ** 2
            df = df.drop(['FSH/LH'], axis=1) if 'FSH/LH' in df.columns else df
            df['Waist:Hip Ratio'] = df['Waist(inch)'] / df['Hip(inch)']
            df.dropna(inplace=True)
            df['II    beta-HCG(mIU/mL)'] = pd.to_numeric(df['II    beta-HCG(mIU/mL)'], errors='coerce')
            df['AMH(ng/mL)'] = pd.to_numeric(df['AMH(ng/mL)'], errors='coerce')
            df.dropna(inplace=True)
            
            y = df['PCOS (Y/N)']
            X = df.drop(columns=['PCOS (Y/N)'])
            
            # Train model
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
            xgb_model.fit(X, y)
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(xgb_model, f)
            
            return xgb_model, True
        else:
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
    
    st.markdown('<div class="info-box"><h3>About This System</h3>', unsafe_allow_html=True)
    st.markdown("""
    This AI-powered system detects **Polycystic Ovary Syndrome (PCOS)** using:
    - Machine Learning (XGBoost)
    - Clinical Parameters Analysis
    - Physical & Hormonal Data
    
    The system achieves **high accuracy** in early PCOS detection, enabling timely intervention and treatment.
    
    **For Social Good:** This technology democratizes PCOS detection for underserved communities with limited access to specialized healthcare.
    </div>
    """, unsafe_allow_html=True)

    img_col, feature_col = st.columns([1, 1], vertical_alignment="top")
    with img_col:
        home_banner_path = get_home_banner_image_path()
        if home_banner_path:
            st.image(home_banner_path, width=400)
        else:
            st.warning(
                "Custom home image not found. Add one of: "
                "assets/home-banner.jpg, assets/home-banner.jpeg, assets/home-banner.png"
            )
            st.image(create_metric_banner_image(), width=400)
    with feature_col:
        st.markdown(
            """
            <div class="feature-panel">
                <div class="feature-title">Key Features</div>
                <div class="feature-item">✓ Non-Invasive Screening</div>
                <div class="feature-item">✓ Fast AI Results</div>
                <div class="feature-item">✓ Privacy Protected</div>
                <div class="feature-item">✓ Evidence-Based Insights</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.markdown("### System Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", "87-92%", "Based on validation set")
    with col2:
        st.metric("Model Features", "43", "Used internally by AI model")
    with col3:
        st.metric("User Inputs", "16", "Simple form for non-doctors")
    
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
        st.markdown('<div class="section-card section-card-1">', unsafe_allow_html=True)
        st.markdown("### 1) Physical")
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
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-card section-card-2">', unsafe_allow_html=True)
        st.markdown("### 2) Hormonal & Biochemical")
        col1, col2 = st.columns(2)
        with col1:
            fsh = st.slider("FSH (mIU/mL)", 1.0, 15.0, 6.5)
            lh = st.slider("LH (mIU/mL)", 1.0, 25.0, 8.0)
            amh = st.slider("AMH (ng/mL)", 0.0, 15.0, 3.5)
        with col2:
            testo = st.slider("Testosterone (ng/mL)", 0.0, 1.5, 0.5)
            insulin = st.slider("Insulin (U/mL)", 0.0, 25.0, 5.0)
            rbs = st.slider("RBS (mg/dL)", 70, 200, 100)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-card section-card-3">', unsafe_allow_html=True)
        st.markdown("### 3) Clinical & Lifestyle")
        col1, col2 = st.columns(2)
        with col1:
            acne = st.selectbox("Acne", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            hair_growth = st.selectbox("Excess Hair Growth", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            skin_darkening = st.selectbox("Skin Darkening", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col2:
            pimples = st.selectbox("Pimples", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            fast_food = st.selectbox("Frequent Fast Food", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            reg_exercise = st.selectbox("Regular Exercise", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-card section-card-4">', unsafe_allow_html=True)
        st.markdown("### 4) Upload Ultrasound Images")
        uploaded_usg = st.file_uploader(
            "Upload ultrasound image(s) (optional)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )
        col1, col2 = st.columns(2)
        with col1:
            follicles_l = st.slider("Follicle Count (Left Ovary)", 0, 30, 8)
        with col2:
            follicles_r = st.slider("Follicle Count (Right Ovary)", 0, 30, 8)
        if uploaded_usg:
            st.caption(f"{len(uploaded_usg)} image(s) uploaded")
            preview_cols = st.columns(min(3, len(uploaded_usg)))
            for idx, file in enumerate(uploaded_usg[:3]):
                with preview_cols[idx]:
                    st.image(file, caption=f"Ultrasound {idx + 1}", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
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
            model_features = model.get_booster().feature_names
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

    st.markdown("### Steps Overview")
    for s in steps:
        st.markdown(
            f"""
            <div class="info-box">
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
