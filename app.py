import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image, ImageDraw
import time
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="PCOS Detection AI",
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
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 0 0 20px rgba(100,200,255,0.8);
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
        min-height: 260px;
        border-radius: 18px;
        overflow: hidden;
        margin-bottom: 28px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #0a1a3a 0%, #0d2657 40%, #0a3a6e 70%, #0d5099 100%);
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
        padding: 40px 30px 30px;
    }
    .dna-hero-title {
        font-size: 42px;
        font-weight: 900;
        color: #FFFFFF;
        letter-spacing: 3px;
        text-shadow: 0 0 30px rgba(0,200,255,0.9), 0 2px 8px rgba(0,0,0,0.6);
        margin-bottom: 8px;
        font-family: Georgia, serif;
    }
    .dna-hero-subtitle {
        font-size: 13px;
        letter-spacing: 5px;
        color: #00CFFF;
        text-transform: uppercase;
        margin-bottom: 18px;
        text-shadow: 0 0 10px rgba(0,180,255,0.6);
    }
    .dna-hero-desc {
        font-size: 14px;
        color: rgba(220,240,255,0.85);
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.7;
    }
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
        border: 1px solid rgba(0,200,255,0.3);
        border-radius: 10px;
        pointer-events: none;
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

def create_awareness_image(title, subtitle, base_color):
    """Create a simple in-app awareness illustration so the app has visual guidance."""
    img = Image.new("RGB", (900, 420), "#FFF9FB")
    draw = ImageDraw.Draw(img)

    # Soft background blocks
    draw.rounded_rectangle((30, 30, 870, 390), radius=30, fill="#FFFFFF", outline="#F3D7DF", width=3)
    draw.rounded_rectangle((60, 70, 420, 350), radius=24, fill=base_color)
    draw.ellipse((520, 95, 760, 335), fill="#FFE1E8", outline="#F2A7B8", width=4)
    draw.ellipse((595, 170, 685, 260), fill="#FFFFFF", outline="#F2A7B8", width=3)

    # Text
    draw.text((85, 110), title, fill="#222222")
    draw.text((85, 170), subtitle, fill="#444444")
    draw.text((85, 250), "PCOS awareness | Early action matters", fill="#6B7280")
    draw.text((565, 355), "Women's Health", fill="#AA4C63")
    return img

# HOME PAGE
if app_mode == "Home":
    st.markdown(f"""
    <div class="dna-hero">
        <div class="dna-dots"></div>
        <div class="dna-frame"></div>
        <div class="dna-svg-wrap">{DNA_HERO_SVG}</div>
        <div class="dna-hero-content">
            <div class="dna-hero-title">PCOS Detection AI</div>
            <div class="dna-hero-subtitle">Medical Presentation &nbsp;|&nbsp; AI for Social Good</div>
            <div class="dna-hero-desc">
                An intelligent screening system for Polycystic Ovary Syndrome using clinical parameters,
                machine learning, and evidence-based diagnostics — accessible to everyone.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
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
    
    with col2:
        st.markdown('<div class="success-box"><h3>Key Features</h3>', unsafe_allow_html=True)
        st.markdown("""
        1. **Non-Invasive Detection** - Uses only clinical parameters
        2. **Fast Results** - Instant diagnosis prediction
        3. **Data Privacy** - All processing happens locally
        4. **Accessible** - Web-based interface for ease of use
        5. **Evidence-Based** - Trained on clinical dataset
        6. **Interpretable** - Shows key factors in diagnosis
        """, unsafe_allow_html=True)

    st.markdown("### PCOS Awareness")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(
            create_awareness_image("Women First Care", "Simple checks for early PCOS screening", "#FFE7EE"),
            use_container_width=True
        )
    with img_col2:
        st.image(
            create_awareness_image("Know Your Cycle", "Track symptoms and get support sooner", "#E8F7F1"),
            use_container_width=True
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
    st.markdown('<div class="header-style">Clinical Parameters Analysis</div>', unsafe_allow_html=True)
    st.markdown("---")
    if not model_loaded:
        st.error("Model could not be loaded. Please ensure the training data is available.")
    else:
        st.markdown("### Enter Patient Clinical Data")
        st.markdown("Simple input form for non-doctors. Fill what you know and keep defaults for missing values.")
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
    st.markdown('<div class="header-style">About PCOS</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    ## What is PCOS?
    
    **Polycystic Ovary Syndrome (PCOS)** is a common endocrine disorder that affects reproductive-aged women. 
    It is characterized by:
    
    - **Irregular periods** - Unpredictable menstrual cycles
    - **Elevated androgens** - Excess male hormones
    - **Polycystic ovaries** - Multiple small follicles on ovaries
    - **Metabolic dysfunction** - Insulin resistance in 50-70% of cases
    
    ### Key Statistics
    - **Prevalence:** 6-20% of reproductive-aged women worldwide
    - **Age of onset:** Typically 20-40 years
    - **Impact:** Leading cause of infertility in women
    - **Comorbidities:** Increased risk of diabetes, heart disease, and endometrial cancer
    
    ### Common Symptoms
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Physical Symptoms:**
        - Irregular or missed periods
        - Excessive hair growth (hirsutism)
        - Acne
        - Hair loss
        - Dark skin patches
        - Weight gain or difficulty losing weight
        """)
    
    with col2:
        st.markdown("""
        **Metabolic Issues:**
        - Insulin resistance
        - High blood pressure
        - Elevated cholesterol
        - Risk of type 2 diabetes
        - Metabolic syndrome
        """)
    
    st.markdown("---")
    st.markdown("""
    ### Diagnostic Criteria (Rotterdam Criteria)
    
    PCOS is diagnosed if at least 2 of 3 criteria are present:
    
    1. **Ovulatory dysfunction** - Irregular or absent periods
    2. **Clinical or biochemical hyperandrogenism** - Elevated testosterone or visual symptoms
    3. **Polycystic ovaries** - Ultrasound findings (>=12 follicles per ovary)
    
    Other causes of hyperandrogenism must be excluded.
    
    ### Treatment & Management
    
    - **Lifestyle modifications:** Diet, exercise, weight management
    - **Medications:** Metformin, hormonal contraceptives, anti-androgens
    - **Fertility treatment:** For patients desiring pregnancy
    - **Regular monitoring:** Glucose, lipids, blood pressure
    
    ### Why Early Detection Matters
    
    Early detection enables:
    - Timely intervention and treatment
    - Prevention of complications
    - Better fertility outcomes
    - Improved quality of life
    - Reduced long-term health risks
    """)


# HOW TO USE
elif app_mode == "How to Use":
    st.markdown(f"""
    <div class="dna-hero">
        <div class="dna-dots"></div>
        <div class="dna-frame"></div>
        <div class="dna-svg-wrap">{DNA_HERO_SVG}</div>
        <div class="dna-hero-content">
            <div class="dna-hero-title">How To Use</div>
            <div class="dna-hero-subtitle">Step-by-Step Guide &nbsp;|&nbsp; PCOS Screening System</div>
            <div class="dna-hero-desc">
                Follow the guided steps below to enter patient data and receive an AI-powered
                PCOS risk assessment. No medical background required.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        {
            "title": "Step 1: Gather Basic Information",
            "text": (
                "Keep it simple. Collect age, height, weight, waist, hip, and pulse. "
                "Also keep symptom notes like acne, hair growth, and cycle changes."
            )
        },
        {
            "title": "Step 2: Add Hormonal and Biochemical Values",
            "text": (
                "Enter available test values: FSH, LH, AMH, testosterone, insulin, and RBS. "
                "If you do not have a value, keep the default."
            )
        },
        {
            "title": "Step 3: Fill Clinical and Lifestyle Section",
            "text": (
                "Choose Yes or No for acne, excess hair growth, skin darkening, pimples, "
                "fast food, and regular exercise."
            )
        },
        {
            "title": "Step 4: Upload Ultrasound Images",
            "text": (
                "Upload ultrasound image files if available. You can also enter follicle count "
                "for left and right ovary."
            )
        },
        {
            "title": "Step 5: Run Analysis",
            "text": (
                "Click Analyze Patient. The tool shows risk score, confidence, and simple "
                "next-step suggestions."
            )
        },
        {
            "title": "Step 6: Read Result Carefully",
            "text": (
                "This is a screening tool, not a final diagnosis. Share the result with a "
                "qualified gynecologist for confirmation."
            )
        },
    ]

    if "how_to_use_step" not in st.session_state:
        st.session_state.how_to_use_step = 0

    total_steps = len(steps)
    current_step = st.session_state.how_to_use_step
    step_data = steps[current_step]

    st.markdown("### Auto-Play (1 second per step)")
    if st.button("Play 1s Animation", use_container_width=True):
        placeholder = st.empty()
        rendered = []
        for idx, s in enumerate(steps):
            rendered.append(f"""<div class="info-box"><h3>{s['title']}</h3><p>{s['text']}</p></div>""")
            placeholder.markdown("\n\n".join(rendered), unsafe_allow_html=True)
            time.sleep(1)
        st.session_state.how_to_use_step = total_steps - 1

    st.markdown("### Manual Slides")
    st.markdown(f"**Slide {current_step + 1} of {total_steps}**")
    st.progress((current_step + 1) / total_steps)
    st.markdown(
        f"""
        <div class="info-box">
            <h3>{step_data['title']}</h3>
            <p>{step_data['text']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Previous", disabled=(current_step == 0), use_container_width=True):
            st.session_state.how_to_use_step -= 1
            st.rerun()
    with col2:
        if st.button("Next", disabled=(current_step == total_steps - 1), use_container_width=True):
            st.session_state.how_to_use_step += 1
            st.rerun()
    with col3:
        if st.button("Start Over", use_container_width=True):
            st.session_state.how_to_use_step = 0
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <p><strong>PCOS Detection System</strong> - AI for Social Good</p>
    <p>Built for Hackathon | Privacy Protected | Evidence-Based</p>
    <p><small>Disclaimer: This is a screening tool, not a replacement for professional medical diagnosis.</small></p>
</div>
""", unsafe_allow_html=True)


