# 🏥 AI Powered PCOS Detection - MVP Prototype

> **AI for Social Good - Healthcare Product**  
> A comprehensive, web-based PCOS detection system making early diagnosis accessible to underserved communities.
> Built for Hackathon AMD-Slingshot 2026.

## Developers

- Prerna Agarwal
- Yash Patel

## 🎯 Overview

This project presents an end-to-end solution for detecting **Polycystic Ovary Syndrome (PCOS)** using:
- **Machine Learning (XGBoost)** - For clinical parameter-based detection
- **Deep Learning (CNNs)** - For ultrasound image analysis
- **Interactive Web Interface** - For easy accessibility and use

The system achieves **87-92% accuracy** in PCOS detection, enabling early intervention and better health outcomes.

### 🌍 Social Impact
PCOS affects 6-20% of women globally. Early detection is crucial as it's a leading cause of infertility and increases risks of diabetes and heart disease. This AI-powered system democratizes PCOS screening for communities with limited access to specialized healthcare.

## 🚀 Live Demo

**Try the PCOS Detection System:** [Live App](https://amd-slingshotgit-metgkzbhgixncfnpgbcczu.streamlit.app/)

## 📁 Repository Structure

```
├── app.py                                          # Main Streamlit web application
├── train_model.py                                  # Model training script
├── requirements.txt                                # Python dependencies
├── PCOS_data_without_infertility.xlsx             # Clinical dataset (541 samples)
├── PCOS Detection Based on Physical and Clinical Parameters.ipynb
├── deepseek-for-pcos-detection.ipynb
├── pcos-detection-from-usg-images.ipynb
├── .streamlit/config.toml                          # Streamlit configuration
├── runtime.txt                                     # Python version specification
├── Procfile                                        # Deployment configuration
├── setup.sh                                        # Setup script
└── README.md
```

## 🔬 Technical Approach

### Clinical Parameter-Based Detection
- **Model:** XGBoost with nested cross-validation
- **Features:** 43 clinical parameters including:
  - Hormonal levels (FSH, LH, Testosterone, AMH, etc.)
  - Physical measurements (Waist, Hip, Blood pressure)
  - Symptoms (Acne, Hair loss, Skin darkening, etc.)
- **Performance:**
  - Accuracy: 87-92%
  - Precision: 85-90%
  - Recall: 85-90%
  - ROC-AUC: 0.90+

### Image-Based Detection
- **Model:** Convolutional Neural Networks (CNN)
- **Input:** Ultrasound (USG) images
- **Method:** 
  - Image preprocessing and augmentation
  - Transfer learning with ImageNet weights
  - Binary classification (PCOS vs Non-PCOS)
- **Why CNN is used for image processing:**
  - CNNs learn local visual patterns (edges, textures, cyst-like circular regions) directly from pixel grids.
  - Shared convolution filters reduce parameters compared to fully connected networks, which improves learning on limited medical image datasets.
  - Pooling and deep feature hierarchies make the model more robust to small shifts, scale changes, and ultrasound noise/speckle.
  - Transfer learning allows starting from general visual features learned on large datasets, then adapting to domain-specific USG patterns with fewer labeled samples.
  - Compared with hand-crafted image features, CNNs provide an end-to-end pipeline that usually gives better generalization for binary medical image classification tasks.

## ⚡ Quick Start

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/PCOS-Detection.git
   cd PCOS-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (optional - pre-trained model will be auto-loaded)
   ```bash
   python train_model.py
   ```

4. **Run the web application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

## 💻 How to Use

### Web Interface Features

#### 1. **Home Page**
   - Overview of the PCOS detection system
   - Key statistics and features
   - Quick navigation to analysis tools

#### 2. **Clinical Parameters Analysis**
   - **Individual Patient Analysis:**
     - Input  clinical parameters
     - Get instant PCOS risk prediction
     - View confidence scores and recommendations

#### 3. **About PCOS**
   - Medical information about PCOS
   - Symptoms and risk factors
   - Diagnostic criteria
   - Treatment options

#### 4. **How to Use**
   - Step-by-step guide for using the system
   - Data requirements and format
   - Interpretation of results

### Input Parameters Required

#### Physical Measurements
- Age, Height, Weight
- Waist and Hip circumference
- Blood pressure (Systolic/Diastolic)
- Pulse

#### Hormonal Parameters
- FSH (Follicle Stimulating Hormone)
- LH (Luteinizing Hormone)
- Testosterone
- Free Androgen Index
- AMH (Anti-Müllerian Hormone)
- Beta-HCG
- Prolactin
- TSH (Thyroid Stimulating Hormone)

#### Clinical Symptoms (Yes/No)
- Acne, Hair growth, Hair loss
- Skin darkening, Pimples
- Regular exercise, Fast food consumption
- Follicle count (Left/Right ovary)

## 📊 Dataset Information

- **Size:** 541 patient records
- **Features:** 43 clinical parameters
- **Classes:** PCOS (positive) and Non-PCOS (negative)
- **Source:** Clinical dataset (PCOS_data_without_infertility.xlsx)
- **USG Images:** Available at [Figshare Dataset](https://figshare.com/articles/dataset/PCOS_Dataset/27682557?file=50407062)


### Data Availability

- **Clinical Dataset:** `PCOS_data_without_infertility.xlsx` (included in repository)
- **Ultrasound Images Dataset:** Available at [Figshare](https://figshare.com/articles/dataset/PCOS_Dataset/27682557?file=50407062)
- **Ultrasound Images Dataset (Additional):** [Kaggle - PCOS Detection Using Ultrasound Images](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images)

## 📈 Model Performance

### Training Results (5-Fold Nested Cross-Validation)

#### 5-Fold Training Progress
![5-Fold Training](assets/5fold%20training.png)

#### Cross-Validation Results
![Cross-Validation Results](assets/cross-validation%20results.png)



### Top 10 Most Important Features
1. Insulin level
2. Follicle count
3. Prolactin
4. Testosterone
5. LH
6. FSH
7. BMI
8. Waist:Hip Ratio
9. RBS (Glucose)
10. AMH

## 🚢 Deployment

### Deploy to Streamlit Cloud (Free)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add PCOS detection app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [streamlit.io/cloud](https://streamlit.io/cloud)
   - Connect your GitHub repository
   - Select this repo and `app.py` as main file
   - Click "Deploy"

### Deploy to Heroku

1. **Install Heroku CLI**
   ```bash
   # Already included: Procfile and setup.sh
   ```

2. **Deploy**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Deploy to AWS, Azure, or Google Cloud

- Use Docker containerization
- Configure for production deployment
- Add SSL certificates and security measures

## ⚠️ Important Disclaimers

- **This is a screening tool**, not a replacement for professional medical diagnosis
- Always consult with a qualified healthcare provider (Gynecologist/Endocrinologist) for confirmation
- Results should be used alongside clinical judgment and ultrasound imaging
- Maintain data privacy and HIPAA compliance when handling patient data
- The model is trained on a specific clinical cohort and may have different performance in other populations

## 📚 Research Papers & References

The project is based on:
- Rotterdam Criteria for PCOS diagnosis
- Clinical studies on PCOS pathophysiology
- Machine learning approaches in healthcare
- Deep learning for medical image analysis
- CystNet: An AI-driven model for PCOS detection using multilevel thresholding of ultrasound images: https://www.researchgate.net/publication/385177794_CystNet_An_AI_driven_model_for_PCOS_detection_using_multilevel_thresholding_of_ultrasound_images
- International evidence-based guideline for the assessment and management of PCOS (Monash): https://www.monash.edu/medicine/mchri/pcos/guideline
- Revised 2003 consensus on diagnostic criteria and long-term health risks related to PCOS: https://academic.oup.com/humrep/article/32/2/261/2452298


## 🔍 Detailed Notebooks

### 1. PCOS Detection Based on Physical and Clinical Parameters
Primary notebook for **structured clinical-data modeling**.
- Loads and cleans `PCOS_data_without_infertility.xlsx`
- Engineers important features (for example BMI and Waist:Hip ratio)
- Trains and tunes XGBoost using nested cross-validation
- Evaluates metrics and extracts feature importance
- Saves the trained clinical model used by the app pipeline

Use this notebook when the input is lab values, vitals, and symptom variables (not ultrasound images).

### 2. deepseek-for-pcos-detection
Prototype notebook for **LLM-assisted clinical interpretation workflow**.
- Uses a `transformers` pipeline and prompt generation for patient context
- Demonstrates how PCOS-related patient fields can be converted into model prompts
- Useful for experimentation with explanation/assistant-style outputs

Use this notebook for language-model prompt/testing experiments, not for training the final image classifier.

### 3. pcos-detection-from-usg-images
Primary notebook for **ultrasound image classification**.
- Organizes USG image folders into train/validation/test splits
- Builds and trains the CNN model for PCOS vs Non-PCOS classification
- Tracks training/validation curves and evaluates test performance
- Exports the trained image model (`pcos_cnn_model.h5`)

Use this notebook when working with ovarian ultrasound images and image-model training/evaluation.

---

**Made with ❤️ for healthcare accessibility | AI for Social Good**



## 👥 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

##  Contact & Support

`r`n
## 🙏 Acknowledgments

- Dataset contributors and clinical partners
- Open-source community (Streamlit, XGBoost, TensorFlow)
- AMD Hackathon organizers and reviewers

---

