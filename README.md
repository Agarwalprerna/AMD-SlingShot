# 🏥 End-to-End PCOS Detection Using AI - MVP Prototype

> **AI for Social Good - Healthcare Product**  
> A comprehensive, web-based PCOS detection system making early diagnosis accessible to underserved communities.

## 🎯 Overview

This project presents an end-to-end solution for detecting **Polycystic Ovary Syndrome (PCOS)** using:
- **Machine Learning (XGBoost)** - For clinical parameter-based detection
- **Deep Learning (CNNs)** - For ultrasound image analysis
- **Interactive Web Interface** - For easy accessibility and use

The system achieves **87-92% accuracy** in PCOS detection, enabling early intervention and better health outcomes.

### 🌍 Social Impact
PCOS affects 6-20% of women globally. Early detection is crucial as it's a leading cause of infertility and increases risks of diabetes and heart disease. This AI-powered system democratizes PCOS screening for communities with limited access to specialized healthcare.

## 🚀 Live Demo

**Try the PCOS Detection System:** [Streamlit Cloud Link] *(Deploy link will be provided)*

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
  - Metabolic markers (BMI, Insulin, Blood glucose, Cholesterol)
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
     - Input 43 clinical parameters
     - Get instant PCOS risk prediction
     - View confidence scores and recommendations
   
   - **Batch Analysis:**
     - Upload CSV file with multiple patient records
     - Get predictions for all patients
     - Download results in CSV format

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

#### Metabolic Markers
- Insulin level
- RBS (Random Blood Sugar)
- HbA1c
- Total Cholesterol
- Vitamin D3

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

## 📈 Model Performance

### Cross-Validation Results (5-Fold)
```
Average Accuracy:  87.43% ± 2.15%
Average Precision: 86.91% ± 2.89%
Average Recall:    87.12% ± 1.98%
Average F1-Score:  87.01% ± 2.10%
Average ROC-AUC:   0.9125 ± 0.0234
```

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

## 👥 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📞 Contact & Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact the development team

## 🙏 Acknowledgments

- Dataset contributors and clinical partners
- Open-source community (Streamlit, XGBoost, TensorFlow)
- Hackathon organizers and reviewers

---

## 📊 Data Availability

- **Clinical Dataset:** `PCOS_data_without_infertility.xlsx` (included in repository)
- **Ultrasound Images Dataset:** Available at [Figshare](https://figshare.com/articles/dataset/PCOS_Dataset/27682557?file=50407062)

## 🔍 Detailed Notebooks

### 1. PCOS Detection Based on Physical and Clinical Parameters
This notebook explores machine learning classification using XGBoost with clinical parameters for PCOS detection.

### 2. deepseek-for-pcos-detection
Advanced deep learning approach using DeepSeek framework for analyzing ultrasound images.

### 3. pcos-detection-from-usg-images
CNN-based analysis of ultrasound images with comprehensive preprocessing and augmentation.

---

**Made with ❤️ for healthcare accessibility | AI for Social Good**
