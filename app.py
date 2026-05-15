import cv2
import base64
import os

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""


def get_severity_hsv(image, pred_class):
    if pred_class == "Healthy":
        return "None (0.00%)"
        
    img_array = np.array(image)
    # Convert RGB/RGBA to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        return "Unknown"
        
    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
    
    # Healthy Green mask (approximate range for green leaves)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Entire Leaf mask (ignoring mostly background - low saturation/value)
    lower_leaf = np.array([0, 20, 20])
    upper_leaf = np.array([179, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)
    
    # Disease area is leaf area that is not healthy green
    disease_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(green_mask))
    
    leaf_area = cv2.countNonZero(leaf_mask)
    disease_area = cv2.countNonZero(disease_mask)
    
    if leaf_area == 0:
        return "Unknown"
        
    ratio = disease_area / leaf_area
    percentage = ratio * 100
    
    if ratio > 0.40:
        sev_level = "High"
    elif ratio > 0.15:
        sev_level = "Medium"
    else:
        sev_level = "Low"
        
    return f"{sev_level} ({percentage:.2f}%)"

disease_info = {
    "Brown_Spot": {
        "description": "Brown spot is a fungal disease caused by Bipolaris oryzae. It significantly affects crop yield and grain quality, often appearing in soils with poor fertility or nutrient deficiency.",
        "symptoms": "Typical symptoms include small, oval, dark-brown lesions on the leaves, which later develop a light brown or gray center.",
        "prevention": "Ensure balanced soil nutrition, use healthy certified seeds, and apply appropriate seed treatments before sowing.",
        "remedy": "Apply fungicides containing Mancozeb, Propiconazole, or Edifenphos. Improve soil fertility with proper N-P-K balance.",
        "buy_links": [
            {"name": "Mancozeb Fungicide", "link": "https://www.amazon.com/s?k=mancozeb+fungicide", "img": "product_images/mancozeb.png"},
            {"name": "Propiconazole Fungicide", "link": "https://www.amazon.com/s?k=propiconazole+fungicide", "img": "product_images/propiconazole.png"}
        ]
    },
    "Bacterial_Blight": {
        "description": "Bacterial leaf blight is a deadly bacterial disease caused by Xanthomonas oryzae. It can cause severe wilting and crop loss, especially in environments with high humidity and strong winds.",
        "symptoms": "Water-soaked streaks appear on leaf margins, expanding and turning yellow to grayish-white, eventually causing leaves to dry out.",
        "prevention": "Grow resistant varieties, ensure proper plant spacing, and avoid excessive nitrogen application which promotes the disease.",
        "remedy": "Copper-based bactericides (like Copper Oxychloride) or antibiotics like Streptocycline can help manage the spread. Drain the field to reduce humidity.",
        "buy_links": [
            {"name": "Copper Oxychloride", "link": "https://www.amazon.com/s?k=copper+oxychloride+fungicide", "img": "product_images/copper_oxychloride.png"},
            {"name": "Streptocycline", "link": "https://www.amazon.com/s?k=streptocycline+for+plants", "img": "product_images/streptocycline.png"}
        ]
    },
    "Leaf_Smut": {  
        "description": "Leaf smut is a minor fungal disease caused by Entyloma oryzae. It usually occurs late in the growing season and rarely causes significant yield losses, but affects the plant's photosynthetic capability.",
        "symptoms": "Small, slightly raised, black, powdery spots or lesions appear on both sides of the leaves.",
        "prevention": "Use clean, treated seeds and ensure field sanitation by removing infected crop debris after harvest.",
        "remedy": "Seed treatment with systemic fungicides. Foliar application of Propiconazole or Hexaconazole if the infection is severe.",
        "buy_links": [
            {"name": "Hexaconazole Fungicide", "link": "https://www.amazon.com/s?k=hexaconazole+fungicide", "img": "product_images/hexaconazole.png"},
            {"name": "Seed Treatment Fungicide", "link": "https://www.amazon.com/s?k=seed+treatment+fungicide", "img": "product_images/seed_treatment.png"}
        ]
    },
    "Healthy": {
        "description": "The plant appears perfectly healthy with no signs of fungal or bacterial infections.",
        "symptoms": "Leaves are uniformly green without any spots, lesions, or yellowing.",
        "prevention": "Continue maintaining good agricultural practices: proper watering, balanced fertilization, and regular field monitoring.",
        "remedy": "No chemical remedies required. Keep up the good work!",
        "buy_links": [
            {"name": "All Purpose Fertilizer", "link": "https://www.amazon.com/s?k=all+purpose+plant+fertilizer", "img": "product_images/fertilizer.png"},
            {"name": "Soil Testing Kit", "link": "https://www.amazon.com/s?k=soil+testing+kit", "img": "product_images/soil_test_kit.png"}
        ]
    }
}

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Rice Disease Detector", page_icon="🌾", layout="wide")

st.markdown("""
<style>
    /* Global background and font */
    .stApp {
        background: radial-gradient(circle at top, #0f2016, #060a08);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Title Styling */
    h1 {
        color: #00E676;
        text-align: center;
        font-weight: 900;
        letter-spacing: 1.5px;
        text-shadow: 0 0 20px rgba(0, 230, 118, 0.4);
    }
    
    /* Improve button appearance */
    .stDownloadButton button {
        background-color: #0a2e16 !important;
        color: #00E676 !important;
        border-radius: 8px !important;
        border: 1px solid #00E676 !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stDownloadButton button:hover {
        background-color: #00E676 !important;
        color: #060a08 !important;
        box-shadow: 0 0 25px rgba(0, 230, 118, 0.6) !important;
        transform: translateY(-2px);
    }

    /* Metric Cards Glassmorphism */
    div[data-testid="metric-container"] {
        background-color: rgba(15, 32, 22, 0.6);
        border: 1px solid rgba(0, 230, 118, 0.2);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        backdrop-filter: blur(10px);
    }
    div[data-testid="metric-container"] label {
        color: #a5d6a7 !important;
        font-weight: 600;
        font-size: 1.05em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #00E676 !important;
        text-shadow: 0 0 12px rgba(0, 230, 118, 0.4);
    }

    /* Tabs dark aesthetic styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(15, 32, 22, 0.6);
        border-radius: 8px 8px 0px 0px;
        border: 1px solid rgba(0, 230, 118, 0.2);
        border-bottom: none;
        padding: 12px 20px;
        color: #a5d6a7;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0a2e16;
        border-bottom: 3px solid #00E676;
        color: #00E676 !important;
        text-shadow: 0 0 8px rgba(0, 230, 118, 0.3);
    }

    /* Product images hover effect */
    .product-link img {
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    }
    .product-link:hover img {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(0, 230, 118, 0.4) !important;
        border: 1px solid rgba(0, 230, 118, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model("rice_disease_model.h5")

model = get_model()

class_names = ['Bacterial_Blight', 'Brown_Spot', 'Healthy', 'Leaf_Smut']

def preprocess_image(image):
    image = image.resize((224,224))
    image = np.array(image)/255.0
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    image = np.expand_dims(image, axis=0)
    return image

st.title("🌾 AI Rice Leaf Disease Detector")
st.markdown("<p style='text-align: center; color: #a5d6a7; font-size: 1.2em; margin-bottom: 30px;'>Upload a leaf image for instant AI diagnosis and actionable treatment plans.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("### 📸 Upload Leaf Image")
    uploaded_file = st.file_uploader("Select an image (JPG, PNG)", type=["jpg","png","jpeg"], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Preview", use_column_width=True)

with col2:
    if uploaded_file:
        with st.spinner('🔬 Analyzing plant biology...'):
            processed = preprocess_image(image)
            prediction = model.predict(processed)
        
        pred_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        severity = get_severity_hsv(image, pred_class)
        info = disease_info[pred_class]
        
        st.markdown("### 📊 AI Diagnosis Results")
        
        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Identified Condition", value=pred_class.replace("_", " "))
        m2.metric(label="AI Confidence", value=f"{confidence*100:.1f}%")
        m3.metric(label="Infection Severity", value=severity.split(" ")[0])
        
        st.progress(confidence, text="Confidence Score")
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 📋 Comprehensive Action Plan")
        
        # Tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["📖 Description", "⚠️ Symptoms", "🛡️ Prevention", "💊 Treatment"])
        
        with tab1:
            st.info(f"**Pathology:** {info['description']}")
        with tab2:
            st.warning(f"**Indicators:** {info['symptoms']}")
        with tab3:
            st.success(f"**Best Practices:** {info['prevention']}")
        with tab4:
            st.error(f"**Immediate Action:** {info['remedy']}")
            st.markdown("#### 🛒 Recommended Products")
            prod_cols = st.columns(len(info['buy_links']))
            for idx, prod in enumerate(info['buy_links']):
                with prod_cols[idx]:
                    img_b64 = get_base64_image(prod['img'])
                    if img_b64:
                        html_img = f'''
                        <a href="{prod['link']}" target="_blank" class="product-link">
                            <img src="data:image/png;base64,{img_b64}" style="width:100%; border-radius:10px; margin-bottom:10px; border: 1px solid rgba(0,230,118,0.2);">
                        </a>
                        <div style="text-align: center;">
                            <a href="{prod['link']}" target="_blank" style="color: #00E676; text-decoration: none; font-weight: bold; font-size: 1.1em; text-shadow: 0 0 5px rgba(0,230,118,0.3);">{prod['name']}</a>
                        </div>
                        '''
                        st.markdown(html_img, unsafe_allow_html=True)
                    else:
                        st.markdown(f"**[{prod['name']}]({prod['link']})**")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Generate PDF report
        from fpdf import FPDF
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt="Rice Leaf Disease Detection Report", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(40, 10, txt="Disease:", border=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, txt=str(pred_class), border=0, ln=True)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(40, 10, txt="Confidence:", border=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, txt=f"{confidence:.2f}", border=0, ln=True)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(40, 10, txt="Severity:", border=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, txt=str(severity), border=0, ln=True)
        
        pdf.ln(10)
        
        def add_pdf_section(title, content):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, txt=title, ln=True)
            pdf.set_font("Arial", '', 12)
            safe_content = content.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, txt=safe_content)
            pdf.ln(4)
            
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="Detailed Information", ln=True)
        
        add_pdf_section("Description:", info['description'])
        add_pdf_section("Symptoms:", info['symptoms'])
        add_pdf_section("Prevention:", info['prevention'])
        add_pdf_section("Recommended Remedy:", info['remedy'])
        
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="Recommended Products", ln=True)
        pdf.set_font("Arial", '', 12)
        for prod in info['buy_links']:
            safe_text = f"- {prod['name']}: {prod['link']}".encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, txt=safe_text)
            
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        
        st.download_button(
            label="📄 Download Full PDF Report",
            data=pdf_bytes,
            file_name=f"{pred_class}_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    