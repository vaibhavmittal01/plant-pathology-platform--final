from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import matplotlib.pyplot as plt
import io
from datetime import datetime
from scraper import AgriculturalScraper

app = Flask(__name__)

# Initialize the scraper
scraper = AgriculturalScraper()

# Load the model
model = tf.keras.models.load_model('model.h5')
class_names = ['Apple scab', 'Apple Black rot', 'Apple Cedar rust', 'Healthy Apple',
               'Cherry Powdery mildew', 'Healthy Cherry',
               'Corn Cercospora leaf spot Gray leaf spot', 'Corn(maize) Common rust',
               'Corn(maize) Northern Leaf Blight', 'Corn(maize) Healthy', 'Grape Black rot',
               'Grape Esca(Black Measles)', 'Grape Leaf blight (Isariopsis Leaf Spot)', 'Grape Healthy',
               'Peach Bacterial spot', 'Peach Healthy', 'Pepper bell Bacterial spot', 'Pepper bell Healthy',
               'Potato Early blight', 'Potato Late blight', 'Potato Healthy', 'Strawberry Leaf scorch',
               'Strawberry Healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
               'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites Two-spotted spider mite',
               'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato  mosaic virus',
               'Tomato Healthy']

# Disease descriptions
disease_data = {
    "Apple scab": {
        "description": "A fungal disease that causes dark, scabby lesions on apple leaves, fruit, and twigs.",
        "stage": "Early to Mid Growth Stage"
    },
    "Apple Black rot": {
        "description": "A fungal infection leading to black, circular lesions on apples and leaves.",
        "stage": "Mid to Late Growth Stage"
    },
    "Apple Cedar rust": {
        "description": "A fungal disease forming orange spore masses, affecting apple trees and junipers.",
        "stage": "Early Growth Stage"
    },
    "Healthy Apple": {
        "description": "No disease detected. The apple plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Cherry Powdery mildew": {
        "description": "A fungal infection that appears as a white powdery coating on cherry leaves and fruit.",
        "stage": "Flowering and Fruit Development Stage"
    },
    "Healthy Cherry": {
        "description": "No disease detected. The cherry plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Corn Cercospora leaf spot Gray leaf spot": {
        "description": "A fungal disease causing grayish leaf spots, leading to reduced photosynthesis.",
        "stage": "Vegetative Stage"
    },
    "Corn(maize) Common rust": {
        "description": "A fungal disease producing reddish-brown pustules on corn leaves.",
        "stage": "Vegetative to Reproductive Stage"
    },
    "Corn(maize) Northern Leaf Blight": {
        "description": "A fungal infection causing cigar-shaped lesions, leading to yield loss.",
        "stage": "Mid to Late Growth Stage"
    },
    "Corn(maize) Healthy": {
        "description": "No disease detected. The corn plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Grape Black rot": {
        "description": "A fungal disease causing dark spots on leaves and shriveling fruit.",
        "stage": "Fruit Development Stage"
    },
    "Grape Esca(Black Measles)": {
        "description": "A disease that affects grapevines, leading to black streaks and wilting.",
        "stage": "Mid to Late Growth Stage"
    },
    "Grape Leaf blight (Isariopsis Leaf Spot)": {
        "description": "A fungal disease causing irregular leaf spots and defoliation.",
        "stage": "Early to Mid Growth Stage"
    },
    "Grape Healthy": {
        "description": "No disease detected. The grape plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Peach Bacterial spot": {
        "description": "A bacterial infection causing sunken, dark lesions on peach fruits and leaves.",
        "stage": "Early to Mid Growth Stage"
    },
    "Peach Healthy": {
        "description": "No disease detected. The peach plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Pepper bell Bacterial spot": {
        "description": "A bacterial disease causing water-soaked lesions on leaves and fruits.",
        "stage": "Vegetative to Fruit Development Stage"
    },
    "Pepper bell Healthy": {
        "description": "No disease detected. The pepper plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Potato Early blight": {
        "description": "A fungal disease causing dark concentric rings on potato leaves.",
        "stage": "Early to Mid Growth Stage"
    },
    "Potato Late blight": {
        "description": "A severe fungal disease causing large, water-soaked lesions leading to crop loss.",
        "stage": "Late Growth Stage"
    },
    "Potato Healthy": {
        "description": "No disease detected. The potato plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Strawberry Leaf scorch": {
        "description": "A fungal disease causing brown, dried leaf edges, reducing fruit yield.",
        "stage": "Mid to Late Growth Stage"
    },
    "Strawberry Healthy": {
        "description": "No disease detected. The strawberry plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Tomato Bacterial spot": {
        "description": "A bacterial infection causing water-soaked spots on tomato leaves and fruit.",
        "stage": "Early to Mid Growth Stage"
    },
    "Tomato Early blight": {
        "description": "A fungal disease causing dark, target-like spots on lower tomato leaves.",
        "stage": "Mid to Late Growth Stage"
    },
    "Tomato Late blight": {
        "description": "A severe fungal disease causing large, dark lesions on leaves and stems.",
        "stage": "Late Growth Stage"
    },
    "Tomato Leaf Mold": {
        "description": "A fungal disease causing yellow spots on leaves, leading to mold growth.",
        "stage": "Mid to Late Growth Stage"
    },
    "Tomato Septoria leaf spot": {
        "description": "A fungal infection causing small, circular, brown spots on tomato leaves.",
        "stage": "Early to Mid Growth Stage"
    },
    "Tomato Spider mites Two-spotted spider mite": {
        "description": "An infestation of tiny spider mites causing leaf bronzing and defoliation.",
        "stage": "All Growth Stages"
    },
    "Tomato Target Spot": {
        "description": "A fungal disease causing circular leaf lesions with a dark center.",
        "stage": "Early to Mid Growth Stage"
    },
    "Tomato Tomato Yellow Leaf Curl Virus": {
        "description": "A viral disease causing yellow, curled leaves and stunted growth.",
        "stage": "Early Growth Stage"
    },
    "Tomato mosaic virus": {
        "description": "A viral infection leading to mottled, yellowed tomato leaves.",
        "stage": "Seedling to Vegetative Stage"
    },
    "Tomato Healthy": {
        "description": "No disease detected. The tomato plant is in a healthy condition.",
        "stage": "All Growth Stages"
    }
}

# Disease descriptions
disease_remedies = {
    "Apple scab": {
        "causes": "Fungus (Venturia inaequalis) that overwinters in fallen leaves",
        "remedies": [
            "Remove and destroy fallen leaves",
            "Apply fungicides during early spring",
            "Plant resistant varieties",
            "Maintain good air circulation"
        ],
        "irrigation": "Drip irrigation to keep foliage dry",
        "fertilizers": ["Nitrogen-rich fertilizers", "Balanced NPK (10-10-10)"],
        "prevention": "Regular pruning and thinning of branches"
    },
    "Apple Black rot": {
        "causes": "Fungus (Botryosphaeria obtusa) that infects through wounds",
        "remedies": [
            "Remove infected branches and fruit",
            "Apply copper-based fungicides",
            "Maintain proper tree spacing",
            "Practice good sanitation"
        ],
        "irrigation": "Avoid overhead irrigation",
        "fertilizers": ["Calcium-rich fertilizers", "Balanced NPK (12-12-12)"],
        "prevention": "Regular inspection and pruning"
    },
    "Apple Cedar rust": {
        "causes": "Fungus (Gymnosporangium juniperi-virginianae) that requires juniper hosts",
        "remedies": [
            "Remove nearby juniper trees",
            "Apply fungicides during spring",
            "Plant resistant varieties",
            "Maintain good air circulation"
        ],
        "irrigation": "Drip irrigation preferred",
        "fertilizers": ["Potassium-rich fertilizers", "Balanced NPK (10-10-10)"],
        "prevention": "Regular monitoring and early treatment"
    },
    "Healthy Apple": {
        "causes": "No disease present",
        "remedies": [
            "Maintain current care practices",
            "Regular monitoring",
            "Proper watering schedule",
            "Balanced fertilization"
        ],
        "irrigation": "Regular watering schedule",
        "fertilizers": ["Balanced NPK (10-10-10)", "Organic compost"],
        "prevention": "Continue preventive measures"
    },
    "Cherry Powdery mildew": {
        "causes": "Fungus (Podosphaera clandestina) that thrives in humid conditions",
        "remedies": [
            "Apply sulfur-based fungicides",
            "Improve air circulation",
            "Remove infected leaves",
            "Use resistant varieties"
        ],
        "irrigation": "Drip irrigation to reduce humidity",
        "fertilizers": ["Low nitrogen fertilizers", "Balanced NPK (8-8-8)"],
        "prevention": "Regular pruning and monitoring"
    },
    "Healthy Cherry": {
        "causes": "No disease present",
        "remedies": [
            "Maintain current care practices",
            "Regular monitoring",
            "Proper watering schedule",
            "Balanced fertilization"
        ],
        "irrigation": "Regular watering schedule",
        "fertilizers": ["Balanced NPK (10-10-10)", "Organic compost"],
        "prevention": "Continue preventive measures"
    },
    "Corn Cercospora leaf spot Gray leaf spot": {
        "causes": "Fungus (Cercospora zeae-maydis) that spreads through wind and rain",
        "remedies": [
            "Plant resistant hybrids",
            "Practice crop rotation",
            "Apply fungicides during early stages",
            "Maintain proper spacing"
        ],
        "irrigation": "Furrow irrigation recommended",
        "fertilizers": ["Nitrogen-rich fertilizers", "Balanced NPK (15-15-15)"],
        "prevention": "Early detection and treatment"
    },
    "Corn(maize) Common rust": {
        "causes": "Fungus (Puccinia sorghi) that spreads through wind",
        "remedies": [
            "Plant resistant hybrids",
            "Apply fungicides early",
            "Remove infected plants",
            "Practice crop rotation"
        ],
        "irrigation": "Furrow irrigation recommended",
        "fertilizers": ["Nitrogen-rich fertilizers", "Balanced NPK (15-15-15)"],
        "prevention": "Early detection and treatment"
    },
    "Corn(maize) Northern Leaf Blight": {
        "causes": "Fungus (Exserohilum turcicum) that overwinters in crop debris",
        "remedies": [
            "Plant resistant hybrids",
            "Practice crop rotation",
            "Apply fungicides early",
            "Remove infected plants"
        ],
        "irrigation": "Furrow irrigation recommended",
        "fertilizers": ["Nitrogen-rich fertilizers", "Balanced NPK (15-15-15)"],
        "prevention": "Early detection and treatment"
    },
    "Corn(maize) Healthy": {
        "causes": "No disease present",
        "remedies": [
            "Maintain current care practices",
            "Regular monitoring",
            "Proper watering schedule",
            "Balanced fertilization"
        ],
        "irrigation": "Regular watering schedule",
        "fertilizers": ["Balanced NPK (15-15-15)", "Organic compost"],
        "prevention": "Continue preventive measures"
    },
    "Grape Black rot": {
        "causes": "Fungus (Guignardia bidwellii) that infects through wounds",
        "remedies": [
            "Remove infected clusters",
            "Apply copper-based fungicides",
            "Improve air circulation",
            "Practice good sanitation"
        ],
        "irrigation": "Drip irrigation to keep foliage dry",
        "fertilizers": ["Balanced NPK (10-10-10)", "Calcium-rich fertilizers"],
        "prevention": "Regular pruning and monitoring"
    },
    "Grape Esca(Black Measles)": {
        "causes": "Fungal complex including Phaeomoniella chlamydospora",
        "remedies": [
            "Remove infected vines",
            "Apply fungicides",
            "Improve drainage",
            "Practice good sanitation"
        ],
        "irrigation": "Drip irrigation preferred",
        "fertilizers": ["Balanced NPK (10-10-10)", "Calcium-rich fertilizers"],
        "prevention": "Regular monitoring and early treatment"
    },
    "Grape Leaf blight (Isariopsis Leaf Spot)": {
        "causes": "Fungus (Isariopsis clavispora) that thrives in humid conditions",
        "remedies": [
            "Remove infected leaves",
            "Apply fungicides",
            "Improve air circulation",
            "Practice good sanitation"
        ],
        "irrigation": "Drip irrigation preferred",
        "fertilizers": ["Balanced NPK (10-10-10)", "Calcium-rich fertilizers"],
        "prevention": "Regular pruning and monitoring"
    },
    "Grape Healthy": {
        "causes": "No disease present",
        "remedies": [
            "Maintain current care practices",
            "Regular monitoring",
            "Proper watering schedule",
            "Balanced fertilization"
        ],
        "irrigation": "Regular watering schedule",
        "fertilizers": ["Balanced NPK (10-10-10)", "Organic compost"],
        "prevention": "Continue preventive measures"
    },
    "Peach Bacterial spot": {
        "causes": "Bacteria (Xanthomonas arboricola pv. pruni) spread by rain and wind",
        "remedies": [
            "Apply copper-based bactericides",
            "Remove infected leaves and fruit",
            "Improve air circulation",
            "Use resistant varieties"
        ],
        "irrigation": "Drip irrigation preferred",
        "fertilizers": ["Balanced NPK (12-12-12)", "Zinc-rich fertilizers"],
        "prevention": "Early detection and treatment"
    },
    "Peach Healthy": {
        "causes": "No disease present",
        "remedies": [
            "Maintain current care practices",
            "Regular monitoring",
            "Proper watering schedule",
            "Balanced fertilization"
        ],
        "irrigation": "Regular watering schedule",
        "fertilizers": ["Balanced NPK (12-12-12)", "Organic compost"],
        "prevention": "Continue preventive measures"
    },
    "Pepper bell Bacterial spot": {
        "causes": "Bacteria (Xanthomonas euvesicatoria) spread by water and tools",
        "remedies": [
            "Apply copper-based bactericides",
            "Remove infected plants",
            "Practice crop rotation",
            "Use disease-free seed"
        ],
        "irrigation": "Drip irrigation preferred",
        "fertilizers": ["Balanced NPK (10-10-10)", "Calcium-rich fertilizers"],
        "prevention": "Early detection and treatment"
    },
    "Pepper bell Healthy": {
        "causes": "No disease present",
        "remedies": [
            "Maintain current care practices",
            "Regular monitoring",
            "Proper watering schedule",
            "Balanced fertilization"
        ],
        "irrigation": "Regular watering schedule",
        "fertilizers": ["Balanced NPK (10-10-10)", "Organic compost"],
        "prevention": "Continue preventive measures"
    },
    "Potato Early blight": {
        "causes": "Fungus (Alternaria solani) that thrives in warm, humid conditions",
        "remedies": [
            "Practice crop rotation",
            "Apply fungicides during early stages",
            "Remove infected leaves",
            "Maintain proper spacing"
        ],
        "irrigation": "Furrow irrigation recommended",
        "fertilizers": ["Potassium-rich fertilizers", "Balanced NPK (15-15-15)"],
        "prevention": "Regular monitoring and early treatment"
    },
    "Potato Late blight": {
        "causes": "Fungus (Phytophthora infestans) that spreads rapidly in cool, wet conditions",
        "remedies": [
            "Apply fungicides early",
            "Remove infected plants",
            "Improve air circulation",
            "Practice crop rotation"
        ],
        "irrigation": "Furrow irrigation recommended",
        "fertilizers": ["Balanced NPK (15-15-15)", "Potassium-rich fertilizers"],
        "prevention": "Early detection and treatment"
    },
    "Potato Healthy": {
        "causes": "No disease present",
        "remedies": [
            "Maintain current care practices",
            "Regular monitoring",
            "Proper watering schedule",
            "Balanced fertilization"
        ],
        "irrigation": "Regular watering schedule",
        "fertilizers": ["Balanced NPK (15-15-15)", "Organic compost"],
        "prevention": "Continue preventive measures"
    },
    "Strawberry Leaf scorch": {
        "causes": "Fungus (Diplocarpon earliana) that overwinters in infected leaves",
        "remedies": [
            "Remove infected leaves",
            "Apply fungicides early",
            "Improve air circulation",
            "Practice good sanitation"
        ],
        "irrigation": "Drip irrigation preferred",
        "fertilizers": ["Balanced NPK (10-10-10)", "Calcium-rich fertilizers"],
        "prevention": "Regular monitoring and early treatment"
    },
    "Strawberry Healthy": {
        "causes": "No disease present",
        "remedies": [
            "Maintain current care practices",
            "Regular monitoring",
            "Proper watering schedule",
            "Balanced fertilization"
        ],
        "irrigation": "Regular watering schedule",
        "fertilizers": ["Balanced NPK (10-10-10)", "Organic compost"],
        "prevention": "Continue preventive measures"
    },
    "Tomato Bacterial spot": {
        "causes": "Bacteria (Xanthomonas euvesicatoria) spread by water and tools",
        "remedies": [
            "Apply copper-based bactericides",
            "Remove infected plants",
            "Practice crop rotation",
            "Use disease-free seed"
        ],
        "irrigation": "Drip irrigation preferred",
        "fertilizers": ["Balanced NPK (10-10-10)", "Calcium-rich fertilizers"],
        "prevention": "Early detection and treatment"
    },
    "Tomato Early blight": {
        "causes": "Fungus (Alternaria solani) that spreads through wind and rain",
        "remedies": [
            "Remove infected leaves",
            "Apply fungicides during early stages",
            "Improve air circulation",
            "Practice crop rotation"
        ],
        "irrigation": "Drip irrigation preferred",
        "fertilizers": ["Balanced NPK (10-10-10)", "Calcium-rich fertilizers"],
        "prevention": "Regular pruning and monitoring"
    },
    "Tomato Late blight": {
        "causes": "Fungus (Phytophthora infestans) that spreads rapidly in cool, wet conditions",
        "remedies": [
            "Apply fungicides early",
            "Remove infected plants",
            "Improve air circulation",
            "Practice crop rotation"
        ],
        "irrigation": "Drip irrigation preferred",
        "fertilizers": ["Balanced NPK (10-10-10)", "Calcium-rich fertilizers"],
        "prevention": "Early detection and treatment"
    },
    "Tomato Leaf Mold": {
        "causes": "Fungus (Fulvia fulva) that thrives in high humidity",
        "remedies": [
            "Improve air circulation",
            "Apply fungicides",
            "Remove infected leaves",
            "Reduce humidity"
        ],
        "irrigation": "Drip irrigation preferred",
        "fertilizers": ["Balanced NPK (10-10-10)", "Calcium-rich fertilizers"],
        "prevention": "Regular monitoring and early treatment"
    },
    "Tomato Septoria leaf spot": {
        "causes": "Fungus (Septoria lycopersici) that overwinters in crop debris",
        "remedies": [
            "Remove infected leaves",
            "Apply fungicides",
            "Practice crop rotation",
            "Improve air circulation"
        ],
        "irrigation": "Drip irrigation preferred",
        "fertilizers": ["Balanced NPK (10-10-10)", "Calcium-rich fertilizers"],
        "prevention": "Regular monitoring and early treatment"
    },
    "Tomato Spider mites Two-spotted spider mite": {
        "causes": "Tiny arachnids (Tetranychus urticae) that feed on plant sap",
        "remedies": [
            "Apply miticides",
            "Increase humidity",
            "Remove infected leaves",
            "Use predatory mites"
        ],
        "irrigation": "Regular watering schedule",
        "fertilizers": ["Balanced NPK (10-10-10)", "Calcium-rich fertilizers"],
        "prevention": "Regular monitoring and early treatment"
    },
    "Tomato Target Spot": {
        "causes": "Fungus (Corynespora cassiicola) that thrives in warm, humid conditions",
        "remedies": [
            "Remove infected leaves",
            "Apply fungicides",
            "Improve air circulation",
            "Practice crop rotation"
        ],
        "irrigation": "Drip irrigation preferred",
        "fertilizers": ["Balanced NPK (10-10-10)", "Calcium-rich fertilizers"],
        "prevention": "Regular monitoring and early treatment"
    },
    "Tomato Yellow Leaf Curl Virus": {
        "causes": "Virus transmitted by whiteflies",
        "remedies": [
            "Remove infected plants",
            "Control whiteflies",
            "Use virus-free transplants",
            "Plant resistant varieties"
        ],
        "irrigation": "Drip irrigation preferred",
        "fertilizers": ["Balanced NPK (10-10-10)", "Calcium-rich fertilizers"],
        "prevention": "Early detection and treatment"
    },
    "Tomato mosaic virus": {
        "causes": "Virus spread through mechanical means and insects",
        "remedies": [
            "Remove infected plants",
            "Use virus-free seed",
            "Control insect vectors",
            "Practice good sanitation"
        ],
        "irrigation": "Drip irrigation preferred",
        "fertilizers": ["Balanced NPK (10-10-10)", "Calcium-rich fertilizers"],
        "prevention": "Early detection and treatment"
    },
    "Tomato Healthy": {
        "causes": "No disease present",
        "remedies": [
            "Maintain current care practices",
            "Regular monitoring",
            "Proper watering schedule",
            "Balanced fertilization"
        ],
        "irrigation": "Regular watering schedule",
        "fertilizers": ["Balanced NPK (10-10-10)", "Organic compost"],
        "prevention": "Continue preventive measures"
    }
}

# Farming resources by language
farming_resources = {
    "hindi": {
        "videos": [
            {
                "title": "खेती की बेसिक जानकारी",
                "url": "https://www.youtube.com/watch?v=wougJaN_Ha0&t=9s&ab_channel=QuickSupport",
                "description": "किसानों के लिए बेसिक खेती की जानकारी"
            },
            {
                "title": "फसल सुरक्षा",
                "url": "https://www.youtube.com/watch?v=jcd5xNOl_5U&ab_channel=RNKushwaha",
                "description": "फसलों को बीमारियों से बचाने के तरीके"
            }
        ],
        "articles": [
            {
                "title": "खेती में सफलता के मूल मंत्र",
                "url": "https://www.nabard.org/pdf/harvesting-gold-from-indian-agriculture-hindi.pdf",
                "description": "किसानों के लिए सफल खेती के टिप्स"
            }
        ],
        "government_schemes": [
            {
                "name": "प्रधानमंत्री किसान सम्मान निधि",
                "description": "किसानों के लिए आय सहायता योजना",
                "url": "https://pmkisan.gov.in"
            }
        ]
    },
    "punjabi": {
        "videos": [
            {
                "title": "ਖੇਤੀ ਦੀ ਬੁਨਿਆਦੀ ਜਾਣਕਾਰੀ",
                "url": "https://www.youtube.com/watch?v=DRFmRC1PtQQ&ab_channel=IndianSafalKisantv",
                "description": "ਕਿਸਾਨਾਂ ਲਈ ਬੁਨਿਆਦੀ ਖੇਤੀ ਦੀ ਜਾਣਕਾਰੀ"
            }
        ],
        "articles": [
            {
                "title": "ਖੇਤੀ ਵਿੱਚ ਸਫਲਤਾ ਦੇ ਮੁੱਖ ਸੂਤਰ",
                "url": "https://www.nabard.org/pdf/harvesting-gold-from-indian-agriculture-hindi.pdf",
                "description": "ਕਿਸਾਨਾਂ ਲਈ ਸਫਲ ਖੇਤੀ ਦੇ ਸੁਝਾਅ"
            }
        ]
    },
    "tamil": {
        "videos": [
            {
                "title": "விவசாய அடிப்படை தகவல்கள்",
                "url": "https://www.youtube.com/watch?v=Zj-pLI1mISo&ab_channel=TrendingVivasayi",
                "description": "விவசாயிகளுக்கான அடிப்படை விவசாய தகவல்கள்"
            }
        ],
        "articles": [
            {
                "title": "விவசாயத்தில் வெற்றிக்கான முக்கிய குறிப்புகள்",
                "url": "https://tinyurl.com/farmingpracticesarticle",
                "description": "விவசாயிகளுக்கான வெற்றிகரமான விவசாய உதவிக்குறிப்புகள்"
            }
        ]
    },
    "telugu": {
        "videos": [
            {
                "title": "వ్యవసాయ ప్రాథమిక సమాచారం",
                "url": "https://www.youtube.com/watch?v=Zj-pLI1mISo&ab_channel=TrendingVivasayi",
                "description": "రైతులకు ప్రాథమిక వ్యవసాయ సమాచారం"
            }
        ],
        "articles": [
            {
                "title": "వ్యవసాయంలో విజయానికి ముఖ్యమైన చిట్కాలు",
                "url": "https://tinyurl.com/farmingpracticesarticle",
                "description": "రైతులకు విజయవంతమైన వ్యవసాయ చిట్కాలు"
            }
        ]
    }
}

IMAGE_SIZE = 255

# Function to preprocess and predict
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)

            # Read and preprocess the image
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))

            # Get prediction
            predicted_class, confidence = predict(img)

            # Retrieve disease details
            disease_info = disease_data.get(predicted_class, {"description": "No description available", "stage": "Unknown"})
            description = disease_info["description"]
            stage = disease_info["stage"]

            # Get additional information from web scraping
            additional_info = scraper.get_all_information(predicted_class)

            # Generate PDF report
            pdf_buffer = generate_pdf_report(predicted_class, confidence, description, stage, filepath)
            
            # Save PDF to static folder
            pdf_filename = f"report_{filename.rsplit('.', 1)[0]}.pdf"
            pdf_path = os.path.join('static', pdf_filename)
            with open(pdf_path, 'wb') as f:
                f.write(pdf_buffer.getvalue())

            return render_template(
                'index.html',
                image_path=filepath,
                predicted_label=predicted_class,
                confidence=confidence,
                description=description,
                stage=stage,
                pdf_path=pdf_filename,
                additional_info=additional_info
            )

    return render_template('index.html', message='Upload an image')

@app.route('/download_report/<filename>')
def download_report(filename):
    return send_file(
        os.path.join('static', filename),
        as_attachment=True,
        download_name=filename
    )

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/resources')
def resources():
    return render_template('resources.html', resources=farming_resources)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def generate_pdf_report(predicted_class, confidence, description, stage, image_path):
    # Get additional information from web scraping
    additional_info = scraper.get_all_information(predicted_class)
    
    # Create a buffer to store the PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles with adjusted spacing
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,  # Reduced from 24
        spaceAfter=20,  # Reduced from 30
        textColor=colors.HexColor('#1a5f7a'),
        alignment=1,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,  # Reduced from 18
        spaceAfter=12,  # Reduced from 15
        textColor=colors.HexColor('#2c3e50'),
        borderWidth=1,
        borderColor=colors.HexColor('#1a5f7a'),
        borderPadding=4,  # Reduced from 5
        borderRadius=4,  # Reduced from 5
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=13,  # Reduced from 14
        spaceAfter=8,  # Reduced from 10
        textColor=colors.HexColor('#34495e'),
        backgroundColor=colors.HexColor('#f8f9fa'),
        borderWidth=1,
        borderColor=colors.HexColor('#e9ecef'),
        borderPadding=4,  # Reduced from 5
        borderRadius=3,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,  # Reduced from 12
        spaceAfter=6,  # Reduced from 8
        textColor=colors.HexColor('#2c3e50'),
        fontName='Helvetica'
    )
    
    elements = []
    
    # Header with reduced spacing
    elements.append(Paragraph("PLANT PATHOLOGY LABORATORY", title_style))
    elements.append(Paragraph("Disease Detection Report", title_style))
    
    # Report Information with compact table
    info_table = Table([
        ["Report ID:", f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"],
        ["Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ["Status:", "Final Report"],
        ["Laboratory:", "Plant Pathology Platform"],
        ["Technician:", "AI Analysis System"]
    ])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Reduced from 12
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),  # Reduced from 12
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 15))  # Reduced from 20
    
    # Executive Summary with compact spacing
    elements.append(Paragraph("Executive Summary", heading_style))
    summary_text = f"""
    This report presents the findings of a comprehensive analysis conducted on the submitted plant sample. 
    The analysis was performed using advanced computer vision and deep learning techniques to identify 
    potential plant diseases and provide detailed recommendations for treatment and prevention.
    
    Key Findings:
    • Disease Detected: {predicted_class}
    • Confidence Level: {confidence}%
    • Growth Stage: {stage}
    
    The analysis indicates a {confidence}% probability of {predicted_class.lower()} in the submitted sample. 
    This finding is based on detailed visual analysis and pattern recognition using our advanced AI model.
    """
    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 15))  # Reduced from 20
    
    # Sample Information with compact table
    elements.append(Paragraph("Sample Information", heading_style))
    sample_table = Table([
        ["Sample Type:", "Plant Leaf/Image"],
        ["Analysis Method:", "Deep Learning Model"],
        ["Detection Method:", "Computer Vision Analysis"],
        ["Confidence Level:", f"{confidence}%"],
        ["Analysis Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ["Report Status:", "Final"]
    ])
    sample_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Reduced from 12
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),  # Reduced from 12
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
    ]))
    elements.append(sample_table)
    elements.append(Spacer(1, 15))  # Reduced from 20
    
    # Detailed Analysis with compact layout
    elements.append(Paragraph("Detailed Analysis", heading_style))
    
    # Disease Analysis with reduced spacing
    elements.append(Paragraph("Disease Analysis", subheading_style))
    disease_analysis_text = f"""
    The submitted sample has been analyzed for potential plant diseases using our advanced AI-powered 
    detection system. The analysis revealed the following findings:
    
    • Primary Diagnosis: {predicted_class}
    • Growth Stage: {stage}
    • Description: {description}
    
    The confidence level of {confidence}% indicates a strong probability of the detected condition. 
    This assessment is based on multiple factors including visual characteristics, pattern recognition, 
    and comparative analysis with our extensive disease database.
    """
    elements.append(Paragraph(disease_analysis_text, normal_style))
    elements.append(Spacer(1, 8))  # Reduced from 10
    
    # Sample Image with reduced size
    elements.append(Paragraph("Sample Image Analysis", subheading_style))
    img = Image(image_path, width=300, height=300)  # Reduced from 400x400
    img.drawHeight = 300
    img.drawWidth = 300
    elements.append(img)
    elements.append(Spacer(1, 15))  # Reduced from 20
    
    # Treatment Recommendations with compact layout
    if predicted_class in disease_remedies:
        remedies = disease_remedies[predicted_class]
        
        elements.append(Paragraph("Treatment Recommendations", heading_style))
        
        # Etiology with reduced spacing
        elements.append(Paragraph("Etiology", subheading_style))
        etiology_text = f"""
        The identified condition is primarily caused by {remedies['causes']}. Understanding the etiology 
        is crucial for implementing effective treatment strategies and preventing future occurrences.
        """
        elements.append(Paragraph(etiology_text, normal_style))
        elements.append(Spacer(1, 8))  # Reduced from 10
        
        # Treatment Protocol with compact table
        elements.append(Paragraph("Treatment Protocol", subheading_style))
        treatment_text = """
        Based on our analysis, we recommend the following treatment protocol:
        """
        elements.append(Paragraph(treatment_text, normal_style))
        treatment_table = Table([[f"• {remedy}"] for remedy in remedies['remedies']])
        treatment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),  # Reduced from 12
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),  # Reduced from 8
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
        ]))
        elements.append(treatment_table)
        elements.append(Spacer(1, 8))  # Reduced from 10
        
        # Cultural Practices with compact table
        elements.append(Paragraph("Cultural Practices", subheading_style))
        cultural_text = """
        To ensure optimal plant health and prevent disease recurrence, we recommend the following cultural practices:
        """
        elements.append(Paragraph(cultural_text, normal_style))
        cultural_table = Table([
            ["Irrigation Method:", remedies['irrigation']],
            ["Recommended Fertilizers:", "\n".join([f"• {f}" for f in remedies['fertilizers']])],
            ["Preventive Measures:", remedies['prevention']]
        ])
        cultural_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),  # Reduced from 12
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),  # Reduced from 12
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
        ]))
        elements.append(cultural_table)
        elements.append(Spacer(1, 15))  # Reduced from 20
    
    # Analysis Results with compact charts
    elements.append(Paragraph("Analysis Results", heading_style))
    
    # Disease Severity Chart with reduced size
    plt.figure(figsize=(4, 3))  # Reduced from (6, 4)
    plt.pie([confidence, 100-confidence], labels=['Disease Severity', 'Healthy'], 
            colors=['#FF6B6B', '#4ECDC4'], autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.title('Disease Severity Analysis', pad=15, fontsize=10)  # Reduced padding and font size
    
    pie_chart_buffer = io.BytesIO()
    plt.savefig(pie_chart_buffer, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    pie_chart_buffer.seek(0)
    
    elements.append(Paragraph("Disease Severity Analysis", subheading_style))
    severity_text = f"""
    The severity analysis indicates a {confidence}% probability of disease presence in the sample. 
    This assessment is based on multiple visual characteristics and pattern recognition algorithms.
    """
    elements.append(Paragraph(severity_text, normal_style))
    pie_chart_img = Image(pie_chart_buffer, width=300, height=225)  # Reduced from 400x300
    elements.append(pie_chart_img)
    elements.append(Spacer(1, 15))  # Reduced from 20
    
    # Treatment Timeline with reduced size
    plt.figure(figsize=(6, 3))  # Reduced from (8, 4)
    treatment_stages = ['Immediate', 'Short-term', 'Long-term']
    treatment_actions = [3, 2, 1]
    bars = plt.bar(treatment_stages, treatment_actions, 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.title('Treatment Timeline', pad=15, fontsize=10)  # Reduced padding and font size
    plt.ylabel('Number of Actions', fontsize=9)  # Reduced font size
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    bar_chart_buffer = io.BytesIO()
    plt.savefig(bar_chart_buffer, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    bar_chart_buffer.seek(0)
    
    elements.append(Paragraph("Treatment Timeline", subheading_style))
    timeline_text = """
    The treatment timeline outlines the recommended actions across different time periods to ensure 
    effective disease management and plant recovery.
    """
    elements.append(Paragraph(timeline_text, normal_style))
    bar_chart_img = Image(bar_chart_buffer, width=300, height=225)  # Reduced from 400x300
    elements.append(bar_chart_img)
    elements.append(Spacer(1, 15))  # Reduced from 20
    
    # Important Notes with compact table
    elements.append(Paragraph("Important Notes", heading_style))
    notes = [
        "• Regular monitoring is essential for early detection",
        "• Follow recommended treatment schedules",
        "• Maintain proper plant hygiene",
        "• Consider consulting with a plant pathologist for severe cases",
        "• Keep records of treatment applications",
        "• Monitor weather conditions for disease development"
    ]
    notes_table = Table([[note] for note in notes])
    notes_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Reduced from 12
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),  # Reduced from 8
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
    ]))
    elements.append(notes_table)
    
    # Add Additional Resources Section
    elements.append(Paragraph("Additional Resources", heading_style))
    
    # Latest Articles
    if additional_info['articles']:
        elements.append(Paragraph("Latest Articles", subheading_style))
        for article in additional_info['articles']:
            article_text = f"""
            <b>{article['title']}</b>
            {article['description']}
            Source: {article['source']}
            Date: {article['date']}
            """
            elements.append(Paragraph(article_text, normal_style))
            elements.append(Spacer(1, 8))
    
    # Natural Remedies
    if additional_info['remedies']:
        elements.append(Paragraph("Natural Remedies", subheading_style))
        for remedy in additional_info['remedies']:
            remedy_text = f"""
            <b>{remedy['title']}</b>
            {remedy['description']}
            Steps:
            """
            elements.append(Paragraph(remedy_text, normal_style))
            for step in remedy['steps']:
                elements.append(Paragraph(f"• {step}", normal_style))
            elements.append(Spacer(1, 8))
    
    # Chemical Treatments
    if additional_info['chemical_treatments']:
        elements.append(Paragraph("Chemical Treatment Options", subheading_style))
        for treatment in additional_info['chemical_treatments']:
            treatment_text = f"""
            <b>{treatment['name']}</b>
            {treatment['description']}
            Dosage: {treatment['dosage']}
            Precautions:
            """
            elements.append(Paragraph(treatment_text, normal_style))
            for precaution in treatment['precautions']:
                elements.append(Paragraph(f"• {precaution}", normal_style))
            elements.append(Spacer(1, 8))
    
    # Conclusion with reduced spacing
    elements.append(Paragraph("Conclusion", heading_style))
    conclusion_text = f"""
    Based on our comprehensive analysis, we have identified {predicted_class.lower()} in the submitted sample 
    with a confidence level of {confidence}%. The recommended treatment protocol and cultural practices 
    should be followed to manage the condition effectively.
    
    Regular monitoring and adherence to the suggested preventive measures will help maintain plant health 
    and prevent disease recurrence. We recommend scheduling follow-up assessments to track the effectiveness 
    of the treatment protocol.
    """
    elements.append(Paragraph(conclusion_text, normal_style))
    
    # Footer with reduced spacing
    elements.append(Spacer(1, 20))  # Reduced from 30
    elements.append(Paragraph("This report was generated by the Plant Pathology Platform", normal_style))
    elements.append(Paragraph("For more information, visit our website", normal_style))
    elements.append(Paragraph("Disclaimer: This report is for informational purposes only. Always consult with a qualified plant pathologist for professional diagnosis and treatment recommendations.", normal_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

if __name__ == '__main__':
    app.run(debug=True)
