import os
import numpy as np
import cv2
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator

# Set the path to the dataset (Update this with your local path)
#dataset_path = "C:\Users\shreyass krishna\Desktop\sem8_project\dataset"
dataset_path = "C:\\Users\\shreyass krishna\\Desktop\\sem8_project\\dataset"


# Define the training and testing directories
train_dir = os.path.join(dataset_path, "Training")
test_dir = os.path.join(dataset_path, "Testing")

# Define the categories
categories = ["glioma", "meningioma", "notumor", "pituitary"]

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

target_size = (224, 224)
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# Load Pretrained Models
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze Initial Layers
for layer in resnet.layers[:100]:
    layer.trainable = False
for layer in vgg16.layers[:15]:
    layer.trainable = False

# Feature Extraction Layers
resnet_output = GlobalAveragePooling2D()(resnet.output)
vgg16_output = GlobalAveragePooling2D()(vgg16.output)

# Concatenate Features
merged_features = Concatenate()([resnet_output, vgg16_output])

# Fully Connected Layers
x = Dense(512, activation='relu')(merged_features)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(len(categories), activation='softmax')(x)  # 4-class classification

# Define Hybrid CNN Model
model = Model(inputs=[resnet.input, vgg16.input], outputs=out)

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Class labels
classes = ['Meningioma', 'Glioma', 'Pituitary', 'No Tumor']
# Explanation of categories:
# - **Meningioma** 🧠: Typically benign tumors arising from the meninges, often treated with surgery.
# - **Glioma** ⚡: A type of tumor that originates from the brain's glial cells, ranging from low-grade to high-grade malignancies.
# - **Pituitary** 🏥: Tumors affecting the pituitary gland, often treated with medication or surgery.
# - **No Tumor** 🚫: Indicates no detectable tumor in the MRI scan.

stages = {'Meningioma': 'Stage 1-3', 'Glioma': 'Stage 1-4', 'Pituitary': 'Non-staged', 'No Tumor': 'N/A'}
# Explanation of stages:
# - **Meningioma (Stage 1-3)**: Reflects the tumor's size and potential impact.
# - **Glioma (Stage 1-4)**: Indicates the tumor's aggressiveness and spread.
# - **Pituitary (Non-staged)**: Typically not staged due to its unique characteristics.
# - **No Tumor (N/A)**: No staging applies as there is no tumor detected.

recovery_methods = {
    'Meningioma': " Primary Treatment for Meningioma 🧑‍⚕️:\n"
                  "- Minimally invasive neurosurgery with neuronavigation 🛠️.\n"
                  "- Alternative Therapy: Proton beam therapy for targeted radiation 🌟.\n"
                  "- Post-Treatment Care: AI-assisted MRI monitoring 📊, personalized rehabilitation 🧘.\n"
                  "- Lifestyle: Anti-inflammatory diet 🍏, meditation 🧘‍♀️, yoga 🧘‍♂️.",

    'Glioma': " Primary Treatment for Glioma 🧑‍⚕️:\n"
              "- Combination therapy—tumor resection, localized radiation, chemotherapy 💊.\n"
              "- Emerging Solutions: Tumor Treating Fields (TTFields), immunotherapy trials 🚀.\n"
              "- Post-Treatment Care: Machine learning-based progression tracking 📈.\n"
              "- Lifestyle: Ketogenic diet research 🍳, cognitive rehabilitation exercises 🧠.",

    'Pituitary': " Primary Treatment for Pituitary Tumors 🧑‍⚕️:\n"
                 "- Dopamine agonists for prolactinomas 💊, transsphenoidal surgery if necessary 🔪.\n"
                 "- Emerging Solutions: Gene therapy, robotic-assisted surgeries 🤖.\n"
                 "- Post-Treatment Care: AI-driven endocrine monitoring 📊, vision and neurological assessments 👁️.\n"
                 "- Lifestyle: Sleep optimization 😴, stress reduction 🌿, balanced nutrition 🍽️.",

    'No Tumor': " Preventive Measures for No Tumor 🚫:\n"
                "- Regular neuro check-ups 🩺, brain-training activities 🎮, mindfulness meditation 🧘.\n"
                "- Cognitive Enhancement: Nootropic supplements (under supervision) 💊, intermittent fasting ⏳.\n"
                "- Physical Health: Cardiovascular exercises 🏃‍♂️, social engagement 🤝.\n"
                "- Nutritional Support: Mediterranean diet rich in healthy fats and greens 🥗."
}

def preprocess_image(image):
    image = cv2.resize(image, target_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

def predict_tumor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = preprocess_image(image)
    prediction = model.predict([processed_image, processed_image])
    class_idx = np.argmax(prediction)
    tumor_type = classes[class_idx]
    stage = stages.get(tumor_type, 'Unknown')
    recovery = recovery_methods.get(tumor_type, 'No specific recovery steps available.')

    result = f"### AI-Powered Brain Tumor Diagnosis 🤖\n\n**Tumor Type Identified:** {tumor_type}\n\n**Stage:** {stage}\n\n### Suggested Recovery Plan\n{recovery}"
    return result

# User Interface Explanation:
# This interface allows users to upload MRI scans for AI-driven tumor detection.
# It provides a simple and intuitive way to classify tumors and offer personalized recovery insights.
demo = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="numpy", label="Upload Your MRI Scan 📸"),
    outputs=gr.Markdown(),
    title="🧠 HYBRID CNN MODEL FOR BRAIN TUMOR DETECTION AND CLASSIFICATION",
    description="Leverage AI-driven MRI analysis for tumor classification, staging, and personalized recovery insights.",
    theme="huggingface",
    live=True,
)

if __name__ == "__main__":
    demo.launch()
