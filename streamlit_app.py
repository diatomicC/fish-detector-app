import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import io

# Set page config
st.set_page_config(page_title="Fish Species Detector", layout="wide")

# Initialize the model
@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet')

def analyze_fish_image(img_array, model):
    # Preprocess the image
    img_array = cv2.resize(img_array, (224, 224))
    x = np.expand_dims(img_array, axis=0)
    x = preprocess_input(x)
    
    # Make predictions
    preds = model.predict(x)
    predictions = decode_predictions(preds, top=5)[0]
    
    # Filter for fish-related predictions
    fish_predictions = [pred for pred in predictions if 'fish' in pred[1].lower()]
    
    return fish_predictions if fish_predictions else predictions

def create_confidence_chart(predictions):
    labels = [pred[1].replace('_', ' ').title() for pred in predictions]
    values = [pred[2] * 100 for pred in predictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker=dict(
                color='rgb(26, 118, 255)',
                line=dict(color='rgb(8, 48, 107)', width=1.5)
            )
        )
    ])
    
    fig.update_layout(
        title='Prediction Confidence Scores',
        xaxis_title='Confidence (%)',
        yaxis_title='Species',
        height=400
    )
    
    return fig

def main():
    st.title("🐟 Fish Species Detector")
    st.write("Take a picture or upload an image to identify fish species!")

    # Load the model
    model = load_model()

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        # Option to upload image or use webcam
        option = st.radio("Choose input method:", ["Upload Image", "Take Picture"])
        
        if option == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image_bytes = uploaded_file.read()
                img = Image.open(io.BytesIO(image_bytes))
                st.image(img, caption="Uploaded Image", use_column_width=True)
                img_array = np.array(img)
        else:
            picture = st.camera_input("Take a picture")
            if picture is not None:
                img = Image.open(picture)
                st.image(picture, caption="Captured Image", use_column_width=True)
                img_array = np.array(img)

        if 'img_array' in locals():
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    # Get predictions
                    predictions = analyze_fish_image(img_array, model)
                    
                    with col2:
                        st.subheader("Analysis Results")
                        # Create and display confidence chart
                        fig = create_confidence_chart(predictions)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display detailed results
                        st.subheader("Detailed Results")
                        for i, (id, label, prob) in enumerate(predictions, 1):
                            st.write(f"{i}. {label.replace('_', ' ').title()}: {prob*100:.2f}%")

if __name__ == "__main__":
    main() 