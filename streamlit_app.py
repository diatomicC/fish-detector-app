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
from ultralytics import YOLO
import tempfile
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Set page config
st.set_page_config(page_title="Fish Species Detector", layout="wide")

# Initialize the models
@st.cache_resource
def load_models():
    resnet = ResNet50(weights='imagenet')
    yolo = YOLO('yolov8n.pt')  # Load YOLOv8 model
    return resnet, yolo

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.yolo_model = YOLO('yolov8n.pt')

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Detect fish using YOLO
        results = self.yolo_model(img)
        
        # Draw detection boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Process all detected objects (not just class 1)
                b = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                class_name = self.yolo_model.names[int(cls)]
                
                # Draw bounding box and label for all objects
                cv2.rectangle(img, 
                            (int(b[0]), int(b[1])), 
                            (int(b[2]), int(b[3])), 
                            (0, 255, 0), 2)
                
                label = f'{class_name} {conf:.2f}'
                cv2.putText(img, label, 
                          (int(b[0]), int(b[1]-10)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.9, (0, 255, 0), 2)
        
        return img

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

def process_video(video_file, yolo_model):
    # Save uploaded video to temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    # Open video file
    cap = cv2.VideoCapture(tfile.name)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect fish using YOLO
        results = yolo_model(frame)
        
        # Draw detection boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                class_name = yolo_model.names[int(cls)]
                
                # Draw bounding box
                cv2.rectangle(frame, 
                            (int(b[0]), int(b[1])), 
                            (int(b[2]), int(b[3])), 
                            (0, 255, 0), 2)
                
                # Add label
                label = f'{class_name} {conf:.2f}'
                cv2.putText(frame, label, 
                          (int(b[0]), int(b[1]-10)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.9, (0, 255, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    return temp_output.name

def main():
    st.title("🐟 Fish Species Detector")
    st.write("Upload an image/video, use webcam, or start live detection!")

    # Load the models
    resnet_model, yolo_model = load_models()

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        # Option to upload image/video or use webcam
        option = st.radio("Choose input method:", ["Upload Image", "Upload Video", "Take Picture", "Live Detection"])
        
        if option == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image_bytes = uploaded_file.read()
                img = Image.open(io.BytesIO(image_bytes))
                st.image(img, caption="Uploaded Image", use_column_width=True)
                img_array = np.array(img)
                
        elif option == "Upload Video":
            uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                st.video(uploaded_file)
                if st.button("Detect Fish in Video"):
                    with st.spinner("Processing video..."):
                        processed_video = process_video(uploaded_file, yolo_model)
                        st.video(processed_video)
                
        elif option == "Take Picture":
            picture = st.camera_input("Take a picture")
            if picture is not None:
                img = Image.open(picture)
                st.image(picture, caption="Captured Image", use_column_width=True)
                img_array = np.array(img)
        
        else:  # Live Detection
            st.write("Live Object Detection")
            RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            webrtc_streamer(
                key="example", 
                video_transformer_factory=VideoTransformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False}
            )

        if 'img_array' in locals() and option not in ["Upload Video", "Live Detection"]:
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    # Get predictions
                    predictions = analyze_fish_image(img_array, resnet_model)
                    
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