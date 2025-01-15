import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

def analyze_fish_image(image_path):
    # Load the pre-trained model
    model = ResNet50(weights='imagenet')
    
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Make predictions
    preds = model.predict(x)
    predictions = decode_predictions(preds, top=3)[0]
    
    # Filter for fish-related predictions
    fish_predictions = [pred for pred in predictions if 'fish' in pred[1].lower()]
    
    return fish_predictions if fish_predictions else predictions

if __name__ == "__main__":
    image_path = "test.jpg"
    results = analyze_fish_image(image_path)
    
    print("\nAnalysis Results:")
    print("-----------------")
    for i, (id, label, prob) in enumerate(results, 1):
        print(f"{i}. {label.replace('_', ' ').title()}: {prob*100:.2f}%") 