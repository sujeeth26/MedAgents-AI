import os
import json
import base64
from mimetypes import guess_type

from typing import TypedDict
from langchain_core.output_parsers import JsonOutputParser

class ClassificationDecision(TypedDict):
    """Output structure for the decision agent."""
    image_type: str
    reasoning: str
    confidence: float

class ImageClassifier:
    """Uses GPT-4o Vision to analyze images and determine their type."""
    
    def __init__(self, vision_model):
        self.vision_model = vision_model
        self.json_parser = JsonOutputParser(pydantic_object=ClassificationDecision)
        
    def local_image_to_data_url(self, image_path: str) -> str:
        """Get the url of a local image"""
        try:
            mime_type, _ = guess_type(image_path)
            if mime_type is None:
                mime_type = "application/octet-stream"

            with open(image_path, "rb") as image_file:
                base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

            return f"data:{mime_type};base64,{base64_encoded_data}"
        except Exception as e:
            print(f"[ImageAnalyzer] Error reading image file {image_path}: {e}")
            raise e
    
    def classify_image(self, image_path: str) -> dict:
        """Analyzes the image to classify it as a medical image and determine its type."""
        print(f"[ImageAnalyzer] Analyzing image: {image_path}")

        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"[ImageAnalyzer] Error: Image file not found: {image_path}")
            return {"image_type": "unknown", "reasoning": "Image file not found", "confidence": 0.0}

        # Try to use vision model first, but fallback to heuristics if it fails
        try:
            vision_prompt = [
                {"role": "system", "content": "You are an expert in medical imaging. Analyze the uploaded image."},
                {"role": "user", "content": [
                    {"type": "text", "text": (
                        """
                        Determine if this is a medical image. If it is, classify it as:
                        'BRAIN MRI SCAN', 'CHEST X-RAY', 'SKIN LESION', or 'OTHER'. If it's not a medical image, return 'NON-MEDICAL'.
                        You must provide your answer in JSON format with the following structure:
                        {{
                        "image_type": "IMAGE TYPE",
                        "reasoning": "Your step-by-step reasoning for selecting this agent",
                        "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this classification task
                        }}
                        """
                    )},
                    {"type": "image_url", "image_url": {"url": self.local_image_to_data_url(image_path)}}
                ]}
            ]
            
            # Invoke LLM to classify the image
            response = self.vision_model.invoke(vision_prompt)

            try:
                # Ensure the response is parsed as JSON
                response_json = self.json_parser.parse(response.content)
                print(f"[ImageAnalyzer] Vision model classification: {response_json}")
                return response_json
            except json.JSONDecodeError as e:
                print(f"[ImageAnalyzer] Warning: Response was not valid JSON: {e}")
                print(f"[ImageAnalyzer] Raw response: {response.content}")
                return {"image_type": "unknown", "reasoning": "Invalid JSON response", "confidence": 0.0}
        
        except Exception as e:
            print(f"[ImageAnalyzer] Vision model failed: {e}")
            print("[ImageAnalyzer] Falling back to heuristic classification...")
            
            # Fallback: Use heuristics to classify image type
            return self._classify_image_heuristic(image_path)
    
    def _classify_image_heuristic(self, image_path: str) -> dict:
        """Fallback method to classify images using heuristics when vision model fails."""
        import cv2
        import numpy as np
        
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return {"image_type": "unknown", "reasoning": "Could not read image", "confidence": 0.0}
            
            # Get image dimensions and basic properties
            height, width = img.shape[:2]
            aspect_ratio = width / height
            
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Analyze color distribution
            mean_color = np.mean(img, axis=(0, 1))
            std_color = np.std(img, axis=(0, 1))
            
            # Heuristic rules for classification
            # Skin lesions typically have:
            # - Warm colors (red, brown, pink tones)
            # - Irregular shapes
            # - Higher color variation
            
            # Check for skin-like colors (warm tones)
            red_channel = mean_color[2]  # BGR format
            green_channel = mean_color[1]
            blue_channel = mean_color[0]
            
            # Skin lesions often have higher red component
            if red_channel > green_channel and red_channel > blue_channel:
                # Additional check for skin lesion characteristics
                if std_color.mean() > 30:  # High color variation
                    return {
                        "image_type": "SKIN LESION",
                        "reasoning": "Heuristic classification: Image shows warm colors with high variation, likely skin lesion",
                        "confidence": 0.7
                    }
            
            # Check for X-ray characteristics (typically grayscale with specific patterns)
            if aspect_ratio > 0.8 and aspect_ratio < 1.2:  # Roughly square
                gray_mean = np.mean(gray)
                if 50 < gray_mean < 200:  # Typical X-ray intensity range
                    return {
                        "image_type": "CHEST X-RAY", 
                        "reasoning": "Heuristic classification: Square aspect ratio with grayscale characteristics",
                        "confidence": 0.6
                    }
            
            # Default to skin lesion for medical images (most common case)
            return {
                "image_type": "SKIN LESION",
                "reasoning": "Heuristic classification: Defaulting to skin lesion for medical image analysis",
                "confidence": 0.5
            }
            
        except Exception as e:
            print(f"[ImageAnalyzer] Heuristic classification failed: {e}")
            return {
                "image_type": "SKIN LESION", 
                "reasoning": "Fallback classification: Assuming skin lesion for medical image",
                "confidence": 0.3
            }

        # return response.content.strip().lower()
