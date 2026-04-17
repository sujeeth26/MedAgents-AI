"""
Brain Tumor Classification Agent using BrainMRI-Tumor-Classifier-Pytorch-main.

This module wraps the BrainMRI-Tumor-Classifier-Pytorch model to classify brain MRI images
into 5 categories: No Tumor, Pituitary, Glioma, Meningioma, Other.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import logging

# Add BrainMRI-Tumor-Classifier-Pytorch-main to path
brain_tumor_base_path = os.path.join(
    os.path.dirname(__file__),
    "BrainMRI-Tumor-Classifier-Pytorch-main"
)
if os.path.exists(brain_tumor_base_path):
    sys.path.insert(0, brain_tumor_base_path)
    sys.path.insert(0, os.path.join(brain_tumor_base_path, "src"))

logger = logging.getLogger(__name__)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Brain Tumor Classifier using device: {DEVICE}")

# Import the model from the BrainMRI classifier
try:
    from src.model import MyModel, load_model
    from src.utils import predict as predict_utils
except ImportError as e:
    logger.warning(f"Could not import BrainMRI classifier modules: {e}")
    MyModel = None
    load_model = None
    predict_utils = None


class BrainTumorAgent:
    """
    Brain Tumor Classification Agent using BrainMRI-Tumor-Classifier-Pytorch model.
    Classifies brain MRI images into: No Tumor, Pituitary, Glioma, Meningioma, Other.
    """
    
    # Label mapping
    LABEL_DICT = {
        0: "No Tumor",
        1: "Pituitary",
        2: "Glioma",
        3: "Meningioma",
        4: "Other"
    }
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the Brain Tumor Classifier.
        
        Args:
            model_path: Path to the trained model file (model_38)
            device: Torch device (cuda/cpu), auto-detected if None
        """
        self.device = device if device is not None else DEVICE
        
        # Default model path
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "BrainMRI-Tumor-Classifier-Pytorch-main/models/model_38"
            )
        
        self.model_path = model_path
        self.model = None
        self.class_names = list(self.LABEL_DICT.values())
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model not found at {self.model_path}. Classification will not be available.")
                return None
            
            if load_model is None:
                logger.error("Could not import load_model from BrainMRI classifier")
                return None
            
            self.model = load_model(self.model_path, self.device)
            logger.info(f"✅ Brain Tumor Classifier model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading brain tumor model from {self.model_path}: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def get_data_transforms(self):
        """Get image preprocessing transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def predict(self, image_path: str) -> dict:
        """
        Predict brain tumor classification.
        
        Args:
            image_path: Path to the brain MRI image
            
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        if self.model is None:
            return {
                "error": "Model not loaded",
                "predicted_class": None,
                "confidence": 0.0,
                "all_probabilities": []
            }
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            transform = self.get_data_transforms()
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                predicted_class_idx = torch.argmax(probabilities).item()
            
            # Format results
            class_probs = {
                self.LABEL_DICT[i]: probabilities[i].item() * 100 
                for i in range(len(self.LABEL_DICT))
            }
            
            sorted_probs = sorted(
                class_probs.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            predicted_class = self.LABEL_DICT[predicted_class_idx]
            confidence = probabilities[predicted_class_idx].item() * 100
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': sorted_probs,
                'no_tumor_probability': class_probs.get("No Tumor", 0.0),
                'pituitary_probability': class_probs.get("Pituitary", 0.0),
                'glioma_probability': class_probs.get("Glioma", 0.0),
                'meningioma_probability': class_probs.get("Meningioma", 0.0),
                'other_probability': class_probs.get("Other", 0.0),
            }
        except Exception as e:
            logger.error(f"Error during brain tumor prediction for {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "predicted_class": None,
                "confidence": 0.0,
                "all_probabilities": []
            }
    
    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        return self.model is not None











