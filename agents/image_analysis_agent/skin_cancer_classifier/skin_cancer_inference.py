"""
Skin Cancer Classification Agent using EfficientNet-B0
Based on skin-cancer-classification-main implementation
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models, transforms
import logging

# Add skin-cancer-classification-main to path (optional, for future enhancements)
skin_cancer_base_path = os.path.join(
    os.path.dirname(__file__),
    "../skin-cancer-classification-main"
)
if os.path.exists(skin_cancer_base_path):
    sys.path.insert(0, skin_cancer_base_path)

logger = logging.getLogger(__name__)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Skin Cancer Classifier using device: {DEVICE}")

class SkinCancerClassifier:
    """
    Skin Cancer Classification Agent using EfficientNet-B0.
    Classifies skin lesions as benign or malignant.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the skin cancer classifier.
        
        Args:
            model_path: Path to the trained EfficientNet model (.pth file)
            device: PyTorch device (auto-detected if None)
        """
        self.device = device if device is not None else DEVICE
        
        # Default model path
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "../skin-cancer-classification-main/skin_classifier/model/best_efficientnet_b0_focal_loss.pth"
            )
        
        self.model_path = model_path
        self.model = None
        self.class_names = ["benign", "malignant"]
        self._load_model()
    
    def _load_model(self):
        """Load the EfficientNet-B0 model with trained weights."""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model not found at {self.model_path}. Classification will not be available.")
                return None
            
            # Load EfficientNet-B0
            model = models.efficientnet_b0(weights=None)
            
            # Customize classifier head (matches training architecture)
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, 2)  # 2 classes: benign, malignant
            )
            
            # Load trained weights
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            
            logger.info(f"✅ Skin Cancer Classifier model loaded from {self.model_path}")
            self.model = model
            return model
            
        except Exception as e:
            logger.error(f"Error loading skin cancer classifier model: {e}")
            self.model = None
            return None
    
    def _get_data_transforms(self):
        """Create data transformations for EfficientNet (matching training)."""
        return transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """
        Predict skin lesion classification (benign or malignant).
        
        Args:
            image_path: Path to the skin lesion image
            
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        if self.model is None:
            logger.warning("Model not loaded. Cannot make prediction.")
            return {
                "error": "Model not available",
                "predicted_class": None,
                "confidence": 0.0,
                "all_probabilities": []
            }
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            transform = self._get_data_transforms()
            img_tensor = transform(image).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                predicted_class_idx = torch.argmax(probabilities).item()
            
            # Format results
            class_probs = {
                self.class_names[i]: float(probabilities[i].item() * 100)
                for i in range(len(self.class_names))
            }
            
            sorted_probs = sorted(
                class_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            result = {
                "predicted_class": self.class_names[predicted_class_idx],
                "confidence": float(probabilities[predicted_class_idx].item() * 100),
                "all_probabilities": sorted_probs,
                "benign_probability": class_probs.get("benign", 0.0),
                "malignant_probability": class_probs.get("malignant", 0.0)
            }
            
            logger.info(f"Prediction: {result['predicted_class']} ({result['confidence']:.2f}% confidence)")
            return result
            
        except Exception as e:
            logger.error(f"Error during skin cancer prediction: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "predicted_class": None,
                "confidence": 0.0,
                "all_probabilities": []
            }
    
    def is_available(self):
        """Check if the model is loaded and available."""
        return self.model is not None

