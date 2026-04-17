"""
MedRAX Wrapper for Enhanced Chest X-ray Analysis
Integrates MedRAX capabilities while preserving the existing interface
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Add MedRAX to Python path
medrax_base_path = os.path.join(
    os.path.dirname(__file__), 
    "../MedRAX-main"
)
if os.path.exists(medrax_base_path):
    sys.path.insert(0, medrax_base_path)

try:
    # Import directly from the classification module to avoid __init__.py dependencies
    import importlib.util
    classification_path = os.path.join(medrax_base_path, "medrax/tools/classification.py")
    if os.path.exists(classification_path):
        spec = importlib.util.spec_from_file_location("medrax_classification", classification_path)
        medrax_classification = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(medrax_classification)
        ChestXRayClassifierTool = medrax_classification.ChestXRayClassifierTool
        MEDRAX_AVAILABLE = True
    else:
        MEDRAX_AVAILABLE = False
        logging.warning(f"MedRAX classification module not found at {classification_path}")
except Exception as e:
    MEDRAX_AVAILABLE = False
    logging.warning(f"MedRAX not available: {e}. Will use fallback classifier.")

class MedRAXChestXRayAgent:
    """
    Enhanced Chest X-ray Agent using MedRAX framework.
    Provides 18-disease classification while maintaining backward compatibility.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize MedRAX chest X-ray agent.
        
        Args:
            model_path: Legacy parameter (kept for compatibility, not used by MedRAX)
            device: Device to run models on ("cuda" or "cpu")
        """
        self.logger = logging.getLogger(__name__)
        
        # Determine device
        if device is None:
            # Auto-detect device: prefer CUDA if available, otherwise CPU
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Initialize MedRAX classifier if available
        if MEDRAX_AVAILABLE:
            try:
                # MedRAX may not support MPS, so use CPU if MPS is detected
                medrax_device = "cpu" if self.device == "mps" else self.device
                self.classifier = ChestXRayClassifierTool(device=medrax_device)
                self.use_medrax = True
                self.logger.info(f"MedRAX ChestXRayClassifierTool initialized on {medrax_device}")
            except Exception as e:
                self.logger.error(f"Failed to initialize MedRAX: {e}")
                self.use_medrax = False
                self.classifier = None
        else:
            self.use_medrax = False
            self.classifier = None
    
    def predict(self, img_path: str) -> str:
        """
        Predict chest X-ray classification (backward compatible method).
        
        This method maintains the same interface as the original ChestXRayClassification.predict()
        Returns: "covid19", "normal", or None
        
        Args:
            img_path: Path to chest X-ray image
            
        Returns:
            str: "covid19", "normal", or None
        """
        if not self.use_medrax or self.classifier is None:
            return None
        
        try:
            # Get comprehensive classification from MedRAX
            pathologies, metadata = self.classifier._run(img_path)
            
            if "error" in pathologies:
                self.logger.error(f"MedRAX classification error: {pathologies['error']}")
                return None
            
            # Map MedRAX pathologies to simple COVID-19/Normal classification
            # COVID-19 is often associated with Pneumonia, Consolidation, Lung Opacity
            covid_indicators = [
                "Pneumonia",
                "Consolidation", 
                "Lung Opacity",
                "Infiltration"
            ]
            
            # Check for COVID-19 indicators
            covid_score = 0.0
            for indicator in covid_indicators:
                if indicator in pathologies:
                    covid_score += pathologies[indicator]
            
            # Normal if no significant pathologies detected
            significant_pathologies = {
                k: v for k, v in pathologies.items() 
                if v > 0.3  # Threshold for significant detection
            }
            
            # If we have strong COVID indicators, classify as COVID-19
            if covid_score > 0.5 or any(
                pathologies.get(path, 0) > 0.5 
                for path in covid_indicators
            ):
                self.logger.info(f"MedRAX classification: COVID-19 (score: {covid_score:.2f})")
                return "covid19"
            
            # If no significant pathologies, classify as normal
            if len(significant_pathologies) == 0:
                self.logger.info("MedRAX classification: Normal (no significant pathologies)")
                return "normal"
            
            # If other pathologies detected but not COVID indicators, still check
            # for pneumonia which could be COVID-related
            if pathologies.get("Pneumonia", 0) > 0.3:
                self.logger.info(f"MedRAX classification: COVID-19 (Pneumonia detected: {pathologies['Pneumonia']:.2f})")
                return "covid19"
            
            # Default to normal if uncertain
            self.logger.info(f"MedRAX classification: Normal (other pathologies: {significant_pathologies})")
            return "normal"
            
        except Exception as e:
            self.logger.error(f"Error during MedRAX prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def classify_detailed(self, img_path: str) -> Dict[str, Any]:
        """
        Get detailed classification with all 18 pathologies.
        
        Args:
            img_path: Path to chest X-ray image
            
        Returns:
            Dict containing:
                - pathologies: Dict of all 18 pathologies with probabilities
                - primary_diagnosis: Main detected condition
                - covid19_probability: Specific COVID-19 probability
                - normal: Boolean indicating if image appears normal
        """
        if not self.use_medrax or self.classifier is None:
            return {
                "error": "MedRAX not available",
                "pathologies": {},
                "primary_diagnosis": None,
                "covid19_probability": 0.0,
                "normal": True
            }
        
        try:
            pathologies, metadata = self.classifier._run(img_path)
            
            if "error" in pathologies:
                return {
                    "error": pathologies["error"],
                    "pathologies": {},
                    "primary_diagnosis": None,
                    "covid19_probability": 0.0,
                    "normal": True
                }
            
            # Find primary diagnosis (highest probability pathology)
            primary_diagnosis = max(pathologies.items(), key=lambda x: x[1]) if pathologies else None
            
            # Calculate COVID-19 probability based on related pathologies
            covid_related = [
                pathologies.get("Pneumonia", 0),
                pathologies.get("Consolidation", 0),
                pathologies.get("Lung Opacity", 0),
                pathologies.get("Infiltration", 0)
            ]
            covid19_probability = max(covid_related) if covid_related else 0.0
            
            # Determine if normal (no significant pathologies)
            significant_pathologies = {
                k: v for k, v in pathologies.items() 
                if v > 0.3
            }
            is_normal = len(significant_pathologies) == 0
            
            return {
                "pathologies": pathologies,
                "primary_diagnosis": primary_diagnosis[0] if primary_diagnosis and primary_diagnosis[1] > 0.3 else None,
                "primary_probability": primary_diagnosis[1] if primary_diagnosis else 0.0,
                "covid19_probability": covid19_probability,
                "normal": is_normal,
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error during detailed classification: {e}")
            return {
                "error": str(e),
                "pathologies": {},
                "primary_diagnosis": None,
                "covid19_probability": 0.0,
                "normal": True
            }

