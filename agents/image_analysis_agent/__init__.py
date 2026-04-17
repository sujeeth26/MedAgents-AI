from .image_classifier import ImageClassifier
from .chest_xray_agent.covid_chest_xray_inference import ChestXRayClassification
from .chest_xray_agent.medrax_wrapper import MedRAXChestXRayAgent
from .brain_tumor_agent.brain_tumor_inference import BrainTumorAgent
from .skin_lesion_agent.skin_lesion_inference import SkinLesionSegmentation
from .skin_cancer_classifier.skin_cancer_inference import SkinCancerClassifier
import logging

logger = logging.getLogger(__name__)

class ImageAnalysisAgent:
    """
    Agent responsible for processing image uploads and classifying them as medical or non-medical, and determining their type.
    """

    def __init__(self, config):
        self.image_classifier = ImageClassifier(vision_model=config.medical_cv.llm)
        
        # Try to use MedRAX first, fallback to basic classifier if unavailable
        try:
            self.chest_xray_agent = MedRAXChestXRayAgent(
                model_path=config.medical_cv.chest_xray_model_path,
                device=None  # Will auto-detect
            )
            if self.chest_xray_agent.use_medrax:
                logger.info("✅ Using MedRAX for chest X-ray analysis (18-disease classification)")
            else:
                logger.warning("⚠️ MedRAX not available, falling back to basic classifier")
                self.chest_xray_agent = ChestXRayClassification(
                    model_path=config.medical_cv.chest_xray_model_path
                )
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize MedRAX: {e}. Using basic classifier.")
            self.chest_xray_agent = ChestXRayClassification(
                model_path=config.medical_cv.chest_xray_model_path
            )
        
        self.brain_tumor_agent = BrainTumorAgent(model_path=config.medical_cv.brain_tumor_model_path)
        
        # Try to use new skin cancer classifier, fallback to old segmentation if unavailable
        try:
            import os
            skin_cancer_model_path = os.path.join(
                os.path.dirname(__file__),
                "skin-cancer-classification-main/skin_classifier/model/best_efficientnet_b0_focal_loss.pth"
            )
            self.skin_cancer_classifier = SkinCancerClassifier(model_path=skin_cancer_model_path)
            if self.skin_cancer_classifier.is_available():
                logger.info("✅ Using Skin Cancer Classifier (EfficientNet-B0) for benign/malignant classification")
            else:
                logger.warning("⚠️ Skin Cancer Classifier model not found, falling back to segmentation")
                self.skin_cancer_classifier = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Skin Cancer Classifier: {e}. Using segmentation fallback.")
            self.skin_cancer_classifier = None
        
        # Keep old segmentation agent for backward compatibility
        self.skin_lesion_agent = SkinLesionSegmentation(model_path=config.medical_cv.skin_lesion_model_path)
        self.skin_lesion_segmentation_output_path = config.medical_cv.skin_lesion_segmentation_output_path
    
    # classify image
    def analyze_image(self, image_path: str) -> str:
        """Classifies images as medical or non-medical and determines their type."""
        return self.image_classifier.classify_image(image_path)
    
    # chest x-ray agent
    def classify_chest_xray(self, image_path: str) -> str:
        """
        Classify chest X-ray image.
        Returns: "covid19", "normal", or None
        
        Uses MedRAX if available (18-disease classification),
        otherwise falls back to basic COVID-19/Normal classifier.
        """
        return self.chest_xray_agent.predict(image_path)
    
    def classify_chest_xray_detailed(self, image_path: str) -> dict:
        """
        Get detailed chest X-ray classification with all 18 pathologies.
        Only available if MedRAX is being used.
        
        Returns:
            Dict with pathologies, primary_diagnosis, covid19_probability, etc.
        """
        if hasattr(self.chest_xray_agent, 'classify_detailed'):
            return self.chest_xray_agent.classify_detailed(image_path)
        else:
            # Fallback: return basic classification
            prediction = self.chest_xray_agent.predict(image_path)
            return {
                "prediction": prediction,
                "method": "basic",
                "pathologies": {prediction: 1.0} if prediction else {}
            }
    
    # brain tumor agent
    def classify_brain_tumor(self, image_path: str) -> dict:
        """
        Analyze brain MRI for tumor detection and classification.
        
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        if hasattr(self.brain_tumor_agent, 'predict'):
            result = self.brain_tumor_agent.predict(image_path)
            # If it returns a dict (new format), return it directly
            if isinstance(result, dict):
                return result
            # If it returns a string (old format), wrap it
            else:
                return {
                    "predicted_class": result,
                    "confidence": 0.0,
                    "all_probabilities": []
                }
        else:
            return {
                "error": "Brain tumor agent not available",
                "predicted_class": None,
                "confidence": 0.0,
                "all_probabilities": []
            }
    
    # skin lesion agent (old segmentation method)
    def segment_skin_lesion(self, image_path: str) -> str:
        return self.skin_lesion_agent.predict(image_path, self.skin_lesion_segmentation_output_path)
    
    # skin cancer classifier (new EfficientNet-based classification)
    def classify_skin_cancer(self, image_path: str) -> dict:
        """
        Classify skin lesion as benign or malignant using EfficientNet-B0.
        
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        if self.skin_cancer_classifier and self.skin_cancer_classifier.is_available():
            return self.skin_cancer_classifier.predict(image_path)
        else:
            # Fallback: return error
            return {
                "error": "Skin cancer classifier not available",
                "predicted_class": None,
                "confidence": 0.0,
                "all_probabilities": []
            }
