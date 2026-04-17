"""
Full MedRAX Integration for Enhanced Chest X-ray Analysis
Integrates all MedRAX tools: Classification, Segmentation, Report Generation, Phrase Grounding, VQA
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import importlib.util
import matplotlib
# Fix for macOS threading crash
matplotlib.use('Agg')

# Add MedRAX to Python path
# Use the provided MedRAX path
medrax_base_path = "/Users/vasupatel/Desktop/MedAgentica/Multi-Agent-Medical-Assistant/agents/image_analysis_agent/MedRAX-main"
# Also try relative path
if not os.path.exists(medrax_base_path):
    medrax_base_path = os.path.join(
        os.path.dirname(__file__),
        "../MedRAX-main"
    )
if os.path.exists(medrax_base_path):
    sys.path.insert(0, medrax_base_path)

logger = logging.getLogger(__name__)

class MedRAXFullIntegration:
    """
    Full MedRAX integration with all tools:
    - Classification (18 diseases)
    - Segmentation (anatomical structures)
    - Report Generation (findings + impression)
    - Phrase Grounding (disease localization)
    - VQA (visual question answering)
    """
    
    def __init__(self, device=None, cache_dir=None, temp_dir=None):
        """
        Initialize MedRAX with all tools.
        
        Args:
            device: Device to run models on ("cuda", "cpu", or "mps")
            cache_dir: Directory for model cache
            temp_dir: Directory for temporary files
        """
        self.logger = logging.getLogger(__name__)
        
        # Determine device
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Use CPU for compatibility if MPS detected
        if self.device == "mps":
            self.device = "cpu"
        
        # Set directories
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.cache_dir = cache_dir or os.path.join(project_root, "model-weights")
        # Use uploads directory for temp files so they can be served
        self.temp_dir = temp_dir or os.path.join(project_root, "uploads", "chest_xray_output")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize tools
        self.classifier = None
        self.segmentation_tool = None
        self.report_generator = None
        self.phrase_grounding = None
        self.vqa_tool = None
        
        # Load tools
        self._load_tools()
    
    def _load_tools(self):
        """Load all MedRAX tools."""
        try:
            # Load Classification Tool
            classification_path = os.path.join(medrax_base_path, "medrax/tools/classification.py")
            if os.path.exists(classification_path):
                spec = importlib.util.spec_from_file_location("medrax_classification", classification_path)
                medrax_classification = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(medrax_classification)
                self.classifier = medrax_classification.ChestXRayClassifierTool(device=self.device)
                self.logger.info("✅ MedRAX Classification Tool loaded")
            
            # Load Segmentation Tool
            segmentation_path = os.path.join(medrax_base_path, "medrax/tools/segmentation.py")
            if os.path.exists(segmentation_path):
                spec = importlib.util.spec_from_file_location("medrax_segmentation", segmentation_path)
                medrax_segmentation = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(medrax_segmentation)
                # Ensure temp_dir exists
                os.makedirs(self.temp_dir, exist_ok=True)
                self.segmentation_tool = medrax_segmentation.ChestXRaySegmentationTool(
                    device=self.device, 
                    temp_dir=Path(self.temp_dir)
                )
                self.logger.info(f"✅ MedRAX Segmentation Tool loaded (temp_dir: {self.temp_dir})")
            
            # Load Report Generator Tool
            report_path = os.path.join(medrax_base_path, "medrax/tools/report_generation.py")
            if os.path.exists(report_path):
                spec = importlib.util.spec_from_file_location("medrax_report", report_path)
                medrax_report = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(medrax_report)
                self.report_generator = medrax_report.ChestXRayReportGeneratorTool(
                    cache_dir=self.cache_dir,
                    device=self.device
                )
                self.logger.info("✅ MedRAX Report Generator Tool loaded")
            
            # Load Phrase Grounding Tool
            grounding_path = os.path.join(medrax_base_path, "medrax/tools/grounding.py")
            if os.path.exists(grounding_path):
                try:
                    spec = importlib.util.spec_from_file_location("medrax_grounding", grounding_path)
                    medrax_grounding = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(medrax_grounding)
                    self.phrase_grounding = medrax_grounding.XRayPhraseGroundingTool(
                        cache_dir=self.cache_dir,
                        temp_dir=self.temp_dir,
                        load_in_8bit=True,
                        device=self.device
                    )
                    self.logger.info("✅ MedRAX Phrase Grounding Tool loaded")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load Phrase Grounding Tool (likely authentication/permissions issue): {e}")
                    self.logger.warning("Continuing without Phrase Grounding capability.")
                    self.phrase_grounding = None
            
            # Load VQA Tool
            vqa_path = os.path.join(medrax_base_path, "medrax/tools/xray_vqa.py")
            if os.path.exists(vqa_path):
                spec = importlib.util.spec_from_file_location("medrax_vqa", vqa_path)
                medrax_vqa = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(medrax_vqa)
                self.vqa_tool = medrax_vqa.XRayVQATool(
                    cache_dir=self.cache_dir,
                    device=self.device
                )
                self.logger.info("✅ MedRAX VQA Tool loaded")
                
        except Exception as e:
            self.logger.error(f"Error loading MedRAX tools: {e}")
            import traceback
            traceback.print_exc()
    
    def classify(self, image_path: str) -> Dict[str, Any]:
        """Classify chest X-ray for 18 diseases."""
        if not self.classifier:
            return {"error": "Classification tool not available"}
        
        try:
            pathologies, metadata = self.classifier._run(image_path)
            return {
                "pathologies": pathologies,
                "metadata": metadata,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return {"error": str(e)}
    
    def segment(self, image_path: str, organs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Segment anatomical structures in chest X-ray."""
        if not self.segmentation_tool:
            return {"error": "Segmentation tool not available"}
        
        try:
            result, metadata = self.segmentation_tool._run(image_path, organs=organs)
            return {
                "segmentation_image_path": result.get("segmentation_image_path"),
                "organ_metrics": result.get("organ_metrics", {}),
                "metadata": metadata,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Segmentation error: {e}")
            return {"error": str(e)}
    
    def generate_report(self, image_path: str) -> Dict[str, Any]:
        """Generate comprehensive radiology report."""
        if not self.report_generator:
            return {"error": "Report generator not available"}
        
        try:
            report, metadata = self.report_generator._run(image_path)
            
            # Parse report to extract findings and impression
            findings = ""
            impression = ""
            
            if isinstance(report, str):
                # Report format: "CHEST X-RAY REPORT\n\nFINDINGS:\n...\n\nIMPRESSION:\n..."
                if "FINDINGS:" in report and "IMPRESSION:" in report:
                    parts = report.split("IMPRESSION:")
                    if len(parts) == 2:
                        findings_part = parts[0].split("FINDINGS:")
                        if len(findings_part) == 2:
                            findings = findings_part[1].strip()
                        impression = parts[1].strip()
                else:
                    # If not in standard format, use whole report as findings
                    findings = report
            
            return {
                "report": report,
                "findings": findings,
                "impression": impression,
                "metadata": metadata,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def ground_phrase(self, image_path: str, phrase: str) -> Dict[str, Any]:
        """Ground (locate) a medical phrase in the image."""
        if not self.phrase_grounding:
            return {"error": "Phrase grounding tool not available"}
        
        try:
            result, metadata = self.phrase_grounding._run(image_path, phrase=phrase)
            
            # Extract bounding boxes from predictions
            predictions = result.get("predictions", [])
            bounding_boxes = []
            if predictions:
                for pred in predictions:
                    if isinstance(pred, dict):
                        # New format: {"phrase": "...", "bounding_boxes": {"image_coordinates": [...]}}
                        bboxes = pred.get("bounding_boxes", {}).get("image_coordinates", [])
                        if bboxes:
                            bounding_boxes.extend(bboxes)
                    elif isinstance(pred, (list, tuple)) and len(pred) == 2:
                        # Old format: (phrase, bboxes)
                        pred_phrase, pred_bboxes = pred
                        if pred_bboxes:
                            bounding_boxes.extend(pred_bboxes)
            
            return {
                "bounding_boxes": bounding_boxes,
                "bounding_box": bounding_boxes[0] if bounding_boxes else None,
                "visualization_path": result.get("visualization_path"),
                "confidence": 0.8 if bounding_boxes else 0.0,  # Default confidence
                "metadata": metadata,
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Phrase grounding error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def ground_diseases(self, image_path: str, pathologies: Dict[str, float]) -> Dict[str, Any]:
        """Ground all detected diseases in the image."""
        if not self.phrase_grounding:
            return {"error": "Phrase grounding tool not available"}
        
        # Get top diseases with probability > 0.3
        significant_pathologies = {
            k: v for k, v in pathologies.items() 
            if v > 0.3
        }
        
        if not significant_pathologies:
            return {"error": "No significant pathologies to ground"}
        
        # Ground each significant pathology
        grounded_diseases = {}
        all_visualizations = []
        
        for disease, prob in sorted(significant_pathologies.items(), key=lambda x: x[1], reverse=True)[:3]:
            try:
                result = self.ground_phrase(image_path, disease)
                if not result.get("error") and result.get("bounding_box"):
                    grounded_diseases[disease] = {
                        "probability": prob,
                        "bounding_box": result.get("bounding_box"),
                        "bounding_boxes": result.get("bounding_boxes", []),
                        "visualization_path": result.get("visualization_path"),
                        "confidence": result.get("confidence", 0.0)
                    }
                    if result.get("visualization_path"):
                        all_visualizations.append(result.get("visualization_path"))
            except Exception as e:
                self.logger.warning(f"Failed to ground {disease}: {e}")
                import traceback
                traceback.print_exc()
        
        # Create combined visualization
        combined_viz_path = None
        if all_visualizations:
            try:
                combined_viz_path = self._create_combined_visualization(
                    image_path, 
                    grounded_diseases,
                    all_visualizations
                )
            except Exception as e:
                self.logger.warning(f"Failed to create combined visualization: {e}")
        
        return {
            "grounded_diseases": grounded_diseases,
            "combined_visualization_path": combined_viz_path,
            "error": None
        }
    
    def _create_combined_visualization(self, image_path: str, grounded_diseases: Dict, visualization_paths: List[str]) -> str:
        """Create a combined visualization showing all grounded diseases."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from PIL import Image
            import numpy as np
            
            # Load original image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(img_array)
            ax.axis('off')
            
            # Add bounding boxes for each disease
            colors = plt.cm.Set3(np.linspace(0, 1, len(grounded_diseases)))
            img_width, img_height = img.size
            
            for idx, (disease, data) in enumerate(grounded_diseases.items()):
                # Try to get bounding boxes (can be list of bboxes or single bbox)
                bboxes = data.get("bounding_boxes", [])
                bbox = data.get("bounding_box")
                
                # If we have a list of bboxes, use all of them
                if bboxes:
                    for b in bboxes:
                        if b and len(b) == 4:
                            x1, y1, x2, y2 = b
                            # Check if coordinates are normalized (0-1) or pixel coordinates
                            if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                                # Normalized coordinates
                                width = (x2 - x1) * img_width
                                height = (y2 - y1) * img_height
                                x = x1 * img_width
                                y = y1 * img_height
                            else:
                                # Pixel coordinates
                                width = x2 - x1
                                height = y2 - y1
                                x = x1
                                y = y1
                            
                            rect = patches.Rectangle(
                                (x, y),
                                width,
                                height,
                                linewidth=2,
                                edgecolor=colors[idx],
                                facecolor='none',
                                label=f"{disease} ({data['probability']:.1%})"
                            )
                            ax.add_patch(rect)
                elif bbox and len(bbox) == 4:
                    # Single bounding box
                    x1, y1, x2, y2 = bbox
                    # Check if coordinates are normalized (0-1) or pixel coordinates
                    if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                        # Normalized coordinates
                        width = (x2 - x1) * img_width
                        height = (y2 - y1) * img_height
                        x = x1 * img_width
                        y = y1 * img_height
                    else:
                        # Pixel coordinates
                        width = x2 - x1
                        height = y2 - y1
                        x = x1
                        y = y1
                    
                    rect = patches.Rectangle(
                        (x, y),
                        width,
                        height,
                        linewidth=2,
                        edgecolor=colors[idx],
                        facecolor='none',
                        label=f"{disease} ({data['probability']:.1%})"
                    )
                    ax.add_patch(rect)
                
                # Add label (only once per disease)
                if bboxes or bbox:
                    first_bbox = bboxes[0] if bboxes else bbox
                    if first_bbox and len(first_bbox) == 4:
                        x1, y1, x2, y2 = first_bbox
                        if x1 <= 1.0 and y1 <= 1.0:
                            x = x1 * img_width
                            y = y1 * img_height
                        else:
                            x = x1
                            y = y1
                        
                        ax.text(
                            x,
                            y - 5,
                            f"{disease} ({data['probability']:.1%})",
                            color=colors[idx],
                            fontsize=10,
                            weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                        )
            
            ax.legend(loc='upper right', fontsize=8)
            ax.set_title("Disease Grounding Visualization", fontsize=14, weight='bold')
            
            # Save combined visualization
            import uuid
            output_filename = f"disease_grounding_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
            output_path = os.path.join(self.temp_dir, output_filename)
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating combined visualization: {e}")
            return None
    
    def comprehensive_analysis(self, image_path: str, user_query: str = "") -> Dict[str, Any]:
        """
        Perform comprehensive analysis with all three outputs:
        1. Original image
        2. Segmentation
        3. Disease grounding
        """
        results = {
            "original_image_path": image_path,
            "classification": None,
            "segmentation": None,
            "report": None,
            "disease_grounding": None,
            "error": None
        }
        
        try:
            # 1. Classification
            results["classification"] = self.classify(image_path)
            
            # 2. Segmentation
            results["segmentation"] = self.segment(image_path)
            
            # 3. Report Generation
            results["report"] = self.generate_report(image_path)
            
            # 4. Disease Grounding (if pathologies detected)
            if results["classification"].get("pathologies"):
                pathologies = results["classification"]["pathologies"]
                results["disease_grounding"] = self.ground_diseases(image_path, pathologies)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis error: {e}")
            results["error"] = str(e)
            return results

