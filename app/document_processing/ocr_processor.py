from typing import List, Dict, Any, Optional, Tuple
import pytesseract
from PIL import Image
import pdf2image
import numpy as np
import cv2
from pathlib import Path
import tempfile
import logging
from datetime import datetime
import io

from app.document_processing.embeddings import generate_embeddings
from config.settings import settings

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Process images and scanned documents with OCR."""
    
    def __init__(self):
        """Initialize OCR processor with default settings."""
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Configure OCR settings
        self.config = {
            'lang': 'eng',  # Default language
            'config': '--psm 3'  # Fully automatic page segmentation
        }
    
    async def process_image(
        self,
        image_path: str,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Process image with OCR.
        
        Args:
            image_path: Path to image file
            preprocess: Whether to apply image preprocessing
        
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            if preprocess:
                image = self._preprocess_image(image)
            
            # Extract text with layout analysis
            ocr_data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                **self.config
            )
            
            # Extract text with confidence scores
            text_blocks = self._extract_text_blocks(ocr_data)
            
            # Generate embedding for full text
            full_text = " ".join(block["text"] for block in text_blocks)
            embedding = await generate_embeddings(full_text)
            
            return {
                "text_blocks": text_blocks,
                "full_text": full_text,
                "embedding": embedding,
                "metadata": self._generate_metadata(image, ocr_data),
                "confidence_score": np.mean([float(conf) for conf in ocr_data["conf"] if conf != "-1"])
            }
            
        except Exception as e:
            logger.error(f"Error processing image with OCR: {str(e)}")
            raise
    
    async def process_pdf_images(
        self,
        pdf_path: str,
        dpi: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Extract and process images from PDF.
        
        Args:
            pdf_path: Path to PDF file
            dpi: DPI for PDF to image conversion
        
        Returns:
            List of dictionaries containing extracted text and metadata
        """
        try:
            # Convert PDF pages to images
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=dpi,
                fmt="png"
            )
            
            results = []
            for idx, image in enumerate(images):
                # Process each page image
                result = await self.process_image(self._save_temp_image(image))
                result["page_number"] = idx + 1
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF images: {str(e)}")
            raise
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to improve OCR accuracy."""
        # Convert PIL Image to OpenCV format
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to handle different lighting conditions
        thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        
        # Noise removal
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Convert back to PIL Image
        return Image.fromarray(denoised)
    
    def _extract_text_blocks(
        self,
        ocr_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract structured text blocks from OCR data."""
        blocks = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 0:  # Filter low confidence
                blocks.append({
                    "text": ocr_data['text'][i],
                    "confidence": float(ocr_data['conf'][i]),
                    "position": {
                        "left": ocr_data['left'][i],
                        "top": ocr_data['top'][i],
                        "width": ocr_data['width'][i],
                        "height": ocr_data['height'][i]
                    },
                    "block_num": ocr_data['block_num'][i],
                    "line_num": ocr_data['line_num'][i],
                    "word_num": ocr_data['word_num'][i]
                })
        
        return self._merge_text_blocks(blocks)
    
    def _merge_text_blocks(
        self,
        blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge text blocks that belong together."""
        merged_blocks = []
        current_block = None
        
        for block in blocks:
            if current_block is None:
                current_block = block
            elif self._should_merge_blocks(current_block, block):
                # Merge blocks
                current_block["text"] += " " + block["text"]
                current_block["confidence"] = np.mean([
                    current_block["confidence"],
                    block["confidence"]
                ])
                # Update position to encompass both blocks
                current_block["position"] = self._merge_positions(
                    current_block["position"],
                    block["position"]
                )
            else:
                merged_blocks.append(current_block)
                current_block = block
        
        if current_block is not None:
            merged_blocks.append(current_block)
        
        return merged_blocks
    
    def _should_merge_blocks(
        self,
        block1: Dict[str, Any],
        block2: Dict[str, Any]
    ) -> bool:
        """Determine if two blocks should be merged."""
        # Same line and reasonable horizontal distance
        return (
            block1["line_num"] == block2["line_num"] and
            block2["position"]["left"] - (block1["position"]["left"] + block1["position"]["width"]) < 50
        )
    
    def _merge_positions(
        self,
        pos1: Dict[str, int],
        pos2: Dict[str, int]
    ) -> Dict[str, int]:
        """Merge two position rectangles."""
        return {
            "left": min(pos1["left"], pos2["left"]),
            "top": min(pos1["top"], pos2["top"]),
            "width": max(
                pos1["left"] + pos1["width"],
                pos2["left"] + pos2["width"]
            ) - min(pos1["left"], pos2["left"]),
            "height": max(
                pos1["top"] + pos1["height"],
                pos2["top"] + pos2["height"]
            ) - min(pos1["top"], pos2["top"])
        }
    
    def _generate_metadata(
        self,
        image: Image.Image,
        ocr_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate metadata for OCR results."""
        return {
            "image_size": image.size,
            "image_mode": image.mode,
            "word_count": len([t for t in ocr_data["text"] if t.strip()]),
            "average_confidence": np.mean([
                float(conf) for conf in ocr_data["conf"]
                if conf != "-1"
            ]),
            "processing_timestamp": datetime.utcnow().isoformat(),
            "language": self.config["lang"]
        }
    
    def _save_temp_image(self, image: Image.Image) -> str:
        """Save image to temporary file."""
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png",
            dir=self.temp_dir
        ) as tmp:
            image.save(tmp.name)
            return tmp.name

# Global OCR processor instance
ocr_processor = OCRProcessor()
