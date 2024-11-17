import pytest
from PIL import Image
import numpy as np
import tempfile
import os
from pathlib import Path
import json
import pytesseract
import pdf2image

from app.document_processing.ocr_processor import OCRProcessor, ocr_processor

@pytest.fixture
def test_image():
    """Create a test image with text."""
    image = Image.new('RGB', (800, 600), color='white')
    return image

@pytest.fixture
def test_image_path(test_image, tmp_path):
    """Save test image to temporary file."""
    image_path = tmp_path / "test.png"
    test_image.save(image_path)
    return str(image_path)

@pytest.fixture
def test_pdf_path(tmp_path):
    """Create a test PDF path."""
    return str(tmp_path / "test.pdf")

@pytest.fixture
def processor():
    """Create OCR processor instance."""
    return OCRProcessor()

@pytest.fixture
def mock_ocr_data():
    """Create mock OCR data."""
    return {
        'text': ['Hello', 'World'],
        'conf': ['90', '85'],
        'left': [10, 100],
        'top': [10, 10],
        'width': [50, 50],
        'height': [20, 20],
        'block_num': [1, 1],
        'line_num': [1, 1],
        'word_num': [1, 2]
    }

@pytest.mark.asyncio
async def test_process_image(processor, test_image_path, mocker):
    """Test image processing with OCR."""
    # Mock pytesseract output
    mock_data = {
        'text': ['Test', 'Text'],
        'conf': ['90', '85'],
        'left': [0, 100],
        'top': [0, 0],
        'width': [50, 50],
        'height': [20, 20],
        'block_num': [1, 1],
        'line_num': [1, 1],
        'word_num': [1, 2]
    }
    mocker.patch('pytesseract.image_to_data', return_value=mock_data)
    
    # Mock embedding generation
    mocker.patch(
        'app.document_processing.embeddings.generate_embeddings',
        return_value=[0.1] * 1536
    )
    
    result = await processor.process_image(test_image_path)
    
    assert 'text_blocks' in result
    assert 'full_text' in result
    assert 'embedding' in result
    assert 'metadata' in result
    assert 'confidence_score' in result
    assert len(result['embedding']) == 1536

def test_preprocess_image(processor, test_image):
    """Test image preprocessing."""
    processed = processor._preprocess_image(test_image)
    
    assert isinstance(processed, Image.Image)
    assert processed.mode == 'L'  # Should be grayscale

def test_extract_text_blocks(processor, mock_ocr_data):
    """Test extraction of text blocks from OCR data."""
    blocks = processor._extract_text_blocks(mock_ocr_data)
    
    assert len(blocks) > 0
    assert 'text' in blocks[0]
    assert 'confidence' in blocks[0]
    assert 'position' in blocks[0]
    assert isinstance(blocks[0]['confidence'], float)

def test_merge_text_blocks(processor):
    """Test merging of adjacent text blocks."""
    blocks = [
        {
            'text': 'Hello',
            'confidence': 90.0,
            'position': {'left': 0, 'top': 0, 'width': 50, 'height': 20},
            'block_num': 1,
            'line_num': 1,
            'word_num': 1
        },
        {
            'text': 'World',
            'confidence': 85.0,
            'position': {'left': 60, 'top': 0, 'width': 50, 'height': 20},
            'block_num': 1,
            'line_num': 1,
            'word_num': 2
        }
    ]
    
    merged = processor._merge_text_blocks(blocks)
    
    assert len(merged) == 1
    assert merged[0]['text'] == 'Hello World'
    assert 85.0 <= merged[0]['confidence'] <= 90.0

def test_should_merge_blocks(processor):
    """Test block merging decision logic."""
    block1 = {
        'line_num': 1,
        'position': {'left': 0, 'width': 50}
    }
    block2 = {
        'line_num': 1,
        'position': {'left': 60}
    }
    block3 = {
        'line_num': 2,
        'position': {'left': 60}
    }
    
    assert processor._should_merge_blocks(block1, block2) is True
    assert processor._should_merge_blocks(block1, block3) is False

def test_merge_positions(processor):
    """Test merging of position rectangles."""
    pos1 = {'left': 0, 'top': 0, 'width': 50, 'height': 20}
    pos2 = {'left': 60, 'top': 0, 'width': 50, 'height': 20}
    
    merged = processor._merge_positions(pos1, pos2)
    
    assert merged['left'] == 0
    assert merged['width'] == 110
    assert merged['height'] == 20

def test_generate_metadata(processor, test_image, mock_ocr_data):
    """Test metadata generation."""
    metadata = processor._generate_metadata(test_image, mock_ocr_data)
    
    assert 'image_size' in metadata
    assert 'image_mode' in metadata
    assert 'word_count' in metadata
    assert 'average_confidence' in metadata
    assert 'processing_timestamp' in metadata
    assert 'language' in metadata

@pytest.mark.asyncio
async def test_process_pdf_images(processor, test_pdf_path, mocker):
    """Test processing of PDF images."""
    # Mock pdf2image
    mock_images = [Image.new('RGB', (800, 600), color='white')]
    mocker.patch('pdf2image.convert_from_path', return_value=mock_images)
    
    # Mock OCR processing
    mock_result = {
        'text_blocks': [],
        'full_text': 'Test',
        'embedding': [0.1] * 1536,
        'metadata': {},
        'confidence_score': 90.0
    }
    mocker.patch.object(
        processor,
        'process_image',
        return_value=mock_result
    )
    
    results = await processor.process_pdf_images(test_pdf_path)
    
    assert len(results) == 1
    assert 'page_number' in results[0]
    assert results[0]['full_text'] == 'Test'

def test_save_temp_image(processor, test_image):
    """Test saving temporary images."""
    temp_path = processor._save_temp_image(test_image)
    
    assert os.path.exists(temp_path)
    assert temp_path.endswith('.png')
    
    # Clean up
    os.unlink(temp_path)

def test_global_instance():
    """Test global OCR processor instance."""
    assert ocr_processor is not None
    assert isinstance(ocr_processor, OCRProcessor)

@pytest.mark.asyncio
async def test_error_handling(processor, test_image_path, mocker):
    """Test error handling in OCR processing."""
    # Mock pytesseract to raise an error
    mocker.patch(
        'pytesseract.image_to_data',
        side_effect=Exception("OCR Error")
    )
    
    with pytest.raises(Exception) as exc_info:
        await processor.process_image(test_image_path)
    
    assert "OCR Error" in str(exc_info.value)

def test_temp_directory_creation(tmp_path):
    """Test temporary directory creation."""
    temp_dir = tmp_path / "temp"
    processor = OCRProcessor()
    processor.temp_dir = temp_dir
    
    assert temp_dir.exists()
    assert temp_dir.is_dir()
