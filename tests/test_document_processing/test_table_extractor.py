import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from PIL import Image

from app.document_processing.table_extractor import TableExtractor, table_extractor

@pytest.fixture
def test_pdf_path(tmp_path):
    """Create a test PDF file with tables."""
    # This would typically be a real PDF file
    # For testing, we'll use a sample file path
    return str(tmp_path / "test.pdf")

@pytest.fixture
def test_image_path(tmp_path):
    """Create a test image file with tables."""
    image_path = tmp_path / "test_table.png"
    # Create a simple image with text
    img = Image.new('RGB', (800, 600), color='white')
    return str(image_path)

@pytest.fixture
def sample_table():
    """Create a sample pandas DataFrame representing a table."""
    return pd.DataFrame({
        'Name': ['John', 'Jane', 'Bob'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Paris']
    })

@pytest.fixture
def extractor():
    """Create a TableExtractor instance."""
    return TableExtractor()

def test_clean_table(extractor, sample_table):
    """Test table cleaning functionality."""
    # Add some noise to the table
    dirty_table = sample_table.copy()
    dirty_table.iloc[0, 0] = " John  "  # Extra spaces
    dirty_table.iloc[1, 1] = np.nan  # Missing value
    
    cleaned = extractor._clean_table(dirty_table)
    
    assert cleaned.iloc[0, 0] == "John"  # Spaces removed
    assert cleaned.iloc[1, 1] == ""  # NaN converted to empty string
    assert list(cleaned.columns) == list(sample_table.columns)

def test_generate_table_metadata(extractor, sample_table):
    """Test metadata generation for tables."""
    metadata = extractor._generate_table_metadata(sample_table, table_id=1)
    
    assert metadata['table_id'] == 1
    assert metadata['rows'] == 3
    assert metadata['columns'] == 3
    assert 'column_names' in metadata
    assert 'timestamp' in metadata
    assert metadata['numeric_columns'] == ['Age']

def test_convert_table_to_dict(extractor, sample_table):
    """Test conversion of table to dictionary format."""
    table_dict = extractor._convert_table_to_dict(sample_table)
    
    assert 'headers' in table_dict
    assert 'data' in table_dict
    assert 'shape' in table_dict
    assert table_dict['shape'] == (3, 3)
    assert len(table_dict['data']) == 3
    assert table_dict['headers'] == ['Name', 'Age', 'City']

def test_table_to_text(extractor, sample_table):
    """Test conversion of table to text format."""
    text = extractor._table_to_text(sample_table)
    
    assert "Headers: Name | Age | City" in text
    assert "John | 25 | New York" in text
    assert "Jane | 30 | London" in text
    assert "Bob | 35 | Paris" in text

@pytest.mark.asyncio
async def test_extract_tables_from_pdf(extractor, test_pdf_path, mocker):
    """Test PDF table extraction."""
    # Mock tabula.read_pdf to return sample data
    mock_tables = [pd.DataFrame({
        'Column1': ['A', 'B'],
        'Column2': [1, 2]
    })]
    mocker.patch('tabula.read_pdf', return_value=mock_tables)
    
    # Mock generate_embeddings
    mocker.patch(
        'app.document_processing.embeddings.generate_embeddings',
        return_value=[0.1] * 1536
    )
    
    tables = await extractor.extract_tables_from_pdf(test_pdf_path)
    
    assert len(tables) == 1
    assert 'table_id' in tables[0]
    assert 'data' in tables[0]
    assert 'metadata' in tables[0]
    assert 'embedding' in tables[0]

@pytest.mark.asyncio
async def test_extract_tables_from_image(extractor, test_image_path, mocker):
    """Test image table extraction."""
    # Mock pytesseract output
    mock_ocr_data = pd.DataFrame({
        'text': ['Header1', 'Header2', 'Value1', 'Value2'],
        'block_num': [1, 1, 2, 2],
        'conf': [90, 90, 90, 90],
        'left': [0, 100, 0, 100],
        'top': [0, 0, 50, 50]
    })
    mocker.patch(
        'pytesseract.image_to_data',
        return_value=mock_ocr_data
    )
    
    # Mock generate_embeddings
    mocker.patch(
        'app.document_processing.embeddings.generate_embeddings',
        return_value=[0.1] * 1536
    )
    
    tables = await extractor.extract_tables_from_image(test_image_path)
    
    assert len(tables) > 0
    for table in tables:
        assert 'table_id' in table
        assert 'data' in table
        assert 'metadata' in table
        assert 'embedding' in table

def test_identify_tables_in_ocr(extractor):
    """Test table structure identification in OCR output."""
    # Create sample OCR data with table-like structure
    ocr_data = pd.DataFrame({
        'text': ['A', 'B', 'C', '1', '2', '3'],
        'block_num': [1, 1, 1, 2, 2, 2],
        'conf': [90] * 6,
        'left': [0, 100, 200] * 2,
        'top': [0, 0, 0, 50, 50, 50]
    })
    
    tables = extractor._identify_tables_in_ocr(ocr_data)
    assert len(tables) > 0

def test_is_table_structure(extractor):
    """Test table structure detection."""
    # Create sample block with table-like structure
    table_block = pd.DataFrame({
        'left': [0, 100, 0, 100],
        'top': [0, 0, 50, 50],
        'text': ['A', 'B', 'C', 'D']
    })
    
    # Create sample block without table structure
    non_table_block = pd.DataFrame({
        'left': [0, 0, 0],
        'top': [0, 50, 100],
        'text': ['A', 'B', 'C']
    })
    
    assert extractor._is_table_structure(table_block) is True
    assert extractor._is_table_structure(non_table_block) is False

def test_extract_table_rows(extractor):
    """Test extraction of rows from table structure."""
    block = pd.DataFrame({
        'text': ['A', 'B', '1', '2'],
        'left': [0, 100, 0, 100],
        'top': [0, 0, 50, 50]
    })
    
    rows = extractor._extract_table_rows(block)
    assert len(rows) == 2
    assert rows[0] == ['A', 'B']
    assert rows[1] == ['1', '2']

def test_global_instance():
    """Test global table extractor instance."""
    assert table_extractor is not None
    assert isinstance(table_extractor, TableExtractor)
