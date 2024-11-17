from typing import List, Dict, Any, Optional
import tabula
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
import logging
from PIL import Image
import pytesseract
import pdf2image

from app.document_processing.embeddings import generate_embeddings
from config.settings import settings

logger = logging.getLogger(__name__)

class TableExtractor:
    """Extract and process tables from documents."""
    
    def __init__(self):
        """Initialize table extractor with default settings."""
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    async def extract_tables_from_pdf(
        self,
        file_path: str,
        pages: str = 'all'
    ) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF file.
        
        Args:
            file_path: Path to PDF file
            pages: Pages to process ('all' or specific pages)
        
        Returns:
            List of dictionaries containing table data and metadata
        """
        try:
            # Extract tables using tabula
            tables = tabula.read_pdf(
                file_path,
                pages=pages,
                multiple_tables=True,
                guess=True,
                lattice=True,
                stream=True
            )
            
            processed_tables = []
            for idx, table in enumerate(tables):
                if not table.empty:
                    # Process and clean table
                    cleaned_table = self._clean_table(table)
                    
                    # Generate table metadata
                    metadata = self._generate_table_metadata(cleaned_table, idx)
                    
                    # Convert table to structured format
                    table_data = self._convert_table_to_dict(cleaned_table)
                    
                    # Generate embeddings for table content
                    table_text = self._table_to_text(cleaned_table)
                    embedding = await generate_embeddings(table_text)
                    
                    processed_tables.append({
                        'table_id': idx,
                        'data': table_data,
                        'metadata': metadata,
                        'embedding': embedding,
                        'text_content': table_text
                    })
            
            return processed_tables
            
        except Exception as e:
            logger.error(f"Error extracting tables from PDF: {str(e)}")
            raise
    
    async def extract_tables_from_image(
        self,
        image_path: str
    ) -> List[Dict[str, Any]]:
        """
        Extract tables from image using OCR.
        
        Args:
            image_path: Path to image file
        
        Returns:
            List of dictionaries containing table data and metadata
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Extract text using OCR
            ocr_data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DATAFRAME
            )
            
            # Identify table structures
            tables = self._identify_tables_in_ocr(ocr_data)
            
            processed_tables = []
            for idx, table in enumerate(tables):
                # Clean and structure table
                cleaned_table = self._clean_table(table)
                
                # Generate metadata
                metadata = self._generate_table_metadata(cleaned_table, idx)
                
                # Convert to structured format
                table_data = self._convert_table_to_dict(cleaned_table)
                
                # Generate embeddings
                table_text = self._table_to_text(cleaned_table)
                embedding = await generate_embeddings(table_text)
                
                processed_tables.append({
                    'table_id': idx,
                    'data': table_data,
                    'metadata': metadata,
                    'embedding': embedding,
                    'text_content': table_text
                })
            
            return processed_tables
            
        except Exception as e:
            logger.error(f"Error extracting tables from image: {str(e)}")
            raise
    
    def _clean_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize table data."""
        # Remove empty rows and columns
        table = table.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # Clean cell values
        table = table.applymap(lambda x: str(x).strip() if pd.notnull(x) else '')
        
        # Ensure column names are strings
        table.columns = table.columns.astype(str)
        
        return table
    
    def _generate_table_metadata(
        self,
        table: pd.DataFrame,
        table_id: int
    ) -> Dict[str, Any]:
        """Generate metadata for table."""
        return {
            'table_id': table_id,
            'rows': len(table),
            'columns': len(table.columns),
            'column_names': list(table.columns),
            'empty_cells': table.isna().sum().sum(),
            'numeric_columns': list(table.select_dtypes(include=[np.number]).columns),
            'has_header': True,  # Assuming first row is header
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _convert_table_to_dict(self, table: pd.DataFrame) -> Dict[str, Any]:
        """Convert table to structured dictionary format."""
        return {
            'headers': list(table.columns),
            'data': table.values.tolist(),
            'shape': table.shape
        }
    
    def _table_to_text(self, table: pd.DataFrame) -> str:
        """Convert table to text format for embedding."""
        text_parts = []
        
        # Add headers
        headers = " | ".join(str(col) for col in table.columns)
        text_parts.append(f"Headers: {headers}")
        
        # Add data rows
        for idx, row in table.iterrows():
            row_text = " | ".join(str(cell) for cell in row)
            text_parts.append(row_text)
        
        return "\n".join(text_parts)
    
    def _identify_tables_in_ocr(
        self,
        ocr_data: pd.DataFrame
    ) -> List[pd.DataFrame]:
        """Identify table structures in OCR output."""
        tables = []
        current_table = []
        
        # Group text blocks by position
        blocks = ocr_data[ocr_data['conf'] > 30].groupby('block_num')
        
        for _, block in blocks:
            # Check if block forms table structure
            if self._is_table_structure(block):
                rows = self._extract_table_rows(block)
                if rows:
                    table = pd.DataFrame(rows)
                    tables.append(table)
        
        return tables
    
    def _is_table_structure(self, block: pd.DataFrame) -> bool:
        """Determine if text block represents a table structure."""
        # Check for regular spacing and alignment
        x_positions = block['left'].unique()
        y_positions = block['top'].unique()
        
        # Table likely has multiple aligned columns and rows
        return len(x_positions) > 1 and len(y_positions) > 1
    
    def _extract_table_rows(self, block: pd.DataFrame) -> List[List[str]]:
        """Extract rows from table structure."""
        rows = []
        
        # Group by vertical position (rows)
        y_groups = block.groupby('top')
        
        for _, row_group in y_groups:
            # Sort by horizontal position (columns)
            row_group = row_group.sort_values('left')
            row = row_group['text'].tolist()
            if row:
                rows.append(row)
        
        return rows

# Global table extractor instance
table_extractor = TableExtractor()
