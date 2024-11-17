import pytest
from pathlib import Path
import tempfile
import os

from app.document_processing.code_parser import CodeParser, code_parser

@pytest.fixture
def parser():
    """Create code parser instance."""
    return CodeParser()

@pytest.fixture
def python_code():
    """Sample Python code for testing."""
    return '''
class TestClass:
    """Test class docstring."""
    def __init__(self, param):
        self.param = param
    
    def test_method(self):
        """Test method docstring."""
        return self.param

def test_function(arg1, arg2):
    """Test function docstring."""
    return arg1 + arg2

import os
from datetime import datetime

TEST_VARIABLE = "test"
'''

@pytest.fixture
def javascript_code():
    """Sample JavaScript code for testing."""
    return '''
class TestClass extends BaseClass {
    constructor(param) {
        super();
        this.param = param;
    }
    
    testMethod() {
        return this.param;
    }
}

const testFunction = async (arg1, arg2) => {
    return arg1 + arg2;
};

import { something } from 'somewhere';
const TEST_VARIABLE = "test";
'''

@pytest.fixture
def test_file_path(tmp_path):
    """Create temporary file for testing."""
    return str(tmp_path / "test_file.py")

@pytest.mark.asyncio
async def test_parse_python_code(parser, python_code, test_file_path):
    """Test parsing Python code."""
    # Write test code to file
    with open(test_file_path, 'w') as f:
        f.write(python_code)
    
    result = await parser.parse_code_file(test_file_path)
    
    assert result["language"] == "Python"
    assert "highlighted_code" in result
    assert "structure" in result
    
    structure = result["structure"]
    assert len(structure["classes"]) == 1
    assert len(structure["functions"]) == 1
    assert len(structure["imports"]) == 2
    assert len(structure["variables"]) == 1
    assert len(structure["docstrings"]) == 3  # Class, method, and function

def test_analyze_python_code(parser, python_code):
    """Test Python code structure analysis."""
    structure = parser._analyze_python_code(python_code)
    
    assert structure["classes"][0]["name"] == "TestClass"
    assert structure["functions"][0]["name"] == "test_function"
    assert "datetime" in structure["imports"]
    assert structure["variables"][0]["name"] == "TEST_VARIABLE"
    assert any("Test class docstring" in d["docstring"] for d in structure["docstrings"])

def test_analyze_js_code(parser, javascript_code):
    """Test JavaScript code structure analysis."""
    structure = parser._analyze_js_code(javascript_code)
    
    assert structure["classes"][0]["name"] == "TestClass"
    assert "testFunction" in [f["name"] for f in structure["functions"]]
    assert "somewhere" in structure["imports"]
    assert "TEST_VARIABLE" in [v["name"] for v in structure["variables"]]

def test_analyze_generic_code(parser):
    """Test generic code structure analysis."""
    generic_code = '''
class TestClass {
    void testMethod() {
        return null;
    }
}
'''
    structure = parser._analyze_generic_code(generic_code)
    
    assert structure["classes"][0]["name"] == "TestClass"
    assert "testMethod" in [f["name"] for f in structure["functions"]]

def test_calculate_complexity(parser):
    """Test code complexity calculation."""
    code = '''
if (condition) {
    while (true) {
        if (x && y) {
            break;
        }
    }
} else if (other) {
    for (let i = 0; i < 10; i++) {
        console.log(i);
    }
}
'''
    complexity = parser._calculate_complexity(code)
    assert complexity["cyclomatic"] > 1
    assert complexity["lines"] > 0
    assert complexity["characters"] > 0

@pytest.mark.asyncio
async def test_generate_code_chunks(parser, python_code):
    """Test code chunk generation and embedding."""
    structure = parser._analyze_python_code(python_code)
    chunks = await parser._generate_code_chunks(python_code, structure)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert "type" in chunk
        assert "name" in chunk
        assert "content" in chunk
        assert "embedding" in chunk
        assert len(chunk["embedding"]) == 1536  # OpenAI embedding dimension

def test_extract_class_content(parser, python_code):
    """Test class content extraction."""
    content = parser._extract_class_content(python_code, "TestClass")
    
    assert content is not None
    assert "TestClass" in content
    assert "test_method" in content
    assert "__init__" in content

def test_extract_function_content(parser, python_code):
    """Test function content extraction."""
    content = parser._extract_function_content(python_code, "test_function")
    
    assert content is not None
    assert "test_function" in content
    assert "arg1" in content
    assert "arg2" in content

def test_generate_metadata(parser, python_code):
    """Test metadata generation."""
    structure = parser._analyze_python_code(python_code)
    metadata = parser._generate_metadata(python_code, "Python", structure)
    
    assert metadata["language"] == "Python"
    assert metadata["size"] > 0
    assert metadata["lines"] > 0
    assert metadata["classes"] == 1
    assert metadata["functions"] == 1
    assert "complexity" in metadata
    assert "processing_timestamp" in metadata

def test_supported_languages(parser):
    """Test supported language detection."""
    assert parser.supported_extensions[".py"] == "Python"
    assert parser.supported_extensions[".js"] == "JavaScript"
    assert parser.supported_extensions[".java"] == "Java"

@pytest.mark.asyncio
async def test_error_handling(parser, test_file_path):
    """Test error handling for invalid code."""
    # Write invalid code to file
    with open(test_file_path, 'w') as f:
        f.write("class Invalid Syntax{")
    
    with pytest.raises(Exception):
        await parser.parse_code_file(test_file_path)

def test_global_instance():
    """Test global code parser instance."""
    assert code_parser is not None
    assert isinstance(code_parser, CodeParser)

@pytest.mark.asyncio
async def test_multiple_languages(parser, tmp_path):
    """Test parsing different programming languages."""
    languages = {
        "test.py": "class Test:\n    pass",
        "test.js": "class Test { }",
        "test.java": "public class Test { }",
        "test.cpp": "class Test { };",
    }
    
    for filename, code in languages.items():
        file_path = tmp_path / filename
        with open(file_path, 'w') as f:
            f.write(code)
        
        result = await parser.parse_code_file(str(file_path))
        assert result["language"] in parser.supported_extensions.values()
        assert len(result["structure"]["classes"]) > 0
