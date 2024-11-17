from typing import List, Dict, Any, Optional
import pygments
from pygments.lexers import get_lexer_for_filename, guess_lexer
from pygments.formatters import get_formatter_by_name
from pygments.token import Token
import ast
import re
from pathlib import Path
import logging
from datetime import datetime

from app.document_processing.embeddings import generate_embeddings
from config.settings import settings

logger = logging.getLogger(__name__)

class CodeParser:
    """Parse and analyze code files with syntax highlighting and structure analysis."""
    
    def __init__(self):
        """Initialize code parser with default settings."""
        self.supported_extensions = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.go': 'Go',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.rs': 'Rust',
            '.scala': 'Scala',
            '.kt': 'Kotlin',
            '.r': 'R',
            '.sql': 'SQL'
        }
    
    async def parse_code_file(
        self,
        file_path: str,
        generate_chunks: bool = True
    ) -> Dict[str, Any]:
        """
        Parse code file with syntax highlighting and structure analysis.
        
        Args:
            file_path: Path to code file
            generate_chunks: Whether to generate embeddings for code chunks
        
        Returns:
            Dictionary containing parsed code data and metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get file extension and language
            ext = Path(file_path).suffix.lower()
            language = self.supported_extensions.get(ext, 'Unknown')
            
            # Get lexer for syntax highlighting
            try:
                lexer = get_lexer_for_filename(file_path)
            except:
                lexer = guess_lexer(content)
            
            # Generate syntax highlighted HTML
            formatter = get_formatter_by_name('html')
            highlighted_code = pygments.highlight(content, lexer, formatter)
            
            # Parse code structure
            structure = self._analyze_code_structure(content, language)
            
            # Generate chunks and embeddings if requested
            chunks = []
            if generate_chunks:
                chunks = await self._generate_code_chunks(content, structure)
            
            return {
                "content": content,
                "highlighted_code": highlighted_code,
                "language": language,
                "structure": structure,
                "chunks": chunks,
                "metadata": self._generate_metadata(content, language, structure)
            }
            
        except Exception as e:
            logger.error(f"Error parsing code file: {str(e)}")
            raise
    
    def _analyze_code_structure(
        self,
        content: str,
        language: str
    ) -> Dict[str, Any]:
        """Analyze code structure based on language."""
        structure = {
            "classes": [],
            "functions": [],
            "imports": [],
            "variables": [],
            "complexity": {}
        }
        
        try:
            if language == "Python":
                structure = self._analyze_python_code(content)
            elif language in ["JavaScript", "TypeScript"]:
                structure = self._analyze_js_code(content)
            else:
                # Basic structure analysis for other languages
                structure = self._analyze_generic_code(content)
            
            # Add complexity metrics
            structure["complexity"] = self._calculate_complexity(content)
            
            return structure
            
        except Exception as e:
            logger.warning(f"Error analyzing code structure: {str(e)}")
            return structure
    
    def _analyze_python_code(self, content: str) -> Dict[str, Any]:
        """Analyze Python code structure using AST."""
        structure = {
            "classes": [],
            "functions": [],
            "imports": [],
            "variables": [],
            "docstrings": []
        }
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    structure["classes"].append({
                        "name": node.name,
                        "lineno": node.lineno,
                        "methods": [
                            method.name
                            for method in node.body
                            if isinstance(method, ast.FunctionDef)
                        ]
                    })
                elif isinstance(node, ast.FunctionDef):
                    structure["functions"].append({
                        "name": node.name,
                        "lineno": node.lineno,
                        "args": [arg.arg for arg in node.args.args]
                    })
                elif isinstance(node, ast.Import):
                    structure["imports"].extend(
                        alias.name for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom):
                    structure["imports"].append(
                        f"from {node.module} import " +
                        ", ".join(alias.name for alias in node.names)
                    )
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            structure["variables"].append({
                                "name": target.id,
                                "lineno": target.lineno
                            })
                
                # Extract docstrings
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Module)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        structure["docstrings"].append({
                            "type": type(node).__name__,
                            "name": getattr(node, "name", "module"),
                            "docstring": docstring
                        })
            
            return structure
            
        except Exception as e:
            logger.warning(f"Error analyzing Python code: {str(e)}")
            return structure
    
    def _analyze_js_code(self, content: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript code structure using regex."""
        structure = {
            "classes": [],
            "functions": [],
            "imports": [],
            "variables": []
        }
        
        # Class pattern
        class_pattern = r"class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{"
        for match in re.finditer(class_pattern, content):
            structure["classes"].append({
                "name": match.group(1),
                "lineno": content[:match.start()].count('\n') + 1
            })
        
        # Function pattern
        function_pattern = r"(?:function|const|let|var)\s+(\w+)\s*=?\s*(?:\(|\basync\b\s*\()"
        for match in re.finditer(function_pattern, content):
            structure["functions"].append({
                "name": match.group(1),
                "lineno": content[:match.start()].count('\n') + 1
            })
        
        # Import pattern
        import_pattern = r"(?:import|require)\s*\(?['\"]([^'\"]+)['\"]"
        structure["imports"].extend(
            re.findall(import_pattern, content)
        )
        
        # Variable pattern
        var_pattern = r"(?:const|let|var)\s+(\w+)\s*="
        for match in re.finditer(var_pattern, content):
            structure["variables"].append({
                "name": match.group(1),
                "lineno": content[:match.start()].count('\n') + 1
            })
        
        return structure
    
    def _analyze_generic_code(self, content: str) -> Dict[str, Any]:
        """Basic code structure analysis for unsupported languages."""
        structure = {
            "classes": [],
            "functions": [],
            "imports": [],
            "variables": []
        }
        
        # Generic patterns
        patterns = {
            "class": r"class\s+(\w+)",
            "function": r"(?:function|def|func)\s+(\w+)\s*\(",
            "import": r"(?:import|include|require)\s+[\"']?([^\s\"']+)",
            "variable": r"(?:var|let|const)\s+(\w+)\s*="
        }
        
        for pattern_type, pattern in patterns.items():
            for match in re.finditer(pattern, content):
                structure[f"{pattern_type}s"].append({
                    "name": match.group(1),
                    "lineno": content[:match.start()].count('\n') + 1
                })
        
        return structure
    
    def _calculate_complexity(self, content: str) -> Dict[str, Any]:
        """Calculate code complexity metrics."""
        return {
            "lines": len(content.splitlines()),
            "characters": len(content),
            "cyclomatic": self._calculate_cyclomatic_complexity(content)
        }
    
    def _calculate_cyclomatic_complexity(self, content: str) -> int:
        """Calculate cyclomatic complexity."""
        # Count decision points
        decision_patterns = [
            r'\bif\b',
            r'\bwhile\b',
            r'\bfor\b',
            r'\bcase\b',
            r'\bcatch\b',
            r'\b&&\b',
            r'\b\|\|\b'
        ]
        
        complexity = 1  # Base complexity
        for pattern in decision_patterns:
            complexity += len(re.findall(pattern, content))
        
        return complexity
    
    async def _generate_code_chunks(
        self,
        content: str,
        structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for code chunks."""
        chunks = []
        
        # Split code into logical chunks
        chunk_candidates = []
        
        # Add class definitions as chunks
        for class_info in structure.get("classes", []):
            class_content = self._extract_class_content(
                content,
                class_info["name"]
            )
            if class_content:
                chunk_candidates.append({
                    "type": "class",
                    "name": class_info["name"],
                    "content": class_content
                })
        
        # Add function definitions as chunks
        for func_info in structure.get("functions", []):
            func_content = self._extract_function_content(
                content,
                func_info["name"]
            )
            if func_content:
                chunk_candidates.append({
                    "type": "function",
                    "name": func_info["name"],
                    "content": func_content
                })
        
        # Process chunks and generate embeddings
        for candidate in chunk_candidates:
            embedding = await generate_embeddings(candidate["content"])
            chunks.append({
                "type": candidate["type"],
                "name": candidate["name"],
                "content": candidate["content"],
                "embedding": embedding
            })
        
        return chunks
    
    def _extract_class_content(
        self,
        content: str,
        class_name: str
    ) -> Optional[str]:
        """Extract class definition and body."""
        pattern = f"class\\s+{class_name}[^{{]*{{([^}}]*)}}"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(0) if match else None
    
    def _extract_function_content(
        self,
        content: str,
        function_name: str
    ) -> Optional[str]:
        """Extract function definition and body."""
        pattern = f"(?:function\\s+{function_name}|const\\s+{function_name}\\s*=\\s*(?:async\\s*)?\\(|def\\s+{function_name})\\s*\\([^{{]*{{([^}}]*)}}"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(0) if match else None
    
    def _generate_metadata(
        self,
        content: str,
        language: str,
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate metadata for code file."""
        return {
            "language": language,
            "size": len(content),
            "lines": len(content.splitlines()),
            "classes": len(structure.get("classes", [])),
            "functions": len(structure.get("functions", [])),
            "imports": len(structure.get("imports", [])),
            "variables": len(structure.get("variables", [])),
            "complexity": structure.get("complexity", {}),
            "processing_timestamp": datetime.utcnow().isoformat()
        }

# Global code parser instance
code_parser = CodeParser()
