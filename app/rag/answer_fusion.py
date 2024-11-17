from typing import List, Dict, Any
import openai
from openai import AzureOpenAI
import json
from config.settings import settings

# Initialize OpenAI clients
openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

azure_client = None
if settings.AZURE_OPENAI_API_KEY:
    azure_client = AzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_API_BASE
    )

def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    use_azure: bool = False
) -> str:
    """
    Generate a coherent answer from retrieved chunks using GPT.
    
    Args:
        query: User's question
        chunks: Retrieved and reranked chunks
        use_azure: Whether to use Azure OpenAI endpoint
    
    Returns:
        Generated answer
    """
    if not chunks:
        return "I couldn't find any relevant information to answer your question."
    
    # Prepare context from chunks
    context = prepare_context(chunks)
    
    # Generate system message based on query type
    system_message = generate_system_message(query)
    
    # Prepare messages for chat completion
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Question: {query}\n\nContext: {context}"}
    ]
    
    try:
        # Try primary client first
        if use_azure and azure_client:
            response = azure_client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                temperature=settings.DEFAULT_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
        else:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=settings.DEFAULT_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Fallback to other client if available
        if use_azure:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=settings.DEFAULT_TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS
                )
                return response.choices[0].message.content
            except Exception as e2:
                return f"Error generating response: {str(e)}, {str(e2)}"
        else:
            if azure_client:
                try:
                    response = azure_client.chat.completions.create(
                        model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                        messages=messages,
                        temperature=settings.DEFAULT_TEMPERATURE,
                        max_tokens=settings.MAX_TOKENS
                    )
                    return response.choices[0].message.content
                except Exception as e2:
                    return f"Error generating response: {str(e)}, {str(e2)}"
            else:
                return f"Error generating response: {str(e)}"

def prepare_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Prepare context from chunks for the language model.
    
    - Orders chunks by relevance
    - Formats metadata and content
    - Ensures context fits within token limits
    """
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        # Format chunk with metadata
        chunk_text = f"\nChunk {i}:\n"
        
        # Add relevant metadata
        metadata = chunk.get("metadata", {})
        if metadata:
            if "section_header" in metadata:
                chunk_text += f"Section: {metadata['section_header']}\n"
            if "doc_type" in metadata:
                chunk_text += f"Document Type: {metadata['doc_type']}\n"
        
        # Add content
        chunk_text += f"Content: {chunk['content']}\n"
        
        # Add relevance score
        chunk_text += f"Relevance Score: {chunk.get('score', 0):.2f}\n"
        
        context_parts.append(chunk_text)
    
    return "\n".join(context_parts)

def generate_system_message(query: str) -> str:
    """
    Generate appropriate system message based on query type.
    """
    # Base system message
    base_message = (
        "You are a helpful AI assistant that provides accurate and relevant "
        "information based on the given context. "
        "Your responses should be:"
    )
    
    # Detect query type and customize message
    query = query.lower()
    
    if any(word in query for word in ["compare", "difference", "versus", "vs"]):
        return base_message + """
        - Structured as a clear comparison
        - Highlighting key differences and similarities
        - Using bullet points or tables when appropriate
        - Balanced in presenting all sides
        Always maintain objectivity and cite specific information from the context.
        """
    
    elif any(word in query for word in ["summarize", "overview", "brief"]):
        return base_message + """
        - Concise and to the point
        - Covering the main ideas only
        - Organized in a logical flow
        - Highlighting key takeaways
        Focus on providing a high-level understanding while maintaining accuracy.
        """
    
    elif any(word in query for word in ["explain", "how", "why"]):
        return base_message + """
        - Detailed and thorough
        - Using clear explanations
        - Including relevant examples
        - Breaking down complex concepts
        Ensure the explanation is clear and builds understanding progressively.
        """
    
    elif any(word in query for word in ["list", "what are", "examples"]):
        return base_message + """
        - Organized in a clear list format
        - Using bullet points or numbering
        - Providing brief explanations for each item
        - Ensuring completeness
        Present the information in an easily scannable format.
        """
    
    else:
        return base_message + """
        - Clear and direct
        - Well-structured
        - Supported by the context
        - Professional in tone
        Always base your response on the provided context and maintain accuracy.
        """

def format_answer(
    answer: str,
    query_type: str = None
) -> str:
    """
    Format the generated answer based on query type and content.
    
    Args:
        answer: Raw generated answer
        query_type: Optional query type for specialized formatting
    
    Returns:
        Formatted answer
    """
    # Remove any references to being an AI
    answer = re.sub(
        r'\b(I am|I\'m|as an AI|as an assistant)\b',
        '',
        answer,
        flags=re.IGNORECASE
    )
    
    # Clean up multiple newlines
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    
    # Format lists consistently
    answer = re.sub(r'(?m)^[-*]\s', '• ', answer)
    
    # Add section breaks for long answers
    if len(answer.split('\n\n')) > 3:
        answer = re.sub(r'\n\n(?=[A-Z])', '\n\n---\n\n', answer)
    
    return answer.strip()

def merge_similar_chunks(
    chunks: List[Dict[str, Any]],
    similarity_threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Merge chunks with similar content to reduce redundancy.
    
    Args:
        chunks: List of chunks to merge
        similarity_threshold: Threshold for considering chunks similar
    
    Returns:
        List of merged chunks
    """
    if not chunks:
        return []
    
    merged_chunks = []
    merged_indices = set()
    
    for i, chunk1 in enumerate(chunks):
        if i in merged_indices:
            continue
        
        similar_chunks = []
        for j, chunk2 in enumerate(chunks[i+1:], i+1):
            if j in merged_indices:
                continue
            
            # Compare embeddings if available
            if "embedding" in chunk1 and "embedding" in chunk2:
                similarity = compute_similarity(chunk1["embedding"], chunk2["embedding"])
                if similarity > similarity_threshold:
                    similar_chunks.append(chunk2)
                    merged_indices.add(j)
        
        if similar_chunks:
            # Merge similar chunks
            merged_content = chunk1["content"]
            merged_score = chunk1.get("score", 0)
            
            for chunk in similar_chunks:
                merged_content += f"\n{chunk['content']}"
                merged_score = max(merged_score, chunk.get("score", 0))
            
            merged_chunks.append({
                "content": merged_content,
                "score": merged_score,
                "metadata": chunk1["metadata"]
            })
        else:
            merged_chunks.append(chunk1)
    
    return merged_chunks
