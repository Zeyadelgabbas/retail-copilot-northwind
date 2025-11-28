from pathlib import Path
from dataclasses import dataclass
from typing import List
import re
from rank_bm25 import BM25Okapi


@dataclass
class DocChunk:
    """A single chunk of text with metadata for retrieval."""
    chunk_id: str
    content: str
    source: str
    score: float = 0.0


class BM25Retriever:
    def __init__(self, docs_dir="docs"):
        self.docs_dir = Path(docs_dir)
        self.chunks: List[DocChunk] = []
        self.tokenized_chunks = []
        self.bm25 = None
        
        self._load_documents()
        self._build_index()

    def _chunk_markdown_headers(self, content: str, source: str) -> List[DocChunk]:
        """Chunk by ## headers (Marketing Calendar & KPI Definitions)."""
        # Split on ## headers
        raw_sections = re.split(r'\n## ', content)
        
        chunks = []
        idx = 0
        
        # First section is title/header
        header = raw_sections[0].strip()
        
        # Process each section
        for section in raw_sections[1:]:
            section_content = f"{header}\n\n## {section.strip()}"
            chunks.append(DocChunk(
                chunk_id=f"{source}::chunk{idx}",
                content=section_content,
                source=source
            ))
            idx += 1
        
        return chunks

    def _policy_chunks(self, content: str, source: str) -> List[DocChunk]:
        """Chunk by bullet points in product policy docs"""
        # Split on bullet points
        raw_sections = re.split(r'\n- ', content)
        
        chunks = []
        idx = 0
        
        # First section is title
        header = raw_sections[0].strip()
        
        # Process each bullet point
        for section in raw_sections[1:]:
            section_content = f"{header}\n\n- {section.strip()}"
            chunks.append(DocChunk(
                chunk_id=f"{source}::chunk{idx}",
                content=section_content,
                source=source
            ))
            idx += 1
        
        return chunks

    def _single_chunk(self, content: str, source: str) -> List[DocChunk]:
        """Used for catalog (short doc)."""
        return [
            DocChunk(
                chunk_id=f"{source}::chunk0",
                content=content.strip(),
                source=source
            )
        ]
    
    def _load_documents(self):
        """Load and chunk documents"""
        for file in self.docs_dir.glob("*.md"):
            content = file.read_text(encoding="utf-8")
            source = file.stem
            
            # Smart chunking based on filename
            if source in ["marketing_calendar", "kpi_definitions"]:
                chunks = self._chunk_markdown_headers(content, source)
            elif source == 'catalog':
                chunks = self._single_chunk(content, source)
            elif source == 'product_policy':
                chunks = self._policy_chunks(content, source)
            else:
                # Default: chunk by paragraphs
                chunks = self._policy_chunks(content, source)
            
            self.chunks.extend(chunks)
        
        print(f"Loaded {len(self.chunks)} chunks from {len(list(self.docs_dir.glob('*.md')))} documents")
        

    def _tokenize(self, text: str):
        """Simple tokenization"""
        return text.lower().split()

    def _build_index(self):
        """Build BM25 index"""
        self.tokenized_chunks = [self._tokenize(c.content) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[DocChunk]:
        """Search for relevant chunks"""
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        
        # Get top-k indices
        best = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        results = []
        for idx in best:
            chunk = self.chunks[idx]
            results.append(
                DocChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    source=chunk.source,
                    score=float(scores[idx])
                )
            )
        
        return results
    
    def get_context_string(self, chunks: List[DocChunk]) -> str:
        """Format chunks into context string"""
        if not chunks:
            return ""
        
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"[{chunk.chunk_id}]\n{chunk.content}")
        
        return "\n\n".join(context_parts)