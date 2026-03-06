"""RAG pipeline for question answering with citations."""

from typing import Optional

from openai import OpenAI

from src.config import settings
from src.models.schema import Block, BoundingBox, Chunk, Citation, DocumentGraph, QAResponse
from src.retrieval.retriever import Retriever


class RAGPipeline:
    """RAG pipeline for answering questions about rulebooks.

    All configuration is loaded from settings.
    """

    SYSTEM_PROMPT = """You are a helpful assistant that answers questions about board game rules.

You will be given relevant excerpts from a rulebook. Use these excerpts to answer the user's question accurately.

Guidelines:
- Only answer based on the provided context. If the context doesn't contain the answer, say so.
- Be precise and cite specific rules when possible.
- When referencing rules, mention the page number if available.
- If there's ambiguity in the rules, explain the different interpretations.
- Keep answers concise but complete.

Format your response as:
1. A direct answer to the question
2. Supporting details from the rules
3. Page references (e.g., "According to page 7...")"""

    def __init__(
        self,
        retriever: Retriever,
        graph: Optional[DocumentGraph] = None,
    ):
        """Initialize the RAG pipeline.

        Args:
            retriever: The retriever for finding relevant chunks.
            graph: Document graph for block lookups.
        """
        self.retriever = retriever
        self.model = settings.openai_chat_model
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._graph = graph

    def answer(
        self,
        question: str,
        top_k: int = 5,
        expand_context: bool = True,
    ) -> QAResponse:
        """Answer a question about the rulebook.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve.
            expand_context: Whether to expand context using the graph.

        Returns:
            QAResponse with answer and citations.
        """
        retrieved = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            expand_context=expand_context,
        )

        context = self._build_context(retrieved)

        answer_text = self._generate_answer(question, context)

        citations = self._extract_citations(retrieved)

        chunks = [chunk for chunk, _, _ in retrieved]

        confidence = self._estimate_confidence(retrieved)

        return QAResponse(
            question=question,
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            retrieved_chunks=chunks,
        )

    def _build_context(
        self,
        retrieved: list[tuple[Chunk, float, list[Block]]],
    ) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []

        for i, (chunk, score, context_blocks) in enumerate(retrieved, 1):
            page_ref = f"Page {chunk.pdf_page + 1}"
            if chunk.mini_page is not None:
                page_ref += f", Section {chunk.mini_page + 1}"

            header = f"[Excerpt {i}] ({page_ref})"
            if chunk.heading:
                header += f" - {chunk.heading}"

            context_text = chunk.text

            if context_blocks:
                before_text = "\n".join(
                    b.text for b in context_blocks if b.reading_order < min(
                        self._get_block_reading_order(chunk, b.block_id) for b in context_blocks
                        if self._get_block_reading_order(chunk, b.block_id) is not None
                    ) if self._get_block_reading_order(chunk, b.block_id) is not None
                )
                after_text = "\n".join(
                    b.text for b in context_blocks if b.reading_order > max(
                        self._get_block_reading_order(chunk, b.block_id) for b in context_blocks
                        if self._get_block_reading_order(chunk, b.block_id) is not None
                    ) if self._get_block_reading_order(chunk, b.block_id) is not None
                )

                if before_text:
                    context_text = f"[...] {before_text}\n\n{context_text}"
                if after_text:
                    context_text = f"{context_text}\n\n{after_text} [...]"

            context_parts.append(f"{header}\n{context_text}")

        return "\n\n---\n\n".join(context_parts)

    def _get_block_reading_order(self, chunk: Chunk, block_id) -> Optional[int]:
        """Get reading order for a block in the chunk."""
        if self._graph is None:
            return None
        block = self._graph.get_block(block_id)
        return block.reading_order if block else None

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using the LLM."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context from the rulebook:\n\n{context}\n\nQuestion: {question}",
            },
        ]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )

        return response.choices[0].message.content or ""

    def _extract_citations(
        self,
        retrieved: list[tuple[Chunk, float, list[Block]]],
    ) -> list[Citation]:
        """Extract citations from retrieved chunks."""
        citations = []

        for chunk, score, _ in retrieved:
            snippet = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text

            citation = Citation(
                chunk_id=chunk.chunk_id,
                block_ids=chunk.block_ids,
                pdf_page=chunk.pdf_page,
                mini_page=chunk.mini_page,
                bbox=chunk.bbox_union,
                text_snippet=snippet,
            )
            citations.append(citation)

        return citations

    def _estimate_confidence(
        self,
        retrieved: list[tuple[Chunk, float, list[Block]]],
    ) -> float:
        """Estimate confidence based on retrieval scores."""
        if not retrieved:
            return 0.0

        top_score = retrieved[0][1]

        if len(retrieved) > 1:
            avg_score = sum(score for _, score, _ in retrieved) / len(retrieved)
            confidence = (top_score + avg_score) / 2
        else:
            confidence = top_score

        return min(1.0, max(0.0, confidence))

    def format_response(self, response: QAResponse) -> str:
        """Format a QAResponse for display.

        Args:
            response: The QA response to format.

        Returns:
            Formatted string for terminal output.
        """
        lines = []

        lines.append("=" * 60)
        lines.append("ANSWER")
        lines.append("=" * 60)
        lines.append("")
        lines.append(response.answer)
        lines.append("")

        if response.citations:
            lines.append("-" * 60)
            lines.append("CITATIONS")
            lines.append("-" * 60)
            lines.append("")

            for i, citation in enumerate(response.citations, 1):
                page_ref = citation.format_reference()
                lines.append(f"[{i}] {page_ref}")
                lines.append(f"    \"{citation.text_snippet}\"")
                lines.append("")

        lines.append("-" * 60)
        lines.append(f"Confidence: {response.confidence:.2%}")
        lines.append(f"Chunks retrieved: {len(response.retrieved_chunks)}")
        lines.append("=" * 60)

        return "\n".join(lines)
