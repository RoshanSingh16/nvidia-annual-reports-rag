import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient

load_dotenv()

class RAGQueryEngine:
    def __init__(self, collection_name: str = "nvidia_reports", db_path: str = "qdrant_storage"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing in .env file!")

        genai.configure(api_key=api_key)
        self.qdrant = QdrantClient(path=db_path)
        self.collection_name = collection_name

    def retrieve(self, query: str, top_k: int = 8):
        # Gemini embedding
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        query_vector = response["embedding"]

        # Vector search using qdrant query_points
        response_points = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )
        return [hit.payload for hit in response_points.points]

    def answer_question(self, query: str):
        retrieved_chunks = self.retrieve(query, top_k=8)

        if not retrieved_chunks:
            return {
                "answer": "I do not have sufficient information in the provided NVIDIA disclosures to answer this question.",
                "sources": [],
                "chart_json": None
            }

        # Build Context String
        context_blocks = []
        for c in retrieved_chunks:
            doc_title = c.get('document_title', c.get('source', 'NVIDIA Report'))
            fiscal_year = c.get('fiscal_year', 'N/A')
            page_num = c.get('page_number', c.get('page', 'N/A'))
            text = c.get('text', '')
            block = f"--- SOURCE: {doc_title} | Fiscal Year: {fiscal_year} | Page: {page_num} ---\n{text}"
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)

        system_prompt = """You are an expert AI assistant specializing in NVIDIA financial reports and disclosures (1998-2026).
STRICT RULES:
1. Answer ONLY using the provided retrieved context.
2. Every factual claim or number MUST be accompanied by an exact inline page citation, e.g., [NVIDIA 10-K 2024, Page 42].
3. Maintain exact numerical precision, reporting units (Millions vs Billions), and fiscal periods.
4. ABSTENTION: If the retrieved context does NOT contain enough evidence to answer the question, state: "I do not have sufficient information in the provided NVIDIA disclosures to answer this question." Do NOT fabricate information.
5. CHART GENERATION: If the user asks for trends, comparisons, or data over time (e.g., revenue over years), include a JSON block at the end of your response under ```json_chart ... ``` with keys: `title`, `x_label`, `y_label`, `x_values` (list), `y_values` (list).
"""

        user_prompt = f"Retrieved Context:\n{context_str}\n\nUser Question: {query}"

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_prompt
        )

        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )

        full_reply = response.text

        # Extract Chart JSON if generated
        chart_data = None
        if "```json_chart" in full_reply:
            try:
                parts = full_reply.split("```json_chart")
                json_part = parts[1].split("```")[0].strip()
                chart_data = json.loads(json_part)
                full_reply = parts[0].strip() + "\n" + parts[1].split("```")[1].strip()
            except Exception:
                pass

        return {
            "answer": full_reply,
            "sources": retrieved_chunks,
            "chart_json": chart_data
        }