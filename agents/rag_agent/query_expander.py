import logging
from typing import Dict, Any


class QueryExpander:
    """
    Conservative medical query expansion.

    The purpose of expansion is ONLY to improve terminology matching.
    It must NOT predict, invent, or enumerate possible answers.
    """

    def __init__(self, config):
        self.logger = logging.getLogger(f"{self.__module__}")
        self.config = config
        self.model = config.rag.llm

    def expand_query(self, original_query: str) -> Dict[str, Any]:
        """
        Expand the user's query conservatively.

        Returns:
            Dictionary containing the original and expanded query.
        """

        self.logger.info(
            f"Expanding query: {original_query}"
        )

        expanded_query = self._generate_expansions(
            original_query
        )

        return {
            "original_query": original_query,
            "expanded_query": expanded_query,
        }

    def _generate_expansions(self, query: str) -> str:
        """
        Generate a conservative retrieval query.

        IMPORTANT:
        - Do not answer the question.
        - Do not list possible answers.
        - Do not add symptoms, treatments, causes, risk factors,
          tests, or diseases that were not present in the query.
        - Only add genuine synonyms/acronyms/terminology.
        """

        prompt = f"""
You are a query normalization assistant for a medical
Retrieval-Augmented Generation (RAG) system.

Your ONLY job is to rewrite the user's query so that it is
easier to search in a medical knowledge base.

STRICT RULES:

1. Preserve the original meaning exactly.

2. Do NOT answer the user's question.

3. Do NOT predict possible answers.

4. Do NOT add lists of symptoms, causes, risk factors,
   treatments, medications, tests, biomarkers, diseases,
   complications, or other facts that were not explicitly
   mentioned by the user.

5. Only add:
   - common medical synonyms
   - standard medical terminology
   - common abbreviations/acronyms
   - equivalent terminology that refers to the SAME concept

6. Do NOT introduce another medical specialty or unrelated
   concept.

7. If no useful terminology expansion is necessary,
   return the original query unchanged.

8. Keep the expanded query concise.

Examples:

User:
"What are the symptoms of diabetes?"

Good:
"What are the symptoms of diabetes mellitus (diabetes)?"

Bad:
"What are the symptoms of diabetes including polyuria,
polydipsia, polyphagia, weight loss, blurred vision,
neuropathy, retinopathy..."

User:
"What is high blood pressure?"

Good:
"What is high blood pressure (hypertension)?"

User:
"What is HbA1c?"

Good:
"What is HbA1c (glycated hemoglobin, hemoglobin A1c)?"

User:
"What are cardiovascular disease risk factors?"

Good:
"What are cardiovascular disease (CVD) risk factors?"

User:
"What deep learning techniques detect brain tumors?"

Good:
"What deep learning techniques are used to detect
brain tumors (brain neoplasms)?"

Return ONLY the normalized query.

User Query:
{query}
"""

        try:
            expansion = self.model.invoke(prompt)

            expanded_query = expansion.content.strip()

            if not expanded_query:
                return query

            return expanded_query

        except Exception as e:
            self.logger.error(
                f"Query expansion failed: {e}"
            )

            # Safe fallback: always use the user's original query.
            return query
