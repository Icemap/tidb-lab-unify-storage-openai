LLM_RESPONSE_STYLE = """
<style>
.llm-response {
    background: rgba(255, 255, 255, 0.05);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin: 15px 0;
    font-size: 1.1em;
    line-height: 1.6;
    color: #e1e4e8;
}
.llm-response:hover {
    background: rgba(255, 255, 255, 0.08);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}
</style>
"""

# RAG prompt template
RAG_PROMPT_TEMPLATE = """Answer the question based on the following reference information.

Reference Information:
{context}

Question: {question}

Please answer:"""