from IPython.display import Markdown, display
from langchain_core.documents import Document
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class RAGPipeline:
    def __init__(
        self,
        retriever,
        gpt_client,
        reranker_client,
        topic_control_client,
        topic_control_model,
        topic_control_prompt=None,
    ):
        self.retriever = retriever
        self.gpt_client = gpt_client
        self.reranker_client = reranker_client
        self.topic_control_client = topic_control_client
        self.topic_control_model = topic_control_model
        self.topic_control_prompt = topic_control_prompt

    # ---------------- TOPIC CONTROL ---------------- #

    def check_topic_control(self, query):
        if not self.topic_control_prompt:
            return True

        control_messages = [
            {"role": "system", "content": self.topic_control_prompt},
            {"role": "user", "content": query}
        ]

        completion = self.topic_control_client.chat.completions.create(
            model=self.topic_control_model,
            messages=control_messages,
            temperature=0.5,
            top_p=1,
            max_tokens=1024
        )

        classification = completion.choices[0].message.content.strip().lower()
        return "on-topic" in classification

    # ---------------- HELPER LLM ---------------- #

    def llm_markdown_response(self, input, prompt_prefix=None):
        if prompt_prefix is None:
            prompt_prefix = (
                "You are an AI assistant that provides clear, concise, and professional refusal responses."
                "When a question is classified as off-topic based on the topic control rules, explain the refusal in a polite and factual manner."
                "Use Markdown formatting with a heading and brief bullet points to summarize why the question is off-topic in this context."
                "Do not apologize or add unnecessary detail; keep it focused and informative."
            )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(prompt_prefix),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        formatted_prompt = prompt.format_prompt(input=input)
        messages = formatted_prompt.to_messages()
        result = self.gpt_client.invoke(messages, max_tokens=1024)
        return result.content.strip()

    # ---------------- DOCUMENT MARKDOWN ---------------- #

    def markdown_for_documents(self, docs, title_prefix="Document"):
        md_output = ""
        for i, doc in enumerate(docs, 1):
            relevance_score = doc.metadata.get("relevance_score")
            md_output += f"""### {title_prefix} {i}

**Source:** {doc.metadata.get('source', 'Unknown Source')}  
**Author:** {doc.metadata.get('author', 'Unknown')}  
**Page:** {doc.metadata.get('page', 'N/A')}
"""
            if relevance_score is not None:
                md_output += f"**Relevance Score:** {relevance_score}\n"

            md_output += f"""

**Snippet:**  
{doc.page_content[:500].strip()}...

"""
        return md_output

    # ---------------- RETRIEVE ---------------- #

    def retrieve(self, query):
        docs = self.retriever.invoke(query)
        return docs

    # ---------------- RERANK ---------------- #

    def rerank(self, query, docs):
        # 1. Call NVIDIA's reranker (compress_documents)
        reranked_docs = self.reranker_client.compress_documents(
            query=query,
            documents=[
                Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                )
                for doc in docs
            ]
        )
        return reranked_docs


    # ---------------- LLM ANSWER ---------------- #

    def generate_rag_answer(self, query, docs):
        if not docs:
            return "The provided documents do not contain this information."

        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
        system_prompt = (
            "You are a helpful, factual AI assistant. Your task is to answer the user's question "
            "STRICTLY based on the provided context.\n\n"
        
            "### Instructions\n"
            "1. Use ONLY the information present in the context.\n"
            "2. Do NOT hallucinate, guess, or add external knowledge.\n"
            "3. Format your final answer in clean Markdown:\n"
            "   - Use headings (##)\n"
            "   - Use bullet points and numbered lists where appropriate\n"
            "   - Highlight key terms using **bold**\n"
            "4. Cite all statements or facts using inline citations.\n"
            "5. If the context contains absolutely no relevant information, explicitly refuse by saying:\n"
            "   \"The provided documents do not contain this information.\"\n\n"
        
            "### Goal\n"
            "Produce a clear, accurate, well-structured Markdown answer based solely on the given context."
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(
                "Context: {context}\n\nQuestion: {query}\nAnswer with citations."
            )
        ])

        formatted_prompt = prompt.format_prompt(context=context_text, query=query)
        messages = formatted_prompt.to_messages()
        result = self.gpt_client.invoke(messages)
        return result.content.strip()

    # ---------------- MAIN QUERY ---------------- #

    def query(self, query, enable_rerank=False, enable_topic_control=False):
        full_output_md = f"## Query\n\n{query}\n\n"

        # ----------- TOPIC CONTROL ----------- #
        if enable_topic_control:
            if not self.check_topic_control(query):
                refusal_context = (
                    f"The user asked: \"{query}\"\n"
                    f"The topic control model prompt is \"{self.topic_control_prompt}\"\n"
                    "The question was classified as off-topic."
                )
                refusal_response = self.llm_markdown_response(refusal_context)
                full_output_md += f"### Topic Control Refusal Response\n\n{refusal_response}\n"
                display(Markdown(full_output_md))
                return

        # ----------- RETRIEVE ----------- #
        docs = self.retrieve(query)

        # ----------- RERANK (with threshold) ----------- #
        if enable_rerank:
            docs = self.rerank(query, docs)

        # ----------- ANSWER ----------- #
        answer = self.generate_rag_answer(query, docs)

        full_output_md += f"## Generated Answer\n\n{answer}\n\n"
        full_output_md += "---\n## Source Documents Used for Answer\n\n"
        full_output_md += self.markdown_for_documents(docs, title_prefix="Source Document")

        display(Markdown(full_output_md))
