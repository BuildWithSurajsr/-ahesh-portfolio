from typing import List, Dict
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.src.config import Config


class ChatManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = None
        self._initialize_components()

    def _initialize_components(self):
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=Config.MODEL_NAME,
                google_api_key=self.api_key,
                temperature=Config.LLM_TEMPERATURE,
                max_output_tokens=Config.MAX_TOKENS,
                top_p=0.95,
                top_k=40
            )
        except Exception as e:
            print(f"Error initializing LLM, retrying: {str(e)}")
            time.sleep(1)
            self.llm = ChatGoogleGenerativeAI(
                model=Config.MODEL_NAME,
                google_api_key=self.api_key
            )

    def generate_response(self, query: str, context_docs: List[Dict], chat_history: List[Dict] = None):
        context_text = "\n".join([doc["content"] for doc in context_docs])

        system_prompt = f"""You are Mahesh Rajendra's AI assistant on his portfolio website.
Answer questions about Mahesh based only on the provided context.
If you cannot find the answer in the context, say so politely.
Be conversational, helpful, and concise. Keep responses under 3-4 sentences unless more detail is asked for.

Context about Mahesh:
{context_text}"""

        messages = [SystemMessage(content=system_prompt)]

        if chat_history:
            for msg in chat_history[-6:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=query))

        try:
            response = self.llm.invoke(messages)
            return {"answer": response.content}
        except Exception as e:
            print(f"LLM error, retrying: {str(e)}")
            try:
                time.sleep(2)
                response = self.llm.invoke(messages)
                return {"answer": response.content}
            except Exception:
                return {"answer": "I'm currently unavailable. Please try again in a moment."}
