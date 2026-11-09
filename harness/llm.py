import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from .tools import tool_specs

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "openai/gpt-oss-20b")


def build_llm(with_tools: bool = True):
    llm = ChatGroq(model=MODEL, temperature=0, max_retries=3, timeout=60.0)
    return llm.bind_tools(tool_specs()) if with_tools else llm