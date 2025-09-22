import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_google_community import GoogleSearchAPIWrapper

# 加载环境变量
load_dotenv(find_dotenv())

# ModelScope 配置
key = os.environ["MODELSCOPE_TOKEN"]

# LLM 配置
llm = ChatOpenAI(
    base_url="https://api-inference.modelscope.cn/v1",
    api_key=key,
    model="deepseek-ai/DeepSeek-V3.1",
    temperature=0,
)

llm_think = llm.bind(
    extra_body={
        "enable_thinking": True
    }
)

# 搜索工具配置
tavily_retriever = TavilySearchAPIRetriever(k=3)

google_search = GoogleSearchAPIWrapper(
    google_api_key=os.environ["GOOGLE_SEARCH_KEY"],
    google_cse_id=os.environ["GOOGLE_CSE_ID"],
    k=3,
)

# MCP
MCP_WEB_URL = os.getenv("MCP_WEB_URL", "http://127.0.0.1:8765/mcp")