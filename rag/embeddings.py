"""
SiliconFlow 嵌入模型封装
使用 OpenAI 兼容接口调用 SiliconFlow 的 BAAI/bge-m3 嵌入模型
"""

from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY, OPENAI_BASE_URL

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024


def get_embeddings() -> OpenAIEmbeddings:
    """获取 OpenAI 兼容的嵌入模型实例（SiliconFlow BGE-M3）"""
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=EMBEDDING_MODEL,
        check_embedding_ctx_length=False,
    )
