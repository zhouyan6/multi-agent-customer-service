"""
基础智能体类
所有专门智能体的基类
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from session_manager import LangChainSessionManager


class BaseAgent(ABC):
    def __init__(self, name: str, role: str, expertise: List[str], rag_category: str = "", session_manager: LangChainSessionManager = None):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.rag_category = rag_category
        self.llm = None  # 将在运行时注入
        self.session_manager = session_manager or LangChainSessionManager()

    def set_llm(self, llm):
        """设置LLM客户端"""
        self.llm = llm

    def set_session_manager(self, session_manager: LangChainSessionManager):
        """设置会话管理器"""
        self.session_manager = session_manager

    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理客户查询的抽象方法"""
        pass
#@abstractmethod:这是个装饰器,标记"这方法是抽象的 —— 父类只声明不实现,子类必须自己写"。
#pass:占位符,表示"这里啥也不做"(因为抽象方法本来就不该有实现)。

#方法名前下划线 _ = 约定"这是内部方法,只给类自己/子类用,外部别调"。
    def _get_conversation_context(
        self,
        session_id: str,
        state: Optional[Dict[str, Any]] = None,
        max_messages: int = 12,
    ) -> str:
        """优先使用图状态中持久化的对话（跨 LangGraph 进程/工作有效），否则回退到 session_manager。"""
        if state is not None:
            records = state.get("persisted_dialogue")
            if records:
                tail = records[-max_messages:]
                context_lines = []
                for msg in tail:
                    role = "用户" if msg.get("is_user", True) else "AI"
                    content = msg.get("content", "")
                    timestamp = msg.get("timestamp", "")
                    context_lines.append(f"[{timestamp}] {role}: {content}")
                return "\n".join(context_lines)
        try:
            conversation_context = self.session_manager.get_conversation_context(session_id, max_messages)

            if not conversation_context:
                return ""

            # 格式化对话历史
            context_lines = []
            for msg in conversation_context:
                role = "用户" if msg.get("is_user", True) else "AI"
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                context_lines.append(f"[{timestamp}] {role}: {content}")

            return "\n".join(context_lines)
        except Exception as e:
            print(f"获取对话上下文时出错: {e}")
            return ""

    def _add_message_to_session(self, session_id: str, message: str, is_user: bool = True):
        """添加消息到会话历史"""
        try:
            self.session_manager.add_message(session_id, message, is_user)
        except Exception as e:
            print(f"添加消息到会话时出错: {e}")

    def _enhance_system_prompt_with_context(self, base_prompt: str) -> str:
        """增强系统提示，添加对话上下文说明"""
        context_instruction = """

重要：请结合对话历史上下文，理解客户之前的问题和需求，提供连贯、个性化的回答。
如果这是多轮对话，请参考之前的对话内容，避免重复信息，并基于客户的新问题提供补充信息。
保持对话的连贯性和自然性，让客户感受到你理解他们的完整需求。"""

        return base_prompt + context_instruction

    def get_info(self) -> Dict[str, Any]:
        """获取智能体信息"""
        return {
            "name": self.name,
            "role": self.role,
            "expertise": self.expertise
        }

    def retrieve_knowledge(self, query: str) -> str:
        """
        使用 RAG 从知识库检索相关信息
        返回格式化的知识文本，无结果时返回空字符串
        """
        try:
            from rag.retriever import retrieve
            return retrieve(query=query, category=self.rag_category)
        except Exception as e:
            print(f"RAG retrieval error in {self.name}: {e}")
            return ""
