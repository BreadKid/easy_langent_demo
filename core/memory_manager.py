import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.chat_message_histories import RedisChatMessageHistory, PostgresChatMessageHistory

# 确保加载 .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

class MemoryManager:
    """记忆管理器：负责各类聊天记录的持久化"""

    @staticmethod
    def get_redis_history(session_id: str) -> RedisChatMessageHistory:
        """获取基于 Redis 的聊天记录存储实例"""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        ttl = int(os.getenv("REDIS_TTL", 3600))
        
        return RedisChatMessageHistory(
            session_id=session_id,
            url=redis_url,
            ttl=ttl
        )

    @staticmethod
    def get_postgres_history(session_id: str) -> PostgresChatMessageHistory:
        """
        获取基于 PostgreSQL 的聊天记录存储实例
        
        Args:
            session_id: 会话唯一标识符
            
        Returns:
            PostgresChatMessageHistory 实例
        """
        connection_string = os.getenv("POSTGRES_URL")
        table_name = os.getenv("POSTGRES_TABLE_NAME", "chat_history")
        
        if not connection_string:
            raise ValueError("未配置 POSTGRES_URL 环境参数")

        # 初始化时会自动创建表（如果不存在）
        return PostgresChatMessageHistory(
            connection_string=connection_string,
            table_name=table_name,
            session_id=session_id
        )

# 方便直接导入使用
get_redis_history = MemoryManager.get_redis_history
get_postgres_history = MemoryManager.get_postgres_history
