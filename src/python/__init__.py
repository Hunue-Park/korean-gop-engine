# 주요 인터페이스 노출
from .engine.korean_engine import KoreanSpeechEngine
from .engine.engine_factory import get_engine_instance

__all__ = ['KoreanSpeechEngine', 'get_engine_instance']