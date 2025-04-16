# example/python-terminal/speech_engine_bridge.py
import os
import time
import json
import numpy as np
from typing import Dict, List, Optional, Callable, Any

class SpeechEngineDelegate:
    """음성 인식 엔진으로부터 결과를 받기 위한 델리게이트 인터페이스"""
    
    def on_engine_init_success(self) -> None:
        """엔진 초기화 성공 시 호출"""
        pass
    
    def on_engine_init_failed(self) -> None:
        """엔진 초기화 실패 시 호출"""
        pass
    
    def on_record_start(self) -> None:
        """녹음 시작 시 호출"""
        pass
    
    def on_record_start_fail(self, error: str) -> None:
        """녹음 시작 실패 시 호출"""
        pass
    
    def on_tick(self, millis_until_finished: float, percent_until_finished: float) -> None:
        """녹음 진행 상태 업데이트 시 호출"""
        pass
    
    def on_record_end(self) -> None:
        """녹음 종료 시 호출"""
        pass
    
    def on_recording(self, result: str) -> None:
        """녹음 중 소리 강도 정보 업데이트 시 호출"""
        pass
    
    def on_score(self, result: str) -> None:
        """인식 결과 및 점수가 도착했을 때 호출"""
        pass


class SpeechEngineConfig:
    """음성 인식 엔진 설정 클래스"""
    
    def __init__(self) -> None:
        # 기본 설정값
        self.app_key = ""
        self.secret_key = ""
        self.user_id = "user_" + str(int(time.time()))
        
        # 엔진 설정
        self.engine_type = "cloud"  # cloud, native, multi
        self.server_address = ""
        self.sdk_cfg_addr = ""
        self.connect_timeout = 10.0
        self.server_timeout = 30.0
        
        # 오디오 설정
        self.core_type = "sent.eval"  # sent.eval, word.eval, para.eval
        self.audio_type = "wav"
        self.sample_rate = 16000.0
        self.is_need_sound_intensity = True
        self.force_record = True
        
        # 경로 설정
        self.record_file_path = os.path.join(os.getcwd(), "recordings")
        self.record_name = f"recording_{int(time.time())}.wav"
        
        # 기타 설정
        self.auto_retry = False
        self.duration = 60.0  # 최대 녹음 시간(초)


class SpeechEngine:
    """음성 인식 엔진 브릿지 클래스"""
    
    def __init__(self) -> None:
        self.delegate = None
        self.config = SpeechEngineConfig()
        self.is_initialized = False
        self.is_recording = False
        self.last_record_path = None
    
    def set_delegate(self, delegate: SpeechEngineDelegate) -> None:
        """델리게이트 설정"""
        self.delegate = delegate
    
    def init_engine(self, app_key: str, secret_key: str, user_id: Optional[str] = None) -> bool:
        """음성 인식 엔진 초기화"""
        self.config.app_key = app_key
        self.config.secret_key = secret_key
        if user_id:
            self.config.user_id = user_id
        
        # 실제 구현에서는 여기서 음성 인식 엔진을 초기화합니다.
        # 현재는 더미 구현이므로 항상 성공합니다.
        self.is_initialized = True
        
        if self.delegate:
            self.delegate.on_engine_init_success()
        
        return True
    
    def start(self, reference_text: str = None) -> bool:
        """음성 녹음 및 인식 시작"""
        if not self.is_initialized:
            if self.delegate:
                self.delegate.on_record_start_fail("엔진이 초기화되지 않았습니다.")
            return False
        
        if self.is_recording:
            if self.delegate:
                self.delegate.on_record_start_fail("이미 녹음 중입니다.")
            return False
        
        # 시작 시간 기록 및 녹음 상태 변경
        self.start_time = time.time()
        self.is_recording = True
        
        # 녹음 디렉토리 생성
        os.makedirs(self.config.record_file_path, exist_ok=True)
        self.last_record_path = os.path.join(self.config.record_file_path, self.config.record_name)
        
        # 참조 텍스트 설정 (있는 경우)
        if reference_text:
            self.reference_text = reference_text
        
        if self.delegate:
            self.delegate.on_record_start()
        
        return True
    
    def stop(self) -> bool:
        """녹음 중지 및 결과 반환"""
        if not self.is_recording:
            return False
        
        self.is_recording = False
        end_time = time.time()
        duration = end_time - self.start_time
        
        if self.delegate:
            self.delegate.on_record_end()
            
            # 더미 결과 생성 (실제 구현에서는 실제 인식 결과를 반환)
            if hasattr(self, 'reference_text') and self.reference_text:
                # 정답 텍스트가 있는 경우 해당 텍스트를 기반으로 점수 생성
                # 실제 구현에서는 음성 인식 및 발음 평가 결과를 반환
                result = self._generate_dummy_result(self.reference_text)
                self.delegate.on_score(json.dumps(result))
        
        return True
    
    def cancel(self) -> bool:
        """녹음 취소"""
        if not self.is_recording:
            return False
        
        self.is_recording = False
        
        if self.delegate:
            self.delegate.on_record_end()
        
        return True
    
    def get_last_record_path(self) -> Optional[str]:
        """마지막 녹음 파일 경로 반환"""
        return self.last_record_path
    
    def get_engine_status(self) -> bool:
        """엔진 상태 확인"""
        return self.is_initialized
    
    def _generate_dummy_result(self, reference_text: str) -> Dict[str, Any]:
        """테스트용 더미 결과 생성 함수"""
        words = reference_text.split()
        word_scores = []
        
        overall_accuracy = np.random.uniform(60, 95)
        overall_fluency = np.random.uniform(60, 95)
        overall_integrity = np.random.uniform(60, 95)
        overall_total_score = (overall_accuracy + overall_fluency + overall_integrity) / 3
        
        # 각 단어별 상세 점수 생성
        for word in words:
            accuracy = np.random.uniform(60, 95)
            fluency = np.random.uniform(60, 95)
            integrity = np.random.uniform(60, 95)
            word_score = (accuracy + fluency + integrity) / 3
            
            word_scores.append({
                "word": word,
                "accuracy": round(accuracy, 1),
                "fluency": round(fluency, 1),
                "integrity": round(integrity, 1),
                "score": round(word_score, 1)
            })
        
        return {
            "text": reference_text,  # 정답 텍스트
            "recognized_text": reference_text,  # 인식된 텍스트 (더미이므로 동일)
            "accuracy": round(overall_accuracy, 1),
            "fluency": round(overall_fluency, 1),
            "integrity": round(overall_integrity, 1),
            "total_score": round(overall_total_score, 1),
            "words": word_scores
        }


# 싱글톤 인스턴스
_shared_instance = None

def get_instance() -> SpeechEngine:
    """음성 인식 엔진 싱글톤 인스턴스 가져오기"""
    global _shared_instance
    if _shared_instance is None:
        _shared_instance = SpeechEngine()
    return _shared_instance