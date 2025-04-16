import numpy as np
import time

class StreamHandler:
    """오디오 스트림 처리 및 특성 추출"""
    
    def __init__(self, model, sample_rate=16000, chunk_size=1024):
        """
        스트림 핸들러 초기화
        
        Args:
            model: ONNX 모델 인스턴스
            sample_rate: 샘플링 레이트 (기본값: 16kHz)
            chunk_size: 처리할 청크 크기
        """
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # 상태 및 버퍼 변수
        self.buffer = []
        self.features = []
        self.is_active = False
        self.total_samples = 0
        self.start_time = None
        self.last_inference_time = 0
        self.inference_interval = 0.5  # 추론 간격(초)
    
    def start(self):
        """스트림 처리 시작"""
        print("🔄 스트림 처리 시작")
        self.buffer = []
        self.features = []
        self.is_active = True
        self.total_samples = 0
        self.start_time = time.time()
        self.last_inference_time = 0
        return True
    
    def process_chunk(self, audio_chunk):
        """오디오 청크 처리 및 실시간 추론"""
        if not self.is_active:
            return None
        
        try:
            # 입력 데이터 변환
            if not isinstance(audio_chunk, np.ndarray):
                audio_chunk = np.array(audio_chunk, dtype=np.float32)
            
            # int16 -> float32 변환
            if audio_chunk.dtype != np.float32:
                if audio_chunk.dtype == np.int16:
                    audio_chunk = audio_chunk.astype(np.float32) / 32768.0
                else:
                    audio_chunk = audio_chunk.astype(np.float32)
            
            # 볼륨 정보 (디버깅용)
            volume = np.abs(audio_chunk).mean()
            print(f"🎤 오디오 청크 수신: {len(audio_chunk)} 샘플, 평균 볼륨: {volume:.4f}")
            
            # 오디오 버퍼에 청크 추가
            self.buffer.append(audio_chunk)
            self.total_samples += len(audio_chunk)
            
            # 특성 추출
            from ..audio.feature_extractor import FeatureExtractor
            features = FeatureExtractor.extract_features(audio_chunk)
            
            # 유효한 특성인지 확인
            if features is not None and features.shape[0] > 0:
                self.features.append(features)
                
                # 실시간 추론 실행 (추가된 부분)
                current_time = time.time()
                if current_time - self.last_inference_time >= self.inference_interval:
                    print("⏱️ 실시간 추론 간격 도달, 추론 실행...")
                    self.last_inference_time = current_time
                    result = self._perform_realtime_inference()
                    return result
                
                return features
            else:
                print("⚠️ 유효한 특성이 추출되지 않았습니다.")
                return None
            
        except Exception as e:
            print(f"❌ 청크 처리 오류 (무시됨): {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _perform_realtime_inference(self):
        """실시간 모델 추론 수행"""
        print(f"🔄 실시간 추론 시작 (버퍼: {len(self.buffer)} 청크, 특성: {len(self.features)}개)")
        
        # 최근 특성 데이터 사용
        if len(self.features) > 0:
            try:
                # 특성 대신 원시 오디오 데이터 사용
                recent_audio = np.concatenate(self.buffer[-3:], axis=0)  # 최근 3개 청크만 사용
                print(f"오디오 데이터 크기: {recent_audio.shape}")
                
                # 모델 추론 실행
                if self.model and hasattr(self.model, 'infer'):
                    # 오디오 데이터 직접 전달 (특성 대신)
                    result = self.model.infer(recent_audio)
                    print(f"✅ 모델 추론 완료")
                    return {"success": True}  # 임시 결과
            
            except Exception as e:
                print(f"⚠️ 실시간 추론 오류: {e}")
        
        return None
    
    def stop(self):
        """
        스트림 처리 종료
        
        Returns:
            all_features: 전체 특성 배열
        """
        if not self.is_active:
            print("⚠️ 이미 중지된 스트림입니다.")
            return np.array([])
        
        self.is_active = False
        duration = time.time() - self.start_time
        print(f"✅ 스트림 처리 완료: 총 {self.total_samples} 샘플, {duration:.2f}초")
        
        # 최종 버퍼 처리
        if not self.buffer:
            print("⚠️ 처리할 오디오 데이터가 없습니다.")
            return np.array([])
        
        # 모든 오디오 데이터 결합
        try:
            all_audio = np.concatenate(self.buffer, axis=0)
            print(f"🔍 결합된 오디오 크기: {all_audio.shape}")
            
            # 특성이 없는 경우 전체 오디오에서 추출
            if not self.features:
                from ..audio.feature_extractor import FeatureExtractor
                all_features = FeatureExtractor.extract_features(all_audio)
            else:
                # 모든 특성 결합
                all_features = np.concatenate(self.features, axis=0)
            
            print(f"🔍 최종 특성 크기: {all_features.shape}")
            
            # 최종 모델 추론 실행
            if self.model and hasattr(self.model, 'infer'):
                try:
                    result = self.model.infer(all_features)
                    print(f"✅ 최종 모델 추론 완료: 결과 형태 {result.shape if hasattr(result, 'shape') else '알 수 없음'}")
                    
                    # CTC 확률 분포 계산
                    from ..recognition.ctc_decoder import CTCDecoder
                    ctc_probs = CTCDecoder.get_probabilities(all_features)
                    
                    # 결과 딕셔너리 반환
                    return {
                        "features": all_features,
                        "ctc_probs": ctc_probs,
                        "is_final": True,
                        "timestamp": duration
                    }
                    
                except Exception as e:
                    print(f"⚠️ 최종 추론 오류: {e}")
            
            return all_features
            
        except Exception as e:
            print(f"❌ 최종 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])
    
    def get_buffer_duration(self):
        """현재 버퍼의 오디오 길이(초)"""
        if not self.total_samples:
            return 0.0
        return self.total_samples / self.sample_rate
    
    def reset(self):
        """모든 상태 초기화"""
        self.buffer = []
        self.features = []
        self.is_active = False
        self.total_samples = 0
        self.start_time = None
        self.last_inference_time = 0
        self.inference_interval = 0.5  # 추론 간격(초)
        print("🔄 스트림 핸들러 초기화됨")