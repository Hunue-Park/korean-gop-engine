import time
import json
import os
from ..utils.config import ConfigManager

class KoreanSpeechEngine:
    """한국어 음성 인식 및 발음 평가 엔진"""
    
    def __init__(self):
        # 모듈 의존성 초기화
        self.delegate = None
        self.onnx_model = None
        self.g2p_converter = None
        self.forced_aligner = None
        self.gop_calculator = None
        self.stream_handler = None
        
        # 상태 변수
        self.is_initialized = False
        self.is_recording = False
        self.reference_text = None
        self.last_record_path = None
        
    def set_delegate(self, delegate):
        """델리게이트 설정"""
        self.delegate = delegate
        
    def init_engine(self, app_key, secret_key, user_id=None):
        print("🔄 init_engine 호출됨")
        """엔진 초기화, ONNX 모델 로드"""
        try:
            # 설정 초기화
            config = self._initialize_config(app_key, secret_key, user_id)
            
            # 모델 경로 로그 출력
            model_path = config.get("model_path")
            print(f"🔍 ONNX 모델 경로: {model_path}")
            
            # 모델 파일 존재 여부 확인
            if not os.path.exists(model_path):
                print(f"❌ 오류: 모델 파일이 존재하지 않습니다. 경로: {model_path}")
                if self.delegate:
                    self.delegate.on_engine_init_failed()
                return False
            
            print(f"✅ 모델 파일 확인됨: {model_path}")
            
            # ONNX 모델 로드
            from ..recognition.onnx_model import OnnxModel
            print("🔄 ONNX 모델 로딩 시작...")
            self.onnx_model = OnnxModel(model_path=model_path)
            
            try:
                self.onnx_model.load()
                print(f"✅ ONNX 모델 로딩 성공: {model_path}")
            except Exception as onnx_error:
                print(f"❌ ONNX 모델 로딩 실패: {onnx_error}")
                if self.delegate:
                    self.delegate.on_engine_init_failed()
                return False
            
            # G2P 변환기 초기화
            print("🔄 G2P 변환기 초기화 중...")
            from ..pronunciation.g2p_converter import G2PConverter
            self.g2p_converter = G2PConverter()
            
            # Forced Aligner 초기화
            print("🔄 Forced Aligner 초기화 중...")
            from ..pronunciation.forced_aligner import ForcedAligner
            self.forced_aligner = ForcedAligner()
            
            # GOP 계산기 초기화
            print("🔄 GOP 계산기 초기화 중...")
            from ..pronunciation.gop_calculator import GOPCalculator
            self.gop_calculator = GOPCalculator()
            
            # 스트림 핸들러 초기화
            print("🔄 스트림 핸들러 초기화 중...")
            from ..recognition.stream_handler import StreamHandler
            self.stream_handler = StreamHandler(self.onnx_model)
            
            self.is_initialized = True
            print("✅ 엔진 초기화 완료")
            
            if self.delegate:
                self.delegate.on_engine_init_success()
                
            return True
            
        except Exception as e:
            print(f"❌ 엔진 초기화 오류: {e}")
            import traceback
            traceback.print_exc()
            if self.delegate:
                self.delegate.on_engine_init_failed()
            return False
    
    def start(self, reference_text=None):
        print("🔄 start 호출됨")
        """음성 인식 시작 및 참조 텍스트 설정"""
        if not self.is_initialized:
            if self.delegate:
                self.delegate.on_record_start_fail("엔진이 초기화되지 않았습니다.")
            return False
        
        if self.is_recording:
            if self.delegate:
                self.delegate.on_record_start_fail("이미 녹음 중입니다.")
            return False
        
        # 참조 텍스트 설정 및 G2P 변환
        if reference_text:
            self.reference_text = reference_text
            # G2P 변환 수행
            self.reference_phonemes = self.g2p_converter.convert(reference_text)
            print("🔄 참조 텍스트 설정 및 G2P 변환 완료", self.reference_phonemes)
        
        # 녹음 상태 변경
        self.is_recording = True
        self.start_time = time.time()
        
        # 녹음 디렉토리 설정
        import os
        record_dir = os.path.join(os.getcwd(), "recordings")
        os.makedirs(record_dir, exist_ok=True)
        self.last_record_path = os.path.join(record_dir, f"recording_{int(time.time())}.wav")
        
        # 스트림 핸들러 초기화
        print("🔄 스트림 핸들러 초기화 중...")
        self.stream_handler.start()
        
        if self.delegate:
            self.delegate.on_record_start()
        
        return True
    
    def process_audio_chunk(self, audio_chunk):
        """오디오 청크 처리 (스트리밍) 및 실시간 결과 반환"""
        if not self.is_recording:
            return False
        
        # 오디오 청크를 스트림 핸들러에 전달
        result = self.stream_handler.process_chunk(audio_chunk)
        
        # 실시간 결과가 있으면 처리
        if isinstance(result, dict) and 'ctc_probs' in result:
            # 참조 텍스트가 있는 경우 실시간 발음 평가 수행
            if self.reference_text and hasattr(self, 'reference_phonemes'):
                try:
                    print(f"🔄 실시간 결과 ctc_probs: {result['ctc_probs']}")
                    # 부분 정렬 및 점수 계산
                    alignment = self.forced_aligner.align(result['ctc_probs'], self.reference_phonemes)
                    gop_scores = self.gop_calculator.calculate(alignment, result['ctc_probs'])
                    print(f"🔄 실시간 결과 gop_scores: {gop_scores}")
                    
                    # 실시간 부분 결과 생성
                    interim_result = self._generate_interim_result(
                        self.reference_text, 
                        self.reference_phonemes, 
                        gop_scores,
                        is_final=result.get('is_final', False)
                    )
                    
                    # 결과 전달
                    if self.delegate:
                        self.delegate.on_recording(json.dumps(interim_result))
                        print(f"✅ 실시간 평가 결과 전송: {len(gop_scores)} 점수")
                except Exception as e:
                    print(f"⚠️ 실시간 평가 오류 (무시됨): {e}")
        
        # 볼륨 정보 전달
        if self.delegate:
            from ..audio.processor import AudioProcessor
            volume_info = AudioProcessor.compute_volume_features(audio_chunk)
            self.delegate.on_recording(json.dumps(volume_info))
        
        return True
    
    def stop(self):
        """녹음 중지 및 결과 평가"""
        if not self.is_recording:
            return False
        
        self.is_recording = False
        
        # 스트림 처리 완료
        final_features = self.stream_handler.stop()
        
        # CTC 확률 분포 계산
        from ..recognition.ctc_decoder import CTCDecoder
        ctc_probs = CTCDecoder.get_probabilities(final_features)
        
        # 참조 텍스트가 있는 경우 발음 평가 수행
        if self.reference_text and self.reference_phonemes:
            # DTW를 사용한 forced alignment 수행
            alignment = self.forced_aligner.align(ctc_probs, self.reference_phonemes)
            
            # GOP 계산
            gop_scores = self.gop_calculator.calculate(alignment, ctc_probs)
            
            # 음절별 점수 결과 생성
            result = self._generate_result(self.reference_text, self.reference_phonemes, gop_scores)
            
            # 결과 전달
            if self.delegate:
                self.delegate.on_record_end()
                self.delegate.on_score(json.dumps(result))
        else:
            # 참조 텍스트 없이 일반 인식만 수행
            if self.delegate:
                self.delegate.on_record_end()
        
        return True
    
    def cancel(self):
        """녹음 취소"""
        if not self.is_recording:
            return False
        
        self.is_recording = False
        self.stream_handler.stop()
        
        if self.delegate:
            self.delegate.on_record_end()
        
        return True
    
    def get_last_record_path(self):
        """마지막 녹음 파일 경로 반환"""
        return self.last_record_path
    
    def get_engine_status(self):
        """엔진 상태 확인"""
        return self.is_initialized
    
    def _initialize_config(self, app_key, secret_key, user_id=None):
        """설정 초기화"""
        
        
        config = ConfigManager()
        config.set("app_key", app_key)
        config.set("secret_key", secret_key)
        config.set("user_id", user_id or f"user_{int(time.time())}")
        
        # 모델 경로 설정 - 상대 경로 사용
        # src/python/engine/korean_engine.py 기준으로 3단계 위로 올라간 후 models 디렉토리로 이동
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
        model_path = os.path.join(project_root, "models", "wav2vec2_quantized.onnx")
        print(f"🔍 모델 경로: {model_path}")
        config.set("model_path", model_path)
        
        return config
    
    def _generate_result(self, reference_text, reference_phonemes, gop_scores):
        """음절별 발음 점수 결과 생성"""
        syllables = list(reference_text)
        syllable_scores = []
        
        for i, syllable in enumerate(syllables):
            # 해당 음절에 대한 점수 계산
            phoneme_indices = reference_phonemes.get_indices_for_syllable(i)
            phoneme_scores = [gop_scores[j] for j in phoneme_indices]
            
            # 음절별 평균 점수 계산
            avg_score = sum(phoneme_scores) / len(phoneme_scores) if phoneme_scores else 0
            
            # 정확도, 유창성, 완성도 점수 계산 (예시 - 실제 구현에 맞게 조정 필요)
            accuracy = avg_score * 100  # GOP 점수를 0-100 범위로 변환
            fluency = min(95, 60 + avg_score * 35)
            integrity = min(95, 70 + avg_score * 25)
            
            # 총점 계산
            total_score = (accuracy + fluency + integrity) / 3
            
            syllable_scores.append({
                "syllable": syllable,
                "accuracy": round(accuracy, 1),
                "fluency": round(fluency, 1),
                "integrity": round(integrity, 1),
                "score": round(total_score, 1)
            })
        
        # 전체 점수 계산
        overall_accuracy = sum(s["accuracy"] for s in syllable_scores) / len(syllable_scores)
        overall_fluency = sum(s["fluency"] for s in syllable_scores) / len(syllable_scores)
        overall_integrity = sum(s["integrity"] for s in syllable_scores) / len(syllable_scores)
        overall_score = (overall_accuracy + overall_fluency + overall_integrity) / 3
        
        # 결과 형식 구성
        result = {
            "text": reference_text,
            "recognized_text": reference_text,  # 음성 인식 결과 (현재는 동일하게 처리)
            "accuracy": round(overall_accuracy, 1),
            "fluency": round(overall_fluency, 1),
            "integrity": round(overall_integrity, 1),
            "total_score": round(overall_score, 1),
            "syllables": syllable_scores
        }
        
        return result

    def _generate_interim_result(self, reference_text, reference_phonemes, gop_scores, is_final=False):
        """실시간 중간 결과 생성"""
        # 현재까지 처리된 음절 수 결정 (전체 또는 부분)
        syllables = list(reference_text)
        processed_count = min(len(syllables), len(gop_scores))
        
        # 처리된 음절에 대한 점수만 계산
        syllable_scores = []
        
        for i in range(processed_count):
            syllable = syllables[i]
            # 음절별 점수 계산 로직
            phoneme_indices = reference_phonemes.get_indices_for_syllable(i)
            phoneme_scores = [gop_scores[j] for j in phoneme_indices if j < len(gop_scores)]
            
            if not phoneme_scores:
                continue
            
            # 음절별 평균 점수 계산
            avg_score = sum(phoneme_scores) / len(phoneme_scores)
            
            # 점수 계산
            accuracy = avg_score * 100
            fluency = min(95, 60 + avg_score * 35)
            integrity = min(95, 70 + avg_score * 25)
            total_score = (accuracy + fluency + integrity) / 3
            
            syllable_scores.append({
                "syllable": syllable,
                "accuracy": round(accuracy, 1),
                "fluency": round(fluency, 1),
                "integrity": round(integrity, 1),
                "score": round(total_score, 1),
                "is_processed": True
            })
        
        # 아직 처리되지 않은 음절은 임시 값으로 추가
        for i in range(processed_count, len(syllables)):
            syllable_scores.append({
                "syllable": syllables[i],
                "accuracy": 0,
                "fluency": 0,
                "integrity": 0,
                "score": 0,
                "is_processed": False
            })
        
        # 처리된 음절에 대한 평균 점수 계산
        if processed_count > 0:
            processed_scores = syllable_scores[:processed_count]
            overall_accuracy = sum(s["accuracy"] for s in processed_scores) / processed_count
            overall_fluency = sum(s["fluency"] for s in processed_scores) / processed_count
            overall_integrity = sum(s["integrity"] for s in processed_scores) / processed_count
            overall_score = (overall_accuracy + overall_fluency + overall_integrity) / 3
        else:
            overall_accuracy = 0
            overall_fluency = 0 
            overall_integrity = 0
            overall_score = 0
        
        # 결과 형식 구성
        result = {
            "text": reference_text,
            "recognized_text": reference_text[:processed_count],  # 현재까지 인식된 부분
            "accuracy": round(overall_accuracy, 1),
            "fluency": round(overall_fluency, 1),
            "integrity": round(overall_integrity, 1),
            "total_score": round(overall_score, 1),
            "syllables": syllable_scores,
            "is_final": is_final,
            "progress": processed_count / len(syllables) * 100 if len(syllables) > 0 else 0
        }
        
        return result