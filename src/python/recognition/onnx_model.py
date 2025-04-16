import onnxruntime as ort
import os
import numpy as np


class OnnxModel:
    """ONNX 런타임 모델 로드 및 추론"""
    
    def __init__(self, model_path):
        """
        ONNX 모델 초기화
        
        Args:
            model_path: ONNX 모델 파일 경로
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.is_loaded = False
    
    def load(self):
        """ONNX 모델 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        print(f"🔄 ONNX 모델 로드 중: {self.model_path}")
        
        # ONNX 런타임 세션 생성
        try:
            # CPU 옵션으로 세션 생성
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                self.model_path,
                session_options,
                providers=['CPUExecutionProvider']
            )
            
            # 입출력 이름 가져오기
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print(f"✅ ONNX 모델 로드 성공: 입력={self.input_name}, 출력={self.output_name}")
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ ONNX 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def infer(self, features):
        """모델 추론 수행"""
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        
        # 입력 데이터 변환
        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)
        
        print(f"원본 입력 형태: {features.shape}")
        
        # 모델의 입력 요구사항: 첫 번째 차원이 1이어야 함
        # 입력 형태 변환: (N, 39) -> (1, N) 형태로 변환 (압축 후 1차원으로)
        if len(features.shape) == 2:
            # 모든 특성을 평균하여 39차원 -> 1차원으로 압축
            compressed_features = np.mean(features, axis=1, keepdims=True)
            # 첫 번째 차원 추가
            features_reshaped = np.expand_dims(compressed_features, axis=0)
        elif len(features.shape) == 1:
            # (N,) -> (1, N) 형태로 변환
            features_reshaped = np.expand_dims(features, axis=0)
        else:
            # 다른 형태의 차원은 첫 번째 차원을 1로 강제 변경
            features_reshaped = np.reshape(features, (1, -1))
        
        print(f"변환된 입력 형태: {features_reshaped.shape}")
        
        # float32 타입으로 변환
        if features_reshaped.dtype != np.float32:
            features_reshaped = features_reshaped.astype(np.float32)
        
        # 추론 실행
        try:
            outputs = self.session.run([self.output_name], {self.input_name: features_reshaped})
            output = outputs[0]
            print(f"✅ 추론 완료: 출력 형태 {output.shape}")
            return output
        except Exception as e:
            print(f"❌ 추론 오류: {e}")
            print(f"입력 정보 - 형태: {features_reshaped.shape}, 타입: {features_reshaped.dtype}")
            
            # 모델 입력 정보 출력
            print("모델 입력 정보:")
            for input in self.session.get_inputs():
                print(f"  이름: {input.name}, 형태: {input.shape}, 타입: {input.type}")
            
            # 다른 차원 시도
            try:
                print("대체 차원으로 재시도...")
                alt_features = np.ones((1, 1), dtype=np.float32)  # 가장 작은 입력 시도
                outputs = self.session.run([self.output_name], {self.input_name: alt_features})
                print("✅ 대체 입력으로 성공! 이 형태를 사용하세요.")
                print(f"대체 입력 형태: {alt_features.shape}")
            except:
                pass
            
            raise
    
    def get_model_info(self):
        """모델 정보 반환"""
        if not self.is_loaded:
            return {"status": "not_loaded", "path": self.model_path}
        
        info = {
            "status": "loaded",
            "path": self.model_path,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "inputs": [],
            "outputs": []
        }
        
        # 입력 정보 추가
        for input_meta in self.session.get_inputs():
            info["inputs"].append({
                "name": input_meta.name,
                "shape": input_meta.shape,
                "type": input_meta.type
            })
        
        # 출력 정보 추가
        for output_meta in self.session.get_outputs():
            info["outputs"].append({
                "name": output_meta.name,
                "shape": output_meta.shape,
                "type": output_meta.type
            })
        
        return info