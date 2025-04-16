import numpy as np


class CTCDecoder:
    """CTC 디코딩 구현"""
    
    @staticmethod
    def get_probabilities(logits):
        """로짓에서 확률 분포 계산"""
        
        # 소프트맥스 적용하여 확률 분포 계산
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        return probs
    
    @staticmethod
    def decode(probs, blank_idx=0):
        """CTC 디코딩 (그리디)"""
        
        # 배치가 포함된 경우 첫 번째 배치만 사용
        if len(probs.shape) == 3:
            probs = probs[0]
        
        # 각 프레임에서 가장 높은 확률을 가진 인덱스 선택
        indices = np.argmax(probs, axis=-1)
        
        # 중복 제거 및 blank 토큰 제거
        prev_idx = -1
        result = []
        
        for idx in indices:
            if idx != blank_idx and idx != prev_idx:
                result.append(idx)
            prev_idx = idx
        
        return result