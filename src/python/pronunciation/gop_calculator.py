import numpy as np
import math

class GOPCalculator:
    """Goodness of Pronunciation 계산"""
    
    def __init__(self):
        pass
    
    def calculate(self, alignment, ctc_probs):
        """
        정렬 결과를 바탕으로 GOP 점수 계산
        
        Args:
            alignment: DTW 정렬 결과
            ctc_probs: CTC 확률 분포
            
        Returns:
            gop_scores: 각 음소별 GOP 점수
        """
        
        # 각 음소별 GOP 점수 계산
        gop_scores = {}
        
        for phoneme_idx, start_frame, end_frame in alignment:
            # 해당 프레임 범위에서 타겟 음소의 로그 확률 평균
            target_probs = ctc_probs[start_frame:end_frame+1, phoneme_idx]
            log_target_probs = np.log(target_probs + 1e-10)  # 수치 안정성을 위한 엡실론 추가
            avg_log_target_prob = np.mean(log_target_probs)
            
            # 모든 음소에 대한 최대 로그 확률 평균
            max_log_probs = []
            for frame in range(start_frame, end_frame+1):
                # 각 프레임에서 가장 확률이 높은 음소의 로그 확률
                max_prob = np.max(ctc_probs[frame])
                max_log_probs.append(np.log(max_prob + 1e-10))
            
            avg_max_log_prob = np.mean(max_log_probs) if max_log_probs else 0
            
            # GOP 계산: 타겟 음소 로그 확률 - 최대 로그 확률
            # 값이 0에 가까울수록 발음이 정확함
            gop = avg_log_target_prob - avg_max_log_prob
            
            # 점수 정규화 (0-1 범위로 변환, 1이 최고 점수)
            normalized_gop = 1 / (1 + math.exp(-gop))
            
            gop_scores[phoneme_idx] = normalized_gop
        
        return gop_scores