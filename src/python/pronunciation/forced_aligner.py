import numpy as np


class ForcedAligner:
    """DTW 기반 Forced Alignment 구현"""
    
    def __init__(self):
        pass
    
    def align(self, ctc_probs, reference_phonemes):
        """
        CTC 확률 분포와 참조 음소 시퀀스 간 강제 정렬
        
        Args:
            ctc_probs: CTC 확률 분포 [time, phonemes]
            reference_phonemes: 참조 음소 시퀀스 (PhonemeSequence 객체)
            
        Returns:
            alignment: 정렬 결과 (각 참조 음소에 대한 프레임 범위)
        """
        
        # 참조 음소 시퀀스 가져오기
        ref_phonemes = reference_phonemes.get_phonemes()
        
        # 음소-인덱스 매핑 (실제 구현에서는 모델에 맞는 매핑 필요)
        phoneme_to_idx = self._get_phoneme_to_idx_mapping()
        
        # 참조 음소를 인덱스로 변환
        ref_indices = [phoneme_to_idx.get(p, 0) for p in ref_phonemes]
        
        # DTW 알고리즘을 사용한 정렬
        alignment = self._dtw_alignment(ctc_probs, ref_indices)
        
        return alignment
    
    def _get_phoneme_to_idx_mapping(self):
        """음소-인덱스 매핑 반환"""
        # 실제 구현에서는 모델의 출력과 일치하는 매핑 필요
        # 간소화된 예시
        phonemes = ['', 'g', 'gg', 'n', 'd', 'r', 'm', 'b', 's', 'j', 'ch', 'k', 't', 'p', 'h',
                   'a', 'ae', 'ya', 'eo', 'e', 'yeo', 'o', 'u', 'yu', 'eu', 'i']
        return {p: i for i, p in enumerate(phonemes)}
    
    def _dtw_alignment(self, ctc_probs, ref_indices):
        """
        DTW 알고리즘을 사용한 정렬
        
        Args:
            ctc_probs: CTC 확률 분포 [time, phonemes]
            ref_indices: 참조 음소 인덱스 배열
            
        Returns:
            alignment: 각 참조 음소에 대한 프레임 범위
        """
        
        T = ctc_probs.shape[0]  # 시간 프레임 수
        R = len(ref_indices)     # 참조 음소 수
        
        # 거리 행렬 계산 (각 프레임과 참조 음소 간의 거리)
        distance = np.zeros((T, R))
        for t in range(T):
            for r in range(R):
                # 거리 = 1 - 해당 음소에 대한 확률
                distance[t, r] = 1 - ctc_probs[t, ref_indices[r]]
        
        # DTW 누적 거리 행렬
        accumulated = np.zeros((T, R))
        accumulated[0, 0] = distance[0, 0]
        
        # 첫 번째 행 초기화
        for r in range(1, R):
            accumulated[0, r] = accumulated[0, r-1] + distance[0, r]
        
        # 첫 번째 열 초기화
        for t in range(1, T):
            accumulated[t, 0] = accumulated[t-1, 0] + distance[t, 0]
        
        # 나머지 셀 채우기
        for t in range(1, T):
            for r in range(1, R):
                accumulated[t, r] = distance[t, r] + min(
                    accumulated[t-1, r-1],  # 대각선
                    accumulated[t-1, r],    # 위
                    accumulated[t, r-1]     # 왼쪽
                )
        
        # 역추적하여 정렬 경로 찾기
        path = []
        t, r = T-1, R-1
        
        while t > 0 and r > 0:
            path.append((t, r))
            
            # 다음 셀 결정 (최소 거리 방향으로)
            diag = accumulated[t-1, r-1]
            up = accumulated[t-1, r]
            left = accumulated[t, r-1]
            
            min_val = min(diag, up, left)
            
            if min_val == diag:
                t, r = t-1, r-1
            elif min_val == up:
                t = t-1
            else:
                r = r-1
        
        # 시작점 추가
        path.append((0, 0))
        path.reverse()
        
        # 각 참조 음소에 대한 프레임 범위 계산
        alignment = []
        current_r = 0
        start_t = 0
        
        for t, r in path:
            if r > current_r:
                # 이전 음소 범위 저장
                if current_r > 0:
                    alignment.append((current_r, start_t, t-1))
                # 새 음소 시작
                current_r = r
                start_t = t
        
        # 마지막 음소 범위 추가
        if current_r > 0:
            alignment.append((current_r, start_t, T-1))
        
        return alignment