class G2PConverter:
    """간단한 한국어 음절 분리기"""
    
    def __init__(self):
        print("✅ G2P 변환기 초기화 완료")
    
    def convert(self, text):
        """텍스트를 음절 단위로 분리"""
        print(f"텍스트 분리 시작: '{text}'")
        
        # 문자열을 음절 단위로 분리
        syllables = list(text)
        
        # 공백 제거
        syllables = [s for s in syllables if s != ' ']
        
        # 각 음절을 하나의 음소로 처리
        phonemes = syllables.copy()
        
        # 각 음절의 경계 정보 생성
        syllable_boundaries = [(i, i) for i in range(len(phonemes))]
        
        print(f"분리된 음절: {syllables}")
        
        return PhonemeSequence(phonemes, syllable_boundaries, text)


class PhonemeSequence:
    """음소(음절) 시퀀스 및 정보 관리"""
    
    def __init__(self, phonemes, syllable_boundaries, original_text):
        self.phonemes = phonemes
        self.syllable_boundaries = syllable_boundaries
        self.original_text = original_text
    
    def get_phonemes(self):
        """모든 음소 반환"""
        return self.phonemes
    
    def get_indices_for_syllable(self, syllable_idx):
        """지정된 음절에 대한 음소 인덱스 반환"""
        if 0 <= syllable_idx < len(self.syllable_boundaries):
            start, end = self.syllable_boundaries[syllable_idx]
            return list(range(start, end + 1))
        return []
    
    def __str__(self):
        """문자열 표현"""
        return f"원본: '{self.original_text}', 음절: {self.phonemes}"