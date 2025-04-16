import numpy as np

class FeatureExtractor:
    """오디오 특성 추출"""
    
    @staticmethod
    def extract_features(audio_data, sample_rate=16000):
        """오디오 데이터에서 특성 추출 (MFCC 등)"""
        try:
            import librosa
            
            # 데이터 길이 확인 및 패딩
            if len(audio_data) < 1024:
                padding = np.zeros(1024 - len(audio_data))
                audio_data = np.concatenate([audio_data, padding])
            
            # MFCC 특성 추출
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=13,
                hop_length=512,
                n_fft=1024
            )
            
            # 전치: [features, frames] -> [frames, features]
            mfccs = mfccs.T
            
            # 충분한 프레임이 있는지 확인
            if mfccs.shape[0] >= 9:  # 기본 width=9를 위한 최소 프레임 수
                # 기본 width 사용
                delta1 = librosa.feature.delta(mfccs.T, order=1).T
                delta2 = librosa.feature.delta(mfccs.T, order=2).T
            elif mfccs.shape[0] >= 5:  # 축소된 width=3을 위한 최소 프레임 수
                # 축소된 width 사용
                delta1 = librosa.feature.delta(mfccs.T, order=1, width=3).T
                delta2 = librosa.feature.delta(mfccs.T, order=2, width=3).T
            else:
                # 프레임이 너무 적음, MFCC만 사용
                print(f"⚠️ 프레임 수가 너무 적음: {mfccs.shape[0]}개, delta 특성 생략")
                return mfccs
            
            # 특성 결합
            features = np.concatenate([mfccs, delta1, delta2], axis=1)
            return features
            
        except Exception as e:
            print(f"❌ 특성 추출 오류: {e}")
            # 빈 특성 배열 반환
            return np.zeros((1, 13))