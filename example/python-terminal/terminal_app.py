# example/python-terminal/terminal_app_sounddevice.py
import os
import sys
import time
import json
import argparse
import wave
import numpy as np
import sounddevice as sd
from typing import Dict, List, Optional, Tuple, Any
from threading import Thread, Event

# 브릿지 모듈 임포트
from speech_engine_bridge import SpeechEngine, SpeechEngineDelegate, get_instance

class TerminalColors:
    """터미널 색상 코드"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class AudioCapture:
    """오디오 캡처 클래스 (sounddevice 기반)"""
    
    def __init__(self,
                 channels=1, 
                 rate=16000, 
                 chunk_size=1024,
                 device_index=None):
        """
        오디오 캡처 초기화
        
        Args:
            channels: 채널 수 (기본값: 모노)
            rate: 샘플링 레이트 (기본값: 16kHz)
            chunk_size: 한 번에 처리할 오디오 청크 크기
            device_index: 사용할 입력 장치 인덱스 (None=기본 마이크)
        """
        self.channels = channels
        self.rate = rate
        self.chunk_size = chunk_size
        self.device_index = device_index
        
        self.is_recording = False
        self.frames = []  # 녹음된 프레임 저장
        self.stop_event = Event()
        self.record_thread = None
        
    def initialize(self) -> bool:
        """오디오 캡처 초기화"""
        try:
            # 장치 목록 출력
            print("사용 가능한 오디오 장치:")
            print(sd.query_devices())
            return True
        except Exception as e:
            print(f"오디오 초기화 오류: {e}")
            return False
    
    def list_devices(self) -> List[Dict]:
        """사용 가능한 오디오 입력 장치 목록 반환"""
        try:
            devices = sd.query_devices()
            info = []
            
            for i, device in enumerate(devices):
                # 입력 채널이 있는 장치만 필터링
                if device['max_input_channels'] > 0:
                    info.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
            
            return info
        except Exception as e:
            print(f"장치 정보 가져오기 오류: {e}")
            return []
    
    def audio_callback(self, indata, frames, time, status):
        """sounddevice 콜백 함수"""
        if status:
            print(f"오디오 콜백 상태: {status}")
            
        if self.is_recording and not self.stop_event.is_set():
            # 데이터를 프레임 리스트에 추가
            self.frames.append(indata.copy())
            
            # 외부 콜백이 있으면 호출
            if self.user_callback:
                self.user_callback(indata.flatten())
    
    def start_recording(self, callback=None) -> bool:
        """
        오디오 녹음 시작
        
        Args:
            callback: 각 오디오 청크가 캡처될 때마다 호출되는 콜백 함수
        
        Returns:
            bool: 녹음 시작 성공 여부
        """
        if self.is_recording:
            print("이미 녹음 중입니다.")
            return False
        
        # 프레임 초기화
        self.frames = []
        self.user_callback = callback
        self.stop_event.clear()
        
        try:
            def record_thread_func():
                with sd.InputStream(
                    samplerate=self.rate,
                    blocksize=self.chunk_size,
                    device=self.device_index,
                    channels=self.channels,
                    dtype='int16',
                    callback=self.audio_callback
                ):
                    print(f"녹음 시작 (샘플링 레이트: {self.rate}Hz, 채널: {self.channels})")
                    self.stop_event.wait()  # 중지 이벤트가 설정될 때까지 대기
            
            # 녹음 스레드 시작
            self.record_thread = Thread(target=record_thread_func)
            self.record_thread.daemon = True
            self.record_thread.start()
            
            self.is_recording = True
            return True
            
        except Exception as e:
            print(f"녹음 시작 오류: {e}")
            return False
    
    def stop_recording(self) -> bool:
        """녹음 중지"""
        if not self.is_recording:
            print("녹음 중이 아닙니다.")
            return False
        
        self.stop_event.set()  # 중지 이벤트 설정
        
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=1.0)  # 최대 1초 대기
        
        self.is_recording = False
        print("녹음 중지됨")
        return True
    
    def save_to_wav(self, filename: str) -> bool:
        """
        녹음된 오디오를 WAV 파일로 저장
        
        Args:
            filename: 저장할 파일 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        if not self.frames or len(self.frames) == 0:
            print("저장할 오디오 데이터가 없습니다.")
            return False
            
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # 모든 프레임을 하나의 numpy 배열로 결합
            audio_data = np.concatenate(self.frames, axis=0)
            
            # int16 형식으로 변환
            audio_data = (audio_data * 32767).astype(np.int16)
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.rate)
                wf.writeframes(audio_data.tobytes())
                
            print(f"오디오 저장됨: {filename}")
            return True
            
        except Exception as e:
            print(f"오디오 저장 오류: {e}")
            return False
    
    def get_audio_data(self) -> np.ndarray:
        """
        녹음된 오디오 데이터를 numpy 배열로 반환
        
        Returns:
            np.ndarray: 오디오 데이터
        """
        if not self.frames or len(self.frames) == 0:
            return np.array([], dtype=np.int16)
            
        # 모든 프레임을 하나의 배열로 결합
        audio_data = np.concatenate(self.frames, axis=0)
        audio_data = audio_data.flatten()
        
        # int16 형식으로 변환
        return (audio_data * 32767).astype(np.int16)


class AudioLevelMeter:
    """오디오 레벨 미터 클래스"""
    
    @staticmethod
    def get_meter(audio_data: np.ndarray, width: int = 50) -> str:
        """
        오디오 레벨을 시각적으로 표시하는 미터 생성
        
        Args:
            audio_data: 오디오 데이터
            width: 미터 너비
            
        Returns:
            str: 시각적 미터
        """
        if len(audio_data) == 0:
            return "| " + " " * width + " |"
            
        # 평균 볼륨 레벨 계산
        volume = np.abs(audio_data).mean()
        
        # 최대 16비트 값 대비 상대적 볼륨 (0-1)
        relative_volume = min(1.0, volume / 32767)
        
        # 볼륨에 따른 바 길이
        bar_length = int(relative_volume * width)
        
        # 볼륨 레벨에 따른 색상 결정
        if relative_volume < 0.2:
            color = TerminalColors.GREEN
        elif relative_volume < 0.6:
            color = TerminalColors.YELLOW
        else:
            color = TerminalColors.RED
            
        # 미터 생성
        bar = color + "=" * bar_length + TerminalColors.END
        padding = " " * (width - bar_length)
        
        return f"|{bar}{padding}| {int(relative_volume * 100):3d}%"


class TerminalApp(SpeechEngineDelegate):
    """터미널 애플리케이션 클래스"""
    
    def __init__(self, app_key: str = "", secret_key: str = "") -> None:
        self.app_key = app_key
        self.secret_key = secret_key
        self.audio_capture = AudioCapture(rate=16000, chunk_size=2048)
        self.engine = get_instance()
        self.engine.set_delegate(self)
        
        # 녹음 상태 관련 변수
        self.is_recording = False
        self.reference_text = ""
        self.last_volume_update = 0
        self.volume_update_interval = 0.1  # 볼륨 미터 업데이트 간격(초)
        
        # 결과 저장 변수
        self.latest_result = None
    
    def initialize(self) -> bool:
        """앱 초기화"""
        # 오디오 초기화
        if not self.audio_capture.initialize():
            print(f"{TerminalColors.RED}오디오 초기화 실패{TerminalColors.END}")
            return False
        
        # 엔진 초기화
        if not self.engine.init_engine(self.app_key, self.secret_key):
            print(f"{TerminalColors.RED}음성 인식 엔진 초기화 실패{TerminalColors.END}")
            return False
        
        print(f"{TerminalColors.GREEN}초기화 완료{TerminalColors.END}")
        return True
    
    def start_evaluation(self, reference_text: str) -> None:
        """음성 평가 시작"""
        self.reference_text = reference_text
        
        if self.is_recording:
            print(f"{TerminalColors.YELLOW}이미 녹음 중입니다.{TerminalColors.END}")
            return
        
        print(f"\n{TerminalColors.BOLD}정답 텍스트:{TerminalColors.END} {reference_text}")
        print(f"\n{TerminalColors.BOLD}준비되면 아무 키나 누르세요. (Ctrl+C를 누르면 취소){TerminalColors.END}")
        input()
        
        # 엔진 시작
        if not self.engine.start(reference_text):
            print(f"{TerminalColors.RED}음성 인식 시작 실패{TerminalColors.END}")
            return
        
        # 오디오 캡처 시작
        if not self.audio_capture.start_recording(callback=self.audio_callback):
            print(f"{TerminalColors.RED}오디오 캡처 시작 실패{TerminalColors.END}")
            self.engine.cancel()
            return
        
        self.is_recording = True
        print(f"\n{TerminalColors.BOLD}녹음 중... (종료하려면 아무 키나 누르세요){TerminalColors.END}")
        
        try:
            # 사용자 입력 대기
            input()
            self.stop_evaluation()
        except KeyboardInterrupt:
            print("\n녹음을 취소합니다.")
            self.audio_capture.stop_recording()
            self.engine.cancel()
            self.is_recording = False
    
    def stop_evaluation(self) -> None:
        """음성 평가 중지"""
        if not self.is_recording:
            return
        
        print("\n녹음을 종료합니다...")
        # 오디오 캡처 중지
        self.audio_capture.stop_recording()
        
        # 엔진 중지
        self.engine.stop()
        
        self.is_recording = False
    
    def audio_callback(self, audio_data: np.ndarray) -> None:
        """오디오 데이터 콜백"""
        current_time = time.time()
        
        # 일정 간격으로 볼륨 미터 업데이트
        if current_time - self.last_volume_update >= self.volume_update_interval:
            meter = AudioLevelMeter.get_meter(audio_data)
            print(f"\r볼륨: {meter}", end="")
            self.last_volume_update = current_time
    
    def show_results(self) -> None:
        """평가 결과 표시"""
        if not self.latest_result:
            print(f"\n{TerminalColors.YELLOW}아직 결과가 없습니다.{TerminalColors.END}")
            return
        
        print(f"\n{TerminalColors.BOLD}===== 평가 결과 ====={TerminalColors.END}")
        print(f"{TerminalColors.BOLD}정답 텍스트:{TerminalColors.END} {self.latest_result.get('text', '')}")
        print(f"{TerminalColors.BOLD}인식된 텍스트:{TerminalColors.END} {self.latest_result.get('recognized_text', '')}")
        print(f"{TerminalColors.BOLD}정확도:{TerminalColors.END} {self.latest_result.get('accuracy', 0)}")
        print(f"{TerminalColors.BOLD}유창성:{TerminalColors.END} {self.latest_result.get('fluency', 0)}")
        print(f"{TerminalColors.BOLD}완성도:{TerminalColors.END} {self.latest_result.get('integrity', 0)}")
        print(f"{TerminalColors.BOLD}총점:{TerminalColors.END} {self.latest_result.get('total_score', 0)}")
        
        # 단어별 점수 출력
        word_scores = self.latest_result.get('words', [])
        if word_scores:
            print(f"\n{TerminalColors.BOLD}단어별 점수:{TerminalColors.END}")
            for word_score in word_scores:
                word = word_score.get('word', '')
                score = word_score.get('score', 0)
                
                # 점수에 따른 색상 선택
                if score >= 80:
                    color = TerminalColors.GREEN
                elif score >= 60:
                    color = TerminalColors.YELLOW
                else:
                    color = TerminalColors.RED
                
                print(f"- {word}: {color}{score}{TerminalColors.END}")
    
    def run(self) -> None:
        """애플리케이션 실행"""
        print(f"\n{TerminalColors.HEADER}{TerminalColors.BOLD}===== 한국어 음성 인식 및 평가 =====\n{TerminalColors.END}")
        
        if not self.initialize():
            print("초기화에 실패했습니다. 프로그램을 종료합니다.")
            return
        
        while True:
            print(f"\n{TerminalColors.BOLD}명령을 선택하세요:{TerminalColors.END}")
            print("1. 음성 평가 시작")
            print("2. 마지막 결과 보기")
            print("3. 장치 목록 보기")
            print("0. 종료")
            
            choice = input("\n선택: ").strip()
            
            if choice == '1':
                reference_text = input("\n정답 텍스트를 입력하세요: ").strip()
                if reference_text:
                    self.start_evaluation(reference_text)
                else:
                    print(f"{TerminalColors.YELLOW}텍스트를 입력해야 합니다.{TerminalColors.END}")
            
            elif choice == '2':
                self.show_results()
            
            elif choice == '3':
                devices = self.audio_capture.list_devices()
                print("\n사용 가능한 입력 장치:")
                for i, device in enumerate(devices):
                    print(f"{i + 1}. {device['name']} (인덱스: {device['index']})")
                
                device_choice = input("\n사용할 장치를 선택하세요 (숫자 입력, 기본값은 현재 장치 유지): ").strip()
                try:
                    device_idx = int(device_choice) - 1
                    if 0 <= device_idx < len(devices):
                        self.audio_capture.device_index = devices[device_idx]['index']
                        print(f"선택된 장치: {devices[device_idx]['name']}")
                    else:
                        print(f"{TerminalColors.YELLOW}유효하지 않은 장치 번호입니다.{TerminalColors.END}")
                except ValueError:
                    if device_choice:
                        print(f"{TerminalColors.YELLOW}유효하지 않은 입력입니다.{TerminalColors.END}")
            
            elif choice == '0':
                print("프로그램을 종료합니다.")
                break
            
            else:
                print(f"{TerminalColors.YELLOW}유효하지 않은 선택입니다.{TerminalColors.END}")
    
    # SpeechEngineDelegate 구현 메서드
    def on_engine_init_success(self) -> None:
        print(f"{TerminalColors.GREEN}엔진 초기화 성공{TerminalColors.END}")
    
    def on_engine_init_failed(self) -> None:
        print(f"{TerminalColors.RED}엔진 초기화 실패{TerminalColors.END}")
    
    def on_record_start(self) -> None:
        print(f"\n{TerminalColors.GREEN}녹음 시작됨{TerminalColors.END}")
    
    def on_record_start_fail(self, error: str) -> None:
        print(f"\n{TerminalColors.RED}녹음 시작 실패: {error}{TerminalColors.END}")
    
    def on_tick(self, millis_until_finished: float, percent_until_finished: float) -> None:
        # 필요시 구현
        pass
    
    def on_record_end(self) -> None:
        print(f"\n{TerminalColors.GREEN}녹음 종료됨{TerminalColors.END}")
    
    def on_recording(self, result: str) -> None:
        # 필요시 구현
        pass
    
    def on_score(self, result: str) -> None:
        try:
            self.latest_result = json.loads(result)
            print(f"\n{TerminalColors.GREEN}결과 수신 완료{TerminalColors.END}")
            self.show_results()
        except Exception as e:
            print(f"\n{TerminalColors.RED}결과 처리 오류: {e}{TerminalColors.END}")


def main() -> None:
    """메인 함수"""
    parser = argparse.ArgumentParser(description="한국어 음성 인식 및 평가 터미널 애플리케이션")
    parser.add_argument('--app-key', type=str, default="demo_app_key", help="음성 인식 엔진 앱 키")
    parser.add_argument('--secret-key', type=str, default="demo_secret_key", help="음성 인식 엔진 시크릿 키")
    args = parser.parse_args()
    
    app = TerminalApp(app_key=args.app_key, secret_key=args.secret_key)
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n{TerminalColors.RED}오류 발생: {e}{TerminalColors.END}")
    finally:
        # 정리 작업이 필요한 경우
        pass


if __name__ == "__main__":
    main()