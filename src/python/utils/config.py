# src/python/utils/config.py
class ConfigManager:
    """설정 관리 클래스"""
    
    def __init__(self):
        self.config = {}
    
    def load_config(self, file_path=None):
        # 설정 파일 로드
        if file_path:
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                return True
            except Exception as e:
                print(f"설정 파일 로드 실패: {e}")
                return False
        return False
    
    def save_config(self, file_path=None):
        # 설정 파일 저장
        if file_path:
            try:
                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                return True
            except Exception as e:
                print(f"설정 파일 저장 실패: {e}")
                return False
        return False
    
    def set(self, key, value):
        # 설정 값 설정
        self.config[key] = value
    
    def get(self, key, default=None):
        # 설정 값 가져오기
        return self.config.get(key, default)