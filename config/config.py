"""
Configuration file for API keys and settings
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Naver API
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def validate_config():
    """필수 환경변수가 설정되어 있는지 확인"""
    missing_keys = []

    if not NAVER_CLIENT_ID:
        missing_keys.append("NAVER_CLIENT_ID")
    if not NAVER_CLIENT_SECRET:
        missing_keys.append("NAVER_CLIENT_SECRET")
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    if not GOOGLE_API_KEY:
        missing_keys.append("GOOGLE_API_KEY")


    if missing_keys:
        raise ValueError(f"다음 환경변수가 설정되지 않았습니다: {', '.join(missing_keys)}")
    
    return True

validate_config()