import requests
import os
from dotenv import load_dotenv

load_dotenv()

# LS 증권 API 정보
APP_KEY = os.getenv('APP_KEY') 
APP_SECRET = os.getenv('APP_SECRET') 

# API URL 정보
BASE_URL = "https://openapi.ls-sec.co.kr:8080"
PATH = "oauth2/token"
URL = f"{BASE_URL}/{PATH}"

# 요청 헤더 및 데이터
headers = {"Content-Type": "application/x-www-form-urlencoded"}
data = {
    "grant_type": "client_credentials",
    "appkey": APP_KEY,
    "appsecretkey": APP_SECRET,
    "scope": "oob"
}

try:
    # POST 요청
    response = requests.post(URL, headers=headers, data=data, verify=False)

    # 응답 상태 코드 확인
    if response.status_code == 200:
        # Access Token 추출
        access_token = response.json().get("access_token")
        print("Access Token 발급 성공!")
        print("Access Token:", access_token)
    else:
        # 오류 메시지 출력
        print("Access Token 발급 실패.")
        print("응답 코드:", response.status_code)
        print("응답 내용:", response.text)

except requests.exceptions.RequestException as e:
    print("Exception occurred:", e)
