import unicodedata
import websocket
import json
import requests
import re
import os
import mysql.connector
import threading
import queue
import multiprocessing
from dotenv import load_dotenv
from langchain_community.chat_models import ChatLlamaCpp
from huggingface_hub import hf_hub_download
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mysql.connector import pooling
from html import unescape
from LLMTradEx34ScenarioScore import MarketScenario

import torch

from keywords import (
    MACRO_KEYWORDS,
    GLOBAL_KEYWORDS,
    DOMESTIC_MARKET_KEYWORDS,
    ETC_KEYWORDS,
)

# ==========================================
# 0. 환경 변수 및 기본 설정
# ==========================================

load_dotenv()

n_cpu_cores = multiprocessing.cpu_count()
# 시스템 여유분 2개 제외
optimal_threads = max(1, n_cpu_cores - 2)

# 필수 값 확인 (누락 시 프로그램 종료)
if not os.getenv("LS_ACCESS_TOKEN") or not os.getenv("DB_PASSWORD"):
    print("❌ [Error] .env 파일이 없거나 필수 환경 변수(TOKEN, PASSWORD)가 누락되었습니다.")
    exit(1)

# ==========================================
# 1. 설정 및 상수 정의 (Configuration)
# ==========================================

WS_URL = "wss://openapi.ls-sec.co.kr:9443/websocket"
API_BASE_URL = "https://openapi.ls-sec.co.kr:8080"
ACCESS_TOKEN = os.getenv("LS_ACCESS_TOKEN")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME", "LLM")
}

news_queue = queue.Queue()

try:
    db_pool = pooling.MySQLConnectionPool(
        pool_name="news_pool",
        pool_size=10,
        pool_reset_session=True,
        **DB_CONFIG
    )
    print("✅ [System] DB Connection Pool 생성 완료")
except Exception as e:
    print(f"❌ [System] DB Pool 생성 실패: {e}")
    exit(1)

# ==========================================
# 2. LLM 초기화 (요약 + 분류 겸용: Qwen2.5-7B-Instruct GGUF)
#    → CPU/메모리 부담 고려한 보수적 설정
# ==========================================

# 7B 모델 (RAM 6~8GB 이상 권장)
repo_idSum = "bartowski/Qwen2.5-7B-Instruct-GGUF"
filenameSum = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"

print(f"⏳ [System] GGUF 모델 다운로드/로드 중: {repo_idSum}...")

model_pathSum = hf_hub_download(
    repo_id=repo_idSum,
    filename=filenameSum,
    cache_dir='v:/PythonProject/hf_cache_gguf'
)

# LLM 호출 직렬화를 위한 전역 락
llm_lock = threading.Lock()

llmSum = ChatLlamaCpp(
    model_path=model_pathSum,
    n_gpu_layers=-1,  # GPU에서 처리할 레이어 수   # GPU 없으면 0, GPU 있으면 10~20으로 올려도 됨
    n_batch=1024,  # 배치 크기 줄여 메모리 피크 완화
    n_ctx=4096,  # 1536,  # 7B에 4096은 무거우므로 1536 선에서 타협
    # 7B + CPU: 너무 높지 않게 (물리 코어 50~70% 수준, 최대 6)
    temperature=0.1,
    max_tokens=512,     # 출력 최대 길이
    repeat_penalty=1.15,
    verbose=False,
    streaming=False,
    stop=["<|im_end|>", "<|endoftext|>", "<|end_of_text|>"]
)


# VRAM 사용량 확인 함수 추가
def check_vram_usage():
    if torch.cuda.is_available():
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 Allocated VRAM: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"📦 Reserved VRAM:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    else:
        print("⚠️ CUDA GPU가 감지되지 않았습니다.")

# 초기 VRAM 상태 확인
check_vram_usage()

# ==========================================
# 3. 유틸리티 함수 (정제, 병합, DB)
# ==========================================

def clean_financial_text(text: str) -> str:
    """금융 텍스트 정제 (줄바꿈 복구 등)"""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'^[A-Za-z0-9]+OutBlock\d+\s+', '', text)
    lines = text.splitlines()
    merged_lines = []

    bullet_pattern = re.compile(r'^[\*\-•※\[]')
    finance_symbol_pattern = re.compile(r'^[▲▼△▽↑↓]')
    starts_with_number = re.compile(r'^[▲▼△▽↑↓]\s*[0-9\.]')

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not merged_lines:
            merged_lines.append(line)
            continue
        prev_line = merged_lines[-1]

        if bullet_pattern.match(line):
            merged_lines.append(line)
        elif finance_symbol_pattern.match(line):
            if starts_with_number.match(line):
                merged_lines[-1] += " " + line
            else:
                merged_lines.append(line)
        elif prev_line.endswith('.') or prev_line.endswith(':'):
            merged_lines.append(line)
        else:
            merged_lines[-1] += " " + line

    result = "\n".join(merged_lines)
    result = re.sub(r'(\n)([▲▼△▽↑↓])(?!\s*[0-9])', r'\n\n\2', result)
    return result


def clean_text(text: str) -> str:
    """HTML 및 특수문자 제거 (금융 기호 유지)"""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)
    text = unescape(text)
    text = re.sub(r'(@media.*?\{.*?\})|(\{.*?\})', '', text, flags=re.DOTALL)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,\'"()%\+/\-▲▼△▽↑↓]', '', text)
    return text


def clean_base_text(text: str) -> str:
    """뉴스 본문 HTML/특수문자 제거 (보다 일반화된 버전)"""
    if not text or not isinstance(text, str):
        return ""

    # 1. HTML 엔티티 변환 (&amp; -> & 등) 및 유니코드 정규화
    text = unescape(text)
    text = unicodedata.normalize('NFKC', text)

    # 2. HTML 태그 및 CSS/Script 제거
    text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)
    text = re.sub(r'(@media.*?\{.*?\})|(\{.*?\})', '', text, flags=re.DOTALL)

    # 3. URL 및 이메일 제거
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    # 4. 금융 데이터 API 노이즈 제거 (OutBlock 등)
    text = re.sub(r'^[A-Za-z0-9]+OutBlock\d+\s+', '', text)

    # 5. 허용된 문자 외 제거 (금융 기호 ▲▼ 등은 유지)
    # 주의: 줄바꿈(\n) 보존을 위해 \s를 유지하되, 불필요한 공백은 나중에 처리
    text = re.sub(r'[^\w\s.,\'"()%\+/\-▲▼△▽↑↓]', '', text)

    return text


def merge_news_bodies(news_bodies):
    """LS API에서 내려오는 뉴스 본문 배열을 자연스러운 문장으로 병합"""
    merged_lines = []
    for news in news_bodies:
        line = news['sBody'].strip()
        if not line:
            continue
        if not merged_lines:
            merged_lines.append(line)
            continue
        if merged_lines[-1].endswith('.') or merged_lines[-1].endswith(':'):
            merged_lines.append(line)
        else:
            merged_lines[-1] += " " + line
    return "\n".join(merged_lines)


def insert_to_db(data):
    """
    MySQL 데이터 저장
    Connection Pool 사용
    """
    conn = None
    try:
        conn = db_pool.get_connection()
        if conn.is_connected():
            cursor = conn.cursor()
            insert_query = """
                           INSERT INTO news_data (date,
                                                  time,
                                                  id,
                                                  realkey,
                                                  title,
                                                  bodysize,
                                                  category,
                                                  body)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s); \
                           """
            cursor.execute(insert_query, data)
            conn.commit()
    except mysql.connector.Error as err:
        print(f"❌ DB 에러: {err}")
    finally:
        if conn:
            conn.close()


# ==========================================
# 4. 키워드 기반 1차 분류 (속도 향상 핵심)
# ==========================================

def quick_keyword_classify(title: str) -> str | None:
    t = title.strip()
    if any(k in t for k in MACRO_KEYWORDS):
        return "거시경제"
    if any(k in t for k in DOMESTIC_MARKET_KEYWORDS):
        return "국내 시황"
    if any(k in t for k in GLOBAL_KEYWORDS):
        return "해외 증시"
    if any(k in t for k in ETC_KEYWORDS):
        return "기타"
    return None


# ==========================================
# 5. 요약 + 분류 통합 LLM 호출 (안정성 튜닝)
# ==========================================

def summarize_and_classify(text: str, title: str) -> tuple[str, str]:
    """
    1) 긴 텍스트는 청크 단위 요약 후 최종 요약
    2) 최종 요약 + 제목을 이용해 LLM으로 카테고리 분류
    → 최종적으로 (summary, category) 반환
    """

    # [수정됨] 기본값 설정 (반복문에서 매칭 안 될 경우 대비)
    category = "기타_LLM"

    if not text:
        return "내용 없음", "기타"

    # 1차: 키워드 분류 (빠른 경로)
    kw_category = quick_keyword_classify(title)

    # 청크 크기를 키워 호출 횟수 감소
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    def _llm_summary_single(chunk_text: str) -> str:
        system_message = SystemMessage(
            content=(
                "당신은 고도로 숙련된 전문 요약가입니다. "
                "사용자가 제공한 뉴스, 기사, 또는 기타 텍스트에서 투자자가 반드시 알아야 할 핵심 정보를 추출하세요. "
                "요약은 최대 3~5문장으로 작성하며, 다음 요소를 포함해야 합니다: "
                "1) 중요한 수치나 통계, 2) 관련 기업명 및 인물, 3) 주요 사건 및 시장 동향, 4) 경제적 영향과 전망. "
                "요약은 독자가 빠르게 핵심을 이해할 수 있도록 간결하고 명확해야 합니다. "
                "불필요한 배경 설명, 추측, 감정적 표현, 과도한 세부사항은 포함하지 마세요. "
                "객관적이고 사실에 기반한 정보를 제공하며, 구어체나 불필요한 문장은 생략하세요. "
                "요약은 제목이나 형식적인 표현 없이 바로 시작하세요."
            )
        )
        human_message = HumanMessage(content=f"뉴스 본문:\n{chunk_text}")

        # llama-cpp는 스레드 세이프하지 않은 경우가 많으므로
        # 반드시 락으로 감싸서 한 번에 하나만 호출
        with llm_lock:
            response = llmSum.invoke([system_message, human_message])

        result = response.content.strip()
        result = re.sub(r"^(\s*요약\s*[:\-\]]?|.*?요약해\s*드리겠습니다[.]?)", "", result).strip()
        return result

    # 1) 청크 요약
    if len(chunks) == 1:
        print(f"🧩 [System] 텍스트 1개의 청크로 처리합니다.")
        final_summary = _llm_summary_single(text)
    else:
        print(f"🧩 [System] 긴 텍스트 분할 처리 ({len(chunks)}개)")
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"   ... {i + 1}/{len(chunks)} 번째 청크 요약 중")

            summary = _llm_summary_single(chunk)
            chunk_summaries.append(summary)

        combined = "\n".join(chunk_summaries)

        # combined가 너무 길 때만 최종 요약 한 번 더
        if len(combined) > 1000:
            print("🏁 [System] 최종 요약본 생성 중...")
            final_summary = _llm_summary_single(combined)
        else:
            print("🏁 [System] 최종 요약본 생성 ...")
            final_summary = combined

    # 3. 카테고리 분류
    #    - 키워드 분류에 성공했으면 그대로 사용
    #    - 실패했으면 LLM 분류 수행
    if kw_category is not None:
        category = kw_category + "_KEY"
    else:
        system_instruction = """
                당신은 금융 뉴스 분류기입니다. 아래 기준에 따라 뉴스를 분류하세요.

                [분류 기준]
                1. 거시경제: 금리, 환율, 유가, CPI, 연준(Fed), 경제지표.
                2. 해외 증시: 미국/해외 증시 지수(나스닥, 다우), 해외 기업(엔비디아, 테슬라 등).
                3. 국내 시황: 코스피/코스닥 지수, 외국인/기관 수급.
                4. 주도 섹터: 국내 개별 기업(삼성전자, 현대차, 에코프로 등) 및 산업(반도체, 2차전지, 바이오).
                5. 기타: 가상자산, 정책, IPO, 지정학적 리스크 등.

                [제약 사항]
                - 설명하지 마세요.
                - 오직 아래 5개 단어 중 하나만 출력하세요.
                  거시경제, 해외 증시, 국내 시황, 주도 섹터, 기타
                """
        user_content = (
            f"뉴스 제목: {title}\n\n"
            f"뉴스 요약:\n{final_summary}\n\n"
            f"이 뉴스의 카테고리는?"
        )

        try:
            with llm_lock:
                response = llmSum.invoke([
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=user_content)
                ])
            result = response.content.strip()
            valid_categories = ["거시경제", "해외 증시", "국내 시황", "주도 섹터", "기타"]

            for cat in valid_categories:
                if cat in result:
                    category = cat + "_LLM"
                    break
        except Exception as e:
            print(f"⚠️ 분류 LLM 호출 에러: {e}")
            category = "기타_LLM"

    return final_summary, category


# ==========================================
# 6. 뉴스 본문 조회
# ==========================================

def refine_financial_structure(text: str) -> str:
    """
    2단계: 끊어진 문장 연결 및 금융 기호 구조화
    """
    lines = text.splitlines()
    merged_lines = []

    bullet_pattern = re.compile(r'^[\*\-•※\[]')
    finance_symbol_pattern = re.compile(r'^[▲▼△▽↑↓]')
    starts_with_number = re.compile(r'^[▲▼△▽↑↓]\s*[0-9\.]')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if not merged_lines:
            merged_lines.append(line)
            continue

        prev_line = merged_lines[-1]

        # 불릿 포인트나 금융 기호는 새 줄로 시작
        if bullet_pattern.match(line):
            merged_lines.append(line)
        elif finance_symbol_pattern.match(line):
            # "▲ 100원" 처럼 숫자와 결합된 경우 앞줄과 합칠지 결정
            if starts_with_number.match(line):
                merged_lines[-1] += " " + line
            else:
                merged_lines.append(line)
        # 문장이 끝났거나 콜론으로 끝나면 새 줄
        elif prev_line.endswith('.') or prev_line.endswith(':'):
            merged_lines.append(line)
        else:
            # 그 외에는 앞줄과 이어 붙임 (중간에 끊긴 문장 복구)
            merged_lines[-1] += " " + line

    result = "\n".join(merged_lines)

    # 금융 기호 앞 줄바꿈 추가 및 공백 정리
    result = re.sub(r'(\n)([▲▼△▽↑↓])(?!\s*[0-9])', r'\n\n\2', result)
    result = re.sub(r'[ \t]+', ' ', result)

    return result

def get_headers(tr_cd, tr_cont="N"):
    """헤더 생성 헬퍼 함수"""
    return {
        "Content-Type": "application/json; charset=UTF-8",
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "tr_cd": tr_cd,
        "tr_cont": tr_cont,
        "mac_address": "00:11:22:33:44:55"
    }


def fetch_news_body(news_id):
    """REST API를 통해 뉴스 상세 본문 조회"""
    url = f"{API_BASE_URL}/stock/investinfo"

    headers = get_headers("t3102")

    data = {
        "t3102InBlock": {
            "sNewsno": news_id
        }
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)

        if response.status_code == 200:
            print("뉴스 본문 요청 성공!")
            response_json = response.json()

            if "t3102OutBlock1" not in response_json:
                print("뉴스 본문 데이터가 없습니다.\n")
                return None

            news_body = response_json["t3102OutBlock1"]

            joined_body = merge_news_bodies(news_body)
            cleaned_body = clean_base_text(joined_body)
            refined_body = refine_financial_structure(cleaned_body)

            print(refined_body + "\n")

            return refined_body

        else:
            print("뉴스 본문 요청 실패.")
            print("응답 코드:", response.status_code)
            print("응답 내용:", response.text)

    except Exception as e:
        print(f"⚠️ 본문 조회 실패 ({news_id}): {e}")

    return None


# ==========================================
# 7. 워커 스레드 (멀티 워커, 빠른 처리)
# ==========================================

def worker():
    """대기열에서 뉴스를 꺼내 처리하는 소비자 함수"""
    print("🚀 뉴스 처리 워커(Worker) 시작됨...")
    while True:
        try:
            news_item = news_queue.get()
            if news_item is None:
                break  # 종료 신호

            # debug (사용자 요청: 기존 print 유지)
            print(f"날짜: {news_item.get('date')}")
            print(f"시간: {news_item.get('time')}")
            print(f"키값: {news_item.get('realkey')}")
            print(f"제목: {news_item.get('title')}")

            raw_date = news_item.get('date')
            raw_time = news_item.get('time')
            raw_title = news_item.get('title')
            news_id = news_item.get('id')
            realkey = news_item.get('realkey')

            title = clean_financial_text(raw_title)

            date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}" if raw_date and len(raw_date) == 8 else raw_date
            time_str = f"{raw_time[:2]}:{raw_time[2:4]}:{raw_time[4:]}" if raw_time and len(raw_time) == 6 else raw_time

            print(f"\n🔄 처리 시작: {title}")

            # 1. 본문 가져오기
            raw_body = fetch_news_body(realkey)

            if raw_body:
                cleaned_body = raw_body  # clean_text(raw_body)

                # debug: 본문 출력 유지
                print("\n뉴스 본문:")
                print(cleaned_body + '\n')

                # 2. 요약 + 분류(통합 LLM 호출)
                summary_body, category = summarize_and_classify(cleaned_body, title)

                db_data = (
                    date,
                    time_str,
                    news_id,
                    realkey,
                    title,
                    len(cleaned_body),
                    category,
                    summary_body
                )

                # 3. DB 저장
                if "기타" not in category and "주도 섹터" not in category:  # True:  # and "주도 섹터" not in category
                    insert_to_db(db_data)
                    print(f"✅ DB 저장 완료: {title} (카테고리: {category})")
                else:
                    print(f"🚫 '기타', '주도 섹터' 카테고리로 분류된 뉴스는 저장하지 않습니다: {title}\n")


            else:
                print("⚠️ 본문 없음, 건너뜀.")

            news_queue.task_done()

        except Exception as e:
            print(f"\n❌ 워커 처리 중 에러: {e}")


# ==========================================
# 8. WebSocket 이벤트 핸들러
# ==========================================

def on_message(ws, message):
    """실시간 메시지 수신 시 Queue에 적재"""
    try:
        response = json.loads(message)

        if "body" in response:
            news_data = response["body"]

            # Queue에 데이터 넣기 (비동기 처리를 위해)
            news_queue.put(news_data)

            print(f"📩 [수신] {news_data.get('title')} -> 대기열 추가됨")
    except Exception as e:
        print(f"메시지 파싱 에러: {e}")


def on_open(ws):
    print("🌐 WebSocket 연결 및 구독 요청")
    sub_msg = {
        "header": {
            "token": ACCESS_TOKEN,
            "tr_type": "3"
        },
        "body": {
            "tr_cd": "NWS",
            "tr_key": "NWS001"
        }
    }
    ws.send(json.dumps(sub_msg))


# ==========================================
# 9. 메인 실행부
# ==========================================

if __name__ == "__main__":
    # CPU가 4코어라면 워커 2개 정도가 무난
    num_workers = min(3, max(1, n_cpu_cores - 2))
    print(f"🧵 워커 스레드 수: {num_workers}")

    worker_threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        worker_threads.append(t)

    websocket.enableTrace(False)
    ws_app = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message,
        on_open=on_open,
        on_close=lambda ws, status_cd, msg: print("WebSocket 연결 종료:", status_cd, msg),
        on_error=lambda ws, error: print("WebSocket 에러:", error)
    )

    try:
        ws_app.run_forever()
    except KeyboardInterrupt:
        print("프로그램 종료 중...")
        for _ in worker_threads:
            news_queue.put(None)
        for t in worker_threads:
            t.join()
