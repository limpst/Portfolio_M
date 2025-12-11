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
from openai import OpenAI  # OpenAI 클라이언트 추가
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mysql.connector import pooling
from html import unescape

# ==========================================
# 0. 환경 변수 및 기본 설정
# ==========================================

load_dotenv()

n_cpu_cores = multiprocessing.cpu_count()

# 필수 값 확인
if not os.getenv("LS_ACCESS_TOKEN") or not os.getenv("DB_PASSWORD"):
    print("❌ [Error] .env 파일이 없거나 필수 환경 변수(TOKEN, PASSWORD)가 누락되었습니다.")
    exit(1)

# ==========================================
# 1. 설정 및 상수 정의 (Configuration)
# ==========================================

WS_URL = "wss://openapi.ls-sec.co.kr:9443/websocket"
API_BASE_URL = "https://openapi.ls-sec.co.kr:8080"
ACCESS_TOKEN = os.getenv("LS_ACCESS_TOKEN")

# 로컬 Llama 서버 설정
LLM_SERVER_URL = "http://localhost:8080/v1"

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME", "LLM")
}

news_queue = queue.Queue()

# DB Pool 생성
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
# 2. LLM 클라이언트 초기화 (OpenAI 호환)
# ==========================================

print(f"⏳ [System] 로컬 LLM 서버({LLM_SERVER_URL})에 연결 설정 중...")

try:
    client = OpenAI(
        base_url=LLM_SERVER_URL,
        api_key="no-key-needed"  # 로컬 서버라 키 불필요
    )

    # 간단한 연결 테스트 (선택 사항)
    # client.models.list()

    print("✅ [System] LLM 클라이언트 준비 완료")
except Exception as e:
    print(f"⚠️ [Warning] LLM 서버 연결 설정 중 에러 (서버가 켜져 있는지 확인하세요): {e}")

# LLM 호출 직렬화를 위한 전역 락 (서버 부하 조절용)
llm_lock = threading.Lock()

# 키워드 정의





from keywords import (
    MACRO_KEYWORDS,  # MACRO_KEYWORDS = ["금리", "환율", "CPI", "PPI", "FOMC", "연준", "GDP", "물가", "유가"]
    GLOBAL_KEYWORDS, # GLOBAL_KEYWORDS = ["나스닥", "다우", "S&P500", "뉴욕증시", "테슬라", "엔비디아", "애플", "TSMC"]
    DOMESTIC_MARKET_KEYWORDS, # DOMESTIC_MARKET_KEYWORDS = ["코스피", "코스닥", "외국인", "기관", "순매수", "공매도"]
    ETC_KEYWORDS, # ETC_KEYWORDS = ["비트코인", "가상화폐", "IPO", "공모주"]
)

# ==========================================
# 3. 유틸리티 함수 (정제, 병합, DB)
# ==========================================

def clean_financial_text(text: str) -> str:
    """금융 텍스트 정제"""
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


def clean_base_text(text: str) -> str:
    """HTML 및 노이즈 제거"""
    if not text or not isinstance(text, str):
        return ""
    text = unescape(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)
    text = re.sub(r'(@media.*?\{.*?\})|(\{.*?\})', '', text, flags=re.DOTALL)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    text = re.sub(r'^[A-Za-z0-9]+OutBlock\d+\s+', '', text)
    text = re.sub(r'[^\w\s.,\'"()%\+/\-▲▼△▽↑↓]', '', text)
    return text


def merge_news_bodies(news_bodies):
    """뉴스 본문 배열 병합"""
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
    """MySQL 데이터 저장"""
    conn = None
    try:
        conn = db_pool.get_connection()
        if conn.is_connected():
            cursor = conn.cursor()
            insert_query = """
                           INSERT INTO news_data (date, time, id, realkey, title, bodysize, category, body)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                           """
            cursor.execute(insert_query, data)
            conn.commit()
    except mysql.connector.Error as err:
        print(f"❌ DB 에러: {err}")
    finally:
        if conn:
            conn.close()


# ==========================================
# 4. 키워드 기반 1차 분류
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
# 5. 요약 + 분류 통합 LLM 호출 (OpenAI API 방식)
# ==========================================

def call_llm_api(system_prompt, user_prompt, max_tokens=1024):
    """OpenAI 스타일 API 호출 헬퍼 함수"""
    try:
        # 스레드 락 사용 (서버 과부하 방지)
        with llm_lock:
            response = client.chat.completions.create(
                model="Qwen2.5-7B-Instruct-Q4_K_M.gguf",  # 로컬 서버에서는 무시됨
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ [LLM API Error] {e}")
        return ""


def summarize_and_classify(text: str, title: str) -> tuple[str, str]:
    """
    1) 긴 텍스트는 청크 단위 요약 후 최종 요약
    2) 최종 요약 + 제목을 이용해 LLM으로 카테고리 분류
    """
    category = "기타_LLM"

    if not text:
        return "내용 없음", "기타"

    # 1차: 키워드 분류
    kw_category = quick_keyword_classify(title)

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    summary_system_prompt = (
        "당신은 고도로 숙련된 전문 요약가입니다. "
        "사용자가 제공한 뉴스, 기사, 또는 기타 텍스트에서 투자자가 반드시 알아야 할 핵심 정보를 추출하세요. "
        "요약은 최대 3~5문장으로 작성하며, 다음 요소를 포함해야 합니다: "
        "1) 중요한 수치나 통계, 2) 관련 기업명 및 인물, 3) 주요 사건 및 시장 동향, 4) 경제적 영향과 전망. "
        "불필요한 배경 설명, 추측, 감정적 표현, 과도한 세부사항은 포함하지 마세요. "
        "요약은 제목이나 형식적인 표현 없이 바로 시작하세요."
    )

    def _llm_summary_single(chunk_text: str) -> str:
        result = call_llm_api(summary_system_prompt, f"뉴스 본문:\n{chunk_text}")
        # 불필요한 서두 제거
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

        if len(combined) > 1000:
            print("🏁 [System] 최종 요약본 생성 중...")
            final_summary = _llm_summary_single(combined)
        else:
            print("🏁 [System] 최종 요약본 생성 ...")
            final_summary = combined

    # 2) 카테고리 분류
    if kw_category is not None:
        category = kw_category + "_KEY"
    else:
        classify_system = """
                당신은 금융 뉴스 분류기입니다. 아래 기준에 따라 뉴스를 분류하세요.
                [분류 기준]
                1. 거시경제: 금리, 환율, 유가, CPI, 연준(Fed), 경제지표.
                2. 해외 증시: 미국/해외 증시 지수, 해외 기업.
                3. 국내 시황: 코스피/코스닥 지수, 외국인/기관 수급.
                4. 주도 섹터: 국내 개별 기업 및 산업(반도체, 2차전지, 바이오).
                5. 기타: 가상자산, 정책, IPO, 지정학적 리스크 등.

                오직 위 5개 단어 중 하나만 출력하세요. 설명은 하지 마세요.
                """
        classify_user = (
            f"뉴스 제목: {title}\n\n"
            f"뉴스 요약:\n{final_summary}\n\n"
            f"이 뉴스의 카테고리는?"
        )

        result = call_llm_api(classify_system, classify_user, max_tokens=50)

        valid_categories = ["거시경제", "해외 증시", "국내 시황", "주도 섹터", "기타"]
        for cat in valid_categories:
            if cat in result:
                category = cat + "_LLM"
                break

    return final_summary, category


# ==========================================
# 6. 뉴스 본문 조회 및 구조화
# ==========================================

def refine_financial_structure(text: str) -> str:
    """끊어진 문장 연결 및 금융 기호 구조화"""
    lines = text.splitlines()
    merged_lines = []
    bullet_pattern = re.compile(r'^[\*\-•※\[]')
    finance_symbol_pattern = re.compile(r'^[▲▼△▽↑↓]')
    starts_with_number = re.compile(r'^[▲▼△▽↑↓]\s*[0-9\.]')

    for line in lines:
        line = line.strip()
        if not line: continue
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
    result = re.sub(r'[ \t]+', ' ', result)
    return result


def get_headers(tr_cd, tr_cont="N"):
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
    data = {"t3102InBlock": {"sNewsno": news_id}}

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
            return refined_body
        else:
            print(f"뉴스 본문 요청 실패: {response.status_code}")
    except Exception as e:
        print(f"⚠️ 본문 조회 실패 ({news_id}): {e}")
    return None


# ==========================================
# 7. 워커 스레드
# ==========================================

def worker():
    """대기열에서 뉴스를 꺼내 처리하는 소비자 함수"""
    print("🚀 뉴스 처리 워커(Worker) 시작됨...")
    while True:
        news_item = news_queue.get()
        try:
            if news_item is None:
                print("🛑 워커 종료 신호 수신")
                return

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
                print("\n뉴스 본문:")
                print(raw_body + "...\n")  # 너무 기니까 앞부분만 출력

                # 2. 요약 + 분류 (OpenAI API 사용)
                summary_body, category = summarize_and_classify(raw_body, title)

                db_data = (
                    date,
                    time_str,
                    news_id,
                    realkey,
                    title,
                    len(raw_body),
                    category,
                    summary_body
               )

                # 3. DB 저장
                if "기타" not in category:    # and "주도 섹터" not in category
                    insert_to_db(db_data)
                    print(f"✅ DB 저장 완료: {title} (카테고리: {category})")
                else:
                    print(f"🚫 '{category}' 카테고리로 분류되어 저장하지 않습니다: {title}\n")
            else:
                print("⚠️ 본문 없음, 건너뜀.")

        except Exception as e:
            print(f"\n❌ 워커 처리 중 에러: {e}")
        finally:
            news_queue.task_done()


# ==========================================
# 8. WebSocket 이벤트 핸들러
# ==========================================

def on_message(ws, message):
    try:
        response = json.loads(message)
        if "body" in response:
            news_data = response["body"]
            news_queue.put(news_data)
            print(f"📩 [수신] {news_data.get('title')} -> 대기열 추가됨")
    except Exception as e:
        print(f"메시지 파싱 에러: {e}")


def on_open(ws):
    print("🌐 WebSocket 연결 및 구독 요청")
    sub_msg = {
        "header": {"token": ACCESS_TOKEN, "tr_type": "3"},
        "body": {"tr_cd": "NWS", "tr_key": "NWS001"}
    }
    ws.send(json.dumps(sub_msg))


# ==========================================
# 9. 메인 실행부
# ==========================================

if __name__ == "__main__":
    # 워커 수 설정
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
