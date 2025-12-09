import os
import json
import re
import time
from concurrent.futures import thread

import mysql
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from mysql.connector import pooling
from scipy.optimize import minimize
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from LLMTradEx34ScenarioScore import analyze_market_scenario, MarketScenario, MarketTrend

load_dotenv()

# API URL
BASE_URL = "https://openapi.ls-sec.co.kr:8080"
PATH = "/stock/investinfo"
URL = f"{BASE_URL}{PATH}"

WS_URL = "wss://openapi.ls-sec.co.kr:9443/websocket"
API_BASE_URL = "https://openapi.ls-sec.co.kr:8080"
# Access Token (발급받은 토큰 입력)
ACCESS_TOKEN = os.getenv("LS_ACCESS_TOKEN")


# ==========================================
# [설정] 상수 및 API 설정
# ==========================================

MULTIPLIER = 250000  # KOSPI 200 승수

# MySQL 연결 설정
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_DATABASE", "LLM"),
}

# DB 연결 풀 생성
try:
    db_pool = pooling.MySQLConnectionPool(
        pool_name="db_pool",
        pool_size=10,
        pool_reset_session=True,
        **DB_CONFIG
    )
    print("✅ [System] DB Connection Pool 생성 완료")
except Exception as e:
    print(f"❌ [System] DB Pool 생성 실패: {e}")
    exit(1)


OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
# os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPEN_AI_KEY)


# 자산 정의
TARGET_ASSETS = [
    {"name": "Deep OTM Call Long"},  # 0. 상승 (볼록성)
    {"name": "OTM Call Short"},  # 1. 하락/횡보 (수익/헤지)
    {"name": "Deep OTM Put Long"},  # 2. 하락 (볼록성)
    {"name": "OTM Put Short"}  # 3. 상승/횡보 (수익/헤지)
]



"""
# [수정 필요] -> 리포팅 단계에서 실시간 델타를 받아와야 정확한 방향성(Bull/Bear) 판단 가능,  # as of 20251209,0328
real_delta = get_option_greeks(strike=..., type=...)['delta']
"""

def fetch_option_price(focode):
    """
    LS증권 OPEN API를 사용하여 옵션 가격 조회

    Args:
        focode (str): 단축코드 (예: 옵션 코드 "201P3000")

    Returns:
        dict: 옵션 가격 관련 데이터
    """
    url = f"{API_BASE_URL}/futureoption/market-data"
    headers = get_headers("t2101")
    body = {"t2101InBlock": {"focode": focode}}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(body), verify=False)
        response.raise_for_status()  # HTTP 오류 확인
        response_data = response.json()

        if response_data.get("rsp_cd") == "00000" and "t2101OutBlock" in response_data:
            print("✅ 정상적으로 조회가 완료되었습니다.")
            return response_data["t2101OutBlock"]
        else:
            print(f"⚠️ API 오류: {response_data.get('rsp_msg', 'Unknown error')}")
            return {"price": 0.0, "delt": 0.0}  # 기본값 반환

    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"⚠️ API 호출 중 오류 발생: {e}")
        return {"price": 0.0, "delt": 0.0}  # 기본값 반환


def fetch_option_prices(strikes, atm) -> (List[float], List[float]):
    """
   여러 옵션의 현재가를 조회하여 가격 리스트를 반환.
   :param strikes: 행사가 리스트
   :param atm: ATM 기준값
   :return: 옵션 가격 리스트, delta 리스트
   """

    prices = []
    deltas = []
    for i, strike in enumerate(strikes):
        # 옵션 코드 생성 (Call/Put 구분)
        if strike > atm:
            # Call 옵션 코드
            focode = f"201WC{int(strike)}"
        else:
            # Put 옵션 코드
            focode = f"301WC{int(strike)}"

        # 옵션 가격 조회
        time.sleep(1)
        option_data = fetch_option_price(focode)
        if option_data and "price" in option_data:
            # 가격 추출
            price = float(option_data.get("price", 0.0))
            prices.append(price)

            # 델타 값 추출
            delta = float(option_data.get("delt", 0.0))
            deltas.append(delta)
        else:
            # 조회 실패 시 기본값 추가
            prices.append(0.0)
            deltas.append(0.0)

    return prices, deltas



class QuantState(TypedDict):
    kospi_index: float
    market_iv: float
    manager_view: str
    risk_aversion: float
    total_capital: float
    expected_returns: List[float]
    covariance_matrix: List[List[float]]
    optimal_weights: List[float]
    final_report: str
    market_trend: str  # 시장 트랜드 추가


def risk_score_to_phrase(score: float, trend: MarketTrend) -> str:
    """
    risk_aversion_score(2.0~10.0)와 trend를 사용해
    '상승/하락/변동성/중립' 같은 확률/심리 문구를 생성.
    """

    # 안전장치: 범위 클램핑
    score = max(2.0, min(10.0, float(score)))

    # 기본 방향 문구
    if trend == MarketTrend.BULLISH:
        direction_word = "상승"
    elif trend == MarketTrend.BEARISH:
        direction_word = "하락"
    elif trend == MarketTrend.VOLATILE:
        direction_word = "급변동"
    elif trend == MarketTrend.REVERSAL:
        direction_word = "추세 반전"
    else:  # NEUTRAL
        direction_word = "뚜렷한 방향성 형성"

    # 점수 구간별 해석
    if 2.0 <= score < 3.5:
        # 강한 위험 선호 → 방향성이 뚜렷한 장
        prob = 80
        mood = "강한 위험 선호 심리로"
        suffix = f"{direction_word} 가능성이 매우 높음."
    elif 3.5 <= score < 5.0:
        prob = 70
        mood = "위험 선호 심리가 우위에 있어"
        suffix = f"{direction_word} 가능성이 높음."
    elif 5.0 <= score < 6.5:
        prob = 60
        mood = "호재와 악재가 혼재된 가운데"
        # NEUTRAL일 때는 약간 중립적으로
        if trend == MarketTrend.NEUTRAL:
            suffix = "단기적으로 제한적인 등락이 반복될 가능성이 큼."
        elif trend == MarketTrend.VOLATILE:
            suffix = "단기적인 등락이 커질 수 있음."
        else:
            suffix = f"{direction_word} 가능성이 다소 우세함."
    elif 6.5 <= score < 8.5:
        prob = 70
        mood = "위험 회피 심리가 강화되면서"
        if trend == MarketTrend.BULLISH:
            # 위험회피지만 BULLISH라면 ‘상승 제한 + 조정 위험’
            suffix = f"상승 여력이 제한되고 {direction_word}보다 조정 가능성을 염두에 둘 필요가 있음."
        elif trend == MarketTrend.VOLATILE:
            suffix = "단기적으로 급락과 반등이 교차하는 높은 변동성이 예상됨."
        else:
            suffix = f"{direction_word} 가능성이 높음."
    else:  # 8.5 ~ 10.0
        prob = 80
        mood = "극도의 위험 회피 심리로"
        if trend == MarketTrend.BULLISH:
            suffix = "상승 신뢰도는 낮고 방어적 대응이 요구됨."
        elif trend == MarketTrend.NEUTRAL:
            suffix = "뚜렷한 방향성은 없지만 급락 리스크에 특히 유의해야 함."
        elif trend == MarketTrend.VOLATILE:
            suffix = "크게 출렁이는 장세가 이어질 가능성이 큼."
        else:
            suffix = f"{direction_word} 가능성이 매우 높음."

    return f"{mood} {suffix} (약 {prob}% 수준)"


#
# model_id = "google/gemma-2-2b-it"
# print(f"⏳ [System] 모델 로드 중: {model_id} (CPU Mode)...")
#
# tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='./hf_cache')
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="cpu",
#     dtype=torch.float32,
#     low_cpu_mem_usage=True,
#     cache_dir='./hf_cache'
# )
#
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=1024,
#     temperature=0.1,
#     repetition_penalty=1.1,
#     return_full_text=False
# )
#
# llm = HuggingFacePipeline(pipeline=pipe)

# category LIKE '%KEY'

def fetch_latest_news(limit: int = 20):
    """
    news_data 테이블에서 category LIKE '%KEY' 인 최신 뉴스 N개 조회
    :param limit:
    :return:
    """
    conn = None
    rows = []
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        query = """
                SELECT date, time, title, body, category
                FROM news_data
                WHERE category LIKE '거시경제%'
                  AND category LIKE '%KEY'
                ORDER BY date DESC, time DESC
                    LIMIT %s;
                """

        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"❌ DB 조회 에러: {err}")
    finally:
        if conn and conn.is_connected():
            conn.close()
    return rows


def market_scenario_to_tuple(ms: MarketScenario):
    """
    MarketScenario 객체를 (title, summary_for_scenarios, risk_aversion_score) 튜플로 변환.
    summary에는 risk_aversion_score를 해석한 '확률풍 문구'까지 포함.
    """

    # 필요에 따라 summary에 '상승/하락 확률 xx%' 같은 문구를 LLM에게서 직접 받거나,
    # 아래처럼 trend/risk를 조합해서 처리할 수도 있음.

    base_desc = ms.summary.strip()

    # 1) trend 기반 한두 문장 추가 (선택 사항)
    trend_extra = ""
    if ms.trend == MarketTrend.BULLISH:
        trend_extra = " 전반적으로 상승 우위의 흐름이 형성된 상태입니다."
    elif ms.trend == MarketTrend.BEARISH:
        trend_extra = " 전반적으로 하락 압력이 우세한 구간입니다."
    elif ms.trend == MarketTrend.VOLATILE:
        trend_extra = " 방향성보다는 변동성 확대 국면으로 보입니다."
    elif ms.trend == MarketTrend.REVERSAL:
        trend_extra = " 기존 추세에서 방향 전환 신호가 감지되고 있습니다."
    else:   # NEUTRAL
        trend_extra = " 뚜렷한 방향성 없이 관망세가 이어지고 있습니다."

    # 2) risk_aversion_score를 자연어 문구로 변환
    risk_phrase = risk_score_to_phrase(ms.risk_aversion_score, ms.trend)

    # 최종 summary 문장 구성
    summary_for_scenarios = f"{base_desc} {trend_extra} {risk_phrase}"

    # summary_for_scenarios = (base_desc, trend_extra).strip()

    return (
        ms.title,  # "Moderate Bull" 같은 시나리오 이름
        summary_for_scenarios, # e.g. "국내 경제 지표 개선으로 ... 상승 가능성이 매우 높음. (약 80%) 수준)"
        float(ms.risk_aversion_score)
    )


def insert_market_scenario(market_scenario: MarketScenario):
    """"
    MarketScenario 데이터를 MySQL 데이터베이스에 저장하는 함수
    """
    conn = None
    try:
        conn = db_pool.get_connection()
        if conn.is_connected():
            cursor = conn.cursor()

            # SQL 쿼리
            insert_query = """
                INSERT INTO MarketScenario (
                    title, 
                    summary_for_scenarios,
                    risk_aversion_score,
                    score_desc,
                    trend,         
                    risk,          
                    driver,        
                    key_factors,            
                    strategy
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """

            sKeyFactors = "\n".join(f"- {factor}" for factor in market_scenario.key_factors)

            # MarketScenario 데이터를 튜플로 변환
            scenario_data = (
                market_scenario.title,
                market_scenario.summary,
                market_scenario.risk_aversion_score,
                market_scenario.score_desc.value,
                market_scenario.trend.value,
                market_scenario.risk.value,
                market_scenario.driver.value,
                sKeyFactors,
                market_scenario.strategy
            )

            # 쿼리 실행
            cursor.execute(insert_query, scenario_data)
            conn.commit()
            print(f"✅ MarketScenario 저장 완료: {market_scenario.title}")

    except mysql.connector.Error as err:
        print(f"❌ DB 에러: {err}")
    finally:
        if conn and conn.is_connected():
            conn.close()



# ==========================================
# 1. Node: Quant Engine
# ==========================================
def quant_engine(state: QuantState):
    view = state['manager_view']
    iv = state['market_iv']

    prompt = (
        f"Analyze the market view and estimate parameters for 4 Option Assets.\n"
        f"View: \"{view}\"\n"
        f"IV: {iv}%\n\n"
        f"Assets:\n"
        f"1. Deep OTM Call Long (Bull/Convexity)\n"
        f"2. OTM Call Short (Bear/Income/Hedge against Bull)\n"
        f"3. Deep OTM Put Long (Bear/Convexity)\n"
        f"4. OTM Put Short (Bull/Income/Hedge against Bear)\n\n"
        f"Strategy Logic:\n"
        f"- **Allow Short positions as Hedge**: Use Call Short to hedge Call Long (Spread), Use Put Short to hedge Put Long.\n"
        f"- Strong Bull: High return for Call Long & Put Short.\n"
        f"- Strong Bear: High return for Put Long & Call Short.\n"
        f"- Range/Neutral: High return for Shorts (Selling Volatility).\n"
        f"- Crisis: High return for Put Long.\n\n"
        f"Return JSON (4x4 matrix):\n"
        f"```json\n"
        f"{{\n"
        f"  \"mu\": [C_Long, C_Short, P_Long, P_Short],\n"
        f"  \"vol\": [0.2, 0.2, 0.3, 0.3],\n"
        f"  \"corr\": [[1.0, -0.7, -0.3, 0.3], ...]\n"
        f"}}\n"
        f"```"
    )

    """
    # [수정 권장] -> 실제 통계적 공분산 행렬 산출      <== LLM이 시나리오에 따라 mu, sigma 결정함. 
    # 과거 데이터를 로딩하여 .cov() 함수로 계산 
    historical_data = load_historical_data() 
    sigma = historical_data.cov().values  # 실제 공분산 행렬
    """

    try:
        response = llm.invoke(prompt).content

        match = re.search(r"```json(.*?)```", response, re.DOTALL)
        json_str = match.group(1).strip() if match else response
        data = json.loads(json_str)

        mu = data.get('mu', [0.0] * 4)
        vol = data.get('vol', [0.2] * 4)
        corr = np.array(data.get('corr', np.eye(4).tolist()))

        sigma = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                sigma[i][j] = corr[i][j] * vol[i] * vol[j]
    except:
        mu = [0.0] * 4
        sigma = np.eye(4)


    """
    mu = np.array([0.05, 0.15, 0.25, 0.1])
    vol = np.array([0.2, 0.2, 0.3, 0.3])
    corr = np.array([[ 1. , -0.7, -0.3,  0.3],
                     [-0.7,  1.,   0.3,  0.7],
                     [-0.3,  0.3,  1.,  -0.3],
                     [ 0.3,  0.7, -0.3,  1. ],
                    ])
    sigma = np.array([
     [ 0.04,  -0.028, -0.018,  0.018],
     [-0.028,  0.04,   0.018, -0.042],
     [-0.018,  0.018,  0.09,  -0.027],
     [ 0.018, -0.042, -0.027,  0.09 ],
     ])
     """
    # debug
    print(mu)
    # print(vol)
    # print(corr)
    print(sigma)

    return {"expected_returns": mu, "covariance_matrix": sigma.tolist()}


# ==========================================
# 2. Node: Optimizer (Hedge Ratio 20%~40% Constraint)
# ==========================================
def portfolio_optimizer(state: QuantState):
    mu = np.array(state['expected_returns'])
    sigma = np.array(state['covariance_matrix'])
    risk_aversion = state['risk_aversion']
    view = state['manager_view']
    n = 4

    # 1. 다이내믹 현금 비중  (AI 코드로 스코어) 나중에 chatgpt 물어보기
    base_cash = 0.10
    if "불확실" in view or "위기" in view:
        base_cash = 0.20
    elif "확신" in view:
        base_cash = 0.05

    # 2. [핵심] 헤지 그룹 식별 (기대수익률 기반)
    # Bullish Score: Call Long(0) + Put Short(3)
    # Bearish Score: Call Short(1) + Put Long(2)
    bull_score = mu[0] + mu[3]
    bear_score = mu[1] + mu[2]

    if bull_score > bear_score:
        # 상승장 뷰 -> 헤지는 하락/횡보 자산 (1, 2)
        hedge_indices = [1, 2]
    else:
        # 하락장 뷰 -> 헤지는 상승/횡보 자산 (0, 3)
        hedge_indices = [0, 3]

    def objective(w):
        w_assets = w[:-1]
        util = np.dot(w_assets, mu) - (risk_aversion * 0.5 * np.dot(w_assets.T, np.dot(sigma, w_assets)))
        return -util

    # 3. 제약 조건 설정
    constraints = [
        # (1) 전체 비중 합 = 1.0
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},

        # (2) 현금 비중 최소치
        {'type': 'ineq', 'fun': lambda x: x[-1] - base_cash},

        # (3) [NEW] 헤지 비율 20% 이상 (안전장치)
        {'type': 'ineq', 'fun': lambda x: np.sum([x[i] for i in hedge_indices]) - 0.20},

        # (4) [NEW] 헤지 비율 40% 이하 (수익성 보존)
        {'type': 'ineq', 'fun': lambda x: 0.40 - np.sum([x[i] for i in hedge_indices])},

        # (5) [추가] Deep OTM Call Long (인덱스 0) 비중 최대 20% 제한
        {'type': 'ineq', 'fun': lambda x: 0.05 - x[0]}, # x[0] <= 0.15

        # (6) [추가] Deep OTM Put Long (인덱스 2) 비중 최대 20% 제한
        {'type': 'ineq', 'fun': lambda x: 0.05 - x[2]}, # x[2] <= 0.25
    ]

    bounds = tuple((0.0, 1.0) for _ in range(n + 1))
    init_w = [0.05, 0.15, 0.05, 0.15, 0.60]  # 초기 가중치

    try:
        result = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x.tolist()

    except Exception as e:
        print(f"Optimization Failed: {e}")
        weights = [1 / (n + 1)] * (n + 1)

    return {"optimal_weights": weights}




def calculate_strikes(atm: float, risk_aversion: float, iv: float, market_trend: str) -> List[float]:
    """
    risk_aversion, 시장 변동성, 시장 트렌드에 따라 스트라이크를 다이내믹하게 계산합니다.

    Args:
        atm (float): ATM 기준값 (현재 KOSPI 200 지수)
        risk_aversion (float): 위험 회피 성향 값 (2.0 ~ 10.0)
        iv (float): 시장 변동성 (예: 15.0, 30.0)
        market_trend (str): 시장 트렌드 (Bullish, Bearish, Neutral)

    Returns:
        List[float]: 계산된 스트라이크 리스트 (OTM 옵션만 포함)
    """
    # 변동성 및 risk_aversion 기반 간격 계산
    base_interval = 5.0  # 기본 간격
    interval = base_interval * (iv / 15.0) * (1 + (10.0 - risk_aversion) / 10.0)

    # 시장 트렌드에 따른 스트라이크 조정 (ITM 옵션 제외)
    if market_trend.lower() == "bullish":
        strikes = [
            atm + interval * 4,  # Deep OTM Call
            atm + interval * 2,  # OTM Call
            atm - interval * 3,  # Deep OTM Put
            atm - interval * 2  # OTM Put
        ]
    elif market_trend.lower() == "bearish":
        strikes = [
            atm + interval * 3,  # OTM Call
            atm + interval * 2,  # OTM Call
            atm - interval * 4,  # Deep OTM Put
            atm - interval * 3   # OTM Put
        ]
    else:  # Neutral
        strikes = [
            atm + interval * 3,  # OTM Call
            atm + interval * 2,  # OTM Call
            atm - interval * 3,  # OTM Put
            atm - interval * 2  # OTM Put
        ]

    # 2.5 단위로 나누어 떨어지도록 조정
    strikes = [round(strike / 2.5) * 2.5 for strike in strikes]

    # 정수로 변환
    strikes = [int(strike) for strike in strikes]

    return strikes

# strikes = calculate_strikes(590, 2.0)

# ==========================================
# 3. Node: Reporter
# ==========================================
def execution_reporter(state: QuantState):
    kospi = state['kospi_index']
    capital = state['total_capital']
    weights = state['optimal_weights']
    view = state['manager_view']
    risk_aversion = state['risk_aversion']
    iv = state['market_iv']     # 시장 변동성
    market_trend = state['market_trend']

    if not weights: return {"final_report": "Optimization Failed"}

    w_assets = weights[:-1]
    w_cash = weights[-1]

    atm = round(kospi / 2.5) * 2.5


    # 스트라이크 계산 (risk_aversion, iv, market_trend 기반)
    strikes = calculate_strikes(atm, risk_aversion, iv, market_trend)

    # strikes = [atm + 30.0, atm + 22.5, atm - 30.0, atm - 20]

    prices, deltas = fetch_option_prices(strikes, atm)

    # 포트폴리오 델타 계산
    port_delta = 0.0
    for i, w in enumerate(w_assets):
        asset = TARGET_ASSETS[i]
        pos_type = "Long" if "Long" in asset['name'] else "Short"
        delta = deltas[i]  # delta = asset['delta']

        # Short 포지션의 경우 델타 부호를 반전
        if pos_type == "Short":
            delta = -delta

        port_delta += w * delta

    # 방향성 및 헤지 식별 (Optimizer와 동일한 로직 적용)
    # 결과의 정합성을 위해 델타 기준으로 다시 확인
    if port_delta > 0.01:
        direction = "Bullish 📈"
        hedge_indices = [1, 2]  # Call Short, Put Long
    elif port_delta < -0.01:
        direction = "Bearish 📉"
        hedge_indices = [0, 3]  # Call Long, Put Short
    else:
        direction = "Neutral ⚖️"
        hedge_indices = [0, 2]  # Long Volatility as Hedge


    # prices = [0.97, 1.78, 1.12, 2.45]           ###  시장 데이터 입수 ###  # as of 20251209,0328

    """
    # [수정 필요] -> 행사가(Strike)에 맞는 실제 옵션 현재가를 조회해야 함
    # 예: Deep OTM Call(390.0)의 현재가가 0.45라면 0.45를 넣어야 함
    prices = [
        get_option_price(strike=strikes[0], type="call"),  # Deep OTM Call
        get_option_price(strike=strikes[1], type="call"),  # OTM Call
        get_option_price(strike=strikes[2], type="put"),  # Deep OTM Put
        get_option_price(strike=strikes[3], type="put")  # OTM Put
    ]
    """
    lines = []
    hedge_total_weight = 0.0
    hedge_desc = []
    total_premium_pnl = 0.0   # 총 프리미엄 P&L 계산을 위한 변수

    for i, w in enumerate(w_assets):
        if w < 0.001: continue

        asset = TARGET_ASSETS[i]
        is_hedge = i in hedge_indices

        role = "🛡️ Hedge" if is_hedge else "🚀 Main"
        if is_hedge:
            hedge_total_weight += w
            hedge_desc.append(f"{asset['name']}")

        pos_type = "Long" if "Long" in asset['name'] else "Short"

        # Short 포지션의 경우 델타 부호 반전
        delta = deltas[i]
        if pos_type == "Short":
            delta = -delta

        price = prices[i]
        if price <= 0:
            print(f"⚠️ 옵션 가격 조회 실패 (Asset {TARGET_ASSETS[i]['name']}). 기본값 사용.")
            continue


        # 가격이 0이거나 너무 작아서 나누기 오류가 나지 않도록 방지
        if prices[i] * MULTIPLIER == 0:
            qty = 0
            asset_premium_pnl = 0
        else:
            qty = int((capital * w) / (price * MULTIPLIER)) if price > 0 else 0

            # 프리미엄 P&L 계산
            # Long 포지션은 프리미엄 지급 (음수), Short 포지션은 프리미엄 수취 (양수)
            if pos_type == "Long":
                asset_premium_pnl = - (qty * prices[i] * MULTIPLIER)
            else:   # Short
                asset_premium_pnl = (qty * prices[i] * MULTIPLIER)

        total_premium_pnl += asset_premium_pnl

        lines.append(
            f"| {asset['name']:<18} | {strikes[i]:<6.1f} | {pos_type:<6} | {delta:>5.2f} | {w * 100:>5.1f}% | {qty:>3} 계약 | {prices[i]:>6.2f} | {role}"
        )

    lines.append(
        f"| {'Cash (KRW)':<18} | {'-':<6} | {'-':<6} | {'0.00':>5} | {w_cash * 100:>5.1f}% | {int(capital * w_cash):,.0f} 원 | {'-':>6} |")

    hedge_text = " + ".join(hedge_desc) if hedge_desc else "None"

    report = (
        f"\n📊 [Balanced Portfolio (Hedge 20~40%)]\n"
        f"==========================================================\n"
        f"• View: {view}\n"
        f"----------------------------------------------------------\n"
        f"🏆 [Metrics]\n"
        f"• Port. Delta : {port_delta:.2f}\n"
        f"• Direction   : {direction}\n"
        f"• Hedge Ratio : {hedge_total_weight * 100:.1f}%  (Target: 20~40%)\n"
        f"• Hedge Assets: {hedge_text}\n"
        f"• Estimated Premium P&L: {total_premium_pnl:.0f} KRW\n"
        f"----------------------------------------------------------\n"
        f"| Asset              | Strike | Pos.   | Delta | Weight | Qty  | Price  | Role       \n"
        f"|--------------------|--------|--------|-------|--------|------|--------|------------\n"
        f"{chr(10).join(lines)}\n"
        f"=========================================================="
    )
    return {"final_report": report}


# ==========================================
# Workflow & Run
# ==========================================
workflow = StateGraph(QuantState)
workflow.add_node("Engine", quant_engine)
workflow.add_node("Optimizer", portfolio_optimizer)
workflow.add_node("Reporter", execution_reporter)
workflow.set_entry_point("Engine")
workflow.add_edge("Engine", "Optimizer")
workflow.add_edge("Optimizer", "Reporter")
workflow.add_edge("Reporter", END)
app = workflow.compile()

"""
# [수정 필요] -> 증권사 API (eBEST, Kiwoom)나 크롤링으로 가져와야 함
"kospi_index": get_realtime_kospi200(),  # 예: 375.45
"market_iv": get_realtime_vkospi(),  # 예: 18.2
"""

def get_headers(tr_cd, tr_cont="N"):
    """헤더 생성 헬퍼 함수"""
    return {
        "Content-Type": "application/json; charset=UTF-8",
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "tr_cd": tr_cd,
        "tr_cont": tr_cont,
        "mac_address": "00:11:22:33:44:55"
    }


def get_kospi200_index():
    """
    [t2101] KOSPI 200 지수 현재가 조회
    ATM 계산을 위한 기준 가격을 가져옵니다.
    """
    url = f"{API_BASE_URL}/futureoption/market-data"
    headers = get_headers("t2101")
    data = {"t2101InBlock": {"focode": "101WC000"}}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
        response.raise_for_status()

        res_json = response.json()
        return float(res_json["t2101OutBlock"]["kospijisu"])
    except Exception as e:
        print(f"⚠️ KOSPI 200 지수 조회 실패: {e}")
        return 0.0  # 기본값 반환


def run_simulation(view_text: str, risk_level: float = 3.0):
    kospi_index = get_kospi200_index()
    if kospi_index == 0:
        print("⚠️ KOSPI 200 지수를 가져오지 못해 시뮬레이션을 건너뜁니다.")
        return

    # 시장 트렌드 해석 (Bullish, Bearish, Neutral)
    if "bull" in view_text.lower():
        market_trend = "Bullish"
    elif "bear" in view_text.lower():
        market_trend = "Bearish"
    else:
        market_trend = "Neutral"

    inputs = {
        "kospi_index": kospi_index, "market_iv": 27.35, "total_capital": 5_000_000, ### 시장 데이터 입수
        "manager_view": view_text, "risk_aversion": risk_level,
        "expected_returns": [], "covariance_matrix": [], "optimal_weights": [], "final_report": "",
        "market_trend" : market_trend
    }

    try:
        result = app.invoke(inputs)
        print(result['final_report'])
    except Exception as e:
        print(f"❌ Simulation Error: {e}")

test_news = fetch_latest_news(20)   # 최근 뉴스 20 개 가져오기

market_scenario = analyze_market_scenario(test_news, llm)  # 마켓 시나리오 생성

insert_market_scenario(market_scenario) # DB 에 마켓 시나리오 분석 결과 저장

scenarios = [market_scenario_to_tuple(market_scenario)] # 시나리오 튜블 저장.

# scenarios = [
# # ("Strong Bull", "외국인 현선물 동반 대량 매수. 삼성전자 반등. 상승 확신 90%.", 2.0),
#     # ("Strong Bear", "미국 CPI 쇼크 및 나스닥 급락. 하락 확신 90%.", 3.0),
#     # ("Neutral/Hedging", "방향성 탐색 구간. 큰 움직임은 없으나 리스크 관리가 필요함.", 3.0),
#     # ("Weak Bull with Fear", "완만한 상승이 예상되나, 지정학적 리스크로 인한 급락 가능성이 있어 하방 헤지가 필수적임.", 5.0),
#     # ("High Volatility Explosion", "곧 중대 발표가 있음. 방향은 알 수 없으나 위든 아래든 5% 이상 급변동할 것으로 예상됨.", 3.0),
#     # ("Box Range", "거래량이 말라붙었고 특별한 모멘텀이 없음. 좁은 박스권 등락 반복 예상.", 3.0),
#     # ("Black Swan Defense", "금융 위기 전조 증상 발생. 수익보다는 자산 방어가 최우선 목표임.", 10.0),
#     # ("Technical Rebound", "과매도 구간 진입에 따른 기술적 반등 구간. 추세 전환은 아니며 단기 트레이딩 관점 접근.", 3.0),
#      ("Moderate Bull", "국내 경제 지표 개선으로 완만한 상승세를 예상. 상승 확률 70%.", 3.0),
#     # ("Moderate Bear", "미국 금리 인상 우려로 하락 가능성이 높음. 하락 확률 70%.", 3.0),
#     # ("Earnings Season", "기업 실적 발표 시즌. 일부 업종 호조, 전반적으로 중립적인 시장 예상.", 4.0),
#     # ("Geopolitical Tension", "지정학적 리스크 증가. 시장의 하락 가능성이 높아지고 변동성이 확대될 것으로 예상됨.", 6.0),
#     # ("Election Rally", "대선 시즌으로 인한 단기 랠리 예상. 상승 가능성 80%.", 3.0),
#     # ("Inflation Fear", "인플레이션 우려로 인해 시장의 하락 압력이 커지고 있음. 하락 가능성 80%.", 3.0),
#     # ("Global Recovery", "글로벌 경기 회복으로 인해 위험 자산 선호도가 증가. 상승 가능성 85%.", 3.0),
#     # ("Stagflation Risk", "경기 침체와 높은 인플레이션이 동시에 발생할 가능성. 하락 가능성 75%.", 7.0),
#     # ("Sector Rotation", "성장주에서 가치주로 섹터 로테이션이 발생. 시장은 중립적이나 특정 업종이 강세.", 3.0),
#     # ("Liquidity Crunch", "유동성 축소로 인해 시장의 하락 압력이 커지고 있음. 하락 가능성 85%.", 8.0),
#     # ("Market Euphoria", "투자 심리가 과열되며 시장 전반적으로 상승세. 상승 가능성 90%.", 3.0),
#     # ("Defensive Positioning", "시장이 고점에 도달한 것으로 보이며 방어적인 포지션이 필요함.", 9.0)
# ]

for i, (name, view, risk_level) in enumerate(scenarios, 1):
    print(f"\n🚀 [Scenario {i}: {name}]")
    run_simulation(view, risk_level)

