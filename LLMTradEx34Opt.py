import os
import json
import re
import numpy as np
import torch
from langchain_huggingface import HuggingFacePipeline
from scipy.optimize import minimize
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==========================================
# [설정] 상수 및 API 설정
# ==========================================

MULTIPLIER = 250000  # KOSPI 200 승수

# 자산 정의
TARGET_ASSETS = [
    {"name": "Deep OTM Call Long", "delta": 0.15},  # 0. 상승 (볼록성)
    {"name": "OTM Call Short", "delta": -0.25},  # 1. 하락/횡보 (수익/헤지)
    {"name": "Deep OTM Put Long", "delta": -0.15},  # 2. 하락 (볼록성)
    {"name": "OTM Put Short", "delta": 0.25}  # 3. 상승/횡보 (수익/헤지)
]
"""
# [수정 필요] -> 리포팅 단계에서 실시간 델타를 받아와야 정확한 방향성(Bull/Bear) 판단 가능
real_delta = get_option_greeks(strike=..., type=...)['delta']
"""


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


OPEN_AI_KEY = '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'
# os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY
#
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPEN_AI_KEY)

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
    # [수정 권장] -> 실제 통계적 공분산 행렬 산출
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
    print(vol)
    print(corr)
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

        # (5) [추가] Deep OTM Call Long (인덱스 0) 비중 최대 5% 제한
        {'type': 'ineq', 'fun': lambda x: 0.15 - x[0]}, # x[0] <= 0.15

        # (6) [추가] Deep OTM Put Long (인덱스 2) 비중 최대 5% 제한
        {'type': 'ineq', 'fun': lambda x: 0.15 - x[2]}, # x[2] <= 0.15
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


# ==========================================
# 3. Node: Reporter
# ==========================================
def execution_reporter(state: QuantState):
    kospi = state['kospi_index']
    capital = state['total_capital']
    weights = state['optimal_weights']
    view = state['manager_view']

    if not weights: return {"final_report": "Optimization Failed"}

    w_assets = weights[:-1]
    w_cash = weights[-1]

    # 포트폴리오 델타 계산
    port_delta = 0.0
    for i, w in enumerate(w_assets):
        port_delta += w * TARGET_ASSETS[i]['delta']

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

    atm = round(kospi / 2.5) * 2.5
    strikes = [atm + 10.0, atm + 2.5, atm - 10.0, atm - 2.5]
    prices = [1.0, 2.0, 1.0, 2.0]           ###  시장 데이터 입수 ###
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
        # 가격이 0이거나 너무 작아서 나누기 오류가 나지 않도록 방지
        if prices[i] * MULTIPLIER == 0:
            qty = 0
            asset_premium_pnl = 0
        else:
            qty = int((capital * w) / (prices[i] * MULTIPLIER))

            # 프리미엄 P&L 계산
            # Long 포지션은 프리미엄 지급 (음수), Short 포지션은 프리미엄 수취 (양수)
            if pos_type == "Long":
                asset_premium_pnl = - (qty * prices[i] * MULTIPLIER)
            else:   # Short
                asset_premium_pnl = (qty * prices[i] * MULTIPLIER)

        total_premium_pnl += asset_premium_pnl

        lines.append(
            f"| {asset['name']:<18} | {strikes[i]:<6.1f} | {pos_type:<6} | {asset['delta']:>5.2f} | {w * 100:>5.1f}% | {qty:>3} 계약 | {prices[i]:>6.2f} | {role}"
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
def run_simulation(view_text: str, risk_level: float = 3.0):
    inputs = {
        "kospi_index": 362.30, "market_iv": 13.5, "total_capital": 10_000_000, ### 시장 데이터 입수
        "manager_view": view_text, "risk_aversion": risk_level,
        "expected_returns": [], "covariance_matrix": [], "optimal_weights": [], "final_report": ""
    }

    try:
        result = app.invoke(inputs)
        print(result['final_report'])
    except Exception as e:
        print(f"❌ Simulation Error: {e}")


scenarios = [
# ("Strong Bull", "외국인 현선물 동반 대량 매수. 삼성전자 반등. 상승 확신 90%.", 2.0),
    # ("Strong Bear", "미국 CPI 쇼크 및 나스닥 급락. 하락 확신 90%.", 3.0),
    # ("Neutral/Hedging", "방향성 탐색 구간. 큰 움직임은 없으나 리스크 관리가 필요함.", 3.0),
    # ("Weak Bull with Fear", "완만한 상승이 예상되나, 지정학적 리스크로 인한 급락 가능성이 있어 하방 헤지가 필수적임.", 5.0),
    # ("High Volatility Explosion", "곧 중대 발표가 있음. 방향은 알 수 없으나 위든 아래든 5% 이상 급변동할 것으로 예상됨.", 3.0),
    # ("Box Range", "거래량이 말라붙었고 특별한 모멘텀이 없음. 좁은 박스권 등락 반복 예상.", 3.0),
    # ("Black Swan Defense", "금융 위기 전조 증상 발생. 수익보다는 자산 방어가 최우선 목표임.", 10.0),
    # ("Technical Rebound", "과매도 구간 진입에 따른 기술적 반등 구간. 추세 전환은 아니며 단기 트레이딩 관점 접근.", 3.0),
     ("Moderate Bull", "국내 경제 지표 개선으로 완만한 상승세를 예상. 상승 확률 70%.", 3.0),
    # ("Moderate Bear", "미국 금리 인상 우려로 하락 가능성이 높음. 하락 확률 70%.", 3.0),
    # ("Earnings Season", "기업 실적 발표 시즌. 일부 업종 호조, 전반적으로 중립적인 시장 예상.", 4.0),
    # ("Geopolitical Tension", "지정학적 리스크 증가. 시장의 하락 가능성이 높아지고 변동성이 확대될 것으로 예상됨.", 6.0),
    # ("Election Rally", "대선 시즌으로 인한 단기 랠리 예상. 상승 가능성 80%.", 3.0),
    # ("Inflation Fear", "인플레이션 우려로 인해 시장의 하락 압력이 커지고 있음. 하락 가능성 80%.", 3.0),
    # ("Global Recovery", "글로벌 경기 회복으로 인해 위험 자산 선호도가 증가. 상승 가능성 85%.", 3.0),
    # ("Stagflation Risk", "경기 침체와 높은 인플레이션이 동시에 발생할 가능성. 하락 가능성 75%.", 7.0),
    # ("Sector Rotation", "성장주에서 가치주로 섹터 로테이션이 발생. 시장은 중립적이나 특정 업종이 강세.", 3.0),
    # ("Liquidity Crunch", "유동성 축소로 인해 시장의 하락 압력이 커지고 있음. 하락 가능성 85%.", 8.0),
    # ("Market Euphoria", "투자 심리가 과열되며 시장 전반적으로 상승세. 상승 가능성 90%.", 3.0),
    # ("Defensive Positioning", "시장이 고점에 도달한 것으로 보이며 방어적인 포지션이 필요함.", 9.0)
]

for i, (name, view, risk_level) in enumerate(scenarios, 1):
    print(f"\n🚀 [Scenario {i}: {name}]")
    run_simulation(view, risk_level)

