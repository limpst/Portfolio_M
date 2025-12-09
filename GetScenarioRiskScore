from enum import Enum
from dataclasses import dataclass
from typing import List
import json
import re

from bisect import bisect

import mysql
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import os

from mysql.connector import pooling

# from LLMTradEx34Opt import fetch_latest_news, insert_market_scenario, market_scenario_to_tuple

load_dotenv()


# ==========================================
# 1. 분석 기준 정의 (Enums)
# ==========================================
class MarketTrend(Enum):
    BULLISH = "상승 (Bullish)"
    BEARISH = "하락 (Bearish)"
    NEUTRAL = "중립/횡보 (Neutral)"
    VOLATILE = "변동성 확대 (Volatile)"
    REVERSAL = "추세 반전 (Reversal)"

class RiskLevel(Enum):
    LOW = "안정 (Low)"
    MODERATE = "보통 (Moderate)"
    HIGH = "주의 (High)"
    EXTREME = "위기/경보 (Extreme)"

class MarketDriver(Enum):
    MACRO = "거시경제(금리/물가)"
    SUPPLY = "수급(외인/기관)"
    GEOPOLITIC = "지정학/정치"
    EARNINGS = "실적/기업 펀더멘털"
    TECHNICAL = "기술적 분석/심리"
    MIXED = "복합 요인"

class ScoreDesc(Enum):
    score_descGt8 = "극도의 공포 (안전자산 선호)"
    score_descGt6_5 = "경계감 확산 (위험 회피)"
    score_descGt5_5 = "중립 (관망)"
    score_descGt3_5 = "위험 선호 (매수세 유입)"
    score_descElse = "극도의 탐욕 (적극 매수)"

# ==========================================
# 2. 최종 시나리오 객체
# ==========================================
@dataclass
class MarketScenario:
    title: str              # 시나리오 명
    trend: MarketTrend      # 시장 추세
    risk: RiskLevel         # 리스크 레벨
    driver: MarketDriver    # 주요 동인
    risk_aversion_score: float # 위험 회피 지수 (2.0 ~ 10.0)
    score_desc: ScoreDesc  # 위험 회피 지수 description
    summary: str            # 종합 분석 요약
    key_factors: List[str]  # 판단 근거
    strategy: str           # 추천 투자 전략

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
# 3. LLM 분석 함수 (프롬프트 수정됨)
# ==========================================
def analyze_market_scenario(news_items: list, llm: ChatOpenAI) -> MarketScenario:
    """
    여러 건의 뉴스를 종합 분석하여 시장 시나리오를 생성합니다.
    """

    # 뉴스 텍스트화
    news_context = ""

    for idx, news in enumerate(news_items, 1):
        news_context += f"[{idx}] 제목: {news['title']}\n    본문요약: {news['body'][:2000]}...\n\n"

    # 시스템 프롬프트 (위험 회피 지수 적용)
    system_prompt = """
    당신은 월스트리트의 수석 투자 전략가(CIO)입니다.
    제공된 여러 건의 금융 뉴스를 종합하여 현재 시장의 국면을 진단하고 투자 시나리오를 작성하세요.

    [분석 지침]
    1. 호재와 악재의 경중을 비교하여 전체적인 추세(Trend)를 결정하세요.
    2. **위험 회피 지수(Risk Aversion Score)를 2.0에서 10.0 사이의 숫자로 산정하세요.**
       점수가 높을수록 시장이 공포를 느끼고 안전자산을 선호한다는 뜻입니다.

       - **9.0 ~ 10.0**: 극도의 위험 회피 (패닉 셀링, 현금/금 선호, 폭락 공포)
       - **7.0 ~ 8.9**: 위험 회피 심리 확산 (보수적 운용, 하락 우위)
       - **6.0**: 중립 (관망세, 호재와 악재 균형)
       - **4.0 ~ 5.9**: 위험 선호 심리 회복 (저가 매수 유입, 반등 시도)
       - **2.0 ~ 3.9**: 강한 위험 선호 (탐욕, 묻지마 매수, 급등장)

    [시나리오 제목 규칙]
    title 필드는 반드시 아래 목록 중 가장 적절한 하나로만 선택하세요.
    ["Strong Bull", "Strong Bear", "Neutral/Hedging", "Weak Bull with Fear",
     "High Volatility Explosion", "Box Range", "Black Swan Defense",
     "Technical Rebound", "Moderate Bull", "Moderate Bear", "Earnings Season",
     "Geopolitical Tension", "Election Rally", "Inflation Fear",
     "Global Recovery", "Stagflation Risk", "Sector Rotation",
     "Liquidity Crunch", "Market Euphoria", "Defensive Positioning"]
 
    [출력 형식]
    반드시 아래 JSON 포맷으로만 출력하세요. 마크다운(```json)을 쓰지 마세요.
    {
        "title": "시나리오 제목",
        "trend": "BULLISH" | "BEARISH" | "NEUTRAL" | "VOLATILE" | "REVERSAL",
        "risk": "LOW" | "MODERATE" | "HIGH" | "EXTREME",
        "driver": "MACRO" | "SUPPLY" | "GEOPOLITIC" | "EARNINGS" | "TECHNICAL" | "MIXED",
        "risk_aversion_score": 7.5,
        "summary": "종합 분석 내용 (한글 3문장 이내)",
        "key_factors": ["판단 근거1", "판단 근거2", "판단 근거3"],
        "strategy": "추천 투자 전략 (한글)"
    }
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"다음 뉴스들을 분석해줘:\n{news_context}")
        ])

        result_text = response.content.strip()

        # JSON 파싱
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())

            # Enum 매핑
            trend_map = {
                "BULLISH": MarketTrend.BULLISH, "BEARISH": MarketTrend.BEARISH,
                "NEUTRAL": MarketTrend.NEUTRAL, "VOLATILE": MarketTrend.VOLATILE,
                "REVERSAL": MarketTrend.REVERSAL
            }
            risk_map = {
                "LOW": RiskLevel.LOW, "MODERATE": RiskLevel.MODERATE,
                "HIGH": RiskLevel.HIGH, "EXTREME": RiskLevel.EXTREME
            }
            driver_map = {
                "MACRO": MarketDriver.MACRO, "SUPPLY": MarketDriver.SUPPLY,
                "GEOPOLITIC": MarketDriver.GEOPOLITIC, "EARNINGS": MarketDriver.EARNINGS,
                "TECHNICAL": MarketDriver.TECHNICAL, "MIXED": MarketDriver.MIXED
            }

            # 점수 기준과 매핑
            score_thresholds = [3.5, 5.5, 6.5, 8.0]
            score_desc_mapping = [
                ScoreDesc.score_descElse,
                ScoreDesc.score_descGt3_5,
                ScoreDesc.score_descGt5_5,
                ScoreDesc.score_descGt6_5,
                ScoreDesc.score_descGt8
            ]

            score = float(data.get("risk_aversion_score", 6.0))
            score_desc = score_desc_mapping[bisect(score_thresholds, score)]

            # MarketScenario 생성
            return MarketScenario(
                title=data.get("title", "분석 실패"),
                trend=trend_map.get(data.get("trend"), MarketTrend.NEUTRAL),
                risk=risk_map.get(data.get("risk"), RiskLevel.MODERATE),
                driver=driver_map.get(data.get("driver"), MarketDriver.MIXED),
                risk_aversion_score=score,  # 기본값 6.0 (중립)
                score_desc=score_desc,
                summary=data.get("summary", ""),
                key_factors=data.get("key_factors", []),
                strategy=data.get("strategy", "")
            )
        else:
            print("JSON 파싱 실패: ", result_text)
            return None

    except Exception as e:
        print(f"시나리오 생성 중 에러: {e}")
        return None


if __name__ == "__main__":

    OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')

    #  os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY
    #
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0, api_key=OPEN_AI_KEY)

    # ==========================================
    # 실행 결과 확인
    # ==========================================

    test_news = fetch_latest_news(20)  # 최근 뉴스 20 개 가져오기

    market_scenario = analyze_market_scenario(test_news, llm)  # 마켓 시나리오 생성

    insert_market_scenario(market_scenario)  # DB 에 마켓 시나리오 분석 결과 저장

    scenarios = [market_scenario_to_tuple(market_scenario)]  # 시나리오 튜블 저장.

    # test_news = [
    #     {"title": "삼성전자, 1분기 영업이익 6조원 '깜짝 실적'", "body": "반도체 업황 회복에 힘입어..."},
    #     {"title": "엔비디아 5% 급락, 서학개미 울상", "body": "AI 거품론이 제기되며 미국 기술주들이 일제히..."},
    #     {"title": "코스피, 외인 팔자에 2600선 턱걸이 마감", "body": "기관이 방어에 나섰지만 외국인 매도세를 이기지 못하고..."},
    #     {"title": "파월 '금리 인하 서두르지 않겠다'", "body": "미국 연준 의장이 매파적 발언을 내놓으며 국채 금리가 급등..."},
    #     {"title": "HD현대마린, 공모가 최상단 확정... 청약 열기 후끈", "body": "기관 수요예측에서 흥행에 성공하며 IPO 시장에 훈풍..."},
    #     {"title": "정부, '밸류업 프로그램' 가이드라인 발표", "body": "저평가된 기업들의 주가 부양을 위해 세제 혜택을..."}
    # ]

    scenario = analyze_market_scenario(test_news, llm)

    if scenario:
        print("\n" + "="*60)
        print(f"📢  [{scenario.title}]")
        print("="*60)
        print(f"📊 추세 (Trend)   : {scenario.trend.value}")
        print(f"⚠️ 위험 (Risk)    : {scenario.risk.value}")
        print(f"🔑 핵심 동인      : {scenario.driver.value}")

        # 점수 시각화 로직
        score = scenario.risk_aversion_score

        # if score >= 8.0:
        #     score_desc = "😱 극도의 공포 (안전자산 선호)"
        # elif score >= 6.5:
        #     score_desc = "😨 경계감 확산 (위험 회피)"
        # elif score >= 5.5:
        #     score_desc = "😐 중립 (관망)"
        # elif score >= 3.5:
        #     score_desc = "🙂 위험 선호 (매수세 유입)"
        # else:
        #     score_desc = "🤑 극도의 탐욕 (적극 매수)"

        print(f"🛡️ 위험 회피 지수 : {score} / 10.0")
        print(f"   ➤ 상태: {scenario.score_desc.value}")
        print("-" * 60)
        print(f"📝 [종합 요약]\n{scenario.summary}")
        print("-" * 60)
        print("🧐 [판단 근거]")
        for factor in scenario.key_factors:
            print(f"  - {factor}")
        print("-" * 60)
        print(f"💡 [투자 전략]\n{scenario.strategy}")
        print("="*60)
