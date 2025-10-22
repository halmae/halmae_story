import requests
import json
from bs4 import BeautifulSoup
import sys
from pathlib import Path

# 상위 디렉토리를 path에 추가 (config import를 위해)
sys.path.append(str(Path(__file__).parent.parent))

from typing import Optional, List, Dict

from pykrx import stock
from datetime import datetime, timedelta

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, GOOGLE_API_KEY


# 커스텀 예외 클래스
class StockDataNotFoundError(Exception):
    """주식 데이터를 찾을 수 없을 때 발생하는 예외"""
    pass


# LLM 초기화
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=False
)


def get_ticker_from_name(stock_name: str, target_date: str) -> str:
    """
    종목명으로 종목코드를 찾는 함수
    
    Input:
        stock_name: 종목명 (예: "삼성전자")
        target_date: 'YYYYMMDD' 형식의 문자열
    
    Output:
        종목코드 (예: "005930")
        
    Raises:
        StockDataNotFoundError: 종목을 찾을 수 없는 경우
    """
    tickers = stock.get_market_ticker_list(target_date)
    
    for ticker in tickers:
        name = stock.get_market_ticker_name(ticker)
        if name == stock_name:
            return ticker
    
    raise StockDataNotFoundError(f"'{stock_name}' 종목을 찾을 수 없습니다.")

def get_stock_state_variables(stock_name: str, target_date: str) -> Dict:
    """
    Input:
        stock_name: 종목명 (예: "삼성전자", "SK하이닉스")
        target_date: 'YYYYMMDD' 형식의 문자열

    Output:
        {'return': 등락률, 'volatility': (고가-저가)/시가} 딕셔너리
        
    Raises:
        StockDataNotFoundError: 데이터를 찾을 수 없는 경우
    """
    ticker = get_ticker_from_name(stock_name, target_date)
    
    target_date_datetime = datetime.strptime(target_date, "%Y%m%d")
    start_date = target_date_datetime - timedelta(days=10)
    start_date_str = start_date.strftime('%Y%m%d')

    df = stock.get_market_ohlcv(start_date_str, target_date, ticker)
    
    if df.empty:
        raise StockDataNotFoundError(
            f"{target_date}에 {stock_name}({ticker})의 거래 데이터가 없습니다."
        )

    ohlcv = df.iloc[-1]
    
    # 시가가 0인 경우 체크
    open_price = ohlcv['시가']
    if open_price == 0:
        raise StockDataNotFoundError(
            f"{target_date}에 {stock_name}의 시가가 0입니다."
        )

    return {
        'return': ohlcv['등락률'],
        'volatility': (ohlcv['고가'] - ohlcv['저가']) / open_price * 100,  # 퍼센트로 통일
    }


def naver_news(query: str, limit: int) -> Optional[List[dict]]:
    naver_url = "https://openapi.naver.com/v1/search/news.json"

    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }

    params = {
        "query": query,
        "display": limit,
        "sort" : "date"
    }

    # API call
    response = requests.get(naver_url, headers=headers, params=params)

    if response.status_code == 200:
        result = response.json()
        return result['items']
    else:
        print(f"오류 발생: {response.status_code}")
        return None

def filter_naver_news(items: Optional[List[dict]]) -> List[dict]:
    """네이버 뉴스만 필터링"""
    if items is None:
        return []
    
    return [item for item in items if 'n.news.naver.com' in item['link']]

def naver_news_to_article(data: Optional[dict]) -> str:
    if data is None:
        print("데이터가 None입니다.")
        return None
    
    news_title = data['title']
    news_url = data['link']

    if 'n.news.naver.com' not in news_url:
        print(f"네이버 뉴스가 아님: {news_url}")
        return None

    try:
        response = requests.get(news_url)
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')
        news_body = soup.find('div', class_='newsct_article _article_body')

        # news_body가 None인 경우 체크 (중요!)
        if news_body is None:
            print(f"본문을 찾을 수 없음: {news_url}")
            return None

        article_content = news_body.get_text(strip=True)

        return news_title, article_content  # Tuple 반환

    except Exception as e:
        print(f"에러 발생 ({news_url}): {e}")
        return None
    

def explain_stock_with_news(stock_name: str, target_date: str, news_limit: int=10):
    """
    주식의 state variables를 뉴스로 설명

    Input:
        stock_name: 종목명 (예: "삼성전자")
        target_date: 'YYYYMMDD' 형식
        news_limit: 가져올 뉴스 개수

    Output:
        state_variables와 관련 뉴스 기사들
    """
    print(f"\n{'='*50}")
    print(f"[{stock_name}] {target_date} 분석")
    print(f"{'='*50}\n")

    # 1. State Variables
    try:
        state_vars = get_stock_state_variables(stock_name, target_date)
    except StockDataNotFoundError as e:
        print(f"❌ {e}")
        return None

    print("State Variables:")
    print(f"    - 등락률: {state_vars['return']:.2f}%")
    print(f"    - 변동성(고가-저가)/시가: {state_vars['volatility']:.2f}%")

    # 2. Search News
    print(f"'{stock_name}' 관련 뉴스 검색 중...")
    news_items = naver_news(stock_name, news_limit)
    if news_items is None:
        print("뉴스를 가져올 수 없습니다.")
        return None

    # 3. Naver News Filtering
    filtered_news = filter_naver_news(news_items)
    print(f"네이버 뉴스 {len(filtered_news)}개 발견\n")

    # 4. Extracting Articles
    articles = []
    for i, news in enumerate(filtered_news):
        print(f"[{i}] 기사 추출 중...")
        article = naver_news_to_article(news)
        if article:
            title, content = article
            articles.append({
                'title': title,
                'content': content,
                'url': news['link']
            })
        print()

    return {
        'stock_name': stock_name,
        'date': target_date,
        'state_variables': state_vars,
        'articles': articles
    }


def summary_agent(title: str, content: str, llm) -> Optional[str]:
    """
    뉴스 제목과 본문을 3문장으로 요약하는 agent
    
    Input:
        title: 뉴스 제목
        content: 뉴스 본문
        llm: LLM 클라이언트
    
    Output:
        3문장 요약 문자열
    """

    if not title or not content:
        print("제목 또는 본문이 비어있습니다.")
        return None
    
    prompt_template = PromptTemplate(
        input_variables=["title", "article"],
        template="""
다음 뉴스 기사를 3문장으로 요약해주세요.
핵심 내용만 간단명료하게 정리해주세요.

기사 제목:
{title}

기사 내용:
{article}

Summary:
"""
    )

    try:
        summary_chain = LLMChain(
            llm=llm,
            prompt=prompt_template
        )

        summary = summary_chain.run(title=title, article=content)
        return summary.strip()
    
    except Exception as e:
        print(f"요약 중 오류 발생: {e}")
        return None


def sentiment_agent(title:str, summary: str, llm) -> Optional[Dict[str, str]]:
    """
    뉴스 제목과 요약을 바탕으로 감성 분석하는 agent

    Input:
        title: 뉴스 제목
        summary: 3문장 요약
        llm: LLM 클라이언트

    Output:
        {
            'sentiment' : '긍정' | '부정' | '중립',
            'reason' : '한 문장 설명'
        }
    """
    if not title or not summary:
        print("제목 또는 요약이 비었습니다.")
        return None

    prompt_template = PromptTemplate(
        input_variables = ["title", "summary"],
        template = """
다음 뉴스 기사의 감성을 분석해주세요.

기사 제목:
{title}

기사 요약:
{summary}

아래 형식으로 정확히 답변해주세요:
감성: [긍정/부정/중립 중 하나만 선택]
이유: [한 문장으로 설명]

감성 분석:
        """
    )

    try:
        sentiment_chain = LLMChain(
            llm=llm,
            prompt=prompt_template
        )

        result = sentiment_chain.run(title=title, summary=summary)

        lines = result.strip().split('\n')
        sentiment = None
        reason = None

        for line in lines:
            if line.startswith('감성:'):
                sentiment = line.replace('감성:', '').strip()
            elif line.startswith('이유:'):
                reason = line.replace('이유:', '').strip()

        if sentiment and reason:
            return {
                'sentiment': sentiment,
                'reason': reason
            }
        else:
            print("감성 분석 결과를 파싱할 수 없습니다.")
            return None

    except Exception as e:
        print(f"감성 분석 중 오류 발생: {e}")
        return None


def explain_state_variables_agent(
    stock_name: str,
    date: str,
    state_variables: Dict,
    news_analysis: List[Dict],
    llm
) -> Optional[str]:
    """
    뉴스 분석을 바탕으로 state variables를 설명하는 agent
    """
    if not state_variables or not news_analysis:
        print("State variables 또는 뉴스 분석 데이터가 없습니다.")
        return None
    
    # 뉴스 분석 결과를 문자열로 포맷팅
    news_summary = ""
    for i, news in enumerate(news_analysis, 1):
        news_summary += f"""
[뉴스 {i}]
제목: {news['title']}
요약: {news['summary']}
감성: {news['sentiment']}
이유: {news['reason']}
"""
    
    prompt_template = PromptTemplate(
        input_variables=["stock_name", "date", "stock_return", "volatility", "news_summary"],
        template="""
당신은 주식 시장 전문 애널리스트입니다.
다음 정보를 바탕으로 {stock_name}의 {date} 주가 변동을 분석하고 설명해주세요.

📊 State Variables:
- 등락률: {stock_return}%
- 변동성 (고가-저가)/시가: {volatility}%

📰 관련 뉴스 분석:
{news_summary}

위 뉴스들을 종합하여, State Variables(등락률, 변동성)가 왜 이렇게 나타났는지 명확하게 설명해주세요.
각 지표와 뉴스의 연관성을 구체적으로 분석해주세요.

분석 결과:
"""
    )
    
    try:
        explain_chain = LLMChain(
            llm=llm,
            prompt=prompt_template
        )
        
        explanation = explain_chain.run(
            stock_name=stock_name,
            date=date,
            stock_return=state_variables['return'],
            volatility=state_variables['volatility'],
            news_summary=news_summary
        )
        
        return explanation.strip()
        
    except Exception as e:
        print(f"설명 생성 중 오류 발생: {e}")
        return None
    

def stock_news_analysis_pipeline(
    stock_name: str,
    target_date: str,
    llm,
    news_limit: int = 100,
    max_articles: int = 10
) -> Optional[Dict]:
    """
    주식의 state variables를 뉴스로 분석하고 설명하는 전체 파이프라인
    
    Input:
        stock_name: 종목명 (예: "삼성전자")
        target_date: 'YYYYMMDD' 형식
        llm: LLM 클라이언트
        news_limit: 검색할 뉴스 개수
        max_articles: 분석할 최대 기사 수
    
    Output:
        {
            'stock_name': str,
            'date': str,
            'state_variables': dict,
            'news_analysis': list,
            'final_explanation': str
        }
    """
    print(f"\n{'='*60}")
    print(f"📊 [{stock_name}] {target_date} 종합 분석 시작")
    print(f"{'='*60}\n")
    
    # ============================================
    # Step 1: State Variables 가져오기
    # ============================================
    print("📈 Step 1: State Variables 수집 중...")
    try:
        state_vars = get_stock_state_variables(stock_name, target_date)
    except StockDataNotFoundError as e:
        print(f"❌ {e}")
        return None
    
    print(f"  ✓ 등락률: {state_vars['return']:.2f}%")
    print(f"  ✓ 변동성: {state_vars['volatility']:.2f}%\n")
    
    # ============================================
    # Step 2: 뉴스 수집 및 필터링
    # ============================================
    print(f"🔍 Step 2: '{stock_name}' 관련 뉴스 검색 중...")
    news_items = naver_news(stock_name, news_limit)
    if news_items is None:
        print("❌ 뉴스를 가져올 수 없습니다.")
        return None
    
    filtered_news = filter_naver_news(news_items)
    print(f"  ✓ 네이버 뉴스 {len(filtered_news)}개 발견\n")
    
    # ============================================
    # Step 3: 각 뉴스 분석 (요약 + 감성분석)
    # ============================================
    print(f"📰 Step 3: 뉴스 분석 중 (최대 {max_articles}개)...")
    news_analysis = []
    
    for i, news in enumerate(filtered_news[:max_articles], 1):
        print(f"\n  [{i}/{min(len(filtered_news), max_articles)}] 분석 중...")
        
        # 기사 본문 추출
        article = naver_news_to_article(news)
        if article is None:
            print(f"    ⚠️  본문 추출 실패")
            continue
        
        title, content = article
        print(f"    제목: {title[:50]}...")
        
        # 요약
        print(f"    - 요약 생성 중...")
        summary = summary_agent(title, content, llm)
        if summary is None:
            print(f"    ⚠️  요약 실패")
            continue
        print(f"    ✓ 요약 완료")
        
        # 감성 분석
        print(f"    - 감성 분석 중...")
        sentiment = sentiment_agent(title, summary, llm)
        if sentiment is None:
            print(f"    ⚠️  감성 분석 실패")
            continue
        print(f"    ✓ 감성: {sentiment['sentiment']}")
        
        news_analysis.append({
            'title': title,
            'summary': summary,
            'sentiment': sentiment['sentiment'],
            'reason': sentiment['reason'],
            'url': news['link']
        })
    
    if not news_analysis:
        print("\n❌ 분석 가능한 뉴스가 없습니다.")
        return None
    
    print(f"\n  ✓ 총 {len(news_analysis)}개 뉴스 분석 완료\n")
    
    # ============================================
    # Step 4: State Variables 종합 설명
    # ============================================
    print("📝 Step 4: State Variables 종합 설명 생성 중...")
    final_explanation = explain_state_variables_agent(
        stock_name=stock_name,
        date=target_date,
        state_variables=state_vars,
        news_analysis=news_analysis,
        llm=llm
    )
    
    if final_explanation is None:
        print("❌ 최종 설명 생성 실패")
        return None
    
    print("  ✓ 설명 생성 완료\n")
    
    # ============================================
    # 결과 반환
    # ============================================
    print(f"{'='*60}")
    print("✅ 분석 완료!")
    print(f"{'='*60}\n")
    
    return {
        'stock_name': stock_name,
        'date': target_date,
        'state_variables': state_vars,
        'news_analysis': news_analysis,
        'final_explanation': final_explanation
    }