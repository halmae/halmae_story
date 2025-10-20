import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import time
from functools import lru_cache
import json

class NewsCollector:
    """뉴스 수집 및 전처리 클래스"""

    def __init__(self, client_id:str, client_secret:str):
        """
        Parameters:
            client_id: Naver API Client ID
            client_secret: Naver API Client Secret
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret
        }
        self.base_url = "https://openai.naver.com/v1/search/news.json"

    
    def search_news(self,
                    query: str,
                    display: int=100,
                    start: int=1,
                    sort: str ="date") -> List[Dict]:
        """
        네이버 뉴스 검색

        Parameters:
            query: 검색어 (종목명)
            display: 한 번에 가져올 결과 수 (최대 100)
            start: 시작 인덱스
            sort: 정렬 방식 (date: 날짜순, sim: 관련도순)
        """
        try:
            params = {
                "query": query,
                "display": display,
                "start": start,
                "sort": sort
            }

            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            return data.get('items', [])
        
        except Exception as e:
            print(f"뉴스 검색 실패 ({query}): {e}")
            return []
        
    
    def filter_naver_news(self, items: List[Dict]) -> List[Dict]:
        """네이버 뉴스만 필터링"""
        return [item for item in items if 'n.news.naver.com' in item.get('link', '')]
    

    def extract_article(self, news_item: Dict) -> Optional[Tuple[str, str, str]]:
        """
        네이버 뉴스 본문 추출

        Returns:
            tuple: (제목, 본문, URL) 또는 None
        """
        news_url = news_item.get('link', '')
        news_title = news_item.get('title', '').replace('<b>', '').replace('</b>', '')

        if 'n.news.naver.com' not in news_url:
            return None
        
        try:
            response = requests.get(news_url, timeout=10)
            response.encoding = 'utf-8'
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            news_body = soup.find('div', class_='newsct_article _article_body')

            if not news_body:
                news_body = soup.find('article', id='dic_area')

            if not news_body:
                return None
            
            article_content = news_body.get_text(strip=True)

            if len(article_content) < 100:
                return None
            
            return news_title, article_content, news_url
        
        except Exception as e:
            print(f"본문 추출 실패 ({news_url}): {e}")
            return None
        
    
    def collect_news_batch(self,
                           ticker_name: str,
                           max_articles: int = 10) -> List[Dict]:
        """
        특정 종목의 뉴스를 배치로 수집

        Parameters:
            ticker_name: 종목명 (예: "삼성전자")
            max_articles: 최대 수집 기사 수

        Returns:
            List[Dict]: 수집된 기사 리스트
        """
        print(f"[{ticker_name}] 뉴스 수집 시작...")

        all_items = self.search_news(ticker_name, display=100)

        naver_items = self.filter_naver_news(all_items)
        print(f"    네이버 뉴스 {len(naver_items)}개 발견")

        articles = []
        for i, item in enumerate(naver_items[:max_articles * 2]):
            if len(articles) >= max_articles:
                break

            result = self.extract_article(item)
            if result:
                title, content, url = result

                articles.append({
                    'ticker_name': ticker_name,
                    'title': title,
                    'content': content,
                    'url': url,
                    'pub_date': item.get('pubDate', '')
                })
                print(f"    {len(articles)} / {max_articles} 수집 완료")

            time.sleep(0.1)

        print(f"[{ticker_name}] 총 {len(articles)}개 기사 수집 완료\n")
        return articles
    

    def collect_multiple_tickers(self,
                                 ticker_dict: Dict[str, str],
                                 max_articles_per_ticker: int = 10) -> Dict[str, List[Dict]]:
        """
        여러 종목의 뉴스를 일괄 수집

        Parameters:
            ticker_dict: {ticker_code, ticker_name} 형태의 딕셔너리
                        예: {"005930": "삼성전자", "000660": "SK하이닉스"}
            max_articles_per_ticker: 종목당 최대 기사 수

        Returns:
            Dict[ticker_code, List[Dict]]: 종목별 뉴스 리스트
        """
        print(f"\n{"="*80}")
        print(f"    총 {len(ticker_dict)}개 종목 뉴스 수집 시작")
        print(f"    종목당 최대 {max_articles_per_ticker}개 기사 수집")
        print(f"\n{"="*80}")

        results = {}

        for i, (ticker_code, ticker_name) in enumerate(ticker_dict.items(), 1):
            print(f"[{i}/{len(ticker_dict)}] {ticker_code} - {ticker_name}")

            articles = self.collect_news_batch(
                ticker_name=ticker_name,
                max_articles=max_articles_per_ticker
            )

            results[ticker_code] = articles

            # API 부하 방지
            if i < len(ticker_dict):
                time.sleep(0.5)


        total_articles = sum(len(articles) for articles in results.values())
        print(f"\n{"="*80}")
        print(f"    수집 완료!")
        print(f"    총 {len(ticker_dict)}개 종목 / {total_articles}개 기사")
        print(f"\n{"="*80}")

        return results