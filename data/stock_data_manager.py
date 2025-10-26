"""
Stock Data Manager with in-memory caching
OHLCV 데이터를 가져오고 메모리에 캐싱하는 모듈
"""

from pykrx import stock
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import pandas as pd

class StockDataNotFoundError(Exception):
    """주식 데이터를 찾을 수 없을 때 발생하는 예외"""
    pass


class StockDataManager:
    """
    주식 데이터를 관리하는 싱글톤 클래스
    메모리 캐싱을 통해 같은 데이터를 여러 번 조회하지 않음
    """

    _instance = None
    _cache = {}    # {(ticker, start_date, end_date): DataFrame}
    _ticker_cache = {}    # {(stock_name, date): ticker}
    _business_days_cache = {}    # {(start_date, end_date): List[str]}
    _kospi_tickers_cache = {}   # {date: List[ticker]}

    def __new__(cls):
        """싱글톤 패턴: 인스턴스가 하나만 존재하도록"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    

    def get_ticker_from_name(self, stock_name: str, target_date: str) -> str:
        """
        종목명으로 종목코드를 찾는 함수 (캐싱 적용)

        Input:
            stock_name: 종목명 (예: "삼성전자")
            taret_date: 'YYYYMMDD' 형식의 문자열

        Output:
            종목코드 (예: "005930)

        Raises:
            StockDataNotFoundError: 종목을 찾을 수 없는 경우
        """
        cache_key = (stock_name, target_date)

        # 캐시 확인
        if cache_key in self._ticker_cache:
            print(f"    캐시에서 티커 로드: {stock_name} -> {self._ticker_cache[cache_key]}")
            return self._ticker_cache[cache_key]
        
        print(f"    API 호출: {stock_name} 티커 검색 중...")
        tickers = stock.get_market_ticker_list(target_date)

        for ticker in tickers:
            name = stock.get_market_ticker_name(ticker)
            if name == stock_name:
                self._ticker_cache[cache_key] = ticker
                print(f"    Ticker found: {stock_name} -> {ticker}")
                return ticker
            
        raise StockDataNotFoundError(f"'{stock_name}' 종목을 찾을 수 없습니다.")


    def get_business_days(
            self,
            start_date: str,
            end_date: str,
            force_refresh: bool = False,
    ) -> List[str]:
        """
        특정 기간의 실제 거래일(business days) 목록을 반환

        Input:
            start_date : 시작일 'YYYYMMDD'
            end_date : 종료일 'YYYYMMDD'
            force_refresh : True면 캐시 무시하고 새로 가져오기

        Output:
            거래일 문자열 리스트 ['20241001', '20241002', ...]
        """
        cache_key  = (start_date, end_date)

        if not force_refresh and cache_key in self._business_days_cache:
            print(f"    Cache에서 거래일 로드: {start_date}~{end_date} ({len(self._business_days_cache[cache_key])}일)")
            return self._business_days_cache[cache_key].copy()
        
        print(f"    API 호출 : 거래일 목록 조회 중... ({start_date}~{end_date})")
        df = stock.get_index_ohlcv(start_date, end_date, "1001")    # KOSPI 지수

        if df.empty:
            raise StockDataNotFoundError(
                f"{start_date}~{end_date} 기간에 거래일이 없습니다."
            )
        
        business_days = [date.strftime('%Y%m%d') for date in df.index]

        self._business_days_cache[cache_key] = business_days
        print(f"    거래일 {len(business_days)}개 캐시에 저장 완료")

        return business_days.copy()
    

    def get_business_days_before(
            self,
            target_date: str,
            window_size: int,
            include_target: bool = True
    ) -> List[str]:
        """
        target_date로부터 과거 window_size개의 거래일 반환

        Input:
            target_date: 기준일 'YYYYMMDD'
            window_size: 가져올 거래일 개수
            include_target: True면 target_date 포함, False면 그 이전부터

        Output:
            거래일 문자열 리스트 (오래된 순서)

        Example:
            target_date = '20241025', window_size=5, include_target=True
            -> ['20241018', '20241021', '20241022', '20241023', '20241024', '20241025']
        """
        target_dt = datetime.strptime(target_date, '%Y%m%d')
        start_dt = target_dt - timedelta(days=window_size * 3)
        start_date = start_dt.strftime('%Y%m%d')

        all_business_days = self.get_business_days(start_date, target_date)

        if target_date not in all_business_days:
            raise StockDataNotFoundError(
                f"{target_date}는 거래일이 아닙니다."
            )
        
        target_idx = all_business_days.index(target_date)

        if include_target:
            start_idx = max(0, target_idx - window_size + 1)
            selected_days = all_business_days[start_idx:target_idx + 1]
        else:
            start_idx = max(0, target_idx - window_size)
            selected_days = all_business_days[start_idx:target_idx]

        if len(selected_days) < window_size:
            print(f"    경고: 요청한 {window_size}개 보다 적은 {len(selected_days)}개만 존재합니다.")

        print(f"    거래일 {len(selected_days)}개 선택: {selected_days[0]} ~ {selected_days[-1]}")
        return 
    

    def get_kospi_tickers(
            self,
            target_date: str,
            force_refresh: bool = False
    ) -> List[str]:
        """
        특정 날짜의 KOSPI 전체 종목 티커 리스트 반환

        Input:
            target_date: 'YYYYMMDD' 형식
            force_refresh: True면 캐시 무시하고 새로 가져오기

        Output:
            티커 리스트 ['005930', '000660', ...]
        """

        if not force_refresh and target_date in self._kospi_tickers_cache:
            print(f"    Cache에서 KOSPI 티커 로드: {len(self._kospi_tickers_cache[target_date])}개")
            return self._kospi_tickers_cache[target_date].copy()
        
        print(f"    API 호출: KOSPI 종목 리스트 조회 중... ({target_date})")
        tickers = stock.get_market_ticker_list(target_date, market="KOSPI")

        if not tickers:
            raise StockDataNotFoundError(
                f"{target_date}에 KOSPI 종목을 찾을 수 없습니다."
            )
        
        self._kospi_tickers_cache[target_date] = tickers
        print(f"    KOSPI 종목 {len(tickers)}개 캐시에 저장 완료")

        return tickers.copy()
    

    def get_ohlcv(
            self,
            ticker: str,
            start_date: str,
            end_date: str,
            force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        OHLCV 데이터를 가져오는 함수 (캐싱 적용)

        Input:
            ticker: 종목코드 (예: "005930")
            start_date: 시작일 'YYYYMMDD'
            end_date: 종료일 'YYYYMMDD'
            force_refresh: True면 캐시 무시하고 새로 가져오기

        Output:
            OHLCV DataFrmae
        """
        cache_key = (ticker, start_date, end_date)

        if not force_refresh and cache_key in self._cache:
            print(f"    Cache에서 OHLCV 로드: {ticker} ({start_date}~{end_date})")
            return self._cache[cache_key].copy()
        
        print(f"    API 호출: {ticker} OHLCV 다운로드 중... ({start_date}~{end_date})")
        df = stock.get_market_ohlcv(start_date, end_date, ticker)

        if df.empty:
            raise StockDataNotFoundError(
                f"{start_date}~{end_date}에 {ticker}의 거래 데이터가 없습니다."
            )
        
        self._cache[cache_key] = df.copy()
        print(f"    OHLCV 데이터 캐시에 저장 완료")
        
        return df.copy()
    

    def get_multiple_stocks_ohlcv(
            self,
            target_date: str,
            window_size: int,
            tickers: Optional[List[str]] = None,
            include_target: bool = True,
            field: str = '종가',
            fill_na: bool = False
    ) -> pd.DataFrame:
        """
        여러 종목의 OHLCV 데이터를 한 번에 가져오기
        target_date 기준으로 window_size만큼의 거래일 데이터 반환

        Input:
            target_date: 기준일 'YYYYMMDD'
            window_size: 가져올 거래일 개수
            tickers: 종목 리스트 (None이면 KOSPI 전체)
            include_target: True면 target_date 포함
            field: 가져올 필드 ('시가', '고가', '저가', '종가', '거래량', '등락률')
            fill_na: True면 NaN을 forward fill로 채움

        Output:
            DataFrame (index: 날짜, columns: 티커들)

        Example:
            >>> df = manager.get_multiple_stocks_ohlcv('20241025', 20, field='종가')
            >>> df.shape
            (20, 900)    # 20 거래일 x 900 KOSPI 종목
        """
        print(f"\n{'='*60}")
        print("여러 종목 데이터 수집 시작")
        print(f"    - 기준일: {target_date}")
        print(f"    - Window size: {window_size}")
        print(f"    - 필드: {field}")
        print(f"{'='*60}\n")

        business_days = self.get_business_days_before(
            target_date,
            window_size,
            include_target
        )

        start_date = business_days[0]
        end_date = business_days[-1]

        if tickers is None:
            tickers = self.get_kospi_tickers(target_date)

        print(f"    대상 종목: {len(tickers)}개")
        print(f"    기간: {start_date} ~ {end_date} ({len(business_days)} 거래일)\n")

        result_dict = {}
        failed_tickers = []

        for i, ticker in enumerate(tickers, 1):
            try:
                # OHLCV 데이터 가져오기
                df = self.get_ohlcv(ticker, start_date, end_date)

                if field in df.columns:
                    result_dict[ticker] = df[field]
                else:
                    print(f"    [{i}/{len(tickers)}] {ticker}: '{field}' 컬럼 없음")
                    failed_tickers.append(ticker)

                
                if i % 10 == 0:
                    print(f"    진행: {i}/{len(tickers)} 종목 완료...")

            except StockDataNotFoundError as e:
                print(f"    [{i}/{len(tickers)}] {ticker}: 데이터 없음")
                failed_tickers.append(ticker)

            except Exception as e:
                print(f"    [{i}/{len(tickers)}] {ticker}: 오류 - {e}")
                failed_tickers.append(ticker)

        if not result_dict:
            raise StockDataNotFoundError("수집된 데이터가 없습니다.")
        
        result_df = pd.DataFrame(result_dict)

        business_days_dt = [datetime.strptime(d, '%Y%m%d') for d in business_days]
        result_df.index = pd.to_datetime(result_df.index)
        result_df = result_df.reindex(business_days_dt)

        if fill_na:
            result_df = result_df.fillna(method='ffill')
            print(f"    NaN을 forward fill로 채움")

        print(f"\n{'='*60}")
        print("데이터 수집 완료!")
        print(f"    - Shape : {result_df.shape} (거래일 x 종목)")
        print(f"    - Success: {len(result_dict)}")
        print(f"    - Failed: {len(failed_tickers)}")
        print(f"\n{'='*60}")

        return result_df


    def get_stock_state_variables(self, stock_name: str, target_date: str) -> Dict:
        """
        특정 날짜의 state variables를 계산
        
        Input:
            stock_name: 종목명 (예: "삼성전자")
            target_date: 'YYYYMMDD' 형식의 문자열
        
        Output:
            {'return': 등락률, 'volatility': (고가-저가)/시가} 딕셔너리
            
        Raises:
            StockDataNotFoundError: 데이터를 찾을 수 없는 경우
        """
        ticker = self.get_ticker_from_name(stock_name, target_date)
        
        target_date_datetime = datetime.strptime(target_date, "%Y%m%d")
        start_date = target_date_datetime - timedelta(days=10)
        start_date_str = start_date.strftime('%Y%m%d')
        
        df = self.get_ohlcv(ticker, start_date_str, target_date)
        
        ohlcv = df.iloc[-1]
        
        open_price = ohlcv['시가']
        if open_price == 0:
            raise StockDataNotFoundError(
                f"{target_date}에 {stock_name}의 시가가 0입니다."
            )
        
        return {
            'return': ohlcv['등락률'],
            'volatility': (ohlcv['고가'] - ohlcv['저가']) / open_price * 100,
        }


    def clear_cache(self):
        """Cache Initialization"""
        self._cache.clear()
        self._ticker_cache.clear()
        self._business_day_cache.clear()
        self._kospi_tickers_cache.clear()
        print("    Cache 초기화 완료")

    
    def get_cache_info(self) -> Dict:
        """캐시 상태 정보 반환"""
        return {
            'ohlcv_cached' : len(self._cache),
            'ticker_cached' : len(self._ticker_cache),
            'business_days_cached' : len(self._business_days_cache),
            'kospi_tickers_cached' : len(self._kospi_tickers_cache),
            'total_memory_items' : (
                len(self._cache) + 
                len(self._ticker_cache) +
                len(self._business_days_cache) + 
                len(self._kospi_tickers_cache)
            )
        }
    

def get_stock_data_manager() -> StockDataManager:
    """StockDataManager 싱글톤 인스턴스 반환"""
    return StockDataManager()