import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pykrx import stock
from covariance_estimator import CovarianceEstimator

from datetime import datetime, timedelta
from typing import Optional, List


def get_kospi200_tickers(date:str, exclude_tickers=None) -> List[str]:
    """
    특정 날짜 기준 KOSPI200 구성종목 가져오기

    Parameters:
    - date: 기준 날짜 (YYYYMMDD 형식 문자열)
    - exclude_tickers: 제외할 종목 코드 리스트

    Returns:
    - KOSPI200 종목 코드 리스트
    """
    if exclude_tickers is None:
        exclude_tickers = []

    # KOSPI200 구성종목 가져오기 (KOSPI200 지수 코드: 1028)
    kospi200_tickers = stock.get_index_portfolio_deposit_file("1028")

    # Filtering
    kospi200_tickers = [ticker for ticker in kospi200_tickers if ticker not in exclude_tickers]
    print(f"[{date}] KOSPI200 종목 수: {len(kospi200_tickers)}개")

    return kospi200_tickers


def get_price_data(ticker_list: List[str], end_date: str, window_size: int) -> pd.DataFrame:
    """
    종목들의 주가 데이터 수집

    Parameters:
    - ticker_list: 종목 코드 리스트
    - end_date: 종료 날짜 (YYYYMMDD)
    - window_size: 필요한 데이터 일수

    Returns:
    - DataFrame: 종목별 종가 데이터 (날짜 x 종목)
    """
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=int(window_size * 2))
    start_date = start_dt.strftime("%Y%m%d")

    print(f"초기 데이터 수집 기간: {start_date} ~ {end_date}")

    price_data = {}

    for i, ticker in enumerate(ticker_list):
        if (i + 1) % 50 == 0:
            print(f"진행 중... {i+1}/{len(ticker_list)}")

        try:
            df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
            if not df.empty:
                price_data[ticker] = df['종가']
        except Exception as e:
            print(f"종목 {ticker} 데이터 수집 실패: {e}")

    # DataFrame으로 변환
    df_prices = pd.DataFrame(price_data)

    # NaN 처리
    missing_ratio = df_prices.isnull().sum() / len(df_prices)
    valid_tickers = missing_ratio[missing_ratio < 0.5].index.tolist()
    df_prices = df_prices[valid_tickers]

    df_prices = df_prices.ffill().dropna()

    if len(df_prices) > window_size + 1:
        df_prices = df_prices.iloc[-(window_size + 1):]

    print(f"\n최종 데이터: {len(df_prices)}일 x {len(df_prices.columns)}개 종목")
    print(f"기간: {df_prices.index[0]} ~ {df_prices.index[-1]}")

    return df_prices


def calculate_returns(df_prices: pd.DataFrame) -> np.ndarray:
    """
    종가 데이터로부터 수익률 계산

    Parameters:
    - df_prices: 종가 DataFrame (날짜 x 종목)

    Returns:
    - numpy array: 수익률 데이터 (시간 x 종목)
    """

    returns = (df_prices / df_prices.shift(1) - 1).dropna()

    print(f"수익률 데이터: {returns.shape[0]}일 x {returns.shape[1]}개 종목")

    return returns.values, returns.columns.tolist()



def compute_covariance_matrix(return_data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Rolling window 공분산 행렬 계산
    
    마지막 시점의 공분산 행렬만 반환
    """
    n_timesteps, n_assets = return_data.shape
    
    if n_timesteps < window_size:
        raise ValueError(f"데이터({n_timesteps}일)가 window_size({window_size}일)보다 적습니다")
    
    cov_estimator = CovarianceEstimator(window_size=window_size, n_assets=n_assets)
    
    cov_matrix = None
    for t in range(n_timesteps):
        cov_matrix = cov_estimator.calculate_covariance(t, return_data)
        
        # 진행상황 표시 (선택적)
        if t % 50 == 0 and t > 0:
            print(f"공분산 계산 진행: {t}/{n_timesteps}")
    
    if cov_matrix is None:
        raise RuntimeError("공분산 행렬 계산 실패")
    
    print(f"✓ Covariance matrix 계산 완료: {cov_matrix.shape}")
    return cov_matrix


def get_correlation_matrix(cov_matrix: np.ndarray) -> np.ndarray:
    std_devs = np.sqrt(np.diag(cov_matrix))

    corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

    return corr_matrix


def find_top_correlated_stocks(corr_matrix: np.ndarray,
                               ticker_list: List,
                               target_ticker: str = "005930",
                               top_n: int = 10):
    """
    특정 종목과 correlation이 높은 상위 N개 종목 찾기

    Parameters:
    - corr_matrix: Correlation matrix
    - ticker_list: 종목 코드 리스트
    - target_ticker: 기준 종목 (삼성전자: 005930)
    - top_n: 상위 N개

    Returns:
    - DataFrame: 상위 종목과 correlation 값
    """

    if target_ticker not in ticker_list:
        print(f"! {target_ticker}가 종목 리스트에 없습니다!")
        return None
    
    target_idx = ticker_list.index(target_ticker)
    
    correlations = corr_matrix[target_idx, :]

    df_corr = pd.DataFrame({
        'ticker': ticker_list,
        'correlation': correlations
    })

    df_corr = df_corr[df_corr['ticker'] != target_ticker]
    df_corr = df_corr.sort_values('correlation', ascending=False)

    top_stocks = df_corr.head(top_n).copy()

    top_stocks['name'] = top_stocks['ticker'].apply(
        lambda x: stock.get_market_ticker_name(x) if x else "N/A"
    )

    print(f"\n{'='*70}")
    print(f"  삼성전자({target_ticker})와 상관관계가 높은 상위 {top_n}개 종목")
    print(f"{'='*70}")
    print(f"{'순위':<8}{'종목코드':<12}{'종목명':<25}{'Correlation':<15}")
    print(f"{'-'*70}")
    
    for rank, (idx, row) in enumerate(top_stocks.iterrows(), 1):
        ticker = row['ticker']
        name = row['name']
        corr = row['correlation']
        print(f"{rank:<8}{ticker:<12}{name:<25}{corr:<15.4f}")
    
    print(f"{'='*70}\n")
    
    return top_stocks


def analyze_samsung_correlation(date: str,
                                window_size: int = 252,
                                top_n: int = 10,
                                exclude_tickers: Optional[List[str]] = None):
    """
    특정 날짜 기준으로 삼성전저와 correlation이 높은 종목 분석

    Parameters:
    - date: 분석 기준 날짜 (YYYYMMDD 형식)
    - window_size: 공분산 계산에 사용할 데이터 일수 (기본: 252일 = 1년)
    - top_n: 상위 N개 종목 (기본: 10개)
    - exclude_tickers: 제외할 종목 리스트

    Returns:
    - DataFrame: 상위 종목 정보 (ticker, name, correlation)
    """
    print(f"\n{'='*80}")
    print(f"    삼성전자 Correlation 분석 시작")
    print(f"    기준 날짜: {date}")
    print(f"    Window Size: {window_size}일")
    print(f"{'='*80}\n")

    # Step 1: KOSPI200 종목 가져오기
    print("[1/5] KOSPI200 종목 로딩...")
    ticker_list = get_kospi200_tickers(date, exclude_tickers)

    # Step 2: 주가 데이터 수집
    print("\n[2/5] 주가 데이터 수집...")
    df_prices = get_price_data(ticker_list, date, window_size)

    # Step 3: 수익률 계산
    print("\n[3/5] 수익률 계산...")
    return_data, final_ticker_list = calculate_returns(df_prices)

    # Step 4: 공분산 행렬 계산
    print("\n[4/5] 공분산 행렬 계산...")
    cov_matrix = compute_covariance_matrix(return_data, window_size)

    # Step 5: Correlation 행렬로 변환
    print("\n[5/5] Correlation 계산 및 상위 종목 추출...")
    corr_matrix = get_correlation_matrix(cov_matrix)

    # 삼성전자와 correlation 높은 종목 찾기
    top_stocks = find_top_correlated_stocks(
        corr_matrix=corr_matrix,
        ticker_list=final_ticker_list,
        target_ticker="005930",
        top_n = top_n
    )

    print("Analysis Complete!\n")

    return top_stocks