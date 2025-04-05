import json
import pandas as pd
import os 

from data_builder import DatasetBuilder
from data_visualizer import DataVisualizer

# 현재 파이썬 파일 기준으로 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, 'dataset.json')

# dataset.json 
with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# 데이터셋 빌더 초기화 (public)
builder = DatasetBuilder(dataset)
# 시각화클래스 초기화 (public)
visualizer = DataVisualizer()


# 표준업종 10차코드 로드 
builder.load_kic()
# 데이터 로드 
builder.load_data()
# 차주정보병합
builder.merge_borrower()
# 대출정보 및 시각화 병합
builder.merge_loan_credit()

# -----------------------------------
# 1) 차입자의 운영 사업체 개수 파이플롯 시각화
# -----------------------------------
visualizer.df_setter(builder.df_차주_기업개요정보_집계)
visualizer.pie('BIZ_COUNT','COUNT','차주 운영 사업체 개수(Pie Chart)')

# -------------------------------------------------
# 2) 평균대출금액 박스플롯 시각화, 연령별 금액 히스토그램 시각화
# -------------------------------------------------
df_차주_대출_신용카드_연체정보_소액 = builder.df_차주_대출_신용카드_연체정보[(builder.df_차주_대출_신용카드_연체정보['AMT'] <= 50000) & (builder.df_차주_대출_신용카드_연체정보['대출신용카드구분']=='대출')]
df_차주_대출_신용카드_연체정보_중고액 = builder.df_차주_대출_신용카드_연체정보[(builder.df_차주_대출_신용카드_연체정보['AMT'] > 50000) & (builder.df_차주_대출_신용카드_연체정보['대출신용카드구분']=='대출')]

df_차주_대출_신용카드_연체정보_소액  = df_차주_대출_신용카드_연체정보_소액[df_차주_대출_신용카드_연체정보_소액['BTH_SECTION'] != '~19세']
df_차주_대출_신용카드_연체정보_중고액  = df_차주_대출_신용카드_연체정보_중고액[df_차주_대출_신용카드_연체정보_중고액['BTH_SECTION'] != '~19세']

visualizer.df_setter(df_차주_대출_신용카드_연체정보_소액)
visualizer.box('BTH_SECTION','AMT','연령구간','대출금액 (천원)','연령대별 대출금액 분포 (Boxplot) - 소액(5천만원 이하)')

visualizer.df_setter(df_차주_대출_신용카드_연체정보_중고액)
visualizer.box('BTH_SECTION','AMT','연령구간','대출금액 (천원)','연령대별 대출금액 분포 (Boxplot) - 5천만원 초과')

# ------------------------------------
# 3) 연령구간별 개인대출과목 분포 트리맵 시각화
# ------------------------------------
visualizer.df_setter(builder.df_차주_대출정보_집계)
visualizer.treemap('BTH_SECTION','LN_CD_NM','LN_AMT')

# ------------------------------------
# 4) 기업의 대출분포
# ------------------------------------
visualizer.df_setter(builder.df_기업대출정보)
visualizer.treemap('BTH_SECTION','LN_CD_NM','LN_AMT')

# ----------------------------------------------------------
# 5) 연령별 연체율 막대 시각화 - 평균연체율 표시 (version1 기관매칭 기준)
# ----------------------------------------------------------
visualizer.df_setter(builder.df_차주_대출_신용카드_연체정보_집계)
전체평균연체율 = builder.df_차주_대출_신용카드_연체정보_집계['연체여부'].mean()
visualizer.bar('BTH_SECTION','연체율','연체율','연령구간','연체율 (%)','연령별 연체율 (대출기관 매칭)',1,전체평균연체율,'전체평균')

# ----------------------------------------------------------
# 6) 연령별 연체율 막대 시각화 - 평균연체율 표시 (version2 차주 매칭기준)
# ----------------------------------------------------------
visualizer.df_setter(builder.df_차주_연체정보_집계)
전체평균연체율 = builder.df_차주_연체정보_집계['연체여부'].mean()
visualizer.bar('BTH_SECTION','연체율','연체율','연령구간','연체율 (%)','연령별 연체율 (대출기관 언매칭)',1,전체평균연체율,'전체평균')

# ----------------------------------------------------------
# 7) 정책금융 상품 한정 연체율 표시 (version3 정책금융 이용차주 한정 기준)
# ----------------------------------------------------------
visualizer.df_setter(builder.df_차주_정책금융이용_연체정보)
전체평균연체율 = builder.df_차주_정책금융이용_연체정보['연체여부'].mean()
visualizer.bar('BTH_SECTION','연체율','연체율','연령구간','연체율 (%)','연령별 연체율 (정책금융 이용한정)',1,전체평균연체율,'전체평균')
