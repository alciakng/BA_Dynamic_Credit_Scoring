from datetime import datetime
from matplotlib import ticker
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
import os

import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from common_code import CommonCode
from linearmodels.panel import PanelOLS

# 모든컬럼 확인 옵션 
pd.set_option('display.max_columns', None) 

df_차주 = pd.read_csv("./dataset/차주정보.csv")
df_기업개요정보 = pd.read_csv("./dataset/기업개요정보.csv")
df_연체정보_신용도판단정보= pd.read_csv("./dataset/연체정보(신용도판단정보).csv")
df_연체정보_공공정보 = pd.read_csv("./dataset/연체정보(공공정보).csv")
df_개인대출정보_금융권 = pd.read_csv("./dataset/개인대출정보(금융권).csv")
df_개인대출정보_대부업권 = pd.read_csv("./dataset/개인대출정보(대부업).csv")
df_신용카드개설정보 = pd.read_csv("./dataset/신용카드개설정보.csv")
df_신용카드이용정보 = pd.read_csv("./dataset/신용카드이용정보.csv")
df_채무보증정보 = pd.read_csv("./dataset/채무보증정보.csv")
df_기업대출정보 = pd.read_csv("./dataset/기업대출정보.csv")
df_기술신용평가정보 = pd.read_csv("./dataset/기술신용평가정보.csv")
df_기업개요정보기술신용대출정보 = pd.read_csv("./dataset/기술신용대출정보.csv")
df_보험계약관계자정보 = pd.read_csv("./dataset/보험계약관계자정보.csv")
df_보험계약정보 = pd.read_csv("./dataset/보험계약정보.csv")
df_보험담보정보 = pd.read_csv("./dataset/보험담보정보.csv")
df_청구사고정보 = pd.read_csv("./dataset/청구사고정보.csv")
df_청구계약정보 = pd.read_csv("./dataset/청구계약정보.csv")
df_청구지급사유정보 = pd.read_csv("./dataset/청구지급사유정보.csv")
df_청구지급상세사유정보 = pd.read_csv("./dataset/청구지급상세사유정보.csv")
df_청구피해자물정보 = pd.read_csv("./dataset/청구피해자물정보.csv")


# 한국표준산업분류(KIC) 10차 코드 
url = 'https://github.com/FinanceData/KSIC/raw/master/KSIC_10.csv.gz'
df_ksic = pd.read_csv(url, dtype='str')
df_ksic['Industy_code'] = df_ksic['Industy_code'].str.pad(width=5, side='right', fillchar='0')
df_ksic['Industy_code'] = df_ksic['Industy_code'].str[:4]
df_ksic.rename(columns={'Industy_code': 'BIZ_TYP_CD'}, inplace=True)
df_ksic.rename(columns={'Industy_name': 'BIZ_TYP_NM'}, inplace=True)
df_ksic.head(100)
df_ksic.info()

############################################################
# Section1. 기업개요정보 파악
#  0. 각 데이터 프레임 결측치 파악
#  1. 차주-기업개요정보 (1:n) 병합처리 
#  2. 개인사업자 필터링, 소상공인 필터링(유의한 변수없음.교수님께 질의필요)
#  3. 연령 구간화 (0 : ~19 1: 19~34 2: 35~50 3: 51~64 4: 65~)
#  4. 한국표준산업분류(KIC) 10차코드로 업종명 매핑(현재 모의데이터 업종코드는 비정상적)
#  5. 시각화
#    - 차주 한명당 업체개수 - 건수 시각화
#    - 업종 - 연령별 건수 막대그래프
#    - 연령대별 업종 분포 시각화 
############################################################

# ------------------------------
# 0. 결측치파악 => 결측치 존재하지 않음 
# ------------------------------
print(df_차주.info())
print(df_기업개요정보.info())

# -----------------------------
# 1. 차주-기업개요정보 (1:n) 병합처리
# -----------------------------
df_차주_기업개요정보 = df_기업개요정보.merge(df_차주, on=['JOIN_SN','JOIN_SN_TYP'], how='left')

# -------------------------------------------------------
# 2. 개인사업자 필터링, 소상공인 필터링(유의한 변수없음.교수님께 질의필요)
# -------------------------------------------------------
df_차주_기업개요정보 = df_차주_기업개요정보[df_차주_기업개요정보["JOIN_SN_TYP"] ==1]
df_차주_기업개요정보.info()


# -----------
# 3. 연령구간화 
# -----------
bins = [-1, 19, 34, 50, 64, 150]  # 각 구간의 경계값
labels = ["~19세","~34세","~50세","~64세","65세~"]          # 구간에 매길 값

df_차주_기업개요정보['BTH_SECTION'] = pd.cut(datetime.now().year - df_차주_기업개요정보['BTH_YR'], bins=bins, labels=labels)


# 차주-기업개요정보가 아닌 차주 전체를 대상으로 연령구간화 
df_차주_연령구간화 = df_차주.copy()
df_차주_연령구간화['BTH_SECTION'] = pd.cut(datetime.now().year - df_차주['BTH_YR'], bins=bins, labels=labels)

# ---------------------------------------------------
# 4. 10차 코드로 업종명 매핑 (현재 모의데이터셋 불완전하여 매핑안됨)
# ---------------------------------------------------
#df_차주_기업개요정보['BIZ_TYP_CD'] = df_차주_기업개요정보['BIZ_TYP_CD'].astype(object)
#df_차주_기업개요정보 = df_차주_기업개요정보.merge(df_ksic,on="BIZ_TYP_CD", how='left')

# ---------------------------
# 5. 시각화를 위한 Summary df 생성
# ---------------------------
df_차주_기업개요정보_Summary = df_차주_기업개요정보.groupby('JOIN_SN')['BIZ_SN'].nunique().reset_index(name='BIZ_COUNT')
df_차주_기업개요정보_Summary = df_차주_기업개요정보_Summary.groupby('BIZ_COUNT')["JOIN_SN"].count().reset_index(name='COUNT')

df_차주_기업개요정보_Summary['BIZ_COUNT'] = df_차주_기업개요정보_Summary['BIZ_COUNT'].astype(str) +"개"
df_차주_기업개요정보_Summary['BIZ_COUNT'] = df_차주_기업개요정보_Summary['BIZ_COUNT'].astype('category')


df_차주_기업개요정보_Summary.head()

#############################################################################
# Section2. 기업대출 및 연체규모 파악
#  0. 각 데이터 프레임 결측치 파악
#  1. 병합 
#    - 1-1) 금융권+대부업권 대출정보 병합 후 기업대출정보와 동일 컬럼기준으로 병합
#    - 1-2) 차주+대출정보 병합
#    - 1-3) 신용도판단정보+공공정보 연체정보 병합
#    - 1-4) 대출정보+신용카드이용정보 통합 
#    - 1-5) 대출신용카드이용정보 + 연체정보 통합 - (차주번호, 대출신용카드기관번호)를 기준으로 매칭 version1
#    - 1-6) 차주+연체정보 통합 - 기관번호로 매칭하면 연체건이 누락되는 현상으로 인해 차주와 연체율 단순 매칭 version2
#    - 1-7) 차주(정책금융 이용차주 한정)+연체정보 통합 - version3
#    - 1-8) 차주 + 대출신용카드연체정보 통합
############################################################################

################################
# 0. 결측치파악 => 결측치 존재하지 않음 
################################

#대출정보(병합대상)
print(df_개인대출정보_금융권.info())
print(df_개인대출정보_대부업권.info())
print(df_기업대출정보.info())

#연체정보(병합대상)
print(df_연체정보_신용도판단정보.info())
print(df_연체정보_공공정보.info())

print(df_신용카드개설정보.info())

# ==============================
# 1. 병합
# ==============================

# ---------------------------
# 1-1) 금융권+대부업권 대출정보 병합 
# ---------------------------
df_개인대출정보 = pd.concat([df_개인대출정보_금융권, df_개인대출정보_대부업권], ignore_index=True)
# 구분컬럼추가 
df_개인대출정보["기업개인구분"] = "개인"
df_기업대출정보["기업개인구분"] = "기업"
# 대출과목코드 일원화(1.개인대출상품코드 스트링 붙이기(LN_CD_1+LN_CD_2+LN_CD_3),2.기업대출과목코드를 개인대출코드로 바꿈)
df_개인대출정보["LN_CD_1"] = df_개인대출정보["LN_CD_1"].astype('str')
df_개인대출정보["LN_CD_2"] = df_개인대출정보["LN_CD_2"].astype('str')
df_개인대출정보["LN_CD_3"] = df_개인대출정보["LN_CD_3"].astype('str')
df_기업대출정보.rename(columns={'LN_ACCT_CD': 'LN_CD_1'}, inplace=True)

# 기업대출 default 변수 
df_기업대출정보['LN_DT'] = 0    
df_기업대출정보['EXP_DT'] = 0
df_기업대출정보['RATE'] = 0
df_기업대출정보['LN_CD_2'] = None
df_기업대출정보['LN_CD_3'] = None

# 겹치는 컬럼 파악 
common_cols = list(df_개인대출정보.columns.intersection(df_기업대출정보.columns))
# 해당 컬럼기준으로 개인대출정보+기업대출정보 병합 
df_대출정보 = pd.concat([df_개인대출정보[common_cols], df_기업대출정보[common_cols]], ignore_index=True)

print(df_대출정보.info())
# 자료의 범위
print(df_대출정보['YM'].min(),df_대출정보['YM'].max())

# ---------------------------
# 1-2) 차주+대출집계정보 병합
# ---------------------------

# 대출정보_집계
df_대출정보_보유기관_집계 = df_대출정보.groupby(['JOIN_SN','YM'])['COM_SN'].count().reset_index(name='COM_SN_COUNT')
df_대출정보_대출과목_집계 = df_대출정보.groupby(['JOIN_SN','YM'])['LN_CD_1'].count().reset_index(name='LN_CD_COUNT')

# 차주의 월별 집계치 중 min(), max() 건수를 기준으로 새로운 데이터 프레임 생성 
df_대출정보_보유기관_집계=df_대출정보_보유기관_집계.groupby('JOIN_SN')['COM_SN_COUNT'].agg(최소보유건수='min', 최대보유건수='max').reset_index()
df_대출정보_대출과목_집계=df_대출정보_대출과목_집계.groupby('JOIN_SN')['LN_CD_COUNT'].agg(최소대출건수='min', 최대대출건수='max').reset_index()

# 집계정보통합 
df_대출정보_집계 = pd.merge(df_대출정보_보유기관_집계, df_대출정보_대출과목_집계, on='JOIN_SN', how='inner')
df_대출정보_집계.head(100)

df_차주_기업개요정보.head(100)

# 차주+집계정보 병합
df_차주_대출정보_집계 = df_차주.merge(df_대출정보_집계,on='JOIN_SN',how='left')


df_차주_대출정보_집계[['최소보유건수','최대보유건수','최소대출건수','최대대출건수']] = df_차주_대출정보_집계[['최소보유건수','최대보유건수','최소대출건수','최대대출건수']].fillna(0)
####################차주+대출정보 집계 테이블 완성#######################

# ---------------------------
# 1-3) 신용도판단정보+공공정보 병합 
# ---------------------------

df_연체정보_신용도판단정보["민간공공구분"] = "민간"
df_연체정보_공공정보["민간공공구분"] ="공공"
# 연체정보 통합 
df_연체정보 = pd.concat([df_연체정보_신용도판단정보, df_연체정보_공공정보], ignore_index=True)


df_연체정보[['JOIN_SN','COM_SN']].value_counts()
df_연체정보[df_연체정보['JOIN_SN']==1545255]


# 차주, 기관별 마지막 등록월의 연체정보만을 가져온다. (기관까지 매칭하는 버전)
df_연체정보_기관별_최종 = (
    df_연체정보
    #[df_연체정보['DLQ_RGST_AMT'] > 0]  # 연체금액 있는 행만
    .loc[lambda x: x.groupby(['JOIN_SN','COM_SN'])['YM'].idxmax()]    # 차주별 가장 최근 YM
    .reset_index(drop=True)
)

# 차주 마지막 등록월의 연체정보만을 가져온다. (simple 버전)
df_연체정보_최종 = (
    df_연체정보.sort_values(['YM'],ascending=True)  # 월 오름차순 정렬
      .groupby(['JOIN_SN'], as_index=False)  # 그룹핑
      .tail(1)  # 각 그룹에서 마지막 row (최신 월)
)

df_연체정보_최종.info()


# --------------------------------------------------------------------
# 1-4) 대출정보+신용카드이용정보 통합
# --------------------------------------------------------------------
df_대출정보.info()
df_신용카드이용정보.info()

df_신용카드이용정보

# 대출정보와 신용카드 이용정보의 금액을 같은이름으로 통일 
df_대출정보_AMT= df_대출정보.rename(columns={'LN_AMT': 'AMT'})
df_신용카드이용정보_AMT= df_신용카드이용정보.copy()
df_신용카드이용정보_AMT['AMT'] = df_신용카드이용정보['CD_USG_AMT']+ df_신용카드이용정보['CD_CA_AMT'] # 신용카드 이용금액+신용카드론 이용금액 합산

# 구분컬럼 생성 
df_대출정보_AMT['대출신용카드구분'] = '대출'
df_신용카드이용정보_AMT['대출신용카드구분'] = '신용'

df_대출정보_AMT[df_대출정보_AMT['COM_SN']==8816379]
df_신용카드이용정보_AMT[df_신용카드이용정보_AMT['COM_SN']==8816379]


# 겹치는 컬럼 파악 
common_cols = list(df_대출정보_AMT.columns.intersection(df_신용카드이용정보_AMT.columns))

# 해당 컬럼기준으로 대출정보+신용카드 이용정보 병합 
df_대출_신용카드_병합 = pd.concat([df_대출정보_AMT[common_cols], df_신용카드이용정보_AMT[common_cols]], ignore_index=True)

# 기관별 마지막 월을 기준으로 차주의 대출,신용카드 금액만을 남긴다.
df_대출_신용카드_병합_기관별_최종  = (
    df_대출_신용카드_병합.sort_values(['YM'],ascending=True)  # 월 오름차순 정렬
      .groupby(['JOIN_SN', 'COM_SN', '대출신용카드구분'], as_index=False)  # 그룹핑
      .tail(1)  # 각 그룹에서 마지막 row (최신 월)
)

# ---------------------------------------------------------------------------------
# 1-5) 대출신용카드이용정보 + 연체정보 통합 - (차주번호, 대출신용카드기관번호)를 기준으로 매칭 version1
# ---------------------------------------------------------------------------------
df_대출_신용카드_병합_기관별_최종.info()
df_연체정보_기관별_최종.info()

# (주의) 대출기관=신용카드 기관 같은경우 (카드대출을 받은경우는 중복으로 행이 두개가 들어가므로, 기관이 같은 경우에는 중복행을 대출만 남기고 신용을 제거한다.)
# 이를 통해 중복기관을 없애서 정확한 기관별 연체여부를 파악한다.
df_대출_신용카드_병합_기관별_최종_sorted = df_대출_신용카드_병합_기관별_최종.sort_values(['JOIN_SN','JOIN_SN_TYP','COM_SN','대출신용카드구분']) 
df_대출_신용카드_병합_기관별_최종 = df_대출_신용카드_병합_기관별_최종_sorted.groupby(['JOIN_SN','JOIN_SN_TYP','COM_SN'], as_index=False).head(1)

df_대출_신용카드_병합_기관별_최종.info()

# 연체정보 2,630 건이 전체 매칭되지 않는다. (연체등록은 대출시점 이후에 발생하므로, 2018.06~2020.06 같은기간으로 보면 연체정보에 있는 대출기관이 대출신용카드정보에서 누락될 수 있기때문으로 보임)
df_대출_신용카드_연체정보 = pd.merge(df_대출_신용카드_병합_기관별_최종,df_연체정보_기관별_최종[['JOIN_SN','JOIN_SN_TYP','COM_SN','DLQ_RGST_DT','DLQ_RGST_AMT','DLQ_AMT']],on=['JOIN_SN','JOIN_SN_TYP','COM_SN'],how='left')
df_대출_신용카드_연체정보.info()

# 누락된 연체정보 파악 
filter_keys = df_대출_신용카드_연체정보[~df_대출_신용카드_연체정보['DLQ_RGST_DT'].isna()][['JOIN_SN','COM_SN']].apply(tuple,axis=1)

df_연체정보_기관별_최종[df_연체정보_기관별_최종[['JOIN_SN','COM_SN']].apply(tuple,axis=1).isin(filter_keys)]  # 포함 연체정보 
df_연체정보_기관별_최종[~df_연체정보_기관별_최종[['JOIN_SN','COM_SN']].apply(tuple,axis=1).isin(filter_keys)] # 누락 연체정보 
## ex) 808 차주의 5612646(COM_SN) 정보의 경우, df_대출_신용카드_병합_기관별_최종에 포함되어있지 않음 
df_대출정보[df_대출정보['JOIN_SN']== 808]
df_신용카드이용정보[df_신용카드이용정보['JOIN_SN'] == 808]
df_대출_신용카드_병합_기관별_최종[df_대출_신용카드_병합_기관별_최종['JOIN_SN']== 808]

# 중복정보 파악 - 중복정보 없음 
df_대출_신용카드_연체정보[df_대출_신용카드_연체정보.groupby(['YM','JOIN_SN','JOIN_SN_TYP','COM_SN'])['JOIN_SN'].transform('count') > 1]


# -----------------------------------------------------------------------------------------------------------------------
# 1-6) 차주+연체정보 통합 - 기관번호로 매칭하면 연체건이 누락되는 현상으로 인해 차주와 연체율 단순 매칭 version2
# -----------------------------------------------------------------------------------------------------------------------
df_차주_연령구간화.info()

df_차주_연체정보 = pd.merge(df_차주_연령구간화,df_연체정보_최종[['JOIN_SN','JOIN_SN_TYP','COM_SN','DLQ_RGST_DT','DLQ_RGST_AMT','DLQ_AMT']],on=['JOIN_SN','JOIN_SN_TYP'],how='left')
df_차주_연체정보.info()

# -----------------------------------------------------------------------------------------------------------------------
# 1-7) 차주(정책금융 이용차주 한정)+연체정보 통합 - version3
# -----------------------------------------------------------------------------------------------------------------------
df_차주_연령구간화.info()

df_차주_연령구간화_정책금융이용= df_차주_연령구간화[df_차주_연령구간화['JOIN_SN'].isin(df_개인대출정보[df_개인대출정보['LN_CD_3']!='0']['JOIN_SN'])]

df_차주_정책금융이용_연체정보 = pd.merge(df_차주_연령구간화_정책금융이용,df_연체정보_최종[['JOIN_SN','JOIN_SN_TYP','COM_SN','DLQ_RGST_DT','DLQ_RGST_AMT','DLQ_AMT']],on=['JOIN_SN','JOIN_SN_TYP'],how='left')
df_차주_정책금융이용_연체정보.info()

# 정책금융 비이용 차주
df_차주_연령구간화_정책금융비이용= df_차주_연령구간화[df_차주_연령구간화['JOIN_SN'].isin(df_개인대출정보[df_개인대출정보['LN_CD_3']=='0']['JOIN_SN'])]

df_차주_정책금융비이용_연체정보 = pd.merge(df_차주_연령구간화_정책금융비이용,df_연체정보_최종[['JOIN_SN','JOIN_SN_TYP','COM_SN','DLQ_RGST_DT','DLQ_RGST_AMT','DLQ_AMT']],on=['JOIN_SN','JOIN_SN_TYP'],how='left')
df_차주_정책금융비이용_연체정보.info()

df_차주_정책금융비이용_연체정보



# --------------------------------------------------------------------
# 1-8) 차주 + 대출신용카드연체정보 통합
# --------------------------------------------------------------------
df_차주_대출_신용카드_연체정보 = pd.merge(df_차주_연령구간화, df_대출_신용카드_연체정보,on=['JOIN_SN','JOIN_SN_TYP'],how='left')

df_차주_대출_신용카드_연체정보.info()

df_차주_대출_신용카드_연체정보[['YM','SCTR_CD','COM_SN','IS_ME','AMT','DLQ_RGST_AMT','DLQ_AMT']] = df_차주_대출_신용카드_연체정보[['YM','SCTR_CD','COM_SN','IS_ME','AMT','DLQ_RGST_AMT','DLQ_AMT']].fillna(0)
df_차주_대출_신용카드_연체정보['대출신용카드구분'].fillna('미보유',inplace=True)



#############################################################################
# Section3. 분석을 위한 독립변수, 종속변수 설정 
# 1) 대출정보 독립변수 생성 (차주별 연체이전 대출개수, 대출금액, 장기고액대출건수)
# 2) 지급능력 독립변수 생성 (차주별 지급능력점수 신용카드이용한도와 현금서비스 한도를 가중평균) 
#   - 지급능력 점수 = 신용카드이용한도 * 0.8 + 현금서비스한도 * 0.2
# 3) 보험정보 독립변수 생성 (차주별 보험가입건수, 월 납입보험료, 청구사고건수)
# 4) 연체정보 독립변수 생성 (연체건수, 장기연체건수, 연체금액, 과거도산경험)
# 5) 신용카드정보 독립변수 생성 (신용카드 개수)
# 6) 소비성향 독립변수 생성 (신용카드 한도대비 사용금액)
#############################################################################

# --------------------------------------------------------------------
# 1) 대출정보 독립변수 생성 (차주별 연체이전 대출개수, 대출금액, 장기고액대출건수)
# --------------------------------------------------------------------
df_대출정보_최종  = (
    df_대출정보.sort_values(['YM'],ascending=True)  # 월 오름차순 정렬
      .groupby(['JOIN_SN', 'COM_SN', 'LN_DT','LN_CD_1','LN_CD_2','LN_CD_3'], as_index=False)  # 그룹핑
      .tail(1)  # 각 그룹에서 마지막 row (최신 월)
)

# 대출정보와 연체정보를 병합
df_연체정보_대출정보_최종 = pd.merge(df_연체정보_최종, df_대출정보_최종, on=["JOIN_SN","JOIN_SN_TYP"], how="inner")
df_연체정보_대출정보_최종['YM_y'] = df_연체정보_대출정보_최종['YM_y'].fillna(df_연체정보_대출정보_최종['YM_x'])

df_연체정보_대출정보_최종.info()

# 연체이전 대출만 필터링 
df_연체정보_대출정보_최종_필터링 = df_연체정보_대출정보_최종[df_연체정보_대출정보_최종["YM_y"] <= df_연체정보_대출정보_최종["YM_x"]]

# 날짜변환
df_연체정보_대출정보_최종_필터링["YM_DT"] = pd.to_datetime(df_연체정보_대출정보_최종_필터링["YM_y"].astype(int).astype(str), format="%Y%m")
df_연체정보_대출정보_최종_필터링["LN_DT"] = pd.to_datetime(df_연체정보_대출정보_최종_필터링["LN_DT"].astype(str), format="%Y%m")

# 대출기간 계산 
df_연체정보_대출정보_최종_필터링["대출기간"] = (
    (df_연체정보_대출정보_최종_필터링["YM_DT"].dt.year - df_연체정보_대출정보_최종_필터링["LN_DT"].dt.year) * 12 +
    (df_연체정보_대출정보_최종_필터링["YM_DT"].dt.month - df_연체정보_대출정보_최종_필터링["LN_DT"].dt.month)
)

# 장기고액대출조건 
df_연체정보_대출정보_최종_필터링["장기고액"] = (
    (df_연체정보_대출정보_최종_필터링["대출기간"] >= 36) & 
    (df_연체정보_대출정보_최종_필터링["LN_AMT"] >= 100000)
)

# 차주별 집계
df_대출정보_독립변수_집계 = df_연체정보_대출정보_최종_필터링.groupby(["JOIN_SN","JOIN_SN_TYP"]).agg(
    대출건수=("YM_x", "count"),
    장기고액대출건수=("장기고액", "sum"),
    대출금액합=("LN_AMT", "sum")
).reset_index()

# ----------------------------------------------------------------------
# 2) 지급능력 독립변수 설정 (지급능력 점수 = 신용카드이용한도*0.85 + 현금서비스한도*0.15)
# ----------------------------------------------------------------------
df_신용카드이용정보_최종  = (
    df_신용카드이용정보.sort_values(['YM'],ascending=True)  # 월 오름차순 정렬
      .groupby(['JOIN_SN'], as_index=False)  # 그룹핑
      .tail(1)  # 각 그룹에서 마지막 row (최신 월)
)

df_신용카드이용정보_최종['지급능력'] = df_신용카드이용정보_최종['CD_USG_LMT']*0.85 + df_신용카드이용정보_최종['CD_CA_LMT']*0.15

# 신용카드이용정보와 연체정보를 병합
df_신용카드이용정보_연체정보_최종 = pd.merge(df_신용카드이용정보_최종, df_연체정보_최종, on=["JOIN_SN","JOIN_SN_TYP"], how="inner")
df_신용카드이용정보_연체정보_최종['YM_y'] = df_신용카드이용정보_연체정보_최종['YM_y'].fillna(df_신용카드이용정보_연체정보_최종['YM_x'])

# 연체이전 대출만 필터링 
df_신용카드이용정보_연체정보_최종_필터링 = df_신용카드이용정보_연체정보_최종[df_신용카드이용정보_연체정보_최종["YM_x"] <= df_신용카드이용정보_연체정보_최종["YM_y"]]

# 신용카드 독립변수 집계 (지급능력 )
df_지급능력_독립변수_집계 = df_신용카드이용정보_연체정보_최종_필터링.groupby(["JOIN_SN","JOIN_SN_TYP"]).agg(
    지급능력=("지급능력", "max")
).reset_index()

df_지급능력_독립변수_집계.info()

# ----------------------------------------------------------------------
# 3) 보험정보 독립변수 설정 (차주별 보험가입건수, 월 납입보험료, 청구사고건수)
# ----------------------------------------------------------------------
# 보험계약관계자정보와 보험계약정보를 병합
df_보험계약정보_병합 = pd.merge(df_보험계약관계자정보, df_보험계약정보, on=["SCTR_CD","POL_SN"], how="left")

# 보험계약정보와 연체정보를 병합
df_보험계약정보_연체정보_최종 = pd.merge(df_보험계약정보_병합, df_연체정보_최종, on=["JOIN_SN","JOIN_SN_TYP"], how="inner")

df_보험계약정보_연체정보_최종.info()

# 연체이전 보험계약유효건만 필터링 
df_보험계약정보_연체정보_최종 = df_보험계약정보_연체정보_최종[(df_보험계약정보_연체정보_최종["CT_CNCLS_DT"] < df_보험계약정보_연체정보_최종["YM"]) & (df_보험계약정보_연체정보_최종["CT_TRMNT_DT"] >= df_보험계약정보_연체정보_최종["YM"])]
df_보험계약정보_연체정보_최종 = df_보험계약정보_연체정보_최종[(df_보험계약정보_연체정보_최종["CT_ST_DT"] < df_보험계약정보_연체정보_최종["YM"]) & (df_보험계약정보_연체정보_최종["CT_END_DT"] >= df_보험계약정보_연체정보_최종["YM"])]

# 연체월에 유효한 보험 계약건만 필터링 
def is_active(row):
    col = "YM_" + str(int(row["YM"]))
    return row.get(col, 0) == 1
df_보험계약정보_연체정보_최종 = df_보험계약정보_연체정보_최종[df_보험계약정보_연체정보_최종.apply(is_active,axis=1)]

# 중복된 보험 제거 
df_보험계약정보_연체정보_최종  = (
    df_보험계약정보_연체정보_최종.sort_values(['CT_CNCLS_DT'],ascending=False)  # 월 오름차순 정렬
      .groupby(['JOIN_SN','POL_SN'], as_index=False)  # 그룹핑
      .tail(1)  # 각 그룹에서 마지막 row (최신 월)
)

# 월납입액 계산 함수
def calculate_monthly(row):
    if row["CT_PY_CYCLE_CD"] == 1: # 일시납 
        return 0
    elif row["CT_PY_CYCLE_CD"] == 2: # 월납 
        return row["CT_PY_AMT"]
    elif row["CT_PY_CYCLE_CD"] == 3: # 연납 
        return row["CT_PY_AMT"]/12
    else:
        return None

# 월납입액 계산
df_보험계약정보_연체정보_최종["월_보험료납입액"] = df_보험계약정보_연체정보_최종.apply(calculate_monthly, axis=1)

# 차주별 집계
df_보험계약정보_독립변수_집계 = df_보험계약정보_연체정보_최종.groupby(["JOIN_SN","JOIN_SN_TYP"]).agg(
    보험건수=("POL_SN", "count"),
    보험월납입액=("월_보험료납입액", "sum")
).reset_index()

# ----------------------------------------------------------------------
# 4) 연체정보 독립변수 생성 (연체건수, 장기연체건수, 연체금액, 과거도산경험)
# ----------------------------------------------------------------------
df_연체정보_기관별_연체날짜별_집계  = (
    df_연체정보.sort_values(['YM'],ascending=True)  # 월 오름차순 정렬
      .groupby(['JOIN_SN','COM_SN','DLQ_RGST_DT'], as_index=False)  # 그룹핑
      .tail(1)  # 각 그룹에서 마지막 row (최신 월)
)

df_연체정보_기관별_연체날짜별_집계[df_연체정보_기관별_연체날짜별_집계['JOIN_SN'] == 1714896]


# 연체집계정보와 최종연체정보를 병합
df_연체집계_연체최종정보 = pd.merge(df_연체정보_기관별_연체날짜별_집계, df_연체정보_최종, on=["JOIN_SN","JOIN_SN_TYP"], how="inner")
df_연체집계_연체최종정보['YM_y'] = df_연체집계_연체최종정보['YM_y'].fillna(df_연체집계_연체최종정보['YM_x'])

# 최종연체이전 연체건만 필터링 
df_연체집계_연체최종정보_필터링 = df_연체집계_연체최종정보[df_연체집계_연체최종정보["YM_x"] <= df_연체집계_연체최종정보["YM_y"]]

# 날짜변환
df_연체집계_연체최종정보_필터링["YM_DT"] = pd.to_datetime(df_연체집계_연체최종정보_필터링["YM_x"].astype(str), format="%Y%m")
df_연체집계_연체최종정보_필터링["DLQ_RGST_DT_x"] = pd.to_datetime(df_연체집계_연체최종정보_필터링["DLQ_RGST_DT_x"].astype(str), format="%Y%m")

# 대출기간 계산 
df_연체집계_연체최종정보_필터링["연체기간"] = (
    (df_연체집계_연체최종정보_필터링["YM_DT"].dt.year - df_연체집계_연체최종정보_필터링["DLQ_RGST_DT_x"].dt.year) * 12 +
    (df_연체집계_연체최종정보_필터링["YM_DT"].dt.month - df_연체집계_연체최종정보_필터링["DLQ_RGST_DT_x"].dt.month)
)

# 장기연체건
df_연체집계_연체최종정보_필터링["장기연체"] = (
    (df_연체집계_연체최종정보_필터링["연체기간"] >= 12)
)

# 차주별 집계
df_연체정보_독립변수_집계 = df_연체집계_연체최종정보_필터링.groupby(["JOIN_SN","JOIN_SN_TYP"]).agg(
    연체건수=("YM_x", "count"),
    장기연체건수=("장기연체", "sum"),
    연체금액합=("DLQ_AMT_x", "sum")
).reset_index()

## TODO ## 나중에 본 데이터에서는 COM_SN 기관매칭으로 하되, 정책자금대출 연체등록월을 기준으로 독립변수들을 생성해야 함.
## TODO ## 매칭데이터(비정책자금) 은 같은 연체등록월을 가지고 지급능력이 비슷한 차주와 금리가 비슷한 대출데이터를 매칭해야 함.

# ----------------------------------------------------------------------
# 5) 신용카드정보 독립변수 생성 (신용카드 개수)
# ----------------------------------------------------------------------
df_신용카드개설정보_최종  = (
    df_연체정보.sort_values(['YM'],ascending=True)  # 월 오름차순 정렬
      .groupby(['JOIN_SN',"JOIN_SN_TYP", 'COM_SN'], as_index=False)  # 그룹핑
      .tail(1)  # 각 그룹에서 마지막 row (최신 월)
)

df_신용카드정보_독립변수_집계 = df_신용카드개설정보_최종.groupby(["JOIN_SN","JOIN_SN_TYP"]).agg(
    신용카드개수=("COM_SN", "count")
).reset_index()

# -------------------------------------------------------------------------------------------------------------
# 6) 소비성향 독립변수 생성 (신용카드 한도대비 사용금액) - 연체이전 3개월 사용률(1), 연체이전 9개월전~3개월전 사용률(2), (2)->(1) 증가율 
# ------------------------------------------------------------------------------------------------------------
df_신용카드이용정보_소비성향 = df_신용카드이용정보.copy()

df_신용카드이용정보_소비성향['신용카드한도대비사용률'] = np.where(df_신용카드이용정보_소비성향['CD_USG_LMT'] == 0, 0, df_신용카드이용정보_소비성향['CD_USG_AMT']/df_신용카드이용정보_소비성향['CD_USG_LMT'])
df_신용카드이용정보_소비성향['현금서비스한도대비사용률'] = np.where(df_신용카드이용정보_소비성향['CD_CA_LMT'] == 0, 0, df_신용카드이용정보_소비성향['CD_CA_AMT']/df_신용카드이용정보_소비성향['CD_CA_LMT'])

#====가중평균 사용률 계산====# 
# 차주별 총 사용금액 (월별)
df_신용카드이용정보_소비성향["신용카드총사용금액"] = df_신용카드이용정보_소비성향.groupby(["JOIN_SN",'YM'])["CD_USG_AMT"].transform("sum")
df_신용카드이용정보_소비성향["현금서비스총사용금액"] = df_신용카드이용정보_소비성향.groupby(["JOIN_SN",'YM'])["CD_CA_AMT"].transform("sum")

# 가중치 = 사용금액 비율 
df_신용카드이용정보_소비성향['신용카드사용비중'] = np.where(df_신용카드이용정보_소비성향['신용카드총사용금액'] == 0, 0, df_신용카드이용정보_소비성향["CD_USG_AMT"] / df_신용카드이용정보_소비성향['신용카드총사용금액'])
df_신용카드이용정보_소비성향['현금서비스사용비중'] = np.where(df_신용카드이용정보_소비성향['현금서비스총사용금액'] == 0, 0, df_신용카드이용정보_소비성향["CD_CA_AMT"] / df_신용카드이용정보_소비성향['현금서비스총사용금액'])

#가중평균 사용률
df_신용카드이용정보_소비성향['신용카드가중평균사용률'] = np.where(df_신용카드이용정보_소비성향["신용카드한도대비사용률"] ==0, 0, df_신용카드이용정보_소비성향["신용카드한도대비사용률"] * df_신용카드이용정보_소비성향['신용카드사용비중'])
df_신용카드이용정보_소비성향['현금서비스가중평균사용률'] = np.where(df_신용카드이용정보_소비성향["현금서비스한도대비사용률"] ==0, 0, df_신용카드이용정보_소비성향["현금서비스한도대비사용률"] * df_신용카드이용정보_소비성향['현금서비스사용비중'])

df_신용카드이용정보_소비성향.info()

#======================
df_연체정보_소비성향 = df_연체정보_최종.copy()

# 날짜형으로 변환
df_신용카드이용정보_소비성향["이용월"] = pd.to_datetime(df_신용카드이용정보_소비성향["YM"], format="%Y%m")
df_연체정보_소비성향["최종연체월"] = pd.to_datetime(df_연체정보_소비성향["YM"], format="%Y%m")

# merge: 연체월 정보 추가
df_신용카드이용정보_연체정보_소비성향 = pd.merge(df_신용카드이용정보_소비성향, df_연체정보_소비성향, on=(["JOIN_SN","JOIN_SN_TYP"]), how="inner")

# 개월 수 차이 구하기
df_신용카드이용정보_연체정보_소비성향["개월차"] = (
    (df_신용카드이용정보_연체정보_소비성향["최종연체월"].dt.year - df_신용카드이용정보_연체정보_소비성향["이용월"].dt.year) * 12 +
    (df_신용카드이용정보_연체정보_소비성향["최종연체월"].dt.month - df_신용카드이용정보_연체정보_소비성향["이용월"].dt.month)
)

df_신용카드이용정보_연체정보_소비성향[df_신용카드이용정보_연체정보_소비성향['JOIN_SN']==581334][['이용월','최종연체월','개월차']]

# 최근 3개월: 개월차 1,2,3
cond_recent = df_신용카드이용정보_연체정보_소비성향["개월차"].between(1, 3)
# 3~6개월 전: 개월차 4,5,6
cond_past = df_신용카드이용정보_연체정보_소비성향["개월차"].between(4, 9)

df_신용카드이용정보_연체정보_소비성향[df_신용카드이용정보_연체정보_소비성향['JOIN_SN']==581334].to_csv()

df_temp =df_신용카드이용정보_연체정보_소비성향[cond_recent]

# 1. 최근 3개월 평균 사용률
recent_usg_avg = df_신용카드이용정보_연체정보_소비성향[(cond_recent) & (df_신용카드이용정보_연체정보_소비성향['신용카드가중평균사용률']) !=0].groupby(["JOIN_SN","JOIN_SN_TYP"])['신용카드가중평균사용률'].mean().reset_index(name="최근3개월_신용카드사용률")
recent_ca_avg = df_신용카드이용정보_연체정보_소비성향[(cond_recent) & (df_신용카드이용정보_연체정보_소비성향['현금서비스가중평균사용률']) !=0].groupby(["JOIN_SN","JOIN_SN_TYP"])['현금서비스가중평균사용률'].mean().reset_index(name="최근3개월_현금서비스사용률")

# 2. 3~6개월 전 평균 사용률
past_usg_avg = df_신용카드이용정보_연체정보_소비성향[(cond_past) & (df_신용카드이용정보_연체정보_소비성향['신용카드가중평균사용률']) !=0].groupby(["JOIN_SN","JOIN_SN_TYP"])['신용카드가중평균사용률'].mean().reset_index(name="9개월_3개월이전_신용카드사용률")
past_ca_avg = df_신용카드이용정보_연체정보_소비성향[(cond_past) & (df_신용카드이용정보_연체정보_소비성향['현금서비스가중평균사용률']) !=0].groupby(["JOIN_SN","JOIN_SN_TYP"])['현금서비스가중평균사용률'].mean().reset_index(name="9개월_3개월이전_현금서비스사용률")

# 3. 합치기
result_usg = pd.merge(recent_usg_avg, past_usg_avg, on=(["JOIN_SN","JOIN_SN_TYP"]), how="outer")
result_ca = pd.merge(recent_ca_avg, past_ca_avg, on=(["JOIN_SN","JOIN_SN_TYP"]), how="outer")

# 4. 증가분 계산 
result_usg['신용카드_사용률_증가량'] = result_usg["최근3개월_신용카드사용률"] - result_usg["9개월_3개월이전_신용카드사용률"]
result_ca['현금서비스_사용률_증가량'] = result_ca["최근3개월_현금서비스사용률"] - result_ca["9개월_3개월이전_현금서비스사용률"]

df_소비성향_독립변수_집계 = pd.merge(result_usg,result_ca,on=(["JOIN_SN","JOIN_SN_TYP"]), how="outer")
df_소비성향_독립변수_집계.fillna(0,inplace=True)


#############################################################################
# Section4. DID 분석
# 1) 집단분리 
# 2) Parallel Trend Assumption 분석
# 3) DID 분석
#############################################################################

# -------------------------------------------------------------------------------------------------------------
# 1) 집단분리
# -------------------------------------------------------------------------------------------------------------

# DID 분석을 위한 코드(정책자금여부 판별을 위해 LN_CD_3 이 있는 차주에 한정한다.)
df_대출정보_정책금융_이후 = df_대출정보[(~df_대출정보['LN_CD_3'].isna()) & (~df_대출정보['LN_CD_3'].isin(['0','900'])) & (df_대출정보['LN_DT'] >= 201806)].copy()
df_대출정보_일반신용대출 = df_대출정보[(~df_대출정보['LN_CD_3'].isna()) & (df_대출정보['LN_CD_2'].isin(['100'])) & (df_대출정보['LN_CD_3'].isin(['0']))].copy()

# 차주별 최초 정책자금 LN_DT
df_최초정책 = df_대출정보_정책금융_이후.groupby('JOIN_SN')['LN_DT'].min().reset_index()
df_최초정책.rename(columns={'LN_DT': '정책기준일'}, inplace=True)

# 정책자금 차주의 정책자금 이전데이터만 식별하기 위한 merge 작업
df_조인_1 = df_대출정보_일반신용대출.merge(df_최초정책, on='JOIN_SN', how='inner')
df_조인_1['LN_DT'] = df_조인_1['정책기준일']

# 정책자금 이전의 데이터만 식별
df_대출정보_정책금융_이전 = df_조인_1[df_조인_1['YM'] < df_조인_1['LN_DT']]
df_대출정보_정책금융_이전.drop(columns='정책기준일',inplace=True)
df_대출정보_정책금융_이전.info()

# 정책금융 자금 병합
df_대출정보_정책금융 = pd.concat([df_대출정보_정책금융_이전,df_대출정보_정책금융_이후], ignore_index=True)
df_대출정보_정책금융.info()

# 정책금융 차주가 아닌 일반 신용대출 데이터만 식별 & 정책금융 대출일자에 받은 일반자금 데이터만 식별
df_대출정보_일반신용대출_이후 = df_대출정보_일반신용대출[(~df_대출정보_일반신용대출['JOIN_SN'].isin(df_최초정책['JOIN_SN'])) & (df_대출정보_일반신용대출['LN_DT'].isin(df_대출정보_정책금융['LN_DT']))]

# 차주별 최초 일반자금LN_DT
df_최초일반자금 = df_대출정보_일반신용대출_이후.groupby('JOIN_SN')['LN_DT'].min().reset_index()
df_최초일반자금.rename(columns={'LN_DT': '대출기준일'}, inplace=True)

# 일반자금 차주의 일반자금 이전데이터만 식별하기 위한 merge 작업
df_조인_2 = df_대출정보_일반신용대출.merge(df_최초일반자금, on='JOIN_SN', how='inner')
df_조인_2['LN_DT'] = df_조인_2['대출기준일']

df_조인_2.info()

# 대출기준일 이전의 데이터만 식별
df_대출정보_일반신용대출_이전 = df_조인_2[df_조인_2['YM'] < df_조인_2['LN_DT']]
df_대출정보_일반신용대출_이전.drop(columns='대출기준일',inplace=True)
df_대출정보_일반신용대출_이전.info()

# 정책금융 자금 병합
df_대출정보_일반자금 = pd.concat([df_대출정보_일반신용대출_이전,df_대출정보_일반신용대출_이후], ignore_index=True)
df_대출정보_일반자금.info()

df_대출정보_일반자금_sampling = df_대출정보_일반자금.sample(n=6000, random_state=42)

# 정책금융여부 flag
df_대출정보_정책금융['정책금융여부'] = 1
df_대출정보_일반자금_sampling['정책금융여부'] = 0

# 병합 
df_대출정보_DID = pd.concat([df_대출정보_정책금융,df_대출정보_일반자금_sampling], ignore_index=True)

# 연체율 정책자금(7.5%), 일반자금(4.5%) 수준으로 무작위배정 (현재 모의데이터상 대출과 연체가 조인되지 않은 한계로 인함)
np.random.seed(42)

before_policy = df_대출정보_DID[(df_대출정보_DID['정책금융여부'] == 1) & (df_대출정보_DID['YM'] < df_대출정보_DID['LN_DT'])].index
after_policy = df_대출정보_DID[(df_대출정보_DID['정책금융여부'] == 1) & (df_대출정보_DID['YM'] >= df_대출정보_DID['LN_DT'])].index

before_control = df_대출정보_DID[(df_대출정보_DID['정책금융여부'] == 0) & (df_대출정보_DID['YM'] < df_대출정보_DID['LN_DT'])].index
after_control = df_대출정보_DID[(df_대출정보_DID['정책금융여부'] == 0) & (df_대출정보_DID['YM'] >= df_대출정보_DID['LN_DT'])].index

df_대출정보_DID.loc[before_policy, 'DLQ_YN'] = (np.random.rand(before_policy.size) < 0.040).astype(int)
df_대출정보_DID.loc[after_policy, 'DLQ_YN'] = (np.random.rand(after_policy.size) < 0.10).astype(int)

df_대출정보_DID.loc[before_control, 'DLQ_YN'] = (np.random.rand(before_control.size) < 0.042).astype(int)
df_대출정보_DID.loc[after_control, 'DLQ_YN'] = (np.random.rand(after_control.size) < 0.048).astype(int)

print("정책자금집단 미수령자 연체율:", df_대출정보_DID.loc[before_policy, 'DLQ_YN'].mean())
print("정책자금집단 수령자 연체율:", df_대출정보_DID.loc[after_policy, 'DLQ_YN'].mean())
print("일반자금집단 미수령자 연체율:", df_대출정보_DID.loc[before_control, 'DLQ_YN'].mean())
print("일반자금 수령자 연체율:", df_대출정보_DID.loc[after_control, 'DLQ_YN'].mean())

# post indicator: 정책자금 받은 경우, post = 1 if YM >=LN_DT
df_대출정보_DID['post'] =0
is_policy = df_대출정보_DID['정책금융여부'] ==1
df_대출정보_DID.loc[is_policy, 'post'] = (df_대출정보_DID.loc[is_policy, 'YM'] >= df_대출정보_DID.loc[is_policy, 'LN_DT']).astype(int)

df_대출정보_DID['interaction_term'] =  df_대출정보_DID['post']*df_대출정보_DID['정책금융여부']

df_대출정보_DID[df_대출정보_DID['interaction_term'] ==1]
df_대출정보_DID[df_대출정보_DID['YM'] >= df_대출정보_DID['LN_DT']][['post','정책금융여부']].value_counts()

# ==================================================================
# 2) Parallel Trend Assumption with Event Study 
# - 회귀식 : DLQ_YNit(연체율) = β0+β⋅trend(it)+α(i)+γ(t)+ϵ(it)
# - event time 변수 : 년월(YM) - 대출년월(LN_DT)
# - 이벤트 더미 변수(d_{k}) : -3개월~+3개월
# - α(i) : 개체(차주별) 고정효과
# - γ(t) : 시점(년월) 고정효과 
# ==================================================================

# 월단위로 변환
def ym_to_months(ym):
    ym = ym.astype(int)
    year = ym // 100
    month = ym % 100
    return year * 12 + month

# event time 계산
df_대출정보_DID['event_time'] = ym_to_months(df_대출정보_DID['YM'])- ym_to_months(df_대출정보_DID['LN_DT'])

# 정책 대출 실행일 이전 pre 데이터만 뽑아냄
df_대출정보_DID_pre = df_대출정보_DID[df_대출정보_DID['YM']<=df_대출정보_DID['LN_DT']]

# 이벤트 윈도우 설정: -3 ~ 0 (0은 기준시점이므로 제외)
window = range(-3, 1)

# 더미변수 생성 (기준 시점은 제외)
for k in window:
    if k == 0:
        continue
    else:
        df_대출정보_DID_pre[f'pre_{abs(k)}'] = (df_대출정보_DID['event_time'] == k).astype(int)

# 이벤트 더미생성 (pre_1 - 1개월전 시점은 기준시점으로 회귀식에 포함하지 않음 )
event_dummies = ' + '.join([
    f'pre_{abs(k)}' if k < 0 else f'post_{k}' 
    for k in window if (k != -1 and k!=0)
])

formula_event = f"DLQ_YN ~ {event_dummies} + EntityEffects"

# 패널 포맷 맞추기: MultiIndex (JOIN_SN, YM)
df_panel_for_PTA = df_대출정보_DID_pre.set_index(['JOIN_SN', 'YM'])

model = PanelOLS.from_formula(formula_event, data=df_panel_for_PTA)
results_PTA = model.fit(cov_type="clustered", cluster_entity=True)

print(results_PTA)

# ===============================================================
# 2) Staggered DiD 분석 수행
# - 회귀식 : DLQ_YNit(연체율) = β0+β⋅post(it)+α(i)+γ(t)+ϵ(it)
# - post(it) : 정책자금적용여부, 1 : YM(월)>= LN_DT(대출일자), 0 : YM(월) < LN_DT(대출일자)
# - α(i) : 개체(차주별) 고정효과
# - γ(t) : 시점(년월) 고정효과 
# ===============================================================

# 패널 포맷 맞추기: MultiIndex (JOIN_SN, YM)
df_panel = df_대출정보_DID.set_index(['JOIN_SN', 'YM'])

# 모형 추정 (fixed effects 내장 처리)
model = PanelOLS.from_formula("DLQ_YN ~ interaction_term + EntityEffects + TimeEffects", data=df_panel)
results = model.fit(cov_type="clustered", cluster_entity=True)
print(results.summary)

# ===============================================================
# Parallel Trend Assumption with Event Study 시각화 그래프 
# ===============================================================
# 2. 집단 분리
df_treated = df_대출정보_DID[df_대출정보_DID['정책금융여부'] == 1].copy()
df_control = df_대출정보_DID[df_대출정보_DID['정책금융여부'] == 0].copy()

# 3. 그룹별 평균 연체율 계산
treated_group = df_treated.groupby('event_time')['DLQ_YN'].mean().reset_index()
treated_group.rename(columns={'DLQ_YN': 'treated_rate'}, inplace=True)

control_group = df_control.groupby('event_time')['DLQ_YN'].mean().reset_index()
control_group.rename(columns={'DLQ_YN': 'control_rate'}, inplace=True)

# 4. event_time 기준으로 병합
merged = pd.merge(treated_group, control_group, on='event_time', how='outer').sort_values('event_time')
merged = merged.fillna(method='ffill').fillna(method='bfill')  # 결측값 채우기

merged_filterling = merged[(merged['event_time'] >=-9) & (merged['event_time'] <=9)]

# 5. 시각화
plt.figure(figsize=(10,6))
plt.plot(merged_filterling['event_time'], merged_filterling['treated_rate'], label='Policy', marker='o', color='teal')
plt.plot(merged_filterling['event_time'], merged_filterling['control_rate'], label='General', marker='s', color='indianred')
plt.axvline(x=-1, color='black', linestyle='--', label='Treatement line')

plt.title('Comparison of changes in delinquency rates based on event time (policy funds vs. general funds)')
plt.xlabel('EventTime (Difference from the time of loan)')
plt.ylabel('delinquency rate')
plt.legend()
plt.grid(True)
plt.xticks(merged_filterling['event_time'])
plt.tight_layout()
plt.show()


# 시각화 (parallel Trend 회귀계수 비교 )

# 회귀 결과에서 계수 추출
coefs = results_PTA.params
conf_int = results_PTA.conf_int()

# pre 계수만 추출
pre_terms = [term for term in coefs.index if term.startswith('pre_')]
pre_times = [-int(term.split('_')[1]) for term in pre_terms]  # pre_3 → -3
pre_coefs = [coefs[term] for term in pre_terms]
pre_lows = [conf_int.loc[term][0] for term in pre_terms]
pre_highs = [conf_int.loc[term][1] for term in pre_terms]

# 시계열 정렬
sorted_idx = sorted(range(len(pre_times)), key=lambda i: pre_times[i])
pre_times = [pre_times[i] for i in sorted_idx]
pre_coefs = [pre_coefs[i] for i in sorted_idx]
pre_lows = [pre_lows[i] for i in sorted_idx]
pre_highs = [pre_highs[i] for i in sorted_idx]

# 시각화
plt.figure(figsize=(8,5))
plt.errorbar(pre_times, pre_coefs, 
             yerr=[ [pre_coefs[i] - pre_lows[i] for i in range(len(pre_lows))],
                    [pre_highs[i] - pre_coefs[i] for i in range(len(pre_highs))] ],
             fmt='o', capsize=5, color='steelblue', label='Coefficient ± CI')

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Event Time (t)")
plt.ylabel("Coefficient (vs pre_1)")
plt.title("Parallel Trends Assumption: Pre-treatment Coefficients")
plt.xticks(pre_times)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===============================================================
# DiD Regression 결과 시각화
# ===============================================================

# TWFE 회귀 결과에서 treatment 계수와 신뢰구간 추출
coef_treatment = results.params['interaction_term']
conf_int_treatment = results.conf_int().loc['interaction_term']

plt.figure(figsize=(6,4))
plt.bar(['Treatment'], [coef_treatment], yerr=[[coef_treatment - conf_int_treatment[0]], [conf_int_treatment[1] - coef_treatment]],
        color='skyblue', capsize=10)
plt.ylabel('Coefficient Estimate')
plt.title('DiD Regression Estimate (Treatment Effect)')
plt.axhline(0, color='gray', linestyle='--')
plt.show()



#############################################################################
# Section5. 전략적 연체자 예측을 위한 feature engineering & 모델 개발 
# 1) 독립변수 병합 
# 2) 모델학습
# 3) SHAP 분석 시각화 
#############################################################################

# 0) post 데이터만 선별 
df_대출정보_DID_post = df_대출정보_DID[df_대출정보_DID['post'] ==1]

# 1) 독립변수 병합
df_예측모델링 = df_대출정보_DID.merge(df_대출정보_독립변수_집계, on=['JOIN_SN','JOIN_SN_TYP'], how="left").merge(df_지급능력_독립변수_집계, on=['JOIN_SN','JOIN_SN_TYP'], how="left").merge(df_보험계약정보_독립변수_집계, on=['JOIN_SN','JOIN_SN_TYP'], how="left").merge(df_연체정보_독립변수_집계, on=['JOIN_SN','JOIN_SN_TYP'], how="left").merge(df_신용카드정보_독립변수_집계, on=['JOIN_SN','JOIN_SN_TYP'], how="left").merge(df_소비성향_독립변수_집계, on=['JOIN_SN','JOIN_SN_TYP'], how="left")

# 2) 결측치처리 (모의데이터 성격상 데이터가 매우 부족해 랜덤배정으로 처리하였음)
def fill_by_policy(row, col, col_min, col_max, col_mean, col_median):
    if pd.notna(row[col]):
        return row[col]  # 결측 아님 → 그대로
    if ((row["정책금융여부"] == 1) & (col in ["최근3개월_신용카드사용률","보험건수","보험월납입액"])):
        # 최대값에 가까운 범위에서 랜덤 (예: 상위 25% 구간)
        if col in ["보험건수","보험월납입액"] :
          return np.random.randint(low=col_median, high=col_median+col_max*0.7)
        else : 
          return np.random.uniform(low=col_median, high=col_median+col_max*0.7)
    else:
        # 평균 근처에서 랜덤
        if col in ["보험건수","대출금액합","보험월납입액","대출건수","지급능력","보험월납입액"] :
          return np.random.randint(low=col_min, high=col_median+col_max*0.2)
        elif col in ["장기고액대출건수"] :
          return np.random.randint(low=col_min, high=col_max)
        else : 
          return np.random.uniform(low=col_min, high=col_max)


policy_cnt = len(df_예측모델링[df_예측모델링['정책금융여부']==1])
general_cnt = len(df_예측모델링[df_예측모델링['정책금융여부']==0])

#장기고액대출건수 결측치 대체
values = [0, 1, 2, 3]
general_probs = [0.966, 0.025, 0.005, 0.004]
policy_probs = [0.920, 0.045, 0.023, 0.012]

df_예측모델링.loc[df_예측모델링['정책금융여부'] == 0,'장기고액대출건수'] = np.random.choice(values, size=general_cnt, p=general_probs)
df_예측모델링.loc[df_예측모델링['정책금융여부'] ==1,'장기고액대출건수'] = np.random.choice(values, size=policy_cnt, p=policy_probs)

#최근3개월_신용카드사용률
values = [0.23, 0.45, 0.67, 0.72]
general_probs = [0.966, 0.025, 0.005, 0.004]
policy_probs = [0.720, 0.183, 0.083, 0.014]

df_예측모델링.loc[df_예측모델링['정책금융여부'] ==0,'최근3개월_신용카드사용률'] = np.random.choice(values, size=general_cnt, p=general_probs)
df_예측모델링.loc[df_예측모델링['정책금융여부'] ==1,'최근3개월_신용카드사용률'] = np.random.choice(values, size=policy_cnt, p=policy_probs)

##9개월_3개월이전_신용카드사용률
values = [0.23, 0.45, 0.67, 0.72]
general_probs = [0.966, 0.025, 0.005, 0.004]
policy_probs = [0.966, 0.025, 0.005, 0.004]

df_예측모델링.loc[df_예측모델링['정책금융여부'] ==0,'9개월_3개월이전_신용카드사용률'] = np.random.choice(values, size=general_cnt, p=general_probs)
df_예측모델링.loc[df_예측모델링['정책금융여부'] ==1,'9개월_3개월이전_신용카드사용률'] = np.random.choice(values, size=policy_cnt, p=policy_probs)


#최근3개월_현금서비스사용률
values = [0, 0.02, 0.03, 0.04]
general_probs = [0.966, 0.025, 0.005, 0.004]
policy_probs = [0.720, 0.183, 0.083, 0.014]

df_예측모델링.loc[df_예측모델링['정책금융여부'] ==0,'최근3개월_현금서비스사용률'] = np.random.choice(values, size=general_cnt, p=general_probs)
df_예측모델링.loc[df_예측모델링['정책금융여부'] ==1,'최근3개월_현금서비스사용률'] = np.random.choice(values, size=policy_cnt, p=policy_probs)

#9개월_3개월이전_현금서비스사용률
values = [0, 0.02, 0.03, 0.04]
general_probs = [0.966, 0.025, 0.005, 0.004]
policy_probs = [0.966, 0.025, 0.005, 0.004]

df_예측모델링.loc[df_예측모델링['정책금융여부'] ==0,'9개월_3개월이전_현금서비스사용률'] = np.random.choice(values, size=general_cnt, p=general_probs)
df_예측모델링.loc[df_예측모델링['정책금융여부'] ==1,'9개월_3개월이전_현금서비스사용률'] = np.random.choice(values, size=policy_cnt, p=policy_probs)


target_cols = ["대출건수", "대출금액합", "지급능력", "보험건수", "보험월납입액","신용카드개수","장기연체건수","연체금액합","연체건수"]

for col in target_cols:
    col_min = df_예측모델링[col].min(skipna=True)
    col_max = df_예측모델링[col].max(skipna=True)
    col_mean = df_예측모델링[col].mean(skipna=True)
    col_median = df_예측모델링[col].median(skipna=True)

    df_예측모델링[col] = df_예측모델링.apply(lambda row: fill_by_policy(row, col, col_min, col_max, col_mean, col_median), axis=1)

df_예측모델링['신용카드_사용률_증가량'] = df_예측모델링['최근3개월_신용카드사용률'] - df_예측모델링['9개월_3개월이전_신용카드사용률']
df_예측모델링['현금서비스_사용률_증가량'] = df_예측모델링['최근3개월_현금서비스사용률'] - df_예측모델링['9개월_3개월이전_현금서비스사용률']
df_예측모델링["지급능력"].info()

df_예측모델링['장기연체건수'].isna()

# ===============================================================
# 정책자금집단, 일반자금집단의 lightgbm 모델학습
# ===============================================================

df_general= df_예측모델링[df_예측모델링['정책금융여부']==0] 
df_policy = df_예측모델링[df_예측모델링['정책금융여부']==1] 

# 1. 변수 정의
features = [
    '대출건수', '장기고액대출건수', '대출금액합', '지급능력',
    '보험건수', '보험월납입액', '연체건수', '장기연체건수', '연체금액합',
    '신용카드개수', '신용카드_사용률_증가량', '현금서비스_사용률_증가량'
]
target = 'DLQ_YN'

X_gen_train, X_gen_test, y_gen_train, y_gen_test = train_test_split(
    df_general[features], df_general[target], test_size=0.2, random_state=42)

X_pol_train, X_pol_test, y_pol_train, y_pol_test = train_test_split(
    df_policy[features], df_policy[target], test_size=0.2, random_state=42)


df_예측모델링['신용카드개수'].isna()

# 4. 모델 학습
model_gen = lgb.LGBMClassifier(random_state=42)
model_pol = lgb.LGBMClassifier(random_state=42)

model_gen.fit(X_gen_train, y_gen_train)
model_pol.fit(X_pol_train, y_pol_train)

# 5. SHAP 분석
explainer_gen = shap.TreeExplainer(model_gen)
explainer_pol = shap.TreeExplainer(model_pol)

shap_values_gen = explainer_gen.shap_values(X_gen_test)
shap_values_pol = explainer_pol.shap_values(X_pol_test)


# ===============================================================
# feature importance 차이를 시각화 
# ===============================================================
# 절대 SHAP 평균값 계산
mean_shap_gen = np.abs(shap_values_gen).mean(axis=0)
mean_shap_pol = np.abs(shap_values_pol).mean(axis=0)

feature_names = X_gen_test.columns.tolist()

df_shap = pd.DataFrame({
    'Feature': feature_names,
    '일반자금': mean_shap_gen,
    '정책자금': mean_shap_pol
})

# Plotly 시각화
fig = go.Figure(data=[
    go.Bar(name='일반자금 차주', x=df_shap['Feature'], y=df_shap['일반자금']),
    go.Bar(name='정책자금 차주', x=df_shap['Feature'], y=df_shap['정책자금'])
])

# 레이아웃 설정
fig.update_layout(
    barmode='group',
    title='SHAP 변수 중요도 비교 (정책자금 vs 일반자금)',
    xaxis_title='변수',
    yaxis_title='평균 SHAP 절대값',
    xaxis_tickangle=-45,
    height=500,
    width=900
)

fig.show()

# ==============================================================
# 2. 시각화 
# 2-1) 차입자의 운영 사업체 개수 파이플롯 시각화
# 2-2) 평균대출금액 박스플롯 시각화, 연령별 금액 히스토그램 시각화
# 2-3) 연령구간별 개인대출과목 분포 트리맵 시각화
# 2-4) 기업의 대출분포
# 2-5) 연령별 연체율 막대 시각화 - 평균연체율 표시 (version1 기관매칭 기준)
# 2-6) 연령별 연체율 막대 시각화 - 평균연체율 표시 (version2 차주 매칭기준)
# 2-7) 정책금융 상품 한정 연체율 표시 (version3 정책금융 이용차주 한정 기준)
# ===============================================================

# ------------------------------------
# 2-1) 차입자의 운영 사업체 개수 파이플롯 시각화   
# ------------------------------------

fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
ax = fig.add_subplot()

df_차주_기업개요정보_Summary

fig = px.pie(
    df_차주_기업개요정보_Summary,
    names='BIZ_COUNT',
    values='COUNT',
    title='차주 운영 사업체 개수(Pie Chart)',
    hole=0.3 # 0이면 일반 파이, 0.4 정도면 도넛
)

fig.show()

# --------------------------------------------------
# 2-2) 평균대출금액 박스플롯 시각화, 연령별 금액 히스토그램 시각화
# --------------------------------------------------
df_차주_대출_신용카드_연체정보.info()
df_차주_대출_신용카드_연체정보_소액 = df_차주_대출_신용카드_연체정보[(df_차주_대출_신용카드_연체정보['AMT'] <= 50000) & (df_차주_대출_신용카드_연체정보['대출신용카드구분']=='대출')]
df_차주_대출_신용카드_연체정보_그외 = df_차주_대출_신용카드_연체정보[(df_차주_대출_신용카드_연체정보['AMT'] > 50000) & (df_차주_대출_신용카드_연체정보['대출신용카드구분']=='대출')]

df_차주_대출_신용카드_연체정보_소액  = df_차주_대출_신용카드_연체정보_소액[df_차주_대출_신용카드_연체정보_소액['BTH_SECTION'] != '~19']
df_차주_대출_신용카드_연체정보_그외  = df_차주_대출_신용카드_연체정보_그외[df_차주_대출_신용카드_연체정보_그외['BTH_SECTION'] != '~19']

# 5천만원 이하 소액 
fig = px.box(
    df_차주_대출_신용카드_연체정보_소액,
    x='BTH_SECTION',
    y='AMT',
    color='BTH_SECTION',
    points='all',  # 이상치까지 점으로 표시
    title='연령대별 대출금액 분포 (Boxplot) - 5천만원 이하',
    labels={'BTH_SECTION': '연령구간', 'AMT': '대출금액 (천원)'}
)

fig.update_layout(
    xaxis=dict(
        categoryorder='array',
        categoryarray=df_차주_대출_신용카드_연체정보_소액['BTH_SECTION'].cat.categories.to_list()
    ),
    yaxis_tickformat=".0f",  # 소수 없이 퍼센트 (%)
)

fig.show()

# 5천만원 초과 그외 
fig = px.box(
    df_차주_대출_신용카드_연체정보_그외,
    x='BTH_SECTION',
    y='AMT',
    color='BTH_SECTION',
    points='all',  # 이상치까지 점으로 표시
    title='연령대별 대출금액 분포 (Boxplot) - 5천만원 초과',
    labels={'BTH_SECTION': '연령구간', 'AMT': '대출금액 (천원)'}
)

fig.update_layout(
    xaxis=dict(
        categoryorder='array',
        categoryarray=df_차주_대출_신용카드_연체정보_그외['BTH_SECTION'].cat.categories.to_list()
    ),
    yaxis_tickformat=",.0f",  # 소수 없이 퍼센트 (%)
)

fig.show()


"""
df_34 = df_차주_대출_신용카드_연체정보_그외[df_차주_대출_신용카드_연체정보_그외['BTH_SECTION'] == "~34"]
df_34 = df_34[['AMT']]
df_50 = df_차주_대출_신용카드_연체정보_그외[df_차주_대출_신용카드_연체정보_그외['BTH_SECTION'] == "~50"]
df_50 = df_34[['AMT']]
df_64 = df_차주_대출_신용카드_연체정보_그외[df_차주_대출_신용카드_연체정보_그외['BTH_SECTION'] == "~64"]
df_64 = df_34[['AMT']]

df_34.head(100)

plt.hist(df_34, color='red', alpah =0.2, bins = 10000000, label ='~34', density=True)
plt.hist(df_50, color='orange', alpah = 0.2, bins = 100000, label ='~50', density=True)
plt.hist(df_64, color='yellow', alpah = 0.2, bins = 100000, label ='~64', density=True)

plt.legend()
plt.show()"
"""

# --------------------------------------------------
# 2-3) 연령구간별 개인대출과목 분포 트리맵 시각화
# --------------------------------------------------
df_개인대출정보_최종  = (
    df_개인대출정보.sort_values(['YM'],ascending=True) 
      .groupby(['JOIN_SN','COM_SN', 'LN_CD_1','LN_CD_2','LN_CD_3'], as_index=False)  # 그룹핑
      .tail(1)  # 각 그룹에서 마지막 row (최신 월)
)

df_개인대출정보_최종.info()
df_차주.info()

df_차주_개인대출정보 = df_차주_연령구간화.merge(df_개인대출정보_최종,on=['JOIN_SN','JOIN_SN_TYP'], how='left')


# 연령구간별 개인대출과목 코드_3 시각화 (대출상품코드2 - 신용대출, 담보대출, 카드대출, 할부, 리스 구분)
df_차주_개인대출정보_집계 = df_차주_개인대출정보.groupby(['BTH_SECTION', 'LN_CD_2'])['LN_AMT'].sum().reset_index()
df_차주_개인대출정보_집계 = df_차주_개인대출정보_집계[df_차주_개인대출정보_집계['LN_AMT'] >0]

df_차주_개인대출정보_집계.head(100)


df_차주_개인대출정보_집계['LN_CD_NM'] = df_차주_개인대출정보_집계['LN_CD_2'].map(CommonCode.LN_CD_2)
df_차주_개인대출정보_집계['LN_CD_NM'] = df_차주_개인대출정보_집계['LN_CD_NM'].astype('category')
df_차주_개인대출정보_집계.info()

fig = px.treemap(df_차주_개인대출정보_집계,
                 path=['BTH_SECTION','LN_CD_NM'],
                 values='LN_AMT',
                 color='LN_AMT',
                 labels={'BTH_SECTION': '연령구간', 'LN_AMT': '대출금액 (천원)'},
                 color_continuous_scale='viridis')

fig.show()


# 연령구간별 개인대출과목 코드_3 시각화 (정책금융 대출 - 현재 모의데이터에는 소진공 기금대출, 지자체 기금대출이 없는 한계점 있음)
df_차주_개인대출정보_집계_2 = df_차주_개인대출정보.groupby(['BTH_SECTION', 'LN_CD_3'])['LN_AMT'].sum().reset_index()
df_차주_개인대출정보_집계_2 = df_차주_개인대출정보_집계_2[df_차주_개인대출정보_집계_2['LN_AMT'] >0].reset_index(drop=True)
df_차주_개인대출정보_집계_2 = df_차주_개인대출정보_집계_2[~df_차주_개인대출정보_집계_2['LN_CD_3'].isin(['0','900'])]


df_차주_개인대출정보_집계_2.head(100)

df_차주_개인대출정보_집계_2['LN_CD_NM'] = df_차주_개인대출정보_집계_2['LN_CD_3'].map(CommonCode.LN_CD_3)
df_차주_개인대출정보_집계_2['LN_CD_NM'] = df_차주_개인대출정보_집계_2['LN_CD_NM'].astype('category')
df_차주_개인대출정보_집계_2.info()

df_차주_개인대출정보_집계_2.info()


fig = px.bar(df_차주_개인대출정보_집계_2, x='BTH_SECTION', y='LN_AMT', color='LN_CD_NM', barmode='stack', text='LN_CD_NM')
fig.update_layout(
    yaxis=dict(
        tickformat=',.0f',  # 1,000 → 1,000 식으로 콤마 포함
        title='금액 (단위: 천원)'
    ),
    xaxis_title='연령구간'
)

fig.update_layout(title='연령구간별 정책금융 대출금액 누적 막대그래프(기타대출 제외)')
fig.show()


# --------------------------------------------------
# 2-4) 기업의 대출분포
# --------------------------------------------------
df_기업대출정보['LN_CD'] = df_기업대출정보['LN_CD'].astype(str)

df_기업대출정보_최종  = (
    df_기업대출정보[df_기업대출정보['JOIN_SN_TYP'] ==1]
      .sort_values(['YM'],ascending=True) 
      .groupby(['JOIN_SN','COM_SN','LN_CD'], as_index=False)  # 그룹핑
      .tail(1)  # 각 그룹에서 마지막 row (최신 월)
)

df_기업대출정보_최종.info()
df_차주.info()

df_차주_기업대출정보 = df_차주_연령구간화.merge(df_기업대출정보_최종,on=['JOIN_SN','JOIN_SN_TYP'], how='left')

# 연령구간별 기업대출과목 코드 시각화 (기업대출과목코드 - LN_ACCT_CD)
df_차주_기업대출정보_집계 = df_차주_기업대출정보.groupby(['BTH_SECTION', 'LN_CD'])['LN_AMT'].sum().reset_index()
df_차주_기업대출정보_집계 = df_차주_기업대출정보_집계[df_차주_기업대출정보_집계['LN_AMT'] >0]

df_차주_기업대출정보_집계.info()
df_차주_기업대출정보_집계.head(100)

df_차주_기업대출정보_집계['LN_CD_NM'] = df_차주_기업대출정보_집계['LN_CD'].map(CommonCode.LN_ACCT_CD)
df_차주_기업대출정보_집계['LN_CD_NM'] = df_차주_기업대출정보_집계['LN_CD_NM'].astype('category')
df_차주_기업대출정보_집계.info()

fig = px.treemap(df_차주_기업대출정보_집계,
                 path=['BTH_SECTION','LN_CD_NM'],
                 values='LN_AMT',
                 color='LN_AMT',
                 labels={'LN_AMT': '대출금액 (천원)'},
                 color_continuous_scale='viridis')

fig.show()


# -------------------------------------------------------------
# 2-5) 연령별 연체율 막대 시각화 - 평균연체율 표시 (version1 기관매칭 기준)
# -------------------------------------------------------------
df_차주_대출_신용카드_연체정보['연체여부'] =  df_차주_대출_신용카드_연체정보['DLQ_RGST_DT'].apply(lambda x : 1 if pd.notna(x) else 0)
df_차주_대출_신용카드_연체정보_집계 = df_차주_대출_신용카드_연체정보.groupby('BTH_SECTION')['연체여부'].agg(['mean', 'count', 'sum']).reset_index()
df_차주_대출_신용카드_연체정보_집계.rename(columns={'mean': '연체율', 'count': '차주수', 'sum': '연체차주수'}, inplace=True)
df_차주_대출_신용카드_연체정보_집계['연체율'] = df_차주_대출_신용카드_연체정보_집계['연체율']*100
print(df_차주_대출_신용카드_연체정보_집계)

전체평균연체율 = df_차주_대출_신용카드_연체정보['연체여부'].mean()

# 막대그래프 + 텍스트
fig = px.bar(
    df_차주_대출_신용카드_연체정보_집계,
    x='BTH_SECTION',
    y='연체율',
    text='연체율',
    color='연체율',
    color_continuous_scale='Reds',
    labels={'연체율': '연체율 (%)'},
    title='연령별 연체율 (대출기관 매칭)'
)

# 평균선 추가
fig.add_hline(
    y=전체평균연체율*100,
    line_dash='dash',
    line_color='red',
    annotation_text=f'전체 평균: {전체평균연체율:.2f}%',
    annotation_position='top left',
    annotation_font_color='red'
)

# y축 여유공간
fig.update_layout(
    xaxis_title='연령 구간',
    yaxis_range=[0, max(df_차주_대출_신용카드_연체정보_집계['연체율']) * 1.3]
)

fig.update_traces(
    texttemplate='%{text:.2f}%',
    textposition='outside'
)

fig.show()

# -------------------------------------------------------------
# 2-6) 연령별 연체율 막대 시각화 - 평균연체율 표시 (version2 차주 매칭기준)
# -------------------------------------------------------------
df_차주_연체정보['연체여부'] = df_차주_연체정보['DLQ_RGST_DT'].apply(lambda x : 1 if pd.notna(x) else 0)
df_차주_연체정보_집계 = df_차주_연체정보.groupby('BTH_SECTION')['연체여부'].agg(['mean', 'count', 'sum']).reset_index()
df_차주_연체정보_집계.rename(columns={'mean': '연체율', 'count': '차주수', 'sum': '연체차주수'}, inplace=True)
df_차주_연체정보_집계['연체율'] = df_차주_연체정보_집계['연체율']*100

전체평균연체율 = df_차주_연체정보['연체여부'].mean()

# 막대그래프 + 텍스트
fig = px.bar(
    df_차주_연체정보_집계,
    x='BTH_SECTION',
    y='연체율',
    text='연체율',
    color='연체율',
    color_continuous_scale='Reds',
    labels={'연체율': '연체율 (%)'},
    title='연령별 연체율 (대출기관 언매칭)'
)

# 평균선 추가
fig.add_hline(
    y=전체평균연체율*100,
    line_dash='dash',
    line_color='red',
    annotation_text=f'전체 평균: {전체평균연체율:.2f}%',
    annotation_position='top left',
    annotation_font_color='red'
)

# y축 여유공간
fig.update_layout(
    xaxis_title='연령 구간',
    yaxis_range=[0, max(df_차주_연체정보_집계['연체율']) * 1.3]
)

fig.update_traces(
    texttemplate='%{text:.2f}%',
    textposition='outside'
)

fig.show()

# -------------------------------------------------------------
# 2-7) 정책금융 상품 한정 연체율 표시 
# -------------------------------------------------------------
df_차주_정책금융이용_연체정보['연체여부'] = df_차주_정책금융이용_연체정보['DLQ_RGST_DT'].apply(lambda x : 1 if pd.notna(x) else 0)
df_차주_정책금융이용_연체정보_집계 = df_차주_정책금융이용_연체정보.groupby('BTH_SECTION')['연체여부'].agg(['mean', 'count', 'sum']).reset_index()
df_차주_정책금융이용_연체정보_집계.rename(columns={'mean': '연체율', 'count': '차주수', 'sum': '연체차주수'}, inplace=True)
df_차주_정책금융이용_연체정보_집계['연체율'] = df_차주_정책금융이용_연체정보_집계['연체율'] *100

전체평균연체율 = df_차주_정책금융이용_연체정보['연체여부'].mean()*100

# 막대그래프 + 텍스트
fig = px.bar(
    df_차주_정책금융이용_연체정보_집계,
    x='BTH_SECTION',
    y='연체율',
    text='연체율',
    color='연체율',
    color_continuous_scale='Reds',
    labels={'연체율': '연체율 (%)'},
    title='연령별 연체율 (정책금융 이용, 대출기관 언매칭) - 향후 본 데이터 시각화 시 소상공인 정책자금 한정'
)

# 평균선 추가
fig.add_hline(
    y=전체평균연체율,
    line_dash='dash',
    line_color='red',
    annotation_text=f'전체 평균: {전체평균연체율:.2f}%',
    annotation_position='top left',
    annotation_font_color='red'
)

# y축 여유공간
fig.update_layout(
    xaxis_title='연령 구간',
    yaxis_tickformat=".2f",  # 소수 없이 퍼센트 (%)
    yaxis_range=[0, max(df_차주_정책금융이용_연체정보_집계['연체율']) * 1.3]
)

fig.update_traces(
    texttemplate='%{text:.2f}%',
    textposition='outside'
)

fig.show()

# -------------------------------------------------------------
# 2-8) 정책금융 비이용 차주 연체율 표시 
# -------------------------------------------------------------
df_차주_정책금융비이용_연체정보['연체여부'] = df_차주_정책금융비이용_연체정보['DLQ_RGST_DT'].apply(lambda x : 1 if pd.notna(x) else 0)
df_차주_정책금융비이용_연체정보_집계 = df_차주_정책금융비이용_연체정보.groupby('BTH_SECTION')['연체여부'].agg(['mean', 'count', 'sum']).reset_index()
df_차주_정책금융비이용_연체정보_집계.rename(columns={'mean': '연체율', 'count': '차주수', 'sum': '연체차주수'}, inplace=True)
df_차주_정책금융비이용_연체정보_집계['연체율'] = df_차주_정책금융비이용_연체정보_집계['연체율'] *100

전체평균연체율 = df_차주_정책금융비이용_연체정보['연체여부'].mean()*100

# 막대그래프 + 텍스트
fig = px.bar(
    df_차주_정책금융비이용_연체정보_집계,
    x='BTH_SECTION',
    y='연체율',
    text='연체율',
    color='연체율',
    color_continuous_scale='Reds',
    labels={'연체율': '연체율 (%)'},
    title='연령별 연체율 (정책금융 비이용, 대출기관 언매칭)'
)

# 평균선 추가
fig.add_hline(
    y=전체평균연체율,
    line_dash='dash',
    line_color='red',
    annotation_text=f'전체 평균: {전체평균연체율:.2f}%',
    annotation_position='top left',
    annotation_font_color='red'
)

# y축 여유공간
fig.update_layout(
    xaxis_title='연령 구간',
    yaxis_tickformat=".2f",  # 소수 없이 퍼센트 (%)
    yaxis_range=[0, max(df_차주_정책금융비이용_연체정보_집계['연체율']) * 1.3]
)

fig.update_traces(
    texttemplate='%{text:.2f}%',
    textposition='outside'
)

fig.show()


df_차주_정책금융이용_연체정보.info()
