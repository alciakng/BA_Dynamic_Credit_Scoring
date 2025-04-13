from datetime import datetime
from sklearn.calibration import expit
from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import os
import scipy.stats as stats

from common_code import CommonCode

class DatasetBuilder:
    def __init__(self,dataset_json):
        
        # 기본 데이터셋
        self.df_차주 = None
        self.dataset_json = dataset_json
        self.df_기업개요정보 = None
        self.df_연체정보_신용도판단정보= None
        self.df_연체정보_공공정보 = None
        self.df_개인대출정보_금융권 = None
        self.df_개인대출정보_대부업권 = None
        self.df_신용카드개설정보 = None
        self.df_신용카드이용정보 = None
        self.df_채무보증정보 = None
        self.df_기업대출정보 = None
        self.df_기술신용평가정보 = None
        self.df_기술신용대출정보 = None
        self.df_보험계약관계자정보 = None
        self.df_보험계약정보 = None
        self.df_보험담보정보 = None
        self.df_청구사고정보 = None
        self.df_청구계약정보 = None
        self.df_청구지급사유정보 = None
        self.df_청구지급상세사유정보 = None
        self.df_청구피해자물정보 = None

        # 10차 업종코드
        self.df_ksic = None

        # merge 후 데이터셋
        self.df_개인대출정보 = None
        self.df_대출정보 = None
        self.df_대출정보_보유기관_집계 = None
        self.df_대출정보_대출과목_집계 = None
        self.df_대출정보_집계 = None
        self.df_차주_기업개요정보 = None
        self.df_차주_기업개요정보_집계 = None
        self.df_차주_대출정보_집계 = None
        self.df_차주_연령구간화 = None
        self.df_차주_연령구간화_정책금융이용 = None
        self.df_차주_연체정보 = None
        self.df_차주_정책금융이용_연체정보 = None
        self.df_차주_대출_신용카드_연체정보 = None
        self.df_연체정보 = None
        self.df_연체정보_기관별_최종 = None
        self.df_연체정보_최종 = None
        self.df_대출_신용카드_병합_기관별_최종 = None
        self.df_대출_신용카드_연체정보 = None
        
        # 독립변수
        self.df_독립변수 = None

        self.df_대출정보_독립변수_집계 = None
        self.df_지급능력_독립변수_집계 = None
        self.df_보험계약정보_독립변수_집계 = None
        self.df_연체정보_독립변수_집계 = None
        self.df_신용카드정보_독립변수_집계 = None
        self.df_소비성향_독립변수_집계 = None

        # 예측모델링을 위한 데이터프레임 
        self.df_예측모델링 = None

        np.random.seed(42)

        self.ln_cd_dist_기업 = {
            str(code): {
                "a": round(np.random.uniform(1.0, 2.5), 2),
                "b": round(np.random.uniform(7.0, 8.0), 2)
            }
            for code in CommonCode.LN_ACCT_CD.keys()
        }

        self.ln_cd_dist_개인 = {
            str(code): {
                "a": round(np.random.uniform(1.0, 2.7), 2),
                "b": round(np.random.uniform(6.5, 8.0), 2)
            }
            for code in CommonCode.LN_CD_2.keys()
        }


    def load_data(self):
        base_dir = os.path.dirname(os.path.dirname(__file__)) # 상위폴더
        datset_dir = os.path.join(base_dir, 'data')

        self.df_차주 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['차주정보']))  
        self.df_기업개요정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['기업개요정보']))
        self.df_연체정보_신용도판단정보= pd.read_csv(os.path.join(datset_dir, self.dataset_json['연체정보(신용도판단정보)']))
        self.df_연체정보_공공정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['연체정보(공공정보)']))
        self.df_개인대출정보_금융권 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['개인대출정보(금융권)']))
        self.df_개인대출정보_대부업권 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['개인대출정보(대부업)']))
        self.df_신용카드개설정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['신용카드개설정보']))
        self.df_신용카드이용정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['신용카드이용정보']))
        self.df_채무보증정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['채무보증정보']))
        self.df_기업대출정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['기업대출정보']))
        self.df_기술신용평가정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['기술신용평가정보']))
        self.df_기술신용대출정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['기술신용대출정보']))
        self.df_보험계약관계자정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['보험계약관계자정보']))
        self.df_보험계약정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['보험계약정보']))
        self.df_보험담보정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['보험담보정보']))
        self.df_청구사고정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['청구사고정보']))
        self.df_청구계약정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['청구계약정보']))
        self.df_청구지급사유정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['청구지급사유정보']))
        self.df_청구지급상세사유정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['청구지급상세사유정보']))
        self.df_청구피해자물정보 = pd.read_csv(os.path.join(datset_dir, self.dataset_json['청구피해자물정보']))

    ############################################################
    #  함수 : 표준산업분류(KIC) 10차코드 Load 
    ############################################################
    def load_kic(self):
        # 한국표준산업분류(KIC) 10차 코드 
        url = 'https://github.com/FinanceData/KSIC/raw/master/KSIC_10.csv.gz'
        self.df_ksic = pd.read_csv(url, dtype='str')
        self.df_ksic['Industy_code'] = self.df_ksic['Industy_code'].str.pad(width=5, side='right', fillchar='0')
        self.df_ksic['Industy_code'] = self.df_ksic['Industy_code'].str[:4]
        self.df_ksic.rename(columns={'Industy_code': 'BIZ_TYP_CD'}, inplace=True)
        self.df_ksic.rename(columns={'Industy_name': 'BIZ_TYP_NM'}, inplace=True)
        self.df_ksic.head(100)
        self.df_ksic.info()


    ####################################################################
    #  함수 : 데이터 셋 가져오기 - df_getter
    ####################################################################
    def df_getter(self, name: str):
        """
        DataFrame 이름(str)을 입력하면 해당 self.df_ 속성 반환.
        예: get_df('df_차주') → self.df_차주
        """
        if not name.startswith("df_"):
            name = "df_" + name
        attr_name = f"self.{name}"
        
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(f"{attr_name} 라는 DataFrame 속성은 존재하지 않습니다.")
        

    def app_initialize(self):
        if not st.session_state.get('builder_initialized'):
            self.load_kic()
            self.load_data()
            self.merge_borrower()
            self.merge_loan()
            self.merge_credit()
            self.independent_variable_modeler()
            self.merge_loan_variable()
            self.handle_missing_values()

            st.session_state['builder_initialized'] = True


    ####################################################################
    #  함수 : 차주정보 merge - merge_borrower
    #  0. 각 데이터 프레임 결측치 파악
    #  1. 차주-기업개요정보 (1:n) 병합처리 
    #  2. 개인사업자 필터링, 소상공인 필터링(유의한 변수없음.교수님께 질의필요)
    #  3. 연령 구간화 (0 : ~19세 1: ~34세 2: ~50세 3: ~64세 4: 65세~)
    #  4. 한국표준산업분류(KIC) 10차코드로 업종명 매핑(현재 모의데이터 업종코드는 비정상적)
    ####################################################################
    def merge_borrower(self):
        # ------------------------------
        # 0. 결측치파악 => 결측치 존재하지 않음 
        # ------------------------------
        print(self.df_차주.info())
        print(self.df_기업개요정보.info())

        # -----------------------------
        # 1. 차주-기업개요정보 (1:n) 병합처리
        # -----------------------------
        self.df_차주_기업개요정보 = self.df_기업개요정보.merge(self.df_차주, on=['JOIN_SN','JOIN_SN_TYP'], how='left')

        # -------------------------------------------------------
        # 2. 개인사업자 필터링, 소상공인 필터링(유의한 변수없음.교수님께 질의필요)
        # -------------------------------------------------------
        self.df_차주_기업개요정보 = self.df_차주_기업개요정보[self.df_차주_기업개요정보["JOIN_SN_TYP"] ==1]
        self.df_차주_기업개요정보.info()

        # -----------
        # 3. 연령구간화 
        # -----------
        bins = [-1, 19, 34, 50, 64, 150]  # 각 구간의 경계값
        labels = ["~19세","~34세","~50세","~64세","65세~"]          # 구간에 매길 값

        self.df_차주_기업개요정보['BTH_SECTION'] = pd.cut(datetime.now().year - self.df_차주_기업개요정보['BTH_YR'], bins=bins, labels=labels)

        # 차주-기업개요정보가 아닌 차주 전체를 대상으로 연령구간화 
        self.df_차주_연령구간화 = self.df_차주.copy()
        self.df_차주_연령구간화['BTH_SECTION'] = pd.cut(datetime.now().year - self.df_차주['BTH_YR'], bins=bins, labels=labels)

        # ---------------------------------------------------
        # 4. 10차 코드로 업종명 매핑 (현재 모의데이터셋 불완전하여 매핑안됨)
        # ---------------------------------------------------
        #df_차주_기업개요정보['BIZ_TYP_CD'] = df_차주_기업개요정보['BIZ_TYP_CD'].astype(object)
        #df_차주_기업개요정보 = df_차주_기업개요정보.merge(df_ksic,on="BIZ_TYP_CD", how='left')

        # ---------------------------
        # 5. 시각화를 위한 Summary df 생성
        # ---------------------------
        self.df_차주_기업개요정보_집계 = self.df_차주_기업개요정보.groupby('JOIN_SN')['BIZ_SN'].nunique().reset_index(name='BIZ_COUNT')
        self.df_차주_기업개요정보_집계 = self.df_차주_기업개요정보_집계.groupby('BIZ_COUNT')["JOIN_SN"].count().reset_index(name='COUNT')

        self.df_차주_기업개요정보_집계['BIZ_COUNT'] = self.df_차주_기업개요정보_집계['BIZ_COUNT'].astype(str) +"개"
        self.df_차주_기업개요정보_집계['BIZ_COUNT'] = self.df_차주_기업개요정보_집계['BIZ_COUNT'].astype('category')


        self.df_차주_기업개요정보_집계.head()

    ####################################################################
    #  함수 : 대출정보 merge - merge_loan
    #  개인대출과 기업대출정보를 병합한다.
    ####################################################################
    def merge_loan(self):
        self.df_개인대출정보 = pd.concat([self.df_개인대출정보_금융권, self.df_개인대출정보_대부업권], ignore_index=True)
        # 구분컬럼추가 
        self.df_개인대출정보["기업개인구분"] = "개인"
        self.df_기업대출정보["기업개인구분"] = "기업"
        # 대출과목코드 일원화(1.개인대출상품코드 스트링 붙이기(LN_CD_1+LN_CD_2+LN_CD_3),2.기업대출과목코드를 개인대출코드로 바꿈)
        self.df_개인대출정보["LN_CD_1"] = self.df_개인대출정보["LN_CD_1"].astype('str')
        self.df_개인대출정보["LN_CD_2"] = self.df_개인대출정보["LN_CD_2"].astype('str')
        self.df_개인대출정보["LN_CD_3"] = self.df_개인대출정보["LN_CD_3"].astype('str')
        self.df_기업대출정보.rename(columns={'LN_ACCT_CD': 'LN_CD_1'}, inplace=True)

        # 기업대출 default 변수 
        self.df_기업대출정보['LN_DT'] = 0    
        self.df_기업대출정보['EXP_DT'] = 0
        self.df_기업대출정보['RATE'] = 0
        self.df_기업대출정보['LN_CD_2'] = None
        self.df_기업대출정보['LN_CD_3'] = None

        # 겹치는 컬럼 파악 
        common_cols = list(self.df_개인대출정보.columns.intersection(self.df_기업대출정보.columns))
        # 해당 컬럼기준으로 개인대출정보+기업대출정보 병합 
        self.df_대출정보 = pd.concat([self.df_개인대출정보[common_cols], self.df_기업대출정보[common_cols]], ignore_index=True)

    ####################################################################
    #  함수 : 신용도정보 merge - merge_credit
    #  민간연체와 공공연체를 병합한다.
    ####################################################################
    def merge_credit(self):
        self.df_연체정보_신용도판단정보["민간공공구분"] = "민간"
        self.df_연체정보_공공정보["민간공공구분"] ="공공"

        # 연체정보 통합 
        self.df_연체정보 = pd.concat([self.df_연체정보_신용도판단정보, self.df_연체정보_공공정보], ignore_index=True)

        # 차주 마지막 등록월의 연체정보만을 가져온다. (simple 버전)
        self.df_연체정보_최종 = (
            self.df_연체정보.sort_values(['YM'],ascending=True)  # 월 오름차순 정렬
            .groupby(['JOIN_SN'], as_index=False)  # 그룹핑
            .tail(1)  # 각 그룹에서 마지막 row (최신 월)
        )

    ##########################################################################################
    #  함수 : 독립변수 결합
    #  독립변수 : '대출건수', '장기고액대출건수', '대출금액합', '지급능력', '보험건수', '보험월납입액', 
    #  '연체건수', '장기연체건수', '연체금액합', '신용카드개수', '신용카드_사용률_증가량', '현금서비스_사용률_증가량'
    ##########################################################################################
    def independent_variable_modeler(self) :
        # --------------------------------------------------------------------
        # 1) 대출정보 독립변수 생성 (차주별 연체이전 대출개수, 대출금액, 장기고액대출건수)
        # --------------------------------------------------------------------
        df_대출정보_최종  = (
            self.df_대출정보.sort_values(['YM'],ascending=True)  # 월 오름차순 정렬
            .groupby(['JOIN_SN', 'COM_SN', 'LN_DT','LN_CD_1','LN_CD_2','LN_CD_3'], as_index=False)  # 그룹핑
            .tail(1)  # 각 그룹에서 마지막 row (최신 월)
        )

        # 대출정보와 연체정보를 병합
        df_연체정보_대출정보_최종 = pd.merge(self.df_연체정보_최종, df_대출정보_최종, on=["JOIN_SN","JOIN_SN_TYP"], how="inner")
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
        self.df_대출정보_독립변수_집계 = df_연체정보_대출정보_최종_필터링.groupby(["JOIN_SN","JOIN_SN_TYP"]).agg(
            대출건수=("YM_x", "count"),
            장기고액대출건수=("장기고액", "sum"),
            대출금액합=("LN_AMT", "sum")
        ).reset_index()

        # ----------------------------------------------------------------------
        # 2) 지급능력 독립변수 설정 (지급능력 점수 = 신용카드이용한도*0.85 + 현금서비스한도*0.15)
        # ----------------------------------------------------------------------
        df_신용카드이용정보_최종  = (
            self.df_신용카드이용정보.sort_values(['YM'],ascending=True)  # 월 오름차순 정렬
            .groupby(['JOIN_SN'], as_index=False)  # 그룹핑
            .tail(1)  # 각 그룹에서 마지막 row (최신 월)
        )

        df_신용카드이용정보_최종['지급능력'] = df_신용카드이용정보_최종['CD_USG_LMT']*0.85 + df_신용카드이용정보_최종['CD_CA_LMT']*0.15

        # 신용카드이용정보와 연체정보를 병합
        df_신용카드이용정보_연체정보_최종 = pd.merge(df_신용카드이용정보_최종, self.df_연체정보_최종, on=["JOIN_SN","JOIN_SN_TYP"], how="inner")
        df_신용카드이용정보_연체정보_최종['YM_y'] = df_신용카드이용정보_연체정보_최종['YM_y'].fillna(df_신용카드이용정보_연체정보_최종['YM_x'])

        # 연체이전 대출만 필터링 
        df_신용카드이용정보_연체정보_최종_필터링 = df_신용카드이용정보_연체정보_최종[df_신용카드이용정보_연체정보_최종["YM_x"] <= df_신용카드이용정보_연체정보_최종["YM_y"]]

        # 신용카드 독립변수 집계 (지급능력 )
        self.df_지급능력_독립변수_집계 = df_신용카드이용정보_연체정보_최종_필터링.groupby(["JOIN_SN","JOIN_SN_TYP"]).agg(
            지급능력=("지급능력", "max")
        ).reset_index()

        # ----------------------------------------------------------------------
        # 3) 보험정보 독립변수 설정 (차주별 보험가입건수, 월 납입보험료, 청구사고건수)
        # ----------------------------------------------------------------------
        # 보험계약관계자정보와 보험계약정보를 병합
        df_보험계약정보_병합 = pd.merge(self.df_보험계약관계자정보, self.df_보험계약정보, on=["SCTR_CD","POL_SN"], how="left")

        # 보험계약정보와 연체정보를 병합
        df_보험계약정보_연체정보_최종 = pd.merge(df_보험계약정보_병합, self.df_연체정보_최종, on=["JOIN_SN","JOIN_SN_TYP"], how="inner")

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
        self.df_보험계약정보_독립변수_집계 = df_보험계약정보_연체정보_최종.groupby(["JOIN_SN","JOIN_SN_TYP"]).agg(
            보험건수=("POL_SN", "count"),
            보험월납입액=("월_보험료납입액", "sum")
        ).reset_index()

        # ----------------------------------------------------------------------
        # 4) 연체정보 독립변수 생성 (연체건수, 장기연체건수, 연체금액, 과거도산경험)
        # ----------------------------------------------------------------------
        df_연체정보_기관별_연체날짜별_집계  = (
            self.df_연체정보.sort_values(['YM'],ascending=True)  # 월 오름차순 정렬
            .groupby(['JOIN_SN','COM_SN','DLQ_RGST_DT'], as_index=False)  # 그룹핑
            .tail(1)  # 각 그룹에서 마지막 row (최신 월)
        )

        df_연체정보_기관별_연체날짜별_집계[df_연체정보_기관별_연체날짜별_집계['JOIN_SN'] == 1714896]


        # 연체집계정보와 최종연체정보를 병합
        df_연체집계_연체최종정보 = pd.merge(df_연체정보_기관별_연체날짜별_집계, self.df_연체정보_최종, on=["JOIN_SN","JOIN_SN_TYP"], how="inner")
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
        self.df_연체정보_독립변수_집계 = df_연체집계_연체최종정보_필터링.groupby(["JOIN_SN","JOIN_SN_TYP"]).agg(
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
            self.df_연체정보.sort_values(['YM'],ascending=True)  # 월 오름차순 정렬
            .groupby(['JOIN_SN',"JOIN_SN_TYP", 'COM_SN'], as_index=False)  # 그룹핑
            .tail(1)  # 각 그룹에서 마지막 row (최신 월)
        )

        self.df_신용카드정보_독립변수_집계 = df_신용카드개설정보_최종.groupby(["JOIN_SN","JOIN_SN_TYP"]).agg(
            신용카드개수=("COM_SN", "count")
        ).reset_index()

        # -------------------------------------------------------------------------------------------------------------
        # 6) 소비성향 독립변수 생성 (신용카드 한도대비 사용금액) - 연체이전 3개월 사용률(1), 연체이전 9개월전~3개월전 사용률(2), (2)->(1) 증가율 
        # ------------------------------------------------------------------------------------------------------------
        df_신용카드이용정보_소비성향 = self.df_신용카드이용정보.copy()

        df_신용카드이용정보_소비성향['신용카드한도대비사용률'] = np.where(df_신용카드이용정보_소비성향['CD_USG_LMT'] == 0, 0, df_신용카드이용정보_소비성향['CD_USG_AMT']/df_신용카드이용정보_소비성향['CD_USG_LMT'])
        df_신용카드이용정보_소비성향['현금서비스한도대비사용률'] = np.where(df_신용카드이용정보_소비성향['CD_CA_LMT'] == 0, 0, df_신용카드이용정보_소비성향['CD_CA_AMT']/df_신용카드이용정보_소비성향['CD_CA_LMT'])

        #====가중평균 사용률 계산=============================================================================================================================================================================# 
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
        #===============================================================================================================================================================================================#
        df_연체정보_소비성향 = self.df_연체정보_최종.copy()

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

        # 최근 3개월: 개월차 1,2,3
        cond_recent = df_신용카드이용정보_연체정보_소비성향["개월차"].between(1, 3)
        # 3~6개월 전: 개월차 4,5,6
        cond_past = df_신용카드이용정보_연체정보_소비성향["개월차"].between(4, 9)

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

        self.df_소비성향_독립변수_집계 = pd.merge(result_usg,result_ca,on=(["JOIN_SN","JOIN_SN_TYP"]), how="outer")

    ####################################################################
    #  함수 : 대출정보 + 독립변수 결합
    ####################################################################
    def merge_loan_variable(self) :
        self.df_예측모델링 = self.df_대출정보.merge(self.df_대출정보_독립변수_집계, on=['JOIN_SN','JOIN_SN_TYP'], how="left").merge(self.df_지급능력_독립변수_집계, on=['JOIN_SN','JOIN_SN_TYP'], how="left").merge(self.df_보험계약정보_독립변수_집계, on=['JOIN_SN','JOIN_SN_TYP'], how="left").merge(self.df_연체정보_독립변수_집계, on=['JOIN_SN','JOIN_SN_TYP'], how="left").merge(self.df_신용카드정보_독립변수_집계, on=['JOIN_SN','JOIN_SN_TYP'], how="left").merge(self.df_소비성향_독립변수_집계, on=['JOIN_SN','JOIN_SN_TYP'], how="left")



    ####################################################################
    #  함수 : 결측치처리 - common_util.py로 추후 이동 
    ####################################################################
    def fill_missing_skewed_by_loan_code(self, target_col, min_val, max_val, dtype='float'):
        df = self.df_예측모델링

        # 1. 결측치 위치 파악
        mask = df[target_col].isna()
        if not mask.any():
            return

        # 2. LN_KEY 생성 (필요한 행만)
        ln_cd_columns = ['LN_CD_1', 'LN_CD_2', 'LN_CD_3']
        df_missing = df.loc[mask].copy()
        df_missing['__LN_KEY__'] = df_missing[ln_cd_columns].astype(str).agg('-'.join, axis=1)

        # 3. LN_KEY별 a, b 파라미터 생성
        unique_keys = df_missing['__LN_KEY__'].unique()
        ab_map = {
            key: (
                round(np.random.uniform(1.0, 2.5), 2),
                round(np.random.uniform(6.5, 8.0), 2)
            )
            for key in unique_keys
        }

        # 4. LN_KEY별로 샘플링
        def generate_scaled_beta(key, size):
            a, b = ab_map[key]
            samples = stats.beta.rvs(a, b, size=size)
            scaled = min_val + samples * (max_val - min_val)
            return np.round(scaled).astype(int) if dtype == 'int' else scaled

        # 5. apply vectorized beta sampling by group
        df_missing['__index__'] = df_missing.index  # index 복원용
        grouped = df_missing.groupby('__LN_KEY__')['__index__'].apply(list).to_dict()

        allindices = []
        allvalues = []

        for key, indices in grouped.items():
            size = len(indices)
            values = generate_scaled_beta(key, size)
            allindices.extend(indices)
            allvalues.extend(values)

        # 6. 결측치 채우기
        df.loc[allindices, target_col] = allvalues

        # 7. 정리
        self.df_예측모델링 = df

    ####################################################################
    #  함수 : dlq(연체여부 생성) - common_util.py로 추후 이동 
    ####################################################################
    def generate_dlq_by_loan_type(self, df: pd.DataFrame, features: list, seed: int = 42, beta_scale: float = 1.5):
        np.random.seed(seed)
        
        # 1. 대출코드 조합 생성
        df['__LN_KEY__'] = df[['LN_CD_1','LN_CD_2','LN_CD_3']].astype(str).agg('-'.join, axis=1)
        ln_keys = df['__LN_KEY__'].unique()
        
        # 2. feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        
        # 3. 대출조합별로 서로 다른 β벡터 설정
        beta_map = {
            key: np.random.randn(len(features)) * beta_scale
            for key in ln_keys
        }

        # 4. 연체확률 계산
        probs = np.zeros(len(df))
        for key in ln_keys:
            mask = df['__LN_KEY__'] == key
            beta = beta_map[key]
            logits = X_scaled[mask] @ beta
            probs[mask] = expit(logits)

        # 5. 연체여부 생성
        dlq = np.random.binomial(1, probs)

        # 6. 결과 저장
        df['DLQ_Prob'] = probs
        df['DLQ_YN'] = dlq

        # 8. 정리
        df.drop(columns='__LN_KEY__', inplace=True)

    ####################################################################
    #  함수 : 결측치처리 - handle_missing_values()
    ####################################################################
    def handle_missing_values(self) :

        # 대출건수 결측치 대체 
        min_val = self.df_예측모델링['대출건수'].min()
        max_val = self.df_예측모델링['대출건수'].max()

        print(min_val)
        print(max_val)

        self.fill_missing_skewed_by_loan_code('대출건수',min_val,max_val,dtype='int')

        # 대출금액합 결측치 대체 
        min_val = self.df_예측모델링['대출금액합'].min()
        max_val = self.df_예측모델링['대출금액합'].max()

        print(min_val)
        print(max_val)

        self.fill_missing_skewed_by_loan_code('대출금액합',min_val,max_val,dtype='int')

        # 지급능력 결측치 대체 
        min_val = self.df_예측모델링['지급능력'].min()
        max_val = self.df_예측모델링['지급능력'].max()

        print(min_val)
        print(max_val)

        self.fill_missing_skewed_by_loan_code('지급능력',min_val,max_val,dtype='int')

        # 보험건수 결측치 대체 
        min_val = self.df_예측모델링['보험건수'].min()
        max_val = self.df_예측모델링['보험건수'].max()

        print(min_val)
        print(max_val)

        self.fill_missing_skewed_by_loan_code('보험건수',min_val,max_val,dtype='int')

        # 보험월납입액 결측치 대체 
        min_val = self.df_예측모델링['보험월납입액'].min()
        max_val = self.df_예측모델링['보험월납입액'].max()

        print(min_val)
        print(max_val)

        self.fill_missing_skewed_by_loan_code('보험월납입액',min_val,max_val,dtype='int')

        # 신용카드개수 결측치 대체 
        min_val = self.df_예측모델링['신용카드개수'].min()
        max_val = self.df_예측모델링['신용카드개수'].max()

        print(min_val)
        print(max_val)

        self.fill_missing_skewed_by_loan_code('신용카드개수',min_val,max_val,dtype='int')

        # 장기연체건수 결측치 대체
        min_val = self.df_예측모델링['장기연체건수'].min()
        max_val = self.df_예측모델링['장기연체건수'].max()

        print(min_val)
        print(max_val)

        self.fill_missing_skewed_by_loan_code('장기연체건수',min_val,max_val,dtype='int')

        # 연체금액합 결측치 대체
        min_val = self.df_예측모델링['연체금액합'].min()
        max_val = self.df_예측모델링['연체금액합'].max()

        print(min_val)
        print(max_val)

        self.fill_missing_skewed_by_loan_code('연체금액합',min_val,max_val,dtype='int')

        # 연체건수 결측치 대체
        min_val = self.df_예측모델링['연체건수'].min()
        max_val = self.df_예측모델링['연체건수'].max()

        print(min_val)
        print(max_val)

        self.fill_missing_skewed_by_loan_code('연체건수',min_val,max_val,dtype='int')

        #장기고액대출건수 결측치 대체
        values = [0, 1, 2, 3]
        general_probs = [0.966, 0.025, 0.005, 0.004]
        self.df_예측모델링['장기고액대출건수'] = np.random.choice(values, size=len(self.df_예측모델링), p=general_probs)

        #최근3개월_신용카드사용률
        values = [0.23, 0.45, 0.67, 0.72]
        general_probs = [0.966, 0.025, 0.005, 0.004]

        self.df_예측모델링['최근3개월_신용카드사용률'] = np.random.choice(values, size=len(self.df_예측모델링), p=general_probs)

        ##9개월_3개월이전_신용카드사용률
        values = [0.23, 0.45, 0.67, 0.72]
        general_probs = [0.966, 0.025, 0.005, 0.004]

        self.df_예측모델링['9개월_3개월이전_신용카드사용률'] = np.random.choice(values, size=len(self.df_예측모델링), p=general_probs)

        #최근3개월_현금서비스사용률
        values = [0, 0.02, 0.03, 0.04]
        general_probs = [0.966, 0.025, 0.005, 0.004]

        self.df_예측모델링['최근3개월_현금서비스사용률'] = np.random.choice(values, size=len(self.df_예측모델링), p=general_probs)

        #9개월_3개월이전_현금서비스사용률
        values = [0, 0.02, 0.03, 0.04]
        general_probs = [0.966, 0.025, 0.005, 0.004]

        self.df_예측모델링['9개월_3개월이전_현금서비스사용률'] = np.random.choice(values, size=len(self.df_예측모델링), p=general_probs)

        self.df_예측모델링['신용카드_사용률_증가량'] = self.df_예측모델링['최근3개월_신용카드사용률'] - self.df_예측모델링['9개월_3개월이전_신용카드사용률']
        self.df_예측모델링['현금서비스_사용률_증가량'] = self.df_예측모델링['최근3개월_현금서비스사용률'] - self.df_예측모델링['9개월_3개월이전_현금서비스사용률']

        features = [
            '대출건수', '장기고액대출건수', '대출금액합', '지급능력',
            '보험건수', '보험월납입액', '연체건수', '장기연체건수', '연체금액합',
            '신용카드개수', '신용카드_사용률_증가량', '현금서비스_사용률_증가량'
        ]    

        self.generate_dlq_by_loan_type(self.df_예측모델링, features)
        # 지급능력 구간화 
        self.df_예측모델링['지급능력_구간'] = pd.qcut(self.df_예측모델링['지급능력'], q=10, labels=False, duplicates='drop')

    ############################################################
    #  함수 : 금융권+대부업권 대출정보 병합
    #  1) 금융권+대부업권 대출정보 병합 후 기업대출정보와 동일 컬럼기준으로 병합
    #  2) 차주+대출정보 병합
    #  3) 신용도판단정보+공공정보 연체정보 병합
    #  4) 대출정보+신용카드이용정보 통합 
    #  5) 대출신용카드이용정보 + 연체정보 통합 - (차주번호, 대출신용카드기관번호)를 기준으로 매칭 version1
    #  6) 차주+연체정보 통합 - 기관번호로 매칭하면 연체건이 누락되는 현상으로 인해 차주와 연체율 단순 매칭 version2
    #  7) 차주(정책금융 이용차주 한정)+연체정보 통합 - version3
    #  8) 차주 + 대출신용카드연체정보 통합
    ############################################################
    def merge_loan_credit(self):
        ################################
        # 0. 결측치파악 => 결측치 존재하지 않음 
        ################################

        #대출정보(병합대상)
        print(self.df_개인대출정보_금융권.info())
        print(self.df_개인대출정보_대부업권.info())
        print(self.df_기업대출정보.info())

        #연체정보(병합대상)
        print(self.df_연체정보_신용도판단정보.info())
        print(self.df_연체정보_공공정보.info())

        print(self.df_신용카드개설정보.info())

        # ==============================
        # 1. 병합
        # ==============================

        # ---------------------------
        # 1-1) 금융권+대부업권 대출정보 병합 
        # ---------------------------
        self.df_개인대출정보 = pd.concat([self.df_개인대출정보_금융권, self.df_개인대출정보_대부업권], ignore_index=True)
        # 구분컬럼추가 
        self.df_개인대출정보["기업개인구분"] = "개인"
        self.df_기업대출정보["기업개인구분"] = "기업"
        # 대출과목코드 일원화(1.개인대출상품코드 스트링 붙이기(LN_CD_1+LN_CD_2+LN_CD_3),2.기업대출과목코드를 개인대출코드로 바꿈)
        self.df_개인대출정보["LN_CD_1"] = self.df_개인대출정보["LN_CD_1"].astype('str')
        self.df_개인대출정보["LN_CD_2"] = self.df_개인대출정보["LN_CD_2"].astype('str')
        self.df_개인대출정보["LN_CD_3"] = self.df_개인대출정보["LN_CD_3"].astype('str')
        self.df_개인대출정보["LN_CD"] = self.df_개인대출정보["LN_CD_1"]+self.df_개인대출정보["LN_CD_2"]+self.df_개인대출정보["LN_CD_3"]
        self.df_기업대출정보.rename(columns={'LN_ACCT_CD': 'LN_CD'}, inplace=True)

        # 겹치는 컬럼 파악 
        common_cols = list(self.df_개인대출정보.columns.intersection(self.df_기업대출정보.columns))
        # 해당 컬럼기준으로 개인대출정보+기업대출정보 병합 
        df_대출정보 = pd.concat([self.df_개인대출정보[common_cols], self.df_기업대출정보[common_cols]], ignore_index=True)

        print(df_대출정보.info())
        # 자료의 범위
        print(df_대출정보['YM'].min(),df_대출정보['YM'].max())

        # ---------------------------
        # 1-2) 차주+대출집계정보 병합
        # ---------------------------

        # 대출정보_집계
        self.df_대출정보_보유기관_집계 = self.df_대출정보.groupby(['JOIN_SN','YM'])['COM_SN'].count().reset_index(name='COM_SN_COUNT')
        self.df_대출정보_대출과목_집계 = self.df_대출정보.groupby(['JOIN_SN','YM'])['LN_CD'].count().reset_index(name='LN_CD_COUNT')

        # 차주의 월별 집계치 중 min(), max() 건ß수를 기준으로 새로운 데이터 프레임 생성 
        self.df_대출정보_보유기관_집계=self.df_대출정보_보유기관_집계.groupby('JOIN_SN')['COM_SN_COUNT'].agg(최소보유건수='min', 최대보유건수='max').reset_index()
        self.df_대출정보_대출과목_집계=self.df_대출정보_대출과목_집계.groupby('JOIN_SN')['LN_CD_COUNT'].agg(최소대출건수='min', 최대대출건수='max').reset_index()

        # 집계정보통합 
        self.df_대출정보_집계 = pd.merge(self.df_대출정보_보유기관_집계, self.df_대출정보_대출과목_집계, on='JOIN_SN', how='inner')
        self.df_대출정보_집계.head(100)

        self.df_차주_기업개요정보.head(100)

        # 차주+집계정보 병합
        self.df_차주_대출정보_집계 = self.df_차주.merge(self.df_대출정보_집계,on='JOIN_SN',how='left')


        self.df_차주_대출정보_집계[['최소보유건수','최대보유건수','최소대출건수','최대대출건수']] = self.df_차주_대출정보_집계[['최소보유건수','최대보유건수','최소대출건수','최대대출건수']].fillna(0)
        ####################차주+대출정보 집계 테이블 완성#######################

        # ---------------------------
        # 1-3) 신용도판단정보+공공정보 병합 
        # ---------------------------

        self.df_연체정보_신용도판단정보["민간공공구분"] = "민간"
        self.df_연체정보_공공정보["민간공공구분"] ="공공"
        # 연체정보 통합 
        self.df_연체정보 = pd.concat([self.df_연체정보_신용도판단정보, self.df_연체정보_공공정보], ignore_index=True)


        # 차주, 기관별 마지막 등록월의 연체정보만을 가져온다. (기관까지 매칭하는 버전)
        df_연체정보_기관별_최종 = (
            self.df_연체정보
            #[df_연체정보['DLQ_RGST_AMT'] > 0]  # 연체금액 있는 행만
            .loc[lambda x: x.groupby(['JOIN_SN','COM_SN'])['YM'].idxmax()]    # 차주별 가장 최근 YM
            .reset_index(drop=True)
        )

        # 차주 마지막 등록월의 연체정보만을 가져온다. (simple 버전)
        df_연체정보_최종 = (
            self.df_연체정보
            #[df_연체정보['DLQ_RGST_AMT'] > 0]  # 연체금액 있는 행만
            .loc[lambda x: x.groupby(['JOIN_SN'])['YM'].idxmax()]    # 차주별 가장 최근 YM
            .reset_index(drop=True)
        )

        # --------------------------------------------------------------------
        # 1-4) 대출정보+신용카드이용정보 통합
        # --------------------------------------------------------------------
        self.df_대출정보.info()
        self.df_신용카드이용정보.info()

        # 대출정보와 신용카드 이용정보의 금액을 같은이름으로 통일 
        df_대출정보_AMT= self.df_대출정보.rename(columns={'LN_AMT': 'AMT'})
        df_신용카드이용정보_AMT= self.df_신용카드이용정보.copy()
        df_신용카드이용정보_AMT['AMT'] = self.df_신용카드이용정보['CD_USG_AMT']+ self.df_신용카드이용정보['CD_CA_AMT'] # 신용카드 이용금액+신용카드론 이용금액 합산

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
        self.df_대출_신용카드_병합_기관별_최종  = (
            df_대출_신용카드_병합.sort_values(['YM'],ascending=True)  # 월 오름차순 정렬
            .groupby(['JOIN_SN', 'COM_SN', '대출신용카드구분'], as_index=False)  # 그룹핑
            .tail(1)  # 각 그룹에서 마지막 row (최신 월)
        )

        # ---------------------------------------------------------------------------------
        # 1-5) 대출신용카드이용정보 + 연체정보 통합 - (차주번호, 대출신용카드기관번호)를 기준으로 매칭 version1
        # ---------------------------------------------------------------------------------
        self.df_대출_신용카드_병합_기관별_최종.info()
        self.df_연체정보_기관별_최종.info()

        # (주의) 대출기관=신용카드 기관 같은경우 (카드대출을 받은경우는 중복으로 행이 두개가 들어가므로, 기관이 같은 경우에는 중복행을 대출만 남기고 신용을 제거한다.)
        # 이를 통해 중복기관을 없애서 정확한 기관별 연체여부를 파악한다.
        df_대출_신용카드_병합_기관별_최종_sorted = self.df_대출_신용카드_병합_기관별_최종.sort_values(['JOIN_SN','JOIN_SN_TYP','COM_SN','대출신용카드구분']) 
        self.df_대출_신용카드_병합_기관별_최종 = df_대출_신용카드_병합_기관별_최종_sorted.groupby(['JOIN_SN','JOIN_SN_TYP','COM_SN'], as_index=False).head(1)

        self.df_대출_신용카드_병합_기관별_최종.info()

        # 연체정보 2,630 건이 전체 매칭되지 않는다. (연체등록은 대출시점 이후에 발생하므로, 2018.06~2020.06 같은기간으로 보면 연체정보에 있는 대출기관이 대출신용카드정보에서 누락될 수 있기때문으로 보임)
        self.df_대출_신용카드_연체정보 = pd.merge(self.df_대출_신용카드_병합_기관별_최종,self.df_연체정보_기관별_최종[['JOIN_SN','JOIN_SN_TYP','COM_SN','DLQ_RGST_DT','DLQ_RGST_AMT','DLQ_AMT']],on=['JOIN_SN','JOIN_SN_TYP','COM_SN'],how='left')
        self.df_대출_신용카드_연체정보.info()

        # 누락된 연체정보 파악 
        filter_keys = self.df_대출_신용카드_연체정보[~self.df_대출_신용카드_연체정보['DLQ_RGST_DT'].isna()][['JOIN_SN','COM_SN']].apply(tuple,axis=1)

        self.df_연체정보_기관별_최종[df_연체정보_기관별_최종[['JOIN_SN','COM_SN']].apply(tuple,axis=1).isin(filter_keys)]  # 포함 연체정보 
        self.df_연체정보_기관별_최종[~df_연체정보_기관별_최종[['JOIN_SN','COM_SN']].apply(tuple,axis=1).isin(filter_keys)] # 누락 연체정보 
        ## ex) 808 차주의 5612646(COM_SN) 정보의 경우, df_대출_신용카드_병합_기관별_최종에 포함되어있지 않음 
        self.df_대출정보[self.df_대출정보['JOIN_SN']== 808]
        self.df_신용카드이용정보[self.df_신용카드이용정보['JOIN_SN'] == 808]
        self.df_대출_신용카드_병합_기관별_최종[self.df_대출_신용카드_병합_기관별_최종['JOIN_SN']== 808]

        # 중복정보 파악 - 중복정보 없음 
        self.df_대출_신용카드_연체정보[self.df_대출_신용카드_연체정보.groupby(['YM','JOIN_SN','JOIN_SN_TYP','COM_SN'])['JOIN_SN'].transform('count') > 1]


        # -----------------------------------------------------------------------------------------------------------------------
        # 1-6) 차주+연체정보 통합 - 기관번호로 매칭하면 연체건이 누락되는 현상으로 인해 차주와 연체율 단순 매칭 version2
        # -----------------------------------------------------------------------------------------------------------------------
        self.df_차주_연령구간화.info()

        self.df_차주_연체정보 = pd.merge(self.df_차주_연령구간화,self.df_연체정보_최종[['JOIN_SN','JOIN_SN_TYP','COM_SN','DLQ_RGST_DT','DLQ_RGST_AMT','DLQ_AMT']],on=['JOIN_SN','JOIN_SN_TYP'],how='left')
        self.df_차주_연체정보.info()


        # -----------------------------------------------------------------------------------------------------------------------
        # 1-7) 차주(정책금융 이용차주 한정)+연체정보 통합 - version3
        # -----------------------------------------------------------------------------------------------------------------------
        self.df_차주_연령구간화.info()

        self.df_차주_연령구간화_정책금융이용= self.df_차주_연령구간화[self.df_차주_연령구간화['JOIN_SN'].isin(self.df_개인대출정보[self.df_개인대출정보['LN_CD_3']!=0]['JOIN_SN'])]

        self.df_차주_정책금융이용_연체정보 = pd.merge(self.df_차주_연령구간화_정책금융이용,self.df_연체정보_최종[['JOIN_SN','JOIN_SN_TYP','COM_SN','DLQ_RGST_DT','DLQ_RGST_AMT','DLQ_AMT']],on=['JOIN_SN','JOIN_SN_TYP'],how='left')
        self.df_차주_정책금융이용_연체정보.info()

        # --------------------------------------------------------------------
        # 1-8) 차주 + 대출신용카드연체정보 통합
        # --------------------------------------------------------------------
        self.df_차주_대출_신용카드_연체정보 = pd.merge(self.df_차주_연령구간화, self.df_대출_신용카드_연체정보,on=['JOIN_SN','JOIN_SN_TYP'],how='left')

        self.df_차주_대출_신용카드_연체정보.info()

        self.df_차주_대출_신용카드_연체정보[['YM','SCTR_CD','COM_SN','IS_ME','AMT','DLQ_RGST_AMT','DLQ_AMT']] = self.df_차주_대출_신용카드_연체정보[['YM','SCTR_CD','COM_SN','IS_ME','AMT','DLQ_RGST_AMT','DLQ_AMT']].fillna(0)
        self.df_차주_대출_신용카드_연체정보['대출신용카드구분'].fillna('미보유',inplace=True)


  