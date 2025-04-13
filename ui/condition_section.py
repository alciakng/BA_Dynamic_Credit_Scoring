import logging
import pprint
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd 
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from common_code import CommonCode

from controllers.data_builder import DatasetBuilder
from controllers.data_visualizer import DataVisualizer

from ui.scoring_section import run_scoring

# -------------------------------------------------------------
#  Condition Section
# -------------------------------------------------------------

def main_condition(builder,visualizer):
    st.title("비교조건 선택")

    col1, col2 = st.columns(2)

    # 📅 월 리스트 생성
    month_options = generate_month_list()

    # 📍 공통 조회 조건 (위에 삽입)
    st.markdown("### 📅 조회기간 선택")

    col_start, col_end = st.columns(2)
    with col_start:
        조회시작년월 = st.selectbox("조회 시작년월", month_options, index=0)
    with col_end:
        조회종료년월 = st.selectbox("조회 종료년월", month_options, index=len(month_options) - 1)

    # 📌 유효성 검사
    if 조회시작년월 > 조회종료년월:
        st.error("⛔ 조회 시작년월은 종료년월보다 이전이어야 합니다.")

    with col1:
        st.subheader("기준 대출")
        loan_type1 = st.selectbox("대출유형을 선택하세요", ["기업대출", "개인대출"],key="기준대출유형")

        기준_대출 = {
            '대출과목': loan_condition(loan_type1,"기준대출과목"),
            '지급능력_구간': st.selectbox("지급능력_구간", options=CommonCode.지급능력.items(),format_func=lambda x: f"{x} - {CommonCode.지급능력[x]}"),
            '대출금액(천원)': st.slider("대출금액", 1000, 1000000, 20000, step=1000, key="기준대출금액"),
            '금리': st.slider("금리(%)", 1.0, 15.0, 5.0)
        }

    with col2:
        st.subheader("비교 대출")
        loan_type2 = st.selectbox("대출유형을 선택하세요", ["기업대출", "개인대출"],key="비교대출유형")

        비교_대출 = {
            '대출과목': loan_condition(loan_type2,"비교대출과목"),
            '지급능력_구간': st.selectbox("소득구지급능력_구간간", options=CommonCode.지급능력.items(),format_func=lambda x: f"{x} - {CommonCode.지급능력[x]}"),
            '대출금액(천원)': st.slider("대출금액", 1000, 1000000, 30000, step=1000, key="비교대출금액"),
            '금리': st.slider("금리(%)", 1.0, 15.0, 7.0, key="비교4")
        }

    변수리스트 = ['대출건수', '장기고액대출건수', '대출금액합', '지급능력', '보험건수', '보험월납입액', '연체건수', '장기연체건수', '연체금액합', '신용카드개수', '신용카드_사용률_증가량', '현금서비스_사용률_증가량']
    선택변수 = st.multiselect("분석할 변수 선택", 변수리스트, default=변수리스트[:5])

    st.markdown("""
    <style>
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Scoring"):
        with st.spinner("⏳ 처리 중입니다... 잠시만 기다려주세요!"):
            run_scoring(builder, visualizer, 기준_대출, 비교_대출, 선택변수, 조회시작년월, 조회종료년월)
        st.toast("작업이 완료되었습니다!")
        


def loan_condition(loan_type,key_type):

    loan_selection = {
        "LN_CD_1": None,
        "LN_CD_2": None,
        "LN_CD_3": None
    }

    if loan_type == "기업대출":
        com_loan = st.selectbox("기업대출 과목 선택", options=list(CommonCode.LN_ACCT_CD.items()),
                            format_func=lambda x: f"{x[0]} - {x[1]}",key=key_type)
        
        st.success(f"선택된 대출: {pprint.pformat(com_loan)}")
        # 선택배정1
        loan_selection["LN_CD_1"] = com_loan[0]

        return loan_selection

    elif loan_type == "개인대출":
        loan_1 = st.selectbox("LN_CD_1 (개인대출 유형)", options=list(CommonCode.LN_CD_1.items()),
                            format_func=lambda x: f"{x[0]} - {x[1]}",key=key_type)
        
        logging.info("개인대출 선택과목 : " +pprint.pformat(loan_1))
        # 선택배정1
        loan_selection["LN_CD_1"] = loan_1[0]

        if loan_1[0] == "0031": # 개인대출 
            loan_2_options = [
                p for  p in list(CommonCode.LN_CD_2.items())
                if str(p[0]) != "0"
            ]
            
            loan_3_options = list(CommonCode.LN_CD_3.items())

            loan_2 = st.selectbox("LN_CD_2 (대출 상세유형)", options=loan_2_options,
                                format_func=lambda x: f"{x[0]} - {x[1]}",key=key_type+"1")
            
            loan_3 = st.selectbox("LN_CD_3 (정책 대출 유형)", options=loan_3_options,
                                format_func=lambda x: f"{x[0]} - {x[1]}",key=key_type+"2")
            # 선택배정 2,3
            loan_selection["LN_CD_2"] = loan_2[0]
            loan_selection["LN_CD_3"] = loan_3[0]
            
            st.success(f"선택된 대출: {loan_1[1]}, {loan_2[1]}, {loan_3[1]}")
        elif loan_1[1] in ["장기카드대출(카드론)", "단기카드대출(현금서비스)"]:
            loan_selection["LN_CD_2"] = list(CommonCode.LN_CD_2.items())[0][0] # 카드대출
            loan_selection["LN_CD_3"] = list(CommonCode.LN_CD_2.items())[0][0] # 일반대출
            st.success(f"선택된 대출: {loan_1[1]}, {CommonCode.LN_CD_2.get('0')},{CommonCode.LN_CD_3.get('0')}")
        
        return loan_selection

# 📅 조회 가능 월 리스트 생성
def generate_month_list(start='201806', end='202006'):
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    start_date = datetime.strptime(start, "%Y%m")
    end_date = datetime.strptime(end, "%Y%m")
    
    months = []
    current = start_date
    while current <= end_date:
        months.append(current.strftime("%Y%m"))
        current += relativedelta(months=1)
    return months

