import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd 

from common_code import CommonCode

# -------------------------------------------------------------
#  Condition Section
# -------------------------------------------------------------

def main_condition():
    st.title("비교조건 선택")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("기준 대출")
        loan_type1 = st.selectbox("대출유형을 선택하세요", ["기업대출", "개인대출"],key=1)

        기준_대출 = {
            '대출과목': loan_condition(loan_type1,2),
            '지급능력_구간': st.selectbox("지급능력_구간", options=CommonCode.지급능력.items()),
            '대출금액': st.slider("대출금액", 1000, 100000, 20000, step=1000),
            '금리': st.slider("금리(%)", 1.0, 15.0, 5.0)
        }

    with col2:
        st.subheader("비교 대출")
        loan_type2 = st.selectbox("대출유형을 선택하세요", ["기업대출", "개인대출"],key=3)

        비교_대출 = {
            '대출과목': loan_condition(loan_type2,4),
            '지급능력_구간': st.selectbox("소득구지급능력_구간간", options=CommonCode.지급능력.items()),
            '대출금액': st.slider("대출금액", 1000, 100000, 30000, step=1000, key="비교3"),
            '금리': st.slider("금리(%)", 1.0, 15.0, 7.0, key="비교4")
        }

    변수리스트 = ['대출건수', '장기고액대출건수', '대출금액합', '지급능력', '보험건수', '보험월납입액', '연체건수', '장기연체건수', '연체금액합', '신용카드개수', '신용카드_사용률_증가량', '현금서비스_사용률_증가량']
    선택변수 = st.multiselect("분석할 변수 선택", 변수리스트, default=변수리스트[:5])


def loan_condition(loan_type,key_type):
    if loan_type == "기업대출":
        loan_name = st.selectbox("기업대출 과목 선택", options=CommonCode.LN_ACCT_CD.items(),
                            format_func=lambda x: f"{x} - {CommonCode.LN_ACCT_CD[x]}",key=key_type)
        st.success(f"선택된 대출: {loan_name}")

    elif loan_type == "개인대출":
        loan_name_1 = st.selectbox("LN_CD_1 (개인대출 유형)", options=CommonCode.LN_CD_1.items(),
                            format_func=lambda x: f"{x} - {CommonCode.LN_CD_1[x]}",key=key_type)
        
        if loan_name_1 == "개인대출":
            loan_name_2_options = [k for k in CommonCode.LN_CD_2.items() if int(k) >= 100]
            loan_name_3_options = CommonCode.LN_CD_3.items()

            loan_name_2 = st.selectbox("LN_CD_2 (대출 상세유형)", options=loan_name_2_options,
                                format_func=lambda x: f"{x} - {CommonCode.LN_CD_2.get(x, 'N/A')}",key=key_type)
            
            loan_name_3 = st.selectbox("LN_CD_3 (정책 대출 유형)", options=loan_name_3_options,
                                format_func=lambda x: f"{x} - {CommonCode.LN_CD_3.get(x, 'N/A')}",key=key_type)
            
            st.success(f"선택된 대출: {loan_name_1}, {loan_name_2}, {loan_name_3}")
        elif loan_name_1 in ["장기카드대출(카드론)", "단기카드대출(현금서비스)"]:
            st.success(f"선택된 코드: {loan_name_1}")
        
        
