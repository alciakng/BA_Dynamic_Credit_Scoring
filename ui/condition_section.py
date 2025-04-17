from datetime import time
import logging
import pprint
import shap
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd 
import lightgbm as lgb
from io import StringIO
import sys

from sklearn.model_selection import train_test_split
from common_code import CommonCode

from controllers.data_builder import DatasetBuilder
from controllers.data_visualizer import DataVisualizer

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

from ui.scoring_section import run_scoring
from ui.dashboard_section import generate_variable_report_aggrid
from ui.dashboard_section import plot_adjusted_proba_threshold_plotly

# -------------------------------------------------------------
#  Condition Section
# -------------------------------------------------------------

def main_condition(builder,visualizer):
    st.title("비교조건 선택")

    col1, col2 = st.columns(2)

    # 월 리스트 생성
    month_options = generate_month_list()

    # 공통 조회 조건 (위에 삽입)
    st.markdown("###  조회기간 선택")

    col_start, col_end = st.columns(2)
    with col_start:
        조회시작년월 = st.selectbox("조회 시작년월", month_options, index=0)
    with col_end:
        조회종료년월 = st.selectbox("조회 종료년월", month_options, index=len(month_options) - 1)

    # 유효성 검사
    if 조회시작년월 > 조회종료년월:
        st.error("조회 시작년월은 종료년월보다 이전이어야 합니다.")

    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 5rem;
        padding-right: 5rem;
        max-width: 100%;
    }
    .element-container {
        padding-left: 0px !important;
        padding-right: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with col1:
        st.subheader("기준 대출")
        loan_type1 = st.selectbox("대출유형을 선택하세요", ["개인대출", "기업대출"],key="기준대출유형")

        기준_대출 = {
            '대출과목': loan_condition(loan_type1,"기준대출과목"),
            '지급능력_구간': st.selectbox("지급능력_구간", options=CommonCode.지급능력.items(),format_func=lambda x: f"{x} - {CommonCode.지급능력[x]}",key="기준_지급구간"),
            '대출금액(천원)': st.slider("대출금액(천원)", 1000, 1000000, 20000, step=1000, key="기준대출금액"),
            '금리': st.slider("금리(%)", 1.0, 15.0, 5.0)
        }

    with col2:
        st.subheader("비교 대출")
        loan_type2 = st.selectbox("대출유형을 선택하세요", ["개인대출", "기업대출"],key="비교대출유형")

        비교_대출 = {
            '대출과목': loan_condition(loan_type2,"비교대출과목"),
            '지급능력_구간': st.selectbox("지급능력_구간", options=CommonCode.지급능력.items(),format_func=lambda x: f"{x} - {CommonCode.지급능력[x]}",key="비교_지급구간"),
            '대출금액(천원)': st.slider("대출금액(천원)", 1000, 1000000, 30000, step=1000, key="비교대출금액"),
            '금리': st.slider("금리(%)", 1.0, 15.0, 7.0, key="비교4")
        }


    변수리스트 = ['대출건수', '장기고액대출건수', '대출금액합', '지급능력', '보험건수', '보험월납입액', '연체건수', '장기연체건수', '연체금액합', '신용카드개수', '신용카드_사용률_증가량', '현금서비스_사용률_증가량']
    선택변수 = st.multiselect("분석할 변수 선택", 변수리스트, default=변수리스트[:5])

    # 선택변수 셋팅처리 
    builder.선택변수 = 선택변수

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
        # 세션상태 업데이트
        st.session_state['based_scored'] = True
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

# 조회 가능 월 리스트 생성
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




def dynamic_condition(builder: DatasetBuilder,visualizer : DataVisualizer):

    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 5rem;
        padding-right: 5rem;
        max-width: 100%;
    }
    .element-container {
        padding-left: 0px !important;
        padding-right: 0px !important;
    }
    /* 탭 영역 전체 스타일 */
    div[data-baseweb="tab-list"] {
        font-size: 20px !important;    /* 글자 크기 */
        height: 60px;                  /* 탭 높이 */
    }

    /* 각 탭 버튼의 스타일 */
    button[role="tab"] {
        padding: 12px 24px !important;  /* 내부 여백 */
        font-size: 18px !important;     /* 글자 크기 */
        font-weight: bold !important;
        color: white !important;
        background-color: #1c1c1c !important;
        border-radius: 8px !important;
        margin-right: 10px !important;
    }

    /* 선택된 탭 스타일 */
    button[aria-selected="true"] {
        background-color: #0e76a8 !important; /* 선택된 탭 배경색 */
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    

    탭1, 탭2, 탭3= st.tabs(["Step1. 요약 리포트", "Step2. 대상 데이터 확인", "Step3. Dynamic Credit Scoring"])

    # 첫 번째 탭: 요약레포트 
    with 탭1:
        st.markdown("### Step1. 요약리포트")
        generate_variable_report_aggrid(builder.df_summary_wo,"요약리포트")

    # 두 번째 탭: 대상데이터 확인 
    with 탭2:
        st.markdown("### Step2. 대상 데이터 확인")

        with st.expander("조회조건 설정", expanded=True):
            col1, col2= st.columns(2)
            dlq_type = col1.selectbox("실제 연체여부", ["여", "부"])
            pred_dlq_type = col2.selectbox("예측 연체여부", ["여", "부"])
            min_dlq_rate, max_dlq_rate = st.slider("추정 연체확률 범위 설정", 0.0, 1.0, (0.3, 0.6), step=0.01)
            threshold_moderator = st.slider("임계구간 설정(threshold moderator)", 0.0, 0.1, 0.05, step=0.01)

        filtered_df = builder.df_css_base.copy()

        # 실제 연체여부
        if dlq_type == "여": # 필터조건
            filtered_df = filtered_df[filtered_df["실제_연체여부"] == 1]  
        else : 
            filtered_df = filtered_df[filtered_df["실제_연체여부"] == 0]

        # 예측 연체여부
        if pred_dlq_type == "여": # 필터조건
            filtered_df = filtered_df[filtered_df["예측_연체여부"] == 1]  
        else : 
            filtered_df = filtered_df[filtered_df["예측_연체여부"] == 0]

        # 추정 연체확률 범위 설정 
        filtered_df = filtered_df[(filtered_df["예측_연체확률"] >= min_dlq_rate) & (filtered_df["예측_연체확률"] <= max_dlq_rate)]

        # JavaScript로 row 스타일 정의
        row_style = JsCode(f"""
        function(params) {{
            const prob = params.data.예측_연체확률;
            const lower = {builder.best_threshold} - {threshold_moderator};
            const upper = {builder.best_threshold} + {threshold_moderator};

            if (prob >= lower && prob <= upper) {{
                return {{
                    'backgroundColor': '#d6f5d6'
                }};
            }}
            return null;
        }}
        """)
    
        # Grid 설정
        gb = GridOptionsBuilder.from_dataframe(filtered_df)
        gb.configure_default_column(filter=True, sortable=True, resizable=True)

        # row Style
        gb.configure_grid_options(getRowStyle=row_style)

        # Grid 옵션 생성
        gridOptions = gb.build()

        grid_response = AgGrid(
            filtered_df,
            gridOptions=gridOptions,
            height=300,
            width='100%',
            data_return_mode='FILTERED_AND_SORTED',
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=False,
            allow_unsafe_jscode=True,
            theme='alpine'
        )

        filtered_df = grid_response['data']

        if not filtered_df.empty:
            csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="필터링된 데이터 CSV 다운로드",
                data=csv,
                file_name='필터링된_데이터.csv',
                mime='text/csv'
            )
    # 세 번째 탭 : Dynamic CSS Modeling 
    with 탭3:
        st.markdown("### Step3. Dynamic CSS Modeling ")

        st.markdown("""
            <style>
            .box-left, .box-right {
                background-color: #1b263b;
                border: 1px solid #3e5c76;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
            }
            .box-left h4, .box-right h4 {
                color: #e0e1dd;
                margin-bottom: 1rem;
            }
            .tag {
                background-color: #204d3c;
                color: white;
                padding: 6px 12px;
                    border-radius: 8px;
                margin-bottom: 6px;
                display: inline-block;
                font-size: 14px;
            }
            </style>
        """, unsafe_allow_html=True)    

        st.success("선택된 변수: " + ", ".join(builder.선택변수))

        if 'df_신용평가_차주' not in st.session_state:
            st.session_state['df_신용평가_차주'] = None
                                    
        fig = None 
        with st.expander("차주 csv 파일 업로드", expanded=True):
            uploaded_file = st.file_uploader("파일을 선택하세요", type=["csv", "xlsx"])
            
            if uploaded_file is not None:
                try:
                    # 확장자 확인
                    if uploaded_file.name.endswith(".csv"):
                        builder.df_신용평가차주 = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".xlsx"):
                        builder.df_신용평가차주 = pd.read_excel(uploaded_file, engine="openpyxl")  # openpyxl 필요
                    else:
                        st.warning("지원되지 않는 파일 형식입니다.")
                        builder.df_신용평가차주 = None

                    if builder.df_신용평가차주 is not None:
                        st.success("파일 업로드 성공")
                        st.write("데이터 미리보기")
                        st.dataframe(builder.df_신용평가차주)


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
                            # 예측 확률
                            builder.df_신용평가차주['예측_연체확률'] = builder.clf_base_model.predict_proba(builder.df_신용평가차주[builder.선택변수])[:, 1]
                            # threshold
                            builder.df_신용평가차주['대출승인_임계값'] = builder.best_threshold
                            # 예측값 생성 
                            builder.df_신용평가차주['예측_연체여부'] = (builder.df_신용평가차주["예측_연체확률"] >= builder.best_threshold).astype(int) 
                            #대출 승인여부 
                            builder.df_신용평가차주['대슬_승인여부'] = 1 - builder.df_신용평가차주['예측_연체여부'] 

                            st.session_state['df_신용평가_차주'] = builder.df_신용평가차주

                            st.session_state['scoring_done'] = True
                            
                except Exception as e:
                    st.error(f"파일을 불러오는 중 오류 발생: {e}")



        if st.session_state.get('scoring_done', False):

            with st.expander("신용평가 결과", expanded=True):

                if st.session_state['df_신용평가_차주'] is None:
                    df_AgGrid = builder.df_신용평가차주
                else :
                    df_AgGrid = st.session_state['df_신용평가_차주']


                # Grid 설정
                gb = GridOptionsBuilder.from_dataframe(df_AgGrid)
                gb.configure_default_column(filter=True, sortable=True, resizable=True)
                # Grid 옵션 생성
                gridOptions = gb.build()

                AgGrid(
                    df_AgGrid,
                    gridOptions=gridOptions,
                    height=100,
                    width='100%',
                    data_return_mode='FILTERED_AND_SORTED',
                    update_mode=GridUpdateMode.MODEL_CHANGED,
                    fit_columns_on_grid_load=False,
                    allow_unsafe_jscode=True,
                    theme='alpine'
                )

            adjusted_weights = {}
            with st.expander("변수별 중요도 가중치 조정", expanded=True):

                if st.session_state['df_신용평가_차주'] is None:
                    df_adjusted_차주 = builder.df_신용평가차주
                else :
                    df_adjusted_차주 = st.session_state['df_신용평가_차주']

                # SHAP 계산
                explainer = shap.TreeExplainer(builder.clf_base_model.best_estimator_)
                shap_vals = explainer.shap_values(df_adjusted_차주[builder.선택변수])
                shap_row = pd.Series(shap_vals[0], index=builder.선택변수)
                shap_row_abs = pd.Series(np.abs(shap_vals[0]), index=builder.선택변수)
                shap_ratio = (shap_row / shap_row_abs.mean()).round(2)

                if 'initial_shap_ratio' not in st.session_state:
                    st.session_state['initial_shap_ratio'] = shap_ratio.to_dict()

                for i in range(0, len(shap_ratio), 4):
                    cols = st.columns(4)
                    for j, feature in enumerate(shap_ratio.index[i:i+4]):
                        with cols[j]:
                            weight = st.slider(
                                f"{feature} - 기여도: {float(shap_ratio[feature]):.2f}",
                                -1.0, 
                                2.0,
                                1.0,
                                0.1,
                                key=f"slider_{feature}"
                            )
                            adjusted_weights[feature] = weight

                if st.session_state['df_신용평가_차주'] is None:
                    builder.df_신용평가차주['조정_연체확률'] = get_adjusted_proba_for_single(builder.df_신용평가차주,builder.clf_base_model,builder.explainer,adjusted_weights,builder.선택변수)
                    builder.df_신용평가차주['조정_연체여부'] = (builder.df_신용평가차주["조정_연체확률"] >= builder.best_threshold).astype(int) 
                    builder.df_신용평가차주['조정_대출승인여부'] = 1-st.session_state['df_신용평가_차주']['조정_연체여부']
                else :
                    st.session_state['df_신용평가_차주']['조정_연체확률'] = get_adjusted_proba_for_single(builder.df_신용평가차주,builder.clf_base_model,builder.explainer,adjusted_weights,builder.선택변수)
                    st.session_state['df_신용평가_차주']['조정_연체여부'] = (st.session_state['df_신용평가_차주']["조정_연체확률"] >= builder.best_threshold).astype(int) 
                    builder.df_신용평가차주['조정_대출승인여부'] = 1-st.session_state['df_신용평가_차주']['조정_연체여부']

            with st.expander("시각화 그래프", expanded=True):
                fig = plot_adjusted_proba_threshold_plotly(st.session_state['df_신용평가_차주'] )
                st.plotly_chart(fig, use_container_width=True)



def get_adjusted_proba_for_single(df_row: pd.DataFrame,
                                   model,
                                   explainer: shap.TreeExplainer,
                                   adjusted_weights: dict,
                                   feature_names: list) -> float:
    """
    단일 차주 row에 대해 SHAP 기반 가중치 조정 후 조정된 연체확률을 반환합니다.

    Parameters:
    - df_row: DataFrame, 단 하나의 row만 포함되어 있어야 함 (shape: [1, n])
    - model: 학습된 LightGBM or XGBoost 등 tree-based 분류 모델
    - explainer: shap.TreeExplainer 객체 (model 기반으로 생성된 것)
    - adjusted_weights: feature별 조정 가중치 dict (예: {'소득': 1.0, '대출건수': 0.8, ...})
    - feature_names: 예측에 사용되는 변수 리스트

    Returns:
    - float: 보정된 연체확률 (adjusted probability)
    """

    # sanity check
    if df_row.shape[0] != 1:
        raise ValueError("df_row must contain exactly one observation (one row)")

    # 1. SHAP 값 계산 (class 1 기준)
    shap_vals = explainer.shap_values(df_row[feature_names])
    shap_df = pd.DataFrame(shap_vals, columns=feature_names)

    # 2. 가중치 조정
    for feature, weight in adjusted_weights.items():
        if feature in shap_df.columns:
            shap_df[feature] *= weight

    # 3. log-odds 계산 (expected_value + SHAP 총합)
    adjusted_log_odds = shap_df.sum(axis=1).values[0] + explainer.expected_value

    # 4. 확률로 변환 (sigmoid)
    adjusted_proba = 1 / (1 + np.exp(-adjusted_log_odds))

    return adjusted_proba