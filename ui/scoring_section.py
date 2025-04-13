
import logging
import pprint
import lightgbm as lgb
from sklearn.metrics import auc, roc_curve
import streamlit as st

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from common_code import CommonCode

from controllers.data_builder import DatasetBuilder
from controllers.data_visualizer import DataVisualizer

from ui.dashboard_section import show_delinquency_ratio
from ui.dashboard_section import show_shap_analysis
from ui.dashboard_section import show_confusion_matrix
from ui.dashboard_section import show_performance_summary
from ui.dashboard_section import show_roc_curve


def run_scoring(builder: DatasetBuilder,visualizer : DataVisualizer, 기준_대출, 비교_대출, 선택변수, 조회시작년월, 조회종료년월):

    logging.info("예측모델링 개수")
    logging.info(len(builder.df_예측모델링))
    logging.info(builder.df_예측모델링.info())

    logging.info("기준대출 : " + pprint.pformat(기준_대출))
    logging.info("비교대출 : " + pprint.pformat(비교_대출))
    logging.info("선택변수 : " + pprint.pformat(선택변수))
    logging.info("선택변수 : " + pprint.pformat(조회시작년월))
    logging.info("선택변수 : " + pprint.pformat(조회종료년월))

    #기준집단
    df_base = builder.df_예측모델링[
        (builder.df_예측모델링['LN_CD_1'] == str(기준_대출['대출과목']['LN_CD_1']).lstrip('0')) &
        (builder.df_예측모델링['LN_CD_2'] == 기준_대출['대출과목']['LN_CD_2']) &
        (builder.df_예측모델링['LN_CD_3'] == 기준_대출['대출과목']['LN_CD_3']) &
        #(builder.df_예측모델링['지급능력_구간'] == 기준_대출['지급능력_구간']) &
        (builder.df_예측모델링['RATE'] <= int(기준_대출['금리']*1000)) &
        (builder.df_예측모델링['LN_AMT'] <= 기준_대출['대출금액(천원)'])
    ]

    # 조회년월 필터링 
    df_base = df_base[
        (df_base['YM'] >= int(조회시작년월)) &
        (df_base['YM'] <= int(조회종료년월))
    ]

    #비교집단
    df_comp = builder.df_예측모델링[
        (builder.df_예측모델링['LN_CD_1'] == str(비교_대출['대출과목']['LN_CD_1']).lstrip('0')) &
        (builder.df_예측모델링['LN_CD_2'] == 비교_대출['대출과목']['LN_CD_2']) &
        (builder.df_예측모델링['LN_CD_3'] == 비교_대출['대출과목']['LN_CD_3']) &
        #(builder.df_예측모델링['지급능력_구간'] == 비교_대출['지급능력_구간']) &
        (builder.df_예측모델링['RATE'] <= int(비교_대출['금리']*1000)) &
        (builder.df_예측모델링['LN_AMT'] <= int(비교_대출['대출금액(천원)']))
    ]

    # 조회년월 필터링 
    df_comp = df_comp[
        (df_comp['YM'] >= int(조회시작년월)) &
        (df_comp['YM'] <= int(조회종료년월))
    ]

    logging.info("기준대출 : "+pprint.pformat(df_base))
    logging.info("비교대출 : "+pprint.pformat(df_comp))

    # 차주수 제한 
    if len(df_base) < 1000 or len(df_comp) < 1000:
        st.warning(f"""
        ⚠️ 집단의 row 수가 너무 적습니다. 조건을 조정하세요.

        - 기준대출: {len(df_base):,}건  
        - 비교대출: {len(df_comp):,}건
        """)
        st.stop()  # 이후 실행 중단
    else : 
        with st.container():
            st.markdown("### 📌 데이터 건수 요약")
            st.markdown(f"""
            - **기준대출:** {len(df_base):,}건  
            - **비교대출:** {len(df_comp):,}건
            """)

            st.markdown("### 🧾 기준대출 조회조건")
            st.code(pprint.pformat(기준_대출), language="python")

            st.markdown("### 🧾 비교대출 조회조건")
            st.code(pprint.pformat(비교_대출), language="python")

            st.markdown("### 🧾 선택변수")
            st.code(선택변수, language="python")

    # 머신러닝 실행
    X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(
        df_base[선택변수], df_base['DLQ_YN'], test_size=0.2, random_state=42)

    X_comp_train, X_comp_test, y_comp_train, y_comp_test = train_test_split(
        df_comp[선택변수], df_comp['DLQ_YN'], test_size=0.2, random_state=42)
    

    # ✅ 2. 오버샘플링 (SMOTE)
    smote = SMOTE(random_state=42)

    X_base_train_res, y_base_train_res = smote.fit_resample(X_base_train, y_base_train)
    X_comp_train_res, y_comp_train_res = smote.fit_resample(X_comp_train, y_comp_train)

    # ✅ 3. 하이퍼파라미터 튜닝 (RandomizedSearchCV)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [15, 31],
        'max_depth': [3, 5],
        'min_child_samples': [10, 20],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    clf_base = RandomizedSearchCV(
        estimator=lgb.LGBMClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=3,  # 탐색 횟수
        cv=2,
        scoring='roc_auc',
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    clf_comp = RandomizedSearchCV(
        estimator=lgb.LGBMClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=3,
        cv=2,
        scoring='roc_auc',
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # ✅ 4. 모델 학습
    clf_base.fit(X_base_train_res, y_base_train_res)
    clf_comp.fit(X_comp_train_res, y_comp_train_res)

    # 예측 확률
    y_base_proba = clf_base.predict_proba(X_base_test)[:, 1]
    y_comp_proba = clf_comp.predict_proba(X_comp_test)[:, 1]

    # ROC 계산
    fpr_base, tpr_base, _ = roc_curve(y_base_test, y_base_proba)
    roc_auc_base = auc(fpr_base, tpr_base)

    fpr_comp, tpr_comp, _ = roc_curve(y_comp_test, y_comp_proba)
    roc_auc_comp = auc(fpr_comp, tpr_comp)

    # 시각화 

    # 탭 생성
    # css 주입
    st.markdown("""
    <style>
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

    탭1, 탭2, 탭3 = st.tabs(["📈 연체율 추이", "🔍 변수중요도 비교", "🔍 ROC-Curve"])

    # 첫 번째 탭: 연체율 추이
    with 탭1:
        st.subheader("연체율 추이")
        show_delinquency_ratio(df_base, df_comp, 조회시작년월, 조회종료년월)

    # 두 번째 탭: 변수중요도
    with 탭2:
        st.subheader("변수중요도 비교")
        show_shap_analysis(clf_base.best_estimator_, clf_comp.best_estimator_, X_base_test, X_comp_test)

        st.subheader("기준대출 변수중요도(Matrix)")
        show_performance_summary(clf_base.best_estimator_,선택변수,X_base_test,y_base_test,y_base_proba)

        st.subheader("비교대출 변수중요도(Matrix)")
        show_performance_summary(clf_comp.best_estimator_,선택변수,X_comp_test,y_comp_test,y_comp_proba)

    with 탭3:
        st.subheader("ROC Curve")
        show_roc_curve(fpr_base,tpr_base,fpr_comp,tpr_comp,roc_auc_base,roc_auc_comp)

        st.subheader("Confusion Matrix(기준대출)")
        show_confusion_matrix(clf_base,X_base_test,y_base_test)

        st.subheader("Confusion Matrix(비교대출)")
        show_confusion_matrix(clf_comp,X_comp_test,y_comp_test)