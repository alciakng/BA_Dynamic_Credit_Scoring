
import logging
import pprint
import lightgbm as lgb
import numpy as np
import shap
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, auc
import streamlit as st
import pandas as pd 

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from scipy.stats import ks_2samp

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
            st.markdown("###  데이터 건수 요약")
            st.markdown(f"""
            - **기준대출:** {len(df_base):,}건  
            - **비교대출:** {len(df_comp):,}건
            """)

            st.markdown("### 기준대출 조회조건")
            st.code(pprint.pformat(기준_대출), language="python")

            st.markdown("### 비교대출 조회조건")
            st.code(pprint.pformat(비교_대출), language="python")

            st.markdown("### 선택변수")
            st.code(선택변수, language="python")

    # 머신러닝 실행
    X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(
        df_base[선택변수], df_base['DLQ_YN'], test_size=0.2, random_state=42)

    X_comp_train, X_comp_test, y_comp_train, y_comp_test = train_test_split(
        df_comp[선택변수], df_comp['DLQ_YN'], test_size=0.2, random_state=42)
    

    # 2. 오버샘플링 (SMOTE)
    smote = SMOTE(random_state=42)

    X_base_train_res, y_base_train_res = smote.fit_resample(X_base_train, y_base_train)
    X_comp_train_res, y_comp_train_res = smote.fit_resample(X_comp_train, y_comp_train)

    # 3. 하이퍼파라미터 튜닝 (RandomizedSearchCV)
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

    builder.clf_base_model = clf_base

    # 4. 모델 학습
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

    # streamlit 용 df 생성 
    columns = ['대출건수', '장기고액대출건수', '대출금액합', '지급능력', '지급능력_구간', '보험건수', '보험월납입액', '연체건수', '장기연체건수', '연체금액합', '신용카드개수', '신용카드_사용률_증가량', '현금서비스_사용률_증가량','DLQ_YN']
    
    # 데이터프레임 복사 
    builder.df_css_base = df_base[columns].copy()
    builder.df_css_comp = df_comp[columns].copy()

    builder.df_css_base.rename(columns={"DLQ_YN": "실제_연체여부"}, inplace=True)

    # X , Y 
    X_base = builder.df_css_base[선택변수]
    Y_base = builder.df_css_base['실제_연체여부']

    builder.df_css_base['예측_연체확률'] = clf_base.predict_proba(X_base)[:, 1]

    # 최적 임계값 산출 
    prec, recall, thresholds = precision_recall_curve(Y_base, builder.df_css_base['예측_연체확률'])
    f1 = 2 * (prec * recall) / (prec + recall + 1e-6)
    builder.best_threshold = thresholds[np.argmax(f1)]

    # threshold
    builder.df_css_base['대출승인_임계값'] = builder.best_threshold

    # 예측값 생성 
    builder.df_css_base['예측_연체여부'] = (builder.df_css_base["예측_연체확률"] >= builder.best_threshold).astype(int)
    builder.df_css_base['대슬_승인여부'] = 1- builder.df_css_base['예측_연체여부']

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

        explainer1 = shap.Explainer(clf_base.best_estimator_)
        explainer2 = shap.Explainer(clf_comp.best_estimator_)

        shap_values_base = explainer1.shap_values(X_base_test)
        shap_values_comp = explainer2.shap_values(X_comp_test)

        # explainer1 builder 저장 
        builder.explainer = explainer1

        # builder 에 shap 값 부여 
        shap_df = pd.DataFrame(shap_values_base, columns=builder.선택변수)

        # 중요도 계산 
        shap_importance = shap_df[builder.선택변수].abs().mean()
        builder.shap_weight_df = (shap_importance / shap_importance.mean()).round(2)

        # 전체 예측 확률 (전체 모델 기준)
        auc_total = roc_auc_score(y_base_test, y_base_proba)
        ks_total = ks_2samp(y_base_proba[y_base_test == 1], y_base_proba[y_base_test == 0]).statistic

        # SHAP값 계산
        explainer = shap.TreeExplainer(clf_base.best_estimator_)
        shap_values = explainer.shap_values(X_base_test)  

        # Gain 값
        feature_gain = clf_base.best_estimator_.booster_.feature_importance(importance_type='gain')
        gain_dict = dict(zip(X_base_test.columns, feature_gain))

        rows = []
        # 변수별 성능 정리
        for i, col in enumerate(선택변수):
            x = X_base_test[col]
            auc_val = roc_auc_score(y_base_test, x)
            ks = ks_2samp(x[y_base_test == 1], x[y_base_test == 0]).statistic
            shap_mean = np.abs(shap_values[:, i]).mean()
            gain = gain_dict.get(col, 0)

            rows.append({
                '변수': col,
                'AUC (단변수)': round(auc_val, 4),
                'KS (단변수)': round(ks, 4),
                'SHAP': round(shap_mean, 4),
                'Gain': round(gain, 2)
            })
        
        
        # 전체 모델 기준 성능 추가
        rows.append({
            '변수': '전체모델',
            'AUC (단변수)': round(auc_total, 4),
            'KS (단변수)': round(ks_total, 4),
            'SHAP': np.abs(shap_values).mean().round(4),
            'Gain': np.sum(feature_gain).round(2)
        })

        df_summary = pd.DataFrame(rows)
        builder.df_summary_wo = df_summary[df_summary['변수'] != '전체모델']

        show_shap_analysis(clf_base.best_estimator_, clf_comp.best_estimator_, X_base_test, X_comp_test)

        st.subheader("기준대출 변수중요도(Matrix)")
        show_performance_summary(df_summary,"기준대출")

        st.subheader("비교대출 변수중요도(Matrix)")
        show_performance_summary(df_summary,"비교대출")

    with 탭3:
        st.subheader("ROC Curve")
        show_roc_curve(fpr_base,tpr_base,fpr_comp,tpr_comp,roc_auc_base,roc_auc_comp)

        col1, col2 = st.columns(2)

        with col1: 
            st.subheader("Confusion Matrix(기준대출)")
            show_confusion_matrix(clf_base,X_base_test,y_base_test)
        with col2:
            st.subheader("Confusion Matrix(비교대출)")
            show_confusion_matrix(clf_comp,X_comp_test,y_comp_test)