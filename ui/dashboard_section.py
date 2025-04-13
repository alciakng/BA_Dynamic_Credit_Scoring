from sklearn.metrics import confusion_matrix, roc_auc_score
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd 
import shap
import plotly.graph_objects as go

from scipy.stats import ks_2samp

from controllers.data_visualizer import DataVisualizer

# -------------------------------------------------------------
#  dashboard section
# -------------------------------------------------------------

def show_delinquency_ratio(group_df1,group_df2,start_ym,end_ym):
    
    group1 = group_df1.groupby('YM')['DLQ_YN'].mean().reset_index()
    group2 = group_df2.groupby('YM')['DLQ_YN'].mean().reset_index()

    group1_filter = group1[(group1['YM'] >=int(start_ym)) & (group1['YM'] <=int(end_ym))]
    group2_filter = group2[(group2['YM'] >=int(start_ym)) & (group2['YM'] <=int(end_ym))]
 
    # 시각화클래스 초기화 (public)
    visualizer = DataVisualizer()

    fig =visualizer.multi_scatter(group1_filter, group2_filter, 'YM', 'DLQ_YN', '년월', '연체율','기준대출','비교대출')

    st.plotly_chart(fig)


def show_shap_analysis(model1,model2, X1, X2):
    explainer1 = shap.Explainer(model1)
    explainer2 = shap.Explainer(model2)

    shap_values_gen = explainer1.shap_values(X1)
    shap_values_pol = explainer2.shap_values(X2)

    # 절대 SHAP 평균값 계산
    mean_shap_gen = np.abs(shap_values_gen).mean(axis=0)
    mean_shap_pol = np.abs(shap_values_pol).mean(axis=0)

    feature_names = X1.columns.tolist()

    df_shap = pd.DataFrame({
        'Feature': feature_names,
        '일반자금': mean_shap_gen,
        '정책자금': mean_shap_pol
    })

    # 시각화클래스 초기화 (public)
    visualizer = DataVisualizer()

    # 차트시각화
    fig = visualizer.multiple_bar(mean_shap_gen,mean_shap_pol,feature_names,'변수','평균 SHAP 절대값','기준대출 집단','비교대출 집단','SHAP 변수 중요도 비교 (기준대출 vs 비교대출)')

    st.plotly_chart(fig)


def show_roc_curve(fpr_base,tpr_base,fpr_comp,tpr_comp,roc_auc_base,roc_auc_comp):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_base, y=tpr_base,
                            mode='lines',
                            name=f'Base Model (AUC = {roc_auc_base:.2f})'))
    fig.add_trace(go.Scatter(x=fpr_comp, y=tpr_comp,
                            mode='lines',
                            name=f'Comp Model (AUC = {roc_auc_comp:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                            mode='lines',
                            line=dict(dash='dash'),
                            name='Random (AUC = 0.50)'))

    fig.update_layout(
        title='ROC Curve (Plotly)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800,
        height=600,
        legend=dict(x=0.6, y=0.05),
        template='plotly_white'
    )

    st.plotly_chart(fig)


def show_confusion_matrix(clf,X_test,y_test):
    y_pred_base = clf.predict(X_test)
    # 혼동행렬 생성 시 레이블 순서 지정 (True=1이 먼저 오게)
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred_base, labels=[1,0])
    cm_percent = cm / cm.sum() * 100
    labels = np.array([[f"{int(cm[i, j])}<br>({cm_percent[i, j]:.1f}%)" for j in range(2)] for i in range(2)])

    # Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        text=labels,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        zmin=0,
        zmax=np.max(cm),
        hoverinfo='text'
    ))

    # 축 설정
    fig.update_layout(
        title='Confusion Matrix<br><sub>(Columns: 실제, Rows: 예측)</sub>',
        xaxis=dict(
            title='실제 값 (Actual)',
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['True (1)', 'False (0)']
        ),
        yaxis=dict(
            title='예측 값 (Predicted)',
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['True (1)', 'False (0)'],
            autorange='reversed'
        ),
        width=500,
        height=500
    )

    st.plotly_chart(fig)


def show_performance_summary(model,columns,X_test,y_test,y_pred_proba):
    feature_cols = columns
    rows = []

    # 전체 예측 확률 (전체 모델 기준)
    auc_total = roc_auc_score(y_test, y_pred_proba)
    ks_total = ks_2samp(y_pred_proba[y_test == 1], y_pred_proba[y_test == 0]).statistic

    # SHAP값 계산
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  

    # Gain 값
    feature_gain = model.booster_.feature_importance(importance_type='gain')
    gain_dict = dict(zip(X_test.columns, feature_gain))

    # 변수별 성능 정리
    for i, col in enumerate(feature_cols):
        x = X_test[col]
        auc = roc_auc_score(y_test, x)
        ks = ks_2samp(x[y_test == 1], x[y_test == 0]).statistic
        shap_mean = np.abs(shap_values[:, i]).mean()
        gain = gain_dict.get(col, 0)

        rows.append({
            '변수': col,
            'AUC (단변수)': round(auc, 4),
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

    """
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df_summary.columns),
            fill_color='lightblue',
            align='center',
            font=dict(size=12, color='black'),
        ),
        cells=dict(
            values=[df_summary[col] for col in df_summary.columns],
            fill_color='white',
            align='center',
            format=["", ".4f", ".4f", ".4f", ".2f"],  # 숫자 포맷 설정
            font=dict(size=11, color='black')  
        )
    )])

    fig.update_layout(
        title='변수별 성능 요약 (AUC / KS / SHAP / Gain)',
        paper_bgcolor='white',  
        plot_bgcolor='white',
        margin=dict(l=10, r=10, t=30, b=10)  # 좌우상하 여백 최소화
    )
    """

    
    # '전체모델' 행 제외하고 수치만 추출해 index를 '변수'로 설정
    df_numeric = df_summary[df_summary['변수'] != '전체모델'].set_index('변수')

    # imshow로 시각화
    fig = px.imshow(
        df_numeric,
        text_auto=".2f",  # 셀에 숫자 자동 표기 (소수 2자리)
        color_continuous_scale=[[0, "white"], [1, "white"]],  # 색상 강조 (Viridis, YlGnBu 등 사용 가능)
        aspect="auto",
        title="변수별 성능 요약 (AUC / KS / SHAP / Gain)"
    )
    fig.update_coloraxes(showscale=False) 

    # ✅ 텍스트 색상과 크기 설정 (모든 셀 텍스트 → 검정색)
    fig.update_traces(
        textfont=dict(color='black', size=16)
    )

    fig.update_layout(
        paper_bgcolor='#0D1B2A',
        plot_bgcolor='#0D1B2A',
        margin=dict(l=20, r=20, t=40, b=20),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    