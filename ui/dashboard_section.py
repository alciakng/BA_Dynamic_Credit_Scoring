import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd 
import shap

from data_visualizer import DataVisualizer

# -------------------------------------------------------------
#  dashboard section
# -------------------------------------------------------------

def show_delinquency_ratio(group_df1,group_df2,start_ym,end_ym):
    
    group1 = group_df1.groupby('YM')['DLQ_YN'].mean().reset_index()
    group1.rename(columns={'DLQ_YN': '기준 대출그룹 연체율'}, inplace=True)

    group2 = group_df2.groupby('YM')['DLQ_YN'].mean().reset_index()
    group2.rename(columns={'DLQ_YN': '비교 대출그룹 연체율'}, inplace=True)

    group1_filter = group1[(group1['YM'] >=start_ym) & (group1['YM'] <=end_ym)]
    group2_filter = group2[(group2['YM'] >=start_ym) & (group2['YM'] <=end_ym)]
 
    # 시각화클래스 초기화 (public)
    visualizer = DataVisualizer()

    visualizer.multi_scatter(group1_filter, group2_filter, 'YM', 'DLQ_YN', '년월', '연체율','기준대출','비교대출')



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
    fig = visualizer.multiple_bar(mean_shap_gen,mean_shap_pol,feature_names,'변수','평균 SHAP 절대값','일반자금 차주','정책자금 차주','SHAP 변수 중요도 비교 (정책자금 vs 일반자금)')

    st.plotly_chart(fig)