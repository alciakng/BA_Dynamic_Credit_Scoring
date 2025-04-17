from sklearn.metrics import confusion_matrix, roc_auc_score
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd 
import shap
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

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


def show_performance_summary(df_summary, key):
    
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
        textfont=dict(color='black', size=17)
    )

    fig.update_layout(
        paper_bgcolor='#0D1B2A',
        plot_bgcolor='#0D1B2A',
        margin=dict(l=20, r=20, t=40, b=20),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    df_summary_wo = df_summary[df_summary['변수'] != '전체모델']
    generate_variable_report_aggrid(df_summary_wo, key)


def generate_variable_report_aggrid(df: pd.DataFrame, key):
    """
    변수 중요도 보고서를 AgGrid 테이블로 표시
    """
    report_rows = []

    for _, row in df.iterrows():
        var = row['변수']
        auc = row['AUC (단변수)']
        ks = row['KS (단변수)']
        shap = row['SHAP']
        gain = row['Gain']

        # 변수 요약 문자열 생성
        summary = f"AUC: {auc:.2f}, KS: {ks:.2f}, SHAP: {shap:.2f}, Gain: {gain:,.0f}"

        # 해석 요약 로직
        shap_level = "상대적으로 높은" if shap > 0.3 else "낮은"
        model_contrib = f"1. 해당 변수는 모델의 성능에 '{shap_level}' 기여를 합니다."
        univar_msg = "단변수 AUC와 KS도 안정적인 분류력을 보여줍니다." if auc > 0.4 and ks > 0.2 else "2. 단변수 중요도는 다소 낮습니다."

        # 특이 케이스 추가 해석
        extra = ""
        has_extra = False
        if (auc < shap) & (shap > 0.45): 
            extra = "⚠️ 3. AUC는 낮지만 SHAP은 높음 → 변수는 비선형적/상호작용적 방식으로 중요한 기여를 합니다."
            has_extra = True

        interpretation = f"{model_contrib} \n {univar_msg} \n {extra}".strip()

        report_rows.append({
            "변수": var,
            "변수 요약": summary,
            "해석 요약": interpretation,
            "has_extra": has_extra  # ✅ 조건부 색상용 플래그
        })


    # DataFrame으로 변환
    report_df = pd.DataFrame(report_rows)

    # ✅ 행 강조 색상 설정 (조건: has_extra == True)
    row_style = JsCode("""
    function(params) {
        if (params.data.has_extra === true) {
            return {
                'backgroundColor': '#e8f5e9',  // 연한 초록 배경
                'fontWeight': 'bold'
            }
        }
    }
    """)

    gb = GridOptionsBuilder.from_dataframe(report_df)

    gb.configure_column("변수", width=120)
    gb.configure_column("변수 요약", width=300)
    gb.configure_column("해석 요약", width=700)

    gb.configure_default_column(wrapText=False, autoHeight=False, resizable=True)
    
    
    gb.configure_column("has_extra", hide=True)

    gb.configure_grid_options(domLayout='normal', getRowStyle=row_style)
    gb.configure_grid_options(domLayout='autoHeight')
    gb.configure_default_column(autoWidth=True)

    with st.expander("변수 중요도 종합보고서",expanded = True):
        AgGrid(report_df,
        gridOptions=gb.build(),
        height=600,
        width='100%',
        data_return_mode='FILTERED_AND_SORTED',
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
        theme='alpine',
        key = key)

        

# 차주 보정결과 시각화 
def plot_adjusted_proba_threshold_plotly(
    df,
    threshold_col='대출승인_임계값',
    orig_proba_col='예측_연체확률',
    adjusted_proba_col='조정_연체확률'
):
    """
    연체확률 시각화 (Plotly)
    - 조정된 확률이 존재하면: 원래/보정 비교
    - 없으면: 원래 예측만 표시
    """

    y_orig = df[orig_proba_col].values[0]
    threshold = df[threshold_col].values[0]

    fig = go.Figure()

    # Always show original prediction
    fig.add_trace(go.Bar(
        x=["원래 예측"],
        y=[y_orig],
        name="원래 예측",
        marker_color='royalblue',
        text=[f"{y_orig:.2f}"],
        textposition='outside'
    ))

    # If adjusted value exists, show it too
    if adjusted_proba_col in df.columns:
        y_adj = df[adjusted_proba_col].values[0]
        fig.add_trace(go.Bar(
            x=["보정 후"],
            y=[y_adj],
            name="보정 후",
            marker_color='seagreen',
            text=[f"{y_adj:.2f}"],
            textposition='outside'
        ))

    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"임계값: {threshold:.2f}",
        annotation_position="top left"
    )

    fig.update_layout(
        title="연체 확률 시각화",
        yaxis_title="연체 확률",
        yaxis=dict(range=[0, 1.1]),
        bargap=0.5,
        showlegend=False,
        height=400
    )

    return fig