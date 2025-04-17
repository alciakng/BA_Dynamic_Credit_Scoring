import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd 

from common_code import CommonCode
from streamlit_option_menu import option_menu

from ui.condition_section import main_condition
from ui.condition_section import dynamic_condition

def main_section(builder, visualizer):

    #based_scored 조건 초기화
    if 'based_scored' not in st.session_state:
        st.session_state['based_scored'] = False

    username = st.session_state.get('username', '사용자')
    st.sidebar.subheader(f"👋 {username}님 환영합니다!")

    with st.sidebar:
        selected = option_menu(
            menu_title='Menu',  # 메뉴 제목
            options=["Based Credit Scoring", "Dynamic Credit Scoring"],
            icons=["bar-chart", "calculator"],
            menu_icon='folder',  # 상단 아이콘
            default_index=0,
            styles={
                "container": {"padding": "10px"},
                "icon": {"color": "green", "font-size": "18px"},
                "nav-link-selected": {"background-color": "#4CAF50", "font-weight": "bold", "color": "white"}
            }
    )
    
    # 콘텐츠 렌더링
    if selected == "Based Credit Scoring":
        st.subheader("📊 Dynamic Credit Scoring System")
        main_condition(builder,visualizer)
        
    elif selected == "Dynamic Credit Scoring":
        st.subheader("🧮 Dynamic Credit Scoring")

        # 필수 세션 키가 없으면 안내
        if not st.session_state.get('based_scored', False):
            # ✅ 중앙에 토스트 스타일 메시지 띄우기 (streamlit-toast 활용 or fallback)
            st.markdown("""
            <div style='
                background-color: #ffdede;
                padding: 0.5rem;
                border-radius: 8px;
                text-align: center;
                font-size: 0.9rem;
                font-weight: bold;
                color: #990000;
                margin: 2rem auto;
                width: 70%;
                box-shadow: 0 0 8px rgba(0,0,0,0.15);'>
                ⚠️ `Based Credit Scoring` 탭에서 먼저 모델학습을 진행하세요.
            </div>
            """, unsafe_allow_html=True)
            
            # 실행 중단 또는 안내만 하고 return
            st.stop()
            return 
    
        dynamic_condition(builder, visualizer)
        