import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd 

from common_code import CommonCode
from streamlit_option_menu import option_menu

from ui.condition_section import main_condition

def main_section(builder, visualizer):

    username = st.session_state.get('username', '사용자')
    st.sidebar.subheader(f"👋 {username}님 환영합니다!")

    with st.sidebar:
        selected = option_menu(
            menu_title='Menu',  # 메뉴 제목
            options=["Dynamic Credit Scoring", "NPV based in Scoring"],
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
    if selected == "Dynamic Credit Scoring":
        st.subheader("📊 Dynamic Credit Scoring System")
        main_condition(builder,visualizer)
        
    elif selected == "NPV based in Scoring":
        st.subheader("🧮 경제적편익 계산 based in Scoring")
        