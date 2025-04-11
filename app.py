import streamlit as st
import json
import pandas as pd
import os 

from data_builder import DatasetBuilder
from data_visualizer import DataVisualizer

from ui.dashboard_section import show_delinquency_ratio
from ui.dashboard_section import show_shap_analysis

from ui.condition_section import main_condition

from ui.login_section import login
from ui.login_section import ensure_login


# 현재 파이썬 파일 기준으로 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, 'dataset.json')

# dataset.json 
with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# 데이터셋 빌더 초기화 (public)
#builder = DatasetBuilder(dataset)
# 시각화클래스 초기화 (public)
visualizer = DataVisualizer()

# 표준업종 10차코드 로드 
#builder.load_kic()
# 데이터 로드 
#builder.load_data()

# -------------------------------------------------------------
# Streamlit Application
# -------------------------------------------------------------
st.write("🔍 상태:", st.session_state.get("logged_in"))
st.write("📍 위치 도달함")

ensure_login()
main_condition()