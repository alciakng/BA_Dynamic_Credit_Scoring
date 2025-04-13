import streamlit as st
import json
import pandas as pd
import os 
import logging

from controllers.data_builder import DatasetBuilder
from controllers.data_visualizer import DataVisualizer

from ui.main_section import main_section

from ui.dashboard_section import show_delinquency_ratio
from ui.dashboard_section import show_shap_analysis

from ui.login_section import login
from ui.login_section import ensure_login


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)

# 현재 파이썬 파일 기준으로 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, 'dataset.json')

# dataset.json 
with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# 시각화클래스 초기화 (public)
visualizer = DataVisualizer()

# 빌더 초기화
if 'builder' not in st.session_state:
    st.session_state['builder'] = DatasetBuilder(dataset)

builder = st.session_state['builder']
builder.app_initialize()

# -------------------------------------------------------------
# Streamlit Application
# -------------------------------------------------------------
# ensure_login 메서드
ensure_login()
# main_section 실행
main_section(builder, visualizer)