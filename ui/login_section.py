import streamlit as st
import yaml
import hashlib

from ui.main_section import main_section
# -------------------------------------------------------------
#  로그인 Section
# -------------------------------------------------------------

# 비밀번호 해싱 함수
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# yaml 파일로부터 사용자 정보 로드
def load_users(yaml_path='users.yaml'):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

# 로그인 함수 구현
def login():
    st.sidebar.title("🔐 로그인")
    username = st.sidebar.text_input("아이디")
    password = st.sidebar.text_input("비밀번호", type="password")

    if st.sidebar.button("로그인"):
        users = load_users()
        hashed_input = hash_password(password)

        if username in users and users[username]['password'] == hashed_input:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.rerun()
            return
        else:
            st.sidebar.error("❌ 아이디 또는 비밀번호가 잘못되었습니다.")

# 로그인 상태 확인 함수
def ensure_login():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        ## 앱제목
        st.markdown(
        """
        <div style='text-align: center; padding-top: 100px;'>
            <h1 style='font-size: 35px; font-weight: 800;'>
                🚀 <span style='color: #4CAF50;'>Dynamic</span> Credit Scoring System
            </h1>
            <h3 style='margin-top: -10px; color: gray;'>- 이종집단 비교 신용평가 모델링</h3>
        </div>
        """,
        unsafe_allow_html=True
        )

        login()
        st.stop()  # 로그인 안 됐으면 여기서 렌더 중단