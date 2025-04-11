import streamlit as st
import yaml
import hashlib

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
            st.sidebar.success(f"{username}님 환영합니다!")
        else:
            st.sidebar.error("❌ 아이디 또는 비밀번호가 잘못되었습니다.")

# 로그인 상태 확인 함수
def ensure_login():
    if not st.session_state.get('logged_in', False):
        login()
        st.stop()