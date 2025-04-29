📊 BA_Financial_6

 - 신용정보원 데이터 기반 분석 프로젝트입니다.  
 - 차주, 대출, 신용카드, 연체정보, 보험정보 등의 다양한 데이터셋을 통합합니다.
 - 정책금융 차주의 전략적 연체율을 Chiappori-Salanié(양의상관성분석), DiD(이중차분분석)을 통해 식별합니다.
 - ML을 통해 전략적 연체 예측 모델을 구성하고, 가능성이 있는 차주를 식별하는 대안신용평가 모델을 기획합니다.

## 📁 프로젝트 구성

### 🗂️ 프로젝트 구조도
<pre><code>📁 BA_Finance_6/ 
 ├── 📂 dataset/ # 금융 데이터셋 (.csv 파일 모음) 
 
 │ ├── 📄 차주정보.csv  # 대출자 정보 
 │ ├── 📄 개인대출정보.csv # 개인 대출 내역 
 │ ├── 📄 신용카드개설정보.csv # 카드 개설 내역 
 │ └── 📄 보험담보정보.csv # 보험 담보 관련 데이터 
 │ └── ... 기타.csv
 │
 ├── common_code.py # 공통 코드 테이블 정의 (static class) 
 │
 ├── data_builder.py # 데이터 로딩 및 병합 처리 
 ├── data_visualizer.py # 시각화 함수 정의 (matplotlib, seaborn) 
 ├── machine_learner.py # ML 처리
 │
 ├── main.py # 프로젝트 실행용 메인 스크립트 
 │
 ├── practice.py # 실험용 코드 (연습, 테스트) 
 ├── dataset.json # 데이터셋에 대한 메타 정보 │ 
 │
 ├── .gitignore # Git에서 추적하지 않을 파일 목록 
 └── README.md # 프로젝트 설명 문서 </code></pre>

### 1. `dataset/`   
##### 신용정보원 모의데이터는 보안상 올리지 않습니다.(신용정보원 AI 학습장 참고 : https://ailp.kcredit.or.kr:3446/frt/main.do)
- 차주, 대출, 신용카드, 연체정보 등 다양한 금융 데이터셋이 저장된 폴더입니다.
- 원시 CSV 또는 전처리된 파일들이 포함됩니다.

### 2. `common_code.py`
- 공통 코드 테이블(LN_ACCT_CD, LN_CD_1 등)을 포함한 **static 클래스** 정의 파일입니다.

### 3. `data_builder.py`
- `dataset` 폴더에 있는 데이터를 로드하고, 필요한 컬럼을 기준으로 병합하여 **분석용 DataFrame**을 생성하는 역할을 합니다.

### 4. `data_visualizer.py`
- 시각화를 담당하는 클래스입니다.
- seaborn, matplotlib 기반 다양한 그래프 (막대그래프, 박스플롯, 트리맵 등) 를 생성합니다.

### 5. `machine_learner.py`
- 머신러닝 담당 클래스입니다.

### 6. `main.py`
- 이 프로젝트의 **메인 실행 파일**입니다.
- 위의 클래스들을 조합하여 전체 분석 파이프라인을 실행합니다.

---

## 📊 시각화 슬라이드

- [데이터 분포 시각화]
https://docs.google.com/presentation/d/1v_GPHuICVvLx6m1Yvaqh9ZcXvozIiHZHRpG-VbFSSFo/edit?usp=sharing

---

## 🧪 Staggered DID 

### 1) 전략적 연체율 식별을 위한 Staggered DID
<img width="882" alt="image" src="https://github.com/user-attachments/assets/bc442d58-9700-4b0c-8a64-d08fef575620" />

### 2) Pre Parallel Trend Assumption 검증
<img width="638" alt="image" src="https://github.com/user-attachments/assets/b1b1fd34-fc1b-4fd3-824c-9372a65a1f65" />

---

## 🧪 전략적연체에 유의한 feature 식별

### 1) 각 집단의 연체율을 예측
<img width="1159" alt="image" src="https://github.com/user-attachments/assets/a28a7f4b-bec1-488a-98ae-b10af92a6dca" />

### 2) SHAP(Feature Importance) 방법론을 통해 유의한 변수식별 
<img width="900" alt="image" src="https://github.com/user-attachments/assets/eeb5a74c-162e-4d06-97b8-9fed6d788495" />

---

## 시연연상

[![Video Label](http://img.youtube.com/vi/QAn6n9Ik5DE/0.jpg)](https://youtu.be/59USvjy2toI)

---

## 사업기획안


---
