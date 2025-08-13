📊 Dynamic_Credit_Scoring

 - 신용정보원 모의 데이터 기반 아이디어 기획 프로젝트입니다.
 - 차주, 대출, 신용카드, 연체정보, 보험정보 등의 다양한 모의 데이터셋을 통합합니다.
 - 대안정보를 이용한 신용평가 정보를 바탕으로 심사자가 개입 및 신용평점 사후 조정(Dynamic_Credit_Scoring) 시뮬레이션 아이디어를 제시합니다.
 - 심사자는 차주의 Credit Scoring에 기여한 SHAP Importance를 조정하고 역산함으로써 신용평점을 조정할 수 있다는 아이디어를 제시합니다.


<img width="200" height="65" alt="image" src="https://github.com/user-attachments/assets/31570db3-0e18-4375-a64b-fe27fde9315c" />
<img width="200" height="75" alt="image" src="https://github.com/user-attachments/assets/8df3ef9e-14e1-4a34-80fa-deb512efe658" />
<img width="300" height="72" alt="image" src="https://github.com/user-attachments/assets/c4e1286c-6ebb-4ebd-946e-20687c012fd0" />



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

## 시연연상
[![Video Label](http://img.youtube.com/vi/QAn6n9Ik5DE/0.jpg)](https://youtu.be/QAn6n9Ik5DE)

---

## 사업화 기획
![Image](https://github.com/user-attachments/assets/de69717b-a03f-4fce-bb37-d21aafd68450)
![Image](https://github.com/user-attachments/assets/cebd46db-b8e2-4b56-af56-ba9fd1abc1d7)
![Image](https://github.com/user-attachments/assets/9ce2fde2-9dfc-478b-8bd5-2cbc350aaa52)
![Image](https://github.com/user-attachments/assets/bc2bff49-0c33-4537-83db-3bfed2618e5f)
![Image](https://github.com/user-attachments/assets/3bfd370f-53d1-4ef4-980a-f1fcc7ed16b2)
![Image](https://github.com/user-attachments/assets/ebb12632-b131-48c8-b82c-2927494c9fa8)
![Image](https://github.com/user-attachments/assets/fbea80bb-abf3-4448-ad50-4d1f66d949b2)
![Image](https://github.com/user-attachments/assets/73ec8d90-4278-443f-bc7c-dc2e8c225138)
![Image](https://github.com/user-attachments/assets/0491ac21-37da-4e77-950d-a2d2db065d6c)
![Image](https://github.com/user-attachments/assets/d2010ee2-8283-4745-8e01-d2bb5faf1c64)
![Image](https://github.com/user-attachments/assets/54f83a97-47ea-4678-8208-a818b0eec211)
![Image](https://github.com/user-attachments/assets/bbdac745-35f5-4a91-a74d-d3f4e6217f0a)
---
