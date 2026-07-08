#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
아토피 중증도 AI — 실험 결과 비교 대시보드 (Streamlit)
실행:  pip install streamlit plotly pandas
       streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="아토피 중증도 AI — 실험 비교", layout="wide")

# ==================== 전역 스타일 (가독성 개선) ====================
st.markdown("""
<style>
/* 기본 글씨 크기·줄간격 확대 */
html, body, [class*="css"] { font-size: 17px; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }

/* 제목 */
h1 { font-size: 2.3rem !important; font-weight: 800 !important; letter-spacing: -0.5px; }
h2, [data-testid="stSubheader"] { font-size: 1.5rem !important; font-weight: 700 !important; }
h3 { font-size: 1.25rem !important; font-weight: 700 !important; }

/* 본문 문단 */
div[data-testid="stMarkdownContainer"] p {
    font-size: 1.06rem; line-height: 1.8; color: inherit;
}

/* 탭 라벨 크게·굵게 */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    font-size: 1.1rem; font-weight: 700; padding: 12px 22px;
    border-radius: 10px 10px 0 0;
}
.stTabs [aria-selected="true"] { background: rgba(23,158,142,0.12); }

/* 메트릭 카드 */
[data-testid="stMetric"] {
    background: rgba(128,128,128,0.06);
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 14px; padding: 16px 18px;
}
[data-testid="stMetricValue"] { font-size: 2.1rem; font-weight: 800; }
[data-testid="stMetricLabel"] { font-size: 1.02rem; font-weight: 600; }
[data-testid="stMetricDelta"] { font-size: 0.95rem; }

/* 알림 박스(info/success/warning) 글씨 크게 */
[data-testid="stAlert"] { font-size: 1.05rem; line-height: 1.75; border-radius: 12px; }

/* 표 글씨 */
[data-testid="stDataFrame"] { font-size: 1.0rem; }
</style>
""", unsafe_allow_html=True)

# ==================== 실험 데이터 (실제 측정치) ====================
GRADING = pd.DataFrame([
    {"방식": "CORN (이미지→CNN, ordinal)", "QWK": 0.634, "accuracy": 0.633, "유형": "CNN 직접", "설명가능성": "낮음"},
    {"방식": "D 스태킹 앙상블",             "QWK": 0.6205, "accuracy": 0.6556, "유형": "앙상블", "설명가능성": "낮음"},
    {"방식": "Baseline (EffNetV2, CE)",     "QWK": 0.590, "accuracy": 0.644, "유형": "CNN 직접", "설명가능성": "낮음"},
    {"방식": "B 개수·면적 (마스크→LightGBM)", "QWK": 0.4997, "accuracy": 0.578, "유형": "정량화+부스팅", "설명가능성": "높음"},
])

SEG = pd.DataFrame([
    {"모델": "DeepLabV3Plus+resnet50", "IoU": 0.680, "Dice": 0.810, "학습(분)": 8.3},
    {"모델": "UnetPlusPlus+resnet50",  "IoU": 0.678, "Dice": 0.808, "학습(분)": 8.3},
    {"모델": "Unet+resnet34",          "IoU": 0.677, "Dice": 0.807, "학습(분)": 8.3},
    {"모델": "FPN+efficientnet-b3",    "IoU": 0.676, "Dice": 0.807, "학습(분)": 8.5},
    {"모델": "MAnet+resnet50",         "IoU": 0.676, "Dice": 0.807, "학습(분)": 8.3},
    {"모델": "DeepLabV3+resnet50",     "IoU": 0.676, "Dice": 0.807, "학습(분)": None},
])

# 혼동행렬 (행=정답, 열=예측) [Mild, Moderate, Severe] — eval셋이 방식마다 다름(주의)
CM = {
    "CORN":       [[23, 10, 1], [11, 22, 7], [0, 4, 12]],
    "D 앙상블":    [[27, 7, 0], [10, 25, 5], [1, 8, 7]],
    "Baseline":   [[20, 13, 1], [11, 28, 1], [0, 6, 10]],
    "B 개수·면적": [[23, 19, 7], [18, 38, 12], [3, 14, 39]],
}
CM_BACKBONES = pd.DataFrame([
    {"백본": "densenet121", "QWK": 0.6220}, {"백본": "resnet50", "QWK": 0.6218},
    {"백본": "efficientnet_v2_s", "QWK": 0.6104}, {"백본": "convnext_tiny", "QWK": 0.5223},
    {"백본": "단순평균 앙상블", "QWK": 0.5870}, {"백본": "스태킹(메타러너)", "QWK": 0.6205},
])
FEAT = pd.DataFrame([
    ("lbp_std (태선화·텍스처)", 1810), ("area_ratio (면적)", 1802), ("redness (홍반)", 1793),
    ("elongation (상처)", 1754), ("glcm_contrast (태선화)", 1590), ("redness_x_area (홍반×면적)", 1517),
    ("glcm_homogeneity (태선화)", 1404), ("mean_lesion_size (구진밀도)", 1239), ("n_lesions (구진개수)", 608),
], columns=["특징", "중요도"])

LAB = ["Mild", "Moderate", "Severe"]

# ==================== 헤더 ====================
st.title("🩺 아토피 중증도 AI — 실험 결과 비교")
st.caption("AI Hub #71863 합성데이터 · 3클래스(Mild/Moderate/Severe) · self-annotated 마스크 (리허설/검증용)")

st.markdown(
    "<div style='padding:10px 14px;border-radius:10px;border:1px solid #B0B8C1;"
    "background:rgba(120,130,150,0.08);font-size:0.9rem;line-height:1.6;'>"
    "📌 <b>데이터 출처</b> — 본 데모는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원(NIA)의 지원을 받아 구축된 "
    "「안면부 피부질환 이미지 <b>합성데이터</b>」(AI 허브, 과제번호 71863)를 활용하여 제작되었습니다. "
    "데이터: AI 허브(<a href='https://www.aihub.or.kr' target='_blank'>www.aihub.or.kr</a>). "
    "원본 저작권은 AI 허브/구축기관에 있으며 연구·시연용으로 무단 재배포를 금합니다."
    "</div>",
    unsafe_allow_html=True)
st.write("")

c1, c2, c3, c4 = st.columns(4)
c1.metric("최고 중증도 QWK", "0.634", "CORN (ordinal)")
c2.metric("최고 검출 IoU", "0.680", "DeepLabV3Plus")
c3.metric("최고 검출 Dice", "0.810", "DeepLabV3Plus")
c4.metric("비교한 모델 수", "10", "등급 4 + 검출 6")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["① 중증도 등급화 비교", "② 혼동행렬", "③ B방식 설명가능성", "④ 병변 검출 모델", "⑤ 전체 파이프라인"])

# ==================== 탭1 — 중증도 등급화 ====================
with tab1:
    st.subheader("중증도 등급화 4방식 — QWK 비교")
    left, right = st.columns([3, 2])
    with left:
        g = GRADING.sort_values("QWK")
        fig = px.bar(g, x="QWK", y="방식", orientation="h", color="설명가능성",
                     color_discrete_map={"높음": "#E0982F", "낮음": "#7B8794"},
                     text="QWK", range_x=[0, 0.75])
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.add_vline(x=0.7, line_dash="dash", line_color="green",
                      annotation_text="목표 0.7", annotation_position="top")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.dataframe(GRADING.style.format({"QWK": "{:.3f}", "accuracy": "{:.3f}"})
                     .background_gradient(subset=["QWK"], cmap="Greens"),
                     hide_index=True, use_container_width=True)

    st.markdown("""
해석 — 성능은 이미지를 직접 학습하는 CNN 계열(CORN 0.634)이 가장 높고,
개수·면적 기반 B(0.500)가 가장 낮다. B는 수작업 특징 9개로 정보를 압축하며 손실이 생기고,
self-annotated 마스크·합성데이터 조건에서 정량화 정확도가 제한된 것이 원인이다.
다만 B는 판단 근거를 제시할 수 있어(→ 탭③) 성능–설명가능성 트레이드오프를 보여준다.
""")

# ==================== 탭2 — 혼동행렬 ====================
with tab2:
    st.subheader("혼동행렬 (행=정답, 열=예측)")
    method = st.selectbox("방식 선택", list(CM.keys()))
    cm = CM[method]
    colA, colB = st.columns([3, 2])
    with colA:
        fig = go.Figure(go.Heatmap(z=cm, x=LAB, y=LAB, text=cm, texttemplate="%{text}",
                                   colorscale="Blues", showscale=False))
        fig.update_layout(height=380, xaxis_title="예측", yaxis_title="정답",
                          yaxis_autorange="reversed", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        total = sum(sum(r) for r in cm)
        diag = sum(cm[i][i] for i in range(3))
        far = cm[0][2] + cm[2][0]   # Mild↔Severe (2칸 오류)
        st.metric("정확히 맞힘", f"{diag}/{total}", f"{diag/total*100:.1f}%")
        st.metric("2칸 오류 (Mild↔Severe)", f"{far}", "적을수록 좋음")
        st.info("대부분의 오류가 인접 등급(Mild↔Moderate, Moderate↔Severe)에 몰려 있고 "
                "2칸 오류는 드물다 → 모델이 중증도 순서를 학습했다는 신호.")
    

# ==================== 탭3 — B방식 설명가능성 ====================
with tab3:
    st.subheader("B 방식 특징 중요도 — 특징이 트리에서 분기(split)에 사용된 횟수")
    st.caption("LightGBM feature importance. CNN·앙상블은 제공 못 하는 B 방식만의 강점.")
    fig = px.bar(FEAT.sort_values("중요도"), x="중요도", y="특징", orientation="h",
                 color="중요도", color_continuous_scale="Oranges", text="중요도")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
해석 — 태선화 텍스처(GLCM,LBP사용), 병변 면적, 홍반(붉기)이 IGA 예측에 고루 기여함.
임상 EASI 채점의 핵심 요소(강도·면적)가 실제로 모델 판단에 반영됐음을 보여줌 —
이 해석가능성이 B 방식을 성능이 낮음에도 채택할 이유가 될 수 있다 (의료 적용 시 근거 제시 가능).
""")

# ==================== 탭4 — 병변 검출 ====================
with tab4:
    st.subheader("병변 검출·분할 모델 6종 — IoU / Dice")
    st.caption("픽셀 단위로 병변 영역을 얼마나 정확히 찾는지 비교 (IoU·Dice 모두 1에 가까울수록 좋음)")

    left, right = st.columns([3, 2])
    with left:
        m = SEG.melt(id_vars="모델", value_vars=["IoU", "Dice"], var_name="지표", value_name="값")
        fig = px.bar(m, x="모델", y="값", color="지표", barmode="group",
                     color_discrete_map={"IoU": "#179E8E", "Dice": "#2F8F4E"},
                     text="값", range_y=[0, 0.95])
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                          textfont_size=13, cliponaxis=False)
        fig.update_layout(
            height=440,
            xaxis_tickangle=-30,
            font=dict(size=14),
            xaxis_title=None,
            yaxis_title="점수",
            legend_title_text="지표",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=10, r=10, t=40, b=90),
        )
        fig.update_xaxes(tickfont_size=13)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.dataframe(SEG.style.format({"IoU": "{:.3f}", "Dice": "{:.3f}", "학습(분)": "{:.1f}"})
                     .background_gradient(subset=["IoU"], cmap="Greens"),
                     hide_index=True, use_container_width=True)
        st.caption("IoU 기준 색이 진할수록 우수. 6종 모두 거의 같은 색 = 성능이 비슷하다는 뜻.")

    st.success(" 6개 모델이 IoU 0.676~0.680으로 사실상 동일. "
               "아키텍처를 바꿔도 성능이 거의 움직이지 않는다는 것 -> 성능의 한계가 "
               "모델이 아니라 데이터(self-annotated 마스크·합성데이터)에서 온다는 강한 증거. "
               "→ 실제 성능은 임상 정답 마스크에서 재검증이 필요")

# ==================== 탭5 — 전체 파이프라인 ====================
with tab5:
    st.subheader("전체 추론 파이프라인 — 검출 → 중증도 분류")
    st.caption("한 장의 피부 이미지가 최종 진단 리포트로 나오기까지의 흐름")

    dot = """
    digraph pipeline {
        rankdir=LR;
        bgcolor="transparent";
        nodesep=0.35; ranksep=0.7;
        node [shape=box, style="rounded,filled", fontname="Malgun Gothic",
              fontsize=13, margin="0.22,0.14", penwidth=1.6];
        edge [color="#7B8794", penwidth=1.8, arrowsize=0.9];

        input [label="피부 병변 이미지\\n(입력)", fillcolor="#E3E8EF", color="#4B5563", fontcolor="#1F2933"];

        seg   [label="① 병변 검출\\nDeepLabV3+  (ResNet50)", fillcolor="#D5F0EB", color="#179E8E", fontcolor="#0B4F47"];
        segout[label="병변 마스크\\nIoU 0.680 · Dice 0.810", fillcolor="#EAF7F4", color="#179E8E", fontcolor="#0B4F47"];

        cls   [label="② 중증도 분류\\nCORN(ordinal)  또는  스태킹 앙상블", fillcolor="#FBE8CC", color="#E0982F", fontcolor="#7A4E12"];
        clsout[label="중증도 등급\\nMild / Moderate / Severe\\nQWK 0.634", fillcolor="#FDF3E2", color="#E0982F", fontcolor="#7A4E12"];

        result[label="최종 결과\\n병변 위치 + 중증도 리포트", fillcolor="#DDECD8", color="#2F8F4E", fontcolor="#1E5631"];

        input -> seg -> segout;
        input -> cls -> clsout;
        segout -> result;
        clsout -> result;
    }
    """
    st.graphviz_chart(dot, use_container_width=True)

    st.markdown("""
파이프라인 설명
- 입력: 피부 병변 사진 한 장이 두 갈래로 동시에 처리됩니다.
- ① 병변 검출: DeepLabV3+(백본 ResNet50)가 병변 영역을 픽셀 단위로 분할 → 병변 마스크(어디에 있는지). 성능 IoU 0.680 · Dice 0.810.
- ② 중증도 분류: CORN(ordinal 회귀) 또는 스태킹 앙상블이 이미지를 직접 학습해 등급(Mild/Moderate/Severe)을 예측. 성능 QWK 0.634.
- 최종 결과: '병변 위치(마스크)'와 '중증도 등급'을 합쳐 진단 리포트를 만듭니다 — 어디가 얼마나 심한지를 함께 제시.
""")
    st.info("검출과 분류는 서로 독립적인 두 모델입니다. 검출은 '위치', 분류는 '심각도'를 담당하며, "
            "두 결과를 결합해 임상적으로 해석 가능한 진단을 만듭니다.")

# ==================== 사이드바 — 결론 ====================
with st.sidebar:
    st.header("핵심 결론")
    st.markdown("""
1. 성능: CNN 직접 학습 > 앙상블 > 개수·면적 기반
2. ordinal 처리(CORN)가 단일 분류 대비 +0.04 (0.59→0.634), 특히 Severe recall 0.625→0.750
3. 앙상블 이득 미미 — 합성데이터라 백본 오차가 상관됨
4. 검출 모델은 다 동률 — 병목은 데이터
5. B 방식: 성능↓ 이지만 설명가능성↑

공통 한계: self-annotated 마스크 + 합성데이터.
실제 성능은 #508(임상 정답)에서 재검증 필요.
""")
    st.divider()
    st.caption("데이터: AI Hub #71863 · 3클래스 · 리허설/검증 실험")
