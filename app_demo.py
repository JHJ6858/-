#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
아토피 검출 데모 페이지 (Streamlit)
왼쪽: test 파일 목록 → 클릭 → 중간: 원본 / 정답마스크(초록) / 예측마스크(빨강) + 중증도·증상

준비: 코랩 make_demo_assets.py 실행 → demo_assets.zip 받아 이 파일 옆에 풀기
      (demo_assets/images, gt, pred, results.csv 구조)
실행: pip install streamlit pandas pillow numpy
      streamlit run app_demo.py
"""
import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops

st.set_page_config(page_title="아토피 검출 데모", layout="wide")
ASSETS = "demo_assets"
GT_COLOR = (0, 200, 0)      # 정답 = 초록
PR_COLOR = (255, 40, 40)    # 예측 = 빨강

SEV_COLOR = {"None": "#9AA5B1", "Mild": "#F2C94C", "Moderate": "#F2994A",
             "Severe": "#EB5757", "Clear": "#9AA5B1", "Almost Clear": "#BDBDBD"}


def _data_url():
    """클라우드 배포용: Streamlit secrets 또는 아래 상수에 demo_assets.zip 주소를 넣으면 자동 다운로드."""
    fallback = ""  # 예: "https://github.com/사용자/저장소/releases/download/v1/demo_assets.zip"
    try:
        return st.secrets.get("DEMO_ASSETS_URL", fallback)
    except Exception:
        return fallback


@st.cache_resource(show_spinner="데이터셋을 내려받는 중… (최초 1회, 1~2분 소요)")
def ensure_assets():
    """demo_assets가 없으면 지정된 URL에서 zip을 받아 압축 해제 (로컬에 있으면 그대로 사용)."""
    if os.path.exists(os.path.join(ASSETS, "results.csv")):
        return True
    url = _data_url()
    if not url:
        return False
    zp = "demo_assets_download.zip"
    urllib.request.urlretrieve(url, zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(".")
    return os.path.exists(os.path.join(ASSETS, "results.csv"))


@st.cache_data
def load_results():
    p = os.path.join(ASSETS, "results.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner="검증셋 임계값 스윕 계산 중… (최초 1회, 수 초 소요)")
def sweep_thresholds(filenames, thresholds):
    """검증셋 전체에서 임계값 τ별 평균 Dice/IoU/Precision/Recall (전경 있는 이미지 대상)."""
    ths = np.asarray(thresholds, dtype=np.float64)
    dice = np.zeros_like(ths); iou = np.zeros_like(ths)
    prec = np.zeros_like(ths); rec = np.zeros_like(ths)
    used = 0
    for fn in filenames:
        gp = f"{ASSETS}/gt/{fn}.png"; pp = f"{ASSETS}/prob/{fn}.png"
        if not (os.path.exists(gp) and os.path.exists(pp)):
            continue
        gt = np.array(Image.open(gp).convert("L")) > 127
        ga = int(gt.sum())
        if ga == 0:
            continue  # 정답 병변이 없는 이미지는 Dice/recall 정의가 어려워 제외
        prob = np.array(Image.open(pp).convert("L")).astype(np.float32) / 255.0
        all_sorted = np.sort(prob.ravel())
        gt_sorted = np.sort(prob[gt])
        # prob >= t 인 픽셀 수 = 전체 - (t보다 작은 것의 개수)
        pa = (all_sorted.size - np.searchsorted(all_sorted, ths, side="left")).astype(np.float64)
        tp = (gt_sorted.size - np.searchsorted(gt_sorted, ths, side="left")).astype(np.float64)
        denom = ga + pa
        union = ga + pa - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            dice += np.where(denom > 0, 2 * tp / denom, 0.0)
            iou += np.where(union > 0, tp / union, 0.0)
            prec += np.where(pa > 0, tp / pa, 0.0)
        rec += tp / ga
        used += 1
    if used == 0:
        return None
    return {"thr": ths, "dice": dice / used, "iou": iou / used,
            "precision": prec / used, "recall": rec / used, "n": used}


def overlay(img, mask, color, alpha=0.45):
    out = img.astype(float).copy()
    m = mask > 127
    for c in range(3):
        out[..., c][m] = (1 - alpha) * out[..., c][m] + alpha * color[c]
    return out.astype(np.uint8)


# 증상 ↔ 이미지 근거 특징 매핑
# (표시명, results.csv 정답컬럼, 특징키, 특징라벨, 정규화 기준max, '왜' 설명)
SYMPTOM_EVIDENCE = [
    ("홍반", "erythema", "redness", "붉기 지표 R-(G+B)/2", 80.0,
     "병변 영역이 붉을수록 홍반이 강함 (혈관 확장·염증)"),
    ("부종/구진", "papulation", "n_lesions", "구진 덩어리 개수", 15.0,
     "볼록한 병변 덩어리(blob)가 많을수록 구진/부종 경향"),
    ("상처", "excoriation", "eccentricity", "형태 길쭉함(이심률)", 1.0,
     "긁은 자국은 가늘고 길쭉함 → 이심률이 높음"),
    ("태선화", "lichenification", "contrast", "표면 거칠기(GLCM 대비)", 0.4,
     "피부가 두꺼워지고 주름지면 텍스처 대비가 커짐"),
]


def symptom_evidence(img_rgb, mask_bool):
    """예측 병변 영역에서 각 증상과 연결되는 해석용 이미지 특징을 계산."""
    area = int(mask_bool.sum())
    if area < 30:
        return None
    R = img_rgb[..., 0].astype(np.float32)
    G = img_rgb[..., 1].astype(np.float32)
    B = img_rgb[..., 2].astype(np.float32)
    redness = float((R - (G + B) / 2.0)[mask_bool].mean())          # 홍반

    lab, n = ndi.label(mask_bool)                                   # 연결요소(구진)
    props = [p for p in regionprops(lab) if p.area >= 20]
    n_lesions = len(props)                                          # 부종/구진
    eccs = sorted([p.eccentricity for p in props], reverse=True)[:5]
    eccentricity = float(np.mean(eccs)) if eccs else 0.0           # 상처

    gray = np.array(Image.fromarray(img_rgb).convert("L"))         # 태선화(텍스처)
    ys, xs = np.where(mask_bool)
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop_q = (crop // 32).astype(np.uint8)                          # 8단계 양자화
    glcm = graycomatrix(crop_q, distances=[1], angles=[0], levels=8,
                        symmetric=True, normed=True)
    contrast = float(graycoprops(glcm, "contrast")[0, 0])

    return {"redness": redness, "n_lesions": float(n_lesions),
            "eccentricity": eccentricity, "contrast": contrast,
            "area_ratio": area / mask_bool.size * 100}


def badge(label, value):
    color = SEV_COLOR.get(str(value), "#9AA5B1")
    st.markdown(
        f"<div style='display:inline-block;margin:3px;padding:6px 12px;border-radius:8px;"
        f"background:{color}22;border:1px solid {color};'>"
        f"<b>{label}</b><br><span style='font-size:1.1em'>{value}</span></div>",
        unsafe_allow_html=True)


st.title("🔬 아토피 병변 검출 — 예측 결과 뷰어")
st.caption("DeepLabV3+ 예측 · test셋 · 정답(초록) vs 예측(빨강) 비교")

# ==================== AI Hub 데이터 출처 표시 (저작권/이용조건 준수) ====================
st.markdown(
    "<div style='padding:10px 14px;border-radius:10px;border:1px solid #B0B8C1;"
    "background:rgba(120,130,150,0.08);font-size:0.9rem;line-height:1.6;'>"
    "📌 <b>데이터 출처</b> — 본 데모는 과학기술정보통신부의 재원으로 "
    "한국지능정보사회진흥원(NIA)의 지원을 받아 구축된 "
    "「안면부 피부질환 이미지 <b>합성데이터</b>」(AI 허브, 과제번호 71863)를 활용하여 제작되었습니다. "
    "데이터는 AI 허브(<a href='https://www.aihub.or.kr' target='_blank'>www.aihub.or.kr</a>)에서 내려받을 수 있습니다.<br>"
    "본 화면의 이미지·마스크는 <b>연구·시연 목적</b>으로만 표시되며, 원본 데이터의 저작권은 AI 허브/구축기관에 있습니다. "
    "무단 복제·재배포를 금합니다. (마스크는 self-annotated · 리허설/검증용)"
    "</div>",
    unsafe_allow_html=True)
st.write("")

ensure_assets()
df = load_results()
if df is None:
    st.error("demo_assets/ 를 찾을 수 없습니다. 로컬은 이 파일 옆에 demo_assets 폴더를 두고, "
             "클라우드 배포는 Streamlit secrets의 DEMO_ASSETS_URL에 demo_assets.zip 주소를 넣어주세요.")
    st.stop()

PROB_READY = os.path.isdir(f"{ASSETS}/prob")

tab_view, tab_sweep, tab_speed, tab_upload = st.tabs(
    ["🖼️ 이미지별 뷰어", "📈 임계값(τ) 분석", "⚡ 추론 속도", "➕ 이미지 추가"])

# ==================== 탭1 — 이미지별 뷰어 ====================
with tab_view:
    left, mid = st.columns([1, 3])
    with left:
        st.subheader("test 파일")
        st.caption(f"총 {len(df)}장")
        sort_by = st.radio("정렬", ["파일명", "IoU 높은순", "IoU 낮은순"], horizontal=False)
        d = df.copy()
        if sort_by == "IoU 높은순": d = d.sort_values("iou", ascending=False)
        elif sort_by == "IoU 낮은순": d = d.sort_values("iou", ascending=True)
        else: d = d.sort_values("filename")
        # IGA 필터
        igas = ["전체"] + sorted(df["iga"].dropna().unique().tolist())
        pick_iga = st.selectbox("IGA 필터", igas)
        if pick_iga != "전체":
            d = d[d["iga"] == pick_iga]
        sel = st.radio("파일 선택", d["filename"].tolist(),
                       format_func=lambda s: f"{s}  (IoU {df[df.filename==s]['iou'].values[0]:.2f})")

    with mid:
        row = df[df["filename"] == sel].iloc[0]
        img = np.array(Image.open(f"{ASSETS}/images/{sel}.png").convert("RGB"))
        gt = np.array(Image.open(f"{ASSETS}/gt/{sel}.png").convert("L"))

        st.subheader(f"📄 {sel}")

        # ----- 예측: 확률맵(prob)이 있으면 Confidence 임계값으로 실시간 판정 -----
        prob_path = f"{ASSETS}/prob/{sel}.png"
        has_prob = os.path.exists(prob_path)
        if has_prob:
            prob = np.array(Image.open(prob_path).convert("L")).astype(np.float32) / 255.0
            thr = st.slider("🎚️ 병변 판정 Confidence 임계값", 0.05, 0.95, 0.50, 0.05,
                            help="이 확률(신뢰도) 이상인 픽셀만 '병변'으로 검출합니다. "
                                 "높이면 확실한 부위만, 낮추면 넓게 검출됩니다.")
            st.caption(f"현재 설정: confidence ≥ {thr:.2f} 이면 병변으로 판정  ·  최적 τ는 '📈 임계값(τ) 분석' 탭 참고")
            pr_bool = prob >= thr
        else:
            pr = np.array(Image.open(f"{ASSETS}/pred/{sel}.png").convert("L"))
            pr_bool = pr > 127
            thr = None
            st.caption("ℹ️ 확률맵(prob)이 없어 임계값 조절은 비활성 상태입니다 "
                       "(기본 0.5로 판정된 마스크 표시). make_demo_assets.py를 재실행하면 슬라이더가 활성화됩니다.")
        pr_mask = (pr_bool.astype(np.uint8)) * 255

        # ----- 지표 실시간 계산 (임계값에 따라 갱신) -----
        gt_bool = gt > 127
        gt_area = int(gt_bool.sum()); pr_area = int(pr_bool.sum())
        inter = int((gt_bool & pr_bool).sum()); union = int((gt_bool | pr_bool).sum())
        iou = inter / union if union else 0.0
        dice = 2 * inter / (gt_area + pr_area) if (gt_area + pr_area) else 0.0

        m1, m2 = st.columns(2)
        m1.metric("이 이미지 IoU", f"{iou:.3f}")
        m2.metric("Dice", f"{dice:.3f}")

        st.markdown("##### 검출 결과 비교")
        i1, i2, i3 = st.columns(3)
        i1.image(img, caption="원본", use_container_width=True)
        i2.image(overlay(img, gt, GT_COLOR), caption="🟢 정답 마스크", use_container_width=True)
        _cap = f"🔴 예측 마스크 (임계값 {thr:.2f})" if has_prob else "🔴 예측 마스크"
        i3.image(overlay(img, pr_mask, PR_COLOR), caption=_cap, use_container_width=True)

        st.markdown("##### 중증도 · 증상 (정답 라벨)")

        # ----- 병변 인식률(재현율): 정답 병변 픽셀 중 모델이 맞게 검출한 비율 -----
        r1, r2 = st.columns(2)
        if gt_area > 0:
            recall = inter / gt_area * 100
            r1.metric("병변 인식률", f"{recall:.1f}%",
                      help="정답(실제) 병변 픽셀 중 모델이 맞게 검출한 비율 = 재현율(recall)")
            r1.caption("정답 병변 중 몇 %를 잡아냈는가")
        else:
            r1.metric("병변 인식률", "—")
            r1.caption("이 이미지엔 정답 병변이 없음")
        if pr_area > 0:
            precision = inter / pr_area * 100
            r2.metric("검출 정확도", f"{precision:.1f}%",
                      help="모델이 병변이라 예측한 픽셀 중 실제로 병변인 비율 = 정밀도(precision)")
            r2.caption("예측한 병변 중 실제 병변 비율")
        st.markdown(" ")
        badge("IGA 등급", row["iga"])

        # ----- 증상별: 정답 라벨 + 예측 병변 영역에서 뽑은 이미지 근거 -----
        st.markdown("###### 증상별 정답 라벨과 모델 근거")
        ev = symptom_evidence(img, pr_bool) if pr_area > 0 else None
        b = st.columns(4)
        for col, (kname, gtcol, feat, flabel, fmax, why) in zip(b, SYMPTOM_EVIDENCE):
            with col:
                st.markdown(f"**{kname}**")
                badge("정답", row.get(gtcol, "-"))
                if ev is not None:
                    val = ev[feat]
                    norm = max(0.0, min(1.0, val / fmax))
                    st.progress(norm)
                    st.caption(f"{flabel}: **{val:.2f}**")
                    st.caption(f"↳ {why}")
                else:
                    st.caption("병변 미검출 → 근거 계산 불가")

        with st.expander("ℹ️ '모델 근거'는 어떻게 계산되나요?"):
            st.markdown("""
이 뷰어의 DeepLabV3+ 모델은 **병변 위치(마스크)** 만 예측하고, 증상 등급을 직접 분류하지는 않습니다.
그래서 위 '근거'는 **예측된 병변 영역에서 뽑은 해석용 이미지 특징**입니다 (프로젝트 B방식의 설명가능성과 동일한 아이디어):

- **홍반** ← 병변 영역의 붉기 `R-(G+B)/2`
- **부종/구진** ← 병변 덩어리(연결요소) 개수
- **상처** ← 병변 형태의 길쭉함(이심률)
- **태선화** ← 표면 텍스처 거칠기(GLCM contrast)

막대는 각 특징을 대략적인 기준값으로 0~100% 정규화한 것으로, **정식 증상 분류기의 확률이 아니라 해석 보조 지표**입니다.
임계값 슬라이더를 움직이면 병변 영역이 바뀌므로 근거 수치도 함께 갱신됩니다.
""")

        with st.expander("두 마스크 겹쳐보기 (정답+예측)"):
            both = overlay(overlay(img, gt, GT_COLOR), pr_mask, PR_COLOR, alpha=0.4)
            st.image(both, caption="🟢정답 + 🔴예측 (겹치면 노랑빛)", use_container_width=True)
            st.caption("초록만=놓침(정답인데 예측X) · 빨강만=오검출 · 겹침=정확")

# ==================== 탭2 — 임계값(τ) 분석 ====================
with tab_sweep:
    st.subheader("임계값(τ) 스윕 — 검증셋 전체 기준 Dice / IoU / Precision / Recall")
    st.caption("τ를 0.02~0.98로 훑으며 검증셋 평균 성능이 어떻게 변하는지. Dice 최적점과 민감도(recall) 우선 권장점을 표시합니다.")

    if not PROB_READY:
        st.info("확률맵(prob)이 없어 스윕을 계산할 수 없습니다. make_demo_assets.py(확률맵 저장 버전)를 재실행해 주세요.")
        st.stop()

    THS = tuple(round(x, 2) for x in np.arange(0.02, 0.99, 0.02))
    res = sweep_thresholds(tuple(sorted(df["filename"].tolist())), THS)
    if res is None:
        st.warning("전경(병변)이 있는 이미지를 찾지 못해 스윕을 계산할 수 없습니다.")
        st.stop()

    thr_arr = res["thr"]
    # (1) Dice 최적 τ
    i_dice = int(np.argmax(res["dice"]))
    tau_dice = float(thr_arr[i_dice]); dice_max = float(res["dice"][i_dice])
    # (2) 민감도 우선 τ: Dice가 최대의 95% 이상인 구간 중 recall이 가장 높은(=가장 낮은) τ
    ok = np.where(res["dice"] >= 0.95 * dice_max)[0]
    i_rec = int(ok[0]) if len(ok) else i_dice     # THS 오름차순 → 첫 인덱스가 최저 τ = 최고 recall
    tau_rec = float(thr_arr[i_rec])

    # ----- 곡선 그래프 -----
    fig = go.Figure()
    for key, color in [("dice", "#2F8F4E"), ("iou", "#179E8E"), ("precision", "#E0982F"), ("recall", "#EB5757")]:
        fig.add_trace(go.Scatter(x=thr_arr, y=res[key], mode="lines", name=key.capitalize(),
                                 line=dict(color=color, width=2.5)))
    fig.add_vline(x=tau_dice, line_dash="dash", line_color="#2F8F4E",
                  annotation_text=f"Dice 최적 τ={tau_dice:.2f}", annotation_position="top")
    fig.add_vline(x=tau_rec, line_dash="dot", line_color="#EB5757",
                  annotation_text=f"민감도 우선 τ={tau_rec:.2f}", annotation_position="bottom")
    fig.update_layout(height=460, font=dict(size=14),
                      xaxis_title="임계값 τ (confidence)", yaxis_title="검증셋 평균 점수",
                      yaxis_range=[0, 1], legend=dict(orientation="h", y=1.08, x=0),
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ----- 권장 임계값 요약 -----
    c1, c2 = st.columns(2)
    c1.metric("Dice 최적 τ", f"{tau_dice:.2f}", f"Dice {dice_max:.3f}")
    c1.caption(f"이 지점: Recall {res['recall'][i_dice]:.3f} · Precision {res['precision'][i_dice]:.3f}")
    c2.metric("민감도(recall) 우선 τ", f"{tau_rec:.2f}", f"Recall {res['recall'][i_rec]:.3f}")
    c2.caption(f"이 지점: Dice {res['dice'][i_rec]:.3f} · Precision {res['precision'][i_rec]:.3f} "
               f"(Dice를 최대의 95% 이내로 유지하며 recall 최대화)")

    st.info(f"검증셋 {res['n']}장(정답 병변 있는 이미지) 기준. "
            "임상적으로 병변을 놓치지 않는 것이 중요하면 Dice 최적 τ보다 조금 낮은 '민감도 우선 τ'를 채택합니다.")

    st.markdown("""
근거 (임계값 선택 방법론)
- 병변은 전경이 작고 클래스가 불균형하므로 ROC보다 Precision–Recall 기반 지표로 τ를 선택하는 것이 타당함 — Saito & Rehmsmeier (2015), PLoS ONE.
- 문제 특성(불균형)·임상 위험(놓침)에 맞춰 지표·임계값을 정해야 함 — Maier-Hein, Reinke et al., "Metrics Reloaded", Nature Methods (2024).
- 본 그래프는 검증셋에서 τ를 스윕해 Dice를 최대화하는 τ를 찾고, 임상적으로 recall을 우선해 그보다 약간 낮은 τ를 채택하는 절차를 시각화한 것.
""")
    with st.expander("τ별 상세 수치 표 보기"):
        tbl = pd.DataFrame({"τ": thr_arr, "Dice": res["dice"], "IoU": res["iou"],
                            "Precision": res["precision"], "Recall": res["recall"]})
        st.dataframe(tbl.style.format({"τ": "{:.2f}", "Dice": "{:.3f}", "IoU": "{:.3f}",
                                       "Precision": "{:.3f}", "Recall": "{:.3f}"}),
                     hide_index=True, use_container_width=True, height=320)


# ==================== 탭3 — 추론 속도 ====================
with tab_speed:
    st.subheader("추론 속도 — DeepLabV3+ (ResNet50), 입력 512×512")
    st.caption("1장 처리(batch=1) 기준 예상 지연시간. 아래 값은 아키텍처·연산량 기반 추정이며, 코랩 실측값으로 교체할 수 있습니다.")

    # ▼ 코랩에서 잰 실측값(ms/장)을 채우면 표 상단에 '실측' 행으로 추가됩니다. 없으면 None.
    MEASURED = {
        "코랩 GPU (A100) — 실측": 8.1,
        "코랩 CPU (6스레드 Xeon) — 실측": 751.1,
    }

    st.markdown("**① 서버·PC 급**")
    rows = [
        {"환경": "고성능 GPU (A100/V100)", "예상 지연(장당)": "5 ~ 15 ms", "처리량(장/초)": "65 ~ 200", "구분": "추정"},
        {"환경": "서버·코랩 GPU (T4~L4급)", "예상 지연(장당)": "8 ~ 40 ms", "처리량(장/초)": "25 ~ 125", "구분": "추정"},
        {"환경": "노트북 CPU (Ryzen 7 7735HS, 8C/16T)", "예상 지연(장당)": "0.3 ~ 0.5 s", "처리량(장/초)": "2 ~ 3.3", "구분": "추정"},
    ]
    measured_rows = []
    for name, ms in MEASURED.items():
        if ms is not None:
            sec = f"{ms/1000:.2f} s" if ms >= 1000 else f"{ms:.1f} ms"
            measured_rows.append({"환경": name, "예상 지연(장당)": sec,
                                  "처리량(장/초)": f"{1000.0/ms:.2f}", "구분": "실측"})
    st.dataframe(pd.DataFrame(measured_rows + rows), hide_index=True, use_container_width=True)
    st.caption("노트북 Ryzen 7 7735HS는 코랩 CPU(6스레드)보다 코어·IPC·AVX2에서 앞서 대략 1.5~2.5배 빠를 것으로 추정. "
               "내장 Radeon(iGPU)은 PyTorch 기본 CPU 경로에서 사용되지 않습니다(DirectML/ROCm 별도 필요).")

    st.markdown("**② 스마트폰 칩별 예상 — CPU만 사용(NPU/GPU 미사용) 기준**")
    st.caption("현재 무거운 모델(ResNet50, 512×512)을 모바일 CPU만으로 돌릴 때의 대략적 예상. 실측 아님(추정).")
    mob = pd.DataFrame([
        {"기기 예시": "iPhone 15 Pro", "칩(CPU)": "Apple A17 Pro", "CPU만 예상": "1.5 ~ 3 s", "NPU/GPU 가속 시": "0.15 ~ 0.5 s"},
        {"기기 예시": "iPhone 14 Pro / 15", "칩(CPU)": "Apple A16", "CPU만 예상": "2 ~ 3.5 s", "NPU/GPU 가속 시": "0.2 ~ 0.6 s"},
        {"기기 예시": "Galaxy S24 (일부)", "칩(CPU)": "Snapdragon 8 Gen 3", "CPU만 예상": "2 ~ 4 s", "NPU/GPU 가속 시": "0.2 ~ 0.6 s"},
        {"기기 예시": "Galaxy S23", "칩(CPU)": "Snapdragon 8 Gen 2", "CPU만 예상": "2.5 ~ 4.5 s", "NPU/GPU 가속 시": "0.25 ~ 0.7 s"},
        {"기기 예시": "Galaxy S24 (Exynos)", "칩(CPU)": "Exynos 2400", "CPU만 예상": "3 ~ 5 s", "NPU/GPU 가속 시": "0.3 ~ 0.8 s"},
        {"기기 예시": "중급기 (Galaxy A 등)", "칩(CPU)": "Snapdragon 7s급", "CPU만 예상": "5 ~ 10 s", "NPU/GPU 가속 시": "0.6 ~ 1.5 s"},
    ])
    st.dataframe(mob, hide_index=True, use_container_width=True)
    st.caption("→ CPU만으로는 수 초가 걸려 실사용이 어렵고, NPU/GPU(TFLite NNAPI·CoreML) 위임 + 양자화가 사실상 필수입니다.")

    st.markdown("""
휴대폰 앱으로 구현할 때 예상 (핵심)
- **현재 모델(DeepLabV3+ / ResNet50, output_stride 8)은 모바일 기준 무거움** (약 40M 파라미터, dilated conv로 연산량 큼).
  고급 스마트폰에서도 FP16 기준 **약 0.4~0.9초/장** 수준으로, 실시간(수십 ms)엔 부적합할 수 있습니다.
- **모바일 최적화 시 예상**: 백본을 **MobileNetV3**로 교체 + **output_stride 16** + **INT8 양자화** + **입력 256~384로 축소**
  → 대략 **30~120 ms/장**(고급폰)까지 단축 가능 → 준실시간 사용 가능.
- **변환 경로**: PyTorch → ONNX → TFLite(안드로이드, NNAPI/GPU delegate) 또는 CoreML(iOS). 양자화(INT8)로 용량·지연 추가 감소.
""")
    with st.expander("🔍 왜 A100과 모바일·PC CPU는 이렇게 차이가 날까? (메모리 병목 등 원인)"):
        st.markdown("같은 코랩 환경에서 **A100 8.1 ms vs CPU 751 ms ≈ 93배**(실측), 모바일 CPU까지 가면 수백 배 차이. "
                    "이는 아래 요인들이 겹친 결과입니다.")
        spec = pd.DataFrame([
            {"하드웨어": "NVIDIA A100", "병렬 연산유닛": "6912 CUDA + Tensor Core", "연산 처리량(대략)": "~156 TFLOPS (TF32)", "메모리 대역폭": "HBM2e ~1,555 GB/s"},
            {"하드웨어": "데스크톱 CPU", "병렬 연산유닛": "8~16 코어", "연산 처리량(대략)": "~0.5~1 TFLOPS", "메모리 대역폭": "DDR4/5 ~50~90 GB/s"},
            {"하드웨어": "모바일 CPU(고급폰)", "병렬 연산유닛": "6~8 코어", "연산 처리량(대략)": "~0.05~0.15 TFLOPS", "메모리 대역폭": "LPDDR5 ~50~68 GB/s"},
        ])
        st.dataframe(spec, hide_index=True, use_container_width=True)
        st.markdown("""
주요 원인
1. **병렬성(연산 처리량) 차이** — A100은 수천 개 코어 + Tensor Core로 행렬곱(MAC)을 수만 개 동시에 처리합니다. 모바일 CPU는 코어가 수십 배 적어 순수 연산량만으로도 **1,000배 이상** 차이가 납니다.

2. **메모리 대역폭 병목** — batch=1 추론은 연산량보다 **활성값·가중치를 메모리에서 얼마나 빨리 나르느냐**에 좌우되는 memory-bound 구간이 많습니다(roofline 관점). A100의 HBM2e(~1.5 TB/s)와 모바일 LPDDR5(~60 GB/s)는 **약 25배** 차이라, 코어가 있어도 데이터를 못 받아 놀게 됩니다.

3. **DeepLabV3+의 dilated convolution 특성** — output_stride 8로 **고해상도 특징맵을 끝까지 유지**해 활성 텐서가 매우 큽니다. 이 큰 텐서가 CPU 캐시를 초과해 메모리 왕복이 폭증 → CPU에서 특히 불리합니다.

4. **전용 가속기 유무** — GPU는 cuDNN·Tensor Core로 conv/GEMM에 최적화된 커널을 쓰지만, 범용 CPU는 그런 전용 유닛이 없어 커널 효율이 낮습니다.

5. **데이터 타입** — GPU는 FP16/TF32로 처리량을 더 높이는 반면, CPU는 보통 FP32라 같은 연산도 느립니다.

한 줄 요약 — 속도차는 단순히 "GPU가 빠르다"가 아니라 **① 압도적 병렬 연산량 + ② 수십 배 넓은 메모리 대역폭**이 핵심이며, 고해상도 dilated conv 구조가 그 격차를 더 벌립니다. 그래서 모바일에선 **경량 백본·양자화·입력 축소·NPU 위임**으로 연산량과 메모리 이동을 함께 줄여야 합니다.
""")

    st.info("정확한 수치는 아래 '코랩 벤치마크 코드'로 실제 모델을 재서 위 MEASURED에 넣으면 표에 '실측' 행으로 반영됩니다. "
            "(코드는 채팅 참고)")

# ==================== 탭4 — 이미지 추가 ====================
with tab_upload:
    st.subheader("새 이미지 테스트")
    st.caption("새 피부 이미지를 올려 병변 검출과 증상 근거를 확인합니다. "
               "정답 마스크가 없으므로 IoU 등 정답기반 지표는 생략됩니다.")

    up_img = st.file_uploader("피부 이미지 (png/jpg)", type=["png", "jpg", "jpeg"], key="up_img")
    up_prob = st.file_uploader(
        "(선택) 코랩 DeepLabV3+로 만든 확률맵 prob PNG — 올리면 실제 모델 결과로 분석",
        type=["png"], key="up_prob")

    if up_img is None:
        st.info("이미지를 업로드하면 분석이 시작됩니다. 실제 모델 결과를 보려면 코랩에서 만든 prob PNG도 함께 올려주세요.")
    else:
        nimg = np.array(Image.open(up_img).convert("RGB").resize((512, 512)))

        if up_prob is not None:
            nprob = np.array(Image.open(up_prob).convert("L").resize((512, 512))).astype(np.float32) / 255.0
            t = st.slider("🎚️ Confidence 임계값", 0.05, 0.95, 0.50, 0.05, key="up_thr")
            nmask = nprob >= t
            st.success("실제 모델 확률맵(prob)으로 분석 중입니다.")
        else:
            # 붉기 기반 간이 검출 (딥러닝 모델 아님 · 참고용 미리보기)
            R = nimg[..., 0].astype(np.float32); G = nimg[..., 1].astype(np.float32); B = nimg[..., 2].astype(np.float32)
            red = R - (G + B) / 2.0
            m0 = red > (red.mean() + 0.8 * red.std())
            m0 = ndi.binary_opening(m0, iterations=2)
            m0 = ndi.binary_closing(m0, iterations=2)
            lab, n = ndi.label(m0)
            if n:
                sizes = ndi.sum(np.ones_like(lab), lab, index=range(1, n + 1))
                m0 = np.isin(lab, np.where(sizes >= 80)[0] + 1)
            nmask = m0
            st.warning("⚠️ 확률맵이 없어 **붉기 기반 간이 검출(딥러닝 모델 아님)** 으로 미리보기 중입니다. "
                       "정확한 결과는 코랩에서 DeepLabV3+로 prob을 생성해 함께 올려주세요.")

        area = int(nmask.sum())
        nmask_u8 = (nmask.astype(np.uint8)) * 255
        cc1, cc2 = st.columns(2)
        cc1.image(nimg, caption="원본", use_container_width=True)
        cc2.image(overlay(nimg, nmask_u8, PR_COLOR), caption="🔴 검출 영역", use_container_width=True)
        st.metric("검출 병변 면적 비율", f"{area / nmask.size * 100:.1f}%")

        st.markdown("###### 증상별 이미지 근거")
        ev = symptom_evidence(nimg, nmask) if area > 30 else None
        cols = st.columns(4)
        for col, (kname, gtcol, feat, flabel, fmax, why) in zip(cols, SYMPTOM_EVIDENCE):
            with col:
                st.markdown(f"**{kname}**")
                if ev is not None:
                    val = ev[feat]; norm = max(0.0, min(1.0, val / fmax))
                    st.progress(norm)
                    st.caption(f"{flabel}: **{val:.2f}**")
                    st.caption(f"↳ {why}")
                else:
                    st.caption("검출 영역이 작아 계산 불가")
        st.caption("※ 새 이미지는 정답 라벨이 없어 증상 '정답'은 표시하지 않습니다. 위 값은 해석 보조 지표입니다.")

# ==================== 하단 출처 푸터 ====================
st.divider()
st.markdown(
    "<div style='font-size:0.82rem;color:#8894A5;line-height:1.6;'>"
    "본 저작물(데모)은 과학기술정보통신부의 재원으로 한국지능정보사회진흥원(NIA)의 지원을 받아 구축된 "
    "「안면부 피부질환 이미지 합성데이터」(AI 허브, 71863)를 활용하여 제작되었습니다. "
    "데이터 출처: AI 허브(www.aihub.or.kr) · 원본 데이터 저작권은 AI 허브/구축기관에 있으며, 본 화면은 연구·시연용으로 무단 재배포를 금합니다."
    "</div>",
    unsafe_allow_html=True)
