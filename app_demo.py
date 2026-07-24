#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
아토피 검출 데모 (Streamlit) — 실시간 추론 버전
앱에서 TFLite 모델을 직접 실행해 결과를 표시합니다.
 · 병변 분할: EfficientNet-B0 + UNet++
 · 중증도 분류: PVTv2-B0 + CORN (IGA + 4증상 멀티헤드)
"""
import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

import models as M

st.set_page_config(page_title="아토피 검출 데모", layout="wide")

ASSETS = "demo_assets"


def _find_model(fname):
    """models/ 폴더 또는 저장소 루트 어디에 있든 .tflite를 찾음."""
    for p in (os.path.join("models", fname), fname):
        if os.path.exists(p):
            return p
    return os.path.join("models", fname)  # 없을 때(에러 메시지용 기본 경로)


SEG_PATH = _find_model("seg_effb0_unetpp_512_fp16.tflite")
SEV_PATH = _find_model("sev_pvt_v2_b0_corn_crop_area_512_fp16.tflite")

GT_COLOR = (0, 200, 0)      # 정답 = 초록
PR_COLOR = (255, 40, 40)    # 예측 = 빨강

SEV_COLOR = {"None": "#9AA5B1", "Mild": "#F2C94C", "Moderate": "#F2994A",
             "Severe": "#EB5757", "Clear": "#9AA5B1", "Almost Clear": "#BDBDBD"}

# 등급 비교(예측 vs 정답) 색 — 맞춤=초록, 틀림=빨강
GRADE_OK = "#27AE60"
GRADE_BAD = "#EB5757"

METRIC_KO = {"iga": "IGA 등급", "erythema": "홍반", "papulation": "구진/부종",
             "excoriation": "상처", "lichenification": "태선화"}

# ==================== 모델 카드 (보고된 성능 · 학습 결과 고정값) ====================
SEG_CARD = {"dice": 0.8318, "iou": 0.7335, "res": 512, "params": "6.57M"}
SEV_CARD = {"iga": 0.720, "params": "3.41M",
            "erythema": 0.619, "papulation": 0.440,
            "excoriation": 0.567, "lichenification": 0.589}  # 증상은 QWK 기준
SEV_CARD_MEAN_QWK = 0.587  # 5개 헤드 평균 QWK



# ==================== 모델 로딩 (캐시) ====================
@st.cache_resource(show_spinner="병변분할 모델(EfficientNet-B0+UNet++) 로딩 중…")
def get_seg():
    return M.SegModel(SEG_PATH)


@st.cache_resource(show_spinner="중증도 모델(PVTv2-B0+CORN) 로딩 중…")
def get_sev():
    return M.SevModel(SEV_PATH)


# ==================== 데이터셋(뷰어용) ====================
def _data_url():
    fallback = "https://github.com/JHJ6858/<저장소이름>/releases/download/v2/demo_assets.zip"
    try:
        return st.secrets.get("DEMO_ASSETS_URL", fallback)
    except Exception:
        return fallback



@st.cache_resource(show_spinner="데이터셋을 내려받는 중… (최초 1회)")
def ensure_assets():
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
    return pd.read_csv(p) if os.path.exists(p) else None


# ==================== 유틸 ====================
def to512(rgb):
    return np.asarray(Image.fromarray(np.asarray(rgb)).convert("RGB").resize((512, 512), Image.BILINEAR))


def overlay(img, mask_bool, color, alpha=0.45):
    out = np.asarray(img).astype(float).copy()
    m = np.asarray(mask_bool, bool)
    for c in range(3):
        out[..., c][m] = (1 - alpha) * out[..., c][m] + alpha * color[c]
    return out.astype(np.uint8)


def badge(label, value, color=None):
    if color is None:
        color = SEV_COLOR.get(str(value), "#9AA5B1")
    st.markdown(
        f"<div style='display:inline-block;margin:3px;padding:6px 12px;border-radius:8px;"
        f"background:{color}22;border:1px solid {color};'>"
        f"<b>{label}</b><br><span style='font-size:1.1em'>{value}</span></div>",
        unsafe_allow_html=True)


@st.cache_data(show_spinner="검증셋 세그 추론 + 임계값 스윕 중… (최초 1회, 이미지 수에 따라 수십 초)")
def sweep_live(filenames, thresholds, seg_tag):
    """검증셋 전체에 세그 모델을 실행해 τ별 평균 Dice/IoU/Precision/Recall 계산."""
    seg = get_seg()
    ths = np.asarray(thresholds, np.float64)
    dice = np.zeros_like(ths); iou = np.zeros_like(ths)
    prec = np.zeros_like(ths); rec = np.zeros_like(ths)
    used = 0
    for fn in filenames:
        ip = f"{ASSETS}/images/{fn}.png"; gp = f"{ASSETS}/gt/{fn}.png"
        if not (os.path.exists(ip) and os.path.exists(gp)):
            continue
        gt = np.array(Image.open(gp).convert("L").resize((512, 512), Image.NEAREST)) > 127
        ga = int(gt.sum())
        if ga == 0:
            continue
        orig = np.array(Image.open(ip).convert("RGB"))
        prob = M.sigmoid(seg.predict_logit(orig)).astype(np.float32)  # 512x512
        all_sorted = np.sort(prob.ravel())
        gt_sorted = np.sort(prob[gt])
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


@st.cache_data(show_spinner="정렬용 세그 추론 중… (최초 1회, 이미지 수에 따라 수십 초)")
def per_image_scores(filenames, tau, seg_tag):
    """정렬용: 이미지별 Dice/IoU를 세그 모델로 계산 (정답 마스크 있는 것만)."""
    seg = get_seg()
    out = {}
    for fn in filenames:
        ip = f"{ASSETS}/images/{fn}.png"; gp = f"{ASSETS}/gt/{fn}.png"
        dice = iou = float("nan")
        if os.path.exists(ip) and os.path.exists(gp):
            orig = np.array(Image.open(ip).convert("RGB"))
            mask = M.sigmoid(seg.predict_logit(orig)) >= tau
            gt = np.array(Image.open(gp).convert("L").resize((512, 512), Image.NEAREST)) > 127
            inter = int((gt & mask).sum()); uni = int((gt | mask).sum())
            ga = int(gt.sum()); pa = int(mask.sum())
            if uni:
                iou = inter / uni
            if ga + pa:
                dice = 2 * inter / (ga + pa)
        out[fn] = {"dice": dice, "iou": iou}
    return out


def show_grades(grades, gt_row=None):
    """예측 등급(모델) + (있으면) 정답 등급(csv)을 나란히 표시.
    정답이 있으면 예측·정답 배지 색을 일치 여부로 표시(맞춤=초록, 틀림=빨강).
    정답 값이 비어있으면 'None'으로 표기."""
    cols = st.columns(5)
    for col, metric in zip(cols, M.SEV_ORDER):
        with col:
            st.markdown(f"**{METRIC_KO[metric]}**")

            pred_label = M.grade_label(metric, grades[metric]) if metric in grades else None

            # 정답 컬럼이 존재하면(gt_row 있음), NaN이어도 'None' 으로 표기
            gt_label = None
            if gt_row is not None and metric in gt_row:
                gt_label = "None" if pd.isna(gt_row[metric]) else str(gt_row[metric])

            # 예측·정답 둘 다 있으면 일치 여부로 색 결정
            cmp_color = None
            if gt_label is not None and pred_label is not None:
                cmp_color = GRADE_OK if str(pred_label) == gt_label else GRADE_BAD

            if pred_label is not None:
                badge("예측", pred_label, color=cmp_color)
            else:
                st.caption("출력 매핑 실패")

            if gt_label is not None:
                badge("정답", gt_label, color=cmp_color)


# ==================== 성능 평가 계산 ====================
def _qwk(cm):
    """Quadratic Weighted Kappa (순서형 등급 일치도)."""
    cm = np.asarray(cm, float); K = cm.shape[0]; N = cm.sum()
    if N == 0 or K < 2:
        return float("nan")
    idx = np.arange(K)
    w = ((idx[:, None] - idx[None, :]) ** 2) / (K - 1) ** 2
    E = np.outer(cm.sum(1), cm.sum(0)) / N
    denom = (w * E).sum()
    if denom == 0:
        return float("nan")
    return 1.0 - (w * cm).sum() / denom


def _cm_metrics(cm):
    cm = np.asarray(cm, float); N = cm.sum(); K = cm.shape[0]
    if N == 0:
        return {"n": 0, "acc": float("nan"), "mae": float("nan"), "qwk": float("nan")}
    idx = np.arange(K)
    acc = np.trace(cm) / N
    mae = (np.abs(idx[:, None] - idx[None, :]) * cm).sum() / N
    return {"n": int(N), "acc": float(acc), "mae": float(mae), "qwk": _qwk(cm)}


@st.cache_data(show_spinner="전체 테스트셋 추론 + 성능 평가 중… (최초 1회, 이미지 수에 따라 수십 초~수 분)")
def evaluate_all(filenames, tau, crop_mode, seg_tag, sev_tag):
    """테스트셋 전체에 두 모델을 실행해 세그·중증도 지표를 집계.
    crop_mode='pred' : 예측 마스크로 크롭 (end-to-end, 실제 배포 성능)
    crop_mode='gt'   : 정답 마스크로 크롭 (중증도 모델 단독 성능, 카드 수치 재현)
    """
    seg = get_seg(); sev = get_sev()
    dices, ious = [], []
    tp = fp = fn = 0
    conf = {m: np.zeros((len(M.IGA_CLASSES if m == "iga" else M.SYMPTOM_CLASSES),) * 2, int)
            for m in M.SEV_ORDER}
    rows = {r["filename"]: r for _, r in df.iterrows()} if df is not None else {}
    for name in filenames:
        ip = f"{ASSETS}/images/{name}.png"
        if not os.path.exists(ip):
            continue
        orig = np.array(Image.open(ip).convert("RGB"))
        mask = M.sigmoid(seg.predict_logit(orig)) >= tau     # 512x512 bool (예측)
        # ---- 세그 지표 + GT 마스크 로드 ----
        gp = f"{ASSETS}/gt/{name}.png"
        gt = None
        if os.path.exists(gp):
            gt = np.array(Image.open(gp).convert("L").resize((512, 512), Image.NEAREST)) > 127
            if gt.sum() > 0:
                inter = int((gt & mask).sum()); uni = int((gt | mask).sum())
                ga = int(gt.sum()); pa = int(mask.sum())
                ious.append(inter / uni if uni else 0.0)
                dices.append(2 * inter / (ga + pa) if (ga + pa) else 0.0)
                tp += inter; fp += pa - inter; fn += ga - inter
        # ---- 중증도: 크롭 소스 선택 ----
        if crop_mode == "gt" and gt is not None:
            cmask = gt                                       # 정답 마스크로 크롭
        else:
            cmask = mask                                     # 예측 마스크로 크롭
        area = float(cmask.sum()) / cmask.size
        crop = M.lesion_bbox_crop(orig, cmask, pad=0.15)
        grades = sev.predict(crop, area)
        r = rows.get(name)
        if r is not None:
            for m in M.SEV_ORDER:
                classes = M.IGA_CLASSES if m == "iga" else M.SYMPTOM_CLASSES
                gl = r.get(m)
                if pd.isna(gl) or gl not in classes:
                    continue
                pi = grades.get(m)
                if pi is None:
                    continue
                ti = classes.index(gl)
                pi = max(0, min(int(pi), len(classes) - 1))
                conf[m][ti, pi] += 1
    seg_res = {"n": len(dices),
               "dice": float(np.mean(dices)) if dices else float("nan"),
               "iou": float(np.mean(ious)) if ious else float("nan"),
               "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
               "recall": tp / (tp + fn) if (tp + fn) else float("nan")}
    sev_res = {m: {"cm": conf[m].tolist(), **_cm_metrics(conf[m])} for m in M.SEV_ORDER}
    return {"seg": seg_res, "sev": sev_res}


def conf_heatmap(cm, classes, title):
    cm = np.asarray(cm)
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=classes, y=classes, colorscale="Blues",
        text=cm, texttemplate="%{text}", textfont=dict(size=13),
        showscale=False, xgap=2, ygap=2))
    fig.update_layout(title=dict(text=title, font=dict(size=15)),
                      xaxis=dict(title="예측", side="top"),
                      yaxis=dict(title="정답", autorange="reversed"),
                      height=300, margin=dict(l=10, r=10, t=60, b=10), font=dict(size=12))
    return fig


def model_cards():
    """학습 때 보고된 성능 카드(고정 숫자)를 표시."""
    seg_html = (
        "<div style='background:linear-gradient(135deg,#16233f,#241a3a);color:#fff;"
        "border-radius:12px;padding:14px 18px;height:100%'>"
        "<div style='color:#7fe3c0;font-weight:700;font-size:1.02rem'>EfficientNet-B0 + UNet++</div>"
        f"<div style='font-size:1.45rem;font-weight:800;margin:6px 0'>DICE {SEG_CARD['dice']:.4f}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;IoU {SEG_CARD['iou']:.4f}</div>"
        f"<div style='color:#aeb8cf;font-size:0.85rem'>해상도 {SEG_CARD['res']} · Params {SEG_CARD['params']}</div>"
        "</div>")
    syms = " &nbsp;|&nbsp; ".join(
        f"{s} {SEV_CARD[k]:.3f}" for k, s in
        [("erythema", "홍반"), ("papulation", "구진"), ("excoriation", "상처"), ("lichenification", "태선")])
    sev_html = (
        "<div style='background:linear-gradient(135deg,#2a1a3f,#16233f);color:#fff;"
        "border-radius:12px;padding:14px 18px;height:100%'>"
        "<div style='color:#f0c674;font-weight:700;font-size:1.02rem'>PVTv2-B0 (Best IGA)</div>"
        f"<div style='font-size:1.45rem;font-weight:800;margin:6px 0'>IGA {SEV_CARD['iga']:.3f}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;평균 QWK {SEV_CARD_MEAN_QWK:.3f}&nbsp;&nbsp;|&nbsp;&nbsp;Params {SEV_CARD['params']}</div>"
        f"<div style='color:#aeb8cf;font-size:0.85rem'>{syms}</div>"
        "</div>")
    c1, c2 = st.columns(2)
    c1.markdown(seg_html, unsafe_allow_html=True)
    c2.markdown(sev_html, unsafe_allow_html=True)
    st.caption("📋 학습 때 보고된 성능(모델 카드). 세그는 Dice/IoU, 중증도는 QWK 기준. "
               "아래 '성능 평가 실행' 결과와 비교해 보세요.")



# ==================== 헤더 ====================
st.title("🔬 아토피 병변 검출 — 실시간 추론 데모")
st.caption("EfficientNet-B0 + UNet++ 병변분할 · PVTv2-B0 + CORN 중증도 분류 · 앱에서 직접 추론")

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

# ---- 모델 존재 확인 ----
if not (os.path.exists(SEG_PATH) and os.path.exists(SEV_PATH)):
    st.error(f"모델 파일을 찾을 수 없습니다. 저장소에 다음 파일을 두세요:\n\n- `{SEG_PATH}`\n- `{SEV_PATH}`")
    st.stop()

ensure_assets()
df = load_results()  # 없으면 None (뷰어 탭 비활성)

tab_view, tab_sweep, tab_eval, tab_upload = st.tabs(
    ["🖼️ 이미지별 뷰어", "📈 임계값(τ) 분석", "📊 성능 평가", "➕ 이미지 추가"])

# ==================== 탭1 — 이미지별 뷰어 ====================
with tab_view:
    if df is None:
        st.info("demo_assets/results.csv 가 없어 테스트셋 뷰어를 사용할 수 없습니다. "
                "'➕ 이미지 추가' 탭에서 직접 이미지를 올려 실시간 추론을 확인하세요.")
    else:
        left, mid = st.columns([1, 3])
        with left:
            st.subheader("test 파일")
            st.caption(f"총 {len(df)}장")
            d = df.copy()
            if "iga" in df.columns:
                igas = ["전체"] + sorted(df["iga"].dropna().unique().tolist())
                pick_iga = st.selectbox("IGA(정답) 필터", igas)
                if pick_iga != "전체":
                    d = d[d["iga"] == pick_iga]

            sort_by = st.radio("정렬", ["파일명", "Dice 높은순", "Dice 낮은순",
                                        "IoU 높은순", "IoU 낮은순"])
            if sort_by == "파일명":
                d = d.sort_values("filename")
                fmt = lambda s: s
            else:
                key = "dice" if "Dice" in sort_by else "iou"
                seg_tag = f"{SEG_PATH}:{os.path.getmtime(SEG_PATH)}"
                scores = per_image_scores(tuple(sorted(df["filename"].tolist())), 0.50, seg_tag)
                d = d.assign(_score=d["filename"].map(lambda f: scores.get(f, {}).get(key, float("nan"))))
                d = d.sort_values("_score", ascending=("낮은순" in sort_by), na_position="last")
                fmt = lambda s: f"{s}  ({key.upper()} {scores.get(s, {}).get(key, float('nan')):.2f})"
                st.caption("정렬 점수는 τ=0.50 기준 (이미지별 슬라이더와 별개)")

            sel = st.radio("파일 선택", d["filename"].tolist(), format_func=fmt)

        with mid:
            row = df[df["filename"] == sel].iloc[0]
            orig = np.array(Image.open(f"{ASSETS}/images/{sel}.png").convert("RGB"))
            has_gt = os.path.exists(f"{ASSETS}/gt/{sel}.png")

            st.subheader(f"📄 {sel}")
            thr = st.slider("🎚️ 병변 판정 Confidence 임계값", 0.05, 0.95, 0.50, 0.05,
                            help="sigmoid(세그 logit) ≥ 이 값이면 병변으로 판정. "
                                 "높이면 확실한 부위만, 낮추면 넓게 검출됩니다.")

            out = M.run_full(orig, get_seg(), get_sev(), tau=thr)
            mask = out["mask"]                                # 512x512 bool
            disp = to512(orig)

            # 지표 (정답 마스크 있으면)
            if has_gt:
                gt = np.array(Image.open(f"{ASSETS}/gt/{sel}.png").convert("L").resize((512, 512), Image.NEAREST)) > 127
                inter = int((gt & mask).sum()); union = int((gt | mask).sum())
                ga = int(gt.sum()); pa = int(mask.sum())
                iou = inter / union if union else 0.0
                dice = 2 * inter / (ga + pa) if (ga + pa) else 0.0
                m1, m2 = st.columns(2)
                m1.metric("이 이미지 IoU", f"{iou:.3f}")
                m2.metric("Dice", f"{dice:.3f}")

            st.markdown("##### 검출 결과 비교")
            i1, i2, i3 = st.columns(3)
            i1.image(disp, caption="원본", use_container_width=True)
            if has_gt:
                i2.image(overlay(disp, gt, GT_COLOR), caption="🟢 정답 마스크", use_container_width=True)
            else:
                i2.image(disp, caption="(정답 마스크 없음)", use_container_width=True)
            i3.image(overlay(disp, mask, PR_COLOR), caption=f"🔴 예측 마스크 (τ={thr:.2f})",
                     use_container_width=True)

            

            st.markdown("##### 중증도 · 증상 — 예측/정답 배지 · 🟢 정답 맞춤 vs 🔴 정답 틀림")
            show_grades(out["grades"], gt_row=row)
            st.caption(f"검출 병변 면적비: {out['area'] * 100:.2f}%  ·  중증도 입력 크롭 크기: {out['crop'].shape[1]}×{out['crop'].shape[0]}")

            with st.expander("🔎 중증도 모델에 입력된 크롭 이미지 보기"):
                st.image(out["crop"], caption="세그 마스크 bbox + 15% 패딩 크롭 → PVTv2 입력",
                         use_container_width=False, width=280)

            with st.expander("두 마스크 겹쳐보기 (정답+예측)"):
                if has_gt:
                    both = overlay(overlay(disp, gt, GT_COLOR), mask, PR_COLOR, alpha=0.4)
                    st.image(both, caption="🟢정답 + 🔴예측 (겹치면 노랑빛)", use_container_width=True)
                    st.caption("초록만=놓침 · 빨강만=오검출 · 겹침=정확")
                else:
                    st.caption("정답 마스크가 없어 비교 불가")

# ==================== 탭2 — 임계값(τ) 분석 ====================
with tab_sweep:
    st.subheader("임계값(τ) 스윕 — 검증셋 전체 실시간 추론 기준")
    st.caption("검증셋 전체에 세그 모델을 실행해 τ별 평균 Dice/IoU/Precision/Recall을 계산합니다. "
               "(최초 1회 전체 추론 후 캐시)")

    if df is None:
        st.info("demo_assets/results.csv 가 없어 스윕을 계산할 수 없습니다.")
    elif st.button("▶ 스윕 계산 실행", type="primary"):
        THS = tuple(round(x, 2) for x in np.arange(0.02, 0.99, 0.02))
        seg_tag = f"{SEG_PATH}:{os.path.getmtime(SEG_PATH)}"
        res = sweep_live(tuple(sorted(df["filename"].tolist())), THS, seg_tag)
        if res is None:
            st.warning("전경(병변) 있는 이미지를 찾지 못해 스윕을 계산할 수 없습니다.")
        else:
            thr_arr = res["thr"]
            i_dice = int(np.argmax(res["dice"]))
            tau_dice = float(thr_arr[i_dice]); dice_max = float(res["dice"][i_dice])
            ok = np.where(res["dice"] >= 0.95 * dice_max)[0]
            i_rec = int(ok[0]) if len(ok) else i_dice
            tau_rec = float(thr_arr[i_rec])

            fig = go.Figure()
            for key, color in [("dice", "#2F8F4E"), ("iou", "#179E8E"),
                               ("precision", "#E0982F"), ("recall", "#EB5757")]:
                fig.add_trace(go.Scatter(x=thr_arr, y=res[key], mode="lines",
                                         name=key.capitalize(), line=dict(color=color, width=2.5)))
            fig.add_vline(x=tau_dice, line_dash="dash", line_color="#2F8F4E",
                          annotation_text=f"Dice 최적 τ={tau_dice:.2f}", annotation_position="top")
            fig.add_vline(x=tau_rec, line_dash="dot", line_color="#EB5757",
                          annotation_text=f"민감도 우선 τ={tau_rec:.2f}", annotation_position="bottom")
            fig.update_layout(height=460, font=dict(size=14),
                              xaxis_title="임계값 τ (confidence)", yaxis_title="검증셋 평균 점수",
                              yaxis_range=[0, 1], legend=dict(orientation="h", y=1.08, x=0),
                              margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Dice 최적 τ", f"{tau_dice:.2f}", f"Dice {dice_max:.3f}")
            c1.caption(f"이 지점: Recall {res['recall'][i_dice]:.3f} · Precision {res['precision'][i_dice]:.3f}")
            c2.metric("민감도(recall) 우선 τ", f"{tau_rec:.2f}", f"Recall {res['recall'][i_rec]:.3f}")
            c2.caption(f"이 지점: Dice {res['dice'][i_rec]:.3f} · Precision {res['precision'][i_rec]:.3f}")

            st.info(f"검증셋 {res['n']}장(정답 병변 있는 이미지) 기준. "
                    "임상적으로 병변을 놓치지 않는 것이 중요하면 Dice 최적보다 조금 낮은 '민감도 우선 τ'를 채택합니다.")
            with st.expander("τ별 상세 수치 표 보기"):
                tbl = pd.DataFrame({"τ": thr_arr, "Dice": res["dice"], "IoU": res["iou"],
                                    "Precision": res["precision"], "Recall": res["recall"]})
                st.dataframe(tbl.style.format({"τ": "{:.2f}", "Dice": "{:.3f}", "IoU": "{:.3f}",
                                               "Precision": "{:.3f}", "Recall": "{:.3f}"}),
                             hide_index=True, use_container_width=True, height=320)

# ==================== 탭3 — 성능 평가 ====================
with tab_eval:
    st.subheader("성능 평가 — 두 모델 (테스트셋 전체 실시간 추론)")
    st.caption("병변분할(EfficientNet-B0+UNet++)과 중증도분류(PVTv2-B0+CORN)를 테스트셋 전체에 실행해 "
               "지표를 집계합니다. (최초 1회 전체 추론 후 캐시)")

    st.markdown("#### 📋 보고된 성능 (모델 카드)")
    model_cards()
    st.divider()

    if df is None:
        st.info("demo_assets/results.csv 가 없어 라이브 성능 평가를 할 수 없습니다. (위 카드는 학습 보고값)")
    else:
        st.markdown("#### 🔬 라이브 평가 (앱에서 실제 추론)")
        mode_label = st.radio(
            "중증도 평가 모드",
            ["end-to-end (예측 마스크 크롭 · 실제 배포 성능)",
             "단독 (GT 마스크 크롭 · 모델 카드 수치 재현)"],
            help="단독 모드는 정답 마스크로 크롭해 중증도 모델 실력만 측정합니다. "
                 "카드의 QWK 수치와 직접 비교하려면 이 모드를 쓰세요.")
        crop_mode = "gt" if mode_label.startswith("단독") else "pred"

        colt, colb = st.columns([2, 1])
        tau_e = colt.slider("세그 임계값 τ (평가 기준)", 0.05, 0.95, 0.50, 0.05, key="eval_tau")
        run_eval = colb.button("▶ 성능 평가 실행", type="primary")

        if not run_eval:
            st.info("위 버튼을 눌러 전체 테스트셋 평가를 실행하세요. (이미지 수에 따라 시간이 걸립니다)")
        else:
            seg_tag = f"{SEG_PATH}:{os.path.getmtime(SEG_PATH)}"
            sev_tag = f"{SEV_PATH}:{os.path.getmtime(SEV_PATH)}"
            ev = evaluate_all(tuple(sorted(df["filename"].tolist())), tau_e, crop_mode, seg_tag, sev_tag)

            # ----- 병변 분할 -----
            st.markdown("### 🟥 병변 분할 — EfficientNet-B0 + UNet++")
            s = ev["seg"]
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("평균 Dice", f"{s['dice']:.3f}")
            g2.metric("평균 IoU", f"{s['iou']:.3f}")
            g3.metric("Precision", f"{s['precision']:.3f}")
            g4.metric("Recall", f"{s['recall']:.3f}")
            st.caption(f"정답 병변 있는 이미지 {s['n']}장 기준 · τ={tau_e:.2f} · "
                       "Dice/IoU는 이미지 평균, Precision/Recall은 픽셀 micro 평균  ·  "
                       f"📋 카드: Dice {SEG_CARD['dice']:.4f} / IoU {SEG_CARD['iou']:.4f}")

            st.divider()
            # ----- 중증도 분류 -----
            _mode_txt = "단독 (GT 마스크 크롭)" if crop_mode == "gt" else "end-to-end (예측 마스크 크롭)"
            st.markdown(f"### 🟦 중증도 분류 — PVTv2-B0 + CORN  ·  모드: {_mode_txt}")
            if crop_mode == "gt":
                st.caption("👉 이 표의 **QWK(라이브) 열**을 QWK(카드) 열과 직접 비교하세요.")
            summ = [{"지표": METRIC_KO[m], "N": ev["sev"][m]["n"],
                     "정확도": ev["sev"][m]["acc"], "MAE": ev["sev"][m]["mae"],
                     "QWK(라이브)": ev["sev"][m]["qwk"], "QWK(카드)": SEV_CARD[m]}
                    for m in M.SEV_ORDER]
            st.dataframe(
                pd.DataFrame(summ).style.format(
                    {"정확도": "{:.3f}", "MAE": "{:.3f}", "QWK(라이브)": "{:.3f}", "QWK(카드)": "{:.3f}"}),
                hide_index=True, use_container_width=True)
            st.caption("정확도=정확히 맞춘 비율(↑좋음) · MAE=평균 등급오차(↓좋음) · "
                       "QWK=순서형 가중 카파(1에 가까울수록↑, 등급 순서까지 반영한 일치도)")

            st.markdown("##### 혼동행렬 (행=정답, 열=예측)")
            for i in range(0, len(M.SEV_ORDER), 2):
                cc = st.columns(2)
                for j, m in enumerate(M.SEV_ORDER[i:i + 2]):
                    classes = M.IGA_CLASSES if m == "iga" else M.SYMPTOM_CLASSES
                    with cc[j]:
                        st.plotly_chart(conf_heatmap(ev["sev"][m]["cm"], classes, METRIC_KO[m]),
                                        use_container_width=True)

# ==================== 탭4 — 이미지 추가 ====================
with tab_upload:
    st.subheader("새 이미지 실시간 추론")
    st.caption("피부 이미지를 올리면 병변분할 + 중증도 분류를 앱에서 직접 실행합니다. "
               "정답 라벨이 없어 IoU 등 정답기반 지표는 생략됩니다.")

    up_img = st.file_uploader("피부 이미지 (png/jpg)", type=["png", "jpg", "jpeg"], key="up_img")
    thr_u = st.slider("🎚️ 병변 판정 Confidence 임계값", 0.05, 0.95, 0.50, 0.05, key="up_thr")

    if up_img is None:
        st.info("이미지를 업로드하면 추론이 시작됩니다.")
    else:
        nimg = np.array(Image.open(up_img).convert("RGB"))
        out = M.run_full(nimg, get_seg(), get_sev(), tau=thr_u)
        disp = to512(nimg)

        cc1, cc2 = st.columns(2)
        cc1.image(disp, caption="원본", use_container_width=True)
        cc2.image(overlay(disp, out["mask"], PR_COLOR), caption=f"🔴 검출 영역 (τ={thr_u:.2f})",
                  use_container_width=True)
        st.metric("검출 병변 면적 비율", f"{out['area'] * 100:.2f}%")

        st.markdown("##### 중증도 · 증상 — 모델 예측")
        show_grades(out["grades"])
        with st.expander("🔎 중증도 모델에 입력된 크롭 이미지 보기"):
            st.image(out["crop"], caption="세그 마스크 bbox + 15% 패딩 크롭",
                     use_container_width=False, width=280)

# ==================== 진단 · 푸터 ====================
with st.expander("🩺 모델 출력 매핑 진단 (onnx2tf 순서 확인용)"):
    try:
        info = get_sev().mapping_info()
        st.write("중증도 멀티헤드 매핑 (지표 → tensor name → shape):")
        st.table(pd.DataFrame(info, columns=["지표", "output name", "shape"]))
        st.caption("지표별 shape가 IGA=[..,4], 증상=[..,3] 로 맞는지, 순서가 예상과 같은지 확인하세요. "
                   "어긋나면 name 파싱 규칙을 알려주시면 조정합니다.")
    except Exception as e:
        st.warning(f"매핑 정보를 불러오지 못했습니다: {e}")

st.divider()
st.markdown(
    "<div style='font-size:0.82rem;color:#8894A5;line-height:1.6;'>"
    "본 저작물(데모)은 과학기술정보통신부의 재원으로 한국지능정보사회진흥원(NIA)의 지원을 받아 구축된 "
    "「안면부 피부질환 이미지 합성데이터」(AI 허브, 71863)를 활용하여 제작되었습니다. "
    "데이터 출처: AI 허브(www.aihub.or.kr) · 원본 데이터 저작권은 AI 허브/구축기관에 있으며, 본 화면은 연구·시연용으로 무단 재배포를 금합니다."
    "</div>",
    unsafe_allow_html=True)
