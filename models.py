# -*- coding: utf-8 -*-
"""
TFLite 추론 모듈
- 병변 분할: EfficientNet-B0 + UNet++  (입력 NCHW [1,3,512,512] f32, 출력 raw logit [1,1,512,512])
- 중증도 분류: PVTv2-B0 + CORN (입력 2개: 크롭이미지 + 면적스칼라, 출력 5개 멀티헤드)

전처리: RGB → resize → /255 → ImageNet 정규화 → 모델이 요구하는 레이아웃(NCHW/NHWC)
후처리: 세그 sigmoid>τ / 중증도 CORN 누적곱 디코딩
"""
import re
import numpy as np
from PIL import Image

# ---- TFLite 런타임 (설치된 것 자동 선택) ----
Interpreter = None
try:
    from ai_edge_litert.interpreter import Interpreter          # 권장 (pip install ai-edge-litert)
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter       # 구 tflite-runtime
    except ImportError:
        try:
            from tensorflow.lite import Interpreter               # full tensorflow
        except ImportError:
            Interpreter = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# CORN 클래스 (등급 index 0..K-1 → 라벨)
IGA_CLASSES = ["Clear", "Almost Clear", "Mild", "Moderate", "Severe"]      # K=5 (logits 4)
SYMPTOM_CLASSES = ["None", "Mild", "Moderate", "Severe"]                    # K=4 (logits 3)
SEV_ORDER = ["iga", "erythema", "papulation", "excoriation", "lichenification"]  # forward :N 순서


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def corn_level(logits):
    """CORN 순서형 회귀 디코딩: sigmoid → 누적곱 > 0.5 개수 = 등급 index."""
    p = sigmoid(np.asarray(logits).ravel())
    return int((np.cumprod(p) > 0.5).sum())


def grade_label(metric, level):
    classes = IGA_CLASSES if metric == "iga" else SYMPTOM_CLASSES
    level = max(0, min(int(level), len(classes) - 1))
    return classes[level]


def preprocess(rgb_uint8, input_detail):
    """이미지를 모델 입력 텐서(NCHW 또는 NHWC)로 변환. ImageNet 정규화."""
    shape = list(input_detail["shape"])
    nchw = (len(shape) == 4 and shape[1] == 3)     # [1,3,H,W]
    size = shape[2] if nchw else shape[1]          # NCHW→H=shape[2], NHWC→H=shape[1]
    im = Image.fromarray(np.asarray(rgb_uint8)).convert("RGB").resize((int(size), int(size)), Image.BILINEAR)
    arr = np.asarray(im, np.float32) / 255.0       # HWC RGB
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    if nchw:
        arr = np.transpose(arr, (2, 0, 1))         # CHW
    return np.ascontiguousarray(arr[None], np.float32)


def lesion_bbox_crop(orig_rgb, mask, pad=0.15):
    """마스크 bbox + 15% 패딩으로 원본을 크롭. 병변 없으면 전체 이미지 폴백."""
    orig_rgb = np.asarray(orig_rgb)
    H0, W0 = orig_rgb.shape[:2]
    mh, mw = mask.shape[:2]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return orig_rgb
    sy, sx = H0 / mh, W0 / mw
    Y0 = ys.min() * sy; Y1 = (ys.max() + 1) * sy
    X0 = xs.min() * sx; X1 = (xs.max() + 1) * sx
    h, w = Y1 - Y0, X1 - X0
    Y0 = max(0, Y0 - pad * h); Y1 = min(H0, Y1 + pad * h)
    X0 = max(0, X0 - pad * w); X1 = min(W0, X1 + pad * w)
    crop = orig_rgb[int(Y0):int(np.ceil(Y1)), int(X0):int(np.ceil(X1))]
    return crop if crop.size else orig_rgb


def _forward_index(name):
    m = re.search(r":(\d+)", name or "")
    return int(m.group(1)) if m else None


class SegModel:
    def __init__(self, path):
        if Interpreter is None:
            raise RuntimeError("TFLite 런타임이 없습니다. requirements.txt에 'ai-edge-litert'를 추가하세요.")
        self.it = Interpreter(model_path=path)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def predict_logit(self, rgb_uint8):
        x = preprocess(rgb_uint8, self.inp)
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        y = np.squeeze(np.asarray(self.it.get_tensor(self.out["index"])))  # [512,512] logit
        return y.astype(np.float32)


class SevModel:
    def __init__(self, path):
        if Interpreter is None:
            raise RuntimeError("TFLite 런타임이 없습니다. requirements.txt에 'ai-edge-litert'를 추가하세요.")
        self.it = Interpreter(model_path=path)
        self.it.allocate_tensors()
        ins = self.it.get_input_details()
        self.img_in = next((d for d in ins if len(d["shape"]) == 4), None)          # 이미지 = 4D
        self.area_in = next((d for d in ins if int(np.prod(d["shape"])) == 1), None)  # 면적 = 크기1
        self.out_map = self._map_outputs(self.it.get_output_details())

    def _map_outputs(self, outs):
        """onnx2tf가 물리적 순서를 뒤섞을 수 있어 name 기준으로 매핑."""
        # 1) name에 지표명 포함
        m = {}
        for d in outs:
            nm = (d["name"] or "").lower()
            for k in SEV_ORDER:
                if k in nm and k not in m:
                    m[k] = d
        if len(m) == 5:
            return m
        # 2) name의 :N (forward 인덱스)
        m = {}
        for d in outs:
            n = _forward_index(d["name"])
            if n is not None and 0 <= n < 5:
                m[SEV_ORDER[n]] = d
        if len(m) == 5:
            return m
        # 3) 폴백: shape[-1]==4 를 iga, 나머지는 물리적 순서로 증상 매핑
        m = {}
        si = 0
        for d in outs:
            if int(d["shape"][-1]) == 4 and "iga" not in m:
                m["iga"] = d
            elif si < 4:
                m[SEV_ORDER[1 + si]] = d
                si += 1
        return m

    def mapping_info(self):
        return [(k, self.out_map[k]["name"], list(self.out_map[k]["shape"]))
                for k in SEV_ORDER if k in self.out_map]

    def predict(self, crop_rgb_uint8, area_frac):
        x = preprocess(crop_rgb_uint8, self.img_in)
        self.it.set_tensor(self.img_in["index"], x)
        a = np.array([float(area_frac)], np.float32).reshape(self.area_in["shape"])
        self.it.set_tensor(self.area_in["index"], a)
        self.it.invoke()
        grades = {}
        for metric, d in self.out_map.items():
            logits = np.asarray(self.it.get_tensor(d["index"])).ravel()
            grades[metric] = corn_level(logits)
        return grades


def run_full(orig_rgb, seg_model, sev_model, tau=0.5):
    """전체 파이프라인: 세그 → 마스크/면적 → 크롭 → 중증도(CORN) → 등급."""
    logit = seg_model.predict_logit(orig_rgb)          # 512x512
    prob = sigmoid(logit).astype(np.float32)
    mask = prob >= tau
    area = float(mask.sum()) / mask.size               # 병변면적비 0~1
    crop = lesion_bbox_crop(orig_rgb, mask, pad=0.15)
    grades = sev_model.predict(crop, area)
    return {"logit": logit, "prob": prob, "mask": mask, "area": area,
            "crop": crop, "grades": grades}
