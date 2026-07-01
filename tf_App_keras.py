"""
tf_App_keras.py  ─  .keras 모델 사용
변경 사항:
  - build_model_architecture() 완전 제거
  - load_model() 에서 keras.models.load_model() 한 줄로 구조+가중치 동시 로드
  - cam_model: Sequential 구조 대응 (vgg16 서브모델 경유)
  - 실시간 웹캠 제거 (streamlit-webrtc 패키지 의존성 문제)
  - 입력: 파일 업로드 / 카메라 스냅샷 (st.camera_input 기본 내장)
"""

import streamlit as st
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ── 한글 폰트 설정 ──
font_path_win   = "C:/Windows/Fonts/malgun.ttf"
font_path_linux = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(font_path_win):
    font_manager.fontManager.addfont(font_path_win)
    matplotlib.rc('font', family='Malgun Gothic')
elif os.path.exists(font_path_linux):
    font_manager.fontManager.addfont(font_path_linux)
    matplotlib.rc('font', family='NanumGothic')
else:
    matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 상수 ──
INPUT_IMG_SIZE = (224, 224)
NEG_CLASS      = 1
CLASSES        = ["정상", "불량"]
MODEL_PATH     = "./weights/leather_model.keras"   # ← .keras 사용
HEATMAP_THRES  = 0.5

# ─────────────────────────────────────────────
# 1. 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:", layout="wide")
st.title("InspectorsAlly")
st.caption("AI 기반 자동 검사로 품질 관리를 한 단계 높이세요")
st.write("제품 이미지를 업로드하면 AI 모델이 **정상 / 불량** 여부를 자동으로 판별합니다.")

with st.sidebar:
    if os.path.exists("./docs/overview_dataset.jpg"):
        st.image(Image.open("./docs/overview_dataset.jpg"))
    st.subheader("InspectorsAlly 소개")
    st.write(
        "InspectorsAlly는 기업의 품질 관리 검사를 효율화하기 위해 설계된 "
        "AI 기반 검사 애플리케이션입니다. VGG16 전이학습 기반으로 "
        "가죽 제품의 스크래치, 찍힘, 변색 등의 결함을 감지합니다."
    )
    st.divider()
    st.write("**모델 정보**")
    st.write(f"- 프레임워크: TensorFlow {tf.__version__}")
    st.write(f"- 백본: VGG16 (ImageNet 사전학습, 전체 동결)")
    st.write(f"- 출력: sigmoid 단일값 (0=정상, 1=불량)")
    st.write(f"- 입력 크기: {INPUT_IMG_SIZE[0]}×{INPUT_IMG_SIZE[1]}")


# ─────────────────────────────────────────────
# 2. 모델 로드  ← build_model_architecture() 제거됨
#    .keras 는 구조+가중치가 함께 저장되어 있으므로
#    load_model() 한 줄로 복원
#
#    ※ Sequential(vgg16 포함) 구조이므로
#      model.input 대신 vgg16 서브모델을 통해 cam_model 재구성
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None

    # 구조 + 가중치 한 번에 로드
    model = tf.keras.models.load_model(MODEL_PATH)

    # CAM 히트맵용 cam_model 재구성
    # Sequential 모델은 model.input 직접 접근 불가 → vgg16 서브모델 경유
    vgg16       = model.get_layer("vgg16")
    inputs      = vgg16.input                                   # (None, 224, 224, 3)
    feature_out = vgg16.get_layer("block5_conv3").output        # (None, 14, 14, 512)

    x = vgg16.output                                            # (None, 7, 7, 512)
    x = model.get_layer("global_average_pooling2d")(x)
    x = model.get_layer("dense")(x)
    x = model.get_layer("dropout")(x)
    predictions = model.get_layer("predictions")(x)             # (None, 1)

    cam_model = keras.Model(inputs=inputs, outputs=[feature_out, predictions])
    return model, cam_model


# ─────────────────────────────────────────────
# 3. 이미지 전처리
# ─────────────────────────────────────────────
def preprocess_image(pil_img):
    img       = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = keras.applications.vgg16.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


# ─────────────────────────────────────────────
# 4. CAM 히트맵 생성
# ─────────────────────────────────────────────
def generate_heatmap(cam_model, img_array):
    feature_maps, pred = cam_model(img_array, training=False)
    feature_maps = feature_maps.numpy()[0]
    prob         = float(pred.numpy()[0][0])
    class_idx    = 1 if prob > 0.5 else 0

    w1 = cam_model.get_layer("dense").get_weights()[0]
    w2 = cam_model.get_layer("predictions").get_weights()[0]
    weights_for_anomaly = (w1 @ w2).squeeze()

    cam     = np.dot(feature_maps, weights_for_anomaly)
    cam_min, cam_max = cam.min(), cam.max()
    norm_cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    heatmap_pil     = Image.fromarray((norm_cam * 255).astype(np.uint8))
    heatmap_resized = np.array(heatmap_pil.resize(INPUT_IMG_SIZE)) / 255.0
    return heatmap_resized, prob, class_idx


def get_bbox_from_heatmap(heatmap, thres=0.5):
    binary_map = heatmap > thres
    if not binary_map.any():
        return None
    x_dim  = np.max(binary_map, axis=0) * np.arange(binary_map.shape[1])
    y_dim  = np.max(binary_map, axis=1) * np.arange(binary_map.shape[0])
    x_vals = x_dim[x_dim > 0]
    y_vals = y_dim[y_dim > 0]
    if len(x_vals) == 0 or len(y_vals) == 0:
        return None
    return int(x_vals.min()), int(y_vals.min()), int(x_dim.max()), int(y_dim.max())


# ─────────────────────────────────────────────
# 5. 결과 시각화
# ─────────────────────────────────────────────
def visualize_result(pil_img, heatmap, class_idx, prob, thres=HEATMAP_THRES):
    img_np = np.array(pil_img.resize(INPUT_IMG_SIZE).convert("RGB"))

    if class_idx == NEG_CLASS:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        axes[0].imshow(img_np)
        axes[0].set_title("원본 이미지", fontsize=11)
        axes[0].axis("off")
        axes[1].imshow(img_np)
        axes[1].imshow(heatmap, cmap="Reds", alpha=0.45)
        axes[1].set_title(f"불량 감지 히트맵 (불량 확률: {prob:.3f})", fontsize=11)
        axes[1].axis("off")
        bbox = get_bbox_from_heatmap(heatmap, thres)
        if bbox:
            x0, y0, x1, y1 = bbox
            rect = mpatches.Rectangle(
                (x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor="red", facecolor="none"
            )
            axes[1].add_patch(rect)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(img_np)
        ax.set_title(f"정상 (불량 확률: {prob:.3f})", fontsize=11)
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)


# ─────────────────────────────────────────────
# 6. 메인 UI
# ─────────────────────────────────────────────
model, cam_model = load_model()

if model is None:
    st.error(
        f"모델 파일을 찾을 수 없습니다: `{MODEL_PATH}`\n\n"
        "노트북에서 아래 코드로 저장하세요:\n\n"
        "```python\nmodel.save('weights/leather_model.keras')\n```"
    )
    st.stop()

st.subheader("이미지 입력 방법 선택")
input_method = st.radio("options", ["파일 업로드", "카메라 촬영"],
                        label_visibility="collapsed")
pil_image = None

if input_method == "파일 업로드":
    uploaded_file = st.file_uploader("이미지 파일을 선택하세요", type=["jpg","jpeg","png"])
    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        st.image(pil_image, caption="업로드된 이미지", width=300)
        st.success("이미지가 성공적으로 업로드되었습니다!")
    else:
        st.warning("검사할 이미지 파일을 업로드해주세요.")

elif input_method == "카메라 촬영":
    st.warning("카메라 접근 권한을 허용해주세요.")
    camera_file = st.camera_input("카메라로 이미지 촬영")
    if camera_file:
        pil_image = Image.open(camera_file).convert("RGB")
        st.image(pil_image, caption="촬영된 이미지", width=300)
        st.success("이미지가 성공적으로 촬영되었습니다!")
    else:
        st.warning("카메라로 이미지를 촬영해주세요.")

submit = st.button(label="가죽 제품 이미지 검사 시작", type="primary")

if submit:
    if pil_image is None:
        st.error("이미지를 먼저 업로드하거나 카메라로 촬영해주세요.")
    else:
        st.subheader("검사 결과")
        with st.spinner("이미지를 분석 중입니다..."):
            img_array                = preprocess_image(pil_image)
            heatmap, prob, class_idx = generate_heatmap(cam_model, img_array)
        label = CLASSES[class_idx]
        if label == "정상":
            st.success(f"✅ **정상** (불량 확률: {prob:.1%})\n\n제품 검사 결과 이상이 감지되지 않았습니다.")
        else:
            st.error(
                f"⚠️ **불량 감지** (불량 확률: {prob:.1%})\n\n"
                "아래 히트맵에서 결함이 의심되는 영역(빨간 박스)을 확인하세요."
            )
        st.write("**검사 결과 시각화**")
        visualize_result(pil_image, heatmap, class_idx, prob)
        st.write("**클래스별 예측 확률**")
        col1, col2 = st.columns(2)
        col1.metric("정상", f"{(1 - prob):.1%}")
        col2.metric("불량", f"{prob:.1%}")
        st.progress(float(prob), text=f"불량 확률: {prob:.1%}")