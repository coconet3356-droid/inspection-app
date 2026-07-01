import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import streamlit as range  # Streamlit 라이브러리
import streamlit as st

# ── 설정 ─────────────────────────────────────────────────────────
MODEL_PATH = "./weights/leather_model.keras"  # .keras 모델 경로
INPUT_IMG_SIZE = (224, 224)
CLASSES = ["정상", "불량"]

# 1. 페이지 설정
st.set_page_config(
    page_title="가죽 이상 탐지 시스템",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 가죽 이상 탐지 시스템")
st.caption("가죽 이미지를 업로드하거나 카메라로 촬영하여 정상/불량 여부를 실시간으로 판별합니다.")


# ─────────────────────────────────────────────────────────────────
# 2. 모델 로드 (캐싱 적용 및 예외 처리)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"🚨 모델 파일을 찾을 수 없습니다. 경로를 확인해주세요: `{MODEL_PATH}`")
        st.stop()  # 앱 실행 중단
    
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# ─────────────────────────────────────────────────────────────────
# 3. 이미지 전처리 로직 (기존 유지)
# ─────────────────────────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.vgg16.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────────────────────────────
# 4. 추론 로직 (기존 유지)
# ─────────────────────────────────────────────────────────────────
def predict(model, pil_img):
    arr = preprocess(pil_img)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = CLASSES[1 if prob > 0.5 else 0]
    return label, prob


# ─────────────────────────────────────────────────────────────────
# 메인 웹 UI 및 제어 흐름
# ─────────────────────────────────────────────────────────────────
def main():
    # 모델 로드
    model = load_model()

    # 3. 이미지 입력 방식 선택
    input_mode = st.radio(
        "이미지 입력 방식을 선택하세요", 
        ["파일 업로드", "카메라 촬영"],
        horizontal=True
    )

    pil_img = None

    if input_mode == "파일 업로드":
        uploaded_file = st.file_uploader("가죽 이미지 파일을 업로드하세요", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)

    else:
        camera_file = st.camera_input("웹캠을 통해 가죽을 촬영하세요")
        if camera_file is not None:
            pil_img = Image.open(camera_file)

    # 이미지 미리보기 표시
    if pil_img is not None:
        st.image(pil_img, caption="입력된 이미지 예시", use_container_width=True)

    st.markdown("---")

    # 4. 검사 실행 버튼
    if st.button("검사 시작", type="primary"):
        if pil_img is None:
            st.warning("⚠️ 검사할 이미지를 먼저 업로드하거나 촬영해 주세요.")
        else:
            with st.spinner("이미지를 분석하고 있습니다..."):
                # 추론 수행
                label, prob = predict(model, pil_img)
                
                normal_prob = 1 - prob
                defect_prob = prob

            # 5. 결과 표시
            st.subheader("📊 분석 결과")
            
            if label == "정상":
                st.success(f"✅ 판정 결과: **{label}** 제품입니다.")
            else:
                st.error(f"🚨 판정 결과: **{label}**품입니다. 주의가 필요합니다.")

            # 확률 표기 (st.metric 이용)
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="정상 확률", value=f"{normal_prob:.1%}")
            with col2:
                st.metric(label="불량 확률", value=f"{defect_prob:.1%}")

            # 불량 확률 시각화 (st.progress 이용)
            st.write("**불량 위험도 수치**")
            st.progress(defect_prob)


if __name__ == "__main__":
    main()