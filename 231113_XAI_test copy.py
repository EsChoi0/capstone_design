import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.models import load_model
import os
from keras.models import Sequential
import tensorflow as tf

# eager execution 비활성화
tf.compat.v1.disable_eager_execution()


def grad_cam(model, img, layer_name, class_index, height, width):
    # 이미지 전처리
    img = image.load_img(img, target_size=(height, width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # 모델 예측
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])

    # Grad-CAM 설정
    last_conv_layer = model.get_layer(layer_name)

    # 예측 클래스의 출력을 가져옴
    output = model.output[:, pred_class]

    # GradientTape을 사용하여 그래디언트 계산
    with tf.GradientTape() as tape:
        last_conv_output = last_conv_layer.output
        grads = tape.gradient(output, last_conv_output)
    
    # None 값을 방지하기 위해 추가
    if grads is not None:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    else:
        # 차원이 없다면 그대로 사용
        pooled_grads = grads

    # 특징맵과 그래디언트 계산을 위해 새로운 모델 생성
    iterate = tf.keras.models.Model(inputs=model.input, outputs=[pooled_grads, last_conv_output])
    
    # 특징맵과 그래디언트 계산
    pooled_grads_value, conv_layer_output_value = iterate([img_array])
    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # 특징맵의 평균값 계산
    grad_cam = np.mean(conv_layer_output_value, axis=-1)

    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = cv2.resize(grad_cam, (height, width))
    grad_cam = grad_cam / grad_cam.max()

    # 원본 이미지 로드
    img = cv2.imread(img)

    # Heatmap 생성
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)

    # Heatmap을 원본 이미지에 적용
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return superimposed_img, pred_class, preds[0][class_index]



def occlusion(model, img_path, class_index, patch_size, stride):
    # 이미지 로드
    img = cv2.imread(img_path)

    # 이미지 전처리
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # 원본 이미지의 높이와 너비
    height, width, _ = img.shape

    # 예측 클래스의 출력을 가져옴
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    pred_prob = preds[0][class_index]

    # 초기화된 이미지로 복사
    occlusion_img = img.copy()

    # Occlusion 적용
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            img_patch = img_array.copy()
            img_patch[:, y:y + patch_size, x:x + patch_size, :] = 0

            # 모델에 패치를 적용하여 예측
            patch_preds = model.predict(img_patch)
            patch_pred_class = np.argmax(patch_preds[0])
            patch_pred_prob = patch_preds[0][class_index]

            # 예측 클래스가 변경된 경우에만 이미지 업데이트
            if patch_pred_class != pred_class:
                color = (0, 0, 255)  # 빨간색
                occlusion_img[y:y + patch_size, x:x + patch_size, :] = color

    return occlusion_img, pred_class, pred_prob

# 모델 로드
model = load_model('EfficientNetB0_finetune.hdf5')

model.summary()

# 이미지 경로
img_dir = r'C:\Users\Esc\Desktop\sum'
# 결과 저장 경로
result_dir = r'C:\Users\Esc\Desktop\XAI'
os.makedirs(result_dir, exist_ok=True)


for img_file in os.listdir(img_dir):
    # 각 이미지 파일의 경로 생성
    img_path = os.path.join(img_dir, img_file)

    # Grad-CAM과 Occlusion을 계산하고 결과를 출력하는 함수 호출
    # grad_cam_img, pred_class, pred_prob = grad_cam(model, img_path, 'efficientnetb0', class_index=1, height=224, width=224)
    occlusion_img, pred_class_occ, pred_prob_occ = occlusion(model, img_path, class_index=1, patch_size=32, stride=16)

    # # 결과 출력
    # plt.figure(figsize=(12, 6))

    # plt.subplot(1, 3, 1)
    # plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    # plt.title('Original Image')

    # plt.subplot(1, 3, 2)
    # plt.imshow(cv2.cvtColor(grad_cam_img, cv2.COLOR_BGR2RGB))
    # plt.title(f'Grad-CAM\nPredicted Class: {pred_class}, Probability: {pred_prob:.4f}')

    # plt.subplot(1, 3, 3)
    # plt.imshow(cv2.cvtColor(occlusion_img, cv2.COLOR_BGR2RGB))
    # plt.title(f'Occlusion\nPredicted Class: {pred_class_occ}, Probability: {pred_prob_occ:.4f}')

    # plt.show()

    # 결과 이미지를 저장할 파일 경로 생성
    # grad_cam_result_path = os.path.join(result_dir, f'grad_cam_{img_file}')
    occlusion_result_path = os.path.join(result_dir, f'occlusion_{img_file}')

    # 결과 이미지 저장
    # cv2.imwrite(grad_cam_result_path, grad_cam_img)
    cv2.imwrite(occlusion_result_path, occlusion_img)








