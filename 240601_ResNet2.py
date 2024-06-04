import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid

# 데이터셋 경로 설정
dataset = 'resized_image'
train_dir = os.path.join(dataset, 'train')
validation_dir = os.path.join(dataset, 'val')
test_dir = os.path.join(dataset, 'test')

IMG_SIZE = (224, 224)  # ResNet의 표준 입력 크기로 수정

# 이미지 데이터셋 로드 및 전처리
train_dataset = image_dataset_from_directory(train_dir, shuffle=True, batch_size=32, image_size=IMG_SIZE, label_mode='categorical')
validation_dataset = image_dataset_from_directory(validation_dir, shuffle=True, batch_size=32, image_size=IMG_SIZE, label_mode='categorical')

# 데이터를 넘파이 배열로 변환 (메모리 사용 주의)
def dataset_to_numpy(dataset):
    images, labels = [], []
    for batch in dataset:
        images.append(batch[0].numpy())
        labels.append(batch[1].numpy())
    return np.concatenate(images), np.concatenate(labels)

train_images, train_labels = dataset_to_numpy(train_dataset)
validation_images, validation_labels = dataset_to_numpy(validation_dataset)

# 클래스 수 확인
num_classes = train_labels.shape[1]

# ResNet50 모델 불러오기 (사전 훈련된 가중치 사용하지 않음)
base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights=None)
base_model.trainable = True  # 모델의 모든 층을 훈련 가능하게 설정

# 새로운 분류 레이어 추가
def create_model(optimizer='adam'):
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')  # 클래스 수에 맞게 수정
    ])
    # 모델 컴파일
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 하이퍼파라미터 그리드 정의
param_grid = {
    'epochs': [5, 10, 15],
    'batch_size': [16, 32, 64],
    'optimizer': ['adam', 'rmsprop']
}

# 수동 그리드 탐색
best_score = 0
best_params = None

for params in ParameterGrid(param_grid):
    model = create_model(optimizer=params['optimizer'])
    model.fit(train_images, train_labels, epochs=params['epochs'], batch_size=params['batch_size'], validation_data=(validation_images, validation_labels))
    predictions = model.predict(validation_images)
    accuracy = accuracy_score(np.argmax(validation_labels, axis=1), np.argmax(predictions, axis=1))
    
    if accuracy > best_score:
        best_score = accuracy
        best_params = params

print("Best: %f using %s" % (best_score, best_params))
