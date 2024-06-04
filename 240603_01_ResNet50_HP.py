import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# 데이터셋 경로 설정
dataset = 'resized_image'
train_dir = os.path.join(dataset, 'train')
validation_dir = os.path.join(dataset, 'val')
test_dir = os.path.join(dataset, 'test')

IMG_SIZE = (224, 224)  # ResNet50의 기본 입력 크기
BATCH_SIZE = 16

# 이미지 데이터셋 로드
train_dataset = image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='categorical')
validation_dataset = image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='categorical')
test_dataset = image_dataset_from_directory(test_dir, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='categorical')

# 데이터 증강 및 정규화 레이어 추가
data_augmentation = Sequential([
    Rescaling(1./255),  # 정규화 먼저
    RandomFlip("horizontal"),
    RandomRotation(0.2),
])

# 클래스 레이블 수집
class_labels = []
for images, labels in train_dataset:
    class_labels.extend(tf.argmax(labels, axis=1).numpy().tolist())

# 중복 제거를 통한 클래스 수 확인
num_classes = len(set(class_labels))

# 모델 생성 함수
def create_model(optimizer='adam'):
    try:
        base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_model.trainable = True  # 모델의 모든 층을 훈련 가능하게 설정
        model = Sequential([
            data_augmentation,  # 데이터 증강 및 정규화 레이어 추가
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model created and compiled successfully")
        return model
    except Exception as e:
        print(f"Error in create_model: {e}")
        return None

# KerasClassifier 래핑
keras_model = KerasClassifier(build_fn=create_model, verbose=0)

# 하이퍼파라미터 그리드 정의
param_grid = {
    'epochs': [5, 10],
    'batch_size': [16, 32],
    'optimizer': ['adam', 'rmsprop']
}

# 그리드 탐색 객체 생성
grid_search = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=3, error_score='raise')

# 데이터를 넘파이 배열로 변환
def dataset_to_numpy(dataset):
    images, labels = [], []
    for batch in dataset:
        images.append(batch[0].numpy())
        labels.append(batch[1].numpy())
    return np.concatenate(images), np.concatenate(labels)

train_images, train_labels = dataset_to_numpy(train_dataset)

# 원-핫 인코딩된 레이블을 정수 레이블로 변환
train_labels = np.argmax(train_labels, axis=1)

# 그리드 탐색 수행
try:
    grid_result = grid_search.fit(train_images, train_labels)
    # 최적의 하이퍼파라미터 출력
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
except Exception as e:
    print(f"Error during grid search: {e}")
