import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation

def dense_block(x, num_layers, growth_rate):
    for i in range(num_layers):
        cb = composite_function(x, growth_rate)
        x = layers.Concatenate()([x, cb])
    return x

def composite_function(x, growth_rate):
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(4 * growth_rate, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(growth_rate, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    return x

def transition_layer(x, reduction):
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(int(x.shape[-1] * reduction), kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = layers.AvgPool2D(pool_size=2, strides=2, padding='same')(x)
    return x

def DenseNet121(input_shape=(224, 224, 3), num_classes=4, growth_rate=32):
    num_blocks = [6, 12, 24, 16]

    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    for i, num_layers in enumerate(num_blocks):
        x = dense_block(x, num_layers, growth_rate)
        if i != len(num_blocks) - 1:
            x = transition_layer(x, 0.5)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def create_model(num_classes=4, optimizer='adam'):
    model = DenseNet121(input_shape=(224, 224, 3), num_classes=num_classes)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# 데이터셋 경로 설정
dataset = 'resized_image'
train_dir = os.path.join(dataset, 'train')
validation_dir = os.path.join(dataset, 'val')
test_dir = os.path.join(dataset, 'test')

# 이미지 데이터셋 로드
train_dataset = image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='categorical')
validation_dataset = image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='categorical')
test_dataset = image_dataset_from_directory(test_dir, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='categorical')

# 데이터 증강 및 전처리 레이어
def preprocess_data(x, y):
    x = Rescaling(1./255)(x)
    x = RandomFlip("horizontal_and_vertical")(x)
    x = RandomRotation(0.2)(x)
    return x, y

# 데이터 증강 및 전처리 적용
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=AUTOTUNE)
validation_dataset = validation_dataset.map(lambda x, y: (Rescaling(1./255)(x), y), num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.map(lambda x, y: (Rescaling(1./255)(x), y), num_parallel_calls=AUTOTUNE)

# 데이터셋 캐싱 및 배치 사전 로드
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# KerasClassifier 래핑
keras_model = KerasClassifier(model=create_model, epochs=10, batch_size=16, verbose=0)

# 하이퍼파라미터 그리드 정의
param_grid = {
    'epochs': [5, 10, 15],
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
test_images, test_labels = dataset_to_numpy(test_dataset)

# 그리드 탐색 수행
try:
    grid_result = grid_search.fit(train_images, train_labels)
    
    # 최적의 하이퍼파라미터로 모델 재학습
    best_model = grid_result.best_estimator_
    best_model.fit(train_images, train_labels)
    
    # 테스트 데이터셋으로 모델 평가
    test_accuracy = best_model.score(test_images, test_labels)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # 최적의 모델 저장
    best_model.model_.save("best_model.h5")
    
    # 최적의 하이퍼파라미터 출력
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
except Exception as e:
    print(f"Error during grid search: {e}")
