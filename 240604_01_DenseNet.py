import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

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

def DenseNet121(input_shape=(224, 224, 3), num_classes=4):
    growth_rate = 32
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

class_names = train_dataset.class_names

# 데이터 증강 및 전처리 레이어
data_augmentation = tf.keras.Sequential([
    Rescaling(1./255),
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
])

# 데이터 증강 및 전처리 적용
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
validation_dataset = validation_dataset.map(lambda x, y: (Rescaling(1./255)(x), y), num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.map(lambda x, y: (Rescaling(1./255)(x), y), num_parallel_calls=AUTOTUNE)

# 데이터셋 캐싱 및 배치 사전 로드
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# 모델 정의 및 컴파일 (DenseNet-121 예제 사용)
model = DenseNet121(input_shape=(224, 224, 3), num_classes=4)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

# 모델 평가
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}')

# 모델 예측
y_true = []
y_pred = []

for images, labels in test_dataset:
    preds = model.predict(images)
    y_true.extend(tf.argmax(labels, axis=1).numpy())
    y_pred.extend(tf.argmax(preds, axis=1).numpy())

# 분류 보고서 출력 
print(classification_report(y_true, y_pred, target_names=class_names))


