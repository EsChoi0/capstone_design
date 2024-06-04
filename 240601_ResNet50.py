import tensorflow as tf
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# 데이터셋 경로 설정
dataset = 'resized_image'
train_dir = os.path.join(dataset, 'train')
validation_dir = os.path.join(dataset, 'val')
test_dir = os.path.join(dataset, 'test')


IMG_SIZE = (244, 244)
BATCH_SIZE = 16
train_dataset = image_dataset_from_directory(train_dir, shuffle = True, batch_size = BATCH_SIZE, image_size = IMG_SIZE, label_mode = 'categorical')
validation_dataset = image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='categorical')
test_dataset = image_dataset_from_directory(test_dir, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE, label_mode='categorical')

# 클래스 레이블 수집
class_labels = []
for images, labels in train_dataset:
    class_labels.extend(tf.argmax(labels, axis=1).numpy().tolist())

# 중복 제거를 통한 클래스 수 확인
num_classes = len(set(class_labels))

# ResNet50 모델 불러오기
base_model = ResNet50(input_shape=(244, 244, 3), include_top=False, weights=None)
base_model.trainable = True  # 모델의 모든 층을 훈련 가능하게 설정

# 새로운 분류 레이어 추가
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


## 하이퍼파라미터 그리드 정의
#param_grid = {
#    'epochs': [5, 10, 15],
#    'batch_size': [16, 32, 64],
#    'optimizer': ['adam', 'rmsprop']
#}

## 그리드 탐색 객체 생성
#grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')

## 그리드 탐색 수행
#grid_result = grid_search.fit(train_dataset, validation_data=validation_dataset)

## 최적의 하이퍼파라미터 출력
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))





# 모델 훈련
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=15)
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
print(classification_report(y_true, y_pred, target_names=train_dataset.class_names))

 # # Summary of the model
 # resnet_model.summary()