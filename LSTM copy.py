import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score

# 데이터 로드
file_path = "Total.csv"
data = pd.read_csv(file_path, encoding='CP949')

# X는 특성, y는 레이블
X = data[['temp1', 'temp2', 'temp3', 'hum1', 'hum2', 'hum3', 'disease']]
y = data['risk']

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 연속형 데이터 스케일링
scaler = MinMaxScaler()
X_train[['temp1', 'temp2', 'temp3', 'hum1', 'hum2', 'hum3']] = scaler.fit_transform(X_train[['temp1', 'temp2', 'temp3', 'hum1', 'hum2', 'hum3']])
X_val[['temp1', 'temp2', 'temp3', 'hum1', 'hum2', 'hum3']] = scaler.transform(X_val[['temp1', 'temp2', 'temp3', 'hum1', 'hum2', 'hum3']])

# 범주형 데이터 원-핫 인코딩
X_train = pd.get_dummies(X_train, columns=['disease'])
X_val = pd.get_dummies(X_val, columns=['disease'])

# 데이터 형태 변환
X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val.values, (X_val.shape[0], 1, X_val.shape[1]))

# y 레이블을 원-핫 인코딩 형태로 변환
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_val_encoded = encoder.transform(y_val)

y_train_onehot = to_categorical(y_train_encoded)
y_val_onehot = to_categorical(y_val_encoded)


print(X_train.shape)
print(y_train_onehot.shape)


# # 조기 종료 설정
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# # LSTM 모델 구성
# model = Sequential()
# model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dense(y_train_onehot.shape[1], activation='softmax'))

# # 모델 컴파일
# optimizer = Adam(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# # 모델 학습 (조기 종료 콜백 추가)
# history = model.fit(X_train, y_train_onehot, epochs=500, batch_size=64, validation_data=(X_val, y_val_onehot), verbose=1, callbacks=[early_stopping])

# # 가중치 저장
# model.save('lstm_model_weights.h5')



# # 모델 요약
# model.summary()

# # 예측 수행
# y_pred = model.predict(X_val)
# y_pred_classes = np.argmax(y_pred, axis=1)

# # 정밀도, 재현율, F1 점수 계산
# precision = precision_score(y_val_encoded, y_pred_classes, average='weighted')
# recall = recall_score(y_val_encoded, y_pred_classes, average='weighted')
# f1 = f1_score(y_val_encoded, y_pred_classes, average='weighted')

# print(f'정밀도 (Precision): {precision:.2f}')
# print(f'재현율 (Recall): {recall:.2f}')
# print(f'F1 점수 (F1 Score): {f1:.2f}')

# # 시각화: 정밀도, 재현율, F1 점수 바 차트
# labels = ['Precision', 'Recall', 'F1 Score']
# values = [precision, recall, f1]

# plt.figure(figsize=(6, 4))
# plt.bar(labels, values, color=['blue', 'green', 'orange'])
# plt.xlabel('Evaluation')
# plt.ylabel('value')
# plt.title('Performance')
# plt.ylim(0, 1.0)  # y 축 범위 설정 (0부터 1까지)
# plt.show()

# # 훈련 손실과 검증 손실 그래프 출력
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 그래프
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss Value')
# plt.legend()
# plt.ylim(0, 1.0)  # y 축 범위 설정 (0부터 1까지)

# # 훈련 정확도와 검증 정확도 그래프 출력
# plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 그래프
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy Value')
# plt.legend()
# plt.ylim(0, 1.0)  # y 축 범위 설정 (0부터 1까지)

# plt.tight_layout()  # 그래프 간격 조정
# plt.show()