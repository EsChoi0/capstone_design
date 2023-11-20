import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

# 데이터 로드
file_path = 'C:\\Users\\Esc\\Desktop\\황순길한테보내줄거\\특성6\\Total.csv'
data = pd.read_csv(file_path)

# # 입력 데이터 선택 (온도, 습도 및 'disease' 열 포함)
# temperature_features = ['temp{}'.format(i) for i in range(1, 4)]
# humidity_features = ['hum{}'.format(i) for i in range(1, 4)]
# X = data[temperature_features + humidity_features + ['disease']]

# X는 특성, y는 레이블
X = data[['temp1', 'temp2', 'temp3', 'hum1', 'hum2', 'hum3', 'disease']]
y = data['risk']

# 연속형 데이터 스케일링
scaler = MinMaxScaler()
X[['temp1', 'temp2', 'temp3', 'hum1', 'hum2', 'hum3']] = scaler.fit_transform(X[['temp1', 'temp2', 'temp3', 'hum1', 'hum2', 'hum3']])


# 범주형 데이터 원-핫 인코딩
X = pd.get_dummies(X, columns=['disease'])

# 테스트 데이터의 레이블을 원-핫 인코딩으로 변환
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(data['risk'])
y_onehot = to_categorical(y_encoded, num_classes=6)

# 데이터 형태 변환
X = np.reshape(X.values, (X.shape[0], 1, X.shape[1]))





# 훈련된 모델 로드 및 평가
model = load_model("lstm_model_weights.h5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
loss, accuracy = model.evaluate(X, y_onehot)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# 테스트 데이터로 예측 수행
predictions = model.predict(X)
predicted_labels = np.argmax(predictions, axis=1)

# 예측 결과 출력
print("Predicted Labels:")
print(predicted_labels)
