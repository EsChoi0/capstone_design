import pandas as pd

# CSV 파일을 읽어옴
file_path = r'C:\Users\Esc\Desktop\231114\Test\updated_normal.csv'  # 파일 경로를 적절히 수정해주세요
df = pd.read_csv(file_path)

# 수치형 열에 대해서만 결측치를 해당 열의 평균값으로 대체
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# 수정된 데이터프레임을 CSV 파일로 저장
df.to_csv(r'C:\Users\Esc\Desktop\231114\Test\edited_normal_result.csv', index=False)
