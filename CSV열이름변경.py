import os
import pandas as pd

# 대상 디렉토리 경로

# 대상 디렉토리 및 목표 디렉토리 경로
source_directory = r'C:\Users\Esc\Desktop\231114\Test\병해'
target_directory = r'C:\Users\Esc\Desktop\231114\Test\fillter_disease'

# 디렉토리 내의 모든 CSV 파일에 대해 작업 수행
for filename in os.listdir(source_directory):
    if filename.endswith(".csv"):
        source_file_path = os.path.join(source_directory, filename)

        # CSV 파일 읽기
        df = pd.read_csv(source_file_path, encoding='euc-kr')

        # 열 이름 변경
        df.columns = df.columns.str.replace('내부.온도.1.평균', '내부 온도 1 평균')
        df.columns = df.columns.str.replace('내부.습도.1.평균', '내부 습도 1 평균')

        # 변경된 데이터프레임을 목표 디렉토리에 저장
        target_file_path = os.path.join(target_directory, f"{filename}")
        df.to_csv(target_file_path, index=False, encoding='euc-kr')

        # 변경된 데이터프레임 출력
        print(f"Modified {filename} and saved to {target_file_path}.")

print("All files processed.")
