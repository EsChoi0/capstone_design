import os
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

# 파일이 있는 폴더 경로
input_folder = r'C:\Users\Esc\Desktop\231114\Test\fillter_disease'
output_file = r'C:\Users\Esc\Desktop\231114\Test\disease_result.csv'

def process_file(file_path, result_list):
    encodings = ['cp949', 'euc-kr']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            print(f"Failed to decode with {encoding} encoding.")

    df['측정시각'] = pd.to_datetime(df['측정시각'], format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values(by='측정시각')

    file_name = os.path.basename(file_path)

    try:
        start_time = df['측정시각'].iloc[0]
    except IndexError as e:
        print(f"No data found in file {file_path}")
        return

    record_time = start_time

    temp_list = []
    humidity_list = []
    while record_time <= pd.to_datetime(df['측정시각'].iloc[-1], format='%Y-%m-%d %H:%M:%S'):
        record_time_str = record_time.strftime('%Y-%m-%d %H:%M:%S')
        if record_time_str in df['측정시각'].dt.strftime('%Y-%m-%d %H:%M:%S').values:
            temp = df.loc[df['측정시각'] == record_time_str, '내부 온도 1 평균'].values[0]
            humidity = df.loc[df['측정시각'] == record_time_str, '내부 습도 1 평균'].values[0]
        else:
            temp = None
            humidity = None

        temp_list.append(temp)
        humidity_list.append(humidity)

        record_time += timedelta(hours=1)

    try:
        row_data = {'File Name': file_name}
        for i, temp in enumerate(temp_list, 1):
            row_data[f'Temperature_{i}'] = temp
        for i, humidity in enumerate(humidity_list, 1):
            row_data[f'Humidity_{i}'] = humidity

        result_list.append(pd.DataFrame([row_data]))

    except ValueError as e:
        print(f"Error adding data from file {file_path} to result_list: {e}")

    return file_name  # 각 프로세스가 처리한 파일 이름을 반환

# 파일 목록 가져오기
files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

if __name__ == '__main__':
    # 각 프로세스의 결과를 저장할 리스트
    result_list = []

    # 멀티프로세싱 코드
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, os.path.join(input_folder, file), result_list) for file in files]

        # as_completed를 사용하여 결과를 기다림
        for future in as_completed(futures):
            # 프로세스의 출력을 화면에 출력
            print(f"Processed file: {future.result()}")

    # 결과 데이터프레임을 CSV 파일로 저장
    result_df = pd.concat(result_list, ignore_index=True)
    result_df.to_csv(output_file, index=False)
