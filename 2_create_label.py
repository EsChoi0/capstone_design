import os
import csv
import pandas as pd
import json

image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.JPG']


# 이미지를 저장할 리스트 생성
image_files = []
# 키와 값을 저장할 빈 리스트 생성
disease_list = []
risk_list = []
points_list = []

# 이미지와 JSON 파일 경로, * 계속 수정해주어야 함
image_directory_path = r'C:\Users\Esc\Desktop\Python\DWP\img\train\raw\abnormal\abnormal03' 
json_directory_path = r'C:\Users\Esc\Desktop\Python\DWP\img\train\label\abnormal\abnormal03'


# 디렉토리 내의 모든 파일 목록을 가져옵니다.
for filename in os.listdir(image_directory_path):
    # 파일의 확장자를 소문자로 변환합니다.
    file_extension = os.path.splitext(filename)[-1].lower()
    
    # 확장자가 이미지 확장자 중 하나라면 리스트에 추가합니다.
    if file_extension in image_extensions:
        image_files.append(filename)

def create_csv():
    # * CSV 파일 생성
    new_csv_file_path = 'abnormal_annotations_03.csv'
    # 빈 DataFrame 생성
    data = pd.DataFrame(columns=["Image_Name", "Image_Path", "Disease", "Risk", "Points"])
    # DataFrame을 CSV 파일로 저장
    data.to_csv(new_csv_file_path, index=False)

create_csv()
# * CSV 파일 경로 계속 수정해주어야 함
csv_file_path = r'C:\Users\Esc\Desktop\Python\DWP\abnormal_annotations_03.csv'
# CSV 파일 경로 및 열 이름 설정
column_names = ["Image_Name", "Image_Path", "Disease", "Risk", "Points"]

# 리스트 요소를 CSV 파일에 추가
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
    
    # CSV 파일에 열 이름을 쓰기
    csv_writer.writeheader()
    
    # 리스트 요소를 'Image_Name' 열에 추가
    for item in image_files:
        csv_writer.writerow({"Image_Name": item})



# JSON 디렉토리에서 모든 JSON 파일을 순회하며 처리
for filename in os.listdir(json_directory_path):
    if filename.endswith('.json'):  # JSON 파일만 처리
        json_file_path = os.path.join(json_directory_path, filename)
        
        # JSON 파일 불러오기
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
        
        # JSON 데이터에서 원하는 키와 값을 추출하여 리스트에 추가
        if 'annotations' in data:
            annotations = data['annotations']
            disease_list.append(annotations.get('disease', None))
            risk_list.append(annotations.get('risk', None))
            points_list.append(annotations.get('points', None))

# CSV 파일 업데이트
dataframe = pd.read_csv(csv_file_path)  # CSV 파일 불러오기

# 각 열에 JSON 데이터 추가
dataframe['Disease'] = disease_list
dataframe['Risk'] = risk_list
dataframe['Points'] = points_list
dataframe['Image_Path'] = image_directory_path
# 업데이트된 데이터프레임을 CSV 파일로 저장
dataframe.to_csv(csv_file_path, index=False)
