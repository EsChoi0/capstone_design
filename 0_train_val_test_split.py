import os
import shutil
import random

# 원본 이미지 및 JSON 파일이 있는 디렉토리 경로
image_dir = r'C:\Users\Esc\Downloads\IMG\Training\raw_abnormal\fruite'
json_dir = r'C:\Users\Esc\Downloads\IMG\Training\label_abnormal\fruite'

# 이미지와 JSON 파일을 따로 저장할 디렉토리 생성
output_image_dir = r'C:\Users\Esc\Desktop\Python\DWP\image'
output_json_dir = r'C:\Users\Esc\Desktop\Python\DWP\json'

# 이미지 및 JSON 파일 목록 가져오기
image_files = os.listdir(image_dir)

# 이미지 및 JSON 파일 목록을 무작위로 섞기
random.shuffle(image_files)

# 분할 비율 설정 (7:2:1)
train_ratio = 0.7
validation_ratio = 0.2
test_ratio = 0.1

total_samples = len(image_files)
train_split = int(train_ratio * total_samples)
validation_split = int(validation_ratio * total_samples)

# 데이터 분할 및 파일 이동
def move_files(source_dir, target_dir, file_list):
    for file_name in file_list:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)
        if not os.path.exists(target_path):  # 폴더가 없는 경우에만 폴더 생성
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.move(source_path, target_path)

train_images = image_files[:train_split]
validation_images = image_files[train_split:train_split+validation_split]
test_images = image_files[train_split+validation_split:]

# 이미지 파일 이동
move_files(image_dir, os.path.join(output_image_dir, 'train'), train_images)
move_files(image_dir, os.path.join(output_image_dir, 'validation'), validation_images)
move_files(image_dir, os.path.join(output_image_dir, 'test'), test_images)

# JSON 파일 이동 (이미지 파일 이름에 ".json" 추가하여 이동)
for image_file in train_images:
    json_file = image_file + '.json'
    if json_file in os.listdir(json_dir):
        move_files(json_dir, os.path.join(output_json_dir, 'train'), [json_file])

for image_file in validation_images:
    json_file = image_file + '.json'
    if json_file in os.listdir(json_dir):
        move_files(json_dir, os.path.join(output_json_dir, 'validation'), [json_file])

for image_file in test_images:
    json_file = image_file + '.json'
    if json_file in os.listdir(json_dir):
        move_files(json_dir, os.path.join(output_json_dir, 'test'), [json_file])