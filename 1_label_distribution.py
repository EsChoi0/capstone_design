import os
import shutil

# 원본 JSON 디렉토리 경로
source_json_directory = r'C:\Users\Esc\Desktop\Python\DWP\img\Training\label_abnormal'

# 대상 폴더의 경로 매핑 (12번째 문자에 따라 1번 폴더로, 2번 폴더로 이동)
target_directory_mapping = {
    '1': r'C:\Users\Esc\Desktop\Python\DWP\img\train\label\abnormal01',
    '2': r'C:\Users\Esc\Desktop\Python\DWP\img\train\label\abnormal02',
    '3': r'C:\Users\Esc\Desktop\Python\DWP\img\train\label\abnormal03',
    '4': r'C:\Users\Esc\Desktop\Python\DWP\img\train\label\abnormal04',
    '5': r'C:\Users\Esc\Desktop\Python\DWP\img\train\label\abnormal05',
    '6': r'C:\Users\Esc\Desktop\Python\DWP\img\train\label\abnormal06',
    '7': r'C:\Users\Esc\Desktop\Python\DWP\img\train\label\abnormal07',
}

def move_json_to_folders_based_on_12th_char(json_dir, target_dir_mapping):
    for json_filename in os.listdir(json_dir):
        json_file_path = os.path.join(json_dir, json_filename)

        # JSON 파일 확인
        if os.path.isfile(json_file_path) and json_filename.endswith('.json'):
            try:
                # JSON 파일 이름에서 12번째 문자 가져오기
                char_12 = json_filename[11]

                # 해당 문자에 대응되는 타겟 폴더 경로 가져오기
                if char_12 in target_dir_mapping:
                    target_directory = target_dir_mapping[char_12]

                    # JSON 파일을 해당 폴더로 이동
                    shutil.move(json_file_path, os.path.join(target_directory, json_filename))
                else:
                    print(f"No target directory found for '{char_12}' in {json_filename}.")

            except IndexError:
                print(f"Skipping '{json_filename}' as it doesn't have an 12th character.")
            except Exception as e:
                print(f"An error occurred while processing '{json_filename}': {str(e)}")

if __name__ == "__main__":
    move_json_to_folders_based_on_12th_char(source_json_directory, target_directory_mapping)
