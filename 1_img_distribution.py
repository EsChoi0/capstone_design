import os
import shutil 

# 원본 이미지 디렉토리 경로
source_directory = r'C:\Users\Esc\Desktop\Python\DWP\img\Training\raw_abnormal'
# 이미지 파일 확장자 리스트
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.JPG']


# 이미지 파일을 분류할 대상 폴더 경로 (총 5 종류)
target_directories = {
    '1': r'C:\Users\Esc\Desktop\Python\DWP\img\train\raw\abnormal01',
    '2': r'C:\Users\Esc\Desktop\Python\DWP\img\train\raw\abnormal02',
    '3': r'C:\Users\Esc\Desktop\Python\DWP\img\train\raw\abnormal03',
    '4': r'C:\Users\Esc\Desktop\Python\DWP\img\train\raw\abnormal04',
    '5': r'C:\Users\Esc\Desktop\Python\DWP\img\train\raw\abnormal05',
    '6': r'C:\Users\Esc\Desktop\Python\DWP\img\train\raw\abnormal06',
    '7': r'C:\Users\Esc\Desktop\Python\DWP\img\train\raw\abnormal07',
}

def move_images_to_folders_based_on_12th_char(source_dir, target_dir_mapping):
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)

        # 파일이 이미지 파일인지 확인
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            try:
                # 이미지 파일 이름에서 12번째 문자 가져오기
                char_12 = filename[11]

                # 12번째 문자에 해당하는 폴더로 이미지 파일 이동
                if char_12 in target_dir_mapping:
                    target_directory = target_dir_mapping[char_12]
                    shutil.move(file_path, os.path.join(target_directory, filename))
                else:
                    print(f"No target directory found for '{char_12}' in {filename}.")

            except IndexError:
                print(f"Skipping '{filename}' as it doesn't have an 12th character.")
            except Exception as e:
                print(f"An error occurred while processing '{filename}': {str(e)}")

if __name__ == "__main__":
    move_images_to_folders_based_on_12th_char(source_directory, target_directories)