import pandas as pd

# CSV 파일 읽기

file_path = 'C:\\Users\\Esc\\Desktop\\231114\\Test\\normal_result.csv'
result_dir = 'C:\\Users\\Esc\\Desktop\\231114\\Test\\updated_normal.csv'


df = pd.read_csv(file_path, encoding='CP949')




# # 병해 작물 기록
# # 질병코드 추출 함수
# def extract_disease(name):
#     parts = name.split('_')
#     fifth_element = parts[4] # 질병코드
    
#     if fifth_element == "a7":
#         return 'A7'
#     elif fifth_element == "a8":
#         return 'A8'
#     else:
#         return None  # 해당되지 않는 패턴의 경우 None 반환

# # 위험도 추출 함수
# def extract_risk(name):
#     parts = name.split('_')
#     fifth_element = parts[4] # 질병코드
#     six_element = int(parts[7]) # 부위코드
#     ninth_element = int(parts[8]) # 피해정도 코드
    
#     if fifth_element == "a7":
#         if ninth_element == 2:
#             return 'A1'
#         elif ninth_element == 3:
#             return 'A2'
#     elif fifth_element == "a8":
#         if ninth_element == 1:
#             return 'B1'
#         if ninth_element == 2:
#             return 'B2'
#         elif ninth_element == 3:
#             return 'B3'
#     # elif fifth_element == "0":
#     #     if six_element == 1:
#     #         return 'E'
#     # elif fifth_element == "0":
#     #     if six_element == 3:
#     #         return 'F'
#     return None  # 해당되지 않는 패턴의 경우 None 반환

# # 질병코드 및 위험도 열 업데이트
# df['disease'] = df['File Name'].apply(extract_disease)
# df['progression'] = df['File Name'].apply(extract_risk)




# 정상 작물 코드 기록하는 부분
# 6번째 요소에 따라 질병 코드를 업데이트하는 함수
def update_disease(name):
    parts = name.split('_')
    sixth_element = int(parts[5])

    if sixth_element == 1:
        return '01'
    elif sixth_element == 3:
        return '03'
    else:
        return None

# 6번째 요소에 따라 진행도를 업데이트하는 함수
def update_progression(name):
    parts = name.split('_')
    sixth_element = int(parts[5])

    if sixth_element == 1:
        return '0'
    elif sixth_element == 3:
        return '0'
    else:
        return None


# 질병코드 및 위험도 열 업데이트
df['disease'] = df['File Name'].apply(update_disease)
df['progression'] = df['File Name'].apply(update_progression)

# 결과를 새로운 CSV 파일로 저장
df.to_csv(result_dir, index=False, encoding='CP949')

print("질병코드 및 위험도 업데이트 완료!")
