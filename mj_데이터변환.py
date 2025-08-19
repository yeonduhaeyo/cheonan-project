import pandas as pd


# 의료기관.csv (from 엑셀 to csv)
excel_path = '의료.xlsx'

# 엑셀 불러오기
df = pd.read_excel(excel_path, sheet_name='1-의료기관', skiprows=6, header=0)
# 영어로 된 행 제거
def is_english_row(row):
    return row.astype(str).str.match(r'^[A-Za-z\s\.\(\)/&-]+$').all()
df = df[~df.apply(is_english_row, axis=1)]
# 지우고 싶은 열 인덱스
cols_to_drop = [2, 4, 6, 8, 9, 11, 13, 15, 16, 17, 19, 21, 23,24,26,31,33,34]
# 🔒 실제 존재하는 인덱스만 필터링
cols_to_drop = [i for i in cols_to_drop if i < len(df.columns)]
# 해당 열 제거
df = df.drop(columns=df.columns[cols_to_drop])
# 1️⃣ 첫 번째 행 제거 (병원수 등의 중복 헤더)
df = df.iloc[1:, :]
# 2️⃣ 마지막 3행 제거
df = df.iloc[:-3, :]
# 3️⃣ 인덱스 초기화 (선택)
df = df.reset_index(drop=True)

df.to_csv('의료기관.csv',index=False)

import pandas as pd
df=pd.read_csv('./data_MJ/응급의료기관.csv',encoding='cp949')
df=pd.read_csv('./data_MJ/노인_복지시설.csv',encoding='utf-8-sig')
df=pd.read_csv('./data_MJ/의료기관.csv',encoding='utf-8-sig')
df=pd.read_csv('./data_MJ/장애인_병원교통수단.csv',encoding='cp949')
df=pd.read_csv('./data_MJ/장애인_외출교통수단.csv',encoding='cp949')
df=pd.read_csv('./data_MJ/장애인_재활시설.csv',encoding='cp949')
df=pd.read_csv('./data_MJ/보건기관.csv')
df=pd.read_csv('./data_MJ/노인_여가복지시설.csv')


# 노인여가복지시설.csv로 변환  (from 엑셀 to csv)
excel_path = '복지.xlsx'
# 엑셀 불러오기
df = pd.read_excel(excel_path, sheet_name='18-노인여가복지시설',header=None)
# 2. 불필요한 열 삭제 (열 인덱스: 2, 3, 4, 5, 8)
df.drop(columns=[3,4,5,6,9], inplace=True)
# 3. 불필요한 행 삭제 (행 인덱스: 0, 1, 2, 4, 53)
df.drop(index=[0,1,2,3,4,5,6,7, 54], inplace=True)
# 4. 열 이름 지정
df.columns = ['읍/면/동','합계', '노인복지관','경로당','노인교실']
# 5. NaN 제거
df=df.dropna()
# 6. 인덱스 리셋
df.reset_index(drop=True, inplace=True)

df.to_csv('노인여가복지시설.csv',index=False)



# [노인복지시설]위도/경도 변환 -> 카카오 API 사용!!
import pandas as pd
import requests
import re
# 1. CSV 불러오기 (주소 열 이름은 '주소'라고 가정)
df = pd.read_csv('./data_MJ/노인복지시설.csv',encoding='utf-8-sig')
df.columns = df.columns.str.strip()
# 2. 카카오 API 키
KAKAO_API_KEY = '5ea52feee429d12bc70b4cc8ca489063'
# 3. 주소 전처리 함수
def clean_address(addr):
    if pd.isna(addr):
        return ""
    addr = str(addr)
    addr = re.sub(r'\(.*?\)', '', addr)        # 괄호 제거
    addr = addr.replace(',', ' ')              # 쉼표 제거
    addr = addr.replace('  ', ' ').strip()     # 중복 공백 제거
    return addr
# 4. 위도/경도 구하는 함수
def get_lat_lon(address):
    url = 'https://dapi.kakao.com/v2/local/search/address.json'
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {'query': address}
    
    res = requests.get(url, headers=headers, params=params)
    result = res.json()
    # 디버깅용 출력
    print(f"[요청 주소] {address}")
    print(f"[응답] {result}")
    if res.status_code == 200 and result.get('documents'):
        lat = result['documents'][0]['y']
        lon = result['documents'][0]['x']
        return pd.Series([lat, lon])
    return pd.Series([None, None])

# 5. 주소 열 전처리 및 적용
df['주소'] = df['주소'].apply(clean_address)
df[['위도', '경도']] = df['주소'].apply(get_lat_lon)


# 결측치 확인
null_rows = df[df['위도'].isna() | df['경도'].isna()]
print(null_rows[['주소', '위도', '경도']])

# 수동 변환
# index만 확인해서 해당 인덱스 수동 입력

df.at[4, '위도'] = 36.8173311499575
df.at[4, '경도'] = 127.141784555121
df.at[12, '위도'] = 36.83770426916
df.at[12, '경도'] = 127.139829899458
df.at[28, '위도'] = 36.8361870289748
df.at[28, '경도'] = 127.137708695265
df.at[41, '위도'] = 36.8356237328293
df.at[41, '경도'] = 127.133935053832
df.at[43, '위도'] = 36.8052363185714
df.at[43, '경도'] = 127.15032774916
df.at[60, '위도'] = 36.7771365591271
df.at[60, '경도'] = 127.211625433146
df.at[78, '위도'] = 36.8069221509273
df.at[78, '경도'] = 127.133617107919
df.at[83, '위도'] = 36.8135104762167
df.at[83, '경도'] = 127.140696017685
df.at[85, '위도'] = 36.8215905550037
df.at[85, '경도'] = 127.125724359546
df.at[87, '위도'] = 36.7800848581284
df.at[87, '경도'] = 127.140332212358
df.at[90, '위도'] = 36.8232234748061
df.at[90, '경도'] = 127.124074774017
df.at[107, '위도'] = 36.7955605300395
df.at[107, '경도'] = 127.122874396073
df.at[111, '위도'] = 36.8209843432287
df.at[111, '경도'] = 127.102157004348
df.at[124, '위도'] = 36.8056233954062
df.at[124, '경도'] = 127.139872961068
df.at[126, '위도'] = 36.8108947734158
df.at[126, '경도'] = 127.153066877525
df.at[168, '위도'] = 36.9185171209731
df.at[168, '경도'] = 127.130050208779
df.at[169, '위도'] = 36.8032502637586
df.at[169, '경도'] = 127.1386113693
df.at[185, '위도'] = 36.7800848581284
df.at[185, '경도'] = 127.140332212358

# 6. 저장
df.to_csv('노인_복지시설.csv', index=False)



# [보건의료기관]
import pandas as pd
import requests
import re
df=pd.read_csv('./보건의료기관.csv',encoding='cp949')
df=df[(df['시군구']=='천안시 동남구')|(df['시군구']=='천안시 서북구')]
# 2. 카카오 API 키
KAKAO_API_KEY = '5ea52feee429d12bc70b4cc8ca489063'
# 3. 주소 전처리 함수
def clean_address(addr):
    if pd.isna(addr):
        return ""
    addr = str(addr)
    addr = re.sub(r'\(.*?\)', '', addr)        # 괄호 제거
    addr = addr.replace(',', ' ')              # 쉼표 제거
    addr = addr.replace('  ', ' ').strip()     # 중복 공백 제거
    return addr
# 4. 위도/경도 구하는 함수
def get_lat_lon(address):
    url = 'https://dapi.kakao.com/v2/local/search/address.json'
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {'query': address}
    
    res = requests.get(url, headers=headers, params=params)
    result = res.json()
    # 디버깅용 출력
    print(f"[요청 주소] {address}")
    print(f"[응답] {result}")
    if res.status_code == 200 and result.get('documents'):
        lat = result['documents'][0]['y']
        lon = result['documents'][0]['x']
        return pd.Series([lat, lon])
    return pd.Series([None, None])
# 5. 주소 열 전처리 및 적용
df['주소'] = df['주소'].apply(clean_address)
df[['위도', '경도']] = df['주소'].apply(get_lat_lon)
# 결측치 확인
null_rows = df[df['위도'].isna() | df['경도'].isna()]
print(null_rows[['주소', '위도', '경도']])