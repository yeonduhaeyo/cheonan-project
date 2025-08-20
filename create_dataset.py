### Import
import requests
import math
import time
import pandas as pd
import numpy as np


### 국토교통부_(TAGO)_버스노선정보

# 노선번호목록 조회
SERVICE_KEY = "key"
API_URL = "http://apis.data.go.kr/1613000/BusRouteInfoInqireService/getRouteNoList"
CITY_CODE = 34010 # 천안시 도시코드

params = {
    "serviceKey": SERVICE_KEY,
    "cityCode": CITY_CODE,
    "pageNo": 1,
    "numOfRows": 999,
    "_type": "json"
}

response = requests.get(API_URL, params=params)

# 응답 상태와 내용 확인
if response.status_code == 200:
    data = response.json()
    total_count = data['response']['body']['totalCount']
    total_pages = math.ceil(total_count / params["numOfRows"])
    print(f"데이터 수 : 총 {total_count}건, {total_pages}페이지")

    all_items = []
    for page in range(1, total_pages + 1):
        params["pageNo"] = page
        res = requests.get(API_URL, params=params)
        if res.status_code == 200:
            data_page = res.json()
            items = data_page['response']['body']['items']
            if items:
                # 단일 아이템일 땐 dict, 복수일 땐 list 임
                page_items = items['item']
                if isinstance(page_items, dict):
                    all_items.append(page_items)
                else:
                    all_items.extend(page_items)
            else:
                print(f"{page}페이지: 데이터 없음")
        else:
            print(f"{page}페이지 요청 실패: {res.status_code}")

    df = pd.DataFrame(all_items)
    print(df.head())
else:
    print(f"첫 요청 실패: {response.status_code}")
    print(response.text)

# 영문 컬럼명 → 국문 컬럼명 매핑 딕셔너리
col_mapping = {
    'routeid': '노선ID',
    'routeno': '노선번호',
    'routetp': '노선유형',
    'endnodenm': '종점',
    'startnodenm': '기점',
    'endvehicletime': '막차시간',
    'startvehicletime': '첫차시간',
}

# 컬럼명 변경
df.rename(columns=col_mapping, inplace=True)

# CSV 저장 (utf-8-sig로 저장하면 한글 깨짐 방지에 좋음)
df.to_csv('./data/천안시_노선번호목록.csv', index=False, encoding='utf-8-sig')



# 노선별경유정류소목록 조회
SERVICE_KEY = "key"
API_URL = "http://apis.data.go.kr/1613000/BusRouteInfoInqireService/getRouteAcctoThrghSttnList"
CITY_CODE = "34010"  # 천안시 도시코드
INPUT_CSV = "./data/천안시_노선번호목록.csv"
OUTPUT_CSV = "./data/천안시_노선별경유정류소목록.csv"

# 최대 재시도 횟수 설정
MAX_RETRIES = 5
DELAY = 2  # 요청 간 대기 시간 (초)

# ============================
route_df = pd.read_csv(INPUT_CSV)
route_ids = route_df['노선ID'].tolist()

# 데이터 저장용 리스트
all_data = []

def fetch_data(route_id, retries=0):
    params = {
        'serviceKey': SERVICE_KEY,
        'cityCode': CITY_CODE,
        'routeId': route_id,
        '_type': 'json',
        'numOfRows': 100,
        'pageNo': 1
    }

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # 4xx, 5xx 오류 발생 시 예외 처리

        json_data = response.json()
        items = json_data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
        
        if isinstance(items, dict):  # 단일 아이템 처리
            items = [items]

        return items
    
    except requests.exceptions.RequestException as e:
        # 에러 발생 시 재시도
        if retries < MAX_RETRIES:
            print(f"[에러] routeId={route_id} 요청 실패. {retries + 1}/{MAX_RETRIES} 재시도 중...")
            time.sleep(DELAY)  # 잠시 대기 후 재시도
            return fetch_data(route_id, retries + 1)  # 재귀적으로 재시도
        else:
            print(f"[에러] routeId={route_id} 요청이 {MAX_RETRIES}번 실패했습니다.")
            return None

# 전체 routeId 목록을 순회하며 데이터 수집
for route_id in route_ids:
    print(f"[시작] routeId={route_id} 데이터 수집")
    data = fetch_data(route_id)

    if data:
        for item in data:
            item['routeid'] = route_id  # routeId 추가
            all_data.append(item)

    time.sleep(0.2)  # API 호출 속도 제한을 고려하여 잠시 대기

len(all_data)

# 3. CSV로 저장
if all_data:
    df = pd.DataFrame(all_data)
    col_mapping = {
    "resultCode": "결과코드",
    "resultMsg": "결과메세지",
    "numOfRows": "한 페이지 결과 수",
    "pageNo": "페이지 수",
    "totalCount": "데이터 총 개수",
    "routeid": "노선ID",
    "nodeid": "정류소ID",
    "nodenm": "정류소명",
    "nodeno": "정류소번호",
    "nodeord": "정류소순번",
    "gpslati": "정류소 Y좌표",
    "gpslong": "정류소 X좌표",
    "updowncd": "상하행구분코드"
    }
    # 컬럼명 변경
    df.rename(columns=col_mapping, inplace=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"총 {len(df)}건 저장 완료 → {OUTPUT_CSV}")
else:
    print("저장할 데이터가 없습니다.")
    
    

#노선정보항목 조회
SERVICE_KEY = "lXgIrPRsjleQjcemMPjgZgqE+z/4mH9us3HiWrjFcAmFMeQPcmmLOhrHqmzYuafJDMgUf5uHR636Xvmrt+/EGQ=="
API_URL = "http://apis.data.go.kr/1613000/BusRouteInfoInqireService/getRouteInfoIem"
CITY_CODE = "34010"  # 천안 도시코드 (확인 필요)
INPUT_CSV = "./data/천안시_노선번호목록.csv"
OUTPUT_CSV = "./data/천안시_노선정보항목.csv"

route_df = pd.read_csv(INPUT_CSV)
route_ids = route_df['노선ID'].tolist()

all_data = []  # 결과 저장용 데이터프레임


# ==== API 호출 ====
for route_id in route_ids:
    params = {
        "serviceKey": SERVICE_KEY,
        "_type": "json",
        "cityCode": CITY_CODE,
        "routeId": route_id
    }
    try:
        response = requests.get(API_URL, params=params)
        data = response.json()

        # 응답에서 item 추출
        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
        if isinstance(items, dict):
            items = [items]  # 단일 데이터도 리스트로 변환

        all_data.extend(items)

    except Exception as e:
        print(f"{route_id} 처리 중 오류: {e}")

    time.sleep(2)  # API 호출 간 딜레이 (tps 제한)

df = pd.DataFrame(all_data)

col_map = {
    "resultCode": "결과코드",
    "resultMsg": "결과메세지",
    "routeid": "노선ID",
    "routeno": "노선번호",
    "routetp": "노선유형",
    "endnodenm": "종점",
    "startnodenm": "기점",
    "endvehicletime": "막차시간",
    "startvehicletime": "첫차시간",
    "intervaltime": "배차간격(평일)",
    "intervalsattime": "배차간격(토요일)",
    "intervalsuntime": "배차간격(일요일)"
}

df.rename(columns=col_map, inplace=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"저장 완료: {OUTPUT_CSV}, 총 {len(df)}건")