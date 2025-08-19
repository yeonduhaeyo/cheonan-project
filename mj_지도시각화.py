#  <장애인 시각화>

import pandas as pd
import json
import geopandas as gpd
import plotly.express as px
# 1. 지도 데이터 불러오고 천안만 분리해 geojson으로 변환
gdf = gpd.read_file('./data_MJ/행정동/행정동.shp', encoding='euc-kr')
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)
else:
    gdf = gdf.to_crs(epsg=4326)

gdf['ADM_CD'] = gdf['ADM_CD'].astype(str).str.strip()

# 천안시 동남구, 서북구 행정동 코드(앞 4자리 기준 필터)
# 동남구 : 34011
# 서북구: 34012
cheonan_gdf = gdf[gdf['ADM_CD'].str.startswith(('34011', '34012'))]
cheonan_gdf.to_file('./data_MJ/cheonan_gdf.geojson',driver='GeoJSON')

# 2. geojson 불러오기
with open('./data_MJ/cheonan_gdf.geojson',encoding='utf-8') as f:
    geojson_data=json.load(f)

# 3. 읍면동별 장애인 인구
disabled=pd.read_csv('./data_MJ/읍면동별_장애인_등록인구_2024.csv',encoding='cp949',skiprows=2)
# 필요한 컬럼만 추출
disabled = disabled.iloc[:, [0, 1, 2]]  # '읍면동별(1)', '읍면동별(2)', '장애인등록 인구'
# 컬럼명 정리
disabled.columns = ['구군', '읍면동', '장애인등록인구']
# 합계, 소계 행 제거
disabled = disabled[~disabled['읍면동'].isin(['소계', '합계'])]
disabled = disabled[~disabled['구군'].isin(['합계'])]
# 장애인등록인구 숫자화
disabled['장애인등록인구'] = pd.to_numeric(disabled['장애인등록인구'], errors='coerce')
# 인덱스 리셋
disabled = disabled.reset_index(drop=True)
disabled= disabled.rename(columns={"읍면동": "ADM_NM"})
# 4. 장애인 인구 데이터 시각화
fig = px.choropleth_mapbox(
    disabled,  # ✅ GeoDataFrame 사용
    geojson=geojson_data,
    locations="ADM_NM",  # ✅ 실제 GeoJSON 속성과 연결되는 컬럼
    featureidkey="properties.ADM_NM",  # ✅ GeoJSON 안에 있는 키 경로
    color='장애인등록인구',
    color_continuous_scale="Blues",
    mapbox_style="carto-positron",
    center={"lat": 36.815, "lon": 127.113},
    zoom=11,
    opacity=0.7,
    hover_name="ADM_NM"
)
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0}) 

# 5. [장애인 재활시설] 위치
df_facilities = pd.read_csv('./data_MJ/장애인_재활시설.csv')
fig.add_scattermapbox(
    lat=df_facilities['위도'],
    lon=df_facilities['경도'],
    mode='markers',
    marker=dict(size=10, color='blue'),
    text=df_facilities['명 칭'] + "<br>" + df_facilities['소 재 지'],
    hoverinfo='text',
    name='장애인 재활시설'
)
# 6. <병원 데이터>추가 (?)

# 7. [응급의료시설] 위치
df_hospital_e = pd.read_csv('./data_MJ/응급의료기관.csv',encoding='cp949')
em_hospital=df_hospital_e[df_hospital_e['주소'].str.contains('천안')]
# em_hospital.columns
fig.add_scattermapbox(
    lat=em_hospital['병원위도'],
    lon=em_hospital['병원경도'],
    mode='markers',
    marker=dict(size=10, color='red'),
    text=em_hospital['기관명'] + "<br>" + em_hospital['주소'],
    hoverinfo='text',
    name='응급의료기관'
)
# 8.[보건기관] 위치
df_healthcenter = pd.read_csv('./data_MJ/보건기관.csv')
# df_healthcenter.columns
fig.add_scattermapbox(
    lat=df_healthcenter['위도'],
    lon=df_healthcenter['경도'],
    mode='markers',
    marker=dict(size=10, color='green'),
    text=df_healthcenter['시설명'] + "<br>" + df_healthcenter['주소'],
    hoverinfo='text',
    name='보건기관'
)

# 9. 레이아웃
fig.update_layout(
    margin={"r": 0, "t": 30, "l": 0, "b": 0},
    title='천안시 장애인 대상 복지시설 위치'
)
fig.update_layout(
    legend=dict(
        orientation="h",     # 수평 정렬
        yanchor="top",
        y=-0.1,              # 아래쪽으로 이동
        xanchor="center",
        x=0.5,
        font=dict(size=13)
    )
)
fig.show()







#  <노인> 시각화

import pandas as pd
import json
import geopandas as gpd
import plotly.express as px
import re
# 1. 지도 데이터 불러오고 천안만 분리해 geojson으로 변환
gdf = gpd.read_file('./data_MJ/행정동/행정동.shp', encoding='euc-kr')
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)
else:
    gdf = gdf.to_crs(epsg=4326)

gdf['ADM_CD'] = gdf['ADM_CD'].astype(str).str.strip()
# 천안시 동남구, 서북구 행정동 코드(앞 4자리 기준 필터)
# 동남구 : 34011
# 서북구: 34012
cheonan_gdf = gdf[gdf['ADM_CD'].str.startswith(('34011', '34012'))]
cheonan_gdf.to_file('./data_MJ/cheonan_gdf.geojson',driver='GeoJSON')

# 2. geojson 불러오기
with open('./data_MJ/cheonan_gdf.geojson',encoding='utf-8') as f:
    geojson_data=json.load(f)

# 3. 읍면동별 노인 인구
older=pd.read_csv('./data_MJ/노인인구.csv',encoding='cp949')
older = older[older['행정구역'].str.startswith('충청남도 천안시')]
older = older[['행정구역', '2025년07월_65세이상전체']]
remove_list = [
    "충청남도 천안시 (4413000000)",
    "충청남도 천안시 동남구 (4413100000)",
    "충청남도 천안시 서북구 (4413300000)"
]
older = older[~older['행정구역'].isin(remove_list)].reset_index(drop=True)
older = older.rename(columns={'2025년07월_65세이상전체': '65세 이상 인구'})
older['행정구역'] = older['행정구역'].str.replace(r'^충청남도 천안시 (동남구|서북구) ', '', regex=True)
older['행정구역'] = older['행정구역'].str.replace(r'\s*\(\d+\)', '', regex=True)
older = older.rename(columns={'행정구역': 'ADM_NM'})
older['65세 이상 인구'] = pd.to_numeric(older['65세 이상 인구'].str.replace(',', ''), errors='coerce')

fig = px.choropleth_mapbox(
    older,  # ✅ GeoDataFrame 사용
    geojson=geojson_data,
    locations="ADM_NM",  # ✅ 실제 GeoJSON 속성과 연결되는 컬럼
    featureidkey="properties.ADM_NM",  # ✅ GeoJSON 안에 있는 키 경로
    color='65세 이상 인구',
    color_continuous_scale="Greens",
    range_color=(older['65세 이상 인구'].min(), older['65세 이상 인구'].max()),
    mapbox_style="carto-positron",
    center={"lat": 36.815, "lon": 127.113},
    zoom=11,
    opacity=0.5,
    hover_name="ADM_NM"
)
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0}) 


# 5. <병원 데이터>추가 (?)

# 6. [응급의료시설]  위치
df_hospital_e = pd.read_csv('./data_MJ/응급의료기관.csv',encoding='cp949')
em_hospital=df_hospital_e[df_hospital_e['주소'].str.contains('천안')]
# em_hospital.columns
fig.add_scattermapbox(
    lat=em_hospital['병원위도'],
    lon=em_hospital['병원경도'],
    mode='markers',
    marker=dict(size=10, color='red'),
    text=em_hospital['기관명'] + "<br>" + em_hospital['주소'],
    hoverinfo='text',
    name='응급의료기관'
)
# 7.[보건기관] 위치
df_healthcenter = pd.read_csv('./data_MJ/보건기관.csv')
# df_healthcenter.columns
fig.add_scattermapbox(
    lat=df_healthcenter['위도'],
    lon=df_healthcenter['경도'],
    mode='markers',
    marker=dict(size=10, color='green'),
    text=df_healthcenter['시설명'] + "<br>" + df_healthcenter['주소'],
    hoverinfo='text',
    name='보건기관'
)
# 8. [노인 복지시설] 위치
df_welfare_facilities = pd.read_csv('./data_MJ/노인_복지시설.csv')
df_welfare_facilities.columns
fig.add_scattermapbox(
    lat=df_welfare_facilities['위도'],
    lon=df_welfare_facilities['경도'],
    mode='markers',
    marker=dict(size=10, color='yellow'),
    text=df_welfare_facilities['기관명'] + "<br>" + df_welfare_facilities['주소'],
    hoverinfo='text',
    name='노인 복지시설'
)
# 9. 레이아웃
fig.update_layout(
    margin={"r": 0, "t": 30, "l": 0, "b": 0},
    title='천안시 노인 대상 복지시설 위치'
)
fig.update_layout(
    legend=dict(
        orientation="h",     # 수평 정렬
        yanchor="top",
        y=-0.1,              # 아래쪽으로 이동
        xanchor="center",
        x=0.6,
        font=dict(size=13)
    )
)
fig.show()

