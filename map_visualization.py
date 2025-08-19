### plotly로 노선별 경유 정류소 목록 표시

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("./data_HWJ/천안시_노선별경유정류소목록.csv")

df.columns

fig = px.scatter_mapbox(
    df,
    lat="정류소 Y좌표",
    lon="정류소 X좌표",
    color="노선ID",
    hover_name="정류소명", # 마우스 오버 시 표시한 텍스트
    # hover_data={},
    # text="text",
    zoom=11,
    height=650,
    );
 # carto-positron : 무료, 지도 배경 스타일 지정
fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
fig.show();


### shp 파일
# .shp 불러오기 → 좌표계 변환 → .geojson 저장
# CSV 불러오기 → 자치구별 데이터 집계
# GeoJSON과 CSV의 키값 맞추기
# Plotly로 색깔 입힌 지도(Choropleth Mapbox) 생성
import pandas as pd
pd.set_option('display.max_columns', None)
import geopandas as gpd
from shapely.geometry import Point
import json
import plotly.express as px

# 1. 읍면동 경계 shp 파일 불러오기, 좌표계 변환
gdf = gpd.read_file("./data_HWJ/map_data/emd.shp")
print("원본 좌표계:", gdf.crs)
gdf.crs = "EPSG:5179"  # 원본 좌표계 지정 (예: UTM-K)
gdf = gdf.to_crs(epsg=4326)  # WGS84 위경도로 변환

# 2. 정류소 CSV 불러오기
df = pd.read_csv("./data_HWJ/천안시_노선별경유정류소목록.csv")
print(df.head())
print(df.info())

# 3. 정류소 DataFrame에 위도,경도 기준 Point geometry 생성 (경도=X, 위도=Y)
geometry = [Point(xy) for xy in zip(df['정류소 X좌표'], df['정류소 Y좌표'])]
gdf_stops = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

# 4. 공간 조인 (정류소 포인트가 포함된 읍면동 폴리곤 가져오기)
gdf_joined = gpd.sjoin(gdf_stops, gdf, how='left', predicate='within')

# 5. 읍면동명 컬럼명 확인 (예: 'EMD_KOR_NM')와 필요한 컬럼 추출
print(gdf_joined.columns)
# 일반적으로 읍면동 명 컬럼이 'EMD_KOR_NM'일 가능성 높음

# 6. 읍면동별 정류소 수 집계
agg_df = gdf_joined.groupby('EMD_KOR_NM').size().reset_index(name='정류소수')
print(agg_df.head())

# 7. GeoJSON 파일로 저장 (Plotly용)
gdf.to_file("./data_HWJ/map_data/mapfile.geojson", driver="GeoJSON")

with open('./data_HWJ/map_data/mapfile.geojson', encoding='utf-8') as f:
    geojson_data = json.load(f)

# 8. Plotly Choropleth Mapbox 시각화
fig = px.choropleth_mapbox(
    agg_df,
    geojson=geojson_data,
    locations='EMD_KOR_NM',
    featureidkey='properties.EMD_KOR_NM',
    color='정류소수',
    color_continuous_scale='Viridis',
    mapbox_style='carto-positron',
    center={'lat': 36.8, 'lon': 127.1},  # 천안시 중심 좌표 설정
    zoom=10,
    opacity=0.6,
    labels={'정류소수': '정류소 개수'},
    title='읍면동별 정류소 개수'
)

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()



import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import plotly.graph_objects as go
import plotly.express as px

# 1. 정류소 데이터 불러오기
df = pd.read_csv('./data_HWJ/천안시_노선별경유정류소목록.csv')

# 2. 노선별 정류소를 순서대로 정렬
df_sorted = df.sort_values(by=['노선ID', '정류소순번'])

# 3. 노선별 LineString 생성 함수
def create_linestring(group):
    points = list(zip(group['정류소 X좌표'], group['정류소 Y좌표']))
    return LineString(points)

# 4. 노선별 LineString GeoDataFrame 생성
lines = df_sorted.groupby('노선ID').apply(create_linestring).reset_index()
lines.columns = ['노선ID', 'geometry']
gdf_lines = gpd.GeoDataFrame(lines, geometry='geometry', crs='EPSG:4326')

# 5. Plotly로 시각화
fig = go.Figure()

colors = px.colors.qualitative.Dark24  # 색상 팔레트

for i, (route_id, row) in enumerate(gdf_lines.iterrows()):
    x, y = row.geometry.xy
    lon = list(x)  # array -> list 변환
    lat = list(y)
    fig.add_trace(go.Scattermapbox(
        lon=lon,
        lat=lat,
        mode='lines+markers',
        name=route_id,
        line=dict(width=1, color=colors[i % len(colors)]),
        marker=dict(size=5)
    ))

fig.update_layout(
    mapbox_style='carto-positron',
    mapbox_center={"lat": 36.8, "lon": 127.1},  # 천안시 중심 좌표
    mapbox_zoom=10,
    margin={"r":0,"t":0,"l":0,"b":0},
    legend_title_text='노선ID별',
    title='노선별 정류소 시각화'
    
)

fig.show()


### 각 정류장별 최단 거리 복지시설 찾기
# KDTree (특히 cKDTree)
# 개념: 점 데이터를 k차원 공간에 트리 구조로 저장하여, 최근접 이웃 탐색을 빠르게 수행.
# 특징:
# 투영 좌표계(예: UTM)로 변환 → 거리 계산이 유클리드 거리(평면거리)가 됨
# 최근접 이웃을 찾을 때 매우 빠름 (O(log n) 수준)
# 데이터가 많거나 반복적으로 거리 계산해야 할 때 효율적
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# --------------------
# 데이터 불러오기
# --------------------
stops_df = pd.read_csv("./data_HWJ/천안시_노선별경유정류소목록.csv")  # 정류소 데이터
facilities_df = pd.read_csv("./data_HWJ/장애인_재활시설.csv")        # 복지시설 데이터

# --------------------
# GeoDataFrame 생성 (WGS84 좌표계)
# --------------------
gdf_stops = gpd.GeoDataFrame(
    stops_df,
    geometry=gpd.points_from_xy(stops_df['정류소 X좌표'], stops_df['정류소 Y좌표']),
    crs="EPSG:4326"
)

gdf_facilities = gpd.GeoDataFrame(
    facilities_df,
    geometry=gpd.points_from_xy(facilities_df['경도'], facilities_df['위도']),
    crs="EPSG:4326"
)

# --------------------
# 투영 좌표계로 변환 (UTM-K : EPSG:5179)
# --------------------
gdf_stops = gdf_stops.to_crs(epsg=5179)
gdf_facilities = gdf_facilities.to_crs(epsg=5179)

# --------------------
# 좌표 배열 추출
# --------------------
stops_coords = np.array(list(zip(gdf_stops.geometry.x, gdf_stops.geometry.y)))
fac_coords = np.array(list(zip(gdf_facilities.geometry.x, gdf_facilities.geometry.y)))

# --------------------
# KDTree 생성 및 최근접 시설 찾기
# --------------------
tree = cKDTree(fac_coords)
dist_m, idx = tree.query(stops_coords, k=1)  # k=1 → 가장 가까운 시설

# --------------------
# 결과 병합
# --------------------
gdf_stops['가장 가까운 복지시설'] = gdf_facilities.iloc[idx]['명 칭'].values
gdf_stops['최단거리_m'] = dist_m  # 이미 미터 단위

# --------------------
# 저장
# --------------------
gdf_stops.drop(columns='geometry').to_csv(
    "정류소별_최단거리복지시설_투영좌표.csv",
    index=False,
    encoding='utf-8-sig'
)

gdf_stops[['정류소명', '가장 가까운 복지시설', '최단거리_m']].head()






import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from scipy.spatial import cKDTree
import plotly.graph_objects as go

# --------------------
# 1. 데이터 불러오기
# --------------------
stops_df = pd.read_csv("./data_HWJ/천안시_노선별경유정류소목록.csv")
facilities_df = pd.read_csv("./data_HWJ/장애인_재활시설.csv")

# --------------------
# 2. GeoDataFrame 생성 (WGS84)
# --------------------
gdf_stops = gpd.GeoDataFrame(
    stops_df,
    geometry=gpd.points_from_xy(stops_df['정류소 X좌표'], stops_df['정류소 Y좌표']),
    crs="EPSG:4326"
)
gdf_facilities = gpd.GeoDataFrame(
    facilities_df,
    geometry=gpd.points_from_xy(facilities_df['경도'], facilities_df['위도']),
    crs="EPSG:4326"
)

# --------------------
# 3. 투영 좌표계 변환 (UTM-K)
# --------------------
gdf_stops = gdf_stops.to_crs(epsg=5179)
gdf_facilities = gdf_facilities.to_crs(epsg=5179)

# --------------------
# 4. KDTree로 최근접 시설 찾기
# --------------------
stops_coords = np.array(list(zip(gdf_stops.geometry.x, gdf_stops.geometry.y)))
fac_coords = np.array(list(zip(gdf_facilities.geometry.x, gdf_facilities.geometry.y)))

tree = cKDTree(fac_coords)
dist_m, idx = tree.query(stops_coords, k=1)

gdf_stops['가장 가까운 복지시설'] = gdf_facilities.iloc[idx]['명 칭'].values
gdf_stops['최단거리_m'] = dist_m

# --------------------
# 5. 연결선(LineString) 생성
# --------------------
lines = []
for i, row in gdf_stops.iterrows():
    facility_point = gdf_facilities.iloc[idx[i]].geometry
    line = LineString([row.geometry, facility_point])
    lines.append(line)

gdf_lines = gpd.GeoDataFrame(geometry=lines, crs="EPSG:5179")

# WGS84로 다시 변환 (Plotly용)
gdf_stops = gdf_stops.to_crs(epsg=4326)
gdf_facilities = gdf_facilities.to_crs(epsg=4326)
gdf_lines = gdf_lines.to_crs(epsg=4326)

# --------------------
# 6. Plotly 시각화
# --------------------
fig = go.Figure()

# 정류소 마커
fig.add_trace(go.Scattermapbox(
    lon=gdf_stops.geometry.x,
    lat=gdf_stops.geometry.y,
    mode='markers',
    name='정류소',
    marker=dict(size=6, color='blue'),
    text=gdf_stops['정류소명']
))

# 복지시설 마커
fig.add_trace(go.Scattermapbox(
    lon=gdf_facilities.geometry.x,
    lat=gdf_facilities.geometry.y,
    mode='markers',
    name='복지시설',
    marker=dict(size=10, color='red', symbol='star'),
    text=gdf_facilities['명 칭']
))

# 연결선
for line in gdf_lines.geometry[:5]:
    x, y = line.xy
    fig.add_trace(go.Scattermapbox(
        lon=list(x),
        lat=list(y),
        mode='lines',
        line=dict(color='green', width=1),
        showlegend=False
    ))

# 레이아웃
fig.update_layout(
    mapbox_style='carto-positron',
    mapbox_zoom=12,
    mapbox_center={"lat": 36.8, "lon": 127.1},
    margin={"r":0,"t":0,"l":0,"b":0},
    legend=dict(title='범례')
)

fig.show()



import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point
import plotly.graph_objects as go

# --------------------
# 1. 데이터 불러오기
# --------------------
stops_df = pd.read_csv("./data_HWJ/천안시_노선별경유정류소목록.csv")  # 정류소
facilities_df = pd.read_csv("./data_HWJ/장애인_재활시설.csv")        # 복지시설

# --------------------
# 2. GeoDataFrame 생성 (WGS84)
# --------------------
gdf_stops = gpd.GeoDataFrame(
    stops_df,
    geometry=gpd.points_from_xy(stops_df['정류소 X좌표'], stops_df['정류소 Y좌표']),
    crs="EPSG:4326"
)
gdf_facilities = gpd.GeoDataFrame(
    facilities_df,
    geometry=gpd.points_from_xy(facilities_df['경도'], facilities_df['위도']),
    crs="EPSG:4326"
)

# --------------------
# 3. 투영 좌표계 변환 (UTM-K: EPSG:5179) - 거리 계산용
# --------------------
gdf_stops_proj = gdf_stops.to_crs(epsg=5179)
gdf_facilities_proj = gdf_facilities.to_crs(epsg=5179)

# --------------------
# 4. KDTree로 최근접 시설 찾기
# --------------------
stops_coords = np.array(list(zip(gdf_stops_proj.geometry.x, gdf_stops_proj.geometry.y)))
fac_coords = np.array(list(zip(gdf_facilities_proj.geometry.x, gdf_facilities_proj.geometry.y)))

tree = cKDTree(fac_coords)
dist_m, idx = tree.query(stops_coords, k=1)

# --------------------
# 5. 결과 병합
# --------------------
gdf_stops['가장 가까운 복지시설'] = gdf_facilities.iloc[idx]['명 칭'].values
gdf_stops['최단거리_m'] = dist_m

# --------------------
# 6. 곡선(LineString) 좌표 생성
# --------------------
def get_curve(lon1, lat1, lon2, lat2, n_points=50, curvature=0.2):
    """출발점-도착점 사이 곡선 좌표 생성"""
    # 직선 중간점
    mid_lon = (lon1 + lon2)/2
    mid_lat = (lat1 + lat2)/2
    # 곡률 적용
    dx = lon2 - lon1
    dy = lat2 - lat1
    ctrl_lon = mid_lon - dy*curvature
    ctrl_lat = mid_lat + dx*curvature

    t = np.linspace(0, 1, n_points)
    # Quadratic Bezier
    lon_curve = (1-t)**2 * lon1 + 2*(1-t)*t*ctrl_lon + t**2 * lon2
    lat_curve = (1-t)**2 * lat1 + 2*(1-t)*t*ctrl_lat + t**2 * lat2
    return lon_curve, lat_curve

# --------------------
# 7. Plotly 시각화
# --------------------
fig = go.Figure()

# 7-1. 정류소 마커
fig.add_trace(go.Scattermapbox(
    lon=gdf_stops.geometry.x,
    lat=gdf_stops.geometry.y,
    mode='markers',
    name='정류소',
    marker = dict(size = 6, color='blue'),
    text=gdf_stops['정류소명']
))

# 7-2. 복지시설 마커
fig.add_trace(go.Scattermapbox(
    lon=gdf_facilities.geometry.x,
    lat=gdf_facilities.geometry.y,
    mode='markers',
    name='복지시설',
    marker=dict(size=10, color='red'),
    text=gdf_facilities['명 칭']
))

# 7-3. 곡선 연결선
for i in range(len(gdf_stops)):
    if i == 100:
        break
    lon0, lat0 = gdf_stops.geometry.x[i], gdf_stops.geometry.y[i]
    facility_point = gdf_facilities.geometry.iloc[idx[i]]
    lon1, lat1 = facility_point.x, facility_point.y
    
    # 곡선 좌표 생성
    x_curve, y_curve = get_curve(lon0, lat0, lon1, lat1)
    
    # 지도에 곡선 추가
    fig.add_trace(go.Scattermapbox(
        lon=x_curve,
        lat=y_curve,
        mode='lines',
        line=dict(color='green', width=2),
        opacity=0.7,
        showlegend=False
    ))

# 7-4. 레이아웃 설정
fig.update_layout(
    mapbox_style='carto-positron',
    mapbox_zoom=12,
    mapbox_center={"lat": 36.8, "lon": 127.1},  # 천안 중심
    margin={"r":0,"t":0,"l":0,"b":0},
    legend=dict(title='범례')
)

fig.show()



import pandas as pd
import pydeck as pdk

# --------------------
# 1. 데이터 불러오기
# --------------------
stops_df = pd.read_csv("./data_HWJ/천안시_노선별경유정류소목록.csv")
facilities_df = pd.read_csv("./data_HWJ/장애인_재활시설.csv")

# --------------------
# 2. 최근접 시설 연결 (KDTree)
# --------------------
import numpy as np
from scipy.spatial import cKDTree

stops_coords = np.array(list(zip(stops_df['정류소 X좌표'], stops_df['정류소 Y좌표'])))
fac_coords = np.array(list(zip(facilities_df['경도'], facilities_df['위도'])))

tree = cKDTree(fac_coords)
dist_m, idx = tree.query(stops_coords, k=1)

# stops_df에 가장 가까운 시설 정보 추가
stops_df['nearest_facility'] = facilities_df.iloc[idx]['명 칭'].values
stops_df['facility_lon'] = facilities_df.iloc[idx]['경도'].values
stops_df['facility_lat'] = facilities_df.iloc[idx]['위도'].values

# --------------------
# 3. ArcLayer용 데이터 생성
# --------------------
arc_data = stops_df[['정류소 X좌표', '정류소 Y좌표', 'facility_lon', 'facility_lat']].copy()
arc_data = arc_data.rename(columns={
    '정류소 X좌표': 'source_lon',
    '정류소 Y좌표': 'source_lat',
    'facility_lon': 'target_lon',
    'facility_lat': 'target_lat'
})

# --------------------
# 4. PyDeck 레이어 정의
# --------------------
# 곡선(Arc) 레이어
arc_layer = pdk.Layer(
    "ArcLayer",
    data=arc_data,
    get_source_position=["source_lon", "source_lat"],
    get_target_position=["target_lon", "target_lat"],
    get_width=2,
    get_source_color=[0, 0, 255],
    get_target_color=[255, 0, 0],
    pickable=True,
)

# 정류소 마커 레이어
stop_layer = pdk.Layer(
    "ScatterplotLayer",
    data=stops_df,
    get_position=["정류소 X좌표", "정류소 Y좌표"],
    get_radius=50,
    get_fill_color=[0, 0, 255],
    pickable=True,
)

# 복지시설 마커 레이어
facility_layer = pdk.Layer(
    "ScatterplotLayer",
    data=facilities_df,
    get_position=["경도", "위도"],
    get_radius=80,
    get_fill_color=[255, 0, 0],
    pickable=True,
)

# --------------------
# 5. PyDeck 뷰 설정
# --------------------
view_state = pdk.ViewState(
    latitude=36.8,
    longitude=127.1,
    zoom=12,
    pitch=0,
)

# --------------------
# 6. PyDeck 지도 생성
# --------------------
pdk.settings.mapbox_api_key = None
r = pdk.Deck(
    layers=[arc_layer, stop_layer, facility_layer],
    initial_view_state=view_state,
    tooltip={"text": "{nearest_facility}"},  # 정류소 마커에 툴팁
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
)

r.show()