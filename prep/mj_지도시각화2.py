import json
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
from shapely.geometry import Point, LineString

def add_facility_trace(fig, df, lat_col, lon_col, name, color, text_col, address_col=None):
    if address_col and address_col in df.columns:
        text = df[text_col] + "<br>" + df[address_col]
    else:
        text = df[text_col]

    fig.add_scattermapbox(
        lat=df[lat_col],
        lon=df[lon_col],
        mode='markers',
        marker=dict(size=10, color=color),
        text=text,
        hoverinfo='text',
        name=name,
        visible='legendonly'  # 기본 토글 OFF 상태
    )


# 1. 천안시 경계 GeoDataFrame 로드
with open('./data_MJ/cheonan_gdf.geojson', encoding='utf-8') as f:
    geojson_data = json.load(f)
gdf_boundary = gpd.GeoDataFrame.from_features(geojson_data, crs='EPSG:4326')
boundary_union = gdf_boundary.union_all()

# 2. 정류소 데이터 불러오기 및 GeoDataFrame 생성
df = pd.read_csv('./data_HWJ/천안시_노선별경유정류소목록.csv')
df_sorted = df.sort_values(by=['노선ID', '정류소순번'])
stops = gpd.GeoDataFrame(
    df_sorted,
    geometry=gpd.points_from_xy(df_sorted['정류소 X좌표'], df_sorted['정류소 Y좌표']),
    crs='EPSG:4326'
)

# 3. OSM 기반 드라이브 도로망 다운로드
bbox_vals = gdf_boundary.total_bounds  # [minx, miny, maxx, maxy] → [west, south, east, north]
west, south, east, north = bbox_vals

# 2) 튜플 형태의 bbox 정의
bbox_tuple = (west, south, east, north)

# 3) OSMnx 그래프 생성 (최신 방식)
G = ox.graph.graph_from_bbox(bbox=bbox_tuple, network_type='drive')
nodes_proj, edges_proj = ox.graph_to_gdfs(G)

# 4. 정류소를 도로망 노드에 스냅
stops['nearest_node'] = ox.distance.nearest_nodes(G, stops.geometry.x, stops.geometry.y)

# 5. 노선별 실제 도로 경로 계산
routes = []
for route_id, grp in stops.groupby('노선ID'):
    grp = grp.sort_values('정류소순번')
    nodes = grp['nearest_node'].tolist()
    full_path = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        try:
            path = nx.shortest_path(G, u, v, weight='length')
            full_path.extend(path)
        except (nx.NetworkXNoPath, KeyError):
            continue
    routes.append((route_id, full_path))

# 6. 천안시 경계 밖 노선 제거
filtered_routes = []

for route_id, path in routes:
    coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
    points = [Point(x, y) for x, y in coords]
    
    # 노선 경로상의 모든 점이 경계 내에 포함되는지 검사
    if all(point.within(boundary_union) for point in points):
        filtered_routes.append((route_id, path))

# 7. 복지시설 및 경계 시각화 + 노선 추가
fig = go.Figure()

# 경계 표시
for polygon in gdf_boundary.geometry:
    if polygon.geom_type == 'Polygon':
        lon, lat = polygon.exterior.xy
        fig.add_trace(go.Scattermapbox(
            lon=list(lon),
            lat=list(lat),
            mode='lines',
            fill='toself',
            fillcolor='rgba(0,0,0,0.1)',
            line=dict(color='gray', width=2),
            name='천안시 경계',
            visible=True,
            showlegend=False
        ))
    elif polygon.geom_type == 'MultiPolygon':
        for poly in polygon.geoms:
            lon, lat = poly.exterior.xy
            fig.add_trace(go.Scattermapbox(
                lon=list(lon),
                lat=list(lat),
                mode='lines',
                fill='toself',
                fillcolor='rgba(0,0,0,0.1)',
                line=dict(color='gray', width=2),
                name='천안시 경계',
                visible=True,
                showlegend=False
            ))

# 8. 노선 시각화 (검은색)
bus_routes=pd.read_csv('./data_HWJ/천안시_노선정보항목.csv',sep='\t')


for route_id, path in filtered_routes:
    coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
    if not coords:
        continue
    lon, lat = zip(*coords)
    fig.add_trace(go.Scattermapbox(
        lon=lon, lat=lat,
        mode='lines+markers',
        line=dict(color='black', width=1),
        marker=dict(size=5),
        name=f'노선 {route_id}',
        visible=True,
        showlegend=False
    ))
    # 노선번호 찾아서 hover text에 추가
    bus = bus_routes[bus_routes['노선ID'] == route_id]
    if not bus.empty:
        bus_number = bus.iloc[0]['노선번호']
        hover_text = f'노선ID: {route_id}<br>버스번호: {bus_number}'
    else:
        hover_text = f'노선ID: {route_id}'
    
    fig.add_trace(go.Scattermapbox(
        lon=lon, lat=lat,
        mode='lines+markers',
        line=dict(color='black', width=1),
        marker=dict(size=5),
        name=f'노선 {route_id}',
        text=[hover_text] * len(lon),  # 각 점에 hover 텍스트 반복 적용
        hoverinfo='text',
        visible=True,
        showlegend=False
    ))

# 9. 복지시설 표시
# 응급의료기관
df_hospital_e = pd.read_csv('./data_MJ/응급의료기관.csv', encoding='cp949')
em_hospital = df_hospital_e[df_hospital_e['주소'].str.contains('천안')]
fig.add_scattermapbox(lat=em_hospital['병원위도'], lon=em_hospital['병원경도'],
                      mode='markers', marker=dict(size=10, color='red'),
                      text=em_hospital['기관명'] + "<br>" + em_hospital['주소'],
                      hoverinfo='text', name='응급의료기관')
# 보건기관
df_health = pd.read_csv('./data_MJ/보건기관.csv', encoding='utf-8-sig')
fig.add_scattermapbox(lat=df_health['위도'], lon=df_health['경도'],
                      mode='markers', marker=dict(size=10, color='green'),
                      text=df_health['보건기관명'] + "<br>" + df_health['주소'],
                      hoverinfo='text', name='보건기관')
# 장애인 재활시설
df_fac = pd.read_csv('./data_MJ/장애인_재활시설.csv')
fig.add_scattermapbox(lat=df_fac['위도'], lon=df_fac['경도'],
                      mode='markers', marker=dict(size=10, color='blue'),
                      text=df_fac['명 칭'] + "<br>" + df_fac['소 재 지'],
                      hoverinfo='text', name='장애인 재활시설')
# 노인 복지시설
df_welf = pd.read_csv('./data_MJ/노인_복지시설.csv')
fig.add_scattermapbox(lat=df_welf['위도'], lon=df_welf['경도'],
                      mode='markers', marker=dict(size=10, color='yellow'),
                      text=df_welf['기관명'] + "<br>" + df_welf['주소'],
                      hoverinfo='text', name='노인 복지시설')



# 10. 레이아웃 설정
fig.update_layout(
    mapbox_style='carto-positron',
    mapbox_center={"lat": 36.8, "lon": 127.1},
    mapbox_zoom=9.5,
    margin=dict(r=0, t=0, l=0, b=0),
    legend_title_text='범례',
    title='천안시 복지시설 및 실도로망 기반 버스 노선 (검정색)'
)

fig.show()

fig.write_html('map_bus.html')

