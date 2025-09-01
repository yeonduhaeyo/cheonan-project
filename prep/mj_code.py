import pandas as pd
import geopandas as gpd
import plotly.express as px
import json

def make_disabled_map():
    # 1. 지도 및 데이터 불러오기
    
    with open("./data_MJ\cheonan_gdf.geojson", encoding='utf-8') as f:
        geojson_data = json.load(f)

    disabled = pd.read_csv("./data_MJ/disabled.csv")

    # 2. 시각화
    fig = px.choropleth_mapbox(
        disabled,
        geojson=geojson_data,
        locations="ADM_NM",
        featureidkey="properties.ADM_NM",
        color='장애인 인구 비율',
        color_continuous_scale="Blues",
        mapbox_style="carto-positron",
        center = {"lat": 36.81, "lon": 127.21},
        zoom=9.5,
        opacity=0.7,
        hover_name="ADM_NM",
        height=600
    )

    # 3. 시설들 마커 추가
    df_facilities = pd.read_csv("./data_MJ/장애인_재활시설.csv")
    fig.add_scattermapbox(
        lat=df_facilities['위도'],
        lon=df_facilities['경도'],
        mode='markers',
        marker=dict(size=10, color='blue'),
        text=df_facilities['명 칭'] + "<br>" + df_facilities['소 재 지'],
        hoverinfo='text',
        name='장애인 재활시설'
    )

    df_hospital = pd.read_csv("./data_MJ/응급의료기관.csv", encoding='utf-8-sig')
    em_hospital = df_hospital[df_hospital['주소'].str.contains('천안')]
    fig.add_scattermapbox(
        lat=em_hospital['병원위도'],
        lon=em_hospital['병원경도'],
        mode='markers',
        marker=dict(size=10, color='red'),
        text=em_hospital['기관명'] + "<br>" + em_hospital['주소'],
        hoverinfo='text',
        name='응급의료기관'
    )

    df_healthcenter = pd.read_csv("./data_MJ/보건기관.csv")
    fig.add_scattermapbox(
        lat=df_healthcenter['위도'],
        lon=df_healthcenter['경도'],
        mode='markers',
        marker=dict(size=10, color='green'),
        text=df_healthcenter['보건기관명'] + "<br>" + df_healthcenter['주소'],
        hoverinfo='text',
        name='보건기관'
    )

    fig.update_layout(
        autosize=True,
        height=600,
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=13)
        )
    )
    return fig





import pandas as pd
import json
import geopandas as gpd
import plotly.express as px
import re

def make_older_map():
    # 1. 지도 및 데이터 불러오기
    
    with open("./data_MJ/cheonan_gdf.geojson", encoding='utf-8') as f:
        geojson_data = json.load(f)

    older = pd.read_csv("./data_MJ/older.csv")

# 2. 시각화
    fig = px.choropleth_mapbox(
        older,  # ✅ GeoDataFrame 사용
        geojson=geojson_data,
        locations="ADM_NM",  # ✅ 실제 GeoJSON 속성과 연결되는 컬럼
        featureidkey="properties.ADM_NM",  # ✅ GeoJSON 안에 있는 키 경로
        color='노인 인구 비율',
        color_continuous_scale="Greens",
        range_color=(older['노인 인구 비율'].min(), older['노인 인구 비율'].max()),
        mapbox_style="carto-positron",
        center = {"lat": 36.81, "lon": 127.21},
        zoom=9.5,
        opacity=0.5,
        hover_name="ADM_NM",
        height=600
    )
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0}) 

# 3. [노인 복지시설]
    df_welfare_facilities = pd.read_csv("data_MJ\노인_복지시설.csv",encoding='utf-8-sig')
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

# 4. [응급의료기관]
    df_hospital_e = pd.read_csv("./data_MJ/응급의료기관.csv",encoding='utf-8-sig')
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
# 5.[보건기관]
    df_healthcenter = pd.read_csv("./data_MJ/보건기관.csv")
    # df_healthcenter.columns
    fig.add_scattermapbox(
        lat=df_healthcenter['위도'],
        lon=df_healthcenter['경도'],
        mode='markers',
        marker=dict(size=10, color='green'),
        text=df_healthcenter['보건기관명'] + "<br>" + df_healthcenter['주소'],
        hoverinfo='text',
        name='보건기관'
    )
    
    # 6. 레이아웃
    fig.update_layout(
        autosize=True,
        height=600,
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        legend=dict(
            orientation="h",     # 수평 정렬
            yanchor="top",
            y=-0.1,              # 아래쪽으로 이동
            xanchor="center",
            x=0.6,
            font=dict(size=13)
        )
    )
    return fig







# [기존 노선 + 복지시설 위치 ]
import json
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
from shapely.geometry import Point


def load_map_figure(selected_facilities):
    # --- 1. 경계 ---
    with open('./data_MJ/cheonan_gdf.geojson', encoding='utf-8') as f:
        geojson_data = json.load(f)
    gdf_boundary = gpd.GeoDataFrame.from_features(geojson_data, crs='EPSG:4326')
    boundary_union = gdf_boundary.unary_union

    # --- 2. 정류소 ---
    df = pd.read_csv('./data_HWJ/천안시_노선별경유정류소목록.csv')
    df = df.sort_values(by=['노선ID', '정류소순번'])
    stops = gpd.GeoDataFrame(df,
        geometry=gpd.points_from_xy(df['정류소 X좌표'], df['정류소 Y좌표']),
        crs='EPSG:4326')

    # --- 3. OSM 도로망 ---
    bbox_vals = gdf_boundary.total_bounds  # [minx, miny, maxx, maxy] → [west, south, east, north]
    west, south, east, north = bbox_vals
    # 2) 튜플 형태의 bbox 정의
    bbox_tuple = (west, south, east, north)
    # 3) OSMnx 그래프 생성 (최신 방식)
    G = ox.graph.graph_from_bbox(bbox=bbox_tuple, network_type='drive')
    stops['nearest_node'] = ox.distance.nearest_nodes(G, stops.geometry.x, stops.geometry.y)

    # --- 4. 노선 경로 계산 ---
    routes = []
    for route_id, grp in stops.groupby('노선ID'):
        grp = grp.sort_values('정류소순번')
        nodes = grp['nearest_node'].tolist()
        full_path = []
        for u, v in zip(nodes[:-1], nodes[1:]):
            try:
                path = nx.shortest_path(G, u, v, weight='length')
                full_path.extend(path)
            except:
                continue
        routes.append((route_id, full_path))

    # --- 5. 경계 내 노선 필터 ---
    filtered_routes = []
    for route_id, path in routes:
        coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
        if all(Point(x, y).within(boundary_union) for x, y in coords):
            filtered_routes.append((route_id, path))

    bus_info = pd.read_csv('./data_HWJ/천안시_노선정보항목.csv', sep='\t')

    # --- 6. 시각화 시작 ---
    fig = go.Figure()

    # 천안시 경계
    for poly in gdf_boundary.geometry:
        if poly.geom_type == 'Polygon':
            lon, lat = poly.exterior.xy
            fig.add_trace(go.Scattermapbox(
                lon=list(lon), lat=list(lat),
                mode='lines', fill='toself',
                fillcolor='rgba(0,0,0,0.1)',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
        elif poly.geom_type == 'MultiPolygon':
            for sub_poly in poly.geoms:
                lon, lat = sub_poly.exterior.xy
                fig.add_trace(go.Scattermapbox(
                    lon=list(lon), lat=list(lat),
                    mode='lines', fill='toself',
                    fillcolor='rgba(0,0,0,0.1)',
                    line=dict(color='gray', width=2),
                    showlegend=False
                ))

    # 버스 노선
    for route_id, path in filtered_routes:
        coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
        if not coords:
            continue
        lon, lat = zip(*coords)
        bus = bus_info[bus_info['노선ID'] == route_id]
        bus_number = bus.iloc[0]['노선번호'] if not bus.empty else 'N/A'
        hover_text = f'노선ID: {route_id}<br>버스번호: {bus_number}'

        fig.add_trace(go.Scattermapbox(
            lon=lon, lat=lat,
            mode='lines+markers',
            line=dict(color='black', width=1),
            marker=dict(size=5),
            name=f'노선 {route_id}',
            text=[hover_text] * len(lon),
            hoverinfo='text',
            showlegend=False
        ))

    # --- 복지시설 ---
    facility_data = {
        '응급의료기관': {
            'df': pd.read_csv('./data_MJ/응급의료기관.csv', encoding='utf-8-sig'),
            'lat': '병원위도', 'lon': '병원경도',
            'text': '기관명', 'addr': '주소', 'color': 'red'
        },
        '보건기관': {
            'df': pd.read_csv('./data_MJ/보건기관.csv', encoding='utf-8-sig'),
            'lat': '위도', 'lon': '경도',
            'text': '보건기관명', 'addr': '주소', 'color': 'green'
        },
        '장애인 재활시설': {
            'df': pd.read_csv('./data_MJ/장애인_재활시설.csv'),
            'lat': '위도', 'lon': '경도',
            'text': '명 칭', 'addr': '소 재 지', 'color': 'blue'
        },
        '노인 복지시설': {
            'df': pd.read_csv('data_MJ\노인_복지시설.csv',encoding='utf-8-sig'),
            'lat': '위도', 'lon': '경도',
            'text': '기관명', 'addr': '주소', 'color': 'yellow'
        }
    }

    for name, info in facility_data.items():
        if name in selected_facilities:
            df = info['df']
            text = df[info['text']] + "<br>" + df[info['addr']]
            fig.add_trace(go.Scattermapbox(
                lat=df[info['lat']],
                lon=df[info['lon']],
                mode='markers',
                marker=dict(size=10, color=info['color']),
                text=text,
                hoverinfo='text',
                name=name
            ))

    # 레이아웃
    fig.update_layout(
        mapbox_style='carto-positron',
        mapbox_center={"lat": 36.81, "lon": 127.21},
        mapbox_zoom=9.5,
        margin=dict(r=0, t=0, l=0, b=0),
        legend_title_text='복지시설',
        title='천안시 복지시설 및 실도로망 기반 버스 노선 (검정색)'
    )

    return fig