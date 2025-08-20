
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point
import plotly.express as px
import numpy as np
import osmnx as ox

# -----------------------------
# 1. 격자 생성 + centroid 계산
# -----------------------------
def create_grid(gdf_city, grid_size=300):
    gdf_city_proj = gdf_city.to_crs(epsg=5179)  # 미터 단위 CRS
    xmin, ymin, xmax, ymax = gdf_city_proj.total_bounds

    grid_cells = [box(x0, y0, x0+grid_size, y0+grid_size)
                  for x0 in range(int(xmin), int(xmax), grid_size)
                  for y0 in range(int(ymin), int(ymax), grid_size)]

    gdf_grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=gdf_city_proj.crs)
    gdf_grid = gpd.overlay(gdf_grid, gdf_city_proj, how='intersection')
    gdf_grid['centroid'] = gdf_grid.geometry.centroid  # centroid 1회 계산
    return gdf_grid

# -----------------------------
# 2. 최근접 시설 거리 계산
# -----------------------------
def compute_nearest_facility(gdf_grid, gdf_facility):
    gdf_facility = gpd.GeoDataFrame(
        gdf_facility,
        geometry=gpd.points_from_xy(gdf_facility['경도'], gdf_facility['위도']),
        crs="EPSG:4326"
    ).to_crs(epsg=5179)

    gdf_grid_proj = gdf_grid.to_crs(epsg=5179)
    
    gdf_grid_proj['min_distance_to_facility'] = gdf_grid_proj['centroid'].apply(
        lambda c: gdf_facility.distance(c).min()
    )

    return gdf_grid_proj.to_crs(epsg=4326)

# -----------------------------
# 3. 저상버스 점수 계산
# -----------------------------
def compute_lowfloor_score(gdf_grid, df_bus_routes, buffer_m=500):
    gdf_routes = gpd.GeoDataFrame(
        df_bus_routes,
        geometry=gpd.points_from_xy(df_bus_routes['X'], df_bus_routes['Y']),
        crs="EPSG:4326"
    ).to_crs(epsg=5179)

    gdf_grid_proj = gdf_grid.to_crs(epsg=5179)

    # buffer 기반 sjoin 벡터화
    gdf_grid_proj['geometry_buffer'] = gdf_grid_proj['centroid'].buffer(buffer_m)
    gdf_grid_buffer = gpd.GeoDataFrame(
        gdf_grid_proj[['geometry_buffer']], geometry='geometry_buffer', crs=gdf_grid_proj.crs
    )

    sjoin_lowfloor = gpd.sjoin(gdf_routes, gdf_grid_buffer, predicate='within')
    lowfloor_score = sjoin_lowfloor.groupby('index_right')['저상버스'].mean()
    gdf_grid_proj['lowfloor_score'] = gdf_grid_proj.index.map(lowfloor_score).fillna(0)

    return gdf_grid_proj.to_crs(epsg=4326)

# -----------------------------
# 4. 노선 수 점수 계산
# -----------------------------
def compute_num_routes_score(gdf_grid, df_bus_routes, buffer_m=500):
    gdf_stops = gpd.GeoDataFrame(
        df_bus_routes,
        geometry=gpd.points_from_xy(df_bus_routes['X'], df_bus_routes['Y']),
        crs="EPSG:4326"
    ).to_crs(epsg=5179)

    gdf_grid_proj = gdf_grid.to_crs(epsg=5179)
    gdf_grid_proj['geometry_buffer'] = gdf_grid_proj['centroid'].buffer(buffer_m)
    gdf_grid_buffer = gpd.GeoDataFrame(
        gdf_grid_proj[['geometry_buffer']], geometry='geometry_buffer', crs=gdf_grid_proj.crs
    )

    sjoin_routes = gpd.sjoin(gdf_stops[['geometry', '노선ID']], gdf_grid_buffer, predicate='within')
    num_routes_score = sjoin_routes.groupby('index_right')['노선ID'].nunique()
    max_routes = df_bus_routes['노선ID'].nunique()
    gdf_grid_proj['num_routes_score'] = gdf_grid_proj.index.map(num_routes_score).fillna(0) / max_routes

    return gdf_grid_proj.to_crs(epsg=4326)


# -----------------------------
# 4-2. 배차간격 점수 계산
# -----------------------------
def compute_headway_score(gdf_grid, df_bus_routes_raw, df_info, buffer_m=500):
    # 정류소 데이터 (geometry 생성)
    gdf_stops = gpd.GeoDataFrame(
        df_bus_routes_raw,
        geometry=gpd.points_from_xy(df_bus_routes_raw['X'], df_bus_routes_raw['Y']),
        crs="EPSG:4326"
    ).to_crs(epsg=5179)

    # 정류소별 평균 배차간격 계산
    stop_headway = df_bus_routes_raw.merge(
        df_info[['노선ID', '배차간격(평일)']], on='노선ID', how='left'
    ).groupby('정류소ID')['배차간격(평일)'].mean()

    gdf_stops['headway'] = gdf_stops['정류소ID'].map(stop_headway)

    # 격자와 매핑
    gdf_grid_proj = gdf_grid.to_crs(epsg=5179)
    gdf_grid_proj['geometry_buffer'] = gdf_grid_proj['centroid'].buffer(buffer_m)
    gdf_grid_buffer = gpd.GeoDataFrame(
        gdf_grid_proj[['geometry_buffer']], geometry='geometry_buffer', crs=gdf_grid_proj.crs
    )

    sjoin_headway = gpd.sjoin(gdf_stops[['geometry', 'headway']], gdf_grid_buffer, predicate='within')
    headway_mean = sjoin_headway.groupby('index_right')['headway'].mean()

    # 점수화 (배차간격이 짧을수록 점수 ↑)
    max_interval = df_info['배차간격(평일)'].max()
    gdf_grid_proj['headway_score'] = 1 - (gdf_grid_proj.index.map(headway_mean) / max_interval)
    gdf_grid_proj['headway_score'] = gdf_grid_proj['headway_score'].fillna(0)

    return gdf_grid_proj.to_crs(epsg=4326)


def compute_vulnerable_density(grid, emd, disabled, older):
    """
    격자 단위 교통약자(장애인+노인) 인구밀도 계산
    
    Parameters
    ----------
    grid : GeoDataFrame
        격자 (geometry, grid_id 포함)
    emd_path : str
        읍면동 경계 geojson 파일 경로
    disabled_csv : str
        장애인 인구 CSV 경로 (ADM_NM, 장애인등록인구 포함)
    older_csv : str
        노인 인구 CSV 경로 (ADM_NM, 65세 이상 인구 포함)

    Returns
    -------
    grid_df : GeoDataFrame
        grid에 교통약자 인구밀도 컬럼이 추가된 GeoDataFrame
    """
    # -----------------------------
    # 1. 데이터 불러오기 및 전처리
    # -----------------------------
    
    def get_grid_id_m(geom, grid_size=300):
        centroid = geom.centroid
        x_id = int(centroid.x // grid_size)
        y_id = int(centroid.y // grid_size)
        return f"{x_id}_{y_id}"

    grid['grid_id'] = grid.geometry.apply(get_grid_id_m)
    emd['ADM_NM'] = emd['ADM_NM'].astype(str).str.strip()

    # 장애인 데이터
    disabled = disabled.rename(columns={"ADM_NM": "DIS_ADM_NM"})
    disabled['DIS_ADM_NM'] = disabled['DIS_ADM_NM'].astype(str).str.strip()
    emd_disabled = emd.merge(disabled, left_on="ADM_NM", right_on="DIS_ADM_NM", how="inner")
    emd_disabled = emd_disabled[['geometry', 'ADM_NM', '장애인등록인구']].copy()
    emd_disabled['geometry'] = emd_disabled.geometry.buffer(0)

    # 노인 데이터
    older = older.rename(columns={'ADM_NM': 'OLD_ADM_NM'})
    older['65세 이상 인구'] = pd.to_numeric(
        older['65세 이상 인구'].astype(str).str.replace(',', '', regex=False),
        errors='coerce'
    )
    emd_older = emd.merge(older, left_on="ADM_NM", right_on="OLD_ADM_NM", how="inner")
    emd_older = emd_older[['geometry', 'ADM_NM', '65세 이상 인구']].copy()
    emd_older['geometry'] = emd_older.geometry.buffer(0)

    # -----------------------------
    # 2. 좌표계 통일
    # -----------------------------
    emd_disabled = emd_disabled.to_crs(epsg=5179)
    emd_older = emd_older.to_crs(epsg=5179)
    grid_clean = grid[['geometry', 'grid_id']].copy().to_crs(epsg=5179)

    # -----------------------------
    # 3. 장애인 인구 밀도
    # -----------------------------
    emd_disabled['읍면동면적'] = emd_disabled.geometry.area
    inter_dis = gpd.overlay(grid_clean, emd_disabled, how='intersection')
    inter_dis['면적'] = inter_dis.geometry.area
    inter_dis['장애인_분배'] = inter_dis['장애인등록인구'] * (inter_dis['면적'] / inter_dis['읍면동면적'])
    dis_by_grid = inter_dis.groupby('grid_id')['장애인_분배'].sum().reset_index()
    dis_by_grid['장애인_인구밀도'] = dis_by_grid['장애인_분배']  # 1km² 가정

    # -----------------------------
    # 4. 노인 인구 밀도
    # -----------------------------
    emd_older['읍면동면적'] = emd_older.geometry.area
    inter_old = gpd.overlay(grid_clean, emd_older, how='intersection')
    inter_old['면적'] = inter_old.geometry.area
    inter_old['노인_분배'] = inter_old['65세 이상 인구'] * (inter_old['면적'] / inter_old['읍면동면적'])
    old_by_grid = inter_old.groupby('grid_id')['노인_분배'].sum().reset_index()
    old_by_grid['노인_인구밀도'] = old_by_grid['노인_분배']

    # -----------------------------
    # 5. 병합 + 합산
    # -----------------------------
    grid_df = grid.copy()
    grid_df = grid_df.merge(dis_by_grid[['grid_id', '장애인_인구밀도']], on='grid_id', how='left')
    grid_df = grid_df.merge(old_by_grid[['grid_id', '노인_인구밀도']], on='grid_id', how='left')

    grid_df['장애인_인구밀도'] = grid_df['장애인_인구밀도'].fillna(0)
    grid_df['노인_인구밀도'] = grid_df['노인_인구밀도'].fillna(0)
    grid_df['교통약자_인구밀도'] = grid_df['장애인_인구밀도'] + grid_df['노인_인구밀도']

    return grid_df




# -----------------------------
# 5. 접근성 점수 계산
# -----------------------------
def compute_accessibility_score(df_grid):
    df_grid['distance_score'] = df_grid['min_distance_to_facility'].apply(
        lambda x: 1 if x <= 400 else (0.5 if x <= 800 else 0)
    )
    base_score = (
        0.25*df_grid['distance_score'] +
        0.25*df_grid['lowfloor_score'] +
        0.25*df_grid['num_routes_score']+
        0.25*df_grid['headway_score']
    )
    
     # 교통약자 인구밀도로 가중치 부여
    max_density = df_grid['교통약자_인구밀도'].max() if df_grid['교통약자_인구밀도'].max() > 0 else 1
    df_grid['accessibility_score'] = base_score * (df_grid['교통약자_인구밀도'] / max_density)

    return df_grid


# -----------------------------
# 6. 지도 시각화
# -----------------------------
def plot_accessibility_plotly(gdf_grid, score_column='accessibility_score'):
    gdf_grid = gdf_grid.to_crs(epsg=4326)
    fig = px.choropleth_map(
        gdf_grid,
        geojson=gdf_grid.geometry,
        locations=gdf_grid.index,
        color=score_column,
        color_continuous_scale="YlOrRd",
        map_style="carto-positron",
        zoom=11,
        center={"lat": gdf_grid.geometry.centroid.y.mean(),
                "lon": gdf_grid.geometry.centroid.x.mean()},
        opacity=0.7,
        hover_data=[score_column]
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, width=1200, height=800)
    return fig



def plot_accessibility_plotly_reverse(gdf_grid, score_column='accessibility_score'):
    gdf_grid = gdf_grid.to_crs(epsg=4326)
    fig = px.choropleth_map(
        gdf_grid,
        geojson=gdf_grid.geometry,
        locations=gdf_grid.index,
        color=score_column,
        color_continuous_scale=px.colors.sequential.YlOrRd[::-1],  # 색상 반전
        map_style="carto-positron",
        zoom=11,
        center={"lat": gdf_grid.geometry.centroid.y.mean(),
                "lon": gdf_grid.geometry.centroid.x.mean()},
        opacity=0.7,
        hover_data=[score_column]
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, width=1200, height=800)
    return fig

# -----------------------------
# 7. 데이터 불러오기 및 전처리
# -----------------------------
df_num = pd.read_csv("./data/천안시_노선번호목록.csv")
df_stops = pd.read_csv("./data/천안시_노선별경유정류소목록.csv")
df_info = pd.read_csv("./data/천안시_노선정보항목.csv", sep='\t')
df_older = pd.read_csv("./data/older.csv")
df_disabled = pd.read_csv("./data/disabled.csv")

emd_path = gpd.read_file("./data/map_data/cheonan_gdf.geojson")


lowfloor_routes = [
    9, 19, 51, 52, 55, 62, 63, 70, 71, 82, 83, 98, 590,
    113, 122, 130, 221, 229, 230, 231, 232, 234, 235, 236,
    237, 238, 251, 270, 310, 402, 510, 512, 520, 540, 614,
    620, 621, 650, 720, 730, 830, 840, 850, 860
]

df_routes = df_info.copy()
df_routes["저상버스"] = (~df_routes["노선번호"].isin(lowfloor_routes)).astype(int)

df_bus_routes_raw = df_stops.merge(
    df_routes[["노선ID", "노선번호", "저상버스"]],
    on="노선ID",
    how="left"
)
df_bus_routes_raw.rename(columns={"정류소 X좌표": "X", "정류소 Y좌표": "Y"}, inplace=True)

df_bus_routes = (
    df_bus_routes_raw.groupby(["정류소ID", "정류소명", "X", "Y"])
    .agg({"저상버스": "mean"})
    .reset_index()
)

gdf_city = gpd.read_file('data/map_data/mapfile.geojson')
gdf_facility = pd.read_csv('data/장애인_재활시설.csv')





# -----------------------------
# 8. 격자 생성 및 점수 계산
# -----------------------------
gdf_grid = create_grid(gdf_city)
gdf_grid = compute_nearest_facility(gdf_grid, gdf_facility)
gdf_grid = compute_lowfloor_score(gdf_grid, df_bus_routes)
gdf_grid = compute_num_routes_score(gdf_grid, df_bus_routes_raw)
gdf_grid = compute_headway_score(gdf_grid, df_bus_routes_raw, df_info)
gdf_grid = compute_vulnerable_density(gdf_grid, emd_path, df_disabled, df_older)
gdf_grid = compute_accessibility_score(gdf_grid)

# -----------------------------
# 9. 지도 시각화
# -----------------------------
map_obj = plot_accessibility_plotly(gdf_grid)
map_obj









### 도로망 없는 경우 회색 표시로..


import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point
import plotly.express as px
import numpy as np
import osmnx as ox

# -----------------------------
# 1. 격자 생성 + centroid 계산
# -----------------------------
def create_grid(gdf_city, grid_size=300):
    gdf_city_proj = gdf_city.to_crs(epsg=5179)  # 미터 단위 CRS
    xmin, ymin, xmax, ymax = gdf_city_proj.total_bounds

    grid_cells = [box(x0, y0, x0+grid_size, y0+grid_size)
                  for x0 in range(int(xmin), int(xmax), grid_size)
                  for y0 in range(int(ymin), int(ymax), grid_size)]

    gdf_grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=gdf_city_proj.crs)
    gdf_grid = gpd.overlay(gdf_grid, gdf_city_proj, how='intersection')
    gdf_grid['centroid'] = gdf_grid.geometry.centroid  # centroid 1회 계산
    return gdf_grid

# -----------------------------
# 2. 최근접 시설 거리 계산
# -----------------------------
def compute_nearest_facility(gdf_grid, gdf_facility):
    gdf_facility = gpd.GeoDataFrame(
        gdf_facility,
        geometry=gpd.points_from_xy(gdf_facility['경도'], gdf_facility['위도']),
        crs="EPSG:4326"
    ).to_crs(epsg=5179)

    gdf_grid_proj = gdf_grid.to_crs(epsg=5179)
    
    gdf_grid_proj['min_distance_to_facility'] = gdf_grid_proj['centroid'].apply(
        lambda c: gdf_facility.distance(c).min()
    )

    return gdf_grid_proj.to_crs(epsg=4326)

# -----------------------------
# 3. 저상버스 점수 계산
# -----------------------------
def compute_lowfloor_score(gdf_grid, df_bus_routes, buffer_m=500):
    gdf_routes = gpd.GeoDataFrame(
        df_bus_routes,
        geometry=gpd.points_from_xy(df_bus_routes['X'], df_bus_routes['Y']),
        crs="EPSG:4326"
    ).to_crs(epsg=5179)

    gdf_grid_proj = gdf_grid.to_crs(epsg=5179)

    # buffer 기반 sjoin 벡터화
    gdf_grid_proj['geometry_buffer'] = gdf_grid_proj['centroid'].buffer(buffer_m)
    gdf_grid_buffer = gpd.GeoDataFrame(
        gdf_grid_proj[['geometry_buffer']], geometry='geometry_buffer', crs=gdf_grid_proj.crs
    )

    sjoin_lowfloor = gpd.sjoin(gdf_routes, gdf_grid_buffer, predicate='within')
    lowfloor_score = sjoin_lowfloor.groupby('index_right')['저상버스'].mean()
    gdf_grid_proj['lowfloor_score'] = gdf_grid_proj.index.map(lowfloor_score).fillna(0)

    return gdf_grid_proj.to_crs(epsg=4326)

# -----------------------------
# 4. 노선 수 점수 계산
# -----------------------------
def compute_num_routes_score(gdf_grid, df_bus_routes, buffer_m=500):
    gdf_stops = gpd.GeoDataFrame(
        df_bus_routes,
        geometry=gpd.points_from_xy(df_bus_routes['X'], df_bus_routes['Y']),
        crs="EPSG:4326"
    ).to_crs(epsg=5179)

    gdf_grid_proj = gdf_grid.to_crs(epsg=5179)
    gdf_grid_proj['geometry_buffer'] = gdf_grid_proj['centroid'].buffer(buffer_m)
    gdf_grid_buffer = gpd.GeoDataFrame(
        gdf_grid_proj[['geometry_buffer']], geometry='geometry_buffer', crs=gdf_grid_proj.crs
    )

    sjoin_routes = gpd.sjoin(gdf_stops[['geometry', '노선ID']], gdf_grid_buffer, predicate='within')
    num_routes_score = sjoin_routes.groupby('index_right')['노선ID'].nunique()
    max_routes = df_bus_routes['노선ID'].nunique()
    gdf_grid_proj['num_routes_score'] = gdf_grid_proj.index.map(num_routes_score).fillna(0) / max_routes

    return gdf_grid_proj.to_crs(epsg=4326)


# -----------------------------
# 4-2. 배차간격 점수 계산
# -----------------------------
def compute_headway_score(gdf_grid, df_bus_routes_raw, df_info, buffer_m=500):
    # 정류소 데이터 (geometry 생성)
    gdf_stops = gpd.GeoDataFrame(
        df_bus_routes_raw,
        geometry=gpd.points_from_xy(df_bus_routes_raw['X'], df_bus_routes_raw['Y']),
        crs="EPSG:4326"
    ).to_crs(epsg=5179)

    # 정류소별 평균 배차간격 계산
    stop_headway = df_bus_routes_raw.merge(
        df_info[['노선ID', '배차간격(평일)']], on='노선ID', how='left'
    ).groupby('정류소ID')['배차간격(평일)'].mean()

    gdf_stops['headway'] = gdf_stops['정류소ID'].map(stop_headway)

    # 격자와 매핑
    gdf_grid_proj = gdf_grid.to_crs(epsg=5179)
    gdf_grid_proj['geometry_buffer'] = gdf_grid_proj['centroid'].buffer(buffer_m)
    gdf_grid_buffer = gpd.GeoDataFrame(
        gdf_grid_proj[['geometry_buffer']], geometry='geometry_buffer', crs=gdf_grid_proj.crs
    )

    sjoin_headway = gpd.sjoin(gdf_stops[['geometry', 'headway']], gdf_grid_buffer, predicate='within')
    headway_mean = sjoin_headway.groupby('index_right')['headway'].mean()

    # 점수화 (배차간격이 짧을수록 점수 ↑)
    max_interval = df_info['배차간격(평일)'].max()
    gdf_grid_proj['headway_score'] = 1 - (gdf_grid_proj.index.map(headway_mean) / max_interval)
    gdf_grid_proj['headway_score'] = gdf_grid_proj['headway_score'].fillna(0)

    return gdf_grid_proj.to_crs(epsg=4326)


def compute_vulnerable_density(grid, emd, disabled, older):
    """
    격자 단위 교통약자(장애인+노인) 인구밀도 계산
    
    Parameters
    ----------
    grid : GeoDataFrame
        격자 (geometry, grid_id 포함)
    emd_path : str
        읍면동 경계 geojson 파일 경로
    disabled_csv : str
        장애인 인구 CSV 경로 (ADM_NM, 장애인등록인구 포함)
    older_csv : str
        노인 인구 CSV 경로 (ADM_NM, 65세 이상 인구 포함)

    Returns
    -------
    grid_df : GeoDataFrame
        grid에 교통약자 인구밀도 컬럼이 추가된 GeoDataFrame
    """
    # -----------------------------
    # 1. 데이터 불러오기 및 전처리
    # -----------------------------
    
    def get_grid_id_m(geom, grid_size=300):
        centroid = geom.centroid
        x_id = int(centroid.x // grid_size)
        y_id = int(centroid.y // grid_size)
        return f"{x_id}_{y_id}"

    grid['grid_id'] = grid.geometry.apply(get_grid_id_m)
    emd['ADM_NM'] = emd['ADM_NM'].astype(str).str.strip()

    # 장애인 데이터
    disabled = disabled.rename(columns={"ADM_NM": "DIS_ADM_NM"})
    disabled['DIS_ADM_NM'] = disabled['DIS_ADM_NM'].astype(str).str.strip()
    emd_disabled = emd.merge(disabled, left_on="ADM_NM", right_on="DIS_ADM_NM", how="inner")
    emd_disabled = emd_disabled[['geometry', 'ADM_NM', '장애인등록인구']].copy()
    emd_disabled['geometry'] = emd_disabled.geometry.buffer(0)

    # 노인 데이터
    older = older.rename(columns={'ADM_NM': 'OLD_ADM_NM'})
    older['65세 이상 인구'] = pd.to_numeric(
        older['65세 이상 인구'].astype(str).str.replace(',', '', regex=False),
        errors='coerce'
    )
    emd_older = emd.merge(older, left_on="ADM_NM", right_on="OLD_ADM_NM", how="inner")
    emd_older = emd_older[['geometry', 'ADM_NM', '65세 이상 인구']].copy()
    emd_older['geometry'] = emd_older.geometry.buffer(0)

    # -----------------------------
    # 2. 좌표계 통일
    # -----------------------------
    emd_disabled = emd_disabled.to_crs(epsg=5179)
    emd_older = emd_older.to_crs(epsg=5179)
    grid_clean = grid[['geometry', 'grid_id']].copy().to_crs(epsg=5179)

    # -----------------------------
    # 3. 장애인 인구 밀도
    # -----------------------------
    emd_disabled['읍면동면적'] = emd_disabled.geometry.area
    inter_dis = gpd.overlay(grid_clean, emd_disabled, how='intersection')
    inter_dis['면적'] = inter_dis.geometry.area
    inter_dis['장애인_분배'] = inter_dis['장애인등록인구'] * (inter_dis['면적'] / inter_dis['읍면동면적'])
    dis_by_grid = inter_dis.groupby('grid_id')['장애인_분배'].sum().reset_index()
    dis_by_grid['장애인_인구밀도'] = dis_by_grid['장애인_분배']  # 1km² 가정

    # -----------------------------
    # 4. 노인 인구 밀도
    # -----------------------------
    emd_older['읍면동면적'] = emd_older.geometry.area
    inter_old = gpd.overlay(grid_clean, emd_older, how='intersection')
    inter_old['면적'] = inter_old.geometry.area
    inter_old['노인_분배'] = inter_old['65세 이상 인구'] * (inter_old['면적'] / inter_old['읍면동면적'])
    old_by_grid = inter_old.groupby('grid_id')['노인_분배'].sum().reset_index()
    old_by_grid['노인_인구밀도'] = old_by_grid['노인_분배']

    # -----------------------------
    # 5. 병합 + 합산
    # -----------------------------
    grid_df = grid.copy()
    grid_df = grid_df.merge(dis_by_grid[['grid_id', '장애인_인구밀도']], on='grid_id', how='left')
    grid_df = grid_df.merge(old_by_grid[['grid_id', '노인_인구밀도']], on='grid_id', how='left')

    grid_df['장애인_인구밀도'] = grid_df['장애인_인구밀도'].fillna(0)
    grid_df['노인_인구밀도'] = grid_df['노인_인구밀도'].fillna(0)
    grid_df['교통약자_인구밀도'] = grid_df['장애인_인구밀도'] + grid_df['노인_인구밀도']

    return grid_df




# -----------------------------
# 5. 접근성 점수 계산
# -----------------------------
def compute_accessibility_score(df_grid):
    df_grid['distance_score'] = df_grid['min_distance_to_facility'].apply(
        lambda x: 1 if x <= 400 else (0.5 if x <= 800 else 0)
    )
    base_score = (
        0.25*df_grid['distance_score'] +
        0.25*df_grid['lowfloor_score'] +
        0.25*df_grid['num_routes_score']+
        0.25*df_grid['headway_score']
    )
    
     # 교통약자 인구밀도로 가중치 부여
    max_density = df_grid['교통약자_인구밀도'].max() if df_grid['교통약자_인구밀도'].max() > 0 else 1
    df_grid['accessibility_score'] = base_score * (df_grid['교통약자_인구밀도'] / max_density)

    return df_grid

def filter_grid_by_roads(gdf_grid, gdf_roads):
    if gdf_roads is None or len(gdf_roads) == 0:
        gdf_grid['has_road'] = True
        return gdf_grid
    
    gdf_grid_proj = gdf_grid.to_crs(epsg=5179).reset_index()  # reset_index 추가
    gdf_roads_proj = gdf_roads.to_crs(epsg=5179)

    # 격자와 도로가 겹치는지 확인
    sjoin = gpd.sjoin(gdf_grid_proj, gdf_roads_proj, predicate='intersects', how='left')

    # index 컬럼 사용
    has_road_idx = sjoin.loc[~sjoin['index_right'].isna(), 'index'].unique()
    
    # 도로가 있는 격자만 선택
    gdf_grid_filtered = gdf_grid_proj[gdf_grid_proj['index'].isin(has_road_idx)].copy()
    return gdf_grid_filtered.to_crs(epsg=4326)



# -----------------------------
# 6. 지도 시각화
# -----------------------------
def plot_accessibility_plotly(gdf_grid, score_column='accessibility_score'):
    gdf_grid = gdf_grid.to_crs(epsg=4326)
    fig = px.choropleth_map(
        gdf_grid,
        geojson=gdf_grid.geometry,
        locations=gdf_grid.index,
        color=score_column,
        color_continuous_scale="YlOrRd",
        map_style="carto-positron",
        zoom=11,
        center={"lat": gdf_grid.geometry.centroid.y.mean(),
                "lon": gdf_grid.geometry.centroid.x.mean()},
        opacity=0.7,
        hover_data=[score_column]
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, width=1200, height=800)
    return fig

def plot_accessibility_with_gray(gdf_grid, score_column='accessibility_score'):
    gdf_grid = gdf_grid.to_crs(epsg=4326).copy()
    
    # 도로 있는 격자 / 없는 격자 나누기
    gdf_road = gdf_grid[gdf_grid['has_road']==True]
    gdf_no_road = gdf_grid[gdf_grid['has_road']==False]
    
    # 지도 그리기
    fig = px.choropleth_map(
        gdf_road,
        geojson=gdf_road.geometry,
        locations=gdf_road.index,
        color=score_column,
        color_continuous_scale="YlOrRd",
        map_style="carto-positron",
        zoom=11,
        center={"lat": gdf_grid.geometry.centroid.y.mean(),
                "lon": gdf_grid.geometry.centroid.x.mean()},
        opacity=0.7,
        hover_data=[score_column]
    )
    
    # 도로 없는 격자는 회색으로 오버레이
    fig.add_trace(px.choropleth_map(
        gdf_no_road,
        geojson=gdf_no_road.geometry,
        locations=gdf_no_road.index,
        color_discrete_sequence=['lightgray'],
        hover_data=[score_column]
    ).data[0])
    
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, width=1200, height=800)
    
    return fig


def plot_accessibility_plotly_reverse(gdf_grid, score_column='accessibility_score'):
    gdf_grid = gdf_grid.to_crs(epsg=4326)
    fig = px.choropleth_map(
        gdf_grid,
        geojson=gdf_grid.geometry,
        locations=gdf_grid.index,
        color=score_column,
        color_continuous_scale=px.colors.sequential.YlOrRd[::-1],  # 색상 반전
        map_style="carto-positron",
        zoom=11,
        center={"lat": gdf_grid.geometry.centroid.y.mean(),
                "lon": gdf_grid.geometry.centroid.x.mean()},
        opacity=0.7,
        hover_data=[score_column]
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, width=1200, height=800)
    return fig

# -----------------------------
# 7. 데이터 불러오기 및 전처리
# -----------------------------
df_num = pd.read_csv("./data/천안시_노선번호목록.csv")
df_stops = pd.read_csv("./data/천안시_노선별경유정류소목록.csv")
df_info = pd.read_csv("./data/천안시_노선정보항목.csv", sep='\t')
df_older = pd.read_csv("./data/older.csv")
df_disabled = pd.read_csv("./data/disabled.csv")

emd_path = gpd.read_file("./data/map_data/cheonan_gdf.geojson")
gdf_roads = gpd.read_file("cheonan_roads.geojson")


lowfloor_routes = [
    9, 19, 51, 52, 55, 62, 63, 70, 71, 82, 83, 98, 590,
    113, 122, 130, 221, 229, 230, 231, 232, 234, 235, 236,
    237, 238, 251, 270, 310, 402, 510, 512, 520, 540, 614,
    620, 621, 650, 720, 730, 830, 840, 850, 860
]

df_routes = df_info.copy()
df_routes["저상버스"] = (~df_routes["노선번호"].isin(lowfloor_routes)).astype(int)

df_bus_routes_raw = df_stops.merge(
    df_routes[["노선ID", "노선번호", "저상버스"]],
    on="노선ID",
    how="left"
)
df_bus_routes_raw.rename(columns={"정류소 X좌표": "X", "정류소 Y좌표": "Y"}, inplace=True)

df_bus_routes = (
    df_bus_routes_raw.groupby(["정류소ID", "정류소명", "X", "Y"])
    .agg({"저상버스": "mean"})
    .reset_index()
)

gdf_city = gpd.read_file('data/map_data/mapfile.geojson')
gdf_facility = pd.read_csv('data/장애인_재활시설.csv')





# -----------------------------
# 8. 격자 생성 및 점수 계산
# -----------------------------
gdf_grid = create_grid(gdf_city)
gdf_grid = compute_nearest_facility(gdf_grid, gdf_facility)
gdf_grid = compute_lowfloor_score(gdf_grid, df_bus_routes)
gdf_grid = compute_num_routes_score(gdf_grid, df_bus_routes_raw)
gdf_grid = compute_headway_score(gdf_grid, df_bus_routes_raw, df_info)
gdf_grid = compute_vulnerable_density(gdf_grid, emd_path, df_disabled, df_older)
gdf_grid = compute_accessibility_score(gdf_grid)
# 도로망 있는 격자만 필터링
gdf_grid_filtered = filter_grid_by_roads(gdf_grid, gdf_roads)
# -----------------------------
# 9. 지도 시각화
# -----------------------------
map_obj = plot_accessibility_with_gray(gdf_grid_filtered)
map_obj