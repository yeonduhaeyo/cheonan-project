# viz/score_map.py
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box, mapping
import plotly.graph_objects as go
import plotly.express as px  # (다른 곳에서 쓰면 유지, 안 쓰면 지워도 됨)

from shared import (
    gdf_cheonan, gdf_roads, df_facility,
    df_bus_routes, df_bus_routes_raw, df_info,
    df_disabled, df_older
)

# -----------------------------
# 1. 격자 생성
# -----------------------------
def create_grid(gdf_city, grid_size=300):
    gdf_city_proj = gdf_city.to_crs(5179)
    xmin, ymin, xmax, ymax = gdf_city_proj.total_bounds
    cells = [box(x0, y0, x0+grid_size, y0+grid_size)
             for x0 in range(int(xmin), int(xmax), grid_size)
             for y0 in range(int(ymin), int(ymax), grid_size)]
    grid = gpd.GeoDataFrame({"geometry": cells}, crs=gdf_city_proj.crs)
    grid = gpd.overlay(grid, gdf_city_proj, how="intersection")
    grid["centroid"] = grid.geometry.centroid
    return grid


def _to_points_5179(df, lon_col='경도', lat_col='위도'):
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=4326
    ).to_crs(5179)
    return gdf

def compute_nearest_facility_multi(gdf_grid, fac_all, type_col='구분', weights=None, score_fn=None):
    grid_5179 = gdf_grid.to_crs(5179).copy()
    centroids = grid_5179['centroid']

    types = fac_all[type_col].unique().tolist()
    for t in types:
        fac_t = fac_all[fac_all[type_col] == t]
        if fac_t.empty:
            grid_5179[f'dist_{t}'] = np.nan
            continue
        gdf_fac_t = _to_points_5179(fac_t)
        dists = centroids.apply(lambda c: gdf_fac_t.distance(c).min() if len(gdf_fac_t) else np.nan)
        grid_5179[f'dist_{t}'] = dists.values

    def default_score_fn(d):
        if pd.isna(d): return 0.0
        if d <= 400:  return 1.0
        if d <= 800:  return 0.5
        return 0.0

    score_fn = score_fn or default_score_fn
    for t in types:
        grid_5179[f'score_{t}'] = grid_5179[f'dist_{t}'].apply(score_fn)

    if weights is None:
        weights = {t: 1.0/len(types) for t in types}
    wsum = sum(weights.values()) or 1.0
    weights = {t: v/wsum for t, v in weights.items()}

    grid_5179['facility_score_weighted'] = 0.0
    for t in types:
        grid_5179['facility_score_weighted'] += weights[t] * grid_5179[f'score_{t}']

    return grid_5179.to_crs(4326)

def compute_lowfloor_score(gdf_grid, df_bus_routes, buffer_m=500):
    gdf_routes = gpd.GeoDataFrame(
        df_bus_routes,
        geometry=gpd.points_from_xy(df_bus_routes['X'], df_bus_routes['Y']),
        crs=4326
    ).to_crs(5179)

    grid = gdf_grid.to_crs(5179)
    grid['geometry_buffer'] = grid['centroid'].buffer(buffer_m)
    grid_buf = grid.set_geometry('geometry_buffer')
    sjoin = gpd.sjoin(gdf_routes, grid_buf, predicate='within')
    score = sjoin.groupby('index_right')['저상버스'].mean()
    grid['lowfloor_score'] = grid.index.map(score).fillna(0)
    return grid.drop(columns=['geometry_buffer']).to_crs(4326)

def compute_num_routes_score(gdf_grid, df_bus_routes_raw, buffer_m=500):
    gdf_stops = gpd.GeoDataFrame(
        df_bus_routes_raw,
        geometry=gpd.points_from_xy(df_bus_routes_raw['X'], df_bus_routes_raw['Y']),
        crs=4326
    ).to_crs(5179)

    grid = gdf_grid.to_crs(5179)
    grid['geometry_buffer'] = grid['centroid'].buffer(buffer_m)
    grid_buf = grid.set_geometry('geometry_buffer')

    sjoin = gpd.sjoin(gdf_stops[['geometry', '노선ID']], grid_buf, predicate='within')
    nuniq = sjoin.groupby('index_right')['노선ID'].nunique()
    max_routes = df_bus_routes_raw['노선ID'].nunique()
    grid['num_routes_score'] = grid.index.map(nuniq).fillna(0) / max_routes
    return grid.drop(columns=['geometry_buffer']).to_crs(4326)

def compute_headway_score(gdf_grid, df_bus_routes_raw, df_info, buffer_m=500):
    gdf_stops = gpd.GeoDataFrame(
        df_bus_routes_raw,
        geometry=gpd.points_from_xy(df_bus_routes_raw['X'], df_bus_routes_raw['Y']),
        crs=4326
    ).to_crs(5179)

    stop_headway = df_bus_routes_raw.merge(
        df_info[['노선ID', '배차간격(평일)']], on='노선ID', how='left'
    ).groupby('정류소ID')['배차간격(평일)'].mean()
    gdf_stops['headway'] = gdf_stops['정류소ID'].map(stop_headway)

    grid = gdf_grid.to_crs(5179)
    grid['geometry_buffer'] = grid['centroid'].buffer(buffer_m)
    grid_buf = grid.set_geometry('geometry_buffer')

    sjoin = gpd.sjoin(gdf_stops[['geometry', 'headway']], grid_buf, predicate='within')
    mean_hw = sjoin.groupby('index_right')['headway'].mean()
    max_interval = df_info['배차간격(평일)'].max()
    grid['headway_score'] = 1 - (grid.index.map(mean_hw) / max_interval)
    grid['headway_score'] = grid['headway_score'].fillna(0)
    return grid.drop(columns=['geometry_buffer']).to_crs(4326)

def compute_vulnerable_density(grid, emd, disabled, older, grid_size=300):
    def gid(geom):
        c = geom.centroid
        return f"{int(c.x // grid_size)}_{int(c.y // grid_size)}"

    grid['grid_id'] = grid.geometry.apply(gid)
    emd['ADM_NM'] = emd['ADM_NM'].astype(str).str.strip()

    disabled = disabled.rename(columns={"ADM_NM": "DIS_ADM_NM"})
    disabled['DIS_ADM_NM'] = disabled['DIS_ADM_NM'].astype(str).str.strip()
    emd_dis = emd.merge(disabled, left_on="ADM_NM", right_on="DIS_ADM_NM", how="inner")
    emd_dis = emd_dis[['geometry', 'ADM_NM', '장애인등록인구']].copy()

    older = older.rename(columns={'ADM_NM': 'OLD_ADM_NM'})
    older['65세 이상 인구'] = pd.to_numeric(older['65세 이상 인구'].astype(str).str.replace(',', ''), errors='coerce')
    emd_old = emd.merge(older, left_on="ADM_NM", right_on="OLD_ADM_NM", how="inner")
    emd_old = emd_old[['geometry', 'ADM_NM', '65세 이상 인구']].copy()

    emd_dis = emd_dis.to_crs(5179)
    emd_old = emd_old.to_crs(5179)
    grid_clean = grid[['geometry', 'grid_id']].copy().to_crs(5179)

    emd_dis['읍면동면적'] = emd_dis.geometry.area
    i_dis = gpd.overlay(grid_clean, emd_dis, how='intersection')
    i_dis['면적'] = i_dis.geometry.area
    i_dis['장애인_분배'] = i_dis['장애인등록인구'] * (i_dis['면적'] / i_dis['읍면동면적'])
    dis_by_grid = i_dis.groupby('grid_id')['장애인_분배'].sum().reset_index()
    dis_by_grid['장애인_인구밀도'] = dis_by_grid['장애인_분배']

    emd_old['읍면동면적'] = emd_old.geometry.area
    i_old = gpd.overlay(grid_clean, emd_old, how='intersection')
    i_old['면적'] = i_old.geometry.area
    i_old['노인_분배'] = i_old['65세 이상 인구'] * (i_old['면적'] / i_old['읍면동면적'])
    old_by_grid = i_old.groupby('grid_id')['노인_분배'].sum().reset_index()
    old_by_grid['노인_인구밀도'] = old_by_grid['노인_분배']

    out = grid.copy()
    out = out.merge(dis_by_grid[['grid_id', '장애인_인구밀도']], on='grid_id', how='left')
    out = out.merge(old_by_grid[['grid_id', '노인_인구밀도']], on='grid_id', how='left')
    out['장애인_인구밀도'] = out['장애인_인구밀도'].fillna(0)
    out['노인_인구밀도'] = out['노인_인구밀도'].fillna(0)
    out['교통약자_인구밀도'] = out['장애인_인구밀도'] + out['노인_인구밀도']
    return out

def flag_has_road(gdf_grid, gdf_roads):
    """
    격자와 도로가 'intersects'하면 has_road=True.
    결과 CRS는 4326으로 돌려줌.
    """
    grid = gdf_grid.to_crs(5179).copy().reset_index(drop=True)

    if gdf_roads is None or len(gdf_roads) == 0:
        grid['has_road'] = True
        return grid.to_crs(4326)

    roads = gdf_roads.to_crs(5179)[['geometry']].copy()

    # 격자(왼쪽)와 도로(오른쪽) 공간조인 (겹치는 것만 True)
    joined = gpd.sjoin(grid[['geometry']], roads, how='left', predicate='intersects')

    # 도로와 한 번이라도 겹친 격자의 left-index 집합
    idx_with_road = joined.index[~joined['index_right'].isna()].unique()

    grid['has_road'] = grid.index.isin(idx_with_road)
    return grid.to_crs(4326)

# -----------------------------
# 7. 접근성 점수 통합
# -----------------------------
def compute_accessibility_score(df_grid, weights):
    base = (
        weights[0]*df_grid['facility_score_weighted'] +
        weights[1]*df_grid['lowfloor_score'] +
        weights[2]*df_grid['num_routes_score'] +
        weights[3]*df_grid['headway_score']
    )
    max_density = df_grid['교통약자_인구밀도'].max() or 1
    df_grid['accessibility_score'] = base * (df_grid['교통약자_인구밀도'] / max_density)
    return df_grid


def _gdf_to_geojson_ids(gdf):
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    ids = [str(i) for i in range(len(gdf))]
    geojson = {"type": "FeatureCollection", "features": []}
    for i, geom in enumerate(gdf.geometry):
        geojson["features"].append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {"id": ids[i]},
        })
    return geojson, ids

def plot_accessibility_plotly(
    gdf_grid,
    score_column='accessibility_score',
    colorscale='RdYlGn',      # 낮은값=빨강, 높은값=초록
    show_city_outline=True,
    outline_color='black',
    outline_width=2,
    grey_color='#222222',      # ← 더 진한 회색
    grey_opacity=0.15          # ← 낮은 alpha (투명도)
):
    gdf = gdf_grid.to_crs(4326).copy()

    # 중심 좌표는 투영 후 계산
    cent_5179 = gdf.to_crs(5179).geometry.centroid
    center = gpd.GeoSeries(cent_5179, crs=5179).to_crs(4326)
    center_lat = float(center.y.mean())
    center_lon = float(center.x.mean())

    # 도로 여부 분리
    if 'has_road' in gdf.columns:
        gdf_road = gdf[gdf['has_road'] == True]
        gdf_no_road = gdf[gdf['has_road'] == False]
    else:
        gdf_road = gdf
        gdf_no_road = gdf.iloc[0:0]

    data_traces = []

    # (1) 도로 없는 격자: 진한 회색 + 낮은 투명도로 바닥 처리
    if len(gdf_no_road) > 0:
        gj_no, ids_no = _gdf_to_geojson_ids(gdf_no_road)
        z_no = np.zeros(len(ids_no), dtype=float)
        trace_no = go.Choroplethmapbox(
            geojson=gj_no,
            locations=ids_no,
            featureidkey="properties.id",
            z=z_no,
            colorscale=[[0, grey_color], [1, grey_color]],  # 진한 회색
            zmin=0, zmax=1,
            marker_opacity=grey_opacity,                    # 낮은 alpha
            marker_line_width=0,
            showscale=False,                                # 컬러바 숨김
            hoverinfo="skip",
            showlegend=False,                               # 범례 숨김
            name=""                                         # 이름 비움
        )
        data_traces.append(trace_no)

    # (2) 도로 있는 격자: 점수 색상
    if len(gdf_road) > 0:
        gj_road, ids_road = _gdf_to_geojson_ids(gdf_road)
        z = gdf_road[score_column].fillna(0.0).astype(float).to_numpy()
        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z)) if float(np.nanmax(z)) > 0 else 1.0

        trace_road = go.Choroplethmapbox(
            geojson=gj_road,
            locations=ids_road,
            featureidkey="properties.id",
            z=z,
            colorscale=colorscale,                         # RdYlGn: 낮은=빨강, 높은=초록
            zmin=zmin, zmax=zmax,
            marker_opacity=0.80,
            marker_line_width=0,
            name="접근성 점수"                               # 컬러바만 생김(범례X)
        )
        data_traces.append(trace_road)

    fig = go.Figure(data=data_traces)

    # (3) 천안시 윤곽선: 범례 숨기기
    if show_city_outline:
        city = gdf_cheonan.to_crs(4326)
        for geom in city.geometry:
            polys = [geom] if geom.geom_type == "Polygon" else (list(geom.geoms) if geom.geom_type == "MultiPolygon" else [])
            for poly in polys:
                xs, ys = poly.exterior.coords.xy
                fig.add_trace(go.Scattermapbox(
                    lon=list(xs), lat=list(ys),
                    mode="lines",
                    line=dict(width=outline_width, color=outline_color),
                    hoverinfo="skip",
                    showlegend=False,                       # ← 여기!
                    name="천안시 경계"
                ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=11,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        margin={"r":0, "t":0, "l":0, "b":0},
        height=800
    )
    return fig

# -----------------------------
# 9. 파이프라인
# -----------------------------
def build_score_map(w=[0.25, 0.25, 0.25, 0.25], return_grid:bool = False):
    gdf_grid = create_grid(gdf_cheonan)

    weights_facility = {'장애인 재활시설':0.4, '노인 복지시설':0.3, '보건기관':0.2, '응급의료기관':0.1}
    gdf_grid = compute_nearest_facility_multi(gdf_grid, df_facility, '구분', weights_facility)
    gdf_grid = compute_lowfloor_score(gdf_grid, df_bus_routes)
    gdf_grid = compute_num_routes_score(gdf_grid, df_bus_routes_raw)
    gdf_grid = compute_headway_score(gdf_grid, df_bus_routes_raw, df_info)
    gdf_grid = compute_vulnerable_density(gdf_grid, gdf_cheonan, df_disabled, df_older)
    gdf_grid = compute_accessibility_score(gdf_grid, w)
    
    gdf_grid = flag_has_road(gdf_grid, gdf_roads)
    fig = plot_accessibility_plotly(gdf_grid)  # 내부에서 4326으로 변환해 그림

    # routes 페이지 등에서 쓰도록 grid를 같이 달라고 하면 4326으로 넘겨줌
    if return_grid:
        return fig, gdf_grid.to_crs(4326)
    return fig