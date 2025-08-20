# 천안시 1km 격자 — 셀별 '고유 노선 수' (Plotly, 0 회색, 깔끔 스타일, opacity 오류 수정)
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point, box
from shapely.ops import unary_union

GEO_PATH    = "./data/map_data/mapfile.geojson"
GRID_SIZE_M = 500
SHOW_LABELS = False
LABEL_MIN   = 1

# df_* 미정의시 로드
g = globals()
try:
    df_stops
except NameError:
    df_stops = pd.read_csv("./data/천안시_노선별경유정류소목록.csv")
try:
    df_num
except NameError:
    df_num = pd.read_csv("./data/천안시_노선번호목록.csv")
try:
    df_info
except NameError:
    df_info = pd.read_csv("./data/천안시_노선정보항목.csv", sep="\t")

def find_coord_cols(df):
    lat = next((c for c in df.columns if ("Y좌표" in c) or ("위도" in c)), None)
    lon = next((c for c in df.columns if ("X좌표" in c) or ("경도" in c)), None)
    if lat is None or lon is None:
        raise ValueError("위도/경도 컬럼을 찾지 못했습니다. (예: '정류소 Y좌표' / '정류소 X좌표')")
    return lat, lon

lat_col, lon_col = find_coord_cols(df_stops)
stops = df_stops.copy()
stops["__lat__"] = pd.to_numeric(stops[lat_col], errors="coerce")
stops["__lon__"] = pd.to_numeric(stops[lon_col], errors="coerce")
stops = stops.dropna(subset=["__lat__", "__lon__"]).copy()

gdf_stops = gpd.GeoDataFrame(
    stops, geometry=[Point(xy) for xy in zip(stops["__lon__"], stops["__lat__"])], crs=4326
)

# 행정경계 → 천안시만
gdf_emd = gpd.read_file(GEO_PATH)
if gdf_emd.crs is None: gdf_emd.set_crs(4326, inplace=True)
else: gdf_emd = gdf_emd.to_crs(4326)

if "SIG_CD" in gdf_emd.columns:
    mask = gdf_emd["SIG_CD"].astype(str).str.startswith(("44130","44131","44133"))
elif "SIG_KOR_NM" in gdf_emd.columns:
    mask = gdf_emd["SIG_KOR_NM"].astype(str).str.contains("천안")
elif "SGG_NM" in gdf_emd.columns:
    mask = gdf_emd["SGG_NM"].astype(str).str.contains("천안")
else:
    mask = pd.Series([True]*len(gdf_emd), index=gdf_emd.index)

gdf_city = gdf_emd[mask].copy()
if gdf_city.empty:
    raise ValueError("GeoJSON에서 '천안시'를 찾지 못했습니다.")

# 천안 내부 정류소만
city_union = unary_union(gdf_city.geometry)
try:
    idx = list(gdf_stops.sindex.intersection(city_union.bounds))
    gdf_stops_bb = gdf_stops.iloc[idx]
except Exception:
    gdf_stops_bb = gdf_stops
gdf_stops_c = gdf_stops_bb[gdf_stops_bb.geometry.intersects(city_union)].copy()

# 5179로 변환 → 500m 격자
city_m  = gdf_city.to_crs(5179)
stops_m = gdf_stops_c.to_crs(5179)
minx, miny, maxx, maxy = city_m.total_bounds

xs = np.arange(minx, maxx + GRID_SIZE_M, GRID_SIZE_M)
ys = np.arange(miny, maxy + GRID_SIZE_M, GRID_SIZE_M)

cells = []
u = city_m.unary_union
for ix, x0 in enumerate(xs[:-1]):
    for iy, y0 in enumerate(ys[:-1]):
        geom = box(x0, y0, x0 + GRID_SIZE_M, y0 + GRID_SIZE_M)
        if geom.intersects(u):
            cells.append({"cell_id": f"{ix}_{iy}", "geometry": geom})
grid_sq = gpd.GeoDataFrame(cells, crs=5179)

# 격자별 고유 노선 수
key_route = "노선ID" if "노선ID" in stops_m.columns else ("노선번호" if "노선번호" in stops_m.columns else None)
if key_route is None:
    raise ValueError("노선 식별 컬럼(노선ID/노선번호)이 필요합니다.")

join = gpd.sjoin(
    stops_m[["geometry", key_route]].dropna(subset=[key_route]),
    grid_sq[["cell_id", "geometry"]],
    how="inner", predicate="within"
)
route_counts = (
    join.groupby("cell_id")[key_route]
        .nunique().rename("route_count").reset_index()
)

# 클리핑 + 병합 → 4326
grid_clip = gpd.overlay(grid_sq, city_m[["geometry"]], how="intersection")
grid_clip = grid_clip.merge(route_counts, on="cell_id", how="left").fillna({"route_count": 0})
grid4326  = grid_clip.to_crs(4326)
city4326  = city_m.to_crs(4326)

# Plotly (opacity 제거, RGBA 색상 사용)
minx, miny, maxx, maxy = city4326.total_bounds
center_lon = float((minx + maxx) / 2.0)
center_lat = float((miny + maxy) / 2.0)

grid_pos  = grid4326[grid4326["route_count"] > 0]
grid_zero = grid4326[grid4326["route_count"] == 0]

# 연속형 팔레트(RGBA로 부드럽게)
ylgnbu_soft_rgba = [
    "rgba(247,252,240,0.95)", "rgba(224,243,219,0.95)", "rgba(204,235,197,0.95)",
    "rgba(168,221,181,0.95)", "rgba(123,204,196,0.95)", "rgba(78,179,211,0.95)",
    "rgba(43,140,190,0.95)",  "rgba(8,104,172,0.95)",  "rgba(8,64,129,0.95)"
]

fig = go.Figure()

# 0 셀: 회색 (RGBA), ✨ opacity 속성 없이 처리
if len(grid_zero) > 0:
    geojson_zero = json.loads(grid_zero.to_json())
    fig.add_trace(go.Choroplethmapbox(
        geojson=geojson_zero,
        featureidkey="properties.cell_id",
        locations=grid_zero["cell_id"],
        z=[0]*len(grid_zero),
        colorscale=[[0, "rgba(200,200,200,0.85)"], [1, "rgba(200,200,200,0.85)"]],
        showscale=False,
        marker_line_width=0.6,
        marker_line_color="white",
        hovertemplate="노선 수: 0<extra></extra>"
    ))

# 양수 셀: 연속색 (RGBA)
if len(grid_pos) > 0:
    geojson_pos = json.loads(grid_pos.to_json())
    vmin, vmax = int(grid_pos["route_count"].min()), int(grid_pos["route_count"].max())
    fig.add_trace(go.Choroplethmapbox(
        geojson=geojson_pos,
        featureidkey="properties.cell_id",
        locations=grid_pos["cell_id"],
        z=grid_pos["route_count"],
        colorscale=ylgnbu_soft_rgba,
        zmin=vmin, zmax=vmax,
        marker_line_width=0.8,
        marker_line_color="white",
        colorbar=dict(title="고유 노선 수", thickness=12, len=0.7, x=0.98, y=0.5, outlinewidth=0),
        hovertemplate="노선 수: %{z}<extra></extra>"
    ))

# (옵션) 라벨
if SHOW_LABELS and len(grid_pos) > 0:
    lab = grid_pos.copy()
    lab = lab[lab["route_count"] >= LABEL_MIN]
    lab["centroid"] = lab.geometry.representative_point()
    fig.add_trace(go.Scattermapbox(
        lat=[p.y for p in lab["centroid"]],
        lon=[p.x for p in lab["centroid"]],
        mode="text",
        text=[str(int(v)) for v in lab["route_count"]],
        textfont=dict(size=10, color="black"),
        hoverinfo="skip"
    ))

# 천안시 외곽선
for geom in city4326.geometry:
    if geom.geom_type == "Polygon":
        xs, ys = geom.exterior.coords.xy
        fig.add_trace(go.Scattermapbox(
            lon=list(xs), lat=list(ys), mode="lines",
            line=dict(width=1.2, color="rgba(0,0,0,0.55)"),
            hoverinfo="skip"
        ))
    elif geom.geom_type == "MultiPolygon":
        for poly in geom:
            xs, ys = poly.exterior.coords.xy
            fig.add_trace(go.Scattermapbox(
                lon=list(xs), lat=list(ys), mode="lines",
                line=dict(width=1.2, color="rgba(0,0,0,0.55)"),
                hoverinfo="skip"
            ))

fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_center={"lat": center_lat, "lon": center_lon},
    mapbox_zoom=11.2,
    margin=dict(l=10, r=10, t=42, b=10),
    title="천안시 500m 격자 — 셀별 ‘고유 노선 수’ (0 회색, 깨끗한 스타일)"
)
fig.show()
