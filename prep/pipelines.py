# prep/pipelines.py
from __future__ import annotations
import os
from pathlib import Path
import json
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ---- 경로 헬퍼 ----
def _must(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"[prep] 파일을 찾을 수 없음: {path}")
    return path

def read_csv_kr(path: Path) -> pd.DataFrame:
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

# ---- 로더: 경계/인구/시설/버스 ----
def load_boundary_geojson() -> tuple[dict, gpd.GeoDataFrame]:
    gj_path = _must(DATA / "map_data" / "cheonan_gdf.geojson")
    with open(gj_path, encoding="utf-8") as f:
        gj = json.load(f)
    gdf = gpd.GeoDataFrame.from_features(gj, crs="EPSG:4326")
    return gj, gdf

def load_disabled_df() -> pd.DataFrame:
    return read_csv_kr(_must(DATA / "population_data" / "disabled.csv"))

def load_older_df() -> pd.DataFrame:
    return read_csv_kr(_must(DATA / "population_data" / "older.csv"))

def load_facility_df() -> pd.DataFrame:
    """
    data/facility_data/통합_복지의료시설.csv
    필수 컬럼: ['구분','위도','경도'] + (이름/주소 컬럼은 다양해도 허용)
    """
    df = read_csv_kr(_must(DATA / "facility_data" / "통합_복지의료시설.csv"))
    # 컬럼 표준화
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    if "위도" not in df.columns or "경도" not in df.columns or "구분" not in df.columns:
        raise KeyError("[prep] 통합_복지의료시설.csv에 '구분/위도/경도' 컬럼이 필요합니다.")
    # 이름/주소 후보
    if "명 칭" in df.columns:
        df["이름"] = df["명 칭"]
    elif "기관명" in df.columns:
        df["이름"] = df["기관명"]
    else:
        df["이름"] = ""
    if "주소" in df.columns:
        df["주소_norm"] = df["주소"]
    elif "소 재 지" in df.columns:
        df["주소_norm"] = df["소 재 지"]
    else:
        df["주소_norm"] = ""
    return df

def load_bus_files() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    stops = read_csv_kr(_must(DATA / "bus_route_data" / "천안시_노선별경유정류소목록.csv"))
    info  = pd.read_csv(_must(DATA / "bus_route_data" / "천안시_노선정보항목.csv"), sep="\t")
    stops = stops.sort_values(["노선ID", "정류소순번"])
    gdf_stops = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["정류소 X좌표"], stops["정류소 Y좌표"]),
        crs="EPSG:4326",
    )
    return gdf_stops, info

# ---- OSM 그래프 & 노선 경로 ----
def build_osm_graph_for_boundary(gdf_boundary: gpd.GeoDataFrame, network_type="drive"):
    west, south, east, north = gdf_boundary.total_bounds
    return ox.graph.graph_from_bbox(bbox=(west, south, east, north), network_type=network_type)

def snap_stops_to_nodes(G, gdf_stops: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf_stops.copy()
    out["nearest_node"] = ox.distance.nearest_nodes(G, out.geometry.x, out.geometry.y)
    return out

def _shortest_path_sequence(G, seq_nodes: list[int]) -> list[int]:
    whole = []
    for u, v in zip(seq_nodes[:-1], seq_nodes[1:]):
        try:
            whole.extend(nx.shortest_path(G, u, v, weight="length"))
        except Exception:
            continue
    return whole

def build_routes_for_all_lines(G, gdf_stops_with_node: gpd.GeoDataFrame):
    routes = []
    for route_id, grp in gdf_stops_with_node.groupby("노선ID"):
        seq = _shortest_path_sequence(G, grp.sort_values("정류소순번")["nearest_node"].tolist())
        routes.append((route_id, seq))
    return routes

def filter_routes_in_boundary(G, routes, gdf_boundary):
    uni = gdf_boundary.unary_union
    kept = []
    for route_id, path in routes:
        coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in path]
        if coords and all(Point(x, y).within(uni) for x, y in coords):
            kept.append((route_id, path))
    return kept

def compute_routes_within_boundary():
    _, gdf_boundary = load_boundary_geojson()
    gdf_stops, df_info = load_bus_files()
    G = build_osm_graph_for_boundary(gdf_boundary, network_type="drive")
    stops_snap = snap_stops_to_nodes(G, gdf_stops)
    routes = build_routes_for_all_lines(G, stops_snap)
    routes = filter_routes_in_boundary(G, routes, gdf_boundary)
    return routes, G, gdf_boundary, df_info
