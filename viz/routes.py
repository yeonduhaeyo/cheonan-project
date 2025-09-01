# viz/routes.py
import os
import math
import hashlib
import numpy as np
import geopandas as gpd
import networkx as nx
import osmnx as ox
from sklearn.cluster import DBSCAN
from shapely.geometry import LineString, Point
from shapely.geometry import mapping as shp_mapping
import pydeck as pdk
import matplotlib.cm as cm

from shared import (
    gdf_cheonan,
    df_facility,
    df_bus_routes_raw,   # 정류장(필수)
)

# ===============================
# 파라미터
# ===============================
OSM_NETWORK = "drive"
OSM_CUSTOM_FILTER = (
    '["highway"~"^(motorway|trunk|primary|secondary|tertiary|'
    'motorway_link|trunk_link|primary_link|secondary_link|tertiary_link)$"]'
)
GRAPH_BUFFER_M = 4000

FAC_TYPES_ALLOWED = [
    "장애인 재활시설",
    "노인 복지시설",
    "보건기관",
    "응급의료기관",
]

FAC_MATCH_TOL_M = 100.0         # 경로 끝점-시설 허용 반경 (살짝 완화)
OFFROAD_TOL_M_DEFAULT = 50.0    # 도로 영역 이탈 허용 길이 ↑
MERGE_MIN_BRANCH_LEN_M = 100.0  # 짧은 가지 병합 기준 ↓
DBSCAN_EPS_M = 1800             # 군집 반경 ↑
DBSCAN_MIN_SAMPLES = 2          # 최소 군집 크기 ↓
LOW_Q_DEFAULT = 0.35            # 하위 35% 선택 (노선 더 많아짐)

# 운영 제약
MIN_ROUTE_LEN_M = 500           # 최소 경로 길이 ↓
MAX_ROUTE_LEN_M = 20000         # 최대 경로 길이 ↑

# 새 정류장 제안
ROUTE_NEARBUF_M = 80
STOP_SPACING_M = 300
MIN_DIST_TO_EXISTING_M = 150
MIN_DIST_BETWEEN_NEW_M = 200
LOWFLOOR_RATIO_FLAG = 0.5       # 저상버스 미운행 비율 조건 완화


# ===============================
# 유틸
# ===============================
def _poly_hash(poly4326) -> str:
    wkb = gpd.GeoSeries([poly4326], crs=4326).to_wkb().iloc[0]
    return hashlib.md5(wkb).hexdigest()

def _load_or_build_graph(area_poly_4326):
    os.makedirs("./_graph_cache", exist_ok=True)
    key = hashlib.md5((_poly_hash(area_poly_4326) + OSM_CUSTOM_FILTER).encode()).hexdigest()
    path = f"./_graph_cache/osm_{OSM_NETWORK}_{key}.graphml"
    if os.path.exists(path):
        return ox.load_graphml(path)
    G = ox.graph_from_polygon(
        area_poly_4326, network_type=OSM_NETWORK, simplify=True, custom_filter=OSM_CUSTOM_FILTER
    )
    ox.save_graphml(G, path)
    return G

def _haversine_m(lon1, lat1, lon2, lat2):
    R=6371000.0
    from math import radians, sin, cos, atan2, sqrt
    dlon=radians(lon2-lon1); dlat=radians(lat2-lat1)
    a=sin(dlat/2)**2+cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))

def _path_length_m(coords):
    if not coords or len(coords) < 2:
        return 0.0
    return sum(_haversine_m(x1,y1,x2,y2) for (x1,y1),(x2,y2) in zip(coords[:-1], coords[1:]))


# ===============================
# 도로 여부(has_road) 계산
# ===============================
def _compute_has_road(gdf_grid, G):
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    if edges.empty:
        out = gdf_grid.copy(); out["has_road"] = False; return out
    grid_m = gdf_grid.to_crs(5179).reset_index(drop=True)
    edges_m = edges.to_crs(5179)[["geometry"]].reset_index(drop=True)
    j = gpd.sjoin(grid_m.reset_index().rename(columns={"index":"gid"}), edges_m, how="left", predicate="intersects")
    has = j.loc[j.index_right.notna(), "gid"].unique()
    grid_m["has_road"] = grid_m.index.isin(has)
    return grid_m.to_crs(4326)


# ===============================
# 하위 지역 군집화 + 시설 매칭
# ===============================
def _cluster_low_access(gdf_grid_scored, low_q=LOW_Q_DEFAULT, eps_m=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_SAMPLES):
    thr = gdf_grid_scored["accessibility_score"].quantile(low_q)
    cands = gdf_grid_scored[(gdf_grid_scored["has_road"]==True) & (gdf_grid_scored["accessibility_score"]<=thr)].copy()
    if cands.empty:
        return cands.assign(cluster=-1)

    cent_m = cands.to_crs(5179).geometry.centroid
    coords = np.c_[cent_m.x.values, cent_m.y.values]
    db = DBSCAN(eps=eps_m, min_samples=min_samples).fit(coords)
    cands["cluster"] = db.labels_
    return cands[cands["cluster"]>=0].copy()

def _pick_facility_per_cluster(cands_clustered, allowed_types=None, top_k=2):
    allowed_types = allowed_types or FAC_TYPES_ALLOWED
    fac_gdf = gpd.GeoDataFrame(
        df_facility, geometry=gpd.points_from_xy(df_facility["경도"], df_facility["위도"]), crs=4326
    ).to_crs(5179)
    fac_gdf = fac_gdf[fac_gdf["구분"].isin(allowed_types)]

    out = []
    for cid, grp in cands_clustered.groupby("cluster"):
        center_m = grp.to_crs(5179).geometry.unary_union.centroid
        if fac_gdf.empty:
            continue
        dists = fac_gdf.geometry.distance(center_m)
        top = dists.nsmallest(top_k).index
        for j in top:
            out.append({"cluster": int(cid), "facility_idx": int(j)})
    return out


# ===============================
# 경로 생성
# ===============================
def _build_routes_from_clusters(cands_clustered, matched, G, allowed_union, fac_gdf_4326):
    fac_ll = fac_gdf_4326.to_crs(4326)
    f_nodes = ox.distance.nearest_nodes(G, fac_ll.geometry.x.values, fac_ll.geometry.y.values)

    routes = []
    for item in matched:
        cid = item["cluster"]; fac_idx = item["facility_idx"]
        grp = cands_clustered[cands_clustered["cluster"]==cid]
        if grp.empty: continue

        center_ll = grp.to_crs(4326).geometry.unary_union.centroid
        center_node = ox.distance.nearest_nodes(G, center_ll.x, center_ll.y)
        fac_node = int(f_nodes[fac_idx])

        Gr = G.reverse(copy=False)
        paths = nx.single_source_dijkstra_path(Gr, fac_node, weight="length")
        if center_node not in paths: continue

        rev_nodes = list(reversed(paths[center_node]))
        coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in rev_nodes]
        if len(coords) < 2: continue

        # 시설 근접 보정
        fp = fac_ll.iloc[fac_idx].geometry
        if _haversine_m(coords[-1][0], coords[-1][1], fp.x, fp.y) > FAC_MATCH_TOL_M:
            coords.append((fp.x, fp.y))

        # 길이 제한
        L = _path_length_m(coords)
        if not (MIN_ROUTE_LEN_M <= L <= MAX_ROUTE_LEN_M):
            continue

        routes.append({
            "cluster": int(cid),
            "facility_name": str(fac_ll.iloc[fac_idx].get("명 칭", "시설")),
            "facility_type": str(fac_ll.iloc[fac_idx].get("구분", "")),
            "path": coords,
            "length_m": float(L),
        })

    return routes


# ===============================
# 새 정류장 후보
# ===============================
def _propose_new_stops_along_routes(routes, stops_raw):
    if stops_raw is None or len(routes)==0: return []

    gdf_stops = gpd.GeoDataFrame(
        stops_raw, geometry=gpd.points_from_xy(stops_raw["X"], stops_raw["Y"]), crs=4326
    ).to_crs(5179)

    new_stops = []
    for r in routes:
        coords = r.get("path", [])
        if len(coords) < 2: continue
        line_m = gpd.GeoSeries([LineString(coords)], crs=4326).to_crs(5179).iloc[0]

        near = gdf_stops[gdf_stops.geometry.within(line_m.buffer(ROUTE_NEARBUF_M))]

        lf_col = "저상버스" if "저상버스" in stops_raw.columns else None
        lowfloor_issue = False
        if lf_col and len(near) > 0:
            ratio_no = float((near[lf_col].fillna(0).astype(float) <= 0).mean())
            lowfloor_issue = (ratio_no >= LOWFLOOR_RATIO_FLAG)

        exist_pts = np.c_[near.geometry.x.values, near.geometry.y.values] if len(near) else np.zeros((0,2))

        total, t, last_new_xy = line_m.length, 0.0, None
        while t <= total:
            p = line_m.interpolate(t); px, py = float(p.x), float(p.y)
            ok_far = True
            if exist_pts.shape[0] > 0:
                d = np.hypot(exist_pts[:,0] - px, exist_pts[:,1] - py).min()
                if d < MIN_DIST_TO_EXISTING_M: ok_far = False
            if ok_far and last_new_xy is not None:
                if math.hypot(last_new_xy[0] - px, last_new_xy[1] - py) < MIN_DIST_BETWEEN_NEW_M:
                    ok_far = False
            if ok_far:
                p_ll = gpd.GeoSeries([Point(px, py)], crs=5179).to_crs(4326).iloc[0]
                reason = "정류장 공백 구간"
                if lowfloor_issue: reason += " / 저상버스 미운행 구간"
                new_stops.append({"lon": float(p_ll.x),"lat": float(p_ll.y),"reason": reason,"cluster": r.get("cluster")})
                last_new_xy = (px, py)
            t += STOP_SPACING_M
    return new_stops


# ===============================
# 시각화
# ===============================
def _get_color_palette(n):
    cmap = cm.get_cmap("tab20", n)
    return [[int(r*255), int(g*255), int(b*255), 220] for r,g,b,_ in cmap(np.linspace(0,1,n))]

def _deck_map(gdf_grid_full, routes, new_stops, score_col="accessibility_score"):
    gdf_ll = gdf_grid_full.to_crs(4326).copy()
    gdf_ll = gdf_ll[gdf_ll.geometry.notna() & (~gdf_ll.geometry.is_empty)]
    gdf_ll = gdf_ll[gdf_ll.geometry.type.isin(["Polygon","MultiPolygon"])]

    city_ctr = gdf_cheonan.to_crs(4326).geometry.unary_union.centroid
    ctr_lat, ctr_lon = float(city_ctr.y), float(city_ctr.x)

    road_yes = gdf_ll[gdf_ll.get("has_road", True)==True]

    vmin, vmax = float(road_yes[score_col].min()), float(road_yes[score_col].max())

    def _color_score(v):
        t = 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin + 1e-9)
        if t < 0.5: return [255, int(255 - (255 - 165) * (t / 0.5)), 0, 160]
        else: return [int(80 + (255 - 80) * ((t - 0.5) / 0.5)), 200, 80, 160]

    feats_yes = [{"type":"Feature","geometry":shp_mapping(geom),
                  "properties":{"fill_color":_color_score(float(v)),"tooltip":f"접근성 점수: {float(v):.3f}"}}
                 for geom,v in zip(road_yes.geometry, road_yes[score_col])]

    layers = [pdk.Layer("GeoJsonLayer", data={"type":"FeatureCollection","features":feats_yes},
                        pickable=True, auto_highlight=True, filled=True,
                        get_fill_color="properties.fill_color")]

    if routes:
        palette = _get_color_palette(len(routes))
        path_data = [{"path": r["path"],"color": palette[i % len(palette)],
                      "tooltip": f"군집 {r['cluster']} → {r['facility_name']} ({r['facility_type']})<br>길이: {r['length_m']:.0f} m"}
                     for i,r in enumerate(routes) if len(r.get("path",[]))>=2]
        layers.append(pdk.Layer("PathLayer", data=path_data, get_path="path", get_color="color",
                                width_scale=1.2, width_min_pixels=3, pickable=True))

    if new_stops:
        layers.append(pdk.Layer("ScatterplotLayer",
                                data=[{"position":[s["lon"], s["lat"]], "tooltip": s["reason"]} for s in new_stops],
                                get_position="position", get_fill_color=[220,20,60,220],
                                get_radius=70, pickable=True))

    view = pdk.ViewState(latitude=ctr_lat, longitude=ctr_lon, zoom=11)
    return pdk.Deck(layers=layers, initial_view_state=view,
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    tooltip={"html":"{tooltip}", "style":{"color":"white"}})


# ===============================
# 외부 API (Shiny에서 호출)
# ===============================
def build_shuttle_routes_html_from_grid(gdf_grid_scored,
    low_q: float = LOW_Q_DEFAULT,
    allowed_facility_types = None,
    dbscan_eps_m: float = DBSCAN_EPS_M,
    dbscan_min_samples: int = DBSCAN_MIN_SAMPLES,
    offroad_tol_m: float = OFFROAD_TOL_M_DEFAULT):

    poly = gdf_cheonan.to_crs(4326).unary_union.convex_hull
    area = gpd.GeoSeries([poly], crs=4326).to_crs(5179).buffer(GRAPH_BUFFER_M).to_crs(4326).iloc[0]

    G = _load_or_build_graph(area)
    grid_full = _compute_has_road(gdf_grid_scored, G)

    cand_clustered = _cluster_low_access(grid_full, low_q=low_q, eps_m=dbscan_eps_m, min_samples=dbscan_min_samples)
    if cand_clustered.empty:
        deck = _deck_map(grid_full.assign(has_road=False), routes=[], new_stops=[])
        return deck.to_html(as_string=True, iframe_height=800)

    fac_gdf_4326 = gpd.GeoDataFrame(
        df_facility, geometry=gpd.points_from_xy(df_facility["경도"], df_facility["위도"]), crs=4326
    )
    fac_gdf_4326 = fac_gdf_4326[fac_gdf_4326["구분"].isin(allowed_facility_types or FAC_TYPES_ALLOWED)].to_crs(4326)

    matched = _pick_facility_per_cluster(cand_clustered, allowed_types=allowed_facility_types, top_k=2)

    allowed = grid_full[grid_full["has_road"]==True][["geometry"]].to_crs(5179)
    allowed_union = allowed.unary_union

    routes = _build_routes_from_clusters(cand_clustered, matched, G, allowed_union, fac_gdf_4326)

    new_stops = _propose_new_stops_along_routes(routes, df_bus_routes_raw)

    deck = _deck_map(grid_full, routes, new_stops)
    return deck.to_html(as_string=True, iframe_height=800)
