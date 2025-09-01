# viz/maps.py
from __future__ import annotations
import json
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os


# 프로젝트의 단일 데이터 소스
from shared import (
    gdf_cheonan, df_facility, df_disabled, df_older,
    df_bus_routes_raw, df_info
)

# ----------------------------
# 내부 유틸
# ----------------------------
def _geojson_from_gdf(gdf: gpd.GeoDataFrame) -> dict:
    """GeoDataFrame -> GeoJSON(dict)"""
    return json.loads(gdf.to_json())

def _ensure_adm_nm(df: pd.DataFrame) -> pd.DataFrame:
    """ADM_NM 열이 없으면 구군/읍면동 이름 열을 ADM_NM으로 맞춤."""
    out = df.copy()
    if "ADM_NM" not in out.columns:
        for cand in ("행정구역", "법정동명", "읍면동", "구군", "시군구", "행정동", "동이름"):
            if cand in out.columns:
                out["ADM_NM"] = out[cand].astype(str)
                break
    return out

def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _add_facility_markers(fig, kinds: list[str]):
    """df_facility에서 구분으로 필터해서 마커 추가."""
    if "구분" not in df_facility.columns:
        return fig
    base = df_facility.copy()
    base = base.dropna(subset=["위도", "경도"])
    base["구분"] = base["구분"].astype(str).str.strip()

    for kind in kinds:
        sub = base[base["구분"] == kind]
        if sub.empty:
            continue
        # 툴팁 텍스트 후보
        name_col = _pick_first_existing(sub, ["명 칭", "기관명", "시설명", "이름", "명칭"])
        addr_col = _pick_first_existing(sub, ["소 재 지", "주소", "도로명주소"])
        if name_col is None:
            name_col = "구분"
            sub = sub.assign(**{name_col: kind})
        text_vals = sub[name_col].astype(str)
        if addr_col:
            text_vals = text_vals + "<br>" + sub[addr_col].astype(str)

        fig.add_scattermapbox(
            lat=sub["위도"],
            lon=sub["경도"],
            mode="markers",
            marker=dict(size=10),
            text=text_vals,
            hoverinfo="text",
            name=kind,
        )
    return fig

# ----------------------------
# 1) 장애인 인구 지표 지도 (+ 선택 시설)
# ----------------------------
def make_disabled_map(add_facilities: bool = True, facility_types: list[str] | None = None):
    """
    df_disabled를 사용해 Choropleth Mapbox 생성.
    - color 컬럼은 ['장애인 인구 비율', '장애인등록인구'] 중 존재하는 것을 자동 선택
      (없으면 숫자형 컬럼 하나를 임의 선택)
    - add_facilities=True면 기본 시설 3종(장애인 재활시설/응급의료기관/보건기관) 마커 표시
    """
    geojson_data = _geojson_from_gdf(gdf_cheonan)
    df = _ensure_adm_nm(df_disabled)

    # 컬러 컬럼 자동 선택
    color_col = _pick_first_existing(df, ["장애인 인구 비율", "장애인등록인구", "장애인 인구 수", "장애인수"])
    if color_col is None:
        # 숫자형 아무거나 하나 선택
        num_cols = df.select_dtypes("number").columns.tolist()
        if not num_cols:
            raise ValueError("색칠할 숫자형 컬럼을 df_disabled에서 찾지 못했습니다.")
        color_col = num_cols[0]

    fig = px.choropleth_mapbox(
        df,
        geojson=geojson_data,
        locations="ADM_NM",
        featureidkey="properties.ADM_NM",
        color=color_col,
        color_continuous_scale="Blues",
        mapbox_style="carto-positron",
        center={"lat": float(gdf_cheonan.geometry.centroid.y.mean()),
                "lon": float(gdf_cheonan.geometry.centroid.x.mean())},
        zoom=9.5,
        opacity=0.7,
        hover_name="ADM_NM",
        height=600,
    )

    if add_facilities:
        kinds = facility_types or ["장애인 재활시설", "응급의료기관", "보건기관"]
        fig = _add_facility_markers(fig, kinds)

    fig.update_layout(
        margin=dict(r=0, t=30, l=0, b=0),
        legend=dict(orientation="h", yanchor="top", y=-0.1,
                    xanchor="center", x=0.5, font=dict(size=12)),
        title=f"장애인 지표(색: {color_col}) + 시설"
    )
    return fig

# ----------------------------
# 2) 노인 인구 지표 지도 (+ 선택 시설)
# ----------------------------
def make_older_map(add_facilities: bool = False, facility_types: list[str] | None = None):
    """
    df_older를 사용한 Choropleth Mapbox.
    - color 컬럼은 ['노인 인구 비율','노인 인구 수','노인인구비율','노인수'] 자동 선택
    - add_facilities=True면 기본 시설 3종(노인 복지시설/응급의료기관/보건기관) 마커 표시
    """
    geojson_data = _geojson_from_gdf(gdf_cheonan)
    df = _ensure_adm_nm(df_older)

    color_col = _pick_first_existing(df, ["노인 인구 비율", "노인 인구 수", "노인인구비율", "노인수"])
    if color_col is None:
        num_cols = df.select_dtypes("number").columns.tolist()
        if not num_cols:
            raise ValueError("색칠할 숫자형 컬럼을 df_older에서 찾지 못했습니다.")
        color_col = num_cols[0]

    vmin = float(pd.to_numeric(df[color_col], errors="coerce").min())
    vmax = float(pd.to_numeric(df[color_col], errors="coerce").max())

    fig = px.choropleth_mapbox(
        df,
        geojson=geojson_data,
        locations="ADM_NM",
        featureidkey="properties.ADM_NM",
        color=color_col,
        color_continuous_scale="Greens",
        range_color=(vmin, vmax),
        mapbox_style="carto-positron",
        center={"lat": float(gdf_cheonan.geometry.centroid.y.mean()),
                "lon": float(gdf_cheonan.geometry.centroid.x.mean())},
        zoom=9.5,
        opacity=0.6,
        hover_name="ADM_NM",
        height=600,
    )

    if add_facilities:
        kinds = facility_types or ["노인 복지시설", "응급의료기관", "보건기관"]
        fig = _add_facility_markers(fig, kinds)

    fig.update_layout(
        margin=dict(r=0, t=30, l=0, b=0),
        legend=dict(orientation="h", yanchor="top", y=-0.1,
                    xanchor="center", x=0.6, font=dict(size=12)),
        title=f"노인 지표(색: {color_col}) + 시설"
    )
    return fig

# ----------------------------
# 3) (간단판) 버스 노선 + 선택 시설 지도
#    - OSM 경로추정 없이 정류소 순서대로 선을 그려 빠르게 표시
# ----------------------------
def build_bus_and_facilities_map(selected_facilities: list[str] | None = None):
    """
    천안시 내부 정류소만 사용하여 버스 노선도 생성.
    - 저상버스 운행 노선: 검정색
    - 저상버스 예외 노선: 빨강색
    - 복지시설: 종류별 색상 + 범례 토글 가능
    """
    fig = go.Figure()

    # -------------------------
    # 천안시 경계
    # -------------------------
    for geom in gdf_cheonan.geometry:
        if geom.geom_type == "Polygon":
            lon, lat = geom.exterior.xy
            fig.add_trace(go.Scattermapbox(
                lon=list(lon), lat=list(lat),
                mode="lines", fill="toself",
                fillcolor="rgba(0,0,0,0.05)",
                line=dict(color="gray", width=1.5),
                showlegend=False
            ))
        elif geom.geom_type == "MultiPolygon":
            for sub in geom.geoms:
                lon, lat = sub.exterior.xy
                fig.add_trace(go.Scattermapbox(
                    lon=list(lon), lat=list(lat),
                    mode="lines", fill="toself",
                    fillcolor="rgba(0,0,0,0.05)",
                    line=dict(color="gray", width=1.5),
                    showlegend=False
                ))

    # -------------------------
    # 정류소 데이터 (천안시 내부만 필터링)
    # -------------------------
    stops = df_bus_routes_raw.dropna(subset=["X", "Y"]).copy()
    gdf_stops = gpd.GeoDataFrame(stops,
                                    geometry=gpd.points_from_xy(stops["X"], stops["Y"]),
                                    crs="EPSG:4326")
    gdf_stops = gdf_stops[gdf_stops.within(gdf_cheonan.unary_union)]
    gdf_stops["정류소순번"] = pd.to_numeric(gdf_stops["정류소순번"], errors="coerce")

    # -------------------------
    # 버스 노선 (운행/예외 구분)
    # -------------------------
    for rid, grp in gdf_stops.groupby("노선ID"):
        grp = grp.sort_values("정류소순번")
        xs, ys = grp["X"].tolist(), grp["Y"].tolist()
        if len(xs) < 2:
            continue

        info = df_info[df_info["노선ID"] == rid]
        bus_no = info.iloc[0]["노선번호"] if not info.empty else "NA"

        low_bus_flag = int(grp["저상버스"].max()) if "저상버스" in grp.columns else 1
        if low_bus_flag == 1:
            color, legend_name, group = "black", "저상버스 운행 노선", "운행"
        else:
            color, legend_name, group = "red", "저상버스 예외 노선", "예외"

        hover = f"노선ID: {rid}<br>버스번호: {bus_no}"

        fig.add_trace(go.Scattermapbox(
            lon=xs, lat=ys,
            mode="lines",
            line=dict(color=color, width=1.5),
            name=legend_name,
            legendgroup=group,          # 동일 그룹 묶기
            text=[hover] * len(xs),
            hoverinfo="text",
            showlegend=(legend_name not in [t.name for t in fig.data])  # 최초 1회만 범례 표시
        ))

    # -------------------------
    # 복지시설 (종류별 색상)
    # -------------------------
    color_map = {
        "응급의료기관": "red",
        "보건기관": "green",
        "장애인 재활시설": "blue",
        "노인 복지시설": "yellow"
    }
    selected_facilities = FAC_TYPES = ["응급의료기관","보건기관","노인 복지시설","장애인 재활시설"]

    if selected_facilities:
        df_fac = df_facility.dropna(subset=["위도", "경도"]).copy()
        for fac in selected_facilities:
            sub = df_fac[df_fac["구분"] == fac]
            if sub.empty:
                continue

            text_vals = sub.apply(
                lambda r: f"{r.get('기관명', r.get('명 칭','시설'))}<br>{r.get('주소', r.get('소 재 지',''))}",
                axis=1
            )

            fig.add_trace(go.Scattermapbox(
                lat=sub["위도"], lon=sub["경도"],
                mode="markers",
                marker=dict(size=10, color=color_map.get(fac, "blue"), opacity=0.7),
                text=text_vals,
                hoverinfo="text",
                name=fac,
                showlegend=True
            ))

    # -------------------------
    # 레이아웃
    # -------------------------
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={
            "lat": float(gdf_cheonan.geometry.centroid.y.mean()),
            "lon": float(gdf_cheonan.geometry.centroid.x.mean())
        },
        mapbox_zoom=9.5,
        margin=dict(r=0, t=0, l=0, b=0),
        legend=dict(title="범례", orientation="h", y=-0.1, x=0.5, xanchor="center"),
        title="천안시 버스 노선 + 복지시설"
    )
    
    # 프로젝트 최상위 기준으로 www 폴더 경로 잡기
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    www_dir = os.path.join(root_dir, "www")
    os.makedirs(www_dir, exist_ok=True)

    filename = "bus_facilities.png"
    path = os.path.join(www_dir, filename)

    pio.write_image(fig, path, format="png", width=1200, height=800, scale=2)
    print(f"✅ 저장 완료: {path}")   # 저장 확인용 로그

    return filename  # UI에서는 bus_facilities.png 로 참조

    # return fig
