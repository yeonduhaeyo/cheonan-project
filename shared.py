# shared.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import os, json
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 기본 경로
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WWW_DIR = ROOT / "www"
HTML_OUT = ROOT / "html_out"

# 서브 폴더(있으면 사용)
MAP_DIR        = DATA_DIR / "map_data"
FACILITY_DIR   = DATA_DIR / "facility_data"
BUS_DIR        = DATA_DIR / "bus_route_data"
POP_DIR        = DATA_DIR / "population_data"

# -------------------------------------------------------------------
# Matplotlib 한글 설정
# -------------------------------------------------------------------
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# -------------------------------------------------------------------
# 공통 유틸
# -------------------------------------------------------------------
def read_csv_kr(path: str | os.PathLike) -> pd.DataFrame:
    """cp949 ↔ UTF-8-SIG 자동 시도하여 CSV 로드"""
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    # 마지막 시도(기본)
    return pd.read_csv(path)

def _find_in_data(file_name: str) -> Optional[Path]:
    """
    data/ 하위(서브폴더 포함)에서 file_name을 찾아 첫 경로 반환.
    """
    # 1) 최상위 data 바로 아래 우선
    p = DATA_DIR / file_name
    if p.exists():
        return p
    # 2) 재귀 탐색
    try:
        for found in DATA_DIR.rglob(file_name):
            return found
    except Exception:
        pass
    return None

# -------------------------------------------------------------------
# 기본 데이터(프로젝트 공통) — 앱 시작 시 미리 읽어두는 것들
# -------------------------------------------------------------------
# 행정구역 경계 (천안시)
gdf_cheonan = gpd.read_file((MAP_DIR / "cheonan_gdf.geojson").resolve())

# 도로 데이터 (필요 없는 페이지에서는 안 써도 OK)
gdf_roads = gpd.read_file((MAP_DIR / "cheonan_roads.geojson").resolve())

# 복지·의료 시설 (필수 컬럼: '구분','위도','경도')
df_facility = read_csv_kr((FACILITY_DIR / "통합_복지의료시설.csv").resolve())
# 타입/좌표 정리
if "구분" in df_facility.columns:
    df_facility["구분"] = df_facility["구분"].astype(str).str.strip()
for c in ("위도", "경도"):
    if c in df_facility.columns:
        df_facility[c] = pd.to_numeric(df_facility[c], errors="coerce")

# 버스 노선/정류소
df_info = pd.read_csv((BUS_DIR / "천안시_노선정보항목.csv").resolve(), sep="\t")
df_bus_routes_raw = read_csv_kr((BUS_DIR / "bus_routes_raw.csv").resolve())
df_bus_routes     = read_csv_kr((BUS_DIR / "bus_routes_summary.csv").resolve())

# 인구 데이터
df_disabled = read_csv_kr((POP_DIR / "disabled.csv").resolve())  # 읍면동별 장애인 등록 인구
df_older    = read_csv_kr((POP_DIR / "older.csv").resolve())     # 읍면동별 65세 이상 인구

# 색상 팔레트
ACCESSIBILITY_COLOR_MAP = {
    "낮음": "#d62728",
    "보통": "#ff7f0e",
    "높음": "#2ca02c",
}

# -------------------------------------------------------------------
# 차트/시각화 모듈에서 사용할 로더(파일명 표준화)
# -------------------------------------------------------------------
def load_disability_registered_raw() -> pd.DataFrame:
    """
    충남 시군구별 등록장애인수 원본.
    파일명: '시군구별_장애정도별_성별_등록장애인수.csv'
    권장 위치: data/population_data/
    """
    cand = (POP_DIR / "시군구별_장애정도별_성별_등록장애인수.csv")
    path = cand if cand.exists() else _find_in_data("시군구별_장애정도별_성별_등록장애인수.csv")
    if not path or not path.exists():
        raise FileNotFoundError("[shared] '시군구별_장애정도별_성별_등록장애인수.csv'를 data/ 하위에서 찾지 못했습니다.")
    return read_csv_kr(path)

def load_age_distribution_file(year: int) -> Optional[pd.DataFrame]:
    """
    연령별 인구현황(연도별) CSV 로드.
    파일명 패턴: '연령별인구현황({year}.7월)-게시용.csv'
    권장 위치: data/population_data/
    """
    fname = f"연령별인구현황({year}.7월)-게시용.csv"
    cand = POP_DIR / fname
    path = cand if cand.exists() else _find_in_data(fname)
    if not path:
        return None
    return read_csv_kr(path)

def load_age_distribution_years(
    years: Iterable[int] = (2021, 2022, 2023, 2024, 2025)
) -> dict[int, pd.DataFrame]:
    """
    여러 연도 파일을 한 번에 로드해서 {연도: DataFrame} 반환.
    없는 연도는 생략.
    """
    out: dict[int, pd.DataFrame] = {}
    for y in years:
        df = load_age_distribution_file(y)
        if df is not None:
            out[y] = df
    if not out:
        raise FileNotFoundError("[shared] 연령별인구현황 CSV들을 data/ 하위에서 찾지 못했습니다.")
    return out

# -------------------------------------------------------------------
# GeoJSON 경계 로더(필요 시 dict+GeoDataFrame 동시 제공)
# -------------------------------------------------------------------
def load_boundary_geojson() -> tuple[dict, gpd.GeoDataFrame]:
    p = (MAP_DIR / "cheonan_gdf.geojson").resolve()
    with open(p, encoding="utf-8") as f:
        gj = json.load(f)
    gdf = gpd.GeoDataFrame.from_features(gj, crs="EPSG:4326")
    return gj, gdf

# -------------------------------------------------------------------
# 시설 필터 유틸
# -------------------------------------------------------------------
def facility_types() -> list[str]:
    """시설 구분 목록(표시용)"""
    if "구분" not in df_facility.columns:
        return []
    return sorted(df_facility["구분"].dropna().astype(str).unique().tolist())

def get_facilities(types: Iterable[str] | None = None) -> pd.DataFrame:
    """
    시설 DF 반환(필요 시 구분 필터)
    """
    if types:
        mask = df_facility["구분"].astype(str).isin([str(t) for t in types])
        return df_facility.loc[mask].copy()
    return df_facility.copy()

# -------------------------------------------------------------------
# 내보낼 심볼
# -------------------------------------------------------------------
__all__ = [
    # 경로
    "ROOT", "DATA_DIR", "WWW_DIR", "HTML_OUT",
    # 공통 오브젝트
    "gdf_cheonan", "gdf_roads", "df_facility",
    "df_info", "df_bus_routes_raw", "df_bus_routes",
    "df_disabled", "df_older",
    "ACCESSIBILITY_COLOR_MAP",
    # 유틸/로더
    "read_csv_kr", "_find_in_data",
    "load_disability_registered_raw",
    "load_age_distribution_file", "load_age_distribution_years",
    "load_boundary_geojson",
    "facility_types", "get_facilities",
]