# -*- coding: utf-8 -*-
# 목적: 천안시 NGII_LUM 데이터 중 "주거지역"만 필터링하여 시각화/저장

import os, glob, warnings, platform, sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# ---------------------------------
# 1) 사용자 경로 (폴더 안에 .shp/.shx/.dbf/.prj 세트가 있어야 함)
# ---------------------------------
DATA_DIR = r"C:\Users\USER\Desktop\project\cheonan-project\data_KHJ\map_data\NGII_LUM_44_충남_천안"

# .shx 누락시 자동 복구 시도
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")

# 경고 억제
warnings.filterwarnings("ignore")

# 한글 폰트 (Windows)
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------
# 2) 폴더 내 모든 SHP 읽기
# ---------------------------------
shps = sorted(glob.glob(os.path.join(DATA_DIR, "*.shp")))
if not shps:
    print(f"[오류] SHP를 찾지 못함: {DATA_DIR}")
    sys.exit(1)

layers = []
for fp in shps:
    try:
        gdf = gpd.read_file(fp)
        if not gdf.empty:
            layers.append((fp, gdf))
            print(f"[로드] {os.path.basename(fp)} ({len(gdf)} rows)")
    except Exception as e:
        print(f"[실패] {os.path.basename(fp)}: {e}")

if not layers:
    print("[오류] 읽을 수 있는 레이어가 없습니다.")
    sys.exit(1)

# ---------------------------------
# 3) CRS(좌표계) 맞추기 (가능하면 EPSG:5179로)
# ---------------------------------
base_crs = None
for _, g in layers:
    if g.crs is not None:
        base_crs = g.crs
        break
if base_crs is None:
    base_crs = "EPSG:5179"
    print("[안내] CRS가 없어 EPSG:5179로 가정합니다.")

poly_list = []
for fp, g in layers:
    g2 = g.copy()
    if g2.crs is None:
        g2.set_crs(base_crs, inplace=True)
    else:
        try:
            if str(g2.crs).lower() != str(base_crs).lower():
                g2 = g2.to_crs(base_crs)
        except Exception:
            pass
    # 폴리곤만 추출
    keep = g2[g2.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if not keep.empty:
        poly_list.append(keep)

if not poly_list:
    print("[오류] 폴리곤 레이어가 없습니다.")
    sys.exit(1)

poly = gpd.GeoDataFrame(pd.concat(poly_list, ignore_index=True), crs=base_crs)

# ---------------------------------
# 4) "주거지역" 필터링 — UCB 코드 사용
#    1110 = 단독주택, 1120 = 공동주택(아파트)
# ---------------------------------
poly["UCB_str"] = poly["UCB"].astype(str).str.strip()
res = poly[poly["UCB_str"].isin(["1110", "1120"])].copy()

if res.empty:
    print("[안내] UCB=1110/1120 조건으로 필터 결과가 비었습니다.")
    print("poly['UCB'].value_counts().head(20) 로 실제 분포 확인 필요.")
    sys.exit(0)

# ---------------------------------
# 5) 시각화
# ---------------------------------
fig, ax = plt.subplots(figsize=(11, 11))

# 전체 윤곽 (옅게 회색)
poly.boundary.plot(ax=ax, linewidth=0.2, alpha=0.4, color="#777777")

# 주거지역 강조 (노랑색)
res.plot(ax=ax, edgecolor="#333333", facecolor="#ffcc66", linewidth=0.4, alpha=0.85)

ax.set_aspect("equal", adjustable="box")
ax.set_title("천안시 NGII LUM — 주거지역 (UCB=1110,1120)", fontsize=16, pad=12)
ax.set_xlabel("X"); ax.set_ylabel("Y")
ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)

plt.tight_layout()

# ---------------------------------
# 6) 저장 (PNG + GeoPackage + GeoJSON)
# ---------------------------------
out_png = os.path.join(DATA_DIR, "cheonan_residential.png")
plt.savefig(out_png, dpi=220, bbox_inches="tight")
print(f"[저장] 주거지역 지도 PNG: {out_png}")
plt.show()

out_gpkg = os.path.join(DATA_DIR, "cheonan_residential.gpkg")
out_geojson = os.path.join(DATA_DIR, "cheonan_residential.geojson")
try:
    res.to_file(out_gpkg, layer="residential", driver="GPKG")
    print(f"[저장] {out_gpkg}")
except Exception as e:
    print("[경고] GPKG 저장 실패:", e)

try:
    res.to_file(out_geojson, driver="GeoJSON")
    print(f"[저장] {out_geojson}")
except Exception as e:
    print("[경고] GeoJSON 저장 실패:", e)
