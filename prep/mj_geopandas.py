import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (맑은 고딕)
font_path = 'C:/Windows/Fonts/malgun.ttf'  # Windows 환경
fontprop = fm.FontProperties(fname=font_path, size=14)
plt.rc('font', family=fontprop.get_name())

# 1. shapefile 불러오기
gdf = gpd.read_file('./data_MJ/시군구/시군구.shp', encoding='euc-kr')

# 2. 좌표계 설정 및 변환
if gdf.crs is None:
    # shapefile 원 좌표계가 EPSG:5179라고 가정 (보통 한국 shp는 이 좌표계일 가능성 높음)
    gdf = gdf.set_crs(epsg=5179)

# 3. WGS84 (EPSG:4326) 위도경도 좌표계로 변환
gdf = gdf.to_crs(epsg=4326)

# 4. 천안시 필터링
cheonan_gdf = gdf[gdf['SIG_KOR_NM'].str.contains('천안')].copy()

# 5. 천안시 경계 단일 폴리곤 병합
cheonan_boundary = unary_union(cheonan_gdf.geometry)

# 6. 경계 GeoDataFrame 생성
boundary_gdf = gpd.GeoDataFrame(geometry=[cheonan_boundary], crs='EPSG:4326')
print(boundary_gdf)
# 7. 재활시설 데이터 불러오기
df_facilities = pd.read_csv('./data_MJ/장애인_재활시설.csv')

# 8. 재활시설 GeoDataFrame 생성
geometry = [Point(xy) for xy in zip(df_facilities['경도'], df_facilities['위도'])]
facilities_gdf = gpd.GeoDataFrame(df_facilities, geometry=geometry, crs='EPSG:4326')

# 9. 시각화
fig, ax = plt.subplots(figsize=(10,10))
boundary_gdf.boundary.plot(ax=ax, edgecolor='blue', linewidth=2, label='천안시 경계')
facilities_gdf.plot(ax=ax, color='red', markersize=50, label='재활시설')

ax.set_title('천안시 경계 및 장애인 재활시설 위치')
ax.set_xlabel('경도')
ax.set_ylabel('위도')
ax.legend()
plt.show()