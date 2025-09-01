import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import osmnx as ox

# 한글 폰트
mpl.rc('font', family='Malgun Gothic')
mpl.rcParams['axes.unicode_minus'] = False

# <격자데이터 생성>
# 1. 천안시 행정구역 불러오기
cheonan = gpd.read_file("./data_MJ/cheonan_gdf.geojson")
# 2. 거리 기반 격자를 만들기 위해 UTM 좌표계로 변환
cheonan = cheonan.to_crs(epsg=5179)  # 한국형 UTM (단위: meter)
# 3. 천안시 영역 전체 범위 구하기
minx, miny, maxx, maxy = cheonan.total_bounds
# 4. 격자 크기 설정 
grid_size = 500  # meters
# 5. 격자 생성
cols = np.arange(minx, maxx + grid_size, grid_size)
rows = np.arange(miny, maxy + grid_size, grid_size)

grid_cells = []
for x in cols:
    for y in rows:
        cell = box(x, y, x + grid_size, y + grid_size)
        grid_cells.append(cell)

grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=cheonan.crs)

# 6. 천안시 영역과 겹치는 격자만 추출
grid = gpd.overlay(grid, cheonan, how='intersection')

# # 7. 시각화
# ax = cheonan.plot(color='white', edgecolor='black', figsize=(10, 10))
# grid.boundary.plot(ax=ax, linewidth=0.5, color='blue')
# # plt.title("천안시 단위공간 격자 (500m x 500m)")
# # plt.show()

# 천안시 중심 좌표
place = "Cheonan, South Korea"
roads = ox.graph_from_place(place, network_type='drive')  # 도로망
gdf_roads = ox.graph_to_gdfs(roads, nodes=False, edges=True)
gdf_roads = gdf_roads.to_crs(epsg=5179)

#  시각화
fig, ax = plt.subplots(figsize=(12, 12))
cheonan.plot(ax=ax, color='white', edgecolor='black', linewidth=1, label="천안시 행정구역")
grid.boundary.plot(ax=ax, linewidth=0.5, color='blue', label="500m 격자")
gdf_roads.plot(ax=ax, linewidth=0.8, color='red', alpha=0.7, label="도로망")

plt.title("천안시 500m 격자 및 도로망", fontsize=15)
plt.legend()
plt.axis("off")
plt.show()




# ④-1)장애인 인구밀도
import geopandas as gpd
import pandas as pd

def get_grid_id(geom, grid_size=500):
    centroid = geom.centroid
    x_id = int(np.floor(centroid.x / grid_size))
    y_id = int(np.floor(centroid.y / grid_size))
    return f"{x_id}_{y_id}"

grid['grid_id'] = grid.geometry.apply(get_grid_id)

emd = gpd.read_file("./data_MJ/cheonan_gdf.geojson")
# 인구 데이터 (CSV) 불러오기
disabled=pd.read_csv('./data_MJ/disabled.csv',encoding='utf-8-sig')
disabled= disabled.rename(columns={"ADM_NM": "DIS_ADM_NM"})
disabled['DIS_ADM_NM'] = disabled['DIS_ADM_NM'].astype(str).str.strip()
# 3. 병합 (읍면동명 기준)
emd['ADM_NM'] = emd['ADM_NM'].astype(str).str.strip()
emd_pop = emd.merge(disabled, left_on="ADM_NM", right_on="DIS_ADM_NM",how='inner')
emd_pop_clean = emd_pop[['geometry', 'ADM_NM', '장애인등록인구']].copy()
emd_pop_clean['geometry'] = emd_pop_clean.geometry.buffer(0)

grid_clean = grid[['geometry', 'grid_id']].copy()
grid_clean['geometry'] = grid_clean.geometry.buffer(0)

# 7. 좌표계 통일 (면적 계산용)
emd_pop_clean = emd_pop_clean.to_crs(epsg=5179)
grid_clean = grid_clean.to_crs(epsg=5179)

# 8. 읍면동 면적 계산
emd_pop_clean['읍면동면적'] = emd_pop_clean.geometry.area

# 9. 겹치는 영역 계산 (overlay)
intersect = gpd.overlay(grid_clean, emd_pop_clean, how='intersection')

# 겹친 영역의 면적 계산 (m² 단위)
intersect['면적'] = intersect.geometry.area

# 11. 장애인 인구를 면적 비율로 분배
intersect['장애인_분배'] = intersect['장애인등록인구'] * (intersect['면적'] / intersect['읍면동면적'])

# 12. 격자별 장애인 인구 합산
pop_by_grid = intersect.groupby('grid_id')['장애인_분배'].sum().reset_index()

# 13. 밀도 계산 (단위: 명/km², 500m * 500m 격자 가정)
grid_area_km2 = 0.25
pop_by_grid['장애인_인구밀도'] = pop_by_grid['장애인_분배'] / grid_area_km2

# 14. 기존 grid에 장애인 인구밀도 병합 (좌표계 다시 원복 필요하면 수행)
grid_df = grid.merge(pop_by_grid[['grid_id', '장애인_인구밀도']], on='grid_id', how='left')
grid_df['장애인_인구밀도'] = grid_df['장애인_인구밀도'].fillna(0)


# ④-2) 노인 인구 밀도
# 노인 인구 데이터 불러오기
older=pd.read_csv('./data_MJ/older.csv',encoding='utf-8-sig')
older = older.rename(columns={'ADM_NM': 'OLD_ADM_NM'})
older['65세 이상 인구'] = pd.to_numeric(
    older['65세 이상 인구'].astype(str).str.replace(',', '', regex=False),
    errors='coerce'
)
# emd 정리
emd['ADM_NM'] = emd['ADM_NM'].astype(str).str.strip()

emd_pop_older = emd.merge(older, left_on='ADM_NM', right_on='OLD_ADM_NM', how='inner')

# 필요한 칼럼만 정리
emd_older_clean = emd_pop_older[['geometry', 'ADM_NM', '65세 이상 인구']].copy()
emd_older_clean['geometry'] = emd_older_clean.geometry.buffer(0)

# 좌표계 EPSG:5179로 통일
emd_older_clean = emd_older_clean.to_crs(epsg=5179)
grid_clean = grid.to_crs(epsg=5179)

# 면적 계산
emd_older_clean['읍면동면적'] = emd_older_clean.geometry.area

# overlay로 교차 계산
intersect_older = gpd.overlay(grid_clean, emd_older_clean, how='intersection')
intersect_older['면적'] = intersect_older.geometry.area

# 인구 분배
intersect_older['노인_비율'] = intersect_older['65세 이상 인구'] * (intersect_older['면적'] / intersect_older['읍면동면적'])

# 격자별 집계
pop_by_grid_older = intersect_older.groupby('grid_id')['노인_비율'].sum().reset_index()
pop_by_grid_older['노인_인구밀도'] = pop_by_grid_older['노인_비율'] / grid_area_km2

# grid_df에 병합
grid_df = grid_df.merge(pop_by_grid_older[['grid_id', '노인_인구밀도']], on='grid_id', how='left')
grid_df['노인_인구밀도'] = grid_df['노인_인구밀도'].fillna(0)

# 노인_인구밀도와 장애인_인구밀도를 합쳐서 교통약자_인구밀도
grid_df['교통약자_인구밀도'] = grid_df['노인_인구밀도'] + grid_df['장애인_인구밀도']

# 시각화
fig, ax = plt.subplots(figsize=(10, 10))

# 교통약자 인구 밀도 기준 색상 표시
grid_df.plot(
    column='교통약자_인구밀도',
    cmap='YlOrRd',  # 노란색~빨간색
    linewidth=0.1,
    edgecolor='gray',
    legend=True,
    ax=ax
)
ax.set_title("천안시 격자별 교통약자 인구 밀도", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()


# ✅ 행정동 단위로 교통약자 인구밀도 집계
import geopandas as gpd
import plotly.express as px
# 1. 좌표계 맞추기
emd = emd.to_crs(epsg=5179)

# 2. grid_df (격자)와 emd (행정동) 겹치는 부분 계산
intersect_adm = gpd.overlay(grid_df, emd, how='intersection')
intersect_adm['면적'] = intersect_adm.geometry.area
# 3. 겹친 영역 면적과 가중치 곱하기
intersect_adm['면적'] = intersect_adm.geometry.area
intersect_adm['가중_밀도'] = intersect_adm['교통약자_인구밀도'] * intersect_adm['면적']

# 4. 읍면동별 가중평균 밀도 계산
adm_density = intersect_adm.groupby('ADM_NM_2').agg(
    total_weighted_density=('가중_밀도', 'sum'),
    total_area=('면적', 'sum')
).reset_index()

adm_density['교통약자_인구밀도'] = adm_density['total_weighted_density'] / adm_density['total_area']
# 단위 면적 당 (장애인+노인) 인구 수
# 행정동별 가중평균 = Σ(가중_밀도) / Σ(면적)


# 행정동별 특징 컬럼 추가
feature_dict = {
    "목천읍": "산간지대 많음",
    "풍세면": "축산업 및 도시근교 농업이 발달되어 있고, 산업단지가 조성되어 있음",
    "광덕면": "청정보존지역",
    "북면": "고소득 작목으로 각광받고 있는 전형적인 농촌지역",
    "성남면": "농업 지역이자 천안제5일반산업단지가 조성되어 있음",
    "수신면":"농촌 지역이자 관내 제5산업단지 및 150여 개의 중소기업체 소재",
    "병천면":"관광명소",
    "동면":"전형적인 농촌 지역이자 노인 인구가 총40%에 육박하는 고령화 지역",
    "중앙동":"유통업체 및 상가 밀집 지역으로 유동인구가 많음",
    "문성동":"천안역과 터미널 사이에 위치한 교통의 요충지",
    "원성1동":"도·농 복합 지역",
    "원성2동":"도심과 인접한 지역",
    "봉명동":"교육기관과 종합병원이 소재한 전형적인 원도심 지역",
    "일봉동":"대규모 아파트단지가 조성되어 많은 인구가 거주하는 천안의 대표적인 주거지역",
    "신방동":"KTX 천안아산역에 인접한 도·농 복합 지역",
    "청룡동":"도·농 복합 지역",
    "신안동":"5개 대학이 위치하고 있으며 교통과 유통의 요충지",
    "성환읍":"수도권 배후지역으로 국도, 철도가 관통하며 교통이 편리한 지역",
    "성거읍":"시내에 인접해있고 154여개 기업체가 소재하는 농공복합 지역",
    "직산읍":"제 4산업단지가 조성 중이며 신흥개발지역으로 변화 중인 지역",
    "입장면":"고소득 작목과 미곡 및 축산이 발달한 지역",
    "성정1동":"상대적으로 낙후된 서편과 새로운 교통 및 유통 중심지로 급부상인 동편이 함께 존재하는 지역",
    "성정2동":"원룸, 빌라, 도시형 주택 밀집지역",
    "쌍용1동":"패션 및 먹거리 상가가 밀집한 소비 중심지역",
    "쌍용2동":"천안의 대표적인 아파트 밀집지역",
    "쌍용3동":"단위면적당 인구밀집도가 가장 높은 지역",
    "백석동":"최첨단복합인프라가 구축되어 있는 지역",
    "불당1동":"행정·주거·교육의 인프라가 완벽하게 구축되어 있는 지역",
    "불당2동":"대규모 아파트단지, 교육시설, 상업지구, 문화시설, 도시공원 등 인프라가 충분한 지역",
    "부성1동":"도·농 복합 지역",
    "부성2동":"상업과 주거의 집적화로 주민생활 거점축을 형성한 지역" }

# 5. 원래 행정동 지오데이터와 병합
emd_density = emd.merge(
    adm_density[['ADM_NM_2', '교통약자_인구밀도','total_area']],
    left_on='ADM_NM', right_on='ADM_NM_2',
    how='left'
)
# 면적 계산 (단위: 제곱킬로미터)
emd_density['면적_km2'] = emd_density['total_area'] / 1e6
emd_density['행정동_특징'] = emd_density['ADM_NM_2'].map(feature_dict).fillna("특징 없음")  

emd_density = emd_density.to_crs(epsg=4326)
geojson = emd_density.__geo_interface__
# ✅ 시각화 (행정동별 교통약자 인구 밀도)
fig = px.choropleth_mapbox(
    emd_density,
    geojson=geojson,
    locations='ADM_NM_2',
    featureidkey='properties.ADM_NM',
    color='교통약자_인구밀도',
    hover_name='ADM_NM',
    hover_data={
        '교통약자_인구밀도': ':.2f',
        '면적_km2': ':.2f',
        '행정동_특징': True
    },
    center={"lat": 36.8, "lon": 127.1},  # 천안시 중심 좌표
    mapbox_style="carto-positron",
    zoom=10,
    color_continuous_scale="YlOrRd"
)

fig.update_layout(
    margin={"r":0,"t":30,"l":0,"b":0},
    title="천안시 행정동별 교통약자 인구 밀도"
)

fig.show()

# ✅ 상하위 동 리스트 출력
top5 = adm_density.sort_values(by='교통약자_인구밀도', ascending=False).head(5)
bottom5 = adm_density.sort_values(by='교통약자_인구밀도').head(5)

print("🔺 교통약자 인구밀도가 높은 상위 5개 행정동:")
print(top5[['ADM_NM_2', '교통약자_인구밀도']], '\n')

print("🔻 교통약자 인구밀도가 낮은 하위 5개 행정동:")
print(bottom5[['ADM_NM_2', '교통약자_인구밀도']])




