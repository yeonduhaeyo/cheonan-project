### <첫 번째 시각화> 천안시 읍/면의 고령화 심각성 ### [2~307]

# =========================
# 0) 한글 폰트 설정(Windows/macOS/Linux)
# =========================
import os, re, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 폰트
if platform.system() == 'Windows':
    for fp in [r"C:/Windows/Fonts/malgun.ttf",
               r"C:/Windows/Fonts/malgunsl.ttf",
               r"C:/Windows/Fonts/gulim.ttc",
               r"C:/Windows/Fonts/batang.ttc"]:
        if os.path.exists(fp):
            try: fm.fontManager.addfont(fp)
            except: pass
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1) 파일 경로 설정
# =========================
local_files = [
    r"C:/Users/USER/Desktop/project/cheonan-project/data_KHJ/연령별인구현황(2021.7월)-게시용.csv",
    r"C:/Users/USER/Desktop/project/cheonan-project/data_KHJ/연령별인구현황(2022.7월)-게시용.csv",
    r"C:/Users/USER/Desktop/project/cheonan-project/data_KHJ/연령별인구현황(2023.7월)-게시용.csv",
    r"C:/Users/USER/Desktop/project/cheonan-project/data_KHJ/연령별인구현황(2024.7월)-게시용.csv",
    r"C:/Users/USER/Desktop/project/cheonan-project/data_KHJ/연령별인구현황(2025.7월)-게시용.csv",
]
upload_files = [
    "/mnt/data/연령별인구현황(2021.7월)-게시용.csv",
    "/mnt/data/연령별인구현황(2022.7월)-게시용.csv",
    "/mnt/data/연령별인구현황(2023.7월)-게시용.csv",
    "/mnt/data/연령별인구현황(2024.7월)-게시용.csv",
    "/mnt/data/연령별인구현황(2025.7월)-게시용.csv",
]
def pick_path(local, uploaded):
    return local if os.path.exists(local) else uploaded

files = [pick_path(l, u) for l, u in zip(local_files, upload_files)]

# 전국 파일 (충청북도 제외)
path_local_k = r"C:/Users/USER/Desktop/project/cheonan-project/data_KHJ/고령인구비율_시도_시_군_구__2021_2025.csv"
path_up_k    = "/mnt/data/고령인구비율_시도_시_군_구__2021_2025.csv"
csv_k = pick_path(path_local_k, path_up_k)

# =========================
# 2) 공통 유틸
# =========================
def read_csv_kr(path):
    for enc in ("cp949", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def clean_columns(df):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out

def to_num(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace(",", "", regex=False)
              .str.replace("%", "", regex=False)
              .str.replace("−", "-", regex=False)
              .str.replace(r"[^\d\.\-]", "", regex=True)
              .replace({"": np.nan, "nan": np.nan})
              .astype(float))

# =========================
# 3) (A) 읍/면·천안시·나머지: 65세 이상 비율(%) 계산
# =========================
def pick_region_col(df):
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    patt = re.compile(r"(천안시|읍|면|동)")
    best, score = None, -1
    for c in obj_cols:
        sc = df[c].astype(str).str.contains(patt).sum()
        if sc > score:
            best, score = c, sc
    return best

def find_cols_for_ratio(df):
    cols = list(df.columns)
    senior = next((c for c in cols if re.search(r"65\s*세\s*이상", str(c))), None)
    total  = next((c for c in cols if re.search(r"(총계|총인구|^계$|합계|전체)", str(c))), None)
    age_cols = [c for c in cols if re.search(r"\d+\s*세", str(c)) or re.search(r"\d+\s*~\s*\d+\s*세", str(c))]
    if senior is None:
        patt_65p = re.compile(r"(65\s*세\s*이상|65\s*~\s*69\s*세|65\s*-\s*69\s*세|70|75|80|85|90|95|100)")
        senior_cols = [c for c in age_cols if re.search(patt_65p, str(c))]
        if senior_cols: senior = senior_cols   # list → 합산
    if total is None and age_cols:
        total = age_cols                       # list → 합산
    return senior, total

def compute_3groups_ratio_from_cols(df_raw):
    df = clean_columns(df_raw)
    region_col = pick_region_col(df)
    if region_col is None:
        raise RuntimeError("지역(행정구역) 열을 찾지 못했습니다.")

    df[region_col] = df[region_col].astype(str).str.strip()
    mask_lowest = df[region_col].str.contains(r"(읍|면|동)")
    sub = df[mask_lowest].copy()
    if sub.empty:
        raise RuntimeError("읍/면/동 단위 행을 찾지 못했습니다.")

    senior_col, total_col = find_cols_for_ratio(sub)
    if senior_col is None or total_col is None:
        raise RuntimeError("'65세 이상' 또는 '총계' 열을 찾지 못했습니다.")

    def series_from(col):
        if isinstance(col, list):
            return to_num(sub[col]).sum(axis=1, numeric_only=True)
        return to_num(sub[col])

    senior = series_from(senior_col).fillna(0)
    total  = series_from(total_col).fillna(0)

    # 그룹 라벨: 읍/면, 나머지(=동 등)
    grp = np.where(sub[region_col].str.contains(r"(읍|면)"), "읍/면", "동/구")
    grouped = pd.DataFrame({"그룹": grp, "총계": total.values, "65세 이상": senior.values})
    grp_sum = grouped.groupby("그룹", as_index=False)[["총계","65세 이상"]].sum()
    grp_sum["65세 이상 비율(%)"] = (grp_sum["65세 이상"]/grp_sum["총계"]*100).round(2)

    # 천안시(전체) = 모든 최하위 합
    city_ratio = round(float(senior.sum()/total.sum()*100), 2)

    out = {"연도": None, "읍/면": np.nan, "천안시": city_ratio, "나머지": np.nan}
    for _, row in grp_sum.iterrows():
        out[row["그룹"]] = row["65세 이상 비율(%)"]
    return out

# 연도별 집계
records = []
for path in files:
    year = int(re.search(r"(\d{4})", os.path.basename(path)).group(1))
    df_raw = read_csv_kr(path)
    ratios = compute_3groups_ratio_from_cols(df_raw)
    ratios["연도"] = year
    records.append(ratios)

trend_df = (pd.DataFrame(records)
              .sort_values("연도")
              .reset_index(drop=True))

# =========================
# 4) (B) 전국: 고령인구비율(%) 추출  ← 충청북도 제외
# =========================
def pick_region_column(df):
    obj_cols = [c for c in df.columns if df[c].dtype == 'object']
    patt = re.compile(r"(전국|충청북도|충북|시도|시군구|행정구역|지역|자치)")
    best, score = None, -1
    for c in obj_cols:
        sc = df[c].astype(str).str.contains(patt).sum()
        if sc > score: best, score = c, sc
    return best or (obj_cols[0] if obj_cols else None)

def find_indicator_column(df):
    obj_cols = [c for c in df.columns if df[c].dtype == 'object']
    for c in obj_cols:
        if df[c].astype(str).str.contains(r"고령\s*인구\s*비율|고령인구비율").any():
            return c
    return None

def extract_year_cols(df):
    year_map = {}
    for c in df.columns:
        m = re.search(r"(20\d{2})", str(c))
        if m:
            y = int(m.group(1))
            if 2021 <= y <= 2025:
                if y not in year_map:
                    year_map[y] = c
                else:
                    if re.search(r"비율|고령", str(c)) and not re.search(r"비율|고령", str(year_map[y])):
                        year_map[y] = c
    return year_map

# 읽기
dfk_raw = read_csv_kr(csv_k)
dfk = clean_columns(dfk_raw)

region_k = pick_region_column(dfk)
dfk[region_k] = dfk[region_k].astype(str).str.strip().replace({"충북":"충청북도"})

# ▼ 충청북도 제외: 전국만 선택
dfk2 = dfk[dfk[region_k].isin(["전국"])].copy()
if dfk2.empty:
    raise RuntimeError(f"'{region_k}' 열에서 '전국' 행을 찾지 못했습니다.")

# 지표행 필터(필요 시)
ind_col = find_indicator_column(dfk2)
if ind_col is not None:
    dfk2 = dfk2[dfk2[ind_col].astype(str).str.contains(r"고령\s*인구\s*비율|고령인구비율")].copy()

# wide/long 대응
tidy_k = None
ymap = extract_year_cols(dfk2)
if ymap:
    rows = []
    for y, c in sorted(ymap.items()):
        vals = to_num(dfk2[c])
        for name, v in zip(dfk2[region_k].values, vals.values):
            rows.append({"지역": name, "연도": y, "고령인구비율": v})
    tidy_k = pd.DataFrame(rows)
else:
    ratio_col = next((c for c in dfk2.columns if re.search(r"고령\s*인구\s*비율|고령인구비율", str(c))), None)
    year_col  = next((c for c in dfk2.columns if re.search(r"연도|시점|기간|기준연도|년도", str(c))), None)
    if ratio_col and year_col:
        sub = dfk2[[region_k, year_col, ratio_col]].copy()
        sub.columns = ["지역","연도","고령인구비율"]
        sub["연도"] = sub["연도"].astype(str).str.extract(r"(20\d{2})", expand=False).astype(float)
        sub = sub.dropna(subset=["연도"])
        sub["연도"] = sub["연도"].astype(int)
        sub["고령인구비율"] = to_num(sub["고령인구비율"])
        tidy_k = sub

if tidy_k is None or tidy_k.empty:
    # 마지막 fallback: 2021~2025가 정확히 열명인 경우
    explicit = [str(y) for y in range(2021,2026)]
    if all(y in dfk2.columns for y in explicit):
        sub = dfk2[[region_k]+explicit].copy()
        m = sub.melt(id_vars=region_k, var_name="연도", value_name="고령인구비율")
        m["지역"] = m[region_k]
        m["연도"] = m["연도"].astype(int)
        m["고령인구비율"] = to_num(m["고령인구비율"])
        tidy_k = m[["지역","연도","고령인구비율"]]
    else:
        raise RuntimeError("연도별 '고령인구비율' 열을 찾지 못했습니다.")

# 전국 wide (충북 컬럼은 처음부터 없음. 혹시 모를 잔존 컬럼 제거 안전장치)
tidy_k = (tidy_k.groupby(["지역","연도"], as_index=False)["고령인구비율"].mean())
wide_k  = tidy_k.pivot(index="연도", columns="지역", values="고령인구비율").reset_index()
wide_k = wide_k.rename_axis(None, axis=1)
wide_k = wide_k.drop(columns=["충청북도"], errors="ignore")  # 안전장치

# =========================
# 5) (C) 병합 및 시각화 — Plotly 2분할(좌: 전국·천안시 / 우: 천안시 점선 + 읍/면·동/구 실선)
# =========================
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

if 'VSCODE_PID' in os.environ:
    pio.renderers.default = 'vscode'

# 병합
merged = trend_df.merge(wide_k, on="연도", how="left")
merged = merged.drop(columns=["충청북도"], errors="ignore")

x = merged["연도"].astype(int).values

series_order = ["전국", "천안시", "읍/면", "동/구"]
present = [s for s in series_order if s in merged.columns]

colors = {
    "읍/면": "#d62728",
    "천안시": "#17A549",
    "동/구": "#0d1ceb",
    "전국": "#090a09",
}

def yvals(name):
    return pd.to_numeric(merged[name], errors="coerce")

# ==== y축 고정 범위 ====
vals = []
for n in present:
    arr = yvals(n).to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size: vals.append(arr)
if vals:
    all_vals = np.concatenate(vals)
    y_min, y_max = float(all_vals.min()), float(all_vals.max())
else:
    y_min, y_max = 0.0, 100.0
pad = max((y_max - y_min) * 0.08, 0.5)
y_range = [y_min - pad * 0.2, y_max + pad]

# ==== 2분할 서브플롯 ====
fig = make_subplots(
    rows=1, cols=2, shared_xaxes=True, shared_yaxes=True,
    subplot_titles=("전국 · 천안시", "읍/면 · 동/구  (천안시=점선)")
)

# 좌측: 전국, 천안시(실선)
if "전국" in present:
    fig.add_trace(
        go.Scatter(x=x, y=yvals("전국"), mode="lines+markers",
                   name="전국", line=dict(width=3, color=colors["전국"]),
                   marker=dict(size=8), showlegend=True),
        row=1, col=1
    )
if "천안시" in present:
    fig.add_trace(
        go.Scatter(x=x, y=yvals("천안시"), mode="lines+markers",
                   name="천안시", line=dict(width=3, color=colors["천안시"]),
                   marker=dict(size=8), showlegend=True),
        row=1, col=1
    )

# 우측: 천안시(점선), 읍/면·동/구(실선)
if "천안시" in present:
    fig.add_trace(
        go.Scatter(x=x, y=yvals("천안시"), mode="lines+markers",
                   name="천안시(점선)", line=dict(width=3, color=colors["천안시"], dash="dash"),
                   marker=dict(size=8), showlegend=False),   # ← 주석(범례)에서 제외
        row=1, col=2
    )
if "읍/면" in present:
    fig.add_trace(
        go.Scatter(x=x, y=yvals("읍/면"), mode="lines+markers",
                   name="읍/면", line=dict(width=3, color=colors["읍/면"]),
                   marker=dict(size=8), showlegend=True),
        row=1, col=2
    )
if "동/구" in present:
    fig.add_trace(
        go.Scatter(x=x, y=yvals("동/구"), mode="lines+markers",
                   name="동/구", line=dict(width=3, color=colors["동/구"]),
                   marker=dict(size=8), showlegend=True),
        row=1, col=2
    )

# ── 범례를 '오른쪽 그래프의 연도 아래'로 배치 ──
x2_dom = fig.layout.xaxis2.domain if fig.layout.xaxis2.domain is not None else [0.5, 1.0]
legend_x = (x2_dom[0] + x2_dom[1]) / 2.0  # 오른쪽 서브플롯 중앙
legend_y = -0.28                           # x축 제목 아래

fig.update_layout(
    title="65세 이상 비율 추이, 2021~2025",
    title_x=0.5,  # 가운데
    margin=dict(l=40, r=40, t=70, b=140),
    legend=dict(
        orientation="h",
        x=legend_x, xanchor="center",
        y=legend_y, yanchor="top",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1
    ),
    height=520
)

# 축
fig.update_xaxes(dtick=1, row=1, col=1, title_text="연도")
fig.update_xaxes(dtick=1, row=1, col=2, title_text="연도")
fig.update_yaxes(title_text="65세 이상 비율(%)", autorange=False, range=y_range,
                 gridcolor="rgba(0,0,0,0.2)", row=1, col=1)
fig.update_yaxes(autorange=False, range=y_range, gridcolor="rgba(0,0,0,0.2)", row=1, col=2)

fig.show()

# <두 번째 시각화> 2025년 7월 천안 지역 별 65세 이상 인구 현황 [309~438]
import pandas as pd
import numpy as np
from pathlib import Path

IN_DIR  = Path("C:/Users/USER/Desktop/project/cheonan-project/data_KHJ")
OUT_DIR = Path("./data_KHJ")
SHEET   = "4.연령별(10세 등)"
YEARS   = range(2021, 2026)  # 2021~2025

OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 열 이름 공백 제거
    df.columns = df.columns.str.strip()
    # '연령' → '지역명'
    df = df.rename(columns={"연령": "지역명"}).reset_index(drop=True)
    # 'Unnamed: 1' → '구분' (없으면 첫 Unnamed 계열을 '구분'으로)
    if "Unnamed: 1" in df.columns:
        df = df.rename(columns={"Unnamed: 1": "구분"})
    else:
        cand = [c for c in df.columns if str(c).startswith("Unnamed")]
        if cand:
            df = df.rename(columns={cand[0]: "구분"})
    # 3행(계/남/여) 블록별 지역명 동일화
    df["지역명"] = df["지역명"].where(df.index % 3 == 0).ffill()
    return df

def select_needed_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 요청: 지역명, 구분, 총계, 65세이상 만 남기기
    return df.loc[:, ["지역명", "구분", "총계", "65세이상"]]

def drop_gender_rows(df: pd.DataFrame) -> pd.DataFrame:
    # 1) '구분' == '남' 또는 '여' 제거
    return df[~df["구분"].isin(["남", "여"])].reset_index(drop=True)

def add_ratio_column(df: pd.DataFrame) -> pd.DataFrame:
    # 2) '65세이상' / '총계' → '65세 이상 비율' (비율)
    # 숫자 변환 (문자/콤마 대비)
    for col in ["총계", "65세이상"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["65세 이상 비율"] = df["65세이상"] / df["총계"]
    # 0으로 나눔/결측 처리
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# 연도별 처리
for y in YEARS:
    xlsx_path = IN_DIR / f"연령별인구현황({y}.7월)-게시용.xlsx"
    df = pd.read_excel(xlsx_path, sheet_name=SHEET, skiprows=2, engine="openpyxl")
    df = clean(df)
    df = select_needed_columns(df)
    df = drop_gender_rows(df)     # ← 남/여 행 삭제
    df = add_ratio_column(df)     # ← 65세 이상 비율 생성

    # CSV 저장
    csv_path = OUT_DIR / f"연령별인구현황({y}.7월)-게시용.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 메모리에 변수로 보관 (예: polulation2021_csv)
    globals()[f"polulation{y}_csv"] = df.copy()

################## [시각화] 65세 이상 주소지 별 ##################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 폰트 후보: Windows(맑은 고딕), macOS(AppleGothic), Linux(나눔고딕)
plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 0) 연도별 DF 결합 (이미 '남/여' 삭제·'65세 이상 비율' 생성 완료 상태 가정)
dfs = []
for y in range(2021, 2026):
    df = globals()[f"polulation{y}_csv"].copy()
    df["year"] = y
    dfs.append(df)
all_df = pd.concat(dfs, ignore_index=True)

# ───────────────────────────────────────────────────────────────
# A. 한 시점 비교: 정렬된 가로 막대 (예: 2025년)
# ───────────────────────────────────────────────────────────────
year_sel = 2025
plot_df = (all_df[all_df["year"] == year_sel]
           .sort_values("65세 이상 비율", ascending=False))

# 지역명에 따른 색상 매핑
colors = []
for name in plot_df["지역명"]:
    if name.endswith(("동", "구")):
        colors.append("lightblue")       # '동' 또는 '구' → 파란색
    elif name.endswith(("읍", "면")):
        colors.append("red")        # '읍' 또는 '면' → 빨간색
    else:
        colors.append("gray")      # 기타 → 검정색 (원하면 다른 색 지정 가능)

plt.figure(figsize=(9, max(4, len(plot_df)*0.35)))
plt.barh(plot_df["지역명"], plot_df["65세 이상 비율"], color=colors)
plt.gca().invert_yaxis()  # 상위가 위로 오도록
plt.title(f"천안 {year_sel}년 지역별 65세 이상 비율")
plt.xlabel("비율(%)")

# 퍼센트 라벨
for i, v in enumerate(plot_df["65세 이상 비율"].values):
    plt.text(v, i, f" {v*100:.1f}%", va='center')

plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────
# B. 연도 변화: 슬로프 차트 (상위 10개 지역만 예시) <추후에 활용>
# 기준 연도(예: 2025)에서 상위 10개 지역 선정
topN = 10
top_regions = (all_df[all_df["year"] == year_sel]
               .nlargest(topN, "65세 이상 비율")["지역명"].tolist())

trend_df = (all_df[all_df["지역명"].isin(top_regions)]
            .pivot_table(index="year", columns="지역명", values="65세 이상 비율"))

plt.figure(figsize=(9, 5))
for col in trend_df.columns:
    plt.plot(trend_df.index, trend_df[col], marker='o', linewidth=1.5, label=col)
plt.title("상위 지역 65세 이상 비율 추이")
plt.ylabel("비율(%)")
plt.xlabel("연도")
plt.xticks(sorted(trend_df.index))
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
# ───────────────────────────────────────────────────────────────

########### <세 번째 시각화> 평소 외출 시 이용 교통수단 [440~703]
# =========================
# 0) Import & Windows 폰트
# =========================
import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

# --- Add: 스크롤 컨테이너(그룹 모드) 도우미 ---
from io import BytesIO
import base64
from IPython.display import HTML, display

# 그룹 모드 전역 상태
GROUP_SCROLL = False          # 실행부에서 True로 켭니다
_SCROLL_ITEMS = []            # 이미지(<img>) HTML을 임시 저장

def _fig_to_img_tag(dpi=160):
    """현재 활성 figure를 PNG(base64) <img> 태그로 반환"""
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close()
    b64 = base64.b64encode(buf.getvalue()).decode()
    # 각 이미지 사이 간격을 12px로 조금 줄임
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto; display:block; margin:0 0 12px 0;"/>'

def show_scrollable(height_px=600, dpi=160):
    """
    단일 모드: 바로 스크롤 박스로 출력
    그룹 모드(GROUP_SCROLL=True): 이미지만 모아두었다가 display_group_scrollbox()에서 한 번에 출력
    """
    img_tag = _fig_to_img_tag(dpi=dpi)
    if GROUP_SCROLL:
        _SCROLL_ITEMS.append(img_tag)   # 박스 없이 이미지 태그만 저장
    else:
        html = f"""
<div style="max-width:100%; height:{height_px}px; overflow:auto;
            border:1px solid #e5e7eb; border-radius:10px; padding:6px; background:#fff;">
  {img_tag}
</div>
"""
        display(HTML(html))

def display_group_scrollbox(height_px=800):
    """모아둔 이미지를 하나의 스크롤 박스에 세로 나열해 출력"""
    if not _SCROLL_ITEMS:
        return
    html = f"""
<div style="max-width:100%; height:{height_px}px; overflow:auto;
            border:1px solid #e5e7eb; border-radius:10px; padding:6px; background:#fff;">
  {''.join(_SCROLL_ITEMS)}
</div>
"""
    display(HTML(html))
    _SCROLL_ITEMS.clear()

# Windows 한글 폰트 등록 + 적용
for fp in [r"C:\Windows\Fonts\malgun.ttf",
           r"C:\Windows\Fonts\malgunsl.ttf",
           r"C:\Windows\Fonts\gulim.ttc",
           r"C:\Windows\Fonts\batang.ttc"]:
    if os.path.exists(fp):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1) 데이터 로드 (이미 dat1 있으면 건너뜀)
# =========================
try:
    dat1
except NameError:
    dat1 = pd.read_csv(
        r"C:\Users\USER\Desktop\project\cheonan-project\data_KHJ\평소_외출_시_이용_교통수단.csv",
        encoding="cp949"
    )

# =========================
# 2) 유틸 함수
# =========================
def to_numeric_series(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace(",", "", regex=False)
              .str.replace("%", "", regex=False)
              .str.replace("−", "-", regex=False)
              .str.replace(r"[^\d\.\-]", "", regex=True)
              .replace("", np.nan)
              .astype(float))

def norm_text(s: str) -> str:
    s = str(s)
    s = re.sub(r"\(.*?\)", "", s)  # 괄호 내용 제거
    s = s.replace(" ", "").strip()
    return s

# =========================
# 3) 멀티헤더 구성 (상단 2행)
# =========================
df0 = dat1.copy()
df0.columns = [str(c).strip() for c in df0.columns]
for c in ["특성별(1)", "특성별(2)"]:
    df0[c] = df0[c].astype(str).str.strip()

top = df0.iloc[0, 2:].astype(str).str.strip().replace("", np.nan).ffill()  # '주된 응답' 등
sub = df0.iloc[1, 2:].astype(str).str.strip().replace("", np.nan).ffill()

# 소항목의 괄호 제거: '버스(%)' → '버스'
sub_clean = (sub.str.replace(r"\s*\(.*?\)\s*", "", regex=True)
                 .str.replace(r"\s+", " ", regex=True)
                 .str.strip())

mi = pd.MultiIndex.from_arrays(
    [pd.Index(["기준", "기준"], name="응답유형").append(pd.Index(top, name="응답유형")),
     pd.Index(["특성별(1)", "특성별(2)"], name="항목").append(pd.Index(sub_clean, name="항목"))]
)

data = df0.iloc[2:].reset_index(drop=True)
data.columns = mi

# =========================
# 4) 행/열 필터
# =========================
# 행: 읍면동별, 연령별만
data[("기준", "특성별(1)")] = data[("기준", "특성별(1)")].replace("", np.nan).ffill()
data = data[data[("기준", "특성별(1)")].isin(["읍면동별", "연령별"])].copy()

# 열: '주된 응답'만
main_cols = [c for c in data.columns if isinstance(c, tuple) and c[0] == "주된 응답"]
base_cols = [("기준", "특성별(1)"), ("기준", "특성별(2)")]
data_main = data.loc[:, base_cols + main_cols].copy()

# 모든 '주된 응답' 열을 숫자형으로 변환
for c in main_cols:
    data_main[c] = to_numeric_series(data_main[c])

# =========================
# 5) 시각화용 데이터 생성
# =========================
def make_eup_views(src: pd.DataFrame):
    tmp = src[src[("기준", "특성별(1)")] == "읍면동별"].copy()
    tmp_index = tmp[("기준", "특성별(2)")].astype(str).str.strip().rename("분류").reset_index(drop=True)
    value_cols = [c for c in tmp.columns if isinstance(c, tuple) and c[0] == "주된 응답"]
    values = pd.concat([tmp[c].reset_index(drop=True) for c in value_cols], axis=1)
    values.columns = [c[1] for c in value_cols]
    eup_all_wide = pd.concat([tmp_index, values], axis=1)
    eup_all_long = eup_all_wide.melt(id_vars="분류", var_name="항목", value_name="비율(%)")
    return eup_all_wide, eup_all_long

def make_age_views(src: pd.DataFrame):
    tmp = src[src[("기준", "특성별(1)")] == "연령별"].copy()
    idx = tmp[("기준", "특성별(2)")].astype(str).str.strip().rename("분류").reset_index(drop=True)
    candidates = [c for c in tmp.columns if isinstance(c, tuple) and c[0] == "주된 응답"]
    cand_norm = {c: norm_text(c[1]) for c in candidates}
    bus_col = next((c for c, n in cand_norm.items() if "버스" in n), None)
    walk_col = next((c for c, n in cand_norm.items() if ("도보" in n or "걸어서" in n)), None)
    if bus_col is None or walk_col is None:
        raise RuntimeError(f"[연령별] 버스/걸어서(도보) 열을 찾지 못했습니다. 실제 소항목: {[c[1] for c in candidates]}")
    vals = pd.concat([tmp[bus_col].reset_index(drop=True),
                      tmp[walk_col].reset_index(drop=True)], axis=1)
    vals.columns = ["버스", "걸어서/도보로"]
    age_2_wide = pd.concat([idx, vals], axis=1)
    age_2_long = age_2_wide.melt(id_vars="분류", var_name="항목", value_name="비율(%)")
    return age_2_wide, age_2_long

eup_all_wide, eup_all_long = make_eup_views(data_main)
age_2_wide, age_2_long     = make_age_views(data_main)

# =========================
# 5.5) 색상맵 정의 (순수 빨강 계조)
# =========================
BUS_CMAP = LinearSegmentedColormap.from_list(
    "pureRedOnly", ["#660000", "#990000", "#cc0000", "#ff0000"]
)
WALK_CMAP = cm.Greys

# =========================
# 6) 읍면동별 그래프
# =========================
def plot_eup_all_items_highlight(df_long: pd.DataFrame,
                                 title="읍면동별: 주된 응답(동부 vs 읍면부) — 버스/도보 강조"):
    df = df_long.copy()
    name_map = {
        '동지역': '동부', '동부(동)': '동부', '동부지역': '동부',
        '읍면지역': '읍면부', '읍·면부': '읍면부', '읍·면지역': '읍면부', '읍면동': '읍면부'
    }
    df['분류'] = df['분류'].astype(str).str.strip().replace(name_map)
    df = df[df['분류'].isin(['동부', '읍면부'])]

    items = df['항목'].drop_duplicates().tolist()
    pv = (df.pivot_table(index='분류', columns='항목', values='비율(%)', aggfunc='first')
            .reindex(index=['동부', '읍면부'])
            .reindex(columns=items)
            .fillna(0))

    x = np.arange(len(pv.index))
    n = pv.shape[1]
    full_width = 0.86
    width = full_width / max(n, 1)
    start = -full_width/2 + width/2

    def _bar_color(colname: str) -> str:
        if colname == '버스':
            return '#d62728'
        if colname == '걸어서/도보로':
            return '#000000'
        return '#BDBDBD'

    colors = [_bar_color(c) for c in pv.columns]

    # 높이 약 500px로 축소 (대략 96dpi 기준)
    plt.figure(figsize=(12, 500/96))
    for i, col in enumerate(pv.columns):
        offset = start + i*width
        plt.bar(x + offset, pv[col].values, width=width,
                label=col, color=colors[i], edgecolor='black', linewidth=0.4)

    plt.xticks(x, pv.index)
    plt.ylabel("비율(%)")
    plt.title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    bylabel = dict(zip(labels, handles))
    plt.legend(bylabel.values(), bylabel.keys(), title='항목', ncol=4, frameon=False)

    # 🔹 가운데 점선
    plt.axvline(0.5, color="#999999", linestyle="--", alpha=0.6)

    plt.tight_layout()
    show_scrollable(height_px=600, dpi=160)  # 개별 컨테이너 높이 축소

# =========================
# 7) 연령별 그래프
# =========================
def plot_age_bus_walk(df_age_long: pd.DataFrame,
                      title="연령별: 주된 응답(버스 vs 걸어서/도보로 — 좌/우 블록 배치)"):
    df = df_age_long.copy()
    df = df[df['항목'].isin(['버스', '걸어서/도보로'])]

    cats = df['분류'].astype(str).drop_duplicates().tolist()
    pv = (df.pivot(index='분류', columns='항목', values='비율(%)')
            .reindex(index=cats)
            .reindex(columns=['버스', '걸어서/도보로'])
            .fillna(0))

    bus_vals  = pv['버스'].values
    walk_vals = pv['걸어서/도보로'].values

    def norm01(arr):
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if vmax - vmin < 1e-12:
            return np.full_like(arr, 0.6, dtype=float)
        return (arr - vmin) / (vmax - vmin)
    bus_n  = norm01(bus_vals)
    walk_n = norm01(walk_vals)

    bus_colors  = [BUS_CMAP(0.2 + 0.7*(1 - t)) for t in bus_n]
    walk_colors = [WALK_CMAP(0.25 + 0.7*t) for t in walk_n]

    n = len(cats)
    gap = 1.0
    bus_x  = np.arange(n)
    walk_x = np.arange(n) + n + gap

    # 높이 약 500px
    plt.figure(figsize=(12, 500/96))
    plt.bar(bus_x,  bus_vals,  width=0.8, color=bus_colors,  edgecolor='black', linewidth=0.3, label='버스')
    plt.bar(walk_x, walk_vals, width=0.8, color=walk_colors, edgecolor='black', linewidth=0.3, label='걸어서/도보로')

    xticks = np.concatenate([bus_x, walk_x])
    xlabels = cats + cats
    plt.xticks(xticks, xlabels, rotation=0)

    # 🔹 가운데 점선
    sep_x = n - 0.5 + gap/2
    plt.axvline(sep_x, color="#999999", linestyle="--", alpha=0.6)

    plt.ylabel('비율(%)')
    plt.title(title)

    legend_handles = [
        Patch(facecolor=BUS_CMAP(0.2),  edgecolor='black', label='버스 (짙음=높음)'),
        Patch(facecolor=WALK_CMAP(0.85), edgecolor='black', label='걸어서/도보로'),
    ]
    plt.legend(handles=legend_handles, frameon=False)

    plt.tight_layout()
    show_scrollable(height_px=500, dpi=160)  # 개별 컨테이너 높이 축소

# =========================
# 8) 실행
# =========================

# 🔻 한 박스에 두 그래프를 모두 넣어 스크롤로 보기 (세로 배치)
GROUP_SCROLL = True
plot_eup_all_items_highlight(eup_all_long, "읍면동별: 주된 응답(동부 vs 읍면부)")
plot_age_bus_walk(age_2_long, "연령별: 주된 응답(버스 vs 걸어서/도보로)")
display_group_scrollbox(height_px=500)  # 최종 스크롤 박스 높이 축소
# GROUP_SCROLL = False  # (선택) 이후 다른 셀에서는 단일 모드로 복귀




# ==========================================
# <네 번째 시각화> 외출 시 불편한 점 — '없음'을 '있음(=100-없음)'으로 변환
# ==========================================

import os, re, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --------------------------
# 0) 한글 폰트
# --------------------------
if platform.system() == 'Windows':
    for fp in [r"C:\Windows\Fonts\malgun.ttf",
               r"C:\Windows\Fonts\malgunsl.ttf",
               r"C:\Windows\Fonts\gulim.ttc",
               r"C:\Windows\Fonts\batang.ttc"]:
        if os.path.exists(fp):
            try: fm.fontManager.addfont(fp)
            except: pass
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 1) 데이터 로드
# --------------------------
path_local  = r"C:\Users\USER\Desktop\project\cheonan-project\data_KHJ\평소_외출_시_불편한_점.csv"
path_upload = "/mnt/data/평소_외출_시_불편한_점.csv"
csv_path = path_local if os.path.exists(path_local) else path_upload

def read_csv_kr(path):
    for enc in ("cp949", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def to_numeric_series(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace(",", "", regex=False)
              .str.replace("%", "", regex=False)
              .str.replace("−", "-", regex=False)
              .str.replace(r"[^\d\.\-]", "", regex=True)
              .replace("", np.nan)
              .astype(float))

def is_multi_response(name: str) -> bool:
    n = str(name).replace(" ", "")
    return ("복수응답" in n) or ("복수" in n and "응답" in n)

def norm_label(s: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"\(.*?\)", "", str(s))).strip()

# --------------------------
# 2) 멀티헤더 처리
# --------------------------
df0 = read_csv_kr(csv_path).copy()
df0.columns = [str(c).strip() for c in df0.columns]
for c in ["특성별(1)", "특성별(2)"]:
    if c in df0.columns:
        df0[c] = df0[c].astype(str).str.strip()

if df0.shape[0] < 3:
    raise RuntimeError("파일 상단 2행(헤더) + 데이터가 필요합니다.")

top = df0.iloc[0, 2:].astype(str).str.strip().replace("", np.nan).ffill()
sub = df0.iloc[1, 2:].astype(str).str.strip().replace("", np.nan).ffill()
sub_clean = (sub.str.replace(r"\s*\(.*?\)\s*", "", regex=True)
                 .str.replace(r"\s+", " ", regex=True)
                 .str.strip())

mi = pd.MultiIndex.from_arrays(
    [pd.Index(["기준","기준"], name="응답유형").append(pd.Index(top, name="응답유형")),
     pd.Index(["특성별(1)","특성별(2)"], name="항목").append(pd.Index(sub_clean, name="항목"))]
)
data = df0.iloc[2:].reset_index(drop=True)
data.columns = mi

# --------------------------
# 3) 전체 '소계'만 추출
# --------------------------
data[("기준","특성별(1)")] = data[("기준","특성별(1)")].replace("", np.nan).ffill()
mask_total = data[("기준","특성별(2)")].astype(str).str.strip() == "소계"
data_total = data[mask_total].copy()
if data_total.empty:
    raise RuntimeError("'특성별(2)'에서 '소계' 행을 찾지 못했습니다.")

mr_cols = [c for c in data_total.columns if isinstance(c, tuple) and is_multi_response(c[0])]
if not mr_cols:
    raise RuntimeError("복수응답(%) 열을 찾지 못했습니다. 상단 2행(응답유형)에 '복수 응답'이 있는지 확인하세요.")
for c in mr_cols:
    data_total[c] = to_numeric_series(data_total[c])

# --------------------------
# 4) ‘있음/없음’ 제거 후 long 변환
# --------------------------
# (소계가 여러 행이면 평균 집계)
values = pd.concat([data_total[c] for c in mr_cols], axis=1).mean(axis=0).to_frame().T
values.columns = [c[1] for c in mr_cols]
wide_total = values.copy()

# 제거 대상: '불편한 점이 있음', '불편한 점이 없음'
drop_targets = []
for col in list(wide_total.columns):
    key = norm_label(col)
    if key in {norm_label("불편한 점이 있음"), norm_label("불편한 점이 없음")}:
        drop_targets.append(col)
wide_total = wide_total.drop(columns=list(set(drop_targets)), errors="ignore")

# long 변환
long_total = wide_total.T.reset_index()
long_total.columns = ["항목", "비율(%)"]
long_total["비율(%)"] = to_numeric_series(long_total["비율(%)"])
long_total = long_total.dropna(subset=["비율(%)"])
if long_total.empty:
    raise RuntimeError("시각화할 항목이 없습니다. (있음/없음만 있었거나 값이 모두 NaN)")

# 내림차순 정렬 후 barh 상단에 큰 값이 오도록 reverse
long_total = long_total.sort_values("비율(%)", ascending=False).reset_index(drop=True)
long_total = long_total[::-1]

# --------------------------
# 5) 시각화 (녹색 단색 + 라벨 겹침 방지)
# --------------------------
fig, ax = plt.subplots(figsize=(10, max(6, len(long_total) * 0.42)))
y = np.arange(len(long_total))

vals = long_total["비율(%)"].astype(float).values
vmax = float(np.nanmax(vals))
# 오른쪽 여백 확보(라벨이 바깥에 나가도 안 겹치도록)
ax.set_xlim(0, vmax * 1.12)

# 단색 녹색 막대
bar_color = "#1A5E1A"  # 녹색
ax.barh(y, vals, color=bar_color, edgecolor="black", linewidth=0.5)

# 값 라벨: 기본은 막대 오른쪽(바깥), 끝에 너무 가까우면 막대 안쪽으로 자동 이동
x_out_pad = max(0.3, vmax * 0.01)   # 바깥 패딩
x_in_pad  = max(0.3, vmax * 0.01)   # 안쪽 패딩
x_limit_right = ax.get_xlim()[1]
threshold = x_limit_right - (vmax * 0.03)  # 이 값보다 크면 안쪽으로

for i, v in enumerate(vals):
    if not np.isfinite(v): 
        continue
    x_out = v + x_out_pad
    if x_out > threshold:
        # 막대 안쪽에 흰색 글자, 오른쪽 정렬
        ax.text(v - x_in_pad, i, f"{v:.1f}%", va="center", ha="right",
                fontsize=9, color="white", clip_on=False)
    else:
        # 막대 바깥쪽에 검은 글자, 왼쪽 정렬
        ax.text(x_out, i, f"{v:.1f}%", va="center", ha="left",
                fontsize=9, color="black", clip_on=False)

# 축/제목/그리드
ax.set_yticks(y, labels=long_total["항목"])
ax.set_xlabel("비율(%)")
ax.set_title("외출 시 불편한 점 — 65세 이상")
ax.grid(axis="x", alpha=0.3, linestyle=":")

fig.tight_layout()
plt.show()

# (선택) 확인
print("▶ 시각화 대상 항목 수:", len(long_total))
print("▶ 상위 5개:\n", long_total[::-1].head(5))
