# viz/charts.py
from __future__ import annotations
import re, base64
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ⬇️ 모든 '읽기'는 shared.py 로 이동
from shared import (
    read_csv_kr,                       # (공유 유틸)
    load_disability_registered_raw,    # 충남 등록장애인 원본
    load_age_distribution_years,       # 연령별 인구현황(여러 연도)
)

KR_FONT_STACK = "Malgun Gothic, AppleGothic, Nanum Gothic, sans-serif"

# ---- 공통 유틸 (계산/시각화 전용) ----
def _wrap_plotly_fragment_with_kr_html(fragment: str, title: str) -> str:
    return f"""<!doctype html><html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>body{{margin:0;font-family:{KR_FONT_STACK}}} body *, .plotly, .main-svg,.legend text{{font-family:{KR_FONT_STACK}!important}}</style>
</head><body>{fragment}</body></html>"""

def _fig_to_img_html(fig, dpi=150) -> str:
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight"); plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;height:auto;display:block">'

def to_num(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.replace(",","",regex=False)
            .str.replace("%","",regex=False).str.replace("−","-",regex=False)
            .str.replace(r"[^\d\.\-]","",regex=True).replace({"":np.nan,"nan":np.nan}).astype(float))

# ---- 좌:충남 등록장애인 수(2024) ----
def disability_2024_df() -> pd.DataFrame:
    df = load_disability_registered_raw()
    df.columns = [str(c).strip() for c in df.columns]
    need = {"시군구별(1)", "시군구별(2)", "2024"}
    if not need.issubset(df.columns):
        raise KeyError(f"[charts] 컬럼 누락: {need - set(df.columns)}")

    sub = df[(df["시군구별(1)"]=="충청남도") & (df["시군구별(2)"]!="소계")].copy()
    sub["2024"] = pd.to_numeric(sub["2024"], errors="coerce")
    sub["시군구별(2)"] = sub["시군구별(2)"].replace({"천안시 동남구":"천안시","천안시 서북구":"천안시"})
    return (sub.groupby("시군구별(2)", as_index=False)["2024"].sum()
              .sort_values("2024", ascending=False).reset_index(drop=True))

# ---- 우: 65+ 비율 추이 ----
def _pick_region_col(df: pd.DataFrame) -> str | None:
    obj = [c for c in df.columns if df[c].dtype=="object"]
    pat = re.compile(r"(천안시|읍|면|동|구|행정구역|지역|자치)")
    best,score=None,-1
    for c in obj:
        sc = df[c].astype(str).str.contains(pat).sum()
        if sc>score: best,score=c,sc
    return best

def _find_cols_for_ratio(df: pd.DataFrame):
    cols = list(df.columns)
    senior = next((c for c in cols if re.search(r"65\s*세\s*이상", str(c))), None)
    age_cols = [c for c in cols if re.search(r"\d+\s*세|\d+\s*[~-]\s*\d+\s*세", str(c))]
    if senior is None:
        patt = re.compile(r"(65\s*세\s*이상|65\s*[~-]\s*69\s*세|70|75|80|85|90|95|100)")
        senior_cols = [c for c in age_cols if re.search(patt, str(c))]
        senior = senior_cols if senior_cols else None
    total = next((c for c in cols if re.search(r"(총계|총인구|^계$|합계|전체)", str(c))), None)
    if total is None and age_cols: total = age_cols
    return senior, total

def _series_from(df, col):
    if isinstance(col, list):
        return to_num(df[col]).sum(axis=1, numeric_only=True)
    return to_num(df[col])

def load_trend_ratios_df() -> pd.DataFrame:
    # ⬇️ shared에서 여러 연도 원본을 모두 로드
    yearly = load_age_distribution_years((2021, 2022, 2023, 2024, 2025))
    rec = []
    for year, df in sorted(yearly.items()):
        df.columns=[str(c).strip() for c in df.columns]
        region = _pick_region_col(df)
        if region is None:
            continue
        sub = df[df[region].astype(str).str.contains(r"(읍|면|동|구)")].copy()
        senior,total = _find_cols_for_ratio(sub)
        if senior is None or total is None:
            continue
        s = _series_from(sub, senior).fillna(0); t = _series_from(sub, total).fillna(0)
        grp_flag = np.where(sub[region].str.contains(r"(읍|면)"), "읍/면", "동/구")
        tmp = pd.DataFrame({"그룹": grp_flag, "총계": t, "65세 이상": s})
        g = tmp.groupby("그룹", as_index=False)[["총계","65세 이상"]].sum()
        g["65세 이상 비율(%)"] = (g["65세 이상"]/g["총계"]*100).round(2)
        city_ratio = round(float(s.sum()/t.sum()*100), 2) if t.sum()>0 else np.nan
        row = {"연도": year, "천안시": city_ratio, "읍/면": np.nan, "동/구": np.nan}
        for _,r in g.iterrows():
            row[r["그룹"]] = r["65세 이상 비율(%)"]
        rec.append(row)
    if not rec:
        raise RuntimeError("[charts] 연령별인구현황 CSV를 찾지 못했습니다.")
    return pd.DataFrame(rec).sort_values("연도").reset_index(drop=True)

# ---- 메인 도표 HTML ----
def dis_old_rat(title="천안시 노인 비율 증가 추이", height=520, as_fragment=False) -> str:
    grp = disability_2024_df()
    trend = load_trend_ratios_df()
    # (왼) 막대
    bar = go.Bar(x=grp["시군구별(2)"], y=grp["2024"], name="등록장애인수", marker_color="#a6a6a6")
    # (오) 선
    x = trend["연도"].astype(int)
    color_map = {"읍/면":"#d62728","천안시":"#17A549","동/구":"#0d1ceb"}
    lines = []
    for k in ["천안시","읍/면","동/구"]:
        if k in trend.columns:
            dash = "dash" if k=="천안시" else "solid"
            lines.append(go.Scatter(x=x, y=pd.to_numeric(trend[k], errors="coerce"),
                                    mode="lines+markers", name=k,
                                    line=dict(width=3, color=color_map[k], dash=dash),
                                    marker=dict(size=8)))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("충남 2024 등록장애인 수", "읍/면·동/구 65세 이상 비율 추이"),
                        horizontal_spacing=0.12)
    fig.add_trace(bar, row=1, col=1)
    for tr in lines: fig.add_trace(tr, row=1, col=2)
    fig.update_xaxes(title_text="시군구", row=1, col=1, tickangle=45, showline=True, linecolor="black")
    fig.update_yaxes(title_text="명", row=1, col=1, showline=True, linecolor="black")
    fig.update_xaxes(title_text="연도", row=1, col=2, dtick=1, showline=True, linecolor="black")
    fig.update_yaxes(title_text="65세 이상 비율(%)", row=1, col=2, showline=True, linecolor="black",
                     gridcolor="rgba(0,0,0,0.15)")
    fig.update_layout(title=title, title_x=0.5, height=height, plot_bgcolor="#fff", paper_bgcolor="#fff",
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15, yanchor="top"))
    frag = fig.to_html(full_html=False, include_plotlyjs="cdn")
    return frag if as_fragment else _wrap_plotly_fragment_with_kr_html(frag, title=title)
