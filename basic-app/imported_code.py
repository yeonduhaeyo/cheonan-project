# --- app.py 상단 import 근처에 추가 ---
import os, re, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from functools import lru_cache

# 안전한 폰트 문자열
def _font_family():
    mf = plt.rcParams.get('font.family', 'Malgun Gothic')
    if isinstance(mf, (list, tuple)):
        for x in mf:
            if isinstance(x, str) and x.strip():
                return x
        return 'Malgun Gothic'
    return mf or 'Malgun Gothic'

# OS별 폰트 등록(최초 1회)
if platform.system() == 'Windows':
    for fp in [r"C:/Windows/Fonts/malgun.ttf", r"C:/Windows/Fonts/malgunsl.ttf",
               r"C:/Windows/Fonts/gulim.ttc", r"C:/Windows/Fonts/batang.ttc"]:
        if os.path.exists(fp):
            try: fm.fontManager.addfont(fp)
            except: pass
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
FONT_FAMILY = _font_family()

# 파일 경로 선택 유틸
def _pick_path(local, uploaded):
    return local if os.path.exists(local) else uploaded

# ---- 여기 함수 안에 "긴 코드"를 그대로 옮겨 넣어 생성만 하게 만듭니다 ----
@lru_cache(maxsize=1)
def build_cheonan_senior_trend_html() -> str:
    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  당신이 보낸 '긴 코드'의 "데이터 읽기~fig 생성" 부분을 이 위치에 붙입니다  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    # (주의) showlegend는 False, annotation은 FONT_FAMILY 사용 유지

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
    files = [_pick_path(l, u) for l, u in zip(local_files, upload_files)]

    path_local_k = r"C:/Users/USER/Desktop/project/cheonan-project/data_KHJ/고령인구비율_시도_시_군_구__2021_2025.csv"
    path_up_k = "/mnt/data/고령인구비율_시도_시_군_구__2021_2025.csv"
    csv_k = _pick_path(path_local_k, path_up_k)

    def read_csv_kr(path):
        for enc in ("cp949", "utf-8-sig"):
            try: return pd.read_csv(path, encoding=enc)
            except Exception: pass
        return pd.read_csv(path)

    def clean_columns(df):
        out = df.copy()
        out.columns = [str(c).strip() for c in out.columns]
        return out

    def to_num(s: pd.Series) -> pd.Series:
        return (s.astype(str).str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("−", "-", regex=False)
                .str.replace(r"[^\d\.\-]", "", regex=True)
                .replace({"": np.nan, "nan": np.nan}).astype(float))

    def pick_region_col(df):
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        patt = re.compile(r"(천안시|읍|면|동)")
        best, score = None, -1
        for c in obj_cols:
            sc = df[c].astype(str).str.contains(patt).sum()
            if sc > score: best, score = c, sc
        return best

    def find_cols_for_ratio(df):
        cols = list(df.columns)
        senior = next((c for c in cols if re.search(r"65\s*세\s*이상", str(c))), None)
        total = next((c for c in cols if re.search(r"(총계|총인구|^계$|합계|전체)", str(c))), None)
        age_cols = [c for c in cols if re.search(r"\d+\s*세", str(c)) or re.search(r"\d+\s*~\s*\d+\s*세", str(c))]
        if senior is None:
            patt_65p = re.compile(r"(65\s*세\s*이상|65\s*~\s*69\s*세|65\s*-\s*69\s*세|70|75|80|85|90|95|100)")
            senior_cols = [c for c in age_cols if re.search(patt_65p, str(c))]
            if senior_cols: senior = senior_cols
        if total is None and age_cols:
            total = age_cols
        return senior, total

    def compute_3groups_ratio_from_cols(df_raw):
        df = clean_columns(df_raw)
        region_col = pick_region_col(df)
        if region_col is None: raise RuntimeError("지역(행정구역) 열을 찾지 못했습니다.")
        df[region_col] = df[region_col].astype(str).str.strip()
        mask_lowest = df[region_col].str.contains(r"(읍|면|동)")
        sub = df[mask_lowest].copy()
        if sub.empty: raise RuntimeError("읍/면/동 단위 행을 찾지 못했습니다.")
        senior_col, total_col = find_cols_for_ratio(sub)
        if senior_col is None or total_col is None: raise RuntimeError("'65세 이상' 또는 '총계' 열을 찾지 못했습니다.")
        def series_from(col):
            if isinstance(col, list): return to_num(sub[col]).sum(axis=1, numeric_only=True)
            return to_num(sub[col])
        senior = series_from(senior_col).fillna(0)
        total  = series_from(total_col).fillna(0)
        grp = np.where(sub[region_col].str.contains(r"(읍|면)"), "읍/면", "동/구")
        grouped = pd.DataFrame({"그룹": grp, "총계": total.values, "65세 이상": senior.values})
        grp_sum = grouped.groupby("그룹", as_index=False)[["총계","65세 이상"]].sum()
        grp_sum["65세 이상 비율(%)"] = (grp_sum["65세 이상"]/grp_sum["총계"]*100).round(2)
        city_ratio = round(float(senior.sum()/total.sum()*100), 2)
        out = {"연도": None, "읍/면": np.nan, "천안시": city_ratio, "나머지": np.nan}
        for _, row in grp_sum.iterrows():
            out[row["그룹"]] = row["65세 이상 비율(%)"]
        return out

    records = []
    for path in files:
        year = int(re.search(r"(\d{4})", os.path.basename(path)).group(1))
        df_raw = read_csv_kr(path)
        ratios = compute_3groups_ratio_from_cols(df_raw)
        ratios["연도"] = year
        records.append(ratios)
    trend_df = pd.DataFrame(records).sort_values("연도").reset_index(drop=True)

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

    dfk_raw = read_csv_kr(csv_k)
    dfk = clean_columns(dfk_raw)
    region_k = pick_region_column(dfk)
    dfk[region_k] = dfk[region_k].astype(str).str.strip().replace({"충북":"충청북도"})
    dfk2 = dfk[dfk[region_k].isin(["전국"])].copy()
    if dfk2.empty: raise RuntimeError(f"'{region_k}' 열에서 '전국' 행을 찾지 못했습니다.")
    ind_col = find_indicator_column(dfk2)
    if ind_col is not None:
        dfk2 = dfk2[dfk2[ind_col].astype(str).str.contains(r"고령\s*인구\s*비율|고령인구비율")].copy()

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
        year_col = next((c for c in dfk2.columns if re.search(r"연도|시점|기간|기준연도|년도", str(c))), None)
        if ratio_col and year_col:
            sub = dfk2[[region_k, year_col, ratio_col]].copy()
            sub.columns = ["지역","연도","고령인구비율"]
            sub["연도"] = sub["연도"].astype(str).str.extract(r"(20\d{2})", expand=False).astype(float)
            sub = sub.dropna(subset=["연도"])
            sub["연도"] = sub["연도"].astype(int)
            sub["고령인구비율"] = to_num(sub["고령인구비율"])
            tidy_k = sub
    if tidy_k is None or tidy_k.empty:
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

    tidy_k = tidy_k.groupby(["지역","연도"], as_index=False)["고령인구비율"].mean()
    wide_k  = tidy_k.pivot(index="연도", columns="지역", values="고령인구비율").reset_index()
    wide_k  = wide_k.rename_axis(None, axis=1).drop(columns=["충청북도"], errors="ignore")

    if 'VSCODE_PID' in os.environ:
        pio.renderers.default = 'vscode'

    merged = trend_df.merge(wide_k, on="연도", how="left").drop(columns=["충청북도"], errors="ignore")
    x = merged["연도"].astype(int).values
    series_order = ["전국", "천안시", "읍/면", "동/구"]
    present = [s for s in series_order if s in merged.columns]
    colors = {"읍/면": "#d62728", "천안시": "#17A549", "동/구": "#0d1ceb", "전국": "#090a09"}

    def yvals(name): return pd.to_numeric(merged[name], errors="coerce")
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

    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=("전국 · 천안시", "읍/면 · 동/구"))

    if "전국" in present:
        fig.add_trace(go.Scatter(x=x, y=yvals("전국"), mode="lines+markers",
                                 line=dict(width=3, color=colors["전국"]),
                                 marker=dict(size=8), showlegend=False), row=1, col=1)
    if "천안시" in present:
        fig.add_trace(go.Scatter(x=x, y=yvals("천안시"), mode="lines+markers",
                                 line=dict(width=3, color=colors["천안시"]),
                                 marker=dict(size=8), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=yvals("천안시"), mode="lines+markers",
                                 line=dict(width=3, color=colors["천안시"], dash="dash"),
                                 marker=dict(size=8), showlegend=False), row=1, col=2)
    if "읍/면" in present:
        fig.add_trace(go.Scatter(x=x, y=yvals("읍/면"), mode="lines+markers",
                                 line=dict(width=3, color=colors["읍/면"]),
                                 marker=dict(size=8), showlegend=False), row=1, col=2)
    if "동/구" in present:
        fig.add_trace(go.Scatter(x=x, y=yvals("동/구"), mode="lines+markers",
                                 line=dict(width=3, color=colors["동/구"]),
                                 marker=dict(size=8), showlegend=False), row=1, col=2)

    def add_label(trace_name, col, label_text=None):
        if trace_name not in present: return
        y = yvals(trace_name)
        xv = x[np.isfinite(y)]
        yv = y[np.isfinite(y)]
        if xv.size == 0: return
        xl = xv[-1]
        yl = yv.iloc[-1] if isinstance(yv, pd.Series) else yv[-1]
        dy = (y_range[1] - y_range[0]) * 0.03
        txt = label_text if label_text is not None else trace_name
        fig.add_annotation(x=xl, y=yl + dy, text=txt, xanchor="left", showarrow=False,
                           font=dict(color=colors[trace_name], size=12, family=FONT_FAMILY),
                           align="left", row=1, col=col)

    x2_dom = fig.layout.xaxis2.domain if fig.layout.xaxis2.domain is not None else [0.5, 1.0]
    legend_x = (x2_dom[0] + x2_dom[1]) / 2.0
    legend_y = -0.28

    fig.update_layout(
        title="65세 이상 비율 추이",
        title_x=0.5,
        margin=dict(l=40, r=40, t=70, b=140),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        legend=dict(orientation="h", x=legend_x, xanchor="center",
                    y=legend_y, yanchor="top",
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="rgba(0,0,0,0.2)", borderwidth=1),
        height=520,
        showlegend=False,
        font=dict(family=FONT_FAMILY)
    )

    fig.update_xaxes(showline=True, linecolor="black", linewidth=2, mirror=False)
    fig.update_yaxes(showline=True, linecolor="black", linewidth=2, mirror=False, row=1, col=1)
    fig.update_yaxes(showline=False, row=1, col=2)

    fig.layout.shapes = ()
    x1_dom = fig.layout.xaxis.domain
    x2_dom = fig.layout.xaxis2.domain
    y_dom  = fig.layout.yaxis.domain
    fig.add_shape(type="line", xref="paper", yref="paper", x0=x1_dom[0], x1=x1_dom[1], y0=y_dom[1], y1=y_dom[1], line=dict(color="black", width=2))
    fig.add_shape(type="line", xref="paper", yref="paper", x0=x1_dom[1], x1=x1_dom[1], y0=y_dom[0], y1=y_dom[1], line=dict(color="black", width=2))
    fig.add_shape(type="line", xref="paper", yref="paper", x0=x2_dom[0], x1=x2_dom[1], y0=y_dom[1], y1=y_dom[1], line=dict(color="black", width=2))
    fig.add_shape(type="line", xref="paper", yref="paper", x0=x2_dom[1], x1=x2_dom[1], y0=y_dom[0], y1=y_dom[1], line=dict(color="black", width=2))
    fig.add_shape(type="line", xref="paper", yref="paper", x0=x2_dom[0], x1=x2_dom[0], y0=y_dom[0], y1=y_dom[1], line=dict(color="black", width=2))

    fig.update_xaxes(dtick=1, row=1, col=1, title_text="연도")
    fig.update_xaxes(dtick=1, row=1, col=2, title_text="연도")
    fig.update_yaxes(title_text="65세 이상 비율(%)", autorange=False, range=y_range, gridcolor="rgba(0,0,0,0.2)", row=1, col=1)
    fig.update_yaxes(autorange=False, range=y_range, gridcolor="rgba(0,0,0,0.2)", row=1, col=2)

    add_label("전국", 1)
    add_label("천안시", 1)
    add_label("천안시", 2, label_text="천안시")
    add_label("읍/면", 2)
    add_label("동/구", 2)

    html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    return html
