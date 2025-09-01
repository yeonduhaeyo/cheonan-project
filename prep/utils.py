# prep/utils.py
import os, re, base64
from io import BytesIO
import numpy as np
import pandas as pd

# 한글 폰트 스택
KR_FONT_STACK = "Malgun Gothic, AppleGothic, Nanum Gothic, sans-serif"

def read_csv_kr(path):
    for enc in ("cp949", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in df.columns]
    return out

def to_num(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace(",", "", regex=False)
              .str.replace("%", "", regex=False)
              .str.replace("−", "-", regex=False)
              .str.replace(r"[^\d\.\-]", "", regex=True)
              .replace({"": np.nan, "nan": np.nan})
              .astype(float))

def fig_to_img_html(fig, dpi=150) -> str:
    from matplotlib import pyplot as plt
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"""
<div style="max-width:100%; border:1px solid #e5e7eb; border-radius:10px; padding:6px; background:#fff;">
  <img src="data:image/png;base64,{b64}" style="width:100%; height:auto; display:block;"/>
</div>
"""

def wrap_plotly_fragment(fragment: str, title: str = "시각화") -> str:
    return f"""<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root{{--kr-font:{KR_FONT_STACK};}}
html,body{{margin:0;padding:0;background:#fff;font-family:var(--kr-font);}}
body *,.plotly,.main-svg,.hoverlayer,.legend text,.gtitle,.xtick text,.ytick text,.annotation-text{{font-family:var(--kr-font)!important;}}
.wrap{{padding:6px;}}
</style>
</head><body><div class="wrap">{fragment}</div></body></html>"""

def norm_label(s: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"\(.*?\)", "", str(s))).strip()
