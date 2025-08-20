# -*- coding: utf-8 -*-
import pandas as pd
import plotly.express as px
from shiny import App, ui, render
from shinywidgets import output_widget, render_plotly

# ---------------------------
# 1) 데이터 불러오기
# ---------------------------
df = pd.read_csv("./data/천안시_노선별경유정류소목록.csv", encoding="utf-8-sig")

# ---------------------------
# 2) UI 정의
# ---------------------------
app_ui = ui.page_fluid(
    ui.panel_title("천안시 노선별 경유 정류소 시각화"),
    output_widget("map")   # 여기서 plotly 출력
)

# ---------------------------
# 3) 서버 로직
# ---------------------------
def server(input, output, session):
    @render_plotly
    def map():
        fig = px.scatter_map(
            df,
            lat="정류소 Y좌표",
            lon="정류소 X좌표",
            color="노선ID",
            hover_name="정류소명",
            zoom=11,
            height=650,
            map_style="carto-positron",  # 무료 타일
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return fig

# ---------------------------
# 4) Shiny App 객체
# ---------------------------
app = App(app_ui, server)
