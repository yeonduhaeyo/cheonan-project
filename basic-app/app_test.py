# app_test.py
from shiny import App, ui, render, reactive
import pandas as pd
import geopandas as gpd
from shinyswatch import theme
import folium
import plotly.express as px
import os
import matplotlib.ticker as ticker
import plotly.graph_objs as go

# --- 외부 시각화 코드 임포트 (./basic-app/imported_code.py) ---
from imported_code import build_cheonan_senior_trend_html

# --------------------------
# 더미 데이터(연결 전 임시)
# --------------------------
df = pd.DataFrame(columns=["읍면동", "건물명", "점수"])
df_population = pd.DataFrame()
region_list = ["지역1", "지역2", "지역3"]
stations_filtered = pd.DataFrame()

# --------------------------
# 1) UI
# --------------------------
def app_ui(request):
    return ui.page_fluid(
        ui.tags.head(
            ui.tags.link(
                href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.2/dist/minty/bootstrap.min.css",
                rel="stylesheet"
            ),
            ui.tags.style(
                """
                .scroll-box {
                    max-height: 600px;
                    overflow-y: auto;
                    border: 1px solid #ccc;
                    padding: 10px;
                }
                table { font-size: 14px; line-height: 1.6; table-layout: fixed; width: 100%; }
                th { white-space: nowrap !important; text-align: center; background-color: #f8f9fa; }
                td { vertical-align: top; white-space: normal !important; padding: 8px; }
                .card { border: none; }

                .navbar, .navbar.bg-primary, .navbar-dark, .navbar-light {
                    background-color: #90ee90 !important;
                    border-color: #90ee90 !important;
                }
                .nav-link, .navbar-brand { color: #084c2e !important; }
                .nav-link.active, .nav-link:focus, .nav-link:hover {
                    color: #063a24 !important;
                    text-decoration: none !important;
                }

                .btn-primary, .btn-danger, .btn-success {
                    background-color: #90ee90 !important;
                    border-color: #90ee90 !important;
                    color: #063a24 !important;
                }
                .btn-primary:hover, .btn-danger:hover, .btn-success:hover {
                    background-color: #76d476 !important;
                    border-color: #76d476 !important;
                    color: #042818 !important;
                }

                .form-range::-webkit-slider-thumb { background-color: #90ee90 !important; }
                .form-check-input:checked { background-color: #90ee90 !important; border-color: #90ee90 !important; }

                .card-header { border-bottom: 2px solid #90ee90 !important; }
                """
            )
        ),

        ui.page_navbar(
            ui.nav_panel("HOME",
                ui.card(
                    ui.card_header("사용자 가중치 설정"),
                    ui.layout_columns(
                        ui.input_slider("w0", "① 건물연차 점수", min=0, max=25, value=25),
                        ui.input_slider("w1", "② 지상층수",   min=0, max=25, value=9),
                        ui.input_slider("w2", "③ 지하층수",   min=0, max=25, value=11),
                        ui.input_slider("w3", "④ 비상용 승강기", min=0, max=25, value=5),
                    ),
                    ui.card(
                        ui.card_header("사용자 설정 기반 취약 점수 지도 및 건물 목록"),
                        ui.output_ui("show_score_map2"),
                        full_screen=True
                    )
                ),
            ),

            ui.nav_panel("건물 취약도 분석",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_checkbox_group("region", "행정동 선택", choices=region_list, selected=region_list),
                        ui.input_action_button("apply_filter", "적용", style="background-color: #90ee90; color: #063a24;"),
                        title="필터 설정"
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("사용자 설정에 따른 건물 분포 시각화"),
                            ui.output_ui("show_filtered_building_map"),
                            full_screen=True
                        ),
                        ui.card(
                            ui.card_header("전체 건물 취약 점수 분포"),
                            ui.output_plot("top_bottom_histogram"),
                            ui.output_data_frame("show_summary"),
                            full_screen=True
                        )
                    )
                )
            ),

            ui.nav_panel("부록1(시각화)",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("천안시 노인 비율 증가 추이"),
                        ui.output_ui("show_building_age_bar"),
                        full_screen=True
                    ),
                    ui.card(
                        ui.card_header("소화전 거리 분포"),
                        ui.output_ui("show_firehydrant_distance_plot")
                    )
                )
            ),

            ui.nav_panel("부록2(기준 및 설명)",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("변수 정의"),
                        ui.output_ui("show_data_table")
                    ),
                    ui.card(
                        ui.card_header("데이터 설명"),
                        ui.output_ui("show_variable_table")
                    )
                )
            ),

            title=" 천안 🏠 복지시설 🚌 대중교통 접근성 분석",
            theme=theme.minty
        )
    )

# --------------------------
# 2) Server
# --------------------------
def server(input, output, session):
    @reactive.calc
    def selected_regions():
        return input.region()

    @reactive.calc
    def filtered_building_df():
        if df.empty:
            return pd.DataFrame(columns=df.columns)
        return df[df["읍면동"].isin(selected_regions())]

    # ▷ 부록1(시각화) ▸ 천안시 노인 비율 증가 추이
    @output
    @render.ui
    def show_building_age_bar():
        try:
            return ui.HTML(build_cheonan_senior_trend_html())
        except Exception as e:
            return ui.HTML(f"<div style='color:#b00020'>그래프 생성 오류: {e}</div>")

    @output
    @render.ui
    def show_firehydrant_distance_plot():
        return ui.HTML("소화전 거리 분포 플롯이 여기에 표시됩니다.")

    @output
    @render.data_frame
    def show_summary():
        f = filtered_building_df()
        if f.empty:
            return pd.DataFrame({"메시지": ["표시할 데이터가 없습니다."]})
        return (
            f.groupby("읍면동")
             .size()
             .reset_index(name="건물수")
             .sort_values("건물수", ascending=False)
        )

    @output
    @render.ui
    def show_score_map2():
        return ui.HTML("사용자 설정 기반 취약 점수 지도가 여기에 표시됩니다.")

    @output
    @render.ui
    def show_filtered_building_map():
        return ui.HTML("필터링된 건물 분포 지도가 여기에 표시됩니다.")

    @output
    @render.plot
    def top_bottom_histogram():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist([1, 2, 3], bins=10)
        ax.set_title("예시 히스토그램")
        return fig

    @output
    @render.ui
    def show_data_table():
        return ui.HTML("변수 정의 테이블이 여기에 표시됩니다.")

    @output
    @render.ui
    def show_variable_table():
        return ui.HTML("데이터 설명 테이블이 여기에 표시됩니다.")

# --------------------------
# 3) App 객체 + 런처
# --------------------------
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
