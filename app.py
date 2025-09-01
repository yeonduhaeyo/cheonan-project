# app.py
from shiny import App, ui, render, reactive
from shinyswatch import theme

from modules import (
    page_accessibility, page_routes,
    page_appendix1, page_appendix2
)

def app_ui(request):
    return ui.page_navbar(
        # ✅ head는 여기서 head_content로
        ui.head_content(
            # Plotly를 외부 CDN으로 쓰고 싶다면 유지
            # ui.tags.script(src="https://cdn.plot.ly/plotly-2.30.0.min.js"),
        ),

        # --- 탭들 ---
        ui.nav_panel(
            "접근성 점수 지도",
            ui.card(
                ui.card_header("사용자 가중치 설정"),
                ui.layout_columns(
                    ui.input_slider("w0", "① 복지시설까지의 거리", min=0, max=25, value=25),
                    ui.input_slider("w1", "② 저상버스 비율",   min=0, max=25, value=25),
                    ui.input_slider("w2", "③ 버스 노선 수",   min=0, max=25, value=25),
                    ui.input_slider("w3", "④ 배차 간격",       min=0, max=25, value=25),
                )
            ),
            page_accessibility.render_ui()
        ),

        ui.nav_panel("셔틀버스 노선 추천", page_routes.render_ui()),

        ui.nav_panel("부록1(인구밀도 시각화)", page_appendix1.render_ui()),
     
        ui.nav_panel("부록2(기준 및 설명)", page_appendix2.render_ui()),

        title="천안 🏠 복지시설 🚌 대중교통 접근성 분석",
        theme=theme.minty  # ✅ 부트스와치 테마는 여기서만
    )

def server(input, output, session):
    # 모듈 간 공유 상태
    shared = {
        "grid": reactive.Value(None),     # 접근성 그리드(GeoDataFrame, EPSG:4326)
        "weights": reactive.Value(None),  # 선택 사항
    }

    page_accessibility.register_server(output, input, session, shared)
    page_routes.register_server(output, input, session, shared)
    page_appendix1.register_server(output, input, session)
    page_appendix2.register_server(output, input, session, shared)

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
