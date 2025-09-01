# modules/page_routes.py
from shiny import ui, render
from viz.routes import build_shuttle_routes_html_from_grid
from html import escape  # ⬅️ 추가

def render_ui():
    return ui.card(
        ui.card_header("점수 기반 셔틀버스 노선 추천"),
        ui.output_ui("routes_map"),
        full_screen=True
    )

def register_server(output, input, session, shared):
    @output
    @render.ui
    def routes_map():
        grid = shared["grid"].get()
        if grid is None:
            return ui.HTML("<div style='padding:1rem'>먼저 접근성 지도를 계산해 주세요.</div>")

        # pydeck HTML 생성
        deck_html = build_shuttle_routes_html_from_grid(grid)

        iframe_html = f"""
        <iframe
            style="width:100%;height:820px;border:0"
            sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
            srcdoc="{escape(deck_html)}">
        </iframe>
        """
        return ui.HTML(iframe_html)
