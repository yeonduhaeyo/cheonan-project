# modules/page_accessibility.py
from shiny import ui, render
from plotly.io import to_html
from viz.score_map import build_score_map

def render_ui():
    return ui.card(
        ui.card_header("사용자 설정 기반 접근성 지수 산출"),
        ui.output_ui("show_score_map"),
        full_screen=True
    )

# ✅ shared 추가
def register_server(output, input, session, shared):
    @output
    @render.ui
    def show_score_map():
        # 슬라이더 값 → 정규화는 build_score_map 내부에서 처리하도록 가정
        weights = [input.w0(), input.w1(), input.w2(), input.w3()]

        # ✅ grid까지 같이 받아서 공유 스토어에 저장
        fig, grid = build_score_map(w=weights, return_grid=True)
        shared["grid"].set(grid)       # ← 다른 탭(노선추천)에서 그대로 사용
        shared["weights"].set(weights) # (옵션) 기록만

        html = to_html(fig, include_plotlyjs=True, full_html=False)
        return ui.HTML(html)
