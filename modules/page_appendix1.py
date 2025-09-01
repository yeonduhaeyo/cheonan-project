# modules/page_appendix.py
import base64
from shiny import ui, render
import os

from viz.charts import dis_old_rat
from viz.maps import make_disabled_map, make_older_map, build_bus_and_facilities_map

FAC_TYPES = ["응급의료기관","보건기관","장애인 재활시설","노인 복지시설"]

def _iframe_from_html(html_text: str, height=700) -> ui.HTML:
    b64 = base64.b64encode(html_text.encode("utf-8")).decode("ascii")
    return ui.HTML(
        f'<iframe style="width:100%;height:{height}px;border:0" '
        f'src="data:text/html;charset=utf-8;base64,{b64}"></iframe>'
    )

def render_ui():
    return ui.page_fluid(
        ui.h3("부록/요약 시각화"),

        # # (1) 상단: Plotly Subplots (왼쪽), 오른쪽은 비워둠
        # ui.row(
        #     ui.column(6, ui.card(
        #         ui.card_header("장애인/노인 지표 (Plotly Subplots)"),
        #         ui.output_ui("out_subplots")
        #     )),
        #     ui.column(6)  # 오른쪽 빈 칸
        # ),

        # (2) 장애인 인구 지도 + 노인 인구 지도 (양쪽 배치)
        ui.row(
            ui.column(6, ui.card(
                ui.card_header("장애인 인구 비율 지도 + 복지시설"),
                ui.output_ui("out_disabled_map")
            )),
            ui.column(6, ui.card(
                ui.card_header("노인 인구 비율 지도 + 복지시설"),
                ui.output_ui("out_older_map")
            ))
        ),

        # # (3) 버스 노선 + 시설
        # ui.row(
        #     ui.column(12, ui.card(
        #         ui.card_header("실도로 기반 버스 노선 + 시설"),
        #         ui.output_ui("out_bus_map", height="700px")
       
        #         # ui.column(4, ui.input_checkbox_group(
        #         #     "fac_types2", "표시할 시설", choices=FAC_TYPES, selected=FAC_TYPES
        #         # )),
        #         )
        #     )
        # ),
        # ui.card(
        # ui.card_header("실도로 기반 버스 노선 + 시설"),
        # # ui.output_ui("out_bus_map")
        # ui.img(src="bus_facilities.png", style="width:100%; height:auto;")
        # )
    )

def register_server(output, input, session):
    @output
    @render.ui
    def out_subplots():
        html = dis_old_rat(as_fragment=False)
        return _iframe_from_html(html, height=560)

    @output
    @render.ui
    def out_disabled_map():
        fig = make_disabled_map()
        frag = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return _iframe_from_html(frag, height=640)

    @output
    @render.ui
    def out_older_map():
        fig = make_older_map(add_facilities=True)
        frag = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return _iframe_from_html(frag, height=640)

    # @output
    # @render.ui
    # def out_bus_map():
    #     # types = list(input.fac_types2()) if input.fac_types2() else FAC_TYPES
    #     types = FAC_TYPES
    #     # filename = build_bus_and_facilities_map(selected_facilities=types)
    #     # frag = fig.to_html(full_html=False, include_plotlyjs="inline")
    #     # return _iframe_from_html(frag, height=760)
        
    #     # root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    #     # www_dir = os.path.join(root_dir, "www")
        
    #     # filename = "bus_facilities.png"
    #     # path = os.path.join(www_dir, filename)
    #     return ui.img(src=f'/{filename}', style="width:100%; height:auto;")