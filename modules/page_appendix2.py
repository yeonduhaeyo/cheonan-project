from shiny import ui, render
import pandas as pd

def render_ui():
    return ui.layout_column_wrap(

        # --------------------
        # 사용 데이터 설명
        # --------------------
        ui.card(
            ui.card_header("사용 데이터"),
            ui.tags.ul(
                ui.tags.li("천안시 행정구역 경계 데이터 (GeoJSON)"),
                ui.tags.li("천안시 버스 정류소 및 노선 데이터 (천안시청, 교통공사 공개자료)"),
                ui.tags.li("복지·의료 시설 데이터 (보건복지부, 공공데이터포털)"),
                ui.tags.li("저상버스 운행 정보 (국토교통부 교통약자 이동편의 데이터)"),
                ui.tags.li("읍면동별 인구 통계 (노인 인구, 장애인 등록 인구 — 행정안전부 주민등록인구 통계)")
            ),
            ui.tags.p("※ 모든 자료는 공개자료를 수집·전처리하여 사용했습니다."),
            style="height:auto;"
        ),

        # --------------------
        # 점수화 기준 설명
        # --------------------
        ui.card(
            ui.card_header("접근성 점수 산정 기준"),
            ui.tags.ul(
                ui.tags.li("① 복지시설까지의 거리: 가까울수록 높은 점수"),
                ui.tags.ul(
                    ui.tags.li("400m 이내 → 1.0점"),
                    ui.tags.li("400~800m → 0.5점"),
                    ui.tags.li("800m 초과 → 0점"),
                    ui.tags.li("시설별 가중치: 장애인재활 40%, 노인복지 30%, 보건기관 20%, 응급의료 10%")
                ),
                ui.tags.li("② 저상버스 점수: 격자 반경 500m 이내 정류장의 저상버스 운행 비율"),
                ui.tags.li("③ 버스 노선 수 점수: 500m 내 접근 가능한 노선 수 / 전체 노선 수"),
                ui.tags.li("④ 배차 간격 점수: 1 - (500m 내 평균 배차 간격 / 전체 최대 배차 간격)"),
                ui.tags.li("⑤ 교통약자 인구 보정: 노인 인구밀도 + 장애인 인구밀도 비례 반영")
            ),
            ui.tags.p("※ 위 지표의 가중합 × (교통약자 인구밀도 / 최대 인구밀도) = 최종 접근성 점수"),
            style="height:auto;"
        ),

        # --------------------
        # 산정 공식
        # --------------------
        ui.card(
            ui.card_header("점수 산정 공식"),
            ui.tags.pre("""
접근성 점수 = (w1 * 시설 점수 
             + w2 * 저상버스 점수 
             + w3 * 노선 수 점수 
             + w4 * 배차 간격 점수) 
             × (교통약자 인구밀도 / 최대 인구밀도)
            """),
            ui.tags.p("※ w1~w4는 분석 목적에 따라 조정 가능 (기본값: 0.25씩 동일)."),
            style="height:auto;"

        ),

        # --------------------
        # 최종 데이터프레임 샘플
        # --------------------
        ui.card(
            ui.card_header("최종 데이터프레임 (일부 예시)"),
            ui.output_table("df_final"),
            style="height:auto;"
        ),

        width=1
    )


def register_server(output, input, session, shared=None):
    @output
    @render.table
    def df_final():
        # shared에서 grid를 받아온다고 가정
        if shared is None or "grid" not in shared:
            return pd.DataFrame({"메시지": ["shared['grid'] 데이터가 없음"]})

        grid = shared["grid"].get()
        COLUMN_RENAME = {
        "BASE_DATE": "기준일자",
        "ADM_CD": "행정코드",
        "ADM_NM": "행정동",
        "geometry": "격자경계",
        "centroid": "격자중심",
        "dist_장애인 재활시설": "거리_장애인재활시설",
        "dist_노인 복지시설": "거리_노인복지시설",
        "dist_보건기관": "거리_보건기관",
        "dist_응급의료기관": "거리_응급의료기관",
        "score_장애인 재활시설": "점수_장애인재활시설",
        "score_노인 복지시설": "점수_노인복지시설",
        "score_보건기관": "점수_보건기관",
        "score_응급의료기관": "점수_응급의료기관",
        "facility_score_weighted": "가중치_시설점수",
        "lowfloor_score": "저상버스점수",
        "num_routes_score": "노선수점수",
        "headway_score": "배차간격점수",
        "grid_id": "격자ID",
        "장애인_인구밀도": "장애인인구밀도",
        "노인_인구밀도": "노인인구밀도",
        "교통약자_인구밀도": "교통약자인구밀도",
        "accessibility_score": "접근성점수",
        "has_road": "도로존재여부"
        }
        # 원하는 컬럼만 선택
        cols = [
            "facility_score_weighted", "lowfloor_score",
            "num_routes_score", "headway_score",
            "교통약자_인구밀도", "accessibility_score"
        ]
        df = pd.DataFrame(grid[cols]).sample(10) # 10개만 샘플

        # 한글 컬럼명으로 변환
        df = df.rename(columns=COLUMN_RENAME)
        
        return df
