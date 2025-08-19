import pandas as pd

# (장애인_병원교통수단) 시각화  => 버스를 가장 많이 이용한다
df=pd.read_csv('./data_MJ/장애인_병원교통수단.csv',encoding='cp949')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
# 한글 폰트
mpl.rc('font', family='Malgun Gothic')
mpl.rcParams['axes.unicode_minus'] = False
df.rename(columns={'일반 저상버스 (%)': '일반/저상버스 (%)'}, inplace=True)
# 대중교통 관련 열만 선택
columns_to_use = [
    '일반/저상버스 (%)',
    '일반택시 (%)',
    '지하철/전철 (%)',
    '장애인콜택시(복지콜 포함) (%)',
    '통학버스/사회복지시설 버스 (%)']

# '전체' 행만 필터링
df_total_only = df[(df['특성별(1)'] == '전체') & (df['특성별(2)'] == '소계')]
# 숫자형 변환
for col in columns_to_use:
    df_total_only[col] = pd.to_numeric(df_total_only[col], errors='coerce')
# 평균 이용률 계산
mean_usage1 = df_total_only[columns_to_use].mean().sort_values(ascending=False)
# ✅ 대중교통 내 구성비율로 백분율 환산
total_usage1 = mean_usage1.sum()
mean_usage1_percent = (mean_usage1 / total_usage1 * 100).round(1)
# ✅ 인덱스 클린업: ' (%)' 제거 → 새 변수에 저장 (기존 Series 유지)
labels = mean_usage1_percent.index.str.replace(' (%)', '', regex=False)
# 색상 설정: 일반/저상버스만 빨간색, 나머지는 회색
colors = ['red' if '일반/저상버스' in label else 'gray' for label in labels]
# ✅ 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x=mean_usage1_percent.values, y=labels, palette=colors)
# 제목
plt.suptitle('외출 시 대중교통 수단별 이용률', fontsize=18, fontweight='bold', x=0.6, y=1.05)
# 축 라벨
plt.xlabel('평균 이용률 (%)', fontsize=13, fontweight='bold', labelpad=20)
plt.ylabel('교통수단 (%)', fontsize=13, fontweight='bold', rotation=90, labelpad=30)
# 막대 수치 표시
for i, value in enumerate(mean_usage1_percent.values):
    is_red = '일반/저상버스' in labels[i]  # 또는 다른 조건
    plt.text(
        value + 0.5,
        i,
        f'{value:.1f}%',
        va='center',
        fontsize=11,
        color='red' if is_red else 'gray',
        fontweight='bold' if is_red else 'normal')  # ✅ 이 줄이 핵심!
# x축 범위 (0~100%)
plt.xlim(0, max(mean_usage1_percent.values) * 1.15)
# 주석
plt.figtext(0.99, -0.05, "* 2018~2023년의 평균 이용률 데이터를 사용함", fontsize=13, color='black', ha='right')
plt.tight_layout()
plt.show()



# (장애인_외출수단).csv 시각화
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
df=pd.read_csv('./data_MJ/장애인_외출교통수단.csv',encoding='cp949')
#  '일반버스 저상버스 (%)' → '일반/저상버스 (%)'로 이름 변경 (저상버스만 의미)
df.rename(columns={'일반버스 저상버스 (%)': '일반/저상버스 (%)'}, inplace=True)
# '사회복지시설버스 (%)' + '장애인 무료 셔틀버스 (%)' 병합
for col in ['사회복지시설버스 (%)', '장애인 무료 셔틀버스 (%)']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['셔틀버스/사회복지시설 버스 (%)'] = df['사회복지시설버스 (%)'].fillna(0) + df['장애인 무료 셔틀버스 (%)'].fillna(0)
df.drop(columns=['사회복지시설버스 (%)', '장애인 무료 셔틀버스 (%)'], inplace=True)

# 한글 폰트
mpl.rc('font', family='Malgun Gothic')
mpl.rcParams['axes.unicode_minus'] = False
# 대중교통 관련 열만 선택
columns_to_use = [
    '일반/저상버스 (%)',
    '일반택시 (%)',
    '지하철/전철 (%)',
    '장애인콜택시(복지콜 포함) (%)',
    '셔틀버스/사회복지시설 버스 (%)'
]
# '전체' 행만 필터링
df_total_only = df[(df['특성별(1)'] == '전체') & (df['특성별(2)'] == '소계')]
# 숫자형 변환
for col in columns_to_use:
    df_total_only[col] = pd.to_numeric(df_total_only[col], errors='coerce')
# 평균 이용률 계산
mean_usage2 = df_total_only[columns_to_use].mean().sort_values(ascending=False)
# ✅ 대중교통 내 구성비율로 백분율 환산
total_usage2 = mean_usage2.sum()
mean_usage2_percent = (mean_usage2 / total_usage2 * 100).round(1)
# 시각화 설정
# 항목 이름에서 ' (%)' 제거
mean_usage2_percent.index = mean_usage2_percent.index.str.replace(' (%)', '', regex=False)
# 색상 설정: 일반/저상버스만 빨간색, 나머지는 회색
colors = ['red' if '일반/저상버스' in label else 'gray' for label in mean_usage2_percent.index]

# 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x=mean_usage2_percent.values, y=mean_usage2_percent.index, palette=colors)
# ✅ 제목 가운데 정렬 (그래프 중심 기준)
plt.suptitle('외출 시 대중교통 수단별 이용률', fontsize=18, fontweight='bold', x=0.6, y=1.05)
# ✅ 축 라벨 설정 (세로 정렬, 패딩 조정)
plt.xlabel('평균 이용률 (%)', fontsize=13, fontweight='bold', labelpad=20)
plt.ylabel('교통수단 (%)', fontsize=13, fontweight='bold', rotation=90, labelpad=30)
# ✅ 막대 오른쪽에 수치 표시
for i, value in enumerate(mean_usage2_percent.values):
    plt.text(value + 0.5, i, f'{value:.1f}%', va='center', fontsize=11,
             color='red' if '일반/저상버스' in mean_usage2_percent.index[i] else 'gray')
# ✅ x축 범위 확장 (겹침 방지)
plt.xlim(0, max(mean_usage2_percent.values) * 1.15)
# ✅ 하단 오른쪽 주석
plt.figtext(0.99, -0.05, "* 2018~2023년의 평균 이용률 데이터를 사용함", fontsize=13, color='black', ha='right')

# 레이아웃 정리
plt.tight_layout()
plt.show()



# 위 두 그래프를 subplot으로 만들기(1행2열)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
df1=pd.read_csv('./data_MJ/장애인_병원교통수단.csv',encoding='cp949')
df2=pd.read_csv('./data_MJ/장애인_외출교통수단.csv',encoding='cp949')
mpl.rc('font', family='Malgun Gothic')
mpl.rcParams['axes.unicode_minus'] = False
df1.rename(columns={'일반 저상버스 (%)': '일반/저상버스 (%)'}, inplace=True)
# df1 전처리
columns_to_use = [
    '일반/저상버스 (%)',
    '일반택시 (%)',
    '지하철/전철 (%)',
    '장애인콜택시(복지콜 포함) (%)',
    '통학버스/사회복지시설 버스 (%)']
# '전체' 행만 필터링
df1_total_only = df1[(df1['특성별(1)'] == '전체') & (df1['특성별(2)'] == '소계')]
# 숫자형 변환
for col in columns_to_use:
    df1_total_only[col] = pd.to_numeric(df1_total_only[col], errors='coerce')
# 평균 이용률 계산
mean_usage1 = df1_total_only[columns_to_use].mean().sort_values(ascending=False)
# ✅ 대중교통 내 구성비율로 백분율 환산
total_usage1 = mean_usage1.sum()
mean_usage1_percent = (mean_usage1 / total_usage1 * 100).round(1)
# df2 전처리
df2=pd.read_csv('./data_MJ/장애인_외출교통수단.csv',encoding='cp949')
#  '일반버스 저상버스 (%)' → '일반/저상버스 (%)'로 이름 변경 (저상버스만 의미)
df2.rename(columns={'일반버스 저상버스 (%)': '일반/저상버스 (%)'}, inplace=True)
# '사회복지시설버스 (%)' + '장애인 무료 셔틀버스 (%)' 병합
for col in ['사회복지시설버스 (%)', '장애인 무료 셔틀버스 (%)']:
    df2[col] = pd.to_numeric(df2[col], errors='coerce')
df2['셔틀버스/사회복지시설 버스 (%)'] = df2['사회복지시설버스 (%)'].fillna(0) + df2['장애인 무료 셔틀버스 (%)'].fillna(0)
df2.drop(columns=['사회복지시설버스 (%)', '장애인 무료 셔틀버스 (%)'], inplace=True)
columns_to_use = [
    '일반/저상버스 (%)',
    '일반택시 (%)',
    '지하철/전철 (%)',
    '장애인콜택시(복지콜 포함) (%)',
    '셔틀버스/사회복지시설 버스 (%)']
# '전체' 행만 필터링
df2_total_only = df2[(df2['특성별(1)'] == '전체') & (df2['특성별(2)'] == '소계')]
# 숫자형 변환
for col in columns_to_use:
    df2_total_only[col] = pd.to_numeric(df2_total_only[col], errors='coerce')
# 평균 이용률 계산
mean_usage2 = df2_total_only[columns_to_use].mean().sort_values(ascending=False)
# ✅ 대중교통 내 구성비율로 백분율 환산
total_usage2 = mean_usage2.sum()
mean_usage2_percent = (mean_usage2 / total_usage2 * 100).round(1)
# ' (%)' 제거
mean_usage1_percent.index = mean_usage1_percent.index.str.replace(' (%)', '', regex=False)
mean_usage2_percent.index = mean_usage2_percent.index.str.replace(' (%)', '', regex=False)
# 색상 지정 (각 시리즈마다)
colors1 = ['red' if '일반/저상버스' in label else 'gray' for label in mean_usage1_percent.index]
colors2 = ['red' if '일반/저상버스' in label else 'gray' for label in mean_usage2_percent.index]
# subplot 설정
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=False)
# 왼쪽 그래프
sns.barplot(x=mean_usage2_percent.values, y=mean_usage2_percent.index, palette=colors2, ax=axes[0])
axes[0].set_title('외출 시', fontsize=16, fontweight='bold',y=1.03)
axes[0].set_xlabel('평균 이용률 (%)', fontsize=12, fontweight='bold', labelpad=15)
axes[0].set_xlim(0,70)
axes[0].set_ylabel('교통수단 (%)', fontsize=13, fontweight='bold', labelpad=25)
axes[0].tick_params(axis='y', labelleft=True)
for i, value in enumerate(mean_usage2_percent.values):
    is_red = '일반/저상버스' in mean_usage2_percent.index[i]
    axes[0].text(value + 0.5, i, f'{value:.1f}%', va='center', fontsize=10,
                 fontweight='bold' if is_red else 'normal',
                 color='red' if is_red else 'gray')
# 오른쪽 그래프
sns.barplot(x=mean_usage1_percent.values, y=mean_usage1_percent.index, palette=colors1, ax=axes[1])
axes[1].set_title('병원 이용 시', fontsize=16, fontweight='bold',y=1.03)
axes[1].set_xlabel('평균 이용률 (%)', fontsize=12, fontweight='bold', labelpad=15)
axes[1].set_xlim(0, 70)
axes[1].set_ylabel('')  # 오른쪽 그래프는 y축 라벨 생략
axes[1].tick_params(axis='y', labelsize=11)
for i, value in enumerate(mean_usage1_percent.values):
    is_red = '일반/저상버스' in mean_usage1_percent.index[i]
    axes[1].text(value + 0.5, i, f'{value:.1f}%', va='center', fontsize=10,
                 fontweight='bold' if is_red else 'normal',
                 color='red' if is_red else 'gray')
# 전체 제목
plt.suptitle('장애인 주요 대중교통 수단', fontsize=20, fontweight='bold', y=1.05)
# 하단 오른쪽 주석
plt.figtext(0.99, -0.05, "* 2018~2023년의 평균 이용률 데이터를 사용함", fontsize=13, color='black', ha='right')
# 레이아웃 조정
plt.tight_layout()
plt.show()


