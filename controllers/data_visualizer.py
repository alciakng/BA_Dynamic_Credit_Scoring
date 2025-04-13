from matplotlib import ticker
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class DataVisualizer:
    def __init__(self):
        self.df = None
        
    # 데이터셋을 셋팅한다.
    def df_setter(self, df):
        self.df = df
            
    def df_plt_setter(self, df, title, x_label, y_label, x_size, y_size):
        self.df = df
        plt.figure(figsize=(x_size,y_size))
        plt.ticklabel_format(style='plain', axis='y')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        return plt
    
    # 파이플롯
    def pie(self, x, y, title):
        fig = px.pie(
            self.df,
            names=x,
            values=y,
            title=title,
            hole=0.3 # 0이면 일반 파이, 0.4 정도면 도넛
        )

        fig.show()

    # 트리맵 차트
    def treemap(self, level1, level2, value):
        fig = px.treemap(self.df,
                        path=[level1,level2],
                        values=value,
                        color=value,
                        color_continuous_scale='viridis')
        fig.show()
    
    # 바(히스토그램) 차트
    def bar(self, x, y, value, x_axis_title,y_axis_title, title, hline_yn, hline_y, hline_text):
        fig = px.bar(self.df, x=x, y=y, color=value, barmode='stack', text=value)

        # 평균선 추가
        if(hline_yn) :
            fig.add_hline(
                y=hline_y,
                line_dash='dash',
                line_color='red',
                annotation_text=f'{hline_text} : {hline_y:.2f}%',
                annotation_position='top left',
                annotation_font_color='red'
            )
    
        # y축 여유공간
        fig.update_layout(     
            xaxis_title=x_axis_title,   
            yaxis=dict(
                tickformat=',.0f',  # 1,000 → 1,000 식으로 콤마 포함
                title=y_axis_title
            ),
            yaxis_range=[0, max(self.df[y]) * 1.3]
        )

        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside'
        )

        fig.update_layout(title=title)
        
        fig.show()
    

    def multiple_bar(self, df1, df2, feature, x_axis_title,y_axis_title, category_title_1, category_title_2,  title):

        # Plotly 시각화
        fig = go.Figure(data=[
            go.Bar(name=category_title_1, x=feature, y=df1),
            go.Bar(name=category_title_2, x=feature, y=df2)
        ])

        # 레이아웃 설정
        fig.update_layout(
            barmode='group',
            title=title,
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            xaxis_tickangle=-45,
            height=500,
            width=900
        )

        return fig

    # 박스플롯 차트
    def box(self,x,y,x_label,y_label,title):
        fig = px.box(
            self.df,
            x=x,
            y=y,
            color=x,
            points='all',  # 이상치까지 점으로 표시
            title=title,
            labels={x: x_label, y: y_label}
        )

        fig.update_layout(
            xaxis=dict(
                categoryorder='array',
                categoryarray=self.df[x].cat.categories.to_list()
            ),
            yaxis_tickformat=",.0f",  # 소수 없이 퍼센트 (%)
        )

        fig.show()

    # 히스토그램 차트
    def hist(self, column, bins=10):
        plt.figure(figsize=(8, 4))
        sns.histplot(self.df[column], bins=bins, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    # 스캐터플룻 차트
    def scatter(self, x, y, hue=None):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=self.df, x=x, y=y, hue=hue)
        plt.title(f'Scatter plot of {y} vs {x}')
        plt.tight_layout()
        plt.show()


    def multi_scatter(self, df1, df2, x, y, x_axis_title, y_axis_title, df1_name, df2_name):
        fig = go.Figure()

        df1['YM'] = df1['YM'].astype('str')
        df2['YM'] = df2['YM'].astype('str')

        # 기준
        fig.add_trace(go.Scatter(
            x=df1[x],
            y=df1[y],
            mode='lines+markers',
            name=df1_name,
            marker=dict(symbol='circle', color='teal'),
            line=dict(color='teal')
        ))

        # 비교 
        fig.add_trace(go.Scatter(
            x=df2[x],
            y=df2[y],
            mode='lines+markers',
            name=df2_name,
            marker=dict(symbol='square', color='indianred'),
            line=dict(color='indianred')
        ))

        # 레이아웃 설정
        fig.update_layout(
            title=df1_name+", "+df2_name +"연체율 추이",
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            xaxis=dict(tickmode='array', tickvals=df1['YM']),
            legend=dict(title='Group'),
            template='simple_white',
            xaxis_tickangle=-45,
            height=500
        )

        return fig


    # 히트맵 차트
    def heatmap_corr(self):
        plt.figure(figsize=(10, 8))
        corr = self.df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()