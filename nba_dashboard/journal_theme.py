"""Shared presentation only: court notes shell and immutable chart restyling."""
from html import escape
from pathlib import Path
import altair as alt
import streamlit as st
from .views import _line_chart as original_line_chart

PAGES=[('歷史回測','dashboard_pages/history.py',':material/query_stats:'),
       ('模型比較','dashboard_pages/comparison.py',':material/layers:'),
       ('未來預測','dashboard_pages/future.py',':material/online_prediction:')]
SECTIONS={'歷史回測':'球員報告','模型比較':'模型評估','未來預測':'賽前觀察'}


def render_shell(title):
    css=Path(__file__).with_name('journal.css').read_text(encoding='utf-8')
    st.markdown('<style>'+css+'</style>',unsafe_allow_html=True)
    season=st.session_state.get('comparison_season','2025-26') if title=='模型比較' else st.session_state.get('history_season','2026-27') if title=='歷史回測' else '2026-27'
    with st.sidebar:
        st.markdown('<div class="journal-brand">court<span>notes.</span></div><div class="journal-brand-caption">NBA 球員表現研究</div>',unsafe_allow_html=True)
        for index,(name,path,icon) in enumerate(PAGES):
            with st.container(key='journal-nav-active' if name==title else 'journal-nav-'+str(index)):
                st.page_link(path,label=name,icon=icon)
        st.markdown('<div class="journal-season-summary"><span>正在觀察</span><strong>'+escape(season.replace('-','–'))+'</strong><span>例行賽</span></div>',unsafe_allow_html=True)
        st.markdown('<div class="journal-sidebar-footer">一場比賽，一份紀錄。<br><span>NBA 得分研究室</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="journal-breadcrumb"><span>'+escape(title)+'<b>/</b>'+escape(SECTIONS.get(title,''))+'</span><span class="journal-edition">COURT NOTES · NBA ANALYTICS</span></div>',unsafe_allow_html=True)


def journal_line_chart(data,x,series):
    palette=[]
    for name in series:
        palette.append('#607D8B' if '實際' in name else '#A6B0A7' if '均值' in name else '#557366')
    return (original_line_chart(data,x,series)
        .encode(color=alt.Color('曲線:N',scale=alt.Scale(domain=series,range=palette),legend=alt.Legend(title=None,orient='bottom')))
        .properties(height=260)
        .configure_view(stroke=None)
        .configure_axis(labelColor='#708078',titleColor='#708078',gridColor='#EDF0EB',domain=False,tickColor='#E4E9E3',titleFontWeight='normal')
        .configure_legend(labelColor='#66766E',symbolStrokeWidth=2))
