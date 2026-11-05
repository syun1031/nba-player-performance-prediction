"""Run with: .venv/Scripts/python.exe -m streamlit run streamlit_app.py"""
import streamlit as st
from nba_dashboard.journal_theme import render_shell

st.set_page_config(page_title='court notes · NBA 得分研究室',page_icon='🏀',layout='wide',initial_sidebar_state='auto')
page=st.navigation([
    st.Page('dashboard_pages/history.py',title='歷史回測',icon='🏀',default=True),
    st.Page('dashboard_pages/comparison.py',title='模型比較',icon='📊'),
    st.Page('dashboard_pages/future.py',title='未來預測',icon='🔮'),
],position='hidden')
render_shell(page.title)
page.run()
