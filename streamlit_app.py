"""Run with: .venv/Scripts/python.exe -m streamlit run streamlit_app.py"""
import streamlit as st

st.set_page_config(page_title='NBA 得分研究室',page_icon='🏀',layout='wide',initial_sidebar_state='collapsed')
page=st.navigation([
    st.Page('dashboard_pages/history.py',title='歷史回測',icon='🏀',default=True),
    st.Page('dashboard_pages/comparison.py',title='模型比較',icon='📊'),
],position='hidden')
links=st.columns(2)
links[0].page_link('dashboard_pages/history.py',label='歷史回測',icon='🏀')
links[1].page_link('dashboard_pages/comparison.py',label='模型比較',icon='📊')
st.divider()
page.run()
