"""Season-aware comparison entry; classic page and production model stay intact."""
from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st
from .data import DataIssue
from .views import render_comparison as classic_comparison
from .research_data import load_research,revision,research_metrics,MODELS,NAMES,EXPECTED

ROOT=Path(__file__).resolve().parents[1]
PERIOD_LABELS={'test':'三月至季末｜已使用的測試資料','validation':'二月驗證｜開發資料','CV':'CV｜三輪時間驗證'}


@st.cache_data(show_spinner='載入已核對的研究結果…')
def cached_research(root,version):return load_research(Path(root))


def csv(frame):return frame.to_csv(index=False).encode('utf-8-sig')


def render_comparison():
    season=st.selectbox('球季',['2025-26','2024-25'],format_func=lambda s:s.replace('-','–')+' 例行賽',key='comparison_season')
    options=['classic','research'] if season=='2025-26' else ['research']
    if st.session_state.get('comparison_scope') not in options:st.session_state.pop('comparison_scope',None)
    scope=st.radio('比較範圍',options,format_func=lambda v:'原有模型' if v=='classic' else '三個研究候選',horizontal=True,key='comparison_scope')
    if scope=='classic':
        classic_comparison();return
    try:bundle=cached_research(str(ROOT),revision(ROOT))
    except DataIssue as exc:st.error(str(exc));st.stop()
    st.caption('NBA 得分研究室 · '+season.replace('-','–')+' 例行賽')
    st.title('模型比較')
    st.write('在同一批比賽比較三個研究候選，分別觀察整體與高得分誤差。')
    st.caption('三者均為 XGBoost 研究版本；原九項使用同樣訓練樣本重訓作對照。比較不會更換已鎖定的線上主模型。')
    periods=['test','validation','CV']
    if st.session_state.get('research_period') not in periods:st.session_state.pop('research_period',None)
    period=st.radio('研究資料期間',periods,format_func=PERIOD_LABELS.get,horizontal=True,key='research_period')
    threshold=st.selectbox('觀察場次',[None,20,30],format_func=lambda n:'全部出賽' if n is None else f'實際得分 {n} 分以上',key='research_threshold')
    frame=bundle.predictions[season][period];start,end,n=EXPECTED[season][period]
    if period=='CV':
        year=int(season[:4]);training_note=f'三輪訓練截止：{year}-11-30、{year}-12-31、{year+1}-01-15，各輪只使用更早資料'
    else:training_note='模型固定訓練至 '+('2025-01-31' if season=='2024-25' else '2026-01-31')
    st.caption(f'{start} 至 {end} · 完整共同樣本 {n:,} 筆 · '+training_note)
    if season=='2025-26' and period=='test':st.info('這是在已使用過的2025–26測試資料上追加模型比較，不是新的獨立最終測試。原有模型的結果仍可切換「原有模型」查看。')
    if period=='test':st.caption('此期間已用於測試或研究比較，不能再當作未使用的新資料，也不依此結果追加調參。')
    else:st.caption('這是開發期資料，已用於研究比較，不能當作獨立最終測試。')
    if threshold is not None:st.info('高分分組依賽後實際得分觀察，預測前不知道哪些比賽會高分。20+與30+互相重疊，不能相加。')
    stats=research_metrics(frame,['baseline']+MODELS,threshold)
    if stats.empty:st.info('目前範圍沒有合格比賽。');return
    cards=st.columns(4)
    for col,v in zip(cards,MODELS):
        r=stats.loc[stats['model_variant']==v].iloc[0];col.metric(NAMES[v]+' MAE',f'{r["mae"]:.3f} 分')
    cards[3].metric('樣本數',f'{int(stats.iloc[0]["n"]):,}')
    st.caption('MAE是平均差幾分，RMSE更重視大誤差；兩者越低越好。表格維持固定模型順序，不按測試結果選模。')
    high=research_metrics(frame,['baseline']+MODELS,30)
    if not high.empty:
        st.warning(f'實際30分以上共 {int(high.iloc[0]["n"]):,} 筆；三模型MAE依序為 '+
            '／'.join(f'{high.loc[high.model_variant==v,"mae"].iloc[0]:.2f}' for v in MODELS)+' 分。這是賽後分組，高得分誤差仍需留意。')
    plot=stats.assign(模型=stats['model_variant'].map(NAMES))
    cols=st.columns(2)
    for col,metric,title in [(cols[0],'mae','平均誤差 MAE'),(cols[1],'rmse','大誤差指標 RMSE')]:
        chart=alt.Chart(plot).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X(metric+':Q',title='分（越低越好）'),y=alt.Y('模型:N',sort=[NAMES[v] for v in ['baseline']+MODELS],title=None,axis=alt.Axis(labelLimit=300)),
            color=alt.value('#243C60'),tooltip=['模型:N',alt.Tooltip('n:Q',title='同樣本數',format=','),alt.Tooltip(metric+':Q',format='.3f')]).properties(height=240,title=title)
        col.altair_chart(chart,width='stretch')
    table=stats.rename(columns={'n':'樣本數','mae':'MAE','rmse':'RMSE','mean_residual':'平均低估分數'}).copy()
    table.insert(0,'模型',table['model_variant'].map(NAMES));table=table.drop(columns='model_variant')
    st.dataframe(table,hide_index=True,width='stretch',column_config={k:st.column_config.NumberColumn(format='%.3f') for k in ['MAE','RMSE','平均低估分數']})
    st.download_button('下載目前研究比較表',csv(table),f'comparison_{season}_{period}.csv','text/csv',key='research_comparison_download')
    with st.expander('模型用了什麼資料？'):
        st.write('原九項：近五場／近最多十場得分，近五場分鐘／籃板／助攻，主場、中立場、休息天數、自己是否先發。')
        st.write('上季表現：上季平均得分、平均分鐘、總分／總分鐘、出賽場數、有無上季資料。沒有上季紀錄時保留缺值，仍可預測。')
        st.write('1＋3＋同區：再加入近期變化／波動、對手近五場得分環境、同區對陣歷史；2倍權重只用於訓練時的實際30+場次，評估誤差不加權。')
        st.write('三版本使用相同個人／球隊／對手五場資格樣本。歷史窗口只用目標日期前的比賽；先發為賽後快照，屬日期隔離的假設性回測。')
        if season=='2024-25':st.write('此球季沿用後續2025–26研究的設定，屬較早球季重現驗證，不是當年鎖定模型的前瞻部署證據。')
    with st.expander('三模型逐場預測與下載'):
        player=st.selectbox('逐場球員',['全部球員']+sorted(frame['player_id'].unique().tolist()),key='research_player')
        selected=frame if player=='全部球員' else frame.loc[frame['player_id']==player]
        if threshold is not None:selected=selected.loc[selected['target_pts']>=threshold]
        human=pd.DataFrame({'球員':selected['player_id'],'日期':selected['GAMEDATE'],'對戰':selected['game_id'].str.rsplit('|',n=1).str[0],
            '實際上場時間（分鐘）':selected['actual_minutes'],'實際得分':selected['target_pts'],'五場均值':selected['pred_baseline']})
        for v in MODELS:
            human[NAMES[v]]=selected['pred_'+v];human[NAMES[v]+'絕對誤差']=(selected['target_pts']-selected['pred_'+v]).abs()
        st.dataframe(human,hide_index=True,width='stretch')
        st.download_button('下載三模型逐場預測',csv(human),f'predictions_{season}_{period}.csv','text/csv',key='research_predictions_download')
    st.caption('所有切換都只讀取已完成的預測，不重新訓練或產生新的測試結果。')
