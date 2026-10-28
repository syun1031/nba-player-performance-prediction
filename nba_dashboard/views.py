"""Human-readable Streamlit views for immutable research predictions."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from .data import DataIssue,load_bundle,file_revision,recent_history,comparison_metrics,prediction_table,MODELS,PERIODS

ROOT=Path(__file__).resolve().parents[1]
FAMILIES={'linear':'線性迴歸','tree':'決策樹','xgboost':'XGBoost'}
FEATURES={'starter':'9項：基礎＋自己是否先發','basic':'8項：基礎特徵',
          'teammates':'366項：再加隊友身分','all':'全部特徵版本'}
DATASETS={'test':'最終測試｜3/1–4/12','validation':'二月驗證｜選模使用'}
COLORS=['#C85D32','#243C60','#8794A5']


@st.cache_data(show_spinner='載入已核對的回測成果…')
def _cached_bundle(root,revision):
    return load_bundle(Path(root))


def _bundle():
    try:
        return _cached_bundle(str(ROOT),file_revision(ROOT))
    except DataIssue as exc:
        st.error(str(exc))
        st.stop()


def _intro(title,description,bundle):
    st.caption('NBA 得分研究室 · 2025–26 例行賽')
    st.title(title)
    st.write(description)
    st.caption(f'回測成果版本：{bundle.version} · 已鎖定主模型：XGBoost＋自己是否先發 · 以出賽為前提')
    st.caption('歷史先發來自賽後快照，屬日期隔離的假設性回測，不能視為完整賽前資訊重建。')


def _risk(frame,primary,period):
    result=comparison_metrics(frame,['baseline',primary],threshold=30)
    if result.empty:
        return
    baseline=result.iloc[0]; model=result.iloc[1]
    source='最終測試' if period=='test' else '二月驗證'
    st.warning(f'{source}中，實際30分以上共 {int(model["n"]):,} 筆，主模型平均誤差 {model["mae"]:.2f} 分，'
               f'五場均值基準為 {baseline["mae"]:.2f} 分。高得分場次仍是限制；此分組是賽後才知道的結果。')


def _csv(frame):
    return frame.to_csv(index=False).encode('utf-8-sig')


def _line_chart(data,x,series):
    long=data.melt(id_vars=[x],value_vars=series,var_name='曲線',value_name='得分')
    return alt.Chart(long).mark_line(point=True).encode(
        x=alt.X(x+':T',title='比賽日期',axis=alt.Axis(format='%m/%d')),
        y=alt.Y('得分:Q',title='得分',scale=alt.Scale(zero=True)),
        color=alt.Color('曲線:N',scale=alt.Scale(domain=series,range=COLORS[:len(series)]),legend=alt.Legend(title=None,orient='bottom')),
        tooltip=[alt.Tooltip(x+':T',title='日期',format='%Y-%m-%d'),'曲線:N',alt.Tooltip('得分:Q',format='.2f')],
    ).properties(height=290)


def render_history():
    bundle=_bundle(); frame=bundle.predictions['test']
    _intro('歷史回測','選一位球員與一場比賽，對照模型預測、實際得分與當時之前的表現。',bundle)
    players=sorted(frame['player_id'].unique().tolist())
    col1,col2=st.columns([1,1.5])
    player=col1.selectbox('球員',players,key='history_player')
    games=frame.loc[frame['player_id']==player].sort_values(['GAMEDATE','game_id'],ascending=False)
    game_id=col2.selectbox('比賽',games['game_id'].tolist(),
        format_func=lambda key:key.rsplit('|',1)[1]+' · '+key.rsplit('|',1)[0],key='history_game')
    row=games.loc[games['game_id']==game_id].iloc[0]
    predicted=float(row['pred_'+bundle.primary]); actual=float(row['target_pts'])
    residual=actual-predicted
    cards=st.columns(4)
    cards[0].metric('模型預測',f'{predicted:.2f} 分')
    cards[1].metric('實際得分',f'{actual:.0f} 分')
    cards[2].metric('絕對誤差',f'{abs(residual):.2f} 分')
    cards[3].metric('五場均值基準',f'{row["pred_baseline"]:.2f} 分')
    direction=f'模型低估 {residual:.2f} 分' if residual>0 else f'模型高估 {-residual:.2f} 分' if residual<0 else '預測與實際相同'
    st.caption(f'{row["GAMEDATE"]} · {direction}。五場均值基準代表直接用此前五場平均得分作預測。')
    _risk(frame,bundle.primary,'test')
    left,right=st.columns([1.6,1])
    with left:
        st.subheader('本場之前的近期表現')
        history=recent_history(bundle.history,player,row['GAMEDATE'])
        if history.empty:
            st.info('目前沒有這場比賽之前的可用出賽紀錄。')
        else:
            trend=pd.DataFrame({'比賽日期':history['GAMEDATE'],'歷史實際得分':history['PTS']})
            st.altair_chart(_line_chart(trend,'比賽日期',['歷史實際得分']),width='stretch')
            st.caption(f'顯示本場之前最近 {len(history)} 場正分鐘出賽；不含本場、同日或後續比賽。')
    with right:
        st.subheader('這位球員的完整測試期')
        st.caption('以下統計涵蓋3/1–4/12所有合格預測，包含目前選定比賽之後的紀錄。')
        stats=comparison_metrics(games,['baseline',bundle.primary])
        st.metric('本球員平均誤差',f'{stats.iloc[1]["mae"]:.2f} 分')
        st.write(f'合格出賽 **{len(games)} 場** · 五場均值基準平均誤差 **{stats.iloc[0]["mae"]:.2f} 分**')
        st.caption('這是同一球員在整個測試期間的描述，不代表每場都只差這麼多。')
    with st.expander('查看完整測試期曲線與逐場紀錄',expanded=True):
        chronological=games.sort_values(['GAMEDATE','game_id'])
        trend=pd.DataFrame({'比賽日期':pd.to_datetime(chronological['GAMEDATE']),
                            '模型預測':chronological['pred_'+bundle.primary],
                            '實際得分':chronological['target_pts'],'五場均值':chronological['pred_baseline']})
        st.altair_chart(_line_chart(trend,'比賽日期',['模型預測','實際得分','五場均值']),width='stretch')
        table=prediction_table(chronological,bundle.primary)
        st.dataframe(table,width='stretch',hide_index=True,column_config={
            k:st.column_config.NumberColumn(format='%.2f') for k in ['模型預測','五場均值','絕對誤差']})
        st.download_button('下載這位球員的回測紀錄',_csv(table),'player_backtest.csv','text/csv',key='history_download')
    st.caption('目前提供歷史研究展示；未來預測需先完成新球季資料與正式先發來源驗收。')


def render_comparison():
    bundle=_bundle()
    _intro('模型比較','用同一批比賽比較模型，並把整體表現與高分場次分開看。',bundle)
    period=st.radio('資料期間',list(DATASETS),format_func=DATASETS.get,horizontal=True,key='comparison_period')
    frame=bundle.predictions[period]
    start,end,n=PERIODS[period]
    st.caption(f'{start} 至 {end} · 完整資料 {n:,} 筆 · '+('測試已完成，不再用來調參或重選模型。' if period=='test' else '這批資料已用於選模，不能當作獨立最終測試。'))
    c1,c2=st.columns(2)
    feature=c1.selectbox('特徵版本',list(FEATURES),format_func=FEATURES.get,key='comparison_features')
    threshold=c2.selectbox('觀察場次',[None,20,30],format_func=lambda value:'全部出賽' if value is None else f'實際得分 {value} 分以上',key='comparison_threshold')
    families=st.multiselect('比較模型',list(FAMILIES),default=list(FAMILIES),format_func=FAMILIES.get,key='comparison_families')
    primary_stats=comparison_metrics(frame,['baseline',bundle.primary],threshold)
    if primary_stats.empty:
        st.info('目前的觀察範圍沒有合格比賽，請調整分組。')
        return
    baseline=primary_stats.iloc[0]; model=primary_stats.iloc[1]
    st.markdown('**鎖定主模型｜XGBoost＋自己是否先發（九項）**')
    cards=st.columns(4)
    cards[0].metric('主模型 MAE',f'{model["mae"]:.3f} 分')
    cards[1].metric('基準 MAE',f'{baseline["mae"]:.3f} 分')
    delta=float(model['mae']-baseline['mae'])
    cards[2].metric('模型減基準',f'{delta:+.3f} 分',help='負值代表主模型平均誤差較低；正值代表較高。')
    cards[3].metric('樣本數',f'{int(model["n"]):,}')
    st.caption('上述卡片始終是鎖定主模型；切換下方比較版本不會更換主模型。MAE是平均差幾分，RMSE會更重視大誤差，兩者越低越好。')
    _risk(frame,bundle.primary,period)
    if threshold is not None:
        st.info('此分組依賽後實際得分觀察；預測前不知道哪些場次會高分。20分以上與30分以上互相重疊，不能相加。')
    selected=['baseline']+[key for key in MODELS if key.rsplit('_',1)[0] in families and (feature=='all' or key.endswith('_'+feature))]
    stats=comparison_metrics(frame,selected,threshold)
    if not families:
        st.info('目前只顯示五場均值基準；可勾選上方模型加入比較。')
    names={key:('五場均值基準' if key=='baseline' else FAMILIES[key.rsplit('_',1)[0]]+' · '+FEATURES[key.rsplit('_',1)[1]].split('：')[0]) for key in selected}
    plot=stats.assign(比較版本=stats['model_variant'].map(names))
    cols=st.columns(2)
    for container,metric,title in [(cols[0],'mae','平均誤差 MAE'),(cols[1],'rmse','大誤差指標 RMSE')]:
        chart=alt.Chart(plot).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X(metric+':Q',title='分（越低越好）'),
            y=alt.Y('比較版本:N',sort=[names[k] for k in selected],title=None,
                    axis=alt.Axis(labelOverlap=False,labelLimit=160)),
            color=alt.condition(alt.datum.model_variant==bundle.primary,alt.value(COLORS[0]),alt.value(COLORS[1])),
            tooltip=['比較版本:N',alt.Tooltip('n:Q',title='同樣本數',format=','),alt.Tooltip(metric+':Q',format='.3f')],
        ).properties(height=max(240,45*len(stats)),title=title)
        container.altair_chart(chart,width='stretch')
    table=pd.DataFrame({'模型':stats['model_variant'].map(lambda key:'五場均值基準' if key=='baseline' else FAMILIES[key.rsplit('_',1)[0]]),
        '特徵版本':stats['model_variant'].map(lambda key:'最近五場平均' if key=='baseline' else FEATURES[key.rsplit('_',1)[1]]),
        '已鎖定':stats['model_variant'].eq(bundle.primary),'樣本數':stats['n'],
        'MAE':stats['mae'],'RMSE':stats['rmse']})
    st.dataframe(table,width='stretch',hide_index=True,column_config={k:st.column_config.NumberColumn(format='%.3f') for k in ['MAE','RMSE']})
    st.download_button('下載目前比較表',_csv(table),'model_comparison.csv','text/csv',key='comparison_download')
    st.caption('橘色表示已鎖定主模型，其他版本為研究比較；表格維持固定模型順序，不按測試排名重新選模。')
    with st.expander('模型用了什麼資料？'):
        st.write('訓練：2025/10/21–2026/1/31；驗證：2026/2；最終測試：2026/3/1–4/12。模型權重與前處理固定，歷史摘要只用目標日期之前的比賽。')
        st.write('九項特徵：近五場得分、近最多十場得分、近五場分鐘／籃板／助攻平均、主場、中立場、個人休息天數、自己是否先發。個人與球隊各需至少五場歷史才預測。')
        st.write('隊友身分版本額外包含356位訓練期隊友旗標與未知隊友人數；改善不穩定，沒有放進主模型。先發隊友缺席特徵尚未實作。')
    st.caption('切換頁面與篩選只讀取既有預測，不會重新訓練、產生新模型預測或對未來比賽給出數字。')
