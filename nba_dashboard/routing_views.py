"""Two-season history using the user-fixed 25-point routing policy."""
from pathlib import Path
import pandas as pd
import streamlit as st
from .data import DataIssue
from .routing_data import load_routing,revision,recent_games,comparison_metrics,SOURCE_NAMES,RULE,ROUTE,NAMES
from .views import _line_chart

ROOT=Path(__file__).resolve().parents[1]

@st.cache_data(show_spinner='載入兩季切換回測資料…')
def cached_bundle(root,version):return load_routing(Path(root))


def bundle_or_stop():
    try:return cached_bundle(str(ROOT),revision(ROOT))
    except DataIssue as exc:st.error(str(exc));st.stop()


def csv(frame):return frame.to_csv(index=False).encode('utf-8-sig')


def ensure_option(key,options):
    if st.session_state.get(key) not in options:st.session_state.pop(key,None)


def render_history():
    season=st.selectbox('球季',['2025-26','2024-25','2026-27'],format_func=lambda s:s.replace('-','–')+' 例行賽',key='history_season')
    if season=='2026-27':
        from .current_views import render_current_history
        render_current_history();return
    bundle=bundle_or_stop()
    period='test'
    frame=bundle.predictions[season][period]
    st.caption('NBA 得分研究室 · '+season.replace('-','–')+' 例行賽')
    st.title('歷史回測')
    st.write('依賽前近5場平均得分，切換直接得分模型或五場均值增減模型。')
    st.caption(RULE)
    st.caption('兩分支均使用25項特徵（已含上季表現），訓練時實際30+場次給2倍權重。以出賽為前提；歷史先發來自賽後快照，屬日期隔離的假設性回測。')
    st.caption('固定顯示3月至季末。此期間已用於測試或追加研究，不能當作未使用的新資料；25分門檻依使用者指定固定，不代表已證明優於單一模型。')
    if season=='2024-25':st.caption('2024–25沿用後續球季研究設定作較早球季重現；本次套用相同25分規則，沒有依此球季再挑門檻。')
    players=sorted(frame.player_id.unique().tolist())
    player_key=f'history_player_{season}_{period}'
    ensure_option(player_key,players)
    preferred=st.session_state.get('last_history_player')
    initial=players.index(preferred) if preferred in players else 0
    left,right=st.columns([1,1.5]);player=left.selectbox('球員',players,index=initial,key=player_key)
    st.session_state['last_history_player']=player
    games=frame.loc[frame.player_id.eq(player)].sort_values(['GAMEDATE','game_id'],ascending=False)
    ids=games.game_id.tolist();game_key=f'history_game_{season}_{period}_{player}';ensure_option(game_key,ids)
    game=right.selectbox('比賽',ids,format_func=lambda key:key.rsplit('|',1)[1]+' · '+key.rsplit('|',1)[0],key=game_key)
    r=games.loc[games.game_id.eq(game)].iloc[0];pred=float(r['pred_'+ROUTE]);actual=float(r.target_pts);residual=actual-pred
    st.info(f'本場使用：{SOURCE_NAMES[r.route25_model]} · 近5場平均 {r.pred_baseline:.2f} 分'+('（達25分門檻）' if r.pred_baseline>=25 else '（低於25分門檻）'))
    cards=st.columns(5)
    for col,label,value in zip(cards,['模型預測','實際上場時間','實際得分','絕對誤差','五場均值基準'],[f'{pred:.2f} 分',f'{r.actual_minutes:.2f} 分鐘',f'{actual:.0f} 分',f'{abs(residual):.2f} 分',f'{r.pred_baseline:.2f} 分']):col.metric(label,value)
    cutoff=f'{int(season[:4])+1}-01-31'
    st.caption(f'{r.GAMEDATE} · 分支模型訓練截止 {cutoff} · '+(f'低估 {residual:.2f} 分' if residual>0 else f'高估 {-residual:.2f} 分' if residual<0 else '與實際相同'))
    high=comparison_metrics(frame,['baseline',ROUTE],30)
    if not high.empty:st.warning(f'本期間實際30分以上共 {int(high.iloc[0]["n"]):,} 筆，切換方案MAE {high.iloc[1]["mae"]:.2f} 分，五場均值基準 {high.iloc[0]["mae"]:.2f} 分。此為賽後分組，不能作為賽前切換條件。')
    history=recent_games(bundle.histories[season],player,r.GAMEDATE)
    col1,col2=st.columns([1.6,1])
    with col1:
        st.subheader('本場之前的近10場出賽')
        if history.empty:st.info('沒有可用的前日歷史。')
        else:
            trend=pd.DataFrame({'比賽日期':pd.to_datetime(history.GAMEDATE),'歷史實際得分':history.PTS})
            st.altair_chart(_line_chart(trend,'比賽日期',['歷史實際得分']),width='stretch')
            st.caption(f'共 {len(history)} 場同季正分鐘紀錄，不含本場、同日或後續比賽；包含當時尚未達預測資格的歷史場次。')
    with col2:
        st.subheader('這位球員的完整三月至季末')
        stats=comparison_metrics(games,['baseline',ROUTE]);st.metric('本球員平均誤差',f'{stats.iloc[1]["mae"]:.2f} 分')
        st.write(f'合格出賽 **{len(games)} 場** · 五場均值基準MAE **{stats.iloc[0]["mae"]:.2f} 分**')
        st.caption('這裡包含整段三月至季末，可能有目前比賽之後的紀錄；下方完整曲線亦同。')
    recent=pd.DataFrame({'比賽日期':history.GAMEDATE,'對戰':history.MATCHUP,'實際上場時間（分鐘）':history.actual_minutes,'實際得分':history.PTS})
    st.dataframe(recent,width='stretch',hide_index=True,column_config={'實際上場時間（分鐘）':st.column_config.NumberColumn(format='%.2f')})
    with st.expander('本場兩個分支各預測多少？'):
        st.write(f'直接25項：{r.pred_combined_prior25_w2:.2f} 分；五場均值增減25項：{r.pred_residual25_w2:.2f} 分。門檻只決定選哪一個，不混合平均。')
    with st.expander('查看三月至季末曲線與逐場紀錄',expanded=True):
        chronological=games.sort_values(['GAMEDATE','game_id'])
        trend=pd.DataFrame({'比賽日期':pd.to_datetime(chronological.GAMEDATE),'切換方案預測':chronological['pred_'+ROUTE],'實際得分':chronological.target_pts,'五場均值':chronological.pred_baseline})
        st.altair_chart(_line_chart(trend,'比賽日期',['切換方案預測','實際得分','五場均值']),width='stretch')
        table=pd.DataFrame({'比賽日期':chronological.GAMEDATE,'對戰':chronological.game_id.str.rsplit('|',n=1).str[0],
            '實際上場時間（分鐘）':chronological.actual_minutes,'實際得分':chronological.target_pts,'模型預測':chronological['pred_'+ROUTE],
            '近5場平均得分':chronological.pred_baseline,'本場選用模型':chronological.route25_model.map(SOURCE_NAMES),
            '直接25項預測':chronological.pred_combined_prior25_w2,'五場均值增減25項預測':chronological.pred_residual25_w2,
            '絕對誤差':(chronological.target_pts-chronological['pred_'+ROUTE]).abs()})
        st.dataframe(table,width='stretch',hide_index=True,column_config={c:st.column_config.NumberColumn(format='%.2f') for c in ['實際上場時間（分鐘）','模型預測','近5場平均得分','直接25項預測','五場均值增減25項預測','絕對誤差']})
        st.download_button('下載這位球員的回測紀錄',csv(table),f'player_backtest_{season}_{period}.csv','text/csv',key='history_download')
    st.caption('頁面只讀取既有預測再按固定規則切換，不重新訓練。未來比賽預測仍待新球季資料與正式先發來源驗收。')
