"""Current-season postgame results, clearly separate from live forecasts."""
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from .data import DataIssue
from .current_season import load_current
from .future_data import load_future
from .future_source import aware
from .routing_data import RULE,SOURCE_NAMES
from .journal_theme import journal_line_chart as _line_chart

ROOT=Path(__file__).resolve().parents[1]


def render_current_history():
    st.caption('NBA 得分研究室 · 2026–27 例行賽')
    heading=st.empty();heading.title('歷史回測')
    st.write('本季開季至今，對已完賽且符合五場門檻的球員出賽回算。')
    st.caption('日期隔離的假設性回測：只使用比賽日期前的歷史，但本場先發來自賽後資料；並非當時留存的賽前預測。')
    with st.expander('資料與模型說明'):
        st.caption(RULE+'；兩個固定模型均訓練至2026-01-31，沒有每日重訓。')
        st.caption('每日台灣17:00由GitHub Actions開始更新；實際發布時間依執行狀況而定。未納入者包含五場不足或較早比賽尚待釐清。')
    try:
        data=load_current(ROOT)
    except DataIssue as exc:st.error(str(exc));return
    m=data['manifest'];frame=data['frame']
    stamp=aware(m['collected_at_utc']).astimezone(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M')
    st.caption(f"資料取得 {stamp}（台灣） · 最近完賽 {m['last_completed_date'] or '尚無'} · 合格 {m['eligible_rows']:,} 筆 · 未納入 {m['excluded_rows']:,} 筆")
    if frame.empty:st.info('本季尚無符合預測資格的已完賽出賽，等待資料累積。');return
    try:history=load_future(ROOT)
    except DataIssue as exc:st.error(str(exc));return
    players=sorted(frame.player_id.unique().tolist());pkey='current_history_player'
    if st.session_state.get(pkey) not in players:st.session_state.pop(pkey,None)
    left,right=st.columns([1,1.5])
    player=left.selectbox('球員',players,key=pkey)
    heading.title(player)
    games=frame.loc[frame.player_id.eq(player)].sort_values(['GAMEDATE','game_id'],ascending=False)
    ids=games.game_id.tolist();gkey='current_history_game_'+player
    if st.session_state.get(gkey) not in ids:st.session_state.pop(gkey,None)
    gid=right.selectbox('比賽',ids,key=gkey,format_func=lambda v:v.rsplit('|',1)[1]+' · '+v.rsplit('|',1)[0])
    r=games.loc[games.game_id.eq(gid)].iloc[0]
    st.info('本場使用：'+SOURCE_NAMES[r.route25_model])
    cols=st.columns(5)
    for col,label,value in zip(cols,['模型預測','實際上場時間','實際得分','絕對誤差','五場均值基準'],
        [f'{r.pred_route25:.2f} 分',f'{r.actual_minutes:.2f} 分鐘',f'{r.target_pts:.0f} 分',f'{abs(r.target_pts-r.pred_route25):.2f} 分',f'{r.pred_baseline:.2f} 分']):col.metric(label,value)
    st.subheader('本場之前的近10場出賽')
    rows=sorted([x for x in history['raw_history'] if str(x['nba_player_id'])==str(r.nba_player_id) and x['GAMEDATE']<r.GAMEDATE],key=lambda x:x['GAMEDATE'])[-10:]
    recent=pd.DataFrame([{'比賽日期':x['GAMEDATE'],'對戰':x['MATCHUP'],'實際上場時間（分鐘）':float(x['MIN']),'實際得分':float(x['PTS'])} for x in rows])
    if not recent.empty:
        st.altair_chart(_line_chart(recent,'比賽日期',['實際得分']),width='stretch')
        st.dataframe(recent,hide_index=True,width='stretch')
    st.subheader('這位球員的本季合格出賽')
    st.caption(f"{len(games)} 場 · 平均絕對誤差 {(games.target_pts-games.pred_route25).abs().mean():.2f} 分。包含所選比賽之後的已完賽紀錄。")
    ordered=games.sort_values(['GAMEDATE','game_id'])
    trend=pd.DataFrame({'比賽日期':pd.to_datetime(ordered.GAMEDATE),'實際得分':ordered.target_pts,'切換方案預測':ordered.pred_route25,'五場均值':ordered.pred_baseline})
    st.altair_chart(_line_chart(trend,'比賽日期',['實際得分','切換方案預測','五場均值']),width='stretch')
    table=pd.DataFrame({'比賽日期':ordered.GAMEDATE,'對戰':ordered.game_id.str.rsplit('|',n=1).str[0],
        '實際上場時間（分鐘）':ordered.actual_minutes,'實際得分':ordered.target_pts,'模型預測':ordered.pred_route25,
        '近5場平均得分':ordered.pred_baseline,'本場選用模型':ordered.route25_model.map(SOURCE_NAMES),
        '直接25項預測':ordered.pred_combined_prior25_w2,'五場均值增減25項預測':ordered.pred_residual25_w2,
        '絕對誤差':(ordered.target_pts-ordered.pred_route25).abs()})
    st.dataframe(table,hide_index=True,width='stretch')
    st.download_button('下載這位球員的本季回測紀錄',table.to_csv(index=False).encode('utf-8-sig'),
        file_name='player_backtest_2026_27.csv',mime='text/csv',key='current_history_download')
