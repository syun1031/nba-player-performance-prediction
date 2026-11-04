"""Future page: official source check first, inference only after every gate passes."""
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import pandas as pd
import streamlit as st
from .data import DataIssue
from .future_data import load_future,history_issues,HISTORY_REASONS
from .future_source import utc_now,aware,query_schedule,query_context,assess_context,SourceIssue,REASONS
from .future_inference import build_features,load_models,make_prediction,prediction_csv_row
from .routing_data import SOURCE_NAMES,RULE
from .views import _line_chart

ROOT=Path(__file__).resolve().parents[1]
TAIPEI=ZoneInfo('Asia/Taipei')
FEATURE_REASONS={**HISTORY_REASONS,**REASONS,
    'insufficient_history':'這位球員的本季有效出賽不足五場。','insufficient_team_history':'本場球隊的本季完賽不足五場。',
    'insufficient_opponent_history':'對手的本季完賽不足五場。','player_not_active':'這位球員未列入本場官網確認的啟用名單。',
    'missing_pregame_evidence':'正式先發與時間檢查尚未全部通過。'}


@st.cache_resource(show_spinner='載入固定預測模型…')
def get_models(root,version,_bundle):return load_models(Path(root),_bundle)


def clear_result():st.session_state.pop('fp_result',None)


def pending_notice(rows,day,now):
    """Describe pending games relative to the selected NBA date, not wall clock alone."""
    games={}
    for row in rows:games.setdefault(row.get('nba_game_id',row['GAMEDATE']),[]).append(row)
    prior=upcoming=other=0
    for records in games.values():
        if any(r['GAMEDATE']<day for r in records):
            prior+=1;continue
        try:scheduled=all(str(r['game_status'])=='1' and aware(r['scheduled_start_utc'])>now for r in records)
        except (KeyError,ValueError,TypeError):scheduled=False
        if scheduled:upcoming+=1
        else:other+=1
    if prior:
        st.warning(f'所選NBA日期之前仍有 {prior} 場尚未收錄為完賽，會先停用預測；請等待每日更新或在GitHub Actions手動補跑。')
    if upcoming:
        st.caption(f'資料包另列有 {upcoming} 場尚未開賽的賽程，不屬於所選日期之前的歷史，不影響歷史完整性檢查。')
    if other:
        st.caption(f'另有 {other} 場尚未收錄為完賽（可能進行中或狀態待核對），不屬於所選日期之前的歷史。')


def recent_table(bundle,pid,day):
    rows=[r for r in bundle['raw_history'] if r['nba_player_id']==str(pid) and r['GAMEDATE']<day]
    rows=sorted(rows,key=lambda r:r['GAMEDATE'])[-10:]
    if not rows:return pd.DataFrame()
    return pd.DataFrame([{'比賽日期':r['GAMEDATE'],'球隊':r['TEAM'],'對戰':r['MATCHUP'],
                          '實際上場時間':float(r['MIN']),'實際得分':float(r['PTS'])} for r in rows])


def render_future():
    st.caption('NBA 得分研究室 · 2026–27 例行賽')
    st.title('未來預測')
    st.write('先查詢NBA官網正式先發，資料與資格通過後，在開賽前30分鐘內產生球員得分預測。')
    st.caption(RULE)
    st.caption('以球員會出賽為前提。使用訓練至2026-01-31的既有模型，本季前瞻準確率尚未建立。')
    try:bundle=load_future(ROOT)
    except DataIssue as exc:st.error(str(exc));st.stop()
    m=bundle['manifest'];now=utc_now()
    st.caption(f"歷史查詢至 {m['through']} · 已收錄 {m.get('completed_games',len(bundle['team_games'])//2)} 場完賽 · 資料取得 {aware(m['collected_at_utc']).astimezone(TAIPEI):%Y-%m-%d %H:%M}（台灣）")
    day=st.date_input('NBA比賽日期（美東日期）',value=now.astimezone(ZoneInfo('America/New_York')).date(),
        min_value=pd.Timestamp('2026-07-01').date(),max_value=pd.Timestamp('2027-06-30').date(),key='fp_day').isoformat()
    st.caption('開賽時間會轉為台灣時間顯示；歷史特徵仍按NBA日期隔離。')
    pending_notice(bundle['pending'],day,now)
    if st.button('查詢官網賽程',key='fp_schedule_query'):
        clear_result();st.session_state.pop('fp_context',None);st.session_state.pop('fp_source_error',None)
        try:
            with st.spinner('查詢NBA官網賽程…'):st.session_state['fp_schedule']={'day':day,**query_schedule(day)}
            st.session_state.pop('fp_schedule_error',None)
        except (SourceIssue,ValueError) as exc:
            st.session_state.pop('fp_schedule',None);st.session_state['fp_schedule_error']=str(exc)
    if st.session_state.get('fp_schedule_error'):st.warning(st.session_state['fp_schedule_error'])
    daily=st.session_state.get('fp_schedule')
    if not daily or daily['day']!=day:
        st.info('選擇日期後，查詢官網賽程。');return
    games=daily['games']
    if not games:st.info('這個NBA日期沒有NBA例行賽，或官網尚未列出賽程。');return
    byid={g['nba_game_id']:g for g in games}
    gamekey='fp_game_'+day
    if st.session_state.get(gamekey) not in byid:st.session_state.pop(gamekey,None)
    gid=st.selectbox('比賽',list(byid),key=gamekey,format_func=lambda g:byid[g]['away_team']+' @ '+byid[g]['home_team']+' · '+aware(byid[g]['scheduled_start_utc']).astimezone(TAIPEI).strftime('%m/%d %H:%M 台灣'))
    selected=byid[gid]
    for code in history_issues(bundle,day,utc_now()):st.warning(FEATURE_REASONS.get(code,code))
    if st.button('查詢官網正式先發',key='fp_lineup_query'):
        clear_result();st.session_state.pop('fp_context',None);st.session_state.pop('fp_source_error',None)
        try:
            with st.spinner('核對官網賽程與兩隊名單…'):st.session_state['fp_context']=query_context(selected)
        except SourceIssue as exc:st.session_state['fp_source_error']=str(exc)
    if st.session_state.get('fp_source_error'):st.warning(st.session_state['fp_source_error'])
    context=st.session_state.get('fp_context')
    if not context or context['schedule']['nba_game_id']!=gid or context['schedule']['GAMEDATE']!=day:
        st.info('等待查詢並確認本場正式先發。');return
    current=assess_context(context['schedule'],*context['sources'],utc_now())
    context={**current,'schedule_source':context.get('schedule_source')}
    for code in current['reasons']:st.info(FEATURE_REASONS.get(code,code))
    options={};columns=st.columns(2)
    for col,(team,t) in zip(columns,current['teams'].items()):
        col.subheader(team+' · 官網名單')
        col.write('先發：'+'、'.join(p['name'] for p in t['players'] if p['nba_player_id'] in t['starters']))
        col.caption('啟用名單 '+str(len(t['active']))+' 人；未啟用者不能選擇。')
        for p in t['players']:
            if p['active']:options[p['nba_player_id']]=team+' · '+p['name']+('（先發）' if p['nba_player_id'] in t['starters'] else '（替補）')
    if not options:
        st.button('產生預測',key='fp_predict',disabled=True);return
    playerkey='fp_player_'+day+'_'+gid
    if st.session_state.get(playerkey) not in options:st.session_state.pop(playerkey,None)
    pid=st.selectbox('球員',list(options),format_func=options.get,key=playerkey)
    f=build_features(bundle,context,pid,utc_now())
    for code in filter(None,f['exclusion_reason'].split(';')):
        if code not in current['reasons']:st.warning(FEATURE_REASONS.get(code,'資料未通過檢查：'+code))
    metrics=st.columns(3)
    for col,key,label in zip(metrics,['history_count','team_history_count','opponent_history_count'],['球員歷史出賽','球隊歷史完賽','對手歷史完賽']):
        col.metric(label,str(f.get(key,0))+' 場')
    st.caption('三項均至少五場；只計NBA目標日期之前的同季紀錄。')
    if st.button('產生預測',key='fp_predict',disabled=not f['eligible']):
        clear_result()
        try:
            with st.spinner('重新確認先發與賽程，再計算預測…'):
                fresh=query_context(selected);st.session_state['fp_context']=fresh
                latest=load_future(ROOT)
                check=build_features(latest,fresh,pid,utc_now())
                if not check['eligible']:raise DataIssue('資格或先發已變更，請重新查詢：'+check['exclusion_reason'])
                version=json.dumps(latest['manifest'].get('models',{}),sort_keys=True)
                models=get_models(str(ROOT),version,latest)
                # Clock check after potentially slow model loading, before inference.
                result=make_prediction(latest,fresh,pid,models,utc_now())
                if utc_now()>=aware(result['scheduled_start_utc']):raise DataIssue('計算期間已到開賽時間，這次不列為賽前預測。')
                st.session_state['fp_result']=result
                st.rerun()
        except (SourceIssue,DataIssue,ValueError) as exc:st.error(str(exc))
    result=st.session_state.get('fp_result')
    if result and result['nba_game_id']==gid and result['nba_player_id']==pid and result['history_version']==m['version']:
        old=context['sources']
        active=current['ready'] and f['eligible'] and utc_now()<aware(result['scheduled_start_utc']) and all(result['sources'][i]['sha256']==old[i]['sha256'] for i in range(2))
        if not active:st.warning('以下為先前已產生的紀錄；已開賽或來源已更新，不能視為目前有效預測。')
        cols=st.columns(3);cols[0].metric('預測得分',f"{result['prediction']:.2f} 分")
        cols[1].metric('近五場均分',f"{result['baseline']:.2f} 分")
        cols[2].metric('本場身分','先發' if result['is_starter'] else '替補')
        st.info('使用：'+SOURCE_NAMES[result['model_variant']])
        st.caption('產生時間：'+aware(result['prediction_at_utc']).astimezone(TAIPEI).strftime('%Y-%m-%d %H:%M:%S 台灣'))
        with st.expander('查看兩個模型與本次資料紀錄'):
            st.write(f"直接模型 {result['direct_prediction']:.2f} 分 · 五場均值增減模型 {result['offset_prediction']:.2f} 分")
            st.json(result['features'])
        payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
        st.download_button('下載完整預測紀錄 JSON',payload,file_name=f"pregame_{gid}_{pid}.json",mime='application/json',on_click='ignore')
        row=prediction_csv_row(result)
        st.download_button('下載預測摘要 CSV',pd.DataFrame([row]).to_csv(index=False).encode('utf-8-sig'),file_name=f"pregame_{gid}_{pid}.csv",mime='text/csv',on_click='ignore')
        st.caption('請下載留存；雲端頁面的暫存紀錄不保證永久保存。')
    recent=recent_table(bundle,pid,day)
    st.subheader('目標比賽之前的近10場出賽')
    if recent.empty:st.info('尚無這位球員的本季正分鐘出賽紀錄。')
    else:
        st.altair_chart(_line_chart(recent,'比賽日期',['實際得分']),width='stretch')
        st.dataframe(recent,hide_index=True,width='stretch',column_config={'實際上場時間':st.column_config.NumberColumn('實際上場時間',format='%.2f 分鐘')})
    with st.expander('資料來源與檢查時間'):
        for receipt in context['sources']:
            st.write(receipt['url']);st.caption('本次取得：'+receipt['observed_at_utc'])
        st.caption('官網無時區的原始timestamp只隨JSON保存，不當成UTC發布時間。')
