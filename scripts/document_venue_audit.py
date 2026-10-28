"""Publish the completed venue audit summary and synchronize project documentation."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
audit=ROOT/'outputs/week1/venue_audit'
summary=json.loads((audit/'validation.json').read_text(encoding='utf-8'))
merge=json.loads((ROOT/'outputs/week1/model_data/merge_validation.json').read_text(encoding='utf-8'))
assert summary['verified'] and summary['official_games_verified']==1230
assert merge['neutral_status_unverified_rows']==0 and merge['official_venue_verified_rows']==26568
n=summary['neutral_games']; regular=summary['non_neutral_games']
neutral_rows=merge['verified_neutral_positive_minute_rows']
ordinary_rows=merge['merged_positive_minute_rows']-neutral_rows
report=f'''# 全季 NBA 官方場地核對結果

核對日期：2026-10-21。已逐場核對保存資料內全部 **1,230 場**例行賽；中立場 {n} 場、非中立場 {regular} 場。不是以「未列例外」就推定非中立。

## 方法與證據

- {summary['official_game_page_count']} 場：從 NBA.com 公開比賽頁的 __NEXT_DATA__ 讀取 isNeutral、arena、homeTeam／awayTeam 與 gameEt，逐場驗證官方 game ID、日期與雙方球隊。
- {summary['official_pdf_home_arena_count']} 場：官方頁資料不可讀，改用官方 scorer's report 的球館與 HOME／VISITOR，並對照同主隊其他官方非中立場的同名主場球館。
- 三場補件為 2025-11-19 WAS／MIN、DEN／NOP、SAC／OKC；PDF 與文字保存於 pdfs/。它們的非中立判定依實際球館比對，不只依 HOME 文字。
- 只涵蓋已保存的 2025–26 例行賽集合，不混入季前賽、季後賽或 NBA Cup 決賽。

## 中立場清單

| 日期 | 兩隊 | 官方球館 | 官方來源 |
| --- | --- | --- | --- |
'''
for r in summary['neutral_game_details']:
    report+=f"| {r['GAMEDATE']} | {' / '.join(r['teams'])} | {r['venue']} | [NBA.com]({r['source_url']}) |\n"
report+=f'''
## 合併資料驗收

- 主合併檔 26,568 筆，is_neutral_site=1 為 {neutral_rows} 筆、=0 為 {ordinary_rows} 筆，無空值。
- is_home 在中立場兩隊皆為 0；非中立場依官方指定主隊決定。官方 designated_home_team 與球館另存。
- 原 MATCHUP、game_id、原九欄數值不變；83 筆零分鐘仍另存排除清單；兩份來源 SHA-256 不變。
- 原五筆方向差異的 53 筆主檔紀錄保留稽核旗標，已由官方場地證據處理。
- 場地完整不代表五場歷史資格、陣容賽前公布時間或模型已完成。

## 重現

核對結果：validation.json；逐場欄位：official_game_venues.csv；供合併器讀取：verified_venues.json；每場保存欄位：games/<官方 game ID>.json。

先執行 scripts/fetch_official_venues.py（需網路，成功結果會快取）；三場缺頁以 scripts/supplement_venue_pdfs.py 補件；再執行 scripts/validate_official_venues.py 及 scripts/merge_game_rosters.py。原 HTTP 批次的錯誤紀錄保留在 fetch_summary.json，三場後續補件後的完整狀態以 validation.json 為準。

以上是賽後取得的場地資料；不代表已重建當時賽程公布／更改時間。
'''
(audit/'README.md').write_text(report,encoding='utf-8')

p=ROOT/'outputs/week1/model_data/README.md'
t=p.read_text(encoding='utf-8')
t=t.replace('五場已核對中立場為 1；其他場次未核對為空，不任意填 0',f'全季已逐場核對：中立場為 1、非中立場為 0，無空值；共 {n}／{regular} 場')
t=t.replace('已確認中立場的場地與官方 scorer\'s report 來源','全季已確認的場地與官方比賽頁／記錄表來源')
t=t.replace('verified_neutral_site、source_agreement 或 unresolved','verified_neutral_site 或 verified_official_venue；原始方向差異旗標仍保留')
start=t.index('五場中立場涵蓋兩隊共')
end=t.index('\n\n## 驗收與限制',start)
t=t[:start]+f'''全季 1,230 場已核對，{n} 場中立場涵蓋兩隊共 {neutral_rows} 筆主檔紀錄，is_home 全為 0；其中 53 筆為原先來源方向差異。其餘 {ordinary_rows} 筆 is_neutral_site=0，is_home 按官方主隊決定。全部 26,568 筆通過接合資格，歷史窗口資格仍未計算。

逐場場地與來源見 [全季核對](../venue_audit/README.md) 與 official_game_venues.csv。''' +t[end:]
t=t.replace('十個單元測試','十二個單元測試').replace('不同對手不套用覆寫。','不同對手不套用覆寫，以及官方非中立場與缺少主隊時的驗證。')
p.write_text(t,encoding='utf-8')

p=ROOT/'TODO.md';t=p.read_text(encoding='utf-8')
t=t.replace('- [ ] 核對其餘場次的中立狀態，再決定 is_neutral_site 缺失處理；不把未核對直接填 0。',f'- [x] 全季 1,230 場官方場地核對完成（{n} 中立／{regular} 非中立），is_neutral_site 無空值；證據見 outputs/week1/venue_audit/。')
p.write_text(t,encoding='utf-8')

p=ROOT/'PROGRESS.md';t=p.read_text(encoding='utf-8')
heading='## 2026-10-21 全季場地核對完成'
if heading not in t:
    t=t.replace('更新日期：2026-10-21',f'''更新日期：2026-10-21

{heading}

全季 1,230 場已逐場核對：{summary['official_game_page_count']} 場官方頁欄位、{summary['official_pdf_home_arena_count']} 場官方記錄表與主場球館交叉核對；{n} 場中立、{regular} 場非中立。證據與完整清單位於 outputs/week1/venue_audit/。

合併檔 26,568 筆全部有場地結果，{neutral_rows} 筆中立、{ordinary_rows} 筆非中立；原始 CSV、原九欄、game_id 與零分鐘排除規則不變。十二項測試、來源雜湊與輸出核對通過。下方未核對場次的描述為較早進度，已由本次完整核對取代。''',1)
p.write_text(t,encoding='utf-8')

p=ROOT/'PRD.md';t=p.read_text(encoding='utf-8')
t=t.replace('其餘場次 is_neutral_site 未核對而留空，需於特徵流程前處理，不能直接猜填 0。',f'全季 1,230 場已逐場核對，{n} 場中立／{regular} 場非中立，is_neutral_site 已完整填入 1／0，無空值。')
p.write_text(t,encoding='utf-8')

p=ROOT/'docs/week1/integration_spec.md';t=p.read_text(encoding='utf-8')
t=t.replace('已核對中立場為 1；其他未核對場次留空，需補核對或確認缺失政策後才納入完整模型',f'全季官方核對完成：{n} 場中立為 1、{regular} 場非中立為 0，無空值')
t=t.replace('其他場次沿用來源一致的主客指標。官方證據見', '全季其餘場次現已依官方非中立場與指定主隊核對。完整證據另見 outputs/week1/venue_audit/；原五場官方證據見')
p.write_text(t,encoding='utf-8')

p=ROOT/'docs/superpowers/plans/2026-10-21-full-season-venues.md'
t=p.read_text(encoding='utf-8').replace('- [ ]','- [x]')
p.write_text(t,encoding='utf-8')
print('Published official season venue audit and updated project documents.')
