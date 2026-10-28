"""Small read-only official NBA HTML probe, saving a local evidence copy."""
import json
import re
import urllib.request
from pathlib import Path

url='https://www.nba.com/game/hou-vs-okc-0022500001'
req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
html=urllib.request.urlopen(req,timeout=15).read().decode('utf-8')
path=Path(__file__).resolve().parents[1]/'outputs/week1/venue_audit'
path.mkdir(parents=True,exist_ok=True)
(path/'probe.html').write_text(html,encoding='utf-8')
match=re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',html,re.S)
print('bytes',len(html),'next_data',bool(match),'arena_position',html.find('arenaName'))
if match:
    data=json.loads(match.group(1))
    (path/'probe_next_data.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    print('props keys',list(data.get('props',{}).get('pageProps',{})))
