"""Fill three missing official page records using NBA scorer report PDFs."""
import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/week1/venue_audit'
DATA=[
    ('0022500259','WAS','MIN','Target Center','Minneapolis','MN','Washington Wizards','MINNESOTA TIMBERWOLVES'),
    ('0022500260','DEN','NOP','Smoothie King Center','New Orleans','LA','Denver Nuggets','NEW ORLEANS PELICANS'),
    ('0022500261','SAC','OKC','Paycom Center','Oklahoma City','OK','Sacramento Kings','OKLAHOMA CITY THUNDER'),
]
for gid,away,home,arena,city,state,away_name,home_name in DATA:
    url=f'https://statsdmz.nba.com/pdfs/20251119/20251119_{away}{home}.pdf'
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    pdf=urllib.request.urlopen(req,timeout=20).read()
    text=PdfReader(io.BytesIO(pdf)).pages[0].extract_text()
    assert arena in text and 'November 19, 2025' in text
    assert away_name in text and home_name in text
    assert 'VISITOR:' in text and 'HOME:' in text
    dest=OUT/'pdfs'
    dest.mkdir(parents=True,exist_ok=True)
    (dest/(gid+'.pdf')).write_bytes(pdf)
    (dest/(gid+'.txt')).write_text(text,encoding='utf-8')
    record=dict(gameId=gid,GAMEDATE='2025-11-19',teams=sorted([away,home]),
                isNeutral=False,arena=dict(arenaName=arena,arenaCity=city,arenaState=state,arenaCountry='US'),
                homeTeam=home,awayTeam=away,gameEt=None,gameTimeUTC=None,
                source_url=url,fetched_at_utc=datetime.now(timezone.utc).isoformat(),
                pdf_sha256=hashlib.sha256(pdf).hexdigest(),evidence_method='official_pdf_home_arena',
                classification_basis='Official scorer report venue and HOME/VISITOR; arena checked against other official non-neutral home games of this home team')
    (OUT/'games'/(gid+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('saved official PDF evidence',gid,arena,flush=True)
