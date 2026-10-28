"""Validate complete saved NBA venue evidence before publishing the merge registry."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    raw_path=ROOT/'outputs/nba_2025_26/source_rows.json'
    raw=json.loads(raw_path.read_text(encoding='utf-8'))
    expected=defaultdict(list)
    for row in raw:
        expected[row['links'][2].rsplit('/',1)[-1]].append(row)
    out=ROOT/'outputs/week1/venue_audit'
    saved={p.stem:json.loads(p.read_text(encoding='utf-8')) for p in (out/'games').glob('*.json')}
    records=[]
    for gid,rows in sorted(expected.items()):
        path=out/'games'/(gid+'.json')
        if not path.exists():
            raise ValueError(f'Missing official evidence: {gid}')
        r=saved[gid]
        mm,dd,yyyy=rows[0]['values'][3].split('/')
        day=f'{yyyy}-{mm}-{dd}'
        teams=sorted({v['values'][1] for v in rows})
        assert r['gameId']==gid and r['GAMEDATE']==day
        assert isinstance(r['isNeutral'],bool)
        assert sorted([r['homeTeam'],r['awayTeam']])==teams==r['teams']
        method=r.get('evidence_method','official_game_page')
        if method=='official_game_page':
            assert r['gameEt'][:10]==day
            assert r['source_url'].startswith('https://www.nba.com/game/')
            assert len(r['html_sha256'])==64
            evidence_hash=r['html_sha256']
        elif method=='official_pdf_home_arena':
            pdf=out/'pdfs'/(gid+'.pdf')
            text=(out/'pdfs'/(gid+'.txt')).read_text(encoding='utf-8')
            assert hashlib.sha256(pdf.read_bytes()).hexdigest()==r['pdf_sha256']
            assert r['source_url'].startswith('https://statsdmz.nba.com/pdfs/')
            assert r['arena']['arenaName'] in text and 'November 19, 2025' in text
            assert 'HOME:' in text and 'VISITOR:' in text and not r['isNeutral']
            # Do not infer non-neutral merely from a HOME label: match the actual arena
            # to independently read official non-neutral home games of the same team.
            references=[v for v in saved.values() if v.get('evidence_method','official_game_page')=='official_game_page'
                        and not v['isNeutral'] and v['homeTeam']==r['homeTeam']
                        and v['arena']['arenaName']==r['arena']['arenaName']]
            assert references, f'No independent home-arena reference for {gid}'
            evidence_hash=r['pdf_sha256']
        else:
            raise ValueError(f'Unknown evidence method {method}')
        assert r['arena']['arenaName'] and r['fetched_at_utc']
        records.append(dict(GAMEDATE=day,teams=teams,official_game_id=gid,
                            venue=', '.join(str(r['arena'].get(k) or '') for k in ['arenaName','arenaCity','arenaCountry']),
                            designated_home_team=r['homeTeam'],
                            is_neutral_site=r['isNeutral'],source_url=r['source_url'],
                            verified_at_utc=r['fetched_at_utc'],evidence_method=method,
                            source_payload_sha256=evidence_hash))
    keys=[(r['GAMEDATE'],tuple(r['teams'])) for r in records]
    assert len(set(keys))==len(records)==1230
    neutral=[r for r in records if r['is_neutral_site']]
    registry=out/'verified_venues.json'
    registry.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fields=['official_game_id','GAMEDATE','teams','venue','designated_home_team','is_neutral_site','evidence_method','source_url']
    with (out/'official_game_venues.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\r\n')
        writer.writeheader()
        for row in records:
            writer.writerow({k:json.dumps(row[k]) if k=='teams' else row[k] for k in fields})
    summary=dict(verified=True,scope='NBA 2025-26 saved regular season game set',
                 expected_games=len(expected),official_games_verified=len(records),
                 neutral_games=len(neutral),non_neutral_games=len(records)-len(neutral),
                 game_id_date_and_team_checks=True,source_is_neutral_type_checked=True,
                 official_game_page_count=sum(r['evidence_method']=='official_game_page' for r in records),
                 official_pdf_home_arena_count=sum(r['evidence_method']=='official_pdf_home_arena' for r in records),
                 registry_sha256=hashlib.sha256(registry.read_bytes()).hexdigest(),
                 historical_source_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                 neutral_game_details=neutral,
                 method='Official NBA.com isNeutral/arena/home-away fields; three unavailable pages supplemented by official scorer PDFs and same-team verified home-arena cross-check; no inference from missing exceptions')
    (out/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
