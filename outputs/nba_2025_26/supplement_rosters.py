import json
import unicodedata
from pathlib import Path

root=Path(__file__).parent
raw=json.loads((root/'source_rows.json').read_text(encoding='utf-8'))
boxes=json.loads((root/'roster_boxes.json').read_text(encoding='utf-8'))
normalize=lambda s: ''.join(c for c in unicodedata.normalize('NFKD',s) if c.isalnum()).casefold()
supplements=[
 ('0022500259','WASMIN',[
  [('Bilal Coulibaly','29:20'),('Khris Middleton','27:27'),('Marvin Bagley III','30:34'),('Kyshawn George','37:22'),('CJ McCollum','30:57'),('Cam Whitmore','19:38'),('Corey Kispert','10:43'),('Tristan Vukcevic','20:01'),('Bub Carrington','16:37'),('Tre Johnson','17:23')],
  [('Anthony Edwards','36:28'),('Julius Randle','32:36'),('Rudy Gobert','36:05'),('Donte DiVincenzo','36:02'),('Mike Conley','27:25'),('Naz Reid','27:19'),('Jaylen Clark','25:49'),('Rob Dillingham','08:39'),('Bones Hyland','09:37')]
 ]),
 ('0022500260','DENNOP',[
  [('Peyton Watson','34:21'),('Zeke Nnaji','26:36'),('Nikola Jokic','34:26'),('Cameron Johnson','32:28'),('Jamal Murray','35:33'),('Tim Hardaway Jr.','23:04'),('Bruce Brown','27:01'),('Spencer Jones','10:31'),('Jonas Valanciunas','16:00')],
  [('Trey Murphy III','35:57'),('Zion Williamson','29:06'),('Derik Queen','30:25'),('Herbert Jones','29:00'),('Jeremiah Fears','25:28'),('Yves Missi','12:21'),('Saddiq Bey','18:54'),('Jose Alvarado','18:54'),('Jordan Hawkins','17:37'),('Micah Peavy','17:04'),('Kevon Looney','05:14')]
 ]),
 ('0022500261','SACOKC',[
  [('DeMar DeRozan','34:55'),('Precious Achiuwa','26:12'),('Drew Eubanks','22:34'),('Zach LaVine','30:09'),('Russell Westbrook','29:37'),('Malik Monk','27:15'),('Nique Clifford','08:21'),('Maxime Raynaud','18:36'),('Dennis Schroder','31:45'),('Keon Ellis','10:36')],
  [('Luguentz Dort','33:04'),('Chet Holmgren','31:01'),('Isaiah Hartenstein','26:31'),('Cason Wallace','26:05'),('Shai Gilgeous-Alexander','35:34'),('Isaiah Joe','20:20'),('Jaylin Williams','09:40'),('Branden Carlson','04:56'),('Ajay Mitchell','29:05'),('Alex Caruso','20:44'),('Chris Youngblood','01:30'),('Brooks Barnhizer','01:30')]
 ])
]
for gid,match,teams in supplements:
 assert gid not in {b['gid'] for b in boxes}
 game=[r for r in raw if r['links'][2].split('/')[-1]==gid]
 names={normalize(r['values'][0]):r for r in game}
 output=[]
 for players in teams:
  team=[]
  for i,(name,minutes) in enumerate(players):
   r=names[normalize(name)]
   team.append({'name':r['values'][0],'pid':r['links'][0].split('/')[-1],'position':['F','F','C','G','G'][i] if i<5 else '', 'min':minutes})
  output.append(team)
 assert len(game)==sum(len(t) for t in teams)
 boxes.append({'gid':gid,'title':match+' 2025-11-19 Official Scorers Report','source':f'https://statsdmz.nba.com/pdfs/20251119/20251119_{match}.pdf','teams':output})
(root/'roster_boxes.json').write_text(json.dumps(boxes,ensure_ascii=False),encoding='utf-8')
print({'games':len(boxes),'supplemented_games':len(supplements)})
