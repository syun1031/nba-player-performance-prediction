import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {Workbook} from '@oai/artifact-tool';
const dir='C:/Users/wangs/codex/outputs/nba_2025_26';
const raw=JSON.parse(await fs.readFile(`${dir}/source_rows.json`,'utf8'));
const boxes=JSON.parse(await fs.readFile(`${dir}/roster_boxes.json`,'utf8'));
const games=new Map();
for(const r of raw){
 const gid=r.links[2].split('/').filter(Boolean).at(-1);
 if(!games.has(gid))games.set(gid,[]);
 games.get(gid).push(r);
}
assert.equal(games.size,1230);
assert.equal(boxes.length,1230);
assert.equal(new Set(boxes.map(b=>b.gid)).size,1230);
const records=[];
const counts={games:1230,teamRows:2460,starters:0,players:0,excludedZero:0,excludedDNP:0,lessThanOneMinute:0};
for(const box of boxes){
 const src=games.get(box.gid);assert.ok(src);
 const away=src.find(r=>r.values[2].includes(' @ '));assert.ok(away);
 const [awayTeam,homeTeam]=away.values[2].split(' @ ');
 const [mm,dd,yyyy]=away.values[3].split('/');
 const date=`${yyyy}-${mm}-${dd}`;
 const matchup=`${awayTeam} @ ${homeTeam}`;
 const pidTeam=new Map(src.map(r=>[r.links[0].split('/').at(-1),r.values[1]]));
 assert.equal(box.teams.length,2);
 const used=new Set();
 for(const [side,team] of [awayTeam,homeTeam].entries()){
  const list=box.teams.find(t=>t.some(p=>pidTeam.get(p.pid)===team));
  assert.ok(list,box.gid+' missing team '+team);
  assert.equal(list.filter(p=>p.position).length,5,box.gid+' source starters');
  const players=[];
  for(const p of list){
   const match=/^(\d+):(\d{2})$/.exec(p.min||'');
   if(!match){assert.match(p.min||'',/DNP|DND|NWT|DID NOT/i);counts.excludedDNP++;continue;}
   const seconds=Number(match[1])*60+Number(match[2]);
   if(seconds===0){counts.excludedZero++;continue;}
   if(seconds<60)counts.lessThanOneMinute++;
   assert.equal(pidTeam.get(p.pid),team,box.gid+' '+p.name+' team');
   assert.ok(!used.has(p.pid),box.gid+' duplicate '+p.pid);used.add(p.pid);
   players.push({name:p.name.trim(),star:!!p.position,pid:p.pid});
  }
  players.sort((a,b)=>Number(b.star)-Number(a.star));
  assert.equal(players.filter(p=>p.star).length,5,box.gid+' retained starters');
  counts.starters+=5;counts.players+=players.length;
  records.push({gid:box.gid,date,matchup,team,side,players});
 }
 // All player-game logs with positive rounded minutes must be present.
 for(const r of src)if(Number(r.values[5])>0)assert.ok(used.has(r.links[0].split('/').at(-1)),box.gid+' missing participant');
}
records.sort((a,b)=>a.date.localeCompare(b.date)||a.matchup.localeCompare(b.matchup)||a.side-b.side);
const maxPlayers=Math.max(...records.map(r=>r.players.length));
const headers=['MATCHUP','GAMEDATE','TEAM',...Array.from({length:maxPlayers},(_,i)=>`PLAYER_${i+1}`)];
const rows=records.map(r=>[r.matchup,r.date,r.team,...r.players.map(p=>(p.star?'*':'')+p.name),...Array(maxPlayers-r.players.length).fill('')]);
const wb=Workbook.create();const sheet=wb.worksheets.add('Game rosters');
sheet.getRangeByIndexes(0,0,rows.length+1,headers.length).values=[headers,...rows];
sheet.getRange('A1:H7').format.font={name:'Arial',size:10};
sheet.getRange('A1:H1').format={fill:'#17408B',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'}};
sheet.getRange('A1:A7').format.columnWidth=18;sheet.getRange('B1:B7').format.columnWidth=15;sheet.getRange('C1:C7').format.columnWidth=9;
sheet.getRange('D1:H7').format.columnWidth=29;sheet.getRange('A1:H7').format.rowHeight=24;
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Game rosters!A1:H3',tableMaxRows:3,tableMaxCols:8,maxChars:2000})).ndjson);
const preview=await wb.render({sheetName:'Game rosters',range:'A1:H7',scale:1.3,format:'png'});
await fs.writeFile(`${dir}/roster_preview.png`,new Uint8Array(await preview.arrayBuffer()));
const grid=sheet.getRangeByIndexes(0,0,rows.length+1,headers.length).values;
assert.deepEqual(grid,[headers,...rows]);
const quote=v=>/[",\r\n]/.test(String(v))?'"'+String(v).replaceAll('"','""')+'"':String(v);
const csv=grid.map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n';
await fs.writeFile(`${dir}/nba_2025_26_regular_season_game_rosters.csv`,'\uFEFF'+csv,'utf8');
await fs.writeFile(`${dir}/roster_validation.json`,JSON.stringify({...counts,maxPlayers,headers,first:rows[0],last:rows.at(-1)},null,2));
console.log(JSON.stringify({...counts,maxPlayers,rows:rows.length}));
