import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { Workbook } from '@oai/artifact-tool';

const dir = 'C:/Users/wangs/codex/outputs/lebron_lakers_2018_2026';
const seasons = JSON.parse(await fs.readFile(`${dir}/source_seasons.json`, 'utf8'));
const expectedSeasons = Array.from({length:8},(_,i)=>`${2018+i}-${String(2019+i).slice(-2)}`);
assert.deepEqual(seasons.map(s=>s.season),expectedSeasons);
const expectedHeaders = ['MATCH UP','W/L','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','OREB','DREB','REB','AST','STL','BLK','TOV','PF','+/-'];
const months = {JAN:'01',FEB:'02',MAR:'03',APR:'04',MAY:'05',JUN:'06',JUL:'07',AUG:'08',SEP:'09',OCT:'10',NOV:'11',DEC:'12'};
const records = [];
const ids = new Set();
for(const s of seasons){
  assert.equal(s.type,'Regular Season');
  assert.equal(s.rows.length,parseInt(s.countText));
  assert.deepEqual(s.headers,expectedHeaders);
  for(const r of s.rows){
    assert.equal(r.values.length,22);
    const gid = r.game.split('/').filter(Boolean).at(-1);
    assert.match(gid,new RegExp(`^002${s.season.slice(2,4)}\\d{5}$`));
    assert.ok(!ids.has(gid),'Duplicate '+gid);
    ids.add(gid);
    const m = /^([A-Z]{3}) (\d{2}), (\d{4}) - (LAL (?:VS\.|@) [A-Z]{3})$/.exec(r.values[0]);
    assert.ok(m,'Unexpected matchup '+r.values[0]);
    const date = `${m[3]}-${months[m[1]]}-${m[2]}`;
    assert.match(r.values[2],/^\d{1,2}:\d{2}$/);
    assert.ok(Number(r.values[2].split(':')[1]) < 60);
    const numeric = [3,6,15,16,17,18].map(i=>Number(r.values[i]));
    assert.ok(numeric.every(Number.isFinite));
    assert.ok(Math.abs(Number(r.values[6])-Number(r.values[4])/Number(r.values[5])*100) <= 0.051);
    records.push({season:s.season,gid,values:[m[4],date,r.values[2],...numeric]});
  }
}
records.sort((a,b)=>a.values[1].localeCompare(b.values[1]));
assert.equal(records.length,479);
const prior = JSON.parse(await fs.readFile('C:/Users/wangs/codex/outputs/nba_2025_26/source_rows.json','utf8')).filter(r=>r.links[0]==='/stats/player/2544');
const current = records.filter(r=>r.season==='2025-26');
assert.equal(current.length,prior.length);
for(const r of current){
  const old = prior.find(p=>p.links[2].split('/').at(-1)===r.gid);
  assert.ok(old);
  assert.equal(old.values[1],'LAL');
  assert.deepEqual(r.values.slice(3),[6,9,18,19,20,21].map(i=>Number(old.values[i])));
}
const headers = ['MATCHUP','GAMEDATE','MIN','PTS','FG%','REB','AST','STL','BLK'];
const wb = Workbook.create();
const sheet = wb.worksheets.add('LeBron Lakers');
sheet.getRange(`A1:I${records.length+1}`).values = [headers,...records.map(r=>r.values)];
sheet.getRange('A1:I12').format.font = {name:'Arial',size:10};
sheet.getRange('A1:I1').format = {fill:'#552583',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'}};
sheet.getRange('A1:A12').format.columnWidth = 22;
sheet.getRange('B1:B12').format.columnWidth = 15;
sheet.getRange('C1:I12').format.columnWidth = 10;
sheet.getRange('A1:I12').format.rowHeight = 21;
sheet.getRange('E2:E12').setNumberFormat('0.0');
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'LeBron Lakers!A1:I4',tableMaxRows:4,tableMaxCols:9,maxChars:2000})).ndjson);
const preview = await wb.render({sheetName:'LeBron Lakers',range:'A1:I12',scale:1.5,format:'png'});
await fs.writeFile(`${dir}/preview.png`,new Uint8Array(await preview.arrayBuffer()));
const grid = sheet.getRange(`A1:I${records.length+1}`).values;
assert.deepEqual(grid,[headers,...records.map(r=>r.values)]);
const quote = v => /[",\r\n]/.test(String(v)) ? `"${String(v).replaceAll('"','""')}"` : String(v);
const csv = grid.map((r,ri)=>r.map((v,ci)=>quote(ri>0&&ci===4?v.toFixed(1):v)).join(',')).join('\r\n')+'\r\n';
const path = `${dir}/lebron_james_lakers_2018_19_to_2025_26_regular_season.csv`;
await fs.writeFile(path,'\uFEFF'+csv,'utf8');
const result = {rows:records.length,seasonCounts:seasons.map(s=>({season:s.season,games:s.rows.length})),first:records[0].values,last:records.at(-1).values,headers,source:'https://www.nba.com/stats/player/2544/boxscores-traditional',path};
await fs.writeFile(`${dir}/validation.json`,JSON.stringify(result,null,2));
console.log(JSON.stringify(result));
