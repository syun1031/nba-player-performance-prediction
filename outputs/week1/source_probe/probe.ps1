$ErrorActionPreference = 'Stop'
$probeDir = Join-Path $PSScriptRoot '.'
$targets = @(
 @{name='nba_schedule'; url='https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json'},
 @{name='nba_roster'; url='https://stats.nba.com/stats/commonallplayers?LeagueID=00&Season=2026-27&IsOnlyCurrentSeason=1'},
 @{name='nba_history'; url='https://stats.nba.com/stats/leaguegamelog?Counter=0&DateFrom=04%2F12%2F2026&DateTo=04%2F12%2F2026&Direction=ASC&LeagueID=00&PlayerOrTeam=P&Season=2025-26&SeasonType=Regular%20Season&Sorter=DATE'}
)
$results = foreach ($target in $targets) {
 $started = [DateTimeOffset]::UtcNow
 $timer = [Diagnostics.Stopwatch]::StartNew()
 try {
  $response = Invoke-WebRequest -Uri $target.url -UseBasicParsing -TimeoutSec 15 -Headers @{Accept='application/json'; Referer='https://www.nba.com/'}
  $parsed = $response.Content | ConvertFrom-Json
  $summary = if ($target.name -eq 'nba_schedule') {
   $games = @($parsed.leagueSchedule.gameDates | ForEach-Object { $_.games })
   @{season=$parsed.leagueSchedule.seasonYear; game_count=$games.Count; sample=@($games | Select-Object -First 2); future_timestamp_count=@($games | Where-Object { $_.gameDateTimeUTC -and ([DateTimeOffset]$_.gameDateTimeUTC) -gt $started }).Count}
  } else {
   @{sets=@($parsed.resultSets | ForEach-Object { @{name=$_.name; headers=$_.headers; row_count=@($_.rowSet).Count; sample=@($_.rowSet | Select-Object -First 2)} })}
  }
  @{name=$target.name; url=$target.url; started_at_utc=$started.ToString('o'); elapsed_ms=$timer.ElapsedMilliseconds; http_status=[int]$response.StatusCode; bytes=$response.RawContentLength; outcome='json_received'; summary=$summary}
 } catch {
  $status = $null
  if ($_.Exception.Response) { try { $status=[int]$_.Exception.Response.StatusCode } catch {} }
  @{name=$target.name; url=$target.url; started_at_utc=$started.ToString('o'); elapsed_ms=$timer.ElapsedMilliseconds; http_status=$status; outcome='failed'; error=$_.Exception.Message}
 }
}
$results | ConvertTo-Json -Depth 12 | Set-Content (Join-Path $probeDir 'results.json') -Encoding utf8
$results | Select-Object name,outcome,http_status,elapsed_ms | Format-Table
