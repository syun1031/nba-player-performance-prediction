"""Merge saved player logs and actual game rosters without changing sources."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    'Bobby Portis': 'Bobby Portis Jr.',
    'Jonas Valančiūnas': 'Jonas Valanciunas',
    'Terrence Shannon Jr.': 'Terrence Shannon Jr',
    'Hansen Yang': 'Yang Hansen',
    'Monté Morris': 'Monte Morris',
    'Dario Šarić': 'Dario Saric',
    'Darius Brown': 'Darius Brown II',
}
LOG_FIELDS = ['PLAYER', 'TEAM', 'MATCHUP', 'GAMEDATE', 'MIN', 'PTS', 'FG%', 'REB', 'AST']
EXTRA_FIELDS = ['player_id', 'team_id', 'game_id', 'is_starter',
                'starting_teammates', 'appeared_players', 'ROSTER_MATCHUP',
                'roster_match_status', 'home_away_conflict', 'is_home',
                'merge_eligible', 'exclusion_reason', 'pregame_verified',
                'is_neutral_site', 'designated_home_team', 'venue',
                'venue_source_url', 'home_away_resolution']


def matchup_info(matchup, team):
    if ' @ ' in matchup:
        away, home = matchup.split(' @ ')
    elif ' VS. ' in matchup:
        home, away = matchup.split(' VS. ')
    else:
        raise ValueError(f'Unknown MATCHUP: {matchup!r}')
    if team not in (away, home) or away == home:
        raise ValueError(f'TEAM not in MATCHUP: {team}, {matchup}')
    return frozenset((away, home)), int(team == home)


def roster_index(rosters):
    index = {}
    for row in rosters:
        date.fromisoformat(row['GAMEDATE'])
        key = row['GAMEDATE'], row['TEAM']
        if key in index:
            raise ValueError(f'Duplicate team/date roster: {key}')
        players, starters = [], []
        for i in range(1, 16):
            raw = row.get(f'PLAYER_{i}', '').strip()
            if not raw:
                continue
            name = raw.removeprefix('*').strip()
            if not name or '*' in name:
                raise ValueError(f'Invalid starred name: {raw!r}')
            name = ALIASES.get(name, name)
            players.append(name)
            if raw.startswith('*'):
                starters.append(name)
        if len(starters) != 5 or len(set(players)) != len(players):
            raise ValueError(f'Invalid roster: {key}')
        pair, home = matchup_info(row['MATCHUP'], row['TEAM'])
        index[key] = dict(players=players, starters=starters,
                          pair=pair, home=home, raw_matchup=row['MATCHUP'])
    return index


def merge_records(logs, rosters, venues=None):
    if venues is None:
        full_registry = ROOT/'outputs/week1/venue_audit/verified_venues.json'
        registry = full_registry if full_registry.exists() else ROOT/'docs/week1/verified_neutral_venues.json'
        venues = json.loads(registry.read_text(encoding='utf-8'))
    venue_index = {}
    for venue in venues:
        date.fromisoformat(venue['GAMEDATE'])
        teams = frozenset(venue['teams'])
        venue_key = venue['GAMEDATE'], teams
        if len(teams) != 2 or venue_key in venue_index:
            raise ValueError('Invalid or duplicate verified venue')
        if venue['designated_home_team'] is not None and venue['designated_home_team'] not in teams:
            raise ValueError('Designated home team not in verified venue teams')
        neutral = venue.get('is_neutral_site', True)
        if not isinstance(neutral, bool) or (not neutral and venue['designated_home_team'] is None):
            raise ValueError('Non-neutral venue must have official home team')
        venue_index[venue_key] = venue
    index = roster_index(rosters)
    seen, seen_player_dates, result = set(), set(), []
    for raw in logs:
        row = {name: raw[name] for name in LOG_FIELDS}
        date.fromisoformat(row['GAMEDATE'])
        minutes = float(row['MIN'])
        if not 0 <= minutes < float('inf'):
            raise ValueError('Invalid minutes')
        game_id = row['MATCHUP'] + '|' + row['GAMEDATE']
        key = row['PLAYER'], game_id
        player_date = row['PLAYER'], row['GAMEDATE']
        if key in seen or player_date in seen_player_dates:
            raise ValueError(f'Duplicate player/game or player/date: {key}')
        seen.add(key)
        seen_player_dates.add(player_date)
        pair, home = matchup_info(row['MATCHUP'], row['TEAM'])
        roster = index.get((row['GAMEDATE'], row['TEAM']))
        if roster is None or roster['pair'] != pair:
            raise ValueError(f'Missing roster or opponent mismatch: {key}')
        found = row['PLAYER'] in roster['players']
        if not found and minutes > 0:
            raise ValueError(f'Positive minutes with no roster player: {key}')
        conflict = roster['home'] != home
        venue = venue_index.get((row['GAMEDATE'], pair))
        neutral = venue.get('is_neutral_site', True) if venue else None
        resolved_home = (0 if neutral else int(row['TEAM'] == venue['designated_home_team'])) if venue else (None if conflict else home)
        reasons = []
        if minutes == 0:
            reasons.append('zero_minutes')
        if not found:
            reasons.append('roster_player_missing')
        if conflict and venue is None:
            reasons.append('home_away_conflict')
        row.update(player_id=row['PLAYER'], team_id=row['TEAM'], game_id=game_id,
                   is_starter=int(row['PLAYER'] in roster['starters']) if found else None,
                   starting_teammates=[p for p in roster['starters'] if p != row['PLAYER']],
                   appeared_players=roster['players'], ROSTER_MATCHUP=roster['raw_matchup'],
                   roster_match_status='matched' if found else 'missing_player',
                   home_away_conflict=int(conflict),
                   is_home=resolved_home,
                   merge_eligible=not reasons, exclusion_reason=';'.join(reasons),
                   pregame_verified=False,
                   is_neutral_site=int(neutral) if venue is not None else None,
                   designated_home_team=venue['designated_home_team'] if venue else None,
                   venue=venue['venue'] if venue else None,
                   venue_source_url=venue['source_url'] if venue else None,
                   home_away_resolution=('verified_neutral_site' if neutral else 'verified_official_venue') if venue else
                       ('unresolved' if conflict else 'source_agreement'))
        result.append(row)
    return result


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\r\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False)
                             if isinstance(row[k], list) else row[k] for k in fields})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/week1/model_data')
    args = parser.parse_args()
    base = ROOT/'outputs/nba_2025_26'
    log_path = base/'nba_2025_26_regular_season_player_game_logs.csv'
    roster_path = base/'nba_2025_26_regular_season_game_rosters.csv'
    original_hashes = {str(p.relative_to(ROOT)): digest(p) for p in (log_path, roster_path)}
    logs, rosters = read_csv(log_path), read_csv(roster_path)
    index = roster_index(rosters)
    games = defaultdict(set)
    for r in rosters:
        pair, _ = matchup_info(r['MATCHUP'], r['TEAM'])
        games[(r['GAMEDATE'], tuple(sorted(pair)))].add(r['TEAM'])
    if any(len(teams) != 2 for teams in games.values()):
        raise ValueError('Every game must have two team rosters')

    rows = merge_records(logs, rosters)
    log_keys = {(r['GAMEDATE'], r['TEAM'], r['PLAYER']) for r in logs}
    roster_keys = {(day, team, name) for (day, team), r in index.items() for name in r['players']}
    if roster_keys - log_keys:
        raise ValueError('Unmatched roster players after explicit alias mapping')
    kept = [r for r in rows if float(r['MIN']) > 0]
    excluded = [r for r in rows if float(r['MIN']) == 0]
    conflicts = sorted({(r['GAMEDATE'], r['TEAM'], r['MATCHUP'], r['ROSTER_MATCHUP'])
                        for r in rows if r['home_away_conflict']})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    merged_path = out/'nba_2025_26_player_games_merged.csv'
    excluded_path = out/'excluded_zero_minutes.csv'
    fields = LOG_FIELDS + EXTRA_FIELDS
    write_csv(merged_path, kept, fields)
    write_csv(excluded_path, excluded, fields)
    write_csv(out/'name_aliases.csv', [dict(roster_name=k, PLAYER=v) for k, v in ALIASES.items()],
              ['roster_name', 'PLAYER'])
    write_csv(out/'matchup_conflicts.csv', [dict(zip(
        ['GAMEDATE', 'TEAM', 'MATCHUP', 'ROSTER_MATCHUP'], values)) for values in conflicts],
        ['GAMEDATE', 'TEAM', 'MATCHUP', 'ROSTER_MATCHUP'])

    # Independent readback: original values, numeric labels and full roster sets.
    saved = read_csv(merged_path)
    if len(saved) != len(kept) or len(read_csv(excluded_path)) != len(excluded):
        raise AssertionError('Output row count mismatch')
    for original, disk in zip([r for r in logs if float(r['MIN']) > 0], saved):
        if any(original[k] != disk[k] for k in LOG_FIELDS):
            raise AssertionError('Original field changed')
        source = next(r for r in rosters if r['TEAM'] == disk['TEAM'] and r['GAMEDATE'] == disk['GAMEDATE'])
        raw_names = [source.get(f'PLAYER_{i}', '').strip() for i in range(1, 16)]
        expected = {ALIASES.get(n.removeprefix('*').strip(), n.removeprefix('*').strip())
                    for n in raw_names if n}
        starters = {ALIASES.get(n[1:].strip(), n[1:].strip()) for n in raw_names if n.startswith('*')}
        if set(json.loads(disk['appeared_players'])) != expected:
            raise AssertionError('Roster lost players')
        if int(disk['is_starter']) != int(disk['PLAYER'] in starters):
            raise AssertionError('Starter label mismatch')
        if set(json.loads(disk['starting_teammates'])) != starters - {disk['PLAYER']}:
            raise AssertionError('Starting teammates mismatch')
        if disk['is_neutral_site'] == '1':
            if disk['is_home'] != '0' or not disk['venue_source_url']:
                raise AssertionError('Verified neutral venue encoding failed')
        elif disk['is_neutral_site'] == '0':
            if disk['is_home'] != str(int(disk['TEAM'] == disk['designated_home_team'])):
                raise AssertionError('Official home team encoding failed')
        if disk['game_id'] != original['MATCHUP'] + '|' + original['GAMEDATE']:
            raise AssertionError('Original game identity changed')
    if len(rows) != len(kept) + len(excluded):
        raise AssertionError('Row accounting failed')
    for path in (log_path, roster_path):
        if digest(path) != original_hashes[str(path.relative_to(ROOT))]:
            raise AssertionError('Source file changed')
    report = dict(
        verified=True, source_sha256=original_hashes,
        input_log_rows=len(logs), input_team_roster_rows=len(rosters), games=len(games),
        roster_appearances=len(roster_keys), merged_positive_minute_rows=len(kept),
        excluded_zero_minute_rows=len(excluded), positive_minute_unmatched=0,
        zero_minute_unmatched=sum(r['roster_match_status'] != 'matched' for r in excluded),
        roster_unmatched_after_aliases=0, alias_pairs=len(ALIASES),
        matchup_conflict_team_games=len(conflicts),
        positive_minute_rows_with_unknown_home=sum(r['is_home'] is None for r in kept),
        resolved_conflict_rows=sum(r['home_away_conflict'] and r['home_away_resolution'] == 'verified_neutral_site' for r in kept),
        verified_neutral_games=len({(r['GAMEDATE'], tuple(sorted(matchup_info(r['MATCHUP'], r['TEAM'])[0])))
                                   for r in kept if r['is_neutral_site'] == 1}),
        verified_neutral_positive_minute_rows=sum(r['is_neutral_site'] == 1 for r in kept),
        neutral_status_unverified_rows=sum(r['is_neutral_site'] is None for r in kept),
        official_venue_verified_rows=sum(r['venue_source_url'] is not None for r in kept),
        merge_eligible_rows=sum(r['merge_eligible'] for r in kept),
        missing_history_eligibility_not_yet_evaluated=True,
        starter_distribution=dict(Counter(r['is_starter'] for r in kept)),
        exclusion_reason_counts=dict(Counter(r['exclusion_reason'] for r in rows if r['exclusion_reason'])),
        source_files_unchanged=True, output_readback_verified=True,
        pregame_information_verified=False,
        limitations=['Actual postgame roster, no pregame publication timestamps',
                     'Alias mappings are explicit snapshot-local contextual matches',
                     'Unverified neutral status is blank, not assumed false',
                     'Raw direction differences preserved; verified neutral venues have is_home=0',
                     'Historical windows and regular-starter absence not implemented'])
    (out/'merge_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
