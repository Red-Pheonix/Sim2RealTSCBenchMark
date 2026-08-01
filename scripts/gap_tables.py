"""Build every results table in tables/ from the raw run logs in logs/.

This is the single source of truth for how logged runs become paper numbers:
`make_figures.ipynb` calls :func:`build_all` so that a fresh clone with just the
`logs/` tree regenerates `tables/*.csv` and then the figures. The four
`analyze_*.ipynb` notebooks import the same builders for their deeper dives.

Selection rule (identical across the observation / transition / action gaps):
every trained cell is scored as the mean of its 5 best REAL_TEST rows by
reward (stable mergesort, episode tie-break); `direct_transfer` is zero-shot
and keeps its single row. The reward gap logs exactly one promoted REAL_TEST
row per run (keep-best), except `reward_oracle`, which is scored at its
curve-best R_real row. Gap = real-side metric minus the pretrained policy's
CityFlow eval (`logs/engine_baselines`).

Only needs numpy + pandas. Run as a script to rebuild everything:

    python scripts/gap_tables.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

TOP_REWARD_K = 5
REAL_EVAL_TARGET = 100  # intended real-eval budget per curve (real_interval 2 or 3)

AGENTS = ['dqn', 'presslight']
NETWORK_ORDER = ['tempe_1x1', 'bullhead_1', 'cologne1', 'ingolstadt1', 'hz1x1',
                 'tempe_16', 'bullhead_3', 'cologne3', 'ingolstadt7', 'hz4x4']

# shared 9-column DTL layout; the actions trainer appends two safety-shield rates
COLUMNS = ['exp_name', 'mode', 'episode', 'travel_time', 'aux', 'reward', 'queue',
           'delay', 'throughput']
ACTION_COLUMNS = [*COLUMNS, 'violation_rate', 'forceoff_rate']
METRIC_COLUMNS = ['travel_time', 'reward', 'queue', 'delay', 'throughput']
CF_GAP_METRICS = ['travel_time', 'queue', 'delay', 'throughput']  # reward excluded (engine units)

LOGS = Path('logs')
TABLES = Path('tables')


# --------------------------------------------------------------------- helpers

def read_dtl(log_path, columns=COLUMNS, numeric=None):
    df = pd.read_csv(log_path, sep='\t', header=None, names=columns)
    for column in numeric if numeric is not None else ['episode', 'aux', *METRIC_COLUMNS]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def dtl_files(logger_dir):
    # filenames are timestamped, so lexicographic order == chronological order
    return sorted(logger_dir.glob('*_DTL.log')) if logger_dir.exists() else []


def summarize_rows(rows, single_shot, log_name, metric_columns=METRIC_COLUMNS):
    if rows.empty:
        return None
    if single_shot:
        picked = rows.iloc[[0]]
    else:
        # stable mergesort + episode tie-break: saturated runs have many exactly-tied
        # rewards (e.g. 0.0), and an unstable sort makes the top-K picks irreproducible
        picked = (rows.sort_values(['reward', 'episode'], ascending=[False, True], kind='mergesort')
                  .head(TOP_REWARD_K))
    result = {
        'real_mode': picked['mode'].iloc[0],
        'selection_count': int(len(picked)),
        'selected_episodes': ','.join(str(int(e)) for e in picked['episode'] if pd.notna(e)),
        'log_file': log_name,
    }
    for column in metric_columns:
        result[column] = float(picked[column].mean())
        result[f'{column}_std'] = float(picked[column].std(ddof=1)) if len(picked) > 1 else np.nan
    return result


def read_cityflow_baseline(agent, network, logs=LOGS):
    # FINAL_TEST metrics of the pretrained tsc policy evaluated in cityflow (project-wide
    # gap reference). gap_<m> = real - cityflow for every metric: positive is worse for
    # travel_time/queue/delay, NEGATIVE is worse for throughput (fewer completions).
    files = dtl_files(logs / 'engine_baselines' / agent / network / 'cityflow_direct_transfer')
    if files:
        df = read_dtl(files[-1], numeric=['episode', *CF_GAP_METRICS])
        final = df[df['mode'] == 'FINAL_TEST']
        if not final.empty:
            return {m: float(final.iloc[-1][m]) for m in CF_GAP_METRICS}
    return {m: np.nan for m in CF_GAP_METRICS}


def fmt(mean, std):
    if pd.isna(mean):
        return ''
    if pd.isna(std):
        return f'{mean:.1f}'
    return f'{mean:.1f} ± {std:.1f}'


# ------------------------------------------------------------------ gap family

class GapFamily:
    """One ATT-gap task: where its DTLs live and how a cell's rows are picked.

    ``run_dir_style`` encodes the run-directory naming:
      per_setting  -- one run dir per (method, setting): <prefix>_<method>_<setting>_s0
      train_once   -- one run dir per method holding every setting's DTLs, mapped
                      via each file's exp_name (<prefix>_<method>_s0); single-shot
                      methods still use per-setting dirs.
    ``curve_fallback`` -- when to fall back from REAL_TEST to REAL_TRAIN rows:
      'empty' -- no REAL_TEST rows at all (GAT-family logs only rollouts)
      'le1'   -- <=1 REAL_TEST row (transitions DA logs one terminal REAL_TEST on
                 top of its REAL_TRAIN curve; score the curve so every
                 multi-episode method obeys the same top-K rule)
    """

    def __init__(self, name, subdir, prefix, methods, settings, out_csv,
                 run_dir_style, curve_fallback='empty', networks=None,
                 columns=COLUMNS, metric_columns=METRIC_COLUMNS,
                 export_metrics=None, single_shot=('direct_transfer',),
                 extra_record=None, brf_safety=False, logs=LOGS):
        self.name = name
        self.exp_root = logs / subdir
        self.logs = logs
        self.prefix = prefix  # '' for obs/trans (their dirs start at baseline_v1)
        self.methods = list(methods)
        self.settings = list(settings)
        self.networks = list(networks) if networks else list(NETWORK_ORDER)
        self.out_csv = out_csv
        self.run_dir_style = run_dir_style
        self.curve_fallback = curve_fallback
        self.columns = columns
        self.metric_columns = list(metric_columns)
        self.export_metrics = list(export_metrics) if export_metrics else self.metric_columns
        self.single_shot = set(single_shot)
        self.extra_record = extra_record or (lambda setting: {})
        self.brf_safety = brf_safety
        self._setting_files_cache = {}

    # -- locating a cell's DTL files ------------------------------------------
    def _run_dir(self, agent, network, token):
        return self.exp_root / f'cityflow_{agent}' / network / f'{self.prefix}{token}_s0' / 'logger'

    def _train_once_setting_files(self, agent, network, method):
        # one run dir per train-once method; map each per-setting DTL via its exp_name
        # (exp_name = f'{network}_{setting}_{agent}_{method_token}'; settings contain
        # no underscore for these families)
        key = (agent, network, method)
        if key not in self._setting_files_cache:
            by_setting = {}
            for log_path in dtl_files(self._run_dir(agent, network, method)):
                try:
                    first = log_path.open().readline().split('\t', 1)[0]
                except OSError:
                    continue
                if first:
                    setting = first[len(network) + 1:].split('_')[0]
                    by_setting.setdefault(setting, []).append(log_path)
            self._setting_files_cache[key] = by_setting
        return self._setting_files_cache[key]

    def candidate_dtls(self, agent, network, method, setting):
        if self.run_dir_style == 'train_once' and method not in self.single_shot:
            return self._train_once_setting_files(agent, network, method).get(setting, [])
        return dtl_files(self._run_dir(agent, network, f'{method}_{setting}'))

    # -- row selection ---------------------------------------------------------
    def read_dtl(self, log_path):
        return read_dtl(log_path, columns=self.columns,
                        numeric=['episode', 'aux', *self.metric_columns])

    def real_rows(self, df):
        # REAL_TEST is the canonical real-env eval; see curve_fallback in the class doc
        rows = df[df['mode'] == 'REAL_TEST']
        curve = df[df['mode'] == 'REAL_TRAIN']
        if self.curve_fallback == 'le1':
            if len(rows) <= 1 and not curve.empty:
                rows = curve
        elif rows.empty:
            rows = curve
        rows = rows[rows['travel_time'].notna() & rows['reward'].notna()].copy()
        # some runs logged real evals every episode instead of on the intended
        # real_interval (e.g. obs recon_baseline: 199 evals vs the latent siblings'
        # 99); subsample back to the intended cadence so every method draws from
        # the same-size pool
        if len(rows) > REAL_EVAL_TARGET * 1.1:
            interval = round(len(rows) / REAL_EVAL_TARGET)
            rows = rows[rows['episode'] % interval == 0]
        return rows

    def read_run_result(self, agent, network, method, setting):
        # newest DTL with usable real rows wins (older files can be truncated re-runs)
        for log_path in reversed(self.candidate_dtls(agent, network, method, setting)):
            result = summarize_rows(self.real_rows(self.read_dtl(log_path)),
                                    method in self.single_shot, log_path.name,
                                    metric_columns=self.metric_columns)
            if result is None:
                continue
            if self.brf_safety:
                self._add_brf_safety(result, log_path)
            return result
        return None

    def _add_brf_safety(self, result, dtl_path):
        # absolute per-episode counts live only in the BRF twin of each DTL (the DTL
        # itself carries per-decision rates)
        brf_path = Path(str(dtl_path).replace('_DTL.log', '_BRF.log'))
        # main BRF eval lines only: "MODE step:N, travel time:..., violations:V, force_offs:F"
        # (the ", " after the step number excludes the *_PER_INT lines)
        pattern = re.compile(r'(\w+) step:(\d+), .*violations:(\d+), force_offs:(\d+)')
        totals = {}
        if brf_path.exists():
            for line in brf_path.open():
                m = pattern.search(line)
                if m:
                    totals[(m.group(1), int(m.group(2)))] = (int(m.group(3)), int(m.group(4)))
        eps = [int(e) for e in result['selected_episodes'].split(',') if e]
        found = [totals[(result['real_mode'], e)] for e in eps if (result['real_mode'], e) in totals]
        for name, i in [('violations', 0), ('force_offs', 1)]:
            vals = np.array([v[i] for v in found], dtype=float)
            result[name] = float(vals.mean()) if len(vals) else np.nan
            result[f'{name}_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan

    # -- the full sweep --------------------------------------------------------
    def collect(self, verbose=True):
        records, missing = [], []
        for agent in AGENTS:
            for network in self.networks:
                cf = read_cityflow_baseline(agent, network, self.logs)
                for setting in self.settings:
                    for method in self.methods:
                        result = self.read_run_result(agent, network, method, setting)
                        if result is None:
                            missing.append(f'{agent}/{network}/{method}/{setting}')
                            continue
                        records.append({
                            'agent': agent, 'network': network, 'setting': setting,
                            'method': method,
                            'cityflow_att': cf['travel_time'],
                            'gap_vs_cityflow': result['travel_time'] - cf['travel_time'],
                            **{f'cityflow_{m}': cf[m] for m in CF_GAP_METRICS[1:]},
                            **{f'gap_{m}': result[m] - cf[m] for m in CF_GAP_METRICS[1:]},
                            **self.extra_record(setting),
                            **result,
                        })
        df = pd.DataFrame(records)
        df['network'] = pd.Categorical(df['network'], self.networks, ordered=True)
        df['setting'] = pd.Categorical(df['setting'], self.settings, ordered=True)
        df['method'] = pd.Categorical(df['method'], self.methods, ordered=True)
        if verbose:
            print(f'{self.name}: {len(df)} runs loaded, {len(missing)} missing')
            for name in missing:
                print('  missing:', name)
        return df

    def export(self, results_df):
        export_metrics = self.export_metrics
        extra_cols = list(self.extra_record(self.settings[0]))
        picked_rows = []
        for _, row in results_df.sort_values(['agent', 'network', 'setting', 'method']).iterrows():
            rec = {'agent': row['agent'], 'network': row['network'],
                   'setting': row['setting'], 'method': row['method'],
                   **{c: row[c] for c in extra_cols},
                   'real_mode': row['real_mode'], 'selection_count': row['selection_count']}
            for column in export_metrics:
                rec[column] = fmt(row[column], row[f'{column}_std'])
            rec['cityflow_att'] = f"{row['cityflow_att']:.1f}"
            rec['gap_vs_cityflow'] = f"{row['gap_vs_cityflow']:.1f}"
            for m, p in [('queue', 1), ('delay', 2), ('throughput', 0)]:
                rec[f'cityflow_{m}'] = f"{row[f'cityflow_{m}']:.{p}f}"
                rec[f'gap_{m}'] = f"{row[f'gap_{m}']:.{p}f}"
            rec['selected_episodes'] = row['selected_episodes']
            rec['log_file'] = row['log_file']
            picked_rows.append(rec)
        picked_df = pd.DataFrame(picked_rows)
        TABLES.mkdir(exist_ok=True)
        picked_df.to_csv(TABLES / self.out_csv, index=False)
        print(f'wrote {len(picked_df)} rows to tables/{self.out_csv}')
        return picked_df

    def build(self, verbose=True):
        """Parse logs, write tables/<out_csv>, return the raw (unformatted) frame."""
        results_df = self.collect(verbose=verbose)
        self.export(results_df)
        return results_df


ACTION_SETTING_DELAY = {'setting1': 20, 'setting2': 30, 'setting3': 40, 'setting4': 60}

OBSERVATIONS = GapFamily(
    'observations', 'sim2real_observations', 'baseline_v1_',
    methods=['direct_transfer', 'domain_randomization', 'vae', 'recon_baseline',
             'darla', 'atc', 'lusr'],
    settings=['noise3', 'noise5', 'noise10', 'noise20',
              'dz10', 'dz20', 'dz50', 'dz100',
              'sensor5', 'sensor20', 'sensor50', 'sensor70',
              'combine1', 'combine2', 'combine3', 'combine4'],
    out_csv='observation_table.csv', run_dir_style='train_once')

TRANSITIONS = GapFamily(
    'transitions', 'sim2real_transitions', 'baseline_v1_',
    methods=['direct_transfer', 'domain_randomization', 'domain_adaptation',
             'gat', 'ugat', 'jlgat'],
    settings=['setting1', 'setting2', 'setting3', 'setting4'],
    out_csv='transition_table.csv', run_dir_style='per_setting', curve_fallback='le1')

ACTION_DELAY = GapFamily(
    'action delay', 'sim2real_actions', 'action_delay_optimized_s0_',
    methods=['direct_transfer', 'naive', 'oblivious_q', 'delayed_q', 'prlight'],
    settings=['setting1', 'setting2', 'setting3', 'setting4'],
    out_csv='action_table.csv', run_dir_style='train_once', columns=ACTION_COLUMNS,
    extra_record=lambda setting: {'action_delay': ACTION_SETTING_DELAY[setting]})

ACTION_PT = GapFamily(
    'action phase-transition', 'sim2real_actions', 'action_pt_s0_',
    # unshielded before shielded within each adaptation pair
    methods=['direct_transfer', 'dr_noshield', 'dr', 'gat', 'gat_shield',
             'ugat', 'ugat_shield'],
    settings=['cyclic', 'flexible', 'barrier_leading_fixed', 'barrier_lagging_fixed',
              'barrier_leading_lagging_fixed'],
    out_csv='action_pt_table.csv', run_dir_style='per_setting', columns=ACTION_COLUMNS,
    networks=['tempe_1x1', 'bullhead_1', 'tempe_16', 'bullhead_3'],
    metric_columns=[*METRIC_COLUMNS, 'violation_rate', 'forceoff_rate'],
    # per-decision rates feed selection/means but the exported CSV carries the
    # absolute BRF counts instead (violations / force_offs)
    export_metrics=[*METRIC_COLUMNS, 'violations', 'force_offs'],
    brf_safety=True)


# --------------------------------------------------------------- rewards tables

REWARD_SETTINGS = ['efficiency_aligned', 'emission_heavy', 'fairness_heavy',
                   'physical_safety_heavy']
# floor -> budgeted methods -> skyline; reward_oracle is the budget-exempt known-w
# reference (oracle_v1 runs), not a contender
REWARD_METHODS = ['direct_transfer', 'reward_inference', 'morl_grid',
                  'dynamic_reward_shaping', 'reward_oracle']
# the raw metric each hidden reward actually weights (lower is better for all)
NATIVE_METRIC = {
    'efficiency_aligned': 'travel_time',
    'emission_heavy': 'emission',
    'fairness_heavy': 'fairness',
    'physical_safety_heavy': 'ssm_conflicts',
}
REWARD_DTL_NUMERIC = ['step', 'travel_time', 'loss', 'rewards', 'queue', 'delay',
                      'throughput', 'train_reward', 'R_real', 'fairness', 'emission',
                      'fuel', 'emergency_stops', 'ssm_conflicts', 'collisions']

REWARD_METHOD_DISPLAY = {
    'direct_transfer': 'Direct-Transfer (sim reward)',
    'reward_inference': 'Reward Inference',
    'morl_grid': 'MORL (grid)',
    'dynamic_reward_shaping': 'Dynamic Reward Shaping',
    'reward_oracle': 'Reward Oracle (known w*, skyline)',
}
# every real-side metric, appendix-complete; (column, display name, unit, round digits)
REWARD_ALL_METRICS = [
    ('R_real', 'R_real (hidden true reward)', 'higher = better', 4),
    ('travel_time', 'Travel time', 's', 1),
    ('queue', 'Queue', 'veh (lane mean)', 1),
    ('delay', 'Delay', 'ratio', 2),
    ('throughput', 'Throughput', 'veh (episode total)', 0),
    ('fairness', 'Fairness', 'max-min served veh', 0),
    ('emission', 'CO2 emission', 'kg (episode total)', 1),
    ('fuel', 'Fuel', 'kg (episode total)', 1),
    ('emergency_stops', 'Emergency stops', 'count (episode total)', 0),
    ('ssm_conflicts', 'Conflicts (TTC < 1.5 s)', 'count (episode total)', 0),
]
REWARD_SINGLE_NETWORKS = ['tempe_1x1', 'bullhead_1', 'cologne1', 'ingolstadt1', 'hz1x1']
REWARD_MULTI_NETWORKS = ['tempe_16', 'bullhead_3', 'cologne3', 'ingolstadt7', 'hz4x4']


def read_rewards_dtl(log_path):
    # reward-gap DTLs carry their 20-column header as the first row
    df = pd.read_csv(log_path, sep='\t', header=0)
    for column in REWARD_DTL_NUMERIC:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def read_real_test_row(method_dir, curve_best=False):
    # default: latest DTL's last REAL_TEST row (each v2 run logs exactly one -- the
    # promoted checkpoint's deterministic eval).
    # curve_best (reward_oracle only): the oracle has no keep-best, so take the
    # max-R_real row of its ~100-point REAL_TEST scoring curve instead of the
    # final-weights eval; the final eval is kept for disclosure.
    log_files = sorted(method_dir.glob('*_DTL.log'))
    if not log_files:
        return None
    df = read_rewards_dtl(log_files[-1])
    real_test = df[df['mode'] == 'REAL_TEST']
    if real_test.empty:
        return None
    if curve_best:
        row = real_test.loc[real_test['R_real'].idxmax()].to_dict()
        row['final_R_real'] = float(real_test.iloc[-1]['R_real'])
    else:
        row = real_test.iloc[-1].to_dict()
    row['log_file'] = log_files[-1].name
    return row


def read_rewards_cityflow_baseline(agent, network, logs=LOGS):
    # FINAL_TEST travel time of the pretrained tsc policy evaluated in cityflow
    log_files = sorted((logs / 'sim2real_rewards' / agent / network
                        / 'cityflow_direct_transfer').glob('*_DTL.log'))
    if not log_files:
        return np.nan
    df = pd.read_csv(log_files[-1], sep='\t', header=None, names=COLUMNS)
    final = df[df['mode'] == 'FINAL_TEST']
    return float(final.iloc[-1]['travel_time']) if not final.empty else np.nan


def collect_rewards(logs=LOGS, verbose=True):
    log_root = logs / 'sim2real_rewards'
    rows = []
    for agent in AGENTS:
        for network in NETWORK_ORDER:
            cityflow_att = read_rewards_cityflow_baseline(agent, network, logs)
            for setting in REWARD_SETTINGS:
                for method in REWARD_METHODS:
                    real = read_real_test_row(log_root / agent / network / method / setting,
                                              curve_best=(method == 'reward_oracle'))
                    if real is None:
                        if verbose:
                            print(f'missing REAL_TEST: {agent}/{network}/{method}/{setting}')
                        continue
                    rows.append({
                        'agent': agent, 'network': network, 'setting': setting,
                        'method': method,
                        'R_real': real['R_real'],
                        'oracle_final_R_real': real.get('final_R_real', np.nan),
                        'travel_time': real['travel_time'],
                        'queue': real['queue'],
                        'delay': real['delay'],
                        'throughput': real['throughput'],
                        'native_metric': NATIVE_METRIC[setting],
                        'native_value': real[NATIVE_METRIC[setting]],
                        'fairness': real['fairness'],
                        'emission': real['emission'],
                        'fuel': real['fuel'],
                        'emergency_stops': real['emergency_stops'],
                        'ssm_conflicts': real['ssm_conflicts'],
                        'cityflow_att': cityflow_att,
                        'att_gap_vs_cityflow': real['travel_time'] - cityflow_att,
                        'log_file': real['log_file'],
                    })
    df = pd.DataFrame(rows)
    if verbose:
        print(f'rewards: {len(df)} rows loaded')
    return df


def build_reward_tables(logs=LOGS, verbose=True):
    """Parse the reward-gap logs; write reward_table.csv + tables/reward_gap/*.csv."""
    results_df = collect_rewards(logs, verbose=verbose)

    export_df = results_df[[
        'agent', 'network', 'setting', 'method',
        'R_real', 'oracle_final_R_real',
        'native_metric', 'native_value',
        'travel_time', 'cityflow_att', 'att_gap_vs_cityflow',
        'queue', 'delay', 'throughput',
        'fairness', 'emission', 'fuel', 'emergency_stops', 'ssm_conflicts', 'log_file',
    ]].copy()
    export_df['R_real'] = export_df['R_real'].round(4)
    export_df['oracle_final_R_real'] = export_df['oracle_final_R_real'].round(4)
    for column in ['native_value', 'travel_time', 'cityflow_att', 'att_gap_vs_cityflow',
                   'queue', 'delay', 'throughput', 'fairness', 'emission', 'fuel',
                   'emergency_stops', 'ssm_conflicts']:
        export_df[column] = export_df[column].round(1)
    TABLES.mkdir(exist_ok=True)
    export_df.to_csv(TABLES / 'reward_table.csv', index=False)
    print(f'wrote {len(export_df)} rows to tables/reward_table.csv')

    # appendix layout tables: one CSV per (agent, setting, single|multi network block)
    table_dir = TABLES / 'reward_gap'
    table_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for agent in AGENTS:
        for setting in REWARD_SETTINGS:
            for networks, scale in ((REWARD_SINGLE_NETWORKS, 'single'),
                                    (REWARD_MULTI_NETWORKS, 'multi')):
                sub = results_df[(results_df['agent'] == agent)
                                 & (results_df['setting'] == setting)]
                nets = [n for n in networks if n in set(sub['network'])]
                out_rows = []
                for column, display, unit, digits in REWARD_ALL_METRICS:
                    pivot = sub.pivot(index='method', columns='network', values=column)
                    for method in REWARD_METHODS:
                        rec = {'metric': display, 'unit': unit,
                               'method': REWARD_METHOD_DISPLAY[method]}
                        for n in nets:
                            rec[n] = round(float(pivot.loc[method, n]), digits)
                        out_rows.append(rec)
                pd.DataFrame(out_rows).to_csv(table_dir / f'{agent}_{setting}_{scale}.csv',
                                              index=False)
                written += 1
    print(f'{written} tables -> {table_dir}/')
    return results_df


# ------------------------------------------------------- engine baseline table

def build_engine_baseline_table(logs=LOGS):
    """cityflow vs sumo ATT of the pretrained tsc policies (the pure engine gap)."""
    rows = []
    for agent in AGENTS:
        for network in NETWORK_ORDER:
            atts = {}
            for world in ('cityflow', 'sumo'):
                att = np.nan
                files = dtl_files(logs / 'engine_baselines' / agent / network
                                  / f'{world}_direct_transfer')
                if files:
                    df = read_dtl(files[-1], numeric=['episode', 'travel_time'])
                    final = df[df['mode'] == 'FINAL_TEST']
                    if not final.empty:
                        att = float(final.iloc[-1]['travel_time'])
                atts[world] = att
            rows.append({'agent': agent, 'network': network,
                         'cityflow_att': round(atts['cityflow'], 1),
                         'sumo_att': round(atts['sumo'], 1),
                         'engine_gap': round(atts['sumo'] - atts['cityflow'], 1)})
    df = pd.DataFrame(rows)
    TABLES.mkdir(exist_ok=True)
    df.to_csv(TABLES / 'engine_baseline_table.csv', index=False)
    print(f'wrote {len(df)} rows to tables/engine_baseline_table.csv')
    return df


# ------------------------------------------------------------------- build all

def build_all(verbose=False):
    """Rebuild every tables/*.csv from logs/. Returns the raw frames by name."""
    return {
        'engine_baseline': build_engine_baseline_table(),
        'observations': OBSERVATIONS.build(verbose=verbose),
        'transitions': TRANSITIONS.build(verbose=verbose),
        'action_delay': ACTION_DELAY.build(verbose=verbose),
        'action_pt': ACTION_PT.build(verbose=verbose),
        'rewards': build_reward_tables(verbose=verbose),
    }


if __name__ == '__main__':
    build_all(verbose=True)
