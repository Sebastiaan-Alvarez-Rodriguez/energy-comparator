import argparse
from pathlib import Path
import functools
import json
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


required_csv_names = ['name','type','gasconstant','gasconstant_unit','gasvariable','powerconstant','powerconstant_unit','powervariable_low','powervariable_high','powergen_low','powergen_high','bonus']

def set_if_not_null(d, key, value):
    if value:
        d[key] = value

def get_profile(path=None, profile_covariance_matrix=None, profile_gas=None, profile_power_low=None, profile_power_high=None, profile_power_gen_low=None, profile_power_gen_high=None):
    if path:
        with open(path, 'r') as f:
            profile = json.load(f)
    else:
        profile = {}

    set_if_not_null(profile, 'gas', profile_gas)
    set_if_not_null(profile, 'power_low', profile_power_low)
    set_if_not_null(profile, 'power_high', profile_power_high)
    set_if_not_null(profile, 'power_gen_low', profile_power_gen_low)
    set_if_not_null(profile, 'power_gen_high', profile_power_gen_high)
    for name in ['gas', 'power_low', 'power_high', 'power_gen_low', 'power_gen_high']:
        if f'{name}' not in profile:
            print(f'Key "{name}" not set, assuming {name} = 0.0')
            profile[name] = 0.0
    if profile_covariance_matrix:
        profile['covariance_matrix'] = np.array(json.loads(profile_covariance_matrix))
        if profile['covariance_matrix'].shape != (5,5,):
            print(f'Found a non-conforming matrix shape. Must be 5 rows, 5 columns. Found {profile["covariance_matrix"].shape[0]} rows, {profile["covariance_matrix"].shape[1]} columns.')
        print(f'The covariance matrix:')
        print(profile['covariance_matrix'])
    return profile


def normalize(df):
    # normalizes data units to year basis
    power_month_mask = df.powerconstant_unit == 'month'
    df['powerconstant'] = df.powerconstant * 12 * power_month_mask + df.powerconstant * (~power_month_mask)
    power_day_mask = df.powerconstant_unit == 'day'
    df['powerconstant'] = df.powerconstant * 12 * power_day_mask + df.powerconstant * (~power_day_mask)
    df['powerconstant_unit'] = 'year'
    gas_month_mask = df.gasconstant_unit == 'month'
    df['gasconstant'] = df.gasconstant * 12 * gas_month_mask + df.gasconstant * (~gas_month_mask)
    gas_day_mask = df.gasconstant_unit == 'day'
    df['gasconstant'] = df.gasconstant * 12 * gas_day_mask + df.gasconstant * (~gas_day_mask)
    df['gasconstant_unit'] = 'year'
    return df


def verify(df):
    # verifies integrity of input csv file.
    ok = True
    for name in required_csv_names:
        if name not in df:
            print(f'[Error] Missing column name "{name}" in csv header.')
            ok = False
    for idx, entry in df.iterrows():
        gas_vars = [x != 0 and not pd.isna(x) for x in [entry.iloc[2], entry.iloc[4]]]
        if any(gas_vars) and not all(gas_vars):
            print(f'[Error] Data entry at line {idx} ("{entry["name"]}") is incorrect: Contains some, but not all of "gasconstant", "gasconstant_unit", "gasvariable". Specified: {gas_vars}')
            ok = False
        power_vars = [x != 0 and not pd.isna(x) for x in [entry.iloc[5], entry.iloc[7], entry.iloc[8], entry.iloc[9], entry.iloc[10]]]
        if any(power_vars) and not all(power_vars):
            print(f'[Error] Data entry at line {idx} ("{entry["name"]}") is incorrect: Contains some, but not all of "powerconstant", "powerconstant_unit", "powervariable_low", "powervariable_high", "powergen_low", "powergen_high". Specified: {power_vars}')
            ok = False
    return ok


def best_offer(df, mean_vector, saldering, outage):
    '''
    Calculates the best contract for power and gas. Considers both combined and separate contracts.
    :param df Dataframe of contracts.
    :param mean_vector Vector containing estimated gas, power_low, power_high, power_generation_low, power_generation_high.
    :param saldering Percentage of power generated that can be subtracted from power usage, between 0 and 1.
    :param outage Expected Percentage of reduction in generated power due to the net turning off the supply.
    :return a (Pandas.Series, Pandas.Series) containing: the cheapest contract row for gas, cheapest contract for power.
    '''
    gas, power_low, power_high, power_gen_low, power_gen_high = mean_vector
    df['calc_gas'] = df.gasconstant + gas * df.gasvariable
    expected_low = power_low-(power_gen_low*saldering*outage)
    expected_high = power_high-(power_gen_high*saldering*outage)
    df['calc_power'] = df.powerconstant + \
        (expected_low * df.powervariable_low if expected_low > 0 else expected_low * df.powergen_low) + \
        (expected_high * df.powervariable_high if expected_high > 0 else expected_high * df.powergen_high) 

    df['total'] = df.calc_gas.fillna(0) + df.calc_power.fillna(0) + df.bonus.fillna(0)

    only_gas_mask = df.powerconstant.isna() & df.powervariable_low.isna() & df.powervariable_high.isna()
    only_power_mask = df.gasconstant.isna() & df.gasvariable.isna()
    # combined contracts comparison
    combined_df = df[(~only_gas_mask) & (~only_power_mask)]
    min_combined_idx = combined_df.loc[combined_df.total.idxmin()] if not combined_df.empty else None

    # separate contracts comparisons
    only_gas_df = df[only_gas_mask]
    min_gas_idx = only_gas_df.loc[only_gas_df.total.idxmin()] if not only_gas_df.empty else None

    only_power_df = df[only_power_mask]
    min_power_idx = only_power_df.loc[only_power_df.total.idxmin()] if not only_power_df.empty else None
    # print(f'individual gas: {min_gas_idx}')
    # print(f'individual power idx: {min_power_idx}')
    if min_combined_idx is None:
        if min_gas_idx is None or min_power_idx is None:
            return None, None
        return min_gas_idx, min_power_idx
    if min_gas_idx is None or min_power_idx is None or min_combined_idx.total <= min_power_idx.total + min_gas_idx.total:
        return pd.Series([min_combined_idx['name'], min_combined_idx.type, min_combined_idx.gasconstant, min_combined_idx.gasconstant_unit, min_combined_idx.gasvariable, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, min_combined_idx.bonus / 2, min_combined_idx.calc_gas, np.nan, min_combined_idx.total / 2], index=df.columns), \
        pd.Series([min_combined_idx['name'], min_combined_idx.type, np.nan, np.nan, np.nan, min_combined_idx.powerconstant, min_combined_idx.powerconstant_unit, min_combined_idx.powervariable_low, min_combined_idx.powervariable_high, min_combined_idx.powergen_low, min_combined_idx.powergen_high, min_combined_idx.bonus / 2, np.nan, min_combined_idx.calc_power, min_combined_idx.total / 2], index=df.columns)
    return min_gas_idx, min_power_idx


def find_minimal_function(df, mean_vector, cov_matrix, saldering, outage, samplesize=1000):
    # Find the linear function with the highest probability for each set of mean and covariance
    gas_dict = {}
    power_dict = {}
    linear_func = functools.partial(best_offer, df, saldering=saldering, outage=outage)
    inputs = np.random.multivariate_normal(mean_vector, cov_matrix, samplesize)
    outputs = [linear_func(sample) for sample in inputs]

    for gas, power in outputs:
        if gas is None or power is None:
            print('[Warning] found no winner. Filtered too much data?')
            continue
        gas_dict.setdefault(gas['name'], []).append(gas.total)
        power_dict.setdefault(power['name'], []).append(power.total)
    for k, v in gas_dict.items():
        gas_dict[k] = (len(v), np.mean(v))
    for k, v in power_dict.items():
        power_dict[k] = (len(v), np.mean(v))
    best_gas = max(gas_dict, key=lambda x: gas_dict[x][0]) # gets the key with most occurrences (=most optimal values)
    best_power = max(power_dict, key=lambda x: power_dict[x][0])
    print('On average, the best contract is:')
    print(f'Gas: "{best_gas}" with price {gas_dict[best_gas][1]} euro/year, {gas_dict[best_gas][1]/12} euro/month (best in {gas_dict[best_gas][0]/samplesize *100}% of situations measured)')
    print(f'Power: "{best_power}" with price {power_dict[best_power][1]} euro/year, {power_dict[best_power][1]/12} euro/month (best in {power_dict[best_power][0]/samplesize *100}% of situations measured)')
    print()
    print('Case summary:')
    print(f'''Gas:      {gas_dict[best_gas][1]}/yr ("{best_gas}")
Power:    {power_dict[best_power][1]}/yr ("{best_power}")
Combined: {gas_dict[best_gas][1] + power_dict[best_power][1]}/yr
''')


def main():
    # Simple program to calculate normalized cost for power & gas grid connections.
    parser = argparse.ArgumentParser()
    parser.add_argument('--offers', default=str(Path.cwd() / 'offers.csv'), help='Data file containing offers.')
    parser.add_argument('--profile', default=None, help='Data file containing usage and generation data.')
    parser.add_argument('--covariance-matrix', default=None, help='Set the 5x5 covariance matrix. Relations to denote: gas, power-low, power-high, power-generation-low, power-generation-high. Of course, the diagonal entries are standard deviations and the non-diagonal entries provide pairwise covariance. should look like json: i.e. `[ [60,0,0,0,0], [0,60,0,0,0], [0,0,60,0,0], [0,0,0,60,0], [0,0,0,0,60] ]`. Use only if you know what you are doing.')
    parser.add_argument('--gas', type=float, default=None, help='Average yearly gas consumption. Takes precedence over file-based data.')
    parser.add_argument('--power-low', type=float, default=None, help='Average yearly power consumption (low tarif). Takes precedence over file-based data.')
    parser.add_argument('--power-high', type=float, default=None, help='Average yearly power consumption (high tarif). Takes precedence over file-based data.')
    parser.add_argument('--power-gen-low', type=float, default=None, help='Average yearly power generation (low tarif). Takes precedence over file-based data.')
    parser.add_argument('--power-gen-high', type=float, default=None, help='Average yearly power generation (high tarif). Takes precedence over file-based data.')
    parser.add_argument('--saldering', type=float, default=1.0, help='Saldering assumption, determines how much generated energy compensates for consumed energy (factor between 0 and 1).')
    parser.add_argument('--outage', type=float, default=1.0, help='Outage assumption, describes when the sun shines but the net is satisfied, leading to no power delivery on the net (factor between 0 and 1).')
    parser.add_argument('--extra-costs', type=float, default=0.0, help='Extra: Other costs, can be negative to denote government aid.')
    parser.add_argument('--filter', type=str, default=None, help='Execute filter query on dataset. Example: --filter "`type==\'vast\' & name!=\'Oxxio 1 jaar\'"`. Exact syntax: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html')
    args = parser.parse_args()

    profile = get_profile(args.profile, args.covariance_matrix, args.gas, args.power_low, args.power_high, args.power_gen_low, args.power_gen_high)
    df = pd.read_csv(args.offers, comment='#')
    df = normalize(df)
    if not verify(df):
        return
    if args.filter:
        df.query(args.filter, inplace=True)
        print('Dataframe after filtering:')
    print(df.to_string())
    # mean_vector contains the usage/generation estimations.
    mean_vector = np.array([profile['gas'], profile['power_low'], profile['power_high'], profile['power_gen_low'], profile['power_gen_high']])

    if not 'covariance_matrix' in profile: # No statistics -> just simple computing
        gas, power = best_offer(df, mean_vector, args.saldering, args.outage)
        year_price = gas.total+power.total + args.extra_costs
        print('No covariance matrix given - computing in exact mode.')
        print(f'The best offer is for {year_price} euro per year, or {year_price/12} euro per month:')
        print('Gas:')
        print(gas)
        print('Power:')
        print(power)
    else:
        print('Covariance matrix given - computing in statistics mode.')
        std_vector = profile['covariance_matrix'].diagonal()
        if np.nan in std_vector or 0.0 in std_vector:
            print('found at least one standard deviation to be 0 or unset. Defaulting to 0.1, indicating almost no variance (nonzero required for SVD algorithm matrix convergence).')
            np.fill_diagonal(profile['covariance_matrix'], [v if v and v != 0.0 else 0.1 for v in std_vector])
        find_minimal_function(df, mean_vector, profile['covariance_matrix'], args.saldering, args.outage)

if __name__ == '__main__':
    main()
