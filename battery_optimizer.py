import argparse
from pathlib import Path
import functools
import json
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


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


def verify(df, *names):
    # verifies integrity of contract prices csv file.
    required_csv_names = ['entity_id', 'state', 'last_changed']
    ok = True
    for name in required_csv_names:
        if name not in df:
            print(f'[Error] Missing column name "{name}" in csv header.')
            ok = False
    # for idx, entry in df.iterrows():
    #     if not found_production:
    #         found_production = entry.iloc[0] == p_name
    #     if not found_consumption:
    #         found_consumption = entry.iloc[0] == c_name

    for n in names:
        if not any(df['entity_id'] == n):
            print(f'[Error] Could not find values for "{n}" in csv file.')
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

def simulate_battery(df, p_col, np_col, nc_col, capacity_kwh=10, inverter_limit_kw=5.0, min_pct=0.20, max_pct=0.80, discard_overflow=False):
    '''
    Simulates a home battery system and calculates adjusted grid metrics.
    
    Parameters:
    - df: Input pandas DataFrame
    - p_col, np_col, nc_col: String names of your columns
    - capacity_kwh: Total battery capacity in kWh
    - inverter_limit_kw: The battery inverter limit on how many kW of power it can safely convert from or to the battery at any time.
    - min_pct: Minimum charge threshold (e.g., 0.20 for 20%)
    - max_pct: Maximum charge threshold (e.g., 0.80 for 80%)
    - discard_overflow: If True, excess solar is wasted when battery is full. If False, excess solar is delivered to the grid.
    '''
    # Calculate physical limits in kWh
    min_charge = capacity_kwh * min_pct
    max_charge = capacity_kwh * max_pct
    
    # Initialize the battery at its minimum threshold
    current_charge = min_charge
    
    # Lists to store the simulated timeline
    sim_battery_soc = []
    sim_nc = []
    sim_np = []
    
    # Sequential simulation loop
    for _, row in df.iterrows():
        p = row[p_col]
        np_val = row[np_col]
        nc = row[nc_col]
        
        # 1. HANDLE SURPLUS GENERATION (Charging)
        if np_val > 0:
            if max_charge - current_charge > 0:
                # we can only charge as much as the battery inverter can safely convert AC->DC
                possible_charge = min(np_val, inverter_limit_kw)
                # Absorb what we can into the battery
                charge_amount = min(possible_charge, (max_charge - current_charge))
                current_charge += charge_amount
                remaining_surplus = np_val - charge_amount
                # Determine what happens to the leftover energy
                new_np = 0.0 if discard_overflow else remaining_surplus
            else:
                new_np = 0.0 if discard_overflow else np_val # no room for power
            
        else:
            new_np = np_val
            
        # 2. HANDLE DEFICIT/CONSUMPTION (Discharging)
        if nc > 0 and (current_charge - min_charge) > 0:
            # we can only discharge as much as the battery inverter can safely convert DC->AC
            possible_charge = min(nc, inverter_limit_kw)
            # Drain what we can from the battery
            discharge_amount = min(possible_charge, (current_charge - min_charge))
            current_charge -= discharge_amount
            new_nc = nc - discharge_amount
        else:
            new_nc = nc
            
        # Save states for this timestamp
        sim_battery_soc.append(current_charge)
        sim_nc.append(new_nc)
        sim_np.append(new_np)
        
    # Build the resulting DataFrame
    sim_df = df.copy()
    sim_df['battery_soc_kwh'] = sim_battery_soc
    sim_df[nc_col] = sim_nc
    sim_df[np_col] = sim_np
    
    return sim_df

def printst(df, p, np, nc):
    print(f'required overal energy: {df[nc].sum()}kWh')
    print(f'net produced energy:    {df[np].sum()}kWh')

def printcmp(df1, p1, np1, nc1, df2, p2, np2, nc2):
    reduction_kwh = df1[nc1].sum() - df2[nc2].sum()
    print(f'Overal energy consumption reduction: {reduction_kwh}kWh ({reduction_kwh / df1[nc1].sum() *100}%)')
    overflow_kwh = df1[np1].sum() - df2[np2].sum()
    print(f'Overal energy overflowing reduction: {overflow_kwh}kWh ({overflow_kwh / df1[np1].sum() *100}%)')


def main():
    # Simple program to calculate optimal home battery size for a given power consumption
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', default=str(Path.cwd() / 'history.csv'), help='Data file containing historical gross production and grid consumption values.')
    parser.add_argument('--p', default='sensor.solaredge_solar_power', help='Index sensor name for production values')
    parser.add_argument('--np', default='sensor.electricity_meter_power_production', help='Index sensor name for net production values (i.e. what is delivered back to the grid)')
    parser.add_argument('--nc', default='sensor.electricity_meter_power_consumption', help='Index sensor name for consumption values')
    # parser.add_argument('--covariance-matrix', default=None, help='Set the 5x5 covariance matrix. Relations to denote: gas, power-low, power-high, power-generation-low, power-generation-high. Of course, the diagonal entries are standard deviations and the non-diagonal entries provide pairwise covariance. should look like json: i.e. `[ [60,0,0,0,0], [0,60,0,0,0], [0,0,60,0,0], [0,0,0,60,0], [0,0,0,0,60] ]`. Use only if you know what you are doing.')
    parser.add_argument('--filter', type=str, default=None, help='Execute filter query on dataset. Example: --filter "`type==\'vast\' & name!=\'Oxxio 1 jaar\'"`. Exact syntax: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html')
    args = parser.parse_args()

    # profile = get_profile(args.profile, args.covariance_matrix, args.gas, args.power_low, args.power_high, args.power_gen_low, args.power_gen_high)
    df = pd.read_csv(args.history, comment='#')
    # df = normalize(df)
    if not verify(df, args.p, args.np, args.nc):
        return

    df = df.pivot(index = 'last_changed', columns='entity_id', values='state')
    df = df.fillna(0.0) # this is valid: used sensors fill NaN sometimes when no production is present.

    print(f'Number of datapoints: {len(df.index)}')
    if args.filter:
        df.query(args.filter, inplace=True)
        print(f'Number of datapoints (post-filter): {len(df.index)}')

    print('Base situation (no battery)')
    argdata = [args.p, args.np, args.nc] # arguments related to dataset
    printst(df, *argdata)

    print()
    print('Case 1a: with overflow, 10kWh battery, 5kW inverter unit.')
    sim_df = simulate_battery(df, *argdata, capacity_kwh=10, inverter_limit_kw=5, min_pct=0.20, max_pct=0.80, discard_overflow=False)
    printst(sim_df, *argdata)

    print()
    print('Case 1b: no overflow, 10kWh battery, 5kW inverter unit.')
    sim_df = simulate_battery(df, *argdata, capacity_kwh=10, inverter_limit_kw=5, min_pct=0.20, max_pct=0.80, discard_overflow=True)
    printst(sim_df, *argdata)

    print()
    printcmp(df, *argdata, sim_df, *argdata)

    print()
    print('Case 1c: with overflow, 5kWh battery, 5kW inverter unit.')
    sim_df = simulate_battery(df, *argdata, capacity_kwh=5, inverter_limit_kw=5, min_pct=0.20, max_pct=0.80, discard_overflow=False)
    printst(sim_df, *argdata)
    print()
    printcmp(df, *argdata, sim_df, *argdata)
    # mean_vector contains the usage/generation estimations.
    # mean_vector = np.array([profile['gas'], profile['power_low'], profile['power_high'], profile['power_gen_low'], profile['power_gen_high']])

    # computes gross need (i.e. amount of power that the house needs in total, from whatever source)
    # need = (df[args.p] - df[args.np]).clip(lower=0.0) + df[args.nc]

    # sufficiency: factor between 0-1, with 0 is 'total grid dependency' and 1 is 'no net needed' for a given hour
    # sufficiency = (1.0 - df[args.nc]/need).clip(upper=1.0)
    # 4. OVERRIDE: If export is more than grid import, 100% self-sufficient (handles np > p misreadings/battery exports)
    # sufficiency.loc[df[args.np] > df[args.nc]] = 1.0

    # print(sufficiency.to_string())

    return
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
