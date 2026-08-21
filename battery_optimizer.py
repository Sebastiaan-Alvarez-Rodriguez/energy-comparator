import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import shared as s

def set_if_not_null(d, key, value):
    if value:
        d[key] = value


def verify_house(df, *names):
    # verifies integrity of house csv file.
    required_csv_names = ['entity_id', 'state', 'last_changed']
    ok = True
    for name in required_csv_names:
        if name not in df:
            print(f'[Error] Missing column name "{name}" in csv header.')
            ok = False

    for n in names:
        if not any(df['entity_id'] == n):
            print(f'[Error] Could not find values for "{n}" in csv file.')
            ok = False
    return ok

def simulate_battery_parametrized(df, p_col, np_col, nc_col, charge, discharge, capacity_kwh=10, inverter_limit_kw=5.0, min_pct=0.20, max_pct=0.80):
    '''
    Simulates a home battery system and calculates adjusted grid metrics.
    Algorithm used is:
    - if power production:
        - if battery space free: in battery (limited by battery inverter strength)
        - the rest to grid / discard (depending on discard_overflow)
    - if power net consumption:
        - if battery power ready: from battery (limited by battery inverter strength)
        - the rest from grid
    Parameters:
    - df: Input pandas DataFrame
    - p_col, np_col, nc_col: String names of your columns (p = gross solar production, np = net production sent to the grid, nc = net consumption from the grid)
    - charge: function determining how much to charge. Function receives idx, row, self-produced available charge, possible_charge
      * note: if returned charge amount > available charge, assumes you make battery charge from the grid
    - discharge: function determining how much to discharge. Function receives idx, row, house-satisfying discharge, possible_discharge
      * note: if returned discharge amount > house-satisfying discharge, assumes you make battery discharge extra to sell to the grid
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
    for idx, row in df.iterrows():
        p = row[p_col]
        np = row[np_col]
        nc = row[nc_col]
        new_nc = 0.0
        new_np = 0.0
        assert max_charge - current_charge >= 0
        assert np >= 0
        assert nc >= 0
        
        # 1. HANDLE CHARGING (from production surplus or buying from grid)
        # charge limit is the minimum of: inverter limit, free-space in battery
        possible_charge = min(inverter_limit_kw, (max_charge - current_charge))
        chosen = charge(idx, row, np, possible_charge)
        assert chosen <= possible_charge
        current_charge += chosen
        new_np += max(0, np-chosen)
        new_nc += max(0, chosen-np) # if more than np is selected, battery charges with more than just our net produced power, i.e. uses grid!

        # 2. HANDLE DEFICIT/CONSUMPTION (from discharging or selling to grid)
        # discharge limit is the minimum of: inverter limit, charged-space in battery
        possible_discharge = min(inverter_limit_kw, (current_charge - min_charge))
        chosen = discharge(idx, row, nc, possible_discharge)
        assert chosen <= possible_discharge
        current_charge -= chosen
        new_np += max(0, chosen-nc) # if more than nc is selected, battery discharges more than house need, i.e. sell to the grid
        new_nc += max(0, nc-chosen)

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

    
def simulate_battery(df, p_col, np_col, nc_col, capacity_kwh=10, inverter_limit_kw=5.0, min_pct=0.20, max_pct=0.80, discard_overflow=False):
    '''
    Simulates a home battery system and calculates adjusted grid metrics.
    Algorithm used is:
    - if power production:
        - if battery space free: in battery (limited by battery inverter strength)
        - the rest to grid / discard (depending on discard_overflow)
    - if power net consumption:
        - if battery power ready: from battery (limited by battery inverter strength)
        - the rest from grid
    Parameters:
    - df: Input pandas DataFrame
    - p_col, np_col, nc_col: String names of your columns (p = gross solar production, np = net production sent to the grid, nc = net consumption from the grid)
    - capacity_kwh: Total battery capacity in kWh
    - inverter_limit_kw: The battery inverter limit on how many kW of power it can safely convert from or to the battery at any time.
    - min_pct: Minimum charge threshold (e.g., 0.20 for 20%)
    - max_pct: Maximum charge threshold (e.g., 0.80 for 80%)
    - discard_overflow: If True, excess solar is wasted when battery is full. If False, excess solar is delivered to the grid.
    '''
    charge = lambda idx, row, available_self_power, possible_charge: min(available_self_power, possible_charge)
    discharge = lambda idx, row, satisfying_self_power, possible_discharge: min(satisfying_self_power, possible_discharge)
    sim_df = simulate_battery_parametrized(df, p_col, np_col, nc_col, charge, discharge, capacity_kwh=capacity_kwh, inverter_limit_kw=inverter_limit_kw, min_pct=min_pct, max_pct=max_pct)
    if discard_overflow:
        sim_df[sim_df[np_col] > 0] = 0
    return sim_df

def printst(df, p, np, nc):
    print(f'grid energy consumed: {df[nc].sum()}kWh')
    print(f'OWN  energy consumed: {df[p].sum() - df[np].sum()}kWh')
    print(f'grid energy produced: {df[np].sum()}kWh')

def printcmp(df1, p1, np1, nc1, df2, p2, np2, nc2):
    reduction_kwh = df1[nc1].sum() - df2[nc2].sum()
    reduction_pct = reduction_kwh/df1[nc1].sum()
    print(f'Overal energy consumption reduction: {reduction_kwh}kWh ({reduction_pct *100}%)')
    overflow_kwh = df1[np1].sum() - df2[np2].sum()
    overflow_pct = overflow_kwh/df1[np1].sum()
    print(f'Overal energy overflowing reduction: {overflow_kwh}kWh ({overflow_pct*100}%)')
    return reduction_kwh, reduction_pct, overflow_kwh, overflow_pct

def summarize_simulation(inv, cap, df, p_col, np_col, nc_col, min_pct, max_pct, discard_overflow):
    '''summarizes a simulation outcome to its most important values'''
    np_sum = df[np_col].sum()
    return {
        'inverter': inv,
        'capacity': cap,
        'min_pct': min_pct,
        'max_pct': max_pct,
        'discard_overflow': discard_overflow,
        'nc_sum': df[nc_col].sum(),
        'np_sum': np_sum,
        'p_np_diff': df[p_col].sum() - np_sum
    }
    
def run_single_simulation(inv, cap, df, p_col, np_col, nc_col, min_pct, max_pct, discard_overflow):
    '''helper function for parallel processing'''
    sim = simulate_battery(df, p_col, np_col, nc_col, capacity_kwh=cap, inverter_limit_kw=inv, min_pct=min_pct, max_pct=max_pct, discard_overflow=discard_overflow)
    return summarize_simulation(inv, cap, sim, p_col, np_col, nc_col, min_pct, max_pct, discard_overflow)

def run_simulations(inverters, capacities, *args, **kwargs):
    '''
    Runs multiple simulations and returns the result as a pandas dataframe.

    Parameters:
    - inverters: list of battery inverter max rating in kW, e.g. [1, 3, 5]
    - capacities: list of battery capacities in kWh, e.g. [2.5, 5, 10]
    All other args can be spotted at 'run_single_simulation' function (except inv and cap, which are provided here)
    '''
    # ProcessPoolExecutor automatically uses all available CPU cores
    with ProcessPoolExecutor() as executor:
        futures = [ # Submit all tasks to the queue
            executor.submit(run_single_simulation, inv, cap, *args, **kwargs) for inv, cap in itertools.product(inverters, capacities)
        ]
        results = [f.result() for f in futures]
    return pd.DataFrame(results)

def plot_battery_cap(outputs_df, df, p_col, np_col, nc_col, min_pct=0.20, max_pct=0.80, discard_overflow=False):
    # plot battery capacity vs amount of kWh saved.
    inverter_colors = {
        1: '#1f77b4',   # Muted Blue
        2: '#ff7f0e',   # Safety Orange
        5: '#2ca02c',   # Grass Green
        10: '#9467bd'   # Deep Purple
    }
    metric_styles = {
        'nc_sum': {'style': '-', 'label': 'grid-consumption'},
        'np_sum': {'style': '--', 'label': 'grid-production'},
        "p_np_diff": {"style": ":", "label": "self-consumption"}
    }
    x_offsets = {1: -0.15, 2: -0.05, 5: 0.05, 10: 0.15}

    plt.figure(figsize=(8, 5))
    for inv, group in outputs_df.groupby('inverter'):
        color = inverter_colors.get(inv, 'gray') # Fallback color if size changes
        offset = x_offsets.get(int(inv), 0)
        shifted_capacity = group['capacity'] + offset
        for col, info in metric_styles.items():
            plt.plot(
                shifted_capacity, 
                group[col], 
                color=color, 
                linestyle=info['style'], 
                linewidth=2
            )

    # Construct a two-part custom legend to avoid a messy 12-item list
    legend_elements = (
        [plt.Line2D([0], [0], color=c, lw=3, label=f'Inverter {k} kW') for k, c in inverter_colors.items()] +
        [plt.Line2D([0], [0], color='gray', ls=s['style'], lw=2, label=s['label']) for s in metric_styles.values()]
    )

    # Figure layout styling
    plt.title('Battery Simulation: Grouped by Inverter Size', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Capacity (kWh)', fontsize=11, labelpad=10)
    plt.ylabel('Energy (kWh)', fontsize=11, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.4, color='gray')

    # Place the clean 7-item legend
    plt.legend(handles=legend_elements, loc='upper left', frameon=True, facecolor='white', edgecolor='none')

    plt.tight_layout()
    plt.show()
    
def plot_battery_rel_gain(outputs_df, inv, df, p_col, np_col, nc_col, min_pct=0.20, max_pct=0.80, discard_overflow=False):
    # create a barplot showing the gain as compared to doing nothing
    nc_base = df[nc_col].sum()
    np_base = df[np_col].sum()

    # Generate a bar plot for the 5kw inverter size
    group = outputs_df[outputs_df['inverter'] == inv]
    # for inv, group in outputs_df.groupby('Inverter'):
        # Ensure data is sorted by capacity for accurate 'previous' calculations
    group = group.sort_values('capacity').copy()

    # Calculate Bar 1 & 3: Reduction compared to 0 kWh baseline
    group['nc_red_base'] = ((nc_base - group['nc_sum']) / nc_base) * 100
    group['np_red_base'] = ((np_base - group['np_sum']) / np_base) * 100

    # Calculate Bar 2 & 4: Reduction compared to the previous capacity step
    # (Using shift() to get the previous row's value; first row defaults to 0)
    group['nc_red_prev'] = ((group['nc_sum'].shift(1) - group['nc_sum']) / group['nc_sum'].shift(1)).fillna(0) * 100
    group['np_red_prev'] = ((group['np_sum'].shift(1) - group['np_sum']) / group['np_sum'].shift(1)).fillna(0) * 100

    plt.figure(figsize=(14, 6))

    capacities_list = group['capacity'].values
    x = np.arange(len(capacities_list))  # Label locations
    width = 0.18                         # Width of each individual bar

    bar1 = plt.bar(x - 1.5 * width, group['nc_red_base'], width, label='NC vs 0kWh (Total Drop)', color='#2ca02c')
    bar2 = plt.bar(x - 0.5 * width, group['nc_red_prev'], width, label='NC vs Prev Step (Marginal)', color='#98df8a')
    bar3 = plt.bar(x + 0.5 * width, group['np_red_base'], width, label='NP vs 0kWh (Total Drop)', color='#1f77b4')
    bar4 = plt.bar(x + 1.5 * width, group['np_red_prev'], width, label='NP vs Prev Step (Marginal)', color='#aec7e8')

    # Add the text labels on top of Bar 2 and Bar 4
    # 'padding=3' spaces the text slightly above the bar; 'fontsize=8' ensures it fits the narrow width
    for bar in [bar1, bar2, bar3, bar4]:
        plt.gca().bar_label(
            bar, 
            fmt='%.1f%%', 
            rotation=90, 
            padding=3, 
            fontsize=8, 
            color='#2c3e50'
        )
    plt.title(f'Relative Energy Reduction Metrics - Inverter: {inv} kW', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Battery Capacity (kWh)', fontsize=11, labelpad=10)
    plt.ylabel('Reduction (%)', fontsize=11, labelpad=10)

    # Map the X-axis tick marks back to the actual capacity numbers
    plt.xticks(x, capacities_list)

    # Append '%' sign to the Y-axis labels
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))

    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')

    plt.tight_layout()
    plt.show()

def main():
    # Simple program to calculate optimal home battery size for a given power consumption
    # all power related specs expected in kWh, or euro/kWh where applicable.
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help='If set, simulates several different battery configurations for a household')
    s.house_args(parser)
    args = parser.parse_args()

    houseargs = [args.hp, args.hnp, args.hnc] # arguments related to house dataset
    df, ok = s.house_read(args.history_house, *houseargs, args.filter)
    if not ok:
        return

    print('Base situation (no battery)')
    printst(df, *houseargs)

    if args.plot:
        capacities = [x*0.5 for x in range (0, 61)]
        inverters = [1, 2, 5, 10]
        outputs_df = run_simulations(inverters, capacities, df, *houseargs, min_pct=0.2, max_pct=0.8, discard_overflow=False)
        plot_battery_cap(outputs_df, df, *houseargs)
        plot_battery_rel_gain(outputs_df, 5, df, *houseargs)

    print('Case: no overflow, 5kWh battery, 5kW inverter unit.')
    inv = 5
    cap = 5
    min_pct = 0
    max_pct = 1
    discard_overflow = True
    # sim_df = run_single_simulation(inv, cap, df, *houseargs, min_pct, max_pct, discard_overflow)
    sim_df = simulate_battery(df, *houseargs, capacity_kwh=cap, inverter_limit_kw=inv, min_pct=min_pct, max_pct=max_pct, discard_overflow=discard_overflow)
    printst(sim_df, *houseargs)
    reduction_kwh, reduction_pct, overflow_kwh, overflow_pct = printcmp(df, *houseargs, sim_df, *houseargs)
    # 2 main options here:
    # 1. calculate exact pricing for my data (less than 1 year of data available!) and assume the optimal contract is optimal for the full year
    # 2. assume the percentage gains for my data applies to the entire year, and calculate the optimal contract for many generated sample years.
    print()
    print('Assumptions for cost-basis analysis')
    print(f'\t- consumption reduction for total year: {reduction_pct}')
    print(f'\t- overflowing reduction for total year: {overflow_pct}')
    # option 2 chosen, because I think it is doing the better assumption.
    # option 2 requires me to either
    # 1. choose 1 cap-inv combo, find optimal contract
    # 2. choose many cap-inv combos, find optimal contracts
    # I will use my other python file for finding an optimal contract.
    # contract_calculations(outputs_df)
    return

if __name__ == '__main__':
    main()
