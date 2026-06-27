import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


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

def verify_prices(df, *names):
    # verifies integrity of house csv file.
    ok = True
    for name in names:
        if name not in df:
            print(f'[Error] Missing column name "{name}" in csv header.')
            ok = False
    return ok

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
    print(f'grid energy consumed: {df[nc].sum()}kWh')
    print(f'OWN  energy consumed: {df[p].sum() - df[np].sum()}kWh')
    print(f'grid energy produced: {df[np].sum()}kWh')

def printcmp(df1, p1, np1, nc1, df2, p2, np2, nc2):
    reduction_kwh = df1[nc1].sum() - df2[nc2].sum()
    print(f'Overal energy consumption reduction: {reduction_kwh}kWh ({reduction_kwh / df1[nc1].sum() *100}%)')
    overflow_kwh = df1[np1].sum() - df2[np2].sum()
    print(f'Overal energy overflowing reduction: {overflow_kwh}kWh ({overflow_kwh / df1[np1].sum() *100}%)')

# helper function for parallel processing
def run_single_simulation(inv, cap, df, p_col, np_col, nc_col, min_pct, max_pct, discard_overflow):
    sim = simulate_battery(df, p_col, np_col, nc_col, capacity_kwh=cap, inverter_limit_kw=inv, min_pct=min_pct, max_pct=max_pct, discard_overflow=discard_overflow)
    # Calculate metrics in one place
    np_sum = sim[np_col].sum()
    return {
        "Inverter": inv,
        "Capacity": cap,
        "NC_Sum": sim[nc_col].sum(),
        "NP_Sum": np_sum,
        "P_NP_Diff": sim[p_col].sum() - np_sum
    }


def plot_battery(df, p_col, np_col, nc_col, min_pct=0.20, max_pct=0.80, discard_overflow=False):
    capacities = [x*0.5 for x in range (0, 61)]
    inverters = [1, 2, 5, 10]
    inverter_colors = {
        1: "#1f77b4",   # Muted Blue
        2: "#ff7f0e",   # Safety Orange
        5: "#2ca02c",   # Grass Green
        10: "#9467bd"   # Deep Purple
    }
    metric_styles = {
        "NC_Sum": {"style": "-", "label": "grid-consumption"},
        "NP_Sum": {"style": "--", "label": "grid-production"},
        # "P_NP_Diff": {"style": ":", "label": "P-NP Diff"}
    }
    x_offsets = {1: -0.15, 2: -0.05, 5: 0.05, 10: 0.15}

    # ProcessPoolExecutor automatically uses all available CPU cores
    with ProcessPoolExecutor() as executor:
        futures = [ # Submit all tasks to the queue
            executor.submit(run_single_simulation, inv, cap, df, p_col, np_col, nc_col, min_pct, max_pct, discard_overflow ) for inv, cap in itertools.product(inverters, capacities)
        ]
        results = [f.result() for f in futures]
    outputs_df = pd.DataFrame(results)




    plt.figure(figsize=(8, 5))
    for inv, group in outputs_df.groupby("Inverter"):
        color = inverter_colors.get(inv, "gray") # Fallback color if size changes
        offset = x_offsets.get(int(inv), 0)
        shifted_capacity = group["Capacity"] + offset
        for col, info in metric_styles.items():
            plt.plot(
                shifted_capacity, 
                group[col], 
                color=color, 
                linestyle=info["style"], 
                linewidth=2
            )

    # Construct a two-part custom legend to avoid a messy 12-item list
    legend_elements = (
        [plt.Line2D([0], [0], color=c, lw=3, label=f"Inverter {k} kW") for k, c in inverter_colors.items()] +
        [plt.Line2D([0], [0], color="gray", ls=s["style"], lw=2, label=s["label"]) for s in metric_styles.values()]
    )

    # Figure layout styling
    plt.title("Battery Simulation: Grouped by Inverter Size", fontsize=14, pad=15, fontweight='bold')
    plt.xlabel("Capacity (kWh)", fontsize=11, labelpad=10)
    plt.ylabel("Energy (kWh)", fontsize=11, labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.4, color="gray")

    # Place the clean 7-item legend
    plt.legend(handles=legend_elements, loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()
    plt.show()
    
    # next we create a barplot showing the gain as compared to doing nothing
    nc_base = df[nc_col].sum()
    np_base = df[np_col].sum()

    # Generate a bar plot for the 5kw inverter size
    inv = 5
    group = outputs_df[outputs_df['Inverter'] == inv]
    # for inv, group in outputs_df.groupby("Inverter"):
        # Ensure data is sorted by capacity for accurate "previous" calculations
    group = group.sort_values("Capacity").copy()

    # Calculate Bar 1 & 3: Reduction compared to 0 kWh baseline
    group["NC_Red_Base"] = ((nc_base - group["NC_Sum"]) / nc_base) * 100
    group["NP_Red_Base"] = ((np_base - group["NP_Sum"]) / np_base) * 100

    # Calculate Bar 2 & 4: Reduction compared to the previous capacity step
    # (Using shift() to get the previous row's value; first row defaults to 0)
    group["NC_Red_Prev"] = ((group["NC_Sum"].shift(1) - group["NC_Sum"]) / group["NC_Sum"].shift(1)).fillna(0) * 100
    group["NP_Red_Prev"] = ((group["NP_Sum"].shift(1) - group["NP_Sum"]) / group["NP_Sum"].shift(1)).fillna(0) * 100

    plt.figure(figsize=(14, 6))

    capacities_list = group["Capacity"].values
    x = np.arange(len(capacities_list))  # Label locations
    width = 0.18                         # Width of each individual bar

    bar1 = plt.bar(x - 1.5 * width, group["NC_Red_Base"], width, label="NC vs 0kWh (Total Drop)", color="#2ca02c")
    bar2 = plt.bar(x - 0.5 * width, group["NC_Red_Prev"], width, label="NC vs Prev Step (Marginal)", color="#98df8a")
    bar3 = plt.bar(x + 0.5 * width, group["NP_Red_Base"], width, label="NP vs 0kWh (Total Drop)", color="#1f77b4")
    bar4 = plt.bar(x + 1.5 * width, group["NP_Red_Prev"], width, label="NP vs Prev Step (Marginal)", color="#aec7e8")

    # Add the text labels on top of Bar 2 and Bar 4
    # 'padding=3' spaces the text slightly above the bar; 'fontsize=8' ensures it fits the narrow width
    for bar in [bar1, bar2, bar3, bar4]:
        plt.gca().bar_label(
            bar, 
            fmt="%.1f%%", 
            rotation=90, 
            padding=3, 
            fontsize=8, 
            color="#2c3e50"
        )
    plt.title(f"Relative Energy Reduction Metrics - Inverter: {inv} kW", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Battery Capacity (kWh)", fontsize=11, labelpad=10)
    plt.ylabel("Reduction (%)", fontsize=11, labelpad=10)

    # Map the X-axis tick marks back to the actual capacity numbers
    plt.xticks(x, capacities_list)

    # Append '%' sign to the Y-axis labels
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))

    plt.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()
    plt.show()

def main():
    # Simple program to calculate optimal home battery size for a given power consumption
    # all power related specs expected in kWh, or euro/kWh where applicable.
    parser = argparse.ArgumentParser()
    parser.add_argument('--history-house', default=str(Path.cwd() / 'history_house.csv'), help='Data file containing house historical gross production and grid consumption values.')
    parser.add_argument('--hp', default='sensor.solaredge_solar_power', help='Index name for sensor production values')
    parser.add_argument('--hnp', default='sensor.electricity_meter_power_production', help='Index name for sensor net production values (i.e. what is delivered back to the grid)')
    parser.add_argument('--hnc', default='sensor.electricity_meter_power_consumption', help='Index name for sensor consumption values')
    parser.add_argument('--history-prices', default=str(Path.cwd() / 'history_prices.csv'), help='Data file containing historical dynamic prices')
    parser.add_argument('--pp', default='prijs_excl_belastingen', help='Index name for prices')
    parser.add_argument('--filter', type=str, default=None, help='Execute filter query on house dataset. Example: --filter "`type==\'vast\' & name!=\'Oxxio 1 jaar\'"`. Exact syntax: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html')

    args = parser.parse_args()

    df = pd.read_csv(args.history_house, comment='#')
    houseargs = [args.hp, args.hnp, args.hnc] # arguments related to house dataset
    if not verify_house(df, *houseargs):
        return
    df = df.pivot(index = 'last_changed', columns='entity_id', values='state')
    df = df.fillna(0.0) # this is valid: used sensors fill NaN sometimes when no production is present.

    prices = pd.read_csv(args.history_prices, comment='#', sep=';')
    priceargs = [args.pp]
    if not verify_prices(prices, *priceargs):
        return
    prices['prijs_excl_belastingen'] = prices['prijs_excl_belastingen'].str.replace(',', '.').astype(float)
    prices['datum_utc'] = pd.to_datetime(prices['datum_utc'])
    prices = prices.set_index('datum_utc')
    prices.index.name = 'last_changed'
    prices = prices[['prijs_excl_belastingen']].resample('1h').mean()
    prices.index = prices.index.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    df = df.merge(prices, left_index=True, right_index=True,  how='left') # with this trick: hourly prices are added to the main df, using left join. If there is no hit for the measured time in 'df', then a NaN is filled in.

    print(f'Number of datapoints: {len(df.index)}')
    if args.filter:
        df.query(args.filter, inplace=True)
        print(f'Number of datapoints (post-filter): {len(df.index)}')

    print('Base situation (no battery)')
    printst(df, *houseargs)

    print()
    print('Case 1a: with overflow, 10kWh battery, 5kW inverter unit.')
    sim_df = simulate_battery(df, *houseargs, capacity_kwh=10, inverter_limit_kw=5, min_pct=0.20, max_pct=0.80, discard_overflow=False)
    printst(sim_df, *houseargs)
    printcmp(df, *houseargs, sim_df, *houseargs)

    # print()
    # print('Case 1b: no overflow, 10kWh battery, 5kW inverter unit.')
    # sim_df = simulate_battery(df, *houseargs, capacity_kwh=10, inverter_limit_kw=5, min_pct=0.20, max_pct=0.80, discard_overflow=True)
    # printst(sim_df, *houseargs)
    # printcmp(df, *houseargs, sim_df, *houseargs)

    # print()
    # print('Case 1c: with overflow, 5kWh battery, 5kW inverter unit.')
    # sim_df = simulate_battery(df, *houseargs, capacity_kwh=5, inverter_limit_kw=5, min_pct=0.20, max_pct=0.80, discard_overflow=False)
    # printst(sim_df, *houseargs)
    # printcmp(df, *houseargs, sim_df, *houseargs)

    plot_battery(df, *houseargs)
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
