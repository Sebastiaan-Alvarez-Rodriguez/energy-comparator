import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import battery_optimizer as b
import shared as s

def set_if_not_null(d, key, value):
    if value:
        d[key] = value

def verify_dynamics(df, *names):
    # verifies integrity of house csv file.
    ok = True
    for name in names:
        if name not in df:
            print(f'[Error] Missing column name "{name}" in csv header.')
            ok = False
    return ok

def read_dynamics(path, colname):
    '''Reads dynamic input prices from jeroen.nl (export as csv). Also normalizes data to hourly samples'''
    df = pd.read_csv(path, comment='#', sep=';')
    ok = False
    if verify_dynamics(df, colname):
        df['prijs_excl_belastingen'] = df['prijs_excl_belastingen'].str.replace(',', '.').astype(float)
        df['datum_utc'] = pd.to_datetime(df['datum_utc'])
        df = s.drop_nonfloats(df, colname)

        df = df.set_index('datum_utc')
        df.index.name = 'last_changed'
        df = df[['prijs_excl_belastingen']].resample('1h').mean()
        df.index = df.index.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        ok = True
    return df, ok

def do_emhass(df, hp, hnp, hnc, pp, energy_tax, vat, fee_cons, fee_prod, discard_overflow=None, bat_capacity_kwh=10, bat_inverter_limit_kw=5.0, bat_min_pct=0.2, bat_max_pct=0.8):
    from emhass.optimization import Optimization
    from emhass.utils import get_root, get_logger

    copy = df.copy()
    hc = 'hc' # gross consumption (i.e. before own production)
    copy[hc] = copy[hnc] + copy[hp] - copy[hnp]

    copy.rename(columns={ hp: 'prod', hc: 'load' }, inplace=True)
    copy['prod'] *= 1000 # was kWh, make it W (note stepsize is 1hr)
    copy['load'] *= 1000 # was kWh, make it W (note stepsize is 1hr)
    copy['price_buy'] = (copy[pp] + energy_tax + fee_cons) * (1.0+vat) # total price to buy a kWh
    copy['price_sell'] = (copy[pp] + fee_prod) # total price to sell a kWh

    retrieve_hass_conf = {
        'optimization_time_step': pd.Timedelta('0 days 01:00:00'),  # e.g. 30 minute steps (I have 60-minute data steps, is that important for this?)
        'time_zone': 'Europe/Amsterdam',
        'Latitude': 52.51,
        'Longitude': 5.0,
        'sensor_power_photovoltaics': 'prod',
        'sensor_power_load_no_var_loads': 'load',
    }
    optim_conf = {
        # best documentation found for parameter meanings: https://github.com/davidusb-geek/emhass/blob/master/src/emhass/static/openapi.json
        'costfun': 'profit',  # Maximaliseer winst / minimaliseer kosten
        'set_use_battery': True,
        # 'lp_solver': 'PULP_CBC_CMD', # works but is a bit slow
        'lp_solver': 'HIghs',
        # 'lp_solver_path': '/usr/bin/cbc', # what is this anyway, I don't have it installed. CAN'T I JUST HAVE THIS SIMPLY RUN?!
        'num_threads': 0,
        'number_of_deferrable_loads': 0, # amount of shiftable loads like e.g. washing machine, to different timepoints.
        'weight_battery_discharge': 0.0, # extra weight or 'cost' for discharging battery (default 0.0) 
        'weight_battery_charge': 0.0, # extra weight or 'cost' for charging battery (default 0.0) 
        'operating_hours_of_each_deferrable_load': [],
        'start_timesteps_of_each_deferrable_load': [],
        'end_timesteps_of_each_deferrable_load': [],
        'nominal_power_of_deferrable_loads': 0,
        'set_nocharge_from_grid': False, # forbid charging battery from the grid?
        'set_nodischarge_to_grid': True, # forbid delivering battery power to the grid?
        'set_battery_dynamic': False, # Set a power dynamic limiting condition to the battery power
        'set_total_pv_sell': False, # assume all PV power is injected directly into the grid, i.e. no self-consumption possible before hitting grid.
        'set_use_pv': True,
        'lp_solver_mip_rel_gap': 0.02,
    }

    plant_conf = {
        'maximum_power_from_grid': 15000,
        'maximum_power_to_grid': 15000,
        # Batterij parameters (Nu in Wh en fracties voor SoC)
        'battery_nominal_energy_capacity': bat_capacity_kwh * 1000,  # Wh
        'battery_charge_power_max': bat_inverter_limit_kw * 1000,    # W
        'battery_discharge_power_max': bat_inverter_limit_kw * 1000, # W
        'battery_charge_efficiency': 0.90,
        'battery_discharge_efficiency': 0.90,
        'battery_minimum_state_of_charge': bat_min_pct,
        'battery_maximum_state_of_charge': bat_max_pct,
        'battery_target_state_of_charge': 0.5,
        'inverter_is_hybrid': False, # not a inverter that does both PV and battery AC/DC conversions
        'compute_curtailment': False, # idk, official statement: When enabled, curtailment that is cost-equivalent is scheduled as late as possible in the optimization horizon
    }
    emhass_conf = {
        'retrieve_hass_conf': retrieve_hass_conf,
        'optim_conf': optim_conf,
        'plant_conf': plant_conf,
        'data_path': Path('/var/tmp'),
    }

    logger, _ = get_logger(fun_name='emhass_perfect_optim', emhass_conf=emhass_conf)
    opt = Optimization(
        retrieve_hass_conf = retrieve_hass_conf,
        optim_conf = optim_conf,
        plant_conf = plant_conf,
        var_load_cost = 'price_buy',
        var_prod_price = 'price_sell',
        costfun = 'profit',
        emhass_conf = emhass_conf,
        logger = logger,
    )

    runtime_data = {
        'pv_power_forecast': copy['prod'].tolist(),
        'load_power_forecast': copy['load'].tolist(),
        'load_cost_forecast': copy['price_buy'].tolist(),
        'prod_price_forecast': copy['price_sell'].tolist(),
    }
    # df_res = opt.perform_perfect_optimization(copy, runtime_data)
    df_res = opt.perform_optimization(
        data_opt=copy,
        p_pv=copy['prod'].values,
        p_load=copy['load'].values,
        unit_load_cost=copy['price_buy'].values,
        unit_prod_price=copy['price_sell'].values,
    )
    return df_res['cost_profit'] * -1 # flip the number signs, i.e. -800 originally meant '800 euros cost' while 50 meant '50 euros returned'

def calc(df, hp, hnp, hnc, pp, energy_tax, vat, fee_cons, fee_prod, discard_overflow=None, bat=None, bat_capacity_kwh=10, bat_inverter_limit_kw=5.0, bat_min_pct=0.20, bat_max_pct=0.80):
    '''
    basic calculation of energy cost.
    Parameters:
    - df: Input pandas DataFrame
    - hp, hnp, hnc, pp: String names of your columns (hp = gross solar production, hnp = net production sent to the grid, hnc = net consumption from the grid, pp = price/kwh)
    - energy_tax: energy tax per kWh, paid only when consuming (not getting anything back when producing)
    - vat: vat as a factor (e.g. 0.21), received when we are the paying party (i.e. producing at a positive price/kWh, or consuming at a negative price/kWh)
    - fee_cons: fee per kWh for consuming
    - fee_prod: fee per kWh for producing
    - discard_overflow: can be None, 'all' or 'postitive'. On 'all', all net production is discarded. On 'positive', only discard when net-production costs money
    - bat: can be None, 'basic', 'simple-buy' or 'smart-buy'.
      - On None, no battery is included in calculations
      - On 'basic', battery follows basic behavior (not checking dynamic power pricing, just fill when possible and prefer discharge to net consumption).
      - On 'simple-buy', battery is charged when power price is 0 or negative
    - bat_capacity_kwh: amount of kwh to simulate battery with
    - bat_inverter_limit_kw: amount of kw a battery can produce/consume at once
    - bat_min_pct: min charge to keep in the battery (between 0-1)
    - bat_max_pct: max charge to keep in the battery (between 0-1)
    '''
    # costs overview:
    # hourly price: pay when consuming, receive when producing
    # energy taxes: pay when consuming,
    # VAT         : pay when consuming, (goes over the energy tax as well)
    # buy/sell pri: pay when consuming, pay when producing
    copy = None
    if bat == 'basic':
        copy = b.simulate_battery(df, hp, hnp, hnc, capacity_kwh=bat_capacity_kwh, inverter_limit_kw=bat_inverter_limit_kw, min_pct=bat_min_pct, max_pct=bat_max_pct)
    elif bat == 'simple-buy':
        charge = lambda idx, row, available_self_power, possible_charge: min(available_self_power, possible_charge) if ((row[pp]+fee_cons+energy_tax)*(1.0+vat) > 0.0) else possible_charge # always charge with own power, if price is negative also with grid power
        discharge = lambda idx, row, satisfying_self_power, possible_discharge: min(satisfying_self_power, possible_discharge) # we are not going to 'smart-sell' here.
        copy = b.simulate_battery_parametrized(df, hp, hnp, hnc, charge, discharge, capacity_kwh=bat_capacity_kwh, inverter_limit_kw=bat_inverter_limit_kw, min_pct=bat_min_pct, max_pct=bat_max_pct)
    elif bat == 'emhass':
        copy = df.copy()
        return do_emhass(copy, hp, hnp, hnc, pp, energy_tax, vat, fee_cons, fee_prod, discard_overflow=discard_overflow, bat_capacity_kwh=bat_capacity_kwh, bat_inverter_limit_kw=bat_inverter_limit_kw, bat_min_pct=bat_min_pct, bat_max_pct=bat_max_pct)
    else:
        copy = df.copy()

    if discard_overflow == 'all':
        copy[hnp] = 0.0 # no net production, discard ALL
    elif discard_overflow == 'positive':
        copy.loc[(copy[hnp] > 0.0) & (copy[pp] <= fee_prod), hnp] = 0.0 # discard production if price turns negative
    return compute_costs(copy, hp, hnp, hnc, pp, energy_tax, vat, fee_cons, fee_prod)

def compute_costs(df, hp, hnp, hnc, pp, energy_tax, vat, fee_cons, fee_prod):
    price_consume = df[hnc] * df[pp] # net consumption * kwh price gross
    price_produce = df[hnp] * df[pp] * -1 # net production * kwh price gross * -1 (because I get the money for production)

    energy_tax = df[hnc] * energy_tax # energy tax in NL is fixed amount 9.something cents per kWh, only paid when consuming, no longer received when producing kWh's

    vat_consume = (price_consume+energy_tax) * vat
    # vat_produce = price_produce * vat # NOTE: I get VAT if the price/kWh + energy_tax is negative, so this is correct
    vat_produce = np.where(price_produce > 0, 0.0, price_produce * vat)
    fee = df[hnc] * fee_cons + df[hnp] * fee_prod
    return price_consume + price_produce + energy_tax + vat_consume + vat_produce + fee
    

def plot(df, prices):
    '''plots the cost development over time'''
    plt.figure(figsize=(8, 5))

    # plt.plot(prices.index, prices)
    # plt.plot(prices.index.astype(np.datetime64), prices[pp], label='paid hourly price')
    # plt.plot(prices.cumsum(), label='total cost')
    prices.plot(label='paid hourly price')
    prices.cumsum().plot(label='total cost')
    plt.title(f'Cost development ({prices.sum():.2f} eu)', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('time', fontsize=11, labelpad=10)
    plt.ylabel('cost (eu)', fontsize=11, labelpad=10)
    plt.show()


def add_args(parser):
    parser.add_argument('--history-prices', default=str(Path.cwd() / 'history_prices.csv'), help='Data file containing historical dynamic prices')
    parser.add_argument('--pp', default='prijs_excl_belastingen', help='Index name for prices')
    parser.add_argument('--energy-tax', default=0.09161, help='Energy taxes per kWh (default is NL 2026 energy tax)')
    parser.add_argument('--vat', default=0.21, help='Energy taxes per kWh (default is NL default VAT)')
    parser.add_argument('--fee-consumption', default=0.01815, help='Energy taxes per kWh (default is NL avg)')
    parser.add_argument('--fee-production', default=0.01270, help='Energy taxes per kWh (default is NL avg)')
    
def main():
    # Simple program to calculate use of dynamic contracts based on real house usage
    # all power related specs expected in kWh, or euro/kWh where applicable.
    parser = argparse.ArgumentParser()
    add_args(parser)
    parser.add_argument('--plot', action='store_true', help='If set, simulates several different battery configurations for a household')
    s.house_args(parser)
    args = parser.parse_args()

    houseargs = [args.hp, args.hnp, args.hnc] # arguments related to house dataset
    df, ok = s.house_read(args.history_house, *houseargs, args.filter)
    if not ok:
        return

    prices, ok = read_dynamics(args.history_prices, args.pp)
    if not ok:
        return
    df = df.merge(prices, left_index=True, right_index=True,  how='left') # with this trick: hourly prices are added to the main df, using left join. If there is no hit for the measured time in 'df', then a NaN is filled in.
    df = df.dropna(subset=[args.pp]) # drop data for which we have no dynamic prices
    print(f'Number of datapoints (dynamics-house combined data): {len(df.index)} (so {len(df.index)/24:.2f} days of data)')
    print(f'Stats:')
    print(f'  {(df[args.hnc].sum()):.2f}kWh net consumed')
    print(f'  {df[args.hnp].sum():.2f}kWh net produced')
    print()

    # battery_modes = [None, 'basic', 'smart']
    battery_modes = [None, 'basic', 'simple-buy', 'emhass']
    discard_policies = [None, 'all', 'positive']
    for pol in discard_policies:
        for mode in battery_modes:
            if mode == 'emhass' and pol != None:
                continue
            print(f'Estimated dynamic contract cost (battery: {mode}) (discard overflow: {pol}) for 6 months...')
            out = calc(df, *houseargs, args.pp, args.energy_tax, args.vat, args.fee_consumption, args.fee_production, discard_overflow=pol, bat=mode, bat_capacity_kwh=5)
            print(f'Total: {out.sum():.3f} euros')
            if args.plot:
                plot(df, out)
    return

if __name__ == '__main__':
    main()
