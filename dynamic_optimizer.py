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
    prices = pd.read_csv(path, comment='#', sep=';')
    ok = False
    if verify_dynamics(prices, colname):
        prices['prijs_excl_belastingen'] = prices['prijs_excl_belastingen'].str.replace(',', '.').astype(float)
        prices['datum_utc'] = pd.to_datetime(prices['datum_utc'])
        prices = prices.set_index('datum_utc')
        prices.index.name = 'last_changed'
        prices = prices[['prijs_excl_belastingen']].resample('1h').mean()
        prices.index = prices.index.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        ok = True
    return prices, ok

def calc(df, hp, hnp, hnc, pp, energy_tax, vat, fee_cons, fee_prod, discard_overflow=None, bat=None, bat_capacity_kwh=10, bat_inverter_limit_kw=5.0):
    '''
    basic calculation of energy cost.
    Parameters:
    - df: Input pandas DataFrame
    - hp, np, nc, pp: String names of your columns (hp = gross solar production, hnp = net production sent to the grid, hnc = net consumption from the grid, pp = price/kwh)
    - energy_tax: energy tax per kWh, paid only when consuming (not getting anything back when producing)
    - vat: vat calculated, received when producing and paid when consuming
    - fee_cons: fee per kWh for consuming
    - fee_prod: fee per kWh for producing
    - discard_overflow: can be None, 'all' or 'postitive'. On 'all', all net production is discarded. On 'positive', only discard when net-production costs money
    - bat: can be None, 'basic', 'simple-buy' or 'smart-buy'.
      - On None, no battery is included in calculations
      - On 'basic', battery follows basic behavior (not checking dynamic power pricing, just fill when possible and prefer discharge to net consumption).
      - On 'simple-buy', battery is charged when power price is 0 or negative
    - bat_capacity_kwh: amount of kwh to simulate battery with
    - bat_inverter_limit_kw: amount of kw a battery can produce/consume at once
    '''
    # costs overview:
    # hourly price: pay when consuming, receive when producing
    # energy taxes: pay when consuming,
    # VAT         : pay when consuming, receive when producing (calculate over hourly price + energy taxes)
    # buy/sell pri: pay when consuming, pay when producing
    copy = None
    if bat == 'basic':
        copy = b.simulate_battery(df, hp, hnp, hnc, capacity_kwh=bat_capacity_kwh, inverter_limit_kw=bat_inverter_limit_kw)
    elif bat == 'simple-buy':
        charge = lambda idx, row, possible_charge: possible_charge # always charge, never sell
        discharge = lambda idx, row, possible_discharge: 0 if (row[pp]*(1.0+vat)+fee_cons <= 0.0) else possible_discharge
        # charge 
        copy = b.simulate_battery_parametrized(df, hp, hnp, hnc, charge, discharge, capacity_kwh=bat_capacity_kwh, inverter_limit_kw=bat_inverter_limit_kw)
    else:
        copy = df.copy()

    if discard_overflow == 'all':
        copy[hnp] = 0.0 # no net production, discard ALL
    elif discard_overflow == 'positive':
        copy[(copy[hnp] > 0.0) & (copy[pp]*(1.0+vat) <= 0.0+fee_prod)] = 0.0 # discard production if price turns negative
    base_power = (copy[hnc] - copy[hnp])
    price = base_power * copy[pp]
    energy_tax = copy[hnc] * energy_tax
    vat = (price+energy_tax) * vat
    fee = copy[hnc] * fee_cons + copy[hnp] * fee_prod
    return price + energy_tax + vat + fee

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
    parser.add_argument('--fee-consumption', default=0.01815, help='Energy taxes per kWh (default is NL default VAT)')
    parser.add_argument('--fee-production', default=0.01270, help='Energy taxes per kWh (default is NL default VAT)')
    
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
    print(f'Number of datapoints: {len(df.index)} (so {len(df.index)/24:.2f} days of data)')

    # battery_modes = [None, 'basic', 'smart']
    battery_modes = [None, 'basic']
    discard_policies = [None, 'all', 'positive']
    for pol in discard_policies:
        for mode in battery_modes:
            print(f'Estimated dynamic contract cost (battery: {mode}) (discard overflow: {pol}) for 6 months...')
            out = calc(df, *houseargs, args.pp, args.energy_tax, args.vat, args.fee_consumption, args.fee_production, discard_overflow=pol, bat=mode, bat_capacity_kwh=5)
            print(f'Total: {out.sum():.3f} euros')
            if args.plot:
                plot(df, out)
    return

if __name__ == '__main__':
    main()
