from pathlib import Path
import pandas as pd

def house_args(parser):
    parser.add_argument('--history-house', default=str(Path.cwd() / 'history_house.csv'), help='Data file containing house historical gross production and grid consumption values.')
    parser.add_argument('--hp', default='sensor.solaredge_solar_power', help='Index name for sensor production values')
    parser.add_argument('--hnp', default='sensor.electricity_meter_power_production', help='Index name for sensor net production values (i.e. what is delivered back to the grid)')
    parser.add_argument('--hnc', default='sensor.electricity_meter_power_consumption', help='Index name for sensor consumption values')
    parser.add_argument('--filter', type=str, default=None, help='Execute filter query on house dataset. Example: --filter "`type==\'vast\' & name!=\'Oxxio 1 jaar\'"`. Exact syntax: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html')

def house_verify(df, *names):
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

def house_read(path, hp, hnp, hnc, filter):
    '''Reads house usage data (hourly statistics)'''
    df = pd.read_csv(path, comment='#')
    ok = False
    if house_verify(df, hp, hnp, hnc):
        df = df.pivot(index = 'last_changed', columns='entity_id', values='state')
        df = df.fillna(0.0) # this is valid: used sensors fill NaN sometimes when no production is present.
        ok = True
        print(f'Number of datapoints read from house: {len(df.index)}')
        if filter:
            df.query(filter, inplace=True)
            print(f'Number of datapoints read from house (post-filter): {len(df.index)}')
    return df, ok
