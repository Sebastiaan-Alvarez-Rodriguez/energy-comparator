# Energy-Comparator
Small project to help picking electricity and gas contracts.

## Requirements
 - numpy
 - pandas

 Also for dynamic optimizer:
 - emhass

## Data requirements
For contract optimizer:
- know your yearly electricity and gas usages

For battery optimizer:
- download quarterly/hourly power statistics from home-assistant, e.g. [here](https://ha.h.mijn.place/history?entity_id=sensor.solaredge_solar_power%2Csensor.electricity_meter_power_production%2Csensor.electricity_meter_power_consumption&start_date=2026-01-01T00%3A00%3A00.000Z&end_date=2026-10-23T23%3A59%3A00.000Z)
  - name it `history_house.csv`
  Just select the range to use for simulations.

For dynamic optimizer:
- download quarterly/hourly power statistics from home-assistant (same as battery optimizer)
  - name it `history_house.csv`
- download historical dynamic power prices from e.g. [jeroen](https://jeroen.nl/account/data/stroomprijzen) as CSV file
  - name it `history_prices.csv`

## Usage
To use this tool:

1. Compute your average gas consumption, power consumption (divided in low-tariff and high-tariff) and power generation (divide in low-tariff and high-tariff).
2. (optional) Compute/estimiate your standard deviation.
3. (optional) Populate a covariance matrix.
4. Obtain (i.e. create/get) a csv file with contracts to compare. The csv file should contain the fields as provided in `contract.csv`
 
Then, run the program using, e.g:
```bash
python3 battery_optimizer.py
python3 contract_optimizer.py --gas 756 --power-low 2903 --power-high 1520 --power-gen-low 4119 --power-gen-high 1622  --covariance-matrix "[ [60,20,20,-7,-14], [20,60,8,0,0], [20,8,60,0,0], [-7,0,0,60,0], [-14,0,0,0,60] ]" --filter "type=='vast' & name.str.contains('united consumers')"
```
