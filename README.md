# Energy-Comparator
Small project to help picking electricity and gas contracts.

## Requirements
 - numpy
 - pandas

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
