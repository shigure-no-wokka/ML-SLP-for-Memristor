from pathlib import Path

conductance_path = Path('./Conductance data')
GP_100_path = conductance_path/'100权重GP.txt'
GD_100_path = conductance_path/'100权重GD.txt'

with open(GP_100_path, 'r') as f:
    GP = []
    for each in f:
        GP.append(float(each.replace('\n', '')))
    print(GP)

with open(GD_100_path, 'r') as f:
    GD = []
    for each in f:
        GD.append(float(each.replace('\n', '')))
    print(GD)
