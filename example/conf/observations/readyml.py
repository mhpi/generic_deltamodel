import yaml

with open("camels_671.yaml", "r") as f:
    data = yaml.safe_load(f)

print(data)
import pprint
pprint.pprint(data)