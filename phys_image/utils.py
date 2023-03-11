import yaml
import json

def read_yaml():
  with open("./variables.yaml", "r") as stream:
    try:
        return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)



def read_json(path):
  with open(path, 'r') as fp:
    json_file = json.load(fp)
  return json_file