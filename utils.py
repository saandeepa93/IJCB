import yaml

def read_yaml():
  with open("./variables.yaml", "r") as stream:
    try:
        return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)