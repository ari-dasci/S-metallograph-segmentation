import configparser

class insensitivedict(dict):
    def __getitem__(self, key : str):
        return super().__getitem__(key.lower())

def get_config(path="model.config", section=None):
    if section is not None:
        print("section is a deprecated parameter in get_config")
    config = configparser.ConfigParser(default_section=None)
    config.read(path)
    parameters = insensitivedict()
    if "general" in config:
        for k in config["general"].keys():
            parameters[k] = config["general"].get(k)
    if "str" in config:
        for k in config["str"].keys():
            parameters[k] = config["str"].get(k)
    if "int" in config:
        for k in config["int"].keys():
            parameters[k] = config["int"].getint(k)
    if "float" in config:
        for k in config["float"].keys():
            parameters[k] = config["float"].getfloat(k)
    if "bool" in config:
        for k in config["bool"].keys():
            parameters[k] = config["bool"].getboolean(k)
    if "boolean" in config:
        for k in config["boolean"].keys():
            parameters[k] = config["boolean"].getboolean(k)
    if "list" in config:
        for k in config["list"].keys():
            parameters[k] = config["list"].get(k).split(",")
    return parameters
