import ConfigParser, json

def read_config(path):
    config = ConfigParser.SafeConfigParser()
    config.read(path)
    return config

def config_parser(path, section, parameter):
    config = read_config(path)
    return config.get(section,parameter)
