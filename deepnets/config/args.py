import configargparse

parser = configargparse.ArgParser()
parser.add_argument("--config", is_config_file=True)


def add_config(parser):
    parser.add_argument("--config", is_config_file=True)
