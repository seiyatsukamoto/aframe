import os

import jsonargparse

from export.main import export
from export.mm_main import mm_export
from utils.logging import configure_logging
from export.mm_modules import separate_model
import logging

def build_parser():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--config", action=jsonargparse.ActionConfigFile)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_function_arguments(export)
    return parser


def main(args=None):
    parser = build_parser()
    args = parser.parse_args(args)
    logfile = args.pop("logfile")
    if logfile is not None:
        logdir = os.path.dirname(logfile)
        os.makedirs(logdir, exist_ok=True)
    verbose = args.pop("verbose")
    configure_logging(logfile, verbose)
    args = args.as_dict()
    logging.info(args['weights'])
    logging.info(args['repository_directory'])
    logging.info(args['clean'])
    logging.info(args['aframe_instances'])
    logging.info(args['batch_file'])
    if ('model_type' in args.keys()) and (args['model_type'] == 'mm'):
        separate_model(**args)
        mm_export(**args)
    else:
        export(**args)

if __name__ == "__main__":
    main()
