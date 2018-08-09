# coding: utf-8
import os
import sys, traceback
import argparse
import json
import array


import random
import numpy as np
import torch
SEED = 496343

def define_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
define_random_seed(SEED)

import torch
cuda_available = torch.cuda.is_available()
print("Is CUDA available? ", cuda_available )

import sys
#print(sys.executable)
#print(sys.path)

parser = argparse.ArgumentParser()
parser.add_argument("-rcp", "--rest_config_path", action='store',
                    help="Path for REST web service config file",
                    required=False) 
parser.add_argument("-tcp", "--train_config_path", action='store',
                    help="Path for train/test config file",
                    required=False) 
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()
if args.verbose:
    print("verbosity turned on")

if(not args.rest_config_path and not args.train_config_path):
    print("At least one config file has to be given: either for train or for REST web service")
    parser.print_help(sys.stderr)
    sys.exit(0)  
REST=False
TRAIN=False
if(args.rest_config_path): 
    if not os.path.isfile(args.rest_config_path):    
        print("Provided rest_config_path does not exists.")
        sys.exit(0) 
    REST=True
    REST_config  = json.load(open(args.rest_config_path))
    print("REST_config:\n\t", REST_config)
    import web_server

if(args.train_config_path): 
    if not os.path.isfile(args.train_config_path):    
        print("Provided train_config_path does not exists.")
        sys.exit(0) 
    TRAIN=True
    train_config = json.load(open(args.train_config_path))
    print("train_config:\n\t",train_config)
    import trainer


def main():
    if(REST): 
        web_server.run_server()
    elif(TRAIN): 
        trainer.train()

if __name__ == "__main__":
    main()