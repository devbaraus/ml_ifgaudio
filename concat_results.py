#%%
import argparse
import time
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False 

#%%
parser = argparse.ArgumentParser(description='Arguments algo')
parser.add_argument('-f', action='store', dest='feature', required=False, help='Extrator de características',
                    default='lpc')
parser.add_argument('-a', nargs='+', type=int, action='store', dest='augmentation', required=False, help='Augmentation',
                    default=[10])
parser.add_argument('-n', nargs='+', type=float, action='store', dest='noise', required=False, help='Noise',
                    default=np.arange(0, 1.01, 0.25).tolist())
parser.add_argument('-t', type=bool, action='store', dest='trim', required=False, help='Use trim dataset',
                    default=False)
parser.add_argument('-m', type=str, action='store', dest='model', required=False, help='Model algorithm',
                    default='mlp')      
#%%
if not isnotebook():
    args = parser.parse_args()
else:
    class Parser:
        def __init__(self):
            self.feature = 'mfcc'
            self.augmentation = [0]
            self.noise = [0]
            self.trim = True
            self.model = 'mlp'
    args = Parser()
now = int(time.time())

def output_dir():
    t = ''
    if args.trim:
        t = '_trim'
    if args.noise != [0]:
        t += '_noise'
    if args.augmentation != [0]:
        t += f'_aug{args.augmentation[0]}'
    
    return t

params = {
    'noise': args.noise,
    'augmentation': args.augmentation,
    'feature': args.feature,
    'output_file': f'{args.feature}_{args.model}_{now}.csv',
    'output_dir': f'results/{args.model}/{args.feature}/{output_dir()}'
}
#%%
dfs = []

for f  in os.listdir(params['output_dir']):
    if f.startswith(f'{args.feature}_{args.model}') and f.endswith('.csv'):
        dfs.append(pd.read_csv(f'{params["output_dir"]}/{f}'))
        # os.remove(f'{params["output_dir"]}/{f}')

if len(dfs):
    if os.path.isfile(f'{params["output_dir"]}/{args.feature}_{args.model}.csv'):
        dfs.append(pd.read_csv(f'{params["output_dir"]}/{args.feature}_{args.model}.csv'))
    pd.concat(dfs).to_csv(f'{params["output_dir"]}/{args.feature}_{args.model}.csv')