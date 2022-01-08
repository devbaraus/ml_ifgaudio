#%%
import argparse
import time
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
    'output_dir': f'results/{args.model}/{args.feature}/{output_dir()}'
}


#%%
df = pd.read_csv(f'{params["output_dir"]}/{args.feature}_{args.model}.csv')
df['f1_micro'] = df['f1_micro'].astype(float)
df['f1_macro'] = df['f1_macro'].astype(float)


#%%
if args.noise == [0]:
    ticks = np.unique(df['feature_coeff'].astype(float).values)
    legends = []
    fig, ax = plt.subplots()
    for segment in np.unique(df['segment_time'].astype(float).values):
        plot_df = df[df['segment_time'] == segment]
        ax.plot(ticks,plot_df['f1_micro'].values, linewidth=2)
        ax.plot(ticks,plot_df['f1_macro'].values, linewidth=2)
        legends.append(f'{segment}s - micro')
        legends.append(f'{segment}s - macro')

    plt.title(f'{args.feature.upper()} - {args.model.upper()}')
    ax.legend(legends)
    ax.xaxis.set_ticks(ticks)
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.10))
    ax.set_xlabel('Feature coefficient')
    ax.set_ylabel('F1 score')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{params["output_dir"]}/{args.feature}_{args.model}.png', dpi=300)
    plt.show()
    plt.close()
else:
    ticks = np.unique(df['noise_percentage'].astype(float).values)

    for segment in np.unique(df['segment_time'].astype(float).values):
        legends = []
        fig, ax = plt.subplots()

        for coeff in np.unique(df['feature_coeff'].astype(float).values):
            plot_df = df[(df['segment_time'] == segment) & (df['feature_coeff'] == coeff)]
            
            # print(plot_df[['feature', 'segment_time', 'noise_percentage', 'feature_coeff','f1_micro', 'f1_macro']])

            ax.plot(ticks,plot_df['f1_micro'].values, linewidth=2)
            ax.plot(ticks,plot_df['f1_macro'].values, linewidth=2)

            legends.append(f'{coeff} - micro')
            legends.append(f'{coeff} - macro')

        plt.title(f'{args.feature.upper()} - {args.model.upper()} - {segment}s')
        ax.legend(legends)
        ax.xaxis.set_ticks(ticks, map(str, ticks))
        ax.yaxis.set_ticks(np.arange(0, 1.01, 0.25))
        ax.set_xlabel('Noise Percentage')
        ax.set_ylabel('F1 score')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{params["output_dir"]}/{args.feature}_{args.model}_{segment}s.png', dpi=300)
        # plt.show()
        plt.close()