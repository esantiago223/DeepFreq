import os
import argparse
import torch
import numpy as np
from data import noise
from data.data import gen_signal
import json



def my_generate_dataset(output_dir=None, overwrite=True, n_test = 1000, signal_dimension=50,minimum_separation=1,max_freq=10,
                        distance='normal',amplitude='normal_floor',floor_amplitude=0.1,dB = [0,5,10,15,20,25,30,35,40,45,50],
                        numpy_seed=105,torch_seed=94):
    

    if os.path.exists(output_dir) and os.listdir(output_dir) and not overwrite:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite to overcome.".format(output_dir))
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    inputArgs = ('output_dir','overwrite','n_test','signal_dimension','minimum_separation','max_freq',
                 'distance','amplitude','floor_amplitude','dB','numpy_seed','torch_seed')
    inputDict = dict.fromkeys(inputArgs)
    inputDict['output_dir']   = output_dir
    inputDict['overwrite']    = overwrite
    inputDict['n_test']       = n_test
    inputDict['signal_dimension'] = signal_dimension
    inputDict['minimum_separation'] = minimum_separation
    inputDict['max_freq']           = max_freq
    inputDict['distance']           = distance
    inputDict['amplitude']          = amplitude
    inputDict['floor_amplitude']    = floor_amplitude
    inputDict['dB']                 = [str(x) for x in dB]
    inputDict['numpy_seed']         = numpy_seed
    inputDict['torch_seed']         = torch_seed
    
    with open(os.path.join(output_dir, 'data.args'), 'w') as f:
        json.dump(inputDict, f, indent=2)

    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)

    s, f, nfreq = gen_signal(
        num_samples=n_test,
        signal_dim=signal_dimension,
        num_freq=max_freq,
        min_sep=minimum_separation,
        distance=distance,
        amplitude=amplitude,
        floor_amplitude=floor_amplitude,
        variable_num_freq=True)

    np.save(os.path.join(output_dir, 'infdB'), s)
    np.save(os.path.join(output_dir, 'f'), f)

    eval_snrs = [np.exp(np.log(10) * float(x) / 10) for x in dB]

    for k, snr in enumerate(eval_snrs):
        noisy_signals = noise.noise_torch(torch.tensor(s), snr, 'gaussian').cpu()
        np.save(os.path.join(output_dir, '{}dB'.format(float(dB[k]))), noisy_signals)

if __name__ == '__main__':
    my_generate_dataset(output_dir='myInputDirTest',overwrite=True)