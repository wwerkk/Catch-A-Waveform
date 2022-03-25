import os
import argparse

from numpy import rec
import torch
from utils import utils

def reconstruction_noise(params, output_folder:str):
    try:
        reconstruction_noise_list = torch.load((os.path.join(output_folder, 'reconstruction_noise_list.pt')), map_location=params.device)
    except FileNotFoundError:
        print(f'ISSUE: missing reconstruction noise .pt file in {output_folder}')
        reconstruction_noise_list = []
    return reconstruction_noise_list

def inspect_model(output_folder:str):
    print(f'loading {output_folder} log.txt')
    params = utils.params_from_log(os.path.join(output_folder, 'log.txt'))
    print('===RECONSTRUCTION NOISE LIST SEARCH===')
    reconstruction_noise_list = reconstruction_noise(params, output_folder)
    [print(n, f'\n{rn.shape[-1]} frames', f'\n\tnoise amplitude == {params.noise_amp_list[n] if n < len(params.noise_amp_list) else -1}') for n, rn in enumerate(reconstruction_noise_list)]
    print('===GENERATORS SEARCH===')
    generators_list = utils.generators_list_from_folder(params)
    if len(generators_list) > 0:
        for idx, generator in enumerate(generators_list):
            print(f'---GENERATOR {idx}---')
            [print(f'{n}->', gen[0]) for n, gen in enumerate(generator.named_modules())]
    else:
        print('ISSUE: missing generators from folder. no trained models found!')
    print('===SCALES TO TRAIN===')
    [print(n, f'{round(scale, 2)}x {params.fs_list[n]}Hz {"trained!" if n < len(generators_list) else ""}') for n, scale in enumerate(params.scales)]
    if len(reconstruction_noise_list) > len(generators_list):
        print('ISSUE: extra reconstruction noise tensors found. run the following to match the number of trained generators')
        print(f'\tpython model_editor.py --run_mode drop_scales --output_folder {output_folder} --max_scale {len(generators_list)}')
    elif len(reconstruction_noise_list) < len(generators_list):
        print(f'ISSUE: missing reconstruction noise tensors for {len(generator_list)} trained generators')
    else:
        print(f'found {len(generators_list)} generators and {len(reconstruction_noise_list)} reconstruction noise matches')
    
    if len(reconstruction_noise_list) > len(params.fs_list):
        print('ISSUE: extra reconstruction noise tensors found. run the following match the total number of frequency scales')
        print(f'\tpython model_editor.py --run_mode drop_scales --output_folder {output_folder} --max_scale {len(params.fs_list)}')
    
    if len(reconstruction_noise_list) != len(params.noise_amp_list):
        print(f'ISSUE: reconstruction noise does not match the {len(params.noise_amp_list)} noise amplitudes. run the following match.')
        print(f'\tpython model_editor.py --run_mode drop_scales --output_folder {output_folder} --max_scale {len(params.noise_amp_list)}')
    else:
        print(f'found {len(params.noise_amp_list)} noise amplitudes and {len(reconstruction_noise_list)} reconstruction noise matches')

def drop_scales(output_folder:str, max_scale:int):
    print('drop scales not implemented')

def drop_noise_amp(output_folder:str, idx:int):
    params = utils.params_from_log(os.path.join(output_folder, 'log.txt'))
    print(f'drop noise amp at {idx}. new list will contain {len(params.noise_amp_list)[-1]} amplitudes')

def drop_reconstruction_noise(output_folder:str, idx:int):
    print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', help='output directory with models and signals', type=str)
    parser.add_argument('--run_mode', default='inspect', type=str, choices=['inspect', 'repair', 'drop_scales'])
    parser.add_argument('--max_scale', default=-1, type=int)
    parsed = parser.parse_args()
    if parsed.run_mode == 'inspect':
        inspect_model(parsed.output_folder)
    elif parsed.run_mode == 'drop_scales':
        drop_scales(parsed.output_folder)
