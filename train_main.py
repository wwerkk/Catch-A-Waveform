from utils.utils import *
import glob
from params import Params
from training import train
#import training_batch.train
from utils.plotters import *
import time
from datetime import datetime
import argparse
from generating import AudioGenerator
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', help='GPU id to use', default=0, type=int)
    parser.add_argument('--input_file', help='Path to input file', default='trump_farewell_address_8.wav')
    parser.add_argument('--start_time', help='Skip beginning, in [sec]', default=0, type=float)
    parser.add_argument('--max_length', help='Max length of signal, in [sec]', default=25, type=float)
    parser.add_argument('--segments_to_train', default=[], type=float, nargs='+',
                        help='Train on several segments of input signal, please provide segements in the form: start1, end1, start2, end2,... in [sec]')
    parser.add_argument('--init_sample_rate', help='Resample input to a given sample rate', default=40000, type=int)
    parser.add_argument('--num_epochs', help='Number of training epoches in each scale', default=4000, type=int)
    parser.add_argument('--num_layers', help='Number of layers in each model', default=8, type=int)
    parser.add_argument('--speech', default=False, action='store_true')
    parser.add_argument('--run_mode', default='normal', type=str, choices=['normal', 'inpainting', 'denoising', 'resume', 'transfer'])
    parser.add_argument('--inpainting_indices', default=[0, 1], nargs='+', type=int,
                        help='Start and end indices of hole (for inpainting)')
    parser.add_argument('--plot_losses', help='Save and plot GAN losses', default=True, action='store_true')
    parser.add_argument('--plot_signals', help='Plot signals', default=False, action='store_true')
    parser.add_argument('--output_folder', help='output directory with models and signals', type=str)
    parser.add_argument('--scale_crop', help='crop the signal to use a fixed frame at each scale - used for fitting high sample rates in memory', default=False, action='store_true')
    parser.add_argument('--lite', help='use a precision reduced (8-bit) version of adam optimizers; reduces the memory load of back prop', default=False, action='store_true')
    parser.add_argument('--skip_connections', help='flag to add residual connections between conv blocks', default=False, action='store_true')
    parser.add_argument('--filter_size', help='size of convolution kernel', default=9, type=int)
    parser.add_argument('--hidden_channels_init', help='number of filters to output in initial 1D convolution layers', default=16, type=int)
    parser.add_argument('--ttur', help='use a "Two Time-Scale Update Rule"; one for generators and one for discriminators; learning_rate_g and learning_rate_d', default=False, action='store_true')
    parser.add_argument('--num_trained', help='scale number to start training from', default=0, type=int)
    parser.add_argument('--learn_milestones', help='Save and plot GAN losses', default=False, action='store_true')
    params_parsed = parser.parse_args()
    if params_parsed.run_mode == 'resume' or params_parsed.run_mode == 'transfer':
        if not params_parsed.output_folder:
            print("ERROR: must define output folder of trained model")
        else:
            params_parsed.output_folder = os.path.join('outputs', params_parsed.output_folder) 

startTime = time.time()
params = Params()
if params_parsed.run_mode == "resume" or params_parsed.run_mode == 'transfer':
    log_file = os.path.join(params_parsed.output_folder,'log.txt')
    if os.path.exists(log_file):
        print(f"Warning: the '{params_parsed.run_mode}' --run_mode will overwrite parameters with those found in log.txt")
        params_logged = params_from_log(log_file)
        params = override_params(params, params_logged)
    else:
        print(f'Warning: unable to load params from previous training run. {log_file} does not exist')

if params.run_mode == 'inpaininting' and len(params.inpainting_indices)%2 != 0:
    raise Exception('Provide START and END indices of each hole!')

params = override_params(params, params_parsed)
params.Fs = params.init_sample_rate
#if params.run_mode == 'transfer':
#    params.learning_rate == 0.0001
if params.is_cuda:
    torch.cuda.set_device(params.gpu_num)
    params.device = torch.device("cuda:%d" % params.gpu_num)

if params.manual_random_seed != -1:
    random.seed(params.manual_random_seed)
    torch.manual_seed(params.manual_random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Get input signal
samples = get_input_signal(params)
# set scales
params.fs_list = [f for f in params.fs_list if f <= params.Fs]
if params.fs_list[-1] != params.Fs:
    params.fs_list.append(params.Fs)
params.scales = [params.Fs / f for f in params.fs_list]

print('Working on file: %s' % params.input_file)

# Create a random hole for inpainting
if params.run_mode == 'inpainting':
    samples_orig = samples.copy()
    params.inpainting_indices = list(zip(params.inpainting_indices[0::2], params.inpainting_indices[1::2]))
    for hole_idx in params.inpainting_indices:
        samples[hole_idx[0]:hole_idx[1]] = 0

# Set params by run_node and signal type
params.scheduler_milestones = [int(params.num_epochs * 2 / 3)]
if params.speech:
    params.alpha1 = 10
    params.alpha2 = 0
    params.add_cond_noise = False
else: # not speech
    if params.run_mode == 'normal' or params.run_mode == 'resume':
        params.alpha1 = 0
        params.alpha2 = 1e-4
        params.add_cond_noise = True
    elif params.run_mode == 'transfer':
        params.alpha1 = 1
        params.alpha2 = 1e-5
        params.add_cond_noise = True
    else: # not resuming/transfering normal or speech training modes
        params.alpha1 = 10
        params.alpha2 = 0
        if params.run_mode == 'inpainting':
            params.add_cond_noise = True
        elif params.run_mode == 'denoising': # denoising 
            params.add_cond_noise = False
        else: # unknown non speech mode
            print('warning: unknown run_mode - there could be unexpected training behavior!')
params.dilation_factors = [2 ** i for i in range(params.num_layers)]

# make directory for new project
if params.run_mode == 'normal' or params.run_mode == 'inpainting' or params.run_mode == 'denoising':
    # Create output folder
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    if os.path.exists(params.output_folder):
        dirs = glob.glob(params.output_folder + '*')
        params.output_folder = params.output_folder + '_' + str(len(dirs) + 1)
    os.mkdir(params.output_folder)
    print('Writing results to %s\n' % params.output_folder)

if params.run_mode == 'inpainting':
    write_signal(os.path.join(params.output_folder, 'Original.wav'), samples_orig, params.Fs)

# samples = samples.reshape((1, -1))

# Create input signal for each scale
###TODO allow to continue with new signal sizes in failed layers
signals_list, fs_list = create_input_signals(params, torch.tensor(samples), params.Fs)
print(fs_list,'== fs_list\n', len(signals_list), '== len signals list')
if len(signals_list) == 0:
    params.set_first_scale_by_energy = False
    params.scales = params.scales[2:]  # Manually start from 500
    signals_list, fs_list = create_input_signals(params, torch.tensor(samples), params.Fs)
params.scales = [params.Fs / f for f in fs_list]
params.fs_list = fs_list
params.inputs_lengths = [len(s) for s in signals_list]

# Write parameters of run to a text file
with open(os.path.join(params.output_folder, 'log.txt'), 'w') as f:
    f.write(''.join(["%s = %s\n" % (k, v) for k, v in params.__dict__.items()]))

if params.run_mode == 'inpainting':
    # create masks for inpainting
    params.masks = []
    for scale, real_signal in zip(params.scales, signals_list):
        idcs = np.array(range(len(real_signal)))
        total_mask = np.ones(len(real_signal), dtype=bool)
        for hole_idx in params.inpainting_indices:
            cur_hole_start_idx = int(hole_idx[0] / scale)
            cur_hole_end_idx = int(hole_idx[1] / scale)
            current_mask = np.logical_or(idcs < cur_hole_start_idx, idcs >= cur_hole_end_idx)
            total_mask = np.logical_and(current_mask, total_mask)
        params.masks.append(torch.Tensor(total_mask).bool().to(params.device))

print('Running on ' + str(params.device))
# Start training
output_signals, loss_vectors, generators_list, noise_amp_list, energy_list, reconstruction_noise_list = train(
    params, signals_list)

# Save reconstruction noise list
torch.save(reconstruction_noise_list, os.path.join(params.output_folder, 'reconstruction_noise_list.pt'))

with open(os.path.join(params.output_folder, 'log.txt'), 'a') as f:
    f.write('\nTotal Runtime is: %d minutes' % ((time.time() - startTime) / 60))
    f.write('\n Finished running in : %s' % datetime.fromtimestamp(time.time()))

##############
# Generating #
##############
audio_generator = AudioGenerator(params, generators_list, noise_amp_list,
                                 reconstruction_noise_list=reconstruction_noise_list)
if not params.run_mode == 'inpainting':
    audio_generator.generate()
    audio_generator.reconstruct()
else:
    audio_generator.inpaint()

#################
# Plotting Area #
#################
# Plot Signals
if params.plot_signals:
    os.mkdir(os.path.join(params.output_folder, 'figures'))
    for real_signal, outputs, fs in zip(signals_list, output_signals, params.fs_list):
        output_file(os.path.join(params.output_folder, 'figures', '%dHz' % fs))
        plot_signal_time_freq(real_signal, outputs['reconstructed_signal'], outputs['fake_signal'], Fs=fs,
                              labels=['Real Signal', 'Reconstructed Signal', 'Fake Signal'])
# Plot losses
if params.plot_losses:
    if not os.path.exists(os.path.join(params.output_folder, 'figures')):
        os.mkdir(os.path.join(params.output_folder, 'figures'))
    plot_losses(params, loss_vectors)
