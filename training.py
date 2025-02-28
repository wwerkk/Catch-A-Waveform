from torch import optim
import bitsandbytes as bnb
from utils.utils import *
from utils.mss_loss import multi_scale_spectrogram_loss
from models import CAW
from utils.plotters import *
import os
import random
import time
from datetime import datetime
import shutil


def train(params, signals_list):
    n_scales = len(params.scales)
    print(n_scales, params.scales)

    # load generators
    if params.run_mode == 'resume' or params.run_mode == 'transfer':
        generators_list = generators_list_from_folder(params)
        print(len(generators_list), 'generators')
        datestamp = datetime.now().date()
        shutil.copyfile(os.path.join(params.output_folder, 'reconstruction_noise_list.pt'), os.path.join(params.output_folder, f'reconstruction_noise_list_{datestamp}.pt'))
        shutil.copyfile(os.path.join(params.output_folder, 'log.txt'), os.path.join(params.output_folder, f'log_{datestamp}.txt'))
    else:
        generators_list = []
    # load reconstruction noise list and noise when resuming
    if params.run_mode == 'transfer':
        params.num_trained = min(params.num_trained, len(generators_list))
        reconstruction_noise_list = torch.load((os.path.join(params.output_folder, 'reconstruction_noise_list.pt')),
                                                   map_location=params.device)[:params.num_trained]
        noise_amp_list = params.noise_amp_list[:params.num_trained]

    if params.run_mode == 'resume':   
        params.num_trained = len(generators_list)
        reconstruction_noise_list = torch.load((os.path.join(params.output_folder, 'reconstruction_noise_list.pt')),
                                                   map_location=params.device)
        ###TODO create a function to create reconstruction noise for the correct n_sample size when draw_signal is called with it
        print('loaded reconstruction noise list:', [r.shape for r in reconstruction_noise_list])
        noise_amp_list = params.noise_amp_list 
        fs_list = params.fs_list
        print('noise amp list', noise_amp_list)
        print('resuming from last saved point', params.num_trained, 'are trained out of', len(fs_list))
    else:
        params.num_trained = 0
        fs_list = params.fs_list
        noise_amp_list = []
        reconstruction_noise_list = []
        
    if params.run_mode == 'inpainting':
        energy_list = [(sig[mask] ** 2).mean().item() for sig, mask in zip(signals_list, params.masks)]
    else:
        energy_list = [(sig ** 2).mean().item() for sig in signals_list]
    output_signals = []
    loss_vectors = []

    if params.manual_random_seed != -1:
        random.seed(params.manual_random_seed)
        torch.manual_seed(params.manual_random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    extra_epochs = 0

    for scale_idx in range(params.num_trained, n_scales):
        output_signals_single_scale, loss_vectors_single_scale, netG, reconstruction_noise_list, noise_amp = train_single_scale(
            params,
            signals_list,
            fs_list,
            generators_list,
            noise_amp_list,
            energy_list,
            reconstruction_noise_list,
            extra_epochs=extra_epochs)

        # Write fake sound
        fake_sound = output_signals_single_scale['fake_signal'].squeeze()
        filename = 'fake@%dHz.wav' % params.fs_list[scale_idx]
        write_signal(os.path.join(params.output_folder, filename), fake_sound,
                     params.fs_list[scale_idx], overwrite=False)

        # Write reconstructed sound
        reconstructed_sound = output_signals_single_scale['reconstructed_signal'].squeeze()
        filename = 'reconstructed@%dHz.wav' % params.fs_list[scale_idx]
        write_signal(os.path.join(params.output_folder, filename),
                     reconstructed_sound, params.fs_list[scale_idx], overwrite=False)
        torch.save(reconstruction_noise_list,
                   os.path.join(params.output_folder, 'reconstruction_noise_list.pt'))

        generators_list.append(netG)
        noise_amp_list.append(noise_amp)
        output_signals.append(output_signals_single_scale)
        loss_vectors.append(loss_vectors_single_scale)
        if params.run_mode != 'transfer':
            # train next scale more
            quarter_time = round(params.num_epochs * 0.25)
            if loss_vectors[-1]['v_rec_loss'][-1] > 0.03 and loss_vectors[-1]['v_rec_loss'][-quarter_time] - loss_vectors[-1]['v_rec_loss'][-1] > 0.01:
                print(f'high reconstruction loss detected: training an extra {quarter_time} epochs on the next scale')
                extra_epochs += quarter_time
            # train next scale less
            overfit_score = loss_vectors[-1]['v_rec_loss'][-quarter_time] - loss_vectors[-1]['v_rec_loss'][-1]
            elif overfit_score < 0.001:
                print(f'overfitting detected {overfit_score}: reducing training by {quarter_time} epochs on the next scale.')
                extra_epochs -= quarter_time


    return output_signals, loss_vectors, generators_list, noise_amp_list, energy_list, reconstruction_noise_list


def train_single_scale(params, signals_list, fs_list, generators_list, noise_amp_list, energy_list,
                       reconstruction_noise_list, extra_epochs=0):
    num_epochs_adjusted = params.num_epochs + extra_epochs 
    # Terminology: 0 is the higher scale (original signal, no downsampling). Higher scale means larger downsampling, e.g shorter signals
    n_scales = len(params.scales)
    if params.run_mode == 'transfer':
        current_scale = n_scales - len(noise_amp_list) - 1
    else:
        current_scale = n_scales - len(generators_list) - 1
    scale_idx = n_scales - current_scale - 1
    print(scale_idx, 'scale idx', current_scale, 'current scale')
    input_signal = signals_list[scale_idx].to(params.device)
    params.current_fs = fs_list[scale_idx]
    N = len(input_signal)

    if params.run_mode == 'inpainting':
        current_mask = params.masks[scale_idx]
        params.current_mask = current_mask
        params.current_holes = torch.Tensor([(int(idx[0] / params.Fs * params.current_fs), int(idx[1] / params.Fs * params.current_fs)) for idx in params.inpainting_indices]).to(params.device)

    # Create inputs
    real_signal = input_signal.reshape(1, 1, N)
    params.hidden_channels = params.hidden_channels_init if scale_idx == 0 else int(
        params.hidden_channels_init * params.growing_hidden_channels_factor)

    scale_num = n_scales - scale_idx - 1
    pad_size = calc_pad_size(params)
    signal_padder = nn.ConstantPad1d(pad_size, 0)

    # Initialize models
    netD = CAW.Discriminator(params).to(params.device)
    netD.apply(CAW.weights_init)
    netG = CAW.Generator(params).to(params.device)
    netG.apply(CAW.weights_init)
    receptive_field = calc_receptive_field(params.filter_size, params.dilation_factors, params.current_fs)
    receptive_field_percent = 100 * receptive_field / 1e3 / (N / params.current_fs)
    print('Signal in scale %d has %d samples, sample rate is %d[Hz].' % (
        scale_num, N, params.current_fs))
    print('Total receptive field is %d[msec] (%.1f%% of input).' % (receptive_field, receptive_field_percent))
    with open(os.path.join(params.output_folder, 'log.txt'), 'a') as f:
        f.write('*' * 30 + ' Scale ' + str(scale_num) + ' (' + str(params.current_fs) + ' [Hz]) ' + '*' * 30)
        f.write('\nreceptive_field = %d[msec] (%.1f%% of input)' % (receptive_field, receptive_field_percent))
        f.write('\nsignal_energy = %.4f' % energy_list[scale_idx])
    
    # get reconstruction noise for this scale
    if scale_idx == 0:
        reconstruction_noise = get_noise(params, real_signal.shape)
    else:
        reconstruction_noise = torch.zeros(real_signal.shape, device=params.device)
        if params.run_mode == 'inpainting':
            reconstruction_noise[:, :, torch.logical_not(current_mask)] = get_noise(params, torch.nonzero(
                torch.logical_not(current_mask)).shape[0]).expand(1, 1, -1).to(params.device)
    reconstruction_noise = signal_padder(reconstruction_noise)

    # load state dictionary
    if scale_idx > 1:
        netG.load_state_dict(
            torch.load('%s/netGScale%d.pth' % (params.output_folder, scale_idx - 1), map_location=params.device))
        netD.load_state_dict(
            torch.load('%s/netDScale%d.pth' % (params.output_folder, scale_idx - 1), map_location=params.device))

    output_folder = params.output_folder
    # create training scheduler milestones
    params.scheduler_milestones = [int(num_epochs_adjusted * (2 / 3))]
    # Create optimizers
    if params.ttur: # two time update rule (independent learning rates for discriminator and generator)
        lr_d = params.learning_rate_d
        lr_g = params.learning_rate_g
    else:
        lr_d = params.learning_rate
        lr_g = params.learning_rate
    if params.lite: # 8bit optimizer
        optimizerD = bnb.optim.Adam8bit(netD.parameters(), lr=lr_d, betas=(params.beta1, 0.999)) 
        optimizerG = bnb.optim.Adam8bit(netG.parameters(), lr=lr_g, betas=(params.beta1, 0.999))
    else: # default optimizer
        optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(params.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(params.beta1, 0.999))
    if params.run_mode != 'transfer': # normaltraining modes
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=params.scheduler_milestones,
                                                        gamma=params.scheduler_lr_decay)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=params.scheduler_milestones,
                                                        gamma=params.scheduler_lr_decay)
    else: # experimental transfer learning
        schedulerD = torch.optim.lr_scheduler.StepLR(optimizer=optimizerD, step_size=10,
                                                        gamma=params.scheduler_lr_decay)
        schedulerG = torch.optim.lr_scheduler.StepLR(optimizer=optimizerG, step_size=10,
                                                        gamma=params.scheduler_lr_decay)

    # Initialize error vectors
    v_err_real = np.zeros(num_epochs_adjusted, )
    v_err_fake = np.zeros(num_epochs_adjusted, )
    v_gp = np.zeros(num_epochs_adjusted, )
    v_rec_loss = np.zeros(num_epochs_adjusted, )

    epochs_start_time = time.time()
    # prepare inputs for gradient penalty
    if not params.run_mode == 'inpainting':
        D_out_shape = torch.Size((1, 1, N - 2 * pad_size))
        _grad_outputs = torch.ones(D_out_shape, device=params.device)
    grad_pen_alpha_vec = torch.rand(num_epochs_adjusted).to(params.device)

    inputs_lengths = params.inputs_lengths
    for epoch_num in range(num_epochs_adjusted):
        if params.run_mode == 'transfer':
            pmod = 10
        else:
            pmod = 100
        print_progress = epoch_num % pmod == 0
        # Create noise
        noise_signal = get_noise(params, real_signal.shape)
        noise_signal = signal_padder(noise_signal)
        #################################################################
        # Optimize D by maximizing D(realSignal)+(1-D(G(noise_signal))) #
        #################################################################
        netD.zero_grad()
        # Run on real signal
        if params.run_mode == 'inpainting':
            out_D_real = netD(real_signal, use_mask=True)
            tot_samples = out_D_real.shape[2]
            params.not_valid_idx_start = [int(idx[0] - receptive_field / 1e3 * params.current_fs + 1) for idx in params.current_holes]
            params.not_valid_idx_end = [int(idx[1] + 1) for idx in params.current_holes]  # +1 is because of pe filter
            out_D_real_cp = out_D_real.clone()
            out_D_real = out_D_real_cp[:, :, :params.not_valid_idx_start[0]]
            if len(params.current_holes) > 1:
                for i in range(len(params.current_holes) - 1):
                    out_D_real = torch.cat((out_D_real, out_D_real_cp[:, :, params.not_valid_idx_end[i] + 1:params.not_valid_idx_start[i+1]]), dim=2)
            out_D_real = torch.cat((out_D_real, out_D_real_cp[:, :, params.not_valid_idx_end[-1] + 1:]), dim=2)
            mask_ratio = tot_samples / out_D_real.shape[2]
        else:
            mask_ratio = 1
            out_D_real = netD(real_signal)
        err_real_D = -out_D_real.mean()
        err_real_D.backward(retain_graph=True)
        err_real_D = err_real_D.detach()
        if print_progress or params.plot_losses:
            err_real_D_val = err_real_D.item()

        if epoch_num == 0:
            if params.run_mode == 'inpainting':
                D_out_shape = out_D_real.shape
                _grad_outputs = torch.ones(D_out_shape, device=params.device)
            if scale_idx == 0:  # We are at coarsest scale
                prev_signal = torch.full(noise_signal.shape, 0, device=params.device, dtype=noise_signal.dtype)
                prev_reconstructed_signal = torch.zeros(reconstruction_noise.shape, device=params.device)
                noise_amp = params.initial_noise_amp
            else:
                prev_signal = draw_signal(params, generators_list, inputs_lengths, fs_list, noise_amp_list)
                prev_signal = signal_padder(prev_signal)
                prev_reconstructed_signal = draw_signal(params, generators_list, inputs_lengths,
                                                        fs_list,
                                                        noise_amp_list,
                                                        reconstruction_noise_list)
                prev_reconstructed_signal = signal_padder(prev_reconstructed_signal)
                innovation = energy_list[scale_idx] - energy_list[scale_idx - 1]
                energy_diff = torch.sqrt(torch.Tensor([innovation])).to(params.device)
                noise_amp = params.noise_amp_factor * max(torch.Tensor([0]).to(params.device),
                                                          energy_diff)

            if scale_idx == 1 and params.add_cond_noise:
                noise_amp = prev_reconstructed_signal.std()

            with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
                f.write('\nnoise_amp: %.6f' % noise_amp)

            reconstruction_noise = reconstruction_noise * noise_amp
            reconstruction_noise_list.append(reconstruction_noise)
        else:
            if scale_idx > 0:
                prev_signal = draw_signal(params, generators_list, inputs_lengths, fs_list, noise_amp_list)
                prev_signal = signal_padder(prev_signal)

        input_noise = noise_signal * noise_amp

        # Run on fake signal
        fake_signal = netG((input_noise + prev_signal).detach(), prev_signal)
        out_D_fake = netD(fake_signal.detach())
        err_fake_D = out_D_fake.mean()
        del out_D_real, out_D_fake
        err_fake_D.backward(retain_graph=True)
        err_fake_D = err_fake_D.detach()
        if print_progress or params.plot_losses:
            err_fake_D_val = err_fake_D.item()

        gradient_penalty = calc_gradient_penalty(params, netD, real_signal, fake_signal, params.lambda_grad,
                                                 grad_pen_alpha_vec[epoch_num], _grad_outputs, mask_ratio)
        gradient_penalty.backward()
        if print_progress or params.plot_losses:
            gradient_penalty_val = gradient_penalty.item()
        del gradient_penalty

        optimizerD.step()

        if params.plot_losses:
            v_err_real[epoch_num] = err_real_D_val
            v_err_fake[epoch_num] = err_fake_D_val
            v_gp[epoch_num] = gradient_penalty_val

        #############################################
        # Update G by maximizing D(G(noise_signal)) #
        #############################################
        netG.zero_grad()
        output = netD(fake_signal)
        errG = -output.mean()
        del output
        errG.backward(retain_graph=True)
        errG = errG.detach()
        if print_progress or params.plot_losses:
            errG_val = errG.item()
        if scale_idx == 0:
            reconstructed_signal = netG((reconstruction_noise + prev_reconstructed_signal).detach(),
                                        prev_reconstructed_signal)
        else:
            reconstructed_signal = netG((reconstruction_noise + prev_reconstructed_signal).detach(),
                                        prev_reconstructed_signal)
        if params.alpha1 > 0:
            if params.run_mode == 'inpainting':
                rec_loss_t = params.alpha1 * torch.mean(
                    (real_signal[:, :, current_mask] - reconstructed_signal[:, :, current_mask]) ** 2)
            else:
                rec_loss_t = params.alpha1 * torch.mean((real_signal - reconstructed_signal) ** 2)
        else:
            rec_loss_t = 0
        if params.alpha2 > 0:
            rec_loss_f = params.alpha2 * multi_scale_spectrogram_loss(params, real_signal.permute(0, 2, 1),
                                                                      reconstructed_signal.permute(0, 2, 1))
        else:
            rec_loss_f = 0
        rec_loss = rec_loss_t + rec_loss_f
        rec_loss.backward(retain_graph=True)
        rec_loss = rec_loss.detach()
        if params.alpha1 > 0:
            rec_loss_t = rec_loss_t.detach()
        if params.alpha2 > 0:
            rec_loss_f = rec_loss_f.detach()
        if print_progress or params.plot_losses:
            rec_loss_val = rec_loss.item()

        optimizerG.step()

        # visualize
        if params.plot_losses:
            v_rec_loss[epoch_num] = rec_loss_val

        # display and save results
        if print_progress:
            print('[%d/%d] D(real): %.2f. D(fake): %.2f. rec_loss: %.4f. gp: %.4f ' % (
                epoch_num, num_epochs_adjusted, -err_real_D_val, err_fake_D_val, rec_loss_val, gradient_penalty_val))
            filename = f'fake@{params.fs_list[scale_idx]}Hz{epoch_num}Ep.wav'
            write_signal(os.path.join(params.output_folder, filename), fake_signal,
                     params.fs_list[scale_idx], overwrite=False)

        # increment epoch scheduler for milestone
        schedulerD.step()
        schedulerG.step()

        # Some memory cleanup
        fake_signal = fake_signal.detach()
        reconstructed_signal = reconstructed_signal.detach()
        if epoch_num < num_epochs_adjusted - 1:
            del fake_signal, reconstructed_signal, rec_loss, rec_loss_t, rec_loss_f
        del noise_signal, input_noise
        if scale_idx > 0:
            del prev_signal

    epochs_stop_time = time.time()
    runtime_msg = 'Total time in scale %d: %d[sec] (%.2f[sec]/epoch on avg.). D(real): %f, D(fake): %f, rec_loss: %.4f. gp: %.4f' % (
        current_scale, epochs_stop_time - epochs_start_time,
        round((epochs_stop_time - epochs_start_time) / (num_epochs_adjusted + 0.0001)),
        -err_real_D_val, err_fake_D_val, rec_loss_val, gradient_penalty_val)
    print(runtime_msg)
    with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
        f.write('\n%s\n' % runtime_msg)

    # Save this scale models
    torch.save(netG.state_dict(), '%s/netGScale%d.pth' % (params.output_folder, scale_idx))
    torch.save(netD.state_dict(), '%s/netDScale%d.pth' % (params.output_folder, scale_idx))
    # Pack outputs
    if params.plot_losses:
        loss_vectors = {'v_err_real': v_err_real,
                        'v_err_fake': v_err_fake,
                        'v_rec_loss': v_rec_loss,
                        'v_gp': v_gp}
    else:
        loss_vectors = []
    fake_signal = fake_signal.detach().cpu().numpy()[:, 0, :]
    reconstructed_signal = reconstructed_signal.detach().cpu().numpy()[:, 0, :]
    output_signals = {'fake_signal': fake_signal, 'reconstructed_signal': reconstructed_signal}
    del fake_signal, real_signal, netD, _grad_outputs, grad_pen_alpha_vec, input_signal, reconstructed_signal, prev_reconstructed_signal, reconstruction_noise
    netG = reset_grads(netG, False)
    netG.eval()
    if params.is_cuda:
        torch.cuda.empty_cache()
    print('*' * 30 + ' Finished working on scale ' + str(current_scale) + ' ' + '*' * 30)
    return output_signals, loss_vectors, netG, reconstruction_noise_list, noise_amp
