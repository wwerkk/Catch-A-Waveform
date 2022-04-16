import argparse
import os
import librosa
import numpy as np

from generating import AudioGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='Folder of trained model', type=str, required=True)
    parser.add_argument('--n_signals', help='Number of signals to generate', type=int, default=1)
    parser.add_argument('--length', help='Length of signals to generate', type=float, default=30)
    parser.add_argument('--generate_all_scales', help='Write signals of all scales', default=False, action='store_true')
    parser.add_argument('--condition',
                        help='Condition the generated signals on the lowest scale of input, to enforce general structure',
                        default=False, action='store_true')
    parser.add_argument('--condition_file', help='file to use to condition the generation of the selected model', type=str, default=None)
    parser.add_argument('--reconstruct', help='Generate the reconstruction of the signal', default=False, action='store_true')
    parser.add_argument('--condition_fs', help='the sample rate to insert the conditioning from the condition_file', type=int, default=None)
    parser.add_argument('--infinite', help='Generate infinite wav signals in chunks', default=False, action='store_true')
    args = parser.parse_args()

    audio_generator = AudioGenerator(os.path.join('outputs', args.input_folder))
    if args.condition_file:
        args.condition = True
    if args.condition:
        if args.condition_fs:
            condition_scale = np.where(np.array(audio_generator.params.fs_list) <= args.condition_fs)[0][-1] + 1
        else:
            condition_scale = 0
        condition_fs = audio_generator.params.fs_list[condition_scale]
        print(condition_scale, condition_fs)
        if args.condition_file:
            print('conditioning with custom file', args.condition_file + '.wav')
            condition_file = f'{args.condition_file}{condition_fs}'.replace('/', '-')
            print(f'conditioning at scale index {condition_scale}: {condition_fs}Hz')
            condition_signal, _ = librosa.load(
                os.path.join('inputs/', args.condition_file + '.wav'), sr=condition_fs,  duration=args.length)
            norm_factor = max(abs(condition_signal.reshape(-1)))
            print(norm_factor, 'norm factor', len(condition_signal), 'condition length')
            condition_signal = condition_signal / norm_factor
        else:
            condition_file = 'real@%dHz.wav' % condition_fs
            condition_signal, _ = librosa.load(
                os.path.join(audio_generator.output_folder, condition_file), sr=condition_fs, duration=args.length)
        condition = {'condition_signal': condition_signal, 'name': condition_file, 'condition_fs': condition_fs}
        audio_generator.condition(condition)

    elif args.reconstruct:
        audio_generator.reconstruct()
    elif args.infinite:
        audio_generator.infinite(window_length=args.length)
    else:
        audio_generator.generate(nSignals=args.n_signals, length=args.length,
                                     generate_all_scales=args.generate_all_scales)
