import generating
import librosa, os

def test_audio_generator_condition():
    condition_file = 'inputs/portraitxo-wire.wav'
    audio_generator = generating.AudioGenerator('outputs/portraitxo-wire_2')
    assert isinstance(audio_generator, generating.AudioGenerator)
    for n, fs in enumerate(audio_generator.params.fs_list):
        condition_signal, _ = librosa.load(os.path.join(audio_generator.output_folder, 'real@%dHz.wav' % fs), sr=None, duration=10)
        condition = {'condition_signal': condition_signal, 'name': f'test_self{fs}', 'condition_fs': _}
        print(condition)
        generated_signal = audio_generator.condition(condition, write=False)
        print(generated_signal.shape)
        assert generated_signal.shape[0] == 1