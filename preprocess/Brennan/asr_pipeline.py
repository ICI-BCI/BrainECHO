import soundfile as sf
import os
import json
# import whisper
import tqdm
import librosa
import sys
import numpy as np
import re

# Get the file path of the current script
current_path = os.path.abspath(__file__)
# Get the path to the project root directory
project_root = os.path.dirname(os.path.dirname(current_path))
# Add the project root directory to sys.path
sys.path.append(project_root)

import argparse


def makedirs(output_dir):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    return output_dir


# python process_dataset/asr_pipeline.py --input_dir="datasets/gwilliams2023/download/stimuli/audio"
# --output_dir="datasets/gwilliams2023/preprocess6/audio"

if __name__ == '__main__':
    home_dir = r'/home/szx/dataset/Brennan'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=str, default=None, help="需要设置的存放音频文件的路径")
    parser.add_argument("--type", type=str, default='wav', help="音频文件的后缀")
    parser.add_argument("--output_dir", type=str, default=None, help="输出的音频和转写文本的位置")
    parser.add_argument("--force_rewrite", type=bool, default=True, help="强制重写所有音频文件到新文件夹")
    parser.add_argument("--language", type=str, default=None, help="设置语言")
    parser.add_argument("--backend", type=str, default='whisperx', help="使用的模型")
    args = parser.parse_args()

    wav_dir = args.input_dir
    wav16_dir = os.path.join(home_dir, args.output_dir, 'wav16')
    transcription_dir = os.path.join(home_dir, args.output_dir, 'transcription')
    wav_files = [i for i in os.listdir(wav_dir) if i.split('.')[-1] == args.type]
    wav_files = sorted(wav_files, key=lambda l: int(re.findall('\d+', l)[0]))
    print(wav_files)
    assert len(wav_files) != 0, 'there is no file under the input dir!'
    target_sr = 16000
    language = args.language
    if args.backend == 'whisper':
        # model = whisper.load_model('large')
        pass
    elif args.backend == 'whisperx':
        import whisperx

        batch_size = 16
        device = 'cuda'
        compute_type = 'float16'
        # path to WhisperX model
        model = whisperx.load_model(
            r"D:\research\eeg2text_own_dataset\models\torch\models--Systran--faster-whisper-large-v2", 'cuda',
            compute_type=compute_type)
    else:
        raise NotImplementedError

    # Transcribe from audio to text. You also need to convert the audio to 16kHz and write it into a json file
    all_wav = []
    wav_sr = 16000
    for wav_path in tqdm.tqdm(wav_files):
        wav_path = os.path.join(wav_dir, wav_path)
        wav_name = os.path.basename(wav_path).split('.')[0]
        wav, wav_sr = sf.read(wav_path, always_2d=True)
        wav = wav[:, 0]
        all_wav.append(wav)

    wav = np.concatenate(all_wav)

    if wav_sr != 16000 or args.force_rewrite:
        wav = librosa.resample(wav, orig_sr=wav_sr, target_sr=target_sr)
        print(len(wav))
        wav16_path = os.path.join(wav16_dir, f'DownTheRabbitHoleFinal_SoundFile_16kHz.wav')
        sf.write(makedirs(wav16_path), wav, samplerate=target_sr)
        wav_path = wav16_path

    if args.backend == 'whisper':
        result = model.transcribe(
            wav_path, language=language, word_timestamps=True,
            without_timestamps=False)
    elif args.backend == 'whisperx':
        audio = whisperx.load_audio(wav_path)
        result = model.transcribe(audio, batch_size=batch_size, chunk_size=12)
        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result['language'], device=device)
        segments_result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                                         return_char_alignments=False)
        for k in ['segments', 'word_segments']:
            result[k] = segments_result[k]
    else:
        raise NotImplementedError

    # write text
    transcribe_path = f"{transcription_dir}/{os.path.basename(wav_path).split('.')[0]}.json"
    transcribe_path = makedirs(transcribe_path)
    with open(transcribe_path, 'w', encoding='utf-8') as write_f:
        json.dump(result, write_f, indent=4, ensure_ascii=False)

    lines = []
    with open(transcribe_path, 'r', encoding='utf-8') as f:
        for line in f:
            if r'--\"' in line:
                lines.append(line.replace(r'--\"', ' '))
            elif r"\"'" in line:
                lines.append(line.replace(r"\"'", ''))
            elif r'\"' in line:
                lines.append(line.replace(r'\"', ''))
            else:
                lines.append(line)
    with open(transcribe_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
