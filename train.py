import os
import subprocess
import soundfile as sf
import pyloudnorm as pyln
import glob
from pathlib import Path
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.trainers import Trainer, TrainerArgs
from TTS.tts.utils.generic_utils import load_config

# Set the necessary paths and variables
ds_name = "vits-ds"
output_directory = "traineroutput"
upload_dir = "dataset"
MODEL_FILE = "/home/ubuntu/.local/share/tts/tts_models--en--ljspeech--vits/model_file.pth"
upload_dir = "/home/ubuntu/ml_train/" + upload_dir
RUN_NAME = "VITS-en"
OUT_PATH = "/home/ubuntu/ml_train/" + ds_name + "/traineroutput/"

# Create necessary directories
os.makedirs(upload_dir, exist_ok=True)
os.makedirs("/home/ubuntu/ml_train/" + ds_name, exist_ok=True)
os.makedirs("/home/ubuntu/ml_train/" + ds_name + "/wavs/", exist_ok=True)

run_type = "restore"
print(run_type + " run selected")

run_denoise = True
run_splits = True
use_audio_filter = True
normalize_audio = True

# Convert and resample uploaded mp3/wav clips to 1 channel, 22kHz
mp3_files = glob.glob(os.path.join(upload_dir, '*.mp3'))
wav_files = glob.glob(os.path.join(upload_dir, '*.wav'))
audio_files = mp3_files + wav_files

for file_path in audio_files:
    base_name = os.path.basename(file_path)
    output_path = os.path.join(upload_dir, "kch", os.path.splitext(base_name)[0] + ".wav")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", file_path, "-acodec", "pcm_s16le", "-ar",
                    "22050", "-ac", "1", output_path])

print("Files converted to 22kHz 1ch wav")

if run_denoise:
    print("Running denoise...")
    orig_wavs = os.path.join(upload_dir, "kch")

    from pathlib import Path

    rnn = "/home/ubuntu/ml_train/rnnoise/examples/rnnoise_demo"
    paths = glob.glob(os.path.join(orig_wavs, '*.wav'))

    for file_path in paths:
        base = os.path.basename(file_path)
        tp_s = os.path.join(upload_dir, "kch", "denoise")
        tf_s = os.path.join(upload_dir, "kch", "denoise", base)
        target_path = Path(tp_s)
        target_file = Path(tf_s)
        print("From:", file_path)
        print("To:", target_file)

        subprocess.run(["sox", "-G", "-v", "0.8", file_path, "ft.wav", "remix", "-", "rate", "48000"])
        subprocess.run(["sox", "ft.wav", "-c", "1", "-r", "48000", "-b", "16", "-e", "signed-integer", "-t", "raw",
                        "temp.raw"])  # convert wav to raw
        subprocess.run([rnn, "temp.raw", "rnn.raw"])  # apply rnnoise
        subprocess.run(["sox", "-G", "-v", "0.8", "-r", "ft", "-b", "16", "-e", "signed-integer", "rnn.raw", "-t",
                        "wav", "rnn.wav"])  # convert raw back to wav

        subprocess.run(["mkdir", "-p", str(target_path)])
        if use_audio_filter:
            print("Running highpass/lowpass & resample")
            subprocess.run(["sox", "rnn.wav", str(target_file), "remix", "-", "highpass", "50", "lowpass", "8000",
                            "rate", "22050"])
            # apply high/low pass filter and change sr to 22050Hz
            data, rate = sf.read(target_file)
        else:
            print("Running resample without filter")
            subprocess.run(["sox", "rnn.wav", str(target_file), "remix", "-", "rate", "22050"])
            # apply high/low pass filter and change sr to 22050Hz
            data, rate = sf.read(target_file)

        if normalize_audio:
            print("Output normalized")
            peak_normalized_audio = pyln.normalize.peak(data, -6.0)

            # measure the loudness first
            meter = pyln.Meter(rate)  # create BS.1770 meter
            loudness = meter.integrated_loudness(data)

            # loudness normalize audio to -25 dB LUFS
            loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -25.0)
            sf.write(target_file, data=loudness_normalized_audio, samplerate=22050)
            print("")
        else:
            print("File written without normalizing")
            sf.write(target_file, data=data, samplerate=22050)
            print("")

    for file_path in [target_file, "/home/ubuntu/ml_train/rnnoise/examples/ft.wav"]:
        if os.path.exists(file_path):
            os.remove(file_path)

elif not run_denoise:
    paths = glob.glob(os.path.join(orig_wavs, '*.wav'))
    for file_path in paths:
        print("Skipping denoise...")
        base = os.path.basename(file_path)
        tp_s = os.path.join(upload_dir, "kch", "denoise")
        tf_s = os.path.join(upload_dir, "kch", "denoise", base)
        target_path = Path(tp_s)
        target_file = Path(tf_s)
        print("From:", file_path)
        print("To:", target_file)
        subprocess.run(["sox", "-G", "-v", "0.8", file_path, "ft.wav", "remix", "-", "rate", "48000"])
        subprocess.run(["sox", "ft.wav", "-c", "1", "-r", "48000", "-b", "16", "-e", "signed-integer", "-t", "raw",
                        "temp.raw"])  # convert wav to raw
        subprocess.run(["sox", "-G", "-v", "0.8", "-r", "ft", "-b", "16", "-e", "signed-integer", "rnn.raw", "-t",
                        "wav", "rnn.wav"])  # convert raw back to wav
        subprocess.run(["mkdir", "-p", str(target_path)])
        if use_audio_filter:
            print("Running filter...")
            subprocess.run(["sox", "rnn.wav", str(target_file), "remix", "-", "highpass", "50", "lowpass", "8000",
                            "rate", "22050"])  # apply high/low pass filter and change sr to 22050Hz
            data, rate = sf.read(target_file)
        else:
            print("Skipping filter...")
            subprocess.run(["sox", "rnn.wav", str(target_file), "remix", "-", "rate", "22050"])  # apply high/low pass filter and change sr to 22050Hz
            data, rate = sf.read(target_file)
        if normalize_audio:
            print("Output normalized")
            peak_normalized_audio = pyln.normalize.peak(data, -6.0)

            # measure the loudness first
            meter = pyln.Meter(rate)  # create BS.1770 meter
            loudness = meter.integrated_loudness(data)

            # loudness normalize audio to -25 dB LUFS
            loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -25.0)
            sf.write(target_file, data=loudness_normalized_audio, samplerate=22050)
            print("")
        else:
            print("File written without normalizing")
            sf.write(target_file, data=data, samplerate=22050)
            print("")

    for file_path in [target_file, "/home/ubuntu/ml_train/rnnoise/examples/ft.wav"]:
        if os.path.exists(file_path):
            os.remove(file_path)

if run_splits:
    print("Splitting output and copying...")
    splits_dir = os.path.join(upload_dir, "kch", "splits")
    resplit_dir = os.path.join(upload_dir, "kch", "splits", "resplit")

    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(resplit_dir, exist_ok=True)

    for file_path in glob.glob(os.path.join(splits_dir, "*.wav")):
        os.remove(file_path)

    for file_path in glob.glob(os.path.join(resplit_dir, "*.wav")):
        os.remove(file_path)

    for file_path in glob.glob(os.path.join(orig_wavs, "*.wav")):
        file_name = os.path.basename(file_path)
        subprocess.run(["sox", file_path, os.path.join(splits_dir, file_name), "--show-progress", "silence", "1",
                        "0.2", "0.1%", "1", "0.2", "0.1%", ":", "newfile", ":", "restart"])

    os.chdir(splits_dir)

    for file_path in glob.glob(os.path.join(splits_dir, "*.wav")):
        file_name = os.path.basename(file_path)
        subprocess.run(["sox", file_path, os.path.join(resplit_dir, file_name), "--show-progress", "trim", "0", "8",
                        ":", "newfile", ":", "restart"])

    for file_path in glob.glob(os.path.join(resplit_dir, "*.wav")):
        if os.path.getsize(file_path) < 500:
            os.remove(file_path)

    os.chdir('/home/ubuntu/ml_train/' + ds_name)

    os.makedirs(os.path.join('/home/ubuntu/ml_train/' + ds_name, 'wavs'), exist_ok=True)

    for file_path in glob.glob(os.path.join(resplit_dir, "*.wav")):
        file_name = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join('/home/ubuntu/ml_train/' + ds_name, 'wavs', file_name))
else:
    print("Copying files without splitting...")
    for file_path in glob.glob(os.path.join(target_path, '*.wav')):
        file_name = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join('/home/ubuntu/ml_train/' + ds_name, 'wavs', file_name))

# Load Whisper model
whisper_model_path = "large-v2"
model = whisper.Whisper(whisper_model_path)

# Load and preprocess samples
samples_path = '/home/ubuntu/ml_train/' + ds_name + '/wavs'
samples = []
with open('/home/ubuntu/ml_train/' + ds_name + '/metadata.csv', 'w', encoding='utf-8') as outfile:
    for file_name in os.listdir(samples_path):
        file_path = os.path.join(samples_path, file_name)
        samples.append((file_name, file_path))

    for file_name, file_path in samples:
        audio, rate = sf.read(file_path)
        text = model.transcribe(file_path)["text"].strip()
        output = "|".join([file_name, text, text]) + '\n'
        outfile.write(output)

# Display dataset
with open('/home/ubuntu/ml_train/' + ds_name + '/metadata.csv', 'r', encoding='utf-8') as infile:
    print(infile.read())

# Load Tensorboard
import torch
# %load_ext tensorboard

# Load Dashboard
# %tensorboard --logdir /home/ubuntu/ml_train/$ds_name/$output_directory/

# Load configs
config = load_config("Vits")

# Initialize audio processor
ap = AudioProcessor(**config.audio)

# Initialize tokenizer
tokenizer = TTSTokenizer(config)

# Load data samples
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="/home/ubuntu/ml_train/" + ds_name + "/metadata.csv",
    path="/home/ubuntu/ml_train/" + ds_name
)
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Initialize the model
model = Vits(**config.model)
model.setup(**config.model_args)

# Initialize the trainer
trainer_args = TrainerArgs(
    restore_path=MODEL_FILE,
    skip_train_epoch=False
)
trainer = Trainer(
    trainer_args,
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples
)

# Run training
trainer.fit()
