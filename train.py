import os

ds_name = "vits-ds" #@param {type:"string"}
output_directory = "traineroutput" #@param {type:"string"}
upload_dir = "dataset" #@param {type:"string"}
MODEL_FILE = " /home/ubuntu/.local/share/tts/tts_models--en--ljspeech--vits/model_file.pth" #@param {type:"string"}
upload_dir = "/home/ubuntu/ml_train/" + upload_dir
RUN_NAME = "VITS-en" #@param {type:"string"}

OUT_PATH = "/home/ubuntu/ml_train/"+ds_name+"/traineroutput/"
!mkdir $upload_dir
!mkdir /home/ubuntu/ml_train/$ds_name
!mkdir /home/ubuntu/ml_train/$ds_name/wavs/

run_type = "restore" #@param ["continue","restore","restore-ckpt"]
print(run_type + " run selected")

run_denoise = "True" #@param ["True", "False"]
run_splits = "True" #@param ["True", "False"]
use_audio_filter = "True" #@param ["True", "False"]
normalize_audio = "True" #@param ["True", "False"]

from pathlib import Path
import os
import subprocess
import soundfile as sf
import pyloudnorm as pyln
import sys
import glob
os.chdir($upload_dir)
!ls -al
!rm -rf $upload_dir/22k_1ch
!mkdir $upload_dir/22k_1ch

#Convert and resample uploaded mp3/wav clips to 1 channel, 22khz
!find . -name '*.mp3' -exec bash -c 'for f; do ffmpeg -hide_banner -loglevel error -i "$f" -acodec pcm_s16le -ar 22050 -ac 1 22k_1ch/"${f%.mp3}".wav ; done' _ {} +
!find . -name '*.wav' -exec bash -c 'for f; do ffmpeg -hide_banner -loglevel error -i "$f" -acodec pcm_s16le -ar 22050 -ac 1 22k_1ch/"${f%.wav}".wav ; done' _ {} +
!ls -al $upload_dir/22k_1ch
print("Files converted to 22khz 1ch wav")
if run_denoise=="True":
  print("Running denoise...")
  orig_wavs= upload_dir + "/22k_1ch/"
  print(orig_wavs)

  from pathlib import Path
  import os
  import subprocess
  import soundfile as sf
  import pyloudnorm as pyln
  import sys
  import glob
  rnn = "/home/ubuntu/ml_train/rnnoise/examples/rnnoise_demo"
  paths = glob.glob(os.path.join(orig_wavs, '*.wav'))
  for filepath in paths:
    base = os.path.basename(filepath)
    tp_s = upload_dir + "/22k_1ch/denoise/"
    tf_s = upload_dir + "/22k_1ch/denoise/" + base
    target_path = Path(tp_s)
    target_file = Path(tf_s)
    print("From: " + str(filepath))
    print("To: " + str(target_file))

  # Stereo to Mono; upsample to 48000Hz
  # added -G to fix gain, -v 0.8
    subprocess.run(["sox", "-G", "-v", "0.8", filepath, "48k.wav", "remix", "-", "rate", "48000"])
    subprocess.run(["sox", "48k.wav", "-c", "1", "-r", "48000", "-b", "16", "-e", "signed-integer", "-t", "raw", "temp.raw"]) # convert wav to raw
    subprocess.run(["/home/ubuntu/ml_train/rnnoise/examples/rnnoise_demo", "temp.raw", "rnn.raw"]) # apply rnnoise
    subprocess.run(["sox", "-G", "-v", "0.8", "-r", "48k", "-b", "16", "-e", "signed-integer", "rnn.raw", "-t", "wav", "rnn.wav"]) # convert raw back to wav

    subprocess.run(["mkdir", "-p", str(target_path)])
    if use_audio_filter=="True":
      print("Running highpass/lowpass & resample")
      subprocess.run(["sox", "rnn.wav", str(target_file), "remix", "-", "highpass", "50", "lowpass", "8000", "rate", "22050"])
      # apply high/low pass filter and change sr to 22050Hz
      data, rate = sf.read(target_file)
    elif use_audio_filter=="False":
      print("Running resample without filter")
      subprocess.run(["sox", "rnn.wav", str(target_file), "remix", "-", "rate", "22050"])
      # apply high/low pass filter and change sr to 22050Hz
      data, rate = sf.read(target_file)
# peak normalize audio to -6 dB
    if normalize_audio=="True":
      print("Output normalized")
      peak_normalized_audio = pyln.normalize.peak(data, -6.0)

# measure the loudness first
      meter = pyln.Meter(rate) # create BS.1770 meter
      loudness = meter.integrated_loudness(data)

# loudness normalize audio to -25 dB LUFS
      loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -25.0)
      sf.write(target_file, data=loudness_normalized_audio, samplerate=22050)
      print("")
    elif normalize_audio=="False":
      print("File written without normalizing")
      sf.write(target_file, data=data, samplerate=22050)
      print("")

  !rm $target_path/rnn.wav
  !rm $target_path/48k.wav

elif run_denoise=="False":
  paths = glob.glob(os.path.join(orig_wavs, '*.wav'))
  for filepath in paths:
    print("Skipping denoise...")
    base = os.path.basename(filepath)
    tp_s = upload_dir + "/22k_1ch/denoise/"
    tf_s = upload_dir + "/22k_1ch/denoise/" + base
    target_path = Path(tp_s)
    target_file = Path(tf_s)
    print("From: " + str(filepath))
    print("To: " + str(target_file))
    subprocess.run(["sox", "-G", "-v", "0.8", filepath, "48k.wav", "remix", "-", "rate", "48000"])
    subprocess.run(["sox", "48k.wav", "-c", "1", "-r", "48000", "-b", "16", "-e", "signed-integer", "-t", "raw", "temp.raw"]) # convert wav to raw
    #subprocess.run(["/home/ubuntu/ml_train/rnnoise/examples/rnnoise_demo", "temp.raw", "rnn.raw"]) # apply rnnoise
    subprocess.run(["sox", "-G", "-v", "0.8", "-r", "48k", "-b", "16", "-e", "signed-integer", "rnn.raw", "-t", "wav", "rnn.wav"]) # convert raw back to wav
    subprocess.run(["mkdir", "-p", str(target_path)])
    if use_audio_filter=="True":
      print("Running filter...")
      subprocess.run(["sox", "rnn.wav", str(target_file), "remix", "-", "highpass", "50", "lowpass", "8000", "rate", "22050"]) # apply high/low pass filter and change sr to 22050Hz
      data, rate = sf.read(target_file)
    elif use_audio_filter=="False":
      print("Skipping filter...")
      subprocess.run(["sox", "rnn.wav", str(target_file), "remix", "-", "rate", "22050"]) # apply high/low pass filter and change sr to 22050Hz
      data, rate = sf.read(target_file)
          # peak normalize audio to -6 dB
    if normalize_audio=="True":
      print("Output normalized")
      peak_normalized_audio = pyln.normalize.peak(data, -6.0)

# measure the loudness first
      meter = pyln.Meter(rate) # create BS.1770 meter
      loudness = meter.integrated_loudness(data)

# loudness normalize audio to -25 dB LUFS
      loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -25.0)
      sf.write(target_file, data=loudness_normalized_audio, samplerate=22050)
      print("")
    if normalize_audio=="False":
      print("File written without normalizing")
      sf.write(target_file, data=data, samplerate=22050)
      print("")
  !rm $target_path/rnn.wav
  !rm $target_path/48k.wav

if run_splits=="False":
  print("Copying files without splitting...")
#   %mkdir /home/ubuntu/ml_train/$ds_name
#   %mkdir /home/ubuntu/ml_train/$ds_name/wavs
  !cp $target_path/*.wav /home/ubuntu/ml_train/$ds_name/wavs
if run_splits=="True":
  print("Splitting output and copying...")
  os.chdir($target_path)
  !rm -rf splits
  !mkdir splits
  !for FILE in *.wav; do sox "$FILE" splits/"$FILE" --show-progress silence 1 0.2 0.1% 1 0.2 0.1% : newfile : restart ; done
#alt split method: force splits of 8 seconds, however this will split words. Comment the above with # and remove the # below to change
#!for FILE in *.wav; do sox "$FILE" splits/"$FILE" --show-progress trim 0 8 : restart ; done
  os.chdir('splits')
  !mkdir resplit
  !for FILE in *.wav; do sox "$FILE" resplit/"$FILE" --show-progress trim 0 8 : newfile : restart ; done
  os.chdir('resplit')
  !find . -name "*.wav" -type f -size -35k -delete
  !ls -al
  os.chdir('/home/ubuntu/ml_train/$ds_name/')

  !mkdir wavs
  !cp $target_path/splits/resplit/*.wav /home/ubuntu/ml_train/$ds_name/wavs


#@title
import whisper
import os, os.path
import glob
import pandas as pd

from pathlib import Path


#model = whisper.load_model("medium.en")
model = whisper.load_model("large-v2")

"""**Run Whisper on generated audio clips, generate transcript named metadata.csv in LJSpeech format in the dataset directory.**"""

#@title
wavs = '/home/ubuntu/ml_train/'+ds_name+'/wavs'

paths = glob.glob(os.path.join(wavs, '*.wav'))
print(len(paths))

all_filenames = []
transcript_text = []
with open('/home/ubuntu/ml_train/'+ds_name+'/metadata.csv', 'w', encoding='utf-8') as outfile:
	for filepath in paths:
		base = os.path.basename(filepath)
		all_filenames.append(base)
	for filepath in paths:
		result = model.transcribe(filepath)
		output = result["text"].lstrip()
		output = output.replace("\n","")
		thefile = str(os.path.basename(filepath).lstrip(".")).rsplit(".")[0]
		outfile.write(thefile + '|' + output + '|' + output + '\n')
		print(thefile + '|' + output + '|' + output + '\n')

"""**Display dataset**"""

#@title
!cat /home/ubuntu/ml_train/$ds_name/metadata.csv


!tts --text "I am the very model of a modern Major General" --model_name "tts_models/en/ljspeech/vits" --out_path /home/ubuntu/ml_train/ljspeech-vits.wav

"""**Load Tensorboard**"""

# Commented out IPython magic to ensure Python compatibility.
import torch
# %load_ext tensorboard

"""**Load Dashboard**
May take several minutes to appear from a blank white box.  Ad blockers probably need to whitelist a bunch of Colab stuff or this won't work.
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir /home/ubuntu/ml_train/$ds_name/$output_directory/

"""**Load libs**"""

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = "/home/ubuntu/ml_train/"+ds_name + "/" + output_directory + "/"
SKIP_TRAIN_EPOCH = False

dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "/home/ubuntu/ml_train/"+ ds_name)
)

audio_config = VitsAudioConfig(
    sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=150, mel_fmax=9000
)

config = VitsConfig(
    audio=audio_config,
    run_name="vits_ljspeech",
    batch_size=16,
    eval_batch_size=16,
    batch_group_size=16,
#    num_loader_workers=8,
    num_loader_workers=2,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=50000,
    save_step=1000,
	  save_checkpoints=True,
	  save_n_checkpoints=4,
	  save_best_after=1000,
    #text_cleaner="english_cleaners",
    text_cleaner="multilingual_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

model = Vits.init_from_config(config)

"""**If continuning a run: use the next cell to list all run directories.**

**Copy and paste the run you want to or restore a checkpoint from into the next box**
"""

#@title
# !ls -al /home/ubuntu/ml_train/$ds_name/traineroutput

"""**Run folder to continue from or Run folder that contains your restore checkpoint**"""

# run_folder = "vits_ljspeech-July-13-2023_10+27PM-0000000" #@param {type:"string"}

"""List checkpoints in run folder. The checkpoint only needs to be selected for a restore run.

Continuing a run will load the last best loss checkpoint according to the stored config.json in the run directory on its own (a directory is specified for a continue run, and a model file is specified for a restore run)
"""

#@title
# !ls -al /home/ubuntu/ml_train/$ds_name/traineroutput/$run_folder

"""**If changing to a different "restore" checkpoint to begin a new training session with a model you are already training, set the checkpoint filename here**"""

# ckpt_file = "checkpoint_1012000.pth" #@param {type:"string"}
# print(ckpt_file + " selected for restore run")
# if run_type=="continue":
#   print("Warning:\n restore checkpoint selected, but run type set to continue.\nTrainer will load best loss from checkpoint directory.\n Are you sure this is what you want to do?\n\nIf not, change the run type below to 'restore'")
# elif run_type=="restore-ckpt":
#   print("Warning:\n restore checkpoint selected, run type set to restore from selected checkpoint, not default base model.\nIf this is not correct, adjust the run type.")

# """**Last chance to change run type**"""

# run_type = "restore" #@param ["continue","restore","restore-ckpt"]
# print(run_type + " run selected")

"""**(Session recovery: Reset selected model file back to default predownloaded path)**"""

#@title
ckpt_file = "/home/ubuntu/.local/share/tts/tts_models--en--ljspeech--vits/model_file.pth"
print(ckpt_file + " selected for restore run")

"""**(Optional) Freeze selected modules. Trainer must be reinitilized if these are changed.**"""

print("Current reinit_text_encoder value: " + str(config.model_args.reinit_text_encoder))
reinit_te_status = "False" #@param ["False", "True"]
if reinit_te_status=="False":
  print("Text encoder will not be reinitilized")
elif reinit_te_status=="True":
  config.model_args.reinit_text_encoder=True
  print("Model arguments set to reinitilize text encoder")
  print("Current reinit_DP value: " + str(config.model_args.reinit_DP))
reinit_DP_status = "False" #@param ["False", "True"]
if reinit_DP_status=="False":
  print("DP will not be reinitilized")
elif reinit_DP_status=="True":
  config.model_args.reinit_DP=True
  print("Model arguments set to reinitilize DP")
print("Current freeze_waveform_decoder value: " + str(config.model_args.freeze_waveform_decoder))
freeze_waveform_decoder_status = "False" #@param ["False", "True"]
if freeze_waveform_decoder_status=="False":
  print("Waveform decoder will NOT be frozen")
  config.model_args.freeze_waveform_decoder=False
elif freeze_waveform_decoder_status=="True":
  config.model_args.freeze_waveform_decoder=True
  print("Waveform decoder FROZEN")
print("Current freeze_flow_decoder value: " + str(config.model_args.freeze_flow_decoder))
freeze_flow_decoder_status = "False" #@param ["False", "True"]
if freeze_flow_decoder_status=="False":
  print("Flow decoder will NOT be frozen")
  config.model_args.freeze_flow_decoder=None
elif freeze_flow_decoder_status=="True":
  config.model_args.freeze_flow_decoder="True"
  print("Flow decoder FROZEN")
print("Current freeze_encoder value: " + str(config.model_args.freeze_encoder))
freeze_encoder_status = "False" #@param ["False", "True"]
if freeze_encoder_status=="False":
  print("Text encoder will NOT be frozen")
  config.model_args.freeze_encoder=False
elif freeze_encoder_status=="True":
  config.model_args.freeze_encoder=True
  print("Text encoder FROZEN")
print("Current freeze_DP value: " + str(config.model_args.freeze_DP))
freeze_DP_status = "False" #@param ["False", "True"]
if freeze_DP_status=="False":
  print("Duration predictor will NOT be frozen")
  config.model_args.freeze_DP=False
elif freeze_DP_status=="True":
  config.model_args.freeze_DP=True
  print("Duration predictor FROZEN")

"""**Init the trainer**"""

#@title
print(run_type)
if run_type=="continue":
  CONTINUE_PATH="/home/ubuntu/ml_train/"+ds_name+"/traineroutput/"+run_folder
  trainer = Trainer(
    TrainerArgs(continue_path=CONTINUE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
elif run_type=="restore":
    trainer = Trainer(
    TrainerArgs(restore_path=MODEL_FILE, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
elif run_type=="restore-ckpt":
  trainer = Trainer(
  TrainerArgs(restore_path="/home/ubuntu/ml_train/"+ds_name+"/traineroutput/"+run_folder+"/"+ckpt_file, skip_train_epoch=SKIP_TRAIN_EPOCH),
  config,
  output_path=OUT_PATH,
  model=model,
  train_samples=train_samples,
  eval_samples=eval_samples,
)

"""**Run training**"""

trainer.fit()




