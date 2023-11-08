import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from howl.client import HowlClient

import logging
import time
from typing import Callable
from howl.context import InferenceContext
from howl.settings import SETTINGS
from howl.settings import HowlSettings
import howl.model as howl_model
from howl.workspace import Workspace
from pathlib import Path
import numpy as np
import pyaudio
import howl.data.transform as transform
import torch
import os
from howl.context import InferenceContext
from howl.model.inference import FrameInferenceEngine, InferenceEngine
from howl.model import ConfusionMatrix, ConvertedStaticModel, RegisteredModel
from howl.data.transform.operator import ZmuvTransform, batchify, compose
from howl.utils import logging_utils
from howl.utils.args_utils import ArgOption, ArgumentParserBuilder
import wave
import torchaudio
import torch.nn.functional as F
from howl.utils import audio_utils

# state machine for recognization
class StateMachine:
    def __init__(self, length , interval_threshold, delay_interval):
        self.goal = length
        self.current_state = 0
        self.last_timestamp = -1
        self.interval_threshold = interval_threshold
        self.delay_interval = delay_interval
        self.last_goal_timestamp = -1

    def transition(self, label):
        isGoal = False
        word,t = label
        if t - self.last_timestamp > self.interval_threshold:
            self.current_state = 0 #reset state if next keyword is too far from previous
        if word == self.current_state:
            self.current_state += 1 # move to next state
            self.last_timestamp = t
        elif word < self.current_state:
            return isGoal
        else:
            self.current_state = 0 # return to initial state if the word doesn't follow sequence
            return isGoal
        if self.current_state == self.goal:
            self.current_state = 0 # return to initial if Goal
            isGoal = True
            if isGoal and t - self.last_goal_timestamp < self.delay_interval: return False # Suppress rapid twice detection
            self.last_goal_timestamp = t
            self.last_timestamp = t
        return isGoal
    
    def update(self, labels):
        # if time.time() - self.last_timestamp > self.interval_threshold: 
        #     self.current_state = 0
        for label in labels:
            isGoal = self.transition(label)
            if isGoal: return True
        return False
    
"""Load a pretrained model using the provided name"""
path = "/media/nontawat/Windows/wakeword_ws/howl_/howl/workspaces/hello-cape-bee-res8-reproduce-neg-talknoise-200k"
# path = '/media/nontawat/Windows/wakeword_ws/howl_/howl/workspaces/hey-ff-res8-reproduce-neg'
device = "cpu"
workspace = Workspace(Path(path), delete_existing=False)
# Load model settings
settings = workspace.load_settings()

# Set up context
use_frame = settings.training.objective == "frame"
ctx = InferenceContext(
    vocab=settings.training.vocab, token_type=settings.training.token_type, use_blank=not use_frame
)

# Load models
zmuv_transform = transform.ZmuvTransform()
model = howl_model.RegisteredModel.find_registered_class("res8")(ctx.num_labels).eval()

# Load pretrained weights
zmuv_transform.load_state_dict(
    torch.load(str(workspace.path / "zmuv.pt.bin"), map_location=torch.device(device))
)
workspace.load_model(model, best=True)

# Load engine
model.streaming()
if use_frame:
    engine = FrameInferenceEngine(
        int(settings.training.max_window_size_seconds * 1000),
        int(settings.training.eval_stride_size_seconds * 1000),
        model,
        zmuv_transform,
        ctx,
    )
else:
    engine = InferenceEngine(model, zmuv_transform, ctx)

model = engine.model
zmuv = engine.zmuv

# wav_file = '/media/nontawat/Windows/wakeword_ws/howl_/howl/audio_clip/hey.wav'
# waveform, sample_rate = torchaudio.load(wav_file)
# print("Waveform shape:", waveform.shape)
# print("sample_rate", sample_rate)

# audio_data = waveform[0]

buffers = []
last_data = None
seq_preds = []
OOV_LABEL = 3
BUFFER_SIZE = 32

stateMachine = StateMachine(3, 1, 1)

def _normalize_audio(audio_data):
    return np.frombuffer(audio_data, dtype=np.int16).astype(np.float) / 32767

def store_buffer(in_data, frame_count, time_info, status_flags):
    global buffers, last_data, seq_preds
    data_ok = (in_data, pyaudio.paContinue)
    # data_ok = (in_data, pyaudio.paContinue)
    last_data = in_data
    # if len(buffers) != 16: return data_ok
    if len(buffers) >= BUFFER_SIZE:
        buffers = buffers[1:]
    buffers.append(in_data)
    return data_ok

def recognize_buffer():
    global buffers, last_data
    if len(buffers) < BUFFER_SIZE: return []
    
    audio_data = b"".join(buffers)
    # buffers = buffers[BUFFER_SIZE:]
    arr =  _normalize_audio(audio_data)
    inp = torch.from_numpy(arr).float().to(device)
    
    audio_data = inp
    # print(inp.abs().mean())
    detected = []
    count = 0
    for window in audio_utils.stride(
            audio_data, engine.max_window_size_ms, engine.eval_stride_size_ms, engine.sample_rate
        ):
        if window.size(-1) < 1000:
            break
        frame = window.squeeze(0)

        engine.std = engine.std.to(frame.device)
        lengths = torch.tensor([frame.size(-1)]).to(frame.device)
        transformed_lengths = engine.std.compute_lengths(lengths)
        transformed_frame = engine.zmuv(engine.std(frame.unsqueeze(0)))
        prediction = engine.model(transformed_frame, transformed_lengths).softmax(-1)[0].cpu().detach().numpy()

        prediction *= engine.inference_weights
        prediction = prediction / prediction.sum()
        label = prediction.argmax()
        # print("predictoin", label)
        if label != OOV_LABEL:
            detected.append((label, time.time()))
        # count += 1
    # print(detected)
    return detected

audioobj = pyaudio.PyAudio()
chosen_idx = 0
for idx in range(audioobj.get_device_count()):
    info = audioobj.get_device_info_by_index(idx)
    if info["name"] == "pulse" or info["name"] == "sysdefault":
        chosen_idx = idx

print("chosen_idx",chosen_idx)
stream = audioobj.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=chosen_idx,
            frames_per_buffer=500,
            stream_callback=store_buffer,
        )
print("open stream")
stream.start_stream()

while stream.is_active():
    time.sleep(0.1)
    labels = recognize_buffer()
    print([i[0] for i in labels])
    print(stateMachine.current_state)
    isGoal = stateMachine.update(labels)
    if isGoal:
        print("Detected : ",time.time())