from howl.client import HowlClient

import logging
import time
from typing import Callable
from howl.context import InferenceContext
from howl.settings import SETTINGS

import numpy as np
import pyaudio
import torch

from howl.context import InferenceContext
from howl.model.inference import InferenceEngine
from howl.model import ConfusionMatrix, ConvertedStaticModel, RegisteredModel
from howl.data.transform.operator import ZmuvTransform, batchify, compose
from howl.utils import logging_utils
from howl.utils.args_utils import ArgOption, ArgumentParserBuilder

device = torch.device(SETTINGS.training.device)

apb = ArgumentParserBuilder()
apb.add_options(
    ArgOption("--model", type=str, choices=RegisteredModel.registered_names(), default="res8",),
    # ArgOption("--workspace", type=str, default=str(Path("workspaces") / "default")),
    ArgOption("--load-weights", action="store_true"),
    ArgOption("--load-last", action="store_true"),
    ArgOption("--dataset-paths", "-i", type=str, nargs="+", default=[SETTINGS.dataset.dataset_path],),
    ArgOption("--eval-freq", type=int, default=10),
    ArgOption("--eval", action="store_true"),
    ArgOption("--use-stitched-datasets", action="store_true"),
)
args = apb.parser.parse_args()

class HowlClientCustom:
    """
    A client for serving Howl models. Users can provide custom listener callbacks
    when a wake word is detected using the `add_listener` method.

    The `from_pretrained` method allows users to directly load a pretrained model by name,
    such as "hey_fire_fox", if the model is not provided upon initialization.
    """

    def __init__(
        self,
        engine: InferenceEngine = None,
        context: InferenceContext = None,
        device: torch.device = torch.device("cpu"),
        chunk_size: int = 500,
        logger: logging.Logger = None,
    ):
        """
        Initializes the client.

        Parameters:
            engine (InferenceEngine): An InferenceEngine for processing audio input, provided
                                        on initialization or by loading a pretrained model.
            context (InferenceContext): An InferenceContext object containing vocab, provided
                                        on initialization or by loading a pretrained model.
            device (torch.device): Device for CPU/GPU support (Default: cpu).
            chunk_size (int): Number of frames per buffer for audio.
        """
        self.logger = logger
        if self.logger is None:
            self.logger = logging_utils.setup_logger(self.__class__.__name__)

        self.listeners = []
        self.chunk_size = chunk_size
        self.device = device

        self.engine: InferenceEngine = engine
        self.ctx: InferenceContext = context
        # PyAudio instance which gets created upon start call
        self._audio = None
        # PyAudio Stream instance which gets created by self._audio upon start call
        self._stream = None

        self._audio_buf = []
        self._audio_buf_len = 16
        self._audio_float_size = 32767
        self._infer_detected = False
        self.last_data = np.zeros(self.chunk_size)

    @staticmethod
    def list_pretrained(force_reload: bool = False):
        """Show a list of available pretrained models"""
        print(torch.hub.list("castorini/howl", force_reload=force_reload))

    def _on_audio(self, in_data, frame_count, time_info, status_flags):
        """triggered when a new frame of data is available
        details: http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#a8a60fb2a5ec9cbade3f54a9c978e2710

        Args:
            in_data: recorded data if input=True; else None
            frame_count: number of frames
            time_info: dictionary
            status_flags: PaCallbackFlags

        Returns:
            (in_data, pyaudio.paContinue)
        """
        # pylint: disable=unused-argument
        data_ok = (in_data, pyaudio.paContinue)
        self.last_data = in_data
        self._audio_buf.append(in_data)
        if len(self._audio_buf) != self._audio_buf_len:
            return data_ok

        audio_data = b"".join(self._audio_buf)
        self._audio_buf = self._audio_buf[2:]
        arr = self._normalize_audio(audio_data)
        inp = torch.from_numpy(arr).float().to(self.device)

        # Inference from input sequence
        if self.engine.infer(inp):
            # Check if inference has already occurred for this sequence to prevent
            # duplicate callback execution
            if self._infer_detected:
                return data_ok

            self._infer_detected = True
            phrase = " ".join(self.ctx.vocab[x] for x in self.engine.sequence).title()
            self.logger.info(f"{phrase} detected")
            # Execute user-provided listener callbacks
            for lis in self.listeners:
                lis(self.engine.sequence)
        else:
            self._infer_detected = False

        return data_ok

    def _normalize_audio(self, audio_data):
        return np.frombuffer(audio_data, dtype=np.int16).astype(np.float) / self._audio_float_size

    def start(self):
        """Start the audio stream for inference"""
        if self.engine is None:
            raise AttributeError("Please provide an InferenceEngine or initialize using from_pretrained.")
        if self.ctx is None:
            raise AttributeError("Please provide an InferenceContext or initialize using from_pretrained.")

        chosen_idx = 0
        self._audio = pyaudio.PyAudio()
        for idx in range(self._audio.get_device_count()):
            info = self._audio.get_device_info_by_index(idx)
            if info["name"] == "pulse" or info["name"] == "sysdefault":
                chosen_idx = idx
                break

        stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=chosen_idx,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._on_audio,
        )
        self._stream = stream
        self.logger.info("Starting Howl inference client...")
        stream.start_stream()
        return self

    def join(self):
        """Block while the audio inference stream is active"""
        while self._stream.is_active():
            time.sleep(0.1)

    def load_model(self, path: str, force_reload: bool = False):
        """Load a pretrained model using the provided name"""
        use_frame = SETTINGS.training.objective == "frame"
        ctx = InferenceContext(
        vocab=SETTINGS.training.vocab, token_type=SETTINGS.training.token_type, use_blank=not use_frame,
        )
        zmuv_transform = ZmuvTransform().to(device)
        engine = InferenceEngine(RegisteredModel.find_registered_class(args.model)(ctx.num_labels).to(device).streaming(), context=ctx, zmuv_transform=zmuv_transform)
        self.engine = engine.to(self.device)
        self.ctx = ctx

    def add_listener(self, listener: Callable):
        """
        Add a listener callback to be executed when a sequence is detected

        Parameters:
            listener (Callable): A function that takes in a list of strings as an argument,
                                 which will contain the detected wake words.
        """
        self.listeners.append(listener)

def hello_callback(detected_words):
    print("Detected: {}".format(detected_words))

client = HowlClientCustom()
client.load_model("/home/nlpintern1/howl-ws/howl/workspaces/hey-robin-res8", force_reload=False)
client.add_listener(hello_callback)
client.start().join()