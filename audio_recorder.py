import queue
import sounddevice as sd
import numpy as np
from typing import Generator, Optional


class AudioRecorder:
    """Record microphone audio in a background thread and yield PCM chunks.

    Uses sounddevice.InputStream and a Queue to hand off audio to the consumer.
    """

    def __init__(self, samplerate: int = 16000, channels: int = 1, chunk_size: int = 1024):
        self.samplerate = samplerate
        self.channels = channels
        self.chunk_size = chunk_size
        self._q: queue.Queue = queue.Queue()
        self._stream: Optional[sd.InputStream] = None

    def _callback(self, indata, frames, time, status):
        if status:
            # push status to queue as a warning
            self._q.put_nowait((None, status))
            return
        # Convert to 16-bit PCM
        pcm = (indata.copy() * 32767).astype(np.int16)
        self._q.put_nowait((pcm, None))

    def start(self):
        if self._stream is not None:
            return
        self._stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels,
                                       blocksize=self.chunk_size, dtype='float32', callback=self._callback)
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def audio_generator(self) -> Generator[np.ndarray, None, None]:
        """Yield PCM numpy arrays as they arrive.

        Yields (pcm, None) tuples. If (None, status) is received, it's a status update from sounddevice.
        """
        while True:
            item = self._q.get()
            yield item
