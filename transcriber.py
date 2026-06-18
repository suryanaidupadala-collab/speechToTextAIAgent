import threading
import time
import logging
from typing import Optional, Iterable

import numpy as np

from faster_whisper import WhisperModel

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TranscriptionWorker:
    """Worker that consumes audio chunks and transcribes them in a background thread.

    Exposes callbacks:
      - on_partial(text)
      - on_final(text)
      - on_status(status)
      - on_error(exception)
    """

    def __init__(self, model_name: str = 'base', device: str = 'cpu', language: Optional[str] = 'en', samplerate: int = 16000):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.samplerate = samplerate

        self._model: Optional[WhisperModel] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        # callbacks
        self.on_partial = lambda text: None
        self.on_final = lambda text: None
        self.on_status = lambda status: None
        self.on_error = lambda e: None

        # tuning: window for transcription (balanced for real-time response)
        self.window_seconds = 5.0   # Smaller window for more responsive feedback
        self.overlap_seconds = 1.0  # Moderate overlap for context

    def load_model(self):
        try:
            self.on_status('Loading model...')
            logger.info(f'Loading {self.model_name} model on {self.device}')
            self._model = WhisperModel(self.model_name, device=self.device, compute_type="int8")
            self.on_status('Model loaded')
            logger.info('Model loaded successfully')
        except Exception as e:
            logger.error(f'Failed to load model: {e}')
            self.on_error(e)

    def start(self, audio_iter: Iterable):
        if self._thread and self._thread.is_alive():
            logger.warning('Transcriber thread already running')
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, args=(audio_iter,), daemon=True)
        self._thread.start()
        logger.info('Transcriber started')

    def stop(self):
        logger.info('Stopping transcriber')
        self._stop_flag.set()

    def _run(self, audio_iter: Iterable):
        try:
            if self._model is None:
                self.load_model()

            if self._model is None:
                raise RuntimeError('Model failed to load')

            self.on_status('Listening...')
            logger.info('Transcription worker listening...')

            buffer = []
            buffered_samples = 0
            window_size = int(self.window_seconds * self.samplerate)
            overlap_size = int(self.overlap_seconds * self.samplerate)
            chunks_processed = 0
            silence_count = 0
            max_silence_chunks = 10  # Process after 10 silent chunks (0.64 sec)
            last_transcription = ""
            consecutive_empty = 0

            for item in audio_iter:
                if self._stop_flag.is_set():
                    logger.info('Stop flag set, exiting loop')
                    break

                pcm, status = item
                if status is not None:
                    logger.warning(f'Audio status: {status}')
                    continue

                # convert to float32 in [-1,1]
                if pcm.dtype == np.int16:
                    audio = pcm.astype(np.float32) / 32767.0
                else:
                    audio = pcm.astype(np.float32)

                audio = audio.reshape(-1)
                buffer.append(audio)
                buffered_samples += audio.shape[0]
                chunks_processed += 1

                # Check for silence (audio level < 1% of max)
                is_silent = np.abs(audio).max() < 0.01
                if is_silent:
                    silence_count += 1
                else:
                    silence_count = 0
                    consecutive_empty = 0

                # Transcribe when:
                # 1. Window is full (5 seconds), OR
                # 2. Silence detected after speech (pause), OR
                # 3. Have at least 2 seconds of audio and long silence
                should_transcribe = (
                    buffered_samples >= window_size or 
                    (silence_count > max_silence_chunks and buffered_samples >= int(1.5 * self.samplerate))
                )
                
                if should_transcribe and buffered_samples > 0:
                    logger.info(f'Window ready: {buffered_samples} samples ({buffered_samples/self.samplerate:.2f}s), processing...')
                    segment = np.concatenate(buffer)

                    try:
                        segments, info = self._model.transcribe(
                            segment, 
                            beam_size=5, 
                            language=self.language,
                            no_speech_threshold=0.4  # Lower threshold to catch quiet speech
                        )
                    except Exception as e:
                        logger.error(f'Transcription error: {e}')
                        self.on_error(e)
                        break

                    # combine segment texts
                    full_text = ' '.join(s.text for s in segments).strip()
                    
                    # Only output if we got new text (avoid duplicates)
                    if full_text and full_text != last_transcription:
                        logger.info(f'Transcribed: "{full_text}"')
                        self.on_final(full_text)
                        last_transcription = full_text
                        consecutive_empty = 0
                    elif not full_text:
                        consecutive_empty += 1
                        if consecutive_empty <= 2:  # Log a few empty detections
                            logger.debug('No speech detected in window')
                    
                    # keep overlap for context
                    if overlap_size > 0 and len(segment) > overlap_size:
                        tail = segment[-overlap_size:]
                        buffer = [tail]
                        buffered_samples = tail.shape[0]
                    else:
                        buffer = []
                        buffered_samples = 0
                    
                    silence_count = 0

            logger.info(f'Transcription worker exiting. Processed {chunks_processed} chunks')
            self.on_status('Stopped')

        except Exception as e:
            logger.error(f'Worker exception: {e}', exc_info=True)
            self.on_error(e)
