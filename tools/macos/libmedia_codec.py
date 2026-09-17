"""Drop-in replacement for DJI's native ``libmedia_codec`` on macOS.

The RoboMaster SDK only ships this decoder as a compiled Linux/Windows module,
which is why ``pip install robomaster`` fails on a Mac. The SDK uses it for:

    H264Decoder().decode(data) -> [(frame_bytes, width, height, linesize), ...]
    OpusDecoder().decode(data) -> pcm_bytes

This module provides the same API on top of PyAV (FFmpeg), which has native
Apple Silicon and Intel wheels. Frames are BGR24, like the original decoder,
so ``camera.read_cv2_image()`` returns normal OpenCV images.
"""

import logging

import av

logger = logging.getLogger("libmedia_codec")


class H264Decoder(object):
    """Decodes an H.264 byte stream delivered in arbitrary TCP chunks."""

    def __init__(self):
        self._codec = None
        self._reset()

    def _reset(self):
        self._codec = av.CodecContext.create("h264", "r")
        # Show frames as soon as they are decoded (the robot never sends B-frames).
        try:
            self._codec.flags2 |= av.codec.context.Flags2.fast
        except Exception:
            pass

    def decode(self, data):
        results = []
        if not data:
            return results
        try:
            packets = self._codec.parse(bytes(data))
        except Exception as error:
            logger.warning("H264Decoder: parse error %s, resetting decoder", error)
            self._reset()
            return results
        for packet in packets:
            try:
                frames = self._codec.decode(packet)
            except Exception as error:
                # Corrupt/partial packet (e.g. stream joined mid-GOP): skip it,
                # the decoder recovers at the next key frame.
                logger.debug("H264Decoder: skipped packet (%s)", error)
                continue
            for frame in frames:
                image = frame.to_ndarray(format="bgr24")
                height, width = image.shape[:2]
                results.append((image.tobytes(), width, height, width * 3))
        return results


class OpusDecoder(object):
    """Decodes the robot's Opus audio stream to 16-bit mono 48 kHz PCM."""

    def __init__(self, sample_rate=48000, channels=1):
        self._codec = av.CodecContext.create("opus", "r")
        self._codec.sample_rate = sample_rate
        try:
            self._codec.layout = "mono" if channels == 1 else "stereo"
        except Exception:
            pass
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)

    def decode(self, data):
        if not data:
            return b""
        pcm = bytearray()
        try:
            for frame in self._codec.decode(av.Packet(bytes(data))):
                for out in self._resampler.resample(frame):
                    pcm += out.to_ndarray().tobytes()
        except Exception as error:
            logger.debug("OpusDecoder: skipped packet (%s)", error)
        return bytes(pcm)
