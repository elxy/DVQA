import av
import av.error
import av.video

import cv2


class VideoCV:

    def __init__(self, video):
        self.cap = cv2.VideoCapture(video)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

    def frames(self, downscale_factor: int = 1):
        index = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield index, frame[::downscale_factor, ::downscale_factor, :]
            index += 1


class VideoAV:

    def __init__(self, video):
        self.video = video

        container = av.open(video)
        stream = container.streams.video[0]
        self.width = stream.codec_context.width
        self.height = stream.codec_context.height
        self._frame_count = stream.frames

        def _get_frame_rate(stream: av.video.stream.VideoStream):
            if stream.average_rate and stream.average_rate.denominator and stream.average_rate.numerator:
                return float(stream.average_rate)
            if stream.base_rate and stream.base_rate.denominator and stream.base_rate.numerator:
                return float(stream.base_rate)
            if stream.guessed_rate and stream.guessed_rate.denominator and stream.guessed_rate.numerator:
                return float(stream.guessed_rate)
            if stream.time_base.denominator and stream.time_base.numerator:
                return 1.0 / float(stream.time_base)
            else:
                raise ValueError("Unable to determine FPS")

        self.frame_rate = _get_frame_rate(stream)
        container.close()

    def frames(self, **reformat_kwargs):
        container = av.open(self.video)
        for index, frame in enumerate(container.decode(video=0)):
            yield index, frame.to_ndarray(**reformat_kwargs)
        container.close()

    @property
    def frame_count(self):
        if self._frame_count == 0:
            index = 0
            container = av.open(self.video)
            try:
                for index, _ in enumerate(container.decode(video=0)):
                    continue
            except av.error.EOFError:
                pass
            container.close()
            self._frame_count = index + 1
        return self._frame_count
