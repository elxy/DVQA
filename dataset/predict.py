import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.dataset import CropSegment
from dataset.video import VideoAV


class VideoPredict(Dataset):
    r"""
    A predict dataset for a pair of videos

    args:
        video_ref (str): the path to the reference video
        video_dis (str): the path to the distorted video
        scene_nb_frames (int, optional): number of frames in each scene, 0 means auto detection
        channel (int, optional): number of channels of a sample
        size_x: horizontal dimension of a segment
        size_y: vertical dimension of a segment
        stride_x: horizontal stride between segments
        stride_y: vertical stride between segments
    """

    def __init__(self,
                 video_ref,
                 video_dis,
                 scene_nb_frames=100,
                 channel=1,
                 size_x=112,
                 size_y=112,
                 stride_x=80,
                 stride_y=80):

        self.video_ref = video_ref
        self.video_dis = video_dis

        self.channel = channel
        self.size_x = size_x
        self.size_y = size_y
        self.stride_x = stride_x
        self.stride_y = stride_y

        self.ref = VideoAV(video_ref)
        self.dis = VideoAV(video_dis)
        # check frame count and frame rate
        assert self.ref.frame_count == self.dis.frame_count and self.ref.frame_rate == self.dis.frame_rate

        self.frame_count = self.ref.frame_count
        self.frame_rate = self.ref.frame_rate
        if self.frame_rate <= 30:
            self.stride_t = 2
        elif self.frame_rate <= 60:
            self.stride_t = 4
        else:
            raise ValueError('Unsupported fps')

        self.scenes = [{
            'start': start,
            'end': min(start + scene_nb_frames - 1, self.frame_count - 1),
        } for start in range(0, self.frame_count, scene_nb_frames)]

        if max(self.ref.height, self.dis.height) >= 1080:
            self.frame_width = 1920
            self.frame_height = 1080
        else:
            self.frame_width = 1280
            self.frame_height = 720
        print(f'Calculated resolution is {self.frame_width}x{self.frame_height}.')

        self.ref_frames = self.ref.frames(width=self.frame_width, height=self.frame_height, format='gray')
        self.dis_frames = self.dis.frames(width=self.frame_width, height=self.frame_height, format='gray')

    def __getitem__(self, index):
        ref = self.load_scene(self.ref_frames,
                              self.frame_height,
                              self.frame_width,
                              **self.scenes[index],
                              stride_t=self.stride_t)
        dis = self.load_scene(self.dis_frames,
                              self.frame_height,
                              self.frame_width,
                              **self.scenes[index],
                              stride_t=self.stride_t)

        offset_v = (self.frame_height - self.size_y) % self.stride_y
        offset_t = int(offset_v / 4 * 2)
        offset_b = offset_v - offset_t
        offset_h = (self.frame_width - self.size_x) % self.stride_x
        offset_l = int(offset_h / 4 * 2)
        offset_r = offset_h - offset_l

        ref = ref[:, :, offset_t:self.frame_height - offset_b, offset_l:self.frame_width - offset_r]
        dis = dis[:, :, offset_t:self.frame_height - offset_b, offset_l:self.frame_width - offset_r]

        spatial_crop = CropSegment(self.size_x, self.size_y, self.stride_x, self.stride_y)
        ref = spatial_crop(ref)
        dis = spatial_crop(dis)

        return ref, dis

    def load_scene(self, frames, frame_height, frame_width, start, end, stride_t):
        r"""
        Load frames on-demand from raw video, currently supports only yuv420p

        args:
            file_path (str): path to yuv file
            frame_height
            frame_width
            stride_t (int): sample the 1st frame from every stride_t frames
            start (int): index of the 1st sampled frame
        return:
            ret (tensor): contains sampled frames (Y channel). dim = (C, D, H, W)
        """

        ret = []
        index, frame = next(frames)
        assert index == start

        def convert_frame(frame):
            frame = frame.astype('float32') / 255.
            frame = frame.reshape(1, 1, frame_height, frame_width)
            return frame

        ret.append(convert_frame(frame))
        while index < end:
            index, frame = next(frames)
            if index % stride_t == 0:
                ret.append(convert_frame(frame))

        ret = np.concatenate(ret, axis=1)
        ret = torch.from_numpy(ret)

        return ret

    def __len__(self):
        return len(self.scenes)
