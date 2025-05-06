
import os
import random
import torch
import pandas as pd
from PIL import Image
from einops import rearrange
import numpy as np
import glob
from torchvideotransforms import video_transforms, volume_transforms

from dataloader.tools import Compose, _Reverse_seq
from utils.utils import get_max_len_128


class Parkinson_former(torch.utils.data.Dataset):
    def __init__(self, args, subset):

        self.subset = subset
        self.class_idx = int(args.class_idx)  # sport class index(from 0 begin)

        self.args = args
        self.max_len = get_max_len_128(self.class_idx)[0]
        self.clip_len = args.clip_len
        self.data_root = args.data_root
        self.traj_root = './data/detect_73_crop'
        self.rgb1x1_root = './data/RGB_1x1_fromLK'

        self.split_path = './data/train_41.csv'
        self.split_data = pd.read_csv(self.split_path)
        self.split = np.array(self.split_data)
        self.split = self.split[self.split[:, 4] == self.class_idx].tolist()  # stored nums are in str

        if self.subset == 'test':
            self.split_path_test = './data/test_41.csv'
            self.split_test = pd.read_csv(self.split_path_test)
            self.split_test = np.array(self.split_test)
            self.split_test = self.split_test[self.split_test[:, 4] == self.class_idx].tolist()

        if self.subset == 'test':
            self.dataset = self.split_test.copy()
        else: # sample 5 clips with different start frame for each video (Augmentation)
            self.dataset = self.split.copy()*5

        if self.args.partition: # 7 x 9 parts
            self.left_eye = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            self.left_brow= np.array([9, 10, 11, 12, 13, 14, 15, 16, 17])
            self.right_eye = np.array([53, 54, 55, 56, 57, 58, 59, 60, 61])
            self.right_brow = np.array([62, 63, 64, 65, 66, 67, 68, 69, 70])
            self.nose = np.array([38, 40, 41, 42, 44, 46, 50, 48, 47])
            self.bottom_lips = np.array([18, 20, 26, 23, 27, 24, 25, 22, 21 ])
            self.upper_lips = np.array([18, 29,  33,  27, 35, 36, 28, 32, 31])
            self.new_idx = np.concatenate((self.left_eye, self.left_brow, self.right_eye, self.right_brow, self.nose, self.bottom_lips, self.upper_lips), axis=-1)

    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        id_v, label, cls, num_frame = sample_1[0], sample_1[1], int(sample_1[4]), int(sample_1[2])
        patient_id, video_id = id_v.split('_')[-1], id_v[:11]

        data = {}
        id_path = os.path.join(self.data_root, patient_id, video_id)
        data['video'], frame_index = self.load_video(id_path, num_frame)
        traj_path = os.path.join(self.traj_root, patient_id, video_id + '-'+ patient_id + '.npy')
        rgb1x1_path = os.path.join(self.rgb1x1_root, patient_id, video_id + '-'+ patient_id + '.npy')

        data['traj'], index_t = self._traj_shape(traj_path, rgb1x1_path, num_frame, frame_index)
        data['final_score'] = label
        data['class'] = cls
        data['len'] = num_frame
        return data, index_t

    def __len__(self):
        return len(self.dataset)

    def car2pol(self, trajectories):
        # [t, n, 2]
        center_first_frame = trajectories[0, 52, :]#52
        new_traj = trajectories.copy()
        for t in range(trajectories.shape[0]):
            if t == 0:
                new_traj[t, :, :] = trajectories[t, :, :] - center_first_frame
            else:
                center_offset = trajectories[t, 52, :] - center_first_frame
                new_traj[t, :, :] = trajectories[t, :, :] - trajectories[t, 52, :] + center_offset
        return new_traj

    def _traj_shape(self, traj_path, rgb1x1_path, num_frame, frame_index):
        select_traj, index_t = self._load_tracked_traj(traj_path, rgb1x1_path, num_frame, self.max_len, frame_index)  # [clip_len, n, 2]
        select_traj = self._traj_transform(select_traj)
        select_traj = rearrange(select_traj, 't n p -> n t p', p=5)
        return select_traj, index_t

    def _load_tracked_traj(self, traj_path, rgb1x1_path, num_frame, max_len, frame_index):
        select_traj = np.load(traj_path, allow_pickle=True)
        RGB_values = np.load(rgb1x1_path, allow_pickle=True)

        if select_traj.shape[1] ==73:
            select_traj = np.delete(select_traj, 1, 1)  # delete index 1, left eye centre
            select_traj = np.delete(select_traj, 54, 1)  # delete index 55, right eye centre
        elif select_traj.shape[1]==74:
            select_traj = select_traj[:, 1:, :2]
            select_traj = np.delete(select_traj, 1, 1)  # delete index 1, left eye centre
            select_traj = np.delete(select_traj, 54, 1)  # delete index 55, right eye centre
        else:
            pass

        select_traj = self.car2pol(select_traj) #52 center
        select_traj = np.concatenate((select_traj, RGB_values), axis=-1)

        if self.args.partition:
            select_traj = select_traj[:, self.new_idx, ...]

        tracked_traj = torch.from_numpy(select_traj)  # [t, n, 2]

        assert tracked_traj.shape[
                   0] == num_frame, f"the number of tracked frames:{tracked_traj.shape[0]} in {traj_path} must be equal to num_frames:{num_frame}"
        # select_traj = torch.stack(([tracked_traj[i - 1] for i in range(1, num_frame+1)]), dim=0)  # [clip_len, n, 2]
        
        if num_frame < max_len:
            if self.args.pad_type == 'loop':
                idx = [i % num_frame for i in range(max_len)]
                tracked_traj = tracked_traj[idx]
                idx_np = np.array(idx)
                index_t = 2 * idx_np.astype(np.float32) / max_len - 1
            else:
                pass
        else:
            if self.subset == 'train':
                if frame_index is None:
                    start_idx = num_frame - max_len
                    rand_start = random.randint(0, start_idx)
                    tracked_traj = tracked_traj[rand_start:rand_start+max_len]  # random consecutive seq.
                    idx = np.linspace(rand_start, rand_start+max_len - 1, max_len).astype(np.int32)
                    index_t = 2 * idx.astype(np.float32) / max_len - 1
                else:
                    if self.args.long_pad_traj=='latter_truncate':
                        start_idx = num_frame - max_len
                        tracked_traj = tracked_traj[start_idx:] # take the latter part of the sequence
                        # print(tracked_traj.shape)
                        idx = np.linspace(start_idx, num_frame - 1, max_len).astype(np.int32)
                        index_t = 2 * idx.astype(np.float32) / max_len - 1

            elif self.subset == 'test':
                    start_idx = num_frame - max_len
                    if self.args.long_pad_traj=='truncate':
                        tracked_traj = tracked_traj[:max_len] # take the former part of the sequence
                        idx = np.linspace(0, max_len - 1, max_len).astype(np.int32)
                        index_t = 2 * idx.astype(np.float32) / max_len - 1
                    elif self.args.long_pad_traj=='latter_truncate':
                        tracked_traj = tracked_traj[start_idx:] # take the latter part of the sequence
                        # print(tracked_traj.shape)
                        idx = np.linspace(start_idx, num_frame - 1, max_len).astype(np.int32)
                        index_t = 2 * idx.astype(np.float32) / max_len - 1
                    else:
                        pass
        
            else:
                pass
        
        return tracked_traj, index_t

    def _traj_transform(self, landmarks):
        if self.subset == 'train':

            augment_string = self.args.data_augment
            aug_dict = {
                '0': 'none',
                '1': 'reverse',
            }

            aug_list = []

            for char in augment_string:
                if char in aug_dict:
                    aug_type = aug_dict[char]
                    if aug_type == 'reverse':
                        aug_list.append(_Reverse_seq())
                    elif aug_type == 'none':
                        pass
            transform = Compose(aug_list, self.args.k)
            transformed_landmarks = transform(landmarks.numpy())
            return torch.from_numpy(transformed_landmarks.copy())
        else:
            return landmarks

    def load_short_clips(self, video_frames_list, clip_len, num_frames):
        video_clip = []
        idx = 0
        start_frame = 1
        sample_rate = 1
        frame_index = []
        for i in range(clip_len):
            cur_img_index = start_frame + idx * sample_rate
            frame_index.append(cur_img_index)
            if (start_frame + (idx + 1) * sample_rate) > num_frames:
                start_frame = 1
                idx = 0
            else:
                idx += 1
        imgs = [Image.open(video_frames_list[i-1]).convert('RGB') for i in frame_index]
        video_clip.extend(imgs)
        return video_clip, frame_index

    def load_short_clips_zero(self, video_frames_list, clip_len, num_frames):
        video_clip = []
        imgs = [Image.open(video_frames_list[i]).convert('RGB') for i in range(num_frames)]
        video_clip.extend(imgs)
        pad_length = clip_len - num_frames
        # Create a black image for padding (assuming the frame size is the same as the first frame)
        frame_size = imgs[0].size
        black_image = Image.new('RGB', frame_size, (0, 0, 0))
        # Add the black images to the video clip
        for _ in range(pad_length):
            video_clip.append(black_image)
        frame_index = [i + 1 for i in range(num_frames)]

        return video_clip, frame_index

    def load_long_clips(self, video_frames_list, clip_len, num_frames):
        video_clip = []
        if self.subset == 'train':
            start_frame = random.randint(1, num_frames - clip_len)
            frame_index = [i for i in range(start_frame, start_frame + clip_len)]
            # print(num_frames, 'index:', frame_index)
            imgs = [Image.open(video_frames_list[i - 1]).convert('RGB') for i in
                    range(start_frame, start_frame + clip_len)]
        elif self.subset == 'test':  
            if self.args.long_pad_traj=='random_avg' or self.args.long_pad_traj=='random_avg_new':
                start_frame = random.randint(1, num_frames - clip_len)
                frame_index = [i for i in range(start_frame, start_frame + clip_len)]
                # print(num_frames, 'index:', frame_index)
                imgs = [Image.open(video_frames_list[i - 1]).convert('RGB') for i in
                    range(start_frame, start_frame + clip_len)]
            else:# sample evenly spaced frames across the sequence for inference
                frame_partition = np.linspace(0, num_frames - 1, num=clip_len, dtype=np.int32)
                frame_index = [i + 1 for i in frame_partition]
                # print(num_frames, 'index:', frame_index)
                imgs = [Image.open(video_frames_list[i]).convert('RGB') for i in frame_partition]
        else:
            assert f"subset must be train or test"

        video_clip.extend(imgs)
        return video_clip, frame_index

    def load_video(self, path, num_frame):
        video_frames_list = sorted((glob.glob(os.path.join(path, '*.jpg'))))
        assert video_frames_list != None, f"check the video dir"
        assert len(
            video_frames_list) == num_frame, f"the number of imgs:{len(video_frames_list)} in {path} must be equal to num_frames:{num_frame}"
            # if clip length <= input length
        if len(video_frames_list) <= self.clip_len:
            if self.args.rgb_pad == 'zero':
                video, frame_index = self.load_short_clips_zero(video_frames_list, self.clip_len, num_frame)
            else:
                video, frame_index = self.load_short_clips(video_frames_list, self.clip_len, num_frame)
        else:
                video, frame_index = self.load_long_clips(video_frames_list, self.clip_len, num_frame)
        return self.transform(video), frame_index

    def transform(self, video):
        trans = []
        if self.subset == 'train':
                trans = video_transforms.Compose([
                    video_transforms.RandomHorizontalFlip(),
                    video_transforms.RandomCrop(224),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        elif self.subset == 'test':
                trans = video_transforms.Compose([
                    video_transforms.Resize((224, 224)),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        return trans(video)

def train_data_loader(args):
    train_data = Parkinson_former(args, subset='train')
    return train_data

def test_data_loader(args):
    test_data = Parkinson_former(args,subset='test')
    return test_data




