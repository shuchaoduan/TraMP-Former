import math
import os
import pickle as p

from einops import rearrange

# path_to_protocol5 = '/Users/alex/Code/dlc_playingdata/MultiMouse-Daniel-2019-12-16/training-datasets/iteration-1/UnaugmentedDataSet_MultiMouseDec16/Documentation_data-MultiMouse_95shuffle0.pickle'

# import pickle5 as p
import random

import cv2
import scipy
import torch
import pandas as pd
from PIL import Image

from scipy import stats
import numpy as np
import glob
import torchvision
from sklearn.model_selection import train_test_split
from torchvideotransforms import video_transforms, volume_transforms

from dataloader.tools import Compose, _Reverse_seq
from utils.utils import get_max_len_128

class NeuroFace_former(torch.utils.data.Dataset):
    def __init__(self, args, subset):

        self.subset = subset
        self.args = args
        self.fold = args.fold
        self.clip_len = args.clip_len
        self.max_len = args.traj_len
        self.data_root = args.data_root
        self.traj_root = args.landmarks_root
        self.rgb1x1_root = args.rgb1x1_root

        self.data_path = './NeuroFace/split_6tasks_sum.csv'
        self.data = pd.read_csv(self.data_path)
        self.data = np.array(self.data)
        subtask_dict = {
            '1': 'BIGSMILE', '2': 'LIPS', '3': 'BROW',
            '4': 'OPEN', '5': 'SPREAD'}
        self.class_idx = subtask_dict[str(args.class_idx)]
        # Merge logic: Combine BLOW and KISS under LIPS
        if self.class_idx == 'LIPS':
            # Filter rows where column 3 contains either 'BLOW' or 'KISS'
            self.split = self.data[(self.data[:, 3] == 'BLOW') | (self.data[:, 3] == 'KISS')]
        else:
            # Standard filtering for other classes
            self.split = self.data[self.data[:, 3] == self.class_idx]

        self.load_fold(self.fold)

        if self.args.partition: # 7 x 9 parts
            self.left_eye = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            self.left_brow= np.array([9, 10, 11, 12, 13, 14, 15, 16, 17])
            self.right_eye = np.array([53, 54, 55, 56, 57, 58, 59, 60, 61])
            self.right_brow = np.array([62, 63, 64, 65, 66, 67, 68, 69, 70])
            self.nose = np.array([38, 40, 41, 42, 44, 46, 50, 48, 47])
            self.bottom_lips = np.array([18, 20, 26, 23, 27, 24, 25, 22, 21 ])
            self.upper_lips = np.array([18, 29,  33,  27, 35, 36, 28, 32, 31])
            self.new_idx = np.concatenate((self.left_eye, self.left_brow, self.right_eye, self.right_brow, self.nose, self.bottom_lips, self.upper_lips), axis=-1)

    def load_fold(self, fold):

        cv_dict = {
            '1': ['N002', 'OP03', 'A009', 'A016', 'S007 ', 'N001', 'S002', 'N008', 'N010', 'N004', 'N012', 'S005'],
            '2': ['A008', 'N011', 'S013', 'N019', 'A017', 'A014', 'A011', 'A015', 'OP02', 'A012', 'S011', 'N003'],
            '3': ['A010', 'S008', 'S001', 'S009', 'N017', 'N007', 'S012', 'A006', 'A002', 'OP01', 'S006', 'S003']
        }

        self.dataset = []
        subject_IDs = cv_dict[str(fold)]
        self.split_test = self.data[np.isin(self.data[:, 1], subject_IDs)].tolist()
        self.split_train = self.data[~np.isin(self.data[:, 1], subject_IDs)].tolist()

        if self.subset == 'test':
            self.dataset = self.split_test.copy()
        else: # sample 5 clips with different start frame for each video (Augmentation)
            self.dataset = self.split_train.copy()*5
        return self.dataset

    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        vid, sid, s_type, subtask, rep_ID, num_frame = sample_1[0], sample_1[1], sample_1[2], sample_1[3], str(sample_1[4]), int(sample_1[5])
        symmetry, rom, speed, variability, fatigue, sum = sample_1[8], sample_1[9], sample_1[10], sample_1[11], sample_1[12], sample_1[13]
        id_path = os.path.join(self.data_root, s_type, subtask, vid, rep_ID)
        traj_path = os.path.join(self.traj_root, s_type, subtask, vid, vid +'_'+ rep_ID+ '.npy')
        rgb1x1_path = os.path.join(self.rgb1x1_root,  s_type, subtask, vid, vid +'_'+ rep_ID+ '.npy')

        data = {}
        data['symmetry'] = symmetry
        data['ROM'] = rom
        data['speed'] = speed
        data['variability'] = variability
        data['fatigue'] = fatigue
        data['final_score'] = sum
        data['len'] = num_frame
        data['video'], frame_index = self.load_video(id_path, num_frame)
        data['traj'], index_t = self._traj_shape(traj_path, rgb1x1_path, num_frame, frame_index)

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
            padding_needed = max_len - num_frame
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
                    if self.args.long_pad_traj=='latter_truncate':
                        tracked_traj = tracked_traj[start_idx:] # take the latter part of the sequence
                        # print(tracked_traj.shape)
                        idx = np.linspace(start_idx, num_frame - 1, max_len).astype(np.int32)
                        index_t = 2 * idx.astype(np.float32) / max_len - 1
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
        frame_index.append(True)
        return video_clip, frame_index

    def load_long_clips(self, video_frames_list, clip_len, num_frames):
        video_clip = []
        if self.subset == 'train':
            start_frame = random.randint(1, num_frames - clip_len)
            frame_index = [i for i in range(start_frame, start_frame + clip_len)]
            # print(num_frames, 'index:', frame_index)
            imgs = [Image.open(video_frames_list[i - 1]).convert('RGB') for i in
                    range(start_frame, start_frame + clip_len)]
        elif self.subset == 'test':  # sample evenly spaced frames across the sequence for inference
            frame_partition = np.linspace(0, num_frames - 1, num=clip_len, dtype=np.int32)
            frame_index = [i + 1 for i in frame_partition]
            # print(num_frames, 'index:', frame_index)
            imgs = [Image.open(video_frames_list[i]).convert('RGB') for i in frame_partition]
        else:
            assert f"subset must be train or test"
        video_clip.extend(imgs)
        return video_clip, frame_index

    def load_video(self, path, num_frame, frame_index=None):
        if frame_index is None:
            video_frames_list = sorted((glob.glob(os.path.join(path, '*.jpg'))))
        else:
            video_frames_list = sorted((glob.glob(os.path.join(path, '*.jpg'))))
        assert video_frames_list != None, f"check the video dir"
        assert len(
            video_frames_list) == num_frame, f"the number of imgs:{len(video_frames_list)} in {path} must be equal to num_frames:{num_frame}"
        if frame_index is None:
            # if clip length <= input length
            if len(video_frames_list) <= self.clip_len:
                video, frame_index = self.load_short_clips(video_frames_list, self.clip_len, num_frame)
            else:
                video, frame_index = self.load_long_clips(video_frames_list, self.clip_len, num_frame)
            return self.transform(video), frame_index
        else:
            video = []
            if frame_index[-1]:
                
                imgs = [Image.open(video_frames_list[i-1]).convert('RGB') for i in frame_index[:-1]]
            else:
                imgs = [Image.open(video_frames_list[i]).convert('RGB') for i in frame_index[:-1]]
            video.extend(imgs)
            return self.transform(video, use_landmark=True)

    def transform(self, video, use_landmark=False):
        trans = []

        if self.subset == 'train':
            trans = video_transforms.Compose([
                video_transforms.RandomHorizontalFlip(),
                # video_transforms.Resize((256, 256)),# input 256x256
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
    train_data = NeuroFace_former(args, subset='train')
    return train_data

def test_data_loader(args):
    test_data = NeuroFace_former(args, subset='test')
    return test_data

