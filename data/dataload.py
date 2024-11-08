"""
data loading & batch packing for MAPS, MAESTRO-V3.0.0
"""
import gc
import json
import math
from typing import Optional

from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import librosa
import torch
import os
import random
import pretty_midi
import joblib

from utils import func


def batch_loader_wav_midi_maestro_seq_single_frm(raw_wav_path: str = None,
                                                 midi_path: str = None,
                                                 json_path: str = None,
                                                 cache_path: str = None,
                                                 cache_piece: int = 30,
                                                 preprocess: str = 'mel',
                                                 time_freq_diagram: bool = False,
                                                 batch_size: int = 32,
                                                 frames_size: int = 32,
                                                 train_for_decoder: bool = False,
                                                 resample: int = 16000,
                                                 frame_length_ms: int = 100,
                                                 hop_length_ratio: float = 0.1,
                                                 n_fft: int = 2048,
                                                 n_mel: int = 512,
                                                 top_db: int = 80,
                                                 pedal_threshold: int = 64,
                                                 max_pedal_speed: float = None,
                                                 min_pedal_range: float = None,
                                                 maximum_pedal_acc: float = 0,
                                                 pedal_speed_calcu_extra_frm_range: int = 1,
                                                 note_minimum_frame: int = 0,
                                                 load_type: str = 'train',
                                                 with_train: bool = True,
                                                 schedule_idx: str = '28',
                                                 load_portion: float = 1.0,
                                                 load_offset: float = 0,
                                                 overlap_size: int = 0,
                                                 seed: int = 14,
                                                 num_workers: int = 0,
                                                 n_jobs: int = -4,
                                                 pin_memory=True,
                                                 dataset_name: str = 'maestro-v3.0.0'):
    """
    note on&off task data loading process ONLY for MAESTRO datasets.
    convert midi targets to midi-like form for Transformer training.
    return a torch.utils.data.Dataloader.
        support for loading from existing dataset or generating from .wav & .midi files;
        support for Mel_frames;
        support for subdirectory traversal;
        support for custom training task [onset&offset, onset, frame]
        need a pre-split .json file to split train, validation and test datasets
    """

    # load type check
    load_type_split = ['train', 'test', 'validation']
    if load_type not in load_type_split:
        raise ValueError(f'Invalid data split: {load_type}'
                         f', only support for train, test, validation.')

    # dataset check
    if dataset_name not in ['maestro-v1.0.0', 'maestro-v2.0.0', 'maestro-v3.0.0', 'maps']:
        raise ValueError(f'Unresolved Dataset: {dataset_name}')

    # preprocess check
    assert preprocess in {'mel', 'cqt', 'stft', 'vqt'}

    # print info
    if load_type == 'train':
        print('[Train Data]')
    elif load_type == 'test':
        print('[Test Data]')
    elif load_type == 'validation':
        print('[Validation Data]')

    print('Start dataset load...')
    # path
    if (raw_wav_path is None) or (midi_path is None) or (json_path is None) or (cache_path is None):
        raise RuntimeError('.wav or .mid or .json or cache path is NONE.')
    if not os.path.exists(cache_path[:-1]):
        os.makedirs(cache_path[:-1])

    # shuffle
    if load_type != 'train' or not with_train:
        seed = -1

    # construct cache name
    notes_name = '_'.join([dataset_name, load_type, preprocess, 'notes_cache.pth'])
    pedal_name = '_'.join([dataset_name, load_type, preprocess, 'pedal_cache.pth'])
    wav_name = '_'.join([dataset_name, load_type, preprocess, 'wav_cache.pth'])
    json_name = '.'.join([dataset_name, 'json'])
    file_list = os.listdir(cache_path)
    if [file.find(wav_name) for file in file_list].count(-1) != len(file_list):
        print(f'Load from existing cache...')
        cache_manager = DataCache(cache_path=cache_path, load_type=load_type, load_portion=load_portion,
                                  load_offset=load_offset, frames_size=frames_size, batch_shuffle=train_for_decoder,
                                  notes_name=notes_name, pedal_name=pedal_name, wav_name=wav_name,
                                  schedule_idx=schedule_idx, n_jobs=n_jobs)
        cache_dataset = CacheDataset(cache_manager=cache_manager, seed=seed,
                                     frames_size=frames_size, time_freq_diagram=time_freq_diagram,
                                     overlap_size=overlap_size)
        print('Succeed.')
    else:
        # util parameters
        frame_length_dots = int(resample * frame_length_ms * 1e-3)
        hop_length_dots = int(frame_length_dots * hop_length_ratio)

        wav_train_list, wav_test_list, wav_val_list = [], [], []
        midi_train_list, midi_test_list, midi_val_list = [], [], []
        root_sub_path = raw_wav_path

        if dataset_name == 'maps':
            a_tune = []
            a_file = {}
            for path_name, dir_names, file_names in os.walk(raw_wav_path):
                for file_name in file_names:
                    if file_name.endswith('.mid'):
                        a_file[file_name[:-4]] = path_name
            a_sorted_file = sorted(a_file.items())
            for file_name, path_name in a_sorted_file:
                file_info = path_name.lstrip(raw_wav_path).split('/')
                code = file_info[1]
                content = file_info[2]
                if (content == 'MUS') and ((code == 'ENSTDkAm') or (code == 'ENSTDkCl')):
                    wav_test_list.append(path_name + '/' + file_name + '.wav')
                    midi_path = path_name + '/' + file_name + '.midi'
                    if not os.path.exists(midi_path):
                        midi_path = path_name + '/' + file_name + '.mid'
                    midi_test_list.append(midi_path)
                    if file_name not in a_tune:
                        a_tune.append(file_name)
            for file_name, path_name in a_sorted_file:
                file_info = path_name.lstrip(raw_wav_path).split('/')
                code = file_info[1]
                content = file_info[2]
                if (content == 'MUS') and ((code != 'ENSTDkAm') or (code != 'ENSTDkCl')):
                    if file_name not in a_tune:
                        wav_train_list.append(path_name + '/' + file_name + '.wav')
                        midi_path = path_name + '/' + file_name + '.midi'
                        if not os.path.exists(midi_path):
                            midi_path = path_name + '/' + file_name + '.mid'
                        midi_train_list.append(midi_path)
                    else:
                        # the test split of the MAPS dataset is also the validation split
                        wav_val_list.append(path_name + '/' + file_name + '.wav')
                        midi_path = path_name + '/' + file_name + '.midi'
                        if not os.path.exists(midi_path):
                            midi_path = path_name + '/' + file_name + '.mid'
                        midi_val_list.append(midi_path)
        elif dataset_name in ['maestro-v1.0.0', 'maestro-v2.0.0', 'maestro-v3.0.0']:
            # load json file
            with open(os.path.join(json_path, json_name), 'r') as j:
                j_file = json.loads(j.read())
            if dataset_name == 'maestro-v3.0.0':
                for idx, split in j_file['split'].items():
                    if split == 'train':
                        wav_train_list.append(j_file['audio_filename'][idx])
                        midi_train_list.append(j_file['midi_filename'][idx])
                    elif split == 'test':
                        wav_test_list.append(j_file['audio_filename'][idx])
                        midi_test_list.append(j_file['midi_filename'][idx])
                    elif split == 'validation':
                        wav_val_list.append(j_file['audio_filename'][idx])
                        midi_val_list.append(j_file['midi_filename'][idx])
            elif dataset_name == 'maestro-v1.0.0' or dataset_name == 'maestro-v2.0.0':
                for info_dict in j_file:
                    if info_dict['split'] == 'train':
                        wav_train_list.append(info_dict['audio_filename'])
                        midi_train_list.append(info_dict['midi_filename'])
                    elif info_dict['split'] == 'test':
                        wav_test_list.append(info_dict['audio_filename'])
                        midi_test_list.append(info_dict['midi_filename'])
                    elif info_dict['split'] == 'validation':
                        wav_val_list.append(info_dict['audio_filename'])
                        midi_val_list.append(info_dict['midi_filename'])
        else:
            raise ValueError(f'Unresolved Dataset: {dataset_name}')

        # dataset
        wav_list = wav_train_list
        midi_list = midi_train_list
        if load_type == 'test':
            wav_list = wav_test_list
            midi_list = midi_test_list
        elif load_type == 'validation':
            wav_list = wav_val_list
            midi_list = midi_val_list

        mel_fb = librosa.filters.mel(sr=resample, n_fft=n_fft, n_mels=n_mel,
                                     fmin=librosa.midi_to_hz(20), fmax=resample / 2.0, norm=None)

        if preprocess == 'mel':
            print('Construct Mel dataset')
        elif preprocess == 'cqt':
            print('Construct CQT dataset')
        elif preprocess == 'vqt':
            print('Construct VQT dataset, using ERBlet transform')
        else:
            print('Construct STFT dataset')

        cache_total = math.ceil(len(wav_list) / cache_piece)
        cache_manager = DataCache(cache_path=cache_path, load_type=load_type, load_portion=load_portion,
                                  load_offset=load_offset, frames_size=frames_size, batch_shuffle=train_for_decoder,
                                  notes_name=notes_name, pedal_name=pedal_name, wav_name=wav_name,
                                  schedule_idx=schedule_idx, cache_total=cache_total, n_jobs=n_jobs)

        # loop over files in file list. cpu parallel available
        # reduce cache_piece to avoid memory overflow
        print(f'Total parallel tasks: {len(wav_list)}\n'
              f'Every {cache_piece} tasks saved as a cache\n'
              f'Total caches: {cache_total}')
        for i in range(cache_total):
            print(f'cache [{i + 1:4}/{cache_total:4}]')
            data_one_cache = {}
            cache_wav_list = wav_list[i * cache_piece:(i + 1) * cache_piece]
            cache_midi_list = midi_list[i * cache_piece:(i + 1) * cache_piece]

            data_total = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
                joblib.delayed(spectrum)(
                    wav_file=wav_file, midi_file=midi_file,
                    root_sub_path=root_sub_path, resample=resample, n_fft=n_fft, hop_length_dots=hop_length_dots,
                    frame_length_dots=frame_length_dots, mel_fb=mel_fb, top_db=top_db, preprocess=preprocess,
                    pedal_threshold=pedal_threshold,
                    max_pedal_speed=max_pedal_speed, min_pedal_range=min_pedal_range, maximum_acc=maximum_pedal_acc,
                    note_minimum_frame=note_minimum_frame,
                    pedal_speed_calcu_extra_frm_range=pedal_speed_calcu_extra_frm_range)
                for (wav_file, midi_file) in zip(cache_wav_list, cache_midi_list))

            for key in data_total[0].keys():
                data_one_cache[key] = []
            for data_pack in data_total:
                for key, value in data_pack.items():
                    data_one_cache[key].append(value)
            del data_total, data_pack

            # save data cache
            for key, value in data_one_cache.items():
                if len(value[0].shape) == 2:
                    data_one_cache[key] = np.vstack(value)
                else:
                    data_one_cache[key] = np.hstack(value)

            total_frames = data_one_cache['frame_roll'].shape[0]
            cache_manager.save_wav_midi_onset_offset(data_one_cache)
            cache_manager.count_sample(total_frames)

            del data_one_cache
        cache_manager.save_cache_info()

        print('\nSucceed & Saved cache.')
        
        cache_manager = DataCache(cache_path=cache_path, load_type=load_type, load_portion=load_portion,
                                  load_offset=load_offset, frames_size=frames_size, batch_shuffle=train_for_decoder,
                                  notes_name=notes_name, pedal_name=pedal_name, wav_name=wav_name,
                                  schedule_idx=schedule_idx, n_jobs=n_jobs)
        cache_dataset = CacheDataset(cache_manager=cache_manager, seed=seed,
                                     frames_size=frames_size, time_freq_diagram=time_freq_diagram,
                                     overlap_size=overlap_size)

    print(
        f'Loaded data:\n{cache_path}\n | {wav_name}\n | {notes_name}\n | {pedal_name}\n'
        f'Split: {load_type}\n'
        f'Load portion: {100 * (load_portion - load_offset):.2f}%\n'
        f'Batch size: {batch_size}\n'
        f'Frame size: {frames_size}\n')

    return DataLoader(cache_dataset, batch_size, pin_memory=pin_memory, num_workers=num_workers), cache_manager


def spectrum(wav_file, midi_file, root_sub_path, resample, n_fft, hop_length_dots, frame_length_dots, mel_fb, top_db,
             preprocess, pedal_threshold, max_pedal_speed,
             min_pedal_range, maximum_acc, note_minimum_frame, pedal_speed_calcu_extra_frm_range):
    """
    generate mel_spectrum & midi target for maestroV3 dataset
    """
    if (wav_file.split('.')[:-1]) == (midi_file.split('.')[:-1]):
        # load file_name.wav
        wav, fs = librosa.load(path=os.path.join(root_sub_path, wav_file), sr=None, mono=True)
        end_time = wav.shape[0] / fs
        wav_resample = librosa.resample(y=wav, orig_sr=fs, target_sr=resample)
        if preprocess == 'cqt' or preprocess == 'vqt':
            gamma = 0 if preprocess == 'cqt' else None
            wav_frames = librosa.vqt(y=wav_resample,
                                     sr=resample,
                                     hop_length=hop_length_dots,
                                     fmin=librosa.midi_to_hz(21),
                                     n_bins=88 * 4,
                                     bins_per_octave=12 * 4,
                                     window='hann',
                                     gamma=gamma,
                                     tuning=0.0)
            wav_frames = np.abs(wav_frames)
            wav_frames = librosa.amplitude_to_db(wav_frames, ref=np.max, top_db=top_db)
            wav_frames += int(top_db / 2)
        elif preprocess == 'mel':
            wav_frames = librosa.stft(y=wav_resample,
                                      n_fft=n_fft,
                                      hop_length=hop_length_dots,
                                      win_length=frame_length_dots,
                                      window='hann',
                                      center=True)  # col: frame number    row: mel number
            wav_frames = np.dot(mel_fb, np.abs(wav_frames))
            # dynamic range equals [-top_db, 0]
            wav_frames = librosa.amplitude_to_db(wav_frames, ref=np.max, top_db=top_db)
            # adjust dynamic range to [-top_db/2, top_db/2]
            wav_frames += int(top_db / 2)
        else:
            wav_frames = librosa.stft(y=wav_resample,
                                      n_fft=n_fft,
                                      hop_length=hop_length_dots,
                                      win_length=frame_length_dots,
                                      window='hann',
                                      center=True)  # col: frame number    row: mel number
            wav_frames = np.abs(wav_frames)
            wav_frames = wav_frames[:-1, ...]
            # dynamic range equals [-top_db, 0]
            wav_frames = librosa.amplitude_to_db(wav_frames, ref=np.max, top_db=top_db)
            # adjust dynamic range to [-top_db/2, top_db/2]
            wav_frames += int(top_db / 2)
        del wav_resample, wav

        # load file_name.mid
        pm_obj = pretty_midi.PrettyMIDI(os.path.join(root_sub_path, midi_file))
        target_prs = (
            func.get_piano_roll(pm_obj.instruments[0], fs=wav_frames.shape[-1] / end_time,
                                times=np.arange(0, end_time, end_time / wav_frames.shape[-1]),
                                pedal_threshold=pedal_threshold,
                                max_pedal_speed=max_pedal_speed, min_pedal_range=min_pedal_range,
                                pedal_speed_calcu_extra_frm_range=pedal_speed_calcu_extra_frm_range,
                                maximum_acc=maximum_acc, note_minimum_frame=note_minimum_frame))
        del pm_obj

        wav_frames = wav_frames.T
        # handle frame number mismatch
        prs_ = target_prs['frame_roll'].shape[0]
        wav_ = wav_frames.shape[0]
        if prs_ != wav_:
            if prs_ < wav_:
                for key, value in target_prs.items():
                    if len(value.shape) == 2:
                        insert_frame = np.zeros((wav_ - prs_, value.shape[-1]), dtype=np.float32)
                        target_prs[key] = np.append(value, insert_frame, axis=0)
                    else:
                        insert_frame = np.zeros(wav_ - prs_)
                        target_prs[key] = np.append(value, insert_frame, axis=0)
            else:
                insert_frame = np.ones((prs_ - wav_, wav_frames.shape[-1]), dtype=np.float32)
                insert_frame = insert_frame * int(- top_db / 2)
                wav_frames = np.append(wav_frames, insert_frame, axis=0)

            if target_prs['frame_roll'].shape[0] != wav_frames.shape[0]:
                raise RuntimeError(f'frame number mismatch. midi_frame: {target_prs["frame_roll"].shape[0]}, '
                                   f'while wav_frame: {wav_frames.shape[0]}')

        target_prs['wav_spectrum'] = wav_frames
        return target_prs
    else:
        raise KeyError(f'Unmatched file: {wav_file}->{midi_file}')


class DataCache(object):
    """
    handle multiple cache data save & load
    """

    def __init__(self, cache_path, load_type, load_portion, load_offset, frames_size, batch_shuffle, notes_name,
                 pedal_name, wav_name, schedule_idx, n_jobs, cache_total: int = 0, cache_idx: int = 1):
        self.load_type = load_type
        self.load_portion = load_portion
        self.load_offset = load_offset
        self.frames_size = frames_size
        self.batch_shuffle = batch_shuffle
        self.cache_path = cache_path
        self.cache_idx = cache_idx
        self.cache_total = int(cache_total)
        self.sample_total = []
        self.sample_portion_total = []
        self.sampler_list = []
        self.notes_name = notes_name
        self.pedal_name = pedal_name
        self.wav_name = wav_name
        self.schedule_idx = schedule_idx
        self.n_jobs = n_jobs

    def init_cache_length_info(self):
        """
        [load cache]
        check cache & sample total nums from local .pth file
        """
        self.refer_cache_num()
        self.refer_sample_num()

    def init_sampler(self, random_seed):
        """
        [load cache]
        generate a sample lookup table for training
        """
        if self.load_type == 'train':
            self.sample_portion_total = [int(sample_num * self.load_portion) - int(sample_num * self.load_offset)
                                         for sample_num in self.sample_total]
            self.sample_portion_total[-1] = (
                    int(sum(self.sample_total) * self.load_portion) - int(sum(self.sample_total) * self.load_offset)
                    - sum(self.sample_portion_total[:-1]))
            if random_seed != -1 and not self.batch_shuffle:
                random.seed(random_seed)
                for cache, sample_portion in enumerate(self.sample_portion_total):
                    self.sampler_list.append(
                        random.sample(range(int(self.sample_total[cache] * self.load_offset),
                                            sample_portion + int(self.sample_total[cache] * self.load_offset)),
                                      sample_portion))
            else:
                for cache, sample_portion in enumerate(self.sample_portion_total):
                    self.sampler_list.append(
                        list(range(int(self.sample_total[cache] * self.load_offset),
                                   sample_portion + int(self.sample_total[cache] * self.load_offset))))
                if self.batch_shuffle and random_seed != -1:
                    # cpu parallel shuffle
                    # sampler_list_pll = self._batch_shuffle_parallel(0, self.sampler_list[0])
                    sampler_list_pll = joblib.Parallel(n_jobs=self.n_jobs)(
                        joblib.delayed(self._batch_shuffle_parallel)(cache_idx, batch_list)
                        for (cache_idx, batch_list) in enumerate(self.sampler_list))
                    for cache_idx, shuffled_list in enumerate(sampler_list_pll):
                        self.sampler_list[cache_idx] = shuffled_list
                        self.sample_portion_total[cache_idx] = len(shuffled_list)
                    del sampler_list_pll, shuffled_list
        else:
            self.sample_portion_total = [int(sample_num * self.load_portion) - int(sample_num * self.load_offset)
                                         for sample_num in self.sample_total]
            self.sample_portion_total[-1] = (
                    int(sum(self.sample_total) * self.load_portion) - int(sum(self.sample_total) * self.load_offset)
                    - sum(self.sample_portion_total[:-1]))
            for cache, sample_portion in enumerate(self.sample_portion_total):
                self.sampler_list.append(list(range(int(self.sample_total[cache] * self.load_offset),
                                                    sample_portion + int(self.sample_total[cache] * self.load_offset))))
        assert (len(self.sampler_list[cache]) == self.sample_portion_total[cache] for cache in range(self.cache_total))

    def _batch_shuffle_parallel(self, cache_idx, batch_list):
        """
        [load cache]
        shuffle each sampler list on parallel
        """
        assert cache_idx < self.cache_total
        rng = np.random.default_rng()
        shifted_batch_list = np.array(batch_list)
        shifted_batch_list = shifted_batch_list[rng.integers(0, self.frames_size):]
        batch_num = math.ceil(len(shifted_batch_list) / self.frames_size)
        padding_num = (batch_num * self.frames_size - len(shifted_batch_list))
        shifted_batch_list = np.pad(shifted_batch_list, (0, padding_num), 'constant', constant_values=(-1, -1))
        batch_shuffle_list = np.array(np.split(np.array(shifted_batch_list), batch_num))
        rng.shuffle(batch_shuffle_list[:-1, :])
        batch_shuffle_list = batch_shuffle_list.reshape(-1)
        if padding_num > 0:
            batch_shuffle_list = batch_shuffle_list[:-padding_num]
        return batch_shuffle_list.tolist()

    def accumulate_cache_idx(self):
        """
        [load cache]
        accumulate cache_idx by 1 during training
        """
        self.cache_idx += 1
        if self.cache_idx > self.cache_total:
            return False
        return True

    def cache_reset(self):
        """
        [load cache]
        """
        self.cache_idx = 1

    def batch_reshuffle(self):
        """
        [load cache]
        """
        random.seed()
        del self.sampler_list
        gc.collect()
        self.sampler_list = []
        self.init_sampler(random.randint(1, 99999999))

    def save_wav_midi_onset_offset(self, data_one_cache):
        """
        [save cache]
        save cache as local .pth file
        """
        if self.cache_idx > self.cache_total:
            raise RuntimeError(f'index: {self.cache_idx} out of total range: {self.cache_total}')

        notes_dict = {}
        pedal_dict = {}
        wav_dict = {}
        notes_keys = ('onset_roll', 'reg_onset_roll', 'offset_roll', 'reg_offset_roll',
                      'offset_NDM_roll', 'reg_offset_NDM_roll', 'frame_roll', 'frame_NDM_roll',
                      'key_roll', 'key_offset_roll', 'reg_key_offset_roll', 'velocity_roll')
        pedal_keys = ('dp_roll', 'dp_NDM_roll', 'dp_onset_roll', 'dp_reg_onset_roll',
                      'dp_offset_roll', 'dp_reg_offset_roll', 'dp_offset_NDM_roll', 'dp_reg_offset_NDM_roll',
                      'sp_roll', 'sp_NDM_roll', 'sp_onset_roll', 'sp_reg_onset_roll',
                      'sp_offset_roll', 'sp_reg_offset_roll', 'sp_offset_NDM_roll', 'sp_reg_offset_NDM_roll',
                      'uc_roll', 'uc_onset_roll', 'uc_reg_onset_roll', 'uc_offset_roll', 'uc_reg_offset_roll')

        keys = list(data_one_cache.keys())
        for key in keys:
            if key in notes_keys:
                notes_dict[key] = torch.tensor(data_one_cache.pop(key))
            elif key in pedal_keys:
                pedal_dict[key] = torch.tensor(data_one_cache.pop(key))
            elif key == 'wav_spectrum':
                wav_dict[key] = torch.tensor(data_one_cache.pop(key))
            else:
                raise ValueError(f'Unknown dictionary keys: {key}')
        notes_name = '_'.join([self._convert_idx, self.notes_name])
        pedal_name = '_'.join([self._convert_idx, self.pedal_name])
        wav_name = '_'.join([self._convert_idx, self.wav_name])
        torch.save(notes_dict, self.cache_path + notes_name)
        torch.save(pedal_dict, self.cache_path + pedal_name)
        torch.save(wav_dict, self.cache_path + wav_name)

        # count cache pointer
        self.cache_idx += 1

    def count_sample(self, count):
        """
        [save cache]
        record each caches sample num
        """
        if self.cache_idx == 1:
            self.sample_total[0] = int(count)
        else:
            self.sample_total.append(int(count))

    def save_cache_info(self):
        """
        [save cache]
        save cache & sample total nums as local .pth file
        """
        cache_info_name = '_'.join(['length_info', self.wav_name])
        cache_info = self.sample_total
        cache_info.append(self.cache_total)
        cache_info = torch.LongTensor(cache_info)
        torch.save(cache_info, self.cache_path + cache_info_name)

    def refer_sample_num(self):
        """
        [load cache]
        load local cache length info .pth file & set total sample num
        """
        cache_info_name = '_'.join(['length_info', self.wav_name])
        self.sample_total = torch.load(self.cache_path + cache_info_name)[:-1].numpy().tolist()
        return sum(self.sample_total)

    def refer_cache_num(self):
        """
        [load cache]
        load local cache length info .pth file & set total cache num
        """
        cache_info_name = '_'.join(['length_info', self.wav_name])
        self.cache_total = torch.load(self.cache_path + cache_info_name)[-1].item()
        return self.cache_total

    def load_cache_to_memory(self):
        """
        [load cache]
        load wav & midi cache from local .pth file, one group at a time
        """
        file_list = os.listdir(self.cache_path)
        data_cache = {}
        wav_cache, notes_cache, pedal_cache = None, None, None
        for file in file_list:
            if ((file.find(self.load_type) != -1)
                    and (file.find(self._convert_idx) != -1)
                    and (file.find('wav_cache') != -1)):
                wav_cache = torch.load(self.cache_path + file)
                data_cache.update(wav_cache)
        if data_cache == {}:
            raise RuntimeError(f'could not find: {self._convert_idx}_{self.wav_name} file in path: {self.cache_path}')
        if self.schedule_idx[0] != 'x' and self.schedule_idx[1] != 'x':
            for file in file_list:
                if ((file.find(self.load_type) != -1)
                        and (file.find(self._convert_idx) != -1)
                        and (file.find('notes_cache') != -1)):
                    notes_cache = torch.load(self.cache_path + file)
                if ((file.find(self.load_type) != -1)
                        and (file.find(self._convert_idx) != -1)
                        and (file.find('pedal_cache') != -1)):
                    pedal_cache = torch.load(self.cache_path + file)
                if notes_cache is not None and pedal_cache is not None:
                    data_cache.update(notes_cache)
                    data_cache.update(pedal_cache)
                    return data_cache
        elif self.schedule_idx[0] != 'x':
            for file in file_list:
                if ((file.find(self.load_type) != -1)
                        and (file.find(self._convert_idx) != -1)
                        and (file.find('notes_cache') != -1)):
                    notes_cache = torch.load(self.cache_path + file)
                if notes_cache is not None:
                    data_cache.update(notes_cache)
                    return data_cache
        elif self.schedule_idx[1] != 'x':
            for file in file_list:
                if ((file.find(self.load_type) != -1)
                        and (file.find(self._convert_idx) != -1)
                        and (file.find('pedal_cache') != -1)):
                    pedal_cache = torch.load(self.cache_path + file)
                if pedal_cache is not None:
                    data_cache.update(pedal_cache)
                    return data_cache
        else:
            raise ValueError(f'At least to load one type of notes or pedal cache.')

    @property
    def _convert_idx(self):
        """
        [save & load cache]
        string handling func. convert cache_idx & cache_total to a string form, e.g.'00001of00005'
        """
        return ''.join([str(self.cache_idx).zfill(5), 'of', str(self.cache_total).zfill(5)])


class CacheDataset(TensorDataset):
    """
    a TensorDataset submits wav_mel, midi_target, sequence_target
    """

    def __init__(self, cache_manager: DataCache,
                 seed: int = -1, frames_size: int = 256, time_freq_diagram: bool = True,
                 overlap_size: Optional[int] = 0):
        super(CacheDataset, self).__init__()
        self.cache_manager = cache_manager
        self.cache_manager.init_cache_length_info()
        self.cache_manager.init_sampler(seed)
        self.frames_size = frames_size
        self.data_cache = cache_manager.load_cache_to_memory()
        self.time_freq_2D_diagram = time_freq_diagram
        self.is_first_batch_in_this_cache = True
        self.inner_cache_shift = 0
        self.overlap_size = overlap_size if overlap_size is not None else 0
        self.inner_cache_sample_pointer = 0

    def __getitem__(self, item):
        """
        submit a group of sample data. each sample includes multi-frames.
            automatically switch between different caches.
            generate sequence target on-the-fly.

        PS: DO NOT set "is_shuffle" in DataLoader, i.e. torch.utils.data.DataLoader(is_shuffle=False)
        """
        if self.time_freq_2D_diagram:
            next_cache_flag = False

            if self.is_first_batch_in_this_cache:
                self.inner_cache_sample_pointer = 0
                self.inner_cache_shift = ((item * self.frames_size) - sum(
                    self.cache_manager.sample_portion_total[:self.cache_manager.cache_idx - 1]))
                overlap_size = 0
                self.is_first_batch_in_this_cache = False
            else:
                overlap_size = self.overlap_size
                self.inner_cache_sample_pointer += 1

            cache_pointer_lower = (
                    (item * self.frames_size) - self.inner_cache_shift -
                    self.inner_cache_sample_pointer * overlap_size -
                    sum(self.cache_manager.sample_portion_total[:self.cache_manager.cache_idx - 1])
            )
            cache_pointer_upper = (
                    (item + 1) * self.frames_size - self.inner_cache_shift -
                    self.inner_cache_sample_pointer * overlap_size -
                    sum(self.cache_manager.sample_portion_total[:self.cache_manager.cache_idx])
            )

            if cache_pointer_lower < 0:
                self.cache_reset()
                self.inner_cache_sample_pointer = 0
                self.inner_cache_shift = ((item * self.frames_size) - sum(
                    self.cache_manager.sample_portion_total[:self.cache_manager.cache_idx - 1]))
                overlap_size = 0
                self.is_first_batch_in_this_cache = False
            if cache_pointer_upper >= 0:
                next_cache_flag = True

            sample_idx_in_cache_min = (
                    (item * self.frames_size) - self.inner_cache_shift -
                    self.inner_cache_sample_pointer * overlap_size -
                    sum(self.cache_manager.sample_portion_total[:self.cache_manager.cache_idx - 1])
            )
            sample_idx_in_cache_max = (
                    (item + 1) * self.frames_size - self.inner_cache_shift -
                    self.inner_cache_sample_pointer * overlap_size -
                    sum(self.cache_manager.sample_portion_total[:self.cache_manager.cache_idx - 1])
            )
            sample_idx_list = self.cache_manager.sampler_list[self.cache_manager.cache_idx - 1][
                              sample_idx_in_cache_min:sample_idx_in_cache_max]

            returned_data_dict = {}

            for key, value in self.data_cache.items():
                returned_data_dict[key] = value[sample_idx_list]
            if len(sample_idx_list) < self.frames_size:
                padding_frames_size = self.frames_size - len(sample_idx_list)
                for key, value in returned_data_dict.items():
                    if key == 'wav_spectrum':
                        returned_data_dict[key] = torch.cat(
                            (value, torch.min(value) * torch.ones(padding_frames_size, value.shape[1])), dim=0)
                    elif len(value.shape) == 2:
                        returned_data_dict[key] = torch.cat(
                            (value, torch.zeros(padding_frames_size, value.shape[1])), dim=0)
                    else:
                        returned_data_dict[key] = torch.cat(
                            (value, torch.zeros(padding_frames_size)), dim=0)

            if next_cache_flag:
                self.next_cache()
            return returned_data_dict

    def __len__(self):
        len_in_batch = [
            math.ceil((sample_num - self.frames_size) / (self.frames_size - self.overlap_size)) + 1
            for sample_num in self.cache_manager.sample_portion_total
        ]
        return sum(len_in_batch)

    def next_cache(self):
        if self.cache_manager.accumulate_cache_idx():
            del self.data_cache
            gc.collect()
            self.data_cache = self.cache_manager.load_cache_to_memory()
            self.is_first_batch_in_this_cache = True

    def cache_reset(self):
        self.cache_manager.cache_reset()
        del self.data_cache
        gc.collect()
        self.data_cache = self.cache_manager.load_cache_to_memory()
        self.is_first_batch_in_this_cache = True
