"""
formal full-music-piece based evaluation
"""
import json
from collections import OrderedDict
from typing import Union

import joblib
import librosa
import pretty_midi
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import os

from constant_macro import *
from utils import func, metric
import checkpoint
from net import mt_film


def init_model(model, model_path, ckp, on_cpu=False):
    pretrained_info = checkpoint.resume(model_path, ckp)
    epoch = pretrained_info.train_epoch
    train_time = pretrained_info.train_time
    model_name = pretrained_info.model_name
    if on_cpu:
        loaded_dict = pretrained_info.model_dict
        cpu_dict = OrderedDict()
        for k, v in loaded_dict.items():
            name = k
            if 'module.' in k:
                name = k[7:]  # remove `module.`
            cpu_dict[name] = v
        model.load_state_dict(
            checkpoint.reload_dict(model.state_dict(), cpu_dict))
        model.eval()
        return model, model_name, torch.device('cpu'), epoch, train_time

    else:
        device_ids = [i for i in range(0, torch.cuda.device_count())]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        if (device.type != 'cpu') and (torch.cuda.device_count() > 1):
            model = nn.DataParallel(model, device_ids=device_ids)

        model.load_state_dict(checkpoint.reload_dict(model.state_dict(), pretrained_info.model_dict))
        model.eval()
        return model, model_name, device, epoch, train_time


def convert_to_spectrum(audio_path=None, preprocess: Union[str, list] = 'cqt',
                        with_ground_truth=True, wav=None, fs=None):
    resample = 16000
    n_fft = 2048
    n_mel = 229
    frame_length_ms = 128
    hop_length_ratio = 0.15625
    top_db = 80

    frame_length_dots = int(resample * frame_length_ms * 1e-3)
    hop_length_dots = int(frame_length_dots * hop_length_ratio)
    mel_fb = librosa.filters.mel(sr=resample, n_fft=n_fft, n_mels=n_mel,
                                 fmin=librosa.midi_to_hz(20), fmax=resample / 2.0, norm=None)
    if wav is None and fs is None:
        wav, fs = librosa.load(path=audio_path, sr=None, mono=True)
    end_time = wav.shape[0] / fs
    wav_resample = librosa.resample(y=wav, orig_sr=fs, target_sr=resample)
    wav_frames = {}
    if 'cqt' in preprocess:
        cqt = librosa.vqt(y=wav_resample,
                          sr=resample,
                          hop_length=hop_length_dots,
                          fmin=librosa.midi_to_hz(21),
                          n_bins=88 * 4,
                          bins_per_octave=12 * 4,
                          window='hann',
                          gamma=0,
                          tuning=0.0)
        cqt = np.abs(cqt)
        cqt = librosa.amplitude_to_db(cqt, ref=np.max, top_db=top_db)
        cqt += int(top_db / 2)
        wav_frames['cqt'] = cqt
    if 'vqt' in preprocess:
        vqt = librosa.vqt(y=wav_resample,
                          sr=resample,
                          hop_length=hop_length_dots,
                          fmin=librosa.midi_to_hz(21),
                          n_bins=88 * 4,
                          bins_per_octave=12 * 4,
                          window='hann',
                          gamma=None,
                          tuning=0.0)
        vqt = np.abs(vqt)
        vqt = librosa.amplitude_to_db(vqt, ref=np.max, top_db=top_db)
        vqt += int(top_db / 2)
        wav_frames['vqt'] = vqt
    if 'mel' in preprocess:
        mel = librosa.stft(y=wav_resample,
                           n_fft=n_fft,
                           hop_length=hop_length_dots,
                           win_length=frame_length_dots,
                           window='hann',
                           center=True)  # col: frame number    row: mel number
        mel = np.dot(mel_fb, np.abs(mel))
        mel = librosa.amplitude_to_db(mel, ref=np.max, top_db=top_db)
        mel += int(top_db / 2)
        wav_frames['mel'] = mel
    if 'stft' in preprocess:
        stft = librosa.stft(y=wav_resample,
                            n_fft=n_fft,
                            hop_length=hop_length_dots,
                            win_length=frame_length_dots,
                            window='hann',
                            center=True)  # col: frame number    row: mel number
        stft = np.abs(stft)
        stft = stft[:-1, ...]
        stft = librosa.amplitude_to_db(stft, ref=np.max, top_db=top_db)
        stft += int(top_db / 2)
        wav_frames['stft'] = stft
    assert wav_frames

    for key_wav, value_wav in wav_frames.items():
        wav_frames[key_wav] = value_wav.T

    spectrum = wav_frames[next(iter(wav_frames))]
    total_frm = spectrum.shape[0]
    fs = total_frm / end_time
    if with_ground_truth:
        pedal_threshold = 64
        max_pedal_speed = 0
        min_pedal_range = 1
        pedal_speed_calcu_extra_frm_range = 2
        maximum_acc = 0
        note_minimum_frame = 40

        midi_path = ''.join([audio_path[:-3], 'midi'])
        if not os.path.exists(midi_path):
            midi_path = ''.join([audio_path[:-3], 'mid'])
        pm_obj = pretty_midi.PrettyMIDI(midi_path)

        target_prs = (
            func.get_piano_roll(pm_obj.instruments[0], fs=fs,
                                times=np.arange(0, end_time, end_time / total_frm),
                                pedal_threshold=pedal_threshold,
                                max_pedal_speed=max_pedal_speed, min_pedal_range=min_pedal_range,
                                pedal_speed_calcu_extra_frm_range=pedal_speed_calcu_extra_frm_range,
                                maximum_acc=maximum_acc, note_minimum_frame=note_minimum_frame))

        # handle frame number mismatch
        for key_wav, value_wav in wav_frames.items():
            prs_ = target_prs['frame_roll'].shape[0]
            wav_ = wav_frames[key_wav].shape[0]
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
                    insert_frame = np.ones((prs_ - wav_, wav_frames[key_wav].shape[-1]), dtype=np.float32)
                    insert_frame = insert_frame * int(- top_db / 2)
                    wav_frames[key_wav] = np.append(wav_frames[key_wav], insert_frame, axis=0)

        for key_wav, value_wav in wav_frames.items():
            wav_frames[key_wav] = torch.tensor(value_wav)
        for key, value in target_prs.items():
            target_prs[key] = torch.tensor(value)

        return wav_frames, target_prs

    for key, value in wav_frames.items():
        wav_frames[key] = torch.tensor(value.T)
    return wav_frames, None


def clip_stride_separation(spectrum, block_stride=0.5):
    block_num, block_len, frequency_bins = spectrum.shape
    edge_len = int(block_len * (1 - block_stride) / 2.)
    stride_len = int(block_len * block_stride)
    assert block_len % stride_len == 0
    spectrum = spectrum.reshape(-1, frequency_bins)
    spectrum = spectrum.reshape(-1, stride_len, frequency_bins)

    folds = int(np.ceil(edge_len / stride_len))
    rest_edge_len = edge_len - stride_len
    front_edge = spectrum[1:, :edge_len, :]
    for f_idx in range(2, folds + 1):
        temp_edge = spectrum[f_idx:, :min(stride_len, rest_edge_len), :]
        temp_edge = torch.cat(
            [temp_edge, torch.zeros((f_idx - 1, temp_edge.shape[1], frequency_bins), device=spectrum.device)], dim=0)
        front_edge = torch.cat([front_edge, temp_edge], dim=1)
        rest_edge_len -= stride_len
    front_edge = torch.cat([front_edge, torch.zeros((1, edge_len, frequency_bins), device=spectrum.device)], dim=0)

    rest_edge_len = edge_len - stride_len
    rear_edge = spectrum[:-1, -edge_len:, :]
    for f_idx in range(2, folds + 1):
        temp_edge = spectrum[:-f_idx, -min(stride_len, rest_edge_len):, :]
        temp_edge = torch.cat(
            [torch.zeros((f_idx - 1, temp_edge.shape[1], frequency_bins), device=spectrum.device), temp_edge], dim=0)
        rear_edge = torch.cat([temp_edge, rear_edge], dim=1)
        rest_edge_len -= stride_len
    rear_edge = torch.cat([torch.zeros((1, edge_len, frequency_bins), device=spectrum.device), rear_edge], dim=0)
    spectrum = torch.cat([rear_edge, spectrum, front_edge], dim=1)
    return spectrum


def clip_stride_recombination(midi_pr, block_len, block_stride=0.5):
    if len(midi_pr.shape) == 3:
        _, stride_len, frequency_bins = midi_pr.shape
        edge_len = int(block_len * (1 - block_stride) / 2.)
        assert block_len % stride_len == 0
        midi_pr = midi_pr[:, edge_len:-edge_len, :]
        midi_pr = midi_pr.reshape(-1, frequency_bins)
    else:
        _, stride_len, frequency_bins, vel_classes = midi_pr.shape
        edge_len = int(block_len * (1 - block_stride) / 2.)
        assert block_len % stride_len == 0
        midi_pr = midi_pr[:, edge_len:-edge_len, :, :]
        midi_pr = midi_pr.reshape(-1, frequency_bins, vel_classes)
    return midi_pr


def piano_transcription(model, spectrum, device, block_stride=0.5, schedule_idx='28', evaluate_term='note'):
    top_k = 2
    spectrum = spectrum.to(device)
    frames, frequency_bins = spectrum.shape
    block_len = 400
    pad = (0, 0, 0, block_len - (frames % block_len) if (frames % block_len) != 0 else 0)
    spectrum = F.pad(spectrum, pad)
    spectrum = spectrum.reshape(-1, block_len, frequency_bins)
    spectrum = clip_stride_separation(spectrum, block_stride=block_stride)
    midi_out = {
        'onset_pred': None,
        'frame_pred': None,
        'offset_pred': None,
        'velocity_pred': None
    }
    for clip_idx in range(0, spectrum.shape[0], 2):
        inp = spectrum[clip_idx:clip_idx + 2, ...]
        with torch.no_grad():
            out_prs = func.transcribe_one_batch(model, inp, schedule_idx, evaluate_term)
            frame_pred = out_prs['frame']
            onset_pred = out_prs['onset']
            offset_pred = out_prs['offset']
            velocity_pred = out_prs['velocity']
            if len(velocity_pred.shape) == 4 and velocity_pred.shape[-1] != 128:
                velocity_pred = velocity_pred.permute(0, 2, 3, 1)

        midi_out['onset_pred'] = (
            onset_pred.cpu() if midi_out['onset_pred'] is None
            else torch.cat([midi_out['onset_pred'], onset_pred.cpu()], dim=0)
        )
        midi_out['frame_pred'] = (
            frame_pred.cpu() if midi_out['frame_pred'] is None
            else torch.cat([midi_out['frame_pred'], frame_pred.cpu()], dim=0)
        )
        midi_out['offset_pred'] = (
            offset_pred.cpu() if midi_out['offset_pred'] is None
            else torch.cat([midi_out['offset_pred'], offset_pred.cpu()], dim=0)
        )
        if len(velocity_pred.shape) == 4:
            velocity_pred = torch.topk(velocity_pred, k=top_k).indices
        midi_out['velocity_pred'] = (
            velocity_pred.cpu() if midi_out['velocity_pred'] is None
            else torch.cat([midi_out['velocity_pred'], velocity_pred.cpu()], dim=0)
        )

    for key, value in midi_out.items():
        value = clip_stride_recombination(value, block_len=block_len, block_stride=block_stride)
        midi_out[key] = value[:frames, ...]

    return midi_out


def init_evaluate_list(dataset_name='maestro-v3.0.0', raw_wav_path="/data/MaestroV3-dataset/maestro-v3.0.0/"):
    load_type = 'test'
    preprocess = 'cqt'
    midi_path = raw_wav_path
    json_path = raw_wav_path

    # load type check
    load_type_split = ['train', 'test', 'validation']
    if load_type not in load_type_split:
        raise ValueError(f'Invalid data split: {load_type}'
                         f', only support for train, test, validation.')

    # dataset check
    if dataset_name not in ['maestro-v1.0.0', 'maestro-v2.0.0', 'maestro-v3.0.0']:
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

    # path
    if (raw_wav_path is None) or (midi_path is None) or (json_path is None):
        raise RuntimeError('.wav or .mid or .json is NONE.')

    json_name = '.'.join([dataset_name, 'json'])

    # load json file
    with open(os.path.join(json_path, json_name), 'r') as j:
        j_file = json.loads(j.read())
    wav_train_list, wav_test_list, wav_val_list = [], [], []
    midi_train_list, midi_test_list, midi_val_list = [], [], []
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

    for idx, wav_file in enumerate(wav_list):
        wav_list[idx] = os.path.join(raw_wav_path, wav_file)
    for idx, midi_file in enumerate(midi_list):
        midi_list[idx] = os.path.join(midi_path, midi_file)

    return wav_list


def init_evaluate_list_MAPS(raw_wav_path=None):
    sub_dir = ['/MAPS_ENSTDkAm_2/ENSTDkAm/MUS/', '/MAPS_ENSTDkCl_2/ENSTDkCl/MUS/']
    wav_list = []
    for sub_split in sub_dir:
        parent_path = ''.join([raw_wav_path, sub_split])
        all_files = os.listdir(parent_path)
        for file in all_files:
            if file[-4:] == '.wav':
                wav_list.append(parent_path + file)
    return wav_list


def metric_computation(midi_out, midi_gt, ons_th=0.3, frm_th=0.4, off_th=0.5, top_k=None, for_pedal=False,
                       hpt_heuristic_decoding=False, decoding_with_offsets=True, discard_zero_vel=False,
                       mpe_extract=False):
    # ons_th = 0.3
    # frm_th = 0.4
    # off_th = 0.5

    eval_obj = {'frame': midi_out['frame_pred'], 'onset': midi_out['onset_pred'], 'offset': midi_out['offset_pred'],
                'velocity': midi_out['velocity_pred']}
    eval_ref = {'frame': midi_gt['frame_ref'], 'onset': midi_gt['onset_ref'], 'offset': midi_gt['offset_ref'],
                'velocity': midi_gt['velocity_ref']}

    metrics_tensorboard = metric.evaluate(
        eval_obj, eval_ref, onset_threshold=ons_th, frame_threshold=frm_th, offset_threshold=off_th, top_k=top_k,
        decoding_with_offsets=decoding_with_offsets, hpt_heuristic_decoding=hpt_heuristic_decoding,
        discard_zero_vel=discard_zero_vel, return_with_metric_counts=True, for_pedal=for_pedal, mpe_extract=mpe_extract
    )
    metrics_frame, metrics_onset, metrics_note_w_offset, metrics_note_w_offset_velocity = (
        metric.metric_unboxing(metrics_tensorboard)
    )

    counts_frame, counts_onset, counts_note_w_offset, counts_note_w_offset_velocity = (
        metric.counts_unboxing(metrics_tensorboard))

    m = {
        'm_frm': metrics_frame,
        'm_ons': metrics_onset,
        'm_off': metrics_note_w_offset,
        'm_vel': metrics_note_w_offset_velocity
    }
    c = {
        'c_frm': counts_frame,
        'c_ons': counts_onset,
        'c_off': counts_note_w_offset,
        'c_vel': counts_note_w_offset_velocity
    }
    return m, c


def log_to_local(log_dict, log_name=None, th=(0.3, 0.4, 0.5),
                 log_path="/data/MaestroV3-dataset/pedal_speed_local_range_grid_search/"):
    print()
    if log_name is None:
        log_name = 'DEFAULT_NAME'
    log_name = log_name + '.txt'
    log_file_path = os.path.join(log_path, log_name)

    if os.path.exists(log_file_path):
        with open(log_file_path, 'a') as log_file:
            log_newline = (
                f"{th[0]}\t{th[1]}\t{th[2]}\t"
                f"{log_dict['m_frm']['pre']:.4f}\t{log_dict['m_frm']['rec']:.4f}\t{log_dict['m_frm']['F_m']:.4f}\t"
                f"{log_dict['m_ons']['pre']:.4f}\t{log_dict['m_ons']['rec']:.4f}\t{log_dict['m_ons']['F_m']:.4f}\t"
                f"{log_dict['m_off']['pre']:.4f}\t{log_dict['m_off']['rec']:.4f}\t{log_dict['m_off']['F_m']:.4f}\t"
                f"{log_dict['m_vel']['pre']:.4f}\t{log_dict['m_vel']['rec']:.4f}\t{log_dict['m_vel']['F_m']:.4f}\n")
            log_file.write(log_newline)
    else:
        with open(log_file_path, 'w') as log_file:
            title = f"ons_th\tfrm_th\toff_th\t" \
                    "frm_pre\tfrm_rec\tfrm_fm\t" \
                    "ons_pre\tons_rec\tons_fm\t" \
                    "off_pre\toff_rec\toff_fm\t" \
                    "vel_pre\tvel_rec\tvel_fm\n"
            log_file.write(title)
            log_newline = (
                f"{th[0]}\t{th[1]}\t{th[2]}\t"
                f"{log_dict['m_frm']['pre']:.4f}\t{log_dict['m_frm']['rec']:.4f}\t{log_dict['m_frm']['F_m']:.4f}\t"
                f"{log_dict['m_ons']['pre']:.4f}\t{log_dict['m_ons']['rec']:.4f}\t{log_dict['m_ons']['F_m']:.4f}\t"
                f"{log_dict['m_off']['pre']:.4f}\t{log_dict['m_off']['rec']:.4f}\t{log_dict['m_off']['F_m']:.4f}\t"
                f"{log_dict['m_vel']['pre']:.4f}\t{log_dict['m_vel']['rec']:.4f}\t{log_dict['m_vel']['F_m']:.4f}\n")
            log_file.write(log_newline)


def log_counts_to_local(log_dict, log_name=None, th=(0.3, 0.4, 0.5),
                        log_path="/data/MaestroV3-dataset/pedal_speed_local_range_grid_search/"):
    print()
    if log_name is None:
        log_name = 'DEFAULT_NAME'
    log_name = log_name + '.txt'
    log_file_path = os.path.join(log_path, log_name)

    if os.path.exists(log_file_path):
        with open(log_file_path, 'a') as log_file:
            log_newline = (
                f"{th[0]}\t{th[1]}\t{th[2]}\t"
                f"{log_dict['c_frm']['tp_sum']:d}\t{log_dict['c_frm']['ref_sum']:d}\t{log_dict['c_frm']['est_sum']:d}\t"
                f"{log_dict['c_ons']['tp_sum']:d}\t{log_dict['c_ons']['ref_sum']:d}\t{log_dict['c_ons']['est_sum']:d}\t"
                f"{log_dict['c_off']['tp_sum']:d}\t{log_dict['c_off']['ref_sum']:d}\t{log_dict['c_off']['est_sum']:d}\t"
                f"{log_dict['c_vel']['tp_sum']:d}\t{log_dict['c_vel']['ref_sum']:d}\t{log_dict['c_vel']['est_sum']:d}\n"
            )
            log_file.write(log_newline)
    else:
        with open(log_file_path, 'w') as log_file:
            title = f"ons_th\tfrm_th\toff_th\t" \
                    "frm_tp\tfrm_ref\tfrm_est\t" \
                    "ons_tp\tons_ref\tons_est\t" \
                    "off_tp\toff_ref\toff_est\t" \
                    "vel_tp\tvel_ref\tvel_est\n"
            log_file.write(title)
            log_newline = (
                f"{th[0]}\t{th[1]}\t{th[2]}\t"
                f"{log_dict['c_frm']['tp_sum']:d}\t{log_dict['c_frm']['ref_sum']:d}\t{log_dict['c_frm']['est_sum']:d}\t"
                f"{log_dict['c_ons']['tp_sum']:d}\t{log_dict['c_ons']['ref_sum']:d}\t{log_dict['c_ons']['est_sum']:d}\t"
                f"{log_dict['c_off']['tp_sum']:d}\t{log_dict['c_off']['ref_sum']:d}\t{log_dict['c_off']['est_sum']:d}\t"
                f"{log_dict['c_vel']['tp_sum']:d}\t{log_dict['c_vel']['ref_sum']:d}\t{log_dict['c_vel']['est_sum']:d}\n"
            )
            log_file.write(log_newline)


def show_time():
    nt = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    print(f'[Time] {nt}')


def evaluate_every_song_parallel(load_saved_outputs=EVA_FROM_LOCAL):
    nt = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    log_path = EVA_LOG_PATH
    dataset_name = DATASET_NAME
    raw_wav_path = RAW_WAV_PATH
    schedule_idx = SCHEDULE_IDX
    evaluate_term = EVALUATE_TERM
    log_name = 'MT-FiLM'
    for_pedal = False
    if schedule_idx[1] != 'x' and evaluate_term != 'note':
        for_pedal = True

    if not load_saved_outputs:
        # model, model_full_name, device, epoch, train_time = init_model(mt_film.MT_FiLMSustainPedal(), CKP_PATH, CHECKPOINT)
        model, model_full_name, device, epoch, train_time = init_model(mt_film.MT_FiLM(), CKP_PATH, CHECKPOINT)
        log_path = log_path + log_name
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_path = log_path + '/'
        show_time()
        print('Model Initialized.\n')
        
        if dataset_name == 'maps':
            wav_list = init_evaluate_list_MAPS(raw_wav_path)
        elif dataset_name in ['maestro-v1.0.0', 'maestro-v2.0.0', 'maestro-v3.0.0']:
            wav_list = init_evaluate_list(dataset_name, raw_wav_path)
           
        show_time()
        print('Audio List Initialized.\n')
        show_time()
        print('Evaluating...')
        print(f'Total Audio Num. [{len(wav_list):4d}]')
        
        ss = joblib.Parallel(n_jobs=-1, verbose=10)(
            joblib.delayed(convert_to_spectrum)(wav_path, 'cqt')
            for wav_path in wav_list)

        midi_gt_all = []
        midi_out_all = []
        show_time()
        for idx, s in enumerate(ss):
            print(f'\rEvaluating on: [{idx + 1:4d}/{len(wav_list):4d}]', end='')
            spectrum, target_prs = s[0], s[1]
            spectrum = spectrum['cqt']
            midi_gt = func.construct_midi_gt(target_prs, schedule_idx=schedule_idx, evaluate_term=evaluate_term)
            midi_out = piano_transcription(model, spectrum, device, block_stride=0.5,
                                           schedule_idx=schedule_idx, evaluate_term=evaluate_term)
            midi_gt_all.append(midi_gt)
            midi_out_all.append(midi_out)
        print()

        torch.save(midi_gt_all, log_path + "midi_gt_all.pt")
        torch.save(midi_out_all, log_path + "midi_out_all.pt")
    else:
        midi_gt_all = torch.load(log_path + log_name + "/midi_gt_all.pt")
        midi_out_all = torch.load(log_path + log_name + "/midi_out_all.pt")

    show_time()
    log_name = nt + '_' + log_name
    # adjust all the thresholds for onset and frame note decoding.
    # the offset thresholds are invalid for default.
    for ons_th in EVA_ONS_TH:
        for frm_th in EVA_FRM_TH:
            for off_th in EVA_OFF_TH:
                log_dict = {
                    'm_frm': {'pre': [], 'rec': [], 'F_m': [], 'loss': []},
                    'm_ons': {'pre': [], 'rec': [], 'F_m': [], 'loss': []},
                    'm_off': {'pre': [], 'rec': [], 'F_m': [], 'loss': []},
                    'm_vel': {'pre': [], 'rec': [], 'F_m': [], 'loss': []}
                }
                counts_dict = {
                    'c_frm': {'tp_sum': [], 'ref_sum': [], 'est_sum': []},
                    'c_ons': {'tp_sum': [], 'ref_sum': [], 'est_sum': []},
                    'c_off': {'tp_sum': [], 'ref_sum': [], 'est_sum': []},
                    'c_vel': {'tp_sum': [], 'ref_sum': [], 'est_sum': []}
                }
                print(f'onset threshold: {ons_th}; frame threshold: {frm_th}; offset threshold: {off_th}')
                mss = joblib.Parallel(n_jobs=-1, verbose=10)(
                    joblib.delayed(metric_computation)(midi_out, midi_gt, ons_th, frm_th, off_th,
                                                       hpt_heuristic_decoding=True, decoding_with_offsets=False,
                                                       for_pedal=for_pedal)
                    for midi_out, midi_gt in zip(midi_out_all, midi_gt_all))
                for ms in mss:
                    for m_key, m_prf in ms[0].items():
                        for prf_key, m_value in m_prf.items():
                            log_dict[m_key][prf_key].append(m_value)
                    for c_key, c_trp in ms[1].items():
                        for trp_key, c_value in c_trp.items():
                            counts_dict[c_key][trp_key].append(c_value)
                for m_key, m_prf in log_dict.items():
                    for prf_key, value in m_prf.items():
                        if prf_key == 'loss':
                            continue
                        log_dict[m_key][prf_key] = np.mean(log_dict[m_key][prf_key])
                for c_key, c_trp in counts_dict.items():
                    for trp_key, c_value in c_trp.items():
                        counts_dict[c_key][trp_key] = np.sum(counts_dict[c_key][trp_key])

                show_time()
                log_to_local(log_dict, log_name, (ons_th, frm_th, off_th), log_path)
                log_counts_to_local(counts_dict, 'Count_' + log_name, (ons_th, frm_th, off_th), log_path)


if __name__ == '__main__':
    evaluate_every_song_parallel()
