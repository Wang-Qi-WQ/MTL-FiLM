"""
model training and music-segment based quick & simple validation
"""
import joblib
import numpy as np
import torch
import torch.nn as nn
import datetime
import os
from torch.utils.tensorboard import SummaryWriter

from constant_macro import *
from utils import func, optim_custom, simplelog, metric
import checkpoint
from data import dataload
from net import mt_film

torch.manual_seed(SAMPLE_SEED)
torch.cuda.manual_seed(SAMPLE_SEED)
torch.cuda.manual_seed_all(SAMPLE_SEED)


def train(train_model, loaded, optimizer, device, cache_manager, lr_scheduler, schedule_idx):
    train_model.train()
    losses_on_batch = []
    for batch_idx, input_dict in enumerate(loaded):
        if schedule_idx[0] != 'x' and input_dict['onset_roll'].max() == 0:
            continue
        losses, _ = func.run_one_batch(run_type='train', model=train_model, schedule_idx=schedule_idx,
                                       input_dict=input_dict, device=device, evaluate_term=None)
        loss = losses['loss']
        losses_on_batch.append(loss.item())
        print(f'\rTrainBatch [{batch_idx + 1:8d}/{len(loaded):8d}]|loss:{np.mean(losses_on_batch):.10f}', end='')

        # backward
        optimizer.zero_grad()
        loss.backward()

        lr_scheduler.step()
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), 10.)
        optimizer.step()

    print()
    cache_manager.batch_reshuffle()

    return np.mean(losses_on_batch)


def validate(epoch, validate_model, loaded, device, schedule_idx):
    validate_model.eval()
    validate_model_out = []

    evaluate_term = EVALUATE_TERM
    for_pedal = False
    if schedule_idx[1] != 'x' and evaluate_term != 'note':
        for_pedal = True

    with torch.no_grad():
        for batch_idx, input_dict in enumerate(loaded):
            if schedule_idx[0] != 'x' and input_dict['onset_roll'].max() == 0:
                continue
            print(f'\rNow transcribing...  Batch[{batch_idx + 1:4d}/{len(loaded):4d}]', end='')
            losses, out_prs = func.run_one_batch(run_type='validate', model=validate_model, schedule_idx=schedule_idx,
                                                 input_dict=input_dict, device=device, evaluate_term=evaluate_term)
            frame_pred, midi_ref = out_prs['frame']
            onset_pred, onset_ref = out_prs['onset']
            offset_pred, offset_ref = out_prs['offset']
            velocity_pred, velocity_ref = out_prs['velocity']

            frame_pred_clip = frame_pred.reshape(-1, frame_pred.shape[-1]).cpu()
            midi_ref_clip = midi_ref.reshape(-1, midi_ref.shape[-1]).cpu()

            onset_ref_clip = onset_ref.reshape(-1, onset_ref.shape[-1]).cpu()
            if onset_pred is None:
                onset_pred_clip = onset_ref_clip
            else:
                onset_pred_clip = onset_pred.reshape(-1, onset_pred.shape[-1]).cpu()

            offset_ref_clip = offset_ref.reshape(-1, offset_ref.shape[-1]).cpu()
            if offset_pred is None:
                offset_pred_clip = offset_ref_clip
            else:
                offset_pred_clip = offset_pred.reshape(-1, offset_pred.shape[-1]).cpu()

            velocity_ref_clip = velocity_ref.reshape(-1, velocity_ref.shape[-1]).cpu()
            if velocity_pred is None:
                velocity_pred_clip = velocity_ref_clip
            else:
                velocity_pred = torch.argmax(velocity_pred, dim=1)
                velocity_pred_clip = velocity_pred.reshape(-1, velocity_pred.shape[-1]).cpu()

            eval_obj = {'frame': frame_pred_clip, 'onset': onset_pred_clip, 'offset': offset_pred_clip,
                        'velocity': velocity_pred_clip}
            eval_ref = {'frame': midi_ref_clip, 'onset': onset_ref_clip, 'offset': offset_ref_clip,
                        'velocity': velocity_ref_clip}

            losses = {k_: v_.cpu() for k_, v_ in losses.items()}
            validate_model_out.append(func.TestModelOutDatapack(
                b_idx=batch_idx, eval_obj=eval_obj, eval_ref=eval_ref, losses=losses))

    print('\nNow evaluating...')
    metric_total = joblib.Parallel(n_jobs=N_JOBS, verbose=0)(
        joblib.delayed(func.validate_parallel_metric_compute)(
            datapack=datapack, ons_th=ONS_TH, frm_th=FRM_TH, off_th=OFF_TH, for_pedal=for_pedal)
        for datapack in validate_model_out)
    metrics_frame, metrics_onset, metrics_note_w_offset, metrics_note_w_offset_velocity = (
        func.validate_batch_metric_mean(metric_total=metric_total))
    # statement print
    simplelog.show_statu('Test', epoch, EPOCH_RANGE, batch_idx + 1, len(loaded), metrics_frame, metrics_onset,
                         metrics_note_w_offset, metrics_note_w_offset_velocity)

    print()
    epoch_metrics = metric.epoch_metric(metrics_frame, metrics_onset, metrics_note_w_offset,
                                        metrics_note_w_offset_velocity)

    return metrics_frame, metrics_onset, metrics_note_w_offset, metrics_note_w_offset_velocity, epoch_metrics


def main():
    # data load
    train_load, cache_manager, vocab_manager = None, None, None
    if WITH_TRAIN:
        nt = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S').replace(':', '_')
        print(f'[Time] {nt}')
        train_load, cache_manager = (
            dataload.batch_loader_wav_midi_maestro_seq_single_frm(
                raw_wav_path=RAW_WAV_PATH, midi_path=MIDI_PATH,
                json_path=JSON_PATH, cache_path=CACHE_PATH,
                cache_piece=CACHE_PIECE, preprocess=PREPROCESS,
                time_freq_diagram=True,
                pedal_threshold=PEDAL_TH,
                max_pedal_speed=MAX_PEDAL_SPEED, min_pedal_range=MIN_PEDAL_RANGE,
                pedal_speed_calcu_extra_frm_range=PEDAL_SPEED_RANGE,
                maximum_pedal_acc=PEDAL_MAXIMUM_ACCELERATED_SPEED,
                note_minimum_frame=NOTE_MINIMUM_FRAME,
                batch_size=BATCH_SIZE,
                frames_size=TRAIN_FRAMES_SIZE, train_for_decoder=ONLY_BATCH_SHUFFLE,
                resample=RESAMPLE_FREQ_DOTS, frame_length_ms=FRAME_LENGTH_MS,
                hop_length_ratio=HOP_LENGTH_RATIO, n_fft=N_FFT,
                n_mel=N_MEL, top_db=TOP_DB,
                load_type='train', schedule_idx=SCHEDULE_IDX, with_train=WITH_TRAIN,
                load_portion=LOAD_PORTION, load_offset=LOAD_OFFSET,
                seed=SAMPLE_SEED,
                num_workers=N_WORKERS, n_jobs=N_JOBS,
                dataset_name=DATASET_NAME, pin_memory=True))
    validate_load = None
    if WITH_VALIDATE:
        nt = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S').replace(':', '_')
        print(f'[Time] {nt}')
        validate_load, _, = (
            dataload.batch_loader_wav_midi_maestro_seq_single_frm(
                raw_wav_path=RAW_WAV_PATH, midi_path=MIDI_PATH,
                json_path=JSON_PATH, cache_path=VALIDATE_CACHE_PATH,
                cache_piece=CACHE_PIECE, preprocess=PREPROCESS,
                time_freq_diagram=True,
                pedal_threshold=PEDAL_TH,
                max_pedal_speed=MAX_PEDAL_SPEED, min_pedal_range=MIN_PEDAL_RANGE,
                pedal_speed_calcu_extra_frm_range=PEDAL_SPEED_RANGE,
                maximum_pedal_acc=PEDAL_MAXIMUM_ACCELERATED_SPEED,
                note_minimum_frame=NOTE_MINIMUM_FRAME,
                batch_size=VALIDATE_BATCH_SIZE, frames_size=VALIDATE_FRAMES_SIZE,
                resample=RESAMPLE_FREQ_DOTS, frame_length_ms=FRAME_LENGTH_MS,
                hop_length_ratio=HOP_LENGTH_RATIO, n_fft=N_FFT,
                n_mel=N_MEL, top_db=TOP_DB,
                load_type='validation', schedule_idx=SCHEDULE_IDX,
                load_portion=VALIDATE_LOAD_PORTION, load_offset=VALIDATE_LOAD_OFFSET,
                overlap_size=OVERLAP_SIZE,
                num_workers=N_WORKERS, n_jobs=N_JOBS,
                dataset_name=DATASET_NAME))

    
    if  MODEL_NAME == 'MT_FiLM':
        model = mt_film.MT_FiLM()
    elif MODEL_NAME == 'TF_FiLMSustainPedal':
        model = mt_film.MT_FiLMSustainPedal()

    train_time = datetime.datetime.now()
    model_name = type(model).__name__ + MODEL_NAME_APPENDIX
    epoch_start = 1

    # batch metrics buffers
    batch_metrics_info_frm = metric.BatchTrainInfo()
    batch_metrics_info_ons = metric.BatchTrainInfo()
    batch_metrics_info_off = metric.BatchTrainInfo()
    batch_metrics_info_vel = metric.BatchTrainInfo()
    bm_info = {'frm': batch_metrics_info_frm, 'ons': batch_metrics_info_ons,
               'off': batch_metrics_info_off, 'vel': batch_metrics_info_vel}
    
    optimizer = optim_custom.register_adam_optimizer(model=model, lr=LEARNING_RATE, sub_modules=SUB_TRAINED)
    lr_scheduler = func.StairDownLR(optimizer=optimizer, lr=LEARNING_RATE, stair_length=STEP_STAIR_LENGTH)
    schedule_idx = SCHEDULE_IDX

    # GPU available
    device_ids = [i for i in range(0, torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = model.to(device)
    if (device.type != 'cpu') and (torch.cuda.device_count() > 1):
        model = nn.DataParallel(model, device_ids=device_ids)

    # model parameters
    model_para = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model Params: {model_para:.1f}M\n'
          f'Test threshold: onset={ONS_TH}, frame={FRM_TH}, offset={OFF_TH}\n'
          f'LR Scheduler Step: {STEP_STAIR_LENGTH}\n'
          f'Velocity Rescale: {VELOCITY_RESCALE}')

    # checkpoint
    load_len = len(train_load) if train_load is not None else len(validate_load)
    if FINE_TUNING:
        ft_obj = checkpoint.fine_tuning(model=model, optimizer=optimizer, load_len=load_len, bm_info=bm_info,
                                        sub_modules=SUB_TRAINED)
        model = ft_obj['model']
        model_name = ft_obj['model_name']
        lr_scheduler = ft_obj['lr_scheduler']
        epoch_start = ft_obj['epoch_start']
        train_time = ft_obj['train_time'] if ft_obj['train_time'] is not None else train_time
    print(f'\nDataset: {DATASET_NAME}')
    print(f'Model Name: {model_name}\n')

    # log file dir
    if DATASET_NAME == 'maestro-v1.0.0':
        ds_name = 'MV1'
    elif DATASET_NAME == 'maestro-v2.0.0':
        ds_name = 'MV2'
    else:
        ds_name = 'MV3'
    train_time_str = train_time.strftime('%y_%m_%d_%H_%M_%S').replace(':', '_')
    log_path = LOG_PATH + f'{train_time_str}_{ds_name}_{model_name}'
    if FORMAL_TRAINING:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_path = log_path + '/'

    # train and validate
    for epoch in range(epoch_start, EPOCH_RANGE + 1):
        nt = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S').replace(':', '_')
        nt_lr = lr_scheduler.lr()
        print(f'Epoch      [{epoch:8}/{EPOCH_RANGE:8}]|time:{nt}|learning rate:{nt_lr}')
        mean_loss = None
        if WITH_TRAIN:
            torch.cuda.empty_cache()
            assert train_load is not None and cache_manager is not None
            mean_loss = train(train_model=model, loaded=train_load, optimizer=optimizer, device=device,
                              cache_manager=cache_manager, lr_scheduler=lr_scheduler, schedule_idx=schedule_idx)

            
            writer = SummaryWriter(log_path + f'train_tensorboard')
            writer.add_scalar(f'train/loss', mean_loss, global_step=epoch)
            writer.flush()
            writer.close()
            nt = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S').replace(':', '_')
            print(f'[Time] {nt}')

        # save model
        if WITH_TRAIN and ((epoch % EPOCH_TRAIN_SAVE_PERIOD == 0) or epoch >= EPOCH_TRAIN_SAVE_MIN_EPOCH):
            torch.cuda.empty_cache()
            model_info = checkpoint.ModelInfo(model_dict=model.state_dict(), optim_dict=optimizer.state_dict(),
                                              model_name=model_name, dataset_name=DATASET_NAME,
                                              train_time=train_time, train_epoch=epoch,
                                              train_batch_idx=batch_metrics_info_frm.refer_batch_idx,
                                              train_batch_tp_fp_fn={
                                                  'frame': batch_metrics_info_frm.refer_tp_fp_fn_batch,
                                                  'onset': batch_metrics_info_ons.refer_tp_fp_fn_batch,
                                                  'offset': batch_metrics_info_off.refer_tp_fp_fn_batch,
                                                  'velocity': batch_metrics_info_vel.refer_tp_fp_fn_batch},
                                              train_batch_metrics={
                                                  'frame': batch_metrics_info_frm.refer_metrics_batch,
                                                  'onset': batch_metrics_info_ons.refer_metrics_batch,
                                                  'offset': batch_metrics_info_off.refer_metrics_batch,
                                                  'velocity': batch_metrics_info_vel.refer_tp_fp_fn_batch})
            checkpoint.save(CKP_PATH, model_info)

            # only save the complete model
            # model_save_to_local = copy.deepcopy(model).to('cpu').module
            # with open(log_path+'model_'+ MODEL_NAME + '_ep' + str(epoch).zfill(3) + '.pkl', 'wb') as f:
            #     pickle.dump(model_save_to_local, f, protocol=4)

        if WITH_VALIDATE and ((epoch % EPOCH_VALIDATE_PERIOD == 0) or epoch >= EPOCH_VALIDATE_DENSE_MIN_EPOCH):
            torch.cuda.empty_cache()
            assert validate_load is not None
            (metrics_frame, metrics_onset, metrics_note_w_offset,
             metrics_note_w_offset_velocity, validate_epoch_metrics) = validate(epoch=epoch, validate_model=model,
                                                                            loaded=validate_load, device=device,
                                                                            schedule_idx=schedule_idx)

            writer = SummaryWriter(log_path + f'validate_tensorboard')
            for key, value in validate_epoch_metrics.items():
                writer.add_scalar(f'validate/' + key.replace(' ', '_'), np.mean(value), global_step=epoch)
            writer.flush()
            writer.close()
            simplelog.log(log_path=log_path, epoch=epoch, F_m=metrics_frame['F_m'], rec=metrics_frame['rec'],
                          pre=metrics_frame['pre'], loss=metrics_frame['loss'],
                          model_name=model_name, train_time=train_time, appendix_str='validate_epoch_frame')
            simplelog.log(log_path=log_path, epoch=epoch, F_m=metrics_onset['F_m'], rec=metrics_onset['rec'],
                          pre=metrics_onset['pre'], loss=metrics_onset['loss'], train_loss=mean_loss,
                          model_name=model_name, train_time=train_time, appendix_str='validate_epoch_note')
            simplelog.log(log_path=log_path, epoch=epoch, F_m=metrics_note_w_offset['F_m'],
                          rec=metrics_note_w_offset['rec'], pre=metrics_note_w_offset['pre'],
                          loss=metrics_note_w_offset['loss'], model_name=model_name, train_time=train_time,
                          appendix_str='validate_epoch_note-w-offset')
            simplelog.log(log_path=log_path, epoch=epoch, F_m=metrics_note_w_offset_velocity['F_m'],
                          rec=metrics_note_w_offset_velocity['rec'], pre=metrics_note_w_offset_velocity['pre'],
                          loss=metrics_note_w_offset_velocity['loss'], model_name=model_name, train_time=train_time,
                          appendix_str='validate_epoch_note-w-offset-velocity')
        print()

        # reset batch metrics buffers
        batch_metrics_info_frm.reset()
        batch_metrics_info_ons.reset()
        batch_metrics_info_off.reset()
        batch_metrics_info_vel.reset()


if __name__ == '__main__':

    main()
