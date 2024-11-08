import datetime as dt
import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def log(log_path, epoch, F_m, rec, pre, loss, model_name, train_time, appendix_str: str = None, train_loss=None):
    train_time = train_time.strftime('%y-%m-%d|%H:%M:%S')
    now_time = dt.datetime.now().strftime('%F|%T')
    log_file_path = f'{log_path}{appendix_str}.txt'
    if os.path.exists(log_file_path):
        with open(log_file_path, 'a') as log_file:
            if train_loss is None:
                log_newline = f"{now_time}\t{epoch:d}\t{pre:.4f}\t{rec:.4f}\t{F_m:.4f}\t{loss:.6f}\n"
            else:
                log_newline = f"{now_time}\t{epoch:d}\t{pre:.4f}\t{rec:.4f}\t{F_m:.4f}\t{loss:.6f}\t{train_loss:.6f}\n"
            log_file.write(log_newline)
    else:
        with open(log_file_path, 'w') as log_file:
            first_line = train_time + '\t' + model_name + '\n'
            if train_loss is None:
                title = "Time\tEpoch\tPre\tRec\tF_m\tLoss\n"
                log_newline = f"{now_time}\t{epoch:d}\t{pre:.4f}\t{rec:.4f}\t{F_m:.4f}\t{loss:.6f}\n"
            else:
                title = "Time\tEpoch\tPre\tRec\tF_m\tLoss\tTrainLoss\n"
                log_newline = f"{now_time}\t{epoch:d}\t{pre:.4f}\t{rec:.4f}\t{F_m:.4f}\t{loss:.6f}\t{train_loss:.6f}\n"
            log_file.write(first_line)
            log_file.write(title)
            log_file.write(log_newline)


def show_statu(des, epoch, epoch_total, batch_index, batch_total, metrics, metrics_onset: dict = None,
               metrics_note_w_offset: dict = None, metrics_note_w_offset_vel: dict = None):
    if metrics_onset is None and metrics_note_w_offset is None:
        print(end='\r')
        print(f'{des:5} Epoch [{epoch:4}/{epoch_total:4}]', end='\0')
        print(f'| Batch [{batch_index:8}/{batch_total:8}]\0|', end='\0\t')
        for item, value in metrics.items():
            print(f'{item}:{metrics[item]:.4f}', end='\t')
    elif metrics_note_w_offset is None:
        print(end='\r')
        print(f'{des:5}', end='')
        print(f'Batch [{batch_index:8}/{batch_total:8}]\0|', end='\0')
        for item, value in metrics.items():
            if item in metrics_onset.keys():
                print(f'{item}:{metrics[item]:.4f}/{metrics_onset[item]:.4f}', end='\0\0')
            else:
                print(f'{item}:{metrics[item]:.4f}', end='\0\0')
    else:
        print(end='\r')
        print(f'{des:5}', end='')
        print(f'Batch [{batch_index:8}/{batch_total:8}]\0|', end='\0')
        for item, value in metrics.items():
            if value is not None:
                if (item in metrics_onset.keys()
                        and item in metrics_note_w_offset.keys()
                        and item in metrics_note_w_offset_vel.keys()):
                    print(f'{item}:{metrics[item]:.4f}/{metrics_onset[item]:.4f}/'
                          f'{metrics_note_w_offset[item]:.4f}/{metrics_note_w_offset_vel[item]:.4f}', end='\0\0')
                else:
                    print(f'{item}:{metrics[item]:.4f}', end='\0\0')
