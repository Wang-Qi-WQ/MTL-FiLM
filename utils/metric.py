import sys
from collections import defaultdict

import numpy as np
import torch
import mir_eval
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from scipy.stats import hmean

from constant_macro import *
from utils.mir_eval_modified import *

eps = sys.float_info.epsilon


class BatchTrainInfo(object):
    def __init__(self):
        self._batch_idx = -1
        self._tp_fp_fn_batch = {'tp': [], 'fp': [], 'fn': []}
        self._metrics_batch = {'pre': [], 'rec': [], 'F_m': [], 'loss': []}

    @property
    def refer_batch_idx(self):
        if self._batch_idx != -1:
            return self._batch_idx
        else:
            return 0

    @property
    def refer_metrics_batch(self):
        return self._metrics_batch

    @property
    def refer_tp_fp_fn_batch(self):
        return self._tp_fp_fn_batch

    def log_batch(self, batch_idx: int):
        self._batch_idx = batch_idx

    def log_metrics(self, true_pos: float, false_pos: float, false_neg: float, loss: torch.Tensor):
        tp_fp_fn_count = {'tp': true_pos, 'fp': false_pos, 'fn': false_neg}
        for key in self._tp_fp_fn_batch.keys():
            self._tp_fp_fn_batch[key].append(tp_fp_fn_count[key])

        epsilon = 1e-20  # avoid dividing by zero
        pre = true_pos / (true_pos + false_pos + epsilon)
        rec = true_pos / (true_pos + false_neg + epsilon)
        fm = (2 * rec * pre) / (rec + pre + epsilon)
        loss_np = loss.data.cpu().numpy()

        metrics = {'pre': pre, 'rec': rec, 'F_m': fm, 'loss': loss_np}
        for key in self._metrics_batch.keys():
            self._metrics_batch[key].append(metrics[key])

    def log_info(self, batch_idx, metrics_task):
        self._batch_idx = batch_idx
        mm = {}
        for item, value in metrics_task.items():
            self._metrics_batch[item].append(value)
            mm[item] = (sum(self._metrics_batch[item]) + eps) / (len(self._metrics_batch[item]) + eps)
        return mm

    def compute_metric(self, acc_step: int = 1):
        assert (self._batch_idx + 1) / acc_step == len(self._metrics_batch['loss'])

        true_pos = sum(self._tp_fp_fn_batch['tp'])
        false_pos = sum(self._tp_fp_fn_batch['fp'])
        false_neg = sum(self._tp_fp_fn_batch['fn'])

        epsilon = 1e-20  # avoid dividing by zero
        pre = true_pos / (true_pos + false_pos + epsilon)
        rec = true_pos / (true_pos + false_neg + epsilon)
        fm = (2 * rec * pre) / (rec + pre + epsilon)
        loss = sum(self._metrics_batch['loss']) / ((self._batch_idx + 1) / acc_step)

        metrics_total = {'pre': pre, 'rec': rec, 'F_m': fm, 'loss': loss}
        return metrics_total

    def reload_batch_train_info(self, reload_batch_idx: int, reload_tp_fp_fn_batch: dict, reload_metrics_batch: dict):
        self._batch_idx = reload_batch_idx
        self._tp_fp_fn_batch = reload_tp_fp_fn_batch
        self._metrics_batch = reload_metrics_batch

    def reset(self):
        self.__init__()


def binary_tp_fp_fn_frm(predict, target, is_tset: bool = False, threshold: float = 0.5):
    """
    calculate TP,FP,FN
    """
    if is_tset:
        # tolerance expand (+-50ms, hop:10ms/frm)
        tolerance_frm = 5
        tolerance_target = torch.zeros(tolerance_frm, target.shape[1]).to(target.device)
        tolerance_target = torch.cat((tolerance_target, target, tolerance_target), dim=0)
        for shift in range(tolerance_frm * 2 + 1):
            tolerance_target[shift:shift + target.shape[0], :] += target
        tolerance_target = torch.where(tolerance_target > 0, 1, 0)
        tolerance_target = tolerance_target[tolerance_frm:tolerance_target.shape[0] - tolerance_frm, :]

        # binary convert
        predict_binary = torch.where(predict > threshold, 1, -1)
        predict_alter = torch.where(predict > threshold, -1, 1)
        target_alter = torch.where(tolerance_target > 0, 0, 1)
        true_pos = torch.where(predict_binary == tolerance_target)[0].numel()
        false_pos = torch.where(predict_binary == target_alter)[0].numel()
        false_neg = torch.where(predict_alter == tolerance_target)[0].numel()

    else:
        # binary convert
        predict_binary = torch.where(predict > threshold, 1, -1)
        predict_alter = torch.where(predict > threshold, -1, 1)
        target_alter = torch.where(target > 0, 0, 1)
        true_pos = torch.where(predict_binary == target)[0].numel()
        false_pos = torch.where(predict_binary == target_alter)[0].numel()
        false_neg = torch.where(predict_alter == target)[0].numel()

    return float(true_pos), float(false_pos), float(false_neg)


def metrics_mean(metrics_task, metrics_task_details):
    mm = {}
    for item, value in metrics_task.items():
        if value is not None:
            metrics_task_details[item].append(value)
            mm[item] = (sum(metrics_task_details[item]) + eps) / (len(metrics_task_details[item]) + eps)
    return mm


def choose_velocity(velocity_region, top_k=None, vel_rescale=False):
    velocity_region = velocity_region.numpy()
    if len(velocity_region.shape) != 2:
        if np.count_nonzero(velocity_region) == 0:
            return velocity_region.max()
        if vel_rescale:
            return velocity_region.sum() / np.count_nonzero(velocity_region) * 128.
        return velocity_region.sum() / np.count_nonzero(velocity_region)
    else:
        top_k_classes = velocity_region.shape[-1]
        if top_k is None:
            top_k = top_k_classes
        for idx in range(top_k):
            if np.count_nonzero(velocity_region[:, idx]) != 0:
                if vel_rescale:
                    return velocity_region[:, idx].sum() / np.count_nonzero(velocity_region[:, idx]) * 128.
                return velocity_region[:, idx].sum() / np.count_nonzero(velocity_region[:, idx])
        return velocity_region[:, 0].max()


def hpt_local_maximum_algorithm(onsets, frames, offsets, velocities,
                                onset_threshold=0.5, frame_threshold=0.5, offset_threshold=0.5, top_k=None,
                                decoding_with_offsets=False, vel_rescale=False, scaling=None, discard_zero_vel=False):
    if scaling is None:
        scaling = FRAME_LENGTH_MS / 1000 * HOP_LENGTH_RATIO

    left = onsets[:1, :] >= onsets[1:2, :]
    right = onsets[-1:, :] >= onsets[-2:-1, :]
    mid = (onsets[1:-1] >= onsets[2:]).float() * (onsets[1:-1] >= onsets[:-2]).float()
    onsets_peak = torch.cat([left, mid, right], dim=0).float() * onsets

    left = offsets[:1, :] >= offsets[1:2, :]
    right = offsets[-1:, :] >= offsets[-2:-1, :]
    mid = (offsets[1:-1] >= offsets[2:]).float() * (offsets[1:-1] >= offsets[:-2]).float()
    offsets_peak = torch.cat([left, mid, right], dim=0).float() * offsets

    onsets_peak = (onsets_peak > onset_threshold).cpu().to(torch.uint8)
    offsets_peak = (offsets_peak > offset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    frames = np.logical_or(frames, onsets_peak)
    onsets_diff = torch.cat([onsets_peak[:1, :], onsets_peak[1:, :] - onsets_peak[:-1, :]], dim=0) == 1
    offsets_diff = torch.cat([offsets_peak[:1, :], offsets_peak[1:, :] - offsets_peak[:-1, :]], dim=0) == 1

    pitches = []
    intervals = []
    note_vels = []

    for nonzero in onsets_diff.nonzero():
        onset = nonzero[0].item()
        pitch = nonzero[1].item()
        velocity = choose_velocity(velocities[max(0, onset - 1):min(onset + 2, onsets.shape[0]), pitch, ...],
                                   top_k=top_k, vel_rescale=vel_rescale)
        if velocity == 0 and discard_zero_vel:
            continue

        if (onset == 0) or (onset == onsets.shape[0] - 1):
            onset_time = onset * scaling
        else:
            if onsets[onset - 1][pitch] == onsets[onset + 1][pitch]:
                onset_time = onset * scaling
            elif onsets[onset - 1][pitch] > onsets[onset + 1][pitch]:
                onset_time = (onset * scaling - (
                        scaling * 0.5 * (onsets[onset - 1][pitch] - onsets[onset + 1][pitch]) /
                        (onsets[onset][pitch] - onsets[onset + 1][pitch])
                ))
            else:
                onset_time = (onset * scaling - (
                        scaling * 0.5 * (onsets[onset + 1][pitch] - onsets[onset - 1][pitch]) /
                        (onsets[onset][pitch] - onsets[onset - 1][pitch])
                ))

        offset = onset
        offset_time = offset * scaling
        while frames[offset, pitch].item():
            offset += 1
            offset_time = offset * scaling
            if (offset == onsets.shape[0]) or (offset == onsets.shape[0] - 1):
                break
            if onsets_diff[offset, pitch].item():
                break
            if decoding_with_offsets and offsets_diff[offset, pitch]:
                if offsets[offset - 1][pitch] == offsets[offset + 1][pitch]:
                    offset_time = offset * scaling
                elif offsets[offset - 1][pitch] > offsets[offset + 1][pitch]:
                    offset_time = (offset * scaling - (
                            scaling * 0.5 * (offsets[offset - 1][pitch] - offsets[offset + 1][pitch]) /
                            (offsets[offset][pitch] - offsets[offset + 1][pitch])
                    ))
                else:
                    offset_time = (offset * scaling - (
                            scaling * 0.5 * (offsets[offset + 1][pitch] - offsets[offset - 1][pitch]) /
                            (offsets[offset][pitch] - offsets[offset - 1][pitch])
                    ))

        if offset_time > onset_time:
            pitches.append(pitch)
            intervals.append([onset_time, offset_time])
            note_vels.append(np.maximum(np.minimum(velocity, 127), 0))

    if pitches == [] and intervals == [] and note_vels == []:
        return np.array([]), np.array([[], []]).T, np.array([])
    return np.array(pitches), np.array(intervals), np.array(note_vels)


# https://github.com/bytedance/piano_transcription/blob/1ade7dcd4348add669a67c6e6282456c8c6633bd/utils/utilities.py#L762
def hpt_output_dict_to_note_pedal_arrays(onsets, frames, offsets, velocities,
                                         onset_threshold=0.5, frame_threshold=0.5, offset_threshold=0.2, top_k=None,
                                         decoding_with_offsets=False, vel_rescale=False, scaling=None,
                                         discard_zero_vel=False):
    if scaling is None:
        scaling = FRAME_LENGTH_MS / 1000 * HOP_LENGTH_RATIO
    pedal_offset_output, pedal_offset_shift_output = get_binarized_output_from_regression(
        reg_output=offsets, threshold=offset_threshold, neighbour=4)
    est_pedal_on_offs = output_dict_to_detected_pedals(
        frames, pedal_offset_output, pedal_offset_shift_output, frame_threshold, scaling)
    pitches = np.zeros(est_pedal_on_offs.shape[0])
    vels = np.ones(est_pedal_on_offs.shape[0]) * 64
    if len(est_pedal_on_offs) == 0:
        return np.array([]), np.array([[], []]).T, np.array([])
    return pitches.astype(np.int64), est_pedal_on_offs, vels.astype(np.int64)


# https://github.com/bytedance/piano_transcription/blob/1ade7dcd4348add669a67c6e6282456c8c6633bd/utils/utilities.py#L839C11-L839C11
def get_binarized_output_from_regression(reg_output, threshold, neighbour):
    """Calculate binarized output and shifts of onsets or offsets from the
    regression results.

    Args:
      reg_output: (frames_num, classes_num)
      threshold: float
      neighbour: int

    Returns:
      binary_output: (frames_num, classes_num)
      shift_output: (frames_num, classes_num)
    """
    binary_output = np.zeros_like(reg_output)
    shift_output = np.zeros_like(reg_output)
    (frames_num, classes_num) = reg_output.shape

    for k in range(classes_num):
        x = reg_output[:, k]
        for n in range(neighbour, frames_num - neighbour):
            if x[n] > threshold and is_monotonic_neighbour(x, n, neighbour):
                binary_output[n, k] = 1

                """See Section III-D in [1] for deduction.
                [1] Q. Kong, et al., High-resolution Piano Transcription 
                with Pedals by Regressing Onsets and Offsets Times, 2020."""
                if x[n - 1] > x[n + 1]:
                    shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                else:
                    shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                shift_output[n, k] = shift

    return binary_output, shift_output


# https://github.com/bytedance/piano_transcription/blob/1ade7dcd4348add669a67c6e6282456c8c6633bd/utils/utilities.py#L873
def is_monotonic_neighbour(x, n, neighbour):
    """Detect if values are monotonic in both side of x[n].

    Args:
      x: (frames_num,)
      n: int
      neighbour: int

    Returns:
      monotonic: bool
    """
    monotonic = True
    for i in range(neighbour):
        if x[n - i] < x[n - i - 1]:
            monotonic = False
        if x[n + i] < x[n + i + 1]:
            monotonic = False

    return monotonic


# https://github.com/bytedance/piano_transcription/blob/1ade7dcd4348add669a67c6e6282456c8c6633bd/utils/utilities.py#L948
def output_dict_to_detected_pedals(frames, pedal_offset_output, pedal_offset_shift_output, frame_th, scaling):
    """Postprocess output_dict to piano pedals.

    Args:
        frames: (frames_num,),
        pedal_offset_output: (frames_num,),
        pedal_offset_shift_output: (frames_num,),
        frame_th: frame threshold,
        scaling: frame hop size scaling factor of absolute time.


    Returns:
      est_on_off: (notes, 2), the two columns are pedal onsets and pedal
        offsets. E.g.,
          [[0.1800, 0.9669],
           [1.1400, 2.6458],
           ...]
    """
    est_tuples = pedal_detection_with_onset_offset_regress(
        frame_output=frames,
        offset_output=pedal_offset_output,
        offset_shift_output=pedal_offset_shift_output,
        frame_threshold=frame_th)

    est_tuples = np.array(est_tuples)
    """(notes, 2), the two columns are pedal onsets and pedal offsets"""

    if len(est_tuples) == 0:
        return np.array([])

    else:
        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) * scaling
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) * scaling
        est_on_off = np.stack((onset_times, offset_times), axis=-1)
        est_on_off = est_on_off.astype(np.float32)
        return est_on_off


# https://github.com/bytedance/piano_transcription/blob/1ade7dcd4348add669a67c6e6282456c8c6633bd/utils/piano_vad.py#L78
def pedal_detection_with_onset_offset_regress(frame_output, offset_output,
                                              offset_shift_output, frame_threshold):
    """Process prediction array to pedal events' information.

    Args:
      frame_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      frame_threshold: float

    Returns:
      output_tuples: list of [bgn, fin, onset_shift, offset_shift],
      e.g., [
        [1821, 1909, 0.4749851, 0.3048533],
        [1909, 1947, 0.30730522, -0.45764327],
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(1, frame_output.shape[0]):
        if frame_output[i] >= frame_threshold and frame_output[i] > frame_output[i - 1]:
            """Pedal onset detected"""
            if bgn:
                pass
            else:
                bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                """Offset detected"""
                offset_occur = i

            if offset_occur:
                fin = offset_occur
                output_tuples.append([bgn, fin, 0., offset_shift_output[fin][0]])
                bgn, frame_disappear, offset_occur = None, None, None

            if frame_disappear and i - frame_disappear >= 5:
                """offset not detected but frame disappear"""
                fin = frame_disappear
                output_tuples.append([bgn, fin, 0., offset_shift_output[fin][0]])
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples


def extract_notes(onsets, frames, offsets, velocity, onset_threshold=0.5, frame_threshold=0.5, offset_threshold=0.5,
                  decoding_with_offsets=False, vel_rescale=False):
    """
    # code from https://github.com/jongwook/onsets-and-frames/blob/master/onsets_and_frames/decoding.py

    Finds the note timings based on the onsets and frames information
    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    offsets: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    offset_threshold: float
    decoding_with_offsets: bool
    vel_rescale: rescale velocity from [0,1] to [0,127]
    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    """
    # https://github.com/WX-Wei/HPPNet/blob/85cfe0955586a7a36002f02e0b7b8cd78fdf45fc/hppnet/decoding.py#LL50C1-L54C67
    # only peaks are consider as onsets
    left = onsets[:1, :] >= onsets[1:2, :]
    right = onsets[-1:, :] >= onsets[-2:-1, :]
    mid = (onsets[1:-1] >= onsets[2:]).float() * (onsets[1:-1] >= onsets[:-2]).float()
    onsets = torch.cat([left, mid, right], dim=0).float() * onsets

    onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    frames = np.logical_or(frames, onsets)
    if decoding_with_offsets:
        offsets = (offsets > offset_threshold).cpu().to(torch.uint8)
        frames[np.where(np.logical_and(frames > 0, offsets > 0))] = 0
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    pitches = []
    intervals = []
    velocities = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if onsets[offset, pitch].item():
                # velocity rescale
                if vel_rescale:
                    velocity_samples.append(velocity[offset, pitch].item() * 128.0)
                else:
                    velocity_samples.append(velocity[offset, pitch].item())
                # velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]:
                break
            if onset_diff[offset, pitch].item():
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(np.maximum(np.max(velocity_samples), 0) if len(velocity_samples) > 0 else 0)

    if pitches == [] and intervals == [] and velocities == []:
        return np.array([]), np.array([[], []]).T, np.array([])
    return np.array(pitches), np.array(intervals), np.array(velocities)


def extract_notes_for_predicted(onsets, frames, offsets, velocity,
                                onset_threshold=0.5, frame_threshold=0.5, offset_threshold=0.5,
                                decoding_with_offsets=False, vel_rescale=False):
    """
    (basically same as func:extract_notes())
    """
    # https://github.com/WX-Wei/HPPNet/blob/85cfe0955586a7a36002f02e0b7b8cd78fdf45fc/hppnet/decoding.py#LL50C1-L54C67
    # only peaks are consider as onsets
    left = onsets[:1, :] >= onsets[1:2, :]
    right = onsets[-1:, :] >= onsets[-2:-1, :]
    mid = (onsets[1:-1] >= onsets[2:]).float() * (onsets[1:-1] >= onsets[:-2]).float()
    onsets = torch.cat([left, mid, right], dim=0).float() * onsets

    onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    frames = np.logical_or(frames, onsets)
    if decoding_with_offsets:
        offsets = (offsets > offset_threshold).cpu().to(torch.uint8)
        frames[np.where(np.logical_and(frames > 0, offsets > 0))] = 0
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    pitches = []
    intervals = []
    velocities = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if onsets[offset, pitch].item():
                # velocity rescale
                if vel_rescale:
                    velocity_samples.append(velocity[offset, pitch].item() * 128.0)
                else:
                    velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]:
                break
            if onset_diff[offset, pitch].item():
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(
                np.minimum(np.maximum(np.max(velocity_samples), 0), 127) if len(velocity_samples) > 0 else 0)

    if pitches == [] and intervals == [] and velocities == []:
        return np.array([]), np.array([[], []]).T, np.array([])
    return np.array(pitches), np.array(intervals), np.array(velocities)


def notes_to_frames(pitches, intervals, shape):
    """
    # code from https://github.com/jongwook/onsets-and-frames/blob/master/onsets_and_frames/decoding.py

    Take lists specifying notes sequences and return
    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]
    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs


def mpe_direct_from_frame_output(frames, scaling, frame_threshold=None):
    if frame_threshold is not None:
        frames = (frames > frame_threshold).float().cpu().to(torch.uint8)
    frames = np.array(frames)
    time = np.arange(frames.shape[0])
    f_ = [frames[t, :].nonzero()[0] for t in time]
    t_ = time.astype(np.float64) * scaling
    f_ = [np.array([mir_eval.util.midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_]
    return t_, f_


def evaluate(eval_obj, eval_ref, losses=None, onset_threshold=0.5, frame_threshold=0.5, offset_threshold=0.5,
             top_k=None, discard_zero_vel=False,
             return_with_metric_counts=False, decoding_with_offsets=False, hpt_heuristic_decoding=False,
             vel_rescale=VELOCITY_RESCALE, hop_ms=None, for_pedal=False, mpe_extract=False):
    """
    # code from https://github.com/jongwook/onsets-and-frames/blob/master/evaluate.py
    """
    if hop_ms is None:
        scaling = FRAME_LENGTH_MS / 1000 * HOP_LENGTH_RATIO
    else:
        scaling = hop_ms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = defaultdict(list)

        if losses is not None:
            for key, loss in losses.items():
                if type(loss) == torch.Tensor:
                    metrics[key].append(loss.item())
                else:
                    metrics[key].append(loss)

        if hpt_heuristic_decoding:
            if for_pedal:
                p_ref, i_ref, v_ref = hpt_output_dict_to_note_pedal_arrays(eval_ref['onset'], eval_ref['frame'],
                                                                           eval_ref['offset'], eval_ref['velocity'],
                                                                           vel_rescale=vel_rescale, scaling=scaling)
                p_est, i_est, v_est = hpt_output_dict_to_note_pedal_arrays(eval_obj['onset'], eval_obj['frame'],
                                                                           eval_obj['offset'],
                                                                           eval_obj['velocity'], onset_threshold,
                                                                           frame_threshold,
                                                                           offset_threshold, top_k,
                                                                           decoding_with_offsets=decoding_with_offsets,
                                                                           vel_rescale=vel_rescale, scaling=scaling,
                                                                           discard_zero_vel=discard_zero_vel)
            else:
                p_ref, i_ref, v_ref = hpt_local_maximum_algorithm(eval_ref['onset'], eval_ref['frame'],
                                                                  eval_ref['offset'], eval_ref['velocity'],
                                                                  decoding_with_offsets=decoding_with_offsets,
                                                                  vel_rescale=vel_rescale, scaling=scaling)
                p_est, i_est, v_est = hpt_local_maximum_algorithm(eval_obj['onset'], eval_obj['frame'],
                                                                  eval_obj['offset'],
                                                                  eval_obj['velocity'], onset_threshold,
                                                                  frame_threshold,
                                                                  offset_threshold, top_k,
                                                                  decoding_with_offsets=decoding_with_offsets,
                                                                  vel_rescale=vel_rescale, scaling=scaling,
                                                                  discard_zero_vel=discard_zero_vel)
            i_ref_frame = (i_ref / scaling).astype(np.int64).reshape(-1, 2)
            i_est_frame = (i_est / scaling).astype(np.int64).reshape(-1, 2)
            t_ref, f_ref = notes_to_frames(p_ref, i_ref_frame, eval_ref['frame'].shape)
            t_est, f_est = notes_to_frames(p_est, i_est_frame, eval_obj['frame'].shape)

            t_ref = t_ref.astype(np.float64) * scaling
            f_ref = [np.array([mir_eval.util.midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref]
            t_est = t_est.astype(np.float64) * scaling
            f_est = [np.array([mir_eval.util.midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

            p_ref = np.array([mir_eval.util.midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
            p_est = np.array([mir_eval.util.midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        else:
            p_ref, i_ref, v_ref = extract_notes(eval_ref['onset'], eval_ref['frame'], eval_ref['offset'],
                                                eval_ref['velocity'], decoding_with_offsets=decoding_with_offsets,
                                                vel_rescale=vel_rescale)
            p_est, i_est, v_est = extract_notes_for_predicted(eval_obj['onset'], eval_obj['frame'], eval_obj['offset'],
                                                              eval_obj['velocity'], onset_threshold, frame_threshold,
                                                              offset_threshold,
                                                              decoding_with_offsets=decoding_with_offsets,
                                                              vel_rescale=vel_rescale)

            t_ref, f_ref = notes_to_frames(p_ref, i_ref, eval_ref['frame'].shape)
            t_est, f_est = notes_to_frames(p_est, i_est, eval_obj['frame'].shape)

            i_ref = (i_ref * scaling).reshape(-1, 2)
            p_ref = np.array([mir_eval.util.midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
            i_est = (i_est * scaling).reshape(-1, 2)
            p_est = np.array([mir_eval.util.midi_to_hz(MIN_MIDI + midi) for midi in p_est])

            t_ref = t_ref.astype(np.float64) * scaling
            f_ref = [np.array([mir_eval.util.midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref]
            t_est = t_est.astype(np.float64) * scaling
            f_est = [np.array([mir_eval.util.midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

        if mpe_extract:
            t_ref, f_ref = mpe_direct_from_frame_output(eval_ref['frame'], scaling)
            t_est, f_est = mpe_direct_from_frame_output(eval_obj['frame'], scaling, frame_threshold)

        onset_tolerance = 0.05
        if for_pedal:
            # https://github.com/bytedance/piano_transcription/blob/1ade7dcd4348add669a67c6e6282456c8c6633bd/pytorch/calculate_score_for_paper.py#L282
            onset_tolerance = 0.2

        if return_with_metric_counts:
            p, r, f, tp, est_sum, ref_sum = note_precision_recall_f1_overlap(i_ref, p_ref, i_est, p_est,
                                                                             onset_tolerance=onset_tolerance,
                                                                             offset_ratio=None)
            metrics['metric/note/precision'].append(p)
            metrics['metric/note/recall'].append(r)
            metrics['metric/note/f1'].append(f)
            metrics['metric/note/tp_sum'].append(tp)
            metrics['metric/note/est_sum'].append(est_sum)
            metrics['metric/note/ref_sum'].append(ref_sum)

            p, r, f, tp, est_sum, ref_sum = note_precision_recall_f1_overlap(i_ref, p_ref, i_est, p_est,
                                                                             onset_tolerance=onset_tolerance)
            metrics['metric/note-with-offsets/precision'].append(p)
            metrics['metric/note-with-offsets/recall'].append(r)
            metrics['metric/note-with-offsets/f1'].append(f)
            metrics['metric/note-with-offsets/tp_sum'].append(tp)
            metrics['metric/note-with-offsets/est_sum'].append(est_sum)
            metrics['metric/note-with-offsets/ref_sum'].append(ref_sum)

            p, r, f, tp, est_sum, ref_sum = vel_precision_recall_f1_overlap(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                                            onset_tolerance=onset_tolerance,
                                                                            offset_ratio=None, velocity_tolerance=0.1)
            metrics['metric/note-with-velocity/precision'].append(p)
            metrics['metric/note-with-velocity/recall'].append(r)
            metrics['metric/note-with-velocity/f1'].append(f)
            metrics['metric/note-with-velocity/tp_sum'].append(tp)
            metrics['metric/note-with-velocity/est_sum'].append(est_sum)
            metrics['metric/note-with-velocity/ref_sum'].append(ref_sum)

            p, r, f, tp, est_sum, ref_sum = vel_precision_recall_f1_overlap(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                                            onset_tolerance=onset_tolerance,
                                                                            velocity_tolerance=0.1)
            metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
            metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
            metrics['metric/note-with-offsets-and-velocity/f1'].append(f)
            metrics['metric/note-with-offsets-and-velocity/tp_sum'].append(tp)
            metrics['metric/note-with-offsets-and-velocity/est_sum'].append(est_sum)
            metrics['metric/note-with-offsets-and-velocity/ref_sum'].append(ref_sum)

            frame_metrics = multipitch_metrics(t_ref, f_ref, t_est, f_est)
            metrics['metric/frame/precision'].append(frame_metrics['Precision'])
            metrics['metric/frame/recall'].append(frame_metrics['Recall'])
            metrics['metric/frame/f1'].append(
                hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)
            metrics['metric/frame/tp_sum'].append(frame_metrics['matched'])
            metrics['metric/frame/est_sum'].append(frame_metrics['n_est'])
            metrics['metric/frame/ref_sum'].append(frame_metrics['n_ref'])

        else:
            p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, onset_tolerance=onset_tolerance, offset_ratio=None)
            metrics['metric/note/precision'].append(p)
            metrics['metric/note/recall'].append(r)
            metrics['metric/note/f1'].append(f)

            p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
            metrics['metric/note-with-offsets/precision'].append(p)
            metrics['metric/note-with-offsets/recall'].append(r)
            metrics['metric/note-with-offsets/f1'].append(f)

            p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                      onset_tolerance=onset_tolerance,
                                                      offset_ratio=None, velocity_tolerance=0.1)
            metrics['metric/note-with-velocity/precision'].append(p)
            metrics['metric/note-with-velocity/recall'].append(r)
            metrics['metric/note-with-velocity/f1'].append(f)

            p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                      onset_tolerance=onset_tolerance, velocity_tolerance=0.1)
            metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
            metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
            metrics['metric/note-with-offsets-and-velocity/f1'].append(f)

            frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
            metrics['metric/frame/precision'].append(frame_metrics['Precision'])
            metrics['metric/frame/recall'].append(frame_metrics['Recall'])
            metrics['metric/frame/f1'].append(
                hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)
    return metrics


def metric_unboxing(metrics: collections.defaultdict):
    metrics_frame = {
        'pre': metrics['metric/frame/precision'][-1],
        'rec': metrics['metric/frame/recall'][-1],
        'F_m': metrics['metric/frame/f1'][-1],
        'loss': metrics['loss/frame'][-1] if 'loss/frame' in metrics.keys() else None}
    metrics_onset = {
        'pre': metrics['metric/note/precision'][-1],
        'rec': metrics['metric/note/recall'][-1],
        'F_m': metrics['metric/note/f1'][-1],
        'loss': metrics['loss/onset'][-1] if 'loss/onset' in metrics.keys() else None}
    metrics_note_w_offset = {
        'pre': metrics['metric/note-with-offsets/precision'][-1],
        'rec': metrics['metric/note-with-offsets/recall'][-1],
        'F_m': metrics['metric/note-with-offsets/f1'][-1],
        'loss': metrics['loss/offset'][-1] if 'loss/offset' in metrics.keys() else None}
    metrics_note_w_offset_velocity = {
        'pre': metrics['metric/note-with-offsets-and-velocity/precision'][-1],
        'rec': metrics['metric/note-with-offsets-and-velocity/recall'][-1],
        'F_m': metrics['metric/note-with-offsets-and-velocity/f1'][-1],
        'loss': metrics['loss/velocity'][-1] if 'loss/velocity' in metrics.keys() else None}

    return metrics_frame, metrics_onset, metrics_note_w_offset, metrics_note_w_offset_velocity


def counts_unboxing(metrics: collections.defaultdict):
    counts_frame = {
        'tp_sum': int(metrics['metric/frame/tp_sum'][-1]),
        'est_sum': int(metrics['metric/frame/est_sum'][-1]),
        'ref_sum': int(metrics['metric/frame/ref_sum'][-1])}
    counts_onset = {
        'tp_sum': int(metrics['metric/note/tp_sum'][-1]),
        'est_sum': int(metrics['metric/note/est_sum'][-1]),
        'ref_sum': int(metrics['metric/note/ref_sum'][-1])}
    counts_note_w_offset = {
        'tp_sum': int(metrics['metric/note-with-offsets/tp_sum'][-1]),
        'est_sum': int(metrics['metric/note-with-offsets/est_sum'][-1]),
        'ref_sum': int(metrics['metric/note-with-offsets/ref_sum'][-1])}
    counts_note_w_offset_velocity = {
        'tp_sum': int(metrics['metric/note-with-offsets-and-velocity/tp_sum'][-1]),
        'est_sum': int(metrics['metric/note-with-offsets-and-velocity/est_sum'][-1]),
        'ref_sum': int(metrics['metric/note-with-offsets-and-velocity/ref_sum'][-1])}

    return counts_frame, counts_onset, counts_note_w_offset, counts_note_w_offset_velocity


def epoch_metric(metrics_frame: dict, metrics_onset: dict, metrics_note_w_offset: dict,
                 metrics_note_w_offset_velocity: dict):
    em = defaultdict(list)
    em['metric/frame/precision'] = metrics_frame['pre']
    em['metric/frame/recall'] = metrics_frame['rec']
    em['metric/frame/f1'] = metrics_frame['F_m']
    if 'loss' in metrics_frame.keys():
        em['loss/frame'] = metrics_frame['loss']

    em['metric/note/precision'] = metrics_onset['pre']
    em['metric/note/recall'] = metrics_onset['rec']
    em['metric/note/f1'] = metrics_onset['F_m']
    if 'loss' in metrics_onset.keys():
        em['loss/note'] = metrics_onset['loss']

    em['metric/note-w-offset/precision'] = metrics_note_w_offset['pre']
    em['metric/note-w-offset/recall'] = metrics_note_w_offset['rec']
    em['metric/note-w-offset/f1'] = metrics_note_w_offset['F_m']
    if 'loss' in metrics_note_w_offset.keys():
        em['loss/note-w-offsets'] = metrics_note_w_offset['loss']

    em['metric/note-w-offsets-velocity/precision'] = metrics_note_w_offset_velocity['pre']
    em['metric/note-w-offsets-velocity/recall'] = metrics_note_w_offset_velocity['rec']
    em['metric/note-w-offsets-velocity/f1'] = metrics_note_w_offset_velocity['F_m']
    if 'loss' in metrics_note_w_offset_velocity.keys():
        em['loss/note-w-offsets-velocity'] = metrics_note_w_offset_velocity['loss']

    return em
