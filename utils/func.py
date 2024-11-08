import copy
import dataclasses
from typing import Optional

import numpy as np
import pretty_midi

import torch
from constant_macro import *
from utils import metric


def get_piano_roll(pm_instrument_object: pretty_midi.Instrument, fs=100, times=None,
                   pedal_threshold: int = 64, simplest_process: bool = False,
                   max_pedal_speed: float = 0, min_pedal_range: float = 1,
                   pedal_speed_calcu_extra_frm_range: int = 2, maximum_acc: float = 0,
                   note_minimum_frame: int = 40
                   ) -> Optional[dict]:
    """Compute a piano roll matrix of this instrument.
    #code modified from https://craffel.github.io/pretty-midi/#pretty_midi.Instrument.get_piano_roll

    Parameters
    ----------
    pm_instrument_object: pretty_midi.Instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    times : np.ndarray
        Times of the start of each column in the piano roll.
        Default ``None`` which is ``np.arange(0, get_end_time(), 1./fs)``.
    pedal_threshold : int
        Value of control change 64 (sustain pedal) message that is less
        than this value is reflected as pedal-off.  Pedals will be
        reflected as elongation of notes in the piano roll.
        Default is 64.
    simplest_process :
        the simplest processing that only returns the onset, frame, offset and velocity piano rolls
        with the standard 64-threshold sustain/damper pedal note extension method.
    max_pedal_speed : float
        max decrease pedal speed.
    min_pedal_range : float
        min pedal speed detecting range.
    pedal_speed_calcu_extra_frm_range : int
        extra frame to calculate pedal speed, greater than 0.
    maximum_acc : float
        pedal minimum accelerated speed.
    note_minimum_frame : int
        minimum note duration frames, default is 40 frames when frame hop size is 20ms.

    Returns
    -------
    target_piano_rolls: dict
        all the required piano rolls with shape of (frames, pitches(i.e., 88)) or (frames).

        the piano rolls named without "NDM(Note Duration Modification)" refers to the widely used
        note duration extension method with 64-threshold sustain/damper pedal.

        the note piano rolls named with "NDM" consider the extension both from the sustain pedal
        and the sostenuto pedal.

        the pedal piano rolls named with "NDM" consider the pedal extension of itself.

    """
    # If there are no notes, return an empty matrix
    if not pm_instrument_object.notes:
        raise ValueError('None of any notes in instrument.')
    # Get the end time of the last event
    end_time = pm_instrument_object.get_end_time()
    # Extend end time if one was provided
    if times is not None and times[-1] > end_time:
        end_time = times[-1]

    if simplest_process:
        piano_roll = np.zeros((128, int(fs * end_time)), dtype=np.float32)
        onset_roll = np.zeros_like(piano_roll)
        offset_roll = np.zeros_like(piano_roll)
        key_roll = np.zeros_like(piano_roll)
        dp_roll = np.zeros(int(fs * end_time), dtype=np.float32)

        # Drum tracks don't have pitch
        if pm_instrument_object.is_drum:
            raise ValueError('Only support for piano mul-pitch object.')

        # Add up piano roll matrix, note-by-note
        for note in pm_instrument_object.notes:
            note_start_frm = int(np.maximum(int(note.start * fs), 0))
            note_end_frm = int(np.minimum(int(note.end * fs), len(dp_roll) - 1))
            if note_start_frm >= note_end_frm:
                note_start_frm = note_end_frm - 1
            key_roll[note.pitch, note_start_frm:note_end_frm] = note.velocity

        # Process sustain pedals
        pedal_threshold = int(np.clip(pedal_threshold, 0, 127))
        CC_SUSTAIN_PEDAL = 64  # a.k.a. damper pedal
        cc_64_event = [_e for _e in pm_instrument_object.control_changes if _e.number == CC_SUSTAIN_PEDAL]
        if len(cc_64_event) > 1:
            previous_pedal = cc_64_event[0]
            previous_pedal.value = np.clip(previous_pedal.value, 0, 127)
            for now_pedal in cc_64_event[1:]:
                now_pedal.value = np.clip(now_pedal.value, 0, 127)
                dp_roll[int(previous_pedal.time * fs):int(now_pedal.time * fs)] += previous_pedal.value
            # Add up the last pedal event value
            if int(previous_pedal.time * fs) <= int(end_time * fs):
                dp_roll[int(previous_pedal.time * fs):int(end_time * fs)] += previous_pedal.value
        elif len(cc_64_event) == 1 and cc_64_event[0].value != 0:
            decide_time = cc_64_event[0].time
            dp_roll[int(decide_time * fs):] += cc_64_event[0].value

        # extend notes
        for note in pm_instrument_object.notes:
            note_start_frm = int(np.maximum(int(note.start * fs), 0))
            note_end_frm = int(np.minimum(int(note.end * fs), len(dp_roll) - 1))
            if note_start_frm >= note_end_frm:
                note_start_frm = note_end_frm - 1
            # binary pedal switch
            sustain_extra_frm = 0
            while dp_roll[note_end_frm + sustain_extra_frm] >= pedal_threshold:
                sustain_extra_frm += 1
                if (note_end_frm + sustain_extra_frm) >= (len(dp_roll) - 1):
                    break
            # handle re-onset
            if sustain_extra_frm > 0 and np.max(
                    key_roll[note.pitch, note_end_frm:note_end_frm + sustain_extra_frm]) > 0:
                sustain_extra_frm = np.argmax(key_roll[note.pitch, note_end_frm:note_end_frm + sustain_extra_frm] > 0)
            piano_roll[note.pitch, note_start_frm:note_end_frm + sustain_extra_frm] = 1
            onset_roll[note.pitch, note_start_frm] = note.velocity
            offset_roll[note.pitch, np.minimum(piano_roll.shape[1] - 1, note_end_frm + sustain_extra_frm)] = 1

        velocity_roll = copy.deepcopy(onset_roll)
        onset_roll = np.where(onset_roll > 0, 1, 0)
        onset_roll = onset_roll.astype(np.float32)
        target_piano_rolls = {
            'frame_roll': piano_roll,
            'onset_roll': onset_roll,
            'offset_roll': offset_roll,
            'velocity_roll': velocity_roll
        }
        for key, value in target_piano_rolls.items():
            if len(value.shape) == 2:
                target_piano_rolls[key] = value[21:109, :].T
        if times is None:
            return target_piano_rolls
        times = np.array(np.round(times * fs), dtype=np.int64)
        if times.shape[0] == int(fs * end_time):
            return target_piano_rolls
        # else times is not None: mean value interpolation
        target_piano_rolls_interpolated = {}
        for key, value in target_piano_rolls.items():
            if len(value.shape) == 2:
                target_piano_rolls_interpolated[key] = np.zeros((times.shape[0], value.shape[-1]), dtype=value.dtype)
            else:
                target_piano_rolls_interpolated[key] = np.zeros(times.shape[0], dtype=value.dtype)
        for n, (start, end) in enumerate(zip(times[:-1], times[1:])):
            if start < piano_roll.shape[1]:  # if start is >=, leave zeros
                if start == end:
                    end = start + 1
                for key, value in target_piano_rolls.items():
                    if len(value.shape) == 2:
                        target_piano_rolls_interpolated[key][n, :] = np.mean(value[start:end, :], axis=0)
                    else:
                        target_piano_rolls_interpolated[key][n] = np.mean(value[start:end])
        return target_piano_rolls_interpolated

    # Allocate a matrix of zeros - we will add in as we go
    piano_roll = np.zeros((128, int(fs * end_time)), dtype=np.float32)
    piano_NDM_roll = np.zeros_like(piano_roll)
    key_roll = np.zeros_like(piano_roll)
    key_offset_roll = np.zeros_like(piano_roll)
    reg_key_offset_roll = np.ones_like(piano_roll)
    onset_roll = np.zeros_like(piano_roll)
    reg_onset_roll = np.ones_like(piano_roll)
    offset_roll = np.zeros_like(piano_roll)
    offset_NDM_roll = np.zeros_like(piano_roll)
    reg_offset_roll = np.ones_like(piano_roll)
    reg_offset_NDM_roll = np.ones_like(piano_roll)

    # damper pedal (i.e. sustain pedal)
    dp_roll = np.zeros(int(fs * end_time), dtype=np.float32)
    dp_onset_roll = np.zeros_like(dp_roll)
    dp_reg_onset_roll = np.ones_like(dp_roll)
    dp_offset_roll = np.zeros_like(dp_roll)
    dp_offset_NDM_roll = np.zeros_like(dp_roll)
    dp_reg_offset_roll = np.ones_like(dp_roll)
    dp_reg_offset_NDM_roll = np.ones_like(dp_roll)
    dp_speed_roll = np.zeros_like(dp_roll)
    dp_accelerated_speed_roll = np.zeros_like(dp_roll)

    # sostenuto pedal
    sp_roll = np.zeros(int(fs * end_time), dtype=np.float32)
    sp_onset_roll = np.zeros_like(sp_roll)
    sp_reg_onset_roll = np.ones_like(sp_roll)
    sp_offset_roll = np.zeros_like(sp_roll)
    sp_offset_NDM_roll = np.zeros_like(sp_roll)
    sp_reg_offset_roll = np.ones_like(sp_roll)
    sp_reg_offset_NDM_roll = np.ones_like(sp_roll)
    sp_speed_roll = np.zeros_like(sp_roll)
    sp_accelerated_speed_roll = np.zeros_like(sp_roll)

    # una corda (i.e.soft pedal)
    uc_roll = np.zeros(int(fs * end_time), dtype=np.float32)
    uc_onset_roll = np.zeros_like(uc_roll)
    uc_reg_onset_roll = np.ones_like(uc_roll)
    uc_offset_roll = np.zeros_like(uc_roll)
    uc_reg_offset_roll = np.ones_like(uc_roll)

    # Drum tracks don't have pitch
    if pm_instrument_object.is_drum:
        raise ValueError('Only support for piano mul-pitch object.')

    # Add up piano roll matrix, note-by-note
    for note in pm_instrument_object.notes:
        note_start_frm = int(np.maximum(int(note.start * fs), 0))
        note_end_frm = int(np.minimum(int(note.end * fs), len(dp_roll) - 1))
        if note_start_frm >= note_end_frm:
            note_start_frm = note_end_frm - 1
        key_roll[note.pitch, note_start_frm:note_end_frm] = note.velocity

    # Process sustain pedals
    pedal_threshold = int(np.clip(pedal_threshold, 0, 127))
    CC_SUSTAIN_PEDAL = 64  # a.k.a. damper pedal
    cc_64_event = [_e for _e in pm_instrument_object.control_changes if _e.number == CC_SUSTAIN_PEDAL]
    if len(cc_64_event) > 1:
        previous_pedal = cc_64_event[0]
        previous_pedal.value = np.clip(previous_pedal.value, 0, 127)
        for now_pedal in cc_64_event[1:]:
            now_pedal.value = np.clip(now_pedal.value, 0, 127)
            dp_roll[int(previous_pedal.time * fs):int(now_pedal.time * fs)] += previous_pedal.value
            decide_time = (
                now_pedal.time
                if abs(now_pedal.value - pedal_threshold) < abs(previous_pedal.value - pedal_threshold)
                else previous_pedal.time
            )
            # pedal onset & reg-onset mark
            if previous_pedal.value < pedal_threshold <= now_pedal.value:
                dp_onset_roll[int(decide_time * fs)] = 1
                dp_reg_onset_roll[int(decide_time * fs)] = decide_time - (int(decide_time * fs) / fs)
            # pedal offset & reg-offset mark
            if previous_pedal.value >= pedal_threshold > now_pedal.value:
                dp_offset_roll[int(decide_time * fs)] = 1
                dp_offset_NDM_roll[int(decide_time * fs)] = 1
                dp_reg_offset_roll[int(decide_time * fs)] = decide_time - (int(decide_time * fs) / fs)
                dp_reg_offset_NDM_roll[int(decide_time * fs)] = decide_time - (int(decide_time * fs) / fs)
            previous_pedal = now_pedal
        # Add up the last pedal event value
        if int(previous_pedal.time * fs) <= int(end_time * fs):
            dp_roll[int(previous_pedal.time * fs):int(end_time * fs)] += previous_pedal.value
        # calculate pedal speed on each frame
        if pedal_speed_calcu_extra_frm_range < 1:
            ps_range = 1
        else:
            ps_range = pedal_speed_calcu_extra_frm_range
        for frm in range(ps_range, len(dp_roll) - ps_range):
            dp_speed_roll[frm] = (dp_roll[frm + ps_range] - dp_roll[frm - ps_range]) / (2 * ps_range + 1)
        # pad speed_roll first and last
        dp_speed_roll[:ps_range] = dp_speed_roll[ps_range]
        dp_speed_roll[-ps_range:] = dp_speed_roll[-ps_range]
        # calculate pedal accelerated speed
        pacc_range = int(max(ps_range // 2, 1))
        for frm in range(pacc_range, len(dp_roll) - pacc_range):
            dp_accelerated_speed_roll[frm] = (
                    (dp_speed_roll[frm + pacc_range] - dp_speed_roll[frm - pacc_range])
                    / (2 * pacc_range + 1))
        dp_ons_nonzero = np.nonzero(dp_onset_roll)[0]
        for dp_ons_idx, dp_ons_frm in enumerate(dp_ons_nonzero):
            dp_extra_frm = 0
            while dp_roll[dp_ons_frm + dp_extra_frm] < pedal_threshold:
                dp_extra_frm += 1
            while dp_roll[dp_ons_frm + dp_extra_frm] >= pedal_threshold:
                dp_extra_frm += 1
                if (dp_ons_frm + dp_extra_frm) >= (len(dp_roll) - 1):
                    break
                # low pedal speed near pedal_th means an earlier offset
                if ((dp_roll[dp_ons_frm + dp_extra_frm] <= pedal_threshold + min_pedal_range) and
                        (dp_speed_roll[dp_ons_frm + dp_extra_frm] <= 0) and
                        (abs(dp_speed_roll[dp_ons_frm + dp_extra_frm]) <= abs(max_pedal_speed)) and
                        (dp_accelerated_speed_roll[dp_ons_frm + dp_extra_frm] >= 0) and
                        (dp_accelerated_speed_roll[dp_ons_frm + dp_extra_frm] <= maximum_acc)):
                    break
            if dp_ons_frm + dp_extra_frm == len(dp_roll) - 1:
                continue
            if len(np.nonzero(dp_offset_roll[dp_ons_frm:])[0]) == 0:
                dp_previous_offset = len(dp_roll) - dp_ons_frm - 1
            else:
                dp_previous_offset = np.nonzero(dp_offset_roll[dp_ons_frm:])[0][0]
            if dp_previous_offset > dp_extra_frm:
                dp_offset_NDM_roll[dp_ons_frm + dp_extra_frm] = 1
                dp_offset_NDM_roll[dp_ons_frm + dp_previous_offset] = 0
                dp_reg_offset_NDM_roll[dp_ons_frm + dp_extra_frm] = dp_reg_offset_roll[
                    dp_ons_frm + dp_previous_offset]
                dp_reg_offset_NDM_roll[dp_ons_frm + dp_previous_offset] = 1
        dp_reg_onset_roll = get_regression(dp_reg_onset_roll, fs)
        dp_reg_offset_roll = get_regression(dp_reg_offset_roll, fs)
        dp_reg_offset_NDM_roll = get_regression(dp_reg_offset_NDM_roll, fs)
    elif len(cc_64_event) == 1:
        if cc_64_event[0].value != 0:
            decide_time = cc_64_event[0].time
            dp_roll[int(decide_time * fs):] += cc_64_event[0].value
            dp_reg_offset_NDM_roll = np.zeros_like(dp_roll)
            dp_reg_offset_roll = np.zeros_like(dp_roll)
            dp_reg_onset_roll[int(decide_time * fs)] = decide_time - (int(decide_time * fs) / fs)
            dp_reg_onset_roll = get_regression(dp_reg_onset_roll, fs)
        else:
            dp_reg_offset_NDM_roll = np.zeros_like(dp_roll)
            dp_reg_offset_roll = np.zeros_like(dp_roll)
            dp_reg_onset_roll = np.zeros_like(dp_roll)
    else:
        dp_reg_offset_NDM_roll = np.zeros_like(dp_roll)
        dp_reg_offset_roll = np.zeros_like(dp_roll)
        dp_reg_onset_roll = np.zeros_like(dp_roll)
    assert dp_reg_onset_roll.sum() != len(dp_roll)
    assert dp_reg_offset_roll.sum() != len(dp_roll)
    assert dp_reg_offset_NDM_roll.sum() != len(dp_roll)

    CC_SOSTENUTO_PEDAL = 66
    cc_66_event = [_e for _e in pm_instrument_object.control_changes if _e.number == CC_SOSTENUTO_PEDAL]
    if len(cc_66_event) > 1:
        previous_pedal = cc_66_event[0]
        previous_pedal.value = np.clip(previous_pedal.value, 0, 127)
        for now_pedal in cc_66_event[1:]:
            now_pedal.value = np.clip(now_pedal.value, 0, 127)
            sp_roll[int(previous_pedal.time * fs):int(now_pedal.time * fs)] += previous_pedal.value
            decide_time = (
                now_pedal.time
                if abs(now_pedal.value - pedal_threshold) < abs(previous_pedal.value - pedal_threshold)
                else previous_pedal.time
            )
            # pedal onset & reg-onset mark
            if previous_pedal.value < pedal_threshold <= now_pedal.value:
                sp_onset_roll[int(decide_time * fs)] = 1
                sp_reg_onset_roll[int(decide_time * fs)] = decide_time - (int(decide_time * fs) / fs)
            # pedal offset & reg-offset mark
            if previous_pedal.value >= pedal_threshold > now_pedal.value:
                sp_offset_roll[int(decide_time * fs)] = 1
                sp_offset_NDM_roll[int(decide_time * fs)] = 1
                sp_reg_offset_roll[int(decide_time * fs)] = decide_time - (int(decide_time * fs) / fs)
                sp_reg_offset_NDM_roll[int(decide_time * fs)] = decide_time - (int(decide_time * fs) / fs)
            previous_pedal = now_pedal
        # Add up the last pedal event value
        if int(previous_pedal.time * fs) <= int(end_time * fs):
            sp_roll[int(previous_pedal.time * fs):int(end_time * fs)] += previous_pedal.value
        # calculate pedal speed on each frame
        if pedal_speed_calcu_extra_frm_range < 1:
            ps_range = 1
        else:
            ps_range = pedal_speed_calcu_extra_frm_range
        for frm in range(ps_range, len(sp_roll) - ps_range):
            sp_speed_roll[frm] = (sp_roll[frm + ps_range] - sp_roll[frm - ps_range]) / (2 * ps_range + 1)
        # pad speed_roll first and last
        sp_speed_roll[:ps_range] = sp_speed_roll[ps_range]
        sp_speed_roll[-ps_range:] = sp_speed_roll[-ps_range]
        # calculate pedal accelerated speed
        pacc_range = int(max(ps_range // 2, 1))
        for frm in range(pacc_range, len(dp_roll) - pacc_range):
            sp_accelerated_speed_roll[frm] = (
                    (sp_speed_roll[frm + pacc_range] - sp_speed_roll[frm - pacc_range])
                    / (2 * pacc_range + 1))
        sp_ons_nonzero = np.nonzero(sp_onset_roll)[0]
        for sp_ons_idx, sp_ons_frm in enumerate(sp_ons_nonzero):
            sp_extra_frm = 0
            while sp_roll[sp_ons_frm + sp_extra_frm] < pedal_threshold:
                sp_extra_frm += 1
            while sp_roll[sp_ons_frm + sp_extra_frm] >= pedal_threshold:
                sp_extra_frm += 1
                if (sp_ons_frm + sp_extra_frm) >= (len(sp_roll) - 1):
                    break
                # low pedal speed near pedal_th means an earlier offset
                if ((sp_roll[sp_ons_frm + sp_extra_frm] <= pedal_threshold + min_pedal_range) and
                        (sp_speed_roll[sp_ons_frm + sp_extra_frm] <= 0) and
                        (abs(sp_speed_roll[sp_ons_frm + sp_extra_frm]) <= abs(max_pedal_speed)) and
                        (sp_accelerated_speed_roll[sp_ons_frm + sp_extra_frm] >= 0) and
                        (sp_accelerated_speed_roll[sp_ons_frm + sp_extra_frm] <= maximum_acc)):
                    break
            if sp_ons_frm + sp_extra_frm == len(sp_roll) - 1:
                continue
            if len(np.nonzero(sp_offset_roll[sp_ons_frm:])[0]) == 0:
                sp_previous_offset = len(sp_roll) - sp_ons_frm - 1
            else:
                sp_previous_offset = np.nonzero(sp_offset_roll[sp_ons_frm:])[0][0]
            if sp_previous_offset > sp_extra_frm:
                sp_offset_NDM_roll[sp_ons_frm + sp_extra_frm] = 1
                sp_offset_NDM_roll[sp_ons_frm + sp_previous_offset] = 0
                sp_reg_offset_NDM_roll[sp_ons_frm + sp_extra_frm] = sp_reg_offset_roll[
                    sp_ons_frm + sp_previous_offset]
                sp_reg_offset_NDM_roll[sp_ons_frm + sp_previous_offset] = 1
        sp_reg_onset_roll = get_regression(sp_reg_onset_roll, fs)
        sp_reg_offset_roll = get_regression(sp_reg_offset_roll, fs)
        sp_reg_offset_NDM_roll = get_regression(sp_reg_offset_NDM_roll, fs)
    elif len(cc_66_event) == 1:
        if cc_66_event[0].value != 0:
            decide_time = cc_66_event[0].time
            sp_roll[int(decide_time * fs):] += cc_66_event[0].value
            sp_reg_offset_NDM_roll = np.zeros_like(sp_roll)
            sp_reg_offset_roll = np.zeros_like(sp_roll)
            sp_reg_onset_roll[int(decide_time * fs)] = decide_time - (int(decide_time * fs) / fs)
            sp_reg_onset_roll = get_regression(sp_reg_onset_roll, fs)
        else:
            sp_reg_offset_NDM_roll = np.zeros_like(sp_roll)
            sp_reg_offset_roll = np.zeros_like(sp_roll)
            sp_reg_onset_roll = np.zeros_like(sp_roll)
    else:
        sp_reg_offset_NDM_roll = np.zeros_like(sp_roll)
        sp_reg_offset_roll = np.zeros_like(sp_roll)
        sp_reg_onset_roll = np.zeros_like(sp_roll)
    assert sp_reg_onset_roll.sum() != len(sp_roll)
    assert sp_reg_offset_roll.sum() != len(sp_roll)
    assert sp_reg_offset_NDM_roll.sum() != len(sp_roll)

    CC_UNA_CORDA = 67
    cc_67_event = [_e for _e in pm_instrument_object.control_changes if _e.number == CC_UNA_CORDA]
    if len(cc_67_event) > 1:
        previous_pedal = cc_67_event[0]
        previous_pedal.value = np.clip(previous_pedal.value, 0, 127)
        for now_pedal in cc_67_event[1:]:
            now_pedal.value = np.clip(now_pedal.value, 0, 127)
            uc_roll[int(previous_pedal.time * fs):int(now_pedal.time * fs)] += previous_pedal.value
            decide_time = (
                now_pedal.time
                if abs(now_pedal.value - pedal_threshold) < abs(previous_pedal.value - pedal_threshold)
                else previous_pedal.time
            )
            # pedal onset & reg-onset mark
            if previous_pedal.value < 64 <= now_pedal.value:
                uc_onset_roll[int(decide_time * fs)] = 1
                uc_reg_onset_roll[int(decide_time * fs)] = decide_time - (int(decide_time * fs) / fs)
            # pedal offset & reg-offset mark
            if previous_pedal.value >= 64 > now_pedal.value:
                uc_offset_roll[int(decide_time * fs)] = 1
                uc_reg_offset_roll[int(decide_time * fs)] = decide_time - (int(decide_time * fs) / fs)
            previous_pedal = now_pedal
        # Add up the last pedal event value
        if int(previous_pedal.time * fs) <= int(end_time * fs):
            uc_roll[int(previous_pedal.time * fs):int(end_time * fs)] += previous_pedal.value
        uc_reg_onset_roll = get_regression(uc_reg_onset_roll, fs)
        uc_reg_offset_roll = get_regression(uc_reg_offset_roll, fs)
    elif len(cc_67_event) == 1:
        if cc_67_event[0].value != 0:
            decide_time = cc_67_event[0].time
            uc_roll[int(decide_time * fs):] += cc_67_event[0].value
            uc_reg_offset_roll = np.zeros_like(uc_roll)
            uc_reg_onset_roll[int(decide_time * fs)] = decide_time - (int(decide_time * fs) / fs)
            uc_reg_onset_roll = get_regression(uc_reg_onset_roll, fs)
        else:
            uc_reg_offset_roll = np.zeros_like(uc_roll)
            uc_reg_onset_roll = np.zeros_like(uc_roll)
    else:
        uc_reg_offset_roll = np.zeros_like(uc_roll)
        uc_reg_onset_roll = np.zeros_like(uc_roll)
    assert uc_reg_onset_roll.sum() != len(uc_roll)
    assert uc_reg_offset_roll.sum() != len(uc_roll)

    # extend notes
    for note in pm_instrument_object.notes:
        note_start_frm = int(np.maximum(int(note.start * fs), 0))
        note_end_frm = int(np.minimum(int(note.end * fs), len(dp_roll) - 1))
        if note_start_frm >= note_end_frm:
            note_start_frm = note_end_frm - 1

        # pedal speed
        dp_extra_frm = 0
        while dp_roll[note_end_frm + dp_extra_frm] >= pedal_threshold:
            dp_extra_frm += 1
            if (note_end_frm + dp_extra_frm) >= (len(dp_roll) - 1):
                break
            # low pedal speed near pedal_th means an earlier offset
            if ((dp_roll[note_end_frm + dp_extra_frm] <= pedal_threshold + min_pedal_range) and
                    (dp_speed_roll[note_end_frm + dp_extra_frm] <= 0) and
                    (abs(dp_speed_roll[note_end_frm + dp_extra_frm]) <= abs(max_pedal_speed)) and
                    (dp_accelerated_speed_roll[note_end_frm + dp_extra_frm] >= 0) and
                    (dp_accelerated_speed_roll[note_end_frm + dp_extra_frm] <= maximum_acc)):
                break
            # cut low velocity note to note_minimum_frame
            if note_minimum_frame is not None:
                if ((dp_roll[note_end_frm + dp_extra_frm] <= pedal_threshold + min_pedal_range) and
                        (dp_extra_frm + note_end_frm - note_start_frm > note_minimum_frame)):
                    break
        sp_extra_frm = 0
        if sp_roll[note_start_frm] < pedal_threshold:
            while sp_roll[note_end_frm + sp_extra_frm] >= pedal_threshold:
                sp_extra_frm += 1
                if (note_end_frm + sp_extra_frm) >= (len(sp_roll) - 1):
                    break
                # low pedal speed near pedal_th means an earlier offset
                if ((sp_roll[note_end_frm + sp_extra_frm] <= pedal_threshold + min_pedal_range) and
                        (sp_speed_roll[note_end_frm + sp_extra_frm] <= 0) and
                        (abs(sp_speed_roll[note_end_frm + sp_extra_frm]) <= abs(max_pedal_speed)) and
                        (sp_accelerated_speed_roll[note_end_frm + sp_extra_frm] >= 0) and
                        (sp_accelerated_speed_roll[note_end_frm + sp_extra_frm] <= maximum_acc)):
                    break
                # cut low velocity note to note_minimum_frame
                if note_minimum_frame is not None:
                    if ((sp_roll[note_end_frm + sp_extra_frm] <= pedal_threshold + min_pedal_range) and
                            (sp_extra_frm + note_end_frm - note_start_frm > note_minimum_frame)):
                        break
        sustain_extra_frm = max(dp_extra_frm, sp_extra_frm)
        # handle re-onset
        if sustain_extra_frm > 0 and np.max(
                key_roll[note.pitch, note_end_frm:note_end_frm + sustain_extra_frm]) > 0:
            sustain_extra_frm = np.argmax(
                key_roll[note.pitch, note_end_frm:note_end_frm + sustain_extra_frm] > 0)
        # Mark down note in NDM piano roll matrix
        piano_NDM_roll[note.pitch, note_start_frm:note_end_frm + sustain_extra_frm] = 1
        sustain_end_frm = np.minimum(piano_roll.shape[1] - 1, note_end_frm + sustain_extra_frm)
        offset_NDM_roll[note.pitch, sustain_end_frm] = 1
        reg_offset_NDM_roll[note.pitch, sustain_end_frm] = note.end - (note_end_frm / fs)

        # binary pedal switch
        sustain_extra_frm = 0
        while dp_roll[note_end_frm + sustain_extra_frm] >= pedal_threshold:
            sustain_extra_frm += 1
            if (note_end_frm + sustain_extra_frm) >= (len(dp_roll) - 1):
                break
        # handle re-onset
        if sustain_extra_frm > 0 and np.max(
                key_roll[note.pitch, note_end_frm:note_end_frm + sustain_extra_frm]) > 0:
            sustain_extra_frm = np.argmax(key_roll[note.pitch, note_end_frm:note_end_frm + sustain_extra_frm] > 0)

        # Mark down note in piano roll matrix
        piano_roll[note.pitch, note_start_frm:note_end_frm + sustain_extra_frm] = 1
        onset_roll[note.pitch, note_start_frm] = note.velocity
        reg_onset_roll[note.pitch, note_start_frm] = note.start - (note_start_frm / fs)
        # note offsets
        sustain_end_frm = np.minimum(piano_roll.shape[1] - 1, note_end_frm + sustain_extra_frm)
        offset_roll[note.pitch, np.minimum(piano_roll.shape[1] - 1, note_end_frm + sustain_extra_frm)] = 1
        reg_offset_roll[note.pitch, sustain_end_frm] = note.end - (note_end_frm / fs)
        # key offsets
        key_end_frm = np.minimum(piano_roll.shape[1] - 1, int(note.end * fs))
        key_offset_roll[note.pitch, key_end_frm] += 1
        reg_key_offset_roll[note.pitch, key_end_frm] += note.end - (key_end_frm * fs)

    for k in range(piano_roll.shape[0]):
        reg_onset_roll[k, :] = get_regression(reg_onset_roll[k, :], fs)
        reg_offset_roll[k, :] = get_regression(reg_offset_roll[k, :], fs)
        reg_key_offset_roll[k, :] = get_regression(reg_key_offset_roll[k, :], fs)
        reg_offset_NDM_roll[k, :] = get_regression(reg_offset_NDM_roll[k, :], fs)

    velocity_roll = copy.deepcopy(onset_roll)
    key_roll = np.where(key_roll > 0, 1, 0)
    key_roll = key_roll.astype(np.float32)
    onset_roll = np.where(onset_roll > 0, 1, 0)
    onset_roll = onset_roll.astype(np.float32)

    target_piano_rolls = {
        'onset_roll': onset_roll, 'reg_onset_roll': reg_onset_roll,
        'offset_roll': offset_roll, 'reg_offset_roll': reg_offset_roll,
        'offset_NDM_roll': offset_NDM_roll, 'reg_offset_NDM_roll': reg_offset_NDM_roll,
        'frame_roll': piano_roll, 'frame_NDM_roll': piano_NDM_roll,
        'key_roll': key_roll, 'key_offset_roll': key_offset_roll, 'reg_key_offset_roll': reg_key_offset_roll,
        'velocity_roll': velocity_roll,
        'dp_roll': dp_roll,
        'dp_onset_roll': dp_onset_roll, 'dp_reg_onset_roll': dp_reg_onset_roll,
        'dp_offset_roll': dp_offset_roll, 'dp_reg_offset_roll': dp_reg_offset_roll,
        'dp_offset_NDM_roll': dp_offset_NDM_roll, 'dp_reg_offset_NDM_roll': dp_reg_offset_NDM_roll,
        'sp_roll': sp_roll,
        'sp_onset_roll': sp_onset_roll, 'sp_reg_onset_roll': sp_reg_onset_roll,
        'sp_offset_roll': sp_offset_roll, 'sp_reg_offset_roll': sp_reg_offset_roll,
        'sp_offset_NDM_roll': sp_offset_NDM_roll, 'sp_reg_offset_NDM_roll': sp_reg_offset_NDM_roll,
        'uc_roll': uc_roll,
        'uc_onset_roll': uc_onset_roll, 'uc_reg_onset_roll': uc_reg_onset_roll,
        'uc_offset_roll': uc_offset_roll, 'uc_reg_offset_roll': uc_reg_offset_roll
    }

    for key, value in target_piano_rolls.items():
        if len(value.shape) == 2:
            target_piano_rolls[key] = value[21:109, :].T

    if times is None:
        return target_piano_rolls
    times = np.array(np.round(times * fs), dtype=np.int64)
    if times.shape[0] == int(fs * end_time):
        return target_piano_rolls
    # else times is not None: mean value interpolation
    target_piano_rolls_interpolated = {}
    for key, value in target_piano_rolls.items():
        if len(value.shape) == 2:
            target_piano_rolls_interpolated[key] = np.zeros((times.shape[0], value.shape[-1]), dtype=value.dtype)
        else:
            target_piano_rolls_interpolated[key] = np.zeros(times.shape[0], dtype=value.dtype)
    for n, (start, end) in enumerate(zip(times[:-1], times[1:])):
        if start < piano_roll.shape[1]:  # if start is >=, leave zeros
            if start == end:
                end = start + 1
            for key, value in target_piano_rolls.items():
                if len(value.shape) == 2:
                    target_piano_rolls_interpolated[key][n, :] = np.mean(value[start:end, :], axis=0)
                else:
                    target_piano_rolls_interpolated[key][n] = np.mean(value[start:end])
    return target_piano_rolls_interpolated


def get_regression(pn, fs, j_hyp=None):
    """
    https://github.com/bytedance/piano_transcription/blob/1ade7dcd4348add669a67c6e6282456c8c6633bd/utils/utilities.py#L527
    :param pn: input ONSET or OFFSET piano_roll
    :param j_hyp: note that J is not used in this regress alg.,
            because the origin codes bellowing work the same when the 'default-J' is 5 or 6 (when j_factor is 20).
    :param fs: frames per. sec.
    :return: regress piano_roll targets
    """
    step = 1. / fs
    output = np.ones_like(pn)
    j_factor = 20

    locts = np.where(pn < 0.5)[0]
    if len(locts) > 0:
        for t in range(0, locts[0]):
            output[t] = step * (t - locts[0]) - pn[locts[0]]

        for i in range(0, len(locts) - 1):
            for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                output[t] = step * (t - locts[i]) - pn[locts[i]]

            for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                output[t] = step * (t - locts[i + 1]) - pn[locts[i]]

        for t in range(locts[-1], len(pn)):
            output[t] = step * (t - locts[-1]) - pn[locts[-1]]

    output_clip = np.clip(np.abs(output), 0., 0.05) * j_factor
    output_clip = (1. - output_clip)

    return output_clip


def draft_schedule(schedule_idx: str = '28'):
    schedule_note = {
        '0': ('frame_roll', 'onset_roll', 'offset_roll', 'velocity_roll'),
        '1': ('frame_NDM_roll', 'onset_roll', 'offset_NDM_roll', 'velocity_roll'),
        '2': ('frame_roll', 'reg_onset_roll', 'reg_offset_roll', 'velocity_roll'),
        '3': ('frame_NDM_roll', 'reg_onset_roll', 'reg_offset_NDM_roll', 'velocity_roll'),
        '4': ('key_roll', 'onset_roll', 'key_offset_roll', 'velocity_roll'),
        '5': ('key_roll', 'onset_roll', 'reg_key_offset_roll', 'velocity_roll'),
        'x': ('#', '#', '#', '#')
    }
    schedule_pedal = {
        '6': ('dp_roll', 'dp_onset_roll', 'dp_offset_roll', 'dp_roll',
              'sp_roll', 'sp_onset_roll', 'sp_offset_roll', 'sp_roll',
              'uc_roll', 'uc_onset_roll', 'uc_offset_roll', 'uc_roll'),
        '7': ('dp_roll', 'dp_onset_roll', 'dp_offset_NDM_roll', 'dp_roll',
              'sp_roll', 'sp_onset_roll', 'sp_offset_NDM_roll', 'sp_roll',
              'uc_roll', 'uc_onset_roll', 'uc_offset_roll', 'uc_roll'),
        '8': ('dp_roll', 'dp_reg_onset_roll', 'dp_reg_offset_roll', 'dp_roll',
              'sp_roll', 'sp_reg_onset_roll', 'sp_reg_offset_roll', 'sp_roll',
              'uc_roll', 'uc_reg_onset_roll', 'uc_reg_offset_roll', 'uc_roll'),
        '9': ('dp_roll', 'dp_reg_onset_roll', 'dp_reg_offset_NDM_roll', 'dp_roll',
              'sp_roll', 'sp_reg_onset_roll', 'sp_reg_offset_NDM_roll', 'sp_roll',
              'uc_roll', 'uc_reg_onset_roll', 'uc_reg_offset_roll', 'uc_roll'),
        'x': ('#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#')
    }
    if schedule_idx[0] in schedule_note.keys() and schedule_idx[1] in schedule_pedal.keys():
        return [*schedule_note[schedule_idx[0]], *schedule_pedal[schedule_idx[1]], 'wav_spectrum']

    raise ValueError('Missing required training schedule index.')


def vel_expand(vel_pr):
    vel_ = vel_pr.detach().clone().reshape(-1, vel_pr.shape[-1])
    vel__ = vel_.detach().clone()
    vel__[1:, :] = torch.maximum(vel_[:-1, :], vel__[1:, :])
    vel__[:-1, :] = torch.maximum(vel_[1:, :], vel__[:-1, :])
    vel = vel__.reshape(*vel_pr.shape)
    return vel


def pedal_binary_label_makeup(p_frm):
    if p_frm is None:
        return None
    if p_frm.max() == 1.:
        return p_frm
    return torch.where(p_frm >= PEDAL_TH, 1, 0).to(dtype=torch.float32)


def run_one_batch(run_type, model, schedule_idx, input_dict, device, evaluate_term,
                  bce_func=torch.nn.functional.binary_cross_entropy, cce_func=torch.nn.functional.cross_entropy):
    target_schedule = draft_schedule(schedule_idx)
    for key in target_schedule:
        if key in input_dict.keys():
            input_dict[key] = input_dict[key].to(device)
            if run_type == 'test':
                if len(input_dict[key].shape) == 2:
                    input_dict[key] = input_dict[key].reshape(2, -1)
                else:
                    input_dict[key] = input_dict[key].reshape(2, -1, input_dict[key].shape[-1])

    input_dict['#'] = None

    frm_ref = input_dict[target_schedule[0]]
    ons_ref = input_dict[target_schedule[1]]
    off_ref = input_dict[target_schedule[2]]
    vel_ref = input_dict[target_schedule[3]]
    dp_frm_ref = input_dict[target_schedule[4]]
    dp_ons_ref = input_dict[target_schedule[5]]
    dp_off_ref = input_dict[target_schedule[6]]
    dp_vel_ref = input_dict[target_schedule[7]]
    sp_frm_ref = input_dict[target_schedule[8]]
    sp_ons_ref = input_dict[target_schedule[9]]
    sp_off_ref = input_dict[target_schedule[10]]
    sp_vel_ref = input_dict[target_schedule[11]]
    uc_frm_ref = input_dict[target_schedule[12]]
    uc_ons_ref = input_dict[target_schedule[13]]
    uc_off_ref = input_dict[target_schedule[14]]
    uc_vel_ref = input_dict[target_schedule[15]]

    if schedule_idx[0] != 'x' and schedule_idx[1] != 'x':
        # the combination of note and pedal model architecture is discarded in this source code
        raise RuntimeError('Discarded training target')

    elif schedule_idx[0] != 'x':
        frm, ons, off, vel = model(
            input_dict['wav_spectrum']
        )
        loss_frm = bce_func(frm, frm_ref)
        loss_ons = bce_func(ons, ons_ref)
        loss_off = bce_func(off, off_ref)
        vel_ref = vel_expand(vel_ref)
        # velocity dimension transpose for cross entropy calculation
        if vel.shape[1] != 128:
            vel = vel.permute(0, 3, 1, 2)
        loss_vel = cce_func(vel, vel_ref.long())

        losses = {
            'loss/frame': loss_frm,
            'loss/onset': loss_ons,
            'loss/offset': loss_off,
            'loss/velocity': loss_vel
        }
        loss = sum(losses.values())
        losses['loss'] = loss
        data = {
            'frame': (frm, frm_ref),
            'onset': (ons, ons_ref),
            'offset': (off, off_ref),
            'velocity': (vel, vel_ref)
        }
        return losses, data

    elif schedule_idx[1] != 'x':
        (dp_frm, dp_ons, dp_off, dp_vel,
         sp_frm, sp_ons, sp_off, sp_vel,
         uc_frm, uc_ons, uc_off, uc_vel
         ) = model(input_dict['wav_spectrum'])

        dp_frm_ref_b = pedal_binary_label_makeup(dp_frm_ref)
        loss_dp_frm = bce_func(dp_frm, dp_frm_ref_b)
        if dp_ons is None:
            loss_dp_ons = torch.tensor(0.)
        else:
            loss_dp_ons = bce_func(dp_ons, dp_ons_ref)
        loss_dp_off = bce_func(dp_off, dp_off_ref)
        if dp_vel is None:
            loss_dp_vel = torch.tensor(0.)
        else:
            loss_dp_vel = cce_func(dp_vel, dp_vel_ref.long())

        losses = {
            'loss/frame': loss_dp_frm,
            'loss/onset': loss_dp_ons,
            'loss/offset': loss_dp_off,
            'loss/velocity': loss_dp_vel
        }
        loss = sum(losses.values())
        losses['loss'] = loss

        sp_frm_ref_b = pedal_binary_label_makeup(sp_frm_ref)
        """please de-annotate the following codes for sostenuto pedal training"""
        # loss_sp_frm = bce_func(sp_frm, sp_frm_ref_b)
        # loss_sp_ons = bce_func(sp_ons, sp_ons_ref)
        # loss_sp_off = bce_func(sp_off, sp_off_ref)
        # loss_sp_vel = cce_func(sp_vel, sp_vel_ref.long())
        #
        # losses = {
        #     'loss/frame': loss_sp_frm,
        #     'loss/onset': loss_sp_ons,
        #     'loss/offset': loss_sp_off,
        #     'loss/velocity': loss_sp_vel
        # }
        # loss = sum(losses.values())
        # losses['loss'] = loss

        uc_frm_ref_b = pedal_binary_label_makeup(uc_frm_ref)
        """please de-annotate the following codes for una corda pedal training"""
        # loss_uc_frm = bce_func(uc_frm, uc_frm_ref_b)
        # loss_uc_ons = bce_func(uc_ons, uc_ons_ref)
        # loss_uc_off = bce_func(uc_off, uc_off_ref)
        # loss_uc_vel = cce_func(uc_vel, uc_vel_ref.long())
        #
        # losses = {
        #     'loss/frame': loss_uc_frm,
        #     'loss/onset': loss_uc_ons,
        #     'loss/offset': loss_uc_off,
        #     'loss/velocity': loss_uc_vel
        # }
        # loss = sum(losses.values())
        # losses['loss'] = loss
        
        if evaluate_term == 'dp':
            data = {
                'frame': (dp_frm.unsqueeze(-1), dp_frm_ref_b.unsqueeze(-1)),
                'onset': (dp_ons.unsqueeze(-1) if dp_ons is not None else None, dp_ons_ref.unsqueeze(-1)),
                'offset': (dp_off.unsqueeze(-1), dp_off_ref.unsqueeze(-1)),
                'velocity': (dp_vel.unsqueeze(-1) if dp_vel is not None else None, dp_vel_ref.unsqueeze(-1))
            }

        elif evaluate_term == 'sp':
            data = {
                'frame': (sp_frm.unsqueeze(-1), sp_frm_ref_b.unsqueeze(-1)),
                'onset': (sp_ons.unsqueeze(-1) if sp_ons is not None else None, sp_ons_ref.unsqueeze(-1)),
                'offset': (sp_off.unsqueeze(-1), sp_off_ref.unsqueeze(-1)),
                'velocity': (sp_vel.unsqueeze(-1) if sp_vel is not None else None, sp_vel_ref.unsqueeze(-1))
            }
        
        elif evaluate_term == 'uc':
            data = {
                'frame': (uc_frm.unsqueeze(-1), uc_frm_ref_b.unsqueeze(-1)),
                'onset': (uc_ons.unsqueeze(-1) if uc_ons is not None else None, uc_ons_ref.unsqueeze(-1)),
                'offset': (uc_off.unsqueeze(-1), uc_off_ref.unsqueeze(-1)),
                'velocity': (uc_vel.unsqueeze(-1) if uc_vel is not None else None, uc_vel_ref.unsqueeze(-1))
            }
        else:
            return losses, None
        return losses, data
        
    raise ValueError(f'Unexpected schedule index: {schedule_idx}')


def transcribe_one_batch(model, inp, schedule_idx, evaluate_term):
    if schedule_idx[0] != 'x' and schedule_idx[1] != 'x':
        # the combination of note and pedal model architecture is discarded in this source code
        raise RuntimeError('Discarded training target')
    elif schedule_idx[0] != 'x':
        frm, ons, off, vel = model(inp)
        data = {
            'frame': frm,
            'onset': ons,
            'offset': off,
            'velocity': vel
        }
    elif schedule_idx[1] != 'x':
        (dp_frm, dp_ons, dp_off, dp_vel,
         sp_frm, sp_ons, sp_off, sp_vel,
         uc_frm, uc_ons, uc_off, uc_vel
         ) = model(inp)
        if evaluate_term == 'dp':
            data = {
                'frame': dp_frm.unsqueeze(-1),
                'onset': dp_ons.unsqueeze(-1) if dp_ons is not None else dp_frm.unsqueeze(-1),
                'offset': dp_off.unsqueeze(-1),
                'velocity': dp_vel.unsqueeze(-1) if dp_vel is not None else dp_frm.unsqueeze(-1)
            }
        elif evaluate_term == 'sp':
            data = {
                'frame': sp_frm.unsqueeze(-1),
                'onset': sp_ons.unsqueeze(-1),
                'offset': sp_off.unsqueeze(-1),
                'velocity': sp_vel.unsqueeze(-1)
            }
        elif evaluate_term == 'uc':
            data = {
                'frame': uc_frm.unsqueeze(-1),
                'onset': uc_ons.unsqueeze(-1),
                'offset': uc_off.unsqueeze(-1),
                'velocity': uc_vel.unsqueeze(-1)
            }
        else:
            data = None
    else:
        data = None
    return data


def construct_midi_gt(target_prs, schedule_idx, evaluate_term):
    target_schedule = draft_schedule(schedule_idx)
    target_prs['#'] = None

    frm_ref = target_prs[target_schedule[0]]
    ons_ref = target_prs[target_schedule[1]]
    off_ref = target_prs[target_schedule[2]]
    vel_ref = target_prs[target_schedule[3]]
    dp_frm_ref = pedal_binary_label_makeup(target_prs[target_schedule[4]])
    dp_ons_ref = target_prs[target_schedule[5]]
    dp_off_ref = target_prs[target_schedule[6]]
    dp_vel_ref = target_prs[target_schedule[7]]
    sp_frm_ref = pedal_binary_label_makeup(target_prs[target_schedule[8]])
    sp_ons_ref = target_prs[target_schedule[9]]
    sp_off_ref = target_prs[target_schedule[10]]
    sp_vel_ref = target_prs[target_schedule[11]]
    uc_frm_ref = pedal_binary_label_makeup(target_prs[target_schedule[12]])
    uc_ons_ref = target_prs[target_schedule[13]]
    uc_off_ref = target_prs[target_schedule[14]]
    uc_vel_ref = target_prs[target_schedule[15]]

    if schedule_idx[0] != 'x' and schedule_idx[1] != 'x':
        if evaluate_term == 'note':
            data = {
                'frame_ref': frm_ref,
                'onset_ref': ons_ref,
                'offset_ref': off_ref,
                'velocity_ref': vel_ref
            }
        elif evaluate_term == 'dp':
            data = {
                'frame_ref': dp_frm_ref.unsqueeze(-1),
                'onset_ref': dp_ons_ref.unsqueeze(-1),
                'offset_ref': dp_off_ref.unsqueeze(-1),
                'velocity_ref': dp_vel_ref.unsqueeze(-1)
            }
        elif evaluate_term == 'sp':
            data = {
                'frame_ref': sp_frm_ref.unsqueeze(-1),
                'onset_ref': sp_ons_ref.unsqueeze(-1),
                'offset_ref': sp_off_ref.unsqueeze(-1),
                'velocity_ref': sp_vel_ref.unsqueeze(-1)
            }
        elif evaluate_term == 'uc':
            data = {
                'frame_ref': uc_frm_ref.unsqueeze(-1),
                'onset_ref': uc_ons_ref.unsqueeze(-1),
                'offset_ref': uc_off_ref.unsqueeze(-1),
                'velocity_ref': uc_vel_ref.unsqueeze(-1)
            }
        else:
            data = None
    elif schedule_idx[0] != 'x':
        data = {
            'frame_ref': frm_ref,
            'onset_ref': ons_ref,
            'offset_ref': off_ref,
            'velocity_ref': vel_ref
        }
    elif schedule_idx[1] != 'x':
        if evaluate_term == 'dp':
            data = {
                'frame_ref': dp_frm_ref.unsqueeze(-1),
                'onset_ref': dp_ons_ref.unsqueeze(-1),
                'offset_ref': dp_off_ref.unsqueeze(-1),
                'velocity_ref': dp_vel_ref.unsqueeze(-1)
            }
        elif evaluate_term == 'sp':
            data = {
                'frame_ref': sp_frm_ref.unsqueeze(-1),
                'onset_ref': sp_ons_ref.unsqueeze(-1),
                'offset_ref': sp_off_ref.unsqueeze(-1),
                'velocity_ref': sp_vel_ref.unsqueeze(-1)
            }
        elif evaluate_term == 'uc':
            data = {
                'frame_ref': uc_frm_ref.unsqueeze(-1),
                'onset_ref': uc_ons_ref.unsqueeze(-1),
                'offset_ref': uc_off_ref.unsqueeze(-1),
                'velocity_ref': uc_vel_ref.unsqueeze(-1)
            }
        else:
            data = None
    else:
        data = None
    return data


class StairDownLR(object):
    def __init__(self, optimizer=None, lr=6e-4, stair_length=int(770 * LOAD_PORTION * 5 * 32 / BATCH_SIZE),
                 decay=0.02, reload_step=0):
        assert optimizer is not None
        self._optimizer = optimizer
        self._lr = lr
        self._stair_length = stair_length
        self._decay = decay
        self._curr_step = reload_step

    def lr(self):
        return self._lr

    def lr_hard_set(self):
        self._lr = self._lr * 0.3
        for parm in self._optimizer.param_groups:
            parm['lr'] = self._lr

    def step(self):
        self._curr_step += 1
        if self._curr_step % self._stair_length == 0:
            self._lr = self._lr * (1 - self._decay)
            for parm in self._optimizer.param_groups:
                parm['lr'] = self._lr


@dataclasses.dataclass
class TestModelOutDatapack:
    b_idx: int
    eval_obj: dict
    eval_ref: dict
    losses: dict


@dataclasses.dataclass
class TestModelOutDatapack:
    b_idx: int
    eval_obj: dict
    eval_ref: dict
    losses: dict


def validate_parallel_metric_compute(datapack: TestModelOutDatapack, ons_th=0.5, frm_th=0.5, off_th=0.5,
                                     for_pedal=False):
    metrics_tensorboard = metric.evaluate(eval_obj=datapack.eval_obj, eval_ref=datapack.eval_ref,
                                          losses=datapack.losses, onset_threshold=ons_th, frame_threshold=frm_th,
                                          offset_threshold=off_th, decoding_with_offsets=False,
                                          hpt_heuristic_decoding=True, for_pedal=for_pedal)
    metrics_frame, metrics_onset, metrics_note_w_offset, metrics_note_w_offset_velocity = (
        metric.metric_unboxing(metrics_tensorboard))
    return metrics_frame, metrics_onset, metrics_note_w_offset, metrics_note_w_offset_velocity


def validate_batch_metric_mean(metric_total):
    metrics_frame = []
    metrics_onset = []
    metrics_note_w_offset = []
    metrics_note_w_offset_velocity = []
    metrics_frame_details = {'pre': [], 'rec': [], 'F_m': [], 'loss': []}
    metrics_onset_details, metrics_note_w_offset_details, metrics_note_w_offset_velocity_details = (
        copy.deepcopy(metrics_frame_details), copy.deepcopy(metrics_frame_details),
        copy.deepcopy(metrics_frame_details))
    for batch_idx, batch_metric in enumerate(metric_total):
        metrics_frame = metric.metrics_mean(batch_metric[0], metrics_frame_details)
        metrics_onset = metric.metrics_mean(batch_metric[1], metrics_onset_details)
        metrics_note_w_offset = metric.metrics_mean(batch_metric[2], metrics_note_w_offset_details)
        metrics_note_w_offset_velocity = metric.metrics_mean(batch_metric[3],
                                                             metrics_note_w_offset_velocity_details)
    return metrics_frame, metrics_onset, metrics_note_w_offset, metrics_note_w_offset_velocity
