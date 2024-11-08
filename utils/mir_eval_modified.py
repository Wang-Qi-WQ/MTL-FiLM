import collections
import mir_eval.util as util
import numpy as np
import warnings
from mir_eval.transcription import validate as note_validate
from mir_eval.transcription import match_notes as note_match_notes
from mir_eval.transcription_velocity import validate as vel_validate
from mir_eval.transcription_velocity import match_notes as vel_match_notes
from mir_eval.multipitch import validate as multipitch_validate
from mir_eval.multipitch import resample_multipitch as multipitch_resample_multipitch
from mir_eval.multipitch import frequencies_to_midi as multipitch_frequencies_to_midi
from mir_eval.multipitch import compute_num_freqs as multipitch_compute_num_freqs
from mir_eval.multipitch import compute_num_true_positives as multipitch_compute_num_true_positives
from mir_eval.multipitch import compute_accuracy as multipitch_compute_accuracy


def note_precision_recall_f1_overlap(ref_intervals, ref_pitches, est_intervals,
                                     est_pitches, onset_tolerance=0.05,
                                     pitch_tolerance=50.0, offset_ratio=0.2,
                                     offset_min_tolerance=0.05, strict=False,
                                     beta=1.0):
    note_validate(ref_intervals, ref_pitches, est_intervals, est_pitches)
    # When reference notes are empty, metrics are undefined, return 0's
    if len(ref_pitches) == 0 or len(est_pitches) == 0:
        return 0., 0., 0., 0., 0., 0.

    matching = note_match_notes(ref_intervals, ref_pitches, est_intervals,
                                est_pitches, onset_tolerance=onset_tolerance,
                                pitch_tolerance=pitch_tolerance,
                                offset_ratio=offset_ratio,
                                offset_min_tolerance=offset_min_tolerance,
                                strict=strict)

    precision = float(len(matching)) / len(est_pitches)
    recall = float(len(matching)) / len(ref_pitches)
    f_measure = util.f_measure(precision, recall, beta=beta)

    return precision, recall, f_measure, len(matching), len(est_pitches), len(ref_pitches)


def vel_precision_recall_f1_overlap(
        ref_intervals, ref_pitches, ref_velocities, est_intervals, est_pitches,
        est_velocities, onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=0.2, offset_min_tolerance=0.05, strict=False,
        velocity_tolerance=0.1, beta=1.0):
    vel_validate(ref_intervals, ref_pitches, ref_velocities, est_intervals,
                 est_pitches, est_velocities)
    # When reference notes are empty, metrics are undefined, return 0's
    if len(ref_pitches) == 0 or len(est_pitches) == 0:
        return 0., 0., 0., 0., 0., 0.

    matching = vel_match_notes(
        ref_intervals, ref_pitches, ref_velocities, est_intervals, est_pitches,
        est_velocities, onset_tolerance, pitch_tolerance, offset_ratio,
        offset_min_tolerance, strict, velocity_tolerance)

    precision = float(len(matching)) / len(est_pitches)
    recall = float(len(matching)) / len(ref_pitches)
    f_measure = util.f_measure(precision, recall, beta=beta)

    return precision, recall, f_measure, len(matching), len(est_pitches), len(ref_pitches)


def multipitch_metrics(ref_time, ref_freqs, est_time, est_freqs, **kwargs):
    multipitch_validate(ref_time, ref_freqs, est_time, est_freqs)

    # resample est_freqs if est_times is different from ref_times
    if est_time.size != ref_time.size or not np.allclose(est_time, ref_time):
        warnings.warn("Estimate times not equal to reference times. "
                      "Resampling to common time base.")
        est_freqs = multipitch_resample_multipitch(est_time, est_freqs, ref_time)

    # convert frequencies from Hz to continuous midi note number
    ref_freqs_midi = multipitch_frequencies_to_midi(ref_freqs)
    est_freqs_midi = multipitch_frequencies_to_midi(est_freqs)

    # count number of occurences
    n_ref = multipitch_compute_num_freqs(ref_freqs_midi)
    n_est = multipitch_compute_num_freqs(est_freqs_midi)

    # compute the number of true positives
    true_positives = util.filter_kwargs(
        multipitch_compute_num_true_positives, ref_freqs_midi, est_freqs_midi, **kwargs)

    # compute accuracy metrics
    precision, recall, accuracy = multipitch_compute_accuracy(
        true_positives, n_ref, n_est)

    scores = collections.OrderedDict()
    scores['Precision'] = precision
    scores['Recall'] = recall
    scores['matched'] = int(true_positives.sum())
    scores['n_est'] = int(n_est.sum())
    scores['n_ref'] = int(n_ref.sum())

    return scores


def note_label_matched(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.05,
                       pitch_tolerance=50.0, offset_ratio=0.2,
                       offset_min_tolerance=0.05, strict=False):
    note_validate(ref_intervals, ref_pitches, est_intervals, est_pitches)

    if len(ref_pitches) == 0 or len(est_pitches) == 0:
        return 0., 0., [0]

    matching = note_match_notes(ref_intervals, ref_pitches, est_intervals,
                                est_pitches, onset_tolerance=onset_tolerance,
                                pitch_tolerance=pitch_tolerance,
                                offset_ratio=offset_ratio,
                                offset_min_tolerance=offset_min_tolerance,
                                strict=strict)

    unmatched_label_pairs = list(range(len(ref_pitches)))
    for ref_note_idx, est_note_idx in matching:
        unmatched_label_pairs.remove(ref_note_idx)
    unmatched_label_intervals = []
    for unmatched_note_idx in unmatched_label_pairs:
        unmatched_label_intervals.append(
            (ref_intervals[unmatched_note_idx][1] - ref_intervals[unmatched_note_idx][0]) -
            (est_intervals[unmatched_note_idx][1] - est_intervals[unmatched_note_idx][0]))
    if len(unmatched_label_pairs) == 0:
        return len(ref_pitches), len(unmatched_label_pairs), [0]

    return len(ref_pitches), len(unmatched_label_pairs), unmatched_label_intervals


def note_label_matched_velocity(
        ref_intervals, ref_pitches, ref_velocities,
        est_intervals, est_pitches, est_velocities,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=0.2,
        offset_min_tolerance=0.05, strict=False, velocity_tolerance=0.1):
    note_validate(ref_intervals, ref_pitches, est_intervals, est_pitches)

    if len(ref_pitches) == 0 or len(est_pitches) == 0:
        return 0., 0., [0]

    matching = vel_match_notes(ref_intervals, ref_pitches, ref_velocities,
                               est_intervals, est_pitches, est_velocities,
                               onset_tolerance=onset_tolerance,
                               pitch_tolerance=pitch_tolerance,
                               offset_ratio=offset_ratio,
                               offset_min_tolerance=offset_min_tolerance,
                               strict=strict, velocity_tolerance=velocity_tolerance)

    unmatched_label_pairs = list(range(len(ref_pitches)))
    for ref_note_idx, est_note_idx in matching:
        unmatched_label_pairs.remove(ref_note_idx)
    unmatched_label_intervals = []
    for unmatched_note_idx in unmatched_label_pairs:
        unmatched_label_intervals.append(
            (ref_intervals[unmatched_note_idx][1] - ref_intervals[unmatched_note_idx][0]) -
            (est_intervals[unmatched_note_idx][1] - est_intervals[unmatched_note_idx][0]))
    if len(unmatched_label_pairs) == 0:
        return len(ref_pitches), len(unmatched_label_pairs), [0]

    return len(ref_pitches), len(unmatched_label_pairs), unmatched_label_intervals
