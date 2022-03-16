import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def acc_precision_recall_f1_score(status,status_pred):
    assert status.shape == status_pred.shape
    
    if type(status)!=np.ndarray:
        status = status.detach().cpu().numpy().squeeze()   
    if type(status_pred)!=np.ndarray: 
        status_pred = status_pred.detach().cpu().numpy().squeeze()
    

    status      = status.reshape(status.shape[0], -1)
    status_pred = status_pred.reshape(status_pred.shape[0],-1)
    accs, precisions, recalls, f1_scores = [], [], [], []


    for i in range(status.shape[0]):
        tn, fp, fn, tp = confusion_matrix(status[i, :], status_pred[i, :], labels=[0, 1]).ravel()
        acc            = (tn + tp) / (tn + fp + fn + tp)
        precision      = tp / np.max((tp + fp, 1e-9))
        recall         = tp / np.max((tp + fn, 1e-9))
        f1_score       = 2 * (precision * recall) / np.max((precision + recall, 1e-9))

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return np.array(accs), np.array(precisions), np.array(recalls), np.array(f1_scores)

def regression_errors(pred, label):
    assert pred.shape == label.shape
    
    if type(pred)!=np.ndarray:
        pred = pred.detach().cpu().numpy().squeeze()
    if type(label)!=np.ndarray:
        label = label.detach().cpu().numpy().squeeze()  

    pred     = pred.reshape(pred.shape[0],-1)
    label    = label.reshape(label.shape[0],-1)
    epsilon  = np.full(label.shape, 1e-9)
    mae_arr, mre_arr = [], []

    for i in range(label.shape[0]):
        abs_diff = np.abs(label[i,:] - pred[i,:])
        mae      = np.mean(abs_diff)
        mre_num  = np.nan_to_num(abs_diff)
        mre_den  = np.max((label[i,:], pred[i,:], epsilon[i,:]), axis=0)
        mre = np.mean(mre_num/mre_den)
        mae_arr.append(mae)
        mre_arr.append(mre)

    return np.array(mae_arr), np.array(mre_arr)




def compute_status(data,threshold,min_on,min_off):
    status = np.zeros(data.shape)
    if len(data.squeeze().shape) == 1:
        columns = 1
    else:
        columns = data.squeeze().shape[-1]

    threshold = [threshold]
    min_on    = [min_on]
    min_off   = [min_off]

    for i in range(columns):
        initial_status = data[:, i] >= threshold[i]
        status_diff = np.diff(initial_status)
        events_idx = status_diff.nonzero()

        events_idx = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(
                events_idx, events_idx.size, initial_status.size)

        events_idx = events_idx.reshape((-1, 2))
        on_events = events_idx[:, 0].copy()
        off_events = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events    = on_events[off_duration > min_off[i]]
            off_events   = off_events[np.roll(off_duration, -1) > min_off[i]]

            on_duration = off_events - on_events
            on_events   = on_events[on_duration >= min_on[i]]
            off_events  = off_events[on_duration >= min_on[i]]
            assert len(on_events) == len(off_events)

        temp_status = data[:, i].copy()
        temp_status[:] = 0
        for on, off in zip(on_events, off_events):
            temp_status[on: off] = 1
        status[:, i] = temp_status

    return status
