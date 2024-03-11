import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from einops import rearrange
from collections import defaultdict
from utils.misc import *



def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    optimizer.scheduler.step(epoch_idx)
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    scaler = GradScaler()
    for batch_idx, data in enumerate(tqdm(loader)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        optimizer.zero_grad()
        with autocast():

            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            if len(device.gpu_list)>1:
                loss = model.module.criterion_calculation(ret_dict, label, label_lgt)
            else:
                loss = model.criterion_calculation(ret_dict, label, label_lgt)

            # b t c h w
            # ret_dict_re = model(torch.flip(vid, dims=[3]), vid_lgt, label=label_re, label_lgt=label_lgt)
            # loss += model.criterion_calculation(ret_dict_re, label_re, label_lgt)

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('loss is nan')
            print(str(data[1])+'  frames', str(data[3])+'  glosses')
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update()
        # nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
        if len(device.gpu_list)>1:
            torch.cuda.synchronize()
            # nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
            torch.distributed.reduce(loss, dst=0)

        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0 and is_main_process():
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
        del ret_dict
        del loss
    optimizer.scheduler.step()
    if is_main_process():
        recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder, evaluate_tool="python"):
    model.eval()
    results=defaultdict(dict)

    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        info = [d['fileid'] for d in data[-1]]
        gloss = [d['label'] for d in data[-1]]
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            for inf, conv_sents, recognized_sents, gl in zip(info, ret_dict['conv_sents'], ret_dict['recognized_sents'], gloss):
                results[inf]['conv_sents'] = conv_sents
                results[inf]['recognized_sents'] = recognized_sents
                results[inf]['gloss'] = gl
    gls_hyp = [' '.join(results[n]['conv_sents']) for n in results]
    gls_ref = [results[n]['gloss'] for n in results]
    wer_results_con = wer_list(hypotheses=gls_hyp, references=gls_ref)
    gls_hyp = [' '.join(results[n]['recognized_sents']) for n in results]
    wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
    if wer_results['wer'] < wer_results_con['wer']:
        reg_per = wer_results
    else:
        reg_per = wer_results_con
    recoder.print_log('\tEpoch: {} {} done. Conv wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results_con['wer'], wer_results_con['ins'], wer_results_con['del']),
        f"{work_dir}/{mode}.txt")
    recoder.print_log('\tEpoch: {} {} done. LSTM wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results['wer'], wer_results['ins'], wer_results['del']), f"{work_dir}/{mode}.txt")

    return {"wer":reg_per['wer'], "ins":reg_per['ins'], 'del':reg_per['del']}
 


from utils.metrics import wer_list
def seq_feature_generation(loader, model, device, predix_dir, recoder, epoch=6666):
    model.eval()
    results=defaultdict(dict)
    prefix = os.path.join(predix_dir, "processed/featuresG/")
    for mode in ['dev', 'test', 'train']:
        if not os.path.exists(prefix+mode):
            os.makedirs(prefix+mode)

    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        info = data[4]
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])

        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)

        for inf, conv_sents, recognized_sents, gloss in zip(info, ret_dict['conv_sents'], ret_dict['recognized_sents'], label):
            results[inf]['conv_sents'] = conv_sents
            results[inf]['recognized_sents'] = recognized_sents
            results[inf]['gloss'] = gloss
        feature_dic= defaultdict()
        # b t d
        # print(type(ret_dict['framewise_features']))
        ret_dict['framewise_features'] = rearrange(ret_dict['framewise_features'], 'b d t -> b t d')
        ret_dict['visual_features'] = rearrange(ret_dict['visual_features'], 't b d -> b t d')
        ret_dict['temproal_features'] = rearrange(ret_dict['temproal_features'], 't b d -> b t d')
        for inf, framewise_features, visual_features, temproal_features, feat_len, conv_logits, sequence_logits, x_lgt in \
                zip(info, ret_dict['framewise_features'], ret_dict['visual_features'], ret_dict['temproal_features'], ret_dict['feat_len'], ret_dict['conv_logits'], ret_dict['sequence_logits'], vid_lgt):
            # print(feat_len, feat_len.int().item())
            feat_len= feat_len.int().item()

            print(framewise_features.shape, visual_features.shape, temproal_features.shape)
            feature_dic = { 'framewise_features': framewise_features[:x_lgt].cpu().detach(),
                            'visual_features': visual_features[:feat_len].cpu().detach(),
                            'temproal_features': temproal_features[:feat_len].cpu().detach(),
                            'conv_logits': conv_logits.cpu().detach(),
                            'sequence_logits': sequence_logits.cpu().detach(),
                            'conv_sents': results[inf]['conv_sents'],
                            'recognized_sents': results[inf]['recognized_sents']
                            }
            # print("processing", prefix+inf)
            torch.save(feature_dic, prefix+inf+'.pkl')

        gls_hyp = [' '.join(results[n]['conv_sents']) for n in results]
        gls_ref = [results[n]['gloss'] for n in results]
        wer_results_con = wer_list(hypotheses=gls_hyp, references=gls_ref)
        gls_hyp = [' '.join(results[n]['recognized_sents']) for n in results]
        wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)

        recoder.print_log('\tEpoch: {} {} done. Conv wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
            epoch, mode, wer_results_con['wer'], wer_results_con['ins'], wer_results_con['del']),
            f"{work_dir}/{mode}.txt")
        recoder.print_log('\tEpoch: {} {} done. LSTM wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
            epoch, mode, wer_results['wer'], wer_results['ins'], wer_results['del']), f"{work_dir}/{mode}.txt")



'''
18.01%   4.67%,  1.81%,  conv reslut  19.34%   5.71%,  1.63
 20.94%   4.15%,  2.39%,  conv reslut  21.08%   5.37%,  2.09%
'''