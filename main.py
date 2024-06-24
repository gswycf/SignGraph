import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
import shutil
import inspect
import time
from collections import OrderedDict

faulthandler.enable()
import utils 
from seq_scripts import seq_train, seq_eval
from torch.cuda.amp import autocast as autocast
from utils.misc import *
class Processor():
    def __init__(self, arg):
        self.arg = arg
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        shutil.copy2(__file__, self.arg.work_dir)
        shutil.copy2('./configs/baseline.yaml', self.arg.work_dir)
        shutil.copy2('./modules/tconv.py', self.arg.work_dir)
        shutil.copy2('./modules/resnet.py', self.arg.work_dir)
        shutil.copy2('./modules/gcn_lib/temgraph.py', self.arg.work_dir)
        torch.backends.cudnn.benchmark = True
        if type(self.arg.device) is not int:
            init_distributed_mode(self.arg)
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()


        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.model, self.optimizer = self.loading()




    def start(self):
        if self.arg.phase == 'train':
            best_dev = {"wer":200.0, "del":100.0,"ins":100.0}
            best_tes = {"wer": 200.0, "del": 100.0, "ins": 100.0}
            best_epoch = 0
            total_time = 0
            epoch_time = 0
            if is_main_process():
                self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            seq_model_list = []
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.optimizer_args['num_epoch']):
                save_model = epoch % self.arg.save_interval == 0
                eval_model = epoch % self.arg.eval_interval == 0
                epoch_time = time.time()
                seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder)
                dev_wer={}
                dev_wer['wer']=0
                if is_main_process():
                    if eval_model:
                        dev_wer = seq_eval(self.arg, self.data_loader['dev'], self.model, self.device,
                                           'dev', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                        test_wer = seq_eval(self.arg, self.data_loader['test'], self.model, self.device,
                                           'test', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                        self.recoder.print_log("Dev WER: {:05.2f}% DEV del {:05.2f}% DEV ins {:05.2f}%".format(dev_wer['wer'], dev_wer['del'], dev_wer['ins']))
                        self.recoder.print_log("Test WER: {:05.2f}% Test del {:05.2f}% Test ins {:05.2f}%".format(test_wer['wer'], test_wer['del'],
                                                                                            test_wer['ins']))

                    if dev_wer["wer"] < best_dev["wer"]:
                        best_dev = dev_wer
                        best_tes = test_wer
                        best_epoch = epoch
                        model_path = "{}_best_model.pt".format(self.arg.work_dir)
                        self.save_model(epoch, model_path)
                        self.recoder.print_log('Save best model')
                    self.recoder.print_log('Best_dev: {:05.2f}, {:05.2f}, {:05.2f}, '
                                           'Best_test: {:05.2f}, {:05.2f}, {:05.2f},'
                                           'Epoch : {}'.format(best_dev["wer"], best_dev["del"], best_dev["ins"],
                                                               best_tes["wer"],best_tes["del"],best_tes["ins"], best_epoch))
                    if save_model:
                        model_path = "{}dev_{:05.2f}_epoch{}_model.pt".format(self.arg.work_dir, dev_wer['wer'], epoch)
                        seq_model_list.append(model_path)
                        print("seq_model_list", seq_model_list)
                        self.save_model(epoch, model_path)
                    epoch_time = time.time() - epoch_time
                    total_time += epoch_time
                    torch.cuda.empty_cache()
                    self.recoder.print_log('Epoch {} costs {} mins {} seconds'.format(epoch, int(epoch_time)//60, int(epoch_time)%60))
                self.recoder.print_log('Training costs {} hours {} mins {} seconds'.format(int(total_time)//60//60, int(total_time)//60%60, int(total_time)%60))
        elif self.arg.phase == 'test' and is_main_process():
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                print('Please appoint --weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            train_wer = seq_eval(self.arg, self.data_loader["train_eval"], self.model, self.device,
                                 "train", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            dev_wer = seq_eval(self.arg, self.data_loader["dev"], self.model, self.device,
                               "dev", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device,
                                "test", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            self.recoder.print_log('Evaluation Done.\n')
        elif self.arg.phase == "features":
            for mode in ["train", "dev", "test"]:
                seq_feature_generation(
                    self.data_loader[mode + "_eval" if mode == "train" else mode],
                    self.model, self.device, mode, self.arg.work_dir, self.recoder
                )

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        if len(self.device.gpu_list)>1:
            model = self.model.module
        else:
            model= self.model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    def loading(self):
        self.device.set_device(self.arg.device)



        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        shutil.copy2(inspect.getfile(model_class), self.arg.work_dir)
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)

        self.kernel_sizes = model.conv1d.kernel_size
        model = self.model_to_device(model)
        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            print("using dataparalleling...")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model.to(self.arg.local_rank))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.arg.local_rank])
        else:
            model.cuda()
        # model.cuda()
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        # weights = self.modified_weights(state_dict['model_state_dict'])
        s_dict = model.state_dict()
        for name in weights:
            if name not in s_dict:
                print(name)
                continue
            if s_dict[name].shape == weights[name].shape:
                s_dict[name] = weights[name]
        model.load_state_dict(s_dict, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.arg.local_rank)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recoder.print_log("Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):
        print("Loading Dataprocessing")
        self.feeder = import_class(self.arg.feeder)
        shutil.copy2(inspect.getfile(self.feeder), self.arg.work_dir)
        if self.arg.dataset == 'CSL':
            dataset_list = zip(["train", "dev"], [True, False])
        elif 'phoenix' in self.arg.dataset:
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False]) 
        elif self.arg.dataset == 'CSL-Daily':
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, kernel_size= self.kernel_sizes, dataset=self.arg.dataset, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading Dataprocessing finished.")
    def init_fn(self, worker_id):
        np.random.seed(int(self.arg.random_seed)+worker_id)


    def build_dataloader(self, dataset, mode, train_flag):
        if len(self.device.gpu_list) > 1:
            if train_flag:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=train_flag)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)
            batch_size = self.arg.batch_size if mode == "train" else self.arg.test_batch_size
            loader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=self.feeder.collate_fn,
                num_workers=self.arg.num_worker,
                pin_memory=True,
                worker_init_fn=self.init_fn,
            )
            return loader
        else:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size= self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
                shuffle=train_flag,
                drop_last=train_flag,
                num_workers=self.arg.num_worker,  # if train_flag else 0
                collate_fn=self.feeder.collate_fn,
                pin_memory=True,
                worker_init_fn=self.init_fn,
            )


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    processor = Processor(args)
    utils.pack_code("./", args.work_dir)
    processor.start()
 
