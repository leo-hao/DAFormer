import argparse
import json
import os
# subprocess 模块允许我们启动一个新进程，并连接到它们的输入/输出/错误管道，从而获取返回值。
import subprocess
# UUID（Universally Unique Identifier）是通用唯一识别码
import uuid
from datetime import datetime

import torch
from experiments import generate_experiment_cfgs
from mmcv import Config, get_git_hash
from tools import train


def run_command(command):
    # Popen 是 subprocess的核心，子进程的创建和管理都靠它处理。
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.decode('utf-8'), end='')

# 来实现将服务器端的文件备份到客户端来
def rsync(src, dst):
    # f’{}’ 用法等同于 format用法的简单使用，更加方便
    rsync_cmd = f'rsync -a {src} {dst}'
    print(rsync_cmd)
    run_command(rsync_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ArgumentParser.add_mutually_exclusive_group(required=False) 创建一个互斥组。 argparse 将会确保互斥组中只有一个参数在命令行中可用
    # required=True，表示在互斥组中至少有一个参数是需要的:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--exp',
        type=int,
        default=None,
        help='Experiment id as defined in experiment.py',
    )
    group.add_argument(
        '--config',
        default=None,
        help='Path to config file',
    )
    parser.add_argument(
        '--machine', type=str, choices=['local'], default='local')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    assert (args.config is None) != (args.exp is None), \
        'Either config or exp has to be defined.'

    GEN_CONFIG_DIR = 'configs/generated/'
    JOB_DIR = 'jobs'
    cfgs, config_files = [], []

    # Training with Predefined Config
    # 从预训练的config开始训练
    if args.config is not None:
        cfg = Config.fromfile(args.config)
        # Specify Name and Work Directory
        exp_name = f'{args.machine}-{cfg["exp"]}'
        unique_name = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
                      f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
        child_cfg = {
            '_base_': args.config.replace('configs', '../..'),
            'name': unique_name,
            'work_dir': os.path.join('work_dirs', exp_name, unique_name),
            'git_rev': get_git_hash()
        }
        cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{child_cfg['name']}.json"
        os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
        assert not os.path.isfile(cfg_out_file)
        with open(cfg_out_file, 'w') as of:
            json.dump(child_cfg, of, indent=4)
        config_files.append(cfg_out_file)
        cfgs.append(cfg)

    # Training with Generated Configs from experiments.py
    if args.exp is not None:
        exp_name = f'{args.machine}-exp{args.exp}'
        cfgs = generate_experiment_cfgs(args.exp)
        # Generate Configs
        for i, cfg in enumerate(cfgs):
            if args.debug:
                cfg.setdefault('log_config', {})['interval'] = 10
                cfg['evaluation'] = dict(interval=200, metric='mIoU')
                if 'dacs' in cfg['name']:
                    cfg.setdefault('uda', {})['debug_img_interval'] = 10
                    # cfg.setdefault('uda', {})['print_grad_magnitude'] = True
            # Generate Config File
            cfg['name'] = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
                          f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
            cfg['work_dir'] = os.path.join('work_dirs', exp_name, cfg['name'])
            cfg['git_rev'] = get_git_hash()
            cfg['_base_'] = ['../../' + e for e in cfg['_base_']]
            cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{cfg['name']}.json"
            os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
            assert not os.path.isfile(cfg_out_file)
            with open(cfg_out_file, 'w') as of:
                json.dump(cfg, of, indent=4)
            config_files.append(cfg_out_file)

    if args.machine == 'local':
        for i, cfg in enumerate(cfgs):
            print('Run job {}'.format(cfg['name']))
            train.main([config_files[i]])
            torch.cuda.empty_cache()
    else:
        raise NotImplementedError(args.machine)
