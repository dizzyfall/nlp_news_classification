'''
@Project ：NLPNewsClassification 
@File    ：checkpoints.py
@Author  ：DZY
@Date    ：2025/3/24 19:07 
'''
import os.path

import torch
from src.utils import scheduler

def save_pretrain_checkpoint(model, optimizer, step, checkpoint_pretrain_info_tuple, checkpoints_relative_path,
                             checkpoint_dir_name,
                             checkpoint_file_name):
    """

    Args:
        model:
        optimizer:
        step:
        checkpoint_pretrain_info_tuple:
        checkpoints_relative_path:
        checkpoint_dir_name:
        checkpoint_file_name:

    Returns:

    """
    script_directory = os.getcwd()
    checkpoint_dir_path = os.path.join(script_directory, checkpoints_relative_path, checkpoint_dir_name)
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)
    # 拼接检查点路径
    checkpoint_file_path = os.path.join(checkpoint_dir_path, checkpoint_file_name)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'total_mlm_loss': checkpoint_pretrain_info_tuple[0],
        'total_processed_samples': checkpoint_pretrain_info_tuple[1],
        'cum_time_list': checkpoint_pretrain_info_tuple[2]
    }, checkpoint_file_path)
    print(f"检查点已保存到 {checkpoint_file_path}")


def save_finetuning_model(model, checkpoints_relative_path,
                          checkpoint_dir_name,
                          checkpoint_file_name):
    """

    Args:
        model:
        checkpoints_relative_path:
        checkpoint_dir_name:
        checkpoint_file_name:

    Returns:

    """
    script_directory = os.getcwd()
    checkpoint_dir_path = os.path.join(script_directory, checkpoints_relative_path, checkpoint_dir_name)
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)
    # 拼接检查点路径
    checkpoint_file_path = os.path.join(checkpoint_dir_path, checkpoint_file_name)
    torch.save({
        'model_state_dict': model.state_dict(),
    }, checkpoint_file_path)
    print(f"模型参数已保存到 {checkpoint_file_path}")


def _process_model_state_dict_keys_name(model_state_dict):
    return {key.replace("module.", ""): value for key, value in model_state_dict.items()}


def load_pretrained_model_params(model, checkpoint_relative_path, device):
    script_directory = os.getcwd()
    checkpoint_path = os.path.join(script_directory, checkpoint_relative_path)
    # 加载检查点
    checkpoint = torch.load(checkpoint_path,map_location=device)
    print("device",device)
    print(f"加载检查点路径{checkpoint_relative_path}")

    checkpoint['model_state_dict'] = _process_model_state_dict_keys_name(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print("加载预训练BERT模型参数成功")
    return checkpoint


def load_pretrain_checkpoint(model,device, optimizer=None,checkpoint_relative_path=None,use_checkpoint=False):
    """
    加载预训练模型检查点，用于恢复训练

    Args:
        model:
        device:
        optimizer:
        checkpoint_relative_path:
        use_checkpoint:

    Returns:

    """
    # 从零开始预训练
    # 计算资源足够，不需要分多批数据训练
    if not use_checkpoint:
        step = 0
        total_mlm_loss = 0.0
        total_processed_samples = 0.0
        cum_time_list = []
        return step, total_mlm_loss, total_processed_samples, cum_time_list

    # 计算资源不够，分多批数据训练
    # 加载检查点，并加载模型参数
    checkpoint = load_pretrained_model_params(model, checkpoint_relative_path, device)

    # 加载优化器参数
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for key, tensor in state.items():
                        if isinstance(tensor, torch.Tensor):
                            state[key] = tensor.to(device)  # 移到 GPU
        print("加载预训练BERT优化器参数成功")
        # for i, param_group in enumerate(optimizer.param_groups):
        #     print(f"参数组 {i}, 超参数：{param_group.keys()}")
        #     for j, param in enumerate(param_group['params']):
        #         # 打印参数形状、设备和是否要求梯度
        #         print(f"  参数 {j}: shape={param.shape}, device={param.device}, requires_grad={param.requires_grad}")

    # 恢复训练步数
    step = checkpoint['step']
    total_mlm_loss = checkpoint['total_mlm_loss'],
    total_processed_samples = checkpoint['total_processed_samples']
    cum_time_list = checkpoint['cum_time_list']
    print(f"恢复步数: {step} "
          f"恢复当前总损失: {total_mlm_loss} "
          f"恢复当前已处理样本数: {total_processed_samples} "
          f"恢复当前训练迭代时间列表: {cum_time_list}")

    # 恢复调度器
    pretrain_scheduler=scheduler.BERTScheduler(optimizer=optimizer,current_step=step)
    print("加载预训练BERT调度器参数成功")

    return step, total_mlm_loss[0], total_processed_samples, cum_time_list,pretrain_scheduler



def load_checkpoint(checkpoint_relative_path):
    script_directory = os.getcwd()
    checkpoint_path = os.path.join(script_directory, checkpoint_relative_path)

    # 加载检查点
    checkpoint = torch.load(checkpoint_path)
    print(f"加载检查点路径{checkpoint_relative_path}")

    # 恢复训练步数
    step = checkpoint['step']
    total_mlm_loss = checkpoint['total_mlm_loss'],
    total_processed_samples = checkpoint['total_processed_samples']
    cum_time_list = checkpoint['cum_time_list']
    print(f"恢复步数: {step} "
          f"恢复当前总损失: {total_mlm_loss} "
          f"恢复当前已处理样本数: {total_processed_samples} "
          f"当前平均损失：{total_mlm_loss[0] / total_processed_samples}")
          # f"恢复当前训练迭代时间列表: {cum_time_list}")

load_checkpoint("../../pretrain_results/checkpoints/checkpoint_step_60000/checkpoint_step_60000.pth")
