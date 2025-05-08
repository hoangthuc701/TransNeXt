# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

"""
Train and eval functions used in main.py
"""

# ====== Import thư viện cần thiết ======
import math
import sys
from typing import Iterable, Optional
import torch
from timm.data import Mixup                        # Augmentation Mixup
from timm.utils import accuracy, ModelEma          # Accuracy tính top-1, top-5; EMA mô hình
from timm.utils.clip_grad import dispatch_clip_grad # Hỗ trợ cắt gradient
from losses import DistillationLoss                # Hàm mất mát có knowledge distillation
import utils                                       # Tiện ích logging & thống kê

# ========== HUẤN LUYỆN 1 EPOCH ==========

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, grad_accum_steps=1,
                    fp32=False):

    # Đặt chế độ train (có thể tắt BatchNorm nếu cần)
    model.train(set_training_mode)

    # Khởi tạo logger để theo dõi loss, learning rate,...
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    # Duyệt từng batch trong DataLoader
    for batch_idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Áp dụng mixup nếu có
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # Tính output và loss với autocast (AMP) nếu không dùng fp32
        with torch.cuda.amp.autocast(enabled=not fp32):
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        # Kiểm tra xem có cần cập nhật gradient không (hỗ trợ gradient accumulation)
        update_grad = (batch_idx + 1) % grad_accum_steps == 0
        loss_update = loss / grad_accum_steps

        loss_value = loss.item()

        # Nếu loss không hợp lệ (NaN, Inf) thì dừng huấn luyện
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Với optimizer hỗ trợ second-order (như AdaHessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        # Gọi loss_scaler (dùng GradScaler) để backward + step
        loss_scaler(loss_update, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order, update_grad=update_grad)

        # Nếu đã step thì xóa gradient (dành cho accumulation)
        if update_grad:
            optimizer.zero_grad()

        # Đồng bộ GPU
        torch.cuda.synchronize()

        # Cập nhật EMA model nếu có
        if update_grad and model_ema is not None:
            model_ema.update(model)

        # Cập nhật thống kê
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # Đồng bộ chỉ số giữa các GPU
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)

    # Trả về loss trung bình và các metric khác
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# ========== HÀM ĐÁNH GIÁ TRÊN TẬP VALIDATION ==========

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()  # Loss mặc định khi eval (không dùng distillation)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()  # Đặt mô hình sang chế độ evaluation (BN, dropout tắt)

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():  # Dùng mixed precision nếu hỗ trợ
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))  # Tính top-1 và top-5 accuracy
        batch_size = images.shape[0]

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# ========== LỚP GRADIENT SCALER TÙY CHỈNH (cho AMP) ==========

class NativeScalerAccum:
    state_dict_key = "amp_scaler"  # Tên key khi lưu vào checkpoint

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()  # Sử dụng scaler mặc định từ PyTorch

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False,
                 update_grad=True):
        # Scale loss rồi backward (tạo graph nếu là second-order optimizer)
        self._scaler.scale(loss).backward(create_graph=create_graph)

        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # Giải scale trước khi clip
                dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()  # Dùng để lưu trạng thái vào checkpoint

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)  # Tải trạng thái từ checkpoint
