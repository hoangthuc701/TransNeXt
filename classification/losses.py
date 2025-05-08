# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

"""
Implements the knowledge distillation loss
"""

import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    Đây là một lớp loss mở rộng từ tiêu chuẩn (cross entropy, MSE,...) 
    bằng cách cộng thêm loss từ mô hình giáo viên (teacher) để hướng dẫn mô hình học sinh (student).
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion            # Hàm loss chính (thường là cross_entropy)
        self.teacher_model = teacher_model              # Mô hình giáo viên (đã pre-trained, không fine-tune)
        assert distillation_type in ['none', 'soft', 'hard']  # Kiểm tra loại distillation
        self.distillation_type = distillation_type      # 'soft' dùng KL-div; 'hard' dùng label của teacher
        self.alpha = alpha                              # Trọng số giữa loss gốc và loss distillation
        self.tau = tau                                  # Nhiệt độ dùng cho soft distillation (temperature T)

    def forward(self, inputs, outputs, labels):
        """
        inputs: đầu vào ban đầu (dùng để chạy teacher model)
        outputs: output của mô hình học sinh
                 - nếu là Tensor: chỉ có output gốc
                 - nếu là Tuple[Tensor, Tensor]: (output, output_distill)
        labels: ground truth label (dùng cho base criterion)
        """

        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # Trường hợp mô hình trả ra tuple: (output chính, output dùng cho distillation)
            outputs, outputs_kd = outputs

        # Tính loss gốc (cross entropy giữa output và label)
        base_loss = self.base_criterion(outputs, labels)

        # Nếu không dùng distillation thì chỉ trả về loss gốc
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            # Nếu bật distillation nhưng không có output phụ để so sánh với teacher → báo lỗi
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")

        # Tính output của teacher nhưng không cho gradient backprop
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # Tính KL-divergence giữa student và teacher với softmax được chia nhiệt độ (T)
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),             # Soft label từ student
                F.log_softmax(teacher_outputs / T, dim=1),        # Soft label từ teacher
                reduction='sum',                                  # Tổng tất cả logit
                log_target=True                                   # Teacher đã log-softmax
            ) * (T * T) / outputs_kd.numel()                      # Nhân với T^2 và chuẩn hóa theo số phần tử

        elif self.distillation_type == 'hard':
            # Teacher không dùng softmax, chỉ lấy lớp có xác suất cao nhất làm "pseudo-label"
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        # Kết hợp giữa loss gốc và loss distillation theo trọng số alpha
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha

        return loss
