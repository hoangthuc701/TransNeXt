# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""

import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import mmcv

# ======= TRACKING METRICS WITH SMOOTHING =======

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a window or the global series average."""

    def __init__(self, window_size=20, fmt=None):
        # Khởi tạo một hàng đợi (deque) giới hạn số lượng phần tử bằng window_size
        # Tổng giá trị, số lượng mẫu, và định dạng in mặc định
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        # Cập nhật giá trị mới, nhân với số lượng mẫu nếu cần
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        # Đồng bộ số lượng và tổng giá trị giữa các tiến trình trong training phân tán
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)  # Gộp các giá trị giữa các GPU
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        # Trả về trung vị
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        # Trung bình trong cửa sổ trượt
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        # Trung bình tổng thể kể từ đầu
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        # Giá trị mới nhất
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

# ======= LOGGER QUẢN LÝ NHIỀU SMOOTHED METRICS =======

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        # Tạo dictionary các metric: tên → SmoothedValue
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        # Cập nhật giá trị cho từng metric
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        # Cho phép gọi như thuộc tính (logger.loss, logger.lr,...)
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        # In ra tất cả các metric
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        # Gọi đồng bộ tất cả metrics
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        # Log mỗi khi chạy 1 vòng lặp huấn luyện (batch)
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',     # Thời gian còn lại
            '{meters}',       # Các metric
            'time: {time}',   # Thời gian 1 bước
            'data: {data}'    # Thời gian load dữ liệu
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')  # Ghi thêm bộ nhớ nếu có GPU
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj  # Yield lại batch cho vòng lặp chính
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self), time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self), time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)')

# ======= CHECKPOINT LOAD FOR EMA =======

def _load_checkpoint_for_ema(model_ema, checkpoint):
    # Giải pháp nạp checkpoint vào Model EMA khi không dùng trực tiếp file
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)

# ======= CHẶN PRINT KHI KHÔNG PHẢI MASTER PROCESS =======

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

# ======= KIỂM TRA CÁC THÔNG SỐ MÔI TRƯỜNG DISTRIBUTED =======

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    # Chỉ lưu mô hình ở tiến trình chính
    if is_main_process():
        torch.save(*args, **kwargs)

# ======= KHỞI TẠO CHẾ ĐỘ PHÂN TÁN (Distributed Mode) =======

def init_distributed_mode(args):
    # Kiểm tra biến môi trường để biết đang dùng gì: PyTorch launcher hay SLURM
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'

    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)

    torch.distributed.init_process_group(backend=args.dist_backend,
                                         init_method=args.dist_url,
                                         world_size=args.world_size,
                                         rank=args.rank)
    torch.distributed.barrier()  # Đồng bộ tiến trình
    setup_for_distributed(args.rank == 0)

# ======= ĐỌC FILE CẤU HÌNH MMCV VÀ GÁN LẠI CHO args =======

def update_from_config(args):
    cfg = mmcv.Config.fromfile(args.config)
    for _, cfg_item in cfg._cfg_dict.items():
        for k, v in cfg_item.items():
            setattr(args, k, v)
    return args
