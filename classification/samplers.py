# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.distributed as dist
import math

# ===== RASampler: Sampler dùng cho phân tán kết hợp repeated augmentation =====
class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """
     """
    Sampler này giới hạn việc load dữ liệu trong môi trường distributed training,
    và đảm bảo mỗi augmented version của 1 ảnh sẽ được đưa đến **một tiến trình khác nhau**.

    Dựa nhiều vào torch.utils.data.DistributedSampler.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        # Xác định số lượng tiến trình (replicas) và rank của tiến trình hiện tại
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # Mỗi ảnh sẽ được lặp lại 3 lần (Repeated Augmentation)
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        # Số lượng mẫu thực sự được dùng cho mỗi GPU (làm tròn bội số 256)
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle # Cho phép xáo trộn dữ liệu theo epoch

    def __iter__(self):
        # deterministically shuffle based on epoch
        # Tạo seed dựa trên epoch để shuffle ổn định
        g = torch.Generator()
        g.manual_seed(self.epoch)
        # Shuffle hoặc không shuffle theo cờ `shuffle`
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        # Nhân mỗi ảnh 3 lần (RA): [a, b] → [a, a, a, b, b, b]
        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        # Thêm padding nếu chưa đủ để chia đều cho num_replicas
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # Cắt ra số lượng mẫu thực tế sẽ dùng trong training (có thể nhỏ hơn)
        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        # Hàm để thiết lập epoch mỗi lần training → thay đổi seed shuffle
        self.epoch = epoch
