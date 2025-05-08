# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# ========== IMPORT ==========
import os
import json

# Datasets và transforms từ torchvision
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

# Constants và transform tool từ timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

# Custom multi-crop loader (nếu có)
from mcloader import ClassificationDataset

# ========== CLASS CHUYỂN ĐỔI DỮ LIỆU iNat ==========

class INatDataset(ImageFolder):
    """
    Dataset cho tập iNaturalist (2018 hoặc 2019).
    Lấy thông tin từ các file JSON (đặc trưng cho iNat).
    Cho phép phân loại theo nhiều cấp độ sinh học như: genus, family, order, name,...
    """
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year

        # Đọc file JSON định nghĩa tập huấn luyện hoặc validation
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        # Đọc thông tin chi tiết về các lớp
        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        # Đọc lại annotation để mapping class ID theo category mong muốn
        path_json_for_targeter = os.path.join(root, f"train{year}.json")
        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        # Tạo ánh xạ tên class → chỉ số class
        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = [data_catg[int(elem['category_id'])][category]]
            if king[0] not in targeter:
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)  # Tổng số lớp

        # Xây dựng danh sách (path ảnh, label) từ dữ liệu JSON
        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])  # id lớp thật
            path_current = os.path.join(root, cut[0], cut[2], cut[3])  # đường dẫn ảnh

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]  # lấy chỉ số theo category
            self.samples.append((path_current, target_current_true))

    # __getitem__ và __len__ kế thừa từ ImageFolder

# ========== HÀM TẠO DATASET THEO LOẠI ==========

def build_dataset(is_train, args):
    # Tạo transform trước
    transform = build_transform(is_train, args)

    # Tùy chọn tập dữ liệu theo args.data_set
    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        if not args.use_mcloader:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        else:
            # MultiCropLoader nếu bật
            dataset = ClassificationDataset(
                'train' if is_train else 'val',
                pipeline=transform
            )
        nb_classes = 1000
    elif args.data_set == 'INAT':
        # Dataset iNaturalist 2018
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        # Dataset iNaturalist 2019
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes  # Trả về tập dữ liệu và số lớp

# ========== HÀM TẠO TRANSFORM ==========

def build_transform(is_train, args):
    resize_im = args.input_size > 32  # Có resize không (CIFAR không cần)

    if is_train:
        # Gọi create_transform từ timm cho training
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # Nếu không resize, dùng RandomCrop (CIFAR)
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    # Nếu là eval → build transform thủ công
    t = []
    if resize_im:
        if args.input_size >= 384:
            # Resize cứng luôn (warping)
            t.append(
                transforms.Resize((args.input_size, args.input_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
            )
            print(f"Warping {args.input_size} size input images...")
        else:
            # Resize giữ tỉ lệ (crop_pct 224/256)
            args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                transforms.Resize(size, interpolation=3),  # interpolation=3 = bicubic
            )
            t.append(transforms.CenterCrop(args.input_size))  # Cắt chính giữa

    # Cuối cùng chuyển tensor và normalize
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    return transforms.Compose(t)
