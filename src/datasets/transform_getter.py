
import torchvision.transforms as transforms



def get_transform(args, train=False):
    if args.dataset == "CIFAR10":
        transform = _get_cifar10(args, train)
    elif args.dataset == "CIFAR100":
        transform = _get_cifar100(args, train)
    elif args.dataset == "FashionMNIST":
        transform = _get_fashionmnist(args, train)

    return transform


def _get_cifar10(args, train):
    normalize = (
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    )
    resize = args.resize        # Resize Input Images
    crop = args.crop            # Crop Input Images
    randrot = args.randrot      # Randomly Rotate Input Images
    randhf = args.randhf        # Randomly Flip Input Horizontally
    randvf = args.randvf        # Randomly Flip Input Vertically
    randjit = args.randjit      # Randomly Change the Brightness and Contrast

    if train:
        transform = transforms.Compose(
            [
                transforms.RandomCrop((crop, crop), 4) if (crop is not None and train)\
                    else transforms.CenterCrop(crop) if (crop is not None and not train)\
                        else transforms.Lambda(lambda x: x),
                transforms.Resize((resize, resize)) if resize is not None\
                    else transforms.Lambda(lambda x: x),
                transforms.RandomRotation(randrot) if (randrot is not None and train)\
                    else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip(randhf) if (randhf is not None and train)\
                    else transforms.Lambda(lambda x: x),
                transforms.RandomVerticalFlip(randvf) if (randvf is not None and train)\
                    else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(brightness=randjit, contrast=randjit) if (randjit is not None and train)\
                    else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]
            + ([normalize] if normalize is not None else [])
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((resize, resize)) if resize is not None\
                    else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]
            + ([normalize] if normalize is not None else [])
        )
    return transform

def _get_cifar100(args, train):
    normalize = (
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    )
    resize = args.resize        # Resize Input Images
    crop = args.crop            # Crop Input Images
    randrot = args.randrot      # Randomly Rotate Input Images
    randhf = args.randhf        # Randomly Flip Input Horizontally
    randvf = args.randvf        # Randomly Flip Input Vertically
    randjit = args.randjit      # Randomly Change the Brightness and Contrast

    if train:
        transform = transforms.Compose(
            [
                transforms.RandomCrop((crop, crop), 4) if (crop is not None and train)\
                    else transforms.CenterCrop(crop) if (crop is not None and not train)\
                        else transforms.Lambda(lambda x: x),
                transforms.Resize((resize, resize)) if resize is not None\
                    else transforms.Lambda(lambda x: x),
                transforms.RandomRotation(randrot) if (randrot is not None and train)\
                    else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip(randhf) if (randhf is not None and train)\
                    else transforms.Lambda(lambda x: x),
                transforms.RandomVerticalFlip(randvf) if (randvf is not None and train)\
                    else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(brightness=randjit, contrast=randjit) if (randjit is not None and train)\
                    else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]
            + ([normalize] if normalize is not None else [])
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((resize, resize)) if resize is not None\
                    else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]
            + ([normalize] if normalize is not None else [])
        )
    return transform


def _get_fashionmnist(args, train):
    normalize = (
        transforms.Normalize((0.5,), (0.5,))
    )
    resize = args.resize        # Resize Input Images
    crop = args.crop            # Crop Input Images
    randrot = args.randrot      # Randomly Rotate Input Images
    randhf = args.randhf        # Randomly Flip Input Horizontally
    randvf = args.randvf        # Randomly Flip Input Vertically
    randjit = args.randjit      # Randomly Change the Brightness and Contrast

    if train:
        transform = transforms.Compose(
            [
                transforms.RandomCrop((crop, crop), 4) if (crop is not None and train)\
                    else transforms.CenterCrop(crop) if (crop is not None and not train)\
                        else transforms.Lambda(lambda x: x),
                transforms.Resize((resize, resize)) if resize is not None\
                    else transforms.Lambda(lambda x: x),
                transforms.RandomRotation(randrot) if (randrot is not None and train)\
                    else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip(randhf) if (randhf is not None and train)\
                    else transforms.Lambda(lambda x: x),
                transforms.RandomVerticalFlip(randvf) if (randvf is not None and train)\
                    else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(brightness=randjit, contrast=randjit) if (randjit is not None and train)\
                    else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]
            + ([normalize] if normalize is not None else [])
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((resize, resize)) if resize is not None\
                    else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]
            + ([normalize] if normalize is not None else [])
        )
    return transform
