import torchvision.transforms as T

def get_transform(height, width, is_eval=True):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    if is_eval:
        return valid_transform
    
    else:
        train_transform = T.Compose([
            T.Resize((height, width)),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])

        return train_transform, valid_transform