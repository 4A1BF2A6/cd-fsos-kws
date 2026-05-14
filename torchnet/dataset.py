from torch.utils.data import Dataset


class ListDataset(Dataset):
    """Minimal torchnet.dataset.ListDataset replacement."""

    def __init__(self, elem_list):
        self.list = elem_list

    def __getitem__(self, index):
        return self.list[index]

    def __len__(self):
        return len(self.list)


class TransformDataset(Dataset):
    """Apply a transform callable to each item from another dataset."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        sample = self.dataset[index]
        return self.transform(sample)

    def __len__(self):
        return len(self.dataset)
