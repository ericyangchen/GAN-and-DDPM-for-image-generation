import os
from tqdm import tqdm
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor


class IclevrDataset(Dataset):

    def __init__(
        self, mode="train", json_root="data", image_root="data/images", num_cpus=8
    ):
        super().__init__()

        assert mode in ["train", "test", "new_test"], "IclevrDataset mode error"

        self.mode = mode
        self.image_root = image_root
        self.json_root = json_root

        # image transformation
        self.image_transformation = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        # load train/test/new_test.json
        self.labels = None
        with open(f"{self.json_root}/{self.mode}.json", "r") as json_file:
            json_data = json.load(json_file)

        if self.mode == "train":
            image_paths = list(json_data.keys())
            self.labels = list(json_data.values())
        elif self.mode == "test" or self.mode == "new_test":
            self.labels = list(json_data)

        # load objects.json
        with open(f"{self.json_root}/objects.json", "r") as json_file:
            self.objects_dict = json.load(json_file)

        # convert labels to one hot encoding
        one_hot_labels = []
        for label in self.labels:
            one_hot_label = torch.zeros(len(self.objects_dict), dtype=torch.long)
            for label_name in label:
                one_hot_label[self.objects_dict[label_name]] = 1
            one_hot_labels.append(one_hot_label)
        self.labels = torch.stack(one_hot_labels)

        # load images in train mode
        self.images = None
        if self.mode == "train":
            with ThreadPoolExecutor(max_workers=num_cpus) as executor:
                self.images = list(
                    tqdm(
                        executor.map(self._load_image, image_paths),
                        total=len(image_paths),
                        desc=f"Loading {self.mode} images",
                    )
                )
            self.images = torch.stack(self.images)

    def _load_image(self, image_path):
        image = Image.open(f"{self.image_root}/{image_path}").convert("RGB")
        image = self.image_transformation(image)
        return image

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.mode == "train":
            return self.images[index], self.labels[index]
        elif self.mode == "test" or self.mode == "new_test":
            return self.labels[index]


if __name__ == "__main__":
    print("Train dataset")
    train_dataset = IclevrDataset(
        image_root="../data/images", json_root="../data", mode="train"
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    for batch in train_loader:
        print(batch[0].shape, batch[1].shape)
        breakpoint()
        break

    print("Test dataset")
    test_dataset = IclevrDataset(
        image_root="../data/images", json_root="../data", mode="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    for batch in test_loader:
        print(batch.shape)
        breakpoint()
        break

    print("New Test dataset")
    new_test_dataset = IclevrDataset(
        image_root="../data/images", json_root="../data", mode="new_test"
    )
    new_test_loader = DataLoader(new_test_dataset, batch_size=32, shuffle=False)
    for batch in new_test_loader:
        print(batch.shape)
        breakpoint()
        break
