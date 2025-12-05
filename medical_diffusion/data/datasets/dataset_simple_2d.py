import torch.utils.data as data
import torch
from torch import nn
from pathlib import Path
from torchvision import transforms as T
import pandas as pd
import json
import os

from PIL import Image


class SimpleDataset2D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers=[],
        crawler_ext="tif",  # other options are ['jpg', 'jpeg', 'png', 'tiff'],
        transform=None,
        image_resize=None,
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        image_crop=None,
    ):
        super().__init__()
        self.path_root = Path(path_root)
        self.crawler_ext = crawler_ext
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext)

        if transform is None:
            self.transform = T.Compose(
                [
                    T.Resize(image_resize)
                    if image_resize is not None
                    else nn.Identity(),
                    T.RandomHorizontalFlip()
                    if augment_horizontal_flip
                    else nn.Identity(),
                    T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                    T.CenterCrop(image_crop)
                    if image_crop is not None
                    else nn.Identity(),
                    T.ToTensor(),
                    # T.Lambda(lambda x: torch.cat([x]*3) if x.shape[0]==1 else x),
                    # ToTensor16bit(),
                    # Normalize(), # [0, 1.0]
                    # T.ConvertImageDtype(torch.float),
                    T.Normalize(
                        mean=0.5, std=0.5
                    ),  # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root / rel_path_item
        # img = Image.open(path_item)
        img = self.load_item(path_item)
        return {"uid": rel_path_item.stem, "source": self.transform(img)}

    def load_item(self, path_item):
        return Image.open(path_item).convert("RGB")
        # return cv2.imread(str(path_item), cv2.IMREAD_UNCHANGED) # NOTE: Only CV2 supports 16bit RGB images

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [
            path.relative_to(path_root)
            for path in Path(path_root).rglob(f"*.{extension}")
        ]

    def get_weights(self):
        """Return list of class-weights for WeightedSampling"""
        return None


class AIROGSDataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = pd.read_csv(
            self.path_root.parent / "train_labels.csv", index_col="challenge_id"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        uid = self.labels.index[index]
        path_item = self.path_root / f"{uid}.jpg"
        img = self.load_item(path_item)
        str_2_int = {"NRG": 0, "RG": 1}  # RG = 3270, NRG = 98172
        target = str_2_int[self.labels.loc[uid, "class"]]
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {"source": self.transform(img), "target": target}

    def get_weights(self):
        n_samples = len(self)
        weight_per_class = 1 / self.labels["class"].value_counts(
            normalize=True
        )  # {'NRG': 1.03, 'RG': 31.02}
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.labels.iloc[index]["class"]
            weights[index] = weight_per_class[target]
        return weights

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []


class MSIvsMSS_Dataset(SimpleDataset2D):
    # https://doi.org/10.5281/zenodo.2530835
    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root / rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        str_2_int = {"MSIMUT": 0, "MSS": 1}
        target = str_2_int[path_item.parent.name]  #
        return {"uid": uid, "source": self.transform(img), "target": target}


class MSIvsMSS_2_Dataset(SimpleDataset2D):
    # https://doi.org/10.5281/zenodo.3832231
    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root / rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        str_2_int = {
            "MSIH": 0,
            "nonMSIH": 1,
        }  # patients with MSI-H = MSIH; patients with MSI-L and MSS = NonMSIH)
        target = str_2_int[path_item.parent.name]
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {"source": self.transform(img), "target": target}


class CheXpert_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode = self.path_root.name
        labels = pd.read_csv(self.path_root.parent / f"{mode}.csv", index_col="Path")
        self.labels = labels.loc[labels["Frontal/Lateral"] == "Frontal"].copy()
        self.labels.index = self.labels.index.str[20:]
        self.labels.loc[self.labels["Sex"] == "Unknown", "Sex"] = (
            "Female"  # Affects 1 case, must be "female" to match stats in publication
        )
        self.labels.fillna(2, inplace=True)  # TODO: Find better solution,
        str_2_int = {
            "Sex": {"Male": 0, "Female": 1},
            "Frontal/Lateral": {"Frontal": 0, "Lateral": 1},
            "AP/PA": {"AP": 0, "PA": 1},
        }
        self.labels.replace(str_2_int, inplace=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        rel_path_item = self.labels.index[index]
        path_item = self.path_root / rel_path_item
        img = self.load_item(path_item)
        uid = str(rel_path_item)
        target = torch.tensor(
            self.labels.loc[uid, "Cardiomegaly"] + 1, dtype=torch.long
        )  # Note Labels are -1=uncertain, 0=negative, 1=positive, NA=not reported -> Map to [0, 2], NA=3
        return {"uid": uid, "source": self.transform(img), "target": target}

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []


class CheXpert_2_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        labels = pd.read_csv(
            self.path_root / "labels/cheXPert_label.csv",
            index_col=["Path", "Image Index"],
        )  # Note: 1 and -1 (uncertain) cases count as positives (1), 0 and NA count as negatives (0)
        labels = labels.loc[labels["fold"] == "train"].copy()
        labels = labels.drop(labels="fold", axis=1)

        labels2 = pd.read_csv(self.path_root / "labels/train.csv", index_col="Path")
        labels2 = labels2.loc[labels2["Frontal/Lateral"] == "Frontal"].copy()
        labels2 = labels2[
            [
                "Cardiomegaly",
            ]
        ].copy()
        labels2[(labels2 < 0) | labels2.isna()] = (
            2  # 0 = Negative, 1 = Positive, 2 = Uncertain
        )
        labels = labels.join(
            labels2["Cardiomegaly"],
            on=[
                "Path",
            ],
            rsuffix="_true",
        )
        # labels = labels[labels['Cardiomegaly_true']!=2]

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path_index, image_index = self.labels.index[index]
        path_item = self.path_root / "data" / f"{image_index:06}.png"
        img = self.load_item(path_item)
        uid = image_index
        target = int(self.labels.loc[(path_index, image_index), "Cardiomegaly"])
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {"source": self.transform(img), "target": target}

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

    def get_weights(self):
        n_samples = len(self)
        weight_per_class = 1 / self.labels["Cardiomegaly"].value_counts(normalize=True)
        # weight_per_class = {2.0: 1.2, 1.0: 8.2, 0.0: 24.3}
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.labels.loc[self.labels.index[index], "Cardiomegaly"]
            weights[index] = weight_per_class[target]
        return weights


class BTXRD_Dataset(SimpleDataset2D):
    def __init__(self, json_dir, split_path=None, split_mode="train", *args, **kwargs):
        """
        Args:
            json_dir: Path to the folder containing .json annotations.
            split_path: Path to the 'data_split.json' file generated by your classifier.
            split_mode: 'train' or 'test'.
            *args, **kwargs: Arguments passed to SimpleDataset2D (path_root, transform, etc.)
        """
        super().__init__(*args, **kwargs)
        self.json_dir = json_dir

        # 1. Define the specific classes
        self.classes = [
            "osteochondroma",
            "osteosarcoma",
            "multiple osteochondromas",
            "simple bone cyst",
            "giant cell tumor",
            "synovial osteochondroma",
            "osteofibroma",
        ]
        # Create mapping: Class String -> Integer
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 2. Gather all valid samples (Image Path, Label Index)
        # We process ALL files first to establish the full dataset, then we subset based on the split.
        self.all_samples = []

        # We use sorted() here to ensure deterministic order, unlike standard os.listdir
        # This is crucial for matching indices if you regenerate splits.
        files = sorted(os.listdir(self.path_root))

        for fname in files:
            if not fname.lower().endswith(".jpeg"):
                continue

            # Construct JSON path
            json_name = fname.replace(".jpeg", ".json")
            json_path = os.path.join(self.json_dir, json_name)

            if not os.path.exists(json_path):
                print(json_path)
                print(f"ERROR: No json for {fname}")
                continue

            with open(json_path, "r") as f:
                data = json.load(f)

            # Extract label
            try:
                label = data["shapes"][0]["label"].lower()
            except (KeyError, IndexError):
                continue  # Skip malformed jsons

            if label not in self.classes:
                continue

            # Store relative path for SimpleDataset2D and the label index
            self.all_samples.append((fname, self.class_to_idx[label]))

        # 3. Apply Splitting (if a split file is provided)
        if split_path is not None and os.path.exists(split_path):
            with open(split_path, "r") as f:
                splits = json.load(f)

            if split_mode in splits:
                indices = splits[split_mode]
                # Filter the samples list to only keep the indices belonging to this split
                self.samples = [self.all_samples[i] for i in indices]
            else:
                raise ValueError(f"Split mode '{split_mode}' not found in {split_path}")
        else:
            # If no split file provided, use all data
            self.samples = self.all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 1. Get filename and label index
        fname, target_idx = self.samples[index]

        # 2. Construct full path
        path_item = self.path_root / fname

        # 3. Load and Transform Image
        # load_item is a helper from the parent SimpleDataset2D class
        img = self.load_item(path_item)

        # 4. Return Dictionary format required by MedFusion
        return {"uid": fname, "source": self.transform(img), "target": target_idx}

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """
        Overwrite to return empty list.
        We handle file crawling manually in __init__ because we need to check the JSONs simultaneously.
        """
        return []

    def get_weights(self):
        """
        Optional: Calculate class weights for WeightedRandomSampler if needed.
        """
        n_samples = len(self)
        # Count occurrences
        counts = {}
        for _, label_idx in self.samples:
            counts[label_idx] = counts.get(label_idx, 0) + 1

        # Calculate weights (inverse frequency)
        weight_per_class = {k: 1.0 / v for k, v in counts.items()}

        weights = [0] * n_samples
        for i, (_, label_idx) in enumerate(self.samples):
            weights[i] = weight_per_class[label_idx]

        return weights
