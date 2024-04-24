import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence

import fire
import laspy
import numpy as np
import torch
import torchmetrics
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from MinkowskiEngine import MinkowskiAlgorithm, SparseTensorQuantizationMode, TensorField

from functional import compose_transforms_from_list
from model import MinkUNet14C


def las2pyg(las: laspy.LasData, path: Path) -> Data:
    gt_key = "classification" if "classification" in set(las.point_format.dimension_names) else "raw_classification"
    return Data(
        xyz=torch.from_numpy(las.xyz.copy()),
        intensity=torch.from_numpy(las.intensity.astype(np.int64)),
        classification=torch.from_numpy(las[gt_key]).long(),
        return_number=torch.from_numpy(np.asarray(las.return_number)).long(),
        number_of_returns=torch.from_numpy(np.asarray(las.number_of_returns)).long(),
        edge_of_flight_line=torch.from_numpy(np.asarray(las.edge_of_flight_line)),
        instance_id=(
            torch.from_numpy(np.asarray(las.instance).copy().astype(np.int64)).long()
            if hasattr(las, "instance")
            else torch.full((len(las.return_number),), fill_value=-1, dtype=torch.long)
        ),
        rgb=torch.stack(
            [
                torch.from_numpy(las.red.astype(np.int64)),
                torch.from_numpy(las.green.astype(np.int64)),
                torch.from_numpy(las.blue.astype(np.int64)),
            ],
            dim=-1,
        ).long()
        if hasattr(las, "red")
        else None,
        filename=path,
    )


def collate_custom_test(batch):
    # item = [coords, feats, labels, (unique_map, inverse_map)]
    batch = [item[:3] for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, config, fnames: Sequence[Path]):
        self.fnames = fnames
        self.transforms = compose_transforms_from_list(config.test_transforms)

    def __getitem__(self, index: int) -> Data:
        tile_path = self.fnames[index]
        las = laspy.read(tile_path)
        pyg_data = las2pyg(las, str(tile_path))
        if self.transforms:
            pyg_data = self.transforms(pyg_data)
        return pyg_data

    def __len__(self) -> int:
        return len(self.fnames)


def seed_everything(seed: int) -> None:
    """Fix a random seed in Numpy, PyTorch, and CUDA in order to improve reproducibility of DL pipelines"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_inference(
    dataset_path: str = None,
    model_weights_file: str = None,
    **kwargs,
):
    from_cli = OmegaConf.create(kwargs)
    base_conf = OmegaConf.load("./configs/test.yaml")
    conf = OmegaConf.merge(base_conf, from_cli)
    
    label_file = Path(dataset_path) / "labels.json"
    with open(label_file, encoding="utf-8") as f:
        all_tiles: List[Dict] = json.load(f)

    test_tiles = [dataset_path + "/" + "pointclouds" + "/" + x["tile_name"] for x in all_tiles if x["split"] == "test"]

    # the dataset
    dataset = TestDataset(conf, test_tiles)
    # the dataloader
    dataloader = PyGDataLoader(
        dataset, batch_size=conf.test_batch_size, collate_fn=collate_custom_test, pin_memory=True
    )

    # define metrics
    metrics = {}
    metrics["f1"] = torchmetrics.classification.MulticlassF1Score(num_classes=conf.num_classes, ignore_index=0)
    metrics["f1_pc"] = torchmetrics.classification.MulticlassF1Score(
        num_classes=conf.num_classes, average=None, ignore_index=0
    )
    metrics = torchmetrics.MetricCollection(metrics)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MinkUNet14C(conf.num_features, conf.num_classes)
    model.load_state_dict(torch.load(model_weights_file))

    model.eval()

    model.to(device)
    metrics.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Model"):
            # for batch in dataloader:
            coords, features, batch_idx = batch.pos / conf.voxel_size, batch.x, batch.batch
            coords = torch.cat([batch_idx.unsqueeze(1), coords], dim=1)

            in_field = TensorField(
                features=features.to(device),
                coordinates=coords.to(device),
                quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
            )
            # Convert to a sparse tensor
            sinput = in_field.sparse()

            # sparse model output
            soutput = model(sinput)
            # dense model output
            out_field = soutput.slice(in_field).F

            # get the true labels
            y_true = batch.classification.long().to(device)

            # mask out the ignored classes
            mask = torch.bitwise_and(y_true != 0, y_true != -100)
            out_field = out_field[mask]
            y_true = y_true[mask]
            pred = out_field.max(dim=1).indices

            # update the metrics
            metrics.update(pred, y_true)

    metric_values = metrics.compute()

    print(f"Per-class F1 score:")
    for i, class_name in conf.class_names.items():
        print(f"{class_name}: {metric_values['f1_pc'][i].item()}")
    print(f"Macro F1: {metric_values['f1'].item()}")


if __name__ == "__main__":
    fire.Fire(run_inference)
