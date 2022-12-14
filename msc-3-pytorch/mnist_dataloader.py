from random import shuffle
import torchvision
import torch
from transformers import ViTConfig, ViTModel, ViTFeatureExtractor
from tqdm import tqdm

from datasets import load_dataset
from torch.utils.data import IterableDataset
class InfiniteMaskedMNIST(IterableDataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
    
    def __iter__(self):
        return infinite_masked_mnist(self.dataset)

def dataset_to_pytorch(dataset):
    imgs = torch.stack([torchvision.transforms.functional.pil_to_tensor(img.convert('RGB')) for img in tqdm(dataset["image"])])
    imgs = imgs.float()
    return imgs

    
def infinite_masked_mnist(dataset):
    """
    Takes an iterable of PyTorch tensors which represent images. Masks, discretizes, and batches them as neccessary.
    """
    # fx = ViTFeatureExtractor()
    while True:
        # data epoch
        idxs = list(range(len(dataset)))
        shuffle(idxs)
        for i in idxs:
            img = dataset[i]["image"]
            img = img.convert('RGB')
            img = torchvision.transforms.functional.pil_to_tensor(img)
            img = img.float()
            shape = img.shape
            img = img.reshape(shape[0], shape[1], shape[2])
            # img = fx(img, return_tensors='pt')

            mask = torch.randint(low=0, high=2, size=(shape[1]*shape[2],))

            yield {
                'pixel_values': img,
                'bool_masked_pos': mask,
            }
