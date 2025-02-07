import sys
from typing import List, Dict
import collections
import collections.abc

import torch
import numpy as np


class BatchElement:
    def __init__(self, data: Dict, batch_transforms: Dict = None):
        """
        Constructs a batch element encapsulating the data
        :param data: dictionary containing the data
        :param batch_transforms: dictionary containing the transformations to apply to the batch of corresponding data entries
        """
        # Implements checks on the data if needed
        self.data = data
        self.batch_transforms = batch_transforms

        if self.batch_transforms is None:
            self.batch_transforms = {}


class Batch:
    def __init__(self, data: Dict, batch_transforms: Dict = None):
        """
        :param data: the batch data
        :param batch_transforms: the transformations to apply to the corresponding batch entries
        """
        self.data = data
        self.batch_transforms = batch_transforms
        if self.batch_transforms is None:
            self.batch_transforms = {}

        self.device = "cpu"
        self.transformations_applied = False

    def maybe_apply_transforms(self):
        """
        Applies all the transformations to the data entries
        Transformations should be applied lazily to avoid that they are called outside the main thread.
        This may cause a big usage of shared memory and transfer overheads if big tensors are instantiated
        :return:
        """
        if self.device == "cpu":
            raise Exception("Batch transformations applied with CPU device. This is probably not the correct behavior if training is happening on the GPU.")

        if not self.transformations_applied:
            # Substitutes all keys for which a transformation exists with the corresponding transformation result
            for current_key in self.data.keys():
                if current_key in self.batch_transforms:
                    self.data[current_key] = self.batch_transforms[current_key](self.data[current_key], self.device)
            self.transformations_applied = True

    def transfer_element_to_device(self, key, element, device):
        """
        Transfers the specified element to the specified device
        """
        element_type = type(element)
        if torch.is_tensor(element):
            return element.to(device)
        elif isinstance(element, list) or isinstance(element, tuple):
            return [self.transfer_element_to_device(key, current_element, device) for current_element in element]
        elif isinstance(element, dict):
            transferred_elements = {}

            # Transfers to the device all elements with the same key
            for current_key in element.keys():
                transferred_elements[current_key] = self.transfer_element_to_device(key + "/" + str(current_key),
                                                                                    element[current_key], device)
        else:
            transferred_elements = element  # Any other type that does not need processing

        return transferred_elements

    def to(self, device):
        """
        Transfers tensors to the specified device
        :return:
        """
        # Records the device
        self.device = device

        keys = set(self.data.keys())
        # Transfers to the device all elements with the same key
        for current_key in keys:
            self.data[current_key] = self.transfer_element_to_device(current_key, self.data[current_key], device)

        # Lazily applies transformations
        self.maybe_apply_transforms()

    def pin_element(self, dictionary, key, element):
        """
        Pins the specified element to memory
        """
        element_type = type(element)
        if torch.is_tensor(element):
            dictionary[key] = element.pin_memory()
        elif isinstance(element, list) or isinstance(element, tuple):
            pass  # Lists nor supported for pinning
        elif isinstance(element, dict):

            # Transfers to the device all elements with the same key
            for current_key in element.keys():
                self.pin_element(element, str(current_key), element[current_key])

    def pin_memory(self):

        for current_key in self.data.keys():
            self.pin_element(self.data, current_key, self.data[current_key])

        return self


def collate_element(key: str, all_elements: List):
    """
    Collates the elements corresponding to a key
    """
    # Add here special rules for special keys that cannot be handled with the default type collate

    element_type = type(all_elements[0])
    if all_elements[0] is None:
        collated_element = all_elements  # If the first element is None, then nothing we can only return a list with all values
    elif isinstance(all_elements[0], str):
        collated_element = all_elements  # If the first element is a str, then we return a list of strings
    elif torch.is_tensor(all_elements[0]):
        collated_element = torch.stack(all_elements)
    elif element_type == np.ndarray:
        collated_element = np.stack(all_elements)
    elif isinstance(all_elements[0], list) or isinstance(all_elements[0], tuple):
        collated_element = all_elements  # Do not do anything by default for lists. A recursive call may wrongly stack an inner dimension of the list
    elif isinstance(all_elements[0], dict):
        collated_element = {}

        # Gets all keys
        keys = set()
        for current_element in all_elements:
            keys.update(current_element.keys())
        # Collates all elements with the same keys
        for current_key in keys:
            collated_element[current_key] = collate_element(key + "/" + str(current_key),
                                                            [current_element[current_key] for current_element in
                                                             all_elements])
    else:
        return all_elements  # Any other type that does not need processing

    return collated_element


def flexible_batch_elements_collate_fn(batch: List[BatchElement]) -> Batch:
    """
    Creates a batch starting from single batch elements

    :param batch: List of batch elements or list of dictionaries, each matching a key to a batch element
    :return: Batch representing the passed batch elements
    """
    all_keys = set()
    for current_batch_element in batch:
        all_keys.update(current_batch_element.data.keys())

    collated_data = {}
    for current_key in all_keys:
        try:
            collated_data[current_key] = collate_element(current_key, [current_batch_element.data[current_key] for current_batch_element in batch])
        # In case of failure logs the data point for which the failure occurred
        except Exception as e:
            print(f"An exception occurred while collating elements with key {current_key}", file=sys.stderr, flush=True)
            print(f"Data: {[current_batch_element.data[current_key] for current_batch_element in batch]}", file=sys.stderr, flush=True)
            for batch_element_idx, current_batch_element in enumerate(batch):
                print(f"Data[{batch_element_idx}]: {current_batch_element.data[current_key]}", file=sys.stderr, flush=True)
            raise e

    # We assume transforms in all batch elements are the same and get the ones from the first one
    first_batch_element = batch[0]
    batch_transforms = first_batch_element.batch_transforms

    return Batch(collated_data, batch_transforms)
