import gc
import torch
import random
import logging
import torchvision
import transformers
import numpy as np

from collections import defaultdict

from .torchvisionparser import fetch_torchvision_dataset
from .transform_getter import get_transform

logger = logging.getLogger(__name__)


class SubsetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, suffix):
        self.subset = subset
        self.suffix = suffix

    def __getitem__(self, index):
        inputs, targets = self.subset[index]
        return inputs, targets

    def __len__(self):
        return len(self.subset)
    
    def __repr__(self):
        return f'{repr(self.subset.dataset.dataset)} {self.suffix}'
    

def load_dataset(args):
    # method to construct per-client dataset
    def _construct_dataset(raw_train, idx, sample_indices):
        subset = torch.utils.data.Subset(raw_train, sample_indices)
        if args.num_classes is None: # regression
            training_set, test_set = torch.utils.data.random_split(subset, [len(subset) - int(len(subset) * args.test_size), int(len(subset) * args.test_size)])
        else: # classification
            training_set, test_set = stratified_split(subset, args.test_size)
            
        traininig_set = SubsetWrapper(training_set, f'< {str(idx).zfill(8)} > (train)')
        if len(subset) * args.test_size > 0:
            test_set = SubsetWrapper(test_set, f'< {str(idx).zfill(8)} > (test)')
        else:
            test_set = None
        return (traininig_set, test_set)
    
    # raw train/test datasets
    raw_train, raw_test = None, None
    split_map, client_datasets = None, None
    transforms = [None, None]

    # fetch dataset
    logger.info(f'[LOAD] Fetch dataset!')

    if args.dataset in torchvision.datasets.__dict__.keys(): # for downloadable datasets in `torchvision.datasets`...
        transforms = [get_transform(args, train=True), get_transform(args, train=False)]
        raw_train, raw_test, args = fetch_torchvision_dataset(args=args, dataset_name=args.dataset, root=args.data_path, transforms=transforms) 

    # check if global holdout set is required or not
    if args.eval_type == 'local':
        if args.test_size == -1: 
            assert raw_test is not None
            _raw_test = raw_test
        raw_test = None
    else:
        if raw_test is None:
            err = f'[LOAD] Dataset `{args.dataset.upper()}` does not support pre-defined validation/test set, which can be used for `global` evluation... please check! (current `eval_type`=`{args.eval_type}`)'
            logger.exception(err)
            raise AssertionError(err)
        
    # get split indices if None
    if split_map is None:
        print(f'[SIMULATE] Simulate dataset split (split scenario: `{args.split_type.upper()}`)!')
        split_map = simulate_split(args, raw_train)
        print(f'[SIMULATE] ...done simulating dataset split (split scenario: `{args.split_type.upper()}`)!')
    
    # construct client datasets if None
    if client_datasets is None:
        print(f'[SIMULATE] Create client datasets!')
        client_datasets = []
        for idx, sample_indices in enumerate(split_map.values()):
            client_datasets.append(_construct_dataset(raw_train, idx, sample_indices))
        print(f'[SIMULATE] ...successfully created client datasets!')
        
        ## when if assigning pre-defined test split as a local holdout set (just divided by the total number of clients)
        if (args.eval_type == 'local') and (args.test_size == -1):  
            holdout_sets = torch.utils.data.random_split(_raw_test, [int(len(_raw_test) / args.K)  for _ in range(args.K)])
            holdout_sets = [SubsetWrapper(holdout_set, f'< {str(idx).zfill(8)} > (test)') for idx, holdout_set in enumerate(holdout_sets)]
            augmented_datasets = []
            for idx, client_dataset in enumerate(client_datasets): 
                augmented_datasets.append((client_dataset[0], holdout_sets[idx]))
            client_datasets = augmented_datasets
    gc.collect()
    return raw_test, client_datasets  


# stratified split for classification tasks
def stratified_split(raw_dataset, test_size):
    indices_per_label = defaultdict(list)
    for index, label in enumerate(np.array(raw_dataset.dataset.targets)[raw_dataset.indices]):
        indices_per_label[label.item()].append(index)
    
    train_indices, test_indices = [], []
    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * test_size)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        test_indices.extend(random_indices_sample)
        train_indices.extend(set(indices) - set(random_indices_sample))
    return torch.utils.data.Subset(raw_dataset, train_indices), torch.utils.data.Subset(raw_dataset, test_indices)

def simulate_split(args, dataset):
    # IID split
    if args.split_type == 'iid':
        # shuffle sample indices
        shuffled_indices = np.random.permutation(len(dataset))

        # get adjusted indices
        split_indices = np.array_split(shuffled_indices, args.K)

        # construct a hashmap
        split_map = {k: split_indices[k] for k in range(args.K)}
        return split_map
    
    # non-IID split by sample unbalancedness (#local samples)
    if args.split_type == 'unbalanced':
        # shuffle sample indices
        shuffled_indices = np.random.permutation(len(dataset))

        # split indices by number of clients
        split_indices = np.array_split(shuffled_indices, args.K)

        # randomly remove some proportion (1% ~ 5%) of data
        keep_ratio = np.random.uniform(low=0.95, high=0.99, size=len(split_indices))

        # get adjusted indices
        split_indices = [indices[:int(len(indices) * ratio)] for indices, ratio in zip(split_indices, keep_ratio)]
        
        # construct a hashmap
        split_map = {k: split_indices[k] for k in range(args.K)}
        return split_map

    # Non-IID split proposed in (McMahan et al., 2016); each client has samples from at least two different classes
    elif args.split_type == 'patho': 
        try:
            assert args.mincls >= 2
        except AssertionError as e:
            raise e
        
        # get indices by class labels
        _, unique_inverse, unique_counts = np.unique(dataset.targets, return_inverse=True, return_counts=True)
        class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_counts[:-1]))
            
        # divide shards
        num_shards_per_class = args.K * args.mincls // args.num_classes
        if num_shards_per_class < 1:
            err = f'[SIMULATE] Increase the number of minimum class (`args.mincls` > {args.mincls}) or the number of participating clients (`args.K` > {args.K})!'
            raise Exception(err)
        
        # split class indices again into groups, each having the designated number of shards
        split_indices = [np.array_split(np.random.permutation(indices), num_shards_per_class) for indices in class_indices]
        
        # make hashmap to track remaining shards to be assigned per client
        class_shards_counts = dict(zip([i for i in range(args.num_classes)], [len(split_idx) for split_idx in split_indices]))

        # assign divided shards to clients
        assigned_shards = []
        for _ in range(args.K):
            # update selection proability according to the count of reamining shards
            # i.e., do NOT sample from class having no remaining shards
            selection_prob = np.where(np.array(list(class_shards_counts.values())) > 0, 1., 0.)
            selection_prob /= sum(selection_prob)
            
            # select classes to be considered
            try:
                selected_classes = np.random.choice(args.num_classes, args.mincls, replace=False, p=selection_prob)
            except: # if shard size is not fit enough, some clients may inevitably have samples from classes less than the number of `mincls`
                selected_classes = np.random.choice(args.num_classes, args.mincls, replace=True, p=selection_prob)
            
            # assign shards in randomly selected classes to current client
            for it, class_idx in enumerate(selected_classes):
                selected_shard_indices = np.random.choice(len(split_indices[class_idx]), 1)[0]
                selected_shards = split_indices[class_idx].pop(selected_shard_indices)
                if it == 0:
                    assigned_shards.append([selected_shards])
                else:
                    assigned_shards[-1].append(selected_shards)
                class_shards_counts[class_idx] -= 1
            else:
                assigned_shards[-1] = np.concatenate(assigned_shards[-1])

        # construct a hashmap
        split_map = {k: assigned_shards[k] for k in range(args.K)}
        return split_map
    
    # Non-IID split proposed in (Hsu et al., 2019); simulation of non-IID split scenario using Dirichlet distribution
    elif args.split_type == 'diri':
        MIN_SAMPLES = args.min_samples

        # get indices by class labels
        total_counts = len(dataset.targets)
        _, unique_inverse, unique_counts = np.unique(dataset.targets, return_inverse=True, return_counts=True)
        class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_counts[:-1]))

        # calculate ideal samples counts per client
        ideal_counts = len(dataset.targets) // args.K
        if ideal_counts < 1:
            err = f'[SIMULATE] Decrease the number of participating clients (`args.K` < {args.K})!'
            logger.exception(err)
            raise Exception(err)

        # split dataset
        ## define temporary container
        assigned_indices = []

        ## NOTE: it is possible that not all samples be consumed, as it is intended for satisfying each clients having at least `MIN_SAMPLES` samples per class
        for k in range(args.K):
            ### for current client of which index is `k`
            curr_indices = []
            satisfied_counts = 0

            ### ...until the number of samples close to ideal counts is filled
            while satisfied_counts < ideal_counts:
                ### define Dirichlet distribution of which prior distribution is an uniform distribution
                diri_prior = np.random.uniform(size=args.num_classes)
                
                ### sample a parameter corresponded to that of categorical distribution
                cat_param = np.random.dirichlet(alpha=args.cncntrtn * diri_prior)

                ### try to sample by amount of `ideal_counts``
                sampled = np.random.choice(args.num_classes, ideal_counts, p=cat_param)

                ### count per-class samples
                unique, counts = np.unique(sampled, return_counts=True)
                if len(unique) < args.mincls: 
                    continue
                
                ### filter out sampled classes not having as much as `MIN_SAMPLES`
                required_counts = counts * (counts > MIN_SAMPLES)

                ### assign from population indices split by classes 
                for idx, required_class in enumerate(unique):
                    if required_counts[idx] == 0: continue
                    sampled_indices = class_indices[required_class][:required_counts[idx]]
                    curr_indices.append(sampled_indices)
                    class_indices[required_class] = class_indices[required_class][:required_counts[idx]]
                satisfied_counts += sum(required_counts)
            
            ### when enough samples are collected, go to next clients!
            assigned_indices.append(np.concatenate(curr_indices))

        # construct a hashmap
        split_map = {k: assigned_indices[k] for k in range(args.K)}
        return split_map
    
    
