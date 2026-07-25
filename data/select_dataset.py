def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type in ['dream']:
        from data.dataset_DREAM import SRData as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))
    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    print(f"Loaded dataset with {len(dataset)} samples from {dataset_opt['dataroot_H']}")
    return dataset
