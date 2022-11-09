import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
import torch

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def get_data(params):
    if params['dataset'] == 'BMS-WebView-1' or params['dataset'] == 'BMS-WebView-2':
        return get_BMS(params)
    else:
        print('No dataset')
        return


def get_BMS(params):
    filename = params['filename']
    bs = params['bs']

    df = pd.read_csv(filename)

    train, test = train_test_split(df, test_size=.3)
    
    test_original = test

    train = torch.tensor(train.values, dtype=torch.float)
    test = torch.tensor(test.values, dtype=torch.float)

    train_loader = DataLoader(train, batch_size=bs)
    test_loader = DataLoader(test, batch_size=bs)

    params['n_items'] = train.shape[1]

    return train_loader, test_loader, test_original, params


def get_mnist():
    # image transformations
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(
        root='../input/data',
        train=True,
        download=True,
        transform=transform
    )
    test_set = datasets.MNIST(
        root='../input/data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


def prior_probs(chosen_itemsets, alpha, k, sigma, plot_dist=False):
    """
    Set the global prior probability distribution of itemsets.
    The formula is:
    priors = normalize(powerlaw(a, k) + gaussian_noise(mean=0, stdv=sigma)).
    """

    assert len(chosen_itemsets) > 0
    assert alpha > 0
    assert k > 0
    assert sigma > 0

    n_itemsets = len(chosen_itemsets)

    priors = alpha * np.power(np.arange(1, n_itemsets + 1, dtype=np.float64), -k)
    priors += np.absolute(np.random.normal(loc=0, scale=sigma, size=n_itemsets))
    priors /= np.sum(priors)

    if plot_dist:
        y_pos = np.arange(n_itemsets)

        plt.figure()
        plt.bar(y_pos, priors)
        plt.title('Itemset priors')
        plt.xticks(y_pos, chosen_itemsets)
        plt.xlabel('Itemsets')
        plt.ylabel('probabilities')
        plt.draw()

    return priors


def parabolic(min_val, max_val, x, e):
    y = (max_val - min_val) * x**e + min_val
    # plt.plot(x, y)
    # plt.show()

    return y


def noisy_itemset(itemset, limit, n_items, min_items=1, max_items=20, e=8, first=True):
    n_noise = int(np.round(parabolic(min_items, max_items, np.random.rand(), e)))

    if first:
        noise_ids = np.random.choice(list(range(limit, n_items)),
                                     size=n_noise, replace=False)
    else:
        noise_ids = np.random.choice(list(range(n_items - limit)),
                                     size=n_noise, replace=False)

    noise = np.array(itemset, dtype=int)
    noise[noise_ids] = 1
    return noise


def get_loader(dataset, params):
    train_loader = DataLoader(TensorDataset(torch.tensor(dataset['x_train'], dtype=torch.float),
                                            torch.tensor(dataset['y_train'], dtype=torch.float)),
                              batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(dataset['x_val'], dtype=torch.float),
                                          torch.tensor(dataset['y_val'], dtype=torch.float)),
                            batch_size=params['batch_size'],  shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(dataset['x_test'], dtype=torch.float),
                                           torch.tensor(dataset['y_test'], dtype=torch.float)),
                             batch_size=params['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader


def generate10patterns(params):
    n_items = params['n_items']
    n_itemsets = params['n_itemsets']
    n_transactions = params['n_transactions']

    columns = ['item_' + str(x) for x in range(1, n_items + 1)]
    compressed_columns = list(columns)
    compressed_columns.append('count')

    edo_x = np.zeros((n_transactions, 4), dtype=int)
    edo_y = np.zeros(n_transactions, dtype=int)

    edo_x[:210] = np.array([1, 0, 1, 1])
    edo_y[:210] = 11
    edo_x[210:350] = np.array([0, 1, 0, 1])
    edo_y[210:350] = 5
    edo_x[350:460] = np.array([1, 1, 1, 0])
    edo_y[350:460] = 14
    edo_x[460:560] = np.array([1, 0, 0, 0])
    edo_y[460:560] = 8
    edo_x[560:650] = np.array([1, 1, 0, 0])
    edo_y[560:650] = 12
    edo_x[650:730] = np.array([0, 0, 0, 1])
    edo_y[650:730] = 1
    edo_x[730:800] = np.array([0, 1, 1, 0])
    edo_y[730:800] = 6
    edo_x[800:870] = np.array([1, 1, 0, 1])
    edo_y[800:870] = 13
    edo_x[870:940] = np.array([0, 1, 0, 0])
    edo_y[870:940] = 4
    edo_x[940:] = np.array([1, 0, 1, 0])
    edo_y[940:] = 10

    test_ids = edo_y

    chosen_itemsets = [11, 5, 14, 8, 12, 1, 6, 13, 4, 10]
    print(f'#itemsets: {len(chosen_itemsets)}')

    c_test_df = pd.DataFrame(edo_x, columns=columns)
    c_test_df = c_test_df.groupby(columns).size().reset_index(name='count')

    print(c_test_df)

    dataset = {}
    dataset['x_train'] = edo_x
    dataset['y_train'] = edo_y

    dataset['x_val'] = edo_x
    dataset['y_val'] = edo_y

    dataset['x_test'] = edo_x
    dataset['y_test'] = edo_y

    dataset['test_ids'] = test_ids
    dataset['chosen_itemsets'] = chosen_itemsets
    dataset['columns'] = columns
    dataset['c_test_df'] = c_test_df

    return dataset


def from_ids_to_datasets(ids, n_items):
    dataset = []
    compressed_dataset = []
    counts = dict(zip(ids, [0] * len(ids)))

    first_id = None
    first_itemset = np.zeros(n_items)
    second_itemset = np.zeros(n_items)
    limit = 100
    first_itemset[:limit] = 1
    second_itemset[-limit:] = 1

    for id in ids:
        # tmp = [int(x) for x in bin(id)[2:]]

        if first_id is None:
            first_id = id

        if id == first_id:
            row = noisy_itemset(first_itemset, limit, n_items=n_items)
        else:
            row = noisy_itemset(second_itemset, limit, n_items=n_items, first=False)

        # row = np.pad(tmp, (n_items - len(tmp), 0),
        #              'constant', constant_values=(0, 0))

        counts[id] += 1
        dataset.append(row)

    '''
    for id, count in counts.items():
      tmp = [int(x) for x in bin(id)[2:]]
      row = np.pad(tmp, (n_items - len(tmp), 1),
                   'constant', constant_values=(0, count))

      compressed_dataset.append(row)
    '''

    return np.array(dataset)  # , np.array(compressed_dataset)


def generate2patterns(params):
    n_items = params['n_items']
    n_itemsets = params['n_itemsets']

    alpha = params['alpha']
    k = params['k']
    sigma = params['sigma']
    n_rows_train = params['n_transactions_train']
    n_rows_val = params['n_transactions_val']
    n_rows_test = params['n_transactions_test']

    chosen_itemsets = np.array([3, 12])

    priors = prior_probs(chosen_itemsets, alpha, k, sigma)

    ids_train = np.random.choice(chosen_itemsets, n_rows_train, True, priors)
    ids_val = np.random.choice(chosen_itemsets, n_rows_val, True, priors)
    ids_test = np.random.choice(chosen_itemsets, n_rows_test, True, priors)

    '''
    train, c_train = from_ids_to_datasets(train_ids)
    val, c_val = from_ids_to_datasets(val_ids)
    test, c_test = from_ids_to_datasets(test_ids)
    '''

    train = from_ids_to_datasets(ids_train, n_items)
    val = from_ids_to_datasets(ids_val, n_items)
    test = from_ids_to_datasets(ids_test, n_items)

    dataset = {}
    dataset['x_train'] = train
    dataset['x_test'] = test
    dataset['x_val'] = val
    dataset['y_train'] = ids_train
    dataset['y_val'] = ids_val
    dataset['y_test'] = ids_test
    dataset['chosen_itemsets'] = chosen_itemsets

    return dataset


def noisy_sub_itemset(sub_itemset, min_items=1, max_items=20, e=8):
    n_noise = int(np.round(parabolic(min_items, max_items,
                                     np.random.rand(), e)))

    noise_ids = np.random.choice(list(range(len(sub_itemset))),
                                 size=n_noise, replace=False)

    noise = np.array(sub_itemset, dtype=int)

    if (sub_itemset == 1).any():
        noise[noise_ids] = 0

    if (sub_itemset == 0).any():
        noise[noise_ids] = 1

    return noise


def from_ids_to_datasetsMoreItems(ids, itemset, n_blocks, n_original_items):

    new_itemset = []

    old_itemset = []

    ids_list = []

    for _id in ids:
        for i in range(itemset.shape[0]):
            index_itemset = int(''.join([str(x) for x in itemset[i]]), 2)

            if _id == index_itemset:
                tmp_itemset = []
                for j in range(itemset.shape[1]):
                    tmp_itemset.extend([itemset[i][j]]*n_blocks)
                new_itemset.append(tmp_itemset)

                old_itemset.append(itemset[i])

                ids_list.append(_id)

    new_itemset = np.array(new_itemset)

    for i in range(new_itemset.shape[0]):
        count = 0

        idx_start = 0
        idx_end = n_blocks

        while count < n_original_items:
            new_itemset[i][idx_start:idx_end] = noisy_sub_itemset(new_itemset[i][idx_start:idx_end])

            idx_start = idx_end
            idx_end += n_blocks

            count += 1

    return new_itemset, np.array(old_itemset), np.array(ids_list)


def generate_10Patterns_MoreItems(params):
    n_items = params['n_items']
    n_original_items = params['original_items']
    n_blocks = n_items//n_original_items

    columns = ['item_' + str(x) for x in range(1, n_original_items + 1)]
    params['n_blocks'] = n_blocks

    n_itemsets = 10

    itemset = np.array([[1, 0, 1, 1], [0, 1, 0, 1],[1, 1, 1, 0], [1, 0, 0, 0],
                        [1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 1],[0, 1, 0, 0],
                        [1, 0, 1, 0]])


    alpha = params['alpha']
    k = params['k']
    sigma = params['sigma']
    n_rows_train = params['n_transactions_train']
    n_rows_val = params['n_transactions_val']
    n_rows_test = params['n_transactions_test']

    chosen_itemsets = [11, 5, 14, 8, 12, 1, 6, 13, 4, 10]

    priors = prior_probs(chosen_itemsets, alpha, k, sigma)

    train_ids = np.random.choice(chosen_itemsets, n_rows_train, True, priors)
    val_ids = np.random.choice(chosen_itemsets, n_rows_val, True, priors)
    test_ids = np.random.choice(chosen_itemsets, n_rows_test, True, priors)

    train, original_train, ids_train = from_ids_to_datasetsMoreItems(train_ids, itemset, n_blocks, n_original_items)
    val, original_val,  ids_val = from_ids_to_datasetsMoreItems(val_ids, itemset, n_blocks, n_original_items)
    test, original_test, ids_test = from_ids_to_datasetsMoreItems(test_ids, itemset, n_blocks, n_original_items)

    c_test_df = pd.DataFrame(original_train, columns=columns)
    c_test_df = c_test_df.groupby(columns).size().reset_index(name='count')

    print(c_test_df)

    dataset = {}
    dataset['x_train'] = train
    dataset['original_train'] = original_train
    dataset['x_test'] = test
    dataset['x_val'] = val
    dataset['y_train'] = ids_train
    dataset['y_val'] = ids_val
    dataset['y_test'] = ids_test
    dataset['chosen_itemsets'] = chosen_itemsets
    dataset['test_ids'] = ids_train
    dataset['columns'] = columns

    return dataset
