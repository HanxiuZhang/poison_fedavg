import numpy
import random
import csv

def generate_experiment_ids(start_idx, num_exp):
    """
    Generate the filenames for all experiment IDs.

    :param start_idx: start index for experiments
    :type start_idx: int
    :param num_exp: number of experiments to run
    :type num_exp: int
    """
    log_files = []
    results_files = []
    models_folders = []
    worker_selections_files = []

    for i in range(num_exp):
        idx = str(start_idx + i)

        log_files.append("logs/" + idx + ".log")
        results_files.append(idx + "_results.csv")
        models_folders.append(idx + "_models")
        worker_selections_files.append(idx + "_workers_selected.csv")

    return log_files, results_files, models_folders, worker_selections_files


def distribute_batches_equally(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers

        distributed_dataset[worker_idx].append((data, target))

    return distributed_dataset

def convert_distributed_data_into_numpy(distributed_dataset):
    """
    Converts a distributed dataset (returned by a data distribution method) from Tensors into numpy arrays.

    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    """
    converted_distributed_dataset = []

    for worker_idx in range(len(distributed_dataset)):
        worker_training_data = distributed_dataset[worker_idx]

        X_ = numpy.array([tensor.numpy() for batch in worker_training_data for tensor in batch[0]])
        Y_ = numpy.array([tensor.numpy() for batch in worker_training_data for tensor in batch[1]])

        converted_distributed_dataset.append((X_, Y_))

    return converted_distributed_dataset

def identify_random_elements(max, num_random_elements):
    """
    Picks a specified number of random elements from 0 - max.

    :param max: Max number to pick from
    :type max: int
    :param num_random_elements: Number of random elements to select
    :type num_random_elements: int
    :return: list
    """
    if num_random_elements > max:
        return []

    ids = []
    x = 0
    while x < num_random_elements:
        rand_int = random.randint(0, max - 1)

        if rand_int not in ids:
            ids.append(rand_int)
            x += 1

    return ids

def log_client_data_statistics(logger, label_class_set, distributed_dataset):
    """
    Logs all client data statistics.

    :param logger: logger
    :type logger: loguru.logger
    :param label_class_set: set of class labels
    :type label_class_set: list
    :param distributed_dataset: distributed dataset
    :type distributed_dataset: list(tuple)
    """
    for client_idx in range(len(distributed_dataset)):
        client_class_nums = {class_val : 0 for class_val in label_class_set}
        for target in distributed_dataset[client_idx][1]:
            client_class_nums[target] += 1

        logger.info("Client #{} has data distribution: {}".format(client_idx, str(list(client_class_nums.values()))))

def convert_results_to_csv(results):
    """
    :param results: list(return data by test_classification() in client.py)
    """
    cleaned_epoch_test_set_results = []

    for row in results:
        components = [row[0], row[1]]

        for class_precision in row[2]:
            components.append(class_precision)
        for class_recall in row[3]:
            components.append(class_recall)

        cleaned_epoch_test_set_results.append(components)

    return cleaned_epoch_test_set_results

def save_results(results, filename):
    """
    :param results: experiment results
    :type results: list()
    :param filename: File name to write results to
    :type filename: String
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for experiment in results:
            writer.writerow(experiment)