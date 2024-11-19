from loguru import logger
from arguments import Arguments
from CIFAR10Dataset import *
from FashionMNISTDataset import *
from dataloaders import *


if __name__ == '__main__':
    args = Arguments(logger)

    # ---------------------------------
    # ------------ CIFAR10 ------------
    # ---------------------------------
    dataset = CIFAR10Dataset(args)
    TRAIN_DATA_LOADER_FILE_PATH = "/mnt/ssd1/zhanghanxiu/dataset/cifar10/train_data_loader.pickle"
    TEST_DATA_LOADER_FILE_PATH = "/mnt/ssd1/zhanghanxiu/dataset/cifar10/test_data_loader.pickle"

    # if not os.path.exists("data_loaders/cifar10"):
    #     pathlib.Path("data_loaders/cifar10").mkdir(parents=True, exist_ok=True)

    train_data_loader = generate_train_loader(args, dataset)
    test_data_loader = generate_test_loader(args, dataset)

    with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(train_data_loader, f)

    with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(test_data_loader, f)

    # ---------------------------------
    # --------- Fashion-MNIST ---------
    # ---------------------------------
    dataset = FashionMNISTDataset(args)
    TRAIN_DATA_LOADER_FILE_PATH = "/mnt/ssd1/zhanghanxiu/dataset/FashionMNIST/train_data_loader.pickle"
    TEST_DATA_LOADER_FILE_PATH = "/mnt/ssd1/zhanghanxiu/dataset/FashionMNIST/test_data_loader.pickle"

    # if not os.path.exists("data_loaders/fashion-mnist"):
    #     pathlib.Path("data_loaders/fashion-mnist").mkdir(parents=True, exist_ok=True)

    train_data_loader = generate_train_loader(args, dataset)
    test_data_loader = generate_test_loader(args, dataset)

    with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(train_data_loader, f)

    with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(test_data_loader, f)
