from scripts.train_model import train
from utils.data_eval import show_data

def main():
    data_dir = 'data/cifar-10-batches-py'
    train(data_dir)

if __name__ == "__main__":
    main()