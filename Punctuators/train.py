from model.model import Punctuator
from data.data import TRAIN_FILE
from data.data import load
from model.config import Config
import os


def main():
    config = Config()
    # build model
    print("Building model...")
    """
    if resuming training then pass resume_training=True. Resumes from the last saved model.
    But make sure the ckpt folder has saved the models.
    """
    punctuator = Punctuator(config)
    print("Loading dataset...")
    train_set = load(TRAIN_FILE)
    ref = os.path.join('.', 'data', 'raw', 'ref.txt')
    asr = os.path.join('.', 'data', 'raw', 'asr.txt')
    print("Training mode...")
    texts = [ref, asr]
    punctuator.train(train_set, batch_size=32, epochs=30, texts=texts)


if __name__ == '__main__':
    main()
