from model.model import Punctuator
from model.config import Config
import os


def main():
    config = Config()
    punctuator = Punctuator(config)
    punctuator.restore_last_session()
    ref_txt = os.path.join('.', 'data', 'raw', 'ref.txt')
    ref_out = os.path.join('.', 'data', 'raw', 'ref.out.txt')
    asr_txt = os.path.join('.', 'data', 'raw', 'asr.txt')
    asr_out = os.path.join('.', 'data', 'raw', 'asr.out.txt')

    punctuator.compute_score(ref_txt, ref_out)
    punctuator.compute_score(asr_txt, asr_out)


if __name__ == '__main__':
    main()
