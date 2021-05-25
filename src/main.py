import fire

from train import train_glove, build_cooccurrence
from transform import clean_text
from evaluate import evaluation, apply
from utils import load_data

available_commands = {
    "load_data": load_data,
    "clean_text": clean_text,
    "build_cooccurrence": build_cooccurrence,
    "train_glove": train_glove,
    "evaluate": evaluation,
    "apply": apply,
}

if __name__ == "__main__":
    fire.Fire(available_commands)
