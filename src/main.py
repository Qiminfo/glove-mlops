import fire

from train import train_glove, build_cooccurrence
from transform import clean_text
from package import package_model
from evaluate import evaluation, apply

available_commands = {
    "clean_text": clean_text,
    "build_cooccurrence": build_cooccurrence,
    "train_glove": train_glove,
    "package_model": package_model,
    "evaluate": evaluation,
    "apply": apply,
}

if __name__ == "__main__":
    fire.Fire(available_commands)
