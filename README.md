# Glove ❤ MLOps

Glove implementation and MLOps configuration supporting our Medium articles.
Full article can be found [here](https://xmoleslo.medium.com/link-to-full-article). TK - modify according to final link.

# Data

TK - Prepare a dataset.

# Usage

One would preferably build the Docker images first in order to have a fully working environment:

```bash
cd glove-mlops
docker build .
```

Once this step completed, and inside the runing container, each step can simply be executed using the main script.

For example, do the following to clean the text.

```bash
cd src
python3 main.py clean_text --data-file=/path/to/data_file.json --output-path=/path/to/output
```

Help on the other commands can be obtained by using the --help subcommand.

```bash
cd src
python3 main.py --help
python3 main.py <command> --help
```

# MLOps

If you have a Valohai account, you can upload the docker image you have just created to a registry of your choice and then adapt the `valohai.yaml` file consequently.

Inside the container you can then run the following commands to run the full pipeline from text cleaning, to evaluation:

```bash
vh login -t $YOUR_SECRET_VH_TOKEN
vh project link $YOUR_VALOHAI_PROJECT
vh project fetch
vh pipeline run "Full Pipeline"
```
