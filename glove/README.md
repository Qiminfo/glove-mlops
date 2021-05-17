# Glove

Cython general implementation of the Glove multi-threaded training.

The code in this folder was cloned from [Jonathan Raiman's glove repository](https://github.com/JonathanRaiman/glove) and adapted to support `scipy.sparse` matrices.

The adaptations contained in this repository will soon make the object of a pull request to the original repository.

*The rest of this file comes from the original README file.*

GloVe is an unsupervised learning algorithm for generating vector representations for words.
Training is done using a co-occcurence matrix from a corpus. The resulting representations contain structure useful for many other tasks.

The paper describing the model is [here](http://nlp.stanford.edu/projects/glove/glove.pdf).

The original implementation for this Machine Learning model can be [found here](http://nlp.stanford.edu/projects/glove/).

@author Jonathan Raiman

## Example

To use this package you need a sparse co-occurence matrix.
This matrix is represented by nested dictionaries that use ints as keys
with a 0-index.

For instance below we have a corpus of 3 indices. Below 0 co-occurs with 2, 3.5 times:

```python
import glove
from scipy.sparse import csr_matrix

rows = [0, 0, 1, 2, 2, 2]
cols = [0, 2, 2, 0, 1, 2]
data = [1.0, 3.5, 0.5, 3.5, 0.5, 1.2]
cooccur = csr_matrix((data, (rows, cols)), shape=(3, 3))

model = glove.Glove(cooccur, d=50, alpha=0.75, x_max=100.0)

for epoch in range(25):
    err = model.train(batch_size=200, workers=9, batch_size=50)
    print("epoch %d, error %.3f" % (epoch, err), flush=True)
```

The trained embeddings are now present under `model.W`.

## Usage

The model is controlled by setting several hyperpameters.

### Glove.__init__()

* `cooccurence` dict<int, dict<int, float>> : the co-occurence matrix
* `alpha` float : (default 0.75) hyperparameter for controlling the exponent for normalized co-occurence counts.
* `x_max` float : (default 100.0) hyperparameter for controlling smoothing for common items in co-occurence matrix.
* `d` int : (default 50) how many embedding dimensions for learnt vectors
* `seed` int : (default 1234) the random seed

### Glove.train

* `step_size` float : the learning rate for the model
* `workers` int : number of worker threads used for training
* `batch_size` int : how many examples should each thread receive (controls the size of the job queue)