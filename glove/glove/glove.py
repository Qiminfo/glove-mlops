import threading
from multiprocessing import Queue

import numpy as np
import pyximport
from scipy.sparse import csr_matrix

pyximport.install(setup_args={"include_dirs": np.get_include()})

from .glove_inner import train_glove


class Glove(object):
    def __init__(self, cooccurence, alpha=0.75, x_max=100.0, d=50, seed=1234):
        """
        Glove model for obtaining dense embeddings from a
        co-occurence (sparse) matrix.
        """
        self.alpha = alpha
        self.x_max = x_max
        self.d = d
        self.cooccurence = self._as_csr(cooccurence)
        self.seed = seed
        np.random.seed(seed)
        self.W = np.random.uniform(-0.5 / d, 0.5 / d, (cooccurence.shape[0], d)).astype(np.float64)
        self.ContextW = np.random.uniform(-0.5 / d, 0.5 / d, (cooccurence.shape[0], d)).astype(np.float64)
        self.b = np.random.uniform(-0.5 / d, 0.5 / d, (cooccurence.shape[0], 1)).astype(np.float64)
        self.ContextB = np.random.uniform(-0.5 / d, 0.5 / d, (cooccurence.shape[0], 1)).astype(np.float64)
        self.gradsqW = np.ones_like(self.W, dtype=np.float64)
        self.gradsqContextW = np.ones_like(self.ContextW, dtype=np.float64)
        self.gradsqb = np.ones_like(self.b, dtype=np.float64)
        self.gradsqContextB = np.ones_like(self.ContextB, dtype=np.float64)

    def _as_csr(self, cooccurence):
        if isinstance(cooccurence, csr_matrix):
            return cooccurence
        else:
            raise NotImplementedError("Glove class needs a csr_matrix")

    def train(self, step_size=0.05, workers=9, batch_size=50, verbose=False):
        jobs = Queue(maxsize=2 * workers)
        lock = (
            threading.Lock()
        )  # for shared state (=number of words trained so far, log reports...)
        total_error = [0.0]
        total_done = [0]

        total_els = self.cooccurence.count_nonzero()

        # Worker function:
        def worker_train():
            error = np.zeros(1, dtype=np.float64)
            while True:
                job = jobs.get()
                if job is None:  # resources finished, exit
                    break
                train_glove(self, job, step_size, error)
                with lock:
                    total_error[0] += error[0]
                    total_done[0] += len(job[0])
                    if verbose:
                        if total_done[0] % 1000 == 0:
                            print(
                                "Completed %.3f%%\r"
                                % (100.0 * total_done[0] / total_els)
                            )
                error[0] = 0.0

        # Create workers
        workers_threads = [
            threading.Thread(target=worker_train) for _ in range(workers)
        ]
        for thread in workers_threads:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        # Batch co-occurrence pieces
        # A list with the length of each batch
        batch_lengths = [batch_size] * (total_els // batch_size) + [total_els % batch_size]

        # Construct the slices
        r_indices = np.cumsum(batch_lengths)
        l_indices = np.concatenate([[0], r_indices[:-1]])
        slices = [slice(l, r) for l, r in zip(l_indices, r_indices)]

        # Retrieve rows and columns to iterate upon
        rows, cols = self.cooccurence.nonzero()

        # Build batches
        for sl in slices:
            jobs.put(
                (rows[sl],
                 cols[sl],
                 self.cooccurence[rows[sl], cols[sl]].getA1()
                 )
            )

        for _ in range(workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers_threads:
            thread.join()

        return total_error[0] / total_els
