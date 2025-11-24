import unittest
import numpy as np
import numpy.typing as npt


def boolean_to_distance(array: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
    """
    Given a numpy array of boolean values, we compute the distance (left to right)
    of the closest True value.
    
    For example, consider the sequence
        [False, False, True, False, False, False, True, False, False, False, False, True]
    In int format, this is
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    The output array is
        [-1, -1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0]

    Explained solution steps
    ------------------------
    The main idea is to create an auxiliary sequence of values whose cumulative sum equals
    the desired output.
    Afterwards, we replace initial 0's with -1's.

    For the example above, the auxiliary sequence we wish to obtain (before cumulative sum) is:
        [1, 1, -2, 1, 1, 1, -3, 1, 1, 1, 1, -4]
    Evidently, the cumulative sum of this sequence gives the desired output (except for the initial -1's).
    The question is how to construct this auxiliary sequence efficiently.

    To construct the sequence efficiently, we follow these steps:
    1. Apply the transformation (0 -> 1) and (1 -> 0) to the input array. This yields
        [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    (note that this yields the first condition of the auxiliary sequence).
    2. Next, we need to find values [-2, -3, -4] at the positions of the True values.
    We note that these values correspond to
        a. the (negative) distance between consecutive True values and
        b. to the (negative) position of the first True value.
    The negative distances and first position can be computed as follows:
        a.  Take the cumulative sum of the transformed array.
            [1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9]
        b. Slice this cumulative sum at the positions of the True values.
            [2, 5, 9]
        c. Take the difference between consecutive values in this sliced array,
            and append the initial value at the start.
            [2, 3, 4]
        d. Negate this array
            [-2, -3, -4]
    3. Replace the values found in step 2. at the positions of the True values in the transformed array.
    4. Take the cumulative sum of the resulting array
    5. Replace initial 0's with -1's.
    """
    if len(array) == 0:
        return np.array([], dtype=np.int64)

    array = array.astype(np.int64)
    
    # position of first "1", if there are no "1"'s,
    #   return array of -1 of same length
    posfirst = array.argmax()

    if posfirst == 0 and array[0] == 0:
        # no True values, return array of -1
        res = np.repeat(-1, len(array))
        return res
    
    # Step 1: Map (0 -> 1) and (1 -> 0)
    aux = np.abs(array - 1)

    # Step 2: Find distance between consecutive 1's and append initial one
    v = aux.cumsum() 
    v = np.r_[
        v[posfirst], # position of first '1'
        np.diff(v[array == 1]) # distance between successive '1's
    ]

    # Step 3: replace values in the auxiliary array
    aux[array == 1] = -v

    # Step 4: cumsum of values
    res = aux.cumsum()

    # Step 5: Replace initial 1's with zeros
    res[:posfirst] = -1

    return res


def convolution(
    array: npt.NDArray[np.float64],
    w: int,
    h1: float,
    h2: float
) -> npt.NDArray[np.float64]:
    """
    Compute an array B with len(array) = len(B) such that

        B[i] = sum(array[j] * exp(-(i-j) / h1) * cos((i-j) / h2);  i - w < j <= i, j >= 0)

    The computational complexity is O(N * w) where N = len(array)
    and w is the window size.

    Parameters
    ----------
    array: input array
    w: window size (N / 10 <= w <= N)
    h1: h1 >= N / 2
    h2: h2 > 0

    Explained solution steps
    ------------------------
    The idea is to realise that for each position i in the array (A), we only need a window of size w.
    Furthermore, the coefficients of the window are fixed and range from f(0) to f(w-1)
    Here, f(x) = exp(-x / h1) * cos(x / h2).
    Denote these coefficients as C1, C1, ..., C(w) and consider the case w=3 for simplicity.
    Then, C = [C1, C2, C3] 
    We note that C multiplies the following terms of A:
        [C1 * A1,        ,        ] -> B0
        [C1 * A2, C0 * A1,        ] -> B1
        [C1 * A3, C0 * A2, C2 * A1] -> B2
        [C1 * A4, C0 * A3, C2 * A2] -> B3
        [C1 * A5, C0 * A4, C2 * A3] -> B4

    Rewriting these terms, we see that we can construct an auxiliary matrix A_repeat of the form:
        A1
        A2 A1
        A3 A2 A1
        A4 A3 A2
        A5 A4 A3
    where the upper triangle is zeroed out.
    We observe that B can be written as the dot product between each row of A_repeat with C.

    Thus, if we construct this auxiliary matrix A_repeat, we can compute B 
    in a single step for all rows using matrix multiplication. 
    (Here, we use einsums to avoid creating an intermediate matrix).
    """
    N = len(array)
    
    if N == 0:
        return np.array([], dtype=np.float64)

    if not (N / 10 <= w <= N):
        raise ValueError("w must be in [N/10, N]")
    elif h1 < N / 2:
        raise ValueError("h1 must be >= N / 2")
    elif h2 <= 0:
        raise ValueError("h2 must be > 0")


    # If jax were allowed:
    # jax.vmap(jnp.roll, in_axes=((None, 0))(A, jnp.arange(w))
    # would be faster and avoid this python loop.
    A_repeat = np.stack([np.roll(array, i) for i in range(w)], axis=1)
    A_repeat[np.triu_indices_from(A_repeat, k=1)] = 0 # zero out upper triangle

    vrange = np.arange(w)
    coefs = np.exp(-vrange / h1) * np.cos(vrange / h2)
    res = np.einsum("nw,w->n", A_repeat, coefs)
    return res


def logsumexp_cumsum(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Adapted from https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    """
    c = x.max()
    return c + np.log(np.cumsum(np.exp(x - c)))


def sum_by_indices(
    A: npt.NDArray[np.float64],
    S: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
    """
    Given a numpy length N floating point array A (1<=A[i]<=2) and a numpy length N integer array S (0<=S[i]<=i),
    return length N floating point numpy array X with elements
        X[i] = sum of A[k] * exp(-50*k/N)) for k in range S[i]<=k<=i.
    
    Explained solution steps
    ------------------------
    To avoid python loops, we consider the following construction:
        1. The cumulative sum of the longest sequence (from i=0 to N) can be precomputed and every other sum (i <= N) is a subset of it.
           Call this cumulative sum V.
        2. At i=1,...,N, the end index is always i but the start index is S[i], which is always <= i.
        3. Thus, we can do V[i] - V[S[i]] to get the sum from S[i] to i. The first term is always the same,
           so we append a value of 0 with chosen index 0 to match dimensions.
        4. Finally, for numerical stability, we apply the log-sum-exp trick when computing V.
           This makes it slightly slower but avoids underflow issues for large N.
    """
    N = len(S)
    K_values = np.arange(N)

    if np.any(A < 1) or np.any(A > 2):
        raise ValueError("All elements of A must be in [1, 2]")
    elif np.any(S < 0):
        raise ValueError("All elements of S must be >= 0")
    elif np.any(S > K_values):
        raise ValueError("All elements of S must satisfy S[i] <= i")
    elif len(A) != N:
        raise ValueError("Arrays A and S must have the same length")
    elif N == 0:
        return np.array([], dtype=np.float64)

    V = logsumexp_cumsum(np.log(A) - 50 * K_values / N)
    V = np.exp(V)

    S = np.concat([[0], S])
    V = np.concat([[0], V])

    # V[i] - V[S[i]] for i = 0, ..., (N-1)
    X = (V - V[S])[1:]
    
    return X


################
#### Tests ####
################

def test_boolean_to_distance(random=True):
    if random:
        np.random.seed(314)
        N = 20
        B = np.random.choice(a=[False, True], size=N)
        D_test = np.zeros_like(B, dtype=np.int64)
        start_count = False
        counter = 0
        for i, n in enumerate(B * 1):
            if (n == 0) and start_count:
                counter += 1
            elif (n == 0) and (not start_count):
                counter = -1
            elif n == 1:
                counter = 0
                start_count = True
            D_test[i] = counter
    else:
        B = np.array([False, False, True, False, False, False, True, False, False, False, False, True])
        D_test = np.array([-1, -1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0])

    D = boolean_to_distance(B)

    return np.array_equal(D, D_test)


def test_convolution(random=True):
    N = 20
    w = 5 # w ∈ [N / 10, N]
    h1 = N / 2 # h1 ∈ [N / 2, inf)
    h2 = 1.0 # h2 > 0

    if random:
        np.random.seed(314)
        A = np.random.rand(N)
    else:
        A = np.arange(N).astype(np.float64) + + 1

    B = convolution(A, w, h1, h2)

    B_test = np.zeros_like(A)
    for i in range(N):
        j = np.arange(max(i - w + 1, 0), i + 1)
        Bi = (A[j] * np.exp(-(i - j) / h1) * np.cos((i - j) / h2)).sum()
        B_test[i] = Bi

    
    return np.allclose(B, B_test)


def test_sum_by_indices(random=True):
    np.random.seed(314)
    N = 20
    if random:
        A = np.random.uniform(1, 2, N)
    else:
        A = np.linspace(1, 2, N)

    K_values = np.arange(N)
    S = np.random.uniform(size=N) * K_values
    S = S.round().astype(np.int32)

    X = sum_by_indices(A, S)

    # Explicit sum accumulation
    X_test = np.zeros_like(A)
    for i in range(N):
        for k in range(S[i], i+1):
            X_test[i] += A[k] * np.exp(-50 * k/N) 

    return np.allclose(X, X_test)


class TestCases(unittest.TestCase):
    def test_boolean_to_distance(self):
        self.assertTrue(test_boolean_to_distance(random=False))

    def test_convolution(self):
        self.assertTrue(test_convolution(random=False))

    def test_sum_by_indices(self):
        self.assertTrue(test_sum_by_indices(random=False))


if __name__ == "__main__":
    unittest.main()
