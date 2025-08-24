"""
This is a didactic implementation that mirrors the *structure* of the HQC (Hamming
Quasi-Cyclic) PKE/KEM as specified in the NIST PQC process, but with **tiny toy
parameters** and a simple, classic error-correcting code (Hamming(7,4)) so that
students can read, run, and understand every line.

- It is **not** secure. Do **not** use for real security.
- It follows the HQC *flow* closely (KeyGen/Encrypt/Decrypt and the decoding
  step on `v - u · y`) while remaining short and readable.
- The small code here uses `n = 7` and a Hamming(7,4) block code to keep all
  vectors tiny and decoding transparent. Real HQC uses large parameters and
  efficient code families; see the official specification for details.

References to the real thing (read these next):
- HQC specification (4th-round version):
  https://pqc-hqc.org/doc/hqc-specification_2024-02-23.pdf
- NIST PQC project (HQC selected for standardization, Mar 11, 2025):
  https://csrc.nist.gov/projects/post-quantum-cryptography
- Hamming(7,4) background: https://en.wikipedia.org/wiki/Hamming(7,4)

File layout
- `Hamming74`  : Minimal encoder/decoder with explicit parity-check & syndrome table.
- `cyclic_convolution` : Binary cyclic (mod x^n - 1) polynomial multiplication over GF(2).
- `HQC`        : A small, HQC-shaped scheme using Hamming(7,4) as C.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np



def sample_sparse_binary_vector(n: int, weight: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a length-n binary vector with Hamming weight = `weight`.

    Args:
        n: vector length.
        weight: number of ones to place (0 <= weight <= n).
        rng: optional NumPy Generator for reproducibility.

    Returns:
        np.ndarray of shape (n,), dtype=int, values in {0,1}.
    """
    if weight < 0 or weight > n:
        raise ValueError("weight must satisfy 0 <= weight <= n")
    rng = rng or np.random.default_rng()
    vec = np.zeros(n, dtype=int)
    if weight == 0:
        return vec
    positions = rng.choice(n, size=weight, replace=False)
    vec[positions] = 1
    return vec


def cyclic_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Binary cyclic convolution a * b over GF(2) modulo x^n - 1.

    Interprets `a` and `b` as polynomials over GF(2), multiplies them, then
    reduces modulo x^n - 1 (i.e., wraps coefficients >= n), and reduces mod 2.

    Args:
        a: binary vector of length n (dtype=int, entries 0/1).
        b: binary vector of length n (dtype=int, entries 0/1).

    Returns:
        np.ndarray: binary vector of length n representing (a · b) mod (x^n - 1) over GF(2).
    """
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("a and b must be 1-D arrays with the same shape")
    n = a.shape[0]
    # Linear convolution, then wrap and reduce modulo 2
    lin = np.convolve(a.astype(int), b.astype(int))  # length = 2n-1
    lin %= 2
    out = np.zeros(n, dtype=int)
    # First n terms
    out[:n] ^= lin[:n]
    # Wrap-around terms
    for i in range(n, lin.shape[0]):
        out[i - n] ^= lin[i]
    out %= 2
    return out


# Hamming(7,4) code (toy C)

class Hamming74:
    """Minimal Hamming(7,4) encoder/decoder with explicit matrices.

    - Code length  n = 7
    - Message size k = 4
    - Corrects up to 1 bit error per codeword via syndrome decoding.

    We use the common systematic form where a codeword is [p1, p2, d1, p3, d2, d3, d4].
    """

    # Parity-check matrix H (3x7) in a standard convention (columns are bit positions 1..7)
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],  # checks positions with LSB of index = 1
        [0, 1, 1, 0, 0, 1, 1],  # checks positions with middle bit of index = 1
        [0, 0, 0, 1, 1, 1, 1],  # checks positions with MSB of index = 1
    ], dtype=int)

    # Syndrome table mapping 3-bit syndrome -> error position (1..7), 0 means "no error"
    SYNDROME_TO_POS = {
        (0, 0, 0): 0,
        (1, 0, 0): 1,
        (0, 1, 0): 2,
        (1, 1, 0): 3,
        (0, 0, 1): 4,
        (1, 0, 1): 5,
        (0, 1, 1): 6,
        (1, 1, 1): 7,
    }

    def encode(d: np.ndarray) -> np.ndarray:
        """Encode a 4-bit message into a 7-bit Hamming(7,4) codeword.

        Uses explicit parity equations for the systematic layout
        [p1, p2, d1, p3, d2, d3, d4] with **even parity**:
            p1 = d1 ⊕ d2 ⊕ d4  (covers positions 1,3,5,7)
            p2 = d1 ⊕ d3 ⊕ d4  (covers positions 2,3,6,7)
            p3 = d2 ⊕ d3 ⊕ d4  (covers positions 4,5,6,7)

        Args:
            d: binary array of shape (4,) with entries in {0,1}.

        Returns:
            np.ndarray: binary array of shape (7,) representing the codeword.
        """
        d = np.asarray(d, dtype=int)
        if d.shape != (4,):
            raise ValueError("Hamming74.encode expects shape (4,)")
        d1, d2, d3, d4 = [int(x) & 1 for x in d]
        p1 = (d1 ^ d2 ^ d4) & 1
        p2 = (d1 ^ d3 ^ d4) & 1
        p3 = (d2 ^ d3 ^ d4) & 1
        codeword = np.array([p1, p2, d1, p3, d2, d3, d4], dtype=int)
        return codeword

   
    def decode(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode a 7-bit received vector using syndrome decoding.

        Args:
            r: binary array of shape (7,).

        Returns:
            (m_hat, c_hat):
                m_hat: recovered 4-bit message (np.ndarray of shape (4,)).
                c_hat: corrected 7-bit codeword (np.ndarray of shape (7,)).
        """
        r = np.asarray(r, dtype=int)
        if r.shape != (7,):
            raise ValueError("Hamming74.decode expects shape (7,)")
        # Syndrome s = H r^T (over GF(2))
        s = (Hamming74.H @ r) % 2  # shape (3,)
        syndrome_tuple = tuple(int(x) for x in s)
        pos = Hamming74.SYNDROME_TO_POS.get(syndrome_tuple, 0)
        c_hat = r.copy()
        if pos != 0:
            # Flip the bit at the indicated position (1-indexed)
            c_hat[pos - 1] ^= 1
        # In our layout, message bits are at positions [2,4,5,6] -> (0-based)
        m_hat = c_hat[[2, 4, 5, 6]]
        return m_hat.astype(int), c_hat.astype(int)


# HQC-shaped PKE (toy params)


@dataclass
class HQCParams:
    """Tiny parameter set for the educational HQC-shaped PKE.

    NOTE: We split the randomness weights (wr1, wr2) and error weights (we1, we2)
    so that we can **guarantee** the Hamming(7,4) decoder only sees ≤ 1 error.
    The default "demo" choice keeps r2 = 0 and all explicit errors = 0.
    This preserves the HQC decryption equation while avoiding decoder overload
    at such a tiny block length. Real HQC uses large BCH codes that correct many errors.
    """
    n: int = 7   # ring dimension (kept equal to 7 to match Hamming(7,4))
    w: int = 1   # Hamming weight for secret polynomials x and y
    wr1: int = 1 # Hamming weight for r1
    wr2: int = 0 # Hamming weight for r2 (0 to keep t within 1-bit radius)
    we1: int = 0 # Hamming weight for e1
    we2: int = 0 # Hamming weight for e2

class HQC:
    """Educational, structurally faithful HQC-like scheme using Hamming(7,4).

    Public key:  pk = (h, s) with s = x + h · y (all ops over GF(2) in R = GF(2)[x]/(x^n - 1)).
    Secret key:  sk = (x, y) with small Hamming weight `w`.

    Encrypt(pk, m):
        - Sample small r1, r2 of weight `wr`, and error vectors e1, e2 of weight `we`.
        - Compute u = r1 + h · r2 + e1
        - Compute v = s · r2 + e2 + C.encode(m)

    Decrypt(sk, (u, v)):
        - Compute t = v - u · y = v + u · y   (same as XOR over GF(2))
        - Decode m_hat = C.decode(t)

    With small parameters, correctness holds if the overall noise that reaches the
    decoder fits within the error-correcting radius of C (here: 1-bit for Hamming(7,4)).
    """

    def __init__(self, params: HQCParams | None = None, rng: np.random.Generator | None = None) -> None:
        self.params = params or HQCParams()
        self.rng = rng or np.random.default_rng()
        self.pk: Tuple[np.ndarray, np.ndarray] | None = None  # (h, s)
        self.sk: Tuple[np.ndarray, np.ndarray] | None = None  # (x, y)


    # Key generation
   
    def keygen(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Generate (pk, sk).

        Returns:
            (pk, sk) where pk=(h, s) and sk=(x, y), each np.ndarray of shape (n,).
        """
        n, w = self.params.n, self.params.w
        x = sample_sparse_binary_vector(n, w, self.rng)
        y = sample_sparse_binary_vector(n, w, self.rng)
        h = self.rng.integers(0, 2, size=n, dtype=int)  # public ring element
        # s = x + h · y
        hy = cyclic_convolution(h, y)
        s = (x ^ hy) % 2
        self.pk = (h, s)
        self.sk = (x, y)
        return self.pk, self.sk

  
    # Encryption
  
    def encrypt(self, m4: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encrypt a 4-bit message using pk and Hamming(7,4) as C.

        Args:
            m4: np.ndarray shape (4,) with bits (the "data" part for Hamming(7,4)).

        Returns:
            (u, v): ciphertext polynomials in R (each np.ndarray of shape (n,)).
        """
        if self.pk is None:
            raise ValueError("Public key not initialized. Call keygen() first.")
        n = self.params.n
        h, s = self.pk
        # Sample per-branch sparsities to keep decoder's error budget ≤ 1 for Hamming(7,4)
        r1 = sample_sparse_binary_vector(n, self.params.wr1, self.rng)
        r2 = sample_sparse_binary_vector(n, self.params.wr2, self.rng)
        e1 = sample_sparse_binary_vector(n, self.params.we1, self.rng)
        e2 = sample_sparse_binary_vector(n, self.params.we2, self.rng)
        # C.encode(m)
        c = Hamming74.encode(m4)  # length 7 = n
        # u = r1 + h · r2 + e1
        hr2 = cyclic_convolution(h, r2)
        u = (r1 ^ hr2 ^ e1) % 2
        # v = s · r2 + e2 + c
        sr2 = cyclic_convolution(s, r2)
        v = (sr2 ^ e2 ^ c) % 2
        return u, v

   
    # Decryption
    
    def decrypt(self, ct: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decrypt a ciphertext using sk.

        Args:
            ct: (u, v) where each is np.ndarray of shape (n,).

        Returns:
            (m_hat, t, c_hat):
                m_hat: recovered 4-bit message (np.ndarray of shape (4,)).
                t: the 7-bit vector fed to the Hamming decoder (= v + u · y).
                c_hat: the corrected codeword output by the Hamming decoder.
        """
        if self.sk is None:
            raise ValueError("Secret key not initialized. Call keygen() first.")
        x, y = self.sk
        u, v = ct
        uy = cyclic_convolution(u, y)
        t = (v ^ uy) % 2  # v - u·y over GF(2)
        m_hat, c_hat = Hamming74.decode(t)
        return m_hat, t, c_hat



# Demo

rng = np.random.default_rng(12345)  # fixed seed for reproducibility in the demo
params = HQCParams(n=7, w=1, wr1=1, wr2=0, we1=0, we2=0)  # keep error budget small for Hamming(7,4)
hqc = HQC(params=params, rng=rng)

pk, sk = hqc.keygen()
print("Public key (h, s):\n h =", pk[0], "\n s =", pk[1])
print("Secret key (x, y):\n x =", sk[0], "\n y =", sk[1])

# 4-bit message, e.g., 1011
m = np.array([1, 0, 1, 1], dtype=int)
print("\nPlain message m (4 bits):", m)

u, v = hqc.encrypt(m)
print("\nCiphertext (u, v):\n u =", u, "\n v =", v)

m_hat, t, c_hat = hqc.decrypt((u, v))
print("\nDecryption intermediate t = v + u·y:", t)
print("Corrected codeword from decoder:", c_hat)
print("Recovered message m_hat:", m_hat)

ok = np.array_equal(m, m_hat)
print("\nSUCCESS:", ok)
