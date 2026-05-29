# Cold-item nearest-neighbor visualization

Top-5 cosine neighbors of three cold items (each with 4 training ratings, 1
positive test rating) in three embedding spaces:

- **SemMF V** — item embedding from `semmf_mse_adaptive` (seed 42, λ_0=0.5),
  pulled toward `f(e)` by the regularizer during training.
- **BPR-MF V** — item embedding from `bpr_mf` (seed 42), no regularizer; for
  cold items V is essentially the random init.
- **Raw LLM e** — the `all-MiniLM-L6-v2` embedding of the item's text.

The contrast shows that the regularizer is doing its job: SemMF's `V[cold]`
stays genre-coherent, BPR-MF's `V[cold]` is dominated by random-init clusters
(very high cosines among unrelated items because the vectors are all small
noise), and the raw LLM picks up surface text patterns (e.g. the word "Dance"
in titles) that the trained `V` smooths out.

---

## 1. Rent-a-Kid (1995) — Comedy

### SemMF-MSE-Adaptive V

| cos | item | genres |
|---:|---|---|
| 0.9410 | Vermin (1998) | Comedy |
| 0.9404 | Odd Couple II, The (1998) | Comedy |
| 0.9315 | Problem Child 2 (1991) | Comedy |
| 0.9314 | Mr. Magoo (1997) | Comedy |
| 0.9193 | 8 Heads in a Duffel Bag (1997) | Comedy |

### BPR-MF V

| cos | item | genres |
|---:|---|---|
| 0.9958 | Strike! (1998) | Comedy |
| 0.9950 | Ugly, The (1997) | **Horror, Thriller** |
| 0.9947 | Country Life (1994) | **Drama, Romance** |
| 0.9941 | Smile Like Yours, A (1997) | Comedy, Romance |
| 0.9917 | Squeeze (1996) | **Drama** |

### Raw LLM e

| cos | item | genres |
|---:|---|---|
| 0.7517 | Rent-A-Cop (1988) | Action, Comedy |
| 0.7501 | Roommates (1995) | Comedy, Drama |
| 0.7325 | SubUrbia (1997) | Comedy |
| 0.7041 | Problem Child (1990) | Comedy |
| 0.7031 | Mutters Courage (1995) | Comedy |

---

## 2. Last Dance (1996) — Drama

### SemMF-MSE-Adaptive V

| cos | item | genres |
|---:|---|---|
| 0.9811 | It's My Party (1995) | Drama |
| 0.9758 | Boy Called Hate, A (1995) | Drama |
| 0.9688 | Love, etc. (1996) | Drama |
| 0.9683 | August (1996) | Drama |
| 0.9648 | Bye-Bye (1995) | Drama |

### BPR-MF V

| cos | item | genres |
|---:|---|---|
| 0.9984 | Zone 39 (1997) | **Sci-Fi** |
| 0.9979 | Under Capricorn (1949) | Drama |
| 0.9976 | Hostile Intentions (1994) | **Action, Drama, Thriller** |
| 0.9974 | Train Ride to Hollywood (1978) | **Comedy** |

### Raw LLM e

| cos | item | genres |
|---:|---|---|
| 0.7710 | Dance with Me (1998) | Drama, Romance |
| 0.7522 | Dancing at Lughnasa (1998) | Drama |
| 0.7152 | Dancer in the Dark (2000) | Drama, Musical |
| 0.7142 | Dancemaker (1998) | Documentary |
| 0.7098 | Tango (1998) | Drama |

(LLM is matching on the word "Dance" in the title — a literal text-overlap artifact.)

---

## 3. Love Bewitched / El Amor Brujo (1986) — Musical, foreign-language

### SemMF-MSE-Adaptive V

| cos | item | genres |
|---:|---|---|
| 0.9389 | Man Facing Southeast / Hombre Mirando al Sudeste (1986) | Drama |
| 0.9364 | Draughtsman's Contract, The (1982) | Drama |
| 0.9176 | Invitation, The / Zaproszenie (1986) | Drama |
| 0.9171 | Saragossa Manuscript, The / Rekopis znaleziony w Saragossie (1965) | Drama |
| 0.9093 | Old Lady Who Walked in the Sea, The / Vieille qui marchait dans la mer (1991) | Comedy |

### BPR-MF V

| cos | item | genres |
|---:|---|---|
| 0.9953 | Nekromantik (1987) | **Comedy, Horror** |
| 0.9939 | Tough and Deadly (1995) | **Action, Drama, Thriller** |
| 0.9939 | End of the Affair, The (1955) | Drama |
| 0.9938 | Always Tell Your Wife (1923) | **Comedy** |
| 0.9935 | Born to Win (1971) | Drama |

### Raw LLM e

| cos | item | genres |
|---:|---|---|
| 0.6720 | Pretty in Pink (1986) | Comedy, Drama, Romance |
| 0.6589 | Ennui, L' (1998) | Drama, Romance |
| 0.6557 | Desert Bloom (1986) | Drama |
| 0.6502 | Cinema Paradiso (1988) | Comedy, Drama, Romance |
| 0.6478 | Big Blue, The / Le Grand Bleu (1988) | Adventure, Romance |

---

## Read

The BPR-MF cosines for cold items are systematically **higher in absolute
value** (0.99+) but lower in meaning — they reflect that all cold V's are
small noise vectors near the random init, so they cluster tightly with each
other regardless of content. SemMF's regularizer pulls `V[cold]` toward
`f(e[cold])`, breaking that random clustering and aligning cold items with
genre/era neighbors. The cosines are smaller (0.91–0.98) but actually
correspond to meaningful relationships.

This is the qualitative signal behind the +20% Cold NDCG@10 lift of
SemMF-MSE-Adaptive over BPR-MF.
