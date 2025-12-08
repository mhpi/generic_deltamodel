
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular, splu
from collections import deque


def reachability_matrix(G, row_nodes, col_nodes):
    """
    M[i, j] = 1 if col_nodes[j] can reach row_nodes[i] (downstream),
    counting self-reach as True.
    """
    row_nodes = [int(n) for n in row_nodes]
    col_nodes = [int(n) for n in col_nodes]
    RG = G.reverse(copy=False)
    col_index = {n: j for j, n in enumerate(col_nodes)}

    # BFS once per unique row
    unique_rows = list(dict.fromkeys(row_nodes))
    cache_row_to_js = {}

    for r in unique_rows:
        if r not in RG:
            cache_row_to_js[r] = [col_index[r]] if r in col_index else []
            continue
        seen = {r}
        dq = deque([r])
        js = set()
        while dq:
            u = dq.popleft()
            j = col_index.get(u)
            if j is not None:
                js.add(j)
            for v in RG[u]:
                if v not in seen:
                    seen.add(v);
                    dq.append(v)
        cache_row_to_js[r] = list(js)

    M = np.zeros((len(row_nodes), len(col_nodes)), dtype=np.int8)
    for i, r in enumerate(row_nodes):
        js = cache_row_to_js[int(r)]
        if js:  # vectorized row fill
            M[i, js] = 1
    return M


class PathWeightedAgg:
    """
    Fast path aggregations on a river-style DiGraph (each node has out-degree <= 1).
    Inclusive path a->b (a, ..., b).

    Features:
      - reduction: "mean" (default) or "sum"
      - if y_attr is None: use naive mean/sum over x
      - if y_attr is provided: use weighted mean/sum with weights y
    """

    def __init__(self, G: nx.DiGraph, x_attr="x", y_attr="y"):
        # Structure checks
        if any(G.out_degree(n) > 1 for n in G.nodes()):
            raise ValueError("Graph must have out-degree ≤ 1 per node (unique downstream).")
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Graph must be a DAG (no cycles).")

        self.G = G
        self.x_attr = x_attr
        self.y_attr = y_attr  # can be None

        nodes = list(G.nodes())
        nid = {n: i for i, n in enumerate(nodes)}
        N = len(nodes)

        parent = np.full(N, -1, dtype=np.int64)
        children = [[] for _ in range(N)]
        for u, v in G.edges():
            iu, iv = nid[u], nid[v]
            parent[iu] = iv
            children[iv].append(iu)

        roots = [nid[n] for n in nodes if G.out_degree(n) == 0]

        # attributes
        x = np.array([G.nodes[n].get(x_attr, 0.0) for n in nodes], dtype=float)
        if y_attr is None:
            y = None
            xy = None
        else:
            y = np.array([G.nodes[n].get(y_attr, 0.0) for n in nodes], dtype=float)
            xy = x * y

        # Prefix sums on reversed forest (root -> node), inclusive
        tin = np.full(N, -1, dtype=np.int64)
        tout = np.full(N, -1, dtype=np.int64)

        # Always keep Sx and Cnt for naive paths
        Sx = np.zeros(N, dtype=float)
        Cnt = np.zeros(N, dtype=np.int64)

        # Weighted prefixes only if y provided
        Sy = np.zeros(N, dtype=float) if y is not None else None
        Sxy = np.zeros(N, dtype=float) if y is not None else None

        timer = 0
        for r in roots:
            if tin[r] != -1:
                continue
            # initialize root
            Sx[r] = x[r]
            Cnt[r] = 1
            if y is not None:
                Sy[r] = y[r]
                Sxy[r] = xy[r]

            stack = [(r, 0)]
            while stack:
                u, state = stack.pop()
                if state == 0:  # enter
                    tin[u] = timer;
                    timer += 1
                    stack.append((u, 1))
                    for v in children[u]:
                        Sx[v] = Sx[u] + x[v]
                        Cnt[v] = Cnt[u] + 1
                        if y is not None:
                            Sy[v] = Sy[u] + y[v]
                            Sxy[v] = Sxy[u] + xy[v]
                        stack.append((v, 0))
                else:  # exit
                    tout[u] = timer  # inclusive-interval style

        self.nodes = nodes
        self.nid = nid
        self.parent = parent
        self.tin = tin
        self.tout = tout
        self.Sx = Sx
        self.Cnt = Cnt
        self.Sy = Sy
        self.Sxy = Sxy
        self.has_weights = (y is not None)

    def _is_ancestor(self, b_idx: int, a_idx: int) -> bool:
        # In reversed forest: b ancestor of a  <=>  b is downstream of a in original graph
        return (self.tin[b_idx] <= self.tin[a_idx]) and (self.tout[a_idx] <= self.tout[b_idx])

    def _inclusive_prefix_slice(self, arr, a_idx, b_idx):
        """Sum over inclusive path a->b using root-to-node prefixes; arr is a prefix array."""
        if arr is None:
            return None
        pb = self.parent[b_idx]
        return arr[a_idx] if pb == -1 else (arr[a_idx] - arr[pb])

    def query(self, a, b, reduction="mean"):
        """
        Compute aggregation over inclusive path a->b.
        reduction: "mean" or "sum"
          - if weights present: weighted mean/sum using y
          - if weights absent (y_attr=None): naive mean/sum over x
        Returns np.nan if no a->b path; for weighted mean with zero total weight, returns np.nan.
        """
        ia, ib = self.nid[a], self.nid[b]
        if not self._is_ancestor(ib, ia):
            return np.nan

        sum_x = self._inclusive_prefix_slice(self.Sx, ia, ib)
        cnt = self._inclusive_prefix_slice(self.Cnt, ia, ib)

        if reduction == "sum":
            if self.has_weights:
                sum_xy = self._inclusive_prefix_slice(self.Sxy, ia, ib)
                return sum_xy
            else:
                return sum_x

        # reduction == "mean"
        if self.has_weights:
            sum_y = self._inclusive_prefix_slice(self.Sy, ia, ib)
            sum_xy = self._inclusive_prefix_slice(self.Sxy, ia, ib)
            return np.nan if (sum_y == 0) else (sum_xy / sum_y)
        else:
            return sum_x / cnt  # cnt >= 1 for a valid path

    def query_many(self, pairs, reduction="mean"):
        """
        Vectorized batch for many (a, b). Directional (a->b only).
        """
        ia = np.fromiter((self.nid[a] for a, _ in pairs), dtype=np.int64)
        ib = np.fromiter((self.nid[b] for _, b in pairs), dtype=np.int64)

        # path existence mask
        ab = (self.tin[ib] <= self.tin[ia]) & (self.tout[ia] <= self.tout[ib])

        pb = self.parent[ib]
        # naive prefixes
        sum_x = np.where(pb == -1, self.Sx[ia], self.Sx[ia] - self.Sx[pb])
        cnt = np.where(pb == -1, self.Cnt[ia], self.Cnt[ia] - self.Cnt[pb])

        out = np.full(len(ia), np.nan, dtype=float)

        if reduction == "sum":
            if self.has_weights:
                sum_xy = np.where(pb == -1, self.Sxy[ia], self.Sxy[ia] - self.Sxy[pb])
                out[ab] = sum_xy[ab]
            else:
                out[ab] = sum_x[ab]
            return out

        # reduction == "mean"
        if self.has_weights:
            sum_y = np.where(pb == -1, self.Sy[ia], self.Sy[ia] - self.Sy[pb])
            sum_xy = np.where(pb == -1, self.Sxy[ia], self.Sxy[ia] - self.Sxy[pb])
            valid = ab & (sum_y != 0)
            out[valid] = (sum_xy[valid] / sum_y[valid])
            # remain NaN where no path or zero total weight
        else:
            out[ab] = (sum_x[ab] / cnt[ab])
        return out


def outlet_accum_attribute(
    G,
    outlets,
    A,
    W,
    agg="mean",           # "mean" or "sum"
    fill_value=np.nan     # value to fill for unreachable nodes (default NaN)
):
    """
    Compute aggregated attribute along flow-paths to each outlet.
    Args:
      G        : networkx.DiGraph with edges upstream -> downstream (must be DAG).
      outlets  : list of node ids; rows of returned matrix follow this order.
      A        : 1D numpy array (topo order) or dict/node->value for attribute.
      W        : 1D numpy array (topo order) or dict/node->value for weight.
                 If agg == "sum" you can pass W=None (then W treated as 1).
      agg      : "mean" (weighted mean = sum(A*W)/sum(W)) or "sum" (sum of A*W).
      fill_value: value to put where node does not reach outlet (or denom==0).
    Returns:
      Out: numpy array shape (No, N) where No=len(outlets) and N=number of topo nodes.
           Columns are in topo-node order (returned topo_nodes maps col idx->node id).
      topo_nodes: list of nodes in topological (upstream->downstream) order (columns order).
    """
    # 1) topo order
    topo_nodes = list(nx.topological_sort(G))
    N = len(topo_nodes)
    node2pos = {n: i for i, n in enumerate(topo_nodes)}

    # helper -> array in topo order
    def to_array(x, default=None):
        if x is None:
            if default is None:
                raise ValueError("x is None but no default provided")
            return np.full(N, default, dtype=float)
        if isinstance(x, np.ndarray):
            if x.shape[0] != N:
                raise ValueError("If array, it must be length N and already in topo order.")
            return x.astype(float).copy()
        # dict-like
        arr = np.empty(N, dtype=float)
        for i, n in enumerate(topo_nodes):
            arr[i] = x[n]
        return arr

    A_arr = to_array(A)
    W_arr = to_array(W, default=1.0)   # default weight = 1 if W is None

    # 2) b = A * W (numerator)
    b_arr = A_arr * W_arr

    # 3) adjacency D in topo order (D[i,j] = 1 if topo_nodes[i] -> topo_nodes[j])
    D = nx.to_scipy_sparse_array(G, format="csr", nodelist=topo_nodes, dtype=float)
    I_minus_D = sp.eye(N, format="csr", dtype=float) - D

    # 4) prepare E matrix where column k is unit vector at topo index of outlets[k]
    outlets = list(outlets)
    for o in outlets:
        if o not in node2pos:
            raise KeyError(f"outlet {o} not in graph nodes")
    No = len(outlets)
    E = np.zeros((N, No), dtype=float)
    for k, o in enumerate(outlets):
        E[node2pos[o], k] = 1.0

    # 5) R_out = (I - D)^{-1} E  (N x No)  -> reachability columns
    try:
        R_out = spsolve_triangular(I_minus_D, E, lower=False)   # fast when triangular solver available
    except Exception:
        lu = splu(I_minus_D.tocsc())
        R_out = lu.solve(E)

    if agg == "sum":
        # Single pair of solves: RHS = b * R_out; then S = (I - D)^{-1} RHS
        RHS = (b_arr[:, None]) * R_out    # N x No
        try:
            S = spsolve_triangular(I_minus_D, RHS, lower=False)
        except Exception:
            lu = splu(I_minus_D.tocsc())
            S = np.column_stack([lu.solve(RHS[:, k]) for k in range(No)])
        Out = S.T  # No x N ; each entry is sum_{v on path u->outlet} A[v]*W[v]
        # Mask unreachable nodes (where R_out==0): set fill_value
        reach_mask = (R_out.T > 0.5)  # No x N
        Out[~reach_mask] = fill_value
        return Out, topo_nodes

    elif agg == "mean":
        # Need numerator (b) and denominator (W) accumulations separately
        RHS_b = (b_arr[:, None]) * R_out
        RHS_w = (W_arr[:, None]) * R_out
        try:
            S_b = spsolve_triangular(I_minus_D, RHS_b, lower=False)
            S_w = spsolve_triangular(I_minus_D, RHS_w, lower=False)
        except Exception:
            lu = splu(I_minus_D.tocsc())
            S_b = np.column_stack([lu.solve(RHS_b[:, k]) for k in range(No)])
            S_w = np.column_stack([lu.solve(RHS_w[:, k]) for k in range(No)])
        # Transpose to No x N
        S_b_T = S_b.T
        S_w_T = S_w.T
        # compute mean; avoid div-by-zero
        Out = np.full_like(S_b_T, fill_value=np.nan, dtype=float)
        denom_pos = S_w_T != 0.0
        Out[denom_pos] = S_b_T[denom_pos] / S_w_T[denom_pos]
        # enforce unreachable nodes -> fill_value (if unreachable but denom>0 through numerical weirdness)
        reach_mask = (R_out.T > 0.5)
        Out[~reach_mask] = fill_value
        return Out, topo_nodes

    else:
        raise ValueError("agg must be 'mean' or 'sum'")






