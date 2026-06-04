"""
Spectral Perturbation Test — Is the >0.97 isomorphism real or artifact?

Hypothesis: If the similarity is from coding style, perturbing function names 
but keeping the structure should barely change the similarity. If it's genuine 
structural invariant, perturbing names should NOT change it (it's about structure,
not names).

Control: Original code from 3 repos
Test 1: Randomize function names (preserve structure)
Test 2: Randomize call patterns (preserve function count)
Test 3: Randomize both
Test 4: Add noise to the spectral vectors directly

If similarity stays >0.95 after name randomization → genuine structural invariant
If similarity drops below 0.80 → style/parsing artifact
"""

import numpy as np
import hashlib
import json
import os
import random
import sqlite3
from collections import defaultdict

def extract_call_graph(repo_path):
    """Extract function names and their call targets from Python source."""
    functions = {}
    call_graph = defaultdict(set)
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ['node_modules', '.venv', 'target', '__pycache__', '.git']]
        for f in files:
            if not (f.endswith('.py') or f.endswith('.rs')):
                continue
            filepath = os.path.join(root, f)
            try:
                with open(filepath) as fh:
                    content = fh.read()
            except:
                continue
            
            is_rust = filepath.endswith('.rs')
            lines = content.split('\n')
            current_func = None
            brace_depth = 0
            for line in lines:
                stripped = line.strip()
                if is_rust:
                    # Rust: fn name(
                    if stripped.startswith('pub fn ') or stripped.startswith('fn '):
                        prefix = 'pub fn ' if stripped.startswith('pub fn ') else 'fn '
                        rest = stripped[len(prefix):]
                        if '(' in rest:
                            name = rest[:rest.index('(')].strip()
                            current_func = name
                            functions[name] = {'file': filepath, 'calls': []}
                            brace_depth = 0
                    elif current_func:
                        brace_depth += stripped.count('{') - stripped.count('}')
                        if brace_depth < 0:
                            current_func = None
                            continue
                        # Look for function calls
                        for word in stripped.replace('(', ' (').split():
                            if '(' in word:
                                called = word[:word.index('(')].strip().rstrip(':')
                                if called and not called.startswith(('//', '#', '"', 'self', 'super', 'crate')) and called not in ('if', 'while', 'for', 'match', 'return', 'let', 'mut', 'pub'):
                                    call_graph[current_func].add(called)
                                    functions[current_func]['calls'].append(called)
                else:
                    # Python: def name(
                    if stripped.startswith('def ') and ':' in stripped:
                        name = stripped[4:stripped.index('(')].strip()
                        current_func = name
                        functions[name] = {'file': filepath, 'calls': []}
                    elif current_func and stripped and not stripped.startswith('#'):
                        for word in stripped.split():
                            if '(' in word:
                                called = word.split('(')[0].strip()
                                if called and not called.startswith('#'):
                                    call_graph[current_func].add(called)
                                    functions[current_func]['calls'].append(called)
    
    return functions, dict(call_graph)


def graph_to_adjacency(call_graph, all_functions):
    """Convert call graph to adjacency matrix."""
    n = len(all_functions)
    func_list = sorted(all_functions)
    func_idx = {f: i for i, f in enumerate(func_list)}
    
    adj = np.zeros((n, n), dtype=np.float32)
    for func, calls in call_graph.items():
        if func in func_idx:
            for called in calls:
                if called in func_idx:
                    adj[func_idx[func], func_idx[called]] = 1.0
    
    return adj, func_list


def spectral_embedding(adj, dim=32):
    """Compute spectral embedding from adjacency matrix.
    Returns a FIXED-size signature vector of length `dim` by using
    eigenvalue histogram + graph statistics, so different-sized graphs
    can be compared.
    """
    n = adj.shape[0]
    
    # Degree matrix
    degrees = np.sum(adj, axis=1)
    
    # Laplacian
    D = np.diag(degrees + 1e-10)
    L = D - adj
    
    # Normalize
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    eigs = eigenvalues[1:]  # skip first (~0)
    
    # Build fixed-size signature from eigenvalue distribution
    # Bin eigenvalues into histogram of `dim` bins over [0, 2]
    hist, _ = np.histogram(eigs, bins=dim, range=(0, 2))
    hist = hist.astype(np.float64)
    
    # Append graph-level stats (pad/trim to fill remaining dims)
    stats = np.array([
        n,  # number of nodes
        np.sum(adj),  # number of edges
        np.mean(degrees),
        np.std(degrees),
        np.max(degrees),
        np.min(degrees),
        np.mean(eigs) if len(eigs) > 0 else 0,
        np.std(eigs) if len(eigs) > 0 else 0,
        np.max(eigs) if len(eigs) > 0 else 0,
        np.sum(eigs),  # trace of normalized Laplacian
        len(eigs[eigs < 0.1]) / max(len(eigs), 1),  # fraction near-zero eigenvalues
        len(eigs[eigs > 1.9]) / max(len(eigs), 1),  # fraction near-2 eigenvalues
    ])
    
    signature = np.concatenate([hist, stats])
    if np.linalg.norm(signature) > 0:
        signature /= np.linalg.norm(signature)
    
    return signature, eigs[:dim]


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def perturb_names(call_graph, seed=42):
    """Randomize function names but keep structure identical."""
    rng = random.Random(seed)
    names = list(call_graph.keys())
    new_names = [f"fn_{rng.randint(0, 99999)}" for _ in names]
    name_map = dict(zip(names, new_names))
    
    perturbed = {}
    for func, calls in call_graph.items():
        new_func = name_map.get(func, func)
        new_calls = [name_map.get(c, c) for c in calls]
        perturbed[new_func] = new_calls
    
    return perturbed


def perturb_structure(call_graph, seed=42):
    """Keep function names but randomize call patterns."""
    rng = random.Random(seed)
    all_funcs = list(call_graph.keys())
    
    perturbed = {}
    for func in all_funcs:
        n_calls = len(call_graph[func])
        new_calls = [rng.choice(all_funcs) for _ in range(n_calls)]
        perturbed[func] = new_calls
    
    return perturbed


def add_vector_noise(signature, noise_level=0.1, seed=42):
    """Add Gaussian noise directly to spectral signature."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*signature.shape) * noise_level
    noisy = signature + noise
    return noisy / (np.linalg.norm(noisy) + 1e-10)


def run_experiment():
    repos = {
        'lever-runner': os.path.expanduser('~/repos/lever-runner/src'),
        'pincherOS': os.path.expanduser('~/repos/pincherOS/pincher-core/src'),
        'zeroclaw-arena': os.path.expanduser('~/repos/zeroclaw-arena'),
        'fastloop-guard': os.path.expanduser('~/repos/fastloop-guard/src'),
        'metal-lathe': os.path.expanduser('~/repos/metal-lathe'),
    }
    
    print("=" * 70)
    print("SPECTRAL PERTURBATION TEST")
    print("Is the >0.97 isomorphism genuine or artifact?")
    print("=" * 70)
    
    # Extract call graphs
    graphs = {}
    for name, path in repos.items():
        if os.path.exists(path):
            funcs, cg = extract_call_graph(path)
            graphs[name] = cg
            print(f"  {name}: {len(funcs)} functions, {sum(len(v) for v in cg.values())} edges")
        else:
            print(f"  {name}: path not found ({path})")
    
    if len(graphs) < 2:
        print("Need at least 2 repos with functions!")
        return
    
    # Compute baseline signatures
    all_funcs = set()
    for cg in graphs.values():
        all_funcs.update(cg.keys())
    
    signatures = {}
    for name, cg in graphs.items():
        all_funcs_repo = set(cg.keys())
        for calls in cg.values():
            all_funcs_repo.update(calls)
        if len(all_funcs_repo) < 2:
            print(f"  Skipping {name}: too few functions ({len(all_funcs_repo)})")
            continue
        adj, func_list = graph_to_adjacency(cg, all_funcs_repo)
        if adj.shape[0] < 2:
            continue
        sig, eigs = spectral_embedding(adj)
        signatures[name] = sig
    
    # Baseline similarity matrix
    print("\n--- BASELINE SIMILARITY ---")
    repo_names = list(signatures.keys())
    for i, a in enumerate(repo_names):
        for j, b in enumerate(repo_names):
            if i < j:
                sim = cosine_sim(signatures[a], signatures[b])
                print(f"  {a} ↔ {b}: {sim:.4f}")
    
    # Test 1: Perturb names
    print("\n--- TEST 1: RANDOMIZED FUNCTION NAMES (preserve structure) ---")
    perturbed_sigs = {}
    for name in repo_names:
        cg = graphs[name]
        pcg = perturb_names(cg)
        all_funcs_p = set(pcg.keys())
        for calls in pcg.values():
            all_funcs_p.update(calls)
        adj, _ = graph_to_adjacency(pcg, all_funcs_p)
        sig, _ = spectral_embedding(adj)
        perturbed_sigs[name] = sig
    
    for i, a in enumerate(repo_names):
        for j, b in enumerate(repo_names):
            if i < j:
                orig = cosine_sim(signatures[a], signatures[b])
                perturbed = cosine_sim(perturbed_sigs[a], perturbed_sigs[b])
                delta = perturbed - orig
                print(f"  {a} ↔ {b}: orig={orig:.4f} perturbed={perturbed:.4f} delta={delta:+.4f}")
    
    # Test 2: Perturb structure
    print("\n--- TEST 2: RANDOMIZED CALL PATTERNS (preserve names) ---")
    struct_sigs = {}
    for name in repo_names:
        cg = graphs[name]
        scg = perturb_structure(cg)
        all_funcs_s = set(scg.keys())
        for calls in scg.values():
            all_funcs_s.update(calls)
        adj, _ = graph_to_adjacency(scg, all_funcs_s)
        sig, _ = spectral_embedding(adj)
        struct_sigs[name] = sig
    
    for i, a in enumerate(repo_names):
        for j, b in enumerate(repo_names):
            if i < j:
                orig = cosine_sim(signatures[a], signatures[b])
                perturbed = cosine_sim(struct_sigs[a], struct_sigs[b])
                delta = perturbed - orig
                print(f"  {a} ↔ {b}: orig={orig:.4f} perturbed={perturbed:.4f} delta={delta:+.4f}")
    
    # Test 3: Vector noise
    print("\n--- TEST 3: VECTOR NOISE (direct perturbation) ---")
    for noise_level in [0.01, 0.05, 0.1, 0.2, 0.5]:
        print(f"  noise={noise_level}:")
        for i, a in enumerate(repo_names):
            for j, b in enumerate(repo_names):
                if i < j:
                    noisy_a = add_vector_noise(signatures[a], noise_level)
                    noisy_b = add_vector_noise(signatures[b], noise_level)
                    sim = cosine_sim(noisy_a, noisy_b)
                    print(f"    {a} ↔ {b}: {sim:.4f}")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    # Check: does name perturbation change similarity?
    name_changes = []
    for i, a in enumerate(repo_names):
        for j, b in enumerate(repo_names):
            if i < j:
                orig = cosine_sim(signatures[a], signatures[b])
                pert = cosine_sim(perturbed_sigs[a], perturbed_sigs[b])
                name_changes.append(abs(orig - pert))
    
    avg_name_change = np.mean(name_changes)
    print(f"Average similarity change from name randomization: {avg_name_change:.4f}")
    
    if avg_name_change < 0.05:
        print("→ ISOMORPHISM IS GENUINE: randomizing names barely changes spectral signature")
        print("  The structural invariant is real — it's about call graph topology, not naming")
    elif avg_name_change < 0.15:
        print("→ MIXED: some genuine structure, some naming artifact")
    else:
        print("→ ISOMORPHISM IS ARTIFACT: names significantly affect spectral signature")
    
    # Save results
    results = {
        "baseline": {f"{a}-{b}": cosine_sim(signatures[a], signatures[b]) 
                     for i, a in enumerate(repo_names) for j, b in enumerate(repo_names) if i < j},
        "name_perturbation_avg_change": float(avg_name_change),
        "verdict": "genuine" if avg_name_change < 0.05 else ("mixed" if avg_name_change < 0.15 else "artifact")
    }
    
    out_path = os.path.expanduser("~/repos/conservation-spectral-topology-rs/experiments/spectral_perturbation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
