"""
Parallel Call Graph Analyzer — 24 Ryzen Cores

Extracts call graphs from ALL SuperInstance repos in parallel,
computes spectral signatures, and finds structural similarities.

Uses multiprocessing to parse all repos simultaneously.
"""

import multiprocessing as mp
import os
import re
import time
import json
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib


def extract_python_calls(filepath):
    """Extract function definitions and calls from a Python file."""
    try:
        with open(filepath, 'r', errors='ignore') as f:
            content = f.read()
    except:
        return {}, {}
    
    functions = {}
    call_graph = defaultdict(set)
    
    lines = content.split('\n')
    current_func = None
    indent_level = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        
        # Detect function definitions
        match = re.match(r'def\s+(\w+)\s*\(', stripped)
        if match:
            func_name = match.group(1)
            indent_level = len(line) - len(line.lstrip())
            current_func = func_name
            functions[func_name] = filepath
            continue
        
        # Detect calls within functions
        if current_func:
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and stripped and not stripped.startswith(('"""', "'''", '@', 'pass', 'return', 'if', 'else', 'for', 'while', 'try', 'except', 'with', 'class')):
                # Check if we exited the function
                if current_indent <= indent_level and re.match(r'def |class ', stripped):
                    current_func = None
                    continue
            
            # Extract function calls
            calls = re.findall(r'(\w+)\s*\(', stripped)
            for call in calls:
                if call not in ('print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr', 'super', 'property'):
                    call_graph[current_func].add(call)
    
    return functions, dict(call_graph)


def extract_rust_calls(filepath):
    """Extract function definitions and calls from a Rust file."""
    try:
        with open(filepath, 'r', errors='ignore') as f:
            content = f.read()
    except:
        return {}, {}
    
    functions = {}
    call_graph = defaultdict(set)
    
    lines = content.split('\n')
    current_func = None
    
    for line in lines:
        stripped = line.strip()
        
        # Detect fn definitions
        match = re.match(r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)', stripped)
        if match:
            current_func = match.group(1)
            functions[current_func] = filepath
            continue
        
        if current_func and '{' not in stripped and '}' in stripped:
            current_func = None
            continue
        
        # Extract calls
        if current_func:
            calls = re.findall(r'(\w+)\s*\(', stripped)
            for call in calls:
                if call not in ('if', 'for', 'while', 'match', 'Some', 'None', 'Ok', 'Err', 'Vec', 'String', 'Box', 'Arc', 'Rc', 'println', 'format', 'vec', 'assert', 'unwrap', 'expect'):
                    call_graph[current_func].add(call)
    
    return functions, dict(call_graph)


def analyze_repo(args):
    """Analyze a single repo's call graph. Called by worker processes."""
    repo_name, repo_path = args
    
    all_functions = {}
    all_calls = defaultdict(set)
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ['node_modules', '.venv', 'target', '__pycache__', '.git', 'dist', 'build']]
        
        for f in files:
            filepath = os.path.join(root, f)
            
            if f.endswith('.py'):
                funcs, calls = extract_python_calls(filepath)
            elif f.endswith('.rs'):
                funcs, calls = extract_rust_calls(filepath)
            else:
                continue
            
            all_functions.update(funcs)
            for func, call_list in calls.items():
                all_calls[func].update(call_list)
    
    return {
        'name': repo_name,
        'functions': len(all_functions),
        'edges': sum(len(v) for v in all_calls.values()),
        'call_graph': {k: list(v) for k, v in all_calls.items()},
    }


def compute_spectral_signature(call_graph, dim=32):
    """Compute fixed-size spectral signature from call graph."""
    all_funcs = set(call_graph.keys())
    for calls in call_graph.values():
        all_funcs.update(calls)
    
    func_list = sorted(all_funcs)
    n = len(func_list)
    if n == 0:
        return np.zeros(dim)
    
    func_idx = {f: i for i, f in enumerate(func_list)}
    adj = np.zeros((n, n), dtype=np.float32)
    
    for func, calls in call_graph.items():
        if func in func_idx:
            for called in calls:
                if called in func_idx:
                    adj[func_idx[func], func_idx[called]] = 1.0
    
    # Spectral embedding via Laplacian
    degrees = np.sum(adj, axis=1) + 1e-10
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L = np.diag(degrees) - adj
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
        # Use eigenvalues (always fixed-size) + top eigenvector moments
        sig = np.zeros(dim, dtype=np.float64)
        # First part: eigenvalue distribution stats
        usable = min(len(eigenvalues), dim)
        sig[:usable] = eigenvalues[:usable]
        # Also encode degree distribution moments
        if dim > len(eigenvalues):
            deg_stats = np.array([
                np.mean(degrees), np.std(degrees), np.min(degrees), np.max(degrees),
                np.median(degrees), np.percentile(degrees, 25), np.percentile(degrees, 75),
            ])
            offset = len(eigenvalues)
            end = min(offset + len(deg_stats), dim)
            sig[offset:end] = deg_stats[:end - offset]
        if np.linalg.norm(sig) > 0:
            sig /= np.linalg.norm(sig)
        return sig
    except:
        return np.zeros(dim)


def main():
    print("=" * 60)
    print("PARALLEL CALL GRAPH ANALYZER — 24 Ryzen Cores")
    print("=" * 60)
    
    repos = {}
    base = os.path.expanduser("~/repos")
    for name in ['lever-runner', 'pincherOS', 'open-minded', 'zeroclaw-arena', 
                  'fastloop-guard', 'metal-lathe', 'conservation-spectral-topology-rs',
                  'agent-template', 'intelligent-terminal']:
        path = os.path.join(base, name)
        if os.path.exists(path):
            repos[name] = path
    
    print(f"Found {len(repos)} repos")
    print(f"Using {mp.cpu_count()} cores")
    
    # Parallel extraction
    print(f"\nExtracting call graphs in parallel...")
    start = time.perf_counter()
    
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(executor.map(analyze_repo, repos.items()))
    
    elapsed = time.perf_counter() - start
    print(f"Extracted in {elapsed:.1f}s")
    
    for r in results:
        print(f"  {r['name']}: {r['functions']} functions, {r['edges']} edges")
    
    # Compute spectral signatures
    print(f"\nComputing spectral signatures...")
    signatures = {}
    for r in results:
        sig = compute_spectral_signature(r['call_graph'])
        signatures[r['name']] = sig
    
    # Cross-repo similarity
    print(f"\n=== SPECTRAL SIMILARITY MATRIX ===")
    names = list(signatures.keys())
    print(f"{'':>25}" + "".join(f"{n[:12]:>13}" for n in names))
    
    for i, a in enumerate(names):
        row = f"{a:>25}"
        for j, b in enumerate(names):
            if i == j:
                row += f"{'1.0000':>13}"
            elif i < j:
                sim = np.dot(signatures[a], signatures[b])
                row += f"{sim:>13.4f}"
            else:
                row += f"{'':>13}"
        print(row)
    
    # Find most/least similar pairs
    pairs = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                sim = np.dot(signatures[a], signatures[b])
                pairs.append((a, b, sim))
    
    pairs.sort(key=lambda x: -x[2])
    
    print(f"\nMost similar:")
    for a, b, sim in pairs[:3]:
        print(f"  {a} ↔ {b}: {sim:.4f}")
    
    print(f"\nLeast similar:")
    for a, b, sim in pairs[-3:]:
        print(f"  {a} ↔ {b}: {sim:.4f}")
    
    # Save
    output = {
        'repos': {r['name']: {'functions': r['functions'], 'edges': r['edges']} for r in results},
        'similarity_matrix': {f"{a}-{b}": float(s) for a, b, s in pairs},
        'extraction_time_s': elapsed,
        'cores_used': mp.cpu_count(),
    }
    
    out = os.path.expanduser("~/repos/conservation-spectral-topology-rs/tools/callgraph-results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
