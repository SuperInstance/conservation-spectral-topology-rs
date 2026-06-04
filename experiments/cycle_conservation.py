"""
Cycle Conservation Law — The Real Structural Invariant

Hypothesis: The ratio of cycle density (self-calls + mutual pairs) to total edges
is a conserved quantity within a codebase's evolutionary trajectory.
It changes when architectural shifts happen, but is stable under normal development.

Method:
1. Extract call graphs from lever-runner at multiple git commits
2. Compute cycle metrics at each commit
3. Track how cycle density changes over time
4. Determine if it's stable (conservation) or drifting (no conservation)
"""

import subprocess
import os
import re
import json
import time
import numpy as np
from collections import defaultdict, Counter


def extract_call_graph(repo_path, commit=None):
    """Extract call graph at a specific commit."""
    cwd = os.getcwd()
    os.chdir(repo_path)
    
    if commit:
        subprocess.run(['git', 'stash'], capture_output=True)
        subprocess.run(['git', 'checkout', commit], capture_output=True)
    
    all_calls = defaultdict(set)
    all_funcs = set()
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ['node_modules', '.venv', 'target', '__pycache__', '.git', '.benchmarks']]
        for f in files:
            if f.endswith(('.py', '.rs')):
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r', errors='ignore') as fh:
                        content = fh.read()
                except:
                    continue
                
                current_func = None
                for line in content.split('\n'):
                    stripped = line.strip()
                    
                    if f.endswith('.py'):
                        match = re.match(r'def\s+(\w+)\s*\(', stripped)
                    else:
                        match = re.match(r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)', stripped)
                    
                    if match:
                        current_func = match.group(1)
                        all_funcs.add(current_func)
                        continue
                    
                    if current_func:
                        calls = re.findall(r'(\w+)\s*\(', stripped)
                        for c in calls:
                            if c not in ('if', 'for', 'while', 'match', 'Some', 'None', 'Ok', 'Err',
                                        'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict',
                                        'Vec', 'String', 'Box', 'println', 'format', 'vec', 'assert'):
                                all_calls[current_func].add(c)
    
    if commit:
        subprocess.run(['git', 'checkout', '-'], capture_output=True)
        subprocess.run(['git', 'stash', 'pop'], capture_output=True)
    
    os.chdir(cwd)
    return all_funcs, dict(all_calls)


def compute_cycle_metrics(funcs, calls):
    """Compute cycle-related metrics."""
    n = len(funcs)
    if n == 0:
        return {}
    
    total_edges = sum(len(v) for v in calls.values())
    
    # Self-calls
    self_calls = sum(1 for f, cs in calls.items() if f in cs)
    
    # Mutual calls (A calls B and B calls A)
    mutual_pairs = 0
    seen = set()
    for f, cs in calls.items():
        for c in cs:
            if c in calls and f in calls[c] and (c, f) not in seen:
                mutual_pairs += 1
                seen.add((f, c))
    
    # Cycle density
    cycle_density = (self_calls + mutual_pairs) / (total_edges + 1e-10)
    
    # Hub analysis (high in-degree nodes that create cycles)
    in_degree = Counter()
    for f, cs in calls.items():
        for c in cs:
            in_degree[c] += 1
    
    mean_in = np.mean(list(in_degree.values())) if in_degree else 0
    hubs = sum(1 for d in in_degree.values() if d > mean_in + 2 * np.std(list(in_degree.values()))) if len(in_degree) > 1 else 0
    
    return {
        'n_functions': n,
        'total_edges': total_edges,
        'self_calls': self_calls,
        'mutual_pairs': mutual_pairs,
        'cycle_density': cycle_density,
        'hub_count': hubs,
        'edge_density': total_edges / (n * (n - 1) + 1e-10),
    }


def run_experiment():
    print("=" * 70)
    print("CYCLE CONSERVATION LAW — Longitudinal Test")
    print("=" * 70)
    
    repos = {
        'lever-runner': os.path.expanduser('~/repos/lever-runner/src'),
        'zeroclaw-arena': os.path.expanduser('~/repos/zeroclaw-arena'),
    }
    
    timeline = []
    
    for repo_name, repo_path in repos.items():
        print(f"\n=== {repo_name} ===")
        
        git_dir = os.path.dirname(repo_path) if repo_path.endswith('/src') else repo_path
        
        if not os.path.exists(git_dir):
            print(f"  SKIP: {git_dir} not found")
            continue
        
        # Get recent commits
        result = subprocess.run(
            ['git', 'log', '--oneline', '-20'],
            cwd=git_dir,
            capture_output=True, text=True
        )
        
        commits = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(' ', 1)
            if len(parts) == 2:
                commits.append(parts[0])
        
        print(f"  Found {len(commits)} commits")
        
        # Sample commits
        if len(commits) >= 5:
            sampled = [commits[0], commits[len(commits)//4], commits[len(commits)//2], commits[3*len(commits)//4], commits[-1]]
        else:
            sampled = commits
        
        # Test HEAD without checkout
        if os.path.exists(repo_path):
            print(f"\n  Analyzing HEAD...")
            funcs, calls = extract_call_graph(repo_path)
            head_metrics = compute_cycle_metrics(funcs, calls)
            print(f"    Functions: {head_metrics.get('n_functions', 0)}, Edges: {head_metrics.get('total_edges', 0)}")
            print(f"    Self-calls: {head_metrics.get('self_calls', 0)}, Mutual: {head_metrics.get('mutual_pairs', 0)}")
            print(f"    Cycle density: {head_metrics.get('cycle_density', 0):.4f}")
        
        repo_timeline = [{'commit': 'HEAD', **head_metrics}] if os.path.exists(repo_path) else []
        
        for commit in sampled[-3:]:
            print(f"  Analyzing {commit[:8]}...")
            try:
                funcs, calls = extract_call_graph(git_dir, commit)
                metrics = compute_cycle_metrics(funcs, calls)
                repo_timeline.append({'commit': commit[:8], **metrics})
                print(f"    Cycle density: {metrics.get('cycle_density', 0):.4f}")
            except Exception as e:
                print(f"    ERROR: {e}")
        
        # Check conservation
        densities = [t.get('cycle_density', 0) for t in repo_timeline]
        if len(densities) > 1:
            mean = np.mean(densities)
            cv = np.std(densities) / (mean + 1e-10)
            
            print(f"\n  Cycle density over {len(repo_timeline)} commits:")
            print(f"    Mean: {mean:.4f}, CV: {cv:.2f}")
            
            if cv < 0.1:
                print(f"    ✅ CONSERVED: Cycle density is stable (CV < 0.1)")
            elif cv < 0.3:
                print(f"    ⚠️ WEAKLY CONSERVED: Some drift (CV < 0.3)")
            else:
                print(f"    ❌ NOT CONSERVED: High variance (CV = {cv:.2f})")
        
        timeline.extend(repo_timeline)
    
    # Cross-repo comparison
    print("\n" + "=" * 70)
    print("CROSS-REPO CYCLE COMPARISON")
    print("=" * 70)
    
    all_repos = {
        'lever-runner': os.path.expanduser('~/repos/lever-runner/src'),
        'pincherOS': os.path.expanduser('~/repos/pincherOS/pincher-core/src'),
        'open-minded': os.path.expanduser('~/repos/open-minded'),
        'zeroclaw-arena': os.path.expanduser('~/repos/zeroclaw-arena'),
        'fastloop-guard': os.path.expanduser('~/repos/fastloop-guard/src'),
        'metal-lathe': os.path.expanduser('~/repos/metal-lathe'),
    }
    
    repo_metrics = {}
    for name, path in all_repos.items():
        if os.path.exists(path):
            funcs, calls = extract_call_graph(path)
            metrics = compute_cycle_metrics(funcs, calls)
            repo_metrics[name] = metrics
            print(f"  {name}: density={metrics.get('cycle_density', 0):.4f}, "
                  f"mutual={metrics.get('mutual_pairs', 0)}, self={metrics.get('self_calls', 0)}")
    
    # Save
    output = {
        'repo_metrics': repo_metrics,
        'timeline': timeline,
    }
    
    out = os.path.expanduser("~/repos/conservation-spectral-topology-rs/experiments/cycle_conservation_results.json")
    with open(out, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run_experiment()
