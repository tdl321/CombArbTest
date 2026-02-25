"""Fix is_partition_constraint function."""
import re

with open("src/backtest/constraint_checker.py", "r") as f:
    content = f.read()

# The pattern to find - first part of is_partition_constraint
old_pattern = r"""def is_partition_constraint\(cluster\) -> bool:
    \"\"\"Check if cluster represents a partition \(exhaustive \+ all pairwise exclusive\)\.
    
    A partition requires:
    1\. At least 3 markets \(for combinatorial arbitrage\)
    2\. At least one exhaustive relationship
    3\. Most pairs being mutually exclusive
    
    Args:
        cluster: MarketCluster from src\.llm\.schema or src\.optimizer\.schema
        
    Returns:
        True if the cluster represents a valid partition constraint
    \"\"\"
    # Require 3\+ markets for combinatorial arbitrage
    if len\(cluster\.market_ids\) < 3:
        return False
    
    if not hasattr\(cluster, 'relationships'\) or not cluster\.relationships:
        return False
    
    # Check for exhaustive constraint
    has_exhaustive = any\(r\.type == "exhaustive" for r in cluster\.relationships\)
    
    if not has_exhaustive:
        return False"""

new_code = """def is_partition_constraint(cluster) -> bool:
    \"\"\"Check if cluster represents a partition (exhaustive + all pairwise exclusive).
    
    A partition requires:
    1. At least 3 markets (for combinatorial arbitrage)
    2. Either is_partition=True flag, OR
    3. At least one exhaustive relationship with mutual exclusivity
    
    Args:
        cluster: MarketCluster from src.llm.schema or src.optimizer.schema
        
    Returns:
        True if the cluster represents a valid partition constraint
    \"\"\"
    # Require 3+ markets for combinatorial arbitrage
    if len(cluster.market_ids) < 3:
        return False
    
    # Fast path: check is_partition flag first
    if hasattr(cluster, "is_partition") and cluster.is_partition:
        logger.debug("[PARTITION] Cluster %s: is_partition=True flag set", cluster.cluster_id)
        return True
    
    # Fallback: check for exhaustive relationship in relationships list
    if not hasattr(cluster, 'relationships') or not cluster.relationships:
        return False
    
    # Check for exhaustive constraint
    has_exhaustive = any(r.type == "exhaustive" for r in cluster.relationships)
    
    if not has_exhaustive:
        return False"""

if re.search(old_pattern, content):
    content = re.sub(old_pattern, new_code, content)
    with open("src/backtest/constraint_checker.py", "w") as f:
        f.write(content)
    print("SUCCESS: Patched is_partition_constraint")
else:
    print("Pattern not found, trying simpler approach...")
    # Simpler approach - just add the is_partition check after the market_ids check
    simple_old = """    # Require 3+ markets for combinatorial arbitrage
    if len(cluster.market_ids) < 3:
        return False
    
    if not hasattr(cluster, 'relationships') or not cluster.relationships:
        return False"""
    
    simple_new = """    # Require 3+ markets for combinatorial arbitrage
    if len(cluster.market_ids) < 3:
        return False
    
    # Fast path: check is_partition flag first
    if hasattr(cluster, "is_partition") and cluster.is_partition:
        logger.debug("[PARTITION] Cluster %s: is_partition=True flag set", cluster.cluster_id)
        return True
    
    if not hasattr(cluster, 'relationships') or not cluster.relationships:
        return False"""
    
    if simple_old in content:
        content = content.replace(simple_old, simple_new)
        with open("src/backtest/constraint_checker.py", "w") as f:
            f.write(content)
        print("SUCCESS: Patched with simple approach")
    else:
        print("ERROR: Could not patch file")
