from typing import Dict, List, Any
from io import TextIOWrapper


def _generate_hierarchy(top_category: str, parent_to_child: Dict[str, str], leaves: List[str]) -> Dict[str, Any]:
    hierarchy = {top_category: {}}
    if top_category in leaves:
        return {top_category: None}
    children = [v for k, v in parent_to_child if k == top_category]
    for child in children:
        hierarchy[top_category].update(_generate_hierarchy(child, parent_to_child, leaves))
    return hierarchy


def _get_hierarchies(child_to_parent: Dict[str, str]) -> Dict[str, Any]:
    parent_to_child = [(v, k) for k, v in child_to_parent.items() if v]
    parents = [parent for parent, child in parent_to_child]
    leaves = [child for parent, child in parent_to_child if child not in parents]
    top_categories = [k for k, v in child_to_parent.items() if not v]
    hierarchies = dict()
    for top_category in top_categories:
        hierarchies.update(_generate_hierarchy(top_category, parent_to_child, leaves))
    return hierarchies


def _write_in_csv(hierarchies: Dict[str, Any], depth: int, output: TextIOWrapper):
    if hierarchies:
        prefix = "," * depth
        for k in sorted(hierarchies.keys()):
            res = prefix + k
            output.write(res + "\n")
            _write_in_csv(hierarchies[k], depth + 1, output)


def get_hierarchies_in_csv(child_to_parent: Dict[str, str], csv_output_filepath: str):
    hierarchies = _get_hierarchies(child_to_parent)
    with open(csv_output_filepath, "w") as output:
        _write_in_csv(hierarchies, 0, output)