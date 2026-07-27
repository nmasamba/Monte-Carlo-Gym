# Analysis scaffold

Analysis code should consume `runs.jsonl` and never import a live planner.
Planned outputs include:

- paired bootstrap tables;
- success-cost-risk Pareto frontiers;
- hypervolume and budget-performance curves;
- calibration and discrepancy plots;
- route and stopping diagnostics;
- naive, IPS, self-normalized IPS, and doubly robust comparisons;
- failure and safety taxonomies.

Each figure/table script must record its input run identifiers and analysis
configuration.
