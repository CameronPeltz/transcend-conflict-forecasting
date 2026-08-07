"""Resume helper: sections 1 (edge ablation) and 2 (propagation mode)
already completed and logged (17 rows in hg_iterations_log.jsonl) before
the prior process was killed by the tool timeout. This just runs the
remaining two sections (3: xgi structural features, 4: text hypergraph),
appending to the same log."""
import run_iterations as ri

ri.section_xgi_structural()
ri.section_text_hypergraph()
print("\nremaining sections complete.")
