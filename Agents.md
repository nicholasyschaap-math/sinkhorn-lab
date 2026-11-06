# Project

Entropic optimal transport with Sinkhorn: vanilla and log-domain solvers.

- Do not overstep bounds on the scope of any assigned tasks.

# Commands

- Test: `pytest -q`
- Lint: `ruff check .`

# Rules

- Do not edit src/sinkhorn_lab/solvers.py unless explicitly asked. Its
  author primarily writes it by hand. If asked to make edits you may not
  commit any to main without author approval. You must be extra rigorous
  in your reports on edits made here if asked.
- Never weaken, skip or delete a test to make it pass. Report the failure.
- Do not add dependencies without asking.
- Show the commands you ran and their output.

# Conventions

- Notation generally follows Peyre and Cuturi's Computational Optimal
  Transport (https://arxiv.org/pdf/1803.00567) unless otherwise
  contradicted by author documentation.
- Treat docs/derivation.md as the formula source, changes in approach to
  some notation, methods, etc. will create conflicts if you attempt to
  pull directly from the text instead.

# Review guidelines

- Report only problems that affect correctness or the stated
  requirements.
- It is better to stop and seek clarification than commit incorrect
  generations. If you perceive a strong potential issue, flag it rather
  than approaching it without approval.
