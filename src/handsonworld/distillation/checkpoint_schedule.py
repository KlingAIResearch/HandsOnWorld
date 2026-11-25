"""Pure helpers for selecting future training checkpoint milestones."""

from numbers import Integral


def checkpoint_steps(initial, total, every, save_steps=None):
    """Return sorted checkpoint steps after ``initial`` through ``total``.

    With no explicit ``save_steps``, checkpoints follow ``every`` and include
    the final step.  An explicit sequence replaces the interval schedule; its
    positive milestones above ``total`` are ignored so the same configuration
    can be reused for shorter runs.  The final step is always included when it
    is still in the future.
    """
    if save_steps is None:
        if not isinstance(every, Integral) or isinstance(every, bool) or every <= 0:
            raise ValueError("save interval must be a positive integer")
        selected = set(range((initial // every + 1) * every, total + 1, every))
    else:
        selected = set()
        for step in save_steps:
            if not isinstance(step, Integral) or isinstance(step, bool) or step <= 0:
                raise ValueError("checkpoint milestones must be positive integers")
            if step <= total:
                selected.add(int(step))
    if total > initial:
        selected.add(total)
    return sorted(step for step in selected if step > initial)
