"""Training-only background-depth curriculum, indexed by committed updates.

The frozen predictor supplies P everywhere. Only valid nontransparent pixels
may interpolate toward G in training and explicitly label-conditioned matched
validation. Pure-P selection validation and inference use P directly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers


@dataclass(frozen=True)
class CurriculumSchedule:
    """Use the optimizer-update count BEFORE the next optimizer step.

    A resumed checkpoint's committed ``updates`` is sufficient to reproduce
    the next alpha. Epochs are complete passes, not separate stage-local runs.
    The transition includes both endpoints: its first update uses alpha=1 and
    its last update uses alpha=0.
    """
    steps_per_epoch: int = 1421
    stage_epochs: tuple = (20, 30, 30)

    def __post_init__(self):
        if (isinstance(self.steps_per_epoch, bool)
                or not isinstance(self.steps_per_epoch, numbers.Integral)
                or self.steps_per_epoch <= 0):
            raise ValueError("steps_per_epoch must be a positive integer")
        stages = tuple(self.stage_epochs)
        if (len(stages) != 3 or any(isinstance(x, bool)
                or not isinstance(x, numbers.Integral) or x <= 0 for x in stages)):
            raise ValueError("stage_epochs must contain three positive integers")
        if stages[1] * self.steps_per_epoch < 2:
            raise ValueError("transition requires at least two updates for both endpoints")
        object.__setattr__(self, "stage_epochs", stages)

    @property
    def stage_steps(self):
        return tuple(int(x * self.steps_per_epoch) for x in self.stage_epochs)

    @property
    def total_steps(self):
        return sum(self.stage_steps)

    def _validate_updates(self, completed_updates):
        if (isinstance(completed_updates, bool)
                or not isinstance(completed_updates, numbers.Integral)
                or completed_updates < 0):
            raise ValueError("completed_updates must be a nonnegative integer")
        return int(completed_updates)

    def phase(self, completed_updates):
        step = self._validate_updates(completed_updates)
        oracle, transition, _ = self.stage_steps
        if step < oracle:
            return "oracle_background"
        if step < oracle + transition:
            return "linear_transition"
        return "predicted_depth"

    def alpha(self, completed_updates):
        step = self._validate_updates(completed_updates)
        oracle, transition, _ = self.stage_steps
        if step < oracle:
            return 1.0
        if step < oracle + transition:
            return 1.0 - (step - oracle) / float(transition - 1)
        return 0.0

    def to_dict(self):
        return {"schema_version": "v2_lidf_spvnas_background_curriculum_v1",
                "steps_per_epoch": int(self.steps_per_epoch),
                "stage_epochs": list(self.stage_epochs),
                "stage_steps": list(self.stage_steps),
                "total_steps": self.total_steps,
                "index": "committed_optimizer_updates_before_next_step",
                "transition_endpoints": "inclusive_alpha_1_to_0"}


def mix_background_depth(predicted, target=None, valid=None, transparent=None,
                         alpha=0.0):
    """Return P + alpha*(G-P) on valid opaque pixels, P everywhere else.

    Alpha zero returns the exact P object before touching any label argument.
    For nonzero alpha only eligible target entries are gathered, so even NaN
    or arbitrarily changed transparent target depths cannot leak into input.
    The alpha-one endpoint copies G exactly, avoiding subtraction roundoff.
    Shapes are matching torch tensors (HW, CHW, or BCHW), masks boolean.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
        raise ValueError("alpha must be a finite real number in [0, 1]")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be a finite real number in [0, 1]")
    if alpha == 0.0:
        return predicted

    import torch
    if not isinstance(predicted, torch.Tensor) or not predicted.is_floating_point():
        raise TypeError("predicted depth must be a floating torch tensor")
    for name, value in (("target", target), ("valid", valid), ("transparent", transparent)):
        if not isinstance(value, torch.Tensor):
            raise TypeError("%s must be a torch tensor for nonzero alpha" % name)
        if value.shape != predicted.shape or value.device != predicted.device:
            raise ValueError("%s shape/device differs from predicted depth" % name)
    if not target.is_floating_point():
        raise TypeError("target depth must be floating")
    if valid.dtype != torch.bool or transparent.dtype != torch.bool:
        raise TypeError("valid and transparent masks must be boolean")
    eligible = valid & ~transparent & torch.isfinite(target) & (target > 0)
    result = predicted.clone()
    gt = target[eligible].to(dtype=predicted.dtype)
    if alpha == 1.0:
        result[eligible] = gt
    else:
        base = predicted[eligible]
        result[eligible] = base + alpha * (gt - base)
    return result


@dataclass(frozen=True)
class ContinuationCurriculumSchedule:
    """Keep the committed parent trajectory and continue its next alpha.

    Real optimizer-update counters never change. Only the remaining transition
    is resampled onto the three-rank full-epoch update grid, including its saved
    starting alpha and the final alpha-zero update of epoch 50.
    """
    parent_steps_per_epoch: int
    steps_per_epoch: int
    parent_completed_epochs: int
    parent_committed_updates: int
    anchor_next_alpha: float
    stage_epochs: tuple = (20, 30, 30)

    def __post_init__(self):
        old = CurriculumSchedule(self.parent_steps_per_epoch, self.stage_epochs)
        CurriculumSchedule(self.steps_per_epoch, self.stage_epochs)
        def require(value, message):
            if not value:
                raise ValueError(message)
        require(self.stage_epochs[0] <= self.parent_completed_epochs < sum(self.stage_epochs[:2]),
                'Continuation anchor must be within the transition epoch interval')
        require(self.parent_committed_updates == self.parent_completed_epochs * self.parent_steps_per_epoch,
                'Continuation anchor does not represent complete parent epochs')
        require(self.anchor_next_alpha == old.alpha(self.parent_committed_updates),
                'Continuation alpha differs from the exact parent next-step value')

    @property
    def remaining_transition_updates(self):
        return (sum(self.stage_epochs[:2]) - self.parent_completed_epochs) * self.steps_per_epoch

    @property
    def transition_end_updates(self):
        return self.parent_committed_updates + self.remaining_transition_updates

    @property
    def total_steps(self):
        return self.parent_committed_updates + (sum(self.stage_epochs) - self.parent_completed_epochs) * self.steps_per_epoch

    def phase(self, completed_updates):
        step = CurriculumSchedule._validate_updates(self, completed_updates)
        if step < self.stage_epochs[0] * self.parent_steps_per_epoch:
            return 'oracle_background'
        return 'linear_transition' if step < self.transition_end_updates else 'predicted_depth'

    def alpha(self, completed_updates):
        step = CurriculumSchedule._validate_updates(self, completed_updates)
        if step < self.parent_committed_updates:
            return CurriculumSchedule(self.parent_steps_per_epoch, self.stage_epochs).alpha(step)
        if step >= self.transition_end_updates:
            return 0.0
        remaining_step = step - self.parent_committed_updates
        return self.anchor_next_alpha * (1.0 - remaining_step / float(self.remaining_transition_updates - 1))

    def to_dict(self):
        return {'schema_version': 'v2_lidf_gpu567_continuation_curriculum_v1',
                'stage_epochs': list(self.stage_epochs), 'parent_steps_per_epoch': self.parent_steps_per_epoch,
                'steps_per_epoch': self.steps_per_epoch, 'parent_completed_epochs': self.parent_completed_epochs,
                'parent_committed_updates': self.parent_committed_updates,
                'anchor_next_alpha': self.anchor_next_alpha,
                'remaining_transition_updates': self.remaining_transition_updates,
                'transition_end_updates': self.transition_end_updates, 'total_steps': self.total_steps,
                'index': 'actual_committed_optimizer_updates_without_rewriting_parent_history',
                'transition_endpoints': 'saved_parent_next_alpha_to_zero_inclusive_over_remaining_full_transition_epochs'}


def schedule_for_config(config):
    migration = config.get('migration')
    if migration is None:
        return CurriculumSchedule(config['curriculum']['steps_per_epoch'], tuple(config['curriculum']['stage_epochs']))
    return ContinuationCurriculumSchedule(migration['parent_steps_per_epoch'], migration['steps_per_epoch'],
        migration['parent_completed_epochs'], migration['parent_committed_updates'], migration['anchor_next_alpha'],
        tuple(config['curriculum']['stage_epochs']))
