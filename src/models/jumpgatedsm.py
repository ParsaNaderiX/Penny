"""JumpGate-DSM: the JumpGate machinery on a *generalized-score* objective.

This is the **score-based** sibling of :class:`~models.jumpgateunet.JumpGateUNet`.
The two share a byte-for-byte identical U-Net architecture — the network always
emits a single ``(B, 1, T, F)`` tensor — so the *only* differences live in the
training objective and in how that output tensor is interpreted:

* ``JumpGateUNet`` (``crypto.train_jumpgate_unet``) regresses the **noise** ``eps``
  with a plain MSE (``L_diff = ||eps_hat - eps||^2``).
* ``JumpGateDSM`` (``crypto.train_jumpgate_dsm``) regresses the **generalized score**
  ``grad_{x_t} log q(x_t|x_0)`` of the Lévy jump-diffusion kernel with the weighted
  denoising-score-matching MSE — exactly as :class:`~models.jointdifflevy.JointDiffusionLevy`
  does, but keeping JumpGate's ``g_phi`` noise-state estimator, W-conditioning,
  gated experts and gated trend head.

Because the architecture is unchanged, this class is a thin subclass of
``JumpGateUNet``: it inherits ``forward`` (whose first return value is now read as
the *score* ``s_hat`` rather than ``eps_hat``), the encoder, the trend head and the
feature-only :meth:`predict`.  It only overrides the eps/score helpers so the
semantics are correct.

Naming note: do **not** confuse this with :class:`~models.jumpgatescore.JumpGateScoreGrad`,
which despite its name is an *eps*-prediction model (the "ScoreGrad" there refers to
its GRU+WaveNet *backbone*, not the objective).  ``JumpGateDSM`` is the one that
actually trains on the score.
"""

from __future__ import annotations

import torch

from models.jumpgateunet import (
    JumpGateUNet,
    NoiseStateEstimator as NoiseStateEstimator,  # re-export
    count_parameters as count_parameters,  # re-export
)


class JumpGateDSM(JumpGateUNet):
    """W-aware, jump-gated **score-prediction** U-Net + trend head.

    Identical architecture to :class:`JumpGateUNet`; ``forward`` returns
    ``(s_hat (B,1,T,F), logits (B,3), logW_hat (B,), pi_logit (B,))`` where the first
    element is the predicted generalized score.  Trained by ``crypto.train_jumpgate_dsm``.
    """

    family = "joint_diffusion"

    @staticmethod
    def recover_score(s_hat: torch.Tensor, W_hat: torch.Tensor) -> torch.Tensor:
        """The network already predicts the score, so recovery is the identity.

        (Overrides ``JumpGateUNet.recover_score``, which converts ``eps -> score``
        via ``-eps/W``.)  ``W_hat`` is accepted for a uniform call signature with the
        eps model but is unused.
        """
        return s_hat
