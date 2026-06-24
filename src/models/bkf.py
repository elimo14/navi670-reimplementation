"""
bkf.py
------
Learned Backpropagation Kalman Filter (BKF).

From Section 2.2 of the base paper (navi_670):

  "We introduce a learnable Kalman filter that updates its state estimates
   and parameters through the use of backpropagation. This approach
   optimizes both the state and parameters based on incoming data."

  State vector: x_t = [x, y, z]  (3D ECEF position in metres)
  State dim = 3  (Table 3)

  Learnable parameters (Table 3):
    Q : (3×3) process noise covariance    — initial: I × 1e-3
    R : (3×3) measurement noise covariance — initial: I × 1e-2

  The measurement z_t is the 3D correction output from the GNN.
  The BKF refines this correction using Kalman filtering across time.

  Key equations (Section 2.2):
    Prediction:
      x_{t|t-1} = A x_{t-1|t-1}             (state transition, A=I for position)
      P_{t|t-1} = A P_{t-1|t-1} A^T + Q     (covariance prediction)

    Update:
      K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}   (Kalman gain)
      x_{t|t} = x_{t|t-1} + K_t (z_t - H x_{t|t-1})    (state update)
      P_{t|t} = (I - K_t H) P_{t|t-1}                   (covariance update)

  H = I (identity, Section 2.2: "measurement model H is set to identity")
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class LearnedBKF(nn.Module):
    """
    Learned Backpropagation Kalman Filter for 3D GNSS positioning.

    Parameters are differentiable → gradients flow through Kalman equations
    to update both Q, R, and the GNN weights jointly.

    Parameters
    ----------
    state_dim   : int   state dimension (default 3 = [x, y, z], Table 3)
    init_Q_scale: float initial process noise scale (default 1e-3, Table 3)
    init_R_scale: float initial measurement noise scale (default 1e-2, Table 3)
    """

    def __init__(self,
                 state_dim:    int   = 3,
                 init_Q_scale: float = 1e-3,
                 init_R_scale: float = 1e-2):
        super().__init__()

        self.state_dim = state_dim

        # ── learnable noise parameters (Table 3) ──────────────────────────────
        # log-parameterised for numerical stability (always positive)
        # Q: process noise, R: measurement noise
        self.log_Q_diag = nn.Parameter(
            torch.log(torch.ones(state_dim) * init_Q_scale)
        )
        self.log_R_diag = nn.Parameter(
            torch.log(torch.ones(state_dim) * init_R_scale)
        )

        # ── state transition: A = I (constant position model) ─────────────────
        # Not learnable — fixed identity (position doesn't self-propagate)
        self.register_buffer("A", torch.eye(state_dim))

        # ── observation matrix: H = I (Section 2.2) ──────────────────────────
        self.register_buffer("H", torch.eye(state_dim))

        # ── filter state (reset per sequence) ─────────────────────────────────
        self._x = None   # (state_dim,) current state estimate
        self._P = None   # (state_dim, state_dim) current covariance

    @property
    def Q(self) -> torch.Tensor:
        """Process noise covariance matrix Q (learnable, always PSD)."""
        return torch.diag(torch.exp(self.log_Q_diag))

    @property
    def R(self) -> torch.Tensor:
        """Measurement noise covariance matrix R (learnable, always PSD)."""
        return torch.diag(torch.exp(self.log_R_diag))

    def reset(self,
              x0: Optional[torch.Tensor] = None,
              P0: Optional[torch.Tensor] = None):
        """
        Reset filter state at the start of a new sequence.

        Parameters
        ----------
        x0 : (state_dim,) initial state estimate
             if None, initialises to zeros
        P0 : (state_dim, state_dim) initial covariance
             if None, initialises to identity
        """
        device = self.log_Q_diag.device

        if x0 is None:
            self._x = torch.zeros(self.state_dim, device=device)
        else:
            self._x = x0.to(device).float()

        if P0 is None:
            self._P = torch.eye(self.state_dim, device=device)
        else:
            self._P = P0.to(device).float()

    def predict(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Kalman prediction step.

        Equations (2-3) from navi_670:
          x_{t|t-1} = A x_{t-1}
          P_{t|t-1} = A P_{t-1} A^T + Q

        Returns
        -------
        x_pred : (state_dim,) predicted state
        P_pred : (state_dim, state_dim) predicted covariance
        """
        A = self.A
        Q = self.Q

        x_pred = A @ self._x                     # (3,)
        P_pred = A @ self._P @ A.T + Q           # (3, 3)

        return x_pred, P_pred

    def update(self,
               z:      torch.Tensor,
               x_pred: torch.Tensor,
               P_pred: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Kalman update step.

        Equations (4-6) from navi_670:
          K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}
          x_{t|t} = x_{t|t-1} + K_t (z_t - H x_{t|t-1})
          P_{t|t} = (I - K_t H) P_{t|t-1}

        Parameters
        ----------
        z      : (state_dim,) measurement from GNN
        x_pred : (state_dim,) predicted state
        P_pred : (state_dim, state_dim) predicted covariance

        Returns
        -------
        x_upd : (state_dim,) updated state estimate
        P_upd : (state_dim, state_dim) updated covariance
        """
        H = self.H
        R = self.R
        I = torch.eye(self.state_dim, device=z.device)

        # innovation covariance: S = H P H^T + R
        S = H @ P_pred @ H.T + R                 # (3, 3)

        # Kalman gain: K = P H^T S^{-1}
        K = P_pred @ H.T @ torch.inverse(S)      # (3, 3)

        # innovation: y = z - H x_pred
        innovation = z - H @ x_pred              # (3,)

        # state update
        x_upd = x_pred + K @ innovation          # (3,)

        # covariance update: P = (I - KH) P_pred
        P_upd = (I - K @ H) @ P_pred            # (3, 3)

        return x_upd, P_upd

    def step(self, z: torch.Tensor) -> torch.Tensor:
        """
        One full predict-update cycle.

        Parameters
        ----------
        z : (state_dim,) or (1, state_dim) measurement vector from GNN

        Returns
        -------
        x_upd : (state_dim,) updated 3D position estimate
        """
        if self._x is None:
            self.reset()

        z = z.squeeze().float()

        # ── clamp state to prevent divergence ─────────────────────────────────
        if self._x is not None:
            self._x = torch.clamp(self._x, -500.0, 500.0)   # max 500m correction


        # predict
        x_pred, P_pred = self.predict()

        # update
        x_upd, P_upd = self.update(z, x_pred, P_pred)

        # store updated state
        self._x = x_upd
        self._P = P_upd

        return x_upd

    def forward(self,
                measurements: torch.Tensor,
                x0: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Run BKF over a sequence of measurements.

        Parameters
        ----------
        measurements : (T, state_dim) sequence of GNN corrections
        x0           : (state_dim,) optional initial state

        Returns
        -------
        states : (T, state_dim) filtered state estimates
        """
        self.reset(x0)
        states = []

        for t in range(measurements.shape[0]):
            x_upd = self.step(measurements[t])
            states.append(x_upd.unsqueeze(0))

        return torch.cat(states, dim=0)   # (T, 3)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    print("Testing LearnedBKF (Table 3 parameters)...")
    bkf = LearnedBKF(state_dim=3, init_Q_scale=1e-3, init_R_scale=1e-2)

    print(f"\nLearnable parameters: {bkf.count_parameters()}")
    print(f"  log_Q_diag shape: {bkf.log_Q_diag.shape}")
    print(f"  log_R_diag shape: {bkf.log_R_diag.shape}")
    print(f"\nInitial Q (process noise):\n{bkf.Q.detach().numpy()}")
    print(f"\nInitial R (measurement noise):\n{bkf.R.detach().numpy()}")

    # ── simulate sequence ─────────────────────────────────────────────────────
    T = 10
    true_pos   = torch.tensor([100.0, 200.0, -150.0])  # true correction
    noisy_meas = true_pos.unsqueeze(0).repeat(T, 1) \
                 + torch.randn(T, 3) * 20.0             # noisy GNN output

    print(f"\nSimulated {T} steps:")
    print(f"  True correction  : {true_pos.numpy()}")
    print(f"  Noisy meas (mean): {noisy_meas.mean(0).numpy().round(2)}")

    bkf.reset()
    states = bkf(noisy_meas)

    print(f"\nBKF filtered states (T, 3):\n{states.detach().numpy().round(2)}")
    print(f"  Final estimate: {states[-1].detach().numpy().round(2)}")
    print(f"  True value    : {true_pos.numpy()}")

    # ── test backpropagation ──────────────────────────────────────────────────
    print(f"\nBackpropagation test:")
    bkf.reset()
    states = bkf(noisy_meas)
    target = true_pos.unsqueeze(0).repeat(T, 1)
    loss   = ((states - target) ** 2).mean()
    loss.backward()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  log_Q_diag grad: {bkf.log_Q_diag.grad.numpy().round(6)}")
    print(f"  log_R_diag grad: {bkf.log_R_diag.grad.numpy().round(6)}")
    print(f"  ✅ Gradients flow through Kalman equations")

    print(f"\n✅ BKF ready for tight coupling with GNN")
