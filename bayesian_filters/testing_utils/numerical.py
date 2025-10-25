"""Analytical solutions and reference implementations for testing."""

import numpy as np


class ConstantVelocitySolution:
    """Analytical solution for constant velocity filter.

    For a 1D system with constant velocity:
    - State: [position, velocity]
    - Process model: x(k+1) = x(k) + v(k)*dt
    - Measurement: position only

    With zero process noise and zero initial velocity error,
    the filter should track perfectly.
    """

    @staticmethod
    def position_at_time(t, x0=0.0, v0=1.0):
        """Analytical position at time t.

        Parameters
        ----------
        t : float
            Time
        x0 : float
            Initial position
        v0 : float
            Constant velocity

        Returns
        -------
        float
            Position at time t
        """
        return x0 + v0 * t

    @staticmethod
    def steady_state_error(R, Q, dt, var_init=None):
        """Calculate steady-state estimation error.

        Parameters
        ----------
        R : float
            Measurement noise variance
        Q : float
            Process noise variance per dt
        dt : float
            Time step
        var_init : float, optional
            Initial position variance

        Returns
        -------
        float
            Steady-state position error variance
        """
        # Simplified steady-state for constant velocity model
        # With proper tuning, error should converge to sqrt(Q*R)
        return np.sqrt(Q * R / dt)


class ConstantAccelerationSolution:
    """Analytical solution for constant acceleration filter.

    For a 1D system with constant acceleration:
    - State: [position, velocity, acceleration]
    - Constant acceleration assumption
    """

    @staticmethod
    def position_at_time(t, x0=0.0, v0=0.0, a=1.0):
        """Analytical position at time t.

        Parameters
        ----------
        t : float
            Time
        x0 : float
            Initial position
        v0 : float
            Initial velocity
        a : float
            Constant acceleration

        Returns
        -------
        float
            Position at time t
        """
        return x0 + v0 * t + 0.5 * a * t**2

    @staticmethod
    def velocity_at_time(t, v0=0.0, a=1.0):
        """Analytical velocity at time t.

        Parameters
        ----------
        t : float
            Time
        v0 : float
            Initial velocity
        a : float
            Constant acceleration

        Returns
        -------
        float
            Velocity at time t
        """
        return v0 + a * t


class LinearSystemSolution:
    """Analytical solution for linear systems.

    Computes the optimal state estimate for a linear system with
    known process and measurement models.
    """

    @staticmethod
    def kalman_gain(P, H, R):
        """Compute Kalman gain.

        Parameters
        ----------
        P : np.ndarray
            Prediction covariance (n x n)
        H : np.ndarray
            Measurement matrix (m x n)
        R : np.ndarray
            Measurement noise covariance (m x m)

        Returns
        -------
        np.ndarray
            Kalman gain (n x m)
        """
        S = np.dot(H, np.dot(P, H.T)) + R  # Innovation covariance
        K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))
        return K

    @staticmethod
    def innovation_covariance(P, H, R):
        """Compute innovation (measurement residual) covariance.

        Parameters
        ----------
        P : np.ndarray
            Prediction covariance (n x n)
        H : np.ndarray
            Measurement matrix (m x n)
        R : np.ndarray
            Measurement noise covariance (m x m)

        Returns
        -------
        np.ndarray
            Innovation covariance (m x m)
        """
        return np.dot(H, np.dot(P, H.T)) + R

    @staticmethod
    def updated_covariance(K, H, P):
        """Compute updated state covariance after Kalman update.

        Parameters
        ----------
        K : np.ndarray
            Kalman gain (n x m)
        H : np.ndarray
            Measurement matrix (m x n)
        P : np.ndarray
            Prediction covariance (n x n)

        Returns
        -------
        np.ndarray
            Updated covariance (n x n)
        """
        return (np.eye(P.shape[0]) - np.dot(K, H)) @ P


class WhiteNoiseAcceleration:
    """White noise acceleration process model.

    Reference: Bar-Shalom et al., "Estimation with Applications
    to Tracking and Navigation"
    """

    @staticmethod
    def discrete_process_noise(var, dt, order=2):
        """Compute discrete process noise covariance.

        For continuous white noise acceleration model with
        discrete sampling at interval dt.

        Parameters
        ----------
        var : float
            Acceleration variance (continuous)
        dt : float
            Sampling interval
        order : int, default=2
            Model order (2=const velocity, 3=const acceleration)

        Returns
        -------
        np.ndarray
            Process noise covariance matrix
        """
        if order == 2:
            # Constant velocity model
            return var * np.array(
                [
                    [dt**3 / 3, dt**2 / 2],
                    [dt**2 / 2, dt],
                ]
            )
        elif order == 3:
            # Constant acceleration model
            return var * np.array(
                [
                    [dt**5 / 20, dt**4 / 8, dt**3 / 6],
                    [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                    [dt**3 / 6, dt**2 / 2, dt],
                ]
            )
        else:
            raise ValueError(f"Unsupported order: {order}")
