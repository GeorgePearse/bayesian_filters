"""Common pytest fixtures for Bayesian filter testing."""

import numpy as np
from bayesian_filters.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from bayesian_filters.kalman import SigmaPointsBase, MerweScaledSigmaPoints
from bayesian_filters.common import Q_discrete_white_noise


class SimpleSensorSim:
    """Simple 1D sensor simulator."""

    def __init__(self, pos=0.0, vel=1.0, dt=0.1, noise_std=1.0):
        """Initialize sensor simulator.

        Parameters
        ----------
        pos : float
            Initial position
        vel : float
            Velocity (units per time step)
        dt : float
            Time step
        noise_std : float
            Measurement noise standard deviation
        """
        self.pos = pos
        self.vel = vel
        self.dt = dt
        self.noise_std = noise_std

    def read(self):
        """Read next measurement with noise."""
        self.pos += self.vel * self.dt
        return self.pos + np.random.normal(0, self.noise_std)

    def true_position(self):
        """Return current true position."""
        return self.pos


def kf_1d():
    """Create a 1D constant-velocity Kalman filter fixture.

    Returns
    -------
    KalmanFilter
        1D Kalman filter with state [position, velocity]
    """
    dt = 0.1
    kf = KalmanFilter(dim_x=2, dim_z=1)

    # State transition matrix (constant velocity model)
    kf.F = np.array([[1.0, dt], [0.0, 1.0]])

    # Measurement matrix (measure position only)
    kf.H = np.array([[1.0, 0.0]])

    # Initial covariance
    kf.P = np.eye(2) * 100.0

    # Measurement noise
    kf.R = np.array([[1.0]])

    # Process noise
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)

    # Initial state
    kf.x = np.array([[0.0], [1.0]])

    return kf


def kf_2d():
    """Create a 2D constant-velocity Kalman filter fixture.

    Returns
    -------
    KalmanFilter
        2D Kalman filter with state [x, vx, y, vy]
    """
    dt = 0.1
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State transition matrix
    F = np.eye(4)
    F[0, 1] = dt
    F[2, 3] = dt
    kf.F = F

    # Measurement matrix (measure both positions)
    kf.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    # Initial covariance
    kf.P = np.eye(4) * 100.0

    # Measurement noise
    kf.R = np.eye(2)

    # Process noise
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)
    kf.Q = np.zeros((4, 4))
    kf.Q[:2, :2] = q
    kf.Q[2:, 2:] = q

    # Initial state
    kf.x = np.array([[0.0], [0.0], [0.0], [0.0]])

    return kf


def ekf_2d():
    """Create a 2D Extended Kalman filter fixture.

    Returns
    -------
    ExtendedKalmanFilter
        2D EKF for nonlinear measurement function
    """
    dt = 0.1
    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)

    # State transition matrix
    ekf.F = np.array([[1.0, dt], [0.0, 1.0]])

    # State transition function
    def fx(x, dt):
        return np.dot(ekf.F, x)

    # Measurement function (range to origin)
    def hx(x):
        return np.sqrt(x[0] ** 2 + x[1] ** 2)

    # Jacobian of measurement function
    def H(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        if r < 1e-10:
            r = 1e-10
        return np.array([[x[0] / r, x[1] / r]])

    ekf.fx = fx
    ekf.hx = hx
    ekf.H = H

    # Covariances
    ekf.P = np.eye(2) * 100.0
    ekf.R = np.array([[1.0]])
    ekf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)

    # Initial state
    ekf.x = np.array([10.0, 0.0])

    return ekf


def ukf_3d():
    """Create a 3D Unscented Kalman filter fixture.

    Returns
    -------
    UnscentedKalmanFilter
        3D UKF with standard configuration
    """
    dt = 0.1

    # Create sigma points
    sigmas = MerweScaledSigmaPoints(n=3, alpha=0.1, beta=2.0, kappa=0.0)

    ukf = UnscentedKalmanFilter(
        dim_x=3,
        dim_z=3,
        dt=dt,
        hx=lambda x: x,  # Identity measurement function
        fx=lambda x, dt: x,  # Identity process function
        points=sigmas,
    )

    # Covariances
    ukf.P = np.eye(3) * 100.0
    ukf.R = np.eye(3)
    ukf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=0.1)

    # Initial state
    ukf.x = np.array([0.0, 0.0, 0.0])

    return ukf


def sensor_sim(pos=0.0, vel=1.0, dt=0.1, noise_std=1.0):
    """Create a simple sensor simulator fixture.

    Parameters
    ----------
    pos : float
        Initial position
    vel : float
        Velocity
    dt : float
        Time step
    noise_std : float
        Measurement noise standard deviation

    Returns
    -------
    SimpleSensorSim
        Configured sensor simulator
    """
    return SimpleSensorSim(pos=pos, vel=vel, dt=dt, noise_std=noise_std)


def noisy_measurements(ground_truth, noise_std, seed=42):
    """Generate noisy measurements from ground truth.

    Parameters
    ----------
    ground_truth : array_like
        True values
    noise_std : float
        Standard deviation of measurement noise
    seed : int
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Measurements with added Gaussian noise
    """
    np.random.seed(seed)
    ground_truth = np.asarray(ground_truth)
    noise = np.random.normal(0, noise_std, size=ground_truth.shape)
    return ground_truth + noise
