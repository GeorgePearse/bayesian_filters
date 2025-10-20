# Quick Start Guide

This guide will walk you through creating your first Kalman filter using Bayesian Filters.

## Basic Kalman Filter Example

Let's create a simple Kalman filter to track a 1D position with constant velocity.

### Import Required Modules

```python
import numpy as np
from bayesian_filters.kalman import KalmanFilter
from bayesian_filters.common import Q_discrete_white_noise
```

### Create the Filter

```python
# Create a Kalman filter with 2 state variables (position, velocity)
# and 1 measurement variable (position)
kf = KalmanFilter(dim_x=2, dim_z=1)
```

### Initialize State Vector

```python
# Initial state: [position, velocity]
kf.x = np.array([[2.],   # initial position
                 [0.]])  # initial velocity
```

### Define State Transition Matrix

```python
# State transition matrix (constant velocity model)
dt = 0.1  # time step
kf.F = np.array([[1., dt],
                 [0., 1.]])
```

### Define Measurement Function

```python
# Measurement function (we only measure position)
kf.H = np.array([[1., 0.]])
```

### Set Covariance Matrices

```python
# Initial covariance matrix (uncertainty in initial state)
kf.P *= 1000.

# Measurement uncertainty
kf.R = 5.

# Process noise
kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)
```

### Run the Filter

```python
# Simulate measurements (in practice, these come from sensors)
measurements = [2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3]

for z in measurements:
    # Predict step
    kf.predict()

    # Update step
    kf.update(z)

    # Get the current state estimate
    print(f"Position: {kf.x[0, 0]:.2f}, Velocity: {kf.x[1, 0]:.2f}")
```

## Understanding the Filter Cycle

The Kalman filter operates in a two-step cycle:

### 1. Predict Step

```python
kf.predict()
```

This step uses the state transition model to predict the next state:

- Projects the state forward: `x = F @ x`
- Projects the covariance forward: `P = F @ P @ F.T + Q`

### 2. Update Step

```python
kf.update(measurement)
```

This step incorporates a new measurement:

- Computes the Kalman gain
- Updates the state estimate
- Updates the covariance estimate

## Complete Working Example

Here's a complete example with plotting:

```python
import numpy as np
import matplotlib.pyplot as plt
from bayesian_filters.kalman import KalmanFilter
from bayesian_filters.common import Q_discrete_white_noise

# Create filter
kf = KalmanFilter(dim_x=2, dim_z=1)

# Initialize
kf.x = np.array([[0.], [1.]])  # start at position 0, velocity 1
kf.F = np.array([[1., 1.], [0., 1.]])  # dt = 1
kf.H = np.array([[1., 0.]])
kf.P *= 1000.
kf.R = 5.
kf.Q = Q_discrete_white_noise(dim=2, dt=1., var=0.1)

# Generate noisy measurements
true_positions = np.arange(0, 50, 1)
measurements = true_positions + np.random.normal(0, 2, len(true_positions))

# Run filter
estimates = []
for z in measurements:
    kf.predict()
    kf.update(z)
    estimates.append(kf.x[0, 0])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(true_positions, label='True Position', linewidth=2)
plt.scatter(range(len(measurements)), measurements,
            label='Measurements', alpha=0.5, s=30)
plt.plot(estimates, label='Kalman Filter Estimate', linewidth=2)
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Kalman Filter Example')
plt.grid(True)
plt.show()
```

## Next Steps

Now that you understand the basics, explore:

- **[Kalman Filter](../filters/kalman-filter.md)**: Detailed documentation on the standard Kalman filter
- **[Extended Kalman Filter](../filters/extended-kalman-filter.md)**: For nonlinear systems
- **[Unscented Kalman Filter](../filters/unscented-kalman-filter.md)**: Alternative approach for nonlinear systems
- **[Examples](../examples.md)**: More complex examples and use cases

## Common Patterns

### Batch Processing

```python
# Process multiple measurements at once
for z in measurements:
    kf.predict()
    kf.update(z)
```

### Accessing Filter State

```python
# Current state estimate
position = kf.x[0, 0]
velocity = kf.x[1, 0]

# Current covariance (uncertainty)
uncertainty = kf.P

# Innovation (measurement residual)
residual = kf.y
```

### Saving Filter History

```python
from bayesian_filters.kalman import Saver

# Create a saver to log filter history
saver = Saver(kf)

for z in measurements:
    kf.predict()
    kf.update(z)
    saver.save()

# Access saved data
saver.x  # All state estimates
saver.P  # All covariances
```
