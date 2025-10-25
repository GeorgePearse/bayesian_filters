# Research and Academic Papers

This page provides a curated collection of influential research papers on Kalman and Bayesian filtering, from foundational works to modern advancements.

## Foundational and Tutorial Works

### An Elementary Introduction to Kalman Filtering
**Authors:** Yan Pei et al.
**Institution:** University of Texas at Austin
**Year:** 2017

Offers a clear conceptual and mathematical introduction to Kalman filtering with examples in robotics and control systems. Excellent starting point for newcomers to the field.

### Bayesian Filtering: From Kalman Filters to Particle Filters, and Beyond
**Author:** Zhe Chen
**Year:** 2003

A highly cited tutorial and survey paper that traces the evolution from Kalman filters to modern Bayesian and particle filtering methods. Provides comprehensive historical context and theoretical foundations.

### A Study about Kalman Filters Applied to Embedded Sensors
**Authors:** Valade et al.
**Year:** 2017

Explains how standard and extended Kalman filters can be applied effectively in embedded and constrained environments like drones and smartphones. Practical focus on real-world implementation challenges.

## Hybrid and Modernized Approaches

### A Review of Kalman Filter with Artificial Intelligence Techniques
**Author:** Kim
**Institution:** Cranfield University

Reviews methods integrating Kalman filters with neural networks, providing a taxonomy of AI-augmented Kalman filter approaches. Covers emerging trends in learning-based filtering.

### A Hybrid Bayesian Kalman Filter and Applications to Numerical Models
**Authors:** Galanis et al.
**Year:** 2017

Presents a hybrid systems approach combining nonlinear Kalman filters and Bayesian models for robust prediction tasks. Applications in meteorology and environmental modeling.

### Developments of Inverse Analysis by Kalman Filters and Bayesian Filtering
**Author:** Murakami
**Year:** 2023

Reviews evolving strategies in engineering using Kalman, Extended Kalman, Ensemble Kalman, and Particle Filters from a Bayesian perspective. Focus on inverse problems and parameter estimation.

## Cutting-Edge and Application-Driven Research

### State of the Art on State Estimation: Kalman Filter Driven by Artificial Neural Networks
**Publisher:** ScienceDirect
**Year:** 2023

Summarizes advanced variants of Kalman filters that incorporate learning and adaptive mechanisms for enhanced performance. Comprehensive review of neural-network-augmented filtering.

### The Discriminative Kalman Filter for Bayesian Filtering with Nonlinear and Non-Gaussian Models
**Authors:** Burkhart et al., Casco-Rodriguez et al.
**Years:** 2020, 2024

Introduces and replicates a modified Kalman filter using discriminative modeling suitable for complex neural decoding and non-Gaussian environments. Applications in neuroscience and biomedical signal processing.

### Quantitative Verification of Kalman Filters
**Author:** Evangelidis
**Year:** 2021

Evaluates and compares various Kalman filter variants using formal quantitative verification techniques. Rigorous analysis of filter performance and stability.

## Filter Comparison and Selection

### Comparing Filter Types for Nonlinear Systems

Different filtering approaches excel in different scenarios. Here's a comprehensive comparison to guide your choice:

#### Kalman Filter (KF)
- **Best for:** Linear systems with Gaussian noise
- **Accuracy:** Optimal for linear models
- **Computational Cost:** Low
- **Limitations:** Fails in nonlinear settings

#### Extended Kalman Filter (EKF)
- **Best for:** Mildly nonlinear systems
- **Accuracy:** Moderate (first-order linearization)
- **Computational Cost:** Low
- **Strengths:** Computationally efficient baseline
- **Limitations:** Can diverge with strong nonlinearities; suboptimal estimation

**When to use EKF:**
- Process noise exceeds measurement noise
- Nonlinearities are smooth and mild
- Computational resources are limited
- Real-time performance is critical

#### Unscented Kalman Filter (UKF)
- **Best for:** Moderately to strongly nonlinear systems
- **Accuracy:** High (captures up to 3rd-order Taylor terms)
- **Computational Cost:** Moderate to High
- **Strengths:** No Jacobian calculation required; more accurate than EKF
- **Limitations:** Higher computational cost; requires parameter tuning

**When to use UKF:**
- Nonlinearities are significant
- Accurate uncertainty estimation is critical
- You want to avoid derivative calculations
- Computational resources allow moderate overhead

#### Particle Filter (PF)
- **Best for:** Highly nonlinear, non-Gaussian systems
- **Accuracy:** Very high (nonparametric approach)
- **Computational Cost:** Very High
- **Strengths:** Handles arbitrary distributions; most flexible
- **Limitations:** Computationally expensive; particle degeneracy in high dimensions

**When to use Particle Filter:**
- System is highly nonlinear
- Noise is non-Gaussian or multimodal
- Accurate tail probability estimation needed
- Computational resources are available

### Performance Comparison Table

| Filter Type      | Nonlinearity Handling        | Accuracy            | Computational Cost | Robustness                      | Best Use Case                           |
|------------------|------------------------------|---------------------|--------------------|---------------------------------|-----------------------------------------|
| Kalman Filter    | Only linear                  | Optimal for linear  | Low                | Limited to Gaussian linear      | Linear systems, optimal baseline        |
| Extended KF      | First-order linearization    | Moderate            | Low                | Stable under mild nonlinearity  | Mildly nonlinear, real-time systems     |
| Unscented KF     | Sigma point sampling         | High                | Moderate to High   | More robust than EKF            | Moderate nonlinearity, better accuracy  |
| Particle Filter  | Full posterior sampling      | Very High           | Very High          | Handles any nonlinearity/noise  | Highly nonlinear, non-Gaussian systems  |

### Selection Guidelines

**Choose based on your constraints:**

1. **Computational Budget:**
   - Very limited → EKF
   - Moderate → UKF
   - High → Particle Filter

2. **Nonlinearity Level:**
   - Linear → Kalman Filter
   - Mild (< 10% deviation from linear) → EKF
   - Moderate (10-30% deviation) → UKF
   - Strong (> 30% deviation) → Particle Filter

3. **Noise Characteristics:**
   - Gaussian → KF, EKF, or UKF
   - Non-Gaussian but unimodal → UKF
   - Multimodal or arbitrary → Particle Filter

4. **Dimensionality:**
   - Low (< 5 states) → Any filter
   - Medium (5-20 states) → KF, EKF, UKF
   - High (> 20 states) → KF, EKF (PF becomes impractical)

## Implementation Notes

### EKF vs UKF Trade-offs

**Use EKF when:**
- You have analytical Jacobians readily available
- Real-time performance is critical
- Process noise >> measurement noise
- Nonlinearities are smooth and well-behaved

**Use UKF when:**
- Jacobians are difficult or expensive to compute
- Nonlinearities are moderate to strong
- Accurate covariance estimation is important
- You can afford ~3x computational cost vs EKF

### Practical Considerations

**Filter Stability:**
- EKF: Can diverge if linearization point is poor
- UKF: More stable, but sigma points can become poorly conditioned
- PF: Requires careful resampling to avoid particle depletion

**Parameter Tuning:**
- EKF: Tune process and measurement noise covariances
- UKF: Additionally tune alpha, beta, kappa parameters
- PF: Tune number of particles and resampling threshold

## Implementation Status

The following table shows which Bayesian filtering methods are currently implemented in this library:

| Filter/Method                        | Implemented | Module/Class                              | Documentation |
|--------------------------------------|-------------|-------------------------------------------|---------------|
| **Linear Filters**                   |             |                                           |               |
| Kalman Filter (KF)                   | ✅          | `bayesian_filters.kalman.KalmanFilter`    | [Docs](filters/kalman-filter.md) |
| Information Filter                   | ✅          | `bayesian_filters.kalman.InformationFilter` | [Docs](filters/information-filter.md) |
| Square Root Filter                   | ✅          | `bayesian_filters.kalman.SquareRootKalmanFilter` | [Docs](filters/square-root-filter.md) |
| Fading Memory Filter                 | ✅          | `bayesian_filters.kalman.FadingMemoryFilter` | [Docs](filters/fading-kalman-filter.md) |
| **Nonlinear Filters**                |             |                                           |               |
| Extended Kalman Filter (EKF)         | ✅          | `bayesian_filters.kalman.ExtendedKalmanFilter` | [Docs](filters/extended-kalman-filter.md) |
| Unscented Kalman Filter (UKF)        | ✅          | `bayesian_filters.kalman.UnscentedKalmanFilter` | [Docs](filters/unscented-kalman-filter.md) |
| Cubature Kalman Filter (CKF)         | ✅          | `bayesian_filters.kalman.CubatureKalmanFilter` | [API](api/kalman.md) |
| Ensemble Kalman Filter (EnKF)        | ✅          | `bayesian_filters.kalman.EnsembleKalmanFilter` | [Docs](filters/ensemble-kalman-filter.md) |
| **Particle Filters**                 |             |                                           |               |
| Particle Filter (Sequential Monte Carlo) | ⚠️    | `bayesian_filters.monte_carlo` (resampling only) | [API](api/monte-carlo.md) |
| Sequential Importance Resampling (SIR) | ❌        | -                                          | - |
| Regularized Particle Filter          | ❌          | -                                          | - |
| **Multiple Model Filters**           |             |                                           |               |
| Interacting Multiple Model (IMM)     | ✅          | `bayesian_filters.kalman.IMMEstimator`    | [Docs](filters/imm-estimator.md) |
| Multiple Model Adaptive Estimation (MMAE) | ✅   | `bayesian_filters.kalman.MMAEFilterBank`  | [Docs](filters/mmae-filter-bank.md) |
| **Smoothers**                        |             |                                           |               |
| Rauch-Tung-Striebel (RTS) Smoother   | ✅          | `bayesian_filters.kalman.rts_smoother`    | [API](api/kalman.md) |
| Fixed-Lag Smoother                   | ✅          | `bayesian_filters.kalman.FixedLagSmoother` | [API](api/kalman.md) |
| Forward-Backward Smoother            | ❌          | -                                          | - |
| **Robust Filters**                   |             |                                           |               |
| H-Infinity Filter                    | ✅          | `bayesian_filters.hinfinity.HInfinityFilter` | [Docs](filters/h-infinity-filter.md) |
| Huber Filter                         | ❌          | -                                          | - |
| **Other Estimation Methods**         |             |                                           |               |
| g-h Filter (Alpha-Beta Filter)       | ✅          | `bayesian_filters.gh`                     | [Docs](algorithms/gh-filter.md) |
| Discrete Bayes Filter                | ✅          | `bayesian_filters.discrete_bayes`         | [Docs](algorithms/discrete-bayes.md) |
| Least Squares Filter                 | ✅          | `bayesian_filters.leastsq`                | [Docs](algorithms/least-squares.md) |
| **Sigma Point Methods**              |             |                                           |               |
| Merwe Scaled Sigma Points            | ✅          | `bayesian_filters.kalman.MerweScaledSigmaPoints` | [API](api/kalman.md) |
| Julier Sigma Points                  | ✅          | `bayesian_filters.kalman.JulierSigmaPoints` | [API](api/kalman.md) |
| Simplex Sigma Points                 | ✅          | `bayesian_filters.kalman.SimplexSigmaPoints` | [API](api/kalman.md) |

**Legend:**
- ✅ Fully implemented and tested
- ⚠️ Partially implemented (components available, full filter not assembled)
- ❌ Not yet implemented

### Future Roadmap

The following filters and methods are candidates for future implementation:

1. **Particle Filters:** Full implementation of Sequential Importance Resampling (SIR) and Regularized Particle Filter
2. **Advanced Smoothers:** Forward-Backward smoother for more complex scenarios
3. **Robust Filters:** Huber filter for outlier rejection
4. **Adaptive Filters:** Filters with online covariance estimation
5. **Constrained Filters:** Filters that handle state and measurement constraints

Contributions are welcome! See our [Contributing Guide](contributing.md) for details on how to add new filter implementations.

## Additional Resources

- **[Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)** - Comprehensive online book with interactive Jupyter notebooks
- **[Bayesian Filters Documentation](https://georgepearse.github.io/bayesian_filters)** - This library's full API documentation
- **[FilterPy Original Repository](https://github.com/rlabbe/filterpy)** - Original FilterPy project

## Contributing

Know of an important paper or resource that should be included? Please [open an issue](https://github.com/GeorgePearse/bayesian_filters/issues) or submit a pull request!

---

*This research compilation is maintained as part of the Bayesian Filters library. Last updated: 2025.*
