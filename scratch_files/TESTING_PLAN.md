# Comprehensive Unit Testing Plan - Worktree/Branch Structure

**Status:** Approved and In Progress
**Created:** 2025-10-25
**Target Coverage:** 80-90% line coverage
**Total Estimated New Tests:** 400-550 tests

## Overview

Create a systematic testing infrastructure organized by module, targeting 80-90% line coverage with emphasis on property-based testing, numerical accuracy, parametrization, and performance benchmarks.

---

## Branch/Worktree Structure

### 1. testing/infrastructure (Foundation - Start Here)

**Worktree:** `/tmp/bayesian-filters-testing-infrastructure`
**Base Branch:** `master`
**Status:** IN PROGRESS

#### Scope
- Add pytest-cov for coverage reporting
- Add hypothesis for property-based testing
- Add pytest-benchmark for performance tests
- Configure pyproject.toml with pytest settings
- Add pytest-xdist for parallel test execution
- Create GitHub Actions workflow for test coverage reporting
- Add test utilities and fixtures in `bayesian_filters/testing_utils/`
- Document testing standards and conventions

#### Deliverables
- `pyproject.toml` updated with `[tool.pytest.ini_options]` and `[tool.coverage.run]`
- `.github/workflows/test-coverage.yml` for CI coverage reporting
- `bayesian_filters/testing_utils/__init__.py`
- `bayesian_filters/testing_utils/fixtures.py` with common test fixtures
- `bayesian_filters/testing_utils/numerical.py` with analytical solutions for comparison
- `bayesian_filters/testing_utils/helpers.py` with test utilities
- `TESTING.md` documentation in root
- `conftest.py` at root level for pytest plugins

#### Key Dependencies
```
pytest>=7.0
pytest-cov>=4.0
hypothesis>=6.0
pytest-benchmark>=4.0
pytest-xdist>=3.0
pytest-mpl>=0.15 (for visual regression)
```

#### Implementation Checklist
- [ ] Update pyproject.toml with pytest configuration
- [ ] Create testing_utils package structure
- [ ] Implement basic fixtures (kf_1d, ekf_2d, ukf_3d, etc.)
- [ ] Implement analytical solutions library
- [ ] Create conftest.py with pytest plugins
- [ ] Add GitHub Actions test-coverage.yml workflow
- [ ] Write TESTING.md documentation
- [ ] Test infrastructure setup

---

### 2. testing/kalman-core (PRIORITY - Core Filters)

**Worktree:** `/tmp/bayesian-filters-testing-kalman-core`
**Base Branch:** `testing/infrastructure`
**Status:** PENDING

#### Modules to Test
- **kalman_filter.py** (KalmanFilter class) - CRITICAL
- **EKF.py** (ExtendedKalmanFilter) - CRITICAL
- **UKF.py** (UnscentedKalmanFilter) - CRITICAL

#### Current Test State
| Module | File | Tests | Lines | Status |
|--------|------|-------|-------|--------|
| KalmanFilter | test_kf.py | 14 | 738 | Decent |
| EKF | test_ekf.py | 1 | 122 | MINIMAL |
| UKF | test_ukf.py | 16 | 1108 | Good |

#### Testing Focus

##### 1. Unit Tests
- Test each method independently (predict, update, batch_filter, etc.)
- Edge cases: singular matrices, zero covariance, dimension mismatches
- Error conditions: invalid inputs, numerical instability
- State shape handling (row vs column vectors)
- Matrix dimension validation

##### 2. Property-Based Tests (Hypothesis)
- Generate random valid filter configurations
- Verify conservation properties (covariance remains positive semi-definite)
- Test invariants (symmetry of P, positive definiteness after update)
- Commutative properties where applicable

##### 3. Numerical Accuracy Tests
- Compare against analytical solutions (constant velocity, constant acceleration)
- Verify against published benchmark results
- Test numerical stability with ill-conditioned matrices
- Cross-validation between different filter implementations

##### 4. Parametrized Tests
- Different state dimensions (1D, 2D, 3D, 4D, 10D)
- Different measurement dimensions
- Various noise levels (high, medium, low SNR)
- Different control signal scenarios
- Batch vs sequential filtering equivalence

##### 5. Performance Benchmarks
- Baseline filter update/predict times
- Scaling with state dimension (O(n²) vs O(n³))
- Memory usage profiles
- Vectorization effectiveness

#### Estimated New Tests
- KalmanFilter: 50-60 new tests
- EKF: 40-50 new tests
- UKF: 30-40 new tests
- **Total: 120-150 new tests**

#### New Test File Targets
- `bayesian_filters/kalman/tests/test_kf_comprehensive.py` - Extended KF tests
- `bayesian_filters/kalman/tests/test_ekf_comprehensive.py` - Extended EKF tests
- `bayesian_filters/kalman/tests/test_ukf_comprehensive.py` - Extended UKF tests

---

### 3. testing/kalman-advanced (Advanced Filters)

**Worktree:** `/tmp/bayesian-filters-testing-kalman-advanced`
**Base Branch:** `testing/infrastructure`
**Status:** PENDING

#### Modules to Test
- **CubatureKalmanFilter.py** (test_ckf.py: 1 test → expand to 20-30)
- **ensemble_kalman_filter.py** (test_enkf.py: 2 tests → expand to 15-25)
- **IMM.py** (test_imm.py: 2 tests → expand to 20-30)
- **information_filter.py** (test_information.py: 12 tests → expand to 25-35)
- **square_root.py** (test_sqrtkf.py: 1 test → expand to 20-30)
- **fading_memory.py** (test_fm.py: 1 test → expand to 15-25)
- **fixed_lag_smoother.py** (test_fls.py: 2 tests → expand to 15-25)
- **mmae.py** (test_mmae.py: 1 test → expand to 20-30)
- **RTS Smoother** (test_rts.py: 1 test → expand to 20-30)
- **Sensor Fusion** (test_sensor_fusion.py: 1 test → expand to 15-25)

#### Testing Approach
- Same comprehensive approach as core filters
- Cross-validation between filter types (same problem, different filters)
- Comparison with standard KF/EKF/UKF where applicable
- Filter-specific edge cases:
  - EnKF: ensemble size, covariance localization
  - IMM: model switching, mixing weights
  - CKF: cubature point distribution
  - Information Filter: information matrix singularity
  - RTS: backward pass correctness
  - Fading Memory: forgetting factor effects

#### Estimated New Tests
- Per module: 20-30 new tests
- **Total: 160-240 new tests**

---

### 4. testing/stats (Statistical Functions)

**Worktree:** `/tmp/bayesian-filters-testing-stats`
**Base Branch:** `testing/infrastructure`
**Status:** PENDING

#### Module
- **stats/stats.py**
- Current: test_stats.py: 5 tests, 301 lines

#### Functions Needing Thorough Testing
- Gaussian operations (gaussian, multivariate_gaussian, mul, add, mul_pdf)
- Mahalanobis distance
- Log-likelihood and likelihood
- Plotting functions (plot_gaussian_pdf, plot_covariance, plot_gaussian_cdf, etc.)
- Statistical utilities (NEES, norm_cdf)
- Covariance ellipse calculations

#### Testing Focus
- Numerical accuracy against scipy.stats
- Edge cases:
  - Singular covariances
  - High dimensions
  - Extreme values (very small/large covariances)
  - Near-zero variances
- Property-based tests for Gaussian algebra:
  - Multiplication is associative where defined
  - Addition is commutative
  - Mahalanobis distance properties
- Visual regression testing for plotting functions (using pytest-mpl)
- Comparison with scipy implementations

#### Estimated New Tests
- **30-40 new tests**

#### Coverage Targets
- 95%+ coverage (mostly plotting, validation code)

---

### 5. testing/common (Utility Functions)

**Worktree:** `/tmp/bayesian-filters-testing-common`
**Base Branch:** `testing/infrastructure`
**Status:** PENDING

#### Modules
- **helpers.py** (test_helpers.py: 7 tests → expand to 20-30)
- **discretization.py** (test_discretization.py: 3 tests → expand to 15-20)
- **kinematic.py** (test_kinematic.py: 29 test methods → expand to 40-50)

#### Testing Focus
- Matrix operations accuracy
- Q_discrete_white_noise correctness (compare to analytical Wiener process)
- Kinematic model validation against analytical solutions
- Helper function edge cases
- Block diagonal and other matrix utilities

#### Specific Tests
- Discretization accuracy for different dt values
- Kinematic state transition equivalence
- Saver class functionality
- Q matrix positive definiteness

#### Estimated New Tests
- **20-30 new tests**

---

### 6. testing/other-modules (Remaining Modules)

**Worktree:** `/tmp/bayesian-filters-testing-other`
**Base Branch:** `testing/infrastructure`
**Status:** PENDING

#### Modules
- **discrete_bayes/** (test_discrete_bayes.py: 1 test → expand to 10-15)
- **gh/** (test_gh.py: 4 tests → expand to 15-20)
- **hinfinity/** (test_hinfinity.py: 1 test → expand to 10-15)
- **leastsq/** (test_lsq.py: 6 tests → expand to 20-25)
- **memory/** (test_fading_memory.py: 1 test → expand to 10-15)
- **monte_carlo/** (test_resampling.py: 23 test methods → expand to 50-60)

#### Testing Focus
- Module-specific algorithms
- Cross-module integration
- Edge cases and error handling
- Algorithm correctness against publications

#### Estimated New Tests
- **30-50 new tests**

---

## Testing Standards

### Test Organization

Each test file should contain sections in this order:
```python
# 1. Imports and setup
# 2. Fixtures and helpers
# 3. Unit tests (test_<method_name>_<scenario>)
# 4. Integration tests (test_integration_<workflow>)
# 5. Property-based tests (test_property_<invariant>)
# 6. Parametrized tests (using @pytest.mark.parametrize)
# 7. Performance benchmarks (benchmark_<operation>)
```

### Coverage Requirements
- **Target:** 80-90% line coverage per module
- **Critical paths:** 100% coverage for predict/update methods
- **Error handling:** All error conditions must be tested
- **Edge cases:** Documented and tested
- **Numerical stability:** Tested with condition numbers >1e6

### Naming Conventions
```python
test_<method_name>_<scenario>              # Unit tests
    e.g., test_predict_constant_velocity
    e.g., test_update_dimension_mismatch

test_integration_<workflow_name>           # Integration tests
    e.g., test_integration_batch_filtering
    e.g., test_integration_multi_model_switching

test_property_<invariant_name>             # Property-based (Hypothesis)
    e.g., test_property_covariance_remains_psd
    e.g., test_property_symmetry_preserved

benchmark_<operation>                      # Performance tests
    e.g., benchmark_predict_1d_filter
    e.g., benchmark_update_scaling_with_dimension
```

### Test Markers
```python
@pytest.mark.unit           # Fast, isolated tests
@pytest.mark.integration    # Tests multiple components
@pytest.mark.slow          # Tests that take >1 second
@pytest.mark.benchmark     # Performance benchmarks
@pytest.mark.parametrize   # Multiple parameter sets
```

### Assertion Style
- Use `pytest.approx()` for float comparisons (not `==`)
- Use `np.allclose()` for array comparisons
- Document tolerance in comments
- Use descriptive assertion messages

---

## Implementation Timeline

### Week 1: Infrastructure
- [ ] Create testing/infrastructure branch
- [ ] Update pyproject.toml
- [ ] Create testing_utils package
- [ ] Implement fixtures and helpers
- [ ] Set up GitHub Actions workflow
- [ ] Write TESTING.md documentation

### Week 2-3: Core Kalman Filters
- [ ] Expand test_kf.py to 50-60 tests
- [ ] Expand test_ekf.py from 1 to 40-50 tests
- [ ] Expand test_ukf.py to 30-40 tests
- [ ] Add property-based tests
- [ ] Add benchmarks
- [ ] Achieve 90%+ coverage

### Week 4: Advanced Kalman Filters
- [ ] Expand all advanced filter tests
- [ ] Cross-validation tests
- [ ] Algorithm correctness verification
- [ ] Achieve 85%+ coverage

### Week 5: Stats & Common
- [ ] Expand stats/stats.py tests
- [ ] Expand common utility tests
- [ ] Add numerical accuracy tests
- [ ] Achieve 90%+ coverage

### Week 6: Other Modules & Integration
- [ ] Expand remaining module tests
- [ ] Add integration tests
- [ ] Full system validation
- [ ] Coverage reporting

---

## Merge Strategy

Each testing branch will be reviewed and merged independently:

1. **Create PR** from testing branch to master
2. **Requirements:**
   - 80%+ coverage in modified modules
   - All tests pass
   - Performance benchmarks must not regress >10%
   - Code review approval
3. **Merge** when all requirements met
4. **Parallel Development:** Branches can be worked on in parallel

### PR Template
```markdown
## Testing Enhancement for [Module]

### Coverage
- Current: X%
- After: Y%
- New tests: N

### Test Types
- [ ] Unit tests
- [ ] Property-based tests
- [ ] Integration tests
- [ ] Numerical accuracy tests
- [ ] Performance benchmarks

### Validation
- [ ] All tests pass
- [ ] Coverage increased
- [ ] No performance regressions
- [ ] Documentation updated
```

---

## Performance Baseline

Benchmarks will establish baselines for:
- KalmanFilter.predict() - 1D to 10D states
- KalmanFilter.update() - 1D to 10D measurements
- EKF operations with nonlinear functions
- UKF operations with different sigma points
- Batch filtering performance

Regression threshold: 10% slowdown triggers investigation

---

## Testing Utilities Overview

### `testing_utils/fixtures.py`
Common pytest fixtures:
- `kf_1d` - 1D Kalman filter
- `kf_2d` - 2D Kalman filter
- `ekf_2d` - 2D Extended Kalman filter
- `ukf_3d` - 3D Unscented Kalman filter
- `sensor_sim` - Simple sensor simulator
- `noisy_measurements` - Generate measurement sequences

### `testing_utils/numerical.py`
Analytical solutions and comparisons:
- Constant velocity filter analytical solution
- Constant acceleration filter solution
- Linear system steady state
- Benchmark filter configurations
- Published test cases

### `testing_utils/helpers.py`
Test utilities:
- Matrix verification functions
- Filter validation helpers
- Performance measurement utilities
- Comparison against scipy/numpy

---

## Success Criteria

✅ **Complete when:**
1. All 6 branches created and merged
2. Total coverage: 80-90% across all modules
3. Critical paths (predict/update): 100%
4. 400-550 new tests implemented
5. All property-based tests passing
6. Benchmarks established and no regressions
7. TESTING.md updated with results
8. CI/CD pipeline reporting coverage
9. Documentation complete

---

## References & Resources

### Testing Frameworks
- [pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Guide](https://hypothesis.readthedocs.io/)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)

### Kalman Filter Testing
- Welch & Bishop: "An Introduction to the Kalman Filter"
- Published benchmark datasets
- Cross-reference with reference implementations

### Numerical Testing
- IEEE 754 floating point standards
- Condition number analysis
- Matrix stability considerations

---

## Notes

- Tests should be deterministic (use fixed seeds for randomness)
- Avoid hardcoded magic numbers - use descriptive constants
- Document assumptions and edge cases in docstrings
- Use type hints in test functions where helpful
- Keep tests focused and independent

---

**Last Updated:** 2025-10-25
**Plan Version:** 1.0
**Approved:** Yes
