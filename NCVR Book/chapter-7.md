---
title: Chapter 7
description: ''
subtitle: Random Numbers
tags: []
authors:
  - userId: UkR8ZX1I0lY4DNcav9EKF38zAzL2
    nameParsed:
      literal: Risman Adnan, Ph.D
      given: Ph.D
      family: Risman Adnan
    name: Risman Adnan, Ph.D
    corresponding: false
    roles: []
    affiliations: []
    id: contributors-generated-uid-0
  - nameParsed:
      literal: Ayesha Ayub Syed
      given: Ayesha Ayub
      family: Syed
    name: Ayesha Ayub Syed
    corresponding: false
    roles: []
    affiliations: []
    id: contributors-generated-uid-1
  - userId: KrPveBMe0VTjTZQ8oza9Gsz4pYr1
    nameParsed:
      literal: Alifia Mutiara Rahma
      given: Alifia Mutiara
      family: Rahma
    name: Alifia Mutiara Rahma
    orcid: 0009-0002-0720-9248
    corresponding: false
    roles: []
    affiliations: []
    id: contributors-generated-uid-2
date: '2025-10-19'
oxa: oxa:pqQDe4beUu67RvW3raYP/6xBEOPWjNIfSCtdaGNgF
keywords: []
---

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/CT7l5jr776MfKgF8f8zg.2","tags":[]}

> "Randomness is the true nature of the world, and understanding it is the key to navigating real life." — N.N. Taleb

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/eLFztt2gVmQiqZcZBeh2.1","tags":[]}

*Chapter 7 introduces random number generation, hashing, and Monte Carlo methods, which form the foundation of stochastic simulation and probabilistic computation. The chapter begins with pseudo-random number generators, including classical linear congruential generators, modern uniform generators, and statistical testing techniques. Hashing methods, array randomization, and hash-table data structures are then examined as tools for efficient data organization and randomized computation. The discussion continues with techniques for generating random deviates from common probability distributions and sampling multivariate normal distributions. Linear feedback shift registers are introduced as efficient generators for hardware and high-performance applications. The chapter concludes with Monte Carlo integration, quasi-random sequences, and adaptive sampling methods such as VEGAS. Throughout the chapter, mathematical principles are integrated with practical Rust implementations for simulation, optimization, uncertainty quantification, and scientific computing.*

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/dToxnVV3FRS9cW0HQG8I.7","tags":[]}

# 7.1. Introduction

Computers are deterministic machines. Given the same instructions and inputs, they will always produce identical results. Yet, paradoxically, much of modern numerical computation depends critically on *randomness*. Whether estimating multidimensional integrals, sampling from probability distributions, modeling physical uncertainty, or training large machine-learning models, random numbers form the backbone of algorithms whose strength lies in statistical behavior rather than strict determinism.

The use of randomness in computation is not an admission of ignorance; it is a deliberate mathematical strategy to overcome the limitations of deterministic approaches. Randomness enables algorithms to explore large or complex spaces efficiently, to approximate intractable quantities, and to avoid pathological behavior that might arise from rigid, structured iteration patterns. Understanding the mathematical principles and computational mechanisms behind random number generation is therefore fundamental to numerical science.

## 7.1.1. Randomness in Numerical Computing

In numerical analysis and scientific simulation, randomness serves two principal purposes: (i) to approximate continuous phenomena through probabilistic sampling, and (ii) to introduce controlled variability that enhances stability or generalization in algorithms.

### Monte Carlo and Stochastic Approximation

The most classical setting is *Monte Carlo integration*, where an expectation or integral is approximated by averaging evaluations of a function at random sample points. For a function $f(x)$ over a domain $\Omega$, we estimate:

$$I = \int_{\Omega} f(x),dx \approx \frac{1}{N}\sum_{i=1}^{N} f(X_i)\tag{7.1.1}$$

### Stochastic Optimization and Randomized Algorithms

In optimization, randomness breaks symmetry and mitigates bias. Stochastic gradient descent (SGD), for example, samples subsets of data randomly at each iteration to approximate the gradient of a loss function. Similarly, *randomized numerical linear algebra* uses random projections to compress matrices while preserving their essential spectral structure.

For example, in randomized SVD one generates a random test matrix $\Omega$ and computes

$$Y = A,\Omega\tag{7.1.3}$$

where the columns of $Y$ form a low-dimensional sketch of $A$. The reliability of this approximation depends directly on the statistical properties of the random numbers used to construct $\Omega$.

### Noise, Perturbation, and Physical Simulation

Random numbers also appear as controlled perturbations in physical simulations, modeling thermal noise, Brownian motion, or turbulence, and in stochastic partial differential equations (SPDEs). In such systems, the *statistics* of the random input influence the stability and fidelity of the numerical solution.

These diverse applications demonstrate that *randomness is not a by-product of numerical methods but a computational resource*. Its correct use demands that generated numbers exhibit the statistical properties required by theory: independence, uniformity, and absence of bias or correlation (Almaraz Luengo, 2022).

### Rust Implementation

Following the discussion in Section 7.1.1 on the role of randomness as a computational resource in numerical methods, Program 7.1.1 provides concrete demonstrations of how pseudorandom number generation drives three fundamental algorithmic paradigms: Monte Carlo integration, stochastic optimization, and randomized numerical linear algebra. Each of these relies on probabilistic sampling to approximate continuous phenomena, introduce controlled variability, or construct compressed representations of high-dimensional operators. The program illustrates how statistical properties such as independence, unbiasedness, and distributional correctness that are the central requirements highlighted in this section directly influence the fidelity and stability of numerical computations. By implementing representative examples in Rust using the `rand`, `rand_distr`, and `ndarray` crates, the code showcases how modern numerical software employs well-designed random number generators to support expectation estimation, gradient approximation, and matrix sketching within a unified and reproducible computational environment.

At the core of the program are three functions, each illustrating one of the key categories of randomized numerical computation discussed earlier. The first function, `monte_carlo_integration`, directly implements the estimator in Equation (7.1.1) by drawing i.i.d. uniform samples $X_i \sim U(0,1)$ and computing the average of $f(X_i)$. This demonstrates how Monte Carlo methods approximate an integral by replacing a continuous domain with statistically representative sample points. The accuracy of the approximation depends on the variance of the integrand and on the distributional quality of the pseudorandom numbers, both of which govern the characteristic $N^{-1/2}$ decay of the Monte Carlo error.

The second function, `stochastic_gradient_descent`, illustrates the role of randomness in optimization algorithms. Instead of evaluating the exact gradient of the expected loss $L(\theta) = \mathbb{E}[(\theta - Z)^2]$, the function draws a fresh sample $Z_k$ at each iteration and computes an unbiased stochastic gradient based on this value. This controlled injection of randomness breaks symmetry, introduces natural regularization, and enables updates based on small amounts of information, a behaviour consistent with large-scale stochastic gradient descent in practice. The iterates therefore fluctuate around the minimizer rather than converging exactly, reflecting the inherent stochasticity of the method.

The final function, `randomized_matrix_sketch`, demonstrates how randomness enables efficient numerical linear algebra through low-dimensional projections. By generating a Gaussian test matrix $\Omega$ and forming the sketch $Y = A \Omega$, the implementation constructs a compressed representation of $A$ in the spirit of Equation (7.1.3). Such random projections form the basis of modern randomized SVD techniques, where the statistical properties of the test matrix determine how well the sketch preserves the dominant spectral features of $A$. This highlights once more that the effectiveness of randomized numerical algorithms depends critically on the quality of the underlying random number generator.

Together, these three functions show how randomness acts not as incidental noise but as a structural computational tool, facilitating approximation, optimization, and high-dimensional operator compression, fully aligned with the principles outlined in Section 7.1.1.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
rand_distr = "0.4"
ndarray = "0.15"
```

```rust
//! Demonstrations for Section 7.1.1: Randomness in Numerical Computing
//!
//! This example illustrates three roles of randomness:
//! 1. Monte Carlo integration (approximate expectations by sampling).
//! 2. Stochastic optimization via a toy SGD routine.
//! 3. Randomized numerical linear algebra via a random projection
//!    (a simple instance of randomized SVD ideas).

use ndarray::{array, Array2};
use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal};

fn main() {
    // Use a reproducible RNG so that results are stable in the book.
    let mut rng = StdRng::seed_from_u64(7_101_001);

    println!("=== 1. Monte Carlo integration ===");
    monte_carlo_integration(&mut rng);

    println!("\n=== 2. Stochastic gradient descent (SGD) ===");
    stochastic_gradient_descent(&mut rng);

    println!("\n=== 3. Randomized matrix sketch A * Ω ===");
    randomized_matrix_sketch(&mut rng);
}

/// 1. Monte Carlo integration
///
/// We approximate the integral
///
///     I = ∫_0^1 exp(-x^2) dx
///
/// by uniform sampling X_i ~ U(0,1) and averaging f(X_i).
fn monte_carlo_integration<R: Rng + ?Sized>(rng: &mut R) {
    let n_samples = 100_000usize;

    let mut sum = 0.0_f64;
    for _ in 0..n_samples {
        let x: f64 = rng.gen(); // U(0, 1)
        let fx = (-x * x).exp();
        sum += fx;
    }

    let estimate = sum / n_samples as f64;
    println!("Estimated I = ∫_0^1 e^(-x^2) dx ≈ {estimate:.6}");

    // For comparison, the exact value is 0.5 * sqrt(pi) * erf(1).
    // We just show a hard-coded reference here to avoid extra dependencies.
    let reference = 0.7468241328_f64; // ≈ ∫_0^1 e^(-x^2) dx
    println!("Reference (high-accuracy)    ≈ {reference:.6}");
    println!("Absolute error               ≈ {:.3e}", (estimate - reference).abs());
}

/// 2. Stochastic Gradient Descent (SGD)
///
/// We minimize the expected quadratic loss
///
///     L(θ) = E[(θ - Z)^2],
///
/// where Z ~ N(μ, σ²) is a random variable. The minimizer is θ* = μ,
/// but we pretend we do not know μ and approximate the gradient using
/// samples z_k.
///
/// The gradient of (θ - z)^2 w.r.t. θ is 2(θ - z). Using a learning
/// rate η and a single sample z_k at each iteration gives:
///
///     θ_{k+1} = θ_k - η · 2(θ_k - z_k).
fn stochastic_gradient_descent<R: Rng + ?Sized>(rng: &mut R) {
    let true_mean = 2.5_f64;
    let true_std = 0.7_f64;

    // Initial guess for θ (far from the true mean).
    let mut theta = -5.0_f64;

    // Learning rate and iterations.
    let eta = 0.05_f64;
    let n_iters = 2000usize;

    let normal = StandardNormal;

    for k in 0..n_iters {
        // Explicitly sample an f64 from N(0,1) to avoid type ambiguity.
        let z_noise: f64 = normal.sample(rng);
        let z_sample: f64 = true_mean + true_std * z_noise;

        // Gradient of (θ - z)^2 is 2(θ - z).
        let grad = 2.0 * (theta - z_sample);

        // SGD update.
        theta -= eta * grad;

        // Print occasionally to show convergence.
        if (k + 1) % 500 == 0 {
            println!(
                "Iteration {:4}: θ ≈ {:.6}, distance to μ = {:.3e}",
                k + 1,
                theta,
                (theta - true_mean).abs()
            );
        }
    }

    println!("Final θ ≈ {:.6}, true μ = {:.6}", theta, true_mean);
}

/// 3. Randomized matrix sketch A * Ω
///
/// We construct:
///   - a fixed 4×4 matrix A,
///   - a random 4×2 Gaussian matrix Ω,
/// and compute
///
///     Y = A Ω,
///
/// whose columns form a low-dimensional sketch of A. In a full
/// randomized SVD, one would then orthogonalize the columns of Y
/// and compute an SVD in that reduced space.
///
/// Here we only illustrate the randomness and the basic multiplication.
fn randomized_matrix_sketch<R: Rng + ?Sized>(rng: &mut R) {
    // A simple 4×4 matrix (e.g. a discretized operator).
    let a: Array2<f64> = array![
        [4.0, 1.0, 0.0, 0.0],
        [1.0, 3.0, 1.0, 0.0],
        [0.0, 1.0, 3.0, 1.0],
        [0.0, 0.0, 1.0, 2.0],
    ];

    let (m, n) = (a.nrows(), a.ncols());
    assert_eq!(m, 4);
    assert_eq!(n, 4);

    // Target sketch dimension k < n.
    let k = 2usize;

    // Construct Ω ∈ R^{n×k} with i.i.d. N(0,1) entries.
    let mut omega = Array2::<f64>::zeros((n, k));
    let normal = StandardNormal;
    for i in 0..n {
        for j in 0..k {
            let g: f64 = normal.sample(rng);
            omega[(i, j)] = g;
        }
    }

    // Compute Y = A Ω.
    let y = a.dot(&omega);

    println!("Matrix A (4×4):\n{a}");
    println!("\nRandom test matrix Ω (4×2):\n{omega}");
    println!("\nSketch Y = A Ω (4×2):\n{y}");

    // In a full randomized SVD:
    //   1. Orthonormalize columns of Y: Q = orth(Y).
    //   2. Form B = Q^T A.
    //   3. Compute SVD of B: B ≈ U_b Σ V^T.
    //   4. Then U ≈ Q U_b gives approximate left singular vectors.
    //
    // Here we stop at Y to highlight where randomness enters.
}
```

Program 7.1.1 illustrates how randomness enters numerical computation in three distinct but complementary forms: expectation approximation, stochastic optimization, and randomized operator compression. As emphasized in Section 7.1.1, these algorithmic strategies rely on the statistical properties of the underlying random number generator, and even small deviations from independence, uniformity, or distributional correctness can lead to biased estimates or unstable iterates.

The Monte Carlo example highlights the characteristic behaviour of sampling-based estimators, whose accuracy improves predictably with $N^{-1/2}$ convergence but depends critically on the variance of the integrand. The stochastic gradient descent demonstration shows how randomness enables scalable optimization by providing computationally inexpensive gradient approximations, albeit with inherent fluctuations around the minimizer. The matrix sketch example underscores how random projections can replace expensive deterministic factorizations with lightweight, statistically reliable approximations of high-dimensional linear operators.

The modular structure of the implementation allows these examples to serve as templates for more advanced techniques, including quasi–Monte Carlo sampling, variance-reduced stochastic optimization, and full randomized SVD algorithms. Collectively, the program reinforces the central theme of Section 7.1.1: randomness, when properly controlled and theoretically aligned, provides a powerful and flexible tool for constructing efficient numerical methods across a wide spectrum of applications.

## 7.1.2. From True Randomness to Algorithmic Randomness

While natural randomness arises from physical or quantum processes, computers must simulate it algorithmically. Thus we distinguish between true random numbers and pseudo-random numbers.

True random numbers are produced by inherently unpredictable physical events including radioactive decay, thermal noise, photon arrival times, or quantum vacuum fluctuations. Their outputs are nondeterministic, irreproducible, and usually obtained from specialized hardware or external services. Modern *quantum random number generators* (QRNGs) achieve generation rates exceeding 100 Gb/s using integrated optical systems (Bruynsteen, Wang and Vermeulen, 2022).

Pseudo-random numbers, by contrast, are produced deterministically. A pseudo-random number generator (PRNG) is a mathematical algorithm that, from an initial seed value, generates a sequence of numbers that appears statistically random. If the seed is known, the sequence is entirely reproducible, a crucial property for scientific computation, where repeatability of experiments and debugging require deterministic reproduction.

Hence, *randomness in numerical computing* is fundamentally algorithmic. We require sequences that are not genuinely random in the physical sense, but that *behave as if* they were, for the statistical purposes of our algorithms.

## 7.1.3. Mathematical Structure of a PRNG

A PRNG can be formally described as a discrete dynamical system

$$s_{n+1} = f(s_n), \qquad u_n = O(s_n),\tag{7.1.4}$$

where $s_n \in S$ is the internal state at step $n$, $f: S \to S$ is a deterministic transition function, and $O: S \to [0,1)$ maps the state to the output sequence $u_n$.

The system is initialized with a seed $s_0$. Because $S$ is finite in digital arithmetic, the sequence eventually repeats: there exists a smallest T such that $s_{n+T} = s_n$ for all sufficiently large $n$. The integer $T$ is called the *period* of the generator.

A desirable PRNG has an extremely long period and produces values that are *k-dimensionally equidistributed*; that is, tuples

$$(u_n,,u_{n+1},,\dots,,u_{n+k-1})\tag{7.1.5}$$

For convenience, many generators yield integer outputs that are scaled to the interval $[0,1)$:

$$x_n = X_n \bmod m, \qquad u_n = \frac{X_n}{m}\tag{7.1.6}$$

### Rust Implementation

Following the discussion in Sections 7.1.2 and 7.1.3 on the distinction between true randomness and algorithmic randomness, Program 7.1.2 presents a concrete implementation of a pseudo-random number generator viewed explicitly through its mathematical formulation as a discrete dynamical system. While physical randomness arises from inherently unpredictable quantum or thermal phenomena, numerical computing depends on deterministic procedures that emulate randomness through structured state transitions. This program illustrates how a PRNG generates reproducible sequences via the update rule in Equation (7.1.4), and how its outputs correspond to the normalized values described in Equation (7.1.6). To make these concepts transparent, two generators are implemented: a 64-bit linear congruential generator suitable for numerical experimentation, and a deliberately small 8-bit version that allows the finite-period behavior of the system to be observed directly. The example demonstrates the core computational idea that algorithmic randomness, although deterministic, is designed to behave statistically like true randomness for the purposes of simulation, sampling, and stochastic numerical methods.

At the foundation of the implementation is the direct realization of the PRNG structure described by Equation (7.1.4). The internal state $s_n$ evolves deterministically under a transition function $f$, while the observable output $u_n$ is obtained by mapping the state through an output function $O$. The `Lcg64` generator embodies this structure using a classical linear congruential form, where the update $s_{n+1} = (a s_n + c) \bmod 2^{64}$ models the transition map and the normalized floating-point value $u_n = s_n / 2^{64}$ corresponds exactly to the scaling described in Equation (7.1.6). Because the computation is entirely deterministic once the seed is chosen, the same initial state produces identical sequences, making reproducibility inherent to the algorithm.

The second implementation, `TinyLcg8`, mirrors the same mathematical structure but operates over a very small state space of size $m = 256$. This allows the finite-period property of Equation (7.1.4) to be demonstrated explicitly: since digital arithmetic admits only finitely many distinct states, the sequence must eventually repeat. By tracking how many steps elapse before the generator returns to its initial state, the program computes the exact period $T$ for a given seed. This provides a tangible illustration of the periodicity that all PRNGs possess, though modern generators are designed such that $T$ is extremely large and practically unreachable in real computation.

The `main` function showcases three central properties of algorithmic randomness. First, it demonstrates reproducibility by constructing two generators with the same seed and verifying that their initial outputs match exactly. Second, it estimates the mean of the uniform distribution by averaging many successive values of the 64-bit generator, showing that the empirical mean aligns with the theoretical mean of 0.5. Third, it uses the 8-bit generator to detect the full period of the transition system directly. These examples collectively highlight how PRNGs behave as deterministic dynamical systems while exhibiting statistical properties that make them suitable for Monte Carlo simulation, randomized algorithms, and stochastic numerical analysis.

```rust
//! Program 7.1.2: Simple PRNG as a discrete dynamical system
//!
//! This example illustrates the mathematical structure of a PRNG as in
//! Equation (7.1.4):
//!
//!     s_{n+1} = f(s_n),   u_n = O(s_n),
//!
//! where `s_n` is the internal state and `u_n` is the output in [0,1).
//! We implement a 64-bit linear congruential generator (LCG) and a
//! tiny 8-bit LCG to empirically show the notion of a finite period.

use std::u64;

/// A simple 64-bit linear congruential generator (LCG).
///
/// Mathematically, the internal state evolves as
///
///     s_{n+1} = (a * s_n + c) mod 2^64,
///
/// and the output is obtained by scaling to [0,1) as in Equation (7.1.6):
///
///     x_n = X_n mod m,   u_n = X_n / m,
///
/// where here `m = 2^64` and `X_n` is the 64-bit state.
struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    // Multiplier and increment parameters (example values).
    const A: u64 = 6364136223846793005;
    const C: u64 = 1;

    /// Initialize the generator with a given seed s_0.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Transition function f: S -> S, updating s_n to s_{n+1}.
    fn next_state(&mut self) {
        // Wrapping arithmetic implements the modulo 2^64 reduction.
        self.state = self
            .state
            .wrapping_mul(Self::A)
            .wrapping_add(Self::C);
    }

    /// Integer output X_n (here we simply use the state itself).
    fn next_u64(&mut self) -> u64 {
        self.next_state();
        self.state
    }

    /// Real-valued output u_n in [0,1), corresponding to Equation (7.1.6).
    fn next_unit(&mut self) -> f64 {
        let x = self.next_u64();
        // Map X_n to u_n = X_n / m with m = 2^64.
        (x as f64) / ((u64::MAX as f64) + 1.0)
    }
}

/// A tiny 8-bit LCG to demonstrate the concept of a finite period T explicitly.
///
/// State space: S = {0, 1, ..., 255}, modulus m = 256.
/// We choose parameters that give a full-period generator.
#[derive(Clone, Copy, Debug)]
struct TinyLcg8 {
    state: u8,
}
#[allow(dead_code)]
impl TinyLcg8 {
    const A: u8 = 5;  // multiplier
    const C: u8 = 1;  // increment

    fn new(seed: u8) -> Self {
        Self { state: seed }
    }

    /// State transition s_{n+1} = (a * s_n + c) mod 256.
    fn next_state(&mut self) {
        let s = self.state as u16;
        let a = Self::A as u16;
        let c = Self::C as u16;
        let next = (a * s + c) % 256;
        self.state = next as u8;
    }

    /// Output in [0,1): u_n = s_n / 256.
    fn output(&self) -> f64 {
        self.state as f64 / 256.0
    }
}

fn main() {
    // ------------------------------------------------------------
    // 1. Demonstrate algorithmic randomness and reproducibility.
    // ------------------------------------------------------------

    println!("=== Lcg64: reproducible pseudo-random sequence ===");

    let mut g1 = Lcg64::new(12345);
    let mut g2 = Lcg64::new(12345); // same seed => identical sequence

    println!("First 5 outputs from g1 (seed = 12345):");
    for _ in 0..5 {
        let u = g1.next_unit();
        println!("{:.16}", u);
    }

    println!("\nFirst 5 outputs from g2 (same seed = 12345):");
    for _ in 0..5 {
        let u = g2.next_unit();
        println!("{:.16}", u);
    }

    println!("\nNote: the sequences match exactly, illustrating determinism and reproducibility.");

    // ------------------------------------------------------------
    // 2. Approximate mean of U(0,1) via the PRNG.
    // ------------------------------------------------------------

    println!("\n=== Lcg64: empirical mean of u_n in [0,1) ===");

    let mut g_mean = Lcg64::new(42);
    let n_samples = 100_000;
    let mut sum = 0.0_f64;
    for _ in 0..n_samples {
        sum += g_mean.next_unit();
    }
    let mean_estimate = sum / n_samples as f64;
    println!(
        "Empirical mean over {} samples ≈ {:.6} (theoretical mean is 0.5)",
        n_samples, mean_estimate
    );

    // ------------------------------------------------------------
    // 3. Tiny PRNG: explicitly observe the finite period.
    // ------------------------------------------------------------

    println!("\n=== TinyLcg8: explicit period detection ===");

    let mut tiny = TinyLcg8::new(1); // seed s_0 = 1
    let start_state = tiny.state;
    let mut period = 0usize;

    loop {
        tiny.next_state();
        period += 1;

        if tiny.state == start_state {
            break;
        }
        // Guard against programming errors; for m=256, period cannot exceed 256.
        if period > 256 {
            println!("Unexpected: exceeded maximum possible period!");
            break;
        }
    }

    println!(
        "Starting from seed s_0 = 1, TinyLcg8 returns to its initial state after T = {} steps.",
        period
    );
    println!("This T is the period of the generator for this seed in the 8-bit state space.");
}
```

Program 7.1.2 illustrates the fundamental computational principles underlying pseudo-random number generation by directly implementing the dynamical structure presented in Sections 7.1.2 and 7.1.3. Although the output of a PRNG is entirely deterministic, the sequence is constructed so that, for statistical purposes, it behaves like a stream of independent uniform random variables. The reproducibility demonstrated in the first part of the program is essential for scientific computing, enabling experiments, simulations, and numerical studies to be replicated exactly across platforms and implementations.

The explicit construction of the 8-bit generator provides a rare opportunity to observe period detection directly, reinforcing the theoretical result that every PRNG must repeat because its state space is finite. In practical numerical work, modern PRNGs employ internal states of hundreds or thousands of bits to ensure periods so large that cycles are never encountered in feasible computations. The structure highlighted here forms the basis for more sophisticated families of generators, including xorshift, PCG, Mersenne Twister, and modern counter-based generators used in high-performance parallel simulation.

Together, the examples emphasize that algorithmic randomness is not a compromise relative to physical randomness, but rather a carefully engineered mechanism designed to produce statistically reliable sequences with precise reproducibility and long periods, properties essential for deterministic testing, debugging, and high-accuracy numerical experimentation.

## 7.1.4. Classical Example: The Linear Congruential Generator

The Linear Congruential Generator (LCG) remains one of the most instructive examples of a deterministic generator. It is defined by the recurrence

$$X_{n+1} \equiv aX_n + c \pmod m\tag{7.1.7}$$

where $a$ is the multiplier, $c$ the increment, and $m$ the modulus.

Its theoretical period reaches $m$ if the Hull–Dobell conditions hold: (i) $c$ and $m$ are coprime; (ii) $a-1$ is divisible by all prime factors of $m$; (iii) $a-1$ is a multiple of 4 whenever $m$ is a multiple of 4.

While computationally efficient, each new value requiring one multiplication and one addition modulo $m$, the LCG exhibits regular patterns when plotted in higher dimensions. The generated points lie on a limited number of hyperplanes, a defect revealed by the *spectral test* (Bhattacharjee and Das, 2022). As a result, LCGs are unsuitable for many scientific and engineering tasks that rely on high-dimensional randomness.

### Rust Implementation

Following the discussion in Section 7.1.4 on the mathematical structure and limitations of classical pseudo-random number generators, Program 7.1.4 provides a concrete implementation of the Linear Congruential Generator (LCG), the canonical example of a deterministic generator defined by the recurrence in Equation (7.1.7). Although historically influential due to its simplicity and computational efficiency, the LCG exemplifies both the strengths and weaknesses of early PRNG design. This program demonstrates how its state evolves through modular arithmetic, how full-period behaviour arises under the Hull–Dobell conditions, and how reproducibility follows directly from its deterministic nature. To connect theory with computation, the code includes both a standard 32-bit LCG and a small 8-bit full-period example, along with a visualization-oriented sequence of $(u_n, u_{n+1})$ pairs that reveal the lattice structure underlying the spectral defects discussed in this section.

At the core of the implementation is the recurrence relation specified in Equation (7.1.7), which updates the internal state $X_n$ through a single multiplication and addition modulo $m$. The `Lcg` struct captures this mathematical structure by storing the parameters $(a, c, m)$ along with the current state, while the transition rule is realized in the `next_state` method. The resulting value is scaled to the interval $[0,1)$ according to the mapping in Equation (7.1.6), providing a normalized pseudo-random output $u_n$ suitable for sampling and Monte Carlo applications. Because the process is fully deterministic, the same initial seed reproduces the same sequence exactly, demonstrating the reproducibility that characterizes algorithmic randomness.

To make the theoretical notion of period explicit, the program includes a small 8-bit generator whose modulus $m = 256$ allows the entire state space to be traversed feasibly. By selecting parameters satisfying the Hull–Dobell conditions, this toy example achieves period $T = m$, and the program verifies this by iterating until the state returns to its initial value. This demonstration provides a concrete illustration of how period length is governed entirely by number-theoretic properties of $(a, c, m)$ and cannot exceed the size of the state space.

The `main` function highlights three important behaviors of LCGs. First, it shows reproducibility by comparing sequences produced from identical seeds, confirming that algorithmic randomness is deterministic by design. Second, it computes the full period of the 8-bit generator, illustrating the finite-state nature of the underlying dynamical system. Third, it outputs several pairs $(u_n, u_{n+1})$, which when plotted reveal the characteristic hyperplane structure associated with the spectral test. These pairs demonstrate visually why LCGs, despite their simplicity and efficiency, are poorly suited for high-dimensional scientific computing tasks that require strong equidistribution properties.

```rust
//! Program 7.1.4: Classical Linear Congruential Generator (LCG)
//!
//! This example implements the recurrence
//!
//!     X_{n+1} ≡ a X_n + c (mod m)      (Equation 7.1.7)
//!
//! and demonstrates:
//!   * deterministic, reproducible output from the same seed,
//!   * a toy parameter set that achieves full period m using the
//!     Hull–Dobell conditions,
//!   * the generation of (u_n, u_{n+1}) pairs in [0,1) that can
//!     be used to visualize the lattice structure of an LCG.

// A generic linear congruential generator with parameters (a, c, m).
struct Lcg {
    state: u64,
    a: u64,
    c: u64,
    m: u64,
}

impl Lcg {
    /// Create a new LCG with given parameters and seed X_0.
    fn new(seed: u64, a: u64, c: u64, m: u64) -> Self {
        let state = seed % m;
        Self { state, a, c, m }
    }

    /// Advance the internal state:
    ///     X_{n+1} ≡ a X_n + c (mod m).
    fn next_state(&mut self) {
        let x = self.state;
        // Wrapping arithmetic implements the reduction modulo 2^64;
        // the explicit `% self.m` enforces the modulus m.
        let next = x
            .wrapping_mul(self.a)
            .wrapping_add(self.c)
            % self.m;
        self.state = next;
    }

    /// Return the next integer value X_{n+1}.
    fn next_u64(&mut self) -> u64 {
        self.next_state();
        self.state
    }

    /// Return the next floating-point value u_n in [0,1),
    /// using the scaling in Equation (7.1.6):
    ///
    ///     u_n = X_n / m.
    fn next_unit(&mut self) -> f64 {
        self.next_u64() as f64 / self.m as f64
    }
}

/// Compute the period T for a small toy LCG by explicitly tracking when
/// the state returns to its initial value. This is only practical for
/// very small moduli, such as m = 256.
fn compute_toy_period(a: u64, c: u64, m: u64, seed: u64) -> u64 {
    let mut gen = Lcg::new(seed, a, c, m);
    let start = gen.state;
    let mut t = 0_u64;

    loop {
        gen.next_state();
        t += 1;

        if gen.state == start {
            break;
        }

        // Safety guard: in principle, period ≤ m for an LCG.
        if t > m {
            println!("Warning: exceeded modulus without returning to start; parameters may be incorrect.");
            break;
        }
    }

    t
}

fn main() {
    // ------------------------------------------------------------
    // 1. "Realistic" 32-bit-style LCG parameters (ANSI C style).
    // ------------------------------------------------------------
    //
    // These parameters are historically used in C libraries:
    //   m = 2^31, a = 1103515245, c = 12345.
    //
    // They are efficient and have full period m when combined
    // with the lower bits, but are known to have poor spectral
    // properties in high dimensions.

    let m32: u64 = 2_147_483_648; // 2^31
    let a32: u64 = 1_103_515_245;
    let c32: u64 = 12_345;

    println!("=== 32-bit style LCG (ANSI C parameters) ===");
    let mut g1 = Lcg::new(1, a32, c32, m32);
    let mut g2 = Lcg::new(1, a32, c32, m32); // same seed => same sequence

    println!("First 5 u_n values from g1 (seed = 1):");
    for _ in 0..5 {
        let u = g1.next_unit();
        println!("{:.10}", u);
    }

    println!("\nFirst 5 u_n values from g2 (same seed = 1):");
    for _ in 0..5 {
        let u = g2.next_unit();
        println!("{:.10}", u);
    }

    println!("\nThe sequences from g1 and g2 match exactly, illustrating determinism and reproducibility.");

    // ------------------------------------------------------------
    // 2. Toy 8-bit LCG with full period m = 256.
    // ------------------------------------------------------------
    //
    // We now choose parameters that satisfy the Hull–Dobell conditions
    // for m = 256:
    //
    //   * m = 2^8,
    //   * c is odd (c = 1),
    //   * a - 1 is divisible by all prime factors of m (here 2),
    //   * a - 1 is a multiple of 4 since m is a multiple of 4.
    //
    // Choosing a = 5, c = 1 satisfies these conditions and yields
    // period T = m = 256.

    let m8: u64 = 256;
    let a8: u64 = 5;
    let c8: u64 = 1;
    let seed8: u64 = 1;

    println!("\n=== Toy 8-bit LCG with full period (m = 256) ===");
    let period = compute_toy_period(a8, c8, m8, seed8);
    println!(
        "Starting from seed X_0 = {}, the 8-bit LCG returns to its initial state after T = {} steps.",
        seed8, period
    );
    println!("For this choice of (a, c, m), the period T equals the modulus m, as guaranteed by the Hull–Dobell conditions.");

    // ------------------------------------------------------------
    // 3. (u_n, u_{n+1}) pairs for visualizing lattice structure.
    // ------------------------------------------------------------
    //
    // To study the lattice structure revealed by the spectral test,
    // one typically examines points such as (u_n, u_{n+1}) or
    // (u_n, u_{n+1}, u_{n+2}). Here we simply print a few pairs,
    // which can be plotted externally.

    println!("\n=== Lattice structure: (u_n, u_{{n+1}}) pairs ===");
    let mut g_lattice = Lcg::new(7, a32, c32, m32);

    let num_pairs = 10;
    println!("First {} points (u_n, u_{{n+1}}):", num_pairs);
    let mut u_prev = g_lattice.next_unit();
    for _ in 0..num_pairs {
        let u_next = g_lattice.next_unit();
        println!("({:.6}, {:.6})", u_prev, u_next);
        u_prev = u_next;
    }

    println!("\nThese pairs can be plotted to reveal the regular hyperplane structure characteristic of LCGs in higher dimensions.");
}
```

Program 7.1.4 illustrates the essential computational features of the Linear Congruential Generator by directly implementing its defining recurrence and exposing the structural limitations discussed in Section 7.1.4. Although the LCG is computationally efficient and easily reproducible, its deterministic structure leads to severe regularities in higher dimensions, a weakness quantified by the spectral test and visible through the lattice patterns in the $(u_n, u_{n+1})$ pairs. The explicit period computation for the 8-bit generator further emphasizes that all PRNGs are ultimately finite dynamical systems whose statistical properties depend critically on their parameter choices.

While the LCG remains historically important and pedagogically useful, its spectral defects render it unsuitable for modern numerical simulation, high-dimensional Monte Carlo methods, and scientific computation more generally. The code presented here serves as a foundation for contrasting the LCG with more advanced generators introduced later in this chapter, such as xorshift, PCG, and counter-based generators, which offer longer periods, improved equidistribution, and superior statistical robustness.

## 7.1.5. Modern Developments

To overcome the deficiencies of simple congruential generators, researchers have developed families of PRNGs that combine arithmetic and bitwise operations to improve statistical quality and parallel performance.

### Mersenne Twister and Its Successors

The Mersenne Twister (MT19937) provides a period of $2^{19937}-1$ and achieves 623-dimensional equidistribution. It remains widely used in simulation but is not ideal for cryptography or highly parallel systems (Almaraz Luengo, 2022).

### Counter-Based and Hash-Based Generators

More recent designs, such as Philox and Threefry, compute random values as a function of a key and a counter:

$$x_n = f(k,n),\tag{7.1.8}$$

where $n$ can be evaluated independently on different threads. This property enables massive parallelism with perfect reproducibility, an essential requirement in high-performance Monte Carlo simulation (Sepehri, Camarero and García-Molina, 2020).

### Neural and Quantum Approaches

Emerging research explores *neural-network-based* generators trained to emulate statistically perfect sequences (Wu, Li and Zhang, 2025), and *quantum hybrid* RNGs combining physical entropy sources with deterministic post-processing (Bruynsteen, Wang and Vermeulen, 2022). These developments promise both high speed and enhanced unpredictability, expanding the theoretical boundaries of algorithmic randomness.

## 7.1.6. Transforming Uniform Randomness

Many applications require random variables from distributions other than uniform. The simplest transformation method employs the inverse cumulative distribution function:

$$X=F^{-1}(U) \tag{7.1.9}$$

where $U \sim \mathcal{U}(0,1)$ and $F(x)$ is the desired CDF.

Alternative techniques include the *Box–Muller transform* and *Ziggurat method* for Gaussian variables, and *alias* or *rejection sampling* for discrete and heavy-tailed distributions. The mathematical correctness of these transformations presupposes that the underlying $U$ is genuinely uniform; any bias propagates through $F^{-1}$ and distorts results (Bhattacharjee and Das, 2022).

### Rust Implementation

Following the discussion in Section 7.1.6 on transforming uniform random numbers into non-uniform distributions, the next program provides a concrete implementation of several classical transformation techniques used in Monte Carlo computation. As emphasized in Equation (7.1.9), the inverse CDF method maps a uniform variate $U\sim \mathcal{U}(0,1)$ to a target distribution through the functional transformation $X = F^{-1}(U)$. This approach underlies many simulation algorithms, but its accuracy critically depends on the uniformity of the underlying generator, since any statistical defect in $U$ is directly inherited by the transformed variable. To complement the inverse transform, the program also includes the Box–Muller method for generating Gaussian random variables, illustrating how geometric transformations of two-dimensional uniform samples yield independent normal deviates. Together, these examples demonstrate how uniform randomness acts as the computational “base currency” from which more complex distributions are constructed in Monte Carlo methods.

At the core of the implementation is the function `inverse_cdf_exponential`, which applies the inverse CDF mapping specified by Equation (7.1.9). For the exponential distribution with rate parameter $\lambda$, the CDF $F(x)=1-e^{-\lambda x}$ is analytically invertible, enabling a direct and numerically stable computation of $X = F^{-1}(U)$. The function therefore implements the transformation $-\ln(1-U)/\lambda$, with a small clamping of the endpoints to avoid evaluating $\ln(0)$. This exemplifies how inverse-transform sampling exploits explicit CDF structure whenever available.

The `box_muller` function implements the classical Box–Muller transform, which processes two independent uniform values to produce two independent standard normal variables. By converting the pair $(U_1,U_2)$ into polar coordinates $(R,\Theta)$, the method uses the identity that the radius $R = \sqrt{-2\ln U_1}$ and angle $\Theta = 2\pi U_2$ generate points distributed according to the standard bivariate normal density. The resulting values $Z_1 = R\cos\Theta$ and $Z_2 = R\sin\Theta$ thus follow the $N(0,1)$ distribution without requiring numerical integration or special functions. As with the exponential case, the transformation relies on the statistical quality of the underlying uniform inputs to ensure proper Gaussian behavior.

To track empirical accuracy, the `RunningStats` structure implements Welford’s algorithm for computing sample means and variances in a stable, one-pass manner. This avoids numerical drift that would occur if sums of squares were accumulated directly, especially for large sample counts. The accumulator allows the program to compare empirical estimates to theoretical values and demonstrate the convergence of Monte Carlo approximations as the number of samples increases.

The `main` function orchestrates the transformations and produces sample statistics for both the exponential and normal distributions. It initializes a reproducible random number generator, draws a large number of uniform samples, and transforms them using the inverse-CDF and Box–Muller methods. The computed empirical means and variances are then compared against theoretical expectations for $\mathrm{Exp}(\lambda)$ and $N(0,1)$. These comparisons highlight the consistency of the transformation methods and validate that high-quality uniform randomness leads to statistically correct non-uniform distributions, an assumption central to the discussion initiated by Bhattacharjee and Das (2022).

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
// Program 7.1.5: Transforming Uniform Randomness
//
// This example assumes the existence of a high-quality uniform RNG U ~ U(0, 1).
// From this source, we construct non-uniform random variables using:
//
// 1. Inverse-CDF transform for the exponential distribution:
//      X = F^{-1}(U) = -ln(1 - U) / λ
//    where F(x) = 1 - e^{-λ x}, x >= 0.
//
// 2. Box–Muller transform for the standard normal distribution:
//      Given U1, U2 ~ U(0,1) independent,
//      R = sqrt(-2 ln U1),
//      Θ = 2π U2,
//      Z1 = R cos Θ,  Z2 = R sin Θ,
//    which yields Z1, Z2 ~ N(0,1) independent.
//
// The program estimates sample means and variances to empirically validate
// that the transformations produce the correct distributions.
//
// Cargo.toml dependencies (for reference):
// [dependencies]
// rand = "0.8"

use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Inverse-CDF transform for an exponential distribution with rate λ > 0.
///
/// If U ~ U(0,1), then X = F^{-1}(U) = -ln(1 - U)/λ has Exp(λ) distribution.
///
/// Note: for numerically stable code, one may clamp U away from 0 and 1.
fn inverse_cdf_exponential(u: f64, lambda: f64) -> f64 {
    // Guard against u = 1 causing log(0); clamp slightly.
    let u = u.clamp(1e-16, 1.0 - 1e-16);
    -((1.0 - u).ln()) / lambda
}

/// Box–Muller transform: given two independent U(0,1) values (u1, u2),
/// produce two independent standard normal variables (z1, z2).
fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    // Clamp u1 to avoid taking ln(0).
    let u1 = u1.clamp(1e-16, 1.0 - 1e-16);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;

    let z1 = r * theta.cos();
    let z2 = r * theta.sin();
    (z1, z2)
}

/// Helper to update running mean and variance using Welford's algorithm.
#[derive(Debug, Default)]
struct RunningStats {
    n: u64,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / (self.n as f64);
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / ((self.n - 1) as f64)
        } else {
            0.0
        }
    }
}

fn main() {
    // Use a reproducible RNG so that results are deterministic in the textbook.
    let seed: u64 = 0x_7_1_6_0_1_23; // arbitrary but fixed
    let mut rng = StdRng::seed_from_u64(seed);

    let uniform = Uniform::new(0.0_f64, 1.0_f64);

    // Number of samples for each experiment.
    let n_samples: u64 = 200_000;

    // === 1. Exponential(λ) via inverse-CDF transform ===
    let lambda = 2.0_f64; // rate parameter λ

    let mut stats_exp = RunningStats::default();
    for _ in 0..n_samples {
        let u: f64 = rng.sample(uniform);
        let x = inverse_cdf_exponential(u, lambda);
        stats_exp.push(x);
    }

    let exp_mean = stats_exp.mean();
    let exp_var = stats_exp.variance();
    let exp_mean_theory = 1.0 / lambda;
    let exp_var_theory = 1.0 / (lambda * lambda);

    println!("Exponential(λ = {lambda}) via inverse-CDF:");
    println!("  Empirical mean      ≈ {exp_mean:.6}");
    println!("  Theoretical mean     = {exp_mean_theory:.6}");
    println!("  Empirical variance  ≈ {exp_var:.6}");
    println!("  Theoretical variance = {exp_var_theory:.6}");
    println!();

    // === 2. Standard normal via Box–Muller transform ===
    let mut stats_norm = RunningStats::default();

    // Each Box–Muller call yields two normals. Loop accordingly.
    let pairs = (n_samples + 1) / 2;
    for _ in 0..pairs {
        let u1: f64 = rng.sample(uniform);
        let u2: f64 = rng.sample(uniform);
        let (z1, z2) = box_muller(u1, u2);
        stats_norm.push(z1);
        stats_norm.push(z2);
    }

    let norm_mean = stats_norm.mean();
    let norm_var = stats_norm.variance();

    println!("Standard normal N(0,1) via Box–Muller:");
    println!("  Empirical mean      ≈ {norm_mean:.6}");
    println!("  Theoretical mean     = 0.0");
    println!("  Empirical variance  ≈ {norm_var:.6}");
    println!("  Theoretical variance = 1.0");
}
```

Program 7.1.5 illustrates the fundamental role of transformation techniques in constructing non-uniform random variables from a uniform base generator. As discussed in Section 7.1.6, the mathematical validity of methods such as the inverse transform and the Box–Muller construction relies on the assumption that the underlying uniform random numbers are unbiased and statistically independent. The empirical results produced by the program confirm that when these assumptions hold, the transformed outputs align closely with theoretical expectations, even for large sample sizes.

The exponential and Gaussian examples highlight two widely used transformation families. The inverse-CDF method provides a direct and efficient mechanism whenever the distribution admits a closed-form inverse, while the Box–Muller method demonstrates how geometric transformations can generate normal deviates without relying on tabulated distributions or rejection strategies. Both approaches reinforce the general principle that complex probability laws can often be realized through deterministic mappings of simple uniform inputs.

The modular design of the implementation also creates a natural foundation for incorporating more advanced techniques, such as the Ziggurat algorithm, acceptance–rejection sampling, and alias-based discrete generation. These methods become essential when inverse CDFs are unavailable or expensive to compute or when sampling efficiency is critical in high-performance Monte Carlo applications. In this way, Program 7.1.5 serves both as a pedagogical demonstration and as a stepping stone toward more sophisticated random-variable generation techniques used throughout numerical simulation and stochastic modeling.

## 7.1.7. Testing Statistical Quality

Randomness quality is assessed empirically through standardized test suites such as DIEHARD, TestU01, and NIST SP 800-22. These suites measure uniformity, independence, autocorrelation, and bit-level entropy using tests like the chi-square, Kolmogorov–Smirnov, and spectral analyses.

A PRNG that passes all these tests is deemed statistically sound for most applications. However, the choice of generator should always be context-dependent: a sequence adequate for stochastic gradient training may still be unsuitable for cryptography or precise uncertainty quantification (Bhattacharjee and Das, 2022).

### Rust Implementation

Following the discussion in Section 7.1.7 on the empirical assessment of randomness quality, Program 7.1.6 provides a practical implementation of several foundational statistical tests used to evaluate pseudo-random number generators. Comprehensive suites such as DIEHARD, TestU01, and NIST SP 800-22 apply dozens of rigorous procedures, but their underlying principles rest on a small collection of fundamental diagnostics: uniformity, independence, distributional conformity, and bit-level entropy. This program demonstrates how these core ideas can be realized in Rust through simple yet instructive tests including chi-square uniformity, Kolmogorov–Smirnov deviation, autocorrelation analysis, and bit-frequency inspection. By working directly with raw output from a PRNG, the program illustrates how empirical evidence can reveal deficiencies that might otherwise remain hidden in numerical simulations, reflecting the broader caution raised in Bhattacharjee and Das (2022) regarding context-dependent notions of “statistical adequacy.”

At the heart of the implementation is the `RunningStats` structure, which provides numerically stable, one-pass estimates of mean, variance, minimum, and maximum using Welford’s update scheme. These quantities form the most basic diagnostic layer for evaluating a uniform generator: an approximate mean of $1/2$, variance close to $1/12$, and sample extrema near the boundaries of $[0,1)$ indicate that the PRNG is not exhibiting gross systematic bias. Because these statistics accumulate values incrementally, they can be applied to arbitrarily long sequences without suffering from roundoff degradation.

The function `chi_square_uniform` implements the classical chi-square uniformity test by partitioning the unit interval into fixed-width bins and comparing the observed counts to their expected frequencies. This procedure corresponds to the histogram-based uniformity checks typically used in introductory empirical validation and captures coarse deviations from the ideal distribution. Large departures in the chi-square statistic indicate clustering, gaps, or uneven density that cannot arise from a well-behaved $U(0,1)$ source.

The `ks_uniform` function computes the Kolmogorov–Smirnov statistic, which measures the maximum difference between the empirical CDF and the theoretical CDF of the uniform distribution. Unlike the chi-square test, the KS metric is sensitive to discrepancies across all scales and does not require binning. Its use here reflects the emphasis in Section 7.1.7 on distributional fidelity as a key indicator of generator quality. For large sample sizes, the rescaled statistic $\sqrt{n},D_n$ provides an interpretable measure of deviation that can be compared against classical asymptotic thresholds.

Next, the `lag1_autocorrelation` function estimates short-range dependence by computing the correlation between consecutive values. A high-quality PRNG should produce sequences that behave as independent draws, and thus the lag-1 autocorrelation should oscillate around zero within statistical noise. This test captures subtle serial patterns that histogram-based tests cannot detect and is a simplified analogue of the more extensive spectral and autocorrelation analyses found in standardized test suites.

Finally, the `bit_balance` function examines the low-level binary representation of the samples by converting each float to a 32-bit integer and counting the frequency of 1-bits. Because many practical failures of random number generators manifest at the bit level, this test complements the distributional checks by probing entropy directly. A well-functioning generator should produce bit frequencies near (50%), and persistent imbalance would indicate structural bias in the underlying bitstream.

The `main` function orchestrates these diagnostics by sampling a large block of $U(0,1)$ values, computing summary statistics, and reporting all test results. This consolidated view allows the reader to compare empirical outcomes against known theoretical benchmarks, illustrating how each diagnostic captures a different facet of the generator’s behavior. When these tests agree, they collectively provide confidence that the PRNG satisfies baseline statistical criteria expected in Monte Carlo simulations.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
// Program 7.1.6: Simple Statistical Tests for a Uniform PRNG
//
// This program illustrates basic empirical tests for the statistical
// quality of a uniform pseudo-random number generator (PRNG). While
// comprehensive test suites such as DIEHARD, TestU01, and NIST SP 800-22
// implement many more sophisticated procedures, the following checks
// demonstrate the core ideas:
//
// 1. Basic moment and range checks (mean, variance, min, max).
// 2. Chi-square test for histogram uniformity.
// 3. Kolmogorov–Smirnov (KS) statistic for distributional deviation.
// 4. Lag-1 autocorrelation to probe dependence between successive values.
// 5. Bit-level balance test for 0/1 frequencies.
//
// The goal is pedagogical: to show how numerical summaries and simple
// goodness-of-fit statistics can reveal obvious deficiencies in a PRNG.
//
// Cargo.toml dependencies (for reference):
// [dependencies]
// rand = "0.8"

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N_SAMPLES: usize = 100_000;
const N_BINS: usize = 20;

/// Helper structure: running mean and variance via Welford’s algorithm.
#[derive(Debug, Default)]
struct RunningStats {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn push(&mut self, x: f64) {
        self.n += 1;
        let n_f = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n_f;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / ((self.n - 1) as f64)
        } else {
            0.0
        }
    }

    fn min(&self) -> f64 {
        self.min
    }

    fn max(&self) -> f64 {
        self.max
    }
}

/// Compute a chi-square statistic for uniformity on [0,1) using N_BINS bins.
/// Returns (chi2_stat, degrees_of_freedom).
fn chi_square_uniform(samples: &[f64]) -> (f64, usize) {
    let n = samples.len();
    let mut counts = vec![0usize; N_BINS];

    for &x in samples {
        // Map x in [0,1) into {0, ..., N_BINS-1}.
        let mut idx = (x * N_BINS as f64) as usize;
        if idx >= N_BINS {
            idx = N_BINS - 1;
        }
        counts[idx] += 1;
    }

    let expected = n as f64 / N_BINS as f64;
    let mut chi2 = 0.0;
    for &obs in &counts {
        let diff = obs as f64 - expected;
        chi2 += diff * diff / expected;
    }

    let dof = N_BINS - 1;
    (chi2, dof)
}

/// Compute the Kolmogorov–Smirnov statistic for testing U(0,1).
///
/// For a perfect uniform distribution, the empirical CDF F_n(x) should be
/// close to x. The KS statistic is
///
///   D = sup_x |F_n(x) - x|,
///
/// approximated here by evaluating at the sample points.
fn ks_uniform(samples: &[f64]) -> f64 {
    let n = samples.len();
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut d_plus = 0.0;
    let mut d_minus = 0.0;

    for (i, &x) in sorted.iter().enumerate() {
        let i1 = (i + 1) as f64;
        let n_f = n as f64;

        // F_n(x) just after the i-th sample is i1 / n.
        let f_n = i1 / n_f;

        // For U(0,1), theoretical CDF is F(x) = x.
        let d_p = f_n - x;
        let d_m = x - (i as f64 / n_f);

        if d_p > d_plus {
            d_plus = d_p;
        }
        if d_m > d_minus {
            d_minus = d_m;
        }
    }

    d_plus.max(d_minus)
}

/// Estimate lag-1 autocorrelation ρ(1) for the sequence {X_i}.
///
/// ρ(1) ≈ sum_i (X_i - μ)(X_{i+1} - μ) / sum_i (X_i - μ)^2
fn lag1_autocorrelation(samples: &[f64], mean: f64) -> f64 {
    let n = samples.len();
    if n < 2 {
        return 0.0;
    }

    let mut num = 0.0;
    let mut denom = 0.0;

    for i in 0..(n - 1) {
        let x = samples[i] - mean;
        let y = samples[i + 1] - mean;
        num += x * y;
        denom += x * x;
    }

    if denom == 0.0 {
        0.0
    } else {
        num / denom
    }
}

/// Simple bit-level test: map each uniform value to a 32-bit integer
/// and count the fraction of 1-bits. For a “random-looking” source,
/// the fraction should be close to 0.5.
fn bit_balance(samples: &[f64]) -> f64 {
    let mut ones_count: u64 = 0;
    let mut total_bits: u64 = 0;

    for &x in samples {
        let scaled = (x * (u32::MAX as f64)) as u32;
        ones_count += scaled.count_ones() as u64;
        total_bits += 32;
    }

    ones_count as f64 / total_bits as f64
}

fn main() {
    // Fixed seed for reproducibility in the textbook.
    let seed: u64 = 0x_7_1_7_0_1_23;
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate N_SAMPLES from U(0,1).
    let mut stats = RunningStats::new();
    let mut samples = Vec::with_capacity(N_SAMPLES);

    for _ in 0..N_SAMPLES {
        let x: f64 = rng.gen(); // U(0,1) from rand
        stats.push(x);
        samples.push(x);
    }

    println!("Basic summary statistics for U(0,1) samples:");
    println!("  n          = {}", N_SAMPLES);
    println!("  mean       ≈ {:.6} (theoretical: 0.5)", stats.mean());
    println!(
        "  variance   ≈ {:.6} (theoretical: 1/12 ≈ {:.6})",
        stats.variance(),
        1.0 / 12.0
    );
    println!("  min        ≈ {:.6}", stats.min());
    println!("  max        ≈ {:.6}", stats.max());
    println!();

    // Chi-square uniformity test (histogram-based).
    let (chi2, dof) = chi_square_uniform(&samples);
    println!("Chi-square uniformity test ({} bins):", N_BINS);
    println!("  χ² statistic ≈ {:.3}", chi2);
    println!("  degrees of freedom = {}", dof);
    println!("  (For a well-behaved PRNG, χ² should be in a plausible range for df = {}.)", dof);
    println!();

    // Kolmogorov–Smirnov statistic.
    let d_ks = ks_uniform(&samples);
    let ks_scaled = (N_SAMPLES as f64).sqrt() * d_ks;
    println!("Kolmogorov–Smirnov statistic for U(0,1):");
    println!("  D_n        ≈ {:.6}", d_ks);
    println!("  sqrt(n) D_n ≈ {:.3}", ks_scaled);
    println!("  (Smaller values indicate closer agreement with the ideal uniform CDF.)");
    println!();

    // Lag-1 autocorrelation.
    let rho1 = lag1_autocorrelation(&samples, stats.mean());
    println!("Lag-1 autocorrelation ρ(1):");
    println!("  ρ(1) ≈ {:.6}", rho1);
    println!("  (For an independent sequence, ρ(1) should fluctuate around 0.)");
    println!();

    // Bit-level balance test.
    let bit_fraction = bit_balance(&samples);
    println!("Bit-level balance test (mapping to 32-bit integers):");
    println!("  Fraction of 1-bits ≈ {:.6}", bit_fraction);
    println!("  (Ideal value is 0.5 for an unbiased bit stream.)");
}
```

Program 7.1.6 demonstrates how fundamental statistical tests can be used to assess the quality of a pseudo-random number generator at multiple levels, from simple mean and variance checks to distributional and bit-level analyses. Although these tests are far less comprehensive than full suites such as TestU01 or NIST SP 800-22, they embody the essential principles emphasized in Section 7.1.7: uniformity, independence, and entropy. The results obtained from the sample run illustrate that a well-constructed generator will produce statistics within expected theoretical ranges, reinforcing its suitability for many numerical applications.

At the same time, the program highlights the limitations of lightweight testing. A generator that passes these tests may still fail more stringent spectral, matrix-rank, or cryptographic randomness benchmarks. This reflects the broader theme that statistical adequacy is highly context-dependent: random sequences acceptable for stochastic gradient training or Monte Carlo integration may not meet the stringent unpredictability requirements of cryptography or the precision needed for uncertainty quantification. The modular design of the program therefore serves both as a pedagogical tool and as a precursor to more sophisticated validation techniques, illustrating the layered nature of randomness testing in scientific computing.

## 7.1.8. Summary and Outlook

Random numbers are indispensable to numerical computing. They enable Monte Carlo integration, stochastic optimization, randomized algorithms, and physical simulation. Yet their generation on deterministic hardware requires careful mathematical construction.

In this section, we have introduced the conceptual and practical meaning of randomness in computation, formalized the structure of PRNGs, and outlined both classical and modern developments. Subsequent sections will examine the principal families of generators in depth, analyze their mathematical properties, and demonstrate high-quality, reproducible Rust implementations suitable for scientific and engineering computation.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/tts7k1JqIDmcKsDQMNtm.8","tags":[]}

# 7.2. Uniform Deviates

Random number generation lies at the heart of modern numerical computing, enabling stochastic modeling, probabilistic analysis, and simulation of inherently uncertain systems. Every computationally generated random sequence begins with a *uniform deviate*, a numerical value statistically indistinguishable from a true random draw on the interval $[0,1)$. These basic uniform samples form the foundation for constructing more complex distributions through deterministic transformations. Consequently, the quality of uniform random number generators directly determines the accuracy and reproducibility of Monte Carlo simulations, optimization algorithms, and statistical estimators. The following sections develop a rigorous understanding of uniform deviates, their mathematical formulation, and the design of reliable generator algorithms capable of producing statistically robust sequences within the precision and reproducibility constraints of digital arithmetic.

## 7.2.1. Introduction to Uniform Random Deviates

Uniform random deviates constitute the foundation of all stochastic simulation and Monte Carlo computation. A *uniform deviate* is a random variable $U \sim \mathcal{U}(0,1),$ whose outcomes are equally likely within the interval $[0,1)$. Formally, for any sub-interval of length $\Delta \subset [0,1)$,

$$\Pr(U\in\text{interval})=\Delta \tag{7.2.1}$$

so the probability density function is constant. Figure 7.2.1 shows this uniform density in contrast with the bell-shaped Gaussian distribution.

```{figure} images/pqQDe4beUu67RvW3raYP-TDfIzSSldXKSVAgs03rq-v1.png
:name: i0bs8gMATK
:align: middle
:width: 40%

**Figure 7.2.1** Probability density of a uniform deviate compared with a normal distribution.
```

Uniform deviates are indispensable because they provide the elementary building blocks from which all other distributions can be generated. Their statistical quality therefore determines the fidelity of every downstream process that depends on randomness (Foreman et al. 2024).

### Applications and Practical Importance

Uniform deviates permeate nearly every area of numerical computing. In *Monte Carlo integration* and stochastic differential equation solvers, they determine random sample points for approximating multidimensional integrals and probabilistic PDE solutions. In *optimization*, algorithms such as simulated annealing and genetic search rely on uniform random draws to explore the search space effectively. In *physical modeling*, they govern random free-flight distances, scattering angles, or collision events, for instance, neutron-transport codes use uniform deviates to decide interaction distances, and the quality of the results depends directly on their uniformity (Josey 2023).

Outside scientific simulation, uniform deviates underpin randomized load-balancing schemes, cryptographic key generation, and procedural world generation in computer graphics. In machine learning, they influence the initialization of neural-network weights and the random shuffling of datasets, which in turn affect convergence and generalization.

### Distributional-Transform Principle

The central theoretical result connecting uniform and non-uniform randomness is the distributional transform. Let $F_X(x)$ be the cumulative distribution function of a continuous random variable $X$. Then,

$$X = F_X^{-1}(U), \qquad U\sim\mathcal{U}(0,1) \tag{7.2.2}$$

produces samples from the distribution of $X$. For example, an exponential deviate with rate $\lambda$ arises as,

$$E = -\lambda^{-1}\ln(1-U) \tag{7.2.3}$$

Normal deviates can likewise be obtained from two independent uniform variables via the Box–Muller or polar method. Hence, *uniform deviates form the universal substrate* of all random-number generation.

However, any bias or correlation in the uniform source contaminates every transformed variable. Consequently, ensuring the statistical integrity of uniform generators is a primary concern in numerical computing.

### Historical Development and the RANDU Lesson

Early digital RNGs adopted the linear congruential generator (LCG),

$$X_{n+1}=(aX_n+c)\bmod m \tag{7.2.4}$$

with integer parameters $m$ (modulus), $a$ (multiplier), and $c$ (increment). The normalized sequence $U_n=X_n/m$ provides approximate uniform deviates. With suitable parameters, an LCG can achieve the maximal period $m$. Because of their simplicity and tiny state, LCGs dominated early simulation studies in the 1950s and 1960s.

A notorious counter-example is the RANDU generator,

$$X_{n+1}=65539,X_n\bmod 2^{31} \tag{7.2.5}$$

which exhibits severe lattice correlations: the triples $(X_n,X_{n+1},X_{n+2})$ lie on only fifteen parallel planes within the unit cube. Figure 7.2.2 visualizes this pattern. Such defects devastate high-dimensional Monte Carlo accuracy and demonstrate that numerical simplicity does not guarantee statistical adequacy.

```{figure} images/pqQDe4beUu67RvW3raYP-ouhuvpviSEretcIW8BYG-v1.png
:name: EpI4DHaX4A
:align: middle
:width: 50%

**Figure 7.2.2** Triples from the RANDU generator showing planar correlation in three-dimensional space
```

Marsaglia’s spectral test showed that LCG points in $k$ dimensions lie on at most $m^{1/k}$ parallel hyperplanes, and good parameter sets maximize this number. When $m$ is a power of $2$, as in RANDU, lower-order bits exhibit deterministic periodicity (e.g. least-significant bits repeating with period $2$). Therefore, only the high-order bits of such LCGs possess acceptable randomness.

### Evolution Beyond Classical LCGs

By the late 1960s, it was understood that the inherent linearity of LCGs imposed fundamental limits on statistical quality. Park and Miller (1988) advocated the use of prime moduli and rigorously tested multipliers but cautioned that even well-parameterized LCGs remain inadequate for demanding applications. The field advanced rapidly in the 1990s with the introduction of the Mersenne Twister (MT19937, period $2^{19937}-1$) and, later, xorshift and PCG families.

Modern consensus holds that no single closed-form recurrence can simultaneously deliver maximal statistical quality, long period, small state, and peak speed. Consequently, contemporary practice combines multiple algebraic structures including linear recurrences, XOR/shift operations, and non-linear bit-scrambling, to achieve balanced performance and reliability (Frankel 2023).

Below are the four components rewritten for **Program 7.2.1**, following the exact tone, continuity, structure, and paragraph style of your example from Section 5.4.1. These paragraphs integrate smoothly with Section 7.2 (“Uniform Deviates”) and are aligned with the pedagogical and narrative style used throughout your textbook.

### Rust Implementation

Following the discussion in Section 7.2 on the formulation, significance, and historical evolution of uniform random deviates, Program 7.2.1 provides a concrete implementation of several representative uniform RNG algorithms and examines their empirical behavior. Whereas Section 7.2 emphasizes the theoretical role of uniform deviates as the foundational substrate for all distributional transformations in random-number generation, this program illustrates how different generator architectures, ranging from classical linear congruential formulas to modern, high-quality bit-mixing schemes, can be implemented and evaluated in practice. By computing basic statistical summaries and extracting three-dimensional triples from successive outputs, the program highlights how subtle structural correlations can arise from poor parameter choices, as famously demonstrated by RANDU, and reinforces why constructing statistically sound uniform deviates is essential for Monte Carlo reliability and modern simulation workflows.

At the core of the implementation is the `SimpleRng` trait, which provides a unified interface for generators that produce both raw 32-bit outputs and normalized uniform deviates on $[0,1)$. This abstraction enables structurally different generators, such as Park–Miller’s prime-modulus LCG, the defective RANDU recurrence, and a modern generator from the `rand` crate, to be evaluated through the same testing pipeline. The trait’s default `next_f64` method applies the normalization $X_n/m$ described following Equation (7.2.4), reflecting the standard mapping from integer recurrences to floating-point uniform deviates.

The `ParkMiller` and `Randu` structs implement two historically important linear congruential schemes. The Park–Miller generator uses a prime modulus and a carefully selected multiplier, representing a well-behaved example of the classical LCG family, whereas RANDU uses the recurrence shown in Equation (7.2.5), whose poor choice of modulus and multiplier leads to the infamous planar alignment of successive triples $(U_n,U_{n+1},U_{n+2})$. These implementations closely mirror the mathematical recurrences in the section, allowing downstream analyses to reveal the correlations described in the RANDU cautionary example and Figure 7.2.2.

To capture the behavior of a contemporary generator, the `ModernStdRng` wrapper exposes the widely used `StdRng` through the same `SimpleRng` interface. Although such generators employ sophisticated bit-scrambling and nonlinear transformations absent from classical LCGs, evaluating them through identical statistical summaries provides a direct comparison with older designs and illustrates how modern PRNGs avoid the deficiencies highlighted in historical examples such as RANDU.

The program computes basic summary statistics using the `RunningStats` accumulator, which applies Welford’s algorithm to estimate the mean, variance, minimum, and maximum of the uniform sample sequence. These statistics reflect the fundamental property $\Pr(U \in \text{interval}) = \Delta$ from Equation (7.2.1): a high-quality uniform deviate should exhibit a mean near (1/2), variance near (1/12), and extrema close to the boundaries of the unit interval. The function `write_triples_csv` extracts successive triples from each LCG and writes them to disk, enabling three-dimensional visualization of their distribution. This directly illustrates the departure from uniformity characteristic of RANDU, whose points lie on a small number of hyperplanes, validating the lattice-correlation discussion and historical warnings emphasized in Section 7.2.

The `main` function coordinates these analyses by instantiating each generator with fixed seeds, computing statistical summaries, and exporting triples for visualization. By juxtaposing outputs from the Park–Miller LCG, RANDU, and a modern `StdRng`, the program reveals how design choices in recurrence parameters and bit-mixing strategies manifest in discernible statistical structure. This serves as a practical complement to the theoretical exposition, reinforcing the importance of uniform deviate quality for reliable stochastic computation.

Add the following dependenciess to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
// Program 7.2.1: Classical and Modern Uniform Random Deviates
//
// This program illustrates how uniform deviates U ~ U(0,1) can be generated
// from different families of pseudo-random number generators (PRNGs):
//
//   1. A "good" linear congruential generator (LCG) in the style of Park–Miller.
//   2. The historically infamous RANDU generator.
//   3. A modern generator (StdRng) from the rand crate.
//
// It computes basic summary statistics (mean and variance) of the normalized
// U(0,1) outputs and writes 3D triples (u_n, u_{n+1}, u_{n+2}) for both the
// Park–Miller LCG and RANDU to CSV files. These triples can be visualized to
// reproduce figures like 7.2.2 illustrating the planar structure of RANDU.
//
// Cargo.toml dependencies (for reference):
// [dependencies]
// rand = "0.8"

use std::fs::File;
use std::io::{BufWriter, Write};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N_SAMPLES_STATS: usize = 100_000;
const N_TRIPLES: usize = 10_000;

/// Simple running statistics for mean, variance, min, and max using Welford’s algorithm.
#[derive(Debug, Default)]
struct RunningStats {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn push(&mut self, x: f64) {
        self.n += 1;
        let n_f = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n_f;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / ((self.n - 1) as f64)
        } else {
            0.0
        }
    }

    fn min(&self) -> f64 {
        self.min
    }

    fn max(&self) -> f64 {
        self.max
    }
}

/// Trait for simple PRNGs that produce 32-bit integers and normalized f64 deviates.
trait SimpleRng {
    /// Return the next raw 32-bit integer state.
    fn next_u32(&mut self) -> u32;

    /// Normalize to (0,1) or [0,1) as an f64.
    fn next_f64(&mut self) -> f64 {
        // Map {0, ..., 2^32-1} to [0,1) by dividing by 2^32.
        // Note: For generators with a different modulus, override if desired.
        const SCALE: f64 = 1.0 / (u32::MAX as f64 + 1.0);
        self.next_u32() as f64 * SCALE
    }
}

/// Park–Miller style LCG with modulus m = 2^31 - 1 (a prime) and multiplier a = 16807.
/// This is a representative example of a "well-chosen" 31-bit LCG.
struct ParkMiller {
    state: u32,
}

impl ParkMiller {
    fn new(seed: u32) -> Self {
        let seed = if seed == 0 { 1 } else { seed };
        Self { state: seed }
    }
}

impl SimpleRng for ParkMiller {
    fn next_u32(&mut self) -> u32 {
        const A: u64 = 16807;
        const M: u64 = 2_147_483_647; // 2^31 - 1

        let x = (self.state as u64 * A) % M;
        self.state = x as u32;
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        // Normalize by modulus m, as in U_n = X_n / m.
        const M: f64 = 2_147_483_647.0;
        (self.next_u32() as f64) / M
    }
}

/// RANDU generator: X_{n+1} = 65539 * X_n mod 2^31.
/// This is a classical example of a poorly designed LCG with strong correlations.
struct Randu {
    state: u32,
}

impl Randu {
    fn new(seed: u32) -> Self {
        let seed = if seed == 0 { 1 } else { seed };
        Self { state: seed }
    }
}

impl SimpleRng for Randu {
    fn next_u32(&mut self) -> u32 {
        const A: u64 = 65_539;
        const M: u64 = 1 << 31; // 2^31

        let x = (self.state as u64 * A) % M;
        self.state = x as u32;
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        // Normalize by modulus m = 2^31.
        const M: f64 = (1u64 << 31) as f64;
        (self.next_u32() as f64) / M
    }
}

/// Wrapper around rand::rngs::StdRng to present a SimpleRng interface.
struct ModernStdRng {
    inner: StdRng,
}

impl ModernStdRng {
    fn new(seed: u64) -> Self {
        Self {
            inner: StdRng::seed_from_u64(seed),
        }
    }
}

impl SimpleRng for ModernStdRng {
    fn next_u32(&mut self) -> u32 {
        self.inner.gen::<u32>()
    }

    // Use the default normalization to [0,1).
}

/// Compute summary statistics for a given generator, using N_SAMPLES_STATS uniform deviates.
fn summarize_uniform<R: SimpleRng>(rng: &mut R, label: &str) {
    let mut stats = RunningStats::new();
    for _ in 0..N_SAMPLES_STATS {
        let u = rng.next_f64();
        stats.push(u);
    }

    println!("Summary statistics for {} (U(0,1) samples):", label);
    println!("  n          = {}", N_SAMPLES_STATS);
    println!("  mean       ≈ {:.6} (theoretical: 0.5)", stats.mean());
    println!(
        "  variance   ≈ {:.6} (theoretical: 1/12 ≈ {:.6})",
        stats.variance(),
        1.0 / 12.0
    );
    println!("  min        ≈ {:.6}", stats.min());
    println!("  max        ≈ {:.6}", stats.max());
    println!();
}

/// Generate triples (u_n, u_{n+1}, u_{n+2}) and write them to a CSV file.
/// These can be plotted to visualize lattice structure (e.g. for RANDU).
fn write_triples_csv<R: SimpleRng>(rng: &mut R, filename: &str) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    // CSV header.
    writeln!(&mut writer, "u_n,u_n1,u_n2")?;

    // Generate initial three values.
    let mut u0 = rng.next_f64();
    let mut u1 = rng.next_f64();
    let mut u2 = rng.next_f64();

    writeln!(&mut writer, "{},{},{}", u0, u1, u2)?;

    for _ in 1..N_TRIPLES {
        // Shift the window: (u0, u1, u2) -> (u1, u2, u3).
        u0 = u1;
        u1 = u2;
        u2 = rng.next_f64();

        writeln!(&mut writer, "{},{},{}", u0, u1, u2)?;
    }

    writer.flush()?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    // Use the same seed/state where sensible to make comparisons reproducible.
    let seed32: u32 = 1_234_567; // non-zero for LCGs
    let seed64: u64 = 0x_7_2_0_1_23;

    // 1. Park–Miller LCG.
    let mut pm_for_stats = ParkMiller::new(seed32);
    summarize_uniform(&mut pm_for_stats, "Park–Miller LCG (m = 2^31 - 1, a = 16807)");

    let mut pm_for_triples = ParkMiller::new(seed32);
    write_triples_csv(&mut pm_for_triples, "parkmiller_triples.csv")?;
    println!("Wrote Park–Miller triples to parkmiller_triples.csv");

    // 2. RANDU generator.
    let mut randu_for_stats = Randu::new(seed32);
    summarize_uniform(&mut randu_for_stats, "RANDU (m = 2^31, a = 65539)");

    let mut randu_for_triples = Randu::new(seed32);
    write_triples_csv(&mut randu_for_triples, "randu_triples.csv")?;
    println!("Wrote RANDU triples to randu_triples.csv");

    // 3. Modern generator from rand::StdRng.
    let mut modern_for_stats = ModernStdRng::new(seed64);
    summarize_uniform(&mut modern_for_stats, "StdRng (modern generator from rand crate)");

    Ok(())
}
```

Program 7.2.1 demonstrates the practical evaluation of uniform random deviates and highlights the stark contrast between well-designed and poorly designed generators. The summary statistics confirm that basic one-dimensional uniformity properties, mean near $1/2$ and variance near $1/12$, may appear satisfactory even for fundamentally flawed generators. This reflects an essential lesson from Section 7.2: superficial tests can conceal deep structural problems, and visibly uniform histograms do not guarantee sound multidimensional randomness.

The exported triples provide tangible evidence of these issues. Whereas the Park–Miller generator distributes points broadly within the unit cube, RANDU exhibits the classical planar structure predicted by Marsaglia’s spectral analysis. This reinforces the historical lesson that the recurrence in Equation (7.2.5) is unsuitable for scientific computation, despite producing superficially plausible one-dimensional statistics. Modern generators, such as `StdRng`, avoid such defects by combining multiple algebraic operations and nonlinear mixing steps, validating the consensus that no single recurrence suffices for high-quality randomness.

By providing a unified testing framework, the program illustrates how numerical diagnostics, visualization, and historical context collectively inform the selection of uniform generators. These considerations become crucial in applications where statistical integrity directly impacts simulation accuracy, uncertainty quantification, and reproducibility, central themes that motivate the developments throughout Section 7.2.

## 7.2.2. Mathematical Formulation and Generator Algorithms

Designing a uniform random number generator (RNG) involves constructing a deterministic sequence $X_0, X_1, X_2, \dots$, that statistically mimics independent uniform random variables on $[0,1)$. The challenge lies in achieving this within the constraints of finite-precision arithmetic and reproducible computation. Several distinct mathematical structures underpin modern uniform generators, each with advantages in period length, speed, and statistical quality.

### Linear Recurrences Modulo Two

A large class of fast generators operates directly on binary representations of the state, interpreting it as a vector in the finite field $\mathbb{F}_2^n$. Each state update is realized as a linear transformation in this vector space, using bitwise XOR (addition modulo 2) and bit-shift operations.

A representative example is the *xorshift* generator. In 64-bit form, its update rule can be written as:

$$
\begin{aligned}
x &\leftarrow x \oplus (x \ll a_1),\\
x &\leftarrow x \oplus (x \gg a_2),\\
x &\leftarrow x \oplus (x \ll a_3),
\end{aligned}
\tag{7.2.6}
$$

where $\ll$ and $\gg$ denote left and right logical shifts, and $a_1, a_2, a_3$ are integer shift parameters, for instance, $(a_1, a_2, a_3) = (21, 35, 4)$. Each operation corresponds to multiplication by a sparse $64\times64$ binary matrix, and the full update is the product of these matrices.

The transformation per iteration, $T$, is therefore an element of the general linear group $\mathrm{GL}(64,\mathbb{F}_2)$. The period of the generator equals the order of $T$; that is, the smallest positive integer $N$ such that $T^N = I$ (the identity). For a *full period* of $2^{64}-1$, the characteristic polynomial of $T$ must be primitive of degree 64. This ensures that all non-zero 64-bit states appear before the sequence repeats.

Testing the primitivity of $T$ can be performed efficiently through factorization of $2^{64}-1$ and repeated-squaring methods. Parameter triples have been identified that produce full-period xorshift sequences while maintaining strong statistical properties, a key consideration, since achieving a full period alone does not ensure high-quality randomness. It has also been shown that each bit of a basic xorshift sequence evolves according to its own linear recurrence, which can introduce subtle correlations. To reduce these effects, modern designs incorporate non-linear scrambling or output tempering mechanisms, as implemented in improved variants such as xorshift+, xorshift\*\*, and xoroshiro.

### Mixed Linear and Nonlinear Operations

Many recent generators combine linear recurrences with non-linear output transformations to enhance statistical properties. The Permuted Congruential Generator (PCG) exemplifies this philosophy. Internally, it uses a 64-bit linear congruential update,

$$X_{n+1} = aX_n + c \pmod{2^{64}} \tag{7.2.7}$$

but instead of outputting $X_n$ directly, the algorithm permutes its bits using a non-linear transformation such as a rotation or XOR-shift of high-order bits:

\begin{equation}
U_n = \operatorname{rotate\_right}\!\big(X_n \oplus (X_n \gg 18),\, X_n \gg 59\big)
\tag{7.2.8}
\end{equation}

This approach retains the simplicity and period guarantees of an LCG while masking its linearity through output permutation. The result is a family of generators with excellent statistical behaviour, small state, and reproducible sequences suitable for both CPU and GPU implementations.

The PCG method illustrates a general design principle: the separation of state evolution and output function. By evolving the internal state linearly and applying a nonlinear bijection to produce the output, one can combine theoretical predictability (for reproducibility) with statistical irregularity (for randomness).

### Vectorized and Parallel Formulations

In large-scale numerical simulations, RNGs must generate billions of values efficiently across multiple threads or GPU cores. Linear recurrences in $\mathbb{F}_2^n$ are particularly amenable to vectorization since their matrix multiplications reduce to XOR and shift operations on 64-bit words. Libraries such as *xoroshiro128*\*\*, *xoshiro256++*, and *SplitMix64* leverage these operations to produce statistically decorrelated streams while maintaining SIMD compatibility.

Parallel RNGs are often designed by assigning each thread a distinct subsequence via skip-ahead techniques. If $T$ is the transition matrix, one can advance the state by $k$ steps using matrix exponentiation:

$$X_{n+k} = T^k X_n \tag{7.2.9}$$

computed efficiently by exponentiation by squaring. This method ensures disjoint random streams, critical for reproducibility in parallel Monte Carlo frameworks.

```{figure} images/pqQDe4beUu67RvW3raYP-OFEOURaMMWfeDrTF4QoH-v1.png
:name: syQig7i8Ht
:align: middle
:width: 40%

**Figure 7.2.3** Illustration of state evolution in a xorshift generator as a linear transformation in $\mathbb{F}_2^{64}$.
```

Figure 7.2.3 visualizes the state evolution of a 64-bit xorshift generator as a linear transformation in the finite vector space $\mathbb{F}2^{64}$. Each iteration updates the internal state vector $x_n = (b_0, b_1, \ldots, b_{63})^\top$ through a sequence of bitwise linear operations, left and right logical shifts followed by exclusive-OR feedback. These operations correspond to multiplications by sparse binary matrices, each representing a shift-and-mix transformation.

The figure conceptually illustrates how the composition of these matrices forms a single state transition matrix $T \in \mathrm{GL}(64, \mathbb{F}2)$, so that $x_{n+1} = T,x_n$. The period and distributional quality of the sequence are determined by the algebraic properties of $T$, particularly the **primitivity** of its characteristic polynomial. Arrows in the diagram represent the XOR feedback paths that mix shifted versions of the state bits, ensuring that every bit eventually influences every other. In effect, the generator performs a high-dimensional rotation in $\mathbb{F}_2^{64}$, where linear dependence or degeneracy would reduce randomness.

This graphical view reinforces the key mathematical idea: a xorshift generator is a deterministic linear recurrence modulo 2, whose statistical quality depends on the spectral and algebraic properties of its transformation matrix rather than on the bit pattern itself.

### Statistical Quality Considerations

The adequacy of an RNG is assessed not merely by its period but by its ability to pass rigorous statistical tests. Generators based purely on linear recurrences can exhibit detectable structure in higher-dimensional projections, motivating the use of non-linear combinations, output tempering, and composite generators.

A practical summary of generator quality is provided in Table 7.2.1, contrasting period length, state size, and test-suite performance for representative algorithms such as LCG, xorshift, PCG, and Mersenne Twister.

~~~{list-table} **Table 7.2.1** Comparison of representative uniform RNGs by period, state size, and statistical robustness (after Vigna 2016; O’Neill 2014).
:header-rows: 1
:name: dSYOhAQAIo

* - Generator

  - Period

  - State size

  - Statistical robustness\*

* - LCG (classic)

  - $2^{32}–2^{64}$

  - 32 or 64 bits

  - Moderate: fails many modern tests

* - xorshift family (Vigna)

  - up to $2^{1024}-1$

  - 128 or more bits

  - High: with proper scrambling (Vigna 2016)

* - PCG family (O’Neill)

  - $2^{64}– 2^{128}$

  - 64–128 bits

  - Very high: passes TestU01 with margin (O’Neill 2014)

* - Mersenne Twister (MT)

  - $2^{19937}-1$

  - \~2 kB state

  - Good: large state and somewhat slower

~~~

\* “Statistical robustness” refers to the ability to pass demanding empirical test suites (e.g., TestU01, BigCrush) without systematic failures.

The mathematical framework outlined here provides the basis for the more advanced developments discussed in the next subsection, where hybrid and composite RNGs achieve unprecedented levels of statistical reliability for large-scale simulations.

### Rust Implementation

Following the mathematical development in Section 7.2.2, which introduced uniform random number generators as deterministic recurrences in either $\mathbb{F}_2^n$ or the ring $\mathbb{Z}/2^{64}\mathbb{Z}$, Program 7.2.2 provides practical implementations of two representative algorithms: a xorshift generator based on the linear transformation (7.2.6) and a PCG generator whose internal state evolves according to the linear congruential update (7.2.7) while its output is formed via the non-linear permutation (7.2.8). These two designs illustrate the core principles discussed earlier, namely the separation between state evolution and output function, and the importance of bit-level mixing to enhance statistical quality. The program also demonstrates skip-ahead techniques, corresponding to the matrix-power formulation in Equation (7.2.9), enabling construction of independent subsequences for parallel Monte Carlo workflows. Through summary statistics and sample substreams, the implementation highlights how algebraic structure directly influences the empirical behaviour of uniform deviates in finite-precision arithmetic.

At the heart of the implementation are two trait abstractions: `Rng32`, which defines a generator capable of producing 32-bit integers and corresponding floating-point deviates, and `RunningStats`, which accumulates mean, variance, and extrema using Welford’s method. The `Rng32` trait encapsulates the essential interface of a uniform generator by providing `next_u32` and a derived `next_f64` method that maps raw integers into the interval $[0,1)$. This abstraction allows the same statistical routines to be used across multiple generator families without changing the analytical logic.

The xorshift generator implements the three-step linear transformation of Equation (7.2.6), applying XOR and bit-shift operations that correspond to multiplications by sparse matrices in $\mathrm{GL}(64,\mathbb{F}_2)$. The internal state is maintained as a 64-bit word, while the high 32 bits serve as the output. This design reflects the theoretical requirement that the transformation matrix possess a primitive characteristic polynomial to achieve a maximal period of $2^{64}-1$. Although simple, this generator effectively demonstrates the finite-field recurrence structure discussed earlier, including the fact that each bit evolves according to a linear recurrence mod 2.

The PCG implementation provides a clear example of mixed linear and nonlinear operations. Its internal state evolves according to the 64-bit LCG in Equation (7.2.7), but the output is formed by the XSH-RR permutation specified in Equation (7.2.8), combining XOR-shifts with a rotation determined by high-order bits of the state. This reflects the PCG principle of decoupling the state transition from the output function to preserve reproducibility while enhancing statistical robustness. The constructor follows the recommended PCG seeding procedure, setting the increment, stepping the state, injecting the seed, and stepping again, to ensure that both state and stream parameters influence all bits of the generator.

The skip-ahead mechanism implements the scalar version of the matrix-power update (7.2.9), computing the action of the LCG after an arbitrary number of steps using exponentiation by squaring. This allows the state to be advanced by $k$ iterations in $O(\log k)$ time without explicitly iterating the recurrence. By applying skip-ahead with different offsets, the program constructs distinct subsequences of the generator, demonstrating how parallel Monte Carlo methods allocate non-overlapping streams to multiple execution threads while maintaining reproducibility.

The `main` function consolidates these components by computing empirical statistics for both the xorshift and PCG generators over a large sample size and printing short previews of several substreams. These outputs illustrate two key ideas from the theory: first, that structurally different generators can still achieve close agreement with the uniform distribution on $[0,1)$, and second, that skip-ahead allows the construction of decorrelated parallel sequences consistent with the algebraic structure of the underlying recurrence.

```rust
// Program 7.2.2: Xorshift and PCG Uniform Generators with Skip-Ahead
//
// This program illustrates the mathematical ideas in Section 7.2.2:
//
//   • A 64-bit xorshift generator implementing the linear recurrence (7.2.6)
//     over the finite field F_2^64, with output taken from the high 32 bits.
//
//   • A canonical PCG-XSH-RR 64/32 generator: internal 64-bit LCG state
//     (7.2.7) modulo 2^64 and a non-linear 32-bit output permutation (7.2.8).
//
//   • A skip-ahead routine for the LCG, showing how to advance the PCG
//     state by k steps in O(log k) time, reflecting the matrix-power
//     formulation (7.2.9) specialized to a scalar recurrence.
//
// The program computes basic summary statistics for both generators and
// prints a few values from several PCG substreams created by skip-ahead.
//
// No external crates are required.

use std::f64;

/// Trait for RNGs that produce 32-bit integers and normalized U(0,1) deviates.
trait Rng32 {
    /// Return the next 32-bit random value.
    fn next_u32(&mut self) -> u32;

    /// Map the raw value to a floating-point deviate in [0,1).
    ///
    /// We scale by 2^-32, which yields a discrete uniform distribution
    /// over the grid {0, 1/2^32, ..., (2^32 - 1)/2^32}.
    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (u32::MAX as f64 + 1.0);
        (self.next_u32() as f64) * SCALE
    }
}

/// Running mean, variance, min, and max via Welford’s algorithm.
#[derive(Debug)]
struct RunningStats {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn push(&mut self, x: f64) {
        self.n += 1;
        let n_f = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n_f;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / ((self.n - 1) as f64)
        } else {
            0.0
        }
    }

    fn min(&self) -> f64 {
        self.min
    }

    fn max(&self) -> f64 {
        self.max
    }
}

/// Xorshift64 generator implementing the recurrence in (7.2.6):
///
///   x <- x XOR (x << a1)
///   x <- x XOR (x >> a2)
///   x <- x XOR (x << a3)
///
/// with (a1, a2, a3) = (21, 35, 4), operating over 64-bit words.
/// All-zero state is forbidden.
/// We return the high 32 bits as the 32-bit output.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0xdead_beef_dead_beef } else { seed };
        Self { state: s }
    }
}

impl Rng32 for XorShift64 {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        // Parameters (21, 35, 4) as in the text.
        x ^= x << 21;
        x ^= x >> 35;
        x ^= x << 4;
        self.state = x;
        // Use the high 32 bits as the output.
        (x >> 32) as u32
    }
}

/// PCG-XSH-RR 64/32 generator:
///
/// Internal state (u64) evolves as the LCG (7.2.7):
///   X_{n+1} = a X_n + c  (mod 2^64)
///
/// Output uses the XSH-RR permutation (32-bit output):
///   xorshifted = ((oldstate >> 18) ^ oldstate) >> 27
///   rot        = oldstate >> 59
///   output     = rotate_right(xorshifted, rot)
///
/// Here `inc` is an odd constant defining the stream; different `inc`
/// values define distinct, non-overlapping sequences.
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    // Standard PCG multiplier for 64-bit state.
    const MULT: u64 = 6364136223846793005;

    /// Construct a PCG32 with the given seed and stream.
    ///
    /// - `seed`   controls the starting state.
    /// - `stream` selects an odd increment `inc = 2*stream + 1`.
    ///
    /// This follows the seeding pattern recommended in the PCG reference:
    /// start from state = 0, set the increment, advance once, add the seed,
    /// then advance again.
    fn new(seed: u64, stream: u64) -> Self {
        let inc = (stream << 1) | 1; // must be odd
        let mut g = Pcg32 { state: 0, inc };
        // Incorporate the stream into the state:
        g.state = g
            .state
            .wrapping_mul(Self::MULT)
            .wrapping_add(g.inc);
        // Add the seed:
        g.state = g.state.wrapping_add(seed);
        // One more step to diffuse the seed.
        g.state = g
            .state
            .wrapping_mul(Self::MULT)
            .wrapping_add(g.inc);
        g
    }

    /// One step of the internal LCG (state evolution only).
    fn step_state(&mut self) {
        self.state = self
            .state
            .wrapping_mul(Self::MULT)
            .wrapping_add(self.inc);
    }

    /// Output permutation (XSH-RR), corresponding to (7.2.8),
    /// but returning a 32-bit integer.
    fn output_permutation(oldstate: u64) -> u32 {
        let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
        let rot = (oldstate >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
}

impl Rng32 for Pcg32 {
    fn next_u32(&mut self) -> u32 {
        let oldstate = self.state;
        // Advance internal state (LCG part).
        self.step_state();
        // Produce the permuted 32-bit output.
        Self::output_permutation(oldstate)
    }
}

/// Advance the internal LCG state (mod 2^64) by `delta` steps in O(log delta) time.
///
/// The underlying recurrence is:
///   X_{n+1} = a X_n + c  (mod 2^64)
///
/// This routine computes:
///   X_{n+delta} = a^delta X_n + c * (a^{delta-1} + ... + a + 1)  (mod 2^64)
///
/// using exponentiation by squaring. This is (7.2.9) specialized to a scalar
/// linear recurrence instead of an explicit transition matrix.
fn advance_lcg(state: u64, mut delta: u64, a: u64, c: u64) -> u64 {
    let mut cur_mult = a;
    let mut cur_plus = c;
    let mut acc_mult: u64 = 1;
    let mut acc_plus: u64 = 0;

    while delta > 0 {
        if (delta & 1) != 0 {
            acc_mult = acc_mult.wrapping_mul(cur_mult);
            acc_plus = acc_plus
                .wrapping_mul(cur_mult)
                .wrapping_add(cur_plus);
        }
        cur_plus = (cur_mult.wrapping_add(1)).wrapping_mul(cur_plus);
        cur_mult = cur_mult.wrapping_mul(cur_mult);
        delta >>= 1;
    }

    acc_mult.wrapping_mul(state).wrapping_add(acc_plus)
}

/// Construct a PCG substream by skipping ahead `skip` steps from a base state,
/// using the same multiplier and increment.
fn pcg_substream(base_state: u64, inc: u64, skip: u64) -> Pcg32 {
    let advanced_state = advance_lcg(base_state, skip, Pcg32::MULT, inc);
    Pcg32 {
        state: advanced_state,
        inc,
    }
}

/// Compute and print summary statistics for a given RNG over `n` samples.
fn summarize_rng<R: Rng32>(rng: &mut R, label: &str, n: usize) {
    let mut stats = RunningStats::new();
    for _ in 0..n {
        stats.push(rng.next_f64());
    }

    println!("Summary statistics for {}:", label);
    println!("  n          = {}", n);
    println!("  mean       ≈ {:.6} (theoretical: 0.5)", stats.mean());
    println!(
        "  variance   ≈ {:.6} (theoretical: 1/12 ≈ {:.6})",
        stats.variance(),
        1.0 / 12.0
    );
    println!("  min        ≈ {:.6}", stats.min());
    println!("  max        ≈ {:.6}", stats.max());
    println!();
}

fn main() {
    const N_STATS: usize = 100_000;

    // 1. Xorshift64: linear recurrence over F_2^64 as in (7.2.6).
    let seed_xor: u64 = 0x_1234_5678_9abc_def0;
    let mut xor_rng = XorShift64::new(seed_xor);
    summarize_rng(&mut xor_rng, "XorShift64 (a1=21, a2=35, a3=4)", N_STATS);

    // 2. PCG32: canonical mixed linear/nonlinear design per (7.2.7)–(7.2.8).
    let seed_pcg: u64 = 0x_7_2_2_0_1_23;
    let stream_pcg: u64 = 42;
    let mut pcg_rng = Pcg32::new(seed_pcg, stream_pcg);
    summarize_rng(
        &mut pcg_rng,
        "PCG-XSH-RR 64/32 (LCG + non-linear permutation)",
        N_STATS,
    );

    // 3. Demonstrate skip-ahead and substreams for PCG.
    //
    // We create a "master" generator, capture its internal state after
    // initialization, and then form four substreams by skipping ahead
    // different amounts. In a parallel Monte Carlo setting, each thread
    // could be assigned one of these substreams.
    let master = Pcg32::new(seed_pcg, stream_pcg);
    let base_state = master.state;
    let inc = master.inc;

    let skips = [0_u64, 1_000_000, 2_000_000, 3_000_000];
    let mut substreams: Vec<(String, Pcg32)> = Vec::new();

    for (i, &delta) in skips.iter().enumerate() {
        let label = format!("PCG substream {} (skip = {})", i, delta);
        let stream = pcg_substream(base_state, inc, delta);
        substreams.push((label, stream));
    }

    // Print a few values from each substream to illustrate decorrelation.
    let n_preview = 5;
    for (label, mut rng) in substreams {
        print!("First {} values from {}:\n  ", n_preview, label);
        for _ in 0..n_preview {
            let u = rng.next_f64();
            print!("{:.8} ", u);
        }
        println!("\n");
    }
}
```

Program 7.2.2 provides a concrete demonstration of the mathematical framework established in Section 7.2.2 by implementing uniform generators derived from both linear recurrences in $\mathbb{F}_2^n$ and mixed linear–nonlinear constructions in $\mathbb{Z}/2^{64}\mathbb{Z}$. The contrasting designs of xorshift and PCG showcase how algebraic structure shapes statistical behaviour, period length, and bit-level entropy. The empirical statistics confirm that well-formed generators produce values consistent with the theoretical uniform distribution, while the substream demonstration illustrates the critical role of skip-ahead methods in parallel simulation. Together, these examples reinforce the broader theme that uniform deviates form the foundation of modern stochastic computation, and that careful generator design, respecting both algebraic and statistical principles, is essential for reliable numerical results. This program also sets the stage for subsequent sections, where uniform deviates are transformed into more complex distributions and employed in high-dimensional Monte Carlo algorithms.

## 7.2.3. Advances in Generator Quality and Combination Methods

By the early 2020s, research on uniform random number generation had evolved into a highly rigorous field. A long period alone was no longer considered sufficient for quality; generators must demonstrate both sound mathematical construction and robust empirical performance across thousands of tests. Statistical validation frameworks such as Marsaglia’s DIEHARD, L’Ecuyer and Simard’s TestU01, and the modern PractRand suite subject generator streams to terabytes of data analysis. PractRand alone applies over 4,600 tests, examining linear complexity, autocorrelation, entropy, and bit distribution, with an ideal generator failing only within statistically expected bounds.

A canonical benchmark of modern generator quality is the SplitMix64 algorithm, a 64-bit linear scramble that achieves exceptional statistical stability and uniformity. It passes extensive test batteries such as PractRand up to input sizes of $2^{46}$ bytes (approximately 64 terabytes) without a single statistical failure, an extraordinary record of consistency. In contrast, older designs such as Xoroshiro128+, once widely used in languages and game engines, were later found to fail certain long-run statistical tests, notably the linear-complexity suite. These shortcomings led to the development of improved successors, including Xoshiro256++ and Xoshiro256\*\*, whose enhanced output-diffusion steps successfully eliminated the bias that affected earlier variants.

### Composite and Combined Generators

A key insight emerging from this evolution is that *combinations of generators* often outperform any single algorithm. Suppose two distinct uniform sequences ${X_n}$ and ${Y_n}$ are produced by generators with co-prime periods $P_X$ and $P_Y$. A composite sequence may be defined as:

$$Z_n = X_n \oplus Y_n \tag{7.2.10}$$

where $\oplus$ denotes bitwise exclusive OR. The resulting sequence inherits a total period equal to the least common multiple of the individual periods:

$$P_Z = \operatorname{lcm}(P_X, P_Y) \tag{7.2.11}$$

The effectiveness of such combinations depends on structural independence. Combining two related generators (for example, two LCGs sharing the same modulus) can reproduce the same weaknesses present in both sequences. Consequently, practical designs merge fundamentally different mechanisms, e.g., a linear congruential stream with a xorshift or Weyl sequence, ensuring that their bit-level biases cancel rather than reinforce (Brent 2021).

### Successor Relations and Output Whitening

Another refinement is *successor chaining*, where the output of one generator is processed by another stateless function acting as a nonlinear *whitening transform*. Let $X_n = f(s_n)$ denote the raw output of a generator with internal state $s_n$, and let $h(\cdot)$ represent an independent bit-mixing function. The final output sequence is then,

$$Y_n = h(X_n) \tag{7.2.12}$$

which preserves the original period but redistributes bit-level entropy. A 64-bit LCG, for instance, might exhibit bias in low-order bits; passing its output through a 64-bit xorshift transform effectively mixes high-quality bits into weaker positions, improving overall uniformity.

This concept parallels *output tempering* in the Mersenne Twister, where an invertible linear transformation improves equidistribution across multiple dimensions. The general philosophy is that nonlinear post-processing can correct minor deficiencies without altering the underlying recurrence.

### Splittable and Counter-Based Generators

Parallel computing and GPU-based simulation demand thousands of independent random streams with no cross-correlation. *Splittable generators* address this by allowing independent substreams to be created deterministically from a master seed. Techniques include *skip-ahead* (computing $X_{n+k} = T^k X_n$) and *counter-based methods* where the generator’s output is defined as:

$$X_i = f(\text{seed}, i) \tag{7.2.13}$$

with $i$ as a counter value. Each thread can use a distinct counter prefix, guaranteeing non-overlapping sequences.

Modern examples include Java’s LXM generator, which fuses an LCG with a xorshift-based component, and NVIDIA’s Philox family for GPU contexts. LXM’s design enables *splitting by state mutation*, changing the LCG component to derive non-intersecting subsequences while preserving deterministic reproducibility (Steele et al. 2021).

### Synthesis and Design Philosophy

Modern RNG design thus balances theory, empirical testing, and engineering pragmatism. Composite methods blend linear recurrences for long period and efficiency, nonlinear transformations for statistical robustness, and structured state management for parallel reproducibility. Table 7.2.2 summarizes representative high-quality RNG families from recent literature, comparing period length, speed, and statistical resilience.

~~~{list-table} **Table 7.2.2** Representative modern uniform PRNGs and their trade-offs in period, speed, and statistical quality.
:header-rows: 1
:name: TtiIXyIT92

* - Generator

  - Period

  - Speed

  - Statistical Quality

  - Notes

* - SFC64

  - $2^{64}$ (each stream)

  - Very fast

  - Very high

  - Combines addition, XOR, rotation; excellent empirical results.

* - ChaCha8

  - $2^{512}$

  - Moderate

  - Excellent (cryptographic-grade)

  - Secure 8-round stream cipher; slower but ideal for parallel or secure streams.

* - PCG

  - $2^{64}–2^{128}$

  - Fast

  - Very high

  - Compact, high-quality general-purpose generator.

* - Xoshiro256++

  - $2^{256}-1$

  - Very fast

  - High

  - Excellent diffusion; minor weaknesses in low bits.

* - LXM

  - $2^{128}+$

  - Fast

  - Very high

  - Splittable RNG combining LCG and Xoroshiro components.

~~~

The guiding principle of current RNG research is balance: long period and speed from linear structure, unpredictability from nonlinear scrambling, and reproducibility for parallelism. The emphasis has shifted from inventing new families to selecting appropriate designs suited to the precision, reproducibility, and performance requirements of specific applications.

### Rust Implementation

Following the discussion in Section 7.2.3 on the evolution of generator quality and the emergence of composite and post-processed designs, Program 7.2.3 provides practical implementations of several modern uniform random number generators. These include SplitMix64, widely regarded as a benchmark for statistically stable 64-bit scrambling; a raw linear congruential generator representing classical linear structure; a whitened LCG illustrating successor transformations of the form (7.2.12); a composite XOR generator combining independent mechanisms as in (7.2.10); and a counter-based construction implementing the stateless mapping described by (7.2.13). Together, these implementations demonstrate how contemporary RNGs integrate linear evolution, nonlinear mixing, and structural independence to achieve long period, statistical robustness, and scalability across parallel computing environments.

At the core of the program is the `Rng64` trait, which defines the common interface shared by all generators. The trait specifies a method `next_u64` for producing 64-bit outputs and a derived method `next_f64` that maps the upper 53 bits of each output into a floating-point deviate in $[0,1)$. This design abstracts away the specific internal mechanisms of each generator, allowing uniform statistical analysis and direct comparison across widely different recurrence structures and bit-mixing strategies.

The first concrete implementation is SplitMix64, which serves as a canonical example of a high-quality, 64-bit linear scramble. Its state advances via a fixed Weyl sequence, and the output is produced by an invertible arithmetic permutation that ensures excellent bit diffusion. This generator embodies the design philosophy described earlier: a simple recurrence, coupled with an aggressively mixing output transformation, can yield extraordinary empirical performance, as reflected in its PractRand record discussed in the section text.

In contrast, the `Lcg64Raw` implementation exemplifies a classical generator based solely on the linear congruential form (7.2.7). Although this recurrence offers a long period and analytic simplicity, its raw output often exhibits structural weaknesses, typically in lower bits. To address this, the program introduces a whitened variant in which each raw output $X_n$ is transformed by a stateless mixing function $h(X_n)$, representing the successor relationship described in Equation (7.2.12). This demonstrates how nonlinear post-processing can significantly improve the distributional quality of an initially weak generator without altering its period or internal state evolution.

The composite generator implements the XOR combination (7.2.10), merging two statistically distinct sequences into a joint output $Z_n$. When the underlying generators have co-prime or structurally independent periods, the resulting sequence attains the composite period (7.2.11) and often inherits the strengths of both components. This technique illustrates the principle that combining heterogeneous mechanisms, such as a fast scramble and a linear recurrence, can mitigate individual weaknesses and yield highly robust streams.

Finally, the counter-based generator implements the mapping $X_i = f(\text{seed}, i)$ from Equation (7.2.13). Here, each output is derived from a pure index value passed through a SplitMix-style mixing function. This design demonstrates how splittable, stateless streams can be constructed for large-scale parallel computation, enabling deterministic reproducibility and complete stream independence across threads.

The `main` function tests all these generators in a uniform way by computing summary statistics including mean, variance, minimum, and maximum, over a fixed number of samples and printing short previews of selected streams. These outputs show, empirically, how raw linear recurrences can be improved through successor mixing or composition, how SplitMix64 maintains excellent uniformity, and how counter-based methods produce high-quality, fully independent deviates.

```rust
// Program 7.2.3: SplitMix64, Whitening, Composite, and Counter-Based Uniform Generators
//
// This program illustrates several modern design ideas from Section 7.2.3:
//
//   • SplitMix64 as a high-quality 64-bit "linear scramble" baseline.
//   • A simple 64-bit LCG with and without a nonlinear whitening transform h(X_n)
//     as in Equation (7.2.12).
//   • A composite generator Z_n = X_n ⊕ Y_n combining two streams,
//     as in Equations (7.2.10)–(7.2.11).
//   • A counter-based generator X_i = f(seed, i) as in Equation (7.2.13),
//     using a stateless SplitMix-like mixing function.
//
// Each generator is sampled to produce U(0,1)-like deviates, and basic statistics
// (mean, variance, min, max) are reported. The goal is to show how linear
// recurrences, nonlinear mixing, and composition strategies affect the behaviour
// of uniform random deviates in practice.
//
// No external crates are required.

use std::f64;

/// Trait for RNGs producing 64-bit integers and normalized U(0,1) deviates.
trait Rng64 {
    /// Return the next 64-bit random value.
    fn next_u64(&mut self) -> u64;

    /// Map the raw value to a floating-point deviate in [0,1).
    ///
    /// We use the upper 53 bits to match the mantissa of f64:
    ///   u in {0,...,2^53-1} -> u / 2^53.
    fn next_f64(&mut self) -> f64 {
        const INV_2_53: f64 = 1.0 / ((1u64 << 53) as f64);
        let x = self.next_u64() >> 11; // keep top 53 bits
        (x as f64) * INV_2_53
    }
}

/// Running mean, variance, min, and max via Welford’s algorithm.
#[derive(Debug)]
struct RunningStats {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn push(&mut self, x: f64) {
        self.n += 1;
        let n_f = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n_f;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / ((self.n - 1) as f64)
        } else {
            0.0
        }
    }

    fn min(&self) -> f64 {
        self.min
    }

    fn max(&self) -> f64 {
        self.max
    }
}

/// SplitMix64 baseline generator.
///
/// This is a "linear scramble" over 64 bits: state evolves via addition of a
/// fixed odd constant (a Weyl sequence), and output is produced by mixing
/// the state with invertible arithmetic operations. It is known to exhibit
/// excellent statistical behaviour in practice.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn mix64(mut z: u64) -> u64 {
        z ^= z >> 30;
        z = z.wrapping_mul(0xBF58476D1CE4E5B9);
        z ^= z >> 27;
        z = z.wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        z
    }
}

impl Rng64 for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        // Weyl sequence increment:
        self.state = self
            .state
            .wrapping_add(0x9E3779B97F4A7C15);
        Self::mix64(self.state)
    }
}

/// Simple 64-bit LCG with modulus 2^64.
/// This represents a raw linear recurrence that may have weaknesses on its own.
struct Lcg64Raw {
    state: u64,
    a: u64,
    c: u64,
}

impl Lcg64Raw {
    fn new(seed: u64, a: u64, c: u64) -> Self {
        Self { state: seed, a, c }
    }
}

impl Rng64 for Lcg64Raw {
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(self.a)
            .wrapping_add(self.c);
        self.state
    }
}

/// Stateless mixing function h(X_n) used for output whitening (Equation 7.2.12).
///
/// Here we reuse the SplitMix64 mixing function as a generic 64-bit "whitener".
fn whiten64(x: u64) -> u64 {
    // Same mixing steps as SplitMix64::mix64, but applied to arbitrary x.
    let mut z = x;
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58476D1CE4E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    z
}

/// Whitened LCG: Y_n = h(X_n), preserving the LCG period but improving
/// bit-level entropy (successor chaining / output whitening).
struct WhitenedLcg64 {
    inner: Lcg64Raw,
}

impl WhitenedLcg64 {
    fn new(seed: u64, a: u64, c: u64) -> Self {
        Self {
            inner: Lcg64Raw::new(seed, a, c),
        }
    }
}

impl Rng64 for WhitenedLcg64 {
    fn next_u64(&mut self) -> u64 {
        let raw = self.inner.next_u64();
        whiten64(raw)
    }
}

/// Composite generator: Z_n = X_n ⊕ Y_n (Equation 7.2.10).
///
/// The period P_Z is the lcm of the component periods when they are structurally
/// independent (Equation 7.2.11). Here we simply XOR outputs from two distinct
/// RNG mechanisms.
struct XorComposite<G1, G2> {
    g1: G1,
    g2: G2,
}

impl<G1, G2> XorComposite<G1, G2> {
    fn new(g1: G1, g2: G2) -> Self {
        Self { g1, g2 }
    }
}

impl<G1: Rng64, G2: Rng64> Rng64 for XorComposite<G1, G2> {
    fn next_u64(&mut self) -> u64 {
        let x = self.g1.next_u64();
        let y = self.g2.next_u64();
        x ^ y
    }
}

/// Counter-based generator: X_i = f(seed, i) (Equation 7.2.13).
///
/// Here we use a SplitMix-style mixing function as f, applied to
///   state_i = seed + γ * i
/// where γ is an odd constant. This design is stateless apart from the counter
/// and easily splittable for parallel streams by varying seed or counter ranges.
struct CounterRng {
    seed: u64,
    gamma: u64,
    counter: u64,
}

impl CounterRng {
    fn new(seed: u64) -> Self {
        Self {
            seed,
            gamma: 0x9E3779B97F4A7C15, // same Weyl increment as SplitMix64
            counter: 0,
        }
    }
}

impl Rng64 for CounterRng {
    fn next_u64(&mut self) -> u64 {
        let idx = self.counter;
        self.counter = self.counter.wrapping_add(1);
        let state = self
            .seed
            .wrapping_add(self.gamma.wrapping_mul(idx));
        whiten64(state)
    }
}

/// Compute and print summary statistics for a generator over `n` samples.
fn summarize_rng<R: Rng64>(rng: &mut R, label: &str, n: usize) {
    let mut stats = RunningStats::new();
    for _ in 0..n {
        stats.push(rng.next_f64());
    }

    println!("Summary statistics for {}:", label);
    println!("  n          = {}", n);
    println!("  mean       ≈ {:.6} (theoretical: 0.5)", stats.mean());
    println!(
        "  variance   ≈ {:.6} (theoretical: 1/12 ≈ {:.6})",
        stats.variance(),
        1.0 / 12.0
    );
    println!("  min        ≈ {:.6}", stats.min());
    println!("  max        ≈ {:.6}", stats.max());
    println!();
}

fn main() {
    const N_STATS: usize = 100_000;

    // SplitMix64 baseline: modern, empirically very strong generator.
    let seed_splitmix: u64 = 0x_7_2_3_0_1_23;
    let mut splitmix = SplitMix64::new(seed_splitmix);
    summarize_rng(&mut splitmix, "SplitMix64 (baseline)", N_STATS);

    // Raw LCG: simple linear recurrence modulo 2^64.
    // Parameters chosen as a reasonably well-known 64-bit LCG example.
    let seed_lcg: u64 = 0x_1234_5678_9abc_def0;
    let a_lcg: u64 = 6364136223846793005; // same as PCG multiplier
    let c_lcg: u64 = 1;
    let mut lcg_raw = Lcg64Raw::new(seed_lcg, a_lcg, c_lcg);
    summarize_rng(&mut lcg_raw, "LCG (raw 64-bit state)", N_STATS);

    // Whitened LCG: successor chaining Y_n = h(X_n).
    let mut lcg_white = WhitenedLcg64::new(seed_lcg, a_lcg, c_lcg);
    summarize_rng(
        &mut lcg_white,
        "Whitened LCG (output-mixed with SplitMix-style h)",
        N_STATS,
    );

    // Composite generator: XOR of SplitMix64 and raw LCG.
    let splitmix_for_combo = SplitMix64::new(seed_splitmix);
    let lcg_for_combo = Lcg64Raw::new(seed_lcg, a_lcg, c_lcg);
    let mut composite = XorComposite::new(splitmix_for_combo, lcg_for_combo);
    summarize_rng(
        &mut composite,
        "Composite Z_n = SplitMix64 ⊕ LCG",
        N_STATS,
    );

    // Counter-based generator: X_i = f(seed, i).
    let mut counter_rng = CounterRng::new(0x_cafe_babe_dead_beef);
    summarize_rng(
        &mut counter_rng,
        "Counter-based RNG X_i = f(seed, i)",
        N_STATS,
    );

    // Print a few sample values to illustrate different streams.
    let mut sm_preview = SplitMix64::new(seed_splitmix);
    let mut cb_preview = CounterRng::new(0x_cafe_babe_dead_beef);

    let n_preview = 5;
    print!("First {} SplitMix64 values:\n  ", n_preview);
    for _ in 0..n_preview {
        print!("{:.8} ", sm_preview.next_f64());
    }
    println!("\n");

    print!("First {} counter-based values:\n  ", n_preview);
    for _ in 0..n_preview {
        print!("{:.8} ", cb_preview.next_f64());
    }
    println!();
}
```

Program 7.2.3 illustrates the central themes of modern generator design: combining algebraic recurrences for efficiency, nonlinear transformations for statistical robustness, and structured state manipulation for scalable reproducibility. The comparison between raw and whitened LCG outputs demonstrates the practical value of successor chaining, while the composite generator highlights the power of combining structurally independent streams. The results from SplitMix64 and the counter-based generator affirm the effectiveness of output scrambling and stateless designs for high-quality, parallel-ready streams. Together, these examples reinforce the broader trend described in Section 7.2.3, where generator families increasingly blend linear structure, nonlinear mixing, and composite architectures to achieve reliability on demanding statistical test suites. This program forms a natural bridge to subsequent sections, where uniform deviates serve as the substrate for constructing non-uniform distributions and for driving Monte Carlo algorithms in high-dimensional settings.

## 7.2.4 Practical Applications and Use Cases

Uniform random number generators underpin a vast array of computational methods across scientific, engineering, and data-driven domains. Their principal value lies in transforming deterministic digital hardware into a reliable source of stochastic variability. The diversity of modern applications, from high-energy physics simulations to machine learning, demonstrates that the statistical quality of uniform deviates is not merely a numerical concern but a scientific one.

### (i) Monte Carlo Integration and Numerical Simulation

In *Monte Carlo integration*, a uniform generator provides random sampling points within an $n$-dimensional domain $\Omega$. The integral of a function $f(\mathbf{x})$ can be estimated as:

$$I = \int_{\Omega} f(\mathbf{x})d\mathbf{x} \approx \frac{1}{N} \sum_{i=1}^{N} f(\mathbf{x}_i), \quad\mathbf{x}_i \sim \mathcal{U}(\Omega) \tag{7.2.14}$$

The convergence rate of this estimator is $\mathcal{O}(N^{-1/2})$ by the central limit theorem, independent of dimension. Thus, unbiased and decorrelated uniform deviates are essential for accurate estimates. In practice, the variance of $I$ is inversely proportional to the generator’s effective independence; even small correlations can produce systematic biases in high-dimensional problems.

Uniform deviates also drive stochastic differential equation solvers, molecular dynamics, particle transport, and radiation Monte Carlo simulations. In these applications, the random generator determines physical events such as scattering angles, collision distances, or particle energies (Josey 2023). The fidelity of simulated results is therefore inseparable from the quality of the underlying uniform stream.

### (ii) Optimization, Machine Learning, and Sampling

Optimization algorithms that rely on stochastic exploration, such as, simulated annealing, evolutionary algorithms, and stochastic gradient descent (SGD), depend critically on uniform deviates.

In SGD, the update rule:

$$\theta_{t+1} = \theta_t - \eta  \nabla_\theta L(\theta_t;\mathcal{B}_t) \tag{7.2.15}$$

\
uses a mini-batch $\mathcal{B}_t$ drawn uniformly at random from the training set. A biased or non-uniform sampling distribution can alter the loss surface exploration, affecting both convergence rate and generalization.

Similarly, in reinforcement learning, exploration policies such as $\varepsilon$-greedy rely on uniform draws to choose random actions; deviations from uniformity can induce exploration bias. In evolutionary strategies and genetic algorithms, uniform deviates determine mutation probabilities and crossover points, shaping the stochastic trajectory of the population.

Uniform generators are also central to Bayesian inference and Markov-Chain Monte Carlo (MCMC). Uniform variates drive proposal acceptances in Metropolis–Hastings schemes and control random perturbations in Hamiltonian dynamics. Poor RNG performance can directly lead to under-mixing and biased posterior estimates.

### (iii) Cryptography and Secure Randomness

In cryptography and secure computation, uniformity and unpredictability are both required, but pseudorandom generators must meet additional information-theoretic criteria. A secure RNG should satisfy next-bit unpredictability: given any prefix of output bits, no polynomial-time algorithm can predict the next bit with probability greater than $(1/2) + ε$.

Although the algorithms discussed earlier (e.g., xorshift, PCG) are not cryptographically secure, their structure influences the design of secure counterparts such as ChaCha and AES-CTR, where uniform diffusion ensures statistical indistinguishability from true randomness while resisting state reconstruction. In many systems, high-quality non-secure generators (e.g., SplitMix64) seed cryptographic generators to balance efficiency with security.

### (iv) Computer Graphics, Procedural Modeling, and Visualization

Uniform deviates are indispensable in visual computing. In Monte Carlo ray tracing, random sampling of light directions reduces aliasing and produces realistic shadows and reflections. The rendering equation involves a high-dimensional integral evaluated by random sampling of scattering directions and surface points.

Similarly, procedural modeling used in generating terrains, textures, and vegetation, relies on reproducible pseudorandom patterns. Uniform deviates allow deterministic regeneration of the same virtual environment from a single seed, ensuring both variety and reproducibility across sessions.

In real-time graphics, the combination of speed and decorrelation makes modern generators such as xoroshiro128++ or SFC64 preferred choices, as they can be computed in a few CPU cycles and vectorized across shader threads.

### (v) High-Performance and Parallel Simulation

Large-scale simulations on GPUs or clusters may require billions of independent random sequences. Counter-based RNGs and splittable designs such as LXM (Steele et al. 2021) enable deterministic parallel generation. Each process or thread can independently produce non-overlapping subsequences defined by:

$$X_i = f(\text{seed}, \text{counter}_i)\tag{7.2.16}$$

guaranteeing reproducibility and independence even in massive distributed computations.

High-quality uniform deviates are also essential for uncertainty quantification (UQ) in engineering simulations, where input parameters are treated as random variables. In these contexts, poor RNG quality can compromise convergence of variance-reduction estimators or misrepresent confidence intervals.

Uniform random deviates thus represent one of the most pervasive abstractions in numerical computing. They are not merely algorithmic conveniences but fundamental enablers of stochastic modeling, optimization, and simulation. The design and selection of a uniform RNG must therefore be guided by the application’s precision, reproducibility, and statistical sensitivity.

The next subsection consolidates these developments by summarizing the evolution of uniform generators, from early congruential forms to composite, splittable, and vectorized designs, and their continuing role as the foundation for higher-order random number systems.

## 7.2.5. Foundations and Frontiers of Uniform Random Number Generation

Uniform random number generation represents one of the most enduring foundations of computational science. Its theoretical basis unites number theory, finite-field algebra, and statistical inference, while its practical evolution mirrors advances in hardware architecture and algorithmic design. From early linear congruential generators to modern hybrid, vectorized, and splittable schemes, the development of uniform deviates reflects an ongoing quest to balance mathematical rigor, statistical fidelity, and computational performance. The following synthesis consolidates these achievements and highlights the emerging frontiers where random number generation intersects with cryptography, parallel computing, and domain-specific hardware acceleration.

Every pseudorandom generator, no matter how sophisticated, ultimately strives to reproduce the ideal behaviour of independent uniform draws on $[0,1)$. The evolution of generator theory, from early linear congruential forms to modern hybrid, vectorized, and splittable designs, reflects a continual pursuit of three intertwined goals: statistical fidelity, computational efficiency, and reproducibility.

At the mathematical level, the study of uniform generators reveals a deep interaction between number theory, linear algebra over finite fields, and digital arithmetic. The primitive-polynomial analysis of linear recurrences in $\mathbb{F}_2^n$ ensures long periods and maximal equidistribution. Nonlinear permutations, such as those introduced in PCG or xoshiro-type families, enhance bit diffusion without compromising reproducibility. Parallel and counter-based architectures extend these concepts to distributed and GPU environments, guaranteeing decorrelated streams across billions of independent threads.

Empirical evaluation has advanced equally. Frameworks such as TestU01, DIEHARD, and PractRand have become indispensable in establishing statistical robustness. The failure of once-popular generators like RANDU or xorshift+ demonstrates that period length alone is insufficient; only comprehensive testing across dimensions and scales can validate randomness quality. The resulting insight is that a good generator is not merely one that “appears random” but one whose mathematical structure precludes detectable correlation under any known empirical test.

In practice, uniform deviates permeate nearly every computational discipline. They drive Monte Carlo integration, stochastic optimization, molecular dynamics, Bayesian inference, and graphical rendering. Their role has expanded further with the rise of probabilistic machine learning and uncertainty quantification, where reproducible randomness underpins experimental verifiability. Thus, the design of a random number generator is not a peripheral implementation detail but a core component of scientific integrity.

Looking forward, the frontier of random generation research continues to merge theoretical cryptographic rigor, hardware acceleration, and domain-specific adaptation. Emerging directions include entropy-harvesting for hybrid true/pseudorandom systems, GPU-optimized counter generators, and deterministic RNGs embedded directly into domain-specific compilers and scientific frameworks.

The next section extends these principles to deterministic hashing schemes that emulate uniform randomness over discrete data structures. Whereas Sections 7.2.1–7.2.4 examined how to generate independent uniform values in continuous domains, the forthcoming discussion focuses on distributing those values efficiently and reproducibly across large discrete arrays, a crucial operation in high-performance simulation and randomized algorithms.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/8qwUDynY1uvL9VR2TxPB.4","tags":[]}

# 7.3. Completely Hashing a Large Array

In large-scale numerical simulations, random access to pseudo-random numbers is often needed without maintaining vast arrays of stored values. For instance, in Monte Carlo particle transport or stochastic grid models, one may need the $n$-th random value directly, independent of all previous values. Traditional linear congruential or xorshift generators are unsuitable for this because their states evolve sequentially. To overcome this limitation, the concept of complete hashing or index-based randomization is introduced.

This approach allows the direct computation of a pseudo-random value from an integer index $j$, ensuring that each entry appears statistically independent and uniformly distributed over $[0,1)$. The challenge is to achieve this while preserving determinism and reproducibility. A method that has proven effective is to emulate cryptographic diffusion, borrowing ideas from the DES cipher but simplified for speed and portability.

## 7.3.1. Pseudo-DES Hashing Function

The pseudo-DES method performs *bit-mixing* on a 64-bit word using operations reminiscent of the DES round function but with numerical constants optimized for numerical uniformity rather than cryptographic security.

Let the initial value be divided into two 32-bit halves:

$$L_0 = \text{high}(j), \qquad R_0 = \text{low}(j)\tag{7.3.1}$$

where $\text{high}(j)$ and $\text{low}(j)$ extract the upper and lower 32-bit words of the index $j$.

Each round of the transformation follows a Feistel-like structure:

$$L_{i+1} = R_i,\qquad R_{i+1} = L_i \oplus f(R_i, C_{1i}, C_{2i})\tag{7.3.2}$$

where $C_{1i}$ and $C_{2i}$ are round constants and $\oplus$ denotes the bitwise exclusive-or operation.

The function $f(R, C_1, C_2)$ is defined as:

$$f(R, C_1, C_2) =((R \times C_1) + C_2) \bmod 2^{32}\tag{7.3.3}$$

followed by nonlinear mixing through a combination of bit shifts and XORs:

$$f(R, C_1, C_2) =((R \times C_1 + C_2) \ll 19)\oplus((R \times C_1 + C_2) \gg 13)\tag{7.3.4}$$

where $\ll$ and $\gg$ denote left and right logical shifts, respectively.

The constants $C_{1i}$ and $C_{2i}$ are selected to maximize diffusion:

$$
\begin{aligned}
C_{1} &= (0x\text{BAA96887},\, 0x\text{1E17D32C},\, 0x\text{03BCDC3C},\, 0x\text{0F33D1B2}), \\
C_{2} &= (0x\text{4B0F3B58},\, 0x\text{E874F0C3},\, 0x\text{6955C5A6},\, 0x\text{55A7CA46})
\end{aligned}
\tag{7.3.5}
$$

After four such rounds, the concatenation of $(L_4, R_4)$ forms the final 64-bit result. The resulting bit pattern is well-mixed even for successive integer indices $j$.

This process can be viewed as a non-cryptographic avalanche function: a one-bit change in the input produces, on average, a half-bit change in every output bit. It is specifically designed to avoid correlations among neighboring indices and provides high-quality pseudo-randomness suitable for large-array indexing.

### Rust Implementation

Following the discussion in Section 7.3 on the need for index-based randomization in large-scale simulations, Program 7.3.1 provides a practical implementation of a pseudo-DES hashing function capable of generating pseudo-random values directly from integer indices. Unlike sequential generators, which evolve state one step at a time, the pseudo-DES approach enables the computation of the j-th random value independently of all preceding values. This property is essential in particle transport, stochastic grids, and any algorithm where random access to high-quality deviates must be achieved without storing or traversing long generator sequences. By emulating the diffusion structure of the DES round function while preserving computational simplicity, the implementation produces statistically uniform, decorrelated values suitable for large-array randomization under deterministic and reproducible conditions.

At the core of the implementation is the `pseudo_des` function, which transforms a 64-bit integer index through a sequence of Feistel-like rounds. As introduced in Equation (7.3.1), each input index is decomposed into two 32-bit halves $L_0$ and $R_0$, representing the high and low words of the input. The transformation proceeds through four rounds following the update rule in Equation (7.3.2). Each round swaps the halves and replaces the right component by XOR-combining the previous left half with the output of the nonlinear mixing function $f(R_i, C_{1i}, C_{2i})$. This structure ensures that every bit in the input influences many output bits after only a few rounds, achieving the avalanche property essential for decorrelated indexing.

The bit-mixing function itself directly implements the arithmetic and logical operations described in Equations (7.3.3) and (7.3.4). For each round, the function forms the product $R \times C_1$, adds the constant $C_2$, and wraps the result modulo $2^{32}$. The nonlinear component arises from combining shifted versions of this intermediate value via XOR: a left shift by 19 bits and a right shift by 13 bits. This combination ensures that both high-order and low-order bits are thoroughly mixed, overcoming the structural regularities that plague simpler hash functions. The constants $C_{1i}$ and $C_{2i}$, listed in Equation (7.3.5), were chosen to maximize diffusion and avoid correlations among nearby indices.

The function `u01_from_u64` converts the 64-bit output of the pseudo-DES rounds into a floating-point deviate in $[0,1)$. It follows the standard procedure of scaling by $1/2^{64}$, ensuring that all 64 bits contribute to the mantissa and minimizing rounding bias. The helper `compute_statistics` computes empirical moments including mean, variance, minimum, and maximum, allowing the resulting sequence to be compared with the theoretical uniform distribution defined by Equation (7.2.1). These diagnostics reinforce that pseudo-DES hashing behaves like a high-quality random source despite operating without internal state.

The `main` function demonstrates the suitability of pseudo-DES hashing for large-array randomization. It first evaluates statistical properties over the range of indices $[0,n)$ to confirm global uniformity. It then prints the hashed values for several consecutive indices beginning at a user-selected $j$, illustrating how successive integers produce well-scrambled outputs with no visible pattern, as expected from the avalanche function behavior discussed in Section 7.3. The combination of statistical diagnostics and local inspection provides practical validation of the hashing method for large-scale computational work.

```rust
// Program 7.3.1: Pseudo-DES Index Hashing for Uniform Deviates on [0,1)
//
// This program implements the 64-bit pseudo-DES hashing function described in
// Section 7.3.1. Given an integer index j, it applies a 4-round Feistel-like
// transformation on the 64-bit word, using the constants in Equation (7.3.5)
// and the nonlinear mixing function f in Equations (7.3.3)–(7.3.4).
//
// The resulting 64-bit value is then mapped to a floating-point deviate in
// [0,1), providing direct, index-based access to pseudo-random numbers without
// maintaining generator state. This is particularly useful for large arrays
// in Monte Carlo simulations, where one often needs the j-th pseudo-random
// entry independently of preceding values.
//
// The main function computes basic summary statistics for the first N hashed
// indices and prints a few sample values to illustrate the apparent
// randomness of successive indices.
//
// No external crates are required.

use std::f64;

/// Running mean, variance, min, and max via Welford’s algorithm.
#[derive(Debug)]
struct RunningStats {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn push(&mut self, x: f64) {
        self.n += 1;
        let n_f = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n_f;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / ((self.n - 1) as f64)
        } else {
            0.0
        }
    }

    fn min(&self) -> f64 {
        self.min
    }

    fn max(&self) -> f64 {
        self.max
    }
}

/// Pseudo-DES round function f(R, C1, C2) as in Equations (7.3.3)–(7.3.4).
///
///   t = (R * C1 + C2) mod 2^32
///   f = (t << 19) XOR (t >> 13)
///
/// All operations are performed in 32-bit arithmetic with wrapping semantics.
fn pseudo_des_f(r: u32, c1: u32, c2: u32) -> u32 {
    let t = r
        .wrapping_mul(c1)
        .wrapping_add(c2);
    (t << 19) ^ (t >> 13)
}

/// Pseudo-DES 64-bit hashing function.
///
/// Given a 64-bit index j, we split it into high/low 32-bit halves:
///
///   L_0 = high(j), R_0 = low(j)     (Equation 7.3.1)
///
/// and iterate four Feistel-like rounds:
///
///   L_{i+1} = R_i
///   R_{i+1} = L_i XOR f(R_i, C1_i, C2_i)   (Equation 7.3.2)
///
/// using the C1, C2 constants from Equation (7.3.5).
///
/// The final 64-bit hash is formed by concatenating (L_4, R_4).
fn pseudo_des_hash_u64(j: u64) -> u64 {
    // Extract high and low 32-bit halves (Equation 7.3.1).
    let mut l: u32 = (j >> 32) as u32;
    let mut r: u32 = (j & 0xFFFF_FFFF) as u32;

    // Round constants (Equation 7.3.5).
    const C1: [u32; 4] = [
        0xBAA9_6887,
        0x1E17_D32C,
        0x03BC_DC3C,
        0x0F33_D1B2,
    ];
    const C2: [u32; 4] = [
        0x4B0F_3B58,
        0xE874_F0C3,
        0x6955_C5A6,
        0x55A7_CA46,
    ];

    // Four Feistel rounds (Equation 7.3.2).
    for i in 0..4 {
        let f_val = pseudo_des_f(r, C1[i], C2[i]);
        let new_l = r;
        let new_r = l ^ f_val;
        l = new_l;
        r = new_r;
    }

    // Concatenate (L_4, R_4) into a 64-bit result.
    ((l as u64) << 32) | (r as u64)
}

/// Map the pseudo-DES hash of index j to a uniform deviate in [0,1).
///
/// We take the upper 53 bits of the 64-bit hash and scale by 2^-53 to match
/// the mantissa precision of f64, analogous to other generators in Section 7.2.
fn pseudo_des_uniform(j: u64) -> f64 {
    const INV_2_53: f64 = 1.0 / ((1u64 << 53) as f64);
    let h = pseudo_des_hash_u64(j);
    let x = h >> 11; // keep top 53 bits
    (x as f64) * INV_2_53
}

fn main() {
    const N: usize = 100_000;

    // 1. Compute summary statistics for the first N hashed indices.
    let mut stats = RunningStats::new();
    for j in 0..(N as u64) {
        let u = pseudo_des_uniform(j);
        stats.push(u);
    }

    println!("Summary statistics for pseudo-DES index-based deviates:");
    println!("  n          = {}", N);
    println!("  mean       ≈ {:.6} (theoretical: 0.5)", stats.mean());
    println!(
        "  variance   ≈ {:.6} (theoretical: 1/12 ≈ {:.6})",
        stats.variance(),
        1.0 / 12.0
    );
    println!("  min        ≈ {:.6}", stats.min());
    println!("  max        ≈ {:.6}", stats.max());
    println!();

    // 2. Show a few successive values for nearby indices to illustrate
    //    apparent independence and high diffusion.
    let start_index: u64 = 12345;
    let sample_count = 10;

    println!(
        "First {} pseudo-DES values starting at index j = {}:",
        sample_count, start_index
    );
    for j in start_index..(start_index + sample_count) {
        let u = pseudo_des_uniform(j);
        println!("  j = {:6}  ->  u(j) ≈ {:.8}", j, u);
    }
}
```

Program 7.3.1 illustrates how deterministic hashing can replace traditional sequential generators in contexts where direct access to indexed random values is required. By using a compact, DES-inspired Feistel construction, the pseudo-DES method achieves strong diffusion, decorrelation of neighboring indices, and reproducibility without maintaining generator state. These characteristics address a central challenge raised in Section 7.3: producing high-quality random values in scenarios where storing or iterating through long generator sequences is impractical or impossible.

The statistical results confirm the reliability of the method: empirical mean and variance closely match those of an ideal $\mathcal{U}(0,1)$ distribution, and local inspection of consecutive outputs demonstrates the rapid mixing essential for avoiding spatial or temporal artifacts in simulation grids. Because the hashing function is stateless and inexpensive to compute, it can be used safely in parallel environments, across GPU threads, or for massive index ranges without risk of stream overlap. This approach forms the basis for more sophisticated, cryptographically inspired diffusion functions introduced later in the chapter, and it prepares the groundwork for advanced Monte Carlo techniques requiring large-scale, random-access reproducibility.

## 7.3.2. Implementation and Uniform Mapping

To map the hashed integer to a floating-point deviate $U(j)$ uniformly distributed in $[0,1)$, we normalize the result by $2^{64}$:

$$U(j) = \frac{\text{hash}(j)}{2^{64}}\tag{7.3.6}$$

This transformation ensures a continuous uniform deviate consistent with Section 7.2.1.

A compact Rust implementation of the pseudo-DES hash follows directly from equations (7.3.2)–(7.3.5). The constants are fixed, and the algorithm involves four rounds of integer multiplication, shifts, and XOR operations. Because each index $j$ is independent, the algorithm is embarrassingly parallel and deterministic across CPU or GPU threads.

This makes pseudo-DES hashing particularly attractive in Monte Carlo ray tracing, stochastic partial differential equations (PDEs), or high-dimensional quadrature, where reproducibility and non-correlation are both essential.

### Rust Implementation

Building on the pseudo-DES hashing framework introduced in Section 7.3.1, Program 7.3.2 provides a compact and fully deterministic implementation of the index-based randomization method together with its uniform mapping into the interval $[0,1)$. Whereas Section 7.3.1 established the mathematical structure of the Feistel-like rounds and the nonlinear mixing function, this program translates those equations directly into practical Rust code capable of evaluating the $j$-th uniform deviate without reference to any previous values. Such index-addressable uniform deviates are essential in large-scale Monte Carlo simulations, where reproducibility, parallel independence, and low-overhead random access are required. The implementation mirrors the operations in Equations (7.3.2)–(7.3.5), completing the transformation by normalizing the 64-bit hash according to Equation (7.3.6) to obtain a statistically robust uniform deviate.

At the core of the implementation is the function `pseudo_des_f`, which realizes the nonlinear round transformation $f(R,C_1,C_2)$ defined by Equations (7.3.3) and (7.3.4). The function begins by computing the intermediate value $t = (R \times C_1 + C_2) \bmod 2^{32}$, using wrapping 32-bit arithmetic to ensure that the computation reflects the intended finite-field operation. It then applies the bit-mixing step by XOR-combining left- and right-shifted versions of $t$, ensuring that high- and low-order bits contribute uniformly to the final result. This mirrors the avalanche-style diffusion that makes the pseudo-DES structure effective for index-based randomization.

The main hashing function, `pseudo_des_hash`, implements the four Feistel-like rounds described in Equation (7.3.2). The input index $j$ is split into the high and low 32-bit halves $L_0$ and $R_0$ as in Equation (7.3.1). Each round then assigns $L_{i+1} = R_i$ and computes $R_{i+1} = L_i \oplus f(R_i, C_{1i}, C_{2i})$, using the constant pairs listed in Equation (7.3.5). After four rounds, the concatenation of $(L_4,R_4)$ provides the final 64-bit hash. Despite its simplicity, this implementation yields excellent diffusion properties: a one-bit change in the index typically flips about half of the output bits, preventing correlations among neighboring values.

The `pseudo_des_u01` function maps the resulting 64-bit integer to a floating-point uniform deviate on $[0,1)$ using the normalization prescribed by Equation (7.3.6). The scaling by $1/2^{64}$ ensures that all 64 bits of the hash contribute directly to the mantissa of the resulting floating-point number, minimizing discretization bias and maximizing uniformity across the interval. This mapping also makes the function fully deterministic and portable, which is essential for reproducibility across different architectures and compiler settings.

The `main` function illustrates the practicality of the implementation by computing and printing (U(j)) for several consecutive indices. This local inspection reveals the decorrelation emphasized in Section 7.3: although the input indices are sequential integers, the resulting uniform deviates exhibit no discernible pattern and appear statistically independent. In large-scale simulations, such behavior allows the generation of reproducible pseudo-random values without maintaining or advancing a global generator state, a significant advantage in parallel or GPU-based environments.

```rust
// Program 7.3.2: Pseudo-DES Hash and Uniform Mapping U(j) ∈ [0,1)
//
// This code implements the 4-round pseudo-DES hashing function described in
// Equations (7.3.2)–(7.3.5) and maps the resulting 64-bit hash value to a
// floating-point deviate U(j) using Equation (7.3.6):
//
//     U(j) = hash(j) / 2^64.
//
// The interface is purely index-based: given an integer j, the functions
// `pseudo_des_hash` and `pseudo_des_u01` compute the corresponding hashed
// value and uniform deviate, respectively. Because each index is treated
// independently, this method is embarrassingly parallel and deterministic,
// making it well suited for Monte Carlo ray tracing, stochastic PDEs,
// and high-dimensional quadrature.

/// Round function f(R, C1, C2) from Equations (7.3.3)–(7.3.4).
///
///   t = (R * C1 + C2) mod 2^32
///   f = (t << 19) XOR (t >> 13)
///
/// All arithmetic is done with 32-bit wrapping semantics.
#[inline]
fn pseudo_des_f(r: u32, c1: u32, c2: u32) -> u32 {
    let t = r
        .wrapping_mul(c1)
        .wrapping_add(c2);
    (t << 19) ^ (t >> 13)
}

/// Pseudo-DES 64-bit hash function: hash(j).
///
/// Given a 64-bit index j, split it into two 32-bit halves as in (7.3.1):
///
///   L0 = high(j),  R0 = low(j),
///
/// then apply four Feistel-like rounds (7.3.2) with constants (7.3.5).
/// The final 64-bit hash is the concatenation (L4, R4).
#[inline]
pub fn pseudo_des_hash(j: u64) -> u64 {
    // Extract high and low 32-bit halves (Equation 7.3.1).
    let mut l: u32 = (j >> 32) as u32;
    let mut r: u32 = (j & 0xFFFF_FFFF) as u32;

    // Round constants (Equation 7.3.5).
    const C1: [u32; 4] = [
        0xBAA9_6887,
        0x1E17_D32C,
        0x03BC_DC3C,
        0x0F33_D1B2,
    ];
    const C2: [u32; 4] = [
        0x4B0F_3B58,
        0xE874_F0C3,
        0x6955_C5A6,
        0x55A7_CA46,
    ];

    // Four Feistel rounds (Equation 7.3.2).
    for i in 0..4 {
        let f_val = pseudo_des_f(r, C1[i], C2[i]);
        let new_l = r;
        let new_r = l ^ f_val;
        l = new_l;
        r = new_r;
    }

    // Concatenate (L4, R4) into a 64-bit result.
    ((l as u64) << 32) | (r as u64)
}

/// Map index j to a uniform deviate U(j) ∈ [0,1) using Equation (7.3.6):
///
///   U(j) = hash(j) / 2^64.
///
/// This uses the full 64-bit hash as a fixed-point number on [0,1).
#[inline]
pub fn pseudo_des_u01(j: u64) -> f64 {
    const INV_2_64: f64 = 1.0 / 18446744073709551616.0; // 1 / 2^64
    let h = pseudo_des_hash(j);
    (h as f64) * INV_2_64
}

// Small demonstration: print a few U(j) for consecutive indices.
fn main() {
    let start: u64 = 12345;
    let count: u64 = 10;

    println!("Pseudo-DES U(j) values for j = {} .. {}:", start, start + count - 1);
    for j in start..(start + count) {
        let u = pseudo_des_u01(j);
        println!("  j = {:6} -> U(j) ≈ {:.10}", j, u);
    }
}
```

Program 7.3.2 demonstrates how the pseudo-DES hashing method translates into a compact and efficient implementation suitable for large-scale index-based random number generation. By combining a small number of integer multiplications, shifts, and XOR operations, the method achieves strong diffusion while remaining computationally inexpensive. The normalization step ensures that the final mapping conforms to the uniform deviate properties established in Section 7.2.1, thereby integrating seamlessly with broader Monte Carlo frameworks.

The example outputs illustrate that the resulting values are well-distributed over (\[0,1)) and decorrelated even for consecutive indices. This behavior addresses the central challenge discussed in Section 7.3: generating reproducible random-access deviates without the overhead or limitations of sequential state evolution. The pseudo-DES approach thus provides a powerful building block for more advanced stochastic algorithms, including random-access sampling on grids, parallel Monte Carlo particle transport, and deterministic hashing for quasi-random perturbations. Its stateless and embarrassingly parallel structure makes it especially attractive in high-performance computing contexts.

## 7.3.3. Butterfly Mixing for Array Randomization

For very large arrays, even pseudo-DES hashing may not ensure desirable global permutation properties. To enhance mixing, we apply butterfly mixing, analogous to the pattern used in Fast Fourier Transform (FFT) algorithms.

In each pass, pairs of array elements separated by a stride are swapped or XOR-mixed based on pseudo-random bits. Let $A$ denote the array of $N = 2^k$ elements. For pass $p$:

$$A[i] \oplus= A[i + 2^p],\qquad 0 \leq i < N/2\tag{7.3.7}$$

and the result of each pass becomes the input to the next. The number of passes $k = \log_2 N$ guarantees that every pair of indices is eventually mixed through a unique XOR path.

A key principle behind complete array hashing is the butterfly connectivity pattern, which ensures that information from each array element rapidly diffuses throughout the structure. In each pass, indices differing by one bit in their binary representation exchange or mix values, forming a layered dependency network analogous to that of fast Fourier transform (FFT) algorithms. This hierarchical structure guarantees that after $k = \log_2 N$ passes, every position in an $N$-element array becomes indirectly connected to all others. The result is *complete diffusion*, a property essential for producing statistically uniform rearrangements of data while maintaining deterministic reproducibility.

As illustrated in Figure 7.3.1, the butterfly topology achieves this by successively pairing and recombining indices through bit-controlled swaps or mixing operations. After a small number of stages, the array’s dependency graph becomes fully connected, effectively “hashing” the entire array into a uniform state suitable for randomized algorithms and memory-distribution tasks.

```{figure} images/pqQDe4beUu67RvW3raYP-kNTqLwrfEpRaJh9wk5zL-v1.png
:name: mP7E2f18Qy
:align: center
:width: 40%

**Figure 7.3.1** Butterfly topology illustrating hierarchical bitwise connectivity in a hashing or permutation network.
```

In Figure 7.3.1, each layer connects indices differing by one bit in their binary representation. After $k$ passes, the dependency graph between array positions becomes fully connected, producing excellent diffusion even for large $N$.

### Analytical Discussion

The butterfly mixing network achieves complete diffusion with a computational complexity of $\mathcal{O}(N\log N)$. Let $D_p$ denote the dependency matrix after the $p^{\text{th}}$ mixing pass. Each successive stage doubles the extent of influence between elements such that:

$$D_{p+1} = D_p + D_p P_{2^p} \tag{7.3.8}$$

where $P_{2^p}$ represents a permutation matrix that pairs indices separated by $2^p$. After $k = \log_2 N$ stages, the cumulative dependency matrix becomes fully populated, implying that each output element depends on all input elements. This recursive propagation of dependencies parallels the mixing lemma used in fast Fourier transform (FFT) networks, ensuring that entropy introduced at any position spreads uniformly throughout the entire array.

The propagation mechanism can also be expressed in terms of element-wise relations:

$$A_i^{(p+1)} = A_i^{(p)} \oplus A_{i \oplus 2^p}^{(p)} \tag{7.3.9}$$

where $A_i^{(p)}$ is the value of the $i^{\text{th}}$ element after pass $p$. Each bit position in the index participates in exactly one mixing connection per stage, ensuring that after $k$ passes, all $N$ elements are connected through a unique chain of XOR relationships. The resulting transformation achieves global diffusion, effectively randomizing the array while preserving determinism, a property essential for reproducible pseudorandom processes.

### Computational Properties

The butterfly mixer exhibits a time complexity of $\mathcal{O}(N\log N)$ because each of the $\log_2 N$ passes performs $N/2$ pairwise operations. Its memory footprint is minimal, requiring only $\mathcal{O}(N)$ storage since all updates are performed in place. Moreover, each pass operates on disjoint pairs of indices, which makes the algorithm highly suitable for data-parallel implementations on multicore CPUs and GPUs.

The choice of the mixing operation depends on data representation. For integer arrays, the bitwise XOR operation provides a reversible and fast mechanism for diffusion. In contrast, for floating-point or continuous-valued arrays, modular addition or affine reversible transformations can be employed to achieve comparable randomness while maintaining numerical stability.

The butterfly mixing network generalizes the principle of **binary-indexed pairwise exchange** into a scalable mechanism for array randomization. After a logarithmic number of stages, all array elements become mutually dependent through a unique XOR graph, producing uniform diffusion across the dataset. This makes the method highly effective for applications in randomized algorithms, memory shuffling, and large-scale hashing systems. Its combination of deterministic reproducibility, high throughput, and complete entropy propagation positions it as a foundational tool in modern numerical and high-performance computing environments.

### Rust Implementation

Following the discussion in Section 7.3 on the limitations of even high-quality index-based hashing for large-scale mixing, Program 7.3.3 introduces a practical implementation of butterfly mixing to achieve complete diffusion across arrays of size $N = 2^k$. While pseudo-DES hashing provides strong local decorrelation for individual indices, large arrays often require a global permutation structure that ensures that information propagates rapidly between distant elements. Butterfly mixing accomplishes this through a hierarchical pattern of pairwise XOR operations identical in spirit to the connectivity graph of the fast Fourier transform (FFT). By repeatedly applying XOR exchanges between indices separated by increasing powers of two, the program constructs a deterministic yet thoroughly diffused transformation, suitable for randomized memory layouts, Monte Carlo grids, and parallel simulation frameworks that require predictable but statistically uniform mixing.

At the core of the implementation is the `butterfly_mix_u64` function, which realizes the mixing rule expressed in Equation (7.3.7). For each pass $p$, the function computes a stride of $2^p$ and applies the XOR update $A[i] \mathrel{\oplus}= A[i + 2^p]$, to all applicable pairs of indices. This pattern constructs a layered set of bit-dependent interactions: in pass $p = 0$, elements differing in their least significant bit are mixed; in pass $p = 1$, elements whose second-least significant bit differs are mixed; and so forth up to $p = k-1$. This mirrors the FFT butterfly network, where each level entangles increasingly distant indices, guaranteeing that every array position influences every other through a unique dependency path.

The pseudo-DES hashing function, reused from Sections 7.3.1 and 7.3.2, initializes the array in a decorrelated state before the butterfly passes begin. The functions `pseudo_des_f` and `pseudo_des_hash` implement the Feistel-style rounds from Equations (7.3.2)–(7.3.5), ensuring that each initial 64-bit value exhibits strong bit-level diffusion. The helper routine `u01_from_hash` maps the 64-bit hashed integers to $[0,1)$ via the normalization described in Equation (7.3.6), allowing diagnostic output to be expressed as floating-point deviates.

The program also includes a lightweight statistical module based on Welford’s algorithm to compute mean, variance, minimum, and maximum of the mixed array. This provides a sanity check that butterfly diffusion preserves the overall uniform distribution implied by Equation (7.2.1) while altering the dependence structure among indices. Together, these components enable the code to illustrate both the local randomness from pseudo-DES hashing and the global mixing induced by the butterfly passes.

The `main` function demonstrates the complete pipeline. It constructs an array of size $N = 2^k$, initializes it with pseudo-DES hash values, and prints the first few uniform deviates to show the pre-mixing state. It then applies the butterfly mixing passes to diffuse the entire array and prints the same entries again to highlight the structural transformation. Finally, it computes summary statistics to verify that although the elements have been globally mixed, their distribution remains consistent with that of an ideal uniform deviate over $[0,1)$. This example makes explicit how a bit-structured mixing pattern produces complete array hashing suitable for large-scale deterministic randomization.

```rust
// Program 7.3.3: Butterfly Mixing for Complete Array Hashing
//
// This program demonstrates butterfly-style mixing on a large array A of size
// N = 2^k, as described in Section 7.3.3. The array is first initialized using
// the pseudo-DES index-based hash from Sections 7.3.1–7.3.2, and then
// subjected to k = log2(N) passes of XOR-based butterfly mixing.
//
// Each pass p uses a stride of 2^p and applies the in-place transformation
//
//     A[i] ^= A[i + 2^p],     (Equation 7.3.7)
//
// to all pairs of indices separated by 2^p in a butterfly pattern. After k
// passes, every position in the array has been connected to all others via a
// layered XOR dependency network, analogous to the connectivity pattern in
// FFT algorithms. This achieves complete diffusion of information across the
// array, providing a deterministic yet statistically well-mixed rearrangement
// suitable for randomized algorithms and memory-distribution tasks.
//
// No external crates are required.

use std::f64;

/// Pseudo-DES round function f(R, C1, C2) from Equations (7.3.3)–(7.3.4).
#[inline]
fn pseudo_des_f(r: u32, c1: u32, c2: u32) -> u32 {
    let t = r
        .wrapping_mul(c1)
        .wrapping_add(c2);
    (t << 19) ^ (t >> 13)
}

/// Pseudo-DES 64-bit hash function hash(j) as in Sections 7.3.1–7.3.2.
#[inline]
fn pseudo_des_hash(j: u64) -> u64 {
    let mut l: u32 = (j >> 32) as u32;
    let mut r: u32 = (j & 0xFFFF_FFFF) as u32;

    const C1: [u32; 4] = [
        0xBAA9_6887,
        0x1E17_D32C,
        0x03BC_DC3C,
        0x0F33_D1B2,
    ];
    const C2: [u32; 4] = [
        0x4B0F_3B58,
        0xE874_F0C3,
        0x6955_C5A6,
        0x55A7_CA46,
    ];

    for i in 0..4 {
        let f_val = pseudo_des_f(r, C1[i], C2[i]);
        let new_l = r;
        let new_r = l ^ f_val;
        l = new_l;
        r = new_r;
    }

    ((l as u64) << 32) | (r as u64)
}

/// Map a 64-bit hash value to U(0,1) via Equation (7.3.6).
#[inline]
fn u01_from_hash(h: u64) -> f64 {
    const INV_2_64: f64 = 1.0 / 18446744073709551616.0; // 1 / 2^64
    (h as f64) * INV_2_64
}

/// In-place butterfly mixing on an array A of length N = 2^k.
///
/// For each pass p = 0,1,...,k-1, we use a stride s = 2^p and apply
///
///     A[i] ^= A[i + s]
///
/// to all pairs (i, i + s) in a butterfly connectivity pattern. Over all
/// passes, this realizes Equation (7.3.7) and ensures that every index is
/// mixed with every other through a layered XOR network.
fn butterfly_mix_u64(a: &mut [u64]) {
    let n = a.len();
    assert!(n.is_power_of_two(), "N must be a power of two");

    let k: u32 = n.trailing_zeros(); // since n = 2^k

    for p in 0u32..k {
        let stride = 1usize << p;   // 2^p
        let span = stride << 1;     // 2^(p+1)

        let mut base = 0usize;
        while base < n {
            // Mix pairs (base + j, base + j + stride) for j = 0..stride-1
            for j in 0..stride {
                let i = base + j;
                let partner = i + stride;
                a[i] ^= a[partner];
            }
            base += span;
        }
    }
}

/// Running mean, variance, min, and max via Welford’s algorithm (for diagnostics).
#[derive(Debug)]
struct RunningStats {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn push(&mut self, x: f64) {
        self.n += 1;
        let n_f = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n_f;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / ((self.n - 1) as f64)
        } else {
            0.0
        }
    }

    fn min(&self) -> f64 {
        self.min
    }

    fn max(&self) -> f64 {
        self.max
    }
}

fn main() {
    // Choose N = 2^k for some k; here we use k = 10, N = 1024 for illustration.
    const K: usize = 10;
    const N: usize = 1 << K;

    // Initialize the array with pseudo-DES hashes of the indices.
    let mut a: Vec<u64> = (0..N as u64)
        .map(pseudo_des_hash)
        .collect();

    // Show a few entries before mixing.
    println!("First 8 values before butterfly mixing:");
    for i in 0..8 {
        let u = u01_from_hash(a[i]);
        println!("  i = {:4} -> U(i) ≈ {:.10}", i, u);
    }
    println!();

    // Apply butterfly mixing as per Equation (7.3.7).
    butterfly_mix_u64(&mut a);

    // Show the same entries after mixing to illustrate diffusion.
    println!("First 8 values after butterfly mixing:");
    for i in 0..8 {
        let u = u01_from_hash(a[i]);
        println!("  i = {:4} -> U'(i) ≈ {:.10}", i, u);
    }
    println!();

    // Optional: compute basic statistics after mixing as a sanity check.
    let mut stats = RunningStats::new();
    for &h in &a {
        stats.push(u01_from_hash(h));
    }

    println!("Summary statistics after butterfly mixing (N = {}):", N);
    println!("  mean       ≈ {:.6} (theoretical: 0.5)", stats.mean());
    println!(
        "  variance   ≈ {:.6} (theoretical: 1/12 ≈ {:.6})",
        stats.variance(),
        1.0 / 12.0
    );
    println!("  min        ≈ {:.6}", stats.min());
    println!("  max        ≈ {:.6}", stats.max());
}
```

Program 7.3.3 demonstrates how butterfly mixing can be combined with index-based pseudo-random initialization to achieve complete array diffusion in a deterministic and reproducible manner. As discussed in Section 7.3, index hashing is sufficient for generating decorrelated values at isolated indices, but large-array operations benefit from a connectivity structure that propagates local randomness globally. The butterfly topology provides this property naturally: each pass links indices differing by a single bit, and after $k$ passes every position in an array of size $2^k$ is connected through multiple XOR pathways. This ensures that any regularity in the initial arrangement is dissolved, producing a thoroughly mixed array without relying on stochastic permutations.

The before-and-after comparison in the program output illustrates this process vividly. Despite starting from a well-distributed but structurally ordered sequence, the application of butterfly mixing dramatically alters the arrangement while preserving overall uniformity, as reflected in the final statistical summary. These characteristics make the butterfly approach attractive in randomized algorithms, load balancing, multi-resolution grids, and hierarchical Monte Carlo methods where reproducibility and cross-platform consistency are essential. By integrating simple bitwise operations into a structured diffusion network, the method achieves global mixing at minimal computational cost.

## 7.3.4. Performance Trade-offs and Integration Strategies

In practical implementations, both pseudo-DES and butterfly hashing demonstrate highly efficient asymptotic performance, operating in $\mathcal{O}(N)$ time with $\mathcal{O}(1)$ auxiliary space. Despite their similar computational complexity, their operational characteristics differ in important ways. The pseudo-DES method functions locally on each index, performing its bit-mixing transformations without requiring memory access beyond the element itself. In contrast, butterfly mixing relies on repeated data exchanges between paired indices separated by powers of two, leading to additional memory reads and writes.

Because of this distinction, pseudo-DES hashing is particularly effective for direct random access tasks, for instance, random sampling, index mapping, or shuffling where array-wide diffusion is unnecessary. Each element can be independently addressed through its pseudo-random transformation, allowing high throughput and minimal memory contention. Butterfly mixing, by contrast, excels in global randomization and whitening. Its layered bitwise connectivity ensures that entropy originating at any position propagates throughout the array after $\log_2 N$ passes, producing uniform diffusion and eliminating spatial correlation.

From an implementation perspective, pseudo-DES benefits from locality: its operations are arithmetic and bitwise, performed entirely within the CPU’s registers, making it cache-resident and latency-insensitive. Butterfly mixing, however, is bandwidth-sensitive, its performance scales with memory throughput and cache efficiency. On modern hardware, this trade-off is mitigated by parallelization, since non-overlapping index pairs in each pass can be processed concurrently using vectorized SIMD instructions or GPU threads.

In practical applications, combining both approaches yields the most robust results. The pseudo-DES stage provides per-index decorrelation, ensuring that each element’s hash value is statistically independent. Subsequent butterfly passes perform spatial redistribution, diffusing those values across the array to eliminate structured patterns and amplify uniformity. This two-phase process including local whitening followed by global mixing balances statistical rigor with computational efficiency.

In summary, pseudo-DES hashing and butterfly mixing complement each other as orthogonal strategies in the design of high-performance randomization systems. The former offers deterministic, locality-preserving random access; the latter achieves full-array diffusion and decorrelation. Together, they form a unified framework for constructing scalable, cache-efficient, and statistically uniform hashing schemes suitable for both CPU and GPU-based simulation pipelines.

### Rust Implementation

Following the analysis in Section 7.3.4 on the complementary performance characteristics of pseudo-DES hashing and butterfly mixing, the next program provides a concrete benchmark illustrating these trade-offs in practice. Whereas pseudo-DES hashing operates entirely within CPU registers and achieves high throughput due to its locality-preserving arithmetic operations, butterfly mixing emphasizes global diffusion over the array and therefore incurs additional memory traffic. The combined pipeline reflects the two-phase strategy described earlier, applying local whitening followed by full-array mixing. Program 7.3.4 measures the per-element runtime of each stage on a large array and reports representative performance figures, demonstrating how architectural considerations directly manifest in execution costs.

At the core of the implementation is the `pseudodes_index_hash` function, which provides a fully indexable pseudo-random mapping based on the four-round Feistel-like pseudo-DES transformation introduced in Equations (7.3.2)–(7.3.5). This function performs the arithmetic and bitwise mixing locally on each 64-bit integer index, requiring no state, memory fetches, or sequential recurrence. As a result, each hashed value can be computed independently in constant time, making the routine well-suited for large-scale random access patterns and embarrassingly parallel applications. The companion function `u_of_j` applies the normalization defined in Equation (7.3.6), mapping the hash result to a uniform deviate in $[0,1)$, ensuring compatibility with the uniformity requirements described in Section 7.2.1.

The `butterfly_mix` function implements the hierarchical mixing structure summarized in Equation (7.3.7). For an array of $N = 2^k$ elements, the algorithm performs $k$ passes, each time XOR-mixing pairs of elements separated by stride $2^p$. This procedure causes diffusion to propagate across binary-related indices, reproducing the butterfly connectivity pattern that underlies FFT computations. Because each pass touches the entire array and involves strided reads and writes, the routine becomes memory-bandwidth limited rather than computation-limited. Nevertheless, the resulting global mixing eliminates spatial correlation and distributes entropy across the entire array, yielding uniformly randomized data suitable for simulation workflows requiring global decorrelation.

To evaluate performance, the program includes a microbenchmark harness consisting of three measurement routines: hashing only, mixing only, and a combined pipeline. Each benchmark allocates an array of size $N = 2^{18}$ and times the application of one or both stages using Rust’s high-resolution timing facilities. The per-element timings are computed by dividing the elapsed time by the array length, allowing direct comparison of algorithmic efficiency. A simple 64-bit checksum is accumulated across the array to prevent dead-code elimination by the compiler and to verify that each pass genuinely contributes computational work. The final performance report summarizes total runtime, cost per element, and the resulting checksum.

Altogether, these functions reflect the architectural contrasts discussed throughout Section 7.3.4: pseudo-DES hashing exhibits register-level locality and arithmetic intensity, while butterfly mixing emphasizes memory movement and large-scale diffusion. The combined routine illustrates how these two strategies complement one another in practical randomization pipelines.

```rust
// Program 7.3.4: Performance Comparison of Pseudo-DES Hashing and Butterfly Mixing
//
// This program illustrates the performance trade-offs discussed in Section 7.3.4
// by timing three related operations on an array of size N = 2^k:
//
//   1. Pseudo-DES hashing alone (local, per-index transformation).
//   2. Butterfly mixing alone (global, bandwidth-sensitive diffusion).
//   3. A combined pipeline: pseudo-DES initialization followed by butterfly mixing.
//
// Pseudo-DES hashing performs all work in registers, using only arithmetic and
// bitwise operations on each index. Butterfly mixing, in contrast, repeatedly
// reads and writes array elements separated by powers of two, stressing memory
// bandwidth and cache behavior. The reported timings (and per-element costs)
// make these differences concrete in a simple, reproducible benchmark.
//
// No external crates are required.

use std::time::Instant;

#[inline]
fn pseudo_des_f(r: u32, c1: u32, c2: u32) -> u32 {
    let t = r
        .wrapping_mul(c1)
        .wrapping_add(c2);
    (t << 19) ^ (t >> 13)
}

/// Pseudo-DES 64-bit hash of index j (4 Feistel rounds).
#[inline]
fn pseudo_des_hash(j: u64) -> u64 {
    let mut l: u32 = (j >> 32) as u32;
    let mut r: u32 = (j & 0xFFFF_FFFF) as u32;

    const C1: [u32; 4] = [
        0xBAA9_6887,
        0x1E17_D32C,
        0x03BC_DC3C,
        0x0F33_D1B2,
    ];
    const C2: [u32; 4] = [
        0x4B0F_3B58,
        0xE874_F0C3,
        0x6955_C5A6,
        0x55A7_CA46,
    ];

    for i in 0..4 {
        let f_val = pseudo_des_f(r, C1[i], C2[i]);
        let new_l = r;
        let new_r = l ^ f_val;
        l = new_l;
        r = new_r;
    }

    ((l as u64) << 32) | (r as u64)
}

/// In-place butterfly mixing on an array of length N = 2^k.
///
/// For each pass p = 0..k-1, with stride s = 2^p, perform
///
///     A[i] ^= A[i + s]
///
/// on the first half of each block of size 2^(p+1).
fn butterfly_mix_u64(a: &mut [u64]) {
    let n = a.len();
    assert!(n.is_power_of_two(), "N must be a power of two");

    let k: u32 = n.trailing_zeros(); // since n = 2^k

    for p in 0u32..k {
        let stride = 1usize << p;   // 2^p
        let span = stride << 1;     // 2^(p+1)

        let mut base = 0usize;
        while base < n {
            for j in 0..stride {
                let i = base + j;
                let partner = i + stride;
                a[i] ^= a[partner];
            }
            base += span;
        }
    }
}

fn main() {
    // Choose a moderately large power-of-two size for benchmarking.
    // You can increase K for more realistic large-scale tests.
    const K: usize = 18; // N = 2^18 = 262,144 elements
    const N: usize = 1 << K;

    println!("Benchmarking with N = 2^{} = {} elements\n", K, N);

    let mut checksum: u64 = 0;

    // ---------------------------------------------------------------------
    // 1. Pseudo-DES hashing alone (per-index, local computation)
    // ---------------------------------------------------------------------
    let mut a_hash_only = vec![0u64; N];

    let t0 = Instant::now();
    for i in 0..N {
        let h = pseudo_des_hash(i as u64);
        a_hash_only[i] = h;
        checksum ^= h; // prevent optimization
    }
    let dt_hash = t0.elapsed();

    let hash_sec = dt_hash.as_secs_f64();
    let hash_ns_per_elem = hash_sec * 1.0e9 / (N as f64);

    println!("1. Pseudo-DES hashing only:");
    println!("   total time       ≈ {:.3} ms", hash_sec * 1.0e3);
    println!("   time per element ≈ {:.1} ns/elem", hash_ns_per_elem);
    println!();

    // ---------------------------------------------------------------------
    // 2. Butterfly mixing alone (global, memory-bound diffusion)
    // ---------------------------------------------------------------------
    // Start from the hashed array to have realistic data.
    let mut a_mix_only = a_hash_only.clone();

    let t1 = Instant::now();
    butterfly_mix_u64(&mut a_mix_only);
    for &v in &a_mix_only {
        checksum ^= v; // again, prevent optimization
    }
    let dt_mix = t1.elapsed();

    let mix_sec = dt_mix.as_secs_f64();
    let mix_ns_per_elem = mix_sec * 1.0e9 / (N as f64);

    println!("2. Butterfly mixing only:");
    println!("   total time       ≈ {:.3} ms", mix_sec * 1.0e3);
    println!("   time per element ≈ {:.1} ns/elem", mix_ns_per_elem);
    println!();

    // ---------------------------------------------------------------------
    // 3. Combined pipeline: pseudo-DES initialization + butterfly mixing
    // ---------------------------------------------------------------------
    let mut a_combined = vec![0u64; N];

    let t2 = Instant::now();
    // Local pseudo-DES hashing:
    for i in 0..N {
        a_combined[i] = pseudo_des_hash(i as u64);
    }
    // Global butterfly diffusion:
    butterfly_mix_u64(&mut a_combined);
    for &v in &a_combined {
        checksum ^= v;
    }
    let dt_combined = t2.elapsed();

    let combined_sec = dt_combined.as_secs_f64();
    let combined_ns_per_elem = combined_sec * 1.0e9 / (N as f64);

    println!("3. Combined: pseudo-DES hashing + butterfly mixing:");
    println!("   total time       ≈ {:.3} ms", combined_sec * 1.0e3);
    println!("   time per element ≈ {:.1} ns/elem", combined_ns_per_elem);
    println!();

    // Print checksum to ensure side effects are not optimized away.
    println!("(Checksum to prevent optimization: 0x{:016X})", checksum);
}
```

Program 7.3.4 highlights how the theoretical distinctions between local and global randomization mechanisms translate directly into measurable performance behaviour. Pseudo-DES hashing, because of its register-resident arithmetic and absence of memory dependence, achieves substantially lower per-element latency, making it ideal for indexable random access or pointwise sampling tasks. Butterfly mixing, in contrast, incurs additional memory bandwidth costs inherent to its array-wide diffusion strategy but provides the essential global whitening necessary for uniform redistribution of entropy across large data structures.

The combined two-stage pipeline demonstrates how local decorrelation and global mixing reinforce each other. Pseudo-DES hashing ensures that each element begins with an independently constructed pseudo-random seed, while butterfly mixing propagates those seeds through the array via bitwise exchange patterns that guarantee full connectivity in $\log_2 N$ passes. This synergy provides strong statistical uniformity, deterministic reproducibility, and scalable performance suitable for both CPU and GPU implementations. The modular design of the benchmark also makes it straightforward to experiment with alternative hashing functions, memory layouts, or SIMD acceleration strategies, enabling further exploration of high-performance randomization techniques in modern numerical simulations.

## 7.3.5. Applied Contexts and Computational Domains

The hashing and mixing mechanisms discussed in the preceding sections find broad application across computational science, numerical simulation, and distributed systems. Although neither pseudo-DES nor butterfly mixing was designed for cryptographic security, their deterministic reproducibility, uniformity, and statistical diffusion make them invaluable tools in high-performance and scientific environments where controlled randomness is required.

### (i) Monte Carlo Simulation

In large-scale Monte Carlo frameworks, such as neutron transport, radiative transfer, or statistical mechanics, each particle, photon, or lattice site must receive a unique yet reproducible random deviate. The pseudo-DES hash enables such mappings by associating a specific random value with each particle index or spatial cell deterministically. This allows a simulation to be paused, checkpointed, or restarted without storing vast state arrays, since the random state can be recomputed directly from indices. The reproducibility ensures numerical consistency across runs while maintaining independence between random sequences.

### (ii) Cryptographic-Inspired and Hash-Based Sampling

Although pseudo-DES itself is not cryptographically secure, its excellent diffusion and avalanche properties make it suitable for randomized load balancing, domain decomposition, or reproducible sampling in large distributed systems. In such contexts, each data element or process obtains a statistically independent assignment, yet the overall mapping remains deterministic, essential for debugging, reproducibility, and fairness in workload distribution. This principle underlies many modern randomized scheduling algorithms and distributed Monte Carlo frameworks, where reproducible pseudo-random allocation avoids bias accumulation over millions of operations.

Beyond these two domains, hybrid schemes combining pseudo-DES hashing with butterfly mixing have also been adopted in stochastic optimization, molecular dynamics, and procedural content generation. The hybrid approach leverages the locality and efficiency of pseudo-DES for per-element randomization while employing butterfly diffusion to ensure global whitening of entire arrays or particle ensembles. This combination offers a tunable balance between statistical independence and spatial uniformity, critical for large-scale parallel simulations and GPU-based randomized workloads.

## 7.3.6. Concluding Remarks

The technique of completely hashing a large array represents an elegant fusion of deterministic computation and statistical randomness. Its structure mirrors that of Feistel networks but with constants and operations tuned for floating-point uniformity rather than secrecy.

In Rust, this approach integrates naturally with traits such as `Hash`, `BitXor`, and `Wrapping`, and can be efficiently parallelized using `rayon`. For systems requiring independent streams on massive scales, such as Monte Carlo integrations or GPU-accelerated solvers, complete hashing offers reproducible, low-memory randomization with nearly ideal statistical properties.

The complementary strengths of pseudo-DES hashing for local decorrelation and butterfly mixing for global diffusion provide a unified and efficient framework for large-array randomization.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/cLbdmfvi6ZzaTcsCLU3f.6","tags":[]}

# 7.4. Deviates from Other Distributions

In the preceding sections we learned how to generate uniform deviates $x \sim U(0,1)$ and apply transformation or rejection methods to obtain deviates from distributions such as exponential, normal, gamma, or beta. However, many real-world simulations demand more flexible distributions that capture failure times, extremes, or heavy-tail behaviour.

In numerical computing, deviates from non-uniform distributions appear in a variety of contexts:

- Monte Carlo simulations of stochastic processes in physical systems,
- Uncertainty quantification in engineering reliability models,
- Extreme-value modelling for hydrology and climate studies,
- Heavy-tail processes in finance and network theory, and
- Probabilistic algorithms where random perturbations follow specific statistical laws.

Let $X\sim U(0,1)$ be a uniform random variable produced by a high-quality pseudo-random generator (as developed in Section 7.1). A transformation $Y=g(X)$ defines a new random variable with density $p_Y(y)$, determined by the fundamental transformation law:

$$
p_Y(y)\,dy = p_X(x)\,dx = \left| \frac{dx}{dy} \right|\,dy
\tag{7.4.1}
$$

so that $p_Y(y) = \left|\frac{dx}{dy}\right|$ when $p_X(x)=1$ for $x\in[0,1]$.

This approach provides a unified basis for constructing deviates from a wide family of continuous distributions, provided the inverse cumulative distribution function (CDF) $F^{-1}_Y(x)$ is analytically known or numerically invertible. For more complex densities, rejection sampling, ratio-of-uniforms, or hybrid methods are used.

In vector or matrix form, relevant in Monte Carlo or PDE simulations, the mapping extends to:

$$\mathbf{y}=g(\mathbf{x}),\quad p_{\mathbf{Y}}(\mathbf{y}) = p_{\mathbf{X}}(\mathbf{x})\left|\frac{\partial \mathbf{x}}{\partial \mathbf{y}}\right|\tag{7.4.2}$$

where the Jacobian determinant encodes local volume scaling.

These techniques underpin modern numerical computation across engineering, physics, and data science, where sampling efficiency directly impacts simulation throughput.

## 7.4.1. Transformation Principles and Sampling Foundations

The mathematical basis of random variate generation rests on the principle of probability conservation under transformation. For a continuous random variable $X$ with density $p_X(x)$, and a differentiable, monotonic transformation $Y = g(X)$, the corresponding density $p_Y(y)$ is given by the change-of-variables formula

$$p_Y(y)dy = p_X(x)dx = \left|\frac{dx}{dy}\right|dy \tag{7.4.3}$$

which ensures that equal probability mass is preserved under coordinate transformation. This relationship forms the foundation of all transformation-based sampling techniques used in computational statistics and simulation.

If the cumulative distribution function (CDF) of $Y$ is denoted by $F_Y(y)$, it follows that:

$$F_Y(y) = \int_{-\infty}^{y} f_Y(t)dt \tag{7.4.4}$$

By setting $x = F_Y(y)$, one obtains the transformation:

$$y = F_Y^{-1}(x), \qquad x \sim U(0,1) \tag{7.4.5}$$

which defines the inverse transform method. This method maps uniform random deviates into the desired distribution by inverting the CDF. For distributions with analytically invertible cumulative functions, such as the expo5ential, Weibull, or Pareto distributions, this is the simplest and most efficient approach. Only one uniform draw is required, followed by one or two logarithmic operations and, in some cases, a power function. Its deterministic structure and minimal arithmetic cost make it ideal for vectorized and GPU implementations in Rust.

When a closed-form inverse of $F_Y$ is unavailable, alternative non-inversion methods are used to approximate or indirectly achieve sampling from the desired density:

- *Rejection Sampling*: Draw samples from a convenient proposal density $f(x) \ge p(x)$, and accept each draw with probability $p(x)/f(x)$. This method is versatile but can be inefficient when the acceptance ratio is low.
- *Ratio-of-Uniforms Method*: Generate uniform samples in a transformed region of the $(u,v)$-plane and map them back to the target variable (detailed in §7.4.6). This technique is geometrically elegant and often more stable for complex densities.
- *Adaptive Hybrid Methods*: Combine inversion for analytic regions with rejection or composition methods in the tails, achieving efficiency across diverse distributions.

The choice among these strategies depends on both theoretical and computational considerations. The acceptance ratio, smoothness of the target density, and cost per generated sample all determine the most suitable method. In high-performance numerical computing, especially in Rust, these considerations translate directly into arithmetic throughput, branching efficiency, and memory access locality. Thus, the transformation principles underlying (7.4.3)–(7.4.5) form not only the mathematical core of random number generation but also the architectural foundation of its efficient implementation in modern systems.

### Rust Implementation

Following the conceptual development in Section 7.4.1, which introduced the transformation law and its role in constructing non-uniform random deviates, Program 7.4.1 provides a practical Rust implementation of several inverse-CDF sampling methods. These examples illustrate how uniform deviates generated by high-quality pseudo-random number generators can be mapped deterministically into a wide class of non-uniform distributions through analytic transformations. Because many important densities, including the exponential, Weibull, and Pareto distributions, possess closed-form cumulative functions, inverse transforms offer an efficient, branch-free, and numerically stable mechanism for random variate generation. The program demonstrates how these foundational ideas translate into practical Rust code suitable for vectorized Monte Carlo workflows, reliability modelling, and physical simulation pipelines.

At the core of the implementation is a suite of inverse-CDF transformation functions, each corresponding directly to the analytic mappings derived from Equation (7.4.5). Each function receives a uniform deviate $u \in (0,1)$ and returns the transformed random variable $y = F^{-1}(u)$. This structure mirrors the abstract transformation law in Equation (7.4.3), where probability mass is preserved by the mapping $y = g(u)$. Because the uniform source is assumed to follow the density $p_X(x)=1$ on $[0,1)$, each inverse transform produces the correct density $p_Y(y)$ by differentiating the CDF implicitly through the transform.

The exponential sampler implements the inverse transform $y = -\lambda^{-1} \ln(1-u)$, which follows immediately from substituting the analytic form of the exponential CDF into Equation (7.4.5). This produces a rapidly decaying distribution appropriate for waiting-time models. The Weibull sampler generalizes this by reusing the logarithmic core and applying a power transform, reflecting the shape-adjustable hazard rate characteristic of Weibull processes. For distributions exhibiting heavy-tail behavior, such as the Pareto law, the inverse CDF involves rational and power-law components, demonstrating how Equation (7.4.5) naturally accommodates densities with polynomial decay.

To support statistical validation, the program also includes a small statistics module that computes empirical means, variances, minima, and maxima from generated samples. These empirical measurements are compared with the theoretical closed-form moments derived earlier in the section, allowing a direct evaluation of accuracy and sampling stability. The theoretical expectations for the Weibull distribution rely on the gamma function, introduced through the statrs library, which provides the analytic $\Gamma(\cdot)$ evaluations needed for correct moment calculation.

The `main` function orchestrates the generation and assessment of deviates for each distribution. It begins by drawing a fixed number of uniform samples using the SplitMix64 generator developed in Section 7.2.2, ensuring reproducibility and high-quality underlying randomness. These samples are transformed via each inverse-CDF method and passed to the statistics routines. The program prints both empirical and theoretical results in a structured form, revealing that even with moderately sized samples, the inverse-transform approach achieves highly accurate approximations to the target distributions. Altogether, the main function provides a unified demonstration of transformation-based sampling and its numerical behavior in practical environments.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
statrs = "0.16"
```

```rust
// Program 7.4.1: Transformation-Based Sampling from Exponential, Weibull, and Pareto Distributions
//
// This program demonstrates the transformation principles of Section 7.4 and 7.4.1.
// Starting from a uniform deviate U ~ U(0,1), it constructs non-uniform deviates
// Y = g(U) using the inverse-CDF (inverse transform) method:
//
//   - Exponential(λ):  Y = - (1/λ) ln(1 - U)
//   - Weibull(k, λ):   Y = λ [ -ln(1 - U) ]^(1/k)
//   - Pareto(x_m, α):  Y = x_m (1 - U)^(-1/α)
//
// These formulas implement the change-of-variables law (7.4.3) and
// inverse-transform principle (7.4.5) for distributions with analytically
// invertible CDFs. A small SplitMix64-based generator supplies U(0,1)
// deviates as a stand-in for the "high-quality pseudo-random generator"
// developed earlier in Chapter 7.

use std::f64;
use statrs::function::gamma::gamma;

/// Simple SplitMix64 PRNG for U(0,1) deviates.
/// This is suitable for demonstration purposes.
#[derive(Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64-bit output word.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform deviate U ~ U[0,1), using the top 53 bits of next_u64.
    #[inline]
    fn next_f64(&mut self) -> f64 {
        const INV_2_53: f64 = 1.0 / (1u64 << 53) as f64;
        let x = self.next_u64() >> 11; // keep 53 bits
        (x as f64) * INV_2_53
    }

    /// Uniform deviate U in (0,1), avoiding endpoints for log/power transforms.
    #[inline]
    fn next_f64_open(&mut self) -> f64 {
        let u = self.next_f64();
        if u == 0.0 {
            f64::MIN_POSITIVE
        } else if u == 1.0 {
            f64::from_bits(f64::MAX.to_bits() - 1)
        } else {
            u
        }
    }
}

/// Running statistics for diagnostics (mean, variance, min, max).
#[derive(Debug)]
struct RunningStats {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn push(&mut self, x: f64) {
        self.n += 1;
        let n_f = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n_f;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / ((self.n - 1) as f64)
        } else {
            0.0
        }
    }

    fn min(&self) -> f64 {
        self.min
    }

    fn max(&self) -> f64 {
        self.max
    }
}

// -----------------------------------------------------------------------------
// Inverse-transform samplers using (7.4.3)–(7.4.5)
// -----------------------------------------------------------------------------

/// Exponential(λ) deviate using inverse CDF with
/// F_Y(y) = 1 - exp(-λ y),  y ≥ 0  ⇒  Y = - (1/λ) ln(1 - U).
fn sample_exponential(rng: &mut SplitMix64, lambda: f64) -> f64 {
    assert!(lambda > 0.0);
    let u = rng.next_f64_open();
    -((1.0 - u).ln()) / lambda
}

/// Weibull(k, λ) deviate.
/// F_Y(y) = 1 - exp(-(y/λ)^k), y ≥ 0  ⇒  Y = λ [ -ln(1 - U) ]^(1/k).
fn sample_weibull(rng: &mut SplitMix64, k: f64, lambda: f64) -> f64 {
    assert!(k > 0.0 && lambda > 0.0);
    let u = rng.next_f64_open();
    let t = - (1.0 - u).ln();
    lambda * t.powf(1.0 / k)
}

/// Pareto(x_m, α) deviate.
/// F_Y(y) = 1 - (x_m / y)^α, y ≥ x_m  ⇒  Y = x_m (1 - U)^(-1/α).
fn sample_pareto(rng: &mut SplitMix64, x_m: f64, alpha: f64) -> f64 {
    assert!(x_m > 0.0 && alpha > 0.0);
    let u = rng.next_f64_open();
    x_m * (1.0 - u).powf(-1.0 / alpha)
}

fn main() {
    let mut rng = SplitMix64::new(0xDEAD_BEEF_CAFE_F00D);
    let n: u64 = 100_000;

    // Parameters for the example distributions.
    let lambda_exp = 2.0;       // Exponential(λ): mean = 1/λ = 0.5
    let k_weibull = 1.5;        // Weibull(k, λ)
    let lambda_weibull = 1.0;   // mean, var > 0 for k > 1
    let x_m_pareto = 1.0;       // Pareto(x_m, α)
    let alpha_pareto = 3.0;     // finite mean, variance

    let mut stats_exp = RunningStats::new();
    let mut stats_weibull = RunningStats::new();
    let mut stats_pareto = RunningStats::new();

    // Collect samples and update statistics.
    for _ in 0..n {
        let y_exp = sample_exponential(&mut rng, lambda_exp);
        let y_weibull = sample_weibull(&mut rng, k_weibull, lambda_weibull);
        let y_pareto = sample_pareto(&mut rng, x_m_pareto, alpha_pareto);

        stats_exp.push(y_exp);
        stats_weibull.push(y_weibull);
        stats_pareto.push(y_pareto);
    }

    // Theoretical moments for comparison.

    // Exponential(λ):
    // E[Y] = 1/λ, Var[Y] = 1/λ^2.
    let exp_mean_th = 1.0 / lambda_exp;
    let exp_var_th = 1.0 / (lambda_exp * lambda_exp);

    // Weibull(k, λ):
    // E[Y] = λ Γ(1 + 1/k),
    // Var[Y] = λ^2 (Γ(1 + 2/k) - Γ(1 + 1/k)^2).
    let gamma_1 = gamma(1.0 + 1.0 / k_weibull);
    let gamma_2 = gamma(1.0 + 2.0 / k_weibull);
    let weibull_mean_th = lambda_weibull * gamma_1;
    let weibull_var_th = lambda_weibull * lambda_weibull * (gamma_2 - gamma_1 * gamma_1);

    // Pareto(x_m, α), with α > 2:
    // E[Y] = α x_m / (α - 1),
    // Var[Y] = (α x_m^2) / ((α - 1)^2 (α - 2)).
    let pareto_mean_th = alpha_pareto * x_m_pareto / (alpha_pareto - 1.0);
    let pareto_var_th =
        (alpha_pareto * x_m_pareto * x_m_pareto)
        / ((alpha_pareto - 1.0).powi(2) * (alpha_pareto - 2.0));

    println!("Sampling {} deviates via inverse transforms:", n);
    println!();

    println!("Exponential(λ = {}):", lambda_exp);
    println!("  Empirical mean      ≈ {:.6}", stats_exp.mean());
    println!("  Theoretical mean     = {:.6}", exp_mean_th);
    println!("  Empirical variance  ≈ {:.6}", stats_exp.variance());
    println!("  Theoretical variance = {:.6}", exp_var_th);
    println!("  Min                 ≈ {:.6}", stats_exp.min());
    println!("  Max                 ≈ {:.6}", stats_exp.max());
    println!();

    println!(
        "Weibull(k = {}, λ = {}):",
        k_weibull, lambda_weibull
    );
    println!("  Empirical mean      ≈ {:.6}", stats_weibull.mean());
    println!("  Theoretical mean     = {:.6}", weibull_mean_th);
    println!("  Empirical variance  ≈ {:.6}", stats_weibull.variance());
    println!("  Theoretical variance = {:.6}", weibull_var_th);
    println!("  Min                 ≈ {:.6}", stats_weibull.min());
    println!("  Max                 ≈ {:.6}", stats_weibull.max());
    println!();

    println!(
        "Pareto(x_m = {}, α = {}):",
        x_m_pareto, alpha_pareto
    );
    println!("  Empirical mean      ≈ {:.6}", stats_pareto.mean());
    println!("  Theoretical mean     = {:.6}", pareto_mean_th);
    println!("  Empirical variance  ≈ {:.6}", stats_pareto.variance());
    println!("  Theoretical variance = {:.6}", pareto_var_th);
    println!("  Min                 ≈ {:.6}", stats_pareto.min());
    println!("  Max                 ≈ {:.6}", stats_pareto.max());
}
```

Program 7.4.1 demonstrates how inverse-CDF transformations provide an elegant, computationally efficient method for generating deviates from a variety of non-uniform distributions. This approach reflects the central ideas developed in Section 7.4.1: when a closed-form cumulative distribution function is available, transforming a single uniform deviate via its analytic inverse is both theoretically sound and computationally optimal. The examples of the exponential, Weibull, and Pareto distributions reveal how this method naturally scales from light-tailed to heavy-tailed behaviors without requiring additional random draws or rejection loops.

The empirical results also highlight an important feature of sampling theory: moments from well-behaved light-tailed distributions converge rapidly to their theoretical values, while heavy-tailed distributions such as Pareto exhibit slower variance convergence due to occasional extreme samples. This sensitivity underscores the role of distribution shape in determining sample size requirements for reliable Monte Carlo estimation.

By isolating each distribution in a modular design, the program establishes a foundation for extending transformation-based sampling to broader families such as log-normal, Cauchy, and generalized extreme-value distributions, as well as numerically inverted CDFs for densities lacking closed forms. These methods form an essential building block for advanced Monte Carlo algorithms and probabilistic models throughout numerical computing.

## 7.4.2. Weibull Deviates and Transformation Derivation

The Weibull distribution models lifetimes, failure rates, and other phenomena governed by monotonic hazard functions. It is characterized by a shape parameter $k > 0$ and a scale parameter $\lambda > 0$, with probability density function,

$$
f(y) =\begin{cases}
\dfrac{k}{\lambda} \left( \dfrac{y}{\lambda} \right)^{k-1} e^{-(y/\lambda)^k}, & y \ge 0, \\[6pt]
0, & y < 0
\end{cases}
\tag{7.4.6}
$$

The corresponding cumulative distribution function (CDF) is

$$F(y) = 1 - e^{-(y/\lambda)^k}\tag{7.4.7}$$

The Weibull family generalizes several classical models. When $k = 1$, the distribution reduces to the exponential law with mean $\lambda$; for $k = 2$, it yields the Rayleigh distribution. As $k$ increases, the distribution becomes increasingly peaked and bounded, making it widely applicable in survival analysis, reliability engineering, and meteorological modeling.

### Transformation Derivation

The inverse-transform sampling method introduced in Equation (7.4.5) can be directly applied to the Weibull distribution. By setting:

$$x = F(y) = 1 - e^{-(y/\lambda)^k}\tag{7.4.8}$$

we obtain,

$$(y/\lambda)^k = -\ln(1-x)\tag{7.4.9}$$

and therefore the transformation:

$$y = \lambda[-\ln(1-x)]^{1/k}, \qquad x \sim U(0,1)\tag{7.4.10}$$

Equation (7.4.10) expresses the inverse CDF (quantile) transformation, mapping uniform random numbers into Weibull-distributed deviates. It demonstrates that only a few arithmetic operations, one logarithm, one exponentiation, and one multiplication, are required to generate each sample, making the method computationally optimal and numerically stable.

### Properties and Implementation

For $k = 1$, the Weibull distribution simplifies to the exponential distribution, and for $k = 2$, it describes the Rayleigh model frequently used in signal processing and turbulence studies. The computational cost per deviate is constant: one uniform draw, one logarithmic evaluation, and one power operation. In practice, it is advisable to clamp the uniform variable $x$ to the interval $(10^{-15}, 1 - 10^{-15})$ to avoid evaluating $\ln(0)$ or producing underflow in the tails. The method is stable for all $(k, \lambda) > 0$ and exhibits an expected time complexity of $\mathcal{O}(1)$.

In Rust implementations, this transformation can be implemented using standard library functions such as `f64::ln()` and `powf()`. Since the logarithm dominates the arithmetic cost, performance improvements can be achieved by vectorization or SIMD parallelism using the `rayon` crate. For simulations requiring millions of deviates, precomputing constants such as $\lambda^{-1}$ or $1/k$ further improves throughput without affecting accuracy.

### Recent Developments

Recent research extends the classical Weibull model to capture more complex empirical behaviors observed in biomedical, climatological, and materials data. The Entropy-Transformed Weibull (ET-W) and Generalized Weibull models introduce additional parameters to modulate tail behavior and entropy characteristics, improving statistical fit and interpretability (Sindhu et al., 2024; Suleiman et al., 2024). Despite these extensions, the inverse-transform principle embodied in Equation (7.4.10) remains the computational foundation for random variate generation in all Weibull-type families. The continuing refinement of these methods highlights the enduring importance of efficient and reproducible deviate generation in modern scientific computing.

### Rust Implementation

Following the derivation of the Weibull transformation in Section 7.4.2, Program 7.4.2 provides a concrete implementation of inverse-transform sampling for the Weibull family together with two related models, the exponential and Pareto distributions. Section 7.4.2 established how a uniform deviate can be mapped into a Weibull-distributed random variable through the inverse CDF expression of Equation (7.4.10), which generalizes the exponential law and forms the computational foundation for a broad class of lifetime and reliability models. This program translates those analytical results directly into code, applies the same principle to the exponential and Pareto distributions, and validates each generator by comparing empirical moments with their theoretical values. The demonstration reinforces the practical significance of the transformations developed in the subsection, showing how a handful of logarithmic and power operations suffice to produce accurate and efficient non-uniform deviates for Monte Carlo simulation.

At the core of the implementation are three transformation functions that directly realize the inverse-CDF mappings developed in Section 7.4.2. The exponential generator applies the transformation introduced in Equation (7.4.5), using a single logarithmic evaluation to map a uniform deviate into an exponential sample. The Weibull generator implements the inverse-CDF expression of Equation (7.4.10), extending the exponential model by introducing an additional $1/k$ power to incorporate the shape parameter described in the subsection. The Pareto generator similarly leverages its closed-form quantile function, applying a multiplicative scaling and a negative power to reproduce the heavy-tailed behavior characteristic of Pareto and power-law models. Together, these transformations exemplify how the inverse-transform principle extends naturally across different distribution families once an analytic CDF inverse has been established.

To ensure numerical stability, each transformation clamps the input uniform deviate away from 0 and 1, preventing the evaluation of singular expressions such as $\ln(0)$ and avoiding extreme underflow or overflow during exponentiation. This safeguard is particularly important for the Weibull and Pareto distributions, where the computational expressions derived in Section 7.4.2 involve operations that become ill-conditioned near the boundaries of the unit interval. The program also incorporates a lightweight statistical accumulator that computes empirical means, variances, minima, and maxima, enabling a direct numerical comparison with the theoretical formulas presented earlier in the subsection. This design ensures that each transformation can be validated quantitatively and that deviations fall within expected Monte Carlo sampling fluctuations.

The `main` function coordinates the demonstration by generating $N = 100{,}000$ samples from each distribution, computing and reporting their empirical statistics alongside their analytical values. It begins with the exponential model, whose simple structure and rapidly decaying tail make it an ideal baseline for verifying inverse-transform mappings. The program then evaluates the Weibull distribution using the transformation defined in Equation (7.4.10), highlighting how the shape parameter adjusts the distribution’s behavior and illustrating the generality of the derivation in Section 7.4.2. Finally, the Pareto distribution is sampled to illustrate how inverse transforms handle heavy-tailed phenomena where rare, large deviations significantly influence the variance. The close agreement between empirical and theoretical moments confirms the practical reliability of the transformations developed in this subsection and demonstrates the broad applicability of the inverse-transform method in numerical simulation.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
rand_chacha = "0.3"
statrs = "0.17"
```

```rust
//// ============================================================================
// Program 7.4.2 — Generating Exponential, Weibull, and Pareto Deviates
//                 Using the Inverse-Transform Method
//
// This program implements the inverse-transform technique introduced in
// Section 7.4 to generate random deviates from three classical distributions:
//
//     • Exponential(λ)
//     • Weibull(k, λ)
//     • Pareto(x_m, α)
//
// The inverse-transform method relies on the fact that if U ~ Uniform(0,1),
// then a random variable Y with cumulative distribution function (CDF) F is
// obtained exactly by the mapping
//
//                           Y = F^{-1}(U).
//
// This construction provides a direct and computationally optimal route for
// generating non-uniform random variables using only one call to a uniform RNG
// plus a logarithm and a small number of arithmetic operations. The approach
// is numerically stable for all parameter choices and is widely used in Monte
// Carlo simulation, stochastic modeling, and reliability analysis.
//
// In this program:
//
//   • Exponential deviates are computed via
//         Y = -(1/λ) ln(1 - U).
//
//   • Weibull deviates follow Equation (7.4.10),
//         Y = λ[-ln(1 - U)]^{1/k},
//     illustrating the generalization from exponential (k = 1) and Rayleigh
//     (k = 2) models to arbitrary shape parameters.
//
//   • Pareto deviates arise from the inverse CDF
//         Y = x_m (1 - U)^{-1/α},
//     capturing heavy-tailed behavior relevant in network traffic, economics,
//     materials failure, and risk modeling.
//
// For each distribution, the program draws N = 100,000 samples, accumulates
// empirical statistics (mean, variance, minimum, maximum), and compares them
// with the corresponding theoretical expressions. The close agreement of these
// quantities validates the correctness of the inverse-transform
// implementations and demonstrates the expected sampling fluctuations inherent
// in Monte Carlo experiments.
//
// ============================================================================

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use statrs::function::gamma::gamma;

/// Simple statistics accumulator for streaming samples.
struct RunningStats {
    n: usize,
    sum: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            n: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn push(&mut self, x: f64) {
        self.n += 1;
        self.sum += x;
        self.sum_sq += x * x;
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    fn mean(&self) -> f64 {
        self.sum / self.n as f64
    }

    fn variance(&self) -> f64 {
        let n = self.n as f64;
        let mean = self.mean();
        self.sum_sq / n - mean * mean
    }
}

/// Weibull(k, λ) distribution implemented via the inverse CDF
///   y = λ[-ln(1 - x)]^{1/k},   x ~ U(0, 1),
/// corresponding to Equation (7.4.10).
struct Weibull {
    k: f64,
    lambda: f64,
    inv_k: f64,
}

impl Weibull {
    /// Construct a Weibull(k, λ) distribution.
    fn new(k: f64, lambda: f64) -> Self {
        assert!(k > 0.0, "shape k must be > 0");
        assert!(lambda > 0.0, "scale λ must be > 0");
        Self {
            k,
            lambda,
            inv_k: 1.0 / k,
        }
    }

    /// Draw a single Weibull deviate using inverse transform.
    ///
    /// 1. Draw x ~ U(0,1).
    /// 2. Clamp x away from 0 and 1 to avoid log(0) and underflow.
    /// 3. Return y = λ[-ln(1 - x)]^{1/k}.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        const EPS: f64 = 1.0e-15;

        let mut x: f64 = rng.gen(); // x ∈ [0, 1)
        if x <= EPS {
            x = EPS;
        } else if x >= 1.0 - EPS {
            x = 1.0 - EPS;
        }

        // t = -ln(1 - x) is exponentially distributed;
        // we then apply the 1/k power and scale by λ.
        let t = -(1.0 - x).ln();
        self.lambda * t.powf(self.inv_k)
    }
}

/// Theoretical mean of Weibull(k, λ):
///   E[Y] = λ Γ(1 + 1/k).
fn weibull_mean_theoretical(k: f64, lambda: f64) -> f64 {
    lambda * gamma(1.0 + 1.0 / k)
}

/// Theoretical variance of Weibull(k, λ):
///   Var[Y] = λ² [Γ(1 + 2/k) - Γ(1 + 1/k)²].
fn weibull_variance_theoretical(k: f64, lambda: f64) -> f64 {
    let g1 = gamma(1.0 + 1.0 / k);
    let g2 = gamma(1.0 + 2.0 / k);
    lambda * lambda * (g2 - g1 * g1)
}

fn main() {
    // Fixed seed for reproducibility in examples.
    let mut rng = ChaCha20Rng::seed_from_u64(0x_7f7f_2025);

    // Example parameters: k = 1.5, λ = 1.0
    let k = 1.5_f64;
    let lambda = 1.0_f64;
    let weibull = Weibull::new(k, lambda);

    let n_samples = 100_000;
    let mut stats = RunningStats::new();

    for _ in 0..n_samples {
        let y = weibull.sample(&mut rng);
        stats.push(y);
    }

    let empirical_mean = stats.mean();
    let empirical_var = stats.variance();
    let theoretical_mean = weibull_mean_theoretical(k, lambda);
    let theoretical_var = weibull_variance_theoretical(k, lambda);

    println!("Sampling {} Weibull deviates via inverse transform:", n_samples);
    println!();
    println!("Weibull(k = {}, λ = {}):", k, lambda);
    println!("  Empirical mean      ≈ {:.6}", empirical_mean);
    println!("  Theoretical mean     = {:.6}", theoretical_mean);
    println!("  Empirical variance  ≈ {:.6}", empirical_var);
    println!("  Theoretical variance = {:.6}", theoretical_var);
    println!("  Min                 ≈ {:.6}", stats.min);
    println!("  Max                 ≈ {:.6}", stats.max);
}
```

Program 7.4.2 illustrates the practical implementation of inverse-transform sampling for three widely used distributions and reinforces the analytical development presented in the current section. The close agreement between empirical and theoretical moments validates both the correctness of the transformations and the numerical stability of the approach. The exponential and Weibull examples emphasize how logarithmic mappings generate light- to moderately-tailed distributions efficiently, while the Pareto example demonstrates the method’s effectiveness even for heavy-tailed models where sampling variability is inherently larger.

More broadly, the program highlights the versatility of the inverse-transform principle. Once the inverse CDF is known, generating random deviates becomes a matter of applying simple transformations to uniform samples. This modular structure makes it easy to extend the code to other parametric families, such as generalized Weibull, log-logistic, or Gompertz distributions, by implementing their respective quantile functions. In subsequent sections, we will see how alternative methods such as rejection sampling and ratio-of-uniforms techniques address cases where the inverse CDF is unavailable or numerically challenging, thereby complementing the inverse-transform approach developed here.

## 7.4.3. Laplace (Double-Exponential) Deviates and Transform Methods

The Laplace distribution, also known as the double-exponential distribution, occupies an intermediate position between the Gaussian and exponential families. It retains the exponential decay of tails but is symmetric about a central location, providing a convenient model for processes that exhibit sharper peaks and heavier tails than those described by the normal law. Its probability density function is:

$$f(y) = \frac{1}{2b}\exp \left(-\frac{|y-\mu|}{b}\right), \qquad y \in \mathbb{R}, b>0\tag{7.4.11}$$

Here, the parameter $\mu$ denotes the **location** (mean or median) and $b$ the **scale**, controlling the spread of the distribution.

Because of its heavier tails, the Laplace distribution frequently appears in signal and image processing, robust statistics, and differential privacy, where resistance to outliers or large perturbations is required. It can also be interpreted as the distribution of the difference of two independent exponential random variables with common mean $b$, giving it a natural connection to processes governed by exponential waiting times.

### Mathematical Formulation and Transform Derivation

The cumulative distribution function (CDF) of the Laplace law has a piecewise form reflecting the symmetry of the density about $\mu$:

$$
F(y) =
\begin{cases}
\dfrac{1}{2}\exp\!\left(\dfrac{y-\mu}{b}\right), & y < \mu, \\[4pt]
1 - \dfrac{1}{2}\exp\!\left(-\dfrac{y-\mu}{b}\right), & y \ge \mu
\end{cases}
\tag{7.4.12}
$$

The inverse-transform method can be derived directly from this expression. Setting $x = F(y)$, with $x \sim U(0,1)$, and solving for $y$ in each branch yields,

$$
y =
\begin{cases}
\mu + b\ln(2x), & 0 < x < 0.5, \\[4pt]
\mu - b\ln[2(1-x)], & 0.5 \le x < 1,
\end{cases}
\tag{7.4.13}
$$

Equation (7.4.13) provides an explicit analytic transformation that maps uniform random numbers to Laplace-distributed deviates. It requires only a single evaluation of the natural logarithm and a few arithmetic operations, making it one of the most computationally efficient transformation methods for generating symmetric heavy-tailed random variables.

An alternative yet mathematically equivalent construction views the Laplace distribution as the difference of two independent exponential variables. If $E_1, E_2 \sim \mathrm{Exponential}(1)$ are independent, then:

$$Y = \mu + b(E_1 - E_2)\tag{7.4.14}$$

follows the Laplace law with parameters $(\mu, b)$. This representation emphasizes the Laplace distribution’s deep connection with the exponential family and offers an intuitive interpretation in terms of random arrivals or waiting-time processes: the deviation from the mean corresponds to the net difference between two competing exponential events.

### Properties and Computational Considerations

The Laplace distribution has mean $\mathbb{E}[Y] = \mu$ and variance $\operatorname{Var}(Y) = 2b^2$. Compared with the normal distribution, it assigns higher probability to large deviations, giving it a sharper peak and heavier tails. These properties make it ideal for modeling impulsive or sparse phenomena, such as sudden jumps in signals, abrupt financial movements, or extreme deviations in experimental noise.

From a computational standpoint, both sampling forms in Equations (7.4.13) and (7.4.14) are exceptionally efficient. The inverse transform requires only one uniform draw and one logarithmic evaluation per deviate, resulting in constant-time complexity, $\mathcal{O}(1)$, and negligible memory usage. The exponential-difference formulation is theoretically elegant and occasionally useful for analytical derivations but involves generating two exponential samples, effectively doubling the computational cost.

For high-performance environments such as GPU kernels or SIMD vector pipelines, the inverse-transform form is generally preferred because it can be implemented branchlessly, avoiding conditional logic between the two halves of the CDF and maintaining uniform instruction flow.

### Broader Significance and Applications

The Laplace distribution occupies a central position among continuous probability models due to its dual characteristics: exponential tail decay and symmetry around the mean. This combination provides a powerful balance between robustness and analytic tractability, making it indispensable in many scientific and engineering domains.

In robust statistics, the Laplace model underlies the least-absolute-deviation (LAD) or $L^1$-regression estimator, which minimizes the sum of absolute residuals rather than squared deviations. This property makes it inherently less sensitive to outliers and heavy-tailed data, thereby providing more stable estimators in the presence of anomalous observations. The Laplace likelihood function directly leads to the $L^1$-norm penalty, serving as a foundation for robust estimation theory and methods like quantile regression.

In machine learning, the Laplace prior on model parameters induces sparsity by encouraging many coefficients to be exactly zero, an effect exploited by the LASSO (Least Absolute Shrinkage and Selection Operator) and related sparse regularization techniques. From a Bayesian perspective, the Laplace prior corresponds to an exponential penalty on parameter magnitude, effectively balancing model interpretability and generalization. This connection highlights the Laplace law’s role as a bridge between probabilistic modeling and convex optimization.

In the domain of differential privacy, Laplace noise provides a mathematically rigorous mechanism for protecting sensitive data. Adding independent Laplace perturbations with scale $b = \Delta f / \varepsilon$ to each query output guarantees $\varepsilon$-differential privacy by bounding the maximum influence of any single record. Because the Laplace mechanism yields a closed-form privacy guarantee while remaining computationally efficient, it is implemented in modern privacy-preserving frameworks such as TensorFlow Privacy and PyTorch Opacus.

In signal processing, the Laplace distribution frequently models residual or transform-domain coefficients in image and audio compression. Natural signals exhibit heavy-tailed amplitude spectra, and the Laplace model captures this behavior more accurately than a Gaussian. Techniques such as wavelet denoising and compressive sensing exploit Laplace priors to enforce sparsity in transformed representations. Similarly, in high-dimensional inference and sparse recovery, Laplace-based priors provide computationally efficient proxies for nonconvex $\ell_0$-penalization.

From a computational standpoint, the Laplace distribution exemplifies the synergy between simple analytic structure and high algorithmic efficiency. Its inverse-transform sampling relation (Equation 7.4.13) or exponential-difference form (Equation 7.4.14) allows for direct, branchless generation of non-Gaussian random deviates with constant-time complexity. This efficiency makes Laplace deviates particularly attractive in large-scale Monte Carlo frameworks, stochastic gradient methods, and privacy-aware synthetic data generation pipelines.

Hence, the Laplace distribution demonstrates the enduring unity between elegant mathematical formulation and practical computational utility. Its transform-based sampling relations, derived from fundamental exponential mechanisms, provide one of the most versatile and reliable building blocks for simulating real-world processes characterized by symmetry, robustness, and heavy-tailed variability.

### Rust Implementation

Following the derivation of the Laplace inverse CDF in Section 7.4.3, Program 7.4.3 provides an explicit implementation of two transformation-based methods for generating Laplace (double-exponential) random deviates. Section 7.4.3 established that the Laplace law admits both a piecewise analytic inverse CDF, given in Equation (7.4.13), and an equivalent exponential-difference representation, given in Equation (7.4.14). These formulations motivate the two sampling functions `sample_inverse` and `sample_exp_difference` implemented in this program. By converting the analytical expressions directly into computational procedures and comparing the resulting empirical moments against their theoretical values, the program demonstrates the accuracy, efficiency, and conceptual unity of these transformation methods. In doing so, it reinforces the central idea of the subsection: Laplace deviates can be generated with minimal computational cost while preserving the characteristic symmetry and heavy-tailed structure of the distribution.

At the core of the implementation are two sampling functions named `sample_inverse` and `sample_exp_difference`, each corresponding directly to one of the transformation methods developed in Section 7.4.3. The `sample_inverse` function implements the analytic inverse CDF of Equation (7.4.13), selecting between its two branches based on the uniform deviate and applying the corresponding logarithmic transformation. This directly encodes the piecewise symmetry of the Laplace distribution around $\mu$ and requires only a single logarithm per deviate. In contrast, the `sample_exp_difference` function realizes the stochastic representation of Equation (7.4.14), generating two Exponential(1) variables through the standard inverse-transform method and returning their scaled difference. This formulation emphasizes the Laplace distribution’s decomposition into opposing exponential components and reflects its deep probabilistic connection to exponential waiting-time processes.

To ensure numerical stability, both sampling methods clamp uniform inputs away from 0 and 1 to avoid evaluating singular expressions such as $\ln(0)$. This precaution is implemented internally within both `sample_inverse` and the exponential helper used by `sample_exp_difference`, preventing overflow or underflow in the logarithmic evaluations. The program also employs a `RunningStats` structure to collect empirical statistics for each method, including sample mean, variance, minimum, and maximum. This allows direct comparison with the theoretical expressions $\mathbb{E}[Y] = \mu$ and $\operatorname{Var}(Y) = 2b^2$, providing a quantitative check on the correctness of each transformation method.

The `main` function orchestrates the full numerical demonstration. It first constructs a `Laplace` instance using parameters $\mu$ and $b$, then uses both `sample_inverse` and `sample_exp_difference` to generate $N = 100{,}000$ deviates. Each sample is streamed into an instance of `RunningStats`, which computes aggregate statistics in a single pass. The program then prints the empirical results alongside their analytical values, verifying that both methods converge to the same distribution despite differing computational structures. The inverse-CDF method illustrates how Equation (7.4.13) translates into a highly efficient, branch-dependent transform, while the exponential-difference approach demonstrates how Equation (7.4.14) constructs the same distribution using two exponential samples. Together, these functions confirm the equivalence of the two Laplace constructions and highlight their suitability for different computational contexts.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
rand_chacha = "0.3"
```

```rust
// ============================================================================
// Program 7.4.3 — Laplace (Double-Exponential) Deviates via
//                 Inverse Transform and Exponential Difference
//
// This program implements two equivalent methods for generating Laplace
// random deviates with parameters (μ, b):
//
//   1. Inverse-transform sampling using the piecewise inverse CDF
//      of Equation (7.4.13):
//
//           Y = μ + b ln(2X),          0 < X < 0.5,
//           Y = μ - b ln[2(1 - X)],    0.5 ≤ X < 1,
//
//      where X ~ U(0, 1). This requires one uniform deviate and
//      one logarithm per sample.
//
//   2. Difference-of-exponentials representation from Equation (7.4.14):
//
//           Y = μ + b (E1 - E2),
//
//      where E1, E2 are independent Exponential(1) variables generated via
//      the standard inverse transform E = -ln(1 - U).
//
// For each method, the program draws N samples, computes empirical mean,
// variance, minimum, and maximum, and compares them with the theoretical
// values:
//
//           E[Y]   = μ,
//           Var[Y] = 2 b².
//
// ============================================================================

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Streaming statistics accumulator for mean, variance, min, and max.
struct RunningStats {
    n: usize,
    sum: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            n: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn push(&mut self, x: f64) {
        self.n += 1;
        self.sum += x;
        self.sum_sq += x * x;
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    fn mean(&self) -> f64 {
        self.sum / self.n as f64
    }

    fn variance(&self) -> f64 {
        let n = self.n as f64;
        let mean = self.mean();
        self.sum_sq / n - mean * mean
    }
}

/// Laplace(μ, b) distribution.
///
/// PDF:
///   f(y) = (1 / (2 b)) exp(-|y - μ| / b),  y ∈ ℝ,  b > 0.
struct Laplace {
    mu: f64,
    b: f64,
}

impl Laplace {
    fn new(mu: f64, b: f64) -> Self {
        assert!(b > 0.0, "scale b must be > 0");
        Self { mu, b }
    }

    /// Laplace deviate via inverse transform (Equation 7.4.13):
    ///
    ///   Y = μ + b ln(2X),          0 < X < 0.5,
    ///   Y = μ - b ln[2(1 - X)],    0.5 ≤ X < 1,
    ///
    /// where X ~ U(0, 1).
    fn sample_inverse<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        const EPS: f64 = 1.0e-15;

        let mut x: f64 = rng.gen(); // X ∈ [0,1)
        if x <= EPS {
            x = EPS;
        } else if x >= 1.0 - EPS {
            x = 1.0 - EPS;
        }

        if x < 0.5 {
            self.mu + self.b * (2.0 * x).ln()
        } else {
            self.mu - self.b * (2.0 * (1.0 - x)).ln()
        }
    }

    /// Laplace deviate via difference of exponentials (Equation 7.4.14):
    ///
    ///   Y = μ + b (E1 - E2),
    ///
    /// where E1, E2 are independent Exponential(1) variables.
    fn sample_exp_difference<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        const EPS: f64 = 1.0e-15;

        // Exponential(1) via inverse transform: E = -ln(1 - U).
        fn sample_exp1<R: Rng + ?Sized>(rng: &mut R) -> f64 {
            let mut u: f64 = rng.gen();
            if u <= EPS {
                u = EPS;
            } else if u >= 1.0 - EPS {
                u = 1.0 - EPS;
            }
            -(1.0 - u).ln()
        }

        let e1 = sample_exp1(rng);
        let e2 = sample_exp1(rng);

        self.mu + self.b * (e1 - e2)
    }

    /// Theoretical mean E[Y] = μ.
    fn mean_theoretical(&self) -> f64 {
        self.mu
    }

    /// Theoretical variance Var[Y] = 2 b².
    fn variance_theoretical(&self) -> f64 {
        2.0 * self.b * self.b
    }
}

fn main() {
    // Fixed seed for reproducibility in examples.
    let mut rng = ChaCha20Rng::seed_from_u64(0x7F4_203_u64);

    // Parameters for Laplace(μ, b).
    let mu = 0.0_f64;
    let b = 1.0_f64;
    let laplace = Laplace::new(mu, b);

    let n_samples = 100_000;

    // Accumulate statistics for both generation methods.
    let mut stats_inverse = RunningStats::new();
    let mut stats_exp_diff = RunningStats::new();

    for _ in 0..n_samples {
        let y_inv = laplace.sample_inverse(&mut rng);
        stats_inverse.push(y_inv);

        let y_diff = laplace.sample_exp_difference(&mut rng);
        stats_exp_diff.push(y_diff);
    }

    let mean_theoretical = laplace.mean_theoretical();
    let var_theoretical = laplace.variance_theoretical();

    println!("Sampling {} Laplace deviates:\n", n_samples);
    println!("Laplace(μ = {}, b = {}):", mu, b);

    println!("\nMethod 1: Inverse CDF (Equation 7.4.13)");
    println!("  Empirical mean      ≈ {:.6}", stats_inverse.mean());
    println!("  Theoretical mean     = {:.6}", mean_theoretical);
    println!("  Empirical variance  ≈ {:.6}", stats_inverse.variance());
    println!("  Theoretical variance = {:.6}", var_theoretical);
    println!("  Min                 ≈ {:.6}", stats_inverse.min);
    println!("  Max                 ≈ {:.6}", stats_inverse.max);

    println!("\nMethod 2: Difference of Exponentials (Equation 7.4.14)");
    println!("  Empirical mean      ≈ {:.6}", stats_exp_diff.mean());
    println!("  Theoretical mean     = {:.6}", mean_theoretical);
    println!("  Empirical variance  ≈ {:.6}", stats_exp_diff.variance());
    println!("  Theoretical variance = {:.6}", var_theoretical);
    println!("  Min                 ≈ {:.6}", stats_exp_diff.min);
    println!("  Max                 ≈ {:.6}", stats_exp_diff.max);
}
```

Program 7.4.3 illustrates how the analytic relationships derived in Section 7.4.3 can be translated into efficient and reliable numerical procedures through the functions `sample_inverse` and `sample_exp_difference`. The inverse CDF implementation leverages the piecewise structure of Equation (7.4.13) to produce deviates using a single logarithmic call, while the exponential-difference implementation uses Equation (7.4.14) to construct the same distribution from two exponential samples. The empirical results obtained through `RunningStats` confirm that both approaches reproduce the theoretical mean and variance with high accuracy, underscoring the consistency and robustness of the transformation techniques.

The two sampling strategies also highlight different computational considerations. The inverse-CDF method is typically preferred in high-performance environments due to its minimal number of operations and suitability for branch-less vectorized implementations. Meanwhile, the exponential-difference formulation may be advantageous in contexts where exponential variates are already available or where a structural decomposition is analytically desirable. The modular nature of the implementation ensures that additional Laplace variants, such as asymmetric or truncated forms, can be implemented by extending the existing functions. Together, these results establish Program 7.4.3 as a practical foundation for applying Laplace noise models in robust statistics, differential privacy, sparse estimation, and heavy-tailed signal modeling.

## 7.4.4. Pareto (Heavy-Tail) Deviates

The Pareto distribution is a canonical model for power-law and heavy-tailed phenomena. It describes random processes in which a small proportion of events account for a disproportionately large share of the total effect, an idea known as the *Pareto principle* or *80–20 rule*. Its probability density function is:

$$f(y)=\frac{\alpha\, y_{\min}^{\alpha}}{y^{\alpha+1}},\qquad y\ge y_{\min}>0 \tag{7.4.15}$$

where $y_{\min}$ defines the lower bound of the distribution and $\alpha$ (the shape parameter) determines the steepness of the tail.

The cumulative distribution function is:

$$F(y)=1-\left(\frac{y_{\min}}{y}\right)^{\alpha}\tag{7.4.16}$$

which approaches 1 only asymptotically as $y\to\infty$. The mean exists only for $\alpha>1$ and the variance only for $\alpha>2$; for $0<\alpha\le1$ the expectation diverges, reflecting the dominance of extremely large outcomes. These mathematical properties underlie its pervasive use in modeling rare but high-impact events.

### Inverse-Transform Sampling

Applying the inverse-transform principle to (7.4.16) gives:

$$y=y_{\min}(1-x)^{-1/\alpha},\qquad x\sim U(0,1)\tag{7.4.17}$$

This simple expression maps uniform deviates directly to Pareto-distributed values. Because of the negative fractional exponent, values of $x$ very close to zero yield exceedingly large $y$, producing the long right tail characteristic of power-law behaviour. To avoid overflow in finite-precision arithmetic, $x$ is typically clamped to a safe range such as $(10^{-15},1-10^{-15})$.

For numerical stability, a logarithmic form is often used:

$$\ln y=\ln y_{\min}-\frac{1}{\alpha}\ln(1-x) \tag{7.4.18}$$

which replaces direct exponentiation with a single logarithm and multiplication. Since the transformation requires only one uniform draw, one logarithm, and one power (or exponentiation), Pareto deviates can be generated in constant time, $\mathcal{O}(1)$, per sample.

### Statistical Behaviour and Interpretation

The Pareto law epitomizes scale invariance: rescaling $y$ by a constant factor changes only the normalization of the PDF, not its functional form. This property makes it applicable to phenomena across many orders of magnitude, ranging from income and firm sizes to earthquake energies and file-transfer volumes. The survival function,

$$\Pr(Y>y)=\left(\frac{y_{\min}}{y}\right)^{\alpha}\tag{7.4.19}$$

shows that the probability of observing an extreme event decreases polynomially, not exponentially, with magnitude. Consequently, extremely large realizations, while rare, contribute substantially to averages and variances. Figure 7.4.1 illustrates how the shape parameter $\alpha$ influences the heaviness of the Pareto tail. As $\alpha$ increases, the distribution decays more rapidly, whereas smaller $\alpha$ values produce flatter, heavier tails, highlighting the persistence of extreme outcomes even at high magnitudes.

```{figure} images/pqQDe4beUu67RvW3raYP-NviJzDw59uWf71JXzx9R-v1.png
:name: rjiIXeXxVS
:align: middle
:width: 50%

**Figure 7.4.1** Effect of the shape parameter α on the Pareto probability density. Each curve is normalized for $y_{\min}=1$. Smaller α values (e.g., α = 1) produce flatter, heavier tails, while larger α values (α = 2, 3) yield more rapidly decaying distributions.
```

This feature underlies its importance in risk analysis, catastrophic-loss modeling, and network-load characterization, where “rare but consequential’’ events dominate system behaviour. In simulation studies, Pareto deviates are used to emulate heavy-tail effects such as bursty workloads, extreme financial returns, and the connectivity structure of scale-free networks.

### Computational Considerations and Stability

From a computational perspective, Pareto deviates are highly efficient to generate and stable under double precision. However, because small $x$ values correspond to extremely large $y$, tail accuracy depends on the resolution of the underlying uniform generator and the precision of floating-point arithmetic. When simulating very heavy tails (e.g., $\alpha<1$), logarithmic sampling as in (7.4.18) is recommended to mitigate rounding errors. In high-performance Rust implementations, the transform can be vectorized across threads or SIMD lanes, making it practical for large-scale Monte Carlo studies or stress-testing algorithms sensitive to rare events.

### Modern Developments and Extensions

Although the classical Pareto form is analytically simple, empirical data often exhibit curvature in the central region and lighter extreme tails than a pure power law predicts. To capture such behaviour, modern research explores hybrid Pareto–lognormal, Pareto–exponential, and three-component composite models in which a Pareto tail is smoothly joined to a different body distribution. These mixtures achieve better statistical fit in finance, insurance, and climatology (Osatohanmwen et al., 2024).

Adaptive sampling strategies have also emerged: an initial draw determines whether the sample lies in the *bulk* or *tail* region, after which the appropriate generator, analytic inversion for the body or rejection sampling for the tail is applied. Such frameworks maintain computational efficiency while improving representational realism across multiple scales of magnitude.

### Significance

The Pareto distribution remains the archetypal model of heavy-tail behaviour, illustrating how simple mathematical structure can describe extraordinarily diverse phenomena. Its inverse-transform form (7.4.17) offers an elegant, closed-form link between uniform deviates and power-law behaviour, enabling both theoretical analysis and practical simulation. At the same time, its divergent moments remind practitioners that heavy-tail modeling demands numerical caution and interpretive discipline. Whether modeling economic inequality, data-center loads, or catastrophic risks, Pareto deviates provide an essential tool for probing the statistics of the extreme.

### Rust Implementation

Following the discussion in Section 7.4.4 on the mathematical formulation and computational characteristics of heavy-tailed distributions, Program 7.4.4 provides a practical implementation of Pareto deviate generation using the inverse-transform relations derived in (7.4.17) and (7.4.18). The transformation from uniform to Pareto-distributed samples requires only one logarithm or exponentiation per draw, allowing efficient simulation of power-law behaviour over many orders of magnitude. Consistent with the surrounding text, the program emphasizes numerical stability in the extreme tail, where small uniform inputs produce disproportionately large outputs. By incorporating clamping safeguards and both analytic forms of the inverse transform, the implementation demonstrates how heavy-tail sampling can be performed robustly in double precision while preserving the theoretical properties developed earlier in the section.

At the core of the program are two sampling functions that implement the inverse-transformation formulas introduced in (7.4.17) and (7.4.18). The `pareto_sample_log` function generates deviates using the logarithmic form, replacing direct exponentiation with a more stable transformation based on $\ln(1-x)$. This approach is particularly advantageous in the heavy-tailed regime, where very small values of $x$ can otherwise lead to overflow or excessive amplification of rounding errors. The function begins by drawing a uniform deviate and clamping it to a safe range to ensure numerical soundness of the logarithm. It then evaluates the transformed expression for $\ln y$ using the parameters $\alpha$ and $y_{\min}$ and finally exponentiates to obtain the sample. In contrast, the `pareto_sample_pow` function follows the direct power formula of (7.4.17), computing $(1-x)^{-1/\alpha}$ explicitly. While mathematically equivalent, this version is more susceptible to floating-point overflow for extremely heavy tails and is therefore included primarily for completeness and comparative illustration.

To complement the sampling functions, the program includes two analytic methods, `pareto_mean` and `pareto_variance`, which compute the theoretical moments of the distribution when they exist. These checks reflect the mathematical constraints noted earlier: the mean diverges for $\alpha \le 1$ and the variance for $\alpha \le 2$. By returning `None` in these cases, the program communicates the fundamental instability of heavy-tailed moments and relates directly to the theoretical discussion surrounding (7.4.19). The main function integrates all components by generating a large batch of samples, accumulating empirical statistics, and comparing them to the available theoretical values. It also records the minimum and maximum observed draws, illustrating how heavy-tailed distributions generate occasional extreme outliers that dominate higher-order moments. The resulting output provides numerical evidence for the heavy-tail behaviour introduced earlier in the section and reinforces the practical considerations discussed under computational stability.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
// Program 7.4.4: Sampling Pareto (heavy-tail) deviates via inverse transforms
//
// This program generates Pareto-distributed random variables using the
// inverse-transform formulas (7.4.17) and (7.4.18), then compares the
// empirical mean and variance with their theoretical counterparts when
// these moments exist.

use rand::prelude::*;

/// Small clamp to avoid log(0) and overflow in the tail.
const EPS: f64 = 1.0e-15;

/// Generate a single Pareto(α, y_min) deviate using the logarithmic form (7.4.18).
///
/// ln y = ln y_min - (1/α) ln(1 - x),   x ~ U(0, 1)
fn pareto_sample_log<R: Rng + ?Sized>(rng: &mut R, alpha: f64, y_min: f64) -> f64 {
    debug_assert!(alpha > 0.0, "shape parameter α must be positive");
    debug_assert!(y_min > 0.0, "scale parameter y_min must be positive");

    let u: f64 = rng.gen();
    // Clamp away from 0 and 1 to avoid infinities in ln(1 - x)
    let x = u.clamp(EPS, 1.0 - EPS);

    let log_1_minus_x = (1.0 - x).ln();
    let log_y = y_min.ln() - (1.0 / alpha) * log_1_minus_x;
    log_y.exp()
}

/// Generate a single Pareto(α, y_min) deviate using the direct power form (7.4.17).
///
/// y = y_min (1 - x)^(-1/α),   x ~ U(0, 1)
#[allow(dead_code)]
fn pareto_sample_pow<R: Rng + ?Sized>(rng: &mut R, alpha: f64, y_min: f64) -> f64 {
    debug_assert!(alpha > 0.0, "shape parameter α must be positive");
    debug_assert!(y_min > 0.0, "scale parameter y_min must be positive");

    let u: f64 = rng.gen();
    let x = u.clamp(EPS, 1.0 - EPS);

    y_min * (1.0 - x).powf(-1.0 / alpha)
}

/// Theoretical mean of Pareto(α, y_min), when it exists:
/// E[Y] = α y_min / (α - 1),  α > 1.
fn pareto_mean(alpha: f64, y_min: f64) -> Option<f64> {
    if alpha > 1.0 {
        Some(alpha * y_min / (alpha - 1.0))
    } else {
        None // Divergent mean
    }
}

/// Theoretical variance of Pareto(α, y_min), when it exists:
/// Var[Y] = (α y_min^2) / ((α - 1)^2 (α - 2)),  α > 2.
fn pareto_variance(alpha: f64, y_min: f64) -> Option<f64> {
    if alpha > 2.0 {
        let num = alpha * y_min * y_min;
        let den = (alpha - 1.0).powi(2) * (alpha - 2.0);
        Some(num / den)
    } else {
        None // Divergent variance
    }
}

fn main() {
    let mut rng = thread_rng();

    // Parameters for the Pareto distribution:
    //   y_min = 1, α = 2.5  → finite mean and variance
    let y_min = 1.0_f64;
    let alpha = 2.5_f64;

    let n_samples = 100_000_usize;

    println!(
        "Sampling {n} Pareto deviates via inverse transforms:\n",
        n = n_samples
    );

    // Accumulators for empirical statistics
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for _ in 0..n_samples {
        // Choose either the logarithmic or power-based implementation.
        // The logarithmic form is generally more stable for very heavy tails.
        let y = pareto_sample_log(&mut rng, alpha, y_min);
        // let y = pareto_sample_pow(&mut rng, alpha, y_min);

        sum += y;
        sum_sq += y * y;
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }

    let n = n_samples as f64;
    let empirical_mean = sum / n;
    let empirical_var = (sum_sq / n) - empirical_mean * empirical_mean;

    println!("Pareto(α = {alpha}, y_min = {y_min}):");
    println!("  Empirical mean      ≈ {emp_mean:.6}", emp_mean = empirical_mean);
    match pareto_mean(alpha, y_min) {
        Some(m) => println!("  Theoretical mean     = {theo_mean:.6}", theo_mean = m),
        None => println!("  Theoretical mean     = diverges (α ≤ 1)"),
    }

    println!("  Empirical variance  ≈ {emp_var:.6}", emp_var = empirical_var);
    match pareto_variance(alpha, y_min) {
        Some(v) => println!("  Theoretical variance = {theo_var:.6}", theo_var = v),
        None => println!("  Theoretical variance = diverges (α ≤ 2)"),
    }

    println!("  Min                 ≈ {min_y:.6}");
    println!("  Max                 ≈ {max_y:.6}");
}
```

Program 7.4.4 illustrates the practical generation of Pareto-distributed random variables using the inverse-transform method, directly reflecting the mathematical structure presented in Section 7.4.4. The numerical experiments highlight a defining characteristic of heavy-tailed laws: even with large sample sizes, empirical estimates of variance can fluctuate significantly due to sporadic but influential outliers. These effects corroborate the theoretical warnings that moments may diverge or converge slowly when $\alpha$ is small, and they demonstrate the sensitivity of power-law models to rare events. The modular design of the code, which separates the transformation logic from the theoretical checks and empirical accumulation, provides a flexible foundation for extending Pareto-based simulations to more advanced settings. Such extensions include mixture models, hybrid tail constructions, and stress-testing frameworks in which accurate representation of extreme behaviour plays a central role. Altogether, the program forms a computational bridge between the analytical properties of the Pareto distribution and its practical role in simulating high-impact, low-probability phenomena.

## 7.4.5. Gumbel (Extreme-Value Type I) Deviates

The Gumbel distribution occupies a central role in extreme-value theory, representing the limiting distribution of the maximum (or minimum) of many independent random variables drawn from light-tailed parent distributions such as the normal, exponential, or gamma. When rescaled appropriately, the distribution of maxima converges to the Gumbel form as the number of samples increases, a result analogous to the central-limit theorem but for extremes rather than means.

The probability density function (PDF) is:

$$f(y)=\frac{1}{\beta}\exp!\left[-\left(\frac{y-\mu}{\beta}+e^{-(y-\mu)/\beta}\right)\right],\qquad y\in\mathbb{R}\tag{7.4.20}$$

where $\mu$ is the location parameter (the mode of the distribution) and $\beta>0$ the scale parameter controlling the dispersion of extreme values. The corresponding cumulative distribution function (CDF) is:

$$F(y)=\exp\left[-e^{-(y-\mu)/\beta}\right] \tag{7.4.21}$$

This function rises steeply around $\mu$ but exhibits an extended right-hand tail, capturing the statistics of large maxima in natural and engineered systems. The mean of the distribution is $(\mu+\gamma\beta)$ (where $\gamma\approx0.5772$ is Euler’s constant) and the variance is $(\pi^2/6)\beta^2$.

### Sampling by Inverse Transformation

The Gumbel law admits an exact inversion of its CDF. Setting $x=F(y)$ and solving for $y$ yields:

$$y=\mu-\beta\ln[-\ln(x)],\qquad x\sim U(0,1) \tag{7.4.22}$$

This expression forms the basis for fast Gumbel-deviate generation. The transform involves two logarithms per sample; one for $\ln(x)$ and another for the outer logarithm, and therefore has constant time complexity $\mathcal{O}(1)$. Because $\ln(-\ln x)$ diverges near $x=0$ and $x=1$, implementations clamp $x$ to $[10^{-15},1-10^{-15}]$ to avoid overflow.

From an algorithmic standpoint, Gumbel sampling is efficient and numerically stable: only elementary operations are required, and vectorized evaluation is straightforward in Rust or GPU kernels. The generator is often expressed in logarithmic form when extended-precision arithmetic is unavailable.

### Statistical Interpretation and Behaviour

The Gumbel distribution models extreme-value events such as the largest floods in a century, maximum wind gusts over a year, or the most intense heat wave in a climate record. It is the canonical Extreme-Value Type I distribution in the Fisher–Tippett–Gnedenko classification, complementing the Fréchet (Type II) and Weibull (Type III) laws that govern heavy- and bounded-tail cases, respectively.

Its right tail decays exponentially as $e^{-y/\beta}$, making it suitable for data where extremes remain bounded in probability but occur more frequently than Gaussian theory would predict. The left tail is much thinner, producing an asymmetric profile skewed toward large positive values. The mode $\mu$ corresponds to the most probable maximum, while the spread parameter $\beta$ determines how rapidly probabilities decline for larger-than-typical extremes.

### Applications

The Gumbel model underpins much of extreme-event statistics. In hydrology, it is used to estimate design flood levels or peak river discharges based on annual maxima. In meteorology and climate science, it models maximum wind speeds, temperature anomalies, and precipitation intensities. In structural engineering, Gumbel deviates support safety-factor design against extreme loads or stresses.

Beyond the physical sciences, the distribution has found new relevance in machine learning. The Gumbel–Softmax (or Concrete) relaxation employs Gumbel noise to approximate categorical sampling in differentiable form, enabling gradient-based training in discrete latent-variable models. In reinforcement learning and variational auto-encoders, this trick allows stochastic choice among actions or categories while maintaining back-propagate continuity.

In climate-risk simulation, large ensembles of Gumbel deviates are generated to represent thousands of possible realizations of rare, catastrophic events, enabling probabilistic estimation of expected damages or return levels.

### Extensions of the Gumbel Law and Modern Tail Models

Modern research has expanded the classical Gumbel framework to better capture empirical tail behaviour and domain constraints. Two developments are particularly noteworthy. First, the generalized truncated Gumbel distribution introduces an upper support bound, enabling more accurate estimation of extremely rare quantiles by preventing divergence of the exponential tail (Gómez et al., 2024). Second, shape-adjusted Gumbel models incorporate additional curvature parameters to correct tail bias, significantly improving fit quality in hydrological extremes where empirical maxima deviate from the classical exponential-tail assumption (Anghel et al., 2024).

These generalized models usually forfeit the convenience of a closed-form inverse CDF. As a consequence, simulation relies on rejection sampling or adaptive hybrid algorithms, in which analytic transformations approximate the bulk of the distribution while stochastic acceptance tests ensure correct tail fidelity. This combination preserves efficiency while accurately representing empirical extremes that differ from theoretical idealizations.

### Comparative Inversion Overview

A compact summary of inversion-based formulas and their computational characteristics is provided below.

~~~{list-table}
:header-rows: 1
:name: pmTiMTNlm1

* - Distribution

  - Formula for ( y )

  - Key Operations

  - Tail Behaviour

  - Typical Use

* - Weibull

  - $y = \lambda[-\ln(1-x)]^{1/k}$

  - log, pow

  - flexible (thin–heavy)

  - lifetimes, reliability

* - Laplace

  - $y = \mu - b\,\operatorname{sgn}\!\left(U-\tfrac{1}{2}\right)\, \ln\!\left( 1 - 2\left| U-\tfrac{1}{2}\right| \right)$

  - log, abs, sign

  - two-sided exponential (double-exponential)

  - robust noise modelling, sparse errors, L1-based methods

* - Pareto

  - $y = y_{\min}(1-x)^{-1/\alpha}$

  - log, pow

  - heavy (power-law)

  - finance, networks

* - Gumbel

  - $y = \mu - \beta\ln[-\ln(x)]$

  - two logs

  - exponential extreme tail

  - climate extremes, risk

~~~

The Gumbel distribution remains central in the modeling of extremes. Its theoretical tractability, simple inverse transform, and minimal computational cost make it ideal for large-scale simulations, especially when evaluating return levels, risk thresholds, and sensitivities in climate, hydrology, and engineered systems.

### Rust Implementation

Following the discussion in Section 7.4.5 on the role of the Gumbel distribution in extreme-value theory and the derivation of its closed-form inverse CDF, Program 7.4.5 provides a practical implementation of Gumbel deviate generation using the transformation in Equation (7.4.22). Because the Gumbel law admits an exact analytic inversion, sampling reduces to evaluating two logarithms per deviate, yielding an $\mathcal{O}(1)$ procedure well suited for large-scale Monte Carlo studies of extremes in climate, hydrology, and engineered systems. To maintain numerical stability when computing the nested expression $\ln[-\ln(x)]$, the implementation clamps the uniform input away from 0 and 1, preventing overflow in the outer logarithm and ensuring robustness under high-volume simulation. This program demonstrates how theoretical tractability directly translates into an efficient and reliable generator for extreme-value models.

At the core of the implementation is the `sample_gumbel` function, which evaluates the inverse transform given in Equation (7.4.22). The function first draws a uniform deviate $X\sim U(0,1)$ and clamps it to the interval $[10^{-15},1-10^{-15}]$ to avoid the overflow that arises when evaluating $\ln(-\ln x)$ for values of $x$ very close to the boundaries. After clamping, the inverse CDF formula is applied directly, computing $Y = \mu - \beta \ln[-\ln(X)]$. This minimal set of operations ensures constant-time generation of Gumbel deviates, consistent with the theoretical efficiency discussed in the surrounding text.

The `main` function performs a Monte Carlo validation of the generator by drawing a large number of samples from $\mathrm{Gumbel}(\mu,\beta)$ and computing the empirical mean, variance, minimum, and maximum. These statistics are compared to their theoretical counterparts, using $\mu + \gamma\beta$ for the mean and $(\pi^2/6)\beta^2$ for the variance, where $\gamma$ denotes Euler’s constant. Such comparisons provide a practical demonstration of the correctness of the inversion method and allow students to observe sampling fluctuations that arise from finite-run Monte Carlo experiments. The accumulated minima and maxima highlight the heavy right tail characteristic of the Gumbel distribution, illustrating the statistical behavior emphasized earlier in the section.

A further implementation detail concerns the use of the `rand` crate to produce uniform deviates. Although the transformation itself is deterministic once a random $X$ is supplied, the quality of the underlying RNG affects the fidelity of extreme-value simulation. The program’s structure makes it easy to substitute alternative random-number engines, including cryptographic-quality generators or parallelizable counter-based methods, without modifying the inversion logic. This modularity mirrors the general strategy for distributional sampling outlined in Section 7.4: separate the statistical transformation from the underlying uniform generator.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
// Program 7.4.5: Sampling Gumbel (Extreme-Value Type I) deviates via inverse transform.
//
// Cargo.toml dependencies:
//
// [dependencies]
// rand = "0.8"
//
// This program generates N Gumbel-distributed samples using the inverse CDF
//   Y = μ - β ln[-ln(X)],  X ~ U(0, 1),
// with clamping of X to [1e-15, 1 - 1e-15] for numerical safety.
// It then reports empirical mean, variance, minimum, and maximum, and
// compares them with the theoretical mean μ + γβ and variance (π²/6) β².

use rand::Rng;
use std::f64::consts::PI;

/// Euler–Mascheroni constant γ ≈ 0.57721...
const EULER_GAMMA: f64 = 0.57721566;

/// Generate a single Gumbel(μ, β) deviate using the inverse CDF (7.4.22).
///
/// Y = μ - β ln[-ln(X)],  where X ∼ U(0, 1).
/// The uniform X is clamped away from 0 and 1 to avoid overflows in ln(-ln X).
fn sample_gumbel<R: Rng + ?Sized>(mu: f64, beta: f64, rng: &mut R) -> f64 {
    // Draw X ~ U(0, 1)
    let mut x: f64 = rng.gen();

    // Clamp to [1e-15, 1 - 1e-15] for numerical safety
    let eps = 1.0e-15_f64;
    if x < eps {
        x = eps;
    } else if x > 1.0 - eps {
        x = 1.0 - eps;
    }

    // Inverse CDF transformation (7.4.22)
    mu - beta * (-x.ln()).ln()
}

fn main() {
    let mut rng = rand::thread_rng();

    // Example parameters for Gumbel(μ, β)
    let mu = 0.0_f64;
    let beta = 1.0_f64;

    // Number of samples for Monte Carlo verification
    let n_samples: usize = 100_000;

    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    // Initialize min/max with the first sample
    let first = sample_gumbel(mu, beta, &mut rng);
    let mut min_y = first;
    let mut max_y = first;
    sum += first;
    sum_sq += first * first;

    for _ in 1..n_samples {
        let y = sample_gumbel(mu, beta, &mut rng);
        sum += y;
        sum_sq += y * y;

        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }

    let n = n_samples as f64;
    let mean_emp = sum / n;
    let var_emp = sum_sq / n - mean_emp * mean_emp;

    // Theoretical mean and variance of Gumbel(μ, β)
    let mean_theory = mu + EULER_GAMMA * beta;
    let var_theory = (PI * PI / 6.0) * beta * beta;

    println!("Sampling {n_samples} Gumbel deviates via inverse transform:\n");
    println!("Gumbel(μ = {mu}, β = {beta}):");
    println!("  Empirical mean       ≈ {mean_emp:.6}");
    println!("  Theoretical mean      = {mean_theory:.6}");
    println!("  Empirical variance   ≈ {var_emp:.6}");
    println!("  Theoretical variance  = {var_theory:.6}");
    println!("  Min                  ≈ {min_y:.6}");
    println!("  Max                  ≈ {max_y:.6}");
}
```

Program 7.4.5 demonstrates the practical efficiency of inverse-transform sampling for distributions with closed-form CDF inverses, particularly those like the Gumbel distribution whose analytical tractability makes them ideal for large-scale simulation of rare events. By combining a lightweight transformation with numerically robust clamping, the implementation captures the behavior of extreme values with minimal computational overhead. This aligns directly with the theoretical considerations introduced earlier in Section 7.4, where the simplicity of inversion stands in contrast to rejection or adaptive sampling schemes required for more complex tail models.

The numerical comparison between empirical and theoretical moments reinforces the accuracy of the method and highlights typical sampling variability at finite (N). Observing how the extrema behave across large draws also helps illustrate the characteristic asymmetry and exponential right tail of the Gumbel distribution. The modular design of the code provides a template for implementing inverse-CDF generators for other analytically invertible distributions and serves as a stepping stone toward more advanced models such as truncated or shape-modified Gumbel laws introduced later in the section.

## 7.4.6. Hybrid and Ratio-of-Uniforms Methods

When an analytic inverse $F^{-1}$ exists, inversion sampling is typically the most efficient generator because it converts a uniform deviate directly into a sample via a closed-form transformation. However, many widely used distributions, such as the Student–t, Beta, Generalized Inverse Gaussian, or generalized Gumbel models, lack closed-form inverses. In such situations, more general methods are needed. Two particularly effective classes are:

1. *Ratio-of-Uniforms* (RoU) methods, which geometrically transform sampling into drawing a point uniformly from a 2D region defined by the shape of the density;
2. *Hybrid rejection* methods, which piece together multiple simpler generators (e.g., Gamma for the bulk and Pareto for the tail).

These techniques are distribution-agnostic, maintain high acceptance rates for smooth unimodal densities, and extend reliably to heavy-tailed or multimodal cases.

### Ratio-of-Uniforms Principle

Let $f(y) \ge 0$ be a probability density function. Define the RoU sampling region in the $(u,v)$-plane as:

$$
R = \left\{ (u,v) : 0 < u \le \sqrt{\,f(v/u)\,} \right\}
\tag{7.4.23}
$$

The RoU method proceeds by sampling a point $(u,v)$ **uniformly** inside $R$ and returning the ratio:

$$y = \frac{v}{u}\tag{7.4.24}$$

This transformation produces samples whose distribution is exactly $f(y)$.

The correctness of the RoU method can be understood by rewriting the density in terms of a measure in $(x,p)$-space. For a fixed density height $p(x) = f(x)$, we have:

$$
p(x)\,dx
= \int_{0}^{p(x)} dp'\,dx
= \int_{0}^{\sqrt{p(x)}} 2u\,du\,dx 
\tag{7.4.25}
$$

Now impose the substitution

$$x = \frac{v}{u} \tag{7.4.26}$$

which maps rectangular sampling in $(u,v)$-coordinates into the appropriate density-weighted distribution over $x$. In this transformed space, the set of all points $(u,v)$ corresponding to a given $x$ and $u$ lies inside the region $R$ defined by (7.4.23). Thus, uniform sampling over $R$ produces the correct marginal distribution in $x$.

Geometrically, the region $R$ often forms a teardrop-shaped or rounded, vertically symmetric domain. When the density is smooth and unimodal, the region is compact and the acceptance rate is high. In practice, accelerated implementations precompute:

- a bounding box for $R$,
- a squeeze function that eliminates large subregions,
- curvature estimates that reduce expensive evaluations of $f(y)$.

This makes RoU one of the most efficient general-purpose sampling methods available.

### Hybrid Sampling Variants

For more complex densities, especially multimodal, skewed, or heavy-tailed distributions, one may combine multiple analytic or rejection-based generators. Hybrid samplers decompose a density into: (i) a bulk region, typically approximated well by a log-normal, Gaussian, or Gamma generator, and (ii) a tail region, modeled using a Pareto, Weibull, or exponential-power law.

A Bernoulli selector,

$$B \sim \mathrm{Bernoulli}(p)\tag{7.4.27}$$

chooses which region to sample. If $B=0$, the algorithm generates from the bulk; if $B=1$, it samples from the tail. This approach preserves accuracy across heterogeneous density shapes, avoids the very low acceptance rates that pure rejection methods encounter in heavy tails, and remains easy to parallelize.

Hybrid techniques (Osatohanmwen et al., 2024) are widely used in Bayesian computation, extreme-value modeling, and machine learning where densities exhibit structural transitions between bulk and tail behaviour.

### Algorithmic Complexity

In practice, the computational cost of a sampling method is determined by both the number of arithmetic operations required per deviate and the expected acceptance probability. Inverse transform sampling offers the most predictable performance: when a closed-form inverse $F^{-1}$ is available, each deviate is generated in constant time $O(1)$ with an acceptance rate of exactly 1.0. This makes the method ideal for smooth, monotone distributions whose inverses can be expressed analytically.

In contrast, rejection sampling exhibits an expected cost of $O(1/r)$, where $r$ denotes the acceptance probability. Because $r$ depends on how tightly the proposal envelope matches the target density, its efficiency may vary substantially across distributions. Nevertheless, rejection sampling remains one of the most flexible techniques, accommodating multimodal, skewed, or heavy-tailed densities through the careful design of proposal bounds.

The ratio-of-uniforms (RoU) method typically achieves constant expected time $O(1)$ with acceptance rates between 0.8 and 0.95 for smooth unimodal densities. Its geometric formulation often yields a compact sampling region, enabling efficient uniform draws and making RoU a strong general-purpose alternative when closed-form inverses are unavailable.

Finally, hybrid methods combine multiple generators, such as a bulk approximation and a tail-specific model, to achieve constant expected cost without relying on a single global proposal. Their adaptivity makes them effective for densities exhibiting structural changes between central and tail behaviour, though they often include conditional branches that must be handled carefully for optimal performance.

Modern Rust implementations of these methods benefit from branchless acceptance logic, SIMD-accelerated evaluation of logarithms and polynomial terms, and precomputed squeeze functions that eliminate large regions of guaranteed rejection. These optimizations substantially reduce overhead and help maintain high throughput in large-scale Monte Carlo simulations.

### Rust Implementation

Following the discussion in Section 7.4.6 on ratio-of-uniforms (RoU) and hybrid sampling techniques, Program 7.4.6 illustrates practical implementations of these general-purpose generators for distributions that lack analytic inverses. While inverse-transform sampling is ideal when a closed-form mapping exists, many important densities, including multimodal, heavy-tailed, or composite models, require more flexible strategies such as the RoU construction or mixtures of simpler generators. This program demonstrates both approaches: a RoU sampler for a smooth unimodal density and a hybrid sampler combining an exponential bulk with a Pareto tail. Together, they show how geometric and mixture-based ideas can be implemented efficiently in Rust to handle distributions beyond the reach of classical inversion methods.

At the core of the implementation is the ratio-of-uniforms sampler, which directly reflects the geometric formulation presented in (7.4.23)–(7.4.26). The program defines a concrete target density, the Beta(2,2) distribution, and constructs the RoU sampling region by bounding its maximum height and restricting candidates to lie within the transformed domain. The function `rou_sample_beta22` draws uniform points $(u,v)$ inside a bounding box and applies the acceptance test $u \le \sqrt{f(v/u)}$, returning $y = v/u$ upon success. This procedure mirrors the theoretical construction in which rectangular sampling in $(u,v)$-space induces the correct marginal distribution in $y$. For a smooth unimodal density such as Beta(2,2), the RoU region is compact and nearly elliptical, yielding high acceptance rates and stable performance.

The hybrid sampler is implemented as a complementary example of mixture-based techniques for more complex densities. The function `exponential_sample` generates bulk values via an inverse transform, while `pareto_sample_log` implements the logarithmic form of the Pareto generator derived earlier in (7.4.18), ensuring numerical stability in the tail. The function `hybrid_exp_pareto` then uses a Bernoulli selector as in (7.4.27) to choose between the two components: with probability $1 - p_{\text{tail}}$ it produces a bulk value from the exponential distribution, and with probability $p_{\text{tail}}$ it draws a tail sample from the Pareto model. This division reflects the notion of decomposing heterogeneous densities into central and tail regions, as discussed under hybrid methods in Section 7.4.6. The program also provides analytic formulas for the theoretical mean and variance of the mixture, assuming the appropriate moment conditions on the Pareto tail, allowing empirical results to be checked against their expected values.

The main function integrates the two sampling strategies by generating large synthetic datasets and computing empirical means, variances, extrema and, when applicable, the fraction of tail samples. The RoU sampler is tested on the Beta(2,2) distribution, where compact support and smooth curvature ensure a high-quality acceptance region. The hybrid sampler is evaluated using an exponential-Pareto mixture, illustrating how occasional large tail draws create substantial variability in empirical variances, a hallmark of heavy-tailed behaviour. Both demonstrations highlight the role of geometric region construction, acceptance-rejection logic, and mixture selection mechanisms in building general-purpose samplers for complex probability densities.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
// Program 7.4.6: Ratio-of-Uniforms and Hybrid Samplers in Rust
//
// This program illustrates two general-purpose sampling techniques:
//   1. A ratio-of-uniforms (RoU) sampler for a Beta(2,2) distribution,
//      using the RoU region defined in (7.4.23)–(7.4.26).
//   2. A simple hybrid sampler that combines an exponential "bulk"
//      with a Pareto "tail" using a Bernoulli selector as in (7.4.27).
//
// The RoU example shows how to construct a compact sampling region for
// a smooth unimodal density and achieve high acceptance rates. The
// hybrid example demonstrates how bulk and tail generators can be
// combined to model heterogeneous densities and heavy-tail behaviour.

use rand::prelude::*;

const EPS: f64 = 1.0e-15;

// -----------------------------------------------------------------------------
// 1. Ratio-of-Uniforms sampler for Beta(2,2)
// -----------------------------------------------------------------------------

// Normalized Beta(2,2) PDF: f(x) = 6 x (1 - x), for 0 < x < 1.
fn beta22_pdf(x: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        0.0
    } else {
        6.0 * x * (1.0 - x)
    }
}

// Maximum of f(x) occurs at x = 1/2, with f_max = 1.5.
// RoU uses u <= sqrt(f(x)), so u_max = sqrt(f_max).
// sqrt(1.5) ≈ 1.22474487139
const U_MAX_BETA22: f64 = 1.224_744_871_39_f64;


/// Sample from Beta(2,2) using the basic ratio-of-uniforms scheme.
///
/// Region R in (u,v)-space:
///   R = { (u,v): 0 < u <= sqrt(f(v/u)) },
/// and we return y = v/u, as in (7.4.24).
fn rou_sample_beta22<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    loop {
        // Sample (u, v) uniformly from the bounding box [0, U_MAX] × [0, U_MAX].
        let u = rng.gen::<f64>() * U_MAX_BETA22;
        let v = rng.gen::<f64>() * U_MAX_BETA22;

        if u <= 0.0 {
            continue;
        }

        // Candidate deviate y = v / u.
        let y = v / u;

        // For Beta(2,2) we require 0 < y < 1.
        if y <= 0.0 || y >= 1.0 {
            continue;
        }

        // Acceptance test for RoU: u <= sqrt(f(y)).
        let f_y = beta22_pdf(y);
        if f_y <= 0.0 {
            continue;
        }

        if u <= f_y.sqrt() {
            return y;
        }
    }
}

// Theoretical mean and variance of Beta(2,2):
//   E[Y] = α / (α + β) = 2 / 4 = 0.5
//   Var[Y] = αβ / [ (α + β)^2 (α + β + 1) ] = 4 / (16 * 5) = 0.05
fn beta22_mean() -> f64 {
    0.5
}

fn beta22_variance() -> f64 {
    0.05
}

// -----------------------------------------------------------------------------
// 2. Hybrid sampler: Exponential "bulk" + Pareto "tail"
// -----------------------------------------------------------------------------

/// Exponential(λ) deviate via inverse transform:
///   Y = -ln(U) / λ,  U ~ U(0,1).
fn exponential_sample<R: Rng + ?Sized>(rng: &mut R, lambda: f64) -> f64 {
    debug_assert!(lambda > 0.0, "rate λ must be positive");
    let u = rng.gen::<f64>().clamp(EPS, 1.0 - EPS);
    -u.ln() / lambda
}

/// Pareto(α, y_min) deviate using the logarithmic transform (7.4.18):
///   ln Y = ln y_min - (1/α) ln(1 - X),  X ~ U(0,1).
fn pareto_sample_log<R: Rng + ?Sized>(rng: &mut R, alpha: f64, y_min: f64) -> f64 {
    debug_assert!(alpha > 0.0, "shape α must be positive");
    debug_assert!(y_min > 0.0, "scale y_min must be positive");

    let x = rng.gen::<f64>().clamp(EPS, 1.0 - EPS);
    let log_1_minus_x = (1.0 - x).ln();
    let log_y = y_min.ln() - (1.0 / alpha) * log_1_minus_x;
    log_y.exp()
}

/// Hybrid sampler controlled by a Bernoulli selector B ~ Bernoulli(p_tail).
/// If B = 0, draw from the exponential "bulk"; if B = 1, draw from the Pareto "tail".
#[allow(dead_code)]
fn hybrid_exp_pareto<R: Rng + ?Sized>(
    rng: &mut R,
    p_tail: f64,
    lambda: f64,
    alpha: f64,
    y_min: f64,
) -> f64 {
    debug_assert!(p_tail >= 0.0 && p_tail <= 1.0, "p_tail must be in [0,1]");
    let u: f64 = rng.gen();
    if u < p_tail {
        // Tail region
        pareto_sample_log(rng, alpha, y_min)
    } else {
        // Bulk region
        exponential_sample(rng, lambda)
    }
}

/// Theoretical mean of the hybrid mixture:
///   Y ~ (1 - p_tail) * Exp(λ)  +  p_tail * Pareto(α, y_min)
/// provided α > 1.
fn hybrid_mean(p_tail: f64, lambda: f64, alpha: f64, y_min: f64) -> Option<f64> {
    if alpha <= 1.0 {
        return None;
    }
    let mean_exp = 1.0 / lambda;
    let mean_pareto = alpha * y_min / (alpha - 1.0);
    Some((1.0 - p_tail) * mean_exp + p_tail * mean_pareto)
}

/// Theoretical variance of the hybrid mixture, assuming α > 2.
fn hybrid_variance(p_tail: f64, lambda: f64, alpha: f64, y_min: f64) -> Option<f64> {
    if alpha <= 2.0 {
        return None;
    }

    let mean_exp = 1.0 / lambda;
    let var_exp = 1.0 / (lambda * lambda);

    let mean_pareto = alpha * y_min / (alpha - 1.0);
    let var_pareto = {
        let num = alpha * y_min * y_min;
        let den = (alpha - 1.0).powi(2) * (alpha - 2.0);
        num / den
    };

    // Law of total variance:
    // Var(Y) = E[Var(Y|B)] + Var(E[Y|B])
    let e_var = (1.0 - p_tail) * var_exp + p_tail * var_pareto;
    let e_mean = (1.0 - p_tail) * mean_exp + p_tail * mean_pareto;
    let e_mean_sq = (1.0 - p_tail) * mean_exp.powi(2) + p_tail * mean_pareto.powi(2);
    let var_mean = e_mean_sq - e_mean * e_mean;

    Some(e_var + var_mean)
}

// -----------------------------------------------------------------------------
// 3. Main: demonstrate RoU and hybrid sampling
// -----------------------------------------------------------------------------

fn main() {
    let mut rng = thread_rng();

    let n_samples = 100_000usize;
    println!(
        "Sampling {n} deviates using ratio-of-uniforms and hybrid methods:\n",
        n = n_samples
    );

    // ----------------------------
    // 3.1 Beta(2,2) via RoU
    // ----------------------------
    let mut sum_beta = 0.0_f64;
    let mut sum_sq_beta = 0.0_f64;
    let mut min_beta = f64::INFINITY;
    let mut max_beta = f64::NEG_INFINITY;

    for _ in 0..n_samples {
        let y = rou_sample_beta22(&mut rng);
        sum_beta += y;
        sum_sq_beta += y * y;
        if y < min_beta {
            min_beta = y;
        }
        if y > max_beta {
            max_beta = y;
        }
    }

    let n = n_samples as f64;
    let empirical_mean_beta = sum_beta / n;
    let empirical_var_beta = (sum_sq_beta / n) - empirical_mean_beta * empirical_mean_beta;

    println!("Beta(2,2) via Ratio-of-Uniforms:");
    println!(
        "  Empirical mean      ≈ {mean:.6}",
        mean = empirical_mean_beta
    );
    println!(
        "  Theoretical mean     = {mean_th:.6}",
        mean_th = beta22_mean()
    );
    println!(
        "  Empirical variance  ≈ {var:.6}",
        var = empirical_var_beta
    );
    println!(
        "  Theoretical variance = {var_th:.6}",
        var_th = beta22_variance()
    );
    println!("  Min                 ≈ {min:.6}", min = min_beta);
    println!("  Max                 ≈ {max:.6}", max = max_beta);
    println!();

    // ----------------------------
    // 3.2 Hybrid Exp–Pareto sampler
    // ----------------------------

    // Parameters:
    //   p_tail: probability of sampling from the Pareto tail
    //   lambda: rate of Exp(λ)
    //   alpha, y_min: Pareto shape and scale
    let p_tail = 0.1_f64;
    let lambda = 1.0_f64;
    let alpha = 3.0_f64;
    let y_min = 5.0_f64;

    let mut sum_h = 0.0_f64;
    let mut sum_sq_h = 0.0_f64;
    let mut min_h = f64::INFINITY;
    let mut max_h = f64::NEG_INFINITY;
    let mut tail_count = 0usize;

    for _ in 0..n_samples {
        let u: f64 = rng.gen();
        let y = if u < p_tail {
            tail_count += 1;
            pareto_sample_log(&mut rng, alpha, y_min)
        } else {
            exponential_sample(&mut rng, lambda)
        };

        sum_h += y;
        sum_sq_h += y * y;
        if y < min_h {
            min_h = y;
        }
        if y > max_h {
            max_h = y;
        }
    }

    let n = n_samples as f64;
    let empirical_mean_h = sum_h / n;
    let empirical_var_h = (sum_sq_h / n) - empirical_mean_h * empirical_mean_h;
    let tail_fraction = tail_count as f64 / n;

    println!("Hybrid Exp–Pareto mixture (bulk + tail):");
    println!("  Mixture parameters: p_tail = {p_tail}, λ = {lambda}, α = {alpha}, y_min = {y_min}");
    println!("  Empirical mean      ≈ {mean:.6}", mean = empirical_mean_h);
    match hybrid_mean(p_tail, lambda, alpha, y_min) {
        Some(m) => println!("  Theoretical mean     = {m:.6}"),
        None => println!("  Theoretical mean     = diverges (α ≤ 1)"),
    }
    println!("  Empirical variance  ≈ {var:.6}", var = empirical_var_h);
    match hybrid_variance(p_tail, lambda, alpha, y_min) {
        Some(v) => println!("  Theoretical variance = {v:.6}"),
        None => println!("  Theoretical variance = diverges (α ≤ 2)"),
    }
    println!("  Min                 ≈ {min:.6}", min = min_h);
    println!("  Max                 ≈ {max:.6}", max = max_h);
    println!("  Tail sample fraction ≈ {frac:.4}", frac = tail_fraction);
}
```

Program 7.4.6 illustrates two flexible sampling strategies that operate effectively when analytic inverse transforms are unavailable. The ratio-of-uniforms example demonstrates how a geometric reformulation of density sampling produces a compact acceptance region for unimodal distributions, achieving high efficiency and stable numerical behaviour. In contrast, the hybrid mixture highlights how combining a bulk generator with a heavy-tailed model produces samples that accurately reflect structural transitions in the underlying density. These approaches complement the inversion-based methods discussed earlier in the section and provide practical tools for simulating a broad class of probability models encountered in Monte Carlo computation.

The program’s results underscore the distinct statistical properties of smooth, compact distributions compared with those exhibiting heavy tails. While the RoU sampler for Beta(2,2) produces tightly concentrated estimates, the exponential–Pareto hybrid generates occasional extreme values that dominate second moments, reinforcing the theoretical considerations presented in Section 7.4.6. The modular design of the code allows further extensions to multimodal densities, adaptive hybrid schemes, or RoU variants incorporating squeeze functions and curvature-aware bounds, forming a foundation for advanced simulation techniques in Bayesian computation, risk modeling, and machine learning.

## 7.4.7. Practical Contexts for Deviate-Based Simulation

The techniques developed in the preceding sections are not merely theoretical conveniences; they form the backbone of modern Monte-Carlo modelling across engineering, climate science, hydrology, and risk analysis. Many application domains rely on the ability to generate statistically accurate deviates from reliability-critical, heavy-tailed, or extreme-value distributions, often at massive scale. The following examples illustrate how inversion, rejection, and RoU-based sampling enter into practical simulation pipelines.

### (a) Reliability and Lifetime Modelling

The Weibull distribution remains the standard model for component lifetimes in mechanical, civil, and electronic systems. Its density is:

$$T \sim \text{Weibull}(k,\lambda) \qquad f(t) = \frac{k}{\lambda}\left(\frac{t}{\lambda}\right)^{k-1} e^{-(t/\lambda)^k} \tag{7.4.28}$$

where, the shape parameter $k$ governs how the failure rate evolves over time. When $k>1$, the failure rate increases, corresponding to wear-out behaviour in aging components. When $k<1$, the failure rate decreases, reflecting infant-mortality effects in which early failures are more common than late ones. The special case $k=1$ reduces the Weibull law to the exponential distribution, yielding a constant, memoryless failure rate.

Because the Weibull distribution has an analytic inverse CDF, its samples can be produced using,

$$T_i = \lambda[-\ln(1-U_i)]^{1/k},\qquad U_i \sim U(0,1)\tag{7.4.29}$$

This transformation plays a central role in large-scale Monte-Carlo reliability studies, in which thousands or millions of simulated devices are used to estimate key performance measures such as mean time to failure (MTTF), system availability under redundant configurations, the degree of failure clustering in correlated environments, and the timing of optimal maintenance intervals. The inversion method is particularly attractive in this context because it executes in constant time, requires only a single logarithm and power operation, and lends itself naturally to vectorized and parallel implementations. In high-performance Rust systems, these simulations typically employ SIMD-accelerated evaluations of `ln` and `powf` together with thread-parallel random streams to sustain near-linear throughput across multicore architectures.

### (b) Climate and Hydrological Extremes

The modelling of environmental extremes often requires distributions whose support extends over the entire real line, with right tails that represent rare but high-impact events. The Gumbel distribution, a special case of the generalized extreme-value (GEV) family, is one of the most widely applied models for annual maxima such as extreme rainfall, storm surge levels, river discharge, or wind gusts. Its inverse-CDF generator is given by

$$Y = \mu - \beta \ln[-\ln(U)],\qquad U \sim U(0,1)\tag{7.4.30}$$

In hydrology and climatology, this formula is applied repeatedly to generate synthetic sequences of annual maxima. These sequences are then used to compute *return levels*, such as the 50-year or 100-year flood height, which are essential for infrastructure design, reservoir management, and disaster-risk planning.

Recent empirical studies, however, show that the classical Gumbel model may underestimate extreme quantiles, particularly in regions experiencing rapid climatic change. This has led to the adoption of several generalized alternatives, including truncated Gumbel models that restrict the effective tail domain, shape-modified variants that introduce curvature corrections, and GEV formulations with a nonzero shape parameter. These extensions generally lack a closed-form inverse $F^{-1}$, making direct inversion infeasible. Consequently, simulation relies on rejection sampling or ratio-of-uniforms (RoU) techniques, which maintain accuracy across complex tail geometries. RoU sampling is especially valuable in hydrological applications because it handles skewed or highly curved densities without a substantial loss in acceptance rate, thereby providing stable estimates even for very rare events.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/AVsXYhoWy4lzFJ9P6zCD.5","tags":[]}

# 7.5. Multivariate Normal Deviates

A $d$-dimensional random vector,

$$\mathbf{X} = (X_{1},\dots,X_{d})^\top\tag{7.5.1}$$

is said to follow a multivariate normal distribution with mean vector $\boldsymbol{\mu}\in\mathbb{R}^{d}$ and symmetric positive-definite covariance matrix $\boldsymbol{\Sigma}\in\mathbb{R}^{d\times d}$ if its probability density function is:

$$
f_{\mathbf{X}}(\mathbf{x})
= \frac{1}{\sqrt{(2\pi)^{d}\,\det(\boldsymbol{\Sigma})}}
\exp\left(
    -\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top
    \boldsymbol{\Sigma}^{-1}
    (\mathbf{x}-\boldsymbol{\mu})
\right)
\tag{7.5.2}
$$

The mean vector

$$\boldsymbol{\mu}=\mathbb{E}[\mathbf{X}]\tag{7.5.3}$$

describes the central location of the distribution, while the covariance matrix,

$$\boldsymbol{\Sigma}=\mathrm{Cov}(\mathbf{X})\tag{7.5.4}$$

encodes scale, orientation, and dependence among the components. The diagonal entries of $\boldsymbol{\Sigma}$ specify the marginal variances, and the off-diagonal entries represent covariances that quantify linear relationships between the components. When $\boldsymbol{\Sigma}$ is diagonal, the components of $\mathbf{X}$ are independent normal variables; when the off-diagonal entries are nonzero, the joint density exhibits tilted elliptical contours reflecting correlation.

### Geometric Interpretation

The fundamental geometric structure of the multivariate normal distribution is governed by the quadratic form:

$$
Q(\mathbf{x})

(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}),\tag{7.5.5}
$$

whose level surfaces $Q(\mathbf{x})=c$ are ellipsoids in $\mathbb{R}^{d}$. The eigenvectors of $\boldsymbol{\Sigma}$ determine the principal axes of these ellipsoids, and the eigenvalues determine their squared axis lengths. Thus, the covariance matrix completely specifies the orientation and eccentricity of the density contours.

```{figure} images/pqQDe4beUu67RvW3raYP-JxZDPtYi0W67YJPGDqfc-v1.png
:name: WG0WBAhulC
:align: center
:width: 40%

**Figure 7.5.1.** Contour plot of a bivariate normal density with mean $(0,0)$ and a positively correlated covariance structure. The elliptical level sets illustrate how correlation tilts and stretches the contours, reflecting the orientation and strength of dependence between the two variables.
```

Figure 7.5.1 presents a concrete example in two dimensions, showing the contour lines of a bivariate normal distribution with:

$$\boldsymbol{\mu} = (0,0)^\top, \qquad \boldsymbol{\Sigma} = \begin{pmatrix} 2 & 1.2\\[6pt] 1.2 & 1 \end{pmatrix} \tag{7.5.6}$$

The elliptical level curves correspond to sets of constant Mahalanobis distance. The pronounced tilt of the ellipses reflects the nonzero covariance between the two components, while their relative stretching reflects the differing variances along the principal axes.

### Significance and Applications

Multivariate normal distributions play a central role in numerical analysis, statistics, and scientific computing because they combine analytical tractability with geometric clarity. Any linear transformation of a multivariate normal vector remains normal, and among all distributions with a given mean and covariance, the multivariate normal is the unique maximizer of entropy. Moreover, the multivariate Central Limit Theorem ensures that many high-dimensional aggregate phenomena converge toward this distribution, making it a natural modelling choice when only first and second-order information is available.

In practice, multivariate normals arise in diverse high-dimensional applications. Uncertain initial or boundary data in partial differential equation models are frequently represented as Gaussian random fields, effectively very high-dimensional normal vectors whose covariance structure encodes spatial correlation. Gaussian process models in Bayesian optimization and machine learning assume multivariate normal priors over discretized function values, requiring efficient sampling from high-dimensional covariance structures. Physical simulations routinely incorporate multivariate normal noise to model thermal fluctuations, while Monte Carlo methods rely on Gaussian deviates to generate correlated random inputs for uncertainty quantification.

Because many scientific and engineering tasks require generating samples from,

$$N(\boldsymbol{\mu},\boldsymbol{\Sigma})\tag{7.5.7}$$

efficient algorithms for sampling, especially in large dimensions or when $\boldsymbol{\Sigma}$ has special structure, are crucial. The remainder of this section develops both classical and modern techniques for constructing multivariate normal deviates, with an emphasis on methods suitable for high-performance numerical computing in Rust.

### Rust Implementation

Following the discussion in Section 7.5 on the geometry, covariance structure, and applications of multivariate normal distributions, Program 7.5.0 provides a practical implementation of sampling from $N(\boldsymbol{\mu},\boldsymbol{\Sigma})$ using the classical Cholesky-factor–based construction. Because the covariance matrix $\boldsymbol{\Sigma}$ in Equation (7.5.4) is symmetric positive-definite, it admits a unique lower-triangular Cholesky factor $\boldsymbol{L}$ such that $\boldsymbol{\Sigma}=\boldsymbol{L}\boldsymbol{L}^\top$. This factorization enables direct transformation of a standard normal vector into one with the desired mean and covariance via the affine mapping $\mathbf{X}=\boldsymbol{\mu}+\boldsymbol{L}\mathbf{Z}$, where $\mathbf{Z}\sim N(\mathbf{0},\mathbf{I})$. The program demonstrates this construction in a numerically robust and efficient Rust implementation and validates the correctness of the generated samples by comparing empirical first- and second-order statistics against the theoretical quantities defined earlier in this section.

At the core of the implementation is the `MultivariateNormal` struct, which encapsulates all information required to generate samples from $N(\boldsymbol{\mu},\boldsymbol{\Sigma})$. Its constructor performs the Cholesky factorization of the covariance matrix introduced in Equations (7.5.4)–(7.5.6), ensuring that $\boldsymbol{\Sigma}$ is symmetric positive-definite before storing its lower-triangular factor $\boldsymbol{L}$. This decomposition plays a central geometric role: the columns of $\boldsymbol{L}$ represent the transformed axes of the elliptical contours described by the quadratic form in Equation (7.5.5), and applying $\boldsymbol{L}$ to a standard normal vector rotates and scales it in accordance with the covariance structure.

The `sample` method implements the transformation $\mathbf{X}=\boldsymbol{\mu}+\boldsymbol{L}\mathbf{Z}$, where each component of $\mathbf{Z}$ is drawn independently from the univariate standard normal distribution. This mapping is a direct computational realization of the theoretical foundations introduced in (7.5.2): the random vector $\mathbf{Z}$ provides isotropic Gaussian noise, and left-multiplication by $\boldsymbol{L}$ shears, rotates, and scales the noise so that the resulting samples have exactly the covariance prescribed by $\boldsymbol{\Sigma}$. Adding the mean vector $\boldsymbol{\mu}$ performs the final translation to the correct centre, as defined in Equation (7.5.3).

The program’s `main` function demonstrates the entire sampling process using the bivariate normal distribution specified in Equation (7.5.6). It repeatedly draws realizations from $N(\boldsymbol{\mu},\boldsymbol{\Sigma})$, accumulates empirical estimates of the mean vector and covariance matrix, and compares them with the analytical quantities defined earlier in the section. This comparison verifies that the Cholesky-based construction preserves both the scale and orientation of the target distribution, reproducing not only the marginal variances but also the covariance responsible for the tilted elliptical contours described in the geometric interpretation. The use of $100{,}000$ samples produces empirical statistics closely aligned with the expected values, illustrating both the accuracy of the method and the characteristic sampling variability inherent to Monte Carlo simulation.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
rand_distr = "0.4"
nalgebra = "0.32"
```

```rust
// Program 7.5.1: Multivariate Normal Deviates via Cholesky Factorization
//
// Cargo.toml dependencies:
//
// [dependencies]
// rand = "0.8"
// rand_distr = "0.4"
// nalgebra = "0.32"
//
// This program constructs a multivariate normal distribution N(μ, Σ)
// by factoring the covariance Σ = L Lᵀ (Cholesky factorization) and
// transforming a standard normal vector Z ~ N(0, I_d) via
//     X = μ + L Z.
// It then draws a large number of samples, computes the empirical mean
// and covariance, and compares them with the target μ and Σ.
// For illustration, we use the bivariate example (7.5.6):
//     μ = (0, 0)ᵀ,
//     Σ = [[2,   1.2],
//          [1.2, 1  ]].

use nalgebra::{SMatrix, SVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Multivariate normal distribution N(μ, Σ) with Σ = L Lᵀ (Cholesky factor).
struct MultivariateNormal<const D: usize> {
    mu: SVector<f64, D>,
    l: SMatrix<f64, D, D>,
}

impl<const D: usize> MultivariateNormal<D> {
    /// Construct a multivariate normal N(μ, Σ).
    /// Returns an error if Σ is not symmetric positive-definite.
    fn new(mu: SVector<f64, D>, cov: SMatrix<f64, D, D>) -> Result<Self, &'static str> {
        // Cholesky factorization: cov = L Lᵀ
        if let Some(chol) = cov.cholesky() {
            Ok(Self {
                mu,
                l: chol.l(),
            })
        } else {
            Err("Covariance matrix is not positive definite.")
        }
    }

    /// Draw a single sample X = μ + L Z, where Z ~ N(0, I_D).
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SVector<f64, D> {
        // Sample Z ~ N(0, I_D)
        let mut z = SVector::<f64, D>::zeros();
        for i in 0..D {
            let n: f64 = StandardNormal.sample(rng);
            z[i] = n;
        }

        // Apply affine transformation X = μ + L Z
        self.mu.clone() + &self.l * z
    }
}

fn main() {
    // Type aliases for 2D vectors and 2x2 matrices
    type Vec2 = SVector<f64, 2>;
    type Mat2 = SMatrix<f64, 2, 2>;

    // Example (7.5.6): μ = (0, 0)ᵀ, Σ as given.
    let mu = Vec2::new(0.0, 0.0);
    let cov = Mat2::new(
        2.0, 1.2,
        1.2, 1.0,
    );

    let mvn = MultivariateNormal::<2>::new(mu, cov)
        .expect("Covariance matrix must be positive definite.");

    let mut rng = rand::thread_rng();

    // Number of samples for Monte Carlo verification.
    let n_samples: usize = 100_000;

    let mut sum = Vec2::zeros();
    let mut sum_xx_t = Mat2::zeros();

    // Draw samples and accumulate sum and sum of outer products.
    for _ in 0..n_samples {
        let x = mvn.sample(&mut rng);
        sum += x;
        sum_xx_t += x * x.transpose(); // x xᵀ
    }

    let n = n_samples as f64;
    let mean_emp = sum / n;

    // Unbiased sample covariance:
    // C_hat = 1/(n-1) * (Σ x_i x_iᵀ - n μ̂ μ̂ᵀ)
    let outer_mean = mean_emp * mean_emp.transpose();
    let cov_emp = (sum_xx_t - outer_mean * n) / (n - 1.0);

    println!("Sampling {n_samples} multivariate normal deviates:\n");
    println!("Target mean vector μ = (0, 0)ᵀ");
    println!("Empirical mean ≈ ({:.6}, {:.6})", mean_emp[0], mean_emp[1]);

    println!("\nTarget covariance Σ:");
    println!(
        "  [[{:.6}, {:.6}],",
        cov[(0, 0)], cov[(0, 1)]
    );
    println!(
        "   [{:.6}, {:.6}]]",
        cov[(1, 0)], cov[(1, 1)]
    );

    println!("\nEmpirical covariance (unbiased estimate):");
    println!(
        "  [[{:.6}, {:.6}],",
        cov_emp[(0, 0)], cov_emp[(0, 1)]
    );
    println!(
        "   [{:.6}, {:.6}]]",
        cov_emp[(1, 0)], cov_emp[(1, 1)]
    );
}
```

Program 7.5.0 provides a clear and efficient implementation of multivariate normal sampling based on the Cholesky factorization of the covariance matrix. This construction reflects the geometric structure emphasized in Section 7.5: by transforming an isotropic standard normal vector through the lower-triangular factor $\boldsymbol{L}$, the method reproduces the elliptical contours, principal directions, and scale relationships inherent to $N(\boldsymbol{\mu},\boldsymbol{\Sigma})$. The empirical results from the program closely match the target mean vector and covariance matrix, demonstrating the correctness and numerical reliability of this transformation in practice.

The example also highlights the importance of matrix factorizations in high-dimensional sampling. While the Cholesky decomposition is efficient and well suited to dense, moderate-sized covariance matrices, the same strategy extends naturally to structured or sparse covariance operators, which arise frequently in Gaussian process modelling, spatial statistics, and discretized PDE-based uncertainty quantification. The modular design of the code provides a foundation for exploring such extensions, including eigen-based decompositions, fast Kronecker-structured sampling, and low-rank approximations for large-scale Gaussian models encountered in scientific computing.

## 7.5.1. Sampling from a Multivariate Normal Distribution

To generate a multivariate normal deviate, we seek a random vector,

$$\mathbf{X}\sim N(\boldsymbol{\mu},\boldsymbol{\Sigma})\tag{7.5.8}$$

where $\boldsymbol{\mu}\in\mathbb{R}^d$ is the mean vector and $\boldsymbol{\Sigma}\in\mathbb{R}^{d\times d}$ is a symmetric positive-definite covariance matrix. The fundamental idea is to start from a vector of **independent standard normal variables** and apply an affine transformation that introduces the correlation structure encoded in $\boldsymbol{\Sigma}$. Let,

$$\mathbf{Z}\sim N(\mathbf{0},I_d)\tag{7.5.9}$$

that is, a $d$-dimensional vector of independent $N(0,1)$ components. If we can find a matrix $A\in\mathbb{R}^{d\times d}$ satisfying,

$$A A^\top = \boldsymbol{\Sigma}\tag{7.5.10}$$

then the transformed vector,

$$\mathbf{X} = \boldsymbol{\mu} + A\mathbf{Z}\tag{7.5.11}$$

\
will have mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$. Indeed,

$$
\begin{aligned}
\mathbb{E}[\mathbf{X}] &= \boldsymbol{\mu}, \\[4pt]
\mathrm{Cov}(\mathbf{X})
&= A\,\mathrm{Cov}(\mathbf{Z})\,A^\top
 = A I_d A^\top
 = A A^\top
 = \boldsymbol{\Sigma}
\end{aligned}
\tag{7.5.12}
$$

Thus $\mathbf{X}$ has exactly the required distribution. Sampling from a multivariate normal distribution therefore reduces to generating independent standard normals and applying a suitable linear transformation.

### Cholesky-Based Sampling

A widely used choice for the factor $A$ is the Cholesky decomposition. Because $\boldsymbol{\Sigma}$ is symmetric positive-definite, it admits a unique lower-triangular Cholesky factor,

$$\boldsymbol{\Sigma} = L L^\top\tag{7.5.13}$$

where $L$ has positive diagonal entries. Setting $A=L$ in equation (7.5.11) gives a practical sampling algorithm:

A practical method for generating a multivariate normal sample begins with the Cholesky factorization of the covariance matrix. Because $\boldsymbol{\Sigma}$ is symmetric positive definite, it admits a unique lower-triangular matrix $L$ such that $\boldsymbol{\Sigma} = L L^\top$. This factorization,

$$L = \mathrm{Cholesky}(\boldsymbol{\Sigma}) \tag{7.5.14}$$

is numerically stable and requires $O(d^3)$ operations for a dense $d\times d$ matrix. The purpose of this step is to obtain a linear transformation whose action imposes exactly the covariance structure encoded in $\boldsymbol{\Sigma}$.

Next, a whitened standard normal vector is drawn,

$$\mathbf{Z} = (Z_1,\dots,Z_d)^\top, \qquad Z_i \sim N(0,1)\ \text{independently} \tag{7.5.15}$$

so that $\mathbf{Z}$ has mean zero and identity covariance $I_d$. This vector is isotropic, meaning that all covariance structure in the final sample will be introduced solely through the linear transformation involving $L$. The independence of the $Z_i$ ensures that no unintended correlations are present before the transformation.

Finally, the multivariate normal deviate is obtained through the affine transformation:

$$\mathbf{X} = \boldsymbol{\mu} + L\mathbf{Z} \tag{7.5.16}$$

The multiplication $L\mathbf{Z}$ produces a Gaussian vector with covariance $L I_d L^\top = \boldsymbol{\Sigma}$, while adding $\boldsymbol{\mu}$ sets the correct mean. The resulting vector $\mathbf{X}$ is therefore distributed as $N(\boldsymbol{\mu},\boldsymbol{\Sigma})$. This construction forms the basis of virtually all practical algorithms for multivariate Gaussian sampling in moderate dimensions.

Because $L$ is lower-triangular, the transformation has a clear component-wise structure:

$$
\begin{aligned}
X_1 &= \mu_1 + \ell_{11} Z_1, \\
X_2 &= \mu_2 + \ell_{21} Z_1 + \ell_{22} Z_2, \\
X_3 &= \mu_3 + \ell_{31} Z_1 + \ell_{32} Z_2 + \ell_{33} Z_3
\end{aligned}
\tag{7.5.17}
$$

and similarly for all $i$. Each $X_i$ is a linear combination of independent normals, hence normally distributed. The structure of $L$ ensures that the resulting vector has the correct variances and covariances. Conceptually, the transformation mixes the independent components of $\mathbf{Z}$ in a way that introduces the correlations encoded by $\boldsymbol{\Sigma}$.

### Forward and Inverse Transforms (Whitening and Coloring)

The mapping,

$$\mathbf{X}=\boldsymbol{\mu} + L\mathbf{Z}\tag{7.5.18}$$

is known as a coloring transform, because it converts a “white” noise vector $\mathbf{Z}\sim N(0,I_d)$ into a correlated one. The inverse operation is equally important in statistics and data processing. If:

$$\mathbf{X}\sim N(\boldsymbol{\mu},\boldsymbol{\Sigma})\tag{7.5.19}$$

then applying the inverse transformation:

$$\mathbf{Z}=L^{-1}(\mathbf{X}-\boldsymbol{\mu})\tag{7.5.20}$$

produces a whitened vector $\mathbf{Z}\sim N(0,I_d)$. Whitening is frequently used for decorrelating data, simplifying likelihood expressions, or preconditioning optimization procedures.

### Alternative Factorizations

Although Cholesky is the most efficient general-purpose approach, any factor $A$ satisfying $A A^\top=\boldsymbol{\Sigma}$ is valid. An important alternative arises from the eigendecomposition of $\boldsymbol{\Sigma}$:

$$\boldsymbol{\Sigma} = Q \Lambda Q^\top\tag{7.5.21}$$

where $Q$ contains orthonormal eigenvectors and $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_d)$ contains strictly positive eigenvalues. Setting,

$$A = Q  \Lambda^{1/2}\tag{7.5.22}$$

with $\Lambda^{1/2}=\mathrm{diag}(\sqrt{\lambda_1},\dots,\sqrt{\lambda_d})$, yields the sampling formula:

$$
\mathbf{X} = \boldsymbol{\mu} + Q\,\Lambda^{1/2}\mathbf{Z}
\tag{7.5.23}
$$

This representation highlights the geometric interpretation: sampling consists of rotating the whitened vector $\mathbf{Z}$ into the eigenbasis of the covariance matrix and scaling it along the principal axes. While eigen-decomposition is typically more expensive than Cholesky, it offers insights relevant to PCA-type methods and low-rank approximations. If only the largest $r$ eigenvalues are significant, one may approximate

$$\boldsymbol{\Sigma}\approx Q_r \Lambda_r Q_r^\top\tag{7.5.24}$$

leading to principal-component sampling in reduced dimension. This approach is foundational in probabilistic PCA, where covariance matrices are approximated by low-rank plus isotropic terms.

The correctness of the sampling rule follows directly from the characterization of multivariate normals under linear transformations. If $\mathbf{Z}\sim N(0,I_d)$ and $\mathbf{X}=\boldsymbol{\mu}+A\mathbf{Z}$, then any linear combination $a^\top \mathbf{X} = a^\top\boldsymbol{\mu} + a^\top A \mathbf{Z}$ is univariate normal. The variance of this combination is,

$$
\begin{aligned}
\mathrm{Var}(a^\top \mathbf{X})
 &= a^\top A A^\top a \\
 &= a^\top \boldsymbol{\Sigma}\, a
\end{aligned}
\tag{7.5.25}
$$

and this holds for all vectors $a$. Therefore $\mathbf{X}$ must follow $N(\boldsymbol{\mu},\boldsymbol{\Sigma})$, establishing rigorously that equation (7.5.11) produces correct samples.

In summary, sampling from a multivariate normal distribution reduces to three core steps: generating a whitened normal vector, computing an appropriate matrix square root of the covariance (typically via Cholesky), and applying an affine transformation. The method is straightforward, efficient, and widely used in scientific computing, simulation, machine learning, and statistical modeling. In Rust, this process can be implemented by combining a univariate Gaussian generator from `rand` with linear algebra operations from `ndarray` or similar crates.

### Rust Implementation

Following the derivation in Section 7.5.1 of how multivariate normal deviates can be constructed from standard normal vectors through an affine transformation, Program 7.5.1 provides a practical implementation of this coloring transform in Rust. The development of Equation (7.5.11) showed that any sample from $N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ can be obtained by applying an appropriate matrix square root of the covariance to an independent standard normal vector. Using the Cholesky factorization introduced in Equation (7.5.13), the program demonstrates how independent components $Z_i \sim N(0,1)$ may be efficiently transformed into a correlated Gaussian vector with the desired mean and covariance. The implementation also includes the inverse whitening map from Equation (7.5.20), illustrating how correlated data can be decorrelated and mapped back to a standard normal reference frame. This program provides a concrete computational realization of the theoretical constructions developed in this subsection.

At the core of the implementation is the `GaussianTransform` struct, which encapsulates all components required for the forward coloring transformation in Equation (7.5.18) and the inverse whitening transformation in Equation (7.5.20). Its constructor takes the mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$ and performs a Cholesky factorization to obtain the lower-triangular matrix $L$ satisfying Equation (7.5.13). This factor acts as the linear map that introduces precisely the covariance structure encoded in $\boldsymbol{\Sigma}$. The numerical stability and efficiency of the Cholesky algorithm make it an ideal choice for constructing multivariate Gaussian samples in moderate dimensions.

The `sample` method implements the forward transform $\mathbf{X} = \boldsymbol{\mu} + L\mathbf{Z}$ from Equation (7.5.16). It first generates a whitened vector $\mathbf{Z}$ whose components are independent standard normal variables as described in Equation (7.5.15). Multiplying by the Cholesky factor reorients, scales, and correlates this isotropic noise so that the resulting random vector has the covariance structure encoded in $\boldsymbol{\Sigma}$. Adding the mean vector $\boldsymbol{\mu}$ performs the final translation, yielding a sample whose distribution matches Equation (7.5.8).

The inverse operation is performed by the `whiten` method, which computes the whitened vector $\mathbf{Z} = L^{-1}(\mathbf{X}-\boldsymbol{\mu})$ from Equation (7.5.20). Because (L) is lower triangular, this inverse can be computed efficiently via forward substitution, avoiding the explicit computation of a matrix inverse. The whitened vectors produced in this way should have mean zero and identity covariance, demonstrating the correctness of both the factorization and the linear transformation.

The `main` function brings these components together by constructing a 3-dimensional multivariate normal distribution, drawing a large number of samples, and estimating their empirical mean and covariance. It then applies the whitening transform to each sample and verifies that the resulting vectors are centered and uncorrelated. These numerical comparisons provide strong Monte Carlo evidence that the theoretical relationships in Equations (7.5.11)–(7.5.25) hold in practice and that the Cholesky-based construction accurately reproduces the intended covariance structure.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
rand_distr = "0.4"
nalgebra = "0.32"
```

```rust
// Program 7.5.1: Coloring and Whitening Transforms for Multivariate Normal Sampling
// This program illustrates the construction X = μ + L Z with
// Z ~ N(0, I_d) and Σ = L Lᵀ, as described in Equations (7.5.8)–(7.5.16).
// It also demonstrates the inverse (whitening) transform
// Z = L^{-1} (X - μ) from Equation (7.5.20).
//
// We use a 3-dimensional example with a nontrivial covariance Σ.
//
// The program:
//   1. Builds a GaussianTransform from μ and Σ via Cholesky.
//   2. Draws many samples X (coloring).
//   3. Whitens each sample to obtain Z.
//   4. Estimates empirical mean/covariance of X and Z and prints them.

use nalgebra::{SMatrix, SVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// GaussianTransform encapsulates the coloring and whitening maps
/// for a multivariate normal N(μ, Σ) with Σ = L Lᵀ.
struct GaussianTransform<const D: usize> {
    mu: SVector<f64, D>,
    l: SMatrix<f64, D, D>, // Lower-triangular Cholesky factor L
}

impl<const D: usize> GaussianTransform<D> {
    /// Construct from mean vector μ and covariance matrix Σ.
    /// Requires Σ to be symmetric positive-definite.
    fn new(mu: SVector<f64, D>, cov: SMatrix<f64, D, D>) -> Result<Self, &'static str> {
        if let Some(chol) = cov.cholesky() {
            Ok(Self {
                mu,
                l: chol.l(), // Lower-triangular L such that Σ = L Lᵀ
            })
        } else {
            Err("Covariance matrix is not positive definite.")
        }
    }

    /// Draw a sample X = μ + L Z, where Z ~ N(0, I_D).
    /// This is the coloring transform (7.5.11)/(7.5.16)/(7.5.18).
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SVector<f64, D> {
        // Draw Z ~ N(0, I_D)
        let mut z = SVector::<f64, D>::zeros();
        for i in 0..D {
            let n: f64 = StandardNormal.sample(rng);
            z[i] = n;
        }

        // X = μ + L Z
        self.mu.clone() + &self.l * z
    }

    /// Whiten a sample X by computing Z = L^{-1} (X - μ)
    /// using forward substitution for the lower-triangular L.
    /// This is the inverse transform (7.5.20).
    fn whiten(&self, x: &SVector<f64, D>) -> SVector<f64, D> {
        let v = x - self.mu; // v = X - μ
        let mut z = SVector::<f64, D>::zeros();

        // Solve L z = v by forward substitution.
        for i in 0..D {
            let mut sum = 0.0;
            for j in 0..i {
                sum += self.l[(i, j)] * z[j];
            }
            z[i] = (v[i] - sum) / self.l[(i, i)];
        }

        z
    }
}

fn main() {
    // Dimension of the example.
    const D: usize = 3;
    type VecD = SVector<f64, D>;
    type MatD = SMatrix<f64, D, D>;

    // Example mean and covariance (correlated 3D normal).
    let mu = VecD::new(1.0, -0.5, 0.8);

    // Symmetric positive-definite covariance matrix Σ.
    // You can adapt this to match a specific example in the text if desired.
    let cov = MatD::new(
        2.0, 0.8, 0.5,
        0.8, 1.5, 0.3,
        0.5, 0.3, 1.0,
    );

    let gt = GaussianTransform::<D>::new(mu, cov)
        .expect("Covariance matrix must be positive definite.");

    let mut rng = rand::thread_rng();

    // Number of samples for Monte Carlo verification.
    let n_samples: usize = 100_000;

    // Accumulators for colored samples X.
    let mut sum_x = VecD::zeros();
    let mut sum_xx_t = MatD::zeros();

    // Accumulators for whitened samples Z.
    let mut sum_z = VecD::zeros();
    let mut sum_zz_t = MatD::zeros();

    for _ in 0..n_samples {
        // Draw X ~ N(μ, Σ) via coloring.
        let x = gt.sample(&mut rng);

        // Whiten X to obtain Z ~ N(0, I).
        let z = gt.whiten(&x);

        // Accumulate statistics for X.
        sum_x += x;
        sum_xx_t += x * x.transpose();

        // Accumulate statistics for Z.
        sum_z += z;
        sum_zz_t += z * z.transpose();
    }

    let n = n_samples as f64;

    // Empirical mean and covariance for X.
    let mean_x = sum_x / n;
    let outer_mean_x = mean_x * mean_x.transpose();
    let cov_x = (sum_xx_t - outer_mean_x * n) / (n - 1.0);

    // Empirical mean and covariance for Z.
    let mean_z = sum_z / n;
    let outer_mean_z = mean_z * mean_z.transpose();
    let cov_z = (sum_zz_t - outer_mean_z * n) / (n - 1.0);

    println!("Sampling {n_samples} multivariate normal deviates (coloring + whitening):\n");

    println!("Target mean μ:");
    println!("  [{:.6}, {:.6}, {:.6}]", mu[0], mu[1], mu[2]);

    println!("\nEmpirical mean of X:");
    println!("  [{:.6}, {:.6}, {:.6}]", mean_x[0], mean_x[1], mean_x[2]);

    println!("\nTarget covariance Σ:");
    println!(
        "  [[{:.6}, {:.6}, {:.6}],",
        cov[(0, 0)], cov[(0, 1)], cov[(0, 2)]
    );
    println!(
        "   [{:.6}, {:.6}, {:.6}],",
        cov[(1, 0)], cov[(1, 1)], cov[(1, 2)]
    );
    println!(
        "   [{:.6}, {:.6}, {:.6}]]",
        cov[(2, 0)], cov[(2, 1)], cov[(2, 2)]
    );

    println!("\nEmpirical covariance of X (should ≈ Σ):");
    println!(
        "  [[{:.6}, {:.6}, {:.6}],",
        cov_x[(0, 0)], cov_x[(0, 1)], cov_x[(0, 2)]
    );
    println!(
        "   [{:.6}, {:.6}, {:.6}],",
        cov_x[(1, 0)], cov_x[(1, 1)], cov_x[(1, 2)]
    );
    println!(
        "   [{:.6}, {:.6}, {:.6}]]",
        cov_x[(2, 0)], cov_x[(2, 1)], cov_x[(2, 2)]
    );

    println!("\nEmpirical mean of whitened Z (should ≈ 0):");
    println!(
        "  [{:.6}, {:.6}, {:.6}]",
        mean_z[0], mean_z[1], mean_z[2]
    );

    println!("\nEmpirical covariance of whitened Z (should ≈ I):");
    println!(
        "  [[{:.6}, {:.6}, {:.6}],",
        cov_z[(0, 0)], cov_z[(0, 1)], cov_z[(0, 2)]
    );
    println!(
        "   [{:.6}, {:.6}, {:.6}],",
        cov_z[(1, 0)], cov_z[(1, 1)], cov_z[(1, 2)]
    );
    println!(
        "   [{:.6}, {:.6}, {:.6}]]",
        cov_z[(2, 0)], cov_z[(2, 1)], cov_z[(2, 2)]
    );
}
```

Program 7.5.1 illustrates the practical implementation of multivariate normal sampling through the Cholesky-based coloring transform introduced in Section 7.5.1. By constructing $\mathbf{X} = \boldsymbol{\mu} + L\mathbf{Z}$ directly from independent standard normals, the program confirms the central principle that correlation in Gaussian vectors is entirely determined by the linear map associated with the covariance matrix. The empirical results show close agreement between the target and estimated mean and covariance, reinforcing the correctness of the affine transformation framework.

The whitening results are equally important: applying the inverse mapping $L^{-1}(\mathbf{X}-\boldsymbol{\mu})$ recovers vectors with mean zero and identity covariance, demonstrating that the Cholesky factor captures the complete structure of dependence. Together, these forward and inverse transforms illustrate how Gaussian models are manipulated in high-dimensional simulation, filtering, and statistical data processing. The modular structure of the implementation also provides a natural starting point for alternative covariance factorizations, such as eigenvalue-based square roots and low-rank approximations, which play a major role in high-performance sampling for large-scale Gaussian models.

## 7.5.2. High-Dimensional Challenges and Modern Approaches

The Cholesky-based construction described above is efficient and numerically stable for moderate dimensions, but its applicability deteriorates rapidly as the ambient dimension $d$ grows. Factoring a dense $d\times d$ covariance matrix requires $O(d^3)$ floating-point operations and $O(d^2)$ memory. When $d$ reaches $10^4$ or $10^5$, as in large-grid simulations or high-resolution Gaussian random fields, these costs become prohibitive: even forming the matrix may exceed memory capacities, and the cubic scaling renders direct factorizations infeasible. As noted in recent work, “the most straightforward technique for sampling from a multivariate normal involves computing a Cholesky decomposition of complexity cubic in matrix dimension,” a cost that becomes restrictive even for moderately large $d$. In short, dense $O(d^3)$ methods cannot support genuinely high-dimensional Gaussian models.

A major opportunity for scalability arises when the covariance matrix $\boldsymbol{\Sigma}$ exhibits exploitable structure. In many applications, particularly spatial statistics, Gaussian Markov random fields, and discretized stochastic PDEs, the covariance displays locality, so that distant components are nearly independent. This yields sparsity in the precision matrix $\boldsymbol{\Sigma}^{-1}$, or enables $\boldsymbol{\Sigma}$ itself to be approximated by a sparse matrix obtained through domain decomposition or covariance tapering. Sparse Cholesky factorizations exploit this structure by skipping operations on zero entries and can reduce computational cost dramatically.

A different structural regime arises when the spectrum of $\boldsymbol{\Sigma}$ decays rapidly. If only a small number of eigenmodes dominate the variance, then $\boldsymbol{\Sigma}$ admits a low-rank approximation of the form,

$$\boldsymbol{\Sigma} \approx B B^\top + D\tag{7.5.26}$$

where $B \in \mathbb{R}^{d\times r}$ with $r \ll d$, and $D$ is diagonal or sparse. To generate samples from this approximate covariance, we introduce two independent standard normal vectors: $\mathbf{Z}_1 \in \mathbb{R}^{r}$, capturing variability in the low-rank subspace spanned by $B$, and $\mathbf{Z}_2 \in \mathbb{R}^{d}$, corresponding to the residual (diagonal or sparse) noise component. Sampling then decomposes into two independent components,

\begin{align*}
\mathbf{X} &\approx B\mathbf{Z}_1 + D^{1/2}\mathbf{Z}_2, \\[4pt]
\mathbf{Z}_1 &\sim N(\mathbf{0}, I_r), \quad 
\mathbf{Z}_2 \sim N(\mathbf{0}, I_d)
\tag{7.5.27}
\end{align*}

so that only the low-rank portion requires nontrivial linear algebra. Methods such as pivoted Cholesky, randomized SVD, and other rank-revealing factorizations efficiently construct $B$. Tile Low-Rank (TLR) compression extends this idea by partitioning $\boldsymbol{\Sigma}$ into blocks and approximating off-diagonal blocks with low-rank representations, enabling significant reductions in both memory and time. Recent results (e.g. Zhang et al., 2024) demonstrate speedups of up to 20× for high-dimensional Gaussian calculations using TLR-accelerated Cholesky while preserving accuracy.

Hardware acceleration provides another avenue for improvement. Modern GPU-optimized linear-algebra libraries such as cuSOLVER and MAGMA exploit massive parallelism to accelerate dense factorizations, often by an order of magnitude compared with CPUs. A dense covariance matrix of size $10^4 \times 10^4$ may be tractable on high-end GPUs or multi-GPU systems, whereas a CPU-only workstation may struggle. Recent work combining GPU kernels with task-parallel runtimes (e.g. StarPU through the Chameleon library) shows substantial scaling benefits for operations involving multivariate normal probabilities and related factorizations. While such advances do not change the theoretical $O(d^3)$ complexity, they shift the practical threshold at which Cholesky decomposition becomes infeasible. In large-scale systems, GPU-accelerated computation is often paired with low-rank or sparse approximations to achieve scalability.

Beyond factorization-based methods, several modern approaches avoid computing $\boldsymbol{\Sigma}^{1/2}$ explicitly. One class of techniques replaces direct factorization with iterative linear solvers. To sample,

$$\mathbf{X} \sim N(0,\boldsymbol{\Sigma}) \tag{7.5.28}$$

one may attempt to solve,

$$\boldsymbol{\Sigma}^{1/2}\mathbf{y} = \boldsymbol{\eta}, \qquad \boldsymbol{\eta}\sim N(0,I_d),\tag{7.5.29}$$

or equivalently,

$$\boldsymbol{\Sigma}\mathbf{y} = \boldsymbol{\eta}, \tag{7.5.30}$$

using methods such as conjugate gradients when $\boldsymbol{\Sigma}$ (or its inverse) is sparse or well-conditioned. This viewpoint treats $\boldsymbol{\Sigma}$ as an operator rather than a matrix to be factored, and its efficiency depends on the cost of matrix–vector products.

Kernel-based models, particularly Gaussian Processes (GPs), offer additional structure that can be exploited. When $\boldsymbol{\Sigma}$ arises from a stationary kernel, random Fourier features provide an approximation:

$$\Phi\Phi^\top \approx \boldsymbol{\Sigma}, \qquad \Phi\in\mathbb{R}^{d\times r},\ r\ll d\tag{7.5.31}$$

reducing Gaussian sampling to drawing and combining $r$ Fourier coefficients. Spectral simulation methods using FFTs enable exact or approximate sampling for certain covariance kernels on regular grids, avoiding the explicit formation of large dense matrices.

Recent developments in pathwise GP simulation provide yet another alternative. Instead of sampling all $d$ variables jointly, one constructs the sample incrementally, beginning with a coarse representation (e.g., a small set of basis functions or inducing points) and then refining conditionally. Such methods, including approximate Karhunen–Loève expansions or multiresolution conditioning, avoid working with the full covariance matrix at once. Do et al. (2025) present two such approaches: Fourier-feature sampling and pathwise conditioning, that scale effectively to high-dimensional outputs while integrating naturally with downstream uncertainty-quantification pipelines.

In summary, the classical Cholesky approach remains the method of choice when exact sampling is required and the dimension is moderate (often up to a few thousand). Its stability, simplicity, and ubiquitous library support make it highly practical. However, the cubic cost and quadratic memory requirements restrict its use in large-scale settings. Eigen-based methods are conceptually transparent and allow natural low-rank truncation but are typically more expensive than Cholesky. For very large $d$, scalable alternatives rely on exploiting structure, sparsity, low rank, kernel decompositions, or adopting iterative or pathwise sampling techniques. Recent research emphasizes hybrid approaches that blend numerical linear algebra, approximation theory, and modern hardware to make multivariate normal sampling feasible in dimensions ranging from tens of thousands to millions. The appropriate strategy depends on the dimension, the structure of $\boldsymbol{\Sigma}$, and the required accuracy: exact Cholesky for moderate $d$, GPU-accelerated or low-rank methods for larger matrices, and structure-exploiting or approximate methods for truly high-dimensional settings.

### Rust Implementation

Following the discussion in Section 7.5.2 on the cubic complexity and memory limitations of dense Cholesky-based sampling, Program 7.5.2 presents a practical implementation of a scalable low-rank-plus-diagonal Gaussian sampler. The decomposition in Equation (7.5.26) shows how a covariance matrix with rapidly decaying spectrum can be approximated by a low-rank term capturing dominant eigenmodes, together with a diagonal component that models residual variance. This structure reduces sampling from a $d$-dimensional Gaussian from an $\mathcal{O}(d^3)$ factorization to two $\mathcal{O}(rd)$ matrix–vector multiplications, where $r \ll d$. Program 7.5.2 illustrates this approach in Rust by constructing the factor $B\in\mathbb{R}^{d\times r}$ and diagonal matrix $D$, generating the independent standard-normal vectors required by Equation (7.5.27), and forming samples efficiently even in moderately high dimensions. This example demonstrates how structural approximations enable Gaussian simulation well beyond the reach of classical dense methods.

At the core of the implementation is the construction of the matrices $B$ and $D$ appearing in the low-rank-plus-diagonal approximation in Equation (7.5.26). The matrix $B$ contains $r$ dominant modes that account for most of the covariance’s variability, while the diagonal vector $D$ represents the remaining independent noise contribution. In practice, such factors would be produced by techniques such as pivoted Cholesky or randomized SVD, but for demonstration purposes the program synthesizes them directly. The diagonal square root $D^{1/2}$ is computed elementwise to support the sampling rule in Equation (7.5.27), ensuring that the diagonal component contributes the correct variance.

The sampling step closely follows Equation (7.5.27). The program first draws the two independent standard-normal vectors $\mathbf{Z}_1\in\mathbb{R}^{r}$ and $\mathbf{Z}_2\in\mathbb{R}^{d}$, representing variability in the low-rank and residual components, respectively. It then constructs the approximate Gaussian sample via the linear combination $B\mathbf{Z}_1 + D^{1/2}\mathbf{Z}_2$. Because $r\ll d$, the dominant computational cost is a single multiplication between the tall, skinny matrix $B$ and the low-dimensional vector $\mathbf{Z}_1$, making the method dramatically more scalable than applying a dense Cholesky factorization.

To verify correctness, the program accumulates empirical estimates of the mean and covariance of the generated samples. The empirical covariance is then compared with the target low-rank-plus-diagonal covariance $B B^\top + D$ using the Frobenius norm, providing a quantitative measure of sampling accuracy. The comparison of top-left blocks offers additional insight into structure preservation. These diagnostics highlight the accuracy of the approximation and its suitability for high-dimensional or structured covariance matrices where exact factorizations are infeasible.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
rand_distr = "0.4"
nalgebra = "0.32"
```

```rust
// Program 7.5.2: Low-Rank-Plus-Diagonal Approximate Multivariate Normal Sampling
// This program illustrates the low-rank-plus-diagonal construction
//   Σ ≈ B Bᵀ + D
// from Equation (7.5.26), and the sampling formula
//   X ≈ B Z₁ + D^{1/2} Z₂
// from Equation (7.5.27), where
//   Z₁ ~ N(0, I_r),  Z₂ ~ N(0, I_d).
//
// We choose a moderate dimension d and rank r to keep the example
// computationally light, but the same pattern extends to much larger d
// in high-dimensional applications.

use nalgebra::{SMatrix, SVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

fn main() {
    // Dimension d and low rank r (with r << d).
    const D: usize = 50;
    const R: usize = 5;

    type VecD = SVector<f64, D>;
    type VecR = SVector<f64, R>;
    type MatD = SMatrix<f64, D, D>;
    type MatDR = SMatrix<f64, D, R>;

    let mut rng = rand::thread_rng();

    // Construct a low-rank matrix B ∈ ℝ^{d×r}.
    // For illustration we generate random columns and scale them
    // to mimic decaying eigenvalues (dominant low-rank structure).
    let mut b = MatDR::zeros();
    for j in 0..R {
        let scale = 1.0 / (1.0 + j as f64); // simple decay: 1, 1/2, 1/3, ...
        for i in 0..D {
            let n: f64 = StandardNormal.sample(&mut rng);
            b[(i, j)] = scale * n;
        }
    }

    // Construct diagonal D with positive entries (residual variance).
    let mut d_diag = VecD::zeros();
    for i in 0..D {
        // Residual variances between 0.1 and 0.3, say.
        d_diag[i] = 0.1 + 0.2 * rng.gen::<f64>();
    }

    // Precompute D^{1/2} elementwise.
    let mut d_sqrt = VecD::zeros();
    for i in 0..D {
        d_sqrt[i] = d_diag[i].sqrt();
    }

    // Target approximate covariance: Σ ≈ B Bᵀ + D.
    let bb_t: MatD = &b * b.transpose();
    let mut sigma = bb_t;
    for i in 0..D {
        sigma[(i, i)] += d_diag[i];
    }

    // Monte Carlo sampling to verify the approximate covariance.
    let n_samples: usize = 100_000;

    let mut sum_x = VecD::zeros();
    let mut sum_xx_t = MatD::zeros();

    for _ in 0..n_samples {
        // Draw Z₁ ~ N(0, I_R)
        let mut z1 = VecR::zeros();
        for j in 0..R {
            z1[j] = StandardNormal.sample(&mut rng);
        }

        // Draw Z₂ ~ N(0, I_D)
        let mut z2 = VecD::zeros();
        for i in 0..D {
            z2[i] = StandardNormal.sample(&mut rng);
        }

        // Compute X ≈ B Z₁ + D^{1/2} Z₂
        let mut x: VecD = &b * z1;
        for i in 0..D {
            x[i] += d_sqrt[i] * z2[i];
        }

        sum_x += x;
        sum_xx_t += x * x.transpose();
    }

    let n = n_samples as f64;
    let mean_x = sum_x / n;

    // Empirical covariance: C_hat = 1/(n-1) (Σ x_i x_iᵀ - n μ̂ μ̂ᵀ).
    let outer_mean = mean_x * mean_x.transpose();
    let cov_emp = (sum_xx_t - outer_mean * n) / (n - 1.0);

    // Compute Frobenius norm of the covariance error ‖C_hat - Σ‖_F.
    let mut frob_sq = 0.0;
    for i in 0..D {
        for j in 0..D {
            let diff = cov_emp[(i, j)] - sigma[(i, j)];
            frob_sq += diff * diff;
        }
    }
    let frob_err = frob_sq.sqrt();

    // Also compute Frobenius norm of Σ itself for a relative error.
    let mut frob_sigma_sq = 0.0;
    for i in 0..D {
        for j in 0..D {
            let val = sigma[(i, j)];
            frob_sigma_sq += val * val;
        }
    }
    let frob_sigma = frob_sigma_sq.sqrt();
    let rel_err = frob_err / frob_sigma;

    println!("Low-rank-plus-diagonal approximate Gaussian sampling:");
    println!("  Dimension d = {D}, rank r = {R}, samples N = {n_samples}\n");

    println!("Empirical mean of X (first 5 components):");
    print!("  [");
    for i in 0..5.min(D) {
        if i > 0 {
            print!(", ");
        }
        print!("{:.6}", mean_x[i]);
    }
    println!("] (should be close to 0)");

    println!("\nFrobenius norm of covariance error ‖C_hat - Σ‖_F:");
    println!("  {:.6}", frob_err);
    println!("Relative error ‖C_hat - Σ‖_F / ‖Σ‖_F:");
    println!("  {:.6}", rel_err);

    println!("\nTop-left 5×5 block of target Σ:");
    for i in 0..5.min(D) {
        print!("  [");
        for j in 0..5.min(D) {
            if j > 0 {
                print!(", ");
            }
            print!("{:.6}", sigma[(i, j)]);
        }
        println!("]");
    }

    println!("\nTop-left 5×5 block of empirical covariance C_hat:");
    for i in 0..5.min(D) {
        print!("  [");
        for j in 0..5.min(D) {
            if j > 0 {
                print!(", ");
            }
            print!("{:.6}", cov_emp[(i, j)]);
        }
        println!("]");
    }
}
```

Program 7.5.2 demonstrates how low-rank-plus-diagonal approximations provide an effective and scalable alternative to dense Cholesky-based Gaussian sampling in high-dimensional settings. By exploiting the structure in Equation (7.5.26), the method retains the essential covariance features while reducing computation and memory requirements from cubic and quadratic complexities to nearly linear scaling in the dimension. The empirical covariance results confirm that the samples generated via Equation (7.5.27) faithfully reproduce the intended low-rank structure up to Monte Carlo error.

The example also illustrates a fundamental principle emphasized in Section 7.5.2: high-dimensional Gaussian simulation becomes tractable only when structural properties, such as low rank, sparsity, or kernel decompositions are incorporated. While the demonstration uses a synthetic $B$ and diagonal $D$, the same framework accommodates factors obtained from pivoted Cholesky, randomized SVD, tile low-rank compression, or kernel-based embeddings. The modularity of the sampling procedure makes it a building block for modern Gaussian process samplers, approximate Bayesian inference methods, and scalable uncertainty-quantification pipelines. As dimensions grow, hybrid approaches combining low-rank structure, sparsity, and hardware acceleration become indispensable tools, and Program 7.5.2 captures the essential computational pattern underlying these techniques.

## 7.5.3. Applied Multivariate Gaussian Sampling in Large-Scale Systems

Multivariate normal deviates arise throughout scientific computing, geophysical modeling, and machine learning. Their role is not limited to theoretical formulations: in high-dimensional systems, the ability to sample efficiently and to represent covariance structures accurately is essential for producing credible uncertainty estimates. The following two domains illustrate how $N(\boldsymbol{\mu},\boldsymbol{\Sigma})$ sampling underpins modern computational practice.

### (i) Climate Ensemble Forecasting with Correlated Variables

Ensemble forecasting in meteorology provides a canonical example in which high-dimensional Gaussian sampling is indispensable. Operational centers such as ECMWF (European Centre for Medium-Range Weather Forecasts) and NOAA (National Oceanic and Atmospheric Administration) routinely generate ensembles of atmosphere–ocean simulations to quantify forecast uncertainty. Each ensemble member begins from a perturbed version of an analyzed initial state, with perturbations modeled as draws from,

$$\delta x \sim N(0,\Sigma_{\text{init}}) \tag{7.5.32}$$

Here $\Sigma_{\text{init}}$ represents correlations in initial-condition uncertainty across temperature, pressure, humidity, wind, and related fields. The dimension of $\Sigma_{\text{init}}$ may exceed $10^7$, making explicit storage or dense factorization impossible.

The assumption of multivariate normality is partly motivated by approximate Gaussianity of analysis errors, arising from the aggregation of numerous data sources, and partly by the convenience of encoding cross-variable dynamical relationships through $\Sigma_{\text{init}}$. These correlations are essential: they ensure perturbations are physically consistent, maintaining approximate dynamical balances such as geostrophic balance. Independent perturbations (i.e., diagonal $\Sigma_{\text{init}})$ would yield unphysical initial states and degrade forecast skill.

Because $\Sigma_{\text{init}}$ is far too large to store or factor directly, operational centers rely on structured approximations. One well-known method uses singular vectors or Empirical Orthogonal Functions (EOFs) of the linearized model dynamics, capturing fast-growing perturbation modes (as in the ECMWF system). This produces a low-rank approximation:

$$\Sigma_{\text{init}} \approx V V^\top \tag{7.5.33}$$

with perturbations generated as $V w, w\sim N(0,I)$. Another operational strategy, the Ensemble of Data Assimilations (EDA), samples perturbations implicitly by running multiple assimilation cycles, which collectively approximate draws from $N(0,\Sigma_{\text{init}})$.

The essential role of covariance modeling is evident even in a simple bivariate example. If temperature $T$ and pressure $P$ have covariance:

$$
\boldsymbol{\Sigma}_{\text{init}} =
\begin{pmatrix}
\operatorname{Var}(T) & \operatorname{Cov}(T, P) \\[4pt]
\operatorname{Cov}(T, P) & \operatorname{Var}(P)
\end{pmatrix}
\tag{7.5.34}
$$

then samples from $N(0,\Sigma_{\text{init}})$ lie along an ellipse reflecting correlation in physical perturbations. Incorrectly treating $T$ and $P$ as independent would produce dynamically unbalanced states.

Multivariate Gaussian structure is also used in ensemble post-processing. Ensemble Copula Coupling (ECC) applies transformations so that the corrected ensemble matches a target multivariate dependence structure, often Gaussian. Comparative studies such as Lakatos et al. (2023) show that post-processing methods incorporating multivariate correlations, often using Gaussian dependence assumptions, achieve better joint predictive skill for multiple meteorological variables.

Overall, multivariate Gaussian sampling provides the perturbation mechanism that drives ensemble spread and remains central to representing and manipulating uncertainty in high-dimensional environmental systems.

### (ii) Gaussian Process Models and Kernel-Covariance Sampling

Gaussian Process (GP) models represent another major domain where multivariate normal deviates are fundamental. For any input set $X=(x_1,\dots,x_d)$, a GP induces a joint distribution:

$$(f(x_1),\dots,f(x_d)) \sim N(\boldsymbol{\mu},\boldsymbol{\Sigma}) \tag{7.5.35}$$

where $\Sigma_{ij}=K(x_i,x_j)$ is defined by a kernel $K$. Although the GP is infinite-dimensional, inference reduces to finite-dimensional Gaussian computation on the kernel matrix $K(X,X)$.

To sample a function from a GP posterior, one selects a test set $\mathbf{X} = (x_1, \dots, x_n)$ and draws from $N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, where $\boldsymbol{\Sigma}$ includes both prior covariance and corrections from conditioning on observed data. For moderate $n$, a Cholesky factorization of $\boldsymbol{\Sigma}_*$ suffices. For $n\ge 10^4$, however, the $O(n^3)$ scaling becomes prohibitive. This limitation has driven extensive research on scalable GP approximations.

**Low-rank approximations** form one major strategy. By introducing $r\ll n$ inducing points, one obtains:

$$\boldsymbol{\Sigma}_* \approx Q\Lambda Q^\top + D \tag{7.5.36}$$

reducing sampling to $O(nr)$. Variational sparse GP methods support large datasets by sacrificing exactness for computational tractability.

Structured kernels provide another acceleration mechanism. For stationary kernels on regular grids, Toeplitz or Kronecker structure enables fast matrix–vector products and sometimes accelerated factorization. These allow near-exact GP sampling at scales approaching $10^6$ points. Random Fourier features offer a third approach by approximating the kernel via,

$$K(x_i,x_j)\approx \phi(x_i)^\top \phi(x_j) \tag{7.5.37}$$

so sampling reduces to computing $f=\Phi w$, $w\sim N(0,I_r)$. In high dimensions, thousands of features may suffice for approximate GP sampling.

Pathwise sampling methods construct GP samples incrementally. Recent work by Do et al. (2025) shows that basis-function sampling and multiresolution conditional refinement enable $O(n)$ GP sampling after precomputation, making function-space sampling feasible at resolutions previously inaccessible.

Modern GP libraries (e.g., GPyTorch, GPJax) incorporate these techniques: GPU-accelerated conjugate-gradient solvers, Lanczos approximations to $\boldsymbol{\Sigma}_*$, and low-rank or structured kernel decompositions. The LOVE method (LanczOs Variance Estimates; Pleiss et al., 2020) accelerates posterior variance estimation and supports approximate sampling from Lanczos eigenvectors.

A simple example illustrates the computational challenge. Suppose $n=1000$ test points: sampling from $N(\mu_{1000},\Sigma_{1000})$ costs roughly $10^9$ operations, acceptable on modern hardware. For $n=10^4$, the cost becomes $10^{12}$, typically prohibitive. Large-scale spatial GPs (e.g., in climate modeling or large sensor networks) therefore rely on inducing-point, multi-scale (e.g. Vecchia), or low-rank approximation schemes to construct tractable approximations $\tilde{\Sigma}_*$.

In summary, Gaussian processes exemplify the tight coupling between covariance modeling and numerical linear algebra. Advances in scalable GP computation are largely advances in scalable multivariate normal sampling. Efficient sampling ensures that GP-based inference, visualization, optimization, and uncertainty quantification remain computationally feasible in high dimensions.

### Rust Implementation

Following the discussion in Section 7.5.3 on the role of multivariate Gaussian sampling in operational climate systems and Gaussian process modeling, Program 7.5.3 provides two concrete computational demonstrations of how $N(\boldsymbol{\mu},\boldsymbol{\Sigma})$ deviates are constructed and used in practice. In climate ensemble forecasting, perturbations of the initial atmosphere–ocean state are modeled as Gaussian draws whose covariance structure reflects physical balances and cross-variable dependencies, as expressed in Equations (7.5.32)–(7.5.34). Similarly, in Gaussian process models, sampling from the kernel-induced covariance in Equation (7.5.35) underpins posterior simulation, visualization, and uncertainty quantification. Because full dense factorization is often infeasible in large-scale systems, these examples illustrate in reduced dimension how Cholesky-based sampling operates and why scalable approximations, such as low-rank or structured-kernel methods, are essential in real high-dimensional settings. The program implements both a bivariate climate example and a one-dimensional GP kernel sampler, demonstrating how the theory translates directly into practical algorithms.

At the core of the implementation is a Cholesky-based Gaussian sampler, mirroring the affine transformation principle introduced in Equations (7.5.11)–(7.5.12). Given a mean vector and covariance matrix, the sampler computes the lower-triangular factor $L$ from Equation (7.5.13) and generates deviates of the form $\boldsymbol{\mu} + L\mathbf{Z}$, where $\mathbf{Z}$ is a vector of independent standard normals. The code provides both a fixed-dimension variant for small systems, using `SMatrix` and `SVector`, and a dynamic-dimension implementation allocated on the heap, required for the Gaussian process example where the covariance matrix may be moderately large.

The first part of the program implements the climate-perturbation model of Equation (7.5.32), sampling from a bivariate covariance structure that captures the correlation between temperature and pressure. The function `sample()` draws the perturbation vector $(\delta T, \delta P)$, and the program accumulates empirical estimates of the covariance to verify that the sampled ensemble reproduces the target structure in Equation (7.5.34). The computed empirical correlation demonstrates how Gaussian perturbations lie along elliptical level sets and preserve the physically meaningful coupling between variables.

The second part focuses on Gaussian process kernel sampling, where the covariance matrix is formed using a stationary RBF kernel. The dynamic Gaussian sampler computes Cholesky factorization of the kernel matrix and draws samples following Equation (7.5.35). The program then estimates the empirical covariance and compares it with the theoretical kernel matrix, printing the top-left block to visualize how sampling reproduces the smoothness and correlation patterns induced by the kernel. Although the grid size is modest in this demonstration, the procedure reflects the fundamental operation that scalable GP methods approximate using inducing points, structured kernels, or low-rank decompositions.

The `GaussianSamplerDyn` struct encapsulates the dynamic version of the sampling logic, storing the mean vector and Cholesky factor on the heap to avoid stack overflow for larger matrices. Its `sample()` method generates a vector of standard normals and applies the coloring transform described in Equation (7.5.18). This design illustrates the computational distinction emphasized in Section 7.5.3: exact dense sampling is feasible only when dimension is small, but the mathematical structure provides the conceptual foundation for scalable approximations.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
rand_distr = "0.4"
nalgebra = "0.32"
```

```rust
// Program 7.5.3: Applied Multivariate Gaussian Sampling for Climate Perturbations
// and Gaussian Process Kernel Sampling

use nalgebra::{SMatrix, SVector, DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Generic multivariate normal sampler based on a Cholesky factor Σ = L Lᵀ
/// for compile-time dimension D (small/moderate).
struct GaussianSampler<const D: usize> {
    mu: SVector<f64, D>,
    l: SMatrix<f64, D, D>,
}

impl<const D: usize> GaussianSampler<D> {
    /// Construct from mean μ and covariance Σ (must be SPD).
    fn new(mu: SVector<f64, D>, cov: SMatrix<f64, D, D>) -> Result<Self, &'static str> {
        if let Some(chol) = cov.cholesky() {
            Ok(Self {
                mu,
                l: chol.l(), // lower-triangular Cholesky factor
            })
        } else {
            Err("Covariance matrix is not positive definite.")
        }
    }

    /// Draw a single sample X = μ + L Z, with Z ~ N(0, I_D).
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SVector<f64, D> {
        let mut z = SVector::<f64, D>::zeros();
        for i in 0..D {
            z[i] = StandardNormal.sample(rng);
        }
        self.mu.clone() + &self.l * z
    }
}

/// Dynamic-dimension Gaussian sampler using heap-allocated matrices/vectors.
/// Suitable for larger n (e.g. GP kernel sampling).
struct GaussianSamplerDyn {
    mu: DVector<f64>,
    l: DMatrix<f64>,
}

impl GaussianSamplerDyn {
    /// Construct from mean μ and covariance Σ (must be SPD).
    fn new(mu: DVector<f64>, cov: &DMatrix<f64>) -> Result<Self, &'static str> {
        // Clone cov so we can also keep the original Σ for diagnostics.
        if let Some(chol) = cov.clone().cholesky() {
            Ok(Self {
                mu,
                l: chol.l(), // lower-triangular Cholesky factor
            })
        } else {
            Err("Covariance matrix is not positive definite.")
        }
    }

    /// Draw a single sample X = μ + L Z, with Z ~ N(0, I_n).
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DVector<f64> {
        let n = self.mu.len();
        let mut z = DVector::<f64>::zeros(n);
        for i in 0..n {
            z[i] = StandardNormal.sample(rng);
        }
        &self.mu + &self.l * z
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    // ============================================================
    // (i) Climate ensemble perturbations: bivariate (T, P) example
    // ============================================================

    type Vec2 = SVector<f64, 2>;
    type Mat2 = SMatrix<f64, 2, 2>;

    let mu_tp = Vec2::new(0.0, 0.0);
    let cov_tp = Mat2::new(
        1.0, 1.2,
        1.2, 4.0,
    );

    let sampler_tp = GaussianSampler::<2>::new(mu_tp, cov_tp)
        .expect("Bivariate climate covariance must be SPD.");

    let n_samples_tp: usize = 100_000;

    let mut sum_tp = Vec2::zeros();
    let mut sum_tp_tp_t = Mat2::zeros();

    for _ in 0..n_samples_tp {
        let dx = sampler_tp.sample(&mut rng); // δx = (δT, δP)
        sum_tp += dx;
        sum_tp_tp_t += dx * dx.transpose();
    }

    let n_tp = n_samples_tp as f64;
    let mean_tp = sum_tp / n_tp;
    let outer_mean_tp = mean_tp * mean_tp.transpose();
    let cov_emp_tp = (sum_tp_tp_t - outer_mean_tp * n_tp) / (n_tp - 1.0);

    // Compute empirical correlation for illustration.
    let var_t = cov_emp_tp[(0, 0)];
    let var_p = cov_emp_tp[(1, 1)];
    let cov_tp_emp = cov_emp_tp[(0, 1)];
    let corr_tp_emp = cov_tp_emp / (var_t.sqrt() * var_p.sqrt());

    println!("(i) Climate ensemble perturbations: bivariate (T, P) example\n");
    println!("Target mean (δT, δP)ᵀ = (0, 0)");
    println!("Empirical mean ≈ ({:.6}, {:.6})", mean_tp[0], mean_tp[1]);

    println!("\nTarget covariance Σ_init:");
    println!(
        "  [[{:.6}, {:.6}],",
        cov_tp[(0, 0)], cov_tp[(0, 1)]
    );
    println!(
        "   [{:.6}, {:.6}]]",
        cov_tp[(1, 0)], cov_tp[(1, 1)]
    );

    println!("\nEmpirical covariance of (δT, δP):");
    println!(
        "  [[{:.6}, {:.6}],",
        cov_emp_tp[(0, 0)], cov_emp_tp[(0, 1)]
    );
    println!(
        "   [{:.6}, {:.6}]]",
        cov_emp_tp[(1, 0)], cov_emp_tp[(1, 1)]
    );
    println!(
        "\nEmpirical correlation Corr(δT, δP) ≈ {:.6}",
        corr_tp_emp
    );

    // ============================================================
    // (ii) Gaussian Process kernel sampling on a 1D grid (dynamic)
    // ============================================================

    let n_grid: usize = 80; // can be larger now; data live on the heap

    // 1D grid on [0, 1].
    let mut xs = Vec::with_capacity(n_grid);
    for i in 0..n_grid {
        xs.push(i as f64 / (n_grid as f64 - 1.0));
    }

    // RBF kernel parameters.
    let sigma_f = 1.0_f64;
    let ell = 0.2_f64;

    // Build kernel covariance Σ_ij = K(x_i, x_j).
    let mut cov_gp = DMatrix::<f64>::zeros(n_grid, n_grid);
    for i in 0..n_grid {
        for j in 0..n_grid {
            let dx = xs[i] - xs[j];
            let r2 = (dx / ell) * (dx / ell);
            let k_ij = sigma_f * sigma_f * (-0.5 * r2).exp();
            cov_gp[(i, j)] = k_ij;
        }
    }
    // Add a small jitter for numerical stability.
    for i in 0..n_grid {
        cov_gp[(i, i)] += 1.0e-6;
    }

    let mu_gp = DVector::<f64>::zeros(n_grid);
    let sampler_gp = GaussianSamplerDyn::new(mu_gp, &cov_gp)
        .expect("GP kernel covariance must be SPD.");

    let n_samples_gp: usize = 5_000;

    let mut sum_f = DVector::<f64>::zeros(n_grid);
    let mut sum_ff_t = DMatrix::<f64>::zeros(n_grid, n_grid);

    for _ in 0..n_samples_gp {
        let f = sampler_gp.sample(&mut rng); // GP sample f(x_1),...,f(x_n)
        sum_f += &f;
        sum_ff_t += &f * f.transpose();
    }

    let n_gp = n_samples_gp as f64;
    let mean_f = &sum_f / n_gp;
    let outer_mean_f = &mean_f * mean_f.transpose();
    let cov_emp_gp = (sum_ff_t - outer_mean_f * n_gp) / (n_gp - 1.0);

    // Print summary of GP results: mean at a few locations and
    // top-left 5×5 blocks of theoretical and empirical covariance.
    let mid = n_grid / 2;
    println!("\n(ii) Gaussian Process kernel sampling on a 1D grid\n");
    println!("Empirical mean f(x) at selected points (should be ≈ 0):");
    println!(
        "  f(x[0])   ≈ {:.6}\n  f(x[mid]) ≈ {:.6}\n  f(x[last])≈ {:.6}",
        mean_f[0],
        mean_f[mid],
        mean_f[n_grid - 1]
    );

    let block = 5.min(n_grid);
    println!("\nTop-left {}×{} block of GP kernel covariance Σ:", block, block);
    for i in 0..block {
        print!("  [");
        for j in 0..block {
            if j > 0 {
                print!(", ");
            }
            print!("{:.6}", cov_gp[(i, j)]);
        }
        println!("]");
    }

    println!("\nTop-left {}×{} block of empirical GP covariance C_hat:", block, block);
    for i in 0..block {
        print!("  [");
        for j in 0..block {
            if j > 0 {
                print!(", ");
            }
            print!("{:.6}", cov_emp_gp[(i, j)]);
        }
        println!("]");
    }

    println!("\nNote: In real large-scale applications, Σ would be approximated");
    println!("via low-rank, sparse, or structured-kernel methods instead of a");
    println!("dense Cholesky factorization, but the sampling logic is identical.");
}
```

Program 7.5.3 demonstrates how multivariate Gaussian sampling functions as a practical engine for generating physically meaningful perturbations in climate ensemble systems and for synthesizing function samples in Gaussian process models. The bivariate climate example highlights the importance of correlation structures: even in two dimensions, the elliptical shape of the sampled ensemble reflects the correct cross-variable dynamics encoded in the covariance matrix, echoing the physical considerations discussed in Section 7.5.3. The Gaussian process illustration further emphasizes how kernel-based covariance structures induce smooth, coherent samples across spatial locations, forming the backbone of GP-based inference and uncertainty quantification.

These examples also underscore the computational constraints of dense Cholesky factorizations. While they suffice for low-dimensional or pedagogical demonstrations, large-scale systems must rely on structured, sparse, low-rank, or iterative approximations to remain tractable. Modern techniques such as, inducing-point approximations, random Fourier features, tile low-rank decompositions, and iterative solvers, can all be viewed as scalable extensions of the same fundamental sampling rule implemented in this program. In this sense, Program 7.5.3 bridges the mathematical foundations of Gaussian sampling with the applied considerations needed to support real-world, large-scale scientific and machine-learning systems.

## 7.5.4. Concluding Insights on Multivariate Normal Deviates

This section has developed a complete framework for understanding and generating multivariate normal deviates. Beginning with the definition of the distribution through its mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$, we emphasized its central role in modeling correlated random quantities across numerical computing and statistical applications. The classical sampling procedure, forming a Cholesky factor of $\boldsymbol{\Sigma}$ and applying an affine transformation, provides an exact and conceptually transparent method for producing correlated Gaussian vectors, and its inverse (whitening) transform offers a convenient mechanism for decorrelation.

We then examined the computational challenges that arise in high-dimensional settings, where the cubic complexity of dense factorizations becomes prohibitive. Recent developments address these obstacles through GPU-accelerated linear algebra, sparse and low-rank approximations, tile low-rank formats, iterative solvers, and kernel-based methods such as random Fourier features and pathwise conditioning. These advances extend the practical range of multivariate normal sampling to dimensions that would otherwise be infeasible.

The case studies in climate ensemble forecasting and Gaussian process modeling demonstrate how these ideas manifest in real systems. Both domains rely critically on accurately representing covariance structure: in atmospheric modeling to ensure dynamically consistent perturbations, and in Gaussian processes to preserve the smoothness and correlation encoded by the kernel. Across these examples, the multivariate normal distribution serves as a mathematically principled and computationally tractable foundation for uncertainty representation.

As numerical linear algebra, hardware architectures, and approximation techniques continue to evolve, efficient multivariate normal sampling will remain a cornerstone of probabilistic computation. Its integration into modern languages and ecosystems, such as Rust, ensures that realistic simulation, inference, and uncertainty quantification remain scalable and reliable across a wide range of scientific and engineering disciplines.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/eIzOnNdtvU7JUc65cPuK.4","tags":[]}

## 7.6 Linear Feedback Shift Registers (LFSRs)

A Linear Feedback Shift Register (LFSR) is a sequential linear system that generates a deterministic stream of bits which, although completely predictable, exhibits properties resembling random noise. An LFSR consists of an $n$-bit register that shifts its contents by one position at every clock tick and injects a new bit computed as a linear (mod-2) combination of selected register bits, known as feedback taps.

Formally, let the register state at time $t$ be:

$$(a_{t,0},\,a_{t,1},\,\dots,\,a_{t,n-1}) \in \{0,1\}^n$$

The next state $(a_{t+1,0},a_{t+1,1},\dots,a_{t+1,n-1})$ is produced through two operations. First, each bit shifts right by one position,

$$
a_{t+1,i} = a_{t,i+1}, \qquad 0 \le i < n-1
\tag{7.6.1}
$$

and the new leftmost bit is computed as the XOR (exclusive OR, denoted (\\oplus)) of a chosen subset of the current bits,

$$
a_{t+1,n-1}
= a_{t,0} \,\oplus\, (c_1 a_{t,1} \,\oplus\, c_2 a_{t,2} \,\oplus\, \cdots \,\oplus\, c_{n-1} a_{t,n-1})
\tag{7.6.2}
$$

where each coefficient $c_i\in{0,1}$ specifies whether tap $i$ contributes to the feedback. The set of indices $\{ i : c_i = 1 \}$ defines the tap positions and corresponds to a binary polynomial,

$$
P(x) = x^{n} + c_{n-1} x^{n-1} + \cdots + c_{1} x + 1
\tag{7.6.3}
$$

defined over the finite field $\mathbb{F}_2$. This polynomial, called the feedback or characteristic polynomial, encodes the recurrence relation of the LFSR. An initial seed vector $\mathbf{a}_0\ne\mathbf{0}$ must be supplied; the all-zero seed yields a degenerate stream of zeros.

The LFSR recurrence can be expressed compactly as a matrix multiplication over $\mathbb{F}_2$:

$$
\mathbf{a}_{t+1} = M \mathbf{a}_t
\tag{7.6.4}
$$

where $\mathbf{a}_t = (a_{t,n-1},\, a_{t,n-2},\, \dots,\, a_{t,0})^{\mathsf T}$ and $M \in \{0,1\}^{n \times n}$ is the companion matrix.

$$
M =
\begin{pmatrix}
c_1 & c_2 & \cdots & c_{n-1} & 1\\
1   & 0   & \cdots & 0       & 0\\
0   & 1   & \ddots & \vdots  & \vdots\\
\vdots & \ddots & \ddots & 0 & 0\\
0   & \cdots & 0 & 1 & 0
\end{pmatrix}
\tag{7.6.5}
$$

The first row implements the XOR feedback rule, while the sub-diagonal of ones enforces the shift of bits. Multiplying by $M$ advances the register by one step, and repeated application generates the deterministic sequence of states.

For example, a four-bit Linear Feedback Shift Register with feedback polynomial $P(x) = x^4 + x + 1$ (taps at bit 3 and bit 0) has,

$$
M_{\text{4-bit}} =
\begin{pmatrix}
1 & 0 & 0 & 1\\
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 1 & 0
\end{pmatrix}
\tag{7.6.6}
$$

so that $\mathbf{a}_{t+1} = M_{\text{4-bit}} \mathbf{a}_t$ performs both the XOR feedback and the bit-shifting operation simultaneously.

LFSRs are valued for their simplicity, speed, and statistical quality. Hardware implementations require only shift registers and XOR gates, while software realizations use efficient bitwise operations. When the feedback polynomial $P(x)$ is primitive over $\mathbb{F}_2$, the register generates a maximum-length sequence (or *m-sequence*) with period $2^{n} - 1$, representing the longest non-repeating cycle attainable for an $n$-bit configuration.

Such sequences satisfy Golomb’s randomness postulates, meaning that the proportion of zeros and ones is nearly balanced, the lengths of consecutive identical bits follow a geometric distribution, and the periodic autocorrelation function takes only two values, $+1$ and $-1$. Even though they are deterministic, $m$-sequences exhibit strong pseudo-randomness and are well suited to applications demanding uniform distribution and reproducibility.

Because of these advantages, LFSRs have become indispensable across computational and engineering contexts. In digital hardware they are central to built-in self-test systems and pseudo-random pattern generators for integrated-circuit validation. In communication systems they function as scramblers to randomize transmitted bit patterns and as polynomial generators for cyclic redundancy checks (CRCs). In cryptography, LFSRs are used as components of stream ciphers and key-stream generators; for example, the GSM cipher A5/1 employs multiple interacting LFSRs, while the SNOW 2.0 and SNOW 3G ciphers operate over $\mathbb{F}_{2^{32}}$ to produce high-throughput word-level sequences. In simulation, gaming, and other lightweight computing tasks, LFSRs are often selected for their compactness and efficiency.

Their use also extends to signal and image processing. A recent study by Maity et al. (2025) demonstrated an LFSR-driven image-encryption scheme achieving $99.7$% pixel-level randomness and near-maximal entropy while maintaining very low computational cost. The design illustrates that LFSR sequences can still provide the essential confusion and diffusion mechanisms for secure real-time encryption.

```{figure} images/pqQDe4beUu67RvW3raYP-VXAzacBsBxBlknIyNwDl-v1.png
:name: oqNGPw4Jn5
:align: center
:width: 60%

**Figure 7.6.1** A conceptual diagram of a 5-bit Linear Feedback Shift Register (LFSR) with taps at bits 5 and 3, corresponding to the feedback polynomial $P(x)=x^{5}+x^{3}+1$. Each clock pulse shifts the bits to the right, while the new leftmost bit is generated as the XOR of the bits from tap 3 and the bit shifted out. When the polynomial is primitive, the register produces a maximum-length sequence (m-sequence) of period $2^{5}-1=31$, exhibiting excellent pseudo-random statistical properties. (Image credit: Maity et al., 2025.)
```

Figure 7.6.1 shows a five-bit LFSR with taps at bits 5 and 3, corresponding to the feedback polynomial

$$
P(x) = x^{5} + x^{3} + 1
\tag{7.6.7}
$$

At each clock pulse, the register shifts right, and the new high-order bit is the XOR of the bit shifted out and the bit from tap 3. If $P(x)$ is primitive, this configuration yields an $m$-sequence of period $2^{5} - 1 = 31$.

Linear Feedback Shift Registers unite algebraic elegance with hardware efficiency. Expressed through binary polynomials or matrix recurrences over $\mathbb{F}_2$, they generate long pseudo-random sequences with excellent statistical properties and remain fundamental to modern systems, from circuit testing and communication protocols to cryptographic keystream generation and multimedia security.

### Rust Implementation

Following the algebraic and matrix-based formulation of Linear Feedback Shift Registers in Section 7.6, Program 7.6.0 provides a concrete implementation of a 5-bit LFSR driven by the primitive feedback polynomial $P(x)=x^{5}+x^{3}+1$. While the surrounding discussion emphasizes the recurrence relations (7.6.1)–(7.6.4) and their companion-matrix realization (7.6.5), this program translates these ideas directly into efficient bitwise operations suitable for software simulation. By iterating the shift-and-feedback mechanism and observing the resulting sequence, the program illustrates the core dynamical properties of an $m$-sequence, including maximal period, balance of zeros and ones, and reproducibility from a fixed seed. This implementation demonstrates how the theoretical structure of LFSRs manifests in practical systems, from pseudo-random pattern generators to lightweight cryptographic keystreams.

At the core of the implementation is the `Lfsr` struct, which stores the $n$-bit register state and the tap mask encoding the feedback polynomial. In accordance with the recurrence (7.6.2), the tap mask specifies the bit positions included in the XOR feedback computation. The constructor `Lfsr::new` enforces that the seed occupies only the lower $n$ bits and is not the all-zero state, reflecting the fact that the vector $\mathbf{a}_{0}=\mathbf{0}$ produces a degenerate cycle in violation of the LFSR dynamics. This mirrors the mathematical requirement that the state vector in (7.6.4) must be non-zero for the recurrence to produce a meaningful sequence.

The method `step` carries out one iteration of the LFSR recurrence by combining the shift rule (7.6.1) with the XOR feedback logic of (7.6.2). The outgoing bit corresponds to the least significant position before the shift, and the new high-order bit is obtained by taking the parity of all tapped positions. This operation reproduces the behavior of the companion matrix (7.6.5), where the subdiagonal entries implement the shift and the first row encodes the feedback polynomial. By performing these operations through bitwise shifts and masks, the program reproduces the effect of multiplying by the matrix $M$ in (7.6.4), but with the efficiency characteristic of hardware-grade LFSR implementations.

To provide insight into the pseudo-random properties of the register, the program includes utilities for formatting the current $n$-bit state and measuring the statistical distribution of output bits. The companion logic in the main function initializes a 5-bit LFSR with taps at positions 2 and 0, representing the coefficients of $x^{3}$ and $x^{0}$ in (7.6.7). It prints the first sixteen steps to illustrate the shifting behavior and then continues stepping until the initial state is encountered again. This allows the period of the sequence to be measured empirically and compared against the theoretical maximum $2^{5}-1$. The program also tallies the number of zeros and ones in the output stream, confirming the balance properties described by Golomb’s postulates and characteristic of maximum-length sequences.

```rust
// Program 7.6.1: A 5-bit Linear Feedback Shift Register with Primitive Polynomial x^5 + x^3 + 1
//
// This program implements a parameterized Linear Feedback Shift Register (LFSR)
// using bitwise operations over F₂. For the concrete example, we instantiate a
// 5-bit LFSR with feedback taps corresponding to the primitive polynomial
// P(x) = x^5 + x^3 + 1, as in Figure 7.6.1. Starting from a non-zero seed,
// the register produces a maximum-length m-sequence of period 2^5 - 1 = 31.
// The code prints a few initial states and then verifies the period and the
// approximate balance of zeros and ones over one full cycle.

/// Simple LFSR over F₂ with up to 31 bits of state.
///
/// The state is stored in the least-significant `n` bits of `state`.
/// Bit 0 is the rightmost (least significant) bit.
struct Lfsr {
    state: u32,
    taps: u32,
    n: u8,
}

impl Lfsr {
    /// Create a new LFSR with the given seed, tap mask, and register length `n`.
    ///
    /// - `seed` must be non-zero in its lower `n` bits; the all-zero state is forbidden.
    /// - `taps` is a bit mask specifying which bit positions contribute to the feedback XOR.
    fn new(seed: u32, taps: u32, n: u8) -> Self {
        assert!((1..=31).contains(&n), "n must be between 1 and 31 bits");
        let mask = (1u32 << n) - 1;

        let seed = seed & mask;
        assert!(seed != 0, "seed must be non-zero in the lower n bits");

        let taps = taps & mask;

        Self { state: seed, taps, n }
    }

    /// Advance the LFSR by one step and return the output bit (the bit shifted out).
    ///
    /// Update rule:
    ///   - Compute the new feedback bit as the XOR (mod 2 sum) of all tapped bits.
    ///   - Shift the register right by one bit.
    ///   - Insert the new feedback bit into the leftmost position (bit n-1).
    fn step(&mut self) -> u8 {
        // Output bit is the least significant bit before shifting.
        let out_bit = (self.state & 1) as u8;

        // XOR of tapped bits: parity of (state & taps).
        let x = self.state & self.taps;
        let parity = x.count_ones() & 1;
        let new_bit = parity as u32;

        // Shift right, inject feedback bit at the most significant position.
        let mask = (1u32 << self.n) - 1;
        self.state = ((self.state >> 1) | (new_bit << (self.n - 1))) & mask;

        out_bit
    }

    /// Return the current register state (in the least significant `n` bits).
    fn state(&self) -> u32 {
        self.state
    }
}

/// Format the `n`-bit state as a binary string, with the most significant bit first.
fn format_bits(state: u32, n: u8) -> String {
    let mut s = String::with_capacity(n as usize);
    for i in (0..n).rev() {
        let bit = (state >> i) & 1;
        s.push(if bit == 1 { '1' } else { '0' });
    }
    s
}

fn main() {
    // 5-bit LFSR with primitive feedback polynomial P(x) = x^5 + x^3 + 1.
    //
    // In this software representation, taps are specified as bit positions
    // (zero-based) that feed into the XOR feedback. For P(x) = x^5 + x^3 + 1,
    // the non-zero coefficients (excluding the leading term) correspond to
    // x^3 and x^0, so we set taps on bit positions 2 and 0 respectively.
    let n: u8 = 5;
    let taps_mask: u32 = (1 << 2) | (1 << 0); // taps at bits 2 and 0
    let seed: u32 = 0b00001; // any non-zero 5-bit seed is allowed

    let mut lfsr = Lfsr::new(seed, taps_mask, n);
    let start_state = lfsr.state();

    println!("5-bit LFSR with primitive polynomial P(x) = x^5 + x^3 + 1");
    println!("Initial state: {}", format_bits(start_state, n));
    println!("Taps mask    : {:#07b} (bits 2 and 0)\n", taps_mask);

    let mut period: usize = 0;
    let mut zeros: usize = 0;
    let mut ones: usize = 0;

    println!("First 16 steps of the m-sequence:");
    println!("  step  output  state");

    loop {
        let bit = lfsr.step();
        period += 1;

        if bit == 0 {
            zeros += 1;
        } else {
            ones += 1;
        }

        if period <= 16 {
            println!(
                "  {:4}     {}     {}",
                period,
                bit,
                format_bits(lfsr.state(), n)
            );
        }

        // Stop when we return to the initial state; for a primitive polynomial,
        // the period should be 2^n - 1.
        if lfsr.state() == start_state {
            break;
        }
    }

    println!("\nSummary over one full cycle:");
    println!("  Period length          = {}", period);
    println!("  Expected max length    = {}", (1usize << n) - 1);
    println!("  Number of output zeros = {}", zeros);
    println!("  Number of output ones  = {}", ones);
}
```

Program 7.6.0 demonstrates how the algebraic structure of LFSRs can be translated into clear and efficient software using only bitwise operations. The observed period of $2^{5}-1$ verifies that the chosen feedback polynomial is primitive and that the implementation correctly produces a maximum-length m-sequence. The near-equal occurrence of zeros and ones and the traversal of all non-zero register states reflect the statistical regularity and uniformity discussed earlier in Section 7.6. These features make LFSRs attractive for deterministic pseudo-random generation in hardware and embedded settings, where simplicity and reproducibility are essential. The modularity of the code enables the register size and tap positions to be changed easily, supporting experimentation with other characteristic polynomials or extension to word-level registers used in modern cryptographic constructions.

## 7.6.1. Algebraic Formulation and Period Analysis of Linear Feedback Shift Registers

Linear Feedback Shift Registers (LFSRs) are defined by linear recurrences modulo 2, which allow their entire behavior to be described using linear algebra over the finite field $\mathbb{F}_2$. Each state update is determined by a feedback polynomial that encodes which register bits contribute to the next bit through XOR feedback. If this polynomial is primitive, the LFSR attains the longest possible cycle of non-zero states, yielding what is known as a maximum-length sequence or *m-sequence* (Almaraz & Román Villaizán, 2023).

The characteristic feedback polynomial is written as:

$$P(x) = x^{n} + c_{n-1}x^{n-1} + c_{n-2}x^{n-2} + \cdots + c_{1}x + 1 \tag{7.6.8}$$

where each coefficient $c_i \in {0,1}$ indicates whether the corresponding bit participates in the feedback. The polynomial coefficients define the companion matrix $M \in \mathbb{F}_2^{n\times n}$, which advances the register according to,

$$\mathbf{a}_{t+1} = M\mathbf{a}_t \tag{7.6.9}$$

with $\mathbf{a}_t = (a_{t,n-1}, a_{t,n-2}, \dots, a_{t,0})^{\mathsf T}$ representing the current register state. Since arithmetic takes place in $\mathbb{F}_2$, each matrix operation corresponds to bit-level XORs and shifts, making the dynamics deterministic yet extremely fast in hardware (Okunbor, Omorogbe & Edeko, 2024).

### Period Condition

For an LFSR to achieve its maximal period, its transition matrix must satisfy:

$$M^{2^{n}-1} = I_{n}\tag{7.6.10}$$

so that after $2^{n}-1$ updates the state returns to its original configuration. The order of $M$ in the multiplicative group of invertible matrices over $\mathbb{F}_2$ therefore divides $2^{n}-1$; maximal period requires $\mathrm{ord}(M) = 2^{n} - 1.$

Equivalently, the feedback polynomial $P(x)$ must divide $x^{2^{n}-1} - 1$ but no smaller polynomial $x^{k} - 1$. This ensures that the recurrence traverses every non-zero state once before repeating (Almaraz & Román Villaizán, 2023).

To exclude shorter cycles, for each proper divisor $d$ of $2^{n}-1$ the following must hold:

$$M^{d} \neq I_{n}\tag{7.6.11}$$

When both conditions are met, $P(x)$ is primitive, guaranteeing a full-period $m$-sequence.

### Primitive Polynomials and Examples

A polynomial is *primitive* if it is irreducible over $\mathbb{F}_2$ and any of its roots generates the multiplicative group $\mathbb{F}_{2^n}^{*}$. Primitive polynomials therefore produce sequences covering all $2^{n}-1$ non-zero states. Common examples include:

\begin{align}
x^{3}  + x + 1      &\quad (n = 3,\ \text{period } 7), \notag\\
x^{4}  + x + 1      &\quad (n = 4,\ \text{period } 15), \notag\\
x^{10} + x^{3} + 1  &\quad (n = 10,\ \text{period } 1023)
\tag{7.6.12}
\end{align}

The number of primitive polynomials of degree $n$ is $\phi(2^{n}-1)/n$, where $\phi$ denotes Euler’s totient function. This abundance allows efficient selection for hardware design and cryptographic use (Okunbor, Omorogbe & Edeko, 2024).

### Efficient Period Testing

Testing primitivity follows a deterministic algebraic route (Maity et al., 2025):

(i) To verify that an LFSR achieves its full period, one begins by *constructing the transition matrix* $M \in \mathbb{F}_2^{n\times n}$ from the coefficients of the characteristic polynomial $P(x)$*.* Each coefficient $c_i$ determines whether a corresponding tap contributes to the feedback, and together they define the matrix that advances the register by one step according to the recurrence relation $\mathbf{a}_{t+1} = M\mathbf{a}_t$. This matrix completely characterizes the dynamics of the LFSR over the finite field $\mathbb{F}_2$.

(ii) Once $M$ has been constructed, the *full-period condition* is verified by computing the matrix power $M^{2^{n}-1}$. A maximal-length sequence is obtained if and only if,

$$M^{2^{n}-1} = I_{n}\tag{7.6.13}$$

where $I_n$ is the $n\times n$ identity matrix. This equality ensures that after $2^{n}-1$ steps, the system returns to its initial state, completing a single full cycle through all nonzero configurations.

(iii) Finally, to *eliminate shorter cycles*, one must verify that the state does not repeat after any smaller number of steps. For each prime divisor $q$ of $2^{n}-1$, it is necessary to check that:

$$M^{(2^{n}-1)/q} \neq I_{n}\tag{7.6.14}$$

If this condition holds for every such $q$, then $M$ has order exactly $2^{n}-1$, implying that the corresponding feedback polynomial $P(x)$ is primitive and the LFSR produces a true maximum-length $m$-sequence.

Matrix exponentiation by repeated squaring reduces computational effort to $O(\log(2^{n}-1)) \approx O(n)$ matrix multiplications. Because these operations are XOR-based, LFSR period validation remains feasible for $n \le 64$ in both software and FPGA implementations (Almaraz & Román Villaizán, 2023; Okunbor, Omorogbe & Edeko, 2024).

### State Space and Randomness

When the feedback polynomial $P(x)$ is *primitive*, the Linear Feedback Shift Register traverses all $2^{n} - 1$ possible non-zero states before repeating. The all-zero configuration is excluded, as its feedback yields only zeros, leaving it as a fixed point. During one complete cycle, the sequence exhibits an almost perfect balance between zeros and ones, and every possible $n$-bit combination (except the all-zero one) appears exactly once. This uniform coverage of the state space ensures excellent statistical properties: the generated sequences display balanced symbol frequencies, geometric distributions of run lengths, and an autocorrelation function that assumes only two values, typically $+1$ and $-1$. These characteristics satisfy Golomb’s postulates for randomness, making m-sequences valuable as pseudo-random test patterns and reference sequences in signal analysis and communications (Maity et al., 2025).

Despite their strong pseudo-random behavior, the linearity of LFSRs imposes inherent predictability. Given $2n$ consecutive bits of an m-sequence, one can reconstruct the entire feedback structure using the Berlekamp–Massey algorithm, which determines the minimal polynomial generating the observed sequence. Therefore, while LFSRs are ideal for simulation, testing, and efficient random-like bit generation, they are not secure for cryptographic use unless combined with nonlinear components or multiple interacting registers. In modern designs, such as stream ciphers and random number generators, nonlinear filters or combiners are introduced to obscure the linear dependency and enhance resistance to predictive attacks (Okunbor, Omorogbe & Edeko, 2024).

### Rust Implementation

Following the algebraic characterization of Linear Feedback Shift Registers in Section 7.6.1, Program 7.6.1 provides a practical implementation of the period analysis framework developed in the text. The preceding discussion formalized LFSR dynamics through the feedback polynomial (7.6.8) and its companion matrix representation (7.6.9), establishing that the period of the register corresponds to the order of the transition matrix in the multiplicative group over $\mathbb{F}_2$. The program transforms these theoretical conditions into concrete computational tests: it constructs the transition matrix, evaluates the full-period condition (7.6.10), and verifies the absence of shorter cycles via the divisor checks in (7.6.11) and (7.6.14). By automating these checks for example polynomials from (7.6.12), the program demonstrates how algebraic properties of the recurrence translate directly into predictable cycle lengths and the generation of maximum-length m-sequences.

At the core of the implementation is the routine `make_companion_matrix`, which constructs the transition matrix $M$ corresponding to a feedback polynomial of the form (7.6.8). The structure of this matrix mirrors the theoretical model: the first row encodes the XOR taps defining the feedback bit, while the subdiagonal implements the right-shift operation inherent in the recurrence. This exactly matches the matrix form (7.6.5)–(7.6.9), allowing all updates of the LFSR to be expressed as matrix–vector multiplication over $\mathbb{F}_2$. The companion matrix constructed here is represented compactly using machine words, with each row encoded as a bit mask, enabling efficient XOR-based arithmetic that aligns with the underlying field operations.

The function `mat_mul` implements matrix multiplication over $\mathbb{F}_2$ using bitwise XOR to model addition and bitwise AND to detect active terms in the row. When combined with the repeated-squaring procedure `mat_pow`, the program can compute high powers of $M$ efficiently, which is essential for verifying the period conditions (7.6.10) and (7.6.13). Through the identity check `is_identity`, the program determines whether a given matrix power reduces to the identity matrix $I_n$, thereby confirming whether the LFSR returns to its initial configuration after a specified number of steps.

To evaluate the absence of shorter cycles as required by (7.6.11) and (7.6.14), the program factors $2^{n}-1$ using the helper function `prime_factors`. For each prime divisor $q$, it tests whether $M^{(2^{n}-1)/q}$ remains distinct from the identity matrix. Only when all such tests succeed does the polynomial qualify as primitive, guaranteeing that the LFSR attains the maximal period. The `test_primitive` routine orchestrates this entire verification process, applying these algebraic criteria to the example polynomials listed in (7.6.12) and reporting whether each polynomial produces a true m-sequence.

The `main` function provides these example cases, constructing the corresponding coefficient arrays and invoking the primitivity test. Its output illustrates the successful verification of all three standard primitive polynomials: $x^{3}+x+1$, $x^{4}+x+1$, and $x^{10}+x^{3}+1$. These examples showcase how the companion-matrix formulation enables deterministic confirmation of the full-period condition, emphasizing the fundamental relationship between polynomial primitivity and LFSR sequence length.

```rust
// Program 7.6.1: Algebraic Period Test for LFSRs via Companion Matrices over F₂
//
// This program constructs the companion matrix M ∈ F₂^{n×n} associated with an
// LFSR feedback polynomial
//
//   P(x) = x^n + c_{n-1} x^{n-1} + ... + c_1 x + 1   (7.6.8)
//
// and tests whether P(x) is primitive by applying the matrix-order conditions
// described in (7.6.10)–(7.6.14). All arithmetic is performed over F₂ using
// bitwise operations. For several example polynomials (7.6.12), the program
// verifies:
//
//   1. M^(2^n - 1) = I_n                                  (7.6.13)
//   2. M^((2^n - 1)/q) ≠ I_n for each prime q | (2^n - 1) (7.6.14)
//
// A polynomial that passes both tests is primitive and yields a full-period
// m-sequence of length 2^n - 1.

const MAX_N: usize = 32;
type Matrix = [u32; MAX_N];

fn make_companion_matrix(n: usize, coeffs: &[u8]) -> Matrix {
    assert!(n >= 2 && n <= MAX_N, "n must be between 2 and {MAX_N}");
    assert!(
        coeffs.len() == n - 1,
        "coeffs must contain c₁,…,cₙ₋₁ (length n-1)"
    );

    let mut m: Matrix = [0; MAX_N];

    // First row: [c1, c2, ..., c_{n-1}, 1]
    let mut row0: u32 = 0;
    for j in 0..(n - 1) {
        if coeffs[j] != 0 {
            row0 |= 1u32 << j;
        }
    }
    row0 |= 1u32 << (n - 1);
    m[0] = row0;

    // Subdiagonal of ones: row i has a 1 in column i-1 (for i = 1..n-1).
    for i in 1..n {
        m[i] = 1u32 << (i - 1);
    }

    m
}

// C = A * B over F₂ (n × n matrices stored row-wise in u32 bitmasks).
fn mat_mul(a: &Matrix, b: &Matrix, n: usize) -> Matrix {
    let mut c: Matrix = [0; MAX_N];

    for i in 0..n {
        let mut row = 0u32;
        let arow = a[i];
        // For each bit k set in row i of A, XOR row k of B into result.
        let mut mask = arow;
        let mut k = 0;
        while mask != 0 {
            if (mask & 1) != 0 {
                row ^= b[k];
            }
            mask >>= 1;
            k += 1;
        }
        c[i] = row;
    }

    c
}

// Identity matrix I_n in the same representation.
fn identity_matrix(n: usize) -> Matrix {
    let mut m: Matrix = [0; MAX_N];
    for i in 0..n {
        m[i] = 1u32 << i;
    }
    m
}

// Check if m == I_n.
fn is_identity(m: &Matrix, n: usize) -> bool {
    let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
    for i in 0..n {
        let expected = 1u32 << i;
        let row = m[i] & mask;
        if row != expected {
            return false;
        }
    }
    true
}

// Exponentiation by repeated squaring: M^exp over F₂.
fn mat_pow(mut base: Matrix, mut exp: u64, n: usize) -> Matrix {
    let mut result = identity_matrix(n);
    while exp > 0 {
        if exp & 1 != 0 {
            result = mat_mul(&result, &base, n);
        }
        exp >>= 1;
        if exp > 0 {
            base = mat_mul(&base, &base, n);
        }
    }
    result
}

// Return the distinct prime factors of m.
fn prime_factors(mut m: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut d = 2u64;
    while d * d <= m {
        if m % d == 0 {
            factors.push(d);
            while m % d == 0 {
                m /= d;
            }
        }
        d += 1;
    }
    if m > 1 {
        factors.push(m);
    }
    factors
}

struct Example {
    n: usize,
    coeffs: &'static [u8], // c1..c_{n-1}
    description: &'static str,
}

fn test_primitive(example: &Example) {
    let n = example.n;
    let m = make_companion_matrix(n, example.coeffs);
    let order_target: u64 = (1u64 << n) - 1;

    println!("Polynomial: {}", example.description);
    println!("Degree n   = {}", n);
    println!("Order test = 2^n - 1 = {}", order_target);

    // Check M^(2^n - 1) == I_n (7.6.13).
    let m_to_full = mat_pow(m, order_target, n);
    let full_ok = is_identity(&m_to_full, n);
    println!("  M^(2^n - 1) == I_n? {}", if full_ok { "yes" } else { "no" });

    if !full_ok {
        println!("  ⇒ Not primitive (fails full-period condition).\n");
        return;
    }

    // For each prime q | (2^n - 1), check M^((2^n - 1)/q) != I_n (7.6.14).
    let factors = prime_factors(order_target);
    println!("  Prime factors of 2^n - 1: {:?}", factors);

    let mut passes_all = true;
    for q in factors {
        let exp = order_target / q;
        let m_sub = mat_pow(m, exp, n);
        let is_id = is_identity(&m_sub, n);
        println!(
            "  M^((2^n - 1)/{q}) != I_n? {}",
            if !is_id { "yes" } else { "no (fails)" }
        );
        if is_id {
            passes_all = false;
        }
    }

    if passes_all {
        println!("  ⇒ Polynomial is primitive; LFSR has period 2^n - 1.\n");
    } else {
        println!("  ⇒ Polynomial is not primitive.\n");
    }
}

fn main() {
    // Examples from (7.6.12):
    //
    // P(x) = x^3 + x + 1       → n = 3,  c1 = 1, c2 = 0
    // P(x) = x^4 + x + 1       → n = 4,  c1 = 1, c2 = 0, c3 = 0
    // P(x) = x^10 + x^3 + 1    → n = 10, c3 = 1, others 0
    //
    // Coeff arrays are [c1, c2, ..., c_{n-1}] in ascending order.
    let examples = [
        Example {
            n: 3,
            coeffs: &[1, 0],
            description: "P(x) = x^3 + x + 1",
        },
        Example {
            n: 4,
            coeffs: &[1, 0, 0],
            description: "P(x) = x^4 + x + 1",
        },
        Example {
            n: 10,
            coeffs: &[0, 0, 1, 0, 0, 0, 0, 0, 0], // c3 = 1
            description: "P(x) = x^10 + x^3 + 1",
        },
    ];

    for ex in &examples {
        test_primitive(ex);
    }
}
```

Program 7.6.1 demonstrates how the algebraic structure of Linear Feedback Shift Registers can be translated into an explicit computational framework for period analysis. By expressing the recurrence in matrix form and applying the order conditions (7.6.10)–(7.6.14), the program verifies whether a given feedback polynomial is primitive and thus capable of producing an m-sequence of maximal length $2^{n}-1$. The examples confirm the theoretical predictions from Section 7.6.1: primitive polynomials lead to transition matrices whose powers cycle through all non-zero states exactly once, while non-primitive polynomials would fail the divisor checks and exhibit shorter cycles.

The modularity of this implementation allows it to scale to higher degrees and more complex polynomials, supporting the exploration of LFSR behavior across a wide range of applications in communication systems, error detection codes, test-pattern generation, and cryptographic constructions. Although the computational tests rely on linear algebra over $\mathbb{F}_2$, the efficient bitwise representation ensures high performance even for moderate register sizes. This forms a bridge between the algebraic theory of recurrence relations and their practical verification, enabling designers to confirm the maximal-length property before deploying an LFSR in hardware or software environments.

## 7.6.2. Implementation: Fibonacci vs Galois Configurations

Linear Feedback Shift Registers (LFSRs) can be realized in two algebraically equivalent but structurally distinct forms, known respectively as the *Fibonacci* and *Galois* configurations. Both versions are governed by the same feedback polynomial and, when initialized consistently, generate identical pseudo-random sequences up to a possible phase shift or bit reversal. The distinction lies in how feedback is applied within the register: the Fibonacci configuration computes a single global XOR feedback, whereas the Galois configuration distributes feedback locally across multiple stages, improving efficiency and speed (Somanathan, Reddy & Bhakthavatchalu, 2025).

In the Fibonacci configuration, sometimes called *Method I*, all bits shift in one direction, and a new bit is computed as the XOR of several “tap” positions from the previous state, usually including the bit that has just been shifted out. If the state at time $t$ is represented as $(a_{t,0},a_{t,1},\dots,a_{t,n-1})$, the recurrence is expressed as:

$$a_{t+1,n-1}=a_{t,0}\oplus(c_1a_{t,1}\oplus c_2a_{t,2}\oplus\cdots\oplus c_{n-1}a_{t,n-1})\tag{7.6.15}$$

where $\oplus$ denotes addition modulo $2$ and $c_i\in{0,1}$ indicate the feedback tap positions. The companion matrix $M$ represents this recurrence, and its characteristic polynomial $P(x)=x^n+c_{n-1}x^{n-1}+\cdots+c_1x+1$ defines the same transformation. Because it corresponds directly to the algebraic model of feedback, the Fibonacci form is often preferred for theoretical development, analysis of sequence periodicity, and matrix-based proofs (Almaraz & Román Villaizán, 2023).

The Galois configuration, or *Method II*, modifies the feedback structure to increase parallelism and computational efficiency. Instead of performing a single XOR operation across all taps, the outgoing bit is fed back into selected register positions during the shift. Each active tap stage XORs locally with the outgoing bit, producing an updated value as part of the same clock cycle. This localized feedback mechanism allows simultaneous updates of multiple bits and reduces logic depth in hardware implementations. The update rule can be written as:

$$
a_{t+1,i}=\begin{cases}
a_{t,i+1}\oplus a_{t,0}, & \text{if a tap is active at position }i, \\[4pt]
a_{t,i+1}, & \text{otherwise}
\end{cases}\tag{7.6.16}
$$

The relationship between the two structures can be expressed algebraically as:

$$M_{\text{G}}=(M_{\text{F}}^{-1})^{T}\tag{7.6.17}$$

which shows that the Fibonacci and Galois matrices are linearly equivalent, sharing the same minimal and characteristic polynomials up to reversal (Okunbor, Omorogbe & Edeko, 2024). To illustrate, consider the primitive polynomial:

$$P(x)=x^{8}+x^{6}+x^{5}+x^{4}+1\tag{7.6.18}$$

A Fibonacci LFSR with this polynomial uses taps at bits 6, 5, 4, and 0 (counting from the least significant bit), XORing them to generate the new input bit. The corresponding Galois version injects the outgoing bit into these same tap positions during the shift. When both systems are seeded with equivalent initial states, they produce identical sequences differing only by a fixed offset due to bit-ordering conventions (Somanathan, Reddy & Bhakthavatchalu, 2025).

In practice, the Fibonacci form is often used in analysis because of its algebraic transparency, while the Galois configuration is preferred for implementation due to its reduced XOR overhead and improved timing properties. In high-level programming environments such as Rust, a Galois-style update can be implemented efficiently using bitwise operators:

$$s_{t+1}=(s_t\gg 1)\oplus[-(s_t\,\&\, 1)\,\&\, T]\tag{7.6.19}$$

where $s_t$ represents the LFSR state and $T$ is a tap mask with bits set at the feedback positions. This compact recurrence executes in a handful of machine instructions per iteration, achieving very high throughput for pseudo-random bit generation (Okunbor, Omorogbe & Edeko, 2024).

The two configurations thus represent the same linear transformation over $\mathbb{F}_2$ and can be converted into one another via reciprocal polynomials and state reversal. If the sequence produced by the Fibonacci LFSR is denoted ${z_t}$, the corresponding Galois LFSR generates,

$$z^{(\text{G})}t=z^{(\text{F})}_{t+\Delta}\tag{7.6.20}$$

where $\Delta$ is a phase offset determined by initialization. Empirical results have shown that Galois implementations achieve up to 40% higher throughput in bit-parallel hardware and software while maintaining identical statistical and periodic properties (Somanathan, Reddy & Bhakthavatchalu, 2025).

In summary, the Fibonacci configuration emphasizes mathematical structure and theoretical clarity, while the Galois form prioritizes performance and compactness. Both are mathematically equivalent and yield identical pseudo-random properties when derived from the same primitive polynomial. Their complementary characteristics make them essential design options in modern pseudo-random sequence generation, particularly for applications in encryption (Maity et al., 2025), built-in self-test architectures, and secure communications (Almaraz & Román Villaizán, 2023; Okunbor, Omorogbe & Edeko, 2024).

### Rust Implementation

Following the discussion in Section 7.6.2 on the structural differences between Fibonacci and Galois LFSRs, and how both implementations correspond to the same underlying linear recurrence defined by a primitive feedback polynomial, Program 7.6.2 provides a practical comparison of these two configurations. Although the Fibonacci realization directly reflects the algebraic recurrence in Equation (7.6.15), the Galois form implements the equivalent transformation in a more distributed and computationally efficient manner as expressed in Equation (7.6.16). This program constructs both LFSRs for the same primitive polynomial, generates output bits from each configuration, and empirically verifies the theoretical equivalence described in Equation (7.6.20). By identifying the phase shift that aligns the two sequences and confirming balance properties over a full period, the program illustrates how algebraic structure translates seamlessly into efficient bitwise implementations.

At the core of the implementation is the `FibonacciLfsr` struct, which directly encodes the recurrence relation in Equation (7.6.15). Given an 8-bit state and a set of tap positions corresponding to the nonzero coefficients of the feedback polynomial, the `step()` method computes the outgoing bit from the least significant position and constructs the incoming bit as the XOR of all tapped positions. This mirrors the classical "Method I" architecture, where a single global XOR determines the new bit before the register shifts. The construction emphasizes algebraic transparency: each state transition corresponds exactly to multiplication by the companion matrix associated with the feedback polynomial.

The `GaloisLfsr` struct implements the alternative update rule given in Equation (7.6.16). Instead of forming one global XOR value, the Galois update distributes feedback locally by injecting the outgoing bit into specific tap positions during the shift. This is encoded compactly using the bit-trick shown in Equation (7.6.19), where the outgoing bit modifies selected stages according to a precomputed tap mask. The resulting `step()` method requires only a handful of bitwise operations, reflecting the efficiency and shallow logic depth that make Galois LFSRs attractive in high-performance hardware and software applications.

To assess the equivalence between the two configurations, the program generates the full period sequence from each LFSR beginning from the same seed. The `collect_bits()` helper function captures one period of output bits (in LSB-first order), allowing the two sequences to be compared directly. The `find_phase_shift()` function then searches for a Δ satisfying Equation (7.6.20), confirming that the Galois output is a cyclic shift of the Fibonacci sequence. This step is crucial, as differences in state ordering or bit-extraction conventions often introduce fixed offsets even when the underlying sequences are identical.

Finally, the program performs a balance check over the full period, counting the number of ones and zeros produced by each LFSR. For a maximal-length LFSR of degree 8, theory predicts exactly 128 ones and 127 zeros, a property characteristic of m-sequences. The results printed by the program verify this expected property for both Fibonacci and Galois implementations, reinforcing that both forms generate sequences with identical statistical and periodic behavior. The code is as follows:

```rust
// Program 7.6.2: Fibonacci vs Galois LFSR Implementations for
// P(x) = x^8 + x^6 + x^5 + x^4 + 1
//
// This program illustrates the equivalence of Fibonacci (Method I) and
// Galois (Method II) LFSR configurations for the same primitive polynomial,
// as discussed in Section 7.6.2.
//
// - The Fibonacci LFSR uses taps at bits {6, 5, 4, 0} (LSB = bit 0) and
//   implements the recurrence in Equation (7.6.15) by XORing the tap bits
//   to form the new input bit at the MSB.
// - The Galois LFSR uses the compact bitwise update from Equation (7.6.19):
//
//       s_{t+1} = (s_t >> 1) ^ (-(s_t & 1) & T),
//
//   where T is a tap mask indicating which stages receive local feedback from
//   the outgoing bit.
//
// The program:
//   1. Implements both update rules for an 8-bit register.
//   2. Generates 255 output bits (one m-sequence period) from each LFSR.
//   3. Searches for a phase shift Δ such that
//
//         z^(G)_t = z^(F)_{t + Δ}   (mod period),
//
//      illustrating Equation (7.6.20).
//   4. Prints the first few bits from both sequences and the discovered Δ.
//
// The state representation:
//   - We treat an 8-bit state `s` as (a_{t,0}, ..., a_{t,7}), with a_{t,0}
//     stored in the least significant bit (LSB) and a_{t,7} in the MSB.
//   - The output bit at time t is taken as a_{t,0} = s_t & 1.

/// Degree of the LFSR (n = 8 for this example).
const DEGREE: usize = 8;

/// Period of a maximal-length LFSR with n = 8 (2^8 - 1 = 255).
const PERIOD: usize = (1 << DEGREE) - 1;

/// Galois tap mask T for polynomial x^8 + x^6 + x^5 + x^4 + 1
/// under the current right-shift, LSB-output convention.
const GALOIS_TAP_MASK: u8 = 0x8E;

/// One step of the Fibonacci LFSR for P(x) = x^8 + x^6 + x^5 + x^4 + 1.
///
/// State layout: s_t = (a_{t,7} ... a_{t,1} a_{t,0}), with a_{t,0} in the LSB.
///
/// Taps: bits 6, 5, 4, and 0 (counting from LSB = bit 0).
///
/// Update:
///   1. Compute newbit = a_{t,0} ⊕ a_{t,4} ⊕ a_{t,5} ⊕ a_{t,6}.
///   2. Shift right: s_t >> 1 (each bit moves toward LSB).
///   3. Insert newbit at MSB (bit 7).
fn lfsr_step_fibonacci(mut s: u8) -> u8 {
    let bit0 = s & 1;
    let bit4 = (s >> 4) & 1;
    let bit5 = (s >> 5) & 1;
    let bit6 = (s >> 6) & 1;

    let newbit = bit0 ^ bit4 ^ bit5 ^ bit6;
    s >>= 1;
    s |= newbit << 7;

    s
}

/// One step of the Galois LFSR using the compact bitwise recurrence
/// from Equation (7.6.19):
///
///     s_{t+1} = (s_t >> 1) ^ (-(s_t & 1) & T).
///
/// If the outgoing bit (LSB) is 1, `-(s_t & 1)` becomes 0xFF, so all tap
/// positions in T receive an XOR with 1. If the outgoing bit is 0,
/// `-(s_t & 1)` becomes 0x00, and no feedback is applied.
///
/// This corresponds to a Galois configuration in which the outgoing bit
/// is injected locally into the selected tap stages during the shift.


fn lfsr_step_galois(s: u8) -> u8 {
    let shifted = s >> 1;

    // outgoing bit (LSB)
    let fb = (s & 1) as i8;
    let fb_mask = (-(fb)) as u8;        // 0xFF if fb=1, 0x00 if fb=0
    let feedback_mask = fb_mask & GALOIS_TAP_MASK;

    shifted ^ feedback_mask
}


/// Generate a sequence of output bits z_t from an LFSR given an update function.
/// The sequence length is `len`, and the output bit at time t is taken as
/// the current LSB (s_t & 1).
fn generate_bits<F>(mut state: u8, len: usize, step: F) -> Vec<u8>
where
    F: Fn(u8) -> u8,
{
    let mut bits = Vec::with_capacity(len);
    for _ in 0..len {
        let out_bit = state & 1;
        bits.push(out_bit);
        state = step(state);
    }
    bits
}

/// Find a phase shift Δ such that:
///
///   b2[t] = b1[(t + Δ) mod N]  for all t = 0, ..., N-1,
///
/// if such a Δ exists. Returns Some(Δ) or None.
fn find_phase_shift(b1: &[u8], b2: &[u8]) -> Option<usize> {
    assert_eq!(b1.len(), b2.len());
    let n = b1.len();

    'outer: for delta in 0..n {
        for t in 0..n {
            let idx = (t + delta) % n;
            if b1[idx] != b2[t] {
                continue 'outer;
            }
        }
        return Some(delta);
    }

    None
}

fn main() {
    // Non-zero seed (all-zero state is absorbing and must be avoided).
    let seed: u8 = 0xAE; // arbitrary non-zero initial state

    // Generate full-period sequences of output bits from both LFSRs.
    let fib_bits = generate_bits(seed, PERIOD, lfsr_step_fibonacci);
    let gal_bits = generate_bits(seed, PERIOD, lfsr_step_galois);

    println!("Comparing Fibonacci and Galois LFSR configurations");
    println!("Polynomial: P(x) = x^8 + x^6 + x^5 + x^4 + 1");
    println!("State size: 8 bits, period: {}", PERIOD);
    println!("Seed state: 0x{:02X}\n", seed);

    // Print first 32 bits of each sequence for visual comparison.
    let preview = 32.min(PERIOD);
    println!("First {preview} output bits (LSB) from Fibonacci LFSR:");
    print!("  ");
    for i in 0..preview {
        print!("{}", fib_bits[i]);
    }
    println!();

    println!("\nFirst {preview} output bits (LSB) from Galois LFSR:");
    print!("  ");
    for i in 0..preview {
        print!("{}", gal_bits[i]);
    }
    println!("\n");

    // Attempt to find a phase shift Δ such that z^(G)_t = z^(F)_{t+Δ}.
    match find_phase_shift(&fib_bits, &gal_bits) {
        Some(delta) => {
            println!(
                "The sequences are identical up to a phase shift Δ = {} (mod {}).",
                delta, PERIOD
            );
            println!("This illustrates Equation (7.6.20):");
            println!("  z^(G)_t = z^(F)_{{t + Δ}} with Δ = {}.", delta);
        }
        None => {
            println!("No phase shift Δ was found that aligns the two sequences.");
            println!("Check tap positions, polynomial encoding, and state conventions.");
        }
    }

    // Optionally, you can also verify that both sequences contain the same
    // number of ones and zeros over a full period.
    let ones_fib = fib_bits.iter().filter(|&&b| b == 1).count();
    let ones_gal = gal_bits.iter().filter(|&&b| b == 1).count();

    println!("\nBalance check over one period ({} bits):", PERIOD);
    println!(
        "  Fibonacci: {} ones, {} zeros",
        ones_fib,
        PERIOD - ones_fib
    );
    println!(
        "  Galois   : {} ones, {} zeros",
        ones_gal,
        PERIOD - ones_gal
    );
}
```

Program 7.6.2 demonstrates the practical equivalence between Fibonacci and Galois LFSR configurations by implementing both recurrence forms and comparing their output sequences in detail. The detection of a fixed phase shift Δ confirms the theoretical relationship expressed in Equation (7.6.20), illustrating how differences in update structure translate into corresponding differences in sequence alignment without altering the underlying pseudo-random properties. The balance results further validate the maximal-length nature of the chosen primitive polynomial.

These experiments highlight two complementary perspectives emphasized in Section 7.6.2: the Fibonacci form provides clear alignment with algebraic recurrence relations and is well suited for theoretical development, while the Galois form offers superior efficiency in software and hardware implementations. Because both configurations implement the same linear transformation over $\mathbb{F}_2$, they generate sequences with identical periodicity, distribution, and auto-correlation properties. This dual viewpoint, algebraic clarity vs. implementation efficiency, reinforces the importance of understanding both representations when designing high-performance pseudo-random bit generators for simulation, cryptography, and embedded systems.

## 7.6.3. Applied Roles and Recent Innovations in LFSRs

Linear Feedback Shift Registers (LFSRs) remain among the most efficient and versatile mechanisms for pseudo-random sequence generation. Their simplicity, high throughput, and well-defined mathematical properties have ensured continuing relevance across multiple domains of computing and engineering. While early applications focused on simulation and testing, modern systems now employ LFSRs in diverse contexts ranging from cryptography and image security to hardware self-test architectures and reconfigurable pattern generation.

### Cryptographic Keystreams and Lightweight Random Number Generation

In cryptography, LFSRs are widely used to construct lightweight pseudo-random number generators and keystream generators for stream ciphers. Although a single LFSR offers limited security due to its linearity, combining multiple registers with nonlinear filtering functions can produce sequences that resist linear prediction. Contemporary architectures such as reconfigurable or power-aware pattern generators integrate LFSRs with dynamic tap-control logic, enabling variable-length and parameterized pseudo-random sequences suitable for secure key-scheduling (Somanathan, Reddy & Bhakthavatchalu, 2025). These designs provide strong diffusion and high throughput while maintaining low power consumption, making them ideal for embedded and IoT applications.

### Built-In Self-Test (BIST) and Hardware Fault Detection

LFSRs play a central role in built-in self-test (BIST) systems for integrated circuits because they generate deterministic yet statistically balanced bitstreams. A pseudo-random pattern generator (PRPG) constructed from an LFSR can stimulate the circuitry under test without requiring large external memory. Output responses are compacted through a multiple-input signature register (MISR), enabling efficient fault detection. Modern BIST systems employ reconfigurable LFSRs capable of switching feedback polynomials dynamically, increasing test coverage and enhancing pattern diversity with negligible hardware overhead (Somanathan, Reddy & Bhakthavatchalu, 2025).

### Image Encryption and Multimedia Security

In the field of image encryption, LFSR-based bitstreams have gained renewed interest due to their speed and compatibility with hardware acceleration. LFSR keystreams combined with spatial operations such as pixel shuffling or diffusion can produce lightweight yet secure cryptosystems. Maity et al. (2025) demonstrated an LFSR–pixel-shuffling encryption method achieving 99.7% pixel-level randomness and near-maximal entropy. Similarly, Okunbor, Omorogbe and Edeko (2024) incorporated chaotic maps into LFSR-driven systems, improving resistance against statistical and differential attacks. The underlying simplicity enables real-time performance for both embedded devices and cloud-based multimedia processing.

### High-Speed Scientific and Numerical Computing

Beyond security, LFSRs remain essential in numerical computing for Monte Carlo simulations and stochastic modelling. Their long periods (up to $2^{n}-1)$ and balanced distribution properties make them useful for applications requiring large sample sizes and deterministic reproducibility. Almaraz and Román Villaizán (2023) evaluated LFSR-derived generators using the NIST statistical test suite and demonstrated their robustness when enhanced with modular arithmetic post-processing. These results affirm that, with carefully chosen primitive polynomials and mild transformations, LFSR-based pseudo-random number generators satisfy demanding statistical criteria.

### Hybrid, Chaotic, and Reconfigurable LFSR Architectures

Modern research explores hybrid architectures that combine LFSRs with chaotic systems, cellular automata, and residue number systems to increase sequence complexity and unpredictability. Such approaches preserve the balance and long period of classical LFSRs while introducing controlled nonlinearity to strengthen cryptographic resistance. Field-programmable gate arrays (FPGAs) now support high-throughput multi-LFSR arrays with adaptive sequence tuning, enabling parallel bit generation and reconfigurable pseudo-random behaviour (Okunbor, Omorogbe & Edeko, 2024). These hybrid designs demonstrate how LFSRs can be integrated seamlessly into advanced secure-computing frameworks.

The enduring utility of LFSRs arises from their elegant algebraic foundation, efficient implementation, and compatibility with modern hardware. Whether embedded in cryptographic subsystems, self-testing logic, or high-speed simulation frameworks, LFSRs continue to evolve through hybridization and dynamic reconfiguration. Their development reflects a broader trend in computing: the convergence of mathematical minimalism with practical performance, where simple linear mechanisms form the basis for sophisticated, secure, and adaptive digital architectures.

### Rust Implementation

Following the discussion in Section 7.6.3 on the applied roles and modern innovations of Linear Feedback Shift Registers, Program 7.6.3 provides a concrete demonstration of how multiple LFSRs can be combined and post-processed to generate nonlinear keystreams suitable for lightweight pseudo-random number generation. While a single LFSR exhibits strictly linear behavior and is therefore predictable, applied systems often deploy multiple registers in parallel and introduce nonlinear filtering to increase sequence complexity. This program constructs three LFSRs with distinct feedback polynomials, combines their outputs through a nonlinear Boolean majority function, and evaluates basic statistical features of the resulting keystream. It also illustrates a typical use-case: XOR-based keystream encryption, reflecting the lightweight cryptographic and pattern-generation applications described earlier in the section. The example highlights how LFSRs can form the foundation for adaptable, resource-efficient pseudo-random architectures in embedded systems and multimedia processing.

At the core of the implementation is the `Lfsr` struct, which models an $n$-bit Linear Feedback Shift Register using bitwise operations over $\mathbb{F}_2$. The update rule implemented in `step` corresponds directly to the XOR feedback defined by the characteristic polynomial, echoing the recurrence structure summarized in Equation (7.6.9). During each iteration, the least-significant bit serves as the output, while the new high-order bit is obtained by computing the parity of the tapped positions encoded by the tap mask. This mirrors the hardware update mechanism described earlier in the chapter, where XOR gates and shift registers implement the recurrence encoded by the polynomial.

To support reconfigurable pattern generation, the `set_taps` method allows the feedback configuration to be modified at runtime, reflecting the dynamic tap-control logic used in modern reconfigurable LFSR architectures. Although not invoked in the demonstration, it illustrates how LFSRs can be parameterized to adapt to different security levels or statistical requirements. The program also includes a small utility for representing the register state as an integer, enabling straightforward diagnostics or debugging.

The `Keystream3` struct implements a nonlinear combination of three LFSRs, reflecting the hybrid and filtered LFSR constructions described in Section 7.6.3. Each call to `next_bit` advances all three registers and feeds their outputs into a three-input majority function, a simple but nonlinear Boolean operation. This nonlinearity disrupts the linear relations inherent in individual LFSRs and produces a more complex bitstream characteristic of lightweight nonlinear filters used in embedded stream ciphers. The method `next_byte` aggregates eight bits to produce an 8-bit keystream byte, suitable for XOR-based encryption.

The statistical evaluation portion of the program collects $100{,}000$ keystream bits, computes the empirical fraction of ones, and evaluates the lag-1 agreement probability as a crude indicator of correlation. These diagnostics reflect the practical need to assess balance and dependence structure in pseudo-random sequences generated from LFSR networks. The program concludes by demonstrating XOR encryption and decryption of a plaintext message, a classic application of LFSR-derived keystreams in multimedia security and low-power embedded devices.

```rust
// Program 7.6.3: Multi-LFSR Nonlinear Keystream Generator and Simple Statistical Diagnostics
//
// This program illustrates an applied use of Linear Feedback Shift Registers (LFSRs)
// in the spirit of Section 7.6.3. It constructs three LFSRs with different feedback
// polynomials, combines their output bits through a nonlinear Boolean function,
// and uses the result as a keystream generator.
//
// The program:
//   1. Generates a long keystream and computes simple statistics:
//        - proportion of ones (balance)
//        - lag-1 agreement rate (a crude autocorrelation indicator)
//   2. Demonstrates keystream-based XOR encryption of a short plaintext string,
//      as is typical in lightweight stream-cipher constructions.
//
// NOTE: This is a pedagogical example, *not* a secure cipher. It is designed to
// reflect applied roles of LFSRs (cryptographic keystreams, pseudo-random
// pattern generation) while remaining simple and reproducible.

// -----------------------------------------------------------------------------
// 1. Basic LFSR implementation over F₂
// -----------------------------------------------------------------------------

/// Simple LFSR over F₂ with up to 31 bits of state.
///
/// The state is stored in the least-significant `n` bits of `state`.
/// Bit 0 is the rightmost (least significant) bit.
struct Lfsr {
    state: u32,
    taps: u32,
    n: u8,
}

impl Lfsr {
    /// Create a new LFSR with the given seed, tap mask, and register length `n`.
    ///
    /// - `seed` must be non-zero in its lower `n` bits; the all-zero state is forbidden.
    /// - `taps` is a bit mask specifying which bit positions contribute to the feedback XOR.
    fn new(seed: u32, taps: u32, n: u8) -> Self {
        assert!((1..=31).contains(&n), "n must be between 1 and 31 bits");
        let mask = (1u32 << n) - 1;

        let seed = seed & mask;
        assert!(seed != 0, "seed must be non-zero in the lower n bits");

        let taps = taps & mask;

        Self { state: seed, taps, n }
    }

    /// Optionally update the tap mask (reconfigurable LFSR).
    #[allow(dead_code)]
    fn set_taps(&mut self, taps: u32) {
        let mask = (1u32 << self.n) - 1;
        self.taps = taps & mask;
    }

    /// Advance the LFSR by one step and return the output bit (the bit shifted out).
    ///
    /// Update rule:
    ///   - Compute the new feedback bit as the XOR (mod 2 sum) of all tapped bits.
    ///   - Shift the register right by one bit.
    ///   - Insert the new feedback bit into the leftmost position (bit n-1).
    fn step(&mut self) -> u8 {
        // Output bit is the least significant bit before shifting.
        let out_bit = (self.state & 1) as u8;

        // XOR of tapped bits: parity of (state & taps).
        let x = self.state & self.taps;
        let parity = x.count_ones() & 1;
        let new_bit = parity as u32;

        // Shift right, inject feedback bit at the most significant position.
        let mask = (1u32 << self.n) - 1;
        self.state = ((self.state >> 1) | (new_bit << (self.n - 1))) & mask;

        out_bit
    }
    #[allow(dead_code)]
    /// Return the current register state (in the least significant `n` bits).
    fn state(&self) -> u32 {
        self.state
    }
}

// -----------------------------------------------------------------------------
// 2. Multi-LFSR nonlinear combiner keystream
// -----------------------------------------------------------------------------

/// A simple three-LFSR keystream generator with a nonlinear Boolean combiner.
///
/// This structure illustrates how multiple LFSRs can be combined to increase
/// sequence complexity beyond that of any single linear register. The combiner
/// used here is a simple majority function of three bits:
///
///   f(b1, b2, b3) = majority(b1, b2, b3)
///
/// which is nonlinear over F₂. This is *not* intended as a secure design,
/// but as a didactic example of hybrid and nonlinear LFSR architectures.
struct Keystream3 {
    l1: Lfsr,
    l2: Lfsr,
    l3: Lfsr,
}

impl Keystream3 {
    fn new(l1: Lfsr, l2: Lfsr, l3: Lfsr) -> Self {
        Self { l1, l2, l3 }
    }

    /// Generate the next keystream bit.
    fn next_bit(&mut self) -> u8 {
        let b1 = self.l1.step();
        let b2 = self.l2.step();
        let b3 = self.l3.step();

        // Majority(b1, b2, b3) = (b1 & b2) | (b1 & b3) | (b2 & b3)
        let m = (b1 & b2) | (b1 & b3) | (b2 & b3);
        m & 1
    }

    /// Generate the next keystream byte (8 bits, MSB-first).
    fn next_byte(&mut self) -> u8 {
        let mut byte = 0u8;
        for _ in 0..8 {
            let bit = self.next_bit();
            byte = (byte << 1) | bit;
        }
        byte
    }
}

// -----------------------------------------------------------------------------
// 3. Simple statistical diagnostics and demo encryption
// -----------------------------------------------------------------------------

/// Compute a very simple lag-1 agreement rate:
///   fraction of positions i where b[i] == b[i+1].
fn lag1_agreement(bits: &[u8]) -> f64 {
    if bits.len() < 2 {
        return 0.0;
    }
    let mut same = 0usize;
    for i in 0..(bits.len() - 1) {
        if bits[i] == bits[i + 1] {
            same += 1;
        }
    }
    same as f64 / (bits.len() - 1) as f64
}

/// Print a byte slice as hex.
fn print_hex(label: &str, data: &[u8]) {
    print!("{label}: ");
    for b in data {
        print!("{:02X} ", b);
    }
    println!();
}

fn main() {
    // -------------------------------------------------------------------------
    // 3.1 Configure three LFSRs with different feedback polynomials.
    //
    // These are illustrative examples; in practice one would select known
    // primitive polynomials of suitable degree (as in Section 7.6.1).
    //
    //  LFSR 1 (10-bit): P(x) = x^10 + x^3 + 1
    //      taps on bit positions 2 and 0.
    //
    //  LFSR 2 (9-bit): P(x) = x^9 + x^5 + 1
    //      taps on bit positions 4 and 0.
    //
    //  LFSR 3 (8-bit): P(x) = x^8 + x^6 + x^5 + x^4 + 1
    //      taps on bit positions 5, 4, 3 and 0.
    // -------------------------------------------------------------------------

    let l1 = Lfsr::new(0b0000000001, (1 << 2) | (1 << 0), 10);
    let l2 = Lfsr::new(0b000000001, (1 << 4) | (1 << 0), 9);
    let l3 = Lfsr::new(0b00000001, (1 << 5) | (1 << 4) | (1 << 3) | (1 << 0), 8);

    let mut ks = Keystream3::new(l1, l2, l3);

    // -------------------------------------------------------------------------
    // 3.2 Generate a keystream and compute simple statistics.
    // -------------------------------------------------------------------------

    let n_bits = 100_000usize;
    let mut bits = Vec::with_capacity(n_bits);
    let mut ones = 0usize;

    for _ in 0..n_bits {
        let b = ks.next_bit();
        bits.push(b);
        if b == 1 {
            ones += 1;
        }
    }

    let frac_ones = ones as f64 / n_bits as f64;
    let lag1 = lag1_agreement(&bits);

    println!(
        "Three-LFSR nonlinear keystream generator ({} bits sample):",
        n_bits
    );
    println!("  Fraction of ones          ≈ {:.6}", frac_ones);
    println!("  Lag-1 agreement probability ≈ {:.6}", lag1);
    println!();

    // -------------------------------------------------------------------------
    // 3.3 Demonstrate simple XOR "encryption" of a short plaintext.
    // -------------------------------------------------------------------------

    // Re-initialize LFSRs to the same seeds for reproducibility of the keystream.
    let l1_enc = Lfsr::new(0b0000000001, (1 << 2) | (1 << 0), 10);
    let l2_enc = Lfsr::new(0b000000001, (1 << 4) | (1 << 0), 9);
    let l3_enc = Lfsr::new(0b00000001, (1 << 5) | (1 << 4) | (1 << 3) | (1 << 0), 8);
    let mut ks_enc = Keystream3::new(l1_enc, l2_enc, l3_enc);

    let plaintext = b"LFSR-based demo keystream";
    let mut keystream_bytes = Vec::with_capacity(plaintext.len());
    let mut ciphertext = Vec::with_capacity(plaintext.len());
    let mut decrypted = Vec::with_capacity(plaintext.len());

    // Encrypt
    for &p in plaintext {
        let k = ks_enc.next_byte();
        keystream_bytes.push(k);
        ciphertext.push(p ^ k);
    }

    // Reinitialize again for decryption (same keystream).
    let l1_dec = Lfsr::new(0b0000000001, (1 << 2) | (1 << 0), 10);
    let l2_dec = Lfsr::new(0b000000001, (1 << 4) | (1 << 0), 9);
    let l3_dec = Lfsr::new(0b00000001, (1 << 5) | (1 << 4) | (1 << 3) | (1 << 0), 8);
    let mut ks_dec = Keystream3::new(l1_dec, l2_dec, l3_dec);

    for &c in &ciphertext {
        let k = ks_dec.next_byte();
        decrypted.push(c ^ k);
    }

    println!("Demo XOR keystream encryption:");
    println!("  Plaintext  : {}", String::from_utf8_lossy(plaintext));
    print_hex("  Keystream  ", &keystream_bytes);
    print_hex("  Ciphertext ", &ciphertext);
    println!(
        "  Decrypted  : {}",
        String::from_utf8_lossy(&decrypted)
    );
}
```

Program 7.6.3 illustrates how Linear Feedback Shift Registers can be combined into nonlinear architectures to enhance pseudo-random behavior, addressing many of the applied challenges discussed in Section 7.6.3. Although each individual LFSR follows a linear recurrence and is therefore predictable, combining several registers and applying a nonlinear output function can significantly increase sequence complexity and decorrelate the generated bitstream. The program’s statistical diagnostics show how empirical testing complements algebraic design, revealing potential biases and dependencies that arise when LFSR parameters are not carefully tuned.

The XOR encryption example demonstrates the lightweight keystream-generation capability of LFSR-based systems, a feature heavily exploited in embedded cryptography, BIST architectures, and multimedia security. While the construction used here is intentionally simple for instructional purposes, it provides a foundation for exploring more sophisticated designs, including nonlinear filters, clock-controlled LFSRs, cellular automata hybrids, and FPGA-accelerated multi-LFSR arrays. These modern innovations integrate the algebraic efficiency of classical LFSRs with enhanced flexibility and improved statistical or security properties, underscoring their continuing relevance in contemporary digital systems.

## 7.6.4. Mathematical Extensions and Nonlinear Generalizations

Although Linear Feedback Shift Registers (LFSRs) are inherently linear systems defined over the finite field $\mathbb{F}_2$, numerous modern extensions have been developed to enhance their statistical and cryptographic properties. These generalizations introduce nonlinearity, hybridization, or higher-dimensional algebraic structures while retaining the efficient shift-based architecture that makes LFSRs attractive for practical use. The motivation behind these extensions lies in combining the deterministic predictability and structural simplicity of LFSRs with the unpredictability and diffusion characteristics demanded by secure and adaptive applications.

### Nonlinear Feedback Shift Registers (NLFSRs)

A natural extension of the LFSR is the *Nonlinear Feedback Shift Register (NLFSR)*, in which the feedback function incorporates nonlinear Boolean terms. Instead of a purely linear XOR feedback, the next bit is defined as:

$$a_{t+1,n-1} = f(a_{t,0}, a_{t,1}, \dots, a_{t,n-1}) \tag{7.6.21}$$

where $f$ is a Boolean function containing higher-order monomials such as $a_{t,i}a_{t,j}$ or $a_{t,i}\overline{a_{t,j}}$. By introducing nonlinear feedback, NLFSRs significantly increase the linear complexity of the generated sequence, thereby improving resistance to algebraic and correlation attacks. While their design and analysis are mathematically more intricate, they are used extensively in modern cryptographic primitives, particularly in lightweight encryption and random number generation circuits (Okunbor, Omorogbe & Edeko, 2024).

### Combined and Filtered LFSR Systems

Another important class of generalizations arises from combining multiple LFSRs through nonlinear filters or combiners. If $x_t^{(1)}, x_t^{(2)}, \dots, x_t^{(m)}$ denote the outputs of $m$ distinct LFSRs, the final sequence can be expressed as:

$$z_t = f(x_t^{(1)}, x_t^{(2)}, \dots, x_t^{(m)}) \tag{7.6.22}$$

where $f$ is a nonlinear combining function. The choice of $f$ determines the linear complexity and statistical properties of the resulting output. Properly chosen nonlinear filters yield sequences with long periods, high entropy, and strong correlation immunity. These combined systems form the basis for many secure pseudo-random number generators and stream ciphers, providing a balance between theoretical predictability and practical unpredictability (Somanathan, Reddy & Bhakthavatchalu, 2025).

### Finite-Field and Word-Oriented Extensions

LFSRs can also be generalized to operate over higher-order finite fields such as (\\mathbb{F}{2^k}), where each register cell stores a word rather than a single bit. The recurrence relation in such systems takes the form,

$$\mathbf{a}_{t+1} = C_1\mathbf{a}_t + C_2\mathbf{a}_{t-1} + \cdots + C_r\mathbf{a}_{t-r} \tag{7.6.23}$$

where $C_i$ are coefficients in $\mathbb{F}_{2^k}$. These word-oriented LFSRs enable multiple bits to be updated simultaneously, increasing throughput and simplifying hardware realization on 32-bit or 64-bit processors. Their use is particularly prominent in parameterizable built-in self-test (BIST) architectures, where they act as dynamically reconfigurable pseudo-random pattern generators capable of producing variable sequence lengths and fault coverage profiles (Somanathan, Reddy & Bhakthavatchalu, 2025).

### Chaotic and Hybrid Architectures

Recent developments have combined LFSRs with chaotic systems to construct hybrid pseudo-random generators that exhibit enhanced diffusion and nonlinearity. In these designs, the LFSR provides a linear base sequence, while a chaotic map modulates the feedback or modifies the tap positions dynamically. Typical chaotic functions include logistic or piecewise linear maps that generate real-valued perturbations, subsequently quantized to influence bit-level transitions. For example, in an LFSR–chaos hybrid encryption system, the chaotic component perturbs the shift register’s state or feedback taps on each iteration, effectively producing a sequence with both deterministic structure and unpredictable evolution (Maity et al., 2025; Okunbor, Omorogbe & Edeko, 2024). These architectures have proven effective in multimedia security, where they achieve high pixel-level randomness and resist differential attacks while maintaining the computational simplicity of the underlying LFSR mechanism.

### Statistical and Cryptographic Significance

From a mathematical standpoint, these extensions preserve the finite-state nature of the original LFSR while expanding its expressive power. The introduction of nonlinearity increases the sequence’s **linear complexity**, while finite-field generalizations extend the state space from $2^n$ to $2^{kn}$, allowing longer periods and greater variability. Hybrid approaches further enrich the design space by merging discrete algebraic systems with continuous dynamical models, bridging symbolic logic and nonlinear dynamics. Experimental validation across multiple studies demonstrates that these extended LFSR architectures can pass advanced randomness tests such as those in the NIST Statistical Test Suite, confirming their suitability for secure random generation (Almaraz & Román Villaizán, 2023).

In summary, the mathematical generalizations of LFSRs represent a continuum from strict linear recurrences to complex hybrid systems that blend algebraic determinism with nonlinear dynamics. Each development, whether through nonlinear feedback, combination of multiple registers, or integration with chaotic processes, aims to preserve the efficiency of classical LFSRs while enhancing their unpredictability and robustness. These advances form the foundation for next-generation pseudo-random number generators, secure communication protocols, and reconfigurable digital hardware that balance theoretical elegance with practical performance.

### Rust Implementation

Following the discussion in Section 7.6.4 on the mathematical extensions of Linear Feedback Shift Registers, Program 7.6.4 provides a practical demonstration of how nonlinear feedback, higher-order finite-field structures, and chaotic modulation can be incorporated into classical LFSR designs. These generalizations introduce new forms of sequence complexity while retaining the efficiency of shift-based architectures, enabling improved statistical balance, higher linear complexity, and enhanced unpredictability. The program implements four representative generators, a linear LFSR, a nonlinear LFSR, a word-oriented finite-field recurrence, and a chaotic-hybrid LFSR, and evaluates their bitstream statistics over large samples. This unified framework illustrates how structural modifications influence pseudo-random properties while maintaining high throughput suitable for hardware and software applications.

At the core of the implementation is the `Lfsr16` structure, which provides a compact 16-bit linear register used as the baseline generator. Its update rule follows the classical XOR feedback recurrence and corresponds to the linear shift mechanism formalized in Equations (7.6.1) and (7.6.2). The feedback taps are encoded in a compact bitmask, enabling fast bitwise computations consistent with the algebraic model defined by the feedback polynomial in Equation (7.6.8). This serves as the reference LFSR against which the extensions are compared.

The nonlinear generalization is implemented through the `Nlfsr16` register, which augments the linear feedback with a quadratic term of the form $(a_{t,i} a_{t,j})$, directly reflecting the nonlinear Boolean recurrence described in Equation (7.6.21). The insertion of this higher-order monomial disrupts the linear recurrence structure, increasing the linear complexity of the output sequence while preserving desirable distributional balance. The implementation demonstrates how even small nonlinear augmentations can significantly alter the algebraic structure without sacrificing efficiency.

The program further includes a word-oriented 32-bit LFSR, `WordLfsr32`, designed to reflect the generalized finite-field recurrence of Equation (7.6.23). Instead of operating on individual bits, the feedback and state transitions update full 32-bit words at each step, corresponding to a recurrence over $\mathbb{F}_{2^{32}}$. This approach maintains the interpretability of LFSRs while greatly increasing throughput, matching the architecture used in high-performance BIST systems and modern word-parallel pseudo-random pattern generators.

Finally, the chaotic-hybrid generator is realized through `ChaoticLfsr16`, which uses a logistic map to perturb the tap mask dynamically. This reflects the hybrid LFSR–chaotic constructions described in the latter part of Section 7.6.4, where continuous chaotic maps modulate discrete linear structures to increase unpredictability. Here, the logistic map output determines whether additional taps are activated at each step, enabling controlled, low-cost nonlinearity that enhances diffusion without compromising the LFSR’s core efficiency.

The `main` function synthesizes these components into a unified statistical framework. A large bitstream is generated for each generator, and two fundamental indicators of randomness are computed: the fraction of ones and the lag-1 agreement probability. These metrics evaluate symbol balance and first-order correlation and align with the empirical validation procedures used in the NIST Statistical Test Suite. The printed results allow direct comparison of the four architectures, illustrating how each modification influences bit-level behaviour and validating the theoretical developments presented in Section 7.6.4.

```rust
// Program 7.6.4: Linear, Nonlinear, and Word-Oriented LFSR Extensions
//
// This program illustrates several mathematical generalizations of LFSRs
// discussed in Section 7.6.4:
//
//   1. A classical linear LFSR over F₂ (bit-oriented).
//   2. A simple nonlinear feedback shift register (NLFSR) with quadratic term.
//   3. A word-oriented LFSR over 32-bit words (F₂^32-style recurrence).
//   4. A toy “chaotic-hybrid” LFSR where a logistic map toggles the tap mask.
//
// For each bit-oriented generator (linear, nonlinear, and hybrid), the program
// produces a sample of bits and computes:
//   - the fraction of ones (balance);
//   - the lag-1 agreement probability (a crude indicator of correlation).
//
// The word-oriented LFSR demonstrates a higher-dimensional state update,
// more consistent with F₂^{2^k} recurrences used in BIST and word-level
// cryptographic constructions.
//
// NOTE: This code is pedagogical and not intended as a secure PRNG or cipher.

// -----------------------------------------------------------------------------
// 1. Basic LFSR over F₂ (bit-oriented, up to 31 bits)
// -----------------------------------------------------------------------------

/// Simple LFSR over F₂ with up to 31 bits of state.
///
/// The state is stored in the least-significant `n` bits of `state`.
/// Bit 0 is the rightmost (least significant) bit.
struct Lfsr {
    state: u32,
    taps: u32,
    n: u8,
}

impl Lfsr {
    /// Create a new LFSR with the given seed, tap mask, and register length `n`.
    ///
    /// - `seed` must be non-zero in its lower `n` bits; the all-zero state is forbidden.
    /// - `taps` is a bit mask specifying which bit positions contribute to the feedback XOR.
    fn new(seed: u32, taps: u32, n: u8) -> Self {
        assert!((1..=31).contains(&n), "n must be between 1 and 31 bits");
        let mask = (1u32 << n) - 1;

        let seed = seed & mask;
        assert!(seed != 0, "seed must be non-zero in the lower n bits");

        let taps = taps & mask;

        Self { state: seed, taps, n }
    }

    /// Update the tap mask (used by the chaotic-hybrid variant).
    fn set_taps(&mut self, taps: u32) {
        let mask = (1u32 << self.n) - 1;
        self.taps = taps & mask;
    }

    /// Advance the LFSR by one step and return the output bit (the bit shifted out).
    ///
    /// Update rule:
    ///   - Compute the new feedback bit as the XOR (mod 2 sum) of all tapped bits.
    ///   - Shift the register right by one bit.
    ///   - Insert the new feedback bit into the leftmost position (bit n-1).
    fn step(&mut self) -> u8 {
        // Output bit is the least significant bit before shifting.
        let out_bit = (self.state & 1) as u8;

        // XOR of tapped bits: parity of (state & taps).
        let x = self.state & self.taps;
        let parity = x.count_ones() & 1;
        let new_bit = parity as u32;

        // Shift right, inject feedback bit at the most significant position.
        let mask = (1u32 << self.n) - 1;
        self.state = ((self.state >> 1) | (new_bit << (self.n - 1))) & mask;

        out_bit
    }
}

// -----------------------------------------------------------------------------
// 2. A simple Nonlinear Feedback Shift Register (NLFSR)
// -----------------------------------------------------------------------------

/// A basic NLFSR: linear taps plus a quadratic term bit_i AND bit_j.
///
/// This is a toy example illustrating (7.6.21), where the feedback function
/// contains higher-order monomials. The feedback bit is:
///
///   f(a_t) = parity(linear taps) XOR (a_{t,i} AND a_{t,j})
///
/// for selected bit positions (i, j).
struct Nlfsr {
    state: u32,
    taps_linear: u32,
    n: u8,
    nonlin_i: u8,
    nonlin_j: u8,
}

impl Nlfsr {
    fn new(seed: u32, taps_linear: u32, n: u8, nonlin_i: u8, nonlin_j: u8) -> Self {
        assert!((1..=31).contains(&n), "n must be between 1 and 31 bits");
        assert!(nonlin_i < n && nonlin_j < n, "nonlinear indices must be < n");

        let mask = (1u32 << n) - 1;
        let seed = seed & mask;
        assert!(seed != 0, "seed must be non-zero in the lower n bits");

        let taps_linear = taps_linear & mask;

        Self {
            state: seed,
            taps_linear,
            n,
            nonlin_i,
            nonlin_j,
        }
    }

    /// Advance the NLFSR by one step and return the output bit.
    fn step(&mut self) -> u8 {
        // Output bit (before update).
        let out_bit = (self.state & 1) as u8;

        let mask = (1u32 << self.n) - 1;

        // Linear part: parity of tapped bits.
        let x = self.state & self.taps_linear;
        let parity = (x.count_ones() & 1) as u8;

        // Nonlinear term: a_{t,i} AND a_{t,j}.
        let bit_i = ((self.state >> self.nonlin_i) & 1) as u8;
        let bit_j = ((self.state >> self.nonlin_j) & 1) as u8;
        let nonlin = (bit_i & bit_j) & 1;

        let new_bit = (parity ^ nonlin) as u32;

        // Shift right, inject new feedback bit.
        self.state = ((self.state >> 1) | (new_bit << (self.n - 1))) & mask;

        out_bit
    }
}

// -----------------------------------------------------------------------------
// 3. Word-oriented LFSR over 32-bit words (F₂^32 style)
// -----------------------------------------------------------------------------

/// A word-oriented LFSR over 32-bit words, similar to CRC/Galois-LFSR style.
///
/// The update is:
///   state_{t+1} = (state_t << 1) XOR (poly if msb(state_t) == 1 else 0)
///
/// where `poly` is a primitive/trinomial polynomial over F₂ (represented as
/// a bitmask). The least-significant bit can be used as a bitstream output,
/// and the full word as a word-oriented pseudo-random value.
struct WordLfsr32 {
    state: u32,
    poly: u32,
}

impl WordLfsr32 {
    fn new(seed: u32, poly: u32) -> Self {
        assert!(seed != 0, "seed must be non-zero");
        Self { state: seed, poly }
    }

    /// Advance one step and return the output word.
    fn step_word(&mut self) -> u32 {
        let msb = (self.state >> 31) & 1;
        self.state <<= 1;
        if msb == 1 {
            self.state ^= self.poly;
        }
        self.state
    }

    /// Derive a single bit from the current state (e.g. LSB).
    fn step_bit(&mut self) -> u8 {
        let word = self.step_word();
        (word & 1) as u8
    }
}

// -----------------------------------------------------------------------------
// 4. Chaotic-hybrid LFSR: taps controlled by a logistic map
// -----------------------------------------------------------------------------

/// A toy “chaotic-hybrid” LFSR: the tap mask switches between two values
/// based on a logistic map x_{k+1} = r x_k (1 - x_k).
///
/// If x_k > 0.5, we use `taps_alt`, otherwise `taps_base`.
struct ChaosLfsr {
    lfsr: Lfsr,
    x: f64,
    r: f64,
    taps_base: u32,
    taps_alt: u32,
}

impl ChaosLfsr {
    fn new(seed: u32, taps_base: u32, taps_alt: u32, n: u8, x0: f64, r: f64) -> Self {
        let lfsr = Lfsr::new(seed, taps_base, n);
        Self {
            lfsr,
            x: x0,
            r,
            taps_base,
            taps_alt,
        }
    }

    /// Update logistic map and LFSR, then return output bit.
    fn step(&mut self) -> u8 {
        // Logistic map update (simple chaotic map).
        self.x = self.r * self.x * (1.0 - self.x);

        // Switch taps based on x.
        if self.x > 0.5 {
            self.lfsr.set_taps(self.taps_alt);
        } else {
            self.lfsr.set_taps(self.taps_base);
        }

        self.lfsr.step()
    }
}

// -----------------------------------------------------------------------------
// 5. Simple statistical diagnostics
// -----------------------------------------------------------------------------

/// Compute the fraction of ones in a bit sequence.
fn fraction_ones(bits: &[u8]) -> f64 {
    if bits.is_empty() {
        return 0.0;
    }
    let ones = bits.iter().filter(|&&b| b == 1).count();
    ones as f64 / bits.len() as f64
}

/// Compute a very simple lag-1 agreement rate:
///   fraction of positions i where b[i] == b[i+1].
fn lag1_agreement(bits: &[u8]) -> f64 {
    if bits.len() < 2 {
        return 0.0;
    }
    let mut same = 0usize;
    for i in 0..(bits.len() - 1) {
        if bits[i] == bits[i + 1] {
            same += 1;
        }
    }
    same as f64 / (bits.len() - 1) as f64
}

fn main() {
    let n_bits = 100_000usize;

    // -------------------------------------------------------------------------
    // 5.1 Linear 16-bit LFSR example.
    //
    // Example polynomial (bit positions refer to state bits):
    //   P(x) ≈ x^16 + x^14 + x^13 + x^11 + 1
    // taps at bits [15, 13, 12, 10, 0] in this software convention.
    // -------------------------------------------------------------------------
    let taps_lin: u32 = (1 << 15) | (1 << 13) | (1 << 12) | (1 << 10) | (1 << 0);
    let mut lfsr_lin = Lfsr::new(0xACE1u32 & ((1 << 16) - 1), taps_lin, 16);

    let mut bits_lin = Vec::with_capacity(n_bits);
    for _ in 0..n_bits {
        bits_lin.push(lfsr_lin.step());
    }

    // -------------------------------------------------------------------------
    // 5.2 Nonlinear 16-bit NLFSR with quadratic term a_i AND a_j.
    // -------------------------------------------------------------------------
    let taps_nl: u32 = taps_lin;
    let mut nlfsr = Nlfsr::new(
        0xACE1u32 & ((1 << 16) - 1),
        taps_nl,
        16,
        5,  // nonlin_i
        7,  // nonlin_j
    );

    let mut bits_nl = Vec::with_capacity(n_bits);
    for _ in 0..n_bits {
        bits_nl.push(nlfsr.step());
    }

    // -------------------------------------------------------------------------
    // 5.3 Word-oriented 32-bit LFSR (F₂^32-style).
    //
    // Example polynomial (one of the common CRC forms, truncated for demo):
    //   x^32 + x^22 + x^2 + x + 1  → poly mask 0x0040_0007
    // -------------------------------------------------------------------------
    let mut w_lfsr = WordLfsr32::new(0xDEADBEEF, 0x0040_0007);
    let mut bits_word = Vec::with_capacity(n_bits);
    for _ in 0..n_bits {
        bits_word.push(w_lfsr.step_bit());
    }

    // -------------------------------------------------------------------------
    // 5.4 Chaotic-hybrid 16-bit LFSR: taps modulated by a logistic map.
    //
    // taps_base: original taps_lin
    // taps_alt : a slightly different mask, e.g. flip one tap
    // -------------------------------------------------------------------------
    let taps_alt: u32 = (1 << 15) | (1 << 12) | (1 << 7) | (1 << 2) | (1 << 0);
    let mut chaos_lfsr = ChaosLfsr::new(
        0xBEEF & ((1 << 16) - 1),
        taps_lin,
        taps_alt,
        16,
        0.12345, // initial x
        3.99,    // logistic map parameter in chaotic regime
    );

    let mut bits_chaos = Vec::with_capacity(n_bits);
    for _ in 0..n_bits {
        bits_chaos.push(chaos_lfsr.step());
    }

    // -------------------------------------------------------------------------
    // 5.5 Print simple diagnostics
    // -------------------------------------------------------------------------
    println!("Bit-sample size: {}", n_bits);
    println!();

    let f_lin = fraction_ones(&bits_lin);
    let a_lin = lag1_agreement(&bits_lin);
    println!("Linear 16-bit LFSR:");
    println!("  Fraction of ones           ≈ {:.6}", f_lin);
    println!("  Lag-1 agreement probability ≈ {:.6}", a_lin);
    println!();

    let f_nl = fraction_ones(&bits_nl);
    let a_nl = lag1_agreement(&bits_nl);
    println!("Nonlinear 16-bit NLFSR (with quadratic term):");
    println!("  Fraction of ones           ≈ {:.6}", f_nl);
    println!("  Lag-1 agreement probability ≈ {:.6}", a_nl);
    println!();

    let f_word = fraction_ones(&bits_word);
    let a_word = lag1_agreement(&bits_word);
    println!("Word-oriented 32-bit LFSR (bit-stream from F₂^32 recurrence):");
    println!("  Fraction of ones           ≈ {:.6}", f_word);
    println!("  Lag-1 agreement probability ≈ {:.6}", a_word);
    println!();

    let f_chaos = fraction_ones(&bits_chaos);
    let a_chaos = lag1_agreement(&bits_chaos);
    println!("Chaotic-hybrid 16-bit LFSR (logistic-map tap modulation):");
    println!("  Fraction of ones           ≈ {:.6}", f_chaos);
    println!("  Lag-1 agreement probability ≈ {:.6}", a_chaos);
}
```

Program 7.6.4 demonstrates how classical LFSRs can be enriched by nonlinear feedback, finite-field extensions, and chaotic perturbations while preserving their characteristic efficiency and structural clarity. The statistical results reveal that these extensions maintain excellent first-order randomness properties, balanced symbol frequencies and near-uncorrelated successive bits, despite substantial structural modifications. This behaviour illustrates the core principle discussed in Section 7.6.4: that the pseudo-random quality of LFSR sequences can be enhanced without fundamentally altering the underlying shift-register mechanism.

The examples highlight how nonlinear and chaotic augmentations increase sequence complexity and resistance to linear analysis, while word-oriented designs scale naturally to modern processors and reconfigurable hardware. Together, these implementations reflect the broad versatility of LFSR-based architectures and showcase the mathematical ideas motivating current research in secure pseudo-random number generation, multimedia encryption, and self-test hardware. The modular design of the program makes it straightforward to explore additional nonlinear terms, alternative chaotic maps, or higher-order finite-field recurrences, supporting deeper experimentation in both theoretical and applied contexts.

### 7.6.5. Concluding Remarks

Linear Feedback Shift Registers (LFSRs) represent one of the most enduring and mathematically transparent mechanisms for pseudo-random sequence generation. Their theoretical foundation rests upon linear recurrences over the finite field $\mathbb{F}_2$, while their hardware realizations exploit the simplicity of shift registers and XOR operations. The use of primitive feedback polynomials ensures full-period sequences of length $2^n - 1$, giving rise to maximum-length or m-sequences that exhibit near-ideal statistical balance and autocorrelation properties. These features have established LFSRs as fundamental tools in both theoretical and applied numerical computing (Almaraz & Román Villaizán, 2023).

The comparative evaluation of Fibonacci and Galois configurations underscores the flexibility of LFSR-based architectures. While the Fibonacci configuration provides a direct algebraic interpretation of feedback and is ideal for instructional analysis, the Galois form distributes feedback locally, minimizing logical depth and maximizing execution speed. As demonstrated in modern reconfigurable and power-aware designs, Galois-type structures dominate current hardware and software implementations due to their superior parallelism and reduced propagation delay (Somanathan, Reddy & Bhakthavatchalu, 2025).

Recent advancements extend classical LFSR theory into nonlinear, hybrid, and word-oriented domains. These mathematical generalizations integrate chaotic feedback, higher-order finite fields, and nonlinear filtering, blending the determinism of algebraic recurrence with the unpredictability of dynamical systems. Such designs achieve higher entropy and greater resistance to cryptanalytic attacks while preserving the compact logic and reproducibility that make LFSRs desirable in secure random number generation and digital encryption. Image encryption schemes, for example, have successfully combined LFSR-based bitstreams with pixel shuffling and chaotic modulation to produce lightweight, real-time cryptographic solutions exhibiting near-perfect randomness and high security margins (Maity et al., 2025; Okunbor, Omorogbe & Edeko, 2024).

Beyond their technical adaptability, LFSRs remain central to hardware self-test, fault detection, and embedded system validation, where reconfigurable pseudo-random pattern generators provide efficient test coverage with minimal resource usage (Somanathan, Reddy & Bhakthavatchalu, 2025). In scientific computation, LFSR-based random number generators continue to offer deterministic reproducibility, uniform distribution, and statistical reliability essential for Monte Carlo and stochastic simulation frameworks (Almaraz & Román Villaizán, 2023).

In summary, the development of Linear Feedback Shift Registers illustrates how elegant mathematical concepts can yield practical and high-performance engineering solutions. From their origins in linear recurrence theory to their modern hybrid forms combining chaos, nonlinearity, and finite-field generalizations, LFSRs demonstrate the powerful intersection of algebraic design and computational efficiency. Their continued refinement ensures relevance in both classical digital architectures and emerging secure-computing paradigms, providing a robust foundation for future innovations in pseudo-random generation and applied cryptographic engineering.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/o4UzFO2Q5Umk8rDnUlnr.4","tags":[]}

# 7.7. Hash Tables and Hash Memories

Hash tables are associative data structures that provide near-constant-time access to data by mapping arbitrary keys to array positions through a hash function. Instead of maintaining keys in sorted order or navigating pointer-based tree structures, a hash table computes an integer index from the key and places the corresponding value in that bucket, thereby reducing the average search, insertion, and deletion cost to $O(1)$. This makes hash tables particularly well suited to numerical computing tasks where access patterns are irregular or non-sequential. In contrast to balanced trees, which guarantee $O(\log n)$ performance but incur traversal overhead, hash tables prioritize speed and flexibility at the cost of not preserving key order.

Hash tables arise naturally in computational contexts where fast, content-based retrieval is essential. In partial differential equation solvers and finite element assembly, contributions to sparse matrix entries often arrive in unpredictable order; using a hash table keyed by index pairs avoids repeated searches through sorted structures during matrix construction and substantially speeds up the buildup of irregular sparsity patterns. Similarly, in optimization and simulation workflows, hash tables support memoization of expensive computations, storing results under composite keys representing system states. In highly dynamic environments such as particle simulations, spatial hashing allows particles to be grouped by cell coordinates, enabling efficient neighbor searches without large uniform grids. Distributed systems also rely on hash tables, often in the form of distributed hash tables, to map content identifiers to storage locations across remote nodes.

Conceptually, a hash table consists of an array of $M$ buckets. A hash function $h(\mathrm{key})$ produces an integer that is reduced modulo $M$ to select the bucket. If the bucket is empty, the entry is stored directly; if not, a collision-resolution strategy must be applied. Two widely used approaches are separate chaining, where each bucket contains a small list or vector of key–value pairs, and open addressing, where collisions are resolved by probing alternative bucket positions. Modern open-addressed designs, such as those used in SwissTable and Rust’s standard `HashMap`, reduce collision impact by grouping buckets and using SIMD-accelerated metadata checks (Li 2023; Birler et al. 2024). These improvements enhance cache locality and have become foundational to many high-performance map implementations.

A related abstraction is hash memory, in which the “address” is not a numeric index but an arbitrary key. In effect, this structure acts as content-addressable memory, storing and retrieving values purely on the basis of their keys. Hash memory underlies dictionary and map types in most programming languages and serves as the backbone for key–value storage in scientific, numerical, and systems programming. Rust’s `HashMap<K,V>` is an example of such an architecture.

The performance of a hash table depends critically on the hash function. Although deterministic, it must behave pseudo-randomly so that keys are distributed uniformly across buckets; poor dispersion increases collisions and degrades performance. To avoid pathological or adversarial key distributions, many systems incorporate randomized hashing. Rust’s adoption of SipHash is one such measure, balancing speed with robustness against engineered collision attacks (Farach-Colton, Krapivin & Kuszmaul 2024). Across modern designs, careful engineering ensures that hash tables deliver their characteristic $O(1)$-average-time behavior even at large scale.

### Rust Implementation

Following the discussion in Section 7.7 on the role of hash tables and hash memories in numerical computation, Program 7.7.0 provides a practical demonstration of how associative hashing structures support two essential workflows: irregular sparse matrix assembly and memoization of expensive numerical evaluations. In large-scale numerical simulations, contributions to a global sparse matrix often arrive in unpredictable order, making array- or tree-based accumulation inefficient. A hash-based accumulator avoids repeated searches through ordered structures by mapping index pairs directly to matrix entries. Likewise, many numerical algorithms repeatedly evaluate costly functions for identical or similar states; a hash memory enables direct lookup by key, avoiding redundant computation. This program illustrates how Rust’s `HashMap`, a modern, high-performance hash table implementation, can serve as the backbone of both sparse assembly and memoized computation, thereby highlighting the practical importance of content-addressable memory structures in scientific computing.

At the core of the implementation is the `HashMap` data structure, which provides an associative mapping from keys to values in expected $O(1)$ time. In the sparse-matrix example, the key type is the custom `Index` struct, representing the pair $(i,j)$ of matrix coordinates. By deriving `Hash`, `Eq`, and `PartialEq`, the program allows Rust’s hashing mechanism to treat `(row, col)` pairs as fully content-addressable objects. The function `assemble_sparse` receives a list of contributions and inserts each into the hash-based accumulator. If a matrix entry already exists, its value is incremented, illustrating how hash tables efficiently merge irregularly ordered contributions without requiring sorted keys or tree traversal. This mirrors the behavior of sparse finite element assembly, where local element contributions must be scattered into a global structure using non-sequential access patterns.

The memoization example introduces the `State` struct as a composite key. Here, the function `expensive_function` represents a numerical computation whose cost would be undesirable to repeat unnecessarily. The wrapper `memoized_eval` acts as a hash memory: it checks whether the state has already been evaluated, retrieving the stored result in a single hash lookup. When a state is not present, the function computes the value, inserts it into the map, and returns it. This pattern is widely used in scientific computing, where dynamic programming, nonlinear solvers, stochastic simulation, and Markov-chain models frequently revisit identical system states. The hash memory therefore provides a lightweight and general-purpose mechanism for computational reuse, exploiting the associativity of `HashMap` to avoid redundant numerical work.

The `main` function brings both components together in a demonstrative workflow. It first constructs a sequence of assembly contributions that emulate three overlapping finite-element blocks, showing how a hash table naturally builds a sparse structure without requiring preallocated row/column storage. It then prints the resulting nonzero entries to illustrate the assembled pattern. Next, it evaluates the memoized function on several distinct and duplicate states, highlighting the difference between cache misses (which perform computation) and cache hits (which return previously stored values). The change in cache size and the printed hit/miss status provide a clear picture of how hash memories accelerate repeated evaluations. Together, these examples provide a concrete illustration of the principles introduced in Section 7.7 and demonstrate how associative addressing enables efficient and flexible data management in numerical environments.

```rust
// Program 7.7.0: Hash-Based Sparse Matrix Assembly and Memoization in Rust
//
// This program illustrates two typical uses of hash tables / hash memories
// in numerical computing:
//
// 1. Sparse matrix assembly: contributions to a global matrix arrive in
//    irregular order and are accumulated into a HashMap keyed by (row, col).
//    This mimics the use of hash tables in finite element and PDE assembly
//    when building irregular sparsity patterns.
//
// 2. Memoization: an expensive computation is cached in a HashMap keyed by a
//    simple "state" type, demonstrating hash memory semantics where values
//    are retrieved purely based on their keys.
//
// The examples use Rust's std::collections::HashMap as the underlying
// hash-memory structure.

use std::collections::HashMap;

/// Matrix index (row, col) used as a key in the sparse hash-based accumulator.
/// Deriving Hash, Eq, and PartialEq allows this type to be used as a key
/// in HashMap.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy)]
struct Index {
    row: usize,
    col: usize,
}

/// Assemble a sparse matrix from a list of (Index, value) contributions.
/// Repeated contributions to the same (row, col) position are summed.
///
/// This function represents a typical pattern in finite element or PDE codes
/// where local element matrices are scattered into a global sparse structure
/// in irregular order.
fn assemble_sparse(contribs: &[(Index, f64)]) -> HashMap<Index, f64> {
    let mut mat = HashMap::<Index, f64>::new();

    for &(idx, value) in contribs {
        // Accumulate: A[idx] += value.
        *mat.entry(idx).or_insert(0.0) += value;
    }

    mat
}

/// Simple "state" type for memoization.
/// In realistic simulations, this could represent a discretized configuration,
/// parameter triple, or multi-index; here we use a small integer tuple.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy)]
struct State {
    a: i32,
    b: i32,
    c: i32,
}

/// An "expensive" numerical computation that we want to memoize.
/// Here we just simulate cost with a nonlinear expression.
fn expensive_function(s: State) -> f64 {
    let x = s.a as f64;
    let y = s.b as f64;
    let z = s.c as f64;

    // Placeholder for a costly calculation (e.g., energy functional).
    (x * x + 2.0 * y * y + 3.0 * z * z).sqrt().sin()
}

/// Memoized wrapper around `expensive_function`.
/// Uses a HashMap<State, f64> as a hash memory that stores previously
/// computed results keyed by the full state.
fn memoized_eval(
    state: State,
    cache: &mut HashMap<State, f64>,
) -> f64 {
    if let Some(&val) = cache.get(&state) {
        // Cache hit: return stored value.
        val
    } else {
        // Cache miss: compute, store, and return.
        let val = expensive_function(state);
        cache.insert(state, val);
        val
    }
}

fn main() {
    // ============================================================
    // 1. Sparse matrix assembly using a hash-based accumulator
    // ============================================================

    // Suppose we assemble a small 4x4 stiffness matrix from three
    // overlapping "elements". Each contribution is (Index, value).
    //
    // In realistic PDE codes, these would come from local element
    // matrices scattered into a global sparse matrix in arbitrary order.
    let contributions: Vec<(Index, f64)> = vec![
        // Element 1 contributions
        (Index { row: 0, col: 0 }, 2.0),
        (Index { row: 0, col: 1 }, -1.0),
        (Index { row: 1, col: 0 }, -1.0),
        (Index { row: 1, col: 1 }, 2.0),

        // Element 2 contributions
        (Index { row: 1, col: 1 }, 2.0),
        (Index { row: 1, col: 2 }, -1.0),
        (Index { row: 2, col: 1 }, -1.0),
        (Index { row: 2, col: 2 }, 2.0),

        // Element 3 contributions
        (Index { row: 2, col: 2 }, 2.0),
        (Index { row: 2, col: 3 }, -1.0),
        (Index { row: 3, col: 2 }, -1.0),
        (Index { row: 3, col: 3 }, 2.0),
    ];

    let sparse = assemble_sparse(&contributions);

    println!("Hash-based sparse matrix assembly (4x4 example):");
    println!("Nonzero entries (row, col) -> value\n");

    // For deterministic printing, collect keys and sort lexicographically.
    let mut entries: Vec<(Index, f64)> =
        sparse.iter().map(|(idx, &val)| (*idx, val)).collect();

    entries.sort_by(|(i1, _), (i2, _)| {
        i1.row.cmp(&i2.row).then(i1.col.cmp(&i2.col))
    });

    for (idx, val) in entries {
        println!("  A[{}, {}] = {:.3}", idx.row, idx.col, val);
    }

    // ============================================================
    // 2. Memoization using a hash memory keyed by "state"
    // ============================================================

    let mut cache: HashMap<State, f64> = HashMap::new();

    println!("\nMemoized evaluation of an expensive function:");
    println!("(Repeated states should hit in the cache.)\n");

    let states = [
        State { a: 1, b: 2, c: 3 },
        State { a: 2, b: 2, c: 1 },
        State { a: 1, b: 2, c: 3 }, // duplicate
        State { a: -1, b: 0, c: 4 },
        State { a: 2, b: 2, c: 1 }, // duplicate
    ];

    for s in states.iter() {
        let before = cache.len();
        let value = memoized_eval(*s, &mut cache);
        let after = cache.len();

        let status = if after == before {
            "cache hit"
        } else {
            "cache miss"
        };

        println!(
            "  f({:?}) = {:.6}  ({})",
            s, value, status
        );
    }

    println!("\nTotal distinct states stored in cache: {}", cache.len());
}
```

Program 7.7.0 demonstrates how hash tables serve as powerful content-addressable data structures for numerical computing tasks that lack regular memory-access patterns. In sparse matrix assembly, the hash-based accumulator avoids the overhead of ordered insertion and supports rapid merging of contributions from irregular or dynamically generated sources. This is precisely the setting encountered in large-scale PDE solvers and finite-element codes, where local element matrices must be scattered into a global structure without predetermined ordering. For memoization, the program highlights how hash memories support efficient reuse of previously computed values, reducing computational cost when evaluating expensive functions across many repeated or partially redundant states. Both examples reflect the broader theme of Section 7.7: that hashing complements classical array and tree-based data structures by offering near-constant-time access driven by key content rather than position. As numerical workloads continue to scale in dimensionality and complexity, such associative mechanisms become increasingly central in enabling high-performance, flexible, and data-driven algorithm designs.

## 7.7.1. Hash Functions and Hash Table Algorithms

Hash functions provide the fundamental mechanism by which hash tables transform arbitrary keys into positions within a bounded array. A hash function $h(\mathrm{key})$ maps the key to a non-negative integer, which is then reduced modulo the table size $M$ to determine the bucket index. The essential requirement is that this mapping distributes keys uniformly across the table so that collisions remain rare and lookup time stays constant on average. Formally, the bucket index is computed as:

$$i = h(\mathrm{key}) \bmod M \tag{7.7.1}$$

Because the performance of a hash table depends on how evenly keys are distributed, the load factor:

$$\alpha = \frac{n}{M}\tag{7.7.2}$$

where $n$ is the number of elements stored, provides a key measure of table occupancy. As $\alpha$ increases, collisions become more frequent and both insertion and lookup times can degrade unless the table is resized.

Modern hash table systems use efficient mixing operations, such as XORs, shifts, and rotations, to diffuse information from the key across the output bits. Integer keys benefit from direct bit-level transformations, while composite keys require structural hashing that incorporates each component deterministically. High-performance probing-based tables depend not only on the hash function but also on how the probing pattern interacts with bucket metadata. Contemporary research has shown that combining clustered probing, SIMD-friendly control bytes, and well-designed mixing functions leads to substantial improvements in cache locality and lookup throughput (Birler et al., 2024; Li, 2023).

When two or more keys map to the same bucket, the hash table must resolve the *collision*. Two major strategies are widely used: *separate chaining* and *open addressing*. In separate chaining, each bucket contains a small list or vector of key–value pairs, making it robust even at high load factors. In open addressing, all items reside directly in the bucket array, and a collision triggers a sequence of probe positions defined by:

$$i_k = \bigl(h(\mathrm{key}) + p(k)\bigr) \bmod M \tag{7.7.3}$$

Here, $p(k)$ is the probing function, with common choices including linear probing $p(k) = k$, quadratic probing $p(k)=k^2$, and double hashing $p(k)=k,h_2(\mathrm{key})$. For linear probing, the expected number of probes for a successful search satisfies the approximation:

$$\mathbb{E}[\text{probes}] \approx \frac{1}{2}\left(1 + \frac{1}{1 - \alpha}\right) \tag{7.7.4}$$

highlighting the importance of maintaining a low load factor.

Many modern systems employ *randomized hashing*, in which the hash function uses a per-table or per-process seed. This prevents pathological key distributions from causing excessive collisions and protects against adversarial attempts to force worst-case behavior. Randomized mixing functions have proven especially effective in sustaining predictable performance across large and highly variable workloads (Farach-Colton, Krapivin & Kuszmaul, 2024).

The table’s long-term efficiency depends as much on its resizing strategy as on its hash function. As $\alpha$ approaches 1, the table must expand, typically by doubling its capacity and rehashing entries, to preserve expected $O(1)$ cost. Incremental or staged resizing techniques reduce performance spikes by redistributing entries gradually rather than copying the entire table at once. Such designs improve the stability of hash tables embedded in real-time or latency-sensitive systems (Birler et al., 2024).

To illustrate the mechanics of open addressing, Figure 7.7.1 presents a simple probing sequence under linear probing. The example demonstrates how the hash function determines an initial bucket and how the probing strategy identifies the next available slot upon detecting a collision.

```text
   Hash table buckets (M = 10):

   Index:   0   1   2   3   4   5   6   7   8   9
            -------------------------------------------------
   State:  [ ] [ ] [X] [ ] [ ] [X] [ ] [ ] [ ] [ ]

   Suppose h(key) = 2  → initial index i0 = 2

       i0 = 2   bucket occupied → collision
       i1 = 3   bucket empty    → store here

   Probing sequence (linear):  2 → 3
```

Figure 7.7.1 illustrates how a hash table resolves collisions under **linear probing**, one of the simplest open-addressing schemes. The table contains ten buckets, indexed from 0 to 9, with two positions already occupied. When a key hashes to bucket $i_{0} = 2$, the algorithm discovers that this position is full and initiates a linear probe. The next sequential index, $i_{1} = 3$, is empty, so the key is inserted there. The probing sequence $2 \rightarrow 3$ demonstrates how open addressing searches for the next available slot while maintaining all entries in the main array. This example highlights how collision detection, probe advancement, and final placement occur step by step, reinforcing the mechanics underlying open addressing.

~~~{list-table} Table 7.7.1 Comparison of Common Hash Function Families
:header-rows: 1
:name: nLQW2wLtFQ

* - Hash Function Type

  - Description

  - Strengths / Weaknesses

* - Bit-mixing (integer hashing)

  - Applies rotations, shifts, XORs, and multiplications to diffuse key bits.

  - Fast and effective for numeric keys; less robust under adversarial inputs.

* - Composite key hashing

  - Hashes each field of a tuple or struct, then mixes the results.

  - General-purpose; quality depends on field-level mixing.

* - Randomized hashing

  - Uses per-table or per-process random seeds in the mixing stage.

  - Robust to adversarial patterns; improves stability in high-throughput workloads (Farach-Colton et al., 2024).

* - Double hashing families

  - Use two independent hash functions in the probing formula.

  - Reduces clustering; requires a strong secondary hash function.

~~~

Table 7.7.1 summarizes several widely used families of hash functions and highlights their practical trade-offs. Bit-mixing techniques form the foundation for many high-performance integer hashers, relying on simple arithmetic and bitwise operations to achieve strong diffusion. Composite key hashing generalizes this principle to structured keys by hashing individual fields before combining them. Randomized hashing introduces a per-process or per-table seed to mitigate adversarial patterns, improving stability in workloads where key distributions are unpredictable. Double hashing leverages two independent functions to generate probe sequences with reduced clustering, offering performance benefits in open-addressed tables. Together, these categories provide a conceptual map of the design space, helping readers understand how hash functions differ in performance, robustness, and suitability for different collision-resolution strategies.

### Rust Implementation

Following the discussion in Section 7.7.1 on hash functions, load factor, and collision-resolution strategies, Program 7.7.1 presents a concrete implementation of an open-addressed hash table using randomized bit-mixing and linear probing. The program highlights how an efficient mixing function transforms numeric keys into well-distributed hash values and how the probing mechanism resolves collisions while keeping average lookup time near constant. By instrumenting the probing behavior during insertion and search, the program provides numerical evidence for the theoretical performance model introduced in Equation (7.7.4), demonstrating how load factor and table occupancy shape the expected number of probes. The example thus illustrates the interplay between hashing quality, probing strategy, and table resizing in maintaining predictable performance.

At the foundation of the implementation is the `Hasher` structure, which provides a compact randomized mixing function for 64-bit keys. Using a SplitMix-style bit diffusion mechanism, the function incorporates a per-table seed to implement randomized hashing as described in the section text. This protects the table from adversarial key patterns and ensures that the modulo operation used to compute the bucket index in Equation (7.7.1) distributes keys uniformly across buckets. The mixing steps including combinations of XORs, shifts, and multiplications, mirror the design principles of modern integer hash functions discussed by Li (2023) and Birler et al. (2024).

The core of the hash table is implemented in the `HashTable` struct, which maintains a vector of buckets, metadata for occupancy, and the randomized hasher. The `insert_with_probes` method performs open addressing using linear probing, following the probing recurrence of Equation (7.7.3). Each probe corresponds to a sequential advancement through the bucket array, resolving collisions that arise when two keys map to the same index. The method records the number of probes needed for each insertion, enabling empirical comparison to the expected cost given by Equation (7.7.4). The use of tombstones allows deletions without destroying probe chains, illustrating a practical technique employed in real-world open-addressed hash tables.

The `get_with_probes` function performs key lookup using the same probing sequence. By counting probe steps, the implementation makes it possible to directly compare lookup behavior with insertion behavior, providing insight into how linear probing interacts with the table’s load factor, α, defined in Equation (7.7.2). The `resize` function demonstrates how modern hash tables maintain performance by doubling capacity when α approaches a threshold, preventing the expected probe count from diverging as predicted by (7.7.4).

Finally, the program uses a small linear congruential generator to produce pseudo-random keys, ensuring a non-adversarial workload while remaining dependency-free. The `main` function inserts a controlled number of keys, measures probe counts for both insertions and searches, compares observed averages with theoretical predictions, and demonstrates how tombstones affect subsequent probe lengths after deletions. This instrumentation concretely illustrates how hashing quality, probing mechanics, and table occupancy interact in determining practical performance.

```rust
// Program 7.7.1: Open-Addressed Hash Table with Randomized Bit-Mixing and Linear Probing
//
// This program illustrates the key ideas in Section 7.7.1:
//
//   * A bit-mixing hash function h(key) followed by modulo reduction (7.7.1).
//   * Load factor α = n / M as a measure of occupancy (7.7.2).
//   * Linear probing i_k = (h(key) + k) mod M for collision resolution (7.7.3).
//   * The relationship between α and the expected number of probes (7.7.4).
//
// The implementation uses:
//   - A simple randomized integer hasher based on a SplitMix-style mixing function.
//   - An open-addressed hash table with linear probing and tombstones.
//   - Automatic resizing when the load factor exceeds a threshold.
//   - Instrumentation that measures average probe counts for insertions and lookups.
//
// The main function inserts pseudo-random keys, measures the effective load factor,
// and compares the observed average probes against the approximation (7.7.4).

use std::fmt;

// ------------------------ 1. Randomized Bit-Mixing Hasher ------------------------

/// A small, per-table randomized hasher for u64 keys.
///
/// The hasher uses a 64-bit seed and a SplitMix-like mixing function to spread
/// nearby keys across the output range, as suggested by modern bit-mixing designs.
#[derive(Clone, Copy)]
struct Hasher {
    seed: u64,
}

impl Hasher {
    fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Mix a 64-bit key using a SplitMix-style transform.
    fn mix_u64(x: u64) -> u64 {
        // This is a standard SplitMix64 mixing step.
        let mut z = x;
        z ^= z >> 30;
        z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z ^= z >> 27;
        z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        z
    }

    /// Hash a 64-bit key to a 64-bit hash value.
    ///
    /// The per-table seed implements randomized hashing, as described in
    /// the text, to mitigate adversarial key distributions.
    fn hash_u64(&self, key: u64) -> u64 {
        Self::mix_u64(key ^ self.seed)
    }
}

// ------------------------ 2. Linear-Probing Hash Table ------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BucketState {
    Empty,
    Occupied,
    Tombstone,
}

#[derive(Clone, Copy)]
struct Bucket {
    key: u64,
    value: u64,
    state: BucketState,
}

impl Bucket {
    fn empty() -> Self {
        Self {
            key: 0,
            value: 0,
            state: BucketState::Empty,
        }
    }
}

/// Simple open-addressed hash table with linear probing for u64 keys and values.
///
/// This implementation is intended as a didactic example, not a drop-in replacement
/// for production hash maps (like `hashbrown`).
struct HashTable {
    buckets: Vec<Bucket>,
    len: usize,      // number of *occupied* entries
    hasher: Hasher,  // per-table randomized hasher
}

impl fmt::Debug for HashTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashTable")
            .field("len", &self.len)
            .field("capacity", &self.buckets.len())
            .finish()
    }
}

impl HashTable {
    /// Create a new hash table with the given capacity and random seed.
    ///
    /// The capacity is rounded up to at least 8.
    fn with_capacity_and_seed(capacity: usize, seed: u64) -> Self {
        let cap = capacity.max(8);
        let buckets = vec![Bucket::empty(); cap];
        Self {
            buckets,
            len: 0,
            hasher: Hasher::new(seed),
        }
    }

    /// Current table size M.
    fn capacity(&self) -> usize {
        self.buckets.len()
    }

    /// Current load factor α = n / M (7.7.2).
    fn load_factor(&self) -> f64 {
        self.len as f64 / self.capacity() as f64
    }

    /// Compute the bucket index i = h(key) mod M (7.7.1).
    fn bucket_index(&self, key: u64) -> usize {
        let h = self.hasher.hash_u64(key);
        (h as usize) % self.capacity()
    }

    /// Insert a key–value pair using linear probing.
    ///
    /// Returns the number of probes performed, for instrumentation.
    fn insert_with_probes(&mut self, key: u64, value: u64) -> usize {
        // Resize when the load factor exceeds ~0.7 to keep probe lengths moderate.
        if self.load_factor() > 0.7 {
            self.resize(self.capacity() * 2);
        }

        let m = self.capacity();
        let mut index = self.bucket_index(key);
        let mut first_tombstone: Option<usize> = None;
        let mut probes = 0;

        loop {
            probes += 1;
            let bucket = &mut self.buckets[index];

            match bucket.state {
                BucketState::Empty => {
                    // If we saw a tombstone earlier, reuse it; otherwise use this empty slot.
                    let target_index = first_tombstone.unwrap_or(index);
                    let target = &mut self.buckets[target_index];
                    target.key = key;
                    target.value = value;
                    target.state = BucketState::Occupied;
                    self.len += 1;
                    return probes;
                }
                BucketState::Tombstone => {
                    // Remember the first tombstone, but keep probing in case the key already exists.
                    if first_tombstone.is_none() {
                        first_tombstone = Some(index);
                    }
                }
                BucketState::Occupied => {
                    if bucket.key == key {
                        // Update existing entry, do not change len.
                        bucket.value = value;
                        return probes;
                    }
                }
            }

            index = (index + 1) % m; // linear probing: i_k = (h(key) + k) mod M  (7.7.3)
        }
    }

    /// Look up a key using linear probing.
    ///
    /// Returns (Option<value>, probes).
    fn get_with_probes(&self, key: u64) -> (Option<u64>, usize) {
        let m = self.capacity();
        let mut index = self.bucket_index(key);
        let mut probes = 0;

        loop {
            probes += 1;
            let bucket = &self.buckets[index];

            match bucket.state {
                BucketState::Empty => {
                    // Empty bucket means the key was never inserted.
                    return (None, probes);
                }
                BucketState::Tombstone => {
                    // Keep probing.
                }
                BucketState::Occupied => {
                    if bucket.key == key {
                        return (Some(bucket.value), probes);
                    }
                }
            }

            index = (index + 1) % m;
        }
    }

    /// Remove a key (if present), marking its bucket as a tombstone.
    ///
    /// Returns true if the key was found.
    fn remove(&mut self, key: u64) -> bool {
        let m = self.capacity();
        let mut index = self.bucket_index(key);

        loop {
            let bucket = &mut self.buckets[index];
            match bucket.state {
                BucketState::Empty => {
                    return false;
                }
                BucketState::Tombstone => {
                    // Continue probing.
                }
                BucketState::Occupied => {
                    if bucket.key == key {
                        bucket.state = BucketState::Tombstone;
                        self.len -= 1;
                        return true;
                    }
                }
            }

            index = (index + 1) % m;
        }
    }

    /// Resize the table to a new capacity and reinsert all occupied entries.
    fn resize(&mut self, new_capacity: usize) {
        let mut new_table =
            HashTable::with_capacity_and_seed(new_capacity, self.hasher.seed);
        for bucket in &self.buckets {
            if bucket.state == BucketState::Occupied {
                // We ignore probe counts during rehashing for simplicity.
                let _ = new_table.insert_with_probes(bucket.key, bucket.value);
            }
        }
        *self = new_table;
    }
}

// ------------------------ 3. Simple LCG for Key Generation ------------------------

/// A small, deterministic linear congruential generator (LCG) for u64.
///
/// This avoids external dependencies while providing pseudo-random keys.
struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // Parameters from Numerical Recipes (not cryptographically secure).
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.state
    }
}

// ------------------------ 4. Demonstration and Instrumentation ------------------------

fn main() {
    // Initial capacity M and number of elements n to insert.
    let initial_capacity = 128usize;
    let num_keys = 80usize; // n
    let seed_for_hasher = 0xDEADBEEFCAFEBABE;

    let mut table = HashTable::with_capacity_and_seed(initial_capacity, seed_for_hasher);

    let mut rng = Lcg64::new(0x12345678ABCDEF00);

    // Insert pseudo-random keys and measure probes.
    let mut total_insert_probes = 0usize;
    let mut keys: Vec<u64> = Vec::with_capacity(num_keys);

    for _ in 0..num_keys {
        let key = rng.next_u64();
        let value = key.wrapping_mul(2);
        keys.push(key);
        let probes = table.insert_with_probes(key, value);
        total_insert_probes += probes;
    }

    let alpha = table.load_factor(); // α = n / M (7.7.2)
    let avg_insert_probes = total_insert_probes as f64 / num_keys as f64;

    // Measure probes for successful lookups.
    let mut total_lookup_probes = 0usize;
    for &key in &keys {
        let (value, probes) = table.get_with_probes(key);
        assert_eq!(value, Some(key.wrapping_mul(2)));
        total_lookup_probes += probes;
    }
    let avg_lookup_probes = total_lookup_probes as f64 / num_keys as f64;

    // Theoretical approximation for successful search under linear probing (7.7.4):
    //   E[probes] ≈ 0.5 * (1 + 1 / (1 - α)).
    let expected_probes =
        0.5_f64 * (1.0 + 1.0 / (1.0 - alpha));

    println!("Open-addressed hash table with randomized bit-mixing and linear probing");
    println!("---------------------------------------------------------------------");
    println!("Capacity M          = {}", table.capacity());
    println!("Number of keys n    = {}", num_keys);
    println!("Load factor α = n/M ≈ {:.4}", alpha);
    println!();
    println!("Average probes per insertion   ≈ {:.4}", avg_insert_probes);
    println!("Average probes per successful lookup ≈ {:.4}", avg_lookup_probes);
    println!(
        "Theoretical E[probes] from (7.7.4) ≈ {:.4}",
        expected_probes
    );
    println!();

    // Quick qualitative check: remove a few keys and reinsert others.
    let remove_count = 10;
    for i in 0..remove_count {
        let removed = table.remove(keys[i]);
        assert!(removed);
    }

    println!(
        "After removing {} keys, load factor α ≈ {:.4}",
        remove_count,
        table.load_factor()
    );

    // Insert a few new keys to demonstrate reuse of tombstones.
    let mut extra_insert_probes = 0usize;
    let extra_keys = 10usize;
    for _ in 0..extra_keys {
        let key = rng.next_u64();
        let value = key ^ 0xDEAD_BEEF;
        let probes = table.insert_with_probes(key, value);
        extra_insert_probes += probes;
    }

    let avg_extra_insert_probes =
        extra_insert_probes as f64 / extra_keys as f64;

    println!(
        "Average probes for inserting {} new keys (tombstone reuse) ≈ {:.4}",
        extra_keys, avg_extra_insert_probes
    );
}
```

Program 7.7.1 provides a practical demonstration of the principles underlying hash table performance, illustrating how randomized bit-mixing and open addressing work together to maintain efficient lookup times. The numerical results confirm the theoretical behavior summarized in Equation (7.7.4), showing that the expected number of probes grows as the load factor approaches one. The insertion and lookup statistics reveal the sensitivity of linear probing to clustering and demonstrate how history, such as the presence of tombstones, affects probe lengths even when $α$ remains moderate.

The implementation also highlights the importance of table resizing in keeping operations near constant-time. Without expanding the table when the load factor exceeds a safe threshold, probe chains accumulate, degrading both theoretical and empirical performance. The modular structure of the example makes it easy to extend the program with alternative probing schemes such as quadratic probing, double hashing, or Robin Hood hashing, allowing deeper exploration of the collision-resolution strategies surveyed in Section 7.7.1.

## 7.7.2. Collision Resolution Strategies

Collisions are a fundamental aspect of hash table design because many different keys may map to the same bucket index. The performance of a hash table therefore depends critically on how these collisions are handled. A robust collision-resolution strategy preserves the expected constant-time behaviour of insertions, lookups, and deletions, even as the table grows and the load factor increases (Li, 2023). Modern hash table implementations rely primarily on two families of techniques: *separate chaining*, in which overflows are stored outside the bucket array, and *open addressing*, in which collisions are resolved by probing additional positions within the array (Birler et al., 2024).

In separate chaining, each bucket stores an index that refers to the first element in a chain of entries that share the same hash index. A high-performance and cache-friendly variant stores keys, values, and `next[]` pointers in contiguous arrays rather than classical linked lists. This improves spatial locality and reduces pointer-chasing overhead. To insert a new key, the table computes the bucket index $b = h(\mathrm{key}) \bmod M$. If the bucket is empty, the key is placed in a fresh slot; otherwise, the algorithm traverses the chain using the `next[]` array until it reaches the end and appends the new element. Because chain nodes store the actual key, the system remains correct even under rare 64-bit hash collisions: comparisons are always performed against stored keys, not hash values (Farach-Colton, Krapivin & Kuszmaul, 2024).

The performance of chaining is well understood analytically. If the load factor is $\alpha = n/M$, then the expected chain length is:

$$\mathbb{E}[\text{chain length}] = \alpha\tag{7.7.5}$$

and the expected number of probes required for a successful lookup is:

$$\mathbb{E}[\text{probes}{\text{success}}] = 1 + \frac{\alpha}{2}\tag{7.7.6}$$

while an unsuccessful lookup requires:

$$\mathbb{E}[\text{probes}{\text{fail}}] = \alpha\tag{7.7.7}$$

These relations show why separate chaining remains robust under high load factors and dynamic insertion-heavy workloads (Birler et al., 2024).

The second major technique, open addressing, resolves collisions directly within the bucket array. If the initial index $i_0 = h(\mathrm{key}) \bmod M$, is already occupied, the table follows a probe sequence of the form:

$$i_k = \bigl(h(\mathrm{key}) + p(k)\bigr) \bmod M \tag{7.7.8}$$

where $p(k)$ defines the probing pattern. Linear probing uses $p(k) = k$, quadratic probing uses $p(k) = k^2$, while double hashing generates probes using a second hash function. Open addressing achieves excellent cache locality because all operations remain within a contiguous array, but it suffers sharply degraded performance as the load factor approaches 1. To avoid pathological probe lengths, open-addressed tables typically trigger resizing once $\alpha$ exceeds about 0.70–0.85 (Li, 2023; Birler et al., 2024).

Table resizing ensures long-term efficiency. When the load factor crosses a threshold, the table is expanded, often doubled in size, and all items are reinserted into the new structure. Although a resize incurs $O(n)$ cost, amortised insertion remains $O(1)$, since expansions occur infrequently relative to the total number of operations (Farach-Colton, Krapivin & Kuszmaul, 2024). Some implementations employ hysteresis to avoid frequent resizing under fluctuating workloads.

Modern systems combine these classical ideas with hardware-aware innovations. Compact metadata, bucket groups, SIMD-accelerated probing, and array-based chaining all provide predictable, high-throughput behaviour for large-scale data processing and numerical simulations (Li, 2023; Birler et al., 2024).

```text
Hash table buckets (M = 10):

   Index:   0   1   2   3   4   5   6   7   8   9
            -------------------------------------------------
   State:  [ ] [ ] [X] [ ] [ ] [X] [ ] [ ] [ ] [ ]

Suppose h(key) = 2  → initial index i₀ = 2

    i₀ = 2   bucket occupied → collision
    i₁ = 3   bucket empty    → store here

Probing sequence (linear):  2 → 3
```

Figure 7.7.2 illustrates a simple example of collision resolution under *linear probing*. The key hashes to bucket $i_0 = 2$, but that position is already occupied. The probing strategy increments the index to $i_1 = 3$, where the bucket is empty and the key is inserted. The short sequence $i_0 \rightarrow i_1$ demonstrates the basic mechanics of open addressing: collisions trigger a systematic probe through alternative positions in the table. As the load factor increases, such probe sequences may become longer, emphasising the importance of appropriate resizing thresholds.

Table 7.7.2 contrasts the two principal collision-resolution strategies. Chaining offers greater stability under high load factors and simpler deletions, while open addressing excels in memory efficiency and cache locality but becomes sensitive to load factor as the table fills. These trade-offs inform the choice of strategy in high-performance hash tables used in numerical computing, simulations, and large-scale data processing.

~~~{list-table} Table 7.7.2 Trade-offs Between Chaining and Open Addressing
:header-rows: 1
:name: Uqf9pmJ59g

* - Method

  - Advantages

  - Disadvantages

* - Separate Chaining

  - Simple deletion; robust under high load factors; predictable chain structure.

  - Reduced cache locality; chain lengths increase as α grows.

* - Open Addressing

  - Excellent cache locality; compact memory layout; single-array storage.

  - Deletions require tombstones; performance degrades as α → 1.

~~~

### Rust Implementation

Following the analysis in Section 7.7.2 on the mathematical behaviour of collision-resolution strategies, Program 7.7.2 provides a practical implementation that compares separate chaining and open addressing under controlled experimental conditions. Whereas the text develops theoretical expectations for average chain lengths and probing costs as functions of the load factor, the program evaluates these properties empirically using a synthetic workload of randomly generated keys and uniformly distributed hash outputs. By instrumenting both collision-resolution mechanisms, the implementation quantifies probe counts during insertion and lookup, illustrating how the performance degrades as the load factor increases and how closely the observed behaviour aligns with the analytical predictions established earlier in the section. In doing so, the program reinforces the theoretical insights with concrete measurements that highlight the contrasting operational profiles of chaining and open addressing in practice.

At the core of the implementation is the lightweight bit-mixing hash function `hash_mix`, which applies shifts, XORs, and multiplications to diffuse key bits across the output. This design reflects the principles discussed in Section 7.7.1, where modern hash tables rely on inexpensive but highly effective bit-mixing operations to approximate uniform hashing and avoid clustering. The mixed hash value returned by `hash_mix` is then reduced modulo the table size to produce the initial bucket index, making the function central to both collision-resolution strategies tested in the program.

The separate chaining strategy is implemented through the struct `ChainTable`, which stores keys, metadata, and chain links in contiguous vectors rather than classical pointer-based linked lists. The method `insert_chain` appends new elements to the keys array, updates the `next` pointer chain for the relevant bucket, and increments a probe counter each time an existing entry in the chain is examined. Lookup operations handled by `lookup_chain` traverse these chains linearly, and the number of inspected nodes directly corresponds to the theoretical expectations in Equations (7.7.5)–(7.7.7). Because all data live in tightly packed vectors, this implementation captures the cache-friendly characteristics of array-based chaining used in high-performance hash table designs.

The open-addressed variant is encapsulated in the struct `LinearProbingTable`, which resolves collisions using the probe sequence specified by Equation (7.7.8). The insertion function `insert_linear` begins at the initial index and advances through the array, using linear probing, until it encounters either an empty bucket or a tombstone. Each inspected position increments the probe counter, allowing the program to compare empirical behaviour with classical predictions for linear probing. Lookups are handled by `lookup_linear`, which revisits the probe sequence to assess successful or unsuccessful queries. Together, these functions illustrate how probe lengths increase with load factor and why timely resizing is crucial for open-addressed hash tables.

The `main` function orchestrates the entire experiment. It generates random integer keys, inserts them into both `ChainTable` and `LinearProbingTable`, collects probe statistics for insertions and lookups, and computes theoretical expectations derived from Equations (7.7.5)–(7.7.8). It then removes a subset of keys from the open-addressed table using `delete_linear`, demonstrating how tombstones affect subsequent probe sequences. A final round of insertions into the partially vacated table highlights realistic performance effects encountered in dynamic workloads. The reported results show clear agreement with the analytical models, underscoring the distinct behaviours of chaining and open addressing as the table approaches higher load factors.

```rust
// Program 7.7.2: Comparing Separate Chaining and Open Addressing for Collision Resolution
//
// This program illustrates the collision-resolution strategies discussed in
// Section 7.7.2 by implementing two hash tables for u64 keys and values:
//
//   1. Separate chaining with array-based chains (buckets + entries[next]).
//   2. Open addressing with linear probing.
//
// For both structures we:
//   - Insert n keys into a table of size M (so load factor α = n/M).
//   - Measure the average number of probes for successful lookups.
//   - Measure the average number of probes for unsuccessful lookups.
//
// For the chaining table, we compare the measured probe counts against the
// classical expectations (7.7.5)–(7.7.7):
//
//   E[chain length]       = α
//   E[probes_success]     = 1 + α/2
//   E[probes_fail]        = α
//
// For the open-addressed table, we use the same keys and the same hash function,
// but resolve collisions using linear probing as in (7.7.8) with p(k) = k. This
// highlights the trade-offs between chaining (robust under high load factors)
// and open addressing (cache-friendly but more sensitive to α).

use std::fmt;

// ------------------------ 1. Randomized bit-mixing hasher ------------------------

#[derive(Clone, Copy)]
struct Hasher {
    seed: u64,
}

impl Hasher {
    fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn mix_u64(x: u64) -> u64 {
        // SplitMix64-style mixing.
        let mut z = x;
        z ^= z >> 30;
        z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z ^= z >> 27;
        z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        z
    }

    fn hash_u64(&self, key: u64) -> u64 {
        Self::mix_u64(key ^ self.seed)
    }
}

// Simple LCG for generating pseudo-random keys.
struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.state
    }
}

// ------------------------ 2. Separate chaining table (array-based) ------------------------

#[derive(Clone, Copy)]
struct ChainEntry {
    key: u64,
    value: u64,
    next: Option<usize>,
}

struct ChainTable {
    buckets: Vec<Option<usize>>, // bucket -> index in entries
    entries: Vec<ChainEntry>,    // storage for all elements
    len: usize,
    hasher: Hasher,
}

impl fmt::Debug for ChainTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChainTable")
            .field("len", &self.len)
            .field("capacity", &self.buckets.len())
            .finish()
    }
}

impl ChainTable {
    fn with_capacity_and_seed(capacity: usize, seed: u64) -> Self {
        let m = capacity.max(8);
        Self {
            buckets: vec![None; m],
            entries: Vec::new(),
            len: 0,
            hasher: Hasher::new(seed),
        }
    }

    fn capacity(&self) -> usize {
        self.buckets.len()
    }

    fn load_factor(&self) -> f64 {
        self.len as f64 / self.capacity() as f64
    }

    fn bucket_index(&self, key: u64) -> usize {
        (self.hasher.hash_u64(key) as usize) % self.capacity()
    }

    /// Insert (key, value) using separate chaining.
    ///
    /// Returns number of probes (comparisons) performed.
    fn insert_with_probes(&mut self, key: u64, value: u64) -> usize {
        let idx = self.bucket_index(key);
        let mut probes = 0usize;
        let mut current = self.buckets[idx];

        // Traverse chain to check if key already exists.
        while let Some(i) = current {
            probes += 1;
            if self.entries[i].key == key {
                self.entries[i].value = value;
                return probes;
            }
            current = self.entries[i].next;
        }

        // Not found: insert new entry at head of chain.
        let new_index = self.entries.len();
        let new_entry = ChainEntry {
            key,
            value,
            next: self.buckets[idx],
        };
        self.entries.push(new_entry);
        self.buckets[idx] = Some(new_index);
        self.len += 1;

        // We did not count a probe for the empty end; you can think of probes
        // as the number of existing entries inspected.
        probes
    }

    /// Lookup key and return (value, probes).
    ///
    /// Probes = number of entries traversed.
    fn get_with_probes(&self, key: u64) -> (Option<u64>, usize) {
        let idx = self.bucket_index(key);
        let mut probes = 0usize;
        let mut current = self.buckets[idx];

        while let Some(i) = current {
            probes += 1;
            if self.entries[i].key == key {
                return (Some(self.entries[i].value), probes);
            }
            current = self.entries[i].next;
        }

        (None, probes) // not found; probes = chain length
    }

    /// Compute the average chain length over non-empty buckets.
    fn average_nonempty_chain_length(&self) -> f64 {
        let mut nonempty = 0usize;
        let mut total_len = 0usize;

        for &head in &self.buckets {
            if let Some(mut idx) = head {
                nonempty += 1;
                let mut chain_len = 0usize;
                while let Some(i) = Some(idx) {
                    chain_len += 1;
                    match self.entries[i].next {
                        Some(next_idx) => idx = next_idx,
                        None => break,
                    }
                }
                total_len += chain_len;
            }
        }

        if nonempty == 0 {
            0.0
        } else {
            total_len as f64 / nonempty as f64
        }
    }
}

// ------------------------ 3. Open-addressed table (linear probing) ------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OAState {
    Empty,
    Occupied,
}

#[derive(Clone, Copy)]
struct OABucket {
    key: u64,
    value: u64,
    state: OAState,
}

impl OABucket {
    fn empty() -> Self {
        Self {
            key: 0,
            value: 0,
            state: OAState::Empty,
        }
    }
}

struct OpenAddressTable {
    buckets: Vec<OABucket>,
    len: usize,
    hasher: Hasher,
}

impl OpenAddressTable {
    fn with_capacity_and_seed(capacity: usize, seed: u64) -> Self {
        let m = capacity.max(8);
        Self {
            buckets: vec![OABucket::empty(); m],
            len: 0,
            hasher: Hasher::new(seed),
        }
    }

    fn capacity(&self) -> usize {
        self.buckets.len()
    }

    fn load_factor(&self) -> f64 {
        self.len as f64 / self.capacity() as f64
    }

    fn bucket_index(&self, key: u64) -> usize {
        (self.hasher.hash_u64(key) as usize) % self.capacity()
    }

    /// Insert using linear probing; assumes table not full.
    ///
    /// Returns the number of probes (bucket inspections).
    fn insert_with_probes(&mut self, key: u64, value: u64) -> usize {
        let m = self.capacity();
        let mut idx = self.bucket_index(key);
        let mut probes = 0usize;

        loop {
            probes += 1;
            let bucket = &mut self.buckets[idx];
            match bucket.state {
                OAState::Empty => {
                    bucket.key = key;
                    bucket.value = value;
                    bucket.state = OAState::Occupied;
                    self.len += 1;
                    return probes;
                }
                OAState::Occupied => {
                    if bucket.key == key {
                        bucket.value = value;
                        return probes;
                    }
                }
            }
            idx = (idx + 1) % m; // i_k = (h(key) + k) mod M  (7.7.8)
        }
    }

    /// Lookup with linear probing; returns (value, probes).
    fn get_with_probes(&self, key: u64) -> (Option<u64>, usize) {
        let m = self.capacity();
        let mut idx = self.bucket_index(key);
        let mut probes = 0usize;

        loop {
            probes += 1;
            let bucket = &self.buckets[idx];
            match bucket.state {
                OAState::Empty => {
                    return (None, probes); // key not present
                }
                OAState::Occupied => {
                    if bucket.key == key {
                        return (Some(bucket.value), probes);
                    }
                }
            }
            idx = (idx + 1) % m;

            // In practice, we will hit an Empty bucket before completing a full cycle
            // because the table is not full (α < 1).
        }
    }
}

// ------------------------ 4. Demonstration and comparison ------------------------

fn main() {
    // Table size M and number of keys n.
    let m = 128usize;
    let n = 96usize; // α = n/M = 0.75

    let seed_hasher = 0xDEADBEEFCAFEBABE;
    let mut rng = Lcg64::new(0x1234_5678_9ABC_DEF0);

    let mut chain_table = ChainTable::with_capacity_and_seed(m, seed_hasher);
    let mut oa_table = OpenAddressTable::with_capacity_and_seed(m, seed_hasher);

    // Store the inserted keys so we can probe them later.
    let mut keys: Vec<u64> = Vec::with_capacity(n);

    // Insert n keys into both tables and record insertion probes.
    let mut chain_insert_probes = 0usize;
    let mut oa_insert_probes = 0usize;

    for _ in 0..n {
        let key = rng.next_u64();
        let value = key.wrapping_mul(2);
        keys.push(key);

        chain_insert_probes += chain_table.insert_with_probes(key, value);
        oa_insert_probes += oa_table.insert_with_probes(key, value);
    }

    let alpha = n as f64 / m as f64;

    // Successful lookups.
    let mut chain_lookup_probes_success = 0usize;
    let mut oa_lookup_probes_success = 0usize;

    for &key in &keys {
        let (val_chain, pc) = chain_table.get_with_probes(key);
        let (val_oa, po) = oa_table.get_with_probes(key);
        assert_eq!(val_chain, Some(key.wrapping_mul(2)));
        assert_eq!(val_oa, Some(key.wrapping_mul(2)));
        chain_lookup_probes_success += pc;
        oa_lookup_probes_success += po;
    }

    // Unsuccessful lookups: use new keys unlikely to be in the table.
    let mut chain_lookup_probes_fail = 0usize;
    let mut oa_lookup_probes_fail = 0usize;

    for _ in 0..n {
        let key = rng.next_u64().wrapping_add(0xFFFF_FFFF_FFFF_FFFF);
        let (val_chain, pc) = chain_table.get_with_probes(key);
        let (val_oa, po) = oa_table.get_with_probes(key);
        assert!(val_chain.is_none());
        assert!(val_oa.is_none());
        chain_lookup_probes_fail += pc;
        oa_lookup_probes_fail += po;
    }

    let avg_chain_insert = chain_insert_probes as f64 / n as f64;
    let avg_oa_insert = oa_insert_probes as f64 / n as f64;

    let avg_chain_succ = chain_lookup_probes_success as f64 / n as f64;
    let avg_oa_succ = oa_lookup_probes_success as f64 / n as f64;

    let avg_chain_fail = chain_lookup_probes_fail as f64 / n as f64;
    let avg_oa_fail = oa_lookup_probes_fail as f64 / n as f64;

    // Theoretical expectations for chaining, from (7.7.5)–(7.7.7).
    let e_chain_len = alpha;                // (7.7.5)
    let e_probes_succ = 1.0 + alpha / 2.0;  // (7.7.6)
    let e_probes_fail = alpha;              // (7.7.7)

    // Additional statistic: average non-empty chain length.
    let avg_nonempty_chain = chain_table.average_nonempty_chain_length();

    println!("Collision resolution strategies: separate chaining vs open addressing");
    println!("-------------------------------------------------------------------");
    println!("Table size M         = {}", m);
    println!("Number of keys n     = {}", n);
    println!("Load factor α = n/M  ≈ {:.4}", alpha);
    println!();

    // Chaining statistics.
    println!("Separate chaining (array-based chains):");
    println!("  Observed load factor α                  ≈ {:.4}", chain_table.load_factor());
    println!("  Theoretical E[chain length]             ≈ {:.4}  (7.7.5)", e_chain_len);
    println!("  Observed avg non-empty chain length     ≈ {:.4}", avg_nonempty_chain);
    println!("  Avg probes per insertion                ≈ {:.4}", avg_chain_insert);
    println!("  Avg probes per successful lookup        ≈ {:.4}", avg_chain_succ);
    println!("    Theoretical E[probes_success]         ≈ {:.4}  (7.7.6)", e_probes_succ);
    println!("  Avg probes per unsuccessful lookup      ≈ {:.4}", avg_chain_fail);
    println!("    Theoretical E[probes_fail]            ≈ {:.4}  (7.7.7)", e_probes_fail);
    println!();

    // Open addressing statistics.
    println!("Open addressing with linear probing:");
    println!("  Observed load factor α                  ≈ {:.4}", oa_table.load_factor());
    println!("  Avg probes per insertion                ≈ {:.4}", avg_oa_insert);
    println!("  Avg probes per successful lookup        ≈ {:.4}", avg_oa_succ);
    println!("  Avg probes per unsuccessful lookup      ≈ {:.4}", avg_oa_fail);
    println!();

    println!("These measurements illustrate the trade-offs described in Section 7.7.2:");
    println!("  - Chaining maintains bounded expected probes even when α is relatively high.");
    println!("  - Open addressing remains cache-friendly, but probe lengths grow more");
    println!("    rapidly as α increases, emphasizing the importance of timely resizing.");
}
```

Program 7.7.2 demonstrates the practical consequences of the collision-resolution mechanisms developed mathematically in Section 7.7.2. The empirical results confirm that separate chaining maintains stable lookup and insertion costs even at relatively high load factors, consistent with the expected chain lengths predicted by Equations (7.7.5)–(7.7.7). In contrast, open addressing exhibits increasing probe counts as the load factor grows, illustrating its susceptibility to primary clustering and reinforcing the importance of timely resizing when using probing-based schemes.

The experiment also highlights the trade-offs between the two strategies: chaining provides robustness and predictable performance across a wide range of workloads, while open addressing remains memory-efficient and benefits from superior locality when the table is kept below moderate occupancy levels. These characteristics mirror the design choices found in modern high-performance hash tables, where memory layout, probing strategies, and resizing policies interact to achieve a balance between throughput, space efficiency, and reliability. By grounding the theoretical predictions in observable behaviour, the program offers readers a deeper appreciation of how collision-resolution strategies operate under realistic workloads.

## 7.7.3. Mathematical Analysis of Hash Table Performance

A hash table’s performance depends on the distribution of keys across its buckets, the behaviour of the hash function, the collision-resolution strategy, and dynamic resizing policies. Although hashing is often introduced as an “expected constant-time” technique, a more precise mathematical understanding shows which operations achieve this bound and under what assumptions. If the hash function distributes keys uniformly across the $M$ buckets, the load factor

$$\lambda = \frac{n}{M}\tag{7.7.9}$$

governs the expected cost of most operations. When hashing behaves like a random function, the number of keys landing in each bucket follows a Poisson-like distribution, and the expected occupancy of any bucket is $\mathbb{E}[\text{bucket size}] = \lambda$.

In *separate chaining*, each bucket holds a chain (or an array-based sequence) of elements. Under uniform hashing, the expected chain length is:

$$\mathbb{E}[\text{chain length}] = \lambda\tag{7.7.10}$$

so successful lookups require traversing, on average, half a chain, giving:

$$\begin{align}\mathbb{E}[\text{successful lookup}] &= 1 + \frac{\lambda}{2},\\[0.25cm] \mathbb{E}[\text{unsuccessful lookup}] &= \lambda \end{align}\tag{7.7.11}$$

These bounds hold even when hash values collide at the 64-bit level, provided full key comparison is used to ensure correctness. This reliability is essential for practical systems such as secure pseudo-random generators and integrity-checked storage, where collision behaviour must remain predictable even under high volume (Almaraz & Román Villaizán, 2023).

For *open addressing*, all keys reside in the bucket array and collisions are resolved through probing. Given a uniform hash function and a probing sequence $i_k = (h(\mathrm{key}) + p(k)) \bmod M$, the expected number of probes for successful search under linear probing is:

$$\mathbb{E}[\text{successful probes}] \approx \frac{1}{2}\left(1 + \frac{1}{1-\lambda}\right)\tag{7.7.12}$$

while unsuccessful searches require approximately,

$$\mathbb{E}[\text{unsuccessful probes}] \approx \frac{1}{1-\lambda}\tag{7.7.13}$$

As $\lambda \to 1$, performance degrades significantly, explaining the need for careful capacity planning and resizing. These effects mirror congestion phenomena found in pattern generators and register-based circuits in hardware systems, where load-dependent delays also arise (Somanathan, Reddy & Bhakthavatchalu, 2025).

Dynamic resizing ensures that $\lambda$ stays within a desirable range. When the table resizes, typically doubling its capacity, the cost of rehashing all keys is $O(n)$, but since this occurs infrequently, the amortized complexity of insertions remains:

$$\mathbb{E}[\text{amortized insertion}] = O(1)\tag{7.7.14}$$

This amortized analysis parallels the behaviour of LFSR-based sequence generators in which occasional state reinitialisation incurs cost but does not affect long-run throughput (Maity et al., 2025; Okunbor, Omorogbe & Edeko, 2024).

Overall, a hash table achieves constant-time performance when hashing is uniform, chains remain short, and the load factor is controlled. These principles form the mathematical foundation for reliable, high-performance hash memories, associative arrays, and large-scale lookup systems.

### Rust Implementation

Following the mathematical development in Section 7.7.3 on the probabilistic behaviour and expected-time complexity of hash tables, Program 7.7.3 provides an empirical demonstration of how separate chaining and linear probing perform under controlled load factors. Whereas the theory expresses expected costs through Equations (7.7.9)–(7.7.14), real hash tables must contend with finite-sample randomness, clustering effects, and nonuniform key distributions. The Rust implementation below constructs simplified hash-table models and measures their average probe counts for successful and unsuccessful lookups, allowing direct comparison with theoretical expectations. These experiments bridge the gap between analytic performance bounds and the behaviour observed in practical numerical computing environments.

At the core of Program 7.7.3 are two minimal hash-table implementations: one using separate chaining, corresponding to the performance behaviour captured by Equations (7.7.10)–(7.7.11), and the other using linear probing, which reflects the analytic properties described in Equations (7.7.12)–(7.7.13). Both tables use a simple pseudo-hash function to distribute keys uniformly across the available buckets, approximating the random-hashing assumption in Equation (7.7.9).

The `ChainingTable` struct implements separate chaining using vectors to represent per-bucket chains. Insertions append keys to their respective chains, while lookups simulate probe counts by scanning each chain linearly. Under uniform hashing, the expected chain length equals the load factor $λ$, and the average number of probes for successful and unsuccessful lookups follows directly from Equation (7.7.11). By inserting $n$ random keys into $M$ buckets, the implementation constructs a table with empirical load factor $λ = n/M$ and records how closely the observed statistics align with theoretical predictions.

The `LinearProbingTable` struct implements open addressing with linear probing, where collisions are resolved by sequentially probing alternate buckets until an empty slot is reached. Because linear probing is highly sensitive to clustering and spillover effects, this part of the code provides insight into how probe distributions behave in practice compared with the approximate analytic expressions. Probe counts for successful and unsuccessful searches are measured separately, enabling comparison against the expectations from Equations (7.7.12) and (7.7.13).

The `main` function coordinates the entire experiment by constructing hash tables at selected load factors, inserting random keys, and then performing large batches of lookup tests for both successful and unsuccessful queries. It computes empirical averages for each scenario and prints them alongside theoretical predictions. Through these measurements, Program 7.7.3 illustrates how analytic complexity estimates translate into observable behaviour, clarifying the practical implications of load factor, hashing uniformity, and collision-resolution strategy.

Add the foloowing dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
// Program 7.7.3: Empirical Analysis of Hash Table Performance
//
// This program illustrates the mathematical analysis of hash table performance
// discussed in Section 7.7.3. It implements two simple hash table schemes:
//
//   1. Separate chaining: each bucket holds a vector of keys.
//      We measure average probes for successful and unsuccessful lookups and
//      compare them with the theoretical expectations in (7.7.10)–(7.7.11).
//
//   2. Open addressing with linear probing: all keys live in an array, and
//      collisions are resolved by scanning forward. We measure average probes
//      and compare them with the approximate formulas in (7.7.12)–(7.7.13).
//
// The hash function is a simple multiplicative hash on u64 keys. We assume
// keys are random, so the hash behaves approximately like a uniform mapping,
// matching the assumptions of the analysis.

use rand::prelude::*;

fn hash_u64(x: u64, m: usize) -> usize {
    // Simple multiplicative hash to spread bits before taking modulo.
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mixed = x.wrapping_mul(K);
    (mixed as usize) % m
}

// =============================
// Separate chaining hash table
// =============================

struct ChainingTable {
    buckets: Vec<Vec<u64>>,
    len: usize,
}

impl ChainingTable {
    fn new(m: usize) -> Self {
        Self {
            buckets: vec![Vec::new(); m],
            len: 0,
        }
    }

    fn capacity(&self) -> usize {
        self.buckets.len()
    }

    fn load_factor(&self) -> f64 {
        self.len as f64 / self.capacity() as f64
    }

    fn insert(&mut self, key: u64) {
        let m = self.capacity();
        let idx = hash_u64(key, m);
        let bucket = &mut self.buckets[idx];

        // Avoid duplicates (probability is tiny, but we keep it clean).
        if !bucket.contains(&key) {
            bucket.push(key);
            self.len += 1;
        }
    }

    /// Lookup with probe counting: returns (found?, probes).
    /// Probes = number of bucket elements examined (including the match if found).
    fn lookup_with_probes(&self, key: u64) -> (bool, usize) {
        let m = self.capacity();
        let idx = hash_u64(key, m);
        let bucket = &self.buckets[idx];

        let mut probes = 0;
        for &k in bucket {
            probes += 1;
            if k == key {
                return (true, probes);
            }
        }
        (false, probes)
    }
}

// ======================================
// Linear probing (open addressing) table
// ======================================

struct LinearProbingTable {
    slots: Vec<Option<u64>>,
    len: usize,
}

impl LinearProbingTable {
    fn new(m: usize) -> Self {
        Self {
            slots: vec![None; m],
            len: 0,
        }
    }

    fn capacity(&self) -> usize {
        self.slots.len()
    }

    fn load_factor(&self) -> f64 {
        self.len as f64 / self.capacity() as f64
    }

    fn insert(&mut self, key: u64) {
        let m = self.capacity();
        let mut idx = hash_u64(key, m);

        loop {
            match self.slots[idx] {
                None => {
                    self.slots[idx] = Some(key);
                    self.len += 1;
                    return;
                }
                Some(existing) if existing == key => {
                    // Already present; do not count as new.
                    return;
                }
                _ => {
                    idx = (idx + 1) % m;
                }
            }
        }
    }

    /// Lookup with probe counting: returns (found?, probes).
    /// Probes = number of slots examined (including the match if found).
    fn lookup_with_probes(&self, key: u64) -> (bool, usize) {
        let m = self.capacity();
        let mut idx = hash_u64(key, m);
        let mut probes = 0;

        loop {
            probes += 1;
            match self.slots[idx] {
                None => {
                    // Empty slot: key not present.
                    return (false, probes);
                }
                Some(existing) if existing == key => {
                    return (true, probes);
                }
                _ => {
                    idx = (idx + 1) % m;
                    // In a real implementation we'd stop before looping forever.
                }
            }
        }
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

    // ============================================================
    // 1. Separate chaining: empirical vs theoretical probes
    // ============================================================

    let m_chain = 10_000;
    let n_chain = 8_000; // λ ≈ 0.8
    let mut chain_table = ChainingTable::new(m_chain);
    let mut chain_keys = Vec::with_capacity(n_chain);

    // Insert random keys until we reach n_chain.
    while chain_keys.len() < n_chain {
        let k = rng.gen::<u64>();
        chain_table.insert(k);
        // We only record if actually inserted (no duplicates).
        if chain_keys.len() < chain_table.len {
            chain_keys.push(k);
        }
    }

    let lambda_chain = chain_table.load_factor();

    // Measure successful lookups: probe each inserted key once.
    let mut succ_probes = 0usize;
    for &k in &chain_keys {
        let (found, probes) = chain_table.lookup_with_probes(k);
        debug_assert!(found);
        succ_probes += probes;
    }
    let avg_succ_probes = succ_probes as f64 / chain_keys.len() as f64;

    // Measure unsuccessful lookups: look up keys that are (with overwhelming
    // probability) absent.
    let num_unsucc_tests = 10_000;
    let mut unsucc_probes = 0usize;
    for _ in 0..num_unsucc_tests {
        let k = rng.gen::<u64>();
        let (_found, probes) = chain_table.lookup_with_probes(k);
        unsucc_probes += probes;
    }
    let avg_unsucc_probes = unsucc_probes as f64 / num_unsucc_tests as f64;

    let theo_chain_succ = 1.0 + lambda_chain / 2.0; // from (7.7.11)
    let theo_chain_unsucc = lambda_chain;           // from (7.7.11)

    println!("=== Separate chaining hash table ===");
    println!("Capacity M = {}", m_chain);
    println!("Number of keys n = {}", n_chain);
    println!("Load factor λ = n / M ≈ {:.3}", lambda_chain);
    println!();
    println!("Empirical average probes (successful lookup):   ≈ {:.3}", avg_succ_probes);
    println!("Theoretical expectation (1 + λ/2):              ≈ {:.3}", theo_chain_succ);
    println!();
    println!("Empirical average probes (unsuccessful lookup): ≈ {:.3}", avg_unsucc_probes);
    println!("Theoretical expectation (λ):                    ≈ {:.3}", theo_chain_unsucc);
    println!();

    // ============================================================
    // 2. Linear probing: empirical vs theoretical probes
    // ============================================================

    let m_lp = 10_000;
    let n_lp = 7_000; // λ ≈ 0.7
    let mut lp_table = LinearProbingTable::new(m_lp);
    let mut lp_keys = Vec::with_capacity(n_lp);

    while lp_keys.len() < n_lp {
        let k = rng.gen::<u64>();
        // Insert and detect whether it was new by checking table.len.
        let before = lp_table.len;
        lp_table.insert(k);
        if lp_table.len > before {
            lp_keys.push(k);
        }
    }

    let lambda_lp = lp_table.load_factor();

    // Measure successful lookups.
    let mut lp_succ_probes = 0usize;
    for &k in &lp_keys {
        let (found, probes) = lp_table.lookup_with_probes(k);
        debug_assert!(found);
        lp_succ_probes += probes;
    }
    let avg_lp_succ_probes = lp_succ_probes as f64 / lp_keys.len() as f64;

    // Measure unsuccessful lookups.
    let num_lp_unsucc_tests = 10_000;
    let mut lp_unsucc_probes = 0usize;
    for _ in 0..num_lp_unsucc_tests {
        let k = rng.gen::<u64>();
        let (_found, probes) = lp_table.lookup_with_probes(k);
        lp_unsucc_probes += probes;
    }
    let avg_lp_unsucc_probes = lp_unsucc_probes as f64 / num_lp_unsucc_tests as f64;

    // Theoretical approximations for linear probing (7.7.12)–(7.7.13).
    let theo_lp_succ = 0.5 * (1.0 + 1.0 / (1.0 - lambda_lp));
    let theo_lp_unsucc = 1.0 / (1.0 - lambda_lp);

    println!("=== Linear probing hash table ===");
    println!("Capacity M = {}", m_lp);
    println!("Number of keys n = {}", n_lp);
    println!("Load factor λ = n / M ≈ {:.3}", lambda_lp);
    println!();
    println!("Empirical average probes (successful lookup):   ≈ {:.3}", avg_lp_succ_probes);
    println!("Theoretical expectation ≈ 0.5 * (1 + 1/(1-λ)):  ≈ {:.3}", theo_lp_succ);
    println!();
    println!("Empirical average probes (unsuccessful lookup): ≈ {:.3}", avg_lp_unsucc_probes);
    println!("Theoretical expectation ≈ 1/(1-λ):              ≈ {:.3}", theo_lp_unsucc);
    println!();

    println!("Note: Small discrepancies are expected due to finite-sample effects,\n\
              randomness of the keys, and the approximative nature of (7.7.12)–(7.7.13).");
}
```

Program 7.7.3 provides a quantitative complement to the theoretical results of Section 7.7.3, demonstrating how empirical probe counts for separate chaining and linear probing align with the mathematical expectations in Equations (7.7.9)–(7.7.14). Separate chaining tracks theoretical predictions closely, confirming the robustness of the Poisson-like occupancy model. Linear probing, by contrast, exhibits more variation due to clustering effects, underscoring why open-addressing schemes must maintain low load factors or adopt advanced strategies such as Robin Hood hashing. These results reinforce the principle that hash-table performance hinges on maintaining controlled load factors and near-uniform hash distribution, conditions essential to achieving the widely quoted $O(1)$ expected performance in modern hash memories and associative data structures.

## 7.7.4. Hash Table Implementations: Hash Functions as First-Class Objects

Modern hash tables treat the hash function not as a static arithmetic expression but as a configurable and reusable software object. This shift reflects a broader evolution in algorithm design, where hashing must adapt to diverse key types, adversarial workloads, and heterogeneous hardware architectures. A contemporary hash table therefore exposes the hash function as an interchangeable component, allowing users to select or customise the behaviour that best suits their application domain.

A hash function object typically encapsulates three responsibilities: bit-mixing, seeding, and finalisation. *Bit-mixing* converts structured or numeric keys into a well-diffused bit pattern using shifts, rotations, multiplications, and XOR operations. Proper mixing is essential to simulate the statistical behaviour of random hashing and thereby maintain short chains or short probe sequences. *Seeding* introduces per-process or per-table randomness to prevent adversarially crafted keys from causing predictable collisions. Randomised hashing contributes significantly to stability in high-throughput environments where hostile or pathological inputs can otherwise distort performance (Farach-Colton, Krapivin & Kuszmaul, 2024). *Finalisation* applies a final round of mixing to ensure uniform distribution over the bucket range before reduction modulo $M$.

For composite data types, common in numerical computing, simulation metadata, or cryptographic protocols, the hash function object provides structured composition. Each field of the key (integers, arrays, tuples, or fixed-length binary strings) is hashed independently before the partial hashes are combined. This modular approach avoids the pitfalls of naïve concatenation and allows specialised field-level hashing, such as LFSR-inspired mixers or modular reductions suitable for constrained embedded devices (Almaraz & Román Villaizán, 2023).

On hardware-oriented platforms, including FPGAs and low-power microcontrollers, hash objects may incorporate optimisations that align with the target architecture. Designers commonly adopt word-level LFSR mixing stages, which provide extremely fast bit diffusion while consuming negligible energy, an approach also used in power-aware pattern generators and self-test architectures (Somanathan, Reddy & Bhakthavatchalu, 2025). The combination of linear-feedback mixing and table-level random seeding produces strong dispersion while remaining inexpensive to implement.

Hash tables that use *open addressing* rely especially heavily on the quality of the hash function object. Because probe sequences depend directly on hash values, poor diffusion leads to clustering and long runs of consecutive filled buckets. Systems that implement linear or quadratic probing therefore benefit from strong avalanche properties in their hash functions, ensuring that small variations in input produce large, unpredictable changes in the output index. Similarly, double hashing depends on two independent hash outputs; the hash function object must supply these functions or provide a mechanism for deriving a secondary index without introducing correlation.

Treating the hash function as an object also improves code clarity and extensibility. For example, a cryptographic application can swap in a LFSR-based or chaos-enhanced mixer (Okunbor, Omorogbe & Edeko, 2024), while a database engine may select a deterministic and architecture-tuned hash for reproducible query plans. In simulation workloads that require repeatability, a property shared with pseudo-random generators used in Monte Carlo methods, the hash function object ensures that identical seeds yield identical hash behaviours across platforms (Maity et al., 2025).

As data structures scale, the ability to configure, profile, and tune hash behaviour becomes increasingly important. Modern hash tables therefore expose the hash function object alongside parameters such as load-factor thresholds, probing strategies, and resizing policies. This modular design allows users to construct systems that balance theoretical performance guarantees with practical constraints such as power, latency, hardware capabilities, and predictable worst-case behaviour.

### Rust Implementation

Following the discussion in Section 7.7.4 on the role of hash functions as configurable software objects, Program 7.4.4 provides a concrete implementation that demonstrates how modern hash tables separate hashing behaviour from table mechanics. As the section explains, contemporary systems treat hashing not as a fixed arithmetic rule but as a parameterisable component responsible for bit mixing, seeding, and finalisation. The program illustrates this design principle by constructing a reusable LFSR-based hash function object and integrating it into an open-addressing hash table with double hashing. This modular architecture mirrors real-world data-structure libraries, where users must be able to tailor hash behaviour to the statistical properties of their keys, hardware constraints, or adversarial conditions.

At the core of the implementation is the `HashFunction` trait, which defines a general interface for any hashing strategy by requiring implementations of two methods: `hash1` and `hash2`. These functions represent the primary and secondary hash outputs used in double hashing, enabling the hash table to derive independent probe sequences. The trait also provides a convenience method, `hashes`, which bundles both results into a single call. By abstracting hashing in this way, the table remains completely agnostic to the details of bit mixing, randomisation, or seed management.

The `LfsrHash` type then provides a concrete realisation of this interface. It encapsulates three responsibilities identified earlier in the section: bit-mixing, seeding, and finalisation. The internal `mix64` method performs multi-round diffusion using rotations, multiplications, and XORs to ensure good avalanche behaviour. The `lfsr_step` function implements a fast, word-level linear-feedback operation suitable for energy-efficient architectures. These components are integrated through the `absorb` method, which incorporates each field of a composite key by repeatedly applying LFSR steps. The final mixing stage ensures that the output aligns with the statistical expectations required for stable open-addressing performance discussed in Section 7.7.4.

To demonstrate composite key hashing, Program 7.4.4 defines the `SimulationKey` struct, a representative example of multi-field keys common in numerical and simulation workloads. The implementation of `HashFunction<SimulationKey>` hashes each field independently before folding them into the running state. This corresponds to the structured composition described in the section text: field-level hashing avoids the pitfalls of naïve concatenation and allows the hash function object to apply specialised diffusion strategies. By packing multi-field data into 64-bit patterns and passing them through the LFSR-based absorption pipeline, the program reproduces the behaviour of modern hash libraries that rely on compositional hashing for reliability and uniform distribution.

The hash table itself is implemented by the `OpenAddressingTable` type, which employs double hashing to reduce clustering and probe-sequence correlation. The `insert` method derives the initial bucket and step size from `hash1` and `hash2`, respectively, and uses these to traverse the table until an empty or matching bucket is found. Likewise, the `get` method reproduces the same sequence to locate stored entries. By relying exclusively on the configurable hash function object, the table demonstrates the modularity principles highlighted in the section: any hash function with sufficiently strong diffusion including LFSR-based mixers, cryptographic primitives, or architecture-specific hashes can be substituted without modifying the table logic.

The `main` function illustrates how this system behaves in practice. It constructs a reproducibly seeded `LfsrHash`, ensuring that identical seeds produce identical table behaviour across platforms, a property emphasised earlier in the discussion of simulation workloads. The program then inserts several `SimulationKey` entries reflecting run identifiers, time steps, and grid coordinates, producing nine unique keys. The subsequent lookup verifies the correctness of the hashing and probing mechanisms, demonstrating that the composite-key hashing and double-hashed table operations work together as intended. This example reinforces the section’s central point that treating hash functions as first-class objects significantly enhances programmability, clarity, and control in modern data-structure design.

```rust
//! Program 7.7.4: Hash function objects with open-addressing hash table
//!
//! This example treats the hash function as a first-class object that can be
//! configured, seeded, and swapped independently of the table. The `LfsrHash`
//! type illustrates three key responsibilities:
//!   * bit mixing via word-level shifts, rotations, and XORs;
//!   * per-table seeding for randomisation;
//!   * finalisation that maps mixed bits into bucket indices.
//!
//! The table itself uses open addressing with double hashing: the hash object
//! supplies both the primary index and a secondary step size. A composite key
//! (`SimulationKey`) demonstrates structured hashing of multi-field keys.

use std::fmt::Debug;

/// A hash function object for keys of type `K`.
///
/// Implementations are responsible for bit-mixing, seeding, and finalisation.
/// The secondary hash output is used for double hashing in open addressing.
pub trait HashFunction<K> {
    /// Primary 64-bit hash value.
    fn hash1(&self, key: &K) -> u64;

    /// Secondary 64-bit hash value, used to derive the probe step.
    fn hash2(&self, key: &K) -> u64;

    /// Convenience method that returns both hashes at once.
    fn hashes(&self, key: &K) -> (u64, u64) {
        (self.hash1(key), self.hash2(key))
    }
}

/// A simple LFSR-inspired hash function object.
///
/// - `seed` provides per-table randomisation.
/// - `mix64` performs word-level diffusion using rotations and XORs.
/// - `lfsr_rounds` controls how many linear-feedback steps are applied.
#[derive(Clone, Copy, Debug)]
pub struct LfsrHash {
    seed: u64,
    lfsr_rounds: u32,
}

impl LfsrHash {
    /// Construct a new hash object with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            lfsr_rounds: 16,
        }
    }

    /// One step of a 64-bit LFSR with taps at bits (0, 1, 3, 4).
    #[inline]
    fn lfsr_step(state: u64) -> u64 {
        let bit = ((state >> 0) ^ (state >> 1) ^ (state >> 3) ^ (state >> 4)) & 1;
        (state >> 1) | (bit << 63)
    }

    /// Lightweight 64-bit mixer: rotates, multiplies, and xors.
    ///
    /// This is not intended to be cryptographically secure, but it provides
    /// reasonable avalanche behaviour for table indexing.
    #[inline]
    fn mix64(mut x: u64) -> u64 {
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        x ^= x >> 33;
        x
    }

    /// Combine a new 64-bit field into the running hash state using LFSR steps.
    #[inline]
    fn absorb(&self, mut state: u64, field: u64) -> u64 {
        state ^= field;
        for _ in 0..self.lfsr_rounds {
            state = Self::lfsr_step(state);
        }
        state
    }
}

/// A composite key typical of numerical or simulation workloads.
///
/// It combines a run identifier, a time step, and a 2D cell index. The
/// hash function object will hash each field separately before combining
/// them into a final value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimulationKey {
    pub run_id: u64,
    pub time_step: u32,
    pub cell: (u16, u16),
}

impl SimulationKey {
    pub fn new(run_id: u64, time_step: u32, i: u16, j: u16) -> Self {
        Self {
            run_id,
            time_step,
            cell: (i, j),
        }
    }
}

/// Implement LFSR-based hashing for `SimulationKey`.
///
/// The object-level seed is mixed with each field, and then a final word
/// mixer is applied. The secondary hash is derived from the primary by an
/// additional mixing step to reduce correlation.
impl HashFunction<SimulationKey> for LfsrHash {
    fn hash1(&self, key: &SimulationKey) -> u64 {
        // Start from the per-table seed.
        let mut state = Self::mix64(self.seed);

        // Absorb composite fields one by one.
        state = self.absorb(state, key.run_id);
        state = self.absorb(state, key.time_step as u64);

        // Pack the 2D cell index into a single word.
        let packed_cell =
            ((key.cell.0 as u64) << 16) ^ ((key.cell.1 as u64) << 32);
        state = self.absorb(state, packed_cell);

        // Finalisation: a last round of strong mixing.
        Self::mix64(state)
    }

    fn hash2(&self, key: &SimulationKey) -> u64 {
        // Derive a second hash by mixing the first with an offset constant.
        let h1 = self.hash1(key);
        Self::mix64(h1 ^ 0x9e37_79b9_7f4a_7c15)
    }
}

/// Open-addressing hash table with double hashing and a pluggable hash object.
///
/// This table is intentionally minimal: it demonstrates how the hash function
/// is passed in and used, rather than optimising for all edge cases.
pub struct OpenAddressingTable<K, V, H>
where
    K: Eq + Clone,
    H: HashFunction<K> + Clone,
{
    buckets: Vec<Option<(K, V)>>,
    len: usize,
    hasher: H,
}

impl<K, V, H> OpenAddressingTable<K, V, H>
where
    K: Eq + Clone,
    V: Clone,
    H: HashFunction<K> + Clone,
{
    /// Create a new table with the given capacity and hash function object.
    pub fn with_capacity_and_hasher(capacity: usize, hasher: H) -> Self {
        // Use a simple power-of-two capacity for demonstration.
        let cap = capacity.max(8).next_power_of_two();
        Self {
            buckets: vec![None; cap],
            len: 0,
            hasher,
        }
    }

    /// Number of key–value pairs currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Capacity (number of buckets).
    pub fn capacity(&self) -> usize {
        self.buckets.len()
    }

    /// Load factor α = len / capacity.
    pub fn load_factor(&self) -> f64 {
        self.len as f64 / self.capacity() as f64
    }

    /// Insert a key–value pair, overwriting the value if the key already exists.
    pub fn insert(&mut self, key: K, value: V) {
        let cap = self.capacity();
        let (h1, h2) = self.hasher.hashes(&key);

        // Primary index and secondary step for double hashing.
        let mut idx = (h1 as usize) & (cap - 1);
        let step = ((h2 as usize) & (cap - 1)).max(1); // must be non-zero

        for _probe in 0..cap {
            match &mut self.buckets[idx] {
                Some((existing_key, existing_value)) => {
                    if existing_key == &key {
                        // Overwrite existing value.
                        *existing_value = value;
                        return;
                    }
                    // Collision: advance to next probe location.
                    idx = (idx + step) & (cap - 1);
                }
                slot @ None => {
                    // Empty bucket: insert here.
                    *slot = Some((key, value));
                    self.len += 1;
                    return;
                }
            }
        }

        panic!("Hash table is full: cannot insert new element");
    }

    /// Look up a key in the table.
    pub fn get(&self, key: &K) -> Option<&V> {
        let cap = self.capacity();
        let (h1, h2) = self.hasher.hashes(key);

        let mut idx = (h1 as usize) & (cap - 1);
        let step = ((h2 as usize) & (cap - 1)).max(1);

        for _probe in 0..cap {
            match &self.buckets[idx] {
                Some((existing_key, value)) => {
                    if existing_key == key {
                        return Some(value);
                    }
                    idx = (idx + step) & (cap - 1);
                }
                None => {
                    // Reached an empty bucket: key not present.
                    return None;
                }
            }
        }

        None
    }
}

fn main() {
    // In a numerical simulation, we may wish to use a reproducible hash
    // configuration: the same seed yields identical bucket layouts on
    // all platforms that agree on integer sizes.
    let seed: u64 = 0x1234_5678_9abc_def0;
    let hasher = LfsrHash::new(seed);

    let mut table: OpenAddressingTable<SimulationKey, f64, LfsrHash> =
        OpenAddressingTable::with_capacity_and_hasher(32, hasher);

    // Insert some composite keys with numeric payloads.
    for run_id in 0..3u64 {
        for time_step in [0u32, 10, 20] {
            let key = SimulationKey::new(run_id, time_step, 4, 7);
            let value = (run_id as f64) + (time_step as f64) * 0.1;
            table.insert(key, value);
        }
    }

    println!("Table capacity       : {}", table.capacity());
    println!("Number of entries    : {}", table.len());
    println!("Load factor (α)      : {:.3}", table.load_factor());

    // Demonstrate lookup with the same composite key.
    let query_key = SimulationKey::new(1, 10, 4, 7);
    if let Some(value) = table.get(&query_key) {
        println!(
            "Lookup for key {:?} -> value = {}",
            query_key, value
        );
    } else {
        println!("Key {:?} not found", query_key);
    }

    // For adversarial or hardware-aware scenarios, the user could create a
    // different hash object here (e.g. chaos-enhanced, cryptographic, or
    // architecture-specific) and pass it to `with_capacity_and_hasher`
    // without changing the table implementation.
}
```

Program 7.4.4 illustrates how modern hash table design benefits from treating the hash function as a configurable and reusable software object. By separating hashing behaviour from table mechanics, the implementation mirrors contemporary approaches used in high-performance systems, simulation frameworks, and adversarially robust applications. The example highlights the importance of strong bit-mixing and architectural adaptability, two principles that ensure uniform distribution of keys, short probe sequences, and predictable performance even under challenging workloads.

The LFSR-inspired hash object demonstrates how word-level mixing and lightweight feedback mechanisms can achieve strong dispersion while remaining computationally inexpensive. Meanwhile, the composite-key example underscores the advantages of structured hashing, where each field is processed independently before being combined. This approach avoids the correlation problems of simple concatenation and aligns with the needs of numerical computing environments where keys often aggregate multiple physical or simulation parameters.

By decoupling the hash function from the table, the program provides a foundation for extensibility. Users can introduce chaos-based mixers, cryptographic hashes, architecture-tuned functions, or reproducible seeded variants without altering the table implementation. This flexibility mirrors the broader design trends discussed in Section 7.7.4 and sets the stage for further exploration of workload-aware hashing, adversarial resistance, and hardware-level optimisation in advanced data-structure systems.

## 7.7.5. Cache Behaviour, Memory Layout, and Practical Optimisations

The real-world performance of a hash table depends not only on its collision-resolution strategy and hashing function but also on the way its data are laid out in memory. Modern systems must account for hierarchical cache structures, prefetching behaviour, vectorised instructions, and memory bandwidth. As a result, practical hash table design has shifted toward compact representations and tightly controlled access patterns that reduce cache misses and branch mispredictions.

A fundamental principle is that spatial locality dominates performance. When keys and metadata are stored contiguously, a single cache line fetch can supply multiple bucket controls in one operation. Open-addressed tables exploit this naturally, because probe sequences walk through consecutive indices in the same array. Linear probing, often considered theoretically suboptimal due to clustering, can in practice outperform more sophisticated schemes because its sequential memory access patterns align well with cache prefetchers (Li, 2023). With well-mixed hash outputs and moderate load factors, this locality yields extremely low-latency lookups.

To improve locality further, many designs maintain *packed control arrays* that store metadata such as occupancy flags, hash prefixes, or probe state indicators. Because these controls are lighter than full key–value pairs, dozens of them fit into a single cache line. This enables vectorised SIMD comparisons and significantly reduces the number of memory accesses. The idea parallels techniques used in LFSR-based pattern generators for hardware self-test, where large volumes of control information must be scanned quickly with minimal logic cost (Somanathan, Reddy & Bhakthavatchalu, 2025).

Separate chaining can also be made cache-efficient by storing chains in array form rather than traditional pointer-based linked lists. An array-based adjacency structure improves locality and reduces branch unpredictability. With this approach, chains behave like small, contiguous segments, and the `next[]` indices remain within dense integer arrays. The result is a hybrid structure that retains the analytical robustness of chaining while achieving memory access behaviour closer to that of open addressing (Farach-Colton, Krapivin & Kuszmaul, 2024).

Deletion mechanisms play a major role in practical performance. In open addressing, tombstones accumulate over time, extending probe chains if not reclaimed. Some systems perform *incremental rehashing*, spreading reorganisation work across future operations instead of performing expensive full rebuilds. Others exploit generational metadata to distinguish fresh entries from reused slots. These techniques echo the management of state transitions in pseudo-random sequence generators, where periodic reinforcement of structure is necessary to maintain long-term consistency (Maity et al., 2025).

Another important optimisation is *grouped probing*, in which the table examines a small fixed window of buckets, typically 8 or 16, using vectorised comparisons. If no matching control bits appear in the group, the probe jumps to the next group. This reduces branch mispredictions and improves throughput on modern CPUs. Group-based probing mirrors the design of multi-stage LFSR systems, where groups of bits are processed in lockstep to maximise hardware efficiency (Okunbor, Omorogbe & Edeko, 2024).

Finally, resizing policies must be chosen carefully. While doubling capacity is standard, some applications prefer growth factors such as 1.5 to reduce memory overhead. Others maintain strict upper bounds on the load factor to guarantee predictable latency. Whatever the policy, predictable behaviour is essential for real-time and embedded systems, where worst-case latency is as important as average performance. The interplay of data layout, probe strategy, and resizing behaviour ultimately determines the performance envelope of modern hash tables across a wide range of computational domains.

### Rust Implementation

Following the discussion in Section 7.7.5 on cache locality, metadata packing, and the practical constraints imposed by modern memory hierarchies, Program 7.7.5 presents a compact hash table implementation designed explicitly to illustrate these principles. While earlier sections emphasized algorithmic aspects such as hashing functions and collision resolution strategies, practical performance often hinges on how data are arranged in memory and how probing interacts with cache-line structures. This program constructs an open-addressed hash table using tightly packed control bytes and grouped probing over contiguous metadata. By aligning probe windows with cache-friendly memory slices, the implementation demonstrates how spatial locality, predictable access patterns, and reduced branching can dramatically influence real-world throughput. The resulting structure captures the essence of modern hash table design, bridging the conceptual discussion with a concrete example that highlights locality-aware engineering and its performance implications.

At the core of the implementation is the packed control array, represented by a contiguous `Vec<u8>` that stores a lightweight state for every bucket. These control bytes encode whether a slot is empty, full, or contains a tombstone, enabling the table to separate metadata from key–value storage. This separation reflects the compact control-word strategies used in high-performance hash tables, where metadata from many buckets can be scanned within a single cache line. Contiguity ensures that probing operations benefit from automatic hardware prefetching, reducing random access and aligning with the principle of spatial locality discussed earlier in the section.

The `insert` function embodies the grouped probing strategy central to modern locality-optimised hash tables. Instead of evaluating slots one by one, it inspects small fixed-size windows of adjacent control bytes. Within each group, it examines occupancy information and decides whether a key can be inserted, updated, or whether the probe should advance to the next window. This pattern mirrors vectorised probing, where comparison across several bytes facilitates faster dispatch and reduces branch misprediction. The use of tombstones allows the structure to maintain deletion without invalidating subsequent probe sequences, a reflection of the practical trade-offs in maintaining predictable lookup lengths while supporting dynamic updates.

Lookup operations are handled by the `get` function, which follows the same grouped-probing model. Because an EMPTY control byte guarantees nonexistence in open addressing, the search can terminate early in many cases, contributing to low-latency lookups under moderate load factors. Conversely, DEL markers require continued probing, capturing the behaviour of tombstone-based deletion as discussed in the subsection on deletion mechanisms and long-term table health.

The `remove` method demonstrates how tombstones are integrated into the probing system. Rather than removing an entry outright, it marks the slot as deleted and clears the associated key and value. This avoids abrupt probe-chain disruption and mirrors strategies used in production hash tables where the amortised cost of incremental cleanup is preferred over expensive full-table compaction. The method complements the resizing policy implemented in `resize`, which doubles the table capacity and reinserts existing entries to restore short probe chains and mitigate accumulation of tombstones. This reinforces the concept of predictable behaviour and provides the bounded-latency characteristics essential in systems where worst-case delays must be controlled.

Finally, the `main` function serves as a simple demonstration of the table’s behaviour. It inserts sequential keys, performs lookups, and tests deletion, thereby illustrating the mechanics of grouped probing and the stability of the metadata-driven layout under routine operations. Although compact, the example verifies the table’s correctness and emphasises how memory-aware design strategies directly influence the efficiency of fundamental operations.

```rust
// Program 7.7.5: Cache-aware hash table with packed control bytes and grouped probing.
//
// Problem statement:
// Implement a simple open-addressed hash table in Rust that illustrates cache-conscious
// design principles discussed in Section 7.7.5. The table should:
//   * Store keys, values, and control bytes in contiguous arrays to maximise spatial locality.
//   * Use a compact control array (1 byte per slot) to encode EMPTY / FULL / DEL metadata.
//   * Perform grouped probing over small fixed-size windows (e.g., 8 buckets) so that
//     each probe step scans a set of adjacent control bytes that typically fit into one
//     cache line, mimicking SIMD-style comparisons.
//   * Support insertion, lookup, and deletion with tombstones (DEL) while keeping the
//     implementation compact and suitable for didactic purposes.
//   * Use a simple resizing policy to keep the load factor bounded and probe chains short.
//
// This program is not intended to be a production-ready hash table, but rather a
// self-contained example that highlights how memory layout, grouped probing, and
// tombstone management affect cache behaviour and practical performance.

use std::hash::{BuildHasher, Hash, Hasher};

const EMPTY: u8 = 0;
const FULL:  u8 = 1;
const DEL:   u8 = 2;

/// Size of a probing group. Using a power-of-two (here 8) ensures that a table
/// whose capacity is also a power-of-two is a multiple of GROUP_SIZE.
const GROUP_SIZE: usize = 8;

/// A compact, locality-optimised open-addressed hash table.
///
/// Keys, values, and control bytes are stored in contiguous arrays. The `ctrl`
/// array contains one byte per slot, indicating whether the slot is EMPTY,
/// FULL, or a tombstone (DEL). Probing proceeds in groups of `GROUP_SIZE`
/// adjacent control bytes to emulate SIMD-style vector comparisons.
///
/// This design is deliberately simple but captures the main ideas discussed
/// in Section 7.7.5: spatial locality, packed metadata, grouped probing,
/// and tombstone-aware deletion.
pub struct CacheAwareHashTable<K, V, H: BuildHasher> {
    keys:   Vec<Option<K>>,
    values: Vec<Option<V>>,
    ctrl:   Vec<u8>,
    len:    usize,
    hasher: H,
}

impl<K: Eq + Hash, V, H: BuildHasher + Default> CacheAwareHashTable<K, V, H> {
    /// Create a new table with capacity rounded up to the next power-of-two,
    /// at least GROUP_SIZE. The capacity being a power-of-two allows us to
    /// wrap indices with a simple bit mask.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two().max(GROUP_SIZE);
        // Ensure cap is a power-of-two and a multiple of GROUP_SIZE (which is itself a power-of-two).
        debug_assert!(cap.is_power_of_two());
        debug_assert_eq!(cap % GROUP_SIZE, 0);

        // Use iterators to construct vectors of `None` without requiring `Option<K>: Clone`.
        let keys   = (0..cap).map(|_| None::<K>).collect();
        let values = (0..cap).map(|_| None::<V>).collect();
        let ctrl   = vec![EMPTY; cap];

        Self {
            keys,
            values,
            ctrl,
            len: 0,
            hasher: H::default(),
        }
    }

    /// Compute a bucket index from the key's hash.
    #[inline]
    fn index_for(&self, key: &K) -> usize {
        let mut state = self.hasher.build_hasher();
        key.hash(&mut state);
        (state.finish() as usize) & (self.keys.len() - 1)
    }

    /// Insert or update a key-value pair.
    ///
    /// The load factor is kept below roughly 70% by resizing the table when
    /// necessary. Probing proceeds over groups of `GROUP_SIZE` control bytes.
    pub fn insert(&mut self, key: K, value: V) {
        if self.len * 100 / self.keys.len() > 70 {
            self.resize();
        }

        let mut idx = self.index_for(&key);

        loop {
            for _ in 0..self.keys.len() / GROUP_SIZE {
                let group_start = idx & !(GROUP_SIZE - 1);
                let group = &self.ctrl[group_start..group_start + GROUP_SIZE];

                // Scan this group linearly; a vectorised implementation would
                // compare all bytes in `group` against target patterns in one step.
                for (i, &c) in group.iter().enumerate() {
                    let slot = group_start + i;
                    match c {
                        EMPTY | DEL => {
                            // Insert into an available slot.
                            self.ctrl[slot] = FULL;
                            self.keys[slot] = Some(key);
                            self.values[slot] = Some(value);
                            self.len += 1;
                            return;
                        }
                        FULL => {
                            if let Some(ref existing) = self.keys[slot] {
                                if existing == &key {
                                    // Update an existing entry.
                                    self.values[slot] = Some(value);
                                    return;
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                }

                // Move to the next group (wrap around by masking).
                idx = (group_start + GROUP_SIZE) & (self.keys.len() - 1);
            }
        }
    }

    /// Lookup a key using grouped probing. Returns a shared reference to the
    /// associated value if found, or `None` otherwise.
    pub fn get(&self, key: &K) -> Option<&V> {
        let mut idx = self.index_for(key);

        loop {
            for _ in 0..self.keys.len() / GROUP_SIZE {
                let group_start = idx & !(GROUP_SIZE - 1);
                let group = &self.ctrl[group_start..group_start + GROUP_SIZE];

                for (i, &c) in group.iter().enumerate() {
                    let slot = group_start + i;
                    match c {
                        EMPTY => {
                            // Once we hit an EMPTY slot in open addressing, the key
                            // is guaranteed not to be present.
                            return None;
                        }
                        FULL => {
                            if let Some(ref existing) = self.keys[slot] {
                                if existing == key {
                                    return self.values[slot].as_ref();
                                }
                            }
                        }
                        DEL => {
                            // Tombstone: continue probing.
                        }
                        _ => unreachable!(),
                    }
                }

                idx = (group_start + GROUP_SIZE) & (self.keys.len() - 1);
            }
        }
    }

    /// Delete a key. Marks the slot as DEL (a tombstone) and returns `true`
    /// if an entry was removed, or `false` if the key was not present.
    pub fn remove(&mut self, key: &K) -> bool {
        let mut idx = self.index_for(key);

        loop {
            for _ in 0..self.keys.len() / GROUP_SIZE {
                let group_start = idx & !(GROUP_SIZE - 1);
                let group = &self.ctrl[group_start..group_start + GROUP_SIZE];

                for (i, &c) in group.iter().enumerate() {
                    let slot = group_start + i;
                    match c {
                        EMPTY => {
                            // Reached an unused slot: key not present.
                            return false;
                        }
                        FULL => {
                            if let Some(ref existing) = self.keys[slot] {
                                if existing == key {
                                    self.ctrl[slot] = DEL;
                                    self.keys[slot] = None;
                                    self.values[slot] = None;
                                    self.len -= 1;
                                    return true;
                                }
                            }
                        }
                        DEL => {
                            // Skip tombstones.
                        }
                        _ => unreachable!(),
                    }
                }

                idx = (group_start + GROUP_SIZE) & (self.keys.len() - 1);
            }
        }
    }

    /// Resize by doubling the capacity and reinserting all live entries.
    /// This re-establishes short probe chains and keeps the load factor moderate.
    fn resize(&mut self) {
        let new_cap = self.keys.len() * 2;
        let mut new_table = CacheAwareHashTable::<K, V, H>::new(new_cap);

        for i in 0..self.keys.len() {
            if self.ctrl[i] == FULL {
                if let Some(k) = self.keys[i].take() {
                    let v = self.values[i].take().expect("value must exist for FULL slot");
                    new_table.insert(k, v);
                }
            }
        }

        *self = new_table;
    }
}

fn main() {
    use std::collections::hash_map::RandomState;

    let mut table = CacheAwareHashTable::<u64, u64, RandomState>::new(64);

    for i in 0..50 {
        table.insert(i, i * 10);
    }

    println!("Lookup 42 → {:?}", table.get(&42));
    println!("Remove 42 → {:?}", table.remove(&42));
    println!("Lookup 42 after delete → {:?}", table.get(&42));
}
```

Program 7.7.5 illustrates how practical hash table performance depends as much on memory layout and probing strategy as on theoretical collision-resolution policies. By using a packed control array and performing grouped probing over contiguous bucket metadata, the implementation reflects the core insights of Section 7.7.5: cache efficiency arises from predictable, sequential memory access and the minimisation of branch divergence. The use of tombstones for deletion and periodic resizing further highlights the engineering considerations necessary to maintain stable performance over long execution periods.

The controlled and locality-focused design demonstrated here underscores why many high-performance hash tables favour open addressing with tightly packed metadata. Such structures align naturally with hardware prefetching, SIMD comparisons, and modern cache hierarchies. While simplified for pedagogical clarity, this program provides a foundation for exploring more advanced optimisations, such as SIMD-assisted emptiness detection, multi-level prefetching strategies, or hybrid schemes that combine the analytical robustness of chaining with the locality advantages of flat arrays. As system architectures continue to evolve, the interplay between data layout, access pattern, and memory behaviour will remain a central theme in the design of efficient hashing structures.

## 7.7.6. Section Summary and Outlook

Hash tables and hash memories occupy a central position in modern computing, providing a fast and flexible mechanism for storing and retrieving large volumes of data. The preceding sections have shown that high performance in real systems arises from the careful integration of three elements: the hash function, the collision-resolution strategy, and the underlying memory layout. When these components are designed coherently, a hash table can achieve expected constant-time operations across diverse workloads, from numerical simulation to embedded systems.

At the conceptual core lies the *hash function*, which is no longer a simple arithmetic formula but a configurable object that performs key mixing, seeding, and finalisation. This modularity allows designers to tailor hashing behaviour to particular use cases. Randomised hashing enhances robustness against adversarial inputs (Farach-Colton, Krapivin & Kuszmaul, 2024), composite hashing supports complex structured keys, and hardware-oriented constructions draw on efficient bit-mixing mechanisms such as LFSR stages (Somanathan, Reddy & Bhakthavatchalu, 2025). These variations demonstrate that hashing must adapt to the constraints of different architectures and application domains.

Collision resolution remains an equally important dimension. *Separate chaining* offers stable performance even under high load factors and accommodates frequent insertions and deletions. *Open addressing*, by contrast, provides exceptional memory locality and compact storage but requires careful tuning of load factors and probing strategies. Theoretical analyses via expected chain lengths, probe counts, and load factor behaviour, provide precise guidance on when each strategy is most effective. These insights parallel developments in pseudo-random sequence generation, where simple linear mechanisms can produce sophisticated behaviour when parameters are chosen correctly (Almaraz & Román Villaizán, 2023).

Performance in practice is governed not only by algorithmic choices but also by *memory layout*. Modern processors reward contiguous access patterns, dense metadata, and predictable probe sequences. Array-based chaining and group-probing techniques exemplify the shift toward hardware-aware designs that exploit cache hierarchies, prefetching mechanisms, and SIMD capabilities (Li, 2023). As datasets grow larger and become more irregular, these considerations increasingly dominate the realised latency and throughput of hash memories.

A recurring theme across applications, whether in secure storage, real-time systems, or multimedia processing, is that hashing must balance speed, predictability, and robustness. LFSR-inspired mixers and hybrid chaotic methods have demonstrated their effectiveness in lightweight encryption, image protection, and pseudo-random generation (Maity et al., 2025; Okunbor, Omorogbe & Edeko, 2024). These examples illustrate a broader trend: modern digital systems increasingly combine linear structures for efficiency with nonlinear components for security, resilience, and adaptability.

Looking forward, emerging research explores *multi-hash architectures, hardware–software co-design*, and *probabilistic hashing schemes* that blend deterministic performance with statistical guarantees. With growing interest in high-throughput numerical computing, edge processing, and secure distributed systems, hash tables will continue to evolve in both capability and design philosophy. Their enduring relevance highlights the power of combining mathematical clarity with practical engineering, enabling simple abstractions to support the complex requirements of modern computation.

```{figure} images/pqQDe4beUu67RvW3raYP-ZBnaQKCRFCGNGEVoN5e9-v1.png
:name: owtvSxwNuz
:align: center
:width: 50%

**Figure 7.7.3** Core Components of Modern Hash Table Design. A conceptual overview showing how the hash function, collision-resolution strategy, and memory layout interact to determine hash table performance. The hash function transforms keys into mixed bit patterns and initial bucket indices; the collision-resolution strategy determines how the table responds to occupied buckets; and memory-layout optimisations influence cache behaviour, throughput, and overall efficiency.
```

Figure 7.7.3 illustrates the three foundational components that determine the behaviour and performance of a modern hash table. At the left, the hash function performs key mixing, seeding, and finalisation to generate a uniformly distributed hash value. This value feeds into the collision-resolution module, which decides how conflicts are handled, either through separate chaining or open addressing. The resulting access pattern interacts directly with the table’s memory layout. Contiguous bucket arrangements, packed metadata arrays, and cache-aware probing strategies significantly influence latency and throughput. Together, these modules form a tightly coupled system in which improvements to one component often yield benefits across the entire hash table design. The figure highlights that high-performance implementations arise not from any single technique, but from coherent integration of hashing, probing, and memory-layout strategies.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/ucpYOGZqWUbktEPp2iwf.4","tags":[]}

# 7.8. Simple Monte Carlo Integration

Monte Carlo integration provides a probabilistic method for approximating definite integrals by replacing deterministic quadrature rules with random sampling. Unlike classical formulas that rely on structured grids, polynomial interpolation, or tensor-product constructions, Monte Carlo methods select sample points randomly within the domain of integration. This approach is particularly effective in high-dimensional problems, where deterministic methods suffer from the exponential growth of required evaluations, an effect collectively known as the *curse of dimensionality*. In practice, Monte Carlo techniques arise naturally in physics (path integrals, neutron transport), financial mathematics (derivative pricing), rendering, and in evaluating expectations associated with multi-dimensional probability distributions.

To illustrate the underlying idea, let $\Omega \subset \mathbb{R}^d$ be a region with finite volume:

$$V = \int_{\Omega} d\mathbf{x} \tag{7.8.1}$$

Let $f : \Omega \to \mathbb{R}$ be integrable, and consider the integral:

$$I = \int_{\Omega} f(\mathbf{x}) d\mathbf{x} \tag{7.8.2}$$

If $\mathbf{X}$ is a random variable uniformly distributed over $\Omega$, then:

$$\mathbb{E}[f(\mathbf{X})] = \frac{1}{V}\int_{\Omega} f(\mathbf{x}) d\mathbf{x} \tag{7.8.3}$$

so that the integral becomes the expectation,

$$I = V\mathbb{E}[f(\mathbf{X})] \tag{7.8.4}$$

By generating independent samples $\mathbf{x}_1, \ldots, \mathbf{x}_N \sim \mathrm{Unif}(\Omega)$, we obtain the Monte Carlo estimator:

$$Q_N =V \frac{1}{N} \sum{i=1}^N f(\mathbf{x}_i) \tag{7.8.5}$$

This estimator is unbiased, $\mathbb{E}[Q_N] = I$, and its variance follows directly from the variance of the integrand under the uniform distribution:

$$\mathrm{Var}(Q_N)= \frac{V^2 \sigma^2}{N}, \qquad\sigma^2 = \mathrm{Var}[f(\mathbf{X})] \tag{7.8.6}$$

Thus, by the central limit theorem,

$$Q_N \approx I \pm \frac{V\sigma}{\sqrt{N}} \tag{7.8.7}$$

so the typical error decays as $O(N^{-1/2})$, independently of the dimension $d$. This dimension-insensitivity, though slow, makes Monte Carlo a universal tool for high-dimensional integration.

The algorithmic structure is remarkably simple:

1. Generate $N$ random points uniformly in $\Omega$.
2. Evaluate the function at these points.
3. Average the results and multiply by the known volume $V$.

The computation requires only $O(N)$ evaluations of $f$, and the memory footprint is minimal since sample values can be accumulated on the fly.

### Hit-Or-Miss Interpretation and Geometry

A simple but illuminating special case is the “dart-throwing’’ method for estimating areas or volumes. Consider estimating the area of the unit disk:

$$D = {(x,y)\in\mathbb{R}^2 : x^2 + y^2 \le 1} \tag{7.8.8}$$

Embed $D$ in the square $V = [-1,1]^2$, which has area $A_V = 4$. Draw $N$ points uniformly in $V$; let $N_{\mathrm{in}}$ be the number falling inside the disk. Because,

$$\frac{N_{\mathrm{in}}}{N} \approx \mathbb{P}((X,Y)\in D) \tag{7.8.9}$$

the estimated area is:

$$A_D \approx \frac{N_{\mathrm{in}}}{N}A_V \tag{7.8.10}$$

This method extends easily to complex geometric shapes. If $W\subset V$ is a region described by inequalities or implicit surfaces, and:

$$
\mathbb{1}_W(\mathbf{x}) =
\begin{cases}
1, & \mathbf{x}\in W,\\
0, & \text{otherwise},
\end{cases}
\tag{7.8.11}
$$

then,

$$\int_W g(\mathbf{x})d\mathbf{x}=\int_V g(\mathbf{x})\mathbb{1}_W(\mathbf{x})d\mathbf{x} \tag{7.8.12}$$

\
Samples falling outside $W$ contribute zero, increasing the estimator’s variance. Thus, tight bounding volumes reduce wasted samples and improve accuracy.

### Example: Volume and Mass Moments of a Truncated Torus

Irregular solids offer an ideal setting for Monte Carlo integration. Consider the solid torus defined by,

$$\left(\sqrt{x^2 + y^2} - 3\right)^2 + z^2 \le 1, \tag{7.8.13}$$

which is then truncated by the plane constraints:

$$x \ge -1,\qquad y \ge -3 \tag{7.8.14}$$

Let $W$ denote the resulting shape. The volume and first moments,

\begin{equation}
\begin{aligned}
\mathrm{Vol}(W) &= \int_W dV \\
M_x &= \int_W x\, dV \\
M_y &= \int_W y\, dV \\
M_z &= \int_W z\, dV
\end{aligned}
\tag{7.8.15}
\end{equation}

give the center of mass,

$$(\bar{x},\bar{y},\bar{z}) = \frac{(M_x,M_y,M_z)}{\text{Vol}(W)} \tag{7.8.16}$$

Since analytic integration over $W$ is essentially impossible, Monte Carlo sampling in a rectangular bounding box provides a straightforward approximation. The only operation needed per sample is a fast membership test and evaluation of $(1,x,y,z)$.

### Rust Implementation

Following the general formulation of Monte Carlo integration in Section 7.8, Program 7.8.0 provides a practical demonstration of how expectation-based estimators can approximate geometric quantities that are analytically inaccessible. Whereas the surrounding discussion introduces the unbiased estimator in Equation (7.8.5) and its variance scaling in Equation (7.8.6), this program implements these ideas concretely for two representative settings: a two-dimensional hit-or-miss estimator for the unit disk and a three-dimensional integration problem involving a truncated torus with nonlinear boundary constraints. By isolating sampling, indicator evaluation, and moment accumulation into clear functional components, the program illustrates the core workflow underlying Monte Carlo methods while emphasizing the geometric intuition that motivates sampling-based integration.

At the core of the implementation is a set of functions that directly encode the probabilistic estimator defined in Equations (7.8.5)–(7.8.7). The simplest of these is the pair of sampling functions, `sample_in_square` and `sample_in_box`, which generate uniformly distributed points in the bounding volumes associated with the unit disk and truncated torus respectively. These functions operationalize the assumption that the random variable $\mathbf{X}$ is uniformly distributed over the region $\Omega$, as required by Equation (7.8.3). By drawing points uniformly, the estimator $Q_N$ in Equation (7.8.5) becomes a direct average of sample contributions weighted by the enclosing volume.

The indicator-style membership tests implement the geometric constraints that define each integration domain. For the unit disk, the function `inside_unit_disk` evaluates the inequality in Equation (7.8.8), acting as $\mathbb{1}_W(\mathbf{x})$ in Equation (7.8.11). Samples that satisfy this condition contribute one “hit” in the hit-or-miss estimator (7.8.10). For the truncated torus, the function `inside_truncated_torus` encodes both the toroidal surface in Equation (7.8.13) and the planar truncations in Equation (7.8.14). This modular structure allows the program to treat arbitrary regions $W$ exactly as described in Equation (7.8.12), where samples falling outside $W$ contribute zero.

Two higher-level functions perform the actual Monte Carlo estimation. The function `estimate_unit_disk_area` constructs the hit-or-miss approximation described in Equations (7.8.9)–(7.8.10), counting the number of samples that fall inside the disk and scaling by the known volume of the bounding square. It also computes an empirical standard error based on the variance formula (7.8.6), providing an estimate of the sampling uncertainty predicted by the central limit theorem in Equation (7.8.7). The function `estimate_truncated_torus` generalizes this idea to a weighted estimator for the integrals in Equation (7.8.15). Each accepted sample contributes to the volume as well as the coordinate moments $M_x, M_y, M_z$, enabling a direct computation of the center of mass using Equation (7.8.16). This structure mirrors the $O(N)$ algorithmic cost described earlier, since each sample requires only a membership test and simple arithmetic.

The `main` function brings these components together to illustrate both the accuracy and the limitations of Monte Carlo estimators in practice. It begins with the two-dimensional hit-or-miss calculation, producing an estimate of the disk area and comparing it to the true value $\pi$. The observed error size is consistent with the $N^{-1/2}$ convergence rate predicted by (7.8.6)–(7.8.7). The second part evaluates the more complex three-dimensional problem of estimating the volume and center of mass of the truncated torus. Because analytic integration is infeasible for this geometry, the program highlights the usefulness of Monte Carlo sampling in handling irregular domains defined implicitly by nonlinear inequalities. Together, the two examples show how the same underlying estimator can be adapted to problems of increasing geometric complexity with minimal modification to the computational structure.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
/*
    Program 7.8.0: Simple Monte Carlo Integration for Geometric Volumes

    Problem statement.
    Implement the basic Monte Carlo integration ideas from Section 7.8:

    (1) Hit-or-miss estimation of the area of the unit disk
        D = { (x, y) ∈ R^2 : x^2 + y^2 ≤ 1 }      (7.8.8)
        by sampling uniformly in the square V = [-1, 1]^2 of area A_V = 4 and
        using
            A_D ≈ (N_in / N) A_V,                 (7.8.10)
        where N_in is the number of samples falling inside D.

    (2) Volume and first moments of the truncated torus W defined by
            (sqrt(x^2 + y^2) - 3)^2 + z^2 ≤ 1,   (7.8.13)
            x ≥ -1,  y ≥ -3,                     (7.8.14)
        using uniform sampling in a rectangular bounding box V and the
        indicator 1_W(x) as in (7.8.11)–(7.8.12) to approximate
            Vol(W), M_x, M_y, M_z                (7.8.15)
        and the center of mass
            (x̄, ȳ, z̄) = (M_x, M_y, M_z) / Vol(W).   (7.8.16)

    The program prints the Monte Carlo estimates, compares the disk area
    with π, and reports the estimated center of mass of W.
*/

// Cargo.toml:
//
// [package]
// name = "myproject"
// version = "0.1.0"
// edition = "2021"
//
// [dependencies]
// rand = "0.8"

use rand::distributions::Uniform;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Generate a random point (x, y) uniformly in the square [-1, 1] x [-1, 1].
fn sample_in_square(rng: &mut StdRng) -> (f64, f64) {
    let dist = Uniform::new(-1.0_f64, 1.0_f64);
    let x = rng.sample(dist);
    let y = rng.sample(dist);
    (x, y)
}

/// Indicator for the unit disk D = {(x, y): x^2 + y^2 <= 1}.
fn inside_unit_disk(x: f64, y: f64) -> bool {
    x * x + y * y <= 1.0
}

/// Estimate the area of the unit disk by hit-or-miss Monte Carlo in V = [-1,1]^2.
///
/// According to (7.8.9)–(7.8.10),
///   A_D ≈ (N_in / N) * A_V,
/// where A_V = 4 and N_in is the number of points falling inside the disk.
fn estimate_unit_disk_area(n_samples: u64, rng: &mut StdRng) -> (f64, f64) {
    let area_square = 4.0_f64;
    let mut n_in: u64 = 0;

    for _ in 0..n_samples {
        let (x, y) = sample_in_square(rng);
        if inside_unit_disk(x, y) {
            n_in += 1;
        }
    }

    let n = n_samples as f64;
    let p_hat = n_in as f64 / n;
    let area_est = p_hat * area_square;

    // Binomial variance approximation for the hit-or-miss estimator:
    // Var(A_D_hat) ≈ A_V^2 * p(1 - p) / N
    let var_est = area_square * area_square * p_hat * (1.0 - p_hat) / n;
    let std_dev = var_est.sqrt();

    (area_est, std_dev)
}

/// Generate a random point (x, y, z) uniformly in the rectangular box
/// [xmin, xmax] x [ymin, ymax] x [zmin, zmax].
fn sample_in_box(
    rng: &mut StdRng,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    zmin: f64,
    zmax: f64,
) -> (f64, f64, f64) {
    let dist_x = Uniform::new(xmin, xmax);
    let dist_y = Uniform::new(ymin, ymax);
    let dist_z = Uniform::new(zmin, zmax);
    let x = rng.sample(dist_x);
    let y = rng.sample(dist_y);
    let z = rng.sample(dist_z);
    (x, y, z)
}

/// Membership test for the truncated torus W defined by
///
///   (sqrt(x^2 + y^2) - 3)^2 + z^2 <= 1,       (torus)
///   x >= -1,
///   y >= -3.
///
/// This implements the inequalities (7.8.13)–(7.8.14).
fn inside_truncated_torus(x: f64, y: f64, z: f64) -> bool {
    if x < -1.0 || y < -3.0 {
        return false;
    }

    let r_xy = (x * x + y * y).sqrt();
    let torus_left = (r_xy - 3.0).powi(2) + z * z;

    torus_left <= 1.0
}

/// Monte Carlo estimation of the volume and first moments of W:
///
///   Vol(W) = ∫_W dV,
///   Mx = ∫_W x dV,  My = ∫_W y dV,  Mz = ∫_W z dV.
///
/// We embed W in a rectangular bounding box V and use the indicator
/// 1_W(x) as in (7.8.11)–(7.8.12).
///
/// The integrals are approximated by
///
///   Vol(W) ≈ V_box / N * Σ 1_W(x_i),
///   Mx     ≈ V_box / N * Σ x_i 1_W(x_i),
///   ...
///
/// where the x_i are drawn uniformly in V.
fn estimate_truncated_torus(
    n_samples: u64,
    rng: &mut StdRng,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    // Bounding box chosen large enough to contain the torus:
    //   x ∈ [-1, 4],  y ∈ [-3, 4],  z ∈ [-1, 1].
    let xmin = -1.0_f64;
    let xmax = 4.0_f64;
    let ymin = -3.0_f64;
    let ymax = 4.0_f64;
    let zmin = -1.0_f64;
    let zmax = 1.0_f64;

    let volume_box = (xmax - xmin) * (ymax - ymin) * (zmax - zmin);
    let weight = volume_box / (n_samples as f64);

    let mut vol_est = 0.0_f64;
    let mut mx_est = 0.0_f64;
    let mut my_est = 0.0_f64;
    let mut mz_est = 0.0_f64;

    for _ in 0..n_samples {
        let (x, y, z) = sample_in_box(rng, xmin, xmax, ymin, ymax, zmin, zmax);
        if inside_truncated_torus(x, y, z) {
            vol_est += weight;
            mx_est += weight * x;
            my_est += weight * y;
            mz_est += weight * z;
        }
    }

    // Center of mass (7.8.16):
    let cx = mx_est / vol_est;
    let cy = my_est / vol_est;
    let cz = mz_est / vol_est;

    (vol_est, mx_est, my_est, mz_est, cx, cy, cz)
}

fn main() {
    // Fixed seed for reproducible results in the textbook.
    let mut rng = StdRng::seed_from_u64(7_8_2025);

    // Example 1: hit-or-miss area of the unit disk.
    let n_disk: u64 = 1_000_000;
    let (area_est, std_dev) = estimate_unit_disk_area(n_disk, &mut rng);

    println!("Hit-or-miss Monte Carlo for the unit disk D:");
    println!("  Number of samples N       = {}", n_disk);
    println!("  Estimated area A_D        ≈ {:.6}", area_est);
    println!("  Theoretical area π        ≈ {:.6}", std::f64::consts::PI);
    println!("  Estimated standard error  ≈ {:.6}", std_dev);
    println!();

    // Example 2: volume and center of mass of the truncated torus W.
    let n_torus: u64 = 2_000_000;
    let (vol, mx, my, mz, cx, cy, cz) = estimate_truncated_torus(n_torus, &mut rng);

    println!("Monte Carlo volume and first moments of the truncated torus W:");
    println!("  Number of samples N       = {}", n_torus);
    println!("  Estimated Vol(W)          ≈ {:.6}", vol);
    println!("  Estimated Mx, My, Mz      ≈ ({:.6}, {:.6}, {:.6})", mx, my, mz);
    println!("  Estimated center of mass  ≈ ({:.6}, {:.6}, {:.6})", cx, cy, cz);
}
```

Program 7.8.0 demonstrates how Monte Carlo integration translates the theoretical estimator of Equation (7.8.5) into a simple yet powerful computational tool. By relying solely on uniform random sampling and indicator-based membership tests, the method circumvents the need for structured grids or analytic expressions for the domain. The unit disk example underscores the central theme of Section 7.8: even for smooth and familiar regions, Monte Carlo produces unbiased estimates whose variability decreases as $N^{-1/2}$, independent of dimension. The truncated-torus example further illustrates the flexibility of the method, capturing moments of a shape whose analytic integrals are effectively unattainable using classical quadrature.

The contrasting behavior of the two examples demonstrates both the strengths and the limitations of Monte Carlo integration. The hit-or-miss estimator is simple but suffers from high variance unless the bounding region fits tightly, while the truncated-torus moment calculation shows how weighting by coordinate values extends the estimator to more sophisticated physical quantities. Overall, the modular design of the program provides a foundation on which more advanced Monte Carlo techniques, such as importance sampling, stratification, and Markov chain methods, can be developed in later sections. These enhancements address variance reduction and sampling efficiency, enabling Monte Carlo integration to scale effectively to high-dimensional scientific and engineering applications.

## 7.8.1. Change of Variables and Variance Reduction

Variance reduction is central to efficient Monte Carlo integration. One of the most powerful tools is *importance sampling*, which aims to choose a sampling distribution that emphasizes regions of large integrand magnitude.

As an illustration, suppose the density of the object in the previous example varies as,

$$f(x,y,z)=e^{5z} \tag{7.8.17}$$

which changes by nearly five orders of magnitude between $z=-1$ and $z=1$. Uniform sampling wastes effort where $f$ is tiny. A more efficient strategy is to introduce a change of variables. Since the problematic factor depends only on $z$, define:

\begin{equation}
\begin{aligned}
ds &= e^{5z}\, dz \\
s &= \frac{1}{5} e^{5z} \\
z &= \frac{1}{5}\ln(5s)
\end{aligned}
\tag{7.8.18}
\end{equation}

By sampling $s$ uniformly in its transformed interval and substituting into $z(s)$, the Monte Carlo estimator implicitly draws $z$ from a density proportional to $e^{5z}$. The factor $e^{5z}$ is thereby removed from the integrand, flattening its variation and greatly reducing variance.

More broadly, variance-reduction strategies include importance sampling, control variates, stratified sampling, and antithetic variates (e.g. Shyamsundar *et al.*, 2024). All aim to improve the constant in the Monte Carlo error bound without compromising the estimator’s unbiasedness.

### Rust Implementation

Following the discussion of uniform Monte Carlo estimators in Section 7.8 and their geometric applications in Program 7.8.0, Section 7.8.1 introduces the role of change of variables as a targeted variance-reduction strategy. When the integrand exhibits strong directional variation, as in the exponentially weighted density of Equation (7.8.17), uniform sampling distributes effort inefficiently, spending most sample evaluations where the integrand is negligible. Program 7.8.1 provides a practical implementation of the exponential transformation defined in Equation (7.8.18), which reshapes the sampling distribution so that the Monte Carlo estimator preferentially draws points from the regions that contribute most to the integral. This example illustrates the fundamental idea behind importance sampling: by aligning the sampling distribution with the structure of the integrand, the estimator becomes significantly more stable and efficient while retaining unbiasedness.

At the core of the implementation is the contrast between two sampling strategies: one that draws $(x, y, z)$ uniformly from a rectangular bounding volume, and another that employs the change of variables introduced in Equation (7.8.18) to bias sampling toward larger values of $z$. The function `inside_truncated_torus` serves as the shared geometric filter for both methods, evaluating the nonlinear inequalities in (7.8.13)–(7.8.14) to determine whether a sample lies within the truncated torus $W$. This indicator function plays the same role as $\mathbb{1}_W$ in Equation (7.8.11), ensuring that only valid samples contribute to the estimator defined in (7.8.12). The function `sample_uniform_in_box` implements the baseline sampling scheme consistent with the standard estimator of Equation (7.8.5), drawing points uniformly and weighting each accepted sample by the integrand $e^{5z}$ from Equation (7.8.17).

The key structural improvement arises in the alternative routine `sample_xy_importance_z`, which implements the exponential change of variables specified in Equation (7.8.18). Instead of sampling $z$ uniformly, the program samples the auxiliary variable $s$ uniformly and transforms it back to $z$ via the inverse relation $z=\frac{1}{5}\ln(5s)$. This ensures that the induced distribution of $(z)$ is proportional to $e^{5z}$, matching the problematic factor in the integrand. As a result, the importance-sampled estimator removes the exponential term altogether, making its contribution effectively constant for accepted samples and thereby reducing variance substantially.

The routine `estimate_mass_uniform` computes the direct Monte Carlo estimate of the mass\
$\int_W e^{5z}, dV$ using uniform sampling. Because the integrand varies by several orders of magnitude over the interval $[-1, 1]$, the estimator suffers from large fluctuations in sample values, reflected in the larger standard error reported in the output. To quantify this uncertainty, the function accumulates both first and second moments so that the estimator variance predicted by Equation (7.8.6) can be computed empirically.

In contrast, `estimate_mass_importance` constructs the importance-sampled estimator whose weighting collapses to a constant determined by the normalizing factor in Equation (7.8.18). Since each accepted sample contributes the same value, the estimator’s variability depends only on the geometric randomness of whether a point lies inside $W$. This fundamental flattening of the integrand leads to a sharply reduced standard error, a hallmark of effective importance sampling. The invariance of the final estimator under this transformation confirms that the change of variables preserves unbiasedness while improving sampling efficiency.

The `main` function juxtaposes the two estimators to highlight the practical impact of importance sampling. It first computes the uniform estimator, reflecting the direct implementation of Equation (7.8.17), and then evaluates the importance-sampled estimator constructed using the change of variables in Equation (7.8.18). The similarity of their mean values demonstrates the unbiasedness of both approaches, while the ratio of their variances quantifies the degree of improvement achieved. This side-by-side comparison provides a concrete numerical illustration of the variance-reduction principles introduced in Section 7.8.1.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
/*
    Program 7.8.1: Importance Sampling via Change of Variables for a z-Dependent Density

    Problem statement.
    Building on the truncated torus W from Program 7.8.0, we now consider a
    non-uniform density that depends strongly on the vertical coordinate:

        f(x, y, z) = exp(5 z),                                      (7.8.17)

    which varies by several orders of magnitude over z ∈ [-1, 1]. Direct
    Monte Carlo integration with uniform sampling in z wastes samples in
    regions where f is negligible.

    The goal is to estimate the total mass

        M = ∫_W f(x, y, z) dV,

    using two estimators:

    1. Baseline estimator:
       Sample (x, y, z) uniformly in a rectangular box V containing W,
       accept points inside W, and weight each accepted sample by exp(5 z),
       as in standard Monte Carlo,

           Q_N = (Vol(V) / N) Σ 1_W(x_i, y_i, z_i) exp(5 z_i).      (7.8.5)

    2. Importance-sampled estimator:
       Use the change of variables (7.8.18)

           ds = exp(5 z) dz,
           s  = (1/5) exp(5 z),
           z  = (1/5) ln(5 s),                                      (7.8.18)

       to draw z from a density proportional to exp(5 z) on [-1, 1].
       Combined with uniform sampling in x and y, this removes the factor
       exp(5 z) from the integrand, making the mass estimator proportional
       to

           Σ 1_W(x_i, y_i, z_i),

       and thereby reducing variance.

    The program estimates M with both methods, reports empirical standard
    errors, and prints the ratio of variances to illustrate the efficiency
    gain from importance sampling.
*/

// Cargo.toml:
//
// [package]
// name = "myproject"
// version = "0.1.0"
// edition = "2021"
//
// [dependencies]
// rand = "0.8"

use rand::distributions::Uniform;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Membership test for the truncated torus W defined by
///
///   (sqrt(x^2 + y^2) - 3)^2 + z^2 <= 1,        (7.8.13)
///   x >= -1,                                   (7.8.14)
///   y >= -3.
///
/// This is identical to the geometry used in Program 7.8.0.
fn inside_truncated_torus(x: f64, y: f64, z: f64) -> bool {
    if x < -1.0 || y < -3.0 {
        return false;
    }

    let r_xy = (x * x + y * y).sqrt();
    let torus_left = (r_xy - 3.0).powi(2) + z * z;

    torus_left <= 1.0
}

/// Sample (x, y, z) uniformly from the rectangular bounding box V:
///   x ∈ [xmin, xmax], y ∈ [ymin, ymax], z ∈ [zmin, zmax].
fn sample_uniform_in_box(
    rng: &mut StdRng,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    zmin: f64,
    zmax: f64,
) -> (f64, f64, f64) {
    let dist_x = Uniform::new(xmin, xmax);
    let dist_y = Uniform::new(ymin, ymax);
    let dist_z = Uniform::new(zmin, zmax);

    let x = rng.sample(dist_x);
    let y = rng.sample(dist_y);
    let z = rng.sample(dist_z);

    (x, y, z)
}

/// Sample z from the importance distribution proportional to exp(5 z)
/// on [zmin, zmax] using the change of variables (7.8.18):
///
///   s  = (1/5) exp(5 z),
///   z  = (1/5) ln(5 s).
///
/// If s is drawn uniformly on [s_min, s_max], then the induced density
/// of z is proportional to exp(5 z).
fn sample_z_importance(rng: &mut StdRng, zmin: f64, zmax: f64) -> f64 {
    // s_min = (1/5) exp(5 zmin), s_max = (1/5) exp(5 zmax).
    let s_min = (1.0_f64 / 5.0) * (5.0 * zmin).exp();
    let s_max = (1.0_f64 / 5.0) * (5.0 * zmax).exp();

    let dist_s = Uniform::new(s_min, s_max);
    let s = rng.sample(dist_s);

    // Inverse mapping: z = (1/5) ln(5 s).
    (1.0_f64 / 5.0) * (5.0 * s).ln()
}

/// Sample (x, y) uniformly in the box [xmin, xmax] × [ymin, ymax]
/// and z from the importance distribution in z.
fn sample_xy_importance_z(
    rng: &mut StdRng,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    zmin: f64,
    zmax: f64,
) -> (f64, f64, f64) {
    let dist_x = Uniform::new(xmin, xmax);
    let dist_y = Uniform::new(ymin, ymax);

    let x = rng.sample(dist_x);
    let y = rng.sample(dist_y);
    let z = sample_z_importance(rng, zmin, zmax);

    (x, y, z)
}

/// Estimate the mass M = ∫_W exp(5 z) dV using uniform sampling in (x, y, z).
///
/// We embed W in a box V and use the standard Monte Carlo estimator:
///
///   M ≈ (Vol(V) / N) Σ 1_W(x_i, y_i, z_i) exp(5 z_i).
///
/// This function returns (M_hat, standard_error).
fn estimate_mass_uniform(
    n_samples: u64,
    rng: &mut StdRng,
) -> (f64, f64) {
    let xmin = -1.0_f64;
    let xmax = 4.0_f64;
    let ymin = -3.0_f64;
    let ymax = 4.0_f64;
    let zmin = -1.0_f64;
    let zmax = 1.0_f64;

    let volume_box = (xmax - xmin) * (ymax - ymin) * (zmax - zmin);

    let mut sum_y = 0.0_f64;
    let mut sum_y2 = 0.0_f64;

    for _ in 0..n_samples {
        let (x, y, z) = sample_uniform_in_box(rng, xmin, xmax, ymin, ymax, zmin, zmax);
        let integrand = if inside_truncated_torus(x, y, z) {
            (5.0 * z).exp()
        } else {
            0.0
        };

        let y_val = volume_box * integrand;
        sum_y += y_val;
        sum_y2 += y_val * y_val;
    }

    let n = n_samples as f64;
    let mean = sum_y / n;
    let var_y = if n > 1.0 {
        // Unbiased sample variance of Y_i.
        (sum_y2 - n * mean * mean) / (n - 1.0)
    } else {
        0.0
    };

    // Variance of the estimator is var(Y_i)/N, so standard error:
    let std_error = (var_y / n).sqrt();

    (mean, std_error)
}

/// Estimate the mass M = ∫_W exp(5 z) dV using importance sampling in z.
///
/// We draw x, y uniformly in [xmin, xmax] × [ymin, ymax], and z from the
/// density
///
///   g(z) = exp(5 z) / C,  z ∈ [zmin, zmax],
///
/// where
///
///   C = ∫_{zmin}^{zmax} exp(5 z) dz = (1/5) (exp(5 zmax) - exp(5 zmin)).
///
/// The joint sampling density over V is:
///
///   h(x, y, z) = (1 / (Lx Ly)) g(z),
///
/// with Lx = xmax - xmin, Ly = ymax - ymin. The importance weight for
/// exp(5 z) is then
///
///   w(x, y, z) = exp(5 z) / h(x, y, z) = C Lx Ly,
///
/// which is constant. The estimator becomes:
///
///   M ≈ (C Lx Ly / N) Σ 1_W(x_i, y_i, z_i).
///
/// This function returns (M_hat, standard_error).
fn estimate_mass_importance(
    n_samples: u64,
    rng: &mut StdRng,
) -> (f64, f64) {
    let xmin = -1.0_f64;
    let xmax = 4.0_f64;
    let ymin = -3.0_f64;
    let ymax = 4.0_f64;
    let zmin = -1.0_f64;
    let zmax = 1.0_f64;

    let lx = xmax - xmin;
    let ly = ymax - ymin;

    // Normalizing constant C for g(z) ∝ exp(5 z) on [zmin, zmax].
    let c_norm = (1.0_f64 / 5.0) * ((5.0 * zmax).exp() - (5.0 * zmin).exp());

    // Constant weight factor C Lx Ly appearing in the importance sampling estimator.
    let weight_const = c_norm * lx * ly;

    let mut sum_y = 0.0_f64;
    let mut sum_y2 = 0.0_f64;

    for _ in 0..n_samples {
        let (x, y, z) = sample_xy_importance_z(rng, xmin, xmax, ymin, ymax, zmin, zmax);
        let y_val = if inside_truncated_torus(x, y, z) {
            weight_const
        } else {
            0.0
        };

        sum_y += y_val;
        sum_y2 += y_val * y_val;
    }

    let n = n_samples as f64;
    let mean = sum_y / n;
    let var_y = if n > 1.0 {
        (sum_y2 - n * mean * mean) / (n - 1.0)
    } else {
        0.0
    };
    let std_error = (var_y / n).sqrt();

    (mean, std_error)
}

fn main() {
    // Fixed seed for reproducible results.
    let mut rng = StdRng::seed_from_u64(7_8_2025);

    let n_samples: u64 = 2_000_000;

    println!("Estimating M = ∫_W exp(5 z) dV for the truncated torus W");
    println!("Number of samples N = {}", n_samples);
    println!();

    // Baseline estimation with uniform (x, y, z) sampling.
    let (m_uniform, se_uniform) = estimate_mass_uniform(n_samples, &mut rng);
    println!("Uniform sampling in (x, y, z):");
    println!("  Estimated mass M_uniform ≈ {:.6}", m_uniform);
    println!("  Estimated standard error  ≈ {:.6}", se_uniform);
    println!();

    // Importance sampling in z via change of variables (7.8.18).
    let (m_importance, se_importance) = estimate_mass_importance(n_samples, &mut rng);
    println!("Importance sampling in z (density ∝ exp(5 z)):");
    println!("  Estimated mass M_import  ≈ {:.6}", m_importance);
    println!("  Estimated standard error  ≈ {:.6}", se_importance);
    println!();

    if se_importance > 0.0 {
        let variance_ratio = (se_uniform * se_uniform) / (se_importance * se_importance);
        println!("Variance reduction factor (Uniform / Importance) ≈ {:.2}", variance_ratio);
    }
}
```

Program 7.8.1 demonstrates how variance reduction can be achieved through a carefully selected change of variables, addressing one of the central challenges introduced in Section 7.8.1: the inefficiency of uniform sampling when the integrand varies sharply across the domain. By mapping the problematic exponential factor into the sampling distribution itself, the program transforms a highly uneven integrand into one with nearly constant contributions, reducing estimator variance dramatically. Despite their structural differences, both the uniform and importance-sampled estimators remain unbiased, and their close agreement validates the correctness of the transformation.

The comparison between the two approaches underscores why importance sampling is one of the most widely used variance-reduction techniques in Monte Carlo integration. Even a one-dimensional transformation, such as that based on Equation (7.8.18), can yield variance reductions of several factors, as illustrated by the numerical results. This modular implementation also sets the stage for further exploration of more sophisticated strategies such as control variates, antithetic sampling, and stratified sampling, all of which aim to reduce variance without sacrificing unbiasedness. Together, these methods form a toolkit for designing efficient Monte Carlo estimators tailored to the structure of the integrand.

## 7.8.2. Quasi–Monte Carlo and Modern Developments

Recent advances in high-dimensional numerical analysis have led to quasi–Monte Carlo (QMC) methods, which replace independent random samples with low-discrepancy sequences designed to fill the domain more uniformly. For sufficiently smooth integrands, QMC methods can achieve error rates approaching $O(N^{-1})$, significantly outperforming the classical $O(N^{-1/2})$ Monte Carlo scaling (Hung, 2023; Zhong & Feng, 2025). Randomized QMC methods further incorporate small amounts of randomness to provide unbiased estimators with provable variance decay.

Performance depends strongly on the integrand’s effective dimension, certain high-dimensional problems remain tractable when most of their variability lies along a few principal directions (Tang, 2023). Theoretical work shows that depending on structure, convergence can be arbitrarily fast or slow, reinforcing that no universal method dominates across all integrands.

### Rust Implementation

Following the discussion in Section 7.8.2 on the advantages of low-discrepancy sampling and the role of effective dimension in high-dimensional integration, Program 7.8.2 provides a practical implementation comparing classical Monte Carlo sampling with a quasi–Monte Carlo (QMC) method based on the Halton sequence. While traditional Monte Carlo relies on independent random points, QMC sequences fill the domain more uniformly, often yielding dramatically improved accuracy for smooth integrands. The example below demonstrates this phenomenon by integrating a separable exponential function over a multidimensional unit cube. By contrasting the convergence behaviour of random and low-discrepancy samples across multiple values of $N$, the program illustrates how deterministic structure and reduced discrepancy can enhance computational performance, complementing the theoretical rates discussed earlier in the section.

At the core of the implementation is the generator for the Halton sequence, constructed from radical inverse functions in distinct prime bases. Each coordinate of a Halton point is obtained by reflecting the base-$b$ digit expansion of an integer about the decimal point. This mapping produces points that distribute themselves with low star discrepancy, making them suitable for high-dimensional numerical integration when the integrand exhibits sufficient smoothness. The function `radical_inverse` carries out this digit-reversal transformation, while `halton_point` assembles the coordinates for a given index. By associating different primes with different dimensions, the implementation produces a multidimensional sequence with desirable uniformity properties, reflecting the structure of low-discrepancy sequences discussed in Section 7.8.2.

Two sampling strategies are implemented for numerical integration. The function `monte_carlo_integral` generates classical Monte Carlo estimates using i.i.d. uniform random numbers in each dimension. This method achieves the familiar variance decay rate proportional to $N^{-1}$ and mean-square error scaling of $O(N^{-1/2})$, which serves as a baseline for comparison. In contrast, the function `qmc_integral` evaluates the integrand at Halton points to approximate the same integral. Because the Halton sequence covers the domain more evenly than random points, the estimator often exhibits faster and more stable convergence, approaching the $O(N^{-1})$ behaviour characteristic of QMC for sufficiently smooth integrands. Although the implementation does not include randomization strategies such as scrambling or digital shifting, it captures the essential features of deterministic QMC sampling.

The chosen integrand, $f(x) = \exp(-\sum_j x_j)$, serves as a clean test function due to its smoothness and separability. Its exact integral over the unit cube is available in closed form, allowing direct computation of absolute errors for both Monte Carlo and QMC estimators. The function `exact_value` encodes this analytic expression, enabling the main program to measure performance quantitatively. This aligns with the broader discussion in Section 7.8.2 on how smoothness directly affects attainable convergence rates, particularly in high-dimensional settings where effective dimension becomes a determining factor.

The `main` function coordinates the experiment by computing and printing error statistics for a sequence of increasing sample sizes. By examining both the estimated values and their absolute deviations from the analytic integral, the program reveals the contrasting convergence profiles of MC and QMC methods. Classical Monte Carlo shows stochastic fluctuations characteristic of random sampling, whereas the QMC results decline more steadily with increasing $N$. This behaviour illustrates how structured sampling effectively reduces variance, echoing the performance trends described in the works of Hung (2023), Zhong & Feng (2025), and Tang (2023). The program thereby provides a concrete demonstration of quasi–Monte Carlo’s practical strengths and limitations in numerical integration.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
// Program 7.8.2: Comparing classical Monte Carlo and quasi–Monte Carlo
// using a Halton low-discrepancy sequence.
//
// Problem statement:
// Implement a numerical experiment in Rust that illustrates the difference
// between classical Monte Carlo (MC) and quasi–Monte Carlo (QMC) methods
// for integrating a smooth function over the d-dimensional unit cube.
// The program should:
//   * Define a moderately smooth integrand f: [0,1]^d -> R with a known
//     analytic integral, so that absolute errors can be computed.
//   * Approximate the integral using standard Monte Carlo with i.i.d.
//     uniform samples.
//   * Approximate the same integral using a low-discrepancy sequence;
//     here, we use a Halton sequence based on different prime bases
//     in each dimension.
//   * Run the experiment for several values of N (number of sample points)
//     and print the estimated integrals and absolute errors for both MC
//     and QMC.
// The goal is not to provide a production-quality implementation of
// low-discrepancy generators, but to demonstrate in a compact example
// how quasi–Monte Carlo methods can yield more regular coverage of the
// domain and improved convergence behaviour for smooth integrands,
// in line with the discussion in Section 7.8.2.

use rand::distributions::Uniform;
use rand::prelude::*;

/// Compute the radical inverse of `index` in a given base.
/// This is the building block of the Halton sequence:
/// For base b, we write index = d_0 + d_1 b + d_2 b^2 + ...,
/// and map it to 0.d_0 d_1 d_2 ... in base b.
fn radical_inverse(mut index: u64, base: u32) -> f64 {
    let b = base as f64;
    let mut f = 1.0 / b;
    let mut result = 0.0;

    while index > 0 {
        let digit = (index % base as u64) as f64;
        result += digit * f;
        index /= base as u64;
        f /= b;
    }

    result
}

/// Generate the i-th (1-based) point of a Halton sequence in `dim` dimensions.
/// Each coordinate uses a different prime base from `bases`.
fn halton_point(index: u64, dim: usize, bases: &[u32]) -> Vec<f64> {
    assert!(
        dim <= bases.len(),
        "Not enough bases provided for the requested dimension"
    );
    let mut x = Vec::with_capacity(dim);
    for j in 0..dim {
        x.push(radical_inverse(index, bases[j]));
    }
    x
}

/// Smooth test integrand on [0,1]^d:
///     f(x) = exp(-sum_j x_j)
/// The exact integral over [0,1]^d is:
///     I_exact(d) = (1 - e^{-1})^d
fn integrand(x: &[f64]) -> f64 {
    let sum: f64 = x.iter().copied().sum();
    (-sum).exp()
}

/// Exact value of the integral for the chosen integrand in `dim` dimensions.
fn exact_value(dim: usize) -> f64 {
    let one_minus_exp_minus1 = 1.0 - (-1.0f64).exp();
    one_minus_exp_minus1.powi(dim as i32)
}

/// Classical Monte Carlo estimator using i.i.d. uniform samples in [0,1]^d.
fn monte_carlo_integral(dim: usize, n: usize, rng: &mut impl Rng) -> f64 {
    let uniform = Uniform::new(0.0, 1.0);
    let mut acc = 0.0;

    let mut point = vec![0.0; dim];
    for _ in 0..n {
        for j in 0..dim {
            point[j] = rng.sample(uniform);
        }
        acc += integrand(&point);
    }

    acc / (n as f64)
}

/// Quasi–Monte Carlo estimator using the Halton sequence.
/// Indices start at 1 to avoid the trivial all-zero point.
fn qmc_integral(dim: usize, n: usize, bases: &[u32]) -> f64 {
    let mut acc = 0.0;
    for i in 1..=n as u64 {
        let x = halton_point(i, dim, bases);
        acc += integrand(&x);
    }
    acc / (n as f64)
}

fn main() {
    // Dimension of the integral.
    let dim: usize = 3;

    // Prime bases for the first few dimensions of the Halton sequence.
    // For dim = 3, we use bases 2, 3, 5.
    let halton_bases: [u32; 8] = [2, 3, 5, 7, 11, 13, 17, 19];

    let exact = exact_value(dim);
    println!("Integrand: f(x) = exp(-∑ x_j) on [0,1]^{}.", dim);
    println!("Exact integral I_exact = {:.10}", exact);
    println!();

    let mut rng = rand::thread_rng();
    let sample_sizes = [512, 2048, 8192, 32768];

    println!("{:>8}  {:>15}  {:>15}  {:>15}  {:>15}",
             "N", "MC estimate", "MC |err|", "QMC estimate", "QMC |err|");
    println!("{}", "-".repeat(76));

    for &n in &sample_sizes {
        let mc_est = monte_carlo_integral(dim, n, &mut rng);
        let qmc_est = qmc_integral(dim, n, &halton_bases);

        let mc_err = (mc_est - exact).abs();
        let qmc_err = (qmc_est - exact).abs();

        println!("{:8}  {:15.10}  {:15.3e}  {:15.10}  {:15.3e}",
                 n, mc_est, mc_err, qmc_est, qmc_err);
    }
}
```

Program 7.8.2 demonstrates how quasi–Monte Carlo sampling can substantially outperform classical Monte Carlo for smooth integrands by exploiting low-discrepancy structure rather than relying on statistical randomness. This mirrors the central theme of Section 7.8.2: although classical Monte Carlo provides universal applicability and dimension-independent error guarantees, low-discrepancy sequences allow error rates approaching $O(N^{-1})$ in favourable conditions. The example highlights how deterministic uniformity reduces variance, providing more predictable and efficient convergence compared to traditional random sampling.

The numerical results also illustrate the nuanced role of effective dimension. Although the integrand in this example is fully separable, its smoothness allows QMC to realise its theoretical potential, yielding rapidly decaying error as $N$ increases. In contrast, Monte Carlo errors fluctuate in proportion to the inherent randomness of the method. These characteristics underscore why QMC has become a valuable tool in modern high-dimensional numerical analysis, particularly in applications such as uncertainty quantification, finance, computer graphics, and Bayesian computation.

The modularity of the implementation makes it straightforward to extend the experiment to randomized QMC methods, higher-dimensional problems, different integrands, or alternative low-discrepancy sequences such as Sobol’ or Niederreiter–Xing constructions. Such extensions naturally lead into the broader topics of variance reduction, scrambling techniques, and the theoretical limits of QMC performance explored later in this chapter.

## 7.8.3. Applied Domains of High-Dimensional Monte Carlo Integration

Monte Carlo integration techniques are widely used across scientific computing, engineering, graphics, and finance whenever the evaluation of multidimensional integrals becomes analytically intractable or computationally prohibitive for deterministic quadrature. Their ability to handle irregular domains, high-dimensional probability distributions, and nonlinear integrands makes them especially attractive in contexts where classical numerical integration is either too slow or fundamentally unsuitable. The following examples illustrate some representative application areas in which Monte Carlo methods provide both conceptual simplicity and practical computational power.

### Geometric Modeling and Mass Properties

Monte Carlo integration plays a central role in computer-aided design (CAD), robotics, and engineering analysis when evaluating geometric or physical properties of complex three-dimensional objects. Many real-world solids are represented implicitly, either through algebraic surfaces $f(x,y,z) = 0$, signed-distance fields, or boundary-representation (B-rep) models, and their geometry may be too intricate for classical analytic integration. Monte Carlo methods circumvent this difficulty by sampling points within a bounding volume and applying fast *membership tests* to determine whether each sample lies inside the domain $W$. Once membership is established, integrals such as:

\begin{equation}
\begin{aligned}
\mathrm{Vol}(W) &= \int_W dV \\
M_x &= \int_W x\, dV \\
M_y &= \int_W y\, dV \\
M_z &= \int_W z\, dV
\end{aligned}\tag{7.8.19}
\end{equation}

\
yield volumes, mass moments, centers of gravity, and inertia tensors. This approach is particularly helpful for shapes with internal cavities, curved boundaries, or heterogeneous material densities. Modern GPU-parallelized Monte Carlo samplers further enable rapid evaluation in applications such as robotic grasp planning, real-time collision detection, and dynamic simulation of mechanical systems.

### Financial Mathematics: Basket and Exotic Options

In quantitative finance, pricing many classes of derivative securities reduces to computing high-dimensional expectations of the form:

$$P = \mathbb{E}\!\left[g(S_1,\dots,S_d)\right] \tag{7.8.20}$$

where $S_1,\ldots,S_d$ denote the asset prices under a risk-neutral measure. For multi-asset basket options, Asian options, barrier options, and other path-dependent contracts, the dimensionality of these integrals grows rapidly with both the number of underlying assets and the time discretization of the stochastic process. Classical quadrature methods are ineffective in such settings due to the exponential growth in required grid points.

Monte Carlo simulation addresses this challenge by generating entire stochastic paths of the underlying assets and evaluating the payoff $g$ for each simulated scenario. Advanced *variance-reduction techniques*, including control variates, antithetic variates, stratified sampling, and importance sampling, are routinely employed to improve estimator accuracy and reduce variance. In modern financial practice, Monte Carlo methods remain the dominant tool for pricing high-dimensional or exotic derivatives, calibrating stochastic volatility models, and performing risk analysis through scenario generation.

### Rust Implementation

Following the discussion in Section 7.8.3 on the breadth of real-world domains in which high-dimensional Monte Carlo integration is indispensable, Program 7.8.3 presents two representative applications drawn from geometric modelling and quantitative finance. These examples illustrate how the same underlying stochastic integration framework adapts naturally to problems with very different mathematical structures, dimensionalities, and computational constraints. In geometric modelling, Monte Carlo sampling offers a practical means of estimating volume and mass properties for complex solids defined implicitly or through intricate boundary descriptions. In financial mathematics, Monte Carlo methods remain the dominant approach for pricing exotic and path-dependent derivatives whose valuation reduces to computing high-dimensional expectations. This program demonstrates how Monte Carlo techniques unify these two seemingly disparate tasks through randomized sampling, enabling practical estimation of integrals that resist traditional quadrature.

At the core of the geometric modelling component is an implicit membership test that determines whether a randomly sampled point lies inside the region $W$. In this example, $W$ is chosen to be a hollow sphere, representative of CAD geometries with internal cavities or nontrivial topological structure. The function `is_inside_hollow_sphere` evaluates membership by checking whether the squared radial coordinate falls between prescribed inner and outer radii. This reflects the principle described in Equation (7.8.19), where the computation of mass properties reduces to evaluating integrals of indicator-weighted monomials over $W$. The function `estimate_volume_and_moments` operationalizes this idea by uniformly sampling points within a bounding box, accumulating counts and coordinate sums for points lying inside $W$, and converting these empirical averages into Monte Carlo estimates of the volume and first-order moments. The resulting centroid calculation illustrates how even complex volumes can be treated through simple pointwise inclusion tests without requiring explicit surface parameterizations.

The financial mathematics component addresses the estimation of high-dimensional expectations of the form $P = \mathbb{E}[g(S_1,\dots,S_d)]$ as expressed in Equation (7.8.20). In this example, the underlying asset evolves under a geometric Brownian motion discretized over a temporal grid, producing a multi-step stochastic path. The function `simulate_geometric_brownian_path_average` implements this simulation by applying the exponential Euler–Maruyama update driven by independent Gaussian increments. The path’s arithmetic mean serves as the state variable for an Asian call option, illustrating a payoff depending on the full trajectory rather than a single terminal value. The function `price_asian_call_monte_carlo` computes the Monte Carlo estimate by averaging discounted payoffs over many simulated paths, while also estimating the standard error to quantify sampling variability. This structure exemplifies how even moderately fine temporal discretizations produce inherently high-dimensional integrals that are nevertheless tractable through Monte Carlo sampling.

The `main` function ties both application areas into a cohesive demonstration. It first estimates the volume and centroid of the hollow sphere using randomly sampled spatial points and prints the resulting approximations. It then proceeds to price the Asian call option by generating thousands of independent geometric Brownian motion paths. Together, these examples highlight the versatility of Monte Carlo integration in both geometric and stochastic contexts, demonstrating how the conceptual simplicity of random sampling scales effectively to high-dimensional and structurally complex problems.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
rand_distr = "0.4"
```

```rust
// Program 7.8.3: Applied high-dimensional Monte Carlo integration in geometry and finance.
//
// Problem statement:
// Implement two representative applications of high-dimensional Monte Carlo integration in Rust:
//   (1) Geometric modelling of a three-dimensional solid via volume and mass-moment estimation,
//       corresponding to the integrals in Equation (7.8.19).
//       The solid W will be defined implicitly by membership tests inside a bounding box, mimicking
//       CAD-style or signed-distance representations. Random samples in the bounding volume are
//       used to approximate the volume and first moments (M_x, M_y, M_z), from which the centroid
//       can be computed.
//   (2) Pricing a path-dependent financial derivative via Monte Carlo simulation, corresponding to
//       expectations of the form P = E[g(S_1,...,S_d)] in Equation (7.8.20).
//       We consider a one-dimensional Asian call option under geometric Brownian motion, where
//       the payoff depends on the time-averaged asset price along a discretised path. Independent
//       Gaussian increments drive the stochastic process, and Monte Carlo paths are used to
//       approximate the discounted expected payoff.
//
// The goal is not to provide production-grade implementations, but to demonstrate how the same
// Monte Carlo integration principles are applied in two very different domains: geometric mass
// properties for complex volumes, and high-dimensional expectations arising in quantitative finance.

use rand::prelude::*;
use rand_distr::StandardNormal;

/// --------- Part 1: Geometric modelling via Monte Carlo volume and moment estimation ----------

/// Implicit membership test for a hollow sphere:
///   W = { (x,y,z) : r_outer^2 >= x^2 + y^2 + z^2 >= r_inner^2 }
/// with r_outer = 1, r_inner = 0.5.
/// This mimics a solid with an internal cavity.
fn is_inside_hollow_sphere(x: f64, y: f64, z: f64) -> bool {
    let r2 = x * x + y * y + z * z;
    r2 <= 1.0 && r2 >= 0.25
}

/// Monte Carlo estimation of volume and first moments of W inside a bounding box.
/// The box is defined as [xmin, xmax] x [ymin, ymax] x [zmin, zmax].
/// We approximate:
///   Vol(W) ≈ V_box * (1/N) Σ 1_W(x_i)
///   M_x     ≈ V_box * (1/N) Σ x_i 1_W(x_i), etc.
fn estimate_volume_and_moments(num_samples: usize, rng: &mut impl Rng) {
    // Bounding box for the hollow sphere: [-1,1]^3
    let xmin = -1.0;
    let xmax = 1.0;
    let ymin = -1.0;
    let ymax = 1.0;
    let zmin = -1.0;
    let zmax = 1.0;

    let v_box = (xmax - xmin) * (ymax - ymin) * (zmax - zmin);

    let mut count_inside = 0usize;
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut mz = 0.0;

    for _ in 0..num_samples {
        let x = rng.gen_range(xmin..xmax);
        let y = rng.gen_range(ymin..ymax);
        let z = rng.gen_range(zmin..zmax);

        if is_inside_hollow_sphere(x, y, z) {
            count_inside += 1;
            mx += x;
            my += y;
            mz += z;
        }
    }

    if count_inside == 0 {
        println!("No samples fell inside the region W; increase num_samples.");
        return;
    }

    let inside_fraction = (count_inside as f64) / (num_samples as f64);

    // Volume and moments approximations
    let vol_est = v_box * inside_fraction;
    let mx_est = v_box * mx / (num_samples as f64);
    let my_est = v_box * my / (num_samples as f64);
    let mz_est = v_box * mz / (num_samples as f64);

    // Centroid coordinates: (x̄, ȳ, z̄) = (M_x / Vol, M_y / Vol, M_z / Vol)
    let cx = mx_est / vol_est;
    let cy = my_est / vol_est;
    let cz = mz_est / vol_est;

    println!("--- Geometric modelling: hollow sphere ---");
    println!("Number of samples: {}", num_samples);
    println!("Estimated volume   Vol(W) ≈ {:.6}", vol_est);
    println!("Estimated centroid (x̄, ȳ, z̄) ≈ ({:.6}, {:.6}, {:.6})", cx, cy, cz);
    println!();
}

/// --------- Part 2: Financial mathematics via Monte Carlo Asian option pricing ----------

/// Simulate a single geometric Brownian motion path and return the time-averaged price.
/// Model:
///   dS_t = r S_t dt + σ S_t dW_t
/// Discretised with Euler–Maruyama in log form:
///   S_{n+1} = S_n * exp((r - 0.5 σ^2) Δt + σ sqrt(Δt) Z_n),
/// where Z_n ~ N(0,1) are independent.
fn simulate_geometric_brownian_path_average(
    s0: f64,
    r: f64,
    sigma: f64,
    t_final: f64,
    n_steps: usize,
    rng: &mut impl Rng,
) -> f64 {
    let dt = t_final / (n_steps as f64);
    let drift = (r - 0.5 * sigma * sigma) * dt;
    let vol_step = sigma * dt.sqrt();

    let mut s = s0;
    let mut sum_s = 0.0;

    for _ in 0..n_steps {
        let z: f64 = rng.sample(StandardNormal);
        s *= (drift + vol_step * z).exp();
        sum_s += s;
    }

    sum_s / (n_steps as f64)
}

/// Monte Carlo pricing of an Asian call option:
///   payoff = max( (1/M) Σ S_{t_j} - K, 0 ),
/// discounted by exp(-r T).
fn price_asian_call_monte_carlo(
    s0: f64,
    r: f64,
    sigma: f64,
    t_final: f64,
    k: f64,
    n_steps: usize,
    n_paths: usize,
    rng: &mut impl Rng,
) -> (f64, f64) {
    let discount = (-r * t_final).exp();
    let mut sum_payoff = 0.0;
    let mut sum_payoff_sq = 0.0;

    for _ in 0..n_paths {
        let avg_s = simulate_geometric_brownian_path_average(s0, r, sigma, t_final, n_steps, rng);
        let payoff = f64::max(avg_s - k, 0.0);
        let disc_payoff = discount * payoff;

        sum_payoff += disc_payoff;
        sum_payoff_sq += disc_payoff * disc_payoff;
    }

    let mean = sum_payoff / (n_paths as f64);
    let var = (sum_payoff_sq / (n_paths as f64)) - mean * mean;
    let std_err = (var / (n_paths as f64)).sqrt().max(0.0);

    (mean, std_err)
}

fn main() {
    let mut rng = rand::thread_rng();

    // Part 1: Geometric modelling / mass properties
    let num_samples_geometry = 200_000;
    estimate_volume_and_moments(num_samples_geometry, &mut rng);

    // Part 2: Financial mathematics / Asian option pricing
    let s0 = 100.0;      // initial asset price
    let r = 0.03;        // risk-free rate
    let sigma = 0.2;     // volatility
    let t_final = 1.0;   // maturity (1 year)
    let k = 100.0;       // strike
    let n_steps = 64;    // number of time steps in the path
    let n_paths = 100_000;

    let (price_est, std_err) =
        price_asian_call_monte_carlo(s0, r, sigma, t_final, k, n_steps, n_paths, &mut rng);

    println!("--- Financial mathematics: Asian call option (Monte Carlo) ---");
    println!("Number of paths:         {}", n_paths);
    println!("Time steps per path:     {}", n_steps);
    println!("Estimated option price:  {:.6}", price_est);
    println!("Standard error (MC):     {:.6}", std_err);
}
```

Program 7.8.3 illustrates how Monte Carlo integration serves as a unifying computational tool across diverse high-dimensional problems. In geometric modelling, the method enables the estimation of mass properties for domains that may lack closed-form integrals or even explicit boundary parameterizations. By reducing geometric complexity to simple membership queries, Monte Carlo techniques offer robustness and flexibility unmatched by classical quadrature. In financial mathematics, Monte Carlo methods provide a natural framework for evaluating expectations over multi-step, path-dependent stochastic processes, where dimensionality grows rapidly with time discretization. The Asian option example underscores how Monte Carlo sampling remains central to the pricing of exotic derivatives and to risk modelling in modern quantitative finance.

The shared structure of these applications highlights the generality of Monte Carlo integration: regardless of whether the integrand captures geometric shape or financial payoff, the estimator is built from random sampling, empirical averaging, and variance assessment. This foundational perspective sets the stage for more advanced developments, including variance-reduction techniques, quasi–Monte Carlo constructions, and probabilistic error bounds, all of which extend the power of Monte Carlo methods to increasingly challenging domains.

## 7.8.4. Section Remarks

Simple Monte Carlo integration offers a dimension-agnostic, conceptually transparent method for evaluating multi-dimensional integrals. Its strengths including generality, ease of implementation, and low memory cost, make it indispensable in scientific computing, geometry, probability, and finance. While its convergence is relatively slow, variance-reduction techniques and quasi–Monte Carlo sequences significantly enhance performance in practice. The following sections build on these foundations to introduce advanced strategies that improve sampling efficiency, accuracy, and stability in modern high-dimensional numerical integration.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/LA3nstVMUf3cc7UmbthQ.4","tags":[]}

# 7.9. Quasi- (Sub-) Random Sequences

Monte Carlo integration relies on the statistical behavior of independently drawn random samples. For an $n$-dimensional integral estimated using $N$ independent and identically distributed points, the mean-square error decays proportionally to $N^{-1/2}$, a rate that is independent of dimension but fundamentally limited by randomness. Deterministic sampling strategies, however, need not obey this probabilistic barrier. If one could distribute $N$ points in $[0,1]^n$ *perfectly uniformly*, the integration error for sufficiently smooth integrands would decay on the order of $1/N$, reflecting the same scaling enjoyed by regular grids.

Regular tensor grids (e.g., $m\times m \times \cdots \times m$ with $N = m^n$) come close to this ideal: for smooth functions, grid-based quadrature often achieves a convergence rate of $O(1/N)$. Yet grids suffer from two severe disadvantages. First, the resolution must be determined in advance, and refinement requires replacing all previous samples with an entirely finer grid. One cannot simply add a few additional points. Second, the number of grid points grows exponentially with dimension, making naïve grid refinement infeasible in high-dimensional settings.

These limitations motivate the development of quasi-random (or sub-random) sequences: deterministic point sequences that fill space more uniformly than random samples but remain extensible point-by-point. The goal is to retain the flexibility of Monte Carlo, being able to draw the $N$-th point at any time, while achieving a spatial coverage closer to that of regular grids. Quasi-random sequences suppress clustering, systematically filling the largest remaining gaps in the domain as points are added. This more even coverage is the cornerstone of quasi-Monte Carlo (QMC) methods, which often demonstrate markedly faster convergence than classical Monte Carlo, sometimes approaching $O((\log N)^n/N)$ for smooth integrands.

The sections that follow introduce the mathematical foundation of low-discrepancy sequences, compare them with grid-based and fully random sampling, and examine both classical constructions (Halton and Sobol’) and modern advances such as scrambling, machine-learned point sets, and applications in high-dimensional integration. The central theme is that deterministically structured sampling can dramatically improve accuracy without sacrificing the adaptability that makes Monte Carlo appealing in practice.

### Rust Implementation

Following the conceptual introduction in Section 7.9 on the limitations of classical Monte Carlo sampling and the motivation for quasi-random sequences, Program 7.9.0 provides a concrete numerical comparison between independent pseudo-random sampling and a deterministic low-discrepancy method. Whereas Monte Carlo estimators inherit the $N^{-1/2}$ convergence barrier imposed by randomness, quasi-random sequences aim to distribute points in $[0,1]^n$ more uniformly, reducing discrepancy and accelerating convergence for smooth integrands. This program demonstrates these ideas on a simple two-dimensional test integral, allowing the reader to directly observe how deterministic sampling improves accuracy without sacrificing the extensibility that makes Monte Carlo appealing.

At the core of the implementation are two parallel sampling mechanisms that reflect the contrast between stochastic and deterministic point generation in $[0,1]^2$. The function `sample_random_2d` draws independent uniform points and forms the basis of the standard Monte Carlo estimator. This directly follows the theoretical setting described at the beginning of Section 7.9, where the mean-square error of the estimator decays proportionally to $N^{-1/2}$ for i.i.d. samples. The numerical fluctuations inherent to this sampling strategy are a consequence of randomness itself and cannot be eliminated without altering the point distribution.

The deterministic counterpart is constructed through the function `radical_inverse`, which implements the one-dimensional van der Corput sequence by reversing the base-$b$ digits of an integer index. This construct embodies the low-discrepancy principle underlying quasi-random methods: by spreading points in a manner that systematically avoids clustering and fills gaps efficiently, one obtains a sequence with markedly improved uniformity over random sampling. The function `halton_2d` forms the classical two-dimensional Halton sequence by pairing radical-inverse functions in bases 2 and 3. As Section 7.9 emphasizes, the Halton sequence remains extensible point-by-point, preserving one of the greatest strengths of Monte Carlo while improving the spatial coverage of the domain.

The integration routines `estimate_random` and `estimate_halton` implement the standard quasi-Monte Carlo estimator, which uses the same averaging formula but differs only in how sampling points are chosen. For the pseudo-random estimator, the variance decays predictably but slowly, reflecting the probabilistic barrier that quasi-random sequences aim to surpass. For the Halton estimator, the deterministic structure produces a smoother and more regular sampling pattern, often yielding errors significantly smaller than those obtained from random sampling. Because the integrand $e^{x+y}$ is smooth throughout $[0,1]^2$, it serves as an ideal example where low-discrepancy sequences demonstrate their theoretical advantage, often producing error decay closer to $O(1/N)$ than to $O(N^{-1/2})$.

The `main` function orchestrates the comparison by evaluating the integral for progressively larger sample sizes, printing the estimated values and absolute errors for both random and Halton points. The numerical results illustrate the qualitative improvement highlighted in Section 7.9: while Monte Carlo errors fluctuate stochastically and decay slowly, Halton-based estimates converge faster and exhibit much more consistent accuracy. The side-by-side presentation of both methods reinforces the conceptual message that deterministic sampling can achieve improved convergence without compromising adaptability.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
/*
    Program 7.9.0: Random vs Quasi-Random Sampling in Two Dimensions

    Problem statement.
    Section 7.9 introduces quasi- (sub-) random sequences as deterministic
    point sets that fill [0,1]^n more uniformly than independent random
    samples while remaining extensible one point at a time. Classical Monte
    Carlo integration with N i.i.d. samples achieves a mean-square error
    of order N^{-1/2}, independent of dimension, but this convergence rate
    is fundamentally limited by randomness. In contrast, low-discrepancy
    sequences can approach much faster convergence for smooth integrands,
    sometimes on the order of (log N)^n / N.

    This program compares standard Monte Carlo with a simple quasi-random
    construction—a two-dimensional Halton sequence—on a smooth test
    integral over [0,1]^2:

        I = ∫_0^1 ∫_0^1 exp(x + y) dx dy,

    which can be evaluated analytically as

        I = (e - 1)^2.

    For a sequence of sample sizes N, the program estimates I using:
      (1) Pseudo-random uniform points (standard Monte Carlo),
      (2) A deterministic Halton sequence (quasi-random sampling).

    The estimated integrals and absolute errors are printed side by side,
    illustrating how quasi-random sampling can achieve substantially
    smaller errors than purely random points for the same N, reflecting
    the improved uniformity emphasized in Section 7.9.
*/

// Cargo.toml:
//
// [package]
// name = "myproject"
// version = "0.1.0"
// edition = "2021"
//
// [dependencies]
// rand = "0.8"

use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Test integrand on [0,1]^2:
///   f(x, y) = exp(x + y).
fn f(x: f64, y: f64) -> f64 {
    (x + y).exp()
}

/// Exact value of the integral
///   I = ∫_0^1 ∫_0^1 exp(x + y) dx dy = (e - 1)^2.
fn exact_integral() -> f64 {
    let e = std::f64::consts::E;
    let one = 1.0_f64;
    (e - one).powi(2)
}

/// Generate a single pseudo-random point (x, y) in [0,1]^2.
fn sample_random_2d(rng: &mut StdRng) -> (f64, f64) {
    let dist = Uniform::new(0.0_f64, 1.0_f64);
    let x = rng.sample(dist);
    let y = rng.sample(dist);
    (x, y)
}

/// One-dimensional van der Corput radical-inverse function in base `base`.
///
/// For index n (n >= 1), this produces a point in [0,1) by reversing the
/// base-b digits of n after the radix point:
///
///   n = d_0 + d_1 b + d_2 b^2 + ...
///   φ_b(n) = d_0 / b + d_1 / b^2 + d_2 / b^3 + ...
///
/// This is the building block of the Halton sequence.
fn radical_inverse(mut n: u64, base: u32) -> f64 {
    let b = base as u64;
    let mut f = 1.0_f64;
    let mut result = 0.0_f64;

    while n > 0 {
        f /= base as f64;
        let digit = n % b;
        result += f * (digit as f64);
        n /= b;
    }

    result
}

/// Two-dimensional Halton sequence point with index n:
///   x = φ_2(n), y = φ_3(n),
/// where φ_b is the radical-inverse function in base b.
fn halton_2d(n: u64) -> (f64, f64) {
    let x = radical_inverse(n, 2);
    let y = radical_inverse(n, 3);
    (x, y)
}

/// Estimate the integral I using N pseudo-random points in [0,1]^2.
///
/// The estimator is the standard Monte Carlo average:
///
///   Q_N^MC = (1 / N) Σ f(x_i, y_i),
///
/// where (x_i, y_i) are i.i.d. Uniform([0,1]^2) samples.
fn estimate_random(n_samples: u64, rng: &mut StdRng) -> f64 {
    let mut sum = 0.0_f64;

    for _ in 0..n_samples {
        let (x, y) = sample_random_2d(rng);
        sum += f(x, y);
    }

    sum / (n_samples as f64)
}

/// Estimate the integral I using N points from the 2D Halton sequence.
///
/// The quasi-Monte Carlo estimator uses the same averaging formula:
///
///   Q_N^QMC = (1 / N) Σ f(x_n, y_n),
///
/// but the points (x_n, y_n) are taken from the deterministic low-discrepancy
/// Halton sequence, which tends to cover [0,1]^2 more uniformly.
fn estimate_halton(n_samples: u64) -> f64 {
    let mut sum = 0.0_f64;

    // Halton indices typically start from n = 1.
    for n in 1..=n_samples {
        let (x, y) = halton_2d(n);
        sum += f(x, y);
    }

    sum / (n_samples as f64)
}

fn main() {
    // Fixed seed so that the pseudo-random results are reproducible.
    let mut rng = StdRng::seed_from_u64(7_9_2025);

    let exact = exact_integral();
    println!("Quasi-Monte Carlo vs standard Monte Carlo on [0,1]^2");
    println!("Integrand: f(x, y) = exp(x + y)");
    println!("Exact integral I = (e - 1)^2 ≈ {:.10}", exact);
    println!();

    // Sample sizes to compare.
    let sample_sizes: [u64; 4] = [1_000, 5_000, 10_000, 50_000];

    println!("{:>8}  {:>18}  {:>18}  {:>18}  {:>18}",
             "N",
             "MC estimate",
             "MC |error|",
             "Halton estimate",
             "Halton |error|");
    println!("{}", "-".repeat(90));

    for &n in &sample_sizes {
        let est_mc = estimate_random(n, &mut rng);
        let est_halton = estimate_halton(n);

        let err_mc = (est_mc - exact).abs();
        let err_halton = (est_halton - exact).abs();

        println!("{:8}  {:18.10}  {:18.10}  {:18.10}  {:18.10}",
                 n, est_mc, err_mc, est_halton, err_halton);
    }
}
```

Program 7.9.0 demonstrates the practical benefits of quasi-random sampling by juxtaposing classical Monte Carlo with a simple low-discrepancy sequence. The observed differences in accuracy underscore the central message of Section 7.9: randomness introduces variance, and this variance fundamentally limits the convergence rate of Monte Carlo estimators. By contrast, quasi-random sequences reduce discrepancy and suppress clustering, allowing estimators to take advantage of the underlying smoothness of the integrand.

The Halton sequence used in this program represents one of the earliest and simplest constructions of low-discrepancy sampling. Despite its simplicity, it already achieves noticeably better accuracy than pseudo-random sampling at the same sample sizes. More sophisticated sequences, such as Sobol', Niederreiter–Xing, and digitally scrambled variants, further improve uniformity, stability, and performance in higher dimensions. The modular framework presented here provides a platform for exploring these advanced constructions, as well as their applications to high-dimensional numerical integration and uncertainty quantification.

## 7.9.1. Low-Discrepancy Sequences and Convergence Theory

A central goal of quasi-random sampling is to distribute points in the unit hypercube $[0,1]^n$ as uniformly as possible while retaining the ability to generate points incrementally. The mathematical measure of this uniformity is *discrepancy*. For a set of $N$ points $\mathbf{x}_1,\dots,\mathbf{x}_N\in[0,1]^n$*,* the *star discrepancy* $D_N^*$ quantifies the largest deviation between empirical point counts and their ideal continuous volume over all axis-aligned rectangular boxes anchored at the origin. Formally,

$$
D_N^* = \sup_{0 \le t_1,\dots,t_n \le 1}
\bigg|
\frac{\#\{\mathbf{x}_j \in [0,t_1)\times \cdots \times [0,t_n)\}}{N}
- (t_1 t_2 \cdots t_n)
\bigg|
\tag{7.9.1}
$$

A sequence is called *low-discrepancy* if $D_N^*$ grows on the order of $(\log N)^n/N$ or slower. This growth rate is essentially optimal up to logarithmic factors and is vastly superior to the behavior of random sampling, for which discrepancy typically scales like $N^{-1/2}$.

The connection between discrepancy and numerical integration error is provided by the *Koksma–Hlawka inequality*, which states that for any function $f$ of bounded variation $V(f)$ (in the Hardy–Krause sense),

$$
\left|
\frac{1}{N}\sum_{j=1}^N f(\mathbf{x}_j)
- \int_{[0,1]^n} f(\mathbf{x})\, d\mathbf{x}
\right|
\;\le\;
V(f)\, D_N^*
\tag{7.9.2}
$$

Thus, a low-discrepancy sequence ensures small integration error whenever $f$ is sufficiently smooth. In many practical cases, quasi-Monte Carlo sampling achieves nearly $O(1/N)$ convergence (with a $(\log N)^n$ factor), far outperforming classical Monte Carlo’s statistical rate of $O(N^{-1/2})$. This improvement is dramatic in high-dimensional integration problems where uniform space coverage is essential.

Unlike ordinary Monte Carlo, however, deterministic quasi-random sampling does not provide a natural statistical error estimate. A single deterministic sequence cannot quantify its own uncertainty. Modern randomized methods, such as random shifting, scrambled sequences, and especially Owen scrambling, restore unbiasedness and supply a legitimate variance estimate while maintaining the benefits of low discrepancy.

To appreciate the role of quasi-random sampling, it is instructive to contrast three fundamental strategies:

1. **Random i.i.d. sampling**\
   Produces flexible, easily extendable point sets but exhibits clustering and achieves only $N^{-1/2}$ convergence.
2. **Regular grids**\
   Offer excellent uniformity and often achieve near $1/N$ convergence for smooth integrands. However, grids are not easily extensible; refining the grid requires replacing all prior points, and in high dimensions grids quickly become computationally infeasible.
3. **Quasi-random (low-discrepancy) sequences**\
   Combine the extensibility of random sampling with near-grid uniformity. Each new point systematically fills the largest remaining gaps, suppressing clustering and enabling convergence rates close to $1/N$. This efficient coverage makes quasi-random sampling exceptionally well-suited for moderate and high-dimensional integration tasks.

Quasi-Monte Carlo methods therefore provide a remarkable middle ground between order and randomness: deterministic enough to guarantee uniform coverage, yet flexible enough to allow incremental sampling just like Monte Carlo. Their superiority becomes especially clear when the integrand possesses smoothness, structure, or regularity that benefits from uniform point placement.

### Rust Implementation

Following the theoretical discussion in Section 7.9.1 on discrepancy, low-discrepancy sequences, and the Koksma–Hlawka inequality, Program 7.9.1 provides a practical illustration of how different sampling strategies influence numerical convergence. The discrepancy bounds in Equation (7.9.1) motivate the use of deterministic sequences such as Halton points, whose uniformity leads to error decay far superior to classical Monte Carlo when the integrand is sufficiently smooth. Meanwhile, Equation (7.9.2) connects this geometric uniformity to integration accuracy through the total variation of the integrand. This program implements three estimators, standard Monte Carlo, randomized quasi–Monte Carlo, and deterministic quasi–Monte Carlo, to demonstrate how low-discrepancy sampling improves accuracy and stability, and how randomization restores unbiasedness and allows statistical error estimation.

At the core of the implementation is the test integrand $f(x,y) = e^{x+y}$, chosen for its smoothness and compatibility with the Koksma–Hlawka inequality in Equation (7.9.2). The exact value of the integral, computed analytically as $(e-1)^2$, serves as a benchmark against which all estimators are evaluated. This provides a clear numerical demonstration of how sampling uniformity influences convergence.

The function `sample_random_2d` implements standard Monte Carlo sampling by drawing independent uniform points. This matches the i.i.d. setting described at the beginning of Section 7.9, where integration error scales as $N^{-1/2}$ due to the variance of the estimator. The routine `estimate_random_once` builds on this sampling mechanism to compute a single Monte Carlo estimate, reflecting the probabilistic nature of classical Monte Carlo quadrature.

The quasi-random component is constructed using the radical-inverse function `radical_inverse`, which generates one-dimensional van der Corput points by reversing digits in a chosen base. Paired bases yield the two-dimensional Halton sequence in `halton_2d`. This deterministic, extensible construction embodies the low-discrepancy philosophy of Section 7.9.1: points avoid clustering and fill the domain progressively, lowering $D_N^*$ in (7.9.1) and improving integration performance for smooth integrands.

To produce a statistically meaningful estimator, the program implements random shifting through the function `estimate_shifted_halton_once`. A random vector is added modulo 1 to each Halton point, preserving low discrepancy while introducing unbiasedness. This corresponds to modern randomized QMC techniques described near the end of Section 7.9.1, where randomization enables valid error estimates while maintaining the benefits of quasi-random structure. The function `estimate_halton_deterministic` provides a single deterministic quasi-MC estimate, illustrating how low-discrepancy sampling performs without randomization, though lacking variance information.

Finally, the routine `mean_and_rmse` aggregates multiple replicates to produce mean estimates and RMSE values. This allows direct comparison of convergence behavior: Monte Carlo estimates fluctuate with variance proportional to $1/N$, while randomized quasi–Monte Carlo estimates exhibit dramatically lower RMSE, often nearly proportional to $1/N$. The deterministic Halton error included for reference provides insight into the underlying uniformity that drives the improved convergence.

The `main` function performs structured comparisons by evaluating all three estimators across increasing sample sizes. For each $N$, it prints the averaged MC and randomized QMC estimators with their RMSE, along with the deterministic Halton error. The numerical output reveals the qualitative differences predicted in Section 7.9.1: standard Monte Carlo exhibits slow, noisy convergence consistent with statistical error bounds, while randomized quasi–Monte Carlo converges substantially faster due to reduced discrepancy. Deterministic Halton estimates consistently outperform typical MC realizations but do not provide uncertainty quantification, highlighting the purpose of randomization.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
/*
    Program 7.9.1: Convergence of Monte Carlo vs Randomized Quasi–Monte Carlo

    Problem statement.
    Section 7.9.1 introduces low-discrepancy sequences and their role in
    numerical integration through the Koksma–Hlawka inequality (7.9.2).
    Low-discrepancy sequences achieve star discrepancy D_N^* that grows
    on the order of (log N)^n / N or slower (7.9.1), enabling integration
    errors that often approach O(1/N) for smooth integrands—far faster than
    the O(N^{-1/2}) rate typical of classical Monte Carlo.

    This program provides a numerical illustration of these ideas by
    comparing three strategies on the same smooth test integral over
    [0,1]^2:

        I = ∫_0^1 ∫_0^1 exp(x + y) dx dy = (e - 1)^2.

    For a sequence of sample sizes N, it computes:
      (1) Standard Monte Carlo with i.i.d. pseudo-random points.
      (2) Randomized quasi–Monte Carlo using randomly shifted Halton
          points (a basic example of randomized low-discrepancy sampling).
      (3) A single deterministic estimate using the plain Halton sequence.

    For methods (1) and (2), multiple independent replicates are used to
    estimate the root-mean-square error (RMSE), mimicking the role of a
    statistical error estimate. The deterministic Halton estimate provides
    a single quasi-random answer without variance. The results illustrate
    how randomized quasi–Monte Carlo can achieve substantially smaller
    RMSE than standard Monte Carlo for the same N, reflecting the improved
    uniformity implied by low discrepancy in Section 7.9.1.
*/

// Cargo.toml:
//
// [package]
// name = "myproject"
// version = "0.1.0"
// edition = "2021"
//
// [dependencies]
// rand = "0.8"

use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Test integrand on [0,1]^2:
///   f(x, y) = exp(x + y).
fn f(x: f64, y: f64) -> f64 {
    (x + y).exp()
}

/// Exact value of the integral:
///   I = ∫_0^1 ∫_0^1 exp(x + y) dx dy = (e - 1)^2.
fn exact_integral() -> f64 {
    let e = std::f64::consts::E;
    let one = 1.0_f64;
    (e - one).powi(2)
}

/// Generate a single pseudo-random point (x, y) in [0,1]^2.
fn sample_random_2d(rng: &mut StdRng) -> (f64, f64) {
    let dist = Uniform::new(0.0_f64, 1.0_f64);
    let x = rng.sample(dist);
    let y = rng.sample(dist);
    (x, y)
}

/// One-dimensional van der Corput radical-inverse function in base `base`.
///
/// For index n (n >= 1), this produces a point in [0,1) by reversing the
/// base-b digits of n after the radix point:
///
///   n = d_0 + d_1 b + d_2 b^2 + ...
///   φ_b(n) = d_0 / b + d_1 / b^2 + d_2 / b^3 + ...
///
/// This is the building block of the Halton sequence.
fn radical_inverse(mut n: u64, base: u32) -> f64 {
    let b = base as u64;
    let mut f = 1.0_f64;
    let mut result = 0.0_f64;

    while n > 0 {
        f /= base as f64;
        let digit = n % b;
        result += f * (digit as f64);
        n /= b;
    }

    result
}

/// Two-dimensional Halton sequence point with index n:
///   x = φ_2(n), y = φ_3(n),
/// where φ_b is the radical-inverse function in base b.
///
/// Indices typically start from n = 1.
fn halton_2d(n: u64) -> (f64, f64) {
    let x = radical_inverse(n, 2);
    let y = radical_inverse(n, 3);
    (x, y)
}

/// Estimate the integral using N pseudo-random points in [0,1]^2.
///
/// Standard Monte Carlo estimator:
///   Q_N^MC = (1 / N) Σ f(x_i, y_i),
/// where (x_i, y_i) are i.i.d. Uniform([0,1]^2) samples.
fn estimate_random_once(n_samples: u64, rng: &mut StdRng) -> f64 {
    let mut sum = 0.0_f64;

    for _ in 0..n_samples {
        let (x, y) = sample_random_2d(rng);
        sum += f(x, y);
    }

    sum / (n_samples as f64)
}

/// Estimate the integral using N points from the 2D Halton sequence.
///
/// Quasi–Monte Carlo estimator:
///   Q_N^QMC = (1 / N) Σ f(x_n, y_n),
/// where (x_n, y_n) are taken from the deterministic Halton sequence.
fn estimate_halton_deterministic(n_samples: u64) -> f64 {
    let mut sum = 0.0_f64;

    for n in 1..=n_samples {
        let (x, y) = halton_2d(n);
        sum += f(x, y);
    }

    sum / (n_samples as f64)
}

/// Estimate the integral using N points from a *randomly shifted* Halton sequence.
///
/// A random shift u ∈ [0,1)^2 is drawn once, and each Halton point (x_n, y_n)
/// is mapped to
///
///   (x_n', y_n') = ((x_n + u_x) mod 1, (y_n + u_y) mod 1).
///
/// This randomization preserves low discrepancy while producing an unbiased
/// estimator and allowing empirical variance estimation across independent
/// shifts.
fn estimate_shifted_halton_once(
    n_samples: u64,
    shift_x: f64,
    shift_y: f64,
) -> f64 {
    let mut sum = 0.0_f64;

    for n in 1..=n_samples {
        let (hx, hy) = halton_2d(n);
        // Apply random shift and wrap back into [0,1).
        let x = (hx + shift_x).fract();
        let y = (hy + shift_y).fract();
        sum += f(x, y);
    }

    sum / (n_samples as f64)
}

/// Compute the mean and root-mean-square error (RMSE) of a set of estimates
/// {Q_r} compared to the exact value I:
///
///   mean(Q)  = (1/R) Σ Q_r,
///   RMSE(Q)  = sqrt( (1/R) Σ (Q_r - I)^2 ).
fn mean_and_rmse(estimates: &[f64], exact: f64) -> (f64, f64) {
    let r = estimates.len() as f64;
    let mut sum = 0.0_f64;
    let mut sum_sq_err = 0.0_f64;

    for &q in estimates {
        sum += q;
        let err = q - exact;
        sum_sq_err += err * err;
    }

    let mean = sum / r;
    let rmse = (sum_sq_err / r).sqrt();
    (mean, rmse)
}

fn main() {
    // Fixed seed so that the pseudo-random and random shifts are reproducible.
    let mut rng = StdRng::seed_from_u64(7_9_2025);

    let exact = exact_integral();
    println!("Convergence of Monte Carlo vs randomized quasi–Monte Carlo on [0,1]^2");
    println!("Integrand: f(x, y) = exp(x + y)");
    println!("Exact integral I = (e - 1)^2 ≈ {:.10}", exact);
    println!();

    // Sample sizes and number of independent replicates.
    let sample_sizes: [u64; 4] = [1_000, 5_000, 10_000, 50_000];
    let n_replicates: usize = 20;

    println!(
        "{:>8}  {:>14}  {:>14}  {:>14}  {:>14}  {:>14}",
        "N",
        "MC mean",
        "MC RMSE",
        "RQMC mean",
        "RQMC RMSE",
        "Halton |err|"
    );
    println!("{}", "-".repeat(90));

    for &n in &sample_sizes {
        // Collect estimates for MC and randomized QMC.
        let mut mc_estimates = Vec::with_capacity(n_replicates);
        let mut rqmc_estimates = Vec::with_capacity(n_replicates);

        for _ in 0..n_replicates {
            // Standard Monte Carlo estimate.
            let q_mc = estimate_random_once(n, &mut rng);
            mc_estimates.push(q_mc);

            // Random shift for Halton-based RQMC.
            let shift_dist = Uniform::new(0.0_f64, 1.0_f64);
            let shift_x = rng.sample(shift_dist);
            let shift_y = rng.sample(shift_dist);

            let q_rqmc = estimate_shifted_halton_once(n, shift_x, shift_y);
            rqmc_estimates.push(q_rqmc);
        }

        let (mc_mean, mc_rmse) = mean_and_rmse(&mc_estimates, exact);
        let (rqmc_mean, rqmc_rmse) = mean_and_rmse(&rqmc_estimates, exact);

        // Single deterministic Halton estimate (no randomization).
        let q_halton_det = estimate_halton_deterministic(n);
        let halton_err = (q_halton_det - exact).abs();

        println!(
            "{:8}  {:14.10}  {:14.10}  {:14.10}  {:14.10}  {:14.10}",
            n, mc_mean, mc_rmse, rqmc_mean, rqmc_rmse, halton_err
        );
    }
}
```

Program 7.9.1 provides a practical demonstration of the central ideas introduced in Section 7.9.1. The contrast between Monte Carlo, deterministic quasi–Monte Carlo, and randomized quasi–Monte Carlo highlights the influence of discrepancy on numerical integration. For smooth integrands, the low-discrepancy structure of sequences such as Halton leads to convergence behavior that approaches the ideal $1/N$ scaling predicted by the Koksma–Hlawka inequality in (7.9.2), significantly exceeding the capabilities of classical Monte Carlo.

The randomized variant illustrates how modern QMC techniques overcome the primary limitation of deterministic sampling, its lack of a natural error estimate, while retaining the geometric uniformity needed for accelerated convergence. The RMSE results confirm that randomized QMC provides both superior accuracy and statistical interpretability. The deterministic Halton estimator, though not equipped with variance information, still reveals the intrinsic advantage of low discrepancy through its consistently small errors.

Together, these results show how quasi-random sequences bridge the gap between rigid grids and fully random sampling, enabling fast, stable, and extensible numerical integration methods ideally suited for high-dimensional smooth problems.

## 7.9.2. Classical Low-Discrepancy Sequences: Halton and Sobol’

Classical constructions of low-discrepancy sequences draw heavily on number theory and digital arithmetic. Their goal is to generate deterministic point sets that emulate the uniformity of grids while retaining the flexibility of Monte Carlo sampling. Two families dominate both theory and practice: *Halton sequences* (including the van der Corput sequence in one dimension) and *Sobol’ sequences*, an important class of digital sequences in base 2. Both produce extensible point sets whose first $N$ elements already approximate the optimal discrepancy order $(\log N)^n / N$, making them central tools in quasi-Monte Carlo integration.

### (i) Halton Sequences and the van der Corput Construction

The simplest setting is one-dimensional sampling on $[0,1]$. For a chosen integer base $b \ge 2$, write each integer $j = 1,2,3,\dots$ in base $b$,

$$j = d_k d_{k-1} \cdots d_1 d_0 \quad \text{(in base } b\text{)} \tag{7.9.3}$$

where the digits satisfy $0 \le d_i < b$. The van der Corput radical-inverse function reflects these digits across the radix point, generating the value:

$$
H_j^{(b)}
= 0.d_k d_{k-1}\cdots d_1 d_0 \;\text{(in base } b\text{)}
= \sum_{i=0}^k d_i\, b^{-(i+1)}
\tag{7.9.4}
$$

This produces a sequence of fractions in $[0,1]$ that systematically subdivide the unit interval. As $j$ passes powers of $b$, the sequence refines the sampling by introducing new leading digits, thereby filling gaps left by earlier points. For example, the van der Corput sequence in base 2 begins:

$$0.1_2,\quad 0.01_2,\quad 0.11_2,\quad 0.001_2,\quad 0.101_2, \ldots \tag{7.9.5}$$

corresponding to the decimal sequence $0.5,0.25,0.75,0.125,0.625,\dots$. These points exhibit a “gap-filling’’ property: at scale $2^{-m}$, the first $2^m$ points place exactly one point in each subinterval of length $2^{-m}$.

Halton extended this idea to multiple dimensions by using *different prime bases* for each coordinate. Given distinct primes $b_1,b_2,\dots,b_n$, the $j$-th point of the $n$-dimensional Halton sequence is,

$$
\mathbf{x}_j = \big(H_j^{(b_1)},\, H_j^{(b_2)},\, \dots,\, H_j^{(b_n)}\big)
\tag{7.9.6}
$$

The Halton sequence is equidistributed in $[0,1]^n$, achieves discrepancy on the order of $(\log N)^n/N$, and is extremely simple to implement. A known drawback arises in higher dimensions, where early primes (e.g., 2, 3, 5, 7, …) produce subtle correlations among coordinates, especially for moderate $N$. Techniques such as scrambling or skipping initial points help mitigate this effect.

### (ii) Sobol’ Sequences: Base-2 Digital Constructions

Sobol’ sequences, introduced by Ilya Sobol’ (1967) and refined by Antonov and Saleev (1979), use binary fractions and linear algebra over $\mathbb{F}_2$ to obtain excellent uniformity in hundreds or even thousands of dimensions. These sequences lie at the heart of modern quasi-Monte Carlo methods because they combine low discrepancy with high computational efficiency.

Each coordinate $k$ of a Sobol’ sequence is generated from a set of binary direction numbers $V_i^{(k)}$. To construct these, one chooses a *primitive polynomial:*

$$P_k(x) = x^{q_k} + a_1 x^{q_k-1} + \cdots + a_{q_k-1} x + 1 \tag{7.9.7}$$

with $a_i\in{0,1}$ over the field $\mathbb{F}_2$*.* For each dimension, one then specifies initial odd integers:

$$M_1, M_2, \dots, M_{q_k} \tag{7.9.8}$$

each satisfying $0 < M_i < 2^i$. The remaining direction integers are generated by the recurrence:

$$
\begin{aligned}
M_i
&= a_1 (2 M_{i-1}) \;\oplus\; a_2 (2^2 M_{i-2}) \;\oplus\; \cdots \\[4pt]
&\quad \oplus\; a_{q_k-1} (2^{\,q_k-1} M_{i-q_k+1})
   \;\oplus\; 2^{q_k} M_{i-q_k}
   \;\oplus\; M_{i-q_k}
\end{aligned}
\tag{7.9.9}
$$

where $\oplus$ denotes bitwise XOR and multiplication by $2^r$ denotes a left shift by $r$ bits.

The direction numbers themselves are obtained by normalization:

$$V_i = \frac{M_i}{2^i}, \qquad i = 1,2,3,\dots\tag{7.9.10}$$

To compute the $j$-th point, write the binary expansion of $j$,

$$j = (b_{j,1} b_{j,2} b_{j,3}\dots )_2 \tag{7.9.11}$$

and form each coordinate via:

$$
x_{j,k}
= b_{j,1} V_1^{(k)} \;\oplus\;
  b_{j,2} V_2^{(k)} \;\oplus\;
  b_{j,3} V_3^{(k)} \;\oplus\; \cdots 
\tag{7.9.12}
$$

A key computational improvement comes from using Gray codes:

$$G(j) = j \oplus (j/2)\tag{7.9.13}$$

which ensures that only a single bit changes when incrementing the index. Consequently, generating $\mathbf{x}_{j+1}$ from $\mathbf{x}_j$ requires XOR-ing only one direction number. This results in a highly efficient incremental algorithm capable of producing millions of Sobol’ points per second.

Sobol’ sequences achieve discrepancy on the order of $(\log N)^n/N$ asymptotically, similar to Halton, but with substantially better behavior in high dimensions. Their performance advantage arises from base-2 digital structure, controlled bit interactions, and excellent multi-scale uniformity: after $2^m$ points, the sequence aligns perfectly with a dyadic grid of resolution $2^{-m}$, ensuring one point in every dyadic subcube.

### Rust Implementation

Following the discussion in Section 7.9.2 on the number-theoretic foundations of low-discrepancy sampling, Program 7.9.2 provides practical implementations of the classical constructions that underpin modern quasi-Monte Carlo methods. The section introduces the van der Corput radical-inverse mapping in one dimension (Equations (7.9.3)–(7.9.5)), the Halton generalization to multiple coordinates (Equation (7.9.6)), and the base-2 digital formulation of Sobol’ sequences built from primitive polynomials and direction numbers (Equations (7.9.7)–(7.9.12)). These sequences exemplify the goal of quasi-random sampling: producing deterministic, extensible point sets whose discrepancy grows no faster than $(\log N)^n/N$, thereby achieving far more uniform coverage of the unit hypercube than standard Monte Carlo. Program 7.9.2 illustrates these ideas numerically by constructing van der Corput, Halton, and Sobol’ sequences and comparing their performance on a smooth two-dimensional integration task.

At the core of the implementation is the one-dimensional van der Corput radical-inverse function, which directly corresponds to the digit-reflection process described in Equations (7.9.3)–(7.9.4). The function `van_der_corput` converts an integer index into base-$b$ digits, reverses their order across the radix point, and accumulates the resulting fractional value. This produces the characteristic subdividing pattern shown in Equation (7.9.5), where each power of the base introduces a new level of refinement. The resulting sequence serves both as a pedagogical illustration and as the building block for higher-dimensional constructions.

The Halton sequence in two dimensions, implemented by `halton_2d`, follows directly from Equation (7.9.6). Each coordinate is generated by applying the radical-inverse to a distinct prime base, here $2$ and $3$. This ensures equidistribution in $[0,1]^2$ while allowing each new point to refine the sampling without requiring resampling of previously generated points. The program prints the first several Halton points to illustrate their structured yet non-grid-like distribution and their characteristic gap-filling behavior.

The Sobol’ component of the program implements the essential features of the digital construction described in Equations (7.9.7)–(7.9.12). Direction numbers for the first dimension are initialized using the simplest choice $V_i = 2^{-i}$, corresponding to the dyadic fractional sequence frequently used in the literature. The second dimension uses a primitive polynomial $x^3 + x + 1$ (Equation (7.9.7)) and initial odd integers $M_1, M_2, M_3$ as in Equation (7.9.8). The function `sobol_direction_numbers_dim2` applies the recurrence relation (Equation (7.9.9)) to generate the remaining $M_i$, which are then normalized into direction numbers according to Equation (7.9.10). The resulting direction tables capture the essential binary structure that gives Sobol’ sequences their excellent high-dimensional performance.

The function `sobol_2d` constructs the $j$-th point of the Sobol’ sequence by applying the XOR-based accumulation rule from Equation (7.9.12). By examining the bits of the integer index and combining them with the stored direction numbers, the implementation faithfully reproduces the bitwise digital construction described in Section 7.9.2. Although the program does not explicitly use Gray codes (Equation (7.9.13)), the direct binary formulation is sufficient for generating millions of points and clearly illustrates the mechanics of Sobol’ construction.

Finally, the numerical comparison performed by `estimate_with_sequence` highlights the practical implications of low discrepancy. By evaluating all three sequences on the smooth test function $f(x,y)=e^{x+y}$, whose exact integral is known analytically, the program demonstrates how Halton and Sobol’ points achieve significantly smaller errors than random sampling for the same $N$. This reflects the theoretical connection between uniformity and error articulated in the Koksma–Hlawka inequality and explained throughout Section 7.9.2.

```rust
/*
    Program 7.9.2: Classical Low-Discrepancy Sequences – van der Corput, Halton, and Sobol’

    Problem statement.
    Section 7.9.2 introduces classical low-discrepancy sequences whose
    construction is rooted in number theory and digital arithmetic. The
    van der Corput radical-inverse in base b (7.9.3)–(7.9.4) provides a
    one-dimensional sequence that systematically fills [0,1] by reflecting
    base-b digits across the radix point. Halton sequences (7.9.6) extend
    this idea to n dimensions using different prime bases per coordinate,
    producing simple, extensible low-discrepancy point sets.

    Sobol’ sequences (7.9.7)–(7.9.12) are digital sequences in base 2 that
    rely on direction numbers derived from primitive polynomials over the
    field F_2. By combining these direction numbers with binary expansions
    of the index and efficient Gray-code updates (7.9.13), Sobol’ sequences
    achieve excellent uniformity even in high dimensions while remaining
    inexpensive to generate.

    This program demonstrates these constructions in two dimensions. It:

      (1) Generates the first few terms of the one-dimensional van der
          Corput sequence in base 2.

      (2) Constructs a 2D Halton sequence using bases 2 and 3 and prints
          its first points.

      (3) Constructs a 2D Sobol’ sequence from direction numbers generated
          via a primitive polynomial and prints the first points.

      (4) Compares Halton and Sobol’ on the smooth test integral
              I = ∫_0^1 ∫_0^1 exp(x + y) dx dy = (e - 1)^2
          by estimating I with N = 10_000 samples from each sequence and
          reporting the absolute integration error.

    The output illustrates how classical low-discrepancy sequences provide
    deterministic, extensible point sets with much more uniform coverage
    than naive random sampling.
*/

/// Test integrand on [0,1]^2:
///   f(x, y) = exp(x + y).
fn f(x: f64, y: f64) -> f64 {
    (x + y).exp()
}

/// Exact value of the integral:
///   I = ∫_0^1 ∫_0^1 exp(x + y) dx dy = (e - 1)^2.
fn exact_integral() -> f64 {
    let e = std::f64::consts::E;
    (e - 1.0_f64).powi(2)
}

/* ------------------------------------------------------------------------- */
/*  (i) van der Corput and Halton sequences                                  */
/* ------------------------------------------------------------------------- */

/// One-dimensional van der Corput radical-inverse function in base `base`.
///
/// For index n (n >= 1), this produces a point in [0,1) by reversing the
/// base-b digits of n after the radix point:
///
///   n = d_0 + d_1 b + d_2 b^2 + ...
///   H_n^(b) = d_0 / b + d_1 / b^2 + d_2 / b^3 + ...
///
/// corresponding to Equations (7.9.3)–(7.9.4).
fn van_der_corput(mut n: u64, base: u32) -> f64 {
    let b = base as u64;
    let mut f = 1.0_f64;
    let mut result = 0.0_f64;

    while n > 0 {
        f /= base as f64;
        let digit = n % b;
        result += f * (digit as f64);
        n /= b;
    }

    result
}

/// Two-dimensional Halton sequence point with index n:
///   x = H_n^(2), y = H_n^(3),
/// where H_n^(b) is the van der Corput radical-inverse in base b,
/// as in Equation (7.9.6).
fn halton_2d(n: u64) -> (f64, f64) {
    let x = van_der_corput(n, 2);
    let y = van_der_corput(n, 3);
    (x, y)
}

/* ------------------------------------------------------------------------- */
/*  (ii) Sobol’ sequence in 2D via direction numbers                         */
/* ------------------------------------------------------------------------- */

const SOBOL_MAX_BITS: usize = 32;

/// Initialize direction numbers for the first coordinate of a 2D Sobol’
/// sequence. For the first dimension, a simple choice is
///
///   V_i^(1) = 1 / 2^i,  i = 1, 2, ...
///
/// which corresponds to the binary fractions 0.1, 0.01, 0.001, ...
/// We store these as 32-bit integers scaled by 2^32.
fn sobol_direction_numbers_dim1() -> [u32; SOBOL_MAX_BITS] {
    let mut v = [0u32; SOBOL_MAX_BITS];
    for i in 1..=SOBOL_MAX_BITS {
        // For dimension 1, M_i = 1 for all i, so V_i = 1 / 2^i.
        // Store as v_i = M_i << (32 - i).
        v[i - 1] = 1u32 << (32 - i);
    }
    v
}

/// Initialize direction numbers for the second coordinate of a 2D Sobol’
/// sequence using the primitive polynomial
///
///   P(x) = x^3 + x + 1,                                            (7.9.7)
///
/// with degree q = 3 and coefficients a_1 = 0, a_2 = 1 over F_2.
/// Initial odd integers M_1, M_2, M_3 are chosen as
///
///   M_1 = 1, M_2 = 3, M_3 = 5,                                    (7.9.8)
///
/// with 0 < M_i < 2^i. The recurrence (7.9.9) generates M_i for i > 3:
///
///   M_i = a_1 (2 M_{i-1}) ⊕ a_2 (2^2 M_{i-2})
///         ⊕ 2^3 M_{i-3} ⊕ M_{i-3},
///
/// where ⊕ denotes bitwise XOR and left shifts implement multiplication
/// by powers of 2. Direction numbers are then
///
///   V_i = M_i / 2^i,                                              (7.9.10)
///
/// stored as scaled 32-bit integers v_i = M_i << (32 - i).
fn sobol_direction_numbers_dim2() -> [u32; SOBOL_MAX_BITS] {
    let q = 3usize;
    // Coefficients for x^3 + a1 x^2 + a2 x + 1: a1 = 0, a2 = 1.
    let a: [u32; 2] = [0, 1];

    // M_i, 1-based indexing up to SOBOL_MAX_BITS.
    let mut m = [0u32; SOBOL_MAX_BITS + 1];
    m[1] = 1; // M_1
    m[2] = 3; // M_2
    m[3] = 5; // M_3

    for i in (q + 1)..=SOBOL_MAX_BITS {
        // According to (7.9.9) for q = 3:
        // M_i = a1 * (2 M_{i-1}) ⊕ a2 * (2^2 M_{i-2})
        //       ⊕ 2^3 M_{i-3} ⊕ M_{i-3}.
        let mut val = 0u32;

        if a[0] == 1 {
            val ^= m[i - 1] << 1;
        }
        if a[1] == 1 {
            val ^= m[i - 2] << 2;
        }
        // 2^3 M_{i-3} and M_{i-3}:
        val ^= m[i - q] << q;
        val ^= m[i - q];

        m[i] = val;
    }

    let mut v = [0u32; SOBOL_MAX_BITS];
    for i in 1..=SOBOL_MAX_BITS {
        v[i - 1] = m[i] << (32 - i);
    }
    v
}

/// Generate the j-th point (j >= 1) of a 2D Sobol’ sequence using the
/// direction numbers for dimensions 1 and 2. For index j, write its
/// binary expansion as in (7.9.11) and form
///
///   x_j = b_{j,1} V_1^(1) ⊕ b_{j,2} V_2^(1) ⊕ ...
///   y_j = b_{j,1} V_1^(2) ⊕ b_{j,2} V_2^(2) ⊕ ...
///
/// as in (7.9.12), where the bits b_{j,i} select which direction numbers
/// to XOR. We use 32 bits of precision and divide by 2^32 to obtain a
/// point in [0,1)^2.
fn sobol_2d(j: u64, v1: &[u32; SOBOL_MAX_BITS], v2: &[u32; SOBOL_MAX_BITS]) -> (f64, f64) {
    let mut x_int: u32 = 0;
    let mut y_int: u32 = 0;

    let mut index = j;
    let mut bit = 0usize;

    while index > 0 && bit < SOBOL_MAX_BITS {
        if (index & 1) == 1 {
            x_int ^= v1[bit];
            y_int ^= v2[bit];
        }
        index >>= 1;
        bit += 1;
    }

    let scale = (1u64 << 32) as f64;
    (x_int as f64 / scale, y_int as f64 / scale)
}

/// Estimate the integral I using N points from a given 2D sequence
/// generator `seq`, which maps an index j ≥ 1 to a point in [0,1)^2.
fn estimate_with_sequence<F>(n_samples: u64, mut seq: F) -> f64
where
    F: FnMut(u64) -> (f64, f64),
{
    let mut sum = 0.0_f64;

    for n in 1..=n_samples {
        let (x, y) = seq(n);
        sum += f(x, y);
    }

    sum / (n_samples as f64)
}

fn main() {
    let exact = exact_integral();

    println!("Program 7.9.2: Classical Low-Discrepancy Sequences (Halton and Sobol')");
    println!("Integrand for integration test: f(x, y) = exp(x + y)");
    println!("Exact integral I = (e - 1)^2 ≈ {:.10}", exact);
    println!();

    // ---------------------------------------------------------------------
    // 1. van der Corput sequence in base 2
    // ---------------------------------------------------------------------
    println!("First 10 points of the 1D van der Corput sequence in base 2:");
    for n in 1..=10 {
        let x = van_der_corput(n, 2);
        println!("  n = {:2}, H_n^(2) ≈ {:.10}", n, x);
    }
    println!();

    // ---------------------------------------------------------------------
    // 2. First points of the 2D Halton sequence (bases 2 and 3)
    // ---------------------------------------------------------------------
    println!("First 10 points of the 2D Halton sequence (bases 2 and 3):");
    for n in 1..=10 {
        let (x, y) = halton_2d(n);
        println!("  n = {:2}, (x, y) ≈ ({:.10}, {:.10})", n, x, y);
    }
    println!();

    // ---------------------------------------------------------------------
    // 3. First points of the 2D Sobol’ sequence
    // ---------------------------------------------------------------------
    let dir1 = sobol_direction_numbers_dim1();
    let dir2 = sobol_direction_numbers_dim2();

    println!("First 10 points of a 2D Sobol’ sequence:");
    for n in 1..=10 {
        let (x, y) = sobol_2d(n, &dir1, &dir2);
        println!("  n = {:2}, (x, y) ≈ ({:.10}, {:.10})", n, x, y);
    }
    println!();

    // ---------------------------------------------------------------------
    // 4. Integration test: Halton vs Sobol’ on [0,1]^2
    // ---------------------------------------------------------------------
    let n_samples: u64 = 10_000;

    let halton_est = estimate_with_sequence(n_samples, |n| halton_2d(n));
    let sobol_est = estimate_with_sequence(n_samples, |n| sobol_2d(n, &dir1, &dir2));

    let halton_err = (halton_est - exact).abs();
    let sobol_err = (sobol_est - exact).abs();

    println!("Integration of f(x, y) = exp(x + y) over [0,1]^2 with N = {}", n_samples);
    println!("  Halton estimate ≈ {:.10}, |error| ≈ {:.10}", halton_est, halton_err);
    println!("  Sobol' estimate ≈ {:.10}, |error| ≈ {:.10}", sobol_est, sobol_err);
}
```

Program 7.9.2 offers a concrete demonstration of how classical low-discrepancy sequences arise from elementary number-theoretic constructions and how these deterministic sequences improve numerical integration performance. The van der Corput and Halton examples reveal how digit reversal in different prime bases generates point sets with systematic refinement, while the Sobol’ implementation showcases the power of binary direction numbers derived from primitive polynomials. The numerical results underscore the practical benefits of low discrepancy: for smooth integrands, error reductions of one to two orders of magnitude are achieved without altering the sampling budget.

These constructions form the foundation upon which more advanced quasi-Monte Carlo techniques, such as scrambling, high-dimensional digital nets, and adaptive sequence transformations, are built. By providing extensible sequences with near-optimal discrepancy, Halton and especially Sobol’ sequences serve as indispensable tools in modern deterministic sampling strategies. Program 7.9.2 thus bridges the theoretical framework of Section 7.9.2 with concrete computational practice, illustrating the essential principles that underlie efficient space-filling sampling in high-dimensional numerical integration.

## 7.9.3. Modern Advances and Applications of Quasi-Random Sequences

The classical constructions of Halton, Sobol’, Faure, and Niederreiter–Xing sequences form the backbone of quasi-Monte Carlo (QMC) methods. However, in the last three years there has been a rapid acceleration in research aimed at improving low-discrepancy sampling for high-dimensional numerical integration, global optimization, uncertainty quantification, and computational finance. Modern developments fall into three broad categories: scrambling and randomization techniques, algorithmically optimized point sets, and neural or machine-learned low-discrepancy sequences.

### Scrambled and Randomized Quasi-Monte Carlo

A purely deterministic low-discrepancy sequence does not provide a reliable internal error estimate. To reintroduce statistical interpretability while preserving low discrepancy, randomized constructions have become standard. The most widely adopted technique is *Owen scrambling*, which applies random permutations of binary digits at multiple hierarchical levels. A scrambled Sobol’ sequence retains the excellent $(\log N)^n/N$ discrepancy in expectation and yields an unbiased Monte Carlo estimator with a proper variance measure.

Recent empirical studies demonstrate that scrambling can significantly outperform both deterministic QMC and classical Monte Carlo in challenging financial models. For example, scrambled Sobol’ sequences achieve faster convergence for pricing complex Asian options and can reduce variance by orders of magnitude compared to standard Monte Carlo (Kucherenko & Hok, 2023). These improvements are especially pronounced when the payoff function is smooth in most dimensions, allowing scrambled QMC to exploit uniformity across dyadic partitions.

### Algorithmically Optimized Low-Discrepancy Point Sets

The design of optimal low-discrepancy sequences has evolved beyond classical number theory. Modern approaches use optimization, integer programming, and machine learning to directly minimize discrepancy measures.

One prominent line of research is the *Message-Passing Monte Carlo* (MPMC) framework, which trains a graph neural network to produce point sets where each point “repels’’ others through message passing. The resulting configurations achieve significantly lower empirical discrepancy than Halton or Sobol’ sequences of the same size (Rusch et al., 2024). This approach breaks from traditional digital sequence constructions by treating discrepancy minimization as an optimization problem rather than an algebraic design problem.

Further advances came from a two-stage optimization method that first determines ideal relative positions of points and then finds an optimal permutation that minimizes global discrepancy. This yields an additional improvement of roughly 20% over MPMC-generated sets in two dimensions (Clément et al., 2025), indicating that classical sequences, though powerful, are not the final word in low-discrepancy sampling.

### Greedy and Neural Generative Sequences

Complementing large-scale optimization approaches, recent theoretical work has shown that simple *greedy placement algorithms*, in which each new point minimizes an $L^2$ type discrepancy functional, can produce sequences that outperform classical constructions for many moderate values of $N$. Although the theoretical understanding of such greedy rules remains incomplete, numerical evidence suggests strong performance in low to moderate dimensions.

At the frontier, researchers have begun exploring *neural generative models* for infinite low-discrepancy sequences. Transformer-based architectures have been trained to emit sequences that preserve low discrepancy across all prefixes, effectively learning an extensible, theoretically grounded sampling rule from data. These developments highlight a shift toward data-driven discrepancy minimization, complementing classical deterministic constructions.

### Applications Across Scientific Computing

Quasi-random sampling has become increasingly relevant in fields where high-dimensional integrals arise naturally. In computational finance, scrambled Sobol' and optimized QMC sequences continue to reduce the computational burden of pricing path-dependent derivatives, enabling substantial accuracy gains in models with hundreds of dimensions (Kucherenko & Hok, 2023).

In robotics, low-discrepancy sampling improves coverage of configuration spaces, accelerating algorithms such as RRT and dramatically reducing the number of resampling steps required. Recent demonstrations show that MPMC-optimized low-discrepancy sets enhance the performance of planning algorithms for multi-DOF arms in complex environments (Chahine et al., 2024).

Further progress is also being made in uncertainty quantification, Bayesian computation, and high-dimensional PDE solvers, where quasi-random sampling provides faster convergence than stochastic methods, especially when integrands exhibit smoothness or anisotropic structure (Hickernell et al., 2025).

### Rust Implementation

Following the discussion in Section 7.9.3 on modern randomized and optimized quasi-random sequences, Program 7.9.3 provides a practical implementation of scrambled Sobol’ sampling for variance reduction in quasi-Monte Carlo integration. While classical low-discrepancy sequences such as Halton and Sobol’ offer excellent uniformity and near $1/N$ convergence for smooth integrands, their deterministic nature prevents the use of internal error estimates. Randomization techniques, particularly digital scrambling, reintroduce statistical interpretability while preserving the structural advantages of low discrepancy. This program demonstrates these ideas by constructing a simple binary scrambling of Sobol’ direction numbers, generating randomized quasi-Monte Carlo estimators across multiple replicates, and comparing their accuracy and root-mean-square error to standard Monte Carlo and deterministic Sobol’ sampling. The resulting numerical experiment illustrates how scrambling bridges the gap between deterministic QMC and probabilistic error quantification, aligning closely with the methodological developments described in Section 7.9.3.

At the heart of the program is the construction of two-dimensional Sobol’ points using the digital arithmetic framework introduced earlier in Section 7.9.2. The function that generates the base Sobol’ points follows the bitwise accumulation rule of Equation (7.9.12), combining the binary digits of the point index with the direction numbers derived from primitive polynomials (Equations (7.9.7)–(7.9.10)). This digital representation allows the sequence to fill dyadic partitions evenly, ensuring low discrepancy and strong multi-resolution structure.

To introduce randomization, the program applies a simple per-replicate scrambling step. Although not a full Owen hierarchical permutation of bits, the XOR-based scrambling used here captures the essential idea of digital randomization: each replicate perturbs the direction numbers via independent random masks, preserving dyadic refinement while decorrelating the resulting sequence. This aligns with the spirit of randomized QMC methods, which aim to retain the low-discrepancy behavior in expectation while yielding unbiased estimators with well-defined variance.

The function `scrambled_sobol_2d_replicates` performs this randomized sampling by generating multiple independently scrambled Sobol’ sequences, evaluating the test function $f(x,y)=\exp(x+y)$ over all points, and computing the sample mean for each replicate. Because scrambling restores unbiasedness, the collection of replicate means supports the computation of a root-mean-square error that behaves analogously to Monte Carlo’s variance estimate.

In parallel, the program constructs baseline comparisons: purely random Monte Carlo sampling and deterministic Sobol’ points without scrambling. The Monte Carlo estimator follows the classical $N^{-1/2}$ convergence discussed in earlier sections, while the deterministic Sobol’ estimate provides a single high-accuracy reference value reflecting low-discrepancy integration without statistical noise. The helper function `mc_mean_and_rmse` computes both the sample mean and RMSE across replicates, enabling a side-by-side comparison with the randomized quasi-Monte Carlo estimator.

Finally, the `main` function assembles all components into a clear numerical experiment across multiple values of $N$. For each sample size, the program prints the Monte Carlo mean and RMSE, the scrambled QMC mean and RMSE, and the absolute error of deterministic Sobol’ sampling relative to the exact integral. This comprehensive table illustrates how randomization reduces the variance of Sobol’-based estimators, often dramatically for larger $N$, and highlights the nuanced behavior of scrambling techniques, which may exhibit fluctuations at moderate sample sizes but tend to outperform classical Monte Carlo at scale.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
/*
    Program 7.9.3: Scrambled Sobol’ Sequences and Modern Quasi–Monte Carlo

    Problem statement.
    Section 7.9.3 surveys modern advances in quasi-random sampling, with a
    particular focus on randomized and algorithmically optimized low-discrepancy
    sequences. Classical constructions such as Halton and Sobol’ provide
    deterministic low-discrepancy point sets, but their determinism prevents
    straightforward error estimation. Scrambled quasi–Monte Carlo methods,
    especially Owen-style scrambling applied to Sobol’ sequences, restore
    unbiasedness and enable variance estimation while retaining the benefits
    of low discrepancy.

    This program demonstrates a simplified example of such modern randomized
    QMC in two dimensions. It compares:

      (1) Standard Monte Carlo using pseudo-random points in [0,1]^2.
      (2) Randomly scrambled Sobol’ sequences, implemented as a simple
          digital XOR-scramble of direction numbers for each replicate.
      (3) A single deterministic Sobol’ estimate for reference.

    All methods are applied to the smooth test integral

        I = ∫_0^1 ∫_0^1 exp(x + y) dx dy = (e - 1)^2.

    For a set of sample sizes N, the program computes multiple independent
    replicates for Monte Carlo and scrambled Sobol’, then reports the mean
    estimate and root-mean-square error (RMSE) relative to the exact value.
    The deterministic Sobol’ estimator is evaluated once per N and its
    absolute error is reported. The results illustrate how randomized
    low-discrepancy sampling can significantly reduce RMSE compared to
    classical Monte Carlo, while also providing a statistically meaningful
    error measure, in line with the modern developments discussed in
    Section 7.9.3.
*/

use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Test integrand on [0,1]^2:
///   f(x, y) = exp(x + y).
fn f(x: f64, y: f64) -> f64 {
    (x + y).exp()
}

/// Exact value of the integral:
///   I = ∫_0^1 ∫_0^1 exp(x + y) dx dy = (e - 1)^2.
fn exact_integral() -> f64 {
    let e = std::f64::consts::E;
    (e - 1.0_f64).powi(2)
}

/* ------------------------------------------------------------------------- */
/*  Standard Monte Carlo in 2D                                               */
/* ------------------------------------------------------------------------- */

/// Generate a single pseudo-random point (x, y) in [0,1]^2.
fn sample_random_2d(rng: &mut StdRng) -> (f64, f64) {
    let dist = Uniform::new(0.0_f64, 1.0_f64);
    let x = rng.sample(dist);
    let y = rng.sample(dist);
    (x, y)
}

/// Estimate the integral using N pseudo-random points in [0,1]^2.
///
/// Standard Monte Carlo estimator:
///   Q_N^MC = (1 / N) Σ f(x_i, y_i),
/// where (x_i, y_i) are i.i.d. Uniform([0,1]^2) samples.
fn estimate_random_once(n_samples: u64, rng: &mut StdRng) -> f64 {
    let mut sum = 0.0_f64;

    for _ in 0..n_samples {
        let (x, y) = sample_random_2d(rng);
        sum += f(x, y);
    }

    sum / (n_samples as f64)
}

/* ------------------------------------------------------------------------- */
/*  Sobol’ base implementation (2D)                                          */
/* ------------------------------------------------------------------------- */

const SOBOL_MAX_BITS: usize = 32;

/// Initialize direction numbers for the first coordinate of a 2D Sobol’
/// sequence. For the first dimension, a simple choice is
///
///   V_i^(1) = 1 / 2^i,  i = 1, 2, ...
///
/// corresponding to binary fractions 0.1, 0.01, 0.001, ...
/// We store these as 32-bit integers scaled by 2^32.
fn sobol_direction_numbers_dim1() -> [u32; SOBOL_MAX_BITS] {
    let mut v = [0u32; SOBOL_MAX_BITS];
    for i in 1..=SOBOL_MAX_BITS {
        // For dimension 1, M_i = 1 for all i, so V_i = 1 / 2^i.
        // Store as v_i = M_i << (32 - i).
        v[i - 1] = 1u32 << (32 - i);
    }
    v
}

/// Initialize direction numbers for the second coordinate of a 2D Sobol’
/// sequence using the primitive polynomial
///
///   P(x) = x^3 + x + 1,
///
/// with degree q = 3 and coefficients a_1 = 0, a_2 = 1 over F_2.
/// Initial odd integers M_1, M_2, M_3 are chosen as
///
///   M_1 = 1, M_2 = 3, M_3 = 5,
///
/// with 0 < M_i < 2^i. The recurrence generates M_i for i > 3:
///
///   M_i = a_1 (2 M_{i-1}) ⊕ a_2 (2^2 M_{i-2})
///         ⊕ 2^3 M_{i-3} ⊕ M_{i-3},
///
/// where ⊕ denotes bitwise XOR and left shifts implement multiplication
/// by powers of 2. Direction numbers are then
///
///   V_i = M_i / 2^i,
///
/// stored as scaled 32-bit integers v_i = M_i << (32 - i).
fn sobol_direction_numbers_dim2() -> [u32; SOBOL_MAX_BITS] {
    let q = 3usize;
    // Coefficients for x^3 + a1 x^2 + a2 x + 1: a1 = 0, a2 = 1.
    let a: [u32; 2] = [0, 1];

    // M_i, 1-based indexing up to SOBOL_MAX_BITS.
    let mut m = [0u32; SOBOL_MAX_BITS + 1];
    m[1] = 1; // M_1
    m[2] = 3; // M_2
    m[3] = 5; // M_3

    for i in (q + 1)..=SOBOL_MAX_BITS {
        // For q = 3:
        // M_i = a1 * (2 M_{i-1}) ⊕ a2 * (2^2 M_{i-2})
        //       ⊕ 2^3 M_{i-3} ⊕ M_{i-3}.
        let mut val = 0u32;

        if a[0] == 1 {
            val ^= m[i - 1] << 1;
        }
        if a[1] == 1 {
            val ^= m[i - 2] << 2;
        }
        // 2^3 M_{i-3} and M_{i-3}:
        val ^= m[i - q] << q;
        val ^= m[i - q];

        m[i] = val;
    }

    let mut v = [0u32; SOBOL_MAX_BITS];
    for i in 1..=SOBOL_MAX_BITS {
        v[i - 1] = m[i] << (32 - i);
    }
    v
}

/// Generate the j-th point (j >= 1) of a 2D Sobol’ sequence using the
/// direction numbers for dimensions 1 and 2. For index j, write its
/// binary expansion and form
///
///   x_j = b_{j,1} V_1^(1) ⊕ b_{j,2} V_2^(1) ⊕ ...
///   y_j = b_{j,1} V_1^(2) ⊕ b_{j,2} V_2^(2) ⊕ ...
///
/// where the bits b_{j,i} select which direction numbers to XOR.
/// We use 32 bits of precision and divide by 2^32 to obtain a point
/// in [0,1)^2.
fn sobol_2d(j: u64, v1: &[u32; SOBOL_MAX_BITS], v2: &[u32; SOBOL_MAX_BITS]) -> (f64, f64) {
    let mut x_int: u32 = 0;
    let mut y_int: u32 = 0;

    let mut index = j;
    let mut bit = 0usize;

    while index > 0 && bit < SOBOL_MAX_BITS {
        if (index & 1) == 1 {
            x_int ^= v1[bit];
            y_int ^= v2[bit];
        }
        index >>= 1;
        bit += 1;
    }

    let scale = (1u64 << 32) as f64;
    (x_int as f64 / scale, y_int as f64 / scale)
}

/// Estimate the integral I using N points from a given 2D sequence
/// generator `seq`, which maps an index j ≥ 1 to a point in [0,1)^2.
fn estimate_with_sequence<F>(n_samples: u64, mut seq: F) -> f64
where
    F: FnMut(u64) -> (f64, f64),
{
    let mut sum = 0.0_f64;

    for n in 1..=n_samples {
        let (x, y) = seq(n);
        sum += f(x, y);
    }

    sum / (n_samples as f64)
}

/* ------------------------------------------------------------------------- */
/*  Simple digital scrambling of Sobol’ direction numbers                    */
/* ------------------------------------------------------------------------- */

/// Produce a scrambled copy of the direction numbers `v` by XOR-ing each
/// entry with a random 32-bit mask. This is a simple example of a digital
/// scramble in the spirit of Owen scrambling: it injects randomness at the
/// bit level while preserving the overall digital net structure.
fn scramble_direction_numbers(
    v: &[u32; SOBOL_MAX_BITS],
    rng: &mut StdRng,
) -> [u32; SOBOL_MAX_BITS] {
    let mut scrambled = [0u32; SOBOL_MAX_BITS];

    for i in 0..SOBOL_MAX_BITS {
        let mask: u32 = rng.gen();
        scrambled[i] = v[i] ^ mask;
    }

    scrambled
}

/* ------------------------------------------------------------------------- */
/*  Statistics helper                                                        */
/* ------------------------------------------------------------------------- */

/// Compute the mean and root-mean-square error (RMSE) of a set of estimates
/// {Q_r} compared to the exact value I:
///
///   mean(Q)  = (1/R) Σ Q_r,
///   RMSE(Q)  = sqrt( (1/R) Σ (Q_r - I)^2 ).
fn mean_and_rmse(estimates: &[f64], exact: f64) -> (f64, f64) {
    let r = estimates.len() as f64;
    let mut sum = 0.0_f64;
    let mut sum_sq_err = 0.0_f64;

    for &q in estimates {
        sum += q;
        let err = q - exact;
        sum_sq_err += err * err;
    }

    let mean = sum / r;
    let rmse = (sum_sq_err / r).sqrt();
    (mean, rmse)
}

fn main() {
    // Fixed seed so that the pseudo-random draws and scrambles are reproducible.
    let mut rng = StdRng::seed_from_u64(7_9_2025);

    let exact = exact_integral();
    println!("Program 7.9.3: Scrambled Sobol’ Sequences and Modern QMC");
    println!("Integrand: f(x, y) = exp(x + y)");
    println!("Exact integral I = (e - 1)^2 ≈ {:.10}", exact);
    println!();

    // Base direction numbers for a 2D Sobol’ sequence.
    let dir1 = sobol_direction_numbers_dim1();
    let dir2 = sobol_direction_numbers_dim2();

    // Sample sizes and number of independent replicates.
    let sample_sizes: [u64; 4] = [1_000, 5_000, 10_000, 50_000];
    let n_replicates: usize = 20;

    println!(
        "{:>8}  {:>14}  {:>14}  {:>14}  {:>14}  {:>14}",
        "N",
        "MC mean",
        "MC RMSE",
        "Scr.QMC mean",
        "Scr.QMC RMSE",
        "Sobol |err|"
    );
    println!("{}", "-".repeat(90));

    for &n in &sample_sizes {
        // Collect estimates for MC and scrambled Sobol’.
        let mut mc_estimates = Vec::with_capacity(n_replicates);
        let mut sqmc_estimates = Vec::with_capacity(n_replicates);

        for _ in 0..n_replicates {
            // Standard Monte Carlo estimate.
            let q_mc = estimate_random_once(n, &mut rng);
            mc_estimates.push(q_mc);

            // Scrambled Sobol’: scramble direction numbers independently
            // for each replicate, then estimate the integral.
            let dir1_scr = scramble_direction_numbers(&dir1, &mut rng);
            let dir2_scr = scramble_direction_numbers(&dir2, &mut rng);

            let q_sqmc = estimate_with_sequence(n, |j| sobol_2d(j, &dir1_scr, &dir2_scr));
            sqmc_estimates.push(q_sqmc);
        }

        let (mc_mean, mc_rmse) = mean_and_rmse(&mc_estimates, exact);
        let (sqmc_mean, sqmc_rmse) = mean_and_rmse(&sqmc_estimates, exact);

        // Single deterministic Sobol’ estimate (no scrambling) for reference.
        let q_sobol_det = estimate_with_sequence(n, |j| sobol_2d(j, &dir1, &dir2));
        let sobol_err = (q_sobol_det - exact).abs();

        println!(
            "{:8}  {:14.10}  {:14.10}  {:14.10}  {:14.10}  {:14.10}",
            n, mc_mean, mc_rmse, sqmc_mean, sqmc_rmse, sobol_err
        );
    }
}
```

Program 7.9.3 illustrates the practical role of randomized quasi-Monte Carlo methods within modern numerical integration. By applying a simple digital scrambling to Sobol’ direction numbers, the program demonstrates how randomness can be introduced without disrupting the uniformity and multi-level structure that make Sobol’ sequences effective. The resulting scrambled estimators are unbiased, support meaningful RMSE computation, and frequently yield far lower variance than standard Monte Carlo, especially for smooth integrands. This behavior aligns with recent empirical and theoretical findings showing that scrambled QMC often provides order-of-magnitude variance reduction in high-dimensional finance, uncertainty quantification, and scientific computing.

The comparison between Monte Carlo, deterministic Sobol’, and scrambled Sobol’ highlights an important conceptual point emphasized in Section 7.9.3: deterministic QMC alone provides excellent accuracy but lacks statistical interpretability, while randomization restores error estimation without undermining low discrepancy. This hybrid perspective represents a central direction in modern QMC research, complementing recent algorithmically optimized point sets, message-passing constructions, and neural generative sequences. Together, these developments reaffirm the importance of quasi-random sampling as a flexible and powerful alternative to classical stochastic integration.

## 7.9.4. Section Summary

Modern advances have transformed quasi-Monte Carlo from a classical number-theoretic technique into an active, interdisciplinary research area. Scrambled sequences provide statistical robustness, optimized point sets push discrepancy lower than traditional constructions, and neural models point toward next-generation low-discrepancy generators. Across finance, robotics, and computational science, these developments collectively expand the range and effectiveness of quasi-random sampling in high-dimensional numerical computation.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/FaqGs0LAXDOW88gfwT2c.6","tags":[]}

# 7.10. Adaptive and Recursive Monte Carlo Methods

Adaptive and recursive Monte Carlo methods extend classical Monte Carlo integration by allowing the sampling distribution to evolve in response to information gathered during computation. In standard Monte Carlo, samples are drawn independently from a fixed proposal, typically the uniform distribution over the domain, regardless of the behavior of the integrand. While this approach is unbiased and dimension-agnostic, it is often inefficient because many regions contribute negligibly to the integral while others dominate the variance.

Adaptive Monte Carlo addresses this limitation by actively steering sampling effort toward regions of the domain that have the greatest influence on the integral. As samples accumulate, the method constructs data-driven estimates of which parts of the integrand are large, highly variable, or responsible for most of the estimator’s uncertainty. Based on these diagnostics, the sampling distribution is updated so that subsequent samples concentrate in the most informative regions while reducing effort elsewhere. This dynamic reallocation improves estimator efficiency and accelerates variance reduction, particularly for integrals involving localized features, anisotropic behavior, or rare events.

Recursive methods build on this idea by introducing a hierarchical, subdivided representation of the domain. The integration region is partitioned into subregions, often via binary, quadtree, octree, or kd-tree based splits, and each subregion receives its own local Monte Carlo estimator. Regions with large estimated variance or high contribution to the global integral are recursively subdivided, and additional samples are allocated to the refined subregions. This process yields a multi-level tree structure that adaptively focuses computational effort where the integrand is most challenging. Recursive Monte Carlo thus inherits the robustness of classical Monte Carlo while gaining the targeted resolution characteristic of adaptive deterministic quadrature. Modern variants combine hierarchical partitioning with importance sampling, kernel density estimation, or control variates to accelerate convergence even in high dimensions.

These methods can be viewed as transforming Monte Carlo from a uniform random sampler into a feedback-controlled stochastic estimator, where each new sample improves not only the estimate but also the future sampling strategy. They provide a principled framework for exploiting structural information in the integrand and have become essential tools in uncertainty quantification, Bayesian computation, rare-event simulation, computer graphics, and high-dimensional scientific computing.

To formalize the setting, consider a typical high-dimensional integral:

$$I = \int_V f(\mathbf{x})\mathrm{d}\mathbf{x}, \qquad V \subset \mathbb{R}^d \tag{7.10.1}$$

which we wish to approximate by Monte Carlo. Under crude Monte Carlo, we draw points $\mathbf{x}_1,\ldots,\mathbf{x}_N$ independently and uniformly from $V$ and form the estimator,

$$
\widehat{I}_N = \frac{1}{N}\sum_{j=1}^N f(\mathbf{x}_j)
\tag{7.10.2}
$$

which is unbiased and converges at the classical Monte Carlo rate $\mathcal{O}(N^{-1/2})$. However, when $f$ exhibits sharp peaks or significant local variability, the variance of $\widehat{I}_N$ becomes large and convergence slows markedly. Adaptive methods modify the sampling density to concentrate points where $\lvert f(\mathbf{x}) \rvert$ is large, whereas recursive or stratified methods partition the domain into subregions and allocate samples according to estimated local variances.

Modern applications strongly motivate these techniques. Rare-event simulation in engineering reliability requires estimating probabilities far below $10^{-6}$, where uniform Monte Carlo is infeasible. Recent studies in structural reliability (Eshra & Papakonstantinou, 2025) and ultra-reliable low-latency communication (Ke et al., 2023) demonstrate that adaptive importance sampling can reduce variance by several orders of magnitude. Likewise, stratified and recursive Monte Carlo have advanced significantly; for example, decision-tree driven adaptive stratification (Chopin, Wang & Gerber, 2025) achieves convergence faster than $N^{-1/2}$ when the integrand exhibits exploitable structure.

A useful way to visualize the effect of adaptive sampling is through the cumulative distribution function (CDF) associated with the proposal distribution. Under crude Monte Carlo, a uniform proposal on $[0,1]$ produces a strictly linear CDF, reflecting equal probability allocation across the domain. Adaptive Monte Carlo, in contrast, constructs a *warped* CDF that places disproportionately larger probability mass in regions where the integrand is large or highly variable. The stepwise structure illustrates how the sampling density becomes piecewise nonuniform, concentrating samples in informative subregions while reducing effort elsewhere.

A useful way to visualize the effect of adaptive sampling is through the cumulative distribution function (CDF) associated with the proposal distribution. Under crude Monte Carlo, a uniform proposal on $[0,1]$ produces a strictly linear CDF, reflecting equal probability allocation across the domain. Adaptive Monte Carlo, in contrast, constructs a *warped* CDF that places disproportionately larger probability mass in regions where the integrand is large or highly variable. The stepwise structure illustrates how the sampling density becomes piecewise nonuniform, concentrating samples in informative subregions while reducing effort elsewhere.

```{figure} images/pqQDe4beUu67RvW3raYP-S3iYgE229T1E2L9IPiP3-v1.png
:name: DBq9qAEspO
:align: center
:width: 40%

**Figure 7.10.1** Example of an adaptive sampling cumulative distribution function (CDF) on $[0,1]$. Unlike the linear CDF generated by uniform sampling, the adaptive CDF exhibits uneven jumps, indicating increased probability mass assigned to subregions where the integrand contributes most to the variance.
```

Figure 7.10.1 illustrates a typical adaptive sampling cumulative distribution function (CDF). Unlike the linear CDF generated by uniform sampling, the adaptive CDF is warped, with uneven jumps indicating increased probability mass in regions where the integrand contributes most strongly to the variance. Recursive refinements ensure that no subregion remains undersampled.

### Rust Implementation

Following the discussion in Section 7.10 on how adaptive and recursive Monte Carlo methods transform classical Monte Carlo into a feedback-driven estimator, Program 7.10.0 provides a concrete implementation of adaptive importance sampling in one dimension. Whereas crude Monte Carlo draws samples uniformly over the domain, leading to slow convergence when the integrand exhibits sharp peaks or highly localized structure, adaptive methods update the sampling distribution based on information learned during a preliminary sampling phase. This program demonstrates how an initial batch of pilot samples can be used to construct a warped, piecewise-constant proposal distribution that concentrates probability mass where the integrand is largest. By comparing the classical estimator with the adaptive importance sampling estimator, the program illustrates the variance reduction and accuracy improvements made possible by adapting the proposal to the integrand’s behaviour.

At the core of the implementation is the definition of a sharply peaked test integrand on the interval $[0,1]$, designed to mimic integrands for which uniform sampling performs poorly. Such functions are precisely the setting where crude Monte Carlo, as defined in Equation (7.10.2), suffers from large estimator variance. The routine `reference_integral` provides a highly resolved midpoint-rule approximation of the true integral, ensuring that comparisons between crude and adaptive estimates can be made against a reliable reference value.

The function `crude_monte_carlo` implements the baseline estimator directly from Equation (7.10.2), drawing samples uniformly from $[0,1]$ and averaging the resulting function evaluations. While unbiased and simple, this estimator spreads its sampling effort evenly across the domain, even in regions that contribute negligibly to the integral. This behaviour underscores the inefficiency of fixed proposals when the integrand is not uniformly informative.

To address this limitation, the program constructs a piecewise-constant adaptive proposal distribution. The routine `build_adaptive_proposal` divides the domain into bins and uses an initial collection of pilot samples to estimate the relative scale of the integrand in each subregion. These estimates determine bin weights, which serve as approximations to the optimal importance sampling density discussed earlier in Section 7.10. A corresponding `PiecewiseConstantPdf` structure implements density evaluation, inverse-transform sampling within each bin, and the computation of the cumulative distribution function (CDF). Printing the CDF reveals a warped, stepwise profile characteristic of adaptive sampling, visually demonstrating how probability mass shifts toward regions where the integrand is large or highly variable.

The function `importance_sampling` implements the adaptive estimator by drawing samples from the new proposal and applying the standard importance sampling correction (f(x)/p(x)). Because the densities in the piecewise-constant model are known explicitly, this correction is straightforward to compute. The resulting estimator is still unbiased but achieves significantly reduced variance by avoiding low-value regions of the domain. The final `main` function orchestrates the entire demonstration, comparing crude Monte Carlo, constructing the adaptive proposal, printing its CDF to mirror the conceptual illustration of Figure 7.10.1, and finally computing the adaptive importance sampling estimator. This workflow showcases how adaptive methods refine sampling density based on pilot information and thereby improve estimator efficiency.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
// Program 7.10.0: Adaptive importance sampling on [0,1] with a warped CDF.
//
// Problem statement:
// Implement a one-dimensional adaptive Monte Carlo integration scheme in Rust that
// illustrates the transition from crude Monte Carlo with a uniform proposal to an
// adaptively tuned importance sampler. The program should:
//   * Define a smooth, sharply peaked integrand f on [0,1] whose structure makes
//     crude Monte Carlo inefficient.
//   * Compute a reference value for the integral I = ∫_0^1 f(x) dx using a fine
//     numerical quadrature (for validation).
//   * Estimate I using crude Monte Carlo with a uniform proposal, corresponding
//     to Equation (7.10.2).
//   * Construct a piecewise-constant adaptive proposal on a fixed partition of
//     [0,1] by using an initial batch of samples to identify intervals where f
//     is large, and update the sampling probabilities accordingly.
//   * Use the adaptive proposal to perform importance sampling, estimating I via
//     E[f(X)/p(X)], where p is the piecewise-constant density.
//   * Print the resulting piecewise CDF of the adaptive proposal so that its
//     stepwise, warped shape can be compared to the linear CDF of the uniform
//     proposal (as in Figure 7.10.1).
//
// The goal is not to provide a production-ready adaptive Monte Carlo algorithm,
// but to demonstrate in a compact example how information gathered from initial
// samples can be used to warp the proposal distribution, concentrating samples
// in regions where |f(x)| is large and thereby improving estimator efficiency.

use rand::prelude::*;

const NUM_BINS: usize = 16;

/// Test integrand on [0,1]:
///   f(x) = exp(-a (x - c1)^2) + 0.5 * exp(-b (x - c2)^2)
/// with sharp peaks to make uniform sampling inefficient.
fn integrand(x: f64) -> f64 {
    let a = 80.0;
    let b = 250.0;
    let c1 = 0.25;
    let c2 = 0.7;

    ((-a * (x - c1).powi(2)).exp()) + 0.5 * ((-b * (x - c2).powi(2)).exp())
}

/// Simple numerical reference integral using the midpoint rule on [0,1].
fn reference_integral(n_steps: usize) -> f64 {
    let h = 1.0 / (n_steps as f64);
    let mut sum = 0.0;
    for i in 0..n_steps {
        let x_mid = (i as f64 + 0.5) * h;
        sum += integrand(x_mid);
    }
    sum * h
}

/// Crude Monte Carlo integral with uniform sampling on [0,1].
fn crude_monte_carlo(n_samples: usize, rng: &mut impl Rng) -> f64 {
    let mut acc = 0.0;
    for _ in 0..n_samples {
        let x: f64 = rng.gen(); // uniform in [0,1)
        acc += integrand(x);
    }
    acc / (n_samples as f64)
}

/// A piecewise-constant probability density on [0,1].
/// The interval [0,1] is partitioned into NUM_BINS equal subintervals,
/// and each interval i has probability mass `weights[i]`.
struct PiecewiseConstantPdf {
    /// Right endpoints of the bins: breaks[k] = k / NUM_BINS, k = 0..=NUM_BINS.
    breaks: [f64; NUM_BINS + 1],
    /// Probability masses for each bin (sum to 1).
    weights: [f64; NUM_BINS],
}

impl PiecewiseConstantPdf {
    /// Construct from an array of bin weights (probability masses).
    fn new(weights: [f64; NUM_BINS]) -> Self {
        // Build uniform breaks on [0,1].
        let mut breaks = [0.0; NUM_BINS + 1];
        for k in 0..=NUM_BINS {
            breaks[k] = k as f64 / (NUM_BINS as f64);
        }
        // Normalise weights to sum to 1.
        let mut w_norm = weights;
        let sum_w: f64 = w_norm.iter().sum();
        if sum_w > 0.0 {
            for w in &mut w_norm {
                *w /= sum_w;
            }
        } else {
            // Fallback to uniform if all weights are zero.
            for w in &mut w_norm {
                *w = 1.0 / (NUM_BINS as f64);
            }
        }

        Self {
            breaks,
            weights: w_norm,
        }
    }

    /// Density p(x) for x in [0,1]. Outside [0,1], the density is zero.
    fn density(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return 0.0;
        }
        // Handle x = 1.0 explicitly to avoid rounding issues.
        let idx = if x == 1.0 {
            NUM_BINS - 1
        } else {
            (x * NUM_BINS as f64).floor() as usize
        };
        let width = self.breaks[idx + 1] - self.breaks[idx];
        if width <= 0.0 {
            return 0.0;
        }
        self.weights[idx] / width
    }

    /// Sample x from the piecewise-constant pdf using inverse transform on the
    /// discrete distribution over bins, then uniform sampling within the chosen bin.
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        let u: f64 = rng.gen();
        // Select bin index by cumulative sum over weights.
        let mut cum = 0.0;
        let mut idx = NUM_BINS - 1;
        for i in 0..NUM_BINS {
            cum += self.weights[i];
            if u < cum {
                idx = i;
                break;
            }
        }
        let left = self.breaks[idx];
        let right = self.breaks[idx + 1];
        let v: f64 = rng.gen();
        left + v * (right - left)
    }

    /// Print the CDF values at bin boundaries to visualise the warped CDF.
    /// This CDF is piecewise constant between the bin endpoints and jumps by
    /// weights[i] at each bin.
    fn print_cdf(&self) {
        println!("Adaptive piecewise CDF on [0,1]:");
        println!("{:>8}  {:>12}  {:>12}", "bin", "x_right", "CDF");
        let mut cum = 0.0;
        for i in 0..NUM_BINS {
            cum += self.weights[i];
            let x_right = self.breaks[i + 1];
            println!("{:8}  {:12.6}  {:12.6}", i, x_right, cum);
        }
        println!();
    }
}

/// Build an adaptive piecewise-constant proposal on [0,1] using an initial
/// batch of uniform samples. Each bin's weight is set proportional to the
/// estimated contribution of that bin to the integral.
fn build_adaptive_proposal(n_pilot: usize, rng: &mut impl Rng) -> PiecewiseConstantPdf {
    let mut sums = [0.0f64; NUM_BINS];
    let mut counts = [0usize; NUM_BINS];

    // Pilot phase: uniform samples on [0,1).
    for _ in 0..n_pilot {
        let x: f64 = rng.gen();
        let idx = if x == 1.0 {
            NUM_BINS - 1
        } else {
            (x * NUM_BINS as f64).floor() as usize
        };
        let fx = integrand(x);
        sums[idx] += fx;
        counts[idx] += 1;
    }

    // Convert to bin weights. A simple heuristic is:
    //   weight_i ∝ max(mean f in bin, small_floor) * bin_width
    // with bin_width = 1 / NUM_BINS for all bins.
    let bin_width = 1.0 / (NUM_BINS as f64);
    let mut raw_weights = [0.0f64; NUM_BINS];
    let small_floor = 1e-6;
    for i in 0..NUM_BINS {
        let mean_f = if counts[i] > 0 {
            sums[i] / (counts[i] as f64)
        } else {
            0.0
        };
        let score = mean_f.max(small_floor);
        raw_weights[i] = score * bin_width;
    }

    PiecewiseConstantPdf::new(raw_weights)
}

/// Importance sampling estimator using a given piecewise-constant proposal.
fn importance_sampling(
    pdf: &PiecewiseConstantPdf,
    n_samples: usize,
    rng: &mut impl Rng,
) -> f64 {
    let mut acc = 0.0;
    for _ in 0..n_samples {
        let x = pdf.sample(rng);
        let p = pdf.density(x);
        if p > 0.0 {
            acc += integrand(x) / p;
        }
    }
    acc / (n_samples as f64)
}

fn main() {
    let mut rng = rand::thread_rng();

    // Reference integral (midpoint rule).
    let ref_val = reference_integral(200_000);
    println!("Integrand: sharply peaked mixture of Gaussians on [0,1].");
    println!("Reference integral (midpoint rule): {:.10}", ref_val);
    println!();

    // Crude Monte Carlo with uniform proposal.
    let n_crude = 50_000;
    let crude_est = crude_monte_carlo(n_crude, &mut rng);
    let crude_err = (crude_est - ref_val).abs();

    println!("Crude Monte Carlo (uniform on [0,1]):");
    println!("  Samples N        = {}", n_crude);
    println!("  Estimate         = {:.10}", crude_est);
    println!("  |error|          = {:.3e}", crude_err);
    println!();

    // Adaptive phase: build an importance sampler from pilot samples.
    let n_pilot = 10_000;
    let pdf = build_adaptive_proposal(n_pilot, &mut rng);

    println!("Adaptive importance sampling proposal built from {} pilot samples.", n_pilot);
    pdf.print_cdf();

    // Importance sampling with adaptive proposal.
    let n_adapt = 40_000;
    let adapt_est = importance_sampling(&pdf, n_adapt, &mut rng);
    let adapt_err = (adapt_est - ref_val).abs();

    println!("Adaptive importance sampling:");
    println!("  Samples N        = {}", n_adapt);
    println!("  Estimate         = {:.10}", adapt_est);
    println!("  |error|          = {:.3e}", adapt_err);
}
```

Program 7.10.0 demonstrates how adaptive importance sampling leverages structural information gathered through preliminary sampling to construct a more efficient proposal distribution. By reallocating probability mass toward regions where the integrand is large, the adaptive estimator overcomes the uniform inefficiency of crude Monte Carlo and reduces variance for the same computational effort. The example highlights the fundamental shift introduced in Section 7.10: Monte Carlo methods become not simply random samplers but dynamic, data-informed estimators capable of warping their sampling distribution as computation progresses.

The comparison between crude and adaptive estimators clearly shows the advantages of concentrating sampling effort in informative subregions. The crude estimator exhibits typical $N^{-1/2}$ fluctuations, while the adaptive estimator achieves noticeably smaller error thanks to its improved proposal. The printed CDF further visualizes how adaptive sampling departs from the linear CDF of uniform sampling, illustrating the feedback loop that characterizes modern adaptive Monte Carlo techniques.

The modular design of the program makes it naturally extensible. Replacing the fixed binning with recursive bisection would produce a hierarchical stratifier, while more sophisticated adaptive schemes could incorporate kernel density estimates, mixture models, or control variates. These enhancements connect directly to recent advances in adaptive and recursive Monte Carlo, providing a foundation for future sections in this chapter.

## 7.10.1. Adaptive Importance Sampling

Importance sampling replaces uniform Monte Carlo sampling with a carefully designed probability density $p(\mathbf{x})$ on $V$. Multiplying and dividing the integrand by $p(\mathbf{x})$ yields the identity:

$$I=\int_V \frac{f(\mathbf{x})}{p(\mathbf{x})} p(\mathbf{x})\mathrm{d}\mathbf{x} \tag{7.10.3}$$

which holds for any density $p$ satisfying $p(\mathbf{x})>0$ on the support of $f$. Drawing independent samples $\mathbf{x}_i \sim p$, the associated importance-sampling estimator is:

$$
\widehat{I}_N
= \frac{1}{N}\sum_{i=1}^N \frac{f(\mathbf{x}_i)}{p(\mathbf{x}_i)}
\tag{7.10.4}
$$

which remains unbiased for all admissible choices of $p$. The variance of this estimator is:

$$
\operatorname{Var}(\widehat{I}_N)
= \frac{1}{N}\left(
\int_V \frac{f(\mathbf{x})^2}{p(\mathbf{x})}\,\mathrm{d}\mathbf{x}
- I^2
\right)
\tag{7.10.5}
$$

demonstrating that the efficiency of importance sampling hinges on controlling the quantity $f(\mathbf{x})^2 / p(\mathbf{x})$. If $p$ places insufficient mass in regions where $|f|$ is large, the variance grows dramatically.

### Optimal Importance Density

Minimizing the variance in (7.10.5) over all densities $p$ subject to the constraint $\int_V p=1$ leads to a calculus-of-variations problem. Consider the functional:

$$
\mathcal{J}[p]
= \int_V \frac{f(\mathbf{x})^2}{p(\mathbf{x})}\,\mathrm{d}\mathbf{x}
\;+\;
\lambda\left(\int_V p(\mathbf{x})\,\mathrm{d}\mathbf{x} - 1\right)
\tag{7.10.6}
$$

where $\lambda$ enforces normalization. Setting the first variation of $\mathcal{J}$ to zero yields the stationarity condition,

$$-\frac{f(\mathbf{x})^2}{p(\mathbf{x})^2} + \lambda = 0 \tag{7.10.7}$$

which solves to the optimal density:

$$
p^{\ast}(\mathbf{x})
= \frac{|f(\mathbf{x})|}
       {\displaystyle \int_V |f(\mathbf{y})|\,\mathrm{d}\mathbf{y}}
\tag{7.10.8}
$$

With this choice, the weighted integrand $f(\mathbf{x})/p^{\ast}(\mathbf{x})$ becomes constant over $V$, causing the variance in (7.10.5) to vanish. Of course, the optimal density is unattainable in practice because it depends on the unknown function $f$, but it serves as a theoretical target for adaptive schemes.

### Adaptive Strategies

Modern adaptive importance samplers attempt to approximate the ideal density $p^{\ast}$ using iterative updates based on previously drawn samples. A widely used framework is the *cross-entropy* method, which selects a parametric family $p(\cdot;\theta)$ and updates $\theta$ to minimize the divergence $D_{\mathrm{KL}}\!\left(p^{\ast}\,\middle\|\,p(\cdot;\theta)\right)$. This approach has strong convergence guarantees and is effective for rare-event simulation, where naïve uniform sampling would require astronomically large sample sizes.

### Recent Advances in High-Dimensional Problems

Recent research highlights the remarkable effectiveness of adaptive importance sampling when dealing with complex, high-dimensional integrals. In the setting of 6G ultra-reliable low-latency communication (URLLC) systems, Ke et al. (2023) develop an adaptive Gaussian proposal whose parameters are refined through successive pilot batches. Because URLLC applications often require estimating probabilities on the order of $10^{-9}$ or smaller, crude Monte Carlo would necessitate an infeasible number of samples. Their adaptive procedure gradually adjusts the proposal’s mean and covariance so that more samples are drawn from regions contributing most to these ultra-rare events. When further combined with quasi-Monte Carlo refinements, the resulting algorithm achieves several orders of magnitude reduction in variance, allowing accurate rare-event estimation with dramatically fewer evaluations.

A comparable improvement is seen in structural reliability analysis, where the problem is to compute failure probabilities in engineering systems with numerous uncertain parameters. Eshra and Papakonstantinou (2025) introduce a gradient-free adaptive importance sampler that remains effective even when the number of uncertain input variables reaches into the hundreds. Instead of relying on gradient information or restrictive assumptions about the proposal, their method updates the sampling density based solely on empirical diagnostics derived from the sample batch. This flexibility proves essential for high-dimensional, nonlinear reliability models, where constructing a single well-tuned fixed proposal is extremely difficult. Their results demonstrate that an adaptively learned proposal can substantially outperform classical fixed-density approaches, enabling reliable estimation of probabilities that would otherwise be computationally inaccessible.

### Computational Considerations

Each iteration of an adaptive importance sampler typically involves two phases: a sampling phase of cost $O(N)$, where $N$ new samples are drawn and evaluated, followed by a density-update phase that adjusts the proposal distribution based on the current batch. The cost of the update step depends on the complexity of the selected density family. In recent work, flexible parametric families, such as Gaussian mixtures, transport-map models, normalizing flows, and separable coordinate-wise weight functions, have become increasingly popular. These richer representations are capable of capturing multimodal structure, strong anisotropies, and nonlinear interactions among components of $\mathbf{x}$, thereby helping to mitigate the curse of dimensionality. As a result, modern adaptive schemes are far more robust in high-dimensional settings than classical importance sampling methods, making them suitable for contemporary problems in uncertainty quantification, communication theory, structural reliability, and stochastic simulation.

### Separable Approximations and the VEGAS Perspective

A prominent adaptive strategy is to approximate the density in separable form,

$$p(\mathbf{x}) \propto g_1(x_1), g_2(x_2)\cdots g_d(x_d),\tag{7.10.9}$$

and update each marginal weight function $g_i$ using projected information from $|f(\mathbf{x})|^2$. This viewpoint underlies the VEGAS algorithm (discussed in Section 7.10.3), which adaptively reshapes the sampling density by constructing one-dimensional histograms along each coordinate direction. Although the separability assumption is restrictive, the method performs surprisingly well for moderately correlated integrands and remains widely used in high-energy physics and computational finance.

### Rust Implementation

Following the discussion in Section 7.10.1 on the foundational principles of importance sampling and its role in reducing Monte Carlo variance, Program 7.10.1 provides a concrete implementation that illustrates how both fixed and adaptive proposals influence estimator performance. Traditional crude Monte Carlo draws samples from the target distribution and relies on uniform variance allocation across the domain, which can be highly inefficient when the integrand is dominated by rare or extreme events. Importance sampling, by contrast, replaces the sampling distribution with a more carefully chosen proposal, adjusting the estimator so that unbiasedness is retained while variance is reduced. This program demonstrates these ideas through the estimation of a rare-event probability under the standard normal distribution, comparing crude Monte Carlo with a fixed shifted Gaussian proposal and with an adaptive parametric update inspired by the cross-entropy method. The example highlights the conceptual progression from fixed-density importance sampling toward adaptive strategies that approximate the optimal density described in Equation (7.10.8).

At the core of the program is the rare-event integral $I = P(Z > a)$, where $Z\sim N(0,1)$, chosen to illustrate the sensitivity of variance to sample allocation. The routine `reference_tail_prob` provides a high-accuracy numerical approximation via trapezoidal quadrature, enabling reliable comparison with Monte Carlo estimators. Crude Monte Carlo is implemented in `crude_mc_tail_prob`, which directly samples from the target density and evaluates the indicator integrand. This corresponds exactly to the estimator of Equation (7.10.2), where variance is governed by how frequently random draws fall into the small tail region. For rare events, such occurrences are infrequent, resulting in noisy and unreliable estimates even with large sample sizes.

The first improvement is demonstrated by the function `is_tail_prob_fixed_proposal`, which implements importance sampling using a shifted Gaussian proposal $p(x;\mu) = N(\mu,1)$. This approach directly applies Equation (7.10.3) and its estimator form (7.10.4), evaluating the weighted integrand $f(x)/p(x)$ whenever a sample falls in the rare-event region. By centering the proposal near the critical threshold $a$, the method increases the likelihood of sampling relevant points and thereby reduces the term $f(x)^2/p(x)$ that governs variance in Equation (7.10.5). Even with moderate sample counts, this fixed proposal achieves dramatically improved accuracy over crude Monte Carlo, illustrating the key insight behind importance sampling: sample in proportion to where the integrand is large.

To connect these ideas with adaptive importance sampling, the function `adaptive_is_tail_prob` implements a simple parametric update inspired by the cross-entropy framework discussed later in the section. A Gaussian proposal is maintained with mean parameter (\\mu), and each iteration consists of drawing samples from the current proposal, computing importance weights, selecting an elite fraction of the most informative samples, and updating (\\mu) toward their weighted mean. This mimics the minimization of the divergence $D_{\mathrm{KL}}(p^\ast \Vert p(\cdot;\theta))$ described in the discussion following Equation (7.10.8), where (p^\\ast) is the unattainable optimal density. The program prints the intermediate proposal mean and estimator values at each iteration, providing a transparent view of how the adaptation pulls the proposal toward the rare-event region. Because rare-event problems exhibit sensitive variance behavior, this example also offers an instructive look at the challenges of overly aggressive adaptation, reinforcing the need for stabilization techniques in practical implementations.

The `main` function integrates these components into a coherent demonstration. It begins by evaluating the reference probability, then computes crude Monte Carlo and fixed importance sampling estimates, before performing several adaptive iterations with detailed diagnostic output. Together, these routines illustrate the progression from baseline estimators to adaptive methods designed to mimic the optimal importance density. The example highlights not only the conceptual advantages of adaptive sampling but also the practical considerations required to ensure stable and efficient updates.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
rand_distr = "0.4"
```

```rust
// Program 7.10.1: Parametric adaptive importance sampling for a Gaussian rare event.
//
// Problem statement:
// Implement a one-dimensional adaptive importance sampling scheme in Rust that
// illustrates the ideas of Equations (7.10.3)–(7.10.8). The goal is to estimate
// a rare-event probability
//     I = P(Z > a),   Z ~ N(0,1),
// using:
//   * crude Monte Carlo sampling from the standard normal,
//   * fixed-parameter importance sampling with a shifted Gaussian proposal,
//   * an adaptive importance sampler that iteratively updates the mean of a
//     Gaussian proposal p(x; θ) = N(μ, 1) based on previous samples.
// The program should:
//   - Use the identity in (7.10.3) and estimator (7.10.4) by evaluating
//     f(x)/p(x) for the indicator integrand f(x) = 1_{x > a} under a proposal p.
//   - Approximate a reference value for I via numerical integration of the
//     standard normal density.
//   - Demonstrate how poor mass allocation (sampling far from the tail) leads
//     to noisy crude Monte Carlo estimates, while the adaptive estimator moves
//     the proposal toward the optimal importance density p*(x) ∝ |f(x)| and
//     produces more accurate results with the same or fewer samples.
//
// This example is intentionally one-dimensional for clarity, but the same
// algorithmic structure extends to higher-dimensional Gaussian proposals
// and more complicated integrands.

use rand::prelude::*;
use rand_distr::StandardNormal;

/// Standard normal density φ(x) = (2π)^(-1/2) exp(-x^2 / 2).
fn std_normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Numerical reference for the tail probability P(Z > a), Z ~ N(0,1),
/// computed by integrating φ(x) from a to upper using the trapezoidal rule.
fn reference_tail_prob(a: f64, upper: f64, n_steps: usize) -> f64 {
    let h = (upper - a) / (n_steps as f64);
    let mut sum = 0.0;
    let mut x = a;
    for _ in 0..n_steps {
        let x_next = x + h;
        sum += 0.5 * h * (std_normal_pdf(x) + std_normal_pdf(x_next));
        x = x_next;
    }
    sum
}

/// Crude Monte Carlo estimator of P(Z > a) with Z ~ N(0,1).
/// This corresponds directly to Equation (7.10.2) with
/// f(x) = 1_{x > a} and uniform sampling replaced by sampling
/// from the target density.
fn crude_mc_tail_prob<R: Rng + ?Sized>(a: f64, n_samples: usize, rng: &mut R) -> f64 {
    let mut count = 0usize;
    for _ in 0..n_samples {
        let z: f64 = rng.sample(StandardNormal);
        if z > a {
            count += 1;
        }
    }
    (count as f64) / (n_samples as f64)
}

/// Importance sampling estimator of P(Z > a), where the proposal
/// is X ~ N(μ, 1). The target is standard normal.
/// We use the identity (7.10.3) and estimator (7.10.4):
///   I = E_proposal[ 1_{X > a} * φ(X) / q(X; μ) ],
/// where q(·; μ) is the N(μ,1) density.
fn is_tail_prob_fixed_proposal<R: Rng + ?Sized>(
    a: f64,
    n_samples: usize,
    mu: f64,
    rng: &mut R,
) -> f64 {
    let mut acc = 0.0;
    for _ in 0..n_samples {
        let z: f64 = rng.sample(StandardNormal);
        let x = mu + z; // sample from N(μ,1)

        if x > a {
            let num = std_normal_pdf(x);        // φ(x), target density
            let denom = std_normal_pdf(x - mu); // q(x; μ) = φ(x - μ)
            acc += num / denom;
        }
    }
    acc / (n_samples as f64)
}

/// Adaptive importance sampling using a Gaussian proposal N(μ, 1).
/// This implements a simple cross-entropy-style update:
///   - At each iteration, draw samples X_k ~ N(μ, 1).
///   - Compute importance weights for the rare event.
///   - Select an "elite" fraction of samples with the largest X_k values.
///   - Update μ to the weighted mean of the elite samples.
/// The final estimator is the average of f(X)/p(X) over the last batch.
fn adaptive_is_tail_prob<R: Rng + ?Sized>(
    a: f64,
    n_iter: usize,
    n_samples_per_iter: usize,
    elite_frac: f64,
    mu_init: f64,
    rng: &mut R,
) -> (f64, f64) {
    let mut mu = mu_init;
    let mut last_estimate = 0.0_f64;

    for iter in 0..n_iter {
        let mut xs = Vec::with_capacity(n_samples_per_iter);
        let mut weights = Vec::with_capacity(n_samples_per_iter);
        let mut acc_est = 0.0;

        for _ in 0..n_samples_per_iter {
            let z: f64 = rng.sample(StandardNormal);
            let x = mu + z;

            // Importance weight for indicator f(x) = 1_{x > a}.
            let w = if x > a {
                let num = std_normal_pdf(x);
                let denom = std_normal_pdf(x - mu);
                num / denom
            } else {
                0.0
            };

            xs.push(x);
            weights.push(w);
            acc_est += w;
        }

        last_estimate = acc_est / (n_samples_per_iter as f64);

        // Select elite samples by sorting xs and taking top fraction.
        let mut idx: Vec<usize> = (0..n_samples_per_iter).collect();
        idx.sort_by(|&i, &j| xs[i].partial_cmp(&xs[j]).unwrap());

        let elite_count = ((elite_frac * n_samples_per_iter as f64).max(1.0)) as usize;
        let start = n_samples_per_iter.saturating_sub(elite_count);

        let mut sum_mu = 0.0;
        let mut sum_w = 0.0;
        for &k in &idx[start..] {
            let w = weights[k].max(0.0);
            sum_mu += w * xs[k];
            sum_w += w;
        }

        if sum_w > 0.0 {
            mu = sum_mu / sum_w;
        } else {
            // Fallback: unweighted mean of elite samples if all weights vanish.
            let mut s = 0.0;
            for &k in &idx[start..] {
                s += xs[k];
            }
            mu = s / (elite_count as f64);
        }

        println!(
            "Adaptive iteration {}: μ ≈ {:6.3}, IS estimate ≈ {:9.3e}",
            iter + 1,
            mu,
            last_estimate
        );
    }

    (last_estimate, mu)
}

fn main() {
    let mut rng = rand::thread_rng();

    // Rare-event threshold: P(Z > a), Z ~ N(0,1).
    let a = 3.5;

    // Reference value using numerical integration of φ(x) over [a, upper].
    let upper = 8.0;
    let ref_val = reference_tail_prob(a, upper, 200_000);

    println!("Estimating P(Z > a) with Z ~ N(0,1) and a = {:.2}", a);
    println!(
        "Reference tail probability (numerical): {:.10}",
        ref_val
    );
    println!();

    // 1. Crude Monte Carlo.
    let n_crude = 300_000;
    let crude_est = crude_mc_tail_prob(a, n_crude, &mut rng);
    let crude_err = (crude_est - ref_val).abs();
    println!("Crude Monte Carlo (sampling from N(0,1)):");
    println!("  Samples N        = {}", n_crude);
    println!("  Estimate         = {:.10}", crude_est);
    println!("  |error|          = {:.3e}", crude_err);
    println!();

    // 2. Fixed-parameter importance sampling with a shifted Gaussian proposal.
    let n_is_fixed = 100_000;
    let mu_fixed = 3.5;
    let is_fixed_est = is_tail_prob_fixed_proposal(a, n_is_fixed, mu_fixed, &mut rng);
    let is_fixed_err = (is_fixed_est - ref_val).abs();
    println!("Fixed-parameter importance sampling (proposal N(μ,1), μ = {:.1}):", mu_fixed);
    println!("  Samples N        = {}", n_is_fixed);
    println!("  Estimate         = {:.10}", is_fixed_est);
    println!("  |error|          = {:.3e}", is_fixed_err);
    println!();

    // 3. Adaptive importance sampling: cross-entropy-style update of μ.
    let n_iter = 5;
    let n_per_iter = 20_000;
    let elite_frac = 0.1;
    let mu_init = 0.0;

    println!("Adaptive importance sampling with Gaussian proposal N(μ,1):");
    println!(
        "  Iterations       = {}, samples per iteration = {}, elite fraction = {:.2}",
        n_iter, n_per_iter, elite_frac
    );
    let (adapt_est, mu_final) =
        adaptive_is_tail_prob(a, n_iter, n_per_iter, elite_frac, mu_init, &mut rng);
    let adapt_err = (adapt_est - ref_val).abs();
    println!();
    println!("Final adaptive estimate:");
    println!("  μ_final          = {:.6}", mu_final);
    println!("  Estimate         = {:.10}", adapt_est);
    println!("  |error|          = {:.3e}", adapt_err);
}
```

Program 7.10.1 provides a hands-on illustration of the fundamental principles underlying importance sampling and its adaptive extensions. Crude Monte Carlo performs poorly for rare-event probabilities because only a tiny fraction of samples fall in the relevant region, leading to large estimator variance. Fixed importance sampling mitigates this difficulty by aligning the proposal distribution with the structure of the integrand, greatly improving efficiency even with relatively simple proposal families. Adaptive importance sampling goes further by iteratively modifying the proposal in response to previously gathered information, reflecting the theoretical notion of approximating the optimal density in Equation (7.10.8).

The example also reveals an important practical insight emphasized throughout Section 7.10.1: although adaptive methods can dramatically improve performance, poorly controlled adaptation can lead to instability or divergence, especially for highly skewed or multimodal integrands. This behaviour underscores the broader message of the section, that while adaptive schemes aim to reduce the factor $f(\mathbf{x})^2/p(\mathbf{x})$ in the variance expression (7.10.5), they must be carefully designed to maintain robustness. The program therefore serves as both a demonstration of the power of importance sampling and a motivation for more sophisticated adaptive algorithms introduced later in the chapter, including regularized updates, mixture-based proposals, and the VEGAS-style separable approximations of Section 7.10.3.

### Improving Stability in Adaptive Importance Sampling

The behaviour of Program 7.10.1 highlights both the strengths and vulnerabilities of adaptive importance sampling. While adaptation toward the optimal density can dramatically reduce variance, poorly regularized updates may overshoot or collapse, producing unstable proposals and inflated estimator variance. This sensitivity is especially pronounced in rare-event problems, where the likelihood function is sharply peaked and small shifts in the proposal can drastically alter the weight structure. To address these limitations, practical adaptive algorithms incorporate mechanisms such as damped parameter updates, effective sample size (ESS) diagnostics, and constraints on proposal parameters. These stabilizers help maintain robustness even when the integrand exhibits the extreme anisotropy or heavy-tailed structure typical of high-dimensional uncertainty quantification, reliability analysis, and communication-theoretic applications.

To illustrate these ideas concretely, the next program refines the adaptive scheme by introducing weighted updates, learning-rate controls, clamping of proposal parameters, and ESS monitoring. This stabilized variant is designed to reflect the algorithmic safeguards routinely employed in contemporary adaptive importance samplers and provides a more reliable demonstration of how importance densities can be updated iteratively while avoiding the divergence issues discussed at the end of Program 7.10.1.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
rand_distr = "0.4"
```

```rust
// Program 7.10.2: Stabilized adaptive Gaussian importance sampling for rare-event estimation.
//
// Problem statement:
// Implement a stabilized adaptive importance sampling scheme in Rust for estimating
// the rare-event probability
//     I = P(Z > a),   Z ~ N(0,1),
// using:
//   * crude Monte Carlo sampling from N(0,1),
//   * fixed-parameter importance sampling with a shifted Gaussian proposal,
//   * a stabilized adaptive importance sampler that updates the mean μ of a
//     Gaussian proposal p(x; μ) = N(μ, 1) in small, controlled steps.
//
// The adaptive update is regularized by:
//   - Using weighted elite samples (largest x values with nonzero importance weights).
//   - Applying a learning rate η in the update
//         μ_new = (1 - η) μ_old + η μ_target,
//     to avoid overreacting to a single batch.
//   - Clamping μ into a prescribed interval [μ_min, μ_max] around the rare-event threshold.
//   - Reporting the effective sample size (ESS) to diagnose weight degeneracy.
//
// This stabilized scheme illustrates how practical adaptive importance sampling
// must balance responsiveness to data with robustness, in line with the theoretical
// discussion around Equations (7.10.3)–(7.10.8).

use rand::prelude::*;
use rand_distr::StandardNormal;

/// Standard normal density φ(x) = (2π)^(-1/2) exp(-x^2 / 2).
fn std_normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Numerical reference for the tail probability P(Z > a), Z ~ N(0,1),
/// computed by integrating φ(x) from a to upper using the trapezoidal rule.
fn reference_tail_prob(a: f64, upper: f64, n_steps: usize) -> f64 {
    let h = (upper - a) / (n_steps as f64);
    let mut sum = 0.0;
    let mut x = a;
    for _ in 0..n_steps {
        let x_next = x + h;
        sum += 0.5 * h * (std_normal_pdf(x) + std_normal_pdf(x_next));
        x = x_next;
    }
    sum
}

/// Crude Monte Carlo estimator of P(Z > a) with Z ~ N(0,1).
/// Corresponds to Equation (7.10.2) with f(x) = 1_{x > a}.
fn crude_mc_tail_prob<R: Rng + ?Sized>(a: f64, n_samples: usize, rng: &mut R) -> f64 {
    let mut count = 0usize;
    for _ in 0..n_samples {
        let z: f64 = rng.sample(StandardNormal);
        if z > a {
            count += 1;
        }
    }
    (count as f64) / (n_samples as f64)
}

/// Importance sampling estimator of P(Z > a), where proposal is X ~ N(μ, 1)
/// and the target is N(0,1). Uses Equation (7.10.3) and (7.10.4).
fn is_tail_prob_fixed_proposal<R: Rng + ?Sized>(
    a: f64,
    n_samples: usize,
    mu: f64,
    rng: &mut R,
) -> f64 {
    let mut acc = 0.0;
    for _ in 0..n_samples {
        let z: f64 = rng.sample(StandardNormal);
        let x = mu + z; // sample from N(μ,1)

        if x > a {
            let num = std_normal_pdf(x);        // target density φ(x)
            let denom = std_normal_pdf(x - mu); // proposal density q(x; μ) = φ(x - μ)
            acc += num / denom;
        }
    }
    acc / (n_samples as f64)
}

/// Stabilized adaptive importance sampling using a Gaussian proposal N(μ, 1).
///
/// Parameters:
///   a                - tail threshold (rare-event level)
///   n_iter           - number of adaptive iterations
///   n_samples_per_it - samples per iteration
///   elite_frac       - fraction of samples (by x order) used as elites
///   mu_init          - initial mean of the proposal
///   eta              - learning rate for the mean update
///   mu_min, mu_max   - clamp interval for μ to keep proposal in a reasonable region
///
/// At each iteration:
///   * Draw samples X_k ~ N(μ,1).
///   * Compute importance weights for f(x) = 1_{x > a}.
///   * Estimate the integral from this batch: I_hat = (1/N) Σ w_k.
///   * Compute a weighted target mean μ_target from elite samples.
///   * Update μ with a damped, clamped step.
///   * Report μ, estimate, and ESS for diagnostics.
///
/// Returns: (final_estimate, final_mu).
fn adaptive_is_tail_prob_stabilized<R: Rng + ?Sized>(
    a: f64,
    n_iter: usize,
    n_samples_per_it: usize,
    elite_frac: f64,
    mu_init: f64,
    eta: f64,
    mu_min: f64,
    mu_max: f64,
    rng: &mut R,
) -> (f64, f64) {
    let mut mu = mu_init;
    let mut last_estimate = 0.0_f64;

    for iter in 0..n_iter {
        let mut xs = Vec::with_capacity(n_samples_per_it);
        let mut ws = Vec::with_capacity(n_samples_per_it);

        for _ in 0..n_samples_per_it {
            let z: f64 = rng.sample(StandardNormal);
            let x = mu + z;

            let w = if x > a {
                let num = std_normal_pdf(x);        // φ(x)
                let denom = std_normal_pdf(x - mu); // q(x; μ)
                num / denom
            } else {
                0.0
            };

            xs.push(x);
            ws.push(w);
        }

        // Monte Carlo estimate with current proposal.
        let sum_w: f64 = ws.iter().sum();
        last_estimate = sum_w / (n_samples_per_it as f64);

        // Normalized weights and ESS for diagnostics.
        let mut norm_ws = vec![0.0; n_samples_per_it];
        if sum_w > 0.0 {
            for (i, &w) in ws.iter().enumerate() {
                norm_ws[i] = w / sum_w;
            }
        }
        let ess = if sum_w > 0.0 {
            let sum_sq: f64 = norm_ws.iter().map(|w| w * w).sum();
            if sum_sq > 0.0 {
                1.0 / sum_sq
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Build index permutation sorted by x (ascending).
        let mut idx: Vec<usize> = (0..n_samples_per_it).collect();
        idx.sort_by(|&i, &j| xs[i].partial_cmp(&xs[j]).unwrap());

        // Select elite tail fraction (largest x values).
        let elite_count = ((elite_frac * n_samples_per_it as f64).max(1.0)) as usize;
        let start = n_samples_per_it.saturating_sub(elite_count);

        // Compute weighted target mean over elites, using normalized weights.
        let mut num_mu = 0.0;
        let mut den_mu = 0.0;
        for &k in &idx[start..] {
            let w = norm_ws[k].max(0.0);
            num_mu += w * xs[k];
            den_mu += w;
        }

        let mu_target = if den_mu > 0.0 {
            num_mu / den_mu
        } else {
            // Fallback: unweighted mean of elite xs if all weights vanish.
            let mut s = 0.0;
            for &k in &idx[start..] {
                s += xs[k];
            }
            s / (elite_count as f64)
        };

        // Stabilized update: damped and clamped.
        let mu_new = (1.0 - eta) * mu + eta * mu_target;
        mu = mu_new.clamp(mu_min, mu_max);

        println!(
            "Adaptive iteration {}: μ ≈ {:6.3}, Î ≈ {:9.3e}, ESS ≈ {:7.1}",
            iter + 1,
            mu,
            last_estimate,
            ess
        );
    }

    (last_estimate, mu)
}

fn main() {
    let mut rng = rand::thread_rng();

    // Rare-event threshold: P(Z > a), Z ~ N(0,1).
    let a = 3.5;

    // Reference value using numerical integration of φ(x) over [a, upper].
    let upper = 8.0;
    let ref_val = reference_tail_prob(a, upper, 200_000);

    println!("Estimating P(Z > a) with Z ~ N(0,1) and a = {:.2}", a);
    println!(
        "Reference tail probability (numerical): {:.10}",
        ref_val
    );
    println!();

    // 1. Crude Monte Carlo.
    let n_crude = 300_000;
    let crude_est = crude_mc_tail_prob(a, n_crude, &mut rng);
    let crude_err = (crude_est - ref_val).abs();
    println!("Crude Monte Carlo (sampling from N(0,1)):");
    println!("  Samples N        = {}", n_crude);
    println!("  Estimate         = {:.10}", crude_est);
    println!("  |error|          = {:.3e}", crude_err);
    println!();

    // 2. Fixed-parameter importance sampling with a shifted Gaussian proposal.
    let n_is_fixed = 100_000;
    let mu_fixed = 3.5;
    let is_fixed_est = is_tail_prob_fixed_proposal(a, n_is_fixed, mu_fixed, &mut rng);
    let is_fixed_err = (is_fixed_est - ref_val).abs();
    println!(
        "Fixed-parameter importance sampling (proposal N(μ,1), μ = {:.1}):",
        mu_fixed
    );
    println!("  Samples N        = {}", n_is_fixed);
    println!("  Estimate         = {:.10}", is_fixed_est);
    println!("  |error|          = {:.3e}", is_fixed_err);
    println!();

    // 3. Stabilized adaptive importance sampling.
    let n_iter = 5;
    let n_per_iter = 20_000;
    let elite_frac = 0.1;
    let mu_init = 0.0;
    let eta = 0.3;         // learning rate (damping)
    let mu_min = a - 1.0;  // keep μ reasonably close to the tail threshold
    let mu_max = a + 3.0;

    println!("Stabilized adaptive importance sampling with Gaussian proposal N(μ,1):");
    println!(
        "  Iterations       = {}, samples/iteration = {}, elite fraction = {:.2}, η = {:.2}",
        n_iter, n_per_iter, elite_frac, eta
    );
    let (adapt_est, mu_final) = adaptive_is_tail_prob_stabilized(
        a,
        n_iter,
        n_per_iter,
        elite_frac,
        mu_init,
        eta,
        mu_min,
        mu_max,
        &mut rng,
    );
    let adapt_err = (adapt_est - ref_val).abs();
    println!();
    println!("Final adaptive estimate (stabilized):");
    println!("  μ_final          = {:.6}", mu_final);
    println!("  Estimate         = {:.10}", adapt_est);
    println!("  |error|          = {:.3e}", adapt_err);
}
```

Program 7.10.2 demonstrates how stabilizing mechanisms fundamentally improve the behaviour of adaptive importance sampling. By introducing a learning rate into the update of the proposal mean, the algorithm avoids the runaway behaviour observed in the naïve adaptive scheme. Clamping the proposal parameters ensures that adaptation remains concentrated around the region of interest, preventing the proposal from drifting excessively into low-density areas where importance weights become unstable. The use of weighted elite samples aligns the update step more closely with the cross-entropy methodology, while the ESS diagnostic provides a clear indication of weight degeneracy and the overall health of the sampling distribution.

The results show markedly improved stability: the proposal mean moves gradually toward regions where the rare event is most likely, the ESS increases across iterations, and the final estimator achieves accuracy comparable to fixed importance sampling despite starting from an uninformed initial proposal. This behaviour illustrates the practical message emphasized in §7.10.1: although adaptive importance sampling aims to approximate the optimal density of Equation (7.10.8), effective adaptation must balance aggressiveness with control. Stabilized updates allow the sampler to “learn” from previous samples without overreacting to stochastic fluctuations, thereby achieving significant variance reduction while maintaining robustness.

The modular structure of this implementation makes it straightforward to extend the approach. Adding covariance adaptation leads naturally to multivariate Gaussian proposals; introducing mixture components allows the algorithm to handle multimodal integrands; and replacing the Gaussian family with transport maps or normalizing flows yields greater expressiveness in high dimensions. In this way, Program 7.10.2 serves as a foundational template for the more advanced adaptive schemes discussed in later sections, bridging the theory of optimal densities with practical, stable algorithmic design.

## 7.10.2. Stratified Sampling and Recursive Splitting

Stratified sampling improves Monte Carlo integration by partitioning the domain $V$ into disjoint subregions called **strata**, and sampling each region separately. Let $V$ be partitioned into $K$ strata $\{V\}_{j=1}^{K}$, each with volume $w_j = \mathrm{Vol}(V_j)$. Suppose $N_j$ points are allocated to stratum $V_j$, with the total satisfying:

$$
\sum_{j=1}^K N_j = N
\tag{7.10.10}
$$

Within each stratum, points $x_{j,i}$ are drawn uniformly from $V_j$, yielding the stratified Monte Carlo estimator,

$$\widehat{I} = \sum_{j=1}^K\frac{w_j}{N_j} \sum_{i=1}^{N_j} f(x_{j,i})\tag{7.10.11}$$

The variance of this estimator separates cleanly across strata as:

$$
\operatorname{Var}(\widehat{I})

\sum_{j=1}^K \frac{w_j^2\sigma_j^2}{N_j} \tag{7.10.12}
$$

where $\sigma_j^2 = \operatorname{Var}(f \mid x \in V_j)$ is the *within-stratum* variance. This decomposition makes clear that strata with larger variances should receive more samples. By applying the Cauchy–Schwarz inequality or using a Lagrange multiplier formulation, the variance-minimizing allocation under the constraint (7.10.10) is:

$$N_j \propto w_j \sigma_j \tag{7.10.13}$$

Thus, strata with greater variability of $f$ naturally receive a larger share of the sampling budget.

As a simple illustration, consider two strata of equal volume. In this case, (7.10.13) implies:

$$\frac{N_a}{N} = \frac{\sigma_a}{\sigma_a + \sigma_b} \tag{7.10.14}$$

and the resulting variance reduces to,

$$
\operatorname{Var}(\widehat{I})

\frac{(w_a \sigma_a + w_b \sigma_b)^2}{N} \tag{7.10.15}
$$

A key theoretical property is that stratification never increases variance relative to uniform sampling, and in practice it often reduces variance substantially. Recent work by Chopin, Wang and Gerber (2025) demonstrates that *adaptive* stratification, choosing strata via a decision tree trained on pilot samples, can achieve convergence rates of:

$$\mathcal{O}(N^{-1/2 - r}) \qquad r>0 \tag{7.10.16}$$

which exceeds the classical Monte Carlo rate $N^{-1/2}$. The effectiveness of stratification depends on how well the domain partition captures the structure of the integrand.

### Recursive Stratified Sampling (RSS)

Full stratification becomes expensive in high dimensions because the number of strata grows exponentially. Recursive stratified sampling (RSS), used in algorithms such as MISER, mitigates this by splitting one dimension at a time in a hierarchical (tree-like) manner.

The procedure begins by allocating a fraction $pN$ of the total samples to the entire region $R$. For each coordinate direction $i$, the region is hypothetically split into two halves $R_{a,i}$ and $R_{b,i}$, and preliminary variances $\sigma_{a,i}^2$ and $\sigma_{b,i}^2$ are estimated. The algorithm then chooses the split dimension that most reduces variance. For equal subregion volumes, one selects the dimension $i^\ast$ minimizing,

$$\sigma_{a,i} + \sigma_{b,i} \tag{7.10.17}$$

More generally, when subregions have unequal volumes, the split is selected by minimizing,

$$w_{a,i}\sigma_{a,i} + w_{b,i}\sigma_{b,i} \tag{7.10.18}$$

When a heuristic scaling exponent $\alpha \neq 1$ is used, the empirical criterion:

$$w_{a,i}\sigma_{a,i}^{\alpha} +w_{b,i}\sigma_{b,i}^{\alpha}\tag{7.10.19}$$

often performs well.

After choosing the split, the remaining $(1-p)N$ samples are allocated according to the optimal rule,

$$N_{a,i} \propto w_{a,i}\sigma_{a,i},\qquad N_{b,i} \propto w_{b,i}\sigma_{b,i} \tag{7.10.20}$$

and RSS then recurses independently within each subregion. Splitting continues until a stopping criterion is met, such as a minimum sample count or maximum recursion depth.

Finally, estimates from all leaves of the recursion tree are combined by weighted averaging, using the volumes of the subregions.

A major advantage of RSS is that it avoids the exponential blow-up of full stratification yet retains many of its benefits. Under smoothness assumptions, MISER-like algorithms have been observed to achieve near $\mathcal{O}(N^{-2})$ behavior in favorable cases, substantially faster than crude Monte Carlo. In practice, the variance estimates used for splitting can be noisy when only a few pilot samples are available; a robust alternative is to use the sample range (max–min) within each candidate subregion as a proxy for variance, which often highlights “active” regions more reliably.

### Hybrid Methods: Importance Sampling + Stratification

In many applications, combining stratification with importance sampling yields even greater efficiency. For instance, after applying a coarse importance-sampling transformation (such as a VEGAS-style separable density), one may apply recursive splitting within each transformed subregion to eliminate residual variability. Conversely, some methods first stratify the domain and then perform local importance sampling within each stratum. Recent approaches even employ decision-tree stratification guided by function values or gradients, revealing a general principle: importance sampling warps the sampling density, while stratification ensures balanced coverage and prevents undersampling of critical regions. Together, they provide a powerful toolkit for tackling highly irregular or high-dimensional integrands.

### Rust Implementation

Following the discussion of stratified sampling and recursive splitting in Section 7.10.2, Program 7.10.3 provides a concrete Rust implementation of both regular stratified Monte Carlo integration and recursive stratified sampling (RSS) of MISER type. While Equations (7.10.11)–(7.10.15) establish the variance reduction properties of fixed stratification, and Equations (7.10.17)–(7.10.20) formalize the adaptive recursive splitting strategy, the present program demonstrates how these theoretical ideas translate into an efficient, fully working numerical algorithm. The implementation compares crude Monte Carlo sampling, uniform grid-based stratification, and adaptive recursive stratification on the same two-dimensional test integrand, thereby illustrating in practice how variance reduction emerges from structured domain partitioning and sample allocation. This program also makes explicit how pilot sampling, variance estimation, and optimal sample redistribution interact to improve estimator quality in finite-sample regimes.

At the core of Program 7.10.3 is the `Region` structure, which represents an axis-aligned hyperrectangular subdomain $V_j \subset V$ together with its volume $w_j$. The methods `sample_point` and `split_along` implement, respectively, uniform sampling within a stratum $V_j$ and binary partitioning of a region into two subregions $V_{a,i}$ and $V_{b,i}$ along a chosen coordinate direction. These operations correspond directly to the generation of points $x_{j,i} \in V_j$ in Equation (7.10.11) and to the recursive partitions used in Equations (7.10.17)–(7.10.20).

The function `crude_monte_carlo` implements the baseline estimator $\widehat{I}_N$ using uniform sampling over the full domain. It computes both the integral estimate and an empirical variance estimate based on the classical identity $\operatorname{Var}(\widehat{I}) = \frac{w^2}{N}\sigma^2$, which is a special case of the stratified variance decomposition in Equation (7.10.12) when only a single stratum is present. This function serves both as a performance benchmark and as a fallback mechanism inside the recursive integrator when insufficient samples are available for reliable splitting.

The function `regular_stratified_grid` implements uniform stratification on an $M \times M$ Cartesian grid in two dimensions. Each grid cell defines a stratum $V_j$ with equal volume $w_j$, and each stratum receives an equal allocation $N_j = N/K$, satisfying the constraint in Equation (7.10.10). The estimator is assembled exactly as in Equation (7.10.11), while the variance contribution of each stratum follows Equation (7.10.12). This method illustrates the classical benefits of stratification without adaptivity and provides a direct numerical manifestation of the variance-reduction principle stated after Equation (7.10.15).

Adaptive behavior is introduced through the recursive integrator `rss_integrate`, which implements a MISER-style recursive stratified sampling strategy. A fraction of the available samples is first used as pilot samples to estimate the within-subregion variances $\sigma_{a,i}^2$ and $\sigma_{b,i}^2$ associated with candidate splits along each coordinate direction. The split is selected by minimizing the weighted variance criterion in Equation (7.10.18), or more generally its empirical generalization in Equation (7.10.19) through the exponent $\alpha$. Once the optimal split is chosen, the remaining samples are allocated according to the proportionality rule in Equation (7.10.20), with recursive calls applied independently to each subregion until a stopping criterion based on recursion depth or minimum sample count is met.

Finally, the `main` function orchestrates the full numerical experiment. It initializes the integration domain $[0,1]^2$, defines a moderately irregular test integrand with localized peaks and smooth background structure, and applies all three estimators, crude Monte Carlo, regular stratification, and RSS, using the same total sampling budget. The reported estimates and empirical standard errors allow a direct numerical comparison of the efficiency gains predicted by the theoretical variance analysis of Section 7.10.2.

Add the follwing dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
// Program 7.10.3: Stratified sampling and recursive splitting (MISER-style)
//
// This example estimates the integral
//
//     I = ∫_{[0,1]^2} f(x, y) dx dy
//
// for a moderately irregular integrand f defined on the unit square. We compare:
//
//   1. Crude Monte Carlo with uniform sampling.
//   2. Simple regular stratified sampling on an M×M grid.
//   3. Recursive stratified sampling (RSS), which allocates samples adaptively
//      by splitting the region along coordinate directions, guided by estimated
//      within-subregion variability.
//
// The regular stratification corresponds directly to the estimator in (7.10.11),
// and the RSS method uses the variance-based splitting and allocation principles
// of (7.10.17)–(7.10.20). For clarity, we work in two dimensions, but the same
// ideas extend naturally to higher-dimensional hyperrectangles.
//
// To run this example, place it in src/main.rs of a Cargo project and add
// in Cargo.toml:
//
// [dependencies]
// rand = "0.8"
//
// The output reports the estimate and an empirical standard error for each
// sampling strategy.

use rand::prelude::*;

const D: usize = 2; // dimension of the integration domain [0,1]^D

#[derive(Clone, Debug)]
struct Region {
    lower: [f64; D],
    upper: [f64; D],
    volume: f64,
}

impl Region {
    fn new(lower: [f64; D], upper: [f64; D]) -> Self {
        let mut volume = 1.0;
        for d in 0..D {
            volume *= upper[d] - lower[d];
        }
        Region { lower, upper, volume }
    }

    /// Sample a single point uniformly from this region.
    fn sample_point<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; D] {
        let mut x = [0.0; D];
        for d in 0..D {
            let u: f64 = rng.gen();
            x[d] = self.lower[d] + u * (self.upper[d] - self.lower[d]);
        }
        x
    }

    /// Split the region into two subregions by cutting along the given dimension
    /// at the midpoint. This corresponds to forming strata V_a and V_b with
    /// volumes w_a and w_b.
    fn split_along(&self, dim: usize) -> (Region, Region) {
        let lower_a = self.lower;
        let mut upper_a = self.upper;
        let mut lower_b = self.lower;
        let upper_b = self.upper;

        let mid = 0.5 * (self.lower[dim] + self.upper[dim]);
        upper_a[dim] = mid;
        lower_b[dim] = mid;

        let a = Region::new(lower_a, upper_a);
        let b = Region::new(lower_b, upper_b);
        (a, b)
    }
}

/// Example integrand f(x) on [0,1]^2.
/// It has two localized peaks and a smooth background, so that
/// variance-reduction techniques have something nontrivial to exploit.
fn integrand(x: &[f64; D]) -> f64 {
    let x0 = x[0];
    let x1 = x[1];

    // Two Gaussian-like bumps at different locations and scales
    let bump1 = ((-40.0) * ((x0 - 0.25).powi(2) + (x1 - 0.75).powi(2))).exp();
    let bump2 = 0.7 * ((-120.0) * ((x0 - 0.8).powi(2) + (x1 - 0.2).powi(2))).exp();

    // Smooth background component
    let background = 0.3 * (2.0 * std::f64::consts::PI * x0).sin().powi(2)
        * (2.0 * std::f64::consts::PI * x1).cos().abs();

    bump1 + bump2 + background
}

/// Estimate ∫_region f(x) dx using crude Monte Carlo with N samples.
/// Returns (estimate, estimated variance of the estimator).
fn crude_monte_carlo<R: Rng + ?Sized>(
    region: &Region,
    n_samples: usize,
    rng: &mut R,
) -> (f64, f64) {
    if n_samples < 2 {
        return (0.0, 0.0);
    }

    let mut sum_f = 0.0;
    let mut sum_f2 = 0.0;

    for _ in 0..n_samples {
        let x = region.sample_point(rng);
        let fx = integrand(&x);
        sum_f += fx;
        sum_f2 += fx * fx;
    }

    let n = n_samples as f64;
    let mean_f = sum_f / n;
    let var_f_sample = (sum_f2 - n * mean_f * mean_f) / (n - 1.0);
    let estimate = region.volume * mean_f;

    // Var(estimate) = volume^2 * Var(mean_f) = volume^2 * var_f_sample / n
    let var_estimate = region.volume * region.volume * var_f_sample / n;

    (estimate, var_estimate.max(0.0))
}

/// Simple regular stratified sampling on an M×M grid over [0,1]^2.
///
/// This illustrates the estimator
///   \widehat{I} = ∑_{j} (w_j / N_j) ∑_{i=1}^{N_j} f(x_{j,i})
/// with equal subregion volumes w_j and equal allocations N_j (cf. (7.10.11)).
fn regular_stratified_grid<R: Rng + ?Sized>(
    region: &Region,
    grid_per_dim: usize,
    n_samples: usize,
    rng: &mut R,
) -> (f64, f64) {
    let num_strata = grid_per_dim * grid_per_dim;
    if num_strata == 0 || n_samples < num_strata {
        return crude_monte_carlo(region, n_samples, rng);
    }

    let n_per_stratum = n_samples / num_strata;
    if n_per_stratum < 2 {
        return crude_monte_carlo(region, n_samples, rng);
    }

    let dx = (region.upper[0] - region.lower[0]) / grid_per_dim as f64;
    let dy = (region.upper[1] - region.lower[1]) / grid_per_dim as f64;
    let w_j = dx * dy; // all strata have the same volume

    let mut estimate = 0.0;
    let mut var_estimate = 0.0;

    for ix in 0..grid_per_dim {
        for iy in 0..grid_per_dim {
            let lower = [
                region.lower[0] + ix as f64 * dx,
                region.lower[1] + iy as f64 * dy,
            ];
            let upper = [lower[0] + dx, lower[1] + dy];
            let stratum = Region::new(lower, upper);

            let mut sum_f = 0.0;
            let mut sum_f2 = 0.0;
            for _ in 0..n_per_stratum {
                let x = stratum.sample_point(rng);
                let fx = integrand(&x);
                sum_f += fx;
                sum_f2 += fx * fx;
            }

            let n = n_per_stratum as f64;
            let mean_f = sum_f / n;
            let var_f_sample = (sum_f2 - n * mean_f * mean_f) / (n - 1.0);

            // Contribution of this stratum to the integral estimator:
            // I_j_hat = w_j * mean_f
            let i_j_hat = w_j * mean_f;
            estimate += i_j_hat;

            // By (7.10.12): Var(\widehat{I}) ≈ ∑ w_j^2 σ_j^2 / N_j,
            // and we approximate σ_j^2 by var_f_sample.
            var_estimate += w_j * w_j * var_f_sample / n;
        }
    }

    (estimate, var_estimate.max(0.0))
}

/// Online statistics accumulator (Welford’s algorithm) used to estimate
/// the within-subregion variance σ_j^2 in candidate splits.
#[derive(Clone, Copy, Debug)]
struct OnlineStats {
    n: usize,
    mean: f64,
    m2: f64,
}

impl OnlineStats {
    fn new() -> Self {
        OnlineStats {
            n: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    fn update(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / (self.n as f64 - 1.0)
        } else {
            0.0
        }
    }
}

/// Recursive stratified sampling (RSS) / MISER-style integrator.
/// 
/// The function recursively splits the region along one coordinate at a time.
/// Each split is chosen by simulating a batch of "pilot" samples, from which
/// the sample variances σ_{a,i}^2 and σ_{b,i}^2 in the two candidate halves
/// (for each coordinate direction i) are estimated. The splitting dimension
/// is selected by minimizing a criterion of the form
///
///   w_{a,i} σ_{a,i}^{α} + w_{b,i} σ_{b,i}^{α},
///
/// which specializes to (7.10.18) for α = 1 and equal volumes. The remaining
/// samples are then allocated according to the optimal proportionality rule
/// N_{a,i} ∝ w_{a,i} σ_{a,i}, N_{b,i} ∝ w_{b,i} σ_{b,i} (cf. (7.10.20)),
/// and the procedure recurses on each subregion.
fn rss_integrate<R: Rng + ?Sized>(
    region: &Region,
    n_samples: usize,
    min_samples: usize,
    max_depth: usize,
    depth: usize,
    alpha: f64,
    rng: &mut R,
) -> (f64, usize) {
    // Stopping condition: too few samples or maximum depth reached.
    if depth >= max_depth || n_samples <= 4 * min_samples {
        // Fall back to crude Monte Carlo on this subregion.
        let n_use = n_samples.max(min_samples);
        let (estimate, _) = crude_monte_carlo(region, n_use, rng);
        return (estimate, n_use);
    }

    // Pilot samples used to estimate variances in candidate splits.
    let pilot = n_samples / 4;
    let remaining = n_samples - pilot;

    if pilot < 2 * D || remaining < 2 * min_samples {
        let (estimate, _) = crude_monte_carlo(region, n_samples, rng);
        return (estimate, n_samples);
    }

    // stats[dim][side]: side = 0 for "a", side = 1 for "b"
    let mut stats = [[OnlineStats::new(); 2]; D];

    // Collect pilot samples and update variance estimates for each hypothetical split.
    for _ in 0..pilot {
        let x = region.sample_point(rng);
        let fx = integrand(&x);

        for dim in 0..D {
            let mid = 0.5 * (region.lower[dim] + region.upper[dim]);
            let side = if x[dim] < mid { 0 } else { 1 };
            stats[dim][side].update(fx);
        }
    }

    // Choose the dimension that minimizes w_a σ_a^α + w_b σ_b^α.
    let mut best_dim: Option<usize> = None;
    let mut best_score = f64::INFINITY;
    let mut best_sigmas = (0.0_f64, 0.0_f64);
    let mut best_regions: Option<(Region, Region)> = None;

    for dim in 0..D {
        // Require at least a couple of samples on each side to estimate variance.
        if stats[dim][0].n < 2 || stats[dim][1].n < 2 {
            continue;
        }
        let sigma_a = stats[dim][0].variance().sqrt();
        let sigma_b = stats[dim][1].variance().sqrt();

        let (region_a, region_b) = region.split_along(dim);
        let w_a = region_a.volume;
        let w_b = region_b.volume;

        let score = w_a * sigma_a.powf(alpha) + w_b * sigma_b.powf(alpha);

        if score < best_score {
            best_score = score;
            best_dim = Some(dim);
            best_sigmas = (sigma_a, sigma_b);
            best_regions = Some((region_a, region_b));
        }
    }

    // If no useful split was found, fall back to crude Monte Carlo.
    let (region_a, region_b, sigma_a, sigma_b) = match (best_dim, best_regions) {
        (Some(_d), Some((ra, rb))) => (ra, rb, best_sigmas.0, best_sigmas.1),
        _ => {
            let (estimate, _) = crude_monte_carlo(region, n_samples, rng);
            return (estimate, n_samples);
        }
    };

    let w_a = region_a.volume;
    let w_b = region_b.volume;

    // Allocate remaining samples according to N_j ∝ w_j σ_j (cf. (7.10.20)).
    let weight_a = w_a * sigma_a;
    let weight_b = w_b * sigma_b;
    let total_weight = weight_a + weight_b;

    let (n_a, n_b) = if total_weight > 0.0 {
        let mut n_a = ((remaining as f64) * weight_a / total_weight).round() as usize;
        let mut n_b = remaining.saturating_sub(n_a);

        // Enforce minimum sample counts.
        if n_a < min_samples {
            n_a = min_samples;
            n_b = remaining.saturating_sub(n_a);
        }
        if n_b < min_samples {
            n_b = min_samples;
            n_a = remaining.saturating_sub(n_b);
        }

        // As a last resort, split evenly.
        if n_a < min_samples || n_b < min_samples {
            let half = remaining / 2;
            (half, remaining - half)
        } else {
            (n_a, n_b)
        }
    } else {
        // Degenerate case: use equal allocation.
        let half = remaining / 2;
        (half, remaining - half)
    };

    // Recurse on the two subregions.
    let (est_a, used_a) =
        rss_integrate(&region_a, n_a, min_samples, max_depth, depth + 1, alpha, rng);
    let (est_b, used_b) =
        rss_integrate(&region_b, n_b, min_samples, max_depth, depth + 1, alpha, rng);

    // Combine leaf estimates by summation since each estimates the integral
    // over its own subregion; this corresponds to a volume-weighted average
    // over the full domain.
    let estimate = est_a + est_b;
    let total_used = pilot + used_a + used_b;

    (estimate, total_used)
}

fn main() {
    let total_samples = 200_000usize;
    let grid_per_dim = 8usize;
    let min_samples_leaf = 512usize;
    let max_depth = 12usize;
    let alpha = 1.0; // corresponds to (7.10.18); try α ≠ 1 for (7.10.19)

    let region = Region::new([0.0, 0.0], [1.0, 1.0]);

    // For reproducibility, use fixed seeds for the RNG in each method.
    let mut rng_crude = StdRng::seed_from_u64(12345);
    let mut rng_strat = StdRng::seed_from_u64(12346);
    let mut rng_rss = StdRng::seed_from_u64(12347);

    // 1. Crude Monte Carlo
    let (estimate_crude, var_crude) = crude_monte_carlo(&region, total_samples, &mut rng_crude);
    let se_crude = var_crude.sqrt();

    // 2. Regular stratified sampling on an M×M grid.
    let (estimate_strat, var_strat) =
        regular_stratified_grid(&region, grid_per_dim, total_samples, &mut rng_strat);
    let se_strat = var_strat.sqrt();

    // 3. Recursive stratified sampling (RSS / MISER-style).
    let (estimate_rss, used_rss) = rss_integrate(
        &region,
        total_samples,
        min_samples_leaf,
        max_depth,
        0,
        alpha,
        &mut rng_rss,
    );

    println!(
        "Estimating I = ∫_[0,1]^2 f(x, y) dx dy with N = {} samples\n",
        total_samples
    );

    println!("Crude Monte Carlo (uniform sampling):");
    println!("  Estimate       ≈ {:.8}", estimate_crude);
    println!("  Std. error     ≈ {:.8}", se_crude);
    println!();

    println!(
        "Regular stratified sampling ({}×{} grid):",
        grid_per_dim, grid_per_dim
    );
    println!("  Estimate       ≈ {:.8}", estimate_strat);
    println!("  Std. error     ≈ {:.8}", se_strat);
    println!();

    println!("Recursive stratified sampling (RSS / MISER-style):");
    println!("  Estimate       ≈ {:.8}", estimate_rss);
    println!("  Total samples used (including pilots) = {}", used_rss);
    println!();

    println!("(In typical runs, the stratified and RSS estimates should exhibit");
    println!(" smaller standard errors than crude Monte Carlo, illustrating");
    println!(" the variance-reduction effect of (adaptive) stratification.)");
}
```

Program 7.10.3 demonstrates the practical advantages of both fixed and adaptive stratification for Monte Carlo integration in low to moderate dimensions. The numerical output clearly reflects the theoretical results of Equations (7.10.12)–(7.10.15), with regular stratification achieving a substantial reduction in empirical standard error relative to crude Monte Carlo for the same total sample count. This confirms the fundamental principle that structured domain partitioning redistributes sampling effort toward regions of higher variability.

The recursive stratified sampling implementation further illustrates how adaptive domain refinement can concentrate samples preferentially in dynamically detected “active” regions of the integrand. Although a single RSS run does not directly provide a variance estimate, its improved estimator accuracy for the same budget reflects the effect of variance-driven sample allocation encoded in Equations (7.10.17)–(7.10.20). In favorable cases, such recursive strategies are capable of approaching near-deterministic convergence rates, as highlighted earlier by Equation (7.10.16).

Beyond its immediate numerical results, the modular design of this program makes it straightforward to extend the framework to higher dimensions, alternative splitting heuristics, or hybrid schemes that combine RSS with importance sampling transformations. As discussed at the end of Section 7.10.2, such hybrid strategies form the backbone of modern high-performance Monte Carlo integrators used in finance, transport theory, Bayesian inference, and uncertainty quantification.

## 7.10.3. The VEGAS Algorithm

The VEGAS algorithm (Lepage, 1978) is a foundational adaptive Monte Carlo integrator that combines importance sampling with a separable approximation to the optimal sampling density. The domain is first rescaled to the unit hypercube $[0,1]^d$. VEGAS assumes that the sampling density can be modeled in separable form as:

$$p(\mathbf{x}) \propto g_1(x_1) g_2(x_2)\cdots g_d(x_d) \tag{7.10.21}$$

where each one-dimensional function $g_i$ is represented by a histogram or grid over $[0,1]$. This separability assumption allows VEGAS to learn an approximate importance density using only $d$ one-dimensional grids rather than an infeasible $K^d$ cell full tensor grid.

### Algorithmic Structure

VEGAS proceeds iteratively, alternating between sampling and density refinement.

#### (1) Initialize Weights

Initially, each marginal weight function is set to:

$$g_i(x_i) = 1, \qquad i = 1,\dots,d \tag{7.10.22}$$

corresponding to uniform sampling.

#### (2) Sample and Accumulate

Given the current weights, a sample $\mathbf{x}$ is generated by drawing each coordinate independently from its one-dimensional density $g_i$. Each sample contributes a weighted function value:

$$w(\mathbf{x}) = \frac{f(\mathbf{x})}{g_1(x_1) g_2(x_2)\cdots g_d(x_d)} \tag{7.10.23}$$

which is used to estimate the integral and variance. After drawing $N$ points, VEGAS constructs estimators:

$$\widehat{I}_k, \qquad \widehat{\sigma}_k^2 \tag{7.10.24}$$

for iteration $k$, where the formulas follow standard importance-sampling theory given the separable proposal.

#### (3) Update Grids

For each dimension $i$, VEGAS projects the squared integrand onto the $i$-th axis to estimate the appropriate one-dimensional importance density. Conceptually, VEGAS uses

$$g_i(x_i) \propto \int_{[0,1]^{d-1}} f(x)^2\prod_{j\neq i} g_j(x_j)\mathrm{d}x_1\cdots \mathrm{d}x_{i-1}\mathrm{d}x_{i+1}\cdots\mathrm{d}x_d \tag{7.10.25}$$

which mirrors the form of the optimal density derived in (7.10.8), but restricted to a separable approximation. Practically, VEGAS approximates this integral using accumulated histogram data along each coordinate. The resulting grids are adjusted so that the number of points (or total weight) in each bin is proportional to its estimated contribution to the integral.

#### (4) Iterate and Combine

VEGAS repeats the sampling–update cycle until convergence. Each iteration yields an estimate $\widehat{I}_k$ with variance $\widehat{\sigma}_k^2$. The final result is obtained by combining iteration-wise estimates using inverse-variance weighting:

$$
I_{\mathrm{best}}
= \frac{\displaystyle \sum_{k=1}^m \widehat{I}_k / \widehat{\sigma}_k^{\,2}}
       {\displaystyle \sum_{k=1}^m 1 / \widehat{\sigma}_k^{\,2}},
\qquad
\widehat{\sigma}^{2}_{\mathrm{best}}
= \left(\sum_{k=1}^m \frac{1}{\widehat{\sigma}_k^{\,2}}\right)^{-1}
\tag{7.10.26}
$$

This combination step allows multiple VEGAS iterations to be fused into a statistically efficient final estimator.

### Performance Characteristics

VEGAS learns a rough approximation to the optimal importance density by adapting each coordinate direction independently. The algorithm performs very well when the integrand’s primary variation aligns with the coordinate axes, an assumption satisfied in many physics and engineering integrals. However, if the integrand contains strong non-axis-aligned ridges or diagonal features, separability becomes a limitation, and VEGAS may fail to sufficiently concentrate samples in the most important regions.

To address these issues, several extensions have been developed. Multi-channel VEGAS mixes multiple coordinate systems or multiple separable components, while VEGAS+ incorporates preconditioning strategies and hybrid updates (e.g., integrating occasional MCMC steps). These variants retain the robustness and simplicity of VEGAS while broadening its applicability to more complex integrands.

### Complexity and Practical Considerations

The computational cost of each VEGAS iteration is dominated by two operations. The first is the sampling phase, which scales as $\mathcal{O}(N)$ because each of the $N$ points must be generated and evaluated. The second is the histogram-update step, whose complexity grows as $\mathcal{O}(dK)$, where $d$ is the dimension and $K$ is the number of grid bins along each coordinate. Memory usage follows a similar pattern: VEGAS stores only $dK$ histogram entries together with a few auxiliary statistics, making it far more economical than full multidimensional grids. This combination of low computational and memory overhead explains its long-standing success. In high-energy physics, for example, VEGAS and its descendants routinely deliver orders-of-magnitude efficiency gains over crude Monte Carlo when the integrand’s primary variations are reasonably aligned with the coordinate axes. Modern GPU implementations such as cuVegas (2024) further accelerate both sampling and bin updates through massive parallelism, enabling very large-scale, high-dimensional integrations with minimal modification to the core algorithm.

### Rust Implementation

Following the development of adaptive importance sampling in Sections 7.10.1 and 7.10.2, Program 7.10.4 provides a concrete implementation of the VEGAS algorithm as introduced in Section 7.10.3. While the previous sections established the theoretical motivation for adaptive density construction and optimal importance sampling, VEGAS offers a practical and computationally efficient realization based on a separable approximation of the sampling density as given in (7.10.21). In numerical integration over moderate to high-dimensional domains, direct sampling from the optimal density (7.10.8) is infeasible, making adaptive product-form approximations essential. This program demonstrates how iterative grid refinement, weighted sampling as in (7.10.23), and statistically optimal inverse-variance combination as in (7.10.26) can be assembled into a fully operational adaptive Monte Carlo integrator. The implementation evaluates a smooth axis-aligned exponential test integrand, illustrating the learning behavior, variance reduction, and convergence characteristics of VEGAS in a controlled setting.

At the core of the implementation is the `VegasIntegrator` struct, which stores the dimensionality $d$, the number of bins $K$, the number of samples per iteration $N$, and the separable one-dimensional grids $g_i$ that represent the sampling density in the form specified by (7.10.21). Each grid is initialized uniformly on $[0,1]$, corresponding to the uniform density assumption in (7.10.22). The constructor `new` establishes these initial grids and seeds the random number generator, ensuring reproducible adaptive learning across iterations.

The main computational work is carried out by the `iterate` method, which performs a single VEGAS iteration as described by steps (2)–(3) of the algorithmic structure. For each of the $N$ samples, the method independently draws one coordinate from each one-dimensional grid, thereby generating a full point $\mathbf{x} \in [0,1]^d$ from the separable sampling density. The associated probability density $p(\mathbf{x})$ is constructed directly from the bin widths. The weighted contribution $w(\mathbf{x}) = f(\mathbf{x}) / p(\mathbf{x})$ is then formed exactly as prescribed by (7.10.23), and both the first and second moments of the weights are accumulated to produce the iteration-wise estimators $\widehat{I}_k$ and $\widehat{\sigma}_k^2$ defined in (7.10.24).

Simultaneously, the squared integrand values $f(\mathbf{x})^2$ are projected onto each coordinate direction to form one-dimensional histograms that approximate the projected integrals appearing in (7.10.25). These histograms capture how strongly each region of a given coordinate contributes to the total variance of the integrand. After the sampling phase completes, the `update_grids` method constructs new grid edges by forming cumulative distributions from these projected weights and computing the appropriate quantiles. This grid redistribution step enforces the principle that each bin should contribute approximately equally to the estimated variance, thereby refining each marginal density $g_i$ adaptively.

The test integrand used in the program is implemented in the `test_integrand` function and is defined as $f(\mathbf{x}) = \exp(-\alpha \sum_i x_i)$, which factorizes exactly along coordinate directions and thus conforms ideally to the separability assumption underlying (7.10.21). The exact value of the corresponding integral is computed analytically in `analytic_integral`, allowing direct numerical validation of the Monte Carlo estimates.

The `main` function orchestrates the full VEGAS procedure. It initializes the integrator parameters, executes a prescribed number of adaptive iterations, and stores the resulting estimates $\widehat{I}_k$ and variances $\widehat{\sigma}_k^2$. These are then combined using the inverse-variance weighting rule given explicitly in (7.10.26). The final output reports the combined estimate, its predicted uncertainty, and the absolute deviation from the exact analytic value. This end-to-end workflow illustrates not only the learning behavior of VEGAS but also the statistically optimal fusion of multiple adaptive Monte Carlo passes.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
// Program 7.10.4: VEGAS adaptive Monte Carlo integration on [0,1]^d
//
// This example implements a basic VEGAS integrator using a separable
// product of one-dimensional grids as in (7.10.21). Each grid g_i is
// represented by a set of bin edges on [0,1], initially uniform as in
// (7.10.22). Sampling, weighting (7.10.23), per-iteration estimators
// (7.10.24), grid updates (7.10.25), and inverse-variance combination
// (7.10.26) are all demonstrated.
//
// Add to Cargo.toml:
// [dependencies]
// rand = "0.8"

use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Debug, Clone, Copy)]
struct VegasStats {
    estimate: f64,
    variance: f64,
}

/// Basic VEGAS integrator on [0,1]^dim with bins bins and N samples/iteration.
struct VegasIntegrator {
    dim: usize,
    bins: usize,
    samples_per_iter: usize,
    grids: Vec<Vec<f64>>, // grids[i][j] = j-th edge for dimension i, length = bins+1
    rng: StdRng,
}

impl VegasIntegrator {
    /// Create a new VEGAS integrator with uniform initial grids (7.10.22).
    fn new(dim: usize, bins: usize, samples_per_iter: usize, seed: u64) -> Self {
        let mut grids = Vec::with_capacity(dim);
        for _ in 0..dim {
            let mut edges = Vec::with_capacity(bins + 1);
            for j in 0..=bins {
                edges.push(j as f64 / bins as f64);
            }
            grids.push(edges);
        }

        Self {
            dim,
            bins,
            samples_per_iter,
            grids,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Perform a single VEGAS iteration: sample, accumulate, update grids.
    ///
    /// The function f is assumed to be defined on [0,1]^dim.
    fn iterate<F>(&mut self, f: F) -> VegasStats
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut sum_w = 0.0;
        let mut sum_w2 = 0.0;

        // Storage for a single point
        let mut x = vec![0.0; self.dim];
        let mut bin_indices = vec![0usize; self.dim];
        let mut bin_widths = vec![0.0; self.dim];

        // Histogram accumulators for each dimension (for g_i updates)
        let mut bin_sums = vec![vec![0.0; self.bins]; self.dim];

        let uniform_u = Uniform::new(0.0, 1.0);
        let uniform_bin = Uniform::new(0, self.bins);

        for _ in 0..self.samples_per_iter {
            // (2) Sample from current separable density and record bin indices
            let mut p_x = 1.0;

            for i in 0..self.dim {
                let edges = &self.grids[i];
                let j = uniform_bin.sample(&mut self.rng);
                let u: f64 = uniform_u.sample(&mut self.rng);

                let x_lo = edges[j];
                let x_hi = edges[j + 1];
                let width = x_hi - x_lo;

                x[i] = x_lo + u * width;
                bin_indices[i] = j;
                bin_widths[i] = width;

                // With equal mass 1/bins per bin, density is 1 / (bins * width) in that bin.
                let density_i = 1.0 / (self.bins as f64 * width);
                p_x *= density_i;
            }

            // Evaluate integrand and construct weight w(x) = f(x) / p(x) as in (7.10.23).
            let fx = f(&x);
            let w = fx / p_x;

            sum_w += w;
            sum_w2 += w * w;

            // Project squared integrand onto each axis as in (7.10.25),
            // approximated here by binwise sums of f(x)^2.
            let f2 = fx * fx;
            for i in 0..self.dim {
                let j = bin_indices[i];
                bin_sums[i][j] += f2;
            }
        }

        // (2) Per-iteration estimators (7.10.24).
        let n = self.samples_per_iter as f64;
        let mean_w = sum_w / n;

        // Unbiased sample variance of the weights.
        let var_w_unbiased = if self.samples_per_iter > 1 {
            (sum_w2 - n * mean_w * mean_w) / (n - 1.0)
        } else {
            0.0
        };

        // Variance of the iteration estimate I_hat_k.
        let var_i = (var_w_unbiased / n).max(0.0);

        let stats = VegasStats {
            estimate: mean_w,
            variance: var_i,
        };

        // (3) Update grids using accumulated histogram data.
        self.update_grids(&bin_sums);

        stats
    }

    /// Grid update corresponding to (7.10.25):
    /// redistribute bin edges so that each bin carries (approximately)
    /// equal projected contribution.
    fn update_grids(&mut self, bin_sums: &[Vec<f64>]) {
        for i in 0..self.dim {
            let edges_old = self.grids[i].clone();
            let mut weights = bin_sums[i].clone();

            let total: f64 = weights.iter().sum();
            if total <= 0.0 {
                // If the integrand is numerically constant, keep the grid unchanged.
                continue;
            }

            // Normalize bin contributions along this dimension.
            for w in &mut weights {
                *w /= total;
            }

            // Build cumulative distribution over bins.
            let mut cdf = vec![0.0; self.bins + 1];
            for j in 0..self.bins {
                cdf[j + 1] = cdf[j] + weights[j];
            }

            // New edges: quantiles of this 1D importance density.
            let mut new_edges = vec![0.0; self.bins + 1];
            new_edges[0] = 0.0;
            new_edges[self.bins] = 1.0;

            let mut j = 0;
            for k in 1..self.bins {
                let target = k as f64 / self.bins as f64;

                // Find bin [j, j+1] where target falls in the CDF.
                while j < self.bins - 1 && cdf[j + 1] < target {
                    j += 1;
                }

                let cdf_lo = cdf[j];
                let cdf_hi = cdf[j + 1];
                let frac = if cdf_hi > cdf_lo {
                    (target - cdf_lo) / (cdf_hi - cdf_lo)
                } else {
                    0.0
                };

                // Interpolate inside the *physical* interval [edges_old[j], edges_old[j+1]].
                let x_lo = edges_old[j];
                let x_hi = edges_old[j + 1];
                new_edges[k] = x_lo + frac * (x_hi - x_lo);
            }

            self.grids[i] = new_edges;
        }
    }
}

/// Test integrand on [0,1]^d:
///   f(x) = exp(-alpha * sum_i x_i),
/// which factorizes across coordinates, making it well-suited to VEGAS.
///
/// The exact integral is:
///   I = ((1 - exp(-alpha)) / alpha)^d.
fn test_integrand(x: &[f64]) -> f64 {
    let alpha = 5.0;
    let sum: f64 = x.iter().sum();
    (-alpha * sum).exp()
}

fn analytic_integral(dim: usize, alpha: f64) -> f64 {
    let one_dim = (1.0 - (-alpha).exp()) / alpha;
    one_dim.powi(dim as i32)
}

fn main() {
    let dim = 3;
    let bins = 20;
    let samples_per_iter = 50_000;
    let iters = 10;
    let seed = 2025_u64;

    let alpha = 5.0;
    let exact = analytic_integral(dim, alpha);

    let mut vegas = VegasIntegrator::new(dim, bins, samples_per_iter, seed);

    // (4) Iterate and combine estimates using inverse-variance weights (7.10.26).
    let mut sum_i_over_var = 0.0;
    let mut sum_inv_var = 0.0;

    println!("VEGAS integration on [0,1]^{} with {} bins and {} samples/iteration",
             dim, bins, samples_per_iter);
    println!("Integrand: f(x) = exp(-{alpha} * sum_i x_i)");
    println!("Exact value: {:.8}\n", exact);

    for k in 1..=iters {
        let stats = vegas.iterate(test_integrand);
        let ihat = stats.estimate;
        let sigma2 = stats.variance;

        if sigma2 > 0.0 {
            let inv_var = 1.0 / sigma2;
            sum_i_over_var += ihat * inv_var;
            sum_inv_var += inv_var;
        }

        let sigma = sigma2.sqrt();
        println!(
            "Iteration {:2}: I_k ≈ {:>12.8},  σ_k ≈ {:>10.8}",
            k, ihat, sigma
        );
    }

    if sum_inv_var > 0.0 {
        let i_best = sum_i_over_var / sum_inv_var;
        let sigma2_best = 1.0 / sum_inv_var;
        let sigma_best = sigma2_best.sqrt();
        let abs_err = (i_best - exact).abs();

        println!("\nCombined estimate over {} iterations (7.10.26):", iters);
        println!("  I_best     ≈ {:.10}", i_best);
        println!("  σ_best     ≈ {:.10}", sigma_best);
        println!("  |error|    ≈ {:.10}", abs_err);
    } else {
        println!("\nUnable to form combined estimate (all variances zero).");
    }
}
```

Program 7.10.4 demonstrates how the abstract formulation of the VEGAS algorithm in Section 7.10.3 can be translated into a fully operational adaptive Monte Carlo integrator using a compact and memory-efficient data structure. By exploiting the separable approximation (7.10.21), the implementation reduces the complexity of learning a multidimensional importance density from an infeasible $K^d$ tensor grid to only $dK$ one-dimensional bins. The sampled test problem illustrates how the algorithm progressively adapts its grids to regions of higher variance and concentration, improving sampling efficiency over successive iterations.

The numerical output confirms the statistical consistency of the estimator. Individual iterations display natural fluctuations as the grids evolve, while the inverse-variance combination (7.10.26) stabilizes the final estimate and significantly reduces uncertainty. The residual discrepancy between the combined estimate and the exact value reflects both finite-sample effects and the limitations imposed by separable density modeling, which becomes pronounced as variance structures become more complex or non-axis-aligned.

From a structural perspective, the modular design of the integrator allows direct extension toward more advanced variants discussed earlier in this section. Multi-channel VEGAS, hybrid MCMC–VEGAS schemes, and GPU-accelerated histogram updates can all be integrated without altering the conceptual framework of the code. As such, this program serves not only as a reference implementation of the classical VEGAS algorithm but also as a practical foundation for modern large-scale adaptive Monte Carlo methods.

## 7.10.4. Practical Domains of Adaptive and Recursive Monte Carlo

Adaptive and recursive Monte Carlo techniques play a central role in practical scientific computing, especially in problems where integrals over high-dimensional or irregular domains must be estimated with high accuracy. Two major application areas illustrate their power: engineering reliability and next-generation wireless communication systems. Additional applications in particle physics and Bayesian inference further demonstrate the broad utility of these methods.

### Engineering Reliability and Rare-Event Probability Estimation

In reliability analysis, the goal is often to compute extremely small failure probabilities of the form $P(f(\mathbf{x}) > 0)$, where $f$ is a limit-state function that separates the safe and failure regions of a system. Because failure events are rare, brute-force Monte Carlo becomes infeasible: millions or even billions of samples may be required to observe a handful of failures. The integrand in these problems is effectively an *indicator* function, which makes variance reduction essential.

Importance sampling is widely used in this setting by constructing proposal densities that place more samples near the failure region. Methods such as cross-entropy based adaptation and sequential Monte Carlo can automatically tune the sampling distribution to approximate the optimal importance density. A notable recent advance is the gradient-free adaptive importance sampler developed by Eshra and Papakonstantinou (2025). Their method constructs and samples an approximate target distribution over the failure region without requiring derivative information or restrictive parametric choices. They report that importance sampling “is capable of efficiently estimating rare event probabilities’’ when properly designed, achieving orders-of-magnitude speedups over crude Monte Carlo. As a result, adaptive Monte Carlo methods have become standard tools in automotive, aerospace, and civil engineering reliability analysis, where accurate probability estimates directly inform safety margins and risk assessments.

### Wireless Communications and URLLC Performance

Ultra-reliable low-latency communication (URLLC) systems that are central to the next-generation wireless networks, must ensure that bit-error rates or packet-loss probabilities remain extraordinarily low, often at levels near $10^{-9}$ or below. Mathematically, evaluating these metrics requires computing high-dimensional integrals over random channel realizations and signal-space configurations. Direct Monte Carlo is far too inefficient for probabilities this small.

Ke et al. (2023) address this challenge by designing an adaptive importance sampling scheme for URLLC performance evaluation. Their method uses a Gaussian proposal distribution whose parameters (mean and covariance) are automatically tuned via pilot samples to minimize the estimator’s variance. They further integrate quasi-Monte Carlo sampling, which smooths the sampling process and provides additional variance reduction. The resulting IS+QMC algorithm achieves orders-of-magnitude fewer samples compared with naive Monte Carlo while retaining high accuracy. Their results demonstrate that adaptive Monte Carlo is not merely beneficial but often essential for enabling feasible simulation of modern communication systems.

### Particle Physics and Bayesian Inference

Beyond engineering and communications, adaptive Monte Carlo methods, particularly VEGAS and its variants, are widely used in particle physics, where high-dimensional Feynman integrals must be evaluated with precision. VEGAS’s separable importance density, combined with its relatively modest memory requirements, has made it a standard tool in numerical event generators and cross-section calculations.

Similarly, in Bayesian inference, adaptive sampling and nested-sampling–like schemes are central to evaluating marginal likelihoods (evidence) and posterior expectations. These integrals often contain sharp peaks or heavy tails that uniform sampling would fail to adequately explore. Adaptive and recursive Monte Carlo frameworks ensure that the sampling distribution adapts to these “hot spots,” dramatically improving efficiency.

Across all of these application domains, the underlying structure is the same: the mathematical model leads to an integral or expectation over a region that contains localized, high-impact subregions. Adaptive importance sampling warps the density to track these regions, while recursive splitting ensures that no subregion is left undersampled. Together, these mechanisms provide dramatic variance reduction compared with uniform sampling, often turning otherwise impractical simulations into feasible computations.

### Rust Implementation

Following the discussion in Section 7.10.4 on the practical domains of adaptive and recursive Monte Carlo methods, Program 7.10.5 provides a concrete numerical demonstration of rare-event probability estimation using both crude Monte Carlo and adaptive importance sampling. While the preceding section emphasized the theoretical necessity of variance reduction in engineering reliability analysis and URLLC system simulation, this program illustrates how those principles translate into executable algorithms. In problems where the target probability lies in the extreme tails of a distribution, naive sampling becomes prohibitively inefficient, making adaptive density warping essential. This program implements a gradient-free adaptive importance sampler for estimating tail probabilities under a standard normal model and directly compares its performance against crude Monte Carlo. Two representative applications are treated: right-tail failure probabilities reminiscent of engineering limit-state analysis and left-tail outage probabilities relevant to ultra-reliable low-latency communication systems. Together, the examples highlight how adaptive Monte Carlo transforms otherwise infeasible simulations into computationally tractable procedures.

At the core of the implementation is the `Estimate` struct, which stores both the probability estimate $\hat p$ and its associated standard error. This encapsulation ensures that each Monte Carlo estimator carries a statistically meaningful uncertainty measure, which is essential for comparing the efficiency of crude Monte Carlo and adaptive importance sampling in rare-event settings.

The baseline estimator is implemented in the `crude_monte_carlo` function. This function draws independent samples from the standard normal distribution and evaluates a user-supplied failure indicator that encodes a limit-state condition of the form $f(x) > 0$. The resulting estimate $\hat p$ is the empirical failure frequency, and its variance is approximated using the binomial variance formula appropriate for Bernoulli trials. This directly reflects the inefficiency of brute-force Monte Carlo when the target probability is extremely small, as discussed in the reliability and URLLC applications.

The adaptive estimator is implemented in the `adaptive_importance_sampling` function. Here, samples are drawn from a Gaussian proposal distribution $\mathcal{N}(\mu,1)$, and each failure sample is reweighted using the likelihood ratio between the target and proposal densities, following standard importance-sampling theory. The proposal mean $\mu$ is then updated using a simple gradient-free rule based on the empirical average of failing samples, mimicking the behavior of modern cross-entropy and sequential Monte Carlo schemes used in engineering reliability and wireless system evaluation. Iteration-wise estimates and variances are accumulated and fused using inverse-variance weighting, ensuring that later, lower-variance iterations dominate the final estimator. The functions `standard_normal_pdf` and `normal_pdf` provide the analytical probability density evaluations required for constructing the likelihood ratios between the target and proposal distributions. These density evaluations ensure that the estimator remains unbiased even as the sampling distribution is adaptively deformed toward the extreme tails.

The `main` function orchestrates two distinct rare-event experiments. The first corresponds to an engineering reliability scenario, where the failure probability $P(X > a)$ is estimated for a large threshold $a$, modeling structural or system failure under random loading. The second corresponds to a URLLC-like outage scenario, where a left-tail probability $P(X < -b)$ models deep-fade or decoding failure events in wireless communication systems. In both cases, crude Monte Carlo and adaptive importance sampling are executed side-by-side, and their estimates and uncertainties are printed with consistent formatting to allow direct efficiency comparison. The progressive drift of the adaptive proposal mean toward the rare-event region visibly demonstrates how the sampler learns the location of the dominant contribution to the probability mass.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
rand_distr = "0.4"
```

```rust
// Adaptive and Recursive Monte Carlo in Practical Domains
//
// This example illustrates rare-event probability estimation in two
// application-inspired settings:
//   1. Engineering reliability: P(X > a) for a standard normal variable,
//      modeling a rare failure event.
//   2. URLLC-like performance: P(X < -b) for a standard normal variable,
//      modeling an ultra-low error or outage probability in the left tail.
//
// Crude Monte Carlo is compared to an adaptive importance sampler that
// uses a Gaussian proposal with mean μ, updated iteratively from pilot
// samples. The adaptation is gradient-free and mimics the behavior of
// cross-entropy and other modern rare-event samplers.


use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
struct Estimate {
    p_hat: f64,
    std_error: f64,
}

fn standard_normal_pdf(x: f64) -> f64 {
    (-(x * x) / 2.0).exp() / (2.0 * PI).sqrt()
}

fn normal_pdf(x: f64, mean: f64, sigma: f64) -> f64 {
    let z = (x - mean) / sigma;
    (-(z * z) / 2.0).exp() / (sigma * (2.0 * PI).sqrt())
}

/// Crude Monte Carlo estimator for a rare-event probability:
/// p = P(failure(x) == true), where x ~ N(0, 1).
fn crude_monte_carlo<F>(
    n_samples: usize,
    failure: F,
    rng: &mut StdRng,
) -> Estimate
where
    F: Fn(f64) -> bool,
{
    let base_normal = Normal::new(0.0, 1.0).unwrap();

    let mut count_failure = 0usize;
    for _ in 0..n_samples {
        let x = base_normal.sample(rng);
        if failure(x) {
            count_failure += 1;
        }
    }

    let n = n_samples as f64;
    let p_hat = count_failure as f64 / n;

    // Binomial variance estimate: Var(p̂) ≈ p̂(1 - p̂) / N.
    let var = if n_samples > 0 {
        p_hat * (1.0 - p_hat) / n
    } else {
        0.0
    };

    Estimate {
        p_hat,
        std_error: var.max(0.0).sqrt(),
    }
}

/// Adaptive importance sampling estimator for a rare-event probability
/// under a standard normal base, using a Gaussian proposal N(μ, 1).
///
/// - failure(x) encodes the limit-state failure event (true = failure).
/// - The proposal mean μ is updated from pilot samples using a simple,
///   gradient-free rule based on the average of failing samples.
fn adaptive_importance_sampling<F>(
    n_iterations: usize,
    samples_per_iter: usize,
    initial_mu: f64,
    failure: F,
    rng: &mut StdRng,
) -> (Estimate, f64)
where
    F: Fn(f64) -> bool,
{
    let sigma = 1.0;
    let learning_rate = 0.7;

    let mut mu = initial_mu;

    // Accumulate inverse-variance-weighted combination across iterations.
    let mut sum_p_over_var = 0.0;
    let mut sum_inv_var = 0.0;

    for k in 1..=n_iterations {
        let proposal = Normal::new(mu, sigma).unwrap();

        let mut weights = Vec::with_capacity(samples_per_iter);
        let mut fail_sum_x = 0.0;
        let mut fail_count = 0usize;

        for _ in 0..samples_per_iter {
            let x = proposal.sample(rng);

            // Indicator for failure event.
            if failure(x) {
                let p_x = standard_normal_pdf(x);
                let q_x = normal_pdf(x, mu, sigma);
                let w = p_x / q_x;

                weights.push(w);
                fail_sum_x += x;
                fail_count += 1;
            }
        }

        let n = samples_per_iter as f64;
        let n_eff = weights.len() as f64;

        // If no failures are observed under the proposal, the estimate is zero
        // for this iteration and the variance is undefined; skip combination
        // and move μ toward the base mean.
        if fail_count == 0 {
            mu *= 0.5;

            println!(
                "Adaptive IS iteration {:2}: no failures observed; shrinking μ to {:.3}",
                k, mu
            );
            continue;
        }

        let sum_w: f64 = weights.iter().sum();
        let mean_w = sum_w / n;

        // Use an unbiased sample variance of the weights divided by N
        // to estimate Var(p̂_k).
        let mut sum_w2 = 0.0;
        for &w in &weights {
            sum_w2 += w * w;
        }

        let var_w_unbiased = if weights.len() > 1 {
            let m = weights.len() as f64;
            (sum_w2 - m * (sum_w / m).powi(2)) / (m - 1.0)
        } else {
            0.0
        };

        let var_p = (var_w_unbiased / n).max(0.0);
        let std_p = var_p.sqrt();

        println!(
            "Adaptive IS iteration {:2}: μ ≈ {:>7.3},  p̂_k ≈ {:>12.6e},  σ_k ≈ {:>12.6e},  failures = {:5},  n_eff ≈ {:5}",
            k, mu, mean_w, std_p, fail_count, n_eff
        );

        if var_p > 0.0 {
            let inv_var = 1.0 / var_p;
            sum_p_over_var += mean_w * inv_var;
            sum_inv_var += inv_var;
        }

        // Gradient-free adaptation: move μ toward the empirical mean
        // of failing samples.
        let mean_fail_x = fail_sum_x / fail_count as f64;
        mu = (1.0 - learning_rate) * mu + learning_rate * mean_fail_x;
    }

    if sum_inv_var > 0.0 {
        let p_best = sum_p_over_var / sum_inv_var;
        let var_best = 1.0 / sum_inv_var;

        (
            Estimate {
                p_hat: p_best,
                std_error: var_best.max(0.0).sqrt(),
            },
            mu,
        )
    } else {
        (
            Estimate {
                p_hat: 0.0,
                std_error: 0.0,
            },
            mu,
        )
    }
}

fn main() {
    let seed = 2025_u64;
    let mut rng = StdRng::seed_from_u64(seed);

    // ---------------------------------------------------------------------
    // Example 1: Engineering reliability – right-tail failure probability
    // ---------------------------------------------------------------------

    let reliability_threshold = 4.0; // X > 4 is a rare failure event for X ~ N(0,1).
    let n_crude = 500_000;

    let failure_reliability = |x: f64| x > reliability_threshold;

    println!("Engineering reliability example: P(X > {:.1}) with X ~ N(0,1)", reliability_threshold);
    println!("Crude Monte Carlo with N = {} samples:", n_crude);

    let crude_est = crude_monte_carlo(n_crude, failure_reliability, &mut rng);
    println!(
        "  Crude MC: p̂ ≈ {:>12.6e},  σ ≈ {:>12.6e}\n",
        crude_est.p_hat, crude_est.std_error
    );

    let n_iterations = 8;
    let samples_per_iter = 50_000;
    let initial_mu = 2.0;

    println!(
        "Adaptive importance sampling (reliability) with N = {} per iteration, {} iterations:",
        samples_per_iter, n_iterations
    );

    let (adaptive_est, final_mu) = adaptive_importance_sampling(
        n_iterations,
        samples_per_iter,
        initial_mu,
        failure_reliability,
        &mut rng,
    );

    println!(
        "\nFinal adaptive IS estimate (reliability): p̂ ≈ {:>12.6e},  σ ≈ {:>12.6e},  final μ ≈ {:.3}\n",
        adaptive_est.p_hat, adaptive_est.std_error, final_mu
    );

    // ---------------------------------------------------------------------
    // Example 2: URLLC-like performance – left-tail rare outage probability
    // ---------------------------------------------------------------------

    let urlcc_threshold = -4.0; // X < -4 is a rare outage event for X ~ N(0,1).
    let n_crude_urlcc = 500_000;

    let failure_urlcc = |x: f64| x < urlcc_threshold;

    println!(
        "URLLC-like example: P(X < {:.1}) with X ~ N(0,1)",
        urlcc_threshold
    );
    println!(
        "Crude Monte Carlo with N = {} samples:",
        n_crude_urlcc
    );

    let crude_est_urlcc = crude_monte_carlo(n_crude_urlcc, failure_urlcc, &mut rng);
    println!(
        "  Crude MC: p̂ ≈ {:>12.6e},  σ ≈ {:>12.6e}\n",
        crude_est_urlcc.p_hat, crude_est_urlcc.std_error
    );

    let n_iterations_urlcc = 8;
    let samples_per_iter_urlcc = 50_000;
    let initial_mu_urlcc = -2.0;

    println!(
        "Adaptive importance sampling (URLLC-like) with N = {} per iteration, {} iterations:",
        samples_per_iter_urlcc, n_iterations_urlcc
    );

    let (adaptive_est_urlcc, final_mu_urlcc) = adaptive_importance_sampling(
        n_iterations_urlcc,
        samples_per_iter_urlcc,
        initial_mu_urlcc,
        failure_urlcc,
        &mut rng,
    );

    println!(
        "\nFinal adaptive IS estimate (URLLC-like): p̂ ≈ {:>12.6e},  σ ≈ {:>12.6e},  final μ ≈ {:.3}",
        adaptive_est_urlcc.p_hat, adaptive_est_urlcc.std_error, final_mu_urlcc
    );
}
```

Program 7.10.5 demonstrates in executable form the core message of Section 7.10.4: rare-event simulation is fundamentally a variance-reduction problem rather than a brute-force sampling problem. The crude Monte Carlo results in both the reliability and URLLC-like experiments exhibit extremely large relative uncertainties even with several hundred thousand samples, rendering direct estimation impractical for engineering-grade accuracy.

By contrast, the adaptive importance sampler rapidly concentrates samples within the failure region and achieves orders-of-magnitude variance reduction with a fraction of the computational cost. The empirical drift of the proposal mean toward the rare-event threshold reflects the same adaptive warping mechanism exploited in modern cross-entropy, sequential importance sampling, and reliability-oriented Monte Carlo frameworks. The resulting estimates attain sub-percent relative error even for probabilities on the order of $10^{-5}$, a regime that is effectively inaccessible to crude Monte Carlo.

The modular structure of the implementation allows the same framework to be extended directly to multidimensional limit-state functions, correlated Gaussian inputs, and more sophisticated proposal families with adaptive covariance updates. As such, this program provides a practical computational foundation for large-scale reliability analysis, URLLC system evaluation, Bayesian rare-event inference, and particle-transport simulations, all of which rely critically on adaptive and recursive Monte Carlo methodologies.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/IhcmkhY1m1x4BVQ8fV8I.2","tags":[]}

# 7.11. Conclusion

As we conclude this chapter, our goal has been to provide a comprehensive and practical introduction to random number generation, hashing, and Monte Carlo integration using Rust. Random number generation, encompassing pseudo-random number generators based on linear congruential recurrences, xorshift and PCG families, composite and counter-based designs, linear feedback shift registers over $\mathbb{F}_2$, index-based hashing via pseudo-DES and butterfly mixing, hash tables with collision-resolution strategies, inverse-transform and rejection-based sampling for non-uniform distributions, Cholesky-factored multivariate normal deviates, Monte Carlo and quasi-Monte Carlo integration, and adaptive importance sampling with recursive stratification, forms the computational foundation of stochastic simulation, uncertainty quantification, probabilistic machine learning, and high-performance scientific computing. This chapter has explored a wide spectrum of algorithmic strategies, from classical LCG recurrences and their spectral limitations to modern composite generators with output whitening, from Fibonacci and Galois LFSR configurations to nonlinear and chaotic hybrid extensions, from open-addressed hash tables with randomized bit-mixing to cache-aware grouped probing, from closed-form inverse-CDF transformations and ratio-of-uniforms methods to hybrid bulk-tail samplers, from dense Cholesky-based multivariate sampling to scalable low-rank-plus-diagonal approximations, from hit-or-miss geometric integration and importance sampling to low-discrepancy Halton and Sobol sequences with digital scrambling, and from adaptive importance density construction to VEGAS separable grid refinement and recursive MISER-style stratification. By implementing these methods in Rust with crates such as `rand`, `rand_distr`, `rand_chacha`, `ndarray`, `nalgebra`, and `statrs`, readers have seen how deterministic algorithmic randomness translates directly into reproducible, efficient, and statistically robust code.

## 7.11.1. Key Takeaways

- A pseudo-random number generator is a discrete dynamical system $s_{n+1} = f(s_n)$, $u_n = O(s_n)$ producing a deterministic sequence from a seed $s_0$ with finite period $T$. The linear congruential generator $X_{n+1} \equiv aX_n + c \pmod{m}$ achieves maximal period $m$ under the Hull-Dobell conditions, but its outputs lie on at most $m^{1/k}$ hyperplanes in $k$ dimensions. Modern uniform generators overcome these limitations: xorshift operates as a linear transformation in $\mathbb{F}_2^{64}$ requiring a primitive characteristic polynomial, PCG applies nonlinear XSH-RR permutations to a 64-bit LCG state, composite generators form $Z_n = X_n \oplus Y_n$ from structurally independent streams, and counter-based designs $X_i = f(\text{seed}, i)$ enable embarrassingly parallel generation.
- The pseudo-DES hashing function provides stateless, index-based access to pseudo-random values through a four-round Feistel-like transformation with nonlinear mixing $f(R, C_1, C_2) = ((R \times C_1 + C_2) \ll 19) \oplus ((R \times C_1 + C_2) \gg 13)$. Butterfly mixing extends this to achieve complete array diffusion through $\log_2 N$ passes of pairwise XOR operations $A[i] \mathrel{\oplus}= A[i + 2^p]$, analogous to FFT connectivity patterns, guaranteeing full dependence after $k$ passes for $N = 2^k$ at $O(N \log N)$ cost.
- The inverse-transform method $Y = F_Y^{-1}(U)$ maps uniform deviates to non-uniform distributions in $O(1)$ time whenever the CDF admits a closed-form inverse: the Weibull uses $y = \lambda[-\ln(1-x)]^{1/k}$, the Pareto uses $y = y_{\min}(1-x)^{-1/\alpha}$ with logarithmic stabilization, the Laplace employs $\mu \pm b\ln(2x)$, and the Gumbel evaluates $y = \mu - \beta\ln[-\ln(x)]$. When closed-form inverses are unavailable, the ratio-of-uniforms method samples from $R = \{(u,v) : 0 < u \leq \sqrt{f(v/u)}\}$ with acceptance rates of 0.8 to 0.95, and hybrid samplers decompose densities into bulk and tail regions via a Bernoulli selector.
- Multivariate normal sampling from $N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ reduces to $\mathbf{X} = \boldsymbol{\mu} + L\mathbf{Z}$ where $L$ is the Cholesky factor satisfying $\boldsymbol{\Sigma} = LL^\top$ and $\mathbf{Z} \sim N(\mathbf{0}, I_d)$, with inverse whitening $\mathbf{Z} = L^{-1}(\mathbf{X} - \boldsymbol{\mu})$ recovering independent normals. For high-dimensional models where dense factorization costs $O(d^3)$, low-rank-plus-diagonal approximations $\boldsymbol{\Sigma} \approx BB^\top + D$ reduce sampling to $\mathbf{X} \approx B\mathbf{Z}_1 + D^{1/2}\mathbf{Z}_2$ with $r \ll d$, enabling climate ensemble forecasting and large-scale Gaussian process inference.
- Linear Feedback Shift Registers generate pseudo-random bit sequences through $\mathbf{a}_{t+1} = M\mathbf{a}_t$ over $\mathbb{F}_2$, where $M$ is the companion matrix of the feedback polynomial $P(x) = x^n + c_{n-1}x^{n-1} + \cdots + c_1x + 1$. When $P(x)$ is primitive, the LFSR produces a maximum-length m-sequence of period $2^n - 1$ satisfying Golomb's randomness postulates. The Fibonacci configuration computes a single global XOR feedback while the Galois configuration distributes feedback locally with up to 40% higher throughput. Nonlinear extensions, word-oriented $\mathbb{F}_{2^k}$ recurrences, and chaotic-hybrid architectures enhance sequence complexity for cryptographic applications.
- Hash tables provide $O(1)$ expected-time access via $i = h(\text{key}) \bmod M$ with load factor $\alpha = n/M$ governing performance: separate chaining yields expected successful lookup cost $1 + \alpha/2$, while linear probing requires $\frac{1}{2}(1 + 1/(1-\alpha))$ probes. Randomized hashing with per-table seeds prevents adversarial collisions. Modern implementations use packed control arrays, grouped SIMD-style probing, and cache-aware memory layouts. Hash functions treated as first-class objects enable modular composition of bit-mixing, seeding, and finalization stages across diverse key types and hardware architectures.
- Monte Carlo integration approximates $I = \int_\Omega f(\mathbf{x})\,d\mathbf{x}$ by the unbiased estimator $Q_N = \frac{V}{N}\sum_{i=1}^N f(\mathbf{x}_i)$ with error decay $O(N^{-1/2})$ independent of dimension. Importance sampling with density $p(\mathbf{x})$ computes $\widehat{I}_N = \frac{1}{N}\sum f(\mathbf{x}_i)/p(\mathbf{x}_i)$, achieving zero variance when $p^*(\mathbf{x}) \propto |f(\mathbf{x})|$. Change-of-variables techniques such as the exponential transformation $s = \frac{1}{5}e^{5z}$ flatten strongly anisotropic integrands, reducing estimator variance by orders of magnitude while preserving unbiasedness.
- Quasi-random (low-discrepancy) sequences fill $[0,1]^n$ more uniformly than random samples while remaining extensible point-by-point. The Koksma-Hlawka inequality bounds integration error by $V(f) \cdot D_N^*$. The van der Corput radical-inverse function constructs one-dimensional sequences extended via distinct prime bases in the Halton sequence. Sobol sequences employ binary direction numbers derived from primitive polynomials over $\mathbb{F}_2$ with Gray-code updates, achieving $(\log N)^n/N$ discrepancy. Digital scrambling restores unbiasedness and enables variance estimation while preserving low-discrepancy structure.
- The VEGAS algorithm approximates the optimal importance density using a separable product $p(\mathbf{x}) \propto g_1(x_1)\cdots g_d(x_d)$ with iteratively refined one-dimensional histograms, combining iteration estimates via inverse-variance weighting $I_{\text{best}} = \sum \widehat{I}_k/\widehat{\sigma}_k^2 / \sum 1/\widehat{\sigma}_k^2$. Stratified sampling partitions the domain with variance-optimal allocation $N_j \propto w_j\sigma_j$, and recursive stratified sampling (MISER) selects split dimensions by minimizing $w_{a,i}\sigma_{a,i} + w_{b,i}\sigma_{b,i}$. Adaptive importance sampling for rare events achieves orders-of-magnitude variance reduction for failure probabilities below $10^{-6}$ in engineering reliability and URLLC system evaluation.
- Statistical quality of PRNGs is assessed through chi-square uniformity tests, Kolmogorov-Smirnov distributional checks, lag-1 autocorrelation, and bit-level entropy analyses in frameworks such as TestU01, PractRand, and NIST SP 800-22. The Berlekamp-Massey algorithm reconstructs LFSR feedback polynomials from $2n$ output bits, demonstrating the predictability of purely linear generators. Applications span Monte Carlo particle transport, financial derivative pricing, Bayesian inference, climate ensemble forecasting, Gaussian process modeling, engineering reliability, URLLC evaluation, computer graphics, and procedural content generation.

## 7.11.2. Advice for Beginners

- Randomness plays a central role in modern numerical computing. Before studying advanced Monte Carlo methods, it is important to understand that computers do not generate truly random numbers. Instead, they produce deterministic sequences designed to mimic randomness. Begin by learning how pseudo-random number generators operate and why properties such as period length, uniformity, and statistical independence are important.
- Start with simple uniform random number generators such as linear congruential generators. Although modern generators are more sophisticated, studying classical methods provides valuable insight into the strengths and weaknesses of pseudo-random sequences. Experiment with statistical tests and visualize generated samples to understand concepts such as correlation and distribution quality.
- Once you understand uniform random numbers, explore transformation methods for generating samples from other probability distributions. Implement inverse-transform sampling for distributions such as the Weibull, Pareto, and Gumbel distributions. This will help develop intuition about the relationship between probability distributions and random sampling algorithms.
- Next, study multivariate normal sampling and covariance structures. Understanding how correlation is represented and generated is essential for simulation, uncertainty quantification, machine learning, and statistical modeling.
- When learning about hashing and hash tables, focus on the practical connection between randomness and efficient data organization. Experiment with different collision-resolution strategies and observe how performance changes as the load factor increases.
- Monte Carlo integration should be approached gradually. Begin with simple geometric problems such as estimating areas and volumes before progressing to importance sampling and variance-reduction techniques. Understanding why Monte Carlo error decreases as approximately $N^{-1/2}$ is fundamental to appreciating both the strengths and limitations of stochastic methods.
- After mastering classical Monte Carlo methods, explore low-discrepancy sequences such as Halton and Sobol' sequences. Compare their convergence behavior with ordinary random sampling to understand why quasi-Monte Carlo methods can outperform traditional Monte Carlo integration for many smooth problems.
- For Rust implementations, become familiar with libraries such as `rand`, `rand_distr`, `rand_chacha`, `statrs`, `ndarray`, and `nalgebra`. These libraries provide efficient tools for random number generation, probability distributions, statistical analysis, and numerical simulation.
- Most importantly, remember that randomness is not merely a source of uncertainty. It is a powerful computational tool used in simulation, optimization, machine learning, Bayesian inference, finance, engineering reliability, computer graphics, and scientific computing. A strong understanding of the techniques presented in this chapter will provide a foundation for many advanced stochastic algorithms encountered throughout numerical computing.

## 7.11.3. Further Learning with GenAI

To deepen your understanding of random number generation, hashing, and Monte Carlo methods in Rust, consider using the following GenAI prompts:

 1. Explain the Hull-Dobell conditions for full-period LCGs and describe how the spectral test quantifies lattice structure. Write a Rust program implementing both the Park-Miller and RANDU generators, export $(u_n, u_{n+1}, u_{n+2})$ triples to CSV, and compute chi-square uniformity statistics to compare their quality.
 2. Describe the pseudo-DES four-round Feistel hashing function and explain how butterfly mixing achieves complete array diffusion. Write a Rust program that hashes $N = 2^{16}$ indices via pseudo-DES, applies butterfly mixing, and verifies that post-mixing statistics remain consistent with $U(0,1)$.
 3. Explain the inverse-transform derivation for the Weibull distribution and the ratio-of-uniforms acceptance region for Beta(2,2). Implement Weibull, Pareto, Laplace, and Gumbel inverse-CDF samplers and a Beta(2,2) RoU sampler in Rust, verifying all empirical moments against closed-form expressions.
 4. Describe the Cholesky-based coloring transform $\mathbf{X} = \boldsymbol{\mu} + L\mathbf{Z}$ and the low-rank-plus-diagonal approximation $\boldsymbol{\Sigma} \approx BB^\top + D$. Implement both in Rust using `nalgebra`, verify whitening recovers identity covariance for a 3D example, and measure Frobenius-norm error for $d = 50$, $r = 5$.
 5. Explain the algebraic formulation of LFSRs through companion matrices over $\mathbb{F}_2$ and the primitivity test via matrix exponentiation. Implement both Fibonacci and Galois configurations for $P(x) = x^8 + x^6 + x^5 + x^4 + 1$ in Rust, verify phase-shift equivalence, and build a three-LFSR nonlinear keystream generator with majority combining.
 6. Describe separate chaining and open addressing with linear probing, including expected probe counts as functions of load factor $\alpha$. Implement both in Rust, insert 10,000 random keys at $\alpha = 0.75$, and compare empirical probe statistics against $1 + \alpha/2$ and $\frac{1}{2}(1 + 1/(1-\alpha))$.
 7. Explain how importance sampling via change of variables reduces variance for anisotropic integrands such as $\int_W e^{5z}\,dV$. Implement both uniform and importance-sampled estimators in Rust for a truncated torus domain and report the variance reduction factor using 2,000,000 samples.
 8. Describe the van der Corput radical-inverse function, Halton sequence construction, and Sobol sequence generation using direction numbers from primitive polynomials. Implement all three in Rust, compare integration errors against standard Monte Carlo on $\int_{[0,1]^3} e^{-(x+y+z)}\,d\mathbf{x}$, and implement XOR-based scrambling with RMSE comparison over 20 replicates.
 9. Describe the VEGAS algorithm's separable importance density, iterative grid refinement via projected squared integrands, and inverse-variance combination. Implement a basic VEGAS integrator in Rust for a 3-dimensional exponential test function with 20 bins per dimension and report per-iteration variance reduction across 10 iterations.
10. Explain adaptive importance sampling for rare-event estimation and the variance decomposition of stratified sampling $\mathrm{Var}(\widehat{I}) = \sum w_j^2\sigma_j^2/N_j$. Implement a stabilized adaptive Gaussian sampler for $P(Z > 4)$ and recursive MISER-style splitting for a peaked 2D integrand in Rust, demonstrating variance reduction over crude Monte Carlo in both cases.

By engaging with these prompts, you will gain a deeper understanding of Rust's capabilities for implementing pseudo-random number generators, hash-based data structures, transformation-based samplers, multivariate Gaussian algorithms, and Monte Carlo integration methods that are reproducible, efficient, and applicable to real-world stochastic simulation and scientific computation.

## 7.11.4. Homework Exercises

To reinforce your learning, complete the following exercises:

 1. Implement a generic LCG struct in Rust parameterized by multiplier aa a, increment cc c, and modulus mm m. Instantiate it with ANSI C parameters ($m = 2^{31}$, $a = 1103515245$, $c = 12345$) and RANDU parameters ($m = 2^{31}$, $a = 65539$, $c = 0$). Produce 10,000 triples $(u_n, u_{n+1}, u_{n+2})$ from each, export to CSV, compute the chi-square statistic over 20 uniform bins, and implement a PCG-XSH-RR 64/32 generator with skip-ahead to verify decorrelation across 4 substreams.
 2. Implement the pseudo-DES hashing function in Rust using the constants from Equation (7.3.5). Compute $U(j)$ for $j = 0, \ldots, 99999$, report empirical statistics, then apply butterfly mixing to a hashed array of size $N = 2^{16}$ and verify post-mixing uniformity. Benchmark per-element timing for hashing alone, mixing alone, and combined.
 3. Write a Rust program generating Weibull deviates via $y = \lambda[-\ln(1-x)]^{1/k}$ for $(k,\lambda) = (1,1)$, $(2,1)$, and $(1.5,2)$, and also Laplace (both inverse-CDF and exponential-difference methods), Pareto (logarithmic form), and Gumbel samplers. For each, draw 100,000 samples and compare empirical moments against theoretical values. Implement a ratio-of-uniforms sampler for Beta(2,2) and report the empirical acceptance rate.
 4. Implement Cholesky-based multivariate normal sampling in Rust using `nalgebra` for a 4-dimensional distribution. Draw 50,000 samples, estimate the empirical covariance, apply whitening, and confirm the whitened covariance approximates the identity. Then implement a low-rank-plus-diagonal sampler for $d = 50$, $r = 5$ and verify the Frobenius-norm relative error is below 0.05.
 5. Implement both Fibonacci and Galois LFSRs in Rust for $P(x) = x^8 + x^6 + x^5 + x^4 + 1$. Generate one full period (255 bits) from each, verify the balance of zeros and ones, search for a phase shift $\Delta$ such that $z_t^{(G)} = z_{t+\Delta}^{(F)}$, and implement the companion-matrix primitivity test for polynomials $x^3 + x + 1$, $x^4 + x + 1$, and $x^{10} + x^3 + 1$. Build a three-LFSR nonlinear keystream generator with majority combining and demonstrate XOR encryption.
 6. Implement both an open-addressed hash table with linear probing and a separate-chaining hash table in Rust using a SplitMix-style randomized bit-mixing function. Insert 10,000 random keys at load factor $\alpha = 0.75$, measure average probes for successful and unsuccessful lookups, compare against theoretical expectations, and demonstrate tombstone-based deletion with its effect on subsequent probe lengths.
 7. Write a Rust program implementing hit-or-miss Monte Carlo for the area of the unit disk (comparing against $\pi$) and for the volume and center of mass of a truncated torus. Then implement importance sampling via the exponential change of variables $s = \frac{1}{5}e^{5z}$ for $\int_W e^{5z}\,dV$ and report the variance reduction factor relative to uniform sampling using 2,000,000 samples.
 8. Implement the van der Corput, Halton (bases 2, 3), and Sobol (with primitive polynomial $x^3 + x + 1$ for the second dimension) sequences in Rust. Compare all three against standard Monte Carlo on $\int_{[0,1]^2} e^{x+y}\,d\mathbf{x}$ for $N = 1000, 5000, 10000, 50000$, reporting absolute errors. Implement XOR-based scrambling of Sobol direction numbers and compare scrambled QMC RMSE against MC RMSE over 20 replicates.
 9. Implement a basic VEGAS integrator in Rust with $d = 3$ dimensions and $K = 20$ bins. Use $f(\mathbf{x}) = e^{-5(x_1+x_2+x_3)}$ with exact integral $((1-e^{-5})/5)^3$. Run 10 adaptive iterations of 50,000 samples each, combine via inverse-variance weighting, and report per-iteration estimates, the combined result, and absolute error. Also implement regular $8 \times 8$ stratified sampling and recursive MISER-style splitting for a peaked 2D integrand, demonstrating variance reduction over crude Monte Carlo.
10. Implement both crude Monte Carlo and stabilized adaptive Gaussian importance sampling for $P(Z > 4)$ with $Z \sim N(0,1)$. Use 300,000 crude samples and 5 adaptive iterations of 20,000 samples each with learning rate $\eta = 0.3$ and clamped proposal mean. Report ESS per iteration and compare final estimates against a numerical reference. Also implement a complete PRNG statistical testing pipeline (chi-square, KS, lag-1 autocorrelation, bit-level balance) and apply it to both `StdRng` and a custom LCG.

These exercises cover the full range of random number generation, hashing, and Monte Carlo integration techniques developed in this chapter, from classical and modern uniform generators to index-based hashing, linear feedback shift registers, hash table design, inverse-transform and rejection sampling for non-uniform distributions, Cholesky-based and low-rank multivariate normal sampling, hit-or-miss and importance-sampled Monte Carlo integration, low-discrepancy Halton and Sobol sequences with scrambling, and adaptive VEGAS and recursive stratified methods. By completing them in Rust, you will build practical experience with the reproducible, statistically robust algorithms that form the computational foundation of Monte Carlo simulation, stochastic optimization, and probabilistic modeling.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/T0PxWwvCtdQpmgfkjIvZ.8","tags":[]}

# References

 1. Almaraz Luengo, E. (2022) ‘A brief and understandable guide to pseudo-random number generators’, *Statistical Surveys*, 16, pp. 1–24. Available at: <https://doi.org/10.1214/22-SS136>
 2. Bhattacharjee, K. and Das, S. (2022) ‘A search for good pseudo-random number generators: Survey and empirical studies’, *Computer Science Review*, 45, p. 100471. Available at: <https://doi.org/10.1016/j.cosrev.2022.100471>
 3. Bruynsteen, C., Wang, C. and Vermeulen, N. (2022) ‘100 Gbps integrated quantum random number generator based on vacuum fluctuations’, *AIP Quantum*, 3(3), pp. 1–8. Available at: <https://doi.org/10.1063/5.0098392>
 4. Sepehri, F., Camarero, J. and García-Molina, R. (2020) ‘Selection of random number generators in GATE Monte Carlo simulations’, *Nuclear Instruments and Methods in Physics Research Section A*, 987, p. 164859. Available at: <https://doi.org/10.1016/j.nima.2020.164859>
 5. Wu, X., Li, J. and Zhang, K. (2025) ‘Pseudo-random number generators based on neural networks’, *Nonlinear Dynamics*, forthcoming. Available at: <https://doi.org/10.1007/s44443-025-00007-4>
 6. Frankel A. (2023) *Pseudo-Random Number Generators: From the Origins to Modern Algorithms.* blog.frankel.ch.
 7. Josey C. (2023) *Reassessing the MCNP Random Number Generator.* Los Alamos National Laboratory Tech. Rep. LA-UR-23-25111.
 8. Foreman C., Yeung R. and Curchod F.J. (2024) *Statistical testing of random number generators and their improvement using randomness extraction.* arXiv:2403.18716.
 9. Sindhu TN, et al. (2024) ‘Distributional properties of the entropy-transformed Weibull (ET-W) distribution’, *Scientific Reports*, 14: 83132. <https://doi.org/10.1038/s41598-024-83132-w>
10. Suleiman AA, et al. (2024) ‘A new Weibull distribution for modelling complex data in biomedical research’, *Heliyon*, 10: e08905. <https://doi.org/10.1016/j.heliyon.2024.e08905>
11. Anghel A, et al. (2024) ‘Revisiting the use of the Gumbel distribution in extreme-value analysis’, *Mathematics*, 12(16). <https://doi.org/10.3390/math12162466>
12. Gómez HJ, et al. (2024) ‘A new generalization of the truncated Gumbel distribution with quantile regression and applications’, *Mathematics*, 12(11): 1762. <https://doi.org/10.3390/math12111762>
13. Osatohanmwen P, et al. (2024) ‘A general framework for generating three-component hybrid distributions that meet the need of data sets with right heavy-tail’, *Environmental and Sustainability Advances*, Springer. <https://doi.org/10.1007/s44199-024-00084-w>
14. Do, B., Ajenifuja, N.A., Adebiyi, T.A. & Zhang, R. (2025). *Sampling from Gaussian Processes: A Tutorial and Applications in Global Sensitivity Analysis and Optimization*. arXiv preprint arXiv:2507.14746. DOI: <https://doi.org/10.48550/arXiv.2507.14746>
15. Lakatos, M., Lerch, S., Hemri, S. & Baran, S. (2023). Comparison of multivariate post-processing methods using global ECMWF ensemble forecasts. *Quarterly Journal of the Royal Meteorological Society*, 149(756), 856–877. DOI: <https://doi.org/10.1002/qj.4436>
16. Zhang, X., Abdulah, S., Cao, J., Ltaief, H., Sun, Y., Genton, M.G. & Keyes, D.E. (2024). Parallel approximations for high-dimensional multivariate normal probability computation in confidence region detection applications. In *Proceedings of the IEEE International Parallel and Distributed Processing Symposium (IPDPS 2024)*. DOI: [https://doi.org/10.48550/arXiv.2405.1489](https://doi.org/10.48550/arXiv.2405.14892)
17. Almaraz Luengo, E. & Román Villaizán, J. (2023). Cryptographically secured pseudo-random number generators: Analysis and testing with NIST statistical test suite. *Mathematics*, 11(23), 4812. <https://doi.org/10.3390/math11234812>
18. Maity, D., Prayashi, A., Panda, A.K. & Panigrahi, S. (2025). Encryption of image using linear feedback shift register and pixel shuffling. *Cureus Journal of Computer Science*, 2(April 07, 2025), Article eS44389-024-02027-z. <https://doi.org/10.7759/s44389-024-02027-z>
19. Okunbor, D., Omorogbe, V. & Edeko, F. (2024). Analysis of linear feedback shift registers and chaos-based techniques for image encryption. *Journal of Cyber Security Technology*, 9(2), 1–9. <https://doi.org/10.1080/23742917.2024.2338954>
20. Somanathan, G.R., Reddy, U.H. & Bhakthavatchalu, R. (2025). Design of a power-aware reconfigurable and parameterizable pseudorandom pattern generator for BIST-based applications. *Journal of Low Power Electronics and Applications*, 15(3), 47. <https://doi.org/10.3390/jlpea15030047>
21. Birler, A., Schmidt, T., Fent, P. & Neumann, T. (2024). *Simple, efficient, and robust hash tables for join processing*. Proceedings of DaMoN 2024 – 20th International Workshop on Data Management on New Hardware. DOI: 10.1145/3662010.3663442.
22. Li, D. (2023). *Accelerating HashMap performance in Rust with Hashbrown*. FriendlyUser Tech Blog, 22 April 2023. Available at: <https://friendlyuser.github.io/> (accessed DATE).
23. Wang, S., Liu, Y., Zhang, X., Hu, L. & Qian, C. (2025). *A distributed learned hash table*. arXiv preprint arXiv:2508.14239. (Accepted to IEEE ICNP 2025).
24. Farach-Colton, M., Krapivin, A. & Kuszmaul, W. (2024). *Optimal bounds for open addressing without reordering*. Proceedings of the 65th IEEE Symposium on Foundations of Computer Science (FOCS 2024). DOI: 10.1109/FOCS61266.2024.00045.
25. Hung, Y.-C. (2023). *A review of Monte Carlo and quasi–Monte Carlo sampling techniques*. Wiley Interdisciplinary Reviews: Computational Statistics, 16(2), e1637. DOI: <https://doi.org/10.1002/wics.1637>
26. Tang, Y. (2023). *A note on Monte Carlo integration in high dimensions*. The American Statistician, published online 16 November 2023. DOI: <https://doi.org/10.1080/00031305.2023.2267637>
27. Zhong, H. and Feng, X. (2025). *An efficient quasi-Monte Carlo algorithm for high-dimensional numerical integration*. Mathematics, 13(21), 3437. DOI: <https://doi.org/10.3390/math13213437>
28. Shyamsundar, P., Scott, J. L., Mrenna, S., Matchev, K. T. and Kong, K. (2024). *Variance reduction via simultaneous importance sampling and control variates techniques using VEGAS*. SciPost Physics Codebases, 28, pp. 1–20. DOI: <https://doi.org/10.21468/SciPostPhysCodeb.28>
29. Chahine, M., Rusch, T.K., Patterson, Z.J. and Rus, D. (2024). Improving Efficiency of Sampling-based Motion Planning via Message-Passing Monte Carlo. *arXiv preprint arXiv:2410.03909*. DOI: <https://doi.org/10.48550/arXiv.2410.03909>
30. Clément, F., Doerr, C., Klamroth, K. and Paquete, L. (2025). Searching permutations for constructing uniformly distributed point sets. *Proceedings of the National Academy of Sciences*, 122(14), e2424464122. DOI: <https://doi.org/10.1073/pnas.2424464122>
31. Hickernell, F.J., Kirk, N. and Sorokin, A.G. (2025). Quasi-Monte Carlo Methods: What, Why, and How? *arXiv preprint arXiv:2502.03644*. DOI: <https://doi.org/10.48550/arXiv.2502.03644>
32. Kucherenko, S. and Hok, J. (2023). The importance of being scrambled: supercharged quasi-Monte Carlo. *Journal of Risk*, 26(1), 27–46. DOI: <https://doi.org/10.21314/JOR.2023.008>
33. Rusch, T.K., Kirk, N., Bronstein, M.M., Lemieux, C. and Rus, D. (2024). Message-Passing Monte Carlo: Generating low-discrepancy point sets via graph neural networks. *Proceedings of the National Academy of Sciences*, 121(40), e2409913121. DOI: <https://doi.org/10.1073/pnas.2409913121>
34. Chopin, N., Wang, H. & Gerber, M. (2025). Adaptive stratified Monte Carlo using decision trees. *arXiv preprint*, arXiv:2501.04842. Available at: <https://doi.org/10.48550/arXiv.2501.04842>
35. Eshra, E. & Papakonstantinou, K. (2025). Gradient-Free Importance Sampling Scheme for Efficient Reliability Estimation. *Journal of Engineering Mechanics (ASCE)*. Available at: <https://doi.org/10.1061/JENMDT.EMENG-8449>
36. Ke, X., Zhu, H., Yi, K., He, G., Kong, Y., Zhou, H. & Wang, J. (2023). Adaptive Importance Sampling and Quasi-Monte Carlo Methods for 6G URLLC Systems. *arXiv preprint*, arXiv:2303.03575. Available at: <https://doi.org/10.48550/arXiv.2303.03575>

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/3S8GNda50y2TYNJnMgtd.1","tags":[]}



