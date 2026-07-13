---
weight: 3600
title: "Chapter 14"
description: "Statistical Description of Data"
icon: "article"
date: "2026-07-06T00:00:00+07:00"
lastmod: "2026-07-06T00:00:00+07:00"
katex: true
draft: false
toc: true
---

{{% alert icon="💡" context="info" %}}
<strong>"<em>Data are just summaries of thousands of stories</em>" — tell a few of those stories to help make the data meaningful." — Chip & Dan Heath</strong>
{{% /alert %}}

{{% alert icon="📘" context="success" %}}
<p style="text-align: justify;"><em>Chapter 14 introduces the fundamental concepts and computational methods of statistical analysis used in scientific computing, data science, and engineering. The chapter begins with descriptive statistics and moment-based characterization of distributions, then develops hypothesis testing, distribution comparison, contingency table analysis, correlation measures, and information-theoretic methods. It further explores multivariate distribution testing and Savitzky-Golay smoothing for noisy data. Throughout the chapter, emphasis is placed on numerical stability, robustness, uncertainty quantification, and practical interpretation of statistical results. Rust implementations demonstrate how modern statistical algorithms can be implemented efficiently and reliably, providing a foundation for data-driven modeling, inference, machine learning, and scientific decision-making.</em></p>
{{% /alert %}}

# 14.1. Introduction

This section establishes the conceptual foundation for the chapter by emphasizing that data in numerical computing are not merely collections of numbers, but structured outputs arising from measurements or simulations. Such data inherently carry provenance, uncertainty, and often latent structure that must be accounted for in analysis. In modern computational settings, datasets are frequently massive, streamed, or distributed, requiring algorithms that are not only statistically meaningful but also numerically stable under finite-precision arithmetic (Li et al., 2024; Grayson et al., 2025; Boldo et al., 2023).

From a mathematical perspective, a dataset is modeled as samples drawn from one or more random variables. In the standard numerical computing formulation, observations of $d$-dimensional features are organized into a data matrix:

$$
X =
\begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1d} \\
x_{21} & x_{22} & \cdots & x_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
x_{N1} & x_{N2} & \cdots & x_{Nd}
\end{pmatrix}
\in \mathbb{R}^{N \times d},
\qquad
x_i = (x_{i1}, \dots, x_{id})^\top \in \mathbb{R}^d
\tag{14.1.1}
$$

The primary descriptive statistics associated with this representation are the column mean vector and the sample covariance matrix, defined by:

$$
\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i \in \mathbb{R}^d
\tag{14.1.2}
$$

$$
S = \frac{1}{N - 1} \sum_{i=1}^{N} (x_i - \bar{x})(x_i - \bar{x})^\top \in \mathbb{R}^{d \times d}
\tag{14.1.3}
$$

These quantities form the computational backbone of uncertainty quantification, model diagnostics, convergence monitoring, and a wide range of statistical procedures embedded within scientific computing workflows (Stanley et al., 2024; Li et al., 2024).

In numerical computing, randomness arises even when the governing equations are deterministic. This occurs for several recurring reasons. First, in simulation ensembles and uncertainty quantification, models such as partial differential equations often depend on uncertain inputs, including initial conditions, boundary conditions, or material parameters. As a result, ensembles of simulations are generated and summarized using statistical measures such as means, variances, and quantiles. In high-dimensional settings, including variational data assimilation, Monte Carlo methods are routinely employed to characterize posterior uncertainty in large-scale inverse problems (Stanley et al., 2024).

Second, in optimization and machine learning, stochastic methods treat gradients and loss functions as random variables whose expectations are estimated from sampled data. In such contexts, heavy-tailed distributions, characterized by rare but extreme observations, can invalidate naïve estimators. This motivates the development of robust statistical techniques and distributionally robust optimization frameworks that explicitly account for tail behavior (Li et al., 2024; Van Parys and Zwart, 2025).

Third, practical constraints related to streaming data and storage limitations play a central role. In domains such as climate and Earth-system simulation, storing full-resolution outputs is often infeasible. Consequently, statistical summaries must be computed on-the-fly using one-pass algorithms. In such settings, computing quantities like means and variances is not merely an implementation detail but a fundamental requirement for scientific analysis (Grayson et al., 2025).

A related aspect of statistical analysis concerns hypothesis testing, particularly the interpretation of tail probabilities. Classical approaches involve selecting a test statistic, deriving its distribution under a null hypothesis, and computing a p-value. However, modern statistical practice emphasizes that p-values do not constitute evidence in favor of the null hypothesis; rather, they measure the degree of incompatibility between observed data and the assumed model. Contemporary methodology therefore advocates for robust testing procedures that are less sensitive to violations of assumptions, and for reporting effect sizes and uncertainty intervals instead of relying solely on binary decisions based on significance thresholds (Höfler, 2026).

In summary, this introduction frames statistical description as an essential component of numerical computing. Data must be interpreted through a lens that integrates probabilistic modeling, computational constraints, and numerical stability. The remainder of the chapter builds on this perspective, developing tools that are both statistically principled and computationally robust.

# 14.2. Moments of a Distribution: Mean, Variance, Skewness, and So Forth

This section develops the statistical descriptors introduced in Section 14.1 into a more formal and computationally grounded framework. While classical treatments present moments primarily as algebraic summaries, modern numerical computing requires a broader perspective that integrates probabilistic validity, numerical stability, and algorithmic scalability. In particular, moments must now be understood not only as theoretical quantities defined through expectations, but also as quantities that must be computed reliably from finite, possibly imperfect data under constraints of precision, memory, and parallelism.

A key extension beyond the classical viewpoint concerns the existence of moments. In many real-world applications, especially those involving financial returns, network traffic, or stochastic physical systems, the underlying distributions may be heavy-tailed. In such cases, higher-order moments may either fail to exist or exist only in a weak theoretical sense while remaining practically unusable due to extreme variability. This raises fundamental questions: when is a moment meaningful, and when does its computation introduce more noise than information? Addressing these questions requires a careful interplay between probability theory and numerical analysis, rather than relying solely on formal definitions.

Equally important is the issue of numerical stability. Even when moments exist mathematically, their computation can be highly sensitive to rounding errors and data ordering. For instance, evaluating central moments involves subtracting nearly equal quantities, which can lead to catastrophic cancellation in floating-point arithmetic. Similarly, high-order moments amplify large values through exponentiation, increasing the risk of overflow or disproportionate influence from outliers. As a result, modern treatments emphasize reformulations that improve conditioning, such as incremental updates, compensated summation, and recentering strategies.

Another essential dimension is the need for robust alternatives. Classical moments such as the mean and variance are optimal under idealized assumptions, particularly for light-tailed distributions like the normal distribution. However, in the presence of outliers or heavy tails, these quantities can become unstable or misleading. Contemporary approaches therefore incorporate robust estimators, including trimmed moments, median-based measures, and M-estimators, which provide more reliable summaries under non-ideal conditions. These alternatives are not merely statistical refinements but are often necessary for ensuring numerical reliability in large-scale computations.

Finally, the computational context itself has evolved. Many modern applications operate in streaming or distributed environments, where data arrive sequentially or are partitioned across multiple processors. In such settings, moment computations must be performed incrementally, with limited storage and minimal communication overhead. This has led to the development of algorithms that support one-pass updates, parallel aggregation, and reproducibility across heterogeneous systems. The design of such algorithms requires a careful balance between statistical accuracy and computational efficiency.

These considerations are central in contemporary large-scale scientific workflows, where statistical estimation and numerical computation are tightly coupled (Minsker, 2025; Liao and Domański, 2025; Boldo et al., 2023; Grayson et al., 2025; Li et al., 2024).

## 14.2.1. Population Moments and Their Limitations

Let $X$ be a real-valued random variable with distribution $P$. The raw moment of order $k$ is defined as:

$$
m_k = \mathbb{E}[X^k], \quad \text{(if finite)}
\tag{14.2.1}
$$

The mean is,

$$
\mu = m_1
\tag{14.2.2}
$$

and the central moment of order $k$ is,

$$
\mu_k = \mathbb{E}[(X - \mu)^k], \quad \text{(if finite)}
\tag{14.2.3}
$$

The variance is:

$$\sigma^2 = \mu_2 \tag{14.2.4}$$

\
while skewness and kurtosis are defined by:

$$
\gamma_1 = \frac{\mu_3}{\mu_2^{3/2}}, \qquad \gamma_2 = \frac{\mu_4}{\mu_2^{2}} - 3
\tag{14.2.5}
$$

Moments can be interpreted as coefficients in a polynomial expansion of the distribution. In particular, when the moment-generating function exists, moments characterize the distribution locally. In many classical settings, the full sequence of moments uniquely determines the distribution. However, from a numerical standpoint, this characterization can be ill-conditioned: small perturbations in higher-order moments may correspond to large changes in the underlying distribution.

Moreover, moments are closely tied to orthogonal polynomial systems and spectral representations, which appear throughout numerical analysis, including quadrature methods and approximation theory. This connection explains why moment computation is fundamental in algorithms such as Gaussian quadrature and Chebyshev approximation.

The qualifier “if finite” is essential and is frequently understated in elementary discussions of moments. The definition of a moment implicitly assumes that the corresponding expectation exists, yet in many practically relevant settings this assumption fails. Distributions encountered in applications such as finance, network traffic analysis, and stochastic simulation often exhibit heavy tails, in which the probability of extreme values decays slowly. In such cases, it is entirely possible for a distribution to possess a finite mean while having an infinite variance, rendering variance-based analyses unreliable. Even when higher-order moments are theoretically finite, they may be dominated by rare but extreme observations, leading to numerical instability and poor interpretability. As a consequence, estimators of skewness and kurtosis can display substantial variability in finite samples, limiting their usefulness for inference and modeling.

These difficulties are particularly pronounced for heavy-tailed distributions, where extreme events exert a disproportionate influence on moment-based summaries. In such regimes, classical descriptors such as variance, skewness, and kurtosis may fail to provide stable or meaningful characterizations of the data. This highlights an important conceptual point: the existence of a moment in a theoretical sense does not guarantee its practical utility in numerical computation or statistical estimation.

From a computational perspective, the evaluation of moments introduces additional challenges. Moments are typically computed as sums of powers of the data, and this structure is inherently sensitive to floating-point effects. Central moment calculations involve subtracting quantities that may be nearly equal, which can result in catastrophic cancellation and significant loss of precision. Higher-order moments further amplify numerical difficulties, as exponentiation increases the dynamic range of the quantities involved, raising the risk of overflow or underflow. Moreover, combining large and small terms in a single summation can degrade numerical accuracy, especially when standard floating-point accumulation is used.

These considerations motivate the development of numerically stable formulations for moment computation. Techniques such as recentered accumulations, compensated summation, and incremental or streaming updates are designed to reduce rounding errors and improve robustness in finite-precision arithmetic. Such methods are essential in modern computational environments, where data sets are large and computations must often be performed in a single pass or across distributed systems.

Because of these limitations, contemporary statistical and numerical practice increasingly supplements or replaces classical moment-based summaries with more robust alternatives (Minsker, 2025; Liao and Domański, 2025). Measures such as the median and interquartile range provide stable descriptions of location and spread that are less sensitive to extreme observations. Trimmed and winsorized moments reduce the influence of outliers by modifying the tails of the data, while M-estimators and influence-function-based approaches offer a principled framework for robust estimation. These alternatives are particularly well suited to large-scale and streaming settings, where data irregularities and outliers are unavoidable, and where numerical stability is as important as statistical efficiency.

## 14.2.2. Sample Estimators and Uncertainty Quantification

Given observations $x_1, \dots, x_N$, the standard estimators are:

$$
\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i
\tag{14.2.6}
$$

$$
s^2 = \frac{1}{N - 1} \sum_{i=1}^{N} (x_i - \bar{x})^2, \qquad s = \sqrt{s^2}
\tag{14.2.7}
$$

Under mild assumptions, uncertainty in $\bar{x}$ is summarized using standard errors or confidence intervals. However, in modern applications involving non-Gaussian or heavy-tailed data, classical formulas for sampling variability may be unreliable. Resampling methods such as the bootstrap and robust estimators are therefore widely used, as they adapt to the data’s tail behavior without relying on fragile parametric assumptions (Colantonio et al., 2024; Ma and Xu, 2024).

Even when higher moments exist, their sample estimates can exhibit substantial variability. In particular, skewness and kurtosis estimates are highly sensitive to finite-sample effects and non-Gaussianity, often displaying large dispersion. Recent studies emphasize that these quantities should be interpreted cautiously and supplemented with resampling-based uncertainty assessments when used in practice (Ma and Xu, 2024).

### Rust Implementation

Following the discussion in Section 14.2.2 on sample estimators and uncertainty quantification, Program 14.2.1 provides a practical implementation of descriptive statistics for finite datasets, including the sample mean (14.2.6), sample variance (14.2.7), and higher-order empirical moments. In modern numerical computing, these quantities must be evaluated not only correctly but also in a manner that is robust to floating-point effects and sensitive to data irregularities such as outliers. The program incorporates compensated summation to improve numerical stability and employs a two-pass formulation for variance computation. In addition, it implements a bootstrap procedure to estimate uncertainty in the sample mean, reflecting the subsection’s emphasis on resampling-based inference in non-Gaussian or heavy-tailed settings. This unified framework demonstrates how classical statistical estimators are extended into reliable computational tools under realistic numerical constraints.

At the core of the implementation is the `KahanSum` structure, which provides a compensated summation mechanism for floating-point accumulation. This directly addresses the numerical stability concerns highlighted in Section 14.2, where naive summation can lead to loss of precision due to rounding errors. By maintaining a correction term, the algorithm significantly improves the accuracy of summations used in computing the mean (14.2.6) and variance (14.2.7).

The function `sample_mean` implements the estimator in Equation (14.2.6) using compensated accumulation. The function `sample_variance` follows a two-pass strategy: it first computes the mean and then accumulates squared deviations, corresponding to Equation (14.2.7). This formulation avoids catastrophic cancellation that would arise from naive single-pass formulas such as $\sum x_i^2 - \bar{x}^2$.

The functions `central_moments_2_3_4`, `sample_skewness`, and `sample_excess_kurtosis` extend the computation to higher-order empirical moments, illustrating the discussion in Section 14.2.2 regarding the variability and instability of skewness and kurtosis in finite samples. These quantities are computed using central moments to ensure consistency with the definitions introduced earlier.

To address uncertainty quantification, the program implements a bootstrap procedure via the function `bootstrap_mean_confidence_interval`. This function repeatedly resamples the dataset with replacement and computes the sample mean for each resample. The resulting empirical distribution is then used to construct a percentile-based confidence interval, reflecting modern practice where classical parametric assumptions may be unreliable. A simple pseudo-random number generator (`XorShift64`) is included to ensure that the program is fully self-contained and does not rely on external dependencies. This design choice aligns with the goal of providing reproducible and portable numerical implementations.

The `main` function demonstrates the complete workflow. It initializes a dataset, computes descriptive statistics, and evaluates a bootstrap confidence interval for the mean. The inclusion of an outlier in the dataset illustrates the sensitivity of higher-order moments and motivates the use of resampling-based uncertainty estimates, as emphasized in Section 14.2.2.

```rust
// Program 14.2.1. Sample Mean, Variance, Higher Moments, and Bootstrap Uncertainty
//
// Problem Statement:
// Given observations x_1, ..., x_N, compute the sample mean, sample variance,
// sample standard deviation, empirical skewness, and empirical excess kurtosis.
// In addition, estimate uncertainty in the sample mean by constructing a
// bootstrap percentile confidence interval. The implementation uses compensated
// summation and a two-pass variance computation to improve numerical reliability.

use std::cmp::Ordering;

// -----------------------------------------------------------------------------
// Compensated summation for improved floating-point accumulation.
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, Copy)]
struct KahanSum {
    sum: f64,
    c: f64,
}

impl KahanSum {
    fn new() -> Self {
        Self { sum: 0.0, c: 0.0 }
    }

    fn add(&mut self, value: f64) {
        let y = value - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    fn total(self) -> f64 {
        self.sum
    }
}

// -----------------------------------------------------------------------------
// A small pseudo-random number generator so the program runs without external
// crates. This is adequate for demonstration and bootstrap resampling.
// -----------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn gen_index(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "upper must be positive");
        (self.next_u64() as usize) % upper
    }
}

// -----------------------------------------------------------------------------
// Descriptive statistics.
// -----------------------------------------------------------------------------
fn sample_mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let mut acc = KahanSum::new();
    for &x in data {
        acc.add(x);
    }
    Some(acc.total() / data.len() as f64)
}

fn sample_variance(data: &[f64]) -> Option<f64> {
    let n = data.len();
    if n < 2 {
        return None;
    }

    let mean = sample_mean(data)?;
    let mut acc = KahanSum::new();

    for &x in data {
        let dx = x - mean;
        acc.add(dx * dx);
    }

    Some(acc.total() / (n as f64 - 1.0))
}

fn sample_stddev(data: &[f64]) -> Option<f64> {
    sample_variance(data).map(f64::sqrt)
}

// Returns empirical central moments with denominator N:
// mu2 = (1/N) sum (x_i - mean)^2
// mu3 = (1/N) sum (x_i - mean)^3
// mu4 = (1/N) sum (x_i - mean)^4
fn central_moments_2_3_4(data: &[f64]) -> Option<(f64, f64, f64)> {
    let n = data.len();
    if n == 0 {
        return None;
    }

    let mean = sample_mean(data)?;
    let mut s2 = KahanSum::new();
    let mut s3 = KahanSum::new();
    let mut s4 = KahanSum::new();

    for &x in data {
        let d = x - mean;
        let d2 = d * d;
        s2.add(d2);
        s3.add(d2 * d);
        s4.add(d2 * d2);
    }

    let nf = n as f64;
    Some((s2.total() / nf, s3.total() / nf, s4.total() / nf))
}

fn sample_skewness(data: &[f64]) -> Option<f64> {
    let (mu2, mu3, _) = central_moments_2_3_4(data)?;
    if mu2 <= 0.0 {
        return None;
    }
    Some(mu3 / mu2.powf(1.5))
}

fn sample_excess_kurtosis(data: &[f64]) -> Option<f64> {
    let (mu2, _, mu4) = central_moments_2_3_4(data)?;
    if mu2 <= 0.0 {
        return None;
    }
    Some(mu4 / (mu2 * mu2) - 3.0)
}

// -----------------------------------------------------------------------------
// Bootstrap percentile interval for the sample mean.
// -----------------------------------------------------------------------------
fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }

    if sorted.len() == 1 {
        return Some(sorted[0]);
    }

    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;

    match lo.cmp(&hi) {
        Ordering::Equal => Some(sorted[lo]),
        _ => {
            let w = pos - lo as f64;
            Some((1.0 - w) * sorted[lo] + w * sorted[hi])
        }
    }
}

fn bootstrap_mean_confidence_interval(
    data: &[f64],
    num_bootstrap: usize,
    alpha: f64,
    rng: &mut XorShift64,
) -> Option<(f64, f64)> {
    if data.is_empty() || num_bootstrap == 0 || !(0.0..1.0).contains(&alpha) {
        return None;
    }

    let n = data.len();
    let mut means = Vec::with_capacity(num_bootstrap);

    for _ in 0..num_bootstrap {
        let mut acc = KahanSum::new();
        for _ in 0..n {
            let idx = rng.gen_index(n);
            acc.add(data[idx]);
        }
        means.push(acc.total() / n as f64);
    }

    means.sort_by(|a, b| a.total_cmp(b));

    let lower = percentile(&means, alpha / 2.0)?;
    let upper = percentile(&means, 1.0 - alpha / 2.0)?;
    Some((lower, upper))
}

// -----------------------------------------------------------------------------
// Utility printing.
// -----------------------------------------------------------------------------
fn print_vector(label: &str, data: &[f64]) {
    println!("{label}");
    println!("{}", "=".repeat(label.len()));
    for (i, x) in data.iter().enumerate() {
        println!("x[{i:>2}] = {:>.10}", x);
    }
    println!();
}

// -----------------------------------------------------------------------------
// Main demonstration.
// -----------------------------------------------------------------------------
fn main() {
    // Example dataset with mild asymmetry and one visible outlier.
    // This makes the behavior of skewness, kurtosis, and bootstrap uncertainty
    // more informative than a perfectly symmetric toy dataset.
    let data = vec![
        10.0, 10.2, 9.8, 10.1, 10.3, 9.9, 10.0, 10.4, 9.7,
        10.2, 10.1, 9.8, 10.0, 10.3, 9.9, 10.2, 10.1, 15.0,
    ];

    print_vector("Input Data", &data);

    let mean = sample_mean(&data).expect("sample mean requires at least one datum");
    let variance = sample_variance(&data).expect("sample variance requires at least two data");
    let stddev = sample_stddev(&data).expect("sample standard deviation requires at least two data");
    let skewness = sample_skewness(&data).unwrap_or(f64::NAN);
    let kurtosis = sample_excess_kurtosis(&data).unwrap_or(f64::NAN);

    let num_bootstrap = 5000;
    let alpha = 0.05;
    let mut rng = XorShift64::new(0x1234_5678_ABCD_EF01);

    let (ci_lo, ci_hi) = bootstrap_mean_confidence_interval(
        &data,
        num_bootstrap,
        alpha,
        &mut rng,
    )
    .expect("bootstrap interval requires nonempty data and valid parameters");

    println!("Sample Descriptive Statistics");
    println!("=============================");
    println!("Number of observations N      = {}", data.len());
    println!("Sample mean                   = {:>.10}", mean);
    println!("Sample variance               = {:>.10}", variance);
    println!("Sample standard deviation     = {:>.10}", stddev);
    println!("Empirical skewness            = {:>.10}", skewness);
    println!("Empirical excess kurtosis     = {:>.10}", kurtosis);
    println!();

    println!("Bootstrap Uncertainty for the Mean");
    println!("==================================");
    println!("Bootstrap replications        = {}", num_bootstrap);
    println!("Confidence level              = {:.1}%", 100.0 * (1.0 - alpha));
    println!("Percentile confidence interval= [{:>.10}, {:>.10}]", ci_lo, ci_hi);
    println!();

    println!("Interpretation");
    println!("==============");
    println!("The program computes the standard sample estimators for mean and variance,");
    println!("then supplements them with higher-moment summaries and a bootstrap-based");
    println!("uncertainty interval for the mean. The outlier in the data illustrates why");
    println!("skewness and kurtosis should be interpreted cautiously in finite samples.");
}
```

Program 14.2.1 demonstrates how classical statistical estimators can be implemented in a numerically stable and computationally robust manner. The use of compensated summation and two-pass variance computation reflects the central theme of Section 14.2: that numerical reliability is as important as statistical correctness in modern computing environments.

The results highlight the differing behavior of statistical summaries under realistic data conditions. While the sample mean and variance remain relatively stable, higher-order moments such as skewness and kurtosis exhibit strong sensitivity to outliers, reinforcing the cautionary discussion in Section 14.2.2. The bootstrap confidence interval provides a practical alternative for uncertainty quantification that does not rely on restrictive distributional assumptions.

The modular design of the implementation allows for straightforward extension to streaming algorithms, distributed aggregation, or robust estimators discussed in Section 14.2.4. In this sense, the program serves as a bridge between classical statistical definitions and modern large-scale computational practice.

## 14.2.3. Cumulants and Structural Properties

Cumulants provide an alternative representation of distributional structure that is particularly useful in numerical computing, especially for analyzing sums of independent variables.

The moment generating function (when it exists) is:

$$M_X(t) = \mathbb{E}[e^{tX}] \tag{14.2.8}$$

and the cumulant generating function is,

$$K_X(t) = \log M_X(t) \tag{14.2.9}$$

The $k$-th cumulant is defined as,

$$\kappa_k = K_X^{(k)}(0) \tag{14.2.10}$$

A fundamental property is additivity: if $X$ and $Y$ are independent, then:

$$
K_{X+Y}(t) = K_X(t) + K_Y(t) \quad \Rightarrow \quad \kappa_k(X + Y) = \kappa_k(X) + \kappa_k(Y)
\tag{14.2.11}
$$

This generalizes the familiar result that means and variances add for independent variables. However, in heavy-tailed regimes, the moment generating function may fail to exist, and cumulant-based descriptions can break down. This limitation reinforces the importance of robust statistical methods based on weaker assumptions (Minsker, 2025).

## 14.2.4. Robust and Scalable Computation of Descriptive Statistics

A modern statistical description of data combines classical summaries with robust alternatives and numerically stable computational strategies, particularly in large-scale or streaming settings.

### Robust Summaries

Quantiles and median-based statistics remain well-defined under heavy-tailed distributions and are often more stable than higher-order moments. Robust mean estimators, including median-of-means, Catoni-type estimators, and Huber-type estimators, provide non-asymptotic guarantees under weak assumptions and are effective in the presence of outliers or adversarial contamination (Minsker, 2025).

Recent developments extend robust estimation to settings with additional constraints, such as differential privacy. In such contexts, statistical procedures must balance robustness with privacy guarantees, leading to new methods for private mean and covariance estimation in heavy-tailed environments (Yu et al., 2024).

L-moments, defined as expectations of linear combinations of order statistics, provide further robust alternatives for describing dispersion and shape. They are particularly useful in heavy-tail modeling and continue to be actively developed in modern statistical research (Alvarez et al., 2025; Liao and Domański, 2025).

### Finite-precision Considerations

In numerical computing, floating-point arithmetic is part of the model rather than a mere implementation detail. Rounding errors can bias estimates, produce catastrophic cancellation, or even lead to invalid quantities such as negative variances. Modern numerical analysis treats floating-point operations as exact arithmetic followed by rounding and emphasizes algorithmic designs that control error propagation (Boldo et al., 2023).

Compensated summation techniques have received renewed attention, particularly in streaming and reduced-precision environments, where accurate accumulation of large numbers of terms is essential. Recent analyses provide detailed error bounds for such methods, reinforcing their importance in reliable statistical computation (Gao and Baidoo, 2026).

### One-pass and Streaming Algorithms

In many applications, storing the full dataset is infeasible, necessitating one-pass algorithms. A numerically stable update for the mean over streaming data or data chunks is given by:

$$\bar{X}_{n+w} = \bar{X}_n + \frac{w}{n+w}(\bar{X}_w - \bar{X}_n) \tag{14.2.12}$$

where $\bar{X}_n$ is the current mean and $\bar{X}_w$ is the mean of an incoming block of size $w$ (Grayson et al., 2025).

For variance, one-pass algorithms maintain a running quantity:

$$
M_{n+1} = M_n + (x_{n+1} - \bar{x}_n)(x_{n+1} - \bar{x}_{n+1})
\tag{14.2.13}
$$

from which the sample variance is computed as:

$$
s^2 = \frac{M_n}{n - 1}
\tag{14.2.14}
$$

In distributed or buffered settings, block updates are given by:

$$
M_{n+w} = M_n + M_w + \frac{n w}{n + w} (\bar{X}_w - \bar{X}_n)^2
\tag{14.2.15}
$$

allowing partial summaries to be merged without revisiting raw data (Grayson et al., 2025).

*Complexity and Scalability.* One-pass mean and variance updates require $O(N)$ time for scalar data. For $d$-dimensional data, mean computation scales as $O(Nd)$, while explicit covariance computation scales as $O(Nd^2)$, motivating dimensionality reduction or approximation strategies in high dimensions (Li et al., 2024).

Memory requirements are minimal for scalar statistics and scale as $O(d)$ for means and $O(d^2)$ for covariance matrices. In large-scale applications, the ability to compute statistics in a streaming fashion with controlled numerical error is a major practical advantage (Grayson et al., 2025).

Finally, numerically unstable formulas, such as computing variance via $x^2 - \bar{x}^2$, can suffer from catastrophic cancellation. Stable recurrences and compensated summation are therefore essential tools in practical implementations (Boldo et al., 2023; Gao and Baidoo, 2026).

### Rust Implementation

Following the discussion in Section 14.2.4 on robust and scalable computation of descriptive statistics, Program 14.2.2 provides a practical implementation of numerically stable and streaming-based statistical estimation. In modern numerical computing, data are often processed in environments where storage is limited, observations arrive sequentially, and numerical precision must be carefully controlled. This program integrates classical summaries such as mean and variance with robust alternatives including median, interquartile range, and median-of-means estimation. It also implements one-pass update formulas for streaming data and block-merging strategies for distributed computation, as described in Equations (14.2.12)–(14.2.15). By combining these techniques within a unified framework, the program demonstrates how statistical estimation can be performed reliably under finite-precision arithmetic and large-scale computational constraints.

At the core of the implementation is the `StreamingStats` structure, which maintains a running estimate of the sample mean and a second-order accumulation term corresponding to the quantity $M_n$ in Equation (14.2.13). The `update` method implements the one-pass recurrence by incorporating each new observation sequentially, avoiding the need to store the entire dataset. The sample variance is then computed from this accumulated quantity using Equation (14.2.14). This formulation ensures numerical stability and is well suited to streaming data scenarios.

The `merge` method of the `StreamingStats` structure implements the block update formulas described in Equations (14.2.12) and (14.2.15). It combines two independently computed summaries into a single aggregate without revisiting the raw data. This capability is essential in distributed or parallel environments, where partial statistics are computed across multiple processors and must be merged efficiently.

To address finite-precision issues, the program includes the `KahanSum` structure for compensated summation. This technique reduces rounding errors during accumulation, particularly when summing large numbers of values with varying magnitudes. It is used in the computation of the mean and variance in the batch setting, reflecting the discussion of floating-point effects in Section 14.2.4.

The program also implements robust statistical summaries. The functions `median`, `percentile`, and `quartiles_and_iqr` compute order-statistic-based measures that remain stable under heavy-tailed distributions and outliers. These quantities provide alternatives to variance-based summaries when classical assumptions are violated. The function `median_of_means` further illustrates a robust estimator of the mean by partitioning the data into blocks, computing block means, and taking their median. This approach reduces sensitivity to contamination and aligns with modern robust estimation techniques.

For comparison purposes, the function `variance_unstable` implements a naive one-pass formula for variance that is known to suffer from catastrophic cancellation. By comparing its output with the numerically stable two-pass and streaming methods, the program illustrates the importance of stable recurrences in practical computation.

The `main` function demonstrates the complete workflow. It evaluates classical and robust statistics, computes streaming summaries, partitions the data into blocks, and verifies consistency between merged and direct computations. This sequence mirrors realistic large-scale workflows, where data are processed incrementally and combined across computational units.

```rust
// Program 14.2.2. Robust and Scalable Computation of Descriptive Statistics
//
// Problem Statement:
// Implement numerically stable and scalable descriptive statistics for scalar data.
// The program should support:
//   1. Classical summaries: mean, variance, and standard deviation.
//   2. Robust summaries: median, quartiles, interquartile range, and median-of-means.
//   3. One-pass streaming updates for mean and variance.
//   4. Block merging formulas for distributed or buffered computation.
//   5. Comparison with a numerically unstable variance formula.
//
// The implementation is aligned with the discussion in Section 14.2.4, especially
// the streaming/block update formulas (14.2.12) through (14.2.15).

use std::cmp::Ordering;

// -----------------------------------------------------------------------------
// Compensated Summation
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, Copy)]
struct KahanSum {
    sum: f64,
    c: f64,
}

impl KahanSum {
    fn new() -> Self {
        Self { sum: 0.0, c: 0.0 }
    }

    fn add(&mut self, value: f64) {
        let y = value - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    fn total(self) -> f64 {
        self.sum
    }
}

// -----------------------------------------------------------------------------
// Percentiles, Median, Quartiles, and IQR
// -----------------------------------------------------------------------------
fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }
    if sorted.len() == 1 {
        return Some(sorted[0]);
    }

    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;

    match lo.cmp(&hi) {
        Ordering::Equal => Some(sorted[lo]),
        _ => {
            let w = pos - lo as f64;
            Some((1.0 - w) * sorted[lo] + w * sorted[hi])
        }
    }
}

fn sorted_copy(data: &[f64]) -> Vec<f64> {
    let mut v = data.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

fn median(data: &[f64]) -> Option<f64> {
    let s = sorted_copy(data);
    percentile(&s, 0.5)
}

fn quartiles_and_iqr(data: &[f64]) -> Option<(f64, f64, f64, f64)> {
    let s = sorted_copy(data);
    let q1 = percentile(&s, 0.25)?;
    let q2 = percentile(&s, 0.50)?;
    let q3 = percentile(&s, 0.75)?;
    Some((q1, q2, q3, q3 - q1))
}

// -----------------------------------------------------------------------------
// Classical Batch Statistics
// -----------------------------------------------------------------------------
fn mean_kahan(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let mut acc = KahanSum::new();
    for &x in data {
        acc.add(x);
    }
    Some(acc.total() / data.len() as f64)
}

fn variance_two_pass(data: &[f64]) -> Option<f64> {
    let n = data.len();
    if n < 2 {
        return None;
    }

    let mu = mean_kahan(data)?;
    let mut acc = KahanSum::new();
    for &x in data {
        let d = x - mu;
        acc.add(d * d);
    }
    Some(acc.total() / (n as f64 - 1.0))
}

fn stddev_two_pass(data: &[f64]) -> Option<f64> {
    variance_two_pass(data).map(f64::sqrt)
}

// A deliberately unstable one-pass formula to demonstrate cancellation:
// s^2 = (sum x_i^2 - N * mean^2) / (N - 1)
fn variance_unstable(data: &[f64]) -> Option<f64> {
    let n = data.len();
    if n < 2 {
        return None;
    }

    let mut sum = 0.0;
    let mut sumsq = 0.0;
    for &x in data {
        sum += x;
        sumsq += x * x;
    }

    let mean = sum / n as f64;
    Some((sumsq - n as f64 * mean * mean) / (n as f64 - 1.0))
}

// -----------------------------------------------------------------------------
// Streaming Mean/Variance State
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, Copy)]
struct StreamingStats {
    n: usize,
    mean: f64,
    m2: f64,
}

impl StreamingStats {
    fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    // One-pass stable update corresponding to (14.2.13).
    fn update(&mut self, x: f64) {
        self.n += 1;
        let n_f = self.n as f64;

        let delta = x - self.mean;
        self.mean += delta / n_f;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn from_slice(data: &[f64]) -> Self {
        let mut s = Self::new();
        for &x in data {
            s.update(x);
        }
        s
    }

    fn count(&self) -> usize {
        self.n
    }

    fn mean(&self) -> Option<f64> {
        if self.n == 0 {
            None
        } else {
            Some(self.mean)
        }
    }

    // Sample variance corresponding to (14.2.14).
    fn sample_variance(&self) -> Option<f64> {
        if self.n < 2 {
            None
        } else {
            Some(self.m2 / (self.n as f64 - 1.0))
        }
    }

    fn sample_stddev(&self) -> Option<f64> {
        self.sample_variance().map(f64::sqrt)
    }

    // Block merge corresponding to (14.2.12) and (14.2.15).
    fn merge(&self, other: &Self) -> Self {
        if self.n == 0 {
            return *other;
        }
        if other.n == 0 {
            return *self;
        }

        let n_a = self.n as f64;
        let n_b = other.n as f64;
        let n_total = self.n + other.n;
        let delta = other.mean - self.mean;

        let mean_total = self.mean + (n_b / (n_a + n_b)) * delta;
        let m2_total = self.m2 + other.m2 + (n_a * n_b / (n_a + n_b)) * delta * delta;

        Self {
            n: n_total,
            mean: mean_total,
            m2: m2_total,
        }
    }
}

// -----------------------------------------------------------------------------
// Robust Mean: Median-of-Means
// -----------------------------------------------------------------------------
fn median_of_means(data: &[f64], num_blocks: usize) -> Option<f64> {
    if data.is_empty() || num_blocks == 0 {
        return None;
    }

    let n = data.len();
    let b = num_blocks.min(n);
    let block_size = n.div_ceil(b);

    let mut block_means = Vec::new();

    for chunk in data.chunks(block_size) {
        if let Some(mu) = mean_kahan(chunk) {
            block_means.push(mu);
        }
    }

    median(&block_means)
}

// -----------------------------------------------------------------------------
// Display Helpers
// -----------------------------------------------------------------------------
fn print_data(label: &str, data: &[f64]) {
    println!("{label}");
    println!("{}", "=".repeat(label.len()));
    for (i, x) in data.iter().enumerate() {
        println!("x[{i:>2}] = {:>.10}", x);
    }
    println!();
}

fn print_stats_report(name: &str, data: &[f64], mom_blocks: usize) {
    println!("{name}");
    println!("{}", "=".repeat(name.len()));

    let mean = mean_kahan(data).unwrap_or(f64::NAN);
    let var_stable = variance_two_pass(data).unwrap_or(f64::NAN);
    let sd_stable = stddev_two_pass(data).unwrap_or(f64::NAN);
    let var_unstable = variance_unstable(data).unwrap_or(f64::NAN);
    let med = median(data).unwrap_or(f64::NAN);
    let (q1, q2, q3, iqr) = quartiles_and_iqr(data).unwrap_or((f64::NAN, f64::NAN, f64::NAN, f64::NAN));
    let mom = median_of_means(data, mom_blocks).unwrap_or(f64::NAN);

    println!("Number of observations             = {}", data.len());
    println!("Mean (compensated)                 = {:>.10}", mean);
    println!("Sample variance (stable two-pass)  = {:>.10}", var_stable);
    println!("Sample std. dev. (stable two-pass) = {:>.10}", sd_stable);
    println!("Variance (unstable formula)        = {:>.10}", var_unstable);
    println!("Median                             = {:>.10}", med);
    println!("Q1                                 = {:>.10}", q1);
    println!("Q2                                 = {:>.10}", q2);
    println!("Q3                                 = {:>.10}", q3);
    println!("Interquartile range                = {:>.10}", iqr);
    println!("Median-of-means ({} blocks)         = {:>.10}", mom_blocks, mom);
    println!();
}

fn print_streaming_state(label: &str, s: &StreamingStats) {
    println!("{label}");
    println!("{}", "-".repeat(label.len()));
    println!("Count                              = {}", s.count());
    println!("Mean                               = {:>.10}", s.mean().unwrap_or(f64::NAN));
    println!("Sample variance                    = {:>.10}", s.sample_variance().unwrap_or(f64::NAN));
    println!("Sample std. dev.                   = {:>.10}", s.sample_stddev().unwrap_or(f64::NAN));
    println!();
}

// -----------------------------------------------------------------------------
// Main Demonstration
// -----------------------------------------------------------------------------
fn main() {
    // A dataset with a tight cluster near 1000, plus a few extreme values.
    // This makes it useful for illustrating both robust summaries and the
    // sensitivity of unstable formulas under finite precision.
    let data = vec![
        1000.0, 1000.1, 999.9, 1000.2, 999.8, 1000.0, 1000.1, 999.9,
        1000.0, 1000.2, 999.8, 1000.1, 1000.0, 999.9, 1000.1, 1000.0,
        2500.0, -400.0, 1000.2, 999.7, 1000.0, 999.9, 1000.1, 1000.0,
    ];

    print_data("Input Data", &data);

    // -------------------------------------------------------------------------
    // Batch descriptive statistics.
    // -------------------------------------------------------------------------
    print_stats_report("Robust and Classical Descriptive Statistics", &data, 4);

    // -------------------------------------------------------------------------
    // Streaming one-pass computation.
    // -------------------------------------------------------------------------
    let mut stream = StreamingStats::new();
    for &x in &data {
        stream.update(x);
    }
    print_streaming_state("One-Pass Streaming Statistics", &stream);

    // -------------------------------------------------------------------------
    // Blocked/distributed computation: split into chunks and merge summaries.
    // -------------------------------------------------------------------------
    let block_a = &data[0..8];
    let block_b = &data[8..16];
    let block_c = &data[16..24];

    let stats_a = StreamingStats::from_slice(block_a);
    let stats_b = StreamingStats::from_slice(block_b);
    let stats_c = StreamingStats::from_slice(block_c);

    print_streaming_state("Block A Summary", &stats_a);
    print_streaming_state("Block B Summary", &stats_b);
    print_streaming_state("Block C Summary", &stats_c);

    let merged_ab = stats_a.merge(&stats_b);
    let merged_all = merged_ab.merge(&stats_c);

    print_streaming_state("Merged Summary from Blocks", &merged_all);

    // -------------------------------------------------------------------------
    // Consistency checks against direct batch computation.
    // -------------------------------------------------------------------------
    let direct_mean = mean_kahan(&data).unwrap();
    let direct_var = variance_two_pass(&data).unwrap();
    let merged_mean = merged_all.mean().unwrap();
    let merged_var = merged_all.sample_variance().unwrap();

    println!("Consistency Check");
    println!("=================");
    println!("Direct batch mean                  = {:>.10}", direct_mean);
    println!("Merged streaming mean              = {:>.10}", merged_mean);
    println!("Absolute difference in mean        = {:>.10e}", (direct_mean - merged_mean).abs());
    println!("Direct batch variance              = {:>.10}", direct_var);
    println!("Merged streaming variance          = {:>.10}", merged_var);
    println!("Absolute difference in variance    = {:>.10e}", (direct_var - merged_var).abs());
    println!();

    println!("Interpretation");
    println!("==============");
    println!("The program combines classical summaries with robust and scalable alternatives.");
    println!("Median, quartiles, and the interquartile range remain stable in the presence");
    println!("of extreme observations, while the median-of-means estimator illustrates a");
    println!("robust approach to mean estimation under contamination or heavy tails.");
    println!("The one-pass and merged summaries show how mean and variance can be computed");
    println!("without storing the full dataset, which is essential in streaming or distributed");
    println!("settings. The comparison with the unstable variance formula highlights the");
    println!("importance of numerically stable recurrences in finite-precision arithmetic.");
}
```

Program 14.2.2 demonstrates how modern statistical computation integrates robustness, scalability, and numerical stability into a unified framework. The results illustrate that classical estimators such as the mean and variance can be computed reliably using stable recurrences, while robust summaries such as the median, interquartile range, and median-of-means provide resilience against outliers and heavy-tailed behavior.

The comparison between stable and unstable variance formulas highlights the practical impact of finite-precision arithmetic, reinforcing the need for carefully designed algorithms. The streaming and block-merging implementations further show how statistical summaries can be computed efficiently in environments where storing the full dataset is infeasible.

The modular design of the program allows it to be extended to higher-dimensional data, parallel implementations, or more advanced robust estimators. It provides a foundation for scalable statistical computation in modern numerical workflows, where both data volume and numerical reliability are critical considerations.

# 14.3. Do Two Distributions Have the Same Means or Variances?

This section develops methods for comparing statistical summaries across datasets, with particular emphasis on means and variances. While classical parametric tests remain widely used, modern practice highlights the importance of assumption checking, robust alternatives, and interpretation beyond binary significance decisions. In numerical computing, these comparisons arise naturally in contexts such as model validation, simulation benchmarking, regression diagnostics, and evaluation of algorithmic changes. Consequently, both statistical validity and computational considerations must be treated as integral parts of the problem (Zhou, 2023; Höfler, 2026; Bonnini et al., 2024).

A central theme is that comparisons are meaningful only under clearly articulated assumptions regarding independence, distributional form, and scale. For example, equality of means is often assessed under normality and homoscedasticity assumptions, yet real-world datasets frequently violate these conditions through skewness, heavy tails, or heterogeneity of variance. In such settings, reliance on classical tests without diagnostic verification can lead to misleading conclusions. Modern workflows therefore incorporate graphical diagnostics, resampling-based procedures, and robust estimators that remain stable under departures from idealized assumptions.

From a computational perspective, the comparison of summaries must also account for numerical stability and scalability. In large-scale simulations or streaming data environments, incremental and mergeable formulations of statistics, such as online mean and variance updates, become essential. These formulations enable consistent comparisons across distributed datasets without requiring full data aggregation, thereby reducing memory overhead and communication costs. Moreover, careful implementation is required to avoid loss of precision, especially when dealing with nearly equal quantities or large sample sizes.

Interpretation has likewise evolved beyond hypothesis rejection toward estimation and uncertainty quantification. Confidence intervals, effect sizes, and distributional distances provide richer information about the magnitude and practical relevance of differences. In numerical computing applications, this perspective aligns naturally with error analysis, where the focus is on quantifying deviations and understanding their propagation through algorithms. As a result, the comparison of statistical summaries serves not merely as a decision tool but as a diagnostic instrument that informs model refinement, algorithm design, and performance evaluation.

Finally, modern comparative analysis often integrates computational techniques such as bootstrapping, permutation testing, and Bayesian inference. These approaches offer flexibility in complex or nonstandard settings and can be implemented efficiently using parallel and high-performance computing frameworks. The interplay between statistical rigor and computational efficiency thus defines the contemporary approach to comparing datasets, ensuring that conclusions are both mathematically sound and practically reliable.

## 14.3.1. The Two-Sample Problem

Let group $A$ consist of observations $(x_1^{(A)}, \dots, x_{n_A}^{(A)})$, and group $B$ consist of observations $(x_1^{(B)}, \dots, x_{n_B}^{(B)})$. The central questions are:

Difference in means is given as:

$$H_0 : \mu_A = \mu_B \quad \text{versus an alternative hypothesis} \tag{14.3.1}$$

Difference in variances is expressed as:

$$H_0: \sigma_A^2 = \sigma_B^2 \quad \text{versus} \quad \sigma_A^2 \ne \sigma_B^2 \tag{14.3.2}$$

Such comparisons appear in a wide range of applications, including before–after studies, treatment–control experiments, regression residual analysis, and validation of changes in simulation codes. In all cases, the statistical question is intertwined with modeling assumptions about independence, distributional form, and variability (Zhou, 2023; Höfler, 2026).

A key structural aspect of the problem is the distinction between independent and paired samples. In independent designs, observations in the two groups arise from separate sources, whereas in paired designs each observation in one group is naturally linked to a counterpart in the other. This distinction determines how variability is represented and how comparisons are formulated. In paired settings, the problem can often be reduced to analyzing within-pair differences, thereby isolating systematic effects from background variability.

Equally important is the role of distributional assumptions. While many classical methods rely on approximate normality, practical datasets encountered in numerical computing often exhibit skewness, heavy tails, or outliers. As a result, the formulation of the two-sample problem must be sufficiently flexible to accommodate departures from idealized models. This motivates the use of diagnostic tools, robust summaries, and, when necessary, alternative comparison frameworks that do not rely strictly on parametric assumptions.

From a computational standpoint, the two-sample problem frequently arises in settings where data are generated, processed, or aggregated at scale. Examples include comparing outputs of stochastic simulations, evaluating algorithmic modifications, or analyzing residual structures in large regression models. In such contexts, numerical stability and data representation become critical. Summary statistics must often be computed incrementally or combined across distributed systems, requiring formulations that are both stable and mergeable.

Finally, interpretation extends beyond the binary decision implied by hypothesis testing. The magnitude, direction, and uncertainty of differences between groups are central to understanding practical significance. In computational applications, this perspective aligns with error analysis and performance evaluation, where the goal is to quantify and explain differences rather than merely detect their presence. The formal testing procedures that operationalize these ideas are developed in the following section.

## 14.3.2. Student and Welch t-Tests

When samples are independent, approximately normally distributed, and have equal variances, the classical two-sample t-statistic is based on the pooled variance,

$$
s_p^2 = \frac{1}{n_A + n_B - 2}
\left(
\sum_{i=1}^{n_A} \left(x_i^{(A)} - \bar{x}_A\right)^2
+ \sum_{i=1}^{n_B} \left(x_i^{(B)} - \bar{x}_B\right)^2
\right) \tag{14.3.3}
$$

and the test statistic,

$$t = \frac{\bar{x}_A - \bar{x}_B}{s_p \sqrt{\frac{1}{n_A} + \frac{1}{n_B}}} \tag{14.3.4}$$

However, the assumption of equal variances is often questionable in practice. Real datasets frequently exhibit heteroscedasticity, motivating the use of Welch’s t-test, which replaces the pooled variance with:

$$t =\frac{\bar{x}_A - \bar{x}_B}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}} \tag{14.3.5}$$

with degrees of freedom approximated by a Welch–Satterthwaite formula.

From a numerical computing perspective, the important point is that while the statistic itself is inexpensive to compute, its validity depends heavily on modeling assumptions. Independence, identical distribution within groups, and tail behavior must be treated as part of the computational problem, not merely as background conditions (Zhou, 2023; Höfler, 2026).

### Rust Implementation

Following the discussion in Sections 14.3.1 and 14.3.2 on the formulation of the two-sample problem and the computation of Student and Welch t-statistics, Program 14.3.1 provides a practical implementation of two-sample comparison using numerically stable summary statistics. In numerical computing, datasets may be large, streaming, or distributed, making it impractical to rely on naïve multi-pass algorithms for computing means and variances. This program adopts an online, mergeable formulation of summary statistics, enabling accurate and scalable computation of sample means, variances, and test statistics. It evaluates both the pooled-variance Student statistic and the heteroscedastic Welch statistic, demonstrating how modeling assumptions influence the resulting inference. The implementation reflects the section’s emphasis on numerical stability, scalability, and the interpretation of statistical comparisons beyond purely theoretical formulas.

At the core of the implementation is the `OnlineStats` structure, which maintains the sample count, running mean, and accumulated centered sum of squares. The `update` function processes each observation sequentially, ensuring that the mean and variance are computed in a numerically stable manner without requiring storage of the full dataset. This directly supports the computational perspective discussed in Section 14.3, where incremental and mergeable formulations are essential for large-scale or distributed data. The `merge` function extends this capability by allowing two independently computed summaries to be combined into a single consistent summary, preserving both accuracy and efficiency.

The functions `pooled_variance`, `student_t_statistic`, and `welch_t_statistic` implement the computations corresponding to equations (14.3.3), (14.3.4), and (14.3.5). The pooled variance function reflects the equal-variance assumption underlying the classical Student test, while the Welch statistic replaces this assumption with separate variance estimates for each sample. The function `welch_satterthwaite_df` computes the approximate degrees of freedom associated with the Welch test, capturing the effect of unequal variances on the effective sample size. These functions illustrate how the mathematical formulations translate directly into computational procedures.

The `analyze_two_samples` function organizes the computation into a coherent workflow, producing a structured summary that includes means, variances, test statistics, and degrees of freedom. The `print_streaming_check` function demonstrates the mergeability property by comparing results obtained from a full pass over the data with those obtained by combining partial summaries. This confirms that the implementation is suitable for streaming and distributed environments, as emphasized in the section.

The `main` function serves as a demonstration of the two-sample comparison framework. It constructs two datasets with similar means but markedly different variances, illustrating a scenario in which the Welch test is more appropriate than the classical Student test. The printed output highlights the difference in variability between the samples, the resulting test statistics, and the reduction in effective degrees of freedom under heteroscedasticity. This example reinforces the importance of aligning statistical methods with the underlying assumptions of the data.

```rust
// Program 14.3.1
// Two-Sample Mean and Variance Comparison via Student and Welch t-Statistics
//
// Problem statement:
// Implement the core computations associated with Sections 14.3.1 and 14.3.2.
// Given two independent samples A and B, compute:
//   1. Sample means and unbiased sample variances
//   2. The pooled variance from equation (14.3.3)
//   3. The classical two-sample Student t-statistic from equation (14.3.4)
//   4. The Welch t-statistic from equation (14.3.5)
//   5. The Welch-Satterthwaite approximate degrees of freedom
//
// The implementation uses online, mergeable summary statistics in order to
// reflect the numerical-stability and scalability discussion in Section 14.3.

#[derive(Clone, Copy, Debug, Default)]
struct OnlineStats {
    n: usize,
    mean: f64,
    m2: f64,
}

impl OnlineStats {
    fn new() -> Self {
        Self::default()
    }

    fn update(&mut self, x: f64) {
        self.n += 1;
        let n_f = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n_f;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn from_slice(data: &[f64]) -> Self {
        let mut stats = Self::new();
        for &x in data {
            stats.update(x);
        }
        stats
    }

    fn merge(&self, other: &Self) -> Self {
        if self.n == 0 {
            return *other;
        }
        if other.n == 0 {
            return *self;
        }

        let n1 = self.n as f64;
        let n2 = other.n as f64;
        let n_total = n1 + n2;
        let delta = other.mean - self.mean;

        let mean = self.mean + delta * (n2 / n_total);
        let m2 = self.m2 + other.m2 + delta * delta * (n1 * n2 / n_total);

        Self {
            n: self.n + other.n,
            mean,
            m2,
        }
    }

    fn count(&self) -> usize {
        self.n
    }

    fn mean(&self) -> Option<f64> {
        if self.n == 0 {
            None
        } else {
            Some(self.mean)
        }
    }

    fn variance_unbiased(&self) -> Option<f64> {
        if self.n < 2 {
            None
        } else {
            Some(self.m2 / ((self.n - 1) as f64))
        }
    }
#[allow(dead_code)]
    fn stddev_unbiased(&self) -> Option<f64> {
        self.variance_unbiased().map(f64::sqrt)
    }
}

#[derive(Debug)]
struct TwoSampleSummary {
    n_a: usize,
    n_b: usize,
    mean_a: f64,
    mean_b: f64,
    var_a: f64,
    var_b: f64,
    pooled_variance: f64,
    student_t: f64,
    welch_t: f64,
    welch_df: f64,
    mean_difference: f64,
}

fn pooled_variance(stats_a: &OnlineStats, stats_b: &OnlineStats) -> Result<f64, String> {
    if stats_a.count() < 2 || stats_b.count() < 2 {
        return Err("Each sample must contain at least two observations.".to_string());
    }

    let n_a = stats_a.count();
    let n_b = stats_b.count();
    let s2_a = stats_a
        .variance_unbiased()
        .ok_or("Could not compute variance for sample A.")?;
    let s2_b = stats_b
        .variance_unbiased()
        .ok_or("Could not compute variance for sample B.")?;

    let numerator = ((n_a - 1) as f64) * s2_a + ((n_b - 1) as f64) * s2_b;
    let denominator = (n_a + n_b - 2) as f64;

    if denominator <= 0.0 {
        return Err("Invalid denominator in pooled variance.".to_string());
    }

    Ok(numerator / denominator)
}

fn student_t_statistic(stats_a: &OnlineStats, stats_b: &OnlineStats) -> Result<f64, String> {
    let n_a = stats_a.count();
    let n_b = stats_b.count();

    if n_a < 2 || n_b < 2 {
        return Err("Each sample must contain at least two observations.".to_string());
    }

    let mean_a = stats_a.mean().ok_or("Missing mean for sample A.")?;
    let mean_b = stats_b.mean().ok_or("Missing mean for sample B.")?;
    let s_p2 = pooled_variance(stats_a, stats_b)?;
    let s_p = s_p2.sqrt();

    let scale = s_p * ((1.0 / n_a as f64) + (1.0 / n_b as f64)).sqrt();
    if scale == 0.0 {
        return Err("Student t-statistic is undefined because the denominator is zero.".to_string());
    }

    Ok((mean_a - mean_b) / scale)
}

fn welch_t_statistic(stats_a: &OnlineStats, stats_b: &OnlineStats) -> Result<f64, String> {
    let n_a = stats_a.count();
    let n_b = stats_b.count();

    if n_a < 2 || n_b < 2 {
        return Err("Each sample must contain at least two observations.".to_string());
    }

    let mean_a = stats_a.mean().ok_or("Missing mean for sample A.")?;
    let mean_b = stats_b.mean().ok_or("Missing mean for sample B.")?;
    let s2_a = stats_a
        .variance_unbiased()
        .ok_or("Missing variance for sample A.")?;
    let s2_b = stats_b
        .variance_unbiased()
        .ok_or("Missing variance for sample B.")?;

    let denom = (s2_a / n_a as f64 + s2_b / n_b as f64).sqrt();
    if denom == 0.0 {
        return Err("Welch t-statistic is undefined because the denominator is zero.".to_string());
    }

    Ok((mean_a - mean_b) / denom)
}

fn welch_satterthwaite_df(stats_a: &OnlineStats, stats_b: &OnlineStats) -> Result<f64, String> {
    let n_a = stats_a.count();
    let n_b = stats_b.count();

    if n_a < 2 || n_b < 2 {
        return Err("Each sample must contain at least two observations.".to_string());
    }

    let s2_a = stats_a
        .variance_unbiased()
        .ok_or("Missing variance for sample A.")?;
    let s2_b = stats_b
        .variance_unbiased()
        .ok_or("Missing variance for sample B.")?;

    let term_a = s2_a / n_a as f64;
    let term_b = s2_b / n_b as f64;
    let numerator = (term_a + term_b).powi(2);

    let denominator =
        (term_a * term_a) / ((n_a - 1) as f64) + (term_b * term_b) / ((n_b - 1) as f64);

    if denominator == 0.0 {
        return Err("Welch degrees of freedom are undefined because the denominator is zero.".to_string());
    }

    Ok(numerator / denominator)
}

fn analyze_two_samples(sample_a: &[f64], sample_b: &[f64]) -> Result<TwoSampleSummary, String> {
    let stats_a = OnlineStats::from_slice(sample_a);
    let stats_b = OnlineStats::from_slice(sample_b);

    let n_a = stats_a.count();
    let n_b = stats_b.count();

    let mean_a = stats_a.mean().ok_or("Sample A is empty.")?;
    let mean_b = stats_b.mean().ok_or("Sample B is empty.")?;
    let var_a = stats_a
        .variance_unbiased()
        .ok_or("Sample A must have at least two observations.")?;
    let var_b = stats_b
        .variance_unbiased()
        .ok_or("Sample B must have at least two observations.")?;

    let pooled = pooled_variance(&stats_a, &stats_b)?;
    let t_student = student_t_statistic(&stats_a, &stats_b)?;
    let t_welch = welch_t_statistic(&stats_a, &stats_b)?;
    let df_welch = welch_satterthwaite_df(&stats_a, &stats_b)?;

    Ok(TwoSampleSummary {
        n_a,
        n_b,
        mean_a,
        mean_b,
        var_a,
        var_b,
        pooled_variance: pooled,
        student_t: t_student,
        welch_t: t_welch,
        welch_df: df_welch,
        mean_difference: mean_a - mean_b,
    })
}

fn print_summary(title: &str, summary: &TwoSampleSummary) {
    println!("{title}");
    println!("{}", "=".repeat(title.len()));

    println!("Sample A size                  = {}", summary.n_a);
    println!("Sample B size                  = {}", summary.n_b);
    println!("Mean of sample A               = {:>.10}", summary.mean_a);
    println!("Mean of sample B               = {:>.10}", summary.mean_b);
    println!("Difference in means (A - B)    = {:>.10}", summary.mean_difference);
    println!("Variance of sample A           = {:>.10}", summary.var_a);
    println!("Variance of sample B           = {:>.10}", summary.var_b);
    println!("Pooled variance                = {:>.10}", summary.pooled_variance);
    println!("Student t-statistic            = {:>.10}", summary.student_t);
    println!("Welch t-statistic              = {:>.10}", summary.welch_t);
    println!("Welch approximate d.o.f.       = {:>.10}", summary.welch_df);
    println!();
}

fn print_streaming_check(sample: &[f64], label: &str) {
    let mid = sample.len() / 2;
    let left = OnlineStats::from_slice(&sample[..mid]);
    let right = OnlineStats::from_slice(&sample[mid..]);
    let merged = left.merge(&right);
    let full = OnlineStats::from_slice(sample);

    println!("Streaming/Merge Check for {label}");
    println!("{}", "-".repeat(32 + label.len()));
    println!(
        "Full-pass mean                 = {:>.10}",
        full.mean().unwrap_or(f64::NAN)
    );
    println!(
        "Merged mean                    = {:>.10}",
        merged.mean().unwrap_or(f64::NAN)
    );
    println!(
        "Full-pass variance             = {:>.10}",
        full.variance_unbiased().unwrap_or(f64::NAN)
    );
    println!(
        "Merged variance                = {:>.10}",
        merged.variance_unbiased().unwrap_or(f64::NAN)
    );
    println!();
}

fn main() {
    // Example data with similar means but unequal variances, making the Welch
    // statistic especially relevant.
    let sample_a = vec![
        10.12, 9.94, 10.08, 10.01, 9.98, 10.10, 10.05, 9.97, 10.03, 10.00,
    ];

    let sample_b = vec![
        10.55, 9.31, 10.84, 9.76, 10.21, 8.95, 11.02, 9.48, 10.37, 9.62,
    ];

    match analyze_two_samples(&sample_a, &sample_b) {
        Ok(summary) => {
            print_summary(
                "Two-Sample Comparison via Student and Welch Statistics",
                &summary,
            );
            print_streaming_check(&sample_a, "Sample A");
            print_streaming_check(&sample_b, "Sample B");
        }
        Err(err) => {
            eprintln!("Error: {err}");
        }
    }
}
```

Program 14.3.1 demonstrates a practical approach to comparing sample means and variances using numerically stable and scalable techniques. The implementation highlights how classical statistical formulas can be adapted to modern computational settings through online updates and mergeable summaries. The comparison between Student and Welch statistics illustrates the impact of variance assumptions on inference, while the streaming validation confirms the robustness of the computational design. This framework provides a foundation for extending two-sample analysis to more advanced settings, including permutation tests, robust estimators, and large-scale simulation studies.

## 14.3.3. Paired Designs and Variance Comparisons

In many applications, observations are naturally paired, such as repeated measurements on the same system or matched simulations. In such cases, ignoring the pairing discards useful information and increases uncertainty. The problem reduces to a one-sample test on the differences:

$$d_i = x_i^{(A)} - x_i^{(B)} \tag{14.3.6}$$

with test statistic,

$$t =\frac{\bar{d}}{s_d / \sqrt{n}} \tag{14.3.7}$$

where $\bar{d}$ and $s_d$ are the sample mean and standard deviation of the differences.

This highlights an important computational principle: incorporating structural information, such as pairing, can reduce variance more effectively than increasing sample size (Zhou, 2023).

For comparing variances, the classical approach is the F-test,

$$F = \frac{s_A^2}{s_B^2} \tag{14.3.8}$$

which is exact under normality assumptions. However, it is highly sensitive to deviations from normality and to outliers. Modern applied work therefore recommends robust alternatives, such as the Levene or Brown–Forsythe tests, which transform data to absolute deviations from a central value before comparison. These methods reduce sensitivity to heavy tails and distributional anomalies (Zhou, 2023; Höfler, 2026).

### Rust Implementation

Following the discussion in Section 14.3.3 on paired designs and variance comparisons, Program 14.3.2 provides a practical implementation of paired hypothesis testing and variance comparison using both classical and robust methods. In numerical computing, paired observations frequently arise in repeated measurements, controlled experiments, and simulation comparisons, where exploiting structural pairing can significantly reduce variability. This program implements the computation of paired differences as defined in equation (14.3.6), evaluates the paired t-statistic from equation (14.3.7), and compares variances using the classical F-statistic from equation (14.3.8). In addition, it incorporates a Brown–Forsythe-style robust transformation based on absolute deviations from the median, reflecting the section’s emphasis on robustness under non-normality and outliers. The implementation highlights how statistical structure and numerical considerations interact in practical computation.

At the core of the implementation is the `OnlineStats` structure, which provides a numerically stable mechanism for computing sample means and variances through incremental updates. The `update` function processes each observation sequentially, maintaining the running mean and centered sum of squares, thereby avoiding the instability associated with naïve summation of squared deviations. This design aligns with the computational perspective discussed in Section 14.3, where stable and scalable formulations are essential. The same structure is reused for both the paired differences and the transformed data used in the robust variance comparison, illustrating how a single abstraction can support multiple statistical procedures.

The function `paired_differences` constructs the sequence $d_i$ defined in equation (14.3.6) by subtracting corresponding elements of the two samples. The `paired_t_test` function then computes the sample mean and variance of these differences and evaluates the t-statistic from equation (14.3.7). By reducing the problem to a one-sample test on differences, the implementation directly exploits the pairing structure, leading to a reduction in variability compared to treating the samples as independent. The function also reports the associated degrees of freedom, completing the classical paired-test formulation.

For variance comparison, the function `f_test_variance_ratio` computes the ratio $s_A^2 / s_B^2$ from equation (14.3.8) using unbiased sample variances obtained from the online statistics. This provides the classical F-statistic, which is exact under normality assumptions but sensitive to deviations from those assumptions. To address this limitation, the program also implements a Brown–Forsythe-style procedure in the function `brown_forsythe_statistic`. This function first computes sample medians, then transforms each observation into its absolute deviation from the median, and finally compares the means of these transformed values using an ANOVA-like construction. This transformation reduces sensitivity to heavy tails and outliers, as emphasized in the section.

The `main` function demonstrates both aspects of the implementation. It first evaluates paired measurements, printing the individual differences and summarizing the paired t-test results. It then applies variance comparison to two samples with visibly different spreads, reporting both the classical F-statistic and the robust Brown–Forsythe diagnostic. The formatted output highlights how the paired design leads to a strong and stable estimate of the mean difference, while the variance comparison illustrates the contrast between classical and robust approaches.

```rust
// Program 14.3.2
// Paired t-Test, F-Test, and Brown-Forsythe Variance Comparison
//
// Problem statement:
// Implement the core computations for Section 14.3.3.
// Given two paired samples A and B, the program computes:
//   1. The paired differences d_i = x_i^(A) - x_i^(B) from Equation (14.3.6)
//   2. The paired t-statistic from Equation (14.3.7)
//   3. The classical F-statistic for variance comparison from Equation (14.3.8)
//   4. A Brown-Forsythe style robust variance-comparison diagnostic based on
//      absolute deviations from the sample medians
//
// The implementation uses numerically stable online statistics for means and
// variances, while the robust comparison is evaluated through transformed data.

#[derive(Clone, Copy, Debug, Default)]
struct OnlineStats {
    n: usize,
    mean: f64,
    m2: f64,
}

impl OnlineStats {
    fn new() -> Self {
        Self::default()
    }

    fn update(&mut self, x: f64) {
        self.n += 1;
        let n_f = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n_f;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn from_slice(data: &[f64]) -> Self {
        let mut stats = Self::new();
        for &x in data {
            stats.update(x);
        }
        stats
    }

    fn count(&self) -> usize {
        self.n
    }

    fn mean(&self) -> Option<f64> {
        if self.n == 0 {
            None
        } else {
            Some(self.mean)
        }
    }

    fn variance_unbiased(&self) -> Option<f64> {
        if self.n < 2 {
            None
        } else {
            Some(self.m2 / (self.n as f64 - 1.0))
        }
    }

    fn stddev_unbiased(&self) -> Option<f64> {
        self.variance_unbiased().map(f64::sqrt)
    }
}

#[derive(Debug)]
struct PairedTestSummary {
    n: usize,
    mean_difference: f64,
    variance_difference: f64,
    stddev_difference: f64,
    paired_t_statistic: f64,
    degrees_of_freedom: usize,
}

#[derive(Debug)]
struct VarianceComparisonSummary {
    variance_a: f64,
    variance_b: f64,
    f_statistic: f64,
    brown_forsythe_zbar_a: f64,
    brown_forsythe_zbar_b: f64,
    brown_forsythe_f_like: f64,
}

fn median(data: &[f64]) -> Result<f64, String> {
    if data.is_empty() {
        return Err("Median is undefined for an empty sample.".to_string());
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let n = sorted.len();
    if n % 2 == 1 {
        Ok(sorted[n / 2])
    } else {
        Ok(0.5 * (sorted[n / 2 - 1] + sorted[n / 2]))
    }
}

fn paired_differences(sample_a: &[f64], sample_b: &[f64]) -> Result<Vec<f64>, String> {
    if sample_a.len() != sample_b.len() {
        return Err("Paired samples must have the same length.".to_string());
    }
    if sample_a.len() < 2 {
        return Err("At least two paired observations are required.".to_string());
    }

    let diffs = sample_a
        .iter()
        .zip(sample_b.iter())
        .map(|(&a, &b)| a - b)
        .collect();

    Ok(diffs)
}

fn paired_t_test(sample_a: &[f64], sample_b: &[f64]) -> Result<PairedTestSummary, String> {
    let diffs = paired_differences(sample_a, sample_b)?;
    let stats = OnlineStats::from_slice(&diffs);

    let n = stats.count();
    let mean_d = stats
        .mean()
        .ok_or("Could not compute the mean of the paired differences.")?;
    let var_d = stats
        .variance_unbiased()
        .ok_or("Could not compute the variance of the paired differences.")?;
    let std_d = stats
        .stddev_unbiased()
        .ok_or("Could not compute the standard deviation of the paired differences.")?;

    let denom = std_d / (n as f64).sqrt();
    if denom == 0.0 {
        return Err("Paired t-statistic is undefined because the denominator is zero.".to_string());
    }

    let t_stat = mean_d / denom;

    Ok(PairedTestSummary {
        n,
        mean_difference: mean_d,
        variance_difference: var_d,
        stddev_difference: std_d,
        paired_t_statistic: t_stat,
        degrees_of_freedom: n - 1,
    })
}

fn f_test_variance_ratio(sample_a: &[f64], sample_b: &[f64]) -> Result<VarianceComparisonSummary, String> {
    if sample_a.len() < 2 || sample_b.len() < 2 {
        return Err("Each sample must contain at least two observations.".to_string());
    }

    let stats_a = OnlineStats::from_slice(sample_a);
    let stats_b = OnlineStats::from_slice(sample_b);

    let var_a = stats_a
        .variance_unbiased()
        .ok_or("Could not compute variance for sample A.")?;
    let var_b = stats_b
        .variance_unbiased()
        .ok_or("Could not compute variance for sample B.")?;

    if var_b == 0.0 {
        return Err("F-statistic is undefined because the variance of sample B is zero.".to_string());
    }

    let f_stat = var_a / var_b;

    let robust = brown_forsythe_statistic(sample_a, sample_b)?;

    Ok(VarianceComparisonSummary {
        variance_a: var_a,
        variance_b: var_b,
        f_statistic: f_stat,
        brown_forsythe_zbar_a: robust.0,
        brown_forsythe_zbar_b: robust.1,
        brown_forsythe_f_like: robust.2,
    })
}

fn brown_forsythe_statistic(sample_a: &[f64], sample_b: &[f64]) -> Result<(f64, f64, f64), String> {
    if sample_a.len() < 2 || sample_b.len() < 2 {
        return Err("Each sample must contain at least two observations.".to_string());
    }

    let med_a = median(sample_a)?;
    let med_b = median(sample_b)?;

    let z_a: Vec<f64> = sample_a.iter().map(|&x| (x - med_a).abs()).collect();
    let z_b: Vec<f64> = sample_b.iter().map(|&x| (x - med_b).abs()).collect();

    let stats_a = OnlineStats::from_slice(&z_a);
    let stats_b = OnlineStats::from_slice(&z_b);

    let n_a = z_a.len();
    let n_b = z_b.len();
    let n_total = n_a + n_b;

    let zbar_a = stats_a
        .mean()
        .ok_or("Could not compute mean absolute deviation for sample A.")?;
    let zbar_b = stats_b
        .mean()
        .ok_or("Could not compute mean absolute deviation for sample B.")?;
    let grand_mean = (z_a.iter().sum::<f64>() + z_b.iter().sum::<f64>()) / n_total as f64;

    let ss_between =
        n_a as f64 * (zbar_a - grand_mean).powi(2) + n_b as f64 * (zbar_b - grand_mean).powi(2);

    let ss_within_a = z_a.iter().map(|&z| (z - zbar_a).powi(2)).sum::<f64>();
    let ss_within_b = z_b.iter().map(|&z| (z - zbar_b).powi(2)).sum::<f64>();
    let ss_within = ss_within_a + ss_within_b;

    let df_between = 1.0;
    let df_within = (n_total - 2) as f64;

    if df_within <= 0.0 || ss_within == 0.0 {
        return Err("Brown-Forsythe statistic is undefined for this input.".to_string());
    }

    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;
    let f_like = ms_between / ms_within;

    Ok((zbar_a, zbar_b, f_like))
}

fn print_paired_summary(summary: &PairedTestSummary) {
    println!("Paired t-Test Summary");
    println!("=====================");
    println!("Number of pairs                  = {}", summary.n);
    println!(
        "Mean of paired differences       = {:>.10}",
        summary.mean_difference
    );
    println!(
        "Variance of paired differences   = {:>.10}",
        summary.variance_difference
    );
    println!(
        "Std. dev. of paired differences  = {:>.10}",
        summary.stddev_difference
    );
    println!(
        "Paired t-statistic               = {:>.10}",
        summary.paired_t_statistic
    );
    println!(
        "Degrees of freedom               = {}",
        summary.degrees_of_freedom
    );
    println!();
}

fn print_variance_summary(summary: &VarianceComparisonSummary) {
    println!("Variance Comparison Summary");
    println!("===========================");
    println!("Variance of sample A            = {:>.10}", summary.variance_a);
    println!("Variance of sample B            = {:>.10}", summary.variance_b);
    println!("F-statistic s_A^2 / s_B^2       = {:>.10}", summary.f_statistic);
    println!(
        "Brown-Forsythe mean |x - med| A = {:>.10}",
        summary.brown_forsythe_zbar_a
    );
    println!(
        "Brown-Forsythe mean |x - med| B = {:>.10}",
        summary.brown_forsythe_zbar_b
    );
    println!(
        "Brown-Forsythe F-like statistic = {:>.10}",
        summary.brown_forsythe_f_like
    );
    println!();
}

fn print_pairs(sample_a: &[f64], sample_b: &[f64]) {
    println!("Paired Observations and Differences");
    println!("===================================");
    println!(" idx        A_i          B_i        d_i = A_i - B_i");
    for (i, (&a, &b)) in sample_a.iter().zip(sample_b.iter()).enumerate() {
        println!("{:>4}  {:>11.6}  {:>11.6}  {:>16.6}", i, a, b, a - b);
    }
    println!();
}

fn main() {
    // Example 1: paired repeated measurements on the same system.
    let sample_a = vec![
        12.14, 11.98, 12.07, 12.21, 12.03, 12.11, 11.95, 12.18, 12.05, 12.09,
    ];
    let sample_b = vec![
        11.88, 11.90, 11.96, 12.05, 11.91, 11.97, 11.84, 12.01, 11.92, 11.98,
    ];

    // Example 2: independent-looking samples for variance comparison.
    let variance_sample_a = vec![
        8.2, 8.4, 8.1, 8.3, 8.5, 8.2, 8.4, 8.3, 8.1, 8.4,
    ];
    let variance_sample_b = vec![
        7.6, 8.9, 8.1, 9.2, 7.4, 8.7, 8.0, 9.0, 7.8, 8.8,
    ];

    print_pairs(&sample_a, &sample_b);

    match paired_t_test(&sample_a, &sample_b) {
        Ok(summary) => print_paired_summary(&summary),
        Err(err) => eprintln!("Paired test error: {err}"),
    }

    match f_test_variance_ratio(&variance_sample_a, &variance_sample_b) {
        Ok(summary) => print_variance_summary(&summary),
        Err(err) => eprintln!("Variance comparison error: {err}"),
    }
}
```

Program 14.3.2 demonstrates a practical approach to incorporating structural information and robustness into statistical comparisons. The paired t-test illustrates how exploiting dependence between observations can significantly reduce variance and improve sensitivity, while the variance comparison highlights the limitations of classical methods under non-ideal conditions and the benefits of robust alternatives. The use of online statistics ensures numerical stability and scalability, making the implementation suitable for modern computational settings. This framework provides a foundation for extending paired and variance-based analyses to more complex scenarios, including resampling methods, high-dimensional data, and distributed computation.

## 14.3.4. Permutation Tests and Modern Perspectives

Permutation tests provide a flexible, computation-driven alternative to classical parametric methods. They rely on the principle that, under the null hypothesis, group labels are exchangeable.

A generic permutation test for the difference in means proceeds as follows:

$$T_{\text{obs}} = \bar{x}_A - \bar{x}B \tag{14.3.9}$$

followed by repeated random permutations of group labels, producing statistics $T^{(b)}$. The $p$-value is estimated as the fraction of permutations satisfying:

$$|T^{(b)}| \ge |T_{\text{obs}}| \tag{14.3.10}$$

The computational cost is $O(BN)$ for (B) permutations, but this is often acceptable in modern environments and can be parallelized efficiently (Bonnini et al., 2024; Li et al., 2024).

Beyond mean and variance comparisons, recent developments emphasize comparisons of broader distributional features, including skewness, kurtosis, and correlation. New effect-size measures enable principled comparisons of such quantities while accounting for sampling variability, extending classical inference frameworks to richer distributional characteristics (Pollo et al., 2026).

More broadly, modern statistical practice discourages reliance on binary “significant/non-significant” conclusions. Instead, it emphasizes reporting effect sizes, uncertainty intervals, and robustness to modeling assumptions, reflecting a shift toward more informative and reliable inference (Höfler, 2026).

### Rust Implementation

Following the discussion in Section 14.3.4 on permutation-based inference and the exchangeability principle, Program 14.3.3 provides a practical implementation of a Monte Carlo permutation test for the difference in means. In numerical computing, such methods are particularly valuable when classical parametric assumptions are questionable or when robustness to distributional irregularities is desired. The program computes the observed statistic defined in equation (14.3.9), generates a permutation distribution through repeated random relabeling of pooled observations, and estimates the two-sided p-value using the exceedance criterion in equation (14.3.10). In addition to the p-value, the implementation reports effect size and summary statistics, reflecting the modern emphasis on interpretation beyond binary hypothesis decisions and aligning with the section’s broader perspective on informative inference.

At the core of the implementation is the computation of the observed difference in means, corresponding directly to equation (14.3.9). The function `difference_in_means` evaluates this quantity using numerically stable mean estimates obtained from the `OnlineStats` structure. This structure, reused from earlier programs, provides incremental updates of sample mean and variance, ensuring stability and consistency across computations. The function `summarize_sample` uses this abstraction to compute basic statistics for each group, which are reported alongside the permutation results to provide context for the observed effect.

The permutation mechanism is implemented through the function `permutation_test_difference_in_means`. This function constructs a pooled dataset under the null hypothesis of exchangeable labels and repeatedly applies the Fisher–Yates shuffle to generate random permutations. For each permutation, the data are split into two groups of the original sizes, and the permuted statistic $T^{(b)}$ is computed using the same function as the observed statistic. The number of exceedances satisfying the condition in equation (14.3.10) is tracked to estimate the two-sided p-value. The inclusion of a plus-one correction ensures numerical stability of the Monte Carlo estimate, particularly when no exceedances are observed.

The implementation also includes a simple linear congruential generator, encapsulated in the `Lcg64` structure, to provide reproducible pseudo-random numbers without external dependencies. The `fisher_yates_shuffle` function uses this generator to produce unbiased permutations of the pooled data. This design keeps the program self-contained while illustrating the essential computational steps of permutation-based inference. The functions `pooled_standard_deviation` and `cohen_d` compute a standardized effect size, providing additional insight into the magnitude of the observed difference, which complements the p-value as recommended in modern statistical practice.

The `main` function demonstrates the full workflow of the permutation test. It defines two samples with a visible mean difference, prints the input data, and executes the permutation test with a specified number of permutations. The output includes sample summaries, the observed statistic, the permutation-based p-value, and the range of permuted statistics. The interpretation block emphasizes the probabilistic meaning of the permutation test under the null hypothesis of exchangeability and highlights the role of effect size in complementing hypothesis-based conclusions.

```rust
// Program 14.3.3
// Permutation Test for the Difference in Means
//
// Problem statement:
// Implement the core computation for Section 14.3.4.
// Given two samples A and B, the program computes:
//   1. The observed difference in means T_obs from Equation (14.3.9)
//   2. A Monte Carlo permutation distribution obtained by repeatedly
//      shuffling pooled observations under the null hypothesis
//   3. The two-sided permutation p-value from Equation (14.3.10)
//   4. Supporting summaries, including sample means, sample variances,
//      and a standardized effect-size diagnostic
//
// The implementation avoids external crates so that it can be compiled
// directly with cargo run in a minimal Rust project.

use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, Default)]
struct OnlineStats {
    n: usize,
    mean: f64,
    m2: f64,
}

impl OnlineStats {
    fn new() -> Self {
        Self::default()
    }

    fn update(&mut self, x: f64) {
        self.n += 1;
        let n_f = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n_f;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn from_slice(data: &[f64]) -> Self {
        let mut stats = Self::new();
        for &x in data {
            stats.update(x);
        }
        stats
    }

    fn count(&self) -> usize {
        self.n
    }

    fn mean(&self) -> Option<f64> {
        if self.n == 0 {
            None
        } else {
            Some(self.mean)
        }
    }

    fn variance_unbiased(&self) -> Option<f64> {
        if self.n < 2 {
            None
        } else {
            Some(self.m2 / ((self.n - 1) as f64))
        }
    }
}

#[derive(Debug)]
struct SampleSummary {
    n: usize,
    mean: f64,
    variance: f64,
}

#[derive(Debug)]
struct PermutationTestResult {
    summary_a: SampleSummary,
    summary_b: SampleSummary,
    observed_statistic: f64,
    effect_size_cohen_d: f64,
    p_value_two_sided: f64,
    exceedances: usize,
    permutations: usize,
    min_perm_stat: f64,
    max_perm_stat: f64,
}

#[derive(Clone, Debug)]
struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    fn seeded_from_clock() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xA5A5_A5A5_A5A5_A5A5);
        Self::new(nanos ^ 0xD1B5_4A32_D192_ED03)
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn gen_index(&mut self, upper_exclusive: usize) -> usize {
        debug_assert!(upper_exclusive > 0);
        (self.next_u64() % upper_exclusive as u64) as usize
    }
}

fn summarize_sample(data: &[f64]) -> Result<SampleSummary, String> {
    if data.len() < 2 {
        return Err("Each sample must contain at least two observations.".to_string());
    }

    let stats = OnlineStats::from_slice(data);
    Ok(SampleSummary {
        n: stats.count(),
        mean: stats.mean().ok_or("Could not compute sample mean.")?,
        variance: stats
            .variance_unbiased()
            .ok_or("Could not compute sample variance.")?,
    })
}

fn difference_in_means(sample_a: &[f64], sample_b: &[f64]) -> Result<f64, String> {
    let mean_a = OnlineStats::from_slice(sample_a)
        .mean()
        .ok_or("Could not compute the mean of sample A.")?;
    let mean_b = OnlineStats::from_slice(sample_b)
        .mean()
        .ok_or("Could not compute the mean of sample B.")?;
    Ok(mean_a - mean_b)
}

fn pooled_standard_deviation(sample_a: &[f64], sample_b: &[f64]) -> Result<f64, String> {
    let stats_a = OnlineStats::from_slice(sample_a);
    let stats_b = OnlineStats::from_slice(sample_b);

    if stats_a.count() < 2 || stats_b.count() < 2 {
        return Err("Each sample must contain at least two observations.".to_string());
    }

    let var_a = stats_a
        .variance_unbiased()
        .ok_or("Could not compute variance for sample A.")?;
    let var_b = stats_b
        .variance_unbiased()
        .ok_or("Could not compute variance for sample B.")?;

    let numerator =
        ((stats_a.count() - 1) as f64) * var_a + ((stats_b.count() - 1) as f64) * var_b;
    let denominator = (stats_a.count() + stats_b.count() - 2) as f64;

    if denominator <= 0.0 {
        return Err("Invalid denominator in pooled standard deviation.".to_string());
    }

    Ok((numerator / denominator).sqrt())
}

fn cohen_d(sample_a: &[f64], sample_b: &[f64]) -> Result<f64, String> {
    let mean_diff = difference_in_means(sample_a, sample_b)?;
    let s_p = pooled_standard_deviation(sample_a, sample_b)?;
    if s_p == 0.0 {
        return Err("Effect size is undefined because the pooled standard deviation is zero.".to_string());
    }
    Ok(mean_diff / s_p)
}

fn fisher_yates_shuffle<T>(data: &mut [T], rng: &mut Lcg64) {
    if data.len() < 2 {
        return;
    }
    for i in (1..data.len()).rev() {
        let j = rng.gen_index(i + 1);
        data.swap(i, j);
    }
}

fn permutation_test_difference_in_means(
    sample_a: &[f64],
    sample_b: &[f64],
    permutations: usize,
    rng: &mut Lcg64,
) -> Result<PermutationTestResult, String> {
    if sample_a.len() < 2 || sample_b.len() < 2 {
        return Err("Each sample must contain at least two observations.".to_string());
    }
    if permutations == 0 {
        return Err("The number of permutations must be positive.".to_string());
    }

    let summary_a = summarize_sample(sample_a)?;
    let summary_b = summarize_sample(sample_b)?;
    let observed = difference_in_means(sample_a, sample_b)?;
    let effect_size = cohen_d(sample_a, sample_b)?;

    let n_a = sample_a.len();
    let n_total = sample_a.len() + sample_b.len();

    let mut pooled = Vec::with_capacity(n_total);
    pooled.extend_from_slice(sample_a);
    pooled.extend_from_slice(sample_b);

    let mut exceedances = 0usize;
    let mut min_perm_stat = f64::INFINITY;
    let mut max_perm_stat = f64::NEG_INFINITY;

    for _ in 0..permutations {
        fisher_yates_shuffle(&mut pooled, rng);

        let perm_a = &pooled[..n_a];
        let perm_b = &pooled[n_a..];
        let perm_stat = difference_in_means(perm_a, perm_b)?;

        if perm_stat < min_perm_stat {
            min_perm_stat = perm_stat;
        }
        if perm_stat > max_perm_stat {
            max_perm_stat = perm_stat;
        }
        if perm_stat.abs() >= observed.abs() {
            exceedances += 1;
        }
    }

    // Plus-one correction for a stable Monte Carlo p-value estimate.
    let p_value = (exceedances as f64 + 1.0) / (permutations as f64 + 1.0);

    Ok(PermutationTestResult {
        summary_a,
        summary_b,
        observed_statistic: observed,
        effect_size_cohen_d: effect_size,
        p_value_two_sided: p_value,
        exceedances,
        permutations,
        min_perm_stat,
        max_perm_stat,
    })
}

fn print_input_samples(sample_a: &[f64], sample_b: &[f64]) {
    println!("Input Samples");
    println!("=============");
    println!("Sample A:");
    for (i, &x) in sample_a.iter().enumerate() {
        println!("  A[{i:>2}] = {:>.10}", x);
    }
    println!();

    println!("Sample B:");
    for (i, &x) in sample_b.iter().enumerate() {
        println!("  B[{i:>2}] = {:>.10}", x);
    }
    println!();
}

fn print_result(result: &PermutationTestResult) {
    println!("Permutation Test for Difference in Means");
    println!("========================================");
    println!("Sample A size                        = {}", result.summary_a.n);
    println!("Sample B size                        = {}", result.summary_b.n);
    println!("Mean of sample A                     = {:>.10}", result.summary_a.mean);
    println!("Mean of sample B                     = {:>.10}", result.summary_b.mean);
    println!("Variance of sample A                 = {:>.10}", result.summary_a.variance);
    println!("Variance of sample B                 = {:>.10}", result.summary_b.variance);
    println!("Observed statistic T_obs             = {:>.10}", result.observed_statistic);
    println!("Effect size (Cohen's d)              = {:>.10}", result.effect_size_cohen_d);
    println!("Number of permutations               = {}", result.permutations);
    println!("Permutation exceedances              = {}", result.exceedances);
    println!("Two-sided permutation p-value        = {:>.10}", result.p_value_two_sided);
    println!("Minimum permuted statistic           = {:>.10}", result.min_perm_stat);
    println!("Maximum permuted statistic           = {:>.10}", result.max_perm_stat);
    println!();
}

fn main() {
    // Example data: two groups with a moderate mean shift.
    let sample_a = vec![
        12.4, 12.1, 12.7, 12.3, 12.6, 12.2, 12.5, 12.8, 12.4, 12.7,
    ];

    let sample_b = vec![
        11.5, 11.8, 11.9, 11.6, 11.7, 12.0, 11.4, 11.8, 11.6, 11.7,
    ];

    let permutations = 20_000usize;
    let mut rng = Lcg64::seeded_from_clock();

    print_input_samples(&sample_a, &sample_b);

    match permutation_test_difference_in_means(&sample_a, &sample_b, permutations, &mut rng) {
        Ok(result) => {
            print_result(&result);
            println!("Interpretation");
            println!("==============");
            println!(
                "The permutation test estimates how often a relabeling of the pooled data\n\
                 produces a difference in means at least as extreme as the observed one."
            );
            println!(
                "The reported effect size complements the p-value by quantifying the magnitude\n\
                 of the mean difference on a standardized scale."
            );
        }
        Err(err) => {
            eprintln!("Error: {err}");
        }
    }
}
```

Program 14.3.3 demonstrates how permutation tests translate theoretical concepts into practical computational procedures. By relying on resampling rather than parametric assumptions, the method provides a flexible and robust framework for inference. The implementation illustrates the trade-off between computational cost and statistical accuracy, as governed by the number of permutations, and shows how modern computing resources make such approaches feasible. This program also reinforces the broader methodological shift emphasized in Section 14.3.4, where inference is viewed not only as a decision problem but as a quantitative assessment of evidence, uncertainty, and effect magnitude.

# 14.4. Are Two Distributions Different?

Suppose we are given two datasets $X$ and $Y$, each viewed as samples drawn from underlying probability distributions $F_X$ and $F_Y$. The central question is whether these distributions are identical or not. Formally, we test:

$$H_0: F_X(x) = F_Y(x) \quad \forall x \tag{14.4.1}$$

against alternatives. This problem goes beyond comparing means or variances and instead examines the *entire distributional structure*, including shape, spread, and tail behavior.

Such questions arise in many scientific applications. In climate science, one may ask whether temperature distributions from different models or time periods differ. In medicine, one may compare biomarker distributions between treatment and control groups. In general, the two-sample distribution problem addresses whether a condition or intervention alters the full distribution of a variable, not merely its average.

## 14.4.1. Empirical Distribution Functions and the Kolmogorov–Smirnov Test

Let $(X_1, \dots, X_n)$ and $(Y_1, \dots, Y_m)$ be independent samples. Their empirical cumulative distribution functions (ECDFs) are defined by,

$$
F_n(x) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}_{\{X_i \le x\}}, \qquad
G_m(x) = \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}_{\{Y_i \le x\}} \tag{14.4.2}
$$

These functions represent the proportion of observations in each sample that are less than or equal to a given value $x$. Thus, $F_n(x)$ and $G_m(x)$ provide empirical approximations to the underlying cumulative distribution functions of the two samples, constructed directly from the observed data.

The Kolmogorov–Smirnov (KS) two-sample statistic is:

$$D = \sup_{x} \left| F_n(x) - G_m(x) \right| \tag{14.4.3}$$

This statistic measures the maximum absolute difference between the two ECDFs over all values of $x$. In other words, it captures the largest vertical distance between the empirical distribution curves of the two samples, providing a single summary of how far apart the distributions are.

Under the null hypothesis $H_0$, the statistic $D$ has a known asymptotic distribution, which enables the computation of $p$-values for testing whether the two samples are drawn from the same distribution. The KS test is nonparametric, meaning that it does not assume any specific functional form for the distributions, and it is distribution-free for continuous data.

However, it has important limitations. The test is most sensitive to differences near the center of the distribution and less sensitive to differences in the tails. As a consequence, deviations that occur in extreme regions may not be detected effectively. In addition, the test assumes continuous data; the presence of ties violates this assumption and can affect the validity of the results. For these reasons, alternative or complementary tests are often employed in practice when these limitations are significant.

### Rust Implementation

Following the discussion in Section 14.4.1 on empirical distribution functions and the Kolmogorov–Smirnov statistic, Program 14.4.1 provides a practical implementation of the two-sample KS test for assessing whether two datasets are drawn from the same underlying distribution. In statistical computation, comparing full distributions rather than summary statistics requires constructing empirical cumulative distribution functions as defined in (14.4.2) and evaluating their maximum deviation as in (14.4.3). This program implements an efficient algorithm based on sorting and a merged traversal of the pooled samples to compute the KS statistic. It also evaluates an asymptotic p-value under the null hypothesis (14.4.1) and reports diagnostic information such as the location of maximal discrepancy and the presence of ties. The implementation emphasizes clarity, numerical robustness, and direct correspondence between the theoretical formulation and computational realization.

At the core of the implementation is the function `two_sample_ks_test`, which computes the empirical cumulative distribution functions $F_n(x)$ and $G_m(x)$ as defined in Equation (14.4.2). Rather than explicitly storing the ECDFs as functions, the algorithm exploits the fact that these functions are step functions that change only at observed data points. By sorting both samples and traversing them simultaneously, the function incrementally updates the proportions of observations less than or equal to the current value, thereby evaluating the ECDFs exactly at all relevant points.

The computation of the KS statistic follows directly from Equation (14.4.3). At each distinct value in the pooled sample, the algorithm evaluates the absolute difference $|F_n(x) - G_m(x)|$ and tracks the maximum value encountered. This merged traversal avoids redundant evaluations and ensures an efficient $O((n+m)\log(n+m))$ complexity dominated by the sorting step. The location at which the maximum difference occurs is also recorded, providing additional insight into where the two distributions diverge most strongly.

The function `ks_two_sample_asymptotic_pvalue` computes an approximate p-value under the null hypothesis (14.4.1) using the asymptotic distribution of the KS statistic. It forms the effective sample size and evaluates the complementary Kolmogorov distribution via a rapidly converging series. While this approximation is valid for continuous data, the implementation also detects ties in the pooled sample using `pooled_has_ties`, reflecting the theoretical limitation noted in the section that ties may affect the distribution-free property of the test.

Supporting functions such as `validate_finite` ensure that all input data are numerically well-defined, preventing propagation of invalid values through the computation. The structure `EcdfPoint` stores intermediate ECDF values and their differences, enabling the construction of a detailed trace of the comparison between the two samples. This trace provides a concrete numerical interpretation of the supremum in Equation (14.4.3), bridging the gap between theory and computation.

The `main` function demonstrates the complete workflow. It defines two representative datasets, invokes the KS test, and prints both summary statistics and the ECDF comparison table. The output includes the KS statistic, p-value, and interpretation relative to a conventional significance level, thereby illustrating how the theoretical hypothesis test in (14.4.1) is implemented and interpreted in practice. The inclusion of formatted output highlights the numerical behavior of the ECDFs and the precise location of their maximal deviation.

```rust
// Program 14.4.1: Two-Sample Kolmogorov-Smirnov Test
//
// Problem statement:
// Given two independent datasets X and Y, compute their empirical cumulative
// distribution functions F_n(x) and G_m(x) as in (14.4.2), then evaluate the
// two-sample Kolmogorov-Smirnov statistic
//
//     D = sup_x |F_n(x) - G_m(x)|
//
// from (14.4.3). Report the statistic, an asymptotic p-value under the
// continuous-data null hypothesis, and the ECDF comparison along the pooled
// support points so that the largest discrepancy can be inspected directly.

use std::cmp::Ordering;

#[derive(Debug, Clone)]
struct EcdfPoint {
    x: f64,
    f_x: f64,
    g_x: f64,
    abs_diff: f64,
}

#[derive(Debug, Clone)]
struct KsResult {
    n: usize,
    m: usize,
    d_stat: f64,
    p_value_asymptotic: f64,
    max_location: f64,
    ecdf_trace: Vec<EcdfPoint>,
    ties_present: bool,
}

fn main() {
    // Example datasets. These may be replaced by any two real-valued samples.
    let x = vec![
        12.1, 12.4, 12.3, 12.2, 12.5, 12.7, 12.6, 12.4, 12.8, 12.5, 12.3, 12.6,
    ];

    let y = vec![
        11.9, 12.0, 12.1, 12.2, 12.2, 12.3, 12.4, 12.4, 12.5, 12.6, 12.7, 12.8,
    ];

    println!("Two-Sample Kolmogorov-Smirnov Test");
    println!("==================================");
    println!();

    print_sample("Sample X", "X", &x);
    print_sample("Sample Y", "Y", &y);

    match two_sample_ks_test(&x, &y) {
        Ok(result) => {
            println!("Summary");
            println!("=======");
            println!("n (size of X)                   = {}", result.n);
            println!("m (size of Y)                   = {}", result.m);
            println!("KS statistic D                  = {:.10}", result.d_stat);
            println!(
                "Approximate asymptotic p-value  = {:.10}",
                result.p_value_asymptotic
            );
            println!("Location of max ECDF gap        = {:.10}", result.max_location);
            println!("Ties present in pooled data     = {}", result.ties_present);
            println!();

            interpret_result(&result);
            println!();

            println!("ECDF Trace on Pooled Support");
            println!("============================");
            println!(
                "{:>14} {:>14} {:>14} {:>14}",
                "x", "F_n(x)", "G_m(x)", "|F_n-G_m|"
            );
            for point in &result.ecdf_trace {
                println!(
                    "{:>14.6} {:>14.6} {:>14.6} {:>14.6}",
                    point.x, point.f_x, point.g_x, point.abs_diff
                );
            }
        }
        Err(err) => {
            eprintln!("Error: {}", err);
        }
    }
}

fn print_sample(name: &str, symbol: &str, data: &[f64]) {
    println!("{}", name);
    println!("{}", "=".repeat(name.len()));
    for (i, value) in data.iter().enumerate() {
        println!("{}[{:>2}] = {:.10}", symbol, i, value);
    }
}

fn two_sample_ks_test(x: &[f64], y: &[f64]) -> Result<KsResult, String> {
    if x.is_empty() || y.is_empty() {
        return Err("both samples must be nonempty".to_string());
    }

    validate_finite(x, "X")?;
    validate_finite(y, "Y")?;

    let mut xs = x.to_vec();
    let mut ys = y.to_vec();

    xs.sort_by(total_cmp_f64);
    ys.sort_by(total_cmp_f64);

    let ties_present = pooled_has_ties(&xs, &ys);

    let mut i = 0usize;
    let mut j = 0usize;
    let n = xs.len();
    let m = ys.len();

    let mut fn_val: f64;
    let mut gm_val: f64;
    let mut d_stat = 0.0_f64;
    let mut max_location = f64::NAN;
    let mut trace: Vec<EcdfPoint> = Vec::new();

    while i < n || j < m {
        let next_x = if i < n { xs[i] } else { f64::INFINITY };
        let next_y = if j < m { ys[j] } else { f64::INFINITY };
        let current = next_x.min(next_y);

        while i < n && xs[i] == current {
            i += 1;
        }
        while j < m && ys[j] == current {
            j += 1;
        }

        fn_val = i as f64 / n as f64;
        gm_val = j as f64 / m as f64;

        let abs_diff = (fn_val - gm_val).abs();
        if abs_diff > d_stat {
            d_stat = abs_diff;
            max_location = current;
        }

        trace.push(EcdfPoint {
            x: current,
            f_x: fn_val,
            g_x: gm_val,
            abs_diff,
        });
    }

    let p_value_asymptotic = ks_two_sample_asymptotic_pvalue(d_stat, n, m);

    Ok(KsResult {
        n,
        m,
        d_stat,
        p_value_asymptotic,
        max_location,
        ecdf_trace: trace,
        ties_present,
    })
}

fn validate_finite(data: &[f64], name: &str) -> Result<(), String> {
    for (k, &value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "sample {} contains a non-finite value at index {}",
                name, k
            ));
        }
    }
    Ok(())
}

fn pooled_has_ties(xs: &[f64], ys: &[f64]) -> bool {
    let mut pooled = Vec::with_capacity(xs.len() + ys.len());
    pooled.extend_from_slice(xs);
    pooled.extend_from_slice(ys);
    pooled.sort_by(total_cmp_f64);

    for w in pooled.windows(2) {
        if w[0] == w[1] {
            return true;
        }
    }
    false
}

fn total_cmp_f64(a: &f64, b: &f64) -> Ordering {
    a.total_cmp(b)
}

fn ks_two_sample_asymptotic_pvalue(d: f64, n: usize, m: usize) -> f64 {
    let n = n as f64;
    let m = m as f64;

    let en = (n * m / (n + m)).sqrt();

    // Finite-sample correction commonly used with the asymptotic series.
    let lambda = (en + 0.12 + 0.11 / en) * d;

    let p = kolmogorov_q(lambda);
    p.clamp(0.0, 1.0)
}

fn kolmogorov_q(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }

    // Complementary Kolmogorov distribution:
    //
    // Q(lambda) = 2 * sum_{k=1}^\infty (-1)^{k-1} exp(-2 k^2 lambda^2).
    //
    // The series converges rapidly for moderate lambda.
    let mut sum = 0.0_f64;
    let mut k = 1_u32;

    loop {
        let kf = k as f64;
        let term = (-2.0 * kf * kf * lambda * lambda).exp();
        let signed_term = if k % 2 == 1 { term } else { -term };
        sum += signed_term;

        if term < 1.0e-14 || k >= 1_000 {
            break;
        }
        k += 1;
    }

    2.0 * sum
}

fn interpret_result(result: &KsResult) {
    println!("Interpretation");
    println!("==============");
    println!(
        "The KS statistic is the largest observed vertical separation between"
    );
    println!(
        "the two empirical distribution functions. Here D = {:.10}, attained",
        result.d_stat
    );
    println!(
        "at x = {:.10}. The reported p-value is based on the continuous-data",
        result.max_location
    );
    println!(
        "asymptotic null distribution for the two-sample KS test."
    );

    if result.ties_present {
        println!();
        println!(
            "Because ties are present in the pooled data, the classical distribution-free"
        );
        println!(
            "interpretation is only approximate. In such settings, a permutation-based"
        );
        println!(
            "calibration is often preferable when high accuracy is required."
        );
    }

    println!();
    if result.p_value_asymptotic < 0.05 {
        println!(
            "At the conventional 5% level, this would suggest rejecting the null"
        );
        println!(
            "hypothesis that the two underlying distributions are identical."
        );
    } else {
        println!(
            "At the conventional 5% level, this does not provide strong evidence"
        );
        println!(
            "against the null hypothesis that the two underlying distributions are identical."
        );
    }
}
```

Program 14.4.1 demonstrates a practical realization of distributional comparison using empirical cumulative distribution functions and the Kolmogorov–Smirnov statistic. This implementation reflects the central idea of Section 14.4.1: that differences between distributions can be quantified by examining their entire cumulative structure rather than relying on summary measures such as means or variances.

The example illustrates how the KS statistic captures the largest discrepancy between two samples and how this discrepancy is interpreted through an asymptotic p-value. The presence of ties in the data highlights an important practical limitation, reinforcing the need for alternative methods such as permutation tests when strict distribution-free assumptions are violated.

The modular structure of the code allows for straightforward extension to other distributional tests discussed in subsequent sections, including weighted statistics and permutation-based approaches. This provides a foundation for more advanced methods that emphasize tail behavior, robustness, or graphical diagnostics, thereby extending the basic KS framework to a broader range of statistical applications.

## 14.4.2. Weighted and Specialized Distributional Tests

To address the limitations of the KS test, several extensions have been developed that modify how discrepancies between empirical distributions are measured.

### Weighted Tests

Statistics such as the Anderson–Darling and Cramér–von Mises tests place greater emphasis on discrepancies in the tails of the distribution. In contrast to the KS statistic, which focuses only on the maximum deviation, these tests aggregate differences across the entire domain, while assigning varying importance to different regions. For example, the Anderson–Darling statistic takes the form:

$$A^2 = \int \frac{\left(F_n(x) - G_m(x)\right)^2}{F(x)\left(1 - F(x)\right)} \, dx \tag{14.4.4}$$

where the denominator $F(x)(1 - F(x))$ acts as a weighting factor. This term becomes small near the extremes of the distribution, thereby increasing the contribution of discrepancies in the tails. As a result, even moderate differences in low-probability regions can have a significant impact on the value of the statistic. While these tests do not admit simple closed-form expressions for p-values, efficient numerical methods are available to evaluate their significance.

### Circular Data Tests

For periodic or bounded data, such as angular measurements, standard distributional comparisons must account for the absence of a natural starting point. In such cases, Kuiper’s statistic is defined as:

$$V = D^+ + D^- \tag{14.4.5}$$

where $D^+$ and $D^-$ denote the maximum positive and negative deviations between the ECDFs. By combining both types of deviations, this statistic captures discrepancies in a way that is insensitive to where the domain is “cut” or represented. This invariance under cyclic shifts ensures that the test treats all regions of the domain equally, leading to uniform sensitivity across the entire range.

These variants illustrate how test design can be adapted to emphasize specific aspects of distributional differences. By modifying the weighting of deviations or accounting for structural features such as periodicity, these methods provide more targeted sensitivity to features such as tail behavior or cyclic structure, thereby complementing the standard KS approach.

### Rust Implementation

Following the discussion in Section 14.4.2 on weighted and specialized distributional tests, Program 14.4.2 provides a practical implementation of distribution comparison methods that extend beyond the Kolmogorov–Smirnov statistic. While the KS test focuses on the maximum deviation between empirical distributions, weighted tests such as the Cramér–von Mises and Anderson–Darling statistics aggregate discrepancies across the entire domain, with the latter emphasizing tail behavior through the weighting in (14.4.4). In addition, for circular or periodic data, the program implements Kuiper’s statistic defined in (14.4.5), which captures both positive and negative deviations in a rotation-invariant manner. Because these statistics generally lack simple closed-form significance measures, the implementation incorporates permutation-based calibration to estimate p-values. The program demonstrates how theoretical formulations of weighted and specialized tests can be translated into efficient and interpretable numerical procedures.

At the core of the implementation is the function `weighted_distribution_tests`, which computes the Cramér–von Mises and Anderson–Darling statistics based on the empirical distribution functions defined in Equation (14.4.2). The function first sorts the two samples and constructs the pooled support. It then treats the empirical distribution functions as step functions that remain constant between successive support points. Over each interval, it evaluates the difference $F_n(x) - G_m(x)$, accumulates its squared contribution for the Cramér–von Mises statistic, and applies the weighting factor from Equation (14.4.4) to compute the Anderson–Darling statistic. This approach reflects the idea that weighted tests integrate discrepancies across the domain while assigning greater importance to regions near the tails.

The structure `WeightedStep` records the numerical quantities associated with each interval, including the ECDF values, their difference, the pooled cumulative value used in the denominator of (14.4.4), and the individual contributions to each statistic. This design allows the program to expose how different regions of the data contribute to the overall discrepancy. In particular, intervals where the pooled cumulative distribution is close to zero or one receive greater weight in the Anderson–Darling statistic, emphasizing tail behavior as described in the section.

The function `kuiper_two_sample` implements Kuiper’s statistic for circular data as defined in Equation (14.4.5). It begins by normalizing all angular observations to a common interval, ensuring consistency under periodicity. After sorting the samples, it evaluates the empirical cumulative distribution functions across the pooled support and determines the maximum positive deviation $D^+$ and maximum negative deviation $D^-$. The sum of these two quantities yields Kuiper’s statistic, which differs from the KS statistic in its invariance under cyclic shifts and its uniform sensitivity across the domain.

Because the statistics in this section do not generally admit simple analytical p-values, the implementation includes the functions `permutation_test_real` and `permutation_test_circular`, which perform Monte Carlo permutation testing. Under the null hypothesis (14.4.1), the sample labels are exchangeable. The algorithm repeatedly shuffles the pooled data, partitions it into samples of the original sizes, and recomputes the statistic. The p-value is then estimated as the fraction of permuted statistics that are at least as extreme as the observed one. This procedure provides a flexible and robust way to assess significance without relying on asymptotic approximations.

The helper functions `real_statistic_cvm`, `real_statistic_ad`, and `circular_statistic_kuiper` serve as adapters that allow each statistic to be evaluated within the general permutation framework. Additional utilities such as `validate_finite`, `pooled_has_ties`, and `normalize_angle` ensure numerical correctness and proper handling of data characteristics, while the lightweight random number generator `SimpleRng` and the `shuffle_in_place` function provide a self-contained mechanism for generating permutations.

The `main` function demonstrates the complete workflow by applying the weighted tests to real-valued samples and the Kuiper test to circular samples. It reports the computed statistics, permutation-based p-values, and representative interval contributions, thereby illustrating how the theoretical concepts of weighted discrepancy and circular invariance are realized in practice.

```rust
// Program 14.4.2: Weighted and Specialized Distributional Tests
//
// Problem statement:
// Given two independent samples, compute weighted distributional discrepancy
// measures that extend the Kolmogorov-Smirnov framework. For ordinary real-valued
// data, evaluate Cramér-von Mises and Anderson-Darling style statistics based on
// the empirical cumulative distribution functions F_n(x) and G_m(x), using the
// tail-sensitive weighting described in Equation (14.4.4). For circular data,
// compute Kuiper's statistic V = D^+ + D^- from Equation (14.4.5), where D^+ and
// D^- are the maximal positive and negative deviations between the two ECDFs.
// Because convenient closed-form p-values are not generally available in these
// forms, estimate significance by permutation testing.

use std::cmp::Ordering;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
struct WeightedStep {
    left: f64,
    right: f64,
    fn_val: f64,
    gm_val: f64,
    pooled_cdf: f64,
    diff: f64,
    cvm_contribution: f64,
    ad_contribution: f64,
}

#[derive(Debug, Clone)]
struct WeightedTestResult {
    n: usize,
    m: usize,
    cvm_stat: f64,
    ad_stat: f64,
    steps: Vec<WeightedStep>,
    ties_present: bool,
}

#[derive(Debug, Clone)]
struct KuiperResult {
    n: usize,
    m: usize,
    d_plus: f64,
    d_minus: f64,
    v_stat: f64,
    location_d_plus: f64,
    location_d_minus: f64,
}

#[derive(Debug, Clone)]
struct PermutationResult {
    observed: f64,
    p_value: f64,
    exceedances: usize,
    permutations: usize,
}

fn main() {
    let x_real = vec![
        -2.4, -1.9, -1.3, -0.8, -0.2, 0.1, 0.6, 0.9, 1.2, 1.8, 2.5, 3.1,
    ];
    let y_real = vec![
        -2.8, -2.1, -1.6, -1.0, -0.4, 0.0, 0.3, 0.7, 1.0, 1.4, 2.2, 3.8,
    ];

    let x_angles = vec![
        0.15, 0.42, 0.71, 1.05, 1.35, 1.91, 2.24, 2.73, 3.10, 5.82,
    ];
    let y_angles = vec![
        0.05, 0.31, 0.68, 0.94, 1.58, 2.05, 2.61, 3.24, 4.92, 5.60,
    ];

    let permutations = 2000usize;
    let seed = 0x1234_5678_9ABC_DEF0u64;

    println!("Weighted and Specialized Distributional Tests");
    println!("============================================");
    println!();

    println!("Real-Valued Samples for Weighted Tests");
    println!("======================================");
    print_sample("X", &x_real);
    println!();
    print_sample("Y", &y_real);
    println!();

    let weighted = weighted_distribution_tests(&x_real, &y_real)
        .expect("real-valued weighted tests should succeed");

    println!("Weighted Test Summary");
    println!("=====================");
    println!("n (size of X)                        = {}", weighted.n);
    println!("m (size of Y)                        = {}", weighted.m);
    println!(
        "Cramér-von Mises statistic           = {:.10}",
        weighted.cvm_stat
    );
    println!(
        "Anderson-Darling weighted statistic  = {:.10}",
        weighted.ad_stat
    );
    println!("Ties present in pooled data          = {}", weighted.ties_present);
    println!();

    let cvm_perm = permutation_test_real(
        &x_real,
        &y_real,
        permutations,
        seed ^ 0xA5A5_A5A5_A5A5_A5A5,
        real_statistic_cvm,
    )
    .expect("permutation test for CvM should succeed");

    let ad_perm = permutation_test_real(
        &x_real,
        &y_real,
        permutations,
        seed ^ 0x5A5A_5A5A_5A5A_5A5A,
        real_statistic_ad,
    )
    .expect("permutation test for AD should succeed");

    println!("Permutation Calibration for Weighted Tests");
    println!("==========================================");
    print_permutation_result("Cramér-von Mises", &cvm_perm);
    print_permutation_result("Anderson-Darling", &ad_perm);
    println!();

    println!("Representative Weighted Contributions");
    println!("=====================================");
    println!(
        "{:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>16} {:>16}",
        "left",
        "right",
        "F_n",
        "G_m",
        "H",
        "diff",
        "CvM contrib",
        "AD contrib"
    );
    for step in weighted.steps.iter().take(10) {
        println!(
            "{:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>16.8e} {:>16.8e}",
            step.left,
            step.right,
            step.fn_val,
            step.gm_val,
            step.pooled_cdf,
            step.diff,
            step.cvm_contribution,
            step.ad_contribution
        );
    }
    println!();

    println!("Circular Samples for Kuiper Test");
    println!("================================");
    print_sample("Theta_X", &x_angles);
    println!();
    print_sample("Theta_Y", &y_angles);
    println!();

    let kuiper = kuiper_two_sample(&x_angles, &y_angles)
        .expect("circular Kuiper test should succeed");

    println!("Kuiper Test Summary");
    println!("===================");
    println!("n (size of Theta_X)                = {}", kuiper.n);
    println!("m (size of Theta_Y)                = {}", kuiper.m);
    println!("D^+                                = {:.10}", kuiper.d_plus);
    println!("D^-                                = {:.10}", kuiper.d_minus);
    println!("Kuiper statistic V                 = {:.10}", kuiper.v_stat);
    println!(
        "Location of D^+                    = {:.10}",
        kuiper.location_d_plus
    );
    println!(
        "Location of D^-                    = {:.10}",
        kuiper.location_d_minus
    );
    println!();

    let kuiper_perm = permutation_test_circular(
        &x_angles,
        &y_angles,
        permutations,
        seed ^ 0xCAFEBABE_DEADC0DE,
        circular_statistic_kuiper,
    )
    .expect("permutation test for Kuiper should succeed");

    println!("Permutation Calibration for Kuiper Test");
    println!("=======================================");
    print_permutation_result("Kuiper", &kuiper_perm);
}

fn print_sample(symbol: &str, data: &[f64]) {
    for (i, value) in data.iter().enumerate() {
        println!("{}[{:>2}] = {:.10}", symbol, i, value);
    }
}

fn print_permutation_result(name: &str, result: &PermutationResult) {
    println!("{} observed statistic            = {:.10}", name, result.observed);
    println!("{} permutation p-value           = {:.10}", name, result.p_value);
    println!(
        "{} exceedances / permutations    = {} / {}",
        name, result.exceedances, result.permutations
    );
}

fn weighted_distribution_tests(x: &[f64], y: &[f64]) -> Result<WeightedTestResult, String> {
    if x.is_empty() || y.is_empty() {
        return Err("both samples must be nonempty".to_string());
    }
    validate_finite(x, "X")?;
    validate_finite(y, "Y")?;

    let mut xs = x.to_vec();
    let mut ys = y.to_vec();
    xs.sort_by(total_cmp_f64);
    ys.sort_by(total_cmp_f64);

    let n = xs.len();
    let m = ys.len();
    let ties_present = pooled_has_ties(&xs, &ys);

    let mut support: Vec<f64> = Vec::with_capacity(n + m);
    support.extend_from_slice(&xs);
    support.extend_from_slice(&ys);
    support.sort_by(total_cmp_f64);
    support.dedup();

    if support.len() < 2 {
        return Err("pooled support must contain at least two distinct values".to_string());
    }

    let mut i = 0usize;
    let mut j = 0usize;
    let mut steps = Vec::with_capacity(support.len() - 1);
    let mut cvm_stat = 0.0f64;
    let mut ad_stat = 0.0f64;
    let eps = 1.0e-12f64;

    for k in 0..(support.len() - 1) {
        let left = support[k];
        let right = support[k + 1];

        while i < n && xs[i] <= left {
            i += 1;
        }
        while j < m && ys[j] <= left {
            j += 1;
        }

        let fn_val = i as f64 / n as f64;
        let gm_val = j as f64 / m as f64;
        let pooled_cdf = (n as f64 * fn_val + m as f64 * gm_val) / (n + m) as f64;
        let diff = fn_val - gm_val;
        let width = right - left;

        let cvm_contribution = diff * diff * width;
        cvm_stat += cvm_contribution;

        let weight = 1.0 / (pooled_cdf * (1.0 - pooled_cdf)).max(eps);
        let ad_contribution = diff * diff * weight * width;
        ad_stat += ad_contribution;

        steps.push(WeightedStep {
            left,
            right,            
            fn_val,
            gm_val,
            pooled_cdf,
            diff,
            cvm_contribution,
            ad_contribution,
        });
    }

    Ok(WeightedTestResult {
        n,
        m,
        cvm_stat,
        ad_stat,
        steps,
        ties_present,
    })
}

fn kuiper_two_sample(x: &[f64], y: &[f64]) -> Result<KuiperResult, String> {
    if x.is_empty() || y.is_empty() {
        return Err("both circular samples must be nonempty".to_string());
    }
    validate_finite(x, "Theta_X")?;
    validate_finite(y, "Theta_Y")?;

    let mut xs: Vec<f64> = x.iter().map(|&v| normalize_angle(v)).collect();
    let mut ys: Vec<f64> = y.iter().map(|&v| normalize_angle(v)).collect();
    xs.sort_by(total_cmp_f64);
    ys.sort_by(total_cmp_f64);

    let n = xs.len();
    let m = ys.len();

    let mut support: Vec<f64> = Vec::with_capacity(n + m);
    support.extend_from_slice(&xs);
    support.extend_from_slice(&ys);
    support.sort_by(total_cmp_f64);
    support.dedup();

    let mut i = 0usize;
    let mut j = 0usize;
    let mut d_plus = 0.0f64;
    let mut d_minus = 0.0f64;
    let mut location_d_plus = 0.0f64;
    let mut location_d_minus = 0.0f64;

    for &theta in &support {
        while i < n && xs[i] <= theta {
            i += 1;
        }
        while j < m && ys[j] <= theta {
            j += 1;
        }

        let fx = i as f64 / n as f64;
        let gy = j as f64 / m as f64;
        let pos = fx - gy;
        let neg = gy - fx;

        if pos > d_plus {
            d_plus = pos;
            location_d_plus = theta;
        }
        if neg > d_minus {
            d_minus = neg;
            location_d_minus = theta;
        }
    }

    Ok(KuiperResult {
        n,
        m,
        d_plus,
        d_minus,
        v_stat: d_plus + d_minus,
        location_d_plus,
        location_d_minus,
    })
}

fn permutation_test_real(
    x: &[f64],
    y: &[f64],
    permutations: usize,
    seed: u64,
    statistic: fn(&[f64], &[f64]) -> Result<f64, String>,
) -> Result<PermutationResult, String> {
    if permutations == 0 {
        return Err("number of permutations must be positive".to_string());
    }

    let observed = statistic(x, y)?;
    let n = x.len();
    let mut pooled = Vec::with_capacity(x.len() + y.len());
    pooled.extend_from_slice(x);
    pooled.extend_from_slice(y);

    let mut rng = SimpleRng::new(seed);
    let mut exceedances = 0usize;
    let mut work = pooled.clone();

    for _ in 0..permutations {
        shuffle_in_place(&mut work, &mut rng);

        let x_perm = &work[..n];
        let y_perm = &work[n..];

        let value = statistic(x_perm, y_perm)?;
        if value >= observed {
            exceedances += 1;
        }
    }

    let p_value = (exceedances as f64 + 1.0) / (permutations as f64 + 1.0);

    Ok(PermutationResult {
        observed,
        p_value,
        exceedances,
        permutations,
    })
}

fn permutation_test_circular(
    x: &[f64],
    y: &[f64],
    permutations: usize,
    seed: u64,
    statistic: fn(&[f64], &[f64]) -> Result<f64, String>,
) -> Result<PermutationResult, String> {
    permutation_test_real(x, y, permutations, seed, statistic)
}

fn real_statistic_cvm(x: &[f64], y: &[f64]) -> Result<f64, String> {
    Ok(weighted_distribution_tests(x, y)?.cvm_stat)
}

fn real_statistic_ad(x: &[f64], y: &[f64]) -> Result<f64, String> {
    Ok(weighted_distribution_tests(x, y)?.ad_stat)
}

fn circular_statistic_kuiper(x: &[f64], y: &[f64]) -> Result<f64, String> {
    Ok(kuiper_two_sample(x, y)?.v_stat)
}

fn validate_finite(data: &[f64], name: &str) -> Result<(), String> {
    for (i, &value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "sample {} contains a non-finite value at index {}",
                name, i
            ));
        }
    }
    Ok(())
}

fn pooled_has_ties(xs: &[f64], ys: &[f64]) -> bool {
    let mut pooled = Vec::with_capacity(xs.len() + ys.len());
    pooled.extend_from_slice(xs);
    pooled.extend_from_slice(ys);
    pooled.sort_by(total_cmp_f64);

    for pair in pooled.windows(2) {
        if pair[0] == pair[1] {
            return true;
        }
    }
    false
}

fn normalize_angle(theta: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let mut value = theta % two_pi;
    if value < 0.0 {
        value += two_pi;
    }
    value
}

fn total_cmp_f64(a: &f64, b: &f64) -> Ordering {
    a.total_cmp(b)
}

fn shuffle_in_place(data: &mut [f64], rng: &mut SimpleRng) {
    if data.len() <= 1 {
        return;
    }
    for i in (1..data.len()).rev() {
        let j = rng.next_usize(i + 1);
        data.swap(i, j);
    }
}

#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.state = x;
        x
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() % upper as u64) as usize
    }
}
```

Program 14.4.2 demonstrates how distributional comparison can be refined by incorporating weighting schemes and structural considerations beyond those captured by the Kolmogorov–Smirnov statistic. By aggregating discrepancies across the entire domain, the Cramér–von Mises statistic provides a more global measure of difference, while the Anderson–Darling statistic further enhances sensitivity to tail behavior through the weighting in (14.4.4). The Kuiper statistic extends these ideas to circular domains, ensuring invariance under cyclic transformations and uniform sensitivity across all regions.

The use of permutation-based calibration highlights an important computational principle discussed in this section: when analytical distributions are unavailable or unreliable, empirical resampling methods provide a flexible and robust alternative. This approach allows the same computational framework to be applied across different statistics and data types, ensuring consistency and adaptability.

The modular structure of the implementation makes it straightforward to extend the framework to additional distributional tests, such as Wasserstein distances or energy statistics, and to incorporate more advanced resampling strategies. This establishes a foundation for further exploration of modern distribution comparison techniques, particularly in settings where tail behavior, periodic structure, or complex dependencies play a critical role.

## 14.4.3. Permutation and Graphical Approaches

Modern practice increasingly relies on permutation (Monte Carlo) tests, which avoid distributional assumptions by exploiting exchangeability under the null hypothesis. The key idea is that, under $H_0$, the labels identifying which sample each observation belongs to are arbitrary, and thus can be rearranged without changing the joint distribution.

A generic procedure is:

- Compute an observed statistic (e.g., $D$), which summarizes the difference between the two samples.
- Pool the samples and randomly permute the labels, thereby generating new samples that are consistent with the null hypothesis.
- Recompute the statistic for each permutation, producing a collection of values that represent the behavior of the statistic under label exchangeability.
- Estimate the p-value as the fraction of permutations with statistic at least as extreme as the observed value, providing a direct measure of how unusual the observed discrepancy is under $H_0$.

This approach is flexible, handles small samples, and accommodates complex dependence structures. Its flexibility arises from the fact that the same procedure can be applied regardless of the specific statistic used, and its effectiveness for small samples follows from relying on the empirical distribution generated by permutations rather than asymptotic approximations. It is widely implemented in modern software, using statistics such as KS, Kuiper, or Wasserstein distances.

A recent development is the global envelope test, which frames the two-sample problem as a graphical Monte Carlo procedure. In this approach, one plots the empirical difference between distributions together with an envelope derived from permutations, where the envelope represents the range of variation expected under the null hypothesis. If the observed curve exits the envelope, the null hypothesis is rejected, indicating that the observed differences are too large to be explained by random variation alone. These methods combine statistical testing with visual diagnostics and are particularly useful for identifying where distributions differ, as the locations where the curve leaves the envelope highlight the regions of greatest discrepancy.

### Rust Implementation

Following the discussion in Section 14.4.3 on permutation and graphical approaches, Program 14.4.3 provides a practical implementation of Monte Carlo two-sample testing based on label exchangeability under the null hypothesis (14.4.1). In situations where analytical distributions of test statistics are unavailable or unreliable, permutation methods offer a flexible and robust alternative by constructing the sampling distribution empirically. This program computes an observed discrepancy statistic, generates randomized samples through permutation, and estimates a p-value based on the fraction of permuted statistics that are at least as extreme as the observed one. In addition, it constructs an envelope for the empirical difference curve between distributions, providing a graphical diagnostic that highlights where deviations occur. The implementation demonstrates how permutation testing and graphical analysis can be combined into a unified computational framework for distributional comparison.

At the core of the implementation is the function `permutation_envelope_test`, which operationalizes the permutation procedure described in this section. It begins by computing the observed discrepancy between the two samples using the Kolmogorov–Smirnov statistic defined in Equation (14.4.3). The samples are then pooled, and their labels are randomly permuted to generate new datasets consistent with the null hypothesis (14.4.1). For each permutation, the statistic is recomputed, producing an empirical distribution that reflects the behavior of the statistic under label exchangeability.

The function `ks_curve` computes the empirical cumulative distribution functions $F_n(x)$ and $G_m(x)$ as defined in Equation (14.4.2), and evaluates their difference across the pooled support. Rather than computing only the maximum deviation, this function constructs the entire difference curve $F_n(x) - G_m(x)$, storing it in the structure `CurvePoint`. This enables both the calculation of the KS statistic and the subsequent graphical analysis. The function also identifies the location at which the maximum deviation occurs, providing additional insight into where the distributions differ most.

The permutation procedure repeatedly invokes `ecdf_difference_on_support`, which evaluates the ECDF difference on a fixed set of support points. By maintaining a consistent support across permutations, the program ensures that comparisons between observed and permuted curves are aligned pointwise. For each support point, the collection of permuted values is sorted to determine lower and upper bounds corresponding to a specified confidence level. These bounds define the envelope stored in the `EnvelopePoint` structure.

The envelope serves as a graphical diagnostic tool. For each point in the support, the program checks whether the observed difference lies outside the permutation-based bounds. If so, that location is recorded as an indication of statistically significant deviation. This approach allows the test not only to determine whether distributions differ, but also to identify where those differences occur.

The Monte Carlo p-value is computed as the fraction of permutations for which the statistic is at least as large as the observed value. This is implemented within `permutation_envelope_test` by counting exceedances across all permutations. The addition of a small correction ensures numerical stability when estimating the p-value.

Supporting functions such as `shuffle_in_place` and the lightweight `SimpleRng` provide a self-contained mechanism for generating random permutations. The functions `validate_finite` and `pooled_has_ties` ensure numerical robustness and proper handling of the data. The `main` function demonstrates the full workflow, including computation of the observed statistic, permutation testing, envelope construction, and formatted output for both numerical and graphical diagnostics.

```rust
// Program 14.4.3: Permutation and Global Envelope Test for Two-Sample Distributions
//
// Problem statement:
// Given two independent samples X and Y, test the null hypothesis that they
// arise from the same distribution by exploiting exchangeability under H_0.
// The program computes the observed two-sample Kolmogorov-Smirnov statistic,
// repeatedly permutes the pooled sample labels, recomputes the statistic for
// each permutation, estimates a Monte Carlo p-value, and constructs a global
// envelope for the ECDF difference curve F_n(x) - G_m(x). The output reports
// both the scalar test result and the locations where the observed difference
// curve exits the permutation envelope.

use std::cmp::Ordering;

#[derive(Debug, Clone)]
struct CurvePoint {
    x: f64,
    f_x: f64,
    g_x: f64,
    diff: f64,
    abs_diff: f64,
}

#[derive(Debug, Clone)]
struct KsCurveResult {
    n: usize,
    m: usize,
    d_stat: f64,
    max_location: f64,
    curve: Vec<CurvePoint>,
    ties_present: bool,
}

#[derive(Debug, Clone)]
struct EnvelopePoint {
    x: f64,
    observed_diff: f64,
    lower: f64,
    upper: f64,
    outside: bool,
}

#[derive(Debug, Clone)]
struct PermutationEnvelopeResult {
    observed_stat: f64,
    p_value: f64,
    exceedances: usize,
    permutations: usize,
    envelope: Vec<EnvelopePoint>,
    outside_locations: Vec<f64>,
}

fn main() {
    let x = vec![
        -2.3, -1.8, -1.4, -0.9, -0.5, -0.1, 0.2, 0.6, 0.9, 1.3, 1.8, 2.4,
    ];
    let y = vec![
        -2.6, -2.0, -1.6, -1.1, -0.7, -0.2, 0.0, 0.3, 0.7, 1.0, 1.5, 3.1,
    ];

    let permutations = 2000usize;
    let alpha = 0.05f64;
    let seed = 0xD1B5_4A32_C7E9_102Fu64;

    println!("Permutation and Global Envelope Test");
    println!("====================================");
    println!();

    println!("Input Samples");
    println!("=============");
    print_sample("X", &x);
    println!();
    print_sample("Y", &y);
    println!();

    let observed = ks_curve(&x, &y).expect("observed KS curve should be computable");
    let perm = permutation_envelope_test(&x, &y, permutations, alpha, seed)
        .expect("permutation envelope test should succeed");

    println!("Observed KS Summary");
    println!("===================");
    println!("n (size of X)                   = {}", observed.n);
    println!("m (size of Y)                   = {}", observed.m);
    println!("KS statistic D                  = {:.10}", observed.d_stat);
    println!("Location of max ECDF gap        = {:.10}", observed.max_location);
    println!("Ties present in pooled data     = {}", observed.ties_present);
    println!();

    println!("Permutation Test Summary");
    println!("========================");
    println!("Number of permutations          = {}", perm.permutations);
    println!("Observed statistic              = {:.10}", perm.observed_stat);
    println!("Exceedances                     = {}", perm.exceedances);
    println!("Monte Carlo p-value             = {:.10}", perm.p_value);
    println!("Envelope level                  = {:.4}", 1.0 - alpha);
    println!(
        "Number of outside envelope pts  = {}",
        perm.outside_locations.len()
    );
    if perm.outside_locations.is_empty() {
        println!("Observed curve exits envelope   = no");
    } else {
        println!("Observed curve exits envelope   = yes");
    }
    println!();

    println!("Representative Envelope Table");
    println!("=============================");
    println!(
        "{:>12} {:>14} {:>14} {:>14} {:>10}",
        "x", "obs diff", "lower", "upper", "outside"
    );
    for point in perm.envelope.iter().take(15) {
        println!(
            "{:>12.6} {:>14.6} {:>14.6} {:>14.6} {:>10}",
            point.x,
            point.observed_diff,
            point.lower,
            point.upper,
            if point.outside { "yes" } else { "no" }
        );
    }
    println!();

    println!("Observed ECDF Difference Curve");
    println!("==============================");
    println!(
        "{:>12} {:>12} {:>12} {:>12} {:>12}",
        "x", "F_n(x)", "G_m(x)", "F_n-G_m", "|diff|"
    );
    for point in observed.curve.iter() {
        println!(
            "{:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            point.x, point.f_x, point.g_x, point.diff, point.abs_diff
        );
    }
    println!();

    interpret_result(&perm);
}

fn print_sample(symbol: &str, data: &[f64]) {
    for (i, value) in data.iter().enumerate() {
        println!("{}[{:>2}] = {:.10}", symbol, i, value);
    }
}

fn interpret_result(result: &PermutationEnvelopeResult) {
    println!("Interpretation");
    println!("==============");
    if result.p_value < 0.05 {
        println!(
            "At the conventional 5% level, the permutation test suggests rejecting"
        );
        println!(
            "the null hypothesis that the two samples come from the same distribution."
        );
    } else {
        println!(
            "At the conventional 5% level, the permutation test does not provide"
        );
        println!(
            "strong evidence against the null hypothesis of identical distributions."
        );
    }

    println!();
    if result.outside_locations.is_empty() {
        println!(
            "The observed ECDF difference curve remains inside the global envelope,"
        );
        println!(
            "so the graphical diagnostic is consistent with the Monte Carlo test."
        );
    } else {
        println!(
            "The observed ECDF difference curve leaves the global envelope at one"
        );
        println!(
            "or more pooled support points, indicating where the samples differ most."
        );
        println!();
        println!("Outside-envelope locations:");
        for x in &result.outside_locations {
            println!("  x = {:.10}", x);
        }
    }
}

fn permutation_envelope_test(
    x: &[f64],
    y: &[f64],
    permutations: usize,
    alpha: f64,
    seed: u64,
) -> Result<PermutationEnvelopeResult, String> {
    if x.is_empty() || y.is_empty() {
        return Err("both samples must be nonempty".to_string());
    }
    if permutations == 0 {
        return Err("number of permutations must be positive".to_string());
    }
    if !(0.0 < alpha && alpha < 1.0) {
        return Err("alpha must lie strictly between 0 and 1".to_string());
    }

    let observed = ks_curve(x, y)?;
    let support: Vec<f64> = observed.curve.iter().map(|p| p.x).collect();
    let curve_len = support.len();
    let n = x.len();

    let mut pooled = Vec::with_capacity(x.len() + y.len());
    pooled.extend_from_slice(x);
    pooled.extend_from_slice(y);

    let mut rng = SimpleRng::new(seed);
    let mut work = pooled.clone();

    let mut exceedances = 0usize;
    let mut permutation_curves: Vec<Vec<f64>> = Vec::with_capacity(permutations);

    for _ in 0..permutations {
        shuffle_in_place(&mut work, &mut rng);
        let x_perm = &work[..n];
        let y_perm = &work[n..];

        let perm_curve = ecdf_difference_on_support(x_perm, y_perm, &support)?;
        let perm_stat = perm_curve
            .iter()
            .map(|v| v.abs())
            .fold(0.0f64, f64::max);

        if perm_stat >= observed.d_stat {
            exceedances += 1;
        }

        permutation_curves.push(perm_curve);
    }

    let p_value = (exceedances as f64 + 1.0) / (permutations as f64 + 1.0);

    let lower_rank = ((alpha / 2.0) * permutations as f64).floor() as usize;
    let upper_rank = ((1.0 - alpha / 2.0) * permutations as f64).ceil() as usize - 1;

    let mut envelope = Vec::with_capacity(curve_len);
    let mut outside_locations = Vec::new();

    for j in 0..curve_len {
        let mut values_at_j: Vec<f64> = permutation_curves.iter().map(|curve| curve[j]).collect();
        values_at_j.sort_by(total_cmp_f64);

        let lo = values_at_j[lower_rank.min(permutations - 1)];
        let hi = values_at_j[upper_rank.min(permutations - 1)];
        let observed_diff = observed.curve[j].diff;
        let outside = observed_diff < lo || observed_diff > hi;

        if outside {
            outside_locations.push(support[j]);
        }

        envelope.push(EnvelopePoint {
            x: support[j],
            observed_diff,
            lower: lo,
            upper: hi,
            outside,
        });
    }

    Ok(PermutationEnvelopeResult {
        observed_stat: observed.d_stat,
        p_value,
        exceedances,
        permutations,
        envelope,
        outside_locations,
    })
}

fn ks_curve(x: &[f64], y: &[f64]) -> Result<KsCurveResult, String> {
    if x.is_empty() || y.is_empty() {
        return Err("both samples must be nonempty".to_string());
    }

    validate_finite(x, "X")?;
    validate_finite(y, "Y")?;

    let mut xs = x.to_vec();
    let mut ys = y.to_vec();
    xs.sort_by(total_cmp_f64);
    ys.sort_by(total_cmp_f64);

    let n = xs.len();
    let m = ys.len();
    let ties_present = pooled_has_ties(&xs, &ys);

    let mut support = Vec::with_capacity(n + m);
    support.extend_from_slice(&xs);
    support.extend_from_slice(&ys);
    support.sort_by(total_cmp_f64);
    support.dedup();

    let diffs = ecdf_difference_on_support(&xs, &ys, &support)?;

    let mut curve = Vec::with_capacity(support.len());
    let mut d_stat = 0.0f64;
    let mut max_location = support[0];

    let mut i = 0usize;
    let mut j = 0usize;

    for (k, &xk) in support.iter().enumerate() {
        while i < n && xs[i] <= xk {
            i += 1;
        }
        while j < m && ys[j] <= xk {
            j += 1;
        }

        let f_x = i as f64 / n as f64;
        let g_x = j as f64 / m as f64;
        let diff = diffs[k];
        let abs_diff = diff.abs();

        if abs_diff > d_stat {
            d_stat = abs_diff;
            max_location = xk;
        }

        curve.push(CurvePoint {
            x: xk,
            f_x,
            g_x,
            diff,
            abs_diff,
        });
    }

    Ok(KsCurveResult {
        n,
        m,
        d_stat,
        max_location,
        curve,
        ties_present,
    })
}

fn ecdf_difference_on_support(x: &[f64], y: &[f64], support: &[f64]) -> Result<Vec<f64>, String> {
    if x.is_empty() || y.is_empty() {
        return Err("both samples must be nonempty".to_string());
    }

    let mut xs = x.to_vec();
    let mut ys = y.to_vec();
    xs.sort_by(total_cmp_f64);
    ys.sort_by(total_cmp_f64);

    let n = xs.len();
    let m = ys.len();

    let mut i = 0usize;
    let mut j = 0usize;
    let mut diffs = Vec::with_capacity(support.len());

    for &s in support {
        while i < n && xs[i] <= s {
            i += 1;
        }
        while j < m && ys[j] <= s {
            j += 1;
        }

        let f_s = i as f64 / n as f64;
        let g_s = j as f64 / m as f64;
        diffs.push(f_s - g_s);
    }

    Ok(diffs)
}

fn validate_finite(data: &[f64], name: &str) -> Result<(), String> {
    for (i, &value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "sample {} contains a non-finite value at index {}",
                name, i
            ));
        }
    }
    Ok(())
}

fn pooled_has_ties(xs: &[f64], ys: &[f64]) -> bool {
    let mut pooled = Vec::with_capacity(xs.len() + ys.len());
    pooled.extend_from_slice(xs);
    pooled.extend_from_slice(ys);
    pooled.sort_by(total_cmp_f64);

    for pair in pooled.windows(2) {
        if pair[0] == pair[1] {
            return true;
        }
    }
    false
}

fn shuffle_in_place(data: &mut [f64], rng: &mut SimpleRng) {
    if data.len() <= 1 {
        return;
    }
    for i in (1..data.len()).rev() {
        let j = rng.next_usize(i + 1);
        data.swap(i, j);
    }
}

fn total_cmp_f64(a: &f64, b: &f64) -> Ordering {
    a.total_cmp(b)
}

#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.state = x;
        x
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() % upper as u64) as usize
    }
}
```

Program 14.4.3 demonstrates how permutation-based methods provide a powerful and flexible framework for testing distributional equality without relying on restrictive assumptions. By constructing the sampling distribution empirically through random relabeling, the method adapts naturally to small samples and complex data structures, overcoming limitations of asymptotic approaches.

The inclusion of the envelope-based diagnostic extends the analysis beyond a single test statistic, allowing the user to visualize where discrepancies between distributions occur. This dual perspective, combining hypothesis testing with graphical interpretation, reflects a modern approach to statistical analysis in which numerical and visual insights complement each other.

The modular design of the implementation enables straightforward extension to other discrepancy measures, such as weighted statistics or distance-based metrics. It also provides a foundation for more advanced graphical techniques, including simultaneous global envelopes and functional data analysis methods. Together, these capabilities illustrate the versatility of permutation approaches in contemporary statistical computation.

## 14.4.4. Discrete Data, Complexity, and Applications

When data are categorical or discretized, distribution comparison is performed using contingency-style methods that operate on counts rather than continuous values. Let bins $i = 1, \dots, K$ have counts $R_i$ for sample $X$ and $S_i$ for sample $Y$. A classical statistic is:

$$\chi^2 = \sum_{i=1}^{K} \frac{(R_i - S_i)^2}{R_i + S_i} \tag{14.4.6}$$

which measures the discrepancy between the two sets of counts across all bins. Each term in the sum reflects how different the counts are in a given bin, scaled by the combined magnitude $R_i + S_i$, so that bins with larger total counts contribute proportionally. Under $H_0$, this statistic approximately follows a chi-square distribution with appropriate degrees of freedom, enabling hypothesis testing based on the overall deviation between the two samples.

Modern insights highlight that for small counts, classical p-values may be inaccurate. This inaccuracy arises because the approximation to the chi-square distribution becomes less reliable when expected counts are low. As a result, adjustments such as Lucy’s $Y^2$ statistic, likelihood-ratio tests, or exact multinomial tests are often preferred, as they provide more reliable assessments in such settings. When total counts differ between samples, normalization adjustments are applied before computing the statistic so that comparisons reflect relative frequencies rather than absolute sample sizes.

### Computational Complexity

KS and related statistics require sorting, costing $O((n+m)\log(n+m))$, followed by linear scans to evaluate differences across the ordered data. This separation into sorting and scanning steps reflects the need to organize the data before computing cumulative quantities.

Permutation tests require $O(B(n+m))$ operations for $B$ permutations, since each permutation involves recomputing the statistic over the pooled data. The computational effort therefore grows linearly with both the sample size and the number of permutations used to approximate the null distribution.

Chi-square tests scale as $O(K)$ for $K$ bins, since the computation involves a single pass over the bin counts. This makes them particularly efficient when the data have already been aggregated into a fixed number of categories.

These methods scale well to large datasets and can be implemented efficiently in streaming or parallel settings by maintaining partial summaries and merging results. In such implementations, counts or intermediate statistics can be updated incrementally, allowing large data streams to be processed without storing all observations simultaneously.

### Illustrative Applications

In climate science, comparing temperature distributions across decades reveals changes not only in the mean but also in variability and extremes. By applying tests such as KS or Kuiper, one can detect shifts in the overall shape of the distribution, while the use of block bootstrap methods accounts for temporal dependence in the data. Ignoring serial correlation can lead to misleading conclusions, as apparent differences may arise from dependence rather than genuine distributional change.

In clinical studies, comparing biomarker distributions between treatment and control groups allows researchers to assess whether an intervention affects the distribution as a whole. Graphical methods, permutation tests, and quantile-based summaries help identify where differences occur, such as in specific regions like the upper tail. This provides a more detailed understanding of treatment effects than comparisons based solely on averages, highlighting differences in spread or extreme values that may be clinically important.

### Rust Implementation

Following the discussion in Section 14.4.4 on discrete data and count-based distribution comparison, Program 14.4.4 provides a practical implementation of contingency-style methods for assessing whether two samples arise from the same categorical distribution. When observations are aggregated into bins, comparison is performed using count statistics such as the chi-square discrepancy defined in (14.4.6), which evaluates differences between bin counts across the two samples. This program computes the classical chi-square statistic, a normalized variant that accounts for unequal sample sizes, and a likelihood-ratio statistic suitable for small-count regimes. To address the limitations of asymptotic approximations, it also implements a Monte Carlo permutation procedure that estimates p-values directly from simulated reallocations of pooled observations. In addition, the code demonstrates incremental accumulation of counts, reflecting the efficiency and scalability of discrete methods highlighted in this section.

At the core of the implementation is the function `discrete_two_sample_tests`, which computes the count-based discrepancy measures using the bin counts $R_i$ and $S_i$ defined in Equation (14.4.6). The function aligns the categories across both samples, computes the contribution of each bin $(R_i - S_i)^2/(R_i + S_i)$, and aggregates these contributions to obtain the chi-square statistic. This directly reflects the mathematical formulation of the test and results in an efficient $O(K)$ computation once the counts are available.

To account for differences in total sample sizes, the function also computes a normalized chi-square statistic based on relative frequencies. By working with $R_i / \sum_j R_j$ and $S_i / \sum_j S_j$, the implementation ensures that comparisons reflect differences in distributional shape rather than differences in scale. This adjustment is particularly important in practical applications where the number of observations in each sample may differ substantially.

The likelihood-ratio statistic is computed using the function `g_component`, which evaluates the contribution of each bin based on observed and expected counts under the null hypothesis (14.4.1). This statistic provides an alternative to the classical chi-square measure and is often more reliable when counts are small or unevenly distributed. The implementation also tracks the number of bins with small expected counts, providing a diagnostic for when asymptotic approximations may be unreliable.

The structure `BinSummary` stores the per-bin contributions to all statistics, including raw counts, relative frequencies, and individual contributions. This allows the program to expose how each category contributes to the overall discrepancy, making the results interpretable and consistent with the decomposition of Equation (14.4.6). The aggregated results are stored in `DiscreteTestResult`, which includes summary statistics, degrees of freedom, and diagnostics.

To address the limitations of asymptotic p-values for small samples, the function `monte_carlo_label_permutation_pvalue` implements a permutation-based calibration. It pools the observations, randomly redistributes them into two samples of the original sizes, recomputes the chosen statistic, and estimates the p-value as the fraction of simulated statistics exceeding the observed value. This approach provides a direct empirical approximation to the null distribution and is applicable regardless of the specific statistic used.

The function `random_partition_from_pooled` performs the random allocation of observations across bins, while `statistic_from_counts` recomputes the selected statistic for each simulated partition. The lightweight random number generator `SimpleRng` and the `shuffle_in_place` function provide a self-contained mechanism for generating permutations without external dependencies.

The `StreamingCounter` structure illustrates how counts can be accumulated incrementally in streaming or large-scale settings. Its `update` method processes observations one at a time, while `to_map` produces the final count representation. This reflects the computational advantage of discrete methods discussed in the section: once data are aggregated, the complexity of comparison depends only on the number of bins rather than the number of observations.

The `main` function demonstrates the full workflow by constructing categorical samples, computing count-based statistics, printing per-bin contributions, and performing Monte Carlo calibration. It also reports complexity characteristics, reinforcing the distinction between sorting-based methods and count-based approaches.

```rust
// Program 14.4.4: Discrete Two-Sample Distribution Comparison by Count Statistics
//
// Problem statement:
// Given two categorical or discretized datasets represented by bin counts
// R_i and S_i, compute the classical chi-square discrepancy
//
//     chi^2 = sum_{i=1}^K (R_i - S_i)^2 / (R_i + S_i)
//
// from Equation (14.4.6). Also compute a normalized chi-square statistic
// for unequal total sample sizes, a likelihood-ratio G statistic for small
// count diagnostics, and a Monte Carlo permutation p-value obtained by
// reallocating pooled observations under the null hypothesis that the two
// samples have the same distribution. The implementation also demonstrates
// incremental count accumulation for streaming or large-scale settings.

use std::collections::BTreeMap;

#[derive(Debug, Clone)]
struct BinSummary {
    label: String,
    r_count: u64,
    s_count: u64,
    pooled: u64,
    r_prop: f64,
    s_prop: f64,
    chi2_contribution: f64,
    normalized_chi2_contribution: f64,
    g_contribution: f64,
}

#[derive(Debug, Clone)]
struct DiscreteTestResult {
    k_bins: usize,
    total_r: u64,
    total_s: u64,
    chi2_stat: f64,
    normalized_chi2_stat: f64,
    g_stat: f64,
    degrees_of_freedom: usize,
    expected_low_count_bins: usize,
    bins: Vec<BinSummary>,
}

#[derive(Debug, Clone)]
struct MonteCarloResult {
    observed: f64,
    exceedances: usize,
    simulations: usize,
    p_value: f64,
}

#[derive(Debug, Clone)]
struct StreamingCounter {
    counts: BTreeMap<String, u64>,
}

impl StreamingCounter {
    fn new() -> Self {
        Self {
            counts: BTreeMap::new(),
        }
    }

    fn update(&mut self, category: &str) {
        let entry = self.counts.entry(category.to_string()).or_insert(0);
        *entry += 1;
    }

    fn to_map(&self) -> BTreeMap<String, u64> {
        self.counts.clone()
    }
}

fn main() {
    // Example categorical samples. In practice these may come from
    // discretized measurements or directly from categorical observations.
    let sample_x = vec![
        "Very Low", "Low", "Low", "Low", "Medium", "Medium", "Medium", "Medium",
        "Medium", "High", "High", "High", "Very High", "High", "Medium", "Low",
        "Medium", "High", "High", "Medium", "Low", "Medium",
    ];

    let sample_y = vec![
        "Very Low", "Very Low", "Low", "Low", "Medium", "Medium", "High",
        "High", "High", "Very High", "Very High", "Medium", "High", "High",
        "Very High", "Medium", "High", "Very High",
    ];

    println!("Discrete Two-Sample Distribution Comparison");
    println!("===========================================");
    println!();

    let counts_x = accumulate_counts(&sample_x);
    let counts_y = accumulate_counts(&sample_y);

    print_counts("Sample X Bin Counts", &counts_x);
    println!();
    print_counts("Sample Y Bin Counts", &counts_y);
    println!();

    let result = discrete_two_sample_tests(&counts_x, &counts_y)
        .expect("discrete two-sample test should succeed");

    println!("Summary Statistics");
    println!("==================");
    println!("Number of bins K                     = {}", result.k_bins);
    println!("Total count in sample X              = {}", result.total_r);
    println!("Total count in sample Y              = {}", result.total_s);
    println!("Chi-square statistic                 = {:.10}", result.chi2_stat);
    println!(
        "Normalized chi-square statistic      = {:.10}",
        result.normalized_chi2_stat
    );
    println!("Likelihood-ratio G statistic         = {:.10}", result.g_stat);
    println!(
        "Approximate degrees of freedom       = {}",
        result.degrees_of_freedom
    );
    println!(
        "Bins with expected count < 5         = {}",
        result.expected_low_count_bins
    );
    println!();

    println!("Per-Bin Contributions");
    println!("=====================");
    println!(
        "{:>12} {:>10} {:>10} {:>10} {:>12} {:>12} {:>16} {:>16} {:>16}",
        "bin",
        "R_i",
        "S_i",
        "R_i+S_i",
        "R_i/N_R",
        "S_i/N_S",
        "chi2 contrib",
        "norm chi2 contrib",
        "G contrib"
    );
    for bin in &result.bins {
        println!(
            "{:>12} {:>10} {:>10} {:>10} {:>12.6} {:>12.6} {:>16.8e} {:>16.8e} {:>16.8e}",
            bin.label,
            bin.r_count,
            bin.s_count,
            bin.pooled,
            bin.r_prop,
            bin.s_prop,
            bin.chi2_contribution,
            bin.normalized_chi2_contribution,
            bin.g_contribution
        );
    }
    println!();

    let mc_chi2 = monte_carlo_label_permutation_pvalue(
        &result.bins,
        result.total_r as usize,
        5000,
        0x1A2B_3C4D_5E6F_7788,
        StatisticKind::ChiSquare,
    )
    .expect("Monte Carlo chi-square calibration should succeed");

    let mc_norm = monte_carlo_label_permutation_pvalue(
        &result.bins,
        result.total_r as usize,
        5000,
        0x9988_7766_5544_3322,
        StatisticKind::NormalizedChiSquare,
    )
    .expect("Monte Carlo normalized chi-square calibration should succeed");

    let mc_g = monte_carlo_label_permutation_pvalue(
        &result.bins,
        result.total_r as usize,
        5000,
        0x0F1E_2D3C_4B5A_6978,
        StatisticKind::GStatistic,
    )
    .expect("Monte Carlo G-statistic calibration should succeed");

    println!("Monte Carlo Calibration");
    println!("=======================");
    print_monte_carlo_result("Chi-square", &mc_chi2);
    print_monte_carlo_result("Normalized chi-square", &mc_norm);
    print_monte_carlo_result("G statistic", &mc_g);
    println!();

    println!("Complexity Notes");
    println!("================");
    println!("Count-based statistics after aggregation scale as O(K),");
    println!("where K is the number of bins. This is reflected here by");
    println!("a single pass over the aligned count vectors.");
    println!();
    println!("The Monte Carlo calibration shown below scales as O(BK),");
    println!("where B is the number of random reallocations.");
    println!("This corresponds to recomputing a count-based statistic");
    println!("for each simulated relabeling of the pooled observations.");
    println!();

    interpret_result(&result, &mc_norm);
}

fn accumulate_counts(sample: &[&str]) -> BTreeMap<String, u64> {
    let mut counter = StreamingCounter::new();
    for &value in sample {
        counter.update(value);
    }
    counter.to_map()
}

fn print_counts(title: &str, counts: &BTreeMap<String, u64>) {
    println!("{}", title);
    println!("{}", "=".repeat(title.len()));
    for (label, count) in counts {
        println!("{:>12} : {}", label, count);
    }
    let total: u64 = counts.values().sum();
    println!("{:>12} : {}", "Total", total);
}

fn discrete_two_sample_tests(
    counts_r: &BTreeMap<String, u64>,
    counts_s: &BTreeMap<String, u64>,
) -> Result<DiscreteTestResult, String> {
    let total_r: u64 = counts_r.values().sum();
    let total_s: u64 = counts_s.values().sum();

    if total_r == 0 || total_s == 0 {
        return Err("both samples must contain at least one observation".to_string());
    }

    let all_labels = union_labels(counts_r, counts_s);
    let mut bins = Vec::with_capacity(all_labels.len());

    let mut chi2_stat = 0.0f64;
    let mut normalized_chi2_stat = 0.0f64;
    let mut g_stat = 0.0f64;
    let mut expected_low_count_bins = 0usize;

    let total_r_f = total_r as f64;
    let total_s_f = total_s as f64;
    let total_all_f = (total_r + total_s) as f64;

    for label in all_labels {
        let r = *counts_r.get(&label).unwrap_or(&0);
        let s = *counts_s.get(&label).unwrap_or(&0);
        let pooled = r + s;

        if pooled == 0 {
            continue;
        }

        let r_f = r as f64;
        let s_f = s as f64;
        let pooled_f = pooled as f64;

        let chi2_contribution = (r_f - s_f) * (r_f - s_f) / pooled_f;
        chi2_stat += chi2_contribution;

        let r_prop = r_f / total_r_f;
        let s_prop = s_f / total_s_f;
        let normalized_chi2_contribution =
            (r_prop - s_prop) * (r_prop - s_prop) / ((r_prop + s_prop).max(1.0e-15));
        normalized_chi2_stat += normalized_chi2_contribution;

        let expected_r = total_r_f * pooled_f / total_all_f;
        let expected_s = total_s_f * pooled_f / total_all_f;

        if expected_r < 5.0 || expected_s < 5.0 {
            expected_low_count_bins += 1;
        }

        let g_contribution = g_component(r_f, expected_r) + g_component(s_f, expected_s);
        g_stat += 2.0 * g_contribution;

        bins.push(BinSummary {
            label,
            r_count: r,
            s_count: s,
            pooled,
            r_prop,
            s_prop,
            chi2_contribution,
            normalized_chi2_contribution,
            g_contribution: 2.0 * g_contribution,
        });
    }

    let degrees_of_freedom = bins.len().saturating_sub(1);

    Ok(DiscreteTestResult {
        k_bins: bins.len(),
        total_r,
        total_s,
        chi2_stat,
        normalized_chi2_stat,
        g_stat,
        degrees_of_freedom,
        expected_low_count_bins,
        bins,
    })
}

fn g_component(observed: f64, expected: f64) -> f64 {
    if observed <= 0.0 || expected <= 0.0 {
        0.0
    } else {
        observed * (observed / expected).ln()
    }
}

fn union_labels(
    counts_r: &BTreeMap<String, u64>,
    counts_s: &BTreeMap<String, u64>,
) -> Vec<String> {
    let mut all = BTreeMap::<String, ()>::new();
    for label in counts_r.keys() {
        all.insert(label.clone(), ());
    }
    for label in counts_s.keys() {
        all.insert(label.clone(), ());
    }
    all.into_keys().collect()
}

#[derive(Debug, Clone, Copy)]
enum StatisticKind {
    ChiSquare,
    NormalizedChiSquare,
    GStatistic,
}

fn monte_carlo_label_permutation_pvalue(
    pooled_bins: &[BinSummary],
    n_r: usize,
    simulations: usize,
    seed: u64,
    statistic_kind: StatisticKind,
) -> Result<MonteCarloResult, String> {
    if pooled_bins.is_empty() {
        return Err("pooled bins must be nonempty".to_string());
    }
    if simulations == 0 {
        return Err("number of simulations must be positive".to_string());
    }

    let observed = statistic_from_bins(pooled_bins, statistic_kind)?;
    let totals: Vec<u64> = pooled_bins.iter().map(|b| b.pooled).collect();
    let mut rng = SimpleRng::new(seed);

    let mut exceedances = 0usize;

    for _ in 0..simulations {
        let simulated = random_partition_from_pooled(&totals, n_r, &mut rng)?;
        let value = statistic_from_counts(&simulated.0, &simulated.1, statistic_kind)?;
        if value >= observed {
            exceedances += 1;
        }
    }

    let p_value = (exceedances as f64 + 1.0) / (simulations as f64 + 1.0);

    Ok(MonteCarloResult {
        observed,
        exceedances,
        simulations,
        p_value,
    })
}

fn statistic_from_bins(
    bins: &[BinSummary],
    statistic_kind: StatisticKind,
) -> Result<f64, String> {
    if bins.is_empty() {
        return Err("bin summaries must be nonempty".to_string());
    }

    let value = match statistic_kind {
        StatisticKind::ChiSquare => bins.iter().map(|b| b.chi2_contribution).sum(),
        StatisticKind::NormalizedChiSquare => {
            bins.iter().map(|b| b.normalized_chi2_contribution).sum()
        }
        StatisticKind::GStatistic => bins.iter().map(|b| b.g_contribution).sum(),
    };

    Ok(value)
}

fn statistic_from_counts(
    r_counts: &[u64],
    s_counts: &[u64],
    statistic_kind: StatisticKind,
) -> Result<f64, String> {
    if r_counts.len() != s_counts.len() || r_counts.is_empty() {
        return Err("count vectors must have equal positive length".to_string());
    }

    let total_r: u64 = r_counts.iter().sum();
    let total_s: u64 = s_counts.iter().sum();

    if total_r == 0 || total_s == 0 {
        return Err("simulated samples must both be nonempty".to_string());
    }

    let total_r_f = total_r as f64;
    let total_s_f = total_s as f64;
    let total_all_f = (total_r + total_s) as f64;

    let mut value = 0.0f64;

    for (&r, &s) in r_counts.iter().zip(s_counts.iter()) {
        let r_f = r as f64;
        let s_f = s as f64;
        let pooled_f = (r + s) as f64;

        if pooled_f <= 0.0 {
            continue;
        }

        match statistic_kind {
            StatisticKind::ChiSquare => {
                value += (r_f - s_f) * (r_f - s_f) / pooled_f;
            }
            StatisticKind::NormalizedChiSquare => {
                let r_prop = r_f / total_r_f;
                let s_prop = s_f / total_s_f;
                value += (r_prop - s_prop) * (r_prop - s_prop)
                    / ((r_prop + s_prop).max(1.0e-15));
            }
            StatisticKind::GStatistic => {
                let expected_r = total_r_f * pooled_f / total_all_f;
                let expected_s = total_s_f * pooled_f / total_all_f;
                value += 2.0 * (g_component(r_f, expected_r) + g_component(s_f, expected_s));
            }
        }
    }

    Ok(value)
}

fn random_partition_from_pooled(
    pooled_counts: &[u64],
    n_r: usize,
    rng: &mut SimpleRng,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let total: usize = pooled_counts.iter().map(|&v| v as usize).sum();
    if n_r > total {
        return Err("requested sample size exceeds pooled count".to_string());
    }

    let mut labels = Vec::with_capacity(total);
    for (i, &count) in pooled_counts.iter().enumerate() {
        for _ in 0..count {
            labels.push(i);
        }
    }

    shuffle_in_place(&mut labels, rng);

    let mut r_counts = vec![0u64; pooled_counts.len()];
    let mut s_counts = vec![0u64; pooled_counts.len()];

    for &idx in labels.iter().take(n_r) {
        r_counts[idx] += 1;
    }
    for &idx in labels.iter().skip(n_r) {
        s_counts[idx] += 1;
    }

    Ok((r_counts, s_counts))
}

fn print_monte_carlo_result(name: &str, result: &MonteCarloResult) {
    println!("{:>24} observed statistic = {:.10}", name, result.observed);
    println!("{:>24} exceedances       = {}", name, result.exceedances);
    println!("{:>24} simulations       = {}", name, result.simulations);
    println!("{:>24} Monte Carlo p-val = {:.10}", name, result.p_value);
    println!();
}

fn interpret_result(result: &DiscreteTestResult, mc_result: &MonteCarloResult) {
    println!("Interpretation");
    println!("==============");
    if result.expected_low_count_bins > 0 {
        println!(
            "Some bins have small expected counts, so asymptotic chi-square calibration"
        );
        println!(
            "should be treated cautiously. The Monte Carlo result is therefore a useful"
        );
        println!(
            "practical complement to the count-based discrepancy statistics."
        );
    } else {
        println!(
            "The bin counts are large enough that classical count-based statistics are"
        );
        println!(
            "likely to behave stably, although Monte Carlo calibration remains useful"
        );
        println!(
            "for direct finite-sample assessment."
        );
    }

    println!();
    if mc_result.p_value < 0.05 {
        println!(
            "At the conventional 5% level, the normalized count comparison suggests"
        );
        println!(
            "that the two samples differ in their underlying categorical distribution."
        );
    } else {
        println!(
            "At the conventional 5% level, the normalized count comparison does not"
        );
        println!(
            "provide strong evidence that the two samples come from different"
        );
        println!(
            "categorical distributions."
        );
    }
}

fn shuffle_in_place<T>(data: &mut [T], rng: &mut SimpleRng) {
    if data.len() <= 1 {
        return;
    }
    for i in (1..data.len()).rev() {
        let j = rng.next_usize(i + 1);
        data.swap(i, j);
    }
}

#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.state = x;
        x
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            0
        } else {
            (self.next_u64() % upper as u64) as usize
        }
    }
}
```

Program 14.4.4 demonstrates how distribution comparison for discrete data can be performed efficiently using count-based statistics. The chi-square statistic in (14.4.6) provides a natural measure of discrepancy across categories, while normalized and likelihood-ratio variants offer improved robustness in practical settings. The implementation highlights how these statistics can be computed in linear time once counts are available, making them well suited for large-scale or streaming applications.

The inclusion of permutation-based calibration addresses a key limitation of classical methods: the inaccuracy of asymptotic p-values when counts are small. By generating an empirical null distribution through random relabeling, the program provides a reliable alternative that adapts naturally to the data at hand.

The modular structure of the implementation allows for straightforward extension to other discrete comparison methods, including exact multinomial tests and divergence-based measures. This establishes a foundation for more advanced statistical techniques that build on count-based representations, particularly in applications where categorical or discretized data play a central role.

# 14.5. Contingency Table Analysis of Two Distributions

In many applications, the variables of interest are categorical rather than numerical. In such cases, the relationship between variables is studied through contingency tables, which record joint frequencies of occurrences. Let $X$ and $Y$ be nominal variables, such as “genotype” and “disease outcome,” or “education level” and “employment status.” The data are summarized in a table of counts:

$$N_{ij}, \qquad i = 1, \dots, I, \qquad j = 1, \dots, J \tag{14.5.1}$$

where $N_{ij}$ denotes the number of observations with $X = i$ and $Y = j$, and the total sample size is:

$$N = \sum_{i=1}^{I} \sum_{j=1}^{J} N_{ij} \tag{14.5.2}$$

The fundamental question is whether $X$ and $Y$ are statistically independent. This corresponds to the null hypothesis:

$$H_0 : P(X = i, Y = j) = P(X = i)\, P(Y = j) \tag{14.5.3}$$

which implies that the joint distribution factorizes. Under this assumption, expected counts are approximated by:

$$E_{ij} \approx \frac{N_{i\cdot}\, N_{\cdot j}}{N} \tag{14.5.4}$$

where $N_{i\cdot}$ and $N_{\cdot j}$ are the row and column totals.

This framework arises in diverse fields, including epidemiology, sociology, and engineering, wherever relationships between categorical variables must be assessed.

## 14.5.1. Chi-Square Test of Independence

The classical test for independence is the chi-square statistic:

$$\chi^2 = \sum_{i=1}^{I} \sum_{j=1}^{J}\frac{(N_{ij} - E_{ij})^2}{E_{ij}} \tag{14.5.5}$$

This statistic compares the observed counts $N_{ij}$ in each cell of the contingency table with the corresponding expected counts $E_{ij}$ under the assumption of independence. Each term in the sum measures the squared deviation between observed and expected counts, scaled by $E_{ij}$, so that cells with larger expected counts contribute proportionally while still allowing meaningful comparison across all cells. When the observed counts closely match the expected counts, the statistic remains small; larger discrepancies across the table lead to larger values of $\chi^2$.

Under the null hypothesis, and assuming sufficiently large expected counts, this statistic approximately follows a chi-square distribution with:

$$\nu = (I - 1)(J - 1) \tag{14.5.6}$$

degrees of freedom for an $(I \times J)$ table. The degrees of freedom reflect the number of independent ways in which the observed counts can vary once the row and column totals are fixed.

A small $p$-value indicates evidence against independence, suggesting that the variables are associated. In practical terms, this means that the pattern of counts across rows and columns deviates from what would be expected if the variables were unrelated. However, when expected counts are small or zero, the approximation may be unreliable, as the distribution of the statistic may differ from the chi-square form. In such cases, exact tests, such as Fisher’s exact test for $(2 \times 2)$ tables, or permutation-based approaches are recommended (Zhou, 2023), as they provide more accurate inference without relying on the large-sample approximation.

### Rust Implementation

Following the discussion in Section 14.5 on contingency table analysis of categorical variables, Program 14.5.1 provides a practical implementation of the chi-square test of independence for two distributions. In many applications, observations are grouped into categories and summarized through joint frequency tables, making it necessary to assess whether two variables are statistically independent as formulated in Equation (14.5.3). This program constructs the contingency table from raw paired observations, computes expected counts using Equation (14.5.4), and evaluates the chi-square statistic defined in Equation (14.5.5). It further reports degrees of freedom from Equation (14.5.6) and an asymptotic p-value, while also diagnosing situations where expected counts are small and the approximation may be unreliable. The implementation demonstrates how categorical data analysis can be translated into a structured and numerically stable computational workflow.

At the core of the implementation is the function `chi_square_test_of_independence`, which constructs the contingency table $N_{ij}$ defined in Equation (14.5.1). The function begins by identifying the unique row and column categories and mapping them to indices. It then accumulates counts for each pair of categories, producing the observed frequency table. From this table, it computes the row totals $N_{i\cdot}$, column totals $N_{\cdot j}$, and the total sample size $N$ as given in Equation (14.5.2). These quantities form the basis for evaluating expected counts under the null hypothesis of independence.

The expected counts are computed using Equation (14.5.4), which assumes that the joint distribution factorizes under $H_0$. For each cell, the expected value is formed from the product of the corresponding row and column totals divided by the overall sample size. The chi-square statistic from Equation (14.5.5) is then evaluated by summing the contributions $(N_{ij} - E_{ij})^2 / E_{ij}$ across all cells. This aggregation reflects how discrepancies between observed and expected counts accumulate over the entire table.

The structure `CellSummary` stores the observed count, expected count, and contribution for each cell, allowing the program to expose how individual cells influence the overall statistic. The structure `ChiSquareResult` aggregates all results, including the contingency table, expected values, totals, chi-square statistic, degrees of freedom from Equation (14.5.6), and the computed p-value. It also records diagnostic information such as the minimum expected count and the number of cells with expected counts below five, which is important for assessing the validity of the asymptotic approximation.

The asymptotic p-value is computed using the function `chi_square_survival_function`, which evaluates the upper tail probability of the chi-square distribution. This is implemented through numerical evaluation of the regularized gamma function, using both series expansion and continued fraction methods for stability across parameter regimes. The helper function `ln_gamma` provides the logarithm of the gamma function using the Lanczos approximation, ensuring accurate evaluation of the distribution function without relying on external libraries.

Additional helper functions support table construction and output formatting. The functions `collect_sorted_labels` and `build_index_map` create consistent indexing for categorical labels, while `print_contingency_table` and `print_expected_table` display the observed and expected frequencies in a structured format. The function `interpret_result` summarizes the statistical conclusion and highlights whether small expected counts may compromise the reliability of the test.

The `main` function demonstrates the complete workflow by constructing a sample dataset, computing the contingency table and associated statistics, and presenting both numerical and diagnostic output. This illustrates how the theoretical framework of contingency table analysis is implemented in practice.

```rust
// Program 14.5.1: Chi-Square Test of Independence for a Contingency Table
//
// Problem statement:
// Given paired categorical observations (X, Y), construct the contingency
// table N_ij from Equation (14.5.1), compute the total sample size N from
// Equation (14.5.2), form the expected counts
//
//     E_ij = N_{i.} N_{.j} / N
//
// from Equation (14.5.4), and evaluate the chi-square statistic
//
//     chi^2 = sum_{i=1}^I sum_{j=1}^J (N_ij - E_ij)^2 / E_ij
//
// from Equation (14.5.5). Report the degrees of freedom
//
//     nu = (I - 1)(J - 1)
//
// from Equation (14.5.6), an asymptotic p-value, and diagnostics for
// small expected counts.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
struct CellSummary {
    row_label: String,
    col_label: String,
    observed: u64,
    expected: f64,
    contribution: f64,
}

#[derive(Debug, Clone)]
struct ChiSquareResult {
    row_labels: Vec<String>,
    col_labels: Vec<String>,
    observed: Vec<Vec<u64>>,
    expected: Vec<Vec<f64>>,
    row_totals: Vec<u64>,
    col_totals: Vec<u64>,
    total: u64,
    chi_square: f64,
    degrees_of_freedom: usize,
    p_value: f64,
    cells: Vec<CellSummary>,
    min_expected: f64,
    small_expected_cells: usize,
}

fn main() {
    // Example dataset:
    // X = genotype category, Y = treatment outcome.
    let observations = vec![
        ("A", "Success"),
        ("A", "Success"),
        ("A", "Failure"),
        ("A", "Success"),
        ("A", "Failure"),
        ("A", "Success"),
        ("B", "Success"),
        ("B", "Failure"),
        ("B", "Failure"),
        ("B", "Success"),
        ("B", "Success"),
        ("B", "Failure"),
        ("B", "Failure"),
        ("C", "Failure"),
        ("C", "Failure"),
        ("C", "Success"),
        ("C", "Failure"),
        ("C", "Failure"),
        ("C", "Failure"),
        ("C", "Success"),
    ];

    println!("Chi-Square Test of Independence");
    println!("===============================");
    println!();

    match chi_square_test_of_independence(&observations) {
        Ok(result) => {
            print_contingency_table(&result);
            println!();
            print_expected_table(&result);
            println!();

            println!("Summary");
            println!("=======");
            println!(
                "Number of row categories I          = {}",
                result.row_labels.len()
            );
            println!(
                "Number of column categories J       = {}",
                result.col_labels.len()
            );
            println!("Total sample size N                 = {}", result.total);
            println!(
                "Chi-square statistic                = {:.10}",
                result.chi_square
            );
            println!(
                "Degrees of freedom                  = {}",
                result.degrees_of_freedom
            );
            println!("Asymptotic p-value                  = {:.10}", result.p_value);
            println!("Minimum expected count              = {:.10}", result.min_expected);
            println!(
                "Cells with expected count < 5       = {}",
                result.small_expected_cells
            );
            println!();

            println!("Cellwise Contributions");
            println!("======================");
            println!(
                "{:>12} {:>12} {:>12} {:>14} {:>14}",
                "Row", "Column", "Observed", "Expected", "Contribution"
            );
            for cell in &result.cells {
                println!(
                    "{:>12} {:>12} {:>12} {:>14.6} {:>14.6}",
                    cell.row_label,
                    cell.col_label,
                    cell.observed,
                    cell.expected,
                    cell.contribution
                );
            }
            println!();

            interpret_result(&result);
        }
        Err(err) => {
            eprintln!("Error: {}", err);
        }
    }
}

fn chi_square_test_of_independence(
    observations: &[(&str, &str)],
) -> Result<ChiSquareResult, String> {
    if observations.is_empty() {
        return Err("the observation list must be nonempty".to_string());
    }

    let row_labels = collect_sorted_labels(observations.iter().map(|(r, _)| *r));
    let col_labels = collect_sorted_labels(observations.iter().map(|(_, c)| *c));

    let row_index = build_index_map(&row_labels);
    let col_index = build_index_map(&col_labels);

    let i_dim = row_labels.len();
    let j_dim = col_labels.len();

    if i_dim < 2 || j_dim < 2 {
        return Err(
            "the contingency table must contain at least two row categories and two column categories"
                .to_string(),
        );
    }

    let mut observed = vec![vec![0u64; j_dim]; i_dim];

    for &(row, col) in observations {
        let i = *row_index
            .get(row)
            .ok_or_else(|| format!("unknown row label encountered: {}", row))?;
        let j = *col_index
            .get(col)
            .ok_or_else(|| format!("unknown column label encountered: {}", col))?;
        observed[i][j] += 1;
    }

    let mut row_totals = vec![0u64; i_dim];
    let mut col_totals = vec![0u64; j_dim];
    let mut total = 0u64;

    for i in 0..i_dim {
        for j in 0..j_dim {
            row_totals[i] += observed[i][j];
            col_totals[j] += observed[i][j];
            total += observed[i][j];
        }
    }

    if total == 0 {
        return Err("the total count must be positive".to_string());
    }

    let total_f = total as f64;
    let mut expected = vec![vec![0.0f64; j_dim]; i_dim];
    let mut cells = Vec::with_capacity(i_dim * j_dim);
    let mut chi_square = 0.0f64;
    let mut min_expected = f64::INFINITY;
    let mut small_expected_cells = 0usize;

    for i in 0..i_dim {
        for j in 0..j_dim {
            let e_ij = (row_totals[i] as f64) * (col_totals[j] as f64) / total_f;
            expected[i][j] = e_ij;

            if e_ij < min_expected {
                min_expected = e_ij;
            }
            if e_ij < 5.0 {
                small_expected_cells += 1;
            }

            let contribution = if e_ij > 0.0 {
                let diff = observed[i][j] as f64 - e_ij;
                diff * diff / e_ij
            } else {
                0.0
            };
            chi_square += contribution;

            cells.push(CellSummary {
                row_label: row_labels[i].clone(),
                col_label: col_labels[j].clone(),
                observed: observed[i][j],
                expected: e_ij,
                contribution,
            });
        }
    }

    let degrees_of_freedom = (i_dim - 1) * (j_dim - 1);
    let p_value = chi_square_survival_function(chi_square, degrees_of_freedom as f64);

    Ok(ChiSquareResult {
        row_labels,
        col_labels,
        observed,
        expected,
        row_totals,
        col_totals,
        total,
        chi_square,
        degrees_of_freedom,
        p_value,
        cells,
        min_expected,
        small_expected_cells,
    })
}

fn collect_sorted_labels<'a, I>(iter: I) -> Vec<String>
where
    I: Iterator<Item = &'a str>,
{
    let mut set = BTreeSet::new();
    for label in iter {
        set.insert(label.to_string());
    }
    set.into_iter().collect()
}

fn build_index_map(labels: &[String]) -> BTreeMap<String, usize> {
    let mut map = BTreeMap::new();
    for (idx, label) in labels.iter().enumerate() {
        map.insert(label.clone(), idx);
    }
    map
}

fn print_contingency_table(result: &ChiSquareResult) {
    println!("Observed Contingency Table N_ij");
    println!("===============================");

    print!("{:>12}", "");
    for col in &result.col_labels {
        print!("{:>12}", col);
    }
    print!("{:>12}", "Row Sum");
    println!();

    for i in 0..result.row_labels.len() {
        print!("{:>12}", result.row_labels[i]);
        for j in 0..result.col_labels.len() {
            print!("{:>12}", result.observed[i][j]);
        }
        print!("{:>12}", result.row_totals[i]);
        println!();
    }

    print!("{:>12}", "Col Sum");
    for total in &result.col_totals {
        print!("{:>12}", total);
    }
    print!("{:>12}", result.total);
    println!();
}

fn print_expected_table(result: &ChiSquareResult) {
    println!("Expected Counts E_ij Under Independence");
    println!("=======================================");

    print!("{:>12}", "");
    for col in &result.col_labels {
        print!("{:>12}", col);
    }
    println!();

    for i in 0..result.row_labels.len() {
        print!("{:>12}", result.row_labels[i]);
        for j in 0..result.col_labels.len() {
            print!("{:>12.4}", result.expected[i][j]);
        }
        println!();
    }
}

fn interpret_result(result: &ChiSquareResult) {
    println!("Interpretation");
    println!("==============");
    if result.p_value < 0.05 {
        println!(
            "At the conventional 5% level, the chi-square test provides evidence"
        );
        println!(
            "against the null hypothesis of independence between the two variables."
        );
    } else {
        println!(
            "At the conventional 5% level, the chi-square test does not provide"
        );
        println!(
            "strong evidence against the null hypothesis of independence."
        );
    }

    println!();
    if result.small_expected_cells > 0 {
        println!(
            "Because one or more expected counts are below 5, the asymptotic"
        );
        println!(
            "chi-square approximation should be interpreted cautiously."
        );
        println!(
            "In such cases, an exact or permutation-based calibration may be preferred."
        );
    } else {
        println!(
            "All expected counts are at least moderately large, so the asymptotic"
        );
        println!(
            "chi-square approximation is more likely to be reliable."
        );
    }
}

fn chi_square_survival_function(x: f64, dof: f64) -> f64 {
    if x < 0.0 || dof <= 0.0 {
        return f64::NAN;
    }
    regularized_gamma_q(0.5 * dof, 0.5 * x)
}

fn regularized_gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - regularized_gamma_p_series(a, x)
    } else {
        regularized_gamma_q_continued_fraction(a, x)
    }
}

fn regularized_gamma_p_series(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    let mut sum = 1.0 / a;
    let mut term = sum;
    let mut n = 1usize;

    loop {
        term *= x / (a + n as f64);
        sum += term;

        if term.abs() < sum.abs() * 1.0e-14 || n > 10_000 {
            break;
        }
        n += 1;
    }

    sum * (-x + a * x.ln() - gln).exp()
}

fn regularized_gamma_q_continued_fraction(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    let tiny = 1.0e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b.max(tiny);
    let mut h = d;

    for i in 1..=10_000 {
        let fi = i as f64;
        let an = -fi * (fi - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < 1.0e-14 {
            break;
        }
    }

    (-x + a * x.ln() - gln).exp() * h
}

fn ln_gamma(z: f64) -> f64 {
    // Lanczos approximation
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - ln_gamma(1.0 - z);
    }

    let z = z - 1.0;
    let mut x = COEFFS[0];
    for (i, coeff) in COEFFS.iter().enumerate().skip(1) {
        x += coeff / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}
```

Program 14.5.1 demonstrates a practical implementation of the chi-square test of independence for categorical data, directly reflecting the theoretical framework developed in Section 14.5. By comparing observed and expected counts across a contingency table, the method provides a global measure of association between two variables. The computation is efficient, requiring only aggregation of counts and evaluation of a summation over the table entries.

The inclusion of diagnostic checks for small expected counts highlights an important limitation of the chi-square approximation. When expected frequencies are low, the theoretical distribution of the test statistic may deviate from the chi-square form, motivating the use of alternative approaches such as exact tests or permutation methods. This reinforces the broader theme that statistical methods must be adapted to the structure and scale of the data.

The modular structure of the implementation allows it to be extended naturally to more advanced contingency table analyses, including measures of association strength, likelihood-ratio tests, and multi-dimensional tables. It also provides a foundation for integrating simulation-based methods, such as permutation or bootstrap techniques, which can improve inference in small-sample settings. Overall, the program illustrates how categorical data analysis can be carried out reliably and efficiently within a modern computational framework.

## 14.5.2. Measures of Association Strength

While the chi-square statistic detects dependence, it does not quantify the strength of association in an easily interpretable way, as it scales with sample size. In particular, even small deviations from independence can produce large values of $\chi^2$ when the sample size $N$ is large, making it difficult to compare results across datasets. To address this, normalized measures are used to rescale the statistic and provide a bounded measure of association.

Two common indices are Cramér’s $V$ and the contingency coefficient $C$:

$$
V = \sqrt{\frac{\chi^2}{N \, \min(I - 1, J - 1)}}, \qquad
C = \sqrt{\frac{\chi^2}{\chi^2 + N}} \tag{14.5.7}
$$

These measures adjust the chi-square statistic by incorporating the sample size and, in the case of Cramér’s $V$, the dimensions of the contingency table. As a result, they provide values that are more directly comparable across studies and table sizes.

Cramér’s $V$ ranges between 0 (no association) and 1 (perfect association in at least one cell). A value close to 0 indicates that the observed counts are close to what would be expected under independence, while values approaching 1 indicate increasingly strong departures from independence. The contingency coefficient $C$ also lies between 0 and an upper bound less than 1, which depends on the size of the table, meaning that its maximum attainable value varies with $I$ and $J$.

These measures are useful for comparing associations across datasets of similar structure, as they provide a standardized scale. However, they do not convey direction, since categorical variables are unordered, and they summarize association in a single number that may not reflect how the dependence is distributed across the table. As a result, their interpretation is often most meaningful at extreme values, where the presence or absence of association is clear. Modern approaches often complement them with information-theoretic quantities such as mutual information or entropy-based measures, which provide more interpretable notions of “information shared” between variables.

### Rust Implementation

Following the discussion in Section 14.5.2 on normalized measures of association strength, Program 14.5.2 provides a practical implementation of standardized indices derived from the chi-square statistic. While the chi-square test detects deviations from independence as described in Equation (14.5.5), its magnitude depends on the sample size and therefore does not directly reflect the strength of association. This program computes the contingency table, evaluates the chi-square statistic, and then rescales it using the formulas in Equation (14.5.7) to obtain Cramér’s $V$ and the contingency coefficient $C$. These measures provide bounded, interpretable summaries of association that are more suitable for comparing results across datasets and table sizes. The implementation demonstrates how normalized association measures can be integrated into the contingency-table framework developed in Section 14.5.

At the core of the implementation is the function `compute_association_measures`, which constructs the contingency table $N_{ij}$ defined in Equation (14.5.1) from the paired categorical observations. The function begins by identifying the distinct row and column categories and mapping them to indices, allowing the observed counts to be accumulated into a two-dimensional table. From this table, it computes the row totals $N_{i\cdot}$, column totals $N_{\cdot j}$, and the total sample size $N_{\cdot j}$ as described in Equation (14.5.2). These quantities are then used to compute expected counts according to Equation (14.5.4), which represents the null hypothesis of independence.

The chi-square statistic is evaluated using the same summation structure introduced in Equation (14.5.5). For each cell, the contribution $(N_{ij} - E_{ij})^2 / E_{ij}$ is computed and accumulated to produce the overall statistic. This step captures the global discrepancy between observed and expected counts across the contingency table.

Once the chi-square statistic has been computed, the program evaluates the normalized association measures defined in Equation (14.5.7). Cramér’s $V$ is obtained by scaling the chi-square statistic by the total sample size and the smaller of $I-1$ and $J-1$, ensuring that the resulting value lies between 0 and 1. The contingency coefficient $C$ is computed by normalizing the statistic using the expression $\chi^2 / (\chi^2 + N)$, which also yields a bounded measure of association, although its upper limit depends on the table dimensions. The implementation additionally computes a reference upper bound for $C$ in the case of square tables, providing context for interpreting its magnitude.

The structure `CellSummary` stores the observed counts, expected counts, and individual contributions for each cell, allowing detailed inspection of how different parts of the table contribute to the overall statistic. The structure `AssociationResult` aggregates the full set of results, including the contingency table, expected counts, chi-square statistic, and normalized measures. This organization ensures that the relationship between the theoretical quantities and their numerical implementation remains transparent.

The helper functions `collect_sorted_labels` and `build_index_map` ensure consistent indexing of categorical variables, while `print_observed_table` and `print_expected_table` provide formatted output for the contingency table and expected counts. The function `interpret_result` offers a qualitative interpretation of the computed association measures, emphasizing the bounded nature of Cramér’s $V$ and the dimension-dependent behavior of the contingency coefficient.

The `main` function demonstrates the full workflow by constructing a sample dataset, computing the contingency table and associated statistics, and printing both numerical and interpretive results. This example illustrates how normalized measures of association can be computed and interpreted alongside the chi-square statistic.

```rust
// Program 14.5.2: Measures of Association Strength for Contingency Tables
//
// Problem statement:
// Given paired categorical observations (X, Y), construct the contingency
// table N_ij, compute the chi-square statistic for independence, and then
// evaluate the normalized association measures
//
//     V = sqrt(chi^2 / (N * min(I - 1, J - 1)))
//     C = sqrt(chi^2 / (chi^2 + N))
//
// from Equation (14.5.7). Report the observed table, expected counts,
// chi-square statistic, and the resulting standardized measures of
// association strength.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
struct CellSummary {
    row_label: String,
    col_label: String,
    observed: u64,
    expected: f64,
    contribution: f64,
}

#[derive(Debug, Clone)]
struct AssociationResult {
    row_labels: Vec<String>,
    col_labels: Vec<String>,
    observed: Vec<Vec<u64>>,
    expected: Vec<Vec<f64>>,
    row_totals: Vec<u64>,
    col_totals: Vec<u64>,
    total: u64,
    chi_square: f64,
    cramers_v: f64,
    contingency_coefficient: f64,
    c_max_square_table: f64,
    cells: Vec<CellSummary>,
}

fn main() {
    // Example dataset:
    // X = education level, Y = employment status.
    let observations = vec![
        ("HighSchool", "Employed"),
        ("HighSchool", "Employed"),
        ("HighSchool", "Unemployed"),
        ("HighSchool", "Employed"),
        ("HighSchool", "PartTime"),
        ("College", "Employed"),
        ("College", "Employed"),
        ("College", "PartTime"),
        ("College", "Employed"),
        ("College", "Unemployed"),
        ("College", "Employed"),
        ("Graduate", "Employed"),
        ("Graduate", "Employed"),
        ("Graduate", "Employed"),
        ("Graduate", "PartTime"),
        ("Graduate", "Employed"),
        ("Graduate", "Employed"),
        ("Graduate", "Employed"),
        ("Graduate", "PartTime"),
        ("Graduate", "Employed"),
    ];

    println!("Measures of Association Strength");
    println!("================================");
    println!();

    match compute_association_measures(&observations) {
        Ok(result) => {
            print_observed_table(&result);
            println!();
            print_expected_table(&result);
            println!();

            println!("Summary");
            println!("=======");
            println!(
                "Number of row categories I          = {}",
                result.row_labels.len()
            );
            println!(
                "Number of column categories J       = {}",
                result.col_labels.len()
            );
            println!("Total sample size N                 = {}", result.total);
            println!(
                "Chi-square statistic                = {:.10}",
                result.chi_square
            );
            println!("Cramer's V                          = {:.10}", result.cramers_v);
            println!(
                "Contingency coefficient C           = {:.10}",
                result.contingency_coefficient
            );
            println!(
                "Reference upper bound for C (square table) = {:.10}",
                result.c_max_square_table
            );
            println!();

            println!("Cellwise Contributions");
            println!("======================");
            println!(
                "{:>14} {:>14} {:>12} {:>14} {:>14}",
                "Row", "Column", "Observed", "Expected", "Contribution"
            );
            for cell in &result.cells {
                println!(
                    "{:>14} {:>14} {:>12} {:>14.6} {:>14.6}",
                    cell.row_label,
                    cell.col_label,
                    cell.observed,
                    cell.expected,
                    cell.contribution
                );
            }
            println!();

            interpret_result(&result);
        }
        Err(err) => {
            eprintln!("Error: {}", err);
        }
    }
}

fn compute_association_measures(
    observations: &[(&str, &str)],
) -> Result<AssociationResult, String> {
    if observations.is_empty() {
        return Err("the observation list must be nonempty".to_string());
    }

    let row_labels = collect_sorted_labels(observations.iter().map(|(r, _)| *r));
    let col_labels = collect_sorted_labels(observations.iter().map(|(_, c)| *c));

    let row_index = build_index_map(&row_labels);
    let col_index = build_index_map(&col_labels);

    let i_dim = row_labels.len();
    let j_dim = col_labels.len();

    if i_dim < 2 || j_dim < 2 {
        return Err(
            "at least two row categories and two column categories are required".to_string(),
        );
    }

    let mut observed = vec![vec![0u64; j_dim]; i_dim];

    for &(row, col) in observations {
        let i = *row_index
            .get(row)
            .ok_or_else(|| format!("unknown row label encountered: {}", row))?;
        let j = *col_index
            .get(col)
            .ok_or_else(|| format!("unknown column label encountered: {}", col))?;
        observed[i][j] += 1;
    }

    let mut row_totals = vec![0u64; i_dim];
    let mut col_totals = vec![0u64; j_dim];
    let mut total = 0u64;

    for i in 0..i_dim {
        for j in 0..j_dim {
            row_totals[i] += observed[i][j];
            col_totals[j] += observed[i][j];
            total += observed[i][j];
        }
    }

    if total == 0 {
        return Err("the total count must be positive".to_string());
    }

    let total_f = total as f64;
    let mut expected = vec![vec![0.0f64; j_dim]; i_dim];
    let mut cells = Vec::with_capacity(i_dim * j_dim);
    let mut chi_square = 0.0f64;

    for i in 0..i_dim {
        for j in 0..j_dim {
            let e_ij = (row_totals[i] as f64) * (col_totals[j] as f64) / total_f;
            expected[i][j] = e_ij;

            let contribution = if e_ij > 0.0 {
                let diff = observed[i][j] as f64 - e_ij;
                diff * diff / e_ij
            } else {
                0.0
            };
            chi_square += contribution;

            cells.push(CellSummary {
                row_label: row_labels[i].clone(),
                col_label: col_labels[j].clone(),
                observed: observed[i][j],
                expected: e_ij,
                contribution,
            });
        }
    }

    let min_dim = (i_dim - 1).min(j_dim - 1) as f64;
    let cramers_v = if min_dim > 0.0 {
        (chi_square / (total_f * min_dim)).sqrt()
    } else {
        0.0
    };

    let contingency_coefficient = (chi_square / (chi_square + total_f)).sqrt();

    // A common reference bound when the table is square of order m:
    // C_max = sqrt((m - 1) / m).
    let m = i_dim.max(j_dim) as f64;
    let c_max_square_table = if m > 0.0 {
        ((m - 1.0) / m).sqrt()
    } else {
        0.0
    };

    Ok(AssociationResult {
        row_labels,
        col_labels,
        observed,
        expected,
        row_totals,
        col_totals,
        total,
        chi_square,
        cramers_v,
        contingency_coefficient,
        c_max_square_table,
        cells,
    })
}

fn collect_sorted_labels<'a, I>(iter: I) -> Vec<String>
where
    I: Iterator<Item = &'a str>,
{
    let mut set = BTreeSet::new();
    for label in iter {
        set.insert(label.to_string());
    }
    set.into_iter().collect()
}

fn build_index_map(labels: &[String]) -> BTreeMap<String, usize> {
    let mut map = BTreeMap::new();
    for (idx, label) in labels.iter().enumerate() {
        map.insert(label.clone(), idx);
    }
    map
}

fn print_observed_table(result: &AssociationResult) {
    println!("Observed Contingency Table N_ij");
    println!("===============================");

    print!("{:>14}", "");
    for col in &result.col_labels {
        print!("{:>14}", col);
    }
    print!("{:>14}", "Row Sum");
    println!();

    for i in 0..result.row_labels.len() {
        print!("{:>14}", result.row_labels[i]);
        for j in 0..result.col_labels.len() {
            print!("{:>14}", result.observed[i][j]);
        }
        print!("{:>14}", result.row_totals[i]);
        println!();
    }

    print!("{:>14}", "Col Sum");
    for total in &result.col_totals {
        print!("{:>14}", total);
    }
    print!("{:>14}", result.total);
    println!();
}

fn print_expected_table(result: &AssociationResult) {
    println!("Expected Counts Under Independence");
    println!("==================================");

    print!("{:>14}", "");
    for col in &result.col_labels {
        print!("{:>14}", col);
    }
    println!();

    for i in 0..result.row_labels.len() {
        print!("{:>14}", result.row_labels[i]);
        for j in 0..result.col_labels.len() {
            print!("{:>14.4}", result.expected[i][j]);
        }
        println!();
    }
}

fn interpret_result(result: &AssociationResult) {
    println!("Interpretation");
    println!("==============");
    println!(
        "Cramer's V rescales the chi-square statistic by the sample size and"
    );
    println!(
        "the smaller effective table dimension, producing a standardized"
    );
    println!(
        "association measure between 0 and 1."
    );
    println!();

    if result.cramers_v < 0.1 {
        println!("Here the value of Cramer's V suggests a very weak association.");
    } else if result.cramers_v < 0.3 {
        println!("Here the value of Cramer's V suggests a weak to moderate association.");
    } else if result.cramers_v < 0.5 {
        println!("Here the value of Cramer's V suggests a moderate association.");
    } else {
        println!("Here the value of Cramer's V suggests a strong association.");
    }

    println!();
    println!(
        "The contingency coefficient C provides an additional normalized summary,"
    );
    println!(
        "although its attainable maximum depends on the table dimensions."
    );
}
```

Program 14.5.2 demonstrates how the chi-square statistic can be transformed into standardized measures of association that are more interpretable and comparable across datasets. While the chi-square test provides a measure of statistical significance, its dependence on sample size limits its usefulness as an indicator of effect size. The normalized measures implemented here address this limitation by rescaling the statistic to produce bounded quantities that reflect the strength of association.

Cramér’s $V$ provides a particularly useful summary, as it adjusts for both sample size and table dimensions, yielding values that can be compared across studies. The contingency coefficient $C$, while also bounded, depends on the size of the table, highlighting the importance of contextual interpretation when using this measure. Together, these indices complement the chi-square test by providing additional insight into the magnitude of dependence.

The modular structure of the implementation allows it to be extended easily to incorporate additional measures of association, such as mutual information or entropy-based quantities, which offer alternative perspectives on dependence. This provides a foundation for more advanced analyses of categorical data, particularly in settings where both statistical significance and effect size must be considered simultaneously.

## 14.5.3. Asymmetric and Predictive Measures

In many applications, variables play asymmetric roles, such as predictor and outcome. In such settings, the interest lies not only in whether the variables are associated, but in how much knowledge of one variable improves the ability to predict the other. Proportional reduction in error (PRE) measures address this by quantifying the decrease in prediction error when information about one variable is taken into account.

A classical example is Goodman and Kruskal’s $\lambda$, which measures the reduction in prediction error of $Y$ given knowledge of $X$. Conceptually, this measure compares the number of prediction errors made when predicting $Y$ without using $X$ to the number of errors made when predictions are based on the information provided by $X$. The resulting value reflects the proportion by which prediction error is reduced, providing an interpretable measure of predictive improvement.

However, recent research highlights limitations of such measures. In particular, they may yield a value of zero even when dependence exists, if not all structural information in the contingency table is utilized. This occurs when the predictive rule used in the calculation does not fully capture the patterns of association present in the data, leading to an apparent lack of improvement despite underlying dependence.

To address this, improved PRE-based measures have been proposed that incorporate more detailed contingency-table information and avoid the “zero despite dependence” issue (Urasaki et al., 2025). These refinements aim to make better use of the available data structure, ensuring that the resulting measure more accurately reflects the strength of predictive relationships.

These developments reflect a broader trend toward association measures with clearer interpretations, such as percentage error reduction or variance explained in categorical settings. By expressing association in terms of predictive improvement, such measures provide a more intuitive understanding of how one variable informs the behavior of another in practical applications.

### Rust Implementation

Following the discussion in Section 14.5.3 on asymmetric and predictive measures, Program 14.5.3 provides a practical implementation of proportional reduction in error (PRE) measures for contingency tables. In many applications, categorical variables play asymmetric roles, such as predictor and response, making it important to quantify how much knowledge of one variable improves prediction of the other. This program constructs the contingency table from paired observations and evaluates Goodman and Kruskal’s $\lambda$, which measures the reduction in prediction error when conditioning on a predictor variable. To address the limitations of modal-based measures discussed in the section, the program also computes directional uncertainty coefficients, which incorporate the full conditional distributions. The implementation demonstrates how predictive improvement can be quantified numerically and interpreted in terms of directional association between categorical variables.

At the core of the implementation is the function `predictive_measures`, which constructs the contingency table $N_{ij}$ defined in Equation (14.5.1) from the paired categorical observations. The function begins by identifying the distinct categories for the predictor and response variables and mapping them to indices. Using this mapping, it accumulates the observed counts into a two-dimensional table and computes the corresponding row totals, column totals, and total sample size as described in Equation (14.5.2). These quantities provide the foundation for evaluating predictive performance both with and without conditioning.

The computation of Goodman and Kruskal’s $\lambda_{Y|X}$ follows the proportional reduction in error framework described in the section. The program first determines the globally modal category of $Y$, which defines the optimal prediction rule when $X$ is ignored. The number of baseline prediction errors is then given by the total sample size minus the frequency of this modal category. Next, for each row of the contingency table corresponding to a fixed value of $X$, the program identifies the rowwise modal category of $Y$. The sum of these rowwise maxima gives the number of correct predictions when $X$ is used. The reduction in prediction error is then normalized by the baseline error to produce $\lambda_{Y|X}$. The reverse-direction quantity $\lambda_{X|Y}$ is computed analogously by performing the same procedure across columns.

The helper function `argmax_u64` supports these computations by identifying the modal category within marginal or conditional distributions. Because $\lambda$ depends only on modal counts, it is computationally simple and directly interpretable as a fraction of error reduction. However, as noted in the section, this reliance on modal values may lead to a value of zero even when dependence exists, since nonmodal structure in the contingency table is ignored.

To address this limitation, the program also computes directional uncertainty coefficients $U(Y|X)$ and $U(X|Y)$, which are based on entropy and conditional entropy. The function `entropy_from_counts` computes the entropy of a discrete distribution, while the functions `conditional_entropy_y_given_x` and `conditional_entropy_x_given_y` evaluate the corresponding conditional entropies by averaging rowwise or columnwise entropies weighted by their probabilities. These quantities measure the reduction in uncertainty achieved by conditioning on the predictor variable and provide a more information-sensitive measure of association than $\lambda$.

The functions `collect_sorted_labels` and `build_index_map` ensure consistent indexing of categorical variables, while `print_contingency_table` presents the observed counts in a structured format. The `interpret_result` function summarizes the predictive implications of the computed measures, highlighting both the reduction in prediction error and the complementary role of entropy-based measures. The `main` function demonstrates the complete workflow on a representative dataset, illustrating how directional predictive relationships can be quantified and interpreted in practice.

```rust
// Program 14.5.3: Asymmetric and Predictive Measures for Contingency Tables
//
// Problem statement:
// Given paired categorical observations (X, Y), construct the contingency
// table N_ij and evaluate directional predictive association measures.
// The program computes Goodman and Kruskal's lambda for predicting Y from X,
// and also the reverse-direction quantity for predicting X from Y. Because
// lambda is based on modal prediction and may become zero even when dependence
// exists, the program also computes directional uncertainty coefficients
// U(Y|X) and U(X|Y), which use the full conditional distributions.
//
// The output reports the contingency table, marginal totals, baseline and
// conditional prediction errors, the directional lambda measures, and the
// corresponding uncertainty coefficients.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
struct PredictiveResult {
    row_labels: Vec<String>,
    col_labels: Vec<String>,
    observed: Vec<Vec<u64>>,
    row_totals: Vec<u64>,
    col_totals: Vec<u64>,
    total: u64,
    modal_y_label: String,
    modal_x_label: String,
    baseline_errors_y: u64,
    conditional_errors_y_given_x: u64,
    lambda_y_given_x: f64,
    baseline_errors_x: u64,
    conditional_errors_x_given_y: u64,
    lambda_x_given_y: f64,
    entropy_y: f64,
    conditional_entropy_y_given_x: f64,
    uncertainty_y_given_x: f64,
    entropy_x: f64,
    conditional_entropy_x_given_y: f64,
    uncertainty_x_given_y: f64,
}

fn main() {
    // Example dataset:
    // X = packaging color, Y = brand choice.
    let observations = vec![
        ("Red", "BrandA"),
        ("Red", "BrandA"),
        ("Red", "BrandB"),
        ("Red", "BrandA"),
        ("Red", "BrandC"),
        ("Blue", "BrandB"),
        ("Blue", "BrandB"),
        ("Blue", "BrandA"),
        ("Blue", "BrandB"),
        ("Blue", "BrandC"),
        ("Blue", "BrandB"),
        ("Green", "BrandC"),
        ("Green", "BrandC"),
        ("Green", "BrandB"),
        ("Green", "BrandC"),
        ("Green", "BrandA"),
        ("Green", "BrandC"),
        ("Green", "BrandB"),
        ("Yellow", "BrandA"),
        ("Yellow", "BrandA"),
        ("Yellow", "BrandB"),
        ("Yellow", "BrandA"),
        ("Yellow", "BrandA"),
        ("Yellow", "BrandC"),
    ];

    println!("Asymmetric and Predictive Measures");
    println!("==================================");
    println!();

    match predictive_measures(&observations) {
        Ok(result) => {
            print_contingency_table(&result);
            println!();

            println!("Prediction-Error Summaries");
            println!("==========================");
            println!(
                "Predicting Y without X: baseline errors              = {}",
                result.baseline_errors_y
            );
            println!(
                "Predicting Y with X: conditional errors              = {}",
                result.conditional_errors_y_given_x
            );
            println!(
                "Goodman-Kruskal lambda_Y|X                           = {:.10}",
                result.lambda_y_given_x
            );
            println!();
            println!(
                "Predicting X without Y: baseline errors              = {}",
                result.baseline_errors_x
            );
            println!(
                "Predicting X with Y: conditional errors              = {}",
                result.conditional_errors_x_given_y
            );
            println!(
                "Goodman-Kruskal lambda_X|Y                           = {:.10}",
                result.lambda_x_given_y
            );
            println!();

            println!("Information-Sensitive Directional Summaries");
            println!("===========================================");
            println!("Entropy H(Y)                               = {:.10}", result.entropy_y);
            println!(
                "Conditional entropy H(Y|X)                  = {:.10}",
                result.conditional_entropy_y_given_x
            );
            println!(
                "Uncertainty coefficient U(Y|X)              = {:.10}",
                result.uncertainty_y_given_x
            );
            println!();
            println!("Entropy H(X)                               = {:.10}", result.entropy_x);
            println!(
                "Conditional entropy H(X|Y)                  = {:.10}",
                result.conditional_entropy_x_given_y
            );
            println!(
                "Uncertainty coefficient U(X|Y)              = {:.10}",
                result.uncertainty_x_given_y
            );
            println!();

            interpret_result(&result);
        }
        Err(err) => {
            eprintln!("Error: {}", err);
        }
    }
}

fn predictive_measures(observations: &[(&str, &str)]) -> Result<PredictiveResult, String> {
    if observations.is_empty() {
        return Err("the observation list must be nonempty".to_string());
    }

    let row_labels = collect_sorted_labels(observations.iter().map(|(x, _)| *x));
    let col_labels = collect_sorted_labels(observations.iter().map(|(_, y)| *y));

    if row_labels.len() < 2 || col_labels.len() < 2 {
        return Err("at least two categories are required in both variables".to_string());
    }

    let row_index = build_index_map(&row_labels);
    let col_index = build_index_map(&col_labels);

    let i_dim = row_labels.len();
    let j_dim = col_labels.len();

    let mut observed = vec![vec![0u64; j_dim]; i_dim];
    for &(x, y) in observations {
        let i = *row_index
            .get(x)
            .ok_or_else(|| format!("unknown row label encountered: {}", x))?;
        let j = *col_index
            .get(y)
            .ok_or_else(|| format!("unknown column label encountered: {}", y))?;
        observed[i][j] += 1;
    }

    let mut row_totals = vec![0u64; i_dim];
    let mut col_totals = vec![0u64; j_dim];
    let mut total = 0u64;

    for i in 0..i_dim {
        for j in 0..j_dim {
            row_totals[i] += observed[i][j];
            col_totals[j] += observed[i][j];
            total += observed[i][j];
        }
    }

    if total == 0 {
        return Err("the total count must be positive".to_string());
    }

    // Goodman-Kruskal lambda for predicting Y from X.
    let (modal_y_idx, modal_y_count) = argmax_u64(&col_totals)
        .ok_or_else(|| "failed to determine modal category of Y".to_string())?;
    let baseline_errors_y = total - modal_y_count;

    let mut conditional_correct_y_given_x = 0u64;
    for row in &observed {
        let (_, row_max) = argmax_u64(row)
            .ok_or_else(|| "failed to determine rowwise modal category of Y".to_string())?;
        conditional_correct_y_given_x += row_max;
    }
    let conditional_errors_y_given_x = total - conditional_correct_y_given_x;
    let lambda_y_given_x = if baseline_errors_y > 0 {
        (baseline_errors_y - conditional_errors_y_given_x) as f64 / baseline_errors_y as f64
    } else {
        0.0
    };

    // Goodman-Kruskal lambda for predicting X from Y.
    let (modal_x_idx, modal_x_count) = argmax_u64(&row_totals)
        .ok_or_else(|| "failed to determine modal category of X".to_string())?;
    let baseline_errors_x = total - modal_x_count;

    let mut conditional_correct_x_given_y = 0u64;
    for j in 0..j_dim {
        let mut column = Vec::with_capacity(i_dim);
        for i in 0..i_dim {
            column.push(observed[i][j]);
        }
        let (_, col_max) = argmax_u64(&column)
            .ok_or_else(|| "failed to determine columnwise modal category of X".to_string())?;
        conditional_correct_x_given_y += col_max;
    }
    let conditional_errors_x_given_y = total - conditional_correct_x_given_y;
    let lambda_x_given_y = if baseline_errors_x > 0 {
        (baseline_errors_x - conditional_errors_x_given_y) as f64 / baseline_errors_x as f64
    } else {
        0.0
    };

    // Information-sensitive directional summaries.
    let entropy_y = entropy_from_counts(&col_totals);
    let conditional_entropy_y_given_x = conditional_entropy_y_given_x(&observed, &row_totals);
    let uncertainty_y_given_x = if entropy_y > 0.0 {
        (entropy_y - conditional_entropy_y_given_x) / entropy_y
    } else {
        0.0
    };

    let entropy_x = entropy_from_counts(&row_totals);
    let conditional_entropy_x_given_y = conditional_entropy_x_given_y(&observed, &col_totals);
    let uncertainty_x_given_y = if entropy_x > 0.0 {
        (entropy_x - conditional_entropy_x_given_y) / entropy_x
    } else {
        0.0
    };

    let modal_y_label = col_labels[modal_y_idx].clone();
    let modal_x_label = row_labels[modal_x_idx].clone();

    Ok(PredictiveResult {
        row_labels,
        col_labels,
        observed,
        row_totals,
        col_totals,
        total,
        modal_y_label,
        modal_x_label,
        baseline_errors_y,
        conditional_errors_y_given_x,
        lambda_y_given_x,
        baseline_errors_x,
        conditional_errors_x_given_y,
        lambda_x_given_y,
        entropy_y,
        conditional_entropy_y_given_x,
        uncertainty_y_given_x,
        entropy_x,
        conditional_entropy_x_given_y,
        uncertainty_x_given_y,
    })
}

fn collect_sorted_labels<'a, I>(iter: I) -> Vec<String>
where
    I: Iterator<Item = &'a str>,
{
    let mut set = BTreeSet::new();
    for label in iter {
        set.insert(label.to_string());
    }
    set.into_iter().collect()
}

fn build_index_map(labels: &[String]) -> BTreeMap<String, usize> {
    let mut map = BTreeMap::new();
    for (idx, label) in labels.iter().enumerate() {
        map.insert(label.clone(), idx);
    }
    map
}

fn argmax_u64(values: &[u64]) -> Option<(usize, u64)> {
    if values.is_empty() {
        return None;
    }
    let mut best_idx = 0usize;
    let mut best_val = values[0];
    for (idx, &value) in values.iter().enumerate().skip(1) {
        if value > best_val {
            best_idx = idx;
            best_val = value;
        }
    }
    Some((best_idx, best_val))
}

fn entropy_from_counts(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }

    let total_f = total as f64;
    let mut h = 0.0f64;
    for &count in counts {
        if count > 0 {
            let p = count as f64 / total_f;
            h -= p * p.ln();
        }
    }
    h
}

fn conditional_entropy_y_given_x(observed: &[Vec<u64>], row_totals: &[u64]) -> f64 {
    let total: u64 = row_totals.iter().sum();
    if total == 0 {
        return 0.0;
    }

    let total_f = total as f64;
    let mut h = 0.0f64;

    for (i, row) in observed.iter().enumerate() {
        let row_total = row_totals[i];
        if row_total == 0 {
            continue;
        }

        let weight = row_total as f64 / total_f;
        let row_entropy = entropy_from_counts(row);
        h += weight * row_entropy;
    }

    h
}

fn conditional_entropy_x_given_y(observed: &[Vec<u64>], col_totals: &[u64]) -> f64 {
    let total: u64 = col_totals.iter().sum();
    if total == 0 {
        return 0.0;
    }

    let total_f = total as f64;
    let i_dim = observed.len();
    let j_dim = if i_dim > 0 { observed[0].len() } else { 0 };
    let mut h = 0.0f64;

    for j in 0..j_dim {
        let col_total = col_totals[j];
        if col_total == 0 {
            continue;
        }

        let mut column = Vec::with_capacity(i_dim);
        for row in observed {
            column.push(row[j]);
        }

        let weight = col_total as f64 / total_f;
        let col_entropy = entropy_from_counts(&column);
        h += weight * col_entropy;
    }

    h
}

fn print_contingency_table(result: &PredictiveResult) {
    println!("Observed Contingency Table N_ij");
    println!("===============================");

    print!("{:>14}", "");
    for col in &result.col_labels {
        print!("{:>14}", col);
    }
    print!("{:>14}", "Row Sum");
    println!();

    for i in 0..result.row_labels.len() {
        print!("{:>14}", result.row_labels[i]);
        for j in 0..result.col_labels.len() {
            print!("{:>14}", result.observed[i][j]);
        }
        print!("{:>14}", result.row_totals[i]);
        println!();
    }

    print!("{:>14}", "Col Sum");
    for total in &result.col_totals {
        print!("{:>14}", total);
    }
    print!("{:>14}", result.total);
    println!();
}

fn interpret_result(result: &PredictiveResult) {
    println!("Interpretation");
    println!("==============");
    println!(
        "The modal category of Y without using X is {}, giving {} baseline errors.",
        result.modal_y_label, result.baseline_errors_y
    );
    println!(
        "Using X reduces this to {} errors, so lambda_Y|X = {:.10}.",
        result.conditional_errors_y_given_x, result.lambda_y_given_x
    );
    println!();

    println!(
        "The modal category of X without using Y is {}, giving {} baseline errors.",
        result.modal_x_label, result.baseline_errors_x
    );
    println!(
        "Using Y reduces this to {} errors, so lambda_X|Y = {:.10}.",
        result.conditional_errors_x_given_y, result.lambda_x_given_y
    );
    println!();

    println!("Because lambda depends only on modal prediction, it can understate");
    println!("association when nonmodal structure is present in the contingency table.");
    println!("The uncertainty coefficients U(Y|X) and U(X|Y) complement lambda by");
    println!("using the full conditional distributions rather than only the largest cell");
    println!("in each row or column.");
}
```

Program 14.5.3 demonstrates how asymmetric association between categorical variables can be quantified through predictive measures. Goodman and Kruskal’s (\\lambda) provides a simple and intuitive measure of error reduction, directly reflecting how much predictive accuracy improves when one variable is used to predict another. However, its reliance on modal categories limits its ability to capture more subtle forms of dependence.

The inclusion of uncertainty coefficients addresses this limitation by incorporating the full conditional distributions, thereby providing a more comprehensive measure of directional association. These entropy-based measures quantify the reduction in uncertainty rather than just prediction error, offering a richer interpretation of how information is shared between variables.

The implementation highlights the importance of choosing appropriate measures depending on the analytical objective. While PRE measures are well suited for interpretability in terms of prediction accuracy, information-theoretic measures provide deeper insight into the structure of dependence. Together, these approaches form a complementary toolkit for analyzing relationships in categorical data and illustrate the broader shift toward interpretable and application-driven measures of association.

## 14.5.4. Computational Aspects and Applications

Constructing the contingency table requires $O(N)$ operations, as each observation contributes to a count in exactly one cell of the table. This process involves identifying the appropriate category combination for each observation and incrementing the corresponding count, making it a single-pass operation over the data.

Computing the chi-square statistic requires $O(IJ)$ operations, where $I$ and $J$ are the numbers of categories for the two variables. Since this computation involves evaluating a fixed formula for each cell in the table, the cost depends only on the table size and is typically small when the number of categories is limited.

Permutation-based extensions introduce a factor of $B$, leading to $O(BN)$ complexity. Each permutation requires reassignment of labels and recomputation of the statistic across the dataset, so the total computational effort scales linearly with both the number of observations and the number of permutations. These computations are lightweight and easily parallelizable, as individual permutations can be processed independently.

### Illustrative Applications

In a medical study, one may analyze the relationship between genotype (categories $(A, B, C)$) and treatment outcome (success/failure). A contingency table is constructed by counting the number of patients in each genotype–outcome combination. The chi-square test then evaluates whether the observed distribution of outcomes differs from what would be expected if genotype and outcome were independent. A large value of Cramér’s $V$ would indicate a strong association, suggesting that genotype is predictive of treatment success. When counts are small, exact or permutation-based methods provide more reliable inference, as they do not rely on large-sample approximations.

In a marketing context, consider a survey where customers choose between brands after being exposed to different packaging colors. The variables are packaging color and brand choice. A contingency table summarizes how often each brand is chosen under each color condition, allowing one to assess whether brand preference depends on packaging color. Even when statistical significance is detected, effect size measures such as Cramér’s $V$ help determine whether the association is practically meaningful. For example, a small value of $V$ may indicate only a weak effect despite a large sample size, emphasizing the distinction between statistical significance and practical importance.

### Rust Implementation

Following the discussion in Section 14.5.4 on computational aspects and applications of contingency-table analysis, Program 14.5.4 provides a practical implementation that integrates table construction, chi-square evaluation, effect-size computation, and permutation-based calibration within a unified computational framework. As described in the section, contingency tables can be constructed in a single pass over the data, while statistical evaluation depends only on the table dimensions rather than the full dataset. This program reflects that structure by separating the linear-time data aggregation phase from the fixed-size table evaluation phase. It further incorporates permutation-based inference to address situations where asymptotic approximations may be unreliable, particularly when expected counts are small. The implementation demonstrates how efficient computation and statistical inference can be combined in practical applications such as medical studies and marketing analysis.

At the core of the implementation is the function `contingency_analysis`, which constructs the contingency table $N_{ij}$ defined in Equation (14.5.1) by processing each observation exactly once. This corresponds directly to the $O(N)$ complexity described in the section. For each observation, the program identifies the corresponding category pair and increments the appropriate cell in the table, thereby building the joint frequency structure required for further analysis. The function then computes the row totals $N_{i\cdot}$, column totals $N_{\cdot j}$, and total sample size $N$ as defined in Equation (14.5.2).

Once the contingency table has been constructed, the program evaluates the chi-square statistic using the formula given in Equation (14.5.5). This involves computing the expected counts according to Equation (14.5.4) and then accumulating the contributions $(N_{ij} - E_{ij})^2 / E_{ij}$ across all cells. Because this computation depends only on the table dimensions, it scales as $O(IJ)$, consistent with the discussion in the section. The implementation records the number of cell evaluations explicitly, illustrating the fixed cost associated with this stage.

The function also computes Cramér’s $V$ using Equation (14.5.7), providing a normalized measure of association strength. This allows the program to distinguish between statistical significance and practical importance, a distinction emphasized in the section. Additional diagnostics, such as the minimum expected count and the number of cells with expected counts below a threshold, are included to assess the reliability of asymptotic approximations.

The permutation-based extension is implemented in the function `permutation_test_independence`. This function reflects the $O(BN)$ complexity described in the section by repeatedly shuffling the labels of one variable and recomputing the chi-square statistic for each permutation. Each iteration requires rebuilding the contingency table and evaluating the statistic, making the total cost proportional to both the number of permutations $B$ and the sample size $N$. The resulting empirical p-value is computed as the fraction of permuted statistics that are at least as large as the observed statistic, providing a nonparametric measure of significance.

The supporting functions `collect_sorted_labels` and `build_index_map` ensure consistent indexing of categorical variables, while `shuffle_in_place` and the lightweight random number generator `SimpleRng` enable reproducible permutation sampling. The functions `print_observed_table` and `print_expected_table` present the results in a structured format, and `interpret_result` provides a qualitative summary that integrates statistical significance and effect size.

The `main` function demonstrates the full workflow on two representative applications discussed in the section. In the medical example, genotype and treatment outcome are analyzed to assess whether the distribution of outcomes depends on genotype. In the marketing example, packaging color and brand choice are examined to determine whether consumer preferences vary across conditions. These examples illustrate how the computational framework supports both statistical testing and practical interpretation in real-world scenarios.

```rust
// Program 14.5.4: Computational Aspects and Applications of Contingency Table Analysis
//
// Problem statement:
// Given paired categorical observations (X, Y), construct the contingency table
// in O(N) time, compute the chi-square statistic and Cramer's V from the table
// in O(IJ) time, and estimate a permutation p-value by repeatedly relabeling
// one variable under the null hypothesis of independence. The program
// demonstrates this workflow on two application-style datasets:
//   1. Medical study: genotype vs treatment outcome
//   2. Marketing study: packaging color vs brand choice
//
// The code reports observed counts, expected counts, chi-square statistic,
// degrees of freedom, asymptotic and permutation p-values, Cramer's V,
// and simple operation counts illustrating the main complexity terms.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
struct ContingencyAnalysis {
    row_labels: Vec<String>,
    col_labels: Vec<String>,
    observed: Vec<Vec<u64>>,
    expected: Vec<Vec<f64>>,
    row_totals: Vec<u64>,
    col_totals: Vec<u64>,
    total: u64,
    chi_square: f64,
    degrees_of_freedom: usize,
    asymptotic_p_value: f64,
    cramers_v: f64,
    min_expected: f64,
    cells_with_expected_below_5: usize,
    table_build_ops: usize,
    table_eval_ops: usize,
}

#[derive(Debug, Clone)]
struct PermutationResult {
    exceedances: usize,
    permutations: usize,
    p_value: f64,
    estimated_ops: usize,
}

#[derive(Debug, Clone)]
struct FullResult {
    title: String,
    analysis: ContingencyAnalysis,
    permutation: PermutationResult,
}

fn main() {
    let medical_data = vec![
        ("A", "Success"),
        ("A", "Success"),
        ("A", "Failure"),
        ("A", "Success"),
        ("A", "Failure"),
        ("A", "Success"),
        ("A", "Success"),
        ("B", "Success"),
        ("B", "Failure"),
        ("B", "Failure"),
        ("B", "Success"),
        ("B", "Failure"),
        ("B", "Failure"),
        ("B", "Success"),
        ("C", "Failure"),
        ("C", "Failure"),
        ("C", "Success"),
        ("C", "Failure"),
        ("C", "Failure"),
        ("C", "Failure"),
        ("C", "Success"),
        ("C", "Failure"),
    ];

    let marketing_data = vec![
        ("Red", "BrandA"),
        ("Red", "BrandA"),
        ("Red", "BrandB"),
        ("Red", "BrandA"),
        ("Red", "BrandC"),
        ("Red", "BrandA"),
        ("Blue", "BrandB"),
        ("Blue", "BrandB"),
        ("Blue", "BrandA"),
        ("Blue", "BrandB"),
        ("Blue", "BrandC"),
        ("Blue", "BrandB"),
        ("Blue", "BrandB"),
        ("Green", "BrandC"),
        ("Green", "BrandC"),
        ("Green", "BrandB"),
        ("Green", "BrandC"),
        ("Green", "BrandA"),
        ("Green", "BrandC"),
        ("Green", "BrandB"),
        ("Yellow", "BrandA"),
        ("Yellow", "BrandA"),
        ("Yellow", "BrandB"),
        ("Yellow", "BrandA"),
        ("Yellow", "BrandA"),
        ("Yellow", "BrandC"),
        ("Yellow", "BrandA"),
    ];

    let medical_result = analyze_with_permutation(
        "Medical Study: Genotype vs Treatment Outcome",
        &medical_data,
        5000,
        0x1357_2468_ACE0_BDF1,
    )
    .expect("medical analysis should succeed");

    let marketing_result = analyze_with_permutation(
        "Marketing Study: Packaging Color vs Brand Choice",
        &marketing_data,
        5000,
        0x2468_1357_BDF1_ACE0,
    )
    .expect("marketing analysis should succeed");

    print_full_result(&medical_result);
    println!();
    print_full_result(&marketing_result);
}

fn analyze_with_permutation(
    title: &str,
    observations: &[(&str, &str)],
    permutations: usize,
    seed: u64,
) -> Result<FullResult, String> {
    let analysis = contingency_analysis(observations)?;
    let permutation = permutation_test_independence(observations, permutations, seed)?;

    Ok(FullResult {
        title: title.to_string(),
        analysis,
        permutation,
    })
}

fn contingency_analysis(observations: &[(&str, &str)]) -> Result<ContingencyAnalysis, String> {
    if observations.is_empty() {
        return Err("observation list must be nonempty".to_string());
    }

    let row_labels = collect_sorted_labels(observations.iter().map(|(r, _)| *r));
    let col_labels = collect_sorted_labels(observations.iter().map(|(_, c)| *c));

    if row_labels.len() < 2 || col_labels.len() < 2 {
        return Err("at least two categories are required for each variable".to_string());
    }

    let row_index = build_index_map(&row_labels);
    let col_index = build_index_map(&col_labels);

    let i_dim = row_labels.len();
    let j_dim = col_labels.len();

    let mut observed = vec![vec![0u64; j_dim]; i_dim];
    let mut table_build_ops = 0usize;

    for &(row, col) in observations {
        let i = *row_index
            .get(row)
            .ok_or_else(|| format!("unknown row label encountered: {}", row))?;
        let j = *col_index
            .get(col)
            .ok_or_else(|| format!("unknown column label encountered: {}", col))?;
        observed[i][j] += 1;
        table_build_ops += 1;
    }

    let mut row_totals = vec![0u64; i_dim];
    let mut col_totals = vec![0u64; j_dim];
    let mut total = 0u64;

    for i in 0..i_dim {
        for j in 0..j_dim {
            row_totals[i] += observed[i][j];
            col_totals[j] += observed[i][j];
            total += observed[i][j];
        }
    }

    if total == 0 {
        return Err("total count must be positive".to_string());
    }

    let total_f = total as f64;
    let mut expected = vec![vec![0.0f64; j_dim]; i_dim];
    let mut chi_square = 0.0f64;
    let mut min_expected = f64::INFINITY;
    let mut cells_with_expected_below_5 = 0usize;
    let mut table_eval_ops = 0usize;

    for i in 0..i_dim {
        for j in 0..j_dim {
            let e_ij = (row_totals[i] as f64) * (col_totals[j] as f64) / total_f;
            expected[i][j] = e_ij;

            if e_ij < min_expected {
                min_expected = e_ij;
            }
            if e_ij < 5.0 {
                cells_with_expected_below_5 += 1;
            }

            if e_ij > 0.0 {
                let diff = observed[i][j] as f64 - e_ij;
                chi_square += diff * diff / e_ij;
            }

            table_eval_ops += 1;
        }
    }

    let degrees_of_freedom = (i_dim - 1) * (j_dim - 1);
    let asymptotic_p_value = chi_square_survival_function(chi_square, degrees_of_freedom as f64);

    let min_dim = (i_dim - 1).min(j_dim - 1) as f64;
    let cramers_v = if min_dim > 0.0 {
        (chi_square / (total_f * min_dim)).sqrt()
    } else {
        0.0
    };

    Ok(ContingencyAnalysis {
        row_labels,
        col_labels,
        observed,
        expected,
        row_totals,
        col_totals,
        total,
        chi_square,
        degrees_of_freedom,
        asymptotic_p_value,
        cramers_v,
        min_expected,
        cells_with_expected_below_5,
        table_build_ops,
        table_eval_ops,
    })
}

fn permutation_test_independence(
    observations: &[(&str, &str)],
    permutations: usize,
    seed: u64,
) -> Result<PermutationResult, String> {
    if observations.is_empty() {
        return Err("observation list must be nonempty".to_string());
    }
    if permutations == 0 {
        return Err("number of permutations must be positive".to_string());
    }

    let observed_analysis = contingency_analysis(observations)?;
    let observed_statistic = observed_analysis.chi_square;

    let x_values: Vec<&str> = observations.iter().map(|(x, _)| *x).collect();
    let mut y_values: Vec<&str> = observations.iter().map(|(_, y)| *y).collect();

    let n = observations.len();
    let mut rng = SimpleRng::new(seed);
    let mut exceedances = 0usize;

    for _ in 0..permutations {
        shuffle_in_place(&mut y_values, &mut rng);

        let permuted: Vec<(&str, &str)> = x_values
            .iter()
            .zip(y_values.iter())
            .map(|(x, y)| (*x, *y))
            .collect();

        let permuted_analysis = contingency_analysis(&permuted)?;
        if permuted_analysis.chi_square >= observed_statistic {
            exceedances += 1;
        }
    }

    let p_value = (exceedances as f64 + 1.0) / (permutations as f64 + 1.0);
    let estimated_ops = permutations * n;

    Ok(PermutationResult {
        exceedances,
        permutations,
        p_value,
        estimated_ops,
    })
}

fn collect_sorted_labels<'a, I>(iter: I) -> Vec<String>
where
    I: Iterator<Item = &'a str>,
{
    let mut set = BTreeSet::new();
    for label in iter {
        set.insert(label.to_string());
    }
    set.into_iter().collect()
}

fn build_index_map(labels: &[String]) -> BTreeMap<String, usize> {
    let mut map = BTreeMap::new();
    for (idx, label) in labels.iter().enumerate() {
        map.insert(label.clone(), idx);
    }
    map
}

fn print_full_result(result: &FullResult) {
    println!("{}", result.title);
    println!("{}", "=".repeat(result.title.len()));
    println!();

    print_observed_table(&result.analysis);
    println!();
    print_expected_table(&result.analysis);
    println!();

    println!("Summary");
    println!("=======");
    println!(
        "Number of row categories I              = {}",
        result.analysis.row_labels.len()
    );
    println!(
        "Number of column categories J           = {}",
        result.analysis.col_labels.len()
    );
    println!(
        "Total sample size N                     = {}",
        result.analysis.total
    );
    println!(
        "Chi-square statistic                    = {:.10}",
        result.analysis.chi_square
    );
    println!(
        "Degrees of freedom                      = {}",
        result.analysis.degrees_of_freedom
    );
    println!(
        "Asymptotic p-value                      = {:.10}",
        result.analysis.asymptotic_p_value
    );
    println!(
        "Permutation p-value                     = {:.10}",
        result.permutation.p_value
    );
    println!(
        "Cramer's V                              = {:.10}",
        result.analysis.cramers_v
    );
    println!(
        "Minimum expected count                  = {:.10}",
        result.analysis.min_expected
    );
    println!(
        "Cells with expected count < 5           = {}",
        result.analysis.cells_with_expected_below_5
    );
    println!();

    println!("Complexity Diagnostics");
    println!("======================");
    println!(
        "Observed table construction ops         = {}",
        result.analysis.table_build_ops
    );
    println!(
        "Observed table evaluation ops           = {}",
        result.analysis.table_eval_ops
    );
    println!(
        "Permutation relabeling count B          = {}",
        result.permutation.permutations
    );
    println!(
        "Estimated permutation-scale ops         = {}",
        result.permutation.estimated_ops
    );
    println!(
        "Exceedances in permutation test         = {}",
        result.permutation.exceedances
    );
    println!();

    interpret_result(result);
}

fn print_observed_table(analysis: &ContingencyAnalysis) {
    println!("Observed Contingency Table N_ij");
    println!("===============================");

    print!("{:>14}", "");
    for col in &analysis.col_labels {
        print!("{:>14}", col);
    }
    print!("{:>14}", "Row Sum");
    println!();

    for i in 0..analysis.row_labels.len() {
        print!("{:>14}", analysis.row_labels[i]);
        for j in 0..analysis.col_labels.len() {
            print!("{:>14}", analysis.observed[i][j]);
        }
        print!("{:>14}", analysis.row_totals[i]);
        println!();
    }

    print!("{:>14}", "Col Sum");
    for total in &analysis.col_totals {
        print!("{:>14}", total);
    }
    print!("{:>14}", analysis.total);
    println!();
}

fn print_expected_table(analysis: &ContingencyAnalysis) {
    println!("Expected Counts Under Independence");
    println!("==================================");

    print!("{:>14}", "");
    for col in &analysis.col_labels {
        print!("{:>14}", col);
    }
    println!();

    for i in 0..analysis.row_labels.len() {
        print!("{:>14}", analysis.row_labels[i]);
        for j in 0..analysis.col_labels.len() {
            print!("{:>14.4}", analysis.expected[i][j]);
        }
        println!();
    }
}

fn interpret_result(result: &FullResult) {
    println!("Interpretation");
    println!("==============");

    if result.permutation.p_value < 0.05 {
        println!(
            "The permutation test suggests that the two categorical variables are associated."
        );
    } else {
        println!(
            "The permutation test does not provide strong evidence against independence."
        );
    }

    if result.analysis.cramers_v < 0.1 {
        println!("Cramer's V indicates a very weak practical association.");
    } else if result.analysis.cramers_v < 0.3 {
        println!("Cramer's V indicates a weak to moderate practical association.");
    } else if result.analysis.cramers_v < 0.5 {
        println!("Cramer's V indicates a moderate practical association.");
    } else {
        println!("Cramer's V indicates a strong practical association.");
    }

    if result.analysis.cells_with_expected_below_5 > 0 {
        println!(
            "Because some expected counts are small, the permutation calibration is especially useful."
        );
    } else {
        println!(
            "Expected counts are reasonably large, so the asymptotic and permutation views can be compared directly."
        );
    }
}

fn chi_square_survival_function(x: f64, dof: f64) -> f64 {
    if x < 0.0 || dof <= 0.0 {
        return f64::NAN;
    }
    regularized_gamma_q(0.5 * dof, 0.5 * x)
}

fn regularized_gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - regularized_gamma_p_series(a, x)
    } else {
        regularized_gamma_q_continued_fraction(a, x)
    }
}

fn regularized_gamma_p_series(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    let mut sum = 1.0 / a;
    let mut term = sum;
    let mut n = 1usize;

    loop {
        term *= x / (a + n as f64);
        sum += term;

        if term.abs() < sum.abs() * 1.0e-14 || n > 10_000 {
            break;
        }
        n += 1;
    }

    sum * (-x + a * x.ln() - gln).exp()
}

fn regularized_gamma_q_continued_fraction(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    let tiny = 1.0e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b.max(tiny);
    let mut h = d;

    for i in 1..=10_000 {
        let fi = i as f64;
        let an = -fi * (fi - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < 1.0e-14 {
            break;
        }
    }

    (-x + a * x.ln() - gln).exp() * h
}

fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - ln_gamma(1.0 - z);
    }

    let z = z - 1.0;
    let mut x = COEFFS[0];
    for (i, coeff) in COEFFS.iter().enumerate().skip(1) {
        x += coeff / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

fn shuffle_in_place<T>(data: &mut [T], rng: &mut SimpleRng) {
    if data.len() <= 1 {
        return;
    }
    for i in (1..data.len()).rev() {
        let j = rng.next_usize(i + 1);
        data.swap(i, j);
    }
}

#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.state = x;
        x
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            0
        } else {
            (self.next_u64() % upper as u64) as usize
        }
    }
}
```

Program 14.5.4 demonstrates how contingency-table analysis can be implemented efficiently while supporting both classical and modern inference methods. The separation between data aggregation and table-based computation reflects the underlying complexity structure, allowing large datasets to be processed efficiently. The inclusion of permutation-based calibration provides a robust alternative to asymptotic methods, particularly in situations where expected counts are small.

The examples highlight the importance of combining statistical significance with measures of practical relevance. While hypothesis tests determine whether an association exists, effect-size measures such as Cramér’s $V$ quantify its magnitude, providing a more complete understanding of the relationship between variables. This distinction is essential in applied settings, where statistically significant results may not always correspond to meaningful effects.

The modular design of the implementation allows it to be extended to additional methods, such as exact tests, information-theoretic measures, or parallelized permutation procedures. This flexibility makes it a foundation for more advanced analyses of categorical data, bridging theoretical concepts and practical applications in modern statistical computing.

# 14.6. Linear Correlation

In many applications, the variables of interest are continuous or ordinal, and the goal is to quantify the strength of association between them. This involves assessing how changes in one variable are accompanied by changes in another, and whether such changes follow a consistent pattern. The most widely used measure is the Pearson product–moment correlation coefficient, which captures linear dependence between two variables by summarizing how closely their joint behavior aligns with a linear relationship.

Correlation plays a central role in numerical computing, underlying regression, principal component analysis, and many scientific interpretations. In regression, correlation reflects the extent to which one variable can be explained by another through a linear model. In principal component analysis, it is used to identify directions of maximum variation and dependence in multivariate data. In scientific contexts, correlation is often used to describe relationships between observed quantities, such as environmental variables, gene expression levels, or financial returns, where identifying patterns of co-variation is essential for analysis and interpretation.

However, modern practice emphasizes two important principles. First, correlation does not imply causation, meaning that the presence of a strong association between variables does not establish a causal relationship. Observed correlations may arise from indirect relationships, shared influences, or coincidental patterns. Second, even strong correlations require careful uncertainty quantification, as estimates based on finite samples may vary due to random fluctuations. Accordingly, both estimation and inference must be considered together when interpreting correlation measures, ensuring that reported values are accompanied by an assessment of their reliability and statistical significance.

## 14.6.1. Pearson Correlation Coefficient

Given paired observations ${(x_i, y_i)}{i=1}^{N}$, the Pearson correlation coefficient is defined as:

$$r = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{N} (x_i - \bar{x})^2} \, \sqrt{\sum_{i=1}^{N} (y_i - \bar{y})^2}} \tag{14.6.1}$$

This expression measures the extent to which the deviations of $x_i$ and $y_i$ from their respective means $\bar{x}$ and $\bar{y}$ vary together. The numerator aggregates the products of these deviations, reflecting whether the variables tend to increase and decrease together or in opposite directions. The denominator normalizes this quantity by the variability of each variable, ensuring that the resulting coefficient is scale-independent and comparable across different datasets.

The coefficient $r$ satisfies,

$$-1 \le r \le 1 \tag{14.6.2}$$

A value $r = 1$ indicates a perfect positive linear relationship, meaning that all points lie exactly on a line with positive slope. A value $r = -1$ indicates a perfect negative linear relationship, where all points lie on a line with negative slope. When $r \approx 0$, the deviations of the variables do not exhibit a consistent linear pattern, suggesting little or no linear association.

Despite its widespread use, correlation measures only linear dependence and may fail to detect nonlinear relationships. In particular, variables may be strongly related in a nonlinear manner while yielding a correlation close to zero. Moreover, interpretation must be cautious: a high correlation does not imply a causal relationship between variables. The coefficient summarizes association, but it does not distinguish between direct relationships and those arising from other factors or coincidental patterns.

### Rust Implementation

Following the discussion in Section 14.6 on linear correlation and the role of association measures in numerical computing, Program 14.6.1 provides a practical implementation of the Pearson product–moment correlation coefficient together with its computational evaluation using a numerically stable one-pass accumulation scheme. In large-scale data analysis, correlation must be computed efficiently while maintaining numerical reliability, particularly when dealing with streaming data or datasets with small variance. This program implements the formulation given in Equation (14.6.1) while avoiding catastrophic cancellation through incremental updates of means and centered sums. It demonstrates the evaluation of correlation across datasets exhibiting strong positive, strong negative, and weak linear relationships, thereby illustrating how correlation reflects patterns of co-variation in practice.

At the core of the implementation is the function `pearson_correlation`, which computes the coefficient defined in Equation (14.6.1) using a Welford-style online algorithm. Instead of computing means and deviations in separate passes, the function updates the sample means incrementally while simultaneously accumulating the centered sums required for the numerator and denominator. This approach improves numerical stability by avoiding subtraction of nearly equal quantities, which is especially important when the data have large magnitudes or small variance. The quantities `sum_sq_x`, `sum_sq_y`, and `sum_cov_xy` correspond directly to the components of the denominator and numerator in Equation (14.6.1), ensuring consistency between the mathematical formulation and its implementation.

The function also includes validation checks to ensure that the inputs are well-defined for correlation computation. It verifies that the input arrays have equal length and contain at least two observations, and it guards against degenerate cases where one of the variables has zero variance, in which case the coefficient is undefined. The final value of the correlation coefficient is clamped to the interval specified in Equation (14.6.2) to account for minor floating-point roundoff errors, ensuring that the computed result satisfies the theoretical bounds.

To facilitate interpretation and structured output, the program defines the `CorrelationSummary` struct, which aggregates all relevant computed quantities, including sample size, means, centered sums, and the final correlation coefficient. This design separates computation from presentation and allows the results to be reused in downstream analysis or extended to inference procedures discussed in Section 14.6.2.

The `print_dataset` function provides a formatted presentation of the input data and computed statistics. It outputs the paired observations, computed means, centered sums, and the resulting correlation coefficient, followed by a range check confirming consistency with Equation (14.6.2). It also includes a qualitative interpretation of the magnitude of the correlation, illustrating how numerical values translate into descriptive assessments of linear association.

The `main` function demonstrates the behavior of the correlation coefficient across three representative datasets. The first dataset exhibits a strong positive linear trend, leading to a correlation close to $1$. The second dataset shows a strong negative linear relationship, resulting in a correlation close to $-1$. The third dataset contains weakly varying values, producing a correlation near zero. These examples highlight the ability of the coefficient to distinguish between different types of linear dependence and reinforce the interpretation of correlation as a normalized measure of co-variation.

```rust
// Program 14.6.1: Pearson Correlation Coefficient
//
// Problem Statement:
// Implement the Pearson product–moment correlation coefficient for paired data
// {(x_i, y_i)}_{i=1}^N as defined in equation (14.6.1). The program should use
// a numerically stable one-pass accumulation strategy, verify the admissible
// range in equation (14.6.2), and demonstrate the computation on several sample
// datasets, including positive, negative, and weakly correlated cases.

#[derive(Debug, Clone, Copy)]
struct CorrelationSummary {
    n: usize,
    mean_x: f64,
    mean_y: f64,
    sum_sq_x: f64,
    sum_sq_y: f64,
    sum_cov_xy: f64,
    r: f64,
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<CorrelationSummary, String> {
    if x.len() != y.len() {
        return Err("Input slices must have the same length.".to_string());
    }
    if x.len() < 2 {
        return Err("At least two paired observations are required.".to_string());
    }

    // Welford-style stable online updates for means, sums of squares,
    // and cross-deviation accumulation.
    let mut n: usize = 0;
    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    let mut sum_sq_x = 0.0_f64;
    let mut sum_sq_y = 0.0_f64;
    let mut sum_cov_xy = 0.0_f64;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if !xi.is_finite() || !yi.is_finite() {
            return Err("All input values must be finite.".to_string());
        }

        n += 1;
        let n_f = n as f64;

        let dx = xi - mean_x;
        let dy = yi - mean_y;

        mean_x += dx / n_f;
        mean_y += dy / n_f;

        // After updating the means, accumulate centered sums.
        sum_sq_x += dx * (xi - mean_x);
        sum_sq_y += dy * (yi - mean_y);
        sum_cov_xy += dx * (yi - mean_y);
    }

    if sum_sq_x <= 0.0 {
        return Err("Correlation is undefined because x has zero variance.".to_string());
    }
    if sum_sq_y <= 0.0 {
        return Err("Correlation is undefined because y has zero variance.".to_string());
    }

    let denom = (sum_sq_x * sum_sq_y).sqrt();
    if denom == 0.0 {
        return Err("Correlation is undefined because the denominator is zero.".to_string());
    }

    // Small floating-point roundoff can push the value slightly outside [-1, 1].
    let r = (sum_cov_xy / denom).clamp(-1.0, 1.0);

    Ok(CorrelationSummary {
        n,
        mean_x,
        mean_y,
        sum_sq_x,
        sum_sq_y,
        sum_cov_xy,
        r,
    })
}

fn print_dataset(name: &str, x: &[f64], y: &[f64]) {
    println!("{name}");
    println!("{}", "=".repeat(name.len()));
    println!("Number of paired observations = {}", x.len());
    println!();

    println!("Data:");
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        println!("  i = {:2}, x = {:>12.6}, y = {:>12.6}", i, xi, yi);
    }
    println!();

    match pearson_correlation(x, y) {
        Ok(summary) => {
            println!("Computed Statistics");
            println!("-------------------");
            println!("n                 = {}", summary.n);
            println!("mean_x            = {:>.10}", summary.mean_x);
            println!("mean_y            = {:>.10}", summary.mean_y);
            println!("sum_sq_x          = {:>.10}", summary.sum_sq_x);
            println!("sum_sq_y          = {:>.10}", summary.sum_sq_y);
            println!("sum_cov_xy        = {:>.10}", summary.sum_cov_xy);
            println!("Pearson r         = {:>.10}", summary.r);
            println!();

            println!("Range Check");
            println!("-----------");
            println!(
                "-1 <= r <= 1      = {}",
                if (-1.0..=1.0).contains(&summary.r) {
                    "satisfied"
                } else {
                    "violated"
                }
            );
            println!();

            println!("Interpretation");
            println!("--------------");
            if summary.r > 0.9 {
                println!("Very strong positive linear association.");
            } else if summary.r > 0.6 {
                println!("Moderately strong positive linear association.");
            } else if summary.r > 0.2 {
                println!("Weak positive linear association.");
            } else if summary.r >= -0.2 {
                println!("Little or no evident linear association.");
            } else if summary.r >= -0.6 {
                println!("Weak negative linear association.");
            } else if summary.r >= -0.9 {
                println!("Moderately strong negative linear association.");
            } else {
                println!("Very strong negative linear association.");
            }
        }
        Err(err) => {
            println!("Correlation could not be computed.");
            println!("Reason: {err}");
        }
    }

    println!();
    println!("{}", "-".repeat(72));
    println!();
}

fn main() {
    let x_positive = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y_positive = vec![1.3, 2.1, 2.9, 4.2, 5.1, 5.8];

    let x_negative = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y_negative = vec![12.0, 10.3, 8.2, 6.4, 4.3, 2.1];

    let x_weak = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y_weak = vec![3.1, 2.7, 3.4, 2.9, 3.0, 3.3, 2.8, 3.2];

    println!("Pearson Product–Moment Correlation Coefficient");
    println!("==============================================");
    println!("This program evaluates the coefficient in equation (14.6.1)");
    println!("using a numerically stable one-pass accumulation scheme.");
    println!();

    print_dataset("Dataset A: Positive Association", &x_positive, &y_positive);
    print_dataset("Dataset B: Negative Association", &x_negative, &y_negative);
    print_dataset("Dataset C: Weak Association", &x_weak, &y_weak);
}
```

Program 14.6.1 demonstrates a numerically stable and computationally efficient approach to evaluating the Pearson correlation coefficient, aligning closely with the theoretical formulation in Section 14.6. The use of incremental updates ensures robustness in finite-precision arithmetic, while the structured design supports extensibility toward statistical inference, robust correlation measures, and large-scale data processing. The examples illustrate how correlation captures linear relationships while also emphasizing the importance of careful interpretation, particularly in the presence of weak or non-existent associations.

## 14.6.2. Statistical Inference for Correlation

To assess whether an observed correlation is statistically significant, one typically considers the null hypothesis of no linear association, meaning that the true correlation between the variables is zero. The goal is to determine whether the observed value of $r$ could plausibly arise from random variation alone, or whether it provides evidence of a genuine relationship.

Under the assumption that the data are independent and follow a bivariate normal distribution, the statistic:

$$t = r \sqrt{\frac{N - 2}{1 - r^2}} \tag{14.6.3}$$

follows a Student’s $t$-distribution with $N - 2$ degrees of freedom. This transformation rescales the correlation coefficient to account for sample size and variability, allowing it to be compared against a known reference distribution. Larger absolute values of $t$ correspond to stronger evidence against the null hypothesis.

This allows computation of $p$-values via:

$$P\bigl(|T| > |t|\bigr), \tag{14.6.4}$$

where $T$ is a $t$-distributed random variable. The $p$-value represents the probability of observing a test statistic at least as extreme as $t$ under the null hypothesis. A small $p$-value indicates that the observed correlation is unlikely to be due to chance alone, providing evidence of a nonzero linear association.

For large $N$, a useful approximation is obtained via Fisher’s $z$-transform:

$$z = \frac{1}{2} \ln\!\left(\frac{1 + r}{1 - r}\right) \tag{14.6.5}$$

which is approximately normally distributed with variance $1/(N - 3)$. This transformation stabilizes the variability of the correlation coefficient, making its distribution more symmetric and easier to approximate using the normal distribution. As a result, it enables the construction of confidence intervals and hypothesis tests for the true correlation in a straightforward manner.

These results rely on the assumption of bivariate normality. When this assumption is violated, the accuracy of parametric inference may degrade, as the sampling distribution of the statistic may differ from the theoretical form. In such cases, alternative approaches that are less sensitive to distributional assumptions may be preferred to ensure reliable inference.

### Rust Implementation

Following the discussion in Section 14.6.2 on statistical inference for correlation, Program 14.6.2 provides a practical implementation of hypothesis testing and confidence interval construction for the Pearson correlation coefficient. While the computation of the coefficient itself, as introduced in Equation (14.6.1), quantifies the strength of linear association, meaningful interpretation requires assessing whether the observed value could arise from random variation. This program implements the transformation in Equation (14.6.3) to obtain a test statistic, evaluates the corresponding two-sided p-value as described in Equation (14.6.4), and applies Fisher’s transformation in Equation (14.6.5) to construct confidence intervals. The implementation emphasizes numerical robustness and self-contained evaluation of special functions, enabling reliable inference without reliance on external libraries.

At the core of the implementation is the function `correlation_inference`, which orchestrates the computation of all inferential quantities from the paired data. It begins by invoking `pearson_correlation` to obtain the coefficient defined in Equation (14.6.1), ensuring that the statistical inference is built directly on the numerically stable accumulation of centered sums. The function `correlation_t_statistic` then computes the test statistic given in Equation (14.6.3), incorporating the dependence on sample size and variability. This transformation maps the correlation coefficient to a scale where standard distributional results apply, enabling hypothesis testing under the assumption of bivariate normality.

The evaluation of p-values is handled by the function `student_t_two_sided_p_value`, which computes the probability in Equation (14.6.4). This requires evaluating the cumulative distribution function of the Student’s t-distribution, which is implemented using the regularized incomplete beta function. The auxiliary functions `log_gamma`, `betacf`, and `regularized_incomplete_beta` provide a numerically stable and self-contained mechanism for computing this quantity. The use of logarithmic transformations and continued fraction expansions ensures stability across a wide range of parameter values, avoiding overflow and loss of precision that would otherwise occur in direct evaluations.

Fisher’s transformation, defined in Equation (14.6.5), is implemented in the function `fisher_z_transform`, which maps the bounded correlation coefficient to an approximately normally distributed variable. The inverse transformation is provided by `fisher_z_inverse`, enabling conversion back to the original scale after interval construction. The function `fisher_confidence_interval` uses this transformation to construct confidence intervals by combining the transformed value with a standard error proportional to $1/\sqrt{N-3}$. The critical value for the normal approximation is obtained via `inverse_standard_normal_cdf`, which implements a rational approximation for the inverse Gaussian distribution. This allows the program to compute confidence intervals without external dependencies while maintaining high numerical accuracy.

The `CorrelationInference` struct encapsulates all inferential outputs, including the test statistic, p-value, Fisher transform, standard error, and confidence interval bounds. This design promotes modularity and clarity, separating statistical computation from presentation. The `print_dataset` function formats and displays both the correlation summary and inferential results, providing a clear interpretation of statistical significance based on the computed p-value. It demonstrates how quantitative outputs translate into decisions about the null hypothesis of zero correlation.

The `main` function illustrates the behavior of the inference framework across datasets with varying levels of association. A strongly correlated dataset produces a large test statistic and a very small p-value, indicating strong evidence against the null hypothesis. A moderately correlated dataset yields a smaller but still significant test statistic, while a weakly correlated dataset produces a small statistic and a large p-value, indicating insufficient evidence to reject the null hypothesis. The corresponding confidence intervals further illustrate how uncertainty varies with the strength of the observed correlation and the sample size.

```rust
// Program 14.6.2: Statistical Inference for Correlation
//
// Problem Statement:
// Given paired observations {(x_i, y_i)}_{i=1}^N, compute the Pearson correlation
// coefficient r, the correlation t-statistic in equation (14.6.3), the associated
// two-sided p-value in equation (14.6.4), and Fisher's z-transform in equation
// (14.6.5). The program should also construct a confidence interval for the true
// correlation using Fisher's normal approximation. The implementation should be
// numerically stable, self-contained, and runnable with `cargo run` without
// requiring external crates.

#[derive(Debug, Clone, Copy)]
struct CorrelationSummary {
    n: usize,
    mean_x: f64,
    mean_y: f64,
    sum_sq_x: f64,
    sum_sq_y: f64,
    sum_cov_xy: f64,
    r: f64,
}

#[derive(Debug, Clone, Copy)]
struct CorrelationInference {
    r: f64,
    t_stat: f64,
    degrees_of_freedom: usize,
    p_value_two_sided: f64,
    fisher_z: f64,
    fisher_se: f64,
    confidence_level: f64,
    ci_lower: f64,
    ci_upper: f64,
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<CorrelationSummary, String> {
    if x.len() != y.len() {
        return Err("Input slices must have the same length.".to_string());
    }
    if x.len() < 2 {
        return Err("At least two paired observations are required.".to_string());
    }

    let mut n: usize = 0;
    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    let mut sum_sq_x = 0.0_f64;
    let mut sum_sq_y = 0.0_f64;
    let mut sum_cov_xy = 0.0_f64;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if !xi.is_finite() || !yi.is_finite() {
            return Err("All input values must be finite.".to_string());
        }

        n += 1;
        let n_f = n as f64;

        let dx = xi - mean_x;
        let dy = yi - mean_y;

        mean_x += dx / n_f;
        mean_y += dy / n_f;

        sum_sq_x += dx * (xi - mean_x);
        sum_sq_y += dy * (yi - mean_y);
        sum_cov_xy += dx * (yi - mean_y);
    }

    if sum_sq_x <= 0.0 {
        return Err("Correlation is undefined because x has zero variance.".to_string());
    }
    if sum_sq_y <= 0.0 {
        return Err("Correlation is undefined because y has zero variance.".to_string());
    }

    let denom = (sum_sq_x * sum_sq_y).sqrt();
    if denom == 0.0 {
        return Err("Correlation is undefined because the denominator is zero.".to_string());
    }

    let r = (sum_cov_xy / denom).clamp(-1.0, 1.0);

    Ok(CorrelationSummary {
        n,
        mean_x,
        mean_y,
        sum_sq_x,
        sum_sq_y,
        sum_cov_xy,
        r,
    })
}

fn correlation_t_statistic(r: f64, n: usize) -> Result<(f64, usize), String> {
    if n < 3 {
        return Err("At least three observations are required for the t-test.".to_string());
    }

    let dof = n - 2;
    let one_minus_r2 = 1.0 - r * r;

    if one_minus_r2 <= 0.0 {
        let t = if r.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        return Ok((t, dof));
    }

    let t = r * ((dof as f64) / one_minus_r2).sqrt();
    Ok((t, dof))
}

fn fisher_z_transform(r: f64) -> f64 {
    let r_clamped = r.clamp(-0.999_999_999_999, 0.999_999_999_999);
    0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln()
}

fn fisher_z_inverse(z: f64) -> f64 {
    z.tanh()
}

fn inverse_standard_normal_cdf(p: f64) -> Result<f64, String> {
    if !(0.0 < p && p < 1.0) {
        return Err("Probability must lie strictly between 0 and 1.".to_string());
    }

    // Acklam's rational approximation.
    let a: [f64; 6] = [
        -3.969_683_028_665_376e+01,
         2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
         1.383_577_518_672_690e+02,
        -3.066_479_806_614_716e+01,
         2.506_628_277_459_239e+00,
    ];
    let b: [f64; 5] = [
        -5.447_609_879_822_406e+01,
         1.615_858_368_580_409e+02,
        -1.556_989_798_598_866e+02,
         6.680_131_188_771_972e+01,
        -1.328_068_155_288_572e+01,
    ];
    let c: [f64; 6] = [
        -7.784_894_002_430_293e-03,
        -3.223_964_580_411_365e-01,
        -2.400_758_277_161_838e+00,
        -2.549_732_539_343_734e+00,
         4.374_664_141_464_968e+00,
         2.938_163_982_698_783e+00,
    ];
    let d: [f64; 4] = [
         7.784_695_709_041_462e-03,
         3.224_671_290_700_398e-01,
         2.445_134_137_142_996e+00,
         3.754_408_661_907_416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    let x = if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    };

    Ok(x)
}

fn fisher_confidence_interval(r: f64, n: usize, confidence_level: f64) -> Result<(f64, f64, f64), String> {
    if n < 4 {
        return Err("At least four observations are required for Fisher confidence intervals.".to_string());
    }
    if !(0.0 < confidence_level && confidence_level < 1.0) {
        return Err("Confidence level must lie strictly between 0 and 1.".to_string());
    }

    let alpha = 1.0 - confidence_level;
    let z_crit = inverse_standard_normal_cdf(1.0 - alpha / 2.0)?;
    let z = fisher_z_transform(r);
    let se = 1.0 / ((n as f64) - 3.0).sqrt();

    let lower_z = z - z_crit * se;
    let upper_z = z + z_crit * se;

    Ok((fisher_z_inverse(lower_z), fisher_z_inverse(upper_z), se))
}

fn log_gamma(z: f64) -> f64 {
    // Lanczos approximation.
    let coefficients: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
       -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
       -176.615_029_162_140_6,
        12.507_343_278_686_905,
       -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - log_gamma(1.0 - z);
    }

    let z_shifted = z - 1.0;
    let mut x = coefficients[0];
    for (i, c) in coefficients.iter().enumerate().skip(1) {
        x += c / (z_shifted + i as f64);
    }

    let t = z_shifted + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z_shifted + 0.5) * t.ln() - t + x.ln()
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 3.0e-14;
    let fpmin = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        let mut aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    h
}

fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> Result<f64, String> {
    if !(a > 0.0 && b > 0.0) {
        return Err("Parameters a and b must be positive.".to_string());
    }
    if !(0.0..=1.0).contains(&x) {
        return Err("x must satisfy 0 <= x <= 1.".to_string());
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if x == 1.0 {
        return Ok(1.0);
    }

    let ln_beta = log_gamma(a) + log_gamma(b) - log_gamma(a + b);
    let front = ((a * x.ln()) + (b * (1.0 - x).ln()) - ln_beta).exp();

    let value = if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    };

    Ok(value.clamp(0.0, 1.0))
}

fn student_t_two_sided_p_value(t: f64, dof: usize) -> Result<f64, String> {
    if dof == 0 {
        return Err("Degrees of freedom must be positive.".to_string());
    }
    if !t.is_finite() {
        return Ok(0.0);
    }

    let nu = dof as f64;
    let x = nu / (nu + t * t);
    let ibeta = regularized_incomplete_beta(0.5 * nu, 0.5, x)?;
    Ok(ibeta.clamp(0.0, 1.0))
}

fn correlation_inference(
    x: &[f64],
    y: &[f64],
    confidence_level: f64,
) -> Result<(CorrelationSummary, CorrelationInference), String> {
    let summary = pearson_correlation(x, y)?;
    let (t_stat, dof) = correlation_t_statistic(summary.r, summary.n)?;
    let p_value = student_t_two_sided_p_value(t_stat, dof)?;
    let fisher_z = fisher_z_transform(summary.r);
    let (ci_lower, ci_upper, fisher_se) =
        fisher_confidence_interval(summary.r, summary.n, confidence_level)?;

    let inference = CorrelationInference {
        r: summary.r,
        t_stat,
        degrees_of_freedom: dof,
        p_value_two_sided: p_value,
        fisher_z,
        fisher_se,
        confidence_level,
        ci_lower,
        ci_upper,
    };

    Ok((summary, inference))
}

fn print_dataset(name: &str, x: &[f64], y: &[f64], confidence_level: f64) {
    println!("{name}");
    println!("{}", "=".repeat(name.len()));
    println!("Number of paired observations = {}", x.len());
    println!();

    println!("Data:");
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        println!("  i = {:2}, x = {:>12.6}, y = {:>12.6}", i, xi, yi);
    }
    println!();

    match correlation_inference(x, y, confidence_level) {
        Ok((summary, inference)) => {
            println!("Correlation Summary");
            println!("-------------------");
            println!("n                 = {}", summary.n);
            println!("mean_x            = {:>.10}", summary.mean_x);
            println!("mean_y            = {:>.10}", summary.mean_y);
            println!("sum_sq_x          = {:>.10}", summary.sum_sq_x);
            println!("sum_sq_y          = {:>.10}", summary.sum_sq_y);
            println!("sum_cov_xy        = {:>.10}", summary.sum_cov_xy);
            println!("Pearson r         = {:>.10}", inference.r);
            println!();

            println!("Inference for Correlation");
            println!("-------------------------");
            println!("degrees_of_freedom = {}", inference.degrees_of_freedom);
            println!("t_statistic        = {:>.10}", inference.t_stat);
            println!("two_sided_p_value  = {:>.10}", inference.p_value_two_sided);
            println!("Fisher z           = {:>.10}", inference.fisher_z);
            println!("Fisher SE          = {:>.10}", inference.fisher_se);
            println!(
                "{:.1}% CI for rho   = [{:>.10}, {:>.10}]",
                100.0 * inference.confidence_level,
                inference.ci_lower,
                inference.ci_upper
            );
            println!();

            println!("Interpretation");
            println!("--------------");
            if inference.p_value_two_sided < 0.001 {
                println!("Very strong evidence against the null hypothesis of zero correlation.");
            } else if inference.p_value_two_sided < 0.01 {
                println!("Strong evidence against the null hypothesis of zero correlation.");
            } else if inference.p_value_two_sided < 0.05 {
                println!("Moderate evidence against the null hypothesis of zero correlation.");
            } else {
                println!("The data do not provide strong evidence against zero correlation.");
            }
        }
        Err(err) => {
            println!("Inference could not be completed.");
            println!("Reason: {err}");
        }
    }

    println!();
    println!("{}", "-".repeat(78));
    println!();
}

fn main() {
    let confidence_level = 0.95_f64;

    let x_strong = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y_strong = vec![1.2, 2.0, 3.1, 3.8, 5.2, 6.1, 6.9, 8.1];

    let x_moderate = vec![2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0];
    let y_moderate = vec![3.4, 2.8, 4.9, 5.5, 5.8, 6.7, 7.2, 7.0];

    let x_weak = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y_weak = vec![3.1, 2.7, 3.4, 2.9, 3.0, 3.3, 2.8, 3.2];

    println!("Statistical Inference for Correlation");
    println!("=====================================");
    println!("This program evaluates the correlation t-statistic in equation (14.6.3),");
    println!("the two-sided p-value in equation (14.6.4), and Fisher's z-transform in");
    println!("equation (14.6.5), together with a confidence interval for the true");
    println!("correlation based on Fisher's normal approximation.");
    println!();

    print_dataset("Dataset A: Strong Positive Correlation", &x_strong, &y_strong, confidence_level);
    print_dataset(
        "Dataset B: Moderate Positive Correlation",
        &x_moderate,
        &y_moderate,
        confidence_level,
    );
    print_dataset("Dataset C: Weak Correlation", &x_weak, &y_weak, confidence_level);
}
```

Program 14.6.2 demonstrates how statistical inference complements the computation of correlation by quantifying uncertainty and enabling hypothesis testing. By integrating numerically stable algorithms for both correlation and distributional evaluation, the program provides a comprehensive framework for assessing linear relationships in finite samples. It highlights the importance of combining estimation and inference, as emphasized in Section 14.6.2, and provides a foundation for extending analysis to robust or nonparametric methods when distributional assumptions are not satisfied.

## 14.6.3. Robust and Nonlinear Correlation Measures

Modern statistical practice extends beyond Pearson’s correlation to address non-normality, outliers, and nonlinear dependence. These extensions aim to provide measures of association that remain reliable when the assumptions underlying classical correlation are not satisfied, while also capturing relationships that are not purely linear.

### Rank-based Correlations

Spearman’s $\rho$ and Kendall’s $\tau$ measure monotonic relationships and are less sensitive to outliers. Instead of using the raw data values, these measures are based on the relative ordering of observations, which reduces the influence of extreme values and makes them more robust in the presence of irregular data. They are particularly useful when the relationship between variables is consistently increasing or decreasing, even if it is not linear. However, standard significance tests for these measures can be unreliable under non-normality, as their theoretical distributions may not accurately reflect the sampling behavior in such cases. Robust alternatives based on permutation tests provide better control of error rates, even for small samples (Yu and Hutson, 2024), by relying on empirical rather than theoretical distributions.

### Distance Correlation

Distance correlation is a modern measure that equals zero if and only if the variables are statistically independent, making it capable of detecting nonlinear relationships. This property distinguishes it from classical correlation measures, which may fail to capture certain forms of dependence. In practice, distance correlation evaluates how pairwise differences within each variable relate to one another, allowing it to detect associations beyond linear or monotonic patterns. While its naive computation requires $O(N^2)$ operations due to the need to consider all pairwise relationships, recent work provides improved estimators and more efficient implementations suitable for large datasets (Monroy-Castillo et al., 2024), making it increasingly practical in modern applications.

### Bootstrap Methods

Bootstrap resampling provides a flexible way to construct confidence intervals for correlation coefficients without relying on parametric assumptions. By repeatedly resampling the observed data and recomputing the correlation measure, one obtains an empirical distribution for $r$ or $\rho$. This distribution reflects the variability inherent in the data and allows direct estimation of uncertainty, such as confidence intervals. Because it does not depend on specific distributional assumptions, the bootstrap is particularly useful in situations where classical inference methods may be unreliable. Such methods are widely used in modern reproducible workflows (Cumming and Calin-Jageman, 2024), where transparent and data-driven uncertainty quantification is essential.

These approaches reflect a broader shift toward robust, assumption-light inference in statistical computing. By emphasizing methods that remain valid under weaker assumptions and that adapt to the structure of the data, modern practice seeks to provide more reliable and interpretable measures of association across a wide range of applications.

### Rust Implementation

Following the discussion in Section 14.6.3 on robust and nonlinear correlation measures, Program 14.6.3 provides a comprehensive implementation of rank-based correlations, distance correlation, and resampling-based inference techniques. While the Pearson coefficient introduced in Equation (14.6.1) captures linear dependence, it is sensitive to outliers and may fail to detect nonlinear relationships. This program extends the analysis by incorporating Spearman’s $\rho$ and Kendall’s $\tau$, which rely on ranks rather than raw values, as well as distance correlation, which is capable of detecting general statistical dependence. In addition, bootstrap confidence intervals and permutation tests are implemented to provide assumption-light inference, addressing the limitations of parametric methods discussed earlier. The program demonstrates how these tools complement classical correlation analysis in modern statistical computing.

At the core of the implementation are functions that compute alternative measures of association using transformations of the input data. The function `spearman_correlation` replaces the raw observations with their ranks via `average_ranks` and then applies the Pearson correlation framework from Equation (14.6.1), thereby measuring monotonic dependence rather than strictly linear association. The function `kendall_tau_b` computes Kendall’s $\tau$ by counting concordant and discordant pairs, providing a measure of ordinal association that is particularly robust to outliers and ties. These rank-based methods reflect the conceptual shift described in Section 14.6.3, where ordering rather than magnitude determines the strength of association.

The function `distance_correlation` implements a more general measure of dependence by constructing pairwise distance matrices and applying double-centering operations. This procedure captures the relationship between pairwise differences in the two variables, enabling detection of nonlinear dependencies that are invisible to classical correlation measures. The auxiliary functions `pairwise_distance_matrix` and `double_center` compute the required matrices and transformations, while the final normalization ensures that the resulting value lies between zero and one. Although the computation involves $O(N^2)$ operations, it provides a powerful tool for identifying general dependence structures.

To support inference without relying on distributional assumptions, the program implements bootstrap and permutation methods. The function `bootstrap_confidence_interval` generates resampled datasets and computes empirical confidence intervals for a given correlation measure, reflecting the variability of the statistic under repeated sampling. This approach aligns with the discussion of bootstrap methods in Section 14.6.3, where uncertainty is estimated directly from the data rather than from theoretical distributions. The function `permutation_test` constructs an empirical null distribution by randomly permuting one variable and recomputing the measure of association, enabling hypothesis testing based on the probability of observing a statistic as extreme as the original.

The program uses a simple pseudorandom number generator implemented in `SimpleRng` to ensure reproducibility of resampling procedures. The function `analyze_dataset` integrates all components by computing multiple association measures and their corresponding inferential summaries for a given dataset. The results are organized into structured outputs, including bootstrap confidence intervals and permutation-based p-values, allowing clear interpretation of the strength and significance of associations.

The `main` function demonstrates the behavior of these methods across three representative datasets. The first dataset exhibits a monotone nonlinear relationship, where rank-based measures detect perfect association while Pearson correlation remains slightly below one. The second dataset illustrates a nonlinear but nonmonotone relationship, where classical and rank-based measures fail to detect dependence, but distance correlation identifies a nonzero association. The third dataset introduces an outlier into an otherwise monotone relationship, showing how rank-based measures remain stable while Pearson correlation is affected. These examples highlight the advantages of robust and nonlinear measures in capturing a wider range of dependence structures.

```rust
// Program 14.6.3: Robust and Nonlinear Correlation Measures
//
// Problem Statement:
// Implement robust and nonlinear measures of association for paired data
// {(x_i, y_i)}_{i=1}^N. The program computes Spearman's rho and Kendall's tau
// as rank-based alternatives to Pearson correlation, evaluates distance
// correlation to detect nonlinear dependence, and applies permutation and
// bootstrap procedures to provide assumption-light inference. The
// implementation is self-contained, uses complete runnable Rust code, and
// demonstrates the behavior of these measures on representative datasets.

use std::cmp::Ordering;

#[derive(Debug, Clone)]
struct AssociationSummary {
    n: usize,
    pearson_r: f64,
    spearman_rho: f64,
    kendall_tau: f64,
    distance_correlation: f64,
}

#[derive(Debug, Clone)]
struct IntervalSummary {
    estimate: f64,
    lower: f64,
    upper: f64,
    confidence_level: f64,
}

#[derive(Debug, Clone)]
struct PermutationSummary {
    observed: f64,
    p_value_two_sided: f64,
    num_permutations: usize,
}

#[derive(Debug, Clone)]
struct AnalysisResult {
    summary: AssociationSummary,
    spearman_bootstrap_ci: IntervalSummary,
    distance_bootstrap_ci: IntervalSummary,
    spearman_permutation: PermutationSummary,
    distance_permutation: PermutationSummary,
}

#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let init = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: init }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64 generator.
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn gen_range_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % upper
    }

    fn shuffle<T>(&mut self, data: &mut [T]) {
        if data.len() <= 1 {
            return;
        }
        for i in (1..data.len()).rev() {
            let j = self.gen_range_usize(i + 1);
            data.swap(i, j);
        }
    }
}

fn validate_inputs(x: &[f64], y: &[f64]) -> Result<(), String> {
    if x.len() != y.len() {
        return Err("Input slices must have the same length.".to_string());
    }
    if x.len() < 2 {
        return Err("At least two paired observations are required.".to_string());
    }
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if !xi.is_finite() || !yi.is_finite() {
            return Err("All input values must be finite.".to_string());
        }
    }
    Ok(())
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64, String> {
    validate_inputs(x, y)?;

    let mut n = 0usize;
    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    let mut sum_sq_x = 0.0_f64;
    let mut sum_sq_y = 0.0_f64;
    let mut sum_cov_xy = 0.0_f64;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        n += 1;
        let n_f = n as f64;

        let dx = xi - mean_x;
        let dy = yi - mean_y;

        mean_x += dx / n_f;
        mean_y += dy / n_f;

        sum_sq_x += dx * (xi - mean_x);
        sum_sq_y += dy * (yi - mean_y);
        sum_cov_xy += dx * (yi - mean_y);
    }

    if sum_sq_x <= 0.0 || sum_sq_y <= 0.0 {
        return Err("Correlation is undefined because one variable has zero variance.".to_string());
    }

    Ok((sum_cov_xy / (sum_sq_x * sum_sq_y).sqrt()).clamp(-1.0, 1.0))
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let n = indexed.len();
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0usize;

    while i < n {
        let mut j = i + 1;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }

        let start_rank = i as f64 + 1.0;
        let end_rank = j as f64;
        let avg_rank = 0.5 * (start_rank + end_rank);

        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }

    ranks
}

fn spearman_correlation(x: &[f64], y: &[f64]) -> Result<f64, String> {
    validate_inputs(x, y)?;
    let rx = average_ranks(x);
    let ry = average_ranks(y);
    pearson_correlation(&rx, &ry)
}

fn signum_diff(a: f64, b: f64) -> i32 {
    if a > b {
        1
    } else if a < b {
        -1
    } else {
        0
    }
}

fn kendall_tau_b(x: &[f64], y: &[f64]) -> Result<f64, String> {
    validate_inputs(x, y)?;

    let n = x.len();
    let mut concordant = 0.0_f64;
    let mut discordant = 0.0_f64;
    let mut ties_x = 0.0_f64;
    let mut ties_y = 0.0_f64;

    for i in 0..n {
        for j in (i + 1)..n {
            let sx = signum_diff(x[i], x[j]);
            let sy = signum_diff(y[i], y[j]);

            match (sx, sy) {
                (0, 0) => {}
                (0, _) => ties_x += 1.0,
                (_, 0) => ties_y += 1.0,
                _ if sx == sy => concordant += 1.0,
                _ => discordant += 1.0,
            }
        }
    }

    let numerator = concordant - discordant;
    let denominator = ((concordant + discordant + ties_x) * (concordant + discordant + ties_y)).sqrt();

    if denominator == 0.0 {
        return Err("Kendall tau is undefined because all observations are tied.".to_string());
    }

    Ok((numerator / denominator).clamp(-1.0, 1.0))
}

fn pairwise_distance_matrix(values: &[f64]) -> Vec<Vec<f64>> {
    let n = values.len();
    let mut dist = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        for j in 0..n {
            dist[i][j] = (values[i] - values[j]).abs();
        }
    }

    dist
}

fn double_center(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut row_means = vec![0.0_f64; n];
    let mut col_means = vec![0.0_f64; n];
    let mut grand_mean = 0.0_f64;

    for i in 0..n {
        for j in 0..n {
            row_means[i] += matrix[i][j];
            col_means[j] += matrix[i][j];
            grand_mean += matrix[i][j];
        }
    }

    let n_f = n as f64;
    for i in 0..n {
        row_means[i] /= n_f;
        col_means[i] /= n_f;
    }
    grand_mean /= n_f * n_f;

    let mut centered = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            centered[i][j] = matrix[i][j] - row_means[i] - col_means[j] + grand_mean;
        }
    }

    centered
}

fn distance_correlation(x: &[f64], y: &[f64]) -> Result<f64, String> {
    validate_inputs(x, y)?;

    let ax = double_center(&pairwise_distance_matrix(x));
    let ay = double_center(&pairwise_distance_matrix(y));
    let n = x.len();
    let n2 = (n * n) as f64;

    let mut dcov2 = 0.0_f64;
    let mut dvarx2 = 0.0_f64;
    let mut dvary2 = 0.0_f64;

    for i in 0..n {
        for j in 0..n {
            dcov2 += ax[i][j] * ay[i][j];
            dvarx2 += ax[i][j] * ax[i][j];
            dvary2 += ay[i][j] * ay[i][j];
        }
    }

    dcov2 /= n2;
    dvarx2 /= n2;
    dvary2 /= n2;

    if dvarx2 <= 0.0 || dvary2 <= 0.0 {
        return Ok(0.0);
    }

    let dcov = dcov2.max(0.0).sqrt();
    let dvarx = dvarx2.sqrt();
    let dvary = dvary2.sqrt();

    if dvarx == 0.0 || dvary == 0.0 {
        return Ok(0.0);
    }

    let value = dcov / (dvarx * dvary).sqrt();
    Ok(value.clamp(0.0, 1.0))
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let q = p.clamp(0.0, 1.0);
    let pos = q * (sorted.len() as f64 - 1.0);
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;

    if lower == upper {
        sorted[lower]
    } else {
        let weight = pos - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn bootstrap_confidence_interval<F>(
    x: &[f64],
    y: &[f64],
    num_bootstrap: usize,
    confidence_level: f64,
    seed: u64,
    measure: F,
) -> Result<IntervalSummary, String>
where
    F: Fn(&[f64], &[f64]) -> Result<f64, String>,
{
    validate_inputs(x, y)?;
    if num_bootstrap == 0 {
        return Err("Number of bootstrap samples must be positive.".to_string());
    }
    if !(0.0 < confidence_level && confidence_level < 1.0) {
        return Err("Confidence level must lie strictly between 0 and 1.".to_string());
    }

    let estimate = measure(x, y)?;
    let n = x.len();
    let mut rng = SimpleRng::new(seed);
    let mut samples = Vec::with_capacity(num_bootstrap);

    for _ in 0..num_bootstrap {
        let mut xb = Vec::with_capacity(n);
        let mut yb = Vec::with_capacity(n);

        for _ in 0..n {
            let idx = rng.gen_range_usize(n);
            xb.push(x[idx]);
            yb.push(y[idx]);
        }

        samples.push(measure(&xb, &yb)?);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let alpha = 1.0 - confidence_level;
    let lower = percentile(&samples, alpha / 2.0);
    let upper = percentile(&samples, 1.0 - alpha / 2.0);

    Ok(IntervalSummary {
        estimate,
        lower,
        upper,
        confidence_level,
    })
}

fn permutation_test<F>(
    x: &[f64],
    y: &[f64],
    num_permutations: usize,
    seed: u64,
    measure: F,
) -> Result<PermutationSummary, String>
where
    F: Fn(&[f64], &[f64]) -> Result<f64, String>,
{
    validate_inputs(x, y)?;
    if num_permutations == 0 {
        return Err("Number of permutations must be positive.".to_string());
    }

    let observed = measure(x, y)?;
    let mut rng = SimpleRng::new(seed);
    let mut y_perm = y.to_vec();
    let mut extreme_count = 0usize;

    for _ in 0..num_permutations {
        rng.shuffle(&mut y_perm);
        let permuted = measure(x, &y_perm)?;
        if permuted.abs() >= observed.abs() {
            extreme_count += 1;
        }
    }

    let p_value = (extreme_count as f64 + 1.0) / (num_permutations as f64 + 1.0);

    Ok(PermutationSummary {
        observed,
        p_value_two_sided: p_value,
        num_permutations,
    })
}

fn analyze_dataset(
    x: &[f64],
    y: &[f64],
    confidence_level: f64,
    num_bootstrap: usize,
    num_permutations: usize,
    seed: u64,
) -> Result<AnalysisResult, String> {
    let summary = AssociationSummary {
        n: x.len(),
        pearson_r: pearson_correlation(x, y)?,
        spearman_rho: spearman_correlation(x, y)?,
        kendall_tau: kendall_tau_b(x, y)?,
        distance_correlation: distance_correlation(x, y)?,
    };

    let spearman_bootstrap_ci = bootstrap_confidence_interval(
        x,
        y,
        num_bootstrap,
        confidence_level,
        seed ^ 0xA5A5A5A5A5A5A5A5,
        spearman_correlation,
    )?;

    let distance_bootstrap_ci = bootstrap_confidence_interval(
        x,
        y,
        num_bootstrap,
        confidence_level,
        seed ^ 0x5A5A5A5A5A5A5A5A,
        distance_correlation,
    )?;

    let spearman_permutation = permutation_test(
        x,
        y,
        num_permutations,
        seed ^ 0x123456789ABCDEF0,
        spearman_correlation,
    )?;

    let distance_permutation = permutation_test(
        x,
        y,
        num_permutations,
        seed ^ 0x0FEDCBA987654321,
        distance_correlation,
    )?;

    Ok(AnalysisResult {
        summary,
        spearman_bootstrap_ci,
        distance_bootstrap_ci,
        spearman_permutation,
        distance_permutation,
    })
}

fn print_data(x: &[f64], y: &[f64]) {
    println!("Data:");
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        println!("  i = {:2}, x = {:>12.6}, y = {:>12.6}", i, xi, yi);
    }
    println!();
}

fn print_interval(label: &str, interval: &IntervalSummary) {
    println!("{label}");
    println!(
        "  estimate           = {:>.10}",
        interval.estimate
    );
    println!(
        "  {:.1}% bootstrap CI = [{:>.10}, {:>.10}]",
        100.0 * interval.confidence_level,
        interval.lower,
        interval.upper
    );
}

fn print_permutation(label: &str, summary: &PermutationSummary) {
    println!("{label}");
    println!("  observed           = {:>.10}", summary.observed);
    println!("  permutations       = {}", summary.num_permutations);
    println!("  two_sided_p_value  = {:>.10}", summary.p_value_two_sided);
}

fn interpret_permutation(p: f64) -> &'static str {
    if p < 0.001 {
        "Very strong evidence of association under permutation inference."
    } else if p < 0.01 {
        "Strong evidence of association under permutation inference."
    } else if p < 0.05 {
        "Moderate evidence of association under permutation inference."
    } else {
        "The permutation analysis does not provide strong evidence of association."
    }
}

fn print_analysis(
    name: &str,
    x: &[f64],
    y: &[f64],
    confidence_level: f64,
    num_bootstrap: usize,
    num_permutations: usize,
    seed: u64,
) {
    println!("{name}");
    println!("{}", "=".repeat(name.len()));
    println!("Number of paired observations = {}", x.len());
    println!();
    print_data(x, y);

    match analyze_dataset(x, y, confidence_level, num_bootstrap, num_permutations, seed) {
        Ok(result) => {
            println!("Association Measures");
            println!("--------------------");
            println!("n                    = {}", result.summary.n);
            println!("Pearson r            = {:>.10}", result.summary.pearson_r);
            println!("Spearman rho         = {:>.10}", result.summary.spearman_rho);
            println!("Kendall tau          = {:>.10}", result.summary.kendall_tau);
            println!(
                "Distance correlation = {:>.10}",
                result.summary.distance_correlation
            );
            println!();

            println!("Bootstrap Confidence Intervals");
            println!("------------------------------");
            print_interval("Spearman rho", &result.spearman_bootstrap_ci);
            print_interval("Distance correlation", &result.distance_bootstrap_ci);
            println!();

            println!("Permutation Tests");
            println!("-----------------");
            print_permutation("Spearman rho", &result.spearman_permutation);
            print_permutation("Distance correlation", &result.distance_permutation);
            println!();

            println!("Interpretation");
            println!("--------------");
            println!(
                "Spearman: {}",
                interpret_permutation(result.spearman_permutation.p_value_two_sided)
            );
            println!(
                "Distance correlation: {}",
                interpret_permutation(result.distance_permutation.p_value_two_sided)
            );
        }
        Err(err) => {
            println!("Analysis could not be completed.");
            println!("Reason: {err}");
        }
    }

    println!();
    println!("{}", "-".repeat(82));
    println!();
}

fn main() {
    let confidence_level = 0.95_f64;
    let num_bootstrap = 2000usize;
    let num_permutations = 2000usize;
    let seed = 20260404_u64;

    // Dataset A: monotone nonlinear association.
    let x_monotone = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y_monotone = vec![1.0, 1.5, 2.2, 3.3, 4.7, 6.5, 8.6, 11.0];

    // Dataset B: nonlinear but nonmonotone association.
    let x_nonlinear = vec![-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    let y_nonlinear: Vec<f64> = x_nonlinear.iter().map(|&v| v * v).collect();

    // Dataset C: contaminated monotone data with an outlier.
    let x_outlier = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let y_outlier = vec![1.2, 2.1, 2.9, 4.2, 5.0, 5.8, 7.1, 8.2, 20.0];

    println!("Robust and Nonlinear Correlation Measures");
    println!("=========================================");
    println!("This program evaluates rank-based measures, distance correlation,");
    println!("bootstrap confidence intervals, and permutation tests for paired data.");
    println!("It illustrates robust and nonlinear association analysis beyond");
    println!("classical Pearson correlation.");
    println!();

    print_analysis(
        "Dataset A: Monotone Nonlinear Association",
        &x_monotone,
        &y_monotone,
        confidence_level,
        num_bootstrap,
        num_permutations,
        seed ^ 0x1111111111111111,
    );

    print_analysis(
        "Dataset B: Nonlinear Nonmonotone Association",
        &x_nonlinear,
        &y_nonlinear,
        confidence_level,
        num_bootstrap,
        num_permutations,
        seed ^ 0x2222222222222222,
    );

    print_analysis(
        "Dataset C: Monotone Data with Outlier",
        &x_outlier,
        &y_outlier,
        confidence_level,
        num_bootstrap,
        num_permutations,
        seed ^ 0x3333333333333333,
    );
}
```

Program 14.6.3 demonstrates how modern correlation analysis extends beyond linear dependence by incorporating rank-based methods, distance-based measures, and resampling techniques. These approaches provide more reliable and interpretable results in the presence of outliers, non-normality, and nonlinear relationships. The modular design of the implementation allows these methods to be applied flexibly across different datasets and extended to more advanced techniques, supporting the broader goal of robust, assumption-light statistical analysis.

## 14.6.4. Computational Aspects and Applications

Computing Pearson’s correlation requires $O(N)$ operations, as it involves a single pass through the paired observations to accumulate the necessary sums. This efficiency arises from the fact that only basic arithmetic operations are required, making it suitable for large datasets and streaming contexts.

Fisher’s transformation and associated inference steps incur negligible additional cost, since they involve simple transformations and evaluations based on already computed quantities. These operations do not significantly increase the overall computational burden.

Permutation tests and bootstrap procedures scale as $O(BN)$, where $B$ is the number of resamples. Each resampling step requires recomputation of the correlation coefficient, so the total cost grows linearly with both the number of observations and the number of repetitions. This makes the choice of $B$ an important practical consideration, balancing computational effort with the accuracy of the resulting estimates.

Distance correlation typically requires $O(N^2)$, as it involves pairwise comparisons between observations. This quadratic scaling reflects the need to evaluate relationships across all pairs of data points. However, approximate methods can reduce this to near-linear complexity for large datasets, making such computations more feasible in practice.

These computations are well suited to vectorized and parallel implementations, as many of the required operations can be performed independently across data points or resamples. This enables efficient analysis of large datasets in modern numerical environments, where parallel processing and optimized numerical libraries are readily available.

### Illustrative Applications

In finance, correlation between asset returns is central to portfolio risk modeling. For example, if two assets exhibit a correlation of approximately $r \approx 0.6$, confidence intervals derived via Fisher’s transformation or bootstrap methods can be used to assess the stability of this estimate. This helps determine whether the observed relationship is consistent across different samples or subject to significant variability. Since financial returns often exhibit heavy tails, robust methods are frequently employed to validate conclusions, ensuring that the estimated correlations are not unduly influenced by extreme observations. Correlation directly affects portfolio variance, highlighting the importance of accurate estimation in risk management.

In biological applications, such as gene expression analysis, correlation is used to detect relationships between genes across large datasets. By comparing expression levels across samples, researchers can identify pairs of genes that exhibit coordinated behavior. A high Pearson correlation may indicate a linear association, while rank-based measures confirm monotonic relationships under non-normal distributions. Robust inference methods, including permutation tests, help ensure that the detected associations are reliable and not artifacts of distributional assumptions, thereby supporting more trustworthy scientific conclusions.

# 14.7. Rank Correlation and Information-Theoretic Measures

In numerical computing, we frequently analyze paired observations $\{(x_i, y_i)\}_{i=1}^N$, where the variables may arise from experiments, simulations, optimization loops, or surrogate models. In each of these settings, the data represent repeated evaluations of related quantities, and understanding how the two variables co-vary is essential for interpretation, validation, and model refinement. Correlation provides a compact summary of this dependence by reducing potentially complex relationships to a single scalar quantity that reflects the overall degree of association between the variables.

However, classical measures such as Pearson’s coefficient are designed specifically for linear relationships and therefore capture only a restricted form of dependence. When the relationship deviates from linearity, even in a systematic way, the resulting correlation value may underrepresent the true strength of association. In addition, these measures can be highly sensitive to outliers, heavy tails, or nonlinear (yet monotone) associations, as extreme values can disproportionately influence the computed coefficient and distort the summary of dependence.

Rank-based methods address these limitations by replacing raw values with their order structure, thereby focusing on the relative ranking of observations rather than their absolute magnitudes. This transformation preserves the essential ordering information while reducing sensitivity to extreme values and scale differences. As a result, rank-based approaches provide a more stable and robust characterization of dependence across a wider range of data conditions.

This approach is particularly natural in settings where preserving ordering is more important than preserving scale, such as surrogate modeling, non-Gaussian physical simulations, and ordinal data pipelines. In these contexts, the primary interest often lies in whether one variable tends to increase or decrease with another, rather than in the exact numerical relationship between them. At the same time, modern developments extend correlation analysis beyond monotonic dependence to general dependence measures grounded in information theory, allowing for a broader characterization of relationships that may not conform to simple linear or monotonic patterns.

## 14.7.1. Rank Transformations and Spearman Correlation

Let,

$$x = (x_1, \dots, x_N)^\top, \qquad y = (y_1, \dots, y_N)^\top \tag{14.7.1}$$

These vectors represent paired observations, where each index $i$ corresponds to a matched pair $(x_i, y_i)$. The goal is to transform these values into ranks so that the analysis depends on their relative ordering rather than their numerical magnitudes.

A sorting permutation $\pi_x$ reorders the data so that,

$$x_{\pi_x(1)} \le \cdots \le x_{\pi_x(N)} \tag{14.7.2}$$

This permutation identifies the indices that would arrange the data in nondecreasing order. In effect, it provides a mapping from the original positions of the observations to their positions in the sorted sequence.

The rank transformation assigns to each observation a rank $R_i$, defined as the position of $x_i$ in the sorted order. This replaces each value with an integer or fractional index that reflects its relative standing among all observations. When ties occur, the standard convention is to assign midranks: if a tied block occupies positions $(a, \dots, b)$, each element receives rank:

$$\frac{a + b}{2} \tag{14.7.3}$$

This preserves the total rank sum and yields a well-defined rank variable even in the presence of ties. By assigning equal ranks within tied groups, the transformation maintains consistency and avoids artificially breaking ties.

Spearman’s rank correlation coefficient is defined as the Pearson correlation of the rank variables:

$$\rho_S = \frac{\sum_{i=1}^{N} (R_i - \bar{R})(S_i - \bar{S})}{\sqrt{\sum_{i=1}^{N} (R_i - \bar{R})^2} \, \sqrt{\sum_{i=1}^{N} (S_i - \bar{S})^2}} \tag{14.7.4}$$

where $R_i = \text{rank}(x_i)$ and $S_i = \text{rank}(y_i)$. Thus, the computation proceeds by first transforming both variables into ranks and then applying the standard correlation formula to these transformed values.

In the absence of ties, a useful shortcut is:

$$\rho_S = 1 - \frac{6 \sum_{i=1}^{N} d_i^2}{N(N^2 - 1)}, \qquad d_i = R_i - S_i \tag{14.7.5}$$

This expression shows that the correlation depends directly on the differences between corresponding ranks. Small values of $d_i$ indicate agreement in ordering between the two variables, leading to values of $\rho_S$ close to $1$, while larger discrepancies reduce the correlation.

Spearman’s correlation is invariant under monotone transformations and can be interpreted as the correlation between marginal cumulative distribution values, explaining its robustness for non-Gaussian data (Tu et al., 2025). Because the transformation depends only on order, any monotonic rescaling of the data leaves the ranks unchanged and therefore does not affect the computed coefficient.

The computational cost is dominated by sorting, yielding,

$$\text{time } = O(N \log N), \qquad \text{space } = O(N) \tag{14.7.6}$$

The sorting step is required to determine the ranks, after which the remaining computations involve linear-time operations over the ranked data.

### Rust Implementation

Following the discussion in Section 14.7.1 on rank transformations and Spearman correlation, Program 14.7.1 provides a practical implementation of rank-based dependence analysis using midranks and correlation of transformed variables. In numerical computing, datasets often contain ties, nonlinear relationships, and non-Gaussian behavior, making classical correlation measures unreliable. This program implements the rank transformation described in equation (14.7.3) and evaluates Spearman’s coefficient via the Pearson correlation of ranks as defined in equation (14.7.4). By separating rank construction from correlation evaluation, the implementation mirrors the theoretical structure of the method while ensuring numerical robustness in practical data settings.

At the core of the implementation is the `midranks` function, which performs the rank transformation by sorting the data and assigning averaged positions to tied values. This directly realizes the definition of midranks given in equation (14.7.3), where all elements in a tied block receive the same fractional rank. The use of sorting reflects the permutation-based formulation in equation (14.7.2), ensuring that the transformation depends only on ordering and not on numerical magnitude. This design guarantees invariance under monotone transformations, a key theoretical property emphasized in the section.

The computation of Spearman’s coefficient is carried out by the `pearson_correlation` function applied to the rank vectors. This corresponds exactly to equation (14.7.4), where the covariance and variance of the ranks are accumulated in a numerically stable, single-pass manner. The function returns an optional value to handle degenerate cases where variance vanishes, ensuring robustness in edge cases such as constant data. The implementation therefore maintains a clear correspondence between the mathematical formulation and its computational realization.

The helper functions `has_ties` and `spearman_shortcut_no_ties` provide additional structure aligned with equation (14.7.5). The shortcut formula is conditionally evaluated only when no ties are present, reflecting its theoretical restriction. This explicit check reinforces the distinction between general and simplified formulas and ensures that incorrect assumptions are not silently introduced into the computation.

The `transformed_example` function demonstrates the invariance of Spearman’s correlation under monotone transformations by applying simple increasing mappings to the data and recomputing the coefficient. Because ranks depend only on ordering, the resulting value remains unchanged, providing a direct computational validation of the theoretical property discussed in the section. The `print_rank_table` and `print_vector` functions support interpretability by displaying both the original data and their corresponding ranks, making the transformation process transparent.

The `main` function illustrates the behavior of the algorithm in two representative scenarios. The first example includes ties, demonstrating the necessity of midranks and the inapplicability of the shortcut formula. The second example uses strictly increasing data, where both the general and shortcut formulas agree exactly, confirming the correctness of the implementation. Together, these examples highlight the robustness and flexibility of rank-based correlation in handling diverse data conditions.

```rust
// Program 14.7.1
// Rank Transformations and Spearman Correlation
//
// Problem statement:
// Given paired observations x = (x_1, ..., x_N)^T and y = (y_1, ..., y_N)^T,
// compute the rank vectors R and S using midranks for tied values, then
// evaluate Spearman's rank correlation coefficient as the Pearson correlation
// of the rank variables. This program also checks whether the shortcut formula
// based on rank differences agrees with the general formula when no ties occur.

use std::cmp::Ordering;

/// Small tolerance for floating-point tie detection in example data.
/// In purely exact data pipelines, this could be replaced by exact equality.
const EPS: f64 = 1.0e-12;

/// Compute the arithmetic mean of a slice.
fn mean(values: &[f64]) -> f64 {
    let n = values.len();
    assert!(n > 0, "mean requires at least one value");
    values.iter().sum::<f64>() / n as f64
}

/// Compute the Pearson correlation coefficient between two equal-length slices.
/// Returns None if either input has zero variance.
fn pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    assert_eq!(x.len(), y.len(), "Input slices must have the same length");
    let n = x.len();
    assert!(n > 0, "Correlation requires at least one observation");

    let mx = mean(x);
    let my = mean(y);

    let mut num = 0.0_f64;
    let mut den_x = 0.0_f64;
    let mut den_y = 0.0_f64;

    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    let den = (den_x * den_y).sqrt();
    if den <= EPS {
        None
    } else {
        Some(num / den)
    }
}

/// Return true if two floating-point values should be treated as tied.
fn nearly_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= EPS
}

/// Compute midranks for a real-valued vector.
///
/// If a tied block occupies sorted positions a..=b in 1-based indexing,
/// every member receives rank (a + b) / 2, matching equation (14.7.3).
fn midranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    assert!(n > 0, "midranks requires at least one value");

    // Pair each value with its original index.
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();

    // Sort by value.
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut ranks = vec![0.0_f64; n];
    let mut start = 0usize;

    while start < n {
        let mut end = start;
        while end + 1 < n && nearly_equal(indexed[end + 1].1, indexed[start].1) {
            end += 1;
        }

        // Convert zero-based positions start..=end to one-based ranks.
        let a = start as f64 + 1.0;
        let b = end as f64 + 1.0;
        let midrank = 0.5 * (a + b);

        for k in start..=end {
            let original_index = indexed[k].0;
            ranks[original_index] = midrank;
        }

        start = end + 1;
    }

    ranks
}

/// Check whether a vector contains ties.
fn has_ties(values: &[f64]) -> bool {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    for i in 1..sorted.len() {
        if nearly_equal(sorted[i - 1], sorted[i]) {
            return true;
        }
    }
    false
}

/// Compute Spearman's rank correlation by:
/// 1. transforming x and y into midranks,
/// 2. applying Pearson correlation to the rank vectors.
fn spearman_correlation(x: &[f64], y: &[f64]) -> Option<(f64, Vec<f64>, Vec<f64>)> {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");
    let rx = midranks(x);
    let ry = midranks(y);
    let rho = pearson_correlation(&rx, &ry)?;
    Some((rho, rx, ry))
}

/// Compute the no-ties shortcut:
/// rho_S = 1 - 6 sum d_i^2 / (N (N^2 - 1))
/// This is valid only when there are no ties in either variable.
fn spearman_shortcut_no_ties(x: &[f64], y: &[f64]) -> Option<f64> {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");
    let n = x.len();
    if n == 0 || has_ties(x) || has_ties(y) {
        return None;
    }

    let rx = midranks(x);
    let ry = midranks(y);

    let sum_d2 = rx
        .iter()
        .zip(ry.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f64>();

    let n_f = n as f64;
    Some(1.0 - 6.0 * sum_d2 / (n_f * (n_f * n_f - 1.0)))
}

/// Print a vector with aligned formatting.
fn print_vector(label: &str, values: &[f64]) {
    println!("{label}");
    for (i, v) in values.iter().enumerate() {
        println!("  [{:>2}] {:>.10}", i, v);
    }
    println!();
}

/// Print observation pairs together with their ranks.
fn print_rank_table(x: &[f64], y: &[f64], rx: &[f64], ry: &[f64]) {
    println!("Paired Data and Corresponding Midranks");
    println!("=====================================");
    println!(
        "{:>5} {:>16} {:>16} {:>16} {:>16}",
        "i", "x_i", "y_i", "R_i", "S_i"
    );
    for i in 0..x.len() {
        println!(
            "{:>5} {:>16.10} {:>16.10} {:>16.10} {:>16.10}",
            i, x[i], y[i], rx[i], ry[i]
        );
    }
    println!();
}

/// Demonstrate monotone invariance by applying monotone transformations
/// and recomputing Spearman's correlation.
fn transformed_example(x: &[f64], y: &[f64]) {
    let x_monotone: Vec<f64> = x.iter().map(|&v| 3.0 * v + 7.0).collect();
    let y_monotone: Vec<f64> = y.iter().map(|&v| v.exp()).collect();

    let original = spearman_correlation(x, y).map(|t| t.0);
    let transformed = spearman_correlation(&x_monotone, &y_monotone).map(|t| t.0);

    println!("Monotone-Transformation Check");
    println!("=============================");
    match (original, transformed) {
        (Some(r1), Some(r2)) => {
            println!("rho_S(original data)        = {:>.10}", r1);
            println!("rho_S(monotone transforms)  = {:>.10}", r2);
            println!("absolute difference         = {:>.10e}", (r1 - r2).abs());
        }
        _ => {
            println!("Unable to evaluate monotone-invariance check.");
        }
    }
    println!();
}

fn main() {
    // Example 1: data with ties, so the general rank-correlation formula is required.
    let x_tied = vec![10.0, 20.0, 20.0, 40.0, 50.0, 50.0, 50.0, 80.0];
    let y_tied = vec![15.0, 18.0, 22.0, 35.0, 35.0, 60.0, 55.0, 90.0];

    println!("Section 14.7.1 Example A: Spearman Correlation with Ties");
    println!("========================================================");
    print_vector("x:", &x_tied);
    print_vector("y:", &y_tied);

    match spearman_correlation(&x_tied, &y_tied) {
        Some((rho, rx, ry)) => {
            print_rank_table(&x_tied, &y_tied, &rx, &ry);
            println!("Spearman rank correlation");
            println!("=========================");
            println!("rho_S = {:>.10}", rho);
            println!();

            println!("Tie diagnostics");
            println!("===============");
            println!("x has ties: {}", has_ties(&x_tied));
            println!("y has ties: {}", has_ties(&y_tied));
            println!(
                "shortcut formula available: {}",
                spearman_shortcut_no_ties(&x_tied, &y_tied).is_some()
            );
            println!();
        }
        None => {
            println!("Could not compute Spearman correlation for Example A.");
            println!();
        }
    }

    transformed_example(&x_tied, &y_tied);

    // Example 2: data without ties, so both formulas can be compared.
    let x_no_ties = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y_no_ties = vec![1.2, 2.7, 2.9, 4.8, 4.9, 6.3];

    println!("Section 14.7.1 Example B: No-Ties Validation");
    println!("============================================");
    print_vector("x:", &x_no_ties);
    print_vector("y:", &y_no_ties);

    let general = spearman_correlation(&x_no_ties, &y_no_ties).map(|t| t.0);
    let shortcut = spearman_shortcut_no_ties(&x_no_ties, &y_no_ties);

    match (general, shortcut) {
        (Some(r_general), Some(r_shortcut)) => {
            println!("General rank-Pearson formula  = {:>.10}", r_general);
            println!("No-ties shortcut formula      = {:>.10}", r_shortcut);
            println!(
                "absolute difference           = {:>.10e}",
                (r_general - r_shortcut).abs()
            );
        }
        _ => {
            println!("Could not compare the two Spearman formulas.");
        }
    }
}
```

Program 14.7.1 demonstrates a practical approach to computing rank-based correlation by separating ordering from magnitude and applying correlation to transformed variables. This approach reflects the central idea of Section 14.7.1: that dependence can be characterized reliably through ordering alone, even when raw numerical relationships are distorted by nonlinearity or outliers.

The examples illustrate two important aspects of rank-based methods. The tied-data case shows how midranks preserve consistency and stability in the presence of repeated values, while the no-ties case validates the equivalence between the general and shortcut formulations. The monotone-transformation check further confirms that the method captures intrinsic ordering relationships rather than scale-dependent effects.

The modular structure of the program allows it to be extended naturally to other rank-based measures, such as Kendall’s $\tau$ or more general dependence statistics discussed later in the section. This provides a foundation for broader nonparametric analysis, where robustness and invariance are essential for reliable interpretation in modern numerical and data-driven applications.

## 14.7.2. Kendall Correlation and Algorithmic Aspects

Kendall’s rank correlation coefficient $\tau_b$ is based on pairwise comparisons between observations. For pairs $(x_i, y_i)$ and $(x_j, y_j)$, with $i < j$, the relationship between the two pairs is classified according to the relative ordering of their components:

- concordant if $(x_i - x_j)(y_i - y_j) > 0$, meaning that both variables agree in their ordering for the pair,
- discordant if $(x_i - x_j)(y_i - y_j) < 0$, meaning that the variables disagree in their ordering,
- ties occur when $x_i = x_j$ and/or $y_i = y_j$, indicating that one or both variables do not distinguish between the two observations.

These pairwise comparisons provide a direct way to assess whether the relative ordering induced by one variable is consistent with that induced by the other.

Let $C$ and $D$ denote the number of concordant and discordant pairs, and $T_x$, $T_y$ the number of ties. Then:

$$\tau_b = \frac{C - D}{\sqrt{(C + D + T_x)(C + D + T_y)}} \tag{14.7.7}$$

The numerator $C - D$ reflects the net agreement in ordering between the two variables, while the denominator normalizes this quantity to account for the total number of comparable pairs, including the effect of ties. This normalization ensures that the coefficient remains bounded and comparable across datasets with different tie structures.

Kendall’s coefficient depends only on pairwise order relations and is often considered “more nonparametric” than Spearman’s, as it does not rely on rank magnitudes. By focusing solely on whether pairs are concordant or discordant, it captures the consistency of ordering without incorporating the numerical spacing between observations.

A naive computation requires $O(N^2)$ comparisons, since all pairs of observations must be examined. However, modern implementations reduce this to:

$$O(N \log N) \tag{14.7.8}$$

by sorting and counting inversions using data structures such as Fenwick trees or merge-sort-based algorithms. These approaches exploit the structure of the problem to avoid explicit enumeration of all pairs, thereby improving computational efficiency for large datasets.

*Inference caveat.* Modern work shows that “exact” rank-based inference may fail when exchangeability assumptions are violated. In such cases, the theoretical distributions used for inference may not accurately describe the behavior of the statistic. Reliable testing often requires studentized or asymptotic corrections, especially in structured or ordinal datasets (Hutson and Yu, 2023), where dependence or constraints in the data can affect the validity of standard procedures.

### Rust Implementation

Following the discussion in Section 14.7.2 on Kendall’s rank correlation and its algorithmic formulation, Program 14.7.2 provides a practical implementation of pairwise order-based dependence analysis using both the direct $O(N^2)$ formulation and an efficient $O(N \log N)$ inversion-count approach. In numerical computing, the evaluation of Kendall’s coefficient requires careful classification of observation pairs into concordant, discordant, and tied categories, as defined prior to equation (14.7.7). This program implements that classification explicitly and then evaluates Kendall’s $\tau_b$ using the normalized form in equation (14.7.7). In addition, it demonstrates how sorting and inversion counting can reduce computational complexity in the absence of ties, reflecting the algorithmic improvements described in equation (14.7.8).

At the core of the implementation is the `kendall_pair_counts_naive` function, which directly realizes the pairwise comparison framework described in the section. For each pair $(i,j)$ with $i<j$, the function determines whether the pair is concordant, discordant, or tied by evaluating the sign of $(x_i - x_j)(y_i - y_j)$ and checking for equality conditions. This corresponds exactly to the classification rules preceding equation (14.7.7). The counts of concordant pairs $C$, discordant pairs $D$, and tie counts $T_x$ and $T_y$ are accumulated in a structured manner, ensuring a transparent mapping between the mathematical definition and its computational realization.

The function `kendall_tau_b_from_counts` then evaluates Kendall’s coefficient using equation (14.7.7). The numerator $C - D$ measures net agreement in ordering, while the denominator incorporates tie adjustments through $T_x$ and $T_y$, ensuring that the coefficient remains properly normalized across datasets with varying tie structures. The implementation guards against degenerate cases by returning an optional value when the denominator is numerically zero, thereby preserving stability.

To reflect the algorithmic considerations discussed in equation (14.7.8), the program includes the function `kendall_tau_no_ties_fast`, which computes Kendall’s coefficient in $O(N \log N)$ time when no ties are present. This function first sorts the data by $x$, then interprets the induced ordering of $y$ as a permutation. Discordant pairs correspond to inversions in this permutation, which are efficiently counted using the `count_inversions_merge_sort` function. This approach avoids explicit enumeration of all pairs and demonstrates how structural properties of the problem can be exploited for improved performance.

The auxiliary functions `has_ties`, `print_vector`, and `print_counts` support both correctness checks and interpretability. The tie-detection mechanism ensures that the fast algorithm is used only when its assumptions are satisfied, while the formatted output provides a clear view of pair classifications and computed statistics. The `monotone_invariance_check` function further illustrates a key theoretical property: since Kendall’s coefficient depends only on pairwise ordering, it remains invariant under monotone transformations of the data.

The `main` function demonstrates the implementation through two representative examples. The first dataset includes ties, illustrating the full $\tau_b$ computation and the role of tie adjustments in the denominator. The second dataset contains no ties, allowing direct comparison between the naive $O(N^2)$ method and the fast inversion-based $O(N \log N)$ method. The agreement between these two computations confirms both correctness and algorithmic consistency.

```rust
// Program 14.7.2
// Kendall Correlation and Algorithmic Aspects
//
// Problem statement:
// Given paired observations (x_i, y_i), classify all pairs (i, j) with i < j
// as concordant, discordant, or tied, and compute Kendall's tau_b
// according to Equation (14.7.7). Also demonstrate the algorithmic idea
// behind faster O(N log N) evaluation by counting inversions after sorting
// when ties are absent.

use std::cmp::Ordering;

const EPS: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy)]
struct KendallCounts {
    concordant: u64,
    discordant: u64,
    tie_x_only: u64,
    tie_y_only: u64,
    tie_both: u64,
}

impl KendallCounts {
    fn total_pairs(&self) -> u64 {
        self.concordant + self.discordant + self.tie_x_only + self.tie_y_only + self.tie_both
    }

    // For tau_b, T_x counts pairs tied in x but not y, and T_y counts pairs tied in y but not x.
    fn tx(&self) -> u64 {
        self.tie_x_only
    }

    fn ty(&self) -> u64 {
        self.tie_y_only
    }
}

fn nearly_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= EPS
}

/// Perform the naive O(N^2) pair classification required by the definition
/// of Kendall's tau_b.
fn kendall_pair_counts_naive(x: &[f64], y: &[f64]) -> KendallCounts {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");
    let n = x.len();
    assert!(n >= 2, "at least two observations are required");

    let mut counts = KendallCounts {
        concordant: 0,
        discordant: 0,
        tie_x_only: 0,
        tie_y_only: 0,
        tie_both: 0,
    };

    for i in 0..n {
        for j in (i + 1)..n {
            let dx_zero = nearly_equal(x[i], x[j]);
            let dy_zero = nearly_equal(y[i], y[j]);

            match (dx_zero, dy_zero) {
                (true, true) => counts.tie_both += 1,
                (true, false) => counts.tie_x_only += 1,
                (false, true) => counts.tie_y_only += 1,
                (false, false) => {
                    let dx = x[i] - x[j];
                    let dy = y[i] - y[j];
                    let prod = dx * dy;
                    if prod > 0.0 {
                        counts.concordant += 1;
                    } else if prod < 0.0 {
                        counts.discordant += 1;
                    }
                }
            }
        }
    }

    counts
}

/// Compute Kendall's tau_b from pair counts using Equation (14.7.7).
fn kendall_tau_b_from_counts(counts: &KendallCounts) -> Option<f64> {
    let c = counts.concordant as f64;
    let d = counts.discordant as f64;
    let tx = counts.tx() as f64;
    let ty = counts.ty() as f64;

    let numerator = c - d;
    let denominator = ((c + d + tx) * (c + d + ty)).sqrt();

    if denominator <= EPS {
        None
    } else {
        Some(numerator / denominator)
    }
}

fn kendall_tau_b_naive(x: &[f64], y: &[f64]) -> Option<(f64, KendallCounts)> {
    let counts = kendall_pair_counts_naive(x, y);
    let tau = kendall_tau_b_from_counts(&counts)?;
    Some((tau, counts))
}

fn has_ties(values: &[f64]) -> bool {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    for i in 1..sorted.len() {
        if nearly_equal(sorted[i - 1], sorted[i]) {
            return true;
        }
    }
    false
}

fn print_vector(label: &str, values: &[f64]) {
    println!("{label}");
    for (i, v) in values.iter().enumerate() {
        println!("  [{:>2}] {:>.10}", i, v);
    }
    println!();
}

fn print_pair_table(x: &[f64], y: &[f64]) {
    println!("Paired Observations");
    println!("===================");
    println!("{:>5} {:>16} {:>16}", "i", "x_i", "y_i");
    for i in 0..x.len() {
        println!("{:>5} {:>16.10} {:>16.10}", i, x[i], y[i]);
    }
    println!();
}

fn print_counts(counts: &KendallCounts) {
    println!("Pair Classification Counts");
    println!("==========================");
    println!("Concordant pairs C         = {}", counts.concordant);
    println!("Discordant pairs D         = {}", counts.discordant);
    println!("Ties in x only   T_x       = {}", counts.tie_x_only);
    println!("Ties in y only   T_y       = {}", counts.tie_y_only);
    println!("Ties in both variables     = {}", counts.tie_both);
    println!("Total pairs               = {}", counts.total_pairs());
    println!();
}

/// Count inversions in an integer array using merge sort.
/// The sorted array is returned in-place through mutation.
fn count_inversions_merge_sort(values: &mut [usize]) -> u64 {
    let n = values.len();
    if n <= 1 {
        return 0;
    }

    let mid = n / 2;
    let mut left = values[..mid].to_vec();
    let mut right = values[mid..].to_vec();

    let mut inv_count = count_inversions_merge_sort(&mut left) + count_inversions_merge_sort(&mut right);

    let mut i = 0usize;
    let mut j = 0usize;
    let mut k = 0usize;

    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            values[k] = left[i];
            i += 1;
        } else {
            values[k] = right[j];
            j += 1;
            inv_count += (left.len() - i) as u64;
        }
        k += 1;
    }

    while i < left.len() {
        values[k] = left[i];
        i += 1;
        k += 1;
    }

    while j < right.len() {
        values[k] = right[j];
        j += 1;
        k += 1;
    }

    inv_count
}

/// Compute Kendall's tau for the special case with no ties by:
/// 1. sorting pairs by x,
/// 2. extracting the induced order of y,
/// 3. counting inversions in y.
/// Then C + D = N(N-1)/2 and D is the inversion count.
fn kendall_tau_no_ties_fast(x: &[f64], y: &[f64]) -> Option<f64> {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");
    let n = x.len();
    if n < 2 || has_ties(x) || has_ties(y) {
        return None;
    }

    // Sort pairs by x.
    let mut pairs: Vec<(f64, f64)> = x.iter().copied().zip(y.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    // Since there are no ties in y, ranking y is just a permutation of 1..N.
    let mut y_with_idx: Vec<(usize, f64)> = pairs.iter().map(|p| p.1).enumerate().collect();
    y_with_idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut y_rank_map = vec![0usize; n];
    for (rank_zero_based, (idx_in_sorted_x, _)) in y_with_idx.iter().enumerate() {
        y_rank_map[*idx_in_sorted_x] = rank_zero_based + 1;
    }

    let mut rank_sequence = y_rank_map;
    let discordant = count_inversions_merge_sort(&mut rank_sequence);
    let total_pairs = (n as u64) * ((n as u64) - 1) / 2;
    let concordant = total_pairs - discordant;

    let numerator = concordant as f64 - discordant as f64;
    let denominator = total_pairs as f64;

    Some(numerator / denominator)
}

fn monotone_invariance_check(x: &[f64], y: &[f64]) {
    let x_monotone: Vec<f64> = x.iter().map(|&v| 2.0 * v + 1.0).collect();
    let y_monotone: Vec<f64> = y.iter().map(|&v| 3.0 * v - 4.0).collect();

    let original = kendall_tau_b_naive(x, y).map(|t| t.0);
    let transformed = kendall_tau_b_naive(&x_monotone, &y_monotone).map(|t| t.0);

    println!("Monotone-Transformation Check");
    println!("=============================");
    match (original, transformed) {
        (Some(t1), Some(t2)) => {
            println!("tau_b(original data)        = {:>.10}", t1);
            println!("tau_b(monotone transforms)  = {:>.10}", t2);
            println!("absolute difference         = {:>.10e}", (t1 - t2).abs());
        }
        _ => {
            println!("Unable to evaluate monotone-invariance check.");
        }
    }
    println!();
}

fn main() {
    // Example A: data with ties in both variables, suitable for tau_b.
    let x_tied = vec![12.0, 15.0, 15.0, 18.0, 21.0, 21.0, 25.0];
    let y_tied = vec![30.0, 35.0, 33.0, 35.0, 40.0, 40.0, 50.0];

    println!("Section 14.7.2 Example A: Kendall tau_b with Ties");
    println!("=================================================");
    print_vector("x:", &x_tied);
    print_vector("y:", &y_tied);
    print_pair_table(&x_tied, &y_tied);

    match kendall_tau_b_naive(&x_tied, &y_tied) {
        Some((tau, counts)) => {
            print_counts(&counts);
            println!("Kendall tau_b");
            println!("=============");
            println!("tau_b = {:>.10}", tau);
            println!();

            println!("Tie diagnostics");
            println!("===============");
            println!("x has ties: {}", has_ties(&x_tied));
            println!("y has ties: {}", has_ties(&y_tied));
            println!(
                "fast no-ties O(N log N) formula available: {}",
                kendall_tau_no_ties_fast(&x_tied, &y_tied).is_some()
            );
            println!();
        }
        None => {
            println!("Could not compute Kendall tau_b for Example A.");
            println!();
        }
    }

    monotone_invariance_check(&x_tied, &y_tied);

    // Example B: no ties, so we can compare the naive O(N^2) method to the
    // inversion-count O(N log N) method.
    let x_no_ties = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y_no_ties = vec![1.5, 3.1, 2.2, 5.4, 4.8, 6.7];

    println!("Section 14.7.2 Example B: No-Ties Algorithmic Validation");
    println!("=========================================================");
    print_vector("x:", &x_no_ties);
    print_vector("y:", &y_no_ties);

    let tau_naive = kendall_tau_b_naive(&x_no_ties, &y_no_ties).map(|t| t.0);
    let tau_fast = kendall_tau_no_ties_fast(&x_no_ties, &y_no_ties);

    match (tau_naive, tau_fast) {
        (Some(t1), Some(t2)) => {
            println!("Naive pairwise tau          = {:>.10}", t1);
            println!("Fast inversion-count tau    = {:>.10}", t2);
            println!("absolute difference         = {:>.10e}", (t1 - t2).abs());
        }
        _ => {
            println!("Could not compare naive and fast Kendall computations.");
        }
    }
}
```

Program 14.7.2 demonstrates a practical approach to evaluating Kendall’s rank correlation by combining a direct pairwise formulation with an efficient algorithmic alternative. This reflects the central computational theme of Section 14.7.2: while the definition of Kendall’s coefficient is inherently pairwise, its computation can be significantly accelerated by exploiting ordering structure.

The examples illustrate two key aspects of Kendall’s method. The tied-data case shows how normalization through $T_x$ and $T_y$ ensures meaningful comparison even when repeated values are present, while the no-ties case highlights the efficiency gains achievable through inversion counting. The invariance under monotone transformations further emphasizes that the measure captures pure ordering consistency, independent of scale or nonlinear transformations.

The modular structure of the program allows straightforward extension to more advanced implementations, including Fenwick-tree-based methods, parallel inversion counting, or streaming approximations for large datasets. This provides a foundation for scalable, nonparametric dependence analysis in modern numerical and data-driven applications.

## 14.7.3. Beyond Classical Rank Correlation

Recent developments extend rank-based dependence measures beyond monotonic relationships. These approaches aim to capture more general forms of dependence that may not be adequately described by classical rank correlations, while still retaining the robustness and interpretability associated with rank-based methods.

*Reproducibility-aware correlation***.** Modern approaches emphasize not only statistical significance but also the probability that a result replicates under repeated sampling. In this view, a correlation estimate is evaluated not just by its magnitude or associated $p$-value, but by how consistently similar results would be obtained if the data-generating process were repeated. This perspective highlights the importance of stability alongside point estimates (Alshahrani, 2026), encouraging the assessment of whether observed associations are reliable and reproducible rather than artifacts of a particular sample.

*Recent rank-based dependence measures for general dependence.* A recent class of rank-based statistics is designed to detect general functional dependence. Let the data be ordered by $x_i$, and let $r_i$ denote the rank of $y_i$. One formulation is:

$$\xi_n = 1 - \frac{n \sum_{i=1}^{n-1} \left| r_{i+1} - r_i \right|}{2 \sum_{i=1}^{n} l_i (n - l_i)} \tag{14.7.9}$$

where $l_i$ counts how many $y$-values are less than or equal to $y_i$. In this construction, the ordering induced by $x_i$ is used to examine how the ranks of $y_i$ vary across adjacent observations, thereby capturing how smoothly or irregularly the values of $y$ change as $x$ varies.

This coefficient satisfies:

$$\xi(X;Y) \in [0,1], \qquad \xi_n \to \xi(X;Y) \text{ as } n \to \infty \tag{14.7.10}$$

with $\xi(X;Y) = 0$ under independence and $\xi(X;Y) = 1$ when $Y$ is a deterministic function of $X$ (Dalitz et al., 2024). The finite-sample statistic $\xi_n$ approximates this behavior, taking values close to zero when the ordering of $x$ provides little information about the ordering of $y$, and values close to one when the behavior of $y$ is strongly determined by $x$. The finite-sample statistic $\xi_n$ may take negative values and does not, in general, attain the full range $[0,1]$; however, it is a consistent estimator of the population dependence measure $\xi(X;Y)$.

Recent improvements address small-sample bias and enhance testing power by incorporating nearest-neighbor ideas (Lin and Han, 2023). These refinements improve the reliability of the measure in finite samples and increase its sensitivity to subtle forms of dependence. Together, these methods broaden the scope of correlation analysis to detect general dependence structures beyond monotonicity, while maintaining the advantages of rank-based approaches.

### Rust Implementation

Following the discussion in Section 14.7.3 on extending rank-based methods beyond monotonic dependence, Program 14.7.3 provides a practical implementation of a general rank-based dependence coefficient together with a reproducibility assessment via bootstrap resampling. In numerical computing, it is often insufficient to rely solely on classical correlation measures when relationships are nonlinear or irregular. The coefficient defined in equation (14.7.9) addresses this limitation by examining how the ranks of one variable evolve when the data are ordered by another. This program implements that construction explicitly and complements it with a stability analysis that reflects the modern emphasis on reproducibility discussed in the section.

At the core of the implementation is the transformation of the paired data into an ordered representation based on $x_i$. The function `sort_pairs_by_x` performs this step, ensuring that the subsequent analysis follows the ordering assumed in equation (14.7.9). Once the data are ordered, the function `upper_ranks` computes the rank values $r_i$ of the corresponding $y_i$, using an upper-rank convention so that each $l_i = \#\{y_j \le y_i\}$ coincides directly with the assigned rank. This design avoids the need for separate counting logic and ensures consistency with the definition of $l_i$ in the formula.

The function `xi_coefficient` then evaluates the dependence measure by assembling the two key components appearing in equation (14.7.9): the sum $\sum_{i=1}^n l_i(n-l_i)$, which captures the global distribution of ranks, and the sum $\sum_{i=1}^{n-1} |r_{i+1} - r_i|$, which measures how rapidly ranks change across adjacent observations in the $x$-ordered sequence. These quantities are combined using the corrected finite-sample form implemented in the code. The function also records whether ties are present in $x$ or $y$, providing useful diagnostic information for interpretation.

To reflect the reproducibility-aware perspective introduced in the section, the program includes the function `bootstrap_xi`, which repeatedly resamples the paired observations and recomputes the coefficient. By summarizing the distribution of these bootstrap values, the program provides an empirical assessment of how stable the dependence estimate is under repeated sampling. This complements the point estimate of $\xi_n$ with information about variability and reliability, aligning with the modern emphasis on reproducibility rather than relying solely on a single computed value.

The `monotone_invariance_check` function illustrates a key structural property of the method: since the coefficient depends only on the ordering of observations, it remains invariant under monotone transformations of the data. This is demonstrated computationally by applying simple transformations to both variables and verifying that the resulting value of $\xi_n$ remains unchanged up to numerical precision.

The `main` function demonstrates the implementation using two representative datasets. The first example exhibits a strong nonlinear dependence with distinct values, illustrating how the coefficient captures structured relationships that are not purely monotonic. The second example presents a more irregular pattern, resulting in a weaker dependence measure and highlighting the sensitivity of the statistic to the ordering behavior of the data. The bootstrap summary further reveals how the estimated dependence varies under resampling, providing insight into its stability.

```rust
// Program 14.7.3
// Beyond Classical Rank Correlation: General Rank-Based Dependence and Reproducibility
//
// Problem statement:
// Given paired observations (x_i, y_i), compute a general rank-based dependence
// coefficient xi_n after ordering the data by x_i, using the finite-sample formula
// discussed in Section 14.7.3. The program also estimates reproducibility by
// bootstrap resampling to illustrate the stability of the dependence estimate
// under repeated sampling.

use std::cmp::Ordering;

const EPS: f64 = 1.0e-12;

#[derive(Debug, Clone)]
struct XiResult {
    xi_n: f64,
    x_sorted: Vec<f64>,
    y_sorted: Vec<f64>,
    r: Vec<usize>, // rank of y_i in x-sorted order
    l: Vec<usize>, // count of y-values <= y_i
    numerator_sum: u64,
    denominator_sum: u64,
    has_x_ties: bool,
    has_y_ties: bool,
}

#[derive(Debug, Clone)]
struct BootstrapSummary {
    mean: f64,
    min: f64,
    max: f64,
    std_dev: f64,
    q05: f64,
    q50: f64,
    q95: f64,
    successful_samples: usize,
    attempted_samples: usize,
}

#[derive(Debug, Clone)]
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn gen_usize(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "upper bound must be positive");
        (self.next_u64() % upper as u64) as usize
    }
}

fn nearly_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= EPS
}

fn mean(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "mean requires nonempty input");
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev_population(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "std_dev requires nonempty input");
    let m = mean(values);
    let var = values
        .iter()
        .map(|v| {
            let d = *v - m;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    var.sqrt()
}

fn quantile(sorted_values: &[f64], p: f64) -> f64 {
    assert!(!sorted_values.is_empty(), "quantile requires nonempty input");
    assert!((0.0..=1.0).contains(&p), "p must lie in [0,1]");

    if sorted_values.len() == 1 {
        return sorted_values[0];
    }

    let pos = p * (sorted_values.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;

    if lo == hi {
        sorted_values[lo]
    } else {
        let w = pos - lo as f64;
        (1.0 - w) * sorted_values[lo] + w * sorted_values[hi]
    }
}

fn has_ties(values: &[f64]) -> bool {
    if values.len() < 2 {
        return false;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    for i in 1..sorted.len() {
        if nearly_equal(sorted[i - 1], sorted[i]) {
            return true;
        }
    }
    false
}

/// Return the stable ordering permutation that sorts values in nondecreasing order.
fn argsort(values: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&i, &j| match values[i].partial_cmp(&values[j]) {
        Some(Ordering::Less) => Ordering::Less,
        Some(Ordering::Greater) => Ordering::Greater,
        _ => i.cmp(&j),
    });
    idx
}

/// Compute 1-based upper ranks:
/// if ties occupy positions a..b in sorted order, each tied value receives rank b.
/// This makes l_i = #{ y_j <= y_i } coincide with the assigned upper rank.
fn upper_ranks(values: &[f64]) -> Vec<usize> {
    let n = values.len();
    assert!(n > 0, "upper_ranks requires nonempty input");

    let order = argsort(values);
    let mut ranks = vec![0usize; n];

    let mut start = 0usize;
    while start < n {
        let mut end = start;
        while end + 1 < n && nearly_equal(values[order[end]], values[order[end + 1]]) {
            end += 1;
        }

        let upper_rank = end + 1; // 1-based
        for k in start..=end {
            ranks[order[k]] = upper_rank;
        }

        start = end + 1;
    }

    ranks
}

/// Sort paired observations by x, then by y, then by original index.
/// This gives a deterministic order for evaluating xi_n.
fn sort_pairs_by_x(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");

    let mut triples: Vec<(usize, f64, f64)> = x
        .iter()
        .copied()
        .zip(y.iter().copied())
        .enumerate()
        .map(|(i, (xi, yi))| (i, xi, yi))
        .collect();

    triples.sort_by(|a, b| {
        let x_cmp = a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal);
        if x_cmp != Ordering::Equal {
            return x_cmp;
        }

        let y_cmp = a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal);
        if y_cmp != Ordering::Equal {
            return y_cmp;
        }

        a.0.cmp(&b.0)
    });

    let x_sorted = triples.iter().map(|t| t.1).collect::<Vec<_>>();
    let y_sorted = triples.iter().map(|t| t.2).collect::<Vec<_>>();
    let original_indices = triples.iter().map(|t| t.0).collect::<Vec<_>>();

    (x_sorted, y_sorted, original_indices)
}

/// Compute the general rank-based dependence coefficient xi_n.
///
/// The corrected finite-sample computation used here is
///
/// xi_n = 1 - [ n * sum_{i=1}^{n-1} |r_{i+1} - r_i| ] / [ 2 * sum_{i=1}^n l_i (n - l_i) ].
///
/// Here:
/// - the data are first ordered by x,
/// - r_i are upper ranks of y in that x-sorted order,
/// - l_i = #{ y_j <= y_i }, which coincides with the upper rank convention.
///
/// Note:
/// finite-sample values may be negative even though the underlying population
/// dependence measure lies in [0, 1].
fn xi_coefficient(x: &[f64], y: &[f64]) -> Option<XiResult> {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");
    let n = x.len();
    if n < 2 {
        return None;
    }

    let has_x_ties = has_ties(x);
    let has_y_ties = has_ties(y);

    let (x_sorted, y_sorted, _) = sort_pairs_by_x(x, y);
    let r = upper_ranks(&y_sorted);
    let l = r.clone();

    let numerator_sum = l
        .iter()
        .map(|&li| {
            let li_u = li as u64;
            li_u * (n as u64 - li_u)
        })
        .sum::<u64>();

    if numerator_sum == 0 {
        return None;
    }

    let denominator_sum = (0..(n - 1))
        .map(|i| {
            let a = r[i] as i64;
            let b = r[i + 1] as i64;
            (b - a).unsigned_abs()
        })
        .sum::<u64>();

    let xi_n = 1.0 - (n as f64 * denominator_sum as f64) / (2.0 * numerator_sum as f64);

    Some(XiResult {
        xi_n,
        x_sorted,
        y_sorted,
        r,
        l,
        numerator_sum,
        denominator_sum,
        has_x_ties,
        has_y_ties,
    })
}

fn bootstrap_xi(x: &[f64], y: &[f64], n_boot: usize, seed: u64) -> Option<BootstrapSummary> {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");
    let n = x.len();
    if n < 2 || n_boot == 0 {
        return None;
    }

    let mut rng = LcgRng::new(seed);
    let mut values = Vec::with_capacity(n_boot);

    for _ in 0..n_boot {
        let mut xb = Vec::with_capacity(n);
        let mut yb = Vec::with_capacity(n);

        for _ in 0..n {
            let idx = rng.gen_usize(n);
            xb.push(x[idx]);
            yb.push(y[idx]);
        }

        if let Some(result) = xi_coefficient(&xb, &yb) {
            values.push(result.xi_n);
        }
    }

    if values.is_empty() {
        return None;
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    Some(BootstrapSummary {
        mean: mean(&values),
        min: values[0],
        max: values[values.len() - 1],
        std_dev: std_dev_population(&values),
        q05: quantile(&values, 0.05),
        q50: quantile(&values, 0.50),
        q95: quantile(&values, 0.95),
        successful_samples: values.len(),
        attempted_samples: n_boot,
    })
}

fn monotone_invariance_check(x: &[f64], y: &[f64]) {
    let x_monotone: Vec<f64> = x.iter().map(|&v| 2.0 * v + 3.0).collect();
    let y_monotone: Vec<f64> = y.iter().map(|&v| v.exp()).collect();

    let original = xi_coefficient(x, y).map(|res| res.xi_n);
    let transformed = xi_coefficient(&x_monotone, &y_monotone).map(|res| res.xi_n);

    println!("Monotone-Transformation Check");
    println!("=============================");
    match (original, transformed) {
        (Some(a), Some(b)) => {
            println!("xi_n(original data)         = {:>.10}", a);
            println!("xi_n(monotone transforms)   = {:>.10}", b);
            println!("absolute difference         = {:>.10e}", (a - b).abs());
        }
        _ => {
            println!("Unable to evaluate monotone-invariance check.");
        }
    }
    println!();
}

fn print_vector(label: &str, values: &[f64]) {
    println!("{label}");
    for (i, v) in values.iter().enumerate() {
        println!("  [{:>2}] {:>.10}", i, v);
    }
    println!();
}

fn print_sorted_rank_table(result: &XiResult) {
    println!("Data Ordered by x_i");
    println!("===================");
    println!(
        "{:>5} {:>16} {:>16} {:>10} {:>10}",
        "i", "x_(i)", "y_(i)", "r_i", "l_i"
    );

    for i in 0..result.x_sorted.len() {
        println!(
            "{:>5} {:>16.10} {:>16.10} {:>10} {:>10}",
            i,
            result.x_sorted[i],
            result.y_sorted[i],
            result.r[i],
            result.l[i]
        );
    }
    println!();
}

fn print_xi_details(result: &XiResult) {
    println!("Coefficient Components");
    println!("======================");
    println!("x contains ties                 = {}", result.has_x_ties);
    println!("y contains ties                 = {}", result.has_y_ties);
    println!("sum_i l_i (n - l_i)             = {}", result.numerator_sum);
    println!("sum_i |r_(i+1) - r_i|           = {}", result.denominator_sum);
    println!("xi_n                            = {:>.10}", result.xi_n);
    println!();
}

fn print_bootstrap_summary(summary: &BootstrapSummary) {
    println!("Bootstrap Reproducibility Summary");
    println!("=================================");
    println!(
        "successful / attempted samples   = {} / {}",
        summary.successful_samples, summary.attempted_samples
    );
    println!("mean xi_n                        = {:>.10}", summary.mean);
    println!("std. dev.                        = {:>.10}", summary.std_dev);
    println!("minimum                          = {:>.10}", summary.min);
    println!("5% quantile                      = {:>.10}", summary.q05);
    println!("median                           = {:>.10}", summary.q50);
    println!("95% quantile                     = {:>.10}", summary.q95);
    println!("maximum                          = {:>.10}", summary.max);
    println!();
}

fn main() {
    // Example A:
    // Strong nonlinear dependence with distinct y-values.
    // This keeps the example deterministic and non-monotone while avoiding tie ambiguity.
    let x_general = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let y_general = vec![8.7, 3.8, 0.9, 0.0, 1.1, 4.2, 9.3];

    println!("Section 14.7.3 Example A: General Rank-Based Dependence");
    println!("=======================================================");
    print_vector("x:", &x_general);
    print_vector("y:", &y_general);

    match xi_coefficient(&x_general, &y_general) {
        Some(result) => {
            print_sorted_rank_table(&result);
            print_xi_details(&result);
        }
        None => {
            println!("Could not compute xi_n for Example A.");
            println!();
        }
    }

    monotone_invariance_check(&x_general, &y_general);

    match bootstrap_xi(&x_general, &y_general, 500, 123456789) {
        Some(summary) => print_bootstrap_summary(&summary),
        None => {
            println!("Bootstrap reproducibility summary unavailable.");
            println!();
        }
    }

    // Example B:
    // A weaker and more irregular dependence pattern.
    let x_weaker = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y_weaker = vec![4.2, 1.3, 7.1, 2.8, 6.4, 3.5, 5.9, 2.1];

    println!("Section 14.7.3 Example B: Weaker General Dependence");
    println!("===================================================");
    print_vector("x:", &x_weaker);
    print_vector("y:", &y_weaker);

    match xi_coefficient(&x_weaker, &y_weaker) {
        Some(result) => {
            print_sorted_rank_table(&result);
            print_xi_details(&result);
        }
        None => {
            println!("Could not compute xi_n for Example B.");
            println!();
        }
    }

    // Optional note for interpretation.
    println!("Interpretation Note");
    println!("===================");
    println!("Finite-sample xi_n values can be negative.");
    println!("The coefficient remains order-based and invariant under monotone transformations,");
    println!("but small-sample realizations need not stay in [0, 1].");
}
```

Program 14.7.3 demonstrates a practical framework for detecting general dependence structures using rank-based methods that extend beyond classical correlation. By focusing on how ranks evolve along an ordered sequence, the method captures a broader class of relationships while retaining robustness to scaling and outliers.

The examples illustrate two important aspects of this approach. The nonlinear deterministic case shows that strong dependence can be detected even when the relationship is not monotonic, while the weaker example highlights how irregular ordering leads to smaller or even negative finite-sample values. The bootstrap analysis reinforces the importance of stability, showing how repeated sampling can be used to assess the reliability of the computed coefficient.

The modular structure of the program makes it straightforward to extend this framework to more advanced variants, including nearest-neighbor refinements or bias-corrected estimators discussed in recent developments. This provides a foundation for modern nonparametric dependence analysis in numerical and data-driven applications, where capturing general structure and ensuring reproducibility are both essential.

## 14.7.4. Information-Theoretic Measures of Dependence

In many numerical pipelines, the primary object is an empirical distribution rather than a simple dataset. Instead of analyzing individual observations directly, one works with estimated probabilities that summarize the frequency of different outcomes. Information theory provides quantitative measures of uncertainty and dependence in this setting, allowing one to describe how much variability is present in a distribution and how strongly different variables are related.

Let a discrete distribution be:

$$p = (p_1, \dots, p_I), \qquad p_i \ge 0, \quad \sum_{i=1}^{I} p_i = 1 \tag{14.7.11}$$

This vector represents the probabilities associated with a finite set of outcomes. Each component $p_i$ corresponds to the likelihood of observing outcome $i$, and the normalization condition ensures that the probabilities sum to one.

The Shannon entropy is:

$$H(p) = - \sum_{i=1}^{I} p_i \log p_i, \quad 0 \log 0 := 0 \tag{14.7.12}$$

This quantity measures the uncertainty or dispersion of the distribution. Larger values of (H(p)) correspond to more spread-out distributions, where probability mass is distributed across many outcomes, while smaller values indicate more concentrated distributions. The convention $0 \log 0 := 0$ ensures that terms corresponding to zero probabilities do not contribute to the sum and that the expression remains well-defined.

Given two distributions $p$ and $q$, the Kullback–Leibler divergence is:

$$D_{\mathrm{KL}}(p \mid q) = \sum_{i} p_i \log \frac{p_i}{q_i} \tag{14.7.13}$$

which quantifies the discrepancy between the two distributions by comparing their probabilities outcome by outcome. It measures how well the distribution $q$ approximates $p$, with larger values indicating greater divergence between the two.

The cross-entropy satisfies,

$$H(p, q) = H(p) + D_{\mathrm{KL}}(p \mid q) \tag{14.7.14}$$

This decomposition shows that cross-entropy combines the intrinsic uncertainty of $p$ with the additional discrepancy introduced when $q$ is used in place of $p$. As a result, it reflects both the variability of the true distribution and the mismatch between the two distributions.

For joint distributions $p_{ij}$, the mutual information is:

$$I(X; Y) = \sum_{i,j} p_{ij} \log \frac{p_{ij}}{p_{i\cdot} p_{\cdot j}} = D_{\mathrm{KL}}(p_{XY} \mid p_X p_Y) \tag{14.7.15}$$

This quantity measures the dependence between two variables by comparing the joint distribution (p\_{ij}) to the product of the marginal distributions (p\_{i\\cdot} p\_{\\cdot j}). When the joint distribution factorizes into the product of marginals, the variables behave independently, and the divergence vanishes.

Mutual information satisfies,

$$I(X; Y) \ge 0, \quad I(X; Y) = 0 \text{ iff } X \text{ and } Y \text{ are independent} \tag{14.7.16}$$

Thus, it provides a nonnegative measure of dependence, with larger values indicating stronger relationships between the variables. Unlike correlation, mutual information captures general dependence, including nonlinear and non-monotonic relationships, since it is based on the full joint distribution rather than specific functional forms. This makes it particularly valuable in modern machine learning, signal processing, and experimental design, where complex dependencies are common (Letizia et al., 2024).

### Computational Considerations

Entropy and divergence calculations must handle zero probabilities carefully, using conventions such as $0 \log 0 = 0$, to ensure numerical stability and well-defined expressions. In practical computations, care must be taken when probabilities are very small, as numerical precision can affect the evaluation of logarithms and ratios.

Estimation in high-dimensional or large-alphabet settings is challenging, as accurately estimating probabilities requires sufficient data across many possible outcomes. When data are limited, naive estimates may be biased or unstable. Modern research emphasizes bias correction, smoothing strategies, and model-based estimation techniques (Pinchas et al., 2024; Altieri et al., 2023), which aim to improve the reliability of entropy and divergence estimates in such settings.

### Rust Implementation

Following the discussion in Section 14.7.4 on information-theoretic measures of dependence, Program 14.7.4 provides a practical implementation of entropy, divergence, and mutual information for discrete probability distributions. In many numerical settings, the primary object of interest is not raw data but an empirical distribution derived from observations, making it essential to compute quantities such as Shannon entropy, Kullback–Leibler divergence, and mutual information in a numerically stable manner. This program implements these measures directly from their definitions in equations (14.7.12)–(14.7.15), while carefully handling edge cases such as zero probabilities using the convention $0 \log 0 = 0$. It evaluates both explicitly defined distributions and empirically constructed joint distributions, illustrating how uncertainty and dependence can be quantified in practical computational workflows.

At the core of the implementation is the `Distribution` structure, which represents a discrete probability vector $p = (p_1, \dots, p_I)$ as defined in equation (14.7.11). The function `normalize_probabilities` constructs a valid probability distribution from raw weights by enforcing nonnegativity and normalization, while `validate_distribution` ensures that the resulting vector satisfies the required constraints. These steps reflect the fundamental requirement that all information-theoretic quantities operate on properly normalized distributions.

The function `shannon_entropy` implements the entropy defined in equation (14.7.12). It evaluates the sum $H(p) = -\sum_i p_i \log p_i$, skipping terms where $p_i = 0$ in accordance with the convention $0 \log 0 = 0$. This avoids undefined logarithmic evaluations and ensures numerical stability when distributions contain zero-probability events.

The function `kl_divergence` computes the Kullback–Leibler divergence given in equation (14.7.13). It evaluates $D_{\mathrm{KL}}(p \mid q)$ by summing contributions of the form $p_i \log(p_i/q_i)$, while explicitly detecting cases where $q_i = 0$ and $p_i > 0$, which would lead to infinite divergence. The related function `cross_entropy` implements equation (14.7.14) by computing (H(p,q)) directly and allows verification of the decomposition $H(p,q) = H(p) + D_{\mathrm{KL}}(p \mid q)$.

To handle dependence between variables, the program introduces the `JointDistribution` structure and constructs empirical joint distributions from paired categorical observations using the function `empirical_joint_distribution`. From this joint representation, the functions `marginal_x` and `marginal_y` compute the marginal distributions $p_{i\cdot}$ and $p_{\cdot j}$, while `joint_entropy` evaluates the entropy of the joint distribution. The function `mutual_information` then implements equation (14.7.15), computing $I(X;Y)$ either directly from the joint distribution or equivalently via the divergence $D_{\mathrm{KL}}(p_{XY} \mid p_X p_Y)$.

The program also includes the function `product_distribution`, which constructs the independent model $p_X p_Y$, enabling a direct computational verification of the identity $I(X;Y) = D_{\mathrm{KL}}(p_{XY} \mid p_X p_Y)$. Supporting functions such as `print_distribution`, `print_joint_distribution`, and summary routines provide structured output that highlights normalization, marginal consistency, and identity checks.

The `main` function demonstrates the implementation through two representative scenarios. First, it evaluates entropy, divergence, and cross-entropy for two explicitly defined distributions $p$ and $q$, verifying the decomposition in equation (14.7.14). Second, it constructs an empirical joint distribution from categorical data and computes the corresponding mutual information, illustrating how dependence can be quantified directly from observed frequencies. A final example with a sparse distribution confirms that zero-probability terms are handled correctly under the $0 \log 0 = 0$ convention.

```rust
// Program 14.7.4
// Information-Theoretic Measures of Dependence
//
// Problem statement:
// Given empirical discrete data or explicitly specified probability distributions,
// compute Shannon entropy, Kullback-Leibler divergence, cross-entropy, and mutual
// information. The implementation must handle zero probabilities carefully using
// the convention 0 log 0 = 0, and it must support dependence analysis through a
// joint distribution p_{ij} together with its marginal distributions.

use std::collections::{BTreeMap, BTreeSet};

const EPS: f64 = 1.0e-15;

#[derive(Debug, Clone)]
struct Distribution {
    probs: Vec<f64>,
}

#[derive(Debug, Clone)]
struct JointDistribution {
    probs: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
}

#[derive(Debug, Clone)]
struct InformationMeasures {
    entropy_p: f64,
    entropy_q: f64,
    kl_pq: f64,
    kl_qp: f64,
    cross_entropy_pq: f64,
    cross_entropy_qp: f64,
}

#[derive(Debug, Clone)]
struct MutualInformationResult {
    joint_entropy: f64,
    entropy_x: f64,
    entropy_y: f64,
    mutual_information: f64,
}

fn log_safe(x: f64) -> f64 {
    x.ln()
}

fn is_nonnegative(x: f64) -> bool {
    x >= -EPS
}

fn normalize_probabilities(weights: &[f64]) -> Result<Distribution, String> {
    if weights.is_empty() {
        return Err("distribution must contain at least one entry".to_string());
    }

    if weights.iter().any(|&w| !is_nonnegative(w)) {
        return Err("distribution contains a negative entry".to_string());
    }

    let sum = weights.iter().sum::<f64>();
    if sum <= EPS {
        return Err("distribution has zero total mass".to_string());
    }

    let probs = weights.iter().map(|&w| w / sum).collect::<Vec<_>>();
    Ok(Distribution { probs })
}

fn validate_distribution(p: &Distribution) -> Result<(), String> {
    if p.probs.is_empty() {
        return Err("distribution must contain at least one entry".to_string());
    }
    if p.probs.iter().any(|&v| !is_nonnegative(v)) {
        return Err("distribution contains a negative probability".to_string());
    }
    let sum = p.probs.iter().sum::<f64>();
    if (sum - 1.0).abs() > 1.0e-10 {
        return Err(format!(
            "distribution is not normalized: sum = {:.16}",
            sum
        ));
    }
    Ok(())
}

fn validate_joint_distribution(joint: &JointDistribution) -> Result<(), String> {
    if joint.rows == 0 || joint.cols == 0 {
        return Err("joint distribution must have positive dimensions".to_string());
    }
    if joint.probs.len() != joint.rows {
        return Err("row count does not match matrix height".to_string());
    }
    if joint.probs.iter().any(|row| row.len() != joint.cols) {
        return Err("column count does not match matrix width".to_string());
    }
    if joint
        .probs
        .iter()
        .flat_map(|row| row.iter())
        .any(|&v| !is_nonnegative(v))
    {
        return Err("joint distribution contains a negative probability".to_string());
    }

    let sum = joint
        .probs
        .iter()
        .flat_map(|row| row.iter())
        .sum::<f64>();

    if (sum - 1.0).abs() > 1.0e-10 {
        return Err(format!(
            "joint distribution is not normalized: sum = {:.16}",
            sum
        ));
    }

    Ok(())
}

fn shannon_entropy(p: &Distribution) -> Result<f64, String> {
    validate_distribution(p)?;

    let mut entropy = 0.0_f64;
    for &pi in &p.probs {
        if pi > EPS {
            entropy -= pi * log_safe(pi);
        }
    }
    Ok(entropy)
}

fn kl_divergence(p: &Distribution, q: &Distribution) -> Result<f64, String> {
    validate_distribution(p)?;
    validate_distribution(q)?;

    if p.probs.len() != q.probs.len() {
        return Err("KL divergence requires distributions of the same length".to_string());
    }

    let mut kl = 0.0_f64;
    for (&pi, &qi) in p.probs.iter().zip(q.probs.iter()) {
        if pi <= EPS {
            continue;
        }
        if qi <= EPS {
            return Err("KL divergence is infinite because q_i = 0 where p_i > 0".to_string());
        }
        kl += pi * log_safe(pi / qi);
    }

    Ok(kl)
}

fn cross_entropy(p: &Distribution, q: &Distribution) -> Result<f64, String> {
    validate_distribution(p)?;
    validate_distribution(q)?;

    if p.probs.len() != q.probs.len() {
        return Err("cross-entropy requires distributions of the same length".to_string());
    }

    let mut ce = 0.0_f64;
    for (&pi, &qi) in p.probs.iter().zip(q.probs.iter()) {
        if pi <= EPS {
            continue;
        }
        if qi <= EPS {
            return Err("cross-entropy is infinite because q_i = 0 where p_i > 0".to_string());
        }
        ce -= pi * log_safe(qi);
    }

    Ok(ce)
}

fn marginal_x(joint: &JointDistribution) -> Result<Distribution, String> {
    validate_joint_distribution(joint)?;

    let mut px = vec![0.0_f64; joint.rows];
    for (i, row) in joint.probs.iter().enumerate() {
        px[i] = row.iter().sum::<f64>();
    }
    Ok(Distribution { probs: px })
}

fn marginal_y(joint: &JointDistribution) -> Result<Distribution, String> {
    validate_joint_distribution(joint)?;

    let mut py = vec![0.0_f64; joint.cols];
    for row in &joint.probs {
        for (j, &v) in row.iter().enumerate() {
            py[j] += v;
        }
    }
    Ok(Distribution { probs: py })
}

fn joint_entropy(joint: &JointDistribution) -> Result<f64, String> {
    validate_joint_distribution(joint)?;

    let mut entropy = 0.0_f64;
    for row in &joint.probs {
        for &pij in row {
            if pij > EPS {
                entropy -= pij * log_safe(pij);
            }
        }
    }
    Ok(entropy)
}

fn mutual_information(joint: &JointDistribution) -> Result<MutualInformationResult, String> {
    validate_joint_distribution(joint)?;

    let px = marginal_x(joint)?;
    let py = marginal_y(joint)?;
    let hx = shannon_entropy(&px)?;
    let hy = shannon_entropy(&py)?;
    let hxy = joint_entropy(joint)?;

    let mut mi = 0.0_f64;
    for i in 0..joint.rows {
        for j in 0..joint.cols {
            let pij = joint.probs[i][j];
            if pij <= EPS {
                continue;
            }
            let denom = px.probs[i] * py.probs[j];
            if denom <= EPS {
                return Err("mutual information encountered zero marginal with positive joint mass".to_string());
            }
            mi += pij * log_safe(pij / denom);
        }
    }

    if mi.abs() <= 1.0e-14 {
        mi = 0.0;
    }

    Ok(MutualInformationResult {
        joint_entropy: hxy,
        entropy_x: hx,
        entropy_y: hy,
        mutual_information: mi,
    })
}

fn product_distribution(px: &Distribution, py: &Distribution) -> Result<JointDistribution, String> {
    validate_distribution(px)?;
    validate_distribution(py)?;

    let rows = px.probs.len();
    let cols = py.probs.len();
    let mut probs = vec![vec![0.0_f64; cols]; rows];

    for (i, &pxi) in px.probs.iter().enumerate() {
        for (j, &pyj) in py.probs.iter().enumerate() {
            probs[i][j] = pxi * pyj;
        }
    }

    Ok(JointDistribution { probs, rows, cols })
}

fn empirical_joint_distribution(
    x: &[&str],
    y: &[&str],
) -> Result<(JointDistribution, Vec<String>, Vec<String>), String> {
    if x.len() != y.len() {
        return Err("x and y must have the same number of observations".to_string());
    }
    if x.is_empty() {
        return Err("empirical data must be nonempty".to_string());
    }

    let x_levels: Vec<String> = x
        .iter()
        .map(|s| s.to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let y_levels: Vec<String> = y
        .iter()
        .map(|s| s.to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let x_index: BTreeMap<String, usize> = x_levels
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();

    let y_index: BTreeMap<String, usize> = y_levels
        .iter()
        .cloned()
        .enumerate()
        .map(|(j, v)| (v, j))
        .collect();

    let rows = x_levels.len();
    let cols = y_levels.len();
    let mut counts = vec![vec![0.0_f64; cols]; rows];

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let i = *x_index
            .get(xi)
            .ok_or_else(|| "failed to index x level".to_string())?;
        let j = *y_index
            .get(yi)
            .ok_or_else(|| "failed to index y level".to_string())?;
        counts[i][j] += 1.0;
    }

    let total = x.len() as f64;
    let probs = counts
        .into_iter()
        .map(|row| row.into_iter().map(|c| c / total).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    Ok((
        JointDistribution { probs, rows, cols },
        x_levels,
        y_levels,
    ))
}

fn print_distribution(name: &str, p: &Distribution) {
    println!("{name}");
    println!("{}", "=".repeat(name.len()));
    for (i, &pi) in p.probs.iter().enumerate() {
        println!("p[{i}] = {:>.10}", pi);
    }
    println!("sum  = {:>.10}", p.probs.iter().sum::<f64>());
    println!();
}

fn print_joint_distribution(
    title: &str,
    joint: &JointDistribution,
    row_labels: &[String],
    col_labels: &[String],
) {
    println!("{title}");
    println!("{}", "=".repeat(title.len()));

    print!("{:>14}", "");
    for label in col_labels {
        print!("{:>14}", label);
    }
    println!("{:>14}", "row sum");

    for (i, row) in joint.probs.iter().enumerate() {
        let row_sum = row.iter().sum::<f64>();
        print!("{:>14}", row_labels[i]);
        for &v in row {
            print!("{:>14.8}", v);
        }
        println!("{:>14.8}", row_sum);
    }

    print!("{:>14}", "col sum");
    for j in 0..joint.cols {
        let col_sum = joint.probs.iter().map(|row| row[j]).sum::<f64>();
        print!("{:>14.8}", col_sum);
    }
    println!("{:>14.8}", 1.0);
    println!();
}

fn print_information_summary(title: &str, summary: &InformationMeasures) {
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
    println!("H(p)             = {:>.10}", summary.entropy_p);
    println!("H(q)             = {:>.10}", summary.entropy_q);
    println!("D_KL(p || q)     = {:>.10}", summary.kl_pq);
    println!("D_KL(q || p)     = {:>.10}", summary.kl_qp);
    println!("H(p, q)          = {:>.10}", summary.cross_entropy_pq);
    println!("H(q, p)          = {:>.10}", summary.cross_entropy_qp);
    println!();
}

fn print_mutual_information_summary(
    title: &str,
    result: &MutualInformationResult,
) {
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
    println!("H(X)             = {:>.10}", result.entropy_x);
    println!("H(Y)             = {:>.10}", result.entropy_y);
    println!("H(X, Y)          = {:>.10}", result.joint_entropy);
    println!("I(X; Y)          = {:>.10}", result.mutual_information);
    println!("H(X) + H(Y) - H(X,Y) = {:>.10}",
        result.entropy_x + result.entropy_y - result.joint_entropy
    );
    println!();
}

fn main() -> Result<(), String> {
    // Example A:
    // Compare two discrete distributions and verify the relationship
    // H(p, q) = H(p) + D_KL(p || q).
    let p = normalize_probabilities(&[0.40, 0.30, 0.20, 0.10])?;
    let q = normalize_probabilities(&[0.35, 0.25, 0.25, 0.15])?;

    print_distribution("Distribution p", &p);
    print_distribution("Distribution q", &q);

    let info_summary = InformationMeasures {
        entropy_p: shannon_entropy(&p)?,
        entropy_q: shannon_entropy(&q)?,
        kl_pq: kl_divergence(&p, &q)?,
        kl_qp: kl_divergence(&q, &p)?,
        cross_entropy_pq: cross_entropy(&p, &q)?,
        cross_entropy_qp: cross_entropy(&q, &p)?,
    };

    print_information_summary(
        "Entropy, Divergence, and Cross-Entropy",
        &info_summary,
    );

    println!("Cross-Entropy Decomposition Check");
    println!("=================================");
    println!(
        "H(p) + D_KL(p || q) = {:>.10}",
        info_summary.entropy_p + info_summary.kl_pq
    );
    println!("H(p, q)             = {:>.10}", info_summary.cross_entropy_pq);
    println!(
        "absolute difference = {:>.10e}",
        (info_summary.entropy_p + info_summary.kl_pq - info_summary.cross_entropy_pq).abs()
    );
    println!();

    // Example B:
    // Build an empirical joint distribution from paired categorical observations,
    // then compute marginals and mutual information.
    let x_samples = vec![
        "Low", "Low", "Low", "Medium", "Medium", "Medium", "Medium",
        "High", "High", "High", "High", "High",
    ];
    let y_samples = vec![
        "A", "A", "B", "A", "B", "B", "C",
        "B", "C", "C", "C", "C",
    ];

    let (joint, x_levels, y_levels) = empirical_joint_distribution(&x_samples, &y_samples)?;
    let px = marginal_x(&joint)?;
    let py = marginal_y(&joint)?;
    let mi_result = mutual_information(&joint)?;

    print_joint_distribution(
        "Empirical Joint Distribution p_{ij}",
        &joint,
        &x_levels,
        &y_levels,
    );

    print_distribution("Marginal Distribution p_X", &px);
    print_distribution("Marginal Distribution p_Y", &py);
    print_mutual_information_summary("Mutual Information Summary", &mi_result);

    // Verify I(X;Y) = D_KL(p_XY || p_X p_Y).
    let product = product_distribution(&px, &py)?;
    let joint_as_flat = Distribution {
        probs: joint
            .probs
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect(),
    };
    let product_as_flat = Distribution {
        probs: product
            .probs
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect(),
    };
    let kl_joint_vs_product = kl_divergence(&joint_as_flat, &product_as_flat)?;

    println!("Mutual Information Identity Check");
    println!("=================================");
    println!("I(X; Y)                    = {:>.10}", mi_result.mutual_information);
    println!("D_KL(p_XY || p_X p_Y)      = {:>.10}", kl_joint_vs_product);
    println!(
        "absolute difference        = {:>.10e}",
        (mi_result.mutual_information - kl_joint_vs_product).abs()
    );
    println!();

    println!("Zero-Probability Convention Check");
    println!("=================================");
    let sparse = normalize_probabilities(&[0.50, 0.50, 0.00, 0.00])?;
    let h_sparse = shannon_entropy(&sparse)?;
    print_distribution("Sparse Distribution", &sparse);
    println!("H(sparse) = {:>.10}", h_sparse);
    println!("The zero entries contribute nothing, consistent with 0 log 0 = 0.");
    println!();

    Ok(())
}
```

Program 14.7.4 demonstrates a practical approach to computing information-theoretic quantities from discrete probability distributions while maintaining numerical stability and mathematical correctness. This reflects the central computational challenge discussed in Section 14.7.4: translating theoretical definitions of entropy and divergence into reliable numerical procedures that remain well-defined even in the presence of zero or near-zero probabilities.

The examples illustrate several key properties. The comparison of distributions $p$ and $q$ confirms the decomposition of cross-entropy into entropy and divergence, while the empirical joint distribution demonstrates how mutual information captures general dependence beyond linear or monotonic relationships. The verification of the identity $I(X;Y) = D_{\mathrm{KL}}(p_{XY} \mid p_X p_Y)$ highlights the deep connection between dependence and divergence.

The modular structure of the code allows straightforward extension to more advanced settings, including continuous approximations, kernel density estimates, or bias-corrected entropy estimators for high-dimensional data. This provides a foundation for modern applications in machine learning, signal processing, and statistical inference, where information-theoretic measures play a central role in understanding uncertainty and dependence.

# 14.8. Information-Theoretic Properties of Distributions

In many numerical computing pipelines, the primary object of interest is not a single dataset but an empirical distribution. Examples include histograms from stochastic simulations, class distributions in machine learning, contingency tables derived from categorical data, and symbolic sequences generated by dynamical systems. In such settings, information theory provides a principled framework for quantifying uncertainty, model mismatch, and dependence.

Modern developments emphasize that estimating these quantities reliably from finite data is a nontrivial problem, particularly in high-dimensional or large-alphabet regimes. As a result, information-theoretic tools are not only theoretical constructs but also computational objects whose numerical behavior must be carefully analyzed (Pinchas et al., 2024; Altieri et al., 2023; Calcagnile et al., 2026).

## 14.8.1. Distributions as Computational Objects

A discrete probability distribution over $I$ outcomes is represented as,

$$p = (p_1, \dots, p_I), \qquad p_i \ge 0, \quad \sum_{i=1}^{I} p_i = 1 \tag{14.8.1}$$

In practice, such distributions are often estimated from observed counts $n_i$ over $N$ samples:

$$\hat{p}_i = \frac{n_i}{N} \tag{14.8.2}$$

This simple “count–normalize” pipeline is ubiquitous in numerical computing. However, when the number of categories $I$ is large relative to $N$, estimation bias and sparsity (including zero counts) become critical issues.

For two discrete variables $X \in {1,\dots,I}$ and $Y \in {1,\dots,J}$, the joint distribution is represented as a matrix:

$$P = [p_{ij}], \qquad p_{ij} \ge 0, \quad \sum_{i=1}^{I} \sum_{j=1}^{J} p_{ij} = 1 \tag{14.8.3}$$

Marginal distributions are obtained by summation:

$$p_{i\cdot} = \sum_{j=1}^{J} p_{ij}, \qquad p_{\cdot j} = \sum_{i=1}^{I} p_{ij} \tag{14.8.4}$$

This matrix viewpoint connects directly to contingency tables and provides a concrete computational representation for entropy and dependence measures (Letizia et al., 2024).

## 14.8.2. Entropy, Cross-Entropy, and KL Divergence

The *Shannon entropy* of a distribution $p$ is defined as:

$$H(p) = - \sum_{i=1}^{I} p_i \log p_i, \qquad 0 \log 0 := 0 \tag{14.8.5}$$

Entropy quantifies uncertainty and plays multiple roles in numerical computing, including regularization, model selection, and coding-theoretic interpretations.

Given a “true” distribution $p$ and a model $q$, the *cross-entropy* is:

$$H(p, q) = - \sum_{i} p_i \log q_i \tag{14.8.6}$$

The *Kullback–Leibler (KL) divergence* is:

$$D_{\mathrm{KL}}(p | q) = \sum_{i} p_i \log \frac{p_i}{q_i} \tag{14.8.7}$$

These quantities are related by:

$$H(p, q) = H(p) + D_{\mathrm{KL}}(p | q) \tag{14.8.8}$$

KL divergence is nonnegative and equals zero if and only if $p = q$, making it a measure of model mismatch. In modern machine learning and signal processing, KL divergence often appears as an optimization objective and as a building block in broader classes of divergence measures (Letizia et al., 2024).

## 14.8.3. Conditional Entropy and Mutual Information

For a joint distribution $p_{ij}$, the entropies of individual variables are:

$$H(X) = - \sum_{i} p_{i\cdot} \log p_{i\cdot}, \qquad H(Y) = - \sum_{j} p_{\cdot j} \log p_{\cdot j} \tag{14.8.9}$$

and the joint entropy is:

$$H(X, Y) = - \sum_{i,j} p_{ij} \log p_{ij} \tag{14.8.10}$$

The *conditional entropy* is:

$$H(Y \mid X) = - \sum_{i,j} p_{ij} \log \frac{p_{ij}}{p_{i\cdot}} \tag{14.8.11}$$

and the chain rule states,

$$H(X, Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y) \tag{14.8.12}$$

The *mutual information* is defined as:

$$I(X; Y) = H(X) + H(Y) - H(X, Y) \tag{14.8.13}$$

which can also be written as:

$$I(X; Y) = \sum_{i,j} p_{ij} \log \frac{p_{ij}}{p_{i\cdot} p_{\cdot j}}= D_{\mathrm{KL}}(p_{XY} | p_X p_Y) \tag{14.8.14}$$

Mutual information satisfies,

$$I(X; Y) \ge 0, \quad I(X; Y) = 0 \iff X \text{ and } Y \text{ are independent} \tag{14.8.15}$$

This formulation highlights a key advantage: unlike correlation, mutual information captures *general dependence*, including nonlinear and non-monotonic relationships.

### Rust Implementation

Following the discussion in Section 14.8 on the information-theoretic properties of distributions, Program 14.8.1 provides a practical implementation of entropy, divergence, and dependence measures for discrete probability distributions. In many numerical computing workflows, distributions are constructed from empirical counts using the normalization process in Equation (14.8.2), and subsequent analysis relies on evaluating quantities such as entropy, cross-entropy, and mutual information. This program translates the definitions in Equations (14.8.5)–(14.8.14) into numerically stable procedures, ensuring correct handling of zero probabilities through the convention $0 \log 0 = 0$. It demonstrates both direct evaluation from prescribed distributions and computation from empirically constructed joint distributions, thereby illustrating how uncertainty and dependence can be quantified in practical computational settings.

At the core of the implementation is the `Distribution` structure, which represents a discrete probability vector as defined in Equation (14.8.1). The functions `normalize_from_weights` and `normalize_from_counts` implement the count–normalize pipeline described in Equation (14.8.2), converting raw weights or counts into valid probability distributions. The function `validate_distribution` ensures that the resulting vectors satisfy nonnegativity and normalization constraints, which are essential for the correct evaluation of all subsequent information-theoretic quantities.

The function `shannon_entropy` implements the entropy defined in Equation (14.8.5). It evaluates the sum $H(p)$ while skipping zero-probability terms to enforce the convention $0 \log 0 = 0$. This guarantees that entropy remains well-defined even for sparse distributions. The functions `cross_entropy` and `kl_divergence` implement Equations (14.8.6) and (14.8.7), respectively. These functions compute the discrepancy between two distributions and explicitly detect invalid cases where $q_i = 0$ while $p_i > 0$, which would lead to infinite divergence.

The program introduces the `JointDistribution` structure to represent the matrix form of a joint distribution described in Equation (14.8.3). The function `empirical_joint_distribution` constructs this matrix from paired categorical observations, while `joint_from_count_matrix` provides a direct pathway from contingency tables. The marginal distributions in Equation (14.8.4) are computed using the functions `marginal_x` and `marginal_y`, which sum over rows and columns of the joint matrix.

The function `joint_entropy` implements Equation (14.8.10), while `conditional_entropy_y_given_x` and `conditional_entropy_x_given_y` implement Equation (14.8.11) in both directions. These functions compute conditional uncertainty by evaluating ratios of joint and marginal probabilities, ensuring numerical stability through careful handling of zero values. The function `mutual_information` implements Equation (14.8.14), computing dependence directly from the joint distribution and its marginals.

The function `summarize_joint_information` aggregates these quantities and verifies the chain rule in Equation (14.8.12) by comparing $H(X,Y)$ with both $H(X)+H(Y|X)$ and $H(Y)+H(X|Y)$. The program also constructs the independent model $p_X p_Y$ using `product_distribution`, enabling verification of the identity $I(X;Y) = D_{\mathrm{KL}}(p_{XY} \mid p_X p_Y)$.

The `main` function demonstrates the implementation through three representative examples. The first evaluates entropy, cross-entropy, and KL divergence for two distributions derived from counts, verifying the decomposition in Equation (14.8.8). The second constructs an empirical joint distribution from categorical data and computes marginals, conditional entropies, and mutual information, verifying both the chain rule and the KL-based identity. The third example demonstrates the correct handling of sparse distributions, confirming that zero-probability terms do not contribute to entropy calculations.

```rust
// Program 14.8.1
// Information-Theoretic Properties of Distributions
//
// Problem statement:
// Given discrete counts or explicitly specified probability distributions,
// construct normalized marginal and joint distributions, then compute
// Shannon entropy, cross-entropy, KL divergence, conditional entropy,
// joint entropy, and mutual information. The program must handle zero
// probabilities correctly using the convention 0 log 0 = 0, and it must
// verify the chain rule and the KL-based identity for mutual information.

use std::collections::{BTreeMap, BTreeSet};

const EPS: f64 = 1.0e-15;

#[derive(Debug, Clone)]
struct Distribution {
    probs: Vec<f64>,
}

#[derive(Debug, Clone)]
struct JointDistribution {
    probs: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
}

#[derive(Debug, Clone)]
struct EntropyAndDivergenceSummary {
    entropy_p: f64,
    entropy_q: f64,
    cross_entropy_pq: f64,
    kl_pq: f64,
    cross_entropy_qp: f64,
    kl_qp: f64,
}

#[derive(Debug, Clone)]
struct JointInformationSummary {
    entropy_x: f64,
    entropy_y: f64,
    joint_entropy: f64,
    conditional_y_given_x: f64,
    conditional_x_given_y: f64,
    mutual_information: f64,
}

fn is_nonnegative(x: f64) -> bool {
    x >= -EPS
}

fn normalize_from_weights(weights: &[f64]) -> Result<Distribution, String> {
    if weights.is_empty() {
        return Err("distribution must contain at least one entry".to_string());
    }
    if weights.iter().any(|&w| !is_nonnegative(w)) {
        return Err("distribution contains a negative entry".to_string());
    }

    let total = weights.iter().sum::<f64>();
    if total <= EPS {
        return Err("distribution has zero total mass".to_string());
    }

    let probs = weights.iter().map(|&w| w / total).collect::<Vec<_>>();
    Ok(Distribution { probs })
}

fn normalize_from_counts(counts: &[usize]) -> Result<Distribution, String> {
    let weights = counts.iter().map(|&c| c as f64).collect::<Vec<_>>();
    normalize_from_weights(&weights)
}

fn validate_distribution(p: &Distribution) -> Result<(), String> {
    if p.probs.is_empty() {
        return Err("distribution must contain at least one entry".to_string());
    }
    if p.probs.iter().any(|&v| !is_nonnegative(v)) {
        return Err("distribution contains a negative probability".to_string());
    }
    let sum = p.probs.iter().sum::<f64>();
    if (sum - 1.0).abs() > 1.0e-10 {
        return Err(format!(
            "distribution is not normalized: sum = {:.16}",
            sum
        ));
    }
    Ok(())
}

fn validate_joint_distribution(joint: &JointDistribution) -> Result<(), String> {
    if joint.rows == 0 || joint.cols == 0 {
        return Err("joint distribution must have positive dimensions".to_string());
    }
    if joint.probs.len() != joint.rows {
        return Err("joint distribution row count mismatch".to_string());
    }
    if joint.probs.iter().any(|row| row.len() != joint.cols) {
        return Err("joint distribution column count mismatch".to_string());
    }
    if joint
        .probs
        .iter()
        .flat_map(|row| row.iter())
        .any(|&v| !is_nonnegative(v))
    {
        return Err("joint distribution contains a negative probability".to_string());
    }

    let sum = joint.probs.iter().flat_map(|row| row.iter()).sum::<f64>();
    if (sum - 1.0).abs() > 1.0e-10 {
        return Err(format!(
            "joint distribution is not normalized: sum = {:.16}",
            sum
        ));
    }
    Ok(())
}

fn shannon_entropy(p: &Distribution) -> Result<f64, String> {
    validate_distribution(p)?;

    let mut entropy = 0.0_f64;
    for &pi in &p.probs {
        if pi > EPS {
            entropy -= pi * pi.ln();
        }
    }
    Ok(entropy)
}

fn cross_entropy(p: &Distribution, q: &Distribution) -> Result<f64, String> {
    validate_distribution(p)?;
    validate_distribution(q)?;

    if p.probs.len() != q.probs.len() {
        return Err("cross-entropy requires equal-length distributions".to_string());
    }

    let mut value = 0.0_f64;
    for (&pi, &qi) in p.probs.iter().zip(q.probs.iter()) {
        if pi <= EPS {
            continue;
        }
        if qi <= EPS {
            return Err("cross-entropy is infinite because q_i = 0 where p_i > 0".to_string());
        }
        value -= pi * qi.ln();
    }
    Ok(value)
}

fn kl_divergence(p: &Distribution, q: &Distribution) -> Result<f64, String> {
    validate_distribution(p)?;
    validate_distribution(q)?;

    if p.probs.len() != q.probs.len() {
        return Err("KL divergence requires equal-length distributions".to_string());
    }

    let mut value = 0.0_f64;
    for (&pi, &qi) in p.probs.iter().zip(q.probs.iter()) {
        if pi <= EPS {
            continue;
        }
        if qi <= EPS {
            return Err("KL divergence is infinite because q_i = 0 where p_i > 0".to_string());
        }
        value += pi * (pi / qi).ln();
    }
    Ok(value)
}

fn summary_entropy_and_divergence(
    p: &Distribution,
    q: &Distribution,
) -> Result<EntropyAndDivergenceSummary, String> {
    Ok(EntropyAndDivergenceSummary {
        entropy_p: shannon_entropy(p)?,
        entropy_q: shannon_entropy(q)?,
        cross_entropy_pq: cross_entropy(p, q)?,
        kl_pq: kl_divergence(p, q)?,
        cross_entropy_qp: cross_entropy(q, p)?,
        kl_qp: kl_divergence(q, p)?,
    })
}

fn joint_from_count_matrix(counts: &[Vec<usize>]) -> Result<JointDistribution, String> {
    if counts.is_empty() || counts[0].is_empty() {
        return Err("count matrix must be nonempty".to_string());
    }

    let rows = counts.len();
    let cols = counts[0].len();

    if counts.iter().any(|row| row.len() != cols) {
        return Err("count matrix rows must all have the same length".to_string());
    }

    let total: usize = counts.iter().flat_map(|row| row.iter()).sum();
    if total == 0 {
        return Err("count matrix has zero total count".to_string());
    }

    let probs = counts
        .iter()
        .map(|row| {
            row.iter()
                .map(|&c| c as f64 / total as f64)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Ok(JointDistribution { probs, rows, cols })
}

fn empirical_joint_distribution(
    x: &[&str],
    y: &[&str],
) -> Result<(JointDistribution, Vec<String>, Vec<String>), String> {
    if x.len() != y.len() {
        return Err("x and y must have the same number of observations".to_string());
    }
    if x.is_empty() {
        return Err("empirical observations must be nonempty".to_string());
    }

    let x_levels: Vec<String> = x
        .iter()
        .map(|s| s.to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let y_levels: Vec<String> = y
        .iter()
        .map(|s| s.to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let x_index: BTreeMap<String, usize> = x_levels
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();

    let y_index: BTreeMap<String, usize> = y_levels
        .iter()
        .cloned()
        .enumerate()
        .map(|(j, v)| (v, j))
        .collect();

    let mut counts = vec![vec![0usize; y_levels.len()]; x_levels.len()];

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let i = *x_index
            .get(xi)
            .ok_or_else(|| "failed to index x label".to_string())?;
        let j = *y_index
            .get(yi)
            .ok_or_else(|| "failed to index y label".to_string())?;
        counts[i][j] += 1;
    }

    let joint = joint_from_count_matrix(&counts)?;
    Ok((joint, x_levels, y_levels))
}

fn marginal_x(joint: &JointDistribution) -> Result<Distribution, String> {
    validate_joint_distribution(joint)?;

    let probs = joint
        .probs
        .iter()
        .map(|row| row.iter().sum::<f64>())
        .collect::<Vec<_>>();

    Ok(Distribution { probs })
}

fn marginal_y(joint: &JointDistribution) -> Result<Distribution, String> {
    validate_joint_distribution(joint)?;

    let mut probs = vec![0.0_f64; joint.cols];
    for row in &joint.probs {
        for (j, &pij) in row.iter().enumerate() {
            probs[j] += pij;
        }
    }

    Ok(Distribution { probs })
}

fn joint_entropy(joint: &JointDistribution) -> Result<f64, String> {
    validate_joint_distribution(joint)?;

    let mut entropy = 0.0_f64;
    for row in &joint.probs {
        for &pij in row {
            if pij > EPS {
                entropy -= pij * pij.ln();
            }
        }
    }
    Ok(entropy)
}

fn conditional_entropy_y_given_x(joint: &JointDistribution) -> Result<f64, String> {
    validate_joint_distribution(joint)?;
    let px = marginal_x(joint)?;

    let mut value = 0.0_f64;
    for i in 0..joint.rows {
        let p_i_dot = px.probs[i];
        for j in 0..joint.cols {
            let p_ij = joint.probs[i][j];
            if p_ij <= EPS {
                continue;
            }
            value -= p_ij * (p_ij / p_i_dot).ln();
        }
    }
    Ok(value)
}

fn conditional_entropy_x_given_y(joint: &JointDistribution) -> Result<f64, String> {
    validate_joint_distribution(joint)?;
    let py = marginal_y(joint)?;

    let mut value = 0.0_f64;
    for i in 0..joint.rows {
        for j in 0..joint.cols {
            let p_j = py.probs[j];
            let p_ij = joint.probs[i][j];
            if p_ij <= EPS {
                continue;
            }
            value -= p_ij * (p_ij / p_j).ln();
        }
    }
    Ok(value)
}

fn product_distribution(px: &Distribution, py: &Distribution) -> Result<JointDistribution, String> {
    validate_distribution(px)?;
    validate_distribution(py)?;

    let rows = px.probs.len();
    let cols = py.probs.len();
    let mut probs = vec![vec![0.0_f64; cols]; rows];

    for (i, &pxi) in px.probs.iter().enumerate() {
        for (j, &pyj) in py.probs.iter().enumerate() {
            probs[i][j] = pxi * pyj;
        }
    }

    Ok(JointDistribution { probs, rows, cols })
}

fn flatten_joint_distribution(joint: &JointDistribution) -> Result<Distribution, String> {
    validate_joint_distribution(joint)?;
    Ok(Distribution {
        probs: joint
            .probs
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>(),
    })
}

fn mutual_information(joint: &JointDistribution) -> Result<f64, String> {
    validate_joint_distribution(joint)?;
    let px = marginal_x(joint)?;
    let py = marginal_y(joint)?;

    let mut value = 0.0_f64;
    for i in 0..joint.rows {
        for j in 0..joint.cols {
            let p_ij = joint.probs[i][j];
            if p_ij <= EPS {
                continue;
            }
            let denom = px.probs[i] * py.probs[j];
            if denom <= EPS {
                return Err("mutual information encountered zero marginal with positive joint mass".to_string());
            }
            value += p_ij * (p_ij / denom).ln();
        }
    }

    if value.abs() <= 1.0e-14 {
        value = 0.0;
    }
    Ok(value)
}

fn summarize_joint_information(joint: &JointDistribution) -> Result<JointInformationSummary, String> {
    let px = marginal_x(joint)?;
    let py = marginal_y(joint)?;
    let hx = shannon_entropy(&px)?;
    let hy = shannon_entropy(&py)?;
    let hxy = joint_entropy(joint)?;
    let hy_given_x = conditional_entropy_y_given_x(joint)?;
    let hx_given_y = conditional_entropy_x_given_y(joint)?;
    let mi = mutual_information(joint)?;

    Ok(JointInformationSummary {
        entropy_x: hx,
        entropy_y: hy,
        joint_entropy: hxy,
        conditional_y_given_x: hy_given_x,
        conditional_x_given_y: hx_given_y,
        mutual_information: mi,
    })
}

fn print_distribution(title: &str, p: &Distribution) {
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
    for (i, &pi) in p.probs.iter().enumerate() {
        println!("p[{i}] = {:>.10}", pi);
    }
    println!("sum  = {:>.10}", p.probs.iter().sum::<f64>());
    println!();
}

fn print_joint_distribution(
    title: &str,
    joint: &JointDistribution,
    row_labels: &[String],
    col_labels: &[String],
) {
    println!("{title}");
    println!("{}", "=".repeat(title.len()));

    print!("{:>14}", "");
    for label in col_labels {
        print!("{:>14}", label);
    }
    println!("{:>14}", "row sum");

    for (i, row) in joint.probs.iter().enumerate() {
        let row_sum = row.iter().sum::<f64>();
        print!("{:>14}", row_labels[i]);
        for &v in row {
            print!("{:>14.8}", v);
        }
        println!("{:>14.8}", row_sum);
    }

    print!("{:>14}", "col sum");
    for j in 0..joint.cols {
        let col_sum = joint.probs.iter().map(|row| row[j]).sum::<f64>();
        print!("{:>14.8}", col_sum);
    }
    println!("{:>14.8}", 1.0);
    println!();
}

fn main() -> Result<(), String> {
    // Example A:
    // Construct distributions from counts and evaluate entropy,
    // cross-entropy, and KL divergence.
    let counts_p = vec![40usize, 30, 20, 10];
    let counts_q = vec![35usize, 25, 25, 15];

    let p = normalize_from_counts(&counts_p)?;
    let q = normalize_from_counts(&counts_q)?;
    let summary = summary_entropy_and_divergence(&p, &q)?;

    println!("Section 14.8 Example A: Entropy, Cross-Entropy, and KL Divergence");
    println!("=================================================================");
    print_distribution("Distribution p", &p);
    print_distribution("Distribution q", &q);

    println!("Information-Theoretic Summary");
    println!("=============================");
    println!("H(p)                 = {:>.10}", summary.entropy_p);
    println!("H(q)                 = {:>.10}", summary.entropy_q);
    println!("H(p, q)              = {:>.10}", summary.cross_entropy_pq);
    println!("D_KL(p | q)          = {:>.10}", summary.kl_pq);
    println!("H(q, p)              = {:>.10}", summary.cross_entropy_qp);
    println!("D_KL(q | p)          = {:>.10}", summary.kl_qp);
    println!();

    println!("Cross-Entropy Identity Check");
    println!("============================");
    println!(
        "H(p) + D_KL(p | q)   = {:>.10}",
        summary.entropy_p + summary.kl_pq
    );
    println!("H(p, q)              = {:>.10}", summary.cross_entropy_pq);
    println!(
        "absolute difference  = {:>.10e}",
        (summary.entropy_p + summary.kl_pq - summary.cross_entropy_pq).abs()
    );
    println!();

    // Example B:
    // Build an empirical joint distribution from paired categorical observations,
    // then compute marginals, conditional entropies, and mutual information.
    let x_samples = vec![
        "Low", "Low", "Low",
        "Medium", "Medium", "Medium", "Medium",
        "High", "High", "High", "High", "High",
    ];

    let y_samples = vec![
        "A", "A", "B",
        "A", "B", "B", "C",
        "B", "C", "C", "C", "C",
    ];

    let (joint, x_levels, y_levels) = empirical_joint_distribution(&x_samples, &y_samples)?;
    let px = marginal_x(&joint)?;
    let py = marginal_y(&joint)?;
    let info = summarize_joint_information(&joint)?;

    println!("Section 14.8 Example B: Joint, Conditional, and Mutual Information");
    println!("==================================================================");
    print_joint_distribution("Empirical Joint Distribution p_{ij}", &joint, &x_levels, &y_levels);
    print_distribution("Marginal Distribution p_X", &px);
    print_distribution("Marginal Distribution p_Y", &py);

    println!("Joint Information Summary");
    println!("=========================");
    println!("H(X)                 = {:>.10}", info.entropy_x);
    println!("H(Y)                 = {:>.10}", info.entropy_y);
    println!("H(X, Y)              = {:>.10}", info.joint_entropy);
    println!("H(Y | X)             = {:>.10}", info.conditional_y_given_x);
    println!("H(X | Y)             = {:>.10}", info.conditional_x_given_y);
    println!("I(X; Y)              = {:>.10}", info.mutual_information);
    println!();

    println!("Chain Rule Checks");
    println!("=================");
    println!(
        "H(X) + H(Y | X)     = {:>.10}",
        info.entropy_x + info.conditional_y_given_x
    );
    println!(
        "H(Y) + H(X | Y)     = {:>.10}",
        info.entropy_y + info.conditional_x_given_y
    );
    println!("H(X, Y)              = {:>.10}", info.joint_entropy);
    println!(
        "difference (first)   = {:>.10e}",
        (info.entropy_x + info.conditional_y_given_x - info.joint_entropy).abs()
    );
    println!(
        "difference (second)  = {:>.10e}",
        (info.entropy_y + info.conditional_x_given_y - info.joint_entropy).abs()
    );
    println!();

    let product = product_distribution(&px, &py)?;
    let joint_flat = flatten_joint_distribution(&joint)?;
    let product_flat = flatten_joint_distribution(&product)?;
    let mi_via_kl = kl_divergence(&joint_flat, &product_flat)?;

    println!("Mutual Information Identity Check");
    println!("=================================");
    println!("I(X; Y)              = {:>.10}", info.mutual_information);
    println!("D_KL(p_XY | p_X p_Y) = {:>.10}", mi_via_kl);
    println!(
        "absolute difference  = {:>.10e}",
        (info.mutual_information - mi_via_kl).abs()
    );
    println!();

    // Example C:
    // Demonstrate the zero-probability convention 0 log 0 = 0.
    let sparse = normalize_from_weights(&[0.5, 0.5, 0.0, 0.0])?;
    let sparse_entropy = shannon_entropy(&sparse)?;

    println!("Section 14.8 Example C: Zero-Probability Convention");
    println!("===================================================");
    print_distribution("Sparse Distribution", &sparse);
    println!("H(sparse)            = {:>.10}", sparse_entropy);
    println!("Zero-mass terms contribute nothing, consistent with 0 log 0 = 0.");
    println!();

    Ok(())
}
```

Program 14.8.1 demonstrates a practical framework for computing information-theoretic measures from discrete distributions while ensuring numerical stability and consistency with theoretical definitions. This reflects the central computational challenge discussed in Section 14.8: translating abstract measures of uncertainty and dependence into reliable numerical procedures that remain robust in finite-precision environments.

The examples illustrate key structural properties of information theory. The comparison of distributions confirms the relationship between entropy, cross-entropy, and KL divergence, while the empirical joint distribution demonstrates how mutual information captures general dependence beyond linear or monotonic relationships. The verification of the chain rule and the KL-based identity highlights the internal consistency of these measures and their interpretation as components of a unified framework.

The modular design of the program allows straightforward extension to more advanced settings, including bias-corrected entropy estimators, high-dimensional distributions, and model-based approaches for structured data. This provides a foundation for modern applications in machine learning, signal processing, and scientific computing, where accurate quantification of uncertainty and dependence is essential.

## 14.8.4. Estimation Challenges, Modern Developments, and Applications

In the large-alphabet, small-sample regime, naive plug-in estimators for entropy can be strongly biased. This bias arises because empirical frequency estimates tend to underrepresent rare events, which play a disproportionately important role in entropy calculations. As a consequence, the plug-in estimator typically underestimates the true entropy when the sample size is insufficient relative to the support size. No single estimator is uniformly optimal, and performance depends sensitively on the underlying distribution, the sparsity of observations, and the sampling regime (Pinchas et al., 2024). This dependence motivates the use of bias-corrected estimators, Bayesian approaches, or shrinkage techniques tailored to the specific data characteristics.

Real-world data often exhibit dependence across space or time. Such dependencies violate the assumptions of independence underlying many classical estimators and require more sophisticated modeling. Model-based entropy estimation approaches incorporate covariates and structured dependencies, enabling estimation of “entropy surfaces” that vary across domain or time (Altieri et al., 2023). These approaches allow entropy to be treated not as a single scalar quantity but as a function reflecting spatial heterogeneity or temporal evolution, thereby providing a richer and more informative description of complex systems.

For dynamical systems and time series, entropy-rate estimation is particularly challenging. The entropy rate captures the average uncertainty per unit time and depends on the full temporal dependence structure of the process. Accurate estimation therefore requires careful handling of correlations, memory effects, and finite sample limitations. Recent benchmarking studies using systems with known theoretical entropy values demonstrate that different estimators can exhibit significant bias depending on the system, emphasizing the need for careful validation and method selection (Calcagnile et al., 2026). In practice, this often necessitates comparing multiple estimators and validating results against known or simulated benchmarks.

### Mutual Information Estimation

Modern methods often rely on variational formulations, including neural-network-based estimators that approximate density ratios. These approaches transform the estimation problem into an optimization problem, where flexible function approximators are trained to capture dependencies between variables. Sampling strategies, such as derangements rather than simple shuffles, can significantly affect estimator accuracy and computational complexity (Letizia et al., 2024). The choice of sampling scheme influences both bias and variance, and careful design is required to ensure stable and reliable estimates in high-dimensional settings.

### Applications

In experimental design, mutual information is used to select measurements that maximize expected information gain about unknown quantities, leading to more efficient and informative data acquisition strategies. In machine learning, it is used to quantify representation quality and dependence between features, playing a central role in tasks such as feature selection, representation learning, and model interpretability. In scientific computing, it provides a powerful alternative to correlation when relationships are nonlinear or high-dimensional, enabling the detection of complex dependencies that are invisible to linear measures.

# 14.9. Do Two-Dimensional Distributions Differ?

In many applications, observations are naturally multivariate, and one must determine whether two datasets arise from the same underlying joint distribution. Such situations occur frequently in fields such as image analysis, spatial statistics, and multivariate experimental measurements, where each observation consists of multiple correlated components. Given samples in $(\mathbb{R}^2)$, the problem generalizes the classical one-dimensional two-sample test to a setting in which both marginal behavior and dependence structure must be taken into account. Formally, we consider two independent samples:

$$
\{\mathbf{X}_i\}_{i=1}^{n_1}, \qquad \{\mathbf{Y}_j\}_{j=1}^{n_2}, \quad
\mathbf{X}_i, \mathbf{Y}_j \in \mathbb{R}^2
\tag{14.9.1}
$$

with unknown distributions $F_1$ and $F_2$, and test,

$$
H_0: F_1 = F_2
\tag{14.9.2}
$$

The hypothesis $H_0$ asserts that the two samples are drawn from the same joint distribution, meaning that not only their individual coordinate distributions but also their dependence structure coincide. Rejecting this hypothesis therefore indicates a difference that may arise from shifts in location, changes in spread, or more subtle alterations in correlation or joint behavior.

Unlike the one-dimensional case, there is no natural ordering of points in $(\mathbb{R}^2)$. In one dimension, many classical procedures rely on sorting observations and comparing cumulative distribution functions along a single axis. In higher dimensions, however, such an ordering does not exist, and the concept of a cumulative distribution must be interpreted in terms of regions rather than intervals. As a result, classical tests such as the Kolmogorov–Smirnov statistic cannot be directly extended without modification.

A widely used approach is the Fasano–Franceschini (FF) test, which generalizes the KS idea by comparing empirical distributions over quadrant regions (Puritz et al., 2023). Specifically, for each point in the combined sample, the plane is partitioned into four quadrants, and the empirical distribution functions of the two samples are compared within each region. The test statistic is defined as the maximum discrepancy observed across all such quadrant-based comparisons, thereby capturing differences in both marginal distributions and spatial structure. This construction preserves the spirit of the KS test while adapting it to the geometric complexity of two-dimensional data.

## 14.9.1. Quadrant-Based Empirical Differences

For any point $(\mathbf{p} \in \mathbb{R}^2)$, the plane is partitioned into four quadrants (orthants) with vertex at $(\mathbf{p})$. Geometrically, each sample point acts as a pivot that divides the plane into four regions, enabling localized comparison of empirical mass. This construction provides a natural way to define cumulative behavior in two dimensions, where intervals are replaced by regions determined by coordinate-wise inequalities relative to $\mathbf{p}$.

Define indicator functions:

$$
I_j(\mathbf{x} \mid \mathbf{p}) =
\begin{cases}
1, & \text{if } \mathbf{x} \text{ lies in quadrant } j \text{ relative to } \mathbf{p}, \\
0, & \text{otherwise},
\end{cases}
\quad j = 1, \dots, 4
\tag{14.9.3}
$$

These indicator functions encode whether a given observation contributes to the empirical mass within a particular quadrant. By summing these indicators over a sample and normalizing by the sample size, one obtains an empirical estimate of the probability mass contained in each quadrant.

For each point $(\mathbf{p})$, define the quadrant-based difference:

$$
D(\mathbf{p} \mid X, Y) =
\max_{j=1,\dots,4}
\left|
\frac{1}{n_1} \sum_{i=1}^{n_1} I_j(\mathbf{X}_i \mid \mathbf{p})
-
\frac{1}{n_2} \sum_{k=1}^{n_2} I_j(\mathbf{Y}_k \mid \mathbf{p})
\right|
\tag{14.9.4}
$$

This quantity measures the largest discrepancy between the two empirical distributions when restricted to any of the four quadrants defined by $\mathbf{p}$. In effect, it captures the maximum local difference in probability mass allocation around the pivot point, making it sensitive to both marginal and joint differences between the samples.

This construction can be interpreted as a two-dimensional empirical distribution function, where probability mass is accumulated over quadrants rather than intervals. The statistic therefore represents a supremum norm of the difference between empirical measures over all quadrant regions. In this sense, it generalizes the Kolmogorov–Smirnov framework by replacing one-dimensional cumulative sums with region-based accumulations adapted to the geometry of $\mathbb{R}^2$.

Evaluating this quantity over all sample points yields:

$$
D_1 = \max_{1 \le i \le n_1} D(\mathbf{X}_i \mid X, Y), \qquad
D_2 = \max_{1 \le j \le n_2} D(\mathbf{Y}_j \mid X, Y)
\tag{14.9.5}
$$

and the combined statistic,

$$
D_0 = \frac{D_1 + D_2}{2}
\tag{14.9.6}
$$

The quantities $D_1$ and $D_2$ correspond to scanning the discrepancy using pivot points drawn from each sample separately. Averaging $D_1$ and $D_2$ ensures symmetry between the two samples, preventing the test from being biased toward one dataset due to differences in sample size or spatial configuration.

A scaled version is often used:

$$
\mathcal{D} = C_{n_1,n_2} \, D_0, \qquad
C_{n_1,n_2} = 2 \sqrt{\frac{n_1 n_2}{n_1 + n_2}}
\tag{14.9.7}
$$

The scaling factor $C_{n_1,n_2}$ adjusts for the sample sizes and places the statistic on a comparable scale across different experiments. This normalization is analogous to that used in classical two-sample tests, where the variability of empirical differences depends on the effective sample size.

Large values of $(\mathcal{D})$ indicate deviations from the null hypothesis $(F_1 = F_2)$. Intuitively, a large statistic reflects the presence of regions in the plane where the two samples allocate substantially different amounts of probability mass, signaling differences in location, spread, or dependence structure between the underlying distributions.

### Rust Implementation

Following the discussion in Section 14.9.1 on quadrant-based empirical differences for two-dimensional distributions, Program 14.9.1 provides a practical implementation of the Fasano–Franceschini two-sample test. In multivariate settings, the absence of a natural ordering prevents direct application of one-dimensional cumulative distribution comparisons, requiring instead a region-based approach. This program implements the quadrant-based construction described in Equations (14.9.3)–(14.9.7), where each sample point serves as a pivot that partitions the plane into four regions. By comparing empirical mass allocations across these quadrants and identifying the maximum discrepancy, the program captures differences in both marginal behavior and dependence structure. The implementation demonstrates how a geometric reinterpretation of cumulative distributions can be translated into an efficient computational procedure for detecting distributional differences in two dimensions.

At the core of the implementation is the `Point2` structure, which represents a two-dimensional observation $(x,y)$. This abstraction allows the samples $X$ and $Y$ in Equation (14.9.1) to be stored as arrays of points in $\mathbb{R}^2$. The geometric structure of the algorithm is built upon the function `quadrant_index`, which implements the indicator functions defined in Equation (14.9.3). For a given pivot $\mathbf{p}$, it determines the quadrant in which each observation lies based on coordinate-wise comparisons.

The function `count_quadrants` computes the empirical mass within each quadrant for a given sample relative to a pivot. By iterating over all points and accumulating counts, it produces the quantities needed to approximate the sums appearing in Equation (14.9.4). These counts are normalized by sample size within `discrepancy_at_pivot`, where the function evaluates the quadrant-based difference $D(\mathbf{p} \mid X, Y)$ by comparing normalized masses for the two samples and selecting the maximum absolute difference across the four quadrants.

The function `max_discrepancy_over_pivots` implements the pivot scanning described in Equation (14.9.5). It evaluates the discrepancy at each candidate pivot drawn from a sample and records the maximum value, yielding either $D_1$ or $D_2$ depending on the choice of pivot set. This approach corresponds to the $O((n_1+n_2)^2)$ strategy discussed in Section 14.9.2, where the search is restricted to sample points rather than all possible locations in the plane.

The function `fasano_franceschini_statistic` combines these results to compute the final test statistic. It evaluates $D_1$ and $D_2$, forms their average $D_0$ as in Equation (14.9.6), and applies the scaling factor $C_{n_1,n_2}$ from Equation (14.9.7) to obtain the normalized statistic $\mathcal{D}$. This scaling ensures that the statistic is comparable across different sample sizes and reflects the effective sample size of the two datasets.

The `main` function demonstrates the implementation using two synthetic samples in $\mathbb{R}^2$. It first prints the sample points, then computes the maximum discrepancy over pivots drawn from each sample, and finally reports the aggregated statistic. Additional diagnostic output shows the quadrant-wise mass differences at the maximizing pivots, providing insight into where the largest discrepancies occur in the plane.

```rust
// Program 14.9.1
// Fasano–Franceschini Quadrant-Based Two-Sample Test in R^2
//
// Problem statement:
// Given two independent samples X = {X_i}_{i=1}^{n1} and Y = {Y_j}_{j=1}^{n2},
// with X_i, Y_j in R^2, compute the quadrant-based empirical discrepancy
// statistic of the Fasano–Franceschini type. For each pivot point p drawn from
// the combined sample, partition the plane into four quadrants relative to p,
// compare empirical masses of the two samples in each quadrant, and record the
// maximum discrepancy. Then compute D1, D2, D0, and the scaled statistic
// 𝒟 = C_{n1,n2} D0.

#[derive(Debug, Clone, Copy)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Debug, Clone, Copy)]
struct QuadrantCounts {
    q1: usize, // x >= px, y >= py
    q2: usize, // x <  px, y >= py
    q3: usize, // x <  px, y <  py
    q4: usize, // x >= px, y <  py
}

impl QuadrantCounts {
    fn zero() -> Self {
        Self {
            q1: 0,
            q2: 0,
            q3: 0,
            q4: 0,
        }
    }

    fn as_array(self) -> [usize; 4] {
        [self.q1, self.q2, self.q3, self.q4]
    }
}

#[derive(Debug, Clone, Copy)]
struct PivotDiscrepancy {
    pivot: Point2,
    value: f64,
    quadrant_x_fractions: [f64; 4],
    quadrant_y_fractions: [f64; 4],
    absolute_differences: [f64; 4],
}

#[derive(Debug, Clone)]
struct FFStatistic {
    d1: f64,
    d2: f64,
    d0: f64,
    scale: f64,
    scaled_statistic: f64,
    best_x_pivot: PivotDiscrepancy,
    best_y_pivot: PivotDiscrepancy,
}

fn quadrant_index(pivot: Point2, point: Point2) -> usize {
    match (point.x >= pivot.x, point.y >= pivot.y) {
        (true, true) => 0,   // q1
        (false, true) => 1,  // q2
        (false, false) => 2, // q3
        (true, false) => 3,  // q4
    }
}

fn count_quadrants(sample: &[Point2], pivot: Point2) -> QuadrantCounts {
    let mut counts = QuadrantCounts::zero();

    for &pt in sample {
        match quadrant_index(pivot, pt) {
            0 => counts.q1 += 1,
            1 => counts.q2 += 1,
            2 => counts.q3 += 1,
            3 => counts.q4 += 1,
            _ => unreachable!("Quadrant index must lie in 0..4"),
        }
    }

    counts
}

fn discrepancy_at_pivot(x_sample: &[Point2], y_sample: &[Point2], pivot: Point2) -> PivotDiscrepancy {
    assert!(
        !x_sample.is_empty() && !y_sample.is_empty(),
        "Both samples must be nonempty."
    );

    let counts_x = count_quadrants(x_sample, pivot).as_array();
    let counts_y = count_quadrants(y_sample, pivot).as_array();

    let n1 = x_sample.len() as f64;
    let n2 = y_sample.len() as f64;

    let mut frac_x = [0.0_f64; 4];
    let mut frac_y = [0.0_f64; 4];
    let mut abs_diff = [0.0_f64; 4];
    let mut max_diff = 0.0_f64;

    for j in 0..4 {
        frac_x[j] = counts_x[j] as f64 / n1;
        frac_y[j] = counts_y[j] as f64 / n2;
        abs_diff[j] = (frac_x[j] - frac_y[j]).abs();
        if abs_diff[j] > max_diff {
            max_diff = abs_diff[j];
        }
    }

    PivotDiscrepancy {
        pivot,
        value: max_diff,
        quadrant_x_fractions: frac_x,
        quadrant_y_fractions: frac_y,
        absolute_differences: abs_diff,
    }
}

fn max_discrepancy_over_pivots(
    pivot_sample: &[Point2],
    x_sample: &[Point2],
    y_sample: &[Point2],
) -> PivotDiscrepancy {
    assert!(
        !pivot_sample.is_empty(),
        "The pivot sample must contain at least one point."
    );

    let mut best = discrepancy_at_pivot(x_sample, y_sample, pivot_sample[0]);

    for &pivot in &pivot_sample[1..] {
        let candidate = discrepancy_at_pivot(x_sample, y_sample, pivot);
        if candidate.value > best.value {
            best = candidate;
        }
    }

    best
}

fn fasano_franceschini_statistic(x_sample: &[Point2], y_sample: &[Point2]) -> FFStatistic {
    assert!(
        !x_sample.is_empty() && !y_sample.is_empty(),
        "Both samples must be nonempty."
    );

    let best_x = max_discrepancy_over_pivots(x_sample, x_sample, y_sample);
    let best_y = max_discrepancy_over_pivots(y_sample, x_sample, y_sample);

    let d1 = best_x.value;
    let d2 = best_y.value;
    let d0 = 0.5 * (d1 + d2);

    let n1 = x_sample.len() as f64;
    let n2 = y_sample.len() as f64;
    let scale = 2.0 * ((n1 * n2) / (n1 + n2)).sqrt();
    let scaled_statistic = scale * d0;

    FFStatistic {
        d1,
        d2,
        d0,
        scale,
        scaled_statistic,
        best_x_pivot: best_x,
        best_y_pivot: best_y,
    }
}

fn print_sample(title: &str, sample: &[Point2]) {
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
    for (i, pt) in sample.iter().enumerate() {
        println!("  [{:>2}] ({:>.10}, {:>.10})", i, pt.x, pt.y);
    }
    println!();
}

fn print_pivot_discrepancy(title: &str, info: &PivotDiscrepancy) {
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
    println!(
        "pivot                     = ({:>.10}, {:>.10})",
        info.pivot.x, info.pivot.y
    );
    println!("D(p | X, Y)               = {:>.10}", info.value);
    println!();

    println!(
        "{:>10} {:>18} {:>18} {:>18}",
        "quadrant", "mass in X", "mass in Y", "abs. difference"
    );
    for j in 0..4 {
        println!(
            "{:>10} {:>18.10} {:>18.10} {:>18.10}",
            format!("Q{}", j + 1),
            info.quadrant_x_fractions[j],
            info.quadrant_y_fractions[j],
            info.absolute_differences[j]
        );
    }
    println!();
}

fn main() {
    // Example:
    // Two planar samples with visibly different location/spread structure.
    // The program evaluates the FF-type quadrant discrepancy statistic.
    let x_sample = vec![
        Point2 { x: 0.80, y: 1.00 },
        Point2 { x: 1.10, y: 1.20 },
        Point2 { x: 1.40, y: 0.90 },
        Point2 { x: 1.70, y: 1.50 },
        Point2 { x: 1.90, y: 1.10 },
        Point2 { x: 2.10, y: 1.80 },
        Point2 { x: 2.40, y: 1.40 },
        Point2 { x: 2.70, y: 2.00 },
    ];

    let y_sample = vec![
        Point2 { x: 1.60, y: 0.40 },
        Point2 { x: 1.90, y: 0.70 },
        Point2 { x: 2.20, y: 0.60 },
        Point2 { x: 2.50, y: 0.90 },
        Point2 { x: 2.80, y: 1.00 },
        Point2 { x: 3.00, y: 1.20 },
        Point2 { x: 3.30, y: 1.10 },
        Point2 { x: 3.50, y: 1.40 },
    ];

    print_sample("Sample X", &x_sample);
    print_sample("Sample Y", &y_sample);

    let ff = fasano_franceschini_statistic(&x_sample, &y_sample);

    print_pivot_discrepancy("Maximum Discrepancy Over X Pivots", &ff.best_x_pivot);
    print_pivot_discrepancy("Maximum Discrepancy Over Y Pivots", &ff.best_y_pivot);

    println!("Fasano–Franceschini Summary");
    println!("===========================");
    println!("n1                        = {}", x_sample.len());
    println!("n2                        = {}", y_sample.len());
    println!("D1                        = {:>.10}", ff.d1);
    println!("D2                        = {:>.10}", ff.d2);
    println!("D0 = (D1 + D2)/2          = {:>.10}", ff.d0);
    println!("C_(n1,n2)                 = {:>.10}", ff.scale);
    println!("𝒟 = C_(n1,n2) * D0        = {:>.10}", ff.scaled_statistic);
    println!();

    println!("Interpretation");
    println!("==============");
    println!("Larger values of 𝒟 indicate stronger evidence that the two planar");
    println!("samples do not arise from the same underlying joint distribution.");
}
```

Program 14.9.1 demonstrates a practical realization of the quadrant-based approach to multivariate two-sample testing. This method extends the intuition of cumulative distribution comparisons to higher dimensions by replacing intervals with geometric regions defined relative to pivot points.

The example illustrates how differences between two samples can be localized within specific quadrants, revealing variations in spatial structure that would not be captured by marginal comparisons alone. The computation of $D_1$, $D_2$, and the scaled statistic $\mathcal{D}$ reflects the combined effect of discrepancies across all candidate pivots, providing a robust measure of distributional difference.

The modular design of the implementation allows for straightforward extensions, including permutation-based significance testing and optimization using spatial data structures. Such extensions are essential for large-scale applications, where efficient evaluation of quadrant counts and repeated resampling become critical. This program thus provides a foundation for modern multivariate hypothesis testing in numerical and data-driven environments.

## 14.9.2. Algorithmic Aspects and Complexity

A naive implementation evaluates quadrant differences at every point and counts observations in each region. For each candidate pivot $\mathbf{p}$, one must determine how many sample points from each dataset fall into each of the four quadrants, which requires scanning all observations. Repeating this procedure for all candidate points leads to:

$$O\big((n_1 + n_2)^3\big)\tag{14.9.8}$$

operations. This cubic complexity quickly becomes prohibitive even for moderately sized datasets, making the direct approach impractical in most applications.

Fasano and Franceschini showed that restricting origins to sample points reduces complexity to,

$$O\big((n_1 + n_2)^2\big) \tag{14.9.9}$$

The key observation is that it is sufficient to evaluate the statistic only at pivot points drawn from the combined sample, rather than over all possible locations in the plane. This reduces the number of candidate pivots from a continuum to a finite set of size $n_1 + n_2$, while still capturing all relevant extrema of the empirical differences.

Further improvements use spatial indexing structures, such as range-counting data structures or KD-trees, which organize the data in a hierarchical manner to enable efficient region queries. These structures allow quadrant counts to be computed without scanning all points explicitly, reducing the cost of each query to,

$$O(\log n) \ \text{or} \ O(\log^2 n)$$

time per query and yielding near,

$$O(n \log n)\tag{14.9.10}$$

performance in practice. The precise complexity depends on the data structure used and the dimensionality, but the improvement over quadratic methods is substantial for large datasets.

Modern implementations exploit these techniques to handle large datasets efficiently (Puritz et al., 2023). In practice, careful engineering of data structures, memory layout, and query strategies is essential to achieve the expected performance gains, especially in applications involving high data volume or repeated evaluations of the test statistic.

## 14.9.3. Alternative Multivariate Two-Sample Tests

While the FF test generalizes the KS approach, modern methods provide alternative ways to compare multivariate distributions. These approaches differ in how they measure discrepancies between distributions, with some focusing on geometric distances, others on functional embeddings, and still others on structural properties of the data. Each method offers distinct advantages depending on the nature of the underlying distributions and the type of differences one aims to detect.

### Energy Tests

A widely used statistic is:

$$
E^2 =
\frac{2}{n_1 n_2} \sum_{i,j} \lvert \mathbf{X}_i - \mathbf{Y}_j \rvert
-
\frac{1}{n_1^2} \sum_{i,k} \lvert \mathbf{X}_i - \mathbf{X}_k \rvert
-
\frac{1}{n_2^2} \sum_{j,\ell} \lvert \mathbf{Y}_j - \mathbf{Y}_\ell \rvert,
\tag{14.9.11}
$$

which equals zero if and only if the distributions match. This statistic compares average pairwise distances across and within samples, effectively measuring how separated the two samples are in the ambient space. The cross-term reflects between-sample distances, while the within-sample terms provide a normalization based on internal variability. As a result, the energy statistic captures both location and distributional differences in a unified geometric framework.

### Kernel-based Methods

Maximum mean discrepancy (MMD) compares distributions in reproducing kernel Hilbert spaces, enabling detection of differences in high-dimensional or structured data (Alden et al., 2025). By mapping data into a feature space defined by a kernel function, MMD transforms the problem into comparing mean embeddings of the distributions. This approach is particularly powerful because appropriately chosen kernels can capture complex, nonlinear relationships that are not accessible through direct distance-based comparisons in the original space.

### Graph-based Tests

Methods based on nearest-neighbor graphs or minimum spanning trees detect differences via connectivity patterns between samples. In these approaches, a graph is constructed on the pooled data, and the extent to which edges connect points from different samples is analyzed. If the two samples are drawn from the same distribution, the graph tends to mix points from both samples uniformly; deviations from this behavior indicate distributional differences. These methods are especially effective in detecting local structural changes and clustering behavior.

Simulation studies show that no single method is uniformly optimal. For example, energy-based tests often achieve high average power, while other methods may perform better for specific alternatives. The performance of each method depends on factors such as dimensionality, sample size, and the type of discrepancy between distributions. In practice, combining multiple methods can improve robustness and provide complementary insights into the nature of the differences (Rolke, 2025).

### Rust Implementation

Following the discussion in Section 14.9.3 on alternative multivariate two-sample tests, Program 14.9.2 provides a practical implementation of the energy-based approach for comparing multivariate distributions. Unlike quadrant-based methods, the energy test measures discrepancies through pairwise distances, offering a geometric interpretation of distributional differences. This program implements the statistic defined in Equation (14.9.11), combining between-sample and within-sample distances to quantify separation between distributions. In addition, it incorporates a permutation-based inference procedure to assess statistical significance, reflecting modern practice in multivariate hypothesis testing where analytic null distributions are often unavailable. The implementation demonstrates how distance-based comparisons can be translated into a robust computational framework for detecting general differences between multivariate samples.

At the core of the implementation is the `Point` structure, which represents a data point in $\mathbb{R}^d$ as a vector of coordinates. This abstraction allows the program to operate in arbitrary dimensions while maintaining a uniform interface for distance computations. The method `distance` computes the Euclidean norm between two points, forming the basic building block for all pairwise comparisons appearing in Equation (14.9.11).

The function `average_cross_distance` evaluates the first term in Equation (14.9.11), computing the average distance between all pairs $(X_i, Y_j)$. This cross-sample term captures the separation between the two datasets in the ambient space. The functions `average_within_distance` applied to each sample compute the remaining terms, corresponding to within-sample distances $(X_i, X_k)$ and $(Y_j, Y_\ell)$. These terms quantify the internal dispersion of each sample and provide the normalization necessary to distinguish true distributional differences from intrinsic variability.

The function `energy_statistic_components` combines these quantities to compute the full statistic $E^2$. By assembling the cross and within components explicitly, the implementation reflects the structure of Equation (14.9.11) and provides intermediate diagnostics that help interpret the contributions of each term.

To assess statistical significance, the program implements a permutation test through the function `permutation_test_energy`. This function pools the two samples, randomly reassigns labels, and recomputes the energy statistic for each permutation. The proportion of permuted statistics exceeding the observed value yields an empirical p-value, with add-one smoothing applied to ensure numerical stability. The randomization is controlled by the `LcgRng` structure, which provides a reproducible pseudo-random number generator and supports in-place shuffling of the pooled sample.

The auxiliary functions `mean`, `standard_deviation`, and `quantile` compute summary statistics of the permutation distribution, enabling diagnostic analysis of the null distribution. These quantities provide insight into the variability and shape of the permutation-based reference distribution.

The `main` function demonstrates the implementation using two two-dimensional samples with differing spatial configurations. It first computes the energy statistic and its components, then performs a permutation test to estimate the p-value. Diagnostic summaries of the permutation distribution are printed, allowing direct comparison between the observed statistic and its null distribution.

```rust
// Program 14.9.2: Energy Distance Two-Sample Test for Multivariate Samples
//
// Problem statement:
// Given two independent samples X = {X_i}_{i=1}^{n1} and Y = {Y_j}_{j=1}^{n2},
// with X_i, Y_j in R^d, compute the energy-test statistic
//
//   E^2 = (2 / (n1 n2)) sum_{i,j} |X_i - Y_j|
//       - (1 / n1^2) sum_{i,k} |X_i - X_k|
//       - (1 / n2^2) sum_{j,l} |Y_j - Y_l|
//
// as described in Equation (14.9.11).
//
// The program also performs a permutation test to estimate a p-value for the
// null hypothesis that the two samples arise from the same underlying
// multivariate distribution. The implementation is written for general
// dimension d, although the demonstration uses two-dimensional samples.

use std::fmt;

#[derive(Debug, Clone)]
struct Point {
    coords: Vec<f64>,
}

impl Point {
    fn new(coords: Vec<f64>) -> Self {
        assert!(!coords.is_empty(), "point dimension must be positive");
        Self { coords }
    }

    fn dim(&self) -> usize {
        self.coords.len()
    }

    fn distance(&self, other: &Self) -> f64 {
        assert_eq!(
            self.dim(),
            other.dim(),
            "points must have the same dimension"
        );
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            .sqrt()
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, x) in self.coords.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.10}", x)?;
        }
        write!(f, ")")
    }
}

#[derive(Debug, Clone)]
struct EnergyComponents {
    cross_average: f64,
    within_x_average: f64,
    within_y_average: f64,
    statistic: f64,
}

#[derive(Debug, Clone)]
struct PermutationResult {
    observed_statistic: f64,
    p_value: f64,
    exceedances: usize,
    permutations: usize,
    permuted_statistics: Vec<f64>,
}

#[derive(Debug, Clone)]
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005_u64)
            .wrapping_add(1442695040888963407_u64);
        self.state
    }

    fn gen_usize(&mut self, upper_exclusive: usize) -> usize {
        assert!(upper_exclusive > 0, "upper bound must be positive");
        (self.next_u64() % upper_exclusive as u64) as usize
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        if values.len() <= 1 {
            return;
        }
        for i in (1..values.len()).rev() {
            let j = self.gen_usize(i + 1);
            values.swap(i, j);
        }
    }
}

fn validate_sample(sample: &[Point]) {
    assert!(!sample.is_empty(), "sample must be nonempty");
    let dim = sample[0].dim();
    assert!(dim > 0, "sample dimension must be positive");
    for p in sample.iter().skip(1) {
        assert_eq!(
            p.dim(),
            dim,
            "all points in a sample must have the same dimension"
        );
    }
}

fn validate_two_samples(x: &[Point], y: &[Point]) {
    validate_sample(x);
    validate_sample(y);
    assert_eq!(
        x[0].dim(),
        y[0].dim(),
        "both samples must have the same point dimension"
    );
}

fn average_cross_distance(x: &[Point], y: &[Point]) -> f64 {
    validate_two_samples(x, y);

    let n1 = x.len();
    let n2 = y.len();
    let mut sum = 0.0_f64;

    for xi in x {
        for yj in y {
            sum += xi.distance(yj);
        }
    }

    sum / (n1 * n2) as f64
}

fn average_within_distance(sample: &[Point]) -> f64 {
    validate_sample(sample);

    let n = sample.len();
    let mut sum = 0.0_f64;

    for i in 0..n {
        for j in 0..n {
            sum += sample[i].distance(&sample[j]);
        }
    }

    sum / (n * n) as f64
}

fn energy_statistic_components(x: &[Point], y: &[Point]) -> EnergyComponents {
    validate_two_samples(x, y);

    let cross_average = average_cross_distance(x, y);
    let within_x_average = average_within_distance(x);
    let within_y_average = average_within_distance(y);

    let statistic = 2.0 * cross_average - within_x_average - within_y_average;

    EnergyComponents {
        cross_average,
        within_x_average,
        within_y_average,
        statistic,
    }
}

fn split_pooled_sample(pooled: &[Point], n1: usize) -> (Vec<Point>, Vec<Point>) {
    assert!(n1 > 0 && n1 < pooled.len(), "invalid split size");
    let x = pooled[..n1].to_vec();
    let y = pooled[n1..].to_vec();
    (x, y)
}

fn permutation_test_energy(
    x: &[Point],
    y: &[Point],
    permutations: usize,
    seed: u64,
) -> PermutationResult {
    validate_two_samples(x, y);
    assert!(permutations > 0, "number of permutations must be positive");

    let observed = energy_statistic_components(x, y).statistic;

    let n1 = x.len();
    let mut pooled = Vec::with_capacity(x.len() + y.len());
    pooled.extend_from_slice(x);
    pooled.extend_from_slice(y);

    let mut rng = LcgRng::new(seed);
    let mut exceedances = 0usize;
    let mut permuted_statistics = Vec::with_capacity(permutations);

    for _ in 0..permutations {
        rng.shuffle(&mut pooled);
        let (xp, yp) = split_pooled_sample(&pooled, n1);
        let stat = energy_statistic_components(&xp, &yp).statistic;
        if stat >= observed {
            exceedances += 1;
        }
        permuted_statistics.push(stat);
    }

    // Add-one smoothing for a stable Monte Carlo p-value.
    let p_value = (exceedances as f64 + 1.0) / (permutations as f64 + 1.0);

    PermutationResult {
        observed_statistic: observed,
        p_value,
        exceedances,
        permutations,
        permuted_statistics,
    }
}

fn mean(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "values must be nonempty");
    values.iter().sum::<f64>() / values.len() as f64
}

fn standard_deviation(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "values must be nonempty");
    let m = mean(values);
    let variance = values
        .iter()
        .map(|v| {
            let d = *v - m;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn quantile(values: &[f64], q: f64) -> f64 {
    assert!(!values.is_empty(), "values must be nonempty");
    assert!((0.0..=1.0).contains(&q), "quantile must lie in [0,1]");

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if sorted.len() == 1 {
        return sorted[0];
    }

    let pos = q * (sorted.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;

    if lo == hi {
        sorted[lo]
    } else {
        let t = pos - lo as f64;
        (1.0 - t) * sorted[lo] + t * sorted[hi]
    }
}

fn print_sample(title: &str, sample: &[Point]) {
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
    for (i, p) in sample.iter().enumerate() {
        println!("  [{:>2}] {}", i, p);
    }
    println!();
}

fn main() {
    // Example:
    // Two two-dimensional samples. The first sample is centered nearer the origin,
    // while the second is shifted and slightly reoriented. The energy statistic
    // should detect the resulting difference in joint distribution.

    let x_sample = vec![
        Point::new(vec![0.20, 0.30]),
        Point::new(vec![0.50, 0.80]),
        Point::new(vec![0.90, 0.40]),
        Point::new(vec![1.10, 1.00]),
        Point::new(vec![1.40, 0.70]),
        Point::new(vec![1.60, 1.30]),
        Point::new(vec![1.90, 1.00]),
        Point::new(vec![2.10, 1.50]),
    ];

    let y_sample = vec![
        Point::new(vec![1.20, -0.10]),
        Point::new(vec![1.50, 0.20]),
        Point::new(vec![1.90, 0.10]),
        Point::new(vec![2.20, 0.50]),
        Point::new(vec![2.50, 0.40]),
        Point::new(vec![2.80, 0.90]),
        Point::new(vec![3.10, 0.80]),
        Point::new(vec![3.40, 1.20]),
    ];

    print_sample("Sample X", &x_sample);
    print_sample("Sample Y", &y_sample);

    let components = energy_statistic_components(&x_sample, &y_sample);

    println!("Energy-Test Components");
    println!("======================");
    println!(
        "Average cross distance          = {:>.10}",
        components.cross_average
    );
    println!(
        "Average within-X distance       = {:>.10}",
        components.within_x_average
    );
    println!(
        "Average within-Y distance       = {:>.10}",
        components.within_y_average
    );
    println!(
        "E^2 = 2*A_XY - A_XX - A_YY      = {:>.10}",
        components.statistic
    );
    println!();

    let permutations = 2000usize;
    let perm = permutation_test_energy(&x_sample, &y_sample, permutations, 20260405);

    println!("Permutation Test Summary");
    println!("========================");
    println!(
        "Observed energy statistic       = {:>.10}",
        perm.observed_statistic
    );
    println!("Permutations                    = {}", perm.permutations);
    println!("Exceedances                     = {}", perm.exceedances);
    println!("Monte Carlo p-value             = {:>.10}", perm.p_value);
    println!();

    println!("Permutation Distribution Diagnostics");
    println!("====================================");
    println!(
        "Mean permuted statistic         = {:>.10}",
        mean(&perm.permuted_statistics)
    );
    println!(
        "Std. dev. permuted statistic    = {:>.10}",
        standard_deviation(&perm.permuted_statistics)
    );
    println!(
        "5% quantile                     = {:>.10}",
        quantile(&perm.permuted_statistics, 0.05)
    );
    println!(
        "50% quantile                    = {:>.10}",
        quantile(&perm.permuted_statistics, 0.50)
    );
    println!(
        "95% quantile                    = {:>.10}",
        quantile(&perm.permuted_statistics, 0.95)
    );
    println!();

    println!("Interpretation");
    println!("==============");
    println!("Large values of the energy statistic indicate that the average");
    println!("between-sample distance is substantially larger than the average");
    println!("within-sample distance, suggesting that the two multivariate");
    println!("samples do not arise from the same underlying distribution.");
}
```

Program 14.9.2 demonstrates a practical implementation of the energy-based framework for multivariate two-sample testing. This approach complements quadrant-based methods by focusing on pairwise distances, providing sensitivity to both location and distributional differences in a unified geometric formulation.

The example illustrates how the energy statistic captures separation between samples through the contrast between cross-sample and within-sample distances. The permutation test further enables reliable inference without relying on asymptotic approximations, making the method applicable in finite-sample and high-dimensional settings.

The modular structure of the implementation allows for straightforward extensions, including alternative distance metrics, weighted samples, or kernel-based generalizations. This flexibility reflects the broader theme discussed in Section 14.9.3: that different multivariate testing methods offer complementary perspectives, and their combination can provide a more comprehensive understanding of distributional differences.

## 14.9.4. Applications and Practical Considerations

In astrophysics, one may compare spatial distributions of stars to detect clustering or filamentary structures, where deviations between samples can reveal underlying physical processes such as gravitational interactions or galaxy formation patterns. In particle physics, joint distributions, for example Dalitz plots, are analyzed to identify subtle differences between decay processes, often requiring sensitive multivariate tests to detect small but meaningful deviations. In climate science, joint distributions of variables such as temperature and humidity can be compared across locations or time periods to detect changes in climatology, providing insight into evolving environmental conditions and long-term trends (Lanzante, 2021). In all these settings, the multivariate nature of the data is essential, and univariate comparisons would fail to capture the full structure of the phenomena.

### Statistical Inference

P-values are typically obtained via permutation, or label shuffling, which relies on exchangeability under the null hypothesis. This procedure constructs the null distribution of the test statistic by repeatedly reassigning sample labels and recomputing the statistic, thereby approximating its sampling distribution without requiring explicit analytical formulas. Alternatively, one may approximate significance using fitted extreme-value distributions, as explored in the energy-test literature. Such approximations aim to reduce computational cost while maintaining accuracy, particularly when a large number of permutations would otherwise be required.

### Computational Considerations

In practice, data are represented as arrays of points $(x, y)$, and the algorithm proceeds by iterating over candidate origin points, computing quadrant counts, tracking maximum differences, and estimating significance via resampling. Each of these steps must be implemented efficiently to ensure scalability, especially for large datasets. The computation of quadrant counts is typically the most expensive component, motivating the use of optimized data structures or vectorized operations. Resampling procedures, such as permutation testing, can further increase computational cost and are often parallelized in modern implementations.

The overall space complexity is $O(n)$, as only the sample points and a small number of auxiliary variables need to be stored. The time complexity ranges from $O(n^2)$ to $O(n \log n)$ depending on the implementation, with more advanced methods leveraging spatial indexing or efficient query structures to reduce computational overhead.

Finally, these ideas extend naturally to higher dimensions and even to structured data, where analogous region-based or kernel-based comparisons are employed. In higher-dimensional settings, the notion of quadrants generalizes to orthants or other partitioning schemes, while kernel-based approaches provide a flexible alternative that avoids explicit geometric partitioning altogether. These extensions broaden the applicability of multivariate two-sample testing to complex data types encountered in modern scientific and computational applications.

# 14.10. Savitzky-Golay Smoothing Filters

Savitzky–Golay (SG) filters are a class of finite-impulse-response (FIR) smoothing filters designed to reduce noise while preserving important features such as peak height, width, and local curvature. Unlike simple moving averages, which tend to distort sharp features by uniformly averaging neighboring values, SG filters perform local polynomial regression, making them particularly effective in signal processing, spectroscopy, and time-series analysis. Their ability to retain the geometric structure of signals makes them especially valuable in applications where accurate shape preservation is critical.

The fundamental idea is to approximate a noisy signal $y[i]$ by fitting a low-degree polynomial over a local window centered at each point. For each index $i$, a set of neighboring samples within a fixed window is selected, and a polynomial of chosen degree is fitted in a least-squares sense to these data points. The smoothed value is then taken as the value of this polynomial at the center of the window. Because the same fitting procedure is applied at every point, the method can be interpreted as a convolution with a fixed set of coefficients determined by the window size and polynomial degree.

This approach preserves low-order moments of the signal while attenuating high-frequency noise. In particular, because the fitted polynomial reproduces polynomials up to a given degree exactly, trends and local curvature within the window are maintained, while rapid fluctuations, typically associated with noise, are smoothed out. This property distinguishes SG filters from simpler averaging methods, which do not preserve such local structure.

An additional advantage of SG filters is that the convolution coefficients can be precomputed, allowing efficient implementation in practice. Once these coefficients are determined, the filtering operation reduces to a simple linear combination of neighboring samples, making the method suitable for real-time or large-scale applications. Furthermore, by adjusting the window size and polynomial degree, one can control the trade-off between smoothing strength and feature preservation.

## 14.10.1. Local Polynomial Approximation

Consider a window of $(2M+1)$ equally spaced points centered at index $i$, with offsets:

$$x_{-M}, x_{-M+1}, \dots, x_M \tag{14.10.1}$$

These offsets are typically chosen symmetrically around zero so that the center of the window corresponds to $x=0$, which simplifies both the analysis and the resulting formulas. This symmetric construction ensures that the fitted polynomial is naturally centered at the point where the smoothed value is to be evaluated.

We fit a polynomial of degree $d$,

$$p(x) = \beta_0 + \beta_1 x + \cdots + \beta_d x^d \tag{14.10.2}$$

to the data in this window using least squares. The choice of degree $d$ controls the flexibility of the approximation: low-degree polynomials provide stronger smoothing, while higher-degree polynomials allow more detailed local structure to be captured. The fitting is performed locally within each window, making the method adaptive to variations in the signal.

Define the Vandermonde design matrix:

$$
X =
\begin{pmatrix}
1 & x_{-M} & x_{-M}^2 & \cdots & x_{-M}^d \\
1 & x_{-M+1} & x_{-M+1}^2 & \cdots & x_{-M+1}^d \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_M & x_M^2 & \cdots & x_M^d
\end{pmatrix}
\tag{14.10.3}
$$

This matrix encodes the polynomial basis evaluated at each point in the window. Each row corresponds to a sample location, and each column corresponds to a power of $x$. The Vandermonde structure reflects the use of monomials as basis functions for the polynomial fit.

Let $y$ denote the vector of observed values in the window. The least-squares solution satisfies:

$$
X^\top X \, \beta = X^\top y
\tag{14.10.4}
$$

These normal equations determine the coefficients $\beta = (\beta_0, \beta_1, \dots, \beta_d)^\top$ that minimize the squared error between the polynomial and the observed data. The matrix $X^\top X$ captures the geometry of the sampling points, while $X^\top y$ incorporates the observed signal values.

The smoothed value at the center corresponds to:

$$y_{\mathrm{smoothed}}[i] = p(0) = \beta_0\tag{14.10.5}$$

Because the polynomial is evaluated at $x=0$, only the constant term $\beta_0$ contributes to the smoothed value. This is a direct consequence of centering the coordinate system at the midpoint of the window.

Using properties of least squares, this can be written as:

$$\beta_0 = e_0^\top (X^\top X)^{-1} X^\top y \tag{14.10.6}$$

where $e_0 = (1, 0, \dots, 0)^\top$. This expression shows that the smoothed value is a linear combination of the observed data, with weights determined by the first row of the matrix $(X^\top X)^{-1} X^\top$. These weights depend only on the window geometry and polynomial degree, not on the data itself, which is why they can be precomputed and reused across the signal.

## 14.10.2. Convolution Interpretation

A key feature of Savitzky–Golay filtering is that it is linear and time-invariant, and therefore equivalent to a convolution. Linearity follows from the least-squares formulation, since the fitted coefficients depend linearly on the data, while time invariance arises because the same window geometry and polynomial fit are applied at every index. As a result, the filtering operation can be expressed as a fixed linear transformation applied uniformly across the signal.

Define weights:

$$w^\top = e_0^\top (X^\top X)^{-1} X^\top \tag{14.10.7}$$

These weights correspond to the coefficients that map the observed data within a window directly to the smoothed value at its center. Importantly, they depend only on the window size $M$ and the polynomial degree $d$, and not on the signal itself. Once computed, these weights can be reused for all positions in the signal.

Then the smoothed signal is given by:

$$
y_{\mathrm{smoothed}}[i]

\sum_{k=-M}^{M} w_k \, y[i + k] \tag{14.10.8}
$$

This expression shows that the smoothing operation is a weighted sum of neighboring samples, which is precisely the definition of a discrete convolution. The coefficients $w_k$ form a symmetric kernel when the sampling points are symmetric, ensuring that no phase shift is introduced into the signal.

Thus, the SG filter can be implemented as a fixed convolution kernel of length $(2M+1)$, where the weights depend only on $M$ and the polynomial degree $d$. This representation is computationally efficient, as it reduces the filtering process to repeated applications of the same set of weights across the signal.

This formulation highlights an important property: the filter exactly reproduces polynomials up to degree $d$, meaning that smooth trends are preserved while noise is suppressed. In particular, if the underlying signal within a window is a polynomial of degree at most $d$, the filtering operation leaves it unchanged. This reproduction property explains why SG filters maintain features such as peak shape and curvature, distinguishing them from simpler smoothing methods that may distort these characteristics.

### Rust Implementation

Following the discussion in Section 14.10 on the construction and interpretation of Savitzky–Golay smoothing filters, Program 14.10.1 provides a practical implementation of local polynomial approximation and its equivalent convolution form. In numerical signal processing, smoothing must balance noise reduction with preservation of underlying structure, particularly low-degree polynomial trends. This program implements the least-squares formulation described in Equations (14.10.3)–(14.10.7), where a polynomial is fitted locally within a moving window and evaluated at the center point. The resulting coefficients are then reused as fixed convolution weights, as expressed in Equation (14.10.8). By combining polynomial fitting with efficient filtering, the program demonstrates how local approximation theory translates into a stable and reusable computational procedure for smoothing discrete data.

At the core of the implementation is the construction of the Vandermonde design matrix, as introduced in Equation (14.10.3). The function `symmetric_offsets` generates the centered grid of offsets $-M, \ldots, M$, which defines the local window around each evaluation point. These offsets are passed to `vandermonde_matrix`, which builds the matrix $X$ by evaluating powers of each offset up to the specified polynomial degree. This matrix encodes the local polynomial basis used in the least-squares approximation.

The least-squares system in Equation (14.10.4) is formed through the normal equations $X^\top X$. The function `transpose` computes $X^\top$, while `matmul` constructs the matrix product $X^\top X$. Rather than explicitly computing the inverse, the program solves the linear system $(X^\top X) z = e_0$, where $e_0$ corresponds to selecting the constant term $\beta_0$. This is carried out by the function `solve_linear_system`, which implements Gaussian elimination with partial pivoting to ensure numerical stability.

Once the coefficients are obtained, the function `savitzky_golay_weights` computes the convolution weights according to Equation (14.10.7). These weights represent the contribution of each point in the window to the smoothed value at the center. The resulting weight vector is symmetric and sums to one, reflecting the reproduction of constant functions and the preservation of polynomial structure.

The convolution interpretation in Equation (14.10.8) is implemented in the function `apply_sg_filter`. This function applies the precomputed weights across the signal, using reflected boundary conditions to handle edge effects. By reusing the same weights at each position, the algorithm avoids repeated least-squares solves, significantly improving computational efficiency. The program includes a verification step through the function `verify_polynomial_reproduction`, which confirms that the filter exactly reproduces polynomials of degree up to $d$ in the interior of the domain. This is a defining property of Savitzky–Golay filters and provides a strong validation of the implementation.

The `main` function demonstrates the full pipeline. It first computes the SG weights, verifies their normalization and symmetry, and then applies the filter to a synthetic noisy signal. The comparison between the clean, noisy, and smoothed signals illustrates the effect of the filter, while RMS error calculations quantify the improvement. The polynomial reproduction test further confirms consistency with the theoretical formulation.

```rust
// Program 14.10.1
// Savitzky-Golay Smoothing Filters
//
// Problem statement:
// Construct the Savitzky-Golay smoothing kernel for a window of length (2M+1)
// and polynomial degree d by forming the Vandermonde design matrix X,
// computing the weights
//
//     w^T = e_0^T (X^T X)^{-1} X^T,
//
// and then applying the resulting fixed convolution kernel to a noisy signal.
// The program also verifies the polynomial reproduction property by checking
// that the filter reproduces a polynomial of degree at most d exactly
// at interior points.

/// Build symmetric offsets x_{-M}, ..., x_M.
fn symmetric_offsets(m: usize) -> Vec<f64> {
    let m_i = m as isize;
    (-m_i..=m_i).map(|k| k as f64).collect()
}

/// Build the Vandermonde design matrix X from offsets and polynomial degree d.
/// X[row, col] = x_row^col.
fn vandermonde_matrix(offsets: &[f64], degree: usize) -> Vec<Vec<f64>> {
    let rows = offsets.len();
    let cols = degree + 1;
    let mut x = vec![vec![0.0_f64; cols]; rows];

    for (i, &xi) in offsets.iter().enumerate() {
        let mut power = 1.0_f64;
        for j in 0..cols {
            x[i][j] = power;
            power *= xi;
        }
    }

    x
}

/// Matrix transpose.
fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = a[0].len();
    let mut at = vec![vec![0.0_f64; rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            at[j][i] = a[i][j];
        }
    }

    at
}

/// Matrix-matrix multiplication.
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let a_rows = a.len();
    let a_cols = a[0].len();
    let b_rows = b.len();
    let b_cols = b[0].len();

    assert_eq!(a_cols, b_rows, "Incompatible matrix dimensions for multiplication.");

    let mut c = vec![vec![0.0_f64; b_cols]; a_rows];
    for i in 0..a_rows {
        for k in 0..a_cols {
            let aik = a[i][k];
            for j in 0..b_cols {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

/// Matrix-vector multiplication.
fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let rows = a.len();
    let cols = a[0].len();
    assert_eq!(cols, x.len(), "Incompatible matrix/vector dimensions.");

    let mut y = vec![0.0_f64; rows];
    for i in 0..rows {
        for j in 0..cols {
            y[i] += a[i][j] * x[j];
        }
    }
    y
}

/// Solve A x = b using Gaussian elimination with partial pivoting.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = a.len();
    assert!(n > 0, "Matrix must be nonempty.");
    assert_eq!(a[0].len(), n, "Matrix must be square.");
    assert_eq!(b.len(), n, "Right-hand side size mismatch.");

    for k in 0..n {
        // Pivot selection.
        let mut pivot_row = k;
        let mut pivot_value = a[k][k].abs();
        for i in (k + 1)..n {
            let cand = a[i][k].abs();
            if cand > pivot_value {
                pivot_value = cand;
                pivot_row = i;
            }
        }

        assert!(
            pivot_value > 1.0e-14,
            "Singular or nearly singular system in Gaussian elimination."
        );

        if pivot_row != k {
            a.swap(k, pivot_row);
            b.swap(k, pivot_row);
        }

        // Eliminate below pivot.
        for i in (k + 1)..n {
            let factor = a[i][k] / a[k][k];
            a[i][k] = 0.0;
            for j in (k + 1)..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    x
}

/// Compute Savitzky-Golay weights:
/// w^T = e_0^T (X^T X)^{-1} X^T.
///
/// Instead of explicitly inverting X^T X, we solve
/// (X^T X) z = e_0
/// and then form w = X z.
fn savitzky_golay_weights(m: usize, degree: usize) -> Vec<f64> {
    assert!(2 * m + 1 >= degree + 1, "Window must contain at least degree+1 points.");

    let offsets = symmetric_offsets(m);
    let x = vandermonde_matrix(&offsets, degree);
    let xt = transpose(&x);
    let xtx = matmul(&xt, &x);

    let mut e0 = vec![0.0_f64; degree + 1];
    e0[0] = 1.0;

    let z = solve_linear_system(xtx, e0);
    matvec(&x, &z)
}

/// Reflect an index into the valid range [0, n-1].
fn reflect_index(idx: isize, n: usize) -> usize {
    assert!(n > 0, "Signal length must be positive.");
    let n_i = n as isize;
    if n == 1 {
        return 0;
    }

    let reflected = if idx < 0 {
        -idx
    } else if idx >= n_i {
        2 * (n_i - 1) - idx
    } else {
        idx
    };

    reflected as usize
}

/// Apply the SG filter by convolution with reflected boundary handling.
fn apply_sg_filter(signal: &[f64], weights: &[f64]) -> Vec<f64> {
    assert!(!signal.is_empty(), "Signal must be nonempty.");
    assert!(!weights.is_empty(), "Weights must be nonempty.");
    assert!(weights.len() % 2 == 1, "Weight vector must have odd length.");

    let n = signal.len();
    let m = (weights.len() - 1) / 2;
    let mut out = vec![0.0_f64; n];

    for i in 0..n {
        let mut acc = 0.0_f64;
        for (j, &w) in weights.iter().enumerate() {
            let offset = j as isize - m as isize;
            let idx = reflect_index(i as isize + offset, n);
            acc += w * signal[idx];
        }
        out[i] = acc;
    }

    out
}

/// Evaluate a polynomial c0 + c1 x + ... + cd x^d.
fn evaluate_polynomial(coeffs: &[f64], x: f64) -> f64 {
    let mut value = 0.0_f64;
    let mut power = 1.0_f64;
    for &c in coeffs {
        value += c * power;
        power *= x;
    }
    value
}

/// Generate a deterministic "noisy" signal:
/// smooth trend + oscillation + small structured perturbation.
fn generate_noisy_signal(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut clean = vec![0.0_f64; n];
    let mut noisy = vec![0.0_f64; n];

    for i in 0..n {
        let x = -3.0 + 6.0 * i as f64 / (n as f64 - 1.0);
        let trend = 0.4 + 0.3 * x - 0.15 * x * x + 0.05 * x * x * x;
        let oscillation = 0.8 * (1.8 * x).sin();
        let structured_noise = 0.12 * (7.0 * x).cos() + 0.06 * (13.0 * x).sin();

        clean[i] = trend + oscillation;
        noisy[i] = clean[i] + structured_noise;
    }

    (clean, noisy)
}

/// Verify that the SG filter reproduces a polynomial of degree <= d
/// at interior points away from the boundary by at least M samples.
fn verify_polynomial_reproduction(
    weights: &[f64],
    m: usize,
    degree: usize,
    n: usize,
) -> f64 {
    let coeffs: Vec<f64> = (0..=degree)
        .map(|k| 0.25 * (k as f64 + 1.0) * if k % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let mut signal = vec![0.0_f64; n];
    let mut xs = vec![0.0_f64; n];
    for i in 0..n {
        xs[i] = -5.0 + 10.0 * i as f64 / (n as f64 - 1.0);
        signal[i] = evaluate_polynomial(&coeffs, xs[i]);
    }

    let filtered = apply_sg_filter(&signal, weights);

    let mut max_error = 0.0_f64;
    for i in m..(n - m) {
        let err = (filtered[i] - signal[i]).abs();
        if err > max_error {
            max_error = err;
        }
    }
    max_error
}

fn print_vector(title: &str, values: &[f64]) {
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
    for (i, v) in values.iter().enumerate() {
        println!("  [{:>2}] {:>.10}", i, v);
    }
    println!();
}

fn main() {
    // Filter parameters.
    let m = 3usize;        // half-window size, total window length = 2M+1 = 7
    let degree = 3usize;   // cubic local polynomial

    let weights = savitzky_golay_weights(m, degree);

    println!("Savitzky-Golay Filter Parameters");
    println!("================================");
    println!("Half-window size M             = {}", m);
    println!("Window length 2M+1             = {}", 2 * m + 1);
    println!("Polynomial degree d            = {}", degree);
    println!();

    print_vector("Precomputed SG Weights", &weights);

    let weight_sum: f64 = weights.iter().sum();
    println!("Weight Checks");
    println!("=============");
    println!("sum_k w_k                      = {:>.10}", weight_sum);
    println!("symmetry error                 = {:>.10e}", {
        let mut err = 0.0_f64;
        for k in 0..weights.len() {
            err = err.max((weights[k] - weights[weights.len() - 1 - k]).abs());
        }
        err
    });
    println!();

    let n = 31usize;
    let (clean, noisy) = generate_noisy_signal(n);
    let smoothed = apply_sg_filter(&noisy, &weights);

    println!("Representative Signal Values");
    println!("============================");
    println!(
        "{:>5} {:>16} {:>16} {:>16}",
        "i", "clean", "noisy", "smoothed"
    );
    for i in 0..n {
        if i < 8 || i >= n - 8 || i == n / 2 {
            println!(
                "{:>5} {:>16.10} {:>16.10} {:>16.10}",
                i, clean[i], noisy[i], smoothed[i]
            );
        }
    }
    println!();

    let rms_noisy = {
        let mut s = 0.0_f64;
        for i in 0..n {
            let e = noisy[i] - clean[i];
            s += e * e;
        }
        (s / n as f64).sqrt()
    };

    let rms_smoothed = {
        let mut s = 0.0_f64;
        for i in 0..n {
            let e = smoothed[i] - clean[i];
            s += e * e;
        }
        (s / n as f64).sqrt()
    };

    println!("RMS Error Relative to Clean Signal");
    println!("==================================");
    println!("RMS(noisy - clean)              = {:>.10}", rms_noisy);
    println!("RMS(smoothed - clean)           = {:>.10}", rms_smoothed);
    println!();

    let reproduction_error = verify_polynomial_reproduction(&weights, m, degree, 41);

    println!("Polynomial Reproduction Check");
    println!("=============================");
    println!("Degree reproduced exactly      = {}", degree);
    println!("max interior reproduction error= {:>.10e}", reproduction_error);
    println!();

    println!("Interpretation");
    println!("==============");
    println!("The precomputed coefficients define a fixed symmetric FIR kernel.");
    println!("Applying that kernel smooths the noisy signal while preserving");
    println!("low-degree polynomial structure, which is the defining property");
    println!("of the Savitzky-Golay filter.");
}
```

Program 14.10.1 demonstrates how local polynomial approximation can be transformed into an efficient convolution-based smoothing algorithm. This approach reflects the central idea of Section 14.10: that fitting a polynomial locally yields a reusable filter kernel that preserves important structural features of the data.

The example highlights several key properties of Savitzky–Golay filters. The symmetry and normalization of the weights ensure stability and preservation of constant signals, while the exact reproduction of low-degree polynomials distinguishes SG filtering from simple averaging methods. At the same time, the modest reduction in RMS error illustrates the trade-off between smoothing and feature preservation, particularly when the noise contains structured components.

The modular design of the implementation allows for straightforward extensions, including higher-degree polynomials, wider windows, derivative estimation, and adaptive filtering strategies. These extensions are widely used in applications such as signal processing, spectroscopy, and time-series analysis, where maintaining the integrity of underlying trends is essential.

## 14.10.3. Computational Aspects and Frequency Behavior

The computational cost of SG filtering consists of two parts. The first is the precomputation of weights,

$$O\big((d+1)^3\big) \tag{14.10.9}$$

which arises from solving the normal equations associated with the local polynomial fit. This cost is incurred only once for a given choice of window size and polynomial degree, and is negligible in practice because (d) is typically small. The second component is the filtering operation itself,

$$O(NM) \tag{14.10.10}$$

for an $N$-point signal, where each output value is computed as a weighted sum over a window of size $(2M+1)$. Since $M$ and $d$ are typically small constants, for example $M=5$ and $d=3$, the overall cost is effectively linear in $N$, making SG filtering efficient for large datasets.

From a signal-processing perspective, SG filters behave as low-pass filters with a relatively flat passband, preserving low-frequency content while attenuating high-frequency noise. This flatness of the passband is a direct consequence of the polynomial reproduction property, which ensures that smooth trends are transmitted with minimal distortion. In addition, SG filters provide improved preservation of peaks compared to moving averages, as the local polynomial fit maintains both amplitude and curvature within the window.

However, standard SG filters may exhibit suboptimal stop-band behavior. In particular, their attenuation of high-frequency components is not as strong as that of some classical FIR designs, which can result in residual noise in certain applications. Modern work proposes modified FIR kernels, for example windowed sinc filters with corrections, that retain the advantages of SG filtering while improving noise suppression (Schmid et al., 2022). These approaches aim to balance the trade-off between smoothness and feature preservation more effectively.

Extensions include weighted SG filters, which assign different importance to points within the window to account for heteroscedastic noise or varying data quality; adaptive window selection, where the window size is adjusted based on local signal characteristics; and multidimensional SG filtering, for example fitting bivariate polynomials over image patches. These extensions broaden the applicability of SG filtering to more complex and structured data, while preserving the core idea of local polynomial approximation.

## 14.10.4. Applications and Practical Implementation

Savitzky–Golay filters are widely used in applications where feature preservation is critical. Their ability to smooth noise while maintaining the geometric structure of the signal makes them particularly valuable in domains where subtle features carry important information.

### Signal and Sensor Data

In environmental monitoring, SG filters are used to smooth high-frequency sensor data, for example water quality or atmospheric measurements, while preserving trends and events (Badrudeen et al., 2026). In such settings, abrupt changes or transient phenomena may correspond to meaningful physical events, and preserving these features during smoothing is essential for accurate interpretation.

### Biomedical Signals

In electrocardiogram (ECG) processing, SG filters reduce noise while maintaining sharp features such as QRS complexes, which are essential for diagnosis (Ádám et al., 2025). The preservation of these features ensures that clinically relevant signal characteristics are not distorted, which is critical for reliable automated or manual analysis.

### Spectroscopy

In analytical chemistry, SG filtering is a standard technique for smoothing spectral data, for example Raman or infrared spectra, without distorting peak positions or widths. Since the location and shape of spectral peaks encode chemical information, maintaining these features while suppressing measurement noise is of central importance.

### Time-series Analysis

In economics or finance, SG filters can be used to reveal underlying trends while suppressing short-term fluctuations. This allows analysts to distinguish between long-term structural behavior and transient noise, improving interpretability of time-dependent data.

### Implementation Considerations

In practice, SG filtering proceeds as follows: one chooses a window size $(2M+1)$ and polynomial degree $d$, computes the convolution weights once, and then applies a sliding window convolution over the data. The precomputation step depends only on the chosen parameters, while the filtering step consists of repeated application of the same kernel, making the method efficient and straightforward to implement.

Boundary handling may involve shrinking the window near edges, padding the signal, or using asymmetric kernels. These strategies address the fact that a full symmetric window is not available near the boundaries of the signal. The choice of method can affect both accuracy and smoothness near the edges and must be selected based on the application requirements.

The method is well suited to efficient implementation in Rust: weights can be precomputed using linear algebra functions, and filtering is performed via a simple loop or convolution. For higher-dimensional data, separable filtering or local polynomial fitting can be applied, extending the same principles to images or multidimensional datasets while maintaining computational efficiency.

# 14.11. Conclusion

This chapter has developed the computational foundations of statistical description, hypothesis testing, association analysis, information-theoretic dependence measures, multivariate distributional comparison, and nonparametric smoothing, progressing from univariate moment estimation through two-sample testing, contingency table analysis, linear and rank-based correlation, entropy and mutual information, two-dimensional distribution tests, and local polynomial regression filters. The central theme throughout is that meaningful statistical inference requires not only the correct mathematical formulas but also careful attention to numerical stability, robustness against distributional violations, appropriate choice of test assumptions, and honest reporting of effect sizes alongside significance levels. Each section has paired classical parametric methods with their nonparametric and resampling-based alternatives, reflecting the practical reality that real-world data rarely conforms perfectly to textbook distributional assumptions. The Rust implementations throughout the chapter demonstrate how these statistical principles translate into efficient, type-safe numerical code that handles edge cases, validates inputs, and produces reproducible results under finite-precision arithmetic.

## 14.11.1. Key Takeaways

- The sample mean $\bar{x} = (1/N) \sum_{i=1}^{N} x_i$ and the sample variance $s^2 = (1/(N-1)) \sum (x_i - \bar{x})^2$ are the fundamental moment estimators for location and spread, but their numerical computation requires care: the naive single-pass variance formula can suffer from catastrophic cancellation when the mean is large relative to the spread, so the two-pass algorithm with compensated summation is preferred for general use. Higher-order standardized moments, specifically skewness $\gamma_1 = \mu_3 / \mu_2^{3/2}$ and excess kurtosis $\gamma_2 = \mu_4 / \mu_2^2 - 3$, characterize distributional asymmetry and tail weight but exhibit substantial variability in finite samples, particularly in the presence of outliers. Cumulants, obtained from the cumulant generating function $K_X(t) = \log M_X(t)$, provide an alternative characterization with the fundamental additivity property $\kappa_k(X + Y) = \kappa_k(X) + \kappa_k(Y)$ for independent variables. When heavy tails or outliers are present, robust alternatives such as the median, interquartile range, median-of-means estimators, and trimmed or winsorized moments provide more stable descriptions of location and spread that are less sensitive to extreme observations.
- For streaming or distributed data where storing the full dataset is infeasible, one-pass algorithms maintain running estimates through the recurrences $\bar{X}_{n+w} = \bar{X}_n + (w/(n+w))(\bar{X}_w - \bar{X}_n)$ for block mean updates and $M_{n+1} = M_n + (x_{n+1} - \bar{x}_n)(x_{n+1} - \bar{x}_{n+1})$ for incremental variance accumulation, from which $s^2 = M_n / (n-1)$. Block merging extends these formulas to distributed settings through $M_{n+w} = M_n + M_w + (nw/(n+w))(\bar{X}_w - \bar{X}_n)^2$, allowing partial summaries computed across multiple processors to be combined without revisiting raw data. Compensated summation techniques such as Kahan summation reduce floating-point rounding errors during accumulation, while the comparison of stable and deliberately unstable variance formulas illustrates why numerically careful algorithm design is essential in practical implementations.
- Comparing the means of two populations begins with the two-sample $t$-test, which takes pooled form $t = (\bar{x}_A - \bar{x}_B) / (s_p \sqrt{1/n_A + 1/n_B})$ when equal variances can be assumed and Welch form $t = (\bar{x}_A - \bar{x}_B) / \sqrt{s_A^2/n_A + s_B^2/n_B}$ with Welch-Satterthwaite approximate degrees of freedom when they cannot. For paired designs where observations are naturally linked, the problem reduces to a one-sample test on differences $d_i = x_i^{(A)} - x_i^{(B)}$ with statistic $t = \bar{d} / (s_d / \sqrt{n})$, which exploits the pairing structure to reduce variability. The $F$-test $F = s_A^2 / s_B^2$ compares variances under normality, but the Brown-Forsythe test provides a robust alternative by comparing absolute deviations from sample medians. Permutation tests generate an empirical null distribution by shuffling group labels and recomputing the observed statistic $T_{\mathrm{obs}} = \bar{x}_A - \bar{x}_B$, counting the fraction of permutations where $|T^{(b)}| \geq |T_{\mathrm{obs}}|$ to estimate the $p$-value. Effect size measures such as Cohen's $d$ quantify the practical magnitude of group differences on a standardized scale.
- Testing whether two distributions differ in their overall shape, not merely their means, requires distribution-level comparison statistics. The two-sample Kolmogorov-Smirnov statistic $D = \sup_x |F_n(x) - G_m(x)|$ measures the maximum vertical distance between the empirical cumulative distribution functions constructed from the two samples, where $F_n(x) = (1/n) \sum \mathbf{1}_{\{X_i \leq x\}}$ and $G_m(x) = (1/m) \sum \mathbf{1}_{\{Y_i \leq x\}}$. The Anderson-Darling statistic $A^2 = \int [F_n(x) - G_m(x)]^2 / [F(x)(1 - F(x))] \, dx$ provides greater sensitivity to tail discrepancies through its variance-weighted formulation, while Kuiper's statistic $V = D^+ + D^-$ combines the maximum positive and negative deviations to achieve invariance under cyclic shifts for circular data. For discrete or categorical data, the two-sample chi-squared statistic $\chi^2 = \sum (R_i - S_i)^2 / (R_i + S_i)$ compares bin counts across the two samples. Permutation tests with global envelope diagnostics extend these methods by constructing ECDF difference curves together with pointwise confidence bands from the permutation distribution, enabling identification of where the two distributions differ most.
- Contingency table analysis tests the association between two categorical variables by comparing observed cell frequencies $N_{ij}$ against expected frequencies $E_{ij} = N_{i\cdot} N_{\cdot j} / N$ under the null hypothesis of independence $P(X = i, Y = j) = P(X = i) P(Y = j)$, where $N_{i\cdot}$ and $N_{\cdot j}$ are row and column marginal totals. The Pearson chi-squared statistic $\chi^2 = \sum_{i,j} (N_{ij} - E_{ij})^2 / E_{ij}$ with $(I-1)(J-1)$ degrees of freedom provides the standard asymptotic test for independence. Because chi-squared scales with sample size, normalized measures are needed to quantify association strength: Cramer's $V = \sqrt{\chi^2 / (N \cdot \min(I-1, J-1))}$ and the contingency coefficient $C = \sqrt{\chi^2 / (\chi^2 + N)}$ rescale the statistic to bounded ranges. Asymmetric predictive measures such as Goodman-Kruskal's $\lambda$ quantify the proportional reduction in prediction error when one variable is used to predict another, while directional uncertainty coefficients based on conditional entropy provide a more information-sensitive alternative that uses the full conditional distributions rather than only modal counts.
- The Pearson correlation coefficient $r = \sum (x_i - \bar{x})(y_i - \bar{y}) / [\sqrt{\sum (x_i - \bar{x})^2} \sqrt{\sum (y_i - \bar{y})^2}]$ measures the strength of linear association between two continuous variables on the scale $[-1, 1]$, with significance tested by transforming to $t = r\sqrt{(N-2)/(1-r^2)}$ with $N - 2$ degrees of freedom. Fisher's $z$-transform $z = (1/2) \ln((1+r)/(1-r))$ with standard error $1/\sqrt{N-3}$ stabilizes the variance of the correlation estimate and enables construction of confidence intervals through normal approximation. Distance correlation, computed from doubly centered pairwise distance matrices, provides a measure that equals zero if and only if the variables are statistically independent, making it capable of detecting nonlinear and nonmonotone relationships that Pearson correlation would miss. Bootstrap resampling and permutation tests provide distribution-free inference for any correlation measure by repeatedly resampling or shuffling the data and computing empirical confidence intervals or $p$-values.
- Spearman's rank correlation $\rho_S$ replaces observed values with their ranks via midrank assignment for ties and computes the Pearson correlation on the rank-transformed variables, yielding a coefficient that measures monotonic association and is invariant under monotone transformations. The shortcut formula $\rho_S = 1 - 6 \sum d_i^2 / [N(N^2 - 1)]$ applies when no ties are present. Kendall's $\tau_b = (C - D) / \sqrt{(C + D + T_x)(C + D + T_y)}$ counts concordant pairs $C$ and discordant pairs $D$ with tie corrections $T_x$ and $T_y$, providing a measure based purely on pairwise ordering. The naive $O(N^2)$ computation of Kendall's $\tau$ can be reduced to $O(N \log N)$ through merge-sort-based inversion counting. A more general rank-based dependence measure $\xi_n = 1 - n \sum |r_{i+1} - r_i| / [2 \sum l_i(n - l_i)]$ captures functional dependence beyond monotonicity, satisfying $\xi(X; Y) = 0$ under independence and $\xi(X; Y) = 1$ when $Y$ is a deterministic function of $X$.
- Information-theoretic measures provide a principled framework for quantifying uncertainty and dependence in discrete distributions. Shannon entropy $H(p) = -\sum p_i \log p_i$ with the convention $0 \log 0 := 0$ measures the uncertainty of a distribution, while the Kullback-Leibler divergence $D_{\mathrm{KL}}(p \| q) = \sum p_i \log(p_i / q_i)$ quantifies the discrepancy between two distributions and satisfies the decomposition $H(p, q) = H(p) + D_{\mathrm{KL}}(p \| q)$ with cross-entropy. For joint distributions $p_{ij}$, the mutual information $I(X; Y) = \sum_{i,j} p_{ij} \log [p_{ij} / (p_{i\cdot} p_{\cdot j})] = D_{\mathrm{KL}}(p_{XY} \| p_X p_Y)$ captures general dependence including nonlinear relationships, satisfying $I(X; Y) \geq 0$ with equality if and only if $X$ and $Y$ are independent. The chain rule $H(X, Y) = H(X) + H(Y | X) = H(Y) + H(X | Y)$ relates joint, marginal, and conditional entropies, providing structural identities that serve as both theoretical tools and computational verification checks. In the large-alphabet small-sample regime, naive plug-in entropy estimators can be strongly biased, motivating bias-corrected and model-based estimation approaches.
- Testing whether two-dimensional point distributions differ requires test statistics that respect the geometry of the plane, since there is no natural ordering of points in $\mathbb{R}^2$. The Fasano-Franceschini test generalizes the KS approach by partitioning the plane into four quadrants around each candidate pivot point, computing the quadrant-based discrepancy $D(\mathbf{p} | X, Y) = \max_{j=1,\ldots,4} |(1/n_1) \sum I_j(\mathbf{X}_i | \mathbf{p}) - (1/n_2) \sum I_j(\mathbf{Y}_k | \mathbf{p})|$, and scanning over all sample points to obtain $D_0 = (D_1 + D_2) / 2$ with scaling factor $\mathcal{D} = 2\sqrt{n_1 n_2 / (n_1 + n_2)} \cdot D_0$. The energy statistic $E^2 = (2/(n_1 n_2)) \sum |\mathbf{X}_i - \mathbf{Y}_j| - (1/n_1^2) \sum |\mathbf{X}_i - \mathbf{X}_k| - (1/n_2^2) \sum |\mathbf{Y}_j - \mathbf{Y}_\ell|$ provides an alternative that equals zero if and only if the distributions match, using pairwise distances to capture both location and spread differences. Maximum mean discrepancy in reproducing kernel Hilbert spaces and graph-based tests using nearest-neighbor connectivity offer further alternatives, with permutation calibration providing $p$-values for all methods.
- The Savitzky-Golay filter performs local polynomial regression by fitting a polynomial of degree $d$ to $2M + 1$ equally spaced data points centered at each sample location through least-squares minimization, where the smoothed value is $\beta_0 = e_0^\top (X^\top X)^{-1} X^\top y$ with $X$ being the Vandermonde design matrix. The key computational insight is that the smoothing operation reduces to convolution with fixed weights $w^\top = e_0^\top (X^\top X)^{-1} X^\top$ that depend only on the half-width $M$ and polynomial degree $d$, yielding the smoothed signal $y_{\mathrm{smoothed}}[i] = \sum_{k=-M}^{M} w_k y[i+k]$. The precomputation of weights costs $O((d+1)^3)$ while the filtering operation requires $O(NM)$ for an $N$-point signal, and the filter exactly reproduces polynomials up to degree $d$, preserving trends and local curvature while attenuating high-frequency noise. Extensions include weighted variants for heteroscedastic noise, adaptive window selection based on local signal characteristics, and multidimensional filtering for image patches.

## 14.11.2. Advice for Beginners

- Statistics is not merely a collection of formulas. It is a framework for understanding uncertainty, extracting patterns from data, and making informed decisions when perfect information is unavailable. As you study this chapter, focus first on the underlying questions each method is designed to answer before concentrating on the mathematical details.
- Begin with descriptive statistics in Section 14.2. Learn the practical meaning of the mean, variance, skewness, and kurtosis, and understand how they describe the center, spread, asymmetry, and tail behavior of a distribution. At the same time, recognize their limitations. Real-world datasets often contain outliers, measurement errors, or heavy-tailed behavior, making robust measures such as the median and interquartile range equally important.
- When studying hypothesis testing in Sections 14.3 through 14.5, avoid thinking of p-values as the sole measure of importance. A statistically significant result may have little practical significance, while a practically important effect may not reach statistical significance in small samples. Always examine effect sizes, confidence intervals, and the assumptions underlying each test. Understanding when to use a Student t-test, Welch t-test, permutation test, chi-square test, or Kolmogorov-Smirnov test is often more important than memorizing their formulas.
- Correlation analysis in Sections 14.6 and 14.7 provides an opportunity to learn an important lesson: correlation does not imply causation. Pearson correlation measures linear relationships, while Spearman and Kendall correlations capture monotonic relationships. More advanced measures such as distance correlation and mutual information can reveal dependencies that traditional correlation coefficients may miss. When analyzing data, always visualize relationships before relying on a single numerical summary.
- The information-theoretic concepts introduced in Section 14.8 form the foundation of modern machine learning, communication theory, and artificial intelligence. Entropy measures uncertainty, while mutual information quantifies dependence between variables. These concepts often provide deeper insights than traditional statistical measures because they can capture nonlinear and complex relationships.
- Section 14.9 extends statistical reasoning into multiple dimensions. As datasets become higher-dimensional, visualization becomes more difficult and classical intuition becomes less reliable. Pay particular attention to energy-based and kernel-based methods, as they represent important modern approaches to multivariate statistical inference and machine learning.
- The Savitzky-Golay filters of Section 14.10 demonstrate a valuable principle in numerical analysis: smoothing should remove noise while preserving important structure. Unlike simple moving averages, these filters preserve local polynomial behavior, making them particularly useful in scientific measurements, spectroscopy, biomedical signals, and sensor data analysis.
- For Rust implementations, focus on numerical reliability as much as statistical correctness. Many statistical computations involve large sums, differences of nearly equal numbers, matrix operations, or repeated resampling. Learn why stable algorithms such as Welford's variance update, Kahan summation, and robust rank computations are preferred in practical software.
- Most importantly, develop the habit of questioning your results. Statistical methods provide evidence, not certainty. Always examine data quality, sample size, assumptions, uncertainty, and possible alternative explanations. The goal of statistical analysis is not simply to compute numbers, but to make trustworthy conclusions from imperfect data. This mindset will serve you well throughout numerical computing, data science, machine learning, and scientific research.

## 14.11.3. Further Learning with GenAI

To deepen your understanding of statistical description and inference in Rust, consider using the following GenAI prompts:

 1. Write a Rust program that computes descriptive statistics for a dataset of 200 values containing a majority cluster near 100 and several extreme outliers. Compute the sample mean, two-pass variance with Kahan-compensated accumulation, skewness, and excess kurtosis. Then compute the median, interquartile range, 10%-trimmed mean, and median-of-means with 5 blocks. Implement Welford's streaming algorithm by processing samples one at a time and verify that the final mean and variance agree with the two-pass results. Compare the sensitivity of classical moments versus robust estimators to the extreme values.
 2. Implement a Rust program that performs two-sample mean and variance comparison. Generate two samples with a moderate mean shift and unequal variances. Compute the pooled Student $t$-statistic, the Welch $t$-statistic with Satterthwaite degrees of freedom, and the $F$-test statistic for variance comparison. Implement the Brown-Forsythe test by computing absolute deviations from sample medians and comparing group means of the transformed values. Then implement a permutation test with 10000 label shuffles and compare the permutation $p$-value against the parametric result. Compute Cohen's $d$ as a standardized effect size measure.
 3. Build a Rust program that tests whether two samples come from the same distribution using complementary approaches. Compute the two-sample Kolmogorov-Smirnov statistic by merging and sorting the combined sample and evaluating the empirical CDFs at all order statistics. Implement the Anderson-Darling weighted statistic that emphasizes tail discrepancies through the denominator $F(x)(1 - F(x))$, and Kuiper's statistic $V = D^+ + D^-$ for rotation-invariant comparison. Calibrate all three statistics using permutation resampling with 2000 shuffles, and construct a global envelope for the ECDF difference curve that identifies where the distributions differ most.
 4. Implement contingency table analysis in Rust for a table of observed categorical frequencies. Compute the expected frequencies $E_{ij} = N_{i\cdot} N_{\cdot j} / N$, the Pearson chi-squared statistic with $(I-1)(J-1)$ degrees of freedom, Cramer's $V$, and the contingency coefficient $C$. Implement Goodman-Kruskal's $\lambda$ for predicting the column variable from the row variable and its reverse, and compute directional uncertainty coefficients $U(Y|X)$ and $U(X|Y)$ based on conditional entropy. Implement a permutation-based calibration of the chi-squared statistic by shuffling one variable's labels and report both the asymptotic and permutation $p$-values.
 5. Write a Rust program that computes and tests linear correlation between two variables using 80 paired observations with a known positive association. Compute the Pearson correlation $r$ using a Welford-style one-pass accumulation, test its significance using the $t$-statistic $t = r\sqrt{(N-2)/(1-r^2)}$, and construct a 95% confidence interval using Fisher's $z$-transform with standard error $1/\sqrt{N-3}$. Implement distance correlation using doubly centered pairwise distance matrices and verify that it detects the association. Implement bootstrap confidence intervals for Pearson $r$ using 5000 resamples and a permutation test for distance correlation using 2000 shuffles.
 6. Implement rank-based correlation analysis in Rust for 50 paired observations with tied values. Compute Spearman's $\rho_S$ using midranks and the Pearson-on-ranks formulation, and verify that the shortcut formula $\rho_S = 1 - 6\sum d_i^2 / [N(N^2 - 1)]$ agrees when ties are absent. Compute Kendall's $\tau_b$ by classifying all $\binom{N}{2}$ pairs as concordant, discordant, or tied. Implement the $O(N \log N)$ inversion-count algorithm for Kendall's $\tau$ in the tie-free case and verify agreement with the naive method. Compute the general dependence coefficient $\xi_n$ by ordering data by one variable and measuring rank variation in the other, and assess its reproducibility using bootstrap resampling.
 7. Build a Rust program that computes information-theoretic measures for discrete distributions. Construct two distributions from empirical counts, compute Shannon entropy, Kullback-Leibler divergence in both directions, and cross-entropy, and verify the decomposition $H(p, q) = H(p) + D_{\mathrm{KL}}(p \| q)$. Construct an empirical joint distribution from paired categorical observations, compute marginal distributions, conditional entropies $H(Y|X)$ and $H(X|Y)$, joint entropy $H(X, Y)$, and mutual information $I(X; Y)$. Verify the chain rule $H(X, Y) = H(X) + H(Y|X)$ and the identity $I(X; Y) = D_{\mathrm{KL}}(p_{XY} \| p_X p_Y)$ numerically.
 8. Write a Rust program that tests whether two sets of two-dimensional points differ in distribution using three methods. Implement the Fasano-Franceschini quadrant-based test by computing the maximum discrepancy over all four quadrants at each sample point and forming the averaged scaled statistic $\mathcal{D}$. Implement the energy distance statistic $E^2 = 2 A_{XY} - A_{XX} - A_{YY}$ from pairwise Euclidean distances. Calibrate both statistics using permutation resampling with 2000 shuffles and compare the resulting $p$-values and rejection decisions across the two methods.
 9. Implement a Rust program that designs and applies Savitzky-Golay smoothing filters for half-widths $M = 3, 5, 7$ with polynomial degree $d = 3$. For each filter, construct the Vandermonde matrix, compute the convolution coefficients $w = (X^\top X)^{-1} X^\top e_0$, and verify the symmetry property $w_k = w_{-k}$ and the normalization $\sum w_k = 1$. Apply each filter to a noisy synthetic signal and compute the RMSE relative to the noise-free signal. Verify the polynomial reproduction property by confirming that the filter exactly reproduces a cubic polynomial at interior points. Compare how wider filters provide stronger smoothing at the cost of reduced feature preservation.
10. Build a comprehensive Rust program that combines techniques from multiple sections of the chapter into a single analysis pipeline. Generate a dataset of 150 paired observations from a structured distribution. Compute descriptive statistics and robust estimators for each variable using streaming algorithms. Compare marginal distributions using the KS test. Construct a contingency table from discretized values and test independence using chi-squared with permutation calibration. Compute Pearson, Spearman, and Kendall correlation measures with bootstrap confidence intervals. Compute mutual information between discretized versions of the variables. Apply Savitzky-Golay smoothing to one variable and compare the smoothed and raw correlation estimates.

By engaging with these prompts, you will gain a deeper understanding of how descriptive statistics, hypothesis tests, correlation measures, information-theoretic quantities, multivariate distributional tests, and smoothing filters form an integrated toolkit for exploratory data analysis and statistical inference, and how the Rust implementations ensure numerical stability, type safety, and reproducibility across all stages of the analysis pipeline.

## 14.11.4. Homework Exercises

To reinforce your learning, complete the following exercises:

 1. Implement a Rust program that computes descriptive statistics for the dataset $x = [1000.0, 1000.1, 999.9, 1000.2, 999.8, 1000.0, 1000.1, 999.9, 1000.0, 1000.2, 999.8, 1000.1, 1000.0, 999.9, 1000.1, 1000.0, 2500.0, -400.0, 1000.2, 999.7]$ containing two extreme values. Compute the sample mean, two-pass variance with Kahan-compensated accumulation, skewness, and excess kurtosis. Compute the median, interquartile range, and median-of-means with 4 blocks. Implement the deliberately unstable variance formula $(\sum x_i^2 - N \bar{x}^2) / (N-1)$ and compare its output against the stable two-pass result. Implement Welford's streaming algorithm by processing samples one at a time, then partition the data into 3 blocks, compute streaming summaries for each block, merge them using the block-merge formula $M_{n+w} = M_n + M_w + (nw/(n+w))(\bar{X}_w - \bar{X}_n)^2$, and verify that the merged mean and variance agree with the direct computation to within $10^{-10}$.
 2. Implement a Rust program that compares two samples $A = [10.12, 9.94, 10.08, 10.01, 9.98, 10.10, 10.05, 9.97, 10.03, 10.00]$ and $B = [10.55, 9.31, 10.84, 9.76, 10.21, 8.95, 11.02, 9.48, 10.37, 9.62]$ which have similar means but different variances. Compute the pooled Student $t$-statistic, the Welch $t$-statistic with Satterthwaite degrees of freedom, and the $F$-test statistic $F = s_A^2 / s_B^2$. Implement the Brown-Forsythe variance comparison by computing absolute deviations from sample medians and forming the between-group versus within-group $F$-statistic on the transformed data. Implement a permutation test with 20000 label shuffles for the difference in means and report the two-sided $p$-value. Compute Cohen's $d$ and verify that the mean difference is small relative to the pooled standard deviation.
 3. Implement a Rust program that tests whether two samples of size 12 come from the same distribution. Use $X = [12.1, 12.4, 12.3, 12.2, 12.5, 12.7, 12.6, 12.4, 12.8, 12.5, 12.3, 12.6]$ and $Y = [11.9, 12.0, 12.1, 12.2, 12.2, 12.3, 12.4, 12.4, 12.5, 12.6, 12.7, 12.8]$. Compute the two-sample KS statistic $D = \sup_x |F_n(x) - G_m(x)|$ by evaluating the ECDFs at all pooled order statistics. Implement the Cramer-von Mises statistic by integrating squared ECDF differences over the support, and the Anderson-Darling statistic with the tail-sensitive weight $1/[F(x)(1-F(x))]$. Detect whether ties are present in the pooled sample. Calibrate all three statistics using permutation tests with 2000 shuffles and report permutation $p$-values.
 4. Implement contingency table analysis in Rust for paired categorical observations. Use the genotype-outcome data from the chapter: genotype $X \in \{A, B, C\}$ crossed with outcome $Y \in \{Success, Failure\}$. Construct the contingency table $N_{ij}$, compute expected frequencies $E_{ij} = N_{i\cdot} N_{\cdot j} / N$, the Pearson chi-squared statistic with $(I-1)(J-1)$ degrees of freedom, and an asymptotic $p$-value. Compute Cramer's $V$ and the contingency coefficient $C$, and report whether any expected counts fall below 5. Implement Goodman-Kruskal's $\lambda_{Y|X}$ by comparing baseline prediction errors to conditional errors, and compute the directional uncertainty coefficient $U(Y|X)$ from marginal and conditional entropies. Implement a permutation test by shuffling the $Y$ labels 5000 times and report the permutation $p$-value for the chi-squared statistic.
 5. Implement a Rust program that analyzes correlation between two continuous variables using 8 paired observations with strong positive association: $x = [1, 2, 3, 4, 5, 6, 7, 8]$ and $y = [1.2, 2.0, 3.1, 3.8, 5.2, 6.1, 6.9, 8.1]$. Compute the Pearson correlation $r$ using a Welford-style one-pass accumulation, the $t$-statistic $t = r\sqrt{(N-2)/(1-r^2)}$ with $N-2$ degrees of freedom, and a two-sided $p$-value via the regularized incomplete beta function. Construct a 95% confidence interval using Fisher's $z$-transform with standard error $1/\sqrt{N-3}$. Compute distance correlation using doubly centered pairwise distance matrices and verify it is positive. Implement a bootstrap 95% confidence interval for $r$ using 2000 resamples and report the percentile bounds.
 6. Implement rank-based correlation analysis in Rust for paired data $X = [12.0, 15.0, 15.0, 18.0, 21.0, 21.0, 25.0]$ and $Y = [30.0, 35.0, 33.0, 35.0, 40.0, 40.0, 50.0]$ which contain ties in both variables. Compute Spearman's $\rho_S$ by assigning midranks to tied values and computing the Pearson correlation on the rank vectors. Compute Kendall's $\tau_b$ by classifying all $\binom{7}{2} = 21$ pairs as concordant, discordant, tied in $X$ only, tied in $Y$ only, or tied in both, and applying the tie-corrected formula $\tau_b = (C - D) / \sqrt{(C+D+T_x)(C+D+T_y)}$. Verify that the shortcut formula is not applicable when ties are present. Verify that both $\rho_S$ and $\tau_b$ are invariant under monotone transformations of the data by applying $f(x) = 2x + 1$ and $g(y) = 3y - 4$ and recomputing.
 7. Implement the general rank-based dependence coefficient $xi_n$ in Rust for detecting functional relationships beyond monotonicity. Use the paired data $X = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]$ and $Y = [8.7, 3.8, 0.9, 0.0, 1.1, 4.2, 9.3]$, which exhibit a strong nonlinear (approximately quadratic) but nonmonotone relationship. Sort the data by $X$, compute upper ranks $r_i$ of the $Y$ values in the sorted order, and evaluate $\xi_n = 1 - n \sum_{i=1}^{n-1} |r_{i+1} - r_i| / [2 \sum_{i=1}^{n} l_i(n - l_i)]$ where $l_i$ counts the number of $Y$ values less than or equal to $Y_i$. Verify that $\xi_n$ is positive, reflecting the strong functional dependence, while Spearman's $\rho_S$ is close to zero due to the nonmonotone relationship. Verify invariance under monotone transformations by applying $f(x) = 2x + 3$ and $g(y) = e^y$ and recomputing $\xi_n$. Implement a bootstrap reproducibility assessment with 500 resamples and report the mean, standard deviation, and the 5th and 95th percentiles of the bootstrap distribution.
 8. Implement information-theoretic measures in Rust. Construct distributions $p$ and $q$ from count vectors $[40, 30, 20, 10]$ and $[35, 25, 25, 15]$ using the normalization $\hat{p}_i = n_i / N$. Compute Shannon entropy $H(p)$ and $H(q)$ with the convention $0 \log 0 = 0$, Kullback-Leibler divergence $D_{\mathrm{KL}}(p \| q)$ and $D_{\mathrm{KL}}(q \| p)$, and cross-entropy $H(p, q)$. Verify the decomposition $H(p, q) = H(p) + D_{\mathrm{KL}}(p \| q)$ to within $10^{-12}$. Construct an empirical joint distribution from paired categorical observations, compute $H(X)$, $H(Y)$, $H(X, Y)$, $H(Y|X)$, $H(X|Y)$, and $I(X; Y)$. Verify the chain rule $H(X, Y) = H(X) + H(Y|X)$ and the identity $I(X; Y) = D_{\mathrm{KL}}(p_{XY} \| p_X p_Y)$ numerically.
 9. Implement a Rust program that tests whether two sets of 8 two-dimensional points differ in distribution. Use sample X = \[(0.80, 1.00), (1.10, 1.20), (1.40, 0.90), (1.70, 1.50), (1.90, 1.10), (2.10, 1.80), (2.40, 1.40), (2.70, 2.00)\] and Y = \[(1.60, 0.40), (1.90, 0.70), (2.20, 0.60), (2.50, 0.90), (2.80, 1.00), (3.00, 1.20), (3.30, 1.10), (3.50, 1.40)\]. Implement the Fasano-Franceschini quadrant-based test by computing quadrant counts relative to each sample point, finding $D_1$ and $D_2$ as maximum discrepancies over pivots from each sample, and forming $D_0 = (D_1 + D_2)/2$ with scaling $\mathcal{D} = 2\sqrt{n_1 n_2 / (n_1 + n_2)} \cdot D_0$. Implement the energy statistic $E^2 = 2 A_{XY} - A_{XX} - A_{YY}$ from pairwise Euclidean distances. Calibrate the energy statistic using 2000 permutation resamples and report the $p$-value.
10. Implement Savitzky-Golay smoothing in Rust with half-width $M = 3$ and polynomial degree $d = 3$. Construct the Vandermonde matrix $X$ for the symmetric offsets $-M, \ldots, M$, solve the normal equations $(X^\top X) z = e_0$ using Gaussian elimination with partial pivoting, and form the convolution weights $w = X z$. Verify that the weights are symmetric with maximum error below $10^{-14}$ and sum to 1. Generate a noisy signal of 31 points composed of a smooth trend plus structured perturbation, apply the SG filter using reflected boundary conditions, and compute the RMS error relative to the clean signal before and after filtering. Verify the polynomial reproduction property by applying the filter to a cubic polynomial and confirming that the maximum interior error is below $10^{-10}$.

These exercises span the full range of statistical methods developed in this chapter, from descriptive statistics and robust estimation through parametric and nonparametric hypothesis tests, contingency table analysis, linear and rank-based correlation measures, information-theoretic dependence quantification, multivariate distributional testing, and Savitzky-Golay smoothing. By implementing them in Rust, you will gain direct experience with the numerical considerations, assumption validation, and interpretive judgment that distinguish reliable statistical analysis from mechanical formula application.

# References

 1. Alshahrani, N.D.A. (2026) ‘Statistical reproducibility of correlation tests: Pearson, Spearman, and Kendall’, *AIMS Mathematics*, 11(1), pp. 957–976. <https://doi.org/10.3934/math.2026042>
 2. Altieri, L., Cocchi, D. and Ventrucci, M. (2023) ‘Model-based entropy estimation for data with covariates and dependence structures’, *Environmental and Ecological Statistics*, 30, pp. 477–499. <https://doi.org/10.1007/s10651-023-00565-8>
 3. Ádám, N., Val’ko, D., Balogh, Z., Madoš, B. and Hurtuk, J. (2025) ‘Comparative evaluation of filtration techniques for ECG signal denoising with emphasis on stationary wavelet transform’, *Scientific Reports*, 15, 42514. <https://doi.org/10.1038/s41598-025-26476-1>
 4. Alvarez, L.A.F., Chiann, C. and Morettin, P.A. (2025) ‘Inference on model parameters with many L-moments’, *Journal of Econometrics*, 252, 106101. <https://doi.org/10.1016/j.jeconom.2025.106101>
 5. Badrudeen, A.T., Sahoo, D., Sawyer, C.B., Pike, J.W. and Harmel, R.D. (2026) ‘A critical review of statistical, signal processing and machine learning methods for continuous and high-frequency water quality data improvement’, *Ecological Informatics*, 94, 103619. <https://doi.org/10.1016/j.ecoinf.2026.103619>
 6. Boldo, S., Jeannerod, C.-P., Melquiond, G. and Muller, J.-M. (2023) ‘Floating-point arithmetic’, *Acta Numerica*, 32, pp. 203–290. <https://doi.org/10.1017/S0962492922000101>
 7. Bonnini, S., Assegie, G.M. and Trzcinska, K. (2024) ‘Review about the permutation approach in hypothesis testing’, *Mathematics*, 12(17), 2617. <https://doi.org/10.3390/math12172617>
 8. Calcagnile, L.M., Di Garbo, A. and Galatolo, S. (2026) ‘A benchmark for entropy estimators’, *Entropy*, 28(3), 311. <https://doi.org/10.3390/e28030311>
 9. Colantonio, L., Equeter, L., Dehombreux, P. and Ducobu, F. (2024) ‘Confidence interval estimation for cutting tool wear prediction in turning using bootstrap-based artificial neural networks’, *Sensors*, 24(11), 3432. <https://doi.org/10.3390/s24113432>
10. Cumming, G. and Calin-Jageman, R. (2024) *Introduction to the New Statistics: Estimation, Open Science, and Beyond*. 2nd edn. Routledge. <https://doi.org/10.4324/9781032689470>
11. Dalitz, C., Arning, J. and Goebbels, S. (2024) ‘A simple bias reduction for Chatterjee’s correlation’, *Journal of Statistical Theory and Practice*, 18, Article 51. <https://doi.org/10.1007/s42519-024-00399-y>
12. Dłotko, P., Hellmer, N., Stettner, Ł. and Topolnicki, R. (2023) ‘Topology-driven goodness-of-fit tests in arbitrary dimensions’, *Statistics and Computing*, 34(1), pp. 1–23. <https://doi.org/10.1007/s11222-023-10333-0>
13. Fasano, G. and Franceschini, A. (1987) ‘A multidimensional version of the Kolmogorov–Smirnov test’, *Monthly Notices of the Royal Astronomical Society*, 225, pp. 155–170.
14. Gao, L. and Baidoo, F. (2026) ‘Dekker’s floating point number system and compensated summation algorithms’, *arXiv* (math.NA), arXiv:2602.19452. <https://doi.org/10.48550/arXiv.2602.19452>
15. Grayson, K., Thober, S., Lacima-Nadolnik, A., Sharifi, E., Lledó, L. and Doblas-Reyes, F. (2025) ‘Statistical summaries for streamed data from climate simulations: one-pass algorithms (v0.6.2)’, *EGUsphere* (preprint), egusphere-2025-28. <https://doi.org/10.5194/egusphere-2025-28>
16. Höfler, M. (2026) ‘Robust tests should be the default, not the backup’, *Peer Community Journal*, 6, e1. <https://doi.org/10.24072/pcjournal.670>
17. Hutson, A.D. and Yu, H. (2023) ‘Exact inference around ordinal measures of association is often not exact’, *Computer Methods and Programs in Biomedicine*, 240, 107725. <https://doi.org/10.1016/j.cmpb.2023.107725>
18. Konstantinou, K., Mrkvička, T. and Myllymäki, M. (2024) ‘Graphical permutation tests for comparing sample distributions’, *arXiv*, arXiv:2403.01838. <https://doi.org/10.48550/arXiv.2403.01838>
19. Lanzante, J.R. (2021) ‘Testing for differences between two distributions in the presence of serial correlation using the Kolmogorov–Smirnov and Kuiper’s tests’, NOAA Technical Report. Available at: <https://repository.library.noaa.gov/view/noaa/32266>
20. Letizia, N.A., Novello, N. and Tonello, A.M. (2024) ‘Mutual information estimation via f-divergence and data derangements’, in *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. <https://doi.org/10.52202/079017-3338>
21. Li, X., Gao, Y., Chang, H., Huang, D., Ma, Y., Pan, R., Qi, H., Wang, F., Wu, S., Xu, K., Zhou, J., Zhu, X., Zhu, Y. and Wang, H. (2024) ‘A selective review on statistical methods for massive data computation: distributed computing, subsampling, and minibatch techniques’, *Statistical Theory and Related Fields*. <https://doi.org/10.1080/24754269.2024.2343151>
22. Liao, X. and Domański, P.D. (2025) ‘On estimation of α-stable distribution using L-moments’, *Fractal and Fractional*, 9(11), 711. <https://doi.org/10.3390/fractalfract9110711>
23. Lin, Z. and Han, F. (2023) ‘On boosting the power of Chatterjee’s rank correlation’, *Biometrika*, 110(2), pp. 283–299. <https://doi.org/10.1093/biomet/asac048>
24. Ma, X. and Xu, F. (2024) ‘Investigation on the sampling distributions of non-Gaussian wind pressure skewness and kurtosis’, *Mechanical Systems and Signal Processing*, 220, 111610. <https://doi.org/10.1016/j.ymssp.2024.111610>
25. Midões, C. and de Crombrugghe, D. (2023) ‘Improving the robustness of t-statistic based inference’, *Journal of Economic Inequality*, 21, pp. 977–1002. <https://doi.org/10.1007/s10888-023-09574-w>
26. Minsker, S. (2025) ‘Uniform bounds for robust mean estimators’, *Stochastic Processes and their Applications*, 190, 104724. <https://doi.org/10.1016/j.spa.2025.104724>
27. Monroy-Castillo, B.E. et al. (2024) ‘Improved distance correlation estimation’, *arXiv*, arXiv:2405.01958. <https://doi.org/10.48550/arXiv.2405.01958>
28. Pinchas, A., Ben-Gal, I. and Painsky, A. (2024) ‘A comparative analysis of discrete entropy estimators for large-alphabet problems’, *Entropy*, 26(5), 369. <https://doi.org/10.3390/e26050369>
29. Plummer, J.T. et al. (2025) ‘Standardized metrics for assessment and reproducibility of imaging-based spatial transcriptomics datasets’, *Nature Biotechnology*. <https://doi.org/10.1038/s41587-025-02811-9>
30. Puritz, C., Ness-Cohn, E. and Braun, R. (2023) ‘fasano.franceschini.test: An implementation of a multivariate KS test in R’, *The R Journal*. <https://doi.org/10.32614/RJ-2023-067>
31. Rolke, W. (2025) ‘Power studies for two-sample methods for multivariate data’, *arXiv*, arXiv:2507.16630. <https://doi.org/10.48550/arXiv.2507.16630>
32. Schmid, M., Rath, D. and Diebold, U. (2022) ‘Why and how Savitzky–Golay filters should be replaced’, *ACS Measurement Science Au*, 2(2), pp. 185–196. <https://doi.org/10.1021/acsmeasuresciau.1c00054>
33. Sepulveda, M.V. (2025) ‘Kendallknight: An R package for efficient implementation of Kendall’s correlation coefficient computation’, *PLOS ONE*, 20(6), e0326090. <https://doi.org/10.1371/journal.pone.0326090>
34. Stanley, M., Kuusela, M., Byrne, B. and Liu, J. (2024) ‘Technical note: Posterior uncertainty estimation via a Monte Carlo procedure specialized for 4D-Var data assimilation’, *Atmospheric Chemistry and Physics*, 24, pp. 9419–9433. <https://doi.org/10.5194/acp-24-9419-2024>
35. Stepanov, A. (2025) ‘Comparison of correlation coefficients’, *Sankhya A: The Indian Journal of Statistics*, 87(1), pp. 191–218. <https://doi.org/10.1007/s13171-025-00378-w>
36. Tu, S. et al. (2025) ‘Between- and within-cluster Spearman rank correlations’, *Statistics in Medicine*. <https://doi.org/10.1002/sim.10326>
37. Urasaki, W., Tahata, K. and Tomizawa, S. (2025) ‘New proportional reduction in error measures for contingency tables’, *arXiv*, arXiv:2503.06538. <https://doi.org/10.48550/arXiv.2503.06538>
38. Van Parys, B.P.G. and Zwart, B. (2025) ‘Robust mean estimation for optimization: The impact of heavy tails’, *arXiv* (math.OC), arXiv:2503.21421. <https://doi.org/10.48550/arXiv.2503.21421>
39. Xu, M. and Hutson, A.D. (2024) ‘A robust Spearman correlation permutation test’, *Communications in Statistics – Theory and Methods*, 53(6), pp. 2141–2153. <https://doi.org/10.1080/03610926.2022.2121144>
40. Yu, M., Ren, Z. and Zhou, W.-X. (2024) ‘Gaussian differentially private robust mean estimation and inference’, *Bernoulli*, 30(4), pp. 3059–3088. <https://doi.org/10.3150/23-BEJ1706>
41. Zhou, Y. (2023) ‘Statistical tests for homogeneity of variance for clinical trials and recommendations’, *Contemporary Clinical Trials Communications*, 33, 101119. <https://doi.org/10.1016/j.conctc.2023.101119>
