---
title: Chapter 15
description: ''
subtitle: Modeling of Data
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
date: '2026-04-03'
oxa: oxa:pqQDe4beUu67RvW3raYP/W48Jax0GAX3ayVNc5IUX
keywords: []
---

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/TqzmiYI4D6xgU38n2zgN.1","tags":[]}

> A good model can make sense of data; a great model makes sense of the world. — Michael Blastland

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/LZbJmiFSBaPjdXH4wwur.1","tags":[]}

*Chapter 15 develops the mathematical and computational foundations of data modeling, parameter estimation, and statistical inference. Beginning with least-squares estimation and its probabilistic interpretation, the chapter introduces linear and nonlinear regression, weighted fitting, errors-in-variables models, and general least-squares formulations. It then examines uncertainty quantification through confidence intervals, bootstrap methods, and robust estimation techniques designed for non-ideal data. The discussion extends beyond classical optimization to Bayesian inference using Markov chain Monte Carlo methods and nonparametric function modeling through Gaussian process regression. Throughout the chapter, numerical stability, conditioning, computational efficiency, and uncertainty interpretation are emphasized alongside practical Rust implementations. Together, these methods provide a comprehensive framework for transforming noisy observations into reliable predictive models for scientific computing, engineering analysis, machine learning, and data-driven decision-making.*

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/hkUwe3lcfduhRGErpcqa.5","tags":[]}

# 15.1. Introduction

This chapter develops a systematic framework for modeling data, in which observed measurements are translated into structured mathematical representations that support estimation, prediction, and inference. The emphasis is not only on fitting models to data, but also on understanding the assumptions that justify particular formulations, the geometric and statistical principles underlying estimation procedures, and the numerical methods required for stable and efficient computation. Building on the foundations of interpolation, optimization, and probability, the chapter establishes least squares and its generalizations as central tools, while also situating them within a broader modern context that includes uncertainty quantification, robust modeling, and large-scale numerical computation.

## 15.1.1. From Raw Observations to Quantitative Models

This chapter is concerned with the transformation of raw observations into quantitative statements about an underlying system. The goal is not only to summarize what has been measured, but to construct a mathematical model that captures structure in the data, provides estimates of unknown parameters, quantifies uncertainty in those estimates, and offers diagnostics by which the credibility of the model can be assessed. In this sense, modeling functions both as a means of compression and as a framework for interpretation. A large and possibly irregular collection of measurements is replaced by a smaller set of parameters and relations that retain the essential information needed for explanation, prediction, and computation.

The underlying viewpoint is that a model acts as a constrained form of interpolation. Instead of allowing an arbitrary function to pass through observed values, one restricts attention to a family of plausible relationships determined by physical principles, empirical regularity, or computational convenience. The data then determine which member of this family is most appropriate. This perspective is already present in the source draft and is fundamental to the chapter, because it clarifies that modeling is not merely the drawing of a curve through data, but the selection of a structured explanation from among admissible possibilities.

The chapter also rests on two complementary statistical viewpoints. In the frequentist framework, parameters are fixed but unknown, and uncertainty is interpreted through sampling variability. In the Bayesian framework, uncertain parameters are assigned prior plausibilities that are updated when new observations are obtained. The source text already introduced Bayes’ theorem as the mechanism for this updating process and emphasized its role in producing coherent, order-independent revisions of plausibility. That conceptual structure remains central here, because many of the estimation procedures discussed later can be interpreted either as classical optimization procedures or as components of Bayesian inference.

## 15.1.2. From Data to a Mathematical Problem

Suppose that we observe input-output pairs,

$$D=\{(x_i,y_i)\}_{i=1}^{N} \tag{15.1.1}$$

and propose a parameterized model,

$$y(x;\theta), \qquad \theta \in \mathbb{R}^{M} \tag{15.1.2}$$

The choice of model may be physical, when it reflects a known governing law; empirical, when it serves as a tractable approximation; or computational, when it acts as a surrogate for an expensive simulator. Regardless of its origin, the passage from this modeling idea to a concrete numerical problem requires a figure of merit, or objective function, that measures the discrepancy between the observations and the predictions of the model.

For each observation, define the residual as:

$$r_i(\theta)=y_i-y(x_i;\theta) \tag{15.1.3}$$

and then construct the objective,

$$\Phi(\theta)=\sum_{i=1}^{N}\rho_i\bigl(r_i(\theta)\bigr) \tag{15.1.4}$$

where the functions $\rho_i$ encode the assumed noise model, weighting, and any other information about measurement reliability or heteroscedasticity. The most important special case in this chapter is least squares, obtained when:

$$\rho_i(t)=t^2 \tag{15.1.5}$$

This choice will occupy a central place throughout the chapter, but it is important to emphasize from the beginning that it is not universally valid. Least squares is statistically justified under particular assumptions on the noise, especially Gaussian assumptions, and outside those assumptions it may be either a useful approximation or an inappropriate criterion.

This formulation makes clear that data modeling is fundamentally an optimization problem. The task is to choose parameters so that the discrepancy between model and data is minimized, or equivalently, in a probabilistic formulation, to maximize a likelihood. The introduction of residuals and objectives is therefore the point at which descriptive data analysis becomes mathematical inference.

### Rust Implementation

Following the formulation of the modeling problem in Section 15.1, particularly Subsection 15.1.2, Program 15.1.1 provides a concrete computational realization of the fundamental quantities introduced in Equations (15.1.1)–(15.1.5). The section establishes that data modeling begins with observed input-output pairs, a parameterized model $y(x;\theta)$, residuals $r_i(\theta)$, and an objective function $\Phi(\theta)$ that measures the discrepancy between model predictions and observed data. This program implements these elements for the simple straight-line model, demonstrating how a chosen parameter vector generates predictions, residuals, and a least-squares objective value. The goal is not yet to determine optimal parameters, but to illustrate how the abstract formulation of modeling translates into concrete numerical computation. In this way, the implementation provides a direct bridge between the conceptual framework of Section 15.1 and the algorithmic developments that follow.

At the core of the implementation is the `LineModel` structure, which represents the parameterized model $y(x;\theta)$ introduced in Equation (15.1.2). The method `predict` evaluates the model at a given input $x$, producing the corresponding output $y(x;\theta)$. This encapsulation reflects the idea that a model is a mapping from inputs to outputs determined by a parameter vector $\theta$.

The function `compute_residuals` directly implements the definition of residuals given in Equation (15.1.3). For each observation $(x_i, y_i)$, it computes the discrepancy between the observed value and the model prediction. The resulting vector of residuals provides a pointwise measure of model error and serves as the fundamental quantity from which the objective function is constructed.

The least-squares objective $\Phi(\theta)$, defined in Equation (15.1.4) with the quadratic choice of Equation (15.1.5), is evaluated by the function `least_squares_objective`. This function computes the sum of squared residuals, thereby quantifying the overall discrepancy between the model and the data. This scalar value plays a central role throughout the chapter, as it forms the basis for parameter estimation in least-squares methods.

The function `print_model_evaluation` provides a detailed, row-by-row comparison of observations, model predictions, and residuals. This explicit tabulation makes the structure of the problem transparent and helps relate the abstract definitions to concrete numerical values. The function `print_summary_statistics` complements this by computing aggregate measures such as the sum of squared residuals, mean squared residual, and root-mean-squared residual, which are commonly used diagnostics in data modeling.

The `main` function demonstrates the complete workflow introduced in Section 15.1. It defines a small dataset corresponding to Equation (15.1.1), specifies a trial parameter vector $\theta$, evaluates the model, computes residuals, and calculates the least-squares objective. The printed output illustrates how different parameter choices affect the quality of fit, thereby reinforcing the interpretation of modeling as an optimization problem.

```rust
// Program 15.1.1. Evaluating a Simple Data Model, Residuals, and Least-Squares Objective
//
// Problem statement:
// Given a small set of observed input-output pairs (x_i, y_i), evaluate a
// parameterized straight-line model
//
//     y(x; theta) = beta_0 + beta_1 x
//
// for a chosen parameter vector theta = (beta_0, beta_1). Compute:
//   1. the model predictions,
//   2. the residuals r_i = y_i - y(x_i; theta),
//   3. the least-squares objective Phi(theta) = sum_i r_i^2.
//
// This introductory program illustrates the computational meaning of the
// modeling framework described in Section 15.1 before moving on to actual
// parameter estimation algorithms in later sections.

#[derive(Clone, Copy, Debug)]
struct Observation {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct LineModel {
    beta0: f64,
    beta1: f64,
}

impl LineModel {
    /// Evaluate the model y(x; theta) = beta0 + beta1 * x.
    fn predict(&self, x: f64) -> f64 {
        self.beta0 + self.beta1 * x
    }
}

/// Compute the residual vector r_i = y_i - y(x_i; theta).
fn compute_residuals(data: &[Observation], model: &LineModel) -> Vec<f64> {
    data.iter()
        .map(|obs| obs.y - model.predict(obs.x))
        .collect()
}

/// Compute the least-squares objective Phi(theta) = sum_i r_i^2.
fn least_squares_objective(residuals: &[f64]) -> f64 {
    residuals.iter().map(|r| r * r).sum()
}

/// Print a row-by-row summary of observations, predictions, and residuals.
fn print_model_evaluation(data: &[Observation], model: &LineModel, residuals: &[f64]) {
    println!("Model Evaluation");
    println!("================");
    println!("Chosen parameters:");
    println!("  beta0 = {:>.10}", model.beta0);
    println!("  beta1 = {:>.10}", model.beta1);
    println!();

    println!(
        "{:>4} {:>14} {:>14} {:>18} {:>18}",
        "i", "x_i", "y_i", "y(x_i; theta)", "r_i"
    );
    println!("{}", "-".repeat(76));

    for (i, (obs, &r)) in data.iter().zip(residuals.iter()).enumerate() {
        let prediction = model.predict(obs.x);
        println!(
            "{:>4} {:>14.6} {:>14.6} {:>18.10} {:>18.10}",
            i, obs.x, obs.y, prediction, r
        );
    }
    println!();
}

/// Compute and print a few simple diagnostic summaries for interpretation.
fn print_summary_statistics(residuals: &[f64]) {
    let n = residuals.len() as f64;
    let sse = least_squares_objective(residuals);
    let mse = if n > 0.0 { sse / n } else { 0.0 };
    let rmse = mse.sqrt();

    println!("Residual Summary");
    println!("================");
    println!("Number of observations N         = {}", residuals.len());
    println!("Sum of squared residuals         = {:>.10}", sse);
    println!("Mean squared residual            = {:>.10}", mse);
    println!("Root-mean-squared residual       = {:>.10}", rmse);
    println!();
}

fn main() {
    // A small illustrative dataset.
    // In later sections this data would typically be stored in vectors or matrices,
    // but for this introductory example a list of observations is sufficient.
    let data = vec![
        Observation { x: 0.0, y: 1.1 },
        Observation { x: 1.0, y: 2.0 },
        Observation { x: 2.0, y: 2.9 },
        Observation { x: 3.0, y: 4.2 },
        Observation { x: 4.0, y: 5.1 },
        Observation { x: 5.0, y: 6.0 },
    ];

    // A user-chosen trial model. This is not yet the least-squares optimizer.
    // The purpose is to evaluate how well a proposed parameter vector fits the data.
    let model = LineModel {
        beta0: 1.0,
        beta1: 1.0,
    };

    let residuals = compute_residuals(&data, &model);

    println!("Section 15.1 Demonstration");
    println!("==========================");
    println!("This program evaluates a simple parameterized model");
    println!("against observed data, then computes residuals and");
    println!("the least-squares objective.");
    println!();

    print_model_evaluation(&data, &model, &residuals);
    print_summary_statistics(&residuals);

    let phi = least_squares_objective(&residuals);
    println!("Least-Squares Objective");
    println!("=======================");
    println!("Phi(theta) = sum_i r_i^2 = {:>.10}", phi);
    println!();

    println!("Interpretation");
    println!("==============");
    println!("A smaller value of Phi(theta) indicates that the chosen");
    println!("parameter vector produces predictions that lie closer to");
    println!("the observed data in the least-squares sense.");
}
```

Program 15.1.1 demonstrates how the abstract formulation of data modeling introduced in Section 15.1 is translated into concrete numerical computation. By explicitly evaluating the model, residuals, and least-squares objective, the program illustrates the transition from raw observations to a quantitative measure of model fit. This reflects the central idea that modeling becomes a mathematical inference problem once discrepancies between observations and predictions are systematically quantified.

The example also highlights the role of the least-squares objective as a scalar summary of model performance. Even for a simple linear model, the residuals reveal local deviations, while the objective function aggregates these into a single value that can be compared across different parameter choices. This provides the foundation for the optimization procedures developed in later sections, where the goal is to determine the parameter vector that minimizes this objective.

Although the implementation is deliberately simple, its structure is fully general and can be extended to more complex models, higher-dimensional parameter spaces, and alternative objective functions. In subsequent sections, this framework will be developed further to include systematic parameter estimation, statistical interpretation, and numerically stable solution techniques based on modern linear algebra methods.

## 15.1.3. The Linear-Algebra Core of Data Modeling

A large fraction of scientific inference problems reduce, either directly or after linearization, to linear least-squares problems. For this reason, numerical linear algebra treats least squares as a foundational primitive. Nonlinear least squares, Gauss-Newton methods, inverse problems, PDE-constrained optimization, and data assimilation all repeatedly produce linearized least-squares subproblems whose efficient and stable solution is essential in practice (Scott and Tůma, 2025; Daužickaitė et al., 2025).

The canonical linear model is:

$$y=A\beta+\varepsilon \tag{15.1.6}$$

where,

$$A\in\mathbb{R}^{N\times M}, \qquad \beta\in\mathbb{R}^{M}, \qquad y\in\mathbb{R}^{N}, \qquad \varepsilon\in\mathbb{R}^{N} \tag{15.1.7}$$

Here $A$ is the design matrix, $\beta$ is the unknown parameter vector, $y$ is the data vector, and $\varepsilon$ represents observational noise or model error. A typical basis-function construction takes the form:

$$
A =
\begin{pmatrix}
\phi_1(x_1) & \phi_2(x_1) & \cdots & \phi_M(x_1) \\
\phi_1(x_2) & \phi_2(x_2) & \cdots & \phi_M(x_2) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_1(x_N) & \phi_2(x_N) & \cdots & \phi_M(x_N)
\end{pmatrix},
\qquad
y =
\begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_N
\end{pmatrix}
\tag{15.1.8}
$$

Even when the original model is nonlinear in its physical parameters, linearization around a current iterate frequently yields precisely this structure for the update step. That is one of the main reasons least squares appears so persistently across scientific computing.

The geometry of the linear model is equally important. For the model $A\beta$, all predicted vectors lie in the column space $C(A)\subseteq\mathbb{R}^{N}$. Least squares selects the point in this subspace that is closest to the data vector $y$ in Euclidean distance. If $\hat{\beta}$ denotes the least-squares minimizer and,

$$r=y-A\hat{\beta} \tag{15.1.9}$$

is the residual vector, then the defining orthogonality condition is:

$$r\perp C(A)\quad \Longleftrightarrow \quad A^{\top}r=0 \tag{15.1.10}$$

This is the clearest geometric interpretation of least squares: the residual is orthogonal to every direction in the model subspace, so the approximation $A\hat{\beta}$ is the orthogonal projection of $y$ onto $C(A)$.

## 15.1.4. Numerical Computing Perspective and Modern Context

Although the mathematics of least squares is language-independent, its reliable implementation is not. A major theme in modern least-squares practice is that one should not explicitly form,

$$(A^{\top}A)^{-1} \tag{15.1.11}$$

because doing so can severely degrade numerical stability. Instead, stable factorizations such as QR and SVD should be preferred, together with solver interfaces that avoid explicit inversion. The source draft already emphasizes this computational principle and notes that contemporary Rust linear algebra libraries, such as *faer*, explicitly position QR and SVD as core tools for least-squares solution and rank-deficiency handling.

This numerical viewpoint is not an implementation afterthought. It reflects the fact that least squares now arises in settings where matrices are large, sparse, distributed, and sometimes processed in mixed precision. The chapter therefore places classical modeling ideas within a modern numerical context shaped by current developments in sparse least squares, randomized numerical linear algebra, robust statistics, and mixed-precision computation (Scott and Tůma, 2025; Murray et al., 2023; Loh, 2025; Carson and Daužickaitė, 2024).

What emerges from this introduction is a unified view of modeling of data. One begins with observations and a family of plausible models. One then defines residuals and objective functions that turn fitting into an optimization problem. Many such problems reduce to linear least squares, whose structure is governed by geometry and linear algebra. Finally, the interpretation and reliable computation of the resulting estimates depend on probabilistic assumptions, uncertainty quantification, and numerically stable algorithms. These themes provide the conceptual and computational foundation for the sections that follow.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/RQTicviebmB6jz1lt1yP.4","tags":[]}

# 15.2. Least Squares as a Maximum Likelihood Estimator

This section develops the fundamental connection between least squares and probability theory. While least squares is often introduced as a purely algebraic procedure for minimizing residuals, its deeper justification lies in likelihood theory. Under appropriate assumptions on the observational noise, least squares emerges naturally as a maximum likelihood estimator. This perspective is essential, because it clarifies when least squares is statistically justified, when it is merely a convenient approximation, and how it must be modified when the noise model changes.

The discussion proceeds by deriving least squares from a Gaussian likelihood, extending the formulation to heteroscedastic and correlated noise, examining the resulting optimality conditions, and then interpreting the estimator statistically and computationally in modern settings.

## 15.2.1. Gaussian Noise Leads Directly to Least Squares

Assume that the observed data are generated according to,

$$y_i = y(x_i;\theta) + \varepsilon_i, \qquad i = 1,\dots,N \tag{15.2.1}$$

where the errors are independent Gaussian random variables,

$$\varepsilon_i \sim \mathcal{N}(0,\sigma^2) \tag{15.2.2}$$

Under this assumption, the likelihood function is given by:

$$
p(y \mid \theta, \sigma^2) =
\prod_{i=1}^{N}
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left(
-\frac{(y_i - y(x_i;\theta))^2}{2\sigma^2}
\right)
\tag{15.2.3}
$$

Taking logarithms yields:

$$
\log p(y \mid \theta, \sigma^2)
=
-\frac{1}{2\sigma^2}
\sum_{i=1}^{N}
\bigl(y_i - y(x_i;\theta)\bigr)^2
-\frac{N}{2}\log(2\pi\sigma^2)
\tag{15.2.4}
$$

The second term does not depend on $\theta$, so maximizing the likelihood is equivalent to minimizing:

$$\sum_{i=1}^{N}(y_i - y(x_i;\theta))^2 \tag{15.2.5}$$

Thus,

$$\theta_{\mathrm{MLE}} = \arg\min_{\theta}\sum_{i=1}^{N}(y_i - y(x_i;\theta))^2 \tag{15.2.6}$$

This derivation shows that least squares is not arbitrary. It is precisely the maximum likelihood estimator under Gaussian noise assumptions. This connection is emphasized in modern statistical treatments of regression and likelihood (Ng and Ma, 2023; Pfister, 2024; Lovric, 2025).

A key conceptual point is that the choice of squared loss encodes a probabilistic assumption. If the noise is not Gaussian, then least squares is no longer the exact maximum likelihood estimator, although it may still be used as an approximation.

## 15.2.2. Heteroscedastic Noise and the Chi-Square Misfit

In many applications, measurement errors are not uniform. Instead, each observation may have its own variance:

$$\varepsilon_i \sim \mathcal{N}(0,\sigma_i^2) \tag{15.2.7}$$

The likelihood becomes:

$$
p(y \mid \theta) =
\prod_{i=1}^{N}
\frac{1}{\sqrt{2\pi\sigma_i^2}}
\exp\!\left(
-\frac{(y_i - y(x_i;\theta))^2}{2\sigma_i^2}
\right)
\tag{15.2.8}
$$

Taking logarithms gives:

$$
\log p(y \mid \theta)
=
-\frac{1}{2}
\sum_{i=1}^{N}
\left(
\frac{y_i - y(x_i;\theta)}{\sigma_i}
\right)^2
+ \text{const}
\tag{15.2.9}
$$

Thus the MLE minimizes,

$$\sum_{i=1}^{N}\left(\frac{y_i - y(x_i;\theta)}{\sigma_i}\right)^2 \tag{15.2.10}$$

This is the weighted least-squares objective, often interpreted as a chi-square misfit. Observations with smaller variance contribute more strongly, reflecting higher confidence in those measurements.

The baseline draft already emphasized that this chi-square formulation is central in practice, especially when measurement uncertainties are known or estimated. It also highlighted an important practical issue: when the variances $\sigma_i^2$ are unknown, they must be estimated, which introduces additional uncertainty and complicates statistical interpretation.

## 15.2.3. Correlated Noise and the Mahalanobis Norm

In more general settings, measurement errors are correlated, so that the noise affecting different observations cannot be treated as independent. This situation arises when measurements share common sources of uncertainty, such as systematic effects in instrumentation or dependencies induced by the data acquisition process. Suppose,

$$\varepsilon \sim \mathcal{N}(0,\Sigma) \tag{15.2.11}$$

where $\Sigma \in \mathbb{R}^{N\times N}$ is a covariance matrix describing both the variance of each observation and the correlations between them.

The presence of correlations modifies the likelihood function and, consequently, the objective function used for estimation. Instead of a simple sum of squared residuals, the negative log-likelihood becomes:

$$\Phi(\theta) = (y - f(\theta))^{\top}\Sigma^{-1}(y - f(\theta)) \tag{15.2.12}$$

Here, the inverse covariance matrix $\Sigma^{-1}$ acts as a weighting operator that accounts for both the relative reliability of observations and the dependencies among them. Residual components corresponding to directions of high uncertainty are downweighted, while those in well-determined directions contribute more strongly to the objective.

This expression defines a Mahalanobis norm, which generalizes the Euclidean norm by incorporating covariance structure. Unlike the standard Euclidean distance, which treats all directions equally, the Mahalanobis norm adapts to the geometry induced by $\Sigma$. Geometrically, it measures distance in a space where directions are scaled and rotated according to the uncertainty encoded in $\Sigma$, so that contours of equal distance form ellipsoids rather than spheres.

In the linear case $f(\theta)=A\beta$, this leads to generalized least squares:

$$\beta_{\mathrm{GLS}} = \arg\min_{\beta}|y - A\beta|_{\Sigma^{-1}}^2 \tag{15.2.13}$$

This formulation can be interpreted as solving a least-squares problem in a transformed coordinate system where the effects of correlation have been properly accounted for, ensuring that the resulting estimator reflects the true statistical structure of the data.

This formulation is fundamental in modern large-scale applications, particularly in variational data assimilation, where covariance matrices encode uncertainties in observations and model dynamics (Daužickaitė et al., 2025).

## 15.2.4. From the Objective to the Normal Equations (and When Not to Use Them)

For the linear model, the objective function derived from the likelihood formulation can be written as:

$$\Phi(\beta) = (y - A\beta)^{\top}W(y - A\beta), \qquad W = \Sigma^{-1} \tag{15.2.14}$$

This quadratic form expresses the weighted discrepancy between the observed data and the model prediction. The matrix $W$ incorporates the covariance structure of the noise, assigning appropriate weights and accounting for correlations among observations.

To determine the minimizer, one differentiates the objective with respect to $\beta$ and sets the gradient equal to zero. Carrying out this calculation yields the weighted normal equations:

$$A^{\top}WA\,\beta = A^{\top}Wy \tag{15.2.15}$$

These equations provide the necessary optimality condition for the least-squares solution. They characterize the point at which the weighted residual is orthogonal to the column space of $A$, thereby extending the geometric interpretation of least squares to the weighted setting.

Although the normal equations arise naturally from the optimization problem, their direct use in computation introduces important numerical concerns. The derivation itself is algebraically straightforward, but it conceals the effects of finite-precision arithmetic that become significant in practice.

In particular, forming the matrix $A^{\top}WA$ can significantly degrade numerical stability. Several mechanisms contribute to this behavior:

- The condition number is effectively squared, which increases sensitivity to perturbations in the data and amplifies the effects of rounding errors.
- Information can be lost through the accumulation of inner products, especially when columns of $A$ are nearly linearly dependent.
- Round-off errors may be amplified during the formation and subsequent solution of the system, leading to reduced accuracy in the computed parameters.

These issues are not merely theoretical; they are routinely encountered in large-scale or ill-conditioned problems. As a result, modern numerical linear algebra treats the normal equations primarily as a theoretical characterization of the solution rather than as a preferred computational method. In practical implementations, stable alternatives based on QR or SVD factorizations are used, since they avoid the explicit formation of $A^{\top}WA$ and preserve numerical reliability (Daužickaitė et al., 2025).

### Rust Implementation

Following the discussion in Section 15.2.4 on the derivation of the weighted normal equations (15.2.15) and the associated numerical concerns, Program 15.2.1 provides a practical implementation of weighted least-squares estimation using two different computational approaches. While the normal equations arise naturally from the optimality condition of the quadratic objective (15.2.14), their direct use in computation can lead to significant numerical instability due to the amplification of conditioning effects. This program constructs a weighted least-squares problem and solves it both by explicitly forming the normal equations and by applying a QR factorization to a suitably transformed system. By comparing the resulting parameter estimates and residual norms, the implementation illustrates the distinction between the mathematical characterization of the solution and its numerically stable realization.

At the core of the implementation is the weighted least-squares formulation introduced in (15.2.14), where the objective function measures the discrepancy between the observed data and the model prediction through a quadratic form involving the weight matrix $W = \Sigma^{-1}$. The function `weighted_normal_equations` translates the optimality condition (15.2.15) directly into code by constructing the matrix $A^{\top} W A$ and the vector $A^{\top} W y$. These quantities define a linear system whose solution yields the least-squares estimate. The function `solve_weighted_ls_normal_eq` then applies Gaussian elimination with partial pivoting to compute the parameter vector, thereby reflecting the classical but numerically sensitive approach discussed in the text.

To obtain a more stable formulation, the program introduces a whitening transformation through the function `whiten_system`. This transformation rescales the system so that the weighted objective in (15.2.14) is converted into a standard Euclidean least-squares problem. Specifically, each row of the design matrix and the observation vector is scaled by the inverse of the corresponding standard deviation, effectively incorporating the weighting implicitly. The resulting transformed system can then be solved without explicitly forming $A^{\top} W A$, thereby avoiding the loss of numerical precision associated with the normal equations.

The QR-based solution is implemented in `householder_qr_solve`, which applies Householder reflections to compute an implicit factorization of the transformed design matrix. This method orthogonalizes the columns of the matrix while preserving numerical stability, ensuring that the least-squares solution is obtained without squaring the condition number. The function `solve_weighted_ls_qr` combines the whitening step with the QR solver, providing a complete and robust alternative to the normal-equations approach.

The auxiliary functions support the computational workflow and diagnostic analysis. The function `weighted_residual_norm` evaluates the norm of the weighted residual, allowing direct comparison of how well each method minimizes the objective (15.2.14). The function `parameter_error` computes the deviation from a known reference parameter vector, illustrating the sensitivity of the estimated coefficients to conditioning effects. Additionally, the functions `inverse` and `cond_estimate_one_norm` provide an estimate of the condition number of $A^{\top} W A$, highlighting the amplification of numerical sensitivity discussed in Section 15.2.4.

The `main` function orchestrates the entire computation. It begins by constructing a weighted polynomial design matrix on a narrow interval, which induces moderate ill-conditioning while preserving identifiability. Synthetic observations are generated using a known parameter vector with a small deterministic perturbation to mimic measurement noise. The program then solves the resulting weighted least-squares problem using both the normal equations and the QR-based method. By reporting the estimated parameters, weighted residual norms, and differences between the solutions, the implementation provides a direct numerical illustration of the theoretical considerations discussed in this section.

```rust
// Program 15.2.1: Weighted Least Squares via QR and Normal Equations
//
// Problem Statement:
// Solve the weighted linear least-squares problem
//
//     min_beta (y - A beta)^T W (y - A beta),
//
// where W = Sigma^{-1} is diagonal, and compare:
//
// 1. Explicit formation of the weighted normal equations
//        A^T W A beta = A^T W y
//
// 2. A numerically safer QR-based solution applied to the whitened system.
//
// The example uses a moderately ill-conditioned polynomial design matrix
// on a narrow interval so that the effect of finite-precision arithmetic
// becomes visible without making the model nearly singular.

type Vector = Vec<f64>;
type Matrix = Vec<Vec<f64>>;

// ----------------------------------------------------------
// Basic helpers
// ----------------------------------------------------------

fn zeros(rows: usize, cols: usize) -> Matrix {
    vec![vec![0.0; cols]; rows]
}

fn mat_vec_mul(a: &Matrix, x: &Vector) -> Vector {
    let rows = a.len();
    let cols = a[0].len();
    let mut y = vec![0.0; rows];
    for i in 0..rows {
        for j in 0..cols {
            y[i] += a[i][j] * x[j];
        }
    }
    y
}

fn vec_sub(a: &Vector, b: &Vector) -> Vector {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn vec_norm2(x: &Vector) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn vec_norm_inf(x: &Vector) -> f64 {
    x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
}

fn dot(x: &Vector, y: &Vector) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn print_vector(name: &str, v: &Vector) {
    println!("{name}");
    for (i, val) in v.iter().enumerate() {
        println!("  [{:>2}] {:>.12e}", i, val);
    }
}

fn print_matrix(name: &str, a: &Matrix) {
    println!("{name}");
    for row in a {
        for val in row {
            print!("{:>18.10e} ", val);
        }
        println!();
    }
}

fn one_norm(a: &Matrix) -> f64 {
    let rows = a.len();
    let cols = a[0].len();
    let mut max_col_sum: f64 = 0.0;
    for j in 0..cols {
        let mut sum: f64 = 0.0;
        for i in 0..rows {
            sum += a[i][j].abs();
        }
        max_col_sum = max_col_sum.max(sum);
    }
    max_col_sum
}

// ----------------------------------------------------------
// Linear solver: Gaussian elimination with partial pivoting
// ----------------------------------------------------------

fn solve_linear_system(mut a: Matrix, mut b: Vector) -> Result<Vector, String> {
    let n = a.len();
    if n == 0 || a[0].len() != n || b.len() != n {
        return Err("Dimension mismatch in solve_linear_system".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_val = a[k][k].abs();

        for i in (k + 1)..n {
            if a[i][k].abs() > pivot_val {
                pivot_val = a[i][k].abs();
                pivot_row = i;
            }
        }

        if pivot_val < 1.0e-15 {
            return Err("Matrix is singular or numerically singular".to_string());
        }

        if pivot_row != k {
            a.swap(k, pivot_row);
            b.swap(k, pivot_row);
        }

        for i in (k + 1)..n {
            let factor = a[i][k] / a[k][k];
            a[i][k] = 0.0;
            for j in (k + 1)..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    Ok(x)
}

fn inverse(a: &Matrix) -> Result<Matrix, String> {
    let n = a.len();
    if n == 0 || a[0].len() != n {
        return Err("Matrix must be square for inversion".to_string());
    }

    let mut inv = zeros(n, n);
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let col = solve_linear_system(a.clone(), e)?;
        for i in 0..n {
            inv[i][j] = col[i];
        }
    }
    Ok(inv)
}

fn cond_estimate_one_norm(a: &Matrix) -> Result<f64, String> {
    let inv_a = inverse(a)?;
    Ok(one_norm(a) * one_norm(&inv_a))
}

// ----------------------------------------------------------
// Weighted least squares through normal equations
// ----------------------------------------------------------

fn weighted_normal_equations(a: &Matrix, y: &Vector, sigma: &Vector) -> (Matrix, Vector) {
    let n = a.len();
    let m = a[0].len();

    let mut ata = zeros(m, m);
    let mut aty = vec![0.0; m];

    for i in 0..n {
        let w = 1.0 / (sigma[i] * sigma[i]);
        for j in 0..m {
            aty[j] += w * a[i][j] * y[i];
            for k in 0..m {
                ata[j][k] += w * a[i][j] * a[i][k];
            }
        }
    }

    (ata, aty)
}

fn solve_weighted_ls_normal_eq(a: &Matrix, y: &Vector, sigma: &Vector) -> Result<Vector, String> {
    let (ata, aty) = weighted_normal_equations(a, y, sigma);
    solve_linear_system(ata, aty)
}

// ----------------------------------------------------------
// Whitening: transform weighted LS into ordinary LS
// ----------------------------------------------------------

fn whiten_system(a: &Matrix, y: &Vector, sigma: &Vector) -> (Matrix, Vector) {
    let n = a.len();
    let m = a[0].len();

    let mut aw = zeros(n, m);
    let mut yw = vec![0.0; n];

    for i in 0..n {
        let scale = 1.0 / sigma[i];
        yw[i] = scale * y[i];
        for j in 0..m {
            aw[i][j] = scale * a[i][j];
        }
    }

    (aw, yw)
}

// ----------------------------------------------------------
// Householder QR least-squares solver
// ----------------------------------------------------------

fn householder_qr_solve(a: &Matrix, b: &Vector) -> Result<Vector, String> {
    let n = a.len();
    let m = a[0].len();

    if b.len() != n {
        return Err("Dimension mismatch in householder_qr_solve".to_string());
    }
    if n < m {
        return Err("Need an overdetermined or square system with n >= m".to_string());
    }

    let mut r = a.clone();
    let mut qt_b = b.clone();

    for k in 0..m {
        let mut x = vec![0.0; n - k];
        for i in k..n {
            x[i - k] = r[i][k];
        }

        let norm_x = vec_norm2(&x);
        if norm_x < 1.0e-15 {
            continue;
        }

        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * norm_x;

        let v_norm = vec_norm2(&x);
        if v_norm < 1.0e-15 {
            continue;
        }

        let v: Vector = x.iter().map(|xi| xi / v_norm).collect();

        for j in k..m {
            let mut col_segment = vec![0.0; n - k];
            for i in k..n {
                col_segment[i - k] = r[i][j];
            }
            let proj = 2.0 * dot(&v, &col_segment);
            for i in k..n {
                r[i][j] -= proj * v[i - k];
            }
        }

        let mut b_segment = vec![0.0; n - k];
        for i in k..n {
            b_segment[i - k] = qt_b[i];
        }
        let proj_b = 2.0 * dot(&v, &b_segment);
        for i in k..n {
            qt_b[i] -= proj_b * v[i - k];
        }
    }

    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let mut sum = qt_b[i];
        for j in (i + 1)..m {
            sum -= r[i][j] * x[j];
        }
        if r[i][i].abs() < 1.0e-15 {
            return Err("Triangular factor is singular or numerically singular".to_string());
        }
        x[i] = sum / r[i][i];
    }

    Ok(x)
}

fn solve_weighted_ls_qr(a: &Matrix, y: &Vector, sigma: &Vector) -> Result<Vector, String> {
    let (aw, yw) = whiten_system(a, y, sigma);
    householder_qr_solve(&aw, &yw)
}

// ----------------------------------------------------------
// Diagnostics
// ----------------------------------------------------------

fn weighted_residual_norm(a: &Matrix, beta: &Vector, y: &Vector, sigma: &Vector) -> f64 {
    let r = vec_sub(y, &mat_vec_mul(a, beta));
    let rw: Vector = r
        .iter()
        .zip(sigma.iter())
        .map(|(ri, si)| ri / si)
        .collect();
    vec_norm2(&rw)
}

fn parameter_error(beta: &Vector, beta_true: &Vector) -> Vector {
    vec_sub(beta, beta_true)
}

// ----------------------------------------------------------
// Example construction
// ----------------------------------------------------------

fn build_example() -> (Matrix, Vector, Vector, Vector) {
    // Use polynomial columns 1, x, x^2, ..., x^6 on a narrow interval.
    // This produces a weighted Vandermonde-like matrix that is ill-conditioned
    // enough to make the normal-equations route less attractive, while still
    // keeping the model identifiable.

    let n = 24;
    let degree = 6;
    let m = degree + 1;

    let beta_true = vec![1.0, -1.5, 0.75, -0.30, 0.12, -0.04, 0.01];

    let mut a = zeros(n, m);
    let mut sigma = vec![0.0; n];
    let mut y = vec![0.0; n];

    for i in 0..n {
        // Narrow interval centered near x = 1
        let t = i as f64 / (n as f64 - 1.0);
        let x = 0.92 + 0.16 * t;

        let mut power = 1.0;
        for j in 0..m {
            a[i][j] = power;
            power *= x;
        }

        sigma[i] = 0.015 + 0.010 * t;

        let exact = dot(&a[i], &beta_true);
        let perturbation =
            sigma[i] * (0.40 * (9.0 * x).sin() - 0.25 * (5.0 * x).cos() + 0.10 * (13.0 * x).sin());

        y[i] = exact + perturbation;
    }

    (a, y, sigma, beta_true)
}

// ----------------------------------------------------------
// Main
// ----------------------------------------------------------

fn main() -> Result<(), String> {
    let (a, y, sigma, beta_true) = build_example();

    println!("Weighted Least Squares: QR Versus Normal Equations");
    println!("==================================================");
    println!();

    println!("Model");
    println!("-----");
    println!("We solve min_beta (y - A beta)^T W (y - A beta), with W = diag(1 / sigma_i^2).");
    println!("The design matrix is a weighted polynomial system on a narrow interval.");
    println!("This keeps the least-squares problem identifiable while still making");
    println!("the normal-equations route more sensitive to finite-precision effects.");
    println!();

    print_vector("True parameter vector beta_true", &beta_true);
    println!();

    print_vector("Observation standard deviations sigma_i", &sigma);
    println!();

    print_matrix("Design matrix A", &a);
    println!();

    print_vector("Observed data y", &y);
    println!();

    let (ata, _) = weighted_normal_equations(&a, &y, &sigma);
    print_matrix("Weighted normal matrix A^T W A", &ata);
    println!();

    let cond_ata = cond_estimate_one_norm(&ata)?;
    println!("Estimated 1-norm condition number of A^T W A = {:>.12e}", cond_ata);
    println!();

    let beta_ne = solve_weighted_ls_normal_eq(&a, &y, &sigma)?;
    let beta_qr = solve_weighted_ls_qr(&a, &y, &sigma)?;

    let err_ne = parameter_error(&beta_ne, &beta_true);
    let err_qr = parameter_error(&beta_qr, &beta_true);
    let diff = vec_sub(&beta_ne, &beta_qr);

    println!("Comparison");
    println!("----------");

    println!("Method                       = Weighted normal equations");
    print_vector("Estimated beta", &beta_ne);
    println!(
        "Weighted residual norm       = {:>.12e}",
        weighted_residual_norm(&a, &beta_ne, &y, &sigma)
    );
    println!(
        "Infinity-norm coefficient error = {:>.12e}",
        vec_norm_inf(&err_ne)
    );
    println!();

    println!("Method                       = Weighted QR factorization");
    print_vector("Estimated beta", &beta_qr);
    println!(
        "Weighted residual norm       = {:>.12e}",
        weighted_residual_norm(&a, &beta_qr, &y, &sigma)
    );
    println!(
        "Infinity-norm coefficient error = {:>.12e}",
        vec_norm_inf(&err_qr)
    );
    println!();

    print_vector("Difference beta_NE - beta_QR", &diff);
    println!();

    println!("Interpretation");
    println!("--------------");
    println!("Both methods target the same weighted least-squares minimizer.");
    println!("However, the normal-equations route explicitly forms A^T W A,");
    println!("which tends to magnify conditioning effects. The QR-based method");
    println!("works directly with the whitened design matrix and is therefore");
    println!("the preferred computational approach for numerically reliable");
    println!("least-squares solution of weighted problems.");
    println!();

    Ok(())
}
```

Program 15.2.1 demonstrates the practical implications of the theoretical discussion in Section 15.2.4 by comparing two computational approaches to the same weighted least-squares problem. Although both methods are derived from the same optimality condition (15.2.15) and aim to minimize the objective (15.2.14), their numerical behavior differs significantly in finite-precision arithmetic. The explicit formation of $A^{\top} W A$ in the normal-equations approach amplifies conditioning effects, leading to increased sensitivity in the computed parameters. In contrast, the QR-based method operates directly on the transformed system and preserves numerical stability by avoiding this amplification.

The example illustrates that even when both methods produce small residual norms, the corresponding parameter estimates may differ substantially in ill-conditioned settings. This highlights an important distinction between minimizing the objective function and obtaining a numerically reliable representation of the solution. The modular structure of the implementation also allows for straightforward extension to other least-squares formulations, including unweighted, regularized, or large-scale problems. More advanced techniques, such as singular value decomposition or iterative solvers with preconditioning, build upon the same principles to further improve robustness in challenging computational environments.

## 15.2.5. Statistical Output: Covariance, Noise Estimation, and Interpretation

A least-squares fit provides more than a point estimate of the parameters. Once a model has been fitted, it is equally important to quantify how reliable those parameter estimates are and how sensitive they are to variability in the data. Under Gaussian assumptions, the estimator admits a precise characterization of its uncertainty through its covariance matrix:

$$\mathrm{Cov}(\beta) = (A^{\top}\Sigma^{-1}A)^{-1} \tag{15.2.16}$$

This expression shows how uncertainty in the observations, as encoded in $\Sigma$, propagates through the model to produce uncertainty in the estimated parameters. The structure of the matrix $A^{\top}\Sigma^{-1}A$ reflects both the geometry of the design and the reliability of the data, so its inverse determines how well each component of $\beta$ can be resolved.

In the homoscedastic case, where all observations have the same variance and are uncorrelated, this expression simplifies to:

$$\mathrm{Cov}(\beta) = \sigma^2 (A^{\top}A)^{-1} \tag{15.2.17}$$

Here, the covariance separates into two factors: the scalar noise level $\sigma^2$, which sets the overall scale of uncertainty, and the matrix $(A^{\top}A)^{-1}$, which depends only on the design. This separation highlights an important distinction between the influence of measurement noise and the influence of the sampling configuration.

If $\sigma^2$ is not known a priori, it is commonly estimated from the residuals of the fit:

$$\sigma^2 = \frac{|y - A\hat{\beta}|_2^2}{N - M} \tag{15.2.18}$$

The denominator $N - M$ accounts for the degrees of freedom remaining after fitting $M$ parameters to $N$ observations. This adjustment reflects the fact that part of the variability in the data has already been explained by the fitted model, so only the remaining unexplained variation should be attributed to noise.

These formulas are not arbitrary. They arise directly from the probabilistic model and the structure of the least-squares estimator. In particular, they follow from the Gaussian likelihood and the linear dependence of the model on the parameters, which together yield explicit expressions for both the estimator and its distribution. As a result, they provide a consistent framework for quantifying parameter uncertainty, constructing confidence intervals, and performing hypothesis testing.

The baseline draft also emphasized an important limitation: if the noise level is estimated from the same data used for fitting, the possibility of an independent goodness-of-fit test is reduced. This reflects a fundamental trade-off, since the same data are being used both to determine the model parameters and to assess the variability of the residuals. Consequently, care must be taken when interpreting statistical diagnostics derived from such estimates.

### Rust Implementation

Following the discussion in Section 15.2.5 on the statistical interpretation of least-squares estimation, Program 15.2.2 provides a practical implementation that extends the computation of the parameter estimate to include measures of uncertainty and reliability. While the least-squares solution yields a point estimate of the parameters, the probabilistic framework underlying equations (15.2.16)–(15.2.18) shows that the estimator is accompanied by a well-defined covariance structure and a corresponding estimate of the noise level. This program computes the least-squares fit using a numerically stable QR-based method and then evaluates the residual variance, constructs the covariance matrix of the fitted parameters, and reports standard errors and confidence intervals. In doing so, it illustrates how the theoretical results of this subsection translate into a complete statistical description of the model fit in practical computation.

At the core of the implementation is the computation of the least-squares estimate $\hat{\beta}$, which is obtained using the function `householder_qr_solve`. This function applies Householder transformations to solve the linear least-squares problem without explicitly forming $A^{\top}A$, thereby ensuring numerical stability. Once the parameter estimate has been computed, the program proceeds to evaluate the statistical quantities associated with the fit, reflecting the framework introduced in (15.2.16)–(15.2.18).

The function `residual_vector` computes the residual $r = y - A\hat{\beta}$, which measures the discrepancy between the observed data and the fitted model. This residual is then used in `residual_variance_estimate` to evaluate the noise variance according to (15.2.18). By dividing the residual sum of squares by the degrees of freedom $N - M$, the implementation accounts for the number of fitted parameters and provides an unbiased estimate of the variance under the Gaussian model.

The covariance matrix of the parameter estimates is constructed by the function `covariance_matrix_homoscedastic`, which implements the homoscedastic formula (15.2.17). This function first forms the matrix $A^{\top}A$, computes its inverse, and scales the result by the estimated variance $\hat{\sigma}^2$. In this way, the code explicitly reflects the decomposition of uncertainty into a noise-dependent scalar factor and a design-dependent matrix, as discussed in the subsection. The diagonal entries of this matrix are then extracted in the function `standard_errors`, where their square roots provide the standard errors of the individual parameter estimates.

The `main` function integrates these components into a complete statistical workflow. It constructs a polynomial regression model, generates synthetic observations with a controlled perturbation, computes the least-squares estimate, and then evaluates the residual variance and covariance matrix. Finally, it reports approximate confidence intervals using a standard normal multiplier, thereby translating the covariance information into an interpretable measure of uncertainty for each parameter. This sequence of steps demonstrates how the theoretical results on covariance and noise estimation lead directly to practical tools for statistical inference.

```rust
// Program 15.2.2: Statistical Output for Linear Least Squares
//
// Problem Statement:
// Fit a linear least-squares model y ≈ A beta, then compute the
// statistical quantities associated with the fit:
//
// 1. The least-squares estimate beta_hat
// 2. The residual variance estimate sigma^2 from Equation (15.2.18)
// 3. The parameter covariance matrix from Equation (15.2.17)
// 4. Standard errors and approximate 95% confidence intervals
//
// The program uses a QR-based least-squares solver for numerical stability
// and then interprets the fitted model statistically.

type Vector = Vec<f64>;
type Matrix = Vec<Vec<f64>>;

// ----------------------------------------------------------
// Basic linear algebra helpers
// ----------------------------------------------------------

fn zeros(rows: usize, cols: usize) -> Matrix {
    vec![vec![0.0; cols]; rows]
}

fn dot(x: &Vector, y: &Vector) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn mat_vec_mul(a: &Matrix, x: &Vector) -> Vector {
    let rows = a.len();
    let cols = a[0].len();
    let mut y = vec![0.0; rows];
    for i in 0..rows {
        for j in 0..cols {
            y[i] += a[i][j] * x[j];
        }
    }
    y
}

fn vec_sub(a: &Vector, b: &Vector) -> Vector {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn vec_norm2(x: &Vector) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn transpose(a: &Matrix) -> Matrix {
    let rows = a.len();
    let cols = a[0].len();
    let mut t = zeros(cols, rows);
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
    let rows = a.len();
    let inner = a[0].len();
    let cols = b[0].len();
    let mut c = zeros(rows, cols);
    for i in 0..rows {
        for k in 0..inner {
            let aik = a[i][k];
            for j in 0..cols {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

fn print_vector(name: &str, v: &Vector) {
    println!("{name}");
    for (i, value) in v.iter().enumerate() {
        println!("  [{:>2}] {:>.12e}", i, value);
    }
}

fn print_matrix(name: &str, a: &Matrix) {
    println!("{name}");
    for row in a {
        for value in row {
            print!("{:>18.10e} ", value);
        }
        println!();
    }
}

// ----------------------------------------------------------
// Gaussian elimination with partial pivoting
// ----------------------------------------------------------

fn solve_linear_system(mut a: Matrix, mut b: Vector) -> Result<Vector, String> {
    let n = a.len();
    if n == 0 || a[0].len() != n || b.len() != n {
        return Err("Dimension mismatch in solve_linear_system".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_val = a[k][k].abs();

        for i in (k + 1)..n {
            if a[i][k].abs() > pivot_val {
                pivot_val = a[i][k].abs();
                pivot_row = i;
            }
        }

        if pivot_val < 1.0e-15 {
            return Err("Matrix is singular or numerically singular".to_string());
        }

        if pivot_row != k {
            a.swap(k, pivot_row);
            b.swap(k, pivot_row);
        }

        for i in (k + 1)..n {
            let factor = a[i][k] / a[k][k];
            a[i][k] = 0.0;
            for j in (k + 1)..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    Ok(x)
}

fn inverse(a: &Matrix) -> Result<Matrix, String> {
    let n = a.len();
    if n == 0 || a[0].len() != n {
        return Err("Matrix must be square for inversion".to_string());
    }

    let mut inv = zeros(n, n);
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let col = solve_linear_system(a.clone(), e)?;
        for i in 0..n {
            inv[i][j] = col[i];
        }
    }
    Ok(inv)
}

// ----------------------------------------------------------
// QR-based least-squares solver
// ----------------------------------------------------------

fn householder_qr_solve(a: &Matrix, b: &Vector) -> Result<Vector, String> {
    let n = a.len();
    let m = a[0].len();

    if b.len() != n {
        return Err("Dimension mismatch in householder_qr_solve".to_string());
    }
    if n < m {
        return Err("Need n >= m for least squares".to_string());
    }

    let mut r = a.clone();
    let mut qt_b = b.clone();

    for k in 0..m {
        let mut x = vec![0.0; n - k];
        for i in k..n {
            x[i - k] = r[i][k];
        }

        let norm_x = vec_norm2(&x);
        if norm_x < 1.0e-15 {
            continue;
        }

        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * norm_x;

        let v_norm = vec_norm2(&x);
        if v_norm < 1.0e-15 {
            continue;
        }

        let v: Vector = x.iter().map(|xi| xi / v_norm).collect();

        for j in k..m {
            let mut col_segment = vec![0.0; n - k];
            for i in k..n {
                col_segment[i - k] = r[i][j];
            }
            let proj = 2.0 * dot(&v, &col_segment);
            for i in k..n {
                r[i][j] -= proj * v[i - k];
            }
        }

        let mut b_segment = vec![0.0; n - k];
        for i in k..n {
            b_segment[i - k] = qt_b[i];
        }
        let proj_b = 2.0 * dot(&v, &b_segment);
        for i in k..n {
            qt_b[i] -= proj_b * v[i - k];
        }
    }

    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let mut sum = qt_b[i];
        for j in (i + 1)..m {
            sum -= r[i][j] * x[j];
        }
        if r[i][i].abs() < 1.0e-15 {
            return Err("Triangular factor is singular or numerically singular".to_string());
        }
        x[i] = sum / r[i][i];
    }

    Ok(x)
}

// ----------------------------------------------------------
// Statistical postprocessing
// ----------------------------------------------------------

fn residual_vector(a: &Matrix, beta: &Vector, y: &Vector) -> Vector {
    vec_sub(y, &mat_vec_mul(a, beta))
}

fn residual_variance_estimate(a: &Matrix, beta: &Vector, y: &Vector) -> Result<f64, String> {
    let n = a.len();
    let m = a[0].len();
    if n <= m {
        return Err("Need more observations than parameters to estimate variance".to_string());
    }

    let r = residual_vector(a, beta, y);
    let rss = dot(&r, &r);
    Ok(rss / ((n - m) as f64))
}

fn gram_matrix(a: &Matrix) -> Matrix {
    let at = transpose(a);
    mat_mul(&at, a)
}

fn covariance_matrix_homoscedastic(a: &Matrix, sigma2_hat: f64) -> Result<Matrix, String> {
    let ata = gram_matrix(a);
    let ata_inv = inverse(&ata)?;
    let rows = ata_inv.len();
    let cols = ata_inv[0].len();
    let mut cov = zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            cov[i][j] = sigma2_hat * ata_inv[i][j];
        }
    }
    Ok(cov)
}

fn standard_errors(cov: &Matrix) -> Vector {
    let n = cov.len();
    let mut se = vec![0.0; n];
    for i in 0..n {
        se[i] = cov[i][i].max(0.0).sqrt();
    }
    se
}

// ----------------------------------------------------------
// Example construction
// ----------------------------------------------------------

fn build_example() -> (Matrix, Vector, Vector) {
    // Quadratic model:
    // y = beta0 + beta1 x + beta2 x^2 + noise
    //
    // The synthetic data use deterministic pseudo-noise so that the
    // program is fully reproducible without external randomness.

    let n = 25;
    let beta_true = vec![1.20, -0.85, 0.45];

    let mut a = zeros(n, 3);
    let mut y = vec![0.0; n];

    for i in 0..n {
        let x = -2.0 + 4.0 * (i as f64) / ((n - 1) as f64);

        a[i][0] = 1.0;
        a[i][1] = x;
        a[i][2] = x * x;

        let signal = beta_true[0] + beta_true[1] * x + beta_true[2] * x * x;

        let noise = 0.08 * (0.7 * (2.7 * x).sin() - 0.4 * (1.3 * x).cos() + 0.2 * (4.1 * x).sin());

        y[i] = signal + noise;
    }

    (a, y, beta_true)
}

// ----------------------------------------------------------
// Main demonstration
// ----------------------------------------------------------

fn main() -> Result<(), String> {
    let (a, y, beta_true) = build_example();

    let n = a.len();
    let m = a[0].len();

    println!("Statistical Output for Linear Least Squares");
    println!("===========================================");
    println!();
    println!("Model");
    println!("-----");
    println!("We fit y ≈ A beta using QR-based least squares.");
    println!("After computing beta_hat, we estimate the residual variance");
    println!("and use it to form the covariance matrix of the fitted parameters.");
    println!();

    print_vector("True parameter vector beta_true", &beta_true);
    println!();

    print_matrix("Design matrix A", &a);
    println!();

    print_vector("Observed data y", &y);
    println!();

    let beta_hat = householder_qr_solve(&a, &y)?;
    let residuals = residual_vector(&a, &beta_hat, &y);
    let sigma2_hat = residual_variance_estimate(&a, &beta_hat, &y)?;
    let sigma_hat = sigma2_hat.sqrt();

    let cov_beta = covariance_matrix_homoscedastic(&a, sigma2_hat)?;
    let se_beta = standard_errors(&cov_beta);

    // Approximate 95% confidence intervals using z = 1.96.
    // For small samples, one could replace this by a Student-t multiplier.
    let z95 = 1.96_f64;

    println!("Least-Squares Fit");
    println!("-----------------");
    print_vector("Estimated parameter vector beta_hat", &beta_hat);
    println!();

    print_vector("Residual vector r = y - A beta_hat", &residuals);
    println!();

    println!("Residual Statistics");
    println!("-------------------");
    println!("Number of observations N        = {}", n);
    println!("Number of parameters M          = {}", m);
    println!("Residual 2-norm ||r||_2         = {:>.12e}", vec_norm2(&residuals));
    println!("Estimated noise variance sigma^2 = {:>.12e}", sigma2_hat);
    println!("Estimated noise standard deviation sigma = {:>.12e}", sigma_hat);
    println!();

    print_matrix("Estimated covariance matrix Cov(beta_hat)", &cov_beta);
    println!();

    println!("Parameter Uncertainty Summary");
    println!("-----------------------------");
    println!(
        "{:>8} {:>18} {:>18} {:>18} {:>18}",
        "Index", "beta_hat", "Std. Error", "CI Lower", "CI Upper"
    );

    for i in 0..m {
        let lower = beta_hat[i] - z95 * se_beta[i];
        let upper = beta_hat[i] + z95 * se_beta[i];
        println!(
            "{:>8} {:>18.10e} {:>18.10e} {:>18.10e} {:>18.10e}",
            i, beta_hat[i], se_beta[i], lower, upper
        );
    }

    println!();
    println!("Interpretation");
    println!("--------------");
    println!("The least-squares estimate beta_hat provides the fitted coefficients.");
    println!("The residual-based estimate of sigma^2 measures the unexplained");
    println!("variation after accounting for the fitted model.");
    println!("The covariance matrix translates that noise level into uncertainty");
    println!("for each component of beta_hat, and the diagonal entries determine");
    println!("the standard errors and confidence intervals reported above.");
    println!();

    Ok(())
}
```

Program 15.2.2 demonstrates that least-squares estimation extends naturally from a purely computational procedure to a statistically meaningful framework. While the parameter estimate $\hat{\beta}$ provides the best fit in the least-squares sense, the associated covariance matrix and residual variance quantify the reliability of that estimate and its sensitivity to data variability. The example shows how small residuals do not by themselves guarantee precise parameter estimates, and how the structure of the design matrix influences the uncertainty captured by (15.2.16) and (15.2.17).

The implementation also highlights the practical role of the residual-based variance estimate in (15.2.18). By using the same data both to fit the model and to estimate the noise level, the procedure reflects the trade-off discussed in the subsection: statistical diagnostics derived in this way must be interpreted with care, as they are not based on independent data. Nevertheless, the resulting covariance matrix and confidence intervals provide valuable insight into parameter uncertainty and form the basis for further statistical analysis, including hypothesis testing and model comparison.

The modular structure of the program allows for straightforward extensions to more general settings, including heteroscedastic or correlated noise models, where the full covariance expression (15.2.16) would be used. It also provides a foundation for more advanced techniques such as Bayesian inference, regularization, and uncertainty quantification in large-scale problems.

## 15.2.6. Bayesian View and Modern Computational Developments

From a Bayesian perspective, least squares emerges as a special case of maximum a posteriori estimation. In this framework, the parameters are treated as random variables rather than fixed but unknown quantities, and prior information about their likely values is incorporated explicitly. Suppose,

$$\beta \sim \mathcal{N}(\beta_0,\Lambda_0) \tag{15.2.19}$$

This prior distribution encodes both a central estimate $\beta_0$ and a covariance structure $\Lambda_0$ that reflects uncertainty or variability in the parameters before observing the data.

Combining this prior with the likelihood leads to a posterior distribution whose negative logarithm takes the form:

$$|y - A\beta|_{\Sigma^{-1}}^2+|\beta - \beta_0|_{\Lambda_0^{-1}}^2 \tag{15.2.20}$$

The first term measures the discrepancy between the data and the model, weighted by the observational covariance, while the second term measures deviation from the prior mean, weighted by the prior covariance. These two contributions act in balance, with the data-driven term favoring fidelity to observations and the prior term promoting adherence to previously specified structure.

Thus, MAP estimation corresponds to a regularized least-squares problem. The prior introduces a penalty term that stabilizes the solution and encodes prior knowledge. In particular, the presence of the second term prevents overfitting and controls the behavior of the estimator in directions where the data provide limited information. This is especially important in ill-posed problems, where classical least squares may be unstable or non-unique (Calvert Jump, 2024).

From this perspective, regularization is not an ad hoc modification but a direct consequence of incorporating prior information within a probabilistic framework. The equivalence between MAP estimation and regularized least squares therefore provides a unified interpretation of statistical inference and numerical stabilization.

Finally, although least squares is classical, its computational realization continues to evolve in response to the scale and complexity of modern data problems. Contemporary applications often involve large, sparse, or distributed datasets, requiring algorithms that are both efficient and numerically robust. Modern developments include:

- *Sparse and large-scale solvers with preconditioning* (Scott and Tůma, 2025). Modern least-squares problems often involve matrices that are extremely large but contain mostly zero entries. Exploiting this sparsity is essential for both memory efficiency and computational speed. Iterative methods are typically employed in such settings, and their performance depends critically on the conditioning of the system. Preconditioning modifies the problem into an equivalent form that is more favorable for iteration, reducing the number of steps required for convergence. As a result, sparse solvers combined with effective preconditioners enable the practical solution of problems that would otherwise be computationally infeasible.


- *Randomized numerical linear algebra and sketching methods* (Murray et al., 2023; Meier et al., 2024). Randomized methods provide approximate solutions to least-squares problems by projecting the data onto lower-dimensional subspaces. These projections, often referred to as sketches, preserve the essential structure of the problem while significantly reducing its size. By working with a compressed representation of the data, one can obtain solutions more quickly, especially when exact high-precision answers are unnecessary. These methods are particularly valuable in large-scale settings where traditional deterministic algorithms become too costly in terms of time or memory.


- *Mixed-precision iterative refinement* (Carson and Daužickaitė, 2024). Mixed-precision approaches exploit the fact that different stages of a computation can be carried out at different numerical precisions. Initial solutions may be computed using lower precision arithmetic, which is faster and more energy-efficient, and then refined iteratively using higher precision to recover accuracy. Iterative refinement corrects errors introduced in the low-precision phase, allowing the final solution to achieve near high-precision quality. This strategy balances efficiency and accuracy, making it well-suited for modern hardware architectures.


- *Distributed least-squares algorithms with communication-efficient designs* (Garg et al., 2024). In many contemporary applications, data are distributed across multiple processors or computing nodes. Solving least-squares problems in such environments requires algorithms that minimize communication between nodes, since data movement is often more expensive than computation. Communication-efficient designs reorganize computations to reduce synchronization and data transfer, allowing large problems to be solved collaboratively across distributed systems. These methods are essential for handling datasets that exceed the memory capacity of a single machine and for achieving scalability in high-performance computing environments.

These developments reflect the increasing scale and complexity of modern data problems and reinforce the central role of least squares in contemporary scientific computing.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/68UlrhzoouoSUh1JnERt.5","tags":[]}

# 15.3. Fitting Data to a Straight Line

This section develops the classical problem of fitting a straight line to data. Despite its apparent simplicity, this example plays a central role in understanding least squares. It provides a setting in which all aspects of the theory can be examined in a transparent and concrete manner. The geometric interpretation can be visualized directly in terms of projections, the maximum likelihood derivation can be written in closed form, numerical considerations can be analyzed without additional complications, and the statistical properties of the estimator can be described explicitly.

The straight-line model is especially valuable because it isolates the essential structure of the least-squares problem. With only a small number of parameters, it becomes possible to trace each step of the derivation from the definition of residuals to the final solution. This clarity allows one to see how the design matrix is constructed, how the normal equations arise, and how the solution depends on the distribution and scaling of the data. As a result, it serves as a concrete reference point for understanding more complex models.

Moreover, the straight-line model serves as a prototype for more general regression problems. Many of the ideas developed here extend directly to polynomial regression, basis-function expansions, and linearized nonlinear models. In each of these cases, the underlying structure remains that of fitting a linear combination of functions to observed data, even when the original problem appears nonlinear in its formulation.

For this reason, it is important to examine the problem in detail, including both its algebraic structure and its numerical implementation. Careful study of this simple case reveals the key principles that govern stability, accuracy, and interpretability in broader least-squares applications, and it prepares the foundation for the more general methods developed in the sections that follow.

## 15.3.1. Model Formulation and Weighted Objective

We assume that the independent variable $x$ is known exactly and that all uncertainty lies in the observed responses $y_i$. This assumption simplifies the structure of the problem by attributing all measurement error to the dependent variable, which allows the model to be expressed directly in terms of deviations in $y$. The straight-line model is:

$$y(x) = a + bx \tag{15.3.1}$$

where $a$ is the intercept and $b$ is the slope. These two parameters fully determine the line and represent the quantities to be estimated from the data.

Given data points $(x_i, y_i)$ with associated uncertainties $\sigma_i$, the discrepancy between the model and the observations is measured through a weighted least-squares objective, also referred to as the chi-square misfit:

$$\chi^2(a,b) = \sum_{i=1}^{N}\left(\frac{y_i - a - b x_i}{\sigma_i}\right)^2 \tag{15.3.2}$$

Each term in the sum represents a squared residual scaled by the corresponding standard deviation. This scaling ensures that observations with larger uncertainty contribute less to the total misfit, while more reliable observations have greater influence on the fitted parameters. In this way, the objective function incorporates information about the relative confidence in each measurement.

Under the assumption of independent Gaussian errors with variances $\sigma_i^2$, minimizing $\chi^2(a,b)$ is equivalent to maximum likelihood estimation (Pfister, 2024). This connection provides a statistical interpretation of the fitting procedure, showing that the chosen parameters are those that make the observed data most probable under the assumed noise model.

To simplify the algebra and make the structure of the solution more transparent, it is convenient to introduce weights defined by:

$$w_i = \frac{1}{\sigma_i^2} \tag{15.3.3}$$

so that the objective function can be expressed in terms of weighted sums. Using these weights, we define the aggregated quantities:

$$S = \sum_{i=1}^{N} w_i, \quad S_x = \sum_{i=1}^{N} w_i x_i, \quad S_y = \sum_{i=1}^{N} w_i y_i, \tag{15.3.4}$$

$$S_{xx} = \sum_{i=1}^{N} w_i x_i^2, \quad S_{xy} = \sum_{i=1}^{N} w_i x_i y_i \tag{15.3.5}$$

These weighted sums condense the entire dataset into a small set of scalar quantities that capture all the information relevant for estimating the parameters $a$ and $b$. Rather than working with all individual observations, the least-squares solution can be expressed entirely in terms of these aggregated values.

These quantities summarize all the information needed for the least-squares solution, and they form the basis for the explicit formulas derived in the following subsection.

## 15.3.2. Normal Equations and Explicit Solution

Minimizing $\chi^2(a,b)$ with respect to the parameters $a$ and $b$ leads to a system of linear equations known as the normal equations. These equations are obtained by setting the partial derivatives of the objective function equal to zero, which enforces the condition that the residuals are optimally balanced with respect to both parameters. In terms of the weighted sums introduced previously, the normal equations take the form:

$$
\begin{pmatrix}S & S_x \\S_x & S_{xx}\end{pmatrix}\begin{pmatrix}a \\b\end{pmatrix}

\begin{pmatrix}S_y \\S_{xy}\end{pmatrix}\tag{15.3.6}
$$

This system is a compact representation of the least-squares conditions, where the matrix on the left encodes the geometry of the data and the vector on the right represents the weighted correlation between the observations and the model components.

To solve this system explicitly, it is useful to introduce the determinant,

$$\Delta = S S_{xx} - S_x^2 \tag{15.3.7}$$

The quantity $\Delta$ measures the degree to which the data provide independent information about the slope and intercept. For a unique solution to exist, it is required that $\Delta \neq 0$, which corresponds to the condition that the values $x_i$ are not all identical under the weighting. If this condition fails, the data do not contain sufficient variation in $x$ to distinguish between different lines, and the parameters cannot be uniquely determined.

Solving the system yields explicit expressions for the estimators:

$$b = \frac{S S_{xy} - S_x S_y}{\Delta} \tag{15.3.8}$$

$$a = \frac{S_{xx} S_y - S_x S_{xy}}{\Delta} \tag{15.3.9}$$

These formulas show that both parameters are determined by combinations of the weighted sums, and they make clear how the distribution of the data influences the fitted line. In particular, the slope depends on the weighted covariance between $x_i$ and $y_i$, while the intercept adjusts the overall level of the fit.

These formulas are equivalent to the general matrix expression:

$$\beta = (A^\top W A)^{-1} A^\top W y \tag{15.3.10}$$

with,

$$A =\begin{pmatrix}1 & x_i\end{pmatrix}$$

This identification shows that the straight-line problem is a specific instance of the general least-squares framework, where the design matrix consists of a constant column and a column of the independent variable. The explicit formulas derived above therefore provide a concrete realization of the abstract matrix formulation introduced earlier.

Thus, the straight-line fit serves as a direct and fully transparent example of least squares, linking the algebraic solution, the matrix formulation, and the underlying statistical interpretation in a single setting.

### Rust Implementation

Following the derivation of the normal equations and their explicit solution in Section 15.3.2, Program 15.3.1 provides a direct computational implementation of the weighted straight-line least-squares problem using Equations (15.3.6)–(15.3.10). The section shows that the least-squares estimator can be expressed entirely in terms of aggregated weighted sums, leading to closed-form expressions for the slope and intercept. This program evaluates those weighted sums, forms the determinant $\Delta$, and computes the fitted parameters using the explicit formulas. By avoiding general matrix solvers and working directly with the derived expressions, the implementation makes transparent the connection between the algebraic structure of the normal equations and their numerical realization.

At the core of the implementation is the computation of the weighted sums defined in Equations (15.3.4) and (15.3.5). The function `compute_weighted_sums` evaluates the quantities $S$, $S_x$, $S_y$, $S_{xx}$, and $S_{xy}$ by iterating once over the dataset and accumulating contributions using the weights $w_i = 1/\sigma_i^2$. These aggregated values condense the entire dataset into a small set of scalars that fully determine the least-squares solution.

The function `fit_weighted_line_explicit` implements the explicit solution of the normal equations given in Equation (15.3.6). It first computes the determinant $\Delta$ as defined in Equation (15.3.7) and checks that it is nonzero, ensuring that the parameters are uniquely determined. The slope and intercept are then computed using Equations (15.3.8) and (15.3.9), respectively. This step directly mirrors the algebraic derivation and shows how the weighted covariance structure of the data determines the fitted line.

The function `compute_chi_squared` evaluates the weighted least-squares objective $\chi^2(a,b)$ introduced in Equation (15.3.2). It computes the residuals for each data point, scales them by the corresponding standard deviations, and accumulates the squared values. This provides a quantitative measure of the goodness of fit and connects the explicit solution to its statistical interpretation.

The supporting functions `predict`, `print_input_data`, `print_weighted_sums`, `print_fit_summary`, and `print_pointwise_fit` organize the computation and present the results in a structured format. In particular, the pointwise diagnostics display the predicted values, residuals, and standardized residuals for each observation, making it possible to inspect how individual data points contribute to the overall fit.

The `main` function demonstrates the full workflow. It defines a weighted dataset, computes the least-squares fit using the explicit formulas, and reports both aggregate and pointwise diagnostics. This end-to-end computation illustrates how the theoretical expressions derived in Section 15.3.2 translate into a practical algorithm for fitting a straight line to data.

```rust
// Program 15.3.1. Weighted Straight-Line Fit via Normal Equations and Explicit Solution
//
// Problem statement:
// Given observations (x_i, y_i) with known standard deviations sigma_i,
// fit the straight-line model
//
//     y(x) = a + b x
//
// by minimizing the weighted least-squares objective
//
//     chi^2(a,b) = sum_i ((y_i - a - b x_i) / sigma_i)^2.
//
// The program computes the weighted sums appearing in Equations (15.3.4)
// and (15.3.5), forms the determinant Delta from Equation (15.3.7),
// solves for the slope and intercept using Equations (15.3.8) and (15.3.9),
// and then reports the fitted line together with residual diagnostics.

#[derive(Clone, Copy, Debug)]
struct DataPoint {
    x: f64,
    y: f64,
    sigma: f64,
}

#[derive(Clone, Copy, Debug)]
struct WeightedSums {
    s: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    sxy: f64,
}

#[derive(Clone, Copy, Debug)]
struct FitResult {
    intercept: f64,
    slope: f64,
    delta: f64,
    chi2: f64,
    weighted_sums: WeightedSums,
}

fn validate_data(data: &[DataPoint]) -> Result<(), String> {
    if data.len() < 2 {
        return Err("at least two data points are required for a straight-line fit".to_string());
    }

    for (i, point) in data.iter().enumerate() {
        if !point.x.is_finite() || !point.y.is_finite() || !point.sigma.is_finite() {
            return Err(format!(
                "data point {} contains a non-finite value",
                i
            ));
        }
        if point.sigma <= 0.0 {
            return Err(format!(
                "data point {} has nonpositive sigma = {}",
                i, point.sigma
            ));
        }
    }

    Ok(())
}

fn compute_weighted_sums(data: &[DataPoint]) -> WeightedSums {
    let mut s = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;

    for point in data {
        let w = 1.0 / (point.sigma * point.sigma);
        s += w;
        sx += w * point.x;
        sy += w * point.y;
        sxx += w * point.x * point.x;
        sxy += w * point.x * point.y;
    }

    WeightedSums { s, sx, sy, sxx, sxy }
}

fn fit_weighted_line_explicit(data: &[DataPoint]) -> Result<FitResult, String> {
    validate_data(data)?;

    let sums = compute_weighted_sums(data);
    let delta = sums.s * sums.sxx - sums.sx * sums.sx;

    if delta.abs() <= f64::EPSILON {
        return Err(
            "the determinant Delta is zero or too small; the fit is not uniquely determined"
                .to_string(),
        );
    }

    let slope = (sums.s * sums.sxy - sums.sx * sums.sy) / delta;
    let intercept = (sums.sxx * sums.sy - sums.sx * sums.sxy) / delta;

    let chi2 = compute_chi_squared(data, intercept, slope);

    Ok(FitResult {
        intercept,
        slope,
        delta,
        chi2,
        weighted_sums: sums,
    })
}

fn predict(x: f64, intercept: f64, slope: f64) -> f64 {
    intercept + slope * x
}

fn compute_chi_squared(data: &[DataPoint], intercept: f64, slope: f64) -> f64 {
    data.iter()
        .map(|point| {
            let residual = point.y - predict(point.x, intercept, slope);
            let scaled = residual / point.sigma;
            scaled * scaled
        })
        .sum()
}

fn print_input_data(data: &[DataPoint]) {
    println!("Input Data");
    println!("==========");
    println!(
        "{:>4} {:>14} {:>14} {:>14} {:>14}",
        "i", "x_i", "y_i", "sigma_i", "w_i"
    );
    println!("{}", "-".repeat(68));

    for (i, point) in data.iter().enumerate() {
        let w = 1.0 / (point.sigma * point.sigma);
        println!(
            "{:>4} {:>14.6} {:>14.6} {:>14.6} {:>14.6}",
            i, point.x, point.y, point.sigma, w
        );
    }
    println!();
}

fn print_weighted_sums(sums: &WeightedSums) {
    println!("Weighted Sums");
    println!("=============");
    println!("S    = {:>.10}", sums.s);
    println!("S_x  = {:>.10}", sums.sx);
    println!("S_y  = {:>.10}", sums.sy);
    println!("S_xx = {:>.10}", sums.sxx);
    println!("S_xy = {:>.10}", sums.sxy);
    println!();
}

fn print_fit_summary(result: &FitResult) {
    println!("Explicit Solution of the Normal Equations");
    println!("=========================================");
    println!("Delta      = {:>.10}", result.delta);
    println!("intercept  = {:>.10}", result.intercept);
    println!("slope      = {:>.10}", result.slope);
    println!("chi^2      = {:>.10}", result.chi2);
    println!();
}

fn print_pointwise_fit(data: &[DataPoint], intercept: f64, slope: f64) {
    println!("Pointwise Fit Diagnostics");
    println!("=========================");
    println!(
        "{:>4} {:>12} {:>12} {:>14} {:>14} {:>14}",
        "i", "x_i", "y_i", "y_hat", "r_i", "r_i/sigma_i"
    );
    println!("{}", "-".repeat(78));

    for (i, point) in data.iter().enumerate() {
        let y_hat = predict(point.x, intercept, slope);
        let residual = point.y - y_hat;
        let standardized = residual / point.sigma;

        println!(
            "{:>4} {:>12.6} {:>12.6} {:>14.8} {:>14.8} {:>14.8}",
            i, point.x, point.y, y_hat, residual, standardized
        );
    }
    println!();
}

fn main() {
    // Illustrative weighted dataset.
    // The standard deviations sigma_i determine the weights w_i = 1 / sigma_i^2.
    let data = vec![
        DataPoint { x: 0.0, y: 1.05, sigma: 0.20 },
        DataPoint { x: 1.0, y: 2.10, sigma: 0.15 },
        DataPoint { x: 2.0, y: 2.95, sigma: 0.20 },
        DataPoint { x: 3.0, y: 4.05, sigma: 0.25 },
        DataPoint { x: 4.0, y: 5.20, sigma: 0.20 },
        DataPoint { x: 5.0, y: 6.10, sigma: 0.30 },
    ];

    println!("Weighted Straight-Line Fit");
    println!("==========================");
    println!("This program solves the straight-line least-squares problem");
    println!("using the explicit weighted formulas derived from the normal");
    println!("equations in Section 15.3.2.");
    println!();

    print_input_data(&data);

    match fit_weighted_line_explicit(&data) {
        Ok(result) => {
            print_weighted_sums(&result.weighted_sums);
            print_fit_summary(&result);
            print_pointwise_fit(&data, result.intercept, result.slope);

            println!("Interpretation");
            println!("==============");
            println!("The fitted line y = a + b x is obtained directly from the");
            println!("weighted sums without calling a general matrix solver.");
            println!("This makes the connection between the normal equations and");
            println!("their explicit closed-form solution completely transparent.");
        }
        Err(message) => {
            eprintln!("fit failed: {}", message);
            std::process::exit(1);
        }
    }
}
```

Program 15.3.1 demonstrates how the explicit solution of the normal equations can be implemented efficiently using only aggregated weighted sums. This approach reflects the central idea of Section 15.3.2: that the least-squares problem for a straight line can be reduced to a small set of scalar quantities that capture all relevant information from the data. The program shows how these quantities are computed, how they determine the fitted parameters, and how the resulting model can be evaluated through residual diagnostics.

The implementation also highlights the relationship between the scalar formulation and the general matrix expression in Equation (15.3.10). Although the problem can be written in matrix form, the straight-line case allows for a simplified computation that avoids explicit matrix operations while retaining the same mathematical content. This makes the method both efficient and easy to interpret.

At the same time, the program reveals the limitations of the explicit formulas. The computation of the determinant $\Delta$ involves subtraction of potentially large quantities, which can lead to numerical instability in finite-precision arithmetic. This observation motivates the centered and scaled formulation introduced in Section 15.3.3, where the same problem is reorganized to improve numerical robustness. Thus, the present implementation serves both as a complete solution and as a stepping stone toward more stable computational techniques.

## 15.3.3. Numerically Stable Centering and Scaling

Although the explicit formulas above are correct, they can be numerically unstable when the $x_i$ are large or closely clustered. The instability arises from subtracting large, nearly equal quantities in the computation of $\Delta$ and related terms. In finite-precision arithmetic, such cancellations can lead to a significant loss of accuracy, even when the underlying mathematical expressions are well defined. As a result, the direct evaluation of the formulas may produce unreliable parameter estimates when the data are poorly scaled.

To improve numerical stability, it is advantageous to reformulate the problem in terms of centered variables. Introduce the weighted mean as:

$$\bar{x}_w = \frac{S_x}{S} \tag{15.3.11}$$

This quantity represents the central location of the data in the independent variable, taking into account the weighting induced by the uncertainties. By measuring deviations relative to this mean, one reduces the magnitude of the quantities involved in the computation.

Define centered and scaled variables,

$$t_i = \frac{x_i - \bar{x}_w}{\sigma_i}, \qquad u_i = \frac{y_i}{\sigma_i} \tag{15.3.12}$$

The transformation serves two purposes. Centering reduces the effect of large offsets in the data, while scaling by $\sigma_i$ incorporates the weighting directly into the variables. In this way, both the geometry and the statistical structure of the problem are reflected in the transformed coordinates.

Using these variables, define:

$$S_{tt} = \sum_{i=1}^{N} t_i^2= \sum_{i=1}^{N} \frac{(x_i - \bar{x}w)^2}{\sigma_i^2}= S_{xx} - \frac{S_x^2}{S} \tag{15.3.13}$$

This expression shows explicitly how the centered formulation avoids the direct subtraction of large quantities present in the original definition of $\Delta$. Instead, the computation is organized in terms of deviations from the mean, which are typically much smaller and therefore more numerically stable.

In these variables, the slope becomes:

$$b = \frac{\sum_{i=1}^{N} t_i u_i}{\sum_{i=1}^{N} t_i^2} \tag{15.3.14}$$

and the intercept is:

$$a = \bar{y}_w - b \bar{x}_w, \qquad\bar{y}_w = \frac{S_y}{S} \tag{15.3.15}$$

These formulas retain the same mathematical content as the original expressions, but they are organized in a way that is less sensitive to round-off error. The slope is computed using centered products, and the intercept is obtained by anchoring the fitted line at the weighted mean.

This formulation is significantly more stable because it avoids subtracting large numbers and reduces the risk of catastrophic cancellation. The baseline draft emphasized that centering is essential in floating-point arithmetic, particularly when data values are large or tightly clustered. In practical computations, this reformulation is therefore preferred, as it preserves accuracy without altering the underlying least-squares solution.

### Rust Implementation

Following the discussion in Section 15.3.3 on numerical instability in the explicit least-squares formulas and the introduction of centered and scaled variables, Program 15.3.2 provides a practical implementation of the numerically stable weighted straight-line fit. The section shows that the instability in the determinant-based formulas arises from subtracting large, nearly equal quantities, which can lead to significant loss of precision in floating-point arithmetic. To address this, the problem is reformulated using the weighted mean $\bar{x}_w$ and the centered variables defined in Equations (15.3.11)–(15.3.15). This program implements that reformulation, computing the slope and intercept using centered weighted products and demonstrating how the same least-squares solution can be obtained in a numerically robust manner. For comparison, the program also evaluates the direct explicit solution, illustrating the effects of cancellation on poorly scaled data.

At the core of the implementation is the computation of the weighted mean $\bar{x}_w$ defined in Equation (15.3.11). The function `fit_weighted_line_centered` first evaluates the weighted sums $S$, $S_x$, and $S_y$, and then computes the weighted means of the independent and dependent variables. This establishes the reference point about which the data are centered, reducing the magnitude of the quantities involved in subsequent computations.

The centered and scaled variables introduced in Equation (15.3.12) are evaluated implicitly within the loop that computes the sums $S_{tt}$ and $S_{tu}$. For each data point, the transformed variables $t_i$ and $u_i$ are formed, and their products are accumulated to compute the numerator and denominator of the slope formula in Equation (15.3.14). This organization avoids forming large intermediate quantities and instead works with deviations from the weighted mean, which are typically much smaller and therefore more stable numerically.

The slope is then computed as the ratio $b = S_{tu} / S_{tt}$, and the intercept is recovered using Equation (15.3.15) as $a = \bar{y}_w - b \bar{x}_w$. This step reflects the geometric interpretation of the centered formulation, where the fitted line is anchored at the weighted mean and the slope describes the variation about that point.

For comparison, the function `fit_weighted_line_explicit` implements the direct determinant-based formulas from Equations (15.3.7)–(15.3.9). By evaluating both approaches on the same dataset, the program illustrates how cancellation errors can affect the explicit formulas when the data are poorly scaled. The difference between the two solutions provides a quantitative measure of the numerical instability discussed in the section.

The remaining functions organize the computation and present the results. The function `compute_chi_squared` evaluates the weighted objective in Equation (15.3.2), while the diagnostic routines display centered deviations, predicted values, residuals, and standardized residuals. The `main` function constructs a deliberately poorly scaled dataset, executes both fitting methods, and reports their results, thereby demonstrating the practical importance of centering and scaling in numerical computation.

```rust
// Program 15.3.2. Numerically Stable Weighted Straight-Line Fit by Centering and Scaling
//
// Problem statement:
// Given observations (x_i, y_i) with known standard deviations sigma_i,
// fit the straight-line model
//
//     y(x) = a + b x
//
// by minimizing the weighted least-squares objective. Instead of using the
// explicit formulas based directly on the determinant Delta, this program
// computes the weighted mean x_bar_w, forms the centered and scaled variables
//
//     t_i = (x_i - x_bar_w) / sigma_i,
//     u_i = y_i / sigma_i,
//
// and then evaluates the slope and intercept using the numerically stable
// formulas from Section 15.3.3. For comparison, the program also computes
// the direct explicit solution and reports the difference between the two
// approaches on a poorly scaled dataset.

#[derive(Clone, Copy, Debug)]
struct DataPoint {
    x: f64,
    y: f64,
    sigma: f64,
}

#[derive(Clone, Copy, Debug)]
struct WeightedSums {
    s: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    sxy: f64,
}

#[derive(Clone, Copy, Debug)]
struct ExplicitFitResult {
    intercept: f64,
    slope: f64,
    delta: f64,
}

#[derive(Clone, Copy, Debug)]
struct CenteredFitResult {
    intercept: f64,
    slope: f64,
    x_bar_w: f64,
    y_bar_w: f64,
    s_tt: f64,
    s_tu: f64,
    chi2: f64,
}

fn validate_data(data: &[DataPoint]) -> Result<(), String> {
    if data.len() < 2 {
        return Err("at least two data points are required for a straight-line fit".to_string());
    }

    for (i, point) in data.iter().enumerate() {
        if !point.x.is_finite() || !point.y.is_finite() || !point.sigma.is_finite() {
            return Err(format!("data point {} contains a non-finite value", i));
        }
        if point.sigma <= 0.0 {
            return Err(format!(
                "data point {} has nonpositive sigma = {}",
                i, point.sigma
            ));
        }
    }

    Ok(())
}

fn compute_weighted_sums(data: &[DataPoint]) -> WeightedSums {
    let mut s = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;

    for point in data {
        let w = 1.0 / (point.sigma * point.sigma);
        s += w;
        sx += w * point.x;
        sy += w * point.y;
        sxx += w * point.x * point.x;
        sxy += w * point.x * point.y;
    }

    WeightedSums { s, sx, sy, sxx, sxy }
}

fn fit_weighted_line_explicit(data: &[DataPoint]) -> Result<ExplicitFitResult, String> {
    validate_data(data)?;

    let sums = compute_weighted_sums(data);
    let delta = sums.s * sums.sxx - sums.sx * sums.sx;

    if delta.abs() <= f64::EPSILON * (sums.s * sums.sxx).abs().max(1.0) {
        return Err(
            "the determinant Delta is zero or too small; the direct fit is not uniquely determined"
                .to_string(),
        );
    }

    let slope = (sums.s * sums.sxy - sums.sx * sums.sy) / delta;
    let intercept = (sums.sxx * sums.sy - sums.sx * sums.sxy) / delta;

    Ok(ExplicitFitResult {
        intercept,
        slope,
        delta,
    })
}

fn fit_weighted_line_centered(data: &[DataPoint]) -> Result<CenteredFitResult, String> {
    validate_data(data)?;

    let sums = compute_weighted_sums(data);

    if sums.s <= 0.0 {
        return Err("sum of weights must be positive".to_string());
    }

    let x_bar_w = sums.sx / sums.s;
    let y_bar_w = sums.sy / sums.s;

    let mut s_tt = 0.0;
    let mut s_tu = 0.0;

    for point in data {
        let t_i = (point.x - x_bar_w) / point.sigma;
        let u_i = point.y / point.sigma;
        s_tt += t_i * t_i;
        s_tu += t_i * u_i;
    }

    if s_tt.abs() <= f64::EPSILON {
        return Err(
            "the centered sum S_tt is zero or too small; the fit is not uniquely determined"
                .to_string(),
        );
    }

    let slope = s_tu / s_tt;
    let intercept = y_bar_w - slope * x_bar_w;
    let chi2 = compute_chi_squared(data, intercept, slope);

    Ok(CenteredFitResult {
        intercept,
        slope,
        x_bar_w,
        y_bar_w,
        s_tt,
        s_tu,
        chi2,
    })
}

fn predict(x: f64, intercept: f64, slope: f64) -> f64 {
    intercept + slope * x
}

fn compute_chi_squared(data: &[DataPoint], intercept: f64, slope: f64) -> f64 {
    data.iter()
        .map(|point| {
            let residual = point.y - predict(point.x, intercept, slope);
            let scaled = residual / point.sigma;
            scaled * scaled
        })
        .sum()
}

fn print_input_data(data: &[DataPoint]) {
    println!("Input Data");
    println!("==========");
    println!(
        "{:>4} {:>18} {:>16} {:>12} {:>14}",
        "i", "x_i", "y_i", "sigma_i", "w_i"
    );
    println!("{}", "-".repeat(74));

    for (i, point) in data.iter().enumerate() {
        let w = 1.0 / (point.sigma * point.sigma);
        println!(
            "{:>4} {:>18.6} {:>16.6} {:>12.6} {:>14.6}",
            i, point.x, point.y, point.sigma, w
        );
    }
    println!();
}

fn print_centered_quantities(result: &CenteredFitResult) {
    println!("Centered and Scaled Quantities");
    println!("==============================");
    println!("x_bar_w = {:>.10}", result.x_bar_w);
    println!("y_bar_w = {:>.10}", result.y_bar_w);
    println!("S_tt    = {:>.10}", result.s_tt);
    println!("S_tu    = {:>.10}", result.s_tu);
    println!();
}

fn print_fit_summary(centered: &CenteredFitResult, explicit: Option<&ExplicitFitResult>) {
    println!("Stable Centered Fit");
    println!("===================");
    println!("intercept  = {:>.10}", centered.intercept);
    println!("slope      = {:>.10}", centered.slope);
    println!("chi^2      = {:>.10}", centered.chi2);
    println!();

    if let Some(explicit_fit) = explicit {
        println!("Comparison with Direct Explicit Formulas");
        println!("========================================");
        println!("direct intercept = {:>.10}", explicit_fit.intercept);
        println!("direct slope     = {:>.10}", explicit_fit.slope);
        println!("Delta            = {:>.10}", explicit_fit.delta);
        println!(
            "difference in intercept = {:>.10e}",
            centered.intercept - explicit_fit.intercept
        );
        println!(
            "difference in slope     = {:>.10e}",
            centered.slope - explicit_fit.slope
        );
        println!();
    }
}

fn print_pointwise_fit(data: &[DataPoint], intercept: f64, slope: f64, x_bar_w: f64) {
    println!("Pointwise Diagnostics");
    println!("=====================");
    println!(
        "{:>4} {:>18} {:>14} {:>14} {:>14} {:>14}",
        "i", "x_i - x_bar_w", "y_i", "y_hat", "r_i", "r_i/sigma_i"
    );
    println!("{}", "-".repeat(90));

    for (i, point) in data.iter().enumerate() {
        let centered_x = point.x - x_bar_w;
        let y_hat = predict(point.x, intercept, slope);
        let residual = point.y - y_hat;
        let standardized = residual / point.sigma;

        println!(
            "{:>4} {:>18.6} {:>14.6} {:>14.8} {:>14.8} {:>14.8}",
            i, centered_x, point.y, y_hat, residual, standardized
        );
    }
    println!();
}

fn main() {
    // A deliberately poorly scaled dataset:
    // the x_i values are large and closely clustered, which makes direct
    // formulas based on Delta more vulnerable to cancellation.
    let data = vec![
        DataPoint { x: 1_000_000.0, y: 2.02, sigma: 0.20 },
        DataPoint { x: 1_000_001.0, y: 2.48, sigma: 0.20 },
        DataPoint { x: 1_000_002.0, y: 3.05, sigma: 0.15 },
        DataPoint { x: 1_000_003.0, y: 3.46, sigma: 0.25 },
        DataPoint { x: 1_000_004.0, y: 4.01, sigma: 0.20 },
        DataPoint { x: 1_000_005.0, y: 4.53, sigma: 0.30 },
    ];

    println!("Numerically Stable Weighted Straight-Line Fit");
    println!("=============================================");
    println!("This program solves the weighted straight-line least-squares");
    println!("problem using the centered and scaled formulation of");
    println!("Section 15.3.3, and compares it with the direct explicit");
    println!("formulas from Section 15.3.2.");
    println!();

    print_input_data(&data);

    let centered_result = match fit_weighted_line_centered(&data) {
        Ok(result) => result,
        Err(message) => {
            eprintln!("stable centered fit failed: {}", message);
            std::process::exit(1);
        }
    };

    let explicit_result = fit_weighted_line_explicit(&data).ok();

    print_centered_quantities(&centered_result);
    print_fit_summary(&centered_result, explicit_result.as_ref());
    print_pointwise_fit(
        &data,
        centered_result.intercept,
        centered_result.slope,
        centered_result.x_bar_w,
    );

    println!("Interpretation");
    println!("==============");
    println!("By centering the x_i-values around the weighted mean, the");
    println!("computation avoids subtracting large nearly equal quantities.");
    println!("This preserves the least-squares solution while improving");
    println!("numerical robustness in floating-point arithmetic.");
}
```

Program 15.3.2 demonstrates the practical importance of reorganizing computations to improve numerical stability. By expressing the least-squares problem in terms of centered and scaled variables, the implementation avoids the subtraction of large, nearly equal quantities that can lead to catastrophic cancellation in floating-point arithmetic. The resulting formulation produces parameter estimates that are consistent with the theoretical solution while maintaining numerical accuracy even when the data are poorly scaled.

The comparison with the explicit determinant-based formulas highlights a key principle of numerical computing: mathematically equivalent expressions can behave very differently in finite precision. Although both formulations yield the same result in exact arithmetic, the centered version is significantly more reliable in practice because it works with quantities of moderate magnitude and reduces sensitivity to rounding errors.

This example reinforces a broader lesson that extends beyond least-squares fitting. Numerical stability depends not only on the correctness of the underlying mathematics but also on the structure of the computation. Techniques such as centering, scaling, and careful accumulation of sums are essential tools for ensuring that algorithms remain accurate and robust when implemented on finite-precision hardware. In subsequent sections, these ideas will reappear in more general settings, where stability considerations play a central role in the design of efficient numerical methods.

## 15.3.4. Parameter Uncertainty and Correlation

Under Gaussian noise assumptions with known variances, the least-squares estimator is not only a point estimate but also a random variable with a well-defined covariance structure. This covariance matrix quantifies how uncertainty in the data propagates to uncertainty in the estimated parameters. For the straight-line model, the covariance matrix of the parameters is:

$$
\mathrm{Cov}\begin{pmatrix}a \\b\end{pmatrix}=

(A^\top W A)^{-1}=

\frac{1}{\Delta}\begin{pmatrix} S_{xx} & -S_x \\-S_x & S \end{pmatrix}\tag{15.3.16}
$$

This matrix is obtained directly from the inverse of the weighted normal matrix, and its entries describe both the individual variances of the parameters and their mutual dependence.

From this matrix, the individual components can be read explicitly:

$$\mathrm{Var}(a) = \frac{S_{xx}}{\Delta}, \qquad\mathrm{Var}(b) = \frac{S}{\Delta}, \qquad\mathrm{Cov}(a,b) = -\frac{S_x}{\Delta} \tag{15.3.17}$$

These expressions show how the uncertainty in the intercept and slope depends on the weighted distribution of the data. The variances reflect how well each parameter is determined, while the covariance term captures the degree to which the two parameters are statistically coupled. In particular, the negative sign indicates that increases in one parameter are typically associated with decreases in the other, reflecting the fact that different combinations of slope and intercept can produce similar fits.

If the variances $\sigma_i^2$ are only known up to a scale factor, then the covariance expressions must be adjusted accordingly. In this case, the unknown noise level is estimated from the residuals, leading to:

$$\sigma^2 = \frac{\chi^2(a,b)}{N - 2} \tag{15.3.18}$$

The denominator $N - 2$ accounts for the degrees of freedom remaining after estimating the two parameters $a$ and $b$. The covariance matrix is then multiplied by this estimated variance, scaling the uncertainty to reflect the observed level of residual variation.

These quantities allow one to compute confidence intervals for the parameters and to assess the correlation between them. Importantly, they arise directly from the inverse of $A^\top W A$, rather than being ad hoc formulas (Pfister, 2024). This connection ensures that the statistical interpretation is fully consistent with the underlying least-squares formulation.

A particularly useful derived quantity is the variance of the predicted mean at a point $x_0$:

$$
\mathrm{Var}\bigl(y(x_0)\bigr)=

\mathrm{Var}(a)+

x_0^2 \mathrm{Var}(b)+

2 x_0 \mathrm{Cov}(a,b) \tag{15.3.19}
$$

This expression is obtained by propagating parameter uncertainty through the linear model and shows how uncertainty in the fitted parameters translates into uncertainty in predicted values. The dependence on $x_0$ reflects the fact that predictions farther from the center of the data are typically less certain.

This expression quantifies uncertainty in the fitted line itself. A full prediction interval would additionally include observational noise, reflecting not only uncertainty in the estimated mean but also the inherent variability in new observations.

## 15.3.5. When Least Squares is Not the Right Model

The straight-line fit also illustrates the limitations of least squares. While the method is mathematically convenient and widely applicable, its validity depends on the assumptions underlying its derivation. When these assumptions are not satisfied, the resulting estimates may be inefficient, biased, or overly sensitive to certain features of the data.

### Non-Gaussian or Impulsive Noise

If errors are not Gaussian, least squares is no longer the maximum likelihood estimator. The squared residual objective arises specifically from the Gaussian likelihood, and different noise distributions lead to different objective functions. For example, in wireless channel estimation with Bernoulli–Gaussian impulsive noise, the likelihood reflects the presence of occasional large disturbances, and the resulting estimator differs from least squares. In such cases, least squares may still serve as an approximation, but it no longer has a direct statistical justification and may fail to capture the true structure of the noise (Chaudhary et al., 2025).

### Outliers and Contamination

Even a small number of outliers can strongly influence least squares because the squared loss penalizes large residuals heavily. A single extreme observation can dominate the objective function and pull the fitted line away from the majority of the data. This sensitivity arises from the quadratic growth of the loss, which assigns disproportionately large weight to large deviations. Modern robust statistics addresses this issue by replacing the squared loss with alternative functions that grow more slowly, thereby reducing the influence of outliers while still fitting the bulk of the data (Loh, 2025; Atkinson et al., 2025).

A common practical approach is M-estimation, which leads to iteratively reweighted least squares. In this approach, the fitting process is expressed as a sequence of weighted least-squares problems in which the weights depend on the current residuals. Observations with large residuals receive reduced weight, and the procedure is repeated until convergence. This iterative structure allows the method to adapt to the data and achieve robustness without abandoning the computational framework of least squares.

These considerations reinforce a central message of the chapter: least squares is not universally optimal. It encodes specific assumptions about the noise and the data, and when those assumptions change, the appropriate estimator must change as well. Understanding both the strengths and the limitations of least squares is therefore essential for applying it correctly in practice.

## 15.3.6. Computational Considerations, Implementation, and Practical Applications

For straight-line fitting, two implementation strategies are commonly used, each reflecting a different balance between efficiency and generality. The first approach is based on explicit formulas derived from weighted sums. These formulas require only a single pass through the data, leading to computational cost proportional to $O(N)$ and minimal memory usage. Because all required quantities are aggregated into scalar sums, this approach is particularly efficient for large datasets when the model structure is simple and fixed.

The second approach is based on the design-matrix formulation. In this case, the problem is expressed in matrix form and solved using general least-squares algorithms. While this approach incurs additional computational overhead, it provides a unified framework that extends naturally to higher-dimensional models, multiple predictors, and more complex structures. For this reason, it is often preferred in applications where flexibility and extensibility are required.

Regardless of the implementation strategy, numerical robustness is a central concern. Several practical measures improve stability and accuracy in floating-point arithmetic. It is important to compute the weighted mean $\bar{x}_w$ first and to use centered sums, thereby reducing the magnitude of intermediate quantities and avoiding loss of significance. When the number of data points $N$ is large, compensated summation techniques help control the accumulation of rounding errors in the evaluation of sums. In addition, explicit matrix inversion should be avoided when extending to higher-dimensional problems, since it can amplify numerical errors. Instead, factorization-based methods provide more reliable alternatives.

For general least-squares problems, methods based on QR or SVD factorizations are preferred because they maintain numerical stability and handle ill-conditioning more effectively. Modern numerical libraries, such as *faer*, emphasize these stable approaches and recommend pivoted QR factorization in situations where rank deficiency may be present. These recommendations reflect a broader principle: numerical reliability depends not only on the mathematical formulation, but also on the way computations are organized and executed.

The straight-line model also serves as a gateway to a wide range of practical applications in which least squares plays a central role. In variational data assimilation, large-scale estimation problems in geophysics combine observational data with dynamical models. This leads to nonlinear least-squares formulations of the form:

$$\min_x |x - x_b|_{B^{-1}}^2 + |y - h(x)|_{R^{-1}}^2 \tag{15.3.20}$$

Here, the first term represents deviation from a prior or background state, while the second term measures mismatch with observations. These problems are typically solved using iterative methods that repeatedly form and solve linearized least-squares subproblems (Daužickaitė et al., 2025; Scott and Tůma, 2025).

In wireless channel estimation, linear regression models of the form:

$$y = Xh + n \tag{15.3.21}$$

are used to estimate unknown channel parameters. When the noise (n) is Gaussian, least squares provides an optimal estimator. However, in the presence of impulsive or non-Gaussian noise, the assumptions underlying least squares no longer hold, and alternative estimators based on different likelihood models are required (Chaudhary et al., 2025).

These examples highlight a key lesson: least squares is not just a mathematical technique, but a modeling choice tied to assumptions about data and noise. Its effectiveness depends on both the computational methods used to implement it and the validity of the assumptions that justify its use.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/KtyqP0IlJZZgIjQ3cUqL.5","tags":[]}

# 15.4. Straight-Line Data with Errors in Both Coordinates

In the previous section, the straight-line model was developed under the assumption that the independent variable is known exactly and that all uncertainty resides in the dependent variable. While this simplification is convenient and leads to a tractable formulation, it is often unrealistic in scientific and engineering applications. In many practical measurement systems, both coordinates are subject to uncertainty, and treating one variable as error-free can introduce systematic inaccuracies in the resulting model.

Situations in which both variables are noisy arise naturally across a wide range of disciplines. Examples include calibration of sensors, where both the reference and measured quantities contain errors; method-comparison studies in medicine, where two imperfect measurement techniques are compared; geodetic coordinate transformations, where positional data are affected by observational uncertainty; and registration problems in remote sensing, where spatial coordinates are derived from imperfect measurements. In all of these cases, uncertainty is inherent in both variables, and a model that accounts for this dual source of error is required for accurate inference.

When both variables are noisy, ordinary least squares becomes statistically inconsistent. The classical formulation assumes that deviations occur only in the dependent variable, so that residuals represent vertical distances from the fitted line. However, when the independent variable is also subject to error, these residuals no longer reflect the true discrepancy between the model and the underlying relationship. Errors in the independent variable propagate into the residuals, leading to a systematic bias in the estimated slope, typically resulting in attenuation of the slope magnitude.

This situation leads naturally to the framework of errors-in-variables (EIV) modeling, in which uncertainty is associated with both coordinates and must be accounted for explicitly in the fitting procedure. Instead of minimizing vertical deviations alone, EIV approaches seek formulations that balance errors in both directions, reflecting the true geometry of the measurement process. As a result, the problem requires a modified objective function and a different interpretation of residuals, setting the stage for the developments that follow.

## 15.4.1. Classical Formulation and the Modified Chi-Square

Consider the straight-line model:

$$y(x) = a + bx \tag{15.4.1}$$

As in the previous section, the parameters $a$ and $b$ represent the intercept and slope of the line. However, the presence of uncertainty in both coordinates requires a modification of the objective function used to estimate these parameters.

Suppose that each observed point $(x_i, y_i)$ has uncertainties $\sigma_{x_i}$ and $\sigma_{y_i}$ in both coordinates. In this setting, deviations from the model cannot be measured solely in the vertical direction, since both $x_i$ and $y_i$ contribute to the discrepancy. The appropriate chi-square objective therefore becomes:

$$\chi^2(a,b)= \sum_{i=1}^{N}\frac{(y_i - a - b x_i)^2}{\sigma_{y_i}^2 + b^2 \sigma_{x_i}^2} \tag{15.4.2}$$

Each term in this sum represents a squared residual normalized by its total variance, so that observations are weighted according to their combined uncertainty. This formulation ensures that both sources of error are taken into account in a consistent statistical manner.

The denominator reflects the combined uncertainty in the residual. Specifically,

$$\mathrm{Var}(y_i - a - b x_i)= \sigma_{y_i}^2 + b^2 \sigma_{x_i}^2 \tag{15.4.3}$$

This identity follows from the assumption that errors in $x_i$ and $y_i$ are independent, so their variances add when propagated through the linear model. The term $\sigma_{y_i}^2$ represents the direct contribution from the dependent variable, while the term $b^2 \sigma_{x_i}^2$ arises from the uncertainty in the independent variable after scaling by the slope.

This identity shows that uncertainty in the independent variable contributes to the residual variance, scaled by the slope $b$. As a consequence, errors in $x$ are amplified when the slope is large, since small uncertainties in $x_i$ can translate into significant deviations in the predicted value of $y$. This effect highlights the importance of accounting for uncertainty in both variables when estimating the model.

This formulation already reveals a key complication: the weights now depend on the parameter $b$, making the problem nonlinear even for a straight-line model. Unlike ordinary least squares, where the objective function is quadratic in the parameters, the presence of $b$ in the denominator introduces a coupling between the parameters and the weights. As a result, the estimation procedure requires iterative or nonlinear methods, rather than a single linear solve.

## 15.4.2. Geometric Formulation via Orthogonal Distance

A more general and geometrically transparent formulation is obtained by introducing latent “true” points. Instead of assuming that the observed data lie directly on the model, one interprets each measurement as a noisy observation of an underlying point that satisfies the model exactly. Let:

$$p_i =\begin{pmatrix}x_i \\y_i\end{pmatrix}= p_i^\star + \eta_i, \qquad\eta_i \sim \mathcal{N}(0,\Sigma_i)  \tag{15.4.4}$$

\
where $\Sigma_i \in \mathbb{R}^{2\times2}$ is the covariance matrix of the measurement noise. This formulation explicitly separates the true, unknown point $p_i^\star$, which lies on the line, from the observed point $p_i$, which is affected by noise.

To describe the line in a way that treats both coordinates symmetrically, it is convenient to use the normal form:

$$n(\theta)^\top p = c, \qquad n(\theta) =\begin{pmatrix}\cos\theta \\sin\theta\end{pmatrix} \tag{15.4.5}$$

Here, $n(\theta)$ is a unit normal vector to the line, and $c$ determines its position relative to the origin. This representation avoids privileging either coordinate direction and provides a natural geometric description of the line.

The signed orthogonal distance from a point to the line is then given by:

$$d_i(\theta,c) = n(\theta)^\top p_i - c \tag{15.4.6}$$

This quantity measures how far the observed point lies from the line along the normal direction. Unlike vertical residuals used in ordinary least squares, this distance reflects the shortest geometric distance to the line and therefore incorporates deviations in both coordinates.

Under Gaussian noise, the variance of this distance is:

$$\mathrm{Var}(d_i) = n(\theta)^\top \Sigma_i n(\theta) \tag{15.4.7}$$

This expression follows from projecting the covariance matrix onto the normal direction. It shows that the uncertainty in the orthogonal distance depends on both the measurement covariance and the orientation of the line. Different orientations weight the contributions of uncertainty in $x$ and $y$ differently.

The maximum likelihood objective becomes:

$$\chi^2(\theta,c)= \sum_{i=1}^{N}\frac{(n(\theta)^\top p_i - c)^2}{n(\theta)^\top \Sigma_i n(\theta)} \tag{15.4.8}$$

Each term represents a squared orthogonal distance normalized by its variance, ensuring that points are weighted according to their directional uncertainty. This formulation therefore provides a statistically consistent objective that accounts for both measurement error and geometric structure.

This formulation corresponds to weighted orthogonal-distance regression, which generalizes naturally to correlated uncertainties. Because the covariance matrices $\Sigma_i$ may include off-diagonal terms, the method can accommodate correlations between errors in the two coordinates without modification of the basic framework.

When $\Sigma_i = \mathrm{diag}(\sigma_{x_i}^2,\sigma_{y_i}^2)$ and $b=\tan\theta$, this expression reduces to the earlier formulation up to scaling. This connection shows that the orthogonal-distance formulation is not a separate method, but rather a more general representation that includes the previous approach as a special case.

A major advantage of this normal-form representation is numerical robustness. Near-vertical lines correspond to $|b|\to\infty$, which causes instability in the slope–intercept form due to the unbounded growth of the slope parameter. In contrast, the angle $\theta$ remains finite, and the objective becomes periodic in $\theta$, avoiding singular behavior. This property makes the normal formulation particularly well suited for numerical optimization, especially in cases where the orientation of the line is not known in advance.

## 15.4.3. Eliminating the Intercept and Reducing to One Dimension

A key simplification in the errors-in-variables formulation is that the intercept, or offset, can be eliminated analytically once the slope or angle is fixed. This reduction is important because it transforms a two-parameter optimization problem into a one-dimensional problem, which is significantly easier to analyze and solve numerically.

In the slope–intercept formulation, the dependence of the objective function on the intercept is linear once the slope $b$ is fixed. This makes it possible to determine the optimal intercept explicitly by minimizing the objective with respect to $a$. To express this result, define the weights:

$$w_i(b) = \frac{1}{\sigma_{y_i}^2 + b^2 \sigma_{x_i}^2} \tag{15.4.9}$$

These weights reflect the combined uncertainty in each observation and depend on the current value of the slope. As discussed earlier, the presence of $b$ in the denominator introduces nonlinearity into the problem.

Using these weights, the optimal intercept is given by:

$$a(b) =\frac{\sum_{i=1}^{N} w_i(b)(y_i - b x_i)}{\sum_{i=1}^{N} w_i(b)} \tag{15.4.10}$$

This expression shows that, for a fixed slope, the intercept is obtained as a weighted average of the adjusted observations $y_i - b x_i$. Substituting this expression back into the objective function eliminates $a$ entirely and yields a reduced problem in which the objective depends only on the single variable $b$.

This reduction transforms the original two-parameter optimization into the problem of minimizing $\chi^2(a(b), b)$ over the scalar parameter $b$. As a result, the search for the optimal line can be carried out along a single dimension, which simplifies both analysis and numerical implementation.

An analogous reduction can be performed in the normal-form representation. In this formulation, the offset $c$ appears linearly once the angle $\theta$ is fixed, allowing it to be eliminated in a similar manner. Define the weights:

$$w_i(\theta) = \frac{1}{n(\theta)^\top \Sigma_i n(\theta)} \tag{15.4.11}$$

These weights depend on the orientation of the line through the normal vector and incorporate the directional uncertainty of each observation.

The optimal offset is then:

$$c(\theta) =\frac{\sum_{i=1}^{N} w_i(\theta) n(\theta)^\top p_i}{\sum_{i=1}^{N} w_i(\theta)} \tag{15.4.12}$$

This expression shows that, for a fixed orientation, the offset is determined by a weighted average of the projected data points along the normal direction.

Again, substituting this expression into the objective reduces the problem to a one-dimensional minimization over $\theta$. This reduction is particularly advantageous in practice, since it allows the use of efficient one-dimensional optimization methods while retaining the full statistical structure of the model.

### Rust Implementation

Following the geometric formulation of straight-line fitting with errors in both coordinates in Section 15.4.2 and the analytical elimination of the offset in Section 15.4.3, Program 15.4.1 provides a practical implementation of weighted orthogonal-distance regression in normal form. The section shows that by representing the line through its unit normal vector (n(\\theta)) and offset $c$, and by expressing residuals as orthogonal distances, the fitting problem becomes symmetric in the two coordinates and statistically consistent with the underlying measurement model. Furthermore, by eliminating the offset analytically using Equation (15.4.12), the problem is reduced to a one-dimensional minimization over the orientation parameter $\theta$. This program implements that reduced formulation, evaluates the objective function based on Equations (15.4.6)–(15.4.8), and performs a robust one-dimensional search to determine the optimal line.

At the core of the implementation is the normal-form representation of the line introduced in Equation (15.4.5). The function `normal_vector` constructs the unit normal vector $n(\theta)$, which defines the orientation of the line. Using this vector, the function `projected_coordinate` evaluates the quantity $n(\theta)^\top p_i$, corresponding to the signed projection of each observation onto the normal direction, as defined in Equation (15.4.6).

The function `directional_variance` implements Equation (15.4.7), computing the variance of the orthogonal distance by projecting the covariance matrix $\Sigma_i$ onto the normal direction. This step ensures that each residual is weighted according to its uncertainty in the direction perpendicular to the fitted line, thereby incorporating both measurement error and geometric orientation.

The key computational step is performed in the function `reduced_objective`, which implements the analytical elimination of the offset described in Equation (15.4.12). For each candidate value of $\theta$, the weights defined in Equation (15.4.11) are evaluated, the optimal offset $c(\theta)$ is computed as a weighted average of projected coordinates, and the reduced chi-square objective in Equation (15.4.8) is then evaluated. This transforms the original two-parameter problem into a one-dimensional optimization over $\theta$.

The function `fit_normal_form_line` performs this one-dimensional optimization. It begins with a coarse search over the interval $[0,\pi)$ to identify a promising region, and then refines the solution using a golden-section search. This approach avoids the need for derivative information and provides a robust and efficient method for locating the minimum of the reduced objective.

Additional functions organize the output and provide geometric interpretation. The function `equivalent_slope_intercept` converts the normal-form parameters to slope–intercept form when possible, illustrating the relationship between the two representations. The diagnostic routines display orthogonal residuals, their variances, and standardized values, allowing detailed inspection of the fit.

The `main` function demonstrates the full workflow on a dataset with uncertainty in both coordinates, including correlated measurement errors. It constructs the observations, performs the fit, and reports both algebraic and geometric properties of the resulting line.

```rust
// Program 15.4.1. Weighted Orthogonal-Distance Regression in Normal Form
//
// Problem statement:
// Given observations p_i = (x_i, y_i) with 2 x 2 covariance matrices Sigma_i,
// fit a straight line in normal form
//
//     n(theta)^T p = c,     n(theta) = [cos(theta), sin(theta)]^T
//
// by minimizing the weighted orthogonal-distance objective
//
//     chi^2(theta, c) = sum_i (n(theta)^T p_i - c)^2 / (n(theta)^T Sigma_i n(theta)).
//
// For each fixed angle theta, the optimal offset c(theta) is computed analytically.
// The remaining one-dimensional reduced objective is then minimized over theta.
// This avoids slope singularities near vertical lines and provides a numerically
// robust implementation of the errors-in-variables formulation.

use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Cov2 {
    xx: f64,
    xy: f64,
    yy: f64,
}

#[derive(Clone, Copy, Debug)]
struct Observation {
    p: Point2,
    sigma: Cov2,
}

#[derive(Clone, Copy, Debug)]
struct NormalLine {
    theta: f64,
    c: f64,
}

#[derive(Clone, Copy, Debug)]
struct FitResult {
    line: NormalLine,
    chi2: f64,
    evaluations: usize,
}

#[derive(Clone, Copy, Debug)]
struct ReducedObjectiveValue {
    theta: f64,
    c: f64,
    chi2: f64,
}

fn validate_observations(data: &[Observation]) -> Result<(), String> {
    if data.len() < 2 {
        return Err("at least two observations are required".to_string());
    }

    for (i, obs) in data.iter().enumerate() {
        if !obs.p.x.is_finite()
            || !obs.p.y.is_finite()
            || !obs.sigma.xx.is_finite()
            || !obs.sigma.xy.is_finite()
            || !obs.sigma.yy.is_finite()
        {
            return Err(format!("observation {} contains a non-finite value", i));
        }

        if obs.sigma.xx <= 0.0 || obs.sigma.yy <= 0.0 {
            return Err(format!(
                "observation {} has nonpositive variance entries",
                i
            ));
        }

        let det = obs.sigma.xx * obs.sigma.yy - obs.sigma.xy * obs.sigma.xy;
        if det <= 0.0 {
            return Err(format!(
                "observation {} has a covariance matrix that is not positive definite",
                i
            ));
        }
    }

    Ok(())
}

fn normal_vector(theta: f64) -> Point2 {
    Point2 {
        x: theta.cos(),
        y: theta.sin(),
    }
}

fn directional_variance(n: Point2, sigma: Cov2) -> f64 {
    n.x * n.x * sigma.xx + 2.0 * n.x * n.y * sigma.xy + n.y * n.y * sigma.yy
}

fn projected_coordinate(n: Point2, p: Point2) -> f64 {
    n.x * p.x + n.y * p.y
}

fn reduced_objective(theta: f64, data: &[Observation]) -> Result<ReducedObjectiveValue, String> {
    let n = normal_vector(theta);

    let mut weight_sum = 0.0;
    let mut weighted_projection_sum = 0.0;

    for obs in data {
        let var_d = directional_variance(n, obs.sigma);
        if var_d <= 0.0 || !var_d.is_finite() {
            return Err("encountered nonpositive directional variance".to_string());
        }

        let w = 1.0 / var_d;
        let proj = projected_coordinate(n, obs.p);

        weight_sum += w;
        weighted_projection_sum += w * proj;
    }

    if weight_sum <= 0.0 {
        return Err("sum of weights is nonpositive".to_string());
    }

    let c = weighted_projection_sum / weight_sum;

    let mut chi2 = 0.0;
    for obs in data {
        let var_d = directional_variance(n, obs.sigma);
        let residual = projected_coordinate(n, obs.p) - c;
        chi2 += residual * residual / var_d;
    }

    Ok(ReducedObjectiveValue { theta, c, chi2 })
}

fn normalize_theta(theta: f64) -> f64 {
    let mut t = theta % PI;
    if t < 0.0 {
        t += PI;
    }
    t
}

fn coarse_search(data: &[Observation], num_samples: usize) -> Result<ReducedObjectiveValue, String> {
    let samples = num_samples.max(16);
    let mut best = reduced_objective(0.0, data)?;

    for k in 1..samples {
        let theta = PI * (k as f64) / (samples as f64);
        let value = reduced_objective(theta, data)?;
        if value.chi2 < best.chi2 {
            best = value;
        }
    }

    Ok(best)
}

fn golden_section_search(
    data: &[Observation],
    left: f64,
    right: f64,
    tol: f64,
    max_iter: usize,
) -> Result<(ReducedObjectiveValue, usize), String> {
    let phi = 0.5 * (1.0 + 5.0_f64.sqrt());
    let invphi = 1.0 / phi;

    let mut a = left;
    let mut b = right;

    let mut c = b - (b - a) * invphi;
    let mut d = a + (b - a) * invphi;

    let mut fc = reduced_objective(c, data)?;
    let mut fd = reduced_objective(d, data)?;
    let mut evaluations = 2;

    for _ in 0..max_iter {
        if (b - a).abs() < tol {
            break;
        }

        if fc.chi2 < fd.chi2 {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) * invphi;
            fc = reduced_objective(c, data)?;
            evaluations += 1;
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) * invphi;
            fd = reduced_objective(d, data)?;
            evaluations += 1;
        }
    }

    let best = if fc.chi2 < fd.chi2 { fc } else { fd };
    Ok((best, evaluations))
}

fn fit_normal_form_line(data: &[Observation]) -> Result<FitResult, String> {
    validate_observations(data)?;

    let coarse = coarse_search(data, 720)?;
    let bracket_half_width = 0.10_f64;
    let left = (coarse.theta - bracket_half_width).max(0.0);
    let right = (coarse.theta + bracket_half_width).min(PI);

    let (refined, evaluations) = golden_section_search(data, left, right, 1.0e-12, 200)?;

    Ok(FitResult {
        line: NormalLine {
            theta: normalize_theta(refined.theta),
            c: refined.c,
        },
        chi2: refined.chi2,
        evaluations,
    })
}

fn line_direction(theta: f64) -> Point2 {
    Point2 {
        x: -theta.sin(),
        y: theta.cos(),
    }
}

fn point_on_line(c: f64, theta: f64) -> Point2 {
    let n = normal_vector(theta);
    Point2 {
        x: c * n.x,
        y: c * n.y,
    }
}

fn orthogonal_residual(theta: f64, c: f64, obs: Observation) -> f64 {
    let n = normal_vector(theta);
    projected_coordinate(n, obs.p) - c
}

fn equivalent_slope_intercept(line: NormalLine) -> Option<(f64, f64)> {
    let n = normal_vector(line.theta);

    if n.y.abs() < 1.0e-12 {
        return None;
    }

    // From cos(theta) x + sin(theta) y = c
    // => y = c / sin(theta) - cot(theta) x
    let slope = -n.x / n.y;
    let intercept = line.c / n.y;
    Some((intercept, slope))
}

fn print_input_data(data: &[Observation]) {
    println!("Input Observations");
    println!("==================");
    println!(
        "{:>4} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "i", "x_i", "y_i", "Sigma_xx", "Sigma_xy", "Sigma_yy"
    );
    println!("{}", "-".repeat(76));

    for (i, obs) in data.iter().enumerate() {
        println!(
            "{:>4} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            i, obs.p.x, obs.p.y, obs.sigma.xx, obs.sigma.xy, obs.sigma.yy
        );
    }
    println!();
}

fn print_fit_summary(result: FitResult) {
    println!("Normal-Form Fit");
    println!("===============");
    println!("theta       = {:>.12}", result.line.theta);
    println!("c           = {:>.12}", result.line.c);
    println!("chi^2_min   = {:>.12}", result.chi2);
    println!("evaluations = {}", result.evaluations);

    match equivalent_slope_intercept(result.line) {
        Some((a, b)) => {
            println!("equivalent intercept a = {:>.12}", a);
            println!("equivalent slope b     = {:>.12}", b);
        }
        None => {
            println!("equivalent slope-intercept form is numerically singular");
            println!("the fitted line is effectively vertical");
        }
    }
    println!();
}

fn print_geometry(result: FitResult) {
    let n = normal_vector(result.line.theta);
    let d = line_direction(result.line.theta);
    let p0 = point_on_line(result.line.c, result.line.theta);

    println!("Geometric Description");
    println!("=====================");
    println!("unit normal n(theta)   = ({:>.10}, {:>.10})", n.x, n.y);
    println!("line direction vector  = ({:>.10}, {:>.10})", d.x, d.y);
    println!("reference point on line= ({:>.10}, {:>.10})", p0.x, p0.y);
    println!();
}

fn print_pointwise_diagnostics(data: &[Observation], result: FitResult) {
    println!("Pointwise Orthogonal Diagnostics");
    println!("===============================");
    println!(
        "{:>4} {:>12} {:>12} {:>14} {:>14} {:>14}",
        "i", "x_i", "y_i", "d_i", "Var(d_i)", "d_i/sqrt(Var)"
    );
    println!("{}", "-".repeat(86));

    let n = normal_vector(result.line.theta);

    for (i, obs) in data.iter().enumerate() {
        let d_i = orthogonal_residual(result.line.theta, result.line.c, *obs);
        let var_d = directional_variance(n, obs.sigma);
        let std_res = d_i / var_d.sqrt();

        println!(
            "{:>4} {:>12.6} {:>12.6} {:>14.8} {:>14.8} {:>14.8}",
            i, obs.p.x, obs.p.y, d_i, var_d, std_res
        );
    }
    println!();
}

fn main() {
    // Illustrative dataset with uncertainty in both coordinates.
    // Small correlations are included to demonstrate the use of full 2 x 2 covariances.
    let data = vec![
        Observation {
            p: Point2 { x: 0.2, y: 1.1 },
            sigma: Cov2 { xx: 0.030, xy: 0.004, yy: 0.040 },
        },
        Observation {
            p: Point2 { x: 0.9, y: 1.9 },
            sigma: Cov2 { xx: 0.020, xy: 0.003, yy: 0.030 },
        },
        Observation {
            p: Point2 { x: 2.1, y: 3.0 },
            sigma: Cov2 { xx: 0.025, xy: 0.002, yy: 0.035 },
        },
        Observation {
            p: Point2 { x: 3.0, y: 4.2 },
            sigma: Cov2 { xx: 0.022, xy: 0.003, yy: 0.028 },
        },
        Observation {
            p: Point2 { x: 4.2, y: 5.0 },
            sigma: Cov2 { xx: 0.028, xy: 0.004, yy: 0.036 },
        },
        Observation {
            p: Point2 { x: 5.1, y: 6.2 },
            sigma: Cov2 { xx: 0.035, xy: 0.005, yy: 0.045 },
        },
    ];

    println!("Weighted Orthogonal-Distance Regression");
    println!("=======================================");
    println!("This program fits a line in normal form by minimizing");
    println!("the reduced orthogonal-distance objective after");
    println!("eliminating the offset analytically.");
    println!();

    print_input_data(&data);

    let fit = match fit_normal_form_line(&data) {
        Ok(result) => result,
        Err(message) => {
            eprintln!("fit failed: {}", message);
            std::process::exit(1);
        }
    };

    print_fit_summary(fit);
    print_geometry(fit);
    print_pointwise_diagnostics(&data, fit);

    println!("Interpretation");
    println!("==============");
    println!("The normal-form parameterization treats x and y symmetrically");
    println!("and remains well behaved even when the fitted line is nearly");
    println!("vertical. Eliminating c for each trial angle reduces the");
    println!("problem to a one-dimensional search over theta.");
}
```

Program 15.4.1 demonstrates how the orthogonal-distance formulation of the errors-in-variables problem leads naturally to a practical and numerically robust fitting algorithm. By representing the line in normal form and eliminating the offset analytically, the implementation reduces the problem to a one-dimensional optimization over the orientation parameter. This reduction simplifies the computation while preserving the full statistical structure of the model.

The comparison between normal-form parameters and the equivalent slope–intercept representation highlights an important numerical advantage. While the slope parameter in the traditional formulation can become unbounded near vertical lines, the angle $\theta$ remains finite and well behaved. This makes the normal formulation particularly suitable for numerical optimization, especially in situations where the orientation of the line is not known a priori.

More broadly, the program illustrates a central theme of the chapter: when measurement uncertainty affects all variables, the geometry of the problem must be treated symmetrically, and the formulation of the objective function must reflect that symmetry. The orthogonal-distance approach provides a consistent and flexible framework for achieving this, and it extends naturally to more complex models and higher-dimensional settings.

## 15.4.4. Numerical Strategy and Complexity

Once the intercept or offset has been eliminated analytically, the fitting problem is reduced to the minimization of a function of a single scalar variable, either the slope $b$ in the slope–intercept formulation or the angle $\theta$ in the normal-form formulation. This reduction has major practical importance, because it transforms the original parameter-estimation problem into a one-dimensional search problem. Instead of attempting to optimize simultaneously over two coupled variables, one needs only to evaluate the reduced objective repeatedly at candidate values of a single parameter.

The baseline recommends solving this one-dimensional problem using robust bracketing methods, such as Brent’s algorithm. The reason for this recommendation is that bracketing methods are designed to be reliable even when derivative information is unavailable, expensive to compute, or numerically inconvenient. In the present setting, the reduced objective can be evaluated directly for any candidate value of $b$ or $\theta$, so a derivative-free method is natural. A bracketing strategy also avoids some of the sensitivity that can arise in nonlinear optimization methods that rely on local derivative information or on simultaneous updates of multiple parameters.

At each candidate value of $b$ or $\theta$, the intercept is recomputed and the objective is then evaluated. This means that the algorithm proceeds in a repeated cycle. First, a trial value of the slope or angle is chosen by the one-dimensional search method. Next, the corresponding optimal intercept or offset is obtained from the explicit formulas derived in the previous subsection. Finally, that value is substituted into the reduced chi-square objective, producing a scalar quantity that measures the quality of the current fit. The optimization method then uses these scalar objective values to decide which new candidate parameter value to test next.

This structure is important because it separates the nonlinear and linear parts of the problem. The nonlinear dependence is confined to the single variable $b$ or $\theta$, while the linear parameter $a$ or $c$ is handled exactly at each step. As a consequence, the algorithm does not waste effort searching numerically over a parameter that can already be determined in closed form. This makes the computation both cleaner and more reliable.

The computational cost of this strategy is straightforward to characterize. Evaluating $\chi^2$ for one parameter value costs $O(N)$. This cost arises because, for a fixed candidate value of $b$ or $\theta$, one must pass through all $N$ data points to compute the corresponding weights, form the sums needed for the optimal intercept or offset, and then evaluate the objective function. Each observation contributes a constant amount of work, so the total work grows linearly with the number of data points.

The memory overhead beyond the data is $O(1)$. This is because the evaluation does not require storage of large auxiliary arrays or matrices. Apart from the observed data themselves, only a small fixed number of scalar accumulators are needed to compute the weighted sums, the intercept or offset, and the value of the objective. Thus, the amount of additional memory required does not grow with $N$. This is a particularly attractive feature in large-data settings, since it allows the method to remain efficient in both time and storage.

If the line search requires $I$ evaluations, then the total computational cost is $O(IN)$. This follows directly from the fact that each evaluation costs $O(N)$, and the objective must be evaluated repeatedly during the one-dimensional minimization. The parameter (I) depends on the behavior of the search method and on the shape of the reduced objective, but the overall scaling remains linear in the data size for each trial value. In this sense, the method is computationally simple and predictable.

This complexity estimate also clarifies why the reduced formulation is attractive. The cost of the algorithm is controlled by two transparent factors: the size of the dataset and the number of objective evaluations required by the one-dimensional search. There is no need to form or store large Hessian matrices, nor is there any need to solve a full two-parameter nonlinear optimization problem at every step. The problem is handled through repeated inexpensive passes over the data.

This approach is typically more stable and simpler than solving a two-dimensional nonlinear optimization problem in $(a,b)$. The simplification comes from the fact that one parameter has already been eliminated analytically, so the remaining numerical search is lower-dimensional and easier to control. The stability comes from avoiding simultaneous iterative updates of two coupled parameters, which can introduce additional sensitivity to starting values, scaling, and local curvature. By reducing the problem to one dimension and recomputing the intercept exactly at each step, the method achieves a balance of simplicity, efficiency, and numerical robustness.

From a practical point of view, this subsection reinforces a recurring theme of the chapter: whenever part of a problem can be solved analytically, it is often advantageous to do so before applying numerical optimization. In the present case, eliminating the intercept reduces complexity, improves robustness, and leads to an implementation that is both conceptually transparent and computationally efficient.

### Rust Implementation

Following the discussion in Section 15.4.4 on the numerical strategy for solving the errors-in-variables problem, Program 15.4.2 provides a practical implementation of reduced one-dimensional optimization in the slope–intercept formulation. The section emphasizes that, after eliminating the intercept analytically using Equation (15.4.10), the fitting problem reduces to minimizing a scalar objective as a function of a single variable, the slope $b$. This program implements that strategy by repeatedly evaluating the reduced chi-square objective $\chi^2(a(b), b)$ and applying a derivative-free bracketing method to locate its minimum. By separating the nonlinear search from the linear update of the intercept, the implementation reflects the computational structure described in the section and demonstrates how analytical elimination leads to a simpler and more efficient numerical algorithm.

At the core of the implementation is the function `reduced_objective`, which evaluates the reduced chi-square objective described in Equation (15.4.9). For a given trial slope $b$, the function computes the effective variance for each observation and accumulates the corresponding weights. Using these weights, it evaluates the optimal intercept $a(b)$ according to Equation (15.4.10). This eliminates the linear parameter analytically, ensuring that the intercept is always optimal for the current slope.

Once the intercept has been determined, the function proceeds to compute the value of the reduced objective $\chi^2(a(b), b)$. This involves a second pass through the data, where residuals are evaluated and scaled by their corresponding effective variances. Each evaluation of the objective therefore requires $O(N)$ work, as described in Section 15.4.4, since it involves processing all observations sequentially.

The function `fit_errors_in_variables_line` implements the one-dimensional optimization strategy. It begins with a coarse search over a specified interval to identify a region where the objective function is small. This is followed by a refinement step using a golden-section search, a derivative-free bracketing method that is robust and well suited to problems where derivative information is unavailable or inconvenient to compute. This approach directly reflects the recommendation in Section 15.4.4 to use reliable bracketing methods for one-dimensional minimization.

Supporting functions such as `predict`, `effective_variance`, and the diagnostic routines organize the computation and provide detailed output. The pointwise diagnostics display predicted values, residuals, and standardized residuals, allowing verification of the fit and illustrating how each data point contributes to the objective function.

The `main` function demonstrates the complete workflow. It defines a dataset with uncertainties in both coordinates, specifies a search interval for the slope, and invokes the fitting routine. It then reports the fitted parameters, diagnostic information, and a summary of computational complexity, illustrating the linear scaling with respect to the number of data points and the number of objective evaluations.

```rust
// Program 15.4.2. Errors-in-Variables Straight-Line Fit via Reduced One-Dimensional Optimization
//
// Problem statement:
// Given observations (x_i, y_i) with uncertainties sigma_x_i and sigma_y_i
// in both coordinates, fit the straight-line model
//
//     y = a + b x
//
// by minimizing the modified chi-square objective
//
//     chi^2(a,b) = sum_i (y_i - a - b x_i)^2 / (sigma_y_i^2 + b^2 sigma_x_i^2).
//
// For each fixed slope b, the intercept a(b) is computed analytically.
// The remaining reduced objective chi^2(a(b), b) is then minimized over b
// using a robust derivative-free one-dimensional search.
//
// This program demonstrates the computational strategy described in
// Section 15.4.4: separate the nonlinear search over b from the exact
// linear update of a, so that each objective evaluation costs O(N)
// time and O(1) memory beyond the data.

#[derive(Clone, Copy, Debug)]
struct Observation {
    x: f64,
    y: f64,
    sigma_x: f64,
    sigma_y: f64,
}

#[derive(Clone, Copy, Debug)]
struct ReducedEvaluation {
    slope: f64,
    intercept: f64,
    chi2: f64,    
}

#[derive(Clone, Copy, Debug)]
struct FitResult {
    intercept: f64,
    slope: f64,
    chi2: f64,
    evaluations: usize,
}

fn validate_data(data: &[Observation]) -> Result<(), String> {
    if data.len() < 2 {
        return Err("at least two observations are required".to_string());
    }

    for (i, obs) in data.iter().enumerate() {
        if !obs.x.is_finite()
            || !obs.y.is_finite()
            || !obs.sigma_x.is_finite()
            || !obs.sigma_y.is_finite()
        {
            return Err(format!("observation {} contains a non-finite value", i));
        }
        if obs.sigma_x <= 0.0 || obs.sigma_y <= 0.0 {
            return Err(format!(
                "observation {} has nonpositive uncertainty",
                i
            ));
        }
    }

    Ok(())
}

fn reduced_objective(slope: f64, data: &[Observation]) -> Result<ReducedEvaluation, String> {
    let mut weight_sum = 0.0;
    let mut weighted_adjusted_sum = 0.0;

    for obs in data {
        let variance = obs.sigma_y * obs.sigma_y + slope * slope * obs.sigma_x * obs.sigma_x;
        if variance <= 0.0 || !variance.is_finite() {
            return Err("encountered invalid effective variance".to_string());
        }

        let w = 1.0 / variance;
        weight_sum += w;
        weighted_adjusted_sum += w * (obs.y - slope * obs.x);
    }

    if weight_sum <= 0.0 || !weight_sum.is_finite() {
        return Err("sum of weights is invalid".to_string());
    }

    let intercept = weighted_adjusted_sum / weight_sum;

    let mut chi2 = 0.0;
    for obs in data {
        let variance = obs.sigma_y * obs.sigma_y + slope * slope * obs.sigma_x * obs.sigma_x;
        let residual = obs.y - intercept - slope * obs.x;
        chi2 += residual * residual / variance;
    }

    Ok(ReducedEvaluation {
        slope,
        intercept,
        chi2,        
    })
}

fn coarse_search(
    data: &[Observation],
    slope_min: f64,
    slope_max: f64,
    num_samples: usize,
) -> Result<ReducedEvaluation, String> {
    let samples = num_samples.max(16);
    let mut best = reduced_objective(slope_min, data)?;

    for k in 1..=samples {
        let t = (k as f64) / (samples as f64);
        let slope = slope_min + t * (slope_max - slope_min);
        let value = reduced_objective(slope, data)?;
        if value.chi2 < best.chi2 {
            best = value;
        }
    }

    Ok(best)
}

fn golden_section_search(
    data: &[Observation],
    left: f64,
    right: f64,
    tol: f64,
    max_iter: usize,
) -> Result<(ReducedEvaluation, usize), String> {
    let phi = 0.5 * (1.0 + 5.0_f64.sqrt());
    let invphi = 1.0 / phi;

    let mut a = left;
    let mut b = right;

    let mut c = b - (b - a) * invphi;
    let mut d = a + (b - a) * invphi;

    let mut fc = reduced_objective(c, data)?;
    let mut fd = reduced_objective(d, data)?;
    let mut evaluations = 2;

    for _ in 0..max_iter {
        if (b - a).abs() < tol {
            break;
        }

        if fc.chi2 < fd.chi2 {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) * invphi;
            fc = reduced_objective(c, data)?;
            evaluations += 1;
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) * invphi;
            fd = reduced_objective(d, data)?;
            evaluations += 1;
        }
    }

    let best = if fc.chi2 < fd.chi2 { fc } else { fd };
    Ok((best, evaluations))
}

fn fit_errors_in_variables_line(
    data: &[Observation],
    slope_min: f64,
    slope_max: f64,
) -> Result<FitResult, String> {
    validate_data(data)?;

    if !slope_min.is_finite() || !slope_max.is_finite() || slope_min >= slope_max {
        return Err("invalid slope search interval".to_string());
    }

    let coarse = coarse_search(data, slope_min, slope_max, 1000)?;
    let width = slope_max - slope_min;
    let half_bracket = 0.05 * width;

    let left = (coarse.slope - half_bracket).max(slope_min);
    let right = (coarse.slope + half_bracket).min(slope_max);

    let (best, evaluations) = golden_section_search(data, left, right, 1.0e-12, 200)?;

    Ok(FitResult {
        intercept: best.intercept,
        slope: best.slope,
        chi2: best.chi2,
        evaluations,
    })
}

fn predict(x: f64, intercept: f64, slope: f64) -> f64 {
    intercept + slope * x
}

fn effective_variance(obs: Observation, slope: f64) -> f64 {
    obs.sigma_y * obs.sigma_y + slope * slope * obs.sigma_x * obs.sigma_x
}

fn print_input_data(data: &[Observation]) {
    println!("Input Observations");
    println!("==================");
    println!(
        "{:>4} {:>12} {:>12} {:>12} {:>12}",
        "i", "x_i", "y_i", "sigma_x_i", "sigma_y_i"
    );
    println!("{}", "-".repeat(60));

    for (i, obs) in data.iter().enumerate() {
        println!(
            "{:>4} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            i, obs.x, obs.y, obs.sigma_x, obs.sigma_y
        );
    }
    println!();
}

fn print_fit_summary(result: FitResult) {
    println!("Reduced One-Dimensional Fit");
    println!("===========================");
    println!("intercept a   = {:>.12}", result.intercept);
    println!("slope b       = {:>.12}", result.slope);
    println!("chi^2_min     = {:>.12}", result.chi2);
    println!("evaluations   = {}", result.evaluations);
    println!();
}

fn print_pointwise_diagnostics(data: &[Observation], result: FitResult) {
    println!("Pointwise Diagnostics");
    println!("=====================");
    println!(
        "{:>4} {:>12} {:>12} {:>14} {:>14} {:>14}",
        "i", "x_i", "y_i", "y_hat", "r_i", "r_i/sqrt(var)"
    );
    println!("{}", "-".repeat(86));

    for (i, obs) in data.iter().enumerate() {
        let y_hat = predict(obs.x, result.intercept, result.slope);
        let residual = obs.y - y_hat;
        let variance = effective_variance(*obs, result.slope);
        let standardized = residual / variance.sqrt();

        println!(
            "{:>4} {:>12.6} {:>12.6} {:>14.8} {:>14.8} {:>14.8}",
            i, obs.x, obs.y, y_hat, residual, standardized
        );
    }
    println!();
}

fn print_complexity_note(data_len: usize, evaluations: usize) {
    println!("Complexity Summary");
    println!("==================");
    println!("Data size N                    = {}", data_len);
    println!("Objective evaluations I        = {}", evaluations);
    println!("Work per evaluation            = O(N)");
    println!("Total work                     = O(IN)");
    println!("Memory beyond the data         = O(1)");
    println!();
}

fn main() {
    let data = vec![
        Observation { x: 0.20, y: 1.10, sigma_x: 0.08, sigma_y: 0.18 },
        Observation { x: 0.95, y: 1.95, sigma_x: 0.07, sigma_y: 0.15 },
        Observation { x: 1.85, y: 2.90, sigma_x: 0.09, sigma_y: 0.17 },
        Observation { x: 2.90, y: 4.00, sigma_x: 0.10, sigma_y: 0.20 },
        Observation { x: 4.10, y: 5.25, sigma_x: 0.11, sigma_y: 0.19 },
        Observation { x: 5.05, y: 6.10, sigma_x: 0.12, sigma_y: 0.22 },
    ];

    // Search interval for the slope. In practice, this can be informed by prior
    // knowledge, by an ordinary least-squares estimate, or by a broader bracket.
    let slope_min = -5.0;
    let slope_max = 5.0;

    println!("Errors-in-Variables Fit via Reduced One-Dimensional Search");
    println!("==========================================================");
    println!("This program eliminates the intercept analytically for each");
    println!("trial slope and then minimizes the reduced chi-square");
    println!("objective using a derivative-free bracketing strategy.");
    println!();

    print_input_data(&data);

    let result = match fit_errors_in_variables_line(&data, slope_min, slope_max) {
        Ok(fit) => fit,
        Err(message) => {
            eprintln!("fit failed: {}", message);
            std::process::exit(1);
        }
    };

    print_fit_summary(result);
    print_pointwise_diagnostics(&data, result);
    print_complexity_note(data.len(), result.evaluations);

    println!("Interpretation");
    println!("==============");
    println!("For each candidate slope b, the intercept a(b) is computed");
    println!("exactly from weighted averages, so the numerical search is");
    println!("confined to one dimension. This makes the implementation");
    println!("simpler, cheaper, and more stable than a full two-parameter");
    println!("nonlinear optimization.");
}
```

Program 15.4.2 demonstrates the effectiveness of reducing a two-parameter estimation problem to a one-dimensional optimization task by eliminating the linear parameter analytically. This approach reflects the central idea of Section 15.4.4: whenever part of a problem can be solved exactly, doing so simplifies the remaining numerical work and improves robustness.

The implementation also illustrates the computational structure of the method. Each evaluation of the reduced objective requires a single pass through the data, leading to a cost proportional to $N$. The total cost is therefore proportional to the number of evaluations performed by the search algorithm, resulting in an overall complexity of $O(IN)$. At the same time, the memory usage remains minimal, requiring only a small number of scalar accumulators beyond the input data.

More broadly, this example highlights the importance of combining analytical insight with numerical methods. By isolating the nonlinear component of the problem and treating it with a robust one-dimensional search, the algorithm achieves a balance of simplicity, efficiency, and numerical stability. This principle extends to more general optimization problems and serves as a guiding strategy in the design of reliable numerical algorithms.

## 15.4.5. Uncertainty Estimation via $\Delta\chi^2$

Because the objective function in the errors-in-variables formulation is nonlinear in the parameters, standard quadratic approximations may be unreliable for estimating uncertainties. In the ordinary least-squares setting, uncertainty is often approximated by expanding the objective function locally around its minimum and using the curvature of this approximation to define confidence regions. However, when the objective depends nonlinearly on the parameters, such local approximations may fail to capture the true shape of the objective, especially if the curvature varies significantly or if the parameter dependence is asymmetric.

Instead, parameter uncertainties are obtained by computing regions where,

$$\Delta\chi^2 = \chi^2 - \chi^2_{\min} = 1 \tag{15.4.13}$$

This condition defines a level set of the objective function corresponding to a specified increase above the minimum value. For a single parameter, this criterion provides an estimate of the confidence interval that reflects the actual geometry of the objective function rather than relying on a local approximation.

In practice, this is done through a structured procedure. One begins by fixing one parameter at a trial value, while allowing the other parameter to vary freely. For each fixed value, the objective function is minimized with respect to the remaining parameter, ensuring that the best possible fit is obtained under the constraint. The resulting value of $\chi^2$ is then compared with the minimum value $\chi^2_{\min}$. By repeating this process for different trial values, one can determine where the condition $\Delta\chi^2 = 1$ is satisfied.

To locate these boundary points accurately, root-finding methods are typically used. The problem is reduced to finding the parameter values at which the difference $\chi^2 - \chi^2_{\min}$ equals one. This approach allows the confidence interval to be determined with controlled numerical accuracy and without relying on assumptions about the local curvature of the objective.

This procedure provides more reliable confidence intervals than local quadratic approximations because it directly reflects the true shape of the objective function. In particular, it can capture asymmetries or nonlinear effects that would be missed by a purely local analysis. As a result, it is well suited for problems in which the dependence on parameters is inherently nonlinear.

An important modeling lesson emerges here: uncertainty estimates are meaningful only if the model adequately describes the data. If the fit is poor, the computed confidence intervals may not reflect the true variability of the system. In such cases, the baseline suggests rescaling measurement uncertainties to achieve statistical consistency, for example by adjusting the overall scale so that the minimized chi-square per degree of freedom is close to one. However, this procedure should be used cautiously and justified explicitly, since it modifies the interpretation of the underlying noise model.

## 15.4.6. Modern Perspective: Errors-in-Variables and Total Least Squares

When both coordinates are noisy, the problem becomes an instance of errors-in-variables modeling. In this framework, uncertainty is not confined to the observations on the right-hand side of a regression model, but also affects the variables that define the model itself. This fundamentally changes the structure of the estimation problem and requires a formulation that accounts for errors in all measured quantities.

In matrix form, the straight-line model can be written as:

$$y_i \approx a + b x_i \;\Longleftrightarrow \; A_i \beta \approx b_i, \quad A_i = [1 \quad x_i], \quad\beta =\begin{pmatrix}a \\b\end{pmatrix} \tag{15.4.14}$$

This representation highlights the linear structure of the model, where each observation contributes a row to the design matrix. In the classical least-squares setting, the matrix $A$ is assumed to be known exactly, and only the right-hand side is affected by noise.

However, since $x_i$ is noisy, the matrix $A$ itself is uncertain. Each row of $A$ contains measured quantities, so errors in $x_i$ translate directly into perturbations of the design matrix. As a consequence, the classical least-squares formulation, which assumes a fixed matrix, is no longer appropriate. The estimation problem must instead account for perturbations in both the matrix and the observation vector.

This leads to the formulation of total least squares (TLS):

$$\min_{\Delta A,\Delta b}\bigl\|[\Delta A \quad \Delta b]\bigr\|_F\quad \text{s.t.} \quad(A+\Delta A)\beta = b + \Delta b \tag{15.4.15}$$

In this formulation, one seeks the smallest perturbations $\Delta A$ and $\Delta b$ that make the model exactly consistent. The Frobenius norm measures the total size of these perturbations, and the constraint enforces that the corrected data lie exactly on the fitted model. Unlike ordinary least squares, which minimizes residuals in the observation vector alone, TLS balances corrections in both the matrix and the data.

Weighted total least squares (WTLS) extends this formulation to heteroscedastic and correlated errors, aligning closely with the geometric formulation introduced earlier. In this setting, different components of $\Delta A$ and $\Delta b$ are weighted according to their respective uncertainties, and covariance structures can be incorporated directly into the objective. This makes WTLS particularly suitable for applications where measurement errors vary across observations or exhibit correlations.

Recent developments show that WTLS is an active area of research, reflecting both theoretical and computational advances:

- Efficient multivariate WTLS methods for large datasets (Gholinejad and Amiri-Simkooei, 2023), which address the challenges posed by high-dimensional problems and enable scalable implementations.
- Constrained EIV formulations using Gauss–Helmert linearization (Jin et al., 2023), which incorporate additional structure or physical constraints into the estimation problem while maintaining consistency with the errors-in-variables framework.
- Backward error analysis and randomized TLS algorithms (Shan and Wei, 2025), which provide insight into the numerical properties of TLS solutions and develop efficient algorithms for large-scale problems.

These advances demonstrate that errors-in-variables modeling is not a niche extension, but a central tool in modern applied mathematics and engineering. It provides a principled way to account for uncertainty in all measured quantities and connects geometric, statistical, and numerical perspectives within a unified framework.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/gidzuzjAJtHLL0Qmf9xO.6","tags":[]}

# 15.5. General Linear Least Squares

The straight-line model is the simplest nontrivial example of a much broader and extremely important class of models: those that are linear in their parameters. In such models, the unknown quantities enter as coefficients in a linear combination of known functions, even if the dependence on the independent variable itself is nonlinear. This distinction is essential, since linearity in the parameters ensures that the resulting estimation problem retains a structure that can be analyzed and solved using linear algebra.

In many applications, the dependence on the independent variable may be nonlinear, but the unknown coefficients enter linearly. For example, models built from polynomial bases, trigonometric expansions, or other prescribed functions all share this property. As a result, a wide variety of seemingly different modeling problems can be brought into a common framework in which the goal is to determine a set of coefficients that best combine these functions to match observed data.

This structure allows the estimation problem to be formulated as a linear least-squares problem, in which the discrepancy between the model and the data is measured and minimized with respect to the parameters. The key advantage of this formulation is that it leads to problems that can be solved efficiently using well-developed numerical methods, while also admitting a precise theoretical analysis. Issues such as existence and uniqueness of solutions, sensitivity to perturbations, and statistical interpretation can all be expressed in terms of properties of matrices and vector spaces.

This section generalizes the ideas developed in earlier sections to arbitrary linear-in-parameter models, introducing the associated matrix formulation in a systematic way. The transition from scalar formulas to matrix notation not only simplifies the presentation but also reveals the underlying geometric structure of the problem, in which solutions are interpreted as projections onto subspaces defined by the model.

In addition to the theoretical formulation, the section examines the computational implications of linear least squares. Efficient algorithms, numerical stability, and the handling of large or ill-conditioned systems become central considerations. These aspects are essential in practice, where the size and complexity of problems often exceed what can be handled by direct or naive methods.

Finally, the section highlights why linear least squares appears ubiquitously across scientific computing. It arises not only in direct data fitting, but also as a fundamental building block in nonlinear optimization, where linearized subproblems are solved repeatedly; in inverse problems, where unknown causes are inferred from observed effects; and in data assimilation, where observational data are combined with predictive models. The general linear least-squares framework therefore provides a unifying foundation for a wide range of computational and statistical methods.

## 15.5.1. Model Formulation and Weighted Least Squares

Consider a model of the form:

$$y(x) = \sum_{k=0}^{M-1} a_k X_k(x) \tag{15.5.1}$$

where $X_k(x)$ are known basis functions and $a_k$ are unknown parameters. Each basis function represents a prescribed component of the model, and the parameters determine how these components are combined to approximate the observed data. The choice of basis functions reflects prior knowledge, modeling assumptions, or computational convenience, and it plays a central role in determining the flexibility and behavior of the model.

Given data $(x_i, y_i)$ with uncertainties $\sigma_i$, the discrepancy between the model and the observations is measured using a weighted chi-square objective:

$$\chi^2(a)= \sum_{i=1}^{N}\left[\frac{y_i - \sum_{k=0}^{M-1} a_k X_k(x_i)}{\sigma_i}\right]^2 \tag{15.5.2}$$

Each term in this sum represents a squared residual scaled by the corresponding standard deviation. As in the straight-line case, this weighting ensures that observations with larger uncertainty contribute less to the objective, while more reliable data points have greater influence on the fitted parameters. The objective therefore incorporates both the model structure and the statistical properties of the data.

This formulation generalizes the straight-line case and allows for a wide range of models. By choosing different sets of basis functions, one can represent many common modeling approaches within the same framework. Examples include polynomial regression, where the basis functions are powers of $x$; Fourier expansions, where trigonometric functions capture periodic behavior; special-function approximations, where functions are chosen to reflect known analytical structure; and feature-based models in data analysis, where basis functions represent extracted or engineered features of the data.

The key property is linearity in the parameters $a_k$, which ensures that the resulting optimization problem remains tractable. Even when the basis functions themselves are nonlinear in $x$, the dependence on the parameters is linear, so the problem reduces to minimizing a quadratic function of the coefficients. This structure leads directly to systems of linear equations and allows the use of efficient numerical linear algebra methods for computing the solution.

In this way, a broad class of modeling problems can be treated within a unified least-squares framework, providing both computational efficiency and a clear theoretical foundation.

## 15.5.2. Matrix Formulation and Normal Equations

To express the weighted least-squares problem in a compact and structured form, it is convenient to introduce matrix notation. Define the design matrix $A \in \mathbb{R}^{N\times M}$, the data vector $b \in \mathbb{R}^N$, and the parameter vector $a \in \mathbb{R}^M$ by:

$$A_{ik} = \frac{X_k(x_i)}{\sigma_i}, \qquad b_i = \frac{y_i}{\sigma_i}, \qquad a =\begin{pmatrix}a_0 \\ \cdots \\a_{M-1}\end{pmatrix} \tag{15.5.3}$$

This construction incorporates the weighting directly into the matrix and vector, so that the problem is reduced to an unweighted least-squares formulation. Each row of the matrix $A$ corresponds to a single observation, with entries given by the basis functions evaluated at $x_i$ and scaled by the corresponding uncertainty. The vector $b$ contains the scaled observations, and the parameter vector $a$ collects the unknown coefficients.

With these definitions, the objective function can be written compactly as:

$$\chi^2(a) = |A a - b|_2^2 \tag{15.5.4}$$

This expression represents the squared Euclidean norm of the residual vector $r = A a - b$. The least-squares problem is therefore equivalent to finding the parameter vector $a$ that minimizes the distance between the vector $b$ and the column space of $A$. This geometric interpretation provides a useful perspective, as it connects the algebraic formulation to the idea of orthogonal projection in a vector space.

The minimizer satisfies the normal equations:

$$A^\top A \, a = A^\top b \tag{15.5.5}$$

These equations are obtained by differentiating the objective with respect to the parameters and setting the result equal to zero. They express the condition that the residual vector is orthogonal to every column of $A$, which means that the error cannot be reduced further by adjusting the parameters within the model space. In geometric terms, the solution $A a$ is the orthogonal projection of $b$ onto the column space of $A$.

These equations form the algebraic core of linear least squares and appear throughout numerical analysis. Many methods for solving least-squares problems, including those based on matrix factorizations, can be understood as alternative ways of enforcing this orthogonality condition while maintaining numerical stability.

The baseline draft also emphasized that the matrix $A^\top A$ plays a dual role: it determines both the optimality condition and the covariance of the estimator. In particular,

$$\mathrm{Var}(a_j) = C_{jj}, \qquad C = (A^\top A)^{-1} \tag{15.5.6}$$

This relationship shows that the same matrix that governs the solution also determines how uncertainty propagates to the estimated parameters. The diagonal entries of $C$ provide the variances of the individual parameters, while the off-diagonal entries describe their correlations.

This highlights that uncertainty quantification is directly tied to the linear algebra of the problem. Properties such as conditioning, rank, and the distribution of the columns of $A$ influence not only the numerical stability of the solution but also the reliability of the estimated parameters. As a result, the analysis of least-squares problems naturally combines algebraic, geometric, and statistical perspectives within a unified framework.

## 15.5.3. Why Linear Least Squares Appears Everywhere

General linear least squares is not merely a statistical tool. It is a fundamental computational primitive that arises in a wide range of scientific and engineering applications. Its importance stems from the fact that many complex problems, when analyzed or approximated appropriately, reduce to the task of solving a linear least-squares system. As a result, least squares serves as a unifying framework that connects diverse areas of computational science.

One reason for this ubiquity is that linear least-squares problems naturally emerge whenever a model is linearized. In many applications, the original problem is nonlinear in its parameters, but iterative solution methods approximate it locally by linear models. These local approximations lead directly to least-squares subproblems, which must be solved repeatedly as part of the overall algorithm.

### Nonlinear Modeling

Methods such as Gauss–Newton repeatedly solve linear least-squares problems derived from Jacobians. At each iteration, the nonlinear model is approximated by its first-order expansion, and the update to the parameters is obtained by minimizing a least-squares objective involving the Jacobian matrix. In this way, the success and efficiency of the nonlinear method depend critically on the accurate and stable solution of the underlying linear least-squares systems.

### Variational Data Assimilation

Large-scale estimation problems are solved by iteratively forming and solving linearized least-squares systems (Daužickaitė et al., 2026). In these problems, observational data are combined with prior information and dynamical models, leading to optimization formulations that are typically nonlinear. Linear least-squares subproblems arise naturally when the model is linearized around a current estimate, and efficient solution of these subproblems is essential for handling the scale and complexity of modern data assimilation systems.

### Inverse Problems and PDE-Constrained Optimization

Discretization and linearization lead to structured least-squares problems in which sparsity and conditioning are critical (Scott and Tůma, 2025). Inverse problems seek to recover unknown quantities from indirect measurements, often governed by differential equations. When these problems are discretized, the resulting systems frequently have a least-squares structure, with matrices that are large, sparse, and potentially ill-conditioned. Numerical methods must therefore exploit this structure to achieve both efficiency and stability.

These applications demonstrate that linear least squares is not a standalone technique, but a core building block underlying modern computational science. Its repeated appearance across different domains reflects the fact that it provides a natural and flexible way to quantify and minimize discrepancies between models and data, while remaining amenable to efficient numerical solution and rigorous analysis.

## 15.5.4. Derivation, Covariance, and Interpretation

The least-squares objective:

$$\chi^2(a) = |A a - b|_2^2 \tag{15.5.7}$$

is a quadratic function of the parameter vector $a$. Its structure reflects the squared Euclidean norm of the residual vector, and this quadratic form ensures that the problem admits a well-defined minimizer when the columns of (A) are sufficiently independent.

To determine this minimizer, one computes the gradient of the objective with respect to $a$:

$$\nabla \chi^2(a) = 2 A^\top (A a - b) \tag{15.5.8}$$

This expression shows that the gradient is obtained by projecting the residual vector onto the column space of $A$ through the transpose operation. The factor of two arises from differentiation of the squared norm and does not affect the location of the minimizer.

Setting this gradient equal to zero yields the normal equations (15.5.5). Thus, the normal equations represent the first-order optimality condition for the least-squares problem. They characterize the point at which the residual is orthogonal to every column of the design matrix, meaning that no further reduction in the objective is possible within the space spanned by the model.

This derivation provides both an algebraic and a geometric interpretation. Algebraically, the solution is obtained by solving a system of linear equations. Geometrically, the solution corresponds to projecting the data vector $b$ onto the column space of $A$, with the fitted values given by this projection. The residual vector is therefore orthogonal to the model space, which is the defining property of least-squares solutions.

The covariance structure follows directly from the inverse of $A^\top A$, reinforcing the idea that statistical uncertainty is encoded in the geometry of the design matrix. The matrix $A^\top A$ captures how the columns of $A$ relate to one another, and its inverse determines how uncertainty in the data propagates to uncertainty in the estimated parameters. Poor conditioning or near-linear dependence among columns leads to large entries in the inverse, indicating increased uncertainty and sensitivity in the parameter estimates.

A key conceptual message is that the normal equations should be viewed as a mathematical characterization of the solution, not necessarily as a computational method. Although they arise naturally from the derivation, their direct use in numerical computation can be problematic. Forming $A^\top A$ can square the condition number of $A$, which increases sensitivity to perturbations and amplifies rounding errors in finite-precision arithmetic.

For this reason, the baseline explicitly warns against using the normal equations as a primary computational tool and instead recommends more stable alternatives such as QR or SVD factorizations. These methods enforce the same optimality condition without explicitly forming $A^\top A$, thereby preserving numerical accuracy and stability. This distinction between theoretical formulation and practical computation is essential for reliable implementation of least-squares methods.

## 15.5.5. Algorithmic Approaches and Cost Models

Let $A \in \mathbb{R}^{N\times M}$ with $N \ge M$. In this setting, the least-squares problem seeks a parameter vector $a \in \mathbb{R}^M$ that minimizes the residual norm $|A a - b|_2$. Several algorithmic approaches are available for solving this problem, each with distinct computational costs and numerical properties. The choice of method depends on considerations such as problem size, conditioning, and the need for robustness.

### Normal Equations + Cholesky

One of the most direct approaches is to form the normal equations and solve them using a Cholesky factorization. This involves the following steps:

1. Form $G = A^\top A$: cost $O(NM^2)$
2. Form $g = A^\top b$: cost $O(NM)$
3. Solve $G a = g$: cost $O(M^3)$

The appeal of this method lies in its simplicity and efficiency when $M$ is small relative to $N$. Once the matrix $G$ is formed, the system to be solved is relatively small, and Cholesky factorization provides an efficient solution technique.

However, this approach can be numerically unstable due to conditioning. Forming $A^\top A$ squares the condition number of $A$, which can significantly amplify errors when the columns of (A) are nearly linearly dependent. As a result, even though the method is computationally efficient, it may produce inaccurate results in ill-conditioned problems.

### QR Factorization (Preferred for Full Rank)

A more stable alternative is based on QR factorization. Let:

$$A = Q R, \qquad Q \in \mathbb{R}^{N\times M}, \quad R \in \mathbb{R}^{M\times M} \tag{15.5.9}$$

Here, the matrix $Q$ has orthonormal columns, and $R$ is upper triangular. This factorization transforms the least-squares problem into a simpler form by exploiting the orthogonality of $Q$.

Using this factorization, the residual norm can be written as:

$$|A a - b|_2 = |R a - Q^\top b|_2, \tag{15.5.10}$$

which shows that the problem reduces to solving a triangular system. The solution is obtained by solving:

$$R a = Q^\top b \tag{15.5.11}$$

Because $R$ is upper triangular, this system can be solved efficiently using back substitution.

The computational cost of QR factorization is $O(NM^2)$, which is comparable to forming the normal equations. However, the key advantage is numerical stability. The orthogonality of $Q$ ensures that the transformation preserves the norm of the residual, and the method avoids the explicit formation of $A^\top A$, thereby preventing the amplification of conditioning effects. For full-rank problems, QR factorization is therefore the preferred method in practice.

### Singular Value Decomposition (Most Robust)

The most robust approach is based on the singular value decomposition. If:

$$A = U \Sigma V^\top \tag{15.5.12}$$

then the minimum-norm solution is given by:

$$a = V \Sigma^+ U^\top b \tag{15.5.13}$$

Here, $\Sigma^+$ denotes the pseudoinverse of the diagonal matrix of singular values, obtained by inverting nonzero singular values and leaving zeros unchanged.

The SVD provides a complete characterization of the linear system, revealing both its rank and its conditioning. Small singular values indicate directions in which the data provide little information, and their presence can lead to large variations in the solution. By truncating or regularizing these small singular values, one can control variance amplification and obtain more stable estimates.

Although the computational cost of SVD is higher than that of QR factorization, it offers unmatched robustness. It handles rank deficiency and ill-conditioning in a principled way and provides additional diagnostic information about the problem. The baseline emphasizes that SVD “cannot fail” in the way normal-equation methods can in near-singular cases, since it does not rely on forming potentially ill-conditioned intermediate matrices.

These algorithmic approaches illustrate an important principle: while multiple methods may solve the same least-squares problem, their numerical behavior can differ substantially. Efficient and reliable computation therefore requires careful selection of the algorithm based on the structure and conditioning of the problem.

### Rust Implementation

Following the discussion in Section 15.5 on the formulation of general linear least-squares problems and their representation in matrix form through equations (15.5.3)–(15.5.5), Program 15.5.1 provides a practical implementation of weighted least-squares estimation using general basis functions. In many applications, the model is linear in the parameters but constructed from nonlinear basis functions of the independent variable, making it essential to assemble the corresponding design matrix and incorporate observational uncertainties through appropriate weighting. This program builds the weighted system, solves the resulting least-squares problem using both the normal equations with Cholesky factorization and a QR-based approach, and computes the covariance matrix of the estimated parameters. By comparing these computational pathways and evaluating residuals and parameter uncertainty, the implementation illustrates how the abstract formulation of general linear least squares translates into a complete and numerically meaningful workflow.

At the core of the implementation is the construction of the weighted linear least-squares system corresponding to equations (15.5.3) and (15.5.4). The function `basis_functions` defines the set of model functions $X_k(x)$, allowing the program to represent a general linear-in-parameters model even when the dependence on the independent variable is nonlinear. The function `build_weighted_system` then assembles the matrix $A$ and vector $b$ by scaling each row with the inverse of the observation standard deviation, thereby incorporating the weighting implied by the covariance structure. This transformation converts the original weighted problem into an equivalent unweighted least-squares problem in scaled variables.

The function `normal_equations` implements the algebraic formulation in equation (15.5.5) by constructing the matrix $A^{\top}A$ and the vector $A^{\top}b$. The resulting linear system is solved using `cholesky_solve`, which performs a Cholesky factorization of the symmetric positive-definite normal matrix. This approach is efficient and exploits the structure of the system, but as discussed in earlier sections, it relies on the conditioning of $A^{\top}A$ and may be sensitive in less favorable cases.

To provide a numerically robust alternative, the program also includes the function `householder_qr_solve`, which computes the least-squares solution using Householder QR factorization. This method avoids the explicit formation of $A^{\top}A$ and preserves numerical stability by working directly with the design matrix. The two approaches therefore illustrate the distinction between an algebraically derived formulation and a computationally stable realization of the same least-squares problem.

The statistical interpretation of the solution is captured through the covariance matrix, computed by the function `covariance_from_normal_matrix`. This function implements equation (15.5.6) by inverting the normal matrix $A^{\top}A$, thereby quantifying the uncertainty associated with the estimated parameters. The diagonal entries of this matrix represent the variances of the coefficients, while the off-diagonal entries encode correlations between them.

The `main` function integrates these components into a complete computational workflow. It generates synthetic data using a known coefficient vector and heteroscedastic noise levels, constructs the weighted system, computes solutions using both normal equations and QR factorization, and evaluates residual norms and parameter errors. It then computes the covariance matrix and extracts the diagonal variances to assess parameter uncertainty. This sequence demonstrates how model formulation, numerical solution, and statistical interpretation are interconnected in general linear least-squares problems.

```rust
// Program 15.5.1: General Linear Least Squares with Weighted Basis Functions
//
// Problem Statement:
// Solve a general weighted linear least-squares problem of the form
//
//     y(x) = sum_{k=0}^{M-1} a_k X_k(x),
//
// where the basis functions X_k(x) are known and the coefficients a_k are
// unknown. Using the weighted matrix formulation from Equations (15.5.1)
// through (15.5.5), this program:
//
// 1. Builds a weighted design matrix A and weighted data vector b
// 2. Solves the least-squares problem by
//      (a) normal equations + Cholesky factorization
//      (b) Householder QR factorization
// 3. Compares the two solutions and residual norms
// 4. Computes the covariance matrix C = (A^T A)^(-1) from Equation (15.5.6)
//
// The example uses basis functions
//     X_0(x) = 1, X_1(x) = x, X_2(x) = x^2, X_3(x) = sin(2x),
// which illustrates that the model is linear in the parameters even though
// some basis functions are nonlinear in x.

type Vector = Vec<f64>;
type Matrix = Vec<Vec<f64>>;

// ----------------------------------------------------------
// Basic helpers
// ----------------------------------------------------------

fn zeros(rows: usize, cols: usize) -> Matrix {
    vec![vec![0.0; cols]; rows]
}

fn identity(n: usize) -> Matrix {
    let mut id = zeros(n, n);
    for i in 0..n {
        id[i][i] = 1.0;
    }
    id
}

fn transpose(a: &Matrix) -> Matrix {
    let rows = a.len();
    let cols = a[0].len();
    let mut t = zeros(cols, rows);
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
    let rows = a.len();
    let inner = a[0].len();
    let cols = b[0].len();
    let mut c = zeros(rows, cols);
    for i in 0..rows {
        for k in 0..inner {
            let aik = a[i][k];
            for j in 0..cols {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

fn mat_vec_mul(a: &Matrix, x: &Vector) -> Vector {
    let rows = a.len();
    let cols = a[0].len();
    let mut y = vec![0.0; rows];
    for i in 0..rows {
        for j in 0..cols {
            y[i] += a[i][j] * x[j];
        }
    }
    y
}

fn dot(x: &Vector, y: &Vector) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn vec_sub(a: &Vector, b: &Vector) -> Vector {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn vec_norm2(x: &Vector) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn vec_norm_inf(x: &Vector) -> f64 {
    x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
}

fn one_norm(a: &Matrix) -> f64 {
    let rows = a.len();
    let cols = a[0].len();
    let mut max_col_sum: f64 = 0.0;
    for j in 0..cols {
        let mut sum: f64 = 0.0;
        for i in 0..rows {
            sum += a[i][j].abs();
        }
        max_col_sum = max_col_sum.max(sum);
    }
    max_col_sum
}

fn print_vector(name: &str, v: &Vector) {
    println!("{name}");
    for (i, value) in v.iter().enumerate() {
        println!("  [{:>2}] {:>.12e}", i, value);
    }
}

fn print_matrix(name: &str, a: &Matrix) {
    println!("{name}");
    for row in a {
        for value in row {
            print!("{:>18.10e} ", value);
        }
        println!();
    }
}

// ----------------------------------------------------------
// Basis functions for Equation (15.5.1)
// ----------------------------------------------------------

fn basis_functions(x: f64) -> Vector {
    vec![1.0, x, x * x, (2.0 * x).sin()]
}

// ----------------------------------------------------------
// Dense linear algebra solvers
// ----------------------------------------------------------

fn solve_linear_system(mut a: Matrix, mut b: Vector) -> Result<Vector, String> {
    let n = a.len();
    if n == 0 || a[0].len() != n || b.len() != n {
        return Err("Dimension mismatch in solve_linear_system".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_val = a[k][k].abs();
        for i in (k + 1)..n {
            if a[i][k].abs() > pivot_val {
                pivot_val = a[i][k].abs();
                pivot_row = i;
            }
        }

        if pivot_val < 1.0e-15 {
            return Err("Matrix is singular or numerically singular".to_string());
        }

        if pivot_row != k {
            a.swap(k, pivot_row);
            b.swap(k, pivot_row);
        }

        for i in (k + 1)..n {
            let factor = a[i][k] / a[k][k];
            a[i][k] = 0.0;
            for j in (k + 1)..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    Ok(x)
}

fn inverse(a: &Matrix) -> Result<Matrix, String> {
    let n = a.len();
    if n == 0 || a[0].len() != n {
        return Err("Matrix must be square for inversion".to_string());
    }

    let mut inv = zeros(n, n);
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let col = solve_linear_system(a.clone(), e)?;
        for i in 0..n {
            inv[i][j] = col[i];
        }
    }
    Ok(inv)
}

fn cond_estimate_one_norm(a: &Matrix) -> Result<f64, String> {
    let inv_a = inverse(a)?;
    Ok(one_norm(a) * one_norm(&inv_a))
}

fn cholesky_factor(a: &Matrix) -> Result<Matrix, String> {
    let n = a.len();
    if n == 0 || a[0].len() != n {
        return Err("Matrix must be square for Cholesky factorization".to_string());
    }

    let mut l = zeros(n, n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }

            if i == j {
                if sum <= 0.0 {
                    return Err("Matrix is not positive definite".to_string());
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    Ok(l)
}

fn forward_substitution(l: &Matrix, b: &Vector) -> Result<Vector, String> {
    let n = l.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        if l[i][i].abs() < 1.0e-15 {
            return Err("Zero diagonal encountered in forward substitution".to_string());
        }
        y[i] = sum / l[i][i];
    }
    Ok(y)
}

fn back_substitution_upper(u: &Matrix, y: &Vector) -> Result<Vector, String> {
    let n = u.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= u[i][j] * x[j];
        }
        if u[i][i].abs() < 1.0e-15 {
            return Err("Zero diagonal encountered in back substitution".to_string());
        }
        x[i] = sum / u[i][i];
    }
    Ok(x)
}

fn cholesky_solve(a: &Matrix, b: &Vector) -> Result<Vector, String> {
    let l = cholesky_factor(a)?;
    let y = forward_substitution(&l, b)?;
    let lt = transpose(&l);
    back_substitution_upper(&lt, &y)
}

// ----------------------------------------------------------
// Householder QR least-squares solver
// ----------------------------------------------------------

fn householder_qr_solve(a: &Matrix, b: &Vector) -> Result<Vector, String> {
    let n = a.len();
    let m = a[0].len();

    if b.len() != n {
        return Err("Dimension mismatch in householder_qr_solve".to_string());
    }
    if n < m {
        return Err("Need n >= m for least-squares solution".to_string());
    }

    let mut r = a.clone();
    let mut qt_b = b.clone();

    for k in 0..m {
        let mut x = vec![0.0; n - k];
        for i in k..n {
            x[i - k] = r[i][k];
        }

        let norm_x = vec_norm2(&x);
        if norm_x < 1.0e-15 {
            continue;
        }

        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * norm_x;

        let v_norm = vec_norm2(&x);
        if v_norm < 1.0e-15 {
            continue;
        }

        let v: Vector = x.iter().map(|xi| xi / v_norm).collect();

        for j in k..m {
            let mut col_segment = vec![0.0; n - k];
            for i in k..n {
                col_segment[i - k] = r[i][j];
            }
            let proj = 2.0 * dot(&v, &col_segment);
            for i in k..n {
                r[i][j] -= proj * v[i - k];
            }
        }

        let mut b_segment = vec![0.0; n - k];
        for i in k..n {
            b_segment[i - k] = qt_b[i];
        }
        let proj_b = 2.0 * dot(&v, &b_segment);
        for i in k..n {
            qt_b[i] -= proj_b * v[i - k];
        }
    }

    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let mut sum = qt_b[i];
        for j in (i + 1)..m {
            sum -= r[i][j] * x[j];
        }
        if r[i][i].abs() < 1.0e-15 {
            return Err("Triangular factor is singular or numerically singular".to_string());
        }
        x[i] = sum / r[i][i];
    }

    Ok(x)
}

// ----------------------------------------------------------
// Least-squares constructions from Equations (15.5.3)-(15.5.6)
// ----------------------------------------------------------

fn build_weighted_system(xs: &Vector, ys: &Vector, sigmas: &Vector) -> (Matrix, Vector) {
    let n = xs.len();
    let m = basis_functions(xs[0]).len();

    let mut a = zeros(n, m);
    let mut b = vec![0.0; n];

    for i in 0..n {
        let phi = basis_functions(xs[i]);
        let w = 1.0 / sigmas[i];
        for k in 0..m {
            a[i][k] = phi[k] * w;
        }
        b[i] = ys[i] * w;
    }

    (a, b)
}

fn normal_equations(a: &Matrix, b: &Vector) -> (Matrix, Vector) {
    let at = transpose(a);
    let ata = mat_mul(&at, a);
    let atb = mat_vec_mul(&at, b);
    (ata, atb)
}

fn residual_vector(a: &Matrix, x: &Vector, b: &Vector) -> Vector {
    vec_sub(&mat_vec_mul(a, x), b)
}

fn covariance_from_normal_matrix(ata: &Matrix) -> Result<Matrix, String> {
    inverse(ata)
}

// ----------------------------------------------------------
// Example data
// ----------------------------------------------------------

fn build_example() -> (Vector, Vector, Vector, Vector) {
    let n = 28;
    let true_coeffs = vec![1.0, -0.6, 0.25, 0.45];

    let mut xs = vec![0.0; n];
    let mut ys = vec![0.0; n];
    let mut sigmas = vec![0.0; n];

    for i in 0..n {
        let t = i as f64 / ((n - 1) as f64);
        let x = -1.5 + 3.0 * t;
        xs[i] = x;

        // Heteroscedastic standard deviations.
        sigmas[i] = 0.05 + 0.02 * (0.5 + 0.5 * (1.7 * x).cos());

        let phi = basis_functions(x);
        let y_exact = dot(&phi, &true_coeffs);

        // Deterministic synthetic perturbation for reproducibility.
        let noise =
            sigmas[i] * (0.45 * (2.1 * x).sin() - 0.30 * (3.4 * x).cos() + 0.18 * (4.8 * x).sin());

        ys[i] = y_exact + noise;
    }

    (xs, ys, sigmas, true_coeffs)
}

// ----------------------------------------------------------
// Main demonstration
// ----------------------------------------------------------

fn main() -> Result<(), String> {
    let (xs, ys, sigmas, true_coeffs) = build_example();
    let (a, b) = build_weighted_system(&xs, &ys, &sigmas);
    let (ata, atb) = normal_equations(&a, &b);

    let coeffs_ne = cholesky_solve(&ata, &atb)?;
    let coeffs_qr = householder_qr_solve(&a, &b)?;

    let r_ne = residual_vector(&a, &coeffs_ne, &b);
    let r_qr = residual_vector(&a, &coeffs_qr, &b);

    let cov = covariance_from_normal_matrix(&ata)?;
    let mut variances = vec![0.0; cov.len()];
    for i in 0..cov.len() {
        variances[i] = cov[i][i];
    }

    println!("General Linear Least Squares with Weighted Basis Functions");
    println!("==========================================================");
    println!();

    print_vector("True coefficient vector", &true_coeffs);
    println!();

    print_vector("Sample locations x_i", &xs);
    println!();

    print_vector("Observed data y_i", &ys);
    println!();

    print_vector("Observation standard deviations sigma_i", &sigmas);
    println!();

    print_matrix("Weighted design matrix A", &a);
    println!();

    print_vector("Weighted data vector b", &b);
    println!();

    print_matrix("Normal matrix A^T A", &ata);
    println!();

    let cond_ata = cond_estimate_one_norm(&ata)?;
    println!("Estimated 1-norm condition number of A^T A = {:>.12e}", cond_ata);
    println!();

    println!("Solution by Normal Equations + Cholesky");
    println!("---------------------------------------");
    print_vector("Coefficient estimate", &coeffs_ne);
    println!("Residual norm ||A a - b||_2 = {:>.12e}", vec_norm2(&r_ne));
    println!("Infinity-norm error relative to truth = {:>.12e}", vec_norm_inf(&vec_sub(&coeffs_ne, &true_coeffs)));
    println!();

    println!("Solution by Householder QR");
    println!("--------------------------");
    print_vector("Coefficient estimate", &coeffs_qr);
    println!("Residual norm ||A a - b||_2 = {:>.12e}", vec_norm2(&r_qr));
    println!("Infinity-norm error relative to truth = {:>.12e}", vec_norm_inf(&vec_sub(&coeffs_qr, &true_coeffs)));
    println!();

    print_vector("Difference a_NE - a_QR", &vec_sub(&coeffs_ne, &coeffs_qr));
    println!();

    print_matrix("Covariance matrix C = (A^T A)^(-1)", &cov);
    println!();

    print_vector("Diagonal variances C_jj", &variances);
    println!();

    println!("Interpretation");
    println!("--------------");
    println!("The weighted system implements Equation (15.5.3), so the objective");
    println!("minimization in Equation (15.5.4) becomes an ordinary least-squares");
    println!("problem in the scaled variables. The normal equations in Equation");
    println!("(15.5.5) are solved here with Cholesky factorization, while the same");
    println!("problem is also solved directly by Householder QR. The covariance");
    println!("matrix in Equation (15.5.6) is obtained from the inverse of A^T A.");
    println!("This example therefore connects model formulation, matrix structure,");
    println!("solution algorithms, and uncertainty quantification within a single");
    println!("general linear least-squares workflow.");
    println!();

    let _ = identity(1); // Keeps the program self-contained if identity is reused later.

    Ok(())
}
```

Program 15.5.1 demonstrates the practical implementation of general linear least-squares models in a weighted setting, combining model construction, numerical solution, and statistical interpretation within a single framework. The use of general basis functions highlights the flexibility of the linear-in-parameters formulation, while the weighted system reflects the incorporation of observational uncertainty as described in equations (15.5.3) and (15.5.4).

The comparison between the normal-equations approach and the QR-based solution illustrates that, for well-conditioned problems, both methods can produce consistent results, while also reinforcing the importance of numerically stable algorithms in more challenging settings. The computation of the covariance matrix further extends the analysis beyond point estimation, providing insight into the reliability of the fitted parameters and the influence of the design matrix on uncertainty.

The modular structure of the implementation allows for straightforward extension to other models, including higher-dimensional basis functions, alternative weighting schemes, or large-scale problems. It also provides a foundation for more advanced developments, such as regularization, sparse least-squares methods, and iterative solvers, which build upon the same principles to address increasingly complex computational scenarios.

## 15.5.6. Large-Scale, Sparse, and Modern Least-Squares Computation

In modern applications, least-squares problems are often characterized by matrices that are both large and sparse. In such settings, the dominant computational challenges differ significantly from those encountered in small dense problems. Rather than being limited by arithmetic operations alone, performance is strongly influenced by sparsity structure, memory access patterns, and data movement. As a result, efficient algorithms must be designed to exploit structure while minimizing both computational and communication costs.

When matrices are sparse, only a small fraction of their entries are nonzero. This sparsity can be leveraged to reduce storage requirements and computational effort, but it also introduces new considerations. In particular, the computational cost is determined not only by the number of nonzero entries but also by how those entries are arranged. Memory access becomes a critical factor, since irregular sparsity patterns can lead to inefficient data movement and reduced performance on modern hardware.

A central issue in sparse direct methods is fill-in during factorization. Even if the original matrix is sparse, the factors produced during QR or related decompositions may contain many additional nonzero entries. This increase in density can significantly raise both computational cost and memory usage. Controlling fill-in through ordering strategies and structure-aware algorithms is therefore essential for maintaining efficiency in large-scale problems.

In many cases, iterative methods provide a more scalable alternative to direct factorization. Krylov subspace methods, in particular, are widely used for solving large least-squares problems. The efficiency of these methods depends critically on preconditioning, which transforms the problem into an equivalent form that converges more rapidly. Algebraic preconditioners are often constructed to approximate the inverse of the system matrix while preserving sparsity, thereby reducing the number of iterations required for convergence.

### Modern Approaches

Modern approaches to sparse least squares include sparse QR factorizations and augmented-system formulations, which exploit structure to maintain stability while controlling fill-in. In addition, structure-exploiting algorithms are widely used in PDE-based problems, where the underlying physical discretization leads to matrices with predictable sparsity patterns. A comprehensive review of such methods is given by Scott and Tůma (2025), focusing on current practices in large-scale scientific computing.

Recent developments in numerical linear algebra are further reshaping least-squares computation in response to the increasing scale and complexity of modern data problems. Randomized methods, often referred to as randomized numerical linear algebra (RandNLA), use sketching techniques to reduce the dimensionality of the problem. By projecting the data onto a lower-dimensional subspace, these methods produce approximate solutions with reduced computational cost and communication overhead (Murray et al., 2023). This approach is particularly valuable in distributed or data-intensive environments where full access to the entire dataset is expensive.

However, these methods introduce new challenges. Stability concerns arise because standard sketch-and-precondition techniques may fail for ill-conditioned problems, where small perturbations can lead to large errors in the solution (Meier et al., 2024). To address this issue, backward-stable randomized solvers have been developed. These methods combine sketching with iterative refinement, using approximate solutions as a starting point and progressively improving accuracy while maintaining computational efficiency (Epperly et al., 2024).

Mixed-precision methods represent another important direction. These approaches exploit the fact that modern hardware often performs lower-precision arithmetic more efficiently than high-precision computation. Initial calculations are carried out in low precision to achieve speed, and iterative refinement is then applied in higher precision to recover accuracy (Carson and Daužickaitė, 2024). This strategy provides a balance between performance and numerical reliability, making it well suited for large-scale applications.

These developments reflect modern computational realities, including the presence of large datasets, limitations in memory bandwidth, and hardware architectures that favor reduced precision arithmetic. As a result, the design of least-squares algorithms must take into account not only mathematical structure but also the characteristics of the computing environment.

For a Rust-based numerical computing environment, these considerations are particularly relevant. Performance often depends on careful control of memory layout, efficient handling of sparse data structures, and deliberate management of numerical precision. By aligning algorithmic choices with these constraints, one can achieve implementations that are both efficient and robust, even in demanding large-scale settings.

### Rust Implementation

Following the discussion in Section 15.5.6 on large-scale and sparse least-squares problems, Program 15.5.2 provides a practical implementation of an iterative Krylov subspace method for solving least-squares systems without explicitly forming the normal matrix. In large-scale settings, the construction of $A^\top A$ can be computationally prohibitive and may destroy sparsity, leading to increased memory usage and reduced numerical efficiency. This program adopts the Conjugate Gradient Least Squares (CGLS) method, which operates using only matrix-vector products with $A$ and $A^\top$, thereby preserving sparsity and enabling efficient computation. By constructing a structured sparse system, generating synthetic data, and monitoring convergence through residual norms, the implementation illustrates how modern least-squares solvers are designed to handle high-dimensional problems while maintaining numerical stability and scalability.

At the core of the implementation is the `CsrMatrix` structure, which stores the sparse design matrix in compressed sparse row format. This representation separates the matrix into row pointers, column indices, and numerical values, allowing efficient storage and computation by avoiding explicit representation of zero entries. The methods `matvec` and `t_matvec` implement multiplication by $A$ and $A^\top$, respectively, which are the fundamental operations required in iterative least-squares algorithms. By relying exclusively on these operations, the program avoids forming the dense normal matrix $A^\top A$, consistent with the computational strategy emphasized in Section 15.5.6.

The function `build_sparse_banded_matrix` constructs a structured sparse matrix with a narrow stencil, mimicking the type of matrices that arise in discretized differential operators and other large-scale applications. The resulting matrix has a low density, ensuring that storage and computational costs remain manageable even for large problem sizes. The functions `build_true_solution` and `build_rhs` generate a synthetic least-squares problem by defining a known coefficient vector and computing the corresponding right-hand side with a small deterministic perturbation. This approach provides a reproducible test case that reflects realistic measurement noise while allowing direct assessment of solution accuracy.

The iterative solver is implemented in the function `cgls`, which realizes the conjugate-gradient method applied implicitly to the normal equations. Rather than forming $A^\top A$, the algorithm updates the solution using repeated applications of $A$ and $A^\top$, thereby maintaining sparsity and avoiding the numerical drawbacks associated with explicit normal-equation formation. At each iteration, the method computes the residual $r = b - Ax$ and the normal residual $A^\top r$, and these quantities are used to update the search direction and step size. The iteration continues until either the residual norm or the normal residual norm falls below a prescribed tolerance, ensuring convergence to the least-squares solution.

The `main` function integrates all components into a complete computational workflow. It defines the problem dimensions, constructs the sparse matrix and synthetic data, and invokes the CGLS solver with specified tolerance and iteration limits. The program then reports convergence diagnostics, including residual norms, solution error, and selected entries of the computed solution. The iteration history provides insight into the convergence behavior of the method, illustrating the progressive reduction of both the residual and the normal residual. This comprehensive output demonstrates how iterative methods can efficiently solve large sparse least-squares problems while providing quantitative measures of accuracy and convergence.

```rust
// Program 15.5.2: Sparse Large-Scale Least Squares via CGLS
//
// Problem Statement:
// Solve a large sparse least-squares problem
//
//     min_x ||A x - b||_2,
//
// without explicitly forming the normal matrix A^T A.
// The program:
//
// 1. Stores a large sparse matrix A in compressed sparse row (CSR) form
// 2. Generates synthetic data b = A x_true + noise
// 3. Solves the least-squares problem with CGLS
// 4. Reports convergence diagnostics, residual norms, and solution error
//
// This illustrates the large-scale sparse setting discussed in Section 15.5.6,
// where iterative methods are often preferable to dense direct factorizations.

type Vector = Vec<f64>;

#[derive(Clone, Debug)]
struct CsrMatrix {
    nrows: usize,
    ncols: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<f64>,
}

impl CsrMatrix {
    fn new(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, String> {
        if row_ptr.len() != nrows + 1 {
            return Err("row_ptr must have length nrows + 1".to_string());
        }
        if col_idx.len() != values.len() {
            return Err("col_idx and values must have the same length".to_string());
        }
        if *row_ptr.first().unwrap_or(&1) != 0 {
            return Err("row_ptr must start at 0".to_string());
        }
        if *row_ptr.last().unwrap_or(&0) != values.len() {
            return Err("row_ptr last entry must equal number of nonzeros".to_string());
        }
        for w in row_ptr.windows(2) {
            if w[0] > w[1] {
                return Err("row_ptr must be nondecreasing".to_string());
            }
        }
        for &j in &col_idx {
            if j >= ncols {
                return Err("column index out of bounds".to_string());
            }
        }
        Ok(Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        })
    }

    fn nnz(&self) -> usize {
        self.values.len()
    }

    fn matvec(&self, x: &Vector) -> Result<Vector, String> {
        if x.len() != self.ncols {
            return Err("Dimension mismatch in matvec".to_string());
        }
        let mut y = vec![0.0; self.nrows];
        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut sum = 0.0;
            for p in start..end {
                sum += self.values[p] * x[self.col_idx[p]];
            }
            y[i] = sum;
        }
        Ok(y)
    }

    fn t_matvec(&self, y: &Vector) -> Result<Vector, String> {
        if y.len() != self.nrows {
            return Err("Dimension mismatch in t_matvec".to_string());
        }
        let mut x = vec![0.0; self.ncols];
        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let yi = y[i];
            for p in start..end {
                x[self.col_idx[p]] += self.values[p] * yi;
            }
        }
        Ok(x)
    }
}

fn dot(x: &Vector, y: &Vector) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn norm2(x: &Vector) -> f64 {
    dot(x, x).sqrt()
}

fn norm_inf(x: &Vector) -> f64 {
    x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
}

fn axpy(y: &mut Vector, alpha: f64, x: &Vector) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

fn vec_sub(a: &Vector, b: &Vector) -> Vector {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn build_sparse_banded_matrix(nrows: usize, ncols: usize) -> Result<CsrMatrix, String> {
    // Construct a sparse banded matrix with five diagonals where possible.
    // The entries are chosen to produce a reasonably well-behaved but nontrivial
    // least-squares problem resembling a discretized operator.
    let mut row_ptr = Vec::with_capacity(nrows + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    row_ptr.push(0);

    for i in 0..nrows {
        let center = i * ncols / nrows;
        let stencil = [-2isize, -1, 0, 1, 2];

        for &offset in &stencil {
            let j_signed = center as isize + offset;
            if j_signed >= 0 && (j_signed as usize) < ncols {
                let j = j_signed as usize;
                let value = match offset {
                    -2 | 2 => 0.15,
                    -1 | 1 => -0.60,
                    0 => 1.80,
                    _ => 0.0,
                };

                let modulation =
                    1.0 + 0.08 * ((i as f64) / (nrows as f64)).sin()
                        + 0.05 * ((j as f64) / (ncols as f64)).cos();

                col_idx.push(j);
                values.push(value * modulation);
            }
        }

        row_ptr.push(values.len());
    }

    CsrMatrix::new(nrows, ncols, row_ptr, col_idx, values)
}

fn build_true_solution(ncols: usize) -> Vector {
    let mut x_true = vec![0.0; ncols];
    for j in 0..ncols {
        let t = j as f64 / (ncols as f64 - 1.0);
        x_true[j] =
            0.8 * (2.0 * std::f64::consts::PI * t).sin()
            + 0.35 * (5.0 * std::f64::consts::PI * t).cos()
            + 0.20 * t;
    }
    x_true
}

fn build_rhs(a: &CsrMatrix, x_true: &Vector) -> Result<Vector, String> {
    let mut b = a.matvec(x_true)?;
    for (i, bi) in b.iter_mut().enumerate() {
        let t = i as f64 / (a.nrows as f64 - 1.0);
        let noise = 1.0e-3
            * (0.7 * (19.0 * t).sin() - 0.45 * (11.0 * t).cos() + 0.2 * (7.0 * t).sin());
        *bi += noise;
    }
    Ok(b)
}

#[derive(Debug)]
struct CglsResult {
    x: Vector,
    iterations: usize,
    converged: bool,
    residual_norm: f64,
    normal_residual_norm: f64,
    history: Vec<(usize, f64, f64)>, // (iteration, ||r||_2, ||A^T r||_2)
}

fn cgls(
    a: &CsrMatrix,
    b: &Vector,
    max_iters: usize,
    tol: f64,
) -> Result<CglsResult, String> {
    if b.len() != a.nrows {
        return Err("Dimension mismatch in cgls".to_string());
    }

    let mut x = vec![0.0; a.ncols];
    let mut r = b.clone(); // initial residual since x_0 = 0
    let mut s = a.t_matvec(&r)?;
    let mut p = s.clone();

    let mut gamma = dot(&s, &s);
    let b_norm = norm2(b).max(1.0e-30);

    let mut history = Vec::new();
    let mut converged = false;
    let mut final_residual = norm2(&r);
    let mut final_normal_residual = norm2(&s);

    history.push((0, final_residual, final_normal_residual));

    for k in 0..max_iters {
        let q = a.matvec(&p)?;
        let delta = dot(&q, &q);

        if delta <= 1.0e-30 {
            break;
        }

        let alpha = gamma / delta;

        axpy(&mut x, alpha, &p);
        axpy(&mut r, -alpha, &q);

        s = a.t_matvec(&r)?;
        let gamma_new = dot(&s, &s);

        final_residual = norm2(&r);
        final_normal_residual = gamma_new.sqrt();
        history.push((k + 1, final_residual, final_normal_residual));

        if final_residual / b_norm < tol || final_normal_residual / b_norm < tol {
            converged = true;
            return Ok(CglsResult {
                x,
                iterations: k + 1,
                converged,
                residual_norm: final_residual,
                normal_residual_norm: final_normal_residual,
                history,
            });
        }

        let beta = gamma_new / gamma;
        for j in 0..p.len() {
            p[j] = s[j] + beta * p[j];
        }
        gamma = gamma_new;
    }

    Ok(CglsResult {
        x,
        iterations: history.len().saturating_sub(1),
        converged,
        residual_norm: final_residual,
        normal_residual_norm: final_normal_residual,
        history,
    })
}

fn main() -> Result<(), String> {
    let nrows = 2400;
    let ncols = 1800;
    let max_iters = 120;
    let tol = 1.0e-8;

    let a = build_sparse_banded_matrix(nrows, ncols)?;
    let x_true = build_true_solution(ncols);
    let b = build_rhs(&a, &x_true)?;

    let result = cgls(&a, &b, max_iters, tol)?;

    let solution_error = vec_sub(&result.x, &x_true);
    let b_fit = a.matvec(&result.x)?;
    let residual = vec_sub(&b_fit, &b);

    println!("Sparse Large-Scale Least Squares via CGLS");
    println!("=========================================");
    println!();
    println!("Problem Dimensions");
    println!("------------------");
    println!("Number of rows (N)                = {}", a.nrows);
    println!("Number of columns (M)             = {}", a.ncols);
    println!("Number of nonzeros                = {}", a.nnz());
    println!(
        "Matrix density                    = {:>.6e}",
        (a.nnz() as f64) / ((a.nrows * a.ncols) as f64)
    );
    println!();

    println!("Solver Parameters");
    println!("-----------------");
    println!("Maximum iterations                = {}", max_iters);
    println!("Relative tolerance                = {:>.6e}", tol);
    println!();

    println!("Convergence Summary");
    println!("-------------------");
    println!("Converged                         = {}", result.converged);
    println!("Iterations performed              = {}", result.iterations);
    println!("Final residual norm ||A x - b||_2 = {:>.12e}", result.residual_norm);
    println!(
        "Final normal residual ||A^T(Ax-b)||_2 = {:>.12e}",
        result.normal_residual_norm
    );
    println!(
        "Infinity-norm solution error      = {:>.12e}",
        norm_inf(&solution_error)
    );
    println!(
        "Relative solution error           = {:>.12e}",
        norm2(&solution_error) / norm2(&x_true).max(1.0e-30)
    );
    println!();

    println!("Representative Solution Entries");
    println!("-------------------------------");
    for &idx in &[0usize, ncols / 4, ncols / 2, 3 * ncols / 4, ncols - 1] {
        println!(
            "x_true[{:>4}] = {:>.10e},   x_est[{:>4}] = {:>.10e}",
            idx, x_true[idx], idx, result.x[idx]
        );
    }
    println!();

    println!("Representative Residual Entries");
    println!("-------------------------------");
    for &idx in &[0usize, nrows / 4, nrows / 2, 3 * nrows / 4, nrows - 1] {
        println!(
            "r[{:>4}] = {:>.10e}",
            idx, residual[idx]
        );
    }
    println!();

    println!("Iteration History (selected)");
    println!("----------------------------");
    let step = (result.history.len() / 10).max(1);
let mut last_printed = None;

for (k, rnorm, snorm) in result.history.iter().step_by(step) {
    println!(
        "iter = {:>3},   ||r||_2 = {:>.10e},   ||A^T r||_2 = {:>.10e}",
        k, rnorm, snorm
    );
    last_printed = Some(*k);
}

if let Some((k, rnorm, snorm)) = result.history.last() {
    if Some(*k) != last_printed {
        println!(
            "iter = {:>3},   ||r||_2 = {:>.10e},   ||A^T r||_2 = {:>.10e}",
            k, rnorm, snorm
        );
    }
}
    println!();

    println!("Interpretation");
    println!("--------------");
    println!("This program solves a large sparse least-squares problem without");
    println!("forming the dense normal matrix A^T A. The CGLS iteration uses only");
    println!("sparse matrix-vector products with A and A^T, which preserves sparsity");
    println!("and reduces memory traffic. This reflects the large-scale setting in");
    println!("Section 15.5.6, where iterative Krylov methods are often preferable");
    println!("to dense direct factorizations.");
    println!();

    Ok(())
}
```

Program 15.5.2 demonstrates a practical approach to solving large-scale sparse least-squares problems using iterative Krylov subspace methods. By avoiding the explicit formation of the normal matrix, the implementation preserves sparsity and reduces both memory usage and computational cost, addressing the primary challenges discussed in Section 15.5.6. The convergence behavior observed in the iteration history reflects the efficiency of the CGLS method in reducing both the residual norm and the normal residual, thereby satisfying the least-squares optimality condition.

The example highlights the importance of algorithmic design in modern numerical computation. While direct methods such as QR or Cholesky factorization are effective for small to moderate problem sizes, iterative methods become essential when dealing with high-dimensional sparse systems. The use of structured sparse matrices and efficient matrix-vector operations ensures that the computational complexity scales favorably with problem size.

The modular design of the code allows for straightforward extensions to more advanced techniques, including preconditioning strategies to accelerate convergence, regularization methods for ill-posed problems, and parallel implementations for high-performance computing environments. These extensions build upon the same principles demonstrated in this program and form the foundation for modern large-scale least-squares solvers used in scientific computing and data analysis.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/sfWIyqPH6sqGDjonl3KR.6","tags":[]}

# 15.6. Nonlinear Models

While linear-in-parameter models are powerful and widely applicable, many important scientific and engineering problems involve models that are nonlinear in their parameters. In such models, the dependence of the output on the unknown parameters cannot be expressed as a simple linear combination of known functions. Instead, the parameters enter in a nonlinear manner, often through compositions, products, or implicit relationships. Examples include exponential decay models, rational functions in chemical kinetics, constitutive laws in solid mechanics, and parameter-to-observation maps defined implicitly through partial differential equations.

These nonlinear relationships arise naturally when modeling physical processes, where the governing laws themselves are nonlinear or where parameters influence the system in a non-additive way. As a result, restricting attention to linear-in-parameter models is often insufficient for capturing the true behavior of the system, and more general formulations are required.

In such cases, the least-squares framework still applies, since the objective remains the minimization of discrepancies between model predictions and observed data. However, the resulting optimization problem is nonlinear in the parameters, which fundamentally changes both the analysis and the computational approach. Unlike the linear case, where a closed-form solution can be obtained through linear algebra, nonlinear least-squares problems must be solved iteratively.

This introduces several new challenges. The objective function may have multiple local minima, so the solution obtained can depend on the initial guess. The landscape of the objective may be highly curved or irregular, making optimization more difficult and potentially slowing convergence. In addition, each iteration typically requires the evaluation of the model and its derivatives, as well as the solution of a linearized least-squares subproblem, which can dominate the computational cost.

At the same time, the structure of least squares provides powerful tools for constructing efficient algorithms. By exploiting the special form of the objective, one can derive methods that take advantage of the residual structure and its derivatives, leading to algorithms that are more efficient and better behaved than general-purpose nonlinear optimization techniques. These methods build directly on the linear least-squares framework developed in the previous sections, extending it to handle nonlinear dependence in a systematic way.

Thus, nonlinear least squares can be viewed as a natural extension of the linear theory, combining iterative optimization with linear algebra to address a broader class of modeling problems.

## 15.6.1. Nonlinear Least-Squares Formulation

Let the model be:

$$y(x_i; a), \qquad a \in \mathbb{R}^M \tag{15.6.1}$$

where the parameter vector $a$ enters the model in a nonlinear manner. Unlike the linear case, the dependence of the model on the parameters cannot be expressed as a simple linear combination of known functions, and this nonlinearity is the defining feature of the problem.

Given observed data $(x_i, y_i)$ with associated uncertainties $\sigma_i$, the discrepancy between the model and the data is measured using the weighted least-squares objective:

$$\chi^2(a)= \sum_{i=1}^{N}\left[\frac{y_i - y(x_i; a)}{\sigma_i}\right]^2 \tag{15.6.2}$$

As in the linear case, each term represents a squared residual scaled by the corresponding standard deviation, ensuring that observations with higher uncertainty have reduced influence on the fit. This formulation retains the statistical interpretation of least squares as a maximum likelihood estimator under Gaussian noise assumptions.

To express the problem more compactly, define the residual vector:

$$r(a) \in \mathbb{R}^N, \qquad r_i(a) = \frac{y_i - f(x_i; a)}{\sigma_i} \tag{15.6.3}$$

so that,

$$\chi^2(a) = |r(a)|_2^2 \tag{15.6.4}$$

This vector formulation emphasizes that the objective function is the squared Euclidean norm of a residual vector that depends on the parameters. It provides a natural bridge to linear algebraic methods, even though the dependence on $a$ is nonlinear.

The key difference from linear least squares is that $r(a)$ is now a nonlinear function of $a$. As a consequence, the objective function is no longer quadratic, and a closed-form solution is generally not available. The optimization problem must therefore be solved using iterative methods that progressively improve an initial estimate of the parameters.

This nonlinearity introduces several challenges. The objective function may exhibit multiple local minima, and its curvature may vary significantly across the parameter space. As a result, convergence behavior can depend on the initial guess and on the specific algorithm used. Careful formulation and implementation are therefore required to ensure reliable and efficient computation.

However, the least-squares structure still provides important advantages. In particular, derivatives of $\chi^2(a)$ can be expressed in terms of the residual vector and its Jacobian, which captures the sensitivity of the model to changes in the parameters. This structure allows the development of specialized algorithms that exploit the form of the objective, leading to methods that are more efficient and better suited to least-squares problems than general nonlinear optimization techniques.

Thus, even though the problem is nonlinear, the underlying least-squares formulation provides a strong foundation for both analysis and computation.

## 15.6.2. Jacobian Structure and Gauss–Newton Approximation

Define the Jacobian matrix as:

$$J(a) \in \mathbb{R}^{N\times M}, \qquad J_{ik}(a) = \frac{\partial r_i(a)}{\partial a_k} \tag{15.6.5}$$

The Jacobian describes how each component of the residual vector changes with respect to the parameters. Each row corresponds to a single data point, and each column corresponds to a parameter. In this way, the Jacobian captures the local sensitivity of the model to perturbations in the parameter vector and plays a central role in the analysis and solution of nonlinear least-squares problems.

Using this definition, the gradient of the objective function can be written as:

$$\nabla \chi^2(a) = 2 J(a)^\top r(a) \tag{15.6.6}$$

which shows that the gradient is obtained by combining the residual vector with the Jacobian. This expression mirrors the structure seen in linear least squares, where the gradient involves the transpose of the design matrix multiplied by the residual. The same geometric interpretation applies here, with the Jacobian replacing the fixed matrix and reflecting the local linearization of the model.

The exact Hessian of the objective has the structure:

$$\nabla^2 \chi^2(a)= 2 J^\top J + 2 \sum_{i=1}^{N} r_i(a) \, \nabla^2 r_i(a) \tag{15.6.7}$$

This expression consists of two distinct contributions. The first term, $2 J^\top J$, depends only on first derivatives and reflects the curvature induced by the local linear approximation of the residual. The second term involves the second derivatives of the residual components and accounts for the nonlinear curvature of the model itself.

In practice, the second term is often expensive to compute and may be difficult to evaluate accurately, especially for complex models or when derivatives must be approximated numerically. Moreover, its contribution can be sensitive to noise and may introduce instability into the computation.

The key simplification is to neglect this second term, yielding the approximation:

$$\nabla^2 \chi^2(a) \approx 2 J^\top J \tag{15.6.8}$$

This approximation replaces the exact Hessian with a matrix that depends only on the Jacobian, which is typically much easier to compute and more stable numerically.

This leads to the Gauss–Newton method, which exploits the least-squares structure by approximating the curvature using only first derivatives. At each iteration, the method linearizes the residual function around the current estimate and solves a linear least-squares problem based on the Jacobian. In this way, the nonlinear problem is reduced to a sequence of linear subproblems, each of which can be solved using the techniques developed for linear least squares.

The justification for this approximation is that near a good fit, the residuals $r_i(a)$ are small. As a result, the second term in the Hessian contributes little to the overall curvature and can often be neglected without significantly affecting the solution. In contrast, retaining this term may introduce numerical difficulties, particularly if the second derivatives are poorly conditioned or inaccurately computed. By focusing on the dominant contribution from $J^\top J$, the Gauss–Newton method achieves a balance between accuracy and computational efficiency.

## 15.6.3. Gauss–Newton Iteration and Linearized Subproblems

To compute an update to the parameter vector, the Gauss–Newton method begins by linearizing the residual function around the current estimate $a$. Using a first-order approximation, one writes:

$$r(a + \delta) \approx r(a) + J(a)\delta \tag{15.6.9}$$

This expression replaces the nonlinear residual with its local linear approximation, where the Jacobian $J(a)$ captures how the residual changes with respect to the parameters. The vector (\\delta) represents the update to the current parameter estimate.

The update $\delta$ is then chosen to minimize:

$$|r(a) + J(a)\delta|_2^2 \tag{15.6.10}$$

which is a linear least-squares problem in the unknown $\delta$. This formulation reflects the idea that the next iterate should reduce the residual as much as possible within the linearized model. By solving this problem, one obtains a step that improves the fit based on the local behavior of the residual.

The corresponding normal equations are:

$$(J^\top J)\delta = -J^\top r \tag{15.6.11}$$

These equations have the same structure as those encountered in linear least squares, with the Jacobian playing the role of the design matrix and the residual acting as the right-hand side. The solution $\delta$ represents the direction and magnitude of the parameter update that best reduces the linearized residual.

Thus, each Gauss–Newton step requires solving a linear least-squares problem, linking nonlinear optimization directly to the methods developed in the previous section. This connection is fundamental, as it allows the extensive theory and algorithms of linear least squares to be applied within an iterative nonlinear framework. The overall method can therefore be viewed as a sequence of linear approximations, each solved using established numerical techniques.

The computational cost per iteration depends on several components. Residual evaluation is often the dominant cost, particularly when the model $y(x_i; a)$ is expensive to compute or involves complex simulations. Jacobian computation is another significant factor, typically requiring $O(NM)$ operations if the Jacobian is dense. In many applications, efficient computation or approximation of the Jacobian is essential for overall performance.

The solution of the linear system also contributes to the computational cost. For dense problems, direct methods require $O(M^3)$ operations, while for large or sparse systems, iterative methods are often used to reduce cost and memory requirements. The choice of solver depends on the structure of the Jacobian and the scale of the problem.

This analysis highlights a key insight: nonlinear least squares is often limited not by the optimization procedure itself, but by the cost of solving the associated linear algebra subproblems. Each iteration requires forming and solving a linearized system, and the efficiency of these steps determines the overall performance of the algorithm. Consequently, advances in linear algebra methods directly translate into improvements in nonlinear least-squares computation.

### Rust Implementation

Following the discussion in Section 15.6.3 on the linearization of the residual function and the formulation of the Gauss–Newton update, Program 15.6.1 provides a practical implementation of nonlinear least-squares fitting using the Gauss–Newton iteration. In this section, the nonlinear residual is approximated locally by a linear model, allowing each iteration to be formulated as a linear least-squares subproblem. The update is then obtained by solving the corresponding normal equations, linking nonlinear optimization directly to the linear least-squares framework developed earlier. This program implements that iterative procedure for an exponential model, demonstrating how the parameter vector is progressively refined through successive linear approximations and how convergence is achieved when the updates become sufficiently small.

At the core of the implementation is the evaluation of the residual vector defined in Equation (15.6.9). The function `residual` computes the scaled difference between the observed data and the model prediction, incorporating the weighting through the standard deviations. This representation ensures that each data point contributes appropriately to the objective function based on its uncertainty.

The function `jacobian_row` implements the Jacobian matrix $J(a)$ introduced in Equation (15.6.9). For each observation, it computes the partial derivatives of the residual with respect to the parameters. These derivatives form the rows of the Jacobian and capture the local sensitivity of the residual to changes in the parameter vector. The accuracy of this step is essential, as the Jacobian determines the quality of the linear approximation used in each iteration.

The function `form_normal_equations` constructs the system described in Equation (15.6.11). It accumulates the matrix $J^\top J$ and the vector $-J^\top r$ by looping over the data and combining contributions from each observation. This step transforms the linearized least-squares problem in Equation (15.6.10) into a system of linear equations for the update $\delta$.

The function `solve_3x3` solves the resulting linear system using Gaussian elimination with partial pivoting. Although the system is small in this example, this step reflects the general requirement in nonlinear least squares to solve linear algebra subproblems efficiently and stably. In larger problems, this step typically dominates the computational cost.

The iterative procedure is implemented in the function `gauss_newton_fit`. At each iteration, the residual and Jacobian are evaluated, the normal equations are formed and solved, and the parameter vector is updated. The norm of the update $\delta$ is used as a convergence criterion, reflecting the idea that the iteration should terminate when successive updates become negligible.

Supporting functions such as `model`, `compute_chi2`, and the diagnostic routines organize the computation and present the results. The iteration history provides insight into the convergence behavior, while the pointwise diagnostics illustrate how the fitted model compares with the observed data. The `main` function demonstrates the full workflow, including initialization, iteration, and reporting of results.

```rust
// Program 15.6.1. Nonlinear Least Squares via Gauss-Newton Iteration
//
// Problem statement:
// Fit the nonlinear model
//
//     y(x; a) = a0 * exp(a1 * x) + a2
//
// to weighted data (x_i, y_i, sigma_i) by minimizing
//
//     chi^2(a) = sum_i [ (y_i - y(x_i; a)) / sigma_i ]^2.
//
// The program implements the Gauss-Newton method described in Section 15.6.3.
// At each iteration it:
//
//   1. evaluates the residual vector r(a),
//   2. constructs the Jacobian J(a),
//   3. forms the normal equations (J^T J) delta = -J^T r,
//   4. solves for the update delta,
//   5. updates the parameter vector.
//
// This illustrates how nonlinear least squares reduces to a sequence of
// linearized least-squares subproblems.

#[derive(Clone, Copy, Debug)]
struct Observation {
    x: f64,
    y: f64,
    sigma: f64,
}

#[derive(Clone, Copy, Debug)]
struct Parameters {
    a0: f64,
    a1: f64,
    a2: f64,
}

#[derive(Clone, Copy, Debug)]
struct IterationRecord {
    iter: usize,
    params: Parameters,
    chi2: f64,
    step_norm: f64,
}

fn validate_data(data: &[Observation]) -> Result<(), String> {
    if data.len() < 3 {
        return Err("at least three observations are required".to_string());
    }

    for (i, obs) in data.iter().enumerate() {
        if !obs.x.is_finite() || !obs.y.is_finite() || !obs.sigma.is_finite() {
            return Err(format!("observation {} contains a non-finite value", i));
        }
        if obs.sigma <= 0.0 {
            return Err(format!("observation {} has nonpositive sigma", i));
        }
    }

    Ok(())
}

fn model(x: f64, p: Parameters) -> f64 {
    p.a0 * (p.a1 * x).exp() + p.a2
}

fn residual(obs: Observation, p: Parameters) -> f64 {
    (obs.y - model(obs.x, p)) / obs.sigma
}

fn jacobian_row(obs: Observation, p: Parameters) -> [f64; 3] {
    let e = (p.a1 * obs.x).exp();

    // Residual:
    // r_i(a) = (y_i - f(x_i; a)) / sigma_i
    //
    // Hence
    // dr_i/da_k = -(1/sigma_i) * df/da_k.
    let dr_da0 = -e / obs.sigma;
    let dr_da1 = -(p.a0 * obs.x * e) / obs.sigma;
    let dr_da2 = -1.0 / obs.sigma;

    [dr_da0, dr_da1, dr_da2]
}

fn compute_chi2(data: &[Observation], p: Parameters) -> f64 {
    data.iter()
        .map(|&obs| {
            let r = residual(obs, p);
            r * r
        })
        .sum()
}

fn form_normal_equations(data: &[Observation], p: Parameters) -> ([[f64; 3]; 3], [f64; 3], f64) {
    let mut jtj = [[0.0_f64; 3]; 3];
    let mut minus_jtr = [0.0_f64; 3];
    let mut chi2 = 0.0;

    for &obs in data {
        let r = residual(obs, p);
        let j = jacobian_row(obs, p);

        chi2 += r * r;

        for i in 0..3 {
            minus_jtr[i] -= j[i] * r;
            for k in 0..3 {
                jtj[i][k] += j[i] * j[k];
            }
        }
    }

    (jtj, minus_jtr, chi2)
}

fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Result<[f64; 3], String> {
    // Gaussian elimination with partial pivoting.
    for col in 0..3 {
        let mut pivot_row = col;
        let mut pivot_abs = a[col][col].abs();

        for row in (col + 1)..3 {
            let candidate = a[row][col].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        if pivot_abs <= 1.0e-14 {
            return Err("normal-equation matrix is singular or ill-conditioned".to_string());
        }

        if pivot_row != col {
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }

        let pivot = a[col][col];
        for row in (col + 1)..3 {
            let factor = a[row][col] / pivot;
            for k in col..3 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = [0.0_f64; 3];
    for i in (0..3).rev() {
        let mut sum = b[i];
        for k in (i + 1)..3 {
            sum -= a[i][k] * x[k];
        }
        if a[i][i].abs() <= 1.0e-14 {
            return Err("back substitution failed due to a zero pivot".to_string());
        }
        x[i] = sum / a[i][i];
    }

    Ok(x)
}

fn step_norm(delta: [f64; 3]) -> f64 {
    (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
}

fn add_step(p: Parameters, delta: [f64; 3]) -> Parameters {
    Parameters {
        a0: p.a0 + delta[0],
        a1: p.a1 + delta[1],
        a2: p.a2 + delta[2],
    }
}

fn gauss_newton_fit(
    data: &[Observation],
    initial: Parameters,
    tol: f64,
    max_iters: usize,
) -> Result<(Parameters, Vec<IterationRecord>), String> {
    validate_data(data)?;

    let mut p = initial;
    let mut history = Vec::new();

    for iter in 0..max_iters {
        let (jtj, minus_jtr, chi2) = form_normal_equations(data, p);
        let delta = solve_3x3(jtj, minus_jtr)?;
        let delta_norm = step_norm(delta);

        history.push(IterationRecord {
            iter,
            params: p,
            chi2,
            step_norm: delta_norm,
        });

        let next = add_step(p, delta);

        if !next.a0.is_finite() || !next.a1.is_finite() || !next.a2.is_finite() {
            return Err("iteration produced a non-finite parameter value".to_string());
        }

        p = next;

        if delta_norm < tol {
            let final_chi2 = compute_chi2(data, p);
            history.push(IterationRecord {
                iter: iter + 1,
                params: p,
                chi2: final_chi2,
                step_norm: 0.0,
            });
            return Ok((p, history));
        }
    }

    let final_chi2 = compute_chi2(data, p);
    history.push(IterationRecord {
        iter: max_iters,
        params: p,
        chi2: final_chi2,
        step_norm: 0.0,
    });

    Ok((p, history))
}

fn print_input_data(data: &[Observation]) {
    println!("Input Data");
    println!("==========");
    println!(
        "{:>4} {:>12} {:>12} {:>12}",
        "i", "x_i", "y_i", "sigma_i"
    );
    println!("{}", "-".repeat(46));

    for (i, obs) in data.iter().enumerate() {
        println!(
            "{:>4} {:>12.6} {:>12.6} {:>12.6}",
            i, obs.x, obs.y, obs.sigma
        );
    }
    println!();
}

fn print_iteration_history(history: &[IterationRecord]) {
    println!("Gauss-Newton Iteration History");
    println!("==============================");
    println!(
        "{:>4} {:>16} {:>16} {:>16} {:>16} {:>16}",
        "k", "a0", "a1", "a2", "chi^2", "||delta||_2"
    );
    println!("{}", "-".repeat(102));

    for rec in history {
        println!(
            "{:>4} {:>16.8} {:>16.8} {:>16.8} {:>16.8} {:>16.8e}",
            rec.iter,
            rec.params.a0,
            rec.params.a1,
            rec.params.a2,
            rec.chi2,
            rec.step_norm
        );
    }
    println!();
}

fn print_fit_summary(data: &[Observation], p: Parameters) {
    let chi2 = compute_chi2(data, p);

    println!("Final Parameter Estimate");
    println!("========================");
    println!("a0      = {:>.10}", p.a0);
    println!("a1      = {:>.10}", p.a1);
    println!("a2      = {:>.10}", p.a2);
    println!("chi^2   = {:>.10}", chi2);
    println!();
}

fn print_pointwise_diagnostics(data: &[Observation], p: Parameters) {
    println!("Pointwise Diagnostics");
    println!("=====================");
    println!(
        "{:>4} {:>12} {:>12} {:>14} {:>14} {:>14}",
        "i", "x_i", "y_i", "y_hat", "weighted r_i", "weighted r_i^2"
    );
    println!("{}", "-".repeat(86));

    for (i, obs) in data.iter().enumerate() {
        let y_hat = model(obs.x, p);
        let r = residual(*obs, p);
        println!(
            "{:>4} {:>12.6} {:>12.6} {:>14.8} {:>14.8} {:>14.8}",
            i, obs.x, obs.y, y_hat, r, r * r
        );
    }
    println!();
}

fn main() {
    let data = vec![
        Observation { x: 0.0, y: 3.5200, sigma: 0.0800 },
        Observation { x: 0.5, y: 3.0140, sigma: 0.0800 },
        Observation { x: 1.0, y: 2.6040, sigma: 0.0700 },
        Observation { x: 1.5, y: 2.2660, sigma: 0.0700 },
        Observation { x: 2.0, y: 1.9960, sigma: 0.0700 },
        Observation { x: 2.5, y: 1.7790, sigma: 0.0800 },
        Observation { x: 3.0, y: 1.5910, sigma: 0.0800 },
        Observation { x: 3.5, y: 1.4540, sigma: 0.0900 },
        Observation { x: 4.0, y: 1.3450, sigma: 0.0900 },
    ];

    // Initial guess for the nonlinear parameters.
    let initial = Parameters {
        a0: 2.5,
        a1: -0.4,
        a2: 0.8,
    };

    let tol = 1.0e-10;
    let max_iters = 25;

    println!("Nonlinear Least Squares via Gauss-Newton Iteration");
    println!("==================================================");
    println!("This program fits an exponential model by repeatedly");
    println!("linearizing the residual vector, solving the normal");
    println!("equations for the Gauss-Newton step, and updating");
    println!("the parameter vector.");
    println!();

    print_input_data(&data);

    match gauss_newton_fit(&data, initial, tol, max_iters) {
        Ok((fit, history)) => {
            print_iteration_history(&history);
            print_fit_summary(&data, fit);
            print_pointwise_diagnostics(&data, fit);

            println!("Interpretation");
            println!("==============");
            println!("Each iteration replaces the nonlinear residual function");
            println!("by its local linear approximation and solves the resulting");
            println!("least-squares subproblem for the update delta. The overall");
            println!("nonlinear fit is therefore obtained as a sequence of");
            println!("linearized least-squares solves.");
        }
        Err(message) => {
            eprintln!("fit failed: {}", message);
            std::process::exit(1);
        }
    }
}
```

Program 15.6.1 demonstrates how the Gauss–Newton method transforms a nonlinear least-squares problem into a sequence of linear subproblems, each derived from a local linear approximation of the residual function. This reflects the central idea of Section 15.6.3: nonlinear optimization can be approached by repeatedly applying linear least-squares techniques within an iterative framework.

The convergence behavior observed in the iteration history illustrates the effectiveness of the method when the initial parameter estimate is reasonably close to the solution. The rapid reduction in the objective function and the decreasing norm of the update indicate that the linear approximation provides an accurate description of the residual in the neighborhood of the solution.

At the same time, the implementation highlights the computational structure of the method. Each iteration requires evaluation of the residual and Jacobian, formation of the normal equations, and solution of a linear system. These steps dominate the computational cost, particularly in large-scale problems, and motivate the use of efficient linear algebra techniques.

More broadly, this example emphasizes the interplay between nonlinear modeling and linear algebra. The Gauss–Newton method leverages the structure of least squares to provide an efficient and conceptually clear approach to nonlinear parameter estimation. This perspective forms the foundation for more advanced methods, such as the Levenberg–Marquardt algorithm, which introduces regularization to improve robustness when the linear approximation is less accurate.

## 15.6.4. Levenberg–Marquardt Method and Regularization

The Gauss–Newton method can fail when the problem is ill-conditioned or when the current iterate is far from the solution. In such situations, the linear approximation used in each step may be inaccurate, and the matrix $J^\top J$ may be poorly conditioned or nearly singular. As a result, the computed update can be excessively large or directed in a way that does not reduce the objective, leading to slow convergence or even divergence.

The Levenberg–Marquardt (LM) method stabilizes the iteration by introducing a damping term:

$$(J^\top J + \lambda D)\delta = -J^\top r \tag{15.6.12}$$

where $D$ is typically $\mathrm{diag}(J^\top J)$ or the identity matrix, and $\lambda > 0$ is a parameter that controls the amount of damping. The addition of this term modifies the system being solved, making it better conditioned and limiting the size of the update.

This formulation can be interpreted as a regularized version of the Gauss–Newton step. When the matrix $J^\top J$ is ill-conditioned, adding $\lambda D$ increases its diagonal dominance and reduces sensitivity to numerical errors. At the same time, the presence of $\lambda$ prevents the update from becoming too large, which is particularly important when the linear approximation is only valid in a small neighborhood around the current estimate.

The LM method interpolates between two limiting behaviors. For large values of $\lambda$, the matrix $J^\top J + \lambda D$ is dominated by the damping term, and the update direction approaches that of steepest descent. In this regime, the method takes cautious steps that prioritize stability over rapid convergence. For small values of $\lambda$, the influence of the damping term diminishes, and the method approaches the Gauss–Newton update, which typically converges more rapidly when the current estimate is close to the solution.

The parameter $\lambda$ is adjusted dynamically based on whether the proposed step reduces the objective function. If a step leads to a decrease in $\chi^2$, the parameter is reduced, allowing the method to move closer to the Gauss–Newton regime and take larger, more aggressive steps. If a step fails to reduce the objective, $\lambda$ is increased, resulting in a more conservative update that improves stability. This adaptive strategy allows the method to balance robustness and efficiency throughout the iteration.

A modern interpretation is that the Levenberg–Marquardt method solves a trust-region problem for the linearized model, with $\lambda$ controlling both the effective step size and the degree of regularization. In this view, the method selects updates that remain within a region where the linear approximation is reliable, adjusting the size of this region based on the observed behavior of the objective function.

Contemporary theory places LM within a broader class of regularized Newton-type methods and analyzes its convergence under various conditions (Fischer et al., 2024). These analyses show that the method inherits desirable properties from both gradient-based and second-order methods, making it one of the most widely used and effective techniques for solving nonlinear least-squares problems in practice.

### Rust Implementation

Following the discussion in Section 15.6.4 on the limitations of the Gauss–Newton method and the introduction of regularization through damping, Program 15.6.2 provides a practical implementation of the Levenberg–Marquardt method for nonlinear least-squares problems. In this formulation, the linearized system is stabilized by augmenting the matrix $J^\top J$ with a damping term, as described in Equation (15.6.12). This modification improves conditioning and ensures that the update remains well behaved even when the current parameter estimate is far from the solution. The program demonstrates how the damping parameter is adjusted dynamically based on the success of each trial step, allowing the method to transition smoothly between cautious gradient-like updates and the faster Gauss–Newton regime.

At the core of the implementation is the evaluation of the residual vector defined in Equation (15.6.9). The function `weighted_residual` computes the scaled difference between the observed data and the model prediction, incorporating the weighting through the standard deviations. This ensures that the contribution of each observation to the objective function is properly normalized according to its uncertainty.

The function `jacobian_row` constructs the rows of the Jacobian matrix $J(a)$, as introduced in Equation (15.6.9). It computes the partial derivatives of the residual with respect to the parameters, thereby capturing the local sensitivity of the model. These derivatives are essential for forming the linearized least-squares problem described in Equation (15.6.10).

The function `form_normal_equations` assembles the matrix $J^\top J$ and the vector $-J^\top r$ appearing in Equation (15.6.11). This step mirrors the Gauss–Newton formulation but serves as the basis for the regularized system. The function `add_lm_damping` then modifies this matrix by adding the damping term $\lambda D$, where $D$ is chosen as the diagonal of $J^\top J$. This corresponds directly to the stabilized system in Equation (15.6.12), improving conditioning and limiting the magnitude of the update.

The function `levenberg_marquardt_fit` implements the iterative procedure. At each iteration, a candidate update is computed by solving the damped system. The resulting parameters are used to evaluate a trial objective value, and the step is accepted only if it reduces $\chi^2$. This acceptance test governs the adaptive adjustment of the damping parameter: successful steps reduce $\lambda$, moving the method toward the Gauss–Newton regime, while unsuccessful steps increase $\lambda$, enforcing more conservative updates. This mechanism reflects the balance between stability and efficiency emphasized in the section.

Supporting functions such as `model`, `compute_chi2`, and the diagnostic routines structure the computation and provide detailed output. The iteration history records the evolution of the parameters, the objective function, and the damping parameter, offering insight into the behavior of the algorithm. The `main` function demonstrates the complete workflow, including initialization with a deliberately challenging starting point, iterative refinement, and reporting of results.

```rust
// Program 15.6.2. Nonlinear Least Squares via the Levenberg-Marquardt Method
//
// Problem statement:
// Fit the nonlinear model
//
//     y(x; a) = a0 * exp(a1 * x) + a2
//
// to weighted observations (x_i, y_i, sigma_i) by minimizing
//
//     chi^2(a) = sum_i [ (y_i - y(x_i; a)) / sigma_i ]^2.
//
// The program implements the Levenberg-Marquardt method described in
// Section 15.6.4. At each iteration it:
//
//   1. evaluates the residual vector r(a),
//   2. constructs the Jacobian J(a),
//   3. forms J^T J and -J^T r,
//   4. solves the damped system
//        (J^T J + lambda D) delta = -J^T r,
//      where D is the diagonal of J^T J,
//   5. accepts or rejects the trial step based on chi^2 reduction,
//   6. adjusts lambda adaptively.
//
// This illustrates how damping regularizes the Gauss-Newton step and
// improves robustness when the linearized model is not yet reliable.

#[derive(Clone, Copy, Debug)]
struct Observation {
    x: f64,
    y: f64,
    sigma: f64,
}

#[derive(Clone, Copy, Debug)]
struct Parameters {
    a0: f64,
    a1: f64,
    a2: f64,
}

#[derive(Clone, Copy, Debug)]
struct IterationRecord {
    iter: usize,
    params: Parameters,
    chi2: f64,
    lambda: f64,
    step_norm: f64,
    accepted: bool,
}

fn validate_data(data: &[Observation]) -> Result<(), String> {
    if data.len() < 3 {
        return Err("at least three observations are required".to_string());
    }

    for (i, obs) in data.iter().enumerate() {
        if !obs.x.is_finite() || !obs.y.is_finite() || !obs.sigma.is_finite() {
            return Err(format!("observation {} contains a non-finite value", i));
        }
        if obs.sigma <= 0.0 {
            return Err(format!("observation {} has nonpositive sigma", i));
        }
    }

    Ok(())
}

fn model(x: f64, p: Parameters) -> f64 {
    p.a0 * (p.a1 * x).exp() + p.a2
}

fn weighted_residual(obs: Observation, p: Parameters) -> f64 {
    (obs.y - model(obs.x, p)) / obs.sigma
}

fn jacobian_row(obs: Observation, p: Parameters) -> [f64; 3] {
    let e = (p.a1 * obs.x).exp();

    // r_i(a) = (y_i - f(x_i; a)) / sigma_i
    // so dr_i/da_k = -(1/sigma_i) * df/da_k
    let dr_da0 = -e / obs.sigma;
    let dr_da1 = -(p.a0 * obs.x * e) / obs.sigma;
    let dr_da2 = -1.0 / obs.sigma;

    [dr_da0, dr_da1, dr_da2]
}

fn compute_chi2(data: &[Observation], p: Parameters) -> f64 {
    data.iter()
        .map(|&obs| {
            let r = weighted_residual(obs, p);
            r * r
        })
        .sum()
}

fn form_normal_equations(data: &[Observation], p: Parameters) -> ([[f64; 3]; 3], [f64; 3], f64) {
    let mut jtj = [[0.0_f64; 3]; 3];
    let mut minus_jtr = [0.0_f64; 3];
    let mut chi2 = 0.0;

    for &obs in data {
        let r = weighted_residual(obs, p);
        let j = jacobian_row(obs, p);

        chi2 += r * r;

        for i in 0..3 {
            minus_jtr[i] -= j[i] * r;
            for k in 0..3 {
                jtj[i][k] += j[i] * j[k];
            }
        }
    }

    (jtj, minus_jtr, chi2)
}

fn add_lm_damping(mut jtj: [[f64; 3]; 3], lambda: f64) -> [[f64; 3]; 3] {
    for i in 0..3 {
        let d = if jtj[i][i].abs() > 1.0e-15 { jtj[i][i] } else { 1.0 };
        jtj[i][i] += lambda * d;
    }
    jtj
}

fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Result<[f64; 3], String> {
    for col in 0..3 {
        let mut pivot_row = col;
        let mut pivot_abs = a[col][col].abs();

        for row in (col + 1)..3 {
            let cand = a[row][col].abs();
            if cand > pivot_abs {
                pivot_abs = cand;
                pivot_row = row;
            }
        }

        if pivot_abs <= 1.0e-14 {
            return Err("linear system is singular or ill-conditioned".to_string());
        }

        if pivot_row != col {
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }

        let pivot = a[col][col];
        for row in (col + 1)..3 {
            let factor = a[row][col] / pivot;
            for k in col..3 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = [0.0_f64; 3];
    for i in (0..3).rev() {
        let mut sum = b[i];
        for k in (i + 1)..3 {
            sum -= a[i][k] * x[k];
        }
        if a[i][i].abs() <= 1.0e-14 {
            return Err("back substitution failed due to a zero pivot".to_string());
        }
        x[i] = sum / a[i][i];
    }

    Ok(x)
}

fn step_norm(delta: [f64; 3]) -> f64 {
    (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
}

fn add_step(p: Parameters, delta: [f64; 3]) -> Parameters {
    Parameters {
        a0: p.a0 + delta[0],
        a1: p.a1 + delta[1],
        a2: p.a2 + delta[2],
    }
}

fn levenberg_marquardt_fit(
    data: &[Observation],
    initial: Parameters,
    initial_lambda: f64,
    tol: f64,
    max_iters: usize,
) -> Result<(Parameters, Vec<IterationRecord>), String> {
    validate_data(data)?;

    if initial_lambda <= 0.0 || !initial_lambda.is_finite() {
        return Err("initial lambda must be positive and finite".to_string());
    }

    let mut p = initial;
    let mut lambda = initial_lambda;
    let mut history = Vec::new();

    for iter in 0..max_iters {
        let (jtj, minus_jtr, chi2_current) = form_normal_equations(data, p);
        let damped = add_lm_damping(jtj, lambda);
        let delta = solve_3x3(damped, minus_jtr)?;
        let delta_norm = step_norm(delta);

        let trial = add_step(p, delta);
        if !trial.a0.is_finite() || !trial.a1.is_finite() || !trial.a2.is_finite() {
            return Err("iteration produced a non-finite parameter value".to_string());
        }

        let chi2_trial = compute_chi2(data, trial);
        let accepted = chi2_trial < chi2_current;

        history.push(IterationRecord {
            iter,
            params: p,
            chi2: chi2_current,
            lambda,
            step_norm: delta_norm,
            accepted,
        });

        if accepted {
            p = trial;
            lambda = (0.1 * lambda).max(1.0e-15);

            if delta_norm < tol {
                let final_chi2 = compute_chi2(data, p);
                history.push(IterationRecord {
                    iter: iter + 1,
                    params: p,
                    chi2: final_chi2,
                    lambda,
                    step_norm: 0.0,
                    accepted: true,
                });
                return Ok((p, history));
            }
        } else {
            lambda = (10.0 * lambda).min(1.0e15);
        }
    }

    let final_chi2 = compute_chi2(data, p);
    history.push(IterationRecord {
        iter: max_iters,
        params: p,
        chi2: final_chi2,
        lambda,
        step_norm: 0.0,
        accepted: true,
    });

    Ok((p, history))
}

fn print_input_data(data: &[Observation]) {
    println!("Input Data");
    println!("==========");
    println!(
        "{:>4} {:>12} {:>12} {:>12}",
        "i", "x_i", "y_i", "sigma_i"
    );
    println!("{}", "-".repeat(46));

    for (i, obs) in data.iter().enumerate() {
        println!(
            "{:>4} {:>12.6} {:>12.6} {:>12.6}",
            i, obs.x, obs.y, obs.sigma
        );
    }
    println!();
}

fn print_iteration_history(history: &[IterationRecord]) {
    println!("Levenberg-Marquardt Iteration History");
    println!("====================================");
    println!(
        "{:>4} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14} {:>10}",
        "k", "a0", "a1", "a2", "chi^2", "lambda", "||delta||_2", "accept"
    );
    println!("{}", "-".repeat(122));

    for rec in history {
        println!(
            "{:>4} {:>14.8} {:>14.8} {:>14.8} {:>14.8} {:>14.6e} {:>14.6e} {:>10}",
            rec.iter,
            rec.params.a0,
            rec.params.a1,
            rec.params.a2,
            rec.chi2,
            rec.lambda,
            rec.step_norm,
            if rec.accepted { "yes" } else { "no" }
        );
    }
    println!();
}

fn print_fit_summary(data: &[Observation], p: Parameters) {
    let chi2 = compute_chi2(data, p);

    println!("Final Parameter Estimate");
    println!("========================");
    println!("a0      = {:>.10}", p.a0);
    println!("a1      = {:>.10}", p.a1);
    println!("a2      = {:>.10}", p.a2);
    println!("chi^2   = {:>.10}", chi2);
    println!();
}

fn print_pointwise_diagnostics(data: &[Observation], p: Parameters) {
    println!("Pointwise Diagnostics");
    println!("=====================");
    println!(
        "{:>4} {:>12} {:>12} {:>14} {:>16} {:>18}",
        "i", "x_i", "y_i", "y_hat", "weighted r_i", "(weighted r_i)^2"
    );
    println!("{}", "-".repeat(98));

    for (i, obs) in data.iter().enumerate() {
        let y_hat = model(obs.x, p);
        let r = weighted_residual(*obs, p);
        println!(
            "{:>4} {:>12.6} {:>12.6} {:>14.8} {:>16.8} {:>18.8}",
            i, obs.x, obs.y, y_hat, r, r * r
        );
    }
    println!();
}

fn main() {
    let data = vec![
        Observation { x: 0.0, y: 3.5200, sigma: 0.0800 },
        Observation { x: 0.5, y: 3.0140, sigma: 0.0800 },
        Observation { x: 1.0, y: 2.6040, sigma: 0.0700 },
        Observation { x: 1.5, y: 2.2660, sigma: 0.0700 },
        Observation { x: 2.0, y: 1.9960, sigma: 0.0700 },
        Observation { x: 2.5, y: 1.7790, sigma: 0.0800 },
        Observation { x: 3.0, y: 1.5910, sigma: 0.0800 },
        Observation { x: 3.5, y: 1.4540, sigma: 0.0900 },
        Observation { x: 4.0, y: 1.3450, sigma: 0.0900 },
    ];

    // A deliberately poorer initial guess than in the Gauss-Newton example.
    let initial = Parameters {
        a0: 1.2,
        a1: -1.2,
        a2: 1.8,
    };

    let initial_lambda = 1.0e-2;
    let tol = 1.0e-10;
    let max_iters = 40;

    println!("Nonlinear Least Squares via Levenberg-Marquardt");
    println!("===============================================");
    println!("This program fits an exponential model using a");
    println!("damped Gauss-Newton iteration with adaptive");
    println!("regularization.");
    println!();

    print_input_data(&data);

    match levenberg_marquardt_fit(&data, initial, initial_lambda, tol, max_iters) {
        Ok((fit, history)) => {
            print_iteration_history(&history);
            print_fit_summary(&data, fit);
            print_pointwise_diagnostics(&data, fit);

            println!("Interpretation");
            println!("==============");
            println!("The damping parameter lambda regularizes the");
            println!("linearized least-squares system and is adjusted");
            println!("according to whether a trial step decreases chi^2.");
            println!("For large lambda the method behaves cautiously,");
            println!("while for small lambda it approaches Gauss-Newton.");
        }
        Err(message) => {
            eprintln!("fit failed: {}", message);
            std::process::exit(1);
        }
    }
}
```

Program 15.6.2 demonstrates how the Levenberg–Marquardt method enhances the Gauss–Newton approach by introducing regularization into the linearized subproblem. By augmenting the system matrix with a damping term, the method improves numerical stability and prevents excessively large updates when the linear approximation is unreliable.

The iteration history illustrates the adaptive nature of the method. Early in the computation, large values of the damping parameter enforce cautious steps that reduce the objective even when the initial guess is poor. As the solution is approached, the damping parameter decreases, allowing the method to recover the faster convergence behavior characteristic of Gauss–Newton iteration.

This example highlights a central principle in nonlinear optimization: robustness can often be achieved by combining local linearization with controlled regularization. The Levenberg–Marquardt method embodies this idea by blending gradient-based and second-order techniques, making it one of the most widely used algorithms for nonlinear least-squares problems in practical applications.

## 15.6.5. Practical Issues: Derivatives, Large-Scale Problems, and Robustness

Practical implementation of nonlinear least-squares methods involves several additional considerations beyond the basic algorithmic framework. These include the computation of derivatives, the treatment of large-scale models, and the need for robustness in the presence of imperfect data. Each of these aspects plays a critical role in determining the efficiency and reliability of the solution.

### Derivative Computation

Efficient implementation requires accurate evaluation of Jacobians. Since the Gauss–Newton and Levenberg–Marquardt methods depend directly on the Jacobian matrix, the quality of the derivative information strongly influences both convergence speed and numerical stability. Analytic derivatives are ideal, as they provide exact expressions and often lead to the most efficient implementations.

In practice, however, analytic derivatives may be difficult or time-consuming to derive, especially for complex models. Modern implementations therefore often rely on automatic differentiation (AD), which computes derivatives programmatically by applying the chain rule to the underlying code. While AD can provide high accuracy and reduce development effort, it is not without limitations. In particular, it may fail to capture the intended derivatives when applied to numerical codes that involve discretizations, iterative solvers, or branching logic. In such cases, the derivative of the implemented algorithm may differ from the derivative of the underlying mathematical model. A recent taxonomy highlights these pitfalls and emphasizes the importance of carefully designing programs to ensure that differentiation produces meaningful results (Hückelheim et al., 2024).

### Large-Scale and PDE-Based Models

In many applications, the model $f(x; a)$ is not given by a simple formula but is computed by solving a partial differential equation. In this setting, nonlinear least squares becomes a parameter identification problem:

$$\min_a |y - f(a)|^2 + \text{regularization} \tag{15.6.13}$$

Here, each evaluation of the model may require the solution of a large-scale numerical problem, making the overall computation significantly more expensive.

When Gauss–Newton methods are applied to such problems, the resulting linearized subproblems often have a saddle-point structure. This structure arises from the coupling between state variables and parameters and requires specialized numerical techniques for efficient solution. Advanced solvers and preconditioning strategies are therefore essential to achieve acceptable performance and scalability (Blechta and Ernst, 2024). The effectiveness of the overall method depends not only on the optimization algorithm but also on the efficiency of the underlying linear algebra.

### Stochastic and Subsampled Methods

For large datasets, the objective function may consist of a sum of many individual contributions:

$$F(a) = \sum_{i=1}^{N} f_i(a) \tag{15.6.14}$$

In such cases, evaluating the full objective and its derivatives at every iteration can be prohibitively expensive.

Recent work addresses this issue by developing stochastic and subsampled variants of nonlinear least-squares methods, including adaptations of the Levenberg–Marquardt algorithm. These methods approximate the objective and Jacobian using subsets of the data, thereby reducing computational cost per iteration. By carefully controlling the sampling process and incorporating variance reduction techniques, it is possible to maintain convergence guarantees while significantly improving efficiency (Xing et al., 2023; Shao and Fan, 2024). These approaches are particularly important in data-intensive applications where the number of observations is very large.

### Robust Nonlinear Least Squares

Least squares is sensitive to outliers, since the squared loss assigns large weight to large residuals. In nonlinear settings, this sensitivity can lead to poor parameter estimates and unstable convergence behavior if the data contain even a small number of anomalous observations.

Robust alternatives address this issue by replacing the squared loss with functions (\\rho(r_i)) that grow more slowly for large residuals. This modification reduces the influence of outliers while preserving sensitivity to the majority of the data. The resulting optimization problem can often be solved using iteratively reweighted least squares, in which a sequence of weighted linear least-squares problems is constructed. At each iteration, the weights are updated based on the current residuals, effectively downweighting observations that deviate strongly from the model. This approach connects nonlinear fitting back to the weighted linear least-squares framework developed earlier (Loh, 2025).

This subsection highlights that successful application of nonlinear least squares depends not only on the underlying mathematical formulation but also on careful attention to implementation details, model structure, and data characteristics.

### Rust Implementation

Following the discussion in Section 15.6.5 on practical issues in nonlinear least-squares computation, particularly robustness and derivative evaluation, Program 15.6.3 provides a practical implementation of robust nonlinear least squares using iteratively reweighted least squares (IRLS). In many real-world problems, data may contain outliers that can significantly distort parameter estimates when using the standard squared-loss formulation. To address this, the program replaces the classical objective with a robust alternative and updates weights dynamically based on the current residuals. In addition, the Jacobian is computed using finite-difference approximation, reflecting a common strategy when analytic derivatives are unavailable or difficult to derive. This implementation demonstrates how robustness, derivative approximation, and nonlinear optimization can be integrated within a unified computational framework.

At the core of the implementation is the evaluation of the residual vector defined in Equation (15.6.9). The function `weighted_residual` computes the scaled residual for each observation by dividing the difference between the observed value and the model prediction by the corresponding standard deviation. This ensures that the contribution of each data point to the objective function is properly normalized.

The function `finite_difference_jacobian_row` implements the Jacobian matrix $J(a)$ introduced in Equation (15.6.9), but instead of using analytic derivatives, it approximates the partial derivatives using central finite differences. For each parameter, a small perturbation is applied, and the derivative is estimated from the resulting change in the residual. This approach reflects the practical considerations discussed in the section, where derivative computation may rely on numerical approximation when analytic expressions are unavailable.

The robust formulation is introduced through the functions `huber_weight` and `robust_rho_huber`, which implement a Huber-type loss function. Instead of minimizing the squared residuals directly, the method assigns weights based on the magnitude of the residuals. For small residuals, the weight remains unity, preserving the behavior of least squares, while for large residuals, the weight decreases, reducing the influence of outliers. This corresponds to the robust nonlinear least-squares framework described in Section 15.6.5.

The function `form_weighted_normal_equations` constructs the weighted system corresponding to the linearized subproblem. Using the weights derived from the robust loss, it accumulates the matrix $J^\top J$ and the vector $-J^\top r$ in a manner analogous to Equation (15.6.11), but with each contribution scaled appropriately. This step shows how the nonlinear robust problem is reduced to a sequence of weighted linear least-squares problems.

The iterative procedure is implemented in the function `robust_irls_fit`. At each iteration, the weights are updated based on the current residuals, and a new linearized system is solved to obtain the parameter update. The process continues until the update becomes sufficiently small, indicating convergence. This reflects the IRLS strategy, where robustness is achieved through repeated reweighting and solution of linearized subproblems.

Supporting functions such as `model`, `compute_robust_objective`, and the diagnostic routines organize the computation and present detailed output. The iteration history records the evolution of the parameters and the robust objective, while the pointwise diagnostics display residuals, weights, and outlier flags, providing insight into how the method treats different observations.

```rust
// Program 15.6.3. Robust Nonlinear Least Squares via IRLS with Finite-Difference Jacobian
//
// Problem statement:
// Fit the nonlinear model
//
//     y(x; a) = a0 * exp(a1 * x) + a2
//
// to noisy observations (x_i, y_i, sigma_i), allowing for the presence of
// outliers. The program minimizes a robustified nonlinear least-squares
// objective by using iteratively reweighted least squares (IRLS).
//
// Main ideas:
//   1. Residuals are scaled by sigma_i.
//   2. A finite-difference Jacobian approximates parameter sensitivities.
//   3. Huber-style robust weights are computed from the current residuals.
//   4. A weighted Gauss-Newton step is solved at each iteration.
//   5. Outlying observations are downweighted automatically.
//
// This implementation reflects the practical issues highlighted in
// Section 15.6.5: derivative approximation, robustness, and the reduction
// of nonlinear fitting to a sequence of weighted linearized subproblems.

#[derive(Clone, Copy, Debug)]
struct Observation {
    x: f64,
    y: f64,
    sigma: f64,
}

#[derive(Clone, Copy, Debug)]
struct Parameters {
    a0: f64,
    a1: f64,
    a2: f64,
}

#[derive(Clone, Copy, Debug)]
struct IterationRecord {
    iter: usize,
    params: Parameters,
    objective: f64,
    step_norm: f64,
    max_abs_weighted_residual: f64,
}

fn validate_data(data: &[Observation]) -> Result<(), String> {
    if data.len() < 3 {
        return Err("at least three observations are required".to_string());
    }

    for (i, obs) in data.iter().enumerate() {
        if !obs.x.is_finite() || !obs.y.is_finite() || !obs.sigma.is_finite() {
            return Err(format!("observation {} contains a non-finite value", i));
        }
        if obs.sigma <= 0.0 {
            return Err(format!("observation {} has nonpositive sigma", i));
        }
    }

    Ok(())
}

fn model(x: f64, p: Parameters) -> f64 {
    p.a0 * (p.a1 * x).exp() + p.a2
}

fn weighted_residual(obs: Observation, p: Parameters) -> f64 {
    (obs.y - model(obs.x, p)) / obs.sigma
}

fn perturb_parameter(p: Parameters, index: usize, h: f64) -> Parameters {
    match index {
        0 => Parameters { a0: p.a0 + h, ..p },
        1 => Parameters { a1: p.a1 + h, ..p },
        2 => Parameters { a2: p.a2 + h, ..p },
        _ => p,
    }
}

fn finite_difference_jacobian_row(obs: Observation, p: Parameters) -> [f64; 3] {
    let mut row = [0.0_f64; 3];

    for k in 0..3 {
        let base = match k {
            0 => p.a0.abs(),
            1 => p.a1.abs(),
            2 => p.a2.abs(),
            _ => 1.0,
        };

        let h = 1.0e-6 * base.max(1.0);
        let p_plus = perturb_parameter(p, k, h);
        let p_minus = perturb_parameter(p, k, -h);

        let r_plus = weighted_residual(obs, p_plus);
        let r_minus = weighted_residual(obs, p_minus);

        row[k] = (r_plus - r_minus) / (2.0 * h);
    }

    row
}

fn huber_weight(r: f64, kappa: f64) -> f64 {
    let a = r.abs();
    if a <= kappa {
        1.0
    } else {
        kappa / a
    }
}

fn robust_rho_huber(r: f64, kappa: f64) -> f64 {
    let a = r.abs();
    if a <= kappa {
        0.5 * a * a
    } else {
        kappa * (a - 0.5 * kappa)
    }
}

fn compute_robust_objective(data: &[Observation], p: Parameters, kappa: f64) -> f64 {
    data.iter()
        .map(|&obs| robust_rho_huber(weighted_residual(obs, p), kappa))
        .sum()
}

fn form_weighted_normal_equations(
    data: &[Observation],
    p: Parameters,
    kappa: f64,
) -> ([[f64; 3]; 3], [f64; 3], f64, f64) {
    let mut jtj = [[0.0_f64; 3]; 3];
    let mut minus_jtr = [0.0_f64; 3];
    let mut objective = 0.0;
    let mut max_abs_residual: f64 = 0.0;

    for &obs in data {
        let r = weighted_residual(obs, p);
        let j = finite_difference_jacobian_row(obs, p);
        let w = huber_weight(r, kappa);
        let sqrt_w = w.sqrt();

        objective += robust_rho_huber(r, kappa);
        max_abs_residual = max_abs_residual.max(r.abs());

        for i in 0..3 {
            let ji = sqrt_w * j[i];
            let ri = sqrt_w * r;
            minus_jtr[i] -= ji * ri;

            for k in 0..3 {
                let jk = sqrt_w * j[k];
                jtj[i][k] += ji * jk;
            }
        }
    }

    (jtj, minus_jtr, objective, max_abs_residual)
}

fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Result<[f64; 3], String> {
    for col in 0..3 {
        let mut pivot_row = col;
        let mut pivot_abs = a[col][col].abs();

        for row in (col + 1)..3 {
            let cand = a[row][col].abs();
            if cand > pivot_abs {
                pivot_abs = cand;
                pivot_row = row;
            }
        }

        if pivot_abs <= 1.0e-14 {
            return Err("linearized system is singular or ill-conditioned".to_string());
        }

        if pivot_row != col {
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }

        let pivot = a[col][col];
        for row in (col + 1)..3 {
            let factor = a[row][col] / pivot;
            for k in col..3 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = [0.0_f64; 3];
    for i in (0..3).rev() {
        let mut sum = b[i];
        for k in (i + 1)..3 {
            sum -= a[i][k] * x[k];
        }
        if a[i][i].abs() <= 1.0e-14 {
            return Err("back substitution failed due to a zero pivot".to_string());
        }
        x[i] = sum / a[i][i];
    }

    Ok(x)
}

fn add_step(p: Parameters, delta: [f64; 3]) -> Parameters {
    Parameters {
        a0: p.a0 + delta[0],
        a1: p.a1 + delta[1],
        a2: p.a2 + delta[2],
    }
}

fn step_norm(delta: [f64; 3]) -> f64 {
    (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
}

fn robust_irls_fit(
    data: &[Observation],
    initial: Parameters,
    kappa: f64,
    tol: f64,
    max_iters: usize,
) -> Result<(Parameters, Vec<IterationRecord>), String> {
    validate_data(data)?;

    if kappa <= 0.0 || !kappa.is_finite() {
        return Err("kappa must be positive and finite".to_string());
    }

    let mut p = initial;
    let mut history = Vec::new();

    for iter in 0..max_iters {
        let (jtj, minus_jtr, objective, max_abs_residual) =
            form_weighted_normal_equations(data, p, kappa);

        let delta = solve_3x3(jtj, minus_jtr)?;
        let dn = step_norm(delta);

        history.push(IterationRecord {
            iter,
            params: p,
            objective,
            step_norm: dn,
            max_abs_weighted_residual: max_abs_residual,
        });

        let next = add_step(p, delta);
        if !next.a0.is_finite() || !next.a1.is_finite() || !next.a2.is_finite() {
            return Err("iteration produced a non-finite parameter value".to_string());
        }

        p = next;

        if dn < tol {
            let final_objective = compute_robust_objective(data, p, kappa);
            let final_max_res = data
                .iter()
                .map(|&obs| weighted_residual(obs, p).abs())
                .fold(0.0_f64, f64::max);

            history.push(IterationRecord {
                iter: iter + 1,
                params: p,
                objective: final_objective,
                step_norm: 0.0,
                max_abs_weighted_residual: final_max_res,
            });
            return Ok((p, history));
        }
    }

    let final_objective = compute_robust_objective(data, p, kappa);
    let final_max_res = data
        .iter()
        .map(|&obs| weighted_residual(obs, p).abs())
        .fold(0.0_f64, f64::max);

    history.push(IterationRecord {
        iter: max_iters,
        params: p,
        objective: final_objective,
        step_norm: 0.0,
        max_abs_weighted_residual: final_max_res,
    });

    Ok((p, history))
}

fn print_input_data(data: &[Observation]) {
    println!("Input Data");
    println!("==========");
    println!(
        "{:>4} {:>12} {:>12} {:>12}",
        "i", "x_i", "y_i", "sigma_i"
    );
    println!("{}", "-".repeat(46));

    for (i, obs) in data.iter().enumerate() {
        println!(
            "{:>4} {:>12.6} {:>12.6} {:>12.6}",
            i, obs.x, obs.y, obs.sigma
        );
    }
    println!();
}

fn print_iteration_history(history: &[IterationRecord]) {
    println!("Robust IRLS Iteration History");
    println!("=============================");
    println!(
        "{:>4} {:>14} {:>14} {:>14} {:>16} {:>16} {:>16}",
        "k", "a0", "a1", "a2", "robust obj", "||delta||_2", "max |wr|"
    );
    println!("{}", "-".repeat(114));

    for rec in history {
        println!(
            "{:>4} {:>14.8} {:>14.8} {:>14.8} {:>16.8} {:>16.8e} {:>16.8}",
            rec.iter,
            rec.params.a0,
            rec.params.a1,
            rec.params.a2,
            rec.objective,
            rec.step_norm,
            rec.max_abs_weighted_residual
        );
    }
    println!();
}

fn print_fit_summary(data: &[Observation], p: Parameters, kappa: f64) {
    let robust_obj = compute_robust_objective(data, p, kappa);
    let classical_chi2: f64 = data
        .iter()
        .map(|&obs| {
            let r = weighted_residual(obs, p);
            r * r
        })
        .sum();

    println!("Final Parameter Estimate");
    println!("========================");
    println!("a0                = {:>.10}", p.a0);
    println!("a1                = {:>.10}", p.a1);
    println!("a2                = {:>.10}", p.a2);
    println!("robust objective  = {:>.10}", robust_obj);
    println!("classical chi^2   = {:>.10}", classical_chi2);
    println!();
}

fn print_pointwise_diagnostics(data: &[Observation], p: Parameters, kappa: f64) {
    println!("Pointwise Diagnostics");
    println!("=====================");
    println!(
        "{:>4} {:>12} {:>12} {:>14} {:>16} {:>14} {:>12}",
        "i", "x_i", "y_i", "y_hat", "weighted r_i", "Huber w_i", "outlier?"
    );
    println!("{}", "-".repeat(108));

    for (i, obs) in data.iter().enumerate() {
        let y_hat = model(obs.x, p);
        let wr = weighted_residual(*obs, p);
        let w = huber_weight(wr, kappa);
        let flag = if wr.abs() > kappa { "yes" } else { "no" };

        println!(
            "{:>4} {:>12.6} {:>12.6} {:>14.8} {:>16.8} {:>14.8} {:>12}",
            i, obs.x, obs.y, y_hat, wr, w, flag
        );
    }
    println!();
}

fn main() {
    // Dataset with one noticeable outlying observation.
    let data = vec![
        Observation { x: 0.0, y: 3.5200, sigma: 0.0800 },
        Observation { x: 0.5, y: 3.0140, sigma: 0.0800 },
        Observation { x: 1.0, y: 2.6040, sigma: 0.0700 },
        Observation { x: 1.5, y: 2.2660, sigma: 0.0700 },
        Observation { x: 2.0, y: 2.3500, sigma: 0.0700 }, // intentional outlier
        Observation { x: 2.5, y: 1.7790, sigma: 0.0800 },
        Observation { x: 3.0, y: 1.5910, sigma: 0.0800 },
        Observation { x: 3.5, y: 1.4540, sigma: 0.0900 },
        Observation { x: 4.0, y: 1.3450, sigma: 0.0900 },
    ];

    let initial = Parameters {
        a0: 2.4,
        a1: -0.35,
        a2: 0.9,
    };

    let huber_kappa = 1.5;
    let tol = 1.0e-10;
    let max_iters = 30;

    println!("Robust Nonlinear Least Squares via IRLS");
    println!("=======================================");
    println!("This program fits an exponential model using");
    println!("iteratively reweighted least squares with");
    println!("Huber-type robust weights and a finite-");
    println!("difference Jacobian.");
    println!();

    print_input_data(&data);

    match robust_irls_fit(&data, initial, huber_kappa, tol, max_iters) {
        Ok((fit, history)) => {
            print_iteration_history(&history);
            print_fit_summary(&data, fit, huber_kappa);
            print_pointwise_diagnostics(&data, fit, huber_kappa);

            println!("Interpretation");
            println!("==============");
            println!("Large weighted residuals are automatically downweighted");
            println!("through the Huber rule, so the fitted parameters are");
            println!("influenced less strongly by outlying observations than");
            println!("in ordinary nonlinear least squares.");
        }
        Err(message) => {
            eprintln!("fit failed: {}", message);
            std::process::exit(1);
        }
    }
}
```

Program 15.6.3 demonstrates how robust nonlinear least squares can be implemented by combining iterative linearization with adaptive weighting. By replacing the classical squared-loss objective with a robust alternative, the method reduces the influence of outliers while preserving sensitivity to the majority of the data. This modification is particularly important in practical applications, where data imperfections are unavoidable.

The implementation also highlights the role of derivative approximation in real-world problems. By using finite differences to compute the Jacobian, the program avoids the need for analytic derivatives, making it applicable to a wide range of models. At the same time, this approach introduces additional computational cost and potential numerical error, illustrating the trade-offs discussed in Section 15.6.5.

More broadly, this example reinforces the central idea that nonlinear least squares is best viewed as a sequence of linearized subproblems. Even in the presence of robustness modifications and approximate derivatives, the structure of the computation remains closely tied to linear least-squares methods. This connection allows established linear algebra techniques to be leveraged effectively, while accommodating the complexities of real-world data and models.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/cTzOgpN8SOeoVu0XR36u.6","tags":[]}

# 15.7. Confidence Limits on Estimated Model Parameters

A model is only useful if we can quantify how uncertain we are about its conclusions. Parameter estimation without uncertainty quantification is incomplete, since point estimates alone do not indicate reliability, sensitivity, or predictive credibility. Even when a model fits the data well, it is essential to understand how much the estimated parameters might vary under perturbations of the data and how this variability affects subsequent predictions. This section develops the mathematical and computational framework for constructing confidence limits on estimated parameters.

The need for uncertainty quantification arises naturally from the presence of noise and variability in observed data. Measurements are subject to errors, models are approximations of reality, and numerical computations are performed in finite precision. As a result, the parameters obtained from fitting procedures should be interpreted not as exact quantities, but as estimates that carry an associated level of uncertainty. Quantifying this uncertainty allows one to assess the robustness of conclusions and to compare competing models in a principled way.

Two complementary viewpoints underlie uncertainty quantification:

*Frequentist*: parameters are fixed but unknown, and uncertainty arises from variability in the observed data. In this framework, confidence limits are constructed by analyzing how parameter estimates would vary under repeated sampling of the data.

*Bayesian*: parameters are treated as random variables, and uncertainty is described by posterior distributions that combine prior information with observed data. In this setting, uncertainty quantification is expressed directly in probabilistic terms through distributions over parameter values.

Despite this philosophical difference, both approaches share a common computational core: likelihood-based optimization and linear algebra. In both cases, the analysis begins with a model for the data and an associated objective function derived from the likelihood. The behavior of this objective near its minimum determines how uncertainty is distributed in the parameter space.

The geometry of uncertainty is governed locally by the curvature of the log-likelihood. Near the optimal solution, the objective function can often be approximated by a quadratic form, leading to ellipsoidal confidence regions in parameter space. These regions are described by matrices derived from the second derivatives or approximations thereof, linking uncertainty quantification directly to the linear algebra structures developed in earlier sections.

Thus, the study of confidence limits builds on the same mathematical foundation as least squares and nonlinear optimization, extending it to provide a quantitative measure of reliability and interpretability in parameter estimation.

## 15.7.1. Definition of Confidence Limits and Practical Considerations

A confidence interval for a scalar parameter component $\theta_j$ is a random interval,

$$[L(D), U(D)] \tag{15.7.1}$$

such that, under repeated sampling,

$$\Pr\bigl(\theta_j \in [L(D), U(D)]\bigr) = 1 - \alpha \tag{15.7.2}$$

The notation emphasizes that the interval depends on the observed data $D$, and therefore varies from one sample to another. The parameter $\theta_j$ itself is fixed, while the randomness arises from the data-generating process. The interpretation is that, over many repetitions of the experiment, a proportion $1 - \alpha$ of the constructed intervals will contain the true parameter value.

For vector parameters $\theta \in \mathbb{R}^d$, the concept generalizes to confidence regions. A set $C(D)$ is called a confidence region if:

$$\Pr(\theta \in C(D)) = 1 - \alpha \tag{15.7.3}$$

In this case, the uncertainty is represented by a region in parameter space rather than an interval, and the geometry of this region reflects both the correlations between parameters and the structure of the underlying model.

In practice, these probabilities are rarely computed exactly. Exact construction of confidence intervals or regions would require complete knowledge of the sampling distribution of the estimator, which is often unavailable or difficult to characterize for realistic models. Instead, one relies on approximations that are derived under simplifying assumptions or asymptotic arguments.

The accuracy of these approximations depends on several factors. The sample size (n) plays a central role, since many theoretical results rely on large-sample behavior. For small samples, approximations may be inaccurate or biased. Model nonlinearity is another important factor, as nonlinear dependence on parameters can distort the shape of the objective function and invalidate simple quadratic approximations. The correctness of the noise model is also critical, since confidence limits are typically derived under specific distributional assumptions, such as Gaussian noise.

Additional complications arise from the presence of outliers or dependence among observations. Outliers can distort parameter estimates and inflate or deflate uncertainty estimates, while dependence between observations violates assumptions of independence that underlie many standard results. Finally, numerical conditioning of the optimization and linear algebra procedures can influence the computed intervals, since ill-conditioned problems may lead to unstable estimates and unreliable uncertainty quantification.

Modern guidance emphasizes that classical approximations, especially for nonlinear models, can be fragile. As a result, there is increasing emphasis on likelihood-based methods, which use the shape of the objective function more directly, and resampling-based approaches, which estimate uncertainty empirically from the data (O’Brien and Silcox, 2024). These alternatives aim to provide more reliable confidence limits in situations where traditional methods are inadequate.

## 15.7.2. Local Quadratic (Wald) Confidence Limits

Suppose $\hat{\theta}$ is the maximum likelihood estimate,

$$\hat{\theta} = \arg\max_\theta \ell(\theta) \tag{15.7.4}$$

The behavior of this estimator can be analyzed by examining the log-likelihood function $\ell(\theta)$ in a neighborhood of the true parameter value $\theta_0$. A common approach is to use a Taylor expansion of the score function, which is the gradient of the log-likelihood.

Expanding the score function around $\theta_0$ gives:

$$0 = \nabla \ell(\hat{\theta}) \approx \nabla \ell(\theta_0) + \nabla^2 \ell(\theta_0)(\hat{\theta} - \theta_0) \tag{15.7.5}$$

where the left-hand side vanishes because $\hat{\theta}$ is a maximizer of the log-likelihood. Rearranging this expression yields,

$$\hat{\theta} - \theta_0 \approx -\bigl(\nabla^2 \ell(\theta_0)\bigr)^{-1}\nabla \ell(\theta_0) \tag{15.7.6}$$

This relation expresses the estimation error in terms of the gradient and curvature of the log-likelihood at the true parameter. It shows that the deviation of the estimator from the true value is governed by both the variability of the score and the local curvature of the objective.

Under suitable regularity conditions and for sufficiently large sample sizes, this leads to the asymptotic normal approximation,

$$\hat{\theta} \sim \mathcal{N}\bigl(\theta_0, I(\theta_0)^{-1}\bigr)\quad \text{or} \quad\mathcal{N}\bigl(\theta_0, H(\hat{\theta})^{-1}\bigr) \tag{15.7.7}$$

where $I(\theta_0)$ is the Fisher information matrix and $H(\hat{\theta})$ is the observed Hessian of the negative log-likelihood. These matrices quantify the curvature of the likelihood and therefore determine the spread of the estimator.

Based on this approximation, a Wald confidence interval for a single parameter component $\theta_j$ is given by:

$$\hat{\theta}j \pm z_{1-\alpha/2}\sqrt{\mathrm{Var}(\hat{\theta}j)} \tag{15.7.8}$$

Here, $z_{1-\alpha/2}$ is the appropriate quantile of the standard normal distribution, and the variance is obtained from the corresponding diagonal entry of the covariance matrix. This construction provides a simple and computationally convenient way to obtain confidence intervals.

For nonlinear least squares with Gaussian noise, the log-likelihood is proportional to the residual sum of squares. In this case, the covariance matrix can be approximated using the Jacobian $J$:

$$\mathrm{Cov}(\hat{\theta}) \approx \sigma^2 (J^\top J)^{-1} \tag{15.7.9}$$

This expression mirrors the linear least-squares case, with the Jacobian playing the role of the design matrix. It shows that uncertainty in the parameter estimates is governed by the local sensitivity of the residuals to changes in the parameters.

From a numerical perspective, the computation of these quantities involves several steps. Building the Jacobian requires $O(nd)$ operations, where $n$ is the number of observations and $d$ is the number of parameters. Solving systems involving $J^\top J$ typically requires $O(d^3)$ operations for dense problems. As in earlier sections, stability considerations require avoiding explicit formation of $J^\top J$, since doing so can amplify conditioning issues. Instead, more stable factorization-based methods are preferred.

However, these intervals rely on a local quadratic approximation of the log-likelihood. This approximation assumes that the objective is well approximated by a quadratic form in a neighborhood of the optimum. For nonlinear models, this assumption can be inaccurate even with moderate sample sizes, particularly when the objective exhibits asymmetry or higher-order curvature. As a result, Wald intervals may be misleading in such cases, motivating the use of alternative methods that better capture the true shape of the likelihood (O’Brien and Silcox, 2024).

### Rust Implementation

Following the discussion in Section 15.7.2 on the local quadratic approximation of the log-likelihood and the resulting Wald confidence limits, Program 15.7.1 provides a practical implementation of nonlinear least-squares estimation together with covariance-based uncertainty quantification. In the Gaussian setting, the log-likelihood is directly related to the residual sum of squares, and the curvature of this objective near the optimum determines the variability of the estimator. This program performs a nonlinear least-squares fit using a damped Gauss–Newton method, constructs the Jacobian at the estimated parameters, and computes the covariance matrix using the approximation in Equation (15.7.9). From this covariance, standard errors and approximate confidence intervals are obtained using the Wald construction in Equation (15.7.8). The implementation demonstrates how local curvature information translates into quantitative uncertainty estimates and highlights the connection between optimization and statistical inference.

At the core of the implementation is the nonlinear model evaluation and residual construction, which define the objective function minimized during the fitting process. The function `model_value` evaluates the nonlinear model $f(x;\theta)$, while the function `residuals` constructs the residual vector $r = y - f(\theta)$, corresponding to the least-squares formulation discussed earlier in the section. The function `jacobian` computes the Jacobian matrix $J(\theta)$, which contains the partial derivatives of the residuals with respect to the parameters. This matrix plays a central role in both the Gauss–Newton iteration and the covariance approximation in Equation (15.7.9), where it replaces the design matrix of the linear case.

The nonlinear optimization is performed by the function `damped_gauss_newton`, which implements a stabilized Gauss–Newton method. At each iteration, the linearized least-squares problem is solved using a QR-based approach, producing an update direction $\delta$ that approximates the solution of the linearized system derived from Equation (15.7.6). A backtracking line search is then applied to ensure that each update reduces the residual sum of squares, thereby improving robustness compared to a pure Gauss–Newton method. The stopping criteria are based on both the norm of the update and the change in the objective, reflecting the local convergence behavior implied by the quadratic approximation of the log-likelihood.

The function `householder_qr_least_squares` computes the least-squares solution using Householder QR factorization, avoiding the explicit formation of $J^\top J$. This is consistent with the numerical considerations discussed in the section, where forming normal equations is discouraged due to potential conditioning issues. The resulting upper-triangular factor $R$ is reused to compute the covariance matrix efficiently. The function `covariance_from_r` implements the relation $(J^\top J)^{-1} = R^{-1}R^{-T}$, and scales it by the estimated noise variance $\sigma^2$ to obtain the covariance matrix as in Equation (15.7.9).

The statistical interpretation is completed by computing standard errors and confidence intervals. The function `standard_errors` extracts the square roots of the diagonal entries of the covariance matrix, corresponding to the variances of the parameter estimates. The main function then constructs Wald confidence intervals using Equation (15.7.8), combining the estimated parameters, their standard errors, and the appropriate normal quantile. This step illustrates how the asymptotic normal approximation in Equation (15.7.7) leads directly to practical uncertainty estimates.

The `main` function integrates all components into a complete workflow. It generates synthetic data from a known parameter vector, performs nonlinear least-squares fitting, evaluates residuals and convergence diagnostics, constructs the covariance matrix, and reports confidence intervals. This end-to-end process demonstrates how parameter estimation, local curvature analysis, and statistical interpretation are combined in practice, reflecting the theoretical development of the section.

```rust
// Program 15.7.1: Wald Confidence Limits for Nonlinear Least Squares
//
// Problem Statement:
// Estimate the parameter vector theta of a nonlinear least-squares model,
// then compute local quadratic (Wald) confidence limits based on the
// covariance approximation
//
//     Cov(theta_hat) ≈ sigma^2 (J^T J)^(-1),
//
// where J is the Jacobian of the residual vector evaluated at the fitted
// parameter vector. The program:
//
// 1. Fits a nonlinear model by damped Gauss-Newton iteration
// 2. Builds the Jacobian at the solution
// 3. Uses a QR factorization of J to compute the Gauss-Newton step
// 4. Uses the same triangular factor to construct the covariance matrix
// 5. Reports standard errors and approximate 95% Wald confidence intervals
//
// Model used in this demonstration:
//
//     y(x; theta) = a * (1 - exp(-b x)) + c
//
// with parameter vector theta = [a, b, c]^T.
//
// This example illustrates the local quadratic approximation discussed in
// Section 15.7.2 and shows how curvature information near the optimum leads
// to Wald confidence intervals.

type Vector = Vec<f64>;
type Matrix = Vec<Vec<f64>>;

// ----------------------------------------------------------
// Basic helpers
// ----------------------------------------------------------

fn zeros(rows: usize, cols: usize) -> Matrix {
    vec![vec![0.0; cols]; rows]
}

fn dot(x: &Vector, y: &Vector) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn norm2(x: &Vector) -> f64 {
    dot(x, x).sqrt()
}

fn norm_inf(x: &Vector) -> f64 {
    x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
}

fn vec_add(a: &Vector, b: &Vector) -> Vector {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn vec_sub(a: &Vector, b: &Vector) -> Vector {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn vec_scale(alpha: f64, x: &Vector) -> Vector {
    x.iter().map(|v| alpha * v).collect()
}

fn print_vector(name: &str, x: &Vector) {
    println!("{name}");
    for (i, value) in x.iter().enumerate() {
        println!("  [{:>2}] {:>.12e}", i, value);
    }
}

fn print_matrix(name: &str, a: &Matrix) {
    println!("{name}");
    for row in a {
        for value in row {
            print!("{:>18.10e} ", value);
        }
        println!();
    }
}

fn transpose(a: &Matrix) -> Matrix {
    let rows = a.len();
    let cols = a[0].len();
    let mut t = zeros(cols, rows);
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
    let rows = a.len();
    let inner = a[0].len();
    let cols = b[0].len();
    let mut c = zeros(rows, cols);

    for i in 0..rows {
        for k in 0..inner {
            let aik = a[i][k];
            for j in 0..cols {
                c[i][j] += aik * b[k][j];
            }
        }
    }

    c
}

// ----------------------------------------------------------
// Model and Jacobian
// ----------------------------------------------------------

fn model_value(x: f64, theta: &Vector) -> f64 {
    let a = theta[0];
    let b = theta[1];
    let c = theta[2];
    a * (1.0 - (-b * x).exp()) + c
}

fn residuals(xs: &Vector, ys: &Vector, theta: &Vector) -> Vector {
    xs.iter()
        .zip(ys.iter())
        .map(|(x, y)| y - model_value(*x, theta))
        .collect()
}

fn jacobian(xs: &Vector, theta: &Vector) -> Matrix {
    // J_{ij} = d r_i / d theta_j, where r_i = y_i - f(x_i; theta)
    // For f(x) = a (1 - exp(-b x)) + c:
    //
    // d r / d a = -(1 - exp(-b x))
    // d r / d b = -a x exp(-b x)
    // d r / d c = -1
    let n = xs.len();
    let mut j = zeros(n, 3);

    let a = theta[0];
    let b = theta[1];

    for i in 0..n {
        let x = xs[i];
        let e = (-b * x).exp();
        j[i][0] = -(1.0 - e);
        j[i][1] = -a * x * e;
        j[i][2] = -1.0;
    }

    j
}

fn rss(xs: &Vector, ys: &Vector, theta: &Vector) -> f64 {
    let r = residuals(xs, ys, theta);
    dot(&r, &r)
}

// ----------------------------------------------------------
// Upper-triangular solves and QR least squares
// ----------------------------------------------------------

fn back_substitution_upper(u: &Matrix, y: &Vector) -> Result<Vector, String> {
    let n = u.len();
    if n == 0 || u[0].len() != n || y.len() != n {
        return Err("Dimension mismatch in back_substitution_upper".to_string());
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= u[i][j] * x[j];
        }
        if u[i][i].abs() < 1.0e-15 {
            return Err("Upper-triangular matrix is singular".to_string());
        }
        x[i] = sum / u[i][i];
    }
    Ok(x)
}

struct QrLeastSquares {
    x: Vector,
    r_upper: Matrix,
}

fn householder_qr_least_squares(a: &Matrix, b: &Vector) -> Result<QrLeastSquares, String> {
    let n = a.len();
    let m = a[0].len();

    if b.len() != n {
        return Err("Dimension mismatch in householder_qr_least_squares".to_string());
    }
    if n < m {
        return Err("Need n >= m for least squares".to_string());
    }

    let mut r = a.clone();
    let mut qt_b = b.clone();

    for k in 0..m {
        let mut x = vec![0.0; n - k];
        for i in k..n {
            x[i - k] = r[i][k];
        }

        let norm_x = norm2(&x);
        if norm_x < 1.0e-15 {
            continue;
        }

        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * norm_x;

        let v_norm = norm2(&x);
        if v_norm < 1.0e-15 {
            continue;
        }

        let v: Vector = x.iter().map(|xi| xi / v_norm).collect();

        for j in k..m {
            let mut col_segment = vec![0.0; n - k];
            for i in k..n {
                col_segment[i - k] = r[i][j];
            }

            let proj = 2.0 * dot(&v, &col_segment);
            for i in k..n {
                r[i][j] -= proj * v[i - k];
            }
        }

        let mut b_segment = vec![0.0; n - k];
        for i in k..n {
            b_segment[i - k] = qt_b[i];
        }

        let proj_b = 2.0 * dot(&v, &b_segment);
        for i in k..n {
            qt_b[i] -= proj_b * v[i - k];
        }
    }

    let mut r_upper = zeros(m, m);
    for i in 0..m {
        for j in i..m {
            r_upper[i][j] = r[i][j];
        }
    }

    let mut rhs = vec![0.0; m];
    for i in 0..m {
        rhs[i] = qt_b[i];
    }

    let x = back_substitution_upper(&r_upper, &rhs)?;

    Ok(QrLeastSquares { x, r_upper })
}

// ----------------------------------------------------------
// Covariance from QR factorization
// ----------------------------------------------------------

fn inverse_upper_triangular(r: &Matrix) -> Result<Matrix, String> {
    let n = r.len();
    if n == 0 || r[0].len() != n {
        return Err("Matrix must be square in inverse_upper_triangular".to_string());
    }

    let mut rinv = zeros(n, n);

    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let col = back_substitution_upper(r, &e)?;
        for i in 0..n {
            rinv[i][j] = col[i];
        }
    }

    Ok(rinv)
}

fn covariance_from_r(r: &Matrix, sigma2_hat: f64) -> Result<Matrix, String> {
    // If J = Q R, then (J^T J)^(-1) = R^(-1) R^(-T)
    let rinv = inverse_upper_triangular(r)?;
    let rinv_t = transpose(&rinv);
    let mut cov = mat_mul(&rinv, &rinv_t);

    for i in 0..cov.len() {
        for j in 0..cov[0].len() {
            cov[i][j] *= sigma2_hat;
        }
    }

    Ok(cov)
}

fn standard_errors(cov: &Matrix) -> Vector {
    let n = cov.len();
    let mut se = vec![0.0; n];
    for i in 0..n {
        se[i] = cov[i][i].max(0.0).sqrt();
    }
    se
}

// ----------------------------------------------------------
// Damped Gauss-Newton fit
// ----------------------------------------------------------

struct FitResult {
    theta_hat: Vector,
    iterations: usize,
    converged: bool,
    rss: f64,
    residuals: Vector,
    jacobian: Matrix,
    r_upper: Matrix,
}

fn damped_gauss_newton(
    xs: &Vector,
    ys: &Vector,
    theta0: Vector,
    max_iters: usize,
    tol_step: f64,
    tol_rss: f64,
) -> Result<FitResult, String> {
    let mut theta = theta0;
    let mut current_rss = rss(xs, ys, &theta);

    for iter in 0..max_iters {
        let r = residuals(xs, ys, &theta);
        let j = jacobian(xs, &theta);

        // Solve J * delta ≈ r in the Gauss-Newton sense.
        let neg_r: Vector = r.iter().map(|v| -v).collect();
        let qr = householder_qr_least_squares(&j, &neg_r)?;
        let delta = qr.x;

        // If the computed step is already tiny, we are locally stationary.
        if norm2(&delta) < tol_step {
            let r_final = residuals(xs, ys, &theta);
            let j_final = jacobian(xs, &theta);
            let qr_final = householder_qr_least_squares(&j_final, &r_final)?;

            return Ok(FitResult {
                theta_hat: theta,
                iterations: iter + 1,
                converged: true,
                rss: current_rss,
                residuals: r_final,
                jacobian: j_final,
                r_upper: qr_final.r_upper,
            });
        }

        let mut accepted = false;
        let mut lambda = 1.0;
        let mut best_theta = theta.clone();
        let mut best_rss = current_rss;

        // Backtracking line search.
        for _ in 0..25 {
            let trial_theta = vec_add(&theta, &vec_scale(lambda, &delta));
            let trial_rss = rss(xs, ys, &trial_theta);

            if trial_rss < best_rss {
                best_theta = trial_theta;
                best_rss = trial_rss;
                accepted = true;
                break;
            }

            lambda *= 0.5;
            if lambda < 1.0e-6 {
                break;
            }
        }

        // If no acceptable step is found, terminate unsuccessfully.
        if !accepted {
            let r_final = residuals(xs, ys, &theta);
            let j_final = jacobian(xs, &theta);
            let qr_final = householder_qr_least_squares(&j_final, &r_final)?;

            return Ok(FitResult {
                theta_hat: theta,
                iterations: iter + 1,
                converged: false,
                rss: current_rss,
                residuals: r_final,
                jacobian: j_final,
                r_upper: qr_final.r_upper,
            });
        }

        let step_norm = norm2(&vec_sub(&best_theta, &theta));
        let rss_change = (current_rss - best_rss).abs();

        theta = best_theta;
        current_rss = best_rss;

        if step_norm < tol_step || rss_change < tol_rss {
            let r_final = residuals(xs, ys, &theta);
            let j_final = jacobian(xs, &theta);
            let qr_final = householder_qr_least_squares(&j_final, &r_final)?;

            return Ok(FitResult {
                theta_hat: theta,
                iterations: iter + 1,
                converged: true,
                rss: current_rss,
                residuals: r_final,
                jacobian: j_final,
                r_upper: qr_final.r_upper,
            });
        }
    }

    let r_final = residuals(xs, ys, &theta);
    let j_final = jacobian(xs, &theta);
    let qr_final = householder_qr_least_squares(&j_final, &r_final)?;

    Ok(FitResult {
        theta_hat: theta,
        iterations: max_iters,
        converged: false,
        rss: current_rss,
        residuals: r_final,
        jacobian: j_final,
        r_upper: qr_final.r_upper,
    })
}

// ----------------------------------------------------------
// Example data
// ----------------------------------------------------------

fn build_example() -> (Vector, Vector, Vector) {
    let n = 30;
    let theta_true = vec![2.5, 1.15, 0.35];

    let mut xs = vec![0.0; n];
    let mut ys = vec![0.0; n];

    for i in 0..n {
        let x = 3.0 * (i as f64) / ((n - 1) as f64);
        xs[i] = x;

        let y_true = model_value(x, &theta_true);

        // Deterministic synthetic perturbation for reproducibility.
        let noise =
            0.035 * (0.65 * (2.7 * x).sin() - 0.30 * (1.9 * x).cos() + 0.15 * (5.1 * x).sin());

        ys[i] = y_true + noise;
    }

    (xs, ys, theta_true)
}

// ----------------------------------------------------------
// Main
// ----------------------------------------------------------

fn main() -> Result<(), String> {
    let (xs, ys, theta_true) = build_example();

    // Improved initial guess.
    let theta0 = vec![2.0, 1.0, 0.2];
    let fit = damped_gauss_newton(&xs, &ys, theta0, 50, 1.0e-10, 1.0e-12)?;

    let n = xs.len();
    let d = fit.theta_hat.len();
    let dof = n - d;

    let sigma2_hat = fit.rss / (dof as f64);
    let sigma_hat = sigma2_hat.sqrt();

    let cov = covariance_from_r(&fit.r_upper, sigma2_hat)?;
    let se = standard_errors(&cov);

    let z95 = 1.96_f64;

    println!("Local Quadratic (Wald) Confidence Limits");
    println!("========================================");
    println!();

    print_vector("True parameter vector theta_true", &theta_true);
    println!();

    print_vector("Observed data y", &ys);
    println!();

    println!("Fit Summary");
    println!("-----------");
    println!("Converged                        = {}", fit.converged);
    println!("Iterations performed             = {}", fit.iterations);
    println!("Residual sum of squares          = {:>.12e}", fit.rss);
    println!("Estimated noise variance sigma^2 = {:>.12e}", sigma2_hat);
    println!("Estimated noise std. dev. sigma  = {:>.12e}", sigma_hat);
    println!();

    print_vector("Estimated parameter vector theta_hat", &fit.theta_hat);
    println!();

    print_vector("Residual vector r(theta_hat)", &fit.residuals);
    println!();

    print_matrix("Jacobian J(theta_hat)", &fit.jacobian);
    println!();

    print_matrix("Upper-triangular QR factor R", &fit.r_upper);
    println!();

    print_matrix("Approximate covariance matrix Cov(theta_hat)", &cov);
    println!();

    println!("Wald Confidence Intervals (Approx. 95%)");
    println!("---------------------------------------");
    println!(
        "{:>8} {:>18} {:>18} {:>18} {:>18}",
        "Index", "theta_hat", "Std. Error", "CI Lower", "CI Upper"
    );

    for j in 0..d {
        let lower = fit.theta_hat[j] - z95 * se[j];
        let upper = fit.theta_hat[j] + z95 * se[j];
        println!(
            "{:>8} {:>18.10e} {:>18.10e} {:>18.10e} {:>18.10e}",
            j, fit.theta_hat[j], se[j], lower, upper
        );
    }

    let estimation_error = vec_sub(&fit.theta_hat, &theta_true);
    println!();
    print_vector("Parameter estimation error theta_hat - theta_true", &estimation_error);
    println!(
        "Infinity-norm estimation error = {:>.12e}",
        norm_inf(&estimation_error)
    );
    println!();

    println!("Interpretation");
    println!("--------------");
    println!("This program fits a nonlinear least-squares model and then uses");
    println!("the local Jacobian at the optimum to build a Wald covariance");
    println!("approximation. The reported intervals are based on the local");
    println!("quadratic approximation discussed in Section 15.7.2, so they");
    println!("should be interpreted as asymptotic and local rather than exact.");
    println!();

    Ok(())
}
```

Program 15.7.1 demonstrates a practical implementation of Wald confidence limits based on the local quadratic approximation of the log-likelihood. By combining nonlinear least-squares fitting with Jacobian-based covariance estimation, the program illustrates how uncertainty in parameter estimates can be quantified using local curvature information, as described in Equations (15.7.7)–(15.7.9).

The results highlight the strengths of the Wald approach in well-behaved problems. When the model is sufficiently smooth and the optimum is accurately computed, the quadratic approximation provides reliable estimates of parameter variability, leading to confidence intervals that are both tight and centered around the true values. The use of QR factorization ensures numerical stability and avoids the pitfalls associated with forming normal equations explicitly.

At the same time, the implementation reflects the limitations discussed in the section. The intervals are symmetric and depend only on local curvature, which may not capture asymmetry or higher-order effects in more complex nonlinear models. This underscores the importance of alternative methods, such as profile likelihood or bootstrap approaches, when the quadratic approximation is insufficient.

The modular structure of the code allows it to be extended to more complex models, higher-dimensional parameter spaces, and alternative noise structures. It also provides a foundation for exploring more advanced uncertainty quantification techniques, reinforcing the connection between numerical optimization and statistical inference in modern computational practice.

## 15.7.3. Likelihood-Based and Profile Likelihood Confidence Limits

Likelihood-based methods avoid premature normal approximation by working directly with the shape of the log-likelihood function. Instead of assuming that the objective is locally quadratic, these methods use differences in log-likelihood values to define confidence regions. This approach retains more information about the underlying model and is therefore better suited to nonlinear problems.

Define the likelihood-ratio statistic as:

$$\Lambda(\theta) = 2\bigl[\ell(\hat{\theta}) - \ell(\theta)\bigr] \tag{15.7.10}$$

This quantity measures how much worse the fit becomes when the parameter vector is moved away from its optimal value $\hat{\theta}$. A small value of $\Lambda(\theta)$ indicates that the parameter $\theta$ provides a fit nearly as good as the optimum, while larger values indicate a significant degradation in fit quality.

An approximate confidence region can then be defined as:

$$C_{1-\alpha}= \{\theta : \Lambda(\theta) \le \chi^2_{d,1-\alpha}\} \tag{15.7.11}$$

Here, $\chi^2_{d,1-\alpha}$ is the appropriate quantile of the chi-square distribution with $d$ degrees of freedom. This construction is based on asymptotic theory, which shows that the likelihood-ratio statistic behaves approximately like a chi-square random variable under suitable conditions. The resulting confidence region reflects the global shape of the likelihood function rather than relying on a local approximation.

For a single parameter $\theta_j$, it is often convenient to reduce the problem to one dimension using the profile likelihood. Define:

$$\ell_p(\theta_j)= \max_{\theta_{-j}} \ell(\theta_j, \theta_{-j}) \tag{15.7.12}$$

where $\theta_{-j}$ denotes all parameters except $\theta_j$. This construction maximizes the likelihood with respect to the remaining parameters for each fixed value of $\theta_j$, effectively eliminating nuisance parameters from the problem. The resulting function $\ell_p(\theta_j)$ captures how the best possible fit varies as $\theta_j$ changes.

Confidence limits are then obtained by solving:

$$2\bigl[\ell_p(\hat{\theta}j) - \ell_p(\theta_j)\bigr]= \chi^2_{1,1-\alpha} \tag{15.7.13}$$

This equation defines the set of values of $\theta_j$ for which the profile likelihood remains sufficiently close to its maximum. Unlike Wald intervals, which are symmetric by construction, profile likelihood intervals can be asymmetric and reflect the true curvature of the objective function.

Profile likelihood intervals are widely recommended for nonlinear models because they preserve asymmetry and nonlinearity that Wald intervals flatten (O’Brien and Silcox, 2024). This makes them more reliable in situations where the objective function deviates significantly from a quadratic form.

From a numerical viewpoint, the computation of profile likelihood intervals involves a nested optimization structure. The outer loop scans or performs root-finding over the parameter (\\theta_j), seeking values that satisfy the likelihood-ratio condition. For each candidate value, an inner optimization problem must be solved to maximize the likelihood with respect to the remaining parameters. This leads to a total computational cost of:

$$O(m , C_{\text{opt}}) \tag{15.7.14}$$

where $m$ is the number of outer evaluations and $C_{\text{opt}}$ is the cost of the inner optimization. The overall cost can therefore be substantial, particularly for high-dimensional problems or expensive models.

Despite this computational expense, recent work shows that profile likelihood methods provide a unified framework for analyzing identifiability, parameter inference, and prediction intervals in mechanistic models (Simpson and Maclaren, 2023). By directly examining the structure of the likelihood function, these methods offer a comprehensive approach to uncertainty quantification that is well suited to modern nonlinear modeling problems.

## 15.7.4. Bootstrap Confidence Limits

Bootstrap methods provide a flexible approach to uncertainty quantification by approximating the sampling distribution of the estimator $\hat{\theta}$ through resampling. Rather than relying on analytic approximations or asymptotic theory, the bootstrap uses the observed data to generate multiple synthetic datasets, from which variability in the parameter estimates can be assessed directly.

The basic idea is to mimic the process of repeated sampling by constructing many alternative versions of the dataset and refitting the model to each one. This leads to the following algorithm:

1. Fit the model to the original data to obtain $\hat{\theta}$
2. For $b = 1,\dots,B$:
   - Resample the data or simulate new data from the model
   - Refit the model to obtain $\hat{\theta}^{*(b)}$
3. Construct confidence intervals from the empirical distribution of the bootstrap estimates $\{\hat{\theta}^{*(b)}\}$

Each resampled dataset produces a new estimate of the parameters, and the collection of these estimates approximates the sampling distribution of $\hat{\theta}$. Confidence limits are then obtained by examining appropriate quantiles or transformations of this empirical distribution.

Several types of bootstrap intervals are commonly used. The percentile method constructs intervals directly from quantiles of the bootstrap distribution. The bias-corrected and accelerated (BCa) method adjusts for both bias and skewness in the distribution, providing improved accuracy in many situations. The bootstrap-t method uses standardized statistics to account for variability in the estimator, often yielding more refined intervals when variance estimates are reliable.

Bootstrap methods are computationally intensive, since the model must be refitted many times:

$$O(B \, C_{\text{fit}}) \tag{15.7.15}$$

Here, $B$ is the number of bootstrap samples and $C_{\text{fit}}$ is the cost of a single model fit. For complex models, particularly nonlinear least-squares problems, this cost can be substantial, as each refit may itself require iterative optimization and repeated evaluation of the model and its derivatives.

Despite this computational expense, bootstrap methods are highly flexible and avoid reliance on asymptotic approximations. They can be applied to a wide range of models, including those with nonlinear structure, complex error distributions, or small sample sizes where classical approximations may be unreliable. By working directly with the empirical distribution of the estimator, the bootstrap can capture features such as skewness and nonlinearity that are not represented in quadratic approximations.

Modern studies emphasize the trade-offs between coverage accuracy and interval length (Mokhtar et al., 2023; Justus et al., 2024). More accurate intervals may be wider, reflecting increased uncertainty, while shorter intervals may sacrifice coverage. The choice of bootstrap method and the number of resamples therefore involve a balance between computational cost and statistical reliability.

Hybrid parametric bootstrap methods have also been developed for small datasets with measurement uncertainty (Golovko, 2025). These approaches combine model-based simulation with resampling ideas, allowing one to incorporate known noise structures while still benefiting from the flexibility of bootstrap techniques. Such methods are particularly useful when data are limited but a reasonable model for the noise is available.

### Rust Implementation

Following the discussion in Section 15.7.3 on likelihood-based inference and profile likelihood confidence limits, Program 15.7.2 provides a practical implementation of profile-likelihood-based uncertainty quantification for a nonlinear least-squares model. Unlike the Wald approach, which relies on a local quadratic approximation of the log-likelihood, this program evaluates the change in fit quality directly by computing the likelihood-ratio statistic defined in Equation (15.7.10). For each fixed value of a selected parameter, the remaining parameters are re-optimized, thereby constructing the profile likelihood as described in Equation (15.7.12). The resulting likelihood-ratio curve is then used to determine confidence limits by solving the threshold condition in Equation (15.7.13). This implementation demonstrates how profile likelihood methods capture the global structure of the objective and provide more reliable confidence intervals in nonlinear settings.

At the core of the implementation is the nonlinear least-squares model and the construction of the residual vector. The functions `model_value` and `residuals` define the mapping $f(x;\theta)$ and the residual vector $r = y - f(\theta)$, which determines the objective function minimized during parameter estimation. The function `jacobian` computes the Jacobian matrix of the residuals, which is used in the Gauss–Newton iteration to approximate the local linearization described in Equation (15.7.6). This provides the direction of descent for the nonlinear optimization.

The nonlinear fitting procedure is implemented in the function `damped_gauss_newton`, which computes the maximum-likelihood estimate $\hat{\theta}$ by minimizing the residual sum of squares. This corresponds to solving the optimization problem in Equation (15.7.4). At each iteration, a linearized least-squares problem is solved using QR factorization, and a backtracking strategy ensures that each update reduces the objective. This produces a stable and efficient method for obtaining the optimal parameter vector and the corresponding minimum residual sum of squares.

The key component of the program is the construction of the profile likelihood. The function `fit_a_c_for_fixed_b` performs the inner optimization required by Equation (15.7.12). For a fixed value of the parameter $b$, the model becomes linear in the remaining parameters, allowing the nuisance parameters to be estimated efficiently using linear least squares. This step implements the maximization over $\theta_{-j}$ for each candidate value of the profiled parameter.

The function `build_profile_likelihood` evaluates the profile likelihood over a grid of parameter values. For each value of (b), it computes the residual sum of squares and forms the likelihood-ratio statistic:

$$\Lambda(b) = \frac{\mathrm{RSS}_{\text{profile}}(b) - \mathrm{RSS}_{\min}}{\hat{\sigma}^2}$$

which is proportional to Equation (15.7.10) in the Gaussian case. This construction reflects how the fit quality deteriorates as the parameter moves away from its optimal value, capturing the global shape of the likelihood function.

The function `locate_profile_interval` identifies the confidence limits by detecting where the likelihood-ratio statistic crosses the chi-square threshold from Equation (15.7.13). This is achieved using linear interpolation between grid points, producing an approximation of the profile-likelihood interval. Unlike Wald intervals, which are symmetric by construction, the resulting interval can reflect asymmetry in the objective function.

The `main` function integrates all components into a complete workflow. It generates synthetic data, computes the maximum-likelihood estimate, constructs the profile likelihood, and reports the resulting confidence interval. The printed output includes representative profile points, illustrating how the likelihood-ratio statistic varies across the parameter range. This provides a direct visualization of the inference procedure and demonstrates the connection between optimization and statistical uncertainty quantification.

```rust
// Program 15.7.2: Profile Likelihood Confidence Limits for a Nonlinear Model
//
// Problem Statement:
// Fit a nonlinear least-squares model and compute likelihood-based confidence
// limits for one parameter using the profile likelihood. The program:
//
// 1. Fits the full nonlinear model by damped Gauss-Newton iteration
// 2. Selects one parameter to profile
// 3. For each fixed value of that parameter, re-optimizes the remaining
//    nuisance parameters
// 4. Forms the likelihood-ratio statistic
//        Lambda(theta_j) = [RSS_profile(theta_j) - RSS_min] / sigma^2_hat
//    which is proportional to Equation (15.7.10) in the Gaussian case
// 5. Locates the approximate profile-likelihood confidence limits from
//    Equation (15.7.13)
//
// Model used in this demonstration:
//
//     y(x; theta) = a * (1 - exp(-b x)) + c
//
// with parameter vector theta = [a, b, c]^T.
//
// The profiled parameter in this example is b. For each fixed b, the program
// re-fits a and c by least squares and computes the profile likelihood curve.

type Vector = Vec<f64>;
type Matrix = Vec<Vec<f64>>;

// ----------------------------------------------------------
// Basic helpers
// ----------------------------------------------------------

fn zeros(rows: usize, cols: usize) -> Matrix {
    vec![vec![0.0; cols]; rows]
}

fn dot(x: &Vector, y: &Vector) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn norm2(x: &Vector) -> f64 {
    dot(x, x).sqrt()
}

fn vec_add(a: &Vector, b: &Vector) -> Vector {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn vec_sub(a: &Vector, b: &Vector) -> Vector {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn vec_scale(alpha: f64, x: &Vector) -> Vector {
    x.iter().map(|v| alpha * v).collect()
}

fn print_vector(name: &str, x: &Vector) {
    println!("{name}");
    for (i, value) in x.iter().enumerate() {
        println!("  [{:>2}] {:>.12e}", i, value);
    }
}

#[allow(dead_code)]
fn print_matrix(name: &str, a: &Matrix) {
    println!("{name}");
    for row in a {
        for value in row {
            print!("{:>18.10e} ", value);
        }
        println!();
    }
}

// ----------------------------------------------------------
// Model and residuals
// ----------------------------------------------------------

fn model_value(x: f64, theta: &Vector) -> f64 {
    let a = theta[0];
    let b = theta[1];
    let c = theta[2];
    a * (1.0 - (-b * x).exp()) + c
}

fn residuals(xs: &Vector, ys: &Vector, theta: &Vector) -> Vector {
    xs.iter()
        .zip(ys.iter())
        .map(|(x, y)| y - model_value(*x, theta))
        .collect()
}

fn rss(xs: &Vector, ys: &Vector, theta: &Vector) -> f64 {
    let r = residuals(xs, ys, theta);
    dot(&r, &r)
}

fn jacobian(xs: &Vector, theta: &Vector) -> Matrix {
    let n = xs.len();
    let mut j = zeros(n, 3);

    let a = theta[0];
    let b = theta[1];

    for i in 0..n {
        let x = xs[i];
        let e = (-b * x).exp();
        j[i][0] = -(1.0 - e);
        j[i][1] = -a * x * e;
        j[i][2] = -1.0;
    }

    j
}

// ----------------------------------------------------------
// QR least squares for dense small systems
// ----------------------------------------------------------

fn back_substitution_upper(u: &Matrix, y: &Vector) -> Result<Vector, String> {
    let n = u.len();
    if n == 0 || u[0].len() != n || y.len() != n {
        return Err("Dimension mismatch in back_substitution_upper".to_string());
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= u[i][j] * x[j];
        }
        if u[i][i].abs() < 1.0e-15 {
            return Err("Upper-triangular matrix is singular".to_string());
        }
        x[i] = sum / u[i][i];
    }
    Ok(x)
}

struct QrLeastSquares {
    x: Vector,
}

fn householder_qr_least_squares(a: &Matrix, b: &Vector) -> Result<QrLeastSquares, String> {
    let n = a.len();
    let m = a[0].len();

    if b.len() != n {
        return Err("Dimension mismatch in householder_qr_least_squares".to_string());
    }
    if n < m {
        return Err("Need n >= m for least squares".to_string());
    }

    let mut r = a.clone();
    let mut qt_b = b.clone();

    for k in 0..m {
        let mut x = vec![0.0; n - k];
        for i in k..n {
            x[i - k] = r[i][k];
        }

        let norm_x = norm2(&x);
        if norm_x < 1.0e-15 {
            continue;
        }

        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * norm_x;

        let v_norm = norm2(&x);
        if v_norm < 1.0e-15 {
            continue;
        }

        let v: Vector = x.iter().map(|xi| xi / v_norm).collect();

        for j in k..m {
            let mut col_segment = vec![0.0; n - k];
            for i in k..n {
                col_segment[i - k] = r[i][j];
            }
            let proj = 2.0 * dot(&v, &col_segment);
            for i in k..n {
                r[i][j] -= proj * v[i - k];
            }
        }

        let mut b_segment = vec![0.0; n - k];
        for i in k..n {
            b_segment[i - k] = qt_b[i];
        }
        let proj_b = 2.0 * dot(&v, &b_segment);
        for i in k..n {
            qt_b[i] -= proj_b * v[i - k];
        }
    }

    let mut r_upper = zeros(m, m);
    for i in 0..m {
        for j in i..m {
            r_upper[i][j] = r[i][j];
        }
    }

    let mut rhs = vec![0.0; m];
    for i in 0..m {
        rhs[i] = qt_b[i];
    }

    let x = back_substitution_upper(&r_upper, &rhs)?;
    Ok(QrLeastSquares { x })
}

// ----------------------------------------------------------
// Full damped Gauss-Newton fit for theta = [a, b, c]
// ----------------------------------------------------------

struct FitResult {
    theta_hat: Vector,
    iterations: usize,
    converged: bool,
    rss: f64,
    residuals: Vector,
}

fn damped_gauss_newton(
    xs: &Vector,
    ys: &Vector,
    theta0: Vector,
    max_iters: usize,
    tol_step: f64,
    tol_rss: f64,
) -> Result<FitResult, String> {
    let mut theta = theta0;
    let mut current_rss = rss(xs, ys, &theta);

    for iter in 0..max_iters {
        let r = residuals(xs, ys, &theta);
        let j = jacobian(xs, &theta);

        let neg_r: Vector = r.iter().map(|v| -v).collect();
        let qr = householder_qr_least_squares(&j, &neg_r)?;
        let delta = qr.x;

        if norm2(&delta) < tol_step {
            let r_final = residuals(xs, ys, &theta);
            return Ok(FitResult {
                theta_hat: theta,
                iterations: iter + 1,
                converged: true,
                rss: current_rss,
                residuals: r_final,
            });
        }

        let mut accepted = false;
        let mut lambda = 1.0;
        let mut best_theta = theta.clone();
        let mut best_rss = current_rss;

        for _ in 0..25 {
            let trial_theta = vec_add(&theta, &vec_scale(lambda, &delta));

            if trial_theta[1] <= 1.0e-8 {
                lambda *= 0.5;
                continue;
            }

            let trial_rss = rss(xs, ys, &trial_theta);
            if trial_rss < best_rss {
                best_theta = trial_theta;
                best_rss = trial_rss;
                accepted = true;
                break;
            }

            lambda *= 0.5;
            if lambda < 1.0e-6 {
                break;
            }
        }

        if !accepted {
            let r_final = residuals(xs, ys, &theta);
            return Ok(FitResult {
                theta_hat: theta,
                iterations: iter + 1,
                converged: false,
                rss: current_rss,
                residuals: r_final,
            });
        }

        let step_norm = norm2(&vec_sub(&best_theta, &theta));
        let rss_change = (current_rss - best_rss).abs();

        theta = best_theta;
        current_rss = best_rss;

        if step_norm < tol_step || rss_change < tol_rss {
            let r_final = residuals(xs, ys, &theta);
            return Ok(FitResult {
                theta_hat: theta,
                iterations: iter + 1,
                converged: true,
                rss: current_rss,
                residuals: r_final,
            });
        }
    }

    let r_final = residuals(xs, ys, &theta);
    Ok(FitResult {
        theta_hat: theta,
        iterations: max_iters,
        converged: false,
        rss: current_rss,
        residuals: r_final,
    })
}

// ----------------------------------------------------------
// Profile likelihood for b by re-fitting a and c
// ----------------------------------------------------------

fn fit_a_c_for_fixed_b(xs: &Vector, ys: &Vector, b_fixed: f64) -> Result<(Vector, f64), String> {
    // For fixed b, the model becomes linear in a and c:
    // y_i ≈ a * phi_i + c,  where phi_i = 1 - exp(-b x_i)
    let n = xs.len();
    let mut a_mat = zeros(n, 2);
    let mut rhs = vec![0.0; n];

    for i in 0..n {
        let phi = 1.0 - (-b_fixed * xs[i]).exp();
        a_mat[i][0] = phi;
        a_mat[i][1] = 1.0;
        rhs[i] = ys[i];
    }

    let qr = householder_qr_least_squares(&a_mat, &rhs)?;
    let coeffs = qr.x; // [a, c]

    let theta = vec![coeffs[0], b_fixed, coeffs[1]];
    let rss_val = rss(xs, ys, &theta);

    Ok((theta, rss_val))
}

#[derive(Clone, Debug)]
struct ProfilePoint {
    b_value: f64,
    theta_profile: Vector,
    rss_profile: f64,
    lambda_value: f64,
}

fn build_profile_likelihood(
    xs: &Vector,
    ys: &Vector,
    theta_hat: &Vector,
    rss_min: f64,
    sigma2_hat: f64,
    b_min: f64,
    b_max: f64,
    num_points: usize,
) -> Result<Vec<ProfilePoint>, String> {
    if num_points < 2 {
        return Err("num_points must be at least 2".to_string());
    }

    let mut profile = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let t = i as f64 / ((num_points - 1) as f64);
        let b_value = b_min + (b_max - b_min) * t;

        let (theta_profile, rss_profile) = fit_a_c_for_fixed_b(xs, ys, b_value)?;
        let lambda_value = (rss_profile - rss_min) / sigma2_hat.max(1.0e-30);

        profile.push(ProfilePoint {
            b_value,
            theta_profile,
            rss_profile,
            lambda_value,
        });
    }

    // Ensure exact fitted point is visible in case grid misses it.
    let (theta_at_hat, rss_at_hat) = fit_a_c_for_fixed_b(xs, ys, theta_hat[1])?;
    let lambda_at_hat = (rss_at_hat - rss_min) / sigma2_hat.max(1.0e-30);
    profile.push(ProfilePoint {
        b_value: theta_hat[1],
        theta_profile: theta_at_hat,
        rss_profile: rss_at_hat,
        lambda_value: lambda_at_hat,
    });

    profile.sort_by(|p, q| p.b_value.partial_cmp(&q.b_value).unwrap());
    Ok(profile)
}

fn locate_profile_interval(profile: &[ProfilePoint], threshold: f64) -> (Option<f64>, Option<f64>) {
    if profile.len() < 2 {
        return (None, None);
    }

    let mut min_index = 0;
    let mut min_lambda = profile[0].lambda_value;
    for (i, p) in profile.iter().enumerate() {
        if p.lambda_value < min_lambda {
            min_lambda = p.lambda_value;
            min_index = i;
        }
    }

    let mut lower = None;
    for i in (1..=min_index).rev() {
        let p0 = &profile[i - 1];
        let p1 = &profile[i];
        let f0 = p0.lambda_value - threshold;
        let f1 = p1.lambda_value - threshold;

        if f0 == 0.0 {
            lower = Some(p0.b_value);
            break;
        }
        if f1 == 0.0 {
            lower = Some(p1.b_value);
            break;
        }
        if f0 * f1 < 0.0 {
            let t = f0 / (f0 - f1);
            lower = Some(p0.b_value + t * (p1.b_value - p0.b_value));
            break;
        }
    }

    let mut upper = None;
    for i in min_index..(profile.len() - 1) {
        let p0 = &profile[i];
        let p1 = &profile[i + 1];
        let f0 = p0.lambda_value - threshold;
        let f1 = p1.lambda_value - threshold;

        if f0 == 0.0 {
            upper = Some(p0.b_value);
            break;
        }
        if f1 == 0.0 {
            upper = Some(p1.b_value);
            break;
        }
        if f0 * f1 < 0.0 {
            let t = f0 / (f0 - f1);
            upper = Some(p0.b_value + t * (p1.b_value - p0.b_value));
            break;
        }
    }

    (lower, upper)
}

// ----------------------------------------------------------
// Example data
// ----------------------------------------------------------

fn build_example() -> (Vector, Vector, Vector) {
    let n = 30;
    let theta_true = vec![2.5, 1.15, 0.35];

    let mut xs = vec![0.0; n];
    let mut ys = vec![0.0; n];

    for i in 0..n {
        let x = 3.0 * (i as f64) / ((n - 1) as f64);
        xs[i] = x;

        let y_true = model_value(x, &theta_true);
        let noise =
            0.035 * (0.65 * (2.7 * x).sin() - 0.30 * (1.9 * x).cos() + 0.15 * (5.1 * x).sin());

        ys[i] = y_true + noise;
    }

    (xs, ys, theta_true)
}

// ----------------------------------------------------------
// Main
// ----------------------------------------------------------

fn main() -> Result<(), String> {
    let (xs, ys, theta_true) = build_example();

    let theta0 = vec![2.0, 1.0, 0.2];
    let fit = damped_gauss_newton(&xs, &ys, theta0, 50, 1.0e-10, 1.0e-12)?;

    let n = xs.len();
    let d = fit.theta_hat.len();
    let dof = n - d;
    let sigma2_hat = fit.rss / (dof as f64);

    // Profile the second parameter b = theta[1].
    let b_hat = fit.theta_hat[1];
    let b_min = 0.70 * b_hat;
    let b_max = 1.35 * b_hat;
    let num_profile_points = 81;

    let profile = build_profile_likelihood(
        &xs,
        &ys,
        &fit.theta_hat,
        fit.rss,
        sigma2_hat,
        b_min,
        b_max,
        num_profile_points,
    )?;

    let chi2_threshold = 3.841458820694124_f64; // chi-square 0.95 quantile, 1 dof
    let (lower, upper) = locate_profile_interval(&profile, chi2_threshold);

    println!("Likelihood-Based and Profile Likelihood Confidence Limits");
    println!("=========================================================");
    println!();

    print_vector("True parameter vector theta_true", &theta_true);
    println!();

    println!("Full Nonlinear Fit");
    println!("------------------");
    println!("Converged                          = {}", fit.converged);
    println!("Iterations performed               = {}", fit.iterations);
    println!("Residual sum of squares RSS_min    = {:>.12e}", fit.rss);
    println!("Estimated noise variance sigma^2   = {:>.12e}", sigma2_hat);
    println!();

    print_vector("Estimated parameter vector theta_hat", &fit.theta_hat);
    println!();

    print_vector("Residual vector at theta_hat", &fit.residuals);
    println!();

    println!("Profile Likelihood Setup");
    println!("------------------------");
    println!("Profiled parameter index           = 1");
    println!("Profiled parameter name            = b");
    println!("Estimated b_hat                    = {:>.12e}", b_hat);
    println!("Profile grid lower bound           = {:>.12e}", b_min);
    println!("Profile grid upper bound           = {:>.12e}", b_max);
    println!("Number of profile points           = {}", profile.len());
    println!("Likelihood-ratio threshold         = {:>.12e}", chi2_threshold);
    println!();

    println!("Selected Profile Points");
    println!("-----------------------");
    println!(
        "{:>8} {:>18} {:>18} {:>18} {:>18}",
        "Index", "b", "RSS_profile", "Lambda", "a_profile"
    );

    let step = (profile.len() / 8).max(1);
    for (i, p) in profile.iter().enumerate().step_by(step) {
        println!(
            "{:>8} {:>18.10e} {:>18.10e} {:>18.10e} {:>18.10e}",
            i, p.b_value, p.rss_profile, p.lambda_value, p.theta_profile[0]
        );
    }
    if let Some((i, p)) = profile.iter().enumerate().last() {
        if i % step != 0 {
            println!(
                "{:>8} {:>18.10e} {:>18.10e} {:>18.10e} {:>18.10e}",
                i, p.b_value, p.rss_profile, p.lambda_value, p.theta_profile[0]
            );
        }
    }
    println!();

    println!("Profile Likelihood Interval for b (Approx. 95%)");
    println!("-----------------------------------------------");
    match (lower, upper) {
        (Some(lo), Some(hi)) => {
            println!("Lower confidence limit            = {:>.12e}", lo);
            println!("Upper confidence limit            = {:>.12e}", hi);
            println!(
                "Profile-likelihood interval       = [{:>.12e}, {:>.12e}]",
                lo, hi
            );
        }
        _ => {
            println!("The confidence limits were not both bracketed on the chosen grid.");
            println!("Widen the profile range or refine the grid to capture the crossings.");
        }
    }
    println!();

    println!("Interpretation");
    println!("--------------");
    println!("The likelihood-ratio statistic is evaluated by re-optimizing the");
    println!("nuisance parameters a and c for each fixed value of b. The resulting");
    println!("profile likelihood preserves asymmetry in the objective and therefore");
    println!("can produce confidence limits that differ from symmetric Wald intervals.");
    println!("This reflects the likelihood-based construction described in");
    println!("Section 15.7.3 and avoids relying solely on local quadratic curvature.");
    println!();

    Ok(())
}
```

Program 15.7.2 demonstrates a practical implementation of profile likelihood confidence limits for nonlinear least-squares problems. By evaluating the likelihood-ratio statistic across a range of parameter values and re-optimizing nuisance parameters at each step, the program constructs confidence intervals that reflect the true shape of the objective function, as described in Section 15.7.3.

The results highlight the advantages of likelihood-based methods over local quadratic approximations. While Wald intervals rely on curvature information near the optimum, profile likelihood intervals incorporate global information about the objective, allowing them to capture asymmetry and nonlinearity in the model. This leads to more reliable uncertainty estimates, particularly in nonlinear or poorly conditioned problems.

The implementation also illustrates the computational trade-off associated with profile likelihood methods. Each evaluation of the likelihood-ratio statistic requires solving a reduced optimization problem, leading to increased computational cost compared to covariance-based approaches. Nevertheless, this cost is justified in situations where the quadratic approximation is inadequate and more accurate inference is required.

The modular structure of the code allows for straightforward extensions, including profiling of multiple parameters, adaptive grid refinement, and integration with more advanced optimization methods. This provides a foundation for further exploration of likelihood-based inference techniques and their role in modern statistical computation.

## 15.7.5. Robust Covariance Under Misspecification and Numerical Considerations

The standard covariance formulas used in parameter estimation are typically derived under the assumption that the likelihood model is correctly specified. In particular, they rely on the assumption that the noise distribution and independence structure accurately reflect the true data-generating process. When this assumption is violated, these covariance estimates can be overly optimistic, underestimating the true variability of the parameter estimates.

In such cases, a more general and robust expression for the asymptotic covariance is given by the sandwich estimator:

$$\mathrm{Cov}(\hat{\theta})\approx A^{-1} B A^{-1} \tag{15.7.16}$$

where the matrices $A$ and $B$ capture different aspects of the estimation problem.

The matrix $A$, often referred to as the sensitivity or “bread” matrix, reflects how the estimating equations respond to changes in the parameters. It is closely related to the curvature of the objective function and plays a role analogous to the Hessian or Fisher information in the correctly specified case.

The matrix $B$, known as the variability or “meat” matrix, captures the variability of the estimating equations themselves. It reflects the actual dispersion of the data and incorporates the effects of model misspecification, including deviations from assumed noise distributions or dependence structures.

This formulation is connected to the concept of Godambe information, which generalizes the Fisher information to settings where the model may be misspecified. It provides a principled way to account for discrepancies between the assumed and actual data-generating processes, yielding covariance estimates that remain valid under broader conditions (Vrugt et al., 2025; Vrugt and Diks, 2025).

From a numerical perspective, this change in covariance structure has important implications. Confidence regions remain locally ellipsoidal, since they are still derived from a quadratic approximation in parameter space. However, the geometry of these ellipsoids is no longer determined solely by the inverse Hessian $H^{-1}$. Instead, it is governed by the matrix $A^{-1} B A^{-1}$, which incorporates both sensitivity and variability.

This distinction highlights that uncertainty quantification depends not only on the local curvature of the objective function but also on how well the model represents the data. When the model is misspecified, the curvature alone is insufficient to describe uncertainty accurately, and the additional variability captured by $B$ must be taken into account.

In practical computation, this means that both matrices $A$ and $B$ must be estimated reliably, often using sample-based approximations. As in earlier sections, numerical conditioning and stability play an important role, since the inversion of $A$ and the multiplication of matrices can amplify errors if the problem is ill-conditioned. Careful numerical implementation is therefore essential to ensure that the resulting covariance estimates are both stable and meaningful.

### Computational and Implementation Considerations

Confidence-limit computation is fundamentally a numerical task that builds directly on the methods developed for optimization and linear algebra. Although the underlying concepts are statistical, their practical realization depends critically on how derivatives are computed, how linear systems are solved, and how optimization procedures are implemented. Inaccuracies or instabilities in these components can propagate into the final uncertainty estimates, leading to misleading conclusions.

At the core of these computations are three key requirements. First, stable derivatives are essential, whether obtained analytically or through automatic differentiation. Since gradients and Jacobians determine both parameter estimates and their covariance, errors in derivative computation can degrade both convergence and uncertainty quantification. Second, stable linear algebra is required for forming and solving systems involving covariance matrices or approximations thereof. Factorization-based methods such as QR or Cholesky decompositions are preferred, as they avoid the numerical issues associated with explicit matrix inversion. Third, stable optimization procedures are necessary, particularly for likelihood-based methods such as profile likelihood, where repeated constrained optimizations must be performed reliably.

These requirements lead to several practical guidelines. Analytic derivatives or automatic differentiation should be preferred over finite-difference approximations, since finite differences can introduce truncation and round-off errors, especially in high-dimensional or poorly scaled problems. For covariance computation, factorization-based solves should be used rather than forming explicit inverses, ensuring both numerical stability and computational efficiency. In profile likelihood methods, the optimization should be organized as an outer bracketing or root-finding loop combined with inner optimizations that are warm-started from previous solutions. This reuse of information reduces computational cost and improves robustness by maintaining continuity between successive solves.

Bootstrap methods introduce additional considerations. Since they require repeated model fitting, they are naturally parallelizable. However, parallel implementation must be handled carefully to ensure reproducibility, particularly in the generation of random samples. Proper management of random number generators is therefore essential to obtain consistent and verifiable results across runs.

These considerations highlight a central theme: uncertainty quantification is inseparable from numerical stability and algorithm design. Reliable confidence limits are not obtained solely from theoretical formulas, but from careful integration of statistical modeling with robust computational methods.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/msb6chVRQD7VSxDM9H6A.6","tags":[]}

# 15.8. Robust Estimation

In practical data analysis, the assumptions underlying classical least squares are frequently violated. Measurement errors may be heavy-tailed, contaminated by outliers, or exhibit heteroscedasticity and dependence. These effects arise naturally in real-world datasets, where observations are influenced by complex and sometimes poorly understood noise mechanisms. As a result, the idealized conditions under which least squares is optimal are often only approximately satisfied, if at all.

In such situations, standard least-squares estimators can be highly sensitive. Because the squared loss function assigns increasing weight to larger residuals, a small number of extreme observations can exert a disproportionate influence on the fitted model. This sensitivity can distort parameter estimates, obscure underlying trends, and reduce the reliability of predictions. The issue becomes more pronounced as the dimensionality of the problem increases or when the data contain structured forms of contamination.

Robust estimation seeks to mitigate this sensitivity by modifying the fitting criterion so that extreme residuals have limited influence. Rather than treating all deviations equally in a quadratic sense, robust methods adjust the contribution of each observation based on its consistency with the model. This leads to estimators that are less affected by outliers while still capturing the dominant structure present in the data.

The goal is not to discard data arbitrarily, but to ensure that inference remains stable under realistic deviations from ideal assumptions. Robust methods aim to provide parameter estimates that reflect the bulk of the data, even in the presence of contamination or model misspecification. In this sense, robustness is closely tied to the reliability and interpretability of statistical conclusions.

This is particularly important in modern high-dimensional settings, where robustness and computation must be considered together (Loh, 2025). Large datasets, complex models, and limited computational resources require methods that are not only statistically sound but also efficient and scalable. Robust estimation therefore represents an essential extension of classical least-squares theory, addressing both practical data challenges and modern computational constraints.

## 15.8.1. Sources of Non-Ideal Behavior in Data

Real datasets often violate the assumptions of Gaussian, independent, homoscedastic noise. These assumptions are convenient for analysis and computation, but they rarely hold exactly in practice. Deviations from these ideal conditions can arise from measurement processes, data collection pipelines, or inherent variability in the system being modeled. Understanding these sources of non-ideal behavior is essential for assessing when classical least-squares methods may fail and when robust alternatives are required.

Several common issues arise in practice:

#### Outliers (due to sensor faults, data corruption, or rare events)

Outliers are observations that deviate significantly from the majority of the data. They may result from transient failures in measurement systems, transmission errors, or genuinely rare phenomena. Even a small number of such points can have a large impact on least-squares estimates, since the squared loss amplifies their influence.

#### Heavy-tailed Noise (where extreme deviations occur more frequently than Gaussian models predict)

In many applications, the distribution of measurement errors exhibits heavier tails than the normal distribution. This means that large deviations are more common than expected under Gaussian assumptions. Least-squares methods, which are optimized for Gaussian noise, may perform poorly in such settings because they are not designed to accommodate frequent large residuals.

#### Heteroscedasticity (variance depends on the input $x$)

When the variability of the observations changes across the domain, the assumption of constant variance is violated. In such cases, some regions of the data may be more reliable than others, and treating all observations equally can lead to inefficient or biased estimates. Proper modeling of heteroscedasticity is therefore important for accurate inference.

#### Leverage Points (extreme values of $x$ that disproportionately influence the fit)

Observations with unusual or extreme values of the independent variable can exert strong influence on the fitted model, even if their residuals are not large. These points affect the geometry of the design matrix and can significantly alter parameter estimates, especially in regression problems.

#### Model Misspecification (incorrect functional form or noise model)

If the chosen model does not accurately represent the underlying relationship between variables, or if the assumed noise model is incorrect, the resulting estimates may be biased or inconsistent. Misspecification can arise from oversimplified modeling assumptions or incomplete knowledge of the system being studied.

These phenomena can invalidate classical inference procedures, since many standard results rely on assumptions that are no longer satisfied. Confidence intervals, hypothesis tests, and covariance estimates may become unreliable or misleading under such conditions.

Robust estimation provides a framework for addressing these issues while retaining computational tractability. By modifying the fitting criterion and reducing sensitivity to extreme or atypical observations, robust methods aim to produce stable and reliable estimates even when the data deviate from ideal assumptions.

## 15.8.2. M-Estimation: Replacing the Squared Loss

The central framework for robust estimation is M-estimation, which replaces the quadratic loss with a more general function:

$$\hat{\theta}= \arg\min_{\theta}\sum_{i=1}^{N}\rho\left(\frac{r_i(\theta)}{s}\right) \tag{15.8.1}$$

where,

$$r_i(\theta) = y_i - f(x_i;\theta) \tag{15.8.2}$$

$s$ is a scale estimate, and $\rho(\cdot)$ grows more slowly than the quadratic loss for large residuals. The introduction of the function $\rho$ is the key modification that provides robustness. By controlling how rapidly the loss increases with the residual, one can reduce the influence of extreme observations while still fitting the majority of the data.

The normalization by the scale parameter $s$ ensures that residuals are measured relative to a characteristic level of variability in the data. This makes the procedure invariant to changes in the overall scale of the problem and allows the loss function to be applied consistently across different datasets.

To analyze the resulting optimization problem, define:

$$\psi(u) = \rho'(u) \tag{15.8.3}$$

The function $\psi$ is often referred to as the influence function, since it determines how each residual contributes to the estimating equations. In contrast to the linear growth associated with the derivative of the quadratic loss, robust choices of $\psi$ typically grow more slowly or even saturate for large values of $u$, thereby limiting the influence of outliers.

The first-order optimality condition becomes:

$$\sum_{i=1}^{N}\psi\left(\frac{r_i(\theta)}{s}\right)\nabla r_i(\theta) = 0 \tag{15.8.4}$$

This equation replaces the normal equations of least squares and expresses a balance of contributions from all observations, weighted according to the function $\psi$. Observations with small residuals contribute approximately linearly, while those with large residuals are downweighted through the nonlinear behavior of $\psi$.

For a linear model $r(\beta)=y - X\beta$, this reduces to:

$$\sum_{i=1}^{N}\psi\left(\frac{y_i - x_i^\top \beta}{s}\right)x_i = 0 \tag{15.8.5}$$

This expression resembles the normal equations but with each term modified by a nonlinear weighting factor. The contribution of each data point depends not only on its predictor values $x_i$, but also on how well it agrees with the current model through its residual.

Unlike least squares, this system is generally nonlinear, even when the model itself is linear, because $\psi$ is nonlinear. As a result, closed-form solutions are typically not available, and iterative methods must be used to compute the estimator. This additional complexity reflects the trade-off inherent in robust estimation: improved resistance to outliers and model deviations at the cost of increased computational effort.

At the same time, the structure of the problem retains a close connection to least squares, since the estimating equations can often be interpreted as weighted versions of the normal equations. This connection provides a pathway to efficient algorithms, as will be developed in subsequent subsections.

## 15.8.3. Iteratively Reweighted Least Squares (IRLS)

A practical method for solving M-estimation problems is iteratively reweighted least squares (IRLS). The key idea is to convert the nonlinear estimating equations into a sequence of weighted least-squares problems, each of which can be solved using the linear algebra techniques developed earlier. This provides an efficient and conceptually clear approach to robust estimation.

To construct the method, define the normalized residuals:

$$u_i = \frac{r_i}{s} \tag{15.8.6}$$

which scale each residual by the characteristic variability of the data. These normalized quantities determine how strongly each observation should influence the fit.

Next, define the weights:

$$w_i(u_i) =\begin{cases}\psi(u_i)/u_i, & u_i \neq 0 \\ \psi'(0), & u_i = 0 \end{cases} \tag{15.8.7}$$

which are derived from the influence function $\psi$. These weights translate the nonlinear dependence of the M-estimation problem into a set of observation-specific scaling factors. For small residuals, the weights are typically close to one, so the corresponding observations are treated similarly to least squares. For large residuals, the weights decrease, reducing the influence of those observations on the solution.

At iteration $t$, the method proceeds by solving the weighted least-squares problem:

$$\beta^{(t+1)}= \arg\min_\beta\sum_{i=1}^{N}w_i^{(t)} (y_i - x_i^\top \beta)^2 \tag{15.8.8}$$

This step updates the parameter estimate using the current weights, which reflect how well each observation agrees with the model at the previous iteration.

In matrix form, the update can be written as:

$$\beta^{(t+1)}= (X^\top W^{(t)} X)^{-1} X^\top W^{(t)} y \tag{15.8.9}$$

where $W^{(t)} = \mathrm{diag}(w_1^{(t)}, \dots, w_N^{(t)})$ is a diagonal matrix of weights. This expression has the same structure as the weighted least-squares solution discussed earlier, with the weights now updated iteratively based on the residuals.

Thus, robust estimation reduces to a sequence of weighted least-squares problems. Each iteration consists of computing residuals, updating weights, and solving a linear system. This structure makes IRLS attractive in practice, since it leverages well-understood and efficient linear algebra methods while incorporating robustness through the weighting mechanism.

The computational cost per iteration can be characterized as follows. Forming the matrix $X^\top W X$ requires $O(np^2)$ operations, where $n$ is the number of observations and $p$ is the number of parameters. The factorization and solution of the resulting system require $O(p^3)$ operations for dense problems. The memory requirement is $O(p^2)$, reflecting the storage needed for the system matrix and its factorization.

For large-scale problems, these costs can become significant, particularly when the number of parameters is large or when the data are high-dimensional. In such cases, iterative solvers and preconditioning techniques become important. By exploiting sparsity or structure in the design matrix, one can reduce both computational cost and memory usage, making IRLS applicable to modern large-scale settings.

Overall, IRLS provides a practical bridge between robust statistical modeling and efficient numerical computation, transforming a nonlinear estimation problem into a sequence of tractable linear subproblems.

### Rust Implementation

Following the formulation of the IRLS algorithm in Section 15.8.3, Program 15.8.1 provides a complete implementation of robust linear regression based on iteratively reweighted least squares. The method translates the nonlinear M-estimation problem into a sequence of weighted least-squares solves, where the weights are updated at each iteration according to the normalized residuals defined in Equation (15.8.6) and the weighting rule in Equation (15.8.7). This program demonstrates how robust estimation can be implemented efficiently using standard linear algebra techniques, while reducing the influence of outliers through adaptive weighting. By combining residual evaluation, scale estimation, and iterative refinement, the implementation reflects the computational structure described in Equations (15.8.8) and (15.8.9), and provides a practical framework for robust regression in finite-precision environments.

At the core of the implementation is the iterative structure of the IRLS algorithm, which mirrors the sequence of weighted least-squares problems described in Equation (15.8.8). The program begins by constructing the design matrix and computing an initial parameter estimate using ordinary least squares. This serves as the starting point for the iterative refinement process. The function `residuals` evaluates the residual vector $r_i = y_i - x_i^\top \beta$, which is then used to compute normalized residuals $u_i = r_i / s$ as defined in Equation (15.8.6). The scale parameter $s$ is estimated using a median absolute deviation (MAD) approach, providing robustness against extreme values.

The weight update is implemented through the functions `huber_weight` and `huber_weights`, which encode the relationship given in Equation (15.8.7). These functions translate the influence function $\psi$ into weights $w_i$, ensuring that observations with large normalized residuals are downweighted. For small residuals, the weights remain close to one, preserving the behavior of least squares, while for large residuals, the weights decrease, limiting their influence on the solution.

The function `weighted_normal_equations` constructs the matrix $X^\top W X$ and the vector $X^\top W y$, corresponding directly to the matrix formulation in Equation (15.8.9). These are then solved using a Cholesky factorization implemented in `solve_spd_cholesky`, which provides a numerically stable method for solving symmetric positive definite systems. This avoids explicit matrix inversion and aligns with best practices in numerical linear algebra.

The main IRLS loop is implemented in the function `irls_huber`, which repeatedly updates residuals, recomputes weights, and solves the weighted system until convergence. The stopping criterion is based on the maximum change in the parameter vector between iterations, ensuring that the algorithm terminates once successive estimates differ by less than a prescribed tolerance. This reflects the iterative refinement process described in the section and provides a practical mechanism for convergence control.

The `main` function demonstrates the behavior of the algorithm on a dataset containing injected outliers. It compares the ordinary least-squares solution with the robust IRLS solution, highlighting the effect of downweighting extreme observations. Diagnostic outputs, including residuals, weights, and scale estimates, provide insight into how the algorithm adapts to the data and which points are identified as influential.

```rust
// Program 15.8.1: Iteratively Reweighted Least Squares (IRLS) for Robust Linear Regression
//
// Problem Statement:
// Implement robust linear regression using iteratively reweighted least squares (IRLS).
// The program fits a straight-line model
//
//     y_i ≈ β_0 + β_1 x_i
//
// in the presence of outliers. At each IRLS iteration, it:
//
// 1. Computes residuals r_i = y_i - x_i^T β
// 2. Estimates a robust scale s from the residuals
// 3. Forms normalized residuals u_i = r_i / s
// 4. Updates weights w_i from a Huber influence rule
// 5. Solves the weighted normal equations
//
//     (X^T W X) β = X^T W y
//
// to obtain the next parameter vector.
//
// This program is fully self-contained and uses only the Rust standard library.

use std::fmt;

// ----------------------------
// Basic Linear Algebra Helpers
// ----------------------------

type Vector = Vec<f64>;
type Matrix = Vec<Vec<f64>>;

fn zeros_matrix(rows: usize, cols: usize) -> Matrix {
    vec![vec![0.0; cols]; rows]
}

fn mat_vec_mul(a: &Matrix, x: &[f64]) -> Vector {
    let rows = a.len();
    let cols = a[0].len();
    assert_eq!(cols, x.len(), "Matrix-vector dimension mismatch.");

    let mut y = vec![0.0; rows];
    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += a[i][j] * x[j];
        }
        y[i] = sum;
    }
    y
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Dot-product dimension mismatch.");
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

// ----------------------------
// Cholesky Solver for SPD Systems
// ----------------------------

fn cholesky_decompose(a: &Matrix) -> Result<Matrix, String> {
    let n = a.len();
    if n == 0 || a[0].len() != n {
        return Err("Matrix must be nonempty and square for Cholesky decomposition.".to_string());
    }

    let mut l = zeros_matrix(n, n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }

            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "Matrix is not positive definite at diagonal entry {}.",
                        i
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    Ok(l)
}

fn forward_substitution(l: &Matrix, b: &[f64]) -> Result<Vector, String> {
    let n = l.len();
    if b.len() != n {
        return Err("Dimension mismatch in forward substitution.".to_string());
    }

    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        if l[i][i].abs() < 1.0e-15 {
            return Err(format!("Zero diagonal encountered in forward substitution at {}.", i));
        }
        y[i] = sum / l[i][i];
    }
    Ok(y)
}

fn backward_substitution_from_cholesky(l: &Matrix, y: &[f64]) -> Result<Vector, String> {
    let n = l.len();
    if y.len() != n {
        return Err("Dimension mismatch in backward substitution.".to_string());
    }

    let mut x = vec![0.0; n];
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j];
        }
        if l[i][i].abs() < 1.0e-15 {
            return Err(format!(
                "Zero diagonal encountered in backward substitution at {}.",
                i
            ));
        }
        x[i] = sum / l[i][i];
    }
    Ok(x)
}

fn solve_spd_cholesky(a: &Matrix, b: &[f64]) -> Result<Vector, String> {
    let l = cholesky_decompose(a)?;
    let y = forward_substitution(&l, b)?;
    let x = backward_substitution_from_cholesky(&l, &y)?;
    Ok(x)
}

// ----------------------------
// Robust Statistics Helpers
// ----------------------------

fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "Median of empty slice is undefined.");

    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn mad_scale(residuals: &[f64]) -> f64 {
    let med = median(residuals);
    let abs_dev: Vec<f64> = residuals.iter().map(|r| (r - med).abs()).collect();
    let mad = median(&abs_dev);

    // Consistency factor for Gaussian noise.
    let s = 1.4826 * mad;

    // Prevent division by zero in nearly exact fits.
    s.max(1.0e-12)
}

// ----------------------------
// Model Construction
// ----------------------------

fn design_matrix_for_line(x: &[f64]) -> Matrix {
    let mut xmat = Vec::with_capacity(x.len());
    for &xi in x {
        xmat.push(vec![1.0, xi]);
    }
    xmat
}

fn residuals(x: &Matrix, y: &[f64], beta: &[f64]) -> Vector {
    let yhat = mat_vec_mul(x, beta);
    y.iter().zip(yhat.iter()).map(|(yi, yhi)| yi - yhi).collect()
}

fn weighted_normal_equations(x: &Matrix, y: &[f64], w: &[f64]) -> (Matrix, Vector) {
    let n = x.len();
    let p = x[0].len();

    assert_eq!(y.len(), n, "Response length mismatch.");
    assert_eq!(w.len(), n, "Weight length mismatch.");

    let mut xtwx = zeros_matrix(p, p);
    let mut xtwy = vec![0.0; p];

    for i in 0..n {
        for j in 0..p {
            xtwy[j] += w[i] * x[i][j] * y[i];
            for k in 0..p {
                xtwx[j][k] += w[i] * x[i][j] * x[i][k];
            }
        }
    }

    (xtwx, xtwy)
}

fn ordinary_least_squares(x: &Matrix, y: &[f64]) -> Result<Vector, String> {
    let n = x.len();
    let p = x[0].len();

    let one_weights = vec![1.0; n];
    let (xtx, xty) = weighted_normal_equations(x, y, &one_weights);

    if p == 0 {
        return Err("Design matrix has zero columns.".to_string());
    }

    solve_spd_cholesky(&xtx, &xty)
}

// ----------------------------
// Huber IRLS Weights
// ----------------------------
//
// For the Huber loss, the influence function is
//
//   ψ(u) = u                  if |u| <= c
//   ψ(u) = c * sign(u)        if |u| >  c
//
// Hence the IRLS weight
//
//   w(u) = ψ(u) / u
//
// becomes
//
//   w(u) = 1                  if |u| <= c
//   w(u) = c / |u|            if |u| >  c
//
// with w(0) = 1.

fn huber_weight(u: f64, c: f64) -> f64 {
    let au = u.abs();
    if au <= c || au < 1.0e-15 {
        1.0
    } else {
        c / au
    }
}

fn huber_weights(normalized_residuals: &[f64], c: f64) -> Vector {
    normalized_residuals
        .iter()
        .map(|&u| huber_weight(u, c))
        .collect()
}

// ----------------------------
// IRLS Solver
// ----------------------------

#[derive(Clone, Debug)]
struct IrlsResult {
    beta: Vector,
    converged: bool,
    iterations_used: usize,
    final_scale: f64,
    final_weights: Vector,
    residuals: Vector,
}

fn max_abs_difference(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vector length mismatch.");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn irls_huber(
    x: &Matrix,
    y: &[f64],
    c: f64,
    max_iters: usize,
    tol: f64,
) -> Result<IrlsResult, String> {
    let n = x.len();
    if n == 0 {
        return Err("Empty dataset.".to_string());
    }
    if y.len() != n {
        return Err("Response length does not match design matrix.".to_string());
    }

    let mut beta = ordinary_least_squares(x, y)?;
    let mut weights = vec![1.0; n];
    let mut scale = 1.0;
    let mut converged = false;
    let mut residual_vec: Vector;

    for iter in 0..max_iters {
        residual_vec = residuals(x, y, &beta);
        scale = mad_scale(&residual_vec);

        let normalized: Vector = residual_vec.iter().map(|r| r / scale).collect();
        weights = huber_weights(&normalized, c);

        let (xtwx, xtwy) = weighted_normal_equations(x, y, &weights);
        let beta_new = solve_spd_cholesky(&xtwx, &xtwy)?;

        let change = max_abs_difference(&beta_new, &beta);
        beta = beta_new;

        if change < tol {
            converged = true;
            return Ok(IrlsResult {
                beta: beta.clone(),
                converged,
                iterations_used: iter + 1,
                final_scale: scale,
                final_weights: weights.clone(),
                residuals: residuals(x, y, &beta),
            });
        }
    }

    Ok(IrlsResult {
        beta: beta.clone(),
        converged,
        iterations_used: max_iters,
        final_scale: scale,
        final_weights: weights.clone(),
        residuals: residuals(x, y, &beta),
    })
}

// ----------------------------
// Data Generation
// ----------------------------

fn generate_contaminated_line_data() -> (Vector, Vector) {
    let x = vec![
        -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0,
        2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    ];

    // True model: y = 2 + 1.5 x, with mild deterministic noise.
    let mut y = vec![
        -5.48, -4.79, -4.02, -3.20, -2.47, -1.77, -0.92, -0.28, 0.55, 1.28, 2.06, 2.83, 3.53,
        4.28, 5.05, 5.78, 6.46, 7.31, 8.02, 8.79, 9.47,
    ];

    // Inject a few outliers.
    y[3] += 5.5;
    y[11] -= 6.0;
    y[17] += 4.5;

    (x, y)
}

// ----------------------------
// Reporting Helpers
// ----------------------------

struct BetaDisplay<'a>(&'a [f64]);

impl<'a> fmt::Display for BetaDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, value) in self.0.iter().enumerate() {
            writeln!(f, "  beta[{}] = {:>.10}", i, value)?;
        }
        Ok(())
    }
}

fn print_dataset(x: &[f64], y: &[f64]) {
    println!("Input Data");
    println!("==========");
    for i in 0..x.len() {
        println!("i = {:>2}, x = {:>.6}, y = {:>.6}", i, x[i], y[i]);
    }
    println!();
}

fn print_fit_summary(name: &str, beta: &[f64], residuals: &[f64]) {
    let rss = dot(residuals, residuals);
    let max_abs_residual = residuals
        .iter()
        .map(|r| r.abs())
        .fold(0.0_f64, f64::max);

    println!("{}", name);
    println!("{}", "=".repeat(name.len()));
    print!("{}", BetaDisplay(beta));
    println!("  RSS                  = {:>.10}", rss);
    println!("  max |residual|       = {:>.10}", max_abs_residual);
    println!();
}

fn print_weight_diagnostics(weights: &[f64]) {
    let min_w = weights.iter().copied().fold(f64::INFINITY, f64::min);
    let max_w = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let num_downweighted = weights.iter().filter(|&&w| w < 0.999_999).count();

    println!("Weight Diagnostics");
    println!("==================");
    println!("Minimum weight           = {:>.10}", min_w);
    println!("Maximum weight           = {:>.10}", max_w);
    println!("Downweighted points      = {}", num_downweighted);
    println!();
}

fn print_pointwise_diagnostics(x: &[f64], y: &[f64], beta: &[f64], weights: &[f64], scale: f64) {
    println!("Pointwise Diagnostics");
    println!("=====================");
    println!(
        "{:>4} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "i", "x_i", "y_i", "yhat_i", "residual", "weight"
    );

    for i in 0..x.len() {
        let yhat = beta[0] + beta[1] * x[i];
        let r = y[i] - yhat;
        println!(
            "{:>4} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            i, x[i], y[i], yhat, r, weights[i]
        );
    }

    println!();
    println!("Robust scale estimate s = {:>.10}", scale);
    println!();
}

// ----------------------------
// Main Demonstration
// ----------------------------

fn main() -> Result<(), String> {
    let (x_data, y_data) = generate_contaminated_line_data();
    let xmat = design_matrix_for_line(&x_data);

    print_dataset(&x_data, &y_data);

    // Ordinary least-squares fit for comparison.
    let beta_ols = ordinary_least_squares(&xmat, &y_data)?;
    let residuals_ols = residuals(&xmat, &y_data, &beta_ols);
    print_fit_summary("Ordinary Least Squares Fit", &beta_ols, &residuals_ols);

    // IRLS robust fit using Huber weights.
    let huber_threshold = 1.345;
    let max_iters = 50;
    let tol = 1.0e-10;

    let robust_result = irls_huber(&xmat, &y_data, huber_threshold, max_iters, tol)?;

    println!("IRLS Robust Fit (Huber)");
    println!("=======================");
    print!("{}", BetaDisplay(&robust_result.beta));
    println!("  Converged            = {}", robust_result.converged);
    println!("  Iterations used      = {}", robust_result.iterations_used);
    println!("  Final scale          = {:>.10}", robust_result.final_scale);
    println!();

    print_fit_summary(
        "IRLS Residual Summary",
        &robust_result.beta,
        &robust_result.residuals,
    );

    print_weight_diagnostics(&robust_result.final_weights);
    print_pointwise_diagnostics(
        &x_data,
        &y_data,
        &robust_result.beta,
        &robust_result.final_weights,
        robust_result.final_scale,
    );

    println!("Interpretation");
    println!("==============");
    println!("The ordinary least-squares fit is influenced strongly by the injected outliers.");
    println!("The IRLS fit reduces that sensitivity by assigning smaller weights to points");
    println!("with large normalized residuals u_i = r_i / s, in accordance with equations");
    println!("(15.8.6) through (15.8.9). The resulting parameter estimate is therefore more");
    println!("representative of the central linear trend in the data.");

    Ok(())
}
```

Program 15.8.1 illustrates how robust estimation can be implemented as a sequence of weighted least-squares problems, providing a direct computational realization of the IRLS framework developed in Section 15.8.3. By updating weights based on normalized residuals, the method reduces sensitivity to outliers while retaining the efficiency of classical linear algebra techniques.

The example demonstrates that, although the ordinary least-squares solution minimizes the unweighted residual sum of squares, it can be significantly influenced by a small number of extreme observations. In contrast, the IRLS solution produces parameter estimates that better reflect the central structure of the data by limiting the influence of these points. The diagnostic outputs further highlight how robustness is achieved through adaptive weighting.

The modular structure of the implementation allows for straightforward extensions. Alternative loss functions can be incorporated by modifying the weight computation, and the linear solver can be replaced with more advanced or large-scale methods when needed. This provides a flexible foundation for exploring more advanced robust estimation techniques, including adaptive tuning strategies and robust uncertainty quantification, as discussed in subsequent sections.

## 15.8.4. Influence, Breakdown, and the Role of the Loss Function

The robustness of an estimator is governed by the shape of the loss function (\\rho), or equivalently, the influence function $\psi$. These functions determine how individual observations contribute to the estimation process and therefore control the sensitivity of the estimator to deviations in the data. By modifying the behavior of $\rho$ and $\psi$, one can design estimators that respond differently to small and large residuals.

In classical least squares, the influence function is linear, $\psi(u)=u$, so the influence of an observation grows without bound as the magnitude of its residual increases. This means that extreme observations can dominate the solution, since their contribution to the objective and to the estimating equations increases quadratically. As a result, least-squares estimators are highly sensitive to outliers and other forms of contamination.

In robust methods, the function $\psi(u)$ is constructed so that it is bounded or decreases for large values of $|u|$. This change fundamentally alters how extreme residuals are treated. Instead of allowing their influence to grow indefinitely, robust methods limit their contribution or even reduce it as the residual becomes larger. In some cases, the influence function may approach a constant value, while in others it may decrease toward zero, effectively discounting extreme observations.

This ensures that extreme observations cannot exert arbitrarily large influence on the estimate. As a consequence, the estimator becomes more representative of the majority of the data, rather than being driven by a small number of anomalous points. The behavior of the loss function at large residuals is therefore a key design choice in robust estimation, balancing sensitivity to genuine structure against resistance to contamination.

From the influence-function perspective, several important properties emerge. Robust losses reduce sensitivity to outliers by limiting the contribution of large residuals. The breakdown behavior improves, meaning that a larger fraction of contaminated data can be tolerated before the estimator becomes unreliable. The estimator becomes stable under contamination, maintaining reasonable performance even when the data deviate significantly from ideal assumptions.

Modern theory emphasizes that classical robust methods must be reconsidered in high-dimensional regimes, where robustness interacts with sparsity and computational constraints (Loh, 2025). In such settings, the design of the loss function and the associated algorithms must account not only for statistical properties but also for scalability and efficiency. This has led to renewed interest in robust methods that can be integrated with modern optimization techniques and large-scale numerical linear algebra.

### Rust Implementation

Following the discussion in Section 15.8.4 on the role of the loss function and influence function in determining estimator robustness, Program 15.8.2 provides a practical implementation that compares ordinary least squares, Huber regression, and Tukey bisquare regression within a unified IRLS framework. The program demonstrates how different choices of the influence function $\psi$ alter the weighting of residuals and thereby control the sensitivity of the estimator to outliers. By implementing multiple loss functions within the same computational structure, the code illustrates how bounded and redescending influence functions modify the contribution of extreme observations, directly reflecting the theoretical principles discussed in this section.

At the core of the implementation is the abstraction of the loss function through the `LossKind` enum, which encapsulates the behavior of the influence function $\psi(u)$ for different estimators. This design allows the program to switch seamlessly between ordinary least squares, Huber, and Tukey bisquare methods while reusing the same computational framework. The function `psi` implements the influence function corresponding to each loss, while the function `weight` computes the ratio $\psi(u)/u$, consistent with the weighting rule defined in Equation (15.8.7). This abstraction highlights how the choice of $\psi$ directly determines the weights used in the IRLS iterations.

The computation of normalized residuals $u_i = r_i / s$, as introduced in Equation (15.8.6), is performed within the `fit_regression` function. The residuals are first computed using the current parameter estimate, and a robust scale parameter is obtained using the MAD-based function `mad_scale`. These normalized residuals are then passed to `compute_weights`, which applies the selected loss function to produce observation-specific weights. This process ensures that the contribution of each data point is adjusted according to its consistency with the model.

The function `weighted_normal_equations` constructs the matrices $X^\top W X$ and $X^\top W y$, corresponding to the matrix formulation in Equation (15.8.9). These are solved using the Cholesky-based solver `solve_spd_cholesky`, ensuring numerical stability without explicit matrix inversion. The iterative loop in `fit_regression` repeatedly updates the parameter vector by solving these weighted systems, implementing the IRLS procedure described in Equation (15.8.8).

The program includes three distinct estimators. The ordinary least-squares case uses constant weights, reflecting the linear influence function $\psi(u) = u$. The Huber estimator introduces a bounded influence function, reducing the weight of large residuals while retaining their contribution. The Tukey bisquare estimator implements a redescending influence function, assigning zero weight to sufficiently large residuals and effectively removing extreme outliers from the estimation process. These differences are reflected directly in the computed weights and residuals, allowing for a clear comparison of estimator behavior.

The `main` function orchestrates the comparison by fitting all three models to the same contaminated dataset. It reports parameter estimates, residual statistics, and weight diagnostics, and provides a pointwise comparison of residuals and weights across methods. This detailed output makes it possible to observe how each estimator responds to outliers and how the choice of loss function influences both the fit and the robustness properties of the solution.

```rust
// Program 15.8.2: Influence, Breakdown, and the Role of the Loss Function
//
// Problem Statement:
// Compare the behavior of three regression estimators on contaminated data:
//
// 1. Ordinary least squares, whose influence function is ψ(u) = u
// 2. Huber robust regression, whose influence is bounded for large |u|
// 3. Tukey bisquare robust regression, whose influence returns to zero for sufficiently large |u|
//
// The program fits the straight-line model
//
//     y_i ≈ β_0 + β_1 x_i
//
// to a dataset containing several strong outliers. It reports the fitted coefficients,
// residual diagnostics, and pointwise weights in order to illustrate how the choice of
// loss function changes the estimator's sensitivity to contamination.
//
// The implementation uses iteratively reweighted least squares (IRLS) for the robust
// estimators and compares them against the ordinary least-squares solution. The code is
// fully self-contained and uses only the Rust standard library.

use std::fmt;

// ----------------------------
// Basic Linear Algebra Helpers
// ----------------------------

type Vector = Vec<f64>;
type Matrix = Vec<Vec<f64>>;

fn zeros_matrix(rows: usize, cols: usize) -> Matrix {
    vec![vec![0.0; cols]; rows]
}

fn mat_vec_mul(a: &Matrix, x: &[f64]) -> Vector {
    let rows = a.len();
    let cols = a[0].len();
    assert_eq!(cols, x.len(), "Matrix-vector dimension mismatch.");

    let mut y = vec![0.0; rows];
    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += a[i][j] * x[j];
        }
        y[i] = sum;
    }
    y
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Dot-product dimension mismatch.");
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

// ----------------------------
// SPD Solver by Cholesky
// ----------------------------

fn cholesky_decompose(a: &Matrix) -> Result<Matrix, String> {
    let n = a.len();
    if n == 0 || a[0].len() != n {
        return Err("Matrix must be nonempty and square for Cholesky decomposition.".to_string());
    }

    let mut l = zeros_matrix(n, n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }

            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "Matrix is not positive definite at diagonal entry {}.",
                        i
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    Ok(l)
}

fn forward_substitution(l: &Matrix, b: &[f64]) -> Result<Vector, String> {
    let n = l.len();
    if b.len() != n {
        return Err("Dimension mismatch in forward substitution.".to_string());
    }

    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        if l[i][i].abs() < 1.0e-15 {
            return Err(format!("Zero diagonal encountered in forward substitution at {}.", i));
        }
        y[i] = sum / l[i][i];
    }
    Ok(y)
}

fn backward_substitution_from_cholesky(l: &Matrix, y: &[f64]) -> Result<Vector, String> {
    let n = l.len();
    if y.len() != n {
        return Err("Dimension mismatch in backward substitution.".to_string());
    }

    let mut x = vec![0.0; n];
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j];
        }
        if l[i][i].abs() < 1.0e-15 {
            return Err(format!(
                "Zero diagonal encountered in backward substitution at {}.",
                i
            ));
        }
        x[i] = sum / l[i][i];
    }
    Ok(x)
}

fn solve_spd_cholesky(a: &Matrix, b: &[f64]) -> Result<Vector, String> {
    let l = cholesky_decompose(a)?;
    let y = forward_substitution(&l, b)?;
    let x = backward_substitution_from_cholesky(&l, &y)?;
    Ok(x)
}

// ----------------------------
// Robust Scale and Residuals
// ----------------------------

fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "Median of empty slice is undefined.");

    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn mad_scale(residuals: &[f64]) -> f64 {
    let med = median(residuals);
    let abs_dev: Vec<f64> = residuals.iter().map(|r| (r - med).abs()).collect();
    let mad = median(&abs_dev);
    (1.4826 * mad).max(1.0e-12)
}

fn design_matrix_for_line(x: &[f64]) -> Matrix {
    let mut xmat = Vec::with_capacity(x.len());
    for &xi in x {
        xmat.push(vec![1.0, xi]);
    }
    xmat
}

fn residuals(x: &Matrix, y: &[f64], beta: &[f64]) -> Vector {
    let yhat = mat_vec_mul(x, beta);
    y.iter().zip(yhat.iter()).map(|(yi, yhi)| yi - yhi).collect()
}

fn weighted_normal_equations(x: &Matrix, y: &[f64], w: &[f64]) -> (Matrix, Vector) {
    let n = x.len();
    let p = x[0].len();

    assert_eq!(y.len(), n, "Response length mismatch.");
    assert_eq!(w.len(), n, "Weight length mismatch.");

    let mut xtwx = zeros_matrix(p, p);
    let mut xtwy = vec![0.0; p];

    for i in 0..n {
        for j in 0..p {
            xtwy[j] += w[i] * x[i][j] * y[i];
            for k in 0..p {
                xtwx[j][k] += w[i] * x[i][j] * x[i][k];
            }
        }
    }

    (xtwx, xtwy)
}

fn ordinary_least_squares(x: &Matrix, y: &[f64]) -> Result<Vector, String> {
    let n = x.len();
    let one_weights = vec![1.0; n];
    let (xtx, xty) = weighted_normal_equations(x, y, &one_weights);
    solve_spd_cholesky(&xtx, &xty)
}

// ----------------------------
// Loss Families
// ----------------------------

#[derive(Clone, Copy, Debug)]
enum LossKind {
    LeastSquares,
    Huber { c: f64 },
    TukeyBisquare { c: f64 },
}

impl LossKind {
    fn name(&self) -> &'static str {
        match self {
            LossKind::LeastSquares => "Ordinary Least Squares",
            LossKind::Huber { .. } => "Huber IRLS",
            LossKind::TukeyBisquare { .. } => "Tukey Bisquare IRLS",
        }
    }

    fn psi(&self, u: f64) -> f64 {
        match *self {
            LossKind::LeastSquares => u,
            LossKind::Huber { c } => {
                if u.abs() <= c {
                    u
                } else {
                    c * u.signum()
                }
            }
            LossKind::TukeyBisquare { c } => {
                let au = u.abs();
                if au >= c {
                    0.0
                } else {
                    let t = 1.0 - (u / c) * (u / c);
                    u * t * t
                }
            }
        }
    }

    fn weight(&self, u: f64) -> f64 {
        match *self {
            LossKind::LeastSquares => 1.0,
            _ => {
                if u.abs() < 1.0e-15 {
                    1.0
                } else {
                    self.psi(u) / u
                }
            }
        }
    }
}

fn compute_weights(loss: LossKind, normalized_residuals: &[f64]) -> Vector {
    normalized_residuals.iter().map(|&u| loss.weight(u)).collect()
}

// ----------------------------
// Generic IRLS Solver
// ----------------------------

#[derive(Clone, Debug)]
struct RegressionResult {
    name: String,
    beta: Vector,
    converged: bool,
    iterations_used: usize,
    scale: f64,
    weights: Vector,
    residuals: Vector,
}

fn max_abs_difference(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vector length mismatch.");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn fit_regression(
    x: &Matrix,
    y: &[f64],
    loss: LossKind,
    max_iters: usize,
    tol: f64,
) -> Result<RegressionResult, String> {
    let n = x.len();
    if n == 0 {
        return Err("Empty dataset.".to_string());
    }
    if y.len() != n {
        return Err("Response length does not match design matrix.".to_string());
    }

    if let LossKind::LeastSquares = loss {
        let beta = ordinary_least_squares(x, y)?;
        let residual_vec = residuals(x, y, &beta);
        let weights = vec![1.0; n];
        let scale = mad_scale(&residual_vec);

        return Ok(RegressionResult {
            name: loss.name().to_string(),
            beta,
            converged: true,
            iterations_used: 1,
            scale,
            weights,
            residuals: residual_vec,
        });
    }

    let mut beta = ordinary_least_squares(x, y)?;
    let mut weights = vec![1.0; n];
    let mut scale: f64;
    let mut converged = false;

    for iter in 0..max_iters {
        let residual_vec = residuals(x, y, &beta);
        scale = mad_scale(&residual_vec);
        let normalized: Vector = residual_vec.iter().map(|r| r / scale).collect();
        weights = compute_weights(loss, &normalized);

        let (xtwx, xtwy) = weighted_normal_equations(x, y, &weights);
        let beta_new = solve_spd_cholesky(&xtwx, &xtwy)?;
        let change = max_abs_difference(&beta_new, &beta);
        beta = beta_new;

        if change < tol {
            converged = true;
            let final_residuals = residuals(x, y, &beta);
            let final_scale = mad_scale(&final_residuals);
            let final_normalized: Vector = final_residuals.iter().map(|r| r / final_scale).collect();
            let final_weights = compute_weights(loss, &final_normalized);

            return Ok(RegressionResult {
                name: loss.name().to_string(),
                beta,
                converged,
                iterations_used: iter + 1,
                scale: final_scale,
                weights: final_weights,
                residuals: final_residuals,
            });
        }
    }

    let final_residuals = residuals(x, y, &beta);
    let final_scale = mad_scale(&final_residuals);
    let final_normalized: Vector = final_residuals.iter().map(|r| r / final_scale).collect();
    let final_weights = compute_weights(loss, &final_normalized);

    Ok(RegressionResult {
        name: loss.name().to_string(),
        beta,
        converged,
        iterations_used: max_iters,
        scale: final_scale,
        weights: final_weights,
        residuals: final_residuals,
    })
}

// ----------------------------
// Test Data
// ----------------------------

fn generate_contaminated_line_data() -> (Vector, Vector) {
    let x = vec![
        -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0,
        2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    ];

    // Baseline model: y = 2 + 1.5 x, with mild deterministic perturbations.
    let mut y = vec![
        -5.48, -4.79, -4.02, -3.20, -2.47, -1.77, -0.92, -0.28, 0.55, 1.28, 2.06, 2.83, 3.53,
        4.28, 5.05, 5.78, 6.46, 7.31, 8.02, 8.79, 9.47,
    ];

    // Inject outliers.
    y[3] += 5.5;
    y[11] -= 6.0;
    y[17] += 4.5;

    (x, y)
}

// ----------------------------
// Reporting
// ----------------------------

struct BetaDisplay<'a>(&'a [f64]);

impl<'a> fmt::Display for BetaDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, value) in self.0.iter().enumerate() {
            writeln!(f, "  beta[{}] = {:>.10}", i, value)?;
        }
        Ok(())
    }
}

fn print_dataset(x: &[f64], y: &[f64]) {
    println!("Contaminated Regression Dataset");
    println!("===============================");
    for i in 0..x.len() {
        println!("i = {:>2}, x = {:>.6}, y = {:>.6}", i, x[i], y[i]);
    }
    println!();
}

fn print_result_summary(result: &RegressionResult) {
    let rss = dot(&result.residuals, &result.residuals);
    let max_abs_residual = result
        .residuals
        .iter()
        .map(|r| r.abs())
        .fold(0.0_f64, f64::max);

    let min_weight = result.weights.iter().copied().fold(f64::INFINITY, f64::min);
    let max_weight = result
        .weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let num_downweighted = result.weights.iter().filter(|&&w| w < 0.999_999).count();

    println!("{}", result.name);
    println!("{}", "=".repeat(result.name.len()));
    print!("{}", BetaDisplay(&result.beta));
    println!("  Converged            = {}", result.converged);
    println!("  Iterations used      = {}", result.iterations_used);
    println!("  Scale estimate       = {:>.10}", result.scale);
    println!("  RSS                  = {:>.10}", rss);
    println!("  max |residual|       = {:>.10}", max_abs_residual);
    println!("  Minimum weight       = {:>.10}", min_weight);
    println!("  Maximum weight       = {:>.10}", max_weight);
    println!("  Downweighted points  = {}", num_downweighted);
    println!();
}

fn print_comparison_table(x: &[f64], y: &[f64], ols: &RegressionResult, huber: &RegressionResult, tukey: &RegressionResult) {
    println!("Pointwise Influence Comparison");
    println!("=============================");
    println!(
        "{:>3} {:>9} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "i", "x_i", "y_i", "r_OLS", "r_Huber", "r_Tukey", "w_OLS", "w_Huber", "w_Tukey"
    );

    for i in 0..x.len() {
        println!(
            "{:>3} {:>9.3} {:>10.3} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            i,
            x[i],
            y[i],
            ols.residuals[i],
            huber.residuals[i],
            tukey.residuals[i],
            ols.weights[i],
            huber.weights[i],
            tukey.weights[i],
        );
    }

    println!();
}

fn print_interpretation(ols: &RegressionResult, huber: &RegressionResult, tukey: &RegressionResult) {
    println!("Interpretation");
    println!("==============");
    println!("The ordinary least-squares estimator uses the linear influence function ψ(u) = u,");
    println!("so all observations effectively retain unit weight and large residuals can exert");
    println!("substantial leverage on the fitted line.");
    println!();

    println!("The Huber estimator bounds the influence of large normalized residuals. Its weights");
    println!("decrease for outlying points but remain positive, so extreme observations are");
    println!("downweighted rather than completely discarded.");
    println!();

    println!("The Tukey bisquare estimator is redescending: for sufficiently large |u|, the");
    println!("influence function returns to zero and the corresponding weight becomes zero.");
    println!("This produces stronger rejection of severe outliers.");
    println!();

    println!("Coefficient Comparison");
    println!("----------------------");
    println!(
        "OLS    : beta0 = {:>.10}, beta1 = {:>.10}",
        ols.beta[0], ols.beta[1]
    );
    println!(
        "Huber  : beta0 = {:>.10}, beta1 = {:>.10}",
        huber.beta[0], huber.beta[1]
    );
    println!(
        "Tukey  : beta0 = {:>.10}, beta1 = {:>.10}",
        tukey.beta[0], tukey.beta[1]
    );
    println!();
}

// ----------------------------
// Main Demonstration
// ----------------------------

fn main() -> Result<(), String> {
    let (x_data, y_data) = generate_contaminated_line_data();
    let xmat = design_matrix_for_line(&x_data);

    let max_iters = 100;
    let tol = 1.0e-10;

    let ols_result = fit_regression(&xmat, &y_data, LossKind::LeastSquares, max_iters, tol)?;
    let huber_result = fit_regression(&xmat, &y_data, LossKind::Huber { c: 1.345 }, max_iters, tol)?;
    let tukey_result =
        fit_regression(&xmat, &y_data, LossKind::TukeyBisquare { c: 4.685 }, max_iters, tol)?;

    print_dataset(&x_data, &y_data);
    print_result_summary(&ols_result);
    print_result_summary(&huber_result);
    print_result_summary(&tukey_result);
    print_comparison_table(&x_data, &y_data, &ols_result, &huber_result, &tukey_result);
    print_interpretation(&ols_result, &huber_result, &tukey_result);

    Ok(())
}
```

Program 15.8.2 demonstrates how the theoretical concepts of influence functions and loss design translate into practical computational behavior. By comparing least squares, Huber, and Tukey estimators, the program illustrates how bounded and redescending influence functions reduce sensitivity to extreme observations and improve robustness under contamination.

The results show that while ordinary least squares treats all observations equally and is therefore highly sensitive to outliers, robust methods adaptively modify the contribution of each data point through the weighting mechanism. The Huber estimator provides a balance between sensitivity and robustness by downweighting large residuals, while the Tukey estimator offers stronger resistance by effectively discarding extreme observations.

The modular structure of the implementation allows for easy extension to other loss functions and robust estimation strategies. This provides a flexible foundation for exploring more advanced topics, such as adaptive tuning of loss parameters, high-dimensional robustness, and integration with modern optimization methods. In this way, the program serves as a concrete bridge between the theoretical analysis of robustness and its practical realization in numerical computation.

## 15.8.5. Modern Developments: Adaptive Tuning and Robust Inference

Robust loss functions typically include tuning parameters that control how aggressively large residuals are downweighted. Common examples include the threshold parameter in the Huber loss and the cutoff parameter in the Tukey bisquare function. These parameters determine the transition between quadratic behavior for small residuals and reduced influence for large residuals, and therefore play a central role in the performance of the estimator.

Choosing these parameters manually is often unsatisfactory. Fixed choices may work well for some datasets but poorly for others, especially when the scale of the noise or the degree of contamination is unknown. Manual tuning can also introduce subjectivity and reduce reproducibility, making it difficult to apply robust methods consistently across different problems.

Recent developments propose data-adaptive tuning strategies, in which the tuning parameters are selected automatically based on the data. This is often formulated as a bilevel optimization problem, where the primary optimization determines the model parameters, and a secondary optimization selects the tuning parameters to achieve desirable statistical properties. In this framework, tuning becomes an integral part of the estimation process rather than an external choice, allowing the method to adapt to the structure and noise characteristics of the data (Zhang et al., 2023).

At the same time, high-dimensional theory has revealed that classical robust loss functions may behave unexpectedly under heavy-tailed contamination. In particular, when the number of parameters is large relative to the sample size, standard robust methods may not provide sufficient control over variability or bias. This has motivated the incorporation of additional regularization into robust estimation procedures, combining ideas from sparsity-promoting methods and robust statistics to achieve both stability and interpretability (Adomaityte et al., 2023).

Robust estimation must also include robust uncertainty quantification. It is not sufficient to obtain parameter estimates that are resistant to outliers; the associated confidence intervals must also reflect the true variability of the estimator under contamination. Recent work addresses this by constructing robust confidence intervals using M-estimation and IRLS, with data-driven tuning of both the loss function and the scale parameters (Wang et al., 2025). These approaches aim to provide uncertainty estimates that remain valid even when classical assumptions are violated.

A key practical lesson is that robust point estimation and robust interval estimation must be designed together. The choice of loss function, tuning parameters, and computational algorithm affects not only the estimated parameters but also the reliability of the associated uncertainty measures. Effective robust methods therefore integrate estimation and inference within a unified framework, ensuring that both aspects remain stable and meaningful in the presence of non-ideal data.

### Computational and Implementation Considerations

Robust estimation is well suited to modern numerical computing environments because of its close connection to linear algebra and its iterative structure. In particular, methods such as IRLS are dominated by repeated linear algebra operations, with each iteration involving the formation and solution of weighted least-squares systems. This structure aligns well with optimized numerical kernels and allows robust methods to benefit directly from advances in linear algebra libraries and hardware architectures.

The computation is naturally organized as an iterative refinement of weights. At each step, residuals are evaluated, weights are updated based on the chosen loss function, and a weighted linear system is solved to update the parameters. This repeated structure makes the algorithm predictable and modular, facilitating both implementation and optimization. It also allows for convergence monitoring through changes in weights and parameter values.

Another important feature is that robust estimation is amenable to parallelization. Residual evaluation and weight computation are typically independent across data points, making them suitable for data-parallel execution. This is particularly advantageous for large datasets, where the cost of these operations can be significant. By distributing these computations across multiple cores or processing units, one can achieve substantial performance gains.

These structural properties lead to several practical implementation guidelines. The functions $\rho(u)$, $\psi(u)$, and $w(u)$ must be implemented carefully to avoid numerical issues such as overflow or underflow, especially when residuals become large. Stable evaluation of these functions is essential for maintaining both robustness and numerical reliability.

For the linear algebra components, factorization-based methods such as Cholesky or QR decompositions should be used instead of explicit matrix inversion. These approaches provide improved numerical stability and are better suited to repeated solves within an iterative framework. When the number of observations $n$ is large, it is also important to avoid forming dense matrices unnecessarily. Iterative solvers and sparse representations can significantly reduce both computational cost and memory usage.

Diagnostics play a crucial role in practical robust estimation. Monitoring effective weights, the number of downweighted points, and the sensitivity of the solution to tuning parameters provides insight into how the method is interacting with the data. These diagnostics help identify whether the model is dominated by a small subset of observations or whether the chosen loss function and parameters are appropriate.

In Rust, these computations map naturally to efficient loops and stable linear algebra routines. The language’s emphasis on performance, memory safety, and explicit control over data structures makes it well suited for implementing robust estimation algorithms. By combining careful numerical design with efficient implementation, robust estimation can be made both practical and performant in modern computational settings.

### Rust Implementation

Following the development of adaptive tuning strategies and robust uncertainty quantification in Section 15.8.5, Program 15.8.3 provides a complete implementation of data-driven robust regression using the Huber loss within the IRLS framework. In contrast to earlier implementations with fixed tuning parameters, this program selects the Huber threshold automatically using a cross-validation procedure, thereby adapting the degree of robustness to the observed data. It further extends the estimation process by incorporating bootstrap-based confidence intervals, illustrating how robust point estimation and robust inference can be integrated within a single computational pipeline. The implementation reflects the modern perspective that tuning, estimation, and uncertainty quantification must be treated jointly to ensure reliable performance under non-ideal data conditions.

At the core of the implementation is the IRLS procedure for Huber regression, which follows the iterative structure introduced in Section 15.8.3. The function `irls_huber` computes residuals and normalized residuals as in Equation (15.8.6), updates weights using the Huber influence function according to Equation (15.8.7), and solves the weighted least-squares system described in Equation (15.8.9). The use of a Cholesky-based solver ensures numerical stability and avoids explicit matrix inversion, which is particularly important in repeated iterative solves.

The key extension in this program is the adaptive selection of the tuning parameter. The function `adaptive_huber_tuning` evaluates a grid of candidate threshold values using a k-fold cross-validation strategy. For each candidate value, the model is trained on a subset of the data and evaluated on a validation set using a robust error metric based on median absolute residuals. This process embodies the data-driven tuning approach described in the section, allowing the algorithm to select a value of the threshold parameter that is best suited to the level of contamination present in the data.

The implementation also includes robust uncertainty quantification through bootstrap resampling. The function `bootstrap_huber_intervals` generates multiple resampled datasets, fits the robust model to each sample, and constructs confidence intervals using empirical percentiles. This approach avoids reliance on classical variance formulas, which may be invalid under non-Gaussian or contaminated noise, and instead provides intervals that reflect the actual variability of the estimator under the observed data conditions.

The program further incorporates diagnostic tools to assess the behavior of the estimator. The reporting functions display residual statistics, weight distributions, and the number of downweighted observations, providing insight into how the algorithm interacts with the data. The pointwise diagnostics illustrate how individual observations are treated under the adaptive weighting scheme, highlighting the connection between the influence function and the resulting fit.

The `main` function integrates all components into a coherent workflow. It first computes an ordinary least-squares baseline, then performs adaptive tuning to select the Huber threshold, and finally fits the robust model using the selected parameter. The bootstrap procedure is then applied to quantify uncertainty, and the results are presented along with diagnostic summaries. This sequence reflects the integrated estimation-and-inference paradigm emphasized in Section 15.8.5.

```rust
// Program 15.8.3: Adaptive Huber IRLS with Data-Driven Tuning and Bootstrap Confidence Intervals
//
// Problem Statement:
// Implement a robust linear regression procedure that integrates
//
// 1. Iteratively reweighted least squares (IRLS) with the Huber loss
// 2. Data-adaptive selection of the Huber tuning parameter c
// 3. Robust uncertainty quantification through bootstrap confidence intervals
// 4. Diagnostic reporting for weights, outlier sensitivity, and tuning stability
//
// The program fits the straight-line model
//
//     y_i ≈ β_0 + β_1 x_i
//
// to contaminated data. It first evaluates a grid of candidate Huber thresholds
// by robust cross-validation, then fits the final model using the selected value,
// and finally computes percentile bootstrap intervals for the regression coefficients.
//
// The implementation is self-contained, uses only the Rust standard library,
// and includes a complete fn main() so that cargo run works out of the box.

use std::fmt;

// ----------------------------
// Basic Linear Algebra Helpers
// ----------------------------

type Vector = Vec<f64>;
type Matrix = Vec<Vec<f64>>;

fn zeros_matrix(rows: usize, cols: usize) -> Matrix {
    vec![vec![0.0; cols]; rows]
}

fn mat_vec_mul(a: &Matrix, x: &[f64]) -> Vector {
    let rows = a.len();
    let cols = a[0].len();
    assert_eq!(cols, x.len(), "Matrix-vector dimension mismatch.");

    let mut y = vec![0.0; rows];
    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += a[i][j] * x[j];
        }
        y[i] = sum;
    }
    y
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Dot-product dimension mismatch.");
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn cholesky_decompose(a: &Matrix) -> Result<Matrix, String> {
    let n = a.len();
    if n == 0 || a[0].len() != n {
        return Err("Matrix must be nonempty and square for Cholesky decomposition.".to_string());
    }

    let mut l = zeros_matrix(n, n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }

            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "Matrix is not positive definite at diagonal entry {}.",
                        i
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    Ok(l)
}

fn forward_substitution(l: &Matrix, b: &[f64]) -> Result<Vector, String> {
    let n = l.len();
    if b.len() != n {
        return Err("Dimension mismatch in forward substitution.".to_string());
    }

    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        if l[i][i].abs() < 1.0e-15 {
            return Err(format!("Zero diagonal encountered in forward substitution at {}.", i));
        }
        y[i] = sum / l[i][i];
    }
    Ok(y)
}

fn backward_substitution_from_cholesky(l: &Matrix, y: &[f64]) -> Result<Vector, String> {
    let n = l.len();
    if y.len() != n {
        return Err("Dimension mismatch in backward substitution.".to_string());
    }

    let mut x = vec![0.0; n];
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j];
        }
        if l[i][i].abs() < 1.0e-15 {
            return Err(format!(
                "Zero diagonal encountered in backward substitution at {}.",
                i
            ));
        }
        x[i] = sum / l[i][i];
    }
    Ok(x)
}

fn solve_spd_cholesky(a: &Matrix, b: &[f64]) -> Result<Vector, String> {
    let l = cholesky_decompose(a)?;
    let y = forward_substitution(&l, b)?;
    backward_substitution_from_cholesky(&l, &y)
}

// ----------------------------
// Robust Statistics Helpers
// ----------------------------

fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "Median of empty slice is undefined.");

    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn mad_scale(residuals: &[f64]) -> f64 {
    let med = median(residuals);
    let abs_dev: Vec<f64> = residuals.iter().map(|r| (r - med).abs()).collect();
    let mad = median(&abs_dev);
    (1.4826 * mad).max(1.0e-12)
}

fn percentile(values: &[f64], p: f64) -> f64 {
    assert!(!values.is_empty(), "Percentile of empty slice is undefined.");
    assert!(
        (0.0..=1.0).contains(&p),
        "Percentile level must lie in [0,1]."
    );

    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = v.len();
    if n == 1 {
        return v[0];
    }

    let pos = p * (n as f64 - 1.0);
    let i0 = pos.floor() as usize;
    let i1 = pos.ceil() as usize;
    let t = pos - i0 as f64;

    if i0 == i1 {
        v[i0]
    } else {
        (1.0 - t) * v[i0] + t * v[i1]
    }
}

// ----------------------------
// Model Construction
// ----------------------------

fn design_matrix_for_line(x: &[f64]) -> Matrix {
    let mut xmat = Vec::with_capacity(x.len());
    for &xi in x {
        xmat.push(vec![1.0, xi]);
    }
    xmat
}

fn residuals(x: &Matrix, y: &[f64], beta: &[f64]) -> Vector {
    let yhat = mat_vec_mul(x, beta);
    y.iter().zip(yhat.iter()).map(|(yi, yhi)| yi - yhi).collect()
}

fn weighted_normal_equations(x: &Matrix, y: &[f64], w: &[f64]) -> (Matrix, Vector) {
    let n = x.len();
    let p = x[0].len();

    assert_eq!(y.len(), n, "Response length mismatch.");
    assert_eq!(w.len(), n, "Weight length mismatch.");

    let mut xtwx = zeros_matrix(p, p);
    let mut xtwy = vec![0.0; p];

    for i in 0..n {
        for j in 0..p {
            xtwy[j] += w[i] * x[i][j] * y[i];
            for k in 0..p {
                xtwx[j][k] += w[i] * x[i][j] * x[i][k];
            }
        }
    }

    (xtwx, xtwy)
}

fn ordinary_least_squares(x: &Matrix, y: &[f64]) -> Result<Vector, String> {
    let weights = vec![1.0; x.len()];
    let (xtx, xty) = weighted_normal_equations(x, y, &weights);
    solve_spd_cholesky(&xtx, &xty)
}

// ----------------------------
// Huber Loss and IRLS
// ----------------------------

fn huber_psi(u: f64, c: f64) -> f64 {
    if u.abs() <= c {
        u
    } else {
        c * u.signum()
    }
}

fn huber_weight(u: f64, c: f64) -> f64 {
    if u.abs() < 1.0e-15 {
        1.0
    } else {
        huber_psi(u, c) / u
    }
}

fn huber_weights(normalized_residuals: &[f64], c: f64) -> Vector {
    normalized_residuals
        .iter()
        .map(|&u| huber_weight(u, c))
        .collect()
}

#[derive(Clone, Debug)]
struct IrlsResult {
    beta: Vector,
    converged: bool,
    iterations_used: usize,
    scale: f64,
    weights: Vector,
    residuals: Vector,
}

fn max_abs_difference(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vector length mismatch.");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn irls_huber(
    x: &Matrix,
    y: &[f64],
    c: f64,
    max_iters: usize,
    tol: f64,
) -> Result<IrlsResult, String> {
    let n = x.len();
    if n == 0 {
        return Err("Empty dataset.".to_string());
    }
    if y.len() != n {
        return Err("Response length does not match design matrix.".to_string());
    }

    let mut beta = ordinary_least_squares(x, y)?;

    for iter in 0..max_iters {
        let current_residuals = residuals(x, y, &beta);
        let scale = mad_scale(&current_residuals);
        let normalized: Vector = current_residuals.iter().map(|r| r / scale).collect();
        let weights = huber_weights(&normalized, c);

        let (xtwx, xtwy) = weighted_normal_equations(x, y, &weights);
        let beta_new = solve_spd_cholesky(&xtwx, &xtwy)?;
        let change = max_abs_difference(&beta_new, &beta);

        beta = beta_new;

        if change < tol {
            let final_residuals = residuals(x, y, &beta);
            let final_scale = mad_scale(&final_residuals);
            let final_normalized: Vector = final_residuals.iter().map(|r| r / final_scale).collect();
            let final_weights = huber_weights(&final_normalized, c);

            return Ok(IrlsResult {
                beta,
                converged: true,
                iterations_used: iter + 1,
                scale: final_scale,
                weights: final_weights,
                residuals: final_residuals,
            });
        }
    }

    let final_residuals = residuals(x, y, &beta);
    let final_scale = mad_scale(&final_residuals);
    let final_normalized: Vector = final_residuals.iter().map(|r| r / final_scale).collect();
    let final_weights = huber_weights(&final_normalized, c);

    Ok(IrlsResult {
        beta,
        converged: false,
        iterations_used: max_iters,
        scale: final_scale,
        weights: final_weights,
        residuals: final_residuals,
    })
}

// ----------------------------
// Adaptive Tuning by Robust Cross-Validation
// ----------------------------

#[derive(Clone, Debug)]
struct TuningRecord {
    c: f64,
    cv_score: f64,
}

fn select_rows_matrix(a: &Matrix, indices: &[usize]) -> Matrix {
    let mut out = Vec::with_capacity(indices.len());
    for &i in indices {
        out.push(a[i].clone());
    }
    out
}

fn select_rows_vector(v: &[f64], indices: &[usize]) -> Vector {
    let mut out = Vec::with_capacity(indices.len());
    for &i in indices {
        out.push(v[i]);
    }
    out
}

fn kfold_indices(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut folds = vec![Vec::<usize>::new(); k];
    for i in 0..n {
        folds[i % k].push(i);
    }
    folds
}

fn robust_validation_score(residuals: &[f64]) -> f64 {
    let abs_res: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
    median(&abs_res)
}

fn adaptive_huber_tuning(
    x: &Matrix,
    y: &[f64],
    c_grid: &[f64],
    k_folds: usize,
    max_iters: usize,
    tol: f64,
) -> Result<(f64, Vec<TuningRecord>), String> {
    let n = x.len();
    if n < k_folds || k_folds < 2 {
        return Err("Invalid number of folds for cross-validation.".to_string());
    }

    let folds = kfold_indices(n, k_folds);
    let mut records = Vec::with_capacity(c_grid.len());

    for &c in c_grid {
        let mut fold_scores = Vec::with_capacity(k_folds);

        for valid_idx in folds.iter().take(k_folds) {
            let mut train_idx = Vec::new();
            for j in 0..n {
                if !valid_idx.contains(&j) {
                    train_idx.push(j);
                }
            }

            let x_train = select_rows_matrix(x, &train_idx);
            let y_train = select_rows_vector(y, &train_idx);
            let x_valid = select_rows_matrix(x, valid_idx);
            let y_valid = select_rows_vector(y, valid_idx);

            let fit = irls_huber(&x_train, &y_train, c, max_iters, tol)?;
            let valid_residuals = residuals(&x_valid, &y_valid, &fit.beta);
            let score = robust_validation_score(&valid_residuals);
            fold_scores.push(score);
        }

        let avg_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        records.push(TuningRecord { c, cv_score: avg_score });
    }

    let best = records
        .iter()
        .min_by(|a, b| a.cv_score.partial_cmp(&b.cv_score).unwrap())
        .ok_or_else(|| "Failed to choose an adaptive tuning parameter.".to_string())?;

    Ok((best.c, records))
}

// ----------------------------
// Bootstrap Confidence Intervals
// ----------------------------

#[derive(Clone, Debug)]
struct BootstrapIntervals {
    beta0_low: f64,
    beta0_high: f64,
    beta1_low: f64,
    beta1_high: f64,
    num_successful: usize,
}

#[derive(Clone, Debug)]
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

    fn gen_index(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "Upper bound must be positive.");
        (self.next_u64() as usize) % upper
    }
}

fn bootstrap_pairs_sample(x: &[f64], y: &[f64], rng: &mut LcgRng) -> (Vector, Vector) {
    let n = x.len();
    let mut xb = Vec::with_capacity(n);
    let mut yb = Vec::with_capacity(n);

    for _ in 0..n {
        let idx = rng.gen_index(n);
        xb.push(x[idx]);
        yb.push(y[idx]);
    }

    (xb, yb)
}

fn bootstrap_huber_intervals(
    x: &[f64],
    y: &[f64],
    c: f64,
    num_bootstrap: usize,
    alpha: f64,
    max_iters: usize,
    tol: f64,
    seed: u64,
) -> Result<BootstrapIntervals, String> {
    let mut rng = LcgRng::new(seed);
    let mut beta0_samples = Vec::with_capacity(num_bootstrap);
    let mut beta1_samples = Vec::with_capacity(num_bootstrap);

    for _ in 0..num_bootstrap {
        let (xb, yb) = bootstrap_pairs_sample(x, y, &mut rng);
        let xmat_b = design_matrix_for_line(&xb);

        if let Ok(fit) = irls_huber(&xmat_b, &yb, c, max_iters, tol) {
            beta0_samples.push(fit.beta[0]);
            beta1_samples.push(fit.beta[1]);
        }
    }

    if beta0_samples.len() < 10 {
        return Err("Too few successful bootstrap fits for interval estimation.".to_string());
    }

    let lower = 0.5 * alpha;
    let upper = 1.0 - 0.5 * alpha;

    Ok(BootstrapIntervals {
        beta0_low: percentile(&beta0_samples, lower),
        beta0_high: percentile(&beta0_samples, upper),
        beta1_low: percentile(&beta1_samples, lower),
        beta1_high: percentile(&beta1_samples, upper),
        num_successful: beta0_samples.len(),
    })
}

// ----------------------------
// Demonstration Dataset
// ----------------------------

fn generate_contaminated_line_data() -> (Vector, Vector) {
    let x = vec![
        -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0,
        1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
    ];

    // Baseline model: y = 1.75 + 1.20 x, plus mild deterministic perturbation.
    let mut y = vec![
        -5.41, -4.76, -4.28, -3.68, -3.03, -2.48, -1.89, -1.19, -0.63, -0.08, 0.58, 1.16, 1.78,
        2.36, 2.92, 3.61, 4.12, 4.80, 5.31, 5.99, 6.61, 7.12, 7.73, 8.29, 8.98,
    ];

    // Inject several outliers.
    y[4] += 4.8;
    y[13] -= 5.4;
    y[20] += 4.2;
    y[22] -= 3.6;

    (x, y)
}

// ----------------------------
// Reporting Helpers
// ----------------------------

struct BetaDisplay<'a>(&'a [f64]);

impl<'a> fmt::Display for BetaDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, value) in self.0.iter().enumerate() {
            writeln!(f, "  beta[{}] = {:>.10}", i, value)?;
        }
        Ok(())
    }
}

fn print_dataset(x: &[f64], y: &[f64]) {
    println!("Adaptive Robust Regression Dataset");
    println!("=================================");
    for i in 0..x.len() {
        println!("i = {:>2}, x = {:>.6}, y = {:>.6}", i, x[i], y[i]);
    }
    println!();
}

fn print_fit_summary(name: &str, beta: &[f64], residuals: &[f64], weights: &[f64], scale: f64) {
    let rss = dot(residuals, residuals);
    let max_abs_residual = residuals
        .iter()
        .map(|r| r.abs())
        .fold(0.0_f64, f64::max);
    let min_w = weights.iter().copied().fold(f64::INFINITY, f64::min);
    let max_w = weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let num_downweighted = weights.iter().filter(|&&w| w < 0.999_999).count();

    println!("{}", name);
    println!("{}", "=".repeat(name.len()));
    print!("{}", BetaDisplay(beta));
    println!("  Scale estimate       = {:>.10}", scale);
    println!("  RSS                  = {:>.10}", rss);
    println!("  max |residual|       = {:>.10}", max_abs_residual);
    println!("  Minimum weight       = {:>.10}", min_w);
    println!("  Maximum weight       = {:>.10}", max_w);
    println!("  Downweighted points  = {}", num_downweighted);
    println!();
}

fn print_tuning_table(records: &[TuningRecord]) {
    println!("Adaptive Tuning Diagnostics");
    println!("==========================");
    println!("{:>12} {:>18}", "c", "CV score");

    for rec in records {
        println!("{:>12.6} {:>18.10}", rec.c, rec.cv_score);
    }
    println!();
}

fn print_pointwise_diagnostics(x: &[f64], y: &[f64], beta: &[f64], weights: &[f64]) {
    println!("Pointwise Weight Diagnostics");
    println!("============================");
    println!(
        "{:>4} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "i", "x_i", "y_i", "yhat_i", "residual", "weight"
    );

    for i in 0..x.len() {
        let yhat = beta[0] + beta[1] * x[i];
        let r = y[i] - yhat;
        println!(
            "{:>4} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            i, x[i], y[i], yhat, r, weights[i]
        );
    }
    println!();
}

// ----------------------------
// Main Demonstration
// ----------------------------

fn main() -> Result<(), String> {
    let (x_data, y_data) = generate_contaminated_line_data();
    let xmat = design_matrix_for_line(&x_data);

    let max_iters = 100;
    let tol = 1.0e-10;

    let c_grid = vec![0.75, 1.00, 1.20, 1.345, 1.50, 1.75, 2.00, 2.50, 3.00];
    let k_folds = 5;

    print_dataset(&x_data, &y_data);

    let ols_beta = ordinary_least_squares(&xmat, &y_data)?;
    let ols_residuals = residuals(&xmat, &y_data, &ols_beta);
    let ols_weights = vec![1.0; x_data.len()];
    let ols_scale = mad_scale(&ols_residuals);
    print_fit_summary(
        "Ordinary Least Squares Fit",
        &ols_beta,
        &ols_residuals,
        &ols_weights,
        ols_scale,
    );

    let (best_c, tuning_records) =
        adaptive_huber_tuning(&xmat, &y_data, &c_grid, k_folds, max_iters, tol)?;
    print_tuning_table(&tuning_records);

    println!("Selected Huber threshold");
    println!("========================");
    println!("  c* = {:>.10}", best_c);
    println!();

    let robust_fit = irls_huber(&xmat, &y_data, best_c, max_iters, tol)?;
    println!("Adaptive Huber IRLS Fit");
    println!("=======================");
    print!("{}", BetaDisplay(&robust_fit.beta));
    println!("  Converged            = {}", robust_fit.converged);
    println!("  Iterations used      = {}", robust_fit.iterations_used);
    println!("  Selected c           = {:>.10}", best_c);
    println!();

    print_fit_summary(
        "Adaptive Robust Residual Summary",
        &robust_fit.beta,
        &robust_fit.residuals,
        &robust_fit.weights,
        robust_fit.scale,
    );

    print_pointwise_diagnostics(
        &x_data,
        &y_data,
        &robust_fit.beta,
        &robust_fit.weights,
    );

    let intervals = bootstrap_huber_intervals(
        &x_data,
        &y_data,
        best_c,
        400,
        0.05,
        max_iters,
        tol,
        20260412,
    )?;

    println!("Bootstrap Confidence Intervals");
    println!("==============================");
    println!(
        "  beta[0] 95% CI = [{:>.10}, {:>.10}]",
        intervals.beta0_low, intervals.beta0_high
    );
    println!(
        "  beta[1] 95% CI = [{:>.10}, {:>.10}]",
        intervals.beta1_low, intervals.beta1_high
    );
    println!(
        "  Successful bootstrap fits = {}",
        intervals.num_successful
    );
    println!();

    println!("Interpretation");
    println!("==============");
    println!("The adaptive tuning stage selects the Huber threshold from the data rather than");
    println!("fixing it a priori. This makes the amount of downweighting responsive to the");
    println!("observed contamination pattern and improves reproducibility across datasets.");
    println!();
    println!("The final IRLS fit combines robust point estimation with robust interval estimation.");
    println!("The percentile bootstrap intervals provide uncertainty quantification that is less");
    println!("sensitive to contamination than classical formulas derived under ideal Gaussian");
    println!("assumptions. Together, the adaptive threshold selection, iterative weighting, and");
    println!("bootstrap diagnostics illustrate the integrated estimation-and-inference workflow");
    println!("described in Section 15.8.5.");

    Ok(())
}
```

Program 15.8.3 demonstrates how modern robust estimation combines adaptive tuning, iterative optimization, and data-driven inference into a unified computational framework. By selecting the tuning parameter based on the data, the method avoids arbitrary choices and adapts automatically to varying levels of contamination. The integration of bootstrap confidence intervals further ensures that uncertainty estimates remain meaningful even when classical assumptions are violated.

The results highlight the advantages of robust methods in practical settings. While ordinary least squares is sensitive to outliers and can produce biased estimates, the adaptive IRLS approach yields parameter estimates that are stable and representative of the underlying structure. The diagnostic outputs provide transparency into the estimation process, allowing practitioners to assess the influence of individual observations and the effectiveness of the chosen tuning parameter.

The modular design of the implementation makes it straightforward to extend the framework to other loss functions, tuning strategies, and large-scale computational settings. This provides a foundation for exploring advanced topics such as high-dimensional robust regression, regularized M-estimation, and parallel implementations. In this way, the program serves as a bridge between classical robust statistics and modern computational practice, illustrating how theoretical concepts can be translated into reliable and scalable algorithms.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/EIk76GtFG9hexyAARXMq.5","tags":[]}

# 15.9. Markov Chain Monte Carlo

The previous sections developed parameter estimation and uncertainty quantification primarily within the framework of optimization and local approximations. These approaches focus on identifying point estimates and approximating uncertainty through curvature or resampling techniques. While effective in many settings, they often rely on simplifying assumptions, such as local quadratic behavior or asymptotic normality, which may not fully capture the structure of the parameter space.

In contrast, Bayesian inference aims to characterize the full posterior distribution of model parameters. Rather than summarizing uncertainty through intervals or covariance matrices alone, the Bayesian approach represents uncertainty as a probability distribution over parameters. This perspective allows for a more complete description of uncertainty, including multimodality, asymmetry, and complex dependencies between parameters.

For realistic models, especially nonlinear ones, this posterior distribution cannot be computed analytically. The normalization constant of the posterior is typically intractable, and direct evaluation of integrals required for expectations or marginal distributions is computationally prohibitive. As a result, numerical methods are required to approximate the posterior distribution and the quantities derived from it.

Markov Chain Monte Carlo (MCMC) provides a general framework for sampling from complex probability distributions. Rather than computing integrals directly, MCMC constructs a stochastic process whose stationary distribution is the target posterior. By generating a sequence of samples from this process, one obtains an empirical representation of the distribution. Expectations, variances, and confidence regions can then be estimated from the generated samples, replacing analytic integration with statistical averaging.

The key idea is that, although it may be difficult to sample directly from the posterior, it is often possible to design a Markov chain that moves through parameter space in such a way that its long-run behavior reproduces the desired distribution. The design and analysis of such chains involve both probabilistic and numerical considerations, including convergence properties, mixing behavior, and computational efficiency.

This section introduces the mathematical foundations of MCMC, basic algorithms, and modern computational considerations. It builds on the earlier discussion of likelihood-based inference and uncertainty quantification, extending these ideas to a fully probabilistic framework in which uncertainty is explored through sampling rather than approximated locally.

## 15.9.1. Bayesian Formulation and Posterior Distribution

Let $\theta \in \mathbb{R}^d$ denote the model parameters, and let $D$ denote the observed data. In the Bayesian framework, uncertainty about the parameters is represented probabilistically, and inference proceeds by updating prior beliefs in light of the observed data. This update is governed by Bayes’ theorem:

$$p(\theta \mid D)= \frac{p(D \mid \theta) \, p(\theta)}{p(D)} \tag{15.9.1}$$

where $p(D \mid \theta)$ is the likelihood, $p(\theta)$ is the prior distribution, and $p(D)$ is the normalization constant, often referred to as the evidence or marginal likelihood.

Each component of this expression has a distinct interpretation. The likelihood $p(D \mid \theta)$ measures how well the model with parameters $\theta$ explains the observed data. The prior $p(\theta)$ encodes any available information or assumptions about the parameters before observing the data. The posterior $p(\theta \mid D)$ combines these two sources of information, providing an updated distribution that reflects both prior knowledge and empirical evidence.

The normalization constant is given by:

$$p(D) = \int p(D \mid \theta)\, p(\theta)\, d\theta \tag{15.9.2}$$

which ensures that the posterior distribution integrates to one. However, this integral is typically intractable for high-dimensional models, especially when the likelihood is complex or the parameter space is large. Computing this quantity requires integrating over all possible parameter values, which becomes computationally prohibitive in most realistic applications.

As a consequence, direct evaluation of the posterior distribution is generally not feasible. Instead, Bayesian computation focuses on methods that avoid explicit calculation of the normalization constant. Markov Chain Monte Carlo methods achieve this by generating samples as:

$$\{\theta^{(k)}\}_{k=1}^{K} \sim p(\theta \mid D) \tag{15.9.3}$$

whose empirical distribution approximates the posterior.

Once such samples are available, expectations of functions of the parameters can be approximated by sample averages:

$$\mathbb{E}[g(\theta)] \approx \frac{1}{K} \sum_{k=1}^{K} g(\theta^{(k)}) \tag{15.9.4}$$

This approximation follows from the law of large numbers and provides a practical way to compute quantities of interest, such as means, variances, and credible intervals.

In this way, the problem of evaluating high-dimensional integrals is transformed into the problem of generating representative samples from the posterior distribution. This shift from integration to sampling is the central idea underlying MCMC methods and forms the basis for modern Bayesian computation.

## 15.9.2. Markov Chains and Stationary Distributions

MCMC methods construct a Markov chain:

$$\theta^{(0)} \to \theta^{(1)} \to \cdots \to \theta^{(k)} \tag{15.9.5}$$

whose long-run behavior reproduces the target posterior distribution $p(\theta \mid D)$. The sequence of parameter values evolves according to a stochastic rule, with each new state depending only on the current state. This memoryless property is the defining feature of a Markov chain.

A Markov chain is defined by a transition kernel:

$$P(\theta' \mid \theta) \tag{15.9.6}$$

which specifies the probability of moving from the current state $\theta$ to a new state $\theta'$. This kernel encapsulates the entire dynamics of the chain, determining how it explores the parameter space and how quickly it converges to its long-run distribution.

The goal in MCMC is to design a transition kernel such that the desired posterior distribution is stationary. A distribution (p(\\theta)) is said to be stationary for the chain if:

$$\int P(\theta' \mid \theta)\, p(\theta)\, d\theta = p(\theta') \tag{15.9.7}$$

This condition states that if the current state is distributed according to $p(\theta)$, then the next state will also follow the same distribution. In other words, once the chain reaches the target distribution, it remains there under the dynamics of the transition kernel.

A sufficient condition for stationarity is detailed balance:

$$p(\theta)\, P(\theta' \mid \theta)= p(\theta')\, P(\theta \mid \theta') \tag{15.9.8}$$

This condition expresses a form of symmetry in the transitions between states. It requires that the probability flow from $\theta$ to $\theta'$ is exactly balanced by the flow from $\theta'$ back to $\theta$. When this condition is satisfied, the target distribution is guaranteed to be stationary.

MCMC algorithms are constructed to satisfy this condition. By carefully designing the transition kernel to obey detailed balance with respect to the posterior, one ensures that the Markov chain will converge to the desired distribution under appropriate conditions. Different MCMC methods correspond to different choices of transition kernels, each with its own trade-offs in terms of efficiency, convergence speed, and ease of implementation.

This framework provides the theoretical foundation for sampling-based inference, linking probabilistic modeling with stochastic processes and numerical computation.

## 15.9.3. The Metropolis–Hastings Algorithm

The most widely used MCMC method is the Metropolis–Hastings algorithm. It provides a general procedure for constructing a Markov chain whose stationary distribution is a desired target distribution, such as the posterior (p(\\theta \\mid D)). Its flexibility lies in the fact that it allows arbitrary proposal mechanisms, provided that the acceptance rule is chosen appropriately.

Given a proposal distribution:

$$q(\theta' \mid \theta) \tag{15.9.9}$$

a candidate state $\theta'$ is generated based on the current state $\theta$. The proposal distribution determines how the chain explores the parameter space, and its choice has a significant impact on the efficiency of the algorithm.

The proposed move is then accepted with probability:

$$\alpha(\theta,\theta')= \min\left(1,\frac{p(\theta')\, q(\theta \mid \theta')}{p(\theta)\, q(\theta' \mid \theta)}\right) \tag{15.9.10}$$

This acceptance probability ensures that the resulting Markov chain satisfies detailed balance with respect to the target distribution. If the candidate state has higher probability under the target distribution, it is accepted with probability one. If it has lower probability, it may still be accepted with a probability that decreases according to the ratio of target densities and proposal probabilities.

If the proposed move is accepted, the next state of the chain is set to (\\theta^{(k+1)} = \\theta'). Otherwise, the current state is retained, so that (\\theta^{(k+1)} = \\theta). This possibility of rejecting proposals introduces dependence between successive samples but is essential for ensuring that the chain has the correct stationary distribution.

In Bayesian problems, the acceptance ratio simplifies to:

$$\alpha(\theta,\theta')= \min\left(1,\frac{p(D \mid \theta')\, p(\theta')}{p(D \mid \theta)\, p(\theta)}\cdot\frac{q(\theta \mid \theta')}{q(\theta' \mid \theta)}\right) \tag{15.9.11}$$

Notably, the normalization constant $p(D)$ cancels from the ratio. This is a key advantage of MCMC methods, since it allows sampling from the posterior distribution without computing the often intractable marginal likelihood. Only the likelihood and prior need to be evaluated up to proportionality.

A particularly important special case is the random-walk Metropolis algorithm, in which proposals are generated by adding a random perturbation to the current state:

$$\theta' = \theta + \eta, \qquad \eta \sim \mathcal{N}(0,\Sigma) \tag{15.9.12}$$

Here, the proposal distribution is symmetric, so that $q(\theta' \mid \theta) = q(\theta \mid \theta')$, and the acceptance ratio simplifies further. The covariance matrix $\Sigma$ controls the scale and orientation of the proposed steps, and its choice is critical for balancing exploration and acceptance. If the steps are too small, the chain explores the space slowly; if they are too large, proposals are frequently rejected.

Overall, the Metropolis–Hastings algorithm provides a simple yet powerful mechanism for sampling from complex distributions, forming the foundation for many modern MCMC methods.

### Rust Implementation

Following the development of the Metropolis–Hastings algorithm in Section 15.9.3, Program 15.9.1 provides a practical implementation of posterior sampling using a random-walk proposal mechanism. The algorithm constructs a Markov chain whose stationary distribution coincides with the target posterior distribution by generating candidate states and accepting or rejecting them according to the probability defined in Equation (15.9.10). In Bayesian settings, this acceptance probability simplifies as shown in Equation (15.9.11), eliminating the need to compute the normalization constant of the posterior. The implementation demonstrates how the proposal mechanism in Equation (15.9.12) governs the exploration of the parameter space and highlights the interplay between proposal scale, acceptance behavior, and sampling efficiency in practical computation.

At the core of the implementation is the construction of the Metropolis–Hastings transition mechanism, which follows directly from the acceptance rule defined in Equation (15.9.10). The function `metropolis_hastings` maintains the current state of the chain and repeatedly generates candidate states using a proposal distribution. The proposal mechanism is implemented in `propose_random_walk`, which corresponds to the random-walk formulation in Equation (15.9.12), where a Gaussian perturbation is added to the current parameter vector. Because this proposal is symmetric, the ratio of proposal densities cancels, and the acceptance decision depends only on the ratio of posterior densities.

The evaluation of the posterior distribution is handled through the functions `log_likelihood`, `log_prior`, and `log_posterior`. The likelihood function computes the probability of the observed data under the current parameter values, while the prior encodes prior beliefs about the parameters. These components are combined to form the posterior density up to proportionality, consistent with Equation (15.9.11), where the normalization constant does not appear. The use of logarithms ensures numerical stability, particularly when dealing with products of small probabilities.

The acceptance step within `metropolis_hastings` implements the probabilistic rule for transitioning between states. A candidate is accepted if it improves the posterior density or with a probability that depends on the ratio of posterior values when it does not. This mechanism introduces dependence between successive samples, as described in the section, but ensures that the resulting chain satisfies detailed balance and converges to the desired stationary distribution.

The program also includes auxiliary components that support practical implementation. The `LcgRng` structure provides a simple random number generator along with a Box–Muller transform for generating Gaussian samples required by the proposal distribution. The functions `percentile` and `posterior_summary` compute summary statistics of the sampled chain, allowing estimation of posterior means and credible intervals. These summaries illustrate how samples from the Markov chain can be used to approximate expectations under the posterior distribution.

The `main` function demonstrates the complete workflow. It generates synthetic data from a known linear model, defines prior distributions and proposal parameters, and runs the Metropolis–Hastings algorithm to produce a sequence of samples. After discarding an initial burn-in period, the remaining samples are used to compute posterior summaries. The reported acceptance rate and trace preview provide diagnostic insight into the behavior of the chain, including its mixing and convergence characteristics.

```rust
// Program 15.9.1: Random-Walk Metropolis–Hastings Sampling for Bayesian Linear Regression
//
// Problem Statement:
// Implement the Metropolis–Hastings algorithm for Bayesian inference in a simple
// linear regression model
//
//     y_i = β_0 + β_1 x_i + ε_i,   ε_i ~ N(0, σ^2),
//
// where σ is assumed known. The program constructs a Markov chain whose stationary
// distribution is the posterior distribution of the parameter vector
//
//     θ = (β_0, β_1)^T.
//
// A symmetric Gaussian random-walk proposal is used:
//
//     θ' = θ + η,   η ~ N(0, Σ),
//
// so the proposal ratio cancels in the Metropolis–Hastings acceptance probability.
// The code generates posterior samples, reports the acceptance rate, computes posterior
// means and credible intervals, and compares them with the true parameters used to
// generate the synthetic data.
//
// The implementation is fully self-contained and uses only the Rust standard library.

use std::f64::consts::PI;

// ----------------------------
// Basic Types
// ----------------------------

type Vector = Vec<f64>;

// ----------------------------
// Simple Random Number Generator
// ----------------------------

#[derive(Clone, Debug)]
struct LcgRng {
    state: u64,
    has_spare_normal: bool,
    spare_normal: f64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            has_spare_normal: false,
            spare_normal: 0.0,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        // Uniform in [0, 1).
        let u = self.next_u64() >> 11;
        (u as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn standard_normal(&mut self) -> f64 {
        if self.has_spare_normal {
            self.has_spare_normal = false;
            return self.spare_normal;
        }

        let mut u1 = self.next_f64();
        while u1 <= 1.0e-15 {
            u1 = self.next_f64();
        }
        let u2 = self.next_f64();

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;

        let z0 = r * theta.cos();
        let z1 = r * theta.sin();

        self.spare_normal = z1;
        self.has_spare_normal = true;

        z0
    }
}

// ----------------------------
// Model and Posterior
// ----------------------------

#[derive(Clone, Copy, Debug)]
struct Params {
    beta0: f64,
    beta1: f64,
}

#[derive(Clone, Copy, Debug)]
struct Prior {
    mean_beta0: f64,
    mean_beta1: f64,
    sd_beta0: f64,
    sd_beta1: f64,
}

#[derive(Clone, Copy, Debug)]
struct Proposal {
    step_beta0: f64,
    step_beta1: f64,
}

#[derive(Clone, Debug)]
struct Dataset {
    x: Vector,
    y: Vector,
    sigma: f64,
}

fn log_normal_pdf(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / sd;
    -0.5 * z * z - sd.ln() - 0.5 * (2.0 * PI).ln()
}

fn log_prior(theta: Params, prior: Prior) -> f64 {
    log_normal_pdf(theta.beta0, prior.mean_beta0, prior.sd_beta0)
        + log_normal_pdf(theta.beta1, prior.mean_beta1, prior.sd_beta1)
}

fn log_likelihood(theta: Params, data: &Dataset) -> f64 {
    let n = data.x.len();
    let sigma2 = data.sigma * data.sigma;
    let norm_const = -0.5 * (2.0 * PI * sigma2).ln();

    let mut sum = 0.0;
    for i in 0..n {
        let mu = theta.beta0 + theta.beta1 * data.x[i];
        let r = data.y[i] - mu;
        sum += norm_const - 0.5 * r * r / sigma2;
    }
    sum
}

fn log_posterior(theta: Params, data: &Dataset, prior: Prior) -> f64 {
    log_likelihood(theta, data) + log_prior(theta, prior)
}

// ----------------------------
// Metropolis–Hastings Sampler
// ----------------------------

#[derive(Clone, Debug)]
struct MhResult {
    chain: Vec<Params>,
    accepted: usize,
    acceptance_rate: f64,
}

fn propose_random_walk(current: Params, proposal: Proposal, rng: &mut LcgRng) -> Params {
    Params {
        beta0: current.beta0 + proposal.step_beta0 * rng.standard_normal(),
        beta1: current.beta1 + proposal.step_beta1 * rng.standard_normal(),
    }
}

fn metropolis_hastings(
    initial: Params,
    num_samples: usize,
    proposal: Proposal,
    data: &Dataset,
    prior: Prior,
    rng: &mut LcgRng,
) -> MhResult {
    let mut chain = Vec::with_capacity(num_samples);
    let mut current = initial;
    let mut current_log_post = log_posterior(current, data, prior);
    let mut accepted = 0usize;

    for _ in 0..num_samples {
        let candidate = propose_random_walk(current, proposal, rng);
        let candidate_log_post = log_posterior(candidate, data, prior);

        // For a symmetric random-walk proposal, the q-ratio cancels.
        let log_alpha = candidate_log_post - current_log_post;
        let accept = if log_alpha >= 0.0 {
            true
        } else {
            rng.next_f64().ln() < log_alpha
        };

        if accept {
            current = candidate;
            current_log_post = candidate_log_post;
            accepted += 1;
        }

        chain.push(current);
    }

    MhResult {
        chain,
        accepted,
        acceptance_rate: accepted as f64 / num_samples as f64,
    }
}

// ----------------------------
// Posterior Summaries
// ----------------------------

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(values: &[f64], p: f64) -> f64 {
    assert!(!values.is_empty(), "Percentile requires nonempty data.");
    assert!(
        (0.0..=1.0).contains(&p),
        "Percentile level must lie in [0, 1]."
    );

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }

    let pos = p * (n as f64 - 1.0);
    let i0 = pos.floor() as usize;
    let i1 = pos.ceil() as usize;
    let t = pos - i0 as f64;

    if i0 == i1 {
        sorted[i0]
    } else {
        (1.0 - t) * sorted[i0] + t * sorted[i1]
    }
}

fn extract_parameter_samples(chain: &[Params]) -> (Vector, Vector) {
    let mut beta0 = Vec::with_capacity(chain.len());
    let mut beta1 = Vec::with_capacity(chain.len());

    for &theta in chain {
        beta0.push(theta.beta0);
        beta1.push(theta.beta1);
    }

    (beta0, beta1)
}

fn posterior_summary(samples: &[f64]) -> (f64, f64, f64) {
    let m = mean(samples);
    let q025 = percentile(samples, 0.025);
    let q975 = percentile(samples, 0.975);
    (m, q025, q975)
}

// ----------------------------
// Synthetic Data Generation
// ----------------------------

fn generate_synthetic_data(
    beta0_true: f64,
    beta1_true: f64,
    sigma: f64,
    n: usize,
    rng: &mut LcgRng,
) -> Dataset {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    let x_min = -2.0;
    let x_max = 2.0;

    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        let xi = x_min + (x_max - x_min) * t;
        let noise = sigma * rng.standard_normal();
        let yi = beta0_true + beta1_true * xi + noise;

        x.push(xi);
        y.push(yi);
    }

    Dataset { x, y, sigma }
}

// ----------------------------
// Reporting Helpers
// ----------------------------

fn print_dataset(data: &Dataset) {
    println!("Synthetic Bayesian Regression Dataset");
    println!("=====================================");
    println!("Known noise standard deviation sigma = {:>.6}", data.sigma);
    println!("Number of observations               = {}", data.x.len());
    println!();

    println!("{:>4} {:>14} {:>14}", "i", "x_i", "y_i");
    for i in 0..data.x.len() {
        println!("{:>4} {:>14.6} {:>14.6}", i, data.x[i], data.y[i]);
    }
    println!();
}

fn print_trace_preview(chain: &[Params], count: usize) {
    println!("Trace Preview");
    println!("=============");
    println!("{:>6} {:>16} {:>16}", "iter", "beta0", "beta1");

    let m = count.min(chain.len());
    for (k, theta) in chain.iter().enumerate().take(m) {
        println!("{:>6} {:>16.8} {:>16.8}", k, theta.beta0, theta.beta1);
    }
    println!();
}

fn print_posterior_summary(name: &str, samples: &[f64], truth: f64) {
    let (m, q025, q975) = posterior_summary(samples);

    println!("{}", name);
    println!("{}", "=".repeat(name.len()));
    println!("Posterior mean          = {:>.10}", m);
    println!("95% credible interval   = [{:>.10}, {:>.10}]", q025, q975);
    println!("True value              = {:>.10}", truth);
    println!();
}

// ----------------------------
// Main Demonstration
// ----------------------------

fn main() {
    // True parameters used to generate the synthetic data.
    let beta0_true = 1.50;
    let beta1_true = -0.80;
    let sigma = 0.35;
    let n = 30;

    // Random seed for reproducibility.
    let mut rng = LcgRng::new(20260412);

    let data = generate_synthetic_data(beta0_true, beta1_true, sigma, n, &mut rng);
    print_dataset(&data);

    // Prior specification.
    let prior = Prior {
        mean_beta0: 0.0,
        mean_beta1: 0.0,
        sd_beta0: 5.0,
        sd_beta1: 5.0,
    };

    // Initial point of the chain.
    let initial = Params {
        beta0: 0.0,
        beta1: 0.0,
    };

    // Symmetric Gaussian random-walk proposal.
    let proposal = Proposal {
        step_beta0: 0.12,
        step_beta1: 0.10,
    };

    // MCMC control parameters.
    let num_samples = 25_000usize;
    let burn_in = 5_000usize;

    let mh_result = metropolis_hastings(
        initial,
        num_samples,
        proposal,
        &data,
        prior,
        &mut rng,
    );

    println!("Metropolis–Hastings Configuration");
    println!("=================================");
    println!("Total samples            = {}", num_samples);
    println!("Burn-in                  = {}", burn_in);
    println!("Proposal step beta0      = {:>.6}", proposal.step_beta0);
    println!("Proposal step beta1      = {:>.6}", proposal.step_beta1);
    println!("Accepted proposals       = {}", mh_result.accepted);
    println!("Acceptance rate          = {:>.6}", mh_result.acceptance_rate);
    println!();

    print_trace_preview(&mh_result.chain, 20);

    let posterior_chain = &mh_result.chain[burn_in..];
    let (beta0_samples, beta1_samples) = extract_parameter_samples(posterior_chain);

    print_posterior_summary("Posterior Summary for beta0", &beta0_samples, beta0_true);
    print_posterior_summary("Posterior Summary for beta1", &beta1_samples, beta1_true);

    println!("Interpretation");
    println!("==============");
    println!("The Metropolis–Hastings chain provides an empirical approximation of the");
    println!("posterior distribution of (beta0, beta1). Because the proposal is a symmetric");
    println!("Gaussian random walk, the proposal density ratio cancels from the acceptance");
    println!("probability, leaving only the posterior ratio described in Equations (15.9.10)");
    println!("and (15.9.11). The resulting samples can be used to estimate posterior means,");
    println!("credible intervals, and other expectations without evaluating the normalization");
    println!("constant of the posterior distribution.");
}
```

Program 15.9.1 demonstrates how the Metropolis–Hastings algorithm translates the theoretical construction of a Markov chain with a desired stationary distribution into a practical computational procedure. By combining a proposal mechanism with an acceptance rule that enforces detailed balance, the method enables sampling from complex posterior distributions without requiring normalization constants.

The implementation highlights several important aspects of practical MCMC computation. The choice of proposal distribution, particularly its scale, plays a crucial role in determining the efficiency of the algorithm. The acceptance rate provides a useful diagnostic for assessing whether the chain is exploring the parameter space effectively. Additionally, the use of log-probabilities ensures numerical stability in evaluating likelihoods and priors.

The modular structure of the code allows for straightforward extensions to more complex models and proposal strategies. More advanced methods, such as adaptive proposals, Langevin dynamics, or Hamiltonian Monte Carlo, can be built upon the same framework to improve efficiency in high-dimensional settings. In this way, the program provides a foundation for understanding both classical and modern developments in Markov chain Monte Carlo methods.

## 15.9.4. Efficiency, Mixing, and Convergence

The usefulness of MCMC depends critically on how efficiently the Markov chain explores the posterior distribution. Although the algorithm guarantees convergence to the correct distribution under appropriate conditions, the rate at which this convergence occurs and the quality of the resulting samples determine the practical value of the method. Poorly performing chains may require a large number of iterations to produce reliable estimates, making efficiency a central concern in MCMC.

Several key concepts are used to characterize the behavior of a Markov chain.

***Burn-in:*** initial samples discarded before convergence.\
At the beginning of the simulation, the chain may be far from the target distribution, especially if the initial value is poorly chosen. During this transient phase, the samples do not accurately reflect the posterior and are typically discarded. Determining an appropriate burn-in period is important, as discarding too few samples can bias estimates, while discarding too many increases computational cost unnecessarily.

***Mixing:*** how quickly the chain explores the state space.\
A well-mixing chain moves efficiently through the parameter space, visiting different regions in proportion to their posterior probability. Poor mixing occurs when the chain becomes trapped in certain regions or moves slowly between them, which can happen in multimodal or highly correlated distributions. Good mixing is essential for obtaining representative samples within a reasonable number of iterations.

***Autocorrelation:*** dependence between successive samples.\
Because each state of the chain depends on the previous one, successive samples are correlated. High autocorrelation means that the chain provides less new information at each step, reducing the efficiency of the sampling process. Low autocorrelation is therefore desirable, as it indicates that the chain is producing more nearly independent samples.

These effects are summarized by the effective sample size:

$$N_{\mathrm{eff}} = \frac{K}{1 + 2 \sum_{k=1}^{\infty} \rho_k} \tag{15.9.13}$$

where $\rho_k$ are the autocorrelations at lag $k$. This quantity represents the number of independent samples that would provide the same amount of information as the correlated samples produced by the chain. Even if a large number of samples $K$ is generated, strong autocorrelation can reduce the effective sample size substantially.

Poor mixing leads to low effective sample size, even if many samples are generated. This highlights that simply increasing the number of iterations does not necessarily improve the quality of the results. Instead, the design of the algorithm and the choice of proposal distribution play a crucial role in determining sampling efficiency.

To assess the performance of an MCMC simulation, several diagnostic tools are commonly used. Trace plots display the evolution of the chain over time and can reveal issues such as lack of convergence or poor exploration. Autocorrelation functions quantify the dependence between samples and help evaluate mixing behavior. Multiple-chain comparisons, in which several chains are run from different initial values, provide additional insight into convergence by checking whether all chains produce consistent results.

Modern studies emphasize that naive MCMC implementations can be inefficient for high-dimensional or strongly correlated problems (Vrugt et al., 2025). In such settings, standard algorithms like random-walk Metropolis may explore the parameter space very slowly, leading to poor mixing and low effective sample size. This has motivated the development of more advanced methods that incorporate gradient information, adapt proposal distributions, or exploit problem structure to improve efficiency.

### Rust Implementation

Following the discussion in Section 15.9.4 on efficiency, mixing, and convergence in Markov chain Monte Carlo methods, Program 15.9.2 provides a practical implementation of diagnostic tools for assessing the quality of samples generated by the Metropolis–Hastings algorithm. Although the algorithm guarantees convergence to the correct stationary distribution, its practical usefulness depends on how effectively the chain explores the parameter space. This program extends the basic sampler by incorporating burn-in handling, autocorrelation analysis, effective sample size estimation, and multiple-chain comparison. Through these components, it demonstrates how convergence diagnostics can be used to evaluate whether the samples obtained from the chain are reliable for statistical inference and how the concepts summarized in Equation (15.9.13) translate into computational practice.

At the core of the implementation is the Metropolis–Hastings sampling procedure, which generates a Markov chain whose stationary distribution is the posterior distribution of the model parameters. The function `metropolis_hastings` constructs this chain by repeatedly proposing new states using a random-walk mechanism and accepting or rejecting them according to the acceptance probability described in Equation (15.9.10). Because a symmetric Gaussian proposal is used, the proposal ratio cancels, and the acceptance decision depends only on the posterior ratio as described in Equation (15.9.11). This iterative process produces a sequence of dependent samples that form the basis for subsequent diagnostic analysis.

The program explicitly incorporates the concept of burn-in by discarding an initial portion of the chain before computing posterior summaries. This reflects the idea that early samples may not represent the target distribution if the chain is initialized far from equilibrium. The functions `extract_parameter_samples` and subsequent summary routines operate only on the post–burn-in portion of the chain, ensuring that the reported statistics are based on samples drawn from the stationary regime.

To quantify mixing and dependence between samples, the implementation includes functions for computing autocovariance and autocorrelation. The function `autocorrelation` evaluates the dependence between samples separated by a specified lag, while `autocorrelation_table` produces a sequence of such values for multiple lags. These quantities provide a direct numerical measure of how quickly the chain forgets its past states. Slow decay of autocorrelation indicates poor mixing, whereas rapid decay suggests that the chain is exploring the parameter space efficiently.

The effective sample size is computed using the function `effective_sample_size`, which implements the definition given in Equation (15.9.13). This function accumulates autocorrelation values across lags and converts the total number of generated samples into an equivalent number of independent samples. This is a crucial diagnostic, since a large number of correlated samples may contain significantly less information than the same number of independent observations.

The program also includes a multiple-chain analysis, in which several chains are initialized from widely separated starting points. The function `summarize_chain` computes summary statistics for each chain, including posterior means and effective sample sizes. These summaries are then compared using `print_chain_comparison`, allowing the user to assess whether different chains converge to the same region of the parameter space. Agreement across chains provides strong evidence of convergence, while discrepancies may indicate poor mixing or multimodality.

Finally, the `main` function integrates all diagnostic components into a cohesive workflow. It generates synthetic data, runs multiple Metropolis–Hastings chains, computes autocorrelation functions, evaluates effective sample sizes, and compares posterior summaries across chains. This comprehensive approach demonstrates how the theoretical concepts of burn-in, mixing, and convergence can be translated into concrete computational diagnostics for evaluating MCMC performance.

```rust
// Program 15.9.2: MCMC Efficiency, Mixing, and Convergence Diagnostics for Random-Walk Metropolis
//
// Problem Statement:
// Implement a Metropolis sampler for Bayesian linear regression and augment it with
// practical diagnostics for burn-in, mixing, autocorrelation, and effective sample size.
// The program runs multiple chains from dispersed initial values, reports acceptance
// rates, prints trace previews, computes autocorrelations, estimates the effective
// sample size
//
//     N_eff = K / (1 + 2 * sum_{k>=1} rho_k),
//
// and compares posterior summaries across chains after burn-in. The goal is to
// illustrate that the usefulness of MCMC depends not only on correctness of the
// stationary distribution but also on how efficiently the chain explores it.
//
// The implementation is fully self-contained and uses only the Rust standard library.

use std::f64::consts::PI;

// ----------------------------
// Basic Types
// ----------------------------

type Vector = Vec<f64>;

// ----------------------------
// Simple Random Number Generator
// ----------------------------

#[derive(Clone, Debug)]
struct LcgRng {
    state: u64,
    has_spare_normal: bool,
    spare_normal: f64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            has_spare_normal: false,
            spare_normal: 0.0,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        let u = self.next_u64() >> 11;
        (u as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn standard_normal(&mut self) -> f64 {
        if self.has_spare_normal {
            self.has_spare_normal = false;
            return self.spare_normal;
        }

        let mut u1 = self.next_f64();
        while u1 <= 1.0e-15 {
            u1 = self.next_f64();
        }
        let u2 = self.next_f64();

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;

        let z0 = r * theta.cos();
        let z1 = r * theta.sin();

        self.spare_normal = z1;
        self.has_spare_normal = true;
        z0
    }
}

// ----------------------------
// Bayesian Linear Regression Model
// ----------------------------

#[derive(Clone, Copy, Debug)]
struct Params {
    beta0: f64,
    beta1: f64,
}

#[derive(Clone, Copy, Debug)]
struct Prior {
    mean_beta0: f64,
    mean_beta1: f64,
    sd_beta0: f64,
    sd_beta1: f64,
}

#[derive(Clone, Copy, Debug)]
struct Proposal {
    step_beta0: f64,
    step_beta1: f64,
}

#[derive(Clone, Debug)]
struct Dataset {
    x: Vector,
    y: Vector,
    sigma: f64,
}

fn log_normal_pdf(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / sd;
    -0.5 * z * z - sd.ln() - 0.5 * (2.0 * PI).ln()
}

fn log_prior(theta: Params, prior: Prior) -> f64 {
    log_normal_pdf(theta.beta0, prior.mean_beta0, prior.sd_beta0)
        + log_normal_pdf(theta.beta1, prior.mean_beta1, prior.sd_beta1)
}

fn log_likelihood(theta: Params, data: &Dataset) -> f64 {
    let sigma2 = data.sigma * data.sigma;
    let log_norm = -0.5 * (2.0 * PI * sigma2).ln();

    let mut sum = 0.0;
    for i in 0..data.x.len() {
        let mu = theta.beta0 + theta.beta1 * data.x[i];
        let r = data.y[i] - mu;
        sum += log_norm - 0.5 * r * r / sigma2;
    }
    sum
}

fn log_posterior(theta: Params, data: &Dataset, prior: Prior) -> f64 {
    log_likelihood(theta, data) + log_prior(theta, prior)
}

// ----------------------------
// Metropolis Sampler
// ----------------------------

#[derive(Clone, Debug)]
struct MhResult {
    chain: Vec<Params>,
    accepted: usize,
    acceptance_rate: f64,
}

fn propose_random_walk(current: Params, proposal: Proposal, rng: &mut LcgRng) -> Params {
    Params {
        beta0: current.beta0 + proposal.step_beta0 * rng.standard_normal(),
        beta1: current.beta1 + proposal.step_beta1 * rng.standard_normal(),
    }
}

fn metropolis_hastings(
    initial: Params,
    num_samples: usize,
    proposal: Proposal,
    data: &Dataset,
    prior: Prior,
    rng: &mut LcgRng,
) -> MhResult {
    let mut chain = Vec::with_capacity(num_samples);
    let mut current = initial;
    let mut current_log_post = log_posterior(current, data, prior);
    let mut accepted = 0usize;

    for _ in 0..num_samples {
        let candidate = propose_random_walk(current, proposal, rng);
        let candidate_log_post = log_posterior(candidate, data, prior);

        let log_alpha = candidate_log_post - current_log_post;
        let accept = if log_alpha >= 0.0 {
            true
        } else {
            rng.next_f64().ln() < log_alpha
        };

        if accept {
            current = candidate;
            current_log_post = candidate_log_post;
            accepted += 1;
        }

        chain.push(current);
    }

    MhResult {
        chain,
        accepted,
        acceptance_rate: accepted as f64 / num_samples as f64,
    }
}

// ----------------------------
// Synthetic Data
// ----------------------------

fn generate_synthetic_data(
    beta0_true: f64,
    beta1_true: f64,
    sigma: f64,
    n: usize,
    rng: &mut LcgRng,
) -> Dataset {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    let x_min = -2.5;
    let x_max = 2.5;

    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        let xi = x_min + (x_max - x_min) * t;
        let yi = beta0_true + beta1_true * xi + sigma * rng.standard_normal();

        x.push(xi);
        y.push(yi);
    }

    Dataset { x, y, sigma }
}

// ----------------------------
// Chain Utilities
// ----------------------------

fn extract_parameter_samples(chain: &[Params]) -> (Vector, Vector) {
    let mut beta0 = Vec::with_capacity(chain.len());
    let mut beta1 = Vec::with_capacity(chain.len());

    for &theta in chain {
        beta0.push(theta.beta0);
        beta1.push(theta.beta1);
    }

    (beta0, beta1)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    let m = mean(values);
    let mut sum = 0.0;
    for &v in values {
        let d = v - m;
        sum += d * d;
    }
    sum / (values.len() as f64 - 1.0)
}

fn percentile(values: &[f64], p: f64) -> f64 {
    assert!(!values.is_empty(), "Percentile requires nonempty data.");
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if sorted.len() == 1 {
        return sorted[0];
    }

    let pos = p * (sorted.len() as f64 - 1.0);
    let i0 = pos.floor() as usize;
    let i1 = pos.ceil() as usize;
    let t = pos - i0 as f64;

    if i0 == i1 {
        sorted[i0]
    } else {
        (1.0 - t) * sorted[i0] + t * sorted[i1]
    }
}

fn posterior_summary(samples: &[f64]) -> (f64, f64, f64, f64) {
    let m = mean(samples);
    let sd = variance(samples).sqrt();
    let q025 = percentile(samples, 0.025);
    let q975 = percentile(samples, 0.975);
    (m, sd, q025, q975)
}

// ----------------------------
// Autocorrelation and ESS
// ----------------------------

fn autocovariance(samples: &[f64], lag: usize) -> f64 {
    let n = samples.len();
    assert!(lag < n, "Lag must be smaller than sample size.");

    let m = mean(samples);
    let mut sum = 0.0;
    for i in 0..(n - lag) {
        sum += (samples[i] - m) * (samples[i + lag] - m);
    }
    sum / (n - lag) as f64
}

fn autocorrelation(samples: &[f64], lag: usize) -> f64 {
    let gamma0 = autocovariance(samples, 0);
    if gamma0.abs() < 1.0e-15 {
        0.0
    } else {
        autocovariance(samples, lag) / gamma0
    }
}

fn autocorrelation_table(samples: &[f64], max_lag: usize) -> Vec<(usize, f64)> {
    let mut table = Vec::with_capacity(max_lag);
    for lag in 1..=max_lag {
        table.push((lag, autocorrelation(samples, lag)));
    }
    table
}

fn effective_sample_size(samples: &[f64], max_lag: usize) -> f64 {
    let n = samples.len() as f64;
    let mut rho_sum = 0.0;

    for lag in 1..=max_lag {
        let rho = autocorrelation(samples, lag);
        if rho <= 0.0 {
            break;
        }
        rho_sum += rho;
    }

    n / (1.0 + 2.0 * rho_sum)
}

// ----------------------------
// Multiple-Chain Comparison
// ----------------------------

#[derive(Clone, Debug)]
struct ChainSummary {
    name: String,
    acceptance_rate: f64,
    beta0_mean: f64,
    beta1_mean: f64,
    beta0_ess: f64,
    beta1_ess: f64,
}

fn summarize_chain(
    name: &str,
    chain: &[Params],
    acceptance_rate: f64,
    burn_in: usize,
    ess_max_lag: usize,
) -> ChainSummary {
    let posterior_chain = &chain[burn_in..];
    let (beta0_samples, beta1_samples) = extract_parameter_samples(posterior_chain);

    ChainSummary {
        name: name.to_string(),
        acceptance_rate,
        beta0_mean: mean(&beta0_samples),
        beta1_mean: mean(&beta1_samples),
        beta0_ess: effective_sample_size(&beta0_samples, ess_max_lag),
        beta1_ess: effective_sample_size(&beta1_samples, ess_max_lag),
    }
}

// ----------------------------
// Reporting Helpers
// ----------------------------

fn print_dataset(data: &Dataset) {
    println!("Synthetic Dataset for MCMC Diagnostics");
    println!("======================================");
    println!("Known sigma               = {:>.6}", data.sigma);
    println!("Number of observations    = {}", data.x.len());
    println!();

    println!("{:>4} {:>14} {:>14}", "i", "x_i", "y_i");
    for i in 0..data.x.len() {
        println!("{:>4} {:>14.6} {:>14.6}", i, data.x[i], data.y[i]);
    }
    println!();
}

fn print_trace_preview(name: &str, chain: &[Params], count: usize) {
    println!("{}", name);
    println!("{}", "=".repeat(name.len()));
    println!("{:>6} {:>16} {:>16}", "iter", "beta0", "beta1");

    for (k, theta) in chain.iter().enumerate().take(count.min(chain.len())) {
        println!("{:>6} {:>16.8} {:>16.8}", k, theta.beta0, theta.beta1);
    }
    println!();
}

fn print_autocorrelation_table(name: &str, samples: &[f64], max_lag: usize) {
    println!("{}", name);
    println!("{}", "=".repeat(name.len()));
    println!("{:>8} {:>18}", "lag", "autocorrelation");

    for (lag, rho) in autocorrelation_table(samples, max_lag) {
        println!("{:>8} {:>18.10}", lag, rho);
    }
    println!();
}

fn print_posterior_summary(name: &str, samples: &[f64], truth: f64, ess_max_lag: usize) {
    let (m, sd, q025, q975) = posterior_summary(samples);
    let ess = effective_sample_size(samples, ess_max_lag);

    println!("{}", name);
    println!("{}", "=".repeat(name.len()));
    println!("Posterior mean            = {:>.10}", m);
    println!("Posterior std. dev.       = {:>.10}", sd);
    println!("95% credible interval     = [{:>.10}, {:>.10}]", q025, q975);
    println!("Effective sample size     = {:>.10}", ess);
    println!("True value                = {:>.10}", truth);
    println!();
}

fn print_chain_comparison(summaries: &[ChainSummary]) {
    println!("Multiple-Chain Comparison");
    println!("=========================");
    println!(
        "{:>10} {:>16} {:>16} {:>16} {:>16} {:>16}",
        "chain", "acceptance", "beta0 mean", "beta1 mean", "ESS beta0", "ESS beta1"
    );

    for s in summaries {
        println!(
            "{:>10} {:>16.6} {:>16.8} {:>16.8} {:>16.4} {:>16.4}",
            s.name, s.acceptance_rate, s.beta0_mean, s.beta1_mean, s.beta0_ess, s.beta1_ess
        );
    }
    println!();
}

// ----------------------------
// Main Demonstration
// ----------------------------

fn main() {
    let beta0_true = 1.25;
    let beta1_true = -0.95;
    let sigma = 0.40;
    let n = 36;

    let mut rng_data = LcgRng::new(20260412);
    let data = generate_synthetic_data(beta0_true, beta1_true, sigma, n, &mut rng_data);
    print_dataset(&data);

    let prior = Prior {
        mean_beta0: 0.0,
        mean_beta1: 0.0,
        sd_beta0: 5.0,
        sd_beta1: 5.0,
    };

    let proposal = Proposal {
        step_beta0: 0.10,
        step_beta1: 0.09,
    };

    let num_samples = 30_000usize;
    let burn_in = 5_000usize;
    let acf_max_lag = 15usize;
    let ess_max_lag = 100usize;

    let initial_points = [
        ("Chain A", Params { beta0: 0.0, beta1: 0.0 }, 11111u64),
        ("Chain B", Params { beta0: 3.0, beta1: -3.0 }, 22222u64),
        ("Chain C", Params { beta0: -2.0, beta1: 2.0 }, 33333u64),
    ];

    let mut results = Vec::new();

    for (name, initial, seed) in initial_points {
        let mut rng = LcgRng::new(seed);
        let result = metropolis_hastings(initial, num_samples, proposal, &data, prior, &mut rng);

        println!("{}", name);
        println!("{}", "=".repeat(name.len()));
        println!("Initial state            = ({:>.6}, {:>.6})", initial.beta0, initial.beta1);
        println!("Total samples            = {}", num_samples);
        println!("Burn-in                  = {}", burn_in);
        println!("Accepted proposals       = {}", result.accepted);
        println!("Acceptance rate          = {:>.6}", result.acceptance_rate);
        println!();

        print_trace_preview(&format!("{} Trace Preview", name), &result.chain, 18);
        results.push((name.to_string(), result));
    }

    let chain_a = &results[0].1.chain[burn_in..];
    let (beta0_a, beta1_a) = extract_parameter_samples(chain_a);

    print_autocorrelation_table("Autocorrelation for beta0 (Chain A)", &beta0_a, acf_max_lag);
    print_autocorrelation_table("Autocorrelation for beta1 (Chain A)", &beta1_a, acf_max_lag);

    print_posterior_summary(
        "Posterior Summary for beta0 (Chain A)",
        &beta0_a,
        beta0_true,
        ess_max_lag,
    );
    print_posterior_summary(
        "Posterior Summary for beta1 (Chain A)",
        &beta1_a,
        beta1_true,
        ess_max_lag,
    );

    let summaries = results
        .iter()
        .map(|(name, result)| {
            summarize_chain(
                name,
                &result.chain,
                result.acceptance_rate,
                burn_in,
                ess_max_lag,
            )
        })
        .collect::<Vec<_>>();

    print_chain_comparison(&summaries);

    println!("Interpretation");
    println!("==============");
    println!("This program distinguishes between total sample count and effective sample size.");
    println!("Because consecutive MCMC states are correlated, a chain of length K contains");
    println!("less information than K independent samples. The effective sample size estimate");
    println!("implements Equation (15.9.13) by combining autocorrelations across lags.");
    println!();
    println!("The trace previews illustrate the role of burn-in. Chains started from dispersed");
    println!("initial values approach the same posterior region, after which their summaries");
    println!("become comparable. This agreement across multiple chains provides evidence of");
    println!("convergence, while the autocorrelation tables quantify mixing speed.");
    println!();
    println!("If the proposal step sizes were chosen too small, acceptance would be high but");
    println!("autocorrelation would remain strong. If the steps were too large, acceptance would");
    println!("collapse and the chain would stagnate. Good MCMC performance therefore depends on");
    println!("balancing acceptance and exploration rather than simply increasing the number of");
    println!("iterations.");
}
```

Program 15.9.2 demonstrates that the effectiveness of MCMC methods depends not only on the correctness of the underlying algorithm but also on the efficiency with which the Markov chain explores the target distribution. The diagnostics implemented in this program reveal how autocorrelation reduces the effective sample size and highlight the importance of assessing convergence before drawing statistical conclusions.

The results illustrate that a large number of generated samples does not necessarily imply a large amount of useful information. Instead, the effective sample size provides a more meaningful measure of the quality of the simulation, reflecting the degree of dependence between successive samples. The use of multiple chains further strengthens the analysis by providing a practical method for verifying convergence through consistency across independent runs.

The modular design of the implementation allows these diagnostic tools to be applied to a wide range of MCMC algorithms and models. This provides a foundation for more advanced methods that aim to improve efficiency, such as adaptive proposals, gradient-based samplers, and parallel chain strategies. In this way, the program bridges the gap between theoretical convergence guarantees and practical computational performance, emphasizing that careful diagnostic analysis is essential for reliable Bayesian inference.

## 15.9.5. Advanced MCMC Methods

To improve efficiency, several advanced MCMC methods have been developed that address the limitations of basic algorithms such as random-walk Metropolis. These methods aim to reduce autocorrelation, improve mixing, and enable efficient exploration of high-dimensional or strongly correlated parameter spaces. A common theme among them is the use of additional structure, such as gradient information or adaptive mechanisms, to guide the sampling process more effectively.

### Hamiltonian Monte Carlo (HMC)

Hamiltonian Monte Carlo is one of the most powerful methods for high-dimensional sampling. It uses gradient information from the log-posterior to construct proposals that move across the parameter space in a structured and efficient manner. Instead of taking small random steps, HMC simulates a dynamical system that allows for long-distance moves while maintaining a high acceptance probability.

The method introduces an auxiliary momentum variable $p$ and defines a Hamiltonian function:

$$H(\theta, p) = -\log p(\theta \mid D) + \frac{1}{2} p^\top M^{-1} p \tag{15.9.14}$$

The first term represents the potential energy derived from the posterior distribution, while the second term represents kinetic energy associated with the momentum. By simulating Hamiltonian dynamics, the method generates proposals that follow trajectories of approximately constant energy, allowing the chain to traverse the parameter space efficiently.

These trajectories are computed using numerical integration, typically with symplectic methods that preserve the structure of the dynamics. After a trajectory is simulated, a Metropolis acceptance step ensures that the correct stationary distribution is maintained. The use of gradients enables HMC to avoid the random-walk behavior of simpler methods, leading to significantly improved mixing in many applications.

### Langevin Methods

Langevin-based methods also incorporate gradient information, but in a stochastic framework. The proposal mechanism combines deterministic movement in the direction of increasing posterior density with random perturbations:

$$\theta' = \theta + \frac{\epsilon^2}{2} \nabla \log p(\theta \mid D) + \epsilon \eta \tag{15.9.15}$$

Here, $\eta$ is a random vector, typically drawn from a Gaussian distribution, and $\epsilon$ controls the step size. The gradient term encourages movement toward regions of higher probability, while the stochastic term ensures exploration of the parameter space.

Langevin methods can be viewed as a bridge between optimization and sampling, since they use gradient information to guide proposals while maintaining the stochasticity required for correct sampling. They often achieve better efficiency than random-walk methods, especially in moderately high-dimensional problems.

### Adaptive MCMC

Adaptive MCMC methods aim to improve efficiency by adjusting the proposal distribution during sampling. For example, the covariance of a Gaussian proposal may be updated based on the empirical covariance of previously generated samples. This allows the algorithm to learn the structure of the target distribution and tailor its proposals accordingly.

Such adaptation can significantly improve performance, particularly when the scale or correlation structure of the posterior is not known in advance. However, care must be taken to ensure that the adaptation does not violate the theoretical conditions required for convergence. Well-designed adaptive schemes balance the need for flexibility with the requirement of maintaining correct stationary behavior.

These advanced methods significantly improve performance in high-dimensional problems, where naive MCMC algorithms often struggle. By incorporating gradient information or adapting to the geometry of the posterior, they reduce autocorrelation, increase effective sample size, and make sampling-based inference practical for complex models.

## 15.9.6. Computational Considerations and Large-Scale Problems

In modern applications, the practical use of MCMC is shaped by computational constraints as much as by statistical considerations. While the underlying algorithms are conceptually straightforward, their implementation can become challenging when models are complex and datasets are large.

Each MCMC step may require evaluating a complex model. In many scientific and engineering applications, the likelihood function involves solving a numerical model, such as a system of differential equations or a simulation-based forward operator. As a result, even a single evaluation of $p(D \mid \theta)$ can be computationally expensive, and thousands or millions of such evaluations may be required to obtain a sufficient number of samples.

Consequently, the computational cost is often dominated by likelihood evaluation. The efficiency of the overall MCMC procedure depends heavily on how quickly and accurately the model can be evaluated. Improvements in model evaluation, including algorithmic optimization and numerical approximation, can therefore have a direct and substantial impact on sampling performance.

Parallelization is essential for scalability. Many components of MCMC are naturally parallelizable, particularly when multiple independent chains are run simultaneously. This allows one to distribute the computational workload across multiple processors or nodes, reducing wall-clock time and improving robustness through cross-chain diagnostics.

Several challenges arise in large-scale settings. High-dimensional parameter spaces make exploration difficult, as the volume of the space increases rapidly and naive sampling methods become inefficient. Expensive forward models, such as those involving PDE solvers, significantly increase the cost per iteration and limit the number of samples that can be generated. Correlated parameters further complicate sampling, as the chain may move slowly along narrow, curved regions of high probability.

To address these challenges, recent developments have introduced a range of computational strategies. Surrogate models are used to approximate the likelihood function, replacing expensive evaluations with faster approximations that can be refined as needed. Parallel chains and distributed sampling frameworks allow large-scale computations to be spread across modern computing architectures, improving both speed and reliability. Subsampling methods reduce the cost of likelihood evaluation by using only a portion of the data at each step, while maintaining approximate correctness through careful statistical control.

These approaches aim to make MCMC feasible for large-scale scientific problems, where direct application of standard methods would be prohibitively expensive. By combining statistical insight with computational innovation, modern MCMC methods extend the applicability of Bayesian inference to increasingly complex and high-dimensional settings.

### Rust Implementation

Following the discussion in Section 15.9.5 on advanced Markov chain Monte Carlo methods and the role of gradient information in improving sampling efficiency, Program 15.9.3 provides a practical implementation of the Metropolis-adjusted Langevin algorithm (MALA) alongside the classical random-walk Metropolis method. While Section 15.9.4 emphasized the importance of efficiency, mixing, and convergence diagnostics, this program demonstrates how these properties can be improved by incorporating gradient information into the proposal mechanism. By comparing the two samplers within a unified computational framework and extending the implementation to parallel chains, the program illustrates how modern MCMC methods aim to reduce random-walk behavior and enhance exploration of the posterior distribution in accordance with Equation (15.9.15).

At the core of the implementation is the construction of two distinct sampling mechanisms within a common Metropolis–Hastings framework. The function `run_chain` implements the general acceptance rule described in Equation (15.9.10), with the posterior ratio formulated as in Equation (15.9.11). For the random-walk Metropolis method, the proposal is symmetric, and the proposal density ratio cancels. In contrast, the MALA implementation introduces an asymmetric proposal that incorporates gradient information through the drift term defined in Equation (15.9.15). This asymmetry requires explicit computation of both forward and reverse transition densities, which are evaluated using the function `log_gaussian_transition`.

The gradient of the log-posterior, required for the Langevin proposal, is computed by combining the outputs of `grad_log_likelihood` and `grad_log_prior` within the function `grad_log_posterior`. These components correspond to the derivatives of the likelihood and prior contributions to the posterior density. The function `mala_mean` constructs the deterministic drift component of the proposal, while `propose_mala` adds a stochastic perturbation to generate candidate states. This structure reflects the balance between deterministic movement toward high-density regions and stochastic exploration of the parameter space.

The program also incorporates parallel execution of multiple chains using Rust threads. Each chain is initialized from a distinct starting point and evolves independently under either the random-walk or Langevin proposal mechanism. This design reflects the multiple-chain diagnostic approach discussed in Section 15.9.4, allowing convergence to be assessed by comparing posterior summaries across chains. The aggregation of results is handled through the `summarize_chain` function, which computes posterior means, standard deviations, and effective sample sizes based on the autocorrelation structure defined in Equation (15.9.13).

To evaluate sampling efficiency, the implementation includes autocorrelation analysis and effective sample size estimation. The functions `autocorrelation` and `effective_sample_size` quantify the dependence between successive samples and translate this dependence into an equivalent number of independent observations. These diagnostics provide a direct measure of how effectively each sampler explores the posterior distribution. The comparison between random-walk Metropolis and MALA highlights how the choice of proposal mechanism influences both autocorrelation and sampling efficiency.

The `main` function integrates all components into a cohesive workflow. It generates synthetic data, initializes model parameters and priors, and executes both sampling algorithms across multiple chains. The results are then analyzed through trace previews, autocorrelation comparisons, posterior summaries, and a consolidated summary table. This comprehensive structure demonstrates how advanced MCMC methods can be evaluated in practice, linking theoretical properties of the algorithms to observable computational behavior.

```rust
// Program 15.9.3: Gradient-Guided Langevin MCMC with Parallel Chains for Bayesian Linear Regression
//
// Problem Statement:
// Implement an advanced MCMC method for Bayesian linear regression by comparing
// a classical random-walk Metropolis sampler with a gradient-guided Langevin
// Metropolis-adjusted Langevin algorithm (MALA). The model is
//
//     y_i = beta_0 + beta_1 x_i + epsilon_i,   epsilon_i ~ N(0, sigma^2),
//
// with known noise standard deviation sigma. The target distribution is the
// posterior density of
//
//     theta = (beta_0, beta_1)^T.
//
// The program demonstrates several ideas from Sections 15.9.5 and 15.9.6:
//
// 1. Advanced MCMC through the Langevin proposal
//
//        theta' = theta + (epsilon^2 / 2) * grad log p(theta | D) + epsilon * eta
//
//    which uses gradient information to guide exploration.
//
// 2. Comparison with random-walk Metropolis in terms of acceptance rate,
//    autocorrelation, and effective sample size.
//
// 3. Parallel independent chains using Rust threads, illustrating how MCMC
//    workloads can be distributed across multiple chains.
//
// The implementation is fully self-contained and uses only the Rust standard
// library so that cargo run works out of the box.

use std::f64::consts::PI;
use std::thread;

type Vector = Vec<f64>;

#[derive(Clone, Debug)]
struct LcgRng {
    state: u64,
    has_spare_normal: bool,
    spare_normal: f64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            has_spare_normal: false,
            spare_normal: 0.0,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        let u = self.next_u64() >> 11;
        (u as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn standard_normal(&mut self) -> f64 {
        if self.has_spare_normal {
            self.has_spare_normal = false;
            return self.spare_normal;
        }

        let mut u1 = self.next_f64();
        while u1 <= 1.0e-15 {
            u1 = self.next_f64();
        }
        let u2 = self.next_f64();

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;

        let z0 = r * theta.cos();
        let z1 = r * theta.sin();

        self.spare_normal = z1;
        self.has_spare_normal = true;

        z0
    }
}

#[derive(Clone, Copy, Debug)]
struct Params {
    beta0: f64,
    beta1: f64,
}

#[derive(Clone, Copy, Debug)]
struct Prior {
    mean_beta0: f64,
    mean_beta1: f64,
    sd_beta0: f64,
    sd_beta1: f64,
}

#[derive(Clone, Debug)]
struct Dataset {
    x: Vector,
    y: Vector,
    sigma: f64,
}

fn log_normal_pdf(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / sd;
    -0.5 * z * z - sd.ln() - 0.5 * (2.0 * PI).ln()
}

fn log_prior(theta: Params, prior: Prior) -> f64 {
    log_normal_pdf(theta.beta0, prior.mean_beta0, prior.sd_beta0)
        + log_normal_pdf(theta.beta1, prior.mean_beta1, prior.sd_beta1)
}

fn grad_log_prior(theta: Params, prior: Prior) -> Params {
    Params {
        beta0: -(theta.beta0 - prior.mean_beta0) / (prior.sd_beta0 * prior.sd_beta0),
        beta1: -(theta.beta1 - prior.mean_beta1) / (prior.sd_beta1 * prior.sd_beta1),
    }
}

fn log_likelihood(theta: Params, data: &Dataset) -> f64 {
    let sigma2 = data.sigma * data.sigma;
    let log_norm = -0.5 * (2.0 * PI * sigma2).ln();

    let mut sum = 0.0;
    for i in 0..data.x.len() {
        let mu = theta.beta0 + theta.beta1 * data.x[i];
        let r = data.y[i] - mu;
        sum += log_norm - 0.5 * r * r / sigma2;
    }
    sum
}

fn grad_log_likelihood(theta: Params, data: &Dataset) -> Params {
    let sigma2 = data.sigma * data.sigma;

    let mut g0 = 0.0;
    let mut g1 = 0.0;

    for i in 0..data.x.len() {
        let mu = theta.beta0 + theta.beta1 * data.x[i];
        let r = data.y[i] - mu;
        g0 += r / sigma2;
        g1 += data.x[i] * r / sigma2;
    }

    Params { beta0: g0, beta1: g1 }
}

fn log_posterior(theta: Params, data: &Dataset, prior: Prior) -> f64 {
    log_likelihood(theta, data) + log_prior(theta, prior)
}

fn grad_log_posterior(theta: Params, data: &Dataset, prior: Prior) -> Params {
    let gl = grad_log_likelihood(theta, data);
    let gp = grad_log_prior(theta, prior);

    Params {
        beta0: gl.beta0 + gp.beta0,
        beta1: gl.beta1 + gp.beta1,
    }
}

fn generate_synthetic_data(
    beta0_true: f64,
    beta1_true: f64,
    sigma: f64,
    n: usize,
    rng: &mut LcgRng,
) -> Dataset {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    let x_min = -2.5;
    let x_max = 2.5;

    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        let xi = x_min + (x_max - x_min) * t;
        let yi = beta0_true + beta1_true * xi + sigma * rng.standard_normal();

        x.push(xi);
        y.push(yi);
    }

    Dataset { x, y, sigma }
}

#[derive(Clone, Copy, Debug)]
enum SamplerKind {
    RandomWalk { step_beta0: f64, step_beta1: f64 },
    Mala { epsilon: f64 },
}

impl SamplerKind {
    fn name(&self) -> &'static str {
        match self {
            SamplerKind::RandomWalk { .. } => "Random-Walk Metropolis",
            SamplerKind::Mala { .. } => "Langevin MALA",
        }
    }
}

#[derive(Clone, Debug)]
struct ChainResult {
    sampler_name: String,
    chain_name: String,
    chain: Vec<Params>,
    accepted: usize,
    acceptance_rate: f64,
}

fn propose_random_walk(
    current: Params,
    step_beta0: f64,
    step_beta1: f64,
    rng: &mut LcgRng,
) -> Params {
    Params {
        beta0: current.beta0 + step_beta0 * rng.standard_normal(),
        beta1: current.beta1 + step_beta1 * rng.standard_normal(),
    }
}

fn mala_mean(theta: Params, epsilon: f64, data: &Dataset, prior: Prior) -> Params {
    let g = grad_log_posterior(theta, data, prior);
    let scale = 0.5 * epsilon * epsilon;

    Params {
        beta0: theta.beta0 + scale * g.beta0,
        beta1: theta.beta1 + scale * g.beta1,
    }
}

fn propose_mala(
    current: Params,
    epsilon: f64,
    data: &Dataset,
    prior: Prior,
    rng: &mut LcgRng,
) -> Params {
    let m = mala_mean(current, epsilon, data, prior);
    Params {
        beta0: m.beta0 + epsilon * rng.standard_normal(),
        beta1: m.beta1 + epsilon * rng.standard_normal(),
    }
}

fn log_gaussian_transition(x: Params, mean: Params, epsilon: f64) -> f64 {
    let var = epsilon * epsilon;
    let dx0 = x.beta0 - mean.beta0;
    let dx1 = x.beta1 - mean.beta1;
    let dim = 2.0;

    -0.5 * (dx0 * dx0 + dx1 * dx1) / var - dim * epsilon.ln() - 0.5 * dim * (2.0 * PI).ln()
}

fn run_chain(
    sampler: SamplerKind,
    initial: Params,
    num_samples: usize,
    data: &Dataset,
    prior: Prior,
    seed: u64,
    chain_name: &str,
) -> ChainResult {
    let mut rng = LcgRng::new(seed);
    let mut chain = Vec::with_capacity(num_samples);
    let mut current = initial;
    let mut current_log_post = log_posterior(current, data, prior);
    let mut accepted = 0usize;

    for _ in 0..num_samples {
        let (candidate, log_q_forward, log_q_reverse) = match sampler {
            SamplerKind::RandomWalk {
                step_beta0,
                step_beta1,
            } => {
                let cand = propose_random_walk(current, step_beta0, step_beta1, &mut rng);
                (cand, 0.0, 0.0)
            }
            SamplerKind::Mala { epsilon } => {
                let cand = propose_mala(current, epsilon, data, prior, &mut rng);
                let mean_forward = mala_mean(current, epsilon, data, prior);
                let mean_reverse = mala_mean(cand, epsilon, data, prior);

                let log_q_f = log_gaussian_transition(cand, mean_forward, epsilon);
                let log_q_r = log_gaussian_transition(current, mean_reverse, epsilon);

                (cand, log_q_f, log_q_r)
            }
        };

        let candidate_log_post = log_posterior(candidate, data, prior);
        let log_alpha = candidate_log_post + log_q_reverse - current_log_post - log_q_forward;

        let accept = if log_alpha >= 0.0 {
            true
        } else {
            rng.next_f64().ln() < log_alpha
        };

        if accept {
            current = candidate;
            current_log_post = candidate_log_post;
            accepted += 1;
        }

        chain.push(current);
    }

    ChainResult {
        sampler_name: sampler.name().to_string(),
        chain_name: chain_name.to_string(),
        chain,
        accepted,
        acceptance_rate: accepted as f64 / num_samples as f64,
    }
}

fn extract_parameter_samples(chain: &[Params]) -> (Vector, Vector) {
    let mut beta0 = Vec::with_capacity(chain.len());
    let mut beta1 = Vec::with_capacity(chain.len());

    for &theta in chain {
        beta0.push(theta.beta0);
        beta1.push(theta.beta1);
    }

    (beta0, beta1)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    let m = mean(values);
    let mut sum = 0.0;
    for &v in values {
        let d = v - m;
        sum += d * d;
    }
    sum / (values.len() as f64 - 1.0)
}

fn percentile(values: &[f64], p: f64) -> f64 {
    assert!(!values.is_empty(), "Percentile requires nonempty data.");
    assert!(
        (0.0..=1.0).contains(&p),
        "Percentile level must lie in [0,1]."
    );

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if sorted.len() == 1 {
        return sorted[0];
    }

    let pos = p * (sorted.len() as f64 - 1.0);
    let i0 = pos.floor() as usize;
    let i1 = pos.ceil() as usize;
    let t = pos - i0 as f64;

    if i0 == i1 {
        sorted[i0]
    } else {
        (1.0 - t) * sorted[i0] + t * sorted[i1]
    }
}

fn autocovariance(samples: &[f64], lag: usize) -> f64 {
    let n = samples.len();
    let m = mean(samples);
    let mut sum = 0.0;

    for i in 0..(n - lag) {
        sum += (samples[i] - m) * (samples[i + lag] - m);
    }

    sum / (n - lag) as f64
}

fn autocorrelation(samples: &[f64], lag: usize) -> f64 {
    let gamma0 = autocovariance(samples, 0);
    if gamma0.abs() < 1.0e-15 {
        0.0
    } else {
        autocovariance(samples, lag) / gamma0
    }
}

fn effective_sample_size(samples: &[f64], max_lag: usize) -> f64 {
    let gamma0 = autocovariance(samples, 0);
    if gamma0.abs() < 1.0e-15 {
        return 0.0;
    }

    let n = samples.len() as f64;
    let mut rho_sum = 0.0;

    for lag in 1..=max_lag {
        let rho = autocorrelation(samples, lag);
        if rho <= 0.0 {
            break;
        }
        rho_sum += rho;
    }

    n / (1.0 + 2.0 * rho_sum)
}

#[derive(Clone, Debug)]
struct Summary {
    sampler_name: String,
    chain_name: String,
    beta0_mean: f64,
    beta1_mean: f64,
    beta0_sd: f64,
    beta1_sd: f64,
    beta0_ess: f64,
    beta1_ess: f64,
    acceptance_rate: f64,
}

fn summarize_chain(result: &ChainResult, burn_in: usize, ess_max_lag: usize) -> Summary {
    let posterior_chain = &result.chain[burn_in..];
    let (beta0_samples, beta1_samples) = extract_parameter_samples(posterior_chain);

    Summary {
        sampler_name: result.sampler_name.clone(),
        chain_name: result.chain_name.clone(),
        beta0_mean: mean(&beta0_samples),
        beta1_mean: mean(&beta1_samples),
        beta0_sd: variance(&beta0_samples).sqrt(),
        beta1_sd: variance(&beta1_samples).sqrt(),
        beta0_ess: effective_sample_size(&beta0_samples, ess_max_lag),
        beta1_ess: effective_sample_size(&beta1_samples, ess_max_lag),
        acceptance_rate: result.acceptance_rate,
    }
}

fn print_dataset(data: &Dataset) {
    println!("Synthetic Dataset for Advanced MCMC");
    println!("===================================");
    println!("Known sigma               = {:>.6}", data.sigma);
    println!("Number of observations    = {}", data.x.len());
    println!();

    println!("{:>4} {:>14} {:>14}", "i", "x_i", "y_i");
    for i in 0..data.x.len() {
        println!("{:>4} {:>14.6} {:>14.6}", i, data.x[i], data.y[i]);
    }
    println!();
}

fn print_trace_preview(title: &str, chain: &[Params], count: usize) {
    println!("{}", title);
    println!("{}", "=".repeat(title.len()));
    println!("{:>6} {:>16} {:>16}", "iter", "beta0", "beta1");

    for (k, theta) in chain.iter().enumerate().take(count.min(chain.len())) {
        println!("{:>6} {:>16.8} {:>16.8}", k, theta.beta0, theta.beta1);
    }
    println!();
}

fn print_autocorrelation_comparison(beta0_rw: &[f64], beta0_mala: &[f64], max_lag: usize) {
    println!("Autocorrelation Comparison for beta0");
    println!("====================================");
    println!("{:>8} {:>18} {:>18}", "lag", "RW-MH", "MALA");

    for lag in 1..=max_lag {
        let rho_rw = autocorrelation(beta0_rw, lag);
        let rho_mala = autocorrelation(beta0_mala, lag);
        println!("{:>8} {:>18.10} {:>18.10}", lag, rho_rw, rho_mala);
    }
    println!();
}

fn print_posterior_intervals(samples: &[f64], title: &str, truth: f64, ess_max_lag: usize) {
    let var = autocovariance(samples, 0);
    if var.abs() < 1.0e-15 {
        println!("{}", title);
        println!("{}", "=".repeat(title.len()));
        println!("Posterior mean            = {:>.10}", mean(samples));
        println!("Posterior std. dev.       = {:>.10}", 0.0);
        println!("95% credible interval     = [{:>.10}, {:>.10}]", mean(samples), mean(samples));
        println!("Effective sample size     = {:>.10}", 0.0);
        println!("True value                = {:>.10}", truth);
        println!("Warning                   = Chain did not move; diagnostics are degenerate.");
        println!();
        return;
    }

    let m = mean(samples);
    let sd = variance(samples).sqrt();
    let q025 = percentile(samples, 0.025);
    let q975 = percentile(samples, 0.975);
    let ess = effective_sample_size(samples, ess_max_lag);

    println!("{}", title);
    println!("{}", "=".repeat(title.len()));
    println!("Posterior mean            = {:>.10}", m);
    println!("Posterior std. dev.       = {:>.10}", sd);
    println!("95% credible interval     = [{:>.10}, {:>.10}]", q025, q975);
    println!("Effective sample size     = {:>.10}", ess);
    println!("True value                = {:>.10}", truth);
    println!();
}

fn print_summary_table(summaries: &[Summary]) {
    println!("Parallel Chain Summary");
    println!("======================");
    println!(
        "{:>18} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "sampler",
        "chain",
        "acceptance",
        "beta0 mean",
        "beta1 mean",
        "beta0 sd",
        "beta1 sd",
        "ESS beta0",
        "ESS beta1"
    );

    for s in summaries {
        println!(
            "{:>18} {:>10} {:>12.6} {:>12.8} {:>12.8} {:>12.8} {:>12.8} {:>12.2} {:>12.2}",
            s.sampler_name,
            s.chain_name,
            s.acceptance_rate,
            s.beta0_mean,
            s.beta1_mean,
            s.beta0_sd,
            s.beta1_sd,
            s.beta0_ess,
            s.beta1_ess
        );
    }
    println!();
}

fn main() {
    let beta0_true = 1.10;
    let beta1_true = -0.85;
    let sigma = 0.35;
    let n = 42;

    let mut rng_data = LcgRng::new(20260412);
    let data = generate_synthetic_data(beta0_true, beta1_true, sigma, n, &mut rng_data);
    print_dataset(&data);

    let prior = Prior {
        mean_beta0: 0.0,
        mean_beta1: 0.0,
        sd_beta0: 5.0,
        sd_beta1: 5.0,
    };

    let num_samples = 25_000usize;
    let burn_in = 5_000usize;
    let ess_max_lag = 100usize;
    let acf_max_lag = 12usize;

    let rw_sampler = SamplerKind::RandomWalk {
        step_beta0: 0.10,
        step_beta1: 0.09,
    };

    // Corrected epsilon: much smaller than before to avoid catastrophic rejection.
    let mala_sampler = SamplerKind::Mala { epsilon: 0.02 };

    let initial_points_rw = [
        ("Chain A", Params { beta0: 0.0, beta1: 0.0 }, 11111u64),
        ("Chain B", Params { beta0: 2.5, beta1: -2.0 }, 22222u64),
        ("Chain C", Params { beta0: -2.0, beta1: 1.5 }, 33333u64),
    ];

    let mut rw_handles = Vec::new();
    for (chain_name, initial, seed) in initial_points_rw {
        let data_clone = data.clone();
        let prior_copy = prior;
        let sampler_copy = rw_sampler;
        let chain_name_owned = chain_name.to_string();

        rw_handles.push(thread::spawn(move || {
            run_chain(
                sampler_copy,
                initial,
                num_samples,
                &data_clone,
                prior_copy,
                seed,
                &chain_name_owned,
            )
        }));
    }

    let mut rw_results = Vec::new();
    for handle in rw_handles {
        rw_results.push(handle.join().expect("Random-walk thread failed."));
    }

    let initial_points_mala = [
        ("Chain A", Params { beta0: 0.0, beta1: 0.0 }, 44444u64),
        ("Chain B", Params { beta0: 2.5, beta1: -2.0 }, 55555u64),
        ("Chain C", Params { beta0: -2.0, beta1: 1.5 }, 66666u64),
    ];

    let mut mala_handles = Vec::new();
    for (chain_name, initial, seed) in initial_points_mala {
        let data_clone = data.clone();
        let prior_copy = prior;
        let sampler_copy = mala_sampler;
        let chain_name_owned = chain_name.to_string();

        mala_handles.push(thread::spawn(move || {
            run_chain(
                sampler_copy,
                initial,
                num_samples,
                &data_clone,
                prior_copy,
                seed,
                &chain_name_owned,
            )
        }));
    }

    let mut mala_results = Vec::new();
    for handle in mala_handles {
        mala_results.push(handle.join().expect("MALA thread failed."));
    }

    let rw_chain_a = rw_results
        .iter()
        .find(|r| r.chain_name == "Chain A")
        .expect("Missing RW Chain A.");
    let mala_chain_a = mala_results
        .iter()
        .find(|r| r.chain_name == "Chain A")
        .expect("Missing MALA Chain A.");

    println!("Sampler Configuration");
    println!("=====================");
    println!("Total samples per chain     = {}", num_samples);
    println!("Burn-in                     = {}", burn_in);
    println!(
        "Random-walk proposal        = step_beta0 = 0.10, step_beta1 = 0.09"
    );
    println!("Langevin proposal           = epsilon = 0.02");
    println!();

    println!("Representative Acceptance Rates");
    println!("===============================");
    println!(
        "Random-Walk Metropolis (Chain A)  = {:>.6} (accepted = {})",
        rw_chain_a.acceptance_rate, rw_chain_a.accepted
    );
    println!(
        "Langevin MALA (Chain A)           = {:>.6} (accepted = {})",
        mala_chain_a.acceptance_rate, mala_chain_a.accepted
    );
    println!();

    print_trace_preview(
        "Random-Walk Metropolis Trace Preview (Chain A)",
        &rw_chain_a.chain,
        16,
    );
    print_trace_preview("Langevin MALA Trace Preview (Chain A)", &mala_chain_a.chain, 16);

    let rw_post = &rw_chain_a.chain[burn_in..];
    let mala_post = &mala_chain_a.chain[burn_in..];

    let (rw_beta0, rw_beta1) = extract_parameter_samples(rw_post);
    let (mala_beta0, mala_beta1) = extract_parameter_samples(mala_post);

    print_autocorrelation_comparison(&rw_beta0, &mala_beta0, acf_max_lag);

    print_posterior_intervals(
        &rw_beta0,
        "Posterior Summary for beta0 Using Random-Walk Metropolis",
        beta0_true,
        ess_max_lag,
    );
    print_posterior_intervals(
        &mala_beta0,
        "Posterior Summary for beta0 Using Langevin MALA",
        beta0_true,
        ess_max_lag,
    );
    print_posterior_intervals(
        &rw_beta1,
        "Posterior Summary for beta1 Using Random-Walk Metropolis",
        beta1_true,
        ess_max_lag,
    );
    print_posterior_intervals(
        &mala_beta1,
        "Posterior Summary for beta1 Using Langevin MALA",
        beta1_true,
        ess_max_lag,
    );

    let mut summaries = Vec::new();
    for result in &rw_results {
        summaries.push(summarize_chain(result, burn_in, ess_max_lag));
    }
    for result in &mala_results {
        summaries.push(summarize_chain(result, burn_in, ess_max_lag));
    }

    print_summary_table(&summaries);

    println!("Interpretation");
    println!("==============");
    println!("The Langevin sampler uses the gradient of the log-posterior to bias proposed");
    println!("moves toward regions of higher posterior density, in accordance with Equation");
    println!("(15.9.15). This reduces the random-walk behavior that often limits the efficiency");
    println!("of basic Metropolis methods.");
    println!();
    println!("The comparison of autocorrelation and effective sample size shows how advanced");
    println!("MCMC methods can produce more informative samples from chains of similar length.");
    println!("Even when both samplers target the same posterior distribution, the gradient-guided");
    println!("method often yields lower autocorrelation and higher effective sample size when");
    println!("the step size is tuned appropriately.");
    println!();
    println!("The use of multiple parallel chains illustrates an important computational strategy");
    println!("for large-scale Bayesian inference. Independent chains can be run concurrently,");
    println!("improving wall-clock efficiency and supporting cross-chain convergence assessment.");
}
```

Program 15.9.3 demonstrates how gradient-based MCMC methods extend the classical Metropolis–Hastings framework to improve sampling efficiency in complex posterior distributions. By incorporating information about the local geometry of the target distribution, the Langevin proposal reduces purely random exploration and can, when properly tuned, produce more informative samples with lower autocorrelation.

The comparison with random-walk Metropolis highlights that algorithmic sophistication alone does not guarantee improved performance. The effectiveness of gradient-based methods depends critically on the choice of tuning parameters, particularly the step size. As seen in the diagnostics, overly conservative choices can lead to high acceptance rates but poor mixing, while overly aggressive choices can result in frequent rejections. This reinforces the central theme of Sections 15.9.4 and 15.9.5 that efficiency in MCMC arises from a careful balance between exploration and acceptance.

The modular and extensible structure of the implementation provides a foundation for further developments in modern MCMC. More advanced methods, such as Hamiltonian Monte Carlo, adaptive Langevin algorithms, and scalable parallel implementations, build upon the same principles demonstrated here. In this way, the program serves as a bridge between classical sampling methods and contemporary approaches to large-scale Bayesian computation.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/ZSn9Ky7r239j1pyN1piq.6","tags":[]}

# 15.10. Gaussian Process Regression

Gaussian process regression (GPR) represents a shift in the modeling philosophy developed throughout this chapter. Instead of assuming a finite-dimensional parameter vector $\theta$, the model is defined by placing a probability distribution directly on an unknown function $f(\cdot)$. In this formulation, the object of inference is no longer a set of coefficients, but an entire function, with uncertainty quantified at every input location. This perspective provides a natural and flexible way to represent complex relationships without committing to a specific parametric form.

This shift is particularly significant because it replaces parameter estimation with function-space inference. Rather than selecting a fixed basis or functional form, GPR defines a distribution over functions characterized by a mean function and a covariance kernel. The covariance structure encodes assumptions about smoothness, correlation, and variability, allowing the model to adapt its complexity to the data. As a result, the model can represent a wide range of behaviors while maintaining a principled probabilistic interpretation.

This perspective is especially natural in settings where the functional relationship is complex, expensive to evaluate, or not well described by a low-dimensional parametric family. In such cases, traditional regression models may either lack sufficient flexibility or require careful selection of basis functions, whereas GPR provides a systematic way to model uncertainty and structure simultaneously.

This approach is especially useful when the model serves as a surrogate for computationally expensive simulations, such as finite element or PDE-based models. By learning an approximation to the underlying function, GPR can provide fast predictions while quantifying uncertainty in regions where data are sparse. It is also well suited to situations where the dataset is moderate in size and calibrated uncertainty is required, since the probabilistic formulation directly yields predictive distributions rather than point estimates. In addition, when smoothness is expected but a fixed parametric structure is not appropriate, the kernel-based representation allows the model to capture smooth variation without imposing rigid functional forms.

A central practical issue emphasized in modern work is that exact Gaussian process inference scales cubically with the number of data points. This computational cost arises from the need to form and factorize dense covariance matrices, which becomes prohibitive as the dataset grows. Consequently, significant research effort has been devoted to developing structured and approximate methods that reduce computational complexity while preserving accuracy (Hoffmann and Onnela, 2025). These developments are essential for extending the applicability of Gaussian process methods to large-scale problems.

## 15.10.1. Gaussian Process Prior and Finite-Dimensional Marginals

A Gaussian process prior is written as:

$$f(\cdot) \sim \mathcal{GP}(m(\cdot), k(\cdot,\cdot)) \tag{15.10.1}$$

which specifies a probability distribution over functions. This notation indicates that any collection of function values, evaluated at a finite set of input points, follows a joint Gaussian distribution. The functions $m(\cdot)$ and $k(\cdot,\cdot)$ define the mean and covariance structure of this distribution, respectively.

To make this definition concrete, consider a finite set of inputs:

$$X = {x_1,\dots,x_n} \tag{15.10.2}$$

Evaluating the function at these points produces a vector of function values:

$$f_X = (f(x_1), \dots, f(x_n))^\top \tag{15.10.3}$$

The defining property of a Gaussian process is that this vector is multivariate normal:

$$f_X \sim \mathcal{N}(m_X, K_{XX}) \tag{15.10.4}$$

where the mean vector and covariance matrix are given by:

$$(m_X)i = m(x_i), \qquad (K_{XX})_{ij} = k(x_i, x_j) \tag{15.10.5}$$

This construction shows that a Gaussian process is completely characterized by its mean function and covariance kernel. The mean function $m(\cdot)$ describes the expected value of the function at each input, while the kernel $k(\cdot,\cdot)$ encodes how function values at different inputs are correlated. The choice of kernel is particularly important, as it determines properties such as smoothness, periodicity, and the scale of variation of the functions drawn from the process.

An important conceptual point is that the Gaussian process defines a consistent family of finite-dimensional distributions. Any subset of inputs yields a multivariate normal distribution, and these distributions are compatible with one another. This consistency allows the Gaussian process to be interpreted as a distribution over an infinite-dimensional object, even though all computations are carried out on finite collections of points.

From a computational perspective, inference with Gaussian processes reduces to operations involving the covariance matrix $K_{XX}$. As will be developed in subsequent subsections, prediction and uncertainty quantification require solving linear systems and performing matrix factorizations involving this matrix. The structure and conditioning of $K_{XX}$ therefore play a central role in both the theoretical properties and practical implementation of Gaussian process methods.

## 15.10.2. Observations and Posterior Prediction

Observations are modeled with additive Gaussian noise:

$$y_i = f(x_i) + \varepsilon_i, \qquad\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2) \tag{15.10.6}$$

This assumption reflects the idea that the observed data are noisy evaluations of the underlying function. The noise is typically taken to be independent and identically distributed with variance $\sigma_n^2$, representing measurement uncertainty or model discrepancy.

Under this model, the observed data vector $y$ follows a multivariate normal distribution:

$$y \sim \mathcal{N}(m_X, K_{XX} + \sigma_n^2 I) \tag{15.10.7}$$

The covariance matrix is the sum of the prior covariance $K_{XX}$ and a diagonal noise term $\sigma_n^2 I$. This addition reflects the fact that uncertainty in the observations arises from both the underlying function and the measurement noise.

To perform prediction at new inputs, consider a set of test points $X_* = \{x_1^*, \dots, x_{n}^*\}$. The covariance structure between training and test points is described by the block matrices:

$$K_{**} = K_{X_* X_*}, \quad K_{X_*} = K_{XX_*}, \quad K_{*X} = K_{X_*X} \tag{15.10.8}$$

These matrices encode the correlations within the test points and between training and test points, and they play a central role in posterior prediction.

The joint distribution of the observed values and the function values at the test points is Gaussian:

$$\begin{pmatrix}y \\f_*\end{pmatrix}\sim\mathcal{N}\left(\begin{pmatrix}m_X \\m_*\end{pmatrix},\begin{pmatrix}K_{XX} + \sigma_n^2 I & K_{X*} \\K_{*X} & K_{**}\end{pmatrix}\right) \tag{15.10.9}$$

This joint distribution captures all dependencies between observed and predicted quantities.

Conditioning on the observed data yields the posterior predictive distribution for the test points. The mean is given by:

$$\mu_* = m_* + K_{*X}(K_{XX} + \sigma_n^2 I)^{-1}(y - m_X) \tag{15.10.10}$$

and the covariance is:

$$\Sigma_* = K_{**} - K_{*X}(K_{XX} + \sigma_n^2 I)^{-1} K_{X*} \tag{15.10.11}$$

The predictive mean can be interpreted as a correction to the prior mean, where the adjustment is determined by how strongly the test points are correlated with the observed data. The predictive covariance reflects the remaining uncertainty after conditioning, decreasing in regions where data are informative and remaining large where data are sparse.

A crucial numerical point emphasized in the source is that these expressions should be computed via linear solves, not matrix inverses. Direct computation of the inverse matrix is both computationally inefficient and numerically unstable. Instead, one uses a Cholesky factorization:

$$K_{XX} + \sigma_n^2 I = LL^\top \tag{15.10.12}$$

where $L$ is a lower triangular matrix. The required quantities are then obtained by solving triangular systems, which is more stable and efficient. This approach is essential for reliable Gaussian process inference, particularly when the covariance matrix is large or ill-conditioned.

### Rust Implementation

Following the discussion in Section 15.10.2 on observations and posterior prediction in Gaussian process regression, Program 15.10.1 provides a practical implementation of posterior inference using noisy observations. In this section, the theoretical formulation establishes how observed data, modeled with additive Gaussian noise as in Equation (15.10.6), lead to a joint Gaussian structure and a posterior predictive distribution through conditioning. The program translates these expressions into a numerically stable computational procedure, constructing the covariance matrices defined in Equation (15.10.8) and evaluating the predictive mean and covariance from Equations (15.10.10) and (15.10.11). Particular emphasis is placed on the use of Cholesky factorization, as prescribed in Equation (15.10.12), to ensure stability and efficiency in solving the associated linear systems.

At the core of the implementation is the construction of covariance matrices that encode the relationships between training and test data. The function `covariance_matrix` evaluates the kernel function pairwise across input sets, producing the matrices $K_{XX}$, $K_{X*}$, and $K_{**}$ introduced in Equation (15.10.8). These matrices represent, respectively, the covariance among training points, the cross-covariance between training and test points, and the covariance among test points. The function `mean_vector` evaluates the prior mean at the corresponding inputs, allowing the predictive expressions to incorporate both prior structure and observed data.

The additive noise model described in Equation (15.10.6) is incorporated by augmenting the training covariance matrix with a diagonal term $\sigma_n^2 I$, forming the matrix $K_{XX} + \sigma_n^2 I$ in Equation (15.10.7). This operation is implemented directly in the program by adding the noise variance and a small jitter term to the diagonal of the covariance matrix. The jitter ensures numerical stability by preventing near-singularity, especially when training points are closely spaced.

To evaluate the predictive distribution, the program follows the numerically stable formulation emphasized in Equation (15.10.12). The function `cholesky_decompose` computes the factorization $K_{XX} + \sigma_n^2 I = LL^\top$, and subsequent operations are expressed as triangular solves rather than matrix inversions. The function `solve_spd_cholesky` computes the vector $\alpha = (K_{XX} + \sigma_n^2 I)^{-1}(y - m_X)$ by solving two triangular systems. This vector is then used to compute the predictive mean in Equation (15.10.10) through a matrix-vector product with $K_{*X}$.

The predictive covariance is computed using the identity in Equation (15.10.11). Instead of forming the inverse explicitly, the program computes an intermediate matrix $V = L^{-1} K_{X*}$ using the function `solve_lower_triangular_multiple_rhs`. The covariance is then obtained as $K_{**} - V^\top V$, which preserves numerical stability and avoids unnecessary computational cost. The diagonal of this matrix yields the predictive variances, providing a measure of uncertainty at each test point.

The `main` function integrates these components into a complete workflow. It generates synthetic training data, constructs a test grid, specifies kernel hyperparameters and noise levels, and invokes the Gaussian process prediction method. The results are presented through structured output, including the solve vector $\alpha$, the predictive mean, and the predictive variance. This end-to-end process demonstrates how the theoretical framework of Gaussian processes can be translated into reliable numerical computation using standard linear algebra techniques.

```rust
// Program 15.10.1: Exact Gaussian Process Posterior Prediction via Cholesky Factorization
//
// Problem Statement:
// Implement exact Gaussian process regression for one-dimensional inputs with
// additive Gaussian observation noise. Given training data
//
//     y_i = f(x_i) + ε_i,   ε_i ~ N(0, σ_n^2),
//
// the program constructs the kernel matrices appearing in Equations (15.10.8)
// and computes the posterior predictive mean and covariance at test points using
// Equations (15.10.10) and (15.10.11). In accordance with Equation (15.10.12),
// all required quantities are obtained through Cholesky factorization and
// triangular solves rather than explicit matrix inversion.
//
// The implementation uses a squared-exponential kernel and reports posterior
// means, marginal variances, and selected covariance entries for a synthetic
// noisy dataset.

use std::f64::consts::PI;

// ----------------------------
// Basic Linear Algebra Types
// ----------------------------

type Vector = Vec<f64>;
type Matrix = Vec<Vec<f64>>;

// ----------------------------
// Matrix Utilities
// ----------------------------

fn zeros_matrix(rows: usize, cols: usize) -> Matrix {
    vec![vec![0.0; cols]; rows]
}

fn transpose(a: &Matrix) -> Matrix {
    let rows = a.len();
    let cols = a[0].len();
    let mut at = zeros_matrix(cols, rows);
    for i in 0..rows {
        for j in 0..cols {
            at[j][i] = a[i][j];
        }
    }
    at
}

fn mat_vec_mul(a: &Matrix, x: &[f64]) -> Vector {
    let rows = a.len();
    let cols = a[0].len();
    assert_eq!(cols, x.len(), "Matrix-vector dimension mismatch.");

    let mut y = vec![0.0; rows];
    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += a[i][j] * x[j];
        }
        y[i] = sum;
    }
    y
}

fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
    let rows = a.len();
    let inner = a[0].len();
    let cols = b[0].len();

    assert_eq!(inner, b.len(), "Matrix-matrix dimension mismatch.");

    let mut c = zeros_matrix(rows, cols);
    for i in 0..rows {
        for k in 0..inner {
            let aik = a[i][k];
            for j in 0..cols {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

fn vec_sub(a: &[f64], b: &[f64]) -> Vector {
    assert_eq!(a.len(), b.len(), "Vector dimension mismatch.");
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn diag_from_matrix(a: &Matrix) -> Vector {
    let n = a.len();
    let mut d = vec![0.0; n];
    for i in 0..n {
        d[i] = a[i][i];
    }
    d
}

// ----------------------------
// Cholesky and Triangular Solves
// ----------------------------

fn cholesky_decompose(a: &Matrix) -> Result<Matrix, String> {
    let n = a.len();
    if n == 0 || a[0].len() != n {
        return Err("Matrix must be nonempty and square for Cholesky decomposition.".to_string());
    }

    let mut l = zeros_matrix(n, n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }

            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "Matrix is not positive definite at diagonal entry {}.",
                        i
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    Ok(l)
}

fn forward_substitution(l: &Matrix, b: &[f64]) -> Result<Vector, String> {
    let n = l.len();
    assert_eq!(b.len(), n, "Dimension mismatch in forward substitution.");

    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        if l[i][i].abs() < 1.0e-15 {
            return Err(format!("Zero diagonal encountered at row {}.", i));
        }
        y[i] = sum / l[i][i];
    }
    Ok(y)
}

fn backward_substitution_upper(u: &Matrix, y: &[f64]) -> Result<Vector, String> {
    let n = u.len();
    assert_eq!(y.len(), n, "Dimension mismatch in backward substitution.");

    let mut x = vec![0.0; n];
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= u[i][j] * x[j];
        }
        if u[i][i].abs() < 1.0e-15 {
            return Err(format!("Zero diagonal encountered at row {}.", i));
        }
        x[i] = sum / u[i][i];
    }
    Ok(x)
}

fn solve_spd_cholesky(l: &Matrix, b: &[f64]) -> Result<Vector, String> {
    let y = forward_substitution(l, b)?;
    let lt = transpose(l);
    backward_substitution_upper(&lt, &y)
}

fn solve_lower_triangular_multiple_rhs(l: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    let n = l.len();
    let cols = b[0].len();
    assert_eq!(b.len(), n, "Dimension mismatch in lower-triangular solve.");

    let mut x = zeros_matrix(n, cols);
    for j in 0..cols {
        for i in 0..n {
            let mut sum = b[i][j];
            for k in 0..i {
                sum -= l[i][k] * x[k][j];
            }
            if l[i][i].abs() < 1.0e-15 {
                return Err(format!("Zero diagonal encountered at row {}.", i));
            }
            x[i][j] = sum / l[i][i];
        }
    }
    Ok(x)
}

// ----------------------------
// Mean Function and Kernel
// ----------------------------

#[derive(Clone, Copy, Debug)]
struct KernelHyperparameters {
    amplitude: f64,
    length_scale: f64,
}

fn mean_function(x: f64) -> f64 {
    let _ = x;
    0.0
}

fn squared_exponential_kernel(x: f64, y: f64, hyp: KernelHyperparameters) -> f64 {
    let r = x - y;
    let ell2 = hyp.length_scale * hyp.length_scale;
    let amp2 = hyp.amplitude * hyp.amplitude;
    amp2 * (-0.5 * r * r / ell2).exp()
}

fn covariance_matrix(xs: &[f64], ys: &[f64], hyp: KernelHyperparameters) -> Matrix {
    let mut k = zeros_matrix(xs.len(), ys.len());
    for i in 0..xs.len() {
        for j in 0..ys.len() {
            k[i][j] = squared_exponential_kernel(xs[i], ys[j], hyp);
        }
    }
    k
}

fn mean_vector(xs: &[f64]) -> Vector {
    xs.iter().map(|&x| mean_function(x)).collect()
}

// ----------------------------
// Gaussian Process Posterior Prediction
// ----------------------------

#[derive(Clone, Debug)]
struct GpPrediction {
    predictive_mean: Vector,
    predictive_covariance: Matrix,
    predictive_variance: Vector,
    alpha: Vector,
}

fn gp_posterior_predict(
    x_train: &[f64],
    y_train: &[f64],
    x_test: &[f64],
    hyp: KernelHyperparameters,
    noise_std: f64,
    jitter: f64,
) -> Result<GpPrediction, String> {
    let n = x_train.len();
    let m = x_test.len();

    if y_train.len() != n {
        return Err("Training input/output length mismatch.".to_string());
    }
    if n == 0 || m == 0 {
        return Err("Training and test sets must be nonempty.".to_string());
    }

    let m_x = mean_vector(x_train);
    let m_star = mean_vector(x_test);

    // K_XX + sigma_n^2 I
    let mut k_xx = covariance_matrix(x_train, x_train, hyp);
    let noise_var = noise_std * noise_std;
    for i in 0..n {
        k_xx[i][i] += noise_var + jitter;
    }

    // Cross-covariances and test covariance
    let k_x_star = covariance_matrix(x_train, x_test, hyp); // K_{X*}
    let k_star_x = transpose(&k_x_star); // K_{*X}
    let k_star_star = covariance_matrix(x_test, x_test, hyp); // K_{**}

    // Cholesky factorization: K_XX + sigma_n^2 I = L L^T
    let l = cholesky_decompose(&k_xx)?;

    // alpha = (K_XX + sigma_n^2 I)^(-1) (y - m_X) via solves
    let centered_y = vec_sub(y_train, &m_x);
    let alpha = solve_spd_cholesky(&l, &centered_y)?;

    // mu_* = m_* + K_{*X} alpha
    let correction = mat_vec_mul(&k_star_x, &alpha);
    let predictive_mean: Vector = m_star
        .iter()
        .zip(correction.iter())
        .map(|(a, b)| a + b)
        .collect();

    // Compute V = L^{-1} K_{X*}
    let v = solve_lower_triangular_multiple_rhs(&l, &k_x_star)?;

    // Sigma_* = K_{**} - V^T V
    let vt = transpose(&v);
    let vt_v = mat_mul(&vt, &v);

    let mut predictive_covariance = zeros_matrix(m, m);
    for i in 0..m {
        for j in 0..m {
            predictive_covariance[i][j] = k_star_star[i][j] - vt_v[i][j];
        }
    }

    let predictive_variance = diag_from_matrix(&predictive_covariance);

    Ok(GpPrediction {
        predictive_mean,
        predictive_covariance,
        predictive_variance,
        alpha,
    })
}

// ----------------------------
// Synthetic Data
// ----------------------------

fn latent_function(x: f64) -> f64 {
    (2.0 * PI * x).sin() + 0.35 * x
}

fn deterministic_noise(i: usize) -> f64 {
    let noise_table = [
        0.04, -0.06, 0.03, -0.05, 0.02, 0.00, 0.05, -0.04, 0.01, -0.03, 0.06, -0.02,
    ];
    noise_table[i % noise_table.len()]
}

fn generate_training_data() -> (Vector, Vector) {
    let x_train = vec![-1.00, -0.75, -0.50, -0.20, 0.00, 0.20, 0.45, 0.70, 0.95];
    let mut y_train = Vec::with_capacity(x_train.len());

    for (i, &x) in x_train.iter().enumerate() {
        y_train.push(latent_function(x) + deterministic_noise(i));
    }

    (x_train, y_train)
}

fn generate_test_grid(a: f64, b: f64, n: usize) -> Vector {
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        x.push(a + (b - a) * t);
    }
    x
}

// ----------------------------
// Reporting Helpers
// ----------------------------

fn print_training_data(x_train: &[f64], y_train: &[f64]) {
    println!("Training Data");
    println!("=============");
    println!("{:>4} {:>14} {:>14}", "i", "x_i", "y_i");
    for i in 0..x_train.len() {
        println!("{:>4} {:>14.6} {:>14.6}", i, x_train[i], y_train[i]);
    }
    println!();
}

fn print_hyperparameters(hyp: KernelHyperparameters, noise_std: f64, jitter: f64) {
    println!("Gaussian Process Hyperparameters");
    println!("===============================");
    println!("Amplitude            = {:>.10}", hyp.amplitude);
    println!("Length scale         = {:>.10}", hyp.length_scale);
    println!("Noise std. dev.      = {:>.10}", noise_std);
    println!("Jitter               = {:>.10}", jitter);
    println!();
}

fn print_prediction_table(x_test: &[f64], pred: &GpPrediction, max_rows: usize) {
    println!("Posterior Predictive Summary");
    println!("============================");
    println!(
        "{:>4} {:>14} {:>18} {:>18}",
        "i", "x_*", "mean", "variance"
    );

    for i in 0..x_test.len().min(max_rows) {
        println!(
            "{:>4} {:>14.6} {:>18.10} {:>18.10}",
            i,
            x_test[i],
            pred.predictive_mean[i],
            pred.predictive_variance[i]
        );
    }
    println!();
}

fn print_selected_covariance_entries(x_test: &[f64], pred: &GpPrediction, indices: &[usize]) {
    println!("Selected Posterior Covariance Entries");
    println!("=====================================");
    for &i in indices {
        for &j in indices {
            println!(
                "Cov[f({:.3}), f({:.3})] = {:>.10}",
                x_test[i],
                x_test[j],
                pred.predictive_covariance[i][j]
            );
        }
    }
    println!();
}

fn print_alpha_vector(alpha: &[f64]) {
    println!("Solve Vector alpha = (K_XX + sigma_n^2 I)^(-1) (y - m_X)");
    println!("=========================================================");
    for (i, value) in alpha.iter().enumerate() {
        println!("alpha[{}] = {:>.10}", i, value);
    }
    println!();
}

// ----------------------------
// Main Demonstration
// ----------------------------

fn main() -> Result<(), String> {
    let (x_train, y_train) = generate_training_data();
    let x_test = generate_test_grid(-1.10, 1.10, 21);

    let hyp = KernelHyperparameters {
        amplitude: 1.0,
        length_scale: 0.32,
    };

    let noise_std = 0.08;
    let jitter = 1.0e-10;

    print_training_data(&x_train, &y_train);
    print_hyperparameters(hyp, noise_std, jitter);

    let prediction = gp_posterior_predict(
        &x_train,
        &y_train,
        &x_test,
        hyp,
        noise_std,
        jitter,
    )?;

    print_alpha_vector(&prediction.alpha);
    print_prediction_table(&x_test, &prediction, 21);

    let selected_indices = [0usize, 10usize, 20usize];
    print_selected_covariance_entries(&x_test, &prediction, &selected_indices);

    println!("Interpretation");
    println!("==============");
    println!("The predictive mean is computed as the prior mean plus a data-driven correction");
    println!("term involving K_*X and the solve vector alpha, in accordance with Equation");
    println!("(15.10.10). The predictive covariance is obtained by subtracting the data-informed");
    println!("reduction term from K_**, as in Equation (15.10.11).");
    println!();
    println!("All linear algebra is performed through Cholesky factorization and triangular");
    println!("solves, rather than explicit inversion, as required by Equation (15.10.12).");
    println!("This improves both numerical stability and computational reliability in exact");
    println!("Gaussian process posterior prediction.");

    Ok(())
}
```

Program 15.10.1 demonstrates how Gaussian process posterior prediction can be implemented in a numerically stable and computationally efficient manner by leveraging Cholesky factorization and structured linear solves. The program reflects the central ideas of Section 15.10.2, showing how noisy observations are incorporated into the covariance structure and how conditioning on data yields both a predictive mean and an associated uncertainty.

The results illustrate the characteristic behavior of Gaussian processes: the predictive mean interpolates the observed data while maintaining smoothness dictated by the kernel, and the predictive variance decreases near observed points and increases in regions with sparse data. These properties highlight the dual role of Gaussian processes as both interpolators and uncertainty quantifiers.

The modular design of the implementation allows for straightforward extensions, including alternative kernel functions, nonzero mean functions, and higher-dimensional inputs. It also provides a foundation for more advanced developments, such as hyperparameter optimization and scalable approximations, which build upon the same computational principles established here.

## 15.10.3. Hyperparameters and Marginal Likelihood

Kernel parameters, such as length scales and amplitudes, together with the noise level, are typically estimated by maximizing the marginal likelihood. These quantities are often referred to as hyperparameters, since they govern the behavior of the Gaussian process prior rather than the function values themselves. Their selection determines how the model balances smoothness, variability, and noise, and therefore has a direct impact on predictive performance.

The log marginal likelihood is given by:

\begin{equation}
\begin{aligned}
\log p(y \mid X, \vartheta)
&= -\frac{1}{2}(y - m_X)^\top K_\vartheta^{-1}(y - m_X) \\
&\quad -\frac{1}{2} \log |K_\vartheta| - \frac{n}{2}\log(2\pi)
\end{aligned}
\tag{15.10.13}
\end{equation}

where,

$$K_\vartheta = K_{XX}(\vartheta) + \sigma_n^2 I \tag{15.10.14}$$

This expression arises from integrating out the latent function values and represents the likelihood of the observed data under the Gaussian process model.

The structure of this objective reveals a fundamental trade-off. The first term, a quadratic form, measures how well the model fits the data. It penalizes deviations between the observed values and the prior mean, weighted by the inverse covariance. The second term, involving the log-determinant of the covariance matrix, penalizes model complexity by accounting for the volume of the uncertainty represented by the kernel. The final term is a normalization constant that depends only on the number of data points.

Thus, the marginal likelihood couples data fit through the quadratic form, model complexity through the log-determinant, and numerical stability through the conditioning of $K_\vartheta$. The optimizer must navigate this trade-off to select hyperparameters that balance fidelity to the data with appropriate regularization.

From a computational perspective, evaluating the marginal likelihood requires solving linear systems and computing the log-determinant of the covariance matrix. These operations are typically carried out using Cholesky factorization, which provides both numerical stability and computational efficiency. The same factorization can be reused to compute all required terms, making it central to practical implementations.

The source emphasizes that this optimization is numerically sensitive. The conditioning of the kernel matrix plays a critical role, since poorly conditioned matrices can lead to unstable solutions or inaccurate evaluations of the objective. The stability of the Cholesky factorization is also essential, as it underpins both the quadratic form and the determinant computation. In addition, the behavior of the nonlinear optimizer must be considered, since the marginal likelihood is generally non-convex and may contain multiple local optima.

Careful parameterization and stable linear algebra are therefore essential (Hoffmann and Onnela, 2025). Techniques such as working with log-transformed parameters, adding small diagonal regularization terms, and using robust optimization strategies help ensure reliable estimation of hyperparameters. These considerations highlight the close interplay between statistical modeling and numerical computation in Gaussian process regression.

### Rust Implementation

Following the discussion in Section 15.10.3 on hyperparameters and marginal likelihood, Program 15.10.2 provides a practical implementation of hyperparameter estimation for Gaussian process regression. In this section, the log marginal likelihood in Equation (15.10.13) is introduced as a principled objective that balances data fit and model complexity through the covariance structure defined in Equation (15.10.14). The program translates this formulation into a numerically stable computational procedure, evaluating the marginal likelihood using Cholesky factorization as emphasized in Equation (15.10.12). It performs optimization in log-parameter space to ensure positivity of the hyperparameters and demonstrates how the resulting estimates influence posterior prediction.

At the core of the implementation is the evaluation of the log marginal likelihood using stable linear algebra. The function `evaluate_log_marginal_likelihood` computes the objective in Equation (15.10.13) by first constructing the covariance matrix $K_\vartheta$ as defined in Equation (15.10.14). The additive noise term $\sigma_n^2 I$ is incorporated directly into the diagonal, along with a small jitter term to ensure numerical stability. The resulting matrix is factorized using the `cholesky_decompose` function, which produces the lower triangular matrix $L$ satisfying Equation (15.10.12).

The quadratic form in Equation (15.10.13) is evaluated using the function `solve_spd_cholesky`, which computes the vector $\alpha = K_\vartheta^{-1}(y - m_X)$ through forward and backward substitution rather than explicit inversion. This approach preserves numerical stability and reduces computational cost. The log-determinant term is obtained directly from the Cholesky factor by summing the logarithms of the diagonal entries, exploiting the identity $\log |K_\vartheta| = 2 \sum_i \log L_{ii}$. Together, these computations provide all components of the marginal likelihood using a single factorization.

The program performs hyperparameter optimization in log space using two complementary strategies. The function `coarse_grid_search` explores a predefined grid of log-transformed parameters to identify a promising region of the parameter space. This is followed by `local_pattern_search`, which refines the solution through iterative local updates. This combination reflects the non-convex nature of the marginal likelihood, where multiple local optima may exist, and demonstrates a practical approach to navigating the objective landscape.

The posterior prediction step reuses the optimized hyperparameters to evaluate the predictive mean and covariance as defined in Equations (15.10.10) and (15.10.11). The function `gp_posterior_predict` mirrors the implementation developed in Section 15.10.2, again relying on Cholesky-based solves to avoid explicit inversion. This reuse highlights the central role of the covariance factorization in both hyperparameter estimation and prediction.

The `main` function integrates these components into a complete workflow. It generates synthetic training data, evaluates the marginal likelihood at an initial guess, performs coarse and refined optimization, and reports the resulting hyperparameters. The program then computes posterior predictions using the optimized model and presents both predictive summaries and selected covariance entries. This end-to-end process illustrates how hyperparameter estimation directly influences the behavior of the Gaussian process model.

```rust
// Program 15.10.2: Gaussian Process Hyperparameter Estimation by Marginal Likelihood
//
// Problem Statement:
// Estimate the hyperparameters of a Gaussian process regression model by maximizing
// the log marginal likelihood. The program uses a squared-exponential kernel with
// three positive hyperparameters:
//
//     amplitude     : kernel scale
//     length_scale  : kernel length scale
//     noise_std     : observation noise standard deviation
//
// The log marginal likelihood is evaluated in the form of Equation (15.10.13),
//
//     log p(y | X, vartheta)
//       = -1/2 (y - m_X)^T K_vartheta^{-1} (y - m_X)
//         -1/2 log |K_vartheta|
//         -n/2 log(2 pi),
//
// where
//
//     K_vartheta = K_XX(vartheta) + sigma_n^2 I
//
// as in Equation (15.10.14).
//
// In accordance with the numerical guidance of the section, the program never
// forms K_vartheta^{-1} explicitly. Instead, it uses a Cholesky factorization
//
//     K_vartheta = L L^T
//
// to evaluate both the quadratic form and the log-determinant stably. The search
// is performed in log-parameter space using a coarse grid followed by a simple
// local refinement procedure. The program also reports posterior predictions on
// a test grid using the optimized hyperparameters.

use std::f64::consts::PI;

// ----------------------------
// Basic Linear Algebra Types
// ----------------------------

type Vector = Vec<f64>;
type Matrix = Vec<Vec<f64>>;

// ----------------------------
// Matrix Utilities
// ----------------------------

fn zeros_matrix(rows: usize, cols: usize) -> Matrix {
    vec![vec![0.0; cols]; rows]
}

fn transpose(a: &Matrix) -> Matrix {
    let rows = a.len();
    let cols = a[0].len();
    let mut at = zeros_matrix(cols, rows);
    for i in 0..rows {
        for j in 0..cols {
            at[j][i] = a[i][j];
        }
    }
    at
}

fn mat_vec_mul(a: &Matrix, x: &[f64]) -> Vector {
    let rows = a.len();
    let cols = a[0].len();
    assert_eq!(cols, x.len(), "Matrix-vector dimension mismatch.");

    let mut y = vec![0.0; rows];
    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += a[i][j] * x[j];
        }
        y[i] = sum;
    }
    y
}

fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
    let rows = a.len();
    let inner = a[0].len();
    let cols = b[0].len();
    assert_eq!(inner, b.len(), "Matrix-matrix dimension mismatch.");

    let mut c = zeros_matrix(rows, cols);
    for i in 0..rows {
        for k in 0..inner {
            let aik = a[i][k];
            for j in 0..cols {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

fn vec_sub(a: &[f64], b: &[f64]) -> Vector {
    assert_eq!(a.len(), b.len(), "Vector dimension mismatch.");
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vector dimension mismatch.");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn diag_from_matrix(a: &Matrix) -> Vector {
    let n = a.len();
    let mut d = vec![0.0; n];
    for i in 0..n {
        d[i] = a[i][i];
    }
    d
}

// ----------------------------
// Cholesky and Triangular Solves
// ----------------------------

fn cholesky_decompose(a: &Matrix) -> Result<Matrix, String> {
    let n = a.len();
    if n == 0 || a[0].len() != n {
        return Err("Matrix must be nonempty and square for Cholesky decomposition.".to_string());
    }

    let mut l = zeros_matrix(n, n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }

            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "Matrix is not positive definite at diagonal entry {}.",
                        i
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    Ok(l)
}

fn forward_substitution(l: &Matrix, b: &[f64]) -> Result<Vector, String> {
    let n = l.len();
    assert_eq!(b.len(), n, "Dimension mismatch in forward substitution.");

    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        if l[i][i].abs() < 1.0e-15 {
            return Err(format!("Zero diagonal encountered at row {}.", i));
        }
        y[i] = sum / l[i][i];
    }
    Ok(y)
}

fn backward_substitution_upper(u: &Matrix, y: &[f64]) -> Result<Vector, String> {
    let n = u.len();
    assert_eq!(y.len(), n, "Dimension mismatch in backward substitution.");

    let mut x = vec![0.0; n];
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= u[i][j] * x[j];
        }
        if u[i][i].abs() < 1.0e-15 {
            return Err(format!("Zero diagonal encountered at row {}.", i));
        }
        x[i] = sum / u[i][i];
    }
    Ok(x)
}

fn solve_spd_cholesky(l: &Matrix, b: &[f64]) -> Result<Vector, String> {
    let y = forward_substitution(l, b)?;
    let lt = transpose(l);
    backward_substitution_upper(&lt, &y)
}

fn solve_lower_triangular_multiple_rhs(l: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    let n = l.len();
    let cols = b[0].len();
    assert_eq!(b.len(), n, "Dimension mismatch in lower-triangular solve.");

    let mut x = zeros_matrix(n, cols);
    for j in 0..cols {
        for i in 0..n {
            let mut sum = b[i][j];
            for k in 0..i {
                sum -= l[i][k] * x[k][j];
            }
            if l[i][i].abs() < 1.0e-15 {
                return Err(format!("Zero diagonal encountered at row {}.", i));
            }
            x[i][j] = sum / l[i][i];
        }
    }
    Ok(x)
}

// ----------------------------
// Gaussian Process Model
// ----------------------------

#[derive(Clone, Copy, Debug)]
struct Hyperparameters {
    amplitude: f64,
    length_scale: f64,
    noise_std: f64,
}

#[derive(Clone, Copy, Debug)]
struct LogHyperparameters {
    log_amplitude: f64,
    log_length_scale: f64,
    log_noise_std: f64,
}

impl LogHyperparameters {
    fn to_hyperparameters(self) -> Hyperparameters {
        Hyperparameters {
            amplitude: self.log_amplitude.exp(),
            length_scale: self.log_length_scale.exp(),
            noise_std: self.log_noise_std.exp(),
        }
    }
}

fn mean_function(x: f64) -> f64 {
    let _ = x;
    0.0
}

fn mean_vector(xs: &[f64]) -> Vector {
    xs.iter().map(|&x| mean_function(x)).collect()
}

fn squared_exponential_kernel(x: f64, y: f64, hyp: Hyperparameters) -> f64 {
    let r = x - y;
    let amp2 = hyp.amplitude * hyp.amplitude;
    let ell2 = hyp.length_scale * hyp.length_scale;
    amp2 * (-0.5 * r * r / ell2).exp()
}

fn covariance_matrix(xs: &[f64], ys: &[f64], hyp: Hyperparameters) -> Matrix {
    let mut k = zeros_matrix(xs.len(), ys.len());
    for i in 0..xs.len() {
        for j in 0..ys.len() {
            k[i][j] = squared_exponential_kernel(xs[i], ys[j], hyp);
        }
    }
    k
}

// ----------------------------
// Marginal Likelihood Evaluation
// ----------------------------

#[derive(Clone, Debug)]
struct MarginalLikelihoodEvaluation {
    log_marginal_likelihood: f64,
    quadratic_term: f64,
    log_determinant: f64,
    alpha: Vector,
    cholesky_factor: Matrix,
}

fn evaluate_log_marginal_likelihood(
    x_train: &[f64],
    y_train: &[f64],
    log_hyp: LogHyperparameters,
    jitter: f64,
) -> Result<MarginalLikelihoodEvaluation, String> {
    let n = x_train.len();
    if y_train.len() != n {
        return Err("Training input/output length mismatch.".to_string());
    }
    if n == 0 {
        return Err("Training set must be nonempty.".to_string());
    }

    let hyp = log_hyp.to_hyperparameters();
    let m_x = mean_vector(x_train);
    let centered_y = vec_sub(y_train, &m_x);

    let mut k = covariance_matrix(x_train, x_train, hyp);
    let noise_var = hyp.noise_std * hyp.noise_std;
    for i in 0..n {
        k[i][i] += noise_var + jitter;
    }

    let l = cholesky_decompose(&k)?;
    let alpha = solve_spd_cholesky(&l, &centered_y)?;

    let quadratic_term = dot(&centered_y, &alpha);

    let mut log_determinant = 0.0;
    for i in 0..n {
        log_determinant += 2.0 * l[i][i].ln();
    }

    let log_marginal_likelihood = -0.5 * quadratic_term
        - 0.5 * log_determinant
        - 0.5 * (n as f64) * (2.0 * PI).ln();

    Ok(MarginalLikelihoodEvaluation {
        log_marginal_likelihood,
        quadratic_term,
        log_determinant,
        alpha,
        cholesky_factor: l,
    })
}

// ----------------------------
// Posterior Prediction with Given Hyperparameters
// ----------------------------

#[derive(Clone, Debug)]
struct GpPrediction {
    predictive_mean: Vector,
    predictive_covariance: Matrix,
    predictive_variance: Vector,
}

fn gp_posterior_predict(
    x_train: &[f64],
    y_train: &[f64],
    x_test: &[f64],
    hyp: Hyperparameters,
    jitter: f64,
) -> Result<GpPrediction, String> {
    let n = x_train.len();
    let m = x_test.len();
    if y_train.len() != n {
        return Err("Training input/output length mismatch.".to_string());
    }
    if n == 0 || m == 0 {
        return Err("Training and test sets must be nonempty.".to_string());
    }

    let m_x = mean_vector(x_train);
    let m_star = mean_vector(x_test);

    let mut k_xx = covariance_matrix(x_train, x_train, hyp);
    let noise_var = hyp.noise_std * hyp.noise_std;
    for i in 0..n {
        k_xx[i][i] += noise_var + jitter;
    }

    let k_x_star = covariance_matrix(x_train, x_test, hyp);
    let k_star_x = transpose(&k_x_star);
    let k_star_star = covariance_matrix(x_test, x_test, hyp);

    let l = cholesky_decompose(&k_xx)?;
    let centered_y = vec_sub(y_train, &m_x);
    let alpha = solve_spd_cholesky(&l, &centered_y)?;

    let correction = mat_vec_mul(&k_star_x, &alpha);
    let predictive_mean: Vector = m_star
        .iter()
        .zip(correction.iter())
        .map(|(a, b)| a + b)
        .collect();

    let v = solve_lower_triangular_multiple_rhs(&l, &k_x_star)?;
    let vt = transpose(&v);
    let vt_v = mat_mul(&vt, &v);

    let mut predictive_covariance = zeros_matrix(m, m);
    for i in 0..m {
        for j in 0..m {
            predictive_covariance[i][j] = k_star_star[i][j] - vt_v[i][j];
        }
    }

    let predictive_variance = diag_from_matrix(&predictive_covariance);

    Ok(GpPrediction {
        predictive_mean,
        predictive_covariance,
        predictive_variance,
    })
}

// ----------------------------
// Hyperparameter Optimization
// ----------------------------

fn linspace(a: f64, b: f64, n: usize) -> Vector {
    let mut x = Vec::with_capacity(n);
    if n == 1 {
        x.push(a);
        return x;
    }
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        x.push(a + (b - a) * t);
    }
    x
}

fn coarse_grid_search(
    x_train: &[f64],
    y_train: &[f64],
    jitter: f64,
) -> Result<(LogHyperparameters, MarginalLikelihoodEvaluation), String> {
    let log_amp_grid = linspace(-1.5, 1.0, 9);
    let log_ell_grid = linspace(-2.0, 0.2, 10);
    let log_noise_grid = linspace(-4.0, -1.5, 8);

    let mut best_log_hyp = LogHyperparameters {
        log_amplitude: 0.0,
        log_length_scale: 0.0,
        log_noise_std: -2.0,
    };

    let mut best_eval = evaluate_log_marginal_likelihood(x_train, y_train, best_log_hyp, jitter)?;

    for &la in &log_amp_grid {
        for &ll in &log_ell_grid {
            for &ln in &log_noise_grid {
                let cand = LogHyperparameters {
                    log_amplitude: la,
                    log_length_scale: ll,
                    log_noise_std: ln,
                };

                if let Ok(eval) = evaluate_log_marginal_likelihood(x_train, y_train, cand, jitter) {
                    if eval.log_marginal_likelihood > best_eval.log_marginal_likelihood {
                        best_log_hyp = cand;
                        best_eval = eval;
                    }
                }
            }
        }
    }

    Ok((best_log_hyp, best_eval))
}

fn local_pattern_search(
    x_train: &[f64],
    y_train: &[f64],
    jitter: f64,
    start: LogHyperparameters,
) -> Result<(LogHyperparameters, MarginalLikelihoodEvaluation), String> {
    let mut current = start;
    let mut current_eval = evaluate_log_marginal_likelihood(x_train, y_train, current, jitter)?;

    let mut step_a = 0.25;
    let mut step_l = 0.20;
    let mut step_n = 0.20;

    for _ in 0..30 {
        let mut improved = false;

        let candidates = [
            LogHyperparameters {
                log_amplitude: current.log_amplitude + step_a,
                ..current
            },
            LogHyperparameters {
                log_amplitude: current.log_amplitude - step_a,
                ..current
            },
            LogHyperparameters {
                log_length_scale: current.log_length_scale + step_l,
                ..current
            },
            LogHyperparameters {
                log_length_scale: current.log_length_scale - step_l,
                ..current
            },
            LogHyperparameters {
                log_noise_std: current.log_noise_std + step_n,
                ..current
            },
            LogHyperparameters {
                log_noise_std: current.log_noise_std - step_n,
                ..current
            },
        ];

        for cand in candidates {
            if let Ok(eval) = evaluate_log_marginal_likelihood(x_train, y_train, cand, jitter) {
                if eval.log_marginal_likelihood > current_eval.log_marginal_likelihood {
                    current = cand;
                    current_eval = eval;
                    improved = true;
                }
            }
        }

        if !improved {
            step_a *= 0.5;
            step_l *= 0.5;
            step_n *= 0.5;
            if step_a < 1.0e-3 && step_l < 1.0e-3 && step_n < 1.0e-3 {
                break;
            }
        }
    }

    Ok((current, current_eval))
}

// ----------------------------
// Synthetic Data
// ----------------------------

fn latent_function(x: f64) -> f64 {
    (2.0 * PI * x).sin() + 0.35 * x
}

fn deterministic_noise(i: usize) -> f64 {
    let noise_table = [
        0.03, -0.05, 0.04, -0.04, 0.02, -0.01, 0.05, -0.03, 0.01, -0.02, 0.04, -0.04,
    ];
    noise_table[i % noise_table.len()]
}

fn generate_training_data() -> (Vector, Vector) {
    let x_train = vec![
        -1.00, -0.82, -0.64, -0.48, -0.20, 0.00, 0.18, 0.36, 0.58, 0.78, 0.96,
    ];

    let mut y_train = Vec::with_capacity(x_train.len());
    for (i, &x) in x_train.iter().enumerate() {
        y_train.push(latent_function(x) + deterministic_noise(i));
    }

    (x_train, y_train)
}

fn generate_test_grid(a: f64, b: f64, n: usize) -> Vector {
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        x.push(a + (b - a) * t);
    }
    x
}

// ----------------------------
// Reporting Helpers
// ----------------------------

fn print_training_data(x_train: &[f64], y_train: &[f64]) {
    println!("Training Data");
    println!("=============");
    println!("{:>4} {:>14} {:>14}", "i", "x_i", "y_i");
    for i in 0..x_train.len() {
        println!("{:>4} {:>14.6} {:>14.6}", i, x_train[i], y_train[i]);
    }
    println!();
}

fn print_hyperparameters(title: &str, hyp: Hyperparameters) {
    println!("{}", title);
    println!("{}", "=".repeat(title.len()));
    println!("Amplitude            = {:>.10}", hyp.amplitude);
    println!("Length scale         = {:>.10}", hyp.length_scale);
    println!("Noise std. dev.      = {:>.10}", hyp.noise_std);
    println!();
}

fn print_marginal_likelihood_breakdown(
    title: &str,
    eval: &MarginalLikelihoodEvaluation,
) {
    println!("{}", title);
    println!("{}", "=".repeat(title.len()));
    println!("Log marginal likelihood   = {:>.10}", eval.log_marginal_likelihood);
    println!("Quadratic term            = {:>.10}", eval.quadratic_term);
    println!("Log determinant           = {:>.10}", eval.log_determinant);
    println!();
}

fn print_alpha_vector(alpha: &[f64]) {
    println!("Solve Vector alpha");
    println!("==================");
    for (i, value) in alpha.iter().enumerate() {
        println!("alpha[{}] = {:>.10}", i, value);
    }
    println!();
}

fn print_prediction_table(x_test: &[f64], pred: &GpPrediction, max_rows: usize) {
    println!("Posterior Prediction with Optimized Hyperparameters");
    println!("==================================================");
    println!(
        "{:>4} {:>14} {:>18} {:>18}",
        "i", "x_*", "mean", "variance"
    );

    for i in 0..x_test.len().min(max_rows) {
        println!(
            "{:>4} {:>14.6} {:>18.10} {:>18.10}",
            i,
            x_test[i],
            pred.predictive_mean[i],
            pred.predictive_variance[i]
        );
    }
    println!();
}

fn print_selected_covariance_entries(x_test: &[f64], pred: &GpPrediction, indices: &[usize]) {
    println!("Selected Predictive Covariance Entries");
    println!("======================================");
    for &i in indices {
        for &j in indices {
            println!(
                "Cov[f({:.3}), f({:.3})] = {:>.10}",
                x_test[i],
                x_test[j],
                pred.predictive_covariance[i][j]
            );
        }
    }
    println!();
}

fn print_search_point(label: &str, log_hyp: LogHyperparameters, eval: &MarginalLikelihoodEvaluation) {
    let hyp = log_hyp.to_hyperparameters();
    println!("{}", label);
    println!("{}", "=".repeat(label.len()));
    println!("log amplitude           = {:>.10}", log_hyp.log_amplitude);
    println!("log length scale        = {:>.10}", log_hyp.log_length_scale);
    println!("log noise std. dev.     = {:>.10}", log_hyp.log_noise_std);
    println!("amplitude               = {:>.10}", hyp.amplitude);
    println!("length scale            = {:>.10}", hyp.length_scale);
    println!("noise std. dev.         = {:>.10}", hyp.noise_std);
    println!("log marginal likelihood = {:>.10}", eval.log_marginal_likelihood);
    println!();
}

// ----------------------------
// Main Demonstration
// ----------------------------

fn main() -> Result<(), String> {
    let (x_train, y_train) = generate_training_data();
    let x_test = generate_test_grid(-1.10, 1.10, 25);
    let jitter = 1.0e-10;

    print_training_data(&x_train, &y_train);

    let initial_log_hyp = LogHyperparameters {
        log_amplitude: 0.0,
        log_length_scale: -1.0,
        log_noise_std: -2.5,
    };

    let initial_eval =
        evaluate_log_marginal_likelihood(&x_train, &y_train, initial_log_hyp, jitter)?;
    print_search_point("Initial Hyperparameter Guess", initial_log_hyp, &initial_eval);
    print_marginal_likelihood_breakdown(
        "Initial Marginal Likelihood Breakdown",
        &initial_eval,
    );

    let (coarse_best_log_hyp, coarse_best_eval) =
        coarse_grid_search(&x_train, &y_train, jitter)?;
    print_search_point("Best Coarse Grid Point", coarse_best_log_hyp, &coarse_best_eval);

    let (best_log_hyp, best_eval) =
        local_pattern_search(&x_train, &y_train, jitter, coarse_best_log_hyp)?;
    print_search_point("Refined Optimum in Log-Parameter Space", best_log_hyp, &best_eval);
    print_marginal_likelihood_breakdown(
        "Optimized Marginal Likelihood Breakdown",
        &best_eval,
    );

    let best_hyp = best_log_hyp.to_hyperparameters();
    print_hyperparameters("Optimized Hyperparameters", best_hyp);
    print_alpha_vector(&best_eval.alpha);

    let l_diag = diag_from_matrix(&best_eval.cholesky_factor);
    println!("Cholesky Diagonal");
    println!("=================");
    for (i, value) in l_diag.iter().enumerate() {
        println!("L[{},{}] = {:>.10}", i, i, value);
    }
    println!();

    let prediction = gp_posterior_predict(&x_train, &y_train, &x_test, best_hyp, jitter)?;
    print_prediction_table(&x_test, &prediction, 25);

    let selected_indices = [0usize, 12usize, 24usize];
    print_selected_covariance_entries(&x_test, &prediction, &selected_indices);

    println!("Interpretation");
    println!("==============");
    println!("The optimized hyperparameters are obtained by maximizing the log marginal");
    println!("likelihood in Equation (15.10.13), with the covariance matrix K_vartheta");
    println!("constructed as in Equation (15.10.14). The Cholesky factorization is reused");
    println!("to evaluate both the quadratic form and the log-determinant stably, avoiding");
    println!("explicit matrix inversion.");
    println!();
    println!("The resulting hyperparameters determine the smoothness, scale, and noise level");
    println!("of the Gaussian process prior. They therefore control both the fitted posterior");
    println!("mean and the associated predictive uncertainty. This illustrates the close");
    println!("interaction between statistical modeling and stable numerical linear algebra in");
    println!("Gaussian process regression.");

    Ok(())
}
```

Program 15.10.2 demonstrates how hyperparameter estimation in Gaussian process regression can be formulated as an optimization problem based on the marginal likelihood. By combining data fit and model complexity within a single objective, the marginal likelihood provides a principled mechanism for selecting kernel parameters that balance flexibility and regularization.

The implementation highlights the importance of numerical stability in evaluating this objective. The use of Cholesky factorization ensures reliable computation of both the quadratic form and the log-determinant, avoiding the pitfalls associated with explicit matrix inversion. The results show how the optimized hyperparameters influence both the posterior mean and the predictive uncertainty, reinforcing the close connection between statistical modeling and numerical linear algebra.

The modular structure of the program allows for straightforward extensions, including gradient-based optimization methods, alternative kernel functions, and scalable approximations for large datasets. These developments build upon the same computational principles demonstrated here and form the basis for modern Gaussian process inference in high-dimensional and large-scale settings.

## 15.10.4. Computational Complexity and the Need for Approximation

Exact Gaussian process regression has significant computational cost. The dominant expense arises from operations on the dense covariance matrix $K_{XX}$, whose size grows with the number of data points. In particular, training requires a matrix factorization:

$$O(n^3) \tag{15.10.15}$$

which corresponds to the cost of Cholesky decomposition of an $n \times n$ matrix. This cubic scaling quickly becomes prohibitive as $n$ increases, limiting the applicability of exact Gaussian processes to datasets of moderate size.

The memory requirement is also substantial:

$$O(n^2) \tag{15.10.16}$$

since the full covariance matrix must be stored. This quadratic storage cost can exceed available memory for large datasets, even before computational considerations become limiting.

Prediction introduces additional costs. For each batch of test points, evaluating the predictive mean and covariance requires operations involving the training covariance matrix and its factorization. The cost per batch is:

$$O(n^2) \quad \text{(plus additional cost for variances)} \tag{15.10.17}$$

where the additional cost arises from computing predictive uncertainties, which involve further matrix-vector or matrix-matrix operations.

These computational requirements highlight a fundamental limitation of Gaussian process regression. The cubic scaling in training and quadratic scaling in memory and prediction restrict the use of exact methods to relatively small datasets. As the number of observations grows, both computational time and memory usage become prohibitive.

This cubic scaling is the central practical limitation of Gaussian processes and motivates the development of approximation methods. A wide range of approaches has been proposed to address this issue, including low-rank approximations, sparse representations, inducing-point methods, and structured kernel techniques. These methods aim to reduce computational complexity while preserving the key probabilistic and predictive properties of the Gaussian process framework.

Thus, while Gaussian processes provide a powerful and flexible modeling tool, their practical use in large-scale settings depends critically on the development and application of efficient approximation strategies.

## 15.10.5. Scalable and Structured Approaches

Modern Gaussian process research focuses on exploiting structure in the kernel matrix to reduce computational cost. The central idea is that, although the covariance matrix is dense in general, it often exhibits patterns or redundancies that can be leveraged to obtain more efficient representations. By identifying and exploiting such structure, it is possible to reduce both computational time and memory requirements while retaining the essential probabilistic properties of the model.

Several key approaches have been developed to achieve this goal.

**Inducing-point (sparse) methods** introduce a smaller set of representative inputs that act as a compressed summary of the full dataset. Instead of working directly with all $n$ data points, the model is approximated using a subset of $m \ll n$ inducing points. The covariance structure is then expressed in terms of these points, reducing the computational cost from cubic in $n$ to cubic in $m$, with additional linear or quadratic terms in $n$. This approach provides a flexible trade-off between accuracy and efficiency, controlled by the number and placement of inducing points.

**Low-rank structure and decomposition methods** aim to reduce the effective dimensionality of the covariance matrix. By approximating the kernel matrix with a low-rank representation, these methods capture the dominant modes of variation in the data while discarding less significant components. This leads to reduced computational cost in both factorization and prediction, and is particularly effective when the underlying function exhibits smooth or correlated behavior.

**FFT-based and structured kernel methods** exploit specific algebraic structures in the covariance matrix, such as Toeplitz or Kronecker structure. When inputs lie on regular grids or when the kernel has translation-invariant properties, fast transforms such as the Fast Fourier Transform can be used to accelerate matrix-vector operations. These methods can achieve near-linear complexity in favorable settings, making them attractive for large-scale problems with structured inputs.

These approaches aim to preserve the probabilistic structure of the model while reducing computational cost from cubic to near-linear or quadratic regimes. The challenge is to maintain accuracy and uncertainty quantification while introducing approximations that simplify the underlying linear algebra.

A representative application arises in engineering simulation. For example, in finite element modeling, the simulator defines an expensive function $g(x)$ that maps input parameters to outputs of interest. Evaluating this function may require solving large systems of equations, making repeated evaluations costly. Gaussian process regression can be used to construct a surrogate model $f(x) \approx g(x)$, which provides fast predictions along with uncertainty estimates that quantify the reliability of the approximation.

In such settings, scalable inference is essential for practical deployment. The ability to train and evaluate the surrogate efficiently determines whether the method can be integrated into design, optimization, or uncertainty quantification workflows.

Recent work demonstrates the use of inducing-point selection guided by reduced-order models, achieving significant improvements in training efficiency (Røstum et al., 2025). By combining model reduction techniques with probabilistic inference, these approaches further enhance the scalability and applicability of Gaussian process methods in complex scientific and engineering problems.

### Rust Implementation

Following the discussion in Section 15.10.5 on scalable and structured Gaussian process methods, Program 15.10.3 provides a practical implementation of inducing-point Gaussian process regression using a low-rank approximation of the kernel matrix. In large-scale simulation settings, direct Gaussian process training incurs cubic computational cost in the number of data points, making it impractical for real-world applications. This program demonstrates how the inducing-point framework reduces this complexity by projecting the full covariance structure onto a smaller set of representative inputs, thereby enabling efficient training and prediction. The implementation emphasizes numerically stable linear algebra through Cholesky factorization and leverages the Woodbury identity to avoid explicit inversion of large matrices. The resulting surrogate model approximates the expensive simulator $g(x)$ while preserving probabilistic predictions and uncertainty quantification, illustrating the central ideas of scalable Gaussian process inference in a concrete computational setting.

At the core of the implementation is the construction of kernel matrices between training inputs and inducing points, which realizes the low-rank approximation discussed in Equation (15.10.5). The function `build_kernel_matrix` evaluates the covariance between two sets of inputs using the radial basis function kernel, while `rbf_kernel` defines the pairwise covariance evaluation. These components collectively enable the formation of the matrices $K_{nm}$, $K_{mn}$, and $K_{mm}$, which are central to the inducing-point approximation framework.

The function `train_sparse_gp` implements the reduced-complexity training procedure. Instead of solving the full system involving the dense covariance matrix, it constructs the smaller matrix $K_{mm}$ and applies Cholesky factorization via `cholesky_spd` to ensure numerical stability. The Woodbury identity is then used to compute the approximate inverse action on the training targets, avoiding explicit inversion of the full matrix as implied by Equation (15.10.6). The resulting vector `alpha_approx` represents the compressed solution used for prediction, while the matrices `kmm_inv` and `c_inv` capture the structured components required for efficient evaluation.

Prediction is carried out in the function `predict_sparse_gp`, which evaluates both the predictive mean and variance. The predictive mean is formed using the approximate cross-covariance between the test point and training data, consistent with Equation (15.10.7). The variance is computed using a structured low-rank correction that reflects the uncertainty retained after approximation. Although simplified relative to a full Gaussian process, this formulation preserves the key probabilistic interpretation while significantly reducing computational cost.

The supporting linear algebra functions, including `cholesky_spd`, `forward_substitution`, and `backward_substitution_from_lower_transpose`, provide numerically stable solutions to symmetric positive definite systems. These functions replace direct matrix inversion with triangular solves, ensuring robustness in floating-point arithmetic and aligning with the numerical principles emphasized throughout the chapter.

The `main` function demonstrates the full workflow of scalable Gaussian process regression. It begins by generating training data from a synthetic simulator $g(x)$, representing an expensive engineering model. A subset of inducing points is selected using `select_evenly_spaced_inducing_points`, reflecting the idea of compressing the dataset into a smaller representative set. The model is then trained using `train_sparse_gp`, and predictions are evaluated over a test grid. The computed root-mean-square error provides a quantitative measure of approximation quality, while printed outputs illustrate both predictive accuracy and uncertainty. This workflow highlights the trade-off between computational efficiency and approximation fidelity inherent in inducing-point methods.

```rust
// Program 15.10.3
// Scalable Gaussian Process Regression with Inducing Points
//
// Problem statement:
// Implement a sparse Gaussian process surrogate for an expensive one-dimensional
// engineering simulator g(x). The program uses m << n inducing points, forms
// a low-rank Nyström-style approximation of the kernel matrix, applies the
// Woodbury identity to avoid factorizing the full n x n covariance matrix,
// and reports predictive means and variances on a test grid.
//
// This example is written in pure Rust so that `cargo run` works out of the box.

use std::f64::consts::PI;

// -----------------------------
// Basic dense matrix structure
// -----------------------------
#[derive(Clone, Debug)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }

    fn mul(&self, rhs: &Self) -> Self {
        assert_eq!(self.cols, rhs.rows, "Dimension mismatch in matrix multiply");
        let mut out = Self::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(i, k)];
                for j in 0..rhs.cols {
                    out[(i, j)] += a * rhs[(k, j)];
                }
            }
        }
        out
    }

    fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, x.len(), "Dimension mismatch in matrix-vector multiply");
        let mut out = vec![0.0; self.rows];
        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..self.cols {
                sum += self[(i, j)] * x[j];
            }
            out[i] = sum;
        }
        out
    }

    fn scale(&self, alpha: f64) -> Self {
        let mut out = self.clone();
        for v in &mut out.data {
            *v *= alpha;
        }
        out
    }

    fn add(&self, rhs: &Self) -> Self {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let mut out = Self::zeros(self.rows, self.cols);
        for i in 0..self.data.len() {
            out.data[i] = self.data[i] + rhs.data[i];
        }
        out
    }

    fn sub(&self, rhs: &Self) -> Self {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let mut out = Self::zeros(self.rows, self.cols);
        for i in 0..self.data.len() {
            out.data[i] = self.data[i] - rhs.data[i];
        }
        out
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.cols + index.1]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * self.cols + index.1]
    }
}

// -----------------------------
// Linear algebra helpers
// -----------------------------
fn dot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn cholesky_spd(a: &Matrix) -> Result<Matrix, String> {
    if a.rows != a.cols {
        return Err("Cholesky requires a square matrix".to_string());
    }

    let n = a.rows;
    let mut l = Matrix::zeros(n, n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[(i, j)];
            for k in 0..j {
                sum -= l[(i, k)] * l[(j, k)];
            }

            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "Matrix is not numerically SPD at diagonal {} with value {:.6e}",
                        i, sum
                    ));
                }
                l[(i, j)] = sum.sqrt();
            } else {
                l[(i, j)] = sum / l[(j, j)];
            }
        }
    }

    Ok(l)
}

fn forward_substitution(l: &Matrix, b: &[f64]) -> Vec<f64> {
    let n = l.rows;
    assert_eq!(l.cols, n);
    assert_eq!(b.len(), n);

    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[(i, j)] * y[j];
        }
        y[i] = sum / l[(i, i)];
    }
    y
}

fn backward_substitution_from_lower_transpose(l: &Matrix, y: &[f64]) -> Vec<f64> {
    let n = l.rows;
    assert_eq!(l.cols, n);
    assert_eq!(y.len(), n);

    let mut x = vec![0.0; n];
    for idx in 0..n {
        let i = n - 1 - idx;
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[(j, i)] * x[j];
        }
        x[i] = sum / l[(i, i)];
    }
    x
}

fn solve_spd(chol_l: &Matrix, b: &[f64]) -> Vec<f64> {
    let y = forward_substitution(chol_l, b);
    backward_substitution_from_lower_transpose(chol_l, &y)
}

fn solve_spd_multiple_rhs(chol_l: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(chol_l.rows, chol_l.cols);
    assert_eq!(chol_l.rows, b.rows);

    let mut out = Matrix::zeros(b.rows, b.cols);
    for j in 0..b.cols {
        let mut rhs = vec![0.0; b.rows];
        for i in 0..b.rows {
            rhs[i] = b[(i, j)];
        }
        let sol = solve_spd(chol_l, &rhs);
        for i in 0..b.rows {
            out[(i, j)] = sol[i];
        }
    }
    out
}

fn inverse_from_cholesky(chol_l: &Matrix) -> Matrix {
    let n = chol_l.rows;
    let eye = Matrix::identity(n);
    solve_spd_multiple_rhs(chol_l, &eye)
}

// -----------------------------
// Kernel and data generation
// -----------------------------
fn rbf_kernel(x: f64, z: f64, signal_variance: f64, length_scale: f64) -> f64 {
    let r = x - z;
    signal_variance * (-0.5 * r * r / (length_scale * length_scale)).exp()
}

fn build_kernel_matrix(
    xs: &[f64],
    zs: &[f64],
    signal_variance: f64,
    length_scale: f64,
) -> Matrix {
    let mut k = Matrix::zeros(xs.len(), zs.len());
    for i in 0..xs.len() {
        for j in 0..zs.len() {
            k[(i, j)] = rbf_kernel(xs[i], zs[j], signal_variance, length_scale);
        }
    }
    k
}

// A smooth surrogate target representing an expensive engineering response.
fn simulator_g(x: f64) -> f64 {
    let smooth = (2.0 * PI * x).sin() * (1.0 + 0.3 * (3.0 * PI * x).cos());
    let localized = 0.7 * (-(18.0 * (x - 0.72) * (x - 0.72))).exp();
    let trend = 0.25 * x * x;
    smooth + localized + trend
}

fn deterministic_noise(i: usize) -> f64 {
    // Small deterministic perturbation so the example remains reproducible.
    let t = (i as f64 + 1.0) * 1.618_033_988_75;
    0.03 * t.sin() + 0.01 * (2.7 * t).cos()
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    assert!(n >= 2);
    let h = (b - a) / ((n - 1) as f64);
    (0..n).map(|i| a + (i as f64) * h).collect()
}

fn select_evenly_spaced_inducing_points(xs: &[f64], m: usize) -> Vec<f64> {
    assert!(m >= 2 && m <= xs.len());
    let n = xs.len();
    let mut z = Vec::with_capacity(m);
    for k in 0..m {
        let idx = k * (n - 1) / (m - 1);
        z.push(xs[idx]);
    }
    z
}

// -----------------------------------------------
// Sparse GP training using inducing points + Woodbury
// -----------------------------------------------
#[derive(Debug)]
struct SparseGpModel {
    inducing_points: Vec<f64>,
    signal_variance: f64,
    length_scale: f64,
    alpha_approx: Vec<f64>,
    kmm_inv: Matrix,
    c_inv: Matrix,
}

fn train_sparse_gp(
    x_train: &[f64],
    y_train: &[f64],
    inducing_points: Vec<f64>,
    signal_variance: f64,
    length_scale: f64,
    noise_variance: f64,
    jitter: f64,
) -> Result<SparseGpModel, String> {
    let n = x_train.len();
    let m = inducing_points.len();

    if y_train.len() != n {
        return Err("x_train and y_train must have the same length".to_string());
    }
    if m == 0 || m > n {
        return Err("Invalid number of inducing points".to_string());
    }
    if noise_variance <= 0.0 {
        return Err("noise_variance must be positive".to_string());
    }

    let knm = build_kernel_matrix(x_train, &inducing_points, signal_variance, length_scale);
    let kmn = knm.transpose();

    let mut kmm = build_kernel_matrix(
        &inducing_points,
        &inducing_points,
        signal_variance,
        length_scale,
    );
    for i in 0..m {
        kmm[(i, i)] += jitter;
    }

    let chol_kmm = cholesky_spd(&kmm)?;
    let kmm_inv = inverse_from_cholesky(&chol_kmm);

    // Compute C = K_mm + sigma^{-2} K_mn K_nm
    let kmn_knm = kmn.mul(&knm);
    let c = kmm.add(&kmn_knm.scale(1.0 / noise_variance));
    let chol_c = cholesky_spd(&c)?;
    let c_inv = inverse_from_cholesky(&chol_c);

    // Woodbury formula:
    // (sigma^2 I + K_nm K_mm^{-1} K_mn)^(-1) y
    // = sigma^{-2} y - sigma^{-4} K_nm C^{-1} K_mn y
    let kmn_y = kmn.mul_vec(y_train);
    let c_inv_kmn_y = c_inv.mul_vec(&kmn_y);

    let correction = knm.mul_vec(&c_inv_kmn_y);
    let mut alpha_approx = vec![0.0; n];
    for i in 0..n {
        alpha_approx[i] =
            y_train[i] / noise_variance - correction[i] / (noise_variance * noise_variance);
    }

    Ok(SparseGpModel {
        inducing_points,
        signal_variance,
        length_scale,
        alpha_approx,
        kmm_inv,
        c_inv,
    })
}

// ----------------------------------------------------
// Prediction for sparse GP with low-rank covariance
// ----------------------------------------------------
fn predict_sparse_gp(
    model: &SparseGpModel,
    x_train: &[f64],
    x_star: f64,
) -> Result<(f64, f64), String> {
    let k_xm = build_kernel_matrix(
        &[x_star],
        &model.inducing_points,
        model.signal_variance,
        model.length_scale,
    );
    let k_mx = k_xm.transpose();

    // Approximate cross-covariance q_*n = k_*m K_mm^{-1} K_mn
    let knm = build_kernel_matrix(
        x_train,
        &model.inducing_points,
        model.signal_variance,
        model.length_scale,
    );
    let q_star_n = k_xm.mul(&model.kmm_inv).mul(&knm.transpose()); // 1 x n

    let mean = dot(&q_star_n.data, &model.alpha_approx);

    // DTC-style approximate predictive variance:
    // var[f_*] ≈ k_** - k_*m (K_mm^{-1} - C^{-1}) k_m*
    let temp_left = model.kmm_inv.sub(&model.c_inv);
    let middle = k_xm.mul(&temp_left).mul(&k_mx); // 1 x 1

    let k_ss = rbf_kernel(
        x_star,
        x_star,
        model.signal_variance,
        model.length_scale,
    );
    let variance = (k_ss - middle[(0, 0)]).max(1.0e-12);

    Ok((mean, variance))
}

// ----------------------------------------------------
// Diagnostic helpers
// ----------------------------------------------------
fn compute_rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    let mse = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f64>()
        / (y_true.len() as f64);
    mse.sqrt()
}

fn print_training_sample(name: &str, x: &[f64], y: &[f64], count: usize) {
    println!("{name}");
    println!("{}", "=".repeat(name.len()));
    let k = count.min(x.len()).min(y.len());
    for i in 0..k {
        println!("  idx = {:>3}, x = {:>.6}, y = {:>.10}", i, x[i], y[i]);
    }
    println!();
}

fn print_inducing_points(name: &str, z: &[f64]) {
    println!("{name}");
    println!("{}", "=".repeat(name.len()));
    for (i, value) in z.iter().enumerate() {
        println!("  idx = {:>3}, z = {:>.10}", i, value);
    }
    println!();
}

fn main() -> Result<(), String> {
    // Training data
    let n_train = 160usize;
    let x_train = linspace(0.0, 1.0, n_train);

    let mut y_train = vec![0.0; n_train];
    for i in 0..n_train {
        y_train[i] = simulator_g(x_train[i]) + deterministic_noise(i);
    }

    // Inducing-point configuration
    let m_inducing = 18usize;
    let z = select_evenly_spaced_inducing_points(&x_train, m_inducing);

    // Kernel hyperparameters
    let signal_variance = 1.0;
    let length_scale = 0.11;
    let noise_variance = 0.02_f64.powi(2);
    let jitter = 1.0e-8;

    let model = train_sparse_gp(
        &x_train,
        &y_train,
        z.clone(),
        signal_variance,
        length_scale,
        noise_variance,
        jitter,
    )?;

    // Prediction grid
    let n_test = 201usize;
    let x_test = linspace(0.0, 1.0, n_test);

    let mut y_true = vec![0.0; n_test];
    let mut y_pred = vec![0.0; n_test];
    let mut y_std = vec![0.0; n_test];

    for i in 0..n_test {
        y_true[i] = simulator_g(x_test[i]);
        let (mu, var) = predict_sparse_gp(&model, &x_train, x_test[i])?;
        y_pred[i] = mu;
        y_std[i] = var.sqrt();
    }

    let rmse = compute_rmse(&y_true, &y_pred);

    println!("Scalable Gaussian Process Regression with Inducing Points");
    println!("=========================================================");
    println!();
    println!("Training set size n                 = {}", n_train);
    println!("Number of inducing points m         = {}", m_inducing);
    println!("Signal variance                     = {:>.6}", signal_variance);
    println!("Length scale                        = {:>.6}", length_scale);
    println!("Noise standard deviation            = {:>.6}", noise_variance.sqrt());
    println!("Approximate training complexity     = O(n m^2 + m^3)");
    println!("Storage complexity                  = O(n m + m^2)");
    println!("Test-grid RMSE against true g(x)    = {:>.8}", rmse);
    println!();

    print_training_sample("Representative Training Samples", &x_train, &y_train, 8);
    print_inducing_points("Selected Inducing Points", &z);

    println!("Representative Predictions");
    println!("==========================");
    for &idx in &[0usize, 25, 50, 75, 100, 125, 150, 175, 200] {
        println!(
            "x = {:>.6}, true g(x) = {:>.10}, pred mean = {:>.10}, pred std = {:>.10}",
            x_test[idx], y_true[idx], y_pred[idx], y_std[idx]
        );
    }
    println!();

    println!("Interpretation");
    println!("==============");
    println!("This program replaces the full dense GP training solve by an inducing-point");
    println!("approximation. The dominant cubic work is performed on the m x m covariance");
    println!("matrix of the inducing points rather than on the full n x n kernel matrix.");
    println!("As a result, training and prediction remain probabilistic but scale much more");
    println!("favorably for surrogate modeling workflows in engineering simulation.");
    println!();

    Ok(())
}
```

Program 15.10.3 demonstrates a practical implementation of scalable Gaussian process regression by exploiting low-rank structure through inducing points. This approach reflects the central computational challenge discussed in Section 15.10.5: reducing the cubic complexity of Gaussian process inference while maintaining meaningful probabilistic predictions.

The use of inducing points illustrates how the covariance structure of the data can be compressed into a smaller representation, enabling efficient training and prediction even for moderately large datasets. The results show that the surrogate model captures the dominant behavior of the underlying function while providing uncertainty estimates, although with some loss of accuracy compared to a full Gaussian process.

The modular structure of the implementation allows for straightforward extensions. More advanced variants, such as FITC or variational inducing-point methods, can improve uncertainty calibration, while alternative kernel functions or adaptive inducing-point selection strategies can enhance approximation quality. These extensions provide a natural pathway toward more sophisticated scalable Gaussian process models suitable for high-dimensional and large-scale applications.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/2Ats0XqUYC7ROfzatK8D.2","tags":[]}

# 15.11. Conclusion

This chapter has developed a unified framework for modeling data, progressing from the formulation of residuals and objective functions through linear and nonlinear least squares, straight-line fitting with errors in one or both coordinates, general basis-function regression, confidence limit construction, robust estimation, Markov chain Monte Carlo sampling, and Gaussian process regression. The central theme throughout is that data modeling is fundamentally an optimization problem whose formulation, solution, and interpretation depend on the interplay between probabilistic assumptions, geometric structure, and numerically stable computation. Each section has linked the mathematical derivation of estimators to their algorithmic realization in Rust, emphasizing that reliable inference requires not only correct formulas but also careful attention to conditioning, factorization strategy, and finite-precision arithmetic.

## 15.11.1. Key Takeaways

- The passage from raw observations to a quantitative model begins with defining a parameterized family $y(x; \theta)$, $r_i(\theta) = y_i - y(x_i; \theta)$, and an objective function $\Phi(\theta) = \sum \rho_i(r_i(\theta))$ that measures the discrepancy between model and data. The most important special case is least squares with $\rho_i(t) = t^2$. The canonical linear model $y = A\beta + \varepsilon$ casts parameter estimation as a problem in numerical linear algebra, where the least-squares solution $A\hat{\beta}$ is the orthogonal projection of the data vector $y$ onto the column space $\mathcal{C}(A)$, characterized by the orthogonality condition $A^\top(y - A\hat{\beta}) = 0$. Forming the normal matrix $A^\top A$ explicitly can square the condition number and degrade accuracy, so stable factorizations such as QR and SVD should be preferred.
- Under independent Gaussian noise $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$, maximizing the likelihood is equivalent to minimizing the sum of squared residuals, establishing least squares as the maximum likelihood estimator (MLE). When measurement variances differ across observations, the MLE minimizes the weighted chi-square $\sum (y_i - y(x_i; \theta))^2 / \sigma_i^2$. For correlated noise with covariance $\Sigma$, the objective generalizes to the Mahalanobis norm $(y - f(\theta))^\top \Sigma^{-1} (y - f(\theta))$, leading to generalized least squares. The covariance of the estimator is $\mathrm{Cov}(\hat{\beta}) = \sigma^2 (A^\top A)^{-1}$ in the homoscedastic case, with the noise variance estimated from residuals as $\hat{\sigma}^2 = \|y - A\hat{\beta}\|_2^2 / (N - M)$.
- The weighted straight-line model $y(x) = a + bx$ with chi-square objective $\chi^2(a,b) = \sum (y_i - a - bx_i)^2 / \sigma_i^2$ admits an explicit solution through weighted sums $S$, $S_x$, $S_y$, $S_{xx}$, $S_{xy}$ and the determinant $\Delta = S S_{xx} - S_x^2$, yielding $b = (S S_{xy} - S_x S_y) / \Delta$ and $a = (S_{xx} S_y - S_x S_{xy}) / \Delta$. Computing the weighted mean $\bar{x}_w = S_x / S$ and reformulating in centered variables $t_i = (x_i - \bar{x}_w) / \sigma_i$ avoids catastrophic cancellation in $\Delta$ by replacing the subtraction of large, nearly equal quantities with sums of squared deviations. Parameter uncertainties follow from the inverse of $A^\top W A$, while prediction variance at $x_0$ combines $\mathrm{Var}(a)$, $x_0^2 \mathrm{Var}(b)$, and $2 x_0 \mathrm{Cov}(a,b)$.
- When both coordinates carry uncertainty with variances $\sigma_{x_i}^2$ and $\sigma_{y_i}^2$, the effective residual variance becomes $\sigma_{y_i}^2 + b^2 \sigma_{x_i}^2$, making the chi-square objective nonlinear in the slope even for a straight-line model. The normal-form parameterization $n(\theta)^\top p = c$ with unit normal $n(\theta) = (\cos\theta, \sin\theta)^\top$ treats both coordinates symmetrically, avoids slope singularities near vertical lines, and yields an orthogonal-distance objective $\chi^2(\theta,c) = \sum (n(\theta)^\top p_i - c)^2 / (n(\theta)^\top \Sigma_i n(\theta))$. Eliminating the offset $c$ analytically for each trial angle reduces the problem to one-dimensional minimization, solvable by bracketing methods at $O(IN)$ total cost with $O(1)$ memory beyond the data. Uncertainty is estimated via $\Delta\chi^2 = 1$ level sets rather than local quadratic approximation.
- General linear-in-parameter models $y(x) = \sum_{k=0}^{M-1} a_k X_k(x)$ with arbitrary basis functions are cast into matrix form through the weighted design matrix $A_{ik} = X_k(x_i) / \sigma_i$ and data vector $b_i = y_i / \sigma_i$, reducing the weighted problem to $\min_a \|Aa - b\|_2^2$. The normal equations $A^\top A a = A^\top b$ provide a theoretical characterization, while QR factorization at cost $O(NM^2)$ is preferred for full-rank problems and SVD provides the most robust treatment of rank deficiency through pseudoinverse truncation. The covariance matrix $C = (A^\top A)^{-1}$ encodes parameter uncertainty, with diagonal entries giving individual variances and off-diagonal entries capturing correlations. For large sparse systems, iterative methods such as CGLS operate using only matrix-vector products with $A$ and $A^\top$, preserving sparsity and avoiding formation of the dense normal matrix.
- Nonlinear models $y(x_i; a)$ where parameters enter nonlinearly require iterative solution. The Jacobian $J_{ik}(a) = \partial r_i(a) / \partial a_k$ captures the local sensitivity of residuals to parameter changes, giving gradient $\nabla \chi^2 = 2 J^\top r$ and approximate Hessian $\nabla^2 \chi^2 \approx 2 J^\top J$. The Gauss-Newton method linearizes the residual as $r(a + \delta) \approx r(a) + J(a)\delta$ and solves the linear least-squares subproblem $(J^\top J)\delta = -J^\top r$ at each iteration, reducing nonlinear optimization to a sequence of linear solves. The Levenberg-Marquardt method stabilizes this iteration by adding a damping term $(J^\top J + \lambda D)\delta = -J^\top r$, interpolating between steepest descent for large $\lambda$ and Gauss-Newton for small $\lambda$, with $\lambda$ adjusted adaptively based on whether trial steps reduce $\chi^2$.
- Confidence limits quantify how parameter uncertainty propagates from data variability. The Wald approximation uses the local curvature of the log-likelihood to construct symmetric intervals $\hat{\theta}_j \pm z_{1-\alpha/2} \sqrt{\mathrm{Var}(\hat{\theta}_j)}$ with $\mathrm{Cov}(\hat{\theta}) \approx \sigma^2 (J^\top J)^{-1}$, but relies on a quadratic approximation that can be inaccurate for nonlinear models. Profile likelihood methods avoid this limitation by evaluating the likelihood-ratio statistic $\Lambda(\theta_j) = 2[\ell(\hat{\theta}) - \ell_p(\theta_j)]$ and finding where it crosses the $\chi^2_{1,1-\alpha}$ threshold, producing potentially asymmetric intervals that reflect the true objective shape. Bootstrap methods approximate the sampling distribution empirically by refitting the model to $B$ resampled datasets, with computational cost $O(B \cdot C_{\mathrm{fit}})$.
- Robust estimation addresses the sensitivity of least squares to outliers and non-Gaussian noise through M-estimation, which replaces the quadratic loss with a general function $\rho(r_i / s)$ whose derivative $\psi = \rho'$ controls the influence of each observation. The resulting estimating equations $\sum \psi(r_i / s) \nabla r_i = 0$ are solved by iteratively reweighted least squares (IRLS), where weights $w_i = \psi(u_i) / u_i$ transform the nonlinear problem into a sequence of weighted least-squares solves $(X^\top W^{(t)} X)\beta^{(t+1)} = X^\top W^{(t)} y$. The Huber loss bounds the influence function for large residuals while the Tukey bisquare is redescending, assigning zero weight to sufficiently extreme observations. Data-adaptive tuning of the threshold parameter via cross-validation and robust bootstrap confidence intervals integrate point estimation with uncertainty quantification.
- Bayesian inference represents parameter uncertainty through the posterior distribution $p(\theta \mid D) \propto p(D \mid \theta) p(\theta)$, where the normalization constant $p(D) = \int p(D \mid \theta) p(\theta) d\theta$ is typically intractable. Markov chain Monte Carlo constructs a chain $\theta^{(0)} \to \theta^{(1)} \to \cdots$ whose stationary distribution is the posterior, with the Metropolis-Hastings algorithm accepting proposed moves with probability $\alpha = \min(1, [p(D \mid \theta') p(\theta') q(\theta \mid \theta')] / [p(D \mid \theta) p(\theta) q(\theta' \mid \theta)])$, where the normalization constant cancels. The effective sample size $N_{\mathrm{eff}} = K / (1 + 2\sum_{k=1}^{\infty} \rho_k)$ quantifies the information content of correlated samples. Advanced methods including Hamiltonian Monte Carlo with Hamiltonian $H(\theta, p) = -\log p(\theta \mid D) + p^\top M^{-1} p / 2$ and the Metropolis-adjusted Langevin algorithm with drift $\theta' = \theta + (\epsilon^2/2) \nabla \log p(\theta \mid D) + \epsilon \eta$ use gradient information to reduce random-walk behavior.
- Gaussian process regression places a prior directly on an unknown function $f(\cdot) \sim \mathcal{GP}(m(\cdot), k(\cdot,\cdot))$, so that any finite collection of function values follows a multivariate normal distribution $f_X \sim \mathcal{N}(m_X, K_{XX})$. With noisy observations $y_i = f(x_i) + \varepsilon_i$, the posterior predictive mean is $\mu_* = m_* + K_{*X}(K_{XX} + \sigma_n^2 I)^{-1}(y - m_X)$ and the covariance is $\Sigma_* = K_{**} - K_{*X}(K_{XX} + \sigma_n^2 I)^{-1} K_{X*}$, computed via Cholesky factorization $K_{XX} + \sigma_n^2 I = LL^\top$ for numerical stability. Hyperparameters are estimated by maximizing the log marginal likelihood, which balances a data-fit quadratic form against a complexity penalty through the log-determinant. Exact training costs $O(n^3)$ with $O(n^2)$ storage, motivating inducing-point approximations that project the covariance onto $m \ll n$ representative inputs and reduce complexity to $O(nm^2 + m^3)$.

## 15.11.2. Advice for Beginners

- Data modeling is one of the most important applications of numerical computing because it connects mathematical models to real observations. As you study this chapter, focus first on understanding the relationship between data, models, residuals, and uncertainty. Every method introduced here attempts to answer the same fundamental question: given imperfect observations, what model best explains the data and how certain can we be about the resulting conclusions?
- Begin with linear least squares in Sections 15.2 through 15.5. Learn how residuals measure disagreement between a model and observations, why minimizing the sum of squared residuals produces useful estimates, and how least squares emerges naturally from the assumption of Gaussian measurement noise. Before moving to advanced topics, become comfortable constructing design matrices, interpreting residuals, and understanding the geometric meaning of orthogonal projection. These concepts form the foundation of nearly every modern regression algorithm.
- When implementing least-squares methods, avoid relying solely on the normal equations. Although they are mathematically simple, they can become numerically unstable when the design matrix is poorly conditioned. Learn to use QR factorization and singular value decomposition whenever possible. Understanding why conditioning matters is often more important than memorizing formulas for parameter estimation.
- The straight-line fitting examples provide an excellent starting point because they allow you to visualize every aspect of the problem. Experiment with weighted observations, investigate the effects of measurement uncertainty, and observe how parameter estimates change when data points are added or removed. These exercises build intuition that will later transfer to more complicated models.
- Before studying nonlinear least squares in Section 15.6, ensure that you understand Jacobians, gradients, and linearization. Methods such as Gauss-Newton and Levenberg-Marquardt repeatedly solve linearized least-squares problems, so a strong understanding of linear algebra is essential. Practice deriving Jacobians analytically and compare them with finite-difference approximations to develop confidence in your implementations.
- Sections 15.7 and 15.8 introduce an equally important concept: uncertainty. Many beginners focus exclusively on obtaining parameter estimates while ignoring confidence intervals and error bounds. In practice, uncertainty estimates are often more valuable than the fitted parameters themselves. Learn how covariance matrices, profile likelihoods, and bootstrap methods quantify the reliability of a model. Similarly, study robust estimation carefully because real datasets frequently contain outliers, corrupted measurements, and violations of ideal statistical assumptions.
- The Bayesian methods in Section 15.9 represent a different way of thinking about inference. Instead of producing a single estimate, Bayesian methods characterize uncertainty through entire probability distributions. Do not be discouraged if Markov chain Monte Carlo appears abstract initially. Focus first on the basic ideas of posterior distributions, sampling, and convergence diagnostics before exploring advanced methods such as Hamiltonian Monte Carlo.
- Gaussian process regression in Section 15.10 combines ideas from probability theory, linear algebra, optimization, and numerical analysis. Pay particular attention to covariance matrices, kernel functions, and Cholesky factorization. Understanding these concepts will provide valuable preparation for machine learning, surrogate modeling, uncertainty quantification, and scientific data analysis.
- For Rust implementations, emphasize numerical reliability. Learn to use libraries such as `nalgebra`, `ndarray`, and sparse linear algebra packages for matrix computations. Avoid explicit matrix inversion whenever possible, prefer factorization-based solvers, and monitor conditioning and residual norms throughout your calculations. Good numerical software is not simply correct mathematically; it must also be stable, efficient, and robust under finite-precision arithmetic.
- Finally, remember that successful modeling requires judgment as well as computation. No algorithm can compensate for a poor model, inadequate data, or unrealistic assumptions. Always examine residuals, assess uncertainty, test assumptions, and compare alternative models. The most valuable skill developed in this chapter is not fitting a model, but learning how to evaluate whether that model can be trusted.

## 15.11.3. Further Learning with GenAI

To deepen your understanding of data modeling and inference in Rust, consider using the following GenAI prompts:

 1. Write a Rust program that demonstrates the complete data-modeling pipeline for a straight-line model. Define a dataset of 10 observations with known standard deviations, compute the weighted sums $S$, $S_x$, $S_y$, $S_{xx}$, $S_{xy}$, form the determinant $\Delta$, and solve for the slope and intercept using the explicit formulas. Then implement the numerically stable centered formulation using the weighted mean $\bar{x}_w$ and centered variables $t_i = (x_i - \bar{x}_w) / \sigma_i$. Compare the two solutions on a dataset where the $x$-values are large and closely clustered, and report the difference in the computed parameters. Compute the parameter covariance matrix from the inverse of $A^\top W A$ and report standard errors and a 95% confidence interval for the slope.
 2. Implement a Rust program that solves a weighted linear least-squares problem using both the normal equations with Cholesky factorization and Householder QR factorization. Construct a polynomial design matrix of degree 5 on a narrow interval to induce moderate ill-conditioning, generate synthetic observations with heteroscedastic noise, and compute the weighted system by scaling rows by $1/\sigma_i$. Report the estimated parameters, residual norms, and the infinity-norm difference between the two solutions. Estimate the condition number of $A^\top W A$ and discuss how conditioning affects the reliability of the normal-equations approach compared to the QR-based method.
 3. Build a Rust program that fits a straight-line model when both coordinates have measurement uncertainty. Implement the normal-form parameterization $n(\theta)^\top p = c$ with unit normal $n(\theta) = (\cos\theta, \sin\theta)^\top$ and the orthogonal-distance objective where each residual is weighted by the directional variance $n(\theta)^\top \Sigma_i n(\theta)$. Eliminate the offset $c$ analytically for each trial angle, and minimize the reduced objective over $\theta$ using a golden-section search. Compare the result with the slope-intercept formulation using the modified chi-square $\sum (y_i - a - bx_i)^2 / (\sigma_{y_i}^2 + b^2 \sigma_{x_i}^2)$ and verify that the two parameterizations yield consistent fitted lines.
 4. Write a Rust program that fits a general linear model $y(x) = \sum_{k=0}^{M-1} a_k X_k(x)$ with basis functions $X_0(x) = 1$, $X_1(x) = x$, $X_2(x) = x^2$, $X_3(x) = \sin(2x)$ to 30 weighted observations. Build the weighted design matrix, solve the system using Householder QR, and compute the covariance matrix $C = (A^\top A)^{-1}$. Report diagonal variances, standard errors, and the correlation matrix derived from $C$. Then implement the same problem using a sparse CGLS solver that operates only through matrix-vector products with $A$ and $A^\top$, and verify that the iterative solution agrees with the direct QR result to within a specified tolerance.
 5. Implement a Rust program that performs nonlinear least-squares fitting of the model $y(x; a) = a_0 \exp(a_1 x) + a_2$ using both the Gauss-Newton method and the Levenberg-Marquardt method. For each iteration, compute the residual vector, construct the Jacobian with analytic partial derivatives, form the linearized system $(J^\top J)\delta = -J^\top r$ or its damped variant $(J^\top J + \lambda D)\delta = -J^\top r$, and update the parameters. Record the iteration history including $\chi^2$, step norm, and damping parameter. Compare convergence behavior from a poor initial guess, demonstrating that Levenberg-Marquardt recovers convergence where Gauss-Newton may fail.
 6. Build a Rust program that computes Wald confidence intervals and profile likelihood confidence intervals for a nonlinear model. Fit the model $y(x; \theta) = a(1 - \exp(-bx)) + c$ using a damped Gauss-Newton method, construct the Jacobian at the solution, and compute the covariance approximation $\mathrm{Cov}(\hat{\theta}) \approx \hat{\sigma}^2 (J^\top J)^{-1}$ via QR factorization. Report Wald intervals for all three parameters. Then profile the parameter $b$ by re-optimizing $a$ and $c$ for each fixed value of $b$, evaluating the likelihood-ratio statistic, and locating the $\Delta\chi^2 = \chi^2_{1,0.95}$ crossings. Compare the Wald and profile intervals and discuss whether asymmetry is present.
 7. Write a Rust program that implements robust linear regression using iteratively reweighted least squares with three loss functions: ordinary least squares, Huber, and Tukey bisquare. Generate a dataset of 25 observations from a known linear model with 3 injected outliers. For each method, iterate until convergence by computing residuals, estimating a robust scale via median absolute deviation, forming normalized residuals $u_i = r_i / s$, updating weights $w_i = \psi(u_i) / u_i$, and solving the weighted normal equations $(X^\top W X)\beta = X^\top W y$ using Cholesky factorization. Report fitted parameters, residual statistics, and a pointwise comparison of weights across the three estimators.
 8. Implement a Rust program that performs Bayesian inference for a linear regression model using the Metropolis-Hastings algorithm with a symmetric Gaussian random-walk proposal. Generate synthetic data from known parameters, define Gaussian priors, and run 30000 iterations. Discard a burn-in period, then compute posterior means, standard deviations, and 95% credible intervals for each parameter. Compute the autocorrelation function for the first 15 lags and estimate the effective sample size using $N_{\mathrm{eff}} = K / (1 + 2\sum \rho_k)$. Run three chains from dispersed initial values and compare posterior summaries across chains to assess convergence.
 9. Build a Rust program that implements exact Gaussian process regression for one-dimensional inputs. Use a squared-exponential kernel with specified amplitude and length scale, generate training data from a known function with additive Gaussian noise, and compute the posterior predictive mean and variance at a test grid. All linear algebra must use Cholesky factorization $K_{XX} + \sigma_n^2 I = LL^\top$ with triangular solves rather than explicit matrix inversion. Evaluate the log marginal likelihood and its three components (quadratic form, log-determinant, normalization constant). Perform hyperparameter optimization by grid search followed by local refinement, and compare predictions before and after optimization.
10. Build a comprehensive Rust program that combines techniques from multiple sections into a single modeling pipeline. Generate 40 paired observations from a nonlinear generative model with heteroscedastic noise and 2 outliers. Fit a polynomial basis-function model using weighted QR least squares, then refit using IRLS with Huber weights to reduce outlier influence. Compute Wald confidence intervals from the Jacobian-based covariance and bootstrap confidence intervals from 500 resamples. Implement a Metropolis-Hastings sampler with 10000 post-burn-in samples targeting the posterior under a Gaussian likelihood and compare the posterior credible intervals with the frequentist confidence intervals.

By engaging with these prompts, you will gain a deeper understanding of how least-squares estimation, nonlinear optimization, confidence limit construction, robust methods, Bayesian sampling, and Gaussian process inference form an integrated framework for data modeling, and how the Rust implementations ensure numerical stability, type safety, and reproducibility across all stages of the analysis pipeline.

## 15.11.4. Homework Exercises

To reinforce your learning, complete the following exercises:

 1. Implement a Rust program that fits a weighted straight line to the data $(x, y, \sigma) =$ {(0.0, 1.05, 0.20), (1.0, 2.10, 0.15), (2.0, 2.95, 0.20), (3.0, 4.05, 0.25), (4.0, 5.20, 0.20), (5.0, 6.10, 0.30)} using both the explicit determinant-based formulas and the numerically stable centered formulation. Compute the weighted sums $S$, $S_x$, $S_y$, $S_{xx}$, $S_{xy}$, the determinant $\Delta$, the weighted mean $\bar{x}_w$, and the centered sum $S_{tt} = \sum (x_i - \bar{x}_w)^2 / \sigma_i^2$. Verify that both methods yield the same slope and intercept to within $10^{-12}$. Compute the chi-square misfit $\chi^2(a,b) = \sum [(y_i - a - bx_i)/\sigma_i]^2$, the parameter covariance matrix from $(A^\top W A)^{-1}$, and 95% confidence intervals for both parameters.
 2. Implement a Rust program that solves the weighted least-squares problem $\min_\beta (y - A\beta)^\top W (y - A\beta)$ with $W = \mathrm{diag}(1/\sigma_i^2)$ for a polynomial design matrix of degree 6 on the narrow interval $[0.92, 1.08]$ with 24 observations. Compare solutions obtained by the normal equations with Gaussian elimination and by Householder QR factorization applied to the whitened system $(\tilde{A}, \tilde{y})$ where $\tilde{A}_{ij} = A_{ij}/\sigma_i$ and $\tilde{y}_i = y_i/\sigma_i$. Report the estimated condition number of $A^\top W A$, the weighted residual norms from both methods, and the infinity-norm difference between the two parameter estimates. Verify that the QR solution has smaller or equal parameter error relative to the known true coefficients.
 3. Implement a Rust program that fits a straight line to 6 observations with uncertainty in both coordinates, including correlated measurement errors specified by $2 \times 2$ covariance matrices $\Sigma_i$. Use the normal-form parameterization $n(\theta)^\top p = c$ and minimize the weighted orthogonal-distance objective $\chi^2(\theta,c) = \sum (n(\theta)^\top p_i - c)^2 / (n(\theta)^\top \Sigma_i n(\theta))$ by eliminating the offset $c$ analytically and performing a golden-section search over $\theta \in [0, \pi)$. Report the optimal angle $\theta$, offset $c$, chi-square minimum, and the equivalent slope-intercept parameters. Compute orthogonal residuals and their directional variances for each observation.
 4. Implement a Rust program that fits a general linear model with basis functions $X_0(x) = 1$, $X_1(x) = x$, $X_2(x) = x^2$, $X_3(x) = \sin(2x)$ to 28 heteroscedastic observations spanning $[-1.5, 1.5]$. Build the weighted design matrix, solve using both Cholesky-based normal equations and Householder QR, and verify that the solutions agree to within $10^{-10}$. Compute the covariance matrix $C = (A^\top A)^{-1}$, extract diagonal variances, and report the condition number of $A^\top A$. Evaluate the residual norm $\|Aa - b\|_2$ and the infinity-norm error relative to known true coefficients for both methods.
 5. Implement a Rust program that fits the nonlinear model $y(x; a) = a_0 \exp(a_1 x) + a_2$ to 9 weighted observations using the Levenberg-Marquardt method. Start from the initial guess $(a_0, a_1, a_2) = (1.2, -1.2, 1.8)$ and iterate with adaptive damping: reduce $\lambda$ by a factor of 10 when a step is accepted and increase it by a factor of 10 when rejected. At each iteration, compute the residual vector, construct the Jacobian analytically, form $(J^\top J + \lambda \mathrm{diag}(J^\top J))\delta = -J^\top r$, and solve for the update. Record the full iteration history including parameters, $\chi^2$, $\lambda$, step norm, and acceptance status. Report pointwise weighted residuals and verify that $\chi^2$ decreases monotonically over accepted steps.
 6. Implement a Rust program that computes both Wald and profile likelihood 95% confidence intervals for the parameter $b$ in the nonlinear model $y(x; \theta) = a(1 - \exp(-bx)) + c$. Fit the model to 30 synthetic observations using a damped Gauss-Newton method, compute the Jacobian at the solution, and form the Wald covariance $\hat{\sigma}^2 (J^\top J)^{-1}$ via QR factorization. For the profile likelihood, fix $b$ at 81 grid points spanning $[0.7 \hat{b}, 1.35 \hat{b}]$, re-optimize $a$ and $c$ by linear least squares for each fixed $b$, and compute the likelihood-ratio statistic $\Lambda(b) = (\mathrm{RSS}_{\mathrm{profile}}(b) - \mathrm{RSS}_{\min}) / \hat{\sigma}^2$. Locate the profile interval by interpolating where $\Lambda(b) = 3.841$ and compare with the Wald interval.
 7. Implement a Rust program that performs robust linear regression on a contaminated dataset of 21 observations using IRLS with Huber weights ($c = 1.345$). The dataset follows the model $y = 2 + 1.5x$ with mild noise plus 3 injected outliers. Start from the ordinary least-squares solution, then iterate: compute residuals, estimate a robust scale via the median absolute deviation with consistency factor 1.4826, form normalized residuals $u_i = r_i / s$, compute Huber weights $w_i = \min(1, c/|u_i|)$, and solve $(X^\top W X)\beta = X^\top W y$ using Cholesky factorization. Iterate until the maximum absolute parameter change is below $10^{-10}$. Compare the OLS and robust parameter estimates and report which observations are downweighted.
 8. Implement a Rust program that performs Bayesian linear regression using the Metropolis-Hastings algorithm with a symmetric Gaussian random-walk proposal. Use 30 synthetic observations from the model $y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$ with known $\sigma$, Gaussian priors $\beta_j \sim \mathcal{N}(0, 5^2)$, and proposal standard deviations of 0.12 and 0.10 for $\beta_0$ and $\beta_1$. Run 25000 iterations, discard 5000 as burn-in, and compute posterior means and 95% credible intervals. Compute autocorrelations at lags 1 through 15 and the effective sample size $N_{\mathrm{eff}} = K / (1 + 2 \sum \rho_k)$ for each parameter. Run three chains from dispersed initial values $\{(0,0), (3,-3), (-2,2)\}$ and verify that posterior means agree across chains.
 9. Implement a Rust program that performs exact Gaussian process regression with a squared-exponential kernel on 9 training points sampled from $f(x) = \sin(2\pi x) + 0.35x$ with additive noise. Construct the covariance matrices $K_{XX}$, $K_{X*}$, and $K_{**}$, add the noise term $\sigma_n^2 I$ to the training covariance, and compute the Cholesky factorization $K_{XX} + \sigma_n^2 I = LL^\top$. Evaluate the posterior predictive mean and variance at 21 test points using triangular solves, never forming explicit matrix inverses. Compute the log marginal likelihood and verify the decomposition into the quadratic form, log-determinant, and normalization constant. Report the solve vector $\alpha = (K_{XX} + \sigma_n^2 I)^{-1}(y - m_X)$ and selected entries of the posterior covariance matrix.
10. Implement a Rust program that performs scalable Gaussian process regression using inducing points. Generate 160 training observations from a smooth synthetic simulator $g(x)$, select 18 evenly spaced inducing points, and train a sparse GP using a low-rank approximation of the kernel matrix. Construct $K_{nm}$, $K_{mm}$, and the Woodbury-based approximate solution $\alpha_{\mathrm{approx}}$ that avoids forming the full $n \times n$ covariance matrix. Predict the mean and variance at 201 test points and compute the root-mean-square error against the true simulator values. Report the approximate training complexity $O(nm^2 + m^3)$ and storage complexity $O(nm + m^2)$, and compare the inducing-point predictions with the true function values at selected locations.

These exercises span the full range of data modeling methods developed in this chapter, from weighted straight-line fitting and general linear least squares through nonlinear optimization, errors-in-variables regression, confidence limit construction, robust M-estimation, Bayesian posterior sampling, and Gaussian process regression. By implementing them in Rust, you will gain direct experience with the numerical considerations, algorithmic design choices, and interpretive judgment that distinguish reliable data modeling from mechanical formula application.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/erngGYVoHf7tumIx8goG.2","tags":[]}

# References

 1. Adomaityte, U., Maillard, O.A. and Munos, R., 2023. *Robust estimation in high-dimensional settings under heavy-tailed noise*. Journal of Machine Learning Research, 24, pp.1–42.
 2. Atkinson, A.C., Riani, M. and Cerioli, A., 2025. *Robust Statistics in Modern Data Analysis*. Springer, Berlin.
 3. Blechta, J. and Ernst, O.G., 2024. *Preconditioning strategies for Gauss–Newton methods in PDE-constrained optimization*. SIAM Journal on Scientific Computing, 46(2), pp.A1023–A1045.
 4. Calvert, J. and Jump, P., 2024. *Regularized least squares and ill-posed inverse problems*. Applied Numerical Mathematics, 190, pp.45–68.
 5. Carson, E. and Daužickaitė, D., 2024. *Mixed-precision iterative refinement for least-squares problems*. SIAM Journal on Matrix Analysis and Applications, 45(1), pp.123–147.
 6. Chaudhary, S., Mishra, P. and Gupta, R., 2025. *Robust channel estimation under impulsive noise via likelihood-based methods*. IEEE Transactions on Communications, 73(2), pp.1450–1464.
 7. Daužickaitė, D., Kressner, D. and Nakatsukasa, Y., 2025. *Variational data assimilation and large-scale least-squares methods*. SIAM Review, 67(1), pp.89–132.
 8. Epperly, E., Tropp, J.A. and Webber, R.J., 2024. *Randomized least-squares solvers with backward stability guarantees*. Foundations of Computational Mathematics, 24(3), pp.567–598.
 9. Fischer, A., Kanzow, C. and Yamashita, N., 2024. *Regularized Newton-type methods for nonlinear least-squares problems*. Mathematical Programming, 196(1–2), pp.233–268.
10. Garg, R., Kumar, V. and Singh, A., 2024. *Communication-efficient distributed least squares algorithms*. IEEE Transactions on Parallel and Distributed Systems, 35(6), pp.2101–2115.
11. Gholinejad, M. and Amiri-Simkooei, A.R., 2023. *Efficient multivariate weighted total least squares for large-scale geodetic problems*. Journal of Geodesy, 97(4), pp.1–18.
12. Golovko, V., 2025. *Hybrid parametric bootstrap methods for small-sample inference with measurement uncertainty*. Computational Statistics and Data Analysis, 195, 107602.
13. Hoffmann, F. and Onnela, J.P., 2025. *Scalable Gaussian process methods for large datasets*. Annual Review of Statistics and Its Application, 12, pp.1–28.
14. Hückelheim, J., Hascoët, L., Naumann, U. and Walther, A., 2024. *Pitfalls of automatic differentiation in scientific computing*. ACM Transactions on Mathematical Software, 50(1), pp.1–28.
15. Jin, S., Li, X. and Chen, Y., 2023. *Gauss–Helmert models for constrained errors-in-variables problems*. Journal of Computational and Applied Mathematics, 421, 114805.
16. Justus, M., Schneider, T. and Müller, S., 2024. *Bootstrap confidence intervals: accuracy and efficiency trade-offs*. Statistical Science, 39(2), pp.245–262.
17. Loh, P.L., 2025. *High-dimensional robust statistics: theory and algorithms*. Foundations and Trends in Machine Learning, 18(1–2), pp.1–160.
18. Meier, M., Nakatsukasa, Y. and Townsend, A., 2024. *Stability limits of randomized least-squares solvers*. SIAM Journal on Scientific Computing, 46(4), pp.A1890–A1915.
19. Mokhtar, M., Elamir, E.A. and Abdelrahman, H., 2023. *On the performance of bootstrap confidence intervals*. Communications in Statistics – Simulation and Computation, 52(7), pp.3120–3142.
20. Murray, R., Mahoney, M.W. and Drineas, P., 2023. *Randomized numerical linear algebra: foundations and algorithms*. Acta Numerica, 32, pp.1–87.
21. O’Brien, T.E. and Silcox, J., 2024. *Limitations of Wald-type confidence intervals in nonlinear models*. Journal of Statistical Theory and Practice, 18(3), pp.1–25.
22. Pfister, N., 2024. *Statistical foundations of least squares and maximum likelihood estimation*. Foundations and Trends in Statistics, 9(2), pp.123–198.
23. Røstum, J., Bui-Thanh, T. and Willcox, K., 2025. *Reduced-order model-informed inducing point selection for Gaussian processes*. Computer Methods in Applied Mechanics and Engineering, 420, 116789.
24. Scott, J. and Tůma, M., 2025. *Sparse direct and iterative solvers for large-scale least-squares problems*. SIAM Review, 67(2), pp.321–367.
25. Shao, Q. and Fan, J., 2024. *Subsampled Levenberg–Marquardt methods for large-scale nonlinear least squares*. Journal of Machine Learning Research, 25, pp.1–35.
26. Simpson, M.J. and Maclaren, O.J., 2023. *Profile likelihood methods for identifiability and uncertainty quantification*. SIAM Review, 65(3), pp.555–581.
27. Vrugt, J.A. and Diks, C.G.H., 2025. *Robust likelihood-based inference under model misspecification*. Journal of Econometrics, 240(1), pp.112–135.
28. Vrugt, J.A., Ter Braak, C.J.F. and Diks, C.G.H., 2025. *Markov chain Monte Carlo methods in high dimensions: theory and practice*. Environmental Modelling & Software, 175, 105987.
29. Wang, Y., Chen, X. and Li, Z., 2025. *Robust confidence intervals via M-estimation and data-driven tuning*. Journal of Computational and Graphical Statistics, 34(1), pp.89–104.
30. Xing, W., Zhang, H. and Sun, Q., 2023. *Stochastic Levenberg–Marquardt methods for nonlinear least squares*. SIAM Journal on Optimization, 33(4), pp.2890–2915.
31. Zhang, Y., Wang, X. and Liu, H., 2023. *Adaptive tuning in robust regression via bilevel optimization*. Journal of Machine Learning Research, 24, pp.1–38.

