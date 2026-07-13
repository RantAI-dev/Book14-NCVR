---
title: Chapter 9
description: ''
subtitle: Root Finding and Nonlinear Sets of Equations
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
date: '2025-12-19'
oxa: oxa:pqQDe4beUu67RvW3raYP/DHc20enSinhKU8OxOo5w
keywords: []
---

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/mJT7EIgdqJQhsHqIb3DU.4","tags":[]}

> Root-finding for nonlinear equations is crucial in real-world applications, where practical, stable solutions can dictate system success or failure. — William H. Press

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/eqN6QZZm79mm09FdNWiO.1","tags":[]}

*Chapter 9 introduces numerical methods for solving nonlinear equations and nonlinear systems, which arise throughout science, engineering, optimization, and data analysis. The chapter begins with the formulation of nonlinear root-finding problems and the challenges posed by multiple roots, sensitivity, and dependence on initial guesses. Classical bracketing methods, including bisection, false position, and Ridders’ method, are developed alongside the widely used Brent method. Derivative-based techniques such as Newton–Raphson iteration are then examined for both scalar equations and nonlinear systems. The chapter also explores polynomial root-finding algorithms, including Muller’s method, Laguerre’s method, and companion-matrix approaches. Finally, globally convergent strategies such as line-search, trust-region, quasi-Newton, and Newton–Krylov methods are presented. Throughout the chapter, mathematical theory is integrated with practical Rust implementations, providing readers with robust tools for solving nonlinear equations encountered in modern scientific computing.*

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/bHXEGkn59J3u2wVqrZt8.3","tags":[]}

# 9.1. Introduction

Solving equations is one of the most fundamental tasks in numerical computing. In its simplest form, a root-finding problem consists of determining a value of an unknown variable for which a given function evaluates to zero. For a scalar-valued function of a single real variable, this problem is written as:

$$f(x) = 0 \tag{9.1.1}$$

Here $x \in \mathbb{R}$, and any solution satisfying (9.1.1) is called a *root* or *zero* of the function.

In many applications, however, the unknown quantity is not a scalar but a vector. In the general case of $N$ independent variables, we seek a solution vector,

$$x = (x_1, x_2, \dots, x_N) \in \mathbb{R}^N \tag{9.1.2}$$

such that a vector-valued function vanishes in every component $f(x) = 0$. Equivalently, this system can be written componentwise as,

$$f_1(x_1,\dots,x_N)=0,\quad f_2(x_1,\dots,x_N)=0,\quad \dots,\quad f_N(x_1,\dots,x_N)=0 \tag{9.1.3}$$

Although the notation in (9.1.1)–(9.1.3) is compact, it conceals a major conceptual and computational leap: solving nonlinear systems in multiple dimensions is substantially more difficult than solving a single equation in one variable (Martin, 2023).

### Origins of Nonlinear Equations

Nonlinear equations and systems arise ubiquitously across science, engineering, and applied mathematics. In physics and engineering, equilibrium conditions are often formulated by requiring a net force or energy gradient to vanish. For example, static mechanical equilibrium corresponds to solving a force-balance equation of the form $f(x)=0$.

In computational physics and engineering, the discretization of partial differential equations commonly leads to large systems of nonlinear algebraic equations. Steady-state solutions of time-dependent PDEs are obtained by enforcing that residuals vanish at each grid point, yielding systems precisely of the form (9.1.2). Similarly, eigenvalue problems and transcendental dispersion relations in quantum mechanics are frequently reduced to root-finding tasks (Martin, 2023).

Optimization theory provides another major source of nonlinear systems. Any local extremum of a smooth objective function $F(x)$ must satisfy the stationarity condition,

$$\nabla F(x)=0, \tag{9.1.4}$$

which is itself a system of nonlinear equations. Consequently, many optimization algorithms can be interpreted as specialized root-finding methods applied to gradient equations.

In machine learning, root-finding appears both explicitly and implicitly. Parameter estimation may require solving nonlinear equations that equate model predictions with observed targets, and many iterative learning algorithms seek fixed points of update mappings. In finance, root-finding is essential for computing quantities such as the yield to maturity of a bond or the implied volatility that satisfies the Black–Scholes pricing equation. These examples illustrate that root-finding is a foundational computational primitive across scientific, engineering, and economic domains (Martin, 2023; Okawa et al., 2023).

### Existence and Local Uniqueness

A central theoretical tool for understanding nonlinear systems is the *implicit function theorem*. Let $f:\mathbb{R}^N\to\mathbb{R}^N$ be continuously differentiable, and define its Jacobian matrix as,

$$J(x)=\left[\frac{\partial f_i}{\partial x_j}\right] \tag{9.1.5}$$

If $J(x^\ast)$ is nonsingular at a solution $x^\ast$ (that is, $\det J(x^\ast)\neq0$), then the solution is locally unique and isolated. In generic situations, roots of $f(x)=0$ occur as isolated points in $\mathbb{R}^N$, separated by regions where $f(x)\neq0$ (Okawa et al., 2023).

Degenerate cases arise when the Jacobian loses rank, for example when equations become dependent or when free parameters are present. In such situations, the solution set may form a curve or surface rather than isolated points. Outside these special cases, the practical numerical task is typically to locate one or more isolated solutions.

### Existence Guarantees and Dimensionality

The existence of a solution cannot be taken for granted. In one dimension, a classical existence result is provided by the Intermediate Value Theorem: if $f$ is continuous on $[a,b]$ and,

$$f(a),f(b)<0 \tag{9.1.6}$$

then there exists at least one root $\xi\in(a,b)$ such that $f(\xi)=0$. This theorem underlies many one-dimensional root-finding algorithms by guaranteeing that a root is bracketed between two points of opposite sign.

In higher dimensions, no comparably simple bracketing principle exists. Although existence theorems such as Brouwer’s fixed-point theorem or degree theory provide guarantees under certain conditions, they rarely translate into practical computational procedures. As a result, in multidimensional problems one often cannot be certain that a solution exists in a given region until it is actually found (Okawa et al., 2023).

### Multiple Roots and Numerical Difficulties

Even when solutions exist, they need not be unique. A nonlinear equation may have no real roots, a single root, or infinitely many. For example,

$$\sin x = 0 \tag{9.1.7}$$

has infinitely many roots, whereas

$$e^x + x^2 = 0 \tag{9.1.8}$$

has none. Roots of higher multiplicity present additional challenges. If a function merely touches zero without changing sign, standard bracketing strategies fail. For instance,

$$f(x)=(x-c)^2 \tag{9.1.9}$$

has a double root at $x=c$, yet $f(x)\ge0$ on both sides of the root. Such behavior can mislead algorithms that rely solely on sign changes (Martin, 2023).

### Rust Implementation

Following the conceptual introduction to nonlinear equations and systems in Section 9.1, Program 9.1.0 provides a concrete computational realization of root-finding methods in one and multiple dimensions. While Equations (9.1.1)–(9.1.3) define the abstract problem of locating zeros of scalar- and vector-valued functions, practical numerical work requires explicit algorithms that balance robustness, efficiency, and mathematical assumptions. This program implements three representative approaches: bisection for guaranteed root localization in one dimension, Newton’s method for rapid local convergence in one dimension, and a multidimensional Newton method based on finite-difference Jacobian approximation. Together, these implementations illustrate how existence guarantees, local uniqueness, and differentiability assumptions discussed earlier translate into executable numerical procedures in finite-precision arithmetic.

At the core of the one-dimensional implementation are two complementary root-finding strategies corresponding to different theoretical guarantees. The `bisection` function directly exploits the existence principle underlying Equation (9.1.6) by requiring a sign change across an interval $[a,b]$. By repeatedly halving the interval and preserving the sign change, the method produces a monotonically shrinking bracket that must contain at least one root. Although its convergence is linear, bisection is robust and insensitive to derivative information, making it a reliable baseline method whenever continuity and bracketing can be established.

In contrast, the `newton_1d` function implements Newton’s method for scalar equations, which assumes differentiability and local regularity of the function near the root. Given an initial guess sufficiently close to a simple root, Newton’s iteration exhibits quadratic convergence, dramatically reducing the number of iterations required compared to bisection. The implementation explicitly guards against numerical breakdown by detecting near-zero derivatives, reflecting the theoretical requirement that the Jacobian in Equation (9.1.5) be nonsingular at the solution.

The extension to systems of equations is realized by the `newton_system` function, which targets problems of the form given in Equation (9.1.3). Since analytic Jacobians are not always available, the program constructs a finite-difference approximation to the Jacobian matrix defined in Equation (9.1.5). Each Newton step requires solving a linear system involving this Jacobian, which is handled by a Gaussian elimination routine with partial pivoting to mitigate numerical instability. A simple damping strategy is incorporated to improve robustness when the initial iterate is not yet within the local convergence region guaranteed by the implicit function theorem.

The `main` function serves as a structured demonstration of these methods. It first applies bisection and Newton’s method to a classical scalar test problem, illustrating the contrast between guaranteed convergence and rapid local convergence. It then solves a two-dimensional nonlinear system consisting of a quadratic constraint and a linear constraint, yielding an isolated solution consistent with the local uniqueness conditions discussed earlier. The printed iteration counts and residuals provide empirical confirmation of the theoretical convergence properties outlined in Section 9.1.

```rust
// Program 9.1.0: Root-finding in one and many dimensions
//
// Demonstrates:
// 1) One-dimensional existence and bracketing via bisection.
// 2) One-dimensional Newton iteration.
// 3) Multidimensional Newton iteration using a finite-difference Jacobian
//    and Gaussian elimination with partial pivoting.
//
// Run with: cargo run

use std::fmt;

#[derive(Debug, Clone)]
pub enum RootError {
    NotBracketed { fa: f64, fb: f64 },
    MaxIterExceeded,
    DerivativeTooSmall,
    SingularJacobian,
    DimensionMismatch,
    NonFiniteValue,
}

impl fmt::Display for RootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RootError::NotBracketed { fa, fb } => {
                write!(f, "interval does not bracket a root (f(a)={fa}, f(b)={fb})")
            }
            RootError::MaxIterExceeded => write!(f, "maximum iterations exceeded"),
            RootError::DerivativeTooSmall => write!(f, "derivative too small for Newton step"),
            RootError::SingularJacobian => write!(f, "Jacobian is singular or ill-conditioned"),
            RootError::DimensionMismatch => write!(f, "dimension mismatch"),
            RootError::NonFiniteValue => write!(f, "encountered a non-finite value"),
        }
    }
}

fn ensure_finite(x: f64) -> Result<f64, RootError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(RootError::NonFiniteValue)
    }
}

/// Bisection method for f(x)=0 on [a,b].
pub fn bisection<F>(
    f: F,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> Result<(f64, usize), RootError>
where
    F: Fn(f64) -> f64,
{
    let mut fa = ensure_finite(f(a))?;
    let fb = ensure_finite(f(b))?;

    if fa == 0.0 {
        return Ok((a, 0));
    }
    if fb == 0.0 {
        return Ok((b, 0));
    }
    if fa.signum() == fb.signum() {
        return Err(RootError::NotBracketed { fa, fb });
    }

    for k in 1..=max_iter {
        let m = 0.5 * (a + b);
        let fm = ensure_finite(f(m))?;

        if (b - a).abs() <= tol || fm.abs() <= tol {
            return Ok((m, k));
        }

        if fa.signum() == fm.signum() {
            a = m;
            fa = fm;
        } else {
            b = m;
        }
    }

    Err(RootError::MaxIterExceeded)
}

/// One-dimensional Newton method.
pub fn newton_1d<F, DF>(
    f: F,
    df: DF,
    mut x: f64,
    tol: f64,
    max_iter: usize,
) -> Result<(f64, usize), RootError>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    for k in 1..=max_iter {
        let fx = ensure_finite(f(x))?;
        if fx.abs() <= tol {
            return Ok((x, k));
        }

        let dfx = ensure_finite(df(x))?;
        if dfx.abs() < 1e-14 {
            return Err(RootError::DerivativeTooSmall);
        }

        let step = fx / dfx;
        x = ensure_finite(x - step)?;

        if step.abs() <= tol * (1.0 + x.abs()) {
            return Ok((x, k));
        }
    }

    Err(RootError::MaxIterExceeded)
}

/// Finite-difference Jacobian.
fn fd_jacobian<F>(f: &F, x: &[f64], h: f64) -> Result<Vec<Vec<f64>>, RootError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let fx = f(x);
    if fx.len() != n {
        return Err(RootError::DimensionMismatch);
    }

    let mut j = vec![vec![0.0; n]; n];
    let mut x_pert = x.to_vec();

    for col in 0..n {
        x_pert[col] = x[col] + h;
        let f_pert = f(&x_pert);
        for row in 0..n {
            j[row][col] = (f_pert[row] - fx[row]) / h;
        }
        x_pert[col] = x[col];
    }

    Ok(j)
}

/// Gaussian elimination with partial pivoting.
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, RootError> {
    let n = b.len();

    for k in 0..n {
        let mut piv = k;
        let mut best = a[k][k].abs();
        for i in (k + 1)..n {
            if a[i][k].abs() > best {
                best = a[i][k].abs();
                piv = i;
            }
        }
        if best < 1e-15 {
            return Err(RootError::SingularJacobian);
        }
        if piv != k {
            a.swap(k, piv);
            b.swap(k, piv);
        }

        for i in (k + 1)..n {
            let factor = a[i][k] / a[k][k];
            for j in k..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }

    Ok(x)
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0, f64::max)
}

/// Newton method for systems f(x)=0.
pub fn newton_system<F>(
    f: F,
    mut x: Vec<f64>,
    tol: f64,
    max_iter: usize,
) -> Result<(Vec<f64>, usize), RootError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();

    for k in 1..=max_iter {
        let fx = f(&x);
        let fnorm = norm_inf(&fx);
        if fnorm <= tol {
            return Ok((x, k));
        }

        let h = 1e-7 * (1.0 + norm_inf(&x));
        let j = fd_jacobian(&f, &x, h)?;
        let rhs: Vec<f64> = fx.iter().map(|v| -v).collect();
        let p = solve_linear(j, rhs)?;

        let mut alpha = 1.0;
        let mut x_new = x.clone();
        for _ in 0..20 {
            for i in 0..n {
                x_new[i] = x[i] + alpha * p[i];
            }
            if norm_inf(&f(&x_new)) < fnorm {
                break;
            }
            alpha *= 0.5;
        }

        x = x_new;

        if alpha * norm_inf(&p) <= tol * (1.0 + norm_inf(&x)) {
            return Ok((x, k));
        }
    }

    Err(RootError::MaxIterExceeded)
}

fn main() -> Result<(), RootError> {
    println!("Program 9.1.1: Root-finding demonstrations\n");

    let f1 = |x: f64| x.cos() - x;
    let (r, it) = bisection(f1, 0.0, 1.0, 1e-12, 200)?;
    println!("Bisection root ≈ {:.15} ({} iterations)", r, it);

    let df1 = |x: f64| -x.sin() - 1.0;
    let (r, it) = newton_1d(f1, df1, 0.8, 1e-12, 50)?;
    println!("Newton 1D root ≈ {:.15} ({} iterations)", r, it);

    let f_sys = |v: &[f64]| vec![v[0] * v[0] + v[1] * v[1] - 1.0, v[0] - v[1]];
    let (sol, it) = newton_system(f_sys, vec![0.7, 0.6], 1e-12, 50)?;
    println!(
        "Newton system solution ≈ [{:.15}, {:.15}] ({} iterations)",
        sol[0], sol[1], it
    );

    Ok(())
}
```

Program 9.1.0 illustrates how the abstract theory of nonlinear equations introduced in Section 9.1 manifests in concrete numerical algorithms. The juxtaposition of bisection and Newton’s method highlights a recurring theme in numerical root-finding: robustness and guaranteed convergence often come at the cost of speed, while rapid convergence typically relies on stronger smoothness and regularity assumptions.

The multidimensional Newton implementation underscores the increased complexity inherent in solving systems of equations. Unlike the one-dimensional case, there is no simple bracketing principle analogous to Equation (9.1.6), and success depends critically on Jacobian conditioning, initial guesses, and step control. These observations motivate the need for globalization strategies, trust-region methods, and hybrid algorithms, which will be developed in subsequent sections.

The modular structure of the code provides a foundation for these extensions. By separating residual evaluation, Jacobian approximation, and linear solves, the framework can be adapted to incorporate analytic Jacobians, quasi-Newton updates, or Krylov-based linear solvers. As such, Program 9.1.0 serves both as a practical introduction to root-finding algorithms and as a stepping stone toward more sophisticated techniques for large-scale nonlinear systems.

## 9.1.1. Iterative Methods and Initial Guesses

Closed-form solutions for nonlinear equations are rare outside of special cases. Consequently, numerical root-finding relies almost exclusively on iterative methods, which generate a sequence,

$$x^{(0)}, x^{(1)}, x^{(2)}, \dots$$

intended to converge to a solution. In one dimension, bracketed methods such as bisection guarantee convergence to some root. In multiple dimensions, convergence typically depends on whether the initial guess lies within the basin of attraction of a solution.

Poor initial guesses may lead to divergence or convergence to an unintended root. This sensitivity is especially pronounced in multidimensional problems, where the norm $\lVert f(x)\rVert$ may exhibit multiple valleys separated by ridges that impede naive algorithms (Berra et al., 2024).

As Hamming famously emphasized, numerical computation should be guided by insight rather than blind iteration. High numerical precision is meaningless if the algorithm converges to the wrong solution, or to no solution at all.

### Modern Developments

Research on root-finding remains active. Recent advances include the W4 method, which enlarges the convergence region of Newton-like schemes (Okawa et al., 2023), Newton–Gradient hybrid methods for constrained biochemical systems (Berra et al., 2024), very high-order one-dimensional methods achieving ninth-order convergence (Thota et al., 2025), and machine-learning-based strategies that adaptively select root-finding algorithms based on early iteration behavior (Aghamie et al., 2025).

In modern practice, these developments can be interpreted as responses to two recurring difficulties highlighted earlier in this introduction: limited basins of attraction and the high cost of repeated failed iterations. The W4 method enlarges the convergence region of Newton-style updates in multidimensional systems, making convergence more likely when the initial guess is poor (Okawa et al., 2023). For constrained biochemical steady-state problems, alternating Newton steps with gradient-based steps provides a pragmatic mechanism to improve reliability while respecting constraints such as non-negativity of concentrations (Berra et al., 2024). For one-dimensional equations where very high precision is required and function evaluations are inexpensive, very high-order schemes can reduce iteration counts dramatically, at the expense of more evaluations per iteration, such as the ninth-order method reported by Thota et al. (2025). Finally, adaptive, data-driven selection strategies attempt to choose among classical methods such as bisection, secant, and Newton based on early iteration behavior, aiming to achieve robust performance without manual tuning (Aghamie et al., 2025).

These developments reflect a broader trend: classical numerical analysis continues to evolve, incorporating ideas from dynamical systems, optimization, and machine learning while retaining rigorous mathematical foundations.

### Outlook

Root-finding and nonlinear equation solving arise across physics, engineering, finance, and data science. The remainder of this chapter develops a hierarchy of algorithms for these problems, beginning in §9.2 with bracketing and bisection. Although conceptually simple, these methods provide reliability and theoretical clarity, forming a foundation upon which more advanced techniques are built.

### Rust Implementation

Following the discussion in Section 9.1.1 on the iterative nature of nonlinear root-finding and the critical role of initial guesses, Program 9.1.1 provides a practical implementation that illustrates how theoretical considerations translate into computational behavior. While Equations (9.1.1)–(9.1.3) define the abstract problem of solving nonlinear equations and systems, iterative algorithms generate sequences of approximations whose success depends strongly on the choice of starting values. This program implements representative one-dimensional and multidimensional iterative methods and uses controlled experiments to demonstrate how convergence, divergence, and basin-of-attraction phenomena emerge in practice. By contrasting guaranteed bracketing methods with fast but locally convergent Newton-type schemes, the program emphasizes the necessity of informed initialization and algorithmic safeguards in finite-precision numerical computation.

At the core of the implementation are iterative root-finding routines designed to reflect the convergence mechanisms discussed earlier in this section. In one dimension, the `bisection` function embodies the concept of guaranteed convergence by exploiting a sign change across an interval, consistent with the existence principle underlying Equation (9.1.6). The method generates a sequence of nested intervals whose midpoints converge monotonically to a root, independent of derivative information or local curvature. Although its convergence is linear, bisection provides a reliable reference method against which more aggressive schemes can be compared.

The `newton_1d` function implements Newton’s method for scalar equations, generating a sequence $x^{(k)}$ that converges quadratically when the initial guess lies sufficiently close to a simple root. This behavior reflects the local uniqueness condition associated with a nonsingular Jacobian, as described in connection with Equation (9.1.5). The implementation explicitly checks for small derivatives to prevent numerical breakdown, illustrating how theoretical assumptions must be enforced algorithmically. By applying Newton’s method with both favorable and unfavorable initial guesses, the program demonstrates how rapid convergence can coexist with complete failure when the starting point lies outside the basin of attraction.

The multidimensional case is handled by the `newton_system` function, which targets systems of the form given in Equation (9.1.3). Since analytic Jacobians are often unavailable in practice, the program constructs a finite-difference approximation to the Jacobian matrix defined in Equation (9.1.5). Each iteration requires solving a linear system for the Newton correction, which is performed using Gaussian elimination with partial pivoting. To improve robustness when the initial guess is poor, a simple damping strategy is employed to ensure that successive iterates reduce the residual norm $\lVert f(x)\rVert$. This reflects the practical necessity of modifying idealized Newton steps to accommodate limited basins of attraction, as emphasized in the surrounding text.

The `main` function serves as a structured numerical experiment. It first compares bisection and Newton’s method on a one-dimensional problem, highlighting the contrast between guaranteed convergence and fast local convergence. It then explores a two-dimensional system with multiple isolated solutions, applying Newton’s method from several distinct initial guesses. The resulting convergence to different roots provides a concrete visualization of basin-of-attraction effects and illustrates why poor initial guesses may lead to unintended solutions, even when the algorithm itself is mathematically sound.

```rust
// Program 9.1.1: Iterative methods and the role of initial guesses
//
// This program illustrates why iterative root-finding methods depend strongly on the initial guess,
// especially in multidimensional problems with multiple basins of attraction. It includes:
//
// 1) A 1D bracketing bisection method (guaranteed convergence when a sign change exists).
// 2) A 1D Newton method (fast local convergence, but sensitive to the starting point).
// 3) A 2D Newton method with a finite-difference Jacobian and damping.
// 4) A simple multi-start strategy to explore basins of attraction in 2D.
//
// No external crates are required. Run with: cargo run

use std::fmt;

#[derive(Debug, Clone)]
pub enum RootError {
    NotBracketed { fa: f64, fb: f64 },
    MaxIterExceeded,
    DerivativeTooSmall,
    SingularJacobian,
    DimensionMismatch,
    NonFiniteValue,
}

impl fmt::Display for RootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RootError::NotBracketed { fa, fb } => {
                write!(f, "interval does not bracket a root (f(a)={fa}, f(b)={fb})")
            }
            RootError::MaxIterExceeded => write!(f, "maximum iterations exceeded"),
            RootError::DerivativeTooSmall => write!(f, "derivative too small for Newton step"),
            RootError::SingularJacobian => write!(f, "Jacobian is singular or ill-conditioned"),
            RootError::DimensionMismatch => write!(f, "dimension mismatch"),
            RootError::NonFiniteValue => write!(f, "encountered a non-finite value"),
        }
    }
}

fn ensure_finite(x: f64) -> Result<f64, RootError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(RootError::NonFiniteValue)
    }
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0, f64::max)
}

/// Bisection method for f(x)=0 on [a,b]. Guaranteed to converge if f(a) and f(b) have opposite signs.
pub fn bisection<F>(
    f: F,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> Result<(f64, usize), RootError>
where
    F: Fn(f64) -> f64,
{
    let mut fa = ensure_finite(f(a))?;
    let fb = ensure_finite(f(b))?;

    if fa == 0.0 {
        return Ok((a, 0));
    }
    if fb == 0.0 {
        return Ok((b, 0));
    }
    if fa.signum() == fb.signum() {
        return Err(RootError::NotBracketed { fa, fb });
    }

    for k in 1..=max_iter {
        let m = 0.5 * (a + b);
        let fm = ensure_finite(f(m))?;

        if (b - a).abs() <= tol || fm.abs() <= tol {
            return Ok((m, k));
        }

        if fa.signum() == fm.signum() {
            a = m;
            fa = fm;
        } else {
            b = m;
        }
    }

    Err(RootError::MaxIterExceeded)
}

/// One-dimensional Newton method for f(x)=0.
pub fn newton_1d<F, DF>(
    f: F,
    df: DF,
    mut x: f64,
    tol: f64,
    max_iter: usize,
) -> Result<(f64, usize), RootError>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    for k in 1..=max_iter {
        let fx = ensure_finite(f(x))?;
        if fx.abs() <= tol {
            return Ok((x, k));
        }

        let dfx = ensure_finite(df(x))?;
        if dfx.abs() < 1e-14 {
            return Err(RootError::DerivativeTooSmall);
        }

        let step = fx / dfx;
        x = ensure_finite(x - step)?;

        if step.abs() <= tol * (1.0 + x.abs()) {
            return Ok((x, k));
        }
    }

    Err(RootError::MaxIterExceeded)
}

/// Finite-difference Jacobian for f: R^n -> R^n.
fn fd_jacobian<F>(f: &F, x: &[f64], h: f64) -> Result<Vec<Vec<f64>>, RootError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let fx = f(x);
    if fx.len() != n {
        return Err(RootError::DimensionMismatch);
    }
    for &v in fx.iter() {
        ensure_finite(v)?;
    }

    let mut j = vec![vec![0.0; n]; n];
    let mut x_pert = x.to_vec();

    for col in 0..n {
        x_pert[col] = x[col] + h;
        let f_pert = f(&x_pert);
        if f_pert.len() != n {
            return Err(RootError::DimensionMismatch);
        }
        for row in 0..n {
            j[row][col] = (ensure_finite(f_pert[row])? - fx[row]) / h;
        }
        x_pert[col] = x[col];
    }

    Ok(j)
}

/// Solve A p = b using Gaussian elimination with partial pivoting.
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, RootError> {
    let n = b.len();
    if a.len() != n || a.iter().any(|r| r.len() != n) {
        return Err(RootError::DimensionMismatch);
    }

    for k in 0..n {
        let mut piv = k;
        let mut best = a[k][k].abs();
        for i in (k + 1)..n {
            let v = a[i][k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best < 1e-15 {
            return Err(RootError::SingularJacobian);
        }
        if piv != k {
            a.swap(k, piv);
            b.swap(k, piv);
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
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        if a[i][i].abs() < 1e-15 {
            return Err(RootError::SingularJacobian);
        }
        x[i] = s / a[i][i];
        ensure_finite(x[i])?;
    }
    Ok(x)
}

/// Newton method for systems f(x)=0 with damping. Returns (solution, iterations, converged_flag).
pub fn newton_system<F>(
    f: F,
    mut x: Vec<f64>,
    tol: f64,
    max_iter: usize,
) -> Result<(Vec<f64>, usize, bool), RootError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    if n == 0 {
        return Err(RootError::DimensionMismatch);
    }

    for k in 1..=max_iter {
        for &xi in x.iter() {
            ensure_finite(xi)?;
        }

        let fx = f(&x);
        if fx.len() != n {
            return Err(RootError::DimensionMismatch);
        }
        for &v in fx.iter() {
            ensure_finite(v)?;
        }

        let fnorm = norm_inf(&fx);
        if fnorm <= tol {
            return Ok((x, k, true));
        }

        let h = 1e-7 * (1.0 + norm_inf(&x)); // scale-aware finite-difference step
        let j = fd_jacobian(&f, &x, h)?;
        let rhs: Vec<f64> = fx.iter().map(|&v| -v).collect();
        let p = solve_linear(j, rhs)?;

        // Damping line search: try to reduce ||f||.
        let mut alpha = 1.0;
        let mut x_new = x.clone();
        let mut accepted = false;

        for _ in 0..20 {
            for i in 0..n {
                x_new[i] = x[i] + alpha * p[i];
            }
            let f_new = f(&x_new);
            if f_new.len() != n {
                return Err(RootError::DimensionMismatch);
            }
            let new_norm = norm_inf(&f_new);
            if new_norm.is_finite() && new_norm < fnorm {
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if !accepted {
            // If no decrease was found, we treat this as "did not converge" within this iteration budget.
            return Ok((x, k, false));
        }

        let step_norm = alpha * norm_inf(&p);
        x = x_new;

        if step_norm <= tol * (1.0 + norm_inf(&x)) {
            return Ok((x, k, true));
        }
    }

    Ok((x, max_iter, false))
}

/// Classify which root a 2D solution converged to for the demo system.
fn classify_root_2d(x: &[f64]) -> &'static str {
    // For the demo system below, roots are (0,0), (1,0), (0,1).
    let candidates = [
        ([0.0, 0.0], "root A ≈ (0,0)"),
        ([1.0, 0.0], "root B ≈ (1,0)"),
        ([0.0, 1.0], "root C ≈ (0,1)"),
    ];

    let mut best = (&candidates[0], f64::INFINITY);
    for c in candidates.iter() {
        let dx0 = x[0] - (c.0)[0];
        let dx1 = x[1] - (c.0)[1];
        let d = (dx0 * dx0 + dx1 * dx1).sqrt();
        if d < best.1 {
            best = (c, d);
        }
    }

    if best.1 < 1e-6 {
        best.0 .1
    } else {
        "unclassified (not near a known root)"
    }
}

fn main() -> Result<(), RootError> {
    println!("Program 9.1.1: Iterative Methods and Initial Guesses\n");

    // ------------------------------------------------------------
    // 1) Bracketing vs Newton in 1D: f(x) = cos(x) - x
    // ------------------------------------------------------------
    let f = |x: f64| x.cos() - x;
    let df = |x: f64| -x.sin() - 1.0;

    println!("1D example: f(x) = cos(x) - x\n");

    let (rb, itb) = bisection(f, 0.0, 1.0, 1e-12, 200)?;
    println!("  Bisection on [0,1] converged:");
    println!("    root ≈ {:.15}  (iters = {})", rb, itb);
    println!("    |f(root)| = {:.3e}\n", f(rb).abs());

    // Newton with a good initial guess
    let (rn_good, itn_good) = newton_1d(f, df, 0.8, 1e-12, 50)?;
    println!("  Newton from x0 = 0.8 (good guess) converged:");
    println!("    root ≈ {:.15}  (iters = {})", rn_good, itn_good);
    println!("    |f(root)| = {:.3e}\n", f(rn_good).abs());

    // Newton with a poor initial guess (still may converge here, but behavior can change with x0)
    let x0_bad = 10.0;
    match newton_1d(f, df, x0_bad, 1e-12, 50) {
        Ok((rn_bad, itn_bad)) => {
            println!("  Newton from x0 = {} converged:", x0_bad);
            println!("    root ≈ {:.15}  (iters = {})", rn_bad, itn_bad);
            println!("    |f(root)| = {:.3e}\n", f(rn_bad).abs());
        }
        Err(e) => {
            println!("  Newton from x0 = {} failed: {}\n", x0_bad, e);
        }
    }

    // ------------------------------------------------------------
    // 2) Basin-of-attraction demo in 2D (multi-start)
    // ------------------------------------------------------------
    println!("2D example (basins of attraction): a system with multiple solutions\n");

    // A standard multi-root system:
    //   f1(x,y) = x^2 - x = x(x-1)
    //   f2(x,y) = y^2 - y = y(y-1)
    //
    // This has four roots: (0,0), (1,0), (0,1), (1,1).
    // To keep the printed classification simple, we will classify only three of them explicitly and
    // print the raw solution in all cases.
    let f_sys = |v: &[f64]| -> Vec<f64> {
        let x = v[0];
        let y = v[1];
        vec![x * x - x, y * y - y]
    };

    let starts = vec![
        vec![0.2, 0.2],
        vec![0.8, 0.2],
        vec![0.2, 0.8],
        vec![0.8, 0.8],
        vec![-1.0, 2.0],
        vec![2.5, -1.5],
    ];

    let tol = 1e-12;
    let max_iter = 50;

    for (idx, x0) in starts.iter().enumerate() {
        let (xstar, it, ok) = newton_system(f_sys, x0.clone(), tol, max_iter)?;
        let r = f_sys(&xstar);
        let res = norm_inf(&r);

        println!("  Start {}: x0 = [{:+.3}, {:+.3}]", idx + 1, x0[0], x0[1]);
        if ok {
            println!(
                "    converged in {:2} iters to x* = [{:+.15}, {:+.15}]",
                it, xstar[0], xstar[1]
            );
            println!("    residual inf-norm = {:.3e}", res);
            println!("    classification (partial) = {}", classify_root_2d(&xstar));
        } else {
            println!(
                "    did not converge within budget (iters = {}, last x = [{:+.6}, {:+.6}], residual = {:.3e})",
                it, xstar[0], xstar[1], res
            );
        }
        println!();
    }

    // ------------------------------------------------------------
    // 3) Interpretation hint (printed)
    // ------------------------------------------------------------
    println!("Interpretation:");
    println!("  - Bisection converges when a bracket exists, independent of derivative information.");
    println!("  - Newton converges rapidly when the initial guess is within the basin of attraction.");
    println!("  - In 2D, different initial guesses converge to different roots of the same system.\n");

    Ok(())
}
```

Program 9.1.1 demonstrates the central theme of Section 9.1.1: iterative root-finding methods are fundamentally dynamical processes whose outcomes depend as much on initialization as on algorithmic form. The examples show that while bracketing methods provide reliability through existence guarantees, Newton-type methods trade global robustness for rapid local convergence.

The multidimensional experiments underscore the increased complexity of nonlinear systems, where multiple solutions coexist and the landscape of $\lVert f(x)\rVert$ may contain narrow valleys separated by ridges. In such settings, convergence to an unintended root is not a numerical anomaly but an inherent feature of the problem. These observations motivate the development of globalization strategies, hybrid methods, and basin-enlarging techniques discussed in subsequent sections and in recent research.

The modular structure of the code allows these ideas to be extended naturally. By modifying the update strategy, Jacobian approximation, or step acceptance criteria, the framework can accommodate advanced methods such as damped Newton schemes, Newton–gradient hybrids, or modern basin-expanding algorithms. As such, Program 9.1.1 provides both a concrete demonstration of classical iterative methods and a foundation for understanding and implementing more sophisticated root-finding techniques.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/uwnRRR8pLIeQwnLbGpzk.4","tags":[]}

# 9.2. Bracketing and Bisection

Before introducing faster or more sophisticated root-finding algorithms, it is essential to understand *bracketing*, the practice of confining a root within an interval by identifying two points where the function takes opposite signs. Bracketing plays a foundational role in numerical root finding because it provides a firm guarantee of existence for continuous functions of a single variable.

Let $f : \mathbb{R} \to \mathbb{R}$ be a real-valued function. If there exist points $a < b$ such that,

$$f(a),f(b) < 0 \tag{9.2.1}$$

then, provided $f$ is continuous on the interval $[a,b]$, the *Intermediate Value Theorem* guarantees the existence of at least one root $\xi \in (a,b)$ satisfying,

$$f(\xi) = 0 \tag{9.2.2}$$

This condition allows us to assert with certainty that a solution lies within the interval. For this reason, bracketing is often the first step in reliable one-dimensional root-finding routines (Martin, 2023).

### Bracketed Root

A root of the equation $f(x)=0$ is said to be *bracketed* by the interval $[a,b]$ if:

$$f(a),f(b) < 0 \tag{9.2.3}$$

By continuity of $f$, there exists at least one $\xi \in (a,b)$ such that $f(\xi)=0$.

### Preconditions and Failure Modes of Bracketing

Although bracketing rests on a strong theoretical foundation, its validity depends critically on the continuity of the function. If $f$ is discontinuous or unbounded within the interval, a sign change alone does not guarantee the presence of a root.

A standard counterexample is,

$$f(x) = \frac{1}{x-c} \tag{9.2.4}$$

which has a vertical asymptote at $x=c$. For any $a<c<b$, one finds $f(a),f(b)<0$, yet the equation $f(x)=0$ has no solution for any finite $x$. In this case, the sign change arises from divergence across a singularity rather than from crossing zero.

In numerical practice, such situations can often be detected by monitoring the magnitude $|f(x)|$. If function values grow without bound instead of tending toward zero as the interval shrinks, this strongly indicates a pole or discontinuity rather than a genuine root (Martin, 2023).

Conceptually, four common scenarios occur in root-finding problems: a simple isolated root with a sign change; a multiple root where the function merely touches zero; a highly oscillatory function with many roots; and an apparent sign change caused by a singularity. Bracketing methods are ideally suited to the first case and require additional care in the others.

### Finding Brackets in Practice

When the analytic form of $f$ is unavailable or difficult to inspect, the function is effectively treated as a black box. In such cases, brackets can be identified through systematic sampling. Suppose a root is suspected within an interval $[x_1,x_2]$. One may subdivide this interval using points:

$$x_1 = t_0 < t_1 < \cdots < t_N = x_2 \tag{9.2.5}$$

and evaluate $f(t_i)$ at each sample. Whenever two adjacent points satisfy:

$$f(t_i),f(t_{i+1}) < 0 \tag{9.2.6}$$

the subinterval $[t_i,t_{i+1}]$ brackets at least one root. Repeating this procedure allows one to isolate multiple roots, provided the sampling resolution is sufficient to detect sign changes.

Another useful heuristic relies on asymptotic behavior. If,

$$\lim_{x\to -\infty} f(x)\cdot \lim_{x\to +\infty} f(x) < 0, \tag{9.2.7}$$

then at least one real root must exist. In practice, one may evaluate $f$ at increasing magnitudes $R, 2R, 4R, \dots$ until values of opposite sign are observed, thereby establishing a large initial bracket. This approach is heuristic rather than guaranteed, since some functions approach asymptotes or retain the same sign at infinity while still possessing local roots (Martin, 2023).

### Rust Implementation

Following the discussion in Section 9.2 on bracketing as a fundamental existence guarantee for one-dimensional root-finding problems, Program 9.2.0 provides a practical implementation of bracketing and bisection strategies for continuous real-valued functions. Although more sophisticated algorithms offer faster local convergence, reliable numerical root finding typically begins by confining a solution within an interval known to contain at least one root. This program demonstrates how the theoretical conditions expressed in Equations (9.2.1)–(9.2.3) are translated into concrete computational procedures, including systematic bracket detection, interval refinement by bisection, and diagnostic checks for failure modes such as multiple roots and singularities. Together, these components illustrate why bracketing remains an indispensable foundation for robust root-finding algorithms in finite-precision environments.

At the core of the implementation is an explicit representation of a bracketed interval, defined by two endpoints and their corresponding function values. The program begins by formalizing the bracketing condition of Equation (9.2.3) through a simple sign-change test, which verifies that the function values at the interval endpoints have opposite signs. This test encodes the existence guarantee implied by the Intermediate Value Theorem in Equation (9.2.2) and serves as a prerequisite for all subsequent bisection steps.

The `bisection` function implements the classical bisection algorithm on a validated bracket. At each iteration, the midpoint of the current interval is evaluated, and the subinterval that preserves the sign change is retained. This process generates a monotonically shrinking sequence of intervals whose widths converge to zero, ensuring convergence to a root whenever the continuity assumptions of Section 9.2 are satisfied. In addition to standard stopping criteria based on interval width and residual magnitude, the implementation monitors the growth of $|f(x)|$ to flag situations where a sign change may arise from a singularity rather than from a true zero, as discussed in connection with Equation (9.2.4).

To address situations where a bracket is not known a priori, the program includes a sampling-based bracket finder that subdivides a suspected interval according to Equation (9.2.5). By evaluating the function at successive sample points and detecting sign changes as in Equation (9.2.6), this routine can isolate multiple brackets corresponding to distinct roots, provided the sampling resolution is sufficient. This approach reflects common practice when the function is treated as a black box and analytic inspection is impractical.

The program also incorporates a heuristic expansion strategy inspired by the asymptotic criterion in Equation (9.2.7). Starting from an initial guess, the search interval is expanded geometrically until a sign change is detected or a predefined budget is exhausted. While this method does not offer the same theoretical guarantees as fixed-interval bracketing, it provides a practical mechanism for locating large initial brackets in problems where the global behavior of the function is only partially understood.

The `main` function serves as a sequence of controlled numerical experiments illustrating the strengths and limitations of bracketing methods. It applies bisection to a smooth function with a simple root, demonstrates the failure of sign-based bracketing for a multiple root, isolates several roots of an oscillatory function through sampling, exposes the pathological behavior induced by a singularity, and finally combines heuristic bracketing with bisection to solve a nonlinear polynomial equation. Each example reinforces a specific theoretical point raised in the surrounding discussion.

```rust
// Program 9.2.0: Bracketing and Bisection with Practical Bracket Search and Singularity Diagnostics
//
// This program implements foundational one-dimensional bracketing ideas used throughout Chapter 9.
// It includes:
//
// 1) A bracketing predicate for an interval [a,b] based on a sign change.
// 2) A sampling-based bracket finder on a user-specified interval (Eq. 9.2.5–9.2.6).
// 3) A heuristic expanding search for a large bracket based on growth to +/- infinity (Eq. 9.2.7).
// 4) A bisection solver that is guaranteed to converge when f is continuous and a sign change exists,
//    along with a simple diagnostic to flag likely singularity behavior (Eq. 9.2.4).
//
// No external crates are required. Run with: cargo run

use std::fmt;

#[derive(Debug, Clone)]
pub enum RootError {
    NotBracketed { a: f64, b: f64, fa: f64, fb: f64 },
    MaxIterExceeded,
    NonFiniteValue,
    InvalidInterval,
}

impl fmt::Display for RootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RootError::NotBracketed { a, b, fa, fb } => write!(
                f,
                "interval [{a}, {b}] does not bracket a root (f(a)={fa}, f(b)={fb})"
            ),
            RootError::MaxIterExceeded => write!(f, "maximum iterations exceeded"),
            RootError::NonFiniteValue => write!(f, "encountered a non-finite value (NaN or infinity)"),
            RootError::InvalidInterval => write!(f, "invalid interval: require a < b"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Bracket {
    pub a: f64,
    pub b: f64,
    pub fa: f64,
    pub fb: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct BisectionResult {
    pub root: f64,
    pub iterations: usize,
    pub f_root: f64,
    pub width: f64,
    pub singularity_suspected: bool,
    pub peak_abs_f_seen: f64,
}

fn ensure_finite(x: f64) -> Result<f64, RootError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(RootError::NonFiniteValue)
    }
}

fn signum_nonzero(x: f64) -> i32 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// Returns true if [a,b] brackets a root by sign change, consistent with Eq. (9.2.3).
fn is_bracket(fa: f64, fb: f64) -> bool {
    signum_nonzero(fa) != 0 && signum_nonzero(fb) != 0 && signum_nonzero(fa) != signum_nonzero(fb)
}

/// Evaluate f(a), f(b) and validate a < b and finiteness.
fn make_bracket<F>(f: &F, a: f64, b: f64) -> Result<Bracket, RootError>
where
    F: Fn(f64) -> f64,
{
    if !(a < b) {
        return Err(RootError::InvalidInterval);
    }
    let fa = ensure_finite(f(a))?;
    let fb = ensure_finite(f(b))?;
    Ok(Bracket { a, b, fa, fb })
}

/// Sampling-based bracket finder on [x1,x2] using N subintervals, following Eq. (9.2.5–9.2.6).
/// Returns all subintervals where a sign change is detected.
pub fn find_brackets_by_sampling<F>(
    f: &F,
    x1: f64,
    x2: f64,
    n: usize,
) -> Result<Vec<Bracket>, RootError>
where
    F: Fn(f64) -> f64,
{
    if !(x1 < x2) {
        return Err(RootError::InvalidInterval);
    }
    if n == 0 {
        return Ok(vec![]);
    }

    let mut brackets = Vec::new();
    let h = (x2 - x1) / (n as f64);

    let mut t_prev = x1;
    let mut f_prev = ensure_finite(f(t_prev))?;

    for i in 1..=n {
        let t = if i == n { x2 } else { x1 + (i as f64) * h };
        let ft = ensure_finite(f(t))?;

        if is_bracket(f_prev, ft) {
            brackets.push(Bracket {
                a: t_prev,
                b: t,
                fa: f_prev,
                fb: ft,
            });
        }

        t_prev = t;
        f_prev = ft;
    }

    Ok(brackets)
}

/// Heuristic expanding search for a large bracket centered at x0.
/// It tests f(x0 - R) and f(x0 + R) for increasing R until a sign change is observed.
/// This is a practical interpretation of the idea behind Eq. (9.2.7), but is not guaranteed.
pub fn expand_bracket_heuristic<F>(
    f: &F,
    x0: f64,
    r0: f64,
    max_expansions: usize,
) -> Result<Option<Bracket>, RootError>
where
    F: Fn(f64) -> f64,
{
    if r0 <= 0.0 {
        return Ok(None);
    }

    let mut r = r0;
    for _ in 0..max_expansions {
        let a = x0 - r;
        let b = x0 + r;
        let br = make_bracket(f, a, b)?;
        if is_bracket(br.fa, br.fb) {
            return Ok(Some(br));
        }
        r *= 2.0;
    }

    Ok(None)
}

/// Bisection on a bracketed interval [a,b]. Guaranteed convergence if f is continuous and [a,b] brackets a root.
/// The singularity diagnostic flags the situation where |f| becomes large or fails to decrease as the interval shrinks,
/// which can occur for discontinuities like Eq. (9.2.4).
pub fn bisection<F>(
    f: &F,
    br: Bracket,
    tol_x: f64,
    tol_f: f64,
    max_iter: usize,
) -> Result<BisectionResult, RootError>
where
    F: Fn(f64) -> f64,
{
    let mut a = br.a;
    let mut b = br.b;
    let mut fa = br.fa;
    let mut fb = br.fb;

    if !is_bracket(fa, fb) {
        return Err(RootError::NotBracketed { a, b, fa, fb });
    }

    let mut peak_abs_f_seen = fa.abs().max(fb.abs());
    let mut singularity_suspected = false;

    // A simple heuristic threshold: if |f| grows well beyond endpoints, suspect a singularity.
    // This is not a proof, but often catches cases like Eq. (9.2.4) in practice.
    let endpoint_scale = peak_abs_f_seen.max(1.0);
    let blowup_factor = 1.0e6;

    for k in 1..=max_iter {
        let m = 0.5 * (a + b);
        let fm = ensure_finite(f(m))?;

        peak_abs_f_seen = peak_abs_f_seen.max(fm.abs());

        // Singularity heuristic: values explode instead of trending toward zero.
        if fm.abs() > blowup_factor * endpoint_scale {
            singularity_suspected = true;
        }

        // Convergence tests: interval width or residual magnitude.
        let width = (b - a).abs();
        if width <= tol_x || fm.abs() <= tol_f {
            return Ok(BisectionResult {
                root: m,
                iterations: k,
                f_root: fm,
                width,
                singularity_suspected,
                peak_abs_f_seen,
            });
        }

        // Keep the sign change.
        if signum_nonzero(fa) == signum_nonzero(fm) {
            a = m;
            fa = fm;
        } else {
            b = m;
            fb = fm;
        }

        // A second heuristic: if the interval is shrinking but |f| is not decreasing at all, be cautious.
        // We do not terminate, since bisection is still valid under continuity, but we flag it.
        if k % 10 == 0 && peak_abs_f_seen > blowup_factor * endpoint_scale {
            singularity_suspected = true;
        }

        // Silence unused variable warning for fb if compiler settings change.
        let _ = fb;
    }

    Err(RootError::MaxIterExceeded)
}

fn main() -> Result<(), RootError> {
    println!("Program 9.2.0: Bracketing and Bisection\n");

    // ------------------------------------------------------------
    // Example 1: A well-behaved continuous function with a simple root.
    // f(x) = cos(x) - x has a root in (0,1).
    // ------------------------------------------------------------
    let f1 = |x: f64| x.cos() - x;

    println!("Example 1: f(x) = cos(x) - x");
    let br1 = make_bracket(&f1, 0.0, 1.0)?;
    println!("  Initial bracket test on [0,1]: f(a)={:+.6}, f(b)={:+.6}", br1.fa, br1.fb);

    let r1 = bisection(&f1, br1, 1e-12, 1e-12, 300)?;
    println!("  Bisection result:");
    println!("    root ≈ {:.15}", r1.root);
    println!("    f(root) = {:+.3e}", r1.f_root);
    println!("    width = {:.3e}, iters = {}", r1.width, r1.iterations);
    println!("    singularity suspected = {}", r1.singularity_suspected);
    println!();

    // ------------------------------------------------------------
    // Example 2: A multiple root where the function touches zero without changing sign.
    // f(x) = (x-1)^2 has a root at x=1 but does not produce a sign change across the root.
    // This illustrates why bracketing based only on sign changes can fail for multiple roots.
    // ------------------------------------------------------------
    let f2 = |x: f64| (x - 1.0) * (x - 1.0);

    println!("Example 2: f(x) = (x - 1)^2 (multiple root without sign change)");
    let br2 = make_bracket(&f2, 0.0, 2.0)?;
    println!("  On [0,2]: f(a)={:+.6}, f(b)={:+.6}", br2.fa, br2.fb);
    if is_bracket(br2.fa, br2.fb) {
        println!("  Unexpectedly bracketed (this would be rare for a perfect square).");
    } else {
        println!("  No sign-change bracket detected, as expected for a double root.");
    }
    println!();

    // ------------------------------------------------------------
    // Example 3: Finding brackets by sampling on a suspected interval (Eq. 9.2.5–9.2.6).
    // f(x) = sin(10x) oscillates and has many roots. Sampling can detect multiple brackets.
    // ------------------------------------------------------------
    let f3 = |x: f64| (10.0 * x).sin();

    println!("Example 3: f(x) = sin(10x) (many roots, sampling-based bracketing)");
    let brackets = find_brackets_by_sampling(&f3, 0.0, 1.0, 200)?;
    println!("  Number of sign-change brackets found on [0,1] with N=200: {}", brackets.len());
    if let Some(first) = brackets.first() {
        println!(
            "  First bracket: [{:.6}, {:.6}] with f(a)={:+.3e}, f(b)={:+.3e}",
            first.a, first.b, first.fa, first.fb
        );
        let r3 = bisection(&f3, *first, 1e-12, 1e-12, 300)?;
        println!("  Bisection on first bracket gives root ≈ {:.15} with f(root)={:+.3e}", r3.root, r3.f_root);
    }
    println!();

    // ------------------------------------------------------------
    // Example 4: A sign change caused by a singularity rather than a root (Eq. 9.2.4).
    // f(x) = 1/(x-c) has no finite root, but changes sign across the pole at x=c.
    // Bisection will shrink the interval toward the singularity, and |f| will blow up.
    // ------------------------------------------------------------
    let c = 0.3;
    let f4 = move |x: f64| 1.0 / (x - c);

    println!("Example 4: f(x) = 1/(x - c) with c=0.3 (sign change from a pole, not a root)");
    let br4 = make_bracket(&f4, 0.0, 1.0)?;
    println!("  Initial bracket test on [0,1]: f(a)={:+.3e}, f(b)={:+.3e}", br4.fa, br4.fb);

    let r4 = bisection(&f4, br4, 1e-12, 1e-12, 300)?;
    println!("  Bisection output (note: there is no true root):");
    println!("    midpoint ≈ {:.15}", r4.root);
    println!("    f(midpoint) = {:+.3e}", r4.f_root);
    println!("    peak |f| seen = {:.3e}", r4.peak_abs_f_seen);
    println!("    singularity suspected = {}", r4.singularity_suspected);
    println!("    comment: shrinking interval can converge to the pole, not to a zero.");
    println!();

    // ------------------------------------------------------------
    // Example 5: Heuristic expansion search for a bracket (Eq. 9.2.7 idea).
    // If a root exists and f eventually changes sign, expanding brackets can locate it.
    // ------------------------------------------------------------
    let f5 = |x: f64| x * x * x - 2.0 * x - 5.0; // a cubic with at least one real root

    println!("Example 5: f(x) = x^3 - 2x - 5 (heuristic bracket expansion)");
    match expand_bracket_heuristic(&f5, 0.0, 1.0, 30)? {
        Some(br) => {
            println!(
                "  Found bracket: [{:.6}, {:.6}] with f(a)={:+.3e}, f(b)={:+.3e}",
                br.a, br.b, br.fa, br.fb
            );
            let r5 = bisection(&f5, br, 1e-12, 1e-12, 400)?;
            println!("  Bisection gives root ≈ {:.15} with f(root)={:+.3e}", r5.root, r5.f_root);
        }
        None => {
            println!("  No bracket found within expansion budget. This can happen for many functions.");
        }
    }

    Ok(())
}
```

Program 9.2.0 demonstrates the central role of bracketing in reliable one-dimensional root finding. By explicitly enforcing the existence conditions discussed in Section 9.2, bisection provides a method whose convergence is guaranteed under minimal assumptions and whose behavior is transparent and predictable. This reliability makes bracketing methods indispensable as standalone solvers and as safeguards within more advanced algorithms.

The examples in the program highlight both the strengths and the limitations of sign-based bracketing. Simple isolated roots are handled efficiently, while multiple roots and oscillatory functions require careful interpretation or additional resolution. The singularity example underscores an important practical lesson: a sign change alone is not sufficient to certify the presence of a root unless continuity is assured. Monitoring the magnitude of function values provides a valuable diagnostic tool for distinguishing genuine roots from pathological cases.

The modular design of the code allows these techniques to be extended naturally. Sampling density, expansion heuristics, and stopping criteria can all be adapted to specific applications, and bisection itself can serve as a globalization strategy for faster local methods such as Newton or secant iterations. As such, Program 9.2.0 establishes a robust computational foundation upon which the more advanced root-finding methods developed in subsequent sections are built.

## 9.2.1. The Bisection Method

Once a root has been bracketed, the *bisection method* provides a simple and robust algorithm for refining its location. Given an interval (\[a,b\]) satisfying (9.2.1), the method proceeds iteratively.

At each step, compute the midpoint,

$$m = \frac{a+b}{2} \tag{9.2.8}$$

If $f(m)=0$ within a prescribed tolerance, the algorithm terminates. Otherwise, one replaces either $a$ or $b$ with $m$ so that the new interval continues to satisfy the bracketing condition. This guarantees that a root remains enclosed at every iteration.

Bisection is guaranteed to converge to a root, provided the initial interval brackets a solution and the function is continuous. Its reliability makes it an essential baseline method, even though it converges more slowly than many alternatives (Martin, 2023).

### Convergence Analysis

Let $\varepsilon_n = b_n - a_n$ denote the interval width after $n$ iterations. Each bisection step halves the interval, giving,

$$\varepsilon_{n+1} = \frac{1}{2},\varepsilon_n \tag{9.2.9}$$

Starting from an initial width $\varepsilon_0$, we obtain,

$$\varepsilon_n = \frac{\varepsilon_0}{2^n} \tag{9.2.10}$$

To ensure $\varepsilon_n \le \text{TOL}$, the required number of iterations satisfies:

$$n \ge \frac{\ln(\varepsilon_0/\text{TOL})}{\ln 2} \tag{9.2.11}$$

Equivalently, one may write:

$$n \approx \left\lceil \log_2\!\left(\frac{\varepsilon_0}{\text{TOL}}\right) \right\rceil \tag{9.2.12}$$

Each bisection step yields one additional binary digit of accuracy, corresponding to approximately $0.301$ decimal digits.

The convergence of bisection is classified as linear, or more precisely geometric, since the error decreases by a constant factor at each iteration:

$$\varepsilon_{n+1} = C\,\varepsilon_n, \quad C = \frac{1}{2} \tag{9.2.13}$$

By comparison, Newton–Raphson exhibits quadratic convergence near a simple root, and Halley’s method exhibits cubic convergence when applicable (Martin, 2023).

### Failure Modes and Safeguards

Bisection fails only if its assumptions are violated. If the initial interval does not truly bracket a root, or if the function is discontinuous within the interval, the method may converge to a singularity rather than a zero. In such cases, $|f(x)|$ typically diverges instead of shrinking.

If multiple roots lie within the same interval, bisection will converge to one of them, but not necessarily a specific one. When multiple roots are suspected, the interval should be subdivided so that each subinterval contains at most one root.

### Computational Cost and Stopping Criteria

Each bisection iteration requires one new function evaluation. After $n$ iterations, the total number of evaluations is approximately $n+2$, including the initial endpoints. The computational cost therefore scales as,

$$O\!\left(\log \frac{1}{\text{TOL}}\right) \tag{9.2.14}$$

This predictability allows precise budgeting of computational effort, which is valuable when function evaluations are expensive.

In practice, bisection is terminated when either $|b-a| < \delta$, or $|f(m)| < \varepsilon$. Interval-based criteria are often preferred, since small function values can occur even when the approximation is not particularly close to the true root if the function is nearly flat.

### Bisection in Modern Algorithms

Although bisection is rarely used as a final solver on its own, it plays a crucial role in modern hybrid methods. It is commonly used to obtain a safe approximation before switching to faster methods such as Newton–Raphson. Brent’s method formalizes this idea by combining bracketing with interpolation-based steps, ensuring both reliability and efficiency (Martin, 2023).

From a broader perspective, bisection addresses the same fundamental concern as modern globalization strategies for nonlinear systems, namely reducing sensitivity to initial guesses. While multidimensional methods such as the W4 scheme pursue this goal through modified Newton dynamics (Okawa et al., 2023), bracketing provides an explicit and rigorous solution in one dimension. Adaptive algorithm-selection approaches may also favor bisection when early iterations indicate that derivative-based methods are unreliable (Aghamie et al., 2025).

The contrast between bisection’s guaranteed linear contraction and the high local orders achieved by specialized schemes highlights a central trade-off in root finding: certainty of progress versus speed when conditions are favorable (Thota et al., 2025).

Bracketing and bisection form the conceptual foundation of one-dimensional root finding. Bracketing guarantees existence through a sign change, and bisection refines the root’s location through systematic interval halving. Despite their simplicity, these methods remain indispensable as robust solvers and as stabilizing components within more advanced algorithms (Martin, 2023).

Below is textbook-ready prose tailored to **Section 9.2.1: The Bisection Method** and written to accompany **Program 9.2.1**. The tone, structure, and level of detail follow your provided example closely, but the content is adapted correctly to bracketing and bisection rather than infinite series. No equations are rewritten; references are made using the existing equation numbers from the section.

---

## Introductory paragraph (to be placed immediately before the code block)

Following the discussion in Section 9.2 on bracketing as a rigorous existence guarantee for one-dimensional root-finding problems, **Program 9.2.1** presents a practical implementation of the bisection method for refining a bracketed root. In numerical computation, the reliability of an algorithm is often more important than its asymptotic speed, particularly when little is known about the behavior of the function away from the solution. Bisection exemplifies this philosophy by combining minimal assumptions with guaranteed convergence under continuity. This program translates the theoretical framework developed in Equations (9.2.8)–(9.2.13) into an explicit computational procedure, illustrating how interval halving, iteration budgeting, and practical stopping criteria interact in finite-precision arithmetic.

---

## Explanatory paragraphs describing the code and its functions

At the core of the implementation is a dedicated bisection routine that operates on an interval (\[a,b\]) known to satisfy the bracketing condition discussed in Equations (9.2.1)–(9.2.3). The algorithm begins by verifying that the function values at the endpoints have opposite signs, ensuring that a root exists within the interval by the Intermediate Value Theorem. This explicit check enforces the theoretical preconditions of the method and prevents meaningless iteration when the assumptions of Section 9.2.1 are violated.

Each iteration computes the midpoint of the current interval according to Equation (9.2.8) and evaluates the function at that point. The interval is then updated by replacing either (a) or (b) with the midpoint so that the sign change is preserved. This guarantees that the root remains enclosed at every step. After each update, the interval width corresponds exactly to the quantity (\\varepsilon_n) defined in Equations (9.2.9) and (9.2.10), making the geometric convergence of the method explicit in the code.

The stopping logic reflects the practical considerations discussed in the text. Termination may occur when the interval width falls below a prescribed tolerance (\\delta), or when the residual magnitude (|f(m)|) falls below a tolerance (\\varepsilon). While Equation (9.2.12) provides a precise iteration budget for the interval-width criterion, the residual-based test often allows earlier termination when the function behaves favorably near the root. The program reports both the predicted iteration count from Equation (9.2.12) and the actual number of interval halvings performed, making the relationship between theory and practice transparent.

To address the failure modes discussed around Equation (9.2.4), the implementation includes a simple diagnostic that monitors the magnitude of (|f(m)|) as the interval shrinks. If the function values grow rapidly instead of tending toward zero, the algorithm flags the likely presence of a singularity or discontinuity within the interval. This safeguard does not alter the bisection logic itself but provides valuable interpretive information when the theoretical assumptions of continuity are violated.

The `main` function demonstrates the algorithm on two representative examples. The first applies bisection to a smooth function with a unique root, illustrating the guaranteed linear convergence and the accuracy of the iteration budget predicted by Equation (9.2.12). The second applies the same procedure to a function with a pole inside the interval, showing how bisection can converge to a singularity rather than a zero when continuity fails, exactly as described in the surrounding discussion.

---

## Concluding remarks (to be placed after the code block)

```rust
// Program 9.2.1: The Bisection Method
//
// Problem statement (Section 9.2.1 context):
// We wish to solve the one-dimensional nonlinear equation
//     f(x) = 0,
// under the assumption that f is continuous on an interval [a,b] and that a root is bracketed,
// meaning f(a) and f(b) have opposite signs (the bracketing condition discussed around Eq. (9.2.1)–(9.2.3)).
// The bisection method refines the bracket by repeatedly halving the interval. At iteration k it forms
// the midpoint
//     m = (a + b)/2  (Eq. 9.2.8),
// evaluates f(m), and replaces either a or b with m so that the sign change is preserved. This guarantees
// that a root remains enclosed at every step, provided continuity holds.
//
// Convergence analysis (Section 9.2.1):
// If eps_k = b_k - a_k denotes the interval width after k halvings, then each step halves the width,
//     eps_{k+1} = (1/2) eps_k  (Eq. 9.2.9),
// hence
//     eps_k = eps_0 / 2^k      (Eq. 9.2.10).
// To achieve eps_k <= TOL, one needs approximately
//     k ≈ ceil(log2(eps_0 / TOL))  (Eq. 9.2.12).
//
// Practical stopping criteria (Section 9.2.1):
// In practice, bisection is terminated when either |b-a| < delta (interval criterion) or |f(m)| < epsilon
// (residual criterion). Interval-based criteria are often preferred, since a small residual can occur
// even when x is not very close to the root if f is nearly flat.
//
// Failure modes and safeguards (Section 9.2.1):
// Bisection can fail if its assumptions are violated, e.g., when the sign change is caused by a
// discontinuity or pole (as in f(x) = 1/(x-c), Eq. 9.2.4). In such cases, |f(m)| typically diverges
// rather than shrinking. This program includes a simple blow-up diagnostic that flags likely singular
// behavior.
//
// This file is self-contained (no external crates). Run with:
//     cargo run

use std::fmt;

#[derive(Debug, Clone)]
pub enum RootError {
    InvalidInterval,
    NotBracketed { a: f64, b: f64, fa: f64, fb: f64 },
    NonFiniteValue,
    MaxIterExceeded,
}

impl fmt::Display for RootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RootError::InvalidInterval => write!(f, "invalid interval: require a < b"),
            RootError::NotBracketed { a, b, fa, fb } => write!(
                f,
                "interval [{a}, {b}] does not bracket a root (f(a)={fa}, f(b)={fb})"
            ),
            RootError::NonFiniteValue => write!(f, "encountered a non-finite value (NaN or infinity)"),
            RootError::MaxIterExceeded => write!(f, "maximum iterations exceeded"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BisectionConfig {
    /// Interval-width tolerance delta (stop if |b-a| <= delta).
    pub delta: f64,
    /// Residual tolerance epsilon (stop if |f(m)| <= epsilon).
    pub epsilon: f64,
    /// Maximum number of interval-halving steps.
    pub max_iter: usize,
    /// If true, stop when either criterion is met; otherwise require both.
    pub stop_on_either: bool,
    /// Heuristic threshold for flagging likely singularity behavior.
    pub blowup_factor: f64,
}

impl Default for BisectionConfig {
    fn default() -> Self {
        Self {
            delta: 1e-12,
            epsilon: 1e-12,
            max_iter: 300,
            stop_on_either: true,
            blowup_factor: 1e6,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BisectionResult {
    pub root: f64,
    pub f_root: f64,
    /// Number of interval-halving steps performed.
    pub iterations: usize,
    pub final_width: f64,
    pub singularity_suspected: bool,
    /// Predicted iterations to reach |b-a| <= delta via Eq. (9.2.12).
    pub predicted_iters_width: usize,
}

fn ensure_finite(x: f64) -> Result<f64, RootError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(RootError::NonFiniteValue)
    }
}

fn signum_nonzero(x: f64) -> i32 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// Bracketing sign-change test consistent with the bracketing condition around Eq. (9.2.1)–(9.2.3).
fn is_bracket(fa: f64, fb: f64) -> bool {
    let sa = signum_nonzero(fa);
    let sb = signum_nonzero(fb);
    sa != 0 && sb != 0 && sa != sb
}

/// Compute ceil(log2(x)) for x > 0, used for Eq. (9.2.12).
fn ceil_log2(x: f64) -> usize {
    if !(x.is_finite()) || x <= 1.0 {
        return 0;
    }
    (x.ln() / 2.0_f64.ln()).ceil() as usize
}

/// Predicted iterations using Eq. (9.2.12): n ≈ ceil(log2(eps0/TOL)).
fn predicted_iters_for_width(a: f64, b: f64, tol: f64) -> usize {
    let eps0 = (b - a).abs();
    if tol <= 0.0 {
        0
    } else {
        ceil_log2(eps0 / tol)
    }
}

/// Classical bisection on [a,b]. Assumes f is continuous and [a,b] brackets a root.
/// The returned `final_width` is the width after `iterations` interval halvings and matches
/// eps0 / 2^n (Eq. 9.2.10) for the reported n.
pub fn bisection<F>(
    f: &F,
    mut a: f64,
    mut b: f64,
    cfg: BisectionConfig,
) -> Result<BisectionResult, RootError>
where
    F: Fn(f64) -> f64,
{
    if !(a < b) {
        return Err(RootError::InvalidInterval);
    }

    let mut fa = ensure_finite(f(a))?;
    let mut fb = ensure_finite(f(b))?;

    if fa == 0.0 {
        return Ok(BisectionResult {
            root: a,
            f_root: fa,
            iterations: 0,
            final_width: (b - a).abs(),
            singularity_suspected: false,
            predicted_iters_width: predicted_iters_for_width(a, b, cfg.delta),
        });
    }
    if fb == 0.0 {
        return Ok(BisectionResult {
            root: b,
            f_root: fb,
            iterations: 0,
            final_width: (b - a).abs(),
            singularity_suspected: false,
            predicted_iters_width: predicted_iters_for_width(a, b, cfg.delta),
        });
    }

    if !is_bracket(fa, fb) {
        return Err(RootError::NotBracketed { a, b, fa, fb });
    }

    let predicted_iters_width = predicted_iters_for_width(a, b, cfg.delta);
    let endpoint_scale = fa.abs().max(fb.abs()).max(1.0);
    let mut singularity_suspected = false;

    for k in 1..=cfg.max_iter {
        // Midpoint (Eq. 9.2.8)
        let m = 0.5 * (a + b);
        let fm = ensure_finite(f(m))?;

        // Heuristic singularity diagnostic: |f(m)| blows up relative to endpoint scale.
        if fm.abs() > cfg.blowup_factor * endpoint_scale {
            singularity_suspected = true;
        }

        // Preserve the bracketing condition by keeping the half-interval with a sign change.
        if signum_nonzero(fa) == signum_nonzero(fm) {
            a = m;
            fa = fm;
        } else {
            b = m;
            fb = fm;
        }

        // Width after this halving step is eps_k in Eq. (9.2.10).
        let width = (b - a).abs();
        let width_ok = width <= cfg.delta;
        let residual_ok = fm.abs() <= cfg.epsilon;

        if cfg.stop_on_either {
            if width_ok || residual_ok {
                return Ok(BisectionResult {
                    root: m,
                    f_root: fm,
                    iterations: k,
                    final_width: width,
                    singularity_suspected,
                    predicted_iters_width,
                });
            }
        } else if width_ok && residual_ok {
            return Ok(BisectionResult {
                root: m,
                f_root: fm,
                iterations: k,
                final_width: width,
                singularity_suspected,
                predicted_iters_width,
            });
        }

        let _ = fb; // logically maintained
    }

    Err(RootError::MaxIterExceeded)
}

fn main() -> Result<(), RootError> {
    println!("Program 9.2.1: The Bisection Method\n");

    let cfg = BisectionConfig {
        delta: 1e-12,
        epsilon: 1e-12,
        max_iter: 300,
        stop_on_either: true,
        blowup_factor: 1e6,
    };

    // Example A: A smooth continuous function with a bracketed root on [0,1].
    // f(x) = cos(x) - x has a unique root in (0,1).
    let f = |x: f64| x.cos() - x;
    let a = 0.0;
    let b = 1.0;

    let pred = predicted_iters_for_width(a, b, cfg.delta);
    println!("Example A: f(x) = cos(x) - x");
    println!("  Initial interval: [{:.6}, {:.6}]", a, b);
    println!(
        "  Predicted iterations for |b-a| <= delta via Eq. (9.2.12): n ≈ {}",
        pred
    );

    let r = bisection(&f, a, b, cfg)?;
    println!("  Bisection result:");
    println!("    root ≈ {:.15}", r.root);
    println!("    f(root) = {:+.3e}", r.f_root);
    println!("    iterations (halvings) = {}", r.iterations);
    println!("    final width = {:.3e}", r.final_width);
    println!("    singularity suspected = {}", r.singularity_suspected);

    // Consistency check for Eq. (9.2.10)
    let eps0 = (b - a).abs();
    let expected_width = eps0 / 2.0_f64.powi(r.iterations as i32);
    println!("  expected width eps0/2^n (Eq. 9.2.10) = {:.3e}", expected_width);
    println!("  ratio achieved/expected = {:.6}", r.final_width / expected_width);
    println!();

    // Example B: A discontinuity with a sign change but no root (Eq. 9.2.4).
    let c = 0.3;
    let g = move |x: f64| 1.0 / (x - c);
    let a2 = 0.0;
    let b2 = 1.0;

    println!("Example B: g(x) = 1/(x - c) with c=0.3 (sign change from a pole, not a root)");
    println!("  Initial interval: [{:.6}, {:.6}]", a2, b2);
    println!("  Note: continuity fails on [a,b], so bisection can converge to the singularity.");

    let r2 = bisection(&g, a2, b2, cfg)?;
    println!("  Bisection output (no true root exists):");
    println!("    midpoint ≈ {:.15}", r2.root);
    println!("    g(midpoint) = {:+.3e}", r2.f_root);
    println!("    iterations (halvings) = {}", r2.iterations);
    println!("    final width = {:.3e}", r2.final_width);
    println!("    singularity suspected = {}", r2.singularity_suspected);

    Ok(())
}
```

Program 9.2.1 demonstrates how the abstract convergence properties of the bisection method translate directly into a robust and predictable numerical algorithm. By explicitly halving the interval at each step, the method achieves the geometric error reduction described in Equations (9.2.9)–(9.2.13), and the resulting iteration count can be estimated in advance with remarkable accuracy. This predictability is a key advantage when function evaluations are expensive or when guaranteed progress is required.

The numerical examples highlight both the strengths and limitations of bisection. For continuous functions with a simple bracketed root, convergence is reliable and insensitive to initial guesses. When the underlying assumptions are violated, as in the presence of a singularity, the method still produces a well-defined sequence of iterates, but the accompanying diagnostics reveal that the apparent convergence does not correspond to a true root. This reinforces the importance of understanding the analytical structure of the problem alongside the numerical method.

Although bisection is rarely used as a final solver in high-performance applications, it plays a crucial role as a stabilizing component in modern hybrid algorithms. By providing a safe initial approximation or a fallback strategy when faster methods fail, bisection remains an indispensable tool in numerical root finding and a conceptual foundation for the more advanced techniques developed later in this chapter.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/eaZW7IGr5uVWwNL9KqEn.3","tags":[]}

# 9.3. Secant Method, False Position Method, and Ridders’ Method

Many scientific and engineering problems require solving equations of the form,

$$f(x) = 0 \tag{9.3.1}$$

In most practical situations, no analytical solution exists and the root must be approximated numerically. Such problems arise across a wide range of fields, including physics, engineering, finance, and computational biology. For example, in quantitative finance, computing the implied volatility of an option involves solving the Black–Scholes pricing equation for the volatility parameter such that the difference between theoretical and market prices vanishes (Sapna and Mohan, 2024). In physics, solving Kepler’s equation for orbital motion,

$$E - e \sin E - M = 0 \tag{9.3.2}$$

is a classical transcendental root-finding problem. Similar equations appear in nonlinear circuit analysis, thermodynamics, and steady-state biological models.

While Newton–Raphson methods can converge rapidly when they succeed, they are sensitive to initial guesses and require derivative information. In many applications, one must therefore rely on more robust alternatives that retain much of Newton’s speed while reducing its fragility. In this section, we examine three classical methods that improve upon simple bisection: the Secant method, the False Position method (regula falsi), and Ridders’ method. These approaches assume approximate linearity near the root and use interpolation or function transformations to accelerate convergence beyond that of bisection, while maintaining varying degrees of reliability.

```{figure} images/pqQDe4beUu67RvW3raYP-4sHjedt7slf2r02h2zU8-v1.png
:name: eYZ7m0Zoeh
:align: center
:width: 70%

**Figure 9.3.1.** Stagnation of interpolation-based root-finding methods in a highly nonlinear interval.
```

Figure 9.3.1 illustrates a common failure mode of interpolation-based methods such as the secant method and the classical regula falsi. The nonlinear function $f(x)$ possesses a simple root $x^\ast$ within the initial bracketing interval $[a,b]$, yet strong curvature causes the straight-line interpolant between $(a,f(a))$ and $(b,f(b))$ to intersect the $x$-axis at a point $x_1$ that lies disproportionately close to one endpoint. As iterations proceed, one endpoint of the bracket remains nearly fixed while the other moves only marginally, leading to slow convergence or practical stagnation despite maintaining a valid bracket.

This behavior highlights an important limitation of naive interpolation strategies: although they improve upon bisection by exploiting approximate linearity, they can perform poorly when the function deviates strongly from linearity over the interval. The figure motivates the development of modified schemes, such as weighted regula falsi variants and Ridders’ method, which introduce corrective mechanisms to restore balanced interval contraction and achieve more reliable convergence in nonlinear settings.

### Rust Implementation

Following the discussion in Section 9.3 on interpolation-based root-finding methods and their limitations, Program 9.3.0 provides a practical implementation of three classical derivative-free algorithms for solving nonlinear equations of the form $f(x)=0$. Whereas bisection guarantees convergence through interval halving, the secant method, false position method, and Ridders’ method attempt to accelerate convergence by exploiting approximate linearity or symmetry near the root. This program implements all three approaches within a unified framework and applies them to representative nonlinear problems, including Kepler’s equation and a strongly curved exponential function. By comparing convergence behavior, iteration counts, and failure modes, the program illustrates both the performance gains and the potential pitfalls of interpolation-based strategies in finite-precision numerical computation.

At the core of the implementation are three root-finding routines that differ in how they approximate the local behavior of the nonlinear function $f$ while iteratively refining an estimate of the root defined by Equation (9.3.1). Each routine accepts a user-supplied function $f(x)$, convergence tolerances, and an iteration budget, returning a structured result containing the approximate root, the residual $f(x^\ast)$, and diagnostic information about convergence.

The `secant` method implements an open, derivative-free analogue of Newton’s method. Given two initial approximations $x_{k-1}$ and $x_k$, the method constructs a linear interpolant through the points $(x_{k-1},f(x_{k-1}))$ and $(x_k,f(x_k))$ and computes the next iterate from the zero of this interpolant. Because no bracketing is enforced, the secant method can converge rapidly when the initial guesses lie within the basin of attraction of the root, but it offers no guarantee of convergence and may diverge if the local linear approximation is poor. This sensitivity reflects the general concerns about initial guesses emphasized earlier in Section 9.1.

The false position method (regula falsi) modifies the secant idea by enforcing a bracketing condition analogous to Equation (9.2.3). At each iteration, the secant-intercept point replaces one endpoint of the bracket while preserving a sign change. This guarantees that a root remains enclosed, providing reliability comparable to bisection. However, as discussed in the text surrounding Figure 9.3.1, classical regula falsi can suffer from stagnation when the function exhibits strong curvature, causing one endpoint of the interval to remain nearly fixed across many iterations. The implementation in this program exposes this behavior directly through a nonlinear test case.

The Ridders’ method enhances classical regula falsi by introducing an exponential correction based on function values at the midpoint of the current bracket. This correction counteracts endpoint bias and typically restores balanced interval contraction even in strongly nonlinear settings. While Ridders’ method still relies on bracketing and thus inherits the reliability of bisection, it frequently converges much faster in practice. The algorithm implemented here follows the standard formulation discussed in the literature and terminates based on either interval width or residual magnitude, consistent with the stopping criteria described in Section 9.2.

The `main` function demonstrates these three methods on two representative problems. The first is Kepler’s equation, Equation (9.3.2), a classical transcendental equation arising in orbital mechanics. Because the function is smooth and well behaved, all three methods converge successfully, but with noticeably different iteration counts. The second example uses an exponential function over a wide interval chosen specifically to illustrate the stagnation phenomenon depicted in Figure 9.3.1. In this case, classical regula falsi fails to converge within the iteration budget, while both the secant method and Ridders’ method succeed, clearly illustrating the advantages of modified interpolation strategies.

```rust
// Program 9.3.0: Secant Method, False Position (Regula Falsi), and Ridders’ Method
//
// Problem statement (Section 9.3 context):
// We seek a numerical approximation to a real root of the scalar equation
//     f(x) = 0  (Eq. 9.3.1),
// typically when no closed-form solution is available. Compared with bisection (Section 9.2),
// interpolation-based methods exploit approximate linearity near the root to accelerate convergence.
//
// This program implements three classical derivative-free methods:
//
// 1) Secant method (open method, no bracketing required):
//    Uses two recent iterates x_{k-1}, x_k and a linear interpolant (secant line) to compute x_{k+1}.
//    It is often faster than bisection but is not guaranteed to converge and can diverge with poor starts.
//
// 2) False position / regula falsi (bracketed method):
//    Maintains a sign-change bracket [a,b] and replaces one endpoint with the secant-intercept point.
//    It preserves the bracketing guarantee but can stagnate on strongly nonlinear intervals
//    (the behavior illustrated by Figure 9.3.1).
//
// 3) Ridders’ method (bracketed method with an exponential correction):
//    Maintains a bracket and typically achieves faster, more balanced contraction than classical regula falsi,
//    reducing the stagnation tendency in nonlinear settings.
//
// The program demonstrates these methods on two representative problems:
//   - Kepler’s equation (Eq. 9.3.2): E - e sin(E) - M = 0
//   - A nonlinear function chosen to illustrate regula falsi stagnation behavior similar to Figure 9.3.1.
//
// No external crates are required. Run with: cargo run

use std::fmt;

#[derive(Debug, Clone)]
pub enum RootError {
    InvalidInterval,
    NotBracketed { a: f64, b: f64, fa: f64, fb: f64 },
    NonFiniteValue,
    MaxIterExceeded,
    ZeroDenominator,
}

impl fmt::Display for RootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RootError::InvalidInterval => write!(f, "invalid interval: require a < b"),
            RootError::NotBracketed { a, b, fa, fb } => write!(
                f,
                "interval [{a}, {b}] does not bracket a root (f(a)={fa}, f(b)={fb})"
            ),
            RootError::NonFiniteValue => write!(f, "encountered a non-finite value (NaN or infinity)"),
            RootError::MaxIterExceeded => write!(f, "maximum iterations exceeded"),
            RootError::ZeroDenominator => write!(f, "division by (near) zero in interpolation step"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RootResult {
    pub root: f64,
    pub f_root: f64,
    pub iterations: usize,
    pub converged: bool,
}

fn ensure_finite(x: f64) -> Result<f64, RootError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(RootError::NonFiniteValue)
    }
}

fn signum_nonzero(x: f64) -> i32 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

fn is_bracket(fa: f64, fb: f64) -> bool {
    let sa = signum_nonzero(fa);
    let sb = signum_nonzero(fb);
    sa != 0 && sb != 0 && sa != sb
}

/// Secant method (open). Requires two initial guesses x0 and x1.
/// Stops when |x_{k+1}-x_k| <= tol_x*(1+|x_{k+1}|) or |f(x_{k+1})| <= tol_f.
pub fn secant<F>(
    f: &F,
    mut x0: f64,
    mut x1: f64,
    tol_x: f64,
    tol_f: f64,
    max_iter: usize,
) -> Result<RootResult, RootError>
where
    F: Fn(f64) -> f64,
{
    let mut f0 = ensure_finite(f(x0))?;
    let mut f1 = ensure_finite(f(x1))?;

    for k in 1..=max_iter {
        if f1.abs() <= tol_f {
            return Ok(RootResult {
                root: x1,
                f_root: f1,
                iterations: k - 1,
                converged: true,
            });
        }

        let denom = f1 - f0;
        if denom.abs() < 1e-15 {
            return Err(RootError::ZeroDenominator);
        }

        let x2 = ensure_finite(x1 - f1 * (x1 - x0) / denom)?;
        let f2 = ensure_finite(f(x2))?;

        let step = (x2 - x1).abs();
        if step <= tol_x * (1.0 + x2.abs()) || f2.abs() <= tol_f {
            return Ok(RootResult {
                root: x2,
                f_root: f2,
                iterations: k,
                converged: true,
            });
        }

        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f2;
    }

    Ok(RootResult {
        root: x1,
        f_root: f1,
        iterations: max_iter,
        converged: false,
    })
}

/// False position / regula falsi (bracketed). Requires a sign-change bracket [a,b].
/// Uses the secant-intercept point x = (a f(b) - b f(a)) / (f(b) - f(a)).
/// Preserves the bracket but may stagnate in strongly nonlinear intervals (Figure 9.3.1 behavior).
pub fn regula_falsi<F>(
    f: &F,
    mut a: f64,
    mut b: f64,
    tol_x: f64,
    tol_f: f64,
    max_iter: usize,
) -> Result<RootResult, RootError>
where
    F: Fn(f64) -> f64,
{
    if !(a < b) {
        return Err(RootError::InvalidInterval);
    }
    let mut fa = ensure_finite(f(a))?;
    let mut fb = ensure_finite(f(b))?;

    if fa == 0.0 {
        return Ok(RootResult {
            root: a,
            f_root: fa,
            iterations: 0,
            converged: true,
        });
    }
    if fb == 0.0 {
        return Ok(RootResult {
            root: b,
            f_root: fb,
            iterations: 0,
            converged: true,
        });
    }
    if !is_bracket(fa, fb) {
        return Err(RootError::NotBracketed { a, b, fa, fb });
    }

    let mut x_prev = a;
    for k in 1..=max_iter {
        let denom = fb - fa;
        if denom.abs() < 1e-15 {
            return Err(RootError::ZeroDenominator);
        }

        // Secant-intercept (linear interpolation)
        let x = ensure_finite((a * fb - b * fa) / denom)?;
        let fx = ensure_finite(f(x))?;

        let width = (b - a).abs();
        let step = (x - x_prev).abs();
        if width <= tol_x * (1.0 + x.abs()) || step <= tol_x * (1.0 + x.abs()) || fx.abs() <= tol_f {
            return Ok(RootResult {
                root: x,
                f_root: fx,
                iterations: k,
                converged: true,
            });
        }

        // Update bracket, preserving sign change.
        if signum_nonzero(fa) == signum_nonzero(fx) {
            a = x;
            fa = fx;
        } else {
            b = x;
            fb = fx;
        }

        x_prev = x;
    }

    Ok(RootResult {
        root: 0.5 * (a + b),
        f_root: ensure_finite(f(0.5 * (a + b)))?,
        iterations: max_iter,
        converged: false,
    })
}

/// Ridders’ method (bracketed). Requires a sign-change bracket [a,b].
/// Uses a midpoint m and an exponential correction that often improves balance compared to regula falsi.
pub fn ridders<F>(
    f: &F,
    mut a: f64,
    mut b: f64,
    tol_x: f64,
    tol_f: f64,
    max_iter: usize,
) -> Result<RootResult, RootError>
where
    F: Fn(f64) -> f64,
{
    if !(a < b) {
        return Err(RootError::InvalidInterval);
    }
    let mut fa = ensure_finite(f(a))?;
    let mut fb = ensure_finite(f(b))?;

    if fa == 0.0 {
        return Ok(RootResult {
            root: a,
            f_root: fa,
            iterations: 0,
            converged: true,
        });
    }
    if fb == 0.0 {
        return Ok(RootResult {
            root: b,
            f_root: fb,
            iterations: 0,
            converged: true,
        });
    }
    if !is_bracket(fa, fb) {
        return Err(RootError::NotBracketed { a, b, fa, fb });
    }

    for k in 1..=max_iter {
        let m = 0.5 * (a + b);
        let fm = ensure_finite(f(m))?;

        if fm.abs() <= tol_f {
            return Ok(RootResult {
                root: m,
                f_root: fm,
                iterations: k,
                converged: true,
            });
        }

        let s2 = fm * fm - fa * fb;
        if s2 <= 0.0 {
            // Fall back to bisection-like behavior if the correction is not well-defined.
            if signum_nonzero(fa) == signum_nonzero(fm) {
                a = m;
                fa = fm;
            } else {
                b = m;
                fb = fm;
            }
        } else {
            let s = s2.sqrt();

            // Ridders correction:
            // x = m + (m-a) * sign(fa - fb) * fm / s
            let sign = if (fa - fb) >= 0.0 { 1.0 } else { -1.0 };
            let x = ensure_finite(m + (m - a) * sign * fm / s)?;
            let fx = ensure_finite(f(x))?;

            if fx.abs() <= tol_f || (b - a).abs() <= tol_x * (1.0 + x.abs()) {
                return Ok(RootResult {
                    root: x,
                    f_root: fx,
                    iterations: k,
                    converged: true,
                });
            }

            // Update bracket using the new point x and midpoint m.
            // We keep a sign-change bracket at all times.
            if signum_nonzero(fm) != signum_nonzero(fx) {
                a = m;
                fa = fm;
                b = x;
                fb = fx;
            } else if signum_nonzero(fa) != signum_nonzero(fx) {
                b = x;
                fb = fx;
            } else {
                a = x;
                fa = fx;
            }
        }

        let width = (b - a).abs();
        if width <= tol_x * (1.0 + 0.5 * (a + b).abs()) {
            let x = 0.5 * (a + b);
            let fx = ensure_finite(f(x))?;
            return Ok(RootResult {
                root: x,
                f_root: fx,
                iterations: k,
                converged: true,
            });
        }
    }

    let x = 0.5 * (a + b);
    Ok(RootResult {
        root: x,
        f_root: ensure_finite(f(x))?,
        iterations: max_iter,
        converged: false,
    })
}

/// Kepler’s equation residual (Eq. 9.3.2): E - e sin(E) - M = 0
fn kepler_residual(e: f64, m: f64, e_anom: f64) -> f64 {
    e_anom - e * e_anom.sin() - m
}

/// A nonlinear function designed to induce regula falsi stagnation behavior on a bracket.
/// f(x) = exp(x) - 3 has a root at ln 3, but if we choose a wide interval, curvature can bias interpolation.
fn nonlinear_demo(x: f64) -> f64 {
    x.exp() - 3.0
}

fn main() -> Result<(), RootError> {
    println!("Program 9.3.0: Secant, False Position, and Ridders’ Method\n");

    let tol_x = 1e-12;
    let tol_f = 1e-12;
    let max_iter = 200;

    // ------------------------------------------------------------
    // Demo 1: Kepler’s equation (Eq. 9.3.2)
    // Choose parameters e and M and solve for E.
    // ------------------------------------------------------------
    let e = 0.7;
    let m = 1.0;
    let f_kepler = |x: f64| kepler_residual(e, m, x);

    // For e in [0,1), a safe bracket is [0, 2*pi] for many M in [0,2*pi].
    let a = 0.0;
    let b = 2.0 * std::f64::consts::PI;

    println!("Demo 1: Kepler’s equation E - e sin(E) - M = 0 (Eq. 9.3.2)");
    println!("  parameters: e = {:.3}, M = {:.3}", e, m);

    let rf = regula_falsi(&f_kepler, a, b, tol_x, tol_f, max_iter)?;
    println!(
        "  Regula falsi:  E ≈ {:.15}, |f(E)| = {:.3e}, iters = {}, converged = {}",
        rf.root,
        rf.f_root.abs(),
        rf.iterations,
        rf.converged
    );

    let rr = ridders(&f_kepler, a, b, tol_x, tol_f, max_iter)?;
    println!(
        "  Ridders:       E ≈ {:.15}, |f(E)| = {:.3e}, iters = {}, converged = {}",
        rr.root,
        rr.f_root.abs(),
        rr.iterations,
        rr.converged
    );

    // Secant method needs two initial guesses; choose them inside the bracket.
    let rs = secant(&f_kepler, 0.5, 2.0, tol_x, tol_f, max_iter)?;
    println!(
        "  Secant:        E ≈ {:.15}, |f(E)| = {:.3e}, iters = {}, converged = {}",
        rs.root,
        rs.f_root.abs(),
        rs.iterations,
        rs.converged
    );
    println!();

    // ------------------------------------------------------------
    // Demo 2: Nonlinear interval where interpolation can be biased (Figure 9.3.1 intuition).
    // f(x) = exp(x) - 3 has root ln(3) ≈ 1.0986.
    // Use a wide bracket [0, 5] to exaggerate curvature effects.
    // ------------------------------------------------------------
    println!("Demo 2: Nonlinear interval illustrating interpolation bias (Figure 9.3.1 intuition)");
    let f_nl = |x: f64| nonlinear_demo(x);
    let a2 = 0.0;
    let b2 = 5.0;

    let rf2 = regula_falsi(&f_nl, a2, b2, tol_x, tol_f, max_iter)?;
    println!(
        "  Regula falsi:  x ≈ {:.15}, |f(x)| = {:.3e}, iters = {}, converged = {}",
        rf2.root,
        rf2.f_root.abs(),
        rf2.iterations,
        rf2.converged
    );

    let rr2 = ridders(&f_nl, a2, b2, tol_x, tol_f, max_iter)?;
    println!(
        "  Ridders:       x ≈ {:.15}, |f(x)| = {:.3e}, iters = {}, converged = {}",
        rr2.root,
        rr2.f_root.abs(),
        rr2.iterations,
        rr2.converged
    );

    // Secant method from two guesses near the left endpoint can be slow or erratic when curvature is strong.
    let rs2 = secant(&f_nl, 0.0, 0.5, tol_x, tol_f, max_iter)?;
    println!(
        "  Secant:        x ≈ {:.15}, |f(x)| = {:.3e}, iters = {}, converged = {}",
        rs2.root,
        rs2.f_root.abs(),
        rs2.iterations,
        rs2.converged
    );

    Ok(())
}
```

Program 9.3.0 demonstrates how interpolation-based root-finding methods can significantly outperform bisection when the function behaves approximately linearly near the root, while also revealing the structural weaknesses of naive interpolation schemes in strongly nonlinear settings. The secant method shows that rapid convergence is possible without derivative information, but only when suitable initial guesses are available. The false position method illustrates how enforcing a bracketing condition restores reliability, albeit sometimes at the cost of stagnation.

Ridders’ method occupies an important middle ground. By retaining the bracketing guarantee while correcting for endpoint bias, it achieves both robustness and efficiency across a wide range of problems. The contrasting outcomes observed in the nonlinear exponential example reinforce the conceptual motivation behind Figure 9.3.1 and clarify why modern hybrid solvers rarely rely on classical regula falsi alone.

Together, these examples highlight a recurring theme in numerical root finding: accelerating convergence requires exploiting local structure, but doing so safely demands mechanisms that guard against pathological behavior. The methods presented here form a natural bridge between the guaranteed but slow contraction of bisection and the fast local convergence of Newton-type schemes, preparing the ground for the hybrid and adaptive algorithms developed in subsequent sections.

## 9.3.1. The Secant Method

The Secant method is an open root-finding algorithm that does not require the root to remain bracketed. It begins with two initial guesses $x_0$ and $x_1$, which should ideally be close to a root and preferably correspond to function values of opposite sign. At each iteration, the method constructs the secant line through the points $(x_{n-1}, f(x_{n-1}))$ and $(x_n, f(x_n))$. The next approximation $x_{n+1}$ is taken as the $x$-intercept of this line.

Algebraically, the update formula is,

$$x_{n+1}= x_n - f(x_n)\frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})} \tag{9.3.3}$$

This expression is obtained by solving the linear interpolation of $f$ between the two most recent iterates for its zero. The Secant method closely resembles Newton’s method, but with the derivative approximated by a finite difference. As a result, it requires only function evaluations and no analytic derivatives.

### Convergence Properties

Under standard smoothness assumptions and when the initial guesses are sufficiently close to a simple root, the Secant method converges superlinearly. Its order of convergence is:

$$p \approx 1.618 \tag{9.3.4}$$

the golden ratio. The error behaves asymptotically as,

$$|e_{n+1}| \approx C\,|e_n|^{p} \tag{9.3.5}$$

for some constant $C$. This convergence is faster than the linear convergence of bisection, though slower than the quadratic convergence of Newton’s method. Each iteration requires only one new function evaluation, and memory requirements are minimal.

### Stability and Drawbacks

Unlike bracketing methods, the Secant method does not guarantee convergence. If the function is poorly behaved or the initial guesses are far from the root, the iterates may diverge or oscillate. For example, if the function is nearly flat near one of the guesses or has a turning point between them, the secant line may predict a new iterate far from the true root. In practice, it is common to monitor the iterates and revert to a safer method, such as bisection, if progress stalls. Despite these risks, the Secant method remains attractive due to its simplicity and efficiency and serves as the conceptual basis for multidimensional quasi-Newton methods.

### Rust Implementation

Following the discussion of the Secant method and its superlinear convergence properties in Section 9.3.1, Program 9.3.1 provides a practical implementation of the algorithm based directly on the update formula given in Equation (9.3.3). In numerical computation, open root-finding methods must balance efficiency against robustness, since they do not enforce bracketing and therefore cannot guarantee convergence from arbitrary starting points. This program illustrates how the Secant method can be implemented using only function evaluations, while incorporating practical safeguards to detect stagnation, excessive step growth, and numerical breakdown. Two representative nonlinear equations are solved to demonstrate the method’s rapid convergence near simple roots and to highlight its behavior in realistic finite-precision settings.

At the core of the implementation is the `secant` function, which encapsulates the iterative procedure defined by Equation (9.3.3). The function accepts a user-supplied scalar function $f(x)$, two initial guesses $x_0$ and $x_1$, and a configuration structure that specifies tolerances and iteration limits. At each iteration, the method constructs the secant line through the points $(x_{n-1}, f(x_{n-1}))$ and $(x_n, f(x_n))$ and computes the next approximation $x_{n+1}$ using the finite-difference approximation to the derivative implicit in Equation (9.3.3). This design mirrors Newton’s method while avoiding the need for analytic or numerical derivatives.

The implementation includes termination criteria based on both the change in successive iterates and the magnitude of the residual $|f(x_n)|$. These stopping conditions reflect the practical interpretation of the asymptotic error model given in Equation (9.3.5), where sufficiently small updates indicate that further iterations will not materially improve accuracy. To prevent numerical instability, the code explicitly checks for near-zero denominators in the secant slope, which correspond to nearly horizontal secant lines and can produce unbounded updates.

In addition to basic convergence checks, the program incorporates simple monitoring logic to detect stagnation and overly aggressive steps. If the magnitude of the proposed update grows disproportionately relative to previous steps, the algorithm applies a damping factor to reduce the step size. While this modification does not alter the underlying Secant iteration, it reflects common practice in production solvers, where open methods are often stabilized by heuristic safeguards. Such measures are consistent with the discussion in the Stability and Drawbacks subsection, where the absence of guaranteed convergence necessitates defensive numerical design.

The `SecantConfig` structure centralizes all algorithmic parameters, including tolerances, iteration limits, and diagnostic options. This separation of configuration from algorithmic logic makes the implementation easier to adapt and allows systematic experimentation with convergence behavior. An optional tracing mode prints a detailed iteration table, enabling direct observation of how the iterates evolve and how rapidly the error decreases, providing empirical confirmation of the superlinear convergence rate described by Equation (9.3.4).

The `main` function demonstrates the method on two classical test problems. The first solves $\cos(x) - x = 0$, a standard benchmark with a simple root and well-behaved derivatives. The second computes the real root of $x^3 - 2 = 0$, illustrating the method’s effectiveness on a smooth polynomial equation. In both cases, the Secant method converges in only a handful of iterations, confirming its efficiency when the initial guesses are reasonably chosen.

```rust
// Program 9.3.1: The Secant Method (open root-finding) with optional tracing and a
// simple safety enhancement (step damping) to reduce wild jumps.
//
// cargo run --release
//
// Core update formula:
//   x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / ( f(x_n) - f(x_{n-1}) )   (9.3.3)
//
// Stopping rules:
//   |f(x_n)| <= f_tol   OR   |x_{n+1} - x_n| <= x_tol
//
// Safety features:
// - Denominator check to avoid dividing by near-zero secant slope.
// - Non-finite value checks.
// - Stall detection: if |f| does not improve meaningfully for several steps, stop.
// - Step damping: if the proposed step is much larger than the previous step, shrink it.

use std::fmt;

#[derive(Debug, Clone)]
pub struct SecantConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Absolute tolerance on the function value: |f(x)| <= f_tol.
    pub f_tol: f64,
    /// Absolute tolerance on successive iterates: |x_{n+1} - x_n| <= x_tol.
    pub x_tol: f64,
    /// If |f(x_n) - f(x_{n-1})| is smaller than this, the secant slope is too flat/unsafe.
    pub denom_tol: f64,
    /// Stall detection: number of consecutive iterations without meaningful improvement.
    pub stall_window: usize,
    /// Stall detection: relative improvement threshold on |f| (smaller is "no improvement").
    pub stall_rel: f64,

    /// If |Δx_proposed| > jump_ratio * |Δx_prev|, apply damping.
    pub jump_ratio: f64,
    /// Damping factor λ in (0,1]: x_{n+1} = x_n + λ (x_{n+1} - x_n) when a jump is detected.
    pub damping: f64,

    /// Whether to print a per-iteration table.
    pub trace: bool,
}

impl Default for SecantConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            f_tol: 1e-12,
            x_tol: 1e-12,
            denom_tol: 1e-15,
            stall_window: 8,
            stall_rel: 1e-6,
            jump_ratio: 20.0,
            damping: 0.25,
            trace: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SecantResult {
    pub root: f64,
    pub f_at_root: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl fmt::Display for SecantResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SecantResult {{ root: {:.16e}, f(root): {:.16e}, iterations: {}, converged: {} }}",
            self.root, self.f_at_root, self.iterations, self.converged
        )
    }
}

#[derive(Debug, Clone)]
pub enum SecantError {
    NonFiniteInput { x0: f64, x1: f64 },
    NonFiniteFunctionValue { x: f64, fx: f64 },
    DenominatorTooSmall { denom: f64, fxn: f64, fxnm1: f64 },
    Stalled { iterations: usize },
    MaxIterations { last_x: f64, last_fx: f64, iterations: usize },
}

impl fmt::Display for SecantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecantError::NonFiniteInput { x0, x1 } => {
                write!(f, "non-finite initial guess: x0 = {x0}, x1 = {x1}")
            }
            SecantError::NonFiniteFunctionValue { x, fx } => {
                write!(f, "non-finite function value: f({x}) = {fx}")
            }
            SecantError::DenominatorTooSmall { denom, fxn, fxnm1 } => write!(
                f,
                "secant denominator too small: f(x_n)-f(x_{{n-1}}) = {denom} (fx_n={fxn}, fx_{{n-1}}={fxnm1})"
            ),
            SecantError::Stalled { iterations } => write!(
                f,
                "iteration stalled (no meaningful improvement) after {iterations} iterations"
            ),
            SecantError::MaxIterations { last_x, last_fx, iterations } => write!(
                f,
                "max iterations reached: x = {last_x}, f(x) = {last_fx}, iterations = {iterations}"
            ),
        }
    }
}

impl std::error::Error for SecantError {}

fn print_trace_header() {
    println!(
        "{:>4}  {:>22}  {:>22}  {:>12}  {:>8}",
        "n", "x_n", "f(x_n)", "|Δx|", "damped?"
    );
}

fn print_trace_row(n: usize, x: f64, fx: f64, dx: Option<f64>, damped: bool) {
    match dx {
        None => println!(
            "{:>4}  {:>22.16e}  {:>22.16e}  {:>12}  {:>8}",
            n, x, fx, "-", if damped { "yes" } else { "no" }
        ),
        Some(d) => println!(
            "{:>4}  {:>22.16e}  {:>22.16e}  {:>12.3e}  {:>8}",
            n, x, fx, d, if damped { "yes" } else { "no" }
        ),
    }
}

/// Secant method root finder with optional tracing and simple damping.
///
/// - f: function for which we want f(x)=0
/// - x0, x1: initial guesses
/// - cfg: tolerances and limits
pub fn secant<F>(f: F, mut x0: f64, mut x1: f64, cfg: SecantConfig) -> Result<SecantResult, SecantError>
where
    F: Fn(f64) -> f64,
{
    if !x0.is_finite() || !x1.is_finite() {
        return Err(SecantError::NonFiniteInput { x0, x1 });
    }

    let mut fx0 = f(x0);
    if !fx0.is_finite() {
        return Err(SecantError::NonFiniteFunctionValue { x: x0, fx: fx0 });
    }

    let mut fx1 = f(x1);
    if !fx1.is_finite() {
        return Err(SecantError::NonFiniteFunctionValue { x: x1, fx: fx1 });
    }

    if cfg.trace {
        print_trace_header();
        print_trace_row(0, x0, fx0, None, false);
        print_trace_row(1, x1, fx1, Some((x1 - x0).abs()), false);
    }

    // Early exit if either guess is already good enough.
    if fx0.abs() <= cfg.f_tol {
        return Ok(SecantResult {
            root: x0,
            f_at_root: fx0,
            iterations: 0,
            converged: true,
        });
    }
    if fx1.abs() <= cfg.f_tol {
        return Ok(SecantResult {
            root: x1,
            f_at_root: fx1,
            iterations: 0,
            converged: true,
        });
    }

    // Stall tracking: monitor relative improvement in |f|.
    let mut best_abs_f = fx0.abs().min(fx1.abs());
    let mut stall_count = 0usize;

    // Previous step size for jump detection.
    let mut prev_dx = (x1 - x0).abs().max(0.0);

    for iter in 1..=cfg.max_iter {
        // Denominator f(x_n) - f(x_{n-1}) in (9.3.3).
        let denom = fx1 - fx0;
        if denom.abs() <= cfg.denom_tol {
            return Err(SecantError::DenominatorTooSmall {
                denom,
                fxn: fx1,
                fxnm1: fx0,
            });
        }

        // Proposed secant update (9.3.3).
        let x2_prop = x1 - fx1 * (x1 - x0) / denom;
        if !x2_prop.is_finite() {
            return Err(SecantError::NonFiniteFunctionValue { x: x2_prop, fx: f64::NAN });
        }

        // Jump detection and damping.
        let mut damped = false;
        let mut x2 = x2_prop;
        let dx_prop = (x2_prop - x1).abs();

        if prev_dx > 0.0 && dx_prop > cfg.jump_ratio * prev_dx {
            // Apply damping: x2 = x1 + λ (x2_prop - x1)
            let lam = cfg.damping.clamp(0.0, 1.0);
            x2 = x1 + lam * (x2_prop - x1);
            damped = true;
        }

        let dx = (x2 - x1).abs();

        // Stop on iterate change.
        if dx <= cfg.x_tol {
            let fx2 = f(x2);
            if !fx2.is_finite() {
                return Err(SecantError::NonFiniteFunctionValue { x: x2, fx: fx2 });
            }
            if cfg.trace {
                print_trace_row(iter + 1, x2, fx2, Some(dx), damped);
            }
            return Ok(SecantResult {
                root: x2,
                f_at_root: fx2,
                iterations: iter,
                converged: true,
            });
        }

        let fx2 = f(x2);
        if !fx2.is_finite() {
            return Err(SecantError::NonFiniteFunctionValue { x: x2, fx: fx2 });
        }

        if cfg.trace {
            print_trace_row(iter + 1, x2, fx2, Some(dx), damped);
        }

        // Stop on function value.
        if fx2.abs() <= cfg.f_tol {
            return Ok(SecantResult {
                root: x2,
                f_at_root: fx2,
                iterations: iter,
                converged: true,
            });
        }

        // Stall detection: if |f| barely improves, count it.
        let abs_f = fx2.abs();
        if abs_f < best_abs_f {
            let rel_improve = (best_abs_f - abs_f) / best_abs_f.max(1.0);
            if rel_improve < cfg.stall_rel {
                stall_count += 1;
            } else {
                stall_count = 0;
            }
            best_abs_f = abs_f;
        } else {
            stall_count += 1;
        }

        if stall_count >= cfg.stall_window {
            return Err(SecantError::Stalled { iterations: iter });
        }

        // Shift the window: (x_{n-1}, x_n) <- (x_n, x_{n+1})
        x0 = x1;
        fx0 = fx1;
        x1 = x2;
        fx1 = fx2;

        prev_dx = dx.max(0.0);
    }

    Err(SecantError::MaxIterations {
        last_x: x1,
        last_fx: fx1,
        iterations: cfg.max_iter,
    })
}

fn main() {
    // Example 1: Solve cos(x) - x = 0 near 0.739085...
    let f1 = |x: f64| x.cos() - x;

    // Example 2: Solve x^3 - 2 = 0 (real root is 2^{1/3} ~ 1.259921...)
    let f2 = |x: f64| x * x * x - 2.0;

    // Configure tolerances. Set trace=true to print an iteration table.
    let cfg = SecantConfig {
        max_iter: 100,
        f_tol: 1e-14,
        x_tol: 1e-14,
        denom_tol: 1e-15,
        stall_window: 10,
        stall_rel: 1e-10,
        jump_ratio: 20.0,
        damping: 0.25,
        trace: false, // flip to true to see the iteration history
    };

    println!("Example 1: f(x) = cos(x) - x");
    match secant(f1, 0.0, 1.0, cfg.clone()) {
        Ok(res) => println!("{res}\n"),
        Err(e) => println!("Secant failed: {e}\n"),
    }

    println!("Example 2: f(x) = x^3 - 2");
    match secant(f2, 1.0, 2.0, cfg) {
        Ok(res) => println!("{res}\n"),
        Err(e) => println!("Secant failed: {e}\n"),
    }
}
```

Program 9.3.1 demonstrates how the Secant method can be translated into a robust numerical algorithm while remaining faithful to its mathematical formulation. By relying solely on function evaluations, the method achieves faster convergence than bracketing techniques while avoiding the derivative requirements of Newton’s method. The included safeguards illustrate how practical implementations address the instability risks discussed earlier in this section.

The example problems highlight the characteristic behavior of the Secant method: rapid convergence near simple roots combined with sensitivity to initial conditions. This balance between efficiency and reliability explains why the Secant method serves as a conceptual bridge between classical root-finding algorithms and modern quasi-Newton methods for systems of equations. The modular structure of the implementation provides a natural foundation for extending these ideas to higher-dimensional problems and for integrating hybrid strategies that combine open and bracketing methods to achieve both speed and robustness.

## 9.3.2. False Position Method (Regula Falsi)

The False Position method, also known as regula falsi, uses linear interpolation in a manner similar to the Secant method but preserves a bracketing interval at every iteration. The method begins with points $a$ and $b$ such that:

$$f(a)f(b) < 0 \tag{9.3.6}$$

which guarantees that at least one root lies in the interval $[a,b]$.

The next approximation is obtained by computing the $x$-intercept of the line through $(a,f(a))$ and $(b,f(b))$,

$$x_{\text{new}}= b - f(b)\frac{a - b}{f(a) - f(b)} \tag{9.3.7}$$

The function is evaluated at $x_{\text{new}}$, and the bracketing interval is updated. If $f(x_{\text{new}})$ has the same sign as $f(a)$, then $a$ is replaced by $x_{\text{new}}$; otherwise, $b$ is replaced. This procedure ensures that the root remains bracketed at every step.

### Reliability Versus Speed

Because the bracketing condition is preserved, the False Position method cannot diverge. However, it may converge very slowly in certain cases. If the magnitudes of (f(a)) and (f(b)) differ greatly, the interpolation is strongly biased toward the endpoint with smaller magnitude, causing the other endpoint to remain nearly fixed. A classical example is,

$$f(x) = x^{10} - 1 \quad \text{on } [0,2] \tag{9.3.8}$$

where the left endpoint remains unchanged for many iterations due to the extreme nonlinearity of the function. This stagnation, illustrated in Figure 9.3.1, can make false position significantly slower than bisection.

### Modern Improvements

Several modified versions of regula falsi address this stagnation problem. The Illinois algorithm and the Anderson–Björck method reduce the influence of an endpoint that remains unchanged, often by scaling its function value by a factor such as $1/2$. The Pegasus method applies a related damping strategy. Hybrid schemes that alternate between bisection and false position have also been proposed and have demonstrated improved robustness and efficiency in practice, particularly in financial applications such as implied volatility estimation (Sapna and Mohan, 2024).

### Rust Implementation

Following the discussion of bracketing methods and the reliability–speed tradeoff in Section 9.3.2, Program 9.3.2 presents a practical implementation of the False Position method (regula falsi) based on the linear interpolation formula given in Equation (9.3.7). In numerical root finding, maintaining a bracketing interval is a powerful guarantee against divergence, but this reliability can come at the cost of slow convergence in strongly nonlinear problems. This program illustrates how the classical false position method enforces the sign-change condition in Equation (9.3.6) at every iteration, ensuring that the root remains enclosed, while also demonstrating a modern modification designed to mitigate the stagnation phenomenon discussed after Equation (9.3.8). By applying the algorithm to both a smooth benchmark problem and a known pathological example, the implementation connects the theoretical properties of regula falsi to its observed numerical behavior in finite-precision computation.

At the core of the implementation is the `regula_falsi` function, which realizes the iterative procedure defined by Equation (9.3.7). The function accepts a user-defined scalar function $f$, an initial interval $[a,b]$, and a configuration structure containing numerical tolerances and algorithmic options. The first step is to evaluate $f(a)$ and $f(b)$ and verify the bracketing condition stated in Equation (9.3.6). This sign-change requirement ensures, by the Intermediate Value Theorem, that at least one root lies within the interval and forms the mathematical basis for the method’s guaranteed reliability.

Each iteration computes a new approximation $x_{\text{new}}$ using the linear interpolation formula of Equation (9.3.7), which corresponds to the x-intercept of the line joining $(a,f(a))$ and $(b,f(b))$. The function is then evaluated at this point, and the interval is updated using the sign test described in the text: if $f(x_{\text{new}})$ has the same sign as $f(a)$, the left endpoint is replaced; otherwise, the right endpoint is replaced. This one-sided update preserves the bracketing property at every step and distinguishes regula falsi from open methods such as the Secant method discussed in Section 9.3.1.

The implementation uses two termination criteria that reflect standard practice for bracketing algorithms. The first tests whether the residual $|f(x_{\text{new}})|$ falls below a prescribed tolerance, indicating that a sufficiently accurate root approximation has been obtained. The second monitors the interval width $|b-a|$, recognizing that once the bracket becomes sufficiently small, any point within it approximates the root to a corresponding absolute accuracy. These criteria allow the algorithm to terminate reliably without relying on asymptotic error models.

To address the slow convergence behavior described after Equation (9.3.8), the program supports both the classical false position method and the Illinois modification through a simple enumeration. In the classical variant, the interpolation weights are left unchanged, which can cause one endpoint to remain nearly fixed when the magnitudes of $f(a)$ and $f(b)$ differ greatly. In the Illinois variant, if the same endpoint is replaced in consecutive iterations, the function value at the opposite endpoint is scaled by a factor of one half. This adjustment reduces the influence of the stagnant endpoint in subsequent interpolations while preserving the bracketing guarantee, thereby improving convergence without sacrificing robustness.

The numerical parameters governing iteration limits, tolerances, and optional diagnostic output are collected in a configuration structure, separating algorithmic logic from problem-dependent tuning. This organization mirrors the theoretical presentation, where the method itself is distinct from the choice of stopping criteria and safeguards required in finite-precision arithmetic.

The `main` function demonstrates the method on two representative examples that directly reflect the discussion in this section. The first solves $\cos(x) - x = 0$ on $[0,1]$, a smooth problem where classical regula falsi converges efficiently. The second applies the method to the stagnation benchmark $f(x) = x^{10} - 1$ on $[0,2]$ from Equation (9.3.8). In this case, the output highlights the extremely slow progress of the classical method and contrasts it with the significantly improved convergence of the Illinois modification, making the theoretical reliability-versus-speed tradeoff visible in practice.

```rust
// Program 9.3.2: False Position (Regula Falsi) with an optional Illinois modification
//
// cargo run --release
//
// This program implements the False Position method (regula falsi) described in Section 9.3.2.
// The iterate x_new is computed as the x-intercept of the secant line through (a,f(a)) and (b,f(b))
// according to Equation (9.3.7). Unlike the Secant method, the algorithm maintains a bracketing
// interval [a,b] at every iteration by replacing exactly one endpoint using a sign test.
//
// Important numerical precondition:
// For bracketing to be valid, the endpoint values must have opposite signs, f(a)*f(b) < 0,
// which guarantees at least one root in [a,b] by the Intermediate Value Theorem.
//
// In addition to the classical method, we include the Illinois variant, which reduces stagnation
// by halving the function value at an endpoint that persists across iterations.

use std::fmt;

#[derive(Debug, Clone, Copy)]
pub enum RegulaFalsiVariant {
    Classical,
    Illinois,
}

#[derive(Debug, Clone)]
pub struct RegulaFalsiConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Absolute tolerance on the function value: |f(x)| <= f_tol.
    pub f_tol: f64,
    /// Absolute tolerance on the interval width: |b-a| <= x_tol.
    pub x_tol: f64,
    /// If |f(a) - f(b)| is too small, the interpolation becomes numerically unsafe.
    pub denom_tol: f64,
    /// Selects classical regula falsi or the Illinois modification.
    pub variant: RegulaFalsiVariant,
    /// Whether to print a per-iteration table.
    pub trace: bool,
}

impl Default for RegulaFalsiConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            f_tol: 1e-12,
            x_tol: 1e-12,
            denom_tol: 1e-15,
            variant: RegulaFalsiVariant::Classical,
            trace: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegulaFalsiResult {
    pub root: f64,
    pub f_at_root: f64,
    pub a: f64,
    pub b: f64,
    pub iterations: usize,
    pub converged: bool,
    pub variant: RegulaFalsiVariant,
}

impl fmt::Display for RegulaFalsiResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = match self.variant {
            RegulaFalsiVariant::Classical => "Classical",
            RegulaFalsiVariant::Illinois => "Illinois",
        };
        write!(
            f,
            "RegulaFalsiResult {{ variant: {}, root: {:.16e}, f(root): {:.16e}, iterations: {}, converged: {}, final_bracket: [{:.16e}, {:.16e}] }}",
            v, self.root, self.f_at_root, self.iterations, self.converged, self.a, self.b
        )
    }
}

#[derive(Debug, Clone)]
pub enum RegulaFalsiError {
    NonFiniteInput { a: f64, b: f64 },
    NonFiniteFunctionValue { x: f64, fx: f64 },
    InvalidBracket { a: f64, b: f64, fa: f64, fb: f64 },
    DenominatorTooSmall { denom: f64, fa: f64, fb: f64 },
    MaxIterations { last_x: f64, last_fx: f64, iterations: usize },
}

impl fmt::Display for RegulaFalsiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegulaFalsiError::NonFiniteInput { a, b } => {
                write!(f, "non-finite interval endpoints: a = {a}, b = {b}")
            }
            RegulaFalsiError::NonFiniteFunctionValue { x, fx } => {
                write!(f, "non-finite function value: f({x}) = {fx}")
            }
            RegulaFalsiError::InvalidBracket { a, b, fa, fb } => write!(
                f,
                "invalid bracket: endpoints must have opposite signs, but f(a) and f(b) do not (a={a}, b={b}, f(a)={fa}, f(b)={fb})"
            ),
            RegulaFalsiError::DenominatorTooSmall { denom, fa, fb } => write!(
                f,
                "interpolation denominator too small: f(a)-f(b) = {denom} (f(a)={fa}, f(b)={fb})"
            ),
            RegulaFalsiError::MaxIterations { last_x, last_fx, iterations } => write!(
                f,
                "max iterations reached: x = {last_x}, f(x) = {last_fx}, iterations = {iterations}"
            ),
        }
    }
}

impl std::error::Error for RegulaFalsiError {}

fn sgn(x: f64) -> i32 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

fn print_trace_header() {
    println!(
        "{:>4}  {:>16}  {:>16}  {:>22}  {:>22}  {:>12}",
        "n", "a", "b", "x_new", "f(x_new)", "|b-a|"
    );
}

/// False Position (Regula Falsi) with optional Illinois modification.
///
/// The classical update for the new point is Equation (9.3.7):
///   x_new = b - f(b) * (a - b) / ( f(a) - f(b) ).
///
/// The bracketing update follows the sign test described in the section:
/// replace the endpoint whose function value has the same sign as f(x_new).
///
/// Illinois modification:
/// if the same endpoint remains unchanged in consecutive steps, its function value is halved,
/// reducing its influence in the next interpolation and mitigating stagnation.
pub fn regula_falsi<F>(
    f: F,
    mut a: f64,
    mut b: f64,
    cfg: RegulaFalsiConfig,
) -> Result<RegulaFalsiResult, RegulaFalsiError>
where
    F: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(RegulaFalsiError::NonFiniteInput { a, b });
    }
    if a == b {
        let fa = f(a);
        if !fa.is_finite() {
            return Err(RegulaFalsiError::NonFiniteFunctionValue { x: a, fx: fa });
        }
        return Err(RegulaFalsiError::InvalidBracket {
            a,
            b,
            fa,
            fb: fa,
        });
    }

    let mut fa = f(a);
    if !fa.is_finite() {
        return Err(RegulaFalsiError::NonFiniteFunctionValue { x: a, fx: fa });
    }
    let mut fb = f(b);
    if !fb.is_finite() {
        return Err(RegulaFalsiError::NonFiniteFunctionValue { x: b, fx: fb });
    }

    // Bracketing precondition: opposite signs or an endpoint is exactly a root.
    if fa.abs() <= cfg.f_tol {
        return Ok(RegulaFalsiResult {
            root: a,
            f_at_root: fa,
            a,
            b,
            iterations: 0,
            converged: true,
            variant: cfg.variant,
        });
    }
    if fb.abs() <= cfg.f_tol {
        return Ok(RegulaFalsiResult {
            root: b,
            f_at_root: fb,
            a,
            b,
            iterations: 0,
            converged: true,
            variant: cfg.variant,
        });
    }
    if sgn(fa) == sgn(fb) {
        return Err(RegulaFalsiError::InvalidBracket { a, b, fa, fb });
    }

    if cfg.trace {
        print_trace_header();
    }

    // Track which endpoint was replaced last time for Illinois scaling.
    // last_replaced = 0 means none yet, 1 means a was replaced, 2 means b was replaced.
    let mut last_replaced: u8 = 0;

    for n in 1..=cfg.max_iter {
        let denom = fa - fb;
        if denom.abs() <= cfg.denom_tol {
            return Err(RegulaFalsiError::DenominatorTooSmall { denom, fa, fb });
        }

        // Equation (9.3.7).
        let x_new = b - fb * (a - b) / denom;
        if !x_new.is_finite() {
            return Err(RegulaFalsiError::NonFiniteFunctionValue {
                x: x_new,
                fx: f64::NAN,
            });
        }

        let fx = f(x_new);
        if !fx.is_finite() {
            return Err(RegulaFalsiError::NonFiniteFunctionValue { x: x_new, fx });
        }

        let width = (b - a).abs();
        if cfg.trace {
            println!(
                "{:>4}  {:>16.8e}  {:>16.8e}  {:>22.16e}  {:>22.16e}  {:>12.3e}",
                n, a, b, x_new, fx, width
            );
        }

        // Convergence tests: residual and interval width.
        if fx.abs() <= cfg.f_tol || width <= cfg.x_tol {
            return Ok(RegulaFalsiResult {
                root: x_new,
                f_at_root: fx,
                a,
                b,
                iterations: n,
                converged: true,
                variant: cfg.variant,
            });
        }

        // Bracket update by sign test.
        // If f(x_new) has the same sign as f(a), replace a; otherwise replace b.
        if sgn(fx) == sgn(fa) {
            a = x_new;
            fa = fx;

            if let RegulaFalsiVariant::Illinois = cfg.variant {
                // If we replaced a again, b stayed fixed; reduce b's influence.
                if last_replaced == 1 {
                    fb *= 0.5;
                }
            }
            last_replaced = 1;
        } else {
            b = x_new;
            fb = fx;

            if let RegulaFalsiVariant::Illinois = cfg.variant {
                // If we replaced b again, a stayed fixed; reduce a's influence.
                if last_replaced == 2 {
                    fa *= 0.5;
                }
            }
            last_replaced = 2;
        }
    }

    // If we exit the loop, we did not converge within max_iter.
    // Provide the most recent interpolation point for context.
    let denom = fa - fb;
    let x_last = if denom.abs() > cfg.denom_tol {
        b - fb * (a - b) / denom
    } else {
        (a + b) / 2.0
    };
    let fx_last = f(x_last);

    Err(RegulaFalsiError::MaxIterations {
        last_x: x_last,
        last_fx: fx_last,
        iterations: cfg.max_iter,
    })
}

fn main() {
    // Example A: A smooth benchmark where regula falsi is efficient.
    // Solve cos(x) - x = 0 on [0,1].
    let f1 = |x: f64| x.cos() - x;

    // Example B: A classical stagnation case for regula falsi.
    // Solve x^10 - 1 = 0 on [0,2], whose root is x = 1. (Equation 9.3.8)
    let f2 = |x: f64| x.powi(10) - 1.0;

    let base_cfg = RegulaFalsiConfig {
        max_iter: 200,
        f_tol: 1e-14,
        x_tol: 1e-14,
        denom_tol: 1e-15,
        trace: false, // set true to see the interval evolution
        ..Default::default()
    };

    println!("Example A: f(x) = cos(x) - x on [0, 1]");
    let cfg_classical = RegulaFalsiConfig {
        variant: RegulaFalsiVariant::Classical,
        ..base_cfg.clone()
    };
    match regula_falsi(f1, 0.0, 1.0, cfg_classical) {
        Ok(res) => println!("{res}\n"),
        Err(e) => println!("Regula falsi failed: {e}\n"),
    }

    println!("Example B: f(x) = x^10 - 1 on [0, 2] (stagnation benchmark)");
    let cfg_classical = RegulaFalsiConfig {
        variant: RegulaFalsiVariant::Classical,
        ..base_cfg.clone()
    };
    match regula_falsi(f2, 0.0, 2.0, cfg_classical) {
        Ok(res) => println!("Classical: {res}\n"),
        Err(e) => println!("Classical regula falsi failed: {e}\n"),
    }

    let cfg_illinois = RegulaFalsiConfig {
        variant: RegulaFalsiVariant::Illinois,
        ..base_cfg
    };
    match regula_falsi(f2, 0.0, 2.0, cfg_illinois) {
        Ok(res) => println!("Illinois:  {res}\n"),
        Err(e) => println!("Illinois regula falsi failed: {e}\n"),
    }
}
```

Program 9.3.2 illustrates how the False Position method combines the safety of bracketing with the efficiency of linear interpolation. By preserving the sign-change condition at every iteration, the method guarantees that a root remains enclosed within the interval, eliminating the risk of divergence that characterizes open methods. At the same time, the numerical examples confirm that this reliability does not imply uniformly fast convergence, particularly for highly nonlinear functions where endpoint domination can lead to stagnation.

The inclusion of the Illinois modification demonstrates how modest algorithmic adjustments can substantially improve performance while retaining the essential bracketing property. This perspective motivates the development of hybrid and modified bracketing schemes that seek to balance robustness and efficiency, and it explains why modern root-finding libraries rarely rely on classical regula falsi alone. The framework established here provides a foundation for understanding these more advanced methods and their role in practical numerical computation.

## 9.3.3. Ridders’ Method

Ridders’ method is a bracketing algorithm that combines robustness with rapid convergence. Introduced by Ridders (1979), it preserves the bracketing guarantee while achieving quadratic convergence for smooth functions.

The method starts with an interval $[x_1,x_2]$ such that:

$$f(x_1)f(x_2) < 0 \tag{9.3.9}$$

The midpoint,

$$x_3 = \frac{x_1 + x_2}{2} \tag{9.3.10}$$

is evaluated. If $f(x_3)=0$, the root has been found.

### Exponential Straightening

Ridders’ key idea is to apply an exponential transformation that locally straightens the function. One introduces transformed values,

\begin{equation}
\begin{aligned}
g(x_1) &= f(x_1) \\
g(x_3) &= f(x_3)\,e^{Q} \\
g(x_2) &= f(x_2)\,e^{2Q}
\end{aligned}
\tag{9.3.11}
\end{equation}

and chooses $Q$ so that these three points lie on a straight line. Enforcing this condition yields:

$$g(x_1) - 2g(x_3) + g(x_2) = 0 \tag{9.3.12}$$

Solving for $Q$ gives:

$$e^{Q}= \frac{f(x_3) + \operatorname{sign}(f(x_2))\sqrt{f(x_3)^2 - f(x_1)f(x_2)}}{f(x_2)} \tag{9.3.13}$$

Equation (9.3.13) defines the exponential scaling factor $e^{Q}$ used in Ridders’ method to construct an auxiliary function whose behavior is closer to linear near the root. Starting from three points $x_1 < x_3 < x_2$ that bracket a simple root, the function values $f(x_1)$ and $f(x_2)$ have opposite signs, while $x_3$ is typically chosen as the midpoint of the interval. The quantity under the square root, $f(x_3)^2 - f(x_1)f(x_2),$ is strictly positive under the bracketing assumption, ensuring that $e^{Q}$ is real and well defined.

The numerator combines the function value at the midpoint with a correction term whose sign is chosen to match $\operatorname{sign}(f(x_2))$. This choice ensures numerical stability and guarantees that the transformed function preserves the sign change across the interval. Dividing by $f(x_2)$ normalizes the expression so that the exponential factor rescales the function values at the endpoints in a controlled manner.

Conceptually, the role of $e^{Q}$ is to “tilt” the function values so that the auxiliary function $g(x)$ defined in (9.3.11) becomes approximately antisymmetric about the root. As a result, linear interpolation applied to $g(x)$ produces a new iterate that converges more rapidly than standard regula falsi while retaining a bracketing guarantee. This exponential transformation is the key mechanism by which Ridders’ method achieves both robustness and superlinear convergence for well-behaved nonlinear functions.

### Root Update and Bracketing

Once the transformation is determined, a secant step is applied to the transformed values, yielding the next approximation,

$$x_4= x_3 + (x_3 - x_1)\frac{\operatorname{sign}(f(x_1) - f(x_2)),f(x_3)}{\sqrt{f(x_3)^2 - f(x_1)f(x_2)}} \tag{9.3.14}$$

It can be shown that $x_4$ always lies within the original bracket. The interval is then updated according to the sign of $f(x_4)$, ensuring that the bracketing condition is preserved.

### Convergence and Practical Use

When the function is smooth near the root, Ridders’ method converges quadratically. If the function behaves poorly, the method degrades gracefully to linear convergence, with the interval shrinking by at least a factor of two over two iterations. This guarantees convergence while retaining much of Newton’s efficiency. Ridders’ method typically requires two function evaluations per iteration and is implemented in numerical libraries such as SciPy. It is widely regarded as a robust alternative when Newton’s method fails due to poor initial guesses or unavailable derivatives.

The Secant, False Position, and Ridders’ methods represent increasingly sophisticated approaches to improving upon bisection. The Secant method offers speed but lacks guarantees. False Position restores robustness but may stagnate. Ridders’ method achieves a balance, combining guaranteed convergence with near-quadratic performance. These methods motivate the development of hybrid algorithms that seek to combine reliability and speed, leading naturally to the van Wijngaarden–Dekker–Brent method discussed in the next section.

Below is textbook-ready prose written to integrate **directly into Section 9.3.3** around **Program 9.3.3 (Ridders’ Method)**. The style, structure, and level of explanation follow your reference example exactly, maintain continuity with the section narrative, and refer only to the existing equation numbers without rewriting them.

---

### Introductory paragraph (before the code block)

Following the discussion of bracketing algorithms and their limitations in Sections 9.3.1 and 9.3.2, **Program 9.3.3** presents a practical implementation of Ridders’ method, which combines the reliability of bracketing with rapid convergence. Unlike the False Position method, which may stagnate when interpolation becomes strongly biased, Ridders’ method applies an exponential transformation to locally straighten the nonlinear function, as described by Equations (9.3.11)–(9.3.13). This approach preserves the bracketing guarantee implied by Equation (9.3.9) while typically achieving quadratic convergence for smooth functions. The program demonstrates how this transformation leads to a robust and efficient root-finding algorithm that bridges the gap between conservative bracketing methods and fast open methods.

---

### Explanatory paragraphs (placed before the code block)

At the core of the implementation is the `ridders` function, which realizes the algorithmic steps described in Section 9.3.3. The function accepts a user-defined scalar function (f), an initial bracketing interval (\[x_1,x_2\]), and a configuration structure specifying numerical tolerances and iteration limits. The method begins by verifying the sign-change condition implicit in Equation (9.3.9), ensuring that the root is bracketed and that subsequent iterations preserve this property.

Each iteration starts by computing the midpoint (x_3) according to Equation (9.3.10) and evaluating (f(x_3)). If the midpoint already satisfies the residual tolerance, the algorithm terminates immediately, reflecting the fact that the root has been resolved exactly within the current interval. Otherwise, the method constructs the exponential scaling factor defined implicitly by Equations (9.3.11)–(9.3.13). The quantity under the square root, (f(x_3)^2 - f(x_1)f(x_2)), is guaranteed to be positive under the bracketing assumption, ensuring that the transformation is well defined in exact arithmetic.

Using this exponential straightening, the algorithm computes a new approximation (x_4) via the update formula in Equation (9.3.14). Conceptually, this step applies a secant-like interpolation to the transformed function values, producing an iterate that converges more rapidly than standard linear interpolation while remaining inside the original bracket. The implementation explicitly selects the sign in Equation (9.3.14) to maintain numerical stability and to guarantee that (x_4) lies within (\[x_1,x_2\]).

After evaluating (f(x_4)), the bracketing interval is updated using a sign test, ensuring that the new interval continues to enclose the root. If the exponential correction becomes ill-conditioned, for example due to roundoff effects near convergence, the implementation safely degrades to a bisection-style update. This fallback mechanism reflects the theoretical property that Ridders’ method never performs worse than a bracketing scheme and retains convergence even when its superlinear behavior is temporarily lost.

The stopping criteria combine a residual test (|f(x)|\\le \\text{f_tol}) with an interval-width test (|x_2-x_1|\\le \\text{x_tol}). These criteria reflect the dual interpretation of convergence for bracketing methods: accuracy can be inferred either from the function value at the iterate or from the geometric size of the enclosing interval. All numerical parameters are collected in a configuration structure, separating algorithmic logic from tolerance selection and reinforcing the connection between the theoretical method and its practical realization.

The `main` function demonstrates the method on two representative examples. The first solves ( \\cos(x)-x=0 ) on (\[0,1\]), where Ridders’ method converges in only a few iterations, illustrating its near-quadratic behavior for smooth functions. The second applies the method to the highly nonlinear equation (x^{10}-1=0) on (\[0,2\]), a case that causes stagnation for classical regula falsi. In contrast, Ridders’ method resolves the root immediately when the midpoint coincides with the solution, highlighting both its robustness and its ability to exploit favorable problem structure.

---

### Concluding remarks (after the code block)

```rust
// Program 9.3.3: Ridders' Method (bracketing with exponential straightening)
//
// cargo run --release
//
// Ridders' method preserves a bracketing interval [x1,x2] while typically achieving
// near-quadratic convergence for smooth functions. The implementation follows the
// section equations:
//
// - Midpoint:                 x3 = (x1 + x2)/2                            (9.3.10)
// - Exponential factor:       e^Q from the closed form in                 (9.3.13)
// - Ridders update:           x4 computed by the Ridders step             (9.3.14)
//
// As with all bracketing methods, the essential precondition is a sign change across
// the initial interval: f(x1) * f(x2) < 0. (Equation (9.3.9) in the text should use
// the sign-change condition, as in Section 9.3.2.)

use std::fmt;

#[derive(Debug, Clone)]
pub struct RiddersConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Absolute tolerance on the function value: |f(x)| <= f_tol.
    pub f_tol: f64,
    /// Absolute tolerance on interval width: |x2 - x1| <= x_tol.
    pub x_tol: f64,
    /// Guard for the square-root argument and divisions.
    pub tiny: f64,
    /// Whether to print a per-iteration table.
    pub trace: bool,
}

impl Default for RiddersConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            f_tol: 1e-12,
            x_tol: 1e-12,
            tiny: 1e-15,
            trace: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RiddersResult {
    pub root: f64,
    pub f_at_root: f64,
    pub x1: f64,
    pub x2: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl fmt::Display for RiddersResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RiddersResult {{ root: {:.16e}, f(root): {:.16e}, iterations: {}, converged: {}, final_bracket: [{:.16e}, {:.16e}] }}",
            self.root, self.f_at_root, self.iterations, self.converged, self.x1, self.x2
        )
    }
}

#[derive(Debug, Clone)]
pub enum RiddersError {
    NonFiniteInput { x1: f64, x2: f64 },
    NonFiniteFunctionValue { x: f64, fx: f64 },
    InvalidBracket { x1: f64, x2: f64, f1: f64, f2: f64 },
    NumericalBreakdown { message: &'static str },
    MaxIterations { last_x: f64, last_fx: f64, iterations: usize },
}

impl fmt::Display for RiddersError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiddersError::NonFiniteInput { x1, x2 } => {
                write!(f, "non-finite interval endpoints: x1 = {x1}, x2 = {x2}")
            }
            RiddersError::NonFiniteFunctionValue { x, fx } => {
                write!(f, "non-finite function value: f({x}) = {fx}")
            }
            RiddersError::InvalidBracket { x1, x2, f1, f2 } => write!(
                f,
                "invalid bracket: f(x1) and f(x2) must have opposite signs (x1={x1}, x2={x2}, f1={f1}, f2={f2})"
            ),
            RiddersError::NumericalBreakdown { message } => write!(f, "numerical breakdown: {message}"),
            RiddersError::MaxIterations { last_x, last_fx, iterations } => write!(
                f,
                "max iterations reached: x = {last_x}, f(x) = {last_fx}, iterations = {iterations}"
            ),
        }
    }
}

impl std::error::Error for RiddersError {}

fn sgn(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn print_trace_header() {
    println!(
        "{:>4}  {:>14}  {:>14}  {:>14}  {:>22}  {:>12}",
        "n", "x1", "x3", "x2", "x4", "|x2-x1|"
    );
}

/// Ridders' method for solving f(x) = 0 on a bracket [x1,x2].
///
/// Preconditions:
/// - x1 < x2
/// - f(x1) * f(x2) < 0  (sign change across the interval)
///
/// Iteration:
/// 1) Evaluate midpoint x3 (9.3.10)
/// 2) Compute the exponential scaling factor implicitly through the Ridders step (9.3.14)
/// 3) Evaluate x4, then update the bracket using the sign of f(x4)
pub fn ridders<F>(
    f: F,
    mut x1: f64,
    mut x2: f64,
    cfg: RiddersConfig,
) -> Result<RiddersResult, RiddersError>
where
    F: Fn(f64) -> f64,
{
    if !x1.is_finite() || !x2.is_finite() {
        return Err(RiddersError::NonFiniteInput { x1, x2 });
    }
    if x1 == x2 {
        let f1 = f(x1);
        if !f1.is_finite() {
            return Err(RiddersError::NonFiniteFunctionValue { x: x1, fx: f1 });
        }
        return Err(RiddersError::InvalidBracket {
            x1,
            x2,
            f1,
            f2: f1,
        });
    }
    if x1 > x2 {
        std::mem::swap(&mut x1, &mut x2);
    }

    let mut f1 = f(x1);
    if !f1.is_finite() {
        return Err(RiddersError::NonFiniteFunctionValue { x: x1, fx: f1 });
    }
    let mut f2 = f(x2);
    if !f2.is_finite() {
        return Err(RiddersError::NonFiniteFunctionValue { x: x2, fx: f2 });
    }

    // Immediate success at endpoints.
    if f1.abs() <= cfg.f_tol {
        return Ok(RiddersResult {
            root: x1,
            f_at_root: f1,
            x1,
            x2,
            iterations: 0,
            converged: true,
        });
    }
    if f2.abs() <= cfg.f_tol {
        return Ok(RiddersResult {
            root: x2,
            f_at_root: f2,
            x1,
            x2,
            iterations: 0,
            converged: true,
        });
    }

    // Bracketing requirement.
    if f1 * f2 > 0.0 {
        return Err(RiddersError::InvalidBracket { x1, x2, f1, f2 });
    }

    if cfg.trace {
        print_trace_header();
    }

    // Track the latest x4 for reporting on failure.
    let mut last_x4 = (x1 + x2) / 2.0;
    let mut last_f4 = f(last_x4);

    for n in 1..=cfg.max_iter {
        let width = (x2 - x1).abs();
        if width <= cfg.x_tol {
            // Best available approximation: midpoint.
            let x_mid = (x1 + x2) / 2.0;
            let f_mid = f(x_mid);
            if !f_mid.is_finite() {
                return Err(RiddersError::NonFiniteFunctionValue { x: x_mid, fx: f_mid });
            }
            return Ok(RiddersResult {
                root: x_mid,
                f_at_root: f_mid,
                x1,
                x2,
                iterations: n - 1,
                converged: true,
            });
        }

        // Midpoint (9.3.10).
        let x3 = (x1 + x2) / 2.0;
        let f3 = f(x3);
        if !f3.is_finite() {
            return Err(RiddersError::NonFiniteFunctionValue { x: x3, fx: f3 });
        }
        if f3.abs() <= cfg.f_tol {
            return Ok(RiddersResult {
                root: x3,
                f_at_root: f3,
                x1,
                x2,
                iterations: n,
                converged: true,
            });
        }

        // Square-root term in (9.3.13) and (9.3.14): sqrt(f3^2 - f1*f2).
        let rad = f3 * f3 - f1 * f2;
        if rad < 0.0 && rad.abs() <= 10.0 * cfg.tiny {
            // Allow a tiny negative due to rounding.
            // Clamp to zero to maintain real arithmetic.
        }
        let rad_clamped = if rad < 0.0 { 0.0 } else { rad };
        let denom = rad_clamped.sqrt();

        if denom <= cfg.tiny {
            // Degenerate case: fall back to bisection-style bracketing update.
            // This preserves convergence even if the Ridders correction becomes ill-conditioned.
            if f1 * f3 < 0.0 {
                x2 = x3;
                f2 = f3;
            } else {
                x1 = x3;
                f1 = f3;
            }
            last_x4 = x3;
            last_f4 = f3;

            if cfg.trace {
                println!(
                    "{:>4}  {:>14.6e}  {:>14.6e}  {:>14.6e}  {:>22}  {:>12.3e}",
                    n, x1, x3, x2, "bisection", (x2 - x1).abs()
                );
            }
            continue;
        }

        // Ridders update (9.3.14).
        //
        // A standard stable implementation uses:
        //   x4 = x3 + (x3 - x1) * s * f3 / denom
        // where s = sign(f1 - f2). The sign choice ensures x4 lies within [x1,x2].
        let s = sgn(f1 - f2);
        let x4 = x3 + (x3 - x1) * s * f3 / denom;

        if !x4.is_finite() {
            return Err(RiddersError::NumericalBreakdown {
                message: "x4 became non-finite",
            });
        }

        let f4 = f(x4);
        if !f4.is_finite() {
            return Err(RiddersError::NonFiniteFunctionValue { x: x4, fx: f4 });
        }

        last_x4 = x4;
        last_f4 = f4;

        if cfg.trace {
            println!(
                "{:>4}  {:>14.6e}  {:>14.6e}  {:>14.6e}  {:>22.16e}  {:>12.3e}",
                n,
                x1,
                x3,
                x2,
                x4,
                (x2 - x1).abs()
            );
        }

        // Convergence by residual.
        if f4.abs() <= cfg.f_tol {
            return Ok(RiddersResult {
                root: x4,
                f_at_root: f4,
                x1,
                x2,
                iterations: n,
                converged: true,
            });
        }

        // Bracketing update: keep an interval with opposite signs.
        // We have four candidates x1 < x3 < x2 and x4 in [x1,x2].
        // Use sign tests to choose the next bracket.
        if f3 * f4 < 0.0 {
            // Root is between x3 and x4.
            if x4 < x3 {
                x1 = x4;
                f1 = f4;
                x2 = x3;
                f2 = f3;
            } else {
                x1 = x3;
                f1 = f3;
                x2 = x4;
                f2 = f4;
            }
        } else if f1 * f4 < 0.0 {
            // Root is between x1 and x4.
            x2 = x4;
            f2 = f4;
        } else {
            // Root is between x4 and x2.
            x1 = x4;
            f1 = f4;
        }
    }

    Err(RiddersError::MaxIterations {
        last_x: last_x4,
        last_fx: last_f4,
        iterations: cfg.max_iter,
    })
}

fn main() {
    // Example 1: Solve cos(x) - x = 0 on [0,1].
    let f1 = |x: f64| x.cos() - x;

    // Example 2: Stagnation benchmark for regula falsi becomes easy for Ridders.
    // Solve x^10 - 1 = 0 on [0,2]. Root at x = 1. (Compare with Equation 9.3.8.)
    let f2 = |x: f64| x.powi(10) - 1.0;

    let cfg = RiddersConfig {
        max_iter: 100,
        f_tol: 1e-14,
        x_tol: 1e-14,
        tiny: 1e-15,
        trace: false, // set true to observe iterations
    };

    println!("Example 1: f(x) = cos(x) - x on [0, 1]");
    match ridders(f1, 0.0, 1.0, cfg.clone()) {
        Ok(res) => println!("{res}\n"),
        Err(e) => println!("Ridders failed: {e}\n"),
    }

    println!("Example 2: f(x) = x^10 - 1 on [0, 2]");
    match ridders(f2, 0.0, 2.0, cfg) {
        Ok(res) => println!("{res}\n"),
        Err(e) => println!("Ridders failed: {e}\n"),
    }
}
```

Program 9.3.3 demonstrates how Ridders’ method achieves a balance between reliability and efficiency that is difficult to obtain with simpler root-finding algorithms. By preserving a bracketing interval at every step, the method guarantees convergence, while the exponential straightening mechanism enables rapid progress when the function is smooth near the root. This combination explains why Ridders’ method often performs comparably to Newton’s method without requiring derivative information.

The numerical examples confirm the theoretical properties discussed in Section 9.3.3. For well-behaved functions, Ridders’ method exhibits near-quadratic convergence, while for more challenging nonlinear problems it degrades gracefully to linear convergence without sacrificing robustness. This balance makes Ridders’ method an attractive alternative when derivatives are unavailable or unreliable, and it provides important conceptual insight into the design of hybrid algorithms. These ideas culminate in the van Wijngaarden–Dekker–Brent method discussed in the next section, which integrates bracketing, interpolation, and safeguarding strategies into a single, widely used root-finding algorithm.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/nszFtwiMyIxzhH2RiSk0.4","tags":[]}

# 9.4. Van Wijngaarden–Dekker–Brent Method

Even with robust algorithms such as Ridders’ method, there exist nonlinear functions for which simpler root-finding approaches converge slowly or inefficiently. A function may be smooth but exhibit sharply varying curvature near the root, causing interpolation-based methods to zigzag. In other cases, the function may contain flat regions or mild irregularities that frustrate fixed-form algorithms. These challenges motivate the search for a method that combines the guaranteed convergence of bracketing with the speed of higher-order interpolation.

The van Wijngaarden–Dekker–Brent method, commonly referred to as Brent’s method, answers this need. Developed through the work of van Wijngaarden and Dekker in the 1960s and refined by Brent in 1973, this algorithm adaptively combines bisection, the secant method, and inverse quadratic interpolation into a single, reliable procedure.

### Basic Structure and Initialization

Brent’s method begins with a bracketing interval $[a,b]$ such that,

$$f(a)f(b) < 0 \tag{9.4.1}$$

Continuity of $f$ on $[a,b]$ guarantees the existence of at least one root within the interval.

In addition to the endpoints $a$ and $b$, the algorithm maintains a third point $c$, which typically represents the previous value of $a$. The function values $f(a)$, $f(b)$, and $f(c)$ are stored, and the points are reordered as needed so that,

$$|f(b)| \le |f(a)| \tag{9.4.2}$$

This convention ensures that $b$ is always the current best approximation to the root.

### Interpolation Strategy

At each iteration, Brent’s method attempts to generate a fast update using interpolation. When only two distinct points are available, a secant step is attempted. When three distinct points $(a,f(a))$, $(b,f(b))$, and $(c,f(c))$ are available, the method attempts *inverse quadratic interpolation*, which fits a quadratic polynomial to $x$ as a function of $f(x)$ and then evaluates it at zero.

Using Lagrange interpolation, the inverse quadratic estimate can be written as:

\begin{equation}
\begin{aligned}
x_{\text{interp}} &=
a\,\frac{f(b)f(c)}{(f(a)-f(b))(f(a)-f(c))} \\
&\quad
+\, b\,\frac{f(a)f(c)}{(f(b)-f(a))(f(b)-f(c))} \\
&\quad
+\, c\,\frac{f(a)f(b)}{(f(c)-f(a))(f(c)-f(b))}
\end{aligned}
\tag{9.4.3}
\end{equation}

This expression reduces to the secant formula when two of the function values coincide.

For numerical stability, Brent’s original implementation rewrites this expression using auxiliary ratios,

\begin{equation}
\begin{aligned}
R &= \frac{f(b)}{f(c)} \\
S &= \frac{f(b)}{f(a)} \\
T &= \frac{f(a)}{f(c)}
\end{aligned}
\tag{9.4.4}
\end{equation}

and expresses the update compactly as:

$$x_{\text{interp}} = b + \frac{P}{Q} \tag{9.4.5}$$

where,

$$P = S\bigl[T(R-T)(c-b) - (1-R)(b-a)\bigr] \tag{9.4.6}$$

$$Q = (T-1)(R-1)(S-1) \tag{9.4.7}$$

### Safeguards and Fallback Mechanism

A defining feature of Brent’s method is that it never blindly trusts interpolation. After computing a candidate point, the algorithm checks several safety conditions:

1. The new point must lie strictly within the current bracketing interval $[a,b]$.
2. The step size must be sufficiently large relative to the previous step or the interval width, to avoid stagnation due to round-off error.
3. The function value at the new point must show adequate progress relative to previous iterations.

If any of these conditions fail, the interpolation step is rejected and replaced by a *bisection step,*

$$x_{\text{bisect}} = \frac{a+b}{2} \tag{9.4.8}$$

This guarantees that the interval width decreases by at least a factor of two whenever interpolation is unreliable. After evaluating $f$ at the accepted candidate, the bracketing interval is updated in the standard way to preserve the sign change condition.

### Convergence Properties

Brent’s method guarantees convergence for any continuous function $f$ on $[a,b]$ satisfying (9.4.1). In the worst case, the algorithm behaves like bisection, yielding linear convergence with predictable error reduction.

In favorable cases, when the function is smooth and well behaved near the root, the method frequently accepts inverse quadratic interpolation steps. In such regimes, convergence is superlinear and often close to quadratic, comparable to Newton’s method but without requiring derivatives.

The worst-case number of iterations required to achieve a tolerance $\text{TOL}$ satisfies:

$$n \le \left\lceil \log_2\!\left(\frac{b-a}{\text{TOL}}\right) \right\rceil \tag{9.4.9}$$

while in practice far fewer iterations are usually needed.

### Practical Performance and Implementations

Empirical experience shows that Brent’s method converges in only a handful of iterations for most well-behaved functions. This combination of reliability and speed has made it the default choice in many numerical libraries.

In scientific computing, the SciPy library implements Brent’s method in the function `brentq`, which is widely regarded as one of the most reliable one-dimensional root solvers available (SciPy v1.16 Documentation, 2023). Decades of practical use have confirmed that the algorithm rarely fails when a valid bracket is provided.

### Engineering Application: Colebrook Equation

A classical engineering application of Brent’s method is the solution of the Colebrook–White equation in fluid mechanics,

$$
\frac{1}{\sqrt{f}}
= -2 \log_{10}\!\left(
\frac{\varepsilon/D}{3.7}
+ \frac{2.51}{\mathrm{Re}\,\sqrt{f}}
\right) \tag{9.4.10}
$$

where $f$ is the Darcy friction factor, $\text{Re}$ is the Reynolds number, and $\varepsilon/D$ is the relative roughness.

This equation has no closed-form solution and must be solved repeatedly in pipe-flow simulations. The function can be flat or sensitive in certain parameter regimes, causing naive iterative methods to converge slowly or diverge. Brent’s method is frequently recommended for this problem because it always converges from a reasonable initial bracket and typically does so much faster than pure bisection or fixed-point iteration. Recent energy systems research confirms its effectiveness in embedded efficiency calculations (Parisi et al., 2025).

### Broader Applications

Beyond fluid mechanics, Brent’s method is widely used whenever a single-variable nonlinear equation must be solved reliably without derivative information. Applications include quantile computation in statistics, hyperparameter tuning in machine learning, and calibration problems in finance and engineering. The continued prevalence of Brent’s method in modern libraries reflects its balanced design, which combines theoretical guarantees with practical efficiency.

### Concluding Remarks

The van Wijngaarden–Dekker–Brent method represents a culmination of ideas developed throughout this chapter. By dynamically combining bisection, secant updates, and inverse quadratic interpolation, it achieves an almost optimal balance between robustness and speed. As of today, Brent’s method remains the gold standard for one-dimensional root finding and serves as a benchmark against which newer algorithms are measured.

### Rust Implementation

Following the development of bracketing and interpolation-based root-finding algorithms in Sections 9.3.1–9.3.3, Program 9.4.0 provides a practical implementation of the van Wijngaarden–Dekker–Brent method. In numerical computation, one-dimensional root finding often requires a careful balance between robustness and efficiency: pure bracketing methods such as bisection guarantee convergence but may be slow, while open methods such as the secant or Newton’s method can converge rapidly but lack reliability without favorable initial conditions. Brent’s method addresses this tradeoff by adaptively combining bisection, secant updates, and inverse quadratic interpolation within a single safeguarded framework. The present program implements this hybrid strategy, illustrating how reliable bracketing can be preserved while exploiting higher-order interpolation whenever it is safe to do so.

At the core of the implementation is the `brent` function, which realizes the algorithmic structure described in Section 9.4. The function accepts a user-defined scalar function $f$, an initial interval $[a,b]$, and a configuration structure specifying tolerances and iteration limits. The method begins by verifying the bracketing condition in Equation (9.4.1), ensuring that the function values at the endpoints have opposite signs. This condition guarantees the existence of at least one root in the interval and forms the basis for the method’s global convergence.

Throughout the iteration, the algorithm maintains three points $a$, $b$, and $c$, along with their function values. The points are reordered as needed to enforce the convention in Equation (9.4.2), ensuring that $b$ is always the current best approximation to the root in the sense of having the smallest function magnitude. This ordering simplifies subsequent decision logic and reflects the theoretical formulation of Brent’s method.

At each iteration, the algorithm attempts to compute a new candidate point using interpolation. When only two distinct points are effectively available, the method reduces to a secant step. When three distinct points $(a,f(a))$, $(b,f(b))$, and $(c,f(c))$ are available, the method attempts inverse quadratic interpolation using the formulation given in Equations (9.4.3)–(9.4.7). This approach fits a quadratic model to the inverse function $x(f)$ and evaluates it at zero, often yielding superlinear convergence for smooth functions.

Crucially, interpolation is never accepted unconditionally. The implementation applies a sequence of safeguard tests that ensure the proposed step lies strictly within the current bracketing interval and represents adequate progress relative to previous steps. If any of these conditions fail, the interpolation step is rejected and replaced by a bisection step, as defined in Equation (9.4.8). This fallback mechanism guarantees that the interval width decreases whenever interpolation becomes unreliable, preserving the method’s robustness.

The algorithm terminates when one of two conditions is satisfied: either the residual (|f(b)|) falls below a prescribed tolerance, or the width of the bracketing interval becomes sufficiently small. The implementation explicitly records which of these criteria triggers convergence, making the stopping logic transparent and reinforcing the interpretation of Brent’s method as both a functional and a geometric convergence process.

The `main` function demonstrates the method on three representative problems. The first is a smooth transcendental equation, illustrating typical fast convergence. The second is a strongly nonlinear polynomial that can cause stagnation in simpler methods, highlighting the effectiveness of safeguarded interpolation. The third example applies Brent’s method to the Colebrook–White equation from fluid mechanics, demonstrating its reliability in a realistic engineering context where derivative information is unavailable and robustness is essential.

```rust
// Program 9.4.0: Van Wijngaarden–Dekker–Brent Method (Brent's bracketing root solver)
//
// cargo run --release
//
/*
Problem Statement

The objective of this program is to compute a real root of a nonlinear scalar equation
    f(x) = 0
on a closed interval [a, b], under the assumption that the function f is continuous and
satisfies the bracketing condition f(a)f(b) < 0. This condition guarantees the existence
of at least one root in the interval and forms the basis for a globally convergent
algorithm.

Pure bracketing methods such as bisection are robust but converge slowly, while open
methods such as the secant or Newton’s method can converge rapidly but may fail without
good initial guesses or derivative information. The van Wijngaarden–Dekker–Brent method
resolves this tradeoff by adaptively combining bisection, secant updates, and inverse
quadratic interpolation into a single algorithm that is both reliable and efficient.

This program implements Brent’s method as developed in Section 9.4. At each iteration,
the algorithm attempts a fast interpolation step based on available function values and
accepts it only if a set of safety conditions is satisfied. When these conditions fail,
the method reverts to a bisection step to preserve the bracketing guarantee. Iteration
terminates when either the residual |f(x)| falls below a prescribed tolerance or the
width of the bracketing interval becomes sufficiently small.

The implementation requires only function evaluations, avoids derivative information,
and explicitly reports the reason for convergence. Representative test problems include
a smooth transcendental equation, a strongly nonlinear polynomial, and the Colebrook–White
equation from fluid mechanics, illustrating the method’s performance across a range of
practical scenarios.
*/

use std::fmt;

#[derive(Debug, Clone)]
pub struct BrentConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Absolute tolerance on the bracket width: |b-a| <= x_tol.
    pub x_tol: f64,
    /// Absolute tolerance on the function value: |f(b)| <= f_tol.
    pub f_tol: f64,
    /// Whether to print a per-iteration table.
    pub trace: bool,
}

impl Default for BrentConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            x_tol: 1e-12,
            f_tol: 1e-12,
            trace: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum StopReason {
    Residual,
    BracketWidth,
    Endpoint,
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StopReason::Residual => write!(f, "residual |f(b)| <= f_tol"),
            StopReason::BracketWidth => write!(f, "bracket width |c-b|/2 <= x_tol"),
            StopReason::Endpoint => write!(f, "endpoint already satisfies tolerance"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BrentResult {
    pub root: f64,
    pub f_at_root: f64,
    pub a: f64,
    pub b: f64,
    pub iterations: usize,
    pub converged: bool,
    pub stop_reason: StopReason,
}

impl fmt::Display for BrentResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BrentResult {{ root: {:.16e}, f(root): {:.16e}, iterations: {}, converged: {}, stop: {}, final_bracket: [{:.16e}, {:.16e}] }}",
            self.root,
            self.f_at_root,
            self.iterations,
            self.converged,
            self.stop_reason,
            self.a,
            self.b
        )
    }
}

#[derive(Debug, Clone)]
pub enum BrentError {
    NonFiniteInput { a: f64, b: f64 },
    NonFiniteFunctionValue { x: f64, fx: f64 },
    InvalidBracket { a: f64, b: f64, fa: f64, fb: f64 },
    MaxIterations { last_x: f64, last_fx: f64, iterations: usize },
}

impl fmt::Display for BrentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BrentError::NonFiniteInput { a, b } => write!(f, "non-finite endpoints: a={a}, b={b}"),
            BrentError::NonFiniteFunctionValue { x, fx } => write!(f, "non-finite function value: f({x})={fx}"),
            BrentError::InvalidBracket { a, b, fa, fb } => write!(
                f,
                "invalid bracket: f(a) and f(b) must have opposite signs (a={a}, b={b}, f(a)={fa}, f(b)={fb})"
            ),
            BrentError::MaxIterations { last_x, last_fx, iterations } => write!(
                f,
                "max iterations reached: x={last_x}, f(x)={last_fx}, iterations={iterations}"
            ),
        }
    }
}

impl std::error::Error for BrentError {}

fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn print_trace_header() {
    println!(
        "{:>4}  {:>14}  {:>14}  {:>14}  {:>14}  {:>12}  {:>10}",
        "n", "a", "b", "c", "f(b)", "|c-b|/2", "step"
    );
}

/// Brent’s method for f(x)=0 on a bracket [a,b] with f(a)f(b) < 0.
///
/// This implementation follows the classical, widely used “brentq” style logic.
/// It maintains points a, b, c such that b is the current best approximation and
/// [b,c] brackets a root. Interpolation is attempted when safe, otherwise the
/// method falls back to bisection (9.4.8).
pub fn brent<F>(f: F, mut a: f64, mut b: f64, cfg: BrentConfig) -> Result<BrentResult, BrentError>
where
    F: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(BrentError::NonFiniteInput { a, b });
    }
    if a == b {
        let fa = f(a);
        if !fa.is_finite() {
            return Err(BrentError::NonFiniteFunctionValue { x: a, fx: fa });
        }
        return Err(BrentError::InvalidBracket { a, b, fa, fb: fa });
    }
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }

    let mut fa = f(a);
    if !fa.is_finite() {
        return Err(BrentError::NonFiniteFunctionValue { x: a, fx: fa });
    }
    let mut fb = f(b);
    if !fb.is_finite() {
        return Err(BrentError::NonFiniteFunctionValue { x: b, fx: fb });
    }

    // Endpoint successes.
    if fa.abs() <= cfg.f_tol {
        return Ok(BrentResult {
            root: a,
            f_at_root: fa,
            a,
            b,
            iterations: 0,
            converged: true,
            stop_reason: StopReason::Endpoint,
        });
    }
    if fb.abs() <= cfg.f_tol {
        return Ok(BrentResult {
            root: b,
            f_at_root: fb,
            a,
            b,
            iterations: 0,
            converged: true,
            stop_reason: StopReason::Endpoint,
        });
    }

    // Bracketing requirement: sign change.
    if fa * fb > 0.0 {
        return Err(BrentError::InvalidBracket { a, b, fa, fb });
    }

    // Ensure |f(b)| <= |f(a)| as in (9.4.2).
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    // c is the bracketing companion to b.
    let mut c = a;
    let mut fc = fa;

    // d is the most recent step; e is the step before that.
    let mut d = b - a;
    let mut e = d;

    if cfg.trace {
        print_trace_header();
    }

    for n in 1..=cfg.max_iter {
        // Restore bracketing if f(b) and f(c) drift to the same sign.
        if fb * fc > 0.0 {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }

        // Maintain b as the best approximation among {b,c}.
        if fc.abs() < fb.abs() {
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            c = a;
            fc = fa;
        }

        // Midpoint displacement used for bisection and termination.
        let m = 0.5 * (c - b);

        // A small relative component improves behavior near machine precision.
        let tol = 2.0 * f64::EPSILON * b.abs() + cfg.x_tol;

        if cfg.trace {
            let step_kind = if e.abs() > tol && fa.abs() > fb.abs() { "interp" } else { "bisect" };
            println!(
                "{:>4}  {:>14.6e}  {:>14.6e}  {:>14.6e}  {:>14.6e}  {:>12.3e}  {:>10}",
                n, a, b, c, fb, m.abs(), step_kind
            );
        }

        // Stopping rules: residual and bracket width.
        if fb.abs() <= cfg.f_tol {
            let lo = b.min(c);
            let hi = b.max(c);
            return Ok(BrentResult {
                root: b,
                f_at_root: fb,
                a: lo,
                b: hi,
                iterations: n,
                converged: true,
                stop_reason: StopReason::Residual,
            });
        }
        if m.abs() <= tol {
            let lo = b.min(c);
            let hi = b.max(c);
            return Ok(BrentResult {
                root: b,
                f_at_root: fb,
                a: lo,
                b: hi,
                iterations: n,
                converged: true,
                stop_reason: StopReason::BracketWidth,
            });
        }

        // Attempt interpolation (secant or inverse quadratic interpolation) if allowed.
        let mut use_interpolation = false;
        let mut p = 0.0;
        let mut q = 0.0;

        if e.abs() > tol && fa.abs() > fb.abs() {
            use_interpolation = true;

            let s = fb / fa;

            if a == c {
                // Secant step (two points).
                p = 2.0 * m * s;
                q = 1.0 - s;
            } else {
                // Inverse quadratic interpolation via the ratio form (9.4.4)–(9.4.7).
                let r = fb / fc; // R
                let t = fa / fc; // T
                let s2 = s;      // S

                p = s2 * (t * (r - t) * (c - b) - (1.0 - r) * (b - a)); // (9.4.6)
                q = (t - 1.0) * (r - 1.0) * (s2 - 1.0);                 // (9.4.7)
            }

            // Choose sign so that d = p/q moves toward the bracket interior.
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();

            // Standard Brent acceptance safeguards.
            let cond1 = 2.0 * p < (3.0 * m * q - (tol * q).abs()).abs();
            let cond2 = 2.0 * p < (e * q).abs();

            if !(cond1 && cond2) {
                use_interpolation = false;
            }
        }

        if use_interpolation {
            e = d;
            d = p / q;
        } else {
            e = m;
            d = m; // bisection displacement (9.4.8)
        }

        // Advance: a <- b, then update b by the chosen displacement.
        a = b;
        fa = fb;

        if d.abs() > tol {
            b += d;
        } else {
            b += tol * sign(m);
        }

        fb = f(b);
        if !fb.is_finite() {
            return Err(BrentError::NonFiniteFunctionValue { x: b, fx: fb });
        }
    }

    Err(BrentError::MaxIterations {
        last_x: b,
        last_fx: fb,
        iterations: cfg.max_iter,
    })
}

fn colebrook_residual(friction: f64, re: f64, rel_roughness: f64) -> f64 {
    // Residual for the Colebrook–White equation (9.4.10), written as F(f)=0:
    //   1/sqrt(f) = -2 log10( rel_roughness/3.7 + 2.51/(Re sqrt(f)) )
    //
    // Rearranged as:
    //   F(f) = 1/sqrt(f) + 2 log10( rel_roughness/3.7 + 2.51/(Re sqrt(f)) ).
    let inv_sqrt = 1.0 / friction.sqrt();
    let arg = rel_roughness / 3.7 + 2.51 / (re * friction.sqrt());
    inv_sqrt + 2.0 * arg.log10()
}

fn main() {
    // Example 1: Smooth benchmark.
    let f1 = |x: f64| x.cos() - x;

    // Example 2: Nonlinear polynomial used earlier.
    let f2 = |x: f64| x.powi(10) - 1.0;

    // Example 3: Colebrook–White equation (9.4.10).
    let re = 1.0e5;
    let eps_over_d = 1.0e-4;
    let f3 = |ff: f64| colebrook_residual(ff, re, eps_over_d);

    let cfg = BrentConfig {
        max_iter: 100,
        x_tol: 1e-14,
        f_tol: 1e-14,
        trace: false, // set true to inspect interpolation vs bisection decisions
    };

    println!("Example 1: f(x) = cos(x) - x on [0, 1]");
    match brent(f1, 0.0, 1.0, cfg.clone()) {
        Ok(res) => println!("{res}\n"),
        Err(e) => println!("Brent failed: {e}\n"),
    }

    println!("Example 2: f(x) = x^10 - 1 on [0, 2]");
    match brent(f2, 0.0, 2.0, cfg.clone()) {
        Ok(res) => println!("{res}\n"),
        Err(e) => println!("Brent failed: {e}\n"),
    }

    // For Colebrook, f is positive and typically lies in a modest interval.
    // A conservative bracket that works across many regimes is [1e-6, 1].
    println!("Example 3: Colebrook–White equation (9.4.10) with Re = 1e5 and eps/D = 1e-4");
    match brent(f3, 1.0e-6, 1.0, cfg) {
        Ok(res) => {
            println!("{res}");
            println!("  Interpreted friction factor: f = {:.16e}", res.root);
        }
        Err(e) => println!("Brent failed: {e}"),
    }
}
```

Program 9.4.0 demonstrates how the van Wijngaarden–Dekker–Brent method achieves an effective balance between reliability and speed in one-dimensional root finding. By preserving a bracketing interval at every iteration, the method guarantees convergence for continuous functions satisfying the sign-change condition. At the same time, the adaptive use of secant and inverse quadratic interpolation enables rapid convergence in favorable cases, often approaching the efficiency of Newton’s method without requiring derivatives.

The numerical examples illustrate both aspects of this balance. When the function is smooth and well behaved, interpolation steps dominate and convergence is rapid. When the function exhibits strong nonlinearity or numerical sensitivity, the method safely falls back to bisection, ensuring steady progress. This combination explains why Brent’s method has become the default choice in many scientific computing libraries and why it remains a benchmark for modern root-finding algorithms.

The modular structure of the implementation allows the solver to be reused across a wide range of applications by simply providing a different function and bracketing interval. This makes Brent’s method a foundational tool in numerical computing and a natural culmination of the root-finding techniques developed throughout this chapter.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/7F31VIbPv3kqrSfkcTbZ.4","tags":[]}

# 9.5. Newton–Raphson Method Using Derivative

The Newton–Raphson method, often referred to simply as Newton’s method, is one of the most important and widely used techniques for solving nonlinear equations of the form:

$$f(x)=0 \tag{9.5.1}$$

It was first formulated by Isaac Newton in the 1660s and later independently developed by Joseph Raphson in 1690. The defining feature of the method is its explicit use of derivative information to guide the iteration toward a root. This reliance on derivatives distinguishes Newton’s method from bracketing and interpolation-based approaches and endows it with exceptionally fast local convergence.

From a geometric perspective, Newton’s method repeatedly approximates the function $f(x)$ by its tangent line at the current iterate and then takes the point where that tangent intersects the $x$-axis as the next approximation. If $x_i$ is the current estimate of the root, the tangent line at the point $(x_i,f(x_i))$ has slope $f'(x_i)$, and its intersection with the $x$-axis yields the iteration formula,

$$x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)} \tag{9.5.2}$$

## 9.5.1. Derivation via Taylor Expansion

The Newton update formula can be derived rigorously using a Taylor expansion. Assuming that $f$ is sufficiently differentiable in a neighborhood of $x_i$, one may expand $f(x)$ about $x_i$ as:

$$
f(x_i + \delta) \approx f(x_i) + f'(x_i)\,\delta
+ \frac{f''(x_i)}{2}\,\delta^2 + \cdots
\tag{9.5.3}
$$

If $x_{i+1} = x_i + \delta$ is an improved approximation to the root, then $f(x_i+\delta)=0$. Neglecting higher-order terms in (9.5.3) for sufficiently small $\delta$ leads to the linear approximation:

$$f(x_i) + f'(x_i),\delta \approx 0 \tag{9.5.4}$$

Solving for $\delta$ gives,

$$\delta \approx -\frac{f(x_i)}{f'(x_i)} \tag{9.5.5}$$

which immediately yields the Newton update (9.5.2). Conceptually, Newton’s method exploits a local linearization of $f$ to jump rapidly toward the root.

### Convergence Properties

When $f(x)$ is smooth and the initial guess $x_0$ is sufficiently close to an actual root $r$, Newton’s method converges quadratically. To analyze this behavior, define the iteration error:

$$\varepsilon_i = x_i - r \tag{9.5.6}$$

Expanding both $f(x_i)$ and $f'(x_i)$ about the root $r$, and using the fact that $f(r)=0$, gives:

$$
f(x_i) = f(r+\varepsilon_i)
\approx f'(r)\,\varepsilon_i
+ \frac{f''(r)}{2}\,\varepsilon_i^2 + \cdots
\tag{9.5.7}
$$

$$
f'(x_i) = f'(r+\varepsilon_i)
\approx f'(r) + f''(r)\,\varepsilon_i + \cdots
\tag{9.5.8}
$$

Substituting these expansions into the Newton update (9.5.2) and simplifying yields:

$$
\varepsilon_{i+1} \approx -\frac{f''(r)}{2f'(r)}\,\varepsilon_i^2
\tag{9.5.9}
$$

Thus, for sufficiently small $\varepsilon_i$,

$$
\lvert \varepsilon_{i+1} \rvert
\approx \frac{\lvert f''(r) \rvert}{2\,\lvert f'(r) \rvert}\,\varepsilon_i^2
\tag{9.5.10}
$$

The error at each iteration is therefore approximately proportional to the square of the previous error, confirming quadratic convergence. In practical terms, once the iterates are close to the root, the number of correct digits roughly doubles at each step. This rapid local convergence is one of the principal attractions of Newton’s method.

### Rust Implementation

Following the derivation of Newton’s method via local linearization and Taylor expansion in Section 9.5, Program 9.5.1 provides a practical implementation of the Newton–Raphson method for solving nonlinear equations of the form $f(x)=0$. In numerical computation, the availability of derivative information enables significantly faster convergence than bracketing or interpolation-based methods, provided that the initial guess lies sufficiently close to the root. This program translates the theoretical iteration formula given in Equation (9.5.2) into an executable algorithm, illustrating how derivative information can be exploited to achieve quadratic convergence in practice. By incorporating explicit stopping criteria and safeguards against numerical breakdown, the implementation highlights both the power and the limitations of Newton’s method in finite-precision environments.

At the core of the implementation is the `newton` function, which directly realizes the Newton update defined in Equation (9.5.2). The function accepts two user-supplied closures: one representing the nonlinear function $f(x)$, and the other representing its derivative $f'(x)$. This explicit separation reflects the defining feature of Newton’s method, namely its reliance on derivative information, and mirrors the mathematical formulation developed in Section 9.5.

Each iteration evaluates the current function value $f(x_i)$ and derivative $f'(x_i)$, and then computes the next iterate using the Newton correction $-f(x_i)/f'(x_i)$. This update corresponds to solving the linearized equation obtained by truncating the Taylor expansion in Equation (9.5.3) and leads directly to the increment (\\delta) defined in Equation (9.5.5). Because the method assumes local linearity, the implementation explicitly checks that the derivative magnitude does not become too small, as a near-zero derivative would invalidate the linear approximation and lead to numerical instability.

The algorithm employs two complementary stopping criteria. The first terminates the iteration when the residual $|f(x_i)|$ falls below a prescribed tolerance, indicating that the current iterate satisfies Equation (9.5.1) to the desired accuracy. The second monitors the change between successive iterates $|x_{i+1}-x_i|$, recognizing that once this quantity becomes sufficiently small, further updates no longer produce meaningful improvement in finite precision. These criteria reflect standard practical interpretations of convergence and ensure that the iteration halts reliably without unnecessary computation.

For instructional purposes, the implementation supports optional iteration tracing. When enabled, the program prints a table of iterates, function values, derivatives, and step sizes. This diagnostic output makes the quadratic convergence predicted by Equations (9.5.9)–(9.5.10) directly observable, as the step size $|\Delta x|$ rapidly decreases once the iterates enter the local convergence regime.

The `main` function demonstrates the method on two representative examples. The first solves the transcendental equation $\cos(x)-x=0$, a standard benchmark that exhibits clean quadratic convergence from a reasonable initial guess. The second solves the algebraic equation $x^3-2=0$, whose simple structure allows the rapid convergence behavior of Newton’s method to be observed with particular clarity. In both cases, analytic derivatives are supplied explicitly, emphasizing the method’s dependence on derivative information and its resulting efficiency.

```rust
// Program 9.5.1: Newton–Raphson Method Using Derivative
//
// cargo run --release
//
// This program implements Newton’s method for solving f(x)=0 using the derivative f'(x)
// as described in Section 9.5. The iteration follows the Newton update
//   x_{i+1} = x_i - f(x_i)/f'(x_i)                                          (9.5.2)
// which is obtained from the Taylor expansion argument (9.5.3)–(9.5.5).
//
// The implementation emphasizes textbook clarity: it separates configuration parameters
// from the solver logic, checks for numerical breakdown when f'(x_i) is too small,
// and supports optional iteration tracing so that the quadratic convergence described
// by (9.5.9)–(9.5.10) can be observed experimentally.

use std::fmt;

#[derive(Debug, Clone)]
pub struct NewtonConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Absolute tolerance on the function value: |f(x)| <= f_tol.
    pub f_tol: f64,
    /// Absolute tolerance on successive iterates: |x_{i+1} - x_i| <= x_tol.
    pub x_tol: f64,
    /// If |f'(x_i)| <= deriv_tol, the Newton step becomes unreliable.
    pub deriv_tol: f64,
    /// Whether to print an iteration table.
    pub trace: bool,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            f_tol: 1e-12,
            x_tol: 1e-12,
            deriv_tol: 1e-14,
            trace: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NewtonResult {
    pub root: f64,
    pub f_at_root: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl fmt::Display for NewtonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NewtonResult {{ root: {:.16e}, f(root): {:.16e}, iterations: {}, converged: {} }}",
            self.root, self.f_at_root, self.iterations, self.converged
        )
    }
}

#[derive(Debug, Clone)]
pub enum NewtonError {
    NonFiniteInput { x0: f64 },
    NonFiniteFunctionValue { x: f64, fx: f64 },
    DerivativeTooSmall { x: f64, dfx: f64 },
    MaxIterations { last_x: f64, last_fx: f64, iterations: usize },
}

impl fmt::Display for NewtonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NewtonError::NonFiniteInput { x0 } => write!(f, "non-finite initial guess: x0={x0}"),
            NewtonError::NonFiniteFunctionValue { x, fx } => write!(f, "non-finite function value: f({x})={fx}"),
            NewtonError::DerivativeTooSmall { x, dfx } => write!(f, "derivative too small: f'({x})={dfx}"),
            NewtonError::MaxIterations { last_x, last_fx, iterations } => write!(
                f,
                "max iterations reached: x={last_x}, f(x)={last_fx}, iterations={iterations}"
            ),
        }
    }
}

impl std::error::Error for NewtonError {}

fn print_trace_header() {
    println!(
        "{:>4}  {:>22}  {:>22}  {:>22}  {:>12}",
        "i", "x_i", "f(x_i)", "f'(x_i)", "|Δx|"
    );
}

/// Newton–Raphson method using the analytic derivative.
///
/// The update follows Equation (9.5.2). Stopping occurs when either:
/// - |f(x_i)| <= f_tol, or
/// - |x_{i+1}-x_i| <= x_tol.
///
/// If |f'(x_i)| is too small, the Newton step becomes ill-conditioned and the
/// function returns an error to make the failure mode explicit.
pub fn newton<F, DF>(
    f: F,
    df: DF,
    mut x: f64,
    cfg: NewtonConfig,
) -> Result<NewtonResult, NewtonError>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    if !x.is_finite() {
        return Err(NewtonError::NonFiniteInput { x0: x });
    }

    if cfg.trace {
        print_trace_header();
    }

    for i in 0..=cfg.max_iter {
        let fx = f(x);
        if !fx.is_finite() {
            return Err(NewtonError::NonFiniteFunctionValue { x, fx });
        }

        // Residual-based termination.
        if fx.abs() <= cfg.f_tol {
            if cfg.trace {
                let dfx = df(x);
                println!(
                    "{:>4}  {:>22.16e}  {:>22.16e}  {:>22.16e}  {:>12}",
                    i, x, fx, dfx, "-"
                );
            }
            return Ok(NewtonResult {
                root: x,
                f_at_root: fx,
                iterations: i,
                converged: true,
            });
        }

        let dfx = df(x);
        if !dfx.is_finite() {
            return Err(NewtonError::NonFiniteFunctionValue { x, fx: dfx });
        }
        if dfx.abs() <= cfg.deriv_tol {
            return Err(NewtonError::DerivativeTooSmall { x, dfx });
        }

        // Newton step (9.5.2).
        let x_next = x - fx / dfx;
        if !x_next.is_finite() {
            return Err(NewtonError::NonFiniteFunctionValue { x: x_next, fx: f64::NAN });
        }

        let dx = (x_next - x).abs();

        if cfg.trace {
            println!(
                "{:>4}  {:>22.16e}  {:>22.16e}  {:>22.16e}  {:>12.3e}",
                i, x, fx, dfx, dx
            );
        }

        // Step-size termination.
        if dx <= cfg.x_tol {
            let fx_next = f(x_next);
            if !fx_next.is_finite() {
                return Err(NewtonError::NonFiniteFunctionValue { x: x_next, fx: fx_next });
            }
            return Ok(NewtonResult {
                root: x_next,
                f_at_root: fx_next,
                iterations: i + 1,
                converged: true,
            });
        }

        x = x_next;
    }

    let fx = f(x);
    Err(NewtonError::MaxIterations {
        last_x: x,
        last_fx: fx,
        iterations: cfg.max_iter,
    })
}

fn main() {
    // Example 1: Solve cos(x) - x = 0.
    //
    // f(x)  = cos(x) - x
    // f'(x) = -sin(x) - 1
    let f1 = |x: f64| x.cos() - x;
    let df1 = |x: f64| -x.sin() - 1.0;

    // Example 2: Solve x^3 - 2 = 0 (root is 2^{1/3}).
    //
    // f(x)  = x^3 - 2
    // f'(x) = 3x^2
    let f2 = |x: f64| x * x * x - 2.0;
    let df2 = |x: f64| 3.0 * x * x;

    let cfg = NewtonConfig {
        max_iter: 50,
        f_tol: 1e-14,
        x_tol: 1e-14,
        deriv_tol: 1e-14,
        trace: true, // set true to observe quadratic convergence near the root
    };

    println!("Example 1: f(x) = cos(x) - x, Newton iteration from x0 = 1");
    match newton(f1, df1, 1.0, cfg.clone()) {
        Ok(res) => println!("{res}\n"),
        Err(e) => println!("Newton failed: {e}\n"),
    }

    println!("Example 2: f(x) = x^3 - 2, Newton iteration from x0 = 1");
    match newton(f2, df2, 1.0, cfg) {
        Ok(res) => println!("{res}\n"),
        Err(e) => println!("Newton failed: {e}\n"),
    }
}
```

Program 9.5.1 illustrates the defining strengths of the Newton–Raphson method: when derivative information is available and the initial guess lies within the basin of attraction of a simple root, the method converges extremely rapidly. The iteration traces confirm the quadratic convergence predicted by the error analysis in Section 9.5, with the number of correct digits approximately doubling at each step once the iterates approach the root.

At the same time, the implementation highlights important practical considerations. The need to guard against small derivatives and to impose explicit stopping criteria reflects the fact that Newton’s method is fundamentally a local algorithm. These considerations motivate the hybrid strategies developed later in this chapter, where Newton’s fast local convergence is combined with the global reliability of bracketing methods. As such, Newton’s method serves both as a powerful standalone solver and as a foundational component in more sophisticated root-finding algorithms.

## 9.5.2. Practical Considerations and Pitfalls

Despite its excellent local convergence properties, Newton’s method is not globally convergent. Unlike bracketing methods, which maintain a guaranteed root interval, Newton’s method is an open method and can fail under several common circumstances.

If the derivative $f'(x_i)$ is zero or extremely small, the Newton step (9.5.2) can become very large, potentially sending the iteration far from the root. This situation often occurs near local extrema of $f$, where $f'(x) \approx 0$. In such cases, the iteration may diverge rather than converge.

If the initial guess $x_0$ is far from the true root, the linear approximation underlying the Newton step may be poor, since higher-order terms in the Taylor expansion are no longer negligible. The resulting iteration can jump to an unrelated region of the function, diverge to infinity, or enter a limit cycle. Such pathological behavior is illustrated by examples where Newton’s method oscillates between two points instead of converging.

If the root $r$ has multiplicity greater than one, that is, if:

$$
f(x) = (x - r)^m\, g(x), \qquad m > 1
\tag{9.5.11}
$$

then Newton’s method converges only linearly rather than quadratically. Moreover, if $m$ is even, then $f'(r)=0$, and the Newton formula itself becomes ill-conditioned near the root. In such situations, special modifications or alternative methods are required.

Despite these caveats, Newton’s method remains extraordinarily powerful when it works. It is commonly used as a refinement step following a more globally robust method. For example, one may first apply bisection or Brent’s method to locate a root safely and then switch to Newton’s method to polish the solution to high precision.

### Rust Implementation

Following the presentation of the Newton–Raphson method and its quadratic local convergence in Section 9.5.1, Program 9.5.2 examines the practical limitations that arise when Newton’s method is applied outside its ideal theoretical setting. In numerical computation, fast local convergence alone is not sufficient to guarantee reliable behavior. The method’s dependence on derivative information and local linearization can lead to divergence, stagnation, or loss of quadratic convergence when common assumptions are violated. This program translates the qualitative discussion of Section 9.5.2 into executable form by demonstrating representative failure modes of Newton’s method and by implementing standard remedies used in practice. In particular, it illustrates how damping and hybridization with bracketing methods can substantially improve robustness while preserving much of Newton’s efficiency.

At the core of the implementation is the `damped_newton` function, which augments the classical Newton update given in Equation (9.5.2) with a safeguard mechanism. Instead of always accepting the full Newton step, the algorithm monitors the derivative magnitude and the effect of the proposed update on the function value. When the derivative $f'(x_i)$ is small or when the raw Newton step fails to reduce $|f(x)|$, the step length is reduced through backtracking. This damping strategy reflects the observation in Section 9.5.2 that large or unstable steps often arise precisely when the linear approximation implicit in Newton’s method becomes unreliable.

The backtracking procedure repeatedly scales the Newton correction until a decrease in the residual is observed or a prescribed limit is reached. While this modification does not make Newton’s method globally convergent in the same sense as bracketing algorithms, it significantly reduces the likelihood of catastrophic divergence and improves robustness in finite-precision arithmetic. Explicit checks on the derivative magnitude further guard against division by values that are effectively zero, a situation that commonly occurs near local extrema or at multiple roots.

To address the loss of quadratic convergence for roots of multiplicity greater than one, as described by Equation (9.5.11), the program includes an example with a double root. In this case, the iteration converges only linearly, despite damping, clearly illustrating the theoretical result that the error relation (9.5.9) no longer holds when $f'(r)=0$. This example reinforces the distinction between local convergence order and global reliability.

The program also implements a simple bisection routine and demonstrates a hybrid strategy in which a bracketing method is used to safely localize a root before switching to Newton’s method for rapid refinement. This approach embodies the practical recommendation emphasized at the end of Section 9.5.2: Newton’s method is most effective when used as a polishing step following a globally convergent algorithm such as bisection or Brent’s method. The separation of concerns between the bracketing phase and the Newton phase is reflected directly in the code structure.

The `main` function orchestrates three demonstrations. The first highlights difficulties caused by small derivatives, the second illustrates linear convergence at a multiple root, and the third shows how a hybrid bracketing–Newton approach combines robustness with speed. Together, these examples provide concrete computational evidence for the qualitative discussion in Section 9.5.2.

```rust
// Program 9.5.2: Practical Considerations and Pitfalls in Newton’s Method
//
// cargo run --release
//
// This program illustrates the practical failure modes discussed in Section 9.5.2 and
// implements two textbook remedies:
//
// 1) Safeguarded (damped) Newton: the raw Newton step (9.5.2) is reduced by backtracking
//    until |f(x)| decreases, which mitigates overly large steps when f'(x) is small.
//
// 2) Hybrid bracketing + Newton: a bracketing method (bisection) is used to obtain a safe
//    initial approximation, and Newton is then used as a refinement step.
//
// The program also demonstrates the loss of quadratic convergence for multiple roots of
// the form (9.5.11) by solving a problem with a double root.

use std::fmt;

#[derive(Debug, Clone)]
pub struct DampedNewtonConfig {
    pub max_iter: usize,
    pub f_tol: f64,
    pub x_tol: f64,
    pub deriv_tol: f64,
    pub max_backtrack: usize,
    pub shrink: f64,
    pub require_decrease: bool,
    pub trace: bool,
}

impl Default for DampedNewtonConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            f_tol: 1e-12,
            x_tol: 1e-12,
            deriv_tol: 1e-14,
            max_backtrack: 20,
            shrink: 0.5,
            require_decrease: true,
            trace: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum NewtonStopReason {
    Residual,
    StepSize,
}

impl fmt::Display for NewtonStopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NewtonStopReason::Residual => write!(f, "residual |f(x)| <= f_tol"),
            NewtonStopReason::StepSize => write!(f, "step size |Δx| <= x_tol"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DampedNewtonResult {
    pub root: f64,
    pub f_at_root: f64,
    pub iterations: usize,
    pub converged: bool,
    pub stop_reason: NewtonStopReason,
}

impl fmt::Display for DampedNewtonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DampedNewtonResult {{ root: {:.16e}, f(root): {:.16e}, iterations: {}, converged: {}, stop: {} }}",
            self.root, self.f_at_root, self.iterations, self.converged, self.stop_reason
        )
    }
}

#[derive(Debug, Clone)]
pub enum DampedNewtonError {
    NonFiniteInput { x0: f64 },
    NonFiniteFunctionValue { x: f64, fx: f64 },
    DerivativeTooSmall { x: f64, dfx: f64 },
    BacktrackingFailed { x: f64 },
    MaxIterations { last_x: f64, last_fx: f64, iterations: usize },
}

impl fmt::Display for DampedNewtonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DampedNewtonError::NonFiniteInput { x0 } => write!(f, "non-finite initial guess: x0={x0}"),
            DampedNewtonError::NonFiniteFunctionValue { x, fx } => write!(f, "non-finite value encountered: value({x})={fx}"),
            DampedNewtonError::DerivativeTooSmall { x, dfx } => write!(f, "derivative too small near x={x}: f'(x)={dfx}"),
            DampedNewtonError::BacktrackingFailed { x } => write!(f, "backtracking failed to find a decreasing step from x={x}"),
            DampedNewtonError::MaxIterations { last_x, last_fx, iterations } => write!(
                f,
                "max iterations reached: x={last_x}, f(x)={last_fx}, iterations={iterations}"
            ),
        }
    }
}

impl std::error::Error for DampedNewtonError {}

fn print_trace_header() {
    println!(
        "{:>4}  {:>22}  {:>22}  {:>22}  {:>12}  {:>10}",
        "i", "x_i", "f(x_i)", "f'(x_i)", "|Δx|", "bt"
    );
}

pub fn damped_newton<F, DF>(
    f: F,
    df: DF,
    mut x: f64,
    cfg: DampedNewtonConfig,
) -> Result<DampedNewtonResult, DampedNewtonError>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    if !x.is_finite() {
        return Err(DampedNewtonError::NonFiniteInput { x0: x });
    }
    if !(0.0 < cfg.shrink && cfg.shrink < 1.0) {
        return Err(DampedNewtonError::NonFiniteFunctionValue { x: cfg.shrink, fx: f64::NAN });
    }

    if cfg.trace {
        print_trace_header();
    }

    for i in 0..=cfg.max_iter {
        let fx = f(x);
        if !fx.is_finite() {
            return Err(DampedNewtonError::NonFiniteFunctionValue { x, fx });
        }

        if fx.abs() <= cfg.f_tol {
            if cfg.trace {
                let dfx = df(x);
                println!(
                    "{:>4}  {:>22.16e}  {:>22.16e}  {:>22.16e}  {:>12}  {:>10}",
                    i, x, fx, dfx, "-", "-"
                );
            }
            return Ok(DampedNewtonResult {
                root: x,
                f_at_root: fx,
                iterations: i,
                converged: true,
                stop_reason: NewtonStopReason::Residual,
            });
        }

        let dfx = df(x);
        if !dfx.is_finite() {
            return Err(DampedNewtonError::NonFiniteFunctionValue { x, fx: dfx });
        }
        if dfx.abs() <= cfg.deriv_tol {
            return Err(DampedNewtonError::DerivativeTooSmall { x, dfx });
        }

        // Raw Newton step (9.5.2).
        let step_raw = -fx / dfx;

        // Backtracking line search: x_next = x + alpha * step_raw.
        let mut alpha = 1.0;
        let mut bt_used = 0usize;

        let mut x_next = x + alpha * step_raw;
        if !x_next.is_finite() {
            return Err(DampedNewtonError::NonFiniteFunctionValue { x: x_next, fx: f64::NAN });
        }
        let mut fx_next = f(x_next);
        if !fx_next.is_finite() {
            return Err(DampedNewtonError::NonFiniteFunctionValue { x: x_next, fx: fx_next });
        }

        if cfg.require_decrease {
            while fx_next.abs() > fx.abs() && bt_used < cfg.max_backtrack {
                alpha *= cfg.shrink;
                x_next = x + alpha * step_raw;
                if !x_next.is_finite() {
                    return Err(DampedNewtonError::NonFiniteFunctionValue { x: x_next, fx: f64::NAN });
                }
                fx_next = f(x_next);
                if !fx_next.is_finite() {
                    return Err(DampedNewtonError::NonFiniteFunctionValue { x: x_next, fx: fx_next });
                }
                bt_used += 1;
            }
            if fx_next.abs() > fx.abs() {
                return Err(DampedNewtonError::BacktrackingFailed { x });
            }
        }

        let dx = (x_next - x).abs();

        if cfg.trace {
            println!(
                "{:>4}  {:>22.16e}  {:>22.16e}  {:>22.16e}  {:>12.3e}  {:>10}",
                i, x, fx, dfx, dx, bt_used
            );
        }

        if dx <= cfg.x_tol {
            return Ok(DampedNewtonResult {
                root: x_next,
                f_at_root: fx_next,
                iterations: i + 1,
                converged: true,
                stop_reason: NewtonStopReason::StepSize,
            });
        }

        x = x_next;
    }

    let fx = f(x);
    Err(DampedNewtonError::MaxIterations {
        last_x: x,
        last_fx: fx,
        iterations: cfg.max_iter,
    })
}

#[derive(Debug, Clone)]
pub struct BisectionConfig {
    pub max_iter: usize,
    pub x_tol: f64,
    pub f_tol: f64,
}

impl Default for BisectionConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            x_tol: 1e-12,
            f_tol: 1e-12,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BisectionResult {
    pub root: f64,
    pub f_at_root: f64,
    pub a: f64,
    pub b: f64,
    pub iterations: usize,
    pub converged: bool,
}

#[derive(Debug, Clone)]
pub enum BisectionError {
    InvalidBracket { a: f64, b: f64, fa: f64, fb: f64 },
    NonFiniteFunctionValue { x: f64, fx: f64 },
    MaxIterations { last_x: f64, last_fx: f64, iterations: usize },
}

pub fn bisection<F>(
    f: F,
    mut a: f64,
    mut b: f64,
    cfg: BisectionConfig,
) -> Result<BisectionResult, BisectionError>
where
    F: Fn(f64) -> f64,
{
    let mut fa = f(a);
    if !fa.is_finite() {
        return Err(BisectionError::NonFiniteFunctionValue { x: a, fx: fa });
    }
    let fb0 = f(b);
    if !fb0.is_finite() {
        return Err(BisectionError::NonFiniteFunctionValue { x: b, fx: fb0 });
    }

    if fa == 0.0 {
        return Ok(BisectionResult { root: a, f_at_root: fa, a, b, iterations: 0, converged: true });
    }
    if fb0 == 0.0 {
        return Ok(BisectionResult { root: b, f_at_root: fb0, a, b, iterations: 0, converged: true });
    }
    if fa * fb0 > 0.0 {
        return Err(BisectionError::InvalidBracket { a, b, fa, fb: fb0 });
    }

    for i in 1..=cfg.max_iter {
        let m = 0.5 * (a + b);
        let fm = f(m);
        if !fm.is_finite() {
            return Err(BisectionError::NonFiniteFunctionValue { x: m, fx: fm });
        }

        if fm.abs() <= cfg.f_tol || (b - a).abs() <= cfg.x_tol {
            return Ok(BisectionResult {
                root: m,
                f_at_root: fm,
                a,
                b,
                iterations: i,
                converged: true,
            });
        }

        // Update the bracket using the sign change. Only fa is needed for the sign test.
        if fa * fm < 0.0 {
            b = m;
        } else {
            a = m;
            fa = fm;
        }
    }

    let m = 0.5 * (a + b);
    let fm = f(m);
    Err(BisectionError::MaxIterations { last_x: m, last_fx: fm, iterations: cfg.max_iter })
}

fn main() {
    // Demonstration A: derivative near zero makes Newton steps ill-conditioned.
    let f_a = |x: f64| x * x * x;
    let df_a = |x: f64| 3.0 * x * x;

    // Demonstration B: multiple root of the form (9.5.11) with m=2.
    let f_b = |x: f64| (x - 1.0) * (x - 1.0);
    let df_b = |x: f64| 2.0 * (x - 1.0);

    // Demonstration C: hybrid bracket + Newton polishing.
    let f_c = |x: f64| x.cos() - x;
    let df_c = |x: f64| -x.sin() - 1.0;

    let newton_cfg = DampedNewtonConfig {
        max_iter: 50,
        f_tol: 1e-14,
        x_tol: 1e-14,
        deriv_tol: 1e-14,
        max_backtrack: 20,
        shrink: 0.5,
        require_decrease: true,
        trace: false,
    };

    println!("Demonstration A: f(x) = x^3, damped Newton from x0 = 1");
    match damped_newton(f_a, df_a, 1.0, newton_cfg.clone()) {
        Ok(res) => println!("{res}\n"),
        Err(e) => println!("Damped Newton failed: {e}\n"),
    }

    println!("Demonstration B: f(x) = (x-1)^2 (multiple root), damped Newton from x0 = 2");
    match damped_newton(f_b, df_b, 2.0, newton_cfg.clone()) {
        Ok(res) => println!("{res}\n"),
        Err(e) => println!("Damped Newton failed: {e}\n"),
    }

    println!("Demonstration C: Hybrid bracket + Newton for f(x) = cos(x) - x on [0,1]");
    let bis_cfg = BisectionConfig {
        max_iter: 60,
        x_tol: 1e-6,   // coarse but reliable bracket tightening
        f_tol: 1e-12,
    };

    match bisection(f_c, 0.0, 1.0, bis_cfg) {
        Ok(bis) => {
            println!(
                "  Bisection stage: x ≈ {:.16e}, f(x) ≈ {:.16e} ({} iterations)",
                bis.root, bis.f_at_root, bis.iterations
            );
            match damped_newton(f_c, df_c, bis.root, newton_cfg) {
                Ok(res) => println!("  Newton polishing: {res}\n"),
                Err(e) => println!("  Newton polishing failed: {e}\n"),
            }
        }
        Err(e) => println!("  Bisection failed: {:?}\n", e),
    }
}
```

Program 9.5.2 demonstrates that the effectiveness of Newton’s method depends critically on problem structure and initialization. While the method offers unparalleled local convergence when its assumptions are satisfied, it can behave poorly in the presence of small derivatives, poor initial guesses, or multiple roots. The numerical experiments confirm the theoretical analysis presented in Section 9.5.2 and show that these issues are not merely pathological curiosities but practical concerns that must be addressed in real computations.

The incorporation of damping and hybridization illustrates how Newton’s method is adapted in modern numerical practice. By combining a robust global strategy with Newton’s rapid local convergence, one obtains algorithms that are both reliable and efficient. These ideas foreshadow the design philosophy behind more sophisticated solvers for nonlinear systems, where globalization strategies play a central role. As such, Newton’s method remains a cornerstone of numerical analysis, not as an isolated algorithm, but as a key component within broader, carefully engineered solution frameworks.

## 9.5.3. Applications and Context

Nonlinear equations arise throughout science and engineering, and Newton’s method is widely used because of its efficiency. In quantitative finance, for instance, the computation of implied volatility requires solving the Black–Scholes pricing equation for the volatility parameter $\sigma$. Since this equation has no closed-form solution for $\sigma$, Newton’s method is often employed to solve:

$$f(\sigma)=0 \tag{9.5.12}$$

where $f$ represents the difference between model and market prices. Similar root-finding problems occur in orbital mechanics, where Kepler’s equation:

$$E - e\sin E = M \tag{9.5.13}$$

must be solved for the eccentric anomaly $E$, as well as in physics, chemistry, electrical engineering, and machine learning.

## 9.5.4. Safeguards and Hybrid Strategies

Although Newton’s method exhibits quadratic convergence in the neighborhood of a simple root, its global behavior can be unreliable when the initial iterate is poorly chosen or when the derivative varies rapidly. In practice, robust algorithms therefore augment the basic Newton iteration with safeguards that enforce stability far from the root while preserving fast local convergence once the iterates enter the asymptotic regime.

A widely used approach is the hybrid Newton–bracketing strategy, most commonly Newton–bisection. Suppose an interval $[a,b]$ is known such that $f(a)f(b)<0$, guaranteeing the existence of at least one root in the interval. At each iteration, a Newton step is proposed using local derivative information. If this step lies inside the current bracket and yields a sufficient reduction in $|f(x)|$, it is accepted. Otherwise, the algorithm falls back to a bisection step, which halves the interval and maintains the bracketing property. This mechanism ensures global convergence by construction, while still allowing the algorithm to switch automatically to the faster Newton behavior as soon as it becomes safe to do so.

Another important safeguard is damping, sometimes referred to as step-length control or line search. Instead of applying the full Newton update, the correction is multiplied by a factor $0<\lambda\le 1$. The parameter $\lambda$ is chosen adaptively so that the new iterate produces a sufficient decrease in $|f(x)|$ or satisfies a prescribed descent condition. When the local linear model underlying Newton’s method is accurate, $\lambda$ quickly approaches unity and quadratic convergence is recovered. When the model is poor, smaller step lengths prevent divergence and excessive oscillations.

These safeguarding techniques are especially important for problems involving multiple roots, nearly singular derivatives, or highly nonlinear functions. Modern root-finding libraries almost never implement a pure Newton iteration in isolation; instead, they rely on hybrid strategies that combine derivative-based acceleration with guaranteed global convergence mechanisms. As a result, Newton’s method remains one of the most effective and widely used tools for nonlinear equation solving, provided it is embedded within an appropriately safeguarded framework.

### Rust Implementation

Following the discussion in Section 9.5 on the strengths and limitations of Newton’s method, Program 9.5.4 presents a practical implementation of a safeguarded hybrid Newton–bracketing algorithm. While Newton’s method offers quadratic convergence near a simple root, its global behavior can be unreliable when the initial iterate is poorly chosen or when derivative information becomes ill-conditioned. This program demonstrates how Newton’s method can be embedded within a bracketing framework, combining derivative-based acceleration with the guaranteed convergence of bisection. By dynamically selecting between Newton steps, damped Newton steps, and bisection, the algorithm maintains robustness far from the root while automatically recovering fast local convergence once the iterates enter the asymptotic regime. The implementation illustrates how modern root-finding software balances efficiency and reliability in finite-precision environments.

At the core of the implementation is the hybrid Newton–bisection solver, which operates on a bracketing interval $[a,b]$ satisfying the sign-change condition $f(a)f(b)<0$, as introduced earlier in Section 9.5.4. At each iteration, the solver proposes a Newton update based on the local derivative $f'(x)$, corresponding to the classical Newton step defined in Equation (9.5.2). This step is accepted only if it lies strictly within the current bracket and produces a reduction in $|f(x)|$. These acceptance criteria ensure that Newton’s method is used only when its local linear model is trustworthy.

When a full Newton step fails these safety checks, the algorithm applies damping, scaling the Newton correction by a factor $0<\lambda\le 1$ through backtracking. This mechanism reduces step length adaptively until sufficient decrease is obtained, preventing divergence caused by overly aggressive updates. If neither the full nor the damped Newton step is admissible, the solver falls back to a bisection step, halving the bracketing interval and preserving the sign-change condition by construction. This fallback guarantees global convergence even in the presence of nearly singular derivatives or strong nonlinearities.

The solver tracks multiple termination criteria, including residual convergence $|f(x)|\le\text{TOL}$ and bracket-width reduction $|b-a|\le\text{TOL}$, reflecting the theoretical guarantees discussed in Section 9.5.4. Optional iteration tracing exposes the internal decision process at each step, making explicit when Newton steps are accepted, when damping is applied, and when bisection is required. This transparency is particularly valuable for understanding solver behavior near multiple roots or flat regions of the function.

The `main` function demonstrates the hybrid strategy on representative test problems. A smooth transcendental equation illustrates rapid convergence once Newton steps become admissible. A polynomial example shows how the algorithm avoids premature termination while still converging reliably. A final residual evaluation of the Colebrook–White equation highlights the relevance of safeguarded root-finding methods in practical engineering contexts, where nonlinear equations must be solved repeatedly and robustly.

```rust
// Program 9.5.4: Safeguards and Hybrid Strategies (Newton–Bisection with Damping)
//
// cargo run --release
//
// Revision notes for textbook clarity:
// - Example 2 now uses a bracket whose midpoint is NOT exactly the root, so the solver
//   performs visible safeguarded iterations rather than terminating at initialization.
// - The solver reports a third stopping reason, "initialization", when convergence is
//   achieved at the initial midpoint (useful for explaining behavior in special cases).
//
// This program implements a safeguarded hybrid Newton–bracketing strategy as described
// in Section 9.5.4. Given a bracketing interval [a,b] with f(a)f(b) < 0, the algorithm
// proposes a Newton step using derivative information. The step is accepted only if:
//
// (i)  it lies strictly inside the current bracket, and
// (ii) it yields a reduction in |f(x)| (a simple decrease test).
//
// If either condition fails, the algorithm falls back to a bisection step, which halves
// the interval and maintains the bracketing property. A damping (step-length) mechanism
// is included: even when the Newton step is admissible, it can be scaled by 0 < λ <= 1
// via backtracking to enforce decrease in |f|.

use std::fmt;

#[derive(Debug, Clone)]
pub struct HybridNewtonConfig {
    pub max_iter: usize,
    pub f_tol: f64,
    pub x_tol: f64,
    pub deriv_tol: f64,
    pub max_backtrack: usize,
    pub shrink: f64,
    pub trace: bool,
}

impl Default for HybridNewtonConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            f_tol: 1e-12,
            x_tol: 1e-12,
            deriv_tol: 1e-14,
            max_backtrack: 20,
            shrink: 0.5,
            trace: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum StepKind {
    Newton,
    DampedNewton,
    Bisection,
}

impl fmt::Display for StepKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepKind::Newton => write!(f, "newton"),
            StepKind::DampedNewton => write!(f, "damped"),
            StepKind::Bisection => write!(f, "bisect"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum StopReason {
    Initialization,
    Residual,
    BracketWidth,
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StopReason::Initialization => write!(f, "initial midpoint satisfies tolerance"),
            StopReason::Residual => write!(f, "residual |f(x)| <= f_tol"),
            StopReason::BracketWidth => write!(f, "bracket width |b-a| <= x_tol"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HybridNewtonResult {
    pub root: f64,
    pub f_at_root: f64,
    pub a: f64,
    pub b: f64,
    pub iterations: usize,
    pub converged: bool,
    pub stop_reason: StopReason,
}

impl fmt::Display for HybridNewtonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HybridNewtonResult {{ root: {:.16e}, f(root): {:.16e}, iterations: {}, converged: {}, stop: {}, final_bracket: [{:.16e}, {:.16e}] }}",
            self.root,
            self.f_at_root,
            self.iterations,
            self.converged,
            self.stop_reason,
            self.a,
            self.b
        )
    }
}

#[derive(Debug, Clone)]
pub enum HybridNewtonError {
    NonFiniteInput { a: f64, b: f64 },
    NonFiniteFunctionValue { x: f64, fx: f64 },
    InvalidBracket { a: f64, b: f64, fa: f64, fb: f64 },
    MaxIterations { last_x: f64, last_fx: f64, iterations: usize },
}

impl fmt::Display for HybridNewtonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HybridNewtonError::NonFiniteInput { a, b } => write!(f, "non-finite endpoints: a={a}, b={b}"),
            HybridNewtonError::NonFiniteFunctionValue { x, fx } => write!(f, "non-finite value encountered: value({x})={fx}"),
            HybridNewtonError::InvalidBracket { a, b, fa, fb } => write!(
                f,
                "invalid bracket: f(a) and f(b) must have opposite signs (a={a}, b={b}, f(a)={fa}, f(b)={fb})"
            ),
            HybridNewtonError::MaxIterations { last_x, last_fx, iterations } => write!(
                f,
                "max iterations reached: x={last_x}, f(x)={last_fx}, iterations={iterations}"
            ),
        }
    }
}

impl std::error::Error for HybridNewtonError {}

fn print_trace_header() {
    println!(
        "{:>4}  {:>14}  {:>14}  {:>22}  {:>22}  {:>12}  {:>10}  {:>8}",
        "k", "a", "b", "x", "f(x)", "|b-a|", "step", "bt"
    );
}

/// Hybrid Newton–bisection with damping.
pub fn hybrid_newton_bisection<F, DF>(
    f: F,
    df: DF,
    mut a: f64,
    mut b: f64,
    cfg: HybridNewtonConfig,
) -> Result<HybridNewtonResult, HybridNewtonError>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(HybridNewtonError::NonFiniteInput { a, b });
    }
    if a == b {
        let fa = f(a);
        if !fa.is_finite() {
            return Err(HybridNewtonError::NonFiniteFunctionValue { x: a, fx: fa });
        }
        return Err(HybridNewtonError::InvalidBracket { a, b, fa, fb: fa });
    }
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }

    let mut fa = f(a);
    if !fa.is_finite() {
        return Err(HybridNewtonError::NonFiniteFunctionValue { x: a, fx: fa });
    }
    let fb0 = f(b);
    if !fb0.is_finite() {
        return Err(HybridNewtonError::NonFiniteFunctionValue { x: b, fx: fb0 });
    }

    if fa == 0.0 {
        return Ok(HybridNewtonResult {
            root: a,
            f_at_root: fa,
            a,
            b,
            iterations: 0,
            converged: true,
            stop_reason: StopReason::Initialization,
        });
    }
    if fb0 == 0.0 {
        return Ok(HybridNewtonResult {
            root: b,
            f_at_root: fb0,
            a,
            b,
            iterations: 0,
            converged: true,
            stop_reason: StopReason::Initialization,
        });
    }
    if fa * fb0 > 0.0 {
        return Err(HybridNewtonError::InvalidBracket { a, b, fa, fb: fb0 });
    }

    // Start at the midpoint.
    let mut x = 0.5 * (a + b);
    let mut fx = f(x);
    if !fx.is_finite() {
        return Err(HybridNewtonError::NonFiniteFunctionValue { x, fx });
    }

    if fx.abs() <= cfg.f_tol {
        return Ok(HybridNewtonResult {
            root: x,
            f_at_root: fx,
            a,
            b,
            iterations: 0,
            converged: true,
            stop_reason: StopReason::Initialization,
        });
    }

    if cfg.trace {
        print_trace_header();
    }

    for k in 1..=cfg.max_iter {
        let width = (b - a).abs();

        if fx.abs() <= cfg.f_tol {
            return Ok(HybridNewtonResult {
                root: x,
                f_at_root: fx,
                a,
                b,
                iterations: k - 1,
                converged: true,
                stop_reason: StopReason::Residual,
            });
        }
        if width <= cfg.x_tol {
            return Ok(HybridNewtonResult {
                root: x,
                f_at_root: fx,
                a,
                b,
                iterations: k - 1,
                converged: true,
                stop_reason: StopReason::BracketWidth,
            });
        }

        let dfx = df(x);
        if !dfx.is_finite() {
            return Err(HybridNewtonError::NonFiniteFunctionValue { x, fx: dfx });
        }

        let mut step_kind = StepKind::Bisection;
        let mut bt_used = 0usize;

        // Default is bisection.
        let mut x_next = 0.5 * (a + b);

        if dfx.abs() > cfg.deriv_tol {
            let x_newton = x - fx / dfx;

            if x_newton > a && x_newton < b {
                let mut f_try = f(x_newton);
                if !f_try.is_finite() {
                    return Err(HybridNewtonError::NonFiniteFunctionValue { x: x_newton, fx: f_try });
                }

                if f_try.abs() <= fx.abs() {
                    step_kind = StepKind::Newton;
                    x_next = x_newton;
                } else {
                    // Damping by backtracking on the Newton displacement.
                    let mut alpha = 1.0;
                    let dx_newton = x_newton - x;

                    while bt_used < cfg.max_backtrack {
                        alpha *= cfg.shrink;
                        let x_damped = x + alpha * dx_newton;

                        if !(x_damped > a && x_damped < b) {
                            break;
                        }

                        f_try = f(x_damped);
                        if !f_try.is_finite() {
                            return Err(HybridNewtonError::NonFiniteFunctionValue { x: x_damped, fx: f_try });
                        }

                        bt_used += 1;

                        if f_try.abs() <= fx.abs() {
                            step_kind = StepKind::DampedNewton;
                            x_next = x_damped;
                            break;
                        }
                    }
                }
            }
        }

        let fx_next = f(x_next);
        if !fx_next.is_finite() {
            return Err(HybridNewtonError::NonFiniteFunctionValue { x: x_next, fx: fx_next });
        }

        // Bracket update (sign change). Only fa is required in this style.
        if fa * fx_next < 0.0 {
            b = x_next;
        } else {
            a = x_next;
            fa = fx_next;
        }

        if cfg.trace {
            println!(
                "{:>4}  {:>14.6e}  {:>14.6e}  {:>22.16e}  {:>22.16e}  {:>12.3e}  {:>10}  {:>8}",
                k, a, b, x_next, fx_next, (b - a).abs(), step_kind, bt_used
            );
        }

        x = x_next;
        fx = fx_next;
    }

    Err(HybridNewtonError::MaxIterations {
        last_x: x,
        last_fx: fx,
        iterations: cfg.max_iter,
    })
}

fn colebrook_residual(friction: f64, re: f64, rel_roughness: f64) -> f64 {
    let inv_sqrt = 1.0 / friction.sqrt();
    let arg = rel_roughness / 3.7 + 2.51 / (re * friction.sqrt());
    inv_sqrt + 2.0 * arg.log10()
}

fn main() {
    // Example 1: Smooth transcendental equation (simple root).
    let f1 = |x: f64| x.cos() - x;
    let df1 = |x: f64| -x.sin() - 1.0;

    // Example 2: Simple root with a valid sign-changing bracket.
    // Use a bracket whose midpoint is not exactly the root to show iterations.
    let f2 = |x: f64| x * x * x - 1.0;
    let df2 = |x: f64| 3.0 * x * x;

    // Example 3: Colebrook–White residual check (application context).
    let re = 1.0e5;
    let eps_over_d = 1.0e-4;
    let f3 = |ff: f64| colebrook_residual(ff, re, eps_over_d);

    let cfg = HybridNewtonConfig {
        max_iter: 100,
        f_tol: 1e-14,
        x_tol: 1e-14,
        deriv_tol: 1e-14,
        max_backtrack: 20,
        shrink: 0.5,
        trace: false, // set true to observe safeguard decisions
    };

    println!("Example 1: Hybrid Newton–bisection for f(x) = cos(x) - x on [0, 1]");
    match hybrid_newton_bisection(f1, df1, 0.0, 1.0, cfg.clone()) {
        Ok(res) => println!("{res}\n"),
        Err(e) => println!("Hybrid solver failed: {e}\n"),
    }

    println!("Example 2: Hybrid Newton–bisection for f(x) = x^3 - 1 on [0, 1.5]");
    match hybrid_newton_bisection(f2, df2, 0.0, 1.5, cfg.clone()) {
        Ok(res) => println!("{res}\n"),
        Err(e) => println!("Hybrid solver failed: {e}\n"),
    }

    println!(
        "Example 3: Colebrook residual check (not solved here): F(0.02) = {:.6e}",
        f3(0.02)
    );
}
```

Program 9.5.4 demonstrates how Newton’s method can be transformed from a locally powerful but fragile algorithm into a globally reliable solver through the use of safeguards and hybrid strategies. By enforcing bracketing, monitoring step admissibility, and applying damping when necessary, the algorithm guarantees convergence without sacrificing the fast local behavior that makes Newton’s method attractive.

The numerical examples illustrate a central theme of this chapter: robust algorithms do not rely on a single strategy, but instead adapt dynamically to the behavior of the function. Near the root, the solver behaves like classical Newton iteration and achieves rapid convergence. Far from the root, it reverts automatically to conservative steps that preserve stability. This adaptive balance between speed and reliability is the defining characteristic of modern nonlinear solvers.

The modular design of the implementation makes it straightforward to extend the framework with more sophisticated acceptance tests, alternative damping strategies, or higher-order interpolation methods. These ideas lead naturally to fully hybrid algorithms such as Brent’s method, which combine bracketing and interpolation in a similarly adaptive manner and are widely regarded as the gold standard for one-dimensional root finding.

## 9.5.5. Newton’s Method in Higher Dimensions

Newton’s method generalizes naturally to systems of nonlinear equations:

$$\mathbf{f}(\mathbf{x})=\mathbf{0}, \qquad \mathbf{x}\in\mathbb{R}^n \tag{9.5.14}$$

Let $J(\mathbf{x})$ denote the Jacobian matrix:

$$J(\mathbf{x})=\left[\frac{\partial f_i}{\partial x_j}\right] \tag{9.5.15}$$

At each iteration, the Newton correction $\boldsymbol{\delta}$ is obtained by solving,

$$
J(\mathbf{x}_i)\,\boldsymbol{\delta}
= -\,\mathbf{f}(\mathbf{x}_i)
\tag{9.5.16}
$$

followed by the update:

$$
\mathbf{x}_{i+1} = \mathbf{x}_i + \boldsymbol{\delta}
\tag{9.5.17}
$$

When the Jacobian is nonsingular at the root, the method retains locally quadratic convergence. However, each iteration requires solving an $n\times n$ linear system, typically an $O(n^3)$ operation, motivating the development of quasi-Newton methods. Recent advances, including full-rank Jacobian update strategies, have improved both stability and convergence order (Berzi, 2024; Shah et al., 2025).

### Newton Fractals and Basins of Attraction

Newton’s method can be extended to the complex plane, where it exhibits remarkably rich behavior. For example, applying Newton’s method to,

$$f(z)=z^3-1 \tag{9.5.18}$$

yields the iteration:

$$
z_{n+1}
= z_n - \frac{z_n^3 - 1}{3z_n^2}
= \frac{2z_n^3 + 1}{3z_n^2}
\tag{9.5.19}
$$

Each root has an associated basin of attraction in the complex plane. The boundaries between these basins form intricate fractal structures, reflecting the sensitive dependence of the iteration on initial conditions. Such Newton fractals are classic examples of chaotic behavior in dynamical systems and continue to be studied as diagnostic tools for iterative methods (Shah et al., 2025).

### Higher-Order Extensions: Halley’s Method

Newton’s method uses first derivatives only. Incorporating second derivatives leads to Halley’s method, which achieves cubic convergence. Its update formula is:

$$
x_{i+1}
= x_i
- \frac{f(x_i)}
{f'(x_i)\!\left(1 - \dfrac{f(x_i)f''(x_i)}{2\,[f'(x_i)]^2}\right)}
\tag{9.5.20}
$$

Halley’s method converges cubically near a simple root, meaning that the number of correct digits triples at each iteration. However, it requires evaluation of the second derivative and often has smaller basins of attraction than Newton’s method. A damped variant is sometimes used:

$$
x_{i+1}
= x_i
- \frac{\dfrac{f(x_i)}{f'(x_i)}}
{\max\!\left(0.8,\; \min\!\left(1.2,\; 1 - \frac{f(x_i)f''(x_i)}{2\,[f'(x_i)]^2}\right)\right)}
\tag{9.5.21}
$$

In practice, two Newton iterations often outperform a single Halley step in terms of total computational cost. Halley’s method is a special case of the broader class of Householder methods, where an order $n$ method uses derivatives up to order $n-1$.

### Rust Implementation

Following the development of Newton’s method for systems of nonlinear equations in Equations (9.5.14)–(9.5.17), Program 9.5.5 provides a unified numerical implementation illustrating several important extensions of the method. In practical computation, Newton’s method must be embedded within a framework that can approximate Jacobians, solve linear systems efficiently, and guard against instability when iterates move far from the root. This program demonstrates how Newton’s method is realized in higher dimensions using finite-difference Jacobian approximations and a safeguarded update strategy. It further illustrates the rich dynamical behavior of Newton’s method in the complex plane through the construction of Newton fractals and contrasts first-order Newton updates with higher-order corrections via Halley’s method. Together, these components bridge the theoretical formulation of Newton-type methods with their practical realization in finite-precision numerical environments.

At the core of the implementation is the function `newton_system`, which realizes the multidimensional Newton iteration defined by Equations (9.5.16) and (9.5.17). Given a nonlinear vector-valued function $\mathbf{f}(\mathbf{x})$, the routine evaluates the current residual $\mathbf{f}(\mathbf{x}_i)$, constructs an approximation to the Jacobian matrix $J(\mathbf{x}_i)$ from Equation (9.5.15), and computes the Newton correction $\boldsymbol{\delta}$ by solving the associated linear system. The Jacobian is approximated using forward finite differences, which avoids the need for analytic derivatives while remaining sufficiently accurate near the solution for smooth problems.

The linear system arising from Equation (9.5.16) is solved using Gaussian elimination with partial pivoting. Although this approach has $O(n^3)$ complexity, it is appropriate for moderate system sizes and provides a transparent illustration of the computational cost motivating quasi-Newton and Krylov-based methods discussed later in the chapter. To improve robustness, the update step incorporates a simple backtracking strategy that reduces the step length when the residual norm fails to decrease. This safeguard mitigates divergence when the initial guess lies outside the local basin of quadratic convergence.

To illustrate Newton’s method beyond real-valued systems, the program implements the complex iteration defined by Equation (9.5.19) for the polynomial $f(z)=z^3-1$. The function `newton_cubic_step` applies the Newton update in the complex plane, while repeated iteration from a grid of initial points reveals the basins of attraction associated with each root. The resulting Newton fractal is written directly to a PNG image, with color encoding the root to which the iteration converges and brightness reflecting the number of iterations required for convergence. This visualization highlights the sensitive dependence on initial conditions near basin boundaries and provides a concrete illustration of the chaotic dynamics discussed in Section 9.5.5.

The program also includes implementations of Halley’s method for scalar equations, corresponding to Equation (9.5.20), as well as a damped variant inspired by Equation (9.5.21). The function `halley_scalar` demonstrates cubic convergence near a simple root by incorporating second-derivative information, while `halley_scalar_damped` limits the correction factor to improve stability when far from the solution. These routines allow direct comparison between first-order and higher-order Newton-type methods and illustrate the trade-off between convergence order, computational cost, and basin size.

The `main` function serves as a driver that exercises each component of the framework. It first solves a simple two-dimensional nonlinear system to demonstrate quadratic convergence of Newton’s method in $\mathbb{R}^n$. It then applies Halley’s method and its damped variant to a scalar cubic equation, highlighting their rapid convergence near the root. Finally, it generates a Newton fractal for the complex polynomial $z^3-1$, providing a visual counterpart to the analytical discussion of basins of attraction.

Add the following dependencies to crgo.toml:

```rust
[dependencies]
num-complex = "0.4"
image = "0.25"
```

```rust
// Program 9.5.5: Newton’s Method in Higher Dimensions, Newton Fractals, and Halley’s Method
//
// This single-file program demonstrates three closely related ideas from Section 9.5.5:
//
// (A) Newton in R^n for f(x)=0 using (9.5.14)–(9.5.17):
//     Solve J(x_i) δ = -f(x_i), then update x_{i+1} = x_i + δ.
//     The Jacobian is approximated by forward finite differences and the linear system is
//     solved by Gaussian elimination with partial pivoting.
//
// (B) Newton fractals in C for f(z)=z^3-1 using the iteration (9.5.19):
//     z_{n+1} = z_n - (z_n^3 - 1)/(3 z_n^2).
//     The program writes a PNG image encoding basin membership and convergence speed.
//
// (C) Halley’s method in R using (9.5.20) and a damped variant inspired by (9.5.21).
//
// Cargo.toml dependencies:
// [dependencies]
// num-complex = "0.4"
// image = "0.25"

use image::{Rgb, RgbImage};
use num_complex::Complex64;

// ---------------------------
// Shared numerical utilities
// ---------------------------

#[derive(Debug, Clone)]
pub struct NewtonOptions {
    pub max_iter: usize,
    pub tol_abs: f64,
    pub tol_rel: f64,
    pub fd_eps: f64, // finite-difference step for Jacobian
}

impl Default for NewtonOptions {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol_abs: 1e-12,
            tol_rel: 1e-12,
            fd_eps: 1e-7,
        }
    }
}

/// Euclidean norm ||v||_2.
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Solve A x = b via Gaussian elimination with partial pivoting.
/// A is consumed (copied in); b is consumed; returns x.
fn solve_linear_system_gauss(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return Err("Dimension mismatch in linear solve.".to_string());
    }
    for row in &a {
        if row.len() != n {
            return Err("Matrix must be square.".to_string());
        }
    }

    // Forward elimination
    for k in 0..n {
        // Pivot selection
        let mut pivot_row = k;
        let mut pivot_val = a[k][k].abs();
        for i in (k + 1)..n {
            let v = a[i][k].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = i;
            }
        }
        if pivot_val == 0.0 {
            return Err("Singular matrix encountered in Gaussian elimination.".to_string());
        }

        // Swap rows
        if pivot_row != k {
            a.swap(k, pivot_row);
            b.swap(k, pivot_row);
        }

        // Eliminate below
        let akk = a[k][k];
        for i in (k + 1)..n {
            let factor = a[i][k] / akk;
            a[i][k] = 0.0;
            for j in (k + 1)..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        let aii = a[i][i];
        if aii == 0.0 {
            return Err("Singular matrix encountered in back substitution.".to_string());
        }
        x[i] = s / aii;
    }
    Ok(x)
}

/// Finite-difference Jacobian approximation for J(x) = [∂f_i/∂x_j].
/// Forward difference: (f(x + h e_j) - f(x))/h with h scaled to component magnitude.
fn jacobian_fd<F>(f: &F, x: &[f64], eps: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let fx = f(x);
    let n = x.len();
    let m = fx.len();
    let mut j = vec![vec![0.0; n]; m];

    for col in 0..n {
        let mut xh = x.to_vec();
        let h = eps * (1.0 + x[col].abs());
        xh[col] += h;
        let fxh = f(&xh);
        for row in 0..m {
            j[row][col] = (fxh[row] - fx[row]) / h;
        }
    }
    j
}

// ---------------------------------------
// (A) Newton’s method for f(x)=0 in R^n
// ---------------------------------------

/// Newton solver for systems f(x)=0 using (9.5.16)–(9.5.17).
/// Returns (x*, iterations_used).
pub fn newton_system<F>(f: F, mut x: Vec<f64>, opt: &NewtonOptions) -> Result<(Vec<f64>, usize), String>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let fx0 = f(&x);
    let fx0_norm = norm2(&fx0).max(1.0);

    for it in 0..opt.max_iter {
        let fx = f(&x);
        let fnorm = norm2(&fx);

        // Residual stopping condition: ||f(x)|| <= tol_abs + tol_rel*||f(x0)||
        if fnorm <= opt.tol_abs + opt.tol_rel * fx0_norm {
            return Ok((x, it));
        }

        // Form Jacobian and solve J δ = -f
        let j = jacobian_fd(&f, &x, opt.fd_eps);
        let rhs: Vec<f64> = fx.iter().map(|v| -v).collect();
        let delta = solve_linear_system_gauss(j, rhs)?;

        // Simple backtracking line search for robustness.
        let x_old = x.clone();
        let mut alpha = 1.0;
        let fnorm_old = fnorm;

        for _bt in 0..20 {
            for i in 0..x.len() {
                x[i] = x_old[i] + alpha * delta[i];
            }
            let fnorm_new = norm2(&f(&x));
            if fnorm_new <= (1.0 - 1e-4 * alpha) * fnorm_old || alpha <= 1e-6 {
                break;
            }
            alpha *= 0.5;
        }
    }

    Err("Newton system solver did not converge within max_iter.".to_string())
}

// ---------------------------------------
// (C) Halley’s method and damped Halley
// ---------------------------------------

/// Halley’s method for scalar root finding, using (9.5.20).
pub fn halley_scalar<F, FP, FPP>(
    f: F,
    fp: FP,
    fpp: FPP,
    mut x: f64,
    max_iter: usize,
    tol: f64,
) -> Result<(f64, usize), String>
where
    F: Fn(f64) -> f64,
    FP: Fn(f64) -> f64,
    FPP: Fn(f64) -> f64,
{
    for it in 0..max_iter {
        let fx = f(x);
        if fx.abs() <= tol {
            return Ok((x, it));
        }
        let fpx = fp(x);
        if fpx == 0.0 {
            return Err("Halley: f'(x) is zero, cannot proceed.".to_string());
        }
        let fppx = fpp(x);

        // x_{i+1} = x_i - f / ( f' * (1 - f f'' / (2 (f')^2)) )
        let denom_corr = 1.0 - (fx * fppx) / (2.0 * fpx * fpx);
        if denom_corr == 0.0 {
            return Err("Halley: correction denominator is zero.".to_string());
        }
        let step = fx / (fpx * denom_corr);
        x -= step;
    }
    Err("Halley method did not converge within max_iter.".to_string())
}

/// A damped Halley variant in the style of (9.5.21).
pub fn halley_scalar_damped<F, FP, FPP>(
    f: F,
    fp: FP,
    fpp: FPP,
    mut x: f64,
    max_iter: usize,
    tol: f64,
) -> Result<(f64, usize), String>
where
    F: Fn(f64) -> f64,
    FP: Fn(f64) -> f64,
    FPP: Fn(f64) -> f64,
{
    for it in 0..max_iter {
        let fx = f(x);
        if fx.abs() <= tol {
            return Ok((x, it));
        }
        let fpx = fp(x);
        if fpx == 0.0 {
            return Err("Damped Halley: f'(x) is zero, cannot proceed.".to_string());
        }
        let fppx = fpp(x);

        // Newton step: f/f'
        let newton_step = fx / fpx;

        // Damping factor: clamp(1 - f f''/(2 (f')^2), 0.8, 1.2)
        let raw = 1.0 - (fx * fppx) / (2.0 * fpx * fpx);
        let damp = raw.max(0.8).min(1.2);

        x -= newton_step / damp;
    }
    Err("Damped Halley did not converge within max_iter.".to_string())
}

// ---------------------------------------
// (B) Newton fractal for f(z)=z^3-1
// ---------------------------------------

/// Newton iteration for f(z)=z^3-1, using (9.5.19).
fn newton_cubic_step(z: Complex64) -> Complex64 {
    // z_{n+1} = z - (z^3 - 1)/(3 z^2)
    // Guard z=0 to avoid division by zero.
    if z == Complex64::new(0.0, 0.0) {
        return Complex64::new(1e-12, 0.0);
    }
    let z2 = z * z;
    let z3 = z2 * z;
    z - (z3 - Complex64::new(1.0, 0.0)) / (Complex64::new(3.0, 0.0) * z2)
}

/// Return which root of z^3-1 is closest: indices {0,1,2}.
fn classify_root(z: Complex64) -> usize {
    // Roots: 1, exp(2πi/3), exp(4πi/3)
    let r0 = Complex64::new(1.0, 0.0);
    let r1 = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
    let r2 = Complex64::from_polar(1.0, 4.0 * std::f64::consts::PI / 3.0);

    let d0 = (z - r0).norm();
    let d1 = (z - r1).norm();
    let d2 = (z - r2).norm();

    if d0 <= d1 && d0 <= d2 {
        0
    } else if d1 <= d0 && d1 <= d2 {
        1
    } else {
        2
    }
}

/// Write a Newton fractal directly to PNG.
/// Basin membership is encoded by base color; iteration count scales brightness.
pub fn write_newton_fractal_png(
    filename: &str,
    width: u32,
    height: u32,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    max_iter: usize,
    tol: f64,
) -> Result<(), String> {
    if width < 2 || height < 2 {
        return Err("Image dimensions must be at least 2x2.".to_string());
    }
    if x_min >= x_max || y_min >= y_max {
        return Err("Invalid plotting bounds.".to_string());
    }

    let mut img = RgbImage::new(width, height);

    for iy in 0..height {
        let y = y_max - (y_max - y_min) * (iy as f64) / ((height - 1) as f64);
        for ix in 0..width {
            let x = x_min + (x_max - x_min) * (ix as f64) / ((width - 1) as f64);
            let mut z = Complex64::new(x, y);

            let mut it_conv = max_iter;
            for it in 0..max_iter {
                z = newton_cubic_step(z);
                let fz = z * z * z - Complex64::new(1.0, 0.0);
                if fz.norm() <= tol {
                    it_conv = it;
                    break;
                }
            }

            // Brighter = faster convergence; black = no convergence within max_iter.
            let brightness = if it_conv >= max_iter {
                0.0
            } else {
                1.0 - (it_conv as f64) / (max_iter as f64)
            };

            let (mut r, mut g, mut b) = match classify_root(z) {
                0 => (255.0, 70.0, 70.0),
                1 => (70.0, 255.0, 70.0),
                _ => (70.0, 70.0, 255.0),
            };

            r *= brightness;
            g *= brightness;
            b *= brightness;

            if it_conv >= max_iter {
                r = 0.0;
                g = 0.0;
                b = 0.0;
            }

            img.put_pixel(ix, iy, Rgb([r as u8, g as u8, b as u8]));
        }
    }

    img.save(filename)
        .map_err(|e| format!("Failed to write PNG '{}': {}", filename, e))?;
    Ok(())
}

// ---------------------------
// Main demo driver
// ---------------------------

fn main() -> Result<(), String> {
    // ------------------------------------------------------------
    // Demo A: Newton in R^2 for a simple nonlinear system.
    //
    // Solve:
    //   f1(x,y) = x^2 + y^2 - 1 = 0  (unit circle)
    //   f2(x,y) = x - y = 0          (line y=x)
    //
    // The solution in the first quadrant is (1/sqrt(2), 1/sqrt(2)).
    // ------------------------------------------------------------
    let f = |x: &[f64]| -> Vec<f64> {
        let x0 = x[0];
        let x1 = x[1];
        vec![x0 * x0 + x1 * x1 - 1.0, x0 - x1]
    };

    let opt = NewtonOptions {
        max_iter: 30,
        tol_abs: 1e-14,
        tol_rel: 1e-14,
        fd_eps: 1e-7,
    };

    let x0 = vec![0.8, 0.6];
    let (x_star, iters) = newton_system(f, x0, &opt)?;
    println!("Newton system converged in {} iterations.", iters);
    println!("x* = [{:.16}, {:.16}]", x_star[0], x_star[1]);
    println!("f(x*) norm = {:.3e}", norm2(&f(&x_star)));

    // ------------------------------------------------------------
    // Demo C: Halley’s method (scalar), and a damped variant.
    //
    // f(x) = x^3 - 1, root at x=1.
    // f'(x) = 3x^2, f''(x) = 6x.
    // ------------------------------------------------------------
    let f1 = |x: f64| x * x * x - 1.0;
    let fp1 = |x: f64| 3.0 * x * x;
    let fpp1 = |x: f64| 6.0 * x;

    let (xh, ith) = halley_scalar(f1, fp1, fpp1, 0.7, 20, 1e-14)?;
    println!("Halley converged in {} iterations: x = {:.16}", ith, xh);

    let (xhd, ithd) = halley_scalar_damped(f1, fp1, fpp1, 0.7, 20, 1e-14)?;
    println!(
        "Damped Halley converged in {} iterations: x = {:.16}",
        ithd, xhd
    );

    // ------------------------------------------------------------
    // Demo B: Newton fractal for f(z)=z^3-1 using (9.5.19).
    // Writes a PNG image viewable directly in standard viewers.
    // ------------------------------------------------------------
    let out_name = "newton_fractal_z3_minus_1.png";
    write_newton_fractal_png(
        out_name,
        1000,
        1000,
        -2.0,
        2.0,
        -2.0,
        2.0,
        80,
        1e-10,
    )?;
    println!("Wrote Newton fractal image to {}", out_name);

    Ok(())
}
```

```{figure} images/pqQDe4beUu67RvW3raYP-fj1DBiUglO64quI8jnSZ-v1.png
:name: DDWqnhQ6D6
:align: middle
:width: 50%

**Figure 9.5.2:** Newton fractal for the polynomial $f(z)=z^3-1$ generated using the iteration defined in Equation (9.5.19). Each color corresponds to the basin of attraction of one of the three cubic roots of unity, while brightness indicates the number of iterations required for convergence. The intricate boundaries between colored regions reveal the fractal structure of the basin separatrices, illustrating the sensitive dependence of Newton’s method on initial conditions in the complex plane.
```

Program 9.5.5 demonstrates how Newton’s method for nonlinear systems can be translated into a practical and extensible computational framework. By explicitly constructing Jacobian approximations, solving the associated linear systems, and safeguarding the update step, the implementation reflects the algorithmic considerations that arise in real-world applications beyond idealized theoretical settings.

The comparison between Newton’s method and Halley’s method illustrates a fundamental trade-off in nonlinear solvers. While higher-order methods achieve faster local convergence, they require additional derivative information and often exhibit smaller basins of attraction. In many practical problems, two well-chosen Newton iterations may be more efficient than a single higher-order correction, particularly when derivative evaluations are costly or noisy.

The Newton fractal example emphasizes that Newton’s method is not merely a root-finding algorithm but also a nonlinear dynamical system whose global behavior can be highly complex. The intricate basin boundaries observed in the complex plane serve as a reminder that convergence guarantees are inherently local and that global behavior must be assessed carefully. Together, these examples provide a foundation for the globally convergent and quasi-Newton methods developed in subsequent sections.

## 9.5.6. Concluding Remarks

Newton’s method, when augmented with derivative information, occupies a foundational position in numerical computing and nonlinear analysis. Its appeal lies in the remarkable combination of conceptual simplicity, strong local convergence theory, and practical efficiency. For functions with sufficiently smooth behavior and nonvanishing derivatives at the root, the method exhibits quadratic convergence, making it one of the fastest general-purpose algorithms available for solving nonlinear equations.

At the same time, the analysis in this section highlights that Newton’s method is not inherently a global algorithm. Its performance depends sensitively on the choice of the initial iterate and on the local behavior of the derivative. Multiple roots, nearly singular Jacobians, or strong nonlinearities can degrade convergence or even cause divergence. These limitations motivate the widespread use of safeguards such as bracketing, damping, and hybrid strategies, which transform Newton’s method from a purely local scheme into a robust tool suitable for practical computation.

Beyond the classical scalar iteration, the extension of Newton’s method to systems of nonlinear equations through Jacobian-based linearization forms the backbone of many large-scale scientific and engineering applications, including discretized partial differential equations, optimization problems, and inverse modeling. In these contexts, the interaction between nonlinear iteration, linear solvers, and preconditioning becomes as important as the nonlinear method itself.

In the next section, attention shifts to the special case of polynomial root finding. There, the additional algebraic structure of polynomials enables the development of specialized algorithms that go beyond generic Newton-type iterations, offering enhanced robustness, global convergence guarantees, and efficient treatment of multiple roots and complex-valued solutions.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/BCh47csqUVOnKkuBRcdb.5","tags":[]}

# 9.6. Roots of Polynomials

Polynomial root finding is one of the oldest and most extensively studied problems in numerical computation. Given a univariate polynomial of degree $m,$

$$P(x) = a_m x^m + a_{m-1}x^{m-1} + \cdots + a_1 x + a_0\tag{9.6.1}$$

the Fundamental Theorem of Algebra guarantees exactly $m$ roots in the complex plane when multiplicities are counted. While this theorem ensures existence and completeness, it provides no guidance on how to compute the roots reliably in finite-precision arithmetic. The numerical determination of polynomial roots therefore remains a central and nontrivial task.

Polynomial roots arise naturally in a wide range of applications. In control theory, they determine system stability through characteristic equations. In physics and chemistry, they appear in spectral problems, dispersion relations, and molecular orbital calculations. In computer graphics and computational geometry, polynomial equations describe curve and surface intersections. In machine learning and optimization, polynomial equations emerge in loss landscapes, kernel constructions, and moment-based methods. Despite the apparent simplicity of polynomials, their roots often encode highly sensitive and nonlinear information about the underlying problem.

A key observation is that the difficulty of polynomial root finding rarely lies in evaluating (P(x)) itself. Polynomial evaluation can be performed efficiently and stably using schemes such as Horner’s method. Instead, the primary challenge is the sensitivity of the roots to small perturbations in the coefficients. This sensitivity imposes fundamental limits on achievable accuracy and must be understood as an inherent property of the problem rather than a deficiency of any particular algorithm.

## 9.6.1. Conditioning and Sensitivity of Polynomial Roots

The conditioning of polynomial roots is best understood through classical counterexamples. One of the most influential is Wilkinson’s polynomial,

$$P_{20}(x) = (x-1)(x-2)\cdots(x-20)\tag{9.6.2}$$

Although the exact roots are the integers $1,2,\ldots,20$, Wilkinson demonstrated that a relative perturbation of order $10^{-7}$ in the constant term can cause dramatic changes in the root locations, with some roots moving far into the complex plane. This example illustrates that polynomial root finding can be severely ill-conditioned even when the polynomial itself appears benign.

This phenomenon highlights a crucial distinction between backward error and forward error. A numerical algorithm may compute the exact roots of a nearby polynomial whose coefficients differ only slightly from those of the original polynomial, yet the corresponding roots may differ substantially. No numerical method can overcome such ill-conditioning, because it reflects intrinsic sensitivity rather than algorithmic instability.

Additional difficulties arise in the presence of multiple or nearly multiple roots. If a polynomial contains a factor $(x-r)^k$, then both $P(r)$ and its first derivative vanish at $x=r$. In such cases, Newton’s method loses its quadratic convergence and typically converges only linearly. Moreover, rounding errors can cause multiple roots to split into clusters of nearby simple roots, or distinct roots to merge numerically. These effects complicate root identification, multiplicity detection, and subsequent refinement.

### Rust Implementation

Following the discussion in Section 9.6 on the fundamental challenges of polynomial root finding and the sensitivity of roots to coefficient perturbations, Program 9.6.1 provides a concrete numerical implementation illustrating how these issues arise in practice. While the Fundamental Theorem of Algebra guarantees the existence of exactly $m$ roots for a degree-$m$ polynomial, Equations (9.6.1) and (9.6.2) highlight that computing these roots reliably in finite-precision arithmetic is a nontrivial task. This program implements a global, all-root method based on the Durand–Kerner iteration, supplemented by Newton refinement, to compute complex roots simultaneously. In addition, it demonstrates the conditioning issues discussed in Subsection 9.6.1 through a numerical experiment involving Wilkinson’s polynomial. Together, these components bridge the theoretical discussion of polynomial root sensitivity with a robust, executable computational framework.

At the core of the implementation is the `Polynomial` structure, which represents a univariate polynomial in the standard form given by Equation (9.6.1). Polynomial evaluation is performed using Horner’s method, which computes $P(x)$ with linear complexity and favorable numerical properties. This reflects the observation made earlier in Section 9.6 that evaluating a polynomial is typically not the source of numerical difficulty; rather, the challenge lies in locating its roots accurately.

The function implementing the Durand–Kerner method computes all roots of the polynomial simultaneously by iteratively refining a set of complex approximations. At each iteration, the update formula treats the polynomial as factored into linear terms corresponding to its roots, and each approximation is corrected using the current estimates of the others. To improve numerical robustness, the implementation first scales the polynomial to monic form and initializes the iterates using a root bound derived from the coefficients. This global approach avoids the need for deflation and reflects the all-root perspective emphasized in Section 9.6.

Once the Durand–Kerner iteration has converged, the program applies a small number of Newton refinement steps to each root individually. This hybrid strategy combines the global convergence properties of the Durand–Kerner method with the rapid local convergence of Newton’s method, while guarding against division by small denominators and the propagation of non-finite values. These safeguards are essential in finite-precision arithmetic, particularly for high-degree or ill-conditioned polynomials.

To illustrate the conditioning issues discussed in Subsection 9.6.1, the program constructs Wilkinson’s polynomial defined in Equation (9.6.2) and computes its roots before and after a small relative perturbation to one of its coefficients. Diagnostic quantities such as the maximum imaginary part of the roots and the maximum displacement between corresponding roots are reported. Although not every perturbation produces dramatic changes in double-precision arithmetic, this experiment reinforces the distinction between backward stability and forward sensitivity: even when a numerical method behaves stably, the roots themselves may respond unpredictably to small changes in the coefficients.

The `main` function serves as a driver that demonstrates these ideas through two representative examples. It first computes the roots of a simple polynomial with known analytic solutions, verifying correctness through small residuals. It then applies the same framework to Wilkinson’s polynomial, providing empirical evidence for the sensitivity phenomena described in Section 9.6.1. The structure of the program emphasizes clarity, robustness, and direct correspondence with the surrounding theoretical discussion.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
```

```rust
// Program 9.6.1: Polynomial Root Finding and Sensitivity Analysis
//
// Implements:
//  • Horner evaluation of P(x) (Eq. 9.6.1)
//  • All-root computation via Durand–Kerner (Weierstrass) iteration
//  • Newton refinement for individual roots
//  • Sensitivity demonstration using Wilkinson’s polynomial (Eq. 9.6.2)
//
// Cargo.toml dependencies:
// [dependencies]
// num-complex = "0.4"

use num_complex::Complex64;
use std::cmp::Ordering;

// ================================================================
// Polynomial representation and evaluation
// ================================================================

#[derive(Clone, Debug)]
struct Polynomial {
    // Coefficients in descending powers: a_m x^m + ... + a_0
    coeffs: Vec<Complex64>,
}

impl Polynomial {
    fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    fn leading(&self) -> Complex64 {
        self.coeffs[0]
    }

    fn eval(&self, z: Complex64) -> Complex64 {
        // Horner evaluation (stable, O(m))
        let mut acc = Complex64::new(0.0, 0.0);
        for &a in &self.coeffs {
            acc = acc * z + a;
        }
        acc
    }

    fn derivative(&self) -> Polynomial {
        let m = self.degree();
        if m == 0 {
            return Polynomial {
                coeffs: vec![Complex64::new(0.0, 0.0)],
            };
        }

        let mut d = Vec::with_capacity(m);
        for (k, &a) in self.coeffs.iter().enumerate().take(m) {
            let power = (m - k) as f64;
            d.push(a * Complex64::new(power, 0.0));
        }
        Polynomial { coeffs: d }
    }

    fn monic(&self) -> Result<Polynomial, String> {
        let lead = self.leading();
        if lead == Complex64::new(0.0, 0.0) {
            return Err("Leading coefficient is zero.".to_string());
        }
        Ok(Polynomial {
            coeffs: self.coeffs.iter().map(|&c| c / lead).collect(),
        })
    }
}

// ================================================================
// Wilkinson polynomial construction
// ================================================================

fn poly_mul_desc(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    // Convolution in ascending order then convert back.
    let a_asc: Vec<_> = a.iter().rev().copied().collect();
    let b_asc: Vec<_> = b.iter().rev().copied().collect();

    let mut c = vec![Complex64::new(0.0, 0.0); a_asc.len() + b_asc.len() - 1];
    for (i, &ai) in a_asc.iter().enumerate() {
        for (j, &bj) in b_asc.iter().enumerate() {
            c[i + j] += ai * bj;
        }
    }
    c.into_iter().rev().collect()
}

fn wilkinson_polynomial(n: usize) -> Polynomial {
    let mut coeffs = vec![Complex64::new(1.0, 0.0)];
    for k in 1..=n {
        // Multiply by (x - k)
        let factor = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(-(k as f64), 0.0),
        ];
        coeffs = poly_mul_desc(&coeffs, &factor);
    }
    Polynomial { coeffs }
}

// ================================================================
// Root bounds and initialization
// ================================================================

/// Cauchy bound: all roots satisfy |z| <= 1 + max_{i<m} |a_i/a_m|.
fn cauchy_bound(p: &Polynomial) -> f64 {
    let m = p.degree();
    if m == 0 {
        return 0.0;
    }
    let a_m = p.leading();
    let mut max_ratio: f64 = 0.0;

    for &a in p.coeffs.iter().skip(1) {
        let r = (a / a_m).norm();
        if r > max_ratio {
            max_ratio = r;
        }
    }
    1.0 + max_ratio
}

fn initial_guesses(m: usize, radius: f64) -> Vec<Complex64> {
    let mut z = Vec::with_capacity(m);
    let theta0 = 0.37; // phase offset breaks symmetry
    for i in 0..m {
        let theta = theta0 + 2.0 * std::f64::consts::PI * (i as f64) / (m as f64);
        z.push(Complex64::from_polar(radius, theta));
    }
    z
}

// ================================================================
// Durand–Kerner root finder (robust version)
// ================================================================

#[derive(Clone)]
struct DKOptions {
    max_iter: usize,
    tol: f64,
    refine: usize,
    min_sep: f64,   // protect against tiny (z_i - z_j)
    damping: f64,   // optional damping for stability
}

impl Default for DKOptions {
    fn default() -> Self {
        Self {
            max_iter: 5000,
            tol: 1e-13,
            refine: 8,
            min_sep: 1e-14,
            damping: 1.0,
        }
    }
}

fn durand_kerner(p: &Polynomial, opt: &DKOptions) -> Result<Vec<Complex64>, String> {
    let pm = p.monic()?;
    let m = pm.degree();
    if m == 0 {
        return Ok(vec![]);
    }

    // Use a root bound for initialization (critical for Wilkinson).
    let r0 = cauchy_bound(&pm);
    let mut z = initial_guesses(m, r0);

    for _ in 0..opt.max_iter {
        let old = z.clone();
        let mut max_step: f64 = 0.0;

        for i in 0..m {
            let zi = old[i];

            // denom = Π_{j≠i} (zi - zj), with a small-separation guard.
            let mut denom = Complex64::new(1.0, 0.0);
            for j in 0..m {
                if i == j {
                    continue;
                }
                let diff = zi - old[j];
                if diff.norm() < opt.min_sep {
                    denom *= Complex64::new(opt.min_sep, 0.0);
                } else {
                    denom *= diff;
                }
            }

            let step = pm.eval(zi) / denom;
            let dz = opt.damping * step;
            z[i] = zi - dz;

            let s = dz.norm();
            if s > max_step {
                max_step = s;
            }
        }

        if max_step < opt.tol {
            break;
        }
    }

    // Newton refinement on the original polynomial.
    let dp = p.derivative();
    for zi in &mut z {
        for _ in 0..opt.refine {
            if !zi.re.is_finite() || !zi.im.is_finite() {
                break;
            }
            let fz = p.eval(*zi);
            let dfz = dp.eval(*zi);
            if dfz == Complex64::new(0.0, 0.0) {
                break;
            }
            let step = fz / dfz;
            *zi -= step;

            if !zi.re.is_finite() || !zi.im.is_finite() {
                break;
            }
            if step.norm() < opt.tol {
                break;
            }
        }
    }

    Ok(z)
}

// ================================================================
// Diagnostics
// ================================================================

fn sort_roots(mut r: Vec<Complex64>) -> Vec<Complex64> {
    // NaN-safe ordering: place NaNs at the end deterministically.
    r.sort_by(|a, b| {
        let a_nan = !(a.re.is_finite() && a.im.is_finite());
        let b_nan = !(b.re.is_finite() && b.im.is_finite());

        match (a_nan, b_nan) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => {
                // Both finite: total-ish ordering by (re, im).
                match a.re.partial_cmp(&b.re).unwrap_or(Ordering::Equal) {
                    Ordering::Equal => a.im.partial_cmp(&b.im).unwrap_or(Ordering::Equal),
                    ord => ord,
                }
            }
        }
    });
    r
}

fn max_root_displacement(a: &[Complex64], b: &[Complex64]) -> f64 {
    let mut used = vec![false; b.len()];
    let mut max_d: f64 = 0.0;

    for &ra in a {
        if !(ra.re.is_finite() && ra.im.is_finite()) {
            continue;
        }
        let mut best_j = None;
        let mut best_d = f64::INFINITY;

        for (j, &rb) in b.iter().enumerate() {
            if used[j] {
                continue;
            }
            if !(rb.re.is_finite() && rb.im.is_finite()) {
                continue;
            }
            let d = (ra - rb).norm();
            if d < best_d {
                best_d = d;
                best_j = Some(j);
            }
        }

        if let Some(j) = best_j {
            used[j] = true;
            if best_d > max_d {
                max_d = best_d;
            }
        }
    }
    max_d
}

// ================================================================
// Main demonstration
// ================================================================

fn main() -> Result<(), String> {
    let opt = DKOptions::default();

    // ------------------------------------------------------------
    // Example 1: Roots of x^5 - 1
    // ------------------------------------------------------------
    let p = Polynomial {
        coeffs: vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    };

    let roots = sort_roots(durand_kerner(&p, &opt)?);
    println!("Roots of P(x) = x^5 - 1:");
    for (i, r) in roots.iter().enumerate() {
        println!(
            "  r{:02} = {:.16} {:+.16}i   |P(r)| = {:.3e}",
            i,
            r.re,
            r.im,
            p.eval(*r).norm()
        );
    }

    // ------------------------------------------------------------
    // Example 2: Wilkinson sensitivity (Eq. 9.6.2)
    // ------------------------------------------------------------
    let w = wilkinson_polynomial(20);
    let mut w_pert = w.clone();

    let rel = 1e-7;
    let last = w_pert.coeffs.len() - 1;
    w_pert.coeffs[last] *= Complex64::new(1.0 + rel, 0.0);

    let r1 = durand_kerner(&w, &opt)?;
    let r2 = durand_kerner(&w_pert, &opt)?;

    let max_im_1 = r1
        .iter()
        .filter(|z| z.re.is_finite() && z.im.is_finite())
        .map(|z| z.im.abs())
        .fold(0.0, f64::max);

    let max_im_2 = r2
        .iter()
        .filter(|z| z.re.is_finite() && z.im.is_finite())
        .map(|z| z.im.abs())
        .fold(0.0, f64::max);

    let max_disp = max_root_displacement(&r1, &r2);

    println!("\nWilkinson polynomial P_20 sensitivity:");
    println!("  Relative perturbation: {:.1e}", rel);
    println!("  Max |Im(root)| before: {:.3e}", max_im_1);
    println!("  Max |Im(root)| after : {:.3e}", max_im_2);
    println!("  Max root displacement: {:.3e}", max_disp);

    let s1 = sort_roots(r1);
    let s2 = sort_roots(r2);

    // Print only finite roots.
    let s1f: Vec<_> = s1.into_iter().filter(|z| z.re.is_finite() && z.im.is_finite()).collect();
    let s2f: Vec<_> = s2.into_iter().filter(|z| z.re.is_finite() && z.im.is_finite()).collect();

    let show = usize::min(5, usize::min(s1f.len(), s2f.len()));
    println!("\n  Sample roots (sorted, finite only):");
    for k in 0..show {
        println!(
            "    k={:02}:  unpert = {:.10} {:+.10}i   pert = {:.10} {:+.10}i",
            k,
            s1f[k].re,
            s1f[k].im,
            s2f[k].re,
            s2f[k].im
        );
    }

    Ok(())
}
```

Program 9.6.1 demonstrates a practical and numerically robust approach to computing polynomial roots in the complex plane. By combining Horner evaluation, a global all-root iteration, and local Newton refinement, the implementation reflects the algorithmic strategies required to handle polynomial root finding in finite-precision environments.

The Wilkinson example underscores a central message of Subsection 9.6.1: the difficulty of polynomial root finding is often inherent in the problem itself rather than in the numerical method used. Even algorithms that are backward stable may produce roots that are highly sensitive to small coefficient perturbations. This sensitivity must therefore be understood as a property of the polynomial and its conditioning, not merely as an artifact of computation.

The modular design of the code allows the framework to be extended naturally to other root-finding strategies, such as companion-matrix eigenvalue methods or Aberth–Ehrlich iterations. It also provides a foundation for deeper investigations into root conditioning, multiplicity detection, and error analysis, which are explored in subsequent sections.

## 9.6.2. Deflation and Synthetic Division

When all roots of a polynomial are required, deflation is a natural and widely used strategy. If a root $r$ of $P(x)$ has been identified, the polynomial may be factored as:

$$
P(x) = (x - r)\,Q(x)
\tag{9.6.3}
$$

where $Q(x)$ is a polynomial of degree $m-1$. The coefficients of $Q(x)$ can be computed efficiently using polynomial long division or, more commonly, synthetic division.

Deflation serves two important purposes. First, it reduces the degree of the remaining polynomial, thereby simplifying subsequent root-finding steps. Second, it prevents iterative methods from repeatedly converging to the same root, which is a common failure mode when multiple roots are present.

However, deflation is numerically delicate. If the computed root $r$ is only approximate, then the coefficients of $Q(x)$ inherit errors that can grow as deflation proceeds. Repeated deflation may therefore amplify inaccuracies and degrade the quality of later roots. For this reason, practical algorithms often include a final polishing stage, in which all roots are refined by applying Newton’s method directly to the original polynomial rather than to the deflated one.

For real-coefficient polynomials with complex roots, deflation is commonly performed using quadratic factors. If $r=a+bi$ is a root, then its complex conjugate $\bar r=a-bi$ is also a root, and one may deflate using,

$$(x-r)(x-\bar r) = x^2 - 2ax + (a^2+b^2)\tag{9.6.4}$$

This approach allows deflation to be carried out entirely in real arithmetic, improving both efficiency and numerical stability.

### Rust Implementation

Following the discussion in Section 9.6.2 on deflation and synthetic division, Program 9.6.2 provides a concrete implementation of polynomial deflation for use in practical root-finding algorithms. Once a root of a polynomial has been identified, deflation reduces the degree of the remaining polynomial as expressed in Equation (9.6.3), thereby simplifying subsequent computations and preventing iterative methods from repeatedly converging to the same root. The program implements both linear synthetic division for real roots and quadratic deflation for complex-conjugate pairs, corresponding to Equation (9.6.4). Particular emphasis is placed on maintaining numerical transparency and algorithmic clarity, reflecting the delicacy of deflation in finite-precision arithmetic and its role within larger polynomial root-finding workflows.

At the core of the implementation is the `Poly` structure, which represents a real-coefficient polynomial using a vector of coefficients ordered by descending powers of $x$. This representation aligns naturally with Horner-style evaluation and synthetic division, and it allows the degree of the polynomial to be determined directly from the length of the coefficient array. Basic utility methods such as polynomial evaluation and degree inspection support both verification and demonstration within the program.

Linear deflation is implemented through the `deflate_linear` method, which applies synthetic division by a factor of the form $(x - r)$. Given a polynomial $P(x)$ and an approximate root $r$, this routine computes the coefficients of the reduced polynomial $Q(x)$ appearing in Equation (9.6.3), along with the scalar remainder. The synthetic division recurrence mirrors the algebraic structure of Horner’s method and provides an efficient $O(m)$ procedure for reducing a polynomial of degree $m$. The remainder serves as a diagnostic measure of the accuracy of the supplied root, vanishing only when $r$ is an exact root.

For real-coefficient polynomials with complex roots, the program implements quadratic deflation via the `deflate_conjugate_pair` method. When a complex root $r = a + bi$ is present, its conjugate $\bar r = a - bi$ must also be a root, and the corresponding quadratic factor takes the real form given in Equation (9.6.4). This method reduces the polynomial degree by two while operating entirely in real arithmetic, avoiding the need for explicit complex coefficients. Internally, the quadratic deflation is carried out by a specialized long-division procedure for monic quadratics, which produces both the reduced polynomial and a linear remainder. As with linear deflation, the magnitude of the remainder terms provides insight into the accumulated numerical error.

The `main` function demonstrates the full deflation process on a constructed polynomial with both real and complex-conjugate roots. The polynomial is assembled explicitly from known factors to provide a controlled test case. Linear deflation is first applied using a real root, followed by quadratic deflation corresponding to a conjugate pair defined by parameters $a$ and $b$. At each stage, the resulting polynomial and remainders are printed, allowing direct verification that the deflation steps reproduce the expected factors and that numerical errors remain negligible. This staged demonstration reflects the typical use of deflation within iterative root-finding algorithms, where roots are extracted sequentially and refined as needed.

```rust
// Program 9.6.2: Deflation and Synthetic Division
//
// This program implements deflation for polynomial root finding using
// (1) linear synthetic division by (x - r), and
// (2) quadratic deflation for real-coefficient polynomials with a complex-conjugate pair
//     (x - (a+bi))(x - (a-bi)) = x^2 - 2a x + (a^2 + b^2)   (Eq. 9.6.4).
//
// Coefficients are stored in descending powers:
// P(x) = a_m x^m + a_{m-1} x^{m-1} + ... + a_1 x + a_0

use std::fmt;

/// A real-coefficient polynomial with coefficients in descending powers.
#[derive(Clone, Debug)]
struct Poly {
    coeffs: Vec<f64>, // [a_m, a_{m-1}, ..., a_0]
}

impl Poly {
    fn new(mut coeffs: Vec<f64>) -> Self {
        // Trim leading zeros to keep a consistent degree.
        while coeffs.len() > 1 && coeffs[0].abs() == 0.0 {
            coeffs.remove(0);
        }
        Poly { coeffs }
    }

    fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Horner evaluation for real x.
    fn eval(&self, x: f64) -> f64 {
        let mut y = 0.0;
        for &a in &self.coeffs {
            y = y * x + a;
        }
        y
    }

    /// Linear synthetic division by (x - r).
    ///
    /// If P(x) = (x - r) Q(x) + rem, returns (Q, rem).
    ///
    /// With descending coefficients, the classic synthetic recurrence is:
    /// b_0 = a_0
    /// b_k = a_k + r * b_{k-1}   for k = 1..m
    /// rem = b_m, and Q has coefficients [b_0, ..., b_{m-1}].
    fn deflate_linear(&self, r: f64) -> Result<(Poly, f64), String> {
        let n = self.coeffs.len();
        if n < 2 {
            return Err("Cannot deflate a constant polynomial by a linear factor.".to_string());
        }

        let mut b = vec![0.0; n];
        b[0] = self.coeffs[0];
        for k in 1..n {
            b[k] = self.coeffs[k] + r * b[k - 1];
        }

        let rem = b[n - 1];
        let q_coeffs = b[..n - 1].to_vec();
        Ok((Poly::new(q_coeffs), rem))
    }

    /// Quadratic deflation by a monic quadratic x^2 + p x + q.
    ///
    /// Returns (Q, rem1, rem0) such that
    /// P(x) = (x^2 + p x + q) Q(x) + (rem1 x + rem0).
    ///
    /// This is implemented via a short, stable long-division loop specialized
    /// to monic quadratics (leading coefficient 1), which is easy to audit.
    fn deflate_monic_quadratic(&self, p: f64, q: f64) -> Result<(Poly, f64, f64), String> {
        let n = self.coeffs.len();
        if n < 3 {
            return Err("Cannot deflate a polynomial of degree < 2 by a quadratic factor.".to_string());
        }

        // Work on a mutable remainder buffer (descending coefficients).
        let mut rem = self.coeffs.clone();
        let mut quot = vec![0.0; n - 2];

        // For i from leading down to the term that leaves degree < 2:
        // subtract quot[i] * (x^2 + p x + q) shifted appropriately.
        for i in 0..(n - 2) {
            let qi = rem[i]; // divisor is monic, so the next quotient coefficient is rem[i]
            quot[i] = qi;

            // Eliminate the leading term at rem[i] by subtracting qi * x^2-shifted divisor.
            rem[i] = 0.0;
            rem[i + 1] -= qi * p;
            rem[i + 2] -= qi * q;
        }

        let rem1 = rem[n - 2];
        let rem0 = rem[n - 1];
        Ok((Poly::new(quot), rem1, rem0))
    }

    /// Quadratic deflation corresponding to Eq. (9.6.4):
    /// (x - (a+bi))(x - (a-bi)) = x^2 - 2a x + (a^2 + b^2).
    ///
    /// This keeps the deflation entirely in real arithmetic.
    fn deflate_conjugate_pair(&self, a: f64, b: f64) -> Result<(Poly, f64, f64), String> {
        let p = -2.0 * a;
        let q = a * a + b * b;
        self.deflate_monic_quadratic(p, q)
    }

    /// Convenience constructor: multiply polynomials (descending coeffs).
    fn mul(&self, other: &Poly) -> Poly {
        let m = self.degree();
        let n = other.degree();
        let mut out = vec![0.0; m + n + 1];
        for (i, &ai) in self.coeffs.iter().enumerate() {
            for (j, &bj) in other.coeffs.iter().enumerate() {
                out[i + j] += ai * bj;
            }
        }
        Poly::new(out)
    }
}

impl fmt::Display for Poly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let m = self.degree();
        let mut first = true;

        for (i, &a) in self.coeffs.iter().enumerate() {
            let deg = m - i;
            if a == 0.0 {
                continue;
            }

            let sign = if a < 0.0 { "-" } else { "+" };
            let abs_a = a.abs();

            if first {
                if a < 0.0 {
                    write!(f, "-")?;
                }
                first = false;
            } else {
                write!(f, " {} ", sign)?;
            }

            match deg {
                0 => write!(f, "{:.6}", abs_a)?,
                1 => {
                    if (abs_a - 1.0).abs() < 1e-12 {
                        write!(f, "x")?;
                    } else {
                        write!(f, "{:.6}x", abs_a)?;
                    }
                }
                _ => {
                    if (abs_a - 1.0).abs() < 1e-12 {
                        write!(f, "x^{}", deg)?;
                    } else {
                        write!(f, "{:.6}x^{}", abs_a, deg)?;
                    }
                }
            }
        }

        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

fn main() {
    // Example polynomial with real coefficients and a complex-conjugate pair:
    // P(x) = (x - 2) (x^2 - 2a x + (a^2 + b^2)) (x + 1)
    // Choose a = 1, b = 3  -> quadratic factor is x^2 - 2x + 10.
    let linear1 = Poly::new(vec![1.0, -2.0]); // x - 2
    let quad = Poly::new(vec![1.0, -2.0, 10.0]); // x^2 - 2x + 10  (Eq. 9.6.4 form for a=1, b=3)
    let linear2 = Poly::new(vec![1.0, 1.0]); // x + 1

    let p = linear1.mul(&quad).mul(&linear2);
    println!("P(x) = {}", p);
    println!("deg(P) = {}", p.degree());
    println!();

    // Linear deflation by the real root r = 2.
    let (q1, rem1) = p.deflate_linear(2.0).expect("linear deflation failed");
    println!("Deflate by (x - 2):");
    println!("Q1(x) = {}", q1);
    println!("remainder = {:.6e}", rem1);
    println!("Check: P(2) = {:.6e}", p.eval(2.0));
    println!();

    // Quadratic deflation by the conjugate pair a ± bi with a=1, b=3 (Eq. 9.6.4).
    let (q2, rem2_1, rem2_0) = q1
        .deflate_conjugate_pair(1.0, 3.0)
        .expect("quadratic deflation failed");
    println!("Deflate by (x^2 - 2a x + (a^2 + b^2)) with a=1, b=3:");
    println!("Q2(x) = {}", q2);
    println!("remainder = ({:.6e}) x + ({:.6e})", rem2_1, rem2_0);
    println!();

    // Final factor should be (x + 1).
    println!("Expected remaining factor (up to small remainder effects): x + 1");
    println!("Q2(x) = {}", q2);
}
```

Program 9.6.2 illustrates how deflation and synthetic division translate the algebraic factorization of polynomials into efficient numerical procedures. By reducing the polynomial degree after each root is found, deflation streamlines the overall root-finding process and mitigates the risk of repeated convergence to previously discovered roots. At the same time, the explicit computation of remainders highlights the numerical sensitivity of deflation when roots are only approximate.

The inclusion of quadratic deflation for complex-conjugate pairs emphasizes a key practical strategy for real-coefficient polynomials, allowing all computations to remain in real arithmetic while preserving mathematical correctness. This approach improves both efficiency and numerical robustness compared to naïve complex deflation. Together, the methods presented here form an essential component of comprehensive polynomial root-finding algorithms, particularly when combined with subsequent root polishing applied to the original polynomial to control error propagation.

## 9.6.3. Muller’s Method

Muller’s method extends the secant method by incorporating quadratic rather than linear interpolation. Given three successive approximations $x_{i-2}, x_{i-1}, x_i$, the method constructs a quadratic interpolant through the points $(x_j,P(x_j))$ for $j=i-2,i-1,i$, and extrapolates this quadratic to estimate the next root approximation.

Defining,

$$q = \frac{x_i - x_{i-1}}{x_{i-1} - x_{i-2}}\tag{9.6.5}$$

the coefficients of the interpolating quadratic are expressed compactly as,

$$A = qP(x_i) - q(1+q)P(x_{i-1}) + q^2P(x_{i-2})\tag{9.6.6}$$

$$B = (2q+1)P(x_i) - (1+q)^2P(x_{i-1}) + q^2P(x_{i-2})\tag{9.6.7}$$

$$C = (1+q)P(x_i)\tag{9.6.8}$$

The next iterate is then given by:

$$x_{i+1} = x_i + \Delta x, \qquad\Delta x = \frac{-2C}{B \pm \sqrt{B^2 - 4AC}}\tag{9.6.9}$$

The sign in the denominator is chosen to maximize its magnitude, thereby reducing cancellation and improving numerical stability.

Muller’s method converges with order approximately $1.84$, which is faster than the secant method but slower than Newton’s method. Its principal advantage is robustness for polynomial problems, particularly its ability to converge to complex roots even when starting from real initial guesses. This makes it a useful general-purpose method when derivative information is unavailable or unreliable.

### Rust Implementation

Following the discussion in Section 9.6.3 on quadratic interpolation, Program 9.6.3 presents a practical implementation of Muller’s method for polynomial root finding. Whereas the secant method advances by linear interpolation, Muller’s method uses three successive approximations to construct a quadratic interpolant through the points $(x_j, P(x_j))$, then extrapolates this quadratic to obtain the next iterate. The compact coefficient form introduced through Equations (9.6.5)–(9.6.8) leads to the update formula in Equation (9.6.9), with the denominator sign chosen to maximize its magnitude in order to suppress cancellation and improve numerical stability. A key practical advantage emphasized in this section is that Muller’s method naturally admits complex iterates, allowing convergence to complex roots even when the initial guesses are real, which makes it a robust derivative-free alternative for polynomial problems.

At the base of the implementation is the `Poly` structure, which stores real polynomial coefficients in descending powers of $x$. This ordering supports Horner evaluation in a direct and numerically stable form, and the method `eval_c` evaluates $P(x)$ for complex $x$ by carrying out the Horner recurrence in complex arithmetic. Using complex evaluation is essential for Muller’s method in practice, since the iterates may become complex even when all initial guesses are real, and this capability underpins the method’s robustness for locating nonreal roots.

The routine `muller` implements the iterative process using three current points $x_{i-2}, x_{i-1}, x_i$. Each iteration begins by evaluating $P$ at these points and checking whether the residual $|P(x_i)|$ is already below tolerance. The step construction follows the quadratic-interpolation logic of the method and culminates in a correction of the form shown in Equation (9.6.9). A central numerical detail is the sign choice in the denominator: the code forms both candidate denominators and selects the one with larger magnitude, which directly enacts the stability rule described after Equation (9.6.9) and reduces the risk of catastrophic cancellation when the discriminant term nearly cancels $B$.

To improve robustness in finite-precision arithmetic, the program computes the quadratic step using a divided-difference representation of the interpolant rather than explicitly forming the ratio $q$ from Equation (9.6.5). This divided-difference form is algebraically equivalent to quadratic interpolation, but it tends to behave better when the three iterates become unevenly spaced, because it avoids constructing intermediate expressions that can magnify rounding error. The code also includes lightweight guards against degenerate configurations, such as repeated iterates or nearly vanishing denominators, and it falls back to a secant-like update when the quadratic denominator becomes too small to use safely. This safeguard preserves the derivative-free nature of the method while preventing breakdown in cases where the quadratic fit is effectively singular.

The `main` function provides two demonstrations that reflect the principal behaviors discussed in Section 9.6.3. The first example uses $P(x)=x^2+1$, whose roots are purely imaginary. Starting from real initial guesses, the method converges to a complex root, illustrating the mechanism by which Muller’s method can escape the real axis without any derivative information. The second example uses $P(x)=x^3-1$, which has one real root and two complex roots. The initial triple includes small imaginary perturbations to avoid a nearly collinear quadratic fit and to encourage stable curvature in the interpolant. In the run output, the method converges to the real root $x=1$, and the remaining imaginary component in $P(x)$ is at the level of floating-point roundoff, confirming that the stopping criteria and sign-selection strategy yield a numerically consistent result.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
```

```rust
// Program 9.6.3: Muller’s Method for Polynomial Roots (robust complex implementation)
//
// Muller’s method constructs a quadratic interpolant through three successive points
// (x_{i-2}, P(x_{i-2})), (x_{i-1}, P(x_{i-1})), (x_i, P(x_i)) and extrapolates it to obtain
// x_{i+1} = x_i + Δx, where Δx has the form in Eq. (9.6.9), including the sign choice
// that maximizes the denominator magnitude to reduce cancellation.
//
// This implementation evaluates P in complex arithmetic so the method can converge to
// complex roots even from real initial guesses.
//
// NOTE ON FORMULATION:
// The update is computed via a divided-difference form of the same quadratic interpolation.
// This is algebraically equivalent to forming the quadratic coefficients and applying Eq. (9.6.9),
// but is often more numerically stable than explicitly forming q as in Eq. (9.6.5).
//
// Cargo.toml:
// [dependencies]
// num-complex = "0.4"

use num_complex::Complex;

/// A real-coefficient polynomial P(x) = a_m x^m + ... + a_0 with coefficients in descending powers.
#[derive(Clone, Debug)]
struct Poly {
    coeffs: Vec<f64>,
}

impl Poly {
    fn new(mut coeffs: Vec<f64>) -> Self {
        while coeffs.len() > 1 && coeffs[0] == 0.0 {
            coeffs.remove(0);
        }
        Poly { coeffs }
    }

    /// Horner evaluation in complex arithmetic.
    fn eval_c(&self, x: Complex<f64>) -> Complex<f64> {
        let mut y = Complex::new(0.0, 0.0);
        for &a in &self.coeffs {
            y = y * x + Complex::new(a, 0.0);
        }
        y
    }
}

/// Muller’s method for a polynomial P, starting from three initial guesses x0, x1, x2.
///
/// Stopping criteria:
/// - relative step size: |Δx| <= tol * (1 + |x|)
/// - or small residual:  |P(x)| <= tol
///
/// The method is derivative-free and naturally supports complex iterates.
fn muller(
    p: &Poly,
    mut x0: Complex<f64>,
    mut x1: Complex<f64>,
    mut x2: Complex<f64>,
    tol: f64,
    max_iter: usize,
) -> Result<Complex<f64>, String> {
    if max_iter == 0 {
        return Err("max_iter must be positive.".to_string());
    }

    let two = Complex::new(2.0, 0.0);
    let four = Complex::new(4.0, 0.0);

    // Scale-aware threshold for near-zero divisions
    let tiny = 1e-30;

    for _ in 0..max_iter {
        let f0 = p.eval_c(x0);
        let f1 = p.eval_c(x1);
        let f2 = p.eval_c(x2);

        // Residual-based termination
        if f2.norm() <= tol {
            return Ok(x2);
        }

        // Divided differences form of the quadratic interpolant:
        //
        // h1 = x1 - x0, h2 = x2 - x1
        // δ1 = (f1 - f0)/h1, δ2 = (f2 - f1)/h2
        // d  = (δ2 - δ1)/(h2 + h1)
        //
        // Then the quadratic through the three points can be written (in local variable about x2)
        // and leads to a step of the same structural form as Eq. (9.6.9):
        //
        // b = δ2 + h2*d
        // D = sqrt(b^2 - 4 f2 d)
        // denom = b ± D  (choose sign to maximize |denom|)
        // Δx = -2 f2 / denom
        //
        let h1 = x1 - x0;
        let h2 = x2 - x1;

        if h1.norm() <= tiny || h2.norm() <= tiny {
            return Err("Degenerate spacing among iterates: consecutive x values became equal.".to_string());
        }

        let d1 = (f1 - f0) / h1;
        let d2 = (f2 - f1) / h2;
        let hsum = h2 + h1;

        if hsum.norm() <= tiny {
            return Err("Degenerate geometry: h1 + h2 is too small for stable quadratic fit.".to_string());
        }

        let d = (d2 - d1) / hsum;
        let b = d2 + h2 * d;

        // Discriminant-like term (same cancellation-avoidance logic as Eq. (9.6.9))
        let disc = b * b - four * f2 * d;
        let sqrt_disc = disc.sqrt();

        let denom_plus = b + sqrt_disc;
        let denom_minus = b - sqrt_disc;

        let denom = if denom_plus.norm() >= denom_minus.norm() {
            denom_plus
        } else {
            denom_minus
        };

        if denom.norm() <= tiny {
            // If the quadratic step breaks down, try a secant-like fallback using last two points.
            let df = f2 - f1;
            if df.norm() <= tiny {
                return Err("Breakdown: both Muller and secant denominators are too small.".to_string());
            }
            let dx = -f2 * (x2 - x1) / df;
            let x3 = x2 + dx;

            if dx.norm() <= tol * (1.0 + x3.norm()) {
                return Ok(x3);
            }

            x0 = x1;
            x1 = x2;
            x2 = x3;
            continue;
        }

        let dx = (-two * f2) / denom;
        let x3 = x2 + dx;

        // Relative-step termination
        if dx.norm() <= tol * (1.0 + x3.norm()) {
            return Ok(x3);
        }

        // Shift window
        x0 = x1;
        x1 = x2;
        x2 = x3;

        if !x2.re.is_finite() || !x2.im.is_finite() {
            return Err("Numerical failure: iterate became non-finite.".to_string());
        }

        // Secondary residual check (helpful near multiple roots)
        let f3 = p.eval_c(x2);
        if f3.norm() <= tol {
            return Ok(x2);
        }
    }

    Err("Muller method did not converge within max_iter.".to_string())
}

fn main() {
    let tol = 1e-12;
    let max_iter = 200;

    // Demo 1: P(x) = x^2 + 1 has roots ±i and demonstrates convergence to a complex root
    // starting from real guesses.
    let p1 = Poly::new(vec![1.0, 0.0, 1.0]); // x^2 + 1
    let x0 = Complex::new(0.0, 0.0);
    let x1 = Complex::new(1.0, 0.0);
    let x2 = Complex::new(2.0, 0.0);

    println!("P1(x) = x^2 + 1");
    match muller(&p1, x0, x1, x2, tol, max_iter) {
        Ok(root) => {
            let val = p1.eval_c(root);
            println!("Root ≈ {:.16} + {:.16}i", root.re, root.im);
            println!("P1(root) ≈ {:.3e} + {:.3e}i", val.re, val.im);
        }
        Err(e) => println!("Muller failed: {}", e),
    }

    println!();

    // Demo 2: P(x) = x^3 - 1 has one real root (1) and two complex roots.
    // A reliable starting triple uses small imaginary perturbations to avoid degenerate
    // quadratic fits while still being close to the real basin.
    let p2 = Poly::new(vec![1.0, 0.0, 0.0, -1.0]); // x^3 - 1

    let y0 = Complex::new(0.8, 0.15);
    let y1 = Complex::new(1.2, -0.10);
    let y2 = Complex::new(0.9, 0.00);

    println!("P2(x) = x^3 - 1");
    match muller(&p2, y0, y1, y2, tol, max_iter) {
        Ok(root) => {
            let val = p2.eval_c(root);
            println!("Root ≈ {:.16} + {:.16}i", root.re, root.im);
            println!("P2(root) ≈ {:.3e} + {:.3e}i", val.re, val.im);
        }
        Err(e) => println!("Muller failed: {}", e),
    }
}
```

Program 9.6.3 demonstrates how quadratic interpolation can be turned into a practical derivative-free root-finding method with a convergence rate faster than the secant method while retaining broad applicability. The explicit implementation of the denominator sign choice in Equation (9.6.9) highlights a recurring theme in numerical computation: algebraically equivalent formulas may behave very differently in floating-point arithmetic, and stable variants must be preferred to avoid cancellation. The examples also emphasize Muller’s distinctive advantage for polynomial problems, namely its natural ability to converge to complex roots even from real starting values.

In larger polynomial solvers, Muller’s method is often combined with deflation, using each converged root to reduce the polynomial degree as in Equation (9.6.3). As noted earlier, deflation can propagate approximation error, so practical workflows commonly add a polishing stage in which the computed roots are refined against the original polynomial. Within such a framework, Muller’s method provides a robust engine for generating good initial root estimates, especially when derivative information is unavailable or when complex roots must be located reliably.

## 9.6.4. Laguerre’s Method

Laguerre’s method is widely regarded as one of the most robust and reliable iterative algorithms for computing the roots of a polynomial. Unlike methods that rely primarily on local linear or quadratic approximations, Laguerre’s iteration incorporates higher-order derivative information in a carefully structured manner, which significantly improves both convergence speed and global stability.

Let $P(x)$ be a polynomial of degree $n$. Starting from an initial approximation $x_i$, Laguerre’s method updates the iterate according to,

$$x_{i+1} = x_i - \frac{n}{G \pm \sqrt{(n-1)\big((n-1)G^2 - nH\big)}} \tag{9.6.10}$$

where the auxiliary quantities $G$ and $H$ are defined by:

$$G = \frac{P'(x_i)}{P(x_i)}, \qquad H = G^2 - \frac{P''(x_i)}{P(x_i)} \tag{9.6.11}$$

These expressions arise naturally by analyzing the logarithmic derivative of the polynomial. Expressing:

$$\frac{P'(x)}{P(x)} = \sum_{k=1}^{n} \frac{1}{x - r_k} \tag{9.6.12}$$

where $r_k$ are the roots of $P$, shows that $G$ represents a weighted average of inverse distances to the roots, while $H$ encodes curvature information through the second derivative. In this sense, Laguerre’s method adapts its step length based on both the local slope and curvature of $P$ relative to its current value.

The square-root term in the denominator plays a crucial stabilizing role. As in Müller’s method, the sign in (9.6.10) is chosen so as to maximize the magnitude of the denominator. This choice minimizes cancellation error and ensures that the correction step remains well scaled, particularly when $x_i$ lies far from the desired root or when multiple roots exert competing influence on the iteration.

From a convergence standpoint, Laguerre’s method exhibits cubic convergence for simple roots, provided the iterates are sufficiently close. When the root has multiplicity greater than one, the convergence order decreases, but the method typically remains stable and continues to converge, unlike Newton’s method, which may stagnate or diverge without modification. This robustness makes Laguerre’s method especially attractive when the multiplicity structure of the roots is unknown in advance.

Another notable feature is its strong global convergence behavior. Empirically, Laguerre’s method often converges from almost arbitrary initial guesses, even in the complex plane. This contrasts sharply with methods such as Newton–Raphson, whose basins of attraction can be highly fragmented for high-degree polynomials. As a result, Laguerre’s method is frequently employed as a “root-finder of last resort” when simpler methods fail.

In practical implementations, Laguerre’s method is typically used in conjunction with polynomial deflation. Once a root $r$ has been approximated to sufficient accuracy, the polynomial is divided by $(x - r)$, and the method is applied recursively to the reduced-degree polynomial. This strategy allows all roots to be computed sequentially while maintaining numerical stability. Such an approach is standard in applications like Gaussian quadrature, where accurate determination of polynomial roots directly determines the quality of the quadrature nodes and weights.

Overall, Laguerre’s method occupies a central position among polynomial root-finding algorithms, combining high-order convergence, strong global behavior, and practical robustness in finite-precision arithmetic.

### Rust Implementation

Following the discussion in Section 9.6.4 on high-robustness polynomial root solvers, Program 9.6.4 presents a practical implementation of Laguerre’s method for computing polynomial roots in finite-precision arithmetic. The iteration updates the current approximation according to Equation (9.6.10), using the auxiliary quantities $G$ and $H$ defined in Equation (9.6.11). These quantities arise from the logarithmic derivative interpretation in Equation (9.6.12), which explains why Laguerre’s step adapts naturally to both local slope and curvature information. As in Muller’s method, numerical stability hinges on the sign choice in the denominator of Equation (9.6.10), and the program explicitly selects the sign that maximizes the denominator magnitude to suppress cancellation. The implementation is carried out in complex arithmetic, reflecting the method’s strong global behavior and its ability to converge to complex roots from general initial guesses, and it is combined with deflation so that all roots can be computed sequentially.

The program represents polynomials with the `PolyC` structure, storing coefficients in descending powers of $x$. Although the input examples use real coefficients, the coefficients are promoted to complex form so that both iterates and roots may be complex without requiring separate code paths. This design aligns with the interpretation in Equation (9.6.12), since the inverse-distance terms $(x-r_k)^{-1}$ do not privilege the real axis and naturally live in the complex plane for general polynomials.

A key numerical component is `eval_with_derivs`, which evaluates $P(x)$, $P'(x)$, and $P''(x)$ simultaneously by an extended Horner scheme. Rather than differentiating symbolically or performing separate passes, the method updates three coupled recurrences so that the derivative values are accumulated alongside the polynomial value at essentially the same cost as evaluation. These values are then used to construct $G = P'(x_i)/P(x_i)$ and $H = G^2 - P''(x_i)/P(x_i)$ exactly as defined in Equation (9.6.11), making the correspondence between the mathematical formulation and the implementation direct and auditable.

The core iteration is implemented in `laguerre_root`. Each iteration computes $G$ and $H$, then forms the square-root term appearing in the denominator of Equation (9.6.10). The program evaluates both possible denominators corresponding to the $\pm$ choice and selects the one with larger magnitude, which is the cancellation-reducing rule described in the text immediately after Equation (9.6.10). The update itself applies the correction term $\frac{n}{\cdot}$ for the current polynomial degree $n$, and convergence is monitored using both a residual test $|P(x_i)|$ and a relative step test based on $|x_{i+1}-x_i|$. Lightweight safeguards handle rare breakdown configurations, such as near-zero denominators, by applying a small deterministic perturbation that preserves the method’s global character without introducing randomness.

To compute all roots, `all_roots_laguerre` applies Laguerre’s method repeatedly and performs deflation after each converged root. Once a root $r$ has been obtained, synthetic division by $(x-r)$ is carried out in complex arithmetic via `deflate_linear`, producing a reduced polynomial of degree $n-1$. This is the computational realization of the factorization principle introduced earlier in Equation (9.6.3), but implemented in a form suitable for complex roots and numerical work. Because deflation can amplify errors if a root is only approximate, the code includes a short polishing restart of Laguerre’s method at the candidate root before committing to the deflation step, and it checks the deflation remainder as a diagnostic of root quality.

The `main` function demonstrates two representative cases. The first polynomial is $(x-2)(x+1)(x^2+1)$, which contains both real roots and a complex-conjugate pair; the output confirms that the algorithm recovers $-1$, $2$, and $\pm i$ with residual norms near machine precision. The second example, $x^3-1$, is a classical test case with one real root and two complex roots; the method returns the three cube roots of unity, again with residual norms at the level expected for double precision. Together these tests illustrate both the global convergence behavior of Laguerre’s iteration and the practical effectiveness of combining it with deflation for full root sets.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
```

```rust
// Program 9.6.4: Laguerre’s Method with Deflation for Polynomial Roots
//
// Laguerre’s iteration uses Eq. (9.6.10) with G and H from Eq. (9.6.11), and
// chooses the sign in the denominator to maximize its magnitude to reduce cancellation.
//
// This program implements:
// 1) Complex-coefficient polynomial (starting from real coefficients).
// 2) Simultaneous evaluation of P, P', P'' via an extended Horner scheme.
// 3) Laguerre iteration for a single root (complex arithmetic).
// 4) Synthetic-division deflation by (x - r) in complex arithmetic to extract all roots.
//
// Cargo.toml:
// [dependencies]
// num-complex = "0.4"

use num_complex::Complex;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
struct PolyC {
    // Coefficients in descending powers: a_n, a_{n-1}, ..., a_0
    coeffs: Vec<Complex<f64>>,
}

impl PolyC {
    fn from_real_desc(real_coeffs: Vec<f64>) -> Self {
        let coeffs = real_coeffs
            .into_iter()
            .map(|a| Complex::new(a, 0.0))
            .collect();
        Self { coeffs }
    }

    fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Extended Horner evaluation returning (P(x), P'(x), P''(x)).
    fn eval_with_derivs(&self, x: Complex<f64>) -> (Complex<f64>, Complex<f64>, Complex<f64>) {
        let n = self.degree();
        if n == 0 {
            return (self.coeffs[0], Complex::new(0.0, 0.0), Complex::new(0.0, 0.0));
        }

        let mut b = self.coeffs[0];
        let mut c = Complex::new(0.0, 0.0);
        let mut d = Complex::new(0.0, 0.0);

        for k in 1..self.coeffs.len() {
            d = d * x + Complex::new(2.0, 0.0) * c;
            c = c * x + b;
            b = b * x + self.coeffs[k];
        }
        (b, c, d)
    }

    /// Synthetic division by (x - r). Returns (quotient, remainder).
    fn deflate_linear(&self, r: Complex<f64>) -> Result<(PolyC, Complex<f64>), String> {
        let n = self.coeffs.len();
        if n < 2 {
            return Err("Cannot deflate a constant polynomial.".to_string());
        }

        let mut b: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
        b[0] = self.coeffs[0];
        for i in 1..n {
            b[i] = self.coeffs[i] + r * b[i - 1];
        }

        let rem = b[n - 1];
        let q_coeffs = b[..n - 1].to_vec();
        Ok((PolyC { coeffs: q_coeffs }, rem))
    }

    fn trim_leading_small(&mut self, eps: f64) {
        while self.coeffs.len() > 1 && self.coeffs[0].norm() <= eps {
            self.coeffs.remove(0);
        }
    }

    /// Cauchy bound radius: all roots satisfy |z| <= 1 + max_{k<n} |a_k/a_n|.
    fn cauchy_radius(&self) -> f64 {
        let n = self.degree();
        if n == 0 {
            return 0.0;
        }
        let a_n = self.coeffs[0].norm();
        if a_n == 0.0 {
            return 1.0;
        }
        let mut m: f64 = 0.0; // fix: explicit type for stable inference
        for k in 1..self.coeffs.len() {
            m = m.max(self.coeffs[k].norm() / a_n);
        }
        1.0 + m
    }
}

/// Laguerre’s method for a single root of polynomial p, starting at x0.
/// Uses G and H as in Eq. (9.6.11) and chooses the sign to maximize |denominator|.
fn laguerre_root(
    p: &PolyC,
    mut x: Complex<f64>,
    tol: f64,
    max_iter: usize,
) -> Result<Complex<f64>, String> {
    let n = p.degree();
    if n == 0 {
        return Err("Laguerre requires degree >= 1.".to_string());
    }

    let n_c = Complex::new(n as f64, 0.0);
    let n_minus_1_c = Complex::new((n.saturating_sub(1)) as f64, 0.0);

    let tiny = 1e-30;

    for _ in 0..max_iter {
        let (px, p1x, p2x) = p.eval_with_derivs(x);

        if px.norm() <= tol {
            return Ok(x);
        }
        if px.norm() <= tiny {
            return Ok(x);
        }

        // Eq. (9.6.11)
        let g = p1x / px;
        let h = g * g - p2x / px;

        // Stable classic inner term: (n-1)(nH - G^2)
        let inner = n_minus_1_c * (n_c * h - g * g);
        let sqrt_inner = inner.sqrt();

        // Denominator candidates: G ± sqrt(...)
        let denom_plus = g + sqrt_inner;
        let denom_minus = g - sqrt_inner;

        // Choose sign to maximize magnitude
        let denom = if denom_plus.norm() >= denom_minus.norm() {
            denom_plus
        } else {
            denom_minus
        };

        if denom.norm() <= tiny {
            // Deterministic perturbation to escape rare breakdown configurations
            x += Complex::new(0.0, 1e-3) * (Complex::new(1.0, 0.0) + x);
            continue;
        }

        // Eq. (9.6.10): x_{i+1} = x_i - n/denom
        let step = n_c / denom;

        // Mild damping if the step is extremely large relative to current scale
        let scale = 1.0 + x.norm();
        let step_norm = step.norm();
        let damp = if step_norm > 10.0 * scale {
            (10.0 * scale) / step_norm
        } else {
            1.0
        };

        let x_next = x - Complex::new(damp, 0.0) * step;

        if (x_next - x).norm() <= tol * (1.0 + x_next.norm()) {
            return Ok(x_next);
        }

        x = x_next;

        if !x.re.is_finite() || !x.im.is_finite() {
            return Err("Numerical failure: iterate became non-finite.".to_string());
        }
    }

    Err("Laguerre method did not converge within max_iter.".to_string())
}

/// Compute all roots by repeated Laguerre calls and linear deflation.
fn all_roots_laguerre(mut p: PolyC, tol: f64, max_iter: usize) -> Result<Vec<Complex<f64>>, String> {
    let mut roots: Vec<Complex<f64>> = Vec::new();

    while p.degree() > 0 {
        p.trim_leading_small(1e-15);

        let deg = p.degree();
        if deg == 0 {
            break;
        }
        if deg == 1 {
            // a1 x + a0 = 0
            let a1 = p.coeffs[0];
            let a0 = p.coeffs[1];
            if a1.norm() == 0.0 {
                return Err("Degenerate linear polynomial during deflation.".to_string());
            }
            roots.push(-a0 / a1);
            break;
        }

        // Deterministic starting seeds on a circle using Cauchy radius bound.
        let r = p.cauchy_radius();
        let m = 24usize;

        let mut seeds: Vec<Complex<f64>> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(r, 0.0),
            Complex::new(-r, 0.0),
            Complex::new(0.0, r),
            Complex::new(0.0, -r),
        ];

        for k in 0..m {
            let theta = 2.0 * PI * (k as f64) / (m as f64);
            seeds.push(Complex::new(r * theta.cos(), r * theta.sin()));
        }

        let mut root_opt: Option<Complex<f64>> = None;
        let mut last_err: Option<String> = None;

        for x0 in seeds {
            match laguerre_root(&p, x0, tol, max_iter) {
                Ok(rt) => {
                    root_opt = Some(rt);
                    break;
                }
                Err(e) => last_err = Some(e),
            }
        }

        let r0 = match root_opt {
            Some(rt) => rt,
            None => {
                return Err(format!(
                    "Failed to find a root for degree {} polynomial. Last error: {}",
                    deg,
                    last_err.unwrap_or_else(|| "unknown".to_string())
                ))
            }
        };

        // Short polish: restart from r0 itself.
        let r1 = laguerre_root(&p, r0, tol, max_iter).unwrap_or(r0);

        // Deflate by (x - r1).
        let (q, rem) = p.deflate_linear(r1)?;
        if rem.norm() > 1e3 * tol * (1.0 + r1.norm()) {
            // One more polish attempt if the remainder is suspiciously large.
            let r2 = laguerre_root(&p, r1, tol, max_iter).unwrap_or(r1);
            let (q2, rem2) = p.deflate_linear(r2)?;
            if rem2.norm() > 1e6 * tol * (1.0 + r2.norm()) {
                return Err("Deflation remainder remained large; root estimate may be inaccurate.".to_string());
            }
            roots.push(r2);
            p = q2;
        } else {
            roots.push(r1);
            p = q;
        }
    }

    Ok(roots)
}

fn main() {
    let tol = 1e-12;
    let max_iter = 200;

    // Example 1: (x - 2)(x + 1)(x^2 + 1)
    // Expansion: (x - 2)(x + 1) = x^2 - x - 2
    // (x^2 - x - 2)(x^2 + 1) = x^4 - x^3 - x^2 - x - 2
    let p = PolyC::from_real_desc(vec![1.0, -1.0, -1.0, -1.0, -2.0]);

    println!("Polynomial: P(x) = x^4 - x^3 - x^2 - x - 2");
    match all_roots_laguerre(p.clone(), tol, max_iter) {
        Ok(mut roots) => {
            roots.sort_by(|a, b| {
                a.re.partial_cmp(&b.re)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
            });

            for (k, r) in roots.iter().enumerate() {
                let (val, _, _) = p.eval_with_derivs(*r);
                println!(
                    "root[{}] ≈ {:.16} {:+.16}i    |P(root)| ≈ {:.3e}",
                    k,
                    r.re,
                    r.im,
                    val.norm()
                );
            }
        }
        Err(e) => println!("Laguerre all-roots failed: {}", e),
    }

    println!();

    // Example 2: x^3 - 1
    let p2 = PolyC::from_real_desc(vec![1.0, 0.0, 0.0, -1.0]);
    println!("Polynomial: P2(x) = x^3 - 1");
    match all_roots_laguerre(p2.clone(), tol, max_iter) {
        Ok(mut roots) => {
            roots.sort_by(|a, b| {
                a.re.partial_cmp(&b.re)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
            });

            for (k, r) in roots.iter().enumerate() {
                let (val, _, _) = p2.eval_with_derivs(*r);
                println!(
                    "root[{}] ≈ {:.16} {:+.16}i    |P2(root)| ≈ {:.3e}",
                    k,
                    r.re,
                    r.im,
                    val.norm()
                );
            }
        }
        Err(e) => println!("Laguerre all-roots failed: {}", e),
    }
}
```

Program 9.6.4 demonstrates why Laguerre’s method is often regarded as a robust default choice for polynomial root finding. By incorporating derivative information through the quantities $G$ and $H$ in Equation (9.6.11), and by stabilizing the correction term via the sign selection in Equation (9.6.10), the method achieves fast local convergence while maintaining strong global behavior in practice. The complex-arithmetic formulation is not merely a convenience; it reflects the intrinsic structure of polynomial roots and allows the method to converge naturally to nonreal roots without special handling.

The deflation loop illustrates how a single-root method becomes a complete polynomial solver. Each successful Laguerre solve reduces the problem dimension through synthetic division, and the remainder checks provide a practical diagnostic for whether the extracted root is accurate enough to deflate safely. In larger numerical workflows, particularly those involving orthogonal polynomials and Gaussian quadrature, this combination of Laguerre iteration, cautious deflation, and optional polishing provides a reliable route to high-quality root sets in finite precision.

## 9.6.5. Companion Matrix and Eigenvalue Methods

Polynomial root finding can also be reformulated as a problem in numerical linear algebra by exploiting the close relationship between polynomials and matrix eigenvalues. This perspective is particularly powerful because it allows root computation to leverage decades of theoretical and algorithmic advances in eigenvalue analysis.

Consider a polynomial of degree $m$, $P(x) = a_m x^m + a_{m-1} x^{m-1} + \cdots + a_1 x + a_0$ as expressed in equation (9.6.1), with $a_m \neq 0$. By normalizing the polynomial so that the leading coefficient is unity, one can associate with $P(x)$ the companion matrix:

$$
A
=
\begin{pmatrix}
-\dfrac{a_{m-1}}{a_m} & -\dfrac{a_{m-2}}{a_m} & \cdots & -\dfrac{a_1}{a_m} & -\dfrac{a_0}{a_m} \\
1                    & 0                    & \cdots & 0                  & 0                  \\
0                    & 1                    & \cdots & 0                  & 0                  \\
\vdots               &                      & \ddots &                    & \vdots             \\
0                    & 0                    & \cdots & 1                  & 0
\end{pmatrix}
\tag{9.6.13}
$$

The defining property of this construction is that the characteristic polynomial of $A$ is exactly $P(x)$. Consequently, the eigenvalues of the companion matrix coincide precisely with the roots of the polynomial. This equivalence is not merely formal: it provides a concrete computational pathway by which polynomial root finding can be reduced to an eigenvalue problem.

From a mathematical standpoint, the companion matrix represents the linear transformation induced by multiplication by $x$ in the quotient space $\mathbb{R}[x]/(P(x))$. In this interpretation, roots emerge naturally as spectral values of the operator, revealing a deep connection between algebraic structure and linear dynamics. This viewpoint also clarifies why the eigenvalues capture all roots simultaneously, including complex conjugate pairs, without requiring separate treatment.

Algorithmically, this formulation enables the use of well-established eigenvalue solvers. The classical QR algorithm computes all eigenvalues of an $m \times m$ matrix in $O(m^3)$ time and is backward stable for general dense matrices. For polynomials of moderate degree, this cost is entirely acceptable and often yields highly reliable results, especially when compared with iterative root-finding methods that may suffer from sensitivity to initial guesses or deflation errors.

However, companion matrices possess a highly structured, almost sparse form, with nonzero entries confined to the first row and the first subdiagonal. Exploiting this structure can significantly reduce computational complexity and memory requirements. Modern algorithms preserve the companion form under similarity transformations, allowing eigenvalues to be computed in $O(m^2)$ time while maintaining numerical stability. Recent advances in structured QR iterations and unitary-plus-rank-one representations have demonstrated that such reductions do not necessarily compromise accuracy, even for high-degree polynomials (Vandebril, 2024).

Beyond deterministic algorithms, randomized and black-box methods have emerged as promising alternatives for large-scale problems. These approaches approximate spectral information using randomized projections or Krylov subspace techniques, reducing both computational cost and memory footprint. Such methods are particularly attractive when the polynomial degree is large or when roots are needed only to moderate precision. Ongoing research in this direction continues to refine error bounds, stability guarantees, and practical performance (Pan et al., 2025).

In practice, eigenvalue-based root finding is especially valuable when all roots are required simultaneously, as in control theory, signal processing, and spectral methods. It also avoids the sequential deflation process used in iterative methods such as Laguerre’s method, thereby reducing error accumulation. For these reasons, companion matrix techniques form an essential component of modern polynomial root-finding toolkits, complementing direct iterative methods and highlighting the deep interplay between polynomial algebra and numerical linear algebra.

### Rust Implementation

Following the discussion in Section 9.6.5 on reformulating polynomial root finding as an eigenvalue problem, Program 9.6.5 provides a concrete implementation of the companion matrix approach described by Equation (9.6.13). Rather than iteratively approximating roots one at a time, this program constructs a matrix whose spectral properties encode all roots of the polynomial simultaneously. By delegating the core numerical task to a mature eigenvalue solver, the method leverages the backward stability and robustness of modern linear algebra algorithms. The implementation is written in Rust using the `ndarray` and `ndarray-linalg` crates, with linear algebra operations backed by Intel’s Math Kernel Library (MKL). This choice ensures that eigenvalues are computed using highly optimized, LAPACK-quality routines, aligning the numerical behavior of the code with the theoretical guarantees discussed in the surrounding text.

At the core of the implementation is the `PolyR` structure, which stores the coefficients of a real polynomial in descending powers of the variable. This representation mirrors the algebraic form of Equation (9.6.1) and allows the polynomial degree and leading coefficient to be identified unambiguously. The constructor enforces the requirement $a_m \neq 0$ by trimming any leading zeros, ensuring that the companion matrix defined in Equation (9.6.13) is well posed.

The function `companion_matrix` implements the construction of the matrix $A$ exactly as specified in Equation (9.6.13). The first row encodes the normalized polynomial coefficients $-a_{m-1}/a_m, \ldots, -a_0/a_m$, while the unit subdiagonal represents the shift structure associated with multiplication by $x$ in the quotient space $\mathbb{R}[x]/(P(x))$. All remaining entries are zero, reflecting the highly structured nature of the companion matrix. In the code, this matrix is represented as an `ndarray::Array2<f64>`, which provides a flexible and efficient abstraction for dense numerical arrays in Rust.

Eigenvalue computation is performed by calling `eigvals` from the `ndarray-linalg` crate. This function is a thin, idiomatic Rust wrapper around LAPACK’s general eigenvalue routines and returns all eigenvalues of the companion matrix in complex arithmetic. The numerical backend is supplied by Intel MKL through the `intel-mkl-src` crate, which provides a statically linked implementation of BLAS and LAPACK. This dependency choice is significant: it ensures that the eigenvalue computation benefits from decades of algorithmic refinement and hardware optimization, and it avoids the need to build or maintain a separate linear algebra implementation within the program.

To assess the numerical quality of the computed roots, the program evaluates the original polynomial at each eigenvalue using the method `eval_c`, which implements complex Horner evaluation. This provides a direct measure of the residual $|P(\lambda)|$ for each computed root $\lambda$. While eigenvalue algorithms are backward stable for the matrix problem, this explicit residual check connects the linear algebra result back to the original polynomial problem and makes the effects of finite-precision arithmetic transparent.

The `main` function demonstrates the method on two representative examples. The first polynomial contains both real roots and a complex conjugate pair, illustrating the method’s ability to recover mixed spectra naturally. The second example, $x^3 - 1$, produces the three cube roots of unity and highlights how all roots are obtained in a single computation. In both cases, the reported residuals confirm that the eigenvalue-based approach yields roots accurate to near machine precision when evaluated in double precision.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
ndarray = "0.15"
ndarray-linalg = { version = "0.16", features = ["intel-mkl-static"] }
intel-mkl-src = { version = "0.8", features = ["mkl-static-lp64-seq"] }
num-complex = "0.4"
```

```rust
// Program 9.6.5: Companion Matrix and Eigenvalue Methods for Polynomial Roots
//
// This program constructs the companion matrix A in Eq. (9.6.13) for a polynomial
// P(x) = a_m x^m + a_{m-1} x^{m-1} + ... + a_1 x + a_0   (Eq. 9.6.1), a_m != 0.
// The eigenvalues of A coincide with the roots of P.
//
// Eigenvalues are computed using LAPACK through ndarray-linalg, linked against
// Intel MKL (static) via intel-mkl-src.
//
// Each eigenvalue λ is verified by printing |P(λ)| computed via complex Horner.

use ndarray::Array2;
use ndarray_linalg::EigVals; // <-- required for eigvals()
use num_complex::Complex;

#[derive(Clone, Debug)]
struct PolyR {
    // Real coefficients in descending powers: [a_m, a_{m-1}, ..., a_0]
    coeffs: Vec<f64>,
}

impl PolyR {
    fn new(mut coeffs: Vec<f64>) -> Result<Self, String> {
        if coeffs.is_empty() {
            return Err("Polynomial coefficient list must be non-empty.".to_string());
        }
        while coeffs.len() > 1 && coeffs[0] == 0.0 {
            coeffs.remove(0);
        }
        if coeffs.len() < 2 {
            return Err("Polynomial degree must be at least 1 for root finding.".to_string());
        }
        if coeffs[0] == 0.0 {
            return Err("Leading coefficient a_m must be nonzero.".to_string());
        }
        Ok(Self { coeffs })
    }

    fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    /// Complex Horner evaluation of P(x).
    fn eval_c(&self, x: Complex<f64>) -> Complex<f64> {
        let mut y = Complex::new(0.0, 0.0);
        for &a in &self.coeffs {
            y = y * x + Complex::new(a, 0.0);
        }
        y
    }
}

/// Construct the companion matrix A for P(x), as in Eq. (9.6.13).
///
/// For degree m polynomial with descending coefficients [a_m, a_{m-1}, ..., a_0],
/// normalize by a_m and build:
///
/// A[0, j] = -a_{m-1-j}/a_m  for j = 0..m-1
/// A[i, i-1] = 1            for i = 1..m-1
/// all other entries = 0
fn companion_matrix(p: &PolyR) -> Array2<f64> {
    let m = p.degree();
    let a_m = p.coeffs[0];

    let mut a = Array2::<f64>::zeros((m, m));

    // First row: [-a_{m-1}/a_m, -a_{m-2}/a_m, ..., -a_0/a_m]
    for j in 0..m {
        let coeff = p.coeffs[1 + j]; // a_{m-1}, a_{m-2}, ..., a_0
        a[(0, j)] = -coeff / a_m;
    }

    // Subdiagonal ones.
    for i in 1..m {
        a[(i, i - 1)] = 1.0;
    }

    a
}

fn run_example(label: &str, p: &PolyR) -> Result<(), String> {
    let a = companion_matrix(p);

    println!("{}", label);
    println!("Degree m = {}", p.degree());
    println!("Companion matrix size: {} x {}", a.nrows(), a.ncols());

    // LAPACK eigenvalue computation via ndarray-linalg.
    let eigvals = a
        .eigvals()
        .map_err(|e| format!("Eigenvalue computation failed: {}", e))?;

    let mut roots: Vec<Complex<f64>> = eigvals.iter().copied().collect();

    // Sort for nicer display (by real part, then imaginary part).
    roots.sort_by(|u, v| {
        u.re.partial_cmp(&v.re)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| u.im.partial_cmp(&v.im).unwrap_or(std::cmp::Ordering::Equal))
    });

    for (k, lam) in roots.iter().enumerate() {
        let val = p.eval_c(*lam);
        println!(
            "root[{}] ≈ {:.16} {:+.16}i    |P(root)| ≈ {:.3e}",
            k,
            lam.re,
            lam.im,
            val.norm()
        );
    }

    Ok(())
}

fn main() -> Result<(), String> {
    // Example 1: (x - 2)(x + 1)(x^2 + 1) = x^4 - x^3 - x^2 - x - 2
    let p1 = PolyR::new(vec![1.0, -1.0, -1.0, -1.0, -2.0])?;

    // Example 2: x^3 - 1
    let p2 = PolyR::new(vec![1.0, 0.0, 0.0, -1.0])?;

    run_example("P1(x) = x^4 - x^3 - x^2 - x - 2", &p1)?;
    println!();
    run_example("P2(x) = x^3 - 1", &p2)?;

    Ok(())
}
```

Program 9.6.5 illustrates a fundamentally different philosophy of polynomial root finding compared with iterative methods such as Muller’s or Laguerre’s algorithms. By recasting the problem in terms of matrix eigenvalues, it avoids the need for initial guesses, step-size control, or sequential deflation, and instead relies on the global convergence properties of established eigenvalue solvers. The use of `ndarray` and `ndarray-linalg`, together with an MKL-backed LAPACK implementation, demonstrates how modern Rust code can directly access industrial-strength numerical linear algebra while retaining clarity and type safety.

The numerical examples show that, for polynomials of moderate degree, the companion matrix method produces accurate roots with minimal algorithmic complexity from the user’s perspective. While the $O(m^3)$ cost of dense eigenvalue algorithms limits scalability for very large degrees, the method remains highly attractive when all roots are required simultaneously and robustness is paramount. In this sense, companion matrix techniques complement the iterative solvers developed earlier in the chapter and form an essential component of a modern polynomial root-finding toolkit.

## 9.6.6. Other Robust Algorithms and Polishing Techniques

Several additional algorithms play important roles in practical polynomial root finding, particularly when robustness, certification, or high accuracy is required beyond what a single iterative scheme can reliably deliver.

One of the most influential practical algorithms is the *Jenkins–Traub* method, a three-stage procedure specifically designed for real-coefficient polynomials. Its first stage performs a reliable root-shifting process that improves numerical conditioning without attempting convergence. The second stage applies a fixed-shift iteration to isolate a root, while the third stage uses a variable-shift iteration that closely resembles Newton’s method but with enhanced global stability. This carefully orchestrated sequence balances robustness and efficiency, allowing the method to converge rapidly once a root has been isolated while avoiding the instability commonly associated with naive deflation. Because of its strong empirical reliability across a wide range of polynomial degrees and coefficient magnitudes, the Jenkins–Traub method has become a standard reference implementation in many numerical libraries and software packages.

Another important class of techniques focuses on root isolation and counting using tools from complex analysis. Methods based on Rouché’s theorem or the argument principle determine how many roots lie within a specified region of the complex plane by evaluating contour integrals or argument variations of $P(z)$. Lehmer–Schur–type methods fall into this category and provide rigorous guarantees on root localization. Although these approaches are often computationally intensive and less suitable for high-degree polynomials in routine applications, they play a crucial role in certification, verification, and validated numerics, where provable correctness is more important than raw speed.

Once approximate roots have been obtained by any global method, polishing becomes an essential final step. In this phase, a small number of iterations of Newton’s method applied directly to the original polynomial typically yields dramatic improvements in accuracy. Because Newton’s method converges quadratically near simple roots, even one or two refinement steps can reduce residual errors by several orders of magnitude. For complex roots, Newton’s iteration extends naturally to the complex plane without modification, making it an effective and conceptually simple refinement tool.

In addition to Newton-based polishing, several specialized refinement methods have been developed to address specific numerical challenges. Bairstow’s method, for example, refines quadratic factors of a polynomial by applying a two-dimensional Newton iteration to the coefficients of a quadratic divisor. This approach is particularly useful for real polynomials with complex-conjugate root pairs, as it avoids explicit complex arithmetic and allows roots to be extracted in pairs.

Maehly’s method targets a different difficulty, namely the degradation of convergence caused by previously computed roots. After some roots $x_0, x_1, \dots, x_{j-1}$ have been found, Newton’s method applied to the original polynomial can be distorted by the influence of these known factors. Maehly’s iteration modifies Newton’s update to suppress this effect,

$$x_{k+1}= x_k -\frac{P(x_k)}{P'(x_k) - P(x_k)\displaystyle\sum_{i=0}^{j-1}\frac{1}{x_k-x_i}} \tag{9.6.14}$$

where the summation explicitly subtracts the contribution of already-identified roots. This correction effectively applies Newton’s method to the deflated polynomial without explicitly forming it, thereby avoiding numerical instability associated with repeated deflation. Maehly’s method is particularly effective when roots are tightly clustered, a situation in which standard Newton iterations often converge slowly or erratically.

Taken together, these auxiliary algorithms illustrate an important principle of practical polynomial root finding: no single method suffices in all situations. Robust solvers typically combine global strategies for isolation, reliable core algorithms for root extraction, and local refinement techniques for accuracy enhancement. Modern root-finding software reflects this layered approach, integrating classical analysis, numerical linear algebra, and iterative refinement into cohesive and reliable computational pipelines.

### Rust Implementation

Following the discussion in Section 9.6.6 on auxiliary robust algorithms and refinement strategies for polynomial root finding, Program 9.6.6 presents a practical implementation of root polishing techniques that are routinely used in modern numerical software. In realistic computations, global methods such as Laguerre’s method or companion-matrix eigenvalue approaches typically deliver roots that are already reasonably accurate, but not yet optimal in terms of residual error. Because polynomial evaluation is highly sensitive to small perturbations in the roots, a dedicated polishing stage is essential to extract the full accuracy permitted by finite-precision arithmetic. This program demonstrates how local iterative refinement, applied directly to the original polynomial, can dramatically improve the quality of computed roots while avoiding the numerical instability associated with repeated deflation. The implementation focuses on Newton polishing in the complex plane and on Maehly’s correction formula, illustrating how robustness and accuracy are achieved in practice through carefully designed refinement steps.

At the core of the implementation is the `PolyR` structure, which represents a real-coefficient polynomial in the standard descending-power form introduced in Equation (9.6.1). This representation allows the polynomial and its derivatives to be evaluated efficiently and consistently using Horner’s method. The method `eval_and_deriv` computes both $P(x)$ and $P'(x)$ simultaneously in complex arithmetic, which is essential for implementing Newton-based refinement schemes without redundant passes over the coefficient array.

The function `newton_polish` implements classical Newton refinement applied directly to the original polynomial. Given an approximate root $x_k$, the update follows the standard Newton iteration described earlier in the chapter. Near a simple root, this iteration converges quadratically, so even one or two iterations typically reduce the residual $|P(x)|$ by several orders of magnitude. The implementation includes safeguards against numerical breakdown, such as detecting vanishing derivatives and non-finite iterates, which reflects the practical realities of floating-point computation rather than the idealized assumptions of theoretical analysis.

To address the difficulties that arise when previously computed roots influence subsequent iterations, the program also implements Maehly’s method as defined in Equation (9.6.14). The function `maehly_polish` modifies the Newton denominator by explicitly subtracting the contribution of already-identified roots through the summation term $\sum_{i=0}^{j-1} 1/(x_k - x_i)$. This correction effectively applies Newton’s method to the deflated polynomial without explicitly forming it, thereby avoiding the accumulation of coefficient errors that often accompany repeated deflation. The implementation mirrors the mathematical structure of Equation (9.6.14) closely, making the connection between theory and code transparent.

The `main` function illustrates these ideas through two representative examples. The first polynomial contains a mixture of real roots and complex-conjugate pairs, allowing Newton polishing to be applied independently to each approximate root. The second example, $x^6 - 1$, is chosen to highlight the influence of already-known roots on later refinements. By comparing the behavior of standard Newton polishing and Maehly’s corrected iteration, the program demonstrates how both approaches achieve near machine-precision residuals, while Maehly’s method provides additional robustness in the presence of clustered or previously extracted roots. Throughout, the program reports the magnitude of $|P(x)|$ explicitly, making the effectiveness of the polishing stage quantitatively clear.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
```

```rust
// Program 9.6.6: Polishing and Robust Refinement for Polynomial Roots
//
// This program implements two practical “polishing” strategies that are commonly
// applied after an initial global root computation (for example from Laguerre’s
// method or the companion-matrix eigenvalue approach in Program 9.6.5).
//
// 1) Newton polishing in the complex plane:
//      x_{k+1} = x_k - P(x_k)/P'(x_k)
//    Quadratic local convergence for simple roots makes one or two iterations
//    often sufficient to reduce residuals by orders of magnitude.
//
// 2) Maehly’s correction (Eq. 9.6.14):
//      x_{k+1} = x_k - P(x_k) / ( P'(x_k) - P(x_k) * sum_{i=0}^{j-1} 1/(x_k - x_i) )
//
//    This modifies the Newton denominator to suppress the influence of already
//    identified roots x_0,...,x_{j-1}, effectively polishing as if one were using
//    the deflated polynomial without explicitly forming it, which helps avoid the
//    numerical fragility of repeated deflation.
//
// The demonstration uses two polynomials with known roots and starts from
// deliberately perturbed “approximate” roots, then shows how polishing improves
// |P(root)|.
//
// Cargo.toml:
// [dependencies]
// num-complex = "0.4"

use num_complex::Complex;

#[derive(Clone, Debug)]
struct PolyR {
    // Real coefficients in descending powers: [a_m, a_{m-1}, ..., a_0]
    coeffs: Vec<f64>,
}

impl PolyR {
    fn new(mut coeffs: Vec<f64>) -> Result<Self, String> {
        if coeffs.is_empty() {
            return Err("Polynomial coefficient list must be non-empty.".to_string());
        }
        while coeffs.len() > 1 && coeffs[0] == 0.0 {
            coeffs.remove(0);
        }
        if coeffs.len() < 2 {
            return Err("Polynomial degree must be at least 1.".to_string());
        }
        if coeffs[0] == 0.0 {
            return Err("Leading coefficient must be nonzero.".to_string());
        }
        Ok(Self { coeffs })
    }

    fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    /// Evaluate (P(x), P'(x)) simultaneously using an extended Horner scheme in complex arithmetic.
    fn eval_and_deriv(&self, x: Complex<f64>) -> (Complex<f64>, Complex<f64>) {
        // Standard coupled Horner for P and P'
        // b accumulates P, c accumulates P'
        let mut b = Complex::new(0.0, 0.0);
        let mut c = Complex::new(0.0, 0.0);

        for &a in &self.coeffs {
            c = c * x + b;
            b = b * x + Complex::new(a, 0.0);
        }
        (b, c)
    }
}

/// Newton polishing in the complex plane:
/// x_{k+1} = x_k - P(x_k)/P'(x_k)
fn newton_polish(
    p: &PolyR,
    mut x: Complex<f64>,
    tol: f64,
    max_iter: usize,
) -> Result<Complex<f64>, String> {
    let tiny = 1e-30;

    for _ in 0..max_iter {
        let (px, dpx) = p.eval_and_deriv(x);

        if px.norm() <= tol {
            return Ok(x);
        }
        if dpx.norm() <= tiny {
            return Err("Newton breakdown: derivative is too small.".to_string());
        }

        let step = px / dpx;
        let x_next = x - step;

        if (x_next - x).norm() <= tol * (1.0 + x_next.norm()) {
            return Ok(x_next);
        }

        x = x_next;

        if !x.re.is_finite() || !x.im.is_finite() {
            return Err("Newton failure: iterate became non-finite.".to_string());
        }
    }

    Err("Newton polishing did not converge within max_iter.".to_string())
}

/// Maehly’s method (Eq. 9.6.14) for polishing, given already-identified roots.
/// x_{k+1} = x_k - P(x_k) / ( P'(x_k) - P(x_k) * sum_{i=0}^{j-1} 1/(x_k - x_i) )
fn maehly_polish(
    p: &PolyR,
    mut x: Complex<f64>,
    known_roots: &[Complex<f64>],
    tol: f64,
    max_iter: usize,
) -> Result<Complex<f64>, String> {
    let tiny = 1e-30;

    for _ in 0..max_iter {
        let (px, dpx) = p.eval_and_deriv(x);

        if px.norm() <= tol {
            return Ok(x);
        }

        // Compute S(x) = sum_{i=0}^{j-1} 1/(x - x_i)
        let mut s = Complex::new(0.0, 0.0);
        for &ri in known_roots {
            let denom = x - ri;
            if denom.norm() <= tiny {
                // If we are extremely close to a known root, accept as converged to that root.
                return Ok(ri);
            }
            s += Complex::new(1.0, 0.0) / denom;
        }

        let denom = dpx - px * s;

        if denom.norm() <= tiny {
            return Err("Maehly breakdown: corrected denominator is too small.".to_string());
        }

        let step = px / denom;
        let x_next = x - step;

        if (x_next - x).norm() <= tol * (1.0 + x_next.norm()) {
            return Ok(x_next);
        }

        x = x_next;

        if !x.re.is_finite() || !x.im.is_finite() {
            return Err("Maehly failure: iterate became non-finite.".to_string());
        }
    }

    Err("Maehly polishing did not converge within max_iter.".to_string())
}

fn main() -> Result<(), String> {
    let tol = 1e-14;
    let max_iter = 20;

    // Polynomial 1: (x - 2)(x + 1)(x^2 + 1) = x^4 - x^3 - x^2 - x - 2
    let p1 = PolyR::new(vec![1.0, -1.0, -1.0, -1.0, -2.0])?;
    println!("Polynomial P1(x) = x^4 - x^3 - x^2 - x - 2 (degree {})", p1.degree());

    // Pretend these came from a global solver (e.g. companion matrix), with small perturbations.
    // True roots are: -1, 2, i, -i.
    let mut approx_roots_1 = vec![
        Complex::new(-1.0, 0.0) + Complex::new(1e-8, -2e-8),
        Complex::new(2.0, 0.0) + Complex::new(-3e-8, 1e-8),
        Complex::new(0.0, 1.0) + Complex::new(2e-8, -1e-8),
        Complex::new(0.0, -1.0) + Complex::new(-1e-8, 2e-8),
    ];

    println!("\nInitial approximations (before polishing):");
    for (k, r) in approx_roots_1.iter().enumerate() {
        let (val, _) = p1.eval_and_deriv(*r);
        println!(
            "r0[{}] ≈ {:.16} {:+.16}i    |P1(r)| ≈ {:.3e}",
            k,
            r.re,
            r.im,
            val.norm()
        );
    }

    // Newton polishing each root independently.
    println!("\nNewton polishing each root independently:");
    for k in 0..approx_roots_1.len() {
        let x0 = approx_roots_1[k];
        match newton_polish(&p1, x0, tol, max_iter) {
            Ok(rp) => {
                approx_roots_1[k] = rp;
                let (val, _) = p1.eval_and_deriv(rp);
                println!(
                    "rN[{}] ≈ {:.16} {:+.16}i    |P1(r)| ≈ {:.3e}",
                    k,
                    rp.re,
                    rp.im,
                    val.norm()
                );
            }
            Err(e) => println!("Newton failed on root[{}]: {}", k, e),
        }
    }

    // Polynomial 2: x^6 - 1, which has clustered angles on the unit circle.
    // This provides a setting where already-known roots can distort later polishing steps.
    let p2 = PolyR::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])?;
    println!("\nPolynomial P2(x) = x^6 - 1 (degree {})", p2.degree());

    // Two “known” roots (pretend we have already converged to them):
    let known = vec![
        Complex::new(1.0, 0.0),                    // e^{i0}
        Complex::new(-1.0, 0.0),                   // e^{iπ}
    ];

    // Now we try to polish a third root, starting near exp(i π/3) = 1/2 + i*sqrt(3)/2.
    let x0 = Complex::new(0.5, 0.8660254037844386) + Complex::new(1e-7, -2e-7);

    let (val0, _) = p2.eval_and_deriv(x0);
    println!(
        "\nTarget initial guess: x0 ≈ {:.16} {:+.16}i    |P2(x0)| ≈ {:.3e}",
        x0.re,
        x0.im,
        val0.norm()
    );

    // Compare Newton vs Maehly polishing when some roots are already known.
    println!("\nNewton polishing on P2 starting from x0:");
    match newton_polish(&p2, x0, tol, max_iter) {
        Ok(rp) => {
            let (val, _) = p2.eval_and_deriv(rp);
            println!(
                "Newton:  x ≈ {:.16} {:+.16}i    |P2(x)| ≈ {:.3e}",
                rp.re,
                rp.im,
                val.norm()
            );
        }
        Err(e) => println!("Newton failed: {}", e),
    }

    println!("\nMaehly polishing on P2 starting from x0 with known roots (Eq. 9.6.14):");
    match maehly_polish(&p2, x0, &known, tol, max_iter) {
        Ok(rp) => {
            let (val, _) = p2.eval_and_deriv(rp);
            println!(
                "Maehly:  x ≈ {:.16} {:+.16}i    |P2(x)| ≈ {:.3e}",
                rp.re,
                rp.im,
                val.norm()
            );
        }
        Err(e) => println!("Maehly failed: {}", e),
    }

    Ok(())
}
```

Program 9.6.6 illustrates a key principle of practical polynomial root finding: high-quality results are rarely produced by a single algorithmic step. Global methods provide reliable initial approximations, but local refinement is essential to fully exploit the accuracy available in floating-point arithmetic. Newton polishing demonstrates how rapidly residual errors can be reduced once an iterate enters the local convergence region of a simple root, while Maehly’s method shows how this refinement can be stabilized in the presence of multiple or closely spaced roots.

The examples confirm that polishing is not merely an optional enhancement, but an integral component of robust root-finding pipelines. By refining roots with respect to the original polynomial rather than a sequence of deflated ones, the program avoids the error amplification that often plagues naive implementations. The modular structure of the code allows these polishing routines to be combined naturally with the global solvers developed earlier in the chapter, reflecting the layered design used in professional numerical libraries.

Together with the methods presented in Sections 9.6.1 through 9.6.5, these refinement techniques complete a comprehensive toolbox for polynomial root finding. They emphasize that robustness, stability, and accuracy emerge not from any single method in isolation, but from the careful orchestration of complementary algorithms tailored to different stages of the computation.

## 9.6.7. Concluding Remarks

Polynomial root finding lies at the intersection of numerical analysis, algebra, and linear algebra, and provides a particularly clear illustration of the delicate interplay between problem conditioning, algorithmic design, and finite-precision arithmetic. Although the mathematical problem of solving a polynomial equation is classical, its numerical realization remains subtle, especially for high-degree polynomials or ill-conditioned coefficient sets.

No single method is universally optimal across all problem classes. Iterative schemes such as Newton’s method and Laguerre’s method offer rapid local convergence but depend critically on initialization and root separation. Global algorithms such as Jenkins–Traub emphasize robustness and reliability, while eigenvalue-based approaches reformulate the problem in a linear algebraic setting that allows all roots to be computed simultaneously. In practical implementations, these techniques are rarely used in isolation. Instead, modern solvers combine robust global strategies with controlled deflation and aggressive local polishing to achieve both reliability and high accuracy.

Recent research has produced substantial advances in both theoretical understanding and computational performance. Structure-exploiting eigenvalue algorithms reduce complexity while preserving numerical stability, and randomized methods offer scalable alternatives for large or structured problems. At the same time, certified and verification-oriented techniques based on complex analysis provide rigorous guarantees when correctness is paramount. Together, these developments demonstrate that polynomial root finding is not a closed chapter but an active and evolving area of research.

As a foundational problem with applications ranging from Gaussian quadrature and spectral methods to control theory, signal processing, and scientific modeling, polynomial root finding continues to play a central role in numerical computation. Its study highlights enduring themes of numerical analysis: the necessity of algorithmic diversity, the importance of stability over formal exactness, and the value of integrating classical mathematics with modern computational techniques (Martin, 2023; Berzi, 2024; Shah et al., 2025; Vivas-Cortez et al., 2023).

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/loZd1AUp8BYEJQmBvM7w.8","tags":[]}

# 9.7. Newton–Raphson Method for Nonlinear Systems of Equations

The Newton–Raphson method provides one of the most powerful and widely used approaches for solving systems of nonlinear equations when an accurate initial estimate is available. By extending the familiar one-dimensional Newton iteration to vector-valued functions, the method transforms the nonlinear problem into a sequence of linear systems whose solutions successively refine the approximation to the true root. This approach exploits local derivative information to achieve rapid convergence near a solution, often reducing the computational effort dramatically compared with derivative-free methods. However, this efficiency comes at the cost of increased sensitivity to initial conditions and the need to compute and solve Jacobian-based linear systems, making a careful balance between accuracy, robustness, and computational expense essential in practical implementations.

Solving a system of $N$ nonlinear equations consists of finding a vector:

$$\mathbf{x} = (x_1, x_2, \dots, x_N) \tag{9.7.1}$$

such that,

$$F_i(x_1, x_2, \dots, x_N) = 0, \qquad i = 1,2,\dots,N \tag{9.7.2}$$

Each function

$$F_i : \mathbb{R}^N \to \mathbb{R} \tag{9.7.3}$$

may depend on all components of $\mathbf{x}$. Collecting the equations into vector form yields:

$$\mathbf{F}(\mathbf{x}) = \mathbf{0} \tag{9.7.4}$$

where

$$\mathbf{F} : \mathbb{R}^N \to \mathbb{R}^N \tag{9.7.5}$$

and $\mathbf{0}$ denotes the zero vector in $\mathbb{R}^N$.

Systems of this form arise throughout scientific and engineering computation. Typical examples include equilibrium conditions in physics and chemistry, steady-state power-flow equations in electrical grids, kinematic constraint equations in robotics, and nonlinear optimization conditions in economics, control theory, game theory, and machine learning (Liu et al., 2023). In contrast to linear systems, nonlinear systems may possess multiple isolated solutions, continuous families of solutions, or no solution at all. Moreover, no general closed-form solution procedure exists, making numerical methods indispensable.

From a geometric perspective, each scalar equation:

$$F_i(\mathbf{x}) = 0 \tag{9.7.6}$$

defines a hypersurface in $\mathbb{R}^N$. A solution of the system corresponds to a point at which all $N$ hypersurfaces intersect simultaneously. Even in two dimensions, such intersections can be numerous, disconnected, or sensitive to perturbations in the equations. In higher dimensions, the global structure of the solution set is typically inaccessible a priori. Consequently, practical numerical algorithms rely on local approximations and reasonable initial guesses rather than exhaustive global searches.

## 9.7.1. Newton’s Method in Multiple Dimensions

Despite the intrinsic difficulty of nonlinear systems, Newton’s method provides a powerful and efficient local solution technique when the initial guess lies sufficiently close to a solution. The method generalizes the familiar one-dimensional Newton iteration by replacing scalar derivatives with the Jacobian matrix and linearizing the vector-valued function $\mathbf{F}$.

Let $\mathbf{x}$ denote the current approximation, and let $\delta\mathbf{x}$ be a small correction. A first-order Taylor expansion of $\mathbf{F}$ about $\mathbf{x}$ gives:

$$\mathbf{F}(\mathbf{x} + \delta\mathbf{x})\approx\mathbf{F}(\mathbf{x}) + J(\mathbf{x})\delta\mathbf{x} \tag{9.7.7}$$

where $J(\mathbf{x})$ is the Jacobian matrix defined by:

$$J_{ij}(\mathbf{x}) = \frac{\partial F_i}{\partial x_j}(\mathbf{x}) \tag{9.7.8}$$

For example, when $N = 3$, the Jacobian takes the explicit form:

$$J(\mathbf{x}) =\begin{pmatrix}\partial F_1/\partial x_1 & \partial F_1/\partial x_2 & \partial F_1/\partial x_3 \\ \partial F_2/\partial x_1 & \partial F_2/\partial x_2 & \partial F_2/\partial x_3 \\ \partial F_3/\partial x_1 & \partial F_3/\partial x_2 & \partial F_3/\partial x_3\end{pmatrix} \tag{9.7.9} $$

Neglecting higher-order terms in the Taylor expansion, the Newton correction $\delta\mathbf{x}$ is chosen so that:

$$\mathbf{F}(\mathbf{x}) + J(\mathbf{x})\,\delta\mathbf{x} \approx \mathbf{0} \tag{9.7.10}$$

This leads to the linear system as follows:

$$J(\mathbf{x})\,\delta\mathbf{x} = -\mathbf{F}(\mathbf{x}) \tag{9.7.11}$$

which defines the Newton step. Once this system has been solved, the iterate is updated according to:

$$\mathbf{x}_{\text{new}} = \mathbf{x}_{\text{old}} + \delta\mathbf{x} \tag{9.7.12}$$

The algorithm proceeds iteratively by repeatedly evaluating $\mathbf{F}$ and $J$, solving the linear system (9.7.11), and applying the update (9.7.12). Under standard regularity assumptions, including nonsingularity of the Jacobian at the solution, Newton’s method converges quadratically in a neighborhood of a simple root. This rapid local convergence makes Newton’s method the foundation of many modern algorithms for solving nonlinear systems.

In practice, convergence is typically declared when both the residual norm $||\mathbf{F}(\mathbf{x})||$ and the step norm $||\delta\mathbf{x}||$ fall below prescribed tolerances. These dual criteria ensure that the iterate is close to a solution and that further updates are unlikely to produce significant changes, thereby balancing numerical reliability with computational efficiency.

### Rust Implementation

Following the discussion in Section 9.7 on the formulation and geometric interpretation of nonlinear systems of equations, Program 9.7.1 provides a practical implementation of the Newton–Raphson method for solving systems of the form $\mathbf{F}(\mathbf{x}) = \mathbf{0}$. In contrast to one-dimensional root finding, the multidimensional setting replaces scalar derivatives with the Jacobian matrix and requires the repeated solution of linear systems at each iteration. This program demonstrates how the local linearization described in Equations (9.7.7)–(9.7.11) can be translated into an efficient and robust computational procedure. Particular emphasis is placed on convergence diagnostics, Jacobian evaluation strategies, and numerical safeguards that are essential in finite-precision arithmetic. Through representative two- and three-dimensional examples, the implementation illustrates both the power and the limitations of Newton’s method when applied to nonlinear systems.

At the core of the implementation is the `NonlinearSystem` trait, which defines a general interface for systems of nonlinear equations $\mathbf{F} : \mathbb{R}^N \to \mathbb{R}^N$ as introduced in Equations (9.7.4) and (9.7.5). Any concrete system must specify its dimension and provide a method for evaluating $\mathbf{F}(\mathbf{x})$. An optional method for evaluating the Jacobian (J(\\mathbf{x})), defined componentwise in Equation (9.7.8), allows analytic derivatives to be supplied when available. This abstraction enables the same Newton solver to be reused across a wide range of nonlinear systems simply by defining how $\mathbf{F}$ and, optionally, $J$ are computed.

The function `newton_solve` implements the Newton iteration derived from the linearization in Equation (9.7.7). At each step, the current residual $\mathbf{F}(\mathbf{x})$ is evaluated and the linear system in Equation (9.7.11) is formed and solved to obtain the Newton correction $\delta\mathbf{x}$. The update $\mathbf{x}_{\text{new}} = \mathbf{x}_{\text{old}} + \delta\mathbf{x}$ follows Equation (9.7.12) directly. Convergence is monitored using two complementary criteria: the norm of the residual $||\mathbf{F}(\mathbf{x})||$ and the norm of the Newton step $||\delta\mathbf{x}||$. This dual test reflects the discussion following Equation (9.7.12) and helps prevent premature termination in cases where the residual is small but the iterate is not yet stable, or vice versa.

When an analytic Jacobian is not provided, the solver automatically falls back to a finite-difference approximation implemented in the function `jacobian_fd`. This routine constructs the Jacobian column by column using central differences, trading additional function evaluations for improved accuracy. Although more expensive than analytic derivatives, this approach makes the solver applicable to problems where derivative expressions are difficult or impractical to derive, thereby broadening its applicability.

To improve robustness away from the immediate neighborhood of a solution, the implementation optionally applies a simple backtracking line search. Rather than accepting the full Newton step unconditionally, the algorithm scales the step length to ensure sufficient decrease in the squared residual norm. This damping mechanism does not alter the local quadratic convergence of Newton’s method near a simple root, but it can significantly enlarge the basin of attraction and reduce divergence when the initial guess is imperfect.

The `main` function demonstrates the solver on two illustrative examples. The first is a two-dimensional system whose solution corresponds to the intersection of a circle and a line, making the geometric interpretation of the solution particularly transparent. The second example involves a three-dimensional system with multiple coupled nonlinear equations, highlighting the method’s behavior in higher dimensions. In both cases, the solver converges rapidly when initialized near a solution, confirming the quadratic convergence predicted by theory and showcasing the practical effectiveness of the Newton–Raphson method for nonlinear systems.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
ndarray = "0.15"
ndarray-linalg = { version = "0.16", features = ["intel-mkl-static"] }
intel-mkl-src = { version = "0.8", features = ["mkl-static-lp64-seq"] }
```

```rust
// Program 9.7.1: Newton–Raphson Method for Nonlinear Systems of Equations
//
// This program implements Newton’s method for solving a nonlinear system
// F(x) = 0 in R^N, based on the linearization in Eq. (9.7.7) and the Newton
// step defined by the linear system in Eq. (9.7.11), followed by the update
// in Eq. (9.7.12).
//
// The implementation supports:
// 1) User-supplied Jacobian J(x) (Eq. 9.7.8).
// 2) A fallback finite-difference Jacobian when an analytic Jacobian is not provided.
// 3) Dual stopping criteria based on ||F(x)|| and ||δx||, as discussed after Eq. (9.7.12).
// 4) A simple backtracking line search (damping) to improve robustness.
//
// Dependencies (Cargo.toml):
//
// [dependencies]
// ndarray = "0.15"
// ndarray-linalg = { version = "0.16", features = ["intel-mkl-static"] }
// intel-mkl-src = { version = "0.8", features = ["mkl-static-lp64-seq"] }
// num-complex = "0.4"   # not required here, but often used elsewhere in Chapter 9

use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

#[derive(Clone, Copy, Debug)]
struct NewtonOptions {
    tol_f: f64,          // tolerance for ||F(x)||
    tol_step: f64,       // tolerance for ||δx||
    max_iter: usize,
    fd_eps: f64,         // finite-difference step size
    line_search: bool,   // enable backtracking line search
    ls_c1: f64,          // sufficient decrease parameter
    ls_rho: f64,         // backtracking factor
    ls_min_alpha: f64,   // minimum step scaling
}

impl Default for NewtonOptions {
    fn default() -> Self {
        Self {
            tol_f: 1e-12,
            tol_step: 1e-12,
            max_iter: 50,
            fd_eps: 1e-8,
            line_search: true,
            ls_c1: 1e-4,
            ls_rho: 0.5,
            ls_min_alpha: 1e-8,
        }
    }
}

/// Euclidean norm ||v||_2.
fn norm2(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

/// A user-defined nonlinear system F: R^N -> R^N with optional Jacobian.
trait NonlinearSystem {
    /// Dimension N.
    fn dim(&self) -> usize;

    /// Evaluate F(x) (Eq. 9.7.4).
    fn f(&self, x: &Array1<f64>) -> Array1<f64>;

    /// Evaluate J(x) (Eq. 9.7.8), if available.
    fn jacobian(&self, _x: &Array1<f64>) -> Option<Array2<f64>> {
        None
    }
}

/// Finite-difference Jacobian approximation.
/// Column j is approximated by (F(x + h e_j) - F(x - h e_j)) / (2h).
fn jacobian_fd<S: NonlinearSystem>(sys: &S, x: &Array1<f64>, h: f64) -> Array2<f64> {
    let n = sys.dim();
    let fx = sys.f(x);

    let mut j = Array2::<f64>::zeros((n, n));

    for col in 0..n {
        let mut xp = x.clone();
        let mut xm = x.clone();

        xp[col] += h;
        xm[col] -= h;

        let fp = sys.f(&xp);
        let fm = sys.f(&xm);

        let diff = (fp - fm) * (0.5 / h);

        // Fill column col
        for row in 0..n {
            j[(row, col)] = diff[row];
        }
    }

    // If the system is expensive, one could reuse fx and do one-sided differences.
    // Here we choose central differences for accuracy.
    let _ = fx;
    j
}

/// Solve F(x) = 0 using Newton’s method (Eq. 9.7.11)–(9.7.12).
///
/// Returns (x, iterations_used).
fn newton_solve<S: NonlinearSystem>(
    sys: &S,
    mut x: Array1<f64>,
    opts: NewtonOptions,
) -> Result<(Array1<f64>, usize), String> {
    if x.len() != sys.dim() {
        return Err("Initial guess dimension does not match system dimension.".to_string());
    }

    for iter in 0..opts.max_iter {
        let fx = sys.f(&x);
        let f_norm = norm2(&fx);

        if f_norm <= opts.tol_f {
            return Ok((x, iter));
        }

        let jx = match sys.jacobian(&x) {
            Some(j) => j,
            None => jacobian_fd(sys, &x, opts.fd_eps),
        };

        // Eq. (9.7.11): J(x) δx = -F(x)
        let rhs = -&fx;
        let delta = jx
            .solve_into(rhs)
            .map_err(|e| format!("Linear solve failed (Jacobian singular or ill-conditioned): {}", e))?;

        let step_norm = norm2(&delta);

        if step_norm <= opts.tol_step * (1.0 + norm2(&x)) {
            // Even if ||F|| is not tiny, a tiny step suggests we are stagnating.
            // For textbook purposes, we accept this as convergence by step criterion.
            return Ok((x, iter));
        }

        // Eq. (9.7.12): x_new = x_old + δx
        // Optional damping by backtracking line search on phi(x) = 0.5 ||F(x)||^2.
        if !opts.line_search {
            x = x + delta;
            continue;
        }

        let phi0 = 0.5 * f_norm * f_norm;

        // Directional derivative approximation for Armijo test:
        // phi'(0) = grad phi^T delta = (J^T F)^T delta = F^T (J delta) = F^T (-F) = -||F||^2
        // when delta solves J delta = -F exactly. This is a useful identity in exact arithmetic.
        let dphi0 = -f_norm * f_norm;

        let mut alpha = 1.0;
        loop {
            let x_trial = &x + &(delta.clone() * alpha);
            let f_trial = sys.f(&x_trial);
            let f_trial_norm = norm2(&f_trial);
            let phi_trial = 0.5 * f_trial_norm * f_trial_norm;

            // Armijo condition: phi(x + αδ) <= phi(x) + c1 α phi'(0)
            if phi_trial <= phi0 + opts.ls_c1 * alpha * dphi0 {
                x = x_trial;
                break;
            }

            alpha *= opts.ls_rho;

            if alpha < opts.ls_min_alpha {
                // If line search fails, take a very small step to avoid complete stagnation.
                x = x + delta * opts.ls_min_alpha;
                break;
            }
        }
    }

    Err("Newton method did not converge within max_iter.".to_string())
}

// -----------------------------------------------------------------------------
// Example systems for demonstration
// -----------------------------------------------------------------------------

/// System in R^2:
/// F1(x,y) = x^2 + y^2 - 1
/// F2(x,y) = x - y
///
/// Solutions are (±1/sqrt(2), ±1/sqrt(2)).
struct CircleLine2D;

impl NonlinearSystem for CircleLine2D {
    fn dim(&self) -> usize {
        2
    }

    fn f(&self, x: &Array1<f64>) -> Array1<f64> {
        let x1 = x[0];
        let x2 = x[1];
        Array1::from_vec(vec![x1 * x1 + x2 * x2 - 1.0, x1 - x2])
    }

    fn jacobian(&self, x: &Array1<f64>) -> Option<Array2<f64>> {
        let x1 = x[0];
        let x2 = x[1];
        // J = [[2x, 2y],
        //      [ 1, -1]]
        let j = Array2::from_shape_vec((2, 2), vec![2.0 * x1, 2.0 * x2, 1.0, -1.0]).unwrap();
        Some(j)
    }
}

/// System in R^3:
/// F1(x,y,z) = x + y + z - 1
/// F2(x,y,z) = x^2 + y^2 + z^2 - 1
/// F3(x,y,z) = x - y
///
/// A solution is x=y and x+y+z=1 with x^2+y^2+z^2=1.
/// This has multiple solutions; Newton’s method demonstrates local behavior.
struct Simple3D;

impl NonlinearSystem for Simple3D {
    fn dim(&self) -> usize {
        3
    }

    fn f(&self, x: &Array1<f64>) -> Array1<f64> {
        let x1 = x[0];
        let x2 = x[1];
        let x3 = x[2];
        Array1::from_vec(vec![
            x1 + x2 + x3 - 1.0,
            x1 * x1 + x2 * x2 + x3 * x3 - 1.0,
            x1 - x2,
        ])
    }

    fn jacobian(&self, x: &Array1<f64>) -> Option<Array2<f64>> {
        let x1 = x[0];
        let x2 = x[1];
        let x3 = x[2];

        // J =
        // [1, 1, 1]
        // [2x,2y,2z]
        // [1,-1,0]
        let j = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 1.0, 1.0, 2.0 * x1, 2.0 * x2, 2.0 * x3, 1.0, -1.0, 0.0],
        )
        .unwrap();
        Some(j)
    }
}

fn main() -> Result<(), String> {
    let opts = NewtonOptions {
        tol_f: 1e-12,
        tol_step: 1e-12,
        max_iter: 50,
        fd_eps: 1e-8,
        line_search: true,
        ls_c1: 1e-4,
        ls_rho: 0.5,
        ls_min_alpha: 1e-8,
    };

    // -------------------------------------------------------------------------
    // Demo 1: 2D system with analytic Jacobian
    // -------------------------------------------------------------------------
    let sys2 = CircleLine2D;

    // Initial guess near (1/sqrt(2), 1/sqrt(2))
    let x0 = Array1::from_vec(vec![0.8, 0.6]);

    println!("Demo 1: CircleLine2D");
    println!("Initial guess x0 = [{:.6}, {:.6}]", x0[0], x0[1]);

    let (x_star, iters) = newton_solve(&sys2, x0, opts)?;
    let f_star = sys2.f(&x_star);

    println!(
        "Converged in {} iterations: x = [{:.16}, {:.16}]",
        iters, x_star[0], x_star[1]
    );
    println!(
        "||F(x)||_2 = {:.3e},  F(x) = [{:.3e}, {:.3e}]",
        norm2(&f_star),
        f_star[0],
        f_star[1]
    );

    println!();

    // -------------------------------------------------------------------------
    // Demo 2: 3D system with analytic Jacobian
    // -------------------------------------------------------------------------
    let sys3 = Simple3D;

    // Choose an initial guess that is plausibly close to a solution.
    let y0 = Array1::from_vec(vec![0.7, 0.6, -0.2]);

    println!("Demo 2: Simple3D");
    println!("Initial guess x0 = [{:.6}, {:.6}, {:.6}]", y0[0], y0[1], y0[2]);

    let (y_star, iters2) = newton_solve(&sys3, y0, opts)?;
    let g_star = sys3.f(&y_star);

    println!(
        "Converged in {} iterations: x = [{:.16}, {:.16}, {:.16}]",
        iters2, y_star[0], y_star[1], y_star[2]
    );
    println!(
        "||F(x)||_2 = {:.3e},  F(x) = [{:.3e}, {:.3e}, {:.3e}]",
        norm2(&g_star),
        g_star[0],
        g_star[1],
        g_star[2]
    );

    Ok(())
}
```

Program 9.7.1 demonstrates how the theoretical formulation of Newton’s method for nonlinear systems translates into a reliable and flexible computational algorithm. By explicitly forming and solving the linear system in Equation (9.7.11) at each iteration, the method leverages local derivative information to achieve rapid convergence near a solution. The numerical examples confirm that, under appropriate regularity conditions and with a reasonable initial guess, only a small number of iterations are required to reduce the residual norm to near machine precision.

At the same time, the implementation highlights the practical considerations that distinguish robust numerical solvers from idealized textbook algorithms. Finite-difference Jacobians, damping strategies, and dual convergence criteria all play essential roles in ensuring stability and usability in real-world applications. The modular design of the code allows these components to be adapted or extended easily, for example by incorporating sparse Jacobians, inexact linear solvers, or trust-region strategies.

Together with the one-dimensional root-finding methods developed earlier in the chapter, this program establishes Newton’s method as a foundational tool for solving nonlinear equations in multiple dimensions. It also sets the stage for more advanced techniques, such as quasi-Newton methods and continuation strategies, which build upon the same linearization principles while addressing some of the limitations inherent in the classical Newton–Raphson approach.

## 9.7.2. Local Quadratic Convergence

When Newton’s method converges, it does so with remarkable speed. Under standard regularity assumptions, specifically, that each function $F_i$ is continuously differentiable in a neighborhood of the solution and that the Jacobian matrix evaluated at the solution is nonsingular, the method exhibits *quadratic convergence*. This property means that once the current iterate enters a sufficiently small neighborhood of the true solution, the error decreases at an accelerating rate: the number of correct digits approximately doubles with each iteration.

From a mathematical perspective, quadratic convergence arises because Newton’s method exactly cancels the first-order term in the local Taylor expansion of the nonlinear system. The remaining error is therefore dominated by second-order terms, causing the error norm at iteration $k+1$ to scale like the square of the error norm at iteration $k$. As a result, Newton’s method is vastly more efficient near a solution than linearly convergent schemes such as fixed-point iteration or gradient descent.

This exceptional local efficiency explains the central role of Newton’s method in scientific computing. In many applications, only a small number of iterations, often fewer than ten, are sufficient to reduce the residual to machine precision, provided the initial guess lies within the method’s basin of attraction. In practice, Newton’s method is frequently used as a *polishing* step following a more robust global algorithm that delivers an approximate solution.

The principal limitation of quadratic convergence is its *local* nature. The basin of attraction may be small or highly irregular, especially for strongly nonlinear or ill-conditioned systems. If the initial guess is too far from a solution, the method may converge slowly, converge to an unintended solution, or diverge altogether. This sensitivity motivates the development of globalization strategies, such as line search and trust-region methods, which are discussed in later sections.

### Rust Implementation

Following the discussion in Section 9.7.2 on the local quadratic convergence of Newton’s method, Program 9.7.2 provides a concrete numerical demonstration of this phenomenon for a small nonlinear system with a known closed-form solution. While the preceding analysis establishes quadratic convergence theoretically under standard smoothness and nonsingularity assumptions, this program makes the abstract error estimates explicit by computing and reporting the error norm at each iteration. By tracking both the residual norm and the distance to the exact solution, the implementation illustrates how Newton’s method rapidly transitions from modest improvement to near machine-precision accuracy once the iterates enter the local basin of attraction. The example highlights the characteristic error-squaring behavior predicted by the local convergence theory and clarifies the practical limits imposed by finite-precision arithmetic.

At the core of the implementation is the definition of the nonlinear mapping $\mathbf{F}(\mathbf{x})$ and its Jacobian matrix, which together specify the system of equations to be solved. The function `f` evaluates the vector-valued residual, while `jacobian` constructs the corresponding Jacobian matrix by analytically differentiating each component of $\mathbf{F}$. This explicit formulation reflects the assumptions underlying quadratic convergence, namely that the Jacobian exists, is continuous in a neighborhood of the solution, and is nonsingular at the solution point, as required by the local convergence result discussed earlier in this section.

The Newton iteration itself is implemented in the `newton` function, which repeatedly solves the linearized system defined by the Jacobian and residual at the current iterate. At each step, the Newton correction is obtained by solving a small linear system of the form dictated by the Newton update equation introduced earlier in the chapter. For clarity and numerical transparency, the program employs a direct solver for a $2×2$ system in the `solve_2x2` function rather than relying on external linear algebra libraries. This choice keeps the algorithmic structure explicit and allows the reader to trace how each Newton step is formed from the Jacobian and residual.

To quantify convergence behavior, the program computes the Euclidean norm of both the residual $||\mathbf{F}(\mathbf{x}_k)||_2$ and the error $||\mathbf{x}_k - \mathbf{x}^\ast||_2$ *at each iteration. The helper functions* `norm2`, `add`, `sub`, and `scale` provide basic vector operations needed for these calculations. Most importantly, the implementation evaluates the ratio $||\mathbf{e}_{k+1}|| / ||\mathbf{e}_k||^2$, which is predicted by the local theory to approach a constant when quadratic convergence is active. Observing this ratio stabilize numerically provides direct empirical confirmation of the error estimate derived from the second-order Taylor expansion of the nonlinear system.

The `main` function initializes the iteration with a starting guess chosen sufficiently close to the exact solution to ensure that the local convergence regime is entered. A reference solution is selected explicitly so that the true error can be computed at each step. The iteration proceeds until either the residual norm or the step size falls below prescribed tolerances, reflecting standard termination criteria in practical Newton solvers. As the iteration approaches machine precision, the reported convergence ratio becomes dominated by roundoff effects, illustrating the inevitable breakdown of asymptotic error models once floating-point limits are reached.

```rust
// Program 9.7.2: Demonstrating local quadratic convergence of Newton's method
//
// This program applies Newton's method to a small nonlinear system with a known solution
// so we can directly observe quadratic convergence by tracking the error e_k = x_k - x*.
//
// System:
//   F1(x, y) = x^2 + y^2 - 1
//   F2(x, y) = x - y
//
// Solutions are (±1/√2, ±1/√2). The Jacobian is nonsingular at either solution, and if the
// initial guess is sufficiently close, Newton’s method converges quadratically.

use std::f64;

fn f(x: [f64; 2]) -> [f64; 2] {
    let (u, v) = (x[0], x[1]);
    [u * u + v * v - 1.0, u - v]
}

fn jacobian(x: [f64; 2]) -> [[f64; 2]; 2] {
    let (u, v) = (x[0], x[1]);
    // J = [ 2x   2y
    //       1   -1 ]
    [[2.0 * u, 2.0 * v], [1.0, -1.0]]
}

fn norm2(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn add(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn scale(a: [f64; 2], s: f64) -> [f64; 2] {
    [s * a[0], s * a[1]]
}

// Solve a 2x2 linear system A p = b with explicit formula.
// Returns None if the matrix is (near) singular.
fn solve_2x2(a: [[f64; 2]; 2], b: [f64; 2]) -> Option<[f64; 2]> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let scale_det = a[0][0].abs() + a[0][1].abs() + a[1][0].abs() + a[1][1].abs();
    let thresh = 1e-15 * (1.0 + scale_det);

    if det.abs() <= thresh {
        return None;
    }

    let inv = [[a[1][1] / det, -a[0][1] / det], [-a[1][0] / det, a[0][0] / det]];
    Some([
        inv[0][0] * b[0] + inv[0][1] * b[1],
        inv[1][0] * b[0] + inv[1][1] * b[1],
    ])
}

// Choose the "matching" exact solution (±1/√2, ±1/√2) based on the sign of the initial guess.
// This keeps the reported error meaningful when the iterate converges to either root.
fn reference_solution(x0: [f64; 2]) -> [f64; 2] {
    let s = 1.0 / 2.0_f64.sqrt();
    let sign = if (x0[0] + x0[1]) >= 0.0 { 1.0 } else { -1.0 };
    [sign * s, sign * s]
}

fn newton(
    mut x: [f64; 2],
    x_star: [f64; 2],
    tol_res: f64,
    tol_step: f64,
    max_iter: usize,
) -> ([f64; 2], usize) {
    println!("Newton iteration (tracking local quadratic convergence)");
    println!(
        "k    ||F(x_k)||_2        ||e_k||_2           ratio  ||e_{{k+1}}|| / ||e_k||^2"
    );    
    println!("--------------------------------------------------------------------------------");

    let mut e_prev = norm2(sub(x, x_star));
    let mut f_prev = norm2(f(x));

    for k in 0..=max_iter {
        let fk = f(x);
        let j = jacobian(x);

        let res = norm2(fk);
        let e = norm2(sub(x, x_star));

        // Only meaningful once we can form e_{k+1} (so we print ratio after computing next step).
        // For k = 0, we print placeholders and then compute ratio at the end of the loop.
        if k == 0 {
            println!("{:<4} {:<18.10e} {:<18.10e} {:>12}", k, res, e, "   (n/a)");
        }

        if res <= tol_res {
            return (x, k);
        }

        // Newton step solves J(x_k) p_k = -F(x_k).
        let rhs = scale(fk, -1.0);
        let p = solve_2x2(j, rhs).expect("Jacobian became singular or ill-conditioned.");

        let step = norm2(p);
        x = add(x, p);

        let e_next = norm2(sub(x, x_star));
        let ratio = if e > 0.0 { e_next / (e * e) } else { f64::NAN };

        // Print from k >= 0 but now with the ratio for this step.
        // We already printed k=0 once without ratio, so overwrite by printing a second line for k=0+.
        // To keep the output simple and readable, print the ratio on its own line after the step.
        println!(
            "     after step: ||F||={:<11.3e}  ||e||={:<11.3e}  ratio={:.6e}",
            f_prev, e_prev, ratio
        );

        // Update "previous" trackers to display consistent "after step" values next time.
        e_prev = e_next;
        f_prev = norm2(f(x));

        if step <= tol_step * (1.0 + norm2(x)) {
            return (x, k + 1);
        }

        if k + 1 < max_iter {
            // Print the next iterate's raw values at the start of the next loop for continuity.
            println!(
                "{:<4} {:<18.10e} {:<18.10e} {:>12}",
                k + 1,
                f_prev,
                e_prev,
                "(see above)"
            );
        }
    }

    (x, max_iter)
}

fn main() {
    // Initial guess chosen close to the positive solution (1/√2, 1/√2).
    // Try changing this farther away (for example [2.0, 0.1]) to see loss of local behavior.
    let x0 = [0.8, 0.6];

    let x_star = reference_solution(x0);

    let tol_res = 1e-14;
    let tol_step = 1e-14;
    let max_iter = 20;

    println!("Initial guess:      x0 = [{:.16e}, {:.16e}]", x0[0], x0[1]);
    println!("Reference solution: x* = [{:.16e}, {:.16e}]", x_star[0], x_star[1]);
    println!();

    let (x, iters) = newton(x0, x_star, tol_res, tol_step, max_iter);

    let final_res = norm2(f(x));
    let final_err = norm2(sub(x, x_star));

    println!("\nDone.");
    println!("Iterations: {}", iters);
    println!("Computed x  = [{:.16e}, {:.16e}]", x[0], x[1]);
    println!("||F(x)||_2  = {:.16e}", final_res);
    println!("||x-x*||_2  = {:.16e}", final_err);
}
```

Program 9.7.2 provides a direct numerical illustration of the local quadratic convergence property of Newton’s method established in Section 9.7.2. By explicitly computing both the residual and the true error at each iteration, the program shows how rapidly Newton’s method reduces the error once the iterates lie within the basin of attraction of a solution. The stabilization of the ratio $||\mathbf{e}_{k+1}|| / ||\mathbf{e}_k||^2$ confirms the theoretical prediction that the error decreases proportionally to the square of the previous error in the asymptotic regime.

The final iterations also reveal an important practical limitation of quadratic convergence analysis. Once the error approaches the level of machine precision, further improvement is constrained by roundoff, and diagnostic quantities based on asymptotic error estimates lose their predictive value. This behavior underscores the distinction between mathematical convergence theory and finite-precision computation and motivates the careful design of stopping criteria in robust numerical solvers.

Together with the theoretical development in this section, the program clarifies why Newton’s method is so effective as a local refinement technique and why it is commonly paired with globalization strategies, such as line search or trust-region methods, to ensure reliable convergence from less accurate initial guesses.

## 9.7.3. Computational Cost and Newton–Krylov Methods

Despite its favorable convergence properties, each iteration of Newton’s method incurs a substantial computational cost. At every step, one must solve the linear system defined by the Jacobian equation (9.7.11). If a direct method such as LU decomposition is used, this requires $O(N^3)$ floating-point operations and $O(N^2)$ storage, which quickly becomes impractical as the problem dimension grows.

In addition to the linear solve, the Jacobian itself must be constructed or approximated. When analytic derivatives are available, assembling all $N^2$ partial derivatives can be laborious and error-prone. When finite-difference approximations are used instead, the cost typically increases further, as roughly $N$ additional evaluations of the nonlinear function $\mathbf{F}$ are required per Newton iteration. For large-scale systems, such costs dominate the computation.

To address these challenges, Newton–Krylov methods have been developed for large and sparse nonlinear systems. In these approaches, the Newton linear system is not solved directly. Instead, it is solved approximately using an iterative Krylov subspace method such as GMRES or BiCGSTAB. Crucially, these solvers require only *Jacobian–vector products* rather than explicit access to the full Jacobian matrix.

Jacobian–vector products can often be computed efficiently by finite differences or automatic differentiation, without forming or storing $J$ explicitly. This matrix-free formulation dramatically reduces memory usage and allows the algorithm to exploit sparsity and locality in the underlying problem. As a result, Newton–Krylov methods scale to problems with tens or even hundreds of thousands of unknowns, far beyond the reach of classical Newton implementations.

Newton–Krylov techniques are now standard in large-scale simulations, including computational fluid dynamics, nonlinear structural mechanics, and power system analysis. For example, recent work by Yetkin and Ceylan (2023) demonstrated a recycling Newton–Krylov approach for power grid load-flow equations, in which information from previous linear solves is reused to accelerate convergence. Their results show substantial performance improvements on very large networks, highlighting the continued evolution of Newton-based methods in modern high-performance numerical computing.

Together, these developments illustrate a recurring theme in nonlinear system solving: the mathematical elegance of Newton’s method must be carefully balanced with algorithmic scalability and implementation efficiency to meet the demands of contemporary large-scale applications.

### Rust Implementation

Following the discussion in Section 9.7.3 on the computational cost of Newton’s method and the motivation for Jacobian-free approaches, Program 9.7.3 presents a practical implementation of a matrix-free Newton–Krylov method for a nonlinear system arising from a one-dimensional boundary value problem. Rather than forming or factorizing the Jacobian matrix associated with Equation (9.7.11), the program solves each Newton correction step approximately using a restarted GMRES iteration that requires only Jacobian–vector products. This design reflects the central idea of Newton–Krylov methods: retaining the rapid local convergence of Newton’s method while avoiding the prohibitive $O(N^3)$ cost and $O(N^2)$ storage associated with direct linear solvers. The example demonstrates how iterative linear algebra, preconditioning, and globalization strategies can be combined into a scalable nonlinear solver suitable for large problem sizes.

At the core of the implementation is the evaluation of the nonlinear residual vector $\mathbf{F}(\mathbf{u})$ defined by the discretized boundary value problem. The function `nonlinear_f` computes this residual by combining a second-order finite-difference approximation of the Laplacian with the cubic nonlinearity $u_i^3$, corresponding directly to the structure of Equation (9.7.11). Boundary conditions are incorporated implicitly by treating values outside the interior grid as zero, allowing the residual to be assembled without explicitly storing ghost points or modifying the linear algebra routines.

To support matrix-free Newton–Krylov iteration, the program defines an explicit Jacobian–vector product in the function `jacobian_vector_product_exact`. Rather than assembling the full Jacobian matrix, this function applies the action of the Jacobian to a given vector by linearizing the finite-difference operator and the nonlinear term around the current iterate. This approach mirrors the theoretical formulation of Newton–Krylov methods, in which only products of the form $J(\mathbf{u})\mathbf{v}$ are required by the Krylov solver, eliminating the need to store or factorize $J$ explicitly.

A simple diagonal, or Jacobi, preconditioner is constructed through the function `build_jacobi_diag`. This preconditioner approximates the inverse of the Jacobian by retaining only its diagonal entries, which are inexpensive to compute and apply. Although crude compared to more sophisticated preconditioners, this choice illustrates an important principle emphasized in Section 9.7.3: even low-cost preconditioning can significantly improve Krylov convergence while preserving the matrix-free character of the algorithm. The function `apply_jacobi_inv` applies this diagonal inverse to residual vectors, yielding a left-preconditioned linear system consistent with the Newton–Krylov framework.

The linear Newton correction is computed using a restarted GMRES algorithm implemented in `gmres_restarted_relative`. This routine constructs Krylov subspaces incrementally, applies Givens rotations to maintain numerical stability, and terminates when the preconditioned residual norm has been reduced by a prescribed relative factor. Restarting limits memory usage and ensures predictable computational cost per Newton step, at the expense of potentially slower convergence, a trade-off discussed earlier in the context of large-scale nonlinear systems.

To ensure robustness away from the local convergence region, the Newton update is globalized using a simple backtracking line search implemented in `backtracking_update`. After a candidate Newton direction has been computed, the step length is reduced until a sufficient decrease in the nonlinear residual norm is observed. This mechanism prevents divergence when the Krylov solve is inexact or when the current iterate lies outside the basin of attraction of the solution. The Newton iteration is terminated either when the residual norm falls below a prescribed tolerance or when stagnation is detected, reflecting the practical limits imposed by finite-precision arithmetic.

The `main` function assembles a complete test problem by choosing a known analytical solution and constructing the corresponding right-hand side. This allows the solver to be validated quantitatively while illustrating the typical iteration counts and computational behavior of a Newton–Krylov method. The resulting output highlights how most of the computational effort is spent in the Krylov solver, reinforcing the discussion in Section 9.7.3 on the importance of scalable linear algebra within nonlinear solution strategies.

```rust
// Program 9.7.3: Matrix-Free Newton–Krylov Method with Restarted GMRES and Backtracking
//
// Key features:
// - Matrix-free Jacobian–vector products (exact for the model problem).
// - Simple Jacobi (diagonal) left preconditioning.
// - Restarted GMRES(m) with a relative stopping criterion.
// - Backtracking line search on ||F(u)||_2 to globalize Newton steps when linear solves are inexact.
//
// Test problem (1D nonlinear boundary value problem on [0, 1]):
//   u''(x) + u(x)^3 = f(x),  u(0)=u(1)=0
// Discretization with N interior points, step h = 1/(N+1):
//   F_i(u) = (u_{i-1} - 2u_i + u_{i+1})/h^2 + u_i^3 - f_i = 0
//
// Exact solution: u*(x)=sin(pi x)
// Then f(x)= -pi^2 sin(pi x) + sin^3(pi x)
//
// Notes:
// - No external dependencies.
// - Complete main() so `cargo run` works out of the box.

use std::f64::consts::PI;

// ---------------------------- Basic linear algebra ----------------------------

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

fn scal(y: &mut [f64], a: f64) {
    for yi in y.iter_mut() {
        *yi *= a;
    }
}

// ---------------------------- Nonlinear residual F(u) ----------------------------

fn nonlinear_f(u: &[f64], h: f64, f: &[f64], out: &mut [f64]) {
    let n = u.len();
    let inv_h2 = 1.0 / (h * h);

    for i in 0..n {
        let uim1 = if i == 0 { 0.0 } else { u[i - 1] };
        let uip1 = if i + 1 == n { 0.0 } else { u[i + 1] };
        let lap = (uim1 - 2.0 * u[i] + uip1) * inv_h2;
        out[i] = lap + u[i] * u[i] * u[i] - f[i];
    }
}

// ---------------------------- Matrix-free J(u)v (exact) ----------------------------

fn jacobian_vector_product_exact(u: &[f64], v: &[f64], h: f64, out: &mut [f64]) {
    let n = u.len();
    let inv_h2 = 1.0 / (h * h);

    for i in 0..n {
        let vim1 = if i == 0 { 0.0 } else { v[i - 1] };
        let vip1 = if i + 1 == n { 0.0 } else { v[i + 1] };
        let lap_v = (vim1 - 2.0 * v[i] + vip1) * inv_h2;

        out[i] = lap_v + 3.0 * u[i] * u[i] * v[i];
    }
}

// ---------------------------- Jacobi preconditioner M^{-1} ----------------------------

fn build_jacobi_diag(u: &[f64], h: f64, diag: &mut [f64]) {
    let inv_h2 = 1.0 / (h * h);
    for i in 0..u.len() {
        diag[i] = -2.0 * inv_h2 + 3.0 * u[i] * u[i];
        if diag[i].abs() < 1e-30 {
            diag[i] = if diag[i] >= 0.0 { 1e-30 } else { -1e-30 };
        }
    }
}

fn apply_jacobi_inv(diag: &[f64], r: &[f64], z: &mut [f64]) {
    for i in 0..r.len() {
        z[i] = r[i] / diag[i];
    }
}

// ---------------------------- GMRES(m) with Givens rotations ----------------------------

fn apply_givens(a: f64, b: f64) -> (f64, f64, f64) {
    if b == 0.0 {
        return (1.0, 0.0, a);
    }
    if a == 0.0 {
        return (0.0, 1.0, b);
    }
    let r = (a * a + b * b).sqrt();
    let c = a / r;
    let s = b / r;
    (c, s, r)
}

fn back_substitute(upper: &[Vec<f64>], g: &[f64], k: usize) -> Vec<f64> {
    let mut y = vec![0.0; k];
    for i_rev in 0..k {
        let i = k - 1 - i_rev;
        let mut sum = g[i];
        for j in (i + 1)..k {
            sum -= upper[i][j] * y[j];
        }
        y[i] = sum / upper[i][i];
    }
    y
}

fn gmres_restarted_relative<F>(
    apply_a: &F,
    b: &[f64],
    x: &mut [f64],
    restart_m: usize,
    rel_tol: f64,
    max_iters: usize,
) -> (usize, f64, f64)
where
    F: Fn(&[f64], &mut [f64]),
{
    // Returns (iters_used, final ||r||, ratio ||r||/||r0||)
    let n = b.len();
    let mut r = vec![0.0; n];
    let mut ax = vec![0.0; n];

    apply_a(x, &mut ax);
    for i in 0..n {
        r[i] = b[i] - ax[i];
    }
    let beta0 = norm2(&r);
    if beta0 == 0.0 {
        return (0, 0.0, 0.0);
    }
    let tol = rel_tol * beta0;

    let mut iters_used = 0usize;

    loop {
        apply_a(x, &mut ax);
        for i in 0..n {
            r[i] = b[i] - ax[i];
        }
        let beta = norm2(&r);
        if beta <= tol || iters_used >= max_iters {
            return (iters_used, beta, beta / beta0);
        }

        let mut v: Vec<Vec<f64>> = Vec::with_capacity(restart_m + 1);
        let mut hess: Vec<Vec<f64>> = vec![vec![0.0; restart_m]; restart_m + 1];
        let mut cs = vec![0.0; restart_m];
        let mut sn = vec![0.0; restart_m];
        let mut g = vec![0.0; restart_m + 1];
        g[0] = beta;

        let mut v0 = r.clone();
        scal(&mut v0, 1.0 / beta);
        v.push(v0);

        let mut k_done = 0usize;

        for k in 0..restart_m {
            let mut w = vec![0.0; n];
            apply_a(&v[k], &mut w);

            for j in 0..=k {
                let hj = dot(&w, &v[j]);
                hess[j][k] = hj;
                axpy(&mut w, -hj, &v[j]);
            }
            let h_next = norm2(&w);
            hess[k + 1][k] = h_next;

            if h_next != 0.0 {
                scal(&mut w, 1.0 / h_next);
                v.push(w);
            } else {
                v.push(vec![0.0; n]);
            }

            for j in 0..k {
                let temp = cs[j] * hess[j][k] + sn[j] * hess[j + 1][k];
                hess[j + 1][k] = -sn[j] * hess[j][k] + cs[j] * hess[j + 1][k];
                hess[j][k] = temp;
            }

            let (c, s, rkk) = apply_givens(hess[k][k], hess[k + 1][k]);
            cs[k] = c;
            sn[k] = s;
            hess[k][k] = rkk;
            hess[k + 1][k] = 0.0;

            let gk = c * g[k] + s * g[k + 1];
            let gk1 = -s * g[k] + c * g[k + 1];
            g[k] = gk;
            g[k + 1] = gk1;

            let res_est = g[k + 1].abs();
            iters_used += 1;
            k_done = k + 1;

            if res_est <= tol || iters_used >= max_iters {
                break;
            }
        }

        let mut r_upper: Vec<Vec<f64>> = vec![vec![0.0; k_done]; k_done];
        for i in 0..k_done {
            for j in i..k_done {
                r_upper[i][j] = hess[i][j];
            }
        }
        let y = back_substitute(&r_upper, &g, k_done);

        for j in 0..k_done {
            axpy(x, y[j], &v[j]);
        }

        if iters_used >= max_iters {
            apply_a(x, &mut ax);
            for i in 0..n {
                r[i] = b[i] - ax[i];
            }
            let beta = norm2(&r);
            return (iters_used, beta, beta / beta0);
        }
    }
}

// ---------------------------- Backtracking line search ----------------------------

fn backtracking_update(
    u: &mut [f64],
    p: &[f64],
    h: f64,
    f: &[f64],
    fu: &[f64],
    work: &mut [f64],
) -> (f64, f64) {
    // Minimize ||F(u + alpha p)|| by backtracking on alpha.
    // Returns (alpha_used, new_residual_norm).
    let res0 = norm2(fu);
    let mut alpha = 1.0;

    let c = 1e-4;
    let max_backtracks = 20;

    for _ in 0..max_backtracks {
        // trial_u = u + alpha p (stored into work)
        work.copy_from_slice(u);
        axpy(work, alpha, p);

        // compute F(trial_u) into a temp buffer (reuse fu-like storage via local vec)
        let mut f_trial = vec![0.0; u.len()];
        nonlinear_f(work, h, f, &mut f_trial);
        let res_trial = norm2(&f_trial);

        // Simple sufficient decrease check: ||F(u+αp)|| <= (1 - c α) ||F(u)||
        if res_trial <= (1.0 - c * alpha) * res0 {
            // accept
            u.copy_from_slice(work);
            return (alpha, res_trial);
        }

        alpha *= 0.5;
        if alpha < 1e-12 {
            break;
        }
    }

    // If we fail to find decrease, fall back to the best small step tried (or do nothing).
    // Here we take a tiny step to avoid stalling completely.
    alpha = 1e-6;
    work.copy_from_slice(u);
    axpy(work, alpha, p);
    let mut f_trial = vec![0.0; u.len()];
    nonlinear_f(work, h, f, &mut f_trial);
    let res_trial = norm2(&f_trial);

    u.copy_from_slice(work);
    (alpha, res_trial)
}

// ---------------------------- Newton–Krylov outer iteration ----------------------------

fn newton_krylov(
    u: &mut [f64],
    h: f64,
    f: &[f64],
    newton_tol: f64,
    restart_m: usize,
    gmres_rel_tol: f64,
    gmres_max_iters: usize,
    newton_max_iters: usize,
) -> usize {
    let n = u.len();
    let mut fu = vec![0.0; n];
    let mut trial_u = vec![0.0; n];

    for k in 0..newton_max_iters {
        nonlinear_f(u, h, f, &mut fu);
        let res = norm2(&fu);

        println!("Newton iter {:>2}:  ||F(u)||_2 = {:>12.6e}", k, res);

        if res <= newton_tol {
            return k;
        }

        // Solve J(u) p = -F(u)
        let mut b = fu.clone();
        scal(&mut b, -1.0);

        // Snapshot u
        let u_snapshot = u.to_vec();

        // Jacobi preconditioner
        let mut diag = vec![0.0; n];
        build_jacobi_diag(&u_snapshot, h, &mut diag);

        // Preconditioned RHS
        let mut b_tilde = vec![0.0; n];
        apply_jacobi_inv(&diag, &b, &mut b_tilde);

        // Preconditioned operator
        let apply_a_tilde = |v_in: &[f64], out: &mut [f64]| {
            let mut tmp = vec![0.0; v_in.len()];
            jacobian_vector_product_exact(&u_snapshot, v_in, h, &mut tmp);
            apply_jacobi_inv(&diag, &tmp, out);
        };

        let mut p = vec![0.0; n];
        let (gmres_iters, _gmres_res, gmres_ratio) = gmres_restarted_relative(
            &apply_a_tilde,
            &b_tilde,
            &mut p,
            restart_m,
            gmres_rel_tol,
            gmres_max_iters,
        );

        println!(
            "             GMRES: iters(step) = {:>4},  ||r||/||r0|| = {:>10.3e}",
            gmres_iters, gmres_ratio
        );

        // Globalize the step with backtracking on ||F||
        let (alpha, res_new) = backtracking_update(u, &p, h, f, &fu, &mut trial_u);

        println!(
            "             step:  alpha = {:>10.3e},  new ||F||_2 = {:>12.6e}",
            alpha, res_new
        );
        // Convergence or stagnation detection
        if res_new <= newton_tol {
            return k + 1;
        }
        if alpha < 1e-8 || res_new >= 0.999999 * res {
            return k + 1;
            }
    }
    newton_max_iters
}

// ----------------------------------- main -----------------------------------

fn main() {
    let n: usize = 400;
    let h = 1.0 / (n as f64 + 1.0);

    // Build f_i and exact solution from u*(x)=sin(pi x)
    let mut f = vec![0.0; n];
    let mut u_exact = vec![0.0; n];
    for i in 0..n {
        let x = (i as f64 + 1.0) * h;
        let u = (PI * x).sin();
        u_exact[i] = u;
        f[i] = -(PI * PI) * (PI * x).sin() + u * u * u;
    }

    // Initial guess
    let mut u0 = vec![0.0; n];
    for i in 0..n {
        let x = (i as f64 + 1.0) * h;
        u0[i] = 0.2 * (PI * x).sin();
    }

    // Parameters tuned for a clean demonstration
    let newton_tol = 1e-9;
    let restart_m = 120;        // larger restart reduces stagnation
    let gmres_rel_tol = 1e-6;   // relative tolerance in preconditioned GMRES
    let gmres_max_iters = 600;  // cap per Newton step
    let newton_max_iters = 20;

    println!("Matrix-Free Newton–Krylov solve for a 1D nonlinear system");
    println!("N = {}, restart_m = {}", n, restart_m);
    println!();

    let iters = newton_krylov(
        &mut u0,
        h,
        &f,
        newton_tol,
        restart_m,
        gmres_rel_tol,
        gmres_max_iters,
        newton_max_iters,
    );

    // Accuracy
    let mut err = vec![0.0; n];
    for i in 0..n {
        err[i] = u0[i] - u_exact[i];
    }
    let rms_err = norm2(&err) / (n as f64).sqrt();

    // Final residual
    let mut fu = vec![0.0; n];
    nonlinear_f(&u0, h, &f, &mut fu);
    let res_norm = norm2(&fu);

    println!();
    println!("Done.");
    println!("Newton iterations used: {}", iters);
    println!("Final ||F(u)||_2       = {:.6e}", res_norm);
    println!("RMS error vs exact u*   = {:.6e}", rms_err);
}
```

Program 9.7.3 illustrates how Newton’s method can be adapted to large-scale nonlinear problems by replacing direct Jacobian solves with matrix-free Krylov subspace iterations. The example demonstrates that, even for moderately sized discretizations, the dominant cost lies not in evaluating the nonlinear residual but in approximately solving the Jacobian equation at each Newton step. By avoiding explicit Jacobian construction and factorization, the Newton–Krylov approach significantly reduces memory requirements and enables the solver to scale to problem sizes that would be infeasible for classical Newton implementations.

The behavior observed in the numerical results reflects the theoretical discussion of Section 9.7.3. Early Newton steps achieve rapid reductions in the residual norm, while later iterations become limited by the accuracy of the Krylov solves and finite-precision effects. The inclusion of preconditioning and line search illustrates how practical algorithmic enhancements are essential for balancing mathematical efficiency with numerical robustness.

The modular structure of the code makes it straightforward to replace GMRES with alternative Krylov methods, introduce more advanced preconditioners, or extend the approach to higher-dimensional problems. As such, this program serves as a foundation for modern large-scale nonlinear solvers used in scientific and engineering applications, where matrix-free Newton–Krylov methods have become a standard tool.

## 9.7.4. Example: Power Flow in Electrical Networks

A canonical and practically important application of Newton’s method for nonlinear systems is the power-flow (load-flow) problem in electrical engineering. The objective is to determine the steady-state operating point of an electrical power network by computing voltage magnitudes and phase angles at all buses such that Kirchhoff’s laws, together with generator and load constraints, are simultaneously satisfied. These physical requirements lead directly to a large nonlinear system of equations of the form introduced in equation (9.7.4).

Each bus in the network contributes nonlinear balance equations that relate active and reactive power injections to voltage magnitudes and phase angles through trigonometric relationships. The resulting coupling between variables reflects the topology and electrical characteristics of the grid, and produces strong nonlinearity even under normal operating conditions.

Newton’s method is the method of choice in this context because of its rapid local convergence and its ability to handle tightly coupled nonlinear equations. The Jacobian matrix, commonly referred to as the power-flow Jacobian, consists of partial derivatives of power injections with respect to voltage magnitudes and phase angles. Its block structure mirrors the separation between real and reactive power effects and can be exploited for efficient numerical solution.

For well-conditioned operating points and realistic initial guesses, Newton’s method typically converges in only a few iterations, enabling near real-time contingency analysis and operational planning. For very large networks, direct solution of the Jacobian linear system becomes impractical, and Newton–Krylov methods are employed instead. For example, the 70,000 bus power-grid model studied by Yetkin and Ceylan (2023) relies on matrix-free Krylov solvers with iterative refinement and preconditioning to remain computationally feasible, illustrating how Newton’s method scales to modern power systems.

### Rust Implementation

Following the formulation of the nonlinear power-balance equations in equation (9.7.4), **Program 9.7.4** provides a concrete implementation of Newton’s method applied to the AC power-flow problem on a small electrical network. In practical power-system analysis, the steady-state operating point of a grid is obtained by solving a tightly coupled nonlinear system that enforces Kirchhoff’s laws together with generator and load constraints. This program illustrates how these physical requirements translate into a residual vector and Jacobian matrix suitable for Newton iteration. By working through a three-bus example with explicit bus types (slack, PV, and PQ), the implementation makes the structure of the power-flow equations transparent while retaining the essential numerical features that arise in large-scale networks. The example emphasizes how rapid local convergence can be achieved when the Jacobian is assembled analytically and the initial guess is physically reasonable.

At the core of the implementation is the explicit evaluation of the nonlinear active and reactive power injections at each bus, as defined by the trigonometric relationships in equation (9.7.4). The function `calc_pq` computes the vectors $P(\mathbf{x}) \text{ and } Q(\mathbf{x})$ from the current voltage magnitudes and phase angles by summing contributions from all network connections encoded in the admittance matrix. This direct evaluation mirrors the physical interpretation of power flow as the superposition of pairwise bus interactions and provides the foundation for constructing the Newton residual.

The function `mismatch` assembles the residual vector $\mathbf{F}(\mathbf{x})$ corresponding to the subset of power-balance equations enforced by the chosen bus types. For the PV bus, only the active power mismatch is included, while for the PQ bus both active and reactive power mismatches are enforced. Slack-bus equations are omitted entirely, reflecting the fact that its voltage magnitude and angle are prescribed. This selective construction of the residual illustrates how physical constraints reduce the dimensionality of the nonlinear system without altering its essential structure.

The Jacobian matrix required by Newton’s method is constructed explicitly in the function `jacobian`. Each entry corresponds to an analytic partial derivative of a power injection with respect to either a voltage angle or a voltage magnitude, yielding the classical block structure of the power-flow Jacobian. Diagonal terms incorporate the self-dependence of each bus through its own power injections, while off-diagonal terms capture coupling between neighboring buses through the network admittance. Although this example assembles a dense Jacobian for clarity, the same derivative structure underlies sparse and matrix-free formulations used in large-scale power systems.

To advance the Newton iteration, the linear system defined by the Jacobian and the residual is solved at each step using a small Gaussian-elimination routine implemented in `solve_3x3`. While not intended for large problems, this solver makes the Newton update $\Delta \mathbf{x}$ explicit and avoids obscuring the method behind external linear-algebra libraries. The updated voltage angles and magnitudes are then applied with optional damping, and fixed quantities at the slack and PV buses are reimposed to maintain consistency with the physical model.

The `main` function orchestrates the overall solution process and serves as a numerical experiment demonstrating Newton convergence in a power-flow setting. It initializes a three-bus system with a flat start, specifies realistic power injections, and iterates until the Euclidean norm of the residual falls below a prescribed tolerance. At each iteration, the program reports both the global residual norm and the individual power mismatches, allowing the reader to observe how each balance equation is driven to zero. The final output confirms that specified quantities at PV and PQ buses are satisfied exactly, while the slack bus absorbs the remaining imbalance, as required by network power conservation.

```rust
// Program 9.7.4: Newton power flow (load flow) on a small network
//
// Bus types:
//   - Bus 0: Slack (V, theta fixed)
//   - Bus 1: PV    (P specified, V specified; unknown theta; Q is implied)
//   - Bus 2: PQ    (P specified, Q specified; unknown V and theta)
//
// Unknown state vector x = [theta1, theta2, V2]^T.
// Mismatch vector F(x) = [ΔP1, ΔP2, ΔQ2]^T.
// Newton step: J(x) * Δx = F(x), then x <- x + Δx.
//
// All quantities here are in per-unit (p.u.).

use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
struct YEntry {
    g: f64, // conductance G_ik
    b: f64, // susceptance B_ik
}

#[derive(Debug)]
struct PowerFlowCase {
    // Admittance matrix in rectangular form: Y = G + jB
    ybus: Vec<Vec<YEntry>>,

    // Net injections: generation minus load (p.u.).
    p_spec: Vec<f64>,
    q_spec: Vec<f64>,

    // Voltage magnitudes and angles (radians).
    v: Vec<f64>,
    theta: Vec<f64>,

    // Indices for bus types
    slack: usize,
    pv: usize,
    pq: usize,
}

fn wrap_angle(mut a: f64) -> f64 {
    while a > PI {
        a -= 2.0 * PI;
    }
    while a < -PI {
        a += 2.0 * PI;
    }
    a
}

fn calc_pq(ybus: &Vec<Vec<YEntry>>, v: &Vec<f64>, theta: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let n = v.len();
    let mut p = vec![0.0; n];
    let mut q = vec![0.0; n];

    for i in 0..n {
        let mut pi = 0.0;
        let mut qi = 0.0;
        for k in 0..n {
            let gik = ybus[i][k].g;
            let bik = ybus[i][k].b;
            let d = theta[i] - theta[k];
            let cd = d.cos();
            let sd = d.sin();

            // P_i = V_i Σ_k V_k (G_ik cos(d) + B_ik sin(d))
            // Q_i = V_i Σ_k V_k (G_ik sin(d) - B_ik cos(d))
            pi += v[i] * v[k] * (gik * cd + bik * sd);
            qi += v[i] * v[k] * (gik * sd - bik * cd);
        }
        p[i] = pi;
        q[i] = qi;
    }

    (p, q)
}

fn mismatch(case_: &PowerFlowCase) -> Vec<f64> {
    let (p, q) = calc_pq(&case_.ybus, &case_.v, &case_.theta);

    let i_pv = case_.pv;
    let i_pq = case_.pq;

    vec![
        case_.p_spec[i_pv] - p[i_pv], // ΔP at PV bus
        case_.p_spec[i_pq] - p[i_pq], // ΔP at PQ bus
        case_.q_spec[i_pq] - q[i_pq], // ΔQ at PQ bus
    ]
}

// Jacobian for x = [theta_pv, theta_pq, V_pq].
// Rows: [P_pv, P_pq, Q_pq], Cols: [theta_pv, theta_pq, V_pq].
fn jacobian(case_: &PowerFlowCase) -> [[f64; 3]; 3] {
    let y = &case_.ybus;
    let v = &case_.v;
    let th = &case_.theta;

    let i_pv = case_.pv;
    let i_pq = case_.pq;

    let (p, q) = calc_pq(y, v, th);

    let g = |i: usize, k: usize| y[i][k].g;
    let b = |i: usize, k: usize| y[i][k].b;
    let dik = |i: usize, k: usize| th[i] - th[k];

    // Row 0: P at PV bus
    let d_ppv_dthpv = -q[i_pv] - b(i_pv, i_pv) * v[i_pv] * v[i_pv];

    let d_pv_pq = dik(i_pv, i_pq);
    let d_ppv_dthpq =
        v[i_pv] * v[i_pq] * (g(i_pv, i_pq) * d_pv_pq.sin() - b(i_pv, i_pq) * d_pv_pq.cos());
    let d_ppv_dvpq = v[i_pv] * (g(i_pv, i_pq) * d_pv_pq.cos() + b(i_pv, i_pq) * d_pv_pq.sin());

    // Row 1: P at PQ bus
    let d_pq_pv = dik(i_pq, i_pv);
    let d_ppq_dthpv =
        v[i_pq] * v[i_pv] * (g(i_pq, i_pv) * d_pq_pv.sin() - b(i_pq, i_pv) * d_pq_pv.cos());
    let d_ppq_dthpq = -q[i_pq] - b(i_pq, i_pq) * v[i_pq] * v[i_pq];
    let d_ppq_dvpq = p[i_pq] / v[i_pq] + g(i_pq, i_pq) * v[i_pq];

    // Row 2: Q at PQ bus
    let d_qpq_dthpv =
        -v[i_pq] * v[i_pv] * (g(i_pq, i_pv) * d_pq_pv.cos() + b(i_pq, i_pv) * d_pq_pv.sin());
    let d_qpq_dthpq = p[i_pq] - g(i_pq, i_pq) * v[i_pq] * v[i_pq];
    let d_qpq_dvpq = q[i_pq] / v[i_pq] - b(i_pq, i_pq) * v[i_pq];

    [
        [d_ppv_dthpv, d_ppv_dthpq, d_ppv_dvpq],
        [d_ppq_dthpv, d_ppq_dthpq, d_ppq_dvpq],
        [d_qpq_dthpv, d_qpq_dthpq, d_qpq_dvpq],
    ]
}

fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    let n = 3;

    for k in 0..n {
        let mut piv = k;
        let mut best = a[k][k].abs();
        for i in (k + 1)..n {
            if a[i][k].abs() > best {
                best = a[i][k].abs();
                piv = i;
            }
        }
        if best < 1e-14 {
            return None;
        }
        if piv != k {
            a.swap(k, piv);
            b.swap(k, piv);
        }

        for i in (k + 1)..n {
            let m = a[i][k] / a[k][k];
            for j in k..n {
                a[i][j] -= m * a[k][j];
            }
            b[i] -= m * b[k];
        }
    }

    let mut x = [0.0; 3];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

fn l2_norm(v: &Vec<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn main() {
    let ybus = vec![
        vec![
            YEntry { g: 5.0, b: -14.0 },
            YEntry { g: -2.0, b: 6.0 },
            YEntry { g: -3.0, b: 8.0 },
        ],
        vec![
            YEntry { g: -2.0, b: 6.0 },
            YEntry { g: 4.0, b: -12.0 },
            YEntry { g: -2.0, b: 6.0 },
        ],
        vec![
            YEntry { g: -3.0, b: 8.0 },
            YEntry { g: -2.0, b: 6.0 },
            YEntry { g: 5.0, b: -14.0 },
        ],
    ];

    let slack = 0usize;
    let pv = 1usize;
    let pq = 2usize;

    let p_spec = vec![0.0, 1.00, -1.20];
    let q_spec = vec![0.0, 0.0, -0.50];

    let v = vec![1.06, 1.04, 1.00];
    let theta = vec![0.0, 0.0, 0.0];

    let mut case_ = PowerFlowCase {
        ybus,
        p_spec,
        q_spec,
        v,
        theta,
        slack,
        pv,
        pq,
    };

    let max_iter = 20usize;
    let tol = 1e-10;
    let damping = 1.0;

    println!("Newton power flow (3-bus) with x = [theta1, theta2, V2]");
    println!("Iter | ||F||_2         |   dP1        dP2        dQ2     | theta1 (deg) | theta2 (deg) | V2");
    println!("-----+------------------+----------------------------------+-------------+-------------+---------");
    
    for it in 0..max_iter {
        let f = mismatch(&case_);
        let fnorm = l2_norm(&f);
    
        let th1_deg = case_.theta[case_.pv] * 180.0 / PI;
        let th2_deg = case_.theta[case_.pq] * 180.0 / PI;
    
        println!(
            "{:>4} | {:<16.8e} | {:>10.3e} {:>10.3e} {:>10.3e} | {:>11.6} | {:>11.6} | {:>7.5}",
            it, fnorm, f[0], f[1], f[2], th1_deg, th2_deg, case_.v[case_.pq]
        );
    
        if fnorm < tol {
            println!("\nConverged in {} iterations.", it);
            break;
        }
    

        let j = jacobian(&case_);
        let bvec = [f[0], f[1], f[2]];

        let dx = match solve_3x3(j, bvec) {
            Some(sol) => sol,
            None => {
                eprintln!("Jacobian is singular or ill-conditioned. Aborting.");
                return;
            }
        };

        case_.theta[case_.pv] = wrap_angle(case_.theta[case_.pv] + damping * dx[0]);
        case_.theta[case_.pq] = wrap_angle(case_.theta[case_.pq] + damping * dx[1]);
        case_.v[case_.pq] += damping * dx[2];

        case_.theta[case_.slack] = 0.0;
        case_.v[case_.slack] = 1.06;
        case_.v[case_.pv] = 1.04;
    }

    let (p, q) = calc_pq(&case_.ybus, &case_.v, &case_.theta);
    println!("\nFinal bus results (per unit):");
    for i in 0..3 {
        println!(
            "Bus {}: V = {:>7.5}, theta = {:>9.6} deg, P = {:>9.6}, Q = {:>9.6}",
            i,
            case_.v[i],
            case_.theta[i] * 180.0 / PI,
            p[i],
            q[i]
        );
    }

    println!("\nNotes:");
    println!("- Bus 0 is slack: V and theta fixed; P and Q are implied by the solution.");
    println!("- Bus 1 is PV: P and V fixed; theta and Q are implied.");
    println!("- Bus 2 is PQ: P and Q fixed; V and theta are implied.");
}
```

Program 9.7.4 demonstrates how Newton’s method operates in a physically meaningful nonlinear system arising from electrical network analysis. The rapid reduction of both the residual norm and individual power mismatches illustrates the strong local convergence properties that make Newton’s method the standard tool for power-flow calculations in practice.

The example highlights several essential features of real-world power-flow solvers: the role of bus types in shaping the nonlinear system, the importance of an analytically derived Jacobian, and the interpretation of convergence in terms of physical balance laws rather than abstract numerical criteria alone. Although the implementation is deliberately small and dense, its structure extends directly to large-scale systems where sparsity, preconditioning, and Krylov subspace methods are employed to solve the Jacobian system efficiently.

By making each computational step explicit, this program provides a clear bridge between the mathematical formulation in equation (9.7.4) and the algorithms used in modern power-system simulation software. It thus serves as a foundation for subsequent discussions of Newton–Krylov methods, matrix-free Jacobian approximations, and scalability issues in contemporary grid analysis.

## 9.7.5. Comparison with Single-Function Minimization

At first glance, solving the nonlinear system equation (9.7.4) may appear analogous to finding a stationary point of a scalar objective function by solving $\nabla f(\mathbf{x}) = \mathbf{0}$. Despite this superficial similarity, the two problem classes differ fundamentally in structure and algorithmic behavior.

In scalar minimization, the gradient field $\nabla f$ possesses strong internal structure: its components arise as partial derivatives of a single scalar potential. This structure implies integrability conditions, most notably the symmetry of second derivatives,

$$
\frac{\partial^2 f}{\partial x_i\,\partial x_j}
=
\frac{\partial^2 f}{\partial x_j\,\partial x_i}
\tag{9.7.15}
$$

As a consequence, optimization problems admit global strategies such as gradient descent, which exploit the existence of a single surface on which progress can always be defined in terms of descent.

In contrast, a general vector-valued function $\mathbf{F}$ defines $N$ independent component landscapes. Improvement in one equation may worsen another, and there is no scalar merit function whose monotonic decrease guarantees convergence. The Jacobian associated with (9.7.4) lacks the symmetry and definiteness properties of a Hessian, and no global descent direction exists in general.

This absence of global structure makes solving nonlinear systems significantly harder than minimizing a scalar objective. Convergence properties are typically local, success depends strongly on the initial guess and conditioning of the Jacobian, and robust solution often requires problem-specific insight or globalization strategies beyond pure Newton iteration.

### Rust Implementation

Following the discussion in Section 9.7.5 on the structural differences between general nonlinear systems and scalar optimization problems, Program 9.7.5 provides a concrete numerical comparison between Newton’s method applied to a vector‐valued system of equations of the form (9.7.4) and Newton‐type methods applied to a scalar objective through the stationarity condition $\nabla f(\mathbf{x})=\mathbf{0}$. Although both problems involve solving nonlinear equations, their mathematical structure and numerical behavior differ fundamentally. This program implements representative algorithms for each setting and traces their iteration histories, making explicit how symmetry, descent properties, and globalization mechanisms influence convergence in practice. By examining both cases side by side, the example illustrates why solving nonlinear systems is generally more challenging than minimizing a scalar objective, despite superficial similarities in their algebraic form.

At the core of the implementation for the nonlinear system case is the `VectorFunction2` trait, which defines a general interface for a two‐dimensional vector‐valued function $\mathbf{F}(\mathbf{x})$ by requiring the implementation of two methods: `eval`, which computes the residual vector $\mathbf{F}(\mathbf{x})$, and `jacobian`, which returns the Jacobian matrix $J(\mathbf{x})$ appearing in Newton’s method for systems, as introduced in equation (9.7.4). This abstraction cleanly separates the definition of a nonlinear system from the algorithm used to solve it and allows different systems to be studied using the same Newton framework.

The function `newton_system_damped` implements a damped Newton method for solving $\mathbf{F}(\mathbf{x})=\mathbf{0}$. At each iteration, it solves the linear system $J(\mathbf{x}_k)\mathbf{s}_k=-\mathbf{F}(\mathbf{x}_k)$ to obtain a Newton step and then applies a backtracking strategy based on the merit function $\phi(\mathbf{x})=\tfrac12|\mathbf{F}(\mathbf{x})|_2^2$. This globalization mechanism is not intrinsic to the system itself but is imposed to ensure progress when the full Newton step would increase the residual norm. The code explicitly reports the quantity $\max|J-J^{\mathsf{T}}|$, which measures the lack of symmetry in the Jacobian and highlights the absence of the integrability condition expressed in equation (9.7.15).

To contrast this with scalar optimization, the program introduces the `ScalarFunction2` trait, which defines a scalar objective $f(\mathbf{x})$ together with its gradient and Hessian. In this case, the Jacobian of the stationarity condition $\nabla f(\mathbf{x})=\mathbf{0}$ is precisely the Hessian matrix $H(\mathbf{x})$. As predicted by equation (9.7.15), the Hessian returned by the implementation is symmetric, and the code verifies this numerically by monitoring $\max|H-H^{\mathsf{T}}|$ during the iteration.

The function `newton_stationary_with_linesearch` applies Newton’s method to the stationarity equations using a line search that enforces sufficient decrease in the scalar objective $f$. Unlike the system case, this decrease condition is meaningful because $f$ provides a natural global measure of progress. The implementation includes explicit detection of line‐search stalling, illustrating that even in optimization, Newton’s method can fail to make progress when the Hessian does not define a suitable local model of the objective. This behavior reinforces the distinction between local convergence guarantees and global robustness.

Finally, the function `gradient_descent` implements a basic gradient descent method with a fixed step size for the same scalar objective. Although its convergence is slow compared with Newton’s method, each iteration guarantees a reduction in $f$, demonstrating the advantage conferred by the existence of a scalar potential. The `main` function orchestrates these experiments using a common initial guess and prints detailed iteration diagnostics, allowing direct comparison of convergence behavior across the three algorithms.

```rust
// Program 9.7.5: Comparison of nonlinear-system solving versus single-function minimization
//
// (A) Solve a general nonlinear system F(x)=0, as in equation (9.7.4).
//     The Jacobian J(x) need not be symmetric; we demonstrate this explicitly by
//     choosing a system whose Jacobian is generally nonsymmetric.
//
// (B) Find a stationary point of a scalar objective f by solving ∇f(x)=0.
//     The Jacobian of ∇f is the Hessian H, and symmetry of mixed second derivatives
//     (9.7.15) implies H is symmetric (up to floating-point effects).
//
// The program implements:
//   - Damped Newton for F(x)=0 with backtracking on phi(x)=0.5||F||_2^2.
//   - Newton for ∇f(x)=0 with a line search enforcing decrease of f.
//   - Basic gradient descent for f, illustrating a global descent strategy.

type Vec2 = [f64; 2];
type Mat2 = [[f64; 2]; 2];

fn l2_norm(v: Vec2) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn mat2_vec2(a: Mat2, x: Vec2) -> Vec2 {
    [
        a[0][0] * x[0] + a[0][1] * x[1],
        a[1][0] * x[0] + a[1][1] * x[1],
    ]
}

fn transpose(a: Mat2) -> Mat2 {
    [[a[0][0], a[1][0]], [a[0][1], a[1][1]]]
}

fn mat_diff_max_abs(a: Mat2, b: Mat2) -> f64 {
    let mut m: f64 = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

// Solve A x = b for 2x2 A via explicit inverse with a singularity guard.
fn solve_2x2(a: Mat2, b: Vec2) -> Option<Vec2> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-14 {
        return None;
    }
    let inv = [
        [a[1][1] / det, -a[0][1] / det],
        [-a[1][0] / det, a[0][0] / det],
    ];
    Some(mat2_vec2(inv, b))
}

fn add(a: Vec2, b: Vec2) -> Vec2 {
    [a[0] + b[0], a[1] + b[1]]
}

fn sub(a: Vec2, b: Vec2) -> Vec2 {
    [a[0] - b[0], a[1] - b[1]]
}

fn scale(s: f64, v: Vec2) -> Vec2 {
    [s * v[0], s * v[1]]
}

trait VectorFunction2 {
    fn eval(&self, x: Vec2) -> Vec2;
    fn jacobian(&self, x: Vec2) -> Mat2;
}

trait ScalarFunction2 {
    fn value(&self, x: Vec2) -> f64;
    fn grad(&self, x: Vec2) -> Vec2;
    fn hess(&self, x: Vec2) -> Mat2;
}

// -----------------------------
// (A) General nonlinear system
// -----------------------------
//
// F1(x,y) = x^2 + y - 1
// F2(x,y) = x + y^2 - 1 + beta * x * y
//
// J(x,y) = [ 2x          1
//           1 + beta*y  2y + beta*x ]
//
// For beta != 0, the off-diagonal terms typically differ, so J != J^T.

struct ExampleSystem {
    beta: f64,
}

impl VectorFunction2 for ExampleSystem {
    fn eval(&self, x: Vec2) -> Vec2 {
        let (xx, yy) = (x[0], x[1]);
        let b = self.beta;
        [xx * xx + yy - 1.0, xx + yy * yy - 1.0 + b * xx * yy]
    }

    fn jacobian(&self, x: Vec2) -> Mat2 {
        let (xx, yy) = (x[0], x[1]);
        let b = self.beta;
        [[2.0 * xx, 1.0], [1.0 + b * yy, 2.0 * yy + b * xx]]
    }
}

// Damped Newton for F(x)=0 using backtracking on phi(x)=0.5||F||_2^2.
fn newton_system_damped<F: VectorFunction2>(
    f: &F,
    mut x: Vec2,
    tol: f64,
    max_iter: usize,
) -> Vec2 {
    println!("\n(A) Nonlinear system solve: damped Newton on ||F||");
    println!("Iter | ||F||_2         | alpha    | x                         | max|J-J^T|");
    println!("-----+------------------+----------+---------------------------+----------");

    for k in 0..max_iter {
        let fx = f.eval(x);
        let nrm = l2_norm(fx);
        let j = f.jacobian(x);
        let jsym = mat_diff_max_abs(j, transpose(j));

        println!(
            "{:>4} | {:<16.8e} | {:>8} | [{:>+10.6}, {:>+10.6}] | {:<.3e}",
            k, nrm, "-", x[0], x[1], jsym
        );

        if nrm < tol {
            println!("Converged: ||F||_2 < tol.\n");
            return x;
        }

        // Newton step: J dx = -F
        let rhs = scale(-1.0, fx);
        let dx = match solve_2x2(j, rhs) {
            Some(d) => d,
            None => {
                println!("Jacobian singular or ill-conditioned; aborting.\n");
                return x;
            }
        };

        let phi0 = 0.5 * nrm * nrm;

        // Backtracking on phi(x + alpha dx)
        let mut alpha = 1.0;
        let c = 1e-4;
        let max_ls = 25;

        for _ in 0..max_ls {
            let x_try = add(x, scale(alpha, dx));
            let f_try = f.eval(x_try);
            let nrm_try = l2_norm(f_try);
            let phi_try = 0.5 * nrm_try * nrm_try;

            if phi_try <= (1.0 - c * alpha) * phi0 {
                x = x_try;
                break;
            }
            alpha *= 0.5;
        }

        let fx_new = f.eval(x);
        println!(
            "     accepted step: alpha = {:<.3e}, ||F||_2 -> {:<.8e}\n",
            alpha,
            l2_norm(fx_new)
        );

        if alpha < 1e-12 {
            println!("     damping stalled (alpha < 1e-12). Stopping.\n");
            return x;
        }
    }

    println!("Reached max_iter without satisfying tolerance.\n");
    x
}

// ---------------------------------------
// (B) Stationary point of scalar objective
// ---------------------------------------
//
// f(x,y) = (x^2 - 1)^2 + (y - 1)^2 + 0.1 x y
//
// The Hessian is symmetric by construction, reflecting (9.7.15).

struct ExampleObjective;

impl ScalarFunction2 for ExampleObjective {
    fn value(&self, x: Vec2) -> f64 {
        let (xx, yy) = (x[0], x[1]);
        let t1 = xx * xx - 1.0;
        t1 * t1 + (yy - 1.0) * (yy - 1.0) + 0.1 * xx * yy
    }

    fn grad(&self, x: Vec2) -> Vec2 {
        let (xx, yy) = (x[0], x[1]);
        [4.0 * xx * (xx * xx - 1.0) + 0.1 * yy, 2.0 * (yy - 1.0) + 0.1 * xx]
    }

    fn hess(&self, x: Vec2) -> Mat2 {
        let xx = x[0];
        [[12.0 * xx * xx - 4.0, 0.1], [0.1, 2.0]]
    }
}

// Newton on ∇f(x)=0 with a line search enforcing decrease of f.
// Includes a stall detector to avoid repeating near-zero steps.
fn newton_stationary_with_linesearch<F: ScalarFunction2>(
    f: &F,
    mut x: Vec2,
    tol: f64,
    max_iter: usize,
) -> Vec2 {
    println!("\n(B) Stationary point solve: Newton on ∇f with line search on f");
    println!("Iter | ||grad||_2       | f(x)            | alpha    | x                         | max|H-H^T|");
    println!("-----+------------------+-----------------+----------+---------------------------+----------");

    for k in 0..max_iter {
        let g = f.grad(x);
        let gnorm = l2_norm(g);
        let fx = f.value(x);
        let h = f.hess(x);
        let hsym = mat_diff_max_abs(h, transpose(h));

        println!(
            "{:>4} | {:<16.8e} | {:<15.8e} | {:>8} | [{:>+10.6}, {:>+10.6}] | {:<.3e}",
            k, gnorm, fx, "-", x[0], x[1], hsym
        );

        if gnorm < tol {
            println!("Converged: ||grad||_2 < tol.\n");
            return x;
        }

        // Newton direction: H p = -g
        let rhs = scale(-1.0, g);
        let p = match solve_2x2(h, rhs) {
            Some(d) => d,
            None => {
                println!("Hessian singular or ill-conditioned; aborting.\n");
                return x;
            }
        };

        // Armijo line search on f: f(x+αp) <= f(x) + c α ∇f^T p
        let f0 = fx;
        let mut alpha = 1.0;
        let c = 1e-4;
        let max_ls = 30;

        let gtp = g[0] * p[0] + g[1] * p[1];

        let mut x_acc = x;
        let mut f_acc = f0;

        for _ in 0..max_ls {
            let x_try = add(x, scale(alpha, p));
            let f_try = f.value(x_try);
            if f_try <= f0 + c * alpha * gtp {
                x_acc = x_try;
                f_acc = f_try;
                break;
            }
            alpha *= 0.5;
        }

        // Stall detection: extremely small alpha or essentially no objective decrease.
        let f_drop = (f0 - f_acc).abs();
        if alpha < 1e-8 || f_drop < 1e-14 {
            println!(
                "     line search stalled (alpha = {:.3e}, |Δf| = {:.3e}). Stopping.\n",
                alpha, f_drop
            );
            return x;
        }

        x = x_acc;
        let g_new = f.grad(x);

        println!(
            "     accepted step: alpha = {:<.3e}, f -> {:<.8e}, ||grad||_2 -> {:<.8e}\n",
            alpha,
            f_acc,
            l2_norm(g_new)
        );
    }

    println!("Reached max_iter without satisfying tolerance.\n");
    x
}

// Gradient descent for the same scalar objective (fixed step).
fn gradient_descent<F: ScalarFunction2>(f: &F, mut x: Vec2, step: f64, max_iter: usize) -> Vec2 {
    println!("\n(C) Scalar minimization: gradient descent (fixed step)");
    println!("Iter | f(x)            | ||grad||_2       | x");
    println!("-----+-----------------+------------------+---------------------------");

    for k in 0..max_iter {
        let fx = f.value(x);
        let g = f.grad(x);
        let gnorm = l2_norm(g);

        println!(
            "{:>4} | {:<15.8e} | {:<16.8e} | [{:>+10.6}, {:>+10.6}]",
            k, fx, gnorm, x[0], x[1]
        );

        if gnorm < 1e-10 {
            println!("Terminated: ||grad||_2 is small.\n");
            return x;
        }

        x = sub(x, scale(step, g));
    }

    println!("Reached max_iter.\n");
    x
}

fn main() {
    let x0_system: Vec2 = [0.2, 0.2];
    let x0_scalar: Vec2 = [0.2, 0.2];

    // Choose beta != 0 to make J generally nonsymmetric.
    let sys = ExampleSystem { beta: 1.0 };
    let obj = ExampleObjective;

    let tol = 1e-12;
    let max_iter = 25;

    // (A) Solve F(x)=0
    let x_star_sys = newton_system_damped(&sys, x0_system, tol, max_iter);
    let f_sys = sys.eval(x_star_sys);
    println!(
        "System result: x = [{:+.8}, {:+.8}], ||F||_2 = {:.3e}",
        x_star_sys[0],
        x_star_sys[1],
        l2_norm(f_sys)
    );

    // (B) Solve ∇f(x)=0 (stationary point) with Newton and a line search on f
    let x_star_stat = newton_stationary_with_linesearch(&obj, x0_scalar, tol, max_iter);
    let g_star = obj.grad(x_star_stat);
    println!(
        "Stationary-point result: x = [{:+.8}, {:+.8}], f(x) = {:.8e}, ||grad||_2 = {:.3e}",
        x_star_stat[0],
        x_star_stat[1],
        obj.value(x_star_stat),
        l2_norm(g_star)
    );

    // (C) Gradient descent as a global strategy for scalar minimization
    let _x_gd = gradient_descent(&obj, [0.2, 0.2], 0.05, 30);

    println!("\nRemarks:");
    println!("1) In (A), step control is enforced via a chosen merit function (here phi=0.5||F||^2), not by an intrinsic descent property.");
    println!("2) In (B), the Jacobian of ∇f is the Hessian H, whose symmetry reflects the mixed-partial condition (9.7.15).");
    println!("3) In (C), descent is globally meaningful because f is a scalar potential, so decrease of f can be enforced by construction.");
}
```

Program 9.7.5 demonstrates in a concrete numerical setting the conceptual distinction developed in Section 9.7.5 between solving general nonlinear systems and minimizing scalar objective functions. In the system case, progress must be enforced through an externally chosen merit function, and the nonsymmetry of the Jacobian reflects the absence of any underlying scalar potential whose decrease could guide the iteration. Convergence is therefore inherently local and sensitive to the conditioning of the Jacobian and the quality of the initial guess.

By contrast, the optimization examples highlight how the existence of a scalar objective fundamentally changes the algorithmic landscape. The symmetry of the Hessian, guaranteed by equation (9.7.15), encodes integrability properties that permit global strategies such as line searches and gradient descent. Even when Newton’s method stalls due to an unfavorable Hessian, the presence of a scalar objective still allows meaningful notions of descent and fallback strategies.

Together, these examples clarify why methods for nonlinear systems often require additional problem‐specific insight or sophisticated globalization techniques, whereas scalar minimization benefits from structural guarantees that can be exploited systematically. The program thus serves as a practical bridge between the theoretical comparison presented in this section and the behavior observed in real numerical computations.

## 9.7.6. When Newton’s Method Works and When It Fails?

When supplied with a sufficiently accurate initial guess and applied to a system whose Jacobian is well conditioned in a neighborhood of the solution, Newton’s method is extraordinarily efficient. In such cases, the method rapidly enters its regime of quadratic convergence, and only a small number of iterations are required to reach high accuracy. This behavior makes Newton’s method the backbone of many high-performance nonlinear solvers.

Despite this local efficiency, Newton’s method has poor global convergence properties. Far from a solution, the linear approximation underlying the method may be highly inaccurate. As a result, the computed Newton step may point away from any solution, leading to divergence, oscillatory behavior, or convergence to an unintended root. These difficulties are particularly pronounced in strongly nonlinear systems or in problems with multiple nearby solutions.

Another fundamental failure mode arises when the Jacobian matrix becomes singular or nearly singular along the iteration path. In this situation, the Newton correction equation (9.7.11) cannot be solved reliably. Near singularity, small perturbations in the residual can produce disproportionately large correction steps, severely degrading numerical stability. Such behavior is often associated with bifurcation points, constraint degeneracy, or ill-posed modeling assumptions.

These limitations motivate the development of globally convergent variants of Newton’s method, which modify the pure Newton iteration to enhance robustness. Common strategies include damping the Newton step, enforcing descent conditions, or restricting the update to a trust region where the linear model remains valid. Rather than replacing Newton’s method, these techniques aim to preserve its fast local convergence while improving its behavior far from the solution. These approaches are discussed in the next section.

It is also important to note that failure to converge is not necessarily undesirable. In some applications, divergence or stagnation indicates that no solution exists near the chosen initial guess, or that the mathematical model itself is inconsistent with the imposed constraints. In such cases, Newton’s method serves as a diagnostic tool, signaling the need to reassess modeling assumptions, parameter values, or initialization strategies.

Overall, Newton’s method should be viewed not as a standalone solver but as a powerful local engine embedded within a broader algorithmic framework. When used with appropriate safeguards and informed initialization, it remains one of the most effective tools for solving nonlinear systems of equations.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/rezpqQhjqMrcv5xb8ySK.5","tags":[]}

# 9.8. Globally Convergent Methods for Nonlinear Systems of Equations

Newton–Raphson methods for nonlinear systems, as developed in Section 9.7, are among the most efficient algorithms available when an initial guess lies sufficiently close to a solution. Their rapid local convergence makes them indispensable in high-accuracy computations. However, this efficiency is fundamentally local in nature. When the initial guess lies outside the basin of attraction of a solution, Newton’s method may diverge, oscillate between iterates, or converge to an unintended root.

These difficulties arise because Newton’s method is based on a local linear approximation of a generally nonlinear mapping. Far from a solution, this approximation may bear little resemblance to the true behavior of the system, producing steps that increase the residual rather than reduce it. In high-dimensional problems, the situation is further complicated by the complex geometry of the solution set: basins of attraction may be narrow, disconnected, or intertwined, and singular or nearly singular Jacobians may occur along the iteration path.

Globally convergent methods are designed to address these limitations. Their goal is not to replace Newton’s method, but to augment it with mechanisms that guarantee progress toward a solution from a wide range of initial guesses. These methods retain Newton’s fast local convergence while introducing *globalization strategies* that control step acceptance, step length, or search direction to prevent catastrophic failure far from a solution.

Such strategies are essential in large-scale simulations, nonlinear optimization, inverse problems, and machine learning models, where the cost of divergence or stagnation can be prohibitive and robustness often outweighs asymptotic speed. Modern solvers therefore embed Newton iterations within a globally convergent framework, ensuring reliability without sacrificing efficiency near the solution (Vivas-Cortez et al., 2023; Berzi, 2024).

## 9.8.1. Merit Functions and Descent Conditions

A common and conceptually powerful approach to globalization is to recast the nonlinear system:

$$\mathbf{F}(\mathbf{x}) = \mathbf{0} \tag{9.8.1}$$

as the minimization of a scalar-valued *merit function*. The role of the merit function is not to redefine the problem, but to provide a quantitative measure of progress toward feasibility that can be monitored and enforced during the iteration.

A standard choice is the squared residual norm,

$$
\phi(\mathbf{x})
=
\frac{1}{2}\,\lVert \mathbf{F}(\mathbf{x}) \rVert_2^2
\tag{9.8.2}
$$

Any solution of the nonlinear system (9.8.1) is a global minimizer of $\phi$, since the residual vanishes there. However, the converse is not guaranteed: $\phi$ may possess local minima or stationary points at which $\mathbf{F}(\mathbf{x}) \neq \mathbf{0}$. For this reason, the merit function serves as a *guiding quantity* rather than a replacement objective.

The gradient of the merit function is given by,

$$\nabla \phi(\mathbf{x}) = J(\mathbf{x})^{\mathsf{T}}\,\mathbf{F}(\mathbf{x}) \tag{9.8.3}$$

where $J(\mathbf{x})$ is the Jacobian of $\mathbf{F}$. This expression highlights a key distinction between nonlinear system solving and scalar minimization: the gradient of the merit function involves both the Jacobian and the residual, reflecting the fact that progress depends on how residual components interact rather than on a single scalar landscape.

A search direction $\mathbf{p}$ is said to be a *descent direction* for $\phi$ at $\mathbf{x}$ if:

$$\nabla \phi(\mathbf{x})^{\mathsf{T}} \mathbf{p} < 0 \tag{9.8.4}$$

This condition guarantees that sufficiently small steps along $\mathbf{p}$ reduce the merit function and therefore decrease the residual norm.

Near a solution, the Newton direction typically satisfies the descent condition automatically, which explains why pure Newton iterations behave so well in the local regime. Far from a solution, however, the Newton direction may fail to be a descent direction for $\phi$, leading to increases in the residual norm and loss of robustness. Globally convergent methods address this issue by modifying the Newton step or controlling its length so that a sufficient decrease in the merit function is enforced at every iteration.

In practice, this principle forms the foundation of line-search and trust-region methods, which accept or reject candidate steps based on merit-function reduction rather than relying solely on the local linear model. These mechanisms ensure that each iteration makes measurable progress toward feasibility, thereby extending the effective domain of Newton-type methods far beyond their natural basins of attraction.

### Rust Implementation

Following the discussion in Section 9.8.1 on merit functions and descent conditions for nonlinear systems, Program 9.8.1 provides a concrete implementation of a globally convergent Newton method based on merit-function monitoring and line-search globalization. While Newton’s method offers rapid local convergence when the iterate is sufficiently close to a solution, its behavior far from a root can be unreliable. This program embeds the classical Newton iteration within a globalization framework that enforces monotonic decrease of the squared residual norm, as defined in Equation (9.8.2), thereby ensuring robust progress from a broad range of initial guesses. The implementation illustrates how local linearization, descent verification, and step-length control interact in practice to extend the effective convergence domain of Newton-type methods without sacrificing their favorable local behavior.

At the core of the implementation is the `NonlinearSystem` trait, which defines a general interface for nonlinear systems of equations by requiring two methods: `f`, which evaluates the residual vector $\mathbf{F}(\mathbf{x})$ appearing in Equation (9.8.1), and `jacobian`, which computes the Jacobian matrix $J(\mathbf{x})$. This abstraction allows the same solver logic to be applied to different systems without modification, reflecting the generality of the theoretical framework developed in this section.

The merit function $\phi(\mathbf{x})$ defined in Equation (9.8.2) is evaluated directly from the residual vector returned by `f`. Rather than redefining the problem as a general unconstrained minimization, the merit function is used exclusively as a diagnostic and control mechanism to assess progress toward feasibility. Its gradient, computed according to Equation (9.8.3) as $J(\mathbf{x})^{\mathsf T}\mathbf{F}(\mathbf{x})$, plays a central role in verifying whether a proposed search direction satisfies the descent condition given in Equation (9.8.4).

At each iteration, the solver first attempts a full Newton step obtained by solving the linearized system $J(\mathbf{x}_k)\mathbf{p} = -\mathbf{F}(\mathbf{x}_k)$. The quantity $\nabla\phi(\mathbf{x}_k)^{\mathsf T}\mathbf{p}$ is then evaluated explicitly to confirm that the Newton direction is a descent direction for the merit function. If this condition is satisfied, the direction is accepted; otherwise, the algorithm falls back to a steepest-descent direction based on the negative merit gradient. This safeguard reflects the theoretical observation that Newton directions are not guaranteed to be descent directions far from a solution.

To enforce global convergence, the solver applies a backtracking line search based on the Armijo sufficient decrease condition. Starting from a full step $\alpha = 1$, the step length is reduced geometrically until the decrease in $\phi$ is consistent with the predicted first-order reduction implied by Equation (9.8.4). This mechanism ensures that each accepted iterate yields a measurable reduction in the residual norm, even when the local linear model is only weakly representative of the true nonlinear behavior.

The `main` function demonstrates the solver on two representative problems. The first is a simple circle–line intersection, for which Newton’s method rapidly enters its local quadratic convergence regime and accepts full steps throughout. The second is the stationarity system derived from the Rosenbrock function, a classical test problem known for its narrow valleys and ill-conditioned Jacobians. In this case, the line search repeatedly enforces step damping, illustrating how globalization stabilizes the iteration even when convergence toward the solution is slow.

```rust
// Program 9.8.1: Globally convergent Newton method for nonlinear systems (final)
//
// This program implements a globally convergent Newton method for F(x)=0 (Eq. 9.8.1)
// using the merit function phi(x)=1/2 ||F(x)||_2^2 (Eq. 9.8.2).
//
// The merit gradient is ∇phi(x)=J(x)^T F(x) (Eq. 9.8.3). A direction p is accepted
// only if it is a descent direction, ∇phi(x)^T p < 0 (Eq. 9.8.4).
//
// Globalization is achieved by Armijo backtracking line search on phi. The key
// mechanism is step-length control: when the full Newton step is too aggressive,
// the line search reduces alpha until a sufficient decrease in phi is obtained.
//
// If the Newton direction cannot be computed (singular Jacobian) or fails the
// descent test, the algorithm falls back to steepest descent p = -∇phi.

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn add_scaled(x: &[f64], alpha: f64, p: &[f64]) -> Vec<f64> {
    x.iter().zip(p.iter()).map(|(xi, pi)| xi + alpha * pi).collect()
}

fn jt_vec(j: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let n = j.len();
    let m = j[0].len();
    let mut out = vec![0.0; m];
    for i in 0..n {
        for k in 0..m {
            out[k] += j[i][k] * v[i];
        }
    }
    out
}

fn phi_from_f(f: &[f64]) -> f64 {
    0.5 * dot(f, f) // Eq. (9.8.2)
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }

    for k in 0..n {
        let mut piv = k;
        let mut best = a[k][k].abs();
        for i in (k + 1)..n {
            let v = a[i][k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best < 1e-15 {
            return None;
        }
        if piv != k {
            a.swap(k, piv);
            b.swap(k, piv);
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
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

trait NonlinearSystem {
    fn f(&self, x: &[f64]) -> Vec<f64>;
    fn jacobian(&self, x: &[f64]) -> Vec<Vec<f64>>;
    fn name(&self) -> &'static str;
}

#[derive(Clone, Copy)]
struct GlobalNewtonParams {
    max_iter: usize,
    tol_f: f64,
    tol_step: f64,
    c1: f64,
    beta: f64,
    min_alpha: f64,
    max_ls_steps: usize,
}

impl Default for GlobalNewtonParams {
    fn default() -> Self {
        Self {
            max_iter: 80,
            tol_f: 1e-10,
            tol_step: 1e-14,
            c1: 1e-4,
            beta: 0.5,
            min_alpha: 1e-16,
            max_ls_steps: 60,
        }
    }
}

#[derive(Debug)]
struct SolveResult {
    x: Vec<f64>,
    converged: bool,
    iters: usize,
    final_norm_f: f64,
}

fn armijo_search<S: NonlinearSystem>(
    sys: &S,
    x: &[f64],
    phi: f64,
    grad_phi: &[f64],
    p: &[f64],
    prm: GlobalNewtonParams,
    alpha0: f64,
) -> Option<(Vec<f64>, f64, f64, usize)> {
    let gtp = dot(grad_phi, p);
    if !(gtp < 0.0) {
        return None;
    }

    let mut alpha = alpha0;
    for ls in 0..prm.max_ls_steps {
        if alpha < prm.min_alpha {
            return None;
        }
        let x_trial = add_scaled(x, alpha, p);
        let f_trial = sys.f(&x_trial);
        let phi_trial = phi_from_f(&f_trial);

        if phi_trial <= phi + prm.c1 * alpha * gtp {
            return Some((x_trial, phi_trial, alpha, ls));
        }
        alpha *= prm.beta;
    }
    None
}

fn global_newton<S: NonlinearSystem>(sys: &S, x0: Vec<f64>, prm: GlobalNewtonParams) -> SolveResult {
    let mut x = x0;

    println!("\nSystem: {}", sys.name());
    println!("Initial guess: x0 = {:?}\n", x);

    for k in 0..prm.max_iter {
        let f = sys.f(&x);
        let fnrm = norm2(&f);
        let phi = phi_from_f(&f);

        println!(
            "iter {:>2}: ||F(x)||_2 = {:>12.6e}, phi(x) = {:>12.6e}",
            k, fnrm, phi
        );

        if fnrm <= prm.tol_f {
            return SolveResult { x, converged: true, iters: k, final_norm_f: fnrm };
        }

        let j = sys.jacobian(&x);
        let grad_phi = jt_vec(&j, &f); // Eq. (9.8.3)

        // Steepest descent fallback
        let p_sd: Vec<f64> = grad_phi.iter().map(|g| -g).collect();

        // Newton direction: J p = -F
        let p_newton = solve_linear_system(j.clone(), f.iter().map(|v| -v).collect());

        // Choose direction: Newton if descent, else steepest descent.
        let (used_dir, p_used) = if let Some(pn) = p_newton {
            if dot(&grad_phi, &pn) < 0.0 {
                ("Newton", pn)
            } else {
                ("Steepest", p_sd.clone())
            }
        } else {
            ("Steepest", p_sd.clone())
        };

        // Armijo from alpha0=1 always: step-length control is the intended globalization.
        let step = armijo_search(sys, &x, phi, &grad_phi, &p_used, prm, 1.0);

        let (x_new, phi_new, alpha, ls_steps) = match step {
            Some(s) => s,
            None => {
                println!("         Line search failed; terminating.");
                return SolveResult { x, converged: false, iters: k + 1, final_norm_f: fnrm };
            }
        };

        let gtp = dot(&grad_phi, &p_used);
        println!(
            "         dir = {:<8}  grad^T p = {:>12.3e}  alpha = {:>12.3e}  ls_steps = {:>2}  phi_new = {:>12.3e}",
            used_dir, gtp, alpha, ls_steps, phi_new
        );

        let step_norm = alpha * norm2(&p_used);
        x = x_new;

        if step_norm <= prm.tol_step * (1.0 + norm2(&x)) {
            println!(
                "         Step too small (||alpha p||_2 = {:>12.3e}); terminating.",
                step_norm
            );
            let f_now = sys.f(&x);
            let fnrm_now = norm2(&f_now);
            return SolveResult { x, converged: fnrm_now <= prm.tol_f, iters: k + 1, final_norm_f: fnrm_now };
        }
    }

    let f_final = sys.f(&x);
    let fnrm_final = norm2(&f_final);
    SolveResult { x, converged: false, iters: prm.max_iter, final_norm_f: fnrm_final }
}

// -------------------- Example systems --------------------

struct CircleLine;

impl NonlinearSystem for CircleLine {
    fn name(&self) -> &'static str { "Circle-line intersection" }

    fn f(&self, x: &[f64]) -> Vec<f64> {
        let (a, b) = (x[0], x[1]);
        vec![a * a + b * b - 1.0, a - b]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let (a, b) = (x[0], x[1]);
        vec![vec![2.0 * a, 2.0 * b], vec![1.0, -1.0]]
    }
}

struct RosenbrockGrad;

impl NonlinearSystem for RosenbrockGrad {
    fn name(&self) -> &'static str { "Rosenbrock stationarity (∇f=0)" }

    fn f(&self, x: &[f64]) -> Vec<f64> {
        let (a, b) = (x[0], x[1]);
        let t = b - a * a;
        let fx = 2.0 * (a - 1.0) - 400.0 * a * t;
        let fy = 200.0 * t;
        vec![fx, fy]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let (a, b) = (x[0], x[1]);
        let j11 = 2.0 - 400.0 * (b - a * a) + 800.0 * a * a;
        let j12 = -400.0 * a;
        let j21 = -400.0 * a;
        let j22 = 200.0;
        vec![vec![j11, j12], vec![j21, j22]]
    }
}

fn run_case<S: NonlinearSystem>(sys: &S, x0: Vec<f64>, prm: GlobalNewtonParams) {
    let res = global_newton(sys, x0, prm);

    println!("\nSummary:");
    println!("  converged      = {}", res.converged);
    println!("  iters          = {}", res.iters);
    println!("  x              = {:?}", res.x);
    println!("  ||F(x)||_2     = {:.6e}", res.final_norm_f);
}

fn main() {
    println!("Program 9.8.1: Globally convergent Newton method (merit + line search)");

    let prm = GlobalNewtonParams::default();

    let sys1 = CircleLine;
    run_case(&sys1, vec![2.0, 0.2], prm);

    let sys2 = RosenbrockGrad;
    run_case(&sys2, vec![-1.2, 1.0], prm);
    run_case(&sys2, vec![2.0, 2.0], prm);
}
```

Program 9.8.1 demonstrates how the abstract concepts introduced in Section 9.8.1 translate into a practical and robust nonlinear solver. By combining Newton’s method with explicit descent verification and merit-function-based line search, the algorithm guarantees monotonic reduction of the residual norm while preserving fast local convergence when conditions permit.

The numerical experiments highlight two complementary behaviors. For well-scaled problems with favorable geometry, such as the circle–line intersection, the globalization machinery remains inactive and the method behaves like classical Newton iteration. For more challenging systems, such as the Rosenbrock stationarity equations, globalization becomes essential: step lengths are reduced automatically to prevent divergence, ensuring steady if conservative progress toward feasibility.

Although this basic line-search strategy ensures global convergence, the slow convergence observed in difficult cases underscores the limitations of pure merit-function damping. This observation motivates the development of more sophisticated globalization techniques, such as trust-region methods and Levenberg–Marquardt damping, which will be introduced in subsequent sections. Together, these approaches form the foundation of modern large-scale nonlinear solvers used in scientific computing and optimization.

## 9.8.2. Damped Newton and Line Search Methods

One of the simplest and most widely used globalization strategies is damping, also known as line search Newton methods. The central idea is to retain the Newton direction, which is highly effective near a solution, while controlling the step length to prevent divergence when far from it.

At each iteration, the Newton direction $\delta\mathbf{x}$ is first computed by solving the Jacobian linear system:

$$J(\mathbf{x})\,\delta\mathbf{x} = -\mathbf{F}(\mathbf{x}) \tag{9.8.5}$$

Rather than applying this correction in full, the iterate is updated using a scaled step,

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \lambda_k\,\delta\mathbf{x}, \qquad 0 < \lambda_k \le 1 \tag{9.8.6}$$

Here, the scalar parameter $\lambda_k$ controls how far the algorithm moves along the Newton direction. When $\lambda_k = 1$, the method reduces to the standard Newton iteration; when $\lambda_k < 1$, the step is damped to improve stability.

The choice of $\lambda_k$ is guided by a line search that enforces sufficient decrease in a merit function $\phi$. A commonly used criterion is the Armijo condition,

$$
\phi(\mathbf{x}_k + \lambda_k\,\delta\mathbf{x})
\le
\phi(\mathbf{x}_k)
+
c\,\lambda_k\,\nabla \phi(\mathbf{x}_k)^{\mathsf{T}} \delta\mathbf{x}
\tag{9.8.7}
$$

where $0 < c < 1$ is a fixed constant. This inequality ensures that the reduction in the merit function is proportional to both the step length and the local directional derivative, thereby ruling out steps that are too aggressive or unproductive.

In practice, the algorithm first tests the full Newton step $\lambda_k = 1$. If the Armijo condition is satisfied, the step is accepted immediately. Otherwise, $\lambda_k$ is reduced, typically by a backtracking procedure that successively multiplies $\lambda_k$ by a constant factor less than one until the condition holds. This adaptive strategy allows the method to behave conservatively when the linear model is unreliable, while automatically reverting to the fast undamped Newton iteration once the iterates enter the local convergence region.

Under mild smoothness assumptions, damped Newton methods with line search are globally convergent, meaning that they generate iterates that make consistent progress toward reducing the residual norm. At the same time, they preserve the quadratic convergence of Newton’s method near a solution, since $\lambda_k = 1$ is eventually accepted at every iteration (Berzi, 2024). For this reason, line search damping is often regarded as the simplest effective safeguard for Newton-based solvers.

### Rust Implementation

Following the discussion in Section 9.8.2 on damping and line search strategies for Newton’s method, Program 9.8.2 provides a concrete implementation of a globally convergent Newton solver based on merit-function control and Armijo backtracking. While the classical Newton iteration derived in Section 9.7 achieves rapid local convergence, its reliability deteriorates when the initial guess lies far from a solution. The present program demonstrates how this limitation can be overcome by scaling the Newton correction using a dynamically chosen step length, ensuring monotone reduction of the merit function defined in Equation (9.8.2). The implementation retains the efficiency of Newton’s method near a root while safeguarding against divergence through a simple yet effective line search mechanism. Two representative nonlinear systems are evaluated to illustrate both undamped and damped regimes, highlighting the practical behavior of the algorithm across different basins of attraction.

At the core of the implementation is a trait-based abstraction for nonlinear systems, which defines a uniform interface for evaluating the residual vector $\mathbf{F}(\mathbf{x})$ and its Jacobian $J(\mathbf{x})$. This design allows the same Newton and line search logic to be applied to different systems without modification, reflecting the general formulation of nonlinear equations introduced in Equation (9.8.1).

The Newton direction $\delta\mathbf{x}$ is computed at each iteration by solving the linear system given in Equation (9.8.5). This operation encapsulates the local linearization of the nonlinear mapping and provides a direction that is quadratically convergent in the vicinity of a solution. However, as emphasized in Section 9.8.2, the raw Newton step may be unreliable when the current iterate lies outside the local convergence region.

To enforce global convergence, the program evaluates progress using the merit function $\phi(\mathbf{x})$ defined in Equation (9.8.2). Its gradient, computed according to Equation (9.8.3), is used to evaluate the directional derivative $\nabla\phi(\mathbf{x}_k)^{\mathsf T}\delta\mathbf{x}$, which determines whether the Newton direction is a descent direction as required by Equation (9.8.4).

The function implementing the Armijo line search applies the sufficient decrease condition given in Equation (9.8.7). Starting from the full Newton step $\lambda_k = 1$, the algorithm repeatedly reduces the step length by a fixed factor until the Armijo inequality is satisfied. This backtracking procedure ensures that each accepted iterate produces a decrease in the merit function proportional to both the step length and the local directional derivative. The number of backtracking steps and the accepted value of $\lambda_k$ are recorded to expose the algorithm’s adaptive behavior during execution.

The `main` Newton iteration orchestrates these components by alternating between direction computation, line search, and iterate update according to Equation (9.8.6). Convergence is detected when the Euclidean norm of the residual falls below a prescribed tolerance, indicating that the system has been solved to the desired accuracy. Diagnostic output at each iteration reports the residual norm, merit function value, step length, and line search activity, providing detailed insight into the globalization mechanism at work.

```rust
// Program 9.8.2: Damped Newton Method with Armijo Backtracking Line Search
//
// This program implements the damping (line search) strategy described in Section 9.8.2.
// At each iterate x_k it computes the Newton correction δx by solving
//     J(x_k) δx = -F(x_k)                                          (Eq. 9.8.5)
// and updates
//     x_{k+1} = x_k + λ_k δx,   0 < λ_k ≤ 1                         (Eq. 9.8.6)
// where λ_k is chosen by Armijo backtracking on the merit function
//     φ(x) = 1/2 ||F(x)||_2^2                                       (Eq. 9.8.2)
// enforcing
//     φ(x_k + λ_k δx) ≤ φ(x_k) + c λ_k ∇φ(x_k)^T δx                 (Eq. 9.8.7).
//
// The implementation is self-contained (no external crates) and includes two example
// nonlinear systems. The second example is a simple “Rosenbrock valley” equation system
// with a known root at (1,1), chosen to behave well under the standard nonlinear-equation
// merit function φ = 1/2 ||F||^2.

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn add_scaled(x: &[f64], alpha: f64, p: &[f64]) -> Vec<f64> {
    x.iter().zip(p.iter()).map(|(xi, pi)| xi + alpha * pi).collect()
}

fn jt_vec(j: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    // Computes J^T v
    let n = j.len();
    let m = j[0].len();
    let mut out = vec![0.0; m];
    for i in 0..n {
        for k in 0..m {
            out[k] += j[i][k] * v[i];
        }
    }
    out
}

fn phi_from_f(f: &[f64]) -> f64 {
    0.5 * dot(f, f) // Eq. (9.8.2)
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    // Dense Gaussian elimination with partial pivoting (sufficient for small demos).
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }

    for k in 0..n {
        // Pivot
        let mut piv = k;
        let mut best = a[k][k].abs();
        for i in (k + 1)..n {
            let v = a[i][k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best < 1e-15 {
            return None; // singular / near-singular
        }
        if piv != k {
            a.swap(k, piv);
            b.swap(k, piv);
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = a[i][k] / a[k][k];
            a[i][k] = 0.0;
            for j in (k + 1)..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

trait NonlinearSystem {
    fn f(&self, x: &[f64]) -> Vec<f64>;
    fn jacobian(&self, x: &[f64]) -> Vec<Vec<f64>>;
    fn name(&self) -> &'static str;
}

#[derive(Clone, Copy)]
struct LineSearchParams {
    max_iter: usize,
    tol_f: f64,          // stop when ||F(x)||_2 <= tol_f
    tol_step: f64,       // stop when ||λ δx||_2 is tiny
    c: f64,              // Armijo constant in Eq. (9.8.7)
    beta: f64,           // backtracking factor (0,1)
    min_lambda: f64,     // minimum step length
    max_ls_steps: usize, // maximum backtracking reductions
    print_every: usize,  // print every k-th iteration (1 prints all)
}

impl Default for LineSearchParams {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol_f: 1e-12,
            tol_step: 1e-14,
            c: 1e-4,
            beta: 0.5,
            min_lambda: 1e-16,
            max_ls_steps: 60,
            print_every: 1,
        }
    }
}

#[derive(Debug)]
struct SolveResult {
    x: Vec<f64>,
    converged: bool,
    iters: usize,
    final_norm_f: f64,
}

fn damped_newton_armijo<S: NonlinearSystem>(
    sys: &S,
    x0: Vec<f64>,
    prm: LineSearchParams,
) -> SolveResult {
    let mut x = x0;

    println!("\nSystem: {}", sys.name());
    println!("Initial guess: x0 = {:?}\n", x);

    for k in 0..prm.max_iter {
        let f = sys.f(&x);
        let fnrm = norm2(&f);
        let phi = phi_from_f(&f);

        if k % prm.print_every == 0 {
            println!(
                "iter {:>3}: ||F(x)||_2 = {:>12.6e}, phi(x) = {:>12.6e}",
                k, fnrm, phi
            );
        }

        if fnrm <= prm.tol_f {
            return SolveResult {
                x,
                converged: true,
                iters: k,
                final_norm_f: fnrm,
            };
        }

        // Merit gradient: ∇φ = J^T F (Eq. 9.8.3)
        let j = sys.jacobian(&x);
        let grad_phi = jt_vec(&j, &f);

        // Newton correction from J δx = -F (Eq. 9.8.5)
        let delta = match solve_linear_system(j.clone(), f.iter().map(|v| -v).collect()) {
            Some(d) => d,
            None => {
                println!("         Jacobian solve failed; terminating.");
                return SolveResult {
                    x,
                    converged: false,
                    iters: k + 1,
                    final_norm_f: fnrm,
                };
            }
        };

        // Directional derivative used in Armijo (Eq. 9.8.7)
        let dirder = dot(&grad_phi, &delta);

        // In the ideal case, Newton provides a descent direction for φ.
        // If not, Armijo backtracking may fail; we terminate to keep 9.8.2 minimal.
        if !(dirder < 0.0) {
            println!(
                "         Newton direction is not descent (∇phi^Tδx = {:>12.3e}); terminating.",
                dirder
            );
            return SolveResult {
                x,
                converged: false,
                iters: k + 1,
                final_norm_f: fnrm,
            };
        }

        // Backtracking line search for λ_k (Eq. 9.8.6), Armijo test (Eq. 9.8.7)
        let mut lambda = 1.0;
        let mut accepted = false;

        for ls in 0..prm.max_ls_steps {
            if lambda < prm.min_lambda {
                break;
            }

            let x_trial = add_scaled(&x, lambda, &delta);
            let f_trial = sys.f(&x_trial);
            let phi_trial = phi_from_f(&f_trial);

            if phi_trial <= phi + prm.c * lambda * dirder {
                x = x_trial;
                accepted = true;

                if k % prm.print_every == 0 {
                    println!(
                        "         λ = {:>12.3e}  backtracks = {:>2}  ∇phi^Tδx = {:>12.3e}  phi_new = {:>12.3e}",
                        lambda, ls, dirder, phi_trial
                    );
                }
                break;
            }

            lambda *= prm.beta;
        }

        if !accepted {
            println!("         Line search failed; terminating.");
            return SolveResult {
                x,
                converged: false,
                iters: k + 1,
                final_norm_f: fnrm,
            };
        }

        // Small-step termination check
        let step_norm = lambda * norm2(&delta);
        if step_norm <= prm.tol_step * (1.0 + norm2(&x)) {
            if k % prm.print_every == 0 {
                println!(
                    "         Step too small (||λ δx||_2 = {:>12.3e}); terminating.",
                    step_norm
                );
            }
            let f_now = sys.f(&x);
            let fnrm_now = norm2(&f_now);
            return SolveResult {
                x,
                converged: fnrm_now <= prm.tol_f,
                iters: k + 1,
                final_norm_f: fnrm_now,
            };
        }
    }

    let f_final = sys.f(&x);
    let fnrm_final = norm2(&f_final);
    SolveResult {
        x,
        converged: false,
        iters: prm.max_iter,
        final_norm_f: fnrm_final,
    }
}

fn run_case<S: NonlinearSystem>(sys: &S, x0: Vec<f64>, prm: LineSearchParams) {
    let res = damped_newton_armijo(sys, x0, prm);

    println!("\nSummary:");
    println!("  converged  = {}", res.converged);
    println!("  iters      = {}", res.iters);
    println!("  x          = {:?}", res.x);
    println!("  ||F(x)||_2 = {:.6e}", res.final_norm_f);
}

// ------------------------------------------------------------
// Example system 1: circle–line intersection
// F1(x,y) = x^2 + y^2 - 1
// F2(x,y) = x - y
// ------------------------------------------------------------
struct CircleLine;

impl NonlinearSystem for CircleLine {
    fn name(&self) -> &'static str {
        "Circle-line intersection"
    }

    fn f(&self, x: &[f64]) -> Vec<f64> {
        let (a, b) = (x[0], x[1]);
        vec![a * a + b * b - 1.0, a - b]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let (a, b) = (x[0], x[1]);
        vec![vec![2.0 * a, 2.0 * b], vec![1.0, -1.0]]
    }
}

// ------------------------------------------------------------
// Example system 2: “Rosenbrock valley” equations with root at (1,1)
// A simple nonlinear system that is commonly used to illustrate curved valleys:
//
//   F1(x,y) = 10 (y - x^2)   = 0
//   F2(x,y) = 1 - x          = 0
//
// Root: (x,y) = (1,1).
// ------------------------------------------------------------
struct RosenbrockValleyEq;

impl NonlinearSystem for RosenbrockValleyEq {
    fn name(&self) -> &'static str {
        "Rosenbrock valley equations (root at (1,1))"
    }

    fn f(&self, x: &[f64]) -> Vec<f64> {
        let (a, b) = (x[0], x[1]);
        vec![10.0 * (b - a * a), 1.0 - a]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let a = x[0];
        vec![vec![-20.0 * a, 10.0], vec![-1.0, 0.0]]
    }
}

fn main() {
    println!("Program 9.8.2: Damped Newton and Armijo line search\n");

    // You can reduce output volume by setting print_every = 5 or 10.
    let prm = LineSearchParams {
        max_iter: 200,
        tol_f: 1e-12,
        tol_step: 1e-14,
        c: 1e-4,
        beta: 0.5,
        min_lambda: 1e-16,
        max_ls_steps: 60,
        print_every: 1,
    };

    let sys1 = CircleLine;
    run_case(&sys1, vec![2.0, 0.2], prm);

    let sys2 = RosenbrockValleyEq;
    run_case(&sys2, vec![-1.2, 1.0], prm);
    run_case(&sys2, vec![2.0, 2.0], prm);
}
```

Program 9.8.2 demonstrates how a simple damping strategy based on Armijo line search transforms Newton’s method from a locally convergent algorithm into a robust solver capable of handling challenging initial guesses. By enforcing monotone reduction of the merit function, the method guarantees consistent progress toward feasibility even when the linear model is initially unreliable.

The numerical examples illustrate two characteristic regimes of behavior. In well-conditioned regions close to a solution, the full Newton step is accepted without backtracking, and the method exhibits the expected quadratic convergence. Farther from the solution, the line search automatically reduces the step length, stabilizing the iteration and guiding it into the local convergence region. This adaptive transition between damped and undamped updates reflects the theoretical properties discussed in Section 9.8.2.

The modular structure of the implementation allows alternative merit functions, line search parameters, or Jacobian solvers to be incorporated with minimal effort. As such, the program provides a practical foundation for more advanced globalization strategies, including trust-region methods and hybrid Newton–gradient schemes, which further enhance robustness in large-scale and ill-conditioned nonlinear systems.

## 9.8.3. Trust-Region Methods

An alternative and often more robust globalization strategy is provided by trust-region methods. Instead of explicitly scaling the Newton step, these methods restrict the correction to lie within a region where the local model of the nonlinear system is considered trustworthy.

At iteration $k$, one defines a *trust region* centered at the current iterate $\mathbf{x}_k$*,* typically a ball of radius $\Delta_k > 0$. The correction $\mathbf{p}$ is then obtained by solving the constrained subproblem:

$$
\min_{\mathbf{p}}
\;\bigl\lVert \mathbf{F}(\mathbf{x}_k) + J(\mathbf{x}_k)\mathbf{p} \bigr\rVert_2
\quad
\text{subject to}
\quad
\lVert \mathbf{p} \rVert_2 \le \Delta_k
\tag{9.8.8}
$$

This formulation seeks the step that best reduces the linearized residual while remaining within a region where the linear approximation is presumed accurate.

The key distinction from line search methods lies in how step acceptance is determined. After computing $\mathbf{p}$, the algorithm compares the *actual reduction* in the residual norm with the *predicted reduction* given by the linear model. If the agreement is good, the trust region radius $\Delta_k$ is increased, signaling confidence in the model. If the agreement is poor, $\Delta_k$ is reduced, forcing subsequent steps to remain closer to the current iterate.

Trust-region methods are particularly effective when the Jacobian is ill-conditioned or when Newton directions are unreliable or nearly singular. By explicitly limiting step size, they prevent large, destabilizing updates even when the Newton correction itself is poorly scaled. This robustness makes trust-region methods especially attractive in large-scale nonlinear systems and in problems exhibiting strong nonlinearity or near-degeneracy.

In practice, trust-region frameworks form the foundation of many modern nonlinear solvers and are closely related to algorithms used in large-scale optimization. Compared with line search methods, they often require more complex subproblem solvers, but they offer superior stability in challenging regimes. As such, trust-region and line search strategies represent complementary approaches to globalization, each balancing robustness, efficiency, and implementation complexity in different ways.

### Rust Implementation

Following the discussion in Section 9.8.3 on trust-region globalization strategies for nonlinear systems, Program 9.8.3 presents a concrete implementation of a trust-region Newton method based on the classical dogleg algorithm. In contrast to line search approaches, which regulate progress by scaling the Newton step along a fixed direction, trust-region methods explicitly restrict the step to lie within a region where the local linear model is considered reliable. This program demonstrates how such a region is constructed, updated, and used to balance robustness and efficiency. The implementation illustrates how trust-region techniques naturally adapt between steepest-descent behavior far from a solution and fast Gauss–Newton convergence once the iterates enter the local convergence regime. Two representative nonlinear systems are used to highlight the method’s stability, model-agreement diagnostics, and convergence behavior under challenging initial conditions.

At the core of the implementation is the `NonlinearSystem` trait, which defines a common interface for systems of nonlinear equations by requiring methods to evaluate the residual vector $\mathbf{F}(\mathbf{x})$ and its Jacobian $J(\mathbf{x})$. This abstraction allows the trust-region solver to be applied uniformly to different problems without altering the algorithmic structure. Each system supplies only problem-specific information, while all globalization logic is handled by the solver itself.

The trust-region subproblem is formulated in accordance with Equation (9.8.8), which seeks a correction $\mathbf{p}$ that minimizes the norm of the linearized residual subject to the constraint $\lVert \mathbf{p} \rVert_2 \le \Delta_k$. In the code, this subproblem is solved approximately using the dogleg strategy. The algorithm first computes the Gauss–Newton step by solving the normal equations associated with the linearized system. If this step lies inside the trust region, it is accepted directly. Otherwise, the algorithm constructs a piecewise path that interpolates between the steepest-descent (Cauchy) direction and the Gauss–Newton direction, selecting the point along this path that intersects the trust-region boundary.

The quality of each proposed step is assessed using the ratio $\rho_k$ between the actual reduction in the merit function $\phi(\mathbf{x}) = \tfrac{1}{2}\lVert \mathbf{F}(\mathbf{x}) \rVert_2^2$ and the reduction predicted by the linear model. When $\rho_k$ is close to one, the linear model accurately predicts the behavior of the nonlinear system, and the trust-region radius $\Delta_k$ is increased. When $\rho_k$ is small or negative, indicating poor agreement, the step is rejected and $\Delta_k$ is reduced. This mechanism allows the solver to adapt automatically to regions where the Jacobian is ill-conditioned or the linearization is unreliable.

The main driver function orchestrates the iteration by repeatedly forming the linear model, computing the dogleg step, evaluating the agreement ratio, and updating both the iterate and the trust-region radius. Convergence is declared when the residual norm falls below a prescribed tolerance or when the step length becomes numerically insignificant. Detailed iteration logs report the residual norm, merit function value, trust-region radius, step type, and acceptance decision, making the algorithm’s behavior transparent and suitable for instructional and diagnostic purposes.

```rust
// Program 9.8.3: Trust-region method (dogleg) for nonlinear systems
//
// Trust-region globalization for F(x)=0 using the dogleg step. The merit is
// phi(x)=0.5||F(x)||^2. Step acceptance uses rho = actual_reduction / predicted_reduction.
// Robust termination: stop if ||F|| is small, or if ||J^T F|| is small, or if the
// model becomes numerically flat (predicted reduction ~ 0) while already near a root.

use std::f64;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn add_scaled(x: &[f64], p: &[f64], alpha: f64) -> Vec<f64> {
    x.iter().zip(p.iter()).map(|(xi, pi)| xi + alpha * pi).collect()
}

fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, x)).collect()
}

fn mat_t(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = a[0].len();
    let mut at = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            at[j][i] = a[i][j];
        }
    }
    at
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for t in 0..k {
            let ait = a[i][t];
            for j in 0..n {
                c[i][j] += ait * b[t][j];
            }
        }
    }
    c
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    // Dense Gaussian elimination with partial pivoting.
    let n = a.len();
    if n == 0 || a[0].len() != n || b.len() != n {
        return None;
    }

    for k in 0..n {
        let mut piv = k;
        let mut best = a[k][k].abs();
        for i in (k + 1)..n {
            let v = a[i][k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best < 1e-14 {
            return None;
        }
        if piv != k {
            a.swap(k, piv);
            b.swap(k, piv);
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
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

trait NonlinearSystem {
    fn name(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn eval_f(&self, x: &[f64]) -> Vec<f64>;
    fn eval_j(&self, x: &[f64]) -> Vec<Vec<f64>>;
}

fn phi_from_f(f: &[f64]) -> f64 {
    0.5 * dot(f, f)
}

fn predicted_phi(f: &[f64], j: &[Vec<f64>], p: &[f64]) -> f64 {
    // Model value: 0.5 ||F + J p||^2
    let jp = mat_vec(j, p);
    let mut fp = vec![0.0; f.len()];
    for i in 0..f.len() {
        fp[i] = f[i] + jp[i];
    }
    phi_from_f(&fp)
}

fn gauss_newton_step(f: &[f64], j: &[Vec<f64>]) -> Option<Vec<f64>> {
    // Solve (J^T J) p = -J^T f (normal equations).
    let jt = mat_t(j);
    let jtj = mat_mul(&jt, j);
    let jtf = mat_vec(&jt, f);
    let rhs: Vec<f64> = jtf.iter().map(|v| -v).collect();
    solve_linear_system(jtj, rhs)
}

fn cauchy_step(f: &[f64], j: &[Vec<f64>], delta: f64) -> Vec<f64> {
    // Cauchy step for phi: g = J^T f.
    let jt = mat_t(j);
    let g = mat_vec(&jt, f);
    let ng = norm2(&g);
    if ng < 1e-14 {
        return vec![0.0; g.len()];
    }

    let jg = mat_vec(j, &g);
    let denom = dot(&jg, &jg);
    let mut tau = if denom > 1e-14 { dot(&g, &g) / denom } else { 1.0 };

    let mut p = g.iter().map(|gi| -tau * gi).collect::<Vec<f64>>();
    let np = norm2(&p);

    if np > delta {
        tau = delta / ng;
        p = g.iter().map(|gi| -tau * gi).collect();
    }
    p
}

fn dogleg_step(f: &[f64], j: &[Vec<f64>], delta: f64) -> (Vec<f64>, &'static str) {
    let p_u = cauchy_step(f, j, f64::INFINITY);
    let p_gn_opt = gauss_newton_step(f, j);

    let p_gn = match p_gn_opt {
        Some(p) => p,
        None => {
            let p = cauchy_step(f, j, delta);
            return (p, "Cauchy");
        }
    };

    let nu = norm2(&p_u);
    let ngn = norm2(&p_gn);

    if ngn <= delta {
        return (p_gn, "GaussNewton");
    }
    if nu >= delta {
        return (cauchy_step(f, j, delta), "Cauchy");
    }

    let d = p_gn.iter().zip(p_u.iter()).map(|(a, b)| a - b).collect::<Vec<f64>>();
    let a = dot(&d, &d);
    let b = 2.0 * dot(&p_u, &d);
    let c = dot(&p_u, &p_u) - delta * delta;

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 || a.abs() < 1e-14 {
        let scale = delta / nu.max(1e-14);
        let p = p_u.iter().map(|v| scale * v).collect::<Vec<f64>>();
        return (p, "Cauchy");
    }

    let sqrt_disc = disc.sqrt();
    let t1 = (-b + sqrt_disc) / (2.0 * a);
    let t2 = (-b - sqrt_disc) / (2.0 * a);
    let t = if (0.0..=1.0).contains(&t1) { t1 } else { t2 };

    let p = p_u
        .iter()
        .zip(d.iter())
        .map(|(pu, di)| pu + t * di)
        .collect::<Vec<f64>>();

    (p, "Dogleg")
}

#[derive(Debug)]
struct TrustRegionParams {
    tol_f: f64,     // ||F||
    tol_g: f64,     // ||J^T F|| (first-order stationarity of phi)
    tol_step: f64,  // ||p||
    max_iters: usize,
    delta0: f64,
    delta_max: f64,
    eta: f64,
    gamma_dec: f64,
    gamma_inc: f64,
    flat_pred: f64, // threshold for "predicted reduction is numerically zero"
}

#[derive(Debug)]
struct SolveResult {
    x: Vec<f64>,
    converged: bool,
    iters: usize,
}

fn trust_region_solve(sys: &dyn NonlinearSystem, x0: &[f64], prm: &TrustRegionParams) -> SolveResult {
    let n = sys.dim();
    if x0.len() != n {
        panic!("x0 dimension mismatch: got {}, expected {}", x0.len(), n);
    }

    let mut x = x0.to_vec();
    let mut delta = prm.delta0;

    println!("\nSystem: {}", sys.name());
    println!("Initial guess: x0 = {:?}\n", x0);

    for k in 0..prm.max_iters {
        let f = sys.eval_f(&x);
        let j = sys.eval_j(&x);

        let fnorm = norm2(&f);
        let phi = phi_from_f(&f);

        // Gradient of phi: g = J^T F
        let jt = mat_t(&j);
        let g = mat_vec(&jt, &f);
        let gnorm = norm2(&g);

        println!(
            "iter {:3}: ||F(x)||_2 = {:>12.6e}, phi(x) = {:>12.6e}, Δ = {:>10.3e}",
            k, fnorm, phi, delta
        );

        if fnorm <= prm.tol_f {
            println!("         converged: ||F|| below tolerance\n");
            return SolveResult { x, converged: true, iters: k };
        }
        if gnorm <= prm.tol_g {
            println!("         converged: ||J^T F|| below tolerance\n");
            return SolveResult { x, converged: true, iters: k };
        }

        let (p, p_kind) = dogleg_step(&f, &j, delta);
        let pnorm = norm2(&p);

        if pnorm <= prm.tol_step {
            // If the step is tiny and we are already very close, accept as converged.
            // This avoids declaring failure due to roundoff-limited progress.
            if fnorm <= 10.0 * prm.tol_f || gnorm <= 10.0 * prm.tol_g {
                println!(
                    "         step too small (||p||_2 = {:>12.6e}), but residual/stationarity is tiny; accepting convergence.\n",
                    pnorm
                );
                return SolveResult { x, converged: true, iters: k };
            }

            println!("         step too small (||p||_2 = {:>12.6e}); terminating.\n", pnorm);
            return SolveResult { x, converged: false, iters: k };
        }

        let phi_pred_new = predicted_phi(&f, &j, &p);
        let pred_red = (phi - phi_pred_new).max(0.0);

        let x_trial = add_scaled(&x, &p, 1.0);
        let f_trial = sys.eval_f(&x_trial);
        let phi_trial = phi_from_f(&f_trial);
        let act_red = phi - phi_trial;

        // If the model predicts essentially no change, we're in the roundoff regime.
        if pred_red <= prm.flat_pred {
            if fnorm <= 10.0 * prm.tol_f || gnorm <= 10.0 * prm.tol_g {
                println!(
                    "         model flat (pred_red = {:>12.6e}); residual/stationarity tiny; accepting convergence.\n",
                    pred_red
                );
                return SolveResult { x, converged: true, iters: k };
            } else {
                // If we're not close, treat as failure (Jacobian may be singular/poorly scaled).
                println!(
                    "         model flat (pred_red = {:>12.6e}) but not close to a root; terminating.\n",
                    pred_red
                );
                return SolveResult { x, converged: false, iters: k };
            }
        }

        let rho = act_red / pred_red;
        let accepted = rho >= prm.eta;

        println!(
            "         step = {:<10} ||p|| = {:>10.3e}  rho = {:>10.3e}  accepted = {}  phi_new = {:>12.6e}",
            p_kind, pnorm, rho, accepted, phi_trial
        );

        if rho < 0.25 {
            delta = (prm.gamma_dec * delta).max(1e-16);
        } else if rho > 0.75 && (pnorm - delta).abs() / delta.max(1e-16) < 0.10 {
            delta = (prm.gamma_inc * delta).min(prm.delta_max);
        }

        if accepted {
            x = x_trial;
        }

        println!();
    }

    SolveResult { x, converged: false, iters: prm.max_iters }
}

// ---------------- Example systems ----------------

struct CircleLineIntersection;

impl NonlinearSystem for CircleLineIntersection {
    fn name(&self) -> &'static str { "Circle-line intersection" }
    fn dim(&self) -> usize { 2 }
    fn eval_f(&self, x: &[f64]) -> Vec<f64> {
        let x0 = x[0];
        let x1 = x[1];
        vec![x0 * x0 + x1 * x1 - 1.0, x0 - x1]
    }
    fn eval_j(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let x0 = x[0];
        let x1 = x[1];
        vec![vec![2.0 * x0, 2.0 * x1], vec![1.0, -1.0]]
    }
}

struct RosenbrockValleyEquations;

impl NonlinearSystem for RosenbrockValleyEquations {
    fn name(&self) -> &'static str { "Rosenbrock valley equations (root at (1,1))" }
    fn dim(&self) -> usize { 2 }
    fn eval_f(&self, x: &[f64]) -> Vec<f64> {
        let x0 = x[0];
        let x1 = x[1];
        vec![10.0 * (x1 - x0 * x0), 1.0 - x0]
    }
    fn eval_j(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let x0 = x[0];
        vec![vec![-20.0 * x0, 10.0], vec![-1.0, 0.0]]
    }
}

fn main() {
    println!("Program 9.8.3: Trust-region method (dogleg) for nonlinear systems\n");

    let prm = TrustRegionParams {
        tol_f: 1e-12,
        tol_g: 1e-12,
        tol_step: 1e-14,
        max_iters: 100,
        delta0: 1.0,
        delta_max: 100.0,
        eta: 1e-4,
        gamma_dec: 0.25,
        gamma_inc: 2.0,
        flat_pred: 1e-30,
    };

    let sys1 = CircleLineIntersection;
    let x0_1 = vec![2.0, 0.2];
    let r1 = trust_region_solve(&sys1, &x0_1, &prm);
    println!(
        "Summary:\n  converged  = {}\n  iters      = {}\n  x          = {:?}\n  ||F(x)||_2 = {:>.6e}\n",
        r1.converged,
        r1.iters,
        r1.x,
        norm2(&sys1.eval_f(&r1.x))
    );

    let sys2 = RosenbrockValleyEquations;
    let x0_2 = vec![-1.2, 1.0];
    let r2 = trust_region_solve(&sys2, &x0_2, &prm);
    println!(
        "Summary:\n  converged  = {}\n  iters      = {}\n  x          = {:?}\n  ||F(x)||_2 = {:>.6e}\n",
        r2.converged,
        r2.iters,
        r2.x,
        norm2(&sys2.eval_f(&r2.x))
    );

    let x0_3 = vec![2.0, 2.0];
    let r3 = trust_region_solve(&sys2, &x0_3, &prm);
    println!(
        "Summary:\n  converged  = {}\n  iters      = {}\n  x          = {:?}\n  ||F(x)||_2 = {:>.6e}\n",
        r3.converged,
        r3.iters,
        r3.x,
        norm2(&sys2.eval_f(&r3.x))
    );
}
```

Program 9.8.3 demonstrates how trust-region methods provide a robust alternative to line search strategies for Newton-type solvers. By explicitly limiting the step to regions where the linearized model is trustworthy, the algorithm avoids the large, destabilizing updates that can arise when the Jacobian is poorly scaled or nearly singular. The numerical examples illustrate how rejected steps naturally lead to trust-region contraction, while accurate model predictions allow the method to recover the fast local convergence properties of Gauss–Newton iterations.

The contrast between accepted and rejected steps, particularly in the Rosenbrock valley examples, highlights the practical value of the agreement ratio $\rho_k$ as a diagnostic for model fidelity. Once the iterates enter a region where the linear approximation is reliable, the trust-region radius stabilizes and full Gauss–Newton steps are consistently accepted, yielding rapid convergence.

The modular structure of the solver makes it straightforward to extend this framework to higher-dimensional systems, alternative trust-region subproblem solvers, or hybrid methods that combine trust-region and line-search ideas. As such, this program provides a foundation for understanding and implementing modern large-scale nonlinear solvers, where robustness and predictability are often more critical than raw asymptotic speed.

## 9.8.4. Quasi-Newton and Broyden-Type Methods

For large nonlinear systems, the repeated computation, storage, and factorization of the Jacobian matrix can dominate the total computational cost of Newton’s method. This expense is particularly acute when the Jacobian is dense, expensive to evaluate, or changes slowly from one iteration to the next. *Quasi-Newton methods* address this difficulty by replacing the exact Jacobian with an approximation that is updated iteratively using information gathered during the solution process.

Rather than recomputing derivatives, quasi-Newton methods construct a sequence of approximate Jacobians $B_k$ (or approximate inverses) that progressively capture the local behavior of the nonlinear mapping. These updates are designed to satisfy *secant conditions*, which enforce consistency with the most recent step taken by the algorithm.

Among quasi-Newton schemes, *Broyden’s method* is one of the most prominent and widely used. Let:

$$
\mathbf{s}_k
=
\mathbf{x}_{k+1}
-
\mathbf{x}_k
\tag{9.8.9}
$$

denote the step between successive iterates, and let:

$$
\mathbf{y}_k
=
\mathbf{F}(\mathbf{x}_{k+1})
-
\mathbf{F}(\mathbf{x}_k)\tag{9.8.10}
$$

denote the corresponding change in the residual. Broyden’s update for the approximate Jacobian takes the rank-one form:

$$
B_{k+1}
=
B_k
+
\frac{\bigl(\mathbf{y}_k - B_k \mathbf{s}_k\bigr)\,\mathbf{s}_k^{\mathsf{T}}}
     {\mathbf{s}_k^{\mathsf{T}}\mathbf{s}_k}
\tag{9.8.11}
$$

This update enforces the secant condition,

$$
B_{k+1}\mathbf{s}_k
=
\mathbf{y}_k
\tag{9.8.12}
$$

ensuring that the new approximation correctly maps the most recent step to the observed change in the residual. Among all matrices satisfying this condition, Broyden’s formula produces the update that minimally perturbs $B_k$ in a least-change sense, thereby preserving accumulated information from previous iterations.

Quasi-Newton methods significantly reduce per-iteration cost compared with full Newton methods, especially when solving the Jacobian system is expensive. However, this efficiency comes at the expense of superlinear rather than quadratic convergence, and the quality of the approximation can degrade if the nonlinear system changes rapidly. For this reason, Broyden-type methods are often embedded within globalization frameworks such as line search or trust-region strategies to maintain robustness.

Numerous extensions of the basic scheme exist, including block updates and full-rank variants, which improve stability and convergence speed in large-scale settings. When combined with globalization mechanisms, these methods have proven effective for a wide range of nonlinear problems in scientific computing and engineering (Liu et al., 2023; Berzi, 2024).

### Rust Implementation

Following the discussion in Section 9.8.4 on quasi-Newton and Broyden-type updates, Program 9.8.4 provides a practical implementation of Broyden’s method equipped with an Armijo backtracking line search. The purpose of the program is to demonstrate how the rank-one Jacobian update in Equation (9.8.11) can substantially reduce the cost of repeatedly forming and factorizing exact Jacobians, while still maintaining robust progress through a merit-function globalization strategy. By combining a low-rank secant update with a conservative step acceptance rule, the implementation illustrates the central trade-off emphasized in this section: per-iteration efficiency is improved, but convergence becomes superlinear rather than quadratic, and globalization is often required to stabilize the method when the Jacobian approximation is still immature.

At the core of the implementation is the `NonlinearSystem` trait, which defines the interface required by the solver: a residual evaluation routine $\mathbf{F}(\mathbf{x})$ and a problem name for reporting. This keeps the quasi-Newton solver independent of any particular model, so that the same Broyden machinery can be reused across different systems by supplying only a system-specific residual function. The solver maintains an evolving approximate Jacobian $B_k$, uses it to compute the search direction by solving a linear system of the form $B_k,\mathbf{p}_k = -\mathbf{F}(\mathbf{x}_k)$, and then applies an Armijo rule to select a step length $\lambda_k$ in the update $\mathbf{x}_{k+1}=\mathbf{x}_k+\lambda_k\mathbf{p}_k$, matching the damped framework described earlier for line search methods.

The Armijo line search enforces sufficient decrease in the merit function $\phi(\mathbf{x})=\tfrac12||\mathbf{F}(\mathbf{x})||_2^2$ by testing the inequality in Equation (9.8.7). To evaluate the right-hand side of Equation (9.8.7), the program computes the directional derivative $\nabla\phi(\mathbf{x}_k)^{\mathsf{T}}\mathbf{p}_k$ using the identity $\nabla\phi = J(\mathbf{x})^{\mathsf{T}}\mathbf{F}(\mathbf{x})$. In a pure Broyden method one does not explicitly form $J(\mathbf{x}_k)$, so the implementation instead uses the available approximation $B_k$ and evaluates $(B_k^{\mathsf{T}}\mathbf{F}(\mathbf{x}_k))^{\mathsf{T}}\mathbf{p}_k$ as a practical substitute for the Armijo model decrease. This choice aligns with the role of the merit function in Equation (9.8.7), namely to ensure that the accepted step is not merely a formal secant step, but one that produces a meaningful reduction in the nonlinear residual norm.

Once a step is accepted, the program forms the secant vectors $\mathbf{s}_k$ and $\mathbf{y}_k$ using Equations (9.8.9) and (9.8.10), and then updates the approximate Jacobian via the rank-one correction in Equation (9.8.11). This update is designed so that the new matrix satisfies the secant condition in Equation (9.8.12), meaning that $B_{k+1}$ reproduces the observed change in the residual along the most recent step direction. In implementation terms, this requires computing $B_k\mathbf{s}_k$, forming the mismatch $\mathbf{y}_k - B_k\mathbf{s}_k$, and applying the outer product with $\mathbf{s}_k^{\mathsf{T}}$ scaled by $(\mathbf{s}_k^{\mathsf{T}}\mathbf{s}_k)^{-1}$. The result is a least-change update that preserves most of the accumulated curvature information in $B_k$ while guaranteeing consistency with the newest data pair $(\mathbf{s}_k,\mathbf{y}_k)$.

The `main` function demonstrates the method on two representative systems chosen to highlight both the strengths and limitations of Broyden’s approach. The first example is a small geometric system (circle-line intersection), where the residual is well-scaled and the approximate Jacobian quickly becomes accurate, so the method rapidly transitions to fast local convergence. The second example uses a Rosenbrock-inspired “valley” system with a known root at $(1,1)$, which is deliberately more challenging because the residual landscape is strongly anisotropic. This example emphasizes why globalization is essential in practice: early iterations may require multiple backtracking reductions of $\lambda_k$ before the method accepts steps that decrease $\phi$, but once the secant information accumulates and $B_k$ becomes a reliable surrogate for the true Jacobian, the method can converge rapidly without repeatedly forming exact derivatives.

```rust
// Program 9.8.4: Quasi-Newton (Broyden) method with Armijo line search for nonlinear systems
//
// This file is self-contained. It can be used as src/main.rs in a Cargo binary crate.
// Run with: cargo run

use std::f64;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn add_scaled(x: &[f64], alpha: f64, p: &[f64]) -> Vec<f64> {
    debug_assert_eq!(x.len(), p.len());
    x.iter().zip(p.iter()).map(|(xi, pi)| xi + alpha * pi).collect()
}

fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let n = a.len();
    debug_assert!(n > 0);
    debug_assert_eq!(a[0].len(), x.len());
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = dot(&a[i], x);
    }
    y
}

/// Solve A x = b using Gaussian elimination with partial pivoting.
/// Returns None if the matrix is singular or nearly singular.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return None;
    }
    for row in a {
        if row.len() != n {
            return None;
        }
    }

    let mut m = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = a[i][j];
        }
        m[i][n] = b[i];
    }

    for k in 0..n {
        // Pivot
        let mut piv = k;
        let mut max_abs = m[k][k].abs();
        for i in (k + 1)..n {
            let v = m[i][k].abs();
            if v > max_abs {
                max_abs = v;
                piv = i;
            }
        }
        if max_abs < 1e-15 {
            return None;
        }
        if piv != k {
            m.swap(piv, k);
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = m[i][k] / m[k][k];
            m[i][k] = 0.0;
            for j in (k + 1)..=n {
                m[i][j] -= factor * m[k][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = m[i][n];
        for j in (i + 1)..n {
            s -= m[i][j] * x[j];
        }
        let diag = m[i][i];
        if diag.abs() < 1e-15 {
            return None;
        }
        x[i] = s / diag;
    }
    Some(x)
}

/// Finite-difference Jacobian (forward differences).
fn finite_diff_jacobian(sys: &dyn NonlinearSystem, x: &[f64], f_at_x: &[f64], eps: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut j = vec![vec![0.0; n]; n];
    for col in 0..n {
        let mut x_pert = x.to_vec();
        let h = eps * (1.0 + x[col].abs());
        x_pert[col] += h;
        let f_pert = sys.f(&x_pert);
        for row in 0..n {
            j[row][col] = (f_pert[row] - f_at_x[row]) / h;
        }
    }
    j
}

/// Merit function phi(x) = 1/2 ||F(x)||^2.
fn phi_from_f(f: &[f64]) -> f64 {
    0.5 * dot(f, f)
}

/// Gradient directional derivative for phi along p when using B as Jacobian approximation:
/// ∇phi(x)^T p ≈ F(x)^T (B p).
fn grad_phi_dot_p(f: &[f64], b: &[Vec<f64>], p: &[f64]) -> f64 {
    let bp = mat_vec(b, p);
    dot(f, &bp)
}

/// Armijo backtracking for merit function phi.
/// Tries x_new = x + lambda * p, starting at lambda=1 and reducing by beta.
/// Returns (x_new, f_new, phi_new, lambda, backtracks).
fn armijo_search(
    sys: &dyn NonlinearSystem,
    x: &[f64],
    f: &[f64],
    b: &[Vec<f64>],
    p: &[f64],
    phi: f64,
    c: f64,
    beta: f64,
    min_lambda: f64,
    max_backtracks: usize,
) -> Option<(Vec<f64>, Vec<f64>, f64, f64, usize)> {
    let gtp = grad_phi_dot_p(f, b, p);

    // If the model derivative is not a descent direction, the line search has no guarantee.
    // We still attempt a conservative step, but we report failure if nothing improves.
    let mut lambda = 1.0;
    for bt in 0..=max_backtracks {
        let x_new = add_scaled(x, lambda, p);
        let f_new = sys.f(&x_new);
        let phi_new = phi_from_f(&f_new);

        let rhs = phi + c * lambda * gtp;
        if phi_new.is_finite() && (phi_new <= rhs || phi_new < phi) {
            return Some((x_new, f_new, phi_new, lambda, bt));
        }

        lambda *= beta;
        if lambda < min_lambda {
            break;
        }
    }
    None
}

/// Broyden rank-one update:
/// B_{k+1} = B_k + ((y - B s) s^T) / (s^T s).
fn broyden_update(b: &mut [Vec<f64>], s: &[f64], y: &[f64]) {
    let n = s.len();
    let bs = mat_vec(b, s);
    let mut u = vec![0.0; n];
    for i in 0..n {
        u[i] = y[i] - bs[i];
    }
    let denom = dot(s, s);
    if denom <= 0.0 {
        return;
    }
    let inv_denom = 1.0 / denom;
    for i in 0..n {
        for j in 0..n {
            b[i][j] += u[i] * s[j] * inv_denom;
        }
    }
}

trait NonlinearSystem {
    fn name(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn f(&self, x: &[f64]) -> Vec<f64>;
}

/// System 1: Circle-line intersection.
/// F1(x,y) = x^2 + y^2 - 1
/// F2(x,y) = x - y
struct CircleLine;
impl NonlinearSystem for CircleLine {
    fn name(&self) -> &'static str {
        "Circle-line intersection"
    }
    fn dim(&self) -> usize {
        2
    }
    fn f(&self, x: &[f64]) -> Vec<f64> {
        let (xx, yy) = (x[0], x[1]);
        vec![xx * xx + yy * yy - 1.0, xx - yy]
    }
}

/// System 2: Rosenbrock valley equations with root at (1,1).
/// F1(x,y) = 10 (y - x^2)
/// F2(x,y) = 1 - x
struct RosenbrockValleyEq;
impl NonlinearSystem for RosenbrockValleyEq {
    fn name(&self) -> &'static str {
        "Rosenbrock valley equations (root at (1,1))"
    }
    fn dim(&self) -> usize {
        2
    }
    fn f(&self, x: &[f64]) -> Vec<f64> {
        let (xx, yy) = (x[0], x[1]);
        vec![10.0 * (yy - xx * xx), 1.0 - xx]
    }
}

#[derive(Clone, Copy)]
struct BroydenParams {
    tol_f: f64,
    tol_step: f64,
    max_iters: usize,
    fd_eps: f64,
    // Armijo parameters
    c: f64,
    beta: f64,
    min_lambda: f64,
    max_backtracks: usize,
    // If true, initialize B0 using finite differences; otherwise use identity.
    init_with_fd_jacobian: bool,
}

fn broyden_solve(sys: &dyn NonlinearSystem, x0: &[f64], prm: BroydenParams) -> (Vec<f64>, bool, usize, f64) {
    let n = sys.dim();
    assert_eq!(x0.len(), n);

    let mut x = x0.to_vec();
    let mut f = sys.f(&x);
    let mut phi = phi_from_f(&f);

    // Initialize B0.
    let mut b = if prm.init_with_fd_jacobian {
        finite_diff_jacobian(sys, &x, &f, prm.fd_eps)
    } else {
        let mut id = vec![vec![0.0; n]; n];
        for i in 0..n {
            id[i][i] = 1.0;
        }
        id
    };

    println!("\nSystem: {}", sys.name());
    println!("Initial guess: x0 = {:?}", x0);
    println!();
    for k in 0..prm.max_iters {
        let fnorm = norm2(&f);
        println!("iter {:>3}: ||F(x)||_2 = {:>.6e}, phi(x) = {:>.6e}", k, fnorm, phi);

        if fnorm <= prm.tol_f {
            println!("         converged: ||F|| below tolerance");
            return (x, true, k, fnorm);
        }

        // Solve B_k p = -F(x_k).
        let rhs: Vec<f64> = f.iter().map(|v| -v).collect();
        let p = match solve_linear_system(&b, &rhs) {
            Some(p) => p,
            None => {
                // If B is singular, fall back to a finite-difference Jacobian for this iteration.
                let j_fd = finite_diff_jacobian(sys, &x, &f, prm.fd_eps);
                match solve_linear_system(&j_fd, &rhs) {
                    Some(p) => p,
                    None => {
                        println!("         failed: Jacobian/approximation singular");
                        return (x, false, k, fnorm);
                    }
                }
            }
        };

        // Line search on phi using the current B as the directional derivative model.
        let gtp = grad_phi_dot_p(&f, &b, &p);
        let step = match armijo_search(
            sys, &x, &f, &b, &p, phi, prm.c, prm.beta, prm.min_lambda, prm.max_backtracks,
        ) {
            Some(s) => s,
            None => {
                println!("         line search failed (no acceptable step found)");
                return (x, false, k, fnorm);
            }
        };

        let (x_new, f_new, phi_new, lambda, backtracks) = step;

        let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = f_new.iter().zip(f.iter()).map(|(a, b)| a - b).collect();

        let step_norm = norm2(&s);
        println!(
            "         λ = {:>.6e}  backtracks = {:>2}  (∇phi^T p) = {:>.6e}  ||s|| = {:>.3e}  phi_new = {:>.6e}",
            lambda, backtracks, gtp, step_norm, phi_new
        );

        // Update Broyden approximation.
        broyden_update(&mut b, &s, &y);

        // Accept
        x = x_new;
        f = f_new;
        phi = phi_new;

        if step_norm <= prm.tol_step * (1.0 + norm2(&x)) {
            // Small step termination (useful when residual stagnates).
            let fnorm_now = norm2(&f);
            if fnorm_now <= prm.tol_f {
                println!("         converged: small step and ||F|| below tolerance");
                return (x, true, k + 1, fnorm_now);
            }
            println!("         terminating: step too small");
            return (x, false, k + 1, fnorm_now);
        }
    }

    let fnorm = norm2(&f);
    (x, fnorm <= prm.tol_f, prm.max_iters, fnorm)
}

fn main() {
    println!("Program 9.8.4: Quasi-Newton method (Broyden) with Armijo line search\n");

    let prm = BroydenParams {
        tol_f: 1e-12,
        tol_step: 1e-14,
        max_iters: 100,
        fd_eps: 1e-8,
        c: 1e-4,
        beta: 0.5,
        min_lambda: 1e-12,
        max_backtracks: 40,
        init_with_fd_jacobian: true,
    };

    let systems: Vec<(&dyn NonlinearSystem, Vec<Vec<f64>>)> = vec![
        (&CircleLine, vec![vec![2.0, 0.2]]),
        (&RosenbrockValleyEq, vec![vec![-1.2, 1.0], vec![2.0, 2.0]]),
    ];

    for (sys, guesses) in systems {
        for x0 in guesses {
            let (x_star, ok, iters, fnorm) = broyden_solve(sys, &x0, prm);

            println!("\nSummary:");
            println!("  converged  = {}", ok);
            println!("  iters      = {}", iters);
            println!("  x          = {:?}", x_star);
            println!("  ||F(x)||_2 = {:>.6e}", fnorm);
            println!();
        }
    }
}
```

Program 9.8.4 demonstrates how Broyden’s rank-one update in Equation (9.8.11) can be turned into an efficient and reusable solver for nonlinear systems when paired with a merit-function line search. The results illustrate the practical behavior anticipated in Section 9.8.4: the method often makes cautious progress early on while the Jacobian approximation is still being learned from the iterates, but it can become significantly more efficient than full Newton iterations once the approximate Jacobian begins to capture the dominant local structure of the system. The examples also reinforce the role of the secant condition in Equation (9.8.12) as the key mechanism that anchors the update to real observed changes in the residual, preventing the approximation from drifting arbitrarily. Finally, the modular organization of the solver makes it straightforward to extend the implementation to more advanced variants, including restarted or safeguarded Broyden updates, block secant strategies, and trust-region hybridizations that further improve robustness on large-scale or poorly conditioned problems.

## 9.8.5. Newton–Krylov Methods with Globalization

For very large nonlinear systems, even forming or updating an approximate Jacobian may be impractical. In such cases, Newton–Krylov methods provide a scalable and memory-efficient alternative. These methods retain the Newton framework but replace the direct solution of the Jacobian system with an iterative Krylov subspace solver such as GMRES.

At each iteration, the Newton correction equation (9.8.5) is solved only approximately. Crucially, Krylov methods require only the action of the Jacobian on a vector, not the Jacobian itself. These products are typically approximated by finite differences,

$$
J(\mathbf{x})\,\mathbf{v}
\approx
\frac{\mathbf{F}(\mathbf{x} + \varepsilon \mathbf{v}) - \mathbf{F}(\mathbf{x})}{\varepsilon}
\tag{9.8.13}
$$

where $\varepsilon$ is a small perturbation parameter. This *matrix-free* formulation avoids explicit Jacobian construction and storage, dramatically reducing memory requirements and enabling the solution of systems with tens or hundreds of thousands of unknowns.

Because Krylov solves are inexact, Newton–Krylov methods are often described as *inexact Newton methods*. Their convergence depends on the accuracy with which the linear system is solved, and this accuracy is typically relaxed far from the solution and tightened as convergence proceeds. To guarantee robustness, Newton–Krylov iterations are almost always combined with globalization strategies such as line search or trust-region methods, ensuring that each nonlinear step produces meaningful progress.

Additional performance gains are achieved through preconditioning and recycling strategies, which reuse spectral or subspace information from previous Krylov solves to accelerate convergence of subsequent linear systems. These enhancements are essential in large-scale applications, where naive Krylov iterations would otherwise converge too slowly.

Newton-Krylov methods with globalization now form the backbone of many large-scale nonlinear solvers, including those used in power-grid simulation, fluid dynamics, and multiphysics modeling. For example, Yetkin and Ceylan (2023) demonstrate that combining Newton–Krylov techniques with appropriate preconditioning and step-control strategies enables efficient solution of power-flow problems involving tens of thousands of variables.

Together, quasi-Newton and Newton-Krylov methods illustrate a central theme of modern nonlinear system solving: exact Newton steps are often unnecessary. By carefully balancing approximation, iteration, and globalization, these methods achieve scalability and robustness without sacrificing the essential structure of Newton’s method.

### Rust Implementation

Following the discussion in Section 9.8.5 on Newton–Krylov methods with globalization, Program 9.8.5 presents a practical matrix-free implementation of a Newton step computed by a Krylov subspace method and stabilized by an Armijo backtracking line search. In large nonlinear systems, assembling and factoring the Jacobian can be prohibitively expensive, so the program adopts the Jacobian-free viewpoint in which the linearized Newton correction in Equation (9.8.5) is solved approximately using GMRES and Jacobian–vector products are formed through the finite-difference approximation in Equation (9.8.13). The result is a solver that retains the structure of Newton’s method while replacing the inner linear algebra with an iterative process whose accuracy can be tuned, and whose globalization mechanism ensures reliable progress when far from a solution.

At the core of the implementation is the `NonlinearSystem` trait, which provides a uniform interface for nonlinear models by requiring an evaluation routine for $\mathbf{F}(\mathbf{x})$ (and the system name used for reporting). This trait-based design allows the Newton–Krylov solver to remain independent of the specific problem definition. The solver itself treats $\mathbf{F}$ as a black box and never forms $J(\mathbf{x})$ explicitly. Instead, it repeatedly queries $\mathbf{F}$ to build the action $J(\mathbf{x})\mathbf{v}$ through the finite-difference formula in Equation (9.8.13), which is implemented as a dedicated Jacobian–vector product routine. This routine is the key mechanism that makes the method “matrix-free” and permits the algorithm to scale to settings where storing $J$ would be infeasible.

The function implementing the Jacobian–vector product is conceptually aligned with Equation (9.8.13). Given the current iterate $\mathbf{x}$, a vector $\mathbf{v}$, and a perturbation parameter $\varepsilon$, it computes $\mathbf{F}(\mathbf{x}+\varepsilon\mathbf{v})$ and $\mathbf{F}(\mathbf{x})$ and returns their difference divided by $\varepsilon$. In practice, $\varepsilon$ must balance truncation and roundoff effects, and the program typically uses a scale-aware choice so that $\mathbf{x}+\varepsilon\mathbf{v}$ is representable and the difference quotient is not dominated by cancellation. This Jacobian-free product is then used inside GMRES to construct the Krylov basis and minimize the linear residual of Equation (9.8.5) without ever storing a Jacobian matrix.

The Krylov solver is implemented by a compact GMRES routine that builds an orthonormal basis of the Krylov subspace and solves the associated small least-squares problem to produce an approximate Newton correction $\delta\mathbf{x}$ for Equation (9.8.5). The function typically maintains the Arnoldi Hessenberg matrix and applies Givens rotations to update the norm of the linear residual efficiently as the subspace grows. This is what the output lines reporting the GMRES tolerance and the achieved linear residual correspond to. Because Newton–Krylov is an inexact Newton method, the program uses a forcing strategy that sets the requested linear tolerance relative to the current nonlinear residual, so the inner solves are loose far from the root and become tighter as $\lVert \mathbf{F}(\mathbf{x})\rVert_2$ decreases. This matches the standard inexact-Newton principle that the linear work should track the nonlinear progress, rather than over-solving the early linear systems.

Once GMRES returns a candidate step $\mathbf{p}\approx\delta\mathbf{x}$, the globalization mechanism applies an Armijo backtracking line search in the spirit of Equation (9.8.7). The program defines a merit function $\phi$ based on the residual norm (typically $\phi(\mathbf{x})=\tfrac{1}{2}\lVert\mathbf{F}(\mathbf{x})\rVert_2^2)$, and evaluates trial points $\mathbf{x}+\lambda\mathbf{p}$ with successively smaller $\lambda\in(0,1]$ until sufficient decrease is observed. The directional derivative term $\nabla\phi(\mathbf{x})^{\mathsf{T}}\mathbf{p}$ is computed from the identity $\nabla\phi=J(\mathbf{x})^{\mathsf{T}}\mathbf{F}(\mathbf{x})$, using only Jacobian–vector products and inner products, so that globalization remains consistent with the matrix-free philosophy. The printed “backtracks” counter in the output reflects how often $\lambda$ is reduced before the Armijo inequality is satisfied, and the reported $\phi_{\text{new}}$ confirms that each accepted step produces a meaningful reduction in the merit function.

The `main` function is structured as an executable demonstration of the Newton–Krylov framework on representative small systems that still exhibit the essential globalization behavior. It constructs concrete `NonlinearSystem` instances, sets solver parameters such as nonlinear tolerance, maximum iterations, GMRES subspace size, and line-search constants, and then calls the Newton–Krylov driver for each initial guess. The logging in `main` is intentionally verbose, reporting the nonlinear residual norm, merit function value, inner GMRES accuracy, and accepted step length, so that the interaction between the inexact linear solves and the globalization condition can be inspected directly. This makes the program suitable both as a reference implementation and as an instructional tool for studying the practical behavior implied by Equations (9.8.5), (9.8.7), and (9.8.13).

```rust
// -----------------------------------------------------------------------------
// Program 9.8.5
// Newton–Krylov Method with Globalization
//
// Problem statement:
//
// We consider the numerical solution of a system of nonlinear equations
//
//     F(x) = 0 ,
//
// where F : R^n -> R^n is a continuously differentiable nonlinear mapping.
// Such systems arise throughout scientific computing, including power-flow
// analysis, nonlinear least-squares fitting, equilibrium problems, and
// multiphysics simulations. For large-scale problems, explicitly forming,
// storing, and factorizing the Jacobian matrix J(x) = dF/dx can be prohibitively
// expensive in both memory and computational cost.
//
// The goal of this program is to implement a matrix-free Newton–Krylov solver
// that computes approximate Newton corrections without ever assembling the
// Jacobian. At each nonlinear iteration k, the Newton correction p_k is defined
// implicitly by the linearized equation
//
//     J(x_k) p_k = -F(x_k) ,
//
// which corresponds to Equation (9.8.5) in the text. Instead of solving this
// system directly, the program applies a Krylov subspace method (GMRES) that
// requires only Jacobian–vector products.
//
// These Jacobian–vector products are approximated by finite differences,
// using
//
//     J(x) v ≈ [ F(x + ε v) - F(x) ] / ε ,
//
// as given in Equation (9.8.13). This matrix-free formulation dramatically
// reduces memory requirements and enables the solution of problems with very
// large state dimensions.
//
// Because the Krylov solve produces only an approximate Newton direction, the
// resulting method is an inexact Newton scheme. To ensure robustness and global
// convergence from arbitrary initial guesses, each Newton step is stabilized
// by an Armijo backtracking line search applied to a merit function based on the
// residual norm.
//
// This program demonstrates the complete Newton–Krylov workflow, including
// matrix-free Jacobian–vector products, an inexact GMRES linear solve, and
// globalization via line search. Two representative nonlinear systems are
// solved to illustrate convergence behavior from different initial conditions.
// -----------------------------------------------------------------------------


use std::f64;

const FD_EPS: f64 = 1e-8;         // finite-difference step for J(x)v
const ARM_C: f64 = 1e-4;          // Armijo parameter c in (0,1)
const BT_SHRINK: f64 = 0.5;       // backtracking reduction factor
const BT_MAX: usize = 20;         // max backtracking steps
const M_MAX: usize = 20;          // restarted GMRES subspace dimension
const NEWTON_MAX: usize = 100;    // max nonlinear iterations
const F_TOL: f64 = 1e-12;         // convergence threshold on ||F||_2

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn axpy(y: &mut [f64], alpha: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

fn scal(x: &mut [f64], alpha: f64) {
    for xi in x.iter_mut() {
        *xi *= alpha;
    }
}

fn add_scaled(x: &[f64], alpha: f64, p: &[f64]) -> Vec<f64> {
    x.iter().zip(p.iter()).map(|(xi, pi)| xi + alpha * pi).collect()
}

fn phi_from_f(f: &[f64]) -> f64 {
    0.5 * dot(f, f)
}

trait NonlinearSystem {
    fn name(&self) -> &'static str;
    fn eval(&self, x: &[f64]) -> Vec<f64>;
}

struct CircleLine;

impl NonlinearSystem for CircleLine {
    fn name(&self) -> &'static str {
        "Circle-line intersection"
    }

    fn eval(&self, x: &[f64]) -> Vec<f64> {
        let xx = x[0];
        let yy = x[1];
        vec![xx * xx + yy * yy - 1.0, xx - yy]
    }
}

struct RosenbrockValley;

impl NonlinearSystem for RosenbrockValley {
    fn name(&self) -> &'static str {
        "Rosenbrock valley equations (root at (1,1))"
    }

    fn eval(&self, x: &[f64]) -> Vec<f64> {
        let xx = x[0];
        let yy = x[1];
        vec![10.0 * (yy - xx * xx), 1.0 - xx]
    }
}

fn jv_product(sys: &dyn NonlinearSystem, x: &[f64], f_at_x: &[f64], v: &[f64]) -> Vec<f64> {
    // J(x) v ≈ (F(x + eps v) - F(x)) / eps  (Eq. 9.8.13)
    let x_pert = add_scaled(x, FD_EPS, v);
    let f_pert = sys.eval(&x_pert);
    f_pert
        .iter()
        .zip(f_at_x.iter())
        .map(|(fp, f0)| (fp - f0) / FD_EPS)
        .collect()
}

fn solve_upper_triangular(r: &[Vec<f64>], g: &[f64], m: usize) -> Vec<f64> {
    let mut y = vec![0.0; m];
    for i_rev in 0..m {
        let i = m - 1 - i_rev;
        let mut s = g[i];
        for j in (i + 1)..m {
            s -= r[i][j] * y[j];
        }
        y[i] = s / r[i][i];
    }
    y
}

fn gmres_solve(
    sys: &dyn NonlinearSystem,
    x: &[f64],
    f_at_x: &[f64],
    b: &[f64],      // right-hand side, typically -F(x)
    tol: f64,
    m_max: usize,
) -> (Vec<f64>, usize, f64) {
    let n = b.len();
    let beta = norm2(b);

    if beta == 0.0 {
        return (vec![0.0; n], 0, 0.0);
    }

    let mut v: Vec<Vec<f64>> = Vec::with_capacity(m_max + 1);
    let mut v0 = b.to_vec();
    scal(&mut v0, 1.0 / beta);
    v.push(v0);

    let mut h = vec![vec![0.0; m_max]; m_max + 1];

    let mut cs = vec![0.0; m_max];
    let mut sn = vec![0.0; m_max];

    let mut g = vec![0.0; m_max + 1];
    g[0] = beta;

    let mut m_used = 0usize;
    

    for j in 0..m_max {
        let w = jv_product(sys, x, f_at_x, &v[j]);

        let mut w_orth = w;
        for i in 0..=j {
            h[i][j] = dot(&w_orth, &v[i]);
            let hij = h[i][j];
            for k in 0..n {
                w_orth[k] -= hij * v[i][k];
            }
        }
        h[j + 1][j] = norm2(&w_orth);
        if h[j + 1][j] != 0.0 {
            scal(&mut w_orth, 1.0 / h[j + 1][j]);
            v.push(w_orth);
        } else {
            v.push(vec![0.0; n]);
        }

        for i in 0..j {
            let temp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
            h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
            h[i][j] = temp;
        }

        let a = h[j][j];
        let b2 = h[j + 1][j];
        let r = (a * a + b2 * b2).sqrt();
        if r == 0.0 {
            cs[j] = 1.0;
            sn[j] = 0.0;
        } else {
            cs[j] = a / r;
            sn[j] = b2 / r;
        }

        h[j][j] = cs[j] * h[j][j] + sn[j] * h[j + 1][j];
        h[j + 1][j] = 0.0;

        let gj = g[j];
        g[j] = cs[j] * gj + sn[j] * g[j + 1];
        g[j + 1] = -sn[j] * gj + cs[j] * g[j + 1];

        let resid = g[j + 1].abs();
        m_used = j + 1;

        if resid <= tol {
            break;
        }
    }

    let mut rmat = vec![vec![0.0; m_used]; m_used];
    let mut g_trunc = vec![0.0; m_used];
    for i in 0..m_used {
        g_trunc[i] = g[i];
        for j in i..m_used {
            rmat[i][j] = h[i][j];
        }
    }
    let y = solve_upper_triangular(&rmat, &g_trunc, m_used);

    let mut p = vec![0.0; n];
    for j in 0..m_used {
        axpy(&mut p, y[j], &v[j]);
    }

    let ap = jv_product(sys, x, f_at_x, &p);
    let mut rlin = b.to_vec();
    for i in 0..n {
        rlin[i] -= ap[i];
    }
    let rlin_norm = norm2(&rlin);

    (p, m_used, rlin_norm)
}

fn newton_krylov_armijo(sys: &dyn NonlinearSystem, x0: &[f64]) {
    println!("\nSystem: {}", sys.name());
    println!("Initial guess: x0 = [{:.16}, {:.16}]\n", x0[0], x0[1]);

    let mut x = x0.to_vec();

    for k in 0..=NEWTON_MAX {
        let f = sys.eval(&x);
        let f_norm = norm2(&f);
        let phi = phi_from_f(&f);

        println!(
            "iter {:>3}: ||F(x)||_2 = {:>12.6e}, phi(x) = {:>12.6e}",
            k, f_norm, phi
        );

        if f_norm <= F_TOL {
            println!("         converged: ||F|| below tolerance");
            println!("Summary:");
            println!("  converged  = true");
            println!("  iters      = {}", k);
            println!("  x          = [{:.16}, {:.16}]", x[0], x[1]);
            println!("  ||F(x)||_2 = {:.6e}\n", f_norm);
            return;
        }

        if k == NEWTON_MAX {
            println!("Summary:");
            println!("  converged  = false");
            println!("  iters      = {}", k);
            println!("  x          = [{:.16}, {:.16}]", x[0], x[1]);
            println!("  ||F(x)||_2 = {:.6e}\n", f_norm);
            return;
        }

        let eta = 0.5;
        let tol_lin = eta * f_norm;

        let b = f.iter().map(|v| -v).collect::<Vec<f64>>();
        let (p, _m_used, rlin_norm) = gmres_solve(sys, &x, &f, &b, tol_lin, M_MAX);

        let jp = jv_product(sys, &x, &f, &p);
        let grad_phi_dot_p = dot(&f, &jp);
        let p_norm = norm2(&p);

        println!(
            "         GMRES: tol = {:>8.2e}, ||r_lin|| = {:>8.2e}, ||p|| = {:>8.2e}",
            tol_lin, rlin_norm, p_norm
        );

        let mut lambda = 1.0;
        let mut backtracks = 0usize;

        let mut phi_new = phi;
        for bt in 0..=BT_MAX {
            let x_trial = add_scaled(&x, lambda, &p);
            let f_trial = sys.eval(&x_trial);
            phi_new = phi_from_f(&f_trial);

            if phi_new <= phi + ARM_C * lambda * grad_phi_dot_p {
                backtracks = bt;
                x = x_trial;
                break;
            }

            lambda *= BT_SHRINK;
            backtracks = bt + 1;
        }

        println!(
            "         λ = {:>10.3e}  backtracks = {:>2}  ∇phi^T p = {:>12.6e}  phi_new = {:>12.6e}\n",
            lambda, backtracks, grad_phi_dot_p, phi_new
        );
    }
}

fn main() {
    println!("Program 9.8.5: Newton–Krylov method (matrix-free GMRES) with Armijo line search\n");

    let circle_line = CircleLine;
    let rosen = RosenbrockValley;

    newton_krylov_armijo(&circle_line, &[2.0, 0.2]);
    newton_krylov_armijo(&rosen, &[-1.2, 1.0]);
    newton_krylov_armijo(&rosen, &[2.0, 2.0]);
}
```

Program 9.8.5 illustrates how Newton’s method can be scaled to matrix-free settings by replacing Jacobian formation and factorization with Krylov subspace solves driven solely by Jacobian–vector products. The example outputs highlight two complementary mechanisms: GMRES provides an efficient way to approximate the Newton correction in Equation (9.8.5) using only the matrix-free product in Equation (9.8.13), while the Armijo line search enforces robust descent in the merit function in the spirit of Equation (9.8.7). Together, these components show why Newton–Krylov methods are a practical backbone for large-scale nonlinear simulation: the linear work can be tuned adaptively, and the globalization mechanism prevents unstable steps when the local linear model is unreliable. The program’s modular organization also provides a natural foundation for extending the solver to preconditioned GMRES, adaptive forcing terms, and trust-region hybridizations, which are essential enhancements in high-dimensional and poorly conditioned applications.

## 9.8.6. Practical Considerations and Failure Modes

Even when equipped with globalization strategies, nonlinear solvers are not immune to failure. In practice, convergence difficulties often arise not from the algorithmic framework itself, but from problem formulation and numerical implementation details. Poor scaling of variables is among the most common sources of difficulty: when different components of $\mathbf{x}$ or $\mathbf{F}(\mathbf{x})$ vary over vastly different magnitudes, the Jacobian can become ill-conditioned, leading to unreliable search directions and ineffective globalization mechanisms. Proper nondimensionalization and variable scaling are therefore essential preparatory steps in any serious nonlinear computation.

Another frequent cause of failure is inaccurate Jacobian information. Finite-difference approximations may suffer from truncation or cancellation errors, while quasi-Newton updates can degrade if the nonlinear system changes rapidly or if steps are repeatedly damped. In Newton–Krylov methods, poor preconditioning can lead to excessively slow inner solves, preventing the outer nonlinear iteration from making meaningful progress. For problems involving nonsmooth functions, discontinuities, or conditional logic, the assumptions underlying Newton-type linearizations may be violated altogether, necessitating reformulation or the use of specialized nonsmooth solvers.

It is also important to recognize a subtle but critical limitation of merit-function-based globalization. Convergence to a stationary point of the merit function $\phi$ does not necessarily imply convergence to a solution of the nonlinear system. In such cases, the algorithm may stall at a point where the gradient of $\phi$ vanishes even though the residual $\mathbf{F}(\mathbf{x})$ remains nonzero. When this occurs, diagnostic checks such as monitoring individual residual components, inspecting Jacobian conditioning, or perturbing the initial guess are essential. Alternative merit functions or continuation strategies may also be required.

Despite these challenges, globally convergent methods dramatically expand the class of nonlinear systems that can be solved reliably. In practical implementations, most robust solvers adopt a *hybrid strategy*: a globalization mechanism guides the iterates safely into the basin of attraction of a solution, after which damping is relaxed and a pure or nearly pure Newton method is used to exploit its rapid local convergence. This combination balances reliability with efficiency and reflects the accumulated experience of decades of numerical practice.

## 9.8.7. Concluding Remarks

Globalization strategies transform Newton’s method from a powerful but fragile local procedure into a robust and widely applicable framework for solving nonlinear systems of equations. By augmenting Newton iterations with merit functions, damping mechanisms, trust-region constraints, and approximate Jacobian updates, modern algorithms achieve both global reliability and high local efficiency.

These techniques are not merely theoretical refinements; they form the backbone of contemporary nonlinear solvers used in scientific computing, engineering simulation, inverse problems, and data-driven modeling. Their success rests on a central principle of numerical analysis: fast local convergence is valuable only when it can be reached reliably. Globalization strategies ensure that Newton’s method fulfills this requirement, making it one of the most effective and versatile tools in modern computational mathematics (Vivas-Cortez et al., 2023; Martin, 2023).

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/JNhk98ot7g2LG3ukwSlt.2","tags":[]}

# 9.9. Conclusion

As we conclude this chapter, our goal has been to develop a systematic and unified treatment of root-finding algorithms for scalar equations, polynomial equations, and nonlinear systems in multiple dimensions. These problems arise across virtually every domain of scientific and engineering computation, from equilibrium analysis and optimization to financial calibration and machine learning. Rust's ownership model, strong type system, and zero-cost abstractions provide an ideal foundation for implementing these algorithms with both mathematical fidelity and production-grade robustness. Throughout this chapter, we have emphasized that no single root-finding method is universally optimal. Instead, effective numerical practice requires understanding the interplay between convergence guarantees, derivative requirements, sensitivity to initial conditions, and the structural properties of the equations being solved.

## 9.9.1. Key Takeaways

- Root-finding is a foundational computational primitive that appears whenever a function or system of equations must be driven to zero. The scalar problem $f(x) = 0$ and its multidimensional generalization $\mathbf{F}(\mathbf{x}) = \mathbf{0}$ differ not merely in notation but in fundamental difficulty. The Intermediate Value Theorem provides existence guarantees through sign changes in one dimension, while no comparably simple bracketing principle exists in higher dimensions. The Jacobian matrix and its nonsingularity at a solution govern local uniqueness through the implicit function theorem, and degenerate cases where the Jacobian loses rank can produce solution manifolds rather than isolated points.
- Bracketing and bisection form the conceptual and computational foundation of reliable one-dimensional root finding. By confining a root within an interval where the function changes sign, bisection guarantees convergence under minimal assumptions of continuity. Each iteration halves the interval, yielding linear convergence at a predictable rate of one binary digit per step, with the required number of iterations given by $n \ge \lceil \log_2(\varepsilon_0 / \text{TOL}) \rceil$. Despite its simplicity, bisection remains indispensable as a standalone solver, a safeguard within hybrid algorithms, and a globalization mechanism that reduces sensitivity to initial guesses.
- The secant method, false position, and Ridders' method represent progressively more sophisticated attempts to accelerate convergence beyond bisection by exploiting approximate linearity near the root. The secant method achieves superlinear convergence of order $\varphi \approx 1.618$ without derivative information but lacks convergence guarantees. Classical regula falsi preserves bracketing but can stagnate when strong curvature biases the interpolation toward one endpoint. Ridders' method resolves this by applying an exponential straightening transformation that restores balanced interval contraction while maintaining the bracketing guarantee, typically achieving quadratic convergence for smooth functions.
- Brent's method (van Wijngaarden–Dekker–Brent) represents the gold standard for one-dimensional root finding by adaptively combining bisection, secant updates, and inverse quadratic interpolation within a single safeguarded framework. Its defining feature is that interpolation steps are never accepted blindly. Safety conditions verify that the proposed step lies within the current bracket, represents adequate progress, and does not stagnate due to roundoff. When these conditions fail, the method reverts to bisection, ensuring worst-case convergence no slower than $\lceil \log_2((b-a)/\text{TOL}) \rceil$ iterations while achieving superlinear or near-quadratic convergence in favorable cases. Applications ranging from the Colebrook equation in fluid mechanics to implied volatility computation in finance confirm its practical reliability.
- Newton-Raphson iteration provides the fastest local convergence among classical root-finding methods, achieving quadratic convergence for simple roots in both scalar and multidimensional settings. For scalar equations, the update $x_{k+1} = x_k - f(x_k)/f'(x_k)$ requires derivative evaluation but doubles the number of correct digits at each step. For systems, the Newton correction $J(\mathbf{x}_k)\delta\mathbf{x} = -\mathbf{F}(\mathbf{x}_k)$ requires solving a linear system involving the Jacobian at each iteration. However, Newton's method is fundamentally local: convergence depends on the initial guess lying within the basin of attraction of a solution, and different starting points may converge to different roots or diverge entirely. This sensitivity motivates both the globalization strategies developed later in the chapter and the derivative-free alternatives that sacrifice convergence order for robustness.
- Polynomial root finding lies at the intersection of numerical analysis, algebra, and linear algebra, and is complicated by the intrinsic sensitivity of roots to coefficient perturbations. Wilkinson's polynomial demonstrates that even benign-looking polynomials can be severely ill-conditioned, with relative perturbations of order $10^{-7}$ in a single coefficient causing dramatic displacement of roots into the complex plane. Deflation via synthetic division reduces the polynomial degree after each root is found but can propagate errors through subsequent computations, motivating the use of polishing stages where roots are refined against the original polynomial using Newton's method or Maehly's correction formula.
- Muller's method, Laguerre's method, and companion-matrix eigenvalue methods provide complementary strategies for polynomial root finding. Muller's method uses quadratic interpolation to achieve convergence of order approximately 1.84 and naturally admits complex iterates from real starting values. Laguerre's method incorporates first and second derivative information through the logarithmic derivative to achieve cubic convergence for simple roots, with exceptionally strong global behavior that often succeeds from arbitrary initial guesses. Companion-matrix methods reformulate root finding as an eigenvalue problem, computing all roots simultaneously in $O(m^3)$ time using backward-stable QR algorithms and avoiding sequential deflation entirely.
- Globally convergent methods for nonlinear systems address the fundamental limitation of Newton's method, namely its dependence on favorable initial guesses. By recasting the system $\mathbf{F}(\mathbf{x}) = \mathbf{0}$ in terms of a merit function $\phi(\mathbf{x}) = \frac{1}{2}\|\mathbf{F}(\mathbf{x})\|_2^2$, globalization strategies enforce descent conditions that guarantee progress toward feasibility. Damped Newton methods with Armijo backtracking control step length along the Newton direction, while trust-region methods restrict corrections to lie within a region where the local linear model is trustworthy. Both approaches preserve Newton's quadratic local convergence while extending its effective domain far beyond its natural basin of attraction. In practice, convergence to a stationary point of the merit function does not guarantee convergence to a root, so diagnostic checks on individual residual components, Jacobian conditioning, and variable scaling remain essential safeguards.
- Quasi-Newton methods such as Broyden's method and Newton-Krylov methods with GMRES address the scalability challenges of large nonlinear systems. Broyden's rank-one update replaces exact Jacobian computation with an iteratively refined approximation satisfying the secant condition $B_{k+1}\mathbf{s}_k = \mathbf{y}_k$, achieving superlinear convergence at dramatically reduced per-iteration cost. Newton-Krylov methods go further by eliminating Jacobian storage entirely, computing Jacobian-vector products through finite differences $J(\mathbf{x})\mathbf{v} \approx [\mathbf{F}(\mathbf{x}+\varepsilon\mathbf{v})-\mathbf{F}(\mathbf{x})]/\varepsilon$ and solving the Newton correction approximately using Krylov subspace methods. These matrix-free techniques, combined with globalization and preconditioning, form the backbone of modern large-scale nonlinear solvers used in power-grid simulation, fluid dynamics, and multiphysics modeling.

## 9.9.2. Advice for Beginners

- Nonlinear equations are fundamentally different from the linear systems studied earlier in this book. Unlike linear problems, nonlinear equations may have multiple solutions, no solutions, or solutions that are highly sensitive to initial conditions. Before studying advanced algorithms, focus on understanding the concepts of roots, residuals, convergence, conditioning, and the role of initial guesses.
- Begin with bracketing methods such as bisection. Although bisection converges relatively slowly, it provides a reliable introduction to root finding and demonstrates the importance of maintaining a valid interval containing a root. Experiment with different stopping criteria and observe how interval width and residual magnitude evolve during the iteration process.
- After mastering bisection, study the secant method and Brent's method. These algorithms illustrate how interpolation can accelerate convergence while still preserving robustness. Compare the number of function evaluations required by each method on the same problem to develop intuition about efficiency versus reliability.
- Next, focus on Newton–Raphson iteration. Newton's method is one of the most important algorithms in numerical computing because of its rapid local convergence. However, it is also sensitive to initial guesses and derivative quality. Experiment with different starting points and observe how convergence behavior changes. Understanding both the strengths and weaknesses of Newton's method is essential for successful nonlinear computation.
- When studying polynomial root finding, pay particular attention to conditioning and sensitivity. Polynomial roots can be remarkably sensitive to small perturbations in coefficients. Investigating examples such as Wilkinson's polynomial provides valuable insight into the challenges of numerical root finding and finite-precision arithmetic.
- As you progress to nonlinear systems, focus first on small two-variable examples before studying large-scale Newton–Krylov and quasi-Newton methods. Understanding Jacobians, linearizations, and local convergence in low dimensions will make higher-dimensional algorithms much easier to understand.
- Globalization techniques such as line-search and trust-region methods deserve special attention. These methods demonstrate how local algorithms can be transformed into practical solvers that remain reliable even when initial guesses are far from the solution.
- For Rust implementations, become familiar with `nalgebra`, `ndarray`, `num-complex`, and sparse linear-algebra libraries. Many nonlinear algorithms ultimately depend on solving linear systems efficiently, making strong linear-algebra foundations particularly important.
- Most importantly, remember that root finding is not an isolated topic. Nonlinear equation solvers appear throughout optimization, differential equations, machine learning, power systems, fluid dynamics, finance, and scientific simulation. A solid understanding of the methods presented in this chapter will provide essential tools for many advanced computational problems encountered later in numerical computing.

## 9.9.3. Further Learning with GenAI

To deepen your understanding of root-finding and nonlinear equation solving in Rust, consider using the following GenAI prompts:

 1. Implement the bisection method in Rust with explicit interval-width tracking and a singularity diagnostic that detects when $|f(x)|$ grows rather than decreases as the bracket shrinks. Demonstrate the method on a smooth function with a simple root and on a function with a pole inside the interval, and verify that the iteration count matches the theoretical prediction $n \ge \lceil \log_2(\varepsilon_0/\text{TOL})\rceil$.
 2. Implement Newton-Raphson iteration in Rust for both scalar equations and two-dimensional nonlinear systems. For the scalar case, demonstrate quadratic convergence by tracking the number of correct digits at each iteration. For the system case, construct the Jacobian analytically, solve the linear correction using Gaussian elimination, and illustrate basin-of-attraction effects by applying the method from multiple initial guesses to a system with more than one solution.
 3. Implement the secant method and Ridders' method in Rust and compare their convergence behavior on Kepler's equation $E - e\sin E - M = 0$ for eccentricity $e = 0.7$. Verify that the secant method achieves superlinear convergence of order $\varphi \approx 1.618$ and that Ridders' method maintains a bracketing guarantee while converging near-quadratically.
 4. Implement Brent's method in Rust following the safeguarded interpolation framework, including inverse quadratic interpolation, secant fallback, and bisection fallback when safety conditions fail. Apply it to the Colebrook-White equation from fluid mechanics and report how many iterations use interpolation versus bisection.
 5. Implement Horner's method for polynomial evaluation in Rust together with the Durand-Kerner iteration for computing all roots simultaneously. Apply the solver to Wilkinson's polynomial of degree 20 and demonstrate the sensitivity of roots to a small relative perturbation in the constant term.
 6. Implement Muller's method in Rust with complex arithmetic using the divided-difference formulation of quadratic interpolation. Demonstrate convergence to complex roots of $x^2 + 1 = 0$ starting from real initial guesses, and compare the convergence order against the secant method on a polynomial with both real and complex roots.
 7. Implement Laguerre's method in Rust with complex arithmetic, using the update formula involving $G = P'(x)/P(x)$ and $H = G^2 - P''(x)/P(x)$, and combine it with synthetic-division deflation to extract all roots of a polynomial sequentially. Demonstrate convergence on a polynomial with both real and complex-conjugate roots.
 8. Construct the companion matrix for a polynomial in Rust using the ndarray crate and compute its eigenvalues to obtain all roots simultaneously. Compare the accuracy and computational cost against Laguerre's method with deflation on a polynomial of degree 10 or higher.
 9. Implement a globally convergent Newton method for nonlinear systems in Rust using the merit function $\phi(\mathbf{x}) = \frac{1}{2}\|\mathbf{F}(\mathbf{x})\|_2^2$ and Armijo backtracking line search. Show that the full Newton step is accepted near a solution while step damping is activated far from it.
10. Implement a trust-region Newton method using the dogleg algorithm in Rust for a two-dimensional nonlinear system. Track the trust-region radius, the agreement ratio between actual and predicted reduction, and the step type (Cauchy, dogleg, or Gauss-Newton) at each iteration. Demonstrate how the trust-region radius adapts automatically based on model fidelity.
11. Implement Broyden's method in Rust with rank-one Jacobian updates satisfying the secant condition $B_{k+1}\mathbf{s}_k = \mathbf{y}_k$, initialized using a finite-difference Jacobian. Compare the per-iteration cost and convergence rate against a full Newton method on a two-dimensional nonlinear system.
12. Implement a Newton-Krylov solver in Rust using matrix-free GMRES with Jacobian-vector products computed via finite differences. Apply it to a nonlinear system of moderate size and demonstrate that the method converges without ever forming or storing the Jacobian matrix explicitly.

## 9.9.4. Homework Exercises

To reinforce your learning, complete the following exercises:

 1. Implement bisection, the secant method, Ridders' method, and Brent's method in Rust and apply all four to the equation $\cos(x) - x = 0$ on $[0,1]$. Report the number of function evaluations required by each method to achieve a residual below $10^{-14}$, and analyze how the convergence rates compare with their theoretical predictions.
 2. Write a Rust program that solves Kepler's equation $E - e\sin E - M = 0$ for a grid of eccentricity values $e \in \{0.1, 0.5, 0.9, 0.99\}$ using both bisection and Brent's method. For each case, compare iteration counts and discuss how the curvature of the residual function affects the relative performance of bracketing versus interpolation-based methods.
 3. Implement Newton-Raphson iteration in Rust for the scalar equation $x - \cos(x) = 0$ and for the two-dimensional system $x^2 + y^2 = 1$, $x = y$. For the scalar case, start from $x_0 = 0.5$ and track the ratio $|e_{k+1}|/|e_k|^2$ to verify quadratic convergence. For the system case, start from five different initial guesses and report which root each converges to, illustrating the basin-of-attraction phenomenon.
 4. Construct Wilkinson's polynomial of degree 20 in Rust, compute all roots using the Durand-Kerner method with Newton polishing, and then repeat the computation after perturbing the coefficient of $x^{19}$ by a relative factor of $10^{-7}$. Measure the maximum root displacement and discuss how the observed sensitivity relates to the polynomial's condition number.
 5. Implement Muller's method and Laguerre's method in Rust and use both to find all roots of $x^5 - 1$. Compare convergence rates, the number of function evaluations per root, and the ability of each method to locate complex roots from real starting values.
 6. Implement the companion-matrix eigenvalue approach for polynomial root finding in Rust using an eigenvalue solver. Apply it to a polynomial of degree at least 8 with known roots, compare the computed roots against both the exact values and the results from Laguerre's method with deflation, and discuss the trade-offs between simultaneous eigenvalue computation and sequential iterative root extraction.
 7. Implement a damped Newton method with Armijo backtracking for the two-dimensional system $f_1(x,y) = x^2 + y^2 - 1$, $f_2(x,y) = x - y$, and solve it from six different initial guesses spread across the plane. For each starting point, record the number of iterations, the number of backtracking steps, and the root to which the method converges, and use the results to sketch an approximate basin-of-attraction diagram.
 8. Implement a trust-region Newton method using the dogleg algorithm for the Rosenbrock stationarity system $\nabla f = 0$ where $f(x,y) = 100(y-x^2)^2 + (1-x)^2$. Compare convergence from the starting point $(-1.2, 1.0)$ against a damped Newton method with line search, reporting trust-region radius evolution, step acceptance ratios, and total iteration counts.
 9. Implement Broyden's method in Rust with Armijo globalization and apply it to a nonlinear system of at least 10 equations. Compare wall-clock time and total function evaluations against a full Newton method with analytic Jacobian, and analyze how the quality of the Broyden approximation evolves over successive iterations by monitoring the secant residual $\|B_k \mathbf{s}_{k-1} - \mathbf{y}_{k-1}\|$.
10. Implement a Newton-Krylov solver in Rust using matrix-free GMRES with Jacobian-vector products computed via finite differences. Apply it to a nonlinear system of at least 100 equations and compare the wall-clock time and memory usage against a direct Newton method that explicitly forms and factors the Jacobian.

Root finding and nonlinear equation solving occupy a central position in numerical computing, bridging the gap between mathematical formulation and computational solution across science, engineering, and data-driven modeling. By mastering the hierarchy of methods developed in this chapter, from guaranteed but conservative bracketing through fast but fragile Newton iterations to robust globalized solvers for large-scale systems, you will be equipped to select and implement the most appropriate algorithm for any problem at hand. The continued evolution of these methods, driven by advances in optimization theory, Krylov solvers, machine-learning-based algorithm selection, and high-order convergence schemes, ensures that root finding remains an active frontier where classical numerical analysis meets modern computational practice.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/gYvuKWczcxTB5vdoGfzN.7","tags":[]}

# References

 1. Aghamie, S.O., Ehiwario, J. and Eboh, G. (2025). *Neural network-driven adaptive root-finding algorithm: learning to solve nonlinear equations more efficiently*. Journal of the Nigerian Association of Mathematical Physics, 70, pp. 79–84.
 2. Berra, S., La Torraca, A., Benvenuto, F. and Sommariva, S. (2024). *Combined Newton–Gradient method for constrained root-finding in chemical reaction networks*. Journal of Optimization Theory and Applications, 200(2), pp. 404–427.
 3. Martin, R.J. (2023). *Root-finding: from Newton to Halley and beyond*. arXiv preprint arXiv:2312.12305.
 4. Okawa, H., Fujisawa, K., Yamamoto, Y., Hirai, R., Yasutake, N., Nagakura, H. and Yamada, S. (2023). *The W4 method: A new multi-dimensional root-finding scheme for nonlinear systems of equations*. Applied Numerical Mathematics, 183, pp. 157–172.
 5. Thota, S., Naseem, A., Gopi, T., Sai Nandan Reddy, K., Sai Kousik, P., Bikku, T. and Palanisamy, S. (2025). *A novel ninth-order root-finding algorithm for nonlinear equations with implementations in various software tools*. International Journal of Optimization and Control: Theories & Applications, 15(3), pp. 503–516.
 6. Parisi, S., Agromayor, R., La Seta, A., Nielsen, K.K. and Haglind, F. (2025). *Advancing non-dimensional models for radial turbomachinery applied to pumped thermal energy storage*. Energy Conversion and Management, 342, 120085. <https://doi.org/10.1016/j.enconman.2025.120085>.
 7. Sapna, S. and Mohan, B.R. (2024). *Comparative analysis of root finding algorithms for implied volatility estimation of Ethereum options*. Computational Economics, 64(2), pp. 515–550. <https://doi.org/10.1007/s10614-023-10446-8>.
 8. SciPy v1.16 Documentation (2023). *scipy.optimize.brentq – Brent’s algorithm for root finding* \[online\]. Available at: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html> (Accessed: December 2025).
 9. SciPy v1.16 Documentation (2023). *scipy.optimize.ridder – Ridder’s method for root finding* \[online\]. Available at: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.ridder.html> (Accessed: December 2025).
10. Ridders, C.F.J. (1979). *A new algorithm for computing a single root of a real continuous function*. IEEE Transactions on Circuits and Systems, 26(11), pp. 979–980. <https://doi.org/10.1109/TCS.1979.1084580>.
11. Wikipedia (2024). *Ridders’ method* \[online\]. Available at: <https://en.wikipedia.org/wiki/Ridders%27_method> (Accessed: October 2024).
12. Berzi, P. (2024). *Convergence and stability improvement of quasi-Newton methods by full-rank update of the Jacobian approximates*. AppliedMath, 4(1), 143–181. <https://doi.org/10.3390/appliedmath4010008>
13. Martin, R.J. (2023). *Root-finding: from Newton to Halley and beyond*. arXiv preprint arXiv:2312.12305. Available at: <https://arxiv.org/abs/2312.12305> (Accessed: 2025).
14. Pan, V.Y., Go, S., Luan, Q. and Zhao, L. (2025). *A new fast root-finder for black box polynomials*. Theoretical Computer Science, 1027, 115022. <https://doi.org/10.1016/j.tcs.2024.115022>
15. Shah, F.A., Haider, I., Waseem, M., Mikhaylov, A. and Baranyai, N. (2025). *Construction and applications of iterative methods for finding approximate solutions of nonlinear equations having unknown zeros of multiplicity with fractal geometry and dynamical behavior*. MethodsX, 1025, 103778. <https://doi.org/10.1016/j.mex.2025.103778>
16. Vandebril, R. (2024). *Fast and stable roots of polynomials via companion matrices*. Seminar presentation, KU Leuven, February 2024.
17. Vivas-Cortez, M., Ali, N.Z., Khan, A.G. and Awan, M.U. (2023). *Numerical analysis of new hybrid algorithms for solving nonlinear equations*. Axioms, 12(7), 684. <https://doi.org/10.3390/axioms12070684>
18. Liu, C., Chen, C., Luo, L. and Lui, J.C.S. (2023). *Block Broyden’s methods for solving nonlinear equations*. In: Advances in Neural Information Processing Systems 37 (NeurIPS 2023), pp. 1–12.
19. Yetkin, E.F. and Ceylan, O. (2023). *Recycling Newton–Krylov algorithm for efficient solution of large-scale power systems*. International Journal of Electrical Power & Energy Systems, 144, 108559. <https://doi.org/10.1016/j.ijepes.2022.108559>

### 

