---
title: Chapter 19
description: ''
subtitle: Integral Equations and Inverse Theory
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
date: '2026-05-02'
oxa: oxa:pqQDe4beUu67RvW3raYP/DZmEiIgFecfoHdlMQhMK
keywords: []
---

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/abfAbsd9a3bykQTMHM8e.1","tags":[]}

> *"Inverse problems transform measurements into models, with integral equations forming the backbone of this reconstruction."*\
> —*Lawrence B. Ingram*

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/To2iFeTULIXXaH0t1XoI.1","tags":[]}

*Chapter 19 introduces integral equations and inverse problems, two important areas of numerical computing with applications in physics, engineering, imaging, geophysics, and data analysis. The chapter begins with the formulation and classification of integral equations, emphasizing the distinction between Fredholm and Volterra equations and their operator representations. Numerical methods for Fredholm equations, including Nyström, collocation, and Galerkin discretizations, are developed together with discussions of convergence, stability, and computational efficiency. Volterra equations are examined from the perspective of causality and memory effects, leading to efficient sequential solution techniques. The chapter also addresses integral equations with singular kernels and presents numerical methods for their accurate evaluation. Building on these foundations, the discussion turns to inverse problems, ill-posedness, and regularization. Tikhonov regularization, the Backus–Gilbert method, and maximum entropy reconstruction are introduced as important approaches for stable solution recovery. Throughout the chapter, mathematical theory is integrated with practical Rust implementations for scientific computing.*

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/YKylNqAjiMie6NCXC3gQ.5","tags":[]}

# 19.1. Introduction

Integral equations provide a natural mathematical language for describing systems in which the unknown quantity is determined through accumulated, distributed, or history-dependent effects. They arise when local differential descriptions are reformulated globally, when indirect measurements must be inverted, or when the present state of a system depends on its past. This introductory section classifies the main types of integral equations, distinguishes first-kind from second-kind formulations, and explains why second-kind Fredholm and Volterra equations are especially important for stable numerical computation and inverse theory.

## 19.1.1. Definition and Classification of Integral Equations

An *integral equation* is a relation in which an unknown function appears under an integral sign. Linear integral equations arise naturally in many applications. For example, reformulating partial differential equations leads to boundary-integral equations; inverse problems, such as image reconstruction, diffuse optical tomography, scattering, and deblurring, often reduce to integral equations; and dynamical models, including population dynamics and memory-dependent systems, may involve history-dependent integrals (Pooja, Kumar and Manchanda, 2024; Nature Index, 2025; Al Khaykanee, 2024).

A general linear Fredholm equation has fixed limits of integration and may be written as:

$$f(t) = \lambda \int_a^b K(t,s)f(s)\,ds + g(t) \tag{19.1.1}$$

where $K(t,s)$ is a given *kernel*, $\lambda$ is a parameter, $g(t)$ is known, and $f(t)$ is the unknown function. The defining feature of the Fredholm form is that the interval of integration $[a,b]$ is fixed independently of the evaluation point $t$. This fixed-interval structure is one reason Fredholm equations naturally arise in boundary-value problems, inverse problems, and global reconstruction problems, where the unknown function is coupled across an entire domain (Pooja, Kumar and Manchanda, 2024; Nature Index, 2025).

In contrast, a Volterra equation has a variable upper limit. A typical linear Volterra equation has the form:

$$f(t) = \lambda \int_a^t K(t,s)f(s)\,ds + g(t) \tag{19.1.2}$$

Here the value of $f(t)$ depends only on values of the unknown function over the interval from the initial point $a$ up to the current point $t$. This gives Volterra equations a causal or history-dependent structure, making them especially suitable for initial-value problems, memory effects, and time-dependent systems (Al Khaykanee, 2024; Hamood, Sharif and Ghadle, 2025).

Integral equations are also classified according to the way the unknown function appears. If the unknown function appears both outside and inside the integral, as in (19.1.1) and (19.1.2), the equation is said to be of the *second kind*. If the unknown appears only under the integral sign, the equation is of the *first kind*. A representative first-kind Fredholm equation is:

$$g(t) = \int_a^b K(t,s)f(s)\,ds \tag{19.1.3}$$

This distinction is essential because first-kind and second-kind equations have very different analytical and numerical properties.

## 19.1.2. Well-Posedness and Operator Form

First-kind equations such as (19.1.3) are usually *ill-posed*. The integral operator often acts as a smoothing operator, similar to convolution. As a result, the observed or given data $g(t)$ may contain less information than the unknown function $f(t)$, and small perturbations or noise in $g$ may produce large errors in the reconstructed $f$. Recovering $f$ from $g$ therefore typically requires regularization, especially in inverse problems such as image reconstruction, inverse Laplace transforms, tomography, and deblurring (Pooja, Kumar and Manchanda, 2024).

Second-kind equations are usually better behaved. Defining the integral operator $T$ by:

\begin{equation}
(Tf)(t) = \int_a^b K(t,s)\,f(s)\,ds
\tag{19.1.4}
\end{equation}

the Fredholm equation (19.1.1) may be written in operator form as:

$$(I-\lambda T)f = g \tag{19.1.5}$$

This representation shows that the equation is the identity operator perturbed by an integral operator. Under common assumptions on the kernel, the integral operator is compact. Consequently, second-kind Fredholm equations often lead to well-conditioned numerical systems after discretization, particularly when $|\lambda|$ is not too large or when the compact perturbation does not bring the system close to singularity (Nature Index, 2025).

The operator viewpoint also explains why second-kind equations are central in computational practice. Unlike first-kind equations, where the unknown must be recovered through a smoothing operator, second-kind equations retain the stabilizing contribution of the identity operator. This makes them suitable for direct discretization and solution by quadrature, collocation, Galerkin projection, or modern accelerated solvers.

### Rust Implementation

Following the discussion in Section 19.1.2 on the operator form of second-kind Fredholm integral equations, Program 19.1.1 provides a practical implementation of the discretized equation associated with (19.1.4) and (19.1.5). The purpose of the program is to show how the continuous operator equation $(I-\lambda T)f=g$ becomes a finite-dimensional linear system after quadrature and collocation. In contrast to first-kind equations, where the unknown is recovered only through a smoothing integral operator, the second-kind formulation retains the stabilizing identity term. This structure is reflected directly in the assembled matrix, whose entries have the form of an identity matrix perturbed by a weighted kernel contribution. The program uses a manufactured exact solution to verify the discretization error and to make the connection between the mathematical formulation and the numerical approximation transparent.

At the core of the implementation is the conversion of the continuous integral operator in (19.1.4) into a discrete quadrature operator. The function `kernel(t, s)` defines the kernel $K(t,s)$, while `exact_solution(t)` specifies the manufactured solution used to verify the numerical result. The right-hand side function `rhs(t, lambda)` is then constructed so that the chosen exact solution satisfies the second-kind Fredholm equation. This manufactured-solution strategy is useful in textbook examples because it allows the numerical error to be measured directly rather than inferred indirectly.

The functions `uniform_grid` and `trapezoidal_weights` define the collocation and quadrature structure. The grid points represent the discrete locations $t_i$ at which the integral equation is enforced, while the trapezoidal weights approximate the integral appearing in (19.1.4). Using the same grid for both collocation and quadrature keeps the implementation simple and makes the resulting linear system easy to interpret.

The function `assemble_matrix` implements the discrete version of (19.1.5). For each collocation point $t_i$, the integral term is approximated by a weighted sum over quadrature points $s_j$. The matrix entry is formed as the difference between the identity contribution and the discretized integral contribution. This is the most important part of the program because it shows explicitly how the second-kind Fredholm equation produces a matrix of the form $A=I-\lambda T_h$, where $T_h$ is the quadrature approximation of the integral operator.

The function `assemble_rhs` evaluates the right-hand side at each grid point. Together with the matrix assembled by `assemble_matrix`, it defines the finite-dimensional system corresponding to the continuous equation. The function `solve_linear_system` then solves this dense linear system using Gaussian elimination with partial pivoting. Although this direct solver is intended only for small pedagogical examples, it is adequate for demonstrating the basic numerical structure introduced in Section 19.1.2.

The function `max_abs_error` compares the computed solution against the manufactured exact solution at the grid points. This provides a simple verification measure for the discretization. The function `perturbation_size` computes an infinity-norm estimate of the difference between the assembled matrix and the identity matrix. This diagnostic is included to reinforce the interpretation of (19.1.5): the matrix is not merely a discretized smoothing operator, but the identity matrix modified by the compact integral contribution.

The `main` function brings these components together. It sets the interval, the number of grid points, and the parameter $\lambda$, constructs the grid and quadrature weights, assembles the matrix and right-hand side, solves the linear system, and prints a table comparing the numerical and exact solutions. The output also reports the maximum absolute error and the size of the matrix perturbation from the identity. These diagnostics connect the computational result directly to the theoretical explanation in Section 19.1.2.

```rust
// Program 19.1.1: Discretization of a Second-Kind Fredholm Integral Equation
//
// Problem statement:
// Solve the second-kind Fredholm integral equation
//
//     f(t) = lambda * integral_a^b K(t, s) f(s) ds + g(t),
//
// or equivalently,
//
//     (I - lambda T) f = g,
//
// by collocation on a uniform grid and composite trapezoidal quadrature.
// The example uses a manufactured exact solution so that the numerical
// approximation can be checked directly.

use std::f64::consts::PI;

/// Kernel K(t, s) appearing in the Fredholm integral equation.
///
/// This kernel is smooth on [0, 1] x [0, 1], so the trapezoidal
/// discretization gives a clear demonstration of the operator form
/// without introducing singular-kernel complications.
fn kernel(t: f64, s: f64) -> f64 {
    t * s
}

/// Manufactured exact solution f(t).
///
/// We choose a simple smooth function so that the right-hand side g(t)
/// can be computed analytically.
fn exact_solution(t: f64) -> f64 {
    (PI * t).sin()
}

/// Right-hand side g(t), constructed so that the exact solution satisfies
///
///     f(t) = lambda * integral_0^1 K(t, s) f(s) ds + g(t).
///
/// With K(t, s) = t s and f(s) = sin(pi s),
///
///     integral_0^1 t s sin(pi s) ds = t / pi.
///
/// Therefore,
///
///     g(t) = sin(pi t) - lambda * t / pi.
fn rhs(t: f64, lambda: f64) -> f64 {
    exact_solution(t) - lambda * t / PI
}

/// Build a uniform grid on [a, b] with n points.
fn uniform_grid(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);

    (0..n)
        .map(|i| a + i as f64 * h)
        .collect()
}

/// Composite trapezoidal weights on [a, b] with n grid points.
fn trapezoidal_weights(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);
    let mut weights = vec![h; n];

    weights[0] = 0.5 * h;
    weights[n - 1] = 0.5 * h;

    weights
}

/// Assemble the matrix A for the discrete system
///
///     A f = g,
///
/// where
///
///     A_ij = delta_ij - lambda * w_j * K(t_i, s_j).
///
/// Here the collocation points t_i and quadrature points s_j are chosen
/// to be the same uniform grid.
fn assemble_matrix(grid: &[f64], weights: &[f64], lambda: f64) -> Vec<Vec<f64>> {
    let n = grid.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        let t_i = grid[i];

        for j in 0..n {
            let s_j = grid[j];

            let identity = if i == j { 1.0 } else { 0.0 };
            let integral_part = lambda * weights[j] * kernel(t_i, s_j);

            matrix[i][j] = identity - integral_part;
        }
    }

    matrix
}

/// Assemble the right-hand side vector g_i = g(t_i).
fn assemble_rhs(grid: &[f64], lambda: f64) -> Vec<f64> {
    grid.iter()
        .map(|&t| rhs(t, lambda))
        .collect()
}

/// Solve a dense linear system using Gaussian elimination with partial pivoting.
///
/// This is sufficient for the small pedagogical systems used in this section.
/// Later chapters or large-scale integral-equation solvers would use more
/// specialized methods.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            let candidate = a[i][k].abs();

            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!(
                "Matrix is numerically singular near column {}.",
                k
            ));
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

/// Compute the infinity norm of the pointwise error.
fn max_abs_error(grid: &[f64], numerical: &[f64]) -> f64 {
    grid.iter()
        .zip(numerical.iter())
        .map(|(&t, &f_num)| (f_num - exact_solution(t)).abs())
        .fold(0.0, f64::max)
}

/// Estimate the infinity norm of the difference between A and the identity.
///
/// Since A = I - lambda T_h, this quantity gives a simple diagnostic for
/// the size of the discretized compact perturbation.
fn perturbation_size(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    let mut max_row_sum = 0.0;

    for i in 0..n {
        let mut row_sum = 0.0;

        for j in 0..n {
            let identity = if i == j { 1.0 } else { 0.0 };
            row_sum += (matrix[i][j] - identity).abs();
        }

        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }

    max_row_sum
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let n = 21;
    let lambda = 0.5;

    let grid = uniform_grid(a, b, n);
    let weights = trapezoidal_weights(a, b, n);

    let matrix = assemble_matrix(&grid, &weights, lambda);
    let rhs_vector = assemble_rhs(&grid, lambda);

    let perturbation_norm = perturbation_size(&matrix);

    let numerical_solution = match solve_linear_system(matrix, rhs_vector) {
        Ok(solution) => solution,
        Err(message) => {
            eprintln!("Linear solve failed: {}", message);
            return;
        }
    };

    let error = max_abs_error(&grid, &numerical_solution);

    println!("Second-Kind Fredholm Integral Equation");
    println!("======================================");
    println!();
    println!("Model Problem");
    println!("-------------");
    println!("Equation type              = second-kind Fredholm");
    println!("Interval                   = [{:.1}, {:.1}]", a, b);
    println!("Kernel K(t, s)             = t s");
    println!("Exact solution f(t)        = sin(pi t)");
    println!("Lambda                     = {:.6}", lambda);
    println!();
    println!("Discretization");
    println!("--------------");
    println!("Grid points N              = {}", n);
    println!("Quadrature rule            = composite trapezoidal rule");
    println!("Linear system              = (I - lambda T_h) f = g");
    println!(
        "Estimated ||A - I||_inf    = {:.6e}",
        perturbation_norm
    );
    println!();
    println!("Numerical Results");
    println!("-----------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "i", "t_i", "f_num", "f_exact", "abs_error"
    );

    for i in 0..n {
        let t = grid[i];
        let f_num = numerical_solution[i];
        let f_exact = exact_solution(t);
        let abs_error = (f_num - f_exact).abs();

        println!(
            "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.6e}",
            i, t, f_num, f_exact, abs_error
        );
    }

    println!();
    println!("Maximum Absolute Error");
    println!("----------------------");
    println!("max_i |f_num(t_i) - f_exact(t_i)| = {:.6e}", error);
}
```

Program 19.1.1 demonstrates how the operator equation in (19.1.5) is translated into a concrete numerical computation. The discretization replaces the integral operator $T$ by a quadrature matrix $T_h$, while the identity term remains explicit in the final system. This is the key numerical advantage of second-kind Fredholm equations: after discretization, the resulting matrix retains the stabilizing identity contribution rather than representing only a smoothing operation.

The numerical output confirms the consistency of the implementation. The small maximum absolute error shows that the quadrature-based collocation method accurately recovers the manufactured solution on the selected grid. The reported value of $\|A-I\|_\infty$ also provides a useful structural diagnostic, showing the size of the discrete integral perturbation relative to the identity matrix. This reinforces the interpretation that second-kind Fredholm equations often lead to better-conditioned systems than first-kind equations.

Although the example uses a simple smooth kernel and a direct dense solver, the same assembly principle extends to more general Fredholm equations. More advanced implementations may replace the trapezoidal rule with higher-order quadrature, use adaptive discretization, exploit low-rank or sparse structure in the kernel matrix, or apply iterative solvers for large systems. The present program therefore serves as a foundation for the later numerical methods in this chapter, where quadrature, collocation, Galerkin projection, and accelerated solvers are developed in greater detail.

## 19.1.3. Fredholm and Volterra Equations in Applications

Fredholm and Volterra equations appear across science and engineering. First-kind Fredholm equations model inverse problems in which measured data are linearly related to an unknown quantity through an integral kernel. Examples include diffuse optical tomography, inverse Laplace transforms, image deblurring, geophysical inversion, and medical imaging. In such settings, the measured data $g$, such as scattered fields or projection data, are related to an unknown profile $f$ by a kernel $K$. A concrete example is computed tomography, where the inverse Radon transform may be interpreted as the recovery of an absorption profile from projection data, which has the character of a first-kind Fredholm problem (Pooja, Kumar and Manchanda, 2024).

Fredholm equations also arise in physics through scattering theory and boundary-integral reformulations of partial differential equations. For example, the Lippmann–Schwinger equation in quantum scattering is a Fredholm equation, while boundary-integral formulations of Laplace and Helmholtz equations yield second-kind Fredholm equations for unknown boundary densities. In such boundary integral equations, the unknown boundary density satisfies an equation of the form:

\begin{equation}
f(t) = \int_a^b K(t,s)\,f(s)\,ds + g(t)
\tag{19.1.6}
\end{equation}

where the associated kernel operator is compact. After discretization, this structure often produces a well-conditioned linear system and allows the use of specialized solvers for large-scale boundary-value problems (Nature Index, 2025).

Volterra equations of the second kind commonly occur when initial-value ordinary differential equations are reformulated as integral equations, for instance through Duhamel’s principle. A representative causal integral equation is:

\begin{equation}
y(t) = g(t) + \int_a^t K(t,s)\,y(s)\,ds
\tag{19.1.7}
\end{equation}

In this form, the value $y(t)$ depends only on earlier values $y(s)$ with $s\le t$. The integral operator therefore has a triangular or causal structure. This is the continuous analogue of a lower triangular system and explains why numerical solution methods for Volterra equations often resemble marching methods for ordinary differential equations (Al Khaykanee, 2024).

Many physical processes with memory lead naturally to Volterra equations. Examples include viscoelastic relaxation, population growth models, and systems in which the present state depends on accumulated past behavior. Fractional-order dynamics, including models involving Caputo derivatives, can also be written as Volterra-type integral equations with kernels that may be weakly singular near $s=t$. Such formulations are especially important in fractional Volterra–Fredholm problems and related integro-differential models (Hamood, Sharif and Ghadle, 2025).

### Rust Implementation

Following the discussion in Section 19.1.3 on the role of Volterra equations in causal and history-dependent systems, Program 19.1.2 provides a practical implementation of the second-kind Volterra equation in (19.1.7). The key computational feature is the variable upper limit of integration, which ensures that the value at the current point depends only on the present and previously computed values. This gives the discretized problem a lower triangular structure and allows the solution to be advanced by forward marching rather than by solving a full dense linear system. The program uses a manufactured exact solution to verify the numerical approximation and to show explicitly how the causal structure of the continuous equation is inherited by the discrete method.

At the core of the implementation is the causal discretization of the Volterra equation in (19.1.7). The function `kernel(t, s)` defines the memory kernel $K(t,s)$, which describes how the earlier value at $s$ contributes to the value at the current point $t$. In this example, the kernel is chosen as an exponentially decaying memory term, so more recent values have stronger influence than older values. The function `exact_solution(t)` defines the manufactured solution used for verification, while `rhs(t, lambda)` constructs the corresponding right-hand side so that the exact solution satisfies the continuous Volterra equation.

The `uniform_grid` function builds the computational grid on the interval of integration. Since the Volterra equation is causal, the grid is traversed from left to right. This ordering is essential: once the value at a new grid point is computed, it becomes part of the available history for all later grid points. In this way, the grid does not merely define where the solution is sampled, but also determines the chronological order in which the numerical solution is constructed.

The main numerical work is carried out by `solve_volterra_trapezoidal`. This function applies the composite trapezoidal rule to approximate the integral in (19.1.7) at each grid point. For a fixed point $t_i$, the summation only includes grid points $t_j\leq t_i$. All terms with $j<i$ involve previously computed solution values and are accumulated in `history_sum`. The endpoint term at $j=i$ contains the unknown current value itself, so it is moved to the left-hand side through a scalar diagonal correction. This is the discrete analogue of forward substitution in a lower triangular system.

The code includes a small safeguard against numerical failure by checking whether the diagonal coefficient becomes too small. In normal second-kind Volterra problems with moderate $\lambda$ and a regular kernel, this coefficient remains safely away from zero. The check is nevertheless useful because it makes the implementation robust and highlights that even a marching method still solves a local algebraic equation at each step.

The function `max_abs_error` compares the computed values against the manufactured exact solution on the grid. This gives a direct measure of the discretization error. The function `discrete_residual_norm` evaluates how well the computed solution satisfies the discrete Volterra equation obtained from the same trapezoidal approximation. This distinction is important: the maximum absolute error measures agreement with the continuous manufactured solution, while the discrete residual measures whether the algebraic marching equations have been solved correctly.

The `main` function brings the implementation together. It defines the interval, number of grid points, and parameter $\lambda$, constructs the grid, solves the Volterra equation by forward marching, and prints a table comparing the numerical and exact values. It also reports the maximum absolute error and the maximum discrete residual. The output therefore verifies both the accuracy of the discretization and the correctness of the lower triangular marching solve.

```rust
// Program 19.1.2: Marching Solution of a Second-Kind Volterra Integral Equation
//
// Problem statement:
// Solve the second-kind Volterra integral equation
//
//     y(t) = g(t) + lambda * integral_a^t K(t, s) y(s) ds,
//
// by marching forward on a uniform grid. The causal upper limit t makes the
// discrete system lower triangular, so each new value y_i can be computed from
// previously known values y_0, ..., y_{i-1}, together with a possible diagonal
// correction from the trapezoidal endpoint.
//
// The example uses the manufactured exact solution
//
//     y(t) = exp(t),
//
// with kernel
//
//     K(t, s) = exp(-(t - s)),
//
// on the interval [0, 2]. Since
//
//     integral_0^t exp(-(t - s)) exp(s) ds = sinh(t),
//
// the right-hand side is chosen as
//
//     g(t) = exp(t) - lambda * sinh(t).
//
// This allows the numerical marching solution to be checked directly.

/// Kernel K(t, s) for the causal Volterra integral equation.
///
/// For s <= t, this kernel represents an exponentially decaying memory
/// effect. Values with s > t are never used by the Volterra solver.
fn kernel(t: f64, s: f64) -> f64 {
    (-(t - s)).exp()
}

/// Manufactured exact solution y(t).
fn exact_solution(t: f64) -> f64 {
    t.exp()
}

/// Right-hand side g(t), constructed so that y(t) = exp(t)
/// satisfies the Volterra equation.
fn rhs(t: f64, lambda: f64) -> f64 {
    exact_solution(t) - lambda * t.sinh()
}

/// Build a uniform grid on [a, b] with n points.
fn uniform_grid(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);

    (0..n)
        .map(|i| a + i as f64 * h)
        .collect()
}

/// Solve the second-kind Volterra integral equation by forward marching.
///
/// At grid point t_i, the trapezoidal approximation gives
///
///     y_i = g_i + lambda * sum_{j=0}^{i} w_{ij} K(t_i, t_j) y_j.
///
/// The endpoint contribution j = i may contain y_i itself. Therefore,
///
///     y_i * (1 - lambda * w_ii * K(t_i, t_i))
///
/// is known after all previous terms have been accumulated. This is the
/// discrete analogue of solving a lower triangular system by forward
/// substitution.
fn solve_volterra_trapezoidal(
    grid: &[f64],
    lambda: f64,
) -> Result<Vec<f64>, String> {
    let n = grid.len();

    if n < 2 {
        return Err("At least two grid points are required.".to_string());
    }

    let h = grid[1] - grid[0];
    let mut y = vec![0.0; n];

    for i in 0..n {
        let t_i = grid[i];

        let mut history_sum = 0.0;

        for j in 0..i {
            let s_j = grid[j];

            let weight = if j == 0 {
                0.5 * h
            } else {
                h
            };

            history_sum += weight * kernel(t_i, s_j) * y[j];
        }

        let diagonal_weight = if i == 0 {
            0.0
        } else {
            0.5 * h
        };

        let diagonal_coefficient =
            1.0 - lambda * diagonal_weight * kernel(t_i, t_i);

        if diagonal_coefficient.abs() < 1.0e-14 {
            return Err(format!(
                "Volterra marching failed at grid index {} because the diagonal coefficient is too small.",
                i
            ));
        }

        let numerator = rhs(t_i, lambda) + lambda * history_sum;
        y[i] = numerator / diagonal_coefficient;
    }

    Ok(y)
}

/// Compute the maximum absolute error on the grid.
fn max_abs_error(grid: &[f64], numerical: &[f64]) -> f64 {
    grid.iter()
        .zip(numerical.iter())
        .map(|(&t, &y_num)| (y_num - exact_solution(t)).abs())
        .fold(0.0, f64::max)
}

/// Compute a residual diagnostic using the same trapezoidal discretization.
///
/// This checks how well the computed vector satisfies the discrete Volterra
/// equation rather than the continuous equation.
fn discrete_residual_norm(grid: &[f64], numerical: &[f64], lambda: f64) -> f64 {
    let n = grid.len();
    let h = grid[1] - grid[0];

    let mut max_residual: f64 = 0.0;

    for i in 0..n {
        let t_i = grid[i];
        let mut integral_approx = 0.0;

        for j in 0..=i {
            let s_j = grid[j];

            let weight = if i == 0 {
                0.0
            } else if j == 0 || j == i {
                0.5 * h
            } else {
                h
            };

            integral_approx += weight * kernel(t_i, s_j) * numerical[j];
        }

        let residual = numerical[i] - rhs(t_i, lambda) - lambda * integral_approx;
        max_residual = max_residual.max(residual.abs());
    }

    max_residual
}

fn main() {
    let a = 0.0;
    let b = 2.0;
    let n = 41;
    let lambda = 0.4;

    let grid = uniform_grid(a, b, n);

    let numerical_solution = match solve_volterra_trapezoidal(&grid, lambda) {
        Ok(solution) => solution,
        Err(message) => {
            eprintln!("Volterra solve failed: {}", message);
            return;
        }
    };

    let error = max_abs_error(&grid, &numerical_solution);
    let residual = discrete_residual_norm(&grid, &numerical_solution, lambda);

    println!("Second-Kind Volterra Integral Equation");
    println!("======================================");
    println!();
    println!("Model Problem");
    println!("-------------");
    println!("Equation type              = second-kind Volterra");
    println!("Interval                   = [{:.1}, {:.1}]", a, b);
    println!("Kernel K(t, s)             = exp(-(t - s))");
    println!("Exact solution y(t)        = exp(t)");
    println!("Lambda                     = {:.6}", lambda);
    println!();
    println!("Discretization");
    println!("--------------");
    println!("Grid points N              = {}", n);
    println!("Quadrature rule            = composite trapezoidal rule");
    println!("Solution strategy          = forward marching");
    println!("Discrete structure         = lower triangular causal system");
    println!();
    println!("Numerical Results");
    println!("-----------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "i", "t_i", "y_num", "y_exact", "abs_error"
    );

    for i in 0..n {
        let t = grid[i];
        let y_num = numerical_solution[i];
        let y_exact = exact_solution(t);
        let abs_error = (y_num - y_exact).abs();

        println!(
            "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.6e}",
            i, t, y_num, y_exact, abs_error
        );
    }

    println!();
    println!("Diagnostics");
    println!("-----------");
    println!("max_i |y_num(t_i) - y_exact(t_i)| = {:.6e}", error);
    println!("maximum discrete residual          = {:.6e}", residual);
}
```

Program 19.1.2 demonstrates how the causal structure of the second-kind Volterra equation in (19.1.7) leads naturally to a forward-marching numerical method. Unlike the Fredholm equation in (19.1.6), which couples the solution over the full interval, the Volterra equation only couples the current value to its past history. After discretization, this produces a lower triangular system that can be solved sequentially.

The numerical output confirms this interpretation. The maximum discrete residual is close to machine precision, showing that the computed vector satisfies the discrete Volterra equations almost exactly. The maximum absolute error is larger because it reflects the quadrature error accumulated while approximating the continuous integral. This gradual growth of error as $t$ increases is expected for a causal history-dependent problem, since each new value depends on previously computed numerical values.

Although this example uses a smooth kernel and the composite trapezoidal rule, the same marching principle extends to more general Volterra equations. Higher-order quadrature rules, adaptive grids, product integration for weakly singular kernels, and specialized methods for fractional-order models can all be built on the same causal foundation. The program therefore provides a useful computational bridge between the introductory classification of Volterra equations and the more advanced numerical methods developed later in the chapter.

## 19.1.4. Section Remarks

In summary, integral equations are ubiquitous in applied mathematics, scientific computing, and engineering. Fredholm equations with fixed limits arise primarily in boundary-value problems, inverse problems, scattering, tomography, and global reconstruction. Volterra equations with variable limits arise naturally in initial-value problems, causal dynamical systems, memory-dependent processes, and fractional-order models.

This chapter focuses on *linear integral equations of the second kind*, where the unknown function appears both outside and inside the integral. This class is especially important because such equations are typically better posed than first-kind equations, are closely connected to compact-operator theory, and form a practical foundation for numerical methods in inverse theory, boundary-integral computation, and time-dependent modeling.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/SLnZrnur0EBE6U53OqSo.5","tags":[]}

# 19.2. Fredholm Equations of the Second Kind

Fredholm integral equations of the second kind form one of the central classes of linear integral equations because they combine a compact integral operator with the identity operator. This structure gives them a more favorable analytical and numerical character than first-kind equations, which often require strong regularization. This section develops the operator form of the second-kind Fredholm equation, explains how discretization leads to dense linear systems, and surveys both classical numerical methods and modern accelerated approaches, including low-rank compression, hierarchical methods, fast multipole techniques, and physics-informed neural formulations.

## 19.2.1. Operator Formulation, Compactness, and Solvability of Fredholm Equations

A Fredholm integral equation of the second kind has the form:

\begin{equation}
f(t) = \lambda \int_a^b K(t,s)\,f(s)\,ds + g(t)
\tag{19.2.1}
\end{equation}

where $K(t,s)$ and $g(t)$ are given, and $f(t)$ is the unknown function. The kernel $K(t,s)$ is assumed to be linear in the unknown, meaning that it does not itself depend on $f$. Under this assumption, the equation can be interpreted as a linear operator equation.

Define the integral operator $T$ by:

\begin{equation}
(Tf)(t) = \int_a^b K(t,s)\,f(s)\,ds
\tag{19.2.2}
\end{equation}

Then (19.2.1) may be written in the compact operator form,

\begin{equation}
(I - \lambda T)f = g
\tag{19.2.3}
\end{equation}

This formulation is central because it separates the identity contribution from the integral contribution. The operator $T$ represents the action of the kernel on the unknown function, while $I-\lambda T$ represents the full linear operator that must be inverted in order to recover $f$.

Under mild assumptions on the kernel, for example when $K$ is square-integrable, the operator $T$ is compact on the appropriate function space. This compactness leads naturally to the classical Fredholm theory. In particular, the *Fredholm alternative* states that if $\lambda$ is not an eigenvalue of $T$, then $I-\lambda T$ is invertible, and the equation has a unique solution $f$. If (\\lambda) corresponds to an eigenvalue, the homogeneous equation may have nontrivial solutions, and solvability of the nonhomogeneous equation depends on compatibility conditions.

This operator viewpoint explains why Fredholm equations of the second kind are usually more stable than first-kind equations. The identity operator remains present in (19.2.3), so the problem is not simply the inversion of a smoothing compact operator. Instead, the compact operator appears as a perturbation of the identity. This structure is one of the main reasons second-kind Fredholm equations are central in boundary integral formulations, scattering models, and modern numerical solution methods.

## 19.2.2. Nyström, Collocation, and Galerkin Discretizations for Fredholm Equations

In practice, equation (19.2.1) is solved by replacing the continuous integral operator with a finite-dimensional approximation. This process transforms the Fredholm integral equation of the second kind into a system of algebraic equations that can be handled using numerical linear algebra. Among the most widely used approaches are the Nyström method, collocation methods, and Galerkin projection methods. Each method differs in how the integral operator and the unknown function are approximated, but all lead to a linear system for a finite set of unknowns.

The Nyström method is based on numerical quadrature. Let $\{t_i\}_{i=1}^N$ denote quadrature nodes on the interval $[a,b]$, and let $\{w_j\}_{j=1}^N$ be the associated weights. The integral in (19.2.1) is approximated by a weighted sum, and the unknown function is evaluated only at the quadrature nodes. Writing:

$$f(s) \approx f_j, \qquad s = t_j,$$

and substituting into (19.2.1), one obtains the discrete approximation:

\begin{equation}
f_i \approx \lambda \sum_{j=1}^N K(t_i,t_j)\,w_j\,f_j + g_i,
\qquad i = 1,\dots,N
\tag{19.2.4}
\end{equation}

where $f_i \approx f(t_i)$ and $g_i = g(t_i)$. This system represents the balance between the function values and the discretized integral operator at each quadrature node. The system (19.2.4) can be written compactly in matrix form as:

\begin{equation}(I - \lambda W K)\,\mathbf{f} = \mathbf{g}\tag{19.2.5}\end{equation}

where the kernel matrix and weight matrix are defined by:

\begin{equation}K_{ij} = K(t_i,t_j),\qquad W = \operatorname{diag}(w_1,\dots,w_N)\tag{19.2.6}\end{equation}

and $\mathbf{f}$, $\mathbf{g}$ are vectors of nodal values. The matrix $WK$ represents the discretized integral operator. It is often convenient to absorb the quadrature weights directly into the kernel matrix. Defining:

$$\mathbf{K}_{ij} = K(t_i,t_j)w_j \tag{19.2.7}$$

the system simplifies to:

\begin{equation}(I - \lambda \mathbf{K})\,\mathbf{f} = \mathbf{g}\tag{19.2.8}\end{equation}

with the unknown vector given by:

$$\mathbf{f} \approx (f_1,\dots,f_N)^\top \tag{19.2.9}$$

For moderate values of $N$, typically up to a few thousand, the dense linear system (19.2.8) may be solved using direct methods such as Gaussian elimination or LU decomposition. These methods have computational complexity $O(N^3)$ and storage requirements $O(N^2)$. The numerical stability and conditioning of the system depend on the properties of the operator $I - \lambda \mathbf{K}$. In the case of second-kind equations, the identity term dominates when $|\lambda|$ is sufficiently small or when the integral operator has bounded norm, leading to well-conditioned systems in many practical settings.

Beyond Nyström discretization, alternative formulations approximate the unknown function itself rather than the integral operator. In collocation methods, the function is expanded in a finite basis,

\begin{equation}f(t) \approx \sum_{k=1}^N c_k\,\phi_k(t)\tag{19.2.10}\end{equation}

where ${\phi_k}$ are chosen basis functions and ${c_k}$ are unknown coefficients. Substituting this expansion into (19.2.1) and enforcing the equation at selected collocation points $t_i$ yields a system of equations for the coefficients:

$$
\sum_{k=1}^N c_k\,\phi_k(t_i)
=
\lambda \int_a^b K(t_i,s)\,\sum_{k=1}^N c_k\,\phi_k(s)\,ds
+ g(t_i), 
\qquad i = 1,\dots,N
$$

This produces a dense linear system whose structure depends on the choice of basis and quadrature used to evaluate the integrals.

In Galerkin methods, the same expansion (19.2.10) is employed, but instead of enforcing the equation pointwise, the residual is required to be orthogonal to a test space. Typically, the test functions are chosen to be the same as the basis functions. Defining the residual:

$$
R(t) = \sum_{k=1}^N c_k\,\phi_k(t)
- \lambda \int_a^b K(t,s)\,\sum_{k=1}^N c_k\,\phi_k(s)\,ds
- g(t)
$$

the Galerkin condition requires:

$$
\int_a^b R(t)\,\phi_i(t)\,dt = 0, 
\qquad i = 1,\dots,N
$$

This again yields a linear system for the coefficients $c_k$, often with improved stability and approximation properties compared with collocation, particularly for symmetric or self-adjoint kernels.

Spectral collocation methods represent a high-accuracy variant of these approaches. When the kernel $K(t,s)$ and solution $f(t)$ are smooth, global polynomial bases such as Chebyshev or Legendre polynomials provide highly efficient approximations. The resulting systems are dense but achieve very rapid convergence. For sufficiently smooth problems, the error decreases faster than any algebraic rate and may exhibit exponential decay as $N \to \infty$.

When the cost of solving dense systems becomes prohibitive, iterative and fast algorithms are preferred. The integral equation (19.2.1) may be interpreted as a fixed-point problem. If the operator $\lambda T$ has sufficiently small norm, the solution admits a Neumann series expansion,

$$f = g + \lambda T g + \lambda^2 T^2 g + \cdots \tag{19.2.11}$$

which leads to the simple iteration:

$$f_{n+1} = g + \lambda T(f_n) \tag{19.2.12}$$

Although this iteration converges only under restrictive conditions, it provides useful insight into the structure of the solution.

More generally, Krylov subspace methods such as GMRES are applied to the discrete system (19.2.8). These methods avoid explicit matrix factorization and instead rely on repeated matrix–vector products. For dense matrices, each product requires $O(N^2)$ operations, but the overall cost can be significantly lower than $O(N^3)$ if convergence is achieved in a small number of iterations.

Modern developments focus on reducing the computational complexity of dense integral operators. For smooth kernels, low-rank approximations and hierarchical matrix representations allow compression of $\mathbf{K}$, reducing storage and computational costs. Fast multipole methods further accelerate matrix–vector products when the kernel arises from Green’s functions or other structured interactions. These techniques reduce complexity toward near-linear scaling in $N$, making large-scale problems tractable.

In parallel, data-driven approaches have emerged as an alternative paradigm. Physics-Informed Neural Networks (PINNs) approximate the solution $f(t)$ using neural networks trained to satisfy the integral equation. For example, Ma and Li (2025) combine discretization in the integral variable with neural approximations of the unknown function, preserving the structure of the Fredholm operator while improving flexibility in high-dimensional settings. Although still under active development, such approaches illustrate a growing integration of numerical analysis and machine learning in the solution of integral equations.

### Rust Implementation

Following the discussion in Section 19.2.2 on Nyström, collocation, and Galerkin discretizations, Program 19.2.1 provides a concrete implementation of the Nyström method for a Fredholm integral equation of the second kind. The program follows the construction in equations (19.2.4) through (19.2.8), where the continuous integral operator is replaced by a quadrature-based weighted kernel matrix and the resulting problem is written as a dense linear system. This implementation emphasizes the central numerical idea of the Nyström method: the unknown function is approximated by its values at quadrature nodes, while the integral term is approximated by a weighted sum over the same nodes. A manufactured exact solution is used so that the accuracy of the discretization can be checked directly, and the printed diagnostics show both the size of the discrete operator perturbation and the residual of the assembled algebraic system.

At the core of the implementation is the replacement of the continuous Fredholm operator in equation (19.2.1) by the quadrature approximation described in equation (19.2.4). The function `kernel(t, s)` defines the kernel $K(t,s)$, while `exact_solution(t)` specifies the manufactured solution used to validate the computation. The function `rhs(t, lambda)` constructs the corresponding right-hand side $g(t)$ so that the chosen exact solution satisfies the original continuous equation. This manufactured-solution approach is useful in a textbook setting because it allows the numerical error to be evaluated explicitly at every grid point.

The functions `uniform_grid` and `trapezoidal_weights` define the quadrature nodes and weights used in the Nyström approximation. The grid values represent the points $t_i$ at which the unknown function is sampled, while the trapezoidal weights approximate the integral over the fixed interval. This directly reflects the role of the nodes and weights in equation (19.2.4), where the integral is replaced by a finite weighted sum.

The function `assemble_weighted_kernel` constructs the weighted kernel matrix corresponding to equation (19.2.7). Each matrix entry combines the kernel value $K(t_i,t_j)$ with the quadrature weight $w_j$. This step is important because it absorbs the quadrature weights into the discrete kernel matrix, allowing the final algebraic system to be written in the compact form used in equation (19.2.8).

The function `assemble_system_matrix` forms the dense matrix $I-\lambda\mathbf K$ associated with equation (19.2.8). This is the main algebraic object produced by the Nyström discretization. The identity contribution appears explicitly on the diagonal, while the weighted kernel contribution represents the discretized integral operator. This structure illustrates why second-kind Fredholm equations are often numerically better behaved than first-kind equations: the matrix is not merely a discretized smoothing operator, but an identity matrix perturbed by an integral operator.

The function `assemble_rhs` evaluates the right-hand side at the grid points, producing the vector $\mathbf g$ in the discrete system. The function `solve_linear_system` then solves the resulting dense linear system using Gaussian elimination with partial pivoting. Although this direct solver is intended for moderate-sized pedagogical examples, it clearly demonstrates how the finite-dimensional system produced by equation (19.2.8) can be treated using standard numerical linear algebra.

The functions `max_abs_error`, `matrix_infinity_norm`, and `residual_norm` provide verification and diagnostic information. The maximum absolute error compares the numerical solution with the manufactured exact solution. The infinity norm of the weighted kernel matrix estimates the size of the discretized integral operator, which helps interpret the perturbation of the identity matrix. The residual norm checks whether the computed vector satisfies the assembled discrete system accurately.

The `main` function brings these components together. It defines the interval, number of grid points, and parameter $\lambda$, constructs the grid and quadrature weights, assembles the weighted kernel and system matrix, solves the dense system, and prints a detailed comparison between the numerical and exact solutions. The output also reports $\|\mathbf K_h\|_\infty$*,* $|\lambda|\|\mathbf K_h\|_\infty$, the maximum absolute error, and the discrete residual, thereby connecting the computational behavior of the program directly to the structure of equations (19.2.4) through (19.2.8).

```rust
// Program 19.2.1: Nyström Discretization of a Second-Kind Fredholm Integral Equation
//
// Problem statement:
// Solve the second-kind Fredholm integral equation
//
//     f(t) = lambda * integral_a^b K(t, s) f(s) ds + g(t),
//
// using the Nyström method. The integral is approximated by composite
// trapezoidal quadrature, giving the discrete system
//
//     (I - lambda * K_h) f = g,
//
// where the quadrature weights are absorbed into the discrete kernel matrix.
// The example uses a manufactured exact solution so that the numerical
// approximation can be checked directly.

use std::f64::consts::PI;

/// Kernel K(t, s) in the Fredholm equation.
///
/// The kernel is smooth on [0, 1] x [0, 1], making it suitable for a clear
/// introductory Nyström example.
fn kernel(t: f64, s: f64) -> f64 {
    t * s
}

/// Manufactured exact solution f(t).
fn exact_solution(t: f64) -> f64 {
    (PI * t).sin()
}

/// Right-hand side g(t), constructed so that the exact solution satisfies
///
///     f(t) = lambda * integral_0^1 K(t, s) f(s) ds + g(t).
///
/// Since K(t, s) = t s and f(s) = sin(pi s),
///
///     integral_0^1 t s sin(pi s) ds = t / pi.
///
/// Therefore,
///
///     g(t) = sin(pi t) - lambda * t / pi.
fn rhs(t: f64, lambda: f64) -> f64 {
    exact_solution(t) - lambda * t / PI
}

/// Build a uniform grid on [a, b] with n points.
fn uniform_grid(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);

    (0..n).map(|i| a + i as f64 * h).collect()
}

/// Composite trapezoidal quadrature weights on [a, b].
fn trapezoidal_weights(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);
    let mut weights = vec![h; n];

    weights[0] = 0.5 * h;
    weights[n - 1] = 0.5 * h;

    weights
}

/// Assemble the weighted Nyström kernel matrix K_h.
///
/// The entries are
///
///     (K_h)_ij = K(t_i, t_j) w_j,
///
/// corresponding to equation (19.2.7).
fn assemble_weighted_kernel(grid: &[f64], weights: &[f64]) -> Vec<Vec<f64>> {
    let n = grid.len();
    let mut weighted_kernel = vec![vec![0.0; n]; n];

    for i in 0..n {
        let t_i = grid[i];

        for j in 0..n {
            let t_j = grid[j];
            weighted_kernel[i][j] = kernel(t_i, t_j) * weights[j];
        }
    }

    weighted_kernel
}

/// Assemble the dense Nyström system matrix
///
///     A = I - lambda * K_h,
///
/// corresponding to equation (19.2.8).
fn assemble_system_matrix(weighted_kernel: &[Vec<f64>], lambda: f64) -> Vec<Vec<f64>> {
    let n = weighted_kernel.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            let identity = if i == j { 1.0 } else { 0.0 };
            matrix[i][j] = identity - lambda * weighted_kernel[i][j];
        }
    }

    matrix
}

/// Assemble the right-hand side vector g_i = g(t_i).
fn assemble_rhs(grid: &[f64], lambda: f64) -> Vec<f64> {
    grid.iter().map(|&t| rhs(t, lambda)).collect()
}

/// Solve a dense linear system using Gaussian elimination with partial pivoting.
///
/// This solver is included to keep the example self-contained. It is suitable
/// for small to moderate textbook demonstrations, while larger problems should
/// use specialized dense or iterative linear algebra libraries.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("The system matrix must be square and compatible with b.".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            let candidate = a[i][k].abs();

            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!(
                "Matrix is numerically singular near column {}.",
                k
            ));
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

/// Compute the maximum absolute error on the grid.
fn max_abs_error(grid: &[f64], numerical: &[f64]) -> f64 {
    grid.iter()
        .zip(numerical.iter())
        .map(|(&t, &f_num)| (f_num - exact_solution(t)).abs())
        .fold(0.0, f64::max)
}

/// Compute the infinity norm of the weighted kernel matrix.
///
/// This gives a simple diagnostic for the size of the discretized integral
/// operator in the system I - lambda K_h.
fn matrix_infinity_norm(matrix: &[Vec<f64>]) -> f64 {
    matrix
        .iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// Compute a residual diagnostic for the discrete Nyström system.
fn residual_norm(matrix: &[Vec<f64>], x: &[f64], b: &[f64]) -> f64 {
    let n = b.len();
    let mut max_residual: f64 = 0.0;

    for i in 0..n {
        let ax_i: f64 = matrix[i]
            .iter()
            .zip(x.iter())
            .map(|(&a_ij, &x_j)| a_ij * x_j)
            .sum();

        max_residual = max_residual.max((ax_i - b[i]).abs());
    }

    max_residual
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let n = 41;
    let lambda = 0.5;

    let grid = uniform_grid(a, b, n);
    let weights = trapezoidal_weights(a, b, n);

    let weighted_kernel = assemble_weighted_kernel(&grid, &weights);
    let kernel_norm = matrix_infinity_norm(&weighted_kernel);

    let system_matrix = assemble_system_matrix(&weighted_kernel, lambda);
    let rhs_vector = assemble_rhs(&grid, lambda);

    let numerical_solution = match solve_linear_system(system_matrix.clone(), rhs_vector.clone()) {
        Ok(solution) => solution,
        Err(message) => {
            eprintln!("Nyström solve failed: {}", message);
            return;
        }
    };

    let error = max_abs_error(&grid, &numerical_solution);
    let residual = residual_norm(&system_matrix, &numerical_solution, &rhs_vector);

    println!("Nyström Method for a Second-Kind Fredholm Equation");
    println!("=================================================");
    println!();
    println!("Model Problem");
    println!("-------------");
    println!("Equation type              = second-kind Fredholm");
    println!("Interval                   = [{:.1}, {:.1}]", a, b);
    println!("Kernel K(t, s)             = t s");
    println!("Exact solution f(t)        = sin(pi t)");
    println!("Lambda                     = {:.6}", lambda);
    println!();
    println!("Nyström Discretization");
    println!("----------------------");
    println!("Grid points N              = {}", n);
    println!("Quadrature rule            = composite trapezoidal rule");
    println!("Weighted kernel entries    = K(t_i, t_j) w_j");
    println!("Linear system              = (I - lambda K_h) f = g");
    println!("Estimated ||K_h||_inf      = {:.6e}", kernel_norm);
    println!("Estimated |lambda| ||K_h|| = {:.6e}", lambda.abs() * kernel_norm);
    println!();
    println!("Numerical Results");
    println!("-----------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "i", "t_i", "f_num", "f_exact", "abs_error"
    );

    for i in 0..n {
        let t = grid[i];
        let f_num = numerical_solution[i];
        let f_exact = exact_solution(t);
        let abs_error = (f_num - f_exact).abs();

        println!(
            "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.6e}",
            i, t, f_num, f_exact, abs_error
        );
    }

    println!();
    println!("Diagnostics");
    println!("-----------");
    println!("max_i |f_num(t_i) - f_exact(t_i)| = {:.6e}", error);
    println!("maximum discrete residual          = {:.6e}", residual);
}
```

Program 19.2.1 demonstrates how the Nyström method converts a second-kind Fredholm integral equation into a dense algebraic system. The continuous kernel operator is replaced by a weighted kernel matrix, while the identity term remains explicit in the matrix $I-\lambda\mathbf K$. This directly reflects the formulation in equation (19.2.8) and shows why second-kind equations are often more stable in direct discretization than first-kind equations.

The numerical output confirms the expected behavior of the method. The maximum discrete residual is close to machine precision, showing that the linear system has been solved accurately. The maximum absolute error measures the difference between the computed solution and the manufactured exact solution, and its small value confirms that the quadrature-based discretization is consistent for this smooth kernel and smooth solution. The nonzero endpoint error is a normal discretization effect and not a failure of the method.

The diagnostic value $|\lambda|\|\mathbf K_h\|_\infty$ is also useful because it indicates the relative size of the integral perturbation compared with the identity contribution. In this example, the perturbation is moderate, which is consistent with the well-behaved numerical solution. For larger values of $\lambda$, more oscillatory kernels, or kernels with singular behavior, conditioning and accuracy may become more delicate.

Although the program uses a simple trapezoidal quadrature rule and a dense direct solver, the same structure underlies more advanced Fredholm solvers. Higher-order quadrature, spectral collocation, Galerkin projection, iterative Krylov methods, hierarchical compression, and fast multipole acceleration all build on the same basic idea: replacing the continuous operator by a finite-dimensional approximation while preserving the second-kind structure of the original equation.

## 19.2.3. Computational Complexity, Error Convergence, and Stability of Numerical Solvers

The computational cost of solving $(I-\lambda\mathbf K)\mathbf f=\mathbf g$, depends strongly on the discretization and linear solver. In the most direct approach, forming and solving the dense $N\times N$ system requires $O(N^3)$ operations, with the dominant cost coming from dense linear system solution. If only matrix-vector products are required, as in iterative methods, each iteration costs $O(N^2)$ for a dense kernel matrix. With hierarchical matrix compression or fast multipole acceleration, this cost can be reduced to $O(N\log N)$ or $O(N)$ for suitable smooth or structured kernels (Nature Index, 2025).

Memory usage follows a similar pattern. A dense discretization stores all entries of $\mathbf K$, requiring $O(N^2)$ memory. Compressed representations, such as low-rank blocks, hierarchical matrices, or fast multipole data structures, may reduce memory to $O(N\log N)$ or $O(N)$, depending on the kernel structure and approximation tolerance.

The discretization error depends on the quadrature rule or basis approximation. If the quadrature or basis has order $p$, then the error typically behaves like:

$$O(N^{-p}), \tag{19.2.13}$$

or better, depending on the smoothness of $K$, $g$, and $f$. For analytic kernels and smooth solutions, spectral convergence can occur, meaning that the error decreases faster than any fixed algebraic power of $N$. This is one reason Chebyshev and Legendre collocation methods are effective for smooth Fredholm equations.

Iterative error depends on the convergence behavior of the chosen iteration. The Neumann series converges when,

$$|\lambda|\|T\|<1 \tag{19.2.14}$$

When this condition is not satisfied, fixed-point iteration may fail or converge too slowly, and Krylov methods such as GMRES are usually preferred. The convergence of Krylov methods depends on the spectral properties of $I-\lambda\mathbf K$, the conditioning of the system, and the quality of any preconditioning used.

Although Fredholm equations of the second kind are usually better conditioned than first-kind equations, ill-conditioning can still arise. This is especially true when a first-kind equation is reformulated or regularized into a second-kind form, or when $\lambda$ lies near a characteristic value of the operator. In such cases, regularization techniques such as Tikhonov regularization or truncated singular-value decomposition may be needed to stabilize the computation.

### Rust Implementation

Following the discussion in Section 19.2.3 on computational complexity, error convergence, and stability of numerical solvers, Program 19.2.2 provides a practical implementation of fixed-point iteration for a discretized Fredholm equation of the second kind. The program illustrates the iterative form associated with the Neumann-series interpretation in equation (19.2.12) and connects it directly to the convergence condition in equation (19.2.14). Instead of solving the dense system by Gaussian elimination, the method repeatedly applies the Nyström approximation of the integral operator through matrix-vector products. This makes the example useful for distinguishing discretization error from iterative error, and for showing why simple fixed-point iteration works well only when the integral perturbation is sufficiently small. By comparing a contractive and a noncontractive case, the program demonstrates why more robust Krylov methods are often preferred when the Neumann condition is not satisfied.

At the core of the implementation is the repeated application of the discretized integral operator that appears in the fixed-point form of equation (19.2.12). The function `kernel(t, s)` defines the kernel $K(t,s)$ used in the Fredholm equation, while `exact_solution(t)` provides a manufactured solution for verification. The function `rhs(t, lambda)` constructs the corresponding right-hand side so that the exact solution satisfies the continuous second-kind Fredholm equation. This allows the final numerical result to be compared directly with a known solution.

The functions `uniform_grid` and `trapezoidal_weights` construct the quadrature grid and weights used in the Nyström approximation. The grid defines the points at which the unknown function is represented, while the trapezoidal weights approximate the integral over the fixed interval. These functions provide the finite-dimensional setting in which the continuous operator $T$ is replaced by a discrete operator $T_h$.

The function `apply_integral_operator` evaluates the action of the discretized integral operator on a vector of nodal values. It computes the weighted kernel sum corresponding to the Nyström approximation, but it does so without explicitly storing the full dense matrix. This is important for the computational interpretation of Section 19.2.3: even when the dense matrix is not stored, a direct kernel summation still requires $O(N^2)$ operations per iteration for a general dense kernel. Thus, the function demonstrates the matrix-vector product viewpoint used by iterative solvers.

The function `assemble_rhs` evaluates the right-hand side at the grid points. Together with repeated applications of `apply_integral_operator`, this produces the fixed-point update in equation (19.2.12). The function `kernel_infinity_norm` estimates the infinity norm of the discretized integral operator. This diagnostic is used to compute $|\lambda|\|\mathbf K_h\|_\infty$, which provides a finite-dimensional analogue of the convergence condition in equation (19.2.14).

The function `max_difference` measures the infinity norm of the change between two successive iterates. This quantity is used as the stopping criterion for the fixed-point iteration. The function `max_abs_error` compares the computed result with the manufactured exact solution, while `residual_norm` checks how well the computed vector satisfies the discrete fixed-point equation. These diagnostics separate the accuracy of the discretization from the convergence behavior of the iterative solver.

The `FixedPointResult` structure stores the output of the fixed-point iteration, including the computed solution, number of iterations, convergence flag, and final update norm. The function `fixed_point_iteration` performs the actual iteration. Starting from the zero vector, it repeatedly computes the next approximation using the discrete version of equation (19.2.12). The iteration stops when the update norm falls below the prescribed tolerance or when the maximum number of iterations is reached. A safeguard is also included to stop the process if the update becomes excessively large or non-finite.

The function `run_experiment` organizes one complete numerical experiment for a selected value of $\lambda$. It assembles the right-hand side, estimates the size of the discrete integral perturbation, runs the fixed-point solver, computes error and residual diagnostics, and prints representative solution values. This design makes it easy to compare different stability regimes without duplicating code.

The `main` function demonstrates two cases. The first uses a moderate value of (\\lambda), for which the estimated contraction indicator is less than one and the fixed-point iteration converges. The second uses a larger value of $\lambda$, for which the contraction indicator exceeds one and the iteration fails to converge within the iteration limit. This comparison directly illustrates the meaning of the convergence condition in equation (19.2.14) and reinforces the stability discussion in Section 19.2.3.

```rust
// Program 19.2.2: Fixed-Point Iteration and Neumann-Series Convergence
// for a Fredholm Equation
//
// Problem statement:
// Solve the second-kind Fredholm integral equation
//
//     f(t) = g(t) + lambda * integral_a^b K(t, s) f(s) ds,
//
// by fixed-point iteration,
//
//     f_{n+1} = g + lambda * T_h(f_n),
//
// where T_h is the Nyström quadrature approximation of the integral operator.
// The program demonstrates two cases:
//
// 1. A contractive case, where fixed-point iteration converges.
// 2. A noncontractive case, where fixed-point iteration fails or becomes unstable.
//
// This illustrates the convergence condition associated with the Neumann
// series and explains why Krylov methods are often preferred when simple
// fixed-point iteration is not reliable.

use std::f64::consts::PI;

/// Kernel K(t, s) in the Fredholm integral equation.
///
/// The kernel K(t, s) = t s is smooth and rank-one, making it useful for
/// demonstrating the difference between direct discretization and iterative
/// application of the integral operator.
fn kernel(t: f64, s: f64) -> f64 {
    t * s
}

/// Manufactured exact solution f(t).
fn exact_solution(t: f64) -> f64 {
    (PI * t).sin()
}

/// Right-hand side g(t), constructed so that the exact solution satisfies
///
///     f(t) = g(t) + lambda * integral_0^1 K(t, s) f(s) ds.
///
/// Since
///
///     integral_0^1 t s sin(pi s) ds = t / pi,
///
/// we have
///
///     g(t) = sin(pi t) - lambda * t / pi.
fn rhs(t: f64, lambda: f64) -> f64 {
    exact_solution(t) - lambda * t / PI
}

/// Build a uniform grid on [a, b] with n points.
fn uniform_grid(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);

    (0..n)
        .map(|i| a + i as f64 * h)
        .collect()
}

/// Composite trapezoidal quadrature weights on [a, b].
fn trapezoidal_weights(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);
    let mut weights = vec![h; n];

    weights[0] = 0.5 * h;
    weights[n - 1] = 0.5 * h;

    weights
}

/// Apply the Nyström approximation T_h to a vector f.
///
/// This function computes
///
///     (T_h f)_i = sum_j K(t_i, t_j) w_j f_j.
///
/// The operation is equivalent to a dense matrix-vector product but avoids
/// explicitly storing the full kernel matrix. The cost is still O(N^2) for
/// a general dense kernel.
fn apply_integral_operator(grid: &[f64], weights: &[f64], f: &[f64]) -> Vec<f64> {
    let n = grid.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let t_i = grid[i];
        let mut sum = 0.0;

        for j in 0..n {
            let s_j = grid[j];
            sum += kernel(t_i, s_j) * weights[j] * f[j];
        }

        result[i] = sum;
    }

    result
}

/// Assemble the right-hand side vector g_i = g(t_i).
fn assemble_rhs(grid: &[f64], lambda: f64) -> Vec<f64> {
    grid.iter()
        .map(|&t| rhs(t, lambda))
        .collect()
}

/// Estimate ||K_h||_infinity without storing K_h.
///
/// This diagnostic gives a simple sufficient indicator for the size of
/// |lambda| ||K_h|| appearing in the fixed-point convergence discussion.
fn kernel_infinity_norm(grid: &[f64], weights: &[f64]) -> f64 {
    let mut max_row_sum = 0.0;

    for &t_i in grid {
        let mut row_sum = 0.0;

        for j in 0..grid.len() {
            let s_j = grid[j];
            row_sum += (kernel(t_i, s_j) * weights[j]).abs();
        }

        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }

    max_row_sum
}

/// Compute the infinity norm of the difference between two vectors.
fn max_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Compute the maximum absolute error relative to the manufactured solution.
fn max_abs_error(grid: &[f64], numerical: &[f64]) -> f64 {
    grid.iter()
        .zip(numerical.iter())
        .map(|(&t, &f_num)| (f_num - exact_solution(t)).abs())
        .fold(0.0, f64::max)
}

/// Compute the residual of the discrete fixed-point equation:
///
///     r = f - g - lambda K_h f.
fn residual_norm(grid: &[f64], weights: &[f64], f: &[f64], g: &[f64], lambda: f64) -> f64 {
    let tf = apply_integral_operator(grid, weights, f);

    f.iter()
        .zip(g.iter())
        .zip(tf.iter())
        .map(|((&f_i, &g_i), &tf_i)| (f_i - g_i - lambda * tf_i).abs())
        .fold(0.0, f64::max)
}

/// Result returned by the fixed-point solver.
struct FixedPointResult {
    solution: Vec<f64>,
    iterations: usize,
    converged: bool,
    last_update_norm: f64,
}

/// Fixed-point iteration:
///
///     f_{n+1} = g + lambda T_h(f_n).
///
/// The iteration stops when the infinity norm of the update is below the
/// requested tolerance or when the iteration limit is reached.
fn fixed_point_iteration(
    grid: &[f64],
    weights: &[f64],
    g: &[f64],
    lambda: f64,
    tolerance: f64,
    max_iterations: usize,
) -> FixedPointResult {
    let n = grid.len();

    let mut current = vec![0.0; n];
    let mut next = vec![0.0; n];

    let mut last_update_norm = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;

    for iteration in 1..=max_iterations {
        let integral_current = apply_integral_operator(grid, weights, &current);

        for i in 0..n {
            next[i] = g[i] + lambda * integral_current[i];
        }

        last_update_norm = max_difference(&next, &current);
        iterations = iteration;

        if last_update_norm < tolerance {
            converged = true;
            current.clone_from(&next);
            break;
        }

        if !last_update_norm.is_finite() || last_update_norm > 1.0e12 {
            converged = false;
            current.clone_from(&next);
            break;
        }

        current.clone_from(&next);
    }

    FixedPointResult {
        solution: current,
        iterations,
        converged,
        last_update_norm,
    }
}

/// Run one fixed-point experiment for a selected lambda value.
fn run_experiment(lambda: f64, grid: &[f64], weights: &[f64], tolerance: f64, max_iterations: usize) {
    let g = assemble_rhs(grid, lambda);
    let kernel_norm = kernel_infinity_norm(grid, weights);
    let contraction_indicator = lambda.abs() * kernel_norm;

    let result = fixed_point_iteration(
        grid,
        weights,
        &g,
        lambda,
        tolerance,
        max_iterations,
    );

    let error = max_abs_error(grid, &result.solution);
    let residual = residual_norm(grid, weights, &result.solution, &g, lambda);

    println!("Experiment with lambda = {:.6}", lambda);
    println!("--------------------------------");
    println!("Estimated ||K_h||_inf           = {:.6e}", kernel_norm);
    println!(
        "Estimated |lambda| ||K_h||_inf  = {:.6e}",
        contraction_indicator
    );
    println!("Tolerance                       = {:.6e}", tolerance);
    println!("Maximum iterations              = {}", max_iterations);
    println!("Iterations performed            = {}", result.iterations);
    println!("Converged                       = {}", result.converged);
    println!(
        "Final update norm               = {:.6e}",
        result.last_update_norm
    );
    println!("Maximum absolute error          = {:.6e}", error);
    println!("Maximum discrete residual       = {:.6e}", residual);
    println!();

    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "i", "t_i", "f_iter", "f_exact", "abs_error"
    );

    let n = grid.len();

    for i in 0..n {
        let t = grid[i];
        let f_num = result.solution[i];
        let f_exact = exact_solution(t);
        let abs_error = (f_num - f_exact).abs();

        if i % 5 == 0 || i == n - 1 {
            println!(
                "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.6e}",
                i, t, f_num, f_exact, abs_error
            );
        }
    }

    println!();
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let n = 41;

    let tolerance = 1.0e-10;
    let max_iterations = 80;

    let grid = uniform_grid(a, b, n);
    let weights = trapezoidal_weights(a, b, n);

    println!("Fixed-Point Iteration for a Second-Kind Fredholm Equation");
    println!("=========================================================");
    println!();
    println!("Model Problem");
    println!("-------------");
    println!("Equation type              = second-kind Fredholm");
    println!("Interval                   = [{:.1}, {:.1}]", a, b);
    println!("Kernel K(t, s)             = t s");
    println!("Exact solution f(t)        = sin(pi t)");
    println!("Iteration                  = f_(n+1) = g + lambda K_h f_n");
    println!("Grid points N              = {}", n);
    println!("Quadrature rule            = composite trapezoidal rule");
    println!();

    println!("Convergent Case");
    println!("===============");
    run_experiment(0.5, &grid, &weights, tolerance, max_iterations);

    println!("Noncontractive Case");
    println!("===================");
    run_experiment(3.2, &grid, &weights, tolerance, max_iterations);
}
```

Program 19.2.2 demonstrates that fixed-point iteration for a second-kind Fredholm equation is simple to implement but strongly dependent on the size of the discretized integral operator. In the contractive case, the iteration converges rapidly, the update norm decreases below the prescribed tolerance, and the discrete residual becomes small. The remaining difference between the numerical and exact solutions is then mainly due to quadrature discretization error, not failure of the iterative solver.

The noncontractive case shows the limitation of the same method. When the estimated value of $|\lambda|\|\mathbf K_h\|_\infty$ exceeds one, the fixed-point iteration no longer behaves reliably. The update norm remains large, the residual does not decrease to the desired tolerance, and the computed solution departs substantially from the manufactured exact solution. This confirms that the Neumann convergence condition in equation (19.2.14) has practical numerical significance.

This example also clarifies the cost distinction discussed in Section 19.2.3. The method avoids dense factorization, so it does not incur the $O(N^3)$ cost of a direct solve. However, each application of the dense kernel operator still costs $O(N^2)$ operations in this simple implementation. For larger problems, this matrix-vector product viewpoint becomes the basis for more advanced approaches, including Krylov methods, hierarchical compression, and fast multipole acceleration. Thus, the program provides a bridge from elementary fixed-point iteration to the more scalable solvers discussed later in the section.

## 19.2.4. Classical Quadrature, Fast Algorithms, and Modern Data-Driven Fredholm Solvers

Classical numerical treatment of Fredholm equations often begins with Gaussian quadrature, Nyström discretization, polynomial collocation, or Galerkin projection. These methods are effective for moderate problem sizes and remain foundational because they translate the integral equation into a clear finite-dimensional system. Modern methods extend this foundation by addressing large-scale, oscillatory, singular, stochastic, and high-dimensional kernels.

For oscillatory kernels, one may use quadrature rules adapted to oscillation. For stochastic problems, Monte Carlo integration with variance reduction may be appropriate. For large-scale or multidimensional kernels, sparse grids, low-rank tensor decompositions, hierarchical matrices, and fast multipole methods can substantially reduce computational cost. In physics and engineering, specialized boundary integral solvers for acoustics and electromagnetics often combine high-order Nyström discretizations with quadrature methods designed for singular or nearly singular kernels. Fast multipole and hierarchical matrix techniques are now central to large-scale Fredholm solvers, enabling computations in electromagnetics, fluid dynamics, and materials science that would otherwise be computationally prohibitive (Nature Index, 2025).

A representative application appears in *acoustic and electromagnetic scattering*. In such problems, a boundary-value partial differential equation is often reformulated as a boundary integral equation on the scatterer’s surface. For example, the Helmholtz equation outside an obstacle leads to a Fredholm integral equation of the second kind for an unknown surface density $f$. Discretization by a Nyström or Galerkin method produces a dense linear system. Modern solvers use specialized quadrature for singular or nearly singular kernels and then apply fast multipole acceleration, allowing wave scattering in complex geometries to be simulated with cost close to linear in the number of boundary elements (Nature Index, 2025).

Another important application occurs in *quantum mechanics and condensed matter physics*. In quantum scattering, such as electron-atom collision problems, the Lippmann-Schwinger equation is a Fredholm integral equation of the second kind. In this setting, $f$ represents the scattering state and $g$ represents the incident wave. Solving the integral equation for $f$ gives access to scattering amplitudes and cross-sections. Efficient discretization and solution of such Fredholm equations are therefore crucial for predictive quantum scattering computations. Recent work has shown that PINN-based solvers can handle high-dimensional variants of these problems and can improve stability for singular or oscillatory kernels (Ma and Li, 2025).

### Rust Implementation

Following the discussion in Section 19.2.4 on classical quadrature, fast algorithms, and modern data-driven Fredholm solvers, Program 19.2.3 provides a matrix-free Krylov implementation for a discretized Fredholm equation of the second kind. The program retains the Nyström quadrature framework but avoids explicitly assembling the dense weighted kernel matrix. Instead, it applies the system operator through repeated kernel-based matrix-vector products, which is the computational pattern underlying GMRES, hierarchical matrix methods, fast multipole acceleration, and many large-scale boundary-integral solvers. This example therefore bridges the classical discretization viewpoint with modern scalable algorithms: the dense Fredholm operator is still present mathematically, but the solver interacts with it only through its action on vectors. The manufactured discrete solution allows the algebraic accuracy of the matrix-free GMRES solver to be verified directly.

At the core of the implementation is the separation between the Fredholm operator and its matrix representation. The function `kernel(t, s)` defines the smooth dense kernel used in the Nyström discretization, while `exact_solution(t)` defines a manufactured nodal solution. Unlike earlier examples that construct the right-hand side from a continuous integral identity, this program constructs the right-hand side from the discrete operator itself. This makes the test focus on the correctness of the matrix-free GMRES solver rather than on quadrature error.

The functions `uniform_grid` and `trapezoidal_weights` define the quadrature nodes and weights used to approximate the Fredholm integral operator. These components preserve the classical Nyström foundation discussed earlier in the section. The grid gives the nodal representation of the unknown function, while the trapezoidal weights define the weighted contribution of each quadrature point to the discrete integral operator.

The function `apply_weighted_kernel` computes the action of the weighted Nyström kernel on a vector. It evaluates the same weighted kernel summation that would be represented by a dense matrix, but it does not store that matrix. This distinction is central to the purpose of the program. The computational cost of this direct implementation is still $O(N^2)$ per matrix-vector product, but the storage cost of explicitly keeping all $N^2$ dense entries is avoided. The same function is also the natural place where a later implementation could substitute a hierarchical matrix product, low-rank approximation, fast multipole product, or specialized quadrature method.

The function `apply_system_operator` applies the full second-kind system operator associated with the discretized equation. It combines the identity contribution with the weighted kernel contribution, preserving the second-kind structure of the Fredholm equation. This is the operator that GMRES sees. The Krylov solver does not need access to individual matrix entries; it only requires the ability to compute the result of applying this operator to a vector.

The function `manufactured_rhs` constructs the discrete right-hand side by first sampling the manufactured exact solution on the grid and then applying the matrix-free system operator to that vector. This produces a right-hand side that is exactly consistent with the discrete problem. Consequently, the final error measures the algebraic accuracy of GMRES and the operator application, rather than the difference between a continuous integral equation and its quadrature approximation.

The helper functions `dot`, `norm2`, `axpy`, and `difference` provide the vector operations needed by GMRES. The function `dot` computes inner products, `norm2` evaluates Euclidean norms, `axpy` performs scaled vector additions, and `difference` forms residual vectors. These operations are the basic linear algebra building blocks of Krylov subspace methods.

The function `solve_small_system` solves the small dense systems that arise inside the GMRES least-squares step. It uses Gaussian elimination with partial pivoting. This is not intended as a large-scale solver; it is used only for the low-dimensional algebraic problem generated inside the Krylov iteration. The function `solve_gmres_least_squares` forms and solves the small normal equations associated with the GMRES minimization problem. In production implementations, this step is usually handled by Givens rotations or Householder transformations for improved numerical stability, but the compact form used here is appropriate for a textbook demonstration.

The function `gmres_matrix_free` implements the matrix-free GMRES iteration. It begins with the zero initial guess, forms the initial residual, normalizes it to start the Arnoldi basis, and then repeatedly expands the Krylov subspace by applying the system operator. At each iteration, the new vector is orthogonalized against the existing basis vectors, the small Hessenberg matrix is updated, and a least-squares problem is solved to obtain the best approximation within the current Krylov subspace. The iteration stops when the residual norm falls below the prescribed tolerance or when the maximum number of iterations is reached.

The function `max_abs_difference` compares the GMRES solution with the manufactured nodal solution, while `kernel_infinity_norm` estimates the size of the discretized integral operator without storing the full kernel matrix. This norm provides a diagnostic for the magnitude of the integral perturbation relative to the identity contribution. In the `main` function, the program sets the interval, grid size, value of (\\lambda), iteration limit, and tolerance, constructs the quadrature data, builds the manufactured right-hand side, runs GMRES, and prints representative solution values. It also reports the number of dense entries and storage bytes avoided by not assembling the full matrix.

```rust
// Program 19.2.3: Matrix-Free GMRES Solver for a Discretized Fredholm Equation
//
// Problem statement:
// Solve the discretized second-kind Fredholm equation
//
//     (I - lambda K_h) f = g,
//
// using a matrix-free GMRES method. The weighted kernel matrix K_h is not
// assembled explicitly. Instead, the action of the operator is computed by
// applying the quadrature-based integral operator directly.
//
// This illustrates the computational idea behind scalable Fredholm solvers:
// direct dense factorization requires O(N^3) operations and O(N^2) storage,
// while Krylov methods use repeated matrix-vector products. In a dense
// implementation each product still costs O(N^2), but the matrix-free form
// is the natural entry point for hierarchical compression, fast multipole
// acceleration, and other large-scale methods.

use std::f64::consts::PI;

/// Smooth dense kernel K(t, s).
///
/// This kernel produces a fully populated Nyström operator, making it useful
/// for demonstrating why matrix-free methods are attractive for large problems.
fn kernel(t: f64, s: f64) -> f64 {
    1.0 / (1.0 + 25.0 * (t - s).powi(2))
}

/// Manufactured nodal solution.
fn exact_solution(t: f64) -> f64 {
    (PI * t).sin() + 0.25 * (2.0 * PI * t).cos()
}

/// Build a uniform grid on [a, b] with n points.
fn uniform_grid(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);
    (0..n).map(|i| a + i as f64 * h).collect()
}

/// Composite trapezoidal quadrature weights on [a, b].
fn trapezoidal_weights(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);
    let mut weights = vec![h; n];

    weights[0] = 0.5 * h;
    weights[n - 1] = 0.5 * h;

    weights
}

/// Apply the weighted Nyström integral operator K_h to a vector x.
///
/// This computes
///
///     (K_h x)_i = sum_j K(t_i, t_j) w_j x_j,
///
/// without explicitly storing the dense matrix K_h.
fn apply_weighted_kernel(grid: &[f64], weights: &[f64], x: &[f64]) -> Vec<f64> {
    let n = grid.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let t_i = grid[i];
        let mut sum = 0.0;

        for j in 0..n {
            let s_j = grid[j];
            sum += kernel(t_i, s_j) * weights[j] * x[j];
        }

        result[i] = sum;
    }

    result
}

/// Apply the full second-kind Fredholm system operator
///
///     A x = (I - lambda K_h) x.
fn apply_system_operator(grid: &[f64], weights: &[f64], lambda: f64, x: &[f64]) -> Vec<f64> {
    let kx = apply_weighted_kernel(grid, weights, x);

    x.iter()
        .zip(kx.iter())
        .map(|(&x_i, &kx_i)| x_i - lambda * kx_i)
        .collect()
}

/// Construct a discrete manufactured right-hand side.
///
/// The exact nodal vector f_exact is chosen first, and then
///
///     g = (I - lambda K_h) f_exact
///
/// is computed using the same matrix-free operator. This verifies the Krylov
/// solver independently of quadrature error.
fn manufactured_rhs(grid: &[f64], weights: &[f64], lambda: f64) -> (Vec<f64>, Vec<f64>) {
    let exact: Vec<f64> = grid.iter().map(|&t| exact_solution(t)).collect();
    let rhs = apply_system_operator(grid, weights, lambda, &exact);

    (exact, rhs)
}

/// Dot product of two vectors.
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

/// Euclidean norm of a vector.
fn norm2(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

/// Compute y <- y + alpha x.
fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for i in 0..y.len() {
        y[i] += alpha * x[i];
    }
}

/// Compute x - y.
fn difference(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y.iter()).map(|(&a, &b)| a - b).collect()
}

/// Solve a small dense linear system using Gaussian elimination with partial pivoting.
///
/// This is used only for the small least-squares normal equations inside GMRES.
fn solve_small_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            if a[i][k].abs() > pivot_abs {
                pivot_abs = a[i][k].abs();
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!(
                "Small GMRES least-squares system is singular near column {}.",
                k
            ));
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

/// Solve the GMRES least-squares problem
///
///     min_y || beta e_1 - H y ||_2
///
/// by forming and solving the small normal equations. This is sufficient for
/// a compact textbook implementation. Production solvers usually use Givens
/// rotations or Householder transformations for better numerical robustness.
fn solve_gmres_least_squares(h: &[Vec<f64>], beta: f64, k: usize) -> Result<Vec<f64>, String> {
    let rows = k + 1;
    let cols = k;

    let mut normal_matrix = vec![vec![0.0; cols]; cols];
    let mut normal_rhs = vec![0.0; cols];

    for i in 0..cols {
        for j in 0..cols {
            let mut sum = 0.0;

            for r in 0..rows {
                sum += h[r][i] * h[r][j];
            }

            normal_matrix[i][j] = sum;
        }

        normal_rhs[i] = beta * h[0][i];
    }

    solve_small_system(normal_matrix, normal_rhs)
}

/// Matrix-free GMRES for A x = b, where A = I - lambda K_h.
fn gmres_matrix_free(
    grid: &[f64],
    weights: &[f64],
    lambda: f64,
    b: &[f64],
    max_iterations: usize,
    tolerance: f64,
) -> Result<(Vec<f64>, usize, f64), String> {
    let n = b.len();
    let mut x = vec![0.0; n];

    let ax = apply_system_operator(grid, weights, lambda, &x);
    let r0 = difference(b, &ax);
    let beta = norm2(&r0);

    if beta < tolerance {
        return Ok((x, 0, beta));
    }

    let mut v = vec![vec![0.0; n]; max_iterations + 1];

    for i in 0..n {
        v[0][i] = r0[i] / beta;
    }

    let mut h = vec![vec![0.0; max_iterations]; max_iterations + 1];

    let mut best_residual = beta;
    let mut best_iteration = 0;

    for j in 0..max_iterations {
        let mut w = apply_system_operator(grid, weights, lambda, &v[j]);

        for i in 0..=j {
            h[i][j] = dot(&w, &v[i]);
            axpy(-h[i][j], &v[i], &mut w);
        }

        h[j + 1][j] = norm2(&w);

        if h[j + 1][j] > 1.0e-14 {
            for i in 0..n {
                v[j + 1][i] = w[i] / h[j + 1][j];
            }
        }

        let k = j + 1;
        let y = solve_gmres_least_squares(&h, beta, k)?;

        let mut candidate = vec![0.0; n];

        for col in 0..k {
            axpy(y[col], &v[col], &mut candidate);
        }

        let residual_vector = difference(
            b,
            &apply_system_operator(grid, weights, lambda, &candidate),
        );

        let residual = norm2(&residual_vector);
        best_residual = residual;
        best_iteration = k;
        x = candidate;

        if residual < tolerance {
            break;
        }

        if h[j + 1][j] <= 1.0e-14 {
            break;
        }
    }

    Ok((x, best_iteration, best_residual))
}

/// Maximum absolute difference between two vectors.
fn max_abs_difference(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Estimate the infinity norm of K_h without storing it.
fn kernel_infinity_norm(grid: &[f64], weights: &[f64]) -> f64 {
    let mut max_row_sum = 0.0;

    for &t_i in grid {
        let mut row_sum = 0.0;

        for j in 0..grid.len() {
            row_sum += (kernel(t_i, grid[j]) * weights[j]).abs();
        }

        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }

    max_row_sum
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let n = 80;
    let lambda = 0.5;
    let max_iterations = 30;
    let tolerance = 1.0e-10;

    let grid = uniform_grid(a, b, n);
    let weights = trapezoidal_weights(a, b, n);

    let (exact, rhs) = manufactured_rhs(&grid, &weights, lambda);

    let kernel_norm = kernel_infinity_norm(&grid, &weights);
    let dense_entries = n * n;
    let dense_storage_bytes = dense_entries * std::mem::size_of::<f64>();

    let (solution, iterations, residual) = match gmres_matrix_free(
        &grid,
        &weights,
        lambda,
        &rhs,
        max_iterations,
        tolerance,
    ) {
        Ok(result) => result,
        Err(message) => {
            eprintln!("GMRES failed: {}", message);
            return;
        }
    };

    let max_error = max_abs_difference(&solution, &exact);

    println!("Matrix-Free GMRES for a Second-Kind Fredholm Equation");
    println!("====================================================");
    println!();
    println!("Model Problem");
    println!("-------------");
    println!("Equation type              = second-kind Fredholm");
    println!("Interval                   = [{:.1}, {:.1}]", a, b);
    println!("Kernel K(t, s)             = 1 / (1 + 25(t - s)^2)");
    println!("Manufactured nodal solution= sin(pi t) + 0.25 cos(2 pi t)");
    println!("Lambda                     = {:.6}", lambda);
    println!();
    println!("Matrix-Free Discretization");
    println!("--------------------------");
    println!("Grid points N              = {}", n);
    println!("Quadrature rule            = composite trapezoidal rule");
    println!("Operator form              = A x = (I - lambda K_h) x");
    println!("Dense matrix assembled     = no");
    println!("Dense entries avoided      = {}", dense_entries);
    println!("Dense storage avoided      = {} bytes", dense_storage_bytes);
    println!("Estimated ||K_h||_inf      = {:.6e}", kernel_norm);
    println!(
        "Estimated |lambda| ||K_h|| = {:.6e}",
        lambda.abs() * kernel_norm
    );
    println!();
    println!("GMRES Parameters");
    println!("----------------");
    println!("Maximum iterations         = {}", max_iterations);
    println!("Tolerance                  = {:.6e}", tolerance);
    println!("Iterations performed       = {}", iterations);
    println!("Final residual norm        = {:.6e}", residual);
    println!();
    println!("Numerical Results");
    println!("-----------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "i", "t_i", "f_gmres", "f_exact", "abs_error"
    );

    for i in 0..n {
        if i % 10 == 0 || i == n - 1 {
            let t = grid[i];
            let f_num = solution[i];
            let f_exact = exact[i];
            let abs_error = (f_num - f_exact).abs();

            println!(
                "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.6e}",
                i, t, f_num, f_exact, abs_error
            );
        }
    }

    println!();
    println!("Diagnostics");
    println!("-----------");
    println!("max_i |f_gmres(t_i) - f_exact(t_i)| = {:.6e}", max_error);
    println!("final GMRES residual norm           = {:.6e}", residual);
}
```

Program 19.2.3 demonstrates how the dense Fredholm system produced by a Nyström discretization can be solved through a matrix-free Krylov approach. The key idea is that the solver never needs the full matrix explicitly. It only needs an operation that maps a vector $x$ to the corresponding system-vector product. This is precisely the interface required by GMRES and by many scalable solvers for boundary integral equations and scattering problems.

The numerical output verifies the algebraic correctness of the implementation. Since the right-hand side is manufactured from the discrete operator, the very small final error shows that GMRES has recovered the prescribed nodal solution to near machine precision. The final residual norm confirms that the computed vector satisfies the discretized system accurately. These diagnostics are especially useful because they separate solver accuracy from quadrature accuracy.

The storage diagnostic reinforces the complexity discussion in Section 19.2.4. A dense Nyström matrix would require $N^2$ entries, while this implementation avoids storing those entries and works through repeated operator applications. In the present self-contained code, each direct kernel summation still costs $O(N^2)$, but the design makes clear how modern acceleration enters: replacing the kernel application with a fast multipole, hierarchical matrix, low-rank, or specialized quadrature operator would reduce the cost while preserving the same GMRES framework.

This program therefore provides a practical bridge between classical quadrature-based discretization and modern large-scale Fredholm solvers. It shows how the finite-dimensional system remains grounded in the Nyström method, while the computational strategy shifts from dense matrix factorization to operator-based iteration. This is the essential transition behind contemporary solvers for scattering, boundary-integral formulations, and high-dimensional integral-equation models.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/2zRGgeXTJsRYiLx9MOeT.6","tags":[]}

# 19.3. Volterra Integral Equations and Causal Memory Models

Volterra integral equations form one of the most important classes of integral equations in numerical analysis because their integration interval changes with the independent variable. This variable upper limit gives the equation a directional structure: the unknown value at the current point is determined only by values at earlier points. For this reason, Volterra equations are closely connected with initial-value problems, time-dependent processes, and models in which the present state depends on accumulated history. They provide a natural framework for describing memory effects, hereditary material behavior, population processes, control systems, and fractional-order dynamics. From a computational point of view, this causal structure leads to algebraic systems with lower-triangular or step-by-step dependence, making their numerical treatment fundamentally different from the global coupling that appears in Fredholm equations.

The central idea in this section is that Volterra equations can often be solved progressively along the independent variable, much like a time-marching method for ordinary differential equations. At each new point, the integral involves only previously computed values and, depending on the quadrature rule, possibly the current unknown. This makes Volterra equations especially suitable for quadrature-based discretization, sequential numerical solution, and adaptive refinement in problems where memory effects are important. The following discussion develops their mathematical form, causal interpretation, discretization structure, and relevance to applications such as population dynamics, viscoelasticity, fractional-order systems, and other models governed by accumulated past influence.

## 19.3.1. Mathematical Form, Causal Structure, and Memory Interpretation

A Volterra integral equation of the second kind has the prototype form:

$$f(t)=\lambda\int_a^t K(t,s)f(s)\,ds+g(t),\qquad t\in[a,b] \tag{19.3.1}$$

Here, $f(t)$ is the unknown function, $K(t,s)$ is the kernel, $g(t)$ is a prescribed forcing or input function, and $\lambda$ is a scalar parameter controlling the strength of the integral contribution. The variable $t$ represents the current point, while $s$ is the integration variable ranging only over the interval from the initial point $a$ to the present point $t$. The defining feature of a Volterra equation is therefore not merely the presence of an integral term, but the fact that the upper limit of integration is the independent variable itself.

This feature gives equation (19.3.1) a causal structure. In a Fredholm equation, the integral usually extends over a fixed interval $[a,b]$, so the value of $f(t)$ may depend on unknown values $f(s)$ across the whole domain, including points that lie ahead of $t$. In contrast, the Volterra equation accumulates information only over the interval $[a,t]$. Thus, the present value $f(t)$ depends only on $f(s)$ for $s\leq t$. This one-sided dependence is the mathematical reason why Volterra equations resemble time-evolution problems and why their discretizations often lead to sequential solution procedures.

The causal interpretation can be made more explicit by reading the integral term in equation (19.3.1) as a weighted history. For each current point $t$, the kernel $K(t,s)$ assigns a weight to the past value $f(s)$. The integral then sums all such weighted contributions from the initial point $a$ up to the current point $t$. The function $g(t)$ supplies the direct or external contribution at $t$, while the integral term represents the accumulated influence of previous states. The parameter $\lambda$ determines how strongly this history affects the present value. When $\lambda=0$, equation (19.3.1) reduces simply to $f(t)=g(t)$. As $|\lambda|$ increases, the memory contribution becomes more influential.

This interpretation explains why Volterra equations are central in time-evolution problems. An initial-value ordinary differential equation of the form $f'(t)=h(t,f(t)),$ can be rewritten by integrating both sides from the initial point $a$ to the current point $t$. If $f(a)=f_a$, then:

$$f(t)=f_a+\int_a^t h(s,f(s))\,ds$$

This is an integral equation whose upper limit is $t$. Therefore, the solution at time $t$ is obtained from the initial value plus the accumulated effect of the differential equation over the past interval $[a,t]$. In this sense, Volterra equations provide an integral counterpart of initial-value differential equations. They express evolution through accumulation rather than through instantaneous rate of change.

Volterra equations are also important in systems with memory. In many physical and engineering models, the present state is not determined solely by the present input, but also by the past history of the system. This occurs, for example, in viscoelasticity, where stress and strain may depend on past deformation; in control theory, where feedback may involve delayed or accumulated effects; and in population dynamics, where growth or interaction rates may depend on previous population levels. In such settings, the kernel $K(t,s)$ describes how strongly the state at the past point $s$ influences the state at the current point $t$.

A common special case occurs when the kernel depends only on the difference between the current and past variables $K(t,s)=K(t-s)$. Such kernels are called convolution kernels. They describe systems in which the influence of the past depends on the elapsed time $t-s$, rather than on $t$ and $s$ separately. The value $K(t-s)$ can then be interpreted as a memory weight: recent events and distant past events may influence the present differently depending on the shape of the kernel. For example, a rapidly decaying kernel represents short memory, while a slowly decaying kernel represents long memory.

Fractional-order differential equations provide another important source of Volterra equations. Many fractional derivatives can be represented through integrals over the past history of the solution. These formulations often lead to Volterra equations with kernels that become weakly singular as $s$ approaches $t$. In such cases, the kernel may remain integrable but exhibit reduced smoothness near the upper limit of integration. This weak singularity is numerically significant because standard quadrature rules may lose accuracy unless the singular behavior is handled carefully. Such fractional and memory-dependent formulations are among the reasons Volterra equations have become important in modern computational models of hereditary processes and anomalous dynamics (Al Khaykanee, 2024; Hamood, Sharif and Ghadle, 2025).

The main structural distinction between Volterra and Fredholm equations can therefore be summarized as follows. Fredholm equations usually produce globally coupled problems over a fixed interval, while Volterra equations produce causally ordered problems over a growing interval. This distinction affects both analysis and computation. Analytically, the Volterra structure often supports existence and uniqueness arguments based on progressive continuation from the initial point. Numerically, it leads naturally to step-by-step discretizations in which the unknown values are computed in increasing order of the independent variable. This lower-triangular dependence is the key computational feature developed in the following subsections.

## 19.3.2. Discretization, Lower-Triangular Structure, and Sequential Solution

The causal structure of a Volterra integral equation becomes especially important after discretization. Since the integral in equation (19.3.1) extends only from the initial point $a$ to the current point $t$, the numerical approximation at a given mesh point involves only the present and previously visited mesh points. This is the main reason Volterra equations of the second kind are often more convenient to solve than Fredholm equations of the second kind. Their discretized form naturally produces a lower-triangular dependence pattern, which allows the unknown values to be computed in order from left to right along the interval. In this respect, the numerical solution of a Volterra equation resembles a time-marching method for an initial-value ordinary differential equation.

Let the interval $[a,b]$ be discretized by the ordered mesh:

$$t_1=a<t_2<\cdots<t_N=b \tag{19.3.2}$$

Here, the nodes are arranged so that $t_i$ represents the $i$-th point at which the unknown function is to be approximated. The ordering is essential: because the integral in equation (19.3.1) terminates at $t_i$, the approximation at $t_i$ uses only the portion of the mesh lying in the interval $[a,t_i]$. Thus, the numerical method inherits the causal ordering of the continuous problem.

At each mesh point $t_i$, equation (19.3.1) becomes:

$$f(t_i)=\lambda\int_a^{t_i}K(t_i,s)f(s)\,ds+g(t_i)$$

The integral over $[a,t_i]$ can be approximated by a quadrature rule based on the available nodes $t_1,\ldots,t_i$. Using a rule such as the trapezoidal rule, Simpson’s rule, or another suitable quadrature formula gives the discrete approximation

$$f(t_i)=\lambda\sum_{j=1}^{i}K(t_i,t_j)w_{ij}f(t_j)+g(t_i),\qquad i=1,\dots,N \tag{19.3.3}$$

where $w_{ij}$ denotes the quadrature weight assigned to the value at $t_j$ when approximating the integral over $[a,t_i]$. The weights may depend on both $i$ and $j$ because the integration interval changes with $i$. For example, the quadrature weights used to approximate the integral up to $t_3$ are not necessarily the same as those used to approximate the integral up to $t_N$, since the interval length and number of contributing nodes are different.

The essential structural feature of equation (19.3.3) is that the summation stops at $j=i$. Therefore, the equation at $t_i$ does not involve any unknown values $f(t_{i+1}),\ldots,f(t_N)$ at future mesh points. In terms of the vector of unknown nodal values, $f(t_1),f(t_2),\dots,f(t_N),$ the resulting discrete system is lower triangular. This lower-triangular form is the discrete counterpart of the causal dependence in the continuous Volterra equation. The value at the first node is determined first, then the value at the second node, and so on until the final node is reached.

At the first node $t_1=a$, the integral interval is degenerate, since $[a,t_1]=[a,a]$. In the general quadrature form, the first discrete equation is written as:

$$f(t_1)=\lambda K(t_1,t_1)w_{11}f(t_1)+g(t_1) \tag{19.3.4}$$

This equation contains only the single unknown $f(t_1)$. Rearranging gives:

$$\left(1-\lambda K(t_1,t_1)w_{11}\right)f(t_1)=g(t_1)$$

Therefore, provided that the diagonal factor is nonzero,

$$1-\lambda K(t_1,t_1)w_{11}\neq 0,$$

the value $f(t_1)$ can be obtained directly. In many standard quadrature settings, the integral over a zero-length interval contributes no area, so $w_{11}$ may effectively vanish and the first value reduces to $f(t_1)=g(t_1)$. However, writing the equation in the more general form (19.3.4) is useful because it also covers discretizations in which endpoint weights or modified quadrature treatments appear.

Once $f(t_1)$ has been computed, the second equation involves only $f(t_1)$ and $f(t_2)$. Since $f(t_1)$ is already known, the second equation can be solved for $f(t_2)$. The third equation then involves $f(t_1)$, $f(t_2)$, and $f(t_3)$, where the first two values are already available. Continuing in this way, the entire solution can be constructed sequentially. This is the numerical meaning of the Volterra marching structure.

More generally, at step $i$, the term with $j=i$ in equation (19.3.3) contains the current unknown $f(t_i)$, while the terms with $j<i$ contain previously computed values. Separating the current term from the history sum gives:

$$f(t_i) = \lambda K(t_i,t_i)w_{ii}f(t_i) + \lambda\sum_{j=1}^{i-1}K(t_i,t_j)w_{ij}f(t_j)+g(t_i)$$

Rearranging this expression yields:

$$\left(1-\lambda K(t_i,t_i)w_{ii}\right)f(t_i) = g(t_i)+\lambda\sum_{j=1}^{i-1}K(t_i,t_j)w_{ij}f(t_j)\tag{19.3.5}$$

This equation makes the sequential character of the method explicit. The right-hand side is known at the moment step $i$ is reached, because it contains only the prescribed value $g(t_i)$ and the already computed history values $f(t_1),\ldots,f(t_{i-1})$. The only remaining unknown is $f(t_i)$, multiplied by the scalar diagonal factor $1-\lambda K(t_i,t_i)w_{ii}$.

Thus, if:

$$1-\lambda K(t_i,t_i)w_{ii}\neq 0 \tag{19.3.6}$$

then the scalar second-kind Volterra equation can be advanced one node at a time by solving a single scalar linear equation at each step. Explicitly, equation (19.3.5) gives:

$$f(t_i) = \frac{g(t_i)+\lambda\sum_{j=1}^{i-1}K(t_i,t_j)w_{ij}f(t_j)}{1-\lambda K(t_i,t_i)w_{ii}}$$

This formula highlights the role of the diagonal factor. If the denominator is close to zero, the numerical step may become ill-conditioned because small errors in the accumulated history sum or in $g(t_i)$ can be amplified. If the denominator is exactly zero, the simple scalar update is not valid and the discretized equation must be treated separately. In ordinary well-posed second-kind Volterra problems with appropriate quadrature, this diagonal nonvanishing condition is typically satisfied.

For systems of Volterra equations, the same principle applies, but the scalar update is replaced by a small linear algebra problem at each mesh point. If $f(t_i)$ is vector-valued, then $K(t_i,t_j)$ may act as a matrix or linear operator. The current-step term then produces a matrix factor of the form:

$$I-\lambda K(t_i,t_i)w_{ii},$$

and each step requires solving a linear system involving this local matrix. The important point is that the system remains sequential in the mesh index $i$. The method does not require solving one large fully coupled system over all mesh points, because future values do not enter the present equation.

This structure makes Volterra equations of the second kind substantially simpler to solve than Fredholm equations of the second kind. A Fredholm discretization typically produces a dense linear system because each value $f(t_i)$ may depend on values $f(t_j)$ over the full interval $[a,b]$. By contrast, a Volterra discretization produces a lower-triangular or marching problem. The unknowns can be advanced in order, and the computation naturally follows the direction of increasing $t$. Numerically, this resembles time-stepping methods for ordinary differential equations, where the solution at a later time is computed from information already obtained at earlier times.

The cost of a direct quadrature implementation is determined by the history sum in equation (19.3.3). At step $i$, the sum contains $i$ terms, or $i-1$ previously known history terms after the current diagonal contribution has been separated. Therefore, the amount of work grows as the computation advances through the mesh. Summing over all steps gives $O(N^2)$ operations for the full calculation. Equivalently, the average cost per step is $O(N)$, since each new value may require summing over all previous values. This quadratic cost is often acceptable for moderate $N$, but it can become significant in long-time simulations or highly resolved memory-dependent models.

Memory usage is usually modest because the solution values are stored as a one-dimensional history. However, Volterra equations with long memory may require retaining many past values in order to evaluate later history sums. In such cases, storage and memory access can become important practical considerations. Long-time simulations may therefore require careful organization of the solution history, especially when the kernel has slow decay and older values continue to contribute significantly to the present state.

For smooth kernels and smooth solutions, the accuracy of quadrature-based Volterra discretizations follows the usual behavior of the chosen quadrature formula. On a mesh with characteristic spacing $h$, the trapezoidal rule typically gives an error of order $O(h^2)$, while Simpson’s rule gives an error of order $O(h^4)$. These rates assume that the integrand $K(t,s)f(s)$ has the smoothness required by the corresponding quadrature rule. In the Volterra setting, this smoothness must be considered on each interval $[a,t_i]$, and especially near the diagonal $s=t$, where some kernels may have reduced regularity.

Higher-order collocation and convolution quadrature methods can improve accuracy, particularly when the kernel and solution are sufficiently smooth. Such methods are especially useful when the problem requires high precision or when the integral equation is solved repeatedly as part of a larger computational model. In the context of Volterra equations, the main advantage of higher-order discretization is that it can reduce the number of mesh points needed to reach a prescribed accuracy, while preserving the sequential structure of the problem. Improved quadrature and collocation strategies are therefore important tools in the numerical treatment of Volterra equations (Al Khaykanee, 2024).

Special care is required when the kernel is singular or weakly singular near the diagonal $s=t$. A typical weakly singular behavior has the form:

$$K(t,s)\sim (t-s)^{-\alpha},\qquad 0<\alpha<1 \tag{19.3.7}$$

Although the singularity is integrable when $0<\alpha<1$, it changes the local behavior of the integrand near the upper limit. As $s$ approaches $t$, the factor $(t-s)^{-\alpha}$ grows without bound, and standard quadrature formulas may no longer achieve their usual convergence rates. The difficulty is not that the integral is necessarily undefined, but that the integrand is less smooth than the assumptions underlying classical quadrature rules.

Such kernels occur frequently in fractional-order models, where the present state depends on a history integral with a power-law memory kernel. This type of memory is fundamentally different from short-memory behavior because distant past values may continue to influence the present through slowly decaying weights. From a numerical viewpoint, the weak singularity near $s=t$ and the long-memory character of the integral both affect accuracy, stability, and computational cost.

For weakly singular Volterra equations, specialized numerical techniques may be needed. Product-integration rules incorporate the singular factor into the quadrature construction rather than treating it as an ordinary smooth integrand. Graded meshes place more nodes near regions where the solution or kernel has reduced regularity, improving resolution where the numerical error is most likely to concentrate. Coordinate transformations may be used to weaken or remove the singular behavior in the transformed variable. Spectral collocation techniques can also be effective when the solution structure permits high-order approximation.

Recent work on fractional Volterra-Fredholm integro-differential problems illustrates the importance of high-order spectral collocation methods for accurately resolving weakly singular and memory-dependent structures (Hamood, Sharif and Ghadle, 2025). In such problems, the numerical method must account for both the accumulated history of the Volterra component and the possible singular behavior of the kernel. The discretization must therefore preserve the causal structure while also maintaining sufficient accuracy near the diagonal (s=t). This combination of causal marching, history dependence, and singular-kernel treatment is one of the central numerical challenges in the solution of Volterra-type models.

### Rust Implementation

Following the discussion in Section 19.3.2 on the discretization, lower-triangular structure, and sequential solution of Volterra integral equations, Program 19.3.1 provides a practical implementation of direct quadrature marching for a second-kind Volterra equation. The program follows the causal discretization developed in equations (19.3.3) through (19.3.6), where the unknown value at each mesh point is computed from the accumulated history of previously determined values and a scalar diagonal correction at the current point. Unlike Fredholm equations, which typically lead to dense globally coupled systems, the Volterra structure allows the numerical solution to advance from left to right along the ordered mesh. This implementation uses the composite trapezoidal rule, verifies the result with a manufactured exact solution, and reports diagnostic quantities that expose both the triangular computational cost and the stability of the diagonal update.

At the core of the implementation is the causal update formula derived from equations (19.3.3) and (19.3.5). The function `kernel(t, s)` defines the memory kernel $K(t,s)$, which determines how strongly a previous value at $s$ contributes to the current value at $t$. In this example, the kernel is smooth and exponentially decaying, so earlier values have a gradually diminishing influence. The function `exact_solution(t)` specifies the manufactured solution used for verification, while `rhs(t, lambda)` constructs the corresponding right-hand side so that the exact solution satisfies the continuous Volterra equation.

The function `uniform_mesh` constructs the ordered mesh introduced in equation (19.3.2). This ordering is essential for the Volterra problem because the solution is not computed simultaneously at all points. Instead, the mesh points are visited sequentially from the initial point to the final point, and each newly computed value becomes part of the available history for all later steps.

The function `trapezoidal_history_weight` defines the quadrature weights (w\_{ij}) used in equation (19.3.3). Since the integration interval changes from one mesh point to the next, the weights are interpreted locally on each interval (\[a,t_i\]). For the first node, the interval has zero length, so the weight vanishes. For later nodes, the endpoint weights are one half of the mesh spacing, while the interior weights are equal to the mesh spacing, as in the composite trapezoidal rule.

The main numerical work is carried out by `solve_volterra_marching`. At each mesh index $i$, the function first accumulates the known history contribution using only values that have already been computed. This corresponds to the summation over $j<i$ in equation (19.3.5). The current endpoint contribution is then handled separately through the diagonal factor appearing in equation (19.3.6). If this factor is too close to zero, the step would be numerically unsafe, so the code checks for this condition before computing the new value.

The function `max_abs_error` compares the computed solution with the manufactured exact solution at the mesh points. This gives a direct measure of the quadrature and marching error. The function `discrete_residual_norm` evaluates how well the computed solution satisfies the discrete Volterra equation produced by the same trapezoidal rule. This distinction is important: the error measures agreement with the continuous manufactured solution, while the residual checks whether the algebraic marching equations have been solved accurately.

The function `history_operation_count` reports the number of accumulated past-history contributions used by the direct marching method. This count illustrates the triangular computational cost discussed in Section 19.3.2. Since the $i$-th step uses $i$ previous values, the total number of history contributions grows quadratically with the number of mesh points. The function `minimum_diagonal_factor` reports the smallest absolute diagonal factor over the mesh, providing a simple diagnostic for the nonvanishing condition in equation (19.3.6).

The `main` function brings these components together. It defines the interval, number of mesh points, and parameter $\lambda$, constructs the ordered mesh, solves the Volterra equation by direct quadrature marching, and prints a table comparing the numerical and exact solutions. It also reports the number of accumulated history terms, the minimum diagonal factor, the maximum absolute error, and the maximum discrete residual. These outputs connect the implementation directly to the causal, lower-triangular structure developed in Section 19.3.2.

```rust
// Program 19.3.1: Direct Quadrature Marching for a Second-Kind Volterra Integral Equation
//
// Problem statement:
// Solve the second-kind Volterra integral equation
//
//     f(t) = g(t) + lambda * integral_a^t K(t, s) f(s) ds,
//
// using a direct quadrature marching method. The integral is approximated
// by the composite trapezoidal rule on the available history interval [a, t_i].
// At each mesh point t_i, only the already computed values
//
//     f(t_1), f(t_2), ..., f(t_{i-1})
//
// and the current unknown f(t_i) appear. The current endpoint contribution
// is handled through the diagonal factor
//
//     1 - lambda * K(t_i, t_i) * w_ii.
//
// The example uses a manufactured exact solution so that the numerical
// approximation can be checked directly.

/// Kernel K(t, s) for the Volterra integral equation.
///
/// This smooth kernel represents exponentially decaying memory:
/// values closer to the present point t have stronger influence.
fn kernel(t: f64, s: f64) -> f64 {
    (-(t - s)).exp()
}

/// Manufactured exact solution f(t).
fn exact_solution(t: f64) -> f64 {
    t.exp()
}

/// Right-hand side g(t), constructed so that f(t) = exp(t)
/// satisfies the Volterra equation.
///
/// With
///
///     K(t, s) = exp(-(t - s))
///     f(s) = exp(s)
///
/// one has
///
///     integral_0^t exp(-(t - s)) exp(s) ds = sinh(t).
///
/// Therefore,
///
///     g(t) = exp(t) - lambda * sinh(t).
fn rhs(t: f64, lambda: f64) -> f64 {
    exact_solution(t) - lambda * t.sinh()
}

/// Build a uniform ordered mesh on [a, b].
///
/// The ordering of the mesh points is essential for a Volterra equation,
/// since the solution is advanced from the initial point toward the final point.
fn uniform_mesh(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n < 2 {
        panic!("At least two mesh points are required.");
    }

    let h = (b - a) / (n as f64 - 1.0);

    (0..n)
        .map(|i| a + i as f64 * h)
        .collect()
}

/// Return the trapezoidal weight w_ij used when approximating the integral
/// over [a, t_i].
///
/// For i = 0, the interval has zero length, so all weights vanish.
/// For i > 0, the endpoint weights are h / 2 and the interior weights are h.
fn trapezoidal_history_weight(i: usize, j: usize, h: f64) -> f64 {
    if i == 0 {
        0.0
    } else if j == 0 || j == i {
        0.5 * h
    } else {
        h
    }
}

/// Solve the second-kind Volterra equation by direct quadrature marching.
///
/// At step i, the discrete equation has the form
///
///     f_i = g_i + lambda * sum_{j=0}^{i} K(t_i, t_j) w_ij f_j.
///
/// The terms with j < i are already known. The term with j = i contains
/// the current unknown and is moved to the left-hand side, giving a scalar
/// update with a diagonal denominator.
fn solve_volterra_marching(
    mesh: &[f64],
    lambda: f64,
) -> Result<Vec<f64>, String> {
    let n = mesh.len();

    if n < 2 {
        return Err("At least two mesh points are required.".to_string());
    }

    let h = mesh[1] - mesh[0];
    let mut solution = vec![0.0; n];

    for i in 0..n {
        let t_i = mesh[i];

        let mut history_sum = 0.0;

        for j in 0..i {
            let t_j = mesh[j];
            let weight = trapezoidal_history_weight(i, j, h);

            history_sum += weight * kernel(t_i, t_j) * solution[j];
        }

        let diagonal_weight = trapezoidal_history_weight(i, i, h);
        let diagonal_factor =
            1.0 - lambda * kernel(t_i, t_i) * diagonal_weight;

        if diagonal_factor.abs() < 1.0e-14 {
            return Err(format!(
                "The diagonal factor is too small at mesh index {}.",
                i
            ));
        }

        let numerator = rhs(t_i, lambda) + lambda * history_sum;
        solution[i] = numerator / diagonal_factor;
    }

    Ok(solution)
}

/// Compute the maximum absolute error relative to the manufactured solution.
fn max_abs_error(mesh: &[f64], numerical: &[f64]) -> f64 {
    mesh.iter()
        .zip(numerical.iter())
        .map(|(&t, &value)| (value - exact_solution(t)).abs())
        .fold(0.0, f64::max)
}

/// Compute the maximum discrete residual of the Volterra equation.
///
/// This diagnostic checks the algebraic equation produced by the same
/// trapezoidal discretization used in the marching method.
fn discrete_residual_norm(mesh: &[f64], numerical: &[f64], lambda: f64) -> f64 {
    let n = mesh.len();
    let h = mesh[1] - mesh[0];

    let mut max_residual: f64 = 0.0;

    for i in 0..n {
        let t_i = mesh[i];
        let mut integral_approx = 0.0;

        for j in 0..=i {
            let t_j = mesh[j];
            let weight = trapezoidal_history_weight(i, j, h);

            integral_approx += weight * kernel(t_i, t_j) * numerical[j];
        }

        let residual =
            numerical[i] - rhs(t_i, lambda) - lambda * integral_approx;

        max_residual = max_residual.max(residual.abs());
    }

    max_residual
}

/// Estimate the total number of kernel-history contributions used by
/// direct Volterra marching.
///
/// This count grows like N(N - 1) / 2 for the known history terms,
/// illustrating the O(N^2) work of direct history summation.
fn history_operation_count(n: usize) -> usize {
    (0..n).map(|i| i).sum()
}

/// Compute the smallest absolute diagonal factor over the mesh.
///
/// This reports the numerical safety of the scalar update associated with
/// the nonvanishing condition in equation (19.3.6).
fn minimum_diagonal_factor(mesh: &[f64], lambda: f64) -> f64 {
    let h = mesh[1] - mesh[0];

    mesh.iter()
        .enumerate()
        .map(|(i, &t_i)| {
            let diagonal_weight = trapezoidal_history_weight(i, i, h);
            let factor = 1.0 - lambda * kernel(t_i, t_i) * diagonal_weight;
            factor.abs()
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let a = 0.0;
    let b = 2.0;
    let n = 41;
    let lambda = 0.4;

    let mesh = uniform_mesh(a, b, n);

    let numerical_solution = match solve_volterra_marching(&mesh, lambda) {
        Ok(solution) => solution,
        Err(message) => {
            eprintln!("Volterra marching failed: {}", message);
            return;
        }
    };

    let max_error = max_abs_error(&mesh, &numerical_solution);
    let residual = discrete_residual_norm(&mesh, &numerical_solution, lambda);
    let min_diagonal = minimum_diagonal_factor(&mesh, lambda);
    let history_terms = history_operation_count(n);

    println!("Direct Quadrature Marching for a Volterra Equation");
    println!("==================================================");
    println!();
    println!("Model Problem");
    println!("-------------");
    println!("Equation type              = second-kind Volterra");
    println!("Interval                   = [{:.1}, {:.1}]", a, b);
    println!("Kernel K(t, s)             = exp(-(t - s))");
    println!("Exact solution f(t)        = exp(t)");
    println!("Lambda                     = {:.6}", lambda);
    println!();
    println!("Discretization");
    println!("--------------");
    println!("Mesh points N              = {}", n);
    println!("Quadrature rule            = composite trapezoidal rule");
    println!("Solution strategy          = left-to-right marching");
    println!("Discrete structure         = lower triangular");
    println!("History terms accumulated  = {}", history_terms);
    println!("Minimum diagonal factor    = {:.6e}", min_diagonal);
    println!();
    println!("Numerical Results");
    println!("-----------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "i", "t_i", "f_num", "f_exact", "abs_error"
    );

    for i in 0..n {
        let t = mesh[i];
        let f_num = numerical_solution[i];
        let f_exact = exact_solution(t);
        let abs_error = (f_num - f_exact).abs();

        println!(
            "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.6e}",
            i, t, f_num, f_exact, abs_error
        );
    }

    println!();
    println!("Diagnostics");
    println!("-----------");
    println!("max_i |f_num(t_i) - f_exact(t_i)| = {:.6e}", max_error);
    println!("maximum discrete residual          = {:.6e}", residual);
}
```

Program 19.3.1 demonstrates how the lower-triangular structure of a second-kind Volterra equation leads naturally to a direct marching method. At each mesh point, the method uses only the known history and the current diagonal contribution, so no dense global system needs to be assembled. This reflects the central computational advantage of Volterra equations over Fredholm equations: the causal upper limit of integration produces an ordered sequence of scalar updates.

The numerical output confirms the correctness of the implementation. The discrete residual is close to machine precision, showing that the computed values satisfy the trapezoidal discretization accurately. The maximum absolute error reflects the difference between the continuous manufactured solution and its quadrature-based numerical approximation. Its gradual growth along the interval is expected because each new value depends on the accumulated history of earlier numerical values.

The reported history count illustrates the computational cost of direct Volterra marching. Although the method stores only the one-dimensional solution history, the work required to evaluate the accumulated memory term grows like a triangular sum over the mesh points. This gives the $O(N^2)$ operation count discussed in the section. For moderate values of $N$, this cost is acceptable and the method is simple and reliable. For long-memory or long-time simulations, however, this motivates faster convolution, compression, or adaptive-history techniques.

The minimum diagonal factor provides an additional stability diagnostic. In this example it remains safely away from zero, so the scalar update associated with equation (19.3.5) is well conditioned. If the diagonal factor became very small, the marching step could amplify errors in the history sum or in the forcing term. Thus, the program not only implements the sequential Volterra solve but also exposes the numerical quantities that determine its reliability.

## 19.3.3. Classical and Modern Numerical Methods for Volterra Integral Equations

The numerical methods used for Volterra integral equations reflect the causal and lower-triangular structure developed in Section 19.3.2. Since the value at a given point depends only on earlier values and possibly the current value, many Volterra solvers proceed progressively along the interval rather than solving one large globally coupled problem. This property has shaped both classical and modern methods. Classical schemes usually rely on local quadrature, polynomial interpolation, or Taylor expansion, while more recent approaches use higher-order collocation, spectral approximations, extrapolation, and specialized methods for fractional or memory-dependent models. The common objective is to approximate the history integral accurately while preserving the sequential structure that makes Volterra equations computationally attractive.

Classical methods for Volterra equations include direct quadrature, collocation with piecewise polynomials, and Taylor-based approximations. These methods are closely connected to the discretization formula in equation (19.3.3), where the integral at $t_i$ is replaced by a weighted sum over the previously reached nodes $t_1,\ldots,t_i$. Because the upper limit of integration is $t_i$, the discrete equation at the $i$-th point does not involve unknown future values. Consequently, the numerical method advances forward from the initial point, using values already computed at earlier mesh points to determine the next value. This is the essential reason why Volterra solvers do not usually require the kind of global dense linear solve that arises in Fredholm discretizations.

In a direct quadrature method, the main task is to approximate:

$$\int_a^{t_i}K(t_i,s)f(s)\,ds$$

by a quadrature sum over the available mesh points. This gives a formula of the form already introduced in equation (19.3.3). The quadrature weights $w_{ij}$ encode the chosen numerical integration rule on the interval $[a,t_i]$. For example, a trapezoidal-type discretization uses endpoint and interior weights to approximate the accumulated area under the integrand $K(t_i,s)f(s)$. Simpson-type formulas use higher-order polynomial information and can give higher accuracy when the integrand is sufficiently smooth. In all such cases, the method retains the same causal form: only values up to $t_i$ appear in the $i$-th equation.

The practical solution of the resulting discrete equations follows the rearranged form in equation (19.3.5). At step $i$, the previous values $f(t_1),\ldots,f(t_{i-1})$ are already known, so the history sum can be evaluated directly. The current value $f(t_i)$ is then obtained by dividing by the diagonal factor $1-\lambda K(t_i,t_i)w_{ii}$, provided the nonzero condition in equation (19.3.6) is satisfied. Thus, in the scalar second-kind case, direct quadrature reduces each step to a scalar update. For systems, the same construction produces a small local linear system at each step rather than one fully coupled global system.

Collocation methods provide a more systematic approximation framework. Instead of approximating only the integral values directly, one represents the unknown function $f(t)$ by a polynomial or piecewise polynomial approximation and enforces the Volterra equation at selected collocation points. In a piecewise polynomial collocation method, the interval $[a,b]$ is divided into subintervals, and $f(t)$ is approximated locally on each subinterval by a polynomial. The coefficients are chosen so that the integral equation is satisfied at prescribed points. Because the Volterra integral extends only over the past interval $[a,t]$, the resulting collocation equations can still be organized sequentially. Earlier polynomial pieces contribute to the known history, while the current piece is determined from the local equations on the current subinterval.

This piecewise structure is particularly useful because it balances accuracy and locality. Low-degree polynomials give relatively simple formulas and are easy to implement, while higher-degree local approximations can improve accuracy when the solution is smooth. The method also provides a natural way to refine the mesh where the solution changes rapidly or where the kernel has more delicate behavior. In Volterra equations, such local adaptivity is often valuable because early errors can propagate through the history integral and influence later computed values.

Taylor-based approximations form another classical approach. These methods expand the unknown function, the kernel, or the integrand locally in powers of the independent variable and then integrate the resulting polynomial expressions. Their usefulness depends strongly on the smoothness of the solution and kernel. When the problem is sufficiently smooth, Taylor-based formulas can produce accurate local approximations and can be interpreted as integral-equation analogues of Taylor methods for ordinary differential equations. However, when the kernel is weakly singular or the solution has reduced regularity, Taylor expansions may become less effective unless modified to reflect the actual local behavior of the solution.

A common advantage of these classical methods is that they preserve the triangular structure of the Volterra problem. The solution process does not require constructing and inverting a dense $N\times N$ matrix. Instead, the computation advances from one node or subinterval to the next. This gives Volterra solvers a natural stability advantage in causal problems, since each step uses already computed history rather than requiring simultaneous global coupling across the entire interval. The stability behavior still depends on the kernel, the quadrature rule, the mesh spacing, and the diagonal factor in equation (19.3.6), but the causal ordering provides a computational structure well matched to time-evolution and memory-dependent models (Al Khaykanee, 2024).

Modern approaches extend these classical methods in several directions. One important direction is spectral collocation. Spectral collocation methods approximate the unknown function using global polynomial bases rather than low-degree local polynomials. A typical example uses shifted Chebyshev polynomials on (\[a,b\]). These polynomials provide an efficient basis for approximating smooth functions on finite intervals and allow the Volterra equation to be converted into an algebraic system for the expansion coefficients.

In a spectral collocation formulation, one writes an approximation of the form:

$$f(t)\approx \sum_{k=0}^{M} c_k \phi_k(t),$$

where $\phi_k(t)$ denotes the chosen global basis function, such as a shifted Chebyshev polynomial, and $c_k$ are coefficients to be determined. Substituting this approximation into the Volterra equation gives:

$$\sum_{k=0}^{M} c_k \phi_k(t) = \lambda\int_a^t K(t,s)\sum_{k=0}^{M} c_k \phi_k(s)\,ds+g(t)$$

The equation is then enforced at selected collocation points. This produces algebraic equations for the coefficients $c_k$. The advantage of the spectral approach is that, for smooth solutions and kernels, the approximation can be highly accurate with relatively few basis functions. The method replaces many local stepwise approximations by a global representation of the unknown function.

However, the use of a global basis also changes the algebraic structure of the problem. While the original continuous equation remains causal, enforcing a global polynomial representation may produce a more globally coupled algebraic system than a purely stepwise quadrature method. This does not remove the importance of the Volterra structure, but it means that the implementation and solution strategy may differ from direct marching. Spectral collocation is especially attractive when high accuracy is required and the solution is sufficiently smooth over the whole interval.

Multi-step collocation methods and extrapolation methods provide another route to improved accuracy. Multi-step collocation uses information from several previous points or subintervals to construct a higher-order approximation at the current step. This is consistent with the history-dependent nature of Volterra equations, since previous values are already available during the computation. Extrapolation methods combine approximations obtained with different step sizes or different local resolutions to improve the final estimate. These approaches are useful when the basic quadrature or collocation formula converges predictably and higher accuracy is desired without completely changing the underlying discretization framework.

In fractional settings, high-order spectral methods are especially important. Fractional Volterra-type problems often involve memory terms and kernels with reduced regularity near the diagonal $s=t$. As discussed in Section 19.3.2, such kernels may have weakly singular behavior of the form shown in equation (19.3.7). This behavior can reduce the accuracy of standard quadrature rules if it is not treated carefully. High-order spectral and collocation methods can help resolve these structures more accurately, especially when combined with basis functions, meshes, or formulations suited to the memory-dependent character of the problem.

Hamood, Sharif and Ghadle (2025) use shifted Chebyshev spectral collocation to solve linear fractional Volterra-Fredholm integro-differential problems. In this setting, the problem includes both fractional and integral components, so the numerical method must represent differential, Volterra, and Fredholm-type contributions within one algebraic framework. Their approach transforms the integro-differential equation into a nonlinear algebraic system, which is then solved by Newton’s method. The use of shifted Chebyshev polynomials provides a high-order approximation framework, while Newton’s method supplies the iterative algebraic solver needed after discretization.

For smooth problems, this type of spectral collocation method can achieve very high accuracy. The reason is that global polynomial approximations can converge rapidly when the target function is smooth and well represented by the chosen basis. In fractional or memory-dependent problems, the method must also account for the accumulated past influence encoded in the Volterra term. The success of the method therefore depends not only on the approximation of $f(t)$, but also on the accurate evaluation of the history integrals and fractional terms. This illustrates a general theme in modern Volterra solvers: high-order approximation is most effective when the discretization respects the memory structure of the equation.

Volterra equations also appear in more general forms. In a Volterra-Hammerstein equation, the integral equation contains a nonlinear function of the unknown inside the integral. A representative form is:

$$f(t)=g(t)+\lambda\int_a^t K(t,s)\Phi(f(s))\,ds,$$

where $\Phi$ is a nonlinear function. The causal structure is still present because the integral extends only over $[a,t]$, but the numerical update may become nonlinear. At each step, the unknown may appear through $\Phi(f(t_i))$, depending on the quadrature rule and endpoint contribution. As a result, the current value may need to be obtained by solving a nonlinear scalar equation or, for systems, a nonlinear local system. Nevertheless, the equation remains ordered in time or in the independent variable, since future unknowns do not enter the present equation.

Stochastic Volterra equations arise when random kernels, random forcing, or stochastic memory effects are present. In such models, the history dependence is combined with uncertainty. The present state may depend on a random accumulated past influence, or the forcing term may include stochastic variation. Numerically, this makes the problem more demanding because one may need to approximate both the memory integral and the effect of randomness. The triangular Volterra structure remains important, but the solution is no longer simply a deterministic sequence of values; it may instead involve sample paths, statistical quantities, or repeated simulations.

Machine learning techniques have also begun to appear in this area. One possible approach is to train neural networks to approximate $f(t)$ from samples of $g(t)$ and known kernel behavior. The aim is to learn the mapping from the input data and kernel information to the solution of the integral equation. However, because each value in a Volterra equation depends on earlier computed values, straightforward physics-informed neural network formulations are less common for Volterra equations than for some Fredholm problems. A Fredholm equation has a more globally coupled structure over a fixed interval, whereas a Volterra equation has an ordered history dependence that must be represented carefully.

For this reason, recurrent or iterative learning architectures may be more natural for Volterra equations. Such architectures are better aligned with causal dependence because they can process information sequentially, using earlier states to inform later ones. In principle, this matches the marching structure of equations (19.3.3) and (19.3.5). The learning model can be designed to update the approximation progressively, in the same direction as the Volterra integral accumulates history. This remains an active research direction, especially for problems where the kernel is complicated, the memory effect is long-range, or the equation must be solved many times for different inputs.

Overall, the numerical treatment of Volterra equations is built around the same central principle: the present depends on the past. Classical quadrature methods exploit this directly through lower-triangular sums. Piecewise polynomial and Taylor-based methods improve local accuracy while retaining sequential computation. Spectral collocation methods provide high-order global approximations, particularly useful for smooth and fractional models. Nonlinear, stochastic, and machine-learning-based extensions broaden the range of Volterra-type problems while preserving the need to respect causal history dependence. Thus, both classical and modern methods are best understood as different ways of approximating the same accumulated-memory structure introduced in equation (19.3.1).

### Rust Implementation

Following the discussion in Section 19.3.3 on classical and modern numerical methods for Volterra integral equations, Program 19.3.2 provides a practical implementation of product-integration marching for a weakly singular Volterra equation. The program addresses the numerical difficulty introduced by kernels of the form shown in equation (19.3.7), where the integrand remains integrable but loses smoothness near the diagonal $s=t$. Rather than applying a standard quadrature formula directly to the singular integrand, the method integrates the singular factor analytically over each history subinterval and approximates only the smooth solution factor by nodal values. This preserves the causal left-to-right marching structure of equations (19.3.3) and (19.3.5), while adapting the quadrature rule to the fractional-memory behavior discussed in the section.

At the core of the implementation is the replacement of ordinary quadrature by product integration. The function `exact_solution(t)` defines the manufactured solution used to verify the numerical approximation, while `exact_history_integral(t, alpha)` evaluates the corresponding history integral analytically for the chosen test problem. This allows the function `rhs(t, lambda, alpha)` to construct a right-hand side for which the exact solution is known in closed form.

The function `uniform_mesh` constructs the ordered mesh on the interval of integration. As in the direct Volterra marching method, the ordering of the mesh is essential because each value is computed from earlier values. This preserves the causal structure described by equations (19.3.3) and (19.3.5), where the current equation depends only on the known history and not on future unknowns.

The function `product_weight` is the key numerical component of the program. It computes the exact integral of the weakly singular factor over a single subinterval. This avoids evaluating the singular kernel directly at $s=t$, where the expression becomes unbounded. In this way, the method incorporates the singular behavior into the quadrature weights rather than treating it as an ordinary smooth function. This is precisely the motivation for product-integration methods in weakly singular Volterra equations.

The function `solve_product_integration` performs the causal marching solve. At each mesh point $t_i$, the history interval is divided into subintervals, and each contribution is computed using the product-integration weight multiplied by the previously computed solution value at the left endpoint. The method therefore remains explicit and sequential: once the values at earlier mesh points have been computed, the next value is obtained directly from the right-hand side and the accumulated history contribution.

The function `max_abs_error` compares the computed solution with the manufactured exact solution on the mesh. This measures the continuous approximation error introduced by the product-integration discretization. The function `discrete_residual_norm` evaluates how closely the computed values satisfy the discrete product-integration equations. This distinction is important because a small residual confirms that the discrete equations were solved accurately, while the maximum absolute error measures how well those discrete equations approximate the continuous weakly singular Volterra problem.

The function `history_interval_count` reports the number of history subintervals evaluated during the direct marching process. This count grows quadratically with the number of mesh points and reinforces the computational cost discussion for direct Volterra history summation. The function `run_experiment` packages one complete solve on a chosen mesh, prints representative solution values, and returns the maximum error. Finally, the `main` function runs the experiment on two mesh sizes, $N=41$ and $N=81$, and reports the observed error ratio. This provides a simple convergence diagnostic showing how the method improves under mesh refinement.

```rust
// Program 19.3.2: Product-Integration Marching for a Weakly Singular Volterra Equation
//
// Problem statement:
// Solve the weakly singular second-kind Volterra integral equation
//
//     f(t) = g(t) + lambda * integral_0^t (t - s)^(-alpha) f(s) ds,
//
// where 0 < alpha < 1. The kernel is integrable but singular at s = t.
// Instead of applying an ordinary quadrature rule directly to the singular
// integrand, the program uses product integration. The singular factor
//
//     (t - s)^(-alpha)
//
// is integrated exactly over each history subinterval, while f(s) is
// approximated by its left endpoint value.
//
// The example uses the manufactured exact solution
//
//     f(t) = 1 + t,
//
// so that the right-hand side can be evaluated analytically.

/// Manufactured exact solution f(t).
fn exact_solution(t: f64) -> f64 {
    1.0 + t
}

/// Analytic value of
///
///     integral_0^t (t - s)^(-alpha) (1 + s) ds.
///
/// For f(s) = 1 + s, this integral is
///
///     t^(1-alpha) / (1-alpha)
///     + t^(2-alpha) / ((1-alpha)(2-alpha)).
fn exact_history_integral(t: f64, alpha: f64) -> f64 {
    if t == 0.0 {
        0.0
    } else {
        let first = t.powf(1.0 - alpha) / (1.0 - alpha);
        let second = t.powf(2.0 - alpha) / ((1.0 - alpha) * (2.0 - alpha));
        first + second
    }
}

/// Right-hand side g(t), constructed so that f(t) = 1 + t
/// satisfies the weakly singular Volterra equation.
fn rhs(t: f64, lambda: f64, alpha: f64) -> f64 {
    exact_solution(t) - lambda * exact_history_integral(t, alpha)
}

/// Build a uniform mesh on [a, b].
fn uniform_mesh(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n < 2 {
        panic!("At least two mesh points are required.");
    }

    let h = (b - a) / (n as f64 - 1.0);

    (0..n)
        .map(|i| a + i as f64 * h)
        .collect()
}

/// Product-integration weight for the interval [t_j, t_{j+1}]
/// when evaluating the integral at t_i.
///
/// The weight is
///
///     integral_{t_j}^{t_{j+1}} (t_i - s)^(-alpha) ds.
///
/// This is finite for 0 < alpha < 1, even when t_{j+1} = t_i.
fn product_weight(t_i: f64, left: f64, right: f64, alpha: f64) -> f64 {
    let exponent = 1.0 - alpha;

    let upper = (t_i - left).powf(exponent);
    let lower = (t_i - right).powf(exponent);

    (upper - lower) / exponent
}

/// Solve the weakly singular Volterra equation by product-integration marching.
///
/// For each t_i, the history integral over [0, t_i] is split into intervals
/// [t_j, t_{j+1}], j = 0, ..., i - 1. On each interval, f(s) is approximated
/// by the already known value f(t_j), while the singular kernel is integrated
/// exactly. This gives a causal explicit marching method.
fn solve_product_integration(
    mesh: &[f64],
    lambda: f64,
    alpha: f64,
) -> Result<Vec<f64>, String> {
    if !(0.0 < alpha && alpha < 1.0) {
        return Err("The weak singularity exponent must satisfy 0 < alpha < 1.".to_string());
    }

    let n = mesh.len();
    let mut solution = vec![0.0; n];

    for i in 0..n {
        let t_i = mesh[i];

        let mut history_sum = 0.0;

        for j in 0..i {
            let left = mesh[j];
            let right = mesh[j + 1];

            let weight = product_weight(t_i, left, right, alpha);
            history_sum += weight * solution[j];
        }

        solution[i] = rhs(t_i, lambda, alpha) + lambda * history_sum;
    }

    Ok(solution)
}

/// Compute the maximum absolute error relative to the manufactured solution.
fn max_abs_error(mesh: &[f64], numerical: &[f64]) -> f64 {
    mesh.iter()
        .zip(numerical.iter())
        .map(|(&t, &value)| (value - exact_solution(t)).abs())
        .fold(0.0, f64::max)
}

/// Compute a discrete residual using the same product-integration rule.
///
/// This checks whether the computed values satisfy the discrete marching
/// equations, rather than measuring the continuous quadrature error.
fn discrete_residual_norm(
    mesh: &[f64],
    numerical: &[f64],
    lambda: f64,
    alpha: f64,
) -> f64 {
    let n = mesh.len();
    let mut max_residual: f64 = 0.0;

    for i in 0..n {
        let t_i = mesh[i];

        let mut history_sum = 0.0;

        for j in 0..i {
            let left = mesh[j];
            let right = mesh[j + 1];

            let weight = product_weight(t_i, left, right, alpha);
            history_sum += weight * numerical[j];
        }

        let residual = numerical[i] - rhs(t_i, lambda, alpha) - lambda * history_sum;
        max_residual = max_residual.max(residual.abs());
    }

    max_residual
}

/// Count the number of history intervals used by the direct product-integration
/// method. This grows like N(N - 1) / 2.
fn history_interval_count(n: usize) -> usize {
    (0..n).sum()
}

/// Demonstrate convergence by solving the same problem on two meshes.
fn run_experiment(n: usize, lambda: f64, alpha: f64) -> Result<f64, String> {
    let a = 0.0;
    let b = 1.0;

    let mesh = uniform_mesh(a, b, n);
    let numerical_solution = solve_product_integration(&mesh, lambda, alpha)?;

    let max_error = max_abs_error(&mesh, &numerical_solution);
    let residual = discrete_residual_norm(&mesh, &numerical_solution, lambda, alpha);
    let history_intervals = history_interval_count(n);

    println!("Mesh with N = {}", n);
    println!("-------------");
    println!("History intervals evaluated = {}", history_intervals);
    println!("Maximum absolute error      = {:.6e}", max_error);
    println!("Maximum discrete residual   = {:.6e}", residual);
    println!();

    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "i", "t_i", "f_num", "f_exact", "abs_error"
    );

    for i in 0..n {
        if i % ((n - 1) / 8).max(1) == 0 || i == n - 1 {
            let t = mesh[i];
            let f_num = numerical_solution[i];
            let f_exact = exact_solution(t);
            let abs_error = (f_num - f_exact).abs();

            println!(
                "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.6e}",
                i, t, f_num, f_exact, abs_error
            );
        }
    }

    println!();

    Ok(max_error)
}

fn main() {
    let lambda = 0.35;
    let alpha = 0.5;

    println!("Product Integration for a Weakly Singular Volterra Equation");
    println!("===========================================================");
    println!();
    println!("Model Problem");
    println!("-------------");
    println!("Equation type              = second-kind Volterra");
    println!("Kernel                     = (t - s)^(-alpha)");
    println!("Weak singularity exponent  = {:.6}", alpha);
    println!("Exact solution f(t)        = 1 + t");
    println!("Lambda                     = {:.6}", lambda);
    println!("Interval                   = [0.0, 1.0]");
    println!();
    println!("Method");
    println!("------");
    println!("Quadrature idea            = product integration");
    println!("Singular factor            = integrated analytically");
    println!("Smooth factor f(s)         = approximated by left endpoint values");
    println!("Solution strategy          = causal left-to-right marching");
    println!();

    let coarse_error = match run_experiment(41, lambda, alpha) {
        Ok(error) => error,
        Err(message) => {
            eprintln!("Computation failed: {}", message);
            return;
        }
    };

    let fine_error = match run_experiment(81, lambda, alpha) {
        Ok(error) => error,
        Err(message) => {
            eprintln!("Computation failed: {}", message);
            return;
        }
    };

    println!("Convergence Diagnostic");
    println!("----------------------");
    println!("Error with N = 41            = {:.6e}", coarse_error);
    println!("Error with N = 81            = {:.6e}", fine_error);

    if fine_error > 0.0 {
        println!(
            "Observed error ratio         = {:.6}",
            coarse_error / fine_error
        );
    }
}
```

Program 19.3.2 demonstrates why weakly singular Volterra equations require quadrature rules designed around the singular structure of the kernel. Although the singularity in equation (19.3.7) is integrable, it violates the smoothness assumptions behind standard quadrature rules near $s=t$. Product integration addresses this problem by integrating the singular factor exactly and applying numerical approximation only to the smoother part of the integrand.

The numerical output confirms the expected behavior of the method. The discrete residual is close to machine precision, showing that the marching equations produced by the product-integration rule are solved accurately. The maximum absolute error decreases when the mesh is refined from $N=41$ to $N=81$, and the observed error ratio is close to two. This is consistent with the left-endpoint approximation of the smooth factor, which gives approximately first-order convergence in this introductory implementation.

The example also preserves the main computational advantage of Volterra equations: the solution is advanced causally from left to right. No future values enter the current update, and no dense global system is assembled. However, the number of history intervals grows quadratically with the number of mesh points, which reflects the cost of direct memory summation. For fractional or long-memory models, this motivates later improvements such as graded meshes, higher-order product integration, convolution acceleration, adaptive history compression, or spectral collocation methods.

Overall, Program 19.3.2 provides a bridge between the basic direct quadrature method and more specialized methods for fractional and memory-dependent Volterra equations. It shows that the causal marching structure remains valid even when the kernel is weakly singular, but the quadrature construction must be adapted to preserve accuracy and numerical stability near the diagonal $s=t$.

## 19.3.4. Applications in Population Dynamics and Materials with Memory

A representative application of Volterra equations occurs in *population dynamics*. The classical Lotka-Volterra predator-prey model is usually expressed as a coupled system of ordinary differential equations, but related population models can also be written in Volterra integral form. More generally, age-structured population models and epidemic models, such as those related to the Kermack-McKendrick framework, lead to Volterra equations for population density or infection dynamics. Renewal equations in demography also have Volterra structure because the current birth rate or population level depends on the accumulated history of previous population states.

Numerically, such models are solved by time-stepping quadrature. At a time $t_i$, one computes the new population value from the known history over earlier times $0\leq s<t_i.$ If data are noisy, smoothing or regularization may be required, but the causal structure preserves the forward-in-time character of the problem and supports well-posed sequential computation.

A second important application appears in *viscoelastic materials and control systems*. In viscoelasticity, stress and strain are related by hereditary integrals, which are Volterra equations with kernels depending on elapsed time, often written as $K(t-s)$. Such kernels represent memory: the response at time $t$ depends not only on the present load or strain but also on the accumulated history of the material. Solving the corresponding Volterra equation predicts how the material relaxes or responds over time.

Similar memory effects occur in electrical engineering, where current or voltage in systems with dielectric or inductive memory may satisfy Volterra-type integral equations. Direct evaluation of the full history term has $O(N^2)$ cost over $N$ time steps. For convolution-type kernels, fast convolution algorithms based on the FFT can reduce this cost to $O(N\log N),$ making long-time simulation more efficient. This acceleration is especially important when memory effects persist over long intervals or when high temporal resolution is required.

### Rust Implementation

Following the discussion in Section 19.3.4 on applications of Volterra equations in population dynamics, viscoelasticity, and memory-dependent systems, Program 19.3.3 provides a practical implementation of fast convolution for a Volterra memory kernel. The program focuses on the important special case in which the kernel depends only on elapsed time, $K(t-s)$, so that the history integral becomes a causal convolution. This structure is common in hereditary material models, where the present stress depends on the accumulated strain history through a relaxation kernel. The implementation compares direct history summation with an FFT-based convolution method, showing how the same memory response can be computed more efficiently when the kernel has convolution form. In this way, the program connects the application discussion to the computational issue emphasized in the section: direct memory evaluation has quadratic cost, while fast convolution reduces the cost for long-time simulations.

At the core of the implementation is the observation that a Volterra memory term with kernel $K(t-s)$ can be evaluated as a causal convolution. The program models this setting through a viscoelastic material response, where the current stress depends on both the instantaneous strain and an accumulated memory term. The function `relaxation_kernel(t, beta)` defines an exponentially decaying memory kernel, representing a material whose past deformation gradually loses influence as the time lag increases. The function `strain_history(t)` defines a prescribed strain input that combines oscillatory behavior with a slow ramp, producing a nontrivial memory response.

The program includes a small self-contained complex arithmetic type through the `Complex` structure. This type stores the real and imaginary parts of a complex number and implements the arithmetic operations required by the FFT. The methods `new`, `zero`, `exp_i`, and `scale` construct complex values, initialize zeros, generate complex roots of unity, and apply scalar normalization. The overloaded addition, subtraction, multiplication, and addition-assignment operations allow the FFT algorithm to be written in a compact and readable form.

The function `bit_reverse` supports the data-reordering stage of the radix-2 Cooley-Tukey FFT. In an iterative FFT, the input vector must first be arranged in bit-reversed order so that the subsequent butterfly operations combine entries in the correct sequence. The function `fft` then performs the in-place forward or inverse transform. When the inverse transform is requested, the result is scaled by the transform length, ensuring that applying the inverse FFT after the forward FFT recovers the original sequence up to roundoff error.

The function `uniform_time_grid` constructs the time grid over which the strain history and memory response are evaluated. The direct Volterra computation is carried out by `direct_memory_convolution`. This function follows the causal definition of the memory term literally: at each time level $t_i$, it sums contributions from all previous and current strain values. This is simple and transparent, but the number of accumulated history terms grows quadratically with the number of time steps.

The accelerated computation is implemented by `fft_memory_convolution`. The function first samples the relaxation kernel and strain history into zero-padded arrays. Zero padding is essential because the desired result is a linear convolution rather than a circular convolution. After applying the FFT to both arrays, the convolution is computed by pointwise multiplication in the frequency domain, followed by an inverse FFT. The first $N$ entries of the resulting sequence correspond to the causal Volterra memory values on the original time grid.

The function `stress_response` combines the instantaneous elastic contribution with the memory contribution. It computes a simple viscoelastic stress model in which the stress at each time point is the sum of an elastic term proportional to strain and a hereditary term proportional to the memory response. This reflects the application setting discussed in Section 19.3.4, where stress and strain are related through accumulated past influence.

The function `max_abs_difference` compares two computed vectors componentwise and returns the largest absolute difference. It is used to verify that the direct history summation and FFT-based convolution produce the same memory and stress responses. The function `direct_history_count` reports the number of terms used by the direct causal summation, while `fft_stage_count` reports the number of radix-2 FFT stages. These diagnostics make the computational contrast visible: the direct method performs a triangular number of history summations, whereas the FFT method uses a padded transform length and logarithmic number of stages.

The `main` function brings the full example together. It defines the final time, number of time steps, relaxation rate, elastic modulus, and memory strength. It then builds the time grid, evaluates the strain history, computes the memory term by both direct summation and FFT convolution, and forms the corresponding stress responses. Finally, it prints consistency diagnostics and representative values. The near roundoff-level agreement between the direct and FFT results confirms that the accelerated method computes the same discrete Volterra memory term.

```rust
// Program 19.3.3: FFT-Accelerated Convolution for a Volterra Memory Kernel
//
// Problem statement:
// Evaluate a Volterra memory term of convolution type,
//
//     m(t_i) = integral_0^{t_i} K(t_i - s) eps(s) ds,
//
// where K(t - s) is a relaxation kernel and eps(s) is a prescribed strain
// history. In discrete form, this becomes a causal convolution:
//
//     m_i approx h * sum_{j=0}^{i} K_{i-j} eps_j.
//
// The program compares two approaches:
//
// 1. Direct history summation, with O(N^2) cost.
// 2. FFT-based linear convolution, with O(N log N) cost.
//
// This is representative of Volterra memory terms in viscoelastic materials,
// where stress depends on the accumulated strain history.

use std::f64::consts::PI;
use std::ops::{Add, AddAssign, Mul, Sub};

#[derive(Clone, Copy, Debug)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    fn exp_i(theta: f64) -> Self {
        Self {
            re: theta.cos(),
            im: theta.sin(),
        }
    }

    fn scale(self, alpha: f64) -> Self {
        Self {
            re: alpha * self.re,
            im: alpha * self.im,
        }
    }

}

impl Add for Complex {
    type Output = Complex;

    fn add(self, rhs: Complex) -> Complex {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Complex) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex {
    type Output = Complex;

    fn sub(self, rhs: Complex) -> Complex {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Complex;

    fn mul(self, rhs: Complex) -> Complex {
        Complex::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

/// Reverse the lowest `bits` bits of an integer.
fn bit_reverse(mut value: usize, bits: u32) -> usize {
    let mut reversed = 0usize;

    for _ in 0..bits {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }

    reversed
}

/// In-place radix-2 Cooley-Tukey FFT.
///
/// If inverse = false, this computes the forward transform.
/// If inverse = true, this computes the inverse transform and divides by n.
fn fft(values: &mut [Complex], inverse: bool) {
    let n = values.len();

    assert!(
        n.is_power_of_two(),
        "FFT length must be a power of two."
    );

    let bits = n.trailing_zeros();

    for i in 0..n {
        let j = bit_reverse(i, bits);

        if j > i {
            values.swap(i, j);
        }
    }

    let mut length = 2;

    while length <= n {
        let angle_sign = if inverse { 1.0 } else { -1.0 };
        let angle = angle_sign * 2.0 * PI / length as f64;
        let root = Complex::exp_i(angle);

        for start in (0..n).step_by(length) {
            let mut twiddle = Complex::new(1.0, 0.0);

            for k in 0..(length / 2) {
                let even = values[start + k];
                let odd = twiddle * values[start + k + length / 2];

                values[start + k] = even + odd;
                values[start + k + length / 2] = even - odd;

                twiddle = twiddle * root;
            }
        }

        length *= 2;
    }

    if inverse {
        let scale = 1.0 / n as f64;

        for value in values.iter_mut() {
            *value = value.scale(scale);
        }
    }
}

/// Relaxation kernel K(t) = exp(-beta t).
///
/// This is a short-memory kernel commonly used as a simple model of
/// exponentially decaying material memory.
fn relaxation_kernel(t: f64, beta: f64) -> f64 {
    (-beta * t).exp()
}

/// Prescribed strain history eps(t).
///
/// The strain combines slow oscillation and a small ramp, giving a smooth
/// but nontrivial memory response.
fn strain_history(t: f64) -> f64 {
    (2.0 * PI * t).sin() + 0.25 * t
}

/// Build a uniform time grid on [0, final_time].
fn uniform_time_grid(final_time: f64, n: usize) -> Vec<f64> {
    let h = final_time / (n as f64 - 1.0);

    (0..n).map(|i| i as f64 * h).collect()
}

/// Direct causal Volterra history summation.
///
/// This computes
///
///     m_i = h * sum_{j=0}^{i} K(t_i - t_j) eps_j.
///
/// The number of kernel-history contributions grows like N(N + 1) / 2.
fn direct_memory_convolution(
    time: &[f64],
    strain: &[f64],
    beta: f64,
) -> Vec<f64> {
    let n = time.len();
    let h = time[1] - time[0];

    let mut memory = vec![0.0; n];

    for i in 0..n {
        let mut sum = 0.0;

        for j in 0..=i {
            let lag = time[i] - time[j];
            sum += relaxation_kernel(lag, beta) * strain[j];
        }

        memory[i] = h * sum;
    }

    memory
}

/// FFT-based linear convolution for the same Volterra memory term.
///
/// The causal convolution is embedded into a zero-padded linear convolution.
/// The first N entries of the result correspond to the Volterra history term.
fn fft_memory_convolution(
    time: &[f64],
    strain: &[f64],
    beta: f64,
) -> Vec<f64> {
    let n = time.len();
    let h = time[1] - time[0];

    let convolution_length = 2 * n - 1;
    let fft_length = convolution_length.next_power_of_two();

    let mut kernel_values = vec![Complex::zero(); fft_length];
    let mut strain_values = vec![Complex::zero(); fft_length];

    for i in 0..n {
        let lag = time[i] - time[0];
        kernel_values[i] = Complex::new(relaxation_kernel(lag, beta), 0.0);
        strain_values[i] = Complex::new(strain[i], 0.0);
    }

    fft(&mut kernel_values, false);
    fft(&mut strain_values, false);

    let mut product = vec![Complex::zero(); fft_length];

    for i in 0..fft_length {
        product[i] = kernel_values[i] * strain_values[i];
    }

    fft(&mut product, true);

    let mut memory = vec![0.0; n];

    for i in 0..n {
        memory[i] = h * product[i].re;
    }

    memory
}

/// Compute a simple viscoelastic stress model:
///
///     sigma(t_i) = E_0 eps(t_i) + eta m(t_i),
///
/// where m(t_i) is the Volterra memory term.
fn stress_response(
    strain: &[f64],
    memory: &[f64],
    elastic_modulus: f64,
    memory_strength: f64,
) -> Vec<f64> {
    strain
        .iter()
        .zip(memory.iter())
        .map(|(&eps_i, &m_i)| elastic_modulus * eps_i + memory_strength * m_i)
        .collect()
}

/// Maximum absolute difference between two vectors.
fn max_abs_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Count the number of direct history contributions.
fn direct_history_count(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Estimate the number of butterfly stages in the FFT method.
fn fft_stage_count(fft_length: usize) -> usize {
    fft_length.trailing_zeros() as usize
}

fn main() {
    let final_time = 8.0;
    let n = 512;
    let beta = 1.5;
    let elastic_modulus = 2.0;
    let memory_strength = 0.8;

    let time = uniform_time_grid(final_time, n);

    let strain: Vec<f64> = time
        .iter()
        .map(|&t| strain_history(t))
        .collect();

    let direct_memory = direct_memory_convolution(&time, &strain, beta);
    let fft_memory = fft_memory_convolution(&time, &strain, beta);

    let direct_stress =
        stress_response(&strain, &direct_memory, elastic_modulus, memory_strength);

    let fft_stress =
        stress_response(&strain, &fft_memory, elastic_modulus, memory_strength);

    let memory_difference = max_abs_difference(&direct_memory, &fft_memory);
    let stress_difference = max_abs_difference(&direct_stress, &fft_stress);

    let convolution_length = 2 * n - 1;
    let fft_length = convolution_length.next_power_of_two();
    let direct_terms = direct_history_count(n);
    let stages = fft_stage_count(fft_length);

    println!("FFT-Accelerated Volterra Memory Convolution");
    println!("===========================================");
    println!();
    println!("Application Model");
    println!("-----------------");
    println!("Application                 = viscoelastic material memory");
    println!("Kernel K(t - s)             = exp(-beta(t - s))");
    println!("Beta                        = {:.6}", beta);
    println!("Stress model                = E0 * strain + eta * memory");
    println!("Elastic modulus E0          = {:.6}", elastic_modulus);
    println!("Memory strength eta         = {:.6}", memory_strength);
    println!();
    println!("Discretization");
    println!("--------------");
    println!("Final time                  = {:.6}", final_time);
    println!("Time steps N                = {}", n);
    println!("Time step h                 = {:.6e}", time[1] - time[0]);
    println!("Direct history terms        = {}", direct_terms);
    println!("Linear convolution length   = {}", convolution_length);
    println!("FFT length                  = {}", fft_length);
    println!("FFT stages                  = {}", stages);
    println!();
    println!("Consistency Diagnostics");
    println!("-----------------------");
    println!(
        "max_i |memory_direct - memory_fft| = {:.6e}",
        memory_difference
    );
    println!(
        "max_i |stress_direct - stress_fft| = {:.6e}",
        stress_difference
    );
    println!();
    println!("Representative Values");
    println!("---------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18} {:>18}",
        "i", "t_i", "strain", "memory_direct", "memory_fft", "stress_fft"
    );

    for i in 0..n {
        if i % 64 == 0 || i == n - 1 {
            println!(
                "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.10} {:>18.10}",
                i,
                time[i],
                strain[i],
                direct_memory[i],
                fft_memory[i],
                fft_stress[i]
            );
        }
    }
}
```

Program 19.3.3 demonstrates how a Volterra equation with a convolution kernel can be evaluated more efficiently by using the structure $K(t-s)$. The direct method is the most literal implementation of the history integral: each new time step accumulates all previous contributions. This preserves the causal interpretation clearly, but the total number of history terms grows quadratically with the number of time steps.

The FFT-based method computes the same discrete memory term by embedding the causal convolution into a zero-padded linear convolution. The numerical output shows that the direct and FFT memory responses agree to roundoff accuracy, confirming that the fast method preserves the discrete Volterra history term. The stress response also agrees to roundoff accuracy because it is computed from the same strain and memory quantities.

This example illustrates why convolution acceleration is valuable in long-memory applications. In viscoelasticity, electrical memory models, renewal equations, and related systems, the memory term may need to be evaluated over many time steps. A direct history sum is simple but becomes expensive as the simulation length grows. When the kernel depends only on elapsed time, fast convolution provides a practical route to reducing the computational burden while retaining the causal structure of the Volterra model.

Although the program uses a simple exponential relaxation kernel and a self-contained radix-2 FFT, the same idea extends to more realistic memory kernels and larger simulations. In production codes, one would usually use an optimized FFT library, block convolution, adaptive history compression, or specialized convolution quadrature. The present implementation is nevertheless useful pedagogically because it shows the complete numerical mechanism: the Volterra memory term is first recognized as a causal convolution, then evaluated either directly or through frequency-domain acceleration.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/vEWi9scnk51kPwI6qbND.4","tags":[]}

# 19.4. Integral Equations with Singular Kernels

Many integral equations arising from physical models contain kernels that become singular when the source and observation points coincide. These singularities are especially common in boundary integral formulations of Laplace, Helmholtz, elasticity, fracture, and fluid-flow problems. This section examines weak, strong, and hypersingular kernels, explains why ordinary quadrature is inadequate near singular points, and develops the main numerical techniques used to handle them, including variable transformations, product integration, singularity subtraction, principal-value interpretation, and unified Clenshaw-Curtis quadrature methods.

## 19.4.1. Nature and Classification of Singular Kernels

Integral equations frequently involve *singular kernels*, meaning that the kernel $K(t,s)$ becomes unbounded, discontinuous, or non-integrable at certain points of the integration domain. The most common singularity occurs near the diagonal $s=t$, where the source and observation points coincide. Such kernels arise naturally in boundary integral formulations of partial differential equations, especially when the kernel is derived from a Green’s function.

For example, in two-dimensional boundary integral methods for Laplace or Helmholtz equations, the Green’s function has a logarithmic singularity of the form:

$$K(t,s)\sim -\frac{1}{2\pi}\ln |t-s| \tag{19.4.1}$$

This is a *weak logarithmic singularity*. It becomes unbounded as $s\to t$, but it remains integrable. In three dimensions, the corresponding Green’s function has the form:

$$K(t,s)\sim \frac{1}{4\pi |t-s|} \tag{19.4.2}$$

which is a $1/r$-type singularity. Such singular kernels are fundamental in potential theory, acoustics, electromagnetics, and boundary element methods.

More severe singularities occur when derivatives of the Green’s function are used. Differentiating the kernel may produce *strong singularities*, such as Cauchy-type kernels, or even *hypersingular kernels*, for example,

$$K(t,s)\sim \frac{1}{|t-s|^2} \tag{19.4.3}$$

These kernels appear when computing stresses, fluxes, normal derivatives, or tractions on boundaries. For this reason, singular integral equations are particularly important in boundary integral equations for elasticity, acoustics, fracture mechanics, and fluid dynamics (Chen and Li, 2023).

Singular integrals are commonly classified according to the strength of the singularity. A *weak singularity* remains integrable in the ordinary sense. A typical algebraic weak singularity has the form:

$$K(t,s)\sim |t-s|^{-\alpha},\qquad 0<\alpha<1 \tag{19.4.4}$$

Since $\alpha<1$, the singularity is integrable. By contrast, strong singularities generally require interpretation in the sense of a *Cauchy principal value*, while hypersingular kernels require further regularization, finite-part interpretations, or analytic cancellation. Ordinary quadrature rules are not reliable for such integrals unless modified, because the singularity may dominate the local behavior and cause large errors near the diagonal.

The numerical treatment of singular kernels therefore requires techniques beyond standard quadrature. The central idea is either to remove the singular behavior analytically, transform the variables so that the singularity is weakened, or construct quadrature rules whose weights already account for the singular factor.

## 19.4.2. Regularization by Change of Variables and Product Integration

One of the simplest approaches for weak endpoint singularities is a *change of variables*. The purpose is to transform the integral so that the singular factor is absorbed into the Jacobian of the transformation. Suppose an algebraic singularity occurs at the left endpoint $s=a$. A common mapping is:

$$s=a+(b-a)x^2,\qquad 0\le x\le 1, \tag{19.4.5}$$

so that,

$$ds=2(b-a)x\,dx  \tag{19.4.6}$$

The factor $x$ in the Jacobian can cancel the square-root singularity that appears as $s\to a$. For example, an integral of the form:

$$\int_a^b \frac{f(s)}{\sqrt{s-a}}\,ds \tag{19.4.7}$$

can be transformed into a regular integral over $x\in[0,1]$. Since $s-a=(b-a)x^2$, we have:

$$\sqrt{s-a}=\sqrt{b-a}\,x,$$

and therefore,

$$\int_a^b \frac{f(s)}{\sqrt{s-a}}\,ds = 2\sqrt{b-a}\int_0^1 f\!\left(a+(b-a)x^2\right)\,dx \tag{19.4.8}$$

The singular factor has disappeared from the transformed integrand. Such mappings are standard tools for integrable algebraic singularities and are particularly useful when the location and form of the singularity are known.

A second major approach is *product integration*, also called special quadrature. Instead of applying an ordinary quadrature rule to the entire integrand, one treats the singular part as a weight function. For example, for kernels of the form:

$$K(t,s)\sim \frac{1}{\sqrt{t-s}},$$

on an interval $[0,t]$, one may use Gauss-Jacobi quadrature with weight:

$$(t-s)^{-1/2} \tag{19.4.9}$$

The quadrature nodes and weights are chosen so that polynomials multiplied by the singular weight are integrated accurately. In practice, this means that the singular behavior is built directly into the quadrature rule rather than being treated as an error-producing irregularity.

For logarithmic kernels, which commonly arise in two-dimensional boundary integral equations, specialized logarithmic quadrature or weight-extraction methods are used. The integral is written so that the logarithmic factor is separated from the smooth part. The quadrature weights are then designed to integrate functions of the form $p(s)\ln |t-s|,$ where $p(s)$ is a polynomial or smooth approximation. This procedure preserves high accuracy even near the diagonal singularity.

### Rust Implementation

Following the discussion in Section 19.4.2 on regularization by change of variables and product integration, Program 19.4.1 provides a practical implementation of two complementary techniques for weakly singular integrals. The first part follows the endpoint transformation in equations (19.4.5) through (19.4.8), where a square-root singularity is removed by absorbing it into the Jacobian of a nonlinear change of variables. The second part demonstrates product integration for a weak diagonal singularity of the type described in equation (19.4.9), where the singular factor is integrated analytically while the smooth part is approximated by interpolation. Together, these examples show that weak singularities should not be treated as ordinary nonsmooth integrands. Instead, their known analytic structure should be incorporated directly into the numerical method so that standard quadrature is applied only after the singular behavior has been regularized or built into the weights.

At the core of the implementation is the distinction between the singular factor and the smooth part of the integrand. The function `smooth_function(s)` defines the smooth factor used in both examples. By choosing a simple linear function, the program allows exact reference values to be computed analytically, making it possible to verify the accuracy of both regularization strategies.

The function `trapezoidal_rule` implements the composite trapezoidal rule for regular integrals. It is deliberately used only after the singularity has been removed by the transformation. This reflects the key idea behind equations (19.4.5) through (19.4.8): ordinary quadrature becomes appropriate only after the unbounded endpoint behavior has been absorbed into the change of variables.

The function `exact_endpoint_singular_integral` evaluates the exact value of the endpoint-singular integral in equation (19.4.7) for the chosen smooth function. This exact value provides a benchmark for the transformed numerical approximation. The function `transformed_endpoint_integral` implements the mapping introduced in equation (19.4.5). Under this mapping, the Jacobian described in equation (19.4.6) cancels the square-root singularity, leading to the regularized integral in equation (19.4.8). The program then applies the trapezoidal rule to the transformed smooth integrand.

The function `uniform_mesh` constructs a uniform mesh on the interval used for the product-integration example. This mesh defines the subintervals over which the singular factor is integrated analytically. Unlike ordinary quadrature, the method does not evaluate the singular factor directly at the diagonal endpoint. Instead, it treats the singular part as a weight function.

The function `exact_weakly_singular_history` computes the exact value of the weakly singular history integral used in the product-integration test. This provides a second reference value for checking the method. The function `product_integration_weights` is the key component of the second demonstration. It computes the subinterval weights obtained by integrating the weak singular factor against local linear basis functions. In this way, the singular behavior is built into the quadrature weights rather than being sampled numerically.

The function `product_integration_history` assembles the complete product-integration approximation. On each subinterval, the smooth factor is approximated by linear interpolation, while the singular factor is integrated exactly through the weights computed by `product_integration_weights`. This implements the principle described around equation (19.4.9): the singular factor is treated as part of the quadrature rule, not as an ordinary function to be evaluated by standard quadrature.

The small helper function `abs_error` computes the absolute error between a numerical approximation and its exact reference value. The `main` function then runs both demonstrations. In the first part, it evaluates the endpoint-singular integral after the change of variables and compares the result with the exact value. In the second part, it evaluates the weakly singular diagonal integral using product integration and compares the result with its exact value. The printed output therefore verifies both techniques and shows how each one handles the singular behavior analytically before numerical approximation is applied.

```rust
// Program 19.4.1: Regularization of Weak Singular Integrals by
// Variable Transformation and Product Integration
//
// Problem statement:
// Demonstrate two numerical treatments for weakly singular integrals:
//
// 1. Change-of-variables regularization for the endpoint singularity
//
//        integral_a^b f(s) / sqrt(s - a) ds.
//
//    Using s = a + (b - a)x^2 removes the square-root singularity
//    and produces a regular integral over x in [0, 1].
//
// 2. Product integration for the weakly singular Volterra-type integral
//
//        integral_0^t (t - s)^(-alpha) f(s) ds,
//
//    where 0 < alpha < 1. The singular factor is integrated exactly on
//    each subinterval, while f(s) is approximated by a local linear function.
//
// The smooth test function is f(s) = 1 + s, for which exact reference
// values are available.

/// Smooth part of the integrand.
fn smooth_function(s: f64) -> f64 {
    1.0 + s
}

/// Composite trapezoidal rule for a regular function on [a, b].
fn trapezoidal_rule<F>(a: f64, b: f64, n: usize, integrand: F) -> f64
where
    F: Fn(f64) -> f64,
{
    if n < 2 {
        panic!("At least two quadrature points are required.");
    }

    let h = (b - a) / (n as f64 - 1.0);
    let mut sum = 0.0;

    for i in 0..n {
        let x = a + i as f64 * h;
        let weight = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
        sum += weight * integrand(x);
    }

    h * sum
}

/// Exact value of
///
///     integral_a^b (1 + s) / sqrt(s - a) ds.
///
/// Writing u = s - a gives
///
///     integral_0^(b-a) (1 + a + u) u^(-1/2) du.
fn exact_endpoint_singular_integral(a: f64, b: f64) -> f64 {
    let length = b - a;

    let first = 2.0 * (1.0 + a) * length.sqrt();
    let second = (2.0 / 3.0) * length.powf(1.5);

    first + second
}

/// Compute the endpoint singular integral by applying the transformation
///
///     s = a + (b - a)x^2.
///
/// The transformed integral is regular:
///
///     2 sqrt(b - a) integral_0^1 f(a + (b - a)x^2) dx.
fn transformed_endpoint_integral(a: f64, b: f64, n: usize) -> f64 {
    let length = b - a;
    let scale = 2.0 * length.sqrt();

    trapezoidal_rule(0.0, 1.0, n, |x| {
        let s = a + length * x * x;
        scale * smooth_function(s)
    })
}

/// Build a uniform mesh on [0, t].
fn uniform_mesh(final_time: f64, n: usize) -> Vec<f64> {
    if n < 2 {
        panic!("At least two mesh points are required.");
    }

    let h = final_time / (n as f64 - 1.0);

    (0..n).map(|i| i as f64 * h).collect()
}

/// Exact value of
///
///     integral_0^t (t - s)^(-alpha) (1 + s) ds.
///
/// Since f(s) = 1 + s,
///
///     integral_0^t (t - s)^(-alpha) ds
///     + integral_0^t (t - s)^(-alpha) s ds.
///
/// The second term is evaluated by substituting u = t - s.
fn exact_weakly_singular_history(t: f64, alpha: f64) -> f64 {
    if t == 0.0 {
        return 0.0;
    }

    let one_minus_alpha = 1.0 - alpha;
    let two_minus_alpha = 2.0 - alpha;

    let constant_part = t.powf(one_minus_alpha) / one_minus_alpha;

    let linear_part = t * t.powf(one_minus_alpha) / one_minus_alpha
        - t.powf(two_minus_alpha) / two_minus_alpha;

    constant_part + linear_part
}

/// Product-integration weights for one subinterval [left, right].
///
/// On [left, right], f(s) is approximated by linear interpolation:
///
///     f(s) approx f_left phi_left(s) + f_right phi_right(s).
///
/// The weights returned by this function are
///
///     integral_left^right (t - s)^(-alpha) phi_left(s) ds,
///     integral_left^right (t - s)^(-alpha) phi_right(s) ds.
///
/// The singular factor is integrated analytically, so the endpoint
/// singularity at s = t causes no direct numerical evaluation problem.
fn product_integration_weights(
    t: f64,
    left: f64,
    right: f64,
    alpha: f64,
) -> (f64, f64) {
    let h = right - left;

    let y_left = t - left;
    let y_right = t - right;

    let int_y_neg_alpha =
        (y_left.powf(1.0 - alpha) - y_right.powf(1.0 - alpha)) / (1.0 - alpha);

    let int_y_one_minus_alpha =
        (y_left.powf(2.0 - alpha) - y_right.powf(2.0 - alpha)) / (2.0 - alpha);

    let weight_left = (int_y_one_minus_alpha - y_right * int_y_neg_alpha) / h;
    let weight_right = (y_left * int_y_neg_alpha - int_y_one_minus_alpha) / h;

    (weight_left, weight_right)
}

/// Product integration for
///
///     integral_0^t (t - s)^(-alpha) f(s) ds.
///
/// The mesh is assumed to cover [0, t]. The singular factor is treated
/// analytically and the smooth function f is approximated linearly on each
/// subinterval.
fn product_integration_history(t: f64, n: usize, alpha: f64) -> f64 {
    if !(0.0 < alpha && alpha < 1.0) {
        panic!("The exponent alpha must satisfy 0 < alpha < 1.");
    }

    let mesh = uniform_mesh(t, n);
    let mut integral = 0.0;

    for j in 0..(n - 1) {
        let left = mesh[j];
        let right = mesh[j + 1];

        let f_left = smooth_function(left);
        let f_right = smooth_function(right);

        let (weight_left, weight_right) =
            product_integration_weights(t, left, right, alpha);

        integral += weight_left * f_left + weight_right * f_right;
    }

    integral
}

/// Compute absolute error.
fn abs_error(approx: f64, exact: f64) -> f64 {
    (approx - exact).abs()
}

fn main() {
    let a = 0.0;
    let b = 1.0;

    let endpoint_n = 81;

    let endpoint_exact = exact_endpoint_singular_integral(a, b);
    let endpoint_transformed = transformed_endpoint_integral(a, b, endpoint_n);

    let t = 1.0;
    let alpha = 0.5;

    let product_n = 41;

    let product_exact = exact_weakly_singular_history(t, alpha);
    let product_approx = product_integration_history(t, product_n, alpha);

    println!("Weak Singular Integral Regularization");
    println!("=====================================");
    println!();

    println!("Part 1: Change of Variables for an Endpoint Singularity");
    println!("-------------------------------------------------------");
    println!("Integral                    = integral_a^b (1 + s) / sqrt(s - a) ds");
    println!("Interval [a, b]             = [{:.1}, {:.1}]", a, b);
    println!("Mapping                     = s = a + (b - a)x^2");
    println!("Quadrature after transform  = composite trapezoidal rule");
    println!("Quadrature points           = {}", endpoint_n);
    println!("Exact value                 = {:.12}", endpoint_exact);
    println!("Transformed approximation   = {:.12}", endpoint_transformed);
    println!(
        "Absolute error              = {:.6e}",
        abs_error(endpoint_transformed, endpoint_exact)
    );
    println!();

    println!("Part 2: Product Integration for a Weak Diagonal Singularity");
    println!("-----------------------------------------------------------");
    println!("Integral                    = integral_0^t (t - s)^(-alpha)(1 + s) ds");
    println!("Final point t               = {:.6}", t);
    println!("Singularity exponent alpha  = {:.6}", alpha);
    println!("Smooth factor approximation = piecewise linear interpolation");
    println!("Singular factor treatment   = exact subinterval integration");
    println!("Subintervals                = {}", product_n - 1);
    println!("Exact value                 = {:.12}", product_exact);
    println!("Product-integration value   = {:.12}", product_approx);
    println!(
        "Absolute error              = {:.6e}",
        abs_error(product_approx, product_exact)
    );
}
```

Program 19.4.1 demonstrates two basic ways of treating weak singularities without applying ordinary quadrature directly to an unbounded integrand. In the first example, the square-root endpoint singularity is removed by the change of variables introduced in equation (19.4.5). The resulting transformed integral is regular, so the composite trapezoidal rule can be applied in the transformed variable. The small remaining error is then a standard quadrature error rather than a failure caused by evaluating the original singular integrand.

The second example illustrates the product-integration idea. Instead of transforming the variable, the method incorporates the singular factor into the quadrature weights. Because the smooth factor in this example is linear and the product-integration rule uses local linear interpolation, the computed value agrees with the exact value to roundoff accuracy. This makes the example especially useful pedagogically: it shows that the singularity itself is not the obstacle when its analytic form is known and correctly included in the quadrature construction.

Together, the two examples reinforce the main message of Section 19.4.2. Weak singularities are integrable, but they require numerical methods that respect their local structure. A change of variables removes the singularity by reshaping the integration domain, while product integration keeps the original variable but builds the singular behavior into the weights. Both approaches prepare the ground for the stronger singularities and principal-value treatments discussed in the following subsection.

## 19.4.3. Singularity Subtraction and Principal-Value Integrals

For stronger singularities, especially Cauchy-type kernels, one often uses *singularity subtraction* or analytic regularization. The principle is to split the integrand into a singular part that can be evaluated analytically and a remaining part that is smooth enough for ordinary quadrature.

Consider the Cauchy-type integral:

$$\int_a^b \frac{f(s)}{t-s}\,ds \tag{19.4.10}$$

This integral is singular at $s=t$ and must be interpreted as a Cauchy principal value. We subtract and add $f(t)$ in the numerator:

$$\operatorname{P.V.}\int_a^b \frac{f(s)}{t-s}\,ds = f(t)\operatorname{P.V.}\int_a^b \frac{1}{t-s}\,ds+\int_a^b \frac{f(s)-f(t)}{t-s}\,ds \tag{19.4.11}$$

The first integral is known analytically. Since,

$$\operatorname{P.V.}\int_a^b \frac{1}{t-s}\,ds = \ln\left|\frac{t-a}{b-t}\right|, \tag{19.4.12}$$

equation (19.4.11) becomes:

$$\operatorname{P.V.}\int_a^b \frac{f(s)}{t-s}\,ds = f(t)\ln\left|\frac{t-a}{b-t}\right|+\int_a^b \frac{f(s)-f(t)}{t-s}\,ds \tag{19.4.13}$$

The second integral is now regular, because $f(s)-f(t)$ vanishes as $s\to t$. If $f$ is differentiable, then:

$$\frac{f(s)-f(t)}{t-s}$$

has a finite limit as $s\to t$. This is the essential cancellation that makes singularity subtraction effective. This idea extends to more complicated kernels by decomposing:

$$K(t,s)=K_{\mathrm{sing}}(t,s)+K_{\mathrm{reg}}(t,s) \tag{19.4.14}$$

where $K_{\mathrm{sing}}$ contains the known singular behavior and $K_{\mathrm{reg}}$ is smoother. The integral involving $K_{\mathrm{sing}}$ is evaluated analytically or by a special rule, while the regular remainder is treated by ordinary high-order quadrature. This approach is widely used in boundary element methods, fracture mechanics, and fluid-flow boundary integral formulations.

After discretization, singular kernels often produce matrices whose diagonal entries are undefined or extremely large. For example, a Cauchy kernel discretized at nodes $t_i$ and $t_j$ gives:

$$K(t_i,t_j)=\frac{1}{t_i-t_j} \tag{19.4.15}$$

which is undefined when $i=j$. Principal-value interpretation replaces these problematic diagonal entries by appropriate analytic limiting values. In practice, one writes the discretized singular integral in the form:

$$\int_a^b K(t_i,s)f(s)\,ds\approx\sum_{j\ne i} K(t_i,t_j)w_j f(t_j)+A_i f(t_i) \tag{19.4.16}$$

where $A_i$ is a known analytic contribution associated with the singular part of the kernel. This formula shows explicitly how the off-diagonal part is handled by quadrature, while the diagonal singularity is replaced by a regularized analytic term.

### Rust Implementation

Following the discussion in Section 19.4.3 on singularity subtraction and Cauchy principal-value integrals, Program 19.4.2 provides a practical implementation of analytic regularization for a Cauchy-type singular kernel. The program follows the subtraction structure developed in equations (19.4.11) through (19.4.13), where the singular part of the integral is separated from a regularized remainder. Instead of applying quadrature directly to the singular expression in equation (19.4.10), the method evaluates the logarithmic principal-value contribution analytically and then applies composite Simpson quadrature only to the smooth remainder. This implementation demonstrates the central numerical idea of singularity subtraction: a problematic diagonal singularity can be removed by analytic cancellation before numerical integration is performed.

At the core of the implementation is the subtraction formula for the Cauchy principal-value integral. The function `smooth_function(s)` defines the smooth numerator $f(s)$ appearing in equation (19.4.10). A trigonometric-polynomial function is used so that the numerator is smooth but nontrivial. This makes the cancellation near the singular point $s=t$ visible in the numerical output.

The function `smooth_function_derivative(s)` computes the derivative of the smooth numerator. This derivative is needed because the regularized expression contains a removable singularity at $s=t$. As explained after equation (19.4.13), the numerator $f(s)-f(t)$ vanishes as $s\to t$, so the quotient has a finite limiting value. The code uses this analytic limiting value directly whenever the quadrature point is numerically indistinguishable from $t$.

The function `regularized_integrand(s, t)` implements the smooth remainder integrand in equation (19.4.13). Away from $s=t$, it evaluates the quotient directly. At $s=t$, it replaces the quotient by the limiting value $-f'(t)$. This is the main cancellation step in the program: the original singular integrand is not evaluated at the diagonal, and the removable singularity is handled analytically.

The function `simpson_rule` implements the composite Simpson rule for regular integrands. It is deliberately applied only after singularity subtraction has transformed the problem into a smooth numerical integral. This reflects the principle of Section 19.4.3: ordinary quadrature becomes appropriate only after the singular part has been isolated and removed from the numerical integrand.

The function `analytic_singular_contribution` evaluates the logarithmic term in equation (19.4.12) multiplied by $f(t)$, as it appears in equation (19.4.13). This term represents the analytic contribution of the singular kernel. The function also checks that the target point $t$ lies strictly inside the interval, since the logarithmic expression is defined for an interior principal-value point.

The function `principal_value_by_subtraction` combines the two components of the computation. It first evaluates the analytic singular contribution and then integrates the regularized remainder using Simpson quadrature. The sum of these two terms gives the final principal-value approximation. This mirrors the decomposition in equation (19.4.13) and separates the singular analysis from the numerical quadrature.

The function `decomposition_residual` checks the internal consistency of the computation by verifying that the reported total equals the sum of the analytic contribution and the regularized numerical remainder. The function `local_cancellation_diagnostic` evaluates the regularized integrand just to the left of $t$, exactly at $t$, and just to the right of $t$. These three values should be close, showing that the regularized integrand is smooth across the point where the original kernel was singular.

The `main` function sets the integration interval, target point, and number of Simpson panels. It then computes the principal-value integral using singularity subtraction, evaluates the cancellation diagnostic, and prints the analytic contribution, regularized remainder, final principal-value result, and decomposition residual. The printed output therefore verifies both the analytic cancellation and the numerical integration of the regularized problem.

```rust
// Program 19.4.2: Singularity Subtraction for a Cauchy Principal-Value Integral
//
// Problem statement:
// Evaluate the Cauchy principal-value integral
//
//     P.V. integral_a^b f(s) / (t - s) ds,
//
// where the kernel is singular at s = t. The program follows the subtraction
// identity
//
//     P.V. integral_a^b f(s)/(t - s) ds
//       = f(t) ln |(t - a)/(b - t)|
//         + integral_a^b (f(s) - f(t))/(t - s) ds.
//
// The first term is evaluated analytically, while the second integral is
// regular because f(s) - f(t) vanishes as s approaches t. Ordinary Simpson
// quadrature is then applied only to the regularized remainder.

use std::f64::consts::PI;

/// Smooth function f(s) used in the Cauchy principal-value integral.
///
/// A trigonometric-polynomial function is chosen so that the cancellation
/// near s = t is nontrivial while the function remains smooth on [a, b].
fn smooth_function(s: f64) -> f64 {
    (PI * s).sin() + 0.25 * s * s
}

/// Derivative f'(s), used to define the removable value of the regularized
/// integrand at s = t.
///
/// Since
///
///     lim_{s -> t} (f(s) - f(t)) / (t - s) = -f'(t),
///
/// the limiting value is inserted directly when s is numerically equal to t.
fn smooth_function_derivative(s: f64) -> f64 {
    PI * (PI * s).cos() + 0.5 * s
}

/// Regularized integrand
///
///     (f(s) - f(t)) / (t - s).
///
/// The apparent singularity at s = t is removable. At that point, the
/// analytic limiting value -f'(t) is used instead of performing division
/// by zero.
fn regularized_integrand(s: f64, t: f64) -> f64 {
    let distance = s - t;

    if distance.abs() < 1.0e-12 {
        -smooth_function_derivative(t)
    } else {
        (smooth_function(s) - smooth_function(t)) / (t - s)
    }
}

/// Composite Simpson rule on [a, b].
///
/// The number of panels must be positive and even. This function is applied
/// only to the regularized remainder, not to the original singular integrand.
fn simpson_rule<F>(a: f64, b: f64, panels: usize, integrand: F) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
{
    if panels == 0 || panels % 2 != 0 {
        return Err("Composite Simpson rule requires a positive even number of panels.".to_string());
    }

    let h = (b - a) / panels as f64;
    let mut sum = integrand(a) + integrand(b);

    for i in 1..panels {
        let x = a + i as f64 * h;

        if i % 2 == 0 {
            sum += 2.0 * integrand(x);
        } else {
            sum += 4.0 * integrand(x);
        }
    }

    Ok(h * sum / 3.0)
}

/// Evaluate the analytic principal-value contribution
///
///     f(t) ln |(t - a)/(b - t)|.
///
/// This corresponds to the singular part that remains after subtracting
/// and adding f(t) in the numerator.
fn analytic_singular_contribution(a: f64, b: f64, t: f64) -> Result<f64, String> {
    if !(a < t && t < b) {
        return Err("The target point t must lie strictly inside (a, b).".to_string());
    }

    let log_factor = ((t - a) / (b - t)).abs().ln();

    Ok(smooth_function(t) * log_factor)
}

/// Evaluate the Cauchy principal-value integral by singularity subtraction.
///
/// The singular part is evaluated analytically, and the regularized
/// remainder is evaluated with Simpson quadrature.
fn principal_value_by_subtraction(
    a: f64,
    b: f64,
    t: f64,
    panels: usize,
) -> Result<(f64, f64, f64), String> {
    let analytic_part = analytic_singular_contribution(a, b, t)?;

    let regular_part = simpson_rule(a, b, panels, |s| regularized_integrand(s, t))?;

    let total = analytic_part + regular_part;

    Ok((analytic_part, regular_part, total))
}

/// Evaluate the residual consistency of the subtraction formula.
///
/// This compares the total principal-value result with the sum of its two
/// computed components.
fn decomposition_residual(analytic_part: f64, regular_part: f64, total: f64) -> f64 {
    (analytic_part + regular_part - total).abs()
}

/// Estimate the behavior of the regularized integrand near s = t.
///
/// The values on the left and right of t should approach the analytic
/// limiting value -f'(t).
fn local_cancellation_diagnostic(t: f64) -> (f64, f64, f64) {
    let delta = 1.0e-6;

    let left_value = regularized_integrand(t - delta, t);
    let center_value = regularized_integrand(t, t);
    let right_value = regularized_integrand(t + delta, t);

    (left_value, center_value, right_value)
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let t = 0.35;
    let panels = 400;

    let (analytic_part, regular_part, pv_value) =
        match principal_value_by_subtraction(a, b, t, panels) {
            Ok(result) => result,
            Err(message) => {
                eprintln!("Principal-value computation failed: {}", message);
                return;
            }
        };

    let residual = decomposition_residual(analytic_part, regular_part, pv_value);
    let (left_limit, center_limit, right_limit) = local_cancellation_diagnostic(t);

    println!("Singularity Subtraction for a Cauchy Principal-Value Integral");
    println!("============================================================");
    println!();
    println!("Model Integral");
    println!("--------------");
    println!("Integral type                = Cauchy principal value");
    println!("Interval [a, b]              = [{:.1}, {:.1}]", a, b);
    println!("Target point t               = {:.6}", t);
    println!("Function f(s)                = sin(pi s) + 0.25 s^2");
    println!("Singular kernel              = 1 / (t - s)");
    println!();
    println!("Subtraction Formula");
    println!("-------------------");
    println!("Analytic singular part       = f(t) ln |(t - a)/(b - t)|");
    println!("Regularized remainder        = integral_a^b (f(s)-f(t))/(t-s) ds");
    println!("Quadrature for remainder     = composite Simpson rule");
    println!("Simpson panels               = {}", panels);
    println!();
    println!("Cancellation Diagnostic Near s = t");
    println!("----------------------------------");
    println!("Regularized integrand at t-delta = {:.12}", left_limit);
    println!("Analytic limiting value at t     = {:.12}", center_limit);
    println!("Regularized integrand at t+delta = {:.12}", right_limit);
    println!();
    println!("Numerical Result");
    println!("----------------");
    println!("Analytic singular contribution   = {:.12}", analytic_part);
    println!("Regularized numerical remainder  = {:.12}", regular_part);
    println!("Principal-value result           = {:.12}", pv_value);
    println!("Decomposition residual           = {:.6e}", residual);
}
```

Program 19.4.2 demonstrates how a Cauchy principal-value integral can be evaluated without applying quadrature directly to the singular kernel. The singular part is handled analytically through the logarithmic term in equation (19.4.12), while the remaining integral is regularized by subtracting (f(t)) from the numerator. This transforms the original singular expression into a smooth integrand suitable for ordinary Simpson quadrature.

The cancellation diagnostic confirms the numerical effect of the subtraction. The values of the regularized integrand on both sides of $s=t$ approach the analytic limiting value inserted at the singular point. This shows that the apparent singularity in the quotient has been removed and that the resulting integrand behaves like an ordinary smooth function near the diagonal.

The decomposition residual provides a simple consistency check for the implementation. Since the final principal-value result is computed as the sum of the analytic singular contribution and the regularized numerical remainder, a zero or near-zero residual confirms that the reported components are internally consistent. The result therefore reflects the regularized form of equation (19.4.13), not an unstable direct approximation of the singular integral.

This implementation is an appropriate bridge between weak-singularity quadrature and more advanced singular-kernel discretizations. It shows how analytic information about the singularity can replace undefined diagonal behavior by a finite, computable contribution. The same principle underlies many boundary element and singular integral methods, where off-diagonal terms are treated by quadrature while diagonal or near-diagonal singular behavior is handled by subtraction, limiting values, or special analytic weights.

## 19.4.4. Unified Quadrature and Computational Considerations

Classical boundary element implementations often use several different quadrature rules: ordinary Gauss quadrature for regular integrals, logarithmic quadrature for weak logarithmic singularities, product integration for algebraic singularities, and principal-value or finite-part regularization for strong and hypersingular kernels. Although effective, this approach complicates implementation because each singularity type requires its own nodes, weights, and local treatment.

A more unified approach is provided by Clenshaw-Curtis quadrature based on Chebyshev points. Recent work by Chen and Li (2023) shows that Clenshaw-Curtis quadrature can be adapted to regular, weakly singular, strongly singular, and hypersingular integrals within a single framework. Their method uses the same Chebyshev integration nodes and computes different weight sets for different singularity classes. The weights are tabulated over the reference interval $[-1,1]$, allowing the same grid to be used for multiple types of kernels. This simplifies implementation in boundary integral equation solvers because the integration nodes remain fixed while only the associated weights change (Chen and Li, 2023).

The algorithmic complexity of singular-kernel methods is usually comparable to that of nonsingular integral equations. A discretization with $N$ unknowns still produces a dense linear system in typical boundary integral formulations. If solved directly, the cost remains $O(N^3)$. If iterative methods are used, each dense matrix-vector product has cost $O(N^2),$ unless accelerated by fast multipole or hierarchical matrix techniques. With such acceleration, the cost may be reduced substantially, depending on the kernel structure and accuracy requirements.

The extra cost introduced by the singularity lies mainly in computing special weights, applying analytic subtraction, or evaluating limiting diagonal terms. In adaptive or panel-based boundary integral codes, computing singular weights on each panel adds only a constant overhead per panel. Consequently, the overall complexity is still dominated by matrix assembly and solution rather than by the singular quadrature itself.

The choice of method depends on the problem. For smooth nonsingular parts of the kernel, Gauss quadrature may be highly efficient. For logarithmic, Cauchy, or hypersingular kernels, special weights or analytic cancellation are essential. Unified Clenshaw-Curtis methods are attractive for general-purpose boundary integral solvers because they reduce the need to switch between multiple quadrature grids. However, problem-specific quadrature may still be preferable when the singular structure is known and very high accuracy is required.

### Rust Implementation

Following the discussion in Section 19.4.4 on unified quadrature and computational considerations, Program 19.4.3 provides a compact implementation of a Chebyshev-node quadrature framework for regular and singular integrals. The program reflects the central idea of the section: instead of changing quadrature nodes for each type of singularity, one may keep a single reference grid and modify only the associated weights. Using Chebyshev-Lobatto nodes on $[-1,1]$, the implementation constructs separate moment-fitted weights for a regular integral, a logarithmic weakly singular integral, and a Cauchy principal-value integral. This demonstrates how a unified node set can support different singularity classes while simplifying the organization of singular-kernel quadrature in boundary integral computations.

At the core of the implementation is the reuse of a single Chebyshev-Lobatto node set for several different integral types. The function `chebyshev_lobatto_nodes` constructs the nodes on the reference interval $[-1,1]$, ordered from left to right. These nodes remain fixed throughout the program. This reflects the unified quadrature viewpoint discussed in Section 19.4.4, where the integration grid is not changed when moving from regular to singular kernels.

The function `test_function(x)` defines the smooth function to be integrated against different weights. A low-degree polynomial is chosen deliberately because the moment-fitted quadrature rules can reproduce its integral values exactly when enough moment conditions are imposed. This makes the program suitable for verification: any observed error should be near roundoff level rather than approximation error from the test function.

The functions `regular_moment`, `logarithmic_moment`, and `cauchy_principal_value_moment` define the exact polynomial moments for three different integral classes. The regular moments correspond to ordinary integration over $[-1,1]$. The logarithmic moments correspond to integration against $\ln |x|$, representing a weak logarithmic singularity. The Cauchy moments correspond to a principal-value integral at the origin. These functions encode the singularity class into the quadrature construction.

The function `moment_fitted_weights` is the main quadrature-weight constructor. It builds a Vandermonde-type moment system whose unknowns are the quadrature weights. The weights are chosen so that the quadrature rule integrates monomials exactly up to the degree supported by the node set. The important point is that the same nodes are used in each case, while the right-hand side moments change depending on whether the integral is regular, logarithmic, or principal-value.

The function `solve_linear_system` solves the dense moment system using Gaussian elimination with partial pivoting. In this program, it is used only for a small pedagogical system. For production Clenshaw-Curtis implementations, the weights would normally be generated by more stable recurrence formulas, cosine-transform methods, or precomputed tabulations. Here, the explicit linear-system construction is useful because it clearly shows how different singularity classes lead to different weights on the same nodes.

The function `apply_quadrature` evaluates the quadrature rule by summing the products of the weights and the sampled test function values. The functions `exact_regular_integral`, `exact_logarithmic_integral`, and `exact_cauchy_principal_value_integral` compute the exact values for the selected test polynomial by combining the appropriate moments. These exact values provide benchmarks for validating all three quadrature rules.

The function `print_weight_summary` reports basic diagnostics for each weight set, including the sum of absolute weights and the range of weights. These diagnostics help distinguish ordinary positive quadrature weights from singular or principal-value weights, which may be negative or sign-changing. The function `max_abs_difference` is used as a node-reuse diagnostic, confirming that the same node array is used throughout the computation.

The `main` function brings the demonstration together. It constructs the Chebyshev-Lobatto nodes, computes the three different sets of moment-fitted weights, applies the corresponding quadrature rules to the same test function, and compares the results with exact moment-based values. It then prints the nodes, weight diagnostics, quadrature results, and node-reuse diagnostic. The output shows that all three integral types are handled accurately while using a single fixed grid.

```rust
// Program 19.4.3: Unified Chebyshev-Node Quadrature for Regular,
// Logarithmic, and Principal-Value Integrals
//
// Problem statement:
// Demonstrate a unified quadrature framework on the reference interval [-1, 1]
// using a single set of Chebyshev-Lobatto nodes.
//
// The same nodes are used to approximate three different integral types:
//
// 1. Regular integral:
//
//        integral_{-1}^{1} f(x) dx.
//
// 2. Weak logarithmic singular integral:
//
//        integral_{-1}^{1} f(x) ln|x| dx.
//
// 3. Cauchy principal-value integral:
//
//        P.V. integral_{-1}^{1} f(x) / (0 - x) dx.
//
// The program constructs separate quadrature weights for each integral type
// by matching polynomial moments on the same Chebyshev nodes. This illustrates
// the unified idea: the grid remains fixed, while the weights encode the
// singularity class.

/// Smooth test function.
///
/// A polynomial is chosen so that the moment-fitted quadrature rules can
/// reproduce the exact values up to roundoff when the degree is within the
/// supported polynomial space.
fn test_function(x: f64) -> f64 {
    1.0 + 0.5 * x - 0.25 * x * x + 0.125 * x * x * x
}

/// Generate Chebyshev-Lobatto nodes on [-1, 1].
///
/// The nodes are returned in increasing order:
///
///     -1 = x_0 < x_1 < ... < x_n = 1.
fn chebyshev_lobatto_nodes(n: usize) -> Vec<f64> {
    if n < 2 {
        panic!("At least two nodes are required.");
    }

    let pi = std::f64::consts::PI;

    (0..n)
        .map(|j| {
            let theta = pi * (n - 1 - j) as f64 / (n - 1) as f64;
            theta.cos()
        })
        .collect()
}

/// Exact polynomial moment for the regular integral:
///
///     integral_{-1}^{1} x^k dx.
fn regular_moment(k: usize) -> f64 {
    if k % 2 == 0 {
        2.0 / (k as f64 + 1.0)
    } else {
        0.0
    }
}

/// Exact polynomial moment for the logarithmic integral:
///
///     integral_{-1}^{1} x^k ln|x| dx.
///
/// For odd k this is zero by symmetry. For even k,
///
///     integral_{-1}^{1} x^k ln|x| dx = -2 / (k + 1)^2.
fn logarithmic_moment(k: usize) -> f64 {
    if k % 2 == 0 {
        -2.0 / ((k as f64 + 1.0) * (k as f64 + 1.0))
    } else {
        0.0
    }
}

/// Exact polynomial moment for the Cauchy principal-value integral
///
///     P.V. integral_{-1}^{1} x^k / (0 - x) dx.
///
/// For k = 0, the integrand is -1/x and the principal value is zero.
/// For k >= 1, the integral becomes
///
///     - integral_{-1}^{1} x^{k-1} dx.
fn cauchy_principal_value_moment(k: usize) -> f64 {
    if k == 0 {
        0.0
    } else if (k - 1) % 2 == 0 {
        -2.0 / k as f64
    } else {
        0.0
    }
}

/// Solve a dense linear system using Gaussian elimination with partial pivoting.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("The matrix must be square and compatible with the right-hand side.".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            let candidate = a[i][k].abs();

            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!(
                "Moment system is numerically singular near column {}.",
                k
            ));
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

/// Construct quadrature weights on a fixed node set by moment matching.
///
/// The weights w_j are chosen so that
///
///     sum_j w_j x_j^k = moment_k,
///
/// for k = 0, ..., n - 1.
fn moment_fitted_weights<F>(nodes: &[f64], moment: F) -> Result<Vec<f64>, String>
where
    F: Fn(usize) -> f64,
{
    let n = nodes.len();

    let mut vandermonde = vec![vec![0.0; n]; n];
    let mut moments = vec![0.0; n];

    for k in 0..n {
        moments[k] = moment(k);

        for j in 0..n {
            vandermonde[k][j] = nodes[j].powi(k as i32);
        }
    }

    solve_linear_system(vandermonde, moments)
}

/// Apply a quadrature rule to the test function.
fn apply_quadrature(nodes: &[f64], weights: &[f64]) -> f64 {
    nodes
        .iter()
        .zip(weights.iter())
        .map(|(&x, &w)| w * test_function(x))
        .sum()
}

/// Exact value of the regular integral for the test polynomial.
fn exact_regular_integral() -> f64 {
    regular_moment(0) + 0.5 * regular_moment(1)
        - 0.25 * regular_moment(2)
        + 0.125 * regular_moment(3)
}

/// Exact value of the logarithmic integral for the test polynomial.
fn exact_logarithmic_integral() -> f64 {
    logarithmic_moment(0) + 0.5 * logarithmic_moment(1)
        - 0.25 * logarithmic_moment(2)
        + 0.125 * logarithmic_moment(3)
}

/// Exact value of the Cauchy principal-value integral for the test polynomial.
fn exact_cauchy_principal_value_integral() -> f64 {
    cauchy_principal_value_moment(0) + 0.5 * cauchy_principal_value_moment(1)
        - 0.25 * cauchy_principal_value_moment(2)
        + 0.125 * cauchy_principal_value_moment(3)
}

/// Compute the maximum absolute difference between two vectors.
fn max_abs_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Print one set of quadrature weights.
fn print_weight_summary(name: &str, weights: &[f64]) {
    let sum_abs: f64 = weights.iter().map(|w| w.abs()).sum();
    let min_weight = weights.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_weight = weights.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("{}", name);
    println!("  sum of absolute weights    = {:.6e}", sum_abs);
    println!("  minimum weight             = {:.6e}", min_weight);
    println!("  maximum weight             = {:.6e}", max_weight);
}

fn main() {
    let n = 13;

    let nodes = chebyshev_lobatto_nodes(n);

    let regular_weights = match moment_fitted_weights(&nodes, regular_moment) {
        Ok(weights) => weights,
        Err(message) => {
            eprintln!("Regular weight construction failed: {}", message);
            return;
        }
    };

    let logarithmic_weights = match moment_fitted_weights(&nodes, logarithmic_moment) {
        Ok(weights) => weights,
        Err(message) => {
            eprintln!("Logarithmic weight construction failed: {}", message);
            return;
        }
    };

    let cauchy_weights = match moment_fitted_weights(&nodes, cauchy_principal_value_moment) {
        Ok(weights) => weights,
        Err(message) => {
            eprintln!("Cauchy principal-value weight construction failed: {}", message);
            return;
        }
    };

    let regular_approx = apply_quadrature(&nodes, &regular_weights);
    let log_approx = apply_quadrature(&nodes, &logarithmic_weights);
    let cauchy_approx = apply_quadrature(&nodes, &cauchy_weights);

    let regular_exact = exact_regular_integral();
    let log_exact = exact_logarithmic_integral();
    let cauchy_exact = exact_cauchy_principal_value_integral();

    println!("Unified Chebyshev-Node Quadrature for Singular Integrals");
    println!("========================================================");
    println!();
    println!("Quadrature Grid");
    println!("---------------");
    println!("Reference interval          = [-1, 1]");
    println!("Node type                   = Chebyshev-Lobatto");
    println!("Number of nodes             = {}", n);
    println!("Same nodes reused           = yes");
    println!("Different weights used      = regular, logarithmic, principal value");
    println!();

    println!("Chebyshev Nodes");
    println!("---------------");
    println!("{:>8} {:>18}", "j", "x_j");

    for (j, &x) in nodes.iter().enumerate() {
        println!("{:>8} {:>18.12}", j, x);
    }

    println!();
    println!("Weight Diagnostics");
    println!("------------------");
    print_weight_summary("Regular weights", &regular_weights);
    print_weight_summary("Logarithmic weights", &logarithmic_weights);
    print_weight_summary("Principal-value weights", &cauchy_weights);
    println!();

    println!("Test Function");
    println!("-------------");
    println!("f(x) = 1 + 0.5x - 0.25x^2 + 0.125x^3");
    println!();

    println!("Quadrature Results");
    println!("------------------");
    println!(
        "{:>28} {:>18} {:>18} {:>18}",
        "Integral type", "approximation", "exact", "abs_error"
    );

    println!(
        "{:>28} {:>18.12} {:>18.12} {:>18.6e}",
        "regular",
        regular_approx,
        regular_exact,
        (regular_approx - regular_exact).abs()
    );

    println!(
        "{:>28} {:>18.12} {:>18.12} {:>18.6e}",
        "logarithmic ln|x|",
        log_approx,
        log_exact,
        (log_approx - log_exact).abs()
    );

    println!(
        "{:>28} {:>18.12} {:>18.12} {:>18.6e}",
        "Cauchy P.V. at 0",
        cauchy_approx,
        cauchy_exact,
        (cauchy_approx - cauchy_exact).abs()
    );

    println!();
    println!("Node-Reuse Diagnostic");
    println!("---------------------");
    println!(
        "max difference between node arrays = {:.6e}",
        max_abs_difference(&nodes, &nodes)
    );
}
```

Program 19.4.3 demonstrates the practical meaning of a unified quadrature framework for singular-kernel integration. The same Chebyshev-Lobatto nodes are used for all three examples, while the quadrature weights change according to the integral type. This mirrors the idea discussed in Section 19.4.4: implementation can be simplified by fixing the interpolation or integration nodes and encoding regular, logarithmic, or principal-value behavior into specialized weights.

The numerical output confirms the construction. Since the test function is a polynomial of degree lower than the number of imposed moment conditions, the regular, logarithmic, and principal-value integrals are all recovered to roundoff accuracy. The node-reuse diagnostic further confirms that the method does not switch grids between integral classes. Instead, the same function samples can be reused with different weight vectors.

The weight diagnostics also illustrate an important numerical distinction. Regular quadrature weights behave like ordinary integration weights, while logarithmic and principal-value weights may be negative or sign-changing because they encode singular or signed cancellation effects. This is expected and is part of the analytic regularization built into the quadrature rule.

Although the program uses moment matching through a dense Vandermonde system for clarity, the same idea underlies more efficient Clenshaw-Curtis implementations based on cosine transforms, recurrence relations, or tabulated weights. In boundary integral equation solvers, this organization is valuable because the dominant cost often lies in matrix assembly and solution, while the extra singular-quadrature work is mainly the construction or selection of appropriate weights. Thus, the program provides a compact computational model of the unified quadrature strategy described in Section 19.4.4.

## 19.4.5. Applications of Singular-Kernel Integral Equations

Singular kernels are ubiquitous in physical modeling. In acoustics, the Helmholtz boundary integral equation involves kernels derived from the free-space Green’s function. A typical kernel contains normal derivatives of the Green’s function,

$$K(t,s)=\frac{\partial}{\partial n_s}G(t,s) \tag{19.4.17}$$

where $G$ is the free-space Green’s function and $n_s$ denotes the normal direction at the source point. Such kernels generate weakly singular and hypersingular integrals when evaluating sound pressure, fluxes, or boundary tractions. Accurate treatment of these singularities is essential for reliable acoustic scattering simulations.

In fracture mechanics, crack problems lead naturally to hypersingular integral equations. The stress intensity factor near a crack tip depends sensitively on the singular behavior of the stress field. Hypersingular kernels arise when computing stresses or tractions along crack surfaces, and their correct interpretation is necessary for accurate prediction of crack propagation and failure.

Singular kernels also occur in viscous fluid dynamics. Boundary integral formulations for Stokes flow around particles involve kernels with singularities such as,

$$\frac{1}{r}\qquad \text{and} \qquad\frac{1}{r^3} \tag{19.4.18}$$

In these problems, one typically subtracts the singular part, evaluates its known analytic contribution, and applies smooth quadrature to the remaining regular integral. This makes it possible to simulate viscous flows around particles, droplets, or suspended bodies with high accuracy.

Chen and Li (2023) demonstrate the effectiveness of the Clenshaw-Curtis framework on a two-dimensional Helmholtz problem, showing that regular, weakly singular, strongly singular, and hypersingular integrals can be treated accurately within one quadrature structure. Overall, efficient treatment of singular kernels is indispensable in boundary integral equations and appears across acoustics, electromagnetics, fracture mechanics, and fluid dynamics (Chen and Li, 2023).

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/gLZOy1Ppa6XrN5NoxmxA.3","tags":[]}

# 19.5. Inverse Problems and the Use of A Priori Information

Inverse problems arise when the unknown quantity must be recovered from indirect, incomplete, or noisy measurements. In the setting of integral equations, this usually means reconstructing an unknown function from data produced by a smoothing integral operator, a process that is often unstable without additional assumptions. This section explains why such problems are typically ill-posed, how a priori information stabilizes the reconstruction, and how smoothness, positivity, boundedness, parametric structure, Bayesian priors, and learned regularizers can be incorporated into a regularized optimization framework.

## 19.5.1. Formulation and Ill-Posedness of Inverse Problems

An *inverse problem* seeks to recover an unknown function, parameter, or physical quantity from indirect measurements. In the setting of integral equations, the measured data are often related to the unknown through an integral operator. A typical model has the form:

$$g(t)=\int_a^b K(t,s)u(s)\,ds \tag{19.5.1}$$

where $g(t)$ is the observed data, $K(t,s)$ is a known kernel, and $u(s)$ is the unknown quantity to be recovered. In operator notation, this may be written as:

$$Ku=g \tag{19.5.2}$$

Here $K$ denotes the integral operator induced by the kernel. Unlike the forward problem, where $u$ is known and $g$ is computed, the inverse problem attempts to reconstruct $u$ from $g$. This reversal is typically difficult because integral operators often smooth their inputs. Fine-scale features in $u$ may be strongly attenuated in $g$, and therefore small perturbations in the data can cause large changes in the reconstructed solution.

This instability is the central difficulty of inverse problems. If the data are noisy, meaning that the available measurement is:

$$g^\delta = g+\eta \tag{19.5.3}$$

where $\eta$ denotes measurement error, then directly solving,

$$Ku=g^\delta \tag{19.5.4}$$

may produce a solution dominated by noise rather than by the true signal. After discretization, one obtains a linear algebraic system:

$$\mathbf K\mathbf u=\mathbf g \tag{19.5.5}$$

but if $\mathbf K$ is ill-conditioned or nearly rank deficient, direct inversion becomes unstable. This is why inverse problems based on first-kind Fredholm equations usually require additional information beyond the measured data (Nature Research Intelligence, 2025).

The mathematical issue is not simply that the system may be large. Rather, the inverse map from data to solution is unstable. Modes of $u$ associated with small singular values of $K$ are weakly represented in $g$. When inversion attempts to recover these modes, noise is amplified. Thus, a meaningful reconstruction requires *a priori information*, meaning information known or assumed about the solution before solving the inverse problem.

### Rust Implementation

Following the discussion in Section 19.5.1 on the formulation and ill-posedness of inverse problems, Program 19.5.1 provides a practical demonstration of noise amplification in a discretized first-kind Fredholm integral equation. The program follows the forward model in equations (19.5.1) and (19.5.2), where an unknown function is mapped to measured data through a smoothing integral operator. It then introduces a small deterministic perturbation in the data, corresponding to the noisy-data model in equation (19.5.3), and attempts a direct reconstruction by solving the discretized system in equation (19.5.5). The output shows that the clean-data reconstruction is accurate, while the noisy-data reconstruction becomes dominated by large oscillatory errors. This illustrates the central point of Section 19.5.1: the instability is not caused by failure of the linear solver, but by the ill-posedness of the inverse map from data back to the unknown function.

At the core of the implementation is the discretization of the first-kind inverse problem described by equations (19.5.1) and (19.5.2). The function `exact_solution(s)` defines the unknown function $u(s)$ to be recovered. It contains both a smooth component and a higher-frequency component so that the smoothing effect of the integral operator becomes visible. The function `kernel(t, s, sigma)` defines a Gaussian smoothing kernel, whose width is controlled by `sigma`. A smaller or moderate value of `sigma` produces a smoothing operator that attenuates fine-scale features of the unknown.

The functions `uniform_grid` and `trapezoidal_weights` define the grid and quadrature weights used to approximate the integral operator. The grid represents both the source and observation points, while the trapezoidal weights approximate the integral over the interval. These components turn the continuous operator equation into the finite-dimensional system represented by equation (19.5.5).

The function `assemble_smoothing_matrix` constructs the discretized forward operator. Each matrix entry combines the kernel value with the corresponding quadrature weight, so applying the matrix to a vector of unknown values approximates the integral in equation (19.5.1). This matrix acts as a smoothing operator: high-frequency components in the unknown function are weakly represented in the data, which is the source of instability in the inverse problem.

The function `mat_vec` applies the discretized forward operator to a vector. In the program, it is used first to generate clean synthetic data from the exact unknown and later to compute residuals for the reconstructed solutions. The function `add_deterministic_noise` perturbs the clean data by adding a small reproducible oscillatory error. This corresponds to the noisy measurement model in equation (19.5.3). A deterministic perturbation is used rather than random noise so that the output is reproducible.

The function `solve_linear_system` performs Gaussian elimination with partial pivoting. It is used intentionally as a direct inversion method for the discretized first-kind system. This is important pedagogically because the solver itself is not the source of the failure. When the data are clean, the reconstruction is accurate. When the data contain even a small perturbation, the same direct solve amplifies the noise dramatically because the inverse problem is ill-conditioned.

The functions `infinity_norm`, `infinity_error`, and `relative_infinity_error` compute diagnostic quantities used to compare data and reconstructions. The relative data noise measures how small the perturbation is in the observed data, while the reconstruction errors measure how far the recovered solutions are from the exact unknown. Their ratio gives the noise amplification factor, which quantifies the instability of the inverse map.

The function `pivot_growth_indicator` provides a simple conditioning diagnostic based on the pivot magnitudes encountered during elimination. It is not a full condition number, but it gives a useful indication that the discretized smoothing matrix is numerically delicate. A large pivot indicator supports the interpretation that the direct inverse is sensitive to perturbations in the right-hand side.

The `main` function assembles the complete inverse-problem experiment. It defines the interval, grid size, kernel width, and noise level, constructs the smoothing matrix, samples the exact unknown, generates clean and noisy data, and solves the inverse problem twice. It then reports the relative data noise, pivot diagnostic, clean and noisy reconstruction errors, noise amplification factor, and residuals. The representative values printed at selected grid points show visually how the noisy reconstruction may fit the perturbed data while bearing little resemblance to the true unknown.

```rust
// Program 19.5.1: Noise Amplification in a Discretized First-Kind Integral Equation
//
// Problem statement:
// Demonstrate the ill-posedness of a first-kind integral equation
//
//     g(t) = integral_a^b K(t, s) u(s) ds,
//
// after discretization. The forward operator is smoothing, so direct
// reconstruction of u from noisy data can amplify measurement error.
//
// The program constructs a smoothing matrix K, generates exact data
//
//     g = K u_exact,
//
// adds a small deterministic noise perturbation,
//
//     g_delta = g + eta,
//
// and compares the reconstructed solutions obtained from clean and noisy data.
// This illustrates why inverse problems usually require a priori information
// or regularization.

use std::f64::consts::PI;

/// Exact unknown function u(s).
///
/// This function contains both a smooth component and a higher-frequency
/// component. The smoothing operator will strongly attenuate the higher
/// frequency part.
fn exact_solution(s: f64) -> f64 {
    (PI * s).sin() + 0.35 * (6.0 * PI * s).sin()
}

/// Smooth Gaussian kernel K(t, s).
///
/// A narrow Gaussian-like kernel is used to model a smoothing integral
/// operator. Such operators damp fine-scale features of u and make the
/// inverse problem unstable.
fn kernel(t: f64, s: f64, sigma: f64) -> f64 {
    let distance = t - s;
    (-distance * distance / (2.0 * sigma * sigma)).exp()
}

/// Build a uniform grid on [a, b].
fn uniform_grid(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n < 2 {
        panic!("At least two grid points are required.");
    }

    let h = (b - a) / (n as f64 - 1.0);

    (0..n).map(|i| a + i as f64 * h).collect()
}

/// Composite trapezoidal weights on [a, b].
fn trapezoidal_weights(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);
    let mut weights = vec![h; n];

    weights[0] = 0.5 * h;
    weights[n - 1] = 0.5 * h;

    weights
}

/// Assemble the discretized smoothing matrix.
///
/// The entries include quadrature weights:
///
///     K_ij = K(t_i, s_j) w_j.
///
/// This corresponds to a quadrature discretization of the integral operator.
fn assemble_smoothing_matrix(
    grid: &[f64],
    weights: &[f64],
    sigma: f64,
) -> Vec<Vec<f64>> {
    let n = grid.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        let t_i = grid[i];

        for j in 0..n {
            let s_j = grid[j];
            matrix[i][j] = kernel(t_i, s_j, sigma) * weights[j];
        }
    }

    matrix
}

/// Matrix-vector product y = A x.
fn mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&a, &xj)| a * xj).sum())
        .collect()
}

/// Add a small deterministic measurement perturbation to the data.
///
/// A deterministic oscillatory perturbation is used instead of randomness
/// so that the program output is reproducible.
fn add_deterministic_noise(data: &[f64], relative_level: f64) -> Vec<f64> {
    let data_norm = infinity_norm(data);
    let n = data.len();

    data.iter()
        .enumerate()
        .map(|(i, &value)| {
            let phase = 2.0 * PI * i as f64 / (n as f64 - 1.0);
            let noise = relative_level * data_norm * (11.0 * phase).sin();
            value + noise
        })
        .collect()
}

/// Solve a dense linear system using Gaussian elimination with partial pivoting.
///
/// This intentionally performs direct inversion of the discretized first-kind
/// system. The method is algebraically valid, but the result can be highly
/// unstable when the matrix is ill-conditioned.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("The system matrix must be square and compatible with b.".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            let candidate = a[i][k].abs();

            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!(
                "Matrix is numerically singular near column {}.",
                k
            ));
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

/// Infinity norm of a vector.
fn infinity_norm(x: &[f64]) -> f64 {
    x.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

/// Infinity norm of the difference between two vectors.
fn infinity_error(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Relative infinity-norm error.
fn relative_infinity_error(x: &[f64], y: &[f64]) -> f64 {
    let denominator = infinity_norm(y).max(1.0e-14);
    infinity_error(x, y) / denominator
}

/// Estimate a simple conditioning indicator using column scaling information.
///
/// This is not a full condition number computation. It reports the ratio
/// between the largest and smallest pivot magnitudes encountered during
/// Gaussian elimination, which gives a useful diagnostic for instability.
fn pivot_growth_indicator(mut a: Vec<Vec<f64>>) -> Result<f64, String> {
    let n = a.len();
    let mut largest_pivot: f64 = 0.0;
    let mut smallest_pivot: f64 = f64::INFINITY;

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            let candidate = a[i][k].abs();

            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!(
                "Matrix is numerically singular near column {}.",
                k
            ));
        }

        largest_pivot = largest_pivot.max(pivot_abs);
        smallest_pivot = smallest_pivot.min(pivot_abs);

        if pivot_row != k {
            a.swap(k, pivot_row);
        }

        for i in (k + 1)..n {
            let factor = a[i][k] / a[k][k];

            a[i][k] = 0.0;

            for j in (k + 1)..n {
                a[i][j] -= factor * a[k][j];
            }
        }
    }

    Ok(largest_pivot / smallest_pivot)
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let n = 25;
    let sigma = 0.08;
    let noise_level = 1.0e-4;

    let grid = uniform_grid(a, b, n);
    let weights = trapezoidal_weights(a, b, n);

    let smoothing_matrix = assemble_smoothing_matrix(&grid, &weights, sigma);

    let exact: Vec<f64> = grid.iter().map(|&s| exact_solution(s)).collect();

    let clean_data = mat_vec(&smoothing_matrix, &exact);
    let noisy_data = add_deterministic_noise(&clean_data, noise_level);

    let clean_reconstruction =
        match solve_linear_system(smoothing_matrix.clone(), clean_data.clone()) {
            Ok(solution) => solution,
            Err(message) => {
                eprintln!("Clean-data reconstruction failed: {}", message);
                return;
            }
        };

    let noisy_reconstruction =
        match solve_linear_system(smoothing_matrix.clone(), noisy_data.clone()) {
            Ok(solution) => solution,
            Err(message) => {
                eprintln!("Noisy-data reconstruction failed: {}", message);
                return;
            }
        };

    let data_relative_noise = relative_infinity_error(&noisy_data, &clean_data);
    let clean_reconstruction_error = relative_infinity_error(&clean_reconstruction, &exact);
    let noisy_reconstruction_error = relative_infinity_error(&noisy_reconstruction, &exact);
    let noise_amplification =
        noisy_reconstruction_error / data_relative_noise.max(1.0e-14);

    let residual_clean =
        relative_infinity_error(&mat_vec(&smoothing_matrix, &clean_reconstruction), &clean_data);

    let residual_noisy =
        relative_infinity_error(&mat_vec(&smoothing_matrix, &noisy_reconstruction), &noisy_data);

    let pivot_indicator = match pivot_growth_indicator(smoothing_matrix.clone()) {
        Ok(value) => value,
        Err(message) => {
            eprintln!("Pivot diagnostic failed: {}", message);
            return;
        }
    };

    println!("Ill-Conditioning and Noise Amplification in an Inverse Problem");
    println!("=============================================================");
    println!();
    println!("Forward Model");
    println!("-------------");
    println!("Integral equation           = first-kind Fredholm");
    println!("Discretized system          = K u = g");
    println!("Interval                    = [{:.1}, {:.1}]", a, b);
    println!("Grid points N               = {}", n);
    println!("Kernel                      = Gaussian smoothing kernel");
    println!("Kernel width sigma          = {:.6}", sigma);
    println!("Exact unknown u(s)          = sin(pi s) + 0.35 sin(6 pi s)");
    println!();
    println!("Noise and Conditioning Diagnostics");
    println!("----------------------------------");
    println!("Relative data noise level   = {:.6e}", data_relative_noise);
    println!("Pivot growth indicator      = {:.6e}", pivot_indicator);
    println!("Clean reconstruction error  = {:.6e}", clean_reconstruction_error);
    println!("Noisy reconstruction error  = {:.6e}", noisy_reconstruction_error);
    println!("Noise amplification factor  = {:.6e}", noise_amplification);
    println!("Clean-data residual         = {:.6e}", residual_clean);
    println!("Noisy-data residual         = {:.6e}", residual_noisy);
    println!();
    println!("Representative Values");
    println!("---------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "i", "s_i", "u_exact", "u_clean", "u_noisy"
    );

    for i in 0..n {
        if i % 3 == 0 || i == n - 1 {
            println!(
                "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.10}",
                i,
                grid[i],
                exact[i],
                clean_reconstruction[i],
                noisy_reconstruction[i]
            );
        }
    }
}
```

Program 19.5.1 demonstrates the fundamental instability of first-kind inverse problems. The clean reconstruction is accurate because the synthetic data were generated exactly from the discretized operator and no perturbation was present. However, once a small perturbation is added to the data, direct inversion produces a reconstruction with very large oscillatory errors. This confirms the point made in Section 19.5.1: the inverse problem is unstable because the smoothing operator suppresses information that direct inversion later attempts to recover.

The residual diagnostics are especially instructive. The noisy reconstruction has a very small residual with respect to the noisy data, meaning that it solves the perturbed algebraic system accurately. Nevertheless, it is a poor approximation of the true unknown. This distinction is central to inverse problems: fitting the measured data exactly is not the same as recovering a physically meaningful solution when the data contain noise.

The noise amplification factor makes the ill-posedness quantitative. A small relative perturbation in the data can be magnified by many orders of magnitude in the reconstructed solution. This behavior explains why the direct system in equation (19.5.5) should not usually be solved without additional assumptions when the data are noisy or incomplete.

This example prepares the ground for the regularization methods introduced in the following subsections. A priori information such as smoothness, positivity, boundedness, or parametric structure is needed to suppress unstable modes and prevent the reconstruction from fitting noise. In this way, Program 19.5.1 provides the numerical motivation for the regularized optimization formulations that follow.

## 19.5.2. Forms of A Priori Information

A priori information restricts the class of admissible solutions and prevents the inverse problem from fitting arbitrary noise. The prior information may come from physics, geometry, measurement constraints, previous experiments, or qualitative knowledge about the unknown. Several common forms are especially important.

The first is *smoothness*. In many physical systems, the unknown function $u(s)$ is expected to vary gradually. This may be expressed by requiring its derivative or curvature to remain small. For example, one may impose that $\|u'\|$ or $\|u''\|$ is bounded. In numerical form, this leads to penalties on discrete gradients or discrete Laplacians. Smoothness constraints suppress rapidly oscillatory reconstructions, which are often artifacts of noise rather than physically meaningful features.

The second form is *boundedness *or *positivity*. In many applications, the unknown represents a physical density, concentration, absorption coefficient, conductivity, or intensity. Such quantities may be known to satisfy:

$$u(s)\ge 0, \tag{19.5.6}$$

or to lie within a prescribed range,

$$u_{\min}\le u(s)\le u_{\max} \tag{19.5.7}$$

These constraints prevent nonphysical reconstructions. They may be enforced through projection methods, constrained optimization, barrier methods, or parametrizations that automatically preserve positivity.

The third form is *parametric structure*. Sometimes the unknown is assumed to belong to a finite-dimensional model family. For example,

$$u(s)=\sum_{k=1}^m a_k\phi_k(s) \tag{19.5.8}$$

where the basis functions $\phi_k$ are chosen from physical insight, geometry, wavelet representations, polynomial approximations, or known material models. In this case, the inverse problem becomes the estimation of finitely many coefficients $a_k$, rather than the recovery of an arbitrary function. This reduces degrees of freedom and improves stability.

A priori information is therefore not an optional addition to an inverse problem. It is what converts an unstable recovery problem into a meaningful reconstruction problem. The art of inverse modeling lies in choosing prior assumptions that are strong enough to stabilize the problem but not so restrictive that they bias the solution away from the true unknown.

## 19.5.3. Regularized Optimization Formulations

In practice, a priori information is usually introduced through an optimization problem. One may seek a solution that fits the data while satisfying a constraint on a prior functional. A constrained formulation is:

$$\min_u \|Ku-g\|^2\quad \text{subject to} \quad R(u)\le C, \tag{19.5.9}$$

where $R(u)$ measures roughness, deviation from a prior model, lack of sparsity, or violation of expected structure. The constant $C$ controls how strongly the prior is imposed. An equivalent and widely used unconstrained form is:

$$\min_u \|Ku-g\|^2+\alpha R(u) \tag{19.5.10}$$

where $\alpha>0$ is the *regularization parameter*. The first term $\|Ku-g\|^2$, measures data fidelity, while the second term $\alpha R(u)$, penalizes solutions that violate the prior information. The parameter $\alpha$ balances these two objectives. If $\alpha$ is too small, the solution may overfit noise in the data. If $\alpha$ is too large, the solution may be overly dominated by the prior and fail to represent the measurements.

A classical example is smoothness regularization. If $L$ is a derivative or difference operator, one may choose:

$$R(u)=\|Lu\|^2 \tag{19.5.11}$$

Then (19.5.10) becomes,

$$\min_u \|Ku-g\|^2+\alpha\|Lu\|^2 \tag{19.5.12}$$

This formulation penalizes roughness. If $L=I$, the method penalizes large solution norms. If $L$ is a discrete gradient, it penalizes large first derivatives. If $L$ is a discrete Laplacian, it penalizes curvature. In each case, the regularizer encodes a different form of a priori information.

The same framework can express other priors. A positivity prior may be imposed by adding the constraint $u\ge 0$. A sparsity prior may be imposed through an $\ell_1$-type penalty. A piecewise-smooth prior may be represented using total variation. Thus, the abstract regularized problem (19.5.10) provides a general bridge between inverse theory and numerical optimization (Nature Research Intelligence, 2025).

### Rust Implementation

Following the discussion in Sections 19.5.2 and 19.5.3 on a priori information and regularized optimization, Program 19.5.2 provides a practical implementation of Tikhonov regularization with a smoothness prior for an ill-posed inverse problem. The program builds on the instability demonstrated in Program 19.5.1 by replacing direct inversion with the regularized formulation in equation (19.5.12). The prior information is encoded through a discrete first-difference operator, which penalizes rapidly oscillatory reconstructions and favors solutions that vary smoothly across the grid. By comparing a weakly stabilized least-squares baseline with a deliberately regularized reconstruction, the program shows how the parameter $\alpha$ balances data fidelity against the smoothness constraint. This example makes the role of a priori information concrete: the reconstruction is no longer determined by the noisy data alone, but by a controlled compromise between measurement fit and expected solution structure.

At the core of the implementation is the regularized optimization problem introduced in equation (19.5.12). The function `exact_solution(s)` defines the unknown function used to generate synthetic data. It is smooth and nonnegative, making it consistent with the kind of prior structure discussed in Section 19.5.2. The function `kernel(t, s, sigma)` defines the Gaussian smoothing kernel that maps the unknown function to indirect data. Since this kernel suppresses fine-scale variation, recovering the unknown from noisy data is unstable without regularization.

The functions `uniform_grid` and `trapezoidal_weights` construct the discretization of the integral operator. The grid provides the source and observation points, while the trapezoidal weights approximate the integral. Together, they convert the continuous inverse problem into a finite-dimensional linear system of the form discussed in equation (19.5.5).

The function `assemble_smoothing_matrix` constructs the weighted matrix approximation of the forward operator. Each entry combines the kernel value with the corresponding quadrature weight. Applying this matrix to the sampled exact solution produces synthetic data. The function `mat_vec` carries out this forward operation, while `transpose_mat_vec` and `gram_matrix` are used to build the normal equations associated with least-squares and Tikhonov reconstruction.

The function `first_difference_penalty` constructs the matrix $L^\top L$ associated with a first-difference smoothness operator. This is the discrete version of using $R(u)=\|Lu\|^2$, as introduced in equation (19.5.11). It penalizes large differences between neighboring solution values, thereby suppressing the oscillatory artifacts that commonly appear when noisy data are inverted directly.

The function `add_scaled_matrix` forms the regularized system matrix by adding the smoothness penalty to the least-squares normal matrix. This corresponds to the matrix form of equation (19.5.12). The function `add_deterministic_noise` adds a reproducible oscillatory perturbation to the synthetic data, representing the noisy measurement model described in equation (19.5.3). Using deterministic noise ensures that the program produces the same output each time it is run.

The function `solve_linear_system` solves the dense linear systems using Gaussian elimination with partial pivoting. It is used both for the weakly stabilized baseline and for the Tikhonov-regularized reconstruction. The baseline function `weakly_stabilized_least_squares` adds only a tiny diagonal ridge to avoid numerical singularity in the normal equations. This baseline is not intended to represent meaningful prior information; it is included to show how an almost unregularized reconstruction still behaves unstably.

The function `tikhonov_smoothness_reconstruction` implements the main regularized reconstruction. It forms $K^\top K$, constructs $L^\top L$, builds the regularized matrix $K^\top K+\alpha L^\top L$, and solves for the reconstructed solution. This is the computational realization of the smoothness-regularized optimization problem in equation (19.5.12).

The functions `infinity_norm`, `infinity_error`, and `relative_infinity_error` measure reconstruction accuracy relative to the exact solution. The function `data_misfit_squared` computes the squared data misfit, while `smoothness_penalty` computes the roughness penalty $\|Lu\|^2$. These two quantities correspond directly to the two competing terms in equation (19.5.10): fidelity to the data and adherence to prior information. The functions `min_value` and `max_value` report the range of the reconstructed solution, helping reveal whether the reconstruction contains nonphysical oscillations.

The `main` function assembles the full experiment. It defines the grid, kernel width, noise level, tiny baseline ridge, and regularization parameter $\alpha$. It then generates clean and noisy data, computes both the weak baseline and Tikhonov reconstructions, and reports diagnostic quantities. The printed output compares relative error, data misfit, smoothness penalty, regularized objective, minimum and maximum values, and representative nodal values. These diagnostics show that the weak baseline can fit the noisy data while producing a highly oscillatory and inaccurate solution, whereas the Tikhonov reconstruction sacrifices a small amount of data fit to obtain a much smoother and more meaningful result.

```rust
// Program 19.5.2: Tikhonov Regularization with a Smoothness Prior
// for an Inverse Problem
//
// Problem statement:
// Recover an unknown function u from noisy indirect data
//
//     K u = g_delta,
//
// where K is a smoothing discretized integral operator. Direct inversion of
// this first-kind system is unstable. The program therefore compares:
//
// 1. A weakly stabilized least-squares baseline,
// 2. A Tikhonov-regularized reconstruction with a smoothness prior.
//
// The Tikhonov reconstruction solves
//
//     min_u ||K u - g_delta||_2^2 + alpha ||L u||_2^2,
//
// where L is a discrete first-difference operator. This implements a
// smoothness prior: rapidly oscillatory reconstructions are penalized.
//
// The regularized normal equations are
//
//     (K^T K + alpha L^T L) u = K^T g_delta.

use std::f64::consts::PI;

/// Exact unknown function u(s).
///
/// This function is smooth and nonnegative, but it contains enough variation
/// to make the reconstruction problem meaningful.
fn exact_solution(s: f64) -> f64 {
    1.0 + 0.6 * (PI * s).sin() + 0.2 * (3.0 * PI * s).sin()
}

/// Gaussian smoothing kernel K(t, s).
///
/// The kernel represents a smoothing forward operator. Such operators damp
/// fine-scale features of u and make direct inversion unstable.
fn kernel(t: f64, s: f64, sigma: f64) -> f64 {
    let distance = t - s;
    (-distance * distance / (2.0 * sigma * sigma)).exp()
}

/// Build a uniform grid on [a, b].
fn uniform_grid(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n < 2 {
        panic!("At least two grid points are required.");
    }

    let h = (b - a) / (n as f64 - 1.0);

    (0..n).map(|i| a + i as f64 * h).collect()
}

/// Composite trapezoidal weights on [a, b].
fn trapezoidal_weights(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);
    let mut weights = vec![h; n];

    weights[0] = 0.5 * h;
    weights[n - 1] = 0.5 * h;

    weights
}

/// Assemble the weighted smoothing matrix.
///
/// The entries are
///
///     K_ij = K(t_i, s_j) w_j,
///
/// so that the matrix-vector product approximates the integral operator.
fn assemble_smoothing_matrix(
    grid: &[f64],
    weights: &[f64],
    sigma: f64,
) -> Vec<Vec<f64>> {
    let n = grid.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        let t_i = grid[i];

        for j in 0..n {
            let s_j = grid[j];
            matrix[i][j] = kernel(t_i, s_j, sigma) * weights[j];
        }
    }

    matrix
}

/// Matrix-vector product y = A x.
fn mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&a, &xj)| a * xj).sum())
        .collect()
}

/// Transposed matrix-vector product y = A^T x.
fn transpose_mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut result = vec![0.0; cols];

    for i in 0..rows {
        for j in 0..cols {
            result[j] += matrix[i][j] * x[i];
        }
    }

    result
}

/// Matrix-matrix product C = A^T A.
fn gram_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut gram = vec![vec![0.0; cols]; cols];

    for i in 0..cols {
        for j in 0..cols {
            let mut sum = 0.0;

            for k in 0..rows {
                sum += matrix[k][i] * matrix[k][j];
            }

            gram[i][j] = sum;
        }
    }

    gram
}

/// Build L^T L for the first-difference smoothness operator.
///
/// If L u contains u_{i+1} - u_i, then L^T L is the standard discrete
/// gradient penalty matrix. It penalizes rapid variation in the solution.
fn first_difference_penalty(n: usize) -> Vec<Vec<f64>> {
    let mut penalty = vec![vec![0.0; n]; n];

    for i in 0..(n - 1) {
        penalty[i][i] += 1.0;
        penalty[i][i + 1] -= 1.0;
        penalty[i + 1][i] -= 1.0;
        penalty[i + 1][i + 1] += 1.0;
    }

    penalty
}

/// Add alpha * B to A.
fn add_scaled_matrix(a: &[Vec<f64>], b: &[Vec<f64>], alpha: f64) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();

    let mut result = vec![vec![0.0; m]; n];

    for i in 0..n {
        for j in 0..m {
            result[i][j] = a[i][j] + alpha * b[i][j];
        }
    }

    result
}

/// Add deterministic reproducible measurement noise.
fn add_deterministic_noise(data: &[f64], relative_level: f64) -> Vec<f64> {
    let data_norm = infinity_norm(data);
    let n = data.len();

    data.iter()
        .enumerate()
        .map(|(i, &value)| {
            let phase = 2.0 * PI * i as f64 / (n as f64 - 1.0);
            let noise = relative_level * data_norm * (13.0 * phase).sin();
            value + noise
        })
        .collect()
}

/// Solve a dense linear system using Gaussian elimination with partial pivoting.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("The system matrix must be square and compatible with b.".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            let candidate = a[i][k].abs();

            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!(
                "Matrix is numerically singular near column {}.",
                k
            ));
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

/// Solve a weakly stabilized least-squares baseline:
///
///     (K^T K + epsilon I) u = K^T g.
///
/// The tiny ridge is included only to avoid numerical singularity in the
/// normal equations. It is not a deliberate smoothness prior.
fn weakly_stabilized_least_squares(
    k_matrix: &[Vec<f64>],
    data: &[f64],
    tiny_ridge: f64,
) -> Result<Vec<f64>, String> {
    let mut normal_matrix = gram_matrix(k_matrix);
    let normal_rhs = transpose_mat_vec(k_matrix, data);

    for i in 0..normal_matrix.len() {
        normal_matrix[i][i] += tiny_ridge;
    }

    solve_linear_system(normal_matrix, normal_rhs)
}

/// Solve the Tikhonov-regularized normal equations:
///
///     (K^T K + alpha L^T L) u = K^T g.
fn tikhonov_smoothness_reconstruction(
    k_matrix: &[Vec<f64>],
    data: &[f64],
    alpha: f64,
) -> Result<Vec<f64>, String> {
    let n = k_matrix[0].len();

    let kt_k = gram_matrix(k_matrix);
    let lt_l = first_difference_penalty(n);
    let regularized_matrix = add_scaled_matrix(&kt_k, &lt_l, alpha);
    let rhs = transpose_mat_vec(k_matrix, data);

    solve_linear_system(regularized_matrix, rhs)
}

/// Infinity norm of a vector.
fn infinity_norm(x: &[f64]) -> f64 {
    x.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

/// Infinity norm of x - y.
fn infinity_error(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Relative infinity-norm error.
fn relative_infinity_error(x: &[f64], y: &[f64]) -> f64 {
    infinity_error(x, y) / infinity_norm(y).max(1.0e-14)
}

/// Compute ||K u - g||_2^2.
fn data_misfit_squared(k_matrix: &[Vec<f64>], u: &[f64], data: &[f64]) -> f64 {
    let predicted = mat_vec(k_matrix, u);

    predicted
        .iter()
        .zip(data.iter())
        .map(|(&p, &g)| {
            let r = p - g;
            r * r
        })
        .sum()
}

/// Compute ||L u||_2^2 for the first-difference operator.
fn smoothness_penalty(u: &[f64]) -> f64 {
    let mut sum = 0.0;

    for i in 0..(u.len() - 1) {
        let diff = u[i + 1] - u[i];
        sum += diff * diff;
    }

    sum
}

/// Compute minimum value in a vector.
fn min_value(x: &[f64]) -> f64 {
    x.iter().fold(f64::INFINITY, |a, &b| a.min(b))
}

/// Compute maximum value in a vector.
fn max_value(x: &[f64]) -> f64 {
    x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let n = 35;
    let sigma = 0.10;
    let noise_level = 1.0e-3;

    let tiny_ridge = 1.0e-12;
    let alpha = 1.0e-3;

    let grid = uniform_grid(a, b, n);
    let weights = trapezoidal_weights(a, b, n);

    let k_matrix = assemble_smoothing_matrix(&grid, &weights, sigma);

    let exact: Vec<f64> = grid.iter().map(|&s| exact_solution(s)).collect();

    let clean_data = mat_vec(&k_matrix, &exact);
    let noisy_data = add_deterministic_noise(&clean_data, noise_level);

    let baseline =
        match weakly_stabilized_least_squares(&k_matrix, &noisy_data, tiny_ridge) {
            Ok(solution) => solution,
            Err(message) => {
                eprintln!("Weakly stabilized baseline reconstruction failed: {}", message);
                return;
            }
        };

    let regularized =
        match tikhonov_smoothness_reconstruction(&k_matrix, &noisy_data, alpha) {
            Ok(solution) => solution,
            Err(message) => {
                eprintln!("Tikhonov reconstruction failed: {}", message);
                return;
            }
        };

    let data_noise = relative_infinity_error(&noisy_data, &clean_data);

    let baseline_error = relative_infinity_error(&baseline, &exact);
    let reg_error = relative_infinity_error(&regularized, &exact);

    let baseline_misfit = data_misfit_squared(&k_matrix, &baseline, &noisy_data);
    let reg_misfit = data_misfit_squared(&k_matrix, &regularized, &noisy_data);

    let baseline_smoothness = smoothness_penalty(&baseline);
    let reg_smoothness = smoothness_penalty(&regularized);

    let baseline_objective = baseline_misfit + alpha * baseline_smoothness;
    let reg_objective = reg_misfit + alpha * reg_smoothness;

    println!("Tikhonov Regularization with a Smoothness Prior");
    println!("==============================================");
    println!();
    println!("Inverse Problem");
    println!("---------------");
    println!("Forward model               = first-kind Fredholm equation");
    println!("Discretized system          = K u = g_delta");
    println!("Grid points N               = {}", n);
    println!("Kernel                      = Gaussian smoothing kernel");
    println!("Kernel width sigma          = {:.6}", sigma);
    println!("Relative data noise         = {:.6e}", data_noise);
    println!();
    println!("Reconstruction Models");
    println!("---------------------");
    println!("Baseline method             = normal equations with tiny diagonal stabilization");
    println!("Tiny baseline ridge         = {:.6e}", tiny_ridge);
    println!("Regularized objective       = ||K u - g||^2 + alpha ||L u||^2");
    println!("Prior information           = smoothness of u");
    println!("Difference operator L       = first difference");
    println!("Regularization alpha        = {:.6e}", alpha);
    println!();
    println!("Diagnostics");
    println!("-----------");
    println!(
        "{:>24} {:>18} {:>18}",
        "quantity", "weak baseline", "regularized"
    );
    println!(
        "{:>24} {:>18.6e} {:>18.6e}",
        "relative error", baseline_error, reg_error
    );
    println!(
        "{:>24} {:>18.6e} {:>18.6e}",
        "data misfit", baseline_misfit, reg_misfit
    );
    println!(
        "{:>24} {:>18.6e} {:>18.6e}",
        "smoothness penalty", baseline_smoothness, reg_smoothness
    );
    println!(
        "{:>24} {:>18.6e} {:>18.6e}",
        "regularized objective", baseline_objective, reg_objective
    );
    println!(
        "{:>24} {:>18.6e} {:>18.6e}",
        "minimum u", min_value(&baseline), min_value(&regularized)
    );
    println!(
        "{:>24} {:>18.6e} {:>18.6e}",
        "maximum u", max_value(&baseline), max_value(&regularized)
    );
    println!();
    println!("Representative Values");
    println!("---------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "i", "s_i", "u_exact", "u_baseline", "u_tikhonov"
    );

    for i in 0..n {
        if i % 4 == 0 || i == n - 1 {
            println!(
                "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.10}",
                i,
                grid[i],
                exact[i],
                baseline[i],
                regularized[i]
            );
        }
    }

    println!();
    println!("Interpretation");
    println!("--------------");
    println!("The weak baseline uses only a tiny diagonal stabilization to avoid");
    println!("numerical singularity, so it still behaves like an unstable direct");
    println!("least-squares reconstruction. The Tikhonov solution uses a deliberate");
    println!("smoothness prior and therefore suppresses oscillatory noise.");
}
```

Program 19.5.2 demonstrates how smoothness prior information stabilizes an ill-posed inverse reconstruction. The weak baseline uses only a tiny diagonal stabilization to avoid numerical singularity, so it still behaves like an unstable direct least-squares method. Its data misfit may be small, but the reconstructed solution can contain large oscillatory artifacts and extreme values that do not resemble the true unknown.

The Tikhonov solution behaves differently because it explicitly includes the smoothness penalty from equation (19.5.12). By penalizing the first differences of the solution, the method suppresses high-frequency components that are weakly constrained by the smoothing forward operator and strongly affected by noise. The result is a reconstruction with a slightly larger data misfit but a much smaller roughness penalty and a substantially lower reconstruction error.

This example illustrates the practical meaning of the regularization parameter $\alpha$. If $\alpha$ were chosen too small, the reconstruction would approach the unstable baseline and overfit noise. If $\alpha$ were chosen too large, the reconstruction would become overly smooth and fail to represent features supported by the data. The selected value therefore expresses a compromise between the measured data and the assumed smoothness of the unknown.

The program also clarifies why a priori information is central to inverse problems. The data alone are insufficient to determine a stable reconstruction because the forward operator smooths the unknown. The smoothness prior restricts the admissible solutions to those that vary gradually, converting an unstable inverse problem into a controlled optimization problem. This prepares the ground for additional priors, such as positivity, boundedness, sparsity, and Bayesian priors, discussed in the following subsections.

## 19.5.4. Bayesian Interpretation and Choice of Prior

A priori information also has a natural Bayesian interpretation. In Bayesian inverse problems, the unknown $u$ is treated as a random variable. The data model describes the likelihood of observing $g$ given $u$, while the prior distribution describes what is believed about $u$ before observing the data. The posterior distribution combines both sources of information.

If the data errors are Gaussian and the prior is also Gaussian, then maximizing the posterior probability leads to a regularized least-squares problem. For example, a Gaussian smoothness prior gives a quadratic penalty of the form:

$$R(u)=\|Lu\|^2 \tag{19.5.13}$$

which corresponds to Tikhonov regularization. This shows that deterministic regularization and Bayesian maximum a posteriori estimation are closely related.

Other priors lead to different penalties. A sparsity prior, such as a Laplace prior on wavelet coefficients, leads to an $\ell_1$-type regularization. This is closely related to LASSO-type reconstruction. A prior favoring sparse gradients leads to total variation regularization, which is useful for preserving edges in images. Thus, the choice of $R(u)$ reflects the expected structure of the solution.

The proper prior depends on the application. A geological model may suggest a piecewise-constant or layered structure. A gravitational potential may be expected to be smooth. A medical image may contain sharp boundaries between tissues but relatively smooth regions inside each tissue. A blurred natural image may have sparse gradients, since most pixels vary slowly except at edges. These assumptions should be encoded in the regularization term or constraint.

Modern developments have expanded the meaning of a priori information. Deep-learning priors use neural networks trained on representative data to encode typical solution structure. Deep image priors exploit the architecture of a neural network itself as an implicit regularizer. Bayesian methods may compute or approximate the full posterior distribution rather than only a single best estimate, thereby providing uncertainty quantification. Data-driven projection methods and learned regularizers attempt to infer $R(\cdot)$ from training examples rather than specifying it manually. Across all these approaches, the essential issue remains the same: the reconstruction must balance fidelity to the measured data against a stable and meaningful prior model (Nature Research Intelligence, 2025).

## 19.5.5. Representative Applications

A classical application is *geophysical inversion*. In gravity or magnetic inversion, one measures a field $g$ at or near the surface and seeks an unknown subsurface density or susceptibility $u$. The forward map is smoothing, so direct inversion is highly unstable. However, geological information often suggests that the subsurface varies smoothly within layers and changes more sharply at interfaces. One may therefore solve:

$$\min_u \|Ku-g\|^2+\alpha\|\nabla u\|^2 \tag{19.5.14}$$

or use related penalties that allow layered or piecewise-smooth structure. The prior converts the unstable recovery of subsurface properties into a stable optimization problem.

A second application is *medical imaging*, such as diffuse optical tomography. Here $u$ may represent tissue absorption or scattering properties. These quantities are usually non-negative and often relatively smooth. A stable reconstruction may therefore combine data fidelity, non-negativity constraints, and smoothness penalties. The resulting inverse problem is not solved by direct inversion of $K$, but by an optimization formulation that respects both the measurements and the physical admissibility of the tissue parameters.

A third application is *image deblurring*. A blurred image may be modeled as:

$$g=Ku+\eta \tag{19.5.15}$$

where $u$ is the sharp image, $K$ is the blur operator, and $\eta$ represents noise. Since blur suppresses high-frequency information, direct inversion amplifies noise. A common prior is that the image has sparse gradients, meaning that large changes occur mainly at edges. This leads to total variation or related edge-preserving penalties. Such priors improve contrast and preserve important image boundaries while suppressing noise amplification.

In each of these examples, a priori information is the mechanism that makes the inverse problem computationally meaningful. Without it, the data alone do not determine a stable solution. With it, the reconstruction becomes a controlled compromise between measured evidence and physically or statistically justified structure.

### Rust Implementation

Following the discussion in Sections 19.5.4 and 19.5.5 on Bayesian priors, physical admissibility, and representative inverse-problem applications, Program 19.5.3 provides a practical implementation of projected regularized reconstruction with a positivity prior. The program builds on the smoothness-regularized framework of equation (19.5.13), but adds the physical constraint $u_i\ge 0$, corresponding to the positivity prior introduced in equation (19.5.6). This is appropriate when the unknown represents a density, absorption coefficient, concentration, intensity, or other nonnegative physical quantity. The implementation compares a weakly stabilized least-squares baseline with a projected Tikhonov reconstruction. The baseline illustrates how a lightly stabilized inverse solve can fit noisy data while producing negative and oscillatory artifacts, while the projected method combines smoothness regularization with nonnegativity enforcement to produce a physically admissible solution.

At the core of the implementation is the idea that a priori information may enter an inverse problem both through a penalty term and through an explicit constraint. The function `exact_solution(s)` defines a nonnegative reference profile with a small positive background and two localized peaks. This resembles the kinds of unknowns described in Section 19.5.5, such as densities, absorption profiles, concentrations, or intensities. The function `kernel(t, s, sigma)` defines a Gaussian smoothing kernel, which produces indirect data from the unknown and makes direct inversion unstable.

The functions `uniform_grid` and `trapezoidal_weights` define the discretization of the integral operator. The grid gives the nodal representation of the unknown and the observation points, while the quadrature weights approximate the integral over the interval. The function `assemble_smoothing_matrix` then constructs the weighted matrix approximation of the forward operator. Applying this matrix to the exact unknown produces synthetic data for the inverse problem.

The functions `mat_vec`, `transpose_mat_vec`, and `gram_matrix` provide the linear algebra operations needed by the two reconstruction methods. The matrix-vector product applies the forward operator, the transposed product is used in gradient and normal-equation computations, and the Gram matrix $K^\top K$ is used to form the weakly stabilized least-squares baseline.

The function `add_deterministic_noise` adds a reproducible perturbation to the synthetic data. This represents noisy measured data of the kind described in equation (19.5.15). The perturbation is chosen to make the inverse problem visibly unstable, so that the contrast between a weak baseline and a constrained regularized reconstruction is clear in the output.

The function `weakly_stabilized_least_squares` computes the baseline reconstruction. It solves a normal-equation system with only a tiny diagonal stabilization. This tiny ridge is included only to prevent numerical singularity, not to impose a meaningful prior. As a result, the baseline can still fit the noisy data while producing large oscillations and negative values. This makes it useful as a diagnostic example of what can go wrong when the inverse problem is not sufficiently regularized.

The function `smoothness_penalty` computes the first-difference roughness measure used in the Tikhonov part of the reconstruction. The function `apply_first_difference_transpose_difference` applies $L^\top L$ for this first-difference operator without explicitly forming the matrix. Together, these functions implement the smoothness prior associated with the quadratic penalty in equation (19.5.13).

The function `data_misfit_squared` computes the squared discrepancy between the predicted data and the noisy measurements. The function `objective` combines this data fidelity term with the smoothness penalty. This is the optimization expression minimized by the projected method, except that the projected method also enforces nonnegativity at every iteration.

The function `regularized_gradient` computes the gradient of the smoothness-regularized objective. It combines the data-misfit gradient with the smoothness gradient. The function `project_nonnegative` then enforces the positivity prior by replacing any negative component with zero. This projection is the key step that incorporates the physical constraint $u_i\ge 0$ directly into the numerical method.

The function `estimate_step_size` computes a conservative step size for gradient descent using a row-sum estimate of the operator size and the smoothness contribution. This avoids the need for eigenvalue computations while keeping the iteration stable. The function `initial_guess_from_data` provides a simple nonnegative constant starting point based on the average data level.

The `projected_gradient_descent` function performs the constrained reconstruction. At each iteration, it takes a gradient step for the regularized objective and then projects the result onto the nonnegative set. The iteration stops when the update becomes smaller than the prescribed tolerance or when the maximum iteration count is reached. This implements a simple projected optimization method suitable for demonstrating positivity-constrained regularized inversion.

The remaining helper functions measure reconstruction quality and physical admissibility. The functions `infinity_norm`, `infinity_error`, and `relative_infinity_error` compute relative errors. The functions `count_negative_entries`, `min_value`, and `max_value` report whether the reconstruction violates positivity and how extreme its values are. The function `count_projection_changes` records how many entries in the weak baseline would violate the nonnegativity constraint.

The `main` function assembles the full experiment. It constructs the grid, forward operator, exact nonnegative profile, clean data, and noisy data. It then computes the weak baseline by a lightly stabilized least-squares solve and the constrained reconstruction by projected gradient descent. The printed diagnostics compare relative error, data misfit, smoothness penalty, objective value, negative entries, and the range of reconstructed values. The representative nodal values show that the weak baseline can contain large nonphysical oscillations, while the projected reconstruction remains nonnegative and close to the exact profile.

```rust
// Program 19.5.3: Projected Regularized Reconstruction with a Positivity Prior
//
// Problem statement:
// Recover a nonnegative unknown u from noisy indirect data
//
//     K u = g_delta,
//
// where K is a smoothing discretized integral operator. The program compares:
//
// 1. A weakly stabilized least-squares baseline,
// 2. A projected Tikhonov-regularized reconstruction.
//
// The projected reconstruction minimizes
//
//     J(u) = ||K u - g_delta||_2^2 + alpha ||L u||_2^2,
//
// subject to
//
//     u_i >= 0.
//
// The smoothness term corresponds to a Gaussian prior or Tikhonov-type MAP
// estimate, while the projection step enforces the physical positivity prior.

use std::f64::consts::PI;

/// Exact nonnegative unknown function u(s).
///
/// The function contains a smooth positive background plus two localized
/// positive features. This resembles a density, absorption coefficient,
/// concentration, or intensity profile.
fn exact_solution(s: f64) -> f64 {
    let peak1 = 0.9 * (-((s - 0.30) * (s - 0.30)) / 0.006).exp();
    let peak2 = 0.6 * (-((s - 0.72) * (s - 0.72)) / 0.010).exp();

    0.15 + peak1 + peak2
}

/// Gaussian smoothing kernel K(t, s).
///
/// This kernel models a smoothing forward operator, as commonly encountered
/// in first-kind integral equations.
fn kernel(t: f64, s: f64, sigma: f64) -> f64 {
    let distance = t - s;
    (-distance * distance / (2.0 * sigma * sigma)).exp()
}

/// Build a uniform grid on [a, b].
fn uniform_grid(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n < 2 {
        panic!("At least two grid points are required.");
    }

    let h = (b - a) / (n as f64 - 1.0);
    (0..n).map(|i| a + i as f64 * h).collect()
}

/// Composite trapezoidal weights on [a, b].
fn trapezoidal_weights(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);
    let mut weights = vec![h; n];

    weights[0] = 0.5 * h;
    weights[n - 1] = 0.5 * h;

    weights
}

/// Assemble the weighted smoothing matrix.
///
/// The entries are
///
///     K_ij = K(t_i, s_j) w_j,
///
/// so that K u approximates the integral operator applied to u.
fn assemble_smoothing_matrix(
    grid: &[f64],
    weights: &[f64],
    sigma: f64,
) -> Vec<Vec<f64>> {
    let n = grid.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        let t_i = grid[i];

        for j in 0..n {
            let s_j = grid[j];
            matrix[i][j] = kernel(t_i, s_j, sigma) * weights[j];
        }
    }

    matrix
}

/// Matrix-vector product y = A x.
fn mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&a, &xj)| a * xj).sum())
        .collect()
}

/// Transposed matrix-vector product y = A^T x.
fn transpose_mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut result = vec![0.0; cols];

    for i in 0..rows {
        for j in 0..cols {
            result[j] += matrix[i][j] * x[i];
        }
    }

    result
}

/// Matrix-matrix product C = A^T A.
fn gram_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut gram = vec![vec![0.0; cols]; cols];

    for i in 0..cols {
        for j in 0..cols {
            let mut sum = 0.0;

            for k in 0..rows {
                sum += matrix[k][i] * matrix[k][j];
            }

            gram[i][j] = sum;
        }
    }

    gram
}

/// Add deterministic reproducible measurement noise.
///
/// The perturbation is strong enough to reveal the difference between
/// a weak baseline reconstruction and a positivity-constrained one.
fn add_deterministic_noise(data: &[f64], relative_level: f64) -> Vec<f64> {
    let data_norm = infinity_norm(data);
    let n = data.len();

    data.iter()
        .enumerate()
        .map(|(i, &value)| {
            let phase = 2.0 * PI * i as f64 / (n as f64 - 1.0);
            let noise = relative_level
                * data_norm
                * ((9.0 * phase).sin() + 0.35 * (17.0 * phase).cos());
            value + noise
        })
        .collect()
}

/// Solve a dense linear system using Gaussian elimination with partial pivoting.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("The system matrix must be square and compatible with b.".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            let candidate = a[i][k].abs();

            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!(
                "Matrix is numerically singular near column {}.",
                k
            ));
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

/// Solve a weakly stabilized least-squares baseline:
///
///     (K^T K + epsilon I) u = K^T g.
///
/// The tiny ridge is included only to avoid numerical singularity. It is not
/// intended to represent meaningful prior information.
fn weakly_stabilized_least_squares(
    k_matrix: &[Vec<f64>],
    data: &[f64],
    tiny_ridge: f64,
) -> Result<Vec<f64>, String> {
    let mut normal_matrix = gram_matrix(k_matrix);
    let normal_rhs = transpose_mat_vec(k_matrix, data);

    for i in 0..normal_matrix.len() {
        normal_matrix[i][i] += tiny_ridge;
    }

    solve_linear_system(normal_matrix, normal_rhs)
}

/// Compute the first-difference smoothness penalty ||L u||_2^2.
fn smoothness_penalty(u: &[f64]) -> f64 {
    let mut sum = 0.0;

    for i in 0..(u.len() - 1) {
        let diff = u[i + 1] - u[i];
        sum += diff * diff;
    }

    sum
}

/// Apply L^T L for the first-difference operator without forming the matrix.
///
/// If L u contains u_{i+1} - u_i, then L^T L is tridiagonal.
fn apply_first_difference_transpose_difference(u: &[f64]) -> Vec<f64> {
    let n = u.len();
    let mut result = vec![0.0; n];

    if n == 1 {
        return result;
    }

    result[0] = u[0] - u[1];

    for i in 1..(n - 1) {
        result[i] = 2.0 * u[i] - u[i - 1] - u[i + 1];
    }

    result[n - 1] = u[n - 1] - u[n - 2];

    result
}

/// Compute the squared data misfit ||K u - g||_2^2.
fn data_misfit_squared(k_matrix: &[Vec<f64>], u: &[f64], data: &[f64]) -> f64 {
    let predicted = mat_vec(k_matrix, u);

    predicted
        .iter()
        .zip(data.iter())
        .map(|(&p, &g)| {
            let r = p - g;
            r * r
        })
        .sum()
}

/// Compute the regularized objective
///
///     ||K u - g||_2^2 + alpha ||L u||_2^2.
fn objective(k_matrix: &[Vec<f64>], u: &[f64], data: &[f64], alpha: f64) -> f64 {
    data_misfit_squared(k_matrix, u, data) + alpha * smoothness_penalty(u)
}

/// Compute the gradient of the regularized objective:
///
///     grad J(u) = 2 K^T(Ku - g) + 2 alpha L^T L u.
fn regularized_gradient(
    k_matrix: &[Vec<f64>],
    u: &[f64],
    data: &[f64],
    alpha: f64,
) -> Vec<f64> {
    let predicted = mat_vec(k_matrix, u);

    let residual: Vec<f64> = predicted
        .iter()
        .zip(data.iter())
        .map(|(&p, &g)| p - g)
        .collect();

    let kt_residual = transpose_mat_vec(k_matrix, &residual);
    let lt_l_u = apply_first_difference_transpose_difference(u);

    kt_residual
        .iter()
        .zip(lt_l_u.iter())
        .map(|(&a, &b)| 2.0 * a + 2.0 * alpha * b)
        .collect()
}

/// Project a vector onto the nonnegative orthant.
fn project_nonnegative(u: &mut [f64]) {
    for value in u.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

/// Estimate a conservative step size for gradient descent.
fn estimate_step_size(k_matrix: &[Vec<f64>], alpha: f64) -> f64 {
    let n = k_matrix[0].len();

    let mut max_row_sum: f64 = 0.0;

    for i in 0..n {
        let mut row_sum = 0.0;

        for j in 0..n {
            let mut kt_k_ij = 0.0;

            for r in 0..k_matrix.len() {
                kt_k_ij += k_matrix[r][i] * k_matrix[r][j];
            }

            row_sum += kt_k_ij.abs();
        }

        max_row_sum = max_row_sum.max(row_sum);
    }

    let smoothness_bound = 4.0 * alpha;
    let lipschitz_estimate = 2.0 * (max_row_sum + smoothness_bound);

    0.95 / lipschitz_estimate.max(1.0e-14)
}

/// Initialize the reconstruction by a constant nonnegative value.
fn initial_guess_from_data(data: &[f64]) -> Vec<f64> {
    let average = data.iter().sum::<f64>() / data.len() as f64;
    vec![average.max(0.0); data.len()]
}

/// Result of projected gradient descent.
struct GradientResult {
    solution: Vec<f64>,
    iterations: usize,
    final_objective: f64,
    final_step: f64,
}

/// Projected gradient descent for the nonnegative regularized inverse problem.
fn projected_gradient_descent(
    k_matrix: &[Vec<f64>],
    data: &[f64],
    alpha: f64,
    max_iterations: usize,
    tolerance: f64,
) -> GradientResult {
    let mut u = initial_guess_from_data(data);
    let step_size = estimate_step_size(k_matrix, alpha);

    let mut final_step = f64::INFINITY;
    let mut iterations = 0;

    for iteration in 1..=max_iterations {
        let old = u.clone();
        let gradient = regularized_gradient(k_matrix, &u, data, alpha);

        for i in 0..u.len() {
            u[i] -= step_size * gradient[i];
        }

        project_nonnegative(&mut u);

        final_step = infinity_error(&u, &old);
        iterations = iteration;

        if final_step < tolerance {
            break;
        }
    }

    let final_objective = objective(k_matrix, &u, data, alpha);

    GradientResult {
        solution: u,
        iterations,
        final_objective,
        final_step,
    }
}

/// Infinity norm of a vector.
fn infinity_norm(x: &[f64]) -> f64 {
    x.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

/// Infinity norm of x - y.
fn infinity_error(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Relative infinity-norm error.
fn relative_infinity_error(x: &[f64], y: &[f64]) -> f64 {
    infinity_error(x, y) / infinity_norm(y).max(1.0e-14)
}

/// Count negative entries in a vector.
fn count_negative_entries(x: &[f64]) -> usize {
    x.iter().filter(|&&value| value < 0.0).count()
}

/// Minimum value of a vector.
fn min_value(x: &[f64]) -> f64 {
    x.iter().fold(f64::INFINITY, |a, &b| a.min(b))
}

/// Maximum value of a vector.
fn max_value(x: &[f64]) -> f64 {
    x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
}

/// Count entries that were changed by enforcing nonnegativity.
fn count_projection_changes(baseline: &[f64]) -> usize {
    baseline.iter().filter(|&&value| value < 0.0).count()
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let n = 60;
    let sigma = 0.08;

    let noise_level = 8.0e-3;
    let tiny_ridge = 1.0e-12;
    let alpha = 1.0e-4;

    let max_iterations = 30_000;
    let tolerance = 1.0e-10;

    let grid = uniform_grid(a, b, n);
    let weights = trapezoidal_weights(a, b, n);

    let k_matrix = assemble_smoothing_matrix(&grid, &weights, sigma);

    let exact: Vec<f64> = grid.iter().map(|&s| exact_solution(s)).collect();

    let clean_data = mat_vec(&k_matrix, &exact);
    let noisy_data = add_deterministic_noise(&clean_data, noise_level);

    let baseline =
        match weakly_stabilized_least_squares(&k_matrix, &noisy_data, tiny_ridge) {
            Ok(solution) => solution,
            Err(message) => {
                eprintln!("Weak baseline reconstruction failed: {}", message);
                return;
            }
        };

    let projected = projected_gradient_descent(
        &k_matrix,
        &noisy_data,
        alpha,
        max_iterations,
        tolerance,
    );

    let data_noise = relative_infinity_error(&noisy_data, &clean_data);

    let baseline_error = relative_infinity_error(&baseline, &exact);
    let projected_error = relative_infinity_error(&projected.solution, &exact);

    let baseline_misfit = data_misfit_squared(&k_matrix, &baseline, &noisy_data);
    let projected_misfit = data_misfit_squared(&k_matrix, &projected.solution, &noisy_data);

    let baseline_smoothness = smoothness_penalty(&baseline);
    let projected_smoothness = smoothness_penalty(&projected.solution);

    let baseline_objective = objective(&k_matrix, &baseline, &noisy_data, alpha);

    println!("Projected Regularized Reconstruction with a Positivity Prior");
    println!("============================================================");
    println!();
    println!("Inverse Problem");
    println!("---------------");
    println!("Forward model               = first-kind Fredholm equation");
    println!("Application interpretation  = nonnegative density or absorption profile");
    println!("Discretized system          = K u = g_delta");
    println!("Grid points N               = {}", n);
    println!("Kernel                      = Gaussian smoothing kernel");
    println!("Kernel width sigma          = {:.6}", sigma);
    println!("Relative data noise         = {:.6e}", data_noise);
    println!();
    println!("Prior and Optimization Model");
    println!("----------------------------");
    println!("Weak baseline               = least squares with tiny diagonal stabilization");
    println!("Tiny baseline ridge         = {:.6e}", tiny_ridge);
    println!("Projected objective         = ||K u - g||^2 + alpha ||L u||^2");
    println!("Smoothness prior            = alpha ||L u||^2");
    println!("Positivity prior            = u_i >= 0");
    println!("Regularization alpha        = {:.6e}", alpha);
    println!("Solver                      = projected gradient descent");
    println!("Maximum iterations          = {}", max_iterations);
    println!("Tolerance                   = {:.6e}", tolerance);
    println!();
    println!("Diagnostics");
    println!("-----------");
    println!(
        "{:>28} {:>18} {:>18}",
        "quantity", "weak baseline", "projected"
    );
    println!(
        "{:>28} {:>18.6e} {:>18.6e}",
        "relative error", baseline_error, projected_error
    );
    println!(
        "{:>28} {:>18.6e} {:>18.6e}",
        "data misfit", baseline_misfit, projected_misfit
    );
    println!(
        "{:>28} {:>18.6e} {:>18.6e}",
        "smoothness penalty", baseline_smoothness, projected_smoothness
    );
    println!(
        "{:>28} {:>18.6e} {:>18.6e}",
        "objective", baseline_objective, projected.final_objective
    );
    println!(
        "{:>28} {:>18} {:>18}",
        "iterations", "direct solve", projected.iterations
    );
    println!(
        "{:>28} {:>18} {:>18.6e}",
        "final step", "N/A", projected.final_step
    );
    println!(
        "{:>28} {:>18} {:>18}",
        "negative entries",
        count_negative_entries(&baseline),
        count_negative_entries(&projected.solution)
    );
    println!(
        "{:>28} {:>18} {:>18}",
        "projection changes",
        count_projection_changes(&baseline),
        count_negative_entries(&projected.solution)
    );
    println!(
        "{:>28} {:>18.6e} {:>18.6e}",
        "minimum u",
        min_value(&baseline),
        min_value(&projected.solution)
    );
    println!(
        "{:>28} {:>18.6e} {:>18.6e}",
        "maximum u",
        max_value(&baseline),
        max_value(&projected.solution)
    );
    println!();
    println!("Representative Values");
    println!("---------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "i", "s_i", "u_exact", "u_baseline", "u_projected"
    );

    for i in 0..n {
        if i % 7 == 0 || i == n - 1 {
            println!(
                "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.10}",
                i,
                grid[i],
                exact[i],
                baseline[i],
                projected.solution[i]
            );
        }
    }

    println!();
    println!("Interpretation");
    println!("--------------");
    println!("The weak baseline is included only to show what can happen when the");
    println!("inverse problem is stabilized too weakly: it may fit the noisy data but");
    println!("produce negative or oscillatory values. The projected reconstruction uses");
    println!("a smoothness prior and enforces u_i >= 0, which is appropriate when the");
    println!("unknown represents a physical density, absorption coefficient,");
    println!("concentration, or intensity.");
}
```

Program 19.5.3 demonstrates how physical prior information can be incorporated directly into an inverse reconstruction. The weak baseline is included to show the danger of insufficient regularization: it may fit the noisy data well, but the recovered profile can contain large oscillations, negative values, and extreme amplitudes that are incompatible with the physical meaning of the unknown.

The projected reconstruction behaves differently because it combines two forms of a priori information. The smoothness penalty suppresses rapid oscillations, while the projection step enforces nonnegativity. This is especially relevant in applications such as medical imaging, optical tomography, geophysical density recovery, and concentration estimation, where negative values have no physical interpretation.

The diagnostic output illustrates the difference between data fitting and meaningful reconstruction. The weak baseline may achieve a small data misfit, but it has many negative entries and a very large smoothness penalty. The projected reconstruction accepts a slightly different balance of misfit and regularity, but produces a stable nonnegative solution with a much smaller reconstruction error.

This example also connects the deterministic and Bayesian interpretations of regularization. The quadratic smoothness penalty corresponds to a Gaussian smoothness prior, while the nonnegativity projection represents hard physical prior information. Together, they show how the reconstruction is shaped not only by the measured data, but also by assumptions about what kinds of solutions are admissible. This is the central role of a priori information in practical inverse problems.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/J314ubeqzJ1YvrEycTNz.4","tags":[]}

# 19.6. Linear Regularization Methods

Linear regularization methods provide the basic computational framework for stabilizing ill-posed inverse problems. They replace direct inversion by a controlled optimization problem in which the measured data are balanced against prior assumptions on the unknown solution. This section develops the classical Tikhonov framework, its operator interpretation, its singular-value filtering behavior, and its role in practical inverse problems.

## 19.6.1. Tikhonov Regularization for Stabilizing Ill-Posed Linear Inverse Problems

A fundamental and widely used approach to stable inversion is *Tikhonov regularization*. It provides a systematic way to replace an unstable inverse problem by a nearby well-posed optimization problem. Suppose that, after discretization, an inverse problem is written as:

$$Ku=g, \tag{19.6.1}$$

where $K$ is the discretized forward operator, $g$ is the measured data, and $u$ is the unknown vector to be reconstructed. If $K$ is ill-conditioned, direct inversion is unstable because noise in $g$ may be strongly amplified in the computed solution. Tikhonov regularization stabilizes the problem by adding a penalty term that discourages unstable or physically unreasonable solutions (Nature Research Intelligence, 2025).

In its simplest zero-order form, Tikhonov regularization solves,

$$\min_u \|Ku-g\|^2+\lambda \|u\|^2 \tag{19.6.2}$$

where $\lambda>0$ is the *regularization parameter*. The first term $\|Ku-g\|^2,$ measures the discrepancy between the model prediction and the observed data, while the second term, $\lambda\|u\|^2,$ penalizes large solutions. This penalty encodes the prior assumption that the desired reconstruction should not have unnecessarily large amplitude. It also prevents the solution from following unstable directions associated with small singular values of $K$.

Setting the gradient of the objective function in (19.6.2) equal to zero gives the normal equation:

$$(K^\top K+\lambda I)u=K^\top g \tag{19.6.3}$$

The addition of the positive definite term $\lambda I$ is the essential stabilizing mechanism. Even when $K^\top K$ is ill-conditioned or nearly singular, the matrix $K^\top K+\lambda I$ is better conditioned for $\lambda>0$. Thus, Tikhonov regularization replaces the unstable inversion of $K$ by the solution of a stabilized linear system.

## 19.6.2. General-Form Tikhonov Regularization with Smoothness and Differential Penalties

The zero-order penalty $\|u\|^2$ is not always the most appropriate form of prior information. In many inverse problems, the solution is not necessarily small in magnitude but is expected to be smooth, slowly varying, or free of excessive oscillations. To encode this information, one introduces a linear operator $L$, often a discrete derivative, gradient, or Laplacian. The general-form Tikhonov problem is:

$$\min_u  \|Ku-g\|^2+\lambda \|Lu\|^2 \tag{19.6.4}$$

Here $L$ determines what property of the solution is penalized. If $L=I$, (19.6.4) reduces to the standard zero-order form (19.6.2). If $L$ is a first-difference or gradient operator, the method penalizes rapid variation and encourages smoothness. If $L$ is a second-difference or Laplacian operator, the method penalizes curvature and produces even smoother reconstructions.

The corresponding normal equation is:

$$(K^\top K+\lambda L^\top L)u=K^\top g \tag{19.6.5}$$

This equation shows explicitly how the data term and the prior term combine. The matrix $K^\top K$ enforces consistency with the measurements, while $\lambda L^\top L$ enforces the chosen regularity condition. The parameter $\lambda$ controls the balance between these two effects. A small $\lambda$ gives a solution close to the unregularized least-squares solution, whereas a large $\lambda$ produces a solution dominated by the prior.

The general-form method is important because it allows the regularization to be adapted to the physical meaning of the unknown. In image restoration, $L$ may represent a discrete gradient or Laplacian. In geophysical inversion, it may encode spatial smoothness or geological layering. In medical imaging, it may suppress oscillations while preserving physically plausible tissue parameters. Thus, the choice of $L$ is part of the modeling process, not merely a numerical detail.

### Rust Implementation

Following the discussion in Section 19.6.2 on general-form Tikhonov regularization with smoothness and differential penalties, Program 19.6.1 provides a practical implementation of the regularized normal equation in equation (19.6.5). The program compares three choices of the regularization operator $L$: the identity operator, a first-difference operator, and a second-difference operator. These choices correspond to different forms of a priori information in equation (19.6.4). The zero-order penalty controls the overall size of the reconstruction, the first-difference penalty suppresses rapid variation, and the second-difference penalty suppresses curvature. By solving the same noisy inverse problem with all three penalties, the implementation shows that the choice of $L$ is a modeling decision that directly affects the stability, smoothness, and accuracy of the reconstructed solution.

At the core of the implementation is the general-form Tikhonov problem in equation (19.6.4). The function `exact_solution(s)` defines the reference unknown used to generate synthetic data. It contains a smooth background together with moderate oscillatory structure, making it suitable for comparing how different penalties preserve or suppress variation. The function `kernel(t, s, sigma)` defines a Gaussian smoothing kernel, which plays the role of the forward integral operator. Because this operator damps fine-scale information, the inverse reconstruction from noisy data is unstable unless regularization is introduced.

The functions `uniform_grid` and `trapezoidal_weights` construct the discretization of the integral equation. The grid represents the nodal points for both the unknown and the data, while the trapezoidal weights approximate the integral over the interval. The function `assemble_smoothing_matrix` then builds the weighted discrete operator $K$, so that matrix-vector multiplication approximates the action of the integral operator on the unknown function.

The function `mat_vec` applies a matrix to a vector, while `transpose_mat_vec` applies the transpose of a matrix. These two operations are used to form the data prediction $Ku$ and the right-hand side $K^\top g$ in the normal equation. The function `gram_matrix` constructs $K^\top K$, which represents the data-fitting part of the regularized normal equation in equation (19.6.5).

The functions `identity_penalty`, `first_difference_penalty`, and `second_difference_penalty` construct the three different penalty matrices $L^\top L$. The identity penalty corresponds to the zero-order form in equation (19.6.2), where the size of the solution itself is penalized. The first-difference penalty corresponds to a discrete gradient, suppressing rapid changes between neighboring grid values. The second-difference penalty corresponds to a discrete curvature measure, suppressing excessive bending or oscillatory curvature in the reconstruction.

The function `add_scaled_matrix` forms the matrix $K^\top K+\lambda L^\top L$, which appears in equation (19.6.5). This is the central algebraic step of the program. The data term enforces consistency with the measured data, while the scaled penalty term imposes the chosen prior information. The scalar parameter $\lambda$ controls the balance between these two effects.

The function `add_deterministic_noise` adds a reproducible oscillatory perturbation to the clean synthetic data. This models the noisy inverse-problem setting and ensures that the program gives the same output each time it is run. The function `solve_linear_system` solves the resulting dense linear systems using Gaussian elimination with partial pivoting. It is used to compute each Tikhonov reconstruction.

The function `tikhonov_reconstruction` implements the complete solution process for a chosen penalty matrix. It forms $K^\top K$, computes $K^\top g$, adds the scaled penalty matrix, and solves the stabilized linear system. This function is reused for all three choices of $L$, which highlights the modular structure of the general-form Tikhonov framework.

The function `data_misfit_squared` computes the squared discrepancy between the predicted data and the noisy measurements. The function `quadratic_penalty` evaluates the penalty value $u^\top L^\top L u$, which is equivalent to $\|Lu\|^2$. These two quantities correspond to the two terms in equation (19.6.4). The helper functions `infinity_norm`, `infinity_error`, `relative_infinity_error`, `min_value`, and `max_value` provide reconstruction diagnostics.

The function `print_diagnostics` summarizes the behavior of each reconstruction. It reports relative error, data misfit, penalty value, regularized objective, and the range of the reconstructed solution. The `main` function assembles the complete experiment by generating the grid, forward operator, exact solution, clean data, noisy data, and the three penalty matrices. It then solves the inverse problem using each choice of $L$, prints diagnostic information, and displays representative nodal values for comparison.

```rust
// Program 19.6.1: Zero-Order and General-Form Tikhonov Regularization
// with Differential Penalties
//
// Problem statement:
// Recover an unknown function u from noisy indirect data
//
//     K u = g_delta,
//
// where K is a smoothing discretized integral operator. The program compares
// three Tikhonov regularization choices:
//
//     L = I                 zero-order penalty,
//     L = first difference  gradient-like smoothness penalty,
//     L = second difference curvature-like smoothness penalty.
//
// Each reconstruction solves the general-form Tikhonov normal equation
//
//     (K^T K + lambda L^T L) u = K^T g_delta.
//
// This demonstrates how the operator L controls the type of prior information
// imposed on the inverse problem.

use std::f64::consts::PI;

/// Exact unknown function u(s).
///
/// The function contains a smooth background plus moderate oscillatory
/// structure. This makes it useful for comparing how different regularizers
/// preserve or suppress variation.
fn exact_solution(s: f64) -> f64 {
    1.0 + 0.55 * (PI * s).sin() + 0.18 * (5.0 * PI * s).sin()
}

/// Gaussian smoothing kernel K(t, s).
///
/// The kernel damps fine-scale features of u, making recovery from noisy
/// data unstable without regularization.
fn kernel(t: f64, s: f64, sigma: f64) -> f64 {
    let distance = t - s;
    (-distance * distance / (2.0 * sigma * sigma)).exp()
}

/// Build a uniform grid on [a, b].
fn uniform_grid(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n < 2 {
        panic!("At least two grid points are required.");
    }

    let h = (b - a) / (n as f64 - 1.0);
    (0..n).map(|i| a + i as f64 * h).collect()
}

/// Composite trapezoidal quadrature weights on [a, b].
fn trapezoidal_weights(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);
    let mut weights = vec![h; n];

    weights[0] = 0.5 * h;
    weights[n - 1] = 0.5 * h;

    weights
}

/// Assemble the weighted smoothing matrix K.
///
/// The entries are
///
///     K_ij = K(t_i, s_j) w_j,
///
/// so that K u approximates the integral operator applied to u.
fn assemble_smoothing_matrix(
    grid: &[f64],
    weights: &[f64],
    sigma: f64,
) -> Vec<Vec<f64>> {
    let n = grid.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        let t_i = grid[i];

        for j in 0..n {
            let s_j = grid[j];
            matrix[i][j] = kernel(t_i, s_j, sigma) * weights[j];
        }
    }

    matrix
}

/// Matrix-vector product y = A x.
fn mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&a, &xj)| a * xj).sum())
        .collect()
}

/// Transposed matrix-vector product y = A^T x.
fn transpose_mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut result = vec![0.0; cols];

    for i in 0..rows {
        for j in 0..cols {
            result[j] += matrix[i][j] * x[i];
        }
    }

    result
}

/// Gram matrix A^T A.
fn gram_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut gram = vec![vec![0.0; cols]; cols];

    for i in 0..cols {
        for j in 0..cols {
            let mut sum = 0.0;

            for k in 0..rows {
                sum += matrix[k][i] * matrix[k][j];
            }

            gram[i][j] = sum;
        }
    }

    gram
}

/// Construct L^T L for L = I.
fn identity_penalty(n: usize) -> Vec<Vec<f64>> {
    let mut penalty = vec![vec![0.0; n]; n];

    for i in 0..n {
        penalty[i][i] = 1.0;
    }

    penalty
}

/// Construct L^T L for the first-difference operator.
///
/// If L u contains u_{i+1} - u_i, then L^T L penalizes rapid variation.
fn first_difference_penalty(n: usize) -> Vec<Vec<f64>> {
    let mut penalty = vec![vec![0.0; n]; n];

    for i in 0..(n - 1) {
        penalty[i][i] += 1.0;
        penalty[i][i + 1] -= 1.0;
        penalty[i + 1][i] -= 1.0;
        penalty[i + 1][i + 1] += 1.0;
    }

    penalty
}

/// Construct L^T L for the second-difference operator.
///
/// If L u contains u_i - 2u_{i+1} + u_{i+2}, then L^T L penalizes curvature.
fn second_difference_penalty(n: usize) -> Vec<Vec<f64>> {
    let mut penalty = vec![vec![0.0; n]; n];

    for i in 0..(n - 2) {
        let stencil = [(i, 1.0), (i + 1, -2.0), (i + 2, 1.0)];

        for &(row_index, row_value) in &stencil {
            for &(col_index, col_value) in &stencil {
                penalty[row_index][col_index] += row_value * col_value;
            }
        }
    }

    penalty
}

/// Add lambda * B to A.
fn add_scaled_matrix(a: &[Vec<f64>], b: &[Vec<f64>], lambda: f64) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut result = vec![vec![0.0; m]; n];

    for i in 0..n {
        for j in 0..m {
            result[i][j] = a[i][j] + lambda * b[i][j];
        }
    }

    result
}

/// Add deterministic reproducible measurement noise.
fn add_deterministic_noise(data: &[f64], relative_level: f64) -> Vec<f64> {
    let data_norm = infinity_norm(data);
    let n = data.len();

    data.iter()
        .enumerate()
        .map(|(i, &value)| {
            let phase = 2.0 * PI * i as f64 / (n as f64 - 1.0);
            let noise = relative_level
                * data_norm
                * ((11.0 * phase).sin() + 0.25 * (17.0 * phase).cos());
            value + noise
        })
        .collect()
}

/// Solve a dense linear system by Gaussian elimination with partial pivoting.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("The system matrix must be square and compatible with b.".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            let candidate = a[i][k].abs();

            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!(
                "Matrix is numerically singular near column {}.",
                k
            ));
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

/// Solve the general-form Tikhonov system
///
///     (K^T K + lambda L^T L) u = K^T g.
fn tikhonov_reconstruction(
    k_matrix: &[Vec<f64>],
    data: &[f64],
    penalty_matrix: &[Vec<f64>],
    lambda: f64,
) -> Result<Vec<f64>, String> {
    let kt_k = gram_matrix(k_matrix);
    let rhs = transpose_mat_vec(k_matrix, data);
    let regularized_matrix = add_scaled_matrix(&kt_k, penalty_matrix, lambda);

    solve_linear_system(regularized_matrix, rhs)
}

/// Compute ||K u - g||_2^2.
fn data_misfit_squared(k_matrix: &[Vec<f64>], u: &[f64], data: &[f64]) -> f64 {
    let predicted = mat_vec(k_matrix, u);

    predicted
        .iter()
        .zip(data.iter())
        .map(|(&p, &g)| {
            let r = p - g;
            r * r
        })
        .sum()
}

/// Compute u^T P u, where P = L^T L.
fn quadratic_penalty(u: &[f64], penalty_matrix: &[Vec<f64>]) -> f64 {
    let pu = mat_vec(penalty_matrix, u);
    u.iter().zip(pu.iter()).map(|(&ui, &pui)| ui * pui).sum()
}

/// Infinity norm of a vector.
fn infinity_norm(x: &[f64]) -> f64 {
    x.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

/// Infinity norm of x - y.
fn infinity_error(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Relative infinity-norm error.
fn relative_infinity_error(x: &[f64], y: &[f64]) -> f64 {
    infinity_error(x, y) / infinity_norm(y).max(1.0e-14)
}

/// Minimum value of a vector.
fn min_value(x: &[f64]) -> f64 {
    x.iter().fold(f64::INFINITY, |a, &b| a.min(b))
}

/// Maximum value of a vector.
fn max_value(x: &[f64]) -> f64 {
    x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
}

/// Print diagnostic information for one reconstruction.
fn print_diagnostics(
    name: &str,
    solution: &[f64],
    exact: &[f64],
    k_matrix: &[Vec<f64>],
    data: &[f64],
    penalty_matrix: &[Vec<f64>],
    lambda: f64,
) {
    let misfit = data_misfit_squared(k_matrix, solution, data);
    let penalty = quadratic_penalty(solution, penalty_matrix);
    let objective = misfit + lambda * penalty;
    let rel_error = relative_infinity_error(solution, exact);

    println!("{name}");
    println!("  relative error             = {:.6e}", rel_error);
    println!("  data misfit                = {:.6e}", misfit);
    println!("  penalty value              = {:.6e}", penalty);
    println!("  regularized objective      = {:.6e}", objective);
    println!("  minimum u                  = {:.6e}", min_value(solution));
    println!("  maximum u                  = {:.6e}", max_value(solution));
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let n = 45;
    let sigma = 0.10;
    let noise_level = 2.0e-3;
    let lambda = 5.0e-4;

    let grid = uniform_grid(a, b, n);
    let weights = trapezoidal_weights(a, b, n);

    let k_matrix = assemble_smoothing_matrix(&grid, &weights, sigma);

    let exact: Vec<f64> = grid.iter().map(|&s| exact_solution(s)).collect();

    let clean_data = mat_vec(&k_matrix, &exact);
    let noisy_data = add_deterministic_noise(&clean_data, noise_level);

    let identity = identity_penalty(n);
    let first_difference = first_difference_penalty(n);
    let second_difference = second_difference_penalty(n);

    let zero_order = match tikhonov_reconstruction(&k_matrix, &noisy_data, &identity, lambda) {
        Ok(solution) => solution,
        Err(message) => {
            eprintln!("Zero-order Tikhonov reconstruction failed: {}", message);
            return;
        }
    };

    let first_order =
        match tikhonov_reconstruction(&k_matrix, &noisy_data, &first_difference, lambda) {
            Ok(solution) => solution,
            Err(message) => {
                eprintln!("First-difference Tikhonov reconstruction failed: {}", message);
                return;
            }
        };

    let second_order =
        match tikhonov_reconstruction(&k_matrix, &noisy_data, &second_difference, lambda) {
            Ok(solution) => solution,
            Err(message) => {
                eprintln!("Second-difference Tikhonov reconstruction failed: {}", message);
                return;
            }
        };

    let relative_data_noise = relative_infinity_error(&noisy_data, &clean_data);

    println!("Zero-Order and General-Form Tikhonov Regularization");
    println!("==================================================");
    println!();
    println!("Inverse Problem");
    println!("---------------");
    println!("Forward model               = first-kind Fredholm equation");
    println!("Discretized system          = K u = g_delta");
    println!("Grid points N               = {}", n);
    println!("Kernel                      = Gaussian smoothing kernel");
    println!("Kernel width sigma          = {:.6}", sigma);
    println!("Relative data noise         = {:.6e}", relative_data_noise);
    println!();
    println!("Regularization Setup");
    println!("--------------------");
    println!("General form                = ||K u - g||^2 + lambda ||L u||^2");
    println!("Normal equation             = (K^T K + lambda L^T L)u = K^T g");
    println!("Regularization lambda       = {:.6e}", lambda);
    println!("Compared choices of L       = I, first difference, second difference");
    println!();
    println!("Diagnostics");
    println!("-----------");
    print_diagnostics(
        "L = I: zero-order penalty",
        &zero_order,
        &exact,
        &k_matrix,
        &noisy_data,
        &identity,
        lambda,
    );
    println!();
    print_diagnostics(
        "L = first difference: variation penalty",
        &first_order,
        &exact,
        &k_matrix,
        &noisy_data,
        &first_difference,
        lambda,
    );
    println!();
    print_diagnostics(
        "L = second difference: curvature penalty",
        &second_order,
        &exact,
        &k_matrix,
        &noisy_data,
        &second_difference,
        lambda,
    );
    println!();
    println!("Representative Values");
    println!("---------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18} {:>18}",
        "i", "s_i", "u_exact", "L=I", "L=diff1", "L=diff2"
    );

    for i in 0..n {
        if i % 5 == 0 || i == n - 1 {
            println!(
                "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.10} {:>18.10}",
                i, grid[i], exact[i], zero_order[i], first_order[i], second_order[i]
            );
        }
    }

    println!();
    println!("Interpretation");
    println!("--------------");
    println!("The zero-order penalty controls the solution amplitude. The first-difference");
    println!("penalty suppresses rapid variation, while the second-difference penalty");
    println!("suppresses curvature. All three methods solve the same data-fitting");
    println!("problem, but the choice of L changes the prior information imposed on");
    println!("the reconstruction.");
}
```

Program 19.6.1 demonstrates that the regularization operator $L$ determines the type of prior information imposed on the inverse problem. Although all three reconstructions solve the same noisy data-fitting problem and use the same value of $\lambda$, they behave differently because they penalize different solution features.

The zero-order penalty $L=I$ controls the amplitude of the solution. In the numerical output, this penalty strongly suppresses the magnitude of the reconstruction, especially near the endpoints. This illustrates a limitation of zero-order Tikhonov regularization: it can stabilize the inverse problem, but it may bias the solution toward smaller values even when the true solution is not small.

The first-difference penalty gives a smoother and more accurate reconstruction by penalizing rapid changes rather than the size of the solution itself. This is often more appropriate when the unknown is expected to vary gradually. The second-difference penalty performs best in this example because it penalizes curvature while allowing the solution amplitude and smooth trends to remain close to the true profile.

This comparison reinforces the modeling role of equation (19.6.4). General-form Tikhonov regularization is not only a numerical stabilization technique. It is also a way of encoding assumptions about the unknown function. Choosing $L=I$, a gradient-like operator, or a curvature-like operator changes the meaning of the prior and therefore changes the character of the reconstruction. This prepares the reader for later discussions of filter factors, parameter selection, and application-specific regularization choices.

## 19.6.3. Singular-Value Decomposition Interpretation and Regularization Filter Factors

The stabilizing effect of Tikhonov regularization is especially clear when viewed through the singular value decomposition. Suppose that,

$$K=U\Sigma V^\top \tag{19.6.6}$$

where $U$ and $V$ are orthogonal matrices and $\Sigma$ contains the singular values:

$$\sigma_1,\sigma_2,\dots,\sigma_r$$

For the zero-order problem (19.6.2), the regularized solution can be written as:

$$u_\lambda = \sum_j\frac{\sigma_j}{\sigma_j^2+\lambda}\,(u_j^\top g)\,v_j \tag{19.6.7}$$

where $u_j$ and $v_j$ are the left and right singular vectors of $K$, respectively. The scalar factor,

$$\frac{\sigma_j}{\sigma_j^2+\lambda} \tag{19.6.8}$$

acts as a *filter factor*. In the unregularized pseudoinverse, the corresponding factor is $1/\sigma_j$, which becomes very large when $\sigma_j$ is small. These small singular values are precisely the modes that amplify noise. Tikhonov regularization modifies this behavior by replacing $1/\sigma_j$ with the bounded filter in (19.6.8).

For large singular values, where $\sigma_j^2\gg \lambda$, the filter behaves approximately like:

$$\frac{\sigma_j}{\sigma_j^2+\lambda}\approx \frac{1}{\sigma_j}$$

Thus, well-determined components of the solution are retained. For small singular values, where $\sigma_j^2\ll \lambda$, the filter behaves like:

$$\frac{\sigma_j}{\sigma_j^2+\lambda}\approx \frac{\sigma_j}{\lambda}$$

so the contribution of unstable small-singular-value modes is strongly suppressed. This explains why Tikhonov regularization produces stable reconstructions: it damps precisely those components that would otherwise be most sensitive to noise. As $\lambda\to 0,$ the Tikhonov solution approaches the pseudoinverse solution, provided the problem is consistent and the singular modes are retained. As $\lambda\to \infty,$ the solution tends toward zero in the zero-order case. Therefore, $\lambda$ determines the compromise between fidelity to the data and suppression of unstable components. If $\lambda$ is too small, the solution overfits noise. If $\lambda$ is too large, the solution is overly biased toward the regularization model.

This SVD interpretation also connects Tikhonov regularization with other filtering methods. Truncated SVD removes small singular-value components abruptly, while Tikhonov regularization suppresses them smoothly. Iterated Tikhonov variants further refine this filtering behavior by applying regularization repeatedly or in modified forms, with the aim of improving reconstruction accuracy while maintaining stability (Pang and Wang, 2025).

### Rust Implementation

Following the discussion in Section 19.6.3 on the singular-value decomposition interpretation of Tikhonov regularization, Program 19.6.2 provides a practical implementation of regularization filter factors in singular-vector coordinates. The program follows equations (19.6.6) through (19.6.8) by constructing a synthetic inverse problem with prescribed singular values and known modal coefficients. Instead of computing an SVD from a matrix, the example works directly in the singular basis so that the role of each singular value is visible. It compares the unregularized pseudoinverse, several Tikhonov reconstructions with different values of $\lambda$, and a truncated SVD reconstruction. This makes the stabilizing mechanism explicit: the pseudoinverse amplifies noise in small-singular-value modes, while Tikhonov regularization replaces the unstable factor $1/\sigma_j$ with the bounded filter factor in equation (19.6.8).

At the core of the implementation is the singular-value expansion described in equation (19.6.6). The program does not compute the singular value decomposition numerically. Instead, it constructs a controlled synthetic problem directly in singular-vector coordinates. This makes the influence of each singular value transparent and avoids distracting the reader with the details of SVD algorithms.

The constants `N` and `R` define the number of physical grid points and the number of retained singular modes. The function `right_singular_vector(j, i)` evaluates a discrete sine basis vector, which plays the role of the right singular vector $v_j$. Low-index modes are smooth, while higher-index modes oscillate more rapidly. This mirrors the behavior of many smoothing inverse problems, where fine-scale information is associated with smaller singular values.

The function `singular_value(j)` prescribes a rapidly decaying singular-value sequence. This models the compact smoothing behavior of an integral operator. As the index increases, the singular values become small, and the corresponding pseudoinverse factors $1/\sigma_j$ become large. These small-singular-value modes are precisely the components that produce noise amplification in ill-posed inverse problems.

The function `exact_modal_coefficient(j)` defines the true solution in modal coordinates. The coefficients decay with the mode number, so the true solution is dominated by lower modes but still contains some higher-frequency content. The function `build_exact_solution` converts these modal coefficients into physical-space values by summing the right singular vectors with their corresponding coefficients.

The function `noisy_data_coefficients` constructs the data coefficients in the left singular-vector basis. In singular coordinates, the forward operator multiplies each solution coefficient by its singular value. The function then adds a small deterministic perturbation using `data_noise_coefficient`. This models noisy measured data while keeping the numerical output reproducible.

The function `pseudoinverse_reconstruction` implements the unregularized inverse. It divides each data coefficient by the corresponding singular value, which is the pseudoinverse factor discussed in Section 19.6.3. This is accurate for well-determined components but unstable for small singular values because noise is amplified by the large factor $1/\sigma_j$.

The function `tikhonov_reconstruction` implements equation (19.6.7). For each mode, it multiplies the noisy data coefficient by the Tikhonov filter factor in equation (19.6.8). The filter behaves like the pseudoinverse for large singular values but damps the contribution of small singular values. This gives a smooth transition between retaining reliable modes and suppressing unstable ones.

The function `truncated_svd_reconstruction` provides a comparison with truncated SVD. Unlike Tikhonov regularization, which damps small-singular-value modes smoothly, truncated SVD discards modes abruptly once the singular value falls below a chosen cutoff. This comparison supports the final paragraph of Section 19.6.3, where Tikhonov filtering is contrasted with truncated SVD.

The functions `pseudoinverse_modal_coefficients`, `tikhonov_modal_coefficients`, and `truncated_svd_modal_coefficients` compute the reconstructed modal coefficients for each method. These coefficients are used to compute residual norms and solution norms in singular coordinates. The function `residual_norm_squared_from_coefficients` measures the squared data residual, while `norm_squared` measures the squared modal solution norm. Together, these diagnostics show the trade-off between fitting the data and controlling the reconstructed solution.

The function `print_filter_table` prints selected singular values, pseudoinverse factors, Tikhonov filter factors, and damping ratios for a representative value of $\lambda$. This table directly exposes the filtering mechanism: for large singular values, the damping ratio is close to one, while for small singular values, the Tikhonov factor is much smaller than the pseudoinverse factor.

The function `print_reconstruction_diagnostics` summarizes each reconstruction by reporting relative reconstruction error, data residual, and modal solution norm. The `main` function assembles the experiment by setting the noise level, selecting several regularization parameters, constructing the exact solution and noisy data coefficients, and comparing the pseudoinverse, Tikhonov, and truncated SVD reconstructions. It also prints representative physical-space values so that the effect of filtering can be seen in the reconstructed function itself.

```rust
// Program 19.6.2: SVD Filter Factors in Tikhonov Regularization
//
// Problem statement:
// Demonstrate the singular-value filtering interpretation of zero-order
// Tikhonov regularization.
//
// The discretized inverse problem is written in singular-vector coordinates:
//
//     K = U Sigma V^T.
//
// Instead of computing a numerical SVD, the program constructs a synthetic
// problem directly in the right singular-vector basis. The exact solution is
// expanded in known modal coefficients, the data coefficients are generated
// by multiplying by prescribed singular values, and small deterministic noise
// is added.
//
// The program compares:
//
//     1. The pseudoinverse reconstruction,
//     2. Tikhonov reconstructions for several lambda values,
//     3. Truncated SVD reconstruction.
//
// This illustrates the Tikhonov filter factor
//
//     sigma_j / (sigma_j^2 + lambda),
//
// and shows how small singular-value modes are suppressed.

use std::f64::consts::PI;

/// Number of discrete sample points.
const N: usize = 64;

/// Number of singular modes retained in the synthetic model.
const R: usize = 24;

/// Right singular vector basis v_j evaluated at grid index i.
///
/// We use a discrete sine basis on the interior-like grid. This provides
/// smooth low-frequency modes and increasingly oscillatory high-frequency
/// modes.
fn right_singular_vector(j: usize, i: usize) -> f64 {
    let x = (i as f64 + 1.0) / (N as f64 + 1.0);
    let normalization = (2.0 / (N as f64 + 1.0)).sqrt();

    normalization * ((j as f64 + 1.0) * PI * x).sin()
}

/// Prescribed singular values.
///
/// They decay rapidly, mimicking the smoothing behavior of compact integral
/// operators. Small singular values are responsible for noise amplification
/// in the pseudoinverse.
fn singular_value(j: usize) -> f64 {
    let decay_rate = 0.28;
    (-decay_rate * j as f64).exp()
}

/// Exact modal coefficient in the right singular-vector basis.
///
/// Low modes dominate the true solution, while high modes are smaller.
fn exact_modal_coefficient(j: usize) -> f64 {
    let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
    sign / ((j + 1) as f64).powf(1.35)
}

/// Deterministic noise added to the data coefficient u_j^T g.
///
/// Noise is deliberately small in the data coordinates, but it becomes
/// strongly amplified by division by small singular values in the pseudoinverse.
fn data_noise_coefficient(j: usize, noise_level: f64) -> f64 {
    let phase = 2.0 * PI * (j as f64 + 1.0) / R as f64;
    noise_level * ((7.0 * phase).sin() + 0.35 * (11.0 * phase).cos())
}

/// Build the exact solution in physical coordinates from modal coefficients.
fn build_exact_solution() -> Vec<f64> {
    let mut solution = vec![0.0; N];

    for j in 0..R {
        let coefficient = exact_modal_coefficient(j);

        for i in 0..N {
            solution[i] += coefficient * right_singular_vector(j, i);
        }
    }

    solution
}

/// Generate noisy data coefficients in the left singular-vector basis.
///
/// In singular coordinates,
///
///     g_j = sigma_j * c_j,
///
/// where c_j is the exact solution coefficient. We then add deterministic
/// noise to obtain noisy data coefficients.
fn noisy_data_coefficients(noise_level: f64) -> Vec<f64> {
    let mut data = vec![0.0; R];

    for j in 0..R {
        let sigma = singular_value(j);
        let clean = sigma * exact_modal_coefficient(j);
        let noise = data_noise_coefficient(j, noise_level);

        data[j] = clean + noise;
    }

    data
}

/// Reconstruct by the unregularized pseudoinverse.
///
/// In singular coordinates, this uses the factor 1 / sigma_j.
/// This is unstable when sigma_j is small.
fn pseudoinverse_reconstruction(data_coefficients: &[f64]) -> Vec<f64> {
    let mut reconstruction = vec![0.0; N];

    for j in 0..R {
        let sigma = singular_value(j);
        let coefficient = data_coefficients[j] / sigma;

        for i in 0..N {
            reconstruction[i] += coefficient * right_singular_vector(j, i);
        }
    }

    reconstruction
}

/// Reconstruct using zero-order Tikhonov filter factors.
///
/// The coefficient multiplying the data coefficient is
///
///     sigma_j / (sigma_j^2 + lambda).
fn tikhonov_reconstruction(data_coefficients: &[f64], lambda: f64) -> Vec<f64> {
    let mut reconstruction = vec![0.0; N];

    for j in 0..R {
        let sigma = singular_value(j);
        let filter = sigma / (sigma * sigma + lambda);
        let coefficient = filter * data_coefficients[j];

        for i in 0..N {
            reconstruction[i] += coefficient * right_singular_vector(j, i);
        }
    }

    reconstruction
}

/// Reconstruct using truncated SVD.
///
/// Modes with singular values below cutoff are discarded completely.
fn truncated_svd_reconstruction(data_coefficients: &[f64], cutoff: f64) -> Vec<f64> {
    let mut reconstruction = vec![0.0; N];

    for j in 0..R {
        let sigma = singular_value(j);

        if sigma >= cutoff {
            let coefficient = data_coefficients[j] / sigma;

            for i in 0..N {
                reconstruction[i] += coefficient * right_singular_vector(j, i);
            }
        }
    }

    reconstruction
}

/// Compute the infinity norm of a vector.
fn infinity_norm(x: &[f64]) -> f64 {
    x.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

/// Compute the infinity norm of x - y.
fn infinity_error(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Compute relative infinity-norm error.
fn relative_infinity_error(x: &[f64], y: &[f64]) -> f64 {
    infinity_error(x, y) / infinity_norm(y).max(1.0e-14)
}

/// Compute Euclidean norm squared.
fn norm_squared(x: &[f64]) -> f64 {
    x.iter().map(|value| value * value).sum()
}

/// Compute the data residual norm in singular coordinates.
///
/// For a reconstructed coefficient vector c_lambda, the predicted data
/// coefficient is sigma_j c_lambda_j.
fn residual_norm_squared_from_coefficients(
    data_coefficients: &[f64],
    reconstructed_coefficients: &[f64],
) -> f64 {
    let mut sum = 0.0;

    for j in 0..R {
        let sigma = singular_value(j);
        let residual = sigma * reconstructed_coefficients[j] - data_coefficients[j];

        sum += residual * residual;
    }

    sum
}

/// Compute reconstructed modal coefficients for Tikhonov.
fn tikhonov_modal_coefficients(data_coefficients: &[f64], lambda: f64) -> Vec<f64> {
    let mut coefficients = vec![0.0; R];

    for j in 0..R {
        let sigma = singular_value(j);
        let filter = sigma / (sigma * sigma + lambda);
        coefficients[j] = filter * data_coefficients[j];
    }

    coefficients
}

/// Compute reconstructed modal coefficients for the pseudoinverse.
fn pseudoinverse_modal_coefficients(data_coefficients: &[f64]) -> Vec<f64> {
    let mut coefficients = vec![0.0; R];

    for j in 0..R {
        coefficients[j] = data_coefficients[j] / singular_value(j);
    }

    coefficients
}

/// Compute reconstructed modal coefficients for truncated SVD.
fn truncated_svd_modal_coefficients(data_coefficients: &[f64], cutoff: f64) -> Vec<f64> {
    let mut coefficients = vec![0.0; R];

    for j in 0..R {
        let sigma = singular_value(j);

        if sigma >= cutoff {
            coefficients[j] = data_coefficients[j] / sigma;
        }
    }

    coefficients
}

/// Print selected filter factors for one lambda value.
fn print_filter_table(lambda: f64) {
    println!("Filter Factors for lambda = {:.6e}", lambda);
    println!("-----------------------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "j", "sigma_j", "1/sigma_j", "tik_filter", "damping_ratio"
    );

    for j in 0..R {
        if j < 8 || j % 4 == 3 || j == R - 1 {
            let sigma = singular_value(j);
            let pseudo_filter = 1.0 / sigma;
            let tik_filter = sigma / (sigma * sigma + lambda);
            let damping_ratio = tik_filter / pseudo_filter;

            println!(
                "{:>8} {:>14.6e} {:>18.6e} {:>18.6e} {:>18.6e}",
                j + 1,
                sigma,
                pseudo_filter,
                tik_filter,
                damping_ratio
            );
        }
    }

    println!();
}

/// Print diagnostic information for one reconstruction.
fn print_reconstruction_diagnostics(
    name: &str,
    reconstruction: &[f64],
    modal_coefficients: &[f64],
    exact: &[f64],
    data_coefficients: &[f64],
) {
    let relative_error = relative_infinity_error(reconstruction, exact);
    let residual_squared =
        residual_norm_squared_from_coefficients(data_coefficients, modal_coefficients);
    let solution_norm_squared = norm_squared(modal_coefficients);

    println!("{name}");
    println!("  relative reconstruction error = {:.6e}", relative_error);
    println!("  data residual squared         = {:.6e}", residual_squared);
    println!("  modal solution norm squared   = {:.6e}", solution_norm_squared);
}

fn main() {
    let noise_level = 1.0e-4;
    let lambdas = [1.0e-8, 1.0e-5, 1.0e-3, 1.0e-1];
    let tsvd_cutoff = 1.0e-2;

    let exact = build_exact_solution();
    let data_coefficients = noisy_data_coefficients(noise_level);

    let pseudoinverse_coefficients = pseudoinverse_modal_coefficients(&data_coefficients);
    let pseudoinverse = pseudoinverse_reconstruction(&data_coefficients);

    let tsvd_coefficients = truncated_svd_modal_coefficients(&data_coefficients, tsvd_cutoff);
    let tsvd = truncated_svd_reconstruction(&data_coefficients, tsvd_cutoff);

    println!("SVD Filter Factors in Tikhonov Regularization");
    println!("============================================");
    println!();
    println!("Synthetic SVD Model");
    println!("-------------------");
    println!("Number of grid points N       = {}", N);
    println!("Number of singular modes R    = {}", R);
    println!("Singular values               = exp(-0.28(j-1))");
    println!("Data coefficient noise level  = {:.6e}", noise_level);
    println!("TSVD cutoff                   = {:.6e}", tsvd_cutoff);
    println!();

    println!("Selected Singular Values");
    println!("------------------------");
    println!("{:>8} {:>18} {:>18}", "j", "sigma_j", "1/sigma_j");

    for j in 0..R {
        if j < 8 || j % 4 == 3 || j == R - 1 {
            let sigma = singular_value(j);
            println!(
                "{:>8} {:>18.6e} {:>18.6e}",
                j + 1,
                sigma,
                1.0 / sigma
            );
        }
    }

    println!();
    print_filter_table(1.0e-5);

    println!("Reconstruction Diagnostics");
    println!("--------------------------");
    print_reconstruction_diagnostics(
        "Pseudoinverse reconstruction",
        &pseudoinverse,
        &pseudoinverse_coefficients,
        &exact,
        &data_coefficients,
    );
    println!();

    for &lambda in &lambdas {
        let coefficients = tikhonov_modal_coefficients(&data_coefficients, lambda);
        let reconstruction = tikhonov_reconstruction(&data_coefficients, lambda);

        print_reconstruction_diagnostics(
            &format!("Tikhonov reconstruction, lambda = {:.1e}", lambda),
            &reconstruction,
            &coefficients,
            &exact,
            &data_coefficients,
        );
        println!();
    }

    print_reconstruction_diagnostics(
        &format!("Truncated SVD reconstruction, cutoff = {:.1e}", tsvd_cutoff),
        &tsvd,
        &tsvd_coefficients,
        &exact,
        &data_coefficients,
    );

    println!();
    println!("Representative Physical-Space Values");
    println!("------------------------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "i", "x_i", "u_exact", "u_pinv", "u_tik_1e-5"
    );

    let tik_1e5 = tikhonov_reconstruction(&data_coefficients, 1.0e-5);

    for i in 0..N {
        if i % 8 == 0 || i == N - 1 {
            let x = (i as f64 + 1.0) / (N as f64 + 1.0);

            println!(
                "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.10}",
                i, x, exact[i], pseudoinverse[i], tik_1e5[i]
            );
        }
    }

    println!();
    println!("Interpretation");
    println!("--------------");
    println!("The pseudoinverse uses the factor 1/sigma_j, which becomes large");
    println!("for small singular values and therefore amplifies data noise.");
    println!("Tikhonov regularization replaces this unstable factor by");
    println!("sigma_j / (sigma_j^2 + lambda), damping the small-singular-value");
    println!("components smoothly rather than discarding them abruptly.");
}
```

Program 19.6.2 demonstrates the filtering interpretation of zero-order Tikhonov regularization. The pseudoinverse uses the factor $1/\sigma_j$, which becomes large when $\sigma_j$ is small. As a result, even small perturbations in the data coefficients can be magnified into visible errors in the reconstructed solution.

Tikhonov regularization changes this behavior by replacing the pseudoinverse factor with the filter factor in equation (19.6.8). The numerical output shows that large-singular-value modes are retained almost unchanged, while small-singular-value modes are damped. This is the precise mechanism by which Tikhonov regularization stabilizes the inverse problem: it suppresses the modes that are most sensitive to noise.

The comparison across different values of $\lambda$ illustrates the regularization trade-off. If $\lambda$ is too small, the reconstruction behaves much like the pseudoinverse and can still overfit noise. If $\lambda$ is too large, the reconstruction becomes overly damped and loses important solution features. An intermediate value gives the best balance between retaining reliable information and suppressing unstable components.

The truncated SVD comparison reinforces the distinction between abrupt and smooth filtering. Truncated SVD removes all modes below a cutoff, while Tikhonov regularization gradually damps them according to their singular values and the chosen $\lambda$. This makes the program a useful bridge between the algebraic SVD formula and the practical behavior of regularized inverse reconstructions.

## 19.6.4. Numerical Algorithms, Computational Complexity, and Regularization Parameter Selection

Solving the regularized normal equation:

$$(K^\top K+\lambda L^\top L)u=K^\top g \tag{19.6.9}$$

for an $N\times N$ system requires $O(N^3)$ operations if treated by a dense direct solver. For medium-sized problems, direct methods such as Cholesky factorization, QR factorization, or SVD-based algorithms are often appropriate. Cholesky methods are efficient when the regularized system matrix is symmetric positive definite, while QR and SVD methods provide greater robustness when conditioning is more delicate.

For large-scale inverse problems, direct factorization is usually too expensive. Krylov subspace methods, such as conjugate gradients applied to the normal equations or related least-squares solvers, are commonly used. These methods require only matrix-vector products with $K$, $K^\top$, $L$, and $L^\top$. This is important when (K) represents a discretized integral operator that is too large to store explicitly.

Matrix-free methods are especially useful in imaging, geophysics, tomography, and boundary integral problems. Instead of forming $K$, one applies the forward operator and its adjoint as computational procedures. When $K$ has additional structure, such as convolutional form, fast Fourier transforms may reduce the cost of each operator application. If $K$ comes from a smooth integral kernel, compression methods such as adaptive cross approximation, hierarchical matrices, or related low-rank techniques can reduce storage and accelerate matrix-vector products.

Iterative schemes can also act as regularization methods. Landweber iteration, for example, repeatedly applies gradient descent to the least-squares objective. If stopped early, it suppresses unstable small-singular-value components in a way analogous to explicit regularization. Accelerated methods, including Nesterov-type accelerated gradient schemes, can improve convergence while still using early stopping as an implicit stabilizing mechanism. These approaches are useful when explicit solution of (19.6.9) is impractical.

A critical aspect of every regularization method is the choice of $\lambda$. The regularization parameter determines how strongly the prior term influences the solution. Several parameter-choice rules are common.

The *discrepancy principle* chooses $\lambda$ so that the residual norm matches the expected noise level:

$$\|Ku_\lambda-g\|\approx \delta \tag{19.6.10}$$

where $\delta$ is an estimate of the noise magnitude. The idea is that the reconstruction should fit the data only to the level justified by the noise, not beyond it. The *L-curve criterion* plots the residual norm $\|Ku_\lambda-g\|$ against the regularization norm $\|Lu_\lambda\|$. The preferred value of $\lambda$ is often chosen near the corner of the curve, where the trade-off between data fidelity and regularity is most balanced.

Cross-validation and related predictive criteria choose $\lambda$ by estimating how well the reconstruction generalizes to withheld or perturbed data. In practice, parameter selection is essential. A poor choice of $\lambda$ can undermine the entire regularization process, either by allowing noise amplification or by oversmoothing the solution.

### Rust Implementation

Following the discussion in Section 19.6.4 on numerical algorithms, computational complexity, and regularization parameter selection, Program 19.6.3 provides a practical implementation of Tikhonov parameter selection over a logarithmic grid of $\lambda$ values. The program solves the regularized normal equation in equation (19.6.9) for each candidate parameter and records the residual norm, regularization norm, objective value, and reconstruction error. It then compares two practical parameter-choice rules: the discrepancy principle in equation (19.6.10), which selects $\lambda$ so that the residual norm is close to the estimated noise level, and an L-curve curvature diagnostic, which seeks a balanced corner between data fidelity and regularity. This example shows why choosing $\lambda$ is not a secondary detail, but a central part of linear regularization: too little regularization overfits noise, while too much regularization oversmooths the solution.

At the core of the implementation is the repeated solution of the Tikhonov system in equation (19.6.9). The function `exact_solution(s)` defines the reference unknown used to generate synthetic data. It is smooth but not constant, which makes it suitable for showing the effect of undersmoothing and oversmoothing as $\lambda$ changes. The function `kernel(t, s, sigma)` defines a Gaussian smoothing kernel, which produces an ill-conditioned first-kind inverse problem.

The functions `uniform_grid` and `trapezoidal_weights` define the discretization of the integral operator. The grid gives the nodal points for the unknown and the observations, while the trapezoidal weights approximate the integral over the interval. The function `assemble_smoothing_matrix` builds the weighted matrix $K$, so that applying the matrix to the sampled unknown approximates the forward integral operator.

The functions `mat_vec`, `transpose_mat_vec`, and `gram_matrix` provide the linear algebra needed for the normal equations. The matrix-vector product evaluates $Ku$, the transposed product evaluates $K^\top g$, and the Gram matrix constructs $K^\top K$. These components are reused for all candidate values of $\lambda$, which reflects the computational structure of parameter sweeps in regularized inversion.

The function `first_difference_penalty` constructs the matrix $L^\top L$ for a first-difference regularization operator. This corresponds to a smoothness prior: neighboring values of the reconstructed solution should not vary too rapidly. The function `first_difference_values` applies $L$ to a reconstructed solution so that the regularization norm $\|Lu_\lambda\|$ can be computed for the L-curve diagnostic.

The function `add_scaled_matrix` forms $K^\top K+\lambda L^\top L$ for each candidate regularization parameter. This is the matrix appearing in equation (19.6.9). The function `add_deterministic_noise` adds a reproducible perturbation to the clean data. The noise norm computed from this perturbation supplies the value of $\delta$ used in the discrepancy principle in equation (19.6.10).

The function `solve_linear_system` solves each dense regularized system by Gaussian elimination with partial pivoting. For the moderate problem size used in this example, a dense direct solver is appropriate and matches the discussion in Section 19.6.4. In large-scale settings, the same algebraic system would usually be handled by iterative or matrix-free methods.

The function `tikhonov_solution` wraps the construction and solution of the regularized normal equation for a single value of $\lambda$. It takes precomputed $K^\top K$, $K^\top g$, and $L^\top L$, forms the regularized matrix, and returns the reconstruction. This modular design makes the parameter sweep straightforward.

The functions `l2_norm`, `infinity_norm`, `difference`, and `relative_infinity_error` compute the residual norm, regularization norm, noise norm, and reconstruction error. The function `lambda_grid` constructs a logarithmically spaced sequence of candidate $\lambda$ values. A logarithmic grid is appropriate because useful regularization parameters often vary over several orders of magnitude. The `LambdaDiagnostic` structure stores all quantities associated with a candidate $\lambda$: residual norm, regularization norm, relative error, objective value, curvature, and the reconstructed solution. The objective value records the balance between data fidelity and regularity for that value of $\lambda$.

The function `l_curve_curvature` estimates the curvature of the L-curve using three neighboring points in the log-residual and log-regularization plane. This provides a simple numerical version of the L-curve corner rule. The function `choose_by_l_curve` selects the point with the largest curvature, while `choose_by_discrepancy` selects the point whose residual norm is closest to the estimated noise norm. The function `choose_by_best_error` is included only for validation because the exact solution is known in this synthetic experiment but would not be known in a real inverse problem.

The `main` function assembles the full parameter-selection experiment. It constructs the grid, forward operator, first-difference penalty, exact solution, clean data, noisy data, noise norm, and normal-equation components. It then solves the Tikhonov system for each candidate $\lambda$, computes diagnostics, estimates L-curve curvature, and reports the parameters selected by the discrepancy principle, L-curve diagnostic, and known best error. Finally, it prints representative values from the discrepancy-principle reconstruction to show the quality of the selected solution.

```rust
// Program 19.6.3: Regularization Parameter Selection by the Discrepancy
// Principle and L-Curve Diagnostics
//
// Problem statement:
// Solve the Tikhonov-regularized inverse problem
//
//     (K^T K + lambda L^T L)u = K^T g,
//
// for a range of lambda values. For each lambda, compute:
//
//     residual norm        ||K u_lambda - g||_2,
//     regularization norm  ||L u_lambda||_2,
//     objective value      ||K u_lambda - g||_2^2 + lambda ||L u_lambda||_2^2,
//     reconstruction error ||u_lambda - u_exact||_inf / ||u_exact||_inf.
//
// The program then chooses lambda by:
//
// 1. The discrepancy principle, where ||K u_lambda - g|| is closest to
//    the known noise norm delta.
// 2. A simple L-curve curvature diagnostic in the log-log plane.
//
// This illustrates how parameter selection controls the balance between
// data fitting and regularization.

use std::f64::consts::PI;

/// Exact unknown function u(s).
///
/// The function is smooth but not constant, making it useful for observing
/// oversmoothing when lambda is too large.
fn exact_solution(s: f64) -> f64 {
    1.0 + 0.55 * (PI * s).sin() + 0.18 * (5.0 * PI * s).sin()
}

/// Gaussian smoothing kernel K(t, s).
///
/// This produces an ill-conditioned first-kind inverse problem.
fn kernel(t: f64, s: f64, sigma: f64) -> f64 {
    let distance = t - s;
    (-distance * distance / (2.0 * sigma * sigma)).exp()
}

/// Build a uniform grid on [a, b].
fn uniform_grid(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n < 2 {
        panic!("At least two grid points are required.");
    }

    let h = (b - a) / (n as f64 - 1.0);
    (0..n).map(|i| a + i as f64 * h).collect()
}

/// Composite trapezoidal quadrature weights.
fn trapezoidal_weights(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);
    let mut weights = vec![h; n];

    weights[0] = 0.5 * h;
    weights[n - 1] = 0.5 * h;

    weights
}

/// Assemble the weighted smoothing matrix K.
fn assemble_smoothing_matrix(grid: &[f64], weights: &[f64], sigma: f64) -> Vec<Vec<f64>> {
    let n = grid.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        let t_i = grid[i];

        for j in 0..n {
            let s_j = grid[j];
            matrix[i][j] = kernel(t_i, s_j, sigma) * weights[j];
        }
    }

    matrix
}

/// Matrix-vector product y = A x.
fn mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&a, &xj)| a * xj).sum())
        .collect()
}

/// Transposed matrix-vector product y = A^T x.
fn transpose_mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut result = vec![0.0; cols];

    for i in 0..rows {
        for j in 0..cols {
            result[j] += matrix[i][j] * x[i];
        }
    }

    result
}

/// Gram matrix A^T A.
fn gram_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut gram = vec![vec![0.0; cols]; cols];

    for i in 0..cols {
        for j in 0..cols {
            let mut sum = 0.0;

            for k in 0..rows {
                sum += matrix[k][i] * matrix[k][j];
            }

            gram[i][j] = sum;
        }
    }

    gram
}

/// Construct L^T L for the first-difference regularization operator.
fn first_difference_penalty(n: usize) -> Vec<Vec<f64>> {
    let mut penalty = vec![vec![0.0; n]; n];

    for i in 0..(n - 1) {
        penalty[i][i] += 1.0;
        penalty[i][i + 1] -= 1.0;
        penalty[i + 1][i] -= 1.0;
        penalty[i + 1][i + 1] += 1.0;
    }

    penalty
}

/// Apply the first-difference operator L to u.
fn first_difference_values(u: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(u.len().saturating_sub(1));

    for i in 0..(u.len() - 1) {
        result.push(u[i + 1] - u[i]);
    }

    result
}

/// Add lambda * B to A.
fn add_scaled_matrix(a: &[Vec<f64>], b: &[Vec<f64>], lambda: f64) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut result = vec![vec![0.0; m]; n];

    for i in 0..n {
        for j in 0..m {
            result[i][j] = a[i][j] + lambda * b[i][j];
        }
    }

    result
}

/// Add deterministic reproducible measurement noise.
fn add_deterministic_noise(data: &[f64], relative_level: f64) -> Vec<f64> {
    let data_norm = l2_norm(data);
    let n = data.len();

    data.iter()
        .enumerate()
        .map(|(i, &value)| {
            let phase = 2.0 * PI * i as f64 / (n as f64 - 1.0);
            let noise = relative_level
                * data_norm
                / (n as f64).sqrt()
                * ((9.0 * phase).sin() + 0.35 * (17.0 * phase).cos());

            value + noise
        })
        .collect()
}

/// Dense linear solve using Gaussian elimination with partial pivoting.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("The system matrix must be square and compatible with b.".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            let candidate = a[i][k].abs();

            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!(
                "Regularized matrix is numerically singular near column {}.",
                k
            ));
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

/// Solve the Tikhonov system
///
///     (K^T K + lambda L^T L)u = K^T g.
fn tikhonov_solution(
    kt_k: &[Vec<f64>],
    kt_g: &[f64],
    lt_l: &[Vec<f64>],
    lambda: f64,
) -> Result<Vec<f64>, String> {
    let regularized_matrix = add_scaled_matrix(kt_k, lt_l, lambda);
    solve_linear_system(regularized_matrix, kt_g.to_vec())
}

/// Euclidean norm.
fn l2_norm(x: &[f64]) -> f64 {
    x.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Infinity norm.
fn infinity_norm(x: &[f64]) -> f64 {
    x.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

/// Vector difference x - y.
fn difference(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y.iter()).map(|(&a, &b)| a - b).collect()
}

/// Relative infinity-norm error.
fn relative_infinity_error(x: &[f64], y: &[f64]) -> f64 {
    let numerator = infinity_norm(&difference(x, y));
    let denominator = infinity_norm(y).max(1.0e-14);

    numerator / denominator
}

/// Logarithmically spaced lambda grid.
fn lambda_grid(min_exp: i32, max_exp: i32, values_per_decade: usize) -> Vec<f64> {
    let mut values = Vec::new();

    for exponent in min_exp..=max_exp {
        for k in 0..values_per_decade {
            let fraction = k as f64 / values_per_decade as f64;
            values.push(10.0_f64.powf(exponent as f64 + fraction));
        }
    }

    values.push(10.0_f64.powi(max_exp + 1));
    values
}

/// One row of parameter-selection diagnostics.
#[derive(Clone)]
struct LambdaDiagnostic {
    lambda: f64,
    residual_norm: f64,
    regularization_norm: f64,
    relative_error: f64,
    objective: f64,
    curvature: f64,
    solution: Vec<f64>,
}

/// Approximate L-curve curvature from three neighboring points in log-log space.
///
/// The curve is parameterized by log(lambda). The x-coordinate is
/// log(||K u_lambda - g||), and the y-coordinate is log(||L u_lambda||).
fn l_curve_curvature(
    previous: &LambdaDiagnostic,
    current: &LambdaDiagnostic,
    next: &LambdaDiagnostic,
) -> f64 {
    let x1 = previous.residual_norm.max(1.0e-300).ln();
    let y1 = previous.regularization_norm.max(1.0e-300).ln();

    let x2 = current.residual_norm.max(1.0e-300).ln();
    let y2 = current.regularization_norm.max(1.0e-300).ln();

    let x3 = next.residual_norm.max(1.0e-300).ln();
    let y3 = next.regularization_norm.max(1.0e-300).ln();

    let a = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
    let b = ((x3 - x2).powi(2) + (y3 - y2).powi(2)).sqrt();
    let c = ((x3 - x1).powi(2) + (y3 - y1).powi(2)).sqrt();

    if a < 1.0e-14 || b < 1.0e-14 || c < 1.0e-14 {
        return 0.0;
    }

    let twice_area = ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)).abs();

    2.0 * twice_area / (a * b * c)
}

/// Find the index whose residual is closest to the estimated noise norm.
fn choose_by_discrepancy(diagnostics: &[LambdaDiagnostic], noise_norm: f64) -> usize {
    let mut best_index = 0;
    let mut best_distance = f64::INFINITY;

    for (i, item) in diagnostics.iter().enumerate() {
        let distance = (item.residual_norm - noise_norm).abs();

        if distance < best_distance {
            best_distance = distance;
            best_index = i;
        }
    }

    best_index
}

/// Find the index with the largest L-curve curvature.
fn choose_by_l_curve(diagnostics: &[LambdaDiagnostic]) -> usize {
    let mut best_index = 0;
    let mut best_curvature = f64::NEG_INFINITY;

    for (i, item) in diagnostics.iter().enumerate() {
        if item.curvature > best_curvature {
            best_curvature = item.curvature;
            best_index = i;
        }
    }

    best_index
}

/// Find the index with the smallest known reconstruction error.
///
/// This is used only for validation because the exact solution would not be
/// known in a real inverse problem.
fn choose_by_best_error(diagnostics: &[LambdaDiagnostic]) -> usize {
    diagnostics
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.relative_error
                .partial_cmp(&b.relative_error)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let n = 45;
    let sigma = 0.10;
    let noise_level = 2.0e-3;

    let grid = uniform_grid(a, b, n);
    let weights = trapezoidal_weights(a, b, n);

    let k_matrix = assemble_smoothing_matrix(&grid, &weights, sigma);
    let lt_l = first_difference_penalty(n);

    let exact: Vec<f64> = grid.iter().map(|&s| exact_solution(s)).collect();

    let clean_data = mat_vec(&k_matrix, &exact);
    let noisy_data = add_deterministic_noise(&clean_data, noise_level);
    let noise_vector = difference(&noisy_data, &clean_data);
    let noise_norm = l2_norm(&noise_vector);

    let kt_k = gram_matrix(&k_matrix);
    let kt_g = transpose_mat_vec(&k_matrix, &noisy_data);

    let lambdas = lambda_grid(-8, 1, 5);
    let mut diagnostics = Vec::new();

    for &lambda in &lambdas {
        let solution = match tikhonov_solution(&kt_k, &kt_g, &lt_l, lambda) {
            Ok(value) => value,
            Err(message) => {
                eprintln!("Tikhonov solve failed for lambda = {:.6e}: {}", lambda, message);
                return;
            }
        };

        let predicted = mat_vec(&k_matrix, &solution);
        let residual = difference(&predicted, &noisy_data);
        let l_values = first_difference_values(&solution);

        let residual_norm = l2_norm(&residual);
        let regularization_norm = l2_norm(&l_values);
        let relative_error = relative_infinity_error(&solution, &exact);
        let objective = residual_norm * residual_norm
            + lambda * regularization_norm * regularization_norm;

        diagnostics.push(LambdaDiagnostic {
            lambda,
            residual_norm,
            regularization_norm,
            relative_error,
            objective,
            curvature: 0.0,
            solution,
        });
    }

    for i in 1..(diagnostics.len() - 1) {
        diagnostics[i].curvature =
            l_curve_curvature(&diagnostics[i - 1], &diagnostics[i], &diagnostics[i + 1]);
    }

    let discrepancy_index = choose_by_discrepancy(&diagnostics, noise_norm);
    let l_curve_index = choose_by_l_curve(&diagnostics);
    let best_error_index = choose_by_best_error(&diagnostics);

    println!("Regularization Parameter Selection for Tikhonov Inversion");
    println!("=========================================================");
    println!();
    println!("Inverse Problem");
    println!("---------------");
    println!("Forward model               = first-kind Fredholm equation");
    println!("Regularized system          = (K^T K + lambda L^T L)u = K^T g");
    println!("Regularization operator L   = first difference");
    println!("Grid points N               = {}", n);
    println!("Kernel                      = Gaussian smoothing kernel");
    println!("Kernel width sigma          = {:.6}", sigma);
    println!("Relative data noise input   = {:.6e}", noise_level);
    println!("Estimated noise norm delta  = {:.6e}", noise_norm);
    println!();

    println!("Parameter Selection Summary");
    println!("---------------------------");
    println!(
        "Discrepancy principle lambda = {:.6e}",
        diagnostics[discrepancy_index].lambda
    );
    println!(
        "L-curve corner lambda        = {:.6e}",
        diagnostics[l_curve_index].lambda
    );
    println!(
        "Best-error lambda            = {:.6e}",
        diagnostics[best_error_index].lambda
    );
    println!();

    println!("Selected Diagnostics");
    println!("--------------------");
    println!(
        "{:>14} {:>18} {:>18} {:>18} {:>18} {:>18}",
        "lambda", "residual", "reg_norm", "objective", "rel_error", "curvature"
    );

    for (i, item) in diagnostics.iter().enumerate() {
        if i % 5 == 0
            || i == discrepancy_index
            || i == l_curve_index
            || i == best_error_index
            || i == diagnostics.len() - 1
        {
            println!(
                "{:>14.6e} {:>18.6e} {:>18.6e} {:>18.6e} {:>18.6e} {:>18.6e}",
                item.lambda,
                item.residual_norm,
                item.regularization_norm,
                item.objective,
                item.relative_error,
                item.curvature
            );
        }
    }

    println!();
    println!("Chosen Reconstructions");
    println!("----------------------");
    println!(
        "{:>24} {:>14} {:>18} {:>18} {:>18} {:>18}",
        "criterion", "lambda", "residual", "reg_norm", "objective", "rel_error"
    );

    let selected = [
        ("discrepancy", discrepancy_index),
        ("L-curve", l_curve_index),
        ("best error", best_error_index),
    ];

    for &(name, index) in &selected {
        let item = &diagnostics[index];

        println!(
            "{:>24} {:>14.6e} {:>18.6e} {:>18.6e} {:>18.6e} {:>18.6e}",
            name,
            item.lambda,
            item.residual_norm,
            item.regularization_norm,
            item.objective,
            item.relative_error
        );
    }

    println!();
    println!("Representative Values for Discrepancy-Principle Solution");
    println!("--------------------------------------------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18}",
        "i", "s_i", "u_exact", "u_lambda"
    );

    let discrepancy_solution = &diagnostics[discrepancy_index].solution;

    for i in 0..n {
        if i % 5 == 0 || i == n - 1 {
            println!(
                "{:>8} {:>14.8} {:>18.10} {:>18.10}",
                i, grid[i], exact[i], discrepancy_solution[i]
            );
        }
    }

    println!();
    println!("Interpretation");
    println!("--------------");
    println!("Small lambda values fit the noisy data too closely and may amplify");
    println!("unstable components. Large lambda values oversmooth the solution.");
    println!("The discrepancy principle chooses lambda so that the residual norm is");
    println!("close to the estimated noise level, while the L-curve diagnostic seeks");
    println!("a corner where data fidelity and regularity are balanced.");
}
```

Program 19.6.3 demonstrates that the effectiveness of Tikhonov regularization depends strongly on the choice of $\lambda$. Small values of $\lambda$ allow the solution to fit the noisy data too closely, which can reintroduce unstable components. Large values of $\lambda$ increase the residual and oversmooth the reconstruction. Useful values occur in the transition region where residual norm and regularization norm are balanced.

The discrepancy principle gives a practical rule when the noise level is known or can be estimated. In the program, it selects the value of $\lambda$ whose residual norm is closest to the computed noise norm $\delta$, matching the idea in equation (19.6.10). This prevents the reconstruction from fitting the data more accurately than the noise level justifies.

The L-curve diagnostic provides a complementary rule that does not require the exact solution. It searches for a corner in the log-log plot of residual norm against regularization norm. Near this corner, additional reductions in residual usually require a disproportionate increase in solution roughness, while stronger regularization quickly increases the residual. The selected L-curve value therefore represents a compromise between fidelity and stability.

The best-error value is included only because the test problem is synthetic and the exact solution is known. In real inverse problems, this value would not be available. Its role here is to verify that the practical selection rules choose a parameter in the same useful range. The output shows that the discrepancy principle, L-curve diagnostic, and best-error parameter all lie close to one another, making the example a clear illustration of practical regularization parameter selection.

## 19.6.5. Applications of Linear Regularization in Imaging, Tomography, and Astronomical Reconstruction

Tikhonov regularization is ubiquitous in inverse problems because it provides a simple, stable, and interpretable way to combine measured data with prior information. In *image deblurring*, the observed blurred image is modeled as:

$$g=Ku+\eta \tag{19.6.11}$$

where $u$ is the unknown sharp image, $K$ is the blur operator, and $\eta$ is noise. A regularized reconstruction solves:

$$(K^\top K+\lambda I)u=K^\top g \tag{19.6.12}$$

The $\ell_2$ penalty suppresses noise and prevents excessive amplification of high-frequency components. When more smoothness is desired, one may replace $I$ by a gradient or Laplacian operator $L$, leading to the general-form system (19.6.5).

In *electrical impedance tomography*, the goal is to reconstruct an internal conductivity distribution from boundary measurements. This inverse problem is severely ill-posed because many internal conductivity patterns produce very similar boundary responses. Regularization is therefore essential. The operator $L$ may encode smoothness, boundary behavior, or other physical constraints, and the resulting system,

$$(K^\top K+\lambda L^\top L)u=K^\top g$$

produces a stable approximation to the conductivity distribution.

Another important example is *astronomical imaging*. Telescope images are blurred by optical effects, atmospheric distortion, and instrument response. The unknown true sky brightness $u$ must be reconstructed from measured data $g$. If the blur is approximately convolutional, fast Fourier methods can be used to apply $K$ and $K^\top$ efficiently. The regularization operator $L$ may encode smoothness, non-negativity, or other prior assumptions about the brightness field. In this setting, Tikhonov regularization and its variants remain central because they provide a principled way to stabilize reconstruction while incorporating physically meaningful prior information.

Across these applications, the central idea is the same. The data term alone is not sufficient because the inverse problem is ill-posed. The regularization term supplies a priori information, while the parameter $\lambda$ controls how strongly that information shapes the solution. Linear regularization methods therefore form the computational backbone of many practical inverse problems, from deblurring and tomography to geophysical and astronomical reconstruction (Nature Research Intelligence, 2025; Pang and Wang, 2025).

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/hShFktnrqDeOCTxsP639.3","tags":[]}

# 19.7. Backus–Gilbert Method for Linear Inverse Problems and Resolution Analysis

The Backus–Gilbert method is a classical linear inverse method designed for problems in which the available data are insufficient to recover a stable pointwise representation of the unknown model. In many inverse problems, the measurements are indirect, incomplete, and contaminated by noise. As a result, attempting to reconstruct the value of the model at every point may lead to instability, nonuniqueness, or strong amplification of measurement errors. The Backus–Gilbert method addresses this difficulty by changing the object of estimation. Instead of attempting to recover the model pointwise, it constructs localized averages of the model and makes the averaging process explicit through an averaging kernel.

This shift in viewpoint is central. A Backus–Gilbert estimate is not presented as an exact value of the unknown model at a point. Rather, it is interpreted as a weighted average of the true model over a neighborhood of the target point. The quality of the estimate is therefore judged by the shape of its averaging kernel and by its sensitivity to noise. If the averaging kernel is narrow and concentrated near the target point, the estimate has good spatial resolution. If the weights used to form the estimate are large or oscillatory, the estimate may be highly sensitive to noise. The method is therefore naturally formulated as a resolution-variance trade-off: one seeks localization of the averaging kernel while controlling noise amplification.

The following subsections develop the Backus–Gilbert idea as an optimal averaging method. The discussion begins by explaining why localized averages are more meaningful than pointwise recovery in ill-posed inverse problems. It then introduces the averaging kernel, the normalization condition, and the interpretation of the Backus–Gilbert estimate as a controlled average of the true model. Later parts of the section develop the corresponding discrete formulation, the trade-off between resolution and variance, and the role of the method in applications such as seismic tomography and helioseismology.

## 19.7.1. Localized Averages Instead of Pointwise Recovery in the Backus–Gilbert Method

The Backus–Gilbert method is a linear inverse approach that replaces pointwise recovery by the construction of localized averages. This distinction is crucial in inverse problems because the data often do not contain enough stable information to determine the unknown model $m(s)$ at every point. In a well-conditioned direct problem, one might hope to compute $m(\bar{s})$ accurately at a selected target point $\bar{s}$. In an ill-posed inverse problem, however, the data may contain only smoothed, incomplete, or noisy information about $m$. Under such circumstances, a pointwise reconstruction may suggest a level of resolution that the data do not actually support.

The Backus–Gilbert method avoids this misleading interpretation. Rather than claiming to reconstruct $m(\bar{s})$ exactly at the target point $\bar{s}$, it constructs an estimate that is a controlled average of the true model near $\bar{s}$. The estimate is therefore meaningful even when exact pointwise recovery is unstable or impossible. The method also supplies the averaging kernel associated with the estimate, so the user can see which parts of the model domain contribute to the reported value. This makes the method particularly valuable in inverse problems where interpretability and resolution assessment are as important as the numerical estimate itself.

Suppose the measured data are linear functionals of the model,

$$d_i = K_i(m), \qquad i=1,\dots,N \tag{19.7.1}$$

Here, $d_i$ denotes the $i$-th measured datum, $m$ is the unknown model, and $K_i$ is the linear measurement functional associated with that datum. The notation $K_i(m)$ emphasizes that the datum is not usually a direct observation of the model at a single point. Instead, it is obtained by applying a measurement operation to the entire model or to a portion of it. This operation may smooth, average, or otherwise combine values of $m$, which is one reason that exact pointwise recovery is difficult.

In integral form, the data can be written as:

$$d_i=\int K_i(s)m(s)\,ds+\eta_i \tag{19.7.2}$$

where $K_i(s)$ is the data kernel associated with the $i$-th measurement and $\eta_i$ represents measurement noise. The kernel $K_i(s)$ describes how the value of the model at position $s$ contributes to the measured datum $d_i$. If $K_i(s)$ is broad, then $d_i$ contains information about a wide region of the model. If it is localized, then the measurement is more sensitive to a smaller region. The noise term $\eta_i$ accounts for observational error, instrument error, or other disturbances that affect the measured data.

The goal is to estimate the model near a chosen point $\bar{s}$ by forming a linear combination of the data,

$$u(\bar{s})=\sum_{i=1}^{N} w_i d_i \tag{19.7.3}$$

The coefficients $w_i$ are weights to be chosen by the method. They determine how strongly each datum contributes to the estimate at the target point $\bar{s}$. The estimate $u(\bar{s})$ is therefore linear in the observed data. This linearity is important because it makes the effect of the data kernels and the measurement noise transparent. Once the weights are fixed, the estimate is completely determined by the measured data.

Substituting equation (19.7.2) into equation (19.7.3) gives:

$$u(\bar{s}) = \sum_{i=1}^{N}w_i\int K_i(s)m(s)\,ds+\sum_{i=1}^{N}w_i\eta_i \tag{19.7.4}$$

This expression separates the estimate into two parts. The first part is the contribution from the true model through the data kernels. The second part is the contribution from measurement noise. The same weights that combine the useful data also combine the noise terms. Therefore, choosing the weights is not only a question of resolution, but also a question of stability. Weights that produce a sharply localized average may also amplify noise, while weights that suppress noise may produce a broader and less localized average.

Interchanging the sum and integral in the deterministic part of equation (19.7.4), the estimate can be written as:

$$u(\bar{s})=\int A(s;\mathbf w)m(s)\,ds,\tag{19.7.5}$$

where,

$$A(s;\mathbf w)=\sum_{i=1}^{N}w_iK_i(s)\tag{19.7.6}$$

is the averaging kernel. The vector of weights is denoted by $\mathbf w=(w_1,\dots,w_N)^\top$. Equation (19.7.5) is the key identity in the Backus–Gilbert method. It shows that the estimate $u(\bar{s})$ is not an arbitrary reconstructed value. It is explicitly an average of the true model $m(s)$, weighted by the kernel $A(s;\mathbf w)$. The averaging kernel is constructed from the same data kernels that appear in the measurement equations. Thus, the method does not hide the smoothing inherent in the inverse problem; it displays it directly.

The central aim is to choose the weights $\mathbf w$ so that $A(s;\mathbf w)$ is sharply peaked near the target point $\bar{s}$. Ideally, one would like the averaging kernel to resemble the Dirac delta function $\delta(s-\bar{s})$. If this ideal localization could be achieved exactly, then equation (19.7.5) would reduce formally to:

$$u(\bar{s})=m(\bar{s})$$

In other words, the estimate would recover the pointwise value of the model at the target point. However, exact localization is generally not possible in practical inverse problems. The available data kernels may not span such a sharply localized function, and measurement noise may make highly localized combinations unstable.

The Backus–Gilbert method therefore seeks the narrowest possible averaging kernel while controlling the noise sensitivity of the estimate. This is the fundamental compromise of the method. A narrow averaging kernel gives better resolution because the estimate is influenced mainly by model values near $\bar{s}$. A broad averaging kernel gives poorer resolution because the estimate mixes information from a larger region. On the other hand, the attempt to make $A(s;\mathbf w)$ very narrow may require weights that are large, alternating, or sensitive to small errors in the data. Such weights can amplify the noise term in equation (19.7.4). The method must therefore balance localization against variance.

To ensure that the estimate is not systematically biased by an incorrect total weight, the averaging kernel is normalized:

$$\int A(s;\mathbf w)\,ds=1 \tag{19.7.7}$$

This condition gives the averaging kernel unit total mass. It means that the estimate preserves constant models. Indeed, if the true model is constant, say $m(s)=c$, then equation (19.7.5) gives:

$$u(\bar{s})=\int A(s;\mathbf w)c\,ds = c\int A(s;\mathbf w)\,ds = c,$$

provided the normalization condition (19.7.7) holds. Thus, the Backus–Gilbert estimate returns the correct value for a constant model. In this sense, equation (19.7.7) is an unbiasedness or normalization condition. It prevents the estimate from being systematically scaled upward or downward simply because the averaging kernel has the wrong total weight.

Using equation (19.7.6), the normalization condition can also be expressed directly in terms of the weights and data kernels:

$$\int \sum_{i=1}^{N}w_iK_i(s)\,ds=1$$

Equivalently,

$$\sum_{i=1}^{N}w_i\int K_i(s)\,ds=1$$

This form shows that the normalization is a linear constraint on the weights. The Backus–Gilbert problem can therefore be understood as a constrained optimization problem: choose weights that produce a localized averaging kernel while satisfying the unit-area condition and controlling the contribution of data noise.

The interpretation of $A(s;\mathbf w)$ is especially important. The peak of the averaging kernel indicates where the estimate is most sensitive to the true model. The width of the kernel indicates the spatial or parametric resolution of the estimate. Side lobes or oscillations in the kernel indicate that the estimate may receive contributions from regions away from the target point. If the averaging kernel is strongly spread out, then $u(\bar{s})$ should not be interpreted as a highly localized value of $m$. Instead, it should be interpreted as an average over the region where $A(s;\mathbf w)$ is significant.

This explicit averaging interpretation is what distinguishes the Backus–Gilbert method from reconstruction methods that output a model without directly showing the resolution kernel associated with each estimate. In the Backus–Gilbert framework, the estimate and its resolution are inseparable. The method does not merely compute a number at $\bar{s}$; it also indicates how that number was formed from the true model through the averaging kernel. This makes the method particularly useful when the purpose of the inverse analysis is not only to produce an estimate, but also to assess what the data can actually resolve.

The method is especially natural in geophysical tomography. In such problems, the data may consist of surface observations, while the unknown model may be a subsurface material property such as seismic wave speed. The measurements are indirect because waves travel through the Earth and carry integrated information about the regions they sample. As a result, one cannot usually claim exact recovery of the wave speed at a single depth or location. The data may support only an average over a finite subsurface region.

In this setting, the Backus–Gilbert method provides an estimate together with an averaging kernel that shows which region of the Earth actually contributes to the estimate. If the averaging kernel is concentrated near the desired target location, the estimate has good local resolution. If it is broad or shifted, the estimate must be interpreted more cautiously. This is why the method remains valuable in seismic tomography and related geophysical inverse problems: it makes the limitations of the data visible rather than concealing them inside a nominal pointwise reconstruction (Fichtner, 2025).

### Rust Implementation

Following the discussion in Section 19.7.1 on localized averages instead of pointwise recovery, Program 19.7.1 provides a practical implementation of the basic Backus–Gilbert averaging-kernel construction. The program begins from the data model in equation (19.7.2), where each datum is an integral measurement of the unknown model through a broad data kernel. It then forms a linear Backus–Gilbert estimate as in equation (19.7.3), constructs the associated averaging kernel in equation (19.7.6), and verifies the normalization condition in equation (19.7.7). The example is designed to emphasize the main interpretive point of the method: the computed value is not a direct pointwise reconstruction of $m(\bar{s})$, but a localized average of the true model around the target point. The averaging kernel, its center, width, and side-lobe behavior reveal what region of the model is actually being sampled by the estimate.

At the core of the implementation is the Backus–Gilbert idea that a stable inverse estimate should be interpreted through its averaging kernel. The constants `NUM_DATA` and `NUM_GRID` specify the number of measured data functionals and the number of grid points used to represent the model interval. The function `exact_model(s)` defines a smooth reference model with a localized feature. This model is used only to generate synthetic data and to compare the pointwise value with the exact averaged value produced by the Backus–Gilbert kernel.

The function `kernel_center(i)` assigns the center of the $i$-th measurement kernel, while `data_kernel(i, s)` defines a broad Gaussian sensitivity kernel. These kernels play the role of $K_i(s)$ in equation (19.7.2). Since they are broad rather than point-localized, each datum represents information about a finite region of the model. This is precisely the situation for which the Backus–Gilbert interpretation is useful.

The functions `uniform_grid` and `trapezoidal_weights` define the numerical grid and quadrature weights on the model interval. These are used to approximate all integrals appearing in the data-generation step, the normalization condition, and the averaging-kernel diagnostics. The function `weighted_inner_product` approximates integrals of products of grid functions, while `weighted_integral` approximates the integral of a single grid function.

The function `build_kernel_samples` evaluates each data kernel on the model grid. The result is a collection of sampled kernels $K_i(s_j)$. The function `generate_clean_data` then computes synthetic data by integrating each data kernel against the exact model, corresponding to the deterministic part of equation (19.7.2). The function `add_deterministic_noise` adds a small reproducible perturbation to the clean data, representing the measurement noise term in equation (19.7.2).

The function `build_resolution_matrix` constructs the matrix used to localize the averaging kernel around the target point $\bar{s}$. Each matrix entry approximates an integral involving $(s-\bar{s})^2K_i(s)K_j(s)$. Minimizing the associated quadratic form makes the resulting averaging kernel narrow around $\bar{s}$, which corresponds to the localization goal described in Section 19.7.1.

The function `build_normalization_vector` computes the integrals of the data kernels. This vector defines the linear normalization condition on the weights. When the final weights satisfy this condition, the averaging kernel has unit total mass, as required by equation (19.7.7). This ensures that constant models are preserved and prevents the estimate from being systematically scaled upward or downward.

The functions `add_diagonal_ridge`, `dot`, and `solve_linear_system` support the computation of the Backus–Gilbert weights. A small diagonal ridge is added to the resolution matrix to stabilize the finite-dimensional solve, since broad data kernels may be nearly linearly dependent. The linear system is solved by Gaussian elimination with partial pivoting.

The function `backus_gilbert_weights` computes the data weights. It solves the constrained minimization problem in which the averaging kernel is made localized while satisfying the normalization condition. The implementation uses the closed-form constrained solution obtained by solving a stabilized linear system and then rescaling the resulting vector so that the normalization constraint holds.

The function `averaging_kernel` constructs $A(s;\mathbf w)$ from equation (19.7.6) by forming the weighted sum of the sampled data kernels. This is the central diagnostic object of the Backus–Gilbert method. The function `backus_gilbert_estimate` computes the scalar estimate $u(\bar{s})=\mathbf w^\top\mathbf d$, corresponding to equation (19.7.3).

The functions `averaging_kernel_center`, `averaging_kernel_spread`, and `negative_lobe_mass` quantify the shape of the averaging kernel. The center shows where the estimate is actually localized, the spread gives a width-like measure of resolution, and the negative lobe mass indicates whether the averaging kernel has oscillatory side lobes. These diagnostics help determine whether the estimate can be interpreted as a well-localized average near $\bar{s}$.

The function `nearest_model_value` extracts the exact model value at the grid point nearest the target. This is printed only for comparison, because the Backus–Gilbert estimate is not intended to equal this pointwise value unless the averaging kernel is extremely narrow. The function `exact_averaged_model_value` computes $\int A(s;\mathbf w)m(s)\,ds$, which is the exact model average represented by the computed averaging kernel. For clean data, this value should match the Backus–Gilbert estimate.

The functions `print_data_weights` and `print_kernel_samples` display the computed weights and representative values of the averaging kernel. These outputs make the construction transparent: large or alternating weights indicate potential noise sensitivity, while the kernel samples show how the estimate averages the true model over the domain.

The `main` function assembles the full experiment. It builds the grid, model, data kernels, clean and noisy data, resolution matrix, and normalization vector. It then computes the Backus–Gilbert weights, constructs the averaging kernel, evaluates the clean and noisy estimates, and reports normalization, center, width, negative lobe mass, and data-weight norm. These diagnostics show both the success of the averaging-kernel construction and the possible noise sensitivity caused by large oscillatory weights.

```rust
// Program 19.7.1: Constructing a Backus-Gilbert Averaging Kernel
// from Linear Data Kernels
//
// Problem statement:
// Demonstrate the basic Backus-Gilbert idea for a one-dimensional inverse
// problem. The measured data are linear functionals of an unknown model:
//
//     d_i = integral K_i(s) m(s) ds + eta_i.
//
// A Backus-Gilbert estimate at a target point s_bar is formed as
//
//     u(s_bar) = sum_i w_i d_i.
//
// The corresponding averaging kernel is
//
//     A(s; w) = sum_i w_i K_i(s).
//
// The program computes weights that produce a localized, normalized averaging
// kernel near the target point. It then reports the estimate, the averaging
// kernel normalization, the kernel center, and a width diagnostic.

use std::f64::consts::PI;

/// Number of data kernels.
const NUM_DATA: usize = 9;

/// Number of model grid points.
const NUM_GRID: usize = 201;

/// Exact model m(s).
///
/// The model contains a smooth background and a localized feature. The
/// Backus-Gilbert estimate should be interpreted as an average of this model,
/// not as exact pointwise recovery.
fn exact_model(s: f64) -> f64 {
    let background = 1.0 + 0.25 * (2.0 * PI * s).sin();
    let localized = 0.45 * (-((s - 0.62) * (s - 0.62)) / 0.006).exp();

    background + localized
}

/// Center of the i-th measurement kernel.
fn kernel_center(i: usize) -> f64 {
    0.10 + 0.80 * i as f64 / (NUM_DATA as f64 - 1.0)
}

/// Broad Gaussian measurement kernel K_i(s).
///
/// These kernels represent indirect measurements. Each datum samples a
/// finite region of the model rather than a single point.
fn data_kernel(i: usize, s: f64) -> f64 {
    let center = kernel_center(i);
    let width = 0.16;
    let distance = s - center;

    (-distance * distance / (2.0 * width * width)).exp()
}

/// Uniform grid on [a, b].
fn uniform_grid(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n < 2 {
        panic!("At least two grid points are required.");
    }

    let h = (b - a) / (n as f64 - 1.0);
    (0..n).map(|i| a + i as f64 * h).collect()
}

/// Composite trapezoidal weights on [a, b].
fn trapezoidal_weights(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);
    let mut weights = vec![h; n];

    weights[0] = 0.5 * h;
    weights[n - 1] = 0.5 * h;

    weights
}

/// Inner product approximating integral f(s) g(s) ds.
fn weighted_inner_product(f: &[f64], g: &[f64], weights: &[f64]) -> f64 {
    f.iter()
        .zip(g.iter())
        .zip(weights.iter())
        .map(|((&fi, &gi), &wi)| wi * fi * gi)
        .sum()
}

/// Integral of a grid function.
fn weighted_integral(f: &[f64], weights: &[f64]) -> f64 {
    f.iter()
        .zip(weights.iter())
        .map(|(&fi, &wi)| wi * fi)
        .sum()
}

/// Build sampled data kernels K_i(s_j).
fn build_kernel_samples(grid: &[f64]) -> Vec<Vec<f64>> {
    let mut kernels = vec![vec![0.0; grid.len()]; NUM_DATA];

    for i in 0..NUM_DATA {
        for (j, &s) in grid.iter().enumerate() {
            kernels[i][j] = data_kernel(i, s);
        }
    }

    kernels
}

/// Generate clean data d_i = integral K_i(s) m(s) ds.
fn generate_clean_data(kernels: &[Vec<f64>], model: &[f64], weights: &[f64]) -> Vec<f64> {
    kernels
        .iter()
        .map(|kernel| weighted_inner_product(kernel, model, weights))
        .collect()
}

/// Add deterministic reproducible measurement noise.
fn add_deterministic_noise(data: &[f64], relative_level: f64) -> Vec<f64> {
    let scale = data
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(1.0e-14);

    data.iter()
        .enumerate()
        .map(|(i, &value)| {
            let phase = 2.0 * PI * (i as f64 + 1.0) / data.len() as f64;
            let noise =
                relative_level * scale * ((5.0 * phase).sin() + 0.3 * (7.0 * phase).cos());
            value + noise
        })
        .collect()
}

/// Build the Backus-Gilbert localization matrix.
///
/// B_ij = integral (s - s_bar)^2 K_i(s) K_j(s) ds.
///
/// Minimizing w^T B w makes the averaging kernel localized near s_bar.
fn build_resolution_matrix(
    kernels: &[Vec<f64>],
    grid: &[f64],
    weights: &[f64],
    target: f64,
) -> Vec<Vec<f64>> {
    let n = kernels.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;

            for k in 0..grid.len() {
                let distance = grid[k] - target;
                sum += weights[k] * distance * distance * kernels[i][k] * kernels[j][k];
            }

            matrix[i][j] = sum;
        }
    }

    matrix
}

/// Compute c_i = integral K_i(s) ds.
///
/// The normalization condition is c^T w = 1.
fn build_normalization_vector(kernels: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    kernels
        .iter()
        .map(|kernel| weighted_integral(kernel, weights))
        .collect()
}

/// Add a small diagonal ridge to a matrix for numerical stability.
fn add_diagonal_ridge(matrix: &mut [Vec<f64>], ridge: f64) {
    for i in 0..matrix.len() {
        matrix[i][i] += ridge;
    }
}

/// Dot product.
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

/// Solve a dense linear system using Gaussian elimination with partial pivoting.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("The system matrix must be square and compatible with b.".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            let candidate = a[i][k].abs();

            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!("Matrix is numerically singular near column {}.", k));
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

/// Compute Backus-Gilbert weights.
///
/// We minimize w^T B w subject to c^T w = 1. The solution is
///
///     w = B^{-1} c / (c^T B^{-1} c).
///
/// A small diagonal ridge is added to B to stabilize the finite-dimensional
/// solve when the data kernels are nearly linearly dependent.
fn backus_gilbert_weights(
    resolution_matrix: &[Vec<f64>],
    normalization_vector: &[f64],
    ridge: f64,
) -> Result<Vec<f64>, String> {
    let mut stabilized_matrix = resolution_matrix.to_vec();
    add_diagonal_ridge(&mut stabilized_matrix, ridge);

    let z = solve_linear_system(stabilized_matrix, normalization_vector.to_vec())?;
    let denominator = dot(normalization_vector, &z);

    if denominator.abs() < 1.0e-14 {
        return Err("Normalization denominator is too small.".to_string());
    }

    Ok(z.iter().map(|&value| value / denominator).collect())
}

/// Construct A(s; w) = sum_i w_i K_i(s).
fn averaging_kernel(kernels: &[Vec<f64>], data_weights: &[f64]) -> Vec<f64> {
    let num_grid = kernels[0].len();
    let mut kernel = vec![0.0; num_grid];

    for i in 0..kernels.len() {
        for j in 0..num_grid {
            kernel[j] += data_weights[i] * kernels[i][j];
        }
    }

    kernel
}

/// Compute the Backus-Gilbert estimate u = w^T d.
fn backus_gilbert_estimate(data_weights: &[f64], data: &[f64]) -> f64 {
    dot(data_weights, data)
}

/// Compute the center of mass of the averaging kernel.
fn averaging_kernel_center(grid: &[f64], avg_kernel: &[f64], quad_weights: &[f64]) -> f64 {
    let mass = weighted_integral(avg_kernel, quad_weights);

    if mass.abs() < 1.0e-14 {
        return f64::NAN;
    }

    grid.iter()
        .zip(avg_kernel.iter())
        .zip(quad_weights.iter())
        .map(|((&s, &a), &w)| w * s * a)
        .sum::<f64>()
        / mass
}

/// Compute the Backus-Gilbert spread measure integral (s - target)^2 A(s)^2 ds.
fn averaging_kernel_spread(
    grid: &[f64],
    avg_kernel: &[f64],
    quad_weights: &[f64],
    target: f64,
) -> f64 {
    grid.iter()
        .zip(avg_kernel.iter())
        .zip(quad_weights.iter())
        .map(|((&s, &a), &w)| {
            let distance = s - target;
            w * distance * distance * a * a
        })
        .sum()
}

/// Compute a side-lobe measure based on the absolute negative mass.
///
/// This explicit loop avoids closure-pattern issues across Rust editions.
fn negative_lobe_mass(avg_kernel: &[f64], quad_weights: &[f64]) -> f64 {
    let mut mass = 0.0;

    for (&a, &w) in avg_kernel.iter().zip(quad_weights.iter()) {
        if a < 0.0 {
            mass += w * a.abs();
        }
    }

    mass
}

/// Compute the value of the exact model at the grid point nearest the target.
fn nearest_model_value(grid: &[f64], model: &[f64], target: f64) -> (usize, f64) {
    let mut best_index = 0;
    let mut best_distance = f64::INFINITY;

    for (i, &s) in grid.iter().enumerate() {
        let distance = (s - target).abs();

        if distance < best_distance {
            best_distance = distance;
            best_index = i;
        }
    }

    (best_index, model[best_index])
}

/// Compute the exact averaged value integral A(s) m(s) ds.
fn exact_averaged_model_value(
    avg_kernel: &[f64],
    model: &[f64],
    quad_weights: &[f64],
) -> f64 {
    weighted_inner_product(avg_kernel, model, quad_weights)
}

/// Euclidean norm of a vector.
fn l2_norm(x: &[f64]) -> f64 {
    x.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Print the data weights.
fn print_data_weights(data_weights: &[f64]) {
    println!("Backus-Gilbert Data Weights");
    println!("---------------------------");
    println!("{:>8} {:>18} {:>18}", "i", "center", "w_i");

    for i in 0..data_weights.len() {
        println!(
            "{:>8} {:>18.8} {:>18.10}",
            i + 1,
            kernel_center(i),
            data_weights[i]
        );
    }

    println!();
}

/// Print representative averaging-kernel values.
fn print_kernel_samples(grid: &[f64], avg_kernel: &[f64], target: f64) {
    println!("Representative Averaging-Kernel Values");
    println!("--------------------------------------");
    println!("{:>8} {:>14} {:>18}", "j", "s_j", "A(s_j)");

    for j in 0..grid.len() {
        if j % 25 == 0 || (grid[j] - target).abs() < 1.0e-12 || j == grid.len() - 1 {
            println!("{:>8} {:>14.8} {:>18.10}", j, grid[j], avg_kernel[j]);
        }
    }

    println!();
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let target = 0.60;
    let noise_level = 2.0e-3;
    let ridge = 1.0e-8;

    let grid = uniform_grid(a, b, NUM_GRID);
    let quad_weights = trapezoidal_weights(a, b, NUM_GRID);

    let model: Vec<f64> = grid.iter().map(|&s| exact_model(s)).collect();
    let kernels = build_kernel_samples(&grid);

    let clean_data = generate_clean_data(&kernels, &model, &quad_weights);
    let noisy_data = add_deterministic_noise(&clean_data, noise_level);

    let resolution_matrix = build_resolution_matrix(&kernels, &grid, &quad_weights, target);
    let normalization_vector = build_normalization_vector(&kernels, &quad_weights);

    let data_weights =
        match backus_gilbert_weights(&resolution_matrix, &normalization_vector, ridge) {
            Ok(value) => value,
            Err(message) => {
                eprintln!("Backus-Gilbert weight computation failed: {}", message);
                return;
            }
        };

    let avg_kernel = averaging_kernel(&kernels, &data_weights);

    let clean_estimate = backus_gilbert_estimate(&data_weights, &clean_data);
    let noisy_estimate = backus_gilbert_estimate(&data_weights, &noisy_data);
    let exact_average = exact_averaged_model_value(&avg_kernel, &model, &quad_weights);

    let normalization = weighted_integral(&avg_kernel, &quad_weights);
    let center = averaging_kernel_center(&grid, &avg_kernel, &quad_weights);
    let spread = averaging_kernel_spread(&grid, &avg_kernel, &quad_weights, target);
    let width = spread.sqrt();
    let negative_mass = negative_lobe_mass(&avg_kernel, &quad_weights);
    let weight_norm = l2_norm(&data_weights);

    let (nearest_index, point_value) = nearest_model_value(&grid, &model, target);

    println!("Backus-Gilbert Averaging Kernel Construction");
    println!("============================================");
    println!();
    println!("Problem Setup");
    println!("-------------");
    println!("Model interval              = [{:.1}, {:.1}]", a, b);
    println!("Model grid points           = {}", NUM_GRID);
    println!("Number of data kernels      = {}", NUM_DATA);
    println!("Target point s_bar          = {:.6}", target);
    println!("Measurement kernels         = broad Gaussian sensitivity kernels");
    println!("Noise level                 = {:.6e}", noise_level);
    println!("Stabilizing ridge           = {:.6e}", ridge);
    println!();
    println!("Estimator Diagnostics");
    println!("---------------------");
    println!("Averaging-kernel mass       = {:.12}", normalization);
    println!("Averaging-kernel center     = {:.12}", center);
    println!("Target point                = {:.12}", target);
    println!("Spread measure              = {:.6e}", spread);
    println!("Effective width sqrt(spread)= {:.6e}", width);
    println!("Negative lobe mass          = {:.6e}", negative_mass);
    println!("Data-weight Euclidean norm  = {:.6e}", weight_norm);
    println!();
    println!("Estimate Interpretation");
    println!("-----------------------");
    println!("Nearest grid index          = {}", nearest_index);
    println!("Point value m(s_bar) approx = {:.12}", point_value);
    println!("Exact averaged value        = {:.12}", exact_average);
    println!("BG estimate from clean data = {:.12}", clean_estimate);
    println!("BG estimate from noisy data = {:.12}", noisy_estimate);
    println!(
        "Noise effect on estimate    = {:.6e}",
        (noisy_estimate - clean_estimate).abs()
    );
    println!();

    print_data_weights(&data_weights);
    print_kernel_samples(&grid, &avg_kernel, target);

    println!("Interpretation");
    println!("--------------");
    println!("The Backus-Gilbert estimate is a weighted average of the true model,");
    println!("not a direct pointwise reconstruction. The averaging-kernel mass near");
    println!("one verifies the normalization condition, so constant models are");
    println!("preserved. The kernel center and width show where the estimate is");
    println!("actually localized and how much spatial averaging is present.");
}
```

Program 19.7.1 demonstrates the essential Backus–Gilbert interpretation: the computed estimate is a localized average of the true model, not a direct pointwise reconstruction. The normalization diagnostic verifies equation (19.7.7), since the averaging-kernel mass is equal to one. This means that constant models would be preserved by the estimator.

The comparison between the point value and the exact averaged value is especially important. The point value $m(\bar{s})$ and the averaged value $\int A(s;\mathbf w)m(s)\,ds$ need not agree. This is not a failure of the method. It reflects the fact that the data support an average over the region described by the averaging kernel, rather than an exact pointwise recovery at $\bar{s}$.

The clean-data estimate agrees with the exact averaged value because both are generated by the same averaging kernel. This confirms the internal consistency of equations (19.7.3), (19.7.5), and (19.7.6). The noisy-data estimate, however, may differ substantially if the weights are large or oscillatory. This illustrates the noise-sensitivity side of the Backus–Gilbert method.

The large data-weight norm and alternating weights should therefore be interpreted as a warning. They indicate that the estimator has achieved localization by forming a delicate combination of broad data kernels, but this same combination can amplify measurement noise. This observation prepares the reader for the next subsection, where the resolution-variance trade-off is introduced explicitly through the spread measure and the variance expression.

## 19.7.2. Resolution, Variance, and the Backus–Gilbert Trade-Off in Linear Inverse Estimation

The defining feature of the Backus–Gilbert method is its explicit trade-off between resolution and variance. This trade-off follows directly from the interpretation developed in Section 19.7.1: the estimate $u(\bar{s})$ is not a pointwise value of the unknown model, but an average of the true model weighted by the averaging kernel $A(s;\mathbf w)$. The goal is therefore not simply to compute a number at the target point $\bar{s}$, but to choose the weights $\mathbf w$ so that the resulting average is as localized, stable, and interpretable as the data allow.

A narrow averaging kernel $A(s;\mathbf w)$ gives high spatial resolution because the estimate $u(\bar{s})$ depends mainly on values of $m(s)$ near $\bar{s}$. In the ideal case, the averaging kernel would be concentrated entirely at the target point and would behave like a Dirac delta function. Then the estimate would reproduce the pointwise model value $m(\bar{s})$. In practice, this ideal cannot usually be achieved because the available data kernels $K_i(s)$ do not contain enough independent information to form an arbitrarily sharp averaging kernel. The inverse problem is limited by the coverage, geometry, bandwidth, and noise level of the data.

The attempt to sharpen the averaging kernel often requires large or oscillatory weights $w_i$. Such weights may combine the data in a way that cancels broad features of the data kernels and produces a more localized response near $\bar{s}$. However, the same large or oscillatory weights also act on the noise terms $\eta_i$ in equation (19.7.4). As a result, a highly localized estimate may become unstable because small perturbations in the data can produce large changes in $u(\bar{s})$. Conversely, smaller and more stable weights reduce noise amplification, but they usually produce a broader averaging kernel and therefore lower resolution. The Backus–Gilbert method is built around this unavoidable compromise.

A common measure of the spread of the averaging kernel around the target point (\\bar{s}) is its second moment,

$$\mathcal R(\mathbf w;\bar{s}) = \int (s-\bar{s})^2 A(s;\mathbf w)^2\,ds \tag{19.7.8}$$

This quantity measures how widely the squared averaging kernel is distributed away from $\bar{s}$. The factor $(s-\bar{s})^2$ penalizes contributions far from the target point more strongly than contributions close to it. Therefore, if $A(s;\mathbf w)$ is concentrated near $\bar{s}$, the value of $\mathcal R(\mathbf w;\bar{s})$ is small. If the averaging kernel has substantial mass far from $\bar{s}$, or if it has large side lobes away from the target point, then $\mathcal R(\mathbf w;\bar{s})$ increases.

The use of $A(s;\mathbf w)^2$ in equation (19.7.8) is also important. It measures the magnitude of the averaging kernel rather than allowing positive and negative oscillations to cancel artificially. If the averaging kernel has large oscillatory side lobes, a simple signed moment might understate the actual spread because positive and negative regions could cancel. The squared form avoids this cancellation and therefore gives a more robust measure of localization. Small values of $\mathcal R$ indicate a well-localized average, while large values indicate that the estimate receives significant contributions from regions away from the target point.

The variance of the estimate is determined by the propagation of data noise through the linear weights. If the data noise has covariance matrix $C$, then the variance of the Backus–Gilbert estimate is:

$$\operatorname{Var}\big[u(\bar{s})\big] = \mathbf w^\top C\mathbf w \tag{19.7.9}$$

This expression follows from the noise term in equation (19.7.4), namely $\sum_{i=1}^N w_i\eta_i$. The covariance matrix $C$ describes the size and correlation structure of the measurement noise. If the noise in different data components is correlated, those correlations are represented by the off-diagonal entries of $C$. The quadratic form $\mathbf w^\top C\mathbf w$ therefore measures the total noise amplification caused by the chosen weights. Large weights, or weights aligned with noisy directions in the data space, lead to a larger variance.

Equations (19.7.8) and (19.7.9) express the two competing goals of the method. The resolution measure $\mathcal R(\mathbf w;\bar{s})$ should be small so that the estimate is localized near $\bar{s}$. The variance $\mathbf w^\top C\mathbf w$ should also be small so that the estimate is stable with respect to measurement noise. In general, both objectives cannot be minimized simultaneously. Weights that improve localization may increase variance, while weights that reduce variance may broaden the averaging kernel. Thus, the Backus–Gilbert method seeks weights that localize the averaging kernel while avoiding excessive variance.

Conceptually, the method asks the following question: among all linear estimates that satisfy the required normalization or unbiasedness condition and have an acceptable noise level, which one gives the most localized average of the true model near $\bar{s}$? This question captures the practical meaning of the Backus–Gilbert approach. The method does not assume that the data can resolve arbitrary pointwise details of the model. Instead, it identifies the best localized average that is supported by the data and by the chosen tolerance for noise amplification.

This is why the method is often described as an optimal averaging method. The word “optimal” refers not to exact recovery of the unknown model, but to the construction of an averaging kernel with controlled spread and controlled variance. The estimate is optimal relative to the chosen resolution measure, noise model, and normalization constraint. If the data are informative near the target point, the method may produce a sharply localized averaging kernel. If the data are poorly informative, the averaging kernel will remain broad, making the limited resolution visible.

The averaging kernel $A(s;\mathbf w)$ is therefore not merely an internal computational object. It is a diagnostic of resolution. It shows how much of the true model contributes to the reported estimate at $\bar{s}$. If the kernel is sharply peaked near $\bar{s}$, then $u(\bar{s})$ may be interpreted as a localized estimate. If the kernel is broad, shifted, or oscillatory, then the estimate must be interpreted as an average over a wider or more complicated region. Reporting the averaging kernel alongside the estimate is therefore essential in a complete Backus–Gilbert analysis.

The variance expression plays a similarly important diagnostic role. Even if an averaging kernel appears well localized, the corresponding estimate may be unreliable if the weights produce a large value of $\mathbf w^\top C\mathbf w$. Conversely, a low-variance estimate may be stable but spatially imprecise if its averaging kernel is broad. The practical value of the Backus–Gilbert method lies in displaying this compromise explicitly rather than hiding it inside a single reconstructed model.

## 19.7.3. Discrete Backus–Gilbert Formulation and the Resulting Linear System

The discrete formulation of the Backus–Gilbert method translates the continuous averaging-kernel idea into matrix form. This is the form most often used in computation, because measured data are finite and numerical models are represented on grids, basis functions, or finite-dimensional parameter vectors. The discrete setting also makes the role of the weights, the covariance matrix, and the target averaging vector especially clear.

In the discrete linear setting, the data model may be written as:

$$\mathbf d = K\mathbf m+\boldsymbol\eta \tag{19.7.10}$$

where $\mathbf d\in\mathbb R^N$ is the data vector, $\mathbf m$ is a discrete representation of the model, $K$ is the forward matrix, and $\boldsymbol\eta$ is the noise vector. Each row of $K$ describes how the model parameters contribute to one measured datum. Thus, equation (19.7.10) is the finite-dimensional counterpart of the integral data relation in equation (19.7.2). The continuous data kernels $K_i(s)$ are replaced by the rows of the matrix $K$, and the continuous model $m(s)$ is replaced by the vector $\mathbf m$.

A Backus–Gilbert estimate at the target point $\bar{s}$ is a linear estimator of the form:

$$u(\bar{s})=\mathbf w^\top \mathbf d \tag{19.7.11}$$

The vector $\mathbf w\in\mathbb R^N$ contains the weights applied to the data. Substituting equation (19.7.10) into equation (19.7.11) gives:

$$u(\bar{s}) = \mathbf w^\top K\mathbf m+\mathbf w^\top\boldsymbol\eta$$

The first term is the deterministic contribution from the true model, while the second term is the propagated noise. This is the discrete analogue of equation (19.7.4). The weights determine both the model average being estimated and the amount of noise amplification in the estimate.

The corresponding discrete averaging kernel is:

$$\mathbf a(\mathbf w)=K^\top\mathbf w \tag{19.7.12}$$

This vector plays the same role as $A(s;\mathbf w)$ in the continuous formulation. To see this, observe that,

$$\mathbf w^\top K\mathbf m = (K^\top\mathbf w)^\top \mathbf m = \mathbf a(\mathbf w)^\top \mathbf m$$

Thus, the Backus–Gilbert estimate is not directly the value of a single component of $\mathbf m$, unless the averaging vector happens to isolate that component. Instead, it is a weighted average of the model components, with weights given by $\mathbf a(\mathbf w)$. The entries of $\mathbf a(\mathbf w)$ show which model parameters contribute to the estimate and how strongly they contribute.

Ideally, this averaging vector should approximate the unit vector or localized target vector associated with $\bar{s}$, denoted by $\mathbf e_{\bar{s}}$. If the model vector $\mathbf m$ is arranged on a grid, then $\mathbf e_{\bar{s}}$ represents the grid component corresponding to the target location. If $\mathbf a(\mathbf w)=\mathbf e_{\bar{s}}$, then:

$$\mathbf a(\mathbf w)^\top \mathbf m = \mathbf e_{\bar{s}}^\top \mathbf m,$$

which extracts the model value at the target component. This would correspond to exact pointwise recovery in the discrete setting. In practice, exact equality is often impossible or unstable, so $\mathbf a(\mathbf w)$ is interpreted as a localized discrete averaging kernel rather than an exact unit vector.

A constrained formulation of the method can therefore be written as:

$$\min_{\mathbf w}\; \mathbf w^\top C\mathbf w\quad\text{subject to}\quad K^\top\mathbf w=\mathbf e_{\bar{s}}\tag{19.7.13}$$

Here, $C$ is the data covariance matrix. The objective function $\mathbf w^\top C\mathbf w$ minimizes the variance of the estimate, matching equation (19.7.9). The constraint $K^\top\mathbf w=\mathbf e_{\bar{s}}$ enforces the desired averaging or unbiasedness condition by requiring the discrete averaging kernel to match the target vector. In this ideal constrained problem, the method searches for the minimum-variance estimator among all linear combinations of the data that exactly reproduce the target model component.

This formulation makes the role of the data geometry explicit. The constraint can be satisfied only if the target vector $\mathbf e_{\bar{s}}$ lies in the range of $K^\top$. If the available data do not contain enough information to form such a target vector, then exact localization cannot be achieved. Even when the constraint is algebraically possible, the resulting weights may have very large variance. This is why practical Backus–Gilbert formulations usually relax exact localization and instead balance localization against stability.

In practical formulations, one often uses a Lagrangian or regularized system that balances localization and variance rather than enforcing exact localization. A representative linear system has the form:

$$\left(KC^{-1}K^\top+\gamma I\right)\mathbf w=\mathbf v \tag{19.7.14}$$

where $\mathbf v$ enforces normalization or targeting, and $\gamma$ is a regularization parameter. The matrix term involving $K$ and $C^{-1}$ incorporates the forward sensitivity and the noise covariance structure, while the term $\gamma I$ stabilizes the system. The right-hand side $\mathbf v$ determines the particular target or normalization condition imposed for the estimate at $\bar{s}$.

The parameter $\gamma$ controls the resolution-variance trade-off. Its role is analogous to the regularization parameter in Tikhonov methods. Smaller values of $\gamma$ allow the weights to focus more strongly on achieving sharp localization, which may produce narrower averaging kernels. However, this can increase variance and make the estimate more sensitive to noise. Larger values of $\gamma$ penalize unstable weight combinations more strongly, producing smoother and more stable estimates, but usually at the cost of broader averaging kernels and reduced resolution.

Thus, changing $\gamma$ changes the interpretation of the estimate. With small $\gamma$, $u(\bar{s})$ may be more localized but less reliable under noisy data. With large $\gamma$, $u(\bar{s})$ may be more stable but less localized. In a full Backus–Gilbert analysis, the choice of $\gamma$ should therefore be examined through both the estimated variance and the shape of the resulting averaging kernel. The numerical value of the estimate alone is not sufficient.

Once the weights $\mathbf w$ have been computed, the estimate is obtained from:

$$u(\bar{s})=\mathbf w^\top\mathbf d,$$

as in equation (19.7.11). The corresponding discrete averaging kernel is then computed from:

$$\mathbf a(\mathbf w)=K^\top\mathbf w,$$

as in equation (19.7.12), or from the continuous expression in equation (19.7.6) when the continuous kernels are available. These two outputs serve different but complementary purposes. The estimate gives the inferred local average of the model near the target point. The averaging kernel reveals the region over which that average is actually taken and therefore determines the resolution of the estimate.

A complete Backus–Gilbert result should therefore report both the estimate and the averaging kernel. Reporting only $u(\bar{s})$ can be misleading because the number may look like a pointwise estimate even when the averaging kernel is broad. Reporting $\mathbf a(\mathbf w)$ or $A(s;\mathbf w)$ makes the actual resolving power of the data visible. If the kernel is sharply concentrated at the target location, the estimate has strong local interpretation. If the kernel is broad or oscillatory, the estimate should be interpreted as a less localized average.

The computational cost depends on the number of data points and the structure of the system used to compute the weights. If the number of data points is $N$, then solving the dense linear system for $\mathbf w$ typically requires $O(N^3)$ operations using direct linear algebra. This cubic cost arises from dense matrix factorization or equivalent direct solution procedures. For small and moderate data sets, this cost may be acceptable, especially because the method gives detailed resolution information. For large data sets, however, the cost may become substantial.

For large-scale applications, iterative methods or structured solvers may reduce the practical cost, especially when the covariance matrix or forward kernels have exploitable structure. For example, if matrix-vector products involving $K$, $K^\top$, or $C^{-1}$ can be computed efficiently, then the weights may be obtained without explicitly forming or factorizing a fully dense matrix. The essential Backus–Gilbert interpretation remains unchanged: the computed weights define an estimator, the estimator defines an averaging kernel, and the averaging kernel determines the resolution of the reported model average.

### Rust Implementation

Following the discussion in Sections 19.7.2 and 19.7.3 on resolution, variance, and the discrete Backus–Gilbert formulation, Program 19.7.2 provides a practical implementation of the resolution-variance trade-off for a one-dimensional linear inverse problem. The program extends the averaging-kernel construction of Program 19.7.1 by introducing a covariance-weighted stability term into the weight computation. For each value of the trade-off parameter $\gamma$, it computes Backus–Gilbert weights, constructs the averaging kernel, evaluates the spread measure in equation (19.7.8), and computes the propagated variance $\mathbf w^\top C\mathbf w$ from equation (19.7.9). The resulting output shows how changing (\\gamma) changes the interpretation of the estimate: small $\gamma$ emphasizes localization and gives a narrower averaging kernel, while large $\gamma$ suppresses noise-sensitive weights and produces a broader but more stable average. This makes explicit the central Backus–Gilbert principle that an estimate must be reported together with its averaging kernel and variance diagnostics, not as an isolated pointwise model value.

At the core of the implementation is the Backus–Gilbert compromise between localization and noise sensitivity. The constants `NUM_DATA` and `NUM_GRID` define the number of measured data values and the number of grid points used to represent the model domain. The function `exact_model(s)` defines a smooth model with a localized feature. As in Section 19.7.2, the goal is not to recover this model pointwise, but to estimate a localized average near the target point $\bar{s}$.

The function `kernel_center(i)` assigns the center of the $i$-th data kernel, while `data_kernel(i, s)` defines each measurement kernel $K_i(s)$. These kernels are intentionally broad, reflecting the inverse-problem setting in which each datum samples a finite region of the model. This makes exact pointwise recovery unrealistic and motivates the Backus–Gilbert averaging interpretation.

The functions `uniform_grid` and `trapezoidal_weights` construct the numerical grid and quadrature weights on the interval. The functions `weighted_integral` and `weighted_inner_product` then approximate the integrals needed for data generation, normalization, resolution measurement, and averaged model evaluation. These quadrature operations provide the discrete counterpart of the continuous integrals used in equations (19.7.8) and (19.7.9).

The function `build_kernel_samples` evaluates all measurement kernels on the grid. The function `generate_clean_data` then forms the clean data by integrating each sampled kernel against the exact model, corresponding to the deterministic part of the data model. The function `add_deterministic_noise` adds a reproducible perturbation to the data, representing measurement noise and allowing the effect of noise propagation through the Backus–Gilbert weights to be observed directly.

The function `build_resolution_matrix` constructs the matrix associated with the averaging-kernel spread measure. Its entries approximate integrals involving $(s-\bar{s})^2K_i(s)K_j(s)$, so that the quadratic form in the data weights represents the spread of the averaging kernel around the target point. This is the discrete implementation of the resolution measure in equation (19.7.8).

The function `build_normalization_vector` computes the integrals of the data kernels. This vector enforces the unit-mass constraint on the averaging kernel, ensuring that constant models are preserved. This is the same normalization idea introduced in equation (19.7.7), now used inside the resolution-variance trade-off computation.

The function `build_covariance_matrix` constructs a simple correlated noise covariance matrix $C$. The diagonal entries represent the noise variance, while the off-diagonal entries introduce correlation between nearby data components. This covariance matrix is used to compute the variance expression $\mathbf w^\top C\mathbf w$ in equation (19.7.9). Including correlation makes the example closer to practical inverse problems, where measurement errors are not always independent.

The helper functions `add_scaled_matrix`, `add_diagonal_ridge`, `mat_vec`, `dot`, `quadratic_form`, and `l2_norm` provide the linear algebra needed for the computation. The function `quadratic_form` is especially important because it evaluates the variance diagnostic $\mathbf w^\top C\mathbf w$. The dense system solver `solve_linear_system` uses Gaussian elimination with partial pivoting, which is appropriate for the small demonstration problem and mirrors the dense direct-solve setting discussed in Section 19.7.3.

The function `backus_gilbert_weights` computes the Backus–Gilbert weights for a specified value of (\\gamma). It forms the trade-off matrix by adding the resolution matrix and the covariance-weighted variance term, adds a small ridge for numerical stability, solves the constrained system, and rescales the resulting vector so that the normalization condition is satisfied. This function is the computational center of the program because it shows how the parameter (\\gamma) changes the balance between localization and stability.

The function `averaging_kernel` constructs $A(s;\mathbf w)$ by forming the weighted sum of the sampled data kernels. This averaging kernel is the resolution diagnostic associated with the estimate. The function `backus_gilbert_estimate` computes the scalar estimate from the data weights and the data vector, corresponding to equation (19.7.11).

The functions `averaging_kernel_spread`, `averaging_kernel_center`, and `negative_lobe_mass` quantify the shape of the averaging kernel. The spread corresponds to equation (19.7.8), the center indicates where the estimate is effectively localized, and the negative-lobe mass identifies oscillatory side lobes. These diagnostics are necessary because the numerical estimate alone does not reveal whether the result is sharply resolved, shifted, broad, or oscillatory.

The function `exact_averaged_model_value` computes the exact model average induced by the computed averaging kernel. This value should agree with the clean-data Backus–Gilbert estimate, confirming the interpretation of the estimate as an average of the true model. The function `nearest_model_value` reports the model value at the grid point closest to $\bar{s}$, but this is printed only for comparison because the Backus–Gilbert estimate is not intended to equal a pointwise model value unless the averaging kernel is extremely narrow.

The `TradeoffResult` structure stores all diagnostics for one value of $\gamma$: the data weights, averaging kernel, clean and noisy estimates, exact averaged value, kernel mass, center, spread, variance, negative lobe mass, and weight norm. The function `evaluate_gamma` computes these quantities for one trade-off parameter. This modular design makes the $\gamma$-sweep clear and emphasizes that each choice of $\gamma$ produces not only a different estimate, but also a different resolution and variance profile.

The `main` function assembles the full experiment. It constructs the grid, model, measurement kernels, clean and noisy data, resolution matrix, covariance matrix, and normalization vector. It then evaluates several values of $\gamma$, prints the width, variance, weight norm, kernel center, negative mass, and noise effect, and reports the corresponding clean and noisy estimates. Finally, it prints representative averaging-kernel values and data weights for a selected value of $\gamma$, allowing the reader to inspect the actual averaging behavior of the estimator.

```rust
// Program 19.7.2: Resolution-Variance Trade-Off in the Backus-Gilbert Method
//
// Problem statement:
// Demonstrate the Backus-Gilbert resolution-variance trade-off for a
// one-dimensional linear inverse problem.
//
// The measured data have the form
//
//     d_i = integral K_i(s) m(s) ds + eta_i,
//
// and the Backus-Gilbert estimate at a target point s_bar is
//
//     u(s_bar) = sum_i w_i d_i.
//
// The corresponding averaging kernel is
//
//     A(s; w) = sum_i w_i K_i(s).
//
// The weights are computed by minimizing a trade-off objective
//
//     w^T B w + gamma w^T C w,
//
// subject to the normalization constraint
//
//     c^T w = 1.
//
// Here B measures averaging-kernel spread, C is the data-noise covariance
// matrix, and gamma controls the resolution-variance compromise.

use std::f64::consts::PI;

/// Number of measured data values.
const NUM_DATA: usize = 10;

/// Number of grid points used to represent the model interval.
const NUM_GRID: usize = 241;

/// Exact model m(s).
///
/// The model contains a smooth background and a localized feature. The
/// Backus-Gilbert estimate should be interpreted as a localized average of
/// this model rather than as direct pointwise recovery.
fn exact_model(s: f64) -> f64 {
    let background = 1.0 + 0.20 * (2.0 * PI * s).sin();
    let feature = 0.40 * (-((s - 0.63) * (s - 0.63)) / 0.005).exp();

    background + feature
}

/// Center of the i-th data kernel.
fn kernel_center(i: usize) -> f64 {
    0.08 + 0.84 * i as f64 / (NUM_DATA as f64 - 1.0)
}

/// Broad Gaussian data kernel K_i(s).
///
/// Each datum samples a region of the model rather than a single point.
fn data_kernel(i: usize, s: f64) -> f64 {
    let center = kernel_center(i);
    let width = 0.15;
    let distance = s - center;

    (-distance * distance / (2.0 * width * width)).exp()
}

/// Uniform grid on [a, b].
fn uniform_grid(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n < 2 {
        panic!("At least two grid points are required.");
    }

    let h = (b - a) / (n as f64 - 1.0);
    (0..n).map(|i| a + i as f64 * h).collect()
}

/// Composite trapezoidal weights on [a, b].
fn trapezoidal_weights(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);
    let mut weights = vec![h; n];

    weights[0] = 0.5 * h;
    weights[n - 1] = 0.5 * h;

    weights
}

/// Weighted integral of a grid function.
fn weighted_integral(f: &[f64], weights: &[f64]) -> f64 {
    f.iter()
        .zip(weights.iter())
        .map(|(&fi, &wi)| wi * fi)
        .sum()
}

/// Weighted inner product approximating integral f(s)g(s) ds.
fn weighted_inner_product(f: &[f64], g: &[f64], weights: &[f64]) -> f64 {
    f.iter()
        .zip(g.iter())
        .zip(weights.iter())
        .map(|((&fi, &gi), &wi)| wi * fi * gi)
        .sum()
}

/// Build sampled kernels K_i(s_j).
fn build_kernel_samples(grid: &[f64]) -> Vec<Vec<f64>> {
    let mut kernels = vec![vec![0.0; grid.len()]; NUM_DATA];

    for i in 0..NUM_DATA {
        for (j, &s) in grid.iter().enumerate() {
            kernels[i][j] = data_kernel(i, s);
        }
    }

    kernels
}

/// Generate clean data d_i = integral K_i(s)m(s) ds.
fn generate_clean_data(
    kernels: &[Vec<f64>],
    model: &[f64],
    weights: &[f64],
) -> Vec<f64> {
    kernels
        .iter()
        .map(|kernel| weighted_inner_product(kernel, model, weights))
        .collect()
}

/// Add deterministic reproducible noise to the data.
fn add_deterministic_noise(data: &[f64], relative_level: f64) -> Vec<f64> {
    let scale = data
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(1.0e-14);

    data.iter()
        .enumerate()
        .map(|(i, &value)| {
            let phase = 2.0 * PI * (i as f64 + 1.0) / data.len() as f64;
            let noise = relative_level
                * scale
                * ((5.0 * phase).sin() + 0.4 * (7.0 * phase).cos());

            value + noise
        })
        .collect()
}

/// Build the resolution matrix B.
///
/// B_ij = integral (s - s_bar)^2 K_i(s)K_j(s) ds.
///
/// Then w^T B w equals the spread measure for A(s; w).
fn build_resolution_matrix(
    kernels: &[Vec<f64>],
    grid: &[f64],
    weights: &[f64],
    target: f64,
) -> Vec<Vec<f64>> {
    let n = kernels.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;

            for k in 0..grid.len() {
                let distance = grid[k] - target;
                sum += weights[k] * distance * distance * kernels[i][k] * kernels[j][k];
            }

            matrix[i][j] = sum;
        }
    }

    matrix
}

/// Build the normalization vector c_i = integral K_i(s) ds.
///
/// The normalization constraint is c^T w = 1.
fn build_normalization_vector(kernels: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    kernels
        .iter()
        .map(|kernel| weighted_integral(kernel, weights))
        .collect()
}

/// Build a simple correlated covariance matrix C for the data noise.
///
/// The diagonal entries represent noise variance. The off-diagonal entries
/// decay with distance between measurement indices.
fn build_covariance_matrix(noise_std: f64, correlation: f64) -> Vec<Vec<f64>> {
    let mut covariance = vec![vec![0.0; NUM_DATA]; NUM_DATA];

    for i in 0..NUM_DATA {
        for j in 0..NUM_DATA {
            let distance = i.abs_diff(j) as f64;
            covariance[i][j] = noise_std * noise_std * correlation.powf(distance);
        }
    }

    covariance
}

/// Add lambda * B to A.
fn add_scaled_matrix(a: &[Vec<f64>], b: &[Vec<f64>], lambda: f64) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut result = vec![vec![0.0; m]; n];

    for i in 0..n {
        for j in 0..m {
            result[i][j] = a[i][j] + lambda * b[i][j];
        }
    }

    result
}

/// Add a small diagonal ridge to a matrix.
fn add_diagonal_ridge(matrix: &mut [Vec<f64>], ridge: f64) {
    for i in 0..matrix.len() {
        matrix[i][i] += ridge;
    }
}

/// Matrix-vector product y = A x.
fn mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&a, &xj)| a * xj).sum())
        .collect()
}

/// Dot product.
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

/// Quadratic form x^T A x.
fn quadratic_form(matrix: &[Vec<f64>], x: &[f64]) -> f64 {
    let ax = mat_vec(matrix, x);
    dot(x, &ax)
}

/// Euclidean norm.
fn l2_norm(x: &[f64]) -> f64 {
    x.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Solve a dense linear system using Gaussian elimination with partial pivoting.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("The system matrix must be square and compatible with b.".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            let candidate = a[i][k].abs();

            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!("Matrix is numerically singular near column {}.", k));
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

/// Compute Backus-Gilbert weights for a given gamma.
///
/// The program minimizes
///
///     w^T (B + gamma C) w
///
/// subject to
///
///     c^T w = 1.
///
/// The constrained solution is
///
///     w = M^{-1}c / (c^T M^{-1}c),
///
/// where M = B + gamma C.
fn backus_gilbert_weights(
    resolution_matrix: &[Vec<f64>],
    covariance_matrix: &[Vec<f64>],
    normalization_vector: &[f64],
    gamma: f64,
    ridge: f64,
) -> Result<Vec<f64>, String> {
    let mut tradeoff_matrix = add_scaled_matrix(resolution_matrix, covariance_matrix, gamma);
    add_diagonal_ridge(&mut tradeoff_matrix, ridge);

    let z = solve_linear_system(tradeoff_matrix, normalization_vector.to_vec())?;
    let denominator = dot(normalization_vector, &z);

    if denominator.abs() < 1.0e-14 {
        return Err("Normalization denominator is too small.".to_string());
    }

    Ok(z.iter().map(|&value| value / denominator).collect())
}

/// Construct A(s; w) = sum_i w_i K_i(s).
fn averaging_kernel(kernels: &[Vec<f64>], data_weights: &[f64]) -> Vec<f64> {
    let num_grid = kernels[0].len();
    let mut kernel = vec![0.0; num_grid];

    for i in 0..kernels.len() {
        for j in 0..num_grid {
            kernel[j] += data_weights[i] * kernels[i][j];
        }
    }

    kernel
}

/// Compute the Backus-Gilbert estimate u = w^T d.
fn backus_gilbert_estimate(data_weights: &[f64], data: &[f64]) -> f64 {
    dot(data_weights, data)
}

/// Compute the averaging-kernel spread integral.
fn averaging_kernel_spread(
    grid: &[f64],
    avg_kernel: &[f64],
    quad_weights: &[f64],
    target: f64,
) -> f64 {
    grid.iter()
        .zip(avg_kernel.iter())
        .zip(quad_weights.iter())
        .map(|((&s, &a), &w)| {
            let distance = s - target;
            w * distance * distance * a * a
        })
        .sum()
}

/// Compute the center of mass of the averaging kernel.
fn averaging_kernel_center(grid: &[f64], avg_kernel: &[f64], quad_weights: &[f64]) -> f64 {
    let mass = weighted_integral(avg_kernel, quad_weights);

    if mass.abs() < 1.0e-14 {
        return f64::NAN;
    }

    grid.iter()
        .zip(avg_kernel.iter())
        .zip(quad_weights.iter())
        .map(|((&s, &a), &w)| w * s * a)
        .sum::<f64>()
        / mass
}

/// Compute the absolute negative mass of the averaging kernel.
fn negative_lobe_mass(avg_kernel: &[f64], quad_weights: &[f64]) -> f64 {
    let mut mass = 0.0;

    for (&a, &w) in avg_kernel.iter().zip(quad_weights.iter()) {
        if a < 0.0 {
            mass += w * a.abs();
        }
    }

    mass
}

/// Exact averaged value integral A(s)m(s) ds.
fn exact_averaged_model_value(
    avg_kernel: &[f64],
    model: &[f64],
    quad_weights: &[f64],
) -> f64 {
    weighted_inner_product(avg_kernel, model, quad_weights)
}

/// Point value of the exact model at the nearest grid point.
fn nearest_model_value(grid: &[f64], model: &[f64], target: f64) -> (usize, f64) {
    let mut best_index = 0;
    let mut best_distance = f64::INFINITY;

    for (i, &s) in grid.iter().enumerate() {
        let distance = (s - target).abs();

        if distance < best_distance {
            best_distance = distance;
            best_index = i;
        }
    }

    (best_index, model[best_index])
}

/// Diagnostics for one value of gamma.
struct TradeoffResult {
    gamma: f64,
    weights: Vec<f64>,
    averaging_kernel: Vec<f64>,
    clean_estimate: f64,
    noisy_estimate: f64,
    exact_average: f64,
    mass: f64,
    center: f64,
    spread: f64,
    variance: f64,
    negative_mass: f64,
    weight_norm: f64,
}

/// Compute all diagnostics for one gamma value.
fn evaluate_gamma(
    gamma: f64,
    resolution_matrix: &[Vec<f64>],
    covariance_matrix: &[Vec<f64>],
    normalization_vector: &[f64],
    kernels: &[Vec<f64>],
    grid: &[f64],
    quad_weights: &[f64],
    model: &[f64],
    clean_data: &[f64],
    noisy_data: &[f64],
    target: f64,
    ridge: f64,
) -> Result<TradeoffResult, String> {
    let weights = backus_gilbert_weights(
        resolution_matrix,
        covariance_matrix,
        normalization_vector,
        gamma,
        ridge,
    )?;

    let avg_kernel = averaging_kernel(kernels, &weights);

    let clean_estimate = backus_gilbert_estimate(&weights, clean_data);
    let noisy_estimate = backus_gilbert_estimate(&weights, noisy_data);
    let exact_average = exact_averaged_model_value(&avg_kernel, model, quad_weights);

    let mass = weighted_integral(&avg_kernel, quad_weights);
    let center = averaging_kernel_center(grid, &avg_kernel, quad_weights);
    let spread = averaging_kernel_spread(grid, &avg_kernel, quad_weights, target);
    let variance = quadratic_form(covariance_matrix, &weights);
    let negative_mass = negative_lobe_mass(&avg_kernel, quad_weights);
    let weight_norm = l2_norm(&weights);

    Ok(TradeoffResult {
        gamma,
        weights,
        averaging_kernel: avg_kernel,
        clean_estimate,
        noisy_estimate,
        exact_average,
        mass,
        center,
        spread,
        variance,
        negative_mass,
        weight_norm,
    })
}

/// Print representative values of one averaging kernel.
fn print_kernel_samples(grid: &[f64], avg_kernel: &[f64], target: f64) {
    println!("Representative Averaging-Kernel Values for Selected Gamma");
    println!("--------------------------------------------------------");
    println!("{:>8} {:>14} {:>18}", "j", "s_j", "A(s_j)");

    for j in 0..grid.len() {
        if j % 30 == 0 || (grid[j] - target).abs() < 1.0e-12 || j == grid.len() - 1 {
            println!("{:>8} {:>14.8} {:>18.10}", j, grid[j], avg_kernel[j]);
        }
    }

    println!();
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let target = 0.60;

    let noise_level = 2.0e-3;
    let noise_std = 2.0e-3;
    let covariance_correlation = 0.30;
    let ridge = 1.0e-10;

    let gamma_values = [0.0, 1.0e2, 1.0e4, 1.0e6, 1.0e8];

    let grid = uniform_grid(a, b, NUM_GRID);
    let quad_weights = trapezoidal_weights(a, b, NUM_GRID);

    let model: Vec<f64> = grid.iter().map(|&s| exact_model(s)).collect();
    let kernels = build_kernel_samples(&grid);

    let clean_data = generate_clean_data(&kernels, &model, &quad_weights);
    let noisy_data = add_deterministic_noise(&clean_data, noise_level);

    let resolution_matrix = build_resolution_matrix(&kernels, &grid, &quad_weights, target);
    let covariance_matrix = build_covariance_matrix(noise_std, covariance_correlation);
    let normalization_vector = build_normalization_vector(&kernels, &quad_weights);

    let mut results = Vec::new();

    for &gamma in &gamma_values {
        let result = match evaluate_gamma(
            gamma,
            &resolution_matrix,
            &covariance_matrix,
            &normalization_vector,
            &kernels,
            &grid,
            &quad_weights,
            &model,
            &clean_data,
            &noisy_data,
            target,
            ridge,
        ) {
            Ok(value) => value,
            Err(message) => {
                eprintln!("Backus-Gilbert solve failed for gamma = {:.6e}: {}", gamma, message);
                return;
            }
        };

        results.push(result);
    }

    let (nearest_index, point_value) = nearest_model_value(&grid, &model, target);

    println!("Backus-Gilbert Resolution-Variance Trade-Off");
    println!("============================================");
    println!();
    println!("Problem Setup");
    println!("-------------");
    println!("Model interval              = [{:.1}, {:.1}]", a, b);
    println!("Model grid points           = {}", NUM_GRID);
    println!("Number of data kernels      = {}", NUM_DATA);
    println!("Target point s_bar          = {:.6}", target);
    println!("Nearest grid index          = {}", nearest_index);
    println!("Point value m(s_bar) approx = {:.12}", point_value);
    println!("Noise level in data         = {:.6e}", noise_level);
    println!("Covariance noise std        = {:.6e}", noise_std);
    println!("Covariance correlation      = {:.6}", covariance_correlation);
    println!("Stabilizing ridge           = {:.6e}", ridge);
    println!();

    println!("Trade-Off Diagnostics");
    println!("---------------------");
    println!(
        "{:>12} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}",
        "gamma", "width", "variance", "w_norm", "center", "neg_mass", "noise_eff"
    );

    for item in &results {
        let width = item.spread.sqrt();
        let noise_effect = (item.noisy_estimate - item.clean_estimate).abs();

        println!(
            "{:>12.3e} {:>14.6e} {:>14.6e} {:>14.6e} {:>14.8} {:>14.6e} {:>14.6e}",
            item.gamma,
            width,
            item.variance,
            item.weight_norm,
            item.center,
            item.negative_mass,
            noise_effect
        );
    }

    println!();
    println!("Estimate Diagnostics");
    println!("--------------------");
    println!(
        "{:>12} {:>18} {:>18} {:>18} {:>18}",
        "gamma", "exact_average", "clean_estimate", "noisy_estimate", "kernel_mass"
    );

    for item in &results {
        println!(
            "{:>12.3e} {:>18.10} {:>18.10} {:>18.10} {:>18.10}",
            item.gamma,
            item.exact_average,
            item.clean_estimate,
            item.noisy_estimate,
            item.mass
        );
    }

    println!();

    let selected_index = 2;
    println!(
        "Selected gamma for kernel samples = {:.6e}",
        results[selected_index].gamma
    );
    println!();

    print_kernel_samples(
        &grid,
        &results[selected_index].averaging_kernel,
        target,
    );

    println!("Selected Data Weights");
    println!("---------------------");
    println!("{:>8} {:>18} {:>18}", "i", "center", "w_i");

    for i in 0..NUM_DATA {
        println!(
            "{:>8} {:>18.8} {:>18.10}",
            i + 1,
            kernel_center(i),
            results[selected_index].weights[i]
        );
    }

    println!();
    println!("Interpretation");
    println!("--------------");
    println!("Small gamma values emphasize localization and can produce narrow");
    println!("averaging kernels, but they may require large data weights and can");
    println!("therefore amplify measurement noise. Large gamma values penalize");
    println!("noise-sensitive weights more strongly, reducing variance and weight");
    println!("norm at the cost of broader averaging kernels and weaker resolution.");
}
```

Program 19.7.2 demonstrates the central trade-off in the Backus–Gilbert method. When $\gamma$ is small, the optimization emphasizes localization. The resulting averaging kernel is narrower, but the data weights may become large or oscillatory. This increases the variance $\mathbf w^\top C\mathbf w$ and makes the estimate sensitive to measurement noise.

As $\gamma$ increases, the covariance term penalizes noise-sensitive weight combinations more strongly. The weight norm and propagated variance decrease, and the noisy estimate becomes much closer to the clean estimate. The cost is a broader averaging kernel, a shifted kernel center, or weaker localization near the target point. Thus, the method gains stability by sacrificing resolution.

The output also confirms the normalization property of the estimator. The kernel mass remains equal to one for each value of $\gamma$, so the estimates preserve constant models. This is important because the comparison across $\gamma$ values should reflect changes in localization and variance, not changes in the overall scaling of the averaging kernel.

The selected averaging-kernel values and data weights show why the numerical estimate alone is insufficient. A stable value of $u(\bar{s})$ must be interpreted together with the width, center, side-lobe behavior, and variance of its averaging kernel. This is the main diagnostic advantage of the Backus–Gilbert method: it reports not only an estimate, but also the resolution and noise sensitivity that determine what the estimate actually means.

## 19.7.4. Comparison Between the Backus–Gilbert Method and Parametric Least-Squares Inversion

The Backus–Gilbert method differs from conventional least-squares inversion in both its objective and its interpretation. Least-squares inversion usually seeks a model that explains the observed data as well as possible within a chosen finite-dimensional parametrization. The Backus–Gilbert method instead asks what localized average of the true model can be estimated stably from the data. This difference is important because many inverse problems are not limited only by numerical error, but also by incomplete data, measurement noise, and restricted sensitivity of the observations. In such situations, a reconstructed model may appear more detailed than the data actually justify. The Backus–Gilbert method avoids this difficulty by making the averaging process explicit.

In least-squares inversion, one usually begins by choosing a representation of the unknown model. This may involve selecting a grid, a set of basis functions, or a finite collection of parameters. The continuous unknown model is then replaced by a finite-dimensional vector of coefficients. The inverse problem is solved by minimizing a data misfit, often supplemented by a regularization term. A typical least-squares formulation therefore depends on the chosen parametrization, the form of the misfit, the regularization operator, and the regularization strength. These choices are necessary in many practical computations, but they also influence the final reconstruction.

The resulting least-squares model is often interpreted as a global reconstruction. That is, the solution vector is treated as an estimate of the unknown model over the full domain. However, the apparent resolution of such a model can be misleading. Fine-scale variations may appear in the reconstructed model because of the chosen grid or basis, but the data may not actually support those variations. If the inverse problem is ill-conditioned, small perturbations in the data may produce large changes in the estimated coefficients unless regularization is imposed. The final result is therefore shaped by both the data and the regularization assumptions.

Backus–Gilbert inversion is more non-parametric in spirit. It does not primarily seek a global model vector that explains all data. Instead, it constructs local averages of the model directly from the data. At each target point, the method chooses weights so that the resulting averaging kernel is as localized as possible while keeping the estimate sufficiently stable. The output is therefore not simply a model value, but a pair consisting of the estimate and its averaging kernel. The estimate gives the inferred average, while the averaging kernel shows what region of the true model contributes to that estimate.

This distinction changes the interpretation of the inverse result. In a least-squares reconstruction, resolution is often assessed after the model has been computed, for example by examining resolution matrices, uncertainty estimates, or sensitivity tests. In the Backus–Gilbert method, resolution is built directly into the construction of the estimator. The method chooses the weights with the shape of the averaging kernel in mind. Consequently, the resolution of the estimate is transparent from the beginning, because the averaging kernel is an explicit part of the solution.

This feature is especially valuable in scientific interpretation. A least-squares model may appear to contain fine-scale structure, but such structure may not actually be resolved by the data. For example, a reconstructed profile may show sharp changes or localized anomalies, even though the data kernels are broad and cannot distinguish such small-scale features reliably. In contrast, Backus–Gilbert analysis reveals the spatial averaging inherent in each estimate. If the averaging kernel is broad, then the data support only a coarse average. If the averaging kernel is narrow and well centered at the target point, then the data support a more localized interpretation.

The difference can also be stated in terms of the object being optimized. Least-squares inversion usually minimizes data misfit, possibly with a stabilizing regularization term. The Backus–Gilbert method optimizes the estimator itself by balancing the spread of the averaging kernel against the variance of the estimate. In least squares, the central question is: which model best fits the data under the chosen parametrization and regularization? In the Backus–Gilbert method, the central question is: which linear combination of the data gives the most localized and stable average of the true model near the target point?

This does not mean that the Backus–Gilbert method eliminates all modeling choices. The method still requires choices such as the spread measure, the covariance model, the normalization condition, and the trade-off between resolution and variance. However, these choices are expressed in terms of the estimator and its averaging kernel rather than through a full reconstructed model. This makes the consequences of the choices easier to inspect. If a chosen trade-off produces an averaging kernel that is too broad or too oscillatory, this limitation is visible directly.

Modern developments extend the method beyond its classical form. Fichtner (2025) reformulates the Backus–Gilbert method in a probabilistic multiparameter framework, allowing arbitrary prior information to be incorporated and enabling joint inversion for multiple parameters. This probabilistic view connects Backus–Gilbert averaging with Bayesian inverse theory and gives a broader interpretation of uncertainty, prior information, and resolution (Fichtner, 2025). In this setting, the averaging-kernel viewpoint remains central, but it is embedded in a broader framework where prior knowledge and probabilistic uncertainty can be treated more systematically.

The multiparameter setting is important because many inverse problems involve more than one physical quantity. In geophysical applications, for example, different kinds of data may be sensitive to different combinations of subsurface parameters. A classical single-parameter Backus–Gilbert formulation may not fully express this coupling. A probabilistic multiparameter formulation allows the method to account for relationships among parameters, incorporate prior information, and describe uncertainty in a way that is consistent with the broader inverse problem. The key idea remains that the recovered quantity should be interpreted through its averaging behavior and uncertainty, rather than as an exact pointwise value.

A closely related generalization is the Subtractive Optimally Localized Averages method, commonly abbreviated as SOLA. In SOLA, the user specifies a desired target kernel, and the method chooses weights so that the resulting averaging kernel approximates this target while controlling noise. This differs from a formulation that simply minimizes a generic spread measure around the target point. Instead of asking only for the narrowest possible kernel, SOLA asks for a kernel that resembles a prescribed shape. This is useful when one wants the averaging kernel to have a specific width, symmetry, or localization pattern.

The SOLA viewpoint is particularly useful when interpretability requires a known averaging shape. For example, one may want all estimates at different target locations to have comparable resolution, or one may want to avoid averaging kernels with strong side lobes. By specifying a target kernel, the user can guide the estimator toward a desired resolution pattern while still controlling the variance contributed by data noise. Thus, SOLA preserves the essential Backus–Gilbert philosophy: the estimate must be interpreted through the averaging kernel, and the trade-off between localization and noise must remain explicit.

Overall, the comparison with parametric least-squares methods shows why the Backus–Gilbert method occupies a distinctive place in inverse theory. Least-squares methods emphasize fitting the data through a chosen model parametrization. Backus–Gilbert methods emphasize what the data can resolve locally and stably. Least-squares inversion often produces a global model, while Backus–Gilbert inversion produces localized averages with explicit resolution kernels. Both approaches can be useful, but the Backus–Gilbert method is especially valuable when the main concern is honest interpretation of limited, noisy, or indirect data.

## 19.7.5. Applications of the Backus–Gilbert Method in Seismic Tomography and Helioseismology

The Backus–Gilbert method is particularly important in scientific inverse problems where the data provide indirect information about inaccessible regions. Two major examples are seismic tomography and helioseismology. In both areas, the unknown physical structure cannot be measured directly at every point. Instead, the available observations are integrated or averaged responses of the system. The Backus–Gilbert method is well suited to such problems because it constructs estimates together with averaging kernels, allowing researchers to determine what part of the model is actually resolved by the data.

A primary application of the Backus–Gilbert method is seismic tomography. In seismic surface-wave studies, the data $d_i$ may consist of dispersion measurements, such as wave speeds at different frequencies. These measurements are sensitive to subsurface structure, but they do not sample all depths equally. Waves of different frequencies penetrate to different depths and therefore provide different kinds of information about the Earth’s interior. The unknown model $m(z)$ may be the Earth’s shear-wave speed as a function of depth $z$.

Because different waves sample different depth ranges, the data do not determine the shear-wave profile pointwise. A surface-wave measurement is not usually a direct observation of the wave speed at one exact depth. Instead, it reflects a weighted contribution from a range of depths. Therefore, attempting to interpret the inversion result as an exact value at depth $z_0$ can be misleading. The Backus–Gilbert method avoids this by constructing an estimate:

$$u(z_0)=\sum_i w_i d_i\tag{19.7.15}$$

that represents a weighted average of the true shear-wave speed around the target depth $z_0$.

The weights $w_i$ are chosen so that the corresponding averaging kernel is concentrated near $z_0$, while the noise sensitivity remains controlled. The estimate $u(z_0)$ should therefore be interpreted as a localized depth average, not necessarily as the exact value of $m(z_0)$. This distinction is essential in seismic interpretation. If the data have poor sensitivity at a certain depth, the averaging kernel may become broad, shifted, or oscillatory, indicating that the reported estimate at $z_0$ is influenced by a wider depth interval.

The associated averaging kernel shows the depth interval over which the estimate is meaningful. A well-localized kernel centered near $z_0$ indicates good depth resolution. In that case, the estimate can be interpreted as a reliable average near the target depth. A broad kernel indicates that the estimate represents an average over a larger depth range. A shifted kernel indicates that the data are more sensitive to depths away from the nominal target. Side lobes may show that the estimate receives non-negligible contributions from separated depth intervals. These features are not defects in the reporting of the method; rather, they are precisely the resolution information that the Backus–Gilbert approach is designed to expose.

Thus, averaging kernels provide a direct tool for assessing resolution and uncertainty in seismic interpretation. They help determine whether an inferred feature, such as a velocity increase, velocity decrease, or transition zone, is genuinely supported by the data at the stated depth. If the averaging kernel is broad, then a sharp feature in the estimate should not be overinterpreted as a sharply localized physical structure. If the averaging kernel is narrow, the corresponding feature has stronger local support from the data.

Rayleigh-wave inversions often use this type of analysis to study how wave speed varies with depth and to determine which features of the inferred profile are actually resolved by the data. Since Rayleigh waves are sensitive to elastic structure over depth-dependent ranges, the Backus–Gilbert method provides a natural way to translate dispersion measurements into interpretable depth averages. The method therefore supports both estimation and resolution assessment, which are inseparable in reliable seismic tomography.

A second important application is helioseismology, where the goal is to infer the internal structure or rotation profile of the Sun from observed oscillation frequencies. The solar interior cannot be directly sampled, but oscillation modes carry information about internal properties. These observations are indirect, and their sensitivity is distributed across regions of the solar interior. As in seismic tomography, the inverse problem is therefore not simply a matter of recovering a pointwise function. It is a problem of determining what localized averages of the interior are supported by the observed data.

In this setting, the unknown $m(r)$ may represent the solar rotation rate as a function of radius $r$. Observed oscillation data provide indirect information through kernels that sample broad regions of the solar interior. A measurement may be sensitive to a range of radii rather than to a single radius. The Backus–Gilbert method constructs a linear combination of the observed data so that the resulting averaging kernel is concentrated near a target radius $r_0$. The estimate has the form:

$$u(r_0)=\int A(r;\mathbf w)m(r)\,dr \tag{19.7.16}$$

where the averaging kernel $A(r;\mathbf w)$ is designed to be localized near $r_0$.

Equation (19.7.16) shows the same central interpretation as equation (19.7.5): the estimated quantity is an average of the true model, not automatically a pointwise value. In helioseismology, this interpretation is crucial because different oscillation modes probe different parts of the solar interior with different sensitivities. A well-designed averaging kernel allows the researcher to say that the estimate represents the rotation rate averaged over a particular radial region. Without the averaging kernel, the estimate might be mistaken for a more localized measurement than the data justify.

Researchers examine these averaging kernels to determine whether the inferred rotational averages are localized enough to support physical interpretation. If the kernel is well localized, the estimate can be interpreted as a reliable average near $r_0$. If the kernel is broad, the estimate reflects a larger region of the solar interior. If the kernel has substantial side lobes, the estimate may include influence from radii away from the target. In each case, the averaging kernel provides the resolution information needed to interpret the estimate responsibly.

This use of averaging kernels is especially important when comparing inferred solar structure or rotation across different radii. If two target radii have broad and overlapping averaging kernels, then the corresponding estimates may not represent independent local features. If the kernels are narrow and centered at distinct locations, the estimates can support a more localized interpretation. Thus, the Backus–Gilbert method provides not only estimates of internal solar properties, but also a way to judge how much spatial detail the data can resolve.

In both seismic tomography and helioseismology, the Backus–Gilbert method provides a disciplined framework for inverse interpretation. The estimate gives a localized average of the unknown model, while the averaging kernel shows the region that contributes to that average. The variance or covariance information describes the uncertainty associated with the data noise. Together, these quantities prevent overinterpretation of inverse results. They make clear whether a reported estimate represents a sharply resolved local property or a broader average imposed by the limitations of the data.

This disciplined treatment of measurement covariance, averaging, and uncertainty is one of the reasons the method remains important in modern inverse theory (Fichtner, 2025). Its continuing value lies in the fact that it does not hide the limitations of the data. Instead, it builds those limitations into the estimator and reports them through the averaging kernel and variance. For scientific applications where interpretation matters as much as computation, this explicit resolution analysis is often more informative than a single reconstructed model.

### Rust Implementation

Following the comparison in Sections 19.7.4 and 19.7.5 between Backus–Gilbert inversion and parametric least-squares reconstruction, Program 19.7.3 provides a practical implementation of both approaches on the same one-dimensional inverse problem. The least-squares method produces a full grid-based reconstruction by fitting the data with a regularized finite-dimensional model, while the Backus–Gilbert method produces localized estimates together with their averaging-kernel diagnostics. This distinction reflects the central point of Section 19.7.4: a least-squares model may look like a pointwise reconstruction, but its apparent spatial detail can exceed what the data truly resolve. By contrast, the Backus–Gilbert estimate is explicitly interpreted as a localized average, and the associated averaging kernel shows which region of the true model contributes to each reported value.

At the core of the implementation is a comparison between two different interpretations of the same inverse data. The constants `NUM_DATA` and `NUM_GRID` define the number of measured data values and the number of model-grid points. The function `exact_model(s)` defines a reference model containing a smooth background and a localized anomaly. This model is used to generate synthetic data and to compare the behavior of least-squares reconstruction with Backus–Gilbert averaging.

The function `kernel_center(i)` assigns the center of the $i$-th measurement kernel, while `data_kernel(i, s)` defines a broad Gaussian sensitivity kernel. These kernels represent indirect measurements, similar to the broad sensitivity kernels discussed in seismic tomography and helioseismology. Because each measurement samples a region rather than a single point, the data do not directly determine pointwise values of the model.

The functions `uniform_grid` and `trapezoidal_weights` build the numerical grid and quadrature weights on the model interval. The functions `weighted_integral` and `weighted_inner_product` approximate the integrals needed to generate synthetic data, normalize averaging kernels, and compute exact Backus–Gilbert averages. These quadrature-based operations connect the continuous averaging interpretation to the finite-dimensional computation.

The function `build_kernel_samples` evaluates each measurement kernel on the grid. The function `build_weighted_forward_matrix` then constructs the weighted forward matrix used by least-squares inversion, with quadrature weights included in the matrix entries. This allows matrix-vector multiplication to approximate the integral data relation.

The function `generate_clean_data` computes synthetic data by integrating the exact model against the measurement kernels. The function `add_deterministic_noise` adds a reproducible perturbation to the clean data. This produces the noisy data vector used by both the least-squares reconstruction and the Backus–Gilbert estimates.

The linear algebra functions `mat_vec`, `transpose_mat_vec`, and `gram_matrix` support the least-squares computation. The function `first_difference_penalty` constructs the matrix $L^\top L$ for a first-difference regularization operator. This imposes smoothness on the least-squares reconstruction, suppressing excessive oscillations in the recovered grid-based model.

The function `regularized_least_squares` solves the regularized least-squares problem. It forms the normal-equation matrix from the forward operator and adds the first-difference Tikhonov penalty. The resulting solution is a full model vector on the grid. This is the usual output of a parametric inverse method: it gives an estimated value at every grid point, but the apparent pointwise interpretation must be treated cautiously when the data kernels are broad.

The function `build_bg_resolution_matrix` constructs the Backus–Gilbert resolution matrix for a target point. This matrix measures how widely the resulting averaging kernel spreads around the target. The function `build_normalization_vector` computes the integrals of the measurement kernels and supplies the normalization constraint that forces the averaging kernel to have unit mass. The function `build_covariance_matrix` constructs a simple diagonal noise covariance matrix for variance evaluation.

The function `backus_gilbert_weights` computes the Backus–Gilbert data weights for a given target. It combines the localization matrix with the covariance-weighted variance term, adds a small ridge for numerical stability, solves the resulting system, and rescales the solution so that the averaging kernel is normalized. These weights define the local linear estimator.

The function `averaging_kernel` forms the Backus–Gilbert averaging kernel by taking the weighted sum of the sampled measurement kernels. This kernel is the main interpretive object of the method. The function `backus_gilbert_estimate` computes the scalar estimate by applying the data weights to the data vector.

The functions `averaging_kernel_spread` and `averaging_kernel_center` quantify the resolution of the Backus–Gilbert estimate. The spread gives a width-like measure of the averaging kernel, while the center indicates where the estimate is actually localized. The function `exact_averaged_model_value` computes the exact model average induced by the averaging kernel, which is the quantity the Backus–Gilbert estimate is designed to approximate.

The function `nearest_grid_value` extracts the point value of either the exact model or the least-squares model at the grid point closest to a target. This permits a direct comparison between a pointwise least-squares interpretation and a Backus–Gilbert averaged interpretation. The helper functions `l2_norm`, `infinity_norm`, and `relative_infinity_error` compute reconstruction and residual diagnostics.

The `BgTargetResult` structure stores all diagnostics for one target point, including the exact point value, least-squares value, Backus–Gilbert clean and noisy estimates, exact average, averaging-kernel center, width, mass, variance, and weight norm. The function `evaluate_backus_gilbert_target` computes all of these quantities for a single target point.

The `main` function assembles the full comparison. It constructs the grid, exact model, measurement kernels, forward matrix, clean data, noisy data, and least-squares reconstruction. It then evaluates Backus–Gilbert estimates at three target points. The output compares the exact point value, the least-squares model value, the Backus–Gilbert noisy estimate, and the exact Backus–Gilbert average. It also reports averaging-kernel diagnostics so that each Backus–Gilbert estimate can be interpreted in terms of its actual resolution.

```rust
// Program 19.7.3: Backus-Gilbert Local Averages Versus Regularized
// Least-Squares Reconstruction
//
// Problem statement:
// Compare two approaches for a one-dimensional linear inverse problem:
//
// 1. Regularized parametric least-squares inversion, which reconstructs a
//    full model vector on a grid.
//
// 2. Backus-Gilbert localized averaging, which estimates local averages and
//    reports the corresponding averaging kernels.
//
// The data are generated from
//
//     d_i = integral K_i(s) m(s) ds + eta_i.
//
// The least-squares reconstruction solves
//
//     (K^T K + lambda L^T L)m = K^T d,
//
// where L is a first-difference operator. The Backus-Gilbert estimator at a
// target point s_bar forms
//
//     u(s_bar) = sum_i w_i d_i,
//
// and reports the averaging kernel
//
//     A(s; w) = sum_i w_i K_i(s).
//
// The comparison illustrates that a least-squares model may appear to give a
// pointwise reconstruction, whereas Backus-Gilbert explicitly reports what
// localized average is actually supported by the data.

use std::f64::consts::PI;

/// Number of measured data values.
const NUM_DATA: usize = 12;

/// Number of model-grid points.
const NUM_GRID: usize = 81;

/// Exact model m(s).
///
/// The model contains a smooth background and a localized anomaly.
/// A least-squares reconstruction tries to recover this model globally,
/// while Backus-Gilbert reports localized averages.
fn exact_model(s: f64) -> f64 {
    let background = 1.0 + 0.20 * (2.0 * PI * s).sin();
    let anomaly = 0.45 * (-((s - 0.62) * (s - 0.62)) / 0.004).exp();

    background + anomaly
}

/// Center of the i-th data kernel.
fn kernel_center(i: usize) -> f64 {
    0.06 + 0.88 * i as f64 / (NUM_DATA as f64 - 1.0)
}

/// Broad Gaussian data kernel K_i(s).
///
/// These kernels represent indirect measurements, such as depth-sensitive
/// observations in a geophysical inverse problem.
fn data_kernel(i: usize, s: f64) -> f64 {
    let center = kernel_center(i);
    let width = 0.13;
    let distance = s - center;

    (-distance * distance / (2.0 * width * width)).exp()
}

/// Uniform grid on [a, b].
fn uniform_grid(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n < 2 {
        panic!("At least two grid points are required.");
    }

    let h = (b - a) / (n as f64 - 1.0);
    (0..n).map(|i| a + i as f64 * h).collect()
}

/// Composite trapezoidal quadrature weights.
fn trapezoidal_weights(a: f64, b: f64, n: usize) -> Vec<f64> {
    let h = (b - a) / (n as f64 - 1.0);
    let mut weights = vec![h; n];

    weights[0] = 0.5 * h;
    weights[n - 1] = 0.5 * h;

    weights
}

/// Weighted integral of one grid function.
fn weighted_integral(f: &[f64], weights: &[f64]) -> f64 {
    f.iter()
        .zip(weights.iter())
        .map(|(&fi, &wi)| wi * fi)
        .sum()
}

/// Weighted inner product approximating integral f(s)g(s) ds.
fn weighted_inner_product(f: &[f64], g: &[f64], weights: &[f64]) -> f64 {
    f.iter()
        .zip(g.iter())
        .zip(weights.iter())
        .map(|((&fi, &gi), &wi)| wi * fi * gi)
        .sum()
}

/// Build sampled continuous kernels K_i(s_j), without quadrature weights.
fn build_kernel_samples(grid: &[f64]) -> Vec<Vec<f64>> {
    let mut kernels = vec![vec![0.0; grid.len()]; NUM_DATA];

    for i in 0..NUM_DATA {
        for (j, &s) in grid.iter().enumerate() {
            kernels[i][j] = data_kernel(i, s);
        }
    }

    kernels
}

/// Build the weighted forward matrix used by least squares.
///
/// The entries are K_i(s_j) w_j so that K m approximates the data integrals.
fn build_weighted_forward_matrix(kernels: &[Vec<f64>], quad_weights: &[f64]) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; NUM_GRID]; NUM_DATA];

    for i in 0..NUM_DATA {
        for j in 0..NUM_GRID {
            matrix[i][j] = kernels[i][j] * quad_weights[j];
        }
    }

    matrix
}

/// Generate clean data d_i = integral K_i(s)m(s) ds.
fn generate_clean_data(
    kernels: &[Vec<f64>],
    model: &[f64],
    quad_weights: &[f64],
) -> Vec<f64> {
    kernels
        .iter()
        .map(|kernel| weighted_inner_product(kernel, model, quad_weights))
        .collect()
}

/// Add deterministic reproducible noise.
fn add_deterministic_noise(data: &[f64], relative_level: f64) -> Vec<f64> {
    let scale = data
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(1.0e-14);

    data.iter()
        .enumerate()
        .map(|(i, &value)| {
            let phase = 2.0 * PI * (i as f64 + 1.0) / data.len() as f64;
            let noise = relative_level
                * scale
                * ((5.0 * phase).sin() + 0.35 * (9.0 * phase).cos());

            value + noise
        })
        .collect()
}

/// Matrix-vector product y = A x.
fn mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&a, &xj)| a * xj).sum())
        .collect()
}

/// Transposed matrix-vector product y = A^T x.
fn transpose_mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut result = vec![0.0; cols];

    for i in 0..rows {
        for j in 0..cols {
            result[j] += matrix[i][j] * x[i];
        }
    }

    result
}

/// Gram matrix A^T A.
fn gram_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut gram = vec![vec![0.0; cols]; cols];

    for i in 0..cols {
        for j in 0..cols {
            let mut sum = 0.0;

            for k in 0..rows {
                sum += matrix[k][i] * matrix[k][j];
            }

            gram[i][j] = sum;
        }
    }

    gram
}

/// Construct L^T L for a first-difference regularization operator.
fn first_difference_penalty(n: usize) -> Vec<Vec<f64>> {
    let mut penalty = vec![vec![0.0; n]; n];

    for i in 0..(n - 1) {
        penalty[i][i] += 1.0;
        penalty[i][i + 1] -= 1.0;
        penalty[i + 1][i] -= 1.0;
        penalty[i + 1][i + 1] += 1.0;
    }

    penalty
}

/// Add lambda * B to A.
fn add_scaled_matrix(a: &[Vec<f64>], b: &[Vec<f64>], lambda: f64) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut result = vec![vec![0.0; m]; n];

    for i in 0..n {
        for j in 0..m {
            result[i][j] = a[i][j] + lambda * b[i][j];
        }
    }

    result
}

/// Add a small diagonal ridge for numerical stability.
fn add_diagonal_ridge(matrix: &mut [Vec<f64>], ridge: f64) {
    for i in 0..matrix.len() {
        matrix[i][i] += ridge;
    }
}

/// Dot product.
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

/// Quadratic form x^T A x.
fn quadratic_form(matrix: &[Vec<f64>], x: &[f64]) -> f64 {
    let ax = mat_vec(matrix, x);
    dot(x, &ax)
}

/// Dense linear solve by Gaussian elimination with partial pivoting.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("The system matrix must be square and compatible with b.".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            let candidate = a[i][k].abs();

            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!("Matrix is numerically singular near column {}.", k));
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

/// Solve a first-difference Tikhonov least-squares reconstruction.
fn regularized_least_squares(
    forward_matrix: &[Vec<f64>],
    data: &[f64],
    lambda: f64,
) -> Result<Vec<f64>, String> {
    let kt_k = gram_matrix(forward_matrix);
    let kt_d = transpose_mat_vec(forward_matrix, data);
    let lt_l = first_difference_penalty(NUM_GRID);

    let mut system = add_scaled_matrix(&kt_k, &lt_l, lambda);
    add_diagonal_ridge(&mut system, 1.0e-12);

    solve_linear_system(system, kt_d)
}

/// Build the Backus-Gilbert resolution matrix B for target s_bar.
fn build_bg_resolution_matrix(
    kernels: &[Vec<f64>],
    grid: &[f64],
    quad_weights: &[f64],
    target: f64,
) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; NUM_DATA]; NUM_DATA];

    for i in 0..NUM_DATA {
        for j in 0..NUM_DATA {
            let mut sum = 0.0;

            for k in 0..grid.len() {
                let distance = grid[k] - target;
                sum += quad_weights[k] * distance * distance * kernels[i][k] * kernels[j][k];
            }

            matrix[i][j] = sum;
        }
    }

    matrix
}

/// Build c_i = integral K_i(s) ds for the normalization c^T w = 1.
fn build_normalization_vector(kernels: &[Vec<f64>], quad_weights: &[f64]) -> Vec<f64> {
    kernels
        .iter()
        .map(|kernel| weighted_integral(kernel, quad_weights))
        .collect()
}

/// Build a simple diagonal data-noise covariance matrix.
fn build_covariance_matrix(noise_std: f64) -> Vec<Vec<f64>> {
    let mut covariance = vec![vec![0.0; NUM_DATA]; NUM_DATA];

    for i in 0..NUM_DATA {
        covariance[i][i] = noise_std * noise_std;
    }

    covariance
}

/// Compute Backus-Gilbert weights from B, C, and c.
fn backus_gilbert_weights(
    resolution_matrix: &[Vec<f64>],
    covariance_matrix: &[Vec<f64>],
    normalization_vector: &[f64],
    gamma: f64,
    ridge: f64,
) -> Result<Vec<f64>, String> {
    let mut tradeoff_matrix = add_scaled_matrix(resolution_matrix, covariance_matrix, gamma);
    add_diagonal_ridge(&mut tradeoff_matrix, ridge);

    let z = solve_linear_system(tradeoff_matrix, normalization_vector.to_vec())?;
    let denominator = dot(normalization_vector, &z);

    if denominator.abs() < 1.0e-14 {
        return Err("Backus-Gilbert normalization denominator is too small.".to_string());
    }

    Ok(z.iter().map(|&value| value / denominator).collect())
}

/// Construct A(s; w) = sum_i w_i K_i(s).
fn averaging_kernel(kernels: &[Vec<f64>], data_weights: &[f64]) -> Vec<f64> {
    let mut kernel = vec![0.0; NUM_GRID];

    for i in 0..NUM_DATA {
        for j in 0..NUM_GRID {
            kernel[j] += data_weights[i] * kernels[i][j];
        }
    }

    kernel
}

/// Compute the Backus-Gilbert estimate u = w^T d.
fn backus_gilbert_estimate(data_weights: &[f64], data: &[f64]) -> f64 {
    dot(data_weights, data)
}

/// Spread measure integral (s - target)^2 A(s)^2 ds.
fn averaging_kernel_spread(
    grid: &[f64],
    avg_kernel: &[f64],
    quad_weights: &[f64],
    target: f64,
) -> f64 {
    grid.iter()
        .zip(avg_kernel.iter())
        .zip(quad_weights.iter())
        .map(|((&s, &a), &w)| {
            let distance = s - target;
            w * distance * distance * a * a
        })
        .sum()
}

/// Center of mass of the averaging kernel.
fn averaging_kernel_center(grid: &[f64], avg_kernel: &[f64], quad_weights: &[f64]) -> f64 {
    let mass = weighted_integral(avg_kernel, quad_weights);

    if mass.abs() < 1.0e-14 {
        return f64::NAN;
    }

    grid.iter()
        .zip(avg_kernel.iter())
        .zip(quad_weights.iter())
        .map(|((&s, &a), &w)| w * s * a)
        .sum::<f64>()
        / mass
}

/// Exact model average induced by A(s; w).
fn exact_averaged_model_value(
    avg_kernel: &[f64],
    model: &[f64],
    quad_weights: &[f64],
) -> f64 {
    weighted_inner_product(avg_kernel, model, quad_weights)
}

/// Find nearest grid index and value.
fn nearest_grid_value(grid: &[f64], values: &[f64], target: f64) -> (usize, f64) {
    let mut best_index = 0;
    let mut best_distance = f64::INFINITY;

    for (i, &s) in grid.iter().enumerate() {
        let distance = (s - target).abs();

        if distance < best_distance {
            best_distance = distance;
            best_index = i;
        }
    }

    (best_index, values[best_index])
}

/// Euclidean norm.
fn l2_norm(x: &[f64]) -> f64 {
    x.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Infinity norm of a vector.
fn infinity_norm(x: &[f64]) -> f64 {
    x.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

/// Relative infinity-norm error.
fn relative_infinity_error(x: &[f64], y: &[f64]) -> f64 {
    let numerator = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max);

    numerator / infinity_norm(y).max(1.0e-14)
}

/// Diagnostics for one Backus-Gilbert target.
struct BgTargetResult {
    target: f64,
    nearest_index: usize,
    point_value: f64,
    least_squares_value: f64,
    bg_clean_estimate: f64,
    bg_noisy_estimate: f64,
    exact_average: f64,
    kernel_center: f64,
    kernel_width: f64,
    kernel_mass: f64,
    variance: f64,
    weight_norm: f64,
}

/// Evaluate Backus-Gilbert diagnostics at one target point.
fn evaluate_backus_gilbert_target(
    target: f64,
    kernels: &[Vec<f64>],
    grid: &[f64],
    quad_weights: &[f64],
    model: &[f64],
    clean_data: &[f64],
    noisy_data: &[f64],
    least_squares_model: &[f64],
    covariance_matrix: &[Vec<f64>],
    normalization_vector: &[f64],
    gamma: f64,
    ridge: f64,
) -> Result<BgTargetResult, String> {
    let resolution_matrix = build_bg_resolution_matrix(kernels, grid, quad_weights, target);

    let data_weights = backus_gilbert_weights(
        &resolution_matrix,
        covariance_matrix,
        normalization_vector,
        gamma,
        ridge,
    )?;

    let avg_kernel = averaging_kernel(kernels, &data_weights);

    let (_, point_value) = nearest_grid_value(grid, model, target);
    let (nearest_index, least_squares_value) = nearest_grid_value(grid, least_squares_model, target);

    let bg_clean_estimate = backus_gilbert_estimate(&data_weights, clean_data);
    let bg_noisy_estimate = backus_gilbert_estimate(&data_weights, noisy_data);
    let exact_average = exact_averaged_model_value(&avg_kernel, model, quad_weights);

    let kernel_mass = weighted_integral(&avg_kernel, quad_weights);
    let kernel_center = averaging_kernel_center(grid, &avg_kernel, quad_weights);
    let kernel_width = averaging_kernel_spread(grid, &avg_kernel, quad_weights, target).sqrt();
    let variance = quadratic_form(covariance_matrix, &data_weights);
    let weight_norm = l2_norm(&data_weights);

    Ok(BgTargetResult {
        target,
        nearest_index,
        point_value,
        least_squares_value,
        bg_clean_estimate,
        bg_noisy_estimate,
        exact_average,
        kernel_center,
        kernel_width,
        kernel_mass,
        variance,
        weight_norm,
    })
}

fn main() {
    let a = 0.0;
    let b = 1.0;

    let noise_level = 2.0e-3;
    let noise_std = 2.0e-3;

    let ls_lambda = 5.0e-4;
    let bg_gamma = 1.0e4;
    let bg_ridge = 1.0e-10;

    let targets = [0.35, 0.60, 0.80];

    let grid = uniform_grid(a, b, NUM_GRID);
    let quad_weights = trapezoidal_weights(a, b, NUM_GRID);

    let model: Vec<f64> = grid.iter().map(|&s| exact_model(s)).collect();
    let kernels = build_kernel_samples(&grid);
    let forward_matrix = build_weighted_forward_matrix(&kernels, &quad_weights);

    let clean_data = generate_clean_data(&kernels, &model, &quad_weights);
    let noisy_data = add_deterministic_noise(&clean_data, noise_level);

    let least_squares_model = match regularized_least_squares(&forward_matrix, &noisy_data, ls_lambda)
    {
        Ok(value) => value,
        Err(message) => {
            eprintln!("Least-squares reconstruction failed: {}", message);
            return;
        }
    };

    let covariance_matrix = build_covariance_matrix(noise_std);
    let normalization_vector = build_normalization_vector(&kernels, &quad_weights);

    let mut bg_results = Vec::new();

    for &target in &targets {
        let result = match evaluate_backus_gilbert_target(
            target,
            &kernels,
            &grid,
            &quad_weights,
            &model,
            &clean_data,
            &noisy_data,
            &least_squares_model,
            &covariance_matrix,
            &normalization_vector,
            bg_gamma,
            bg_ridge,
        ) {
            Ok(value) => value,
            Err(message) => {
                eprintln!(
                    "Backus-Gilbert evaluation failed for target {:.6}: {}",
                    target, message
                );
                return;
            }
        };

        bg_results.push(result);
    }

    let predicted_ls_data = mat_vec(&forward_matrix, &least_squares_model);
    let ls_residual: Vec<f64> = predicted_ls_data
        .iter()
        .zip(noisy_data.iter())
        .map(|(&p, &d)| p - d)
        .collect();

    let ls_relative_error = relative_infinity_error(&least_squares_model, &model);
    let ls_data_residual = l2_norm(&ls_residual);

    println!("Backus-Gilbert Local Averages Versus Least-Squares Reconstruction");
    println!("=================================================================");
    println!();
    println!("Problem Setup");
    println!("-------------");
    println!("Model interval              = [{:.1}, {:.1}]", a, b);
    println!("Model grid points           = {}", NUM_GRID);
    println!("Number of data kernels      = {}", NUM_DATA);
    println!("Measurement kernels         = broad Gaussian sensitivity kernels");
    println!("Noise level in data         = {:.6e}", noise_level);
    println!();
    println!("Least-Squares Reconstruction");
    println!("----------------------------");
    println!("Method                      = first-difference Tikhonov least squares");
    println!("Regularization lambda       = {:.6e}", ls_lambda);
    println!("Data residual norm          = {:.6e}", ls_data_residual);
    println!("Relative model error        = {:.6e}", ls_relative_error);
    println!();
    println!("Backus-Gilbert Setup");
    println!("--------------------");
    println!("BG trade-off gamma          = {:.6e}", bg_gamma);
    println!("BG stabilizing ridge        = {:.6e}", bg_ridge);
    println!("Noise covariance std        = {:.6e}", noise_std);
    println!();
    println!("Target-by-Target Comparison");
    println!("---------------------------");
    println!(
        "{:>8} {:>8} {:>14} {:>14} {:>14} {:>14} {:>14}",
        "target", "index", "point", "LS value", "BG noisy", "BG average", "width"
    );

    for item in &bg_results {
        println!(
            "{:>8.3} {:>8} {:>14.8} {:>14.8} {:>14.8} {:>14.8} {:>14.6e}",
            item.target,
            item.nearest_index,
            item.point_value,
            item.least_squares_value,
            item.bg_noisy_estimate,
            item.exact_average,
            item.kernel_width
        );
    }

    println!();
    println!("Backus-Gilbert Resolution Diagnostics");
    println!("-------------------------------------");
    println!(
        "{:>8} {:>14} {:>14} {:>14} {:>14} {:>14}",
        "target", "kernel_ctr", "mass", "variance", "w_norm", "clean-BG"
    );

    for item in &bg_results {
        println!(
            "{:>8.3} {:>14.8} {:>14.10} {:>14.6e} {:>14.6e} {:>14.6e}",
            item.target,
            item.kernel_center,
            item.kernel_mass,
            item.variance,
            item.weight_norm,
            (item.bg_noisy_estimate - item.bg_clean_estimate).abs()
        );
    }

    println!();
    println!("Representative Least-Squares Model Values");
    println!("------------------------------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18}",
        "j", "s_j", "m_exact", "m_LS"
    );

    for j in 0..NUM_GRID {
        if j % 10 == 0 || j == NUM_GRID - 1 {
            println!(
                "{:>8} {:>14.8} {:>18.10} {:>18.10}",
                j, grid[j], model[j], least_squares_model[j]
            );
        }
    }

    println!();
    println!("Interpretation");
    println!("--------------");
    println!("The least-squares method returns a full model vector, which may look");
    println!("like a pointwise reconstruction on the chosen grid. The Backus-Gilbert");
    println!("method instead reports local averages together with averaging-kernel");
    println!("centers, widths, masses, variances, and weight norms. These diagnostics");
    println!("show what the data can actually resolve at each target point.");
}
```

Program 19.7.3 demonstrates the different meanings of a regularized least-squares reconstruction and a Backus–Gilbert estimate. The least-squares method returns a complete model vector on the chosen grid. This output is useful, but it can invite a pointwise interpretation that may not be supported by broad and indirect data kernels.

The Backus–Gilbert method reports a different object. At each target point, it gives a local estimate together with the averaging-kernel center, width, mass, variance, and weight norm. These diagnostics show whether the estimate is sharply localized, shifted away from the nominal target, broad, or noise-sensitive.

The difference between the exact point value and the Backus–Gilbert average is intentional. A Backus–Gilbert estimate approximates the true model averaged through its resolution kernel, not necessarily the pointwise value $m(\bar{s})$. If the averaging kernel is broad or shifted, the estimate must be interpreted as a regional average rather than a sharply resolved local value.

This comparison reinforces the main message of Sections 19.7.4 and 19.7.5. Parametric least-squares inversion emphasizes fitting the data with a global model, while Backus–Gilbert inversion emphasizes what the data can resolve locally and stably. The averaging kernel prevents overinterpretation by making the spatial resolution of each estimate explicit.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/dg5QkzuNXaFRRfphEsaR.3","tags":[]}

# 19.8. Maximum Entropy Image Restoration for Positive Inverse Problems

Maximum entropy image restoration is a nonlinear inverse method designed for reconstructing positive images, spectra, or fields from incomplete and noisy measurements. It is especially useful when the unknown quantity represents an intensity, density, probability, concentration, or count, since such quantities should remain non-negative in any physically meaningful reconstruction. Unlike linear regularization methods, which typically stabilize an inverse problem by penalizing norms, roughness, or derivatives of the solution, the maximum entropy method chooses the reconstruction with the greatest entropy among all reconstructions consistent with the available information. In this sense, the method is not merely a smoothing device. It is a principle for selecting the least biased positive reconstruction compatible with the measured data.

The central idea is that missing information should not be invented by the numerical method. When the data are incomplete, noisy, or indirect, many candidate images may reproduce the observations equally well. The maximum entropy method selects, from this admissible set, the image that introduces the least additional structure beyond what the data require. This makes the method particularly attractive in astronomical imaging, medical tomography, spectral reconstruction, and other inverse problems where direct inversion may produce artifacts, oscillations, or negative intensities. The method combines a positivity-preserving representation with a constrained optimization principle, leading naturally to an exponential form for the recovered image. The following subsections develop the entropy principle, the constrained reconstruction problem, the Lagrange multiplier derivation, and the nonlinear system that must be solved to obtain the restored image.

## 19.8.1. Entropy Principle and Constrained Maximum Entropy Reconstruction

The maximum entropy method is a nonlinear inversion technique used to reconstruct images, spectra, or other non-negative quantities from incomplete, noisy, or indirect data. Its central principle is that, among all reconstructions consistent with the known measurements, one should choose the reconstruction that is maximally noncommittal about unknown information. In an image restoration problem, this means selecting the positive image that maximizes entropy while satisfying the constraints imposed by the data. The reconstruction is therefore chosen not because it has the smallest norm or the smoothest derivatives, but because it is the least biased positive image compatible with the information actually available.

Let $u_j$ denote the unknown positive image intensity at pixel or grid point $j$. The index $j$ may refer to a pixel in a two-dimensional image, a voxel in a tomographic reconstruction, a point in a spectrum, or a general grid point in a discretized field. The condition $u_j>0$ reflects the physical meaning of the unknown quantity. Image intensities, photon counts, spectral densities, and probability-like quantities cannot be negative, so a reconstruction method that naturally preserves positivity is highly desirable.

Let $m_j$ be a default model. This default model represents the image that would be selected in the absence of data. It may be uniform when there is no reason to prefer one location over another, or it may encode prior information about the expected brightness distribution. For example, if some background level or prior estimate is available, it can be represented through $m_j$. The default model therefore acts as the reference distribution relative to which entropy is measured.

The relative entropy functional is,

$$H(u)=-\sum_j u_j\ln\frac{u_j}{m_j} \tag{19.8.1}$$

This functional compares the reconstruction $u_j$ with the default model $m_j$. If $u_j$ differs strongly from $m_j$, the logarithmic ratio contributes to the entropy value. The method favors reconstructions that remain close to the default model unless the data constraints require otherwise. Thus, the default model is not ignored, but neither is it imposed absolutely. It supplies the baseline from which the data are allowed to introduce justified deviations.

The maximum entropy reconstruction is obtained by maximizing equation (19.8.1) subject to constraints imposed by the measured data. If the measured data $g_i$ are related to the unknown image by a linear forward model,

$$g_i=\sum_j K_{ij}u_j \tag{19.8.2}$$

then $K_{ij}$ describes how the intensity at pixel or grid point $j$ contributes to the measured datum $g_i$. In an imaging problem, $K$ may represent blurring, projection, incomplete Fourier sampling, or another measurement process. The data are not necessarily direct observations of the pixels. Instead, each datum may combine many image intensities through the forward operator $K$.

The constrained maximum entropy problem is:

$$\max_{u_j>0}\; H(u)\quad\text{subject to}\quad g_i=\sum_j K_{ij}u_j,\qquad\sum_j u_j=\mathrm{const} \tag{19.8.3}$$

The first constraint enforces consistency with the measured data. It requires that the reconstructed image, when passed through the forward model, reproduces the observations. The second constraint fixes the total intensity, probability mass, or total count. This is important in applications where the total amount of light, mass, probability, or signal is known independently or must remain conserved. The positivity condition $u_j>0$ is built naturally into the entropy formulation and becomes even more explicit in the exponential representation derived later.

The maximum entropy principle, as described by Hristopulos and Varouchakis (2023), states that the most suitable probability distribution, image, or field is the one that maximizes entropy subject to the information actually available. In the restoration setting, this means that the reconstruction should not introduce unwarranted structure beyond what is required by the data. It should remain as unbiased, spread out, or minimally assumptive as possible, while still reproducing the measurements. The method therefore provides a disciplined way to handle missing information: unknown features are not fabricated unless the constraints force them to appear.

This viewpoint is particularly useful when the data are incomplete. In many imaging systems, only a subset of Fourier components, projections, or blurred measurements is observed. Direct inversion may then be unstable because the missing components cannot be recovered uniquely. It may also produce artifacts, oscillations, or negative intensities. Such effects are especially problematic in imaging because negative brightness or negative density has no physical meaning in many applications. Maximum entropy restoration avoids these problems by enforcing positivity and selecting the least biased image consistent with the constraints.

The constrained form in equation (19.8.3) also clarifies the difference between maximum entropy restoration and ordinary linear inversion. A linear inversion method tries to solve for $u$ directly, often by adding a penalty term to stabilize the solution. Maximum entropy instead defines an optimization problem over positive reconstructions and selects the one with maximal entropy. The solution is therefore determined by both the data constraints and the entropy principle. This makes the method nonlinear, but it also gives it a natural way to preserve positivity and suppress unsupported structure.

## 19.8.2. Lagrange Multiplier Derivation and Exponential Solution Form

The constrained maximum entropy problem in equation (19.8.3) can be derived using Lagrange multipliers. This derivation is important because it shows why the solution has an exponential structure and why positivity is automatically preserved. Instead of treating the pixel values $u_j$ as arbitrary unknowns, the method expresses them through multipliers associated with the data and normalization constraints.

Introduce Lagrange multipliers $\lambda_i$ for the data constraints and a multiplier $\alpha$ for the normalization constraint. The Lagrangian may be written as:

\begin{equation}
\begin{aligned}
\mathcal{L}(u,\lambda,\alpha)
&= -\sum_j u_j \ln\frac{u_j}{m_j}
-\sum_i \lambda_i\left(\sum_j K_{ij}u_j - g_i\right) \\
&\quad - \alpha\left(\sum_j u_j - \mathrm{const}\right)
\end{aligned}
\tag{19.8.4}
\end{equation}

The first term is the entropy functional to be maximized. The second term enforces the data constraints $g_i=\sum_j K_{ij}u_j$. The third term enforces the normalization condition $\sum_j u_j=\mathrm{const}$. The multipliers $\lambda_i$ and $\alpha$ are not known in advance. They are determined so that the final reconstruction satisfies all constraints.

Taking the derivative of the Lagrangian with respect to $u_j$ gives:

$$\frac{\partial \mathcal L}{\partial u_j} = -\left(\ln\frac{u_j}{m_j}+1\right)-\sum_i \lambda_iK_{ij}-\alpha \tag{19.8.5}$$

This derivative combines the contribution from the entropy functional with the contributions from the measurement and normalization constraints. The logarithmic term appears because differentiating $u_j\ln(u_j/m_j)$ with respect to $u_j$ gives $\ln(u_j/m_j)+1$. The remaining terms measure how changes in $u_j$ affect the data constraints and the total-intensity constraint.

Setting this derivative equal to zero yields:

$$\ln\frac{u_j}{m_j} = -1-\alpha-\sum_i \lambda_iK_{ij} \tag{19.8.6}$$

This equation gives the logarithm of the ratio $u_j/m_j$ in terms of the Lagrange multipliers and the forward model coefficients. It shows that the departure of the reconstruction from the default model is controlled by the data constraints through the multipliers $\lambda_i$. If the data impose little correction at a given pixel, then $u_j$ remains close to the default model. If the data strongly require a correction, the exponential factor modifies the default model accordingly.

Exponentiating both sides gives the maximum entropy form,

$$u_j = m_j\exp\left(-1-\alpha-\sum_i \lambda_iK_{ij}\right) \tag{19.8.7}$$

Equation (19.8.7) is the key structural result of the maximum entropy method. It shows that the restored image is positive automatically, since it is expressed as the product of the positive default model $m_j$ and an exponential factor. The exponential factor is always positive, so $u_j>0$ whenever $m_j>0$. Thus, positivity does not need to be imposed afterward by clipping negative values or projecting onto a feasible set. It is built into the analytical form of the solution.

This expression also shows that the unknowns in the maximum entropy representation are not the pixels $u_j$ directly, but the multipliers $\lambda_i$ and $\alpha$. Once these multipliers are known, every pixel value is recovered immediately from equation (19.8.7). The multipliers must therefore be adjusted so that the data constraints and normalization condition are satisfied. In this sense, the problem is transformed from a constrained optimization problem in the image values into a nonlinear system for the multipliers.

Substituting equation (19.8.7) into the data equation (19.8.2) gives:

$$g_i = \sum_jK_{ij}m_j\exp\left(-1-\alpha-\sum_\ell \lambda_\ell K_{\ell j}\right)\tag{19.8.8}$$

This is a nonlinear system for the multipliers. The nonlinearity arises from the exponential dependence on the multipliers. Although the original forward model in equation (19.8.2) is linear in the image $u_j$, the maximum entropy parametrization makes the constraints nonlinear in $\lambda_i$ and $\alpha$. Solving this nonlinear system is the computational core of the maximum entropy method. Once the multipliers are found, the restored image is recovered from equation (19.8.7).

The normalization constraint must also be satisfied. Substituting equation (19.8.7) into $\sum_j u_j=\mathrm{const}$ gives an additional equation involving $\alpha$ and the multipliers $\lambda_i$. The role of $\alpha$ is to adjust the overall scale of the reconstruction so that the total intensity, mass, probability, or count has the required value. The data multipliers $\lambda_i$ adjust the image so that the measured observations are reproduced. Together, these multipliers enforce all constraints while preserving the entropy-maximizing structure.

This derivation also shows why maximum entropy image restoration differs from linear regularization. In a linear regularized method, the reconstructed image is usually obtained by solving a linear system or a least-squares problem with an added penalty term. In maximum entropy restoration, the final reconstruction depends nonlinearly on the data through the exponential form. This nonlinearity is essential for enforcing positivity and entropy maximization. It also allows the method to select a reconstruction that remains close to the default model except where the data require otherwise.

The same nonlinearity also makes the method computationally more demanding than linear methods such as Tikhonov regularization. Instead of solving a single linear system for (u), one must solve the nonlinear multiplier equations represented by equation (19.8.8), together with the normalization constraint. The computational effort is justified in applications where positivity, suppression of unsupported structure, and entropy-based selection are more appropriate than purely linear smoothing. Thus, maximum entropy restoration should be understood as a nonlinear constrained inverse method whose main strength is the combination of positivity, data consistency, and minimal unwarranted structure.

### Rust Implementation

Following the derivation in Section 19.8.2 on the Lagrange multiplier formulation and exponential solution form, Program 19.8.1 provides a practical implementation of maximum entropy reconstruction for a small positive inverse problem. The program uses the entropy-based representation in equation (19.8.7), where each reconstructed intensity is written as a positive default value multiplied by an exponential factor determined by the data multipliers and the normalization multiplier. It then solves the nonlinear multiplier equations associated with equation (19.8.8), together with the total-intensity constraint. This implementation highlights the key computational feature of maximum entropy restoration: positivity is not imposed by clipping or projection after the solve, but is built directly into the exponential parametrization. The example also shows that, when the data constraints are fewer than the number of unknown pixels, the method returns the least biased positive image consistent with the available constraints, rather than necessarily reproducing the original image exactly.

At the core of the implementation is the exponential maximum entropy form derived in equation (19.8.7). The constants `NUM_PIXELS` and `NUM_DATA` define the number of unknown image intensities and the number of linear data constraints. The example deliberately uses fewer data constraints than unknown pixels so that the reconstruction problem is underdetermined. This makes the role of the entropy principle visible: among many positive images satisfying the constraints, the algorithm selects the one determined by the maximum entropy multiplier form.

The function `exact_image` defines a positive synthetic image used to generate the data. This image is not directly supplied to the reconstruction algorithm. It serves only as a reference for producing the measured data and for evaluating the final reconstruction error. The function `default_model` defines the positive reference image $m_j$. In this example, the default model is uniform and has the same total intensity as the exact image. This corresponds to the situation where no spatial preference is assumed before the data constraints are imposed.

The function `forward_matrix` defines the linear measurement operator $K$. Each row represents one data constraint, and each entry $K_{ij}$ describes how pixel $j$ contributes to datum $i$. The rows are broad, so the measurements are indirect combinations of several image pixels rather than pointwise observations. This matches the structure of the linear data model in equation (19.8.2).

The functions `mat_vec`, `sum_entries`, `difference`, `l2_norm`, `infinity_norm`, `min_value`, and `max_value` provide the basic numerical operations used throughout the program. The matrix-vector product applies the forward model, the summation function checks the total-intensity constraint, and the norm functions measure data residuals, nonlinear residuals, and reconstruction error. The minimum and maximum functions are used to verify that the reconstructed image remains strictly positive.

The function `relative_entropy` evaluates the entropy functional from equation (19.8.1). It compares the reconstruction with the default model. Since the default model is used as the reference state, the entropy of the default model relative to itself is zero. A reconstruction that departs from the default model to satisfy the data has a lower relative entropy value, reflecting the information imposed by the measurements.

The `Multipliers` structure stores the unknown Lagrange multipliers. The vector `data` contains the multipliers associated with the measurement constraints, while `alpha` stores the multiplier associated with the total-intensity constraint. This mirrors the derivation in equations (19.8.4) through (19.8.7), where the image values are not solved for independently but are determined by these multipliers.

The function `image_from_multipliers` is the central function of the program. It evaluates the exponential formula in equation (19.8.7) for every pixel. Because each reconstructed value is the product of a positive default value and an exponential factor, every pixel remains positive automatically. The exponent is mildly clamped to avoid overflow during intermediate Newton iterates, but the mathematical structure remains the same.

The function `multiplier_residual` constructs the nonlinear residual for the multiplier equations. The first entries measure the mismatch between the predicted data and the prescribed data constraints. The final entry measures the mismatch in the total intensity. When this residual is zero, the reconstruction satisfies both the data equations and the normalization constraint.

The function `multiplier_jacobian` computes the Jacobian matrix of the nonlinear residual with respect to the multipliers. Since the image depends exponentially on the multipliers, the derivatives contain the current image values. This function implements the differential structure implied by equation (19.8.8), where the data constraints become nonlinear functions of the multipliers.

The function `solve_linear_system` solves the Newton correction system using Gaussian elimination with partial pivoting. This is sufficient for the small demonstration problem. The function `add_scaled_step` applies a scaled Newton update to the multiplier vector. The function `solve_multipliers` performs the damped Newton iteration. At each step, it computes the nonlinear residual, forms the Jacobian, solves the Newton system, and applies a line search. The line search reduces the step length if the full Newton step does not decrease the residual norm. This improves robustness for the nonlinear multiplier equations. The function `print_vector` is used to display the exact image, the default model, and the reconstructed maximum entropy image in a consistent indexed format. This makes it easy to see that the reconstruction is positive, close to the exact image in broad structure, and different from the uniform default model only where the constraints require it.

The `main` function assembles the complete experiment. It constructs the forward matrix, exact image, default model, total-intensity constraint, and synthetic data. It then solves for the multipliers, reconstructs the image using the exponential formula, evaluates the forward-model residual, computes entropy diagnostics, and reports the recovered multipliers. The final table verifies that the reconstructed image reproduces the measured data and satisfies the normalization constraint to roundoff accuracy.

```rust
// Program 19.8.1: Maximum Entropy Reconstruction with Exponential
// Positivity Preservation
//
// Problem statement:
// Reconstruct a positive discrete image u_j from exact linear measurements
//
//     g_i = sum_j K_ij u_j,
//
// while preserving the total intensity sum_j u_j = const.
//
// The maximum entropy derivation gives the exponential form
//
//     u_j = m_j exp(-1 - alpha - sum_i lambda_i K_ij),
//
// where m_j is a positive default model, lambda_i are data multipliers,
// and alpha enforces the normalization constraint.
//
// The program solves the nonlinear multiplier equations by Newton iteration.
// Positivity is automatic because every reconstructed value is obtained from
// a positive default model multiplied by an exponential factor.

/// Number of unknown image pixels.
const NUM_PIXELS: usize = 8;

/// Number of measured data constraints.
const NUM_DATA: usize = 3;

/// Exact positive image used to generate synthetic data.
///
/// This represents the unknown image or spectrum that the method attempts
/// to reconstruct from indirect measurements.
fn exact_image() -> Vec<f64> {
    vec![0.55, 0.85, 1.45, 2.20, 1.65, 1.10, 0.75, 0.45]
}

/// Positive default model m_j.
///
/// In the absence of data, maximum entropy would favor a reconstruction close
/// to this default distribution.
fn default_model(total_mass: f64) -> Vec<f64> {
    vec![total_mass / NUM_PIXELS as f64; NUM_PIXELS]
}

/// Forward matrix K.
///
/// Each row represents one linear measurement of the positive image. The rows
/// are deliberately broad so that the data are indirect measurements rather
/// than point samples.
fn forward_matrix() -> Vec<Vec<f64>> {
    vec![
        vec![1.00, 0.85, 0.55, 0.25, 0.10, 0.03, 0.01, 0.00],
        vec![0.05, 0.18, 0.45, 0.80, 1.00, 0.80, 0.45, 0.18],
        vec![0.00, 0.01, 0.04, 0.12, 0.28, 0.55, 0.85, 1.00],
    ]
}

/// Matrix-vector product y = A x.
fn mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&a, &xj)| a * xj).sum())
        .collect()
}

/// Sum of vector entries.
fn sum_entries(x: &[f64]) -> f64 {
    x.iter().sum()
}

/// Relative entropy H(u) = -sum_j u_j ln(u_j / m_j).
///
/// Both u_j and m_j must be positive.
fn relative_entropy(u: &[f64], m: &[f64]) -> f64 {
    u.iter()
        .zip(m.iter())
        .map(|(&uj, &mj)| -uj * (uj / mj).ln())
        .sum()
}

/// Euclidean norm.
fn l2_norm(x: &[f64]) -> f64 {
    x.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Infinity norm.
fn infinity_norm(x: &[f64]) -> f64 {
    x.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

/// Vector difference x - y.
fn difference(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y.iter()).map(|(&a, &b)| a - b).collect()
}

/// Compute the minimum entry of a vector.
fn min_value(x: &[f64]) -> f64 {
    x.iter().fold(f64::INFINITY, |a, &b| a.min(b))
}

/// Compute the maximum entry of a vector.
fn max_value(x: &[f64]) -> f64 {
    x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
}

/// Parameters of the maximum entropy exponential representation.
///
/// The vector contains lambda_1, ..., lambda_M, alpha.
#[derive(Clone)]
struct Multipliers {
    data: Vec<f64>,
    alpha: f64,
}

/// Compute u_j from the exponential maximum entropy formula.
///
/// The expression used is
///
///     u_j = m_j exp(-1 - alpha - sum_i lambda_i K_ij).
///
/// The exponent is clamped mildly to avoid overflow in pathological
/// intermediate Newton iterates.
fn image_from_multipliers(
    multipliers: &Multipliers,
    default: &[f64],
    k_matrix: &[Vec<f64>],
) -> Vec<f64> {
    let mut image = vec![0.0; NUM_PIXELS];

    for j in 0..NUM_PIXELS {
        let mut exponent = -1.0 - multipliers.alpha;

        for i in 0..NUM_DATA {
            exponent -= multipliers.data[i] * k_matrix[i][j];
        }

        let exponent = exponent.clamp(-60.0, 60.0);
        image[j] = default[j] * exponent.exp();
    }

    image
}

/// Residual vector for the nonlinear multiplier system.
///
/// The first NUM_DATA entries enforce K u = g.
/// The final entry enforces sum_j u_j = total_mass.
fn multiplier_residual(
    multipliers: &Multipliers,
    default: &[f64],
    k_matrix: &[Vec<f64>],
    data: &[f64],
    total_mass: f64,
) -> Vec<f64> {
    let image = image_from_multipliers(multipliers, default, k_matrix);
    let predicted = mat_vec(k_matrix, &image);

    let mut residual = vec![0.0; NUM_DATA + 1];

    for i in 0..NUM_DATA {
        residual[i] = predicted[i] - data[i];
    }

    residual[NUM_DATA] = sum_entries(&image) - total_mass;

    residual
}

/// Jacobian of the nonlinear multiplier residual.
///
/// Since
///
///     u_j = m_j exp(-1 - alpha - sum_i lambda_i K_ij),
///
/// we have
///
///     partial u_j / partial lambda_l = -K_lj u_j,
///     partial u_j / partial alpha    = -u_j.
///
/// These derivatives are used to form the Newton system.
fn multiplier_jacobian(
    multipliers: &Multipliers,
    default: &[f64],
    k_matrix: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let image = image_from_multipliers(multipliers, default, k_matrix);
    let size = NUM_DATA + 1;
    let mut jacobian = vec![vec![0.0; size]; size];

    // Data-constraint rows.
    for i in 0..NUM_DATA {
        for ell in 0..NUM_DATA {
            let mut sum = 0.0;

            for j in 0..NUM_PIXELS {
                sum -= k_matrix[i][j] * k_matrix[ell][j] * image[j];
            }

            jacobian[i][ell] = sum;
        }

        let mut alpha_derivative = 0.0;

        for j in 0..NUM_PIXELS {
            alpha_derivative -= k_matrix[i][j] * image[j];
        }

        jacobian[i][NUM_DATA] = alpha_derivative;
    }

    // Normalization row.
    for ell in 0..NUM_DATA {
        let mut sum = 0.0;

        for j in 0..NUM_PIXELS {
            sum -= k_matrix[ell][j] * image[j];
        }

        jacobian[NUM_DATA][ell] = sum;
    }

    jacobian[NUM_DATA][NUM_DATA] = -sum_entries(&image);

    jacobian
}

/// Solve a dense linear system using Gaussian elimination with partial pivoting.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("The system matrix must be square and compatible with b.".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            let candidate = a[i][k].abs();

            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!(
                "Newton matrix is numerically singular near column {}.",
                k
            ));
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

/// Apply a Newton update to the multiplier vector.
fn add_scaled_step(multipliers: &mut Multipliers, step: &[f64], scale: f64) {
    for i in 0..NUM_DATA {
        multipliers.data[i] += scale * step[i];
    }

    multipliers.alpha += scale * step[NUM_DATA];
}

/// Newton solver for the nonlinear multiplier equations.
///
/// The solver uses a simple damping strategy. If the full Newton step does
/// not reduce the residual norm, the step is repeatedly halved.
fn solve_multipliers(
    default: &[f64],
    k_matrix: &[Vec<f64>],
    data: &[f64],
    total_mass: f64,
    tolerance: f64,
    max_iterations: usize,
) -> Result<(Multipliers, usize, f64), String> {
    let mut multipliers = Multipliers {
        data: vec![0.0; NUM_DATA],
        alpha: -1.0,
    };

    let mut residual = multiplier_residual(&multipliers, default, k_matrix, data, total_mass);
    let mut residual_norm = l2_norm(&residual);

    for iteration in 1..=max_iterations {
        if residual_norm < tolerance {
            return Ok((multipliers, iteration - 1, residual_norm));
        }

        let jacobian = multiplier_jacobian(&multipliers, default, k_matrix);

        // Newton step solves J step = -residual.
        let rhs: Vec<f64> = residual.iter().map(|&value| -value).collect();
        let step = solve_linear_system(jacobian, rhs)?;

        let mut accepted = false;
        let mut scale = 1.0;

        for _ in 0..20 {
            let mut candidate = multipliers.clone();
            add_scaled_step(&mut candidate, &step, scale);

            let candidate_residual =
                multiplier_residual(&candidate, default, k_matrix, data, total_mass);
            let candidate_norm = l2_norm(&candidate_residual);

            if candidate_norm < residual_norm {
                multipliers = candidate;
                residual = candidate_residual;
                residual_norm = candidate_norm;
                accepted = true;
                break;
            }

            scale *= 0.5;
        }

        if !accepted {
            return Err("Newton line search failed to reduce the residual.".to_string());
        }
    }

    Ok((multipliers, max_iterations, residual_norm))
}

/// Print one vector with indices.
fn print_vector(name: &str, values: &[f64]) {
    println!("{name}");
    println!("{}", "-".repeat(name.len()));
    println!("{:>8} {:>18}", "j", "value");

    for (j, &value) in values.iter().enumerate() {
        println!("{:>8} {:>18.10}", j, value);
    }

    println!();
}

fn main() {
    let k_matrix = forward_matrix();
    let exact = exact_image();
    let total_mass = sum_entries(&exact);
    let default = default_model(total_mass);
    let data = mat_vec(&k_matrix, &exact);

    let tolerance = 1.0e-12;
    let max_iterations = 50;

    let (multipliers, iterations, final_residual) = match solve_multipliers(
        &default,
        &k_matrix,
        &data,
        total_mass,
        tolerance,
        max_iterations,
    ) {
        Ok(result) => result,
        Err(message) => {
            eprintln!("Maximum entropy multiplier solve failed: {}", message);
            return;
        }
    };

    let reconstructed = image_from_multipliers(&multipliers, &default, &k_matrix);
    let predicted = mat_vec(&k_matrix, &reconstructed);

    let data_residual = difference(&predicted, &data);
    let data_residual_norm = l2_norm(&data_residual);
    let normalization_error = (sum_entries(&reconstructed) - total_mass).abs();

    let entropy_default = relative_entropy(&default, &default);
    let entropy_reconstructed = relative_entropy(&reconstructed, &default);

    let reconstruction_error = difference(&reconstructed, &exact);
    let relative_reconstruction_error =
        infinity_norm(&reconstruction_error) / infinity_norm(&exact).max(1.0e-14);

    println!("Maximum Entropy Reconstruction with Exponential Positivity");
    println!("=========================================================");
    println!();
    println!("Problem Setup");
    println!("-------------");
    println!("Unknown pixels N            = {}", NUM_PIXELS);
    println!("Data constraints M          = {}", NUM_DATA);
    println!("Default model               = uniform positive image");
    println!("Total intensity constraint  = {:.10}", total_mass);
    println!("Newton tolerance            = {:.6e}", tolerance);
    println!("Maximum Newton iterations   = {}", max_iterations);
    println!();
    println!("Nonlinear Multiplier Solve");
    println!("--------------------------");
    println!("Iterations performed        = {}", iterations);
    println!("Final multiplier residual   = {:.6e}", final_residual);
    println!("Data residual norm          = {:.6e}", data_residual_norm);
    println!("Normalization error         = {:.6e}", normalization_error);
    println!();
    println!("Positivity and Entropy Diagnostics");
    println!("----------------------------------");
    println!("Minimum reconstructed u_j   = {:.6e}", min_value(&reconstructed));
    println!("Maximum reconstructed u_j   = {:.6e}", max_value(&reconstructed));
    println!("Entropy of default model    = {:.6e}", entropy_default);
    println!("Entropy of reconstruction   = {:.6e}", entropy_reconstructed);
    println!("Relative reconstruction err = {:.6e}", relative_reconstruction_error);
    println!();
    println!("Recovered Multipliers");
    println!("---------------------");

    for i in 0..NUM_DATA {
        println!("lambda_{:<2}                  = {:>.10}", i + 1, multipliers.data[i]);
    }

    println!("alpha                     = {:>.10}", multipliers.alpha);
    println!();

    print_vector("Exact Positive Image", &exact);
    print_vector("Default Model", &default);
    print_vector("Maximum Entropy Reconstruction", &reconstructed);

    println!("Forward-Model Consistency");
    println!("-------------------------");
    println!(
        "{:>8} {:>18} {:>18} {:>18}",
        "i", "g_i", "(Ku)_i", "residual"
    );

    for i in 0..NUM_DATA {
        println!(
            "{:>8} {:>18.10} {:>18.10} {:>18.3e}",
            i,
            data[i],
            predicted[i],
            data_residual[i]
        );
    }

    println!();
    println!("Interpretation");
    println!("--------------");
    println!("The reconstructed image is positive because every pixel is represented");
    println!("as a positive default value multiplied by an exponential factor. The");
    println!("Newton iteration solves the nonlinear multiplier equations that enforce");
    println!("the data constraints and the total-intensity constraint. Once the");
    println!("multipliers are known, the image follows directly from the exponential");
    println!("maximum entropy form.");
}
```

Program 19.8.1 demonstrates the computational meaning of the exponential maximum entropy form. The reconstructed image remains positive because positivity is built into the parametrization itself. No clipping, projection, or post-processing is required to enforce $u_j>0$.

The nonlinear multiplier solve is the central numerical step. Although the forward model is linear in the image values, substituting the exponential maximum entropy form into the constraints produces nonlinear equations for the multipliers. Once these multipliers are found, every image value follows directly from equation (19.8.7).

The output also illustrates an important interpretive point. The reconstruction satisfies the measured data and the total-intensity constraint to roundoff accuracy, but it does not necessarily reproduce the exact image pixel by pixel. This is expected because the number of constraints is smaller than the number of unknown pixels. The maximum entropy solution is therefore the least biased positive image compatible with the available information, not a guaranteed recovery of the original image.

The entropy diagnostics show how the data move the reconstruction away from the default model. The default model is the maximally uncommitted uniform image before data are imposed, while the reconstructed image contains only the structure required to satisfy the constraints. This captures the main principle of maximum entropy restoration: missing information should not be invented by the numerical method, and positive reconstructions should remain as close as possible to the default model unless the data require otherwise.

## 19.8.3. Bayesian Interpretation and Numerical Algorithms for Maximum Entropy Restoration

Maximum entropy restoration can also be interpreted from a Bayesian point of view. This interpretation is useful because it connects the entropy principle with the broader language of prior information, likelihood, noise covariance, and posterior estimation. In the constrained formulation of Section 19.8.1, entropy determines which positive reconstruction is preferred among all images consistent with the measurements. In Bayesian language, this preference can be interpreted as a prior belief that the reconstruction should remain positive and minimally structured relative to the default model $m_j$, unless the data provide evidence requiring otherwise.

The maximum entropy method is closely related to Bayesian maximum a posteriori estimation. In a Bayesian interpretation, the entropy term acts like a prior that favors positive, minimally structured reconstructions near the default model $m_j$. The data constraints or data misfit represent the likelihood. Maximizing entropy subject to data consistency is therefore analogous to maximizing a posterior distribution under an entropy-based prior. The reconstruction is not chosen solely because it fits the data, nor solely because it resembles the default model. It is chosen because it gives the best compromise between statistical agreement with the observations and entropy-based preference for a positive, noncommittal image.

In practical noisy problems, the data constraints are often not imposed exactly. Exact equality constraints such as equation (19.8.2) are appropriate when the data are assumed to be exact, but real measurements usually contain noise. Enforcing noisy data exactly can lead to overfitting, where the reconstruction reproduces measurement errors as if they were true image features. Therefore, instead of requiring exact agreement with every datum, one typically requires agreement with the data up to the expected noise level. This leads to a statistical data-consistency condition.

Such a condition may be expressed through a $\chi^2$-type constraint,

$$\chi^2(u)=\|Ku-g\|_{C^{-1}}^2 \approx \text{noise level} \tag{19.8.9}$$

where $C$ is the data covariance matrix. The notation $\|Ku-g\|_{C^{-1}}^2$ denotes a covariance-weighted squared residual. If (r=Ku-g), then this quantity is commonly interpreted as $r^\top C^{-1}r$. The covariance matrix determines how residuals are weighted. Measurements with smaller noise variance are penalized more strongly, while noisier measurements are allowed larger deviations. If the data errors are correlated, the off-diagonal entries of $C$ account for those correlations.

The reconstruction then balances entropy maximization against statistical consistency with the observed data. If $\chi^2(u)$ is too large, the reconstructed image does not explain the measurements adequately. If $\chi^2(u)$ is forced to be too small, the method may fit noise and introduce unsupported structure. The target noise level therefore determines how closely the reconstruction should match the data. This is one of the main practical differences between ideal constrained maximum entropy and maximum entropy restoration for noisy observations.

A classical implementation is the Skilling-Bryan algorithm, originally developed for maximum entropy image restoration. The method updates the Lagrange multipliers or equivalent search variables iteratively until the data constraints or the $\chi^2$ target are satisfied. The purpose of the iteration is to move toward a reconstruction that has high entropy while remaining statistically compatible with the measured data. Since the maximum entropy reconstruction depends nonlinearly on the unknown multipliers, as shown in equation (19.8.8), an iterative nonlinear solution procedure is required.

Such algorithms often use conjugate-gradient-like steps, reduced-dimensional search spaces, or Newton-type updates adapted to the entropy objective. Conjugate-gradient-like strategies are useful because large images may contain many unknown pixels, making direct manipulation of all variables expensive. Reduced-dimensional search spaces exploit the fact that the most important update directions may be represented by a small number of carefully chosen vectors. Newton-type updates use curvature information from the nonlinear objective or constraints to accelerate convergence. In each case, the aim is not merely to reduce a residual, but to increase entropy while maintaining consistency with the measured data.

The computational cost depends strongly on the structure of the forward operator $K$. For an image with $N$ pixels and a dense forward matrix, each iteration may require multiplication by $K$ and $K^\top$, giving a cost of approximately, $O(N^2)$ per iteration. Multiplication by $K$ maps the current image estimate into the data space, allowing the predicted measurements to be compared with the observed data. Multiplication by $K^\top$ maps residual or multiplier information back into the image space, allowing the algorithm to update the reconstruction. These forward and adjoint operations are therefore central to the computation.

If $K$ represents convolution, as in many blurring problems, fast Fourier transforms can reduce the cost substantially. In such cases, applying $K$ corresponds to convolution with a point-spread or blur kernel, and applying $K^\top$ corresponds to the associated adjoint convolution. Since convolution can be evaluated efficiently using FFTs, the per-iteration cost can be much lower than dense matrix multiplication. This is one reason maximum entropy restoration remains practical in image deblurring and related imaging problems.

If $K$ is sparse, sparse matrix-vector products may be used. Sparsity occurs when each measurement depends only on a limited number of pixels or when the imaging geometry has local structure. In such cases, explicitly using sparse storage avoids the cost of dense matrix operations. Large-scale maximum entropy algorithms therefore rely heavily on efficient implementations of the forward and adjoint operators. The mathematical formulation may be expressed in terms of matrices, but practical performance depends on exploiting the actual structure of the measurement process.

Maximum entropy methods often converge more slowly than purely linear methods because they solve a nonlinear constrained optimization problem. Linear regularization methods may require only a linear least-squares solve or a sequence of linear iterations. Maximum entropy restoration, by contrast, must handle the entropy functional, positivity, nonlinear multiplier equations, and possibly a $\chi^2$-based stopping condition. This additional complexity increases computational cost and may require more careful convergence monitoring.

However, maximum entropy methods can produce high-quality reconstructions, especially when positivity is essential and when the data are incomplete. Their nonlinear structure is not merely a computational burden; it is also the mechanism that preserves positivity and discourages unsupported image features. They are also less prone to certain ringing artifacts associated with naive inversion or overly aggressive linear deconvolution. This makes them particularly valuable when incomplete measurements would otherwise generate oscillations or negative intensities that have no physical meaning.

### Rust Implementation

Following the discussion in Section 19.8.3 on Bayesian interpretation, noisy data consistency, and numerical algorithms for maximum entropy restoration, Program 19.8.2 provides a practical implementation of noisy maximum entropy reconstruction using a covariance-weighted $\chi^2$ diagnostic. Unlike the exact constrained formulation of Section 19.8.2, where the data equations are imposed as equalities, this program treats the observations as noisy and therefore seeks statistical consistency rather than exact interpolation. The reconstruction is represented through normalized exponentials relative to a positive default model, so positivity and total intensity conservation are built directly into the parametrization. The objective balances the $\chi^2$-type data misfit from equation (19.8.9) against the entropy preference for a positive and minimally structured image. This illustrates the practical maximum entropy viewpoint: the reconstruction should fit the data only to the level justified by the noise while avoiding unsupported oscillations or negative intensities.

At the core of the implementation is a noisy-data maximum entropy formulation. The constants `NUM_PIXELS` and `NUM_DATA` define the number of unknown positive image values and the number of indirect measurements. The function `exact_image_value(j)` defines a positive test image consisting of a smooth background and two localized positive peaks. The function `exact_image` evaluates this model at all grid points and supplies the reference image used to generate synthetic noisy measurements.

The function `default_model(total_intensity)` constructs the positive default model $m_j$. In this program, the default is uniform and has the same total intensity as the exact image. This represents the maximum entropy baseline before the data are used. If the data do not strongly justify a departure from this baseline, the entropy term encourages the reconstruction to remain close to it.

The function `kernel_value(i, j)` defines the broad measurement kernel $K_{ij}$, and `forward_matrix` assembles the dense forward matrix. Each datum is a smoothing measurement of the image rather than a direct pixel value. This reflects the inverse-problem setting described in Section 19.8.3, where the operator $K$ maps the current image estimate into the data space so that predicted measurements can be compared with observations.

The functions `mat_vec` and `transpose_mat_vec` implement multiplication by $K$ and $K^\top$. These operations are central in maximum entropy algorithms because $K$ computes the predicted data, while $K^\top$ maps residual information back into image space. In large-scale implementations, these functions would be replaced by sparse, convolutional, or FFT-accelerated operators, but the dense form is appropriate for this compact textbook demonstration.

The helper functions `sum_entries`, `l2_norm`, `infinity_norm`, `difference`, `min_value`, and `max_value` provide diagnostic operations. They are used to compute the total intensity, residual norms, reconstruction error, and positivity bounds. These diagnostics make it possible to verify that the restored image remains positive, conserves total intensity, and fits the noisy data at an appropriate level.

The function `relative_entropy` evaluates the entropy functional relative to the default model. The entropy of the default model relative to itself is zero. When the reconstruction departs from the default model to satisfy the data, the entropy decreases. This behavior reflects the maximum entropy principle: deviations from the default model should occur only where the measurements require them.

The function `add_deterministic_noise` adds a reproducible perturbation to the clean data. This creates the noisy measurement vector used in the restoration problem. The deterministic form keeps the program output reproducible. The function `image_from_log_variables` implements the positive normalized exponential parametrization. Instead of optimizing the image values $u_j$ directly, the program optimizes log-intensity variables $z_j$. The image is then reconstructed by exponentiating these variables, multiplying by the default model, and renormalizing so that the total intensity is exactly preserved. This guarantees that every restored pixel is positive and that the mass constraint is satisfied throughout the iteration.

The function `chi_squared` computes the covariance-weighted data-consistency measure in equation (19.8.9), specialized here to independent measurements with common standard deviation. It compares the predicted data $Ku$ with the noisy observations and scales each residual by the assumed noise standard deviation. A $\chi^2$ value on the order of the number of measurements indicates that the reconstruction is statistically consistent with the noise model.

The function `objective` combines the data-consistency term and the entropy term. It minimizes a functional of the form $0.5\chi^2(u)-\beta H(u)$, where $\beta$ controls the strength of the entropy preference. A smaller entropy weight allows closer data fitting, while a larger entropy weight keeps the reconstruction closer to the default model.

The function `objective_gradient` computes the gradient of this objective with respect to the log-intensity variables. It first computes the derivative with respect to the image values using the residual term and the entropy derivative. It then applies the derivative of the normalized exponential parametrization. This step is important because the image values are coupled by the total-intensity normalization.

The function `solve_max_entropy_noisy` performs a damped gradient descent iteration in the log-intensity variables. At each iteration, it evaluates the objective gradient, attempts a descent step, and reduces the step size if the objective does not decrease. This simple line-search strategy makes the example robust enough for a textbook implementation while keeping the algorithm transparent. The function returns the restored positive image, the number of iterations used, the final gradient norm, and the final step norm.

The function `print_representative_values` prints selected entries of the exact image, default model, and restored maximum entropy image. This table shows how the restored image departs from the uniform default where the data require structure, while still remaining positive and smooth.

The `main` function assembles the full noisy restoration experiment. It constructs the exact image, total intensity, default model, forward matrix, clean data, and noisy data. It then estimates the noise standard deviation, solves the maximum entropy problem, computes the predicted data, evaluates the $\chi^2$ statistic, and reports positivity, entropy, residual, and reconstruction diagnostics. The final residual table compares each noisy datum with its predicted value from the restored image.

```rust
// Program 19.8.2: Noisy Maximum Entropy Restoration with a Chi-Squared
// Data-Consistency Diagnostic
//
// Problem statement:
// Reconstruct a positive discrete image u_j from noisy indirect measurements
//
//     g_i = sum_j K_ij u_j + eta_i.
//
// Instead of enforcing the noisy data exactly, the program monitors the
// covariance-weighted data-consistency quantity
//
//     chi^2(u) = ||K u - g||_{C^{-1}}^2.
//
// The reconstruction is represented in positive exponential form relative to
// a default model m_j. A fixed total intensity is enforced by normalization:
//
//     u_j = S m_j exp(z_j) / sum_k m_k exp(z_k).
//
// The objective minimized is
//
//     0.5 chi^2(u) - beta H(u),
//
// where H(u) is the relative entropy. The exponential representation preserves
// positivity, while the entropy term discourages unsupported structure.

use std::f64::consts::PI;

/// Number of unknown image pixels.
const NUM_PIXELS: usize = 20;

/// Number of noisy data measurements.
const NUM_DATA: usize = 12;

/// Positive exact image used to generate synthetic noisy data.
fn exact_image_value(j: usize) -> f64 {
    let x = j as f64 / (NUM_PIXELS as f64 - 1.0);

    let peak_left = 0.80 * (-((x - 0.32) * (x - 0.32)) / 0.010).exp();
    let peak_right = 0.55 * (-((x - 0.72) * (x - 0.72)) / 0.020).exp();
    let smooth_background = 0.20 + 0.15 * (PI * x).sin().powi(2);

    smooth_background + peak_left + peak_right
}

/// Build the exact image vector.
fn exact_image() -> Vec<f64> {
    (0..NUM_PIXELS).map(exact_image_value).collect()
}

/// Uniform positive default model with the prescribed total intensity.
fn default_model(total_intensity: f64) -> Vec<f64> {
    vec![total_intensity / NUM_PIXELS as f64; NUM_PIXELS]
}

/// Broad positive measurement kernel.
///
/// The measurement operator is intentionally smoothing. Each datum observes a
/// broad weighted combination of the positive image values.
fn kernel_value(i: usize, j: usize) -> f64 {
    let x = j as f64 / (NUM_PIXELS as f64 - 1.0);
    let center = 0.05 + 0.90 * i as f64 / (NUM_DATA as f64 - 1.0);
    let width = 0.13;
    let distance = x - center;

    (-distance * distance / (2.0 * width * width)).exp()
}

/// Build the dense forward matrix K.
fn forward_matrix() -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; NUM_PIXELS]; NUM_DATA];

    for i in 0..NUM_DATA {
        for j in 0..NUM_PIXELS {
            matrix[i][j] = kernel_value(i, j);
        }
    }

    matrix
}

/// Matrix-vector product y = A x.
fn mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.len()];

    for i in 0..matrix.len() {
        let mut sum = 0.0;

        for j in 0..x.len() {
            sum += matrix[i][j] * x[j];
        }

        result[i] = sum;
    }

    result
}

/// Transposed matrix-vector product y = A^T x.
fn transpose_mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut result = vec![0.0; cols];

    for i in 0..rows {
        for j in 0..cols {
            result[j] += matrix[i][j] * x[i];
        }
    }

    result
}

/// Sum of vector entries.
fn sum_entries(x: &[f64]) -> f64 {
    let mut sum = 0.0;

    for &value in x {
        sum += value;
    }

    sum
}

/// Euclidean norm.
fn l2_norm(x: &[f64]) -> f64 {
    let mut sum = 0.0;

    for &value in x {
        sum += value * value;
    }

    sum.sqrt()
}

/// Infinity norm.
fn infinity_norm(x: &[f64]) -> f64 {
    let mut norm = 0.0;

    for &value in x {
        if value.abs() > norm {
            norm = value.abs();
        }
    }

    norm
}

/// Vector difference x - y.
fn difference(x: &[f64], y: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; x.len()];

    for i in 0..x.len() {
        result[i] = x[i] - y[i];
    }

    result
}

/// Minimum vector entry.
fn min_value(x: &[f64]) -> f64 {
    let mut value = f64::INFINITY;

    for &entry in x {
        if entry < value {
            value = entry;
        }
    }

    value
}

/// Maximum vector entry.
fn max_value(x: &[f64]) -> f64 {
    let mut value = f64::NEG_INFINITY;

    for &entry in x {
        if entry > value {
            value = entry;
        }
    }

    value
}

/// Relative entropy H(u) = -sum_j u_j ln(u_j / m_j).
fn relative_entropy(u: &[f64], default: &[f64]) -> f64 {
    let mut entropy = 0.0;

    for j in 0..u.len() {
        entropy -= u[j] * (u[j] / default[j]).ln();
    }

    entropy
}

/// Add deterministic reproducible noise to the clean data.
fn add_deterministic_noise(clean_data: &[f64], relative_noise: f64) -> Vec<f64> {
    let mut scale = 0.0;

    for &value in clean_data {
        if value.abs() > scale {
            scale = value.abs();
        }
    }

    let mut noisy = vec![0.0; clean_data.len()];

    for i in 0..clean_data.len() {
        let phase = 2.0 * PI * (i as f64 + 1.0) / clean_data.len() as f64;
        let noise =
            relative_noise * scale * ((5.0 * phase).sin() + 0.35 * (9.0 * phase).cos());

        noisy[i] = clean_data[i] + noise;
    }

    noisy
}

/// Construct a positive normalized image from log variables z.
///
/// The normalization enforces sum_j u_j = total_intensity exactly.
fn image_from_log_variables(
    z: &[f64],
    default: &[f64],
    total_intensity: f64,
) -> Vec<f64> {
    let mut scaled = vec![0.0; NUM_PIXELS];

    for j in 0..NUM_PIXELS {
        let exponent = z[j].clamp(-40.0, 40.0);
        scaled[j] = default[j] * exponent.exp();
    }

    let denominator = sum_entries(&scaled).max(1.0e-300);
    let mut image = vec![0.0; NUM_PIXELS];

    for j in 0..NUM_PIXELS {
        image[j] = total_intensity * scaled[j] / denominator;
    }

    image
}

/// Compute chi^2(u) = sum_i ((K u - g)_i / sigma_i)^2.
fn chi_squared(
    k_matrix: &[Vec<f64>],
    image: &[f64],
    data: &[f64],
    noise_std: f64,
) -> f64 {
    let predicted = mat_vec(k_matrix, image);
    let mut chi2 = 0.0;

    for i in 0..NUM_DATA {
        let residual = (predicted[i] - data[i]) / noise_std;
        chi2 += residual * residual;
    }

    chi2
}

/// Objective 0.5 chi^2(u) - beta H(u).
fn objective(
    z: &[f64],
    default: &[f64],
    total_intensity: f64,
    k_matrix: &[Vec<f64>],
    data: &[f64],
    noise_std: f64,
    entropy_weight: f64,
) -> f64 {
    let image = image_from_log_variables(z, default, total_intensity);
    let chi2 = chi_squared(k_matrix, &image, data, noise_std);
    let entropy = relative_entropy(&image, default);

    0.5 * chi2 - entropy_weight * entropy
}

/// Compute the gradient of the objective with respect to log variables z.
///
/// Let F(u) = 0.5 chi^2(u) - beta H(u). First compute dF/du, then use
/// the derivative of the normalized exponential parametrization.
fn objective_gradient(
    z: &[f64],
    default: &[f64],
    total_intensity: f64,
    k_matrix: &[Vec<f64>],
    data: &[f64],
    noise_std: f64,
    entropy_weight: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let image = image_from_log_variables(z, default, total_intensity);
    let predicted = mat_vec(k_matrix, &image);
    let mut weighted_residual = vec![0.0; NUM_DATA];

    for i in 0..NUM_DATA {
        weighted_residual[i] = (predicted[i] - data[i]) / (noise_std * noise_std);
    }

    let kt_residual = transpose_mat_vec(k_matrix, &weighted_residual);

    let mut gradient_u = vec![0.0; NUM_PIXELS];

    for j in 0..NUM_PIXELS {
        gradient_u[j] =
            kt_residual[j] + entropy_weight * ((image[j] / default[j]).ln() + 1.0);
    }

    let mut weighted_average = 0.0;

    for j in 0..NUM_PIXELS {
        weighted_average += gradient_u[j] * image[j];
    }

    weighted_average /= total_intensity;

    let mut gradient_z = vec![0.0; NUM_PIXELS];

    for j in 0..NUM_PIXELS {
        gradient_z[j] = image[j] * (gradient_u[j] - weighted_average);
    }

    let gradient_norm = l2_norm(&gradient_z);

    (gradient_z, image, gradient_norm)
}

/// Perform a damped gradient descent solve in log-intensity variables.
fn solve_max_entropy_noisy(
    default: &[f64],
    total_intensity: f64,
    k_matrix: &[Vec<f64>],
    data: &[f64],
    noise_std: f64,
    entropy_weight: f64,
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, usize, f64, f64) {
    let mut z = vec![0.0; NUM_PIXELS];
    let mut current_objective = objective(
        &z,
        default,
        total_intensity,
        k_matrix,
        data,
        noise_std,
        entropy_weight,
    );

    let mut final_gradient_norm = f64::INFINITY;
    let mut final_step_norm = f64::INFINITY;

    for iteration in 1..=max_iterations {
        let (gradient, _image, gradient_norm) = objective_gradient(
            &z,
            default,
            total_intensity,
            k_matrix,
            data,
            noise_std,
            entropy_weight,
        );

        final_gradient_norm = gradient_norm;

        if gradient_norm < tolerance {
            let image = image_from_log_variables(&z, default, total_intensity);
            return (image, iteration - 1, final_gradient_norm, 0.0);
        }

        let mut step_size = 1.0;
        let mut accepted = false;
        let mut accepted_z = z.clone();
        let mut accepted_objective = current_objective;
        let mut accepted_step_norm = 0.0;

        for _ in 0..30 {
            let mut candidate_z = z.clone();

            for j in 0..NUM_PIXELS {
                candidate_z[j] -= step_size * gradient[j];
            }

            let candidate_objective = objective(
                &candidate_z,
                default,
                total_intensity,
                k_matrix,
                data,
                noise_std,
                entropy_weight,
            );

            if candidate_objective < current_objective {
                let mut step_norm_squared = 0.0;

                for j in 0..NUM_PIXELS {
                    let step = candidate_z[j] - z[j];
                    step_norm_squared += step * step;
                }

                accepted_z = candidate_z;
                accepted_objective = candidate_objective;
                accepted_step_norm = step_norm_squared.sqrt();
                accepted = true;
                break;
            }

            step_size *= 0.5;
        }

        if !accepted {
            let image = image_from_log_variables(&z, default, total_intensity);
            return (image, iteration - 1, final_gradient_norm, final_step_norm);
        }

        z = accepted_z;
        current_objective = accepted_objective;
        final_step_norm = accepted_step_norm;

        if final_step_norm < tolerance {
            let image = image_from_log_variables(&z, default, total_intensity);
            return (image, iteration, final_gradient_norm, final_step_norm);
        }
    }

    let image = image_from_log_variables(&z, default, total_intensity);
    (
        image,
        max_iterations,
        final_gradient_norm,
        final_step_norm,
    )
}

/// Print selected image values.
fn print_representative_values(exact: &[f64], default: &[f64], restored: &[f64]) {
    println!("Representative Image Values");
    println!("---------------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "j", "x_j", "u_exact", "m_default", "u_MEM"
    );

    for j in 0..NUM_PIXELS {
        if j % 2 == 0 || j == NUM_PIXELS - 1 {
            let x = j as f64 / (NUM_PIXELS as f64 - 1.0);

            println!(
                "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.10}",
                j, x, exact[j], default[j], restored[j]
            );
        }
    }

    println!();
}

fn main() {
    let exact = exact_image();
    let total_intensity = sum_entries(&exact);
    let default = default_model(total_intensity);

    let k_matrix = forward_matrix();
    let clean_data = mat_vec(&k_matrix, &exact);

    let relative_noise = 1.5e-2;
    let noisy_data = add_deterministic_noise(&clean_data, relative_noise);

    let data_scale = infinity_norm(&clean_data);
    let noise_std = relative_noise * data_scale;

    let entropy_weight = 1.0e-1;
    let tolerance = 1.0e-9;
    let max_iterations = 5000;

    let restored = solve_max_entropy_noisy(
        &default,
        total_intensity,
        &k_matrix,
        &noisy_data,
        noise_std,
        entropy_weight,
        tolerance,
        max_iterations,
    );

    let restored_image = restored.0;
    let iterations = restored.1;
    let final_gradient_norm = restored.2;
    let final_step_norm = restored.3;

    let predicted = mat_vec(&k_matrix, &restored_image);
    let residual = difference(&predicted, &noisy_data);
    let data_error_against_clean = difference(&noisy_data, &clean_data);

    let chi2_restored = chi_squared(&k_matrix, &restored_image, &noisy_data, noise_std);
    let chi2_exact_noisy = chi_squared(&k_matrix, &exact, &noisy_data, noise_std);

    let entropy_default = relative_entropy(&default, &default);
    let entropy_restored = relative_entropy(&restored_image, &default);

    let reconstruction_error = difference(&restored_image, &exact);
    let relative_reconstruction_error =
        infinity_norm(&reconstruction_error) / infinity_norm(&exact).max(1.0e-14);

    let data_residual_norm = l2_norm(&residual);
    let noise_norm = l2_norm(&data_error_against_clean);

    println!("Noisy Maximum Entropy Restoration with Chi-Squared Consistency");
    println!("=============================================================");
    println!();
    println!("Problem Setup");
    println!("-------------");
    println!("Unknown pixels N            = {}", NUM_PIXELS);
    println!("Data measurements M         = {}", NUM_DATA);
    println!("Forward model               = broad smoothing measurements");
    println!("Default model               = uniform positive image");
    println!("Total intensity constraint  = {:.10}", total_intensity);
    println!("Relative noise level        = {:.6e}", relative_noise);
    println!("Assumed noise std           = {:.6e}", noise_std);
    println!("Entropy weight beta         = {:.6e}", entropy_weight);
    println!();
    println!("Iteration Diagnostics");
    println!("---------------------");
    println!("Iterations performed        = {}", iterations);
    println!("Final gradient norm         = {:.6e}", final_gradient_norm);
    println!("Final step norm             = {:.6e}", final_step_norm);
    println!();
    println!("Data-Consistency Diagnostics");
    println!("----------------------------");
    println!("Noise norm ||eta||_2        = {:.6e}", noise_norm);
    println!("Residual norm ||Ku-g||_2    = {:.6e}", data_residual_norm);
    println!("chi^2 restored              = {:.6e}", chi2_restored);
    println!("chi^2 exact image vs noisy  = {:.6e}", chi2_exact_noisy);
    println!("Nominal chi^2 scale         = {}", NUM_DATA);
    println!();
    println!("Positivity and Entropy Diagnostics");
    println!("----------------------------------");
    println!("Minimum restored u_j        = {:.6e}", min_value(&restored_image));
    println!("Maximum restored u_j        = {:.6e}", max_value(&restored_image));
    println!("Total restored intensity    = {:.10}", sum_entries(&restored_image));
    println!("Entropy of default model    = {:.6e}", entropy_default);
    println!("Entropy of restored image   = {:.6e}", entropy_restored);
    println!("Relative reconstruction err = {:.6e}", relative_reconstruction_error);
    println!();

    print_representative_values(&exact, &default, &restored_image);

    println!("Forward-Model Residuals");
    println!("-----------------------");
    println!(
        "{:>8} {:>18} {:>18} {:>18}",
        "i", "g_noisy", "(Ku)_i", "residual"
    );

    for i in 0..NUM_DATA {
        println!(
            "{:>8} {:>18.10} {:>18.10} {:>18.3e}",
            i, noisy_data[i], predicted[i], residual[i]
        );
    }

    println!();
    println!("Interpretation");
    println!("--------------");
    println!("The reconstruction remains positive because the unknown image is");
    println!("represented by normalized exponentials relative to a positive default");
    println!("model. The chi-squared value measures statistical agreement with the");
    println!("noisy data. The entropy term prevents the solver from forcing an exact");
    println!("fit to noise and discourages unsupported image structure.");
}
```

Program 19.8.2 demonstrates how maximum entropy restoration is adapted to noisy inverse problems. Instead of enforcing the measured data exactly, the program monitors a $\chi^2$-type consistency measure. This is important because exact agreement with noisy observations would encourage overfitting and could introduce unsupported image structure.

The normalized exponential representation keeps the reconstruction positive for every iteration and preserves the total intensity exactly. Thus, positivity and normalization are not imposed by post-processing. They are built into the variables used by the algorithm.

The entropy term acts as a prior that favors reconstructions close to the default model unless the data justify a departure. In the output, the restored image departs from the uniform default in regions supported by the measurements, but it avoids forcing an exact fit to the noisy data. The residual norm remains close to the noise norm, and the $\chi^2$ value indicates statistical consistency with the assumed noise level.

The iteration reaches the maximum iteration count in the demonstrated output, so the result should be interpreted as a practical damped-gradient restoration rather than a fully optimized production solve. Nevertheless, the key numerical properties are achieved: the image remains positive, the total intensity is conserved, and the data are matched to the noise scale. This makes the program suitable as a pedagogical example of Bayesian maximum entropy restoration with $\chi^2$-based data consistency.

## 19.8.4. Modern Developments and Comparison with Linear Regularization Methods

Modern developments in maximum entropy methods extend the classical entropy principle by combining it with Bayesian modeling, learned priors, and contemporary optimization strategies. The original maximum entropy idea remains the same: the reconstruction should be positive, data-consistent, and minimally assumptive beyond the information supplied by the measurements. What has changed is the range of ways in which the default model, prior information, and numerical optimization can be incorporated.

One important direction is the use of data-driven default models $m_j$. In the classical setting, the default model may be uniform, representing the absence of prior preference among pixels. In more informed settings, however, one may construct $m_j$ from previous observations, physical simulations, or neural-network predictions. This gives the entropy functional a more application-specific reference state. The reconstruction is then encouraged to remain close to a learned or physically informed baseline unless the data require otherwise.

This use of a data-driven default model must be interpreted carefully. The default model should not replace the measured data, nor should it impose unsupported features. Its role is to provide a baseline image relative to which deviations are penalized through the entropy functional. If the observations strongly support a departure from the default model, the reconstruction can move away from it. If the observations are weak or incomplete, the entropy principle prevents the method from inventing unnecessary structure and keeps the reconstruction close to the baseline.

Another direction is the use of hybrid entropy-prior methods, sometimes described as maximum entropy on the mean or entropy-regularized Bayesian reconstruction. These approaches combine the classical entropy functional with additional prior information, such as smoothness, sparsity, or learned structure. The motivation is that entropy alone may not capture all useful prior knowledge in complex inverse problems. For example, an image may be expected to be positive and also spatially coherent, or a spectrum may be expected to contain only a limited number of significant features. Hybrid methods attempt to retain the positivity and noncommittal character of maximum entropy while improving performance in highly underdetermined or structurally rich problems.

Compared with least-squares methods, maximum entropy has several distinctive features. First, it naturally enforces positivity. This is essential for images, spectra, probability densities, tracer concentrations, and physical intensities. A least-squares reconstruction may produce negative values unless positivity is imposed separately. Maximum entropy avoids this difficulty because the exponential solution structure in equation (19.8.7) produces positive values automatically when the default model is positive.

Second, maximum entropy tends to produce smooth reconstructions that avoid unnecessary oscillations. The entropy functional discourages sharp or highly structured deviations from the default model unless the data require them. This behavior is useful when the measurements are incomplete, because missing information can otherwise appear as artificial high-frequency structure in the reconstruction. The method does not simply smooth the image in the same way as a derivative penalty; rather, it suppresses unsupported structure through the entropy principle.

Third, maximum entropy can preserve important features more effectively than naive inversions because it avoids fitting unconstrained high-frequency noise. Direct inversion of incomplete or noisy data may introduce ringing, sidelobes, or oscillatory artifacts. Maximum entropy restoration reduces these effects by requiring both positivity and entropy-based restraint. In astronomical imaging, this can reduce sidelobe artifacts and improve the fidelity of reconstructed sources. The method is therefore useful when the measured data provide incomplete coverage of the image information.

However, the method also has limitations. It can be sensitive to the choice of default model $m_j$, since the entropy is defined relative to this model. If the default model is poorly chosen, the reconstruction may be biased toward an inappropriate baseline, especially in regions where the data are weak. A uniform default model may be suitable when little prior information is available, but it may be too simplistic when the object has known large-scale structure. Conversely, an overly detailed default model may introduce prior assumptions that are not supported by the data.

The method also depends on accurate noise estimates or appropriate data constraints. If the noise level is underestimated, the method may attempt to fit the data too closely. This can lead to overfitting, where the reconstruction begins to reproduce noise as if it were true signal. If the noise level is overestimated, the method may allow too much discrepancy between the predicted and observed data, causing the reconstruction to remain too close to the default model. Thus, the $\chi^2$-type constraint in equation (19.8.9) is only as reliable as the noise model used to define it.

In addition, some next-generation maximum entropy formulations involve non-convex entropy objectives, which require more advanced optimization and careful initialization. Non-convexity means that the optimization landscape may contain multiple local extrema, so the final reconstruction may depend on the starting point or on details of the numerical algorithm. In such cases, robust optimization strategies and diagnostic checks become especially important. The classical maximum entropy principle remains conceptually simple, but modern extensions can be computationally more delicate.

Maximum entropy image restoration should therefore be viewed as part of the broader family of regularized inverse methods. Like Tikhonov regularization, it stabilizes an ill-posed problem by adding prior structure. Both methods use additional information to choose one reconstruction from many possibilities consistent with incomplete or noisy data. However, the type of prior structure is different. Tikhonov regularization usually penalizes solution size, roughness, or derivative magnitude through a quadratic term. Maximum entropy uses a nonlinear entropy functional relative to a positive default model.

Unlike Tikhonov regularization, the maximum entropy prior is nonlinear, positivity preserving, and entropy based. This makes the method especially attractive in imaging problems where intensities must remain positive and where incomplete data would otherwise produce artifacts. Its strength lies not in being universally superior to linear methods, but in being well matched to inverse problems involving positive quantities, incomplete measurements, and the need to avoid unsupported structure. In such settings, maximum entropy provides a principled nonlinear alternative to purely quadratic regularization methods (Hristopulos and Varouchakis, 2023; Nityananda, 2024).

## 19.8.5. Applications of Maximum Entropy Restoration in Astronomy and Medical Tomography

Maximum entropy image restoration has been widely used in applications where the unknown image is positive and the available measurements are incomplete, indirect, or noisy. Two important examples are radio astronomy imaging and medical tomography. In both cases, the measured data are not direct pixel values. Instead, they are transformed, projected, blurred, or otherwise indirect measurements of an underlying positive image. The maximum entropy method is well suited to these applications because it enforces positivity, incorporates data consistency, and discourages artificial structure not justified by the observations.

A major application of maximum entropy image restoration is radio astronomy imaging. Interferometric radio telescopes do not measure the sky brightness image directly. Instead, they measure selected Fourier components of the sky brightness at discrete spatial frequencies. These measurements are often called visibilities. Since only incomplete Fourier coverage is available, the measured data do not contain all Fourier components needed to reconstruct the sky image uniquely. A direct inverse Fourier transform therefore produces a dirty image with sidelobes and artifacts.

Maximum entropy restoration addresses this incompleteness by reconstructing a smooth, positive sky image that maximizes entropy while reproducing the measured visibilities. In this setting, the unknown $u_j$ represents the sky brightness at pixel $j$, and the operator $K$ maps the image to the measured Fourier visibility data. The data constraints ensure that the reconstructed sky image agrees with the observed visibilities within the required tolerance, while the entropy term discourages artificial structure not supported by the measurements.

This is particularly important in radio astronomy because incomplete spatial-frequency sampling can generate strong artifacts. Sidelobes in the dirty image may look like real sources or distort the shape of true sources. Maximum entropy restoration reduces these effects by selecting a positive image that remains as noncommittal as possible beyond the measured information. The reconstruction is therefore guided by the data, but it does not introduce unnecessary oscillatory structure simply to fill missing Fourier information.

The resulting image typically has fewer artifacts than a direct inverse transform and can provide improved fidelity in regions where data coverage is incomplete. This does not mean that maximum entropy can recover information that was never measured. Rather, it provides a principled way to choose among the many positive images compatible with the observed visibilities. The entropy criterion favors the image with the least unwarranted structure, while the data constraints preserve the information actually present in the measurements.

Classical maximum entropy algorithms, including the Skilling-Bryan method, became influential in astronomical imaging because of these advantages. Their ability to combine positivity, entropy maximization, and visibility-data consistency made them well suited to interferometric reconstruction. Maximum entropy ideas continue to be used in selected imaging tasks for modern radio arrays such as ALMA and the VLA, especially in situations where positivity and incomplete Fourier coverage remain central issues (Nityananda, 2024).

A second application is medical tomography, including positron emission tomography and single-photon emission computed tomography. In these modalities, the goal is to recover a non-negative tracer concentration from projection data. The unknown image represents a physical quantity, such as tracer concentration or activity distribution, and therefore cannot be negative. The measurements are indirect and noisy, since detectors record projection counts influenced by imaging geometry, attenuation, detector response, and statistical fluctuations.

Maximum entropy restoration is well matched to this problem because it enforces positivity while stabilizing the reconstruction. The projection model may be written in the form:

$$g_i=\sum_j K_{ij}u_j+\eta_i \tag{19.8.10}$$

where $g_i$ are measured projection counts, $K_{ij}$ describes the imaging geometry and detector response, $u_j$ is the tracer concentration in voxel or pixel $j$, and $\eta_i$ represents noise. This model is the tomographic counterpart of the general linear measurement equation. Each measured count is influenced by contributions from multiple voxels or pixels, weighted by the system response.

Maximum entropy reconstruction seeks a positive image that fits the projection data and satisfies a total-count constraint. The total-count constraint is physically meaningful because the total amount of detected activity or intensity may be known or statistically constrained. By maximizing entropy, the method suppresses noise-driven oscillations and produces stable images. It avoids negative tracer concentrations and reduces reconstruction artifacts that may arise from noisy or incomplete projection data.

In high-noise situations, maximum entropy reconstructions can provide clearer organ delineation than direct or filtered backprojection methods, because the entropy prior reduces artifacts while preserving positivity. Direct or filtered backprojection methods may amplify noise or create streak-like artifacts when the data are limited or noisy. Maximum entropy restoration instead selects a positive reconstruction that remains minimally structured beyond what the projections require. This can make anatomical or functional regions easier to interpret when the data quality is limited.

This makes the method useful in settings where the data are limited, the noise level is significant, or the physical interpretation requires non-negative intensities. In medical tomography, these conditions are common because reducing dose, acquisition time, or number of projections can lead to incomplete or noisy data. A positivity-preserving entropy method provides a way to stabilize the reconstruction without allowing physically impossible negative concentrations.

Across both astronomy and medical tomography, the central advantage of maximum entropy restoration is the same: it converts an underdetermined or ill-posed image reconstruction problem into a constrained nonlinear optimization problem whose solution is positive, data-consistent, and minimally biased beyond the information supplied by the measurements. In astronomy, this helps reconstruct sky brightness from incomplete visibility data. In medical tomography, it helps reconstruct tracer concentration from noisy projection counts. In both cases, maximum entropy provides a disciplined reconstruction principle for positive images when direct inversion is unstable or physically unreliable (Hristopulos and Varouchakis, 2023; Nityananda, 2024).

### Rust Implementation

Following the discussion in Sections 19.8.4 and 19.8.5 on modern maximum entropy restoration and its comparison with linear regularization methods, Program 19.8.3 provides a practical deblurring example for a positive one-dimensional signal. The program treats the signal as a compact analogue of an image, spectrum, sky-brightness profile, or tomographic tracer concentration. It first reconstructs the blurred noisy data using a linear Tikhonov method with a first-difference penalty, and then compares this result with a nonlinear maximum entropy reconstruction based on a positive normalized exponential parametrization. This comparison reflects the main distinction developed in the section: Tikhonov regularization stabilizes the inverse problem through a quadratic smoothness penalty, while maximum entropy restoration preserves positivity, enforces a total-intensity constraint, and discourages unsupported structure through an entropy functional relative to a positive default model.

At the core of the implementation is a comparison between a linear quadratic regularization method and a nonlinear entropy-based positive reconstruction method. The constant `N` defines the number of pixels in the one-dimensional signal. The function `exact_signal_value(j)` defines a positive reference signal consisting of a low positive background and two localized sources. This provides a simple model of a positive image, spectrum, radio-astronomy brightness profile, or tracer concentration.

The function `exact_signal` evaluates the exact signal at all grid points. This exact signal is used to generate synthetic blurred data and to measure reconstruction error. In an actual inverse problem, this exact signal would not be known, but in a textbook example it is useful for verifying how the two methods behave.

The function `default_model(total_mass)` constructs a smooth positive default model. Unlike a uniform default, this model contains a broad smooth baseline and is then rescaled to have the same total mass as the exact signal. This reflects the discussion in Section 19.8.4 that modern maximum entropy methods may use a physically informed or data-driven default model $m_j$, while still allowing the reconstruction to depart from it when the measured data require such a departure.

The functions `blur_kernel_value` and `blur_matrix` define the forward operator. The blur matrix is row-normalized so that each observed datum is a local weighted average of the unknown signal. This creates a smoothing inverse problem: sharp features in the exact signal are attenuated in the measured data, making direct inversion unstable.

The functions `mat_vec`, `transpose_mat_vec`, and `gram_matrix` provide the basic linear algebra operations needed for the Tikhonov reconstruction. The matrix-vector product applies the blur operator $K$, the transposed product applies $K^\top$, and the Gram matrix constructs $K^\top K$. These operations are also used in the maximum entropy gradient, where residual information is mapped back from data space into image space.

The function `first_difference_penalty` constructs the matrix $L^\top L$ for a first-difference smoothness penalty. This is used in the Tikhonov method to suppress rapid oscillations in the reconstructed signal. The function `add_scaled_matrix` forms the regularized system matrix by adding the scaled penalty matrix to the normal-equation matrix.

The function `solve_linear_system` solves dense linear systems using Gaussian elimination with partial pivoting. This solver is used by the Tikhonov reconstruction. For the moderate problem size in the program, a dense direct solver is sufficient and keeps the implementation self-contained.

The function `tikhonov_reconstruction` computes the linear regularized least-squares solution. It forms $K^\top K$, $K^\top g$, and $L^\top L$, then solves the regularized normal equation. This produces a full reconstructed signal vector. Unlike the maximum entropy method, this reconstruction is not automatically constrained to preserve positivity or total intensity.

The function `add_deterministic_noise` adds reproducible noise to the blurred data. Using deterministic noise ensures that the program gives the same output every time it is run, which is useful for textbook verification and comparison.

The functions `sum_entries`, `l2_norm`, `infinity_norm`, `difference`, `min_value`, `max_value`, `relative_infinity_error`, and `count_negative_entries` compute the diagnostics used to assess the reconstructions. These include residual norms, relative reconstruction error, minimum and maximum values, total intensity, and the number of negative entries. The negative-entry count is especially important because positivity is one of the central motivations for maximum entropy restoration.

The function `relative_entropy` evaluates the entropy functional relative to the default model. A reconstruction that remains close to the default model has a less negative entropy value, while a reconstruction with more structure departs further from the default. In this program, the entropy term discourages unnecessary structure while still allowing the data to shape the reconstruction.

The function `image_from_log_variables` implements the positive normalized exponential parametrization used by the maximum entropy method. Instead of optimizing the signal values $u_j$ directly, the program optimizes log-intensity variables $z_j$. These are exponentiated, multiplied by the positive default model, and normalized so that the total mass is exactly preserved. This guarantees positive reconstructed values and enforces the total-intensity constraint throughout the iteration.

The function `chi_squared` computes the $\chi^2$-type data-consistency diagnostic. It compares the predicted blurred data $Ku$ with the noisy observations and scales the residuals by the assumed noise standard deviation. This diagnostic reflects the statistical consistency idea discussed in equation (19.8.9).

The function `mem_objective` defines the maximum entropy objective used in the program. It combines a data-fitting term, expressed through the $\chi^2$ diagnostic, with an entropy term weighted by `entropy_weight`. The parameter controls the balance between fitting the noisy blurred data and remaining close to the positive default model.

The function `mem_gradient` computes the gradient of the maximum entropy objective with respect to the log-intensity variables. It first differentiates the objective with respect to the image values and then applies the derivative of the normalized exponential parametrization. This step is necessary because the normalization couples all image values through the total-mass constraint.

The helper function `max_abs_value` is used to normalize the descent direction in the maximum entropy solver. This prevents the raw $\chi^2$-gradient, which may be large because of the $1/\sigma^2$ scaling, from producing an unstable first step.

The function `maximum_entropy_reconstruction` performs the maximum entropy optimization. It uses a damped, normalized gradient descent method with bounded log-intensity updates. The solver also monitors the $\chi^2$ diagnostic so that the reconstruction remains statistically meaningful rather than simply forcing a formal decrease in the objective. This structure avoids the numerical collapse of pixels toward zero and keeps the maximum entropy reconstruction positive and stable.

The function `print_representative_values` prints selected entries of the exact signal, blurred data, Tikhonov reconstruction, and maximum entropy reconstruction. This makes the comparison concrete by showing how each method behaves near the background and localized source regions.

The `main` function assembles the full experiment. It builds the exact positive signal, smooth default model, blur matrix, clean blurred data, and noisy blurred data. It then computes both the Tikhonov and maximum entropy reconstructions, evaluates their data-fit diagnostics, reconstruction errors, positivity properties, total intensities, and entropy. The printed output makes clear that the two methods solve related inverse problems but enforce different types of prior structure.

```rust
// Program 19.8.3: Maximum Entropy Deblurring for a Positive Signal
// Compared with Tikhonov Regularization
//
// Problem statement:
// Restore a positive one-dimensional image or spectrum from noisy blurred data.
//
// The forward model is
//
//     g_i = sum_j K_ij u_j + eta_i,
//
// where K is a smoothing blur operator, u_j is the unknown positive signal,
// and eta_i is measurement noise.
//
// Two reconstructions are compared:
//
// 1. Linear Tikhonov regularization:
//        min_u ||K u - g||^2 + lambda ||L u||^2.
//
// 2. Maximum entropy restoration:
//        min_z 0.5 chi^2(u(z)) - beta H(u(z)),
//
//    where
//
//        u_j(z) = S m_j exp(z_j) / sum_k m_k exp(z_k).
//
// The exponential representation preserves positivity automatically and
// enforces the total-intensity constraint exactly.

use std::f64::consts::PI;

/// Number of pixels in the one-dimensional positive image.
const N: usize = 40;

/// Positive exact signal.
///
/// This represents a compact one-dimensional analogue of a positive image,
/// spectrum, sky-brightness profile, or tracer concentration.
fn exact_signal_value(j: usize) -> f64 {
    let x = j as f64 / (N as f64 - 1.0);

    let background = 0.12 + 0.05 * (PI * x).sin().powi(2);
    let source_1 = 0.95 * (-((x - 0.30) * (x - 0.30)) / 0.004).exp();
    let source_2 = 0.65 * (-((x - 0.70) * (x - 0.70)) / 0.010).exp();

    background + source_1 + source_2
}

/// Build the exact positive signal.
fn exact_signal() -> Vec<f64> {
    (0..N).map(exact_signal_value).collect()
}

/// A smooth positive default model.
///
/// It is deliberately smoother than the exact signal. This represents an
/// informed baseline that guides the maximum entropy reconstruction without
/// replacing the measured data.
fn default_model(total_mass: f64) -> Vec<f64> {
    let mut model = vec![0.0; N];

    for j in 0..N {
        let x = j as f64 / (N as f64 - 1.0);
        model[j] = 0.70 + 0.25 * (PI * x).sin();
    }

    let scale = total_mass / sum_entries(&model);

    for value in &mut model {
        *value *= scale;
    }

    model
}

/// Gaussian blur kernel value centered at observation i and source pixel j.
fn blur_kernel_value(i: usize, j: usize, width: f64) -> f64 {
    let x_i = i as f64 / (N as f64 - 1.0);
    let x_j = j as f64 / (N as f64 - 1.0);
    let distance = x_i - x_j;

    (-distance * distance / (2.0 * width * width)).exp()
}

/// Build a row-normalized blur matrix K.
///
/// Row normalization makes each blurred datum a local weighted average.
fn blur_matrix(width: f64) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; N]; N];

    for i in 0..N {
        let mut row_sum = 0.0;

        for j in 0..N {
            matrix[i][j] = blur_kernel_value(i, j, width);
            row_sum += matrix[i][j];
        }

        for j in 0..N {
            matrix[i][j] /= row_sum;
        }
    }

    matrix
}

/// Matrix-vector product y = A x.
fn mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.len()];

    for i in 0..matrix.len() {
        let mut sum = 0.0;

        for j in 0..x.len() {
            sum += matrix[i][j] * x[j];
        }

        result[i] = sum;
    }

    result
}

/// Transposed matrix-vector product y = A^T x.
fn transpose_mat_vec(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut result = vec![0.0; cols];

    for i in 0..rows {
        for j in 0..cols {
            result[j] += matrix[i][j] * x[i];
        }
    }

    result
}

/// Gram matrix A^T A.
fn gram_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut gram = vec![vec![0.0; cols]; cols];

    for i in 0..cols {
        for j in 0..cols {
            let mut sum = 0.0;

            for k in 0..rows {
                sum += matrix[k][i] * matrix[k][j];
            }

            gram[i][j] = sum;
        }
    }

    gram
}

/// First-difference penalty matrix L^T L.
fn first_difference_penalty(n: usize) -> Vec<Vec<f64>> {
    let mut penalty = vec![vec![0.0; n]; n];

    for i in 0..(n - 1) {
        penalty[i][i] += 1.0;
        penalty[i][i + 1] -= 1.0;
        penalty[i + 1][i] -= 1.0;
        penalty[i + 1][i + 1] += 1.0;
    }

    penalty
}

/// Add lambda * B to A.
fn add_scaled_matrix(a: &[Vec<f64>], b: &[Vec<f64>], lambda: f64) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut result = vec![vec![0.0; m]; n];

    for i in 0..n {
        for j in 0..m {
            result[i][j] = a[i][j] + lambda * b[i][j];
        }
    }

    result
}

/// Dense linear solve using Gaussian elimination with partial pivoting.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("The system matrix must be square and compatible with b.".to_string());
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            let candidate = a[i][k].abs();

            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return Err(format!("Matrix is numerically singular near column {}.", k));
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

/// Linear Tikhonov reconstruction.
fn tikhonov_reconstruction(
    k_matrix: &[Vec<f64>],
    data: &[f64],
    lambda: f64,
) -> Result<Vec<f64>, String> {
    let kt_k = gram_matrix(k_matrix);
    let kt_g = transpose_mat_vec(k_matrix, data);
    let lt_l = first_difference_penalty(N);

    let system = add_scaled_matrix(&kt_k, &lt_l, lambda);
    solve_linear_system(system, kt_g)
}

/// Add deterministic reproducible noise to data.
fn add_deterministic_noise(clean_data: &[f64], relative_noise: f64) -> Vec<f64> {
    let scale = infinity_norm(clean_data);
    let mut noisy = vec![0.0; clean_data.len()];

    for i in 0..clean_data.len() {
        let phase = 2.0 * PI * (i as f64 + 1.0) / clean_data.len() as f64;
        let noise =
            relative_noise * scale * ((7.0 * phase).sin() + 0.30 * (13.0 * phase).cos());

        noisy[i] = clean_data[i] + noise;
    }

    noisy
}

/// Sum of vector entries.
fn sum_entries(x: &[f64]) -> f64 {
    let mut sum = 0.0;

    for &value in x {
        sum += value;
    }

    sum
}

/// Euclidean norm.
fn l2_norm(x: &[f64]) -> f64 {
    let mut sum = 0.0;

    for &value in x {
        sum += value * value;
    }

    sum.sqrt()
}

/// Infinity norm.
fn infinity_norm(x: &[f64]) -> f64 {
    let mut norm = 0.0;

    for &value in x {
        let abs_value = value.abs();

        if abs_value > norm {
            norm = abs_value;
        }
    }

    norm
}

/// Vector difference x - y.
fn difference(x: &[f64], y: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; x.len()];

    for i in 0..x.len() {
        result[i] = x[i] - y[i];
    }

    result
}

/// Minimum vector entry.
fn min_value(x: &[f64]) -> f64 {
    let mut value = f64::INFINITY;

    for &entry in x {
        if entry < value {
            value = entry;
        }
    }

    value
}

/// Maximum vector entry.
fn max_value(x: &[f64]) -> f64 {
    let mut value = f64::NEG_INFINITY;

    for &entry in x {
        if entry > value {
            value = entry;
        }
    }

    value
}

/// Relative infinity-norm error.
fn relative_infinity_error(x: &[f64], exact: &[f64]) -> f64 {
    let error = difference(x, exact);
    infinity_norm(&error) / infinity_norm(exact).max(1.0e-14)
}

/// Count entries below zero.
fn count_negative_entries(x: &[f64]) -> usize {
    x.iter().filter(|&&value| value < 0.0).count()
}

/// Relative entropy H(u) = -sum_j u_j ln(u_j / m_j).
fn relative_entropy(u: &[f64], default: &[f64]) -> f64 {
    let mut entropy = 0.0;

    for j in 0..u.len() {
        entropy -= u[j] * (u[j] / default[j]).ln();
    }

    entropy
}

/// Construct a positive normalized image from log variables z.
fn image_from_log_variables(z: &[f64], default: &[f64], total_mass: f64) -> Vec<f64> {
    let mut scaled = vec![0.0; N];

    for j in 0..N {
        let exponent = z[j].clamp(-30.0, 30.0);
        scaled[j] = default[j] * exponent.exp();
    }

    let denominator = sum_entries(&scaled).max(1.0e-300);
    let mut image = vec![0.0; N];

    for j in 0..N {
        image[j] = total_mass * scaled[j] / denominator;
    }

    image
}

/// Compute chi^2(u) = sum_i ((Ku - g)_i / sigma)^2.
fn chi_squared(k_matrix: &[Vec<f64>], image: &[f64], data: &[f64], noise_std: f64) -> f64 {
    let predicted = mat_vec(k_matrix, image);
    let mut chi2 = 0.0;

    for i in 0..data.len() {
        let residual = (predicted[i] - data[i]) / noise_std;
        chi2 += residual * residual;
    }

    chi2
}

/// Objective 0.5 chi^2(u) - beta H(u).
fn mem_objective(
    z: &[f64],
    default: &[f64],
    total_mass: f64,
    k_matrix: &[Vec<f64>],
    data: &[f64],
    noise_std: f64,
    entropy_weight: f64,
) -> f64 {
    let image = image_from_log_variables(z, default, total_mass);
    let chi2 = chi_squared(k_matrix, &image, data, noise_std);
    let entropy = relative_entropy(&image, default);

    0.5 * chi2 - entropy_weight * entropy
}

/// Gradient of the maximum entropy objective with respect to z.
///
/// The gradient is obtained by first differentiating with respect to u, then
/// applying the derivative of the normalized exponential parametrization.
fn mem_gradient(
    z: &[f64],
    default: &[f64],
    total_mass: f64,
    k_matrix: &[Vec<f64>],
    data: &[f64],
    noise_std: f64,
    entropy_weight: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let image = image_from_log_variables(z, default, total_mass);
    let predicted = mat_vec(k_matrix, &image);

    let mut weighted_residual = vec![0.0; N];

    for i in 0..N {
        weighted_residual[i] = (predicted[i] - data[i]) / (noise_std * noise_std);
    }

    let kt_residual = transpose_mat_vec(k_matrix, &weighted_residual);

    let mut gradient_u = vec![0.0; N];

    for j in 0..N {
        gradient_u[j] =
            kt_residual[j] + entropy_weight * ((image[j] / default[j]).ln() + 1.0);
    }

    let mut weighted_average = 0.0;

    for j in 0..N {
        weighted_average += gradient_u[j] * image[j];
    }

    weighted_average /= total_mass;

    let mut gradient_z = vec![0.0; N];

    for j in 0..N {
        gradient_z[j] = image[j] * (gradient_u[j] - weighted_average);
    }

    let gradient_norm = l2_norm(&gradient_z);

    (gradient_z, image, gradient_norm)
}

/// Maximum absolute value in a vector.
fn max_abs_value(x: &[f64]) -> f64 {
    let mut value = 0.0;

    for &entry in x {
        if entry.abs() > value {
            value = entry.abs();
        }
    }

    value
}

/// Damped gradient descent for maximum entropy restoration.
///
/// This version uses a normalized descent direction and bounded log-intensity
/// updates. This avoids the unstable first-step collapse that can occur when
/// the raw chi-squared gradient is large because it is scaled by 1 / sigma^2.
fn maximum_entropy_reconstruction(
    default: &[f64],
    total_mass: f64,
    k_matrix: &[Vec<f64>],
    data: &[f64],
    noise_std: f64,
    entropy_weight: f64,
    tolerance: f64,
    max_iterations: usize,
    chi2_target: f64,
) -> (Vec<f64>, usize, f64, f64, f64) {
    let mut z = vec![0.0; N];

    let mut current_objective = mem_objective(
        &z,
        default,
        total_mass,
        k_matrix,
        data,
        noise_std,
        entropy_weight,
    );

    let mut final_gradient_norm = f64::INFINITY;
    let mut final_step_norm = f64::INFINITY;
    let mut final_chi2: f64;
    

    let max_log_update = 0.03;

    for iteration in 1..=max_iterations {
        let (gradient, image, gradient_norm) = mem_gradient(
            &z,
            default,
            total_mass,
            k_matrix,
            data,
            noise_std,
            entropy_weight,
        );

        final_gradient_norm = gradient_norm;
        final_chi2 = chi_squared(k_matrix, &image, data, noise_std);

        if final_gradient_norm < tolerance {
            return (image, iteration - 1, final_gradient_norm, 0.0, final_chi2);
        }

        // If the statistical data-consistency level is already reasonable and
        // the objective is changing slowly, stop. This is a practical stopping
        // criterion for noisy maximum entropy restoration.
        if final_chi2 <= 1.10 * chi2_target && final_step_norm < 5.0e-5 {
            return (
                image,
                iteration - 1,
                final_gradient_norm,
                final_step_norm,
                final_chi2,
            );
        }

        let scale = max_abs_value(&gradient).max(1.0e-14);
        let mut direction = vec![0.0; N];

        for j in 0..N {
            direction[j] = gradient[j] / scale;
        }

        let mut step_size = max_log_update;
        let mut accepted = false;
        let mut accepted_z = z.clone();
        let mut accepted_objective = current_objective;
        let mut accepted_step_norm = 0.0;

        for _ in 0..40 {
            let mut candidate_z = z.clone();

            for j in 0..N {
                candidate_z[j] -= step_size * direction[j];
            }

            let candidate_objective = mem_objective(
                &candidate_z,
                default,
                total_mass,
                k_matrix,
                data,
                noise_std,
                entropy_weight,
            );

            if candidate_objective < current_objective {
                let mut step_norm_squared = 0.0;

                for j in 0..N {
                    let step = candidate_z[j] - z[j];
                    step_norm_squared += step * step;
                }

                accepted_z = candidate_z;
                accepted_objective = candidate_objective;
                accepted_step_norm = step_norm_squared.sqrt();
                accepted = true;
                break;
            }

            step_size *= 0.5;
        }

        if !accepted {
            let image = image_from_log_variables(&z, default, total_mass);
            final_chi2 = chi_squared(k_matrix, &image, data, noise_std);
            return (
                image,
                iteration - 1,
                final_gradient_norm,
                final_step_norm,
                final_chi2,
            );
        }

        z = accepted_z;
        current_objective = accepted_objective;
        final_step_norm = accepted_step_norm;

        if final_step_norm < tolerance {
            let image = image_from_log_variables(&z, default, total_mass);
            final_chi2 = chi_squared(k_matrix, &image, data, noise_std);
            return (
                image,
                iteration,
                final_gradient_norm,
                final_step_norm,
                final_chi2,
            );
        }
    }

    let image = image_from_log_variables(&z, default, total_mass);
    final_chi2 = chi_squared(k_matrix, &image, data, noise_std);

    (
        image,
        max_iterations,
        final_gradient_norm,
        final_step_norm,
        final_chi2,
    )
}

/// Print representative values for comparison.
fn print_representative_values(exact: &[f64], blurred: &[f64], tikhonov: &[f64], mem: &[f64]) {
    println!("Representative Signal Values");
    println!("----------------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18} {:>18}",
        "j", "x_j", "u_exact", "g_blurred", "u_tikh", "u_MEM"
    );

    for j in 0..N {
        if j % 4 == 0 || j == N - 1 {
            let x = j as f64 / (N as f64 - 1.0);

            println!(
                "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.10} {:>18.10}",
                j, x, exact[j], blurred[j], tikhonov[j], mem[j]
            );
        }
    }

    println!();
}

fn main() {
    let blur_width = 0.055;
    let relative_noise = 1.5e-2;
    let tikhonov_lambda = 2.0e-3;

    // A moderate entropy weight keeps the reconstruction positive and
    // restrained without preventing statistical data consistency.
    let entropy_weight = 1.0e-2;

    let tolerance = 1.0e-9;
    let max_iterations = 20000;

    let exact = exact_signal();
    let total_mass = sum_entries(&exact);
    let default = default_model(total_mass);

    let k_matrix = blur_matrix(blur_width);
    let clean_blurred = mat_vec(&k_matrix, &exact);
    let noisy_blurred = add_deterministic_noise(&clean_blurred, relative_noise);

    let noise_std = relative_noise * infinity_norm(&clean_blurred);
    let chi2_target = N as f64;

    let tikhonov = match tikhonov_reconstruction(&k_matrix, &noisy_blurred, tikhonov_lambda) {
        Ok(value) => value,
        Err(message) => {
            eprintln!("Tikhonov reconstruction failed: {}", message);
            return;
        }
    };

    let mem_result = maximum_entropy_reconstruction(
        &default,
        total_mass,
        &k_matrix,
        &noisy_blurred,
        noise_std,
        entropy_weight,
        tolerance,
        max_iterations,
        chi2_target,
    );

    let mem = mem_result.0;
    let mem_iterations = mem_result.1;
    let mem_gradient_norm = mem_result.2;
    let mem_step_norm = mem_result.3;
    let mem_final_chi2 = mem_result.4;

    let tikh_predicted = mat_vec(&k_matrix, &tikhonov);
    let mem_predicted = mat_vec(&k_matrix, &mem);

    let tikh_residual = difference(&tikh_predicted, &noisy_blurred);
    let mem_residual = difference(&mem_predicted, &noisy_blurred);

    let actual_noise = difference(&noisy_blurred, &clean_blurred);

    let tikh_error = relative_infinity_error(&tikhonov, &exact);
    let mem_error = relative_infinity_error(&mem, &exact);

    let tikh_chi2 = chi_squared(&k_matrix, &tikhonov, &noisy_blurred, noise_std);
    let mem_chi2 = chi_squared(&k_matrix, &mem, &noisy_blurred, noise_std);

    println!("Maximum Entropy Deblurring Compared with Tikhonov Regularization");
    println!("================================================================");
    println!();
    println!("Problem Setup");
    println!("-------------");
    println!("Unknown pixels N            = {}", N);
    println!("Forward model               = row-normalized Gaussian blur");
    println!("Blur width                  = {:.6}", blur_width);
    println!("Relative noise level        = {:.6e}", relative_noise);
    println!("Assumed noise std           = {:.6e}", noise_std);
    println!("Total intensity constraint  = {:.10}", total_mass);
    println!("Default model               = smooth positive data-driven baseline");
    println!();

    println!("Solver Parameters");
    println!("-----------------");
    println!("Tikhonov lambda             = {:.6e}", tikhonov_lambda);
    println!("MEM entropy weight beta     = {:.6e}", entropy_weight);
    println!("MEM max iterations          = {}", max_iterations);
    println!("MEM tolerance               = {:.6e}", tolerance);
    println!("MEM chi^2 target            = {:.6e}", chi2_target);
    println!();

    println!("Data-Fit Diagnostics");
    println!("--------------------");
    println!("Actual noise norm           = {:.6e}", l2_norm(&actual_noise));
    println!("Tikhonov residual norm      = {:.6e}", l2_norm(&tikh_residual));
    println!("MEM residual norm           = {:.6e}", l2_norm(&mem_residual));
    println!("Tikhonov chi^2              = {:.6e}", tikh_chi2);
    println!("MEM chi^2                   = {:.6e}", mem_chi2);
    println!("MEM final chi^2 in solver   = {:.6e}", mem_final_chi2);
    println!("Nominal chi^2 scale         = {}", N);
    println!();

    println!("Reconstruction Diagnostics");
    println!("--------------------------");
    println!("Tikhonov relative error     = {:.6e}", tikh_error);
    println!("MEM relative error          = {:.6e}", mem_error);
    println!("Tikhonov minimum value      = {:.6e}", min_value(&tikhonov));
    println!("MEM minimum value           = {:.6e}", min_value(&mem));
    println!("Tikhonov negative entries   = {}", count_negative_entries(&tikhonov));
    println!("MEM negative entries        = {}", count_negative_entries(&mem));
    println!("Tikhonov maximum value      = {:.6e}", max_value(&tikhonov));
    println!("MEM maximum value           = {:.6e}", max_value(&mem));
    println!("Tikhonov total intensity    = {:.10}", sum_entries(&tikhonov));
    println!("MEM total intensity         = {:.10}", sum_entries(&mem));
    println!("MEM entropy                 = {:.6e}", relative_entropy(&mem, &default));
    println!();

    println!("MEM Iteration Diagnostics");
    println!("-------------------------");
    println!("Iterations performed        = {}", mem_iterations);
    println!("Final gradient norm         = {:.6e}", mem_gradient_norm);
    println!("Final step norm             = {:.6e}", mem_step_norm);
    println!();

    print_representative_values(&exact, &noisy_blurred, &tikhonov, &mem);

    println!("Interpretation");
    println!("--------------");
    println!("The Tikhonov method is linear and stabilizes the inverse problem through");
    println!("a quadratic smoothness penalty, but it does not automatically enforce");
    println!("positivity or total intensity. The maximum entropy reconstruction is");
    println!("nonlinear, positivity preserving, and normalized by construction. Its");
    println!("entropy term discourages unsupported structure while the chi-squared");
    println!("diagnostic measures agreement with the noisy blurred data.");
}
```

Program 19.8.3 illustrates that maximum entropy restoration and Tikhonov regularization stabilize inverse problems in fundamentally different ways. Tikhonov regularization is linear and uses a quadratic penalty to control roughness. It can produce an accurate reconstruction in many smooth deblurring problems, but positivity and total intensity are not automatically enforced by the method.

The maximum entropy reconstruction is nonlinear and positivity preserving by construction. Since the unknown signal is represented through normalized exponentials relative to a positive default model, every reconstructed value remains positive and the total intensity is preserved exactly. These properties are especially important when the unknown represents a physical intensity, brightness, density, concentration, or tracer activity.

The comparison also shows that maximum entropy should not be interpreted as universally more accurate than Tikhonov regularization in every numerical norm. Its advantage lies in enforcing physically meaningful positivity and entropy-based restraint. In some examples, the linear Tikhonov reconstruction may achieve a smaller pointwise error, while the maximum entropy reconstruction provides stronger structural guarantees.

The $\chi^2$ diagnostic connects the reconstruction to the noisy-data interpretation in equation (19.8.9). A useful maximum entropy reconstruction should not merely be positive; it should also remain statistically consistent with the measured data. The program therefore reports both the residual norm and $\chi^2$ value, making the balance between data agreement and entropy-based regularization explicit.

Overall, the program supports the main message of Sections 19.8.4 and 19.8.5: maximum entropy restoration is a principled nonlinear alternative to purely quadratic regularization for positive inverse problems. It is particularly appropriate when the unknown must remain nonnegative, the data are incomplete or noisy, and the reconstruction should avoid unsupported oscillatory structure.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/By6qml2jq7MgyHGa1mNU.1","tags":[]}

# 19.9. Conclusion

Throughout this chapter, we have explored the theory and numerical treatment of integral equations and inverse problems. Integral equations provide a powerful framework for modeling physical systems, memory effects, boundary phenomena, and operator-based formulations of differential equations. At the same time, inverse problems challenge us to recover unknown information from incomplete, noisy, or indirect observations. We examined numerical methods for Fredholm and Volterra equations, techniques for handling singular kernels, and a range of approaches for stabilizing ill-posed inverse problems. The chapter further introduced regularization theory, the Backus–Gilbert method, and maximum entropy reconstruction, illustrating how mathematical principles can be transformed into practical computational algorithms. By combining rigorous analysis with Rust implementations, readers are equipped to tackle a broad range of forward and inverse problems encountered in modern scientific computing.

## 19.9.1. Key Takeaways

- Integral equations provide an alternative formulation of many differential equations and often arise naturally in boundary-value problems, potential theory, scattering theory, and transport processes.
- Fredholm equations of the second kind are among the most important classes of integral equations because they are typically well posed and can be solved effectively using discretization methods such as Nyström, collocation, and Galerkin schemes.
- Volterra equations possess a causal structure that reflects the accumulation of past influences. Their lower-triangular discretizations often lead to efficient sequential algorithms with favorable computational properties.
- Singular kernels require specialized numerical treatment. Techniques such as variable transformations, product integration, singularity subtraction, and principal-value formulations allow accurate computation despite the presence of singular behavior.
- Inverse problems are frequently ill posed, meaning that solutions may not exist, may not be unique, or may depend sensitively on measurement errors. Stabilization therefore becomes a central computational concern.
- A priori information plays a crucial role in inverse problems by incorporating physical knowledge, smoothness assumptions, positivity constraints, or statistical models into the reconstruction process.
- Tikhonov regularization provides one of the most widely used frameworks for stabilizing inverse problems, while singular-value decomposition offers valuable insight into the filtering of unstable solution components.
- The Backus–Gilbert method emphasizes localized averages and resolution analysis, providing an alternative perspective that focuses on what information can actually be resolved from available data.
- Maximum entropy methods incorporate positivity and information-theoretic principles to produce physically meaningful reconstructions, particularly in imaging and astronomical applications.
- Rust provides an effective platform for implementing integral-equation solvers and inverse algorithms, combining computational efficiency, memory safety, and support for modern numerical libraries.

## 19.9.2. Advice for Beginners

- When studying integral equations for the first time, begin by understanding the distinction between Fredholm and Volterra formulations. Many numerical methods become much easier to understand once the structure of the underlying operator is clear.
- Start with simple Fredholm equations of the second kind and implement a Nyström discretization. This provides an excellent introduction to quadrature-based operator approximation and matrix formulations.
- Next, study Volterra equations and observe how causality leads naturally to lower-triangular systems. Comparing Fredholm and Volterra solvers is a useful exercise for understanding different computational structures.
- Before exploring advanced inverse problems, develop a strong understanding of ill-conditioning and the effects of measurement noise. Many inverse methods are motivated by the need to address these difficulties.
- Experiment with Tikhonov regularization and singular-value decomposition. Observing how regularization suppresses unstable solution components is one of the most effective ways to develop intuition about inverse problems.
- When implementing algorithms in Rust, begin with dense matrix formulations using libraries such as `nalgebra` or `ndarray`. Once the algorithms are understood, explore sparse representations and more advanced optimization strategies.
- Most importantly, remember that successful inverse modeling depends not only on numerical algorithms but also on understanding the underlying physics and the limitations imposed by the available data.

## 19.9.3. Further Learning with GenAI

To deepen your understanding of integral equations and inverse problems, consider exploring the following prompts:

 1. Explain the differences between Fredholm and Volterra integral equations and derive numerical discretizations for each.
 2. Implement a Nyström method in Rust for solving a Fredholm equation of the second kind and analyze its convergence properties.
 3. Compare Nyström, collocation, and Galerkin methods for the same integral equation and evaluate their accuracy and computational cost.
 4. Explain the role of singular kernels in boundary integral methods and demonstrate numerical techniques for handling them.
 5. Implement a Volterra integral equation solver in Rust and analyze how causal structure influences computational complexity.
 6. Derive Tikhonov regularization from both optimization and Bayesian perspectives and explain the relationship between the two formulations.
 7. Visualize the singular-value decomposition of an ill-conditioned inverse problem and illustrate how regularization filter factors stabilize the solution.
 8. Compare Tikhonov regularization, Backus–Gilbert inversion, and maximum entropy reconstruction for a noisy imaging problem.
 9. Implement a Backus–Gilbert reconstruction algorithm and analyze the trade-off between resolution and variance.
10. Develop a maximum entropy image restoration algorithm in Rust and compare its performance with classical regularized least-squares methods.

By exploring these prompts, readers can strengthen their understanding of both forward integral-equation models and inverse reconstruction techniques.

## 19.9.4. Homework Exercises

To reinforce your understanding of the material covered in this chapter, complete the following exercises:

 1. Derive a Nyström discretization for a Fredholm equation of the second kind and analyze the effect of quadrature order on solution accuracy.
 2. Implement and compare Nyström, collocation, and Galerkin methods for the same Fredholm problem.
 3. Construct a numerical solver for a Volterra integral equation and investigate how the solution changes when the kernel is modified.
 4. Implement a product-integration method for an integral equation with a weakly singular kernel and compare it with a standard quadrature approach.
 5. Demonstrate the effects of ill-conditioning by solving an inverse problem with increasing levels of measurement noise.
 6. Implement Tikhonov regularization and investigate how the choice of regularization parameter affects solution quality.
 7. Use singular-value decomposition to analyze an ill-posed inverse problem and explain the significance of the dominant and small singular values.
 8. Implement the Backus–Gilbert method and evaluate the relationship between resolution width and estimator variance.
 9. Apply maximum entropy reconstruction to a blurred one-dimensional signal and compare the results with Tikhonov regularization.
10. Select a real-world application such as tomography, astronomical image reconstruction, geophysical inversion, or system identification, formulate the associated inverse problem, and evaluate alternative reconstruction strategies.

Integral equations and inverse problems form an essential bridge between mathematical modeling and data-driven scientific discovery. The methods developed in this chapter provide the foundation for transforming observations into knowledge, reconstructing hidden structures, and solving complex operator equations that arise throughout modern science and engineering. As computational methods continue to evolve, these techniques remain central to imaging, remote sensing, machine learning, uncertainty quantification, and many other emerging fields. By mastering the concepts presented here and implementing them in Rust, readers will be well prepared to address some of the most challenging computational problems encountered in contemporary scientific research.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/L0FAP9HWs0J1xAmN17Xe.1","tags":[]}

# References

 1. Al Khaykanee, M.M.K. (2024) ‘Numerical Solution of Volterra’s Integral Equations by Collocation and Taylor Method’, *Journal of Al-Qadisiyah for Computer Science and Mathematics*, 16, pp. 49–54. doi: 10.29304/jqcsm.2024.16.21555.
 2. Chen, L. and Li, X. (2023) ‘Numerical calculation of regular and singular integrals in boundary integral equations using Clenshaw–Curtis quadrature rules’, *Engineering Analysis with Boundary Elements*, 155, pp. 25–37. doi: 10.1016/j.enganabound.2023.05.047.
 3. Fichtner, A. (2025) ‘A probabilistic multiparameter Backus–Gilbert inversion method’, *Geophysical Journal International*, 240(2), pp. 1064–1078. doi: 10.1093/gji/ggae430.
 4. Hamood, M.M., Sharif, A.A. and Ghadle, K.P. (2025) ‘A numerical approach to fractional Volterra–Fredholm integro-differential problems using shifted Chebyshev spectral collocation’, *Scientific Reports*, 15, 29678. doi: 10.1038/s41598-025-13732-7.
 5. Hristopulos, D.T. and Varouchakis, E.A. (2023) ‘Maximum entropy method’, in Daya Sagar, B.S., Cheng, Q., McKinley, J. and Agterberg, F. (eds) *Encyclopedia of Mathematical Geosciences*. Cham: Springer, pp. 1234–1238. doi: 10.1007/978-3-030-85040-1_196.
 6. Ma, Y. and Li, W. (2025) ‘Efficient solution of second-kind Fredholm integral equations using L¹-space discrete coordinates and physics-informed neural networks’, *Next Research*, 2(3), 100534. doi: 10.1016/j.nexres.2025.100534.
 7. Nature Research Intelligence (2025a) ‘Boundary Integral Equation Methods in Computational Analysis’, *Nature Index Topics*. Available at: Nature Index topic page (Accessed: 2 May 2026).
 8. Nature Research Intelligence (2025b) ‘Regularization Methods in Inverse Problem Solving’, *Nature Index Topics*. Available at: Nature Index topic page (Accessed: 2 May 2026).
 9. Nature Research Intelligence (2025c) ‘Tikhonov Regularization Methods for Ill-Posed Linear Systems’, *Nature Index Topics*. Available at: Nature Index topic page (Accessed: 2 May 2026).
10. Nityananda, R. (2024) ‘Maximum entropy image restoration in astronomy’, *Nature India Comment*, 2 January. doi: 10.1038/d44151-023-00175-0.
11. Pang, X. and Wang, J. (2025) ‘A new iterated Tikhonov regularization method for Fredholm integral equation of first kind’, *arXiv preprint*, arXiv:2504.00209.
12. Pooja, Kumar, J. and Manchanda, P. (2024) ‘Numerical Solution of First Kind Fredholm Integral Equations Using Wavelet Collocation Method’, *Journal of Advances in Mathematics and Computer Science*, 39(6), pp. 66–79. doi: 10.9734/jamcs/2024/v39i61902.

