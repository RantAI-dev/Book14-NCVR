---
title: Chapter 18
description: ''
subtitle: Two-Point Boundary Value Problems
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
date: '2026-04-26'
oxa: oxa:pqQDe4beUu67RvW3raYP/ZV4qFyf9VJCVZb1T4XUQ
keywords: []
---

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/WD7EPacoPz5G3eMLndqr.1","tags":[]}

> *The boundary conditions are the real data of the problem, and it is from these that the solution emerges. — Henri Poincaré*

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/fSXT7qwuUyV7KU8vtIJu.1","tags":[]}

*Chapter 18 introduces the numerical solution of two-point boundary value problems (TPBVPs), an important class of differential equations arising in science, engineering, physics, and optimal control. Unlike initial value problems, boundary value problems require the solution to satisfy conditions imposed at multiple locations, leading to globally coupled systems and unique numerical challenges. The chapter begins with the formulation of TPBVPs, residual representations, and their numerical interpretation. It then develops the shooting method, including Newton-based corrections, Jacobian evaluation techniques, conditioning issues, and modern improvements for enhanced robustness. Extensions such as fitting-point and multiple-shooting methods are presented to improve stability and convergence for difficult nonlinear problems. The chapter also introduces relaxation methods based on finite-difference discretization, emphasizing their efficiency and robustness for solving large coupled systems. A worked example involving spheroidal harmonics illustrates the practical application of both shooting and relaxation approaches to eigenvalue problems. Adaptive mesh allocation techniques are then examined as a means of improving accuracy while controlling computational cost. Finally, methods for handling internal boundary conditions, multipoint constraints, and singular points are discussed. Throughout the chapter, mathematical theory is integrated with practical Rust implementations, providing readers with the tools needed to develop accurate and efficient solvers for a wide range of boundary value problems.*

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/dWKns9h4IVT00RVyLjfZ.3","tags":[]}

# 18.1. Introduction

Two-point boundary value problems form a central class of problems in numerical analysis, distinguished from initial value problems by the presence of constraints imposed at more than one point of the independent variable. Instead of specifying all conditions at a single starting location, the solution must simultaneously satisfy differential equations over an interval and boundary conditions at both endpoints. This coupling across the domain introduces global dependencies that significantly influence both the analytical structure of the problem and the design of numerical algorithms. In contrast to initial value problems, where solutions can be propagated sequentially, boundary value problems require methods that enforce consistency across the entire interval.

From a computational perspective, this global nature leads to challenges in stability, convergence, and sensitivity. Small changes in boundary conditions can produce nonlocal effects throughout the solution, and numerical methods must ensure that these endpoint constraints are satisfied without sacrificing accuracy in the interior. Consequently, the study of TPBVPs serves as a bridge between differential equations, optimization, and numerical linear algebra, particularly when discretization leads to large coupled systems of equations.

## 18.1.1. Formulation of Two-Point Boundary Value Problems

A two-point boundary value problem (TPBVP) seeks a function:

$$\mathbf{y} : [a,b] \to \mathbb{R}^N$$

that satisfies a system of differential equations throughout a closed interval together with boundary conditions prescribed at both endpoints. In its standard first-order representation, the problem is written as:

$$\frac{d\mathbf y}{dx}=\mathbf g(x,\mathbf y,\mu), \qquad x\in[a,b], \tag{18.1.1}$$

with boundary conditions:

$$\mathbf B_1(\mathbf y(a),\mu)=\mathbf 0, \qquad \mathbf B_2(\mathbf y(b),\mu)=\mathbf 0 \tag{18.1.2}$$

Here, the unknown function $\mathbf y \in \mathbb{R}^N$ represents a vector of state variables, while $\mu$ denotes a set of auxiliary parameters that may either be specified or determined as part of the solution. The functions $\mathbf B_1$ and $\mathbf B_2$ impose constraints at the left and right endpoints, respectively, and map the state space into lower-dimensional spaces $\mathbb{R}^{n_1}$ and $\mathbb{R}^{n_2}$. The compatibility condition $n_1 + n_2 = N$ ensures that the total number of scalar boundary conditions matches the dimension of the system, thereby making the problem well-posed in the sense of having a unique solution under appropriate regularity assumptions.

The formulation in equation (18.1.1) is completely general, since any higher-order ordinary differential equation can be rewritten as a first-order system by introducing additional variables corresponding to successive derivatives. For example, an $m$-th order scalar equation can be transformed into an $m$-dimensional first-order system, with each new variable representing a derivative of the original unknown function. This reduction is not merely a formal device; it allows the use of unified numerical techniques applicable to systems of first-order equations.

A key structural feature of TPBVPs is that the boundary conditions are split across the domain. Unlike initial value problems, where the solution is determined by data at a single point, here the values at $x=a$ and $x=b$ must simultaneously satisfy the governing equations. This introduces a coupling between the solution behavior at opposite ends of the interval. As a result, the solution cannot generally be constructed by forward integration alone; instead, one must determine a function that globally satisfies both the differential equation and the boundary constraints.

The presence of parameters $\mu$ further enriches the formulation. In many applications, these parameters are not known a priori but must be adjusted so that the boundary conditions are satisfied. This leads to a nonlinear system in which both the function $\mathbf y(x)$ and the parameters $\mu$ are unknown, increasing the dimensionality and complexity of the problem.

Such problems arise naturally in a wide range of applications where physical, geometric, or optimality constraints are imposed at multiple locations. Steady-state reductions of partial differential equations often yield boundary value problems in one spatial dimension, where conditions are specified at the boundaries of a domain. In optimal control, TPBVPs emerge from the coupling of state equations with adjoint equations, together with initial conditions for the state and terminal conditions for the adjoint variables.

Modern applications highlight the breadth and importance of this formulation. Nonisothermal reaction–diffusion systems involve boundary conditions reflecting physical constraints at reactor walls. Multibody robotic control problems require the satisfaction of constraints at initial and final configurations. Powered-descent guidance and orbital transfer modeling involve trajectory design subject to conditions at launch and target states. These applications illustrate that TPBVPs are not merely theoretical constructs but arise in practical, high-dimensional systems with complex coupling between variables (Soares et al., 2024; Eichmeir et al., 2025; Eichmeir and Steiner, 2026; Zou and Jiang, 2026).

The general formulation given by equations (18.1.1) and (18.1.2) therefore provides a unifying framework for a broad class of problems, setting the stage for the numerical methods developed in subsequent sections, where the central challenge is to construct approximations that satisfy both the differential equations and the boundary conditions with high accuracy and stability.

## 18.1.2. Global Coupling and Conceptual Distinction

The defining feature that distinguishes a two-point boundary value problem from an initial value problem is the presence of *global coupling* across the interval $[a,b]$. In an initial value problem, the specification of the state $\mathbf y(a)$ completely determines the solution trajectory, at least locally, through the differential equation. Under suitable smoothness conditions, the solution can be obtained by straightforward forward integration, and the influence of the initial condition propagates unidirectionally from $x=a$ to $x=b$.

In contrast, a TPBVP does not generally prescribe the full state at the initial point. Instead, only a subset of the components is specified at $x=a$, with the remaining conditions imposed at $x=b$. This partitioning introduces degrees of freedom in the initial state, and only particular choices of these free components will yield a trajectory that satisfies the terminal boundary conditions. Consequently, the problem is no longer one of forward propagation, but rather one of determining a solution that is globally consistent with constraints at both endpoints.

This global coupling has several important implications. First, the solution depends simultaneously on information from both boundaries, so local integration alone is insufficient. Second, the problem becomes inherently nonlinear even when the underlying differential equation is linear, because the unknown initial conditions must be adjusted to satisfy the boundary constraints at $x=b$. Third, numerical methods must incorporate mechanisms for enforcing this global consistency, typically through iterative procedures that adjust unknown parameters or the entire solution profile.

The conceptual distinction becomes particularly evident in the context of optimal control. Indirect methods based on Pontryagin’s principle lead naturally to TPBVPs in which the state variables and associated costate variables satisfy coupled differential equations, with boundary conditions split between initial and terminal times. In this formulation, the unknown initial values of the costate variables must be chosen so that the terminal conditions are satisfied, leading to a highly sensitive global problem. By contrast, direct collocation methods reformulate the problem by discretizing the trajectory and converting it into a sparse nonlinear programming problem, thereby avoiding the explicit TPBVP structure. Contemporary studies indicate that direct methods tend to exhibit greater robustness for strongly constrained systems, while indirect TPBVP formulations can achieve higher accuracy when their intrinsic structure is effectively exploited (Zou and Jiang, 2026; Haman and Rao, 2025).

Thus, the notion of global coupling is not merely a conceptual distinction but a defining characteristic that shapes both the analytical properties of the problem and the strategies employed for its numerical solution.

## 18.1.3. Residual Formulation and Numerical Perspective

A unified and mathematically precise framework for analyzing TPBVPs is obtained by introducing a residual operator that encapsulates both the differential equation and the boundary conditions. Specifically, define:

\begin{equation}
\mathcal{R}[\mathbf{y}] =
\begin{bmatrix}
\mathbf{y}' - \mathbf{g}(x,\mathbf{y},\mu) \\[4pt]
\mathbf{B}_1(\mathbf{y}(a),\mu) \\[2pt]
\mathbf{B}_2(\mathbf{y}(b),\mu)
\end{bmatrix}
\tag{18.1.3}
\end{equation}

In this formulation, the first component measures the deviation from the differential equation throughout the interval, while the remaining components enforce the boundary conditions at $x=a$ and $x=b$. The two-point boundary value problem is then equivalently expressed as the operator equation:

$$\mathcal R[\mathbf y]=\mathbf 0 \tag{18.1.4}$$

This perspective highlights that a TPBVP is fundamentally a nonlinear operator equation posed in a function space. The unknown is the entire function $\mathbf y(x)$, and the goal is to find a function for which the residual vanishes identically. From a numerical standpoint, this viewpoint is particularly valuable because it provides a common foundation for different classes of solution methods.

All numerical approaches to TPBVPs can be interpreted as strategies for approximating the zero of the residual operator. Shooting methods reduce the infinite-dimensional problem to a finite-dimensional one by parameterizing the unknown initial conditions. In this case, one introduces a vector $\mathbf V$ representing the free components of $\mathbf y(a)$, integrates the differential equation forward, and defines a mismatch function that measures the violation of the boundary conditions at $x=b$. The problem is thereby transformed into a finite-dimensional root-finding problem.

In contrast, relaxation methods adopt a discretization-based approach. The interval $[a,b]$ is partitioned into a mesh, and the unknown function $\mathbf y(x)$ is approximated by a vector $\mathbf Y$ containing its values at the mesh points. The differential equation is then replaced by a system of algebraic equations derived from finite difference or collocation approximations, and the boundary conditions are incorporated directly into this system. The resulting problem is a large but structured nonlinear algebraic system, whose solution yields an approximation to the continuous problem.

These two perspectives can be summarized schematically as:

\begin{equation}
\boxed{\text{continuous TPBVP}}
\quad \Longrightarrow \quad
\begin{cases}
\text{shooting: } \mathbf{V} \mapsto \mathbf{F}(\mathbf{V}), \\[4pt]
\text{relaxation: } \mathbf{Y} \mapsto \mathbf{E}(\mathbf{Y})
\end{cases}
\tag{18.1.5}
\end{equation}

Here, $\mathbf V$ denotes a reduced set of parameters corresponding to the unspecified initial conditions, while $\mathbf Y$ represents the full discretized solution vector over the computational mesh. The functions $\mathbf F$ and $\mathbf E$ measure the residuals in the respective formulations, and their zeros correspond to approximate solutions of the original TPBVP.

This residual-based formulation provides a unifying lens through which all numerical methods for boundary value problems can be understood. It emphasizes that, regardless of the specific technique employed, the essential task is to construct an approximation that simultaneously satisfies the differential equation and the boundary conditions, thereby driving the residual to zero in an appropriate sense.

### Rust Implementation

Following the residual-based formulation introduced in Section 18.1.3, Program 18.1.1 provides a concrete implementation of the abstract operator equation $\mathcal R[\mathbf y] = \mathbf 0$ by constructing finite-dimensional residuals corresponding to both shooting and relaxation approaches. In the continuous formulation given by equations (18.1.3)–(18.1.5), the solution of a two-point boundary value problem is characterized by the vanishing of a nonlinear operator that combines the differential equation with the boundary conditions. From a computational perspective, this infinite-dimensional problem must be approximated by a finite-dimensional residual that can be evaluated and driven toward zero. The present program illustrates this transition explicitly by evaluating a shooting residual, obtained through forward integration with an unknown initial slope, and a relaxation residual, obtained from a finite-difference discretization of the entire solution profile. The implementation demonstrates how both approaches arise naturally from the same residual framework and provides a practical foundation for the numerical methods developed in subsequent sections.

At the core of the implementation are two complementary residual constructions that correspond directly to the mappings introduced in equation (18.1.5). The function `shooting_residual` evaluates the finite-dimensional function $\mathbf F(\mathbf V)$ by parameterizing the unknown initial condition through the scalar variable $V = y'(0)$. The differential equation is first rewritten as a first-order system, and the resulting initial value problem is integrated forward using a classical fourth-order Runge–Kutta method. The residual is then defined as the mismatch in the terminal boundary condition at $x=b$, thereby measuring the deviation from the second constraint in equation (18.1.2). In this way, the infinite-dimensional operator equation (18.1.4) is reduced to a scalar root-finding problem in the shooting parameter.

The relaxation formulation is implemented through the function `relaxation_residual`, which constructs the vector-valued function $\mathbf E(\mathbf Y)$ by discretizing the domain into a uniform mesh. The second derivative appearing in the differential equation is approximated using a centered finite-difference scheme, leading to a system of algebraic equations that approximate the first component of the residual operator in equation (18.1.3). The boundary conditions are incorporated directly into the residual vector by enforcing the endpoint constraints at the first and last mesh points. This produces a structured nonlinear system whose vanishing corresponds to an approximate solution of the original TPBVP.

To support the shooting formulation, the program includes the function `rk4_step`, which implements a single step of the classical fourth-order Runge–Kutta method for systems of first-order equations. This function is used iteratively to propagate the solution across the interval, providing a stable and accurate approximation of the continuous trajectory. The function `rhs` defines the vector field corresponding to the first-order system, while auxiliary functions such as `forcing`, `exact_solution`, and `exact_derivative` provide the analytical expressions required for validation and comparison.

The `main` function demonstrates the behavior of both residual formulations. It begins by evaluating the shooting residual for both a trial initial slope and the exact slope, illustrating how the residual decreases as the parameter approaches the correct value. It then constructs two discrete solution profiles: one corresponding to the exact solution and another containing a small perturbation. By computing the relaxation residual for both cases, the program shows that the exact solution yields a small residual up to discretization error, while the perturbed solution produces a significantly larger residual. This comparison highlights the role of the residual as a quantitative measure of how well a candidate function satisfies both the differential equation and the boundary conditions.

```rust
// Program 18.1.1: Residual Evaluation for a Simple Two-Point Boundary Value Problem
//
// Problem statement:
// This program illustrates the residual viewpoint for a two-point boundary value
// problem. The model problem is
//
//     y''(x) = -pi^2 sin(pi x),   0 <= x <= 1,
//     y(0) = 0,                   y(1) = 0.
//
// The exact solution is y(x) = sin(pi x). We evaluate two residuals:
//
// 1. A shooting residual F(V), where V is the unknown initial slope y'(0).
// 2. A relaxation residual E(Y), where Y stores approximate values of y on a mesh.
//
// The purpose is not yet to solve the TPBVP, but to show how the abstract
// operator R[y] = 0 becomes a finite-dimensional residual in numerical form.

use std::f64::consts::PI;

fn forcing(x: f64) -> f64 {
    -PI * PI * (PI * x).sin()
}

fn exact_solution(x: f64) -> f64 {
    (PI * x).sin()
}

fn exact_derivative(x: f64) -> f64 {
    PI * (PI * x).cos()
}

// Right-hand side for the first-order system
//
//     y1' = y2,
//     y2' = -pi^2 sin(pi x).
//
// Here y1 = y and y2 = y'.
fn rhs(x: f64, y: [f64; 2]) -> [f64; 2] {
    [y[1], forcing(x)]
}

// One classical fourth-order Runge-Kutta step for the first-order system.
fn rk4_step(x: f64, y: [f64; 2], h: f64) -> [f64; 2] {
    let k1 = rhs(x, y);

    let y2 = [y[0] + 0.5 * h * k1[0], y[1] + 0.5 * h * k1[1]];
    let k2 = rhs(x + 0.5 * h, y2);

    let y3 = [y[0] + 0.5 * h * k2[0], y[1] + 0.5 * h * k2[1]];
    let k3 = rhs(x + 0.5 * h, y3);

    let y4 = [y[0] + h * k3[0], y[1] + h * k3[1]];
    let k4 = rhs(x + h, y4);

    [
        y[0] + h * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        y[1] + h * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    ]
}

// Shooting residual F(V).
//
// The unknown shooting parameter V represents y'(0). We integrate from x = 0
// to x = 1 with y(0) = 0 and y'(0) = V. The residual is the mismatch in the
// right boundary condition y(1) = 0.
fn shooting_residual(initial_slope: f64, steps: usize) -> f64 {
    let a = 0.0;
    let b = 1.0;
    let h = (b - a) / steps as f64;

    let mut x = a;
    let mut y = [0.0, initial_slope];

    for _ in 0..steps {
        y = rk4_step(x, y, h);
        x += h;
    }

    y[0] - 0.0
}

// Relaxation residual E(Y).
//
// The vector y_values stores the approximate solution values on an equally
// spaced mesh. The interior residual is formed by the centered finite-difference
// approximation
//
//     (Y_{j-1} - 2Y_j + Y_{j+1}) / h^2 - f(x_j),
//
// where f(x) = -pi^2 sin(pi x). Boundary residuals enforce Y_0 = 0 and Y_n = 0.
fn relaxation_residual(y_values: &[f64]) -> Vec<f64> {
    let n = y_values.len() - 1;
    let h = 1.0 / n as f64;

    let mut residual = vec![0.0; y_values.len()];

    residual[0] = y_values[0] - 0.0;

    for j in 1..n {
        let xj = j as f64 * h;
        let second_derivative =
            (y_values[j - 1] - 2.0 * y_values[j] + y_values[j + 1]) / (h * h);

        residual[j] = second_derivative - forcing(xj);
    }

    residual[n] = y_values[n] - 0.0;

    residual
}

fn max_abs(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |current_max, value| current_max.max(value.abs()))
}

fn main() {
    let steps = 40;

    println!("Residual Evaluation for a Two-Point Boundary Value Problem");
    println!("==========================================================");
    println!();
    println!("Model problem:");
    println!("  y''(x) = -pi^2 sin(pi x),   0 <= x <= 1");
    println!("  y(0) = 0,                   y(1) = 0");
    println!("  Exact solution: y(x) = sin(pi x)");
    println!();

    let exact_initial_slope = exact_derivative(0.0);
    let trial_initial_slope = 3.0;

    let shooting_residual_trial = shooting_residual(trial_initial_slope, steps);
    let shooting_residual_exact = shooting_residual(exact_initial_slope, steps);

    println!("Shooting Residual F(V)");
    println!("----------------------");
    println!("Trial initial slope V              = {:.12}", trial_initial_slope);
    println!(
        "Residual F(V) = y(1; V) - 0       = {:.12e}",
        shooting_residual_trial
    );
    println!("Exact initial slope V              = {:.12}", exact_initial_slope);
    println!(
        "Residual F(V_exact)                = {:.12e}",
        shooting_residual_exact
    );
    println!();

    let mesh_points = 41;
    let h = 1.0 / (mesh_points - 1) as f64;

    let mut exact_grid_values = Vec::with_capacity(mesh_points);
    let mut perturbed_grid_values = Vec::with_capacity(mesh_points);

    for j in 0..mesh_points {
        let x = j as f64 * h;
        let exact = exact_solution(x);

        exact_grid_values.push(exact);

        let perturbation = 0.02 * x * (1.0 - x);
        perturbed_grid_values.push(exact + perturbation);
    }

    let residual_exact_grid = relaxation_residual(&exact_grid_values);
    let residual_perturbed_grid = relaxation_residual(&perturbed_grid_values);

    println!("Relaxation Residual E(Y)");
    println!("------------------------");
    println!(
        "Mesh intervals                      = {}",
        mesh_points - 1
    );
    println!("Step size h                         = {:.12}", h);
    println!(
        "max |E(Y_exact)|                    = {:.12e}",
        max_abs(&residual_exact_grid)
    );
    println!(
        "max |E(Y_perturbed)|                = {:.12e}",
        max_abs(&residual_perturbed_grid)
    );
    println!();

    println!("Representative Relaxation Residual Entries");
    println!("------------------------------------------");
    println!("{:>8} {:>14} {:>18} {:>18}", "j", "x_j", "Y_j", "E_j(Y)");

    for j in [0, 10, 20, 30, 40] {
        let x = j as f64 * h;
        println!(
            "{:>8} {:>14.8} {:>18.10} {:>18.10e}",
            j, x, perturbed_grid_values[j], residual_perturbed_grid[j]
        );
    }
}
```

Program 18.1.1 demonstrates how the abstract residual operator formulation of two-point boundary value problems can be translated into practical computational terms. By constructing both shooting and relaxation residuals, the implementation makes explicit the connection between the continuous operator equation (18.1.4) and the finite-dimensional problems solved in numerical practice.

The results illustrate two key aspects of the residual framework. First, in the shooting formulation, the residual provides a direct measure of the sensitivity of the terminal boundary condition to variations in the initial parameters. Second, in the relaxation formulation, the residual reflects both discretization error and deviations from the exact solution, thereby serving as a diagnostic tool for assessing solution quality.

The modular structure of the program allows these ideas to be extended naturally in subsequent sections. In particular, the shooting residual can be embedded within Newton or secant methods to determine the correct initial conditions, while the relaxation residual forms the basis of large-scale nonlinear solvers for discretized systems. This unified perspective reinforces the central theme of Section 18.1.3: regardless of the numerical strategy employed, the solution of a TPBVP is fundamentally the process of driving an appropriately defined residual to zero.

## 18.1.4. Local Theory and Modeling Contexts

The analytical and numerical behavior of two-point boundary value problems is most clearly understood through local linearization about a candidate solution. Let $\mathbf y^\star(x)$ denote a solution of the nonlinear problem. A perturbation of this solution can be written in the form $\mathbf y = \mathbf y^\star + \boldsymbol\eta$, where $\boldsymbol\eta(x)$ represents a small variation. Substituting this expression into the governing equations and retaining only first-order terms yields the linearized, or variational, problem:

\begin{equation}
\boldsymbol{\eta}' = \mathbf{g}_{\mathbf{y}}(x,\mathbf{y}^\star)\,\boldsymbol{\eta}, 
\qquad
\mathbf{B}_{1,\mathbf{y}}\,\boldsymbol{\eta}(a)=\mathbf{0}, 
\qquad
\mathbf{B}_{2,\mathbf{y}}\,\boldsymbol{\eta}(b)=\mathbf{0}
\tag{18.1.6}
\end{equation}

This system is a homogeneous linear TPBVP governing the evolution of infinitesimal perturbations. The matrix $\mathbf g_{\mathbf y}(x,\mathbf y^\star)$ denotes the Jacobian of the vector field with respect to the state variables, evaluated along the reference solution. Similarly, $\mathbf B_{1,\mathbf y}$ and $\mathbf B_{2,\mathbf y}$ represent the linearizations of the boundary operators.

A fundamental consequence of this formulation is that the local structure of the nonlinear problem is determined by the properties of the linearized system. If the only solution of equation (18.1.6) is the trivial solution $\boldsymbol\eta \equiv \mathbf 0$, then the original solution $\mathbf y^\star$ is locally isolated. In this case, small perturbations cannot satisfy both the differential equation and the boundary conditions simultaneously, which implies local uniqueness and well-posedness. From a numerical standpoint, this condition ensures that the associated Jacobian operator is nonsingular, a property that is essential for the success of Newton-type methods.

This observation explains the central role of Newton linearization in the numerical treatment of TPBVPs. Whether one adopts a shooting formulation or a relaxation-based discretization, the resulting nonlinear system is typically solved by iterative methods that repeatedly linearize the problem and solve the corresponding linear system. The convergence of these methods depends critically on the invertibility and conditioning of the linearized operator derived from (18.1.6). Thus, the local theory not only provides insight into existence and uniqueness but also directly informs the design and performance of numerical algorithms.

The structure of TPBVPs is further illuminated by considering representative modeling contexts in which such problems arise naturally. One important example is indirect optimal control. Consider a dynamical system governed by:

\begin{equation}
\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x},\mathbf{u},t), 
\qquad
J = \Phi(\mathbf{x}(t_f),t_f) + \int_{t_0}^{t_f} L(\mathbf{x},\mathbf{u},t)\,dt
\tag{18.1.7}
\end{equation}

Application of Pontryagin’s principle leads to a coupled system of differential equations for the state variables $\mathbf x(t)$ and costate variables, together with boundary conditions specified at different times. Typically, the state is prescribed at the initial time $t_0$, while the costate satisfies terminal conditions at $t_f$. This partitioning of boundary conditions produces a TPBVP in which the unknown initial values of the costate must be determined so that the terminal constraints are satisfied. The resulting problem inherits the global coupling and sensitivity characteristics discussed earlier, making it a prototypical example of a TPBVP.

A second representative context is provided by porous catalyst modeling. In such problems, steady-state reaction–diffusion equations are reduced to ordinary differential equations in a spatial coordinate, often radial. The boundary conditions reflect physical constraints: symmetry conditions at the center of the catalyst and mass or heat transfer conditions at the surface. These conditions are imposed at distinct spatial locations, leading naturally to a boundary value formulation. The resulting equations are frequently nonlinear and may exhibit sharp gradients, further emphasizing the need for robust numerical methods capable of resolving complex solution structures.

These application domains continue to drive the development of advanced numerical techniques. Adaptive mesh refinement is employed to concentrate computational effort in regions where the solution exhibits rapid variation, thereby improving accuracy without excessive cost. Continuation methods are used to trace solution branches as parameters vary, providing a systematic approach to handling sensitivity and multiplicity of solutions. Sparse automatic differentiation enables efficient and accurate computation of Jacobians required for Newton iterations, particularly in large-scale problems. Matrix-free Newton–Krylov solvers address the challenges posed by high-dimensional discretizations, allowing the solution of large linearized systems without explicit formation of the Jacobian matrix. These developments reflect the interplay between mathematical structure and computational strategy in the modern treatment of TPBVPs (Soares et al., 2024; Krishnakumar et al., 2025; Eichmeir et al., 2025; Zou and Jiang, 2026).

Taken together, the local linearization framework and the modeling contexts discussed above reinforce the central viewpoint of this chapter. Two-point boundary value problems are inherently nonlinear and globally coupled, and their effective numerical treatment requires methods that integrate local accuracy, as revealed by linearization, with the enforcement of global consistency embodied in the boundary conditions.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/Jus2yCAMgaZrRXBZ3ih4.5","tags":[]}

# 18.2. The Shooting Method

The shooting method provides one of the most direct and conceptually intuitive approaches for solving two-point boundary value problems. Its central idea is to convert the globally constrained boundary value problem into a sequence of initial value problems, supplemented by a finite-dimensional nonlinear system that enforces the terminal conditions. In doing so, the method leverages well-established techniques for initial value problems while addressing the global coupling through parameter adjustment.

From a structural perspective, the shooting method exploits the fact that the differential equation itself is naturally posed as an initial value problem. The difficulty lies not in integrating the equation, but in selecting the correct initial conditions so that the solution satisfies the boundary conditions at the opposite endpoint. The method therefore introduces a parameterization of the unknown components of the initial state and seeks values of these parameters that produce a consistent solution over the entire interval.

## 18.2.1. Basic Idea and Formulation

The shooting method transforms a two-point boundary value problem into an initial value problem together with a finite-dimensional nonlinear system. The left boundary conditions are used to parameterize all admissible initial states, while the right boundary conditions define a residual that must be driven to zero.

Suppose that the boundary conditions at $x=a$ determine $n_1$ components of the vector $\mathbf y(a)$. Since the full state lies in $\mathbb{R}^N$, there remain $q = n_2 = N - n_1$ components that are not specified at the left endpoint. These free components must be chosen so that the solution satisfies the boundary conditions at $x=b$.

To formalize this, introduce a parameter vector $\mathbf V \in \mathbb{R}^q$, which represents the degrees of freedom in the initial condition. The admissible initial states are then described by a *loading map:*

$$\mathbf y(a) = \boldsymbol\ell(\mathbf V) \tag{18.2.1}$$

constructed so that the left boundary conditions are automatically satisfied. That is, for all values of $\mathbf V$,

$$\mathbf B_1(\boldsymbol\ell(\mathbf V), \mu) = \mathbf 0 \tag{18.2.2}$$

This construction reduces the problem to selecting the correct parameter vector $\mathbf V$. For each choice of $\mathbf V$, the system:

$$\frac{d\mathbf y}{dx} = \mathbf g(x,\mathbf y,\mu)$$

is integrated as an initial value problem from $x=a$ to $x=b$, producing a trajectory $\mathbf y(x;\mathbf V)$.

The right boundary conditions are then evaluated at the endpoint $x=b$. In general, the computed solution will not satisfy these conditions exactly, and the discrepancy is measured by the *residual function:*

$$\mathbf F(\mathbf V) = \mathbf B_2(\mathbf y(b;\mathbf V), \mu) \tag{18.2.3}$$

The TPBVP is solved precisely when this residual vanishes, that is, when:

$$\mathbf F(\mathbf V) = \mathbf 0 \tag{18.2.4}$$

Thus, the original boundary value problem is reduced to solving a nonlinear system of (q) equations in the unknown parameter vector $\mathbf V$. This reduction is fundamental: instead of seeking an entire function $\mathbf y(x)$, one seeks a finite-dimensional vector that determines the correct initial condition.

From a numerical standpoint, this formulation highlights both the strengths and limitations of the shooting method. On one hand, it allows the direct use of high-accuracy initial value solvers, preserving their stability and efficiency. On the other hand, the mapping $\mathbf V \mapsto \mathbf F(\mathbf V)$ can be highly nonlinear and sensitive, especially for stiff or unstable systems, since small changes in $\mathbf V$ may lead to large deviations in $\mathbf y(b)$. As a result, solving equation (18.2.4) typically requires robust nonlinear solvers, often based on Newton or quasi-Newton iterations, together with careful evaluation of sensitivities.

The formulation (18.2.1)–(18.2.4) therefore encapsulates the essential idea of shooting: convert the global boundary value problem into a parameter identification problem, where the correct initial conditions are “shot” so that the trajectory lands precisely on the required boundary at $x=b$.

## 18.2.2. Scalar Example and Interpretation

The essential mechanism of the shooting method is most clearly illustrated in the case of a second-order scalar boundary value problem. Consider,

$$y'' = f(x,y,y'), \qquad y(a)=\alpha, \qquad y(b)=\beta \tag{18.2.5}$$

To bring this problem into the standard first-order framework, introduce the variables:

$$y_1 = y, \qquad y_2 = y',$$

so that the equation is rewritten as a system,

$$y_1' = y_2, \qquad y_2' = f(x,y_1,y_2)$$

The boundary conditions become:

$$y_1(a) = \alpha, \qquad y_1(b) = \beta$$

At the initial point $x=a$, the value $y_1(a)=\alpha$ is prescribed, but the slope $y_2(a)=y'(a)$ is not specified. This missing initial value represents the single degree of freedom in the problem, and it is natural to denote it by $s = y'(a)$. Thus, for each choice of $s$, one obtains a well-defined initial value problem:

$$y_1(a)=\alpha, \qquad y_2(a)=s$$

which can be integrated forward to $x=b$. The resulting solution depends parametrically on $s$, and in particular the terminal value $y_1(b)$ becomes a function of $s$, denoted by $y(b;s)$. The mismatch with the required boundary condition at $x=b$ is then measured by the scalar residual function:

$$F(s) = y(b;s) - \beta \tag{18.2.6}$$

The boundary value problem is solved when this residual vanishes, that is, when the chosen initial slope $s$ produces a trajectory that exactly reaches the prescribed value $y(b)=\beta$.

From an interpretive standpoint, this formulation provides a clear geometric picture of the shooting method. Each choice of the initial slope $s$ generates a trajectory starting at $(a,\alpha)$. These trajectories form a family of curves, and the objective is to select the unique trajectory that passes through the target point $(b,\beta)$. The process of “shooting” refers precisely to this adjustment of the initial slope so that the endpoint condition is satisfied.

Numerically, the procedure consists of the following iterative cycle: an initial guess for $s$ is made, the corresponding initial value problem is integrated, the deviation $F(s)$ at $x=b$ is computed, and the guess is updated based on this deviation. The update is typically performed using a root-finding method, such as Newton’s method or a secant iteration, applied to the nonlinear equation $F(s)=0$.

This scalar example encapsulates the fundamental characteristics of the shooting method. It demonstrates how a boundary value problem is reduced to a parameter identification problem, highlights the role of the residual as a measure of boundary mismatch, and illustrates the sensitivity of the solution to the choice of initial conditions. These features extend directly to higher-dimensional systems, where the parameter $s$ is replaced by a vector of unknown initial components, and the residual becomes a vector-valued function.

### Rust Implementation

Following the formulation developed in Section 18.2.1–18.2.2, Program 18.2.1 provides a practical implementation of the shooting method for a scalar two-point boundary value problem. In the theoretical framework, the original boundary value problem is reduced to a nonlinear equation in the unknown initial slope, as expressed through the residual function defined in equation (18.2.6). From a computational perspective, this transformation converts a globally constrained problem into a sequence of initial value problems coupled with a root-finding procedure. The present program realizes this idea by combining a high-order Runge–Kutta integrator with a secant iteration to determine the correct initial slope. It demonstrates how the shooting parameter is adjusted iteratively so that the computed trajectory satisfies the boundary condition at the terminal point, thereby illustrating the central mechanism of the shooting method in a concrete and numerically robust setting.

At the core of the implementation is the function `shooting_residual`, which defines the scalar residual $F(s)$ introduced in equation (18.2.6). This function encapsulates the essential mapping $s \mapsto F(s)$, where the unknown initial slope determines the solution trajectory through the differential equation. For each trial value of $s$, the associated initial value problem is solved, and the deviation from the boundary condition at $x=b$ is computed. This abstraction directly reflects the formulation given in equations (18.2.3)–(18.2.4), where the boundary value problem is recast as a root-finding problem.

The integration of the differential equation is performed by the function `integrate_with_slope`, which constructs the initial state according to the loading map described in equation (18.2.1). The system is rewritten in first-order form, and the resulting initial value problem is advanced from $x=a$ to $x=b$ using the classical fourth-order Runge–Kutta method implemented in `rk4_step`. The auxiliary function `rhs` defines the vector field corresponding to the first-order system, thereby completing the numerical realization of the differential equation.

The root-finding procedure is implemented through the function `secant_shooting`, which applies the secant method to the nonlinear equation $F(s)=0$. Starting from two initial guesses, the method iteratively updates the estimate of the shooting parameter using a finite-difference approximation of the derivative. This approach avoids explicit computation of the Jacobian while still achieving rapid convergence in the scalar case. The iteration process mirrors the general Newton-type framework described in equation (18.2.7), but replaces the exact derivative with a secant approximation, making it both simple and effective for one-dimensional problems.

The `main` function demonstrates the complete shooting procedure. It begins by specifying the boundary value problem and selecting initial guesses for the unknown slope. The secant iteration is then executed, and the convergence behavior is displayed through iteration history and final residual values. After convergence, the computed solution is evaluated at selected mesh points and compared with the exact analytical solution. This comparison provides a direct measure of numerical accuracy and illustrates how the correct choice of the shooting parameter ensures that the trajectory satisfies both boundary conditions.

```rust
// Program 18.2.1: Basic Shooting Method with Secant Root Finding
//
// Problem statement:
// This program solves a scalar two-point boundary value problem by the shooting
// method. The model problem is
//
//     y''(x) = -pi^2 sin(pi x),   0 <= x <= 1,
//     y(0) = 0,                   y(1) = 0.
//
// The exact solution is
//
//     y(x) = sin(pi x).
//
// The unknown initial slope s = y'(0) is adjusted so that the terminal
// residual
//
//     F(s) = y(1; s) - 0
//
// vanishes. The differential equation is rewritten as a first-order system and
// integrated by the classical fourth-order Runge-Kutta method. The scalar
// residual equation F(s) = 0 is solved by the secant method.

use std::f64::consts::PI;

#[derive(Clone, Copy)]
struct State {
    y1: f64, // y
    y2: f64, // y'
}

fn forcing(x: f64) -> f64 {
    -PI * PI * (PI * x).sin()
}

fn exact_solution(x: f64) -> f64 {
    (PI * x).sin()
}

fn exact_initial_slope() -> f64 {
    PI
}

// First-order system:
//
//     y1' = y2,
//     y2' = -pi^2 sin(pi x).
fn rhs(x: f64, state: State) -> State {
    State {
        y1: state.y2,
        y2: forcing(x),
    }
}

fn add_scaled(a: State, b: State, scale: f64) -> State {
    State {
        y1: a.y1 + scale * b.y1,
        y2: a.y2 + scale * b.y2,
    }
}

// One fourth-order Runge-Kutta step.
fn rk4_step(x: f64, state: State, h: f64) -> State {
    let k1 = rhs(x, state);

    let s2 = add_scaled(state, k1, 0.5 * h);
    let k2 = rhs(x + 0.5 * h, s2);

    let s3 = add_scaled(state, k2, 0.5 * h);
    let k3 = rhs(x + 0.5 * h, s3);

    let s4 = add_scaled(state, k3, h);
    let k4 = rhs(x + h, s4);

    State {
        y1: state.y1 + h * (k1.y1 + 2.0 * k2.y1 + 2.0 * k3.y1 + k4.y1) / 6.0,
        y2: state.y2 + h * (k1.y2 + 2.0 * k2.y2 + 2.0 * k3.y2 + k4.y2) / 6.0,
    }
}

// Integrates the initial value problem generated by a trial slope s.
//
// The loading map is
//
//     y(0)  = alpha,
//     y'(0) = s.
//
// The returned value is the terminal state at x = b.
fn integrate_with_slope(
    initial_slope: f64,
    alpha: f64,
    a: f64,
    b: f64,
    steps: usize,
) -> State {
    let h = (b - a) / steps as f64;

    let mut x = a;
    let mut state = State {
        y1: alpha,
        y2: initial_slope,
    };

    for _ in 0..steps {
        state = rk4_step(x, state, h);
        x += h;
    }

    state
}

// Shooting residual:
//
//     F(s) = y(b; s) - beta.
fn shooting_residual(
    initial_slope: f64,
    alpha: f64,
    beta: f64,
    a: f64,
    b: f64,
    steps: usize,
) -> f64 {
    let terminal_state = integrate_with_slope(initial_slope, alpha, a, b, steps);
    terminal_state.y1 - beta
}

struct SecantResult {
    slope: f64,
    residual: f64,
    iterations: usize,
    converged: bool,
}

// Solves F(s) = 0 using the secant method.
fn secant_shooting(
    s0: f64,
    s1: f64,
    alpha: f64,
    beta: f64,
    a: f64,
    b: f64,
    steps: usize,
    tolerance: f64,
    max_iterations: usize,
) -> SecantResult {
    let mut previous_slope = s0;
    let mut current_slope = s1;

    let mut previous_residual =
        shooting_residual(previous_slope, alpha, beta, a, b, steps);
    let mut current_residual =
        shooting_residual(current_slope, alpha, beta, a, b, steps);

    println!("Secant Iteration History");
    println!("------------------------");
    println!(
        "{:>6} {:>18} {:>18}",
        "iter", "s", "F(s)"
    );
    println!(
        "{:>6} {:>18.10} {:>18.10e}",
        0, previous_slope, previous_residual
    );
    println!(
        "{:>6} {:>18.10} {:>18.10e}",
        1, current_slope, current_residual
    );

    for iteration in 2..=max_iterations {
        let denominator = current_residual - previous_residual;

        if denominator.abs() < 1.0e-14 {
            return SecantResult {
                slope: current_slope,
                residual: current_residual,
                iterations: iteration - 1,
                converged: false,
            };
        }

        let next_slope =
            current_slope - current_residual * (current_slope - previous_slope) / denominator;

        let next_residual =
            shooting_residual(next_slope, alpha, beta, a, b, steps);

        println!(
            "{:>6} {:>18.10} {:>18.10e}",
            iteration, next_slope, next_residual
        );

        if next_residual.abs() < tolerance {
            return SecantResult {
                slope: next_slope,
                residual: next_residual,
                iterations: iteration,
                converged: true,
            };
        }

        previous_slope = current_slope;
        previous_residual = current_residual;

        current_slope = next_slope;
        current_residual = next_residual;
    }

    SecantResult {
        slope: current_slope,
        residual: current_residual,
        iterations: max_iterations,
        converged: false,
    }
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let alpha = 0.0;
    let beta = 0.0;

    let steps = 80;
    let tolerance = 1.0e-10;
    let max_iterations = 20;

    let s0 = 2.5;
    let s1 = 3.5;

    println!("Basic Shooting Method for a Two-Point Boundary Value Problem");
    println!("============================================================");
    println!();
    println!("Model problem:");
    println!("  y''(x) = -pi^2 sin(pi x),   0 <= x <= 1");
    println!("  y(0) = 0,                   y(1) = 0");
    println!("  Exact solution: y(x) = sin(pi x)");
    println!();
    println!("Shooting formulation:");
    println!("  Unknown initial slope: s = y'(0)");
    println!("  Residual: F(s) = y(1; s) - 0");
    println!();

    let result = secant_shooting(
        s0,
        s1,
        alpha,
        beta,
        a,
        b,
        steps,
        tolerance,
        max_iterations,
    );

    let terminal_state = integrate_with_slope(result.slope, alpha, a, b, steps);

    println!();
    println!("Final Shooting Result");
    println!("---------------------");
    println!("Converged                         = {}", result.converged);
    println!("Iterations performed              = {}", result.iterations);
    println!("Computed initial slope s          = {:.12}", result.slope);
    println!("Exact initial slope pi            = {:.12}", exact_initial_slope());
    println!(
        "Absolute slope error              = {:.12e}",
        (result.slope - exact_initial_slope()).abs()
    );
    println!("Final residual F(s)               = {:.12e}", result.residual);
    println!("Computed terminal value y(1)      = {:.12}", terminal_state.y1);
    println!("Required terminal value beta      = {:.12}", beta);
    println!();

    println!("Representative Solution Values");
    println!("------------------------------");
    println!("{:>8} {:>14} {:>18} {:>18} {:>18}", "j", "x", "y_num", "y_exact", "abs error");

    for j in [0, 20, 40, 60, 80] {
        let x_target = a + (b - a) * j as f64 / steps as f64;
        let mut x = a;
        let h = (b - a) / steps as f64;
        let mut state = State {
            y1: alpha,
            y2: result.slope,
        };

        for _ in 0..j {
            state = rk4_step(x, state, h);
            x += h;
        }

        let exact = exact_solution(x_target);

        println!(
            "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.10e}",
            j,
            x_target,
            state.y1,
            exact,
            (state.y1 - exact).abs()
        );
    }
}
```

Program 18.2.1 demonstrates the practical realization of the shooting method by combining initial value integration with nonlinear root finding. This implementation reflects the central idea of Section 18.2: the reduction of a two-point boundary value problem to a parameter identification problem governed by the residual function $F(s)$.

The results illustrate how the residual serves as a quantitative measure of boundary mismatch and how iterative updates of the shooting parameter progressively reduce this mismatch to zero. The rapid convergence observed in the secant iteration highlights the effectiveness of the method in low-dimensional settings, particularly when the underlying differential equation is well-behaved.

At the same time, the implementation provides insight into the strengths and limitations of the shooting approach. While it leverages accurate initial value solvers and is straightforward to implement, its performance depends on the sensitivity of the mapping $s \mapsto y(b;s)$. This observation motivates the more advanced techniques discussed in subsequent subsections, where improved Jacobian approximations and sensitivity analysis are introduced.

Overall, the program establishes a clear computational foundation for the shooting method and prepares the ground for more sophisticated formulations, including Newton-based methods and sensitivity-driven approaches.

## 18.2.3. Newton Iteration and Finite-Difference Jacobians

In the general vector formulation of the shooting method, the nonlinear system $\mathbf F(\mathbf V)=\mathbf 0$, is solved using Newton’s method. This approach is a natural consequence of the residual formulation introduced earlier, where the objective is to drive the boundary mismatch to zero by iteratively refining the parameter vector $\mathbf V$.

Given an iterate $\mathbf V_k$, Newton’s method constructs a local linear approximation of the nonlinear mapping $\mathbf F$ and computes a correction $\Delta \mathbf V_k$ by solving:

$$\mathbf J(\mathbf V_k),\Delta \mathbf V_k = -\mathbf F(\mathbf V_k),\qquad\mathbf V_{k+1} = \mathbf V_k + \alpha_k \Delta \mathbf V_k, \tag{18.2.7}$$

where $\alpha_k \in (0,1]$ is a damping parameter. The inclusion of $\alpha_k$ is important in practice, as it improves robustness by controlling the step length, particularly when the current iterate is far from the solution or when the nonlinear mapping exhibits strong sensitivity.

The matrix,

$$\mathbf J(\mathbf V) = \frac{\partial \mathbf F}{\partial \mathbf V} \tag{18.2.8}$$

is the Jacobian of the residual with respect to the parameter vector. Its entries measure how perturbations in the initial conditions affect the boundary mismatch at $x=b$. From a computational perspective, this Jacobian encapsulates the sensitivity of the terminal state $\mathbf y(b;\mathbf V)$ with respect to the initial parameters, making it central to the convergence behavior of the method.

When analytical expressions for $\mathbf J$ are not readily available, a common approach is to approximate it using finite differences. In this case, each column of the Jacobian is obtained by perturbing one component of $\mathbf V$, re-integrating the initial value problem, and measuring the resulting change in $\mathbf F(\mathbf V)$. Consequently, for a parameter vector of dimension $q$, each Newton iteration requires approximately $q$ additional integrations, plus one for evaluating $\mathbf F(\mathbf V_k)$.

This leads to the computational cost estimate:

$$T_{\text{FD-shoot}} \approx (q+1) \,C_{\text{IVP}} + O(q^3) \tag{18.2.9}$$

where $C_{\text{IVP}}$ denotes the cost of a single initial value problem solve. The term $O(q^3)$ arises from solving the dense linear system associated with the Newton step. This cubic cost is typically negligible when $q$ is small, but it becomes significant as the number of free parameters increases.

The associated memory requirement is:

$$S_{\text{FD-shoot}} = O(N + q^2) \tag{18.2.10}$$

where $O(N)$ accounts for storing the state variables during integration, and $O(q^2)$ corresponds to storing the dense Jacobian matrix.

These estimates highlight a key strength of the shooting method. When the number of free parameters $q$ is small, the method is highly efficient, since it leverages accurate initial value solvers and requires only a modest number of additional integrations per iteration. However, the dependence on repeated integrations and the formation of a dense Jacobian also indicate potential limitations for problems with large parameter dimensions or strong sensitivity, where the cost and conditioning of the Newton system may become prohibitive.

Thus, the effectiveness of Newton-based shooting methods is closely tied to the dimensionality of the parameter space and the sensitivity of the solution with respect to the initial conditions, reinforcing the importance of careful formulation and implementation in practical applications.

## 18.2.4. Variational Equations and Sensitivity-Based Jacobians

The finite-difference approximation of the Jacobian, while straightforward, can become computationally expensive due to the repeated integrations of the initial value problem. A more efficient and mathematically structured alternative is obtained by introducing *variational equations*, which directly propagate the sensitivity of the solution with respect to the shooting parameters.

To formalize this idea, define the sensitivity matrix:

$$\mathbf S(x) = \frac{\partial \mathbf y(x;\mathbf V)}{\partial \mathbf V} \in \mathbb{R}^{N \times q} \tag{18.2.11}$$

Each column of $\mathbf S(x)$ represents the derivative of the solution $\mathbf y(x;\mathbf V)$ with respect to one component of the parameter vector $\mathbf V$. Thus, $\mathbf S(x)$ captures how perturbations in the initial conditions influence the solution at any point $x$ in the interval.

To derive an equation for $\mathbf S(x)$, differentiate the governing system:

$$\frac{d\mathbf y}{dx} = \mathbf g(x,\mathbf y,\mu)$$

with respect to $\mathbf V$. Applying the chain rule yields the linear matrix differential equation:

$$\mathbf S'(x) = \mathbf g_{\mathbf y}(x,\mathbf y(x;\mathbf V))\,\mathbf S(x),\qquad\mathbf S(a) = \boldsymbol\ell_{\mathbf V}(\mathbf V) \tag{18.2.12}$$

Here, $\mathbf g_{\mathbf y}$ denotes the Jacobian of the vector field with respect to the state variables, evaluated along the trajectory $\mathbf y(x;\mathbf V)$. The initial condition $\mathbf S(a)$ is obtained by differentiating the loading map $\boldsymbol\ell(\mathbf V)$, ensuring consistency with the parameterization of admissible initial states.

This system of variational equations evolves alongside the original state equations. Importantly, it is linear in $\mathbf S$, although it depends on the nonlinear state trajectory $\mathbf y(x;\mathbf V)$. By integrating the augmented system consisting of $\mathbf y(x)$ and $\mathbf S(x)$ simultaneously from $x=a$ to $x=b$, one obtains both the solution and its sensitivities in a single computation.

The Jacobian of the shooting residual is then obtained by applying the chain rule to the boundary condition at $x=b$:

$$\mathbf J(\mathbf V) = \mathbf B_{2,\mathbf y}(\mathbf y(b;\mathbf V))\,\mathbf S(b) \tag{18.2.13}$$

This expression has a clear interpretation. The matrix $\mathbf S(b)$ describes how the terminal state depends on the initial parameters, while $\mathbf B_{2,\mathbf y}$ maps variations in the terminal state to variations in the boundary residual. Their product therefore yields the full sensitivity of the residual $\mathbf F(\mathbf V)$ with respect to $\mathbf V$.

From a computational standpoint, this formulation offers significant advantages. Instead of requiring $q+1$ separate integrations per Newton iteration, the sensitivity-based approach requires only a *single augmented integration* of dimension $N + Nq$. This reduces the computational cost when $q$ is moderate and eliminates the truncation errors associated with finite-difference approximations. Moreover, the resulting Jacobian is typically more accurate, which improves the convergence behavior of Newton’s method.

At the same time, the structure of the variational equations reflects the underlying linearization discussed earlier in Section 18.1.4. The matrix $\mathbf g_{\mathbf y}$ governing the evolution of $\mathbf S(x)$ is precisely the same Jacobian that appears in the local theory, reinforcing the connection between sensitivity analysis and Newton linearization.

Thus, the use of variational equations provides a principled and efficient mechanism for computing Jacobians in shooting methods. It replaces repeated independent integrations with a single, coupled computation that simultaneously captures both the solution and its dependence on the initial parameters, thereby enhancing both efficiency and numerical reliability.

### Rust Implementation

Following the development of Newton iteration and Jacobian construction in Sections 18.2.3 and 18.2.4, Program 18.2.2 provides a practical implementation of Newton-based shooting for a vector two-point boundary value problem. In the theoretical formulation, the boundary value problem is reduced to solving the nonlinear system $\mathbf F(\mathbf V)=\mathbf 0$ using Newton’s method, as described in equation (18.2.7), where the Jacobian matrix defined in equation (18.2.8) plays a central role in determining convergence. When analytical derivatives are not readily available, finite-difference approximations offer a straightforward but computationally expensive alternative, as reflected in the cost estimate (18.2.9). In contrast, the introduction of variational equations in Section 18.2.4 provides a more efficient and structured approach to computing sensitivities. The present program implements both strategies, enabling a direct comparison between finite-difference and sensitivity-based Jacobians, and thereby illustrating their impact on computational efficiency and convergence behavior in Newton shooting methods.

At the core of the implementation is the function `newton_shooting`, which realizes the Newton iteration described in equation (18.2.7). Given an initial guess for the parameter vector $\mathbf V$, this function evaluates the residual $\mathbf F(\mathbf V)$, constructs an approximation to the Jacobian $\mathbf J(\mathbf V)$, and computes the correction $\Delta \mathbf V$ by solving a linear system. The update is applied with a damping parameter $\alpha_k$, which is determined through a simple backtracking strategy to ensure reduction of the residual norm. This reflects the practical importance of step control in maintaining robustness, particularly when the residual mapping exhibits strong sensitivity.

The function `shooting_residual` implements the mapping $\mathbf V \mapsto \mathbf F(\mathbf V)$ defined in equations (18.2.3)–(18.2.4). For each parameter vector, the associated initial value problem is constructed through the loading map described in equation (18.2.1), and the resulting system is integrated from $x=a$ to $x=b$. The mismatch in the terminal boundary conditions is then returned as a vector-valued residual. This function provides the fundamental link between the continuous boundary value problem and its finite-dimensional nonlinear representation.

Two distinct approaches are used to construct the Jacobian matrix $\mathbf J(\mathbf V)$. The function `finite_difference_jacobian` implements the approximation described in Section 18.2.3, where each column of the Jacobian is obtained by perturbing a single component of $\mathbf V$ and re-evaluating the residual. This approach is simple and broadly applicable, but it requires multiple integrations of the initial value problem per iteration, consistent with the cost estimate (18.2.9). The function `variational_residual_and_jacobian`, on the other hand, implements the sensitivity-based approach derived in equations (18.2.11)–(18.2.13). In this formulation, the sensitivity matrix $\mathbf S(x)$ is propagated alongside the state variables using the variational equations, allowing both the residual and the Jacobian to be computed simultaneously in a single augmented integration.

The numerical integration of both the state equations and the variational equations is performed using the classical fourth-order Runge–Kutta method. The functions `rk4_step_state` and `rk4_step_augmented` implement this procedure for the state-only and augmented systems, respectively. The function `rhs` defines the underlying differential system, while `jacobian_state` provides the Jacobian of the vector field required for the variational equations. Together, these components ensure a consistent and accurate discretization of both the solution and its sensitivities.

The `main` function demonstrates the complete Newton shooting procedure for both Jacobian constructions. Starting from an initial guess, it performs iterations using either the finite-difference or variational approach and reports convergence statistics, including the number of initial value problem integrations required. The results clearly illustrate the efficiency advantage of the variational formulation, which achieves convergence with significantly fewer integrations while maintaining comparable accuracy. This comparison highlights the practical importance of sensitivity-based Jacobians in improving the performance of Newton shooting methods.

```rust
// Program 18.2.2: Newton Shooting with Finite-Difference and Variational Jacobians
//
// Problem statement:
// This program demonstrates Newton-based shooting for a vector two-point
// boundary value problem. The model problem consists of two independent
// second-order equations:
//
//     y''(x) = -pi^2 sin(pi x),
//     z''(x) = -4pi^2 sin(2pi x),        0 <= x <= 1,
//
// with boundary conditions
//
//     y(0) = 0,   z(0) = 0,
//     y(1) = 0,   z(1) = 0.
//
// The exact solution is
//
//     y(x) = sin(pi x),
//     z(x) = sin(2pi x).
//
// The unknown shooting vector is
//
//     V = [y'(0), z'(0)]^T.
//
// The program solves F(V) = 0 by Newton's method using two Jacobian
// constructions:
//
// 1. A finite-difference Jacobian, corresponding to Section 18.2.3.
// 2. A variational-equation Jacobian, corresponding to Section 18.2.4.

use std::f64::consts::PI;

const N: usize = 4;
const Q: usize = 2;

type Vector2 = [f64; Q];
type State = [f64; N];
type Matrix2 = [[f64; Q]; Q];
type Sensitivity = [[f64; Q]; N];

fn rhs(x: f64, y: &State) -> State {
    [
        y[1],
        -PI * PI * (PI * x).sin(),
        y[3],
        -4.0 * PI * PI * (2.0 * PI * x).sin(),
    ]
}

fn jacobian_state() -> [[f64; N]; N] {
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
}

fn add_scaled_state(a: &State, b: &State, scale: f64) -> State {
    let mut result = [0.0; N];
    for i in 0..N {
        result[i] = a[i] + scale * b[i];
    }
    result
}

fn rk4_step_state(x: f64, y: &State, h: f64) -> State {
    let k1 = rhs(x, y);

    let y2 = add_scaled_state(y, &k1, 0.5 * h);
    let k2 = rhs(x + 0.5 * h, &y2);

    let y3 = add_scaled_state(y, &k2, 0.5 * h);
    let k3 = rhs(x + 0.5 * h, &y3);

    let y4 = add_scaled_state(y, &k3, h);
    let k4 = rhs(x + h, &y4);

    let mut result = [0.0; N];
    for i in 0..N {
        result[i] = y[i] + h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }

    result
}

fn initial_state(v: &Vector2) -> State {
    [0.0, v[0], 0.0, v[1]]
}

fn integrate_state(v: &Vector2, steps: usize) -> State {
    let a = 0.0;
    let b = 1.0;
    let h = (b - a) / steps as f64;

    let mut x = a;
    let mut y = initial_state(v);

    for _ in 0..steps {
        y = rk4_step_state(x, &y, h);
        x += h;
    }

    y
}

fn shooting_residual(v: &Vector2, steps: usize) -> Vector2 {
    let terminal = integrate_state(v, steps);

    [
        terminal[0], // y(1) - 0
        terminal[2], // z(1) - 0
    ]
}

fn finite_difference_jacobian(v: &Vector2, steps: usize, epsilon: f64) -> Matrix2 {
    let base = shooting_residual(v, steps);
    let mut jacobian = [[0.0; Q]; Q];

    for col in 0..Q {
        let mut perturbed = *v;
        perturbed[col] += epsilon;

        let residual_perturbed = shooting_residual(&perturbed, steps);

        for row in 0..Q {
            jacobian[row][col] = (residual_perturbed[row] - base[row]) / epsilon;
        }
    }

    jacobian
}

fn sensitivity_rhs(_x: f64, s: &Sensitivity) -> Sensitivity {
    let gy = jacobian_state();
    let mut result = [[0.0; Q]; N];

    for i in 0..N {
        for j in 0..Q {
            let mut sum = 0.0;
            for k in 0..N {
                sum += gy[i][k] * s[k][j];
            }
            result[i][j] = sum;
        }
    }

    result
}

fn add_scaled_sensitivity(
    a: &Sensitivity,
    b: &Sensitivity,
    scale: f64,
) -> Sensitivity {
    let mut result = [[0.0; Q]; N];

    for i in 0..N {
        for j in 0..Q {
            result[i][j] = a[i][j] + scale * b[i][j];
        }
    }

    result
}

fn rk4_step_augmented(
    x: f64,
    y: &State,
    s: &Sensitivity,
    h: f64,
) -> (State, Sensitivity) {
    let k1_y = rhs(x, y);
    let k1_s = sensitivity_rhs(x, s);

    let y2 = add_scaled_state(y, &k1_y, 0.5 * h);
    let s2 = add_scaled_sensitivity(s, &k1_s, 0.5 * h);
    let k2_y = rhs(x + 0.5 * h, &y2);
    let k2_s = sensitivity_rhs(x + 0.5 * h, &s2);

    let y3 = add_scaled_state(y, &k2_y, 0.5 * h);
    let s3 = add_scaled_sensitivity(s, &k2_s, 0.5 * h);
    let k3_y = rhs(x + 0.5 * h, &y3);
    let k3_s = sensitivity_rhs(x + 0.5 * h, &s3);

    let y4 = add_scaled_state(y, &k3_y, h);
    let s4 = add_scaled_sensitivity(s, &k3_s, h);
    let k4_y = rhs(x + h, &y4);
    let k4_s = sensitivity_rhs(x + h, &s4);

    let mut next_y = [0.0; N];
    let mut next_s = [[0.0; Q]; N];

    for i in 0..N {
        next_y[i] =
            y[i] + h * (k1_y[i] + 2.0 * k2_y[i] + 2.0 * k3_y[i] + k4_y[i]) / 6.0;

        for j in 0..Q {
            next_s[i][j] = s[i][j]
                + h * (k1_s[i][j] + 2.0 * k2_s[i][j] + 2.0 * k3_s[i][j] + k4_s[i][j])
                    / 6.0;
        }
    }

    (next_y, next_s)
}

fn variational_residual_and_jacobian(v: &Vector2, steps: usize) -> (Vector2, Matrix2) {
    let a = 0.0;
    let b = 1.0;
    let h = (b - a) / steps as f64;

    let mut x = a;
    let mut y = initial_state(v);

    let mut s = [[0.0; Q]; N];

    // S(0) = d y(0; V) / dV.
    //
    // Since V = [y'(0), z'(0)]^T, the sensitivities are:
    //
    // d[y(0), y'(0), z(0), z'(0)]^T / dV
    //
    // column 0 = [0, 1, 0, 0]^T,
    // column 1 = [0, 0, 0, 1]^T.
    s[1][0] = 1.0;
    s[3][1] = 1.0;

    for _ in 0..steps {
        let next = rk4_step_augmented(x, &y, &s, h);
        y = next.0;
        s = next.1;
        x += h;
    }

    let residual = [y[0], y[2]];

    // The terminal boundary map selects y(1) and z(1), so B_{2,y}
    // selects rows 0 and 2 of the terminal sensitivity matrix S(1).
    let jacobian = [
        [s[0][0], s[0][1]],
        [s[2][0], s[2][1]],
    ];

    (residual, jacobian)
}

fn norm2(v: &Vector2) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn solve_2_by_2(a: &Matrix2, b: &Vector2) -> Option<Vector2> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];

    if det.abs() < 1.0e-14 {
        return None;
    }

    Some([
        (b[0] * a[1][1] - a[0][1] * b[1]) / det,
        (a[0][0] * b[1] - b[0] * a[1][0]) / det,
    ])
}

enum JacobianMode {
    FiniteDifference,
    Variational,
}

struct NewtonResult {
    v: Vector2,
    residual: Vector2,
    iterations: usize,
    converged: bool,
    ivp_solves: usize,
}

fn newton_shooting(
    initial_guess: Vector2,
    steps: usize,
    tolerance: f64,
    max_iterations: usize,
    mode: JacobianMode,
) -> NewtonResult {
    let mut v = initial_guess;
    let mut ivp_solves = 0usize;

    println!("{:>6} {:>15} {:>15} {:>15} {:>15} {:>10}",
        "iter", "V_1", "V_2", "||F(V)||", "alpha", "solves"
    );

    for iteration in 0..=max_iterations {
        let (residual, jacobian, solves_this_step) = match mode {
            JacobianMode::FiniteDifference => {
                let residual = shooting_residual(&v, steps);
                let jacobian = finite_difference_jacobian(&v, steps, 1.0e-6);
                (residual, jacobian, Q + 1)
            }
            JacobianMode::Variational => {
                let (residual, jacobian) = variational_residual_and_jacobian(&v, steps);
                (residual, jacobian, 1)
            }
        };

        ivp_solves += solves_this_step;

        let residual_norm = norm2(&residual);

        println!(
            "{:>6} {:>15.8} {:>15.8} {:>15.6e} {:>15} {:>10}",
            iteration,
            v[0],
            v[1],
            residual_norm,
            "-",
            ivp_solves
        );

        if residual_norm < tolerance {
            return NewtonResult {
                v,
                residual,
                iterations: iteration,
                converged: true,
                ivp_solves,
            };
        }

        let right_hand_side = [-residual[0], -residual[1]];

        let Some(delta) = solve_2_by_2(&jacobian, &right_hand_side) else {
            return NewtonResult {
                v,
                residual,
                iterations: iteration,
                converged: false,
                ivp_solves,
            };
        };

        let mut alpha = 1.0;
        let mut accepted = false;

        for _ in 0..12 {
            let trial_v = [
                v[0] + alpha * delta[0],
                v[1] + alpha * delta[1],
            ];

            let trial_residual = shooting_residual(&trial_v, steps);
            ivp_solves += 1;

            if norm2(&trial_residual) < residual_norm {
                v = trial_v;
                accepted = true;

                println!(
                    "{:>6} {:>15} {:>15} {:>15} {:>15.6} {:>10}",
                    "", "", "", "", alpha, ivp_solves
                );

                break;
            }

            alpha *= 0.5;
        }

        if !accepted {
            return NewtonResult {
                v,
                residual,
                iterations: iteration,
                converged: false,
                ivp_solves,
            };
        }
    }

    let final_residual = shooting_residual(&v, steps);
    ivp_solves += 1;

    NewtonResult {
        v,
        residual: final_residual,
        iterations: max_iterations,
        converged: false,
        ivp_solves,
    }
}

fn print_result(label: &str, result: &NewtonResult) {
    let exact = [PI, 2.0 * PI];

    println!();
    println!("{}", label);
    println!("{}", "-".repeat(label.len()));
    println!("Converged                         = {}", result.converged);
    println!("Iterations performed              = {}", result.iterations);
    println!("IVP integrations counted           = {}", result.ivp_solves);
    println!("Computed V_1 = y'(0)               = {:.12}", result.v[0]);
    println!("Computed V_2 = z'(0)               = {:.12}", result.v[1]);
    println!("Exact V_1 = pi                     = {:.12}", exact[0]);
    println!("Exact V_2 = 2pi                    = {:.12}", exact[1]);
    println!(
        "Absolute error in V_1              = {:.12e}",
        (result.v[0] - exact[0]).abs()
    );
    println!(
        "Absolute error in V_2              = {:.12e}",
        (result.v[1] - exact[1]).abs()
    );
    println!("Final residual F_1(V)              = {:.12e}", result.residual[0]);
    println!("Final residual F_2(V)              = {:.12e}", result.residual[1]);
    println!("Final residual norm                = {:.12e}", norm2(&result.residual));
}

fn main() {
    let steps = 100;
    let tolerance = 1.0e-10;
    let max_iterations = 12;

    let initial_guess = [2.5, 5.5];

    println!("Newton Shooting for a Vector Two-Point Boundary Value Problem");
    println!("=============================================================");
    println!();
    println!("Model problem:");
    println!("  y''(x) = -pi^2 sin(pi x)");
    println!("  z''(x) = -4pi^2 sin(2pi x)");
    println!("  0 <= x <= 1");
    println!();
    println!("Boundary conditions:");
    println!("  y(0) = 0,   z(0) = 0");
    println!("  y(1) = 0,   z(1) = 0");
    println!();
    println!("Unknown shooting vector:");
    println!("  V = [y'(0), z'(0)]^T");
    println!();

    println!("Newton Iteration with Finite-Difference Jacobian");
    println!("------------------------------------------------");
    let fd_result = newton_shooting(
        initial_guess,
        steps,
        tolerance,
        max_iterations,
        JacobianMode::FiniteDifference,
    );

    print_result("Finite-Difference Jacobian Result", &fd_result);

    println!();
    println!("Newton Iteration with Variational Jacobian");
    println!("------------------------------------------");
    let var_result = newton_shooting(
        initial_guess,
        steps,
        tolerance,
        max_iterations,
        JacobianMode::Variational,
    );

    print_result("Variational Jacobian Result", &var_result);
}
```

Program 18.2.2 demonstrates the implementation of Newton-based shooting methods for vector boundary value problems and provides a direct comparison between finite-difference and sensitivity-based Jacobian constructions. This reflects the central computational framework described in Sections 18.2.3 and 18.2.4, where the efficiency and accuracy of the method depend critically on how the Jacobian is computed.

The results illustrate two key aspects of the method. First, Newton iteration provides rapid convergence when the Jacobian accurately captures the sensitivity of the residual with respect to the shooting parameters. Second, the variational approach significantly reduces computational cost by replacing multiple independent integrations with a single augmented integration, thereby improving both efficiency and numerical stability.

The modular design of the implementation allows for straightforward extension to higher-dimensional systems and more complex boundary conditions. It also provides a foundation for incorporating advanced techniques, such as sparse Jacobian representations, matrix-free Newton–Krylov methods, and adaptive integration strategies, which are essential for large-scale applications.

Overall, the program reinforces the importance of sensitivity analysis in numerical boundary value methods and demonstrates how the theoretical framework of variational equations can be translated into practical computational algorithms.

## 18.2.5. Conditioning and Instability

The principal difficulty of the shooting method arises from the potential instability of the underlying differential system. While the formulation reduces the TPBVP to a finite-dimensional nonlinear problem, the mapping from initial conditions to terminal values can be highly sensitive, and this sensitivity directly affects the conditioning of the residual function.

To analyze this behavior, consider a perturbation $\boldsymbol\eta(x)$ of the solution. The evolution of this perturbation is governed by the linearized system discussed earlier, and its propagation from $x=a$ to $x=b$ can be expressed in terms of the *fundamental matrix* $\boldsymbol\Phi(x,a)$:

$$\boldsymbol\eta(b) = \boldsymbol\Phi(b,a)\,\boldsymbol\eta(a) \tag{18.2.14}$$

The fundamental matrix satisfies the matrix differential equation:

$$\boldsymbol\Phi'(x,a) = \mathbf g_{\mathbf y}(x,\mathbf y^\star(x))\,\boldsymbol\Phi(x,a),\qquad\boldsymbol\Phi(a,a) = \mathbf I \tag{18.2.15}$$

This matrix describes how infinitesimal perturbations in the initial condition are transported along the solution trajectory. Its norm therefore provides a quantitative measure of the sensitivity of the terminal state with respect to the initial data.

If the system exhibits unstable modes, the fundamental matrix may grow rapidly as $x$ increases. In such cases, the magnitude $|\boldsymbol\Phi(b,a)|$ becomes large, meaning that even very small perturbations in $\boldsymbol\eta(a)$ can lead to large deviations in $\boldsymbol\eta(b)$. This amplification mechanism has direct consequences for the shooting method.

First, the residual function $\mathbf F(\mathbf V)$, which depends on the terminal value $\mathbf y(b;\mathbf V)$, becomes highly sensitive to changes in the parameter vector $\mathbf V$. As a result, the Jacobian $\mathbf J(\mathbf V)$ may be poorly conditioned, with large variations in its entries. Second, this ill-conditioning leads to numerical difficulties in solving the Newton system, as small errors in the parameter update can produce disproportionately large changes in the residual. Third, the nonlinear mapping itself may exhibit steep gradients, causing Newton iterations to overshoot, stagnate, or behave erratically unless carefully damped.

From a geometric perspective, this instability reflects the divergence of nearby trajectories in the phase space. The shooting method attempts to select a trajectory that satisfies a condition at $x=b$, but when trajectories diverge rapidly, small inaccuracies in the initial slope or parameter vector result in large deviations at the endpoint. This makes the problem of “hitting” the target boundary condition increasingly difficult.

Thus, the conditioning of the shooting formulation is intrinsically linked to the stability properties of the differential equation. When the system is stable or only mildly unstable, the method performs efficiently and accurately. However, in the presence of strong instability, the amplification encoded in the fundamental matrix can render the problem ill-conditioned, necessitating more sophisticated strategies or alternative formulations to achieve reliable convergence.

## 18.2.6. Modern Improvements and Practical Performance Considerations

The classical shooting method, while conceptually simple and effective in low-dimensional settings, can suffer from sensitivity and instability, as discussed in the preceding section. Modern developments therefore aim to enhance its robustness and reliability while preserving its fundamental structure of reducing a TPBVP to an initial value problem coupled with a finite-dimensional nonlinear system.

One important line of improvement focuses on *enhanced differential-correction strategies*. In these approaches, the sensitivity information required for Newton updates is constructed more robustly than in standard formulations. Rather than relying solely on direct Jacobian evaluations or finite differences, higher-order correction schemes employ derivative-free least-squares approximations to capture the relationship between parameter variations and boundary residuals. This leads to improved numerical stability and reduces the impact of local ill-conditioning in the Jacobian. As a result, convergence behavior is often more reliable, particularly in problems where the residual function exhibits strong nonlinearity or noise sensitivity (Sharan et al., 2025).

A second direction addresses one of the practical bottlenecks of shooting methods: the selection of an appropriate initial guess for the parameter vector $\mathbf V$. Since the nonlinear system $\mathbf F(\mathbf V)=\mathbf 0$ may possess multiple solutions or exhibit strong sensitivity, the success of Newton-type iterations depends heavily on the quality of the initial estimate. Recent approaches incorporate predictive models to generate informed initial guesses for the missing initial conditions. In this framework, a trainable model produces a preliminary estimate of $\mathbf V$, which is then refined through classical shooting iterations. This hybrid strategy retains the accuracy of deterministic integration while improving convergence in challenging or highly nonlinear problems (Luong et al., 2026).

These developments share a common objective: to mitigate the sensitivity inherent in the shooting formulation without abandoning its computational advantages. By improving either the approximation of sensitivities or the initialization of parameters, they enhance the stability and efficiency of the method in practical applications.

From a broader computational perspective, the performance of the shooting method depends on several interrelated factors. The dimensionality $q$ of the parameter space plays a central role, as both the cost of Jacobian evaluation and the complexity of the nonlinear solve scale with $q$. When $q$ is small, the method is particularly attractive, since the associated Newton system is inexpensive and the number of required integrations remains modest. Conversely, as (q) increases, both computational cost and sensitivity effects become more pronounced.

The stability properties of the underlying differential system are equally important. As shown in Section 18.2.5, unstable modes lead to amplification of perturbations through the fundamental matrix, which in turn degrades the conditioning of the residual function. In such cases, even improved Jacobian approximations or better initial guesses may not fully eliminate convergence difficulties, although they can significantly alleviate them.

In practical terms, the shooting method is most effective under the following conditions: the number of unknown initial values is small, the associated initial value problem is not strongly unstable, and a reliable and accurate ODE integrator is available. Under these circumstances, the method offers several notable advantages. It is straightforward to implement, requires relatively little memory, and allows direct reuse of high-quality initial value solvers. These properties make it an attractive first choice for many boundary value problems.

At the same time, its primary limitation remains its sensitivity to instability and the resulting ill-conditioning of the nonlinear system. When these effects become significant, convergence may deteriorate, and alternative formulations, such as relaxation or collocation methods, may be preferable.

In summary, the shooting method occupies an important place among numerical techniques for TPBVPs. It combines conceptual clarity with computational efficiency in favorable settings, and modern enhancements further extend its applicability. In practice, it is often employed as an initial approach, with more robust variants or alternative methods introduced when the problem exhibits strong sensitivity or instability.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/1QHRt17mTJHRCe0DO2Z5.4","tags":[]}

# 18.3. Shooting to a Fitting Point

The difficulties encountered in standard shooting methods, particularly those arising from instability and sensitivity, motivate the development of modified formulations that reduce the impact of long-range error propagation. Shooting to a fitting point, often referred to as *double shooting*, is one such approach. It preserves the conceptual framework of shooting while introducing a decomposition of the interval that improves numerical conditioning and stability.

## 18.3.1. Motivation and Basic Idea

When the standard shooting method becomes ill-conditioned, the principal source of difficulty is the need to propagate the solution across the entire interval $[a,b]$ in a single integration. As discussed earlier, if the differential system contains unstable modes, small perturbations in the initial data may grow exponentially as the solution advances. This amplification leads to large deviations at the terminal boundary, making the residual function highly sensitive and the associated nonlinear system poorly conditioned.

The core idea of shooting to a fitting point is to limit this error amplification by reducing the distance over which any single integration is performed. To achieve this, an interior point $x_f \in (a,b)$ is introduced, and the interval is divided into two subintervals: $[a,x_f]$ and $[x_f,b]$. Instead of propagating the solution from one endpoint to the other, the method integrates from both ends toward the fitting point.

More precisely, one constructs two initial value problems. The first is integrated forward from $x=a$, using parameters that satisfy the left boundary conditions. The second is integrated backward from $x=b$, using parameters that satisfy the right boundary conditions. Each of these integrations produces a solution defined on its respective subinterval, and both solutions are evaluated at the fitting point $x_f$.

The key requirement is that these two solutions must agree at $x_f$. This matching condition enforces continuity of the state (and, depending on the formulation, possibly its derivatives) at the fitting point. The discrepancy between the forward and backward solutions at $x_f$ defines a residual that must be driven to zero.

By splitting the interval in this manner, the method significantly reduces the effect of unstable growth. Errors introduced in the forward integration are only amplified over the shorter interval $[a,x_f]$, and similarly for the backward integration over $[x_f,b]$. This leads to a better-conditioned problem, as the sensitivity of the matching condition is typically much lower than that of the full-interval boundary mismatch.

Thus, double shooting retains the advantages of the shooting framework, including the use of initial value solvers and a finite-dimensional nonlinear system, while mitigating its principal weakness by introducing an intermediate consistency condition.

## 18.3.2. Formulation of the Matching Problem

The introduction of a fitting point $x_f \in (a,b)$ leads to a reformulation of the boundary value problem as a *matching problem* between two independently integrated solutions. This formulation retains the essential structure of the shooting method while distributing the unknown degrees of freedom between the two ends of the interval.

Let the boundary conditions at $x=a$ determine $n_1$ components of the state, leaving $n_2 = N - n_1$ components free. Similarly, the boundary conditions at $x=b$ determine $n_2$ components, leaving $n_1$ components free. Accordingly, introduce parameter vectors:

$$\mathbf V^{(L)} \in \mathbb{R}^{n_2}, \qquad \mathbf V^{(R)} \in \mathbb{R}^{n_1} \tag{18.3.1}$$

which represent the degrees of freedom in the initial conditions at the left and right endpoints, respectively. These parameters are used to define *loading maps* that enforce the boundary conditions at each endpoint:

$$\mathbf y(a) = \boldsymbol\ell_L(\mathbf V^{(L)}), \qquad\mathbf y(b) = \boldsymbol\ell_R(\mathbf V^{(R)}) \tag{18.3.2}$$

By construction, these maps ensure that the left and right boundary conditions are satisfied for all choices of $\mathbf V^{(L)}$ and $\mathbf V^{(R)}$.

With these parameterizations in place, two initial value problems are solved. The first is integrated forward from $x=a$ to the fitting point $x=x_f$, producing a solution:

$$\mathbf y^{(L)}(x;\mathbf V^{(L)})$$

The second is integrated backward from $x=b$ to $x=x_f$, producing a solution:

$$\mathbf y^{(R)}(x;\mathbf V^{(R)})$$

The central requirement of the method is that these two solutions must coincide at the fitting point. This leads to the matching condition:

\begin{equation}
\mathbf{M}\big(\mathbf{V}^{(L)},\mathbf{V}^{(R)}\big)
=
\mathbf{y}^{(L)}(x_f;\mathbf{V}^{(L)}) 
- \mathbf{y}^{(R)}(x_f;\mathbf{V}^{(R)})
= \mathbf{0}
\tag{18.3.3}
\end{equation}

This equation defines a nonlinear system in the combined parameter vector $(\mathbf V^{(L)}, \mathbf V^{(R)})$. Its dimension is $N$, corresponding to the number of components of the state vector. Since the total number of unknown parameters is $n_2 + n_1 = N$, the system is square and, under appropriate conditions, admits a unique solution.

An important structural aspect of this formulation is that the original differential equation has already been reduced to first order. Consequently, the state vector $\mathbf y(x)$ contains not only the primary variables but also all necessary derivative information. Therefore, enforcing equality of the full state at $x_f$ ensures continuity of the solution and all its derivatives represented in the system. No additional matching conditions are required.

From a numerical standpoint, the matching formulation has a significant advantage over single shooting. Each integration is performed over a shorter interval, reducing the amplification of perturbations associated with unstable modes. The residual $\mathbf M$ is therefore typically better conditioned than the terminal residual in standard shooting. At the same time, the problem remains finite-dimensional and can be solved using Newton-type methods, with Jacobians constructed from sensitivity information associated with both forward and backward integrations.

Thus, the formulation (18.3.1)–(18.3.3) provides a balanced compromise between the simplicity of shooting and the need for improved numerical stability, forming the basis for the double shooting method developed in subsequent sections.

## 18.3.3. Newton Iteration and Jacobian Structure

The matching condition (18.3.3) defines a nonlinear system in the combined parameter vector $(\mathbf V^{(L)}, \mathbf V^{(R)})$. As in the standard shooting formulation, this system is solved using Newton’s method, now applied to the matching residual $\mathbf M$. The Newton update takes the form:

\begin{equation}
\mathbf{J}_M
\begin{bmatrix}
\Delta \mathbf{V}^{(L)} \\[4pt]
\Delta \mathbf{V}^{(R)}
\end{bmatrix}
=
-\mathbf{M}
\tag{18.3.4}
\end{equation}

where $\mathbf J_M$ denotes the Jacobian of the matching function with respect to the combined parameter vector. The updated parameters are then obtained by adding these corrections, typically with an appropriate damping strategy to ensure robustness.

The structure of the Jacobian $\mathbf J_M$ reflects the dependence of the matching residual on both sets of parameters. Specifically, the residual:

$$\mathbf M = \mathbf y^{(L)}(x_f;\mathbf V^{(L)}) - \mathbf y^{(R)}(x_f;\mathbf V^{(R)})$$

depends on $\mathbf V^{(L)}$ through the forward integration and on $\mathbf V^{(R)}$ through the backward integration. Consequently, the Jacobian naturally decomposes into contributions from these two sensitivities. When variational equations are employed, these sensitivities are obtained directly from the corresponding sensitivity matrices evaluated at the fitting point. This allows the Jacobian to be assembled analytically without requiring repeated integrations, leading to improved efficiency and accuracy.

Alternatively, if sensitivity equations are not used, the Jacobian may be approximated by finite differences. In this case, each component of $\mathbf V^{(L)}$ and $\mathbf V^{(R)}$ is perturbed in turn, and the resulting change in the matching residual is computed by re-integrating the corresponding initial value problem. While straightforward, this approach increases the computational cost due to the additional integrations required for each Newton step.

A useful way to interpret the overall structure of the method is through the mapping:

\begin{equation}
\underbrace{
\begin{bmatrix}
\mathbf{V}^{(L)} \\[2pt]
\mathbf{V}^{(R)}
\end{bmatrix}
}_{\text{unknowns}}
\;\xrightarrow{\text{two IVP solves}}\;
\underbrace{
\begin{bmatrix}
\mathbf{y}^{(L)}(x_f) \\[2pt]
\mathbf{y}^{(R)}(x_f)
\end{bmatrix}
}_{\text{evaluations}}
\;\xrightarrow{\text{matching}}\;
\underbrace{\mathbf{M}}_{\text{residual}}
= \mathbf{0}
\tag{18.3.5}
\end{equation}

This representation makes explicit the sequence of operations: the unknown parameters define initial conditions at both endpoints, these conditions are propagated via two initial value problem solves toward the fitting point, and the discrepancy between the resulting states defines the residual. The Newton iteration then seeks to eliminate this discrepancy by adjusting the parameters.

The principal advantage of this formulation is that it replaces a single long integration, which may be strongly affected by instability, with two shorter integrations over subintervals $[a,x_f]$ and $[x_f,b]$. Since perturbations are only propagated over reduced distances, the amplification associated with unstable modes is significantly diminished. As a result, the Jacobian $\mathbf J_M$ is typically better conditioned than the corresponding Jacobian in single shooting, leading to more stable and reliable Newton iterations.

Thus, the double shooting formulation combines the conceptual simplicity of shooting with improved numerical conditioning, achieved through the introduction of a fitting point and the associated decomposition of the interval.

### Rust Implementation

Following the development of shooting to a fitting point in Sections 18.3.1–18.3.3, Program 18.3.1 provides a practical implementation of the double shooting method for a two-point boundary value problem. In the theoretical formulation, the interval is decomposed into two subintervals and independent initial value problems are solved from each endpoint toward an interior fitting point, as described by equations (18.3.1)–(18.3.3). The solution is then obtained by enforcing a matching condition at this interior point, thereby transforming the boundary value problem into a finite-dimensional nonlinear system. From a computational perspective, this approach mitigates the instability associated with long-range propagation by restricting integrations to shorter intervals. The present program implements this strategy using a Runge–Kutta integrator combined with Newton iteration applied to the matching residual, illustrating how the fitting-point formulation improves conditioning while retaining the conceptual simplicity of shooting methods.

At the core of the implementation is the function `matching_residual`, which defines the vector-valued residual $\mathbf M(\mathbf V^{(L)}, \mathbf V^{(R)})$ introduced in equation (18.3.3). This function evaluates the difference between the forward solution $\mathbf y^{(L)}(x_f;\mathbf V^{(L)})$ and the backward solution $\mathbf y^{(R)}(x_f;\mathbf V^{(R)})$ at the fitting point. By returning both the state and derivative mismatch, it enforces equality of the full first-order system, ensuring continuity of all components of the solution at $x_f$.

The forward and backward integrations are performed by the function `integrate`, which constructs the initial value problems corresponding to the loading maps described in equation (18.3.2). For the left integration, the initial condition satisfies the boundary condition at $x=a$, while the right integration begins at $x=b$ and proceeds backward toward the fitting point. The classical fourth-order Runge–Kutta method, implemented in `rk4_step`, is used to advance the solution in both directions, providing a stable and accurate discretization of the differential equation.

The nonlinear system defined by the matching condition is solved using Newton’s method, implemented in the function `newton_double_shooting`. This function follows the update structure given in equation (18.3.4), where the Jacobian of the matching residual is approximated using finite differences. The function `finite_difference_jacobian` computes this approximation by perturbing each component of the parameter vector and re-evaluating the matching residual. Although this approach requires multiple integrations per iteration, it provides a straightforward and general mechanism for constructing the Jacobian.

The `main` function demonstrates the complete double shooting procedure. Starting from an initial guess for the parameters $\mathbf V^{(L)}$ and $\mathbf V^{(R)}$, it iteratively refines these values using Newton updates until the matching residual is driven to zero. The final results include the computed endpoint slopes, the matching residual at the fitting point, and a comparison between the forward and backward solutions. This verification step illustrates that the two independently integrated trajectories coincide at $x_f$, thereby satisfying the matching condition and confirming the correctness of the solution.

```rust
// Program 18.3.1: Double Shooting to a Fitting Point
//
// Problem statement:
// This program solves a scalar two-point boundary value problem using shooting
// to a fitting point. The model problem is
//
//     y''(x) = -pi^2 sin(pi x),   0 <= x <= 1,
//     y(0) = 0,                   y(1) = 0.
//
// The exact solution is
//
//     y(x) = sin(pi x).
//
// The interval is split at x_f = 0.5. One solution is integrated forward from
// x = 0, while another is integrated backward from x = 1. The unknowns are
//
//     V_L = y'(0),     V_R = y'(1).
//
// Newton's method is applied to the matching residual
//
//     M(V_L, V_R) = y_L(x_f; V_L) - y_R(x_f; V_R),
//
// where the full first-order state is matched at the fitting point.

use std::f64::consts::PI;

#[derive(Clone, Copy)]
struct State {
    y1: f64, // y
    y2: f64, // y'
}

type Vector2 = [f64; 2];
type Matrix2 = [[f64; 2]; 2];

fn forcing(x: f64) -> f64 {
    -PI * PI * (PI * x).sin()
}

fn exact_solution(x: f64) -> f64 {
    (PI * x).sin()
}

fn exact_derivative(x: f64) -> f64 {
    PI * (PI * x).cos()
}

fn rhs(x: f64, state: State) -> State {
    State {
        y1: state.y2,
        y2: forcing(x),
    }
}

fn add_scaled(a: State, b: State, scale: f64) -> State {
    State {
        y1: a.y1 + scale * b.y1,
        y2: a.y2 + scale * b.y2,
    }
}

fn rk4_step(x: f64, state: State, h: f64) -> State {
    let k1 = rhs(x, state);

    let s2 = add_scaled(state, k1, 0.5 * h);
    let k2 = rhs(x + 0.5 * h, s2);

    let s3 = add_scaled(state, k2, 0.5 * h);
    let k3 = rhs(x + 0.5 * h, s3);

    let s4 = add_scaled(state, k3, h);
    let k4 = rhs(x + h, s4);

    State {
        y1: state.y1 + h * (k1.y1 + 2.0 * k2.y1 + 2.0 * k3.y1 + k4.y1) / 6.0,
        y2: state.y2 + h * (k1.y2 + 2.0 * k2.y2 + 2.0 * k3.y2 + k4.y2) / 6.0,
    }
}

fn integrate(x0: f64, xf: f64, initial_state: State, steps: usize) -> State {
    let h = (xf - x0) / steps as f64;

    let mut x = x0;
    let mut state = initial_state;

    for _ in 0..steps {
        state = rk4_step(x, state, h);
        x += h;
    }

    state
}

fn matching_residual(v: Vector2, steps_per_half: usize) -> Vector2 {
    let a = 0.0;
    let b = 1.0;
    let xf = 0.5;

    let v_left = v[0];
    let v_right = v[1];

    let left_initial = State {
        y1: 0.0,
        y2: v_left,
    };

    let right_initial = State {
        y1: 0.0,
        y2: v_right,
    };

    let left_at_fit = integrate(a, xf, left_initial, steps_per_half);
    let right_at_fit = integrate(b, xf, right_initial, steps_per_half);

    [
        left_at_fit.y1 - right_at_fit.y1,
        left_at_fit.y2 - right_at_fit.y2,
    ]
}

fn finite_difference_jacobian(v: Vector2, steps_per_half: usize, eps: f64) -> Matrix2 {
    let base = matching_residual(v, steps_per_half);
    let mut jacobian = [[0.0; 2]; 2];

    for col in 0..2 {
        let mut perturbed = v;
        perturbed[col] += eps;

        let perturbed_residual = matching_residual(perturbed, steps_per_half);

        for row in 0..2 {
            jacobian[row][col] = (perturbed_residual[row] - base[row]) / eps;
        }
    }

    jacobian
}

fn solve_2_by_2(a: Matrix2, b: Vector2) -> Option<Vector2> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];

    if det.abs() < 1.0e-14 {
        return None;
    }

    Some([
        (b[0] * a[1][1] - a[0][1] * b[1]) / det,
        (a[0][0] * b[1] - b[0] * a[1][0]) / det,
    ])
}

fn norm2(v: Vector2) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

struct NewtonResult {
    v: Vector2,
    residual: Vector2,
    iterations: usize,
    converged: bool,
}

fn newton_double_shooting(
    initial_guess: Vector2,
    steps_per_half: usize,
    tolerance: f64,
    max_iterations: usize,
) -> NewtonResult {
    let mut v = initial_guess;

    println!("Newton Iteration History");
    println!("------------------------");
    println!(
        "{:>6} {:>16} {:>16} {:>16}",
        "iter", "V_L", "V_R", "||M||"
    );

    for iteration in 0..=max_iterations {
        let residual = matching_residual(v, steps_per_half);
        let residual_norm = norm2(residual);

        println!(
            "{:>6} {:>16.10} {:>16.10} {:>16.8e}",
            iteration, v[0], v[1], residual_norm
        );

        if residual_norm < tolerance {
            return NewtonResult {
                v,
                residual,
                iterations: iteration,
                converged: true,
            };
        }

        let jacobian = finite_difference_jacobian(v, steps_per_half, 1.0e-6);
        let rhs = [-residual[0], -residual[1]];

        let Some(delta) = solve_2_by_2(jacobian, rhs) else {
            return NewtonResult {
                v,
                residual,
                iterations: iteration,
                converged: false,
            };
        };

        let mut alpha = 1.0;
        let mut accepted = false;

        for _ in 0..12 {
            let trial_v = [
                v[0] + alpha * delta[0],
                v[1] + alpha * delta[1],
            ];

            let trial_residual = matching_residual(trial_v, steps_per_half);

            if norm2(trial_residual) < residual_norm {
                v = trial_v;
                accepted = true;
                break;
            }

            alpha *= 0.5;
        }

        if !accepted {
            return NewtonResult {
                v,
                residual,
                iterations: iteration,
                converged: false,
            };
        }
    }

    let residual = matching_residual(v, steps_per_half);

    NewtonResult {
        v,
        residual,
        iterations: max_iterations,
        converged: false,
    }
}

fn main() {
    let steps_per_half = 60;
    let tolerance = 1.0e-10;
    let max_iterations = 12;

    let initial_guess = [2.5, -2.5];

    let result = newton_double_shooting(
        initial_guess,
        steps_per_half,
        tolerance,
        max_iterations,
    );

    let xf = 0.5;

    let left_at_fit = integrate(
        0.0,
        xf,
        State {
            y1: 0.0,
            y2: result.v[0],
        },
        steps_per_half,
    );

    let right_at_fit = integrate(
        1.0,
        xf,
        State {
            y1: 0.0,
            y2: result.v[1],
        },
        steps_per_half,
    );

    println!();
    println!("Double Shooting Result");
    println!("----------------------");
    println!("Converged                         = {}", result.converged);
    println!("Iterations performed              = {}", result.iterations);
    println!("Computed V_L = y'(0)              = {:.12}", result.v[0]);
    println!("Computed V_R = y'(1)              = {:.12}", result.v[1]);
    println!("Exact y'(0)                       = {:.12}", exact_derivative(0.0));
    println!("Exact y'(1)                       = {:.12}", exact_derivative(1.0));
    println!(
        "Absolute error in V_L             = {:.12e}",
        (result.v[0] - exact_derivative(0.0)).abs()
    );
    println!(
        "Absolute error in V_R             = {:.12e}",
        (result.v[1] - exact_derivative(1.0)).abs()
    );
    println!("Matching residual M_1             = {:.12e}", result.residual[0]);
    println!("Matching residual M_2             = {:.12e}", result.residual[1]);
    println!("Matching residual norm            = {:.12e}", norm2(result.residual));

    println!();
    println!("State Matching at the Fitting Point");
    println!("-----------------------------------");
    println!("Fitting point x_f                  = {:.6}", xf);
    println!("Left  y(x_f)                       = {:.12}", left_at_fit.y1);
    println!("Right y(x_f)                       = {:.12}", right_at_fit.y1);
    println!("Exact y(x_f)                       = {:.12}", exact_solution(xf));
    println!("Left  y'(x_f)                      = {:.12}", left_at_fit.y2);
    println!("Right y'(x_f)                      = {:.12}", right_at_fit.y2);
    println!("Exact y'(x_f)                      = {:.12}", exact_derivative(xf));
}
```

Program 18.3.1 demonstrates the practical implementation of the double shooting method and illustrates how the introduction of a fitting point improves the numerical behavior of shooting-based approaches. By replacing a single long integration with two shorter integrations, the method reduces the amplification of perturbations and leads to a better-conditioned nonlinear system.

The results highlight the effectiveness of the matching formulation. The residual at the fitting point is driven to machine precision, and the forward and backward solutions coincide with high accuracy. This confirms that the method successfully enforces continuity of the solution across the interval while maintaining stability.

The structure of the implementation reflects the mapping described in equation (18.3.5), where the unknown parameters define initial conditions, the system is propagated via two initial value solves, and the resulting mismatch defines the residual. This modular structure makes the method readily extensible to more complex systems and higher-dimensional problems.

Overall, the program provides a clear computational realization of the theoretical framework developed in Section 18.3 and serves as a foundation for further extensions, including multiple shooting methods and sensitivity-based Jacobian constructions.

## 18.3.4. Extension to Multiple Shooting

The concept of introducing an interior fitting point can be extended systematically to a larger number of subintervals, leading to the *multiple shooting* method. This generalization further improves stability and conditioning by limiting the propagation of errors to short segments of the domain, while preserving the flexibility and structure of shooting-based approaches.

To construct the method, partition the interval $[a,b]$ into $m$ subintervals:

$$a = x_0 < x_1 < \cdots < x_m = b \tag{18.3.6}$$

At each node $x_k$, introduce an unknown vector:

$$\mathbf s_k \approx \mathbf y(x_k), \qquad k = 0,\dots,m \tag{18.3.7}$$

These vectors represent approximations to the solution at the mesh points and serve as the primary unknowns of the method.

For each subinterval $[x_k, x_{k+1}]$, the differential equation is solved as an initial value problem starting from $\mathbf s_k$. Denote by $\boldsymbol\varphi_k(\mathbf s_k)$ the resulting state at $x_{k+1}$, obtained by integrating the system forward from $x_k$ to $x_{k+1}$. The consistency requirement is that this propagated value must agree with the next unknown state $\mathbf s_{k+1}$. This leads to the propagation residual:

\begin{equation}
\mathbf{R}_k(\mathbf{s}_k,\mathbf{s}_{k+1})
=
\boldsymbol{\varphi}_k(\mathbf{s}_k) - \mathbf{s}_{k+1}
= \mathbf{0}, 
\qquad k = 0,\dots,m-1
\tag{18.3.8}
\end{equation}

These equations enforce continuity of the solution across subinterval boundaries. In addition, the boundary conditions at $x=a$ and $x=b$ are imposed on $\mathbf s_0$ and $\mathbf s_m$, respectively. Taken together, the propagation residuals and boundary conditions define a nonlinear system in the variables $\{\mathbf s_k\}_{k=0}^m$.

A key structural feature of this system is its *sparse block form*. Each equation $\mathbf R_k$ couples only two consecutive unknowns, $\mathbf s_k$ and $\mathbf s_{k+1}$, resulting in a block-banded Jacobian matrix. If each state vector has dimension $N$, then each block has size $N \times N$, and the overall system has dimension $(m+1)N$.

When solved using direct methods with dense block elimination, the computational cost is approximately,

$$T_{\text{multi-shoot}} = O(m N^3) \tag{18.3.9}$$

This reflects the cost of solving a sequence of coupled linear systems involving (N \\times N) blocks. Although this scaling may appear substantial, the block structure can often be exploited to improve efficiency, particularly when specialized linear algebra techniques are employed.

The principal advantage of multiple shooting lies in its improved numerical conditioning. By restricting each integration to a short subinterval, the growth of perturbations associated with unstable modes is significantly reduced. As a result, the sensitivity of the solution with respect to the unknown states is more localized, and the overall nonlinear system is typically better conditioned than in single or double shooting.

An additional and practically important advantage is the potential for *parallelism*. Since the integrations over different subintervals depend only on the initial states $\mathbf s_k$, they can often be performed independently once the current iterate is specified. This enables parallel execution of the IVP solves, which can lead to substantial gains in computational efficiency, especially for large-scale problems or when high-resolution discretizations are required.

Thus, multiple shooting represents a natural and powerful extension of the shooting framework. It combines improved stability, structured sparsity, and opportunities for parallel computation, making it particularly well suited for complex or high-dimensional boundary value problems where standard shooting methods may struggle.

### Rust Implementation

Following the extension of the shooting framework to multiple subintervals in Section 18.3.4, Program 18.3.2 provides a practical implementation of the multiple shooting method for a two-point boundary value problem. In this formulation, the interval is partitioned as in equation (18.3.6), and the solution is represented by a sequence of unknown state vectors at the mesh points, as introduced in equation (18.3.7). The differential equation is then enforced locally on each subinterval through propagation, while continuity across subinterval boundaries is imposed through the residual equations defined in (18.3.8). This transforms the original boundary value problem into a structured nonlinear system whose Jacobian exhibits a block-banded form. The present program realizes this formulation using a Runge–Kutta integrator and Newton iteration, demonstrating how multiple shooting improves stability and conditioning by limiting error propagation to short segments of the domain while preserving the flexibility of shooting-based methods.

At the core of the implementation is the function `multiple_shooting_residual`, which constructs the nonlinear residual system corresponding to equation (18.3.8). For each subinterval, the function integrates the initial value problem starting from the state $\mathbf s_k$ and compares the propagated result $\boldsymbol\varphi_k(\mathbf s_k)$ with the next unknown state $\mathbf s_{k+1}$. The difference between these two quantities defines the propagation residual, ensuring continuity of the solution across subinterval boundaries. In addition, the boundary conditions are imposed directly on the first and last states, completing the nonlinear system.

The numerical integration on each subinterval is performed by the function `integrate_subinterval`, which uses the classical fourth-order Runge–Kutta method implemented in `rk4_step`. This function advances the solution from $x_k$ to $x_{k+1}$, thereby realizing the mapping $\boldsymbol\varphi_k(\mathbf s_k)$ described in equation (18.3.8). Since each subinterval is treated independently, this structure naturally reflects the localized nature of the multiple shooting formulation.

The unknown states ${\mathbf s_k}$ are stored in a flattened vector and converted to structured form using the functions `pack_states` and `unpack_states`. This representation allows the nonlinear system to be expressed in a standard vector form while preserving the interpretation of each component as a state at a mesh point. The resulting system has dimension $(m+1)N$, consistent with the discussion following equation (18.3.8).

The nonlinear system is solved using Newton’s method, implemented in the function `newton_multiple_shooting`. At each iteration, the residual is evaluated and a Jacobian matrix is constructed using finite differences. Although this approach does not explicitly exploit the block structure described in equation (18.3.9), it provides a general and transparent implementation suitable for moderate problem sizes. The resulting linear system is solved using a dense Gaussian elimination routine, reflecting the $O(mN^3)$ computational cost discussed in equation (18.3.9).

The `main` function demonstrates the complete multiple shooting procedure. It initializes the unknown states with an approximate profile, performs Newton iterations to drive the residual toward zero, and then evaluates the resulting solution at the mesh points. The output includes comparisons with the exact solution and a continuity check across subinterval boundaries, verifying that the propagation residuals have been effectively eliminated.

```rust
// Program 18.3.2: Multiple Shooting with Continuity Residuals
//
// Problem statement:
// This program demonstrates the multiple shooting method for the scalar
// two-point boundary value problem
//
//     y''(x) = -pi^2 sin(pi x),   0 <= x <= 1,
//     y(0) = 0,                   y(1) = 0.
//
// The exact solution is
//
//     y(x) = sin(pi x).
//
// The second-order equation is written as a first-order system:
//
//     y1' = y2,
//     y2' = -pi^2 sin(pi x).
//
// The interval is divided into m subintervals. At each node x_k, the unknown
// vector s_k approximates the full state [y(x_k), y'(x_k)]^T. Each subinterval
// is integrated as an initial value problem starting from s_k, and the
// propagation residual
//
//     R_k(s_k, s_{k+1}) = phi_k(s_k) - s_{k+1}
//
// enforces continuity across neighboring subintervals. The endpoint boundary
// conditions are added to form a square nonlinear system.

use std::f64::consts::PI;

const STATE_DIM: usize = 2;

#[derive(Clone, Copy)]
struct State {
    y1: f64,
    y2: f64,
}

impl State {
    
    fn add_scaled(self, other: State, scale: f64) -> Self {
        Self {
            y1: self.y1 + scale * other.y1,
            y2: self.y2 + scale * other.y2,
        }
    }

    fn sub(self, other: State) -> Self {
        Self {
            y1: self.y1 - other.y1,
            y2: self.y2 - other.y2,
        }
    }
}

fn forcing(x: f64) -> f64 {
    -PI * PI * (PI * x).sin()
}

fn exact_solution(x: f64) -> f64 {
    (PI * x).sin()
}

fn exact_derivative(x: f64) -> f64 {
    PI * (PI * x).cos()
}

fn rhs(x: f64, state: State) -> State {
    State {
        y1: state.y2,
        y2: forcing(x),
    }
}

fn rk4_step(x: f64, state: State, h: f64) -> State {
    let k1 = rhs(x, state);

    let s2 = state.add_scaled(k1, 0.5 * h);
    let k2 = rhs(x + 0.5 * h, s2);

    let s3 = state.add_scaled(k2, 0.5 * h);
    let k3 = rhs(x + 0.5 * h, s3);

    let s4 = state.add_scaled(k3, h);
    let k4 = rhs(x + h, s4);

    State {
        y1: state.y1 + h * (k1.y1 + 2.0 * k2.y1 + 2.0 * k3.y1 + k4.y1) / 6.0,
        y2: state.y2 + h * (k1.y2 + 2.0 * k2.y2 + 2.0 * k3.y2 + k4.y2) / 6.0,
    }
}

fn integrate_subinterval(x_left: f64, x_right: f64, initial_state: State, steps: usize) -> State {
    let h = (x_right - x_left) / steps as f64;

    let mut x = x_left;
    let mut state = initial_state;

    for _ in 0..steps {
        state = rk4_step(x, state, h);
        x += h;
    }

    state
}

fn unpack_states(u: &[f64], m: usize) -> Vec<State> {
    let mut states = Vec::with_capacity(m + 1);

    for k in 0..=m {
        states.push(State {
            y1: u[STATE_DIM * k],
            y2: u[STATE_DIM * k + 1],
        });
    }

    states
}

fn pack_states(states: &[State]) -> Vec<f64> {
    let mut u = Vec::with_capacity(STATE_DIM * states.len());

    for state in states {
        u.push(state.y1);
        u.push(state.y2);
    }

    u
}

fn multiple_shooting_residual(u: &[f64], m: usize, steps_per_segment: usize) -> Vec<f64> {
    let states = unpack_states(u, m);
    let h_mesh = 1.0 / m as f64;

    let mut residual = vec![0.0; STATE_DIM * (m + 1)];

    // Left boundary condition: y(0) = 0.
    residual[0] = states[0].y1;

    // Propagation residuals:
    //
    // R_k = phi_k(s_k) - s_{k+1}, for k = 0, ..., m - 1.
    //
    // These occupy entries 1 through 2m.
    let mut row = 1;

    for k in 0..m {
        let x_left = k as f64 * h_mesh;
        let x_right = (k + 1) as f64 * h_mesh;

        let propagated = integrate_subinterval(x_left, x_right, states[k], steps_per_segment);
        let jump = propagated.sub(states[k + 1]);

        residual[row] = jump.y1;
        residual[row + 1] = jump.y2;

        row += 2;
    }

    // Right boundary condition: y(1) = 0.
    residual[STATE_DIM * (m + 1) - 1] = states[m].y1;

    residual
}

fn finite_difference_jacobian(
    u: &[f64],
    m: usize,
    steps_per_segment: usize,
    epsilon: f64,
) -> Vec<Vec<f64>> {
    let n = u.len();
    let base = multiple_shooting_residual(u, m, steps_per_segment);
    let mut jacobian = vec![vec![0.0; n]; n];

    for col in 0..n {
        let mut perturbed = u.to_vec();
        perturbed[col] += epsilon;

        let residual_perturbed = multiple_shooting_residual(&perturbed, m, steps_per_segment);

        for row in 0..n {
            jacobian[row][col] = (residual_perturbed[row] - base[row]) / epsilon;
        }
    }

    jacobian
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn solve_dense_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
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
            return None;
        }

        if pivot_row != k {
            a.swap(k, pivot_row);
            b.swap(k, pivot_row);
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
        let mut sum = b[i];

        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }

        x[i] = sum / a[i][i];
    }

    Some(x)
}

fn initial_guess(m: usize) -> Vec<f64> {
    let mut states = Vec::with_capacity(m + 1);

    for k in 0..=m {
        let x = k as f64 / m as f64;

        // A deliberately approximate initial profile.
        // The sine shape is scaled slightly below the exact solution so that
        // Newton iteration has a visible correction to make.
        states.push(State {
            y1: 0.9 * exact_solution(x),
            y2: 0.9 * exact_derivative(x),
        });
    }

    pack_states(&states)
}

struct NewtonResult {
    u: Vec<f64>,
    residual: Vec<f64>,
    iterations: usize,
    converged: bool,
}

fn newton_multiple_shooting(
    m: usize,
    steps_per_segment: usize,
    tolerance: f64,
    max_iterations: usize,
) -> NewtonResult {
    let mut u = initial_guess(m);

    println!("Newton Iteration History");
    println!("------------------------");
    println!(
        "{:>6} {:>18} {:>18}",
        "iter", "||R(u)||_2", "max |R_i|"
    );

    for iteration in 0..=max_iterations {
        let residual = multiple_shooting_residual(&u, m, steps_per_segment);
        let residual_norm = norm2(&residual);
        let residual_max = max_abs(&residual);

        println!(
            "{:>6} {:>18.10e} {:>18.10e}",
            iteration, residual_norm, residual_max
        );

        if residual_norm < tolerance {
            return NewtonResult {
                u,
                residual,
                iterations: iteration,
                converged: true,
            };
        }

        let jacobian = finite_difference_jacobian(&u, m, steps_per_segment, 1.0e-6);
        let rhs: Vec<f64> = residual.iter().map(|r| -r).collect();

        let Some(delta) = solve_dense_linear_system(jacobian, rhs) else {
            return NewtonResult {
                u,
                residual,
                iterations: iteration,
                converged: false,
            };
        };

        let mut alpha = 1.0;
        let mut accepted = false;

        for _ in 0..12 {
            let trial_u: Vec<f64> = u
                .iter()
                .zip(delta.iter())
                .map(|(ui, di)| ui + alpha * di)
                .collect();

            let trial_residual = multiple_shooting_residual(&trial_u, m, steps_per_segment);

            if norm2(&trial_residual) < residual_norm {
                u = trial_u;
                accepted = true;
                break;
            }

            alpha *= 0.5;
        }

        if !accepted {
            let residual = multiple_shooting_residual(&u, m, steps_per_segment);

            return NewtonResult {
                u,
                residual,
                iterations: iteration,
                converged: false,
            };
        }
    }

    let residual = multiple_shooting_residual(&u, m, steps_per_segment);

    NewtonResult {
        u,
        residual,
        iterations: max_iterations,
        converged: false,
    }
}

fn main() {
    let m = 5;
    let steps_per_segment = 20;
    let tolerance = 1.0e-10;
    let max_iterations = 12;

    println!("Multiple Shooting for a Two-Point Boundary Value Problem");
    println!("=======================================================");
    println!();
    println!("Model problem:");
    println!("  y''(x) = -pi^2 sin(pi x),   0 <= x <= 1");
    println!("  y(0) = 0,                   y(1) = 0");
    println!("  Exact solution: y(x) = sin(pi x)");
    println!();
    println!("Multiple shooting setup:");
    println!("  Number of subintervals m        = {}", m);
    println!("  Number of nodes m + 1           = {}", m + 1);
    println!("  State dimension N               = {}", STATE_DIM);
    println!(
        "  Nonlinear system dimension      = {}",
        STATE_DIM * (m + 1)
    );
    println!("  RK4 steps per subinterval       = {}", steps_per_segment);
    println!();

    let result = newton_multiple_shooting(m, steps_per_segment, tolerance, max_iterations);
    let states = unpack_states(&result.u, m);

    println!();
    println!("Multiple Shooting Result");
    println!("------------------------");
    println!("Converged                         = {}", result.converged);
    println!("Iterations performed              = {}", result.iterations);
    println!("Final residual norm               = {:.12e}", norm2(&result.residual));
    println!("Final maximum residual            = {:.12e}", max_abs(&result.residual));
    println!();

    println!("Node Values and Exact Comparison");
    println!("--------------------------------");
    println!(
        "{:>6} {:>12} {:>16} {:>16} {:>16} {:>16}",
        "k", "x_k", "y_k", "y_exact", "y'_k", "y'_exact"
    );

    for k in 0..=m {
        let x = k as f64 / m as f64;
        println!(
            "{:>6} {:>12.6} {:>16.10} {:>16.10} {:>16.10} {:>16.10}",
            k,
            x,
            states[k].y1,
            exact_solution(x),
            states[k].y2,
            exact_derivative(x)
        );
    }

    println!();
    println!("Continuity Check Across Subintervals");
    println!("------------------------------------");
    println!(
        "{:>6} {:>14} {:>18} {:>18}",
        "k", "x_{k+1}", "jump in y", "jump in y'"
    );

    for k in 0..m {
        let x_left = k as f64 / m as f64;
        let x_right = (k + 1) as f64 / m as f64;

        let propagated =
            integrate_subinterval(x_left, x_right, states[k], steps_per_segment);
        let jump = propagated.sub(states[k + 1]);

        println!(
            "{:>6} {:>14.8} {:>18.10e} {:>18.10e}",
            k,
            x_right,
            jump.y1,
            jump.y2
        );
    }
}
```

Program 18.3.2 demonstrates the practical realization of the multiple shooting method and highlights its advantages over single and double shooting formulations. By introducing intermediate states and enforcing continuity locally, the method reduces the sensitivity associated with long-range propagation and leads to a better-conditioned nonlinear system.

The results illustrate that the propagation residuals are driven to machine precision, confirming that the solution is continuous across subinterval boundaries and satisfies the differential equation on each segment. The close agreement between the computed and exact solutions further demonstrates the accuracy of the method.

An important structural feature of the implementation is the block-coupled nature of the nonlinear system, which arises from the local coupling between neighboring states. While the present implementation uses a dense solver for simplicity, this structure can be exploited in more advanced implementations to achieve greater efficiency through block elimination or sparse linear algebra techniques.

The modular design of the code also reflects the inherent parallelism of multiple shooting. Since each subinterval integration depends only on the local state, these computations can be performed independently, offering significant opportunities for parallel execution in large-scale applications.

Overall, the program provides a clear computational illustration of the theoretical framework developed in Section 18.3.4 and establishes a foundation for extending multiple shooting methods to more complex and higher-dimensional boundary value problems.

## 18.3.5. Conditioning and Singular Problem Handling

The principal advantage of fitting-point and multiple-shooting methods lies in their ability to improve the conditioning of the boundary value problem by localizing the propagation of perturbations. As established in Section 18.2.5, the sensitivity of the solution is governed by the fundamental matrix $\boldsymbol\Phi(x,a)$, whose growth reflects the presence of unstable modes. In standard shooting, perturbations introduced at $x=a$ are transported across the entire interval, leading to amplification proportional to $|\boldsymbol\Phi(b,a)|$. When this quantity is large, the residual function becomes highly sensitive, and the associated Newton system is poorly conditioned.

By contrast, fitting-point and multiple-shooting methods partition the interval into smaller subintervals. In each subinterval $[x_k, x_{k+1}]$, the propagation of perturbations is governed by a local fundamental matrix $\boldsymbol\Phi(x_{k+1},x_k)$. The overall sensitivity is thus distributed across multiple segments rather than concentrated in a single long propagation. This leads to a significant reduction in the effective amplification of errors, since the norm of each local propagator is typically much smaller than that of the full-interval propagator. As a result, the Jacobian matrices arising in Newton iterations exhibit improved conditioning, and the convergence behavior becomes more stable and predictable.

An additional advantage emerges in the treatment of *singular boundary points*. In many physically relevant problems, the governing differential equation becomes singular at one or both endpoints. Examples include radial formulations of diffusion problems, where coefficients may vanish or diverge at the origin. Direct numerical integration starting exactly at such singular points is generally not feasible, as the equations may be undefined or highly ill-conditioned.

A standard analytical remedy is to construct an asymptotic expansion of the solution in a neighborhood of the singular point. This expansion provides a regularized representation of the solution that can be evaluated at a small distance $\varepsilon > 0$ from the singularity. The numerical integration is then initiated at $x=a+\varepsilon$ (or $(x=b-\varepsilon)$) using this asymptotic approximation as initial data.

Fitting-point and multiple-shooting methods integrate naturally with this strategy. One may initialize the solution near the singular endpoint using the asymptotic expansion and then propagate toward an interior fitting point or across subintervals. Since the singular region is confined to a small portion of the domain, and matching is enforced away from the singularity, the numerical method avoids direct exposure to the ill-conditioned behavior at the endpoint. This makes these methods particularly effective for *hypersensitive* or singular boundary value problems, where standard shooting would fail due to extreme sensitivity or breakdown of the differential formulation at the boundary.

## 18.3.6. Application to Indirect Optimal Control Systems

A major class of applications for fitting-point and multiple-shooting methods arises in indirect optimal control. In this setting, the necessary conditions for optimality lead to a coupled system of differential equations for the state and costate variables. The governing equations take the form:

\begin{equation}
\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x},\mathbf{u},t), 
\qquad
\dot{\boldsymbol{\lambda}} = -\nabla_{\mathbf{x}} H(\mathbf{x},\mathbf{u},\boldsymbol{\lambda},t), 
\qquad
\mathbf{0} = \nabla_{\mathbf{u}} H(\mathbf{x},\mathbf{u},\boldsymbol{\lambda},t)
\tag{18.3.10}
\end{equation}

where $\mathbf x(t)$ denotes the state, $\boldsymbol\lambda(t)$ the costate, $\mathbf u(t)$ the control, and $H$ the Hamiltonian function.

In this formulation, the state variables are typically prescribed at the initial time, while the costate variables are subject to terminal conditions. This leads to a TPBVP in which the initial values of $\boldsymbol\lambda(t_0)$ are unknown and must be determined so that the terminal constraints are satisfied. The resulting problem is inherently nonlinear and globally coupled.

A defining numerical challenge in such systems is the presence of unstable directions in the adjoint dynamics. When the costate equations are integrated forward in time, perturbations may grow rapidly due to the structure of the linearized Hamiltonian system. This instability directly affects the sensitivity of the terminal conditions with respect to the initial costate values, making the shooting formulation highly ill-conditioned.

Fitting-point and multiple-shooting methods provide an effective mechanism for addressing this difficulty. By dividing the time interval into smaller segments, the growth of perturbations is restricted within each subinterval. In double shooting, forward integration of the state and backward integration of the costate naturally align with the directions of stability of the respective subsystems. In multiple shooting, the segmentation further reduces the effective propagation length of unstable modes, leading to a more balanced and stable numerical formulation.

Another important aspect is the enlargement of the convergence region for Newton iterations. In standard shooting, the strong sensitivity of the terminal state to initial parameters often leads to narrow basins of attraction, requiring highly accurate initial guesses. By contrast, the improved conditioning of fitting-point and multiple-shooting formulations allows for a wider range of initial guesses to converge, thereby enhancing robustness in practical computations.

Thus, in indirect optimal control problems, these methods are not merely advantageous but often essential, as they provide the stability and flexibility required to handle the coupled and potentially unstable dynamics of state–costate systems.

## 18.3.7. Practical Guidelines for Method Selection

The choice between standard shooting, fitting-point shooting, and multiple shooting depends on the structural and numerical properties of the problem under consideration. While all these methods share a common conceptual foundation, their performance can differ significantly depending on stability, sensitivity, and problem scale.

Fitting-point and multiple-shooting methods should be preferred over pure shooting in the following situations:

- **Presence of unstable modes:**\
  When the differential system exhibits exponential growth of perturbations, the sensitivity of the solution with respect to initial conditions becomes large. Dividing the interval reduces this amplification and improves conditioning.
- **Long integration intervals:**\
  If the interval $[a,b]$ is large relative to the characteristic growth or decay rates of the system, even moderate instability can accumulate over the full domain. Subinterval formulations mitigate this accumulation.
- **Singular boundary points:**\
  When singularities occur at endpoints, asymptotic expansions can be used to initialize the solution near the boundary, and matching can be performed away from the singular region. This is naturally accommodated within fitting-point and multiple-shooting frameworks.
- **Indirect optimal control problems:**\
  Systems involving coupled state and costate variables often exhibit strong instability in the adjoint dynamics. Interval decomposition improves stability and enlarges the convergence region for nonlinear solvers.

In addition to these criteria, practical considerations such as computational resources and implementation complexity may also influence the choice of method. Standard shooting remains attractive for low-dimensional, well-conditioned problems due to its simplicity and minimal memory requirements. However, as the problem becomes more sensitive or higher-dimensional, the advantages of fitting-point and multiple-shooting methods become increasingly significant.

In summary, fitting-point and multiple-shooting methods retain the conceptual simplicity of shooting while substantially improving robustness. By localizing instability and distributing computational effort across subintervals, they provide a flexible and effective framework for solving challenging boundary value problems encountered in modern applications.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/EDKDvjsKN9kzDyyz16db.4","tags":[]}

# 18.4. Relaxation Methods

Relaxation methods provide an alternative to shooting-based approaches by treating the two-point boundary value problem as a *global system* rather than a sequence of initial value problems. Instead of determining unknown initial conditions and propagating solutions across the interval, relaxation methods represent the solution explicitly over a discrete mesh and enforce the governing equations simultaneously at all points. This leads to a nonlinear algebraic system whose solution approximates the continuous problem.

From a conceptual standpoint, relaxation replaces the idea of trajectory propagation with that of *global consistency*. Every component of the solution is adjusted iteratively so that the differential equation and boundary conditions are satisfied throughout the entire domain. This global coupling avoids the instability issues associated with long integrations in shooting methods and provides a natural framework for handling stiff, unstable, or highly sensitive systems.

## 18.4.1. Discrete Formulation and Residual System

To construct the relaxation formulation, the interval $[a,b]$ is discretized into a mesh:

\begin{equation}
a = x_0 < x_1 < \cdots < x_M = b, 
\qquad 
\mathbf{y}_k \approx \mathbf{y}(x_k)
\tag{18.4.1}
\end{equation}

The unknown function $\mathbf y(x)$ is thus represented by the collection of vectors $\{\mathbf y_k\}_{k=0}^M$, which together form the global solution vector.

The differential equation is replaced by a set of *discrete residual equations* that approximate the continuous system on each subinterval. For a first-order system, a second-order accurate midpoint discretization yields:

\begin{equation}
\begin{aligned}
\mathbf{E}_k(\mathbf{y}_{k-1},\mathbf{y}_k)
&= \mathbf{y}_k - \mathbf{y}_{k-1}
- h_k\,\mathbf{g}\!\left(\frac{x_{k-1}+x_k}{2}, \frac{\mathbf{y}_{k-1}+\mathbf{y}_k}{2}\right) \\
&= \mathbf{0}, \qquad k = 1,\dots,M
\end{aligned}
\tag{18.4.2}
\end{equation}

where $h_k = x_k - x_{k-1}$ denotes the step size of the $k$-th subinterval. This expression enforces the differential equation in an averaged sense, using midpoint values of both the independent variable and the state. The resulting scheme is second-order accurate and provides a balanced approximation that depends symmetrically on adjacent mesh points.

The boundary conditions are incorporated directly into the system as residual equations at the endpoints:

\begin{equation}
\mathbf{E}_0(\mathbf{y}_0) = \mathbf{B}_1(\mathbf{y}_0) = \mathbf{0}, 
\qquad
\mathbf{E}_{M+1}(\mathbf{y}_M) = \mathbf{B}_2(\mathbf{y}_M) = \mathbf{0}
\tag{18.4.3}
\end{equation}

This ensures that the discrete solution satisfies the prescribed constraints at (x=a) and (x=b) without requiring any special treatment during iteration.

Collecting all residual equations, including both interior discretization conditions and boundary constraints, yields a global nonlinear algebraic system:

\begin{equation}
\mathbf{E}(\mathbf{Y}) = \mathbf{0}, 
\qquad
\mathbf{Y} =
\begin{bmatrix}
\mathbf{y}_0 \\
\mathbf{y}_1 \\
\vdots \\
\mathbf{y}_M
\end{bmatrix}
\in \mathbb{R}^{N(M+1)}
\tag{18.4.4}
\end{equation}

This formulation represents a fundamental shift in perspective. The unknown is no longer a function determined indirectly through initial conditions, but rather a finite-dimensional vector containing the solution at all mesh points. The residual $\mathbf E(\mathbf Y)$ encodes both the differential equation and the boundary conditions, and solving the TPBVP amounts to finding a vector $\mathbf Y$ that simultaneously satisfies all these constraints.

From a numerical standpoint, the structure of the system is highly organized. Each interior residual $\mathbf E_k$ couples only neighboring variables $\mathbf y_{k-1}$ and $\mathbf y_k$, leading to a sparse, banded Jacobian matrix. This sparsity is a key advantage, as it allows efficient storage and solution of the resulting linear systems during Newton iterations. Moreover, the global nature of the formulation ensures that errors are distributed across the domain rather than amplified through propagation, contributing to improved stability for difficult problems.

Thus, relaxation methods convert the TPBVP into a structured nonlinear system defined over the entire interval, providing a robust and flexible framework that complements shooting-based approaches, particularly in cases where stability and conditioning are of primary concern.

## 18.4.2. Newton Iteration and Block-Tridiagonal Structure

The nonlinear algebraic system (18.4.4) arising from the relaxation formulation is solved using Newton’s method, applied directly to the global residual operator. Given a current iterate $\mathbf Y^{(m)}$, the Newton correction $\Delta \mathbf Y$ is obtained by solving the linear system:

$$\mathbf J(\mathbf Y^{(m)})\,\Delta \mathbf Y = -\mathbf E(\mathbf Y^{(m)}),\qquad\mathbf Y^{(m+1)} = \mathbf Y^{(m)} + \alpha_m \Delta \mathbf Y, \tag{18.4.5}$$

where $\alpha_m \in (0,1]$ is a damping parameter introduced to enhance robustness, particularly in the early stages of iteration or when the initial guess is not sufficiently close to the solution.

The Jacobian matrix $\mathbf J(\mathbf Y)$ reflects the dependence of each residual equation on the neighboring solution values. For the midpoint discretization (18.4.2), each residual $\mathbf E_k$ depends only on $\mathbf y_{k-1}$ and $\mathbf y_k$, leading to a highly structured sparsity pattern. Specifically, the Jacobian has a *block-tridiagonal* form, with each block of size $N \times N$.

The interior Jacobian blocks are obtained by differentiating the residual (18.4.2) with respect to the neighboring variables. This yields:

\begin{equation}
\mathbf{B}_k
= \frac{\partial \mathbf{E}_k}{\partial \mathbf{y}_{k-1}}
= -\mathbf{I} - \frac{h_k}{2}\,\mathbf{g}_{\mathbf{y}}(\bar{x}_k,\bar{y}_k)
\tag{18.4.6}
\end{equation}

\begin{equation}
\mathbf{A}_k
= \frac{\partial \mathbf{E}_k}{\partial \mathbf{y}_k}
= \mathbf{I} - \frac{h_k}{2}\,\mathbf{g}_{\mathbf{y}}(\bar{x}_k,\bar{y}_k)
\tag{18.4.7}
\end{equation}

where the midpoint quantities are defined as:

\begin{equation}
\bar{x}_k = \frac{x_{k-1}+x_k}{2}, 
\qquad
\bar{y}_k = \frac{\mathbf{y}_{k-1}+\mathbf{y}_k}{2}
\tag{18.4.8}
\end{equation}

These expressions reveal that the Jacobian blocks incorporate both the identity contribution from the discrete difference $\mathbf y_k - \mathbf y_{k-1}$ and the linearized effect of the differential equation through the Jacobian $\mathbf g_{\mathbf y}$. The factor of $1/2$ arises from the midpoint approximation, which distributes the dependence symmetrically between adjacent nodes.

In addition to the diagonal blocks $\mathbf A_k$ and subdiagonal blocks $\mathbf B_k$, the system may include superdiagonal blocks $\mathbf C_k$, depending on the ordering of variables and residuals. Collectively, the Newton system takes the form:

\begin{equation}
\begin{bmatrix}
\mathbf{A}_0 & \mathbf{C}_0 & & & 0 \\
\mathbf{B}_1 & \mathbf{A}_1 & \mathbf{C}_1 & & \\
& \ddots & \ddots & \ddots & \\
0 & & \mathbf{B}_{M-1} & \mathbf{A}_{M-1} & \mathbf{C}_{M-1} \\
0 & & & \mathbf{B}_M & \mathbf{A}_M
\end{bmatrix}
\begin{bmatrix}
\Delta \mathbf{y}_0 \\
\Delta \mathbf{y}_1 \\
\vdots \\
\Delta \mathbf{y}_{M-1} \\
\Delta \mathbf{y}_M
\end{bmatrix}
=
-
\begin{bmatrix}
\mathbf{r}_0 \\
\mathbf{r}_1 \\
\vdots \\
\mathbf{r}_{M-1} \\
\mathbf{r}_M
\end{bmatrix}
\tag{18.4.9}
\end{equation}

Here, $\mathbf r_k = -\mathbf E_k(\mathbf Y^{(m)})$ denotes the residual vector at each node. The boundary conditions contribute to the first and last block rows, modifying $\mathbf A_0$ and $\mathbf A_M$ accordingly.

This block-tridiagonal structure is a direct consequence of the *local coupling* inherent in the discretization. Each equation involves only neighboring unknowns, so the global system, while large, has a sparse and highly structured form. This property is fundamental to the efficiency of relaxation methods. It allows the use of specialized linear solvers for banded or block-tridiagonal systems, reducing both computational cost and memory requirements compared to dense formulations.

From a computational perspective, this structure embodies a key distinction between relaxation and shooting methods. Relaxation treats the problem globally, enforcing consistency across the entire domain in a single system, yet the interactions remain local, confined to adjacent mesh points. This combination of global formulation and local coupling provides both stability and efficiency, particularly for large-scale or stiff problems where shooting methods may encounter difficulties.

Thus, the Newton iteration for relaxation methods leverages the structured sparsity of the discretized system, enabling efficient and robust solution of the TPBVP while maintaining a clear correspondence between the continuous problem and its discrete approximation.

## 18.4.3. Computational Complexity and Efficiency

A central advantage of relaxation methods is the ability to exploit the *block-tridiagonal structure* of the Jacobian matrix arising from the discretization. As shown in Section 18.4.2, each residual equation couples only neighboring mesh points, leading to a sparse linear system in which nonzero blocks appear only on the main diagonal and the first sub- and super-diagonals.

This structure permits the use of efficient *direct block elimination* methods, which are the natural generalization of the Thomas algorithm for tridiagonal systems. In this approach, the system is solved by forward elimination followed by backward substitution, operating on blocks of size $N \times N$. Because each step involves matrix operations on these blocks, the computational cost per step scales as $O(N^3)$, and the total cost over all $M$ intervals is:

\begin{equation}
T_{\text{relax}} = O(MN^3), 
\qquad
S_{\text{relax}} = O(MN^2)
\tag{18.4.10}
\end{equation}

Here, $T_{\text{relax}}$ denotes the total computational cost, while $S_{\text{relax}}$ represents the storage requirement. The $O(MN^2)$ memory usage reflects the need to store the block-tridiagonal matrix, with $M$ blocks of size $N \times N$.

These complexity estimates highlight the efficiency gains obtained by exploiting structure. If the same problem were treated as a fully dense system of dimension $N(M+1) \times N(M+1)$, the cost of direct solution would scale as $O\big((MN)^3\big)$, which is prohibitively expensive for even moderate values of $M$. The block-tridiagonal formulation therefore reduces the computational complexity by several orders of magnitude, making large-scale discretizations feasible.

The efficiency is particularly favorable when the state dimension $N$ is small. For example, in scalar second-order boundary value problems reduced to first-order systems with $N=2$, each block operation is inexpensive, and the overall computational cost grows essentially linearly with the number of mesh intervals $M$. In such cases, relaxation methods provide a highly efficient and robust approach for obtaining accurate solutions.

From a broader perspective, these results emphasize a key strength of relaxation methods: they combine *global formulation* with *local computational structure*. The global coupling ensures stability and robustness, while the local sparsity enables efficient numerical solution. This balance makes relaxation methods especially attractive for problems where shooting methods become inefficient or ill-conditioned, particularly when fine spatial resolution or complex dynamics are involved.

### Rust Implementation

Following the development of relaxation methods in Sections 18.4.1–18.4.3, Program 18.4.1 provides a practical implementation of the midpoint residual formulation for two-point boundary value problems. In this approach, the continuous problem is discretized on a mesh, and the differential equation is enforced locally on each subinterval through the midpoint approximation, as introduced in equations (18.4.1)–(18.4.4). The resulting nonlinear system is then solved using Newton’s method, consistent with the framework described in equation (18.4.5). Unlike shooting methods, which propagate solutions across the entire domain, relaxation methods treat all mesh values simultaneously, leading to improved stability and robustness. The present program demonstrates this formulation by combining a midpoint discretization with a Newton iteration and a dense linear solve, thereby illustrating how the global residual is driven to zero across the entire mesh.

At the core of the implementation is the function `residual`, which constructs the discrete residual system corresponding to the midpoint formulation in equations (18.4.1)–(18.4.4). For each subinterval, the midpoint value of the state is computed and used to evaluate the vector field, while the difference between consecutive mesh values enforces the discrete evolution of the system. This produces a set of algebraic equations that approximate the differential equation locally, while the boundary conditions are incorporated by fixing the values at the endpoints. Together, these constraints define a nonlinear system whose solution approximates the continuous boundary value problem.

The mapping between the continuous formulation and the discrete unknowns is handled by the functions `pack` and `unpack`, which convert between structured state vectors and a flattened representation suitable for numerical linear algebra. This abstraction allows the global system to be expressed in a standard vector form while preserving the interpretation of each component as a state variable at a mesh point. The initial guess is constructed by the function `initial_guess`, which provides a smooth approximation to the solution and ensures that the Newton iteration begins in a reasonable region of the solution space.

The nonlinear system is solved using Newton’s method, implemented in the function `newton_relaxation`. At each iteration, the residual is evaluated and a Jacobian matrix is constructed using finite differences. This Jacobian approximates the linearization of the residual system described in equation (18.4.5), and the resulting linear system is solved using the function `solve_dense`, which performs Gaussian elimination. Although this approach does not explicitly exploit the block-tridiagonal structure discussed in equations (18.4.6)–(18.4.10), it provides a clear and general implementation suitable for moderate problem sizes.

The convergence of the method is monitored using both the Euclidean norm and the maximum absolute value of the residual. A simple backtracking strategy is employed to ensure that each Newton update reduces the residual norm, thereby improving robustness. This reflects the practical importance of damping in nonlinear solvers, particularly when the initial guess is not sufficiently close to the solution.

The `main` function demonstrates the complete relaxation procedure. It initializes the mesh, performs Newton iterations, and then evaluates the numerical solution at selected points. The computed values are compared with the exact solution and its derivative, providing a direct assessment of accuracy. This comparison highlights the second-order accuracy of the midpoint discretization and illustrates how the error decreases with mesh refinement.

```rust
// Program 18.4.1: Relaxation Method with Midpoint Residuals and Newton Iteration
//
// Problem statement:
// Solve
//
//     y''(x) = -pi^2 sin(pi x),   0 <= x <= 1,
//     y(0) = 0,                   y(1) = 0.
//
// The exact solution is y(x) = sin(pi x).
//
// The second-order equation is rewritten as
//
//     y1' = y2,
//     y2' = -pi^2 sin(pi x).
//
// The midpoint relaxation residual is imposed on every subinterval, while
// the two boundary conditions are imposed at the endpoints.

use std::f64::consts::PI;

const N: usize = 2;

type State = [f64; N];

fn forcing(x: f64) -> f64 {
    -PI * PI * (PI * x).sin()
}

fn exact_solution(x: f64) -> f64 {
    (PI * x).sin()
}

fn exact_derivative(x: f64) -> f64 {
    PI * (PI * x).cos()
}

fn g(x: f64, y: State) -> State {
    [y[1], forcing(x)]
}

fn pack(states: &[State]) -> Vec<f64> {
    let mut u = Vec::with_capacity(2 * states.len());
    for s in states {
        u.push(s[0]);
        u.push(s[1]);
    }
    u
}

fn unpack(u: &[f64]) -> Vec<State> {
    let mut states = Vec::with_capacity(u.len() / 2);
    for k in 0..u.len() / 2 {
        states.push([u[2 * k], u[2 * k + 1]]);
    }
    states
}

fn initial_guess(m: usize) -> Vec<f64> {
    let mut states = Vec::with_capacity(m + 1);

    for k in 0..=m {
        let x = k as f64 / m as f64;
        states.push([
            0.85 * exact_solution(x),
            0.85 * exact_derivative(x),
        ]);
    }

    pack(&states)
}

fn residual(u: &[f64], m: usize) -> Vec<f64> {
    let states = unpack(u);
    let h = 1.0 / m as f64;

    let mut r = vec![0.0; 2 * (m + 1)];

    // Boundary condition y(0) = 0.
    r[0] = states[0][0];

    // Midpoint residuals on all subintervals.
    // These occupy rows 1 through 2M.
    let mut row = 1;

    for k in 1..=m {
        let x_left = (k - 1) as f64 * h;
        let x_right = k as f64 * h;
        let x_mid = 0.5 * (x_left + x_right);

        let y_left = states[k - 1];
        let y_right = states[k];

        let y_mid = [
            0.5 * (y_left[0] + y_right[0]),
            0.5 * (y_left[1] + y_right[1]),
        ];

        let g_mid = g(x_mid, y_mid);

        r[row] = y_right[0] - y_left[0] - h * g_mid[0];
        r[row + 1] = y_right[1] - y_left[1] - h * g_mid[1];

        row += 2;
    }

    // Boundary condition y(1) = 0.
    r[2 * (m + 1) - 1] = states[m][0];

    r
}

fn finite_difference_jacobian(u: &[f64], m: usize, eps: f64) -> Vec<Vec<f64>> {
    let n = u.len();
    let base = residual(u, m);

    let mut j = vec![vec![0.0; n]; n];

    for col in 0..n {
        let mut perturbed = u.to_vec();
        perturbed[col] += eps;

        let rp = residual(&perturbed, m);

        for row in 0..n {
            j[row][col] = (rp[row] - base[row]) / eps;
        }
    }

    j
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();

    for k in 0..n {
        let mut pivot = k;
        let mut pivot_abs = a[k][k].abs();

        for i in k + 1..n {
            if a[i][k].abs() > pivot_abs {
                pivot_abs = a[i][k].abs();
                pivot = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return None;
        }

        if pivot != k {
            a.swap(k, pivot);
            b.swap(k, pivot);
        }

        for i in k + 1..n {
            let factor = a[i][k] / a[k][k];

            for j in k..n {
                a[i][j] -= factor * a[k][j];
            }

            b[i] -= factor * b[k];
        }
    }

    let mut x = vec![0.0; n];

    for i in (0..n).rev() {
        let mut sum = b[i];

        for j in i + 1..n {
            sum -= a[i][j] * x[j];
        }

        x[i] = sum / a[i][i];
    }

    Some(x)
}

fn newton_relaxation(
    m: usize,
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, Vec<f64>, usize, bool) {
    let mut u = initial_guess(m);

    println!("Newton Iteration History");
    println!("------------------------");
    println!(
        "{:>6} {:>18} {:>18} {:>12}",
        "iter", "||E(Y)||_2", "max |E_i|", "alpha"
    );

    for iteration in 0..=max_iterations {
        let r = residual(&u, m);
        let r_norm = norm2(&r);
        let r_max = max_abs(&r);

        println!(
            "{:>6} {:>18.10e} {:>18.10e} {:>12}",
            iteration, r_norm, r_max, "-"
        );

        if r_norm < tolerance {
            return (u, r, iteration, true);
        }

        let j = finite_difference_jacobian(&u, m, 1.0e-6);
        let rhs: Vec<f64> = r.iter().map(|x| -x).collect();

        let Some(delta) = solve_dense(j, rhs) else {
            return (u, r, iteration, false);
        };

        let mut alpha = 1.0;
        let mut accepted = false;

        for _ in 0..12 {
            let trial_u: Vec<f64> = u
                .iter()
                .zip(delta.iter())
                .map(|(ui, di)| ui + alpha * di)
                .collect();

            let trial_r = residual(&trial_u, m);

            if norm2(&trial_r) < r_norm {
                u = trial_u;
                accepted = true;

                println!(
                    "{:>6} {:>18} {:>18} {:>12.6}",
                    "", "", "", alpha
                );

                break;
            }

            alpha *= 0.5;
        }

        if !accepted {
            let r = residual(&u, m);
            return (u, r, iteration, false);
        }
    }

    let r = residual(&u, m);
    (u, r, max_iterations, false)
}

fn main() {
    let m = 40;
    let tolerance = 1.0e-10;
    let max_iterations = 20;

    println!("Relaxation Method for a Two-Point Boundary Value Problem");
    println!("========================================================");
    println!();
    println!("Model problem:");
    println!("  y''(x) = -pi^2 sin(pi x),   0 <= x <= 1");
    println!("  y(0) = 0,                   y(1) = 0");
    println!("  Exact solution: y(x) = sin(pi x)");
    println!();
    println!("Relaxation setup:");
    println!("  Mesh intervals M                 = {}", m);
    println!("  Mesh points M + 1                = {}", m + 1);
    println!("  State dimension N                = {}", N);
    println!("  Global unknown dimension         = {}", N * (m + 1));
    println!("  Midpoint residual discretization = second order");
    println!("  Linear solver                    = dense Newton solve");
    println!();

    let (u, final_residual, iterations, converged) =
        newton_relaxation(m, tolerance, max_iterations);

    let solution = unpack(&u);

    println!();
    println!("Relaxation Result");
    println!("-----------------");
    println!("Converged                         = {}", converged);
    println!("Iterations performed              = {}", iterations);
    println!("Final residual norm               = {:.12e}", norm2(&final_residual));
    println!("Final maximum residual            = {:.12e}", max_abs(&final_residual));
    println!();

    println!("Representative Mesh Values");
    println!("--------------------------");
    println!(
        "{:>8} {:>12} {:>18} {:>18} {:>18}",
        "k", "x_k", "y_num", "y_exact", "abs error"
    );

    for k in [0, 10, 20, 30, 40] {
        let x = k as f64 / m as f64;
        let exact = exact_solution(x);

        println!(
            "{:>8} {:>12.6} {:>18.10} {:>18.10} {:>18.10e}",
            k,
            x,
            solution[k][0],
            exact,
            (solution[k][0] - exact).abs()
        );
    }

    println!();
    println!("Derivative Values");
    println!("-----------------");
    println!(
        "{:>8} {:>12} {:>18} {:>18} {:>18}",
        "k", "x_k", "y'_num", "y'_exact", "abs error"
    );

    for k in [0, 10, 20, 30, 40] {
        let x = k as f64 / m as f64;
        let exact = exact_derivative(x);

        println!(
            "{:>8} {:>12.6} {:>18.10} {:>18.10} {:>18.10e}",
            k,
            x,
            solution[k][1],
            exact,
            (solution[k][1] - exact).abs()
        );
    }
}
```

Program 18.4.1 demonstrates the practical implementation of the relaxation method using a midpoint discretization and Newton iteration. This approach reflects the central idea of Section 18.4: transforming a boundary value problem into a system of algebraic equations defined over a mesh and solving this system simultaneously.

The results illustrate that the residual can be driven to machine precision, indicating that the discrete system is satisfied to a high degree of accuracy. At the same time, the numerical solution closely approximates the exact solution, with errors consistent with the second-order accuracy of the discretization. This confirms the effectiveness of the relaxation method in producing stable and accurate solutions.

An important advantage of the relaxation approach is its global nature. By treating all mesh values as unknowns, the method avoids the sensitivity issues associated with shooting methods and provides a more robust framework for solving boundary value problems, particularly in the presence of stiff or unstable dynamics.

The modular structure of the implementation also allows for straightforward extension to more advanced formulations. In particular, the use of structured Jacobians and block-tridiagonal solvers can significantly improve efficiency for large systems, while adaptive mesh refinement and higher-order discretizations can further enhance accuracy. Overall, the program provides a clear computational realization of the relaxation framework and establishes a foundation for extending these methods to more complex and high-dimensional problems.

## 18.4.4. Stability Advantages of Relaxation Methods and Global Consistency Enforcement

A fundamental strength of relaxation methods lies in their *global enforcement of consistency*, which directly leads to superior numerical stability when compared with shooting-based approaches. This advantage originates from the way the solution is constructed. Instead of propagating the solution sequentially from one endpoint, relaxation determines the entire discrete solution vector simultaneously by solving a coupled nonlinear system over the full mesh.

In shooting methods, the numerical solution is obtained by integrating an initial value problem from $x=a$ to $x=b$. When the underlying differential system contains unstable modes, perturbations introduced at the initial point are amplified during this propagation. Since the terminal boundary condition is only evaluated after the full integration, there is no mechanism to control or suppress this growth during the evolution. As a consequence, unstable components may dominate the solution, leading to severe sensitivity, ill-conditioning of the residual function, and potential failure of Newton iterations.

Relaxation methods avoid this mechanism entirely. The discrete system (18.4.2)–(18.4.4) enforces the governing equations at every mesh interval while simultaneously incorporating both boundary conditions. The solution is therefore not constructed through forward or backward propagation, but rather through a *global adjustment* of all unknowns so that the residual vanishes everywhere on the domain. This eliminates the possibility of uncontrolled growth of unstable modes, since any deviation must satisfy the coupled constraints imposed across the entire system.

From a linearized viewpoint, the global Newton system incorporates the effects of both stable and unstable modes in a balanced manner. The block-tridiagonal Jacobian couples neighboring mesh points, ensuring that perturbations are distributed locally and constrained globally. Unlike shooting, where instability manifests as exponential growth in the fundamental matrix, relaxation embeds the stability properties directly into the algebraic structure of the problem.

This global consistency yields several important advantages:

- **Suppression of unstable modes:**\
  Unstable components cannot grow freely, as the simultaneous enforcement of boundary conditions constrains their influence across the entire mesh.
- **Improved conditioning:**\
  The residual reflects discrepancies distributed over the domain rather than concentrated at a single endpoint, leading to better-conditioned Newton systems.
- **Robustness for hypersensitive problems:**\
  In problems where small perturbations produce large changes in the solution, relaxation avoids the amplification mechanisms inherent in propagation-based methods.
- **Effective handling of singular behavior:**\
  Singularities or near-singular regions at the boundaries do not disrupt the computation in the same way as in shooting, since the solution is determined collectively rather than through sequential integration.
- **Stable convergence of Newton iterations:**\
  The global structure of the residual leads to more reliable convergence behavior, even when the underlying differential system exhibits strong instability.

In summary, the stability advantages of relaxation methods are a direct consequence of their global formulation. By enforcing both the differential equations and boundary conditions simultaneously across the mesh, they provide a robust and well-conditioned framework for solving boundary value problems that are unstable, singular, or highly sensitive. This makes relaxation methods an essential complement to shooting-based techniques in the numerical treatment of TPBVPs.

## 18.4.5. Extension of Relaxation Methods to Coupled Differential–Algebraic Systems

An important practical advantage of relaxation methods is their flexibility in the representation of unknowns. In contrast to shooting-based approaches, where the formulation is tightly tied to the differential structure of the original system, relaxation methods operate at the level of a *discrete residual system*. This allows the introduction of additional variables that do not necessarily correspond directly to the original differential state variables.

In many applications, the governing physics is not expressed purely in differential form. Instead, it may include *implicit constitutive relations*, constraints, or equations of state that relate variables algebraically. Examples include relations between thermodynamic quantities, nonlinear material laws, or constraints arising from conservation principles. In a shooting framework, such relations often require nested nonlinear solves or implicit inversions within each step of the integration, increasing both complexity and computational cost.

Relaxation methods provide a natural alternative. Additional algebraic variables can be introduced directly at the discrete level, augmenting the solution vector. The governing equations are then expressed as a combined system of discrete differential residuals and algebraic constraints, all enforced simultaneously. The resulting formulation takes the form of a *coupled differential–algebraic residual system*, in which both types of equations are treated on equal footing within the global nonlinear system.

Although this approach increases the number of unknowns, it often simplifies the numerical treatment. By incorporating algebraic relations explicitly into the residual system, one avoids the need for repeated internal nonlinear inversions during integration. Instead, all nonlinearities are handled collectively within the Newton iteration applied to the full system. This leads to a more uniform and systematic solution process, where both differential and algebraic components are resolved simultaneously.

From a structural perspective, the inclusion of algebraic variables preserves the key advantages of relaxation methods. The resulting Jacobian matrix retains a sparse, block-structured form, with additional rows and columns corresponding to the algebraic constraints. The locality of coupling is typically maintained, so that efficient linear solution techniques remain applicable.

This interpretation significantly broadens the scope of relaxation methods. Rather than being limited to ordinary differential equations, they extend naturally to problems formulated as general residual systems, including those with mixed differential and algebraic components. In this sense, relaxation can be viewed as a general framework for solving nonlinear systems arising from discretized physical models, where the distinction between differential and algebraic equations is handled seamlessly within a unified numerical approach.

## 18.4.6. Application: Reaction–Diffusion in Porous Catalysts

A representative and practically important application of relaxation methods arises in the modeling of steady reaction–diffusion processes in porous catalysts. These systems are characterized by strong nonlinear coupling, spatial gradients, and sensitivity to parameters, making them challenging for shooting-based approaches but well suited to global discretization techniques.

In spherical symmetry, a typical concentration equation takes the form:

\begin{equation}
\frac{1}{\xi^2}\frac{d}{d\xi}\!\left(\xi^2 D(c,\theta)\frac{dc}{d\xi}\right)
= R(c,\theta), 
\qquad 0 < \xi < 1
\tag{18.4.11}
\end{equation}

where $\xi$ is the dimensionless radial coordinate, $c(\xi)$ is the concentration, $D(c,\theta)$ is a possibly nonlinear diffusivity, and $R(c,\theta)$ represents the reaction term. The dependence on $\theta$ reflects coupling with thermal or other physical variables.

The boundary conditions reflect the physical constraints of the system. At the center of the catalyst particle, symmetry requires:

$$\frac{dc}{d\xi}(0)=0 \tag{18.4.12}$$

which ensures that there is no flux across the center. At the outer surface, a transfer condition models the exchange of mass between the catalyst and the surrounding medium:

$$D(c,\theta)\frac{dc}{d\xi}(1)=k_m\big(c(1)-c_\infty\big) \tag{18.4.13}$$

Here, $k_m$ is a mass transfer coefficient, and $c_\infty$ denotes the external concentration.

Several features of this problem illustrate the challenges inherent in boundary value formulations. First, the differential operator includes a geometric singularity at $\xi=0$ due to the factor $1/\xi^2$, requiring careful numerical treatment. Second, the nonlinear dependence of both diffusion and reaction terms on $c$ and $\theta$ introduces strong coupling and potential stiffness. Third, the boundary conditions are imposed at distinct physical locations, leading naturally to a TPBVP.

In many realistic models, the concentration equation is coupled with an energy equation describing heat generation and transport within the catalyst. This results in a system of nonlinear differential equations with multiple interacting variables. The combined system may exhibit *multiple steady states*, bifurcations, and sharp internal layers, particularly when reaction rates are highly temperature-dependent.

Relaxation methods are particularly effective in this setting for several reasons. By discretizing the entire domain and enforcing the governing equations and boundary conditions simultaneously, they avoid the instability associated with propagating solutions from a singular or sensitive boundary. The global formulation ensures that both the symmetry condition at $\xi=0$ and the transfer condition at $\xi=1$ are satisfied within the same nonlinear system, providing a balanced and stable solution.

Moreover, the ability to incorporate additional variables directly into the discretized system, as discussed in Section 18.4.5, allows coupled concentration–temperature models to be handled naturally. The resulting block-structured Jacobian can still be exploited for efficient computation, even in the presence of strong nonlinearities.

Contemporary numerical studies further enhance this framework by combining relaxation with *orthogonal collocation* and *continuation methods*. Orthogonal collocation provides high-order spatial accuracy by approximating the solution using polynomial bases within each subinterval, while continuation techniques allow systematic exploration of solution branches as parameters vary. This is particularly important in reaction–diffusion systems, where multiple steady states may exist and transitions between regimes are of practical interest (Soares et al., 2024; Krishnakumar et al., 2025).

Thus, reaction–diffusion modeling in porous catalysts exemplifies the strengths of relaxation methods. The combination of global consistency, flexibility in handling nonlinear and coupled equations, and compatibility with advanced discretization and continuation techniques makes relaxation a powerful and reliable approach for solving complex TPBVPs arising in chemical and physical systems.

### Rust Implementation

Following the extension of relaxation methods to coupled and constrained systems in Sections 18.4.5–18.4.6, Program 18.4.2 provides a practical implementation of a differential–algebraic boundary value problem using the relaxation framework. In the theoretical formulation, the governing differential equations are discretized using the midpoint residual approach described earlier in equations (18.4.1)–(18.4.4), while additional algebraic constraints are incorporated directly into the global residual system. This leads to a unified nonlinear system in which both differential and algebraic relations are enforced simultaneously. The present program implements this formulation using Newton iteration and finite-difference Jacobians, demonstrating how relaxation methods naturally accommodate coupled physics and constraints. This example highlights the flexibility of the relaxation framework in handling extended models, including reaction–diffusion systems and differential–algebraic structures, within a single computational paradigm.

At the core of the implementation is the function `residual`, which constructs the global nonlinear system combining both differential and algebraic components. The differential part is formulated using the midpoint discretization introduced in equations (18.4.1)–(18.4.4), where each subinterval contributes a set of residual equations enforcing consistency between neighboring mesh values. In addition to these equations, the algebraic relation is imposed directly at each node, ensuring that the auxiliary variable satisfies the constraint at all points. This unified residual formulation reflects the extension discussed in Section 18.4.5, where differential and algebraic conditions are treated on equal footing within the relaxation framework.

The mapping between structured state variables and the global unknown vector is handled by the functions `pack` and `unpack`. These functions allow the multidimensional state at each mesh point to be assembled into a single vector suitable for Newton iteration, while preserving the interpretation of each component as a physical variable. This abstraction enables the formulation of a large nonlinear system whose dimension reflects both the number of mesh points and the number of variables per node, consistent with the discussion following equation (18.4.11).

The nonlinear system is solved using Newton’s method, implemented in the function `newton_solve`. At each iteration, the residual is evaluated and the Jacobian matrix is approximated using the function `finite_difference_jacobian`. This approach corresponds to the linearization described in equation (18.4.5), where the correction is obtained by solving a linear system involving the Jacobian of the residual. Although finite differences introduce additional computational cost, they provide a general mechanism for constructing the Jacobian without requiring explicit analytical derivatives.

The function `solve_dense` performs the linear solve using Gaussian elimination. While this implementation treats the Jacobian as a dense matrix, the underlying structure of the system is in fact sparse and block-coupled, as discussed in Section 18.4.6. More advanced implementations could exploit this structure to improve computational efficiency, particularly for large-scale problems.

The `main` function demonstrates the complete relaxation procedure. It initializes the system with a simple guess, performs Newton iterations to reduce the residual, and then evaluates the solution at selected mesh points. The output includes both the differential variables and the algebraic variable, as well as a direct check of the algebraic constraint. This verification step confirms that the constraint is satisfied to machine precision, illustrating the effectiveness of the unified residual formulation.

```rust
// Program 18.4.2: Relaxation for a Coupled Differential-Algebraic Reaction-Diffusion Model
//
// Problem statement:
// This program demonstrates how relaxation methods can incorporate algebraic
// variables directly into the global residual system.
//
// The model is a simplified one-dimensional reaction-diffusion system:
//
//     c''(x) = Da * c(x) * theta(x),
//     theta''(x) = -beta * Da * c(x) * theta(x),
//     q(x) = c(x) * theta(x),
//
// where q is an algebraic variable introduced at every mesh point.
//
// Boundary conditions:
//
//     c'(0) = 0,      theta'(0) = 0,
//     c(1) = 1,       theta(1) = 1.
//
// The unknown at each mesh point is
//
//     U_k = [c_k, c'_k, theta_k, theta'_k, q_k]^T.
//
// The differential equations are enforced using midpoint relaxation residuals,
// while the algebraic relation q = c theta is enforced directly at each node.

const VARS: usize = 5;

type State = [f64; VARS];

fn reaction(c: f64, theta: f64, da: f64) -> f64 {
    da * c * theta
}

fn rhs(y: State, da: f64, beta: f64) -> [f64; 4] {
    let c = y[0];
    let cp = y[1];
    let theta = y[2];
    let thetap = y[3];

    let r = reaction(c, theta, da);

    [
        cp,
        r,
        thetap,
        -beta * r,
    ]
}

fn pack(states: &[State]) -> Vec<f64> {
    let mut u = Vec::with_capacity(VARS * states.len());

    for state in states {
        for value in state {
            u.push(*value);
        }
    }

    u
}

fn unpack(u: &[f64]) -> Vec<State> {
    let nodes = u.len() / VARS;
    let mut states = vec![[0.0; VARS]; nodes];

    for k in 0..nodes {
        for j in 0..VARS {
            states[k][j] = u[VARS * k + j];
        }
    }

    states
}

fn initial_guess(m: usize) -> Vec<f64> {
    let mut states = Vec::with_capacity(m + 1);

    for _ in 0..=m {
        let c = 1.0;
        let theta = 1.0;
        let q = c * theta;

        states.push([c, 0.0, theta, 0.0, q]);
    }

    pack(&states)
}

fn residual(u: &[f64], m: usize, da: f64, beta: f64) -> Vec<f64> {
    let states = unpack(u);
    let h = 1.0 / m as f64;

    let mut r = vec![0.0; VARS * (m + 1)];

    // Left boundary conditions:
    // c'(0) = 0, theta'(0) = 0.
    r[0] = states[0][1];
    r[1] = states[0][3];

    let mut row = 2;

    // Midpoint differential residuals on each subinterval.
    for k in 1..=m {
        let y_left = states[k - 1];
        let y_right = states[k];

        let y_mid = [
            0.5 * (y_left[0] + y_right[0]),
            0.5 * (y_left[1] + y_right[1]),
            0.5 * (y_left[2] + y_right[2]),
            0.5 * (y_left[3] + y_right[3]),
            0.5 * (y_left[4] + y_right[4]),
        ];

        let f_mid = rhs(y_mid, da, beta);

        r[row] = y_right[0] - y_left[0] - h * f_mid[0];
        r[row + 1] = y_right[1] - y_left[1] - h * f_mid[1];
        r[row + 2] = y_right[2] - y_left[2] - h * f_mid[2];
        r[row + 3] = y_right[3] - y_left[3] - h * f_mid[3];

        row += 4;
    }

    // Algebraic residuals at every mesh point:
    // q_k - c_k theta_k = 0.
    for k in 0..=m {
        r[row] = states[k][4] - states[k][0] * states[k][2];
        row += 1;
    }

    // Right boundary conditions:
    // c(1) = 1, theta(1) = 1.
    r[row] = states[m][0] - 1.0;
    r[row + 1] = states[m][2] - 1.0;

    r
}

fn finite_difference_jacobian(
    u: &[f64],
    m: usize,
    da: f64,
    beta: f64,
    eps: f64,
) -> Vec<Vec<f64>> {
    let n = u.len();
    let base = residual(u, m, da, beta);
    let mut jac = vec![vec![0.0; n]; n];

    for col in 0..n {
        let mut perturbed = u.to_vec();
        perturbed[col] += eps;

        let rp = residual(&perturbed, m, da, beta);

        for row in 0..n {
            jac[row][col] = (rp[row] - base[row]) / eps;
        }
    }

    jac
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();

    for k in 0..n {
        let mut pivot = k;
        let mut pivot_abs = a[k][k].abs();

        for i in (k + 1)..n {
            if a[i][k].abs() > pivot_abs {
                pivot_abs = a[i][k].abs();
                pivot = i;
            }
        }

        if pivot_abs < 1.0e-14 {
            return None;
        }

        if pivot != k {
            a.swap(k, pivot);
            b.swap(k, pivot);
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
        let mut sum = b[i];

        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }

        x[i] = sum / a[i][i];
    }

    Some(x)
}

fn newton_solve(
    m: usize,
    da: f64,
    beta: f64,
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, Vec<f64>, usize, bool) {
    let mut u = initial_guess(m);

    println!("Newton Iteration History");
    println!("------------------------");
    println!(
        "{:>6} {:>18} {:>18} {:>12}",
        "iter", "||R(U)||_2", "max |R_i|", "alpha"
    );

    for iteration in 0..=max_iterations {
        let r = residual(&u, m, da, beta);
        let r_norm = norm2(&r);
        let r_max = max_abs(&r);

        println!(
            "{:>6} {:>18.10e} {:>18.10e} {:>12}",
            iteration, r_norm, r_max, "-"
        );

        if r_norm < tolerance {
            return (u, r, iteration, true);
        }

        let jac = finite_difference_jacobian(&u, m, da, beta, 1.0e-6);
        let rhs: Vec<f64> = r.iter().map(|value| -value).collect();

        let Some(delta) = solve_dense(jac, rhs) else {
            return (u, r, iteration, false);
        };

        let mut alpha = 1.0;
        let mut accepted = false;

        for _ in 0..12 {
            let trial_u: Vec<f64> = u
                .iter()
                .zip(delta.iter())
                .map(|(ui, di)| ui + alpha * di)
                .collect();

            let trial_r = residual(&trial_u, m, da, beta);

            if norm2(&trial_r) < r_norm {
                u = trial_u;
                accepted = true;

                println!(
                    "{:>6} {:>18} {:>18} {:>12.6}",
                    "", "", "", alpha
                );

                break;
            }

            alpha *= 0.5;
        }

        if !accepted {
            let r = residual(&u, m, da, beta);
            return (u, r, iteration, false);
        }
    }

    let r = residual(&u, m, da, beta);
    (u, r, max_iterations, false)
}

fn main() {
    let m = 20;
    let da = 2.0;
    let beta = 0.4;
    let tolerance = 1.0e-10;
    let max_iterations = 20;

    println!("Relaxation for a Coupled Differential-Algebraic System");
    println!("======================================================");
    println!();
    println!("Model problem:");
    println!("  c''(x)     = Da c(x) theta(x)");
    println!("  theta''(x) = -beta Da c(x) theta(x)");
    println!("  q(x)       = c(x) theta(x)");
    println!();
    println!("Boundary conditions:");
    println!("  c'(0) = 0,       theta'(0) = 0");
    println!("  c(1)  = 1,       theta(1)  = 1");
    println!();
    println!("Parameters:");
    println!("  Da     = {:.6}", da);
    println!("  beta   = {:.6}", beta);
    println!("  M      = {}", m);
    println!("  nodes  = {}", m + 1);
    println!("  variables per node = {}", VARS);
    println!("  global unknowns    = {}", VARS * (m + 1));
    println!();

    let (u, final_residual, iterations, converged) =
        newton_solve(m, da, beta, tolerance, max_iterations);

    let states = unpack(&u);

    println!();
    println!("Relaxation Result");
    println!("-----------------");
    println!("Converged                         = {}", converged);
    println!("Iterations performed              = {}", iterations);
    println!("Final residual norm               = {:.12e}", norm2(&final_residual));
    println!("Final maximum residual            = {:.12e}", max_abs(&final_residual));
    println!();

    println!("Representative Solution Values");
    println!("------------------------------");
    println!(
        "{:>8} {:>12} {:>14} {:>14} {:>14} {:>14}",
        "k", "x_k", "c", "c'", "theta", "q"
    );

    for k in [0, 5, 10, 15, 20] {
        let x = k as f64 / m as f64;
        println!(
            "{:>8} {:>12.6} {:>14.8} {:>14.8} {:>14.8} {:>14.8}",
            k,
            x,
            states[k][0],
            states[k][1],
            states[k][2],
            states[k][4]
        );
    }

    println!();
    println!("Algebraic Constraint Check");
    println!("--------------------------");
    println!(
        "{:>8} {:>12} {:>18}",
        "k", "x_k", "|q - c theta|"
    );

    for k in [0, 5, 10, 15, 20] {
        let x = k as f64 / m as f64;
        let defect = (states[k][4] - states[k][0] * states[k][2]).abs();

        println!(
            "{:>8} {:>12.6} {:>18.10e}",
            k, x, defect
        );
    }
}
```

Program 18.4.2 demonstrates how relaxation methods can be extended to coupled differential–algebraic systems by incorporating algebraic constraints directly into the residual formulation. This approach reflects the central idea developed in Sections 18.4.5–18.4.6: treating all governing equations, whether differential or algebraic, within a single nonlinear system.

The results show that the residual is reduced to machine precision, indicating that both the differential equations and the algebraic constraints are satisfied simultaneously. The algebraic constraint is enforced exactly at all mesh points, up to roundoff error, demonstrating the robustness of the method.

An important advantage of this formulation is its flexibility. By augmenting the state vector and residual system, additional physical constraints can be incorporated without fundamentally altering the solution strategy. This makes relaxation methods particularly well suited for complex multiphysics problems, including reaction–diffusion systems, constrained dynamics, and differential–algebraic equations.

The modular structure of the implementation also provides a foundation for further extensions. In particular, exploiting the block structure of the Jacobian, introducing adaptive mesh refinement, or incorporating more advanced nonlinear solvers can significantly enhance performance and scalability. Overall, the program illustrates how the relaxation framework provides a unified and powerful approach for solving complex boundary value problems involving both differential and algebraic relations.

## 18.4.7. Modern Developments and Extensions of Relaxation-Based Methods

Modern relaxation methods have evolved significantly beyond their classical finite-difference origins and are now best understood as part of a broader class of *sparse residual-based nonlinear solvers*. In this framework, the boundary value problem is viewed as a structured nonlinear system, and advances in numerical linear algebra, discretization, and differentiation techniques are leveraged to improve efficiency, scalability, and robustness.

A central development is the use of *adaptive mesh refinement*. Rather than employing a fixed discretization, modern methods adjust the mesh dynamically based on error indicators derived from the current solution. These indicators typically measure local residuals, gradient variations, or estimated truncation errors. By refining the mesh in regions where the solution exhibits rapid variation and coarsening it where the solution is smooth, adaptive strategies achieve higher accuracy with fewer degrees of freedom. This leads to improved computational efficiency while maintaining the global consistency inherent in relaxation methods (Haman and Rao, 2025).

Another important direction involves *hp-adaptive schemes*, in which both the mesh spacing (h-refinement) and the local approximation order (p-refinement) are varied. Approaches based on the Theory of Functional Connections provide a systematic way to embed boundary conditions analytically into the solution representation, allowing high-order approximations to be constructed efficiently. These methods have demonstrated improved accuracy for hypersensitive TPBVPs, where classical discretizations may require extremely fine meshes. At the same time, they highlight a practical challenge: the computational cost associated with constructing and evaluating Jacobians, particularly when numerical differentiation is used (Drozd et al., 2024).

To address the cost of Jacobian evaluation, *sparse and vectorized automatic differentiation* techniques have become increasingly important. These methods exploit the structured sparsity of the discretized system to compute derivatives efficiently, avoiding the overhead of dense finite-difference approximations. In large-scale problems, such as those arising in optimal control discretizations with many collocation nodes, automatic differentiation scales effectively with problem size and provides accurate Jacobian information essential for Newton-type methods (Zou and Jiang, 2026).

In addition, *Jacobian-free Newton–Krylov (JFNK) methods* offer an alternative approach when explicit formation of the Jacobian becomes prohibitively expensive. In these methods, the action of the Jacobian on a vector is approximated implicitly, typically through directional finite differences of the residual. The resulting linear systems are solved using Krylov subspace methods, which require only matrix-vector products rather than explicit matrix storage. This significantly reduces memory requirements and can improve computational efficiency for very large systems. However, the success of JFNK methods depends critically on the availability of effective preconditioning strategies, as poor conditioning can severely degrade convergence (Cardiff et al., 2026).

Taken together, these developments demonstrate that relaxation methods have evolved into a flexible and powerful computational paradigm. They are no longer limited to simple finite-difference discretizations, but instead form part of a modern framework for solving large-scale, nonlinear residual systems. By integrating adaptive discretization, advanced differentiation techniques, and scalable linear solvers, contemporary relaxation-based methods are capable of addressing increasingly complex boundary value problems arising in science and engineering.

## 18.4.8. Practical Perspective and Implementation Strategy for Relaxation Methods

From a practical standpoint, relaxation methods provide a robust and versatile framework for solving two-point boundary value problems, particularly in situations where shooting-based approaches encounter fundamental difficulties. Their global formulation makes them especially well suited for problems characterized by instability, singular behavior, or strong nonlinear coupling between variables.

In systems with unstable modes, relaxation avoids the exponential amplification of perturbations inherent in propagation-based methods by enforcing consistency across the entire domain. Similarly, when singularities occur at endpoints, the global discretization distributes their influence across the mesh, preventing localized breakdown of the numerical procedure. For problems involving strong nonlinear coupling, such as those arising in reaction–diffusion systems or optimal control, the simultaneous solution of all equations ensures that interdependencies are resolved coherently rather than iteratively through repeated integrations.

An additional advantage arises in problems exhibiting *multiple solution branches*. In such cases, continuation methods are often employed to trace solutions as parameters vary. Relaxation methods integrate naturally with continuation strategies, since the solution is represented explicitly on the mesh and can be smoothly updated as parameters change. This makes it possible to compute bifurcation diagrams and explore multiple steady states in a systematic manner.

A practical implementation strategy typically proceeds in a staged and incremental manner, reflecting the flexibility of the relaxation framework:

- **Initial formulation:**\
  Begin with a basic mesh discretization and construct the residual system using a standard scheme such as midpoint or collocation discretization. Apply Newton’s method to solve the resulting nonlinear system, using straightforward linear solvers.
- **Exploitation of structure:**\
  Incorporate structured linear algebra techniques, such as block-tridiagonal solvers, to take advantage of the sparsity pattern of the Jacobian. This significantly reduces computational cost and memory usage.
- **Adaptive refinement:**\
  Introduce adaptive mesh refinement based on residual or error indicators. This step improves accuracy by concentrating computational effort in regions where the solution exhibits sharp gradients or nonlinear features.
- **Advanced differentiation:**\
  Replace finite-difference Jacobian approximations with automatic differentiation or sensitivity-based methods to improve accuracy and efficiency, particularly for large-scale systems.
- **Scalable solvers:**\
  For very large problems, adopt Jacobian-free Newton–Krylov methods or other scalable linear solvers, combined with appropriate preconditioning strategies.

This incremental approach allows the method to evolve naturally from a simple prototype to a highly efficient and scalable solver. At each stage, improvements are introduced only as required by the complexity of the problem, preserving both clarity and flexibility in the implementation.

In summary, relaxation methods offer a comprehensive and extensible framework for solving TPBVPs. Their ability to handle instability, singularities, nonlinear coupling, and multiple solution branches, combined with their compatibility with modern numerical techniques, makes them a powerful tool for tackling increasingly complex problems in scientific and engineering applications.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/85PjoXOwqbp4XFM8XVOp.4","tags":[]}

# 18.5. A Worked Example: Spheroidal Harmonics

Spheroidal harmonics provide a classical and instructive example of a two-point boundary value problem in which many of the challenges discussed earlier arise simultaneously. The problem combines singular behavior at the endpoints, an eigenvalue structure, and sensitivity to parameters, making it an ideal setting for illustrating both analytical structure and numerical methodology.

## 18.5.1. Governing Equation and Problem Structure of Spheroidal Harmonics

Spheroidal harmonics arise naturally in the separation of variables for partial differential equations expressed in spheroidal coordinate systems. A prominent example is the separation of the Helmholtz equation in oblate or prolate geometries, where the angular component satisfies a second-order ordinary differential equation known as the *spheroidal wave equation*.

Introducing the variable transformation,

$$x = \cos\theta, \qquad -1 \le x \le 1$$

the governing equation takes the form:

\begin{equation}
\frac{d}{dx}\!\Big[(1-x^2)\frac{dS}{dx}\Big]
+ \Big(\lambda - c^2 x^2 - \frac{m^2}{1-x^2}\Big) S = 0
\tag{18.5.1}
\end{equation}

In this equation, $S(x)$ denotes the angular function, $m$ is an integer representing the azimuthal order, $c$ is a real parameter associated with the geometry or frequency of the system, and $\lambda$ is the separation constant. The parameter $\lambda$ plays the role of an eigenvalue, and its admissible values are determined as part of the solution process.

Several structural features of equation (18.5.1) are noteworthy. First, the coefficient $(1-x^2)$ multiplying the derivative vanishes at the endpoints $x=\pm1$, introducing *regular singular points* into the differential equation. Second, the term $\frac{m^2}{1-x^2}$ becomes singular as $x \to \pm1$, further emphasizing the special behavior near the boundaries. Despite these singularities, the equation remains well-posed in the sense that physically meaningful solutions are those that remain finite throughout the interval.

The boundary conditions are therefore not prescribed in the usual explicit form but are instead imposed through *regularity requirements*. Specifically, the solution $S(x)$ must remain finite as $x \to \pm1$. These conditions restrict the allowable solutions of the differential equation and, crucially, determine the admissible values of the parameter $\lambda$.

As a consequence, the problem is not merely a boundary value problem but an *eigenvalue boundary value problem*. Only for discrete values of $\lambda$ does the equation admit solutions that satisfy the regularity conditions at both endpoints. Each such value of $\lambda$ corresponds to a distinct eigenfunction $S(x)$, and the resulting set of eigenpairs forms the system of spheroidal harmonics.

From a numerical perspective, this structure introduces additional complexity. The solution process must simultaneously determine the function $S(x)$ and the parameter $\lambda$, while ensuring that the solution remains finite at the singular endpoints. This leads to a coupled nonlinear problem in which both the function and the eigenvalue must be adjusted to satisfy the global conditions.

Thus, the spheroidal harmonic equation exemplifies a class of TPBVPs characterized by singular endpoints, parameter-dependent coefficients, and an eigenvalue structure. It provides a rich setting for applying the methods developed in this chapter, particularly relaxation and shooting techniques adapted to handle eigenvalue constraints and singular behavior.

## 18.5.2. Sturm–Liouville Structure and Regularization of Endpoint Singularities

Expanding the derivative term in equation (18.5.1) yields the explicit second-order form:

\begin{equation}
(1 - x^2)\,S'' - 2x\,S' 
+ \Big(\lambda - c^2 x^2 - \frac{m^2}{1 - x^2}\Big) S = 0
\tag{18.5.2}
\end{equation}

This representation makes the analytical structure of the equation more transparent. In particular, it reveals that the coefficients of the differential equation become singular at the endpoints $x=\pm1$, due to the term $\frac{1}{1-x^2}$. These are *regular singular points*, meaning that although the coefficients diverge, the equation still admits well-behaved solutions under appropriate conditions. However, from a numerical perspective, these singularities must be handled carefully to avoid instability or loss of accuracy.

The equation can be viewed as a special case of a Sturm–Liouville problem with a singular weight and coefficient structure. In such problems, admissible solutions are selected not by explicit boundary values but by *regularity requirements* at the singular endpoints. Specifically, only those solutions that remain finite as $x \to \pm1$ are physically meaningful. This requirement imposes implicit constraints on both the solution and the eigenvalue $\lambda$.

A standard analytical technique for handling such singular behavior is to factor out the dominant singularity using a Frobenius-type substitution. The idea is to separate the known asymptotic behavior near the endpoints from the remaining, regular part of the solution. To this end, introduce the transformation:

$$S(x) = (1-x^2)^{|m|/2}\,T(x) \tag{18.5.3}$$

where the factor $(1-x^2)^{|m|/2}$ captures the leading-order behavior of the solution near $x=\pm1$, and the function $T(x)$ is expected to remain finite and smooth throughout the interval.

Substituting this expression into equation (18.5.2) yields a modified differential equation for $T(x)$. In this transformed equation, the most severe singular terms arising from $\frac{m^2}{1-x^2}$ are canceled by the derivative contributions of the prefactor. As a result, the dominant singularities are removed, and the new equation exhibits significantly improved regularity properties near the endpoints.

From a numerical standpoint, this regularization has important consequences. By working with the transformed variable $T(x)$, one avoids the need to resolve sharply varying or singular behavior directly. The solution becomes smoother, and standard discretization methods, such as relaxation or collocation, can be applied more effectively. In particular, the boundary conditions can now be imposed in a more conventional manner, since $T(x)$ remains finite at $x=\pm1$.

In practical computations, two complementary approaches are commonly used. One may explicitly adopt the transformed formulation (18.5.3) and solve the resulting regularized equation for $T(x)$. Alternatively, one may retain the original formulation and enforce the regularity condition implicitly by selecting boundary or initial conditions that ensure finiteness of $S(x)$ at the endpoints. Both approaches are consistent with the underlying Sturm–Liouville structure, but the transformed formulation often provides improved numerical stability and accuracy.

Thus, the Frobenius-type regularization serves as a crucial step in the numerical treatment of spheroidal harmonics, converting a singular boundary value problem into a form that is more amenable to analysis and computation while preserving the essential spectral properties of the system.

## 18.5.3. Shooting Method Formulation for the Spheroidal Eigenvalue Problem

The spheroidal wave equation provides a natural setting for applying the shooting method to an *eigenvalue boundary value problem*. In this formulation, the unknown separation constant $\lambda$ is treated as an additional parameter to be determined so that the solution satisfies the regularity conditions at both endpoints.

To apply shooting, the second-order equation (18.5.1) is first converted into a first-order system. A convenient choice of variables is:

$$y_1(x) = S(x), \qquad y_2(x) = (1-x^2)\,S'(x) \tag{18.5.4}$$

which incorporates the degeneracy of the coefficient $(1-x^2)$ directly into the state variable. This transformation is advantageous because it avoids explicit division by small quantities near the endpoints when forming the second equation.

Differentiating $y_1$ gives:

$$y_1' = \frac{y_2}{1-x^2} \tag{18.5.5}$$

and differentiating $y_2$, followed by substitution from the original equation, yields a corresponding expression for $y_2'$. Together, these relations define a first-order system of the form:

$$\mathbf y' = \mathbf F(x,\mathbf y;\lambda) \tag{18.5.6}$$

in which the eigenvalue $\lambda$ appears as a parameter.

The shooting procedure proceeds by selecting a trial value of $\lambda$, solving the resulting initial value problem, and assessing whether the computed solution satisfies the required behavior at the opposite endpoint. Because the equation has regular singular points at $x=\pm1$, integration cannot be started exactly at these points. Instead, one begins at a nearby point $x=-1+\varepsilon$, where $\varepsilon>0$ is small.

The initial conditions must be chosen to reflect the regularity requirement at $x=-1$. A typical choice is:

$$y_1(-1+\varepsilon)=0, \qquad y_2(-1+\varepsilon)=c_0 \tag{18.5.7}$$

where $c_0$ is a free scaling parameter. Since the equation is linear in $S(x)$, the overall amplitude is arbitrary, and only the relative behavior of the solution is relevant. Thus, $c_0$ can be fixed (for example, $c_0=1$) without loss of generality.

With these initial conditions, the system is integrated numerically from $x=-1+\varepsilon$ to $x=1-\varepsilon$. For a general value of $\lambda$, the solution will not satisfy the regularity condition at $x=1$; instead, it may diverge or exhibit nonphysical behavior. The requirement that $S(x)$ remain finite at $x=1$ defines a *nonlinear scalar condition* on $\lambda$. This condition can be expressed as a residual function, whose root corresponds to an admissible eigenvalue.

In practice, the eigenvalue is determined by applying a root-finding method, such as Newton or secant iteration, to this residual. Each evaluation of the residual requires a full integration of the initial value problem, making the method computationally efficient in memory but potentially sensitive to instability and endpoint behavior.

An alternative and often more stable formulation employs *double shooting*. In this approach, one integrates a solution from $x=-1+\varepsilon$ forward and another from $x=1-\varepsilon$ backward, each satisfying the appropriate regularity condition at its respective endpoint. The two solutions are then matched at an interior point, typically by requiring that their Wronskian vanish. This condition ensures linear dependence of the two solutions and thus enforces global regularity. The resulting formulation reduces sensitivity to endpoint singularities and improves numerical conditioning.

Overall, the shooting method reduces the eigenvalue boundary value problem to a sequence of initial value integrations combined with a scalar root-finding procedure for $\lambda$. Its principal advantages are conceptual simplicity, low memory requirements, and direct use of high-accuracy ODE solvers. However, careful handling of singular endpoints and sensitivity to instability are essential for reliable performance in this class of problems.

### Rust Implementation

Following the formulation of the spheroidal harmonic boundary value problem in Section 18.5.3, Program 18.5.1 provides a practical implementation of the shooting method for computing eigenvalues associated with the differential equation. In this approach, the second-order equation is first rewritten in the first-order form introduced in equations (18.5.4)–(18.5.6), and the singular endpoints at $x = \pm 1$ are avoided by initiating the integration slightly inside the domain, as described in equation (18.5.7). The eigenvalue $\lambda$ is treated as an unknown parameter and is determined by enforcing a regularity condition at the terminal boundary. From a numerical perspective, this converts the boundary value problem into a scalar root-finding problem, which is solved here using a secant iteration. The program demonstrates how the shooting formulation can be adapted to eigenvalue problems and illustrates the interplay between differential equation integration and parameter estimation.

At the core of the implementation is the function `rhs`, which defines the first-order system corresponding to the transformation introduced in equations (18.5.4)–(18.5.6). By expressing the second-order equation in terms of the variables $y_1 = S$ and $y_2 = (1 - x^2)S'$, the system becomes suitable for numerical integration while avoiding explicit singular behavior at the endpoints. This formulation allows the shooting method to operate on a well-defined initial value problem over a truncated domain.

The numerical integration is performed by the function `rk4_step`, which implements the classical fourth-order Runge–Kutta method. This function advances the solution across the interval using a fixed step size and ensures sufficient accuracy for the eigenvalue computation. The function `integrate` applies this step repeatedly, starting from a point near $x = -1$ and proceeding toward $x = 1$, consistent with the regularization strategy described around equation (18.5.7). The initial conditions are chosen to reflect the regular behavior of the solution at the endpoint.

The function `shooting_residual` evaluates the boundary condition at the terminal point. In accordance with the regularity condition discussed in the text, the residual is defined in terms of the flux variable $y_2$, which must vanish for a physically admissible solution. This scalar residual serves as the objective function in the eigenvalue search and represents the deviation from the required boundary condition.

The eigenvalue is determined using the function `secant_eigenvalue_search`, which implements the secant iteration. Starting from two initial guesses, the method updates the eigenvalue estimate based on the residual values, progressively refining the solution until the residual is sufficiently small. This procedure corresponds to solving the nonlinear equation defined implicitly by the shooting residual and reflects the root-finding perspective discussed in Section 18.5.3.

The `main` function orchestrates the complete computation. It defines the numerical parameters, performs the secant iteration, and evaluates the resulting solution. The computed eigenvalue is compared with the known analytical value in the Legendre limit, and representative solution values are printed for validation. This comparison demonstrates that the numerical solution closely approximates the expected eigenfunction, confirming the correctness of the implementation.

```rust
// Program 18.5.1: Shooting Formulation for a Spheroidal Eigenvalue Problem
//
// Problem statement:
// This program demonstrates a shooting approach for the spheroidal harmonic
// eigenvalue problem in the special case m = 0.
//
// The equation is
//
//     d/dx[(1 - x^2) dS/dx] + (lambda - c^2 x^2) S = 0,
//     -1 < x < 1.
//
// For m = 0 and c = 0, the equation reduces to the Legendre equation, whose
// eigenvalues are lambda = l(l + 1). This program targets the l = 2 mode,
// whose exact eigenvalue is lambda = 6.
//
// The first-order variables are
//
//     y1 = S,
//     y2 = (1 - x^2) S'.
//
// Integration starts near x = -1 and ends near x = 1 to avoid evaluating
// the singular endpoint coefficients directly. A secant iteration adjusts
// lambda until the endpoint regularity residual is driven to zero.

fn exact_lambda_l2() -> f64 {
    6.0
}

#[derive(Clone, Copy)]
struct State {
    y1: f64, // S
    y2: f64, // (1 - x^2) S'
}

fn rhs(x: f64, state: State, lambda: f64, c_param: f64) -> State {
    let denom = 1.0 - x * x;

    State {
        y1: state.y2 / denom,
        y2: -(lambda - c_param * c_param * x * x) * state.y1,
    }
}

fn add_scaled(a: State, b: State, scale: f64) -> State {
    State {
        y1: a.y1 + scale * b.y1,
        y2: a.y2 + scale * b.y2,
    }
}

fn rk4_step(x: f64, state: State, h: f64, lambda: f64, c_param: f64) -> State {
    let k1 = rhs(x, state, lambda, c_param);

    let s2 = add_scaled(state, k1, 0.5 * h);
    let k2 = rhs(x + 0.5 * h, s2, lambda, c_param);

    let s3 = add_scaled(state, k2, 0.5 * h);
    let k3 = rhs(x + 0.5 * h, s3, lambda, c_param);

    let s4 = add_scaled(state, k3, h);
    let k4 = rhs(x + h, s4, lambda, c_param);

    State {
        y1: state.y1 + h * (k1.y1 + 2.0 * k2.y1 + 2.0 * k3.y1 + k4.y1) / 6.0,
        y2: state.y2 + h * (k1.y2 + 2.0 * k2.y2 + 2.0 * k3.y2 + k4.y2) / 6.0,
    }
}

fn integrate(lambda: f64, c_param: f64, eps: f64, steps: usize) -> State {
    let x0 = -1.0 + eps;
    let x1 = 1.0 - eps;
    let h = (x1 - x0) / steps as f64;

    // For the even l = 2 mode, P_2(-1) = 1 and
    // (1 - x^2)S' tends to zero at the regular endpoint.
    let mut state = State { y1: 1.0, y2: 0.0 };
    let mut x = x0;

    for _ in 0..steps {
        state = rk4_step(x, state, h, lambda, c_param);
        x += h;
    }

    state
}

// For a regular solution at x = 1 and m = 0,
// y2 = (1 - x^2)S' should tend to zero.
// Hence the shooting residual is y2(1 - eps).
fn shooting_residual(lambda: f64, c_param: f64, eps: f64, steps: usize) -> f64 {
    let terminal = integrate(lambda, c_param, eps, steps);
    terminal.y2
}

struct SecantResult {
    lambda: f64,
    residual: f64,
    iterations: usize,
    converged: bool,
}

fn secant_eigenvalue_search(
    lambda0: f64,
    lambda1: f64,
    c_param: f64,
    eps: f64,
    steps: usize,
    tolerance: f64,
    max_iterations: usize,
) -> SecantResult {
    let mut previous_lambda = lambda0;
    let mut current_lambda = lambda1;

    let mut previous_residual = shooting_residual(previous_lambda, c_param, eps, steps);
    let mut current_residual = shooting_residual(current_lambda, c_param, eps, steps);

    println!("Secant Iteration History");
    println!("------------------------");
    println!(
        "{:>6} {:>18} {:>18}",
        "iter", "lambda", "residual"
    );
    println!(
        "{:>6} {:>18.10} {:>18.10e}",
        0, previous_lambda, previous_residual
    );
    println!(
        "{:>6} {:>18.10} {:>18.10e}",
        1, current_lambda, current_residual
    );

    for iteration in 2..=max_iterations {
        let denominator = current_residual - previous_residual;

        if denominator.abs() < 1.0e-14 {
            return SecantResult {
                lambda: current_lambda,
                residual: current_residual,
                iterations: iteration - 1,
                converged: false,
            };
        }

        let next_lambda = current_lambda
            - current_residual * (current_lambda - previous_lambda) / denominator;

        let next_residual = shooting_residual(next_lambda, c_param, eps, steps);

        println!(
            "{:>6} {:>18.10} {:>18.10e}",
            iteration, next_lambda, next_residual
        );

        if next_residual.abs() < tolerance {
            return SecantResult {
                lambda: next_lambda,
                residual: next_residual,
                iterations: iteration,
                converged: true,
            };
        }

        previous_lambda = current_lambda;
        previous_residual = current_residual;

        current_lambda = next_lambda;
        current_residual = next_residual;
    }

    SecantResult {
        lambda: current_lambda,
        residual: current_residual,
        iterations: max_iterations,
        converged: false,
    }
}

fn legendre_p2(x: f64) -> f64 {
    0.5 * (3.0 * x * x - 1.0)
}

fn main() {
    let c_param = 0.0;
    let eps = 1.0e-4;
    let steps = 20_000;
    let tolerance = 1.0e-10;
    let max_iterations = 30;

    let lambda0 = 5.5;
    let lambda1 = 6.5;

    println!("Shooting Method for a Spheroidal Eigenvalue Problem");
    println!("===================================================");
    println!();
    println!("Special case:");
    println!("  m = 0, c = 0");
    println!("  Equation reduces to the Legendre eigenvalue problem");
    println!("  Target mode: l = 2");
    println!("  Exact eigenvalue: lambda = l(l + 1) = 6");
    println!();
    println!("Numerical setup:");
    println!("  left endpoint  = -1 + eps");
    println!("  right endpoint =  1 - eps");
    println!("  eps            = {:.2e}", eps);
    println!("  RK4 steps      = {}", steps);
    println!();

    let result = secant_eigenvalue_search(
        lambda0,
        lambda1,
        c_param,
        eps,
        steps,
        tolerance,
        max_iterations,
    );

    let terminal = integrate(result.lambda, c_param, eps, steps);

    println!();
    println!("Eigenvalue Shooting Result");
    println!("--------------------------");
    println!("Converged                         = {}", result.converged);
    println!("Iterations performed              = {}", result.iterations);
    println!("Computed lambda                   = {:.12}", result.lambda);
    println!("Exact lambda                      = {:.12}", exact_lambda_l2());
    println!(
        "Absolute eigenvalue error         = {:.12e}",
        (result.lambda - exact_lambda_l2()).abs()
    );
    println!("Final shooting residual           = {:.12e}", result.residual);
    println!("Terminal S(1 - eps)               = {:.12}", terminal.y1);
    println!("Terminal flux y2(1 - eps)         = {:.12e}", terminal.y2);
    println!();

    println!("Representative Solution Values");
    println!("------------------------------");
    println!(
        "{:>10} {:>18} {:>18}",
        "x", "S_num scaled", "P_2(x)"
    );

    let sample_points = [-0.75, -0.50, 0.0, 0.50, 0.75];

    // The shooting solution is normalized by S(-1 + eps) = 1.
    // This is consistent with P_2(-1) = 1.
    for x_target in sample_points {
        let x0 = -1.0 + eps;
        let h = (x_target - x0) / steps as f64;

        let mut x = x0;
        let mut state = State { y1: 1.0, y2: 0.0 };

        for _ in 0..steps {
            state = rk4_step(x, state, h, result.lambda, c_param);
            x += h;
        }

        println!(
            "{:>10.4} {:>18.10} {:>18.10}",
            x_target,
            state.y1,
            legendre_p2(x_target)
        );
    }
}
```

Program 18.5.1 demonstrates the application of the shooting method to eigenvalue problems arising from boundary value formulations. By combining numerical integration with a root-finding procedure, the method transforms the original problem into a sequence of initial value solves and scalar updates.

The results illustrate that the shooting residual can be driven to near machine precision, indicating that the boundary condition is satisfied accurately. The computed eigenvalue is close to the analytical value, with the remaining discrepancy attributable to the truncation of the domain and the numerical integration error. The corresponding solution profile also matches the expected eigenfunction, confirming the validity of the approach.

An important feature of this method is its conceptual simplicity. The eigenvalue problem is reduced to a one-dimensional root-finding problem, making the implementation straightforward and intuitive. However, the sensitivity of the solution near singular endpoints and the dependence on accurate integration highlight potential limitations, particularly for higher modes or more complex parameter regimes.

The modular design of the implementation allows for straightforward extensions, including higher-order integrators, adaptive step-size control, and more sophisticated root-finding algorithms. These enhancements can improve both accuracy and robustness, making the shooting method a powerful tool for solving eigenvalue problems in practical applications. Overall, the program provides a clear computational realization of the theoretical framework developed in Section 18.5.3 and establishes a foundation for more advanced numerical treatments of spheroidal and related eigenvalue problems.

## 18.5.4. Relaxation and Finite-Difference Formulation for the Spheroidal Eigenvalue Problem

An alternative and often more robust approach to solving the spheroidal wave equation is to discretize the problem directly using a finite-difference relaxation framework. In contrast to shooting, which relies on repeated integrations, this approach transforms the differential equation into a structured algebraic eigenvalue problem defined over the entire domain.

To construct the discretization, introduce a uniform mesh on the interval $[-1,1]$:

$$x_j = -1 + jh, \qquad j=0,\dots,N, \qquad h=\frac{2}{N} \tag{18.5.8}$$

The unknown function $S(x)$ is approximated at the mesh points by values $S_j \approx S(x_j)$.

Using centered finite differences to approximate the derivatives in equation (18.5.2), one obtains a second-order discretization that leads to a linear system of equations of the form:

$$c_j S_{j-1} + d_j(\lambda) S_j + e_j S_{j+1} = 0 \tag{18.5.9}$$

valid for interior nodes $j=1,\dots,N-1$.

The coefficients in this system reflect both the differential operator and the singular structure of the equation. They are given by:

$$c_j = \frac{1}{h^2}(1-x_{j-\frac12}^2), \qquad e_j = \frac{1}{h^2}(1-x_{j+\frac12}^2), \tag{18.5.10}$$

where the half-grid points $x_{j\pm\frac12}$ arise naturally from the centered discretization. The diagonal coefficient depends explicitly on the eigenvalue parameter:

\begin{equation}
d_j(\lambda)
=
-\left(
\frac{1}{h^2}\big[(1 - x_{j+\frac{1}{2}}^2) + (1 - x_{j-\frac{1}{2}}^2)\big]
+ \frac{m^2}{1 - x_j^2}
\right)
+ \lambda - c^2 x_j^2
\tag{18.5.11}
\end{equation}

These expressions show how the discrete operator captures both the regular part of the differential equation and the singular behavior near the endpoints. In particular, the term $\frac{m^2}{1-x_j^2}$ reflects the same singular structure present in the continuous equation and must be handled carefully in computations.

Collecting all interior equations, the system can be written compactly in matrix form as:

$$\mathbf A(\lambda)\,\mathbf S = \mathbf 0 \tag{18.5.12}$$

where,

$$\mathbf S = [S_1,\dots,S_{N-1}]^T$$

is the vector of unknown values at interior nodes, and $\mathbf A(\lambda)$ is a tridiagonal matrix whose entries depend on $\lambda$.

This formulation reveals the problem as a *parameter-dependent eigenvalue problem*. Nontrivial solutions exist only when the matrix (\\mathbf A(\\lambda)) is singular, that is, when:

$$\det \mathbf A(\lambda) = 0 \tag{18.5.13}$$

From a numerical perspective, this condition defines the discrete analogue of the continuous spectral problem. Several computational strategies can be employed:

- One may treat $\lambda$ as unknown and apply a root-finding procedure to approximate values for which $\mathbf A(\lambda)$ becomes singular.
- Alternatively, one may fix $\lambda$, solve the resulting linear system subject to a normalization condition (such as fixing the norm of $\mathbf S)$, and evaluate a consistency measure to guide iteration.
- In practice, specialized eigenvalue algorithms for tridiagonal matrices may also be used, exploiting the structure of $\mathbf A(\lambda)$.

A key advantage of this relaxation-based formulation is that it reduces the problem to structured linear algebra. The tridiagonal nature of $\mathbf A(\lambda)$ allows efficient solution using $O(N)$ storage and $O(N)$ or $O(N^2)$ computational cost, depending on the method employed. This is significantly more efficient than dense formulations and enables high-resolution discretizations.

Moreover, the global nature of the method provides improved numerical stability. Since the solution is determined simultaneously across the entire domain, the method avoids the amplification of errors associated with long integrations in shooting. This makes it particularly effective for problems with large eigenvalues, strong parameter dependence, or singular coefficients.

Thus, the finite-difference relaxation formulation offers a powerful alternative to shooting for the spheroidal eigenvalue problem, combining the advantages of global consistency, structured sparsity, and efficient linear algebra to achieve accurate and stable numerical solutions.

### Rust Implementation

Following the development of the finite-difference relaxation formulation in Section 18.5.4, Program 18.5.2 provides a practical implementation of the spheroidal eigenvalue problem using a structured discretization approach. In contrast to the shooting formulation of Section 18.5.3, the present method discretizes the differential operator directly on a uniform mesh, as introduced in equation (18.5.8), and constructs a tridiagonal algebraic system based on the coefficients defined in equations (18.5.9)–(18.5.11). This leads to a parameter-dependent matrix problem of the form (18.5.12), whose eigenvalues approximate those of the continuous problem. From a numerical perspective, this approach transforms the boundary value problem into a structured eigenvalue computation, enabling efficient and stable solution using linear algebra techniques. The program demonstrates this formulation in the Legendre limit, illustrating how the discrete operator captures the spectral structure of the continuous problem.

At the core of the implementation is the function `build_symmetric_operator`, which constructs the discrete operator corresponding to the finite-difference formulation described in equations (18.5.8)–(18.5.11). This function assembles the matrix representation of the differential operator by combining contributions from the discretized flux term and the potential term. The coefficients are evaluated at mesh points and half-grid locations, ensuring that the discretization accurately reflects the structure of the continuous equation, including the behavior near the endpoints.

The transformation of the generalized eigenvalue problem into a standard symmetric form is achieved within this function by incorporating the mesh weights into the matrix. This corresponds to rewriting the system in a form suitable for numerical eigenvalue computation, consistent with the matrix formulation introduced in equation (18.5.12). The resulting matrix retains the essential structure of the original operator while enabling the use of standard numerical techniques.

The eigenvalues are computed using the function `jacobi_eigenvalues`, which implements the classical Jacobi rotation method. This iterative algorithm diagonalizes the symmetric matrix by successively eliminating off-diagonal entries through orthogonal transformations. Although not the most efficient method for large systems, it provides a clear and robust approach for computing eigenvalues in a pedagogical setting and is well suited to demonstrating the spectral properties of the discretized operator.

The auxiliary function `max_offdiag` is used to monitor convergence of the Jacobi iteration by measuring the magnitude of off-diagonal entries. This serves as a stopping criterion, ensuring that the matrix has been sufficiently diagonalized before extracting the eigenvalues.

The `main` function demonstrates the complete finite-difference relaxation procedure. It constructs the discrete operator for a given mesh resolution, computes its eigenvalues, and compares the lowest modes with the exact Legendre eigenvalues. The output illustrates how the numerical eigenvalues converge toward the analytical values as the discretization resolves the underlying structure of the differential operator.

```rust
// Program 18.5.2: Finite-Difference Relaxation for the Spheroidal Eigenvalue Problem
//
// Corrected version:
// For m = 0 and c = 0, the spheroidal equation reduces to the Legendre problem.
// Regularity at x = ±1 is represented by the natural zero-flux condition
// because p(x) = 1 - x^2 vanishes at the endpoints.
//
// The discrete eigenproblem is assembled as
//
//     K S = lambda W S,
//
// where K is the finite-volume stiffness matrix and W is a diagonal mass matrix.
// It is converted to the symmetric standard form
//
//     W^{-1/2} K W^{-1/2} z = lambda z.

fn exact_legendre_lambda(l: usize) -> f64 {
    (l * (l + 1)) as f64
}

fn build_symmetric_operator(n_intervals: usize, m_order: i32, c_param: f64) -> Vec<Vec<f64>> {
    let nodes = n_intervals + 1;
    let h = 2.0 / n_intervals as f64;

    let mut k_mat = vec![vec![0.0; nodes]; nodes];
    let mut weights = vec![h; nodes];

    weights[0] = 0.5 * h;
    weights[n_intervals] = 0.5 * h;

    for face in 0..n_intervals {
        let x_face = -1.0 + (face as f64 + 0.5) * h;
        let p_face = 1.0 - x_face * x_face;
        let coeff = p_face / h;

        k_mat[face][face] += coeff;
        k_mat[face][face + 1] -= coeff;
        k_mat[face + 1][face] -= coeff;
        k_mat[face + 1][face + 1] += coeff;
    }

    for j in 0..nodes {
        let x = -1.0 + j as f64 * h;

        let singular = if m_order == 0 {
            0.0
        } else {
            let denom = (1.0 - x * x).max(1.0e-12);
            (m_order * m_order) as f64 / denom
        };

        let potential = singular + c_param * c_param * x * x;
        k_mat[j][j] += weights[j] * potential;
    }

    let mut a = vec![vec![0.0; nodes]; nodes];

    for i in 0..nodes {
        for j in 0..nodes {
            a[i][j] = k_mat[i][j] / (weights[i].sqrt() * weights[j].sqrt());
        }
    }

    a
}

fn max_offdiag(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut max_value = 0.0_f64;

    for i in 0..n {
        for j in 0..n {
            if i != j {
                max_value = max_value.max(a[i][j].abs());
            }
        }
    }

    max_value
}

fn jacobi_eigenvalues(mut a: Vec<Vec<f64>>, tolerance: f64, max_sweeps: usize) -> Vec<f64> {
    let n = a.len();

    for _ in 0..max_sweeps {
        if max_offdiag(&a) < tolerance {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < tolerance {
                    continue;
                }

                let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                let app = a[p][p];
                let aqq = a[q][q];
                let apq = a[p][q];

                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                for k in 0..n {
                    if k != p && k != q {
                        let akp = a[k][p];
                        let akq = a[k][q];

                        a[k][p] = c * akp - s * akq;
                        a[p][k] = a[k][p];

                        a[k][q] = s * akp + c * akq;
                        a[q][k] = a[k][q];
                    }
                }
            }
        }
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    eigenvalues.sort_by(|x, y| x.partial_cmp(y).unwrap());
    eigenvalues
}

fn main() {
    let n_intervals = 80;
    let m_order = 0;
    let c_param = 0.0;

    println!("Finite-Difference Relaxation for the Spheroidal Eigenvalue Problem");
    println!("==================================================================");
    println!();
    println!("Special case:");
    println!("  m = 0, c = 0");
    println!("  Equation reduces to the Legendre eigenvalue problem");
    println!("  Exact eigenvalues: lambda_l = l(l + 1)");
    println!();
    println!("Discretization:");
    println!("  Interval                         = [-1, 1]");
    println!("  Mesh intervals N                 = {}", n_intervals);
    println!("  Mesh points N + 1                = {}", n_intervals + 1);
    println!("  Endpoint treatment               = natural zero-flux regularity");
    println!("  Matrix type                      = symmetric generalized eigenproblem");
    println!();

    let operator = build_symmetric_operator(n_intervals, m_order, c_param);
    let eigenvalues = jacobi_eigenvalues(operator, 1.0e-12, 200);

    println!("Lowest Discrete Eigenvalues");
    println!("---------------------------");
    println!(
        "{:>6} {:>18} {:>18} {:>18}",
        "l", "lambda_num", "lambda_exact", "abs error"
    );

    for l in 0..6 {
        let exact = exact_legendre_lambda(l);
        let computed = eigenvalues[l];

        println!(
            "{:>6} {:>18.10} {:>18.10} {:>18.10e}",
            l,
            computed,
            exact,
            (computed - exact).abs()
        );
    }

    println!();
    println!("Interpretation");
    println!("--------------");
    println!("The lowest eigenvalue should be close to 0 for the constant mode.");
    println!("The next eigenvalues should approach 2, 6, 12, 20, and 30.");
}
```

Program 18.5.2 demonstrates the finite-difference relaxation approach to the spheroidal eigenvalue problem and highlights its advantages over shooting-based methods. By discretizing the operator globally, the method avoids the sensitivity associated with long integrations and provides a stable framework for computing eigenvalues.

The results show that the computed eigenvalues closely approximate the exact Legendre values, with increasing accuracy for lower modes and gradual degradation for higher modes due to discretization error. This behavior is consistent with the second-order accuracy of the finite-difference scheme and the increased oscillatory nature of higher eigenfunctions.

An important strength of this formulation is its structured sparsity. The tridiagonal nature of the underlying operator allows efficient storage and computation, making the method scalable to large systems. In practical applications, specialized eigenvalue solvers for tridiagonal matrices can be employed to achieve significantly higher efficiency.

The modular design of the implementation also facilitates extensions to more general spheroidal problems, including nonzero parameters $m$ and $c$, as well as higher-dimensional analogues. Improvements such as adaptive mesh refinement or higher-order discretization schemes can further enhance accuracy.

Overall, the program provides a clear computational realization of the finite-difference relaxation framework and illustrates how boundary value eigenproblems can be effectively transformed into structured linear algebra problems.

## 18.5.5. Comparison of Shooting and Relaxation Methods and Numerical Considerations

The shooting and relaxation approaches represent two fundamentally different, yet complementary, strategies for solving the spheroidal eigenvalue boundary value problem. Their relative effectiveness depends on the interplay between stability, accuracy, and computational cost, as well as the structural properties of the governing equation.

The shooting method reduces the problem to repeated integrations of an initial value problem combined with a scalar root-finding procedure for the eigenvalue $\lambda$. Its principal advantage lies in its *computational efficiency and low memory requirements*. Only the state variables need to be stored during integration, and high-order adaptive ODE solvers can be employed to achieve accurate solutions with relatively few function evaluations. When the system is well-conditioned and the solution remains smooth throughout the interval, shooting can provide highly accurate results with minimal computational overhead.

However, the method is sensitive to the structural difficulties inherent in the spheroidal equation. The presence of regular singular points at $x=\pm1$ requires careful initialization near the endpoints, and the propagation of the solution across the full interval introduces sensitivity to instability. For higher eigenvalues or larger values of the parameter $c$, the solution may exhibit oscillatory or rapidly varying behavior, which amplifies numerical errors during integration. As a result, the residual function used to determine $\lambda$ can become ill-conditioned, and convergence of the root-finding procedure may deteriorate.

Relaxation methods, in contrast, adopt a *global discretization strategy*. The entire domain is represented simultaneously on a mesh, and the differential equation and boundary conditions are enforced through a structured algebraic system. This formulation avoids the propagation of errors inherent in shooting, since no sequential integration is performed. Instead, the solution is obtained through a balanced adjustment of all degrees of freedom, leading to improved numerical stability.

A key advantage of relaxation in this context is its ability to handle *singular coefficients and endpoint behavior* in a systematic manner. The discretized system incorporates the singular structure directly, and regularity conditions are enforced through the global solution process. This makes the method more robust for large eigenvalues, strong parameter dependence, or cases where the solution exhibits sharp gradients. Additionally, the tridiagonal structure of the resulting system allows efficient solution using specialized linear algebra techniques, even for fine discretizations.

The trade-off is that relaxation methods require more memory and involve the solution of a global nonlinear or eigenvalue problem. The computational cost increases with the number of mesh points, and careful attention must be paid to mesh resolution and convergence criteria to ensure accuracy.

Beyond these classical approaches, *spectral methods* provide a third, high-accuracy alternative. In these methods, the solution is expanded in terms of orthogonal polynomial bases, such as Legendre or Chebyshev polynomials. This leads to dense matrix eigenvalue problems in which the differential operator is represented globally. For smooth solutions, spectral methods exhibit *rapid convergence*, often achieving high accuracy with relatively few degrees of freedom. However, this comes at the cost of dense linear algebra operations and reduced sparsity, which may limit scalability for very large problems.

In summary, the choice of method depends on the specific characteristics of the problem:

- **Shooting methods** are preferable when the system is well-conditioned, the solution is smooth, and memory efficiency is important.
- **Relaxation methods** are advantageous for problems with singularities, instability, or strong nonlinear coupling, where global consistency improves robustness.
- **Spectral methods** are ideal when very high accuracy is required for smooth solutions and the computational cost of dense matrix operations is acceptable.

Thus, these approaches should not be viewed as competing alternatives but as complementary tools. An effective numerical strategy often involves selecting the method that best aligns with the stability properties, regularity, and computational demands of the problem at hand.

## 18.5.6. Physical Applications of Spheroidal Harmonics and Computational Significance

Spheroidal harmonics play a central role in a wide range of physical applications where the underlying geometry deviates from perfect spherical symmetry. Their importance arises from the fact that they form a natural basis for representing solutions of partial differential equations in spheroidal coordinate systems, much in the same way that spherical harmonics are used for spherical geometries.

One prominent application occurs in *geodesy and planetary science*, where the gravitational potential of an oblate or prolate body must be modeled accurately. Real planets are not perfectly spherical; rotational effects lead to flattening at the poles and bulging at the equator. In such cases, expansions in spheroidal harmonics provide a more accurate representation of the gravitational field than spherical harmonics, capturing deviations due to ellipticity. The associated eigenvalues $\lambda$ determine the structure of these modes and influence the spatial distribution of the gravitational potential.

Another important application arises in **electromagnetic scattering and wave propagation**. When electromagnetic waves interact with objects that are elongated or flattened, such as antennas, aircraft bodies, or radar targets, the governing equations are naturally expressed in spheroidal coordinates. Spheroidal harmonics then describe the angular dependence of the scattered or radiated fields. Accurate computation of these functions is essential for predicting scattering cross-sections, resonance behavior, and radiation patterns in engineering applications.

In these physical contexts, the eigenvalue $\lambda$ is not merely a mathematical parameter but often corresponds to a *characteristic mode quantity*, such as a frequency, propagation constant, or energy level. The accuracy with which these eigenvalues are computed directly affects the reliability of physical predictions. Small errors in $\lambda$ can lead to significant discrepancies in mode shapes, resonance conditions, or field distributions.

From a numerical standpoint, the challenges discussed throughout this section become particularly relevant. For large values of the parameter $c$, corresponding to high frequencies or strong geometric deformation, the spheroidal equation becomes increasingly oscillatory and sensitive. This places greater demands on numerical methods, requiring careful handling of singularities, stability, and resolution. Modern computational techniques, including refined shooting methods, relaxation-based discretizations, and spectral approaches, enable efficient and accurate computation of spheroidal eigenmodes even in these challenging regimes.

This worked example illustrates how the abstract framework developed for two-point boundary value problems applies directly to a physically significant eigenvalue problem. It highlights the interplay between analytical structure, such as singular endpoints and Sturm–Liouville properties, and numerical solution strategies, such as shooting and relaxation. More broadly, it demonstrates how the methods of this chapter provide practical tools for solving complex problems arising in physics and engineering, where accurate and stable computation of eigenvalues and eigenfunctions is essential.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/C0liLo09FyFDFljFbpGc.5","tags":[]}

# 18.6. Automated Allocation of Mesh Points

The efficiency and accuracy of numerical methods for two-point boundary value problems depend critically on the distribution of mesh points. In many practical problems, the solution exhibits localized features such as boundary layers, sharp gradients, or regions of rapid variation. A uniform discretization distributes computational effort evenly across the domain, which is often inefficient, as smooth regions are over-resolved while critical regions remain under-resolved. Automated mesh allocation provides a systematic way to adapt the grid to the structure of the solution, thereby improving accuracy without excessive computational cost.

### 18.6.1 Motivation and Equidistribution Principle for Adaptive Meshes

Two-point boundary value problems frequently involve solutions with highly non-uniform behavior. Examples include singular perturbation problems with thin boundary layers, reaction–diffusion systems with steep internal fronts, and eigenvalue problems with rapidly oscillating modes. In such cases, achieving uniform accuracy with a fixed mesh requires a very large number of grid points, leading to increased computational cost and memory usage.

Adaptive mesh allocation addresses this difficulty by redistributing grid points according to the local behavior of the solution. Instead of prescribing the mesh a priori, the grid is constructed so that points are concentrated in regions where the solution varies rapidly and spaced more widely where the solution is smooth. This leads to a more efficient use of computational resources while maintaining the desired level of accuracy.

A widely used principle for constructing such meshes is the *equidistribution principle*. Let $M(x) > 0$ be a *monitor function* that quantifies the local importance or difficulty of the solution. Typical choices of $M(x)$ include measures of curvature, such as $|u''(x)|$, or estimates of the local residual or discretization error. The mesh points $\{x_j\}_{j=0}^N$ are then chosen so that:

$$\int{x_{j-1}}^{x_j} M(x),dx \approx \frac{1}{N}\int_a^b M(x),dx, \qquad j=1,\dots,N. \tag{18.6.1}$$

This condition ensures that each subinterval carries approximately the same “weight” with respect to the monitor function. As a result, the mesh automatically adapts to the structure of the solution. In regions where $M(x)$ is large, indicating rapid variation or high error, the integral accumulates quickly, forcing the subinterval length $x_j - x_{j-1}$ to be small. Conversely, in regions where $M(x)$ is small, the mesh intervals become larger.

From a geometric perspective, the equidistribution principle can be interpreted as constructing a transformation from the physical domain $[a,b]$ to a computational domain in which the monitor function is uniformly distributed. This transformation effectively stretches the coordinate system in regions of interest and compresses it elsewhere, leading to a nonuniform mesh that aligns with the features of the solution.

In practice, the monitor function is often not known in advance and must be estimated from an approximate solution. This leads to an iterative process in which the solution is computed on an initial mesh, the monitor function is evaluated, and a new mesh is generated based on the equidistribution condition. The solution is then recomputed on the adapted mesh, and the process is repeated until the desired accuracy is achieved.

Thus, the equidistribution principle provides a systematic and mathematically grounded approach to adaptive mesh generation. It enables efficient resolution of localized features in TPBVPs and forms the basis for modern adaptive algorithms used in conjunction with relaxation, collocation, and other global discretization methods.

## 18.6.2. Iterative Mesh Adaptation Algorithm and Equidistributed Grid Construction

In practical computations, the adaptive mesh cannot be determined in advance, since the monitor function $M(x)$ depends on the unknown solution. As a result, mesh generation must be carried out iteratively, in conjunction with the numerical solution of the boundary value problem. This leads to a coupled procedure in which the solution and the mesh are updated alternately until a consistent distribution is obtained.

A typical adaptive mesh algorithm proceeds through the following steps:

1. **Solve the boundary value problem on the current mesh:**\
   Begin with an initial mesh, often uniform or based on prior knowledge of the problem. Compute an approximate solution using a suitable method, such as relaxation or collocation.
2. **Estimate the monitor function** $M(x)$**:**\
   Using the computed solution, evaluate a monitor function that reflects local solution behavior. Common choices include curvature-based measures such as $|u''(x)|$, gradient-based quantities, or residual-based error indicators derived from the discrete equations.
3. **Redistribute mesh points using equidistribution:**\
   Construct a new mesh by enforcing the equidistribution condition, so that each subinterval carries an equal share of the total monitor weight.
4. **Iterate until convergence:**\
   Recompute the solution on the updated mesh and repeat the process until changes in the mesh and solution become sufficiently small.

The equidistribution condition can be expressed in cumulative form as:

$$\int_a^{x_j} M(x)\,dx = \frac{j}{N}\int_a^b M(x)\,dx, \qquad j=1,\dots,N \tag{18.6.2}$$

This formulation ensures that the mesh points $x_j$ are positioned so that the cumulative monitor function is distributed uniformly across the interval. In effect, the transformation from the physical coordinate $x$ to a computational coordinate $\xi = j/N$ produces a grid in which the monitor function is constant with respect to $\xi$.

From a computational perspective, the redistribution step typically involves approximating the integrals in (18.6.2) using quadrature rules based on the current mesh, followed by interpolation to determine the new node locations. Care must be taken to ensure that the resulting mesh remains ordered and avoids excessive clustering, which could lead to numerical ill-conditioning.

This iterative procedure is closely related to *moving mesh methods*, in which the mesh evolves dynamically in response to changes in the solution. In the present context of steady boundary value problems, the adaptation is performed until a stationary mesh is obtained that reflects the final solution structure. The resulting grid provides enhanced resolution in regions of high activity while maintaining efficiency in smooth regions.

Thus, the iterative mesh adaptation algorithm provides a practical realization of the equidistribution principle, enabling the construction of meshes that are optimally aligned with the features of the solution. This significantly improves both the accuracy and efficiency of numerical methods for TPBVPs, particularly when combined with global discretization techniques such as relaxation.

### Rust Implementation

Following the development of the equidistribution principle and the iterative mesh adaptation algorithm in Sections 18.6.1–18.6.2, Program 18.6.1 provides a practical implementation of automated mesh allocation based on a prescribed monitor function. In numerical computation of two-point boundary value problems, the efficient placement of mesh points is essential for resolving localized features such as boundary layers and steep gradients. The equidistribution principle, introduced in equations (18.6.1)–(18.6.2), offers a systematic way to construct such meshes by ensuring that each subinterval carries an equal share of the monitor function. This program implements the cumulative equidistribution strategy by first approximating the integral of the monitor function and then inverting this cumulative distribution to generate a nonuniform grid. The resulting mesh automatically concentrates points in regions of rapid variation, demonstrating how adaptive discretization improves efficiency and accuracy in boundary value computations.

At the core of the implementation is the function `monitor`, which defines the monitor function $M(x)$ used in the equidistribution principle of equation (18.6.1). This function quantifies the local importance of different regions of the domain and determines where mesh points should be concentrated. In the present example, the monitor is constructed to have a sharp peak near a specified location, mimicking the behavior of a solution with a localized boundary layer or steep gradient.

The function `cumulative_monitor` computes a numerical approximation of the cumulative integral of the monitor function, corresponding to the left-hand side of equation (18.6.2). Using a fine auxiliary grid and the trapezoidal rule, it constructs the cumulative distribution that represents the total monitor weight up to each point. This cumulative function serves as the key link between the physical coordinate and the equidistributed computational coordinate.

The inverse mapping required by equation (18.6.2) is implemented in the function `interpolate_inverse_cumulative`. Given a target value of the cumulative monitor function, this function locates the corresponding physical coordinate by linear interpolation. This step effectively inverts the cumulative distribution, allowing mesh points to be placed so that the monitor function is uniformly distributed across subintervals.

The function `equidistributed_mesh` combines these components to generate the adaptive mesh. It evaluates the total monitor weight, divides it into equal portions, and computes the corresponding mesh points by repeatedly applying the inverse cumulative mapping. This realizes the equidistribution condition in a discrete setting and produces a nonuniform grid aligned with the structure of the monitor function.

To verify the quality of the mesh, the function `interval_monitor_weight` computes the monitor weight on each subinterval using numerical quadrature. This provides a direct check of the equidistribution condition in equation (18.6.1), ensuring that each interval carries approximately the same weight. The `main` function orchestrates the entire process, generating the mesh, displaying interval lengths, and evaluating the uniformity of the monitor distribution across the domain.

```rust
// Program 18.6.1: Equidistributed Mesh Generation from a Monitor Function
//
// Problem statement:
// This program constructs an adaptive mesh using the equidistribution principle.
// A positive monitor function M(x) is prescribed on the interval [a, b].
// The mesh points are chosen so that each subinterval carries approximately
// the same monitor weight.
//
// The example monitor is concentrated near x = 0.75, mimicking a boundary layer
// or region of rapid variation in a TPBVP solution.

fn monitor(x: f64) -> f64 {
    let center = 0.75;
    let width = 0.05;
    let strength = 25.0;

    1.0 + strength * (-(x - center).powi(2) / (width * width)).exp()
}

fn cumulative_monitor(fine_points: usize, a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
    let h = (b - a) / (fine_points - 1) as f64;

    let mut x = vec![0.0; fine_points];
    let mut cumulative = vec![0.0; fine_points];

    for i in 0..fine_points {
        x[i] = a + i as f64 * h;
    }

    for i in 1..fine_points {
        let m_left = monitor(x[i - 1]);
        let m_right = monitor(x[i]);

        cumulative[i] = cumulative[i - 1] + 0.5 * h * (m_left + m_right);
    }

    (x, cumulative)
}

fn interpolate_inverse_cumulative(x: &[f64], cumulative: &[f64], target: f64) -> f64 {
    let n = x.len();

    if target <= cumulative[0] {
        return x[0];
    }

    if target >= cumulative[n - 1] {
        return x[n - 1];
    }

    for i in 1..n {
        if cumulative[i] >= target {
            let c0 = cumulative[i - 1];
            let c1 = cumulative[i];
            let x0 = x[i - 1];
            let x1 = x[i];

            let theta = (target - c0) / (c1 - c0);
            return x0 + theta * (x1 - x0);
        }
    }

    x[n - 1]
}

fn equidistributed_mesh(
    intervals: usize,
    fine_points: usize,
    a: f64,
    b: f64,
) -> Vec<f64> {
    let (fine_x, cumulative) = cumulative_monitor(fine_points, a, b);
    let total_weight = cumulative[fine_points - 1];

    let mut mesh = vec![0.0; intervals + 1];

    for j in 0..=intervals {
        let target = j as f64 * total_weight / intervals as f64;
        mesh[j] = interpolate_inverse_cumulative(&fine_x, &cumulative, target);
    }

    mesh[0] = a;
    mesh[intervals] = b;

    mesh
}

fn interval_monitor_weight(x_left: f64, x_right: f64, quadrature_points: usize) -> f64 {
    let h = (x_right - x_left) / (quadrature_points - 1) as f64;

    let mut weight = 0.0;

    for i in 0..quadrature_points {
        let x = x_left + i as f64 * h;
        let factor = if i == 0 || i == quadrature_points - 1 {
            0.5
        } else {
            1.0
        };

        weight += factor * monitor(x);
    }

    weight * h
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let intervals = 20;
    let fine_points = 10_001;

    let mesh = equidistributed_mesh(intervals, fine_points, a, b);

    println!("Equidistributed Mesh Generation");
    println!("==============================");
    println!();
    println!("Monitor function:");
    println!("  M(x) = 1 + 25 exp(-((x - 0.75)^2)/(0.05^2))");
    println!();
    println!("Mesh parameters:");
    println!("  interval [a, b]       = [{:.3}, {:.3}]", a, b);
    println!("  number of intervals N = {}", intervals);
    println!("  fine quadrature grid  = {}", fine_points);
    println!();

    println!("Generated Mesh Points");
    println!("---------------------");
    println!("{:>6} {:>16} {:>16}", "j", "x_j", "h_j");

    for j in 0..=intervals {
        if j == 0 {
            println!("{:>6} {:>16.10} {:>16}", j, mesh[j], "-");
        } else {
            println!(
                "{:>6} {:>16.10} {:>16.10}",
                j,
                mesh[j],
                mesh[j] - mesh[j - 1]
            );
        }
    }

    println!();
    println!("Equidistribution Check");
    println!("----------------------");
    println!(
        "{:>6} {:>16} {:>20}",
        "j", "interval length", "monitor weight"
    );

    let mut weights = Vec::with_capacity(intervals);

    for j in 1..=intervals {
        let weight = interval_monitor_weight(mesh[j - 1], mesh[j], 101);
        weights.push(weight);

        println!(
            "{:>6} {:>16.10} {:>20.10e}",
            j,
            mesh[j] - mesh[j - 1],
            weight
        );
    }

    let average_weight = weights.iter().sum::<f64>() / intervals as f64;
    let max_deviation = weights
        .iter()
        .map(|w| (w - average_weight).abs())
        .fold(0.0_f64, |a, b| a.max(b));

    println!();
    println!("Summary");
    println!("-------");
    println!("Average monitor weight per interval = {:.12e}", average_weight);
    println!("Maximum absolute weight deviation   = {:.12e}", max_deviation);
    println!(
        "Smallest interval length            = {:.12e}",
        mesh.windows(2)
            .map(|w| w[1] - w[0])
            .fold(f64::INFINITY, |a, b| a.min(b))
    );
    println!(
        "Largest interval length             = {:.12e}",
        mesh.windows(2)
            .map(|w| w[1] - w[0])
            .fold(0.0_f64, |a, b| a.max(b))
    );
}
```

Program 18.6.1 demonstrates a practical implementation of the equidistribution principle for adaptive mesh generation. By constructing a mesh that equalizes the monitor function across subintervals, the method ensures that computational effort is concentrated in regions where the solution exhibits significant variation.

The results illustrate that the generated mesh successfully clusters points in regions where the monitor function is large, while maintaining larger intervals in smooth regions. The equidistribution check confirms that each subinterval carries nearly the same monitor weight, validating the numerical realization of the principle.

An important advantage of this approach is its generality. The monitor function can be tailored to reflect different aspects of the solution, such as curvature, gradients, or residual errors, making the method adaptable to a wide range of problems. When combined with iterative solution procedures, this framework enables dynamic mesh refinement that evolves in response to the computed solution.

The modular structure of the implementation also facilitates extensions to more advanced adaptive strategies, including moving mesh methods, higher-dimensional grid generation, and coupling with global discretization techniques such as relaxation and collocation. Overall, the program provides a clear computational realization of adaptive mesh allocation and establishes a foundation for integrating mesh adaptation into numerical methods for boundary value problems.

## 18.6.3. hp-Adaptivity and Multiresolution Strategies for TPBVPs

While adaptive mesh refinement based on equidistribution focuses on adjusting the *spacing* of grid points (h-adaptivity), modern numerical methods extend this idea by also adapting the *local approximation order* (p-adaptivity). The combination of these two strategies leads to *hp-adaptive methods*, which provide a powerful framework for achieving high accuracy with efficient use of computational resources.

In hp-adaptive methods, the domain $[a,b]$ is partitioned into a set of subintervals or elements. Within each element, the solution is approximated using a finite-dimensional basis, typically polynomial. The adaptive strategy then proceeds by deciding, for each element, whether to:

- refine the mesh locally (h-refinement) by subdividing the element, or
- increase the approximation order (p-refinement) by using higher-degree basis functions.

This decision is guided by estimates of the local truncation error or residual. In regions where the solution exhibits smooth behavior, increasing the polynomial order is often more efficient, as high-order approximations can achieve exponential convergence. In contrast, in regions with sharp gradients, boundary layers, or singularities, mesh refinement is more effective, as it allows the discretization to resolve localized features.

A representative implementation of this approach is given by Drozd et al. (2024), who develop an hp-adaptive mesh refinement strategy based on the Theory of Functional Connections. In their formulation, the solution is represented on ea ch segment using Chebyshev polynomial expansions, which provide high-order accuracy and favorable numerical properties. An error estimate is used to determine whether a given segment should be refined (h-adaptivity) or enriched with higher-order basis functions (p-adaptivity). This selective refinement leads to highly accurate solutions with relatively few degrees of freedom compared to uniform discretizations. However, the method introduces additional complexity in both implementation and data management, particularly in handling variable polynomial orders and ensuring continuity across segment boundaries.

Beyond classical polynomial-based approaches, *multiresolution and learning-based methods* are emerging as promising directions for adaptive discretization. In these approaches, the adaptive strategy is not prescribed explicitly but is instead inferred from data or learned through optimization. For example, Luong et al. (2026) propose a neural-network-based shooting framework for multipoint boundary value problems. In this method, the model effectively learns both the optimal distribution of collocation points and suitable initial conditions during training. The adaptive behavior is thus embedded within the learning process, allowing the method to identify regions of interest and allocate computational effort accordingly.

From a broader perspective, these developments highlight a shift toward *flexible, data-informed discretization strategies*. Traditional h-adaptive methods rely on explicit monitor functions, while hp-adaptive and neural approaches incorporate richer representations of solution behavior. The goal in all cases is to concentrate computational effort where it yields the greatest improvement in accuracy.

In summary, hp-adaptivity and multiresolution methods extend the principles of mesh adaptation by combining geometric refinement with variable approximation order and, in some cases, data-driven optimization. These techniques enable highly efficient and accurate solutions of TPBVPs, particularly for problems with complex structure, localized features, or varying regularity across the domain.

## 18.6.4. Computational Complexity and Efficiency of Adaptive Mesh Methods

Adaptive mesh methods introduce an additional layer of computation beyond the base boundary value solver. Specifically, the solution process becomes *iterative at two levels*: an inner iteration that solves the discretized TPBVP on a given mesh, and an outer iteration that updates the mesh based on the computed solution. As a result, each mesh adaptation cycle requires recomputation of the solution, increasing the total computational effort relative to a single fixed-mesh solve.

However, this apparent overhead is offset by a significant gain in efficiency. An adapted mesh concentrates points only where they are needed, allowing the same level of accuracy to be achieved with far fewer grid points than a uniformly refined mesh. Consequently, although multiple solves are required, each solve is performed on a smaller and more efficient discretization. For problems with localized features such as boundary layers or sharp gradients, this leads to a substantial reduction in overall computational cost for a given accuracy target.

The computational complexity per iteration is dominated by the cost of solving the discretized boundary value problem. Depending on the underlying method, this typically scales as:

$$O(N) \quad \text{or} \quad O(N \log N),$$

where $N$ denotes the number of mesh points or degrees of freedom. Linear or near-linear scaling is achieved when structured solvers, such as block-tridiagonal elimination or sparse iterative methods, are employed. In contrast, the cost of mesh redistribution itself, including evaluation of the monitor function and computation of new node locations, is relatively small and does not significantly affect the overall complexity.

In practice, the number of outer iterations required for mesh adaptation is modest. For many problems, convergence is achieved in only a few adaptation cycles, as the mesh rapidly aligns with the dominant features of the solution. Only in cases involving extreme stiffness, strong nonlinearities, or highly localized structures does the number of iterations increase appreciably.

An important practical consideration is the balance between mesh resolution and solver cost. While increasing $N$ improves accuracy, it also increases the cost of each inner solve. Adaptive methods address this trade-off by minimizing $N$ while maintaining accuracy, thereby optimizing computational efficiency. The effectiveness of this approach depends on the quality of the monitor function and the robustness of the underlying solver.

In summary, although adaptive mesh methods introduce additional iterations, they often reduce the total computational effort required to achieve a desired level of accuracy. By concentrating computational resources in critical regions and exploiting efficient solvers for the resulting structured systems, they provide a highly effective strategy for solving TPBVPs with non-uniform solution behavior.

### Rust Implementation

Following the development of the equidistribution principle in equations (18.6.1)–(18.6.2) and the iterative mesh adaptation framework described in Section 18.6.2, Program 18.6.2 provides a practical implementation of an adaptive mesh strategy coupled with the numerical solution of a two-point boundary value problem. In many problems of practical interest, particularly those involving boundary layers or localized sharp gradients, a uniform mesh leads to inefficient use of computational resources. The adaptive approach addresses this limitation by dynamically redistributing mesh points according to a monitor function derived from the current solution. This program implements a complete adaptive loop in which the boundary value problem is solved on a given mesh, a monitor function is constructed based on solution features, and a new mesh is generated via equidistribution. The process is repeated until the mesh stabilizes, demonstrating how adaptive discretization enhances resolution in critical regions while maintaining computational efficiency.

At the core of the implementation is the function `solve_bvp_on_mesh`, which computes the numerical solution of the boundary value problem on a given nonuniform mesh. This function constructs a discrete system based on finite-difference approximations consistent with the relaxation framework discussed earlier in Section 18.4. The resulting linear system is tridiagonal and is solved efficiently using the function `solve_tridiagonal`, which implements a forward elimination and back substitution procedure. This corresponds to solving the discrete analogue of the differential equation at each stage of the adaptive process.

The adaptive component is driven by the monitor function constructed in `monitor_values`. This function estimates the curvature of the solution using the second-derivative approximation provided by `estimate_second_derivative`. The monitor function reflects the principle introduced in equation (18.6.1), where regions of high curvature correspond to larger values of the monitor and therefore attract a higher density of mesh points. The normalization applied in this function ensures that the monitor remains well-scaled across different iterations.

The mesh redistribution is performed by the function `redistribute_mesh`, which implements the equidistribution condition described in equation (18.6.2). It constructs a cumulative distribution of the monitor function and inverts this distribution to obtain a new mesh in which each subinterval carries approximately equal monitor weight. This step is analogous to the inverse mapping used in Program 18.6.1 and forms the central mechanism for adaptive mesh refinement.

The function `interpolate_monitor` provides a continuous representation of the monitor function over the domain by interpolating between mesh points. This allows the redistribution procedure to evaluate the monitor on a fine auxiliary grid, ensuring a smooth and accurate approximation of the cumulative distribution.

The adaptive loop in the `main` function orchestrates the entire process. Starting from an initial uniform mesh, the program alternates between solving the boundary value problem, computing the monitor function, and redistributing the mesh. The functions `max_mesh_change` and `max_solution_error` are used to quantify the evolution of the mesh and the accuracy of the solution, respectively. These diagnostics provide insight into the convergence of the adaptive procedure and its effectiveness in resolving the boundary layer structure of the solution.

```rust
// Program 18.6.2: Adaptive Mesh Relaxation with Iterative Equidistribution
//
// Problem statement:
// This program demonstrates an adaptive mesh workflow for the boundary layer
// problem
//
//     eps y''(x) + y'(x) = 0,   0 <= x <= 1,
//     y(0) = 0,                 y(1) = 1.
//
// For small eps, the solution has a sharp boundary layer near x = 0.
// The program alternates between solving the TPBVP on the current mesh,
// estimating a curvature-based monitor function, and redistributing mesh
// points by equidistribution.

fn exact_solution(x: f64, eps: f64) -> f64 {
    let denom = 1.0 - (-1.0 / eps).exp();
    (1.0 - (-x / eps).exp()) / denom
}

fn initial_mesh(n: usize) -> Vec<f64> {
    (0..=n).map(|j| j as f64 / n as f64).collect()
}

fn solve_tridiagonal(lower: &[f64], diag: &[f64], upper: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();

    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let denom = diag[i] - lower[i] * c_prime[i - 1];

        if i < n - 1 {
            c_prime[i] = upper[i] / denom;
        }

        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom;
    }

    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];

    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    x
}

fn solve_bvp_on_mesh(mesh: &[f64], eps: f64) -> Vec<f64> {
    let n = mesh.len() - 1;
    let interior = n - 1;

    let mut lower = vec![0.0; interior];
    let mut diag = vec![0.0; interior];
    let mut upper = vec![0.0; interior];
    let mut rhs = vec![0.0; interior];

    for j in 1..n {
        let row = j - 1;

        let h_left = mesh[j] - mesh[j - 1];
        let h_right = mesh[j + 1] - mesh[j];

        let second_left = 2.0 / (h_left * (h_left + h_right));
        let second_mid = -2.0 / (h_left * h_right);
        let second_right = 2.0 / (h_right * (h_left + h_right));

        let first_left = -h_right / (h_left * (h_left + h_right));
        let first_mid = (h_right - h_left) / (h_left * h_right);
        let first_right = h_left / (h_right * (h_left + h_right));

        lower[row] = eps * second_left + first_left;
        diag[row] = eps * second_mid + first_mid;
        upper[row] = eps * second_right + first_right;

        rhs[row] = 0.0;
    }

    let y0 = 0.0;
    let yn = 1.0;

    rhs[0] -= lower[0] * y0;
    lower[0] = 0.0;

    rhs[interior - 1] -= upper[interior - 1] * yn;
    upper[interior - 1] = 0.0;

    let interior_solution = solve_tridiagonal(&lower, &diag, &upper, &rhs);

    let mut y = vec![0.0; n + 1];
    y[0] = y0;
    y[n] = yn;

    for j in 1..n {
        y[j] = interior_solution[j - 1];
    }

    y
}

fn estimate_second_derivative(mesh: &[f64], y: &[f64]) -> Vec<f64> {
    let n = mesh.len() - 1;
    let mut curvature = vec![0.0; n + 1];

    for j in 1..n {
        let h_left = mesh[j] - mesh[j - 1];
        let h_right = mesh[j + 1] - mesh[j];

        let slope_left = (y[j] - y[j - 1]) / h_left;
        let slope_right = (y[j + 1] - y[j]) / h_right;

        curvature[j] = 2.0 * (slope_right - slope_left) / (h_left + h_right);
    }

    curvature[0] = curvature[1];
    curvature[n] = curvature[n - 1];

    curvature
}

fn monitor_values(mesh: &[f64], y: &[f64]) -> Vec<f64> {
    let curvature = estimate_second_derivative(mesh, y);
    let scale = curvature.iter().fold(0.0_f64, |m, v| m.max(v.abs()));

    curvature
        .iter()
        .map(|v| 1.0 + 5.0 * v.abs() / scale.max(1.0e-14))
        .collect()
}

fn interpolate_monitor(x: f64, mesh: &[f64], monitor: &[f64]) -> f64 {
    let n = mesh.len() - 1;

    if x <= mesh[0] {
        return monitor[0];
    }

    if x >= mesh[n] {
        return monitor[n];
    }

    for j in 1..=n {
        if x <= mesh[j] {
            let theta = (x - mesh[j - 1]) / (mesh[j] - mesh[j - 1]);
            return (1.0 - theta) * monitor[j - 1] + theta * monitor[j];
        }
    }

    monitor[n]
}

fn redistribute_mesh(mesh: &[f64], monitor: &[f64], fine_points: usize) -> Vec<f64> {
    let intervals = mesh.len() - 1;
    let a = mesh[0];
    let b = mesh[intervals];

    let mut fine_x = vec![0.0; fine_points];
    let mut cumulative = vec![0.0; fine_points];

    for i in 0..fine_points {
        fine_x[i] = a + (b - a) * i as f64 / (fine_points - 1) as f64;
    }

    for i in 1..fine_points {
        let h = fine_x[i] - fine_x[i - 1];
        let m_left = interpolate_monitor(fine_x[i - 1], mesh, monitor);
        let m_right = interpolate_monitor(fine_x[i], mesh, monitor);

        cumulative[i] = cumulative[i - 1] + 0.5 * h * (m_left + m_right);
    }

    let total = cumulative[fine_points - 1];
    let mut new_mesh = vec![0.0; intervals + 1];

    for j in 0..=intervals {
        let target = j as f64 * total / intervals as f64;

        for i in 1..fine_points {
            if cumulative[i] >= target {
                let theta =
                    (target - cumulative[i - 1]) / (cumulative[i] - cumulative[i - 1]);
                new_mesh[j] = fine_x[i - 1] + theta * (fine_x[i] - fine_x[i - 1]);
                break;
            }
        }
    }

    new_mesh[0] = a;
    new_mesh[intervals] = b;

    new_mesh
}

fn max_mesh_change(old_mesh: &[f64], new_mesh: &[f64]) -> f64 {
    old_mesh
        .iter()
        .zip(new_mesh.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, |m, v| m.max(v))
}

fn max_solution_error(mesh: &[f64], y: &[f64], eps: f64) -> f64 {
    mesh.iter()
        .zip(y.iter())
        .map(|(x, yi)| (yi - exact_solution(*x, eps)).abs())
        .fold(0.0_f64, |m, v| m.max(v))
}

fn main() {
    let eps = 0.03;
    let intervals = 40;
    let adaptation_cycles = 6;
    let fine_points = 10_001;

    println!("Adaptive Mesh Relaxation for a Boundary Layer Problem");
    println!("====================================================");
    println!();
    println!("Model problem:");
    println!("  eps y''(x) + y'(x) = 0,   0 <= x <= 1");
    println!("  y(0) = 0,                 y(1) = 1");
    println!("  eps = {:.6}", eps);
    println!();
    println!("Adaptive setup:");
    println!("  mesh intervals      = {}", intervals);
    println!("  adaptation cycles   = {}", adaptation_cycles);
    println!("  monitor             = curvature-based");
    println!();

    let mut mesh = initial_mesh(intervals);
    let mut solution;

    println!("Adaptation History");
    println!("------------------");
    println!(
        "{:>6} {:>18} {:>18} {:>18} {:>18}",
        "cycle", "max mesh change", "min h", "max h", "max error"
    );

    for cycle in 0..adaptation_cycles {
        solution = solve_bvp_on_mesh(&mesh, eps);
        let monitor = monitor_values(&mesh, &solution);
        let new_mesh = redistribute_mesh(&mesh, &monitor, fine_points);

        let change = max_mesh_change(&mesh, &new_mesh);
        let min_h = new_mesh
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(f64::INFINITY, |a, b| a.min(b));
        let max_h = new_mesh
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(0.0_f64, |a, b| a.max(b));
        let error = max_solution_error(&mesh, &solution, eps);

        println!(
            "{:>6} {:>18.10e} {:>18.10e} {:>18.10e} {:>18.10e}",
            cycle, change, min_h, max_h, error
        );

        mesh = new_mesh;
    }

    solution = solve_bvp_on_mesh(&mesh, eps);

    println!();
    println!("Representative Adaptive Mesh Values");
    println!("-----------------------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "j", "x_j", "y_num", "y_exact", "abs error"
    );

    for j in [0, 1, 2, 3, 5, 10, 20, 30, 40] {
        let exact = exact_solution(mesh[j], eps);

        println!(
            "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.10e}",
            j,
            mesh[j],
            solution[j],
            exact,
            (solution[j] - exact).abs()
        );
    }
}
```

Program 18.6.2 demonstrates a complete adaptive mesh refinement strategy for boundary value problems, integrating mesh generation and solution computation within a unified framework. This approach reflects the central idea of Section 18.6: that mesh points should be allocated dynamically based on the evolving structure of the solution.

The results show that the mesh rapidly adapts to concentrate points in regions of steep gradients, such as boundary layers, while maintaining coarser resolution elsewhere. The convergence of the mesh, as indicated by the decreasing mesh change across iterations, confirms the stability of the equidistribution process. At the same time, the solution accuracy improves significantly in regions where the adaptive mesh provides enhanced resolution.

An important observation is that the choice of monitor function plays a critical role in determining the effectiveness of the adaptation. In this program, a curvature-based monitor is used, which captures regions of rapid variation but does not directly minimize the global solution error. This highlights a key aspect of adaptive methods: the monitor function must be carefully designed to balance local resolution and global accuracy.

The modular structure of the implementation allows for straightforward extensions. More sophisticated monitor functions based on residuals or error estimates can be incorporated, and the framework can be extended to higher-dimensional problems or coupled systems. Additionally, adaptive strategies can be combined with higher-order discretization methods to further enhance accuracy. Overall, the program provides a clear computational realization of adaptive mesh refinement and illustrates how equidistribution can be used effectively to improve numerical solutions of boundary value problems.

## 18.6.5. Comparison with Fixed-Mesh and High-Order Approximation Methods

Adaptive mesh strategies provide a clear advantage over fixed, uniformly distributed meshes when solving TPBVPs with non-uniform solution behavior. In a fixed mesh, resolution is distributed evenly across the domain, which leads to inefficiency: smooth regions are over-resolved, while regions containing boundary layers, sharp gradients, or localized nonlinear effects may remain under-resolved unless a very fine global discretization is used. This results in a rapid increase in computational cost without proportional gains in accuracy.

Adaptive meshes address this imbalance by redistributing nodes according to the local behavior of the solution. Regions of rapid variation receive a higher density of points, while smooth regions are covered with fewer points. This targeted allocation improves both *accuracy* and *efficiency*, as computational effort is concentrated where it has the greatest impact. For problems with localized features, the difference in performance between uniform and adaptive meshes can be substantial.

When compared to purely p-adaptive or spectral methods, adaptive mesh refinement offers a different type of flexibility. High-order methods achieve excellent accuracy for smooth solutions through global or local polynomial approximations, often exhibiting rapid convergence. However, their performance can degrade in the presence of localized singularities, steep gradients, or discontinuities, where high-order approximations may require excessive polynomial degrees or exhibit oscillatory behavior.

Adaptive mesh methods, by contrast, handle such localized phenomena more naturally by refining the mesh rather than increasing the approximation order. This makes them particularly effective for problems with boundary layers or singular perturbations. For example, spline-based adaptive mesh methods have been developed for singularly perturbed reaction–diffusion systems, demonstrating uniform convergence by clustering mesh points within boundary layers and maintaining coarser resolution elsewhere (Kaushik et al., 2024).

In practice, the most effective strategies often combine both approaches, leading to hp-adaptive methods that exploit the strengths of geometric refinement and high-order approximation. Nevertheless, pure mesh adaptation remains a robust and widely applicable technique, especially when the solution exhibits limited regularity or strong localization.

## 18.6.6. Applications and Practical Importance of Adaptive Mesh Allocation

Adaptive mesh allocation plays a critical role in a wide range of real-world applications, where the solution of boundary value problems exhibits strong spatial variation. In such settings, uniform discretizations are often impractical, and adaptive strategies become essential for achieving accurate and efficient numerical solutions.

In *optimal control problems*, the state and control variables frequently exhibit sharp transitions over short intervals, such as switching behavior or boundary-layer-type structures. Capturing these features accurately requires high resolution in localized regions, which adaptive meshes provide naturally. Hp-adaptive methods have been successfully applied in this context, allowing efficient resolution of narrow regions where the solution changes rapidly (Drozd et al., 2024).

In *fluid dynamics*, adaptive meshes are indispensable for resolving boundary layers, shear layers, and shock regions. These features are typically confined to small portions of the domain but have a dominant influence on the overall solution. By concentrating grid points near boundaries or regions of high gradient, adaptive methods enable accurate simulation of laminar boundary layers and related phenomena without excessive global refinement.

Similarly, in *thermal conduction and reaction–diffusion problems*, steep gradients may occur near interfaces, material boundaries, or reaction zones. Adaptive mesh allocation allows these regions to be resolved with high precision while maintaining computational efficiency in the rest of the domain. This is particularly important in multiphysics problems, where different processes operate on widely varying spatial scales.

More broadly, adaptive mesh techniques transform challenging boundary value problems into tractable computational problems. By aligning the discretization with the intrinsic structure of the solution, they reduce the number of degrees of freedom required for a given accuracy and improve the robustness of numerical solvers. This makes them an essential component of modern numerical methods for TPBVPs, especially in applications involving complex geometries, nonlinear interactions, and localized phenomena.

In summary, adaptive mesh allocation provides a powerful mechanism for bridging the gap between theoretical formulations and practical computation. It enables efficient resolution of complex solution features and supports the application of TPBVP methods to a wide range of problems in science and engineering.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/IfOsTwj3O2Khw2puBt3l.3","tags":[]}

# 18.7. Handling Internal Boundary Conditions or Singular Points

Boundary value problems frequently involve constraints not only at the endpoints but also at interior locations or singular points within the domain. These situations arise in multi-material systems, phase interfaces, or problems with internal sources, and require extensions of the standard TPBVP framework. The key challenge is to enforce additional conditions while maintaining consistency of the solution across the entire domain.

## 18.7.1. Internal (Multipoint) Boundary Conditions and Domain Decomposition

Many boundary value problems include conditions imposed at interior points of the domain. Such problems are commonly referred to as *multipoint boundary value problems*. A typical setting occurs in multi-material or multiphysics systems, where physical quantities must satisfy continuity or jump conditions at an interface $x=\xi$.

A representative formulation includes endpoint conditions:

$$u(a)=A, \qquad u(b)=B, \tag{18.7.1}$$

together with interface conditions:

$$[u]\xi = 0, \qquad [p(x)u']\xi = J \tag{18.7.2}$$

where the jump operator is defined by:

$$[f]_\xi = f(\xi^+) - f(\xi^-)$$

The first condition enforces continuity of the solution across the interface, while the second specifies a prescribed jump in the flux, reflecting physical effects such as material discontinuities or internal sources.

A standard numerical strategy for handling such problems is *domain decomposition*. The interval $[a,b]$ is split into subdomains $[a,\xi]$ and $[\xi,b]$, and separate solutions are constructed on each subdomain. Let $u_1(x)$ denote the solution on the left interval and $u_2(x)$ the solution on the right. At the interface, introduce unknown parameters:

$$u_1(\xi^-)=U, \qquad u_1'(\xi^-)=V, \qquad u_2(\xi^+)=U, \qquad u_2'(\xi^+)=V' \tag{18.7.3}$$

These parameters represent the interface values of the solution and its derivatives. The continuity condition ensures that the function values match, while the flux condition relates the derivatives through the coefficient $p(x)$. The unknowns $U$, $V$, and $V'$ are determined by enforcing the interface conditions (18.7.2), leading to a small nonlinear system.

Within a *shooting framework*, this formulation leads naturally to a multi-segment shooting method. One integrates from the left endpoint $x=a$ toward $x=\xi$ using initial conditions consistent with $u(a)=A$, and separately integrates from the right endpoint $x=b$ toward $x=\xi$ using conditions consistent with $u(b)=B$. The interface parameters are then adjusted so that the continuity and flux conditions are satisfied. This approach extends the ideas of double shooting to problems with prescribed interior constraints.

In a *relaxation or collocation framework*, the treatment is more direct. The mesh is constructed so that one of the nodes coincides with the interface $x=\xi$. At this node, the standard discretization equations are replaced by algebraic constraints enforcing the interface conditions. For example, one imposes continuity of the discrete solution and enforces the flux jump condition through appropriate finite-difference or collocation expressions. This approach integrates seamlessly with the global residual formulation, treating interface conditions in the same way as boundary conditions.

An important advantage of the relaxation formulation is that it maintains a *unified system of equations* over the entire domain. Interior conditions are incorporated directly into the residual vector, and the resulting nonlinear system is solved using the same Newton-based framework. This avoids the need for separate matching procedures and improves robustness, particularly for problems with multiple interfaces or complex coupling.

Recent developments extend these ideas to *machine-learning-based solvers*. In neural-network shooting or collocation methods, multipoint boundary conditions are incorporated explicitly into the loss function. The solution is represented by a parameterized model, and deviations from both endpoint and interior conditions are penalized during optimization. This ensures that the learned solution satisfies all constraints simultaneously, including those at interior points (Luong et al., 2026).

Thus, multipoint boundary value problems generalize the standard TPBVP framework by introducing interior constraints. Whether handled through domain decomposition in shooting methods or directly incorporated into the residual system in relaxation methods, these problems can be treated systematically using extensions of the techniques developed earlier in this chapter.

### Rust Implementation

Following the discussion in Section 18.7.1 on the formulation and numerical treatment of multipoint boundary value problems with internal interface conditions, Program 18.7.1 provides a practical implementation of a finite-difference relaxation method that incorporates both endpoint constraints (18.7.1) and interface conditions (18.7.2) directly into a unified linear system. In contrast to shooting-based approaches, which require matching solutions across subdomains, the relaxation formulation constructs a global discretization in which the interface condition is enforced by replacing the standard finite-difference equation at the interface node. This approach demonstrates how continuity of the solution and prescribed flux jumps can be handled seamlessly within a single system of algebraic equations. The program also includes an exact piecewise-linear solution for validation, illustrating how the numerical method reproduces the analytical behavior of the solution to machine precision.

At the core of the implementation is the `InterfaceProblem` struct, which encapsulates all parameters defining the boundary value problem, including the domain $[a,b]$, the interface location $\xi$, boundary values from (18.7.1), and coefficients associated with the flux jump condition in (18.7.2). This abstraction allows the problem to be specified cleanly and passed consistently to all computational components.

The function `assemble_interface_system` constructs the global finite-difference system corresponding to the relaxation formulation. A uniform mesh is generated over the domain, with the requirement that the interface location $\xi$ coincides with a grid node. For interior nodes away from the interface, standard second-order finite-difference approximations are used to discretize the differential operator. At the interface node, however, the usual discretization is replaced by an algebraic equation enforcing the flux jump condition (18.7.2). This directly implements the interface constraint within the linear system, while continuity of the solution is automatically satisfied by using a single unknown at the interface point, consistent with the formulation in (18.7.3).

The linear system is solved using the `gaussian_elimination` function, which implements a basic pivoted Gaussian elimination method. Although not optimized for large-scale systems, this solver is sufficient for demonstrating the structure and correctness of the relaxation approach. It highlights how the inclusion of interface conditions does not alter the overall solution strategy, but only modifies specific rows of the system matrix.

To verify correctness, the function `exact_piecewise_solution` constructs an analytical solution consistent with the interface conditions. This solution is piecewise linear, with slopes determined by the coefficients on each side of the interface and adjusted to satisfy both continuity and the prescribed flux jump. The comparison between numerical and exact solutions provides a direct measure of accuracy and confirms that the discretization correctly captures the behavior imposed by (18.7.2).

The `main` function orchestrates the computation. It initializes the problem parameters, assembles and solves the linear system, and evaluates diagnostic quantities at the interface, including the computed fluxes and their difference. These diagnostics explicitly verify that the jump condition is satisfied to machine precision. The program also computes the maximum error relative to the exact solution and prints the full grid solution, demonstrating the accuracy and stability of the method across the domain.

```rust
// Program 18.7.1: Finite-Difference Relaxation for an Interface Boundary Value Problem
//
// Problem statement:
// Solve a one-dimensional boundary value problem with an internal interface at x = xi.
// The solution satisfies endpoint boundary conditions u(a) = A and u(b) = B,
// together with continuity of u and a prescribed jump in the flux:
//
//     [u]_xi = 0,
//     [p(x)u']_xi = J.
//
// The mesh is chosen so that xi is a grid point. The interface condition replaces
// the usual finite-difference equation at that node.

#[derive(Clone, Copy)]
struct InterfaceProblem {
    a: f64,
    b: f64,
    xi: f64,
    u_a: f64,
    u_b: f64,
    p_left: f64,
    p_right: f64,
    flux_jump: f64,
}

fn gaussian_elimination(mut a: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();

    for k in 0..n {
        let mut pivot = k;
        let mut max_value = a[k][k].abs();

        for i in (k + 1)..n {
            if a[i][k].abs() > max_value {
                max_value = a[i][k].abs();
                pivot = i;
            }
        }

        if max_value < 1.0e-14 {
            panic!("Singular or nearly singular linear system.");
        }

        if pivot != k {
            a.swap(k, pivot);
            rhs.swap(k, pivot);
        }

        for i in (k + 1)..n {
            let factor = a[i][k] / a[k][k];

            for j in k..n {
                a[i][j] -= factor * a[k][j];
            }

            rhs[i] -= factor * rhs[k];
        }
    }

    let mut x = vec![0.0; n];

    for i in (0..n).rev() {
        let mut sum = rhs[i];

        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }

        x[i] = sum / a[i][i];
    }

    x
}

fn assemble_interface_system(problem: InterfaceProblem, n_intervals: usize) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    assert!(
        n_intervals % 2 == 0,
        "Use an even number of intervals so that the interface is a mesh point."
    );

    let n_nodes = n_intervals + 1;
    let h = (problem.b - problem.a) / n_intervals as f64;

    let mut x = vec![0.0; n_nodes];
    for j in 0..n_nodes {
        x[j] = problem.a + j as f64 * h;
    }

    let interface_index = ((problem.xi - problem.a) / h).round() as usize;

    if (x[interface_index] - problem.xi).abs() > 1.0e-12 {
        panic!("The interface point xi must coincide with a mesh node.");
    }

    let mut matrix = vec![vec![0.0; n_nodes]; n_nodes];
    let mut rhs = vec![0.0; n_nodes];

    matrix[0][0] = 1.0;
    rhs[0] = problem.u_a;

    matrix[n_nodes - 1][n_nodes - 1] = 1.0;
    rhs[n_nodes - 1] = problem.u_b;

    for j in 1..(n_nodes - 1) {
        if j == interface_index {
            matrix[j][j - 1] = problem.p_left / h;
            matrix[j][j] = -(problem.p_left + problem.p_right) / h;
            matrix[j][j + 1] = problem.p_right / h;
            rhs[j] = problem.flux_jump;
        } else if j < interface_index {
            matrix[j][j - 1] = problem.p_left;
            matrix[j][j] = -2.0 * problem.p_left;
            matrix[j][j + 1] = problem.p_left;
            rhs[j] = 0.0;
        } else {
            matrix[j][j - 1] = problem.p_right;
            matrix[j][j] = -2.0 * problem.p_right;
            matrix[j][j + 1] = problem.p_right;
            rhs[j] = 0.0;
        }
    }

    (x, matrix, rhs)
}

fn exact_piecewise_solution(problem: InterfaceProblem, x: f64) -> f64 {
    let left_length = problem.xi - problem.a;
    let right_length = problem.b - problem.xi;

    let numerator =
        problem.p_right * (problem.u_b / right_length)
        + problem.p_left * (problem.u_a / left_length)
        - problem.flux_jump;

    let denominator =
        problem.p_right / right_length
        + problem.p_left / left_length;

    let interface_value = numerator / denominator;

    if x <= problem.xi {
        problem.u_a + (interface_value - problem.u_a) * (x - problem.a) / left_length
    } else {
        interface_value + (problem.u_b - interface_value) * (x - problem.xi) / right_length
    }
}

fn main() {
    let problem = InterfaceProblem {
        a: 0.0,
        b: 1.0,
        xi: 0.5,
        u_a: 0.0,
        u_b: 1.0,
        p_left: 1.0,
        p_right: 4.0,
        flux_jump: 2.0,
    };

    let n_intervals = 20;

    let (x, matrix, rhs) = assemble_interface_system(problem, n_intervals);
    let u = gaussian_elimination(matrix, rhs);

    let h = (problem.b - problem.a) / n_intervals as f64;
    let interface_index = ((problem.xi - problem.a) / h).round() as usize;

    let left_flux = problem.p_left * (u[interface_index] - u[interface_index - 1]) / h;
    let right_flux = problem.p_right * (u[interface_index + 1] - u[interface_index]) / h;
    let computed_jump = right_flux - left_flux;

    let mut max_error: f64 = 0.0;

    for j in 0..u.len() {
        let exact = exact_piecewise_solution(problem, x[j]);
        max_error = max_error.max((u[j] - exact).abs());
    }

    println!("Interface Boundary Value Problem by Relaxation");
    println!("==============================================");
    println!();
    println!("Problem Parameters");
    println!("------------------");
    println!("Domain                         = [{:.6}, {:.6}]", problem.a, problem.b);
    println!("Interface xi                   = {:.6}", problem.xi);
    println!("Left coefficient p_left         = {:.6}", problem.p_left);
    println!("Right coefficient p_right       = {:.6}", problem.p_right);
    println!("Prescribed flux jump J          = {:.6}", problem.flux_jump);
    println!("Boundary value u(a)             = {:.6}", problem.u_a);
    println!("Boundary value u(b)             = {:.6}", problem.u_b);
    println!("Number of intervals             = {}", n_intervals);
    println!("Step size h                     = {:.6}", h);
    println!();

    println!("Interface Diagnostics");
    println!("---------------------");
    println!("Computed u(xi)                  = {:.12}", u[interface_index]);
    println!("Left flux p_left u'(xi-)        = {:.12}", left_flux);
    println!("Right flux p_right u'(xi+)      = {:.12}", right_flux);
    println!("Computed flux jump              = {:.12}", computed_jump);
    println!("Jump residual                   = {:.3e}", computed_jump - problem.flux_jump);
    println!("Maximum error against exact form = {:.3e}", max_error);
    println!();

    println!("Grid Solution");
    println!("-------------");
    println!("{:>6} {:>14} {:>18} {:>18}", "j", "x_j", "u_j", "exact");

    for j in 0..u.len() {
        let exact = exact_piecewise_solution(problem, x[j]);
        println!("{:>6} {:>14.8} {:>18.10} {:>18.10}", j, x[j], u[j], exact);
    }
}
```

Program 18.7.1 demonstrates how internal boundary conditions can be incorporated directly into a finite-difference relaxation framework by modifying the discretization at specific nodes. This reflects the central idea of Section 18.7.1: that multipoint boundary value problems can be treated within the same global residual formulation used for standard TPBVPs, without requiring separate matching procedures or domain splitting during the solution phase.

The results illustrate that enforcing the interface condition through a single algebraic constraint yields a solution that satisfies both continuity and the prescribed flux jump to machine precision. This confirms the robustness of the relaxation approach when extended to problems with internal constraints. Moreover, the close agreement with the analytical solution highlights the consistency of the discretization with the underlying physical model.

The modular structure of the implementation allows straightforward extension to more complex scenarios, including multiple interfaces, nonlinear coefficients, or higher-order discretizations. This provides a foundation for exploring more advanced formulations, such as collocation methods or Newton-based nonlinear solvers, which further enhance stability and accuracy in challenging multiphysics applications.

## 18.7.2. Singular Points and Regularization Techniques for Boundary Value Problems

Another important class of two-point boundary value problems involves *singular points*, where one or more coefficients of the differential equation become unbounded or degenerate. Such behavior frequently arises in problems expressed in radial coordinates, in scaling limits, or in equations derived from physical conservation laws. The presence of singularities introduces both analytical and numerical challenges, as standard formulations and discretizations may break down near the singular point.

A typical example of a regular singular point occurs in second-order equations of the form:

$$x^2 y'' + x p(x) y' + q(x) y = 0 \tag{18.7.4}$$

At $x=0$, the coefficients multiplying the derivatives vanish or diverge in such a way that the equation remains well-defined in a limiting sense. These points are classified as *regular singular points*, meaning that the solution can still be expressed in a controlled form, although standard Taylor expansions may not apply directly.

A classical analytical technique for handling such problems is the *Frobenius method*. The idea is to factor out the dominant singular behavior by assuming a solution of the form:

$$y(x) = x^\alpha v(x) \tag{18.7.5}$$

where $\alpha$ is determined by an indicial equation, and $v(x)$ is assumed to be analytic in a neighborhood of the singular point. Substituting this expression into the differential equation leads to a modified equation for $v(x)$ in which the leading-order singularity has been removed. This transformation converts the original singular problem into a *regularized problem* that is more amenable to numerical treatment.

In practical computations, one typically avoids evaluating the solution exactly at the singular point. Instead, the numerical domain is truncated to $[\varepsilon,b]$, where $\varepsilon>0$ is small. Initial conditions at $x=\varepsilon$ are then obtained from the series expansion derived via the Frobenius method, ensuring consistency with the analytical behavior of the solution.

A well-known example illustrating these ideas is the *Lane–Emden equation*:

$$y'' + \frac{2}{x}y' + y^n = 0, \qquad y(0)=1, \qquad y'(0)=0 \tag{18.7.6}$$

The coefficient $\frac{2}{x}$ is singular at $x=0$, preventing direct numerical integration from the origin. However, the solution can be expanded near $x=0$ as:

$$y(x) = 1 + a x^2 + O(x^4) \tag{18.7.7}$$

where the coefficient $a$ is determined by substituting the series into the differential equation. This expansion provides consistent initial values at a small distance $x=\varepsilon$, from which numerical integration can proceed reliably.

In *finite-difference or relaxation methods*, singular points are handled through modifications of the discretization near the singularity. One common approach is to use *one-sided difference schemes* that avoid evaluating undefined expressions at the singular point. Another technique involves the introduction of *ghost points*, which allow the enforcement of boundary conditions or symmetry constraints while maintaining the structure of the discretized equations. In all cases, the numerical scheme must be constructed so that it respects the known analytical behavior of the solution near the singular point.

More advanced discretization techniques, such as *collocation methods*, have been shown to be particularly robust in the presence of singularities. Polynomial collocation schemes approximate the solution using global or piecewise polynomial bases and enforce the differential equation at selected collocation points. Because these methods incorporate the solution behavior in an integral or weighted sense, they tend to maintain stability and accuracy even when coefficients are singular. In contrast, finite-difference methods, which rely on local approximations, require more careful treatment to avoid loss of accuracy or numerical instability. Recent studies demonstrate that collocation-based approaches provide reliable performance for regular-singular equations, while finite-difference methods demand tailored discretizations near singular points (Hohenegger et al., 2024).

In summary, the treatment of singular points in TPBVPs requires a combination of analytical insight and numerical adaptation. Techniques such as Frobenius regularization, domain truncation, modified discretization, and collocation-based formulations enable stable and accurate computation in the presence of singular behavior. These methods ensure that the numerical solution remains consistent with the underlying structure of the differential equation, even in regions where standard approaches would fail.

### Rust Implementation

Following the discussion in Section 18.7.2 on the treatment of singular boundary value problems through analytical regularization and domain truncation, Program 18.7.2 provides a practical implementation of the numerical solution of the Lane–Emden equation (18.7.6) using Frobenius-based initialization derived from the local expansion (18.7.7). Since the coefficient $\frac{2}{x}$ becomes singular at $x=0$, direct numerical integration from the origin is not feasible. Instead, the computation is initiated at a small positive value $x=\varepsilon$, where consistent initial conditions are constructed using the series expansion implied by (18.7.7). The resulting regularized problem is then integrated using a classical fourth-order Runge–Kutta method. This implementation illustrates how analytical insight into the local structure of the solution can be combined with standard numerical integration techniques to obtain accurate and stable solutions in the presence of singular behavior.

At the core of the implementation is the `LaneEmdenProblem` struct, which encapsulates the parameters defining the singular boundary value problem, including the polytropic index $n$, the truncation point $\varepsilon$, the final integration point, and the step size. This abstraction allows the numerical configuration to be clearly specified and consistently passed to all computational routines.

The function `regularized_initial_state` implements the Frobenius-based initialization corresponding to the expansion (18.7.7). Rather than evaluating the differential equation at the singular point $x=0$, it computes approximate values of $y(\varepsilon)$ and $y'(\varepsilon)$ using the truncated series expansion. This ensures that the numerical solution begins with values that are consistent with the analytical behavior of the solution near the singularity.

The Lane–Emden equation (18.7.6) is transformed into a first-order system, and its right-hand side is implemented in the function `lane_emden_rhs`. This function evaluates the derivatives of the state variables while explicitly avoiding evaluation at $x=0$. The structure of this function reflects the singular nature of the original equation and enforces the requirement that all computations occur strictly within the truncated domain $[\varepsilon,b]$.

Time stepping is performed using the `rk4_step` function, which implements the classical fourth-order Runge–Kutta method. This method provides a good balance between accuracy and computational efficiency and is well suited for smooth problems such as the regularized Lane–Emden equation. The auxiliary function `add_scaled` is used to simplify the vector arithmetic required in the Runge–Kutta stages, improving code clarity and modularity.

For validation, the function `exact_lane_emden_n1` provides the analytical solution for the special case $n=1$, namely $y(x)=\sin(x)/x$. This allows the numerical solution to be compared directly against the exact solution, providing a quantitative measure of accuracy. The error is monitored throughout the integration, and the maximum deviation is reported.

The `main` function coordinates the overall computation. It initializes the problem, computes the regularized initial state, and advances the solution across the domain using adaptive handling of the final step to ensure that the integration terminates exactly at the prescribed endpoint. Diagnostic output includes intermediate solution values, error comparisons, and final accuracy measures, demonstrating the effectiveness of the regularization approach.

```rust
// Program 18.7.2: Regularized Integration of the Lane-Emden Equation
//
// Problem statement:
// Solve the Lane-Emden equation
//
//     y'' + (2/x)y' + y^n = 0,
//     y(0) = 1,  y'(0) = 0,
//
// without evaluating the singular coefficient 2/x at x = 0.
// The computation starts instead at x = epsilon > 0, where the initial
// values are obtained from the local regular expansion
//
//     y(x)  = 1 - x^2/6 + n x^4/120 + O(x^6),
//     y'(x) = -x/3 + n x^3/30 + O(x^5).
//
// The regularized initial data are then advanced using a fourth-order
// Runge-Kutta method applied to the equivalent first-order system.

#[derive(Clone, Copy)]
struct LaneEmdenProblem {
    n: f64,
    epsilon: f64,
    x_end: f64,
    step_size: f64,
}

#[derive(Clone, Copy)]
struct State {
    y: f64,
    dy: f64,
}

fn regularized_initial_state(problem: LaneEmdenProblem) -> State {
    let x = problem.epsilon;
    let n = problem.n;

    let y = 1.0 - x.powi(2) / 6.0 + n * x.powi(4) / 120.0;
    let dy = -x / 3.0 + n * x.powi(3) / 30.0;

    State { y, dy }
}

fn lane_emden_rhs(x: f64, state: State, n: f64) -> State {
    if x <= 0.0 {
        panic!("The Lane-Emden equation must not be evaluated at x = 0.");
    }

    State {
        y: state.dy,
        dy: -2.0 * state.dy / x - state.y.powf(n),
    }
}

fn add_scaled(a: State, b: State, scale: f64) -> State {
    State {
        y: a.y + scale * b.y,
        dy: a.dy + scale * b.dy,
    }
}

fn rk4_step(x: f64, state: State, h: f64, n: f64) -> State {
    let k1 = lane_emden_rhs(x, state, n);
    let k2 = lane_emden_rhs(x + 0.5 * h, add_scaled(state, k1, 0.5 * h), n);
    let k3 = lane_emden_rhs(x + 0.5 * h, add_scaled(state, k2, 0.5 * h), n);
    let k4 = lane_emden_rhs(x + h, add_scaled(state, k3, h), n);

    State {
        y: state.y + h * (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y) / 6.0,
        dy: state.dy + h * (k1.dy + 2.0 * k2.dy + 2.0 * k3.dy + k4.dy) / 6.0,
    }
}

fn exact_lane_emden_n1(x: f64) -> f64 {
    if x.abs() < 1.0e-14 {
        1.0
    } else {
        x.sin() / x
    }
}

fn main() {
    let problem = LaneEmdenProblem {
        n: 1.0,
        epsilon: 1.0e-4,
        x_end: 3.0,
        step_size: 0.01,
    };

    let mut x = problem.epsilon;
    let mut state = regularized_initial_state(problem);

    let mut max_error: f64 = 0.0;
    let mut step_count = 0usize;

    println!("Regularized Lane-Emden Integration");
    println!("===================================");
    println!();
    println!("Problem Parameters");
    println!("------------------");
    println!("Polytropic index n          = {:.6}", problem.n);
    println!("Starting point epsilon      = {:.6e}", problem.epsilon);
    println!("Final point                 = {:.6}", problem.x_end);
    println!("Nominal step size h         = {:.6}", problem.step_size);
    println!();

    println!("Regularized Initial Data");
    println!("------------------------");
    println!("y(epsilon)                 = {:.12}", state.y);
    println!("y'(epsilon)                = {:.12}", state.dy);
    println!();

    println!("Numerical Solution");
    println!("------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "step", "x", "y_num", "y_exact", "abs_error"
    );

    loop {
        let exact = exact_lane_emden_n1(x);
        let error = (state.y - exact).abs();
        max_error = max_error.max(error);

        if step_count % 25 == 0 || (problem.x_end - x).abs() < 1.0e-14 {
            println!(
                "{:>8} {:>14.8} {:>18.10} {:>18.10} {:>18.3e}",
                step_count, x, state.y, exact, error
            );
        }

        if x >= problem.x_end - 1.0e-14 {
            break;
        }

        let h = if x + problem.step_size > problem.x_end {
            problem.x_end - x
        } else {
            problem.step_size
        };

        state = rk4_step(x, state, h, problem.n);
        x += h;
        step_count += 1;
    }

    println!();
    println!("Diagnostics");
    println!("-----------");
    println!("Steps performed             = {}", step_count);
    println!("Final x                     = {:.8}", x);
    println!("Computed y(final)           = {:.12}", state.y);
    println!("Exact y(final), n = 1       = {:.12}", exact_lane_emden_n1(x));
    println!("Maximum absolute error      = {:.3e}", max_error);
}
```

Program 18.7.2 demonstrates how singular boundary value problems can be handled effectively by combining analytical regularization with standard numerical integration techniques. This reflects the central idea of Section 18.7.2: that singularities do not necessarily require fundamentally different numerical methods, but rather careful adaptation of initial conditions and computational domains to respect the structure of the underlying differential equation.

The results confirm that starting the integration at a small positive value $x=\varepsilon$ and using Frobenius-based initial conditions yields a solution that is both stable and highly accurate. The close agreement with the analytical solution for the case $n=1$ illustrates that the numerical method correctly captures the behavior of the solution across the domain, including regions close to the singular point.

The modular design of the implementation allows for straightforward extension to other values of the polytropic index $n$, as well as to more general singular boundary value problems. This provides a foundation for further exploration of advanced techniques such as adaptive step-size control, higher-order regularization strategies, or collocation-based formulations, which can enhance performance and robustness in more challenging settings.

## 18.7.3. Comparison of Numerical Strategies for Multipoint and Singular Problems

Internal boundary conditions and singular points introduce additional layers of complexity into two-point boundary value problems, but they do not alter the fundamental structure of the numerical solution process. In both cases, the essential strategy consists of either augmenting the system with additional constraints or transforming the problem to remove or regularize singular behavior. The underlying numerical frameworks, such as shooting, relaxation, and collocation, remain applicable with suitable modifications.

For multipoint (internal boundary) problems, the principal challenge lies in enforcing interface conditions. These conditions typically introduce a small number of additional unknowns, such as interface values or flux parameters, which must be determined so that continuity and jump conditions are satisfied. The resulting system includes a low-dimensional nonlinear subsystem associated with the interface, coupled to the main problem. In most cases, the computational cost of solving this auxiliary system is negligible compared to the overall effort required to solve the discretized boundary value problem. The primary concern is therefore not computational complexity but the correct and consistent formulation of the interface conditions within the numerical scheme.

For singular problems, the focus shifts to incorporating the correct asymptotic behavior of the solution near the singular point. Since standard discretizations may fail or lose accuracy in the presence of singular coefficients, the numerical method must reflect the known analytical structure of the solution. This is typically achieved through regularization techniques, such as Frobenius-type transformations, series expansions, or modified discretizations near the singularity. The effectiveness of the method depends on how well these techniques capture the dominant behavior of the solution and integrate it into the numerical formulation.

Among the available numerical approaches, relaxation and collocation methods are particularly well suited to handling both multipoint and singular problems. Their global residual-based formulation allows additional constraints, variable coefficients, and interface conditions to be incorporated directly into the system of equations. This leads to a unified treatment in which all conditions, whether at endpoints or interior points, are enforced simultaneously. The structured sparsity of the resulting system is preserved, enabling efficient computation even in the presence of added complexity.

Shooting methods can also be adapted to these situations, although the modifications are more problem-specific. For multipoint problems, domain decomposition and multiple shooting provide a natural framework for handling interface conditions. For singular problems, shooting requires careful initialization near the singular point, often based on asymptotic expansions, and may involve splitting the domain to avoid unstable propagation. While these adaptations can be effective, they tend to increase the sensitivity of the method and require careful implementation.

In summary, both multipoint boundary conditions and singularities can be handled within the standard numerical frameworks for TPBVPs, provided that appropriate modifications are made. Relaxation and collocation methods offer a particularly flexible and robust approach due to their ability to incorporate additional constraints directly, while shooting methods remain viable when combined with domain decomposition and regularization techniques. The choice of method ultimately depends on the balance between simplicity, stability, and the specific structural features of the problem.

## 18.7.4. Applications Involving Internal Conditions and Singular Behavior

The treatment of internal boundary conditions and singular points is not merely a theoretical extension but a practical necessity in many real-world problems. In applied settings, boundary value problems frequently involve interfaces, discontinuities, or singular geometries that require careful numerical handling to ensure physically meaningful and accurate solutions.

A common class of applications arises in **composite and layered materials**, where different regions of the domain are governed by distinct physical laws or material properties. For example, in heat conduction through a layered medium, the temperature field must satisfy continuity across material interfaces, while the heat flux may exhibit discontinuities depending on material properties. This leads to interface conditions of the type introduced in (18.7.2), where both continuity of the solution and prescribed flux behavior must be enforced. Numerically, such problems are naturally treated using domain decomposition or global residual formulations that incorporate interface conditions directly into the system.

Another important application is found in **astrophysics**, particularly in the modeling of stellar structure. The Lane–Emden equation (18.7.6) describes the radial distribution of density or temperature in a polytropic star. The singularity at the center (x=0) reflects the geometric structure of the problem, and the regularity condition at this point replaces a conventional boundary condition. Accurate numerical treatment requires incorporating the correct asymptotic behavior near the origin, typically through series expansions such as (18.7.7) or through specialized discretization techniques that respect the symmetry of the problem.

Beyond these examples, internal conditions and singularities appear in a wide range of applications, including fluid flow with internal interfaces, electromagnetic problems with material discontinuities, and reaction–diffusion systems with localized sources or sinks. In all such cases, the governing equations must be supplemented by additional constraints that reflect the underlying physical processes.

These examples highlight a key principle: real-world boundary value problems often possess structural features that go beyond simple endpoint conditions. Effective numerical methods must therefore be sufficiently flexible to incorporate internal constraints and singular behavior without compromising stability or accuracy. The techniques discussed in this section including domain decomposition, regularization, modified discretization, and global residual formulations, provide systematic ways to achieve this.

This section completes the treatment of practical complications in two-point boundary value problems. It demonstrates that, although internal conditions and singularities introduce additional complexity, they can be accommodated within the general numerical frameworks developed in this chapter. By extending these frameworks appropriately, one can handle a broad class of problems while preserving the essential structure and robustness of the underlying solution strategies.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/8xNO9AodfRFI8g09DpYw.1","tags":[]}

# 18.8. Conclusion

Throughout this chapter, we have explored the theory and numerical solution of two-point boundary value problems. Unlike initial value problems, boundary value formulations require the solution to satisfy constraints imposed at multiple locations simultaneously, resulting in globally coupled systems that demand specialized numerical techniques. We examined shooting methods, fitting-point and multiple-shooting strategies, relaxation methods, adaptive mesh allocation procedures, and techniques for treating singular points and internal boundary conditions. These methods form a comprehensive framework for solving boundary value problems that arise in applications ranging from mathematical physics and engineering analysis to eigenvalue computations and optimal control. By combining rigorous mathematical foundations with practical Rust implementations, this chapter has provided the tools necessary for developing robust, accurate, and efficient boundary value solvers.

## 18.8.1. Key Takeaways

- Boundary value problems differ fundamentally from initial value problems because information is constrained at multiple points in the computational domain. This global coupling often produces numerical difficulties that require specialized algorithms and careful analysis.
- Residual formulations provide a unifying perspective for boundary value methods by expressing both the differential equations and boundary conditions as a coupled system of nonlinear equations whose solution satisfies the entire problem simultaneously.
- The shooting method transforms a boundary value problem into an initial value problem with unknown parameters. While intuitive and relatively easy to implement, its effectiveness depends strongly on the conditioning and stability properties of the underlying system.
- Newton-based shooting algorithms can be significantly improved through the use of finite-difference Jacobians, variational equations, and sensitivity analysis. Accurate Jacobian information often determines the efficiency and reliability of convergence.
- Fitting-point and multiple-shooting techniques overcome many of the limitations of classical shooting by dividing the domain into smaller segments and enforcing matching conditions. These approaches are particularly useful for stiff and unstable problems.
- Relaxation methods solve the discretized boundary value problem as a globally coupled algebraic system. Their ability to enforce consistency throughout the domain makes them among the most robust techniques for challenging nonlinear problems.
- Adaptive mesh allocation allows computational effort to be concentrated where it is most needed, improving solution accuracy while controlling computational cost. Such techniques are especially valuable for boundary layers, singularities, and rapidly varying solutions.
- Many practical problems involve singular points, multipoint boundary conditions, eigenvalue structures, or coupled differential-algebraic systems. These applications often require extensions of the standard methods discussed throughout the chapter.
- Rust provides a powerful environment for implementing boundary value algorithms, combining high performance with memory safety, expressive abstractions, and modern numerical computing capabilities.

## 18.8.2. Advice for Beginners

- If you are learning boundary value methods for the first time, begin by understanding the conceptual distinction between initial value problems and boundary value problems. Many implementation difficulties arise because boundary value problems cannot generally be solved through simple forward integration alone.
- Start with a simple linear boundary value problem and implement a basic shooting method. This exercise helps develop intuition about how unknown initial conditions influence the satisfaction of boundary conditions at the opposite end of the domain.
- After becoming comfortable with shooting methods, implement a finite-difference relaxation method for the same problem. Comparing the behavior of the two approaches provides valuable insight into conditioning, convergence, and numerical stability.
- Pay particular attention to residuals, Jacobians, and convergence criteria. A numerical method may appear to converge while still producing an inaccurate solution if residuals and boundary conditions are not monitored carefully.
- Experiment with mesh refinement and adaptive mesh allocation. Observing how solution accuracy changes as the mesh is modified is one of the most effective ways to understand numerical error and convergence behavior.
- For Rust implementations, libraries such as `nalgebra`, `ndarray`, and `sprs` provide useful support for matrix computations and sparse linear algebra. Focus first on correctness and clarity before introducing advanced optimizations.
- Most importantly, remember that successful numerical computing depends on understanding both the mathematics and the implementation. Robust software is built upon a solid understanding of the underlying numerical methods.

## 18.8.3. Further Learning with GenAI

To deepen your understanding of boundary value methods and their implementation in Rust, consider exploring the following prompts:

 1. Explain the mathematical differences between initial value problems and two-point boundary value problems, and illustrate them with numerical examples.
 2. Implement a shooting method in Rust for a nonlinear boundary value problem and analyze its convergence under different initial guesses.
 3. Compare finite-difference Jacobians and sensitivity-based Jacobians in shooting methods. Discuss their advantages, disadvantages, and computational costs.
 4. Explain how multiple-shooting methods improve stability compared with classical shooting methods, and provide a detailed implementation strategy.
 5. Visualize the matching conditions used in fitting-point and multiple-shooting methods and explain their role in achieving convergence.
 6. Implement a relaxation-based solver in Rust using finite differences and Newton iteration. Analyze its computational complexity.
 7. Develop an adaptive mesh allocation algorithm based on equidistribution principles and compare its performance with a uniform mesh approach.
 8. Analyze the numerical challenges associated with singular boundary value problems and compare alternative regularization strategies.
 9. Explain the role of boundary value methods in indirect optimal control and derive the associated multipoint boundary conditions.
10. Compare shooting, multiple-shooting, relaxation, and adaptive-mesh methods for a challenging nonlinear boundary value problem and evaluate their relative strengths and weaknesses.

By exploring these prompts, readers can deepen their understanding of both the mathematical foundations and practical implementation aspects of boundary value methods.

## 18.8.4. Homework Exercises

To reinforce your understanding of the material covered in this chapter, complete the following exercises:

 1. Derive the residual equations associated with a second-order linear boundary value problem and show how they can be discretized using finite differences.
 2. Implement a shooting method in Rust for a nonlinear two-point boundary value problem and investigate how convergence depends on the initial parameter estimate.
 3. Compare finite-difference Jacobians and sensitivity-based Jacobians for a shooting algorithm. Evaluate both accuracy and computational efficiency.
 4. Implement a multiple-shooting method and compare its convergence behavior with that of a classical shooting method on a stiff problem.
 5. Analyze the conditioning of a shooting formulation by studying how perturbations in the unknown initial conditions affect the boundary residuals.
 6. Develop a relaxation-based solver using a finite-difference discretization and investigate how computational cost scales with the number of mesh points.
 7. Solve a boundary-layer problem using both a uniform mesh and an adaptive mesh. Compare solution accuracy and computational efficiency.
 8. Implement the spheroidal harmonic eigenvalue problem using both shooting and relaxation methods and compare the numerical results.
 9. Consider a reaction-diffusion model arising in chemical engineering. Formulate the corresponding boundary value problem and analyze the physical significance of the computed solution.
10. Design a numerical framework capable of handling internal boundary conditions or singular points. Compare alternative numerical strategies and justify the method selected.

Boundary value problems are among the most important and widely used mathematical models in scientific computing. The methods developed in this chapter provide a foundation for solving a broad range of practical problems involving equilibrium states, steady transport processes, eigenvalue computations, and optimal control systems. As you continue your study of numerical computing, the ideas introduced here will reappear in increasingly sophisticated forms across many areas of computational science. By combining mathematical insight with careful Rust implementations, you will be well prepared to tackle complex boundary value problems encountered in modern research and engineering practice.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/9UbwzzBeQZvfIyb87O90.2","tags":[]}

# References

 1. Cardiff, P., Armfield, D., Tuković, Ž. and Batistić, I. (2026) ‘A Jacobian-free Newton–Krylov method for cell-centred finite volume solid mechanics’, *International Journal for Numerical Methods in Engineering*, 127(3). doi:10.1002/nme.70268.
 2. Drozd, K., Furfaro, R. and D’Ambrosio, A. (2024) ‘A theory of functional connections-based hp-adaptive mesh refinement algorithm for solving hypersensitive two-point boundary-value problems’, *Mathematics*, 12(9), 1360. doi:10.3390/math12091360.
 3. Eichmeir, P., Nachbagauer, K. and Steiner, W. (2025) ‘The adjoint method for optimal control of multibody systems for free end time and final constraints’, *Multibody System Dynamics*. doi:10.1007/s11044-025-10114-9.
 4. Eichmeir, P. and Steiner, W. (2026) ‘A double shooting method for two-point boundary value problems in multibody dynamics’, *Multibody System Dynamics*. doi:10.1007/s11044-025-10140-7.
 5. Haman, G. V. III and Rao, A. V. (2025) ‘Adaptive mesh refinement and error estimation method for optimal control using direct collocation’, *Journal of Dynamic Systems, Measurement, and Control*, 147(6), 061001. doi:10.1115/1.4068206.
 6. Hohenegger, M., Settanni, G., Weinmüller, E. B. and Wolde, M. (2024) ‘Numerical treatment of singular ODEs using finite difference and collocation methods’, *Applied Numerical Mathematics*, 205, pp. 184–194. doi:10.1016/j.apnum.2024.07.002.
 7. Kaushik, A., Gupta, A., Jain, S., Toprakseven, Ş. and Sharma, M. (2024) ‘An adaptive mesh generation and higher-order difference approximation for the system of singularly perturbed reaction–diffusion problems’, *Partial Differential Equations in Applied Mathematics*, 11, 100750. doi:10.1016/j.padiff.2024.100750.
 8. Krishnakumar, S., Jeyabarathi, P., Abukhaled, M. and Rajendran, L. (2025) ‘A semi-analytical solution of a nonlinear boundary value problem arises in porous catalysts’, *International Journal of Electrochemical Science*, 20(3), 100953. doi:10.1016/j.ijoes.2025.100953.
 9. Luong, K. A., Nguyen-Xuan, H. and Lee, J. (2026) ‘Shooting neural network for multipoint boundary value problems’, *Thin-Walled Structures*, 218, 114146. doi:10.1016/j.tws.2025.114146.
10. Sharan, S., Singla, P., Eapen, R. T. and Melton, R. G. (2025) ‘Higher-order differential correction schemes for the two-point boundary value problem’, *Journal of Guidance, Control, and Dynamics*, 48(7), pp. 1477–1491. doi:10.2514/1.G008381.
11. Soares, A. V.-H., Binous, H., Peixoto, F. C. and Bellagi, A. (2024) ‘Solving boundary value problems in heterogeneous catalysis with orthogonal collocation and arc-length continuation’, *Computer Applications in Engineering Education*, 32(2), e22701. doi:10.1002/cae.22701.
12. Zou, Y. and Jiang, F. (2026) ‘Vectorized sparse second-order forward automatic differentiation for optimal control direct methods’, *Astronautics*, 1(1), 8. doi:10.3390/astronautics1010008.

