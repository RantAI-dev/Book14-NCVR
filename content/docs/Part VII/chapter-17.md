---
weight: 4200
title: "Chapter 17"
description: "Integration of Ordinary Differential Equations"
icon: "article"
date: "2026-07-06T00:00:00+07:00"
lastmod: "2026-07-06T00:00:00+07:00"
katex: true
draft: false
toc: true
---

{{% alert icon="💡" context="info" %}}
<strong>"<em>Mathematics compares the most diverse phenomena and discovers the secret analogies that unite them.</em>" — Joseph Fourier</strong>
{{% /alert %}}

{{% alert icon="📘" context="success" %}}
<p style="text-align: justify;"><em>Chapter 17 presents the fundamental numerical methods used to solve ordinary differential equations and dynamical systems. Beginning with the formulation of initial value problems, the chapter develops explicit and adaptive Runge–Kutta methods, Richardson extrapolation, the Bulirsch–Stoer algorithm, structure-preserving symplectic integrators, stiff-system solvers, multistep and predictor–corrector methods, and stochastic simulation algorithms for chemical reaction networks. Emphasis is placed on the relationship between mathematical structure and algorithm design, showing how accuracy, stability, conservation properties, and computational efficiency influence the selection of numerical methods. Modern developments and Rust implementations demonstrate how robust software can be constructed for scientific computing applications ranging from physics and engineering to systems biology and stochastic modeling.</em></p>
{{% /alert %}}

# 17.1. Introduction

The numerical solution of ordinary differential equations forms a fundamental component of scientific computing, arising naturally in the mathematical modeling of dynamical systems across physics, engineering, chemistry, and biology. In practical applications, analytical solutions are rarely available, and one is therefore compelled to construct reliable numerical approximations that capture the qualitative and quantitative behavior of the underlying system over a finite interval of interest. This chapter develops the core framework for such approximations, beginning with the formulation of the initial value problem and proceeding toward increasingly sophisticated numerical techniques.

From a computational perspective, the primary objective is to replace the continuous evolution of a system by a discrete sequence of approximations that can be efficiently computed while maintaining control over accuracy and stability. This requires careful consideration of local truncation error, global error propagation, and the structural properties of the differential equation itself. The methods introduced in this chapter, particularly one-step schemes, are designed to advance the solution incrementally while preserving essential features such as consistency and convergence. These considerations underpin modern algorithmic developments, including high-order explicit and implicit integrators.

## 17.1.1. The Initial Value Problem

The central mathematical object in this chapter is the initial value problem (IVP) for a system of ordinary differential equations, written in the standard form:

$$\frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0, \quad y : [t_0, T] \to \mathbb{R}^m \tag{17.1.1}$$

In this formulation, the scalar variable $t$ represents the independent variable, typically interpreted as time, while the vector-valued function $y(t) \in \mathbb{R}^m$ denotes the state of the system at time $t$. The function $f(t, y)$ defines the instantaneous rate of change of the state, thereby encoding the governing dynamics of the system. The specification of the initial condition $y(t_0) = y_0$ selects a unique trajectory from the family of possible solutions.

The mapping $y : [t_0, T] \to \mathbb{R}^m$ emphasizes that the solution is sought over a finite interval, which is particularly important in numerical computation, where the interval is discretized into a sequence of points. The dimension $m$ may vary depending on the application, ranging from scalar equations to large systems arising from spatial discretizations of partial differential equations.

For the problem to be well-posed, certain regularity conditions on the function $f$ are required. In particular, it is assumed that $f$ is sufficiently smooth with respect to its arguments to support the derivation of error estimates and order conditions for numerical methods. Additionally, local Lipschitz continuity in the variable $y$ is imposed. This condition ensures that small perturbations in the state do not lead to disproportionately large changes in the rate of evolution, thereby guaranteeing both existence and uniqueness of solutions to the IVP.

From a numerical standpoint, these assumptions are not merely theoretical conveniences; they play a critical role in ensuring that discretization errors remain controlled and that the computed solution approximates a well-defined continuous trajectory. In particular, Lipschitz continuity provides the foundation for bounding the growth of errors as the solution is advanced step by step.

This formulation serves as the standard foundation for modern numerical methods, especially one-step schemes such as Runge–Kutta methods, where the solution at a new time level depends only on information from the current state (Mitsui and Hu, 2023). Such methods are constructed to approximate the integral form of the differential equation over a small interval, using evaluations of $f(t, y)$ to achieve a desired order of accuracy.

## 17.1.2. Reformulation and Modeling Perspective

The formulation given in (17.1.1) provides a unified mathematical framework capable of representing a broad class of dynamical systems arising in both classical and modern applications. One of its most important features is its generality: differential equations of higher order, as well as systems originating from spatially distributed models, can all be systematically transformed into this canonical first-order form. This unification is not merely notational, but has significant implications for numerical computation, since most numerical integration methods are designed specifically for first-order systems.

A fundamental observation is that any higher-order scalar differential equation can be rewritten as a system of first-order equations by introducing additional variables to represent successive derivatives. For example, consider the second-order equation $q''(t) = g(t, q, q')$. By defining an auxiliary variable $v = q'$, the equation can be reformulated as the first-order system:

$$\frac{d}{dt} \begin{bmatrix} q \\ v \end{bmatrix} = \begin{bmatrix}v \\g(t, q, v)\end{bmatrix}, \quad v = q' \tag{17.1.2}$$

This transformation enlarges the state space while preserving the dynamics of the original problem. The resulting system fits directly into the IVP framework, allowing the application of standard numerical methods without modification. From a computational standpoint, this reformulation is essential, since it ensures that algorithms developed for first-order systems can be applied uniformly across problems of varying order. Moreover, it provides a structured representation in which the evolution of each component of the state vector is explicitly defined.

An analogous reduction occurs in the numerical treatment of partial differential equations. When spatial derivatives are discretized using techniques such as finite differences, finite volumes, or finite elements, the continuous problem is converted into a system of ordinary differential equations in time. This approach, known as the method of lines, yields systems of the form:

$$M \frac{du}{dt} = F(t, u), \quad M \in \mathbb{R}^{m \times m} \tag{17.1.3}$$

Here, the vector $u(t)$ represents the discrete approximation of the solution over the spatial domain, while the matrix $M$ encodes structural properties of the discretization, such as mass or weighting effects. In many cases, $M$ is invertible, allowing the system to be written explicitly in the standard IVP form. However, even when this is not done explicitly, the structure remains compatible with numerical integration techniques designed for first-order systems.

Beyond classical physical models, the same mathematical structure appears in contemporary applications. Continuous-time optimization methods, for instance, can be expressed as dynamical systems where the evolution is governed by gradient information. A typical example is given by the gradient flow equation $\dot{x} = -\nabla \Phi(x)$, in which the state evolves in the direction of steepest descent of an objective function. Similarly, in machine learning, neural ordinary differential equations adopt the form $\dot{h} = f_\theta(t, h)$, where the function $f_\theta$ is parameterized and learned from data.

These formulations demonstrate that the IVP framework is not limited to traditional scientific computing, but extends naturally to data-driven modeling and modern computational paradigms. In all such cases, the essential task remains the same: to approximate the evolution of a state governed by a differential relationship over a finite interval. The versatility of the IVP formulation therefore underlies its central role in numerical analysis, providing a common foundation for problems ranging from heat conduction and reaction–diffusion systems to optimization and learning-based models (Saleh, Kovács and Kallur, 2023; Cano and Moreta, 2024; Worsham and Kalita, 2025).

## 17.1.3. Integral Formulation and Numerical Approximation

From a numerical standpoint, the differential formulation of the initial value problem is most effectively interpreted through its equivalent integral representation. This perspective provides the fundamental basis for the construction and analysis of numerical integration methods. Specifically, integrating both sides of (17.1.1) over the interval $[t_n, t_{n+1}]$, one obtains:

$$y(t_{n+1}) = y(t_n) + \int_{t_n}^{t_{n+1}} f(t, y(t))\,dt, \qquad t_{n+1} = t_n + h \tag{17.1.4}$$

This identity is exact and holds for the true solution of the IVP. It expresses the value of the solution at the next time point as the current value plus the accumulated effect of the rate of change over the interval. The step size $h$ determines the length of this interval and therefore directly influences both the accuracy and computational cost of the numerical method.

All one-step numerical methods can be interpreted as approximations to the integral appearing in (17.1.4). The central task is therefore to construct accurate and computationally efficient approximations to this integral using only a finite number of evaluations of the function $f(t, y)$. The simplest such approximation is obtained by replacing the integrand with its value at the left endpoint, leading to Euler’s method. While this approach is computationally inexpensive, it provides only a first-order approximation and may require very small step sizes to achieve acceptable accuracy.

Higher-order methods improve upon this by incorporating additional information about the behavior of $f$ within the interval. Runge–Kutta methods, for example, evaluate $f$ at several intermediate points between $t_n$ and $t_{n+1}$, effectively constructing a more accurate quadrature approximation to the integral. These intermediate evaluations, often referred to as stages, are combined in a carefully designed manner to achieve a desired order of accuracy while maintaining numerical stability.

An important refinement of this framework is the use of adaptive step size control. Rather than fixing $h$ in advance, adaptive methods adjust the step size dynamically based on estimates of the local truncation error. The goal is to ensure that the error at each step remains close to a prescribed tolerance, thereby balancing efficiency and accuracy. When the solution varies rapidly, smaller step sizes are employed to maintain accuracy; when the solution is smooth, larger steps are taken to reduce computational effort.

Recent developments have extended and refined this classical integral viewpoint. Stability-aware step size controllers incorporate information about the stability properties of the numerical method, ensuring that the chosen step size not only controls error but also avoids numerical instabilities. Multirate Runge–Kutta schemes address problems involving multiple time scales by allowing different components of the system to be advanced with different effective step sizes within a unified framework. Adaptive strong stability preserving methods are designed for problems with sharp gradients or discontinuities, such as shock-dominated flows, where maintaining stability properties is essential.

In addition, structure-preserving integrators have been developed to maintain qualitative features of the underlying system, such as invariants or conserved quantities, over long time integrations. These methods reflect a deeper alignment between the numerical scheme and the mathematical structure of the differential equation.

Taken together, these advances demonstrate that the integral formulation (17.1.4) remains central to modern algorithmic design. It provides not only a conceptual foundation for classical methods but also a guiding principle for the development of advanced techniques that address increasingly complex and demanding applications (Ranocha and Giesselmann, 2024; Bachmann et al., 2025; Hu and Wang, 2024; D’Afiero, 2026).

## 17.1.4. Computational Abstraction and Implementation View

The transition from mathematical formulation to executable numerical algorithms requires a clear computational abstraction that preserves the structure of the IVP while enabling flexibility and efficiency in implementation. In practice, especially within a Rust-based numerical framework, it is advantageous to organize solvers around a small number of fundamental components that directly mirror the underlying mathematics. This approach ensures that the implementation remains both transparent and extensible, while maintaining a precise correspondence with the theoretical formulation introduced earlier.

At the core of this abstraction are three essential elements: the state vector $y$, the right-hand side function $f(t, y)$, and the one-step update map:

$$y_{n+1} = \Psi_h(t_n, y_n) \tag{17.1.5}$$

The state vector $y$ represents the evolving solution and is typically implemented as a structured numerical object, such as a fixed-size or dynamically sized vector. Its representation must support efficient arithmetic operations, since vector updates are performed repeatedly throughout the integration process. The function $f(t, y)$ encapsulates the model itself, providing the mapping from the current state and time to the instantaneous rate of change. In a Rust setting, this is naturally expressed as a function or closure, allowing different models to be supplied without altering the solver infrastructure.

The update map $\Psi_h$ defines the numerical method. It specifies how the solution is advanced from one time level to the next using a step size $h$. Conceptually, $\Psi_h$ approximates the exact flow implied by the integral formulation (17.1.4). Different numerical schemes correspond to different realizations of this map. For example, explicit Runge–Kutta methods construct $\Psi_h$ through a sequence of intermediate evaluations of $f$, while other methods may incorporate additional structural or stability considerations.

This separation of concerns is critical for modularity. By isolating the state representation, the model definition, and the integration mechanism, one can modify or extend each component independently. Numerical ingredients such as Runge–Kutta tableaux, which determine the coefficients used in stage evaluations, can be altered without changing the definition of $f$ or the structure of the state vector. Similarly, error estimators used for adaptive step size control can be integrated into the update process without affecting the core solver logic.

The same modular structure also accommodates enhancements such as dense-output interpolants, which provide continuous approximations of the solution between discrete time steps, and variations in floating-point precision, which may be required for high-accuracy or performance-critical applications. Because these features are incorporated at well-defined points in the abstraction, they do not disrupt the overall organization of the solver.

From a software engineering perspective, this design reflects modern numerical practice, where clarity, flexibility, and performance must coexist. It allows systematic experimentation with different numerical methods and parameter choices while preserving a clean and direct mapping between the mathematical formulation and its computational realization. As a result, the implementation remains both robust and adaptable, supporting a wide range of applications without sacrificing conceptual coherence.

### Rust Implementation

Following the discussion in Section 17.1.4 on the computational abstraction of initial value problems, Program 17.1.1 provides a concrete implementation of the core components underlying numerical time integration. The formulation of the IVP in Equation (17.1.1), together with its integral representation in Equation (17.1.4) and the abstract update map in Equation (17.1.5), naturally leads to a modular computational structure consisting of a state vector, a right-hand-side function, and a one-step evolution operator. This program realizes that structure using a simple explicit method, thereby illustrating how continuous dynamical systems can be approximated through discrete updates. A second-order differential equation is first reformulated as a first-order system as in Equation (17.1.2), demonstrating the generality of the IVP framework and its direct compatibility with numerical solvers.

At the core of the implementation is the representation of the state vector $y$, which encodes the evolving solution of the IVP defined in Equation (17.1.1). In the program, this is implemented as a dynamic vector of floating-point values, allowing systems of arbitrary dimension to be handled uniformly. The right-hand-side function $f(t, y)$, which defines the rate of change of the state, is expressed as a closure. This design reflects the abstraction introduced in Section 17.1.4, where the model is separated from the numerical method, enabling flexibility in defining different dynamical systems without modifying the solver infrastructure.

The function `vector_add_scaled` provides a fundamental linear algebra operation required for time stepping. It computes an expression of the form $y + h f(t,y)$, which corresponds directly to the increment appearing in the integral formulation of the IVP in Equation (17.1.4). Although simple, this operation encapsulates the essential structure of many numerical methods, where the next state is obtained by combining the current state with a scaled evaluation of the derivative.

The function `euler_step` implements the one-step update map $\Psi_h(t_n, y_n)$ introduced in Equation (17.1.5). Specifically, it evaluates the right-hand side at the current state and advances the solution using a forward Euler approximation. This represents the simplest realization of the integral approximation discussed in Section 17.1.3, where the integral in Equation (17.1.4) is replaced by a single evaluation at the left endpoint. While low-order, this method clearly demonstrates how the abstract update map is translated into executable code.

The `solve_ivp` function constructs the full discrete trajectory by repeatedly applying the update map. Starting from the initial condition specified in Equation (17.1.1), it advances the solution over successive time steps, storing the pair $(t_n, y_n)$ at each stage. This iterative process reflects the step-by-step propagation of the solution emphasized throughout Section 17.1 and provides a direct computational realization of the discrete approximation to the continuous trajectory.

To demonstrate the framework, the program includes an example based on the second-order differential equation reformulated as a first-order system in Equation (17.1.2). The harmonic oscillator serves as a canonical test problem, where the state vector consists of position and velocity components. This example highlights how higher-order equations are systematically transformed into first-order systems and then integrated using the same numerical infrastructure.

The `main` function initializes the problem, defines numerical parameters such as the step size and number of steps, and invokes the solver. It then prints the computed trajectory in a structured format, allowing direct inspection of the numerical solution. This demonstrates not only the correctness of the implementation but also the practical workflow of defining, solving, and analyzing an initial value problem within a unified computational framework.

```rust
// Program 17.1.1: A Minimal Computational Abstraction for an Initial Value Problem
//
// Problem statement:
// This program demonstrates the basic computational structure of an initial value
// problem in the form
//
//     dy/dt = f(t, y),    y(t0) = y0,
//
// by representing the state as a vector, the model as a right-hand-side function,
// and the numerical method as a one-step update map Psi_h.  Euler's method is
// used only as the simplest possible example of such an update map.

type State = Vec<f64>;

fn vector_add_scaled(y: &State, scale: f64, dy: &State) -> State {
    y.iter()
        .zip(dy.iter())
        .map(|(yi, dyi)| yi + scale * dyi)
        .collect()
}

fn euler_step<F>(f: &F, t: f64, y: &State, h: f64) -> State
where
    F: Fn(f64, &State) -> State,
{
    let dydt = f(t, y);
    vector_add_scaled(y, h, &dydt)
}

fn solve_ivp<F>(
    f: F,
    t0: f64,
    y0: State,
    h: f64,
    n_steps: usize,
) -> Vec<(f64, State)>
where
    F: Fn(f64, &State) -> State,
{
    let mut solution = Vec::with_capacity(n_steps + 1);

    let mut t = t0;
    let mut y = y0;

    solution.push((t, y.clone()));

    for _ in 0..n_steps {
        y = euler_step(&f, t, &y, h);
        t += h;
        solution.push((t, y.clone()));
    }

    solution
}

fn main() {
    println!("Initial Value Problem Abstraction");
    println!("=================================\n");

    // Example:
    // Convert the second-order equation
    //
    //     q''(t) = -q(t)
    //
    // into the first-order system
    //
    //     q' = v,
    //     v' = -q.
    //
    // The state vector is y = [q, v].
    let harmonic_oscillator = |_: f64, y: &State| -> State {
        let q = y[0];
        let v = y[1];

        vec![v, -q]
    };

    let t0 = 0.0;
    let y0 = vec![1.0, 0.0];
    let h = 0.05;
    let n_steps = 20;

    let solution = solve_ivp(harmonic_oscillator, t0, y0, h, n_steps);

    println!("Model");
    println!("-----");
    println!("Second-order equation: q''(t) = -q(t)");
    println!("First-order state:     y = [q, v]");
    println!("Right-hand side:       f(t, y) = [v, -q]\n");

    println!("Numerical Parameters");
    println!("--------------------");
    println!("Initial time t0        = {:.6}", t0);
    println!("Step size h            = {:.6}", h);
    println!("Number of steps        = {}", n_steps);
    println!("Initial state y0       = [1.000000, 0.000000]\n");

    println!("Computed Solution");
    println!("-----------------");
    println!("{:>10} {:>18} {:>18}", "t", "q(t)", "v(t)");

    for (t, y) in solution {
        println!("{:>10.4} {:>18.10} {:>18.10}", t, y[0], y[1]);
    }
}
```

Program 17.1.1 demonstrates the essential computational structure underlying the numerical solution of initial value problems. By implementing the state representation, the right-hand-side function, and the one-step update map in a modular fashion, the program reflects the abstraction introduced in Section 17.1.4 and provides a clear bridge between mathematical formulation and executable algorithm.

The use of Euler’s method illustrates the simplest possible realization of the integral approximation in Equation (17.1.4), making explicit how discrete updates approximate continuous evolution. Although this method is limited in accuracy and stability, it serves as a transparent foundation upon which more advanced methods, such as Runge–Kutta schemes and adaptive step size integrators, can be constructed.

The harmonic oscillator example further emphasizes the importance of reformulating higher-order equations into first-order systems, as described in Equation (17.1.2), enabling uniform treatment across a wide range of applications. The resulting framework is readily extensible, allowing alternative update maps, error control strategies, and higher-order methods to be incorporated without altering the core structure.

This program therefore establishes the computational foundation for the numerical methods developed in subsequent sections, where more sophisticated integration techniques will build upon the same abstraction to achieve higher accuracy, improved stability, and greater efficiency.

# 17.2. Runge–Kutta Methods for One-Step Time Integration

Runge–Kutta methods form a central class of one-step integration schemes for solving initial value problems. Their formulation is closely tied to the integral representation introduced in Section 17.1, and they provide a systematic mechanism for constructing high-order accurate approximations using only evaluations of the right-hand side function. This makes them particularly well suited for problems in which the governing dynamics are available in functional form but higher derivatives are not easily accessible.

A key strength of Runge–Kutta methods lies in their balance between generality and structure. They can be applied to a wide range of problems without modification, yet their internal organization allows precise control over accuracy and stability through carefully chosen coefficients. As a result, they serve as a foundational tool in both classical numerical analysis and modern computational applications.

## 17.2.1. General Formulation of Explicit Runge–Kutta Schemes

Runge–Kutta methods arise from approximating the integral representation (17.1.4) by evaluating the right-hand side function at several carefully chosen points within a single time step. For an explicit $s$-stage Runge–Kutta method, the stage derivatives are defined by:

$$k_i = f\!\left(t_n + c_i h,\; y_n + h \sum_{j=1}^{i-1} a_{ij} k_j \right), \qquad i = 1, \dots, s \tag{17.2.1}$$

and the solution is advanced according to:

$$y_{n+1} = y_n + h \sum_{i=1}^{s} b_i k_i \tag{17.2.2}$$

In this representation, the coefficients $a_{ij}$, $b_i$, and $c_i$ define the structure of the method and are typically organized in a tabular form known as the Butcher tableau. The step size $h$ determines the width of the interval over which the solution is advanced. Each stage derivative $k_i$ approximates the slope of the exact solution at an intermediate point within the interval, with the location of that point determined by $c_i$.

The explicit nature of the method is reflected in the summation limits in (17.2.1), where the computation of each stage depends only on previously computed stages. This recursive structure eliminates the need to solve algebraic systems during each time step, making explicit Runge–Kutta methods computationally efficient and straightforward to implement.

The stage construction described above admits a clear geometric interpretation. Rather than relying on a single slope evaluation, the method samples the vector field at several intermediate points within the step and combines these contributions to approximate the solution increment. In particular, each stage $k_i$ corresponds to a slope evaluation at a different point within the interval, and the final update combines these slopes to approximate the evolution of the solution over one step.

<div class="row justify-content-center">
    <div class="rounded p-4 position-relative overflow-hidden border-1 text-center" style="width: 50%">
        {{< figure src="/images/pqQDe4beUu67RvW3raYP-sYbhx8P9deWUJKVX4M0L-v1.png" >}}
        <p>**Figure 17.2.1:** Geometric interpretation of a Runge–Kutta step using multiple intermediate stage slopes within a single timestep.</p>
    </div>
</div>

As shown in Figure 17.2.1, the intermediate stage slopes provide local information about the vector field, and their combination yields a higher-order approximation to the integral representation of the solution.

From the viewpoint of the integral formulation, the update formula (17.2.2) can be interpreted as a weighted quadrature approximation to the integral in (17.1.4). The coefficients $b_i$ serve as weights, while the stage evaluations provide sampled values of the integrand at intermediate points. The accuracy of this approximation depends critically on the choice of coefficients, which are selected to satisfy a set of order conditions ensuring that the numerical solution matches the Taylor expansion of the exact solution up to a prescribed order.

The flexibility inherent in this formulation allows the construction of methods ranging from low-order schemes with minimal computational cost to high-order methods that achieve greater accuracy through additional stage evaluations. This adaptability is one of the principal reasons for the widespread use of Runge–Kutta methods in practice.

The formulation presented here is standard in modern numerical analysis and remains central in contemporary applications. It underpins numerical schemes used in neural ordinary differential equations and reduced-order modeling, where efficient and accurate time integration plays a critical role (Mitsui and Hu, 2023; Caldana and Hesthaven, 2024).

## 17.2.2. Butcher Tableau and Algebraic Structure of Runge–Kutta Methods

The coefficients defining a Runge–Kutta method admit a compact and highly structured representation known as the Butcher tableau. This tabular form provides a unified way to encode all parameters appearing in the stage equations (17.2.1) and the update formula (17.2.2), while simultaneously exposing the internal algebraic relationships that govern the behavior of the method. The tableau is written as:

$$\begin{array}{c|cccc}c_1 & \\c_2 & a_{21} \\c_3 & a_{31} & a_{32} \\ \vdots & \vdots & \vdots & \ddots \\c_s & a_{s1} & a_{s2} & \cdots & a_{s,s-1} \\ \hline& b_1 & b_2 & \cdots & b_s\end{array} \tag{17.2.3}$$

In this representation, the first column lists the nodes $c_i$, which determine the locations of the intermediate evaluation points within each time step. The strictly lower triangular matrix $(a_{ij})$ encodes how previously computed stage derivatives contribute to the construction of subsequent stages. The bottom row contains the weights $b_i$, which define how the stage derivatives are combined to produce the final update.

For explicit Runge–Kutta methods, the tableau has a strictly lower triangular structure, reflecting the fact that each stage depends only on earlier stages. In this case, the coefficients satisfy the row-sum relation $c_i = \sum_{j<i} a_{ij},$ which ensures consistency between the stage definitions and the temporal locations at which the function $f$ is evaluated. This relation arises naturally from the interpretation of the method as an approximation to the integral formulation (17.1.4), where the stage points must align with the accumulated contributions of the preceding increments.

The Butcher tableau serves not only as a convenient notation but also as a fundamental tool for analyzing and constructing Runge–Kutta methods. The order of accuracy of a method is determined by a set of algebraic conditions imposed on the coefficients $a_{ij}$, $b_i$, and $c_i$. These conditions ensure that the numerical solution reproduces the Taylor expansion of the exact solution up to a desired order. The tableau provides a systematic framework in which these order conditions can be expressed and verified.

In addition to accuracy, the tableau encodes information about stability properties. For example, the distribution of the coefficients influences how the method behaves when applied to stiff or oscillatory problems. The structure of the tableau also determines storage requirements and computational complexity, since the number of stages directly impacts the number of function evaluations per time step.

Another important aspect captured by the tableau is the possibility of embedded methods. By augmenting the set of weights $b_i$ with an additional set of coefficients, one can construct a secondary approximation of different order without additional stage evaluations. This forms the basis for adaptive step size control, where the difference between the two approximations provides an estimate of the local truncation error.

Modern research continues to extend this algebraic framework. Investigations into weak stage order address situations in which certain components of the method achieve lower effective accuracy, leading to phenomena such as order reduction in stiff or highly constrained systems. Multiderivative generalizations expand the tableau structure to incorporate higher derivatives of the solution, thereby enriching the class of available methods. These developments demonstrate that the Butcher tableau is not merely a static representation, but an active area of study that continues to inform the design of advanced numerical integrators (Biswas et al., 2025; Qin et al., 2024).

## 17.2.3. Order Conditions and Accuracy of Runge–Kutta Methods

The accuracy of a Runge–Kutta method is determined by how closely its discrete update reproduces the behavior of the exact solution over a single time step. This is quantified through the concept of order, which measures the degree to which the numerical approximation matches the Taylor expansion of the exact solution. The derivation of order conditions therefore proceeds by expanding both the exact solution and the numerical update in powers of the step size $h$, and then equating corresponding terms.

Starting from the integral formulation (17.1.4), the exact solution admits a Taylor expansion about $t_n$ in which successive terms involve higher derivatives of $y(t)$, and hence higher derivatives of the function $f(t, y)$. The numerical update (17.2.2), when expanded in a similar manner using the stage definitions (17.2.1), produces a series whose coefficients depend entirely on the parameters $a_{ij}$, $b_i$, and $c_i$. The requirement that the numerical series agree with the exact series up to a given order leads to a system of algebraic constraints on these coefficients.

For consistency and low-order accuracy, the coefficients must satisfy the conditions:

$$\begin{align} &\sum_{i=1}^{s} b_i = 1, \\ &\sum_{i=1}^{s} b_i c_i = \frac{1}{2}, \\ &\sum_{i=1}^{s} b_i c_i^2 = \frac{1}{3}, \\ &\sum_{i,j} b_i a_{ij} c_j = \frac{1}{6} \end{align} \tag{17.2.4}$$

which ensure that the method reproduces the Taylor expansion of the exact solution up to third order. Each of these conditions corresponds to matching a specific term in the expansion. The first condition enforces basic consistency, guaranteeing that the method correctly integrates constant functions. The subsequent conditions ensure that linear and quadratic variations in the solution are captured with increasing accuracy.

To achieve higher-order accuracy, additional constraints must be imposed. For example, fourth-order accuracy requires conditions such as:

$$\begin{align} &\sum_i b_i c_i^3 = \frac{1}{4}, \\ &\sum_{i,j} b_i c_i a_{ij} c_j = \frac{1}{8}, \\ &\sum_{i,j} b_i a_{ij} c_j^2 = \frac{1}{12}, \\ &\sum_{i,j,\ell} b_i a_{ij} a_{j\ell} c_\ell = \frac{1}{24} \end{align} \tag{17.2.5}$$

These relations reflect increasingly intricate interactions among the coefficients, involving nested sums that correspond to higher-order derivative terms in the Taylor expansion. As the desired order increases, the number and complexity of these conditions grow rapidly, making manual derivation of high-order methods impractical. Nevertheless, the structure of these conditions provides essential insight into how the coefficients control the accuracy of the method.

A key principle emerges from this analysis: the construction of Runge–Kutta methods reduces to solving an algebraic matching problem between series expansions. The coefficients are chosen so that the numerical method reproduces the behavior of the exact solution up to a prescribed order in $h$. This viewpoint explains both the power and the limitations of the approach. On one hand, it provides a systematic path to designing methods of arbitrary order. On the other hand, it reveals the increasing algebraic complexity associated with higher-order schemes.

Although in practice high-order Runge–Kutta methods are typically obtained from established tables or generated using symbolic or computational tools, understanding these order conditions remains essential. It clarifies the origin of the coefficients, explains the relationship between stage structure and accuracy, and provides a foundation for analyzing extensions and generalizations of the method (Mitsui and Hu, 2023; Biswas et al., 2025).

## 17.2.4. Classical Runge–Kutta Schemes and Their Properties

To illustrate the general formulation and algebraic structure developed in the preceding sections, it is instructive to examine specific Runge–Kutta methods that are widely used in practice. These classical schemes demonstrate how the abstract coefficients $a_{ij}$, $b_i$, and $c_i$ translate into concrete computational procedures, and how the order conditions manifest in explicit algorithms.

### The Explicit Midpoint Method (Second-Order Scheme)

A simple yet important example of a second-order Runge–Kutta method is the explicit midpoint method, defined by:

$$
\begin{aligned}
k_1 &= f(t_n, y_n), \\
k_2 &= f\!\left(t_n + \frac{h}{2},\; y_n + \frac{h}{2} k_1 \right), \\
y_{n+1} &= y_n + h k_2
\end{aligned}
\tag{17.2.6}
$$

This method employs two stages within each time step. The first stage evaluates the slope at the initial point $(t_n, y_n)$, providing a basic estimate of the local behavior of the solution. The second stage then uses this information to approximate the solution at the midpoint $t_n + \frac{h}{2}$, where a refined slope is computed. The final update uses this midpoint slope to advance the solution over the full interval.

From the perspective of the integral formulation (17.1.4), this procedure corresponds to approximating the integral using a midpoint quadrature rule. By sampling the integrand at a representative interior point rather than solely at the endpoint, the method captures additional information about the variation of $f(t, y)$ over the interval. As a result, it achieves second-order accuracy, improving upon the first-order behavior of Euler’s method while retaining a relatively low computational cost.

### The Classical Fourth-Order Runge–Kutta Method (RK4)

The most widely known higher-order Runge–Kutta scheme is the classical fourth-order method, commonly referred to as RK4. It is defined by:

$$
\begin{aligned}
k_1 &= f(t_n, y_n), \\
k_2 &= f\!\left(t_n + \frac{h}{2},\; y_n + \frac{h}{2} k_1 \right), \\
k_3 &= f\!\left(t_n + \frac{h}{2},\; y_n + \frac{h}{2} k_2 \right), \\
k_4 &= f\!\left(t_n + h,\; y_n + h k_3 \right), \\
y_{n+1} &= y_n + \frac{h}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)
\end{aligned}
\tag{17.2.7}
$$

This method uses four stages to construct a highly accurate approximation to the integral in (17.1.4). The stages are distributed symmetrically within the interval, with two evaluations at the midpoint and one at the endpoint. The weighted combination of slopes in the final update reflects a carefully balanced quadrature rule, where the coefficients $1, 2, 2, 1$ ensure that the order conditions up to fourth order are satisfied.

For sufficiently smooth functions $f$, the RK4 method achieves a local truncation error of order $O(h^5)$ and a global error of order $O(h^4)$. This level of accuracy, combined with its relatively simple structure, has made RK4 a standard reference method in both teaching and practice. The method provides a clear illustration of how multiple stage evaluations can be combined to approximate higher-order terms in the Taylor expansion without explicitly computing derivatives beyond the first.

From a structural viewpoint, RK4 exemplifies the principles discussed in Section 17.2.3. The coefficients satisfy the required order conditions, and the method can be represented compactly using a Butcher tableau. Its symmetry and balanced weighting contribute to its favorable accuracy properties, while its explicit nature ensures ease of implementation.

Despite these advantages, modern numerical solvers often favor embedded Runge–Kutta methods with adaptive step size control. Such methods extend the RK framework by incorporating error estimation mechanisms that allow the step size to vary dynamically, thereby improving efficiency for problems with nonuniform behavior. Nevertheless, classical schemes such as the midpoint method and RK4 remain fundamental, both as practical tools for moderate accuracy requirements and as illustrative examples of the underlying theory (Mitsui and Hu, 2023; Worsham and Kalita, 2025).

### Rust Implementation

Following the development of explicit Runge–Kutta schemes and their algebraic structure in Section 17.2.4, Program 17.2.1 provides a concrete implementation of two classical methods: the explicit midpoint method defined in Equation (17.2.6) and the classical fourth-order Runge–Kutta method defined in Equation (17.2.7). These methods exemplify how the abstract coefficients $a_{ij}, b_i, c_i$ introduced earlier translate into practical computational procedures through staged evaluations of the right-hand side function. By applying both schemes to a representative second-order system reformulated as a first-order IVP, the program highlights the relationship between stage construction, quadrature interpretation of the integral formulation (17.1.4), and the resulting order of accuracy.

At the core of the implementation is the representation of the state vector $y$, which corresponds to the solution of the initial value problem defined in Equation (17.1.1). The right-hand side function $f(t,y)$ is implemented as a closure, reflecting the abstraction introduced in Section 17.1.4. This design allows different dynamical systems to be integrated using the same solver infrastructure, preserving modularity and extensibility.

The function `add_scaled` implements a basic vector operation of the form $y + \alpha k$, which appears repeatedly in the stage definitions of both methods in Equations (17.2.6) and (17.2.7). This operation corresponds directly to the incremental construction of intermediate states used to evaluate stage derivatives. The function `add_weighted4` performs the weighted combination of stage derivatives required in Equation (17.2.7), where the coefficients $1, 2, 2, 1$ are applied to $k_1, k_2, k_3, k_4$, respectively. This reflects the quadrature-based interpretation of the update formula as an approximation to the integral in Equation (17.1.4).

The function `midpoint_step` implements the explicit midpoint method exactly as described in Equation (17.2.6). It computes the first stage derivative $k_1$ at the initial point, constructs an intermediate state at the midpoint of the interval, and then evaluates the second stage derivative $k_2$ at this midpoint. The final update uses $k_2$ to advance the solution, demonstrating how midpoint sampling improves accuracy relative to single-point methods.

The function `rk4_step` implements the classical fourth-order Runge–Kutta method defined in Equation (17.2.7). It computes four stage derivatives at strategically chosen points within the interval and combines them using a weighted average. This sequence of operations reflects the satisfaction of higher-order conditions discussed in Section 17.2.3, allowing the method to achieve fourth-order accuracy without explicitly computing higher derivatives.

The `solve` function provides a generic time-stepping loop that repeatedly applies a given step function. This abstraction mirrors the one-step update map introduced in Equation (17.1.5), allowing different Runge–Kutta methods to be used interchangeably by passing different step functions. The function stores the computed trajectory, enabling both final-time accuracy analysis and step-by-step comparison.

To validate the implementation, the program includes the harmonic oscillator example, which arises from reformulating a second-order differential equation as a first-order system in Equation (17.1.2). The exact solution is known analytically, allowing the functions `exact_solution` and `max_abs_error` to quantify the numerical error. This comparison provides a direct illustration of the difference in accuracy between the second-order midpoint method and the fourth-order RK4 method.

The `main` function initializes the problem, defines numerical parameters, and computes solutions using both methods. It then compares the numerical results with the exact solution at the final time and prints a step-by-step comparison of the trajectories. This demonstrates not only correctness but also the practical impact of method order on numerical accuracy.

```rust
// Program 17.2.1: Explicit Midpoint and Classical RK4 Methods for One-Step Integration
//
// Problem statement:
// This program implements two classical explicit Runge-Kutta methods for the
// initial value problem
//
//     dy/dt = f(t, y),    y(t0) = y0.
//
// The explicit midpoint method follows Equation (17.2.6), while the classical
// fourth-order Runge-Kutta method follows Equation (17.2.7). The same harmonic
// oscillator test problem is solved by both methods so that their different
// accuracy levels can be compared directly.

type State = Vec<f64>;

fn add_scaled(y: &State, scale: f64, k: &State) -> State {
    y.iter()
        .zip(k.iter())
        .map(|(yi, ki)| yi + scale * ki)
        .collect()
}

fn add_weighted4(
    y: &State,
    h: f64,
    k1: &State,
    k2: &State,
    k3: &State,
    k4: &State,
) -> State {
    y.iter()
        .zip(k1.iter())
        .zip(k2.iter())
        .zip(k3.iter())
        .zip(k4.iter())
        .map(|((((yi, k1i), k2i), k3i), k4i)| {
            yi + h * (k1i + 2.0 * k2i + 2.0 * k3i + k4i) / 6.0
        })
        .collect()
}

fn midpoint_step<F>(f: &F, t: f64, y: &State, h: f64) -> State
where
    F: Fn(f64, &State) -> State,
{
    let k1 = f(t, y);
    let y_mid = add_scaled(y, 0.5 * h, &k1);
    let k2 = f(t + 0.5 * h, &y_mid);

    add_scaled(y, h, &k2)
}

fn rk4_step<F>(f: &F, t: f64, y: &State, h: f64) -> State
where
    F: Fn(f64, &State) -> State,
{
    let k1 = f(t, y);

    let y_stage2 = add_scaled(y, 0.5 * h, &k1);
    let k2 = f(t + 0.5 * h, &y_stage2);

    let y_stage3 = add_scaled(y, 0.5 * h, &k2);
    let k3 = f(t + 0.5 * h, &y_stage3);

    let y_stage4 = add_scaled(y, h, &k3);
    let k4 = f(t + h, &y_stage4);

    add_weighted4(y, h, &k1, &k2, &k3, &k4)
}

fn solve<F, S>(
    f: &F,
    stepper: S,
    t0: f64,
    y0: State,
    h: f64,
    n_steps: usize,
) -> Vec<(f64, State)>
where
    F: Fn(f64, &State) -> State,
    S: Fn(&F, f64, &State, f64) -> State,
{
    let mut trajectory = Vec::with_capacity(n_steps + 1);

    let mut t = t0;
    let mut y = y0;

    trajectory.push((t, y.clone()));

    for _ in 0..n_steps {
        y = stepper(f, t, &y, h);
        t += h;
        trajectory.push((t, y.clone()));
    }

    trajectory
}

fn exact_solution(t: f64) -> State {
    vec![t.cos(), -t.sin()]
}

fn max_abs_error(numerical: &State, exact: &State) -> f64 {
    numerical
        .iter()
        .zip(exact.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn print_final_comparison(
    label: &str,
    trajectory: &[(f64, State)],
) {
    let (t_final, y_final) = trajectory.last().expect("trajectory is nonempty");
    let exact = exact_solution(*t_final);
    let error = max_abs_error(y_final, &exact);

    println!("{label}");
    println!("{}", "-".repeat(label.len()));
    println!("Final time t                 = {:.6}", t_final);
    println!("Computed q(t)                = {:.12}", y_final[0]);
    println!("Computed v(t)                = {:.12}", y_final[1]);
    println!("Exact q(t)                   = {:.12}", exact[0]);
    println!("Exact v(t)                   = {:.12}", exact[1]);
    println!("Maximum absolute error       = {:.6e}", error);
    println!();
}

fn main() {
    println!("Classical Explicit Runge-Kutta Methods");
    println!("======================================\n");

    let harmonic_oscillator = |_: f64, y: &State| -> State {
        let q = y[0];
        let v = y[1];

        vec![v, -q]
    };

    let t0 = 0.0;
    let y0 = vec![1.0, 0.0];
    let h = 0.1;
    let n_steps = 10;

    println!("Model Problem");
    println!("-------------");
    println!("Second-order equation: q''(t) = -q(t)");
    println!("First-order system:    q' = v,   v' = -q");
    println!("Initial state:         q(0) = 1, v(0) = 0\n");

    println!("Numerical Parameters");
    println!("--------------------");
    println!("Initial time t0        = {:.6}", t0);
    println!("Step size h            = {:.6}", h);
    println!("Number of steps        = {}", n_steps);
    println!("Final time             = {:.6}\n", t0 + h * n_steps as f64);

    let midpoint_solution = solve(
        &harmonic_oscillator,
        midpoint_step,
        t0,
        y0.clone(),
        h,
        n_steps,
    );

    let rk4_solution = solve(
        &harmonic_oscillator,
        rk4_step,
        t0,
        y0,
        h,
        n_steps,
    );

    println!("Final-Time Accuracy Comparison");
    println!("------------------------------\n");

    print_final_comparison("Explicit Midpoint Method", &midpoint_solution);
    print_final_comparison("Classical Fourth-Order Runge-Kutta Method", &rk4_solution);

    println!("Step-by-Step Comparison");
    println!("-----------------------");
    println!(
        "{:>8} {:>16} {:>16} {:>16} {:>16}",
        "t", "q_midpoint", "v_midpoint", "q_rk4", "v_rk4"
    );

    for ((t_mid, y_mid), (_, y_rk4)) in midpoint_solution.iter().zip(rk4_solution.iter()) {
        println!(
            "{:>8.4} {:>16.10} {:>16.10} {:>16.10} {:>16.10}",
            t_mid, y_mid[0], y_mid[1], y_rk4[0], y_rk4[1]
        );
    }
}
```

Program 17.2.1 demonstrates how classical Runge–Kutta methods translate the abstract formulation of staged derivatives and weighted updates into concrete computational algorithms. The explicit midpoint method illustrates the transition from first-order to second-order accuracy through midpoint sampling, while the RK4 method shows how multiple carefully weighted stage evaluations achieve significantly higher accuracy.

The numerical results clearly reflect the theoretical order properties discussed in Section 17.2.3. The midpoint method exhibits moderate error consistent with second-order convergence, whereas RK4 produces highly accurate results with substantially reduced error. This comparison highlights the trade-off between computational cost and accuracy, as higher-order methods require additional function evaluations but achieve superior precision.

The modular structure of the implementation, particularly the separation between the solver and the step functions, reflects the computational abstraction introduced earlier and provides a foundation for extending the framework. More advanced Runge–Kutta methods, including embedded and adaptive schemes, can be incorporated by modifying the stage definitions and update rules while preserving the overall structure. This program therefore serves as a bridge between the theoretical development of Runge–Kutta methods and their practical realization, preparing the groundwork for more sophisticated time integration strategies discussed in subsequent sections.

## 17.2.5. Stability and Computational Properties of Runge–Kutta Methods

The practical effectiveness of a Runge–Kutta method depends not only on its order of accuracy but also on its stability behavior and computational efficiency. These aspects determine whether a method can reliably approximate solutions over long time intervals and whether it remains feasible for large-scale problems. A systematic analysis of stability begins with a canonical model problem, while computational considerations focus on the cost and storage requirements associated with each time step.

A standard approach to analyzing stability is to apply the numerical method to the linear test equation:

$$y' = \lambda y, \qquad z = h\lambda \tag{17.2.8}$$

This equation serves as a simplified model for the local behavior of more general systems, particularly near equilibrium points. When a Runge–Kutta method is applied to this equation, the update takes the form:

$$y_{n+1} = R(z)\, y_n, \quad R(z) = 1 + z\, b^\top (I - zA)^{-1} \mathbf{1} \tag{17.2.9}$$

where $A = (a_{ij})$ is the matrix of stage coefficients, $b$ is the vector of weights, and $\mathbf{1}$ is the vector of ones. The function $R(z)$, known as the stability function or amplification factor, characterizes how errors and perturbations are propagated from one step to the next.

For explicit $s$-stage Runge–Kutta methods, the matrix $A$ is strictly lower triangular, and the inverse $(I - zA)^{-1}$ reduces to a finite series. As a result, the stability function $R(z)$ is a polynomial of degree $s$. This polynomial structure directly reflects the number of stages and the order conditions satisfied by the method.

A particularly illustrative example is provided by the classical fourth-order Runge–Kutta method, for which the stability function is:

$$R_4(z) = 1 + z + \frac{z^2}{2} + \frac{z^3}{6} + \frac{z^4}{24} \tag{17.2.10}$$

This expression coincides with the fourth-order Taylor polynomial of the exponential function $e^z$, which is the exact solution of the test equation. This correspondence highlights the connection between order conditions and stability: the method reproduces the exponential behavior of the exact solution up to the order of accuracy, but deviates for larger values of $|z|$. This behavior is reflected geometrically in the structure of the stability region in the complex plane.

<div class="row justify-content-center">
    <div class="rounded p-4 position-relative overflow-hidden border-1 text-center" style="width: 40%">
        {{< figure src="/images/pqQDe4beUu67RvW3raYP-36AhqB70jVAongAC8iIv-v1.png" >}}
        <p>**Figure 17.2.2.** Stability regions in the complex plane for the explicit Euler method and the classical fourth-order Runge–Kutta method. The circle |1+z|=1 corresponds to explicit Euler, while the outer curve represents the stability boundary $|R_4(z)|=1$. The shaded region indicates values of $z=h\lambda$ for which the numerical solution remains stable.</p>
    </div>
</div>

As shown in Figure 17.2.2, the stability region of the explicit Euler method is confined to a unit disk centered at $-1$, whereas the fourth-order Runge–Kutta method exhibits a larger but still bounded stability region. This boundedness explains why explicit Runge–Kutta methods are effective for nonstiff problems but require small stepsizes when applied to stiff systems.

The concept of absolute stability is defined by the condition $|R(z)| \leq 1$. The set of values of $z$ for which this inequality holds is known as the region of absolute stability. For explicit Runge–Kutta methods, this region is bounded in the complex plane, which has important implications. In particular, when applied to stiff systems, where eigenvalues of the Jacobian may have large negative real parts, the product $z = h\lambda$ can lie outside the stability region unless the step size $h$ is chosen extremely small. This explains why explicit methods are highly effective for nonstiff problems but become inefficient or unstable for stiff systems (Caldana and Hesthaven, 2024; Cano and Moreta, 2024).

From a computational perspective, the cost of a Runge–Kutta method is determined primarily by the number of stage evaluations. If the state dimension is $m$, the cost of evaluating the function $f$ is $C_f(m)$, and the method has $s$ stages, then a single time step requires:

$$s \, C_f(m) + O(sm) \tag{17.2.11}$$

operations. The dominant contribution arises from the $s$ evaluations of $f$, while the additional $O(sm)$ term accounts for vector operations such as linear combinations and updates.

Storage requirements are typically proportional to the number of stages, leading to $O(sm)$ memory usage. Each stage derivative must be stored until it is combined in the final update. However, specialized low-storage formulations reorganize the computation so that only a small number of intermediate vectors are retained at any given time, reducing the storage requirement to $O(m)$. This reduction is particularly important in large-scale applications, such as those arising from the spatial discretization of partial differential equations, where the state dimension $m$ can be very large.

These computational and stability considerations motivate ongoing developments in the design of Runge–Kutta methods. Multirate schemes aim to handle systems with multiple time scales by allocating computational effort selectively across components. Mixed-precision implementations exploit modern hardware capabilities to balance accuracy and performance. Low-storage variants address memory constraints in large-scale simulations, while stability-optimized methods seek to enlarge the stability region for improved performance on challenging problems.

Collectively, these advances reflect the continuing evolution of Runge–Kutta methods, driven by the dual demands of numerical stability and computational efficiency in increasingly complex applications (Vermeire, 2023; Bachmann et al., 2025; Dravins et al., 2025).

## 17.2.6. Comparison with Related Time Integration Methods

Runge–Kutta methods occupy a central position within the broader landscape of numerical integrators for initial value problems. Their significance arises from a balance of generality, accuracy, and ease of implementation, which distinguishes them from other classical approaches. By situating Runge–Kutta methods relative to these alternatives, one gains a clearer understanding of their strengths and the contexts in which they are most effectively applied.

In comparison with Taylor series methods, Runge–Kutta schemes achieve high-order accuracy without requiring the explicit computation of higher derivatives of the function $f(t, y)$. Taylor methods rely on successive derivatives of the solution, which must be expressed in terms of derivatives of $f$ through repeated application of the chain rule. In many practical problems, especially those involving complex or black-box models, such derivatives are difficult or computationally expensive to obtain. Runge–Kutta methods circumvent this limitation by replacing derivative information with multiple evaluations of $f$ at carefully chosen points within each time step, thereby retaining high-order accuracy while remaining broadly applicable.

When compared with linear multistep methods, Runge–Kutta methods exhibit a different set of advantages. Multistep methods construct the solution at a new time level using information from several previous steps, which can lead to high efficiency once the method is fully initialized. However, they require a startup procedure to generate the initial values, and their reliance on past data complicates their use in situations involving variable step sizes, event detection, or irregular time grids. In contrast, Runge–Kutta methods are self-starting: each step depends only on the current state $(t_n, y_n)$. This property simplifies their implementation and makes them particularly well suited for adaptive step size control and problems where the integration process must respond dynamically to changing conditions.

Relative to Euler’s method, Runge–Kutta methods achieve significantly higher accuracy by incorporating multiple slope evaluations within a single step. Euler’s method uses a single evaluation of $f$, resulting in a first-order approximation that may require very small step sizes to achieve acceptable accuracy. Runge–Kutta methods, by contrast, combine several evaluations to approximate the integral in (17.1.4) more accurately, thereby increasing the order of the method and reducing the number of steps required for a given accuracy level.

Modern developments have further extended the capabilities of the Runge–Kutta framework. Paired explicit methods are designed to address localized stiffness by combining schemes with different stability properties. Multirate methods exploit the presence of multiple time scales within a system by advancing different components with different effective step sizes, thereby improving efficiency without sacrificing accuracy. Exponential Runge–Kutta methods incorporate analytic treatment of stiff linear components, reducing the numerical burden associated with rapidly decaying modes. Modified schemes have also been developed to enhance nonlinear stability when additional structure is present in the underlying problem.

These extensions demonstrate that the Runge–Kutta framework is not static but continues to evolve in response to emerging computational challenges. Its adaptability, combined with its strong theoretical foundation, ensures that it remains a cornerstone of contemporary scientific computing, bridging classical numerical analysis and modern applications (Vermeire, 2023; Bachmann et al., 2025; Cano and Moreta, 2024; Hu and Wang, 2024).

## 17.2.7. Implementation Perspective for Runge–Kutta Methods in Rust

The practical realization of Runge–Kutta methods in a systems-level language such as Rust benefits directly from the structured formulation developed in (17.2.1)–(17.2.2). The mathematical decomposition into stages, coefficients, and weighted updates translates naturally into a modular and efficient implementation strategy, where numerical clarity and performance considerations can be addressed simultaneously.

At the core of the implementation lies the representation of the Runge–Kutta coefficients. These are most naturally stored as immutable data structures encoding the Butcher tableau, including the arrays $(a_{ij})$, $(b_i)$, and $(c_i)$. Treating these coefficients as immutable ensures both safety and predictability, as the defining parameters of the method remain fixed throughout execution. This design also allows different methods to be interchanged seamlessly by substituting one tableau for another, without altering the solver logic.

The solver itself operates by iterating over the stages defined in (17.2.1). For each stage $i$, the intermediate state $y_n + h \sum_{j=1}^{i-1} a_{ij} k_j$, is constructed using previously computed stage derivatives. This is followed by evaluation of the function $f(t, y)$ at the corresponding stage point $t_n + c_i h$. Once all stage derivatives $k_i$ have been computed, they are combined according to (17.2.2) to produce the updated solution $y_{n+1}$.

From a performance standpoint, efficient implementations rely on careful memory management. Stage buffers for the vectors $k_i$ are typically preallocated before the time-stepping loop begins, avoiding repeated dynamic memory allocation during execution. This is particularly important in high-performance contexts, where allocation overhead can significantly degrade efficiency. By reusing preallocated buffers, the implementation achieves predictable memory usage and improved cache locality.

Vector operations are organized to exploit contiguous memory layouts, enabling efficient access patterns and minimizing memory latency. The summations appearing in (17.2.1) and (17.2.2) can be implemented as fused operations, where intermediate results are accumulated directly into working buffers. This reduces the number of passes over the data and improves overall computational throughput.

The structure of the Runge–Kutta method aligns closely with a loop-based implementation. Each stage corresponds to an iteration of a loop, with dependencies only on previously computed values. This sequential yet structured computation maps well onto modern processor architectures, allowing effective use of vectorization and parallel execution where applicable. At the same time, the absence of complex control flow or indirect dependencies contributes to predictable performance.

The close correspondence between the mathematical formulation and its computational realization is a defining advantage of Runge–Kutta methods. The equations (17.2.1)–(17.2.2) translate almost directly into code, preserving both clarity and correctness. This alignment is particularly valuable in Rust, where explicit control over memory and data structures enables high-performance implementations without sacrificing safety. As a result, Runge–Kutta methods are especially well suited for large-scale simulations and performance-critical applications in modern numerical computing.

### Rust Implementation

Following the discussion in Section 17.2.7 on the implementation structure of Runge–Kutta methods and their representation through the Butcher tableau, Program 17.2.2 provides a practical realization of a general explicit Runge–Kutta solver. Building on the formulation in Equations (17.2.1)–(17.2.2), the program encodes the coefficients $a_{ij}, b_i, c_i$ as immutable data structures and uses them to construct stage derivatives and perform the final weighted update. By applying the same solver to multiple tableaux, including the explicit midpoint method and the classical fourth-order Runge–Kutta method, the implementation demonstrates how different numerical schemes can be expressed within a unified computational framework. This abstraction highlights the close correspondence between the mathematical structure of Runge–Kutta methods and their efficient realization in a systems-level language such as Rust.

At the core of the implementation is the `ButcherTableau` structure, which encodes the coefficients appearing in Equations (17.2.1) and (17.2.2). The arrays $a_{ij}$, $b_i$, and $c_i$ define the stage dependencies, weights, and evaluation nodes, respectively. By storing these coefficients in an immutable structure, the program ensures that the defining characteristics of each Runge–Kutta method remain fixed and can be reused consistently throughout the computation. This abstraction allows different methods to be interchanged simply by providing different tableaux, without modifying the solver logic.

The function `rk_step` implements the stage-based update described in Equation (17.2.1). For each stage, it constructs the intermediate state $y_n + h \sum_{j<i} a_{ij} k_j$ using previously computed stage derivatives. This is followed by evaluation of the right-hand side function at the corresponding stage point $t_n + c_i h$. Once all stage derivatives have been computed, the function performs the final update according to Equation (17.2.2), combining the stages with weights $b_i$ to obtain the next state $y_{n+1}$. This direct translation of the mathematical formulation into code highlights the clarity and structure of Runge–Kutta methods.

The `solve` function provides a generic time-stepping loop that repeatedly applies the update map defined by `rk_step`. This corresponds to the iterative application of the one-step method introduced in Equation (17.1.5). By separating the solver from the specific method coefficients, the implementation achieves a high degree of modularity, allowing the same integration loop to be used for different Runge–Kutta schemes.

To illustrate the framework, the program defines two specific tableaux: the explicit midpoint method and the classical fourth-order Runge–Kutta method. These correspond to the schemes presented in Equations (17.2.6) and (17.2.7), respectively. The midpoint method demonstrates a simple two-stage structure, while the RK4 method uses four stages with carefully chosen weights to achieve higher accuracy. The `print_summary` function reports key characteristics of each method, including the number of stages and the resulting numerical error.

The program also includes the harmonic oscillator as a test problem, obtained by reformulating a second-order differential equation as a first-order system as in Equation (17.1.2). The exact solution is known analytically, allowing the functions `exact_solution` and `max_abs_error` to quantify the numerical accuracy of each method. This comparison illustrates how the theoretical order of a method translates into practical error behavior.

The `main` function initializes the problem, constructs the tableaux, and invokes the solver for each method. It then prints a summary of the results and a step-by-step trajectory for the RK4 method. This demonstrates the flexibility of the implementation and provides a clear comparison of the performance of different Runge–Kutta schemes within the same computational framework.

```rust
// Program 17.2.2: Butcher-Tableau-Based Explicit Runge-Kutta Solver
//
// Problem statement:
// This program implements a reusable explicit Runge-Kutta solver based on the
// Butcher tableau representation. The coefficients a_ij, b_i, and c_i define
// the stages and final weighted update of the method. The same solver is used
// with two different tableaux: the explicit midpoint method and the classical
// fourth-order Runge-Kutta method.

type State = Vec<f64>;

#[derive(Clone)]
struct ButcherTableau {
    name: &'static str,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    c: Vec<f64>,
}

impl ButcherTableau {
    fn explicit_midpoint() -> Self {
        Self {
            name: "Explicit Midpoint Method",
            a: vec![
                vec![0.0, 0.0],
                vec![0.5, 0.0],
            ],
            b: vec![0.0, 1.0],
            c: vec![0.0, 0.5],
        }
    }

    fn classical_rk4() -> Self {
        Self {
            name: "Classical Fourth-Order Runge-Kutta Method",
            a: vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.5, 0.0, 0.0, 0.0],
                vec![0.0, 0.5, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ],
            b: vec![1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
            c: vec![0.0, 0.5, 0.5, 1.0],
        }
    }

    fn stages(&self) -> usize {
        self.b.len()
    }
}

fn zeros(n: usize) -> State {
    vec![0.0; n]
}

fn rk_step<F>(
    tableau: &ButcherTableau,
    f: &F,
    t: f64,
    y: &State,
    h: f64,
) -> State
where
    F: Fn(f64, &State) -> State,
{
    let s = tableau.stages();
    let m = y.len();

    let mut k: Vec<State> = vec![zeros(m); s];

    for i in 0..s {
        let mut y_stage = y.clone();

        for j in 0..i {
            let a_ij = tableau.a[i][j];

            for component in 0..m {
                y_stage[component] += h * a_ij * k[j][component];
            }
        }

        let t_stage = t + tableau.c[i] * h;
        k[i] = f(t_stage, &y_stage);
    }

    let mut y_next = y.clone();

    for i in 0..s {
        for component in 0..m {
            y_next[component] += h * tableau.b[i] * k[i][component];
        }
    }

    y_next
}

fn solve<F>(
    tableau: &ButcherTableau,
    f: &F,
    t0: f64,
    y0: State,
    h: f64,
    n_steps: usize,
) -> Vec<(f64, State)>
where
    F: Fn(f64, &State) -> State,
{
    let mut trajectory = Vec::with_capacity(n_steps + 1);

    let mut t = t0;
    let mut y = y0;

    trajectory.push((t, y.clone()));

    for _ in 0..n_steps {
        y = rk_step(tableau, f, t, &y, h);
        t += h;
        trajectory.push((t, y.clone()));
    }

    trajectory
}

fn exact_solution(t: f64) -> State {
    vec![t.cos(), -t.sin()]
}

fn max_abs_error(numerical: &State, exact: &State) -> f64 {
    numerical
        .iter()
        .zip(exact.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn print_summary(tableau: &ButcherTableau, trajectory: &[(f64, State)]) {
    let (t_final, y_final) = trajectory.last().expect("trajectory must not be empty");
    let exact = exact_solution(*t_final);
    let error = max_abs_error(y_final, &exact);

    println!("{}", tableau.name);
    println!("{}", "-".repeat(tableau.name.len()));
    println!("Number of stages              = {}", tableau.stages());
    println!("Final time t                  = {:.6}", t_final);
    println!("Computed q(t)                 = {:.12}", y_final[0]);
    println!("Computed v(t)                 = {:.12}", y_final[1]);
    println!("Exact q(t)                    = {:.12}", exact[0]);
    println!("Exact v(t)                    = {:.12}", exact[1]);
    println!("Maximum absolute error        = {:.6e}", error);
    println!();
}

fn main() {
    println!("Butcher-Tableau-Based Explicit Runge-Kutta Solver");
    println!("=================================================\n");

    let harmonic_oscillator = |_: f64, y: &State| -> State {
        let q = y[0];
        let v = y[1];

        vec![v, -q]
    };

    let midpoint = ButcherTableau::explicit_midpoint();
    let rk4 = ButcherTableau::classical_rk4();

    let t0 = 0.0;
    let y0 = vec![1.0, 0.0];
    let h = 0.1;
    let n_steps = 10;

    println!("Model Problem");
    println!("-------------");
    println!("Second-order equation: q''(t) = -q(t)");
    println!("First-order system:    q' = v,   v' = -q");
    println!("Initial state:         q(0) = 1, v(0) = 0\n");

    println!("Numerical Parameters");
    println!("--------------------");
    println!("Initial time t0        = {:.6}", t0);
    println!("Step size h            = {:.6}", h);
    println!("Number of steps        = {}", n_steps);
    println!("Final time             = {:.6}\n", t0 + h * n_steps as f64);

    let midpoint_solution = solve(&midpoint, &harmonic_oscillator, t0, y0.clone(), h, n_steps);
    let rk4_solution = solve(&rk4, &harmonic_oscillator, t0, y0, h, n_steps);

    println!("Final-Time Accuracy Comparison");
    println!("------------------------------\n");

    print_summary(&midpoint, &midpoint_solution);
    print_summary(&rk4, &rk4_solution);

    println!("Step-by-Step RK4 Trajectory");
    println!("---------------------------");
    println!("{:>8} {:>18} {:>18}", "t", "q(t)", "v(t)");

    for (t, y) in rk4_solution {
        println!("{:>8.4} {:>18.10} {:>18.10}", t, y[0], y[1]);
    }
}
```

Program 17.2.2 demonstrates a general and extensible approach to implementing explicit Runge–Kutta methods using the Butcher tableau representation. By encoding the coefficients in a structured form and separating them from the solver logic, the program reflects the modular design principles discussed in Section 17.2.7. This approach allows a wide range of numerical methods to be implemented within a single unified framework.

The comparison between the explicit midpoint method and the classical RK4 method illustrates how differences in stage structure and weighting translate into differences in numerical accuracy. While the midpoint method provides a computationally efficient second-order approximation, the RK4 method achieves significantly higher accuracy through additional stage evaluations. This highlights the trade-off between computational cost and accuracy that underlies the design of Runge–Kutta methods.

The implementation also demonstrates the importance of efficient memory usage and structured computation. By preallocating stage buffers and organizing operations in a loop-based structure, the program aligns with the performance considerations discussed in Section 17.2.5, where computational cost and storage requirements are key factors.

The modular design of the code makes it straightforward to extend the framework to more advanced methods, including embedded Runge–Kutta schemes for adaptive step size control, low-storage variants for large-scale problems, and stability-optimized methods for challenging systems. As such, this program provides a foundation for further exploration of modern time integration techniques.

# 17.3. Adaptive Stepsize Control for Runge-Kutta

In many practical applications, the behavior of the solution to an initial value problem is far from uniform over the integration interval. Periods of smooth variation may be interspersed with rapid transients, steep gradients, or localized features that demand higher resolution. A numerical method with a fixed step size must be chosen conservatively to handle the most demanding region, often resulting in unnecessary computational effort elsewhere. This inefficiency motivates the development of adaptive stepsize strategies, which adjust the step size dynamically in response to the evolving behavior of the solution.

Adaptive Runge–Kutta methods build directly on the one-step framework introduced in Section 17.2. By augmenting the basic update mechanism with error estimation and control, they achieve a balance between accuracy and efficiency. The step size becomes a variable quantity, selected at each step to satisfy a prescribed tolerance while minimizing computational cost. This approach is fundamental to modern ODE solvers and plays a critical role in large-scale simulations and real-time applications.

## 17.3.1. Motivation and Basic Idea of Adaptive Stepsize Selection

A fixed stepsize is often inefficient for practical problems, since the solution behavior may vary significantly over time. In regions where the solution evolves smoothly, the use of small steps leads to unnecessary computations, while in regions with rapid changes, a large step size may fail to capture essential features of the solution, resulting in significant numerical error. This mismatch between the fixed discretization and the variable nature of the solution motivates the use of adaptive strategies.

The central idea of adaptive integration is not to minimize the step size globally, but to choose each step so that it is just small enough to satisfy a prescribed error tolerance. In other words, the step size is selected locally, based on an estimate of the error incurred in advancing the solution over a single step. If the estimated error is too large, the step is reduced; if the error is sufficiently small, the step size may be increased to improve efficiency.

This approach reflects a shift from uniform discretization to error-driven computation. Rather than imposing a fixed resolution across the entire interval, the method adapts to the intrinsic complexity of the solution. Regions requiring high accuracy are treated with smaller steps, while smoother regions are traversed more quickly with larger steps. This dynamic adjustment leads to significant gains in efficiency without compromising accuracy.

From a numerical analysis perspective, adaptive stepsize control introduces additional considerations beyond those encountered in fixed-step methods. The stability and responsiveness of the stepsize controller itself become important, as overly aggressive adjustments can lead to oscillations in the step size, while overly conservative strategies may reduce efficiency. Recent analyses have therefore focused not only on error estimation but also on the stability properties of the control mechanisms governing step size selection.

These considerations underscore the importance of adaptive methods in modern numerical computing. By aligning computational effort with the local behavior of the solution, adaptive Runge–Kutta methods provide a robust and efficient framework for solving a wide range of initial value problems (Saleh, Kovács and Kallur, 2023; Ranocha and Giesselmann, 2024).

## 17.3.2. Step Doubling and Richardson Extrapolation

A conceptually simple and historically important approach to adaptive stepsize control is based on step doubling combined with Richardson extrapolation. This strategy provides both an estimate of the local truncation error and a mechanism for improving the accuracy of the computed solution, all within the framework of a standard Runge–Kutta method.

The basic idea is to compare two approximations over the same interval $[t_n, t_{n+1}]$. First, a single Runge–Kutta step of length $h$ is performed, yielding an approximation $y_h$. Second, the same interval is traversed using two successive steps of length $h/2$, producing a refined approximation denoted by $y_{h/2}^{(2)}$. Because the smaller step size reduces the truncation error, the two approximations differ by an amount that reflects the local error behavior of the method.

For the classical fourth-order Runge–Kutta method, the local truncation error is proportional to $h^5$. Accordingly, the two approximations satisfy the expansions:

$$
\begin{aligned}
y_h &= y(t_{n+1}) + C h^5 + O(h^6), \\
y_{h/2}^{(2)} &= y(t_{n+1}) + \frac{C}{16} h^5 + O(h^6)
\end{aligned}
\tag{17.3.1}
$$

Here, the factor $\frac{1}{16}$ arises because halving the step size reduces the leading error term by a factor of $2^5 = 32$, and two such half-steps accumulate twice that reduced error. Subtracting the two expressions eliminates the exact solution $y(t_{n+1})$ and yields:

$$\Delta = y_{h/2}^{(2)} - y_h = -\frac{15}{16} C h^5 + O(h^6) \tag{17.3.2}$$

This difference provides an estimate of the leading-order error term. Since the constant $C$ is unknown, the quantity $\Delta$ serves as a practical proxy for the local truncation error. More importantly, it can be used to construct an improved approximation through Richardson extrapolation. By combining the two approximations in a manner that cancels the leading error term, one obtains:

$$y(t_{n+1}) \approx y_{h/2}^{(2)} + \frac{1}{15}\left(y_{h/2}^{(2)} - y_h\right) \tag{17.3.3}$$

This extrapolated value achieves one additional order of accuracy compared to the underlying Runge–Kutta method. In effect, the leading $O(h^5)$ error term is eliminated, leaving an error of order $O(h^6)$. This demonstrates how multiple approximations at different step sizes can be combined to enhance accuracy without modifying the underlying integration scheme.

Despite its conceptual clarity, the step doubling approach has significant computational drawbacks. A single Runge–Kutta step of length $h$ requires four function evaluations in the case of RK4. Performing two half-steps requires eight evaluations, and when combined with the full step, the total cost rises to eleven evaluations. This is considerably more expensive than performing only the two half-steps, which already provide a higher-accuracy approximation.

For this reason, while step doubling and Richardson extrapolation are valuable for understanding the principles of error estimation and adaptive control, they are generally less efficient than modern embedded Runge–Kutta methods. These alternatives achieve similar goals with fewer function evaluations by reusing stage computations within a single integration step.

## 17.3.3. Embedded Runge–Kutta Pairs and Error Estimation

A more efficient and widely used approach to adaptive stepsize control is based on embedded Runge–Kutta pairs. These methods are designed to compute two approximations of different orders within a single time step, using the same set of stage evaluations. This eliminates the need for redundant function evaluations, as encountered in step doubling, and provides an economical mechanism for estimating the local truncation error.

The fundamental idea is to construct two numerical updates from the same stage derivatives $k_i$. Specifically, the solution is advanced using two sets of weights,

$$
\begin{aligned}
y_{n+1} &= y_n + h \sum_{i=1}^{s} b_i k_i, \\
\widehat{y}_{n+1} &= y_n + h \sum_{i=1}^{s} \widehat{b}_i k_i
\end{aligned}
\tag{17.3.4}
$$

where the coefficients $b_i$ define a method of order $p$, while the coefficients $\widehat{b}_i$ correspond to a method of lower order, typically $p-1$. Because both approximations are constructed from the same stages, the additional cost of obtaining the second solution is negligible.

The difference between these two approximations,

$$e_{n+1} = y_{n+1} - \widehat{y}_{n+1} \tag{17.3.5}$$

serves as an estimate of the local truncation error. Since the two methods have different orders of accuracy, their difference isolates the leading-order error term, providing a practical measure of the accuracy of the step.

To make this error estimate meaningful across components of varying magnitude, it is normalized in a scale-aware manner. For each component $i$, a scaling factor is defined as:

$$
\mathrm{sci}_i = \mathrm{atol}_i + \mathrm{rtol}_i \max\!\left( |y_{n,i}|,\; |y_{n+1,i}| \right)
\tag{17.3.6}
$$

where $\mathrm{atol}_i$ and $\mathrm{rtol}_i$ represent the absolute and relative tolerances, respectively. This formulation ensures that the error is measured relative to the size of the solution, while also maintaining sensitivity for components that are close to zero.

The normalized error is then aggregated into a single scalar measure using a root-mean-square norm,

$$\mathrm{err} = \left( \frac{1}{m} \sum_{i=1}^{m} \left( \frac{e_{n+1,i}}{\mathrm{sci}_i} \right)^2 \right)^{1/2} \tag{17.3.7}$$

This measure provides a dimensionless quantity that reflects the overall accuracy of the step. A step is accepted if $\mathrm{err} \leq 1$, indicating that the estimated error lies within the prescribed tolerance. If this condition is not satisfied, the step is rejected and recomputed with a smaller step size.

This framework achieves an effective balance between accuracy and efficiency. By combining absolute and relative tolerances, it adapts naturally to problems with widely varying solution scales. Moreover, because the error estimate is obtained without additional function evaluations, embedded Runge–Kutta pairs are significantly more efficient than step doubling approaches.

The resulting methodology forms the foundation of modern adaptive ODE solvers. It provides a robust mechanism for controlling local error while maintaining computational efficiency, and extends seamlessly to vector-valued problems and large-scale systems (Caldana and Hesthaven, 2024).

## 17.3.4. Stepsize Selection and Controller Design

Once an estimate of the local truncation error has been obtained, the next step is to determine an appropriate value for the subsequent stepsize. This decision is central to the effectiveness of adaptive integration, as it governs both the efficiency and stability of the numerical method. The stepsize must be adjusted in response to the observed error so that future steps remain within the prescribed tolerance while avoiding unnecessary reductions that would increase computational cost.

The starting point for stepsize selection is the asymptotic behavior of the local error. If the error estimate satisfies $e_{n+1} = O(h^q)$, then the dependence of the error on the stepsize can be used to predict how the stepsize should be modified. A simple and widely used strategy is the integral, or I-type, controller, which updates the stepsize according to:

$$h_{n+1} = \eta \, h_n \, \mathrm{err}_n^{-1/q}, \qquad 0 < \eta < 1 \tag{17.3.8}$$

Here, $\mathrm{err}_n$ is the normalized error defined in (17.3.7), and $\eta$ is a safety factor introduced to prevent overly aggressive changes in the stepsize. The exponent $-1/q$ reflects the scaling relationship between the error and the stepsize, ensuring that the new step is chosen to bring the error closer to the target value.

In practical implementations, the stepsize update is further constrained to avoid excessive variation between successive steps. This is achieved by imposing bounds of the form:

$$h_{n+1} \in [f_{\min} h_n, \; f_{\max} h_n] \tag{17.3.9}$$

where $f_{\min}$ and $f_{\max}$ are prescribed factors limiting how much the stepsize can decrease or increase in a single update. These bounds help maintain numerical stability and prevent erratic behavior in the integration process.

For a Runge–Kutta method of order $p$, the exponent $q$ is typically chosen as $p$ or $p+1$, depending on the nature of the error estimator. In many embedded methods, the estimator reflects the difference between solutions of orders $p$ and $p-1$, leading to an effective exponent based on the higher-order behavior. For example, the Dormand–Prince pair employs an exponent of $1/5$, corresponding to a fifth-order error scaling.

While the I-type controller provides a simple and effective mechanism for stepsize adjustment, it can exhibit undesirable behavior in certain situations. In particular, it may lead to oscillations in the stepsize when the error estimates fluctuate, especially in explicit Runge–Kutta methods applied to challenging problems. To address this issue, more sophisticated controllers incorporate additional information from previous steps.

A commonly used enhancement is the proportional–integral (PI) controller, given by:

$$h_{n+1} = \eta \, h_n \, \mathrm{err}_n^{-\alpha} \, \mathrm{err}_{n-1}^{\beta}, \qquad \alpha , \beta > 0 \tag{17.3.10}$$

In this formulation, the current error $\mathrm{err}_n$ provides the primary adjustment, while the previous error $\mathrm{err}_{n-1}$ introduces a damping effect that reduces oscillations in the stepsize sequence. The parameters $\alpha$ and $\beta$ control the relative influence of these terms, allowing the controller to balance responsiveness and stability.

Modern analysis has shown that pure I-type controllers may be unstable in certain explicit Runge–Kutta settings, particularly when the error estimates exhibit rapid variation. By incorporating memory of past errors, PI and more general PID controllers improve robustness and produce smoother stepsize sequences. This leads to more reliable performance, especially in problems with complex or rapidly changing dynamics.

Recent developments have further refined these ideas. Adaptive strong stability preserving (SSP) controllers incorporate techniques such as median filtering to stabilize the sequence of stepsizes in regimes characterized by shocks or discontinuities. These approaches aim to maintain stability properties while still achieving efficient error control.

Overall, the design of stepsize controllers represents a crucial component of adaptive Runge–Kutta methods. By linking error estimation with dynamic adjustment of the stepsize, these controllers ensure that the numerical method remains both accurate and efficient across a wide range of problem types (Ranocha and Giesselmann, 2024; D’Afiero, 2026).

## 17.3.5. Local Versus Global Error in Adaptive Integration

A crucial conceptual distinction in adaptive numerical integration is the difference between local truncation error and global error. Adaptive Runge–Kutta methods are designed to estimate and control the error incurred over a single time step, yet the quantity of ultimate interest is the accumulated deviation of the numerical solution from the exact solution over the entire integration interval.

The local truncation error refers to the error introduced in advancing the solution from $t_n$ to $t_{n+1}$, assuming that the starting value $y_n$ is exact. This is precisely the quantity estimated in (17.3.5)–(17.3.7), and it forms the basis for stepsize selection through the controllers described in (17.3.8)–(17.3.10). By enforcing the condition $\mathrm{err} \leq 1$, the method ensures that each individual step satisfies a prescribed tolerance relative to the local behavior of the solution.

In contrast, the global error measures the cumulative effect of all local errors over the interval $[t_0, T]$. It reflects not only the magnitude of the local truncation errors but also how these errors propagate and interact as the solution is advanced step by step. This propagation depends on the stability properties of both the differential equation and the numerical method. In particular, even small local errors can grow significantly if the underlying system amplifies perturbations.

Because adaptive methods regulate only the local error, there is no direct guarantee that the global error will remain within a specified bound. In many well-behaved problems, however, reducing the local tolerance leads to a proportional reduction in global error. This empirical relationship arises when the error propagation remains controlled and the numerical method exhibits stable behavior. Nevertheless, this correspondence is not universal and may break down in problems with strong sensitivity, stiffness, or instability.

This distinction has important implications for both analysis and implementation. It highlights that error control mechanisms operate indirectly, influencing the global accuracy through local adjustments rather than enforcing it explicitly. As a result, the choice of tolerances and controller parameters must be guided by both theoretical insight and practical experimentation.

Modern analyses address this issue by treating the numerical solver and the stepsize controller as a coupled dynamical system. Rather than considering the integration method in isolation, the combined behavior of the discretization and the adaptive strategy is examined. This perspective allows one to study stability and convergence properties that emerge from the interaction between error estimation and stepsize adjustment.

Such an approach provides a more comprehensive understanding of adaptive methods, particularly in complex or highly nonlinear settings. It explains why certain controller designs yield stable and efficient performance, while others may lead to oscillatory or unstable stepsize sequences. By analyzing the solver-controller combination as a whole, one obtains a more accurate characterization of the behavior of adaptive Runge–Kutta methods in practice (Ranocha and Giesselmann, 2024; Saleh, Kovács and Kallur, 2023).

## 17.3.6. Computational Considerations and Implementation of Adaptive Runge–Kutta Methods

Adaptive Runge–Kutta methods based on embedded pairs achieve high computational efficiency by reusing the same stage evaluations required for the primary solution update. Unlike step doubling, which incurs substantial additional cost through repeated integrations, embedded methods extract an error estimate from existing computations. The additional overhead is limited to a second weighted combination of stage derivatives, evaluation of an error norm, and the occasional recomputation of a step when the error tolerance is not satisfied. As a result, adaptive embedded methods provide accurate error control at a significantly lower computational cost than alternative strategies.

From an implementation perspective, each adaptive time step can be organized into three clearly defined phases that mirror the mathematical formulation. The first phase consists of computing the stage derivatives $k_i$ according to (17.2.1). These evaluations dominate the computational cost and must be performed efficiently, typically using preallocated buffers and contiguous memory layouts as discussed in Section 17.2.7.

The second phase involves forming both the primary solution $y_{n+1}$ and the embedded approximation $\widehat{y}_{n+1}$ using the two sets of weights in (17.3.4). Since both updates depend on the same stage derivatives, this step requires only additional linear combinations of already available data. The difference between these two approximations yields the error estimate defined in (17.3.5).

The third phase consists of computing the normalized error measure (17.3.7) and determining whether the step should be accepted. If the error satisfies the acceptance criterion, the solution is advanced and the stepsize is updated according to the controller described in (17.3.8)–(17.3.10). If the error exceeds the tolerance, the step is rejected, the stepsize is reduced, and the computation is repeated.

A key requirement in this process is that a rejected step must not alter the accepted solution state. This ensures that the numerical trajectory remains consistent and that no inaccurate intermediate values are propagated. In practice, this is achieved by maintaining temporary storage for trial solutions and stage derivatives. Only when a step is accepted are the updated values committed to the main solution variables.

In a Rust implementation, this design is naturally expressed through careful management of ownership and mutability. Temporary variables are used to hold intermediate results during trial steps, while the primary state vector remains unchanged until acceptance. This approach aligns well with Rust’s emphasis on safety and explicit control over data flow, reducing the risk of unintended side effects.

These computational design principles ensure that adaptive Runge–Kutta methods remain both numerically reliable and efficient. By combining accurate error estimation with disciplined state management and efficient memory usage, they provide a robust foundation for modern nonstiff ODE solvers.

### Rust Implementation

Following the discussion in Section 17.3 on adaptive stepsize control and the implementation principles outlined in Subsection 17.3.6, Program 17.3.1 provides a practical realization of an adaptive Runge–Kutta solver based on an embedded pair. The formulation builds directly on the stage-based structure of Runge–Kutta methods in Equations (17.2.1)–(17.2.2), augmenting it with dual solution estimates as described in Equation (17.3.4) and an error estimator defined in Equation (17.3.5). By normalizing the error according to Equation (17.3.7) and adjusting the stepsize using the controller in Equation (17.3.8), the program demonstrates how local error control leads to efficient and reliable time integration. The Bogacki–Shampine RK3(2) pair is employed to illustrate how embedded methods achieve adaptive accuracy with minimal additional computational cost.

At the core of the implementation is the representation of the state vector $y$, which corresponds to the solution of the initial value problem defined in Equation (17.1.1). The right-hand side function $f(t,y)$ is implemented as a closure, preserving the abstraction introduced in Section 17.1.4. This design allows the same solver to be applied to different dynamical systems without modification, ensuring flexibility and modularity.

The function `bogacki_shampine_step` implements the embedded Runge–Kutta pair described in Equation (17.3.4). It computes a sequence of stage derivatives according to the structure in Equation (17.2.1), using intermediate states constructed from previously computed stages. These stage derivatives are then combined in two different ways to produce a higher-order approximation $y_{n+1}$ and a lower-order approximation $\widehat{y}_{n+1}$. The difference between these two approximations provides the error estimate defined in Equation (17.3.5), allowing the solver to assess the quality of the step without additional function evaluations.

The function `rms_normalized_error` implements the scale-aware error normalization described in Equations (17.3.6)–(17.3.7). For each component, the error is divided by a combination of absolute and relative tolerances, ensuring that the error measure is meaningful across different magnitudes of the solution. The root-mean-square aggregation produces a single scalar quantity that determines whether the step is acceptable.

The function `adapt_stepsize` implements the I-type stepsize controller given in Equation (17.3.8). Based on the estimated error, it adjusts the stepsize to maintain the desired accuracy. The inclusion of safety factors and bounds reflects the practical considerations discussed in Section 17.3.4, preventing overly aggressive changes and ensuring stable progression of the integration.

The `solve_adaptive` function organizes the adaptive integration process into the three phases described in Section 17.3.6. First, it computes the stage derivatives and solution estimates. Second, it evaluates the normalized error and determines whether the step should be accepted. Third, it updates the stepsize accordingly and either advances the solution or repeats the computation with a smaller step. The use of temporary variables ensures that rejected steps do not affect the accepted solution, preserving numerical consistency.

The `main` function demonstrates the solver using the harmonic oscillator, which is obtained by reformulating a second-order equation as a first-order system as in Equation (17.1.2). The exact solution is known analytically, allowing the program to compute and report the global error. The printed statistics provide insight into the efficiency of the adaptive method, including the number of accepted and rejected steps and the total number of function evaluations.

```rust
// Program 17.3.1: Adaptive Runge-Kutta Integration with Embedded Error Control
//
// Problem statement:
// This program implements adaptive stepsize control using the Bogacki-Shampine
// RK3(2) embedded Runge-Kutta pair. The third-order approximation is used as
// the accepted solution, while the second-order embedded approximation provides
// the local error estimate. The normalized error determines whether a trial
// step is accepted or rejected, and the next stepsize is chosen using an
// I-type controller.

type State = Vec<f64>;

#[derive(Clone)]
struct AdaptiveResult {
    trajectory: Vec<(f64, State)>,
    accepted_steps: usize,
    rejected_steps: usize,
    function_evaluations: usize,
}

fn add_scaled(y: &State, scale: f64, k: &State) -> State {
    y.iter()
        .zip(k.iter())
        .map(|(yi, ki)| yi + scale * ki)
        .collect()
}

fn add_three_scaled(
    y: &State,
    scale1: f64,
    k1: &State,
    scale2: f64,
    k2: &State,
    scale3: f64,
    k3: &State,
) -> State {
    y.iter()
        .zip(k1.iter())
        .zip(k2.iter())
        .zip(k3.iter())
        .map(|(((yi, k1i), k2i), k3i)| yi + scale1 * k1i + scale2 * k2i + scale3 * k3i)
        .collect()
}

fn bogacki_shampine_step<F>(f: &F, t: f64, y: &State, h: f64) -> (State, State, usize)
where
    F: Fn(f64, &State) -> State,
{
    let k1 = f(t, y);

    let y2 = add_scaled(y, 0.5 * h, &k1);
    let k2 = f(t + 0.5 * h, &y2);

    let y3 = add_scaled(y, 0.75 * h, &k2);
    let k3 = f(t + 0.75 * h, &y3);

    // Third-order solution:
    // y_high = y_n + h(2/9 k1 + 1/3 k2 + 4/9 k3).
    let y_high = add_three_scaled(
        y,
        h * 2.0 / 9.0,
        &k1,
        h * 1.0 / 3.0,
        &k2,
        h * 4.0 / 9.0,
        &k3,
    );

    let k4 = f(t + h, &y_high);

    // Second-order embedded solution:
    // y_low = y_n + h(7/24 k1 + 1/4 k2 + 1/3 k3 + 1/8 k4).
    let mut y_low = y.clone();

    for i in 0..y.len() {
        y_low[i] += h
            * ((7.0 / 24.0) * k1[i]
                + (1.0 / 4.0) * k2[i]
                + (1.0 / 3.0) * k3[i]
                + (1.0 / 8.0) * k4[i]);
    }

    (y_high, y_low, 4)
}

fn rms_normalized_error(
    y_old: &State,
    y_new: &State,
    y_embedded: &State,
    atol: f64,
    rtol: f64,
) -> f64 {
    let m = y_old.len();
    let mut sum = 0.0_f64;

    for i in 0..m {
        let error_i = y_new[i] - y_embedded[i];
        let scale_i = atol + rtol * y_old[i].abs().max(y_new[i].abs());
        let ratio = error_i / scale_i;
        sum += ratio * ratio;
    }

    (sum / m as f64).sqrt()
}

fn adapt_stepsize(h: f64, error: f64, order: f64) -> f64 {
    let safety = 0.9;
    let min_factor = 0.2;
    let max_factor = 5.0;

    let factor = if error == 0.0 {
        max_factor
    } else {
        safety * error.powf(-1.0 / order)
    };

    h * factor.clamp(min_factor, max_factor)
}

fn solve_adaptive<F>(
    f: &F,
    t0: f64,
    y0: State,
    t_end: f64,
    h_initial: f64,
    atol: f64,
    rtol: f64,
    h_min: f64,
    h_max: f64,
) -> AdaptiveResult
where
    F: Fn(f64, &State) -> State,
{
    let mut trajectory = Vec::new();

    let mut t = t0;
    let mut y = y0;
    let mut h = h_initial.clamp(h_min, h_max);

    let mut accepted_steps = 0;
    let mut rejected_steps = 0;
    let mut function_evaluations = 0;

    trajectory.push((t, y.clone()));

    while t < t_end {
        if t + h > t_end {
            h = t_end - t;
        }

        let (y_high, y_low, evals) = bogacki_shampine_step(f, t, &y, h);
        function_evaluations += evals;

        let error = rms_normalized_error(&y, &y_high, &y_low, atol, rtol);

        if error <= 1.0 {
            t += h;
            y = y_high;
            accepted_steps += 1;
            trajectory.push((t, y.clone()));
        } else {
            rejected_steps += 1;
        }

        // The accepted solution is third order, and the local error estimate
        // scales like h^4, so q = 4 is used in the controller.
        h = adapt_stepsize(h, error, 4.0).clamp(h_min, h_max);

        if h <= h_min && error > 1.0 {
            panic!("Stepsize reached h_min before satisfying the error tolerance.");
        }
    }

    AdaptiveResult {
        trajectory,
        accepted_steps,
        rejected_steps,
        function_evaluations,
    }
}

fn exact_solution(t: f64) -> State {
    vec![t.cos(), -t.sin()]
}

fn max_abs_error(numerical: &State, exact: &State) -> f64 {
    numerical
        .iter()
        .zip(exact.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn main() {
    println!("Adaptive Runge-Kutta Integration with Embedded Error Control");
    println!("===========================================================\n");

    let harmonic_oscillator = |_: f64, y: &State| -> State {
        let q = y[0];
        let v = y[1];

        vec![v, -q]
    };

    let t0 = 0.0;
    let t_end = 10.0;
    let y0 = vec![1.0, 0.0];

    let h_initial = 0.25;
    let h_min = 1.0e-8;
    let h_max = 0.5;

    let atol = 1.0e-8;
    let rtol = 1.0e-6;

    let result = solve_adaptive(
        &harmonic_oscillator,
        t0,
        y0,
        t_end,
        h_initial,
        atol,
        rtol,
        h_min,
        h_max,
    );

    let (t_final, y_final) = result
        .trajectory
        .last()
        .expect("trajectory should contain at least one point");

    let exact = exact_solution(*t_final);
    let error = max_abs_error(y_final, &exact);

    println!("Model Problem");
    println!("-------------");
    println!("Second-order equation: q''(t) = -q(t)");
    println!("First-order system:    q' = v,   v' = -q");
    println!("Initial state:         q(0) = 1, v(0) = 0\n");

    println!("Adaptive Solver Parameters");
    println!("--------------------------");
    println!("Embedded pair                = Bogacki-Shampine RK3(2)");
    println!("Initial time t0              = {:.6}", t0);
    println!("Final time                   = {:.6}", t_end);
    println!("Initial stepsize             = {:.6}", h_initial);
    println!("Minimum stepsize             = {:.2e}", h_min);
    println!("Maximum stepsize             = {:.6}", h_max);
    println!("Absolute tolerance           = {:.2e}", atol);
    println!("Relative tolerance           = {:.2e}\n", rtol);

    println!("Solver Statistics");
    println!("-----------------");
    println!("Accepted steps               = {}", result.accepted_steps);
    println!("Rejected steps               = {}", result.rejected_steps);
    println!("Function evaluations         = {}", result.function_evaluations);
    println!("Stored solution points       = {}\n", result.trajectory.len());

    println!("Final-Time Accuracy");
    println!("-------------------");
    println!("Final time t                 = {:.6}", t_final);
    println!("Computed q(t)                = {:.12}", y_final[0]);
    println!("Computed v(t)                = {:.12}", y_final[1]);
    println!("Exact q(t)                   = {:.12}", exact[0]);
    println!("Exact v(t)                   = {:.12}", exact[1]);
    println!("Maximum absolute error       = {:.6e}\n", error);

    println!("Selected Trajectory Points");
    println!("--------------------------");
    println!("{:>10} {:>18} {:>18}", "t", "q(t)", "v(t)");

    let stride = (result.trajectory.len() / 10).max(1);

    for (index, (t, y)) in result.trajectory.iter().enumerate() {
        if index % stride == 0 || index + 1 == result.trajectory.len() {
            println!("{:>10.6} {:>18.10} {:>18.10}", t, y[0], y[1]);
        }
    }
}
```

Program 17.3.1 demonstrates a practical implementation of adaptive Runge–Kutta integration using embedded error estimation and dynamic stepsize control. By combining the stage-based formulation of Runge–Kutta methods with the error estimation framework of Section 17.3, the program achieves an effective balance between accuracy and computational efficiency.

The use of the Bogacki–Shampine embedded pair illustrates how two approximations of different orders can be obtained from the same stage evaluations, providing a reliable error estimate at minimal additional cost. The normalized error measure ensures that the solver adapts appropriately to variations in the solution magnitude, while the stepsize controller maintains stability and efficiency across the integration interval.

The results highlight the advantages of adaptive methods over fixed stepsize approaches. Instead of using a uniformly small stepsize, the solver adjusts its resolution dynamically, taking larger steps in smooth regions and smaller steps where the solution varies rapidly. This leads to a significant reduction in computational effort while maintaining the desired accuracy.

The modular structure of the implementation makes it straightforward to extend the framework to more advanced methods, such as higher-order embedded pairs, PI or PID controllers, and stability-enhanced adaptive schemes. As such, this program provides a foundation for modern adaptive ODE solvers and illustrates the practical realization of the concepts developed in Section 17.3.

# 17.4. Richardson Extrapolation and the Bulirsch-Stoer Method

Richardson extrapolation and the Bulirsch–Stoer method represent a distinct approach to numerical integration, aimed at achieving very high accuracy for smooth problems. Unlike fixed-order Runge–Kutta schemes, which increase accuracy by adding more stages within a single step, extrapolation methods systematically combine solutions computed at different step sizes to eliminate leading error terms. This approach is particularly effective when the underlying solution is sufficiently smooth, allowing asymptotic error expansions to be exploited in a controlled manner.

The central idea is to treat the numerical method itself as a sequence of approximations parameterized by the step size, and then to accelerate convergence by canceling dominant error components. This leads to methods that can achieve high accuracy with relatively few function evaluations per unit of accuracy, provided that stability constraints are not restrictive. The Bulirsch–Stoer method builds on this principle, combining extrapolation with efficient base integrators and adaptive control to form a powerful technique for nonstiff problems.

## 17.4.1. Problem Context and Numerical Priorities

In practical scientific computing, three broad classes of ordinary differential equation models arise repeatedly. These classes reflect distinct mathematical structures and impose different requirements on numerical methods. The first class consists of general initial value problems of the form:

$$y'(t) = f(t, y) \tag{17.4.1}$$

which serve as the standard framework for a wide range of applications. For such systems, particularly when they are smooth and nonstiff, the primary objective is often to achieve high accuracy with minimal computational effort.

A second important class is given by conservative second-order systems,

$$M q''(t) = -\nabla V(q) \tag{17.4.2}$$

which arise naturally in mechanics. In this setting, the emphasis shifts from purely local accuracy to the preservation of geometric structure and invariants, such as energy or momentum, over long time intervals. Numerical methods applied to such systems must respect these properties to avoid the accumulation of systematic errors, often referred to as secular drift.

The third class consists of stiff or split systems of the form:

$$y'(t) = A y + g(t, y) \tag{17.4.3}$$

which are common in multiscale problems, including chemical kinetics and the method-of-lines discretization of partial differential equations. Here, the dominant concern is stability, particularly in the presence of widely separated time scales. Numerical methods must be able to handle rapidly decaying modes without requiring prohibitively small step sizes.

These three formulations correspond to distinct numerical priorities. For smooth nonstiff problems, the focus is on maximizing accuracy and efficiency. For conservative mechanical systems, the preservation of structure becomes paramount. For stiff systems, stability considerations dominate. Modern large-scale simulation codes often encounter all three situations simultaneously, especially in contexts where partial differential equations are reduced to systems of ordinary differential equations through spatial discretization or operator splitting.

A useful conceptual framework for understanding this transition is given by the mapping:

$$
\underbrace{\partial_t U = \mathcal{T}(U) + \mathcal{R}(U)}_{\text{PDE}}
\quad \xrightarrow{\text{space discretization / operator splitting}} \quad
\underbrace{\dot{y} = F(y)}_{\text{ODE}}
\tag{17.4.4}
$$

This transformation highlights how complex spatially distributed systems are reduced to time-dependent systems of ordinary differential equations, which can then be treated using the methods developed in this chapter.

Similarly, in mechanics, second-order systems can be reformulated as first-order systems in phase space,

$$
\underbrace{M q'' = -\nabla V(q)}_{\text{Newton / Lagrange}}
\quad \Longleftrightarrow \quad
\underbrace{
\frac{d}{dt}
\begin{bmatrix}
q \\ p
\end{bmatrix}
=
\begin{bmatrix}
M^{-1} p \\ -\nabla V(q)
\end{bmatrix}
}_{\text{Hamiltonian first-order form}},
\qquad p = M q'
\tag{17.4.5}
$$

This reformulation reveals additional structure, such as the Hamiltonian nature of the system, which may guide the choice of numerical method. These representations are not merely formal transformations; they expose algebraic and geometric properties that influence the design and performance of numerical integrators.

This section focuses on the high-accuracy regime associated with smooth, nonstiff problems. In this setting, extrapolation methods such as Richardson extrapolation and the Bulirsch–Stoer method are particularly effective. By exploiting the smoothness of the solution and the predictable structure of truncation errors, these methods achieve high levels of accuracy with controlled computational effort, providing a powerful alternative to standard fixed-order schemes (Balos et al., 2025; Ye et al., 2025).

## 17.4.2. Richardson Extrapolation and Order Acceleration

Richardson extrapolation provides a systematic mechanism for increasing the accuracy of a numerical approximation by exploiting the asymptotic structure of its truncation error. Rather than modifying the underlying numerical method, the idea is to combine approximations computed at different resolutions in such a way that the dominant error terms cancel. This approach is particularly effective when the solution is smooth and the error admits a regular expansion in powers of the stepsize.

Let $A(h)$ denote a numerical approximation to the exact solution over a fixed macro-interval of length $H$, where the internal stepsize is $h = H/n$. For a method of order $p$, the approximation is assumed to admit an asymptotic expansion of the form:

$$
A(h) = y(t + H) + c_1 h^p + c_2 h^{p+1} + c_3 h^{p+2} + \cdots
\tag{17.4.6}
$$

Here, the coefficients $c_1, c_2, \ldots$ depend on the solution and the method, but not on the stepsize. The leading error term is proportional to $h^p$, which determines the order of the method.

If the same approximation is computed using a refined stepsize $h/2$, then the corresponding expansion becomes:

$$
A\!\left(\frac{h}{2}\right) = y(t + H)
+ c_1 \left(\frac{h}{2}\right)^p
+ c_2 \left(\frac{h}{2}\right)^{p+1}
+ c_3 \left(\frac{h}{2}\right)^{p+2}
+ \cdots
\tag{17.4.7}
$$

Because the leading error term scales predictably with the stepsize, these two approximations can be combined to eliminate the $h^p$ contribution. Specifically, forming the linear combination:

$$R(h) = \frac{2^p A(h/2) - A(h)}{2^p - 1}$$

yields,

$$R(h) = y(t+H) + \mathcal{O}(h^{p+1}) \tag{17.4.8}$$

This expression defines the classical Richardson extrapolation formula. The key observation is that the coefficient of the leading error term cancels exactly, leaving a higher-order approximation without requiring any modification of the underlying numerical scheme. In effect, the method has been upgraded from order $p$ to order $p+1$ through a purely algebraic transformation.

The process can be repeated by applying extrapolation recursively to a sequence of approximations computed with progressively smaller stepsizes. Each stage of extrapolation eliminates another term in the error expansion, thereby increasing the formal order of accuracy. In practical implementations, these extrapolated values are often organized in a triangular scheme, where successive refinements build upon previously computed approximations.

Recent analyses have shown that when explicit Runge–Kutta methods are used as the base integrator, multiple Richardson extrapolation can raise the effective order more rapidly, for example from $p$ to $p+2$ under suitable conditions. This reflects the structured nature of the error expansion and the interplay between the base method and the extrapolation procedure.

More recent developments extend this classical viewpoint further by interpreting Richardson extrapolation within a probabilistic framework. In this setting, approximations at different resolutions are viewed as observations corresponding to different fidelity levels, and the extrapolation process is formulated as a regression problem. This perspective allows the incorporation of uncertainty quantification and experimental design principles into the extrapolation process, providing a richer interpretation of error behavior and convergence (Bayleyegn, Faragó and Havasi, 2024; Oates et al., 2025).

Despite these modern extensions, the essential idea remains unchanged: by performing computations at multiple resolutions and combining them appropriately, one can cancel dominant error terms and achieve higher accuracy with controlled additional cost.

## 17.4.3. The Modified Midpoint Method as an Extrapolation Base Scheme

The effectiveness of Richardson extrapolation depends critically on the choice of the underlying numerical method. In the Bulirsch–Stoer approach, the modified midpoint method is selected as the base integrator because of its favorable structural properties, particularly the form of its truncation error expansion. This choice is not incidental but is motivated by the way in which the method interacts with the extrapolation process to achieve rapid increases in accuracy.

Over a single macro-step from $t_n$ to $t_n + H$, the method proceeds by subdividing the interval into $n$ equal substeps of size $h = H/n$. The computation begins with the initialization:

$$z_0 = y_n, \qquad z_1 = z_0 + h \, f(t_n, z_0) \tag{17.4.9}$$

which provides the first two values needed to start the recurrence. Subsequent values are generated using a leapfrog-type recurrence relation,

$$z_{m+1} = z_{m-1} + 2h \, f(t_n + mh, z_m), \qquad m = 1,2,\dots,n-1 \tag{17.4.10}$$

This recurrence advances the solution by alternating between previously computed points, effectively propagating the solution across the interval using information centered at intermediate stages. The structure resembles a centered difference scheme and plays a key role in the symmetry properties of the method.

After completing the $n$ substeps, a corrected approximation to the endpoint is formed as:

$$A_n(H) = \frac{1}{2}\Bigl(z_n + z_{n-1} + h \, f(t_n + H, z_n)\Bigr) \tag{17.4.11}$$

This correction combines the final two iterates with an additional slope evaluation at the endpoint, producing a more accurate approximation to the solution at $t_n + H$.

The principal reason for selecting the modified midpoint method lies in the structure of its truncation error. For sufficiently smooth functions $f$ and fixed macro-step $H$, the approximation admits an expansion of the form:

$$A_n(H) = y(t_n + H) + \alpha_1 h^2 + \alpha_2 h^4 + \alpha_3 h^6 + \cdots \tag{17.4.12}$$

A notable feature of this expansion is that only even powers of $h$ appear. The absence of odd powers is a consequence of the symmetry inherent in the midpoint formulation, particularly the centered nature of the recurrence relation. This symmetry leads to cancellation of odd-order error terms, resulting in a more structured and predictable error behavior.

This property has a direct and significant impact on the effectiveness of Richardson extrapolation. Because the leading error term is proportional to $h^2$, and subsequent terms involve only even powers, each extrapolation step can eliminate the dominant error term and raise the order by two, rather than by one as in the general case. This accelerated order improvement is the key reason why the modified midpoint method is so well suited to extrapolation-based techniques.

The combination of the modified midpoint method with Richardson extrapolation forms the foundation of the Bulirsch–Stoer method. The base integrator provides a sequence of approximations with a highly regular error structure, while the extrapolation process systematically removes successive error terms. Together, they yield a method capable of achieving very high accuracy with efficient use of computational resources for smooth, nonstiff problems.

## 17.4.4. Extrapolation Tableau and Practical Construction

The Bulirsch–Stoer method becomes most transparent and systematically implementable when expressed in terms of an extrapolation tableau. This representation organizes approximations computed at different resolutions and applies Richardson extrapolation in a structured, recursive manner. The tableau not only clarifies the algorithmic flow but also provides a natural mechanism for estimating error and selecting the most accurate approximation available at a given stage.

To construct the tableau, consider a sequence of substep counts $n_1 < n_2 < \cdots$, and define the corresponding step parameters:

$$x_j = \left(\frac{H}{n_j}\right)^2, \qquad T_{j,1} = A_{n_j}(H) \tag{17.4.13}$$

Here, $A_{n_j}(H)$ denotes the approximation obtained from the modified midpoint method using $n_j$ substeps over the macro-interval $H$. The choice of $x_j = h^2$ reflects the even-power error expansion (17.4.12), ensuring that the extrapolation is performed with respect to the correct asymptotic variable.

The extrapolation process is then carried out using the Aitken–Neville recursion,

$$T_{j,k}= T_{j,k-1} + \frac{T_{j,k-1} - T_{j-1,k-1}}{x_{j-k+1}/x_j - 1},\qquad 2 \le k \le j \tag{17.4.14}$$

This formula combines successive approximations to eliminate leading error terms, progressively increasing the order of accuracy. Each column of the tableau corresponds to a higher level of extrapolation, with the recursion building upon previously computed entries.

The resulting tableau has the triangular structure:

$$
\begin{bmatrix}
T_{1,1} \\
T_{2,1} & T_{2,2} \\
T_{3,1} & T_{3,2} & T_{3,3} \\
\vdots  & \vdots  & \vdots  & \ddots
\end{bmatrix}
\tag{17.4.15}
$$

Within this structure, the diagonal elements $T_{j,j}$ represent the highest-order approximations available after $j$ levels of extrapolation. As $j$ increases, these diagonal entries converge rapidly to the exact solution, provided the underlying assumptions of smoothness and asymptotic error behavior are satisfied.

For practical implementation, the extrapolation tableau also provides a convenient mechanism for estimating the local error. A common approach is to use the difference between successive diagonal approximations,

$$e_j = T_{j,j} - T_{j,j-1} \tag{17.4.16}$$

which serves as an estimate of the remaining truncation error. This estimate reflects the improvement gained by the most recent extrapolation step and is therefore indicative of the current accuracy level.

To make this error measure meaningful across all components of the solution vector, it is typically evaluated using a weighted root-mean-square norm, analogous to the normalization introduced in (17.3.6)–(17.3.7). The macro-step is accepted if this norm is less than or equal to one, indicating that the desired tolerance has been met. Otherwise, the step is rejected, and the computation is repeated with a refined configuration, such as increasing the number of substeps.

This tableau-based formulation integrates naturally with adaptive control strategies. It provides a clear progression of approximations, a built-in error estimate, and a systematic way to improve accuracy through additional extrapolation levels. As such, it forms a central component of the Bulirsch–Stoer method, enabling efficient and reliable high-accuracy integration for smooth, nonstiff problems.

## 17.4.5. Cost Model and Applicability of the Bulirsch–Stoer Method

The efficiency of the Bulirsch–Stoer method is best understood through a cost model that balances the expense of function evaluations against the gains achieved through extrapolation. This analysis clarifies the circumstances under which the method is advantageous and highlights its role as a high-accuracy technique for smooth problems.

Assume that the state vector has dimension $d$, and that a single evaluation of the right-hand side function $f(t,y)$ incurs a cost $C_f(d)$. The modified midpoint method, used as the base integrator, advances the solution over a macro-step of length $H$ using $n_j$ substeps of size $h = H/n_j$. Each such sweep requires approximately $n_j + 1$ evaluations of $f$, corresponding to the initialization and the recurrence steps in (17.4.9)–(17.4.10).

The computational structure of the modified midpoint method is particularly efficient in terms of memory usage. The leapfrog recurrence requires only a small number of working vectors, so that the storage cost for the base integrator itself is $O(d)$. When combined with the extrapolation process, however, additional storage is required to maintain the tableau entries. If only the active diagonals of the tableau are retained, the total storage requirement scales as $O(Kd)$, where $K$ is the number of rows in the extrapolation table.

The total computational cost of constructing a $K$-row extrapolation table can be approximated as:

$$
O\!\left(C_f(d)\sum_{j=1}^{K} n_j\right) \quad \text{work}
\qquad \text{and} \qquad
O(Kd) \quad \text{storage}
\tag{17.4.17}
$$

The dominant contribution arises from the evaluations of the function $f$, which are required for each modified midpoint sweep. The additional arithmetic associated with the extrapolation tableau, including the Aitken–Neville recursion, involves operations on vectors of dimension $d$ and scales as $O(K^2 d)$. In most practical settings, particularly when $f$ is expensive to evaluate, this algebraic overhead is negligible compared to the cost of function evaluations.

This cost structure reveals the fundamental trade-off underlying the Bulirsch–Stoer method. The method invests additional computational effort in evaluating the solution at multiple resolutions and combining these results through extrapolation. The payoff is a rapid increase in accuracy, provided that the solution is sufficiently smooth for the asymptotic error expansion (17.4.12) to hold. In such cases, successive extrapolated approximations converge quickly, and high accuracy can be achieved with relatively few macro-steps.

Conversely, if the solution lacks sufficient smoothness, or if the error expansion does not exhibit the expected regularity, the extrapolation process may fail to enter the asymptotic regime. In these situations, the additional computational effort does not translate into improved efficiency, and simpler methods may be preferable.

Thus, the Bulirsch–Stoer method is best viewed as a high-accuracy strategy tailored to smooth, nonstiff problems. Its cost model emphasizes that the extra algebraic work associated with extrapolation is justified only when the underlying solution permits rapid convergence of the extrapolated sequence. Under these conditions, the method provides an effective means of achieving high precision with controlled computational effort.

## 17.4.6. Comparison with Embedded Runge–Kutta Methods

The preceding cost analysis highlights the essential distinction between extrapolation-based methods and embedded Runge–Kutta schemes. Both approaches aim to achieve high accuracy with controlled computational effort, but they rely on fundamentally different mechanisms. Embedded Runge–Kutta methods adjust the stepsize based on local error estimates obtained within a single step, whereas the Bulirsch–Stoer method increases accuracy by combining multiple approximations across a hierarchy of internal resolutions.

In practical terms, embedded Runge–Kutta methods are generally easier to implement and more robust across a wide range of problem types. Their one-step structure, together with built-in error estimation, makes them well suited for adaptive integration, event detection, and problems involving nonsmooth forcing or discontinuities. Because each step is self-contained, these methods respond naturally to changes in the solution behavior, adjusting the stepsize without requiring a global restructuring of the computation.

By contrast, the Bulirsch–Stoer method relies on the existence of a smooth solution and a well-behaved asymptotic error expansion. Its efficiency depends on the ability of the extrapolation process to rapidly eliminate successive error terms. When these conditions are satisfied, the method can achieve very high accuracy with relatively few macro-steps, making it particularly attractive for problems where tight tolerances are required.

At moderate tolerances, however, the advantages of Bulirsch–Stoer are less pronounced. The overhead associated with constructing the extrapolation tableau and performing multiple modified midpoint sweeps may not be offset by the gains in accuracy. In such regimes, embedded Runge–Kutta methods often provide a better balance between simplicity, efficiency, and robustness.

The distinction becomes even more significant in challenging problem settings. For stiff systems or problems with strong nonlinearities, stability considerations dominate the cost and performance of the numerical method. In these cases, the rational-polynomial extrapolation framework underlying the Bulirsch–Stoer method is less effective, as it does not directly address stability constraints. Embedded Runge–Kutta methods, particularly when extended to implicit or stability-enhanced variants, are better suited to handle such situations.

Consequently, the choice between these approaches depends on the characteristics of the problem. Bulirsch–Stoer is most advantageous for smooth, nonstiff problems requiring high precision, while embedded Runge–Kutta methods offer greater flexibility and robustness across a broader range of applications. This comparison underscores the importance of aligning the numerical method with the dominant features of the underlying differential equation (Murugesh et al., 2025).

## 17.4.7. Rust Implementation Perspective for Bulirsch–Stoer Extrapolation

From an implementation standpoint, the Bulirsch–Stoer method is best organized in a layered manner that mirrors its mathematical construction. In a Rust-based numerical framework, the clearest design separates the algorithm into three distinct components: a reusable modified-midpoint kernel that advances the solution over one macro-step, an extrapolation tableau responsible for constructing successively higher-order approximations, and a controller that determines whether the macro-step is accepted and which subdivision sequence should be used next. This decomposition preserves the conceptual structure of the method and prevents the implementation from collapsing into a single opaque block of logic.

The first layer is the modified-midpoint kernel. Its role is to take the current solution $y_n$, a macro-step length $H$, and a chosen number of substeps $n$, then return the approximation $A_n(H)$ defined by the recurrence in (17.4.9)–(17.4.11). Because this kernel is reused repeatedly for different subdivision counts, it should be written as an independent computational unit with clearly defined inputs and outputs. In practice, this means allocating only the temporary vectors required by the recurrence, updating them in place, and returning the corrected endpoint approximation without exposing the internal stepping details to higher layers of the solver.

The second layer is the extrapolation tableau. This component receives the sequence of modified-midpoint approximations $T_{j,1} = A_{n_j}(H)$ and applies the recursion (17.4.14) to construct higher-order extrapolated values. A full two-dimensional table is not generally necessary. Since only the current and previous diagonals are needed to continue the extrapolation and estimate the error, an efficient implementation stores only the active entries. This keeps memory usage proportional to the number of extrapolation levels rather than the square of that number. It also aligns well with Rust’s preference for explicit, tightly controlled data structures.

The third layer is the controller. Its responsibility is to interpret the extrapolation results, compute the error estimate from successive diagonal entries such as (17.4.16), decide whether the macro-step should be accepted, and determine what subdivision count or macro-step size should be attempted next. This separation is important because acceptance logic, tolerance handling, and substep-sequence selection are conceptually distinct from both the midpoint recurrence and the extrapolation algebra. Keeping these concerns isolated makes the implementation easier to verify, maintain, and extend.

This layered organization has several practical advantages. It keeps the mathematical structure visible in the code, so that each part of the implementation corresponds directly to a clearly identifiable part of the numerical method. It also improves reusability, since the modified-midpoint kernel and tableau logic can be tested independently. Most importantly, it avoids turning the Bulirsch–Stoer algorithm into one monolithic function in which stepping, extrapolation, error estimation, and control flow are interwoven in a way that obscures both correctness and purpose.

In a Rust setting, this design also supports safe and efficient memory management. Temporary workspace for midpoint sweeps can be reused across calls, the tableau can be represented by compact vector-based storage, and solution updates can be committed only after the controller accepts a step. The result is an implementation that remains faithful to the numerical analysis while also satisfying the practical demands of clarity, modularity, and performance.

### Rust Implementation

Following the discussion in Section 17.4 on Richardson extrapolation and the Bulirsch–Stoer method, Program 17.4.1 provides a practical implementation of high-accuracy integration using a modified midpoint base scheme combined with an extrapolation tableau. The construction builds directly on the asymptotic error expansion in Equation (17.4.12) and the extrapolation mechanism in Equation (17.4.14), enabling successive cancellation of leading error terms. By organizing the computation into a midpoint kernel, a recursive extrapolation structure, and an acceptance controller based on the error estimate in Equation (17.4.16), the program demonstrates how very high accuracy can be achieved efficiently for smooth, nonstiff problems. The implementation reflects the layered design described in Subsection 17.4.7, maintaining a clear correspondence between the mathematical formulation and the computational structure.

At the core of the implementation is the `modified_midpoint` function, which realizes the base integration scheme defined in Equations (17.4.9)–(17.4.11). This function advances the solution over a macro-step by subdividing the interval into equal substeps and applying a leapfrog-style recurrence. The recurrence alternates between previously computed states, producing a sequence of intermediate approximations that culminate in a corrected endpoint value. The symmetry of this construction ensures that the resulting truncation error contains only even powers of the stepsize, as described in Equation (17.4.12), which is essential for the efficiency of the extrapolation process.

The extrapolation process is implemented within the `bulirsch_stoer_step` function. It constructs a sequence of approximations corresponding to different substep counts and organizes them into a triangular structure consistent with Equation (17.4.15). The Aitken–Neville recursion from Equation (17.4.14) is applied iteratively to eliminate leading error terms and generate higher-order approximations. Each new entry in the tableau builds upon previously computed values, reflecting the recursive nature of the extrapolation process.

The function `rms_error` computes the error estimate based on the difference between successive diagonal entries of the extrapolation tableau, as defined in Equation (17.4.16). This difference captures the improvement achieved by the most recent extrapolation step and serves as a practical indicator of the remaining truncation error. The error is normalized using a scale-aware measure consistent with the approach introduced in Section 17.3, ensuring that the acceptance criterion is meaningful across all components of the solution.

The `bulirsch_stoer_step` function integrates the midpoint kernel and extrapolation tableau into a single macro-step procedure. It evaluates the sequence of midpoint approximations, applies extrapolation, and checks whether the resulting approximation satisfies the prescribed tolerance. If the error criterion is met, the step is accepted and the highest-order approximation is returned. Otherwise, the computation continues with additional refinement levels or signals rejection to the outer solver.

The `solve_bulirsch_stoer` function implements the controller described in Subsection 17.4.7. It manages the progression of macro-steps, updates the solution state only upon acceptance, and adjusts the macro-step size using a heuristic adaptation rule. This separation of concerns ensures that the stepping logic, extrapolation, and control flow remain clearly delineated, improving both clarity and maintainability.

The `main` function demonstrates the complete solver using the harmonic oscillator, which is expressed as a first-order system as in Equation (17.4.5). The exact solution is available analytically, allowing the program to compute the final error and verify the high-accuracy behavior of the method. The reported statistics illustrate the efficiency of the extrapolation approach, showing that a small number of macro-steps can achieve very high precision.

```rust
// Program 17.4.1: Bulirsch-Stoer Extrapolation with a Modified Midpoint Kernel
//
// Problem statement:
// This program implements a simplified Bulirsch-Stoer integrator for smooth,
// nonstiff initial value problems. The method uses the modified midpoint
// method as a base integrator, constructs an extrapolation tableau using
// Aitken-Neville recursion, estimates the local error from successive
// extrapolated values, and accepts or rejects macro-steps according to a
// normalized tolerance criterion.

type State = Vec<f64>;

#[derive(Clone)]
struct BulirschStoerResult {
    trajectory: Vec<(f64, State)>,
    accepted_steps: usize,
    rejected_steps: usize,
    function_evaluations: usize,
}

fn add_scaled(y: &State, scale: f64, k: &State) -> State {
    y.iter()
        .zip(k.iter())
        .map(|(yi, ki)| yi + scale * ki)
        .collect()
}

fn modified_midpoint<F>(
    f: &F,
    t: f64,
    y: &State,
    h_macro: f64,
    n_substeps: usize,
) -> (State, usize)
where
    F: Fn(f64, &State) -> State,
{
    let h = h_macro / n_substeps as f64;

    let f0 = f(t, y);
    let mut evaluations = 1;

    let mut z_prev = y.clone();
    let mut z_curr = add_scaled(y, h, &f0);

    for m in 1..n_substeps {
        let t_mid = t + m as f64 * h;
        let f_mid = f(t_mid, &z_curr);
        evaluations += 1;

        let mut z_next = z_prev.clone();
        for i in 0..z_next.len() {
            z_next[i] += 2.0 * h * f_mid[i];
        }

        z_prev = z_curr;
        z_curr = z_next;
    }

    let f_end = f(t + h_macro, &z_curr);
    evaluations += 1;

    let mut corrected = z_curr.clone();
    for i in 0..corrected.len() {
        corrected[i] = 0.5 * (z_curr[i] + z_prev[i] + h * f_end[i]);
    }

    (corrected, evaluations)
}

fn rms_error(new_value: &State, previous_value: &State, y_reference: &State, atol: f64, rtol: f64) -> f64 {
    let m = new_value.len();
    let mut sum = 0.0_f64;

    for i in 0..m {
        let diff = new_value[i] - previous_value[i];
        let scale = atol + rtol * y_reference[i].abs().max(new_value[i].abs());
        let ratio = diff / scale;
        sum += ratio * ratio;
    }

    (sum / m as f64).sqrt()
}

fn bulirsch_stoer_step<F>(
    f: &F,
    t: f64,
    y: &State,
    h_macro: f64,
    atol: f64,
    rtol: f64,
    substep_sequence: &[usize],
) -> (Option<State>, f64, usize)
where
    F: Fn(f64, &State) -> State,
{
    let dimension = y.len();
    let levels = substep_sequence.len();

    let mut tableau: Vec<Vec<State>> = vec![vec![vec![0.0; dimension]; levels]; levels];
    let mut x_values = vec![0.0_f64; levels];

    let mut total_evaluations = 0;
    let mut previous_diagonal: Option<State> = None;

    for j in 0..levels {
        let n_substeps = substep_sequence[j];
        let h_internal = h_macro / n_substeps as f64;
        x_values[j] = h_internal * h_internal;

        let (base_approximation, evals) = modified_midpoint(f, t, y, h_macro, n_substeps);
        total_evaluations += evals;

        tableau[j][0] = base_approximation;

        for k in 1..=j {
            let denominator = x_values[j - k] / x_values[j] - 1.0;

            for i in 0..dimension {
                tableau[j][k][i] =
                    tableau[j][k - 1][i]
                        + (tableau[j][k - 1][i] - tableau[j - 1][k - 1][i]) / denominator;
            }
        }

        let current_diagonal = tableau[j][j].clone();

        if let Some(previous) = previous_diagonal {
            let err = rms_error(&current_diagonal, &previous, y, atol, rtol);

            if err <= 1.0 {
                return (Some(current_diagonal), err, total_evaluations);
            }
        }

        previous_diagonal = Some(current_diagonal);
    }

    let final_estimate = tableau[levels - 1][levels - 1].clone();
    let previous_estimate = tableau[levels - 1][levels - 2].clone();
    let err = rms_error(&final_estimate, &previous_estimate, y, atol, rtol);

    if err <= 1.0 {
        (Some(final_estimate), err, total_evaluations)
    } else {
        (None, err, total_evaluations)
    }
}

fn adapt_macro_step(h: f64, error: f64) -> f64 {
    let safety = 0.9;
    let min_factor = 0.25;
    let max_factor = 2.0;

    let factor = if error == 0.0 {
        max_factor
    } else {
        safety * error.powf(-0.25)
    };

    h * factor.clamp(min_factor, max_factor)
}

fn solve_bulirsch_stoer<F>(
    f: &F,
    t0: f64,
    y0: State,
    t_end: f64,
    h_initial: f64,
    h_min: f64,
    h_max: f64,
    atol: f64,
    rtol: f64,
) -> BulirschStoerResult
where
    F: Fn(f64, &State) -> State,
{
    let substep_sequence = vec![2, 4, 6, 8, 10, 12];

    let mut trajectory = Vec::new();
    let mut t = t0;
    let mut y = y0;
    let mut h = h_initial.clamp(h_min, h_max);

    let mut accepted_steps = 0;
    let mut rejected_steps = 0;
    let mut function_evaluations = 0;

    trajectory.push((t, y.clone()));

    while t < t_end {
        if t + h > t_end {
            h = t_end - t;
        }

        let (candidate, error, evals) =
            bulirsch_stoer_step(f, t, &y, h, atol, rtol, &substep_sequence);

        function_evaluations += evals;

        match candidate {
            Some(y_next) => {
                t += h;
                y = y_next;
                accepted_steps += 1;
                trajectory.push((t, y.clone()));

                h = adapt_macro_step(h, error).clamp(h_min, h_max);
            }
            None => {
                rejected_steps += 1;
                h = adapt_macro_step(h, error).clamp(h_min, h_max);

                if h <= h_min {
                    panic!("Macro-step reached h_min before satisfying the tolerance.");
                }
            }
        }
    }

    BulirschStoerResult {
        trajectory,
        accepted_steps,
        rejected_steps,
        function_evaluations,
    }
}

fn exact_solution(t: f64) -> State {
    vec![t.cos(), -t.sin()]
}

fn max_abs_error(numerical: &State, exact: &State) -> f64 {
    numerical
        .iter()
        .zip(exact.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn main() {
    println!("Bulirsch-Stoer Extrapolation with Modified Midpoint Kernel");
    println!("=========================================================\n");

    let harmonic_oscillator = |_: f64, y: &State| -> State {
        let q = y[0];
        let v = y[1];

        vec![v, -q]
    };

    let t0 = 0.0;
    let t_end = 10.0;
    let y0 = vec![1.0, 0.0];

    let h_initial = 0.5;
    let h_min = 1.0e-8;
    let h_max = 1.0;

    let atol = 1.0e-10;
    let rtol = 1.0e-8;

    let result = solve_bulirsch_stoer(
        &harmonic_oscillator,
        t0,
        y0,
        t_end,
        h_initial,
        h_min,
        h_max,
        atol,
        rtol,
    );

    let (t_final, y_final) = result
        .trajectory
        .last()
        .expect("trajectory should contain at least one point");

    let exact = exact_solution(*t_final);
    let error = max_abs_error(y_final, &exact);

    println!("Model Problem");
    println!("-------------");
    println!("Second-order equation: q''(t) = -q(t)");
    println!("First-order system:    q' = v,   v' = -q");
    println!("Initial state:         q(0) = 1, v(0) = 0\n");

    println!("Solver Parameters");
    println!("-----------------");
    println!("Initial time t0              = {:.6}", t0);
    println!("Final time                   = {:.6}", t_end);
    println!("Initial macro-step H         = {:.6}", h_initial);
    println!("Minimum macro-step           = {:.2e}", h_min);
    println!("Maximum macro-step           = {:.6}", h_max);
    println!("Absolute tolerance           = {:.2e}", atol);
    println!("Relative tolerance           = {:.2e}", rtol);
    println!("Substep sequence             = [2, 4, 6, 8, 10, 12]\n");

    println!("Solver Statistics");
    println!("-----------------");
    println!("Accepted macro-steps         = {}", result.accepted_steps);
    println!("Rejected macro-steps         = {}", result.rejected_steps);
    println!("Function evaluations         = {}", result.function_evaluations);
    println!("Stored solution points       = {}\n", result.trajectory.len());

    println!("Final-Time Accuracy");
    println!("-------------------");
    println!("Final time t                 = {:.6}", t_final);
    println!("Computed q(t)                = {:.12}", y_final[0]);
    println!("Computed v(t)                = {:.12}", y_final[1]);
    println!("Exact q(t)                   = {:.12}", exact[0]);
    println!("Exact v(t)                   = {:.12}", exact[1]);
    println!("Maximum absolute error       = {:.6e}\n", error);

    println!("Selected Trajectory Points");
    println!("--------------------------");
    println!("{:>10} {:>18} {:>18}", "t", "q(t)", "v(t)");

    let stride = (result.trajectory.len() / 10).max(1);

    for (index, (t, y)) in result.trajectory.iter().enumerate() {
        if index % stride == 0 || index + 1 == result.trajectory.len() {
            println!("{:>10.6} {:>18.10} {:>18.10}", t, y[0], y[1]);
        }
    }
}
```

Program 17.4.1 demonstrates the practical realization of Richardson extrapolation within the Bulirsch–Stoer framework. By combining a modified midpoint base integrator with a structured extrapolation tableau, the program systematically eliminates leading error terms and achieves rapid convergence to the exact solution.

The results highlight the key advantage of extrapolation methods: the ability to attain very high accuracy with relatively few macro-steps when the solution is smooth. The even-power error structure of the midpoint method plays a crucial role in this efficiency, allowing each extrapolation level to significantly increase the order of accuracy.

At the same time, the implementation illustrates the trade-offs discussed in Section 17.4.5. While the method achieves excellent accuracy, it requires multiple evaluations at different resolutions and careful management of the extrapolation process. The layered design adopted here ensures that these complexities remain manageable and that each component of the method is clearly defined.

The modular structure of the code makes it straightforward to extend the framework to more advanced configurations, including optimized substep sequences, adaptive selection of extrapolation depth, and integration with more sophisticated controllers. As such, this program provides a solid foundation for high-accuracy numerical integration and illustrates the practical implementation of the theoretical concepts developed in Section 17.4.

# 17.5. Second-Order Conservative Equations and Structure-Preserving Integration

A large and important class of differential equations arising in scientific computing possesses an intrinsic second-order structure. Unlike the general first-order formulation introduced earlier, these systems encode additional geometric and physical properties, such as conservation of energy and symplectic structure. Recognizing and preserving these properties in numerical computation is essential, particularly for long-time integration, where small structural errors can accumulate into significant qualitative deviations.

This section focuses on such second-order conservative systems, emphasizing their Hamiltonian formulation and the implications for numerical method design. Rather than treating them as generic first-order systems, it is often advantageous to exploit their inherent structure to achieve improved stability, efficiency, and physical fidelity.

## 17.5.1. Formulation and Hamiltonian Structure of Second-Order Systems

Many important models in science and engineering arise naturally in second-order form,

$$q''(t) = a(t, q) \tag{17.5.1}$$

or, in the autonomous conservative case,

$$M q''(t) = -\nabla V(q(t)) \tag{17.5.2}$$

These formulations appear in a wide range of applications, including semidiscrete wave equations, structural vibration models, molecular dynamics simulations, and gravitational $N$-body systems. In such problems, the variable $q(t) \in \mathbb{R}^d$ typically represents position, while the second derivative encodes acceleration governed by physical forces.

To expose the underlying structure, it is customary to introduce the momentum variable $p = M q'$, which transforms the second-order equation into a first-order system in phase space. In this representation, the dynamics can be expressed in Hamiltonian form with the energy function:

$$H(q,p) = \frac{1}{2} p^\top M^{-1} p + V(q) \tag{17.5.3}$$

The corresponding equations of motion are given by:

$$
\begin{aligned}
\dot{q} &= \nabla_p H = M^{-1} p, \\
\dot{p} &= -\nabla_q H = -\nabla V(q)
\end{aligned}
\tag{17.5.4}
$$

This formulation reveals that the system evolves according to the gradients of a scalar energy function, with the first equation describing the relationship between momentum and velocity, and the second encoding the forces derived from the potential $V(q)$. The Hamiltonian $H(q,p)$ represents the total energy of the system, combining kinetic and potential contributions, and is conserved along the exact solution trajectory.

For linearized systems, such as those of the form $M q'' + K q = 0$, it is often useful to write the equations in matrix form,

$$
\frac{d}{dt}
\begin{bmatrix}
q \\ p
\end{bmatrix}
=
\begin{bmatrix}
0 & M^{-1} \\
K & 0
\end{bmatrix}
\begin{bmatrix}
q \\ p
\end{bmatrix}
\tag{17.5.5}
$$

This representation makes explicit the coupling between position and momentum variables and highlights the skew-symmetric structure characteristic of Hamiltonian systems. Such structure is closely tied to conservation properties and plays a central role in the qualitative behavior of the solution.

The numerical implications of this formulation are significant. If one simply converts the system to first order and applies a general-purpose integration method, the resulting numerical solution may fail to preserve key invariants such as energy or phase-space geometry. Over long time intervals, this can lead to artificial dissipation or growth of energy, distorting the physical interpretation of the solution.

Therefore, when the differential equation possesses an exploitable structure, numerical methods should be designed to respect it. Structure-preserving integrators, which maintain invariants or geometric properties of the system, are particularly important in this context. They provide improved long-time behavior and more faithful representations of the underlying dynamics, especially in applications where conservation laws are fundamental (Blanes, Casas and Murua, 2024).

## 17.5.2. The Störmer Difference Scheme and Centered Discretization

A classical and highly effective approach to second-order differential equations is based on centered finite difference approximations. This approach exploits the symmetry of the second derivative and leads to numerical schemes that are both accurate and structurally compatible with the underlying dynamics, particularly in conservative systems.

To derive the method, consider the Taylor expansions of the solution about a point $t$:

$$q(t+h) = q(t) + h q'(t) + \frac{h^2}{2} q''(t) + \frac{h^3}{6} q^{(3)}(t) + \frac{h^4}{24} q^{(4)}(t) + \cdots \tag{17.5.6}$$

$$q(t-h) = q(t) - h q'(t) + \frac{h^2}{2} q''(t) - \frac{h^3}{6} q^{(3)}(t) + \frac{h^4}{24} q^{(4)}(t) + \cdots \tag{17.5.7}$$

Adding these two expansions eliminates all odd-order derivative terms due to symmetry, yielding:

$$q(t+h) - 2q(t) + q(t-h)= h^2 q''(t) + \mathcal{O}(h^4) \tag{17.5.8}$$

This identity provides a second-order accurate approximation to the second derivative, with a truncation error of order $\mathcal{O}(h^4)$. Substituting the differential equation $q''(t) = a(t, q)$ into this expression leads directly to the Störmer difference scheme,

$$q_{n+1} - 2q_n + q_{n-1} = h^2 a(t_n, q_n) \tag{17.5.9}$$

This scheme advances the solution using values at two previous time levels, reflecting its origin as a centered discretization. The method is explicit and requires only evaluations of the acceleration function $a(t, q)$, making it computationally efficient for large systems.

Because the scheme involves two-step recurrence, it requires an initialization procedure. Given initial position $q_0$ and velocity $v_0 = q'(t_0)$, a consistent starting value can be obtained using a Taylor expansion,

$$q_1 = q_0 + h v_0 + \frac{h^2}{2} a(t_0, q_0) \tag{17.5.10}$$

This initialization ensures that the method begins with the correct second-order accuracy.

In practice, a more numerically stable formulation is often preferred. Instead of working directly with the positions $q_n$, one introduces increments:

$$\Delta_n = q_{n+1} - q_n$$

which represent discrete velocities. The method can then be rewritten as:

$$\Delta_0 = h v_0 + \frac{h^2}{2} a(t_0, q_0), \qquad q_1 = q_0 + \Delta_0 \tag{17.5.11}$$

$$\Delta_n = \Delta_{n-1} + h^2 a(t_n, q_n), \qquad q_{n+1} = q_n + \Delta_n \tag{17.5.12}$$

This formulation avoids repeated subtraction of nearly equal quantities, which can lead to loss of numerical precision due to cancellation errors. By updating increments directly, the method maintains better numerical stability and improves the reliability of long-time integration.

The Störmer scheme thus provides a simple yet powerful method for second-order systems. Its centered structure, absence of odd-order error terms, and compatibility with conservative dynamics make it particularly suitable for problems where preserving qualitative features of the solution is important.

### Rust Implementation

Following the discussion in Section 17.5.2 on centered discretization and the Störmer difference scheme, Program 17.5.1 provides a practical implementation of the increment-based formulation of the method. The scheme, derived from the symmetric Taylor expansions in Equations (17.5.6)–(17.5.8) and expressed in the recurrence relations (17.5.11)–(17.5.12), advances the solution using position updates that inherently respect the second-order structure of the underlying differential equation. By applying this formulation to the harmonic oscillator, the program illustrates how the centered nature of the discretization leads to stable and accurate position evolution, while also highlighting the distinction between position updates and velocity diagnostics in structure-aware methods.

At the core of the implementation is the `stormer_solve` function, which realizes the increment-based formulation introduced in Equations (17.5.11)–(17.5.12). Rather than directly using the two-step recurrence in Equation (17.5.9), the method evolves the solution through increments $\Delta_n$, which represent discrete changes in position. This approach improves numerical stability by avoiding repeated subtraction of nearly equal quantities, thereby reducing cancellation errors over long integrations.

The function begins by computing the initial increment $\Delta_0$ using the Taylor-based initialization given in Equation (17.5.10). This step ensures that the method achieves second-order accuracy from the outset. The subsequent updates follow a simple pattern: the increment is first updated using the acceleration evaluated at the current state, and then the position is advanced using this updated increment. This two-stage update reflects the structure of Equation (17.5.12) and preserves the centered nature of the discretization.

The `add_scaled` function provides a reusable vector operation that computes expressions of the form $x + \alpha y$, which appear frequently in the increment and position updates. This abstraction simplifies the implementation and maintains clarity in the translation from mathematical formulation to code.

The function `centered_velocity_at_interior` computes diagnostic velocities using the centered difference approximation, which is consistent with the symmetry of the Störmer scheme. By evaluating velocities as $(q_{n+1} - q_{n-1}) / (2h)$, the program obtains a second-order accurate estimate that aligns with the centered discretization in Equation (17.5.8). Importantly, this diagnostic is applied only at interior points, reflecting the fact that the Störmer method is fundamentally position-based and does not directly propagate velocity as a primary variable.

To validate the implementation, the harmonic oscillator is used as a test problem. The exact solution is known analytically, allowing direct comparison between numerical and exact positions. This comparison highlights the second-order accuracy of the method and demonstrates how the centered discretization effectively captures the oscillatory behavior of the system.

The `main` function initializes the problem parameters, executes the solver, and reports both final position accuracy and selected interior diagnostics. By presenting both numerical and exact values, it provides insight into the behavior of the method across the integration interval and illustrates the relationship between discretization structure and numerical accuracy.

```rust
// Program 17.5.1: Störmer Difference Scheme for Second-Order Conservative Equations
//
// Problem statement:
// This program implements the Störmer difference scheme for a second-order
// initial value problem
//
//     q''(t) = a(t, q),
//
// using the increment formulation described in Equations (17.5.11) and
// (17.5.12). The harmonic oscillator q'' = -q is used as a test problem.
// The method advances positions directly; velocities are reported only as
// centered-difference diagnostics at interior points.

type State = Vec<f64>;

#[derive(Clone)]
struct StormerResult {
    positions: Vec<(f64, State)>,
}

fn acceleration(_: f64, q: &State) -> State {
    q.iter().map(|qi| -qi).collect()
}

fn add_scaled(x: &State, scale: f64, y: &State) -> State {
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| xi + scale * yi)
        .collect()
}

fn stormer_solve<A>(
    acceleration: A,
    t0: f64,
    q0: State,
    v0: State,
    h: f64,
    n_steps: usize,
) -> StormerResult
where
    A: Fn(f64, &State) -> State,
{
    let mut positions = Vec::with_capacity(n_steps + 1);

    let mut t = t0;
    let mut q = q0.clone();

    positions.push((t, q.clone()));

    let a0 = acceleration(t, &q);

    // Delta_0 = h v_0 + (h^2 / 2) a(t_0, q_0).
    let mut delta = vec![0.0; q.len()];
    for i in 0..delta.len() {
        delta[i] = h * v0[i] + 0.5 * h * h * a0[i];
    }

    // q_1 = q_0 + Delta_0.
    q = add_scaled(&q, 1.0, &delta);
    t += h;
    positions.push((t, q.clone()));

    for _ in 1..n_steps {
        let a = acceleration(t, &q);

        // Delta_n = Delta_{n-1} + h^2 a(t_n, q_n).
        for i in 0..delta.len() {
            delta[i] += h * h * a[i];
        }

        // q_{n+1} = q_n + Delta_n.
        q = add_scaled(&q, 1.0, &delta);
        t += h;
        positions.push((t, q.clone()));
    }

    StormerResult { positions }
}

fn centered_velocity_at_interior(
    positions: &[(f64, State)],
    index: usize,
    h: f64,
) -> Option<State> {
    if index == 0 || index + 1 >= positions.len() {
        return None;
    }

    let q_next = &positions[index + 1].1;
    let q_prev = &positions[index - 1].1;

    Some(
        q_next
            .iter()
            .zip(q_prev.iter())
            .map(|(qn, qp)| (qn - qp) / (2.0 * h))
            .collect(),
    )
}

fn exact_solution(t: f64) -> (f64, f64) {
    (t.cos(), -t.sin())
}

fn main() {
    println!("Störmer Difference Scheme for Second-Order Conservative Equations");
    println!("================================================================\n");

    let t0 = 0.0;
    let h = 0.05;
    let n_steps = 200;

    let q0 = vec![1.0];
    let v0 = vec![0.0];

    let result = stormer_solve(acceleration, t0, q0, v0, h, n_steps);

    let final_index = result.positions.len() - 1;
    let (t_final, q_final) = &result.positions[final_index];
    let (q_exact, _) = exact_solution(*t_final);

    println!("Model Problem");
    println!("-------------");
    println!("Second-order equation: q''(t) = -q(t)");
    println!("Initial position:       q(0) = 1");
    println!("Initial velocity:       v(0) = 0\n");

    println!("Numerical Parameters");
    println!("--------------------");
    println!("Initial time t0        = {:.6}", t0);
    println!("Step size h            = {:.6}", h);
    println!("Number of steps        = {}", n_steps);
    println!("Final time             = {:.6}\n", t_final);

    println!("Final-Time Position Accuracy");
    println!("----------------------------");
    println!("Computed q(t)           = {:.12}", q_final[0]);
    println!("Exact q(t)              = {:.12}", q_exact);
    println!(
        "Absolute position error = {:.6e}\n",
        (q_final[0] - q_exact).abs()
    );

    println!("Selected Interior Diagnostics");
    println!("-----------------------------");
    println!(
        "{:>10} {:>18} {:>18} {:>18} {:>18}",
        "t", "q_n", "v_n centered", "q_exact", "v_exact"
    );

    let stride = (result.positions.len() / 10).max(1);

    for (index, (t, q)) in result.positions.iter().enumerate() {
        if index == 0 || index + 1 == result.positions.len() {
            continue;
        }

        if index % stride == 0 {
            if let Some(v_diag) = centered_velocity_at_interior(&result.positions, index, h) {
                let (q_ex, v_ex) = exact_solution(*t);

                println!(
                    "{:>10.6} {:>18.10} {:>18.10} {:>18.10} {:>18.10}",
                    t, q[0], v_diag[0], q_ex, v_ex
                );
            }
        }
    }
}
```

Program 17.5.1 demonstrates the practical implementation of the Störmer difference scheme using its increment formulation. By exploiting the symmetry of the second-order derivative approximation and avoiding direct reliance on first-order reformulations, the method provides a natural and efficient approach for integrating second-order conservative systems.

The results illustrate the key advantage of centered discretizations: improved numerical stability and preservation of qualitative behavior over time. While the method focuses on position updates, the use of centered differences for velocity diagnostics ensures consistency with the underlying discretization without introducing additional numerical artifacts.

The implementation also highlights an important principle emphasized in Section 17.5. When a problem possesses an inherent second-order structure, it is often beneficial to preserve that structure in the numerical method rather than converting it into a first-order system. This approach leads to algorithms that are not only efficient but also better aligned with the physical and geometric properties of the system.

The modular design of the code allows for straightforward extension to more advanced structure-preserving methods, such as the velocity–Verlet scheme introduced in Section 17.5.3. These methods build on the same principles while incorporating additional features, such as explicit momentum updates and symplectic structure preservation, further enhancing long-time numerical behavior.

## 17.5.3. Velocity–Verlet Method and Symplectic Structure

For autonomous separable Hamiltonian systems, the Störmer scheme admits an equivalent and particularly insightful formulation known as the velocity–Verlet method. In terms of position $q$ and momentum $p$, the update is written as:

$$p_{n+\frac{1}{2}} = p_n - \frac{h}{2} \nabla V(q_n) \tag{17.5.13}$$

$$q_{n+1} = q_n + h \, M^{-1} p_{n+\frac{1}{2}} \tag{17.5.14}$$

$$p_{n+1} = p_{n+\frac{1}{2}} - \frac{h}{2} \nabla V(q_{n+1}) \tag{17.5.15}$$

This formulation separates the update into a sequence of half-step and full-step operations. First, the momentum is advanced by half a step under the influence of the potential. Next, the position is updated over a full step using the intermediate momentum. Finally, the momentum is updated again by another half step, now using the force evaluated at the new position. This symmetric arrangement ensures that the method treats forward and backward time directions in a balanced manner.

A deeper interpretation of this scheme is obtained by viewing it as a splitting of the Hamiltonian system into kinetic and potential components. The exact flow of the system can be decomposed into subflows corresponding to the kinetic energy $T(p) = \frac{1}{2} p^\top M^{-1} p$ and the potential energy $V(q)$. The velocity–Verlet method can then be expressed as the composition:

$$\Phi_h^{\mathrm{VV}} = \Phi_{h/2}^{V} \circ \Phi_h^{T} \circ \Phi_{h/2}^{V} \tag{17.5.16}$$

Here, $\Phi_h^{T}$ denotes the exact flow generated by the kinetic part, which advances positions while keeping momenta constant, and $\Phi_h^{V}$ denotes the flow generated by the potential part, which updates momenta while leaving positions unchanged. This composition is known as Strang splitting and provides a second-order accurate approximation to the full Hamiltonian flow.

A fundamental property of this construction is that each subflow is symplectic, meaning that it preserves the geometric structure of phase space associated with Hamiltonian systems. Because the composition of symplectic maps is itself symplectic, the velocity–Verlet method inherits this property. As a result, the numerical solution preserves the qualitative features of the exact dynamics, including the structure of phase space and the near-conservation of energy over long time intervals.

This property has profound implications for numerical simulation. While higher-order methods such as classical Runge–Kutta schemes may achieve smaller local truncation errors, they do not, in general, preserve the symplectic structure. Over long integrations, this can lead to systematic drift in conserved quantities, distorting the physical behavior of the system. In contrast, symplectic methods like velocity–Verlet maintain the underlying geometric structure, leading to more accurate long-term behavior even if their local accuracy is lower.

This explains why relatively low-order symplectic methods remain central in applications such as molecular dynamics and celestial mechanics. In these contexts, the preservation of invariants and qualitative behavior is often more important than minimizing local error at each step. The velocity–Verlet method thus represents a compelling example of how exploiting problem structure can lead to superior numerical performance in practice (Blanes, Casas and Murua, 2024).

### Rust Implementation

Following the discussion in Section 17.5.3 on the velocity–Verlet method and its symplectic structure, Program 17.5.2 provides a practical implementation of this structure-preserving integrator for separable Hamiltonian systems. The update sequence defined in Equations (17.5.13)–(17.5.15) is realized through a sequence of half-step momentum updates and full-step position updates, reflecting the Strang splitting formulation in Equation (17.5.16). By applying the method to the harmonic oscillator, the program demonstrates how symplectic integration preserves the qualitative features of the dynamics, particularly the near-conservation of energy over long time intervals, even when local truncation errors accumulate.

At the core of the implementation is the `velocity_verlet_step` function, which directly realizes the update sequence described in Equations (17.5.13)–(17.5.15). The function first computes the force at the current position and performs a half-step update of the momentum. It then advances the position over a full time step using the updated momentum, corresponding to the kinetic flow component of the Hamiltonian splitting. Finally, it evaluates the force at the new position and completes the second half-step momentum update. This symmetric arrangement ensures time-reversibility and reflects the Strang splitting structure introduced in Equation (17.5.16).

The `force` function defines the gradient of the potential energy, which in this case corresponds to the harmonic oscillator with potential $V(q) = \frac{1}{2} q^2$. This abstraction allows the method to be applied to a wide range of systems by simply modifying the force function, without altering the integration scheme itself. The `hamiltonian` function computes the total energy of the system as defined in Equation (17.5.3), combining kinetic and potential contributions. This provides a diagnostic tool for evaluating the long-time behavior of the numerical method.

The `solve_velocity_verlet` function implements the time-stepping loop. Starting from the initial state, it repeatedly applies the `velocity_verlet_step` function and records the trajectory of the system, including the energy at each step. This structure mirrors the iterative application of the symplectic map and allows for detailed analysis of both the solution and its invariants over time.

The `main` function initializes the problem parameters and executes the solver. It computes the exact solution of the harmonic oscillator for comparison and evaluates both the final-time error and the evolution of the Hamiltonian. The inclusion of energy diagnostics highlights the key advantage of symplectic integrators: while position and momentum errors may grow over time, the total energy remains nearly constant, reflecting the preservation of the underlying geometric structure.

```rust
// Program 17.5.2: Velocity-Verlet Method for Separable Hamiltonian Systems
//
// Problem statement:
// This program implements the velocity-Verlet method for an autonomous
// separable Hamiltonian system. The method advances momentum by a half step,
// position by a full step, and momentum again by a half step, corresponding
// to Equations (17.5.13)--(17.5.15). The harmonic oscillator is used as a
// test problem to demonstrate long-time near-conservation of energy.

type State = Vec<f64>;

#[derive(Clone)]
struct VerletResult {
    trajectory: Vec<(f64, State, State, f64)>,
}

fn force(q: &State) -> State {
    // For the harmonic oscillator V(q) = 1/2 q^2,
    // the force is -grad V(q) = -q.
    q.iter().map(|qi| -qi).collect()
}

fn hamiltonian(q: &State, p: &State) -> f64 {
    let kinetic: f64 = p.iter().map(|pi| 0.5 * pi * pi).sum();
    let potential: f64 = q.iter().map(|qi| 0.5 * qi * qi).sum();

    kinetic + potential
}

fn velocity_verlet_step(q: &mut State, p: &mut State, h: f64) {
    let f_old = force(q);

    // First half-step momentum update.
    for i in 0..p.len() {
        p[i] += 0.5 * h * f_old[i];
    }

    // Full-step position update.
    // Here M = I, so M^{-1}p = p.
    for i in 0..q.len() {
        q[i] += h * p[i];
    }

    let f_new = force(q);

    // Second half-step momentum update.
    for i in 0..p.len() {
        p[i] += 0.5 * h * f_new[i];
    }
}

fn solve_velocity_verlet(
    t0: f64,
    q0: State,
    p0: State,
    h: f64,
    n_steps: usize,
) -> VerletResult {
    let mut trajectory = Vec::with_capacity(n_steps + 1);

    let mut t = t0;
    let mut q = q0;
    let mut p = p0;

    let energy0 = hamiltonian(&q, &p);
    trajectory.push((t, q.clone(), p.clone(), energy0));

    for _ in 0..n_steps {
        velocity_verlet_step(&mut q, &mut p, h);
        t += h;

        let energy = hamiltonian(&q, &p);
        trajectory.push((t, q.clone(), p.clone(), energy));
    }

    VerletResult { trajectory }
}

fn exact_solution(t: f64) -> (f64, f64) {
    (t.cos(), -t.sin())
}

fn main() {
    println!("Velocity-Verlet Method for Separable Hamiltonian Systems");
    println!("=======================================================\n");

    let t0 = 0.0;
    let h = 0.05;
    let n_steps = 2000;

    let q0 = vec![1.0];
    let p0 = vec![0.0];

    let result = solve_velocity_verlet(t0, q0, p0, h, n_steps);

    let (t_final, q_final, p_final, energy_final) = result
        .trajectory
        .last()
        .expect("trajectory should contain at least one point");

    let energy_initial = result.trajectory[0].3;
    let (q_exact, p_exact) = exact_solution(*t_final);

    let max_energy_deviation = result
        .trajectory
        .iter()
        .map(|(_, _, _, energy)| (energy - energy_initial).abs())
        .fold(0.0_f64, f64::max);

    println!("Model Problem");
    println!("-------------");
    println!("Hamiltonian:             H(q,p) = 1/2 p^2 + 1/2 q^2");
    println!("Equation:                q''(t) = -q(t)");
    println!("Initial position:         q(0) = 1");
    println!("Initial momentum:         p(0) = 0\n");

    println!("Numerical Parameters");
    println!("--------------------");
    println!("Initial time t0            = {:.6}", t0);
    println!("Step size h                = {:.6}", h);
    println!("Number of steps            = {}", n_steps);
    println!("Final time                 = {:.6}\n", t_final);

    println!("Final-Time Accuracy");
    println!("-------------------");
    println!("Computed q(t)              = {:.12}", q_final[0]);
    println!("Computed p(t)              = {:.12}", p_final[0]);
    println!("Exact q(t)                 = {:.12}", q_exact);
    println!("Exact p(t)                 = {:.12}", p_exact);
    println!(
        "Absolute position error    = {:.6e}",
        (q_final[0] - q_exact).abs()
    );
    println!(
        "Absolute momentum error    = {:.6e}\n",
        (p_final[0] - p_exact).abs()
    );

    println!("Energy Behavior");
    println!("---------------");
    println!("Initial energy             = {:.12}", energy_initial);
    println!("Final energy               = {:.12}", energy_final);
    println!(
        "Final energy deviation     = {:.6e}",
        (energy_final - energy_initial).abs()
    );
    println!(
        "Maximum energy deviation   = {:.6e}\n",
        max_energy_deviation
    );

    println!("Selected Trajectory Points");
    println!("--------------------------");
    println!(
        "{:>10} {:>18} {:>18} {:>18}",
        "t", "q_n", "p_n", "H(q,p)"
    );

    let stride = (result.trajectory.len() / 10).max(1);

    for (index, (t, q, p, energy)) in result.trajectory.iter().enumerate() {
        if index % stride == 0 || index + 1 == result.trajectory.len() {
            println!(
                "{:>10.6} {:>18.10} {:>18.10} {:>18.12}",
                t, q[0], p[0], energy
            );
        }
    }
}
```

Program 17.5.2 demonstrates the practical implementation of the velocity–Verlet method as a symplectic integrator for second-order conservative systems. By decomposing the Hamiltonian flow into kinetic and potential components and applying a symmetric composition, the method preserves the geometric structure of phase space and maintains near-conservation of energy over long integrations.

The numerical results clearly illustrate the distinction between local accuracy and long-time qualitative behavior. Although the position and momentum errors increase over extended time intervals, the total energy remains bounded and oscillates around its exact value. This behavior is characteristic of symplectic methods and underscores their importance in applications where preserving invariants is critical.

The modular design of the code allows for straightforward extension to more complex systems, including higher-dimensional Hamiltonian problems and alternative potential functions. It also provides a foundation for exploring higher-order symplectic integrators and adaptive strategies, as discussed in Section 17.5.4. Overall, this program exemplifies the central principle of structure-preserving integration: by aligning the numerical method with the intrinsic properties of the differential equation, one can achieve superior long-time performance even with relatively simple algorithms.

## 17.5.4. Computational Properties and Extensions of Symplectic Second-Order Methods

The computational efficiency of symplectic methods for second-order systems is one of their most compelling features, particularly in large-scale simulations where both accuracy and cost must be carefully balanced. The velocity–Verlet method, in particular, achieves a highly favorable balance between computational effort, memory usage, and long-time qualitative fidelity.

For a system of dimension $d$, suppose that one evaluation of the force $-\nabla V(q)$ incurs a cost $C_F(d)$. After initialization, the velocity–Verlet method requires only one new force evaluation per time step. This efficiency arises because the force computed at the end of one step can be reused at the beginning of the next, reducing the number of evaluations needed. In addition, the method requires only $O(d)$ storage, since it operates with a small number of vectors representing positions, momenta, and intermediate quantities. These properties make the method especially attractive for large particle systems, such as those encountered in molecular dynamics and astrophysical simulations.

More advanced symplectic integrators extend these ideas to achieve higher-order accuracy while preserving the underlying geometric structure. Symplectic Runge–Kutta–Nyström methods with $s$ stages, for example, generalize the velocity–Verlet scheme by incorporating multiple force evaluations within each step. These methods require $s$ force evaluations and $O(sd)$ storage, reflecting the need to store intermediate stage values. Despite the increased cost, they can achieve significantly higher order of accuracy without sacrificing symplecticity, making them suitable for problems where both long-time stability and high precision are required.

Recent research has further expanded the scope and capability of symplectic methods. New sixth-order explicit symplectic Runge–Kutta–Nyström schemes have been developed with improved stability and accuracy properties, providing more efficient high-order alternatives for conservative systems (Pan, Zhang and Zhang, 2025). In addition, explicit symplectic integrators have been constructed for nonseparable Hamiltonians by embedding the system in an extended phase space and exploiting invariant submanifolds. This approach allows the preservation of symplectic structure even in cases where the Hamiltonian cannot be decomposed into simple kinetic and potential components.

Another important direction involves adaptive symplectic methods. Traditional symplectic integrators typically rely on fixed step sizes to maintain their structure-preserving properties. However, recent developments have introduced adaptive strategies that combine time transformations, interpolation techniques, and quasi-Newton solvers. These methods aim to retain long-time energy stability while allowing flexibility in step size selection, thereby improving efficiency in problems with varying dynamical behavior (Ye et al., 2025).

Taken together, these computational and theoretical developments demonstrate that symplectic second-order methods provide a powerful framework for the numerical integration of conservative systems. Their combination of low computational cost, minimal storage requirements, and excellent long-time behavior makes them indispensable in applications where preserving the qualitative structure of the solution is as important as achieving numerical accuracy.

## 17.5.5. Practical Implications and Applications of Symplectic Second-Order Methods

The theoretical advantages of structure-preserving methods become most evident in long-time simulations, where qualitative fidelity is as important as pointwise accuracy. A representative and widely studied application is long-time orbital dynamics, particularly in gravitational $N$-body systems, governed by:

$$m_i q_i'' = -\sum_{j \ne i} G m_i m_j \frac{q_i - q_j}{|q_i - q_j|^3} \tag{17.5.17}$$

In such systems, the numerical objective extends beyond minimizing short-term truncation error. Instead, the focus is on preserving essential qualitative features of the dynamics over extended time intervals. These include the structure of phase space, the presence of resonances, and the approximate conservation of invariants such as total energy and angular momentum. Small violations of these properties can accumulate over time, leading to unphysical behavior such as artificial orbital decay or energy drift.

The challenge is further compounded by the presence of multiple time scales and close encounters between particles. When two bodies approach each other closely, the forces become large and rapidly varying, requiring smaller time steps to resolve the interaction accurately. At the same time, other parts of the system may evolve more slowly, creating a tension between local accuracy and global efficiency. These features motivate the development of adaptive symplectic integrators, which aim to retain structure-preserving properties while accommodating variable time stepping (Ye et al., 2025).

From a practical standpoint, a clear guideline emerges. When the underlying model is genuinely conservative and naturally expressed as a second-order system in position, it is generally preferable to preserve this structure in the numerical method. Symplectic integrators such as the Störmer–Verlet scheme or higher-order Runge–Kutta–Nyström methods are specifically designed to exploit this formulation. By maintaining the geometric properties of the system, they provide superior long-time behavior compared to general-purpose methods.

In contrast, converting the system to a generic first-order form and applying a black-box integration method may yield acceptable short-term accuracy but can degrade the qualitative behavior over long integrations. The loss of symplectic structure can lead to systematic drift in conserved quantities, undermining the reliability of the simulation.

These considerations underscore a broader principle in numerical analysis: the structure of the differential equation should guide the choice of numerical method. For conservative second-order systems, preserving the Hamiltonian framework is often more important than achieving higher local accuracy, particularly in applications where long-time dynamics are of primary interest.

# 17.6. Stiff Sets of Equations and Stability-Dominated Integration

In many applications, the principal limitation on numerical integration is not the accuracy of the approximation, but the stability of the method. This situation arises when the underlying system exhibits widely separated time scales, leading to stiffness. In such cases, standard explicit methods become inefficient because the step size must be chosen according to stability constraints rather than the desired level of accuracy. Understanding the mechanism of stiffness is therefore essential for selecting appropriate numerical strategies.

## 17.6.1. Definition and Fundamental Mechanism of Stiffness

A system of differential equations is called stiff when stability, rather than accuracy, determines the allowable stepsize. This phenomenon can be illustrated using the linear system:

$$y'(t) = A y(t), \qquad A = S \Lambda S^{-1}, \qquad \Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_d) \tag{17.6.1}$$

where the eigenvalues satisfy $\Re(\lambda_i) < 0$, ensuring that all components of the solution decay over time.

The defining feature of stiffness is the presence of eigenvalues with widely varying magnitudes. For example, one component of the solution may decay at a moderate rate, such as $e^{-t}$, while another decays extremely rapidly, such as $e^{-1000t}$. Although the fast-decaying components quickly become negligible from a modeling perspective, they impose severe restrictions on the numerical method.

Explicit integration schemes must choose a step size small enough to remain stable with respect to the fastest decaying mode. In the example above, stability may require $h \ll 10^{-3}$, even though the slow dynamics evolve on a much larger time scale. As a result, the numerical method is forced to take many unnecessarily small steps to resolve dynamics that are no longer relevant to the overall solution.

This mismatch between physical time scales and numerical stability constraints is the essence of stiffness. The system itself may not require fine temporal resolution for accurate modeling, yet the numerical method imposes such a requirement to avoid instability. Consequently, computational effort is expended on resolving transient behavior that has little impact on the long-term solution.

From a broader perspective, stiffness reflects the coexistence of fast and slow dynamics within the same system. Efficient numerical methods must therefore distinguish between these components and allocate computational resources appropriately. In particular, they should handle rapidly decaying modes in a stable manner without forcing the step size to be determined by the smallest time scale.

This observation motivates the development of specialized methods for stiff systems, which are designed to maintain stability even for large step sizes. Such methods enable the integration to proceed at a rate dictated by the slow dynamics, thereby improving efficiency while preserving accuracy (Günther and Sandu, 2025).

## 17.6.2. Stability Functions and the Dahlquist Test Equation

The mechanism underlying stiffness is most clearly revealed through the analysis of numerical stability on a simple model problem, known as the Dahlquist test equation,

$$y' = \lambda y, \qquad \Re(\lambda) < 0 \tag{17.6.2}$$

This equation captures the essential behavior of linear systems with decaying modes. When a one-step numerical method is applied, the continuous evolution is replaced by a discrete recurrence of the form:

$$y_{n+1} = R(z) \, y_n, \qquad z = h\lambda \tag{17.6.3}$$

where $R(z)$ is the stability function of the method. The variable $z$ combines the step size $h$ with the eigenvalue $\lambda$, and therefore encodes both the dynamics of the system and the discretization.

The behavior of the numerical method is governed entirely by the properties of $R(z)$. In particular, stability requires that repeated application of the recurrence does not amplify errors, which leads to the condition $|R(z)| \le 1$. The set of values of $z$ satisfying this condition defines the region of absolute stability.

For the explicit Euler method, the stability function is:

$$R(z) = 1 + z \tag{17.6.4}$$

In this case, stability requires $|1 + z| < 1$, which restricts $z$ to a bounded region in the complex plane. For real negative $z$, this condition implies that $h|\lambda|$ must be sufficiently small. Consequently, when $\lambda$ has large magnitude, the step size must be correspondingly small to maintain stability. This limitation is the source of inefficiency in explicit methods when applied to stiff problems.

In contrast, the implicit (backward) Euler method leads to the relation:

$$y_{n+1} = y_n + h\lambda y_{n+1}, \qquad R(z) = \frac{1}{1 - z} \tag{17.6.5}$$

For this method, the stability function satisfies $|R(z)| \le 1$ for all $\Re(z) \le 0$. This property is known as A-stability and implies that the entire left half of the complex plane lies within the stability region. As a result, the method remains stable for arbitrarily large step sizes when applied to systems with decaying modes.

An even stronger property is obtained by examining the behavior of $R(z)$ as $z \to -\infty$,

$$\lim_{z \to -\infty} R(z) = 0 \tag{17.6.6}$$

This limit shows that highly stable modes are strongly damped by the numerical method. Such behavior is referred to as L-stability and is particularly desirable in stiff problems, where rapidly decaying components should not influence the long-term solution.

These concepts explain the fundamental distinction between explicit and implicit methods in the context of stiffness. Explicit methods have bounded stability regions and therefore require small step sizes to handle large negative eigenvalues. Implicit methods, by contrast, possess stability regions that include the entire left half-plane, allowing them to take large steps without instability. This makes them the preferred choice for stiff systems, where stability considerations dominate the numerical integration process.

## 17.6.3. Implicit Methods and Newton Linearization for Stiff Systems

For nonlinear systems of ordinary differential equations,

$$\dot{y} = f(t,y) \tag{17.6.7}$$

implicit methods provide the stability properties required to handle stiffness, but at the cost of solving nonlinear equations at each time step. This introduces a fundamentally different computational structure compared to explicit methods, where each step is obtained through direct evaluation.

In the case of the implicit (backward) Euler method, advancing the solution from $t_n$ to $t_{n+1}$ requires solving the nonlinear equation:

$$G(y_{n+1}) := y_{n+1} - y_n - h f(t_{n+1}, y_{n+1}) = 0 \tag{17.6.8}$$

This equation defines $y_{n+1}$ implicitly, since the unknown appears inside the function $f$. As a result, each time step involves the solution of a nonlinear system of equations in $\mathbb{R}^d$.

A standard approach to solving this system is to apply Newton’s method. In its simplest form, a single Newton linearization around the known value $y_n$ leads to the linear system:

$$
\begin{aligned}
\left(I - h J_n\right)\delta_n &= h f(t_n, y_n) \\
J_n &= \frac{\partial f}{\partial y}(t_n, y_n)
\end{aligned}
\tag{17.6.9}
$$

followed by the update:

$$y_{n+1} = y_n + \delta_n \tag{17.6.10}$$

Here, $J_n$ denotes the Jacobian matrix of $f$ evaluated at $(t_n, y_n)$, and $\delta_n$ represents the correction applied to the current solution estimate. This linearization approximates the nonlinear problem by a local linear model, allowing the step to be computed through the solution of a linear system.

This formulation reveals a key computational feature of stiff integration. Unlike explicit methods, where the dominant cost is the evaluation of $f(t,y)$, implicit methods require the repeated solution of linear systems involving matrices of the form:

$$W = I - \gamma h J \tag{17.6.11}$$

The matrix $W$ arises from the Jacobian of the nonlinear system and depends on both the step size $h$ and the structure of the differential equation. Solving systems involving $W$ typically dominates the computational cost, especially for large-scale problems where $d$ is large.

Consequently, the efficiency of stiff solvers depends heavily on linear algebra techniques. Efficient factorization, reuse of matrix structures across steps, and approximation strategies for the Jacobian all play a central role in practical implementations. In many cases, the cost of forming and solving these linear systems far exceeds the cost of evaluating the function $f$, making linear algebra the primary computational bottleneck.

This shift in computational emphasis distinguishes stiff solvers from nonstiff methods. It also explains why the design of efficient stiff integrators involves not only numerical analysis of differential equations, but also careful consideration of matrix computations and their implementation.

### Rust Implementation

Following the discussion in Section 17.6.3 on implicit methods and Newton linearization for stiff systems, Program 17.6.1 provides a practical implementation of the backward Euler method with a Newton-based solution strategy for the nonlinear system defined in Equation (17.6.8). Unlike explicit methods, where each step is computed directly, the implicit formulation requires solving a nonlinear equation at every time step. This program demonstrates how Newton’s method, as introduced in Equations (17.6.9)–(17.6.10), transforms the nonlinear problem into a sequence of linear systems involving the matrix $W = I - hJ$ (Equation (17.6.11)). Using a stiff test equation with a known analytical solution, the implementation illustrates the characteristic shift in computational cost toward linear algebra operations and highlights the stability advantages of implicit integration for stiff problems.

At the core of the implementation is the `backward_euler_newton_step` function, which realizes the implicit update defined by Equation (17.6.8). For each time step, the method seeks a solution $y_{n+1}$ such that the residual function vanishes. This residual is constructed in the `residual_backward_euler` function, which evaluates the difference between the trial solution and the backward Euler update formula. This formulation directly reflects the implicit nature of the method, where the unknown solution appears inside the function evaluation.

To solve the nonlinear equation, the program applies Newton’s method. The Jacobian matrix $J_n = \partial f / \partial y$ is computed using the `stiff_jacobian` function, and the matrix $W = I - hJ_n$ is formed in the `backward_euler_jacobian` function according to Equation (17.6.11). This matrix defines the linear system that must be solved at each Newton iteration. The function `solve_linear_system` implements a basic Gaussian elimination procedure to compute the Newton correction $\delta_n$, which is then used to update the trial solution as described in Equation (17.6.10).

The `backward_euler_newton_step` function combines these components into an iterative procedure. Starting from an initial guess obtained via an explicit Euler prediction, it repeatedly evaluates the residual and solves the corresponding linear system until convergence is achieved. The convergence criteria are based on both the residual norm and the magnitude of the Newton correction, ensuring that the solution satisfies the nonlinear equation to the prescribed tolerance.

The `solve_backward_euler` function orchestrates the time-stepping process. It applies the Newton-based solver at each step, accumulates statistics on the number of Newton iterations, and records whether any steps fail to converge. This structure highlights the computational overhead associated with implicit methods, where each time step involves solving one or more linear systems rather than a simple function evaluation.

The `main` function demonstrates the solver using a stiff scalar equation with a known analytical solution. By comparing the numerical and exact solutions, the program verifies the accuracy of the method. The reported statistics provide insight into the efficiency of the Newton iteration, showing that for this linear problem, convergence is achieved rapidly with minimal iterations per step.

```rust
// Program 17.6.1: Backward Euler with Newton Linearization for a Stiff System
//
// Problem statement:
// This program implements the backward Euler method for a stiff nonlinear
// initial value problem. Each implicit step is solved by Newton iteration,
// following the nonlinear residual formulation in Equation (17.6.8). The
// Newton correction is obtained by solving a linear system involving the
// Jacobian matrix, as described in Equations (17.6.9)--(17.6.11).

type State = Vec<f64>;
type Matrix = Vec<Vec<f64>>;

#[derive(Clone)]
struct SolverResult {
    trajectory: Vec<(f64, State)>,
    newton_iterations: usize,
    failed_steps: usize,
}

fn stiff_rhs_time_dependent(t: f64, y: &State) -> State {
    vec![-1000.0 * (y[0] - t.cos()) - t.sin()]
}

fn stiff_jacobian(_: f64, _: &State) -> Matrix {
    vec![vec![-1000.0]]
}

fn residual_backward_euler<F>(
    f: &F,
    t_next: f64,
    y_old: &State,
    y_trial: &State,
    h: f64,
) -> State
where
    F: Fn(f64, &State) -> State,
{
    let f_value = f(t_next, y_trial);

    y_trial
        .iter()
        .zip(y_old.iter())
        .zip(f_value.iter())
        .map(|((ynext, yold), fi)| ynext - yold - h * fi)
        .collect()
}

fn backward_euler_jacobian<J>(
    jacobian: &J,
    t_next: f64,
    y_trial: &State,
    h: f64,
) -> Matrix
where
    J: Fn(f64, &State) -> Matrix,
{
    let j = jacobian(t_next, y_trial);
    let n = j.len();

    let mut w = vec![vec![0.0; n]; n];

    for i in 0..n {
        for k in 0..n {
            w[i][k] = -h * j[i][k];
        }
        w[i][i] += 1.0;
    }

    w
}

fn solve_linear_system(mut a: Matrix, mut b: State) -> State {
    let n = b.len();

    for pivot in 0..n {
        let mut max_row = pivot;
        let mut max_value = a[pivot][pivot].abs();

        for row in pivot + 1..n {
            if a[row][pivot].abs() > max_value {
                max_value = a[row][pivot].abs();
                max_row = row;
            }
        }

        if max_value < 1.0e-14 {
            panic!("Singular or nearly singular matrix in Newton solve.");
        }

        if max_row != pivot {
            a.swap(pivot, max_row);
            b.swap(pivot, max_row);
        }

        for row in pivot + 1..n {
            let factor = a[row][pivot] / a[pivot][pivot];

            for col in pivot..n {
                a[row][col] -= factor * a[pivot][col];
            }

            b[row] -= factor * b[pivot];
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

    x
}

fn norm_inf(x: &State) -> f64 {
    x.iter().map(|xi| xi.abs()).fold(0.0_f64, f64::max)
}

fn backward_euler_newton_step<F, J>(
    f: &F,
    jacobian: &J,
    t_old: f64,
    y_old: &State,
    h: f64,
    newton_tol: f64,
    max_newton: usize,
) -> (State, usize, bool)
where
    F: Fn(f64, &State) -> State,
    J: Fn(f64, &State) -> Matrix,
{
    let t_next = t_old + h;

    // Explicit Euler prediction as the initial Newton guess.
    let f_old = f(t_old, y_old);
    let mut y_trial: State = y_old
        .iter()
        .zip(f_old.iter())
        .map(|(yi, fi)| yi + h * fi)
        .collect();

    for iteration in 1..=max_newton {
        let residual = residual_backward_euler(f, t_next, y_old, &y_trial, h);

        if norm_inf(&residual) < newton_tol {
            return (y_trial, iteration - 1, true);
        }

        let w = backward_euler_jacobian(jacobian, t_next, &y_trial, h);

        // Newton correction solves:
        //
        //     W delta = -G(y_trial).
        let rhs: State = residual.iter().map(|ri| -ri).collect();
        let delta = solve_linear_system(w, rhs);

        for i in 0..y_trial.len() {
            y_trial[i] += delta[i];
        }

        if norm_inf(&delta) < newton_tol {
            return (y_trial, iteration, true);
        }
    }

    (y_trial, max_newton, false)
}

fn solve_backward_euler<F, J>(
    f: &F,
    jacobian: &J,
    t0: f64,
    y0: State,
    h: f64,
    n_steps: usize,
    newton_tol: f64,
    max_newton: usize,
) -> SolverResult
where
    F: Fn(f64, &State) -> State,
    J: Fn(f64, &State) -> Matrix,
{
    let mut trajectory = Vec::with_capacity(n_steps + 1);

    let mut t = t0;
    let mut y = y0;

    let mut total_newton_iterations = 0;
    let mut failed_steps = 0;

    trajectory.push((t, y.clone()));

    for _ in 0..n_steps {
        let (y_next, iterations, converged) =
            backward_euler_newton_step(f, jacobian, t, &y, h, newton_tol, max_newton);

        total_newton_iterations += iterations;

        if !converged {
            failed_steps += 1;
        }

        t += h;
        y = y_next;

        trajectory.push((t, y.clone()));
    }

    SolverResult {
        trajectory,
        newton_iterations: total_newton_iterations,
        failed_steps,
    }
}

fn exact_solution(t: f64) -> f64 {
    t.cos()
}

fn main() {
    println!("Backward Euler with Newton Linearization for a Stiff System");
    println!("==========================================================\n");

    let t0 = 0.0;
    let t_end = 1.0;
    let h = 0.05;
    let n_steps = ((t_end - t0) / h) as usize;

    let y0 = vec![1.0];

    let newton_tol = 1.0e-12;
    let max_newton = 20;

    let result = solve_backward_euler(
        &stiff_rhs_time_dependent,
        &stiff_jacobian,
        t0,
        y0,
        h,
        n_steps,
        newton_tol,
        max_newton,
    );

    let (t_final, y_final) = result
        .trajectory
        .last()
        .expect("trajectory should contain at least one point");

    let y_exact = exact_solution(*t_final);

    println!("Model Problem");
    println!("-------------");
    println!("Stiff scalar equation:");
    println!("  y'(t) = -1000(y(t) - cos(t)) - sin(t)");
    println!("Exact solution:");
    println!("  y(t) = cos(t), provided y(0) = 1\n");

    println!("Numerical Parameters");
    println!("--------------------");
    println!("Initial time t0              = {:.6}", t0);
    println!("Final time                   = {:.6}", t_end);
    println!("Step size h                  = {:.6}", h);
    println!("Number of steps              = {}", n_steps);
    println!("Newton tolerance             = {:.2e}", newton_tol);
    println!("Maximum Newton iterations    = {}\n", max_newton);

    println!("Solver Statistics");
    println!("-----------------");
    println!(
        "Total Newton iterations      = {}",
        result.newton_iterations
    );
    println!("Failed Newton steps          = {}\n", result.failed_steps);

    println!("Final-Time Accuracy");
    println!("-------------------");
    println!("Computed y(t)                = {:.12}", y_final[0]);
    println!("Exact y(t)                   = {:.12}", y_exact);
    println!(
        "Absolute error               = {:.6e}\n",
        (y_final[0] - y_exact).abs()
    );

    println!("Selected Trajectory Points");
    println!("--------------------------");
    println!("{:>10} {:>18} {:>18} {:>18}", "t", "y_n", "exact", "abs error");

    let stride = (result.trajectory.len() / 10).max(1);

    for (index, (t, y)) in result.trajectory.iter().enumerate() {
        if index % stride == 0 || index + 1 == result.trajectory.len() {
            let exact = exact_solution(*t);
            println!(
                "{:>10.6} {:>18.10} {:>18.10} {:>18.6e}",
                t,
                y[0],
                exact,
                (y[0] - exact).abs()
            );
        }
    }
}
```

Program 17.6.1 demonstrates the practical implementation of backward Euler integration for stiff systems using Newton linearization. The results highlight the fundamental trade-off discussed in Section 17.6.3: while implicit methods require solving nonlinear equations at each step, they provide superior stability properties that allow larger step sizes compared to explicit methods.

The example illustrates how the computational focus shifts from function evaluation to linear algebra operations. The construction and solution of linear systems involving the matrix $W = I - hJ$ dominate the cost of the method, emphasizing the importance of efficient matrix factorization and reuse strategies in large-scale applications.

The Newton iteration converges rapidly in this example due to the linear structure of the test problem, but for more complex nonlinear systems, additional considerations such as damping strategies or inexact solves may be required. This highlights the broader role of numerical linear algebra in the design of stiff solvers.

The modular structure of the implementation allows it to be extended to more advanced methods, such as Rosenbrock schemes and fully implicit Runge–Kutta methods, which build on the same principles while improving efficiency and robustness. As such, this program provides a foundational example of stiff integration and illustrates the practical realization of the concepts developed in Section 17.6.3.

## 17.6.4. Rosenbrock Methods and Linearly Implicit Integration

Rosenbrock methods provide an important compromise between fully implicit Runge–Kutta schemes and explicit methods for stiff systems. Their defining feature is that they avoid nonlinear solves at each stage while still retaining strong stability properties. This is achieved by linearizing the problem within each step and solving a sequence of linear systems involving a fixed Jacobian.

For a nonlinear system $\dot{y} = f(t,y)$, Rosenbrock methods construct stage increments $k_i$ by solving linear systems of the form:

$$
\left(I - \gamma h J_n\right) k_i
= h\, f\!\left(t_n + c_i h,\; y_n + \sum_{j<i} a_{ij} k_j \right)
+ h\, J_n \sum_{j<i} \gamma_{ij} k_j
\tag{17.6.12}
$$

followed by the update:

$$y_{n+1} = y_n + \sum_{i=1}^{s} b_i k_i \tag{17.6.13}$$

Here, $J_n = \frac{\partial f}{\partial y}(t_n, y_n)$ is the Jacobian evaluated at the beginning of the step, and the coefficients $a_{ij}$, $b_i$, $c_i$, and $\gamma_{ij}$ define the particular Rosenbrock scheme. The parameter $\gamma$ determines the structure of the linear systems and plays a role analogous to that of implicitness in fully implicit methods.

The key computational advantage of this formulation lies in the matrix $I - \gamma h J_n$, which remains the same for all stages within a given time step. This matrix can therefore be factorized once and reused across all stage computations. As a result, the cost of solving the linear systems is significantly reduced compared to fully implicit Runge–Kutta methods, where each stage typically requires solving a distinct nonlinear system.

From a numerical perspective, Rosenbrock methods inherit much of the stability behavior of implicit methods. Because the linear systems incorporate the Jacobian, the methods are able to handle stiff components effectively, allowing larger step sizes than explicit schemes. At the same time, by avoiding nonlinear iteration, they reduce the complexity and computational overhead associated with implicit solvers.

This balance makes Rosenbrock methods particularly attractive for moderately stiff systems, where full implicitness may be unnecessarily expensive, but explicit methods are insufficiently stable. They are especially useful in applications where the Jacobian can be computed or approximated efficiently and where linear system solves can be performed with moderate cost.

In summary, Rosenbrock methods occupy an intermediate position in the spectrum of numerical integrators. They combine the stability advantages of implicit methods with a computational structure that is closer to explicit schemes, providing an efficient and practical approach to stiff integration in many applications.

### Rust Implementation

Following the discussion in Section 17.6.4 on Rosenbrock methods and linearly implicit integration, Program 17.6.2 provides a practical implementation of a one-stage Rosenbrock scheme for stiff systems. Unlike fully implicit methods introduced in Section 17.6.3, which require nonlinear solves at each step, Rosenbrock methods linearize the problem within each step and solve a sequence of linear systems involving the matrix defined in Equation (17.6.11). This program demonstrates how the stage equation in Equation (17.6.12) and the update formula in Equation (17.6.13) can be realized computationally using a fixed Jacobian per step. By applying the method to a stiff autonomous problem, the implementation highlights the efficiency gained by avoiding Newton iterations while retaining stability properties appropriate for stiff dynamics.

At the core of the implementation is the `rosenbrock_euler_step` function, which realizes the stage computation described in Equation (17.6.12). For each time step, the Jacobian matrix is evaluated at the current state and used to construct the matrix $W = I - \gamma h J_n$ as defined in Equation (17.6.11). This matrix remains fixed throughout the stage computation, reflecting the defining feature of Rosenbrock methods: the reuse of a single linear system structure within each step. The right-hand side of the linear system is formed using the function evaluation at the current state, and solving this system yields the stage increment $k_1$, which is then used to update the solution according to Equation (17.6.13).

The function `form_rosenbrock_matrix` constructs the matrix $W$ by combining the identity matrix with the scaled Jacobian. This step translates the theoretical formulation into a concrete matrix suitable for numerical linear algebra operations. The `solve_linear_system` function implements Gaussian elimination to solve the resulting linear system. Although simple, this implementation clearly illustrates the central computational task in Rosenbrock methods: solving linear systems efficiently at each step.

The `solve_rosenbrock` function manages the time integration process. It repeatedly applies the Rosenbrock step, updates the solution, and records the trajectory. A key feature of this implementation is that only one linear system is solved per step, and no nonlinear iterations are required. This reflects the computational advantage emphasized in Section 17.6.4, where Rosenbrock methods reduce the complexity of implicit integration by avoiding repeated nonlinear solves.

The `stiff_rhs` and `stiff_jacobian` functions define the test problem and its Jacobian. Because the problem is autonomous, the Jacobian remains constant, which simplifies the implementation and highlights the efficiency of the method. The `exact_solution` function provides a reference solution, allowing the numerical results to be validated and the accuracy of the method to be assessed.

The `main` function initializes the problem parameters, executes the solver, and reports the results. It includes diagnostics such as the number of linear solves and the absence of nonlinear iterations, reinforcing the distinction between Rosenbrock methods and fully implicit schemes. The output also compares the numerical solution with the exact solution, demonstrating the method’s ability to handle stiff behavior effectively.

```rust
// Program 17.6.2: Rosenbrock-Euler Method for a Stiff Autonomous Problem
//
// Problem statement:
// This program implements a one-stage Rosenbrock method for a stiff autonomous
// initial value problem. The method forms the matrix W = I - gamma h J_n once
// per step and solves one linear system for the stage increment. This
// demonstrates the linearly implicit structure described in Equation (17.6.12),
// followed by the update in Equation (17.6.13).

type State = Vec<f64>;
type Matrix = Vec<Vec<f64>>;

#[derive(Clone)]
struct SolverResult {
    trajectory: Vec<(f64, State)>,
    linear_solves: usize,
}

fn stiff_rhs(_: f64, y: &State) -> State {
    // Stiff autonomous test problem:
    //
    //     y'(t) = -1000(y(t) - 1)
    //
    // with exact solution y(t) = 1 - exp(-1000t), provided y(0) = 0.
    vec![-1000.0 * (y[0] - 1.0)]
}

fn stiff_jacobian(_: f64, _: &State) -> Matrix {
    vec![vec![-1000.0]]
}

fn form_rosenbrock_matrix(jacobian: &Matrix, h: f64, gamma: f64) -> Matrix {
    let n = jacobian.len();
    let mut w = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            w[i][j] = -gamma * h * jacobian[i][j];
        }
        w[i][i] += 1.0;
    }

    w
}

fn solve_linear_system(mut a: Matrix, mut b: State) -> State {
    let n = b.len();

    for pivot in 0..n {
        let mut max_row = pivot;
        let mut max_value = a[pivot][pivot].abs();

        for row in pivot + 1..n {
            if a[row][pivot].abs() > max_value {
                max_value = a[row][pivot].abs();
                max_row = row;
            }
        }

        if max_value < 1.0e-14 {
            panic!("Singular or nearly singular matrix in Rosenbrock solve.");
        }

        if max_row != pivot {
            a.swap(pivot, max_row);
            b.swap(pivot, max_row);
        }

        for row in pivot + 1..n {
            let factor = a[row][pivot] / a[pivot][pivot];

            for col in pivot..n {
                a[row][col] -= factor * a[pivot][col];
            }

            b[row] -= factor * b[pivot];
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

    x
}

fn rosenbrock_euler_step<F, J>(
    f: &F,
    jacobian: &J,
    t: f64,
    y: &State,
    h: f64,
    gamma: f64,
) -> State
where
    F: Fn(f64, &State) -> State,
    J: Fn(f64, &State) -> Matrix,
{
    let j_n = jacobian(t, y);
    let w = form_rosenbrock_matrix(&j_n, h, gamma);

    let f_n = f(t, y);
    let rhs: State = f_n.iter().map(|fi| h * fi).collect();

    let k1 = solve_linear_system(w, rhs);

    y.iter()
        .zip(k1.iter())
        .map(|(yi, ki)| yi + ki)
        .collect()
}

fn solve_rosenbrock<F, J>(
    f: &F,
    jacobian: &J,
    t0: f64,
    y0: State,
    h: f64,
    n_steps: usize,
    gamma: f64,
) -> SolverResult
where
    F: Fn(f64, &State) -> State,
    J: Fn(f64, &State) -> Matrix,
{
    let mut trajectory = Vec::with_capacity(n_steps + 1);

    let mut t = t0;
    let mut y = y0;
    let mut linear_solves = 0;

    trajectory.push((t, y.clone()));

    for _ in 0..n_steps {
        y = rosenbrock_euler_step(f, jacobian, t, &y, h, gamma);
        linear_solves += 1;

        t += h;
        trajectory.push((t, y.clone()));
    }

    SolverResult {
        trajectory,
        linear_solves,
    }
}

fn exact_solution(t: f64) -> f64 {
    1.0 - (-1000.0 * t).exp()
}

fn main() {
    println!("Rosenbrock-Euler Method for a Stiff Autonomous Problem");
    println!("=====================================================\n");

    let t0 = 0.0;
    let t_end = 1.0;
    let h = 0.05;
    let n_steps = ((t_end - t0) / h) as usize;

    let y0 = vec![0.0];

    // gamma = 1 gives the simplest linearly implicit Rosenbrock-Euler scheme.
    let gamma = 1.0;

    let result = solve_rosenbrock(
        &stiff_rhs,
        &stiff_jacobian,
        t0,
        y0,
        h,
        n_steps,
        gamma,
    );

    let (t_final, y_final) = result
        .trajectory
        .last()
        .expect("trajectory should contain at least one point");

    let y_exact = exact_solution(*t_final);

    println!("Model Problem");
    println!("-------------");
    println!("Stiff autonomous scalar equation:");
    println!("  y'(t) = -1000(y(t) - 1)");
    println!("Exact solution:");
    println!("  y(t) = 1 - exp(-1000t), provided y(0) = 0\n");

    println!("Numerical Parameters");
    println!("--------------------");
    println!("Initial time t0          = {:.6}", t0);
    println!("Final time               = {:.6}", t_end);
    println!("Step size h              = {:.6}", h);
    println!("Number of steps          = {}", n_steps);
    println!("Rosenbrock gamma         = {:.6}\n", gamma);

    println!("Solver Statistics");
    println!("-----------------");
    println!("Linear solves performed  = {}", result.linear_solves);
    println!("Nonlinear Newton solves  = 0\n");

    println!("Final-Time Accuracy");
    println!("-------------------");
    println!("Computed y(t)            = {:.12}", y_final[0]);
    println!("Exact y(t)               = {:.12}", y_exact);
    println!(
        "Absolute error           = {:.6e}\n",
        (y_final[0] - y_exact).abs()
    );

    println!("Selected Trajectory Points");
    println!("--------------------------");
    println!("{:>10} {:>18} {:>18} {:>18}", "t", "y_n", "exact", "abs error");

    let stride = (result.trajectory.len() / 10).max(1);

    for (index, (t, y)) in result.trajectory.iter().enumerate() {
        if index % stride == 0 || index + 1 == result.trajectory.len() {
            let exact = exact_solution(*t);

            println!(
                "{:>10.6} {:>18.10} {:>18.10} {:>18.6e}",
                t,
                y[0],
                exact,
                (y[0] - exact).abs()
            );
        }
    }
}
```

Program 17.6.2 demonstrates the practical implementation of a Rosenbrock method as a linearly implicit integrator for stiff systems. By replacing nonlinear solves with linear system solves involving a fixed Jacobian matrix, the method achieves a balance between stability and computational efficiency. This reflects the central idea of Section 17.6.4, where Rosenbrock methods are positioned between fully implicit and explicit schemes.

The numerical results illustrate how the method handles stiffness effectively, rapidly damping transient behavior and approaching the steady-state solution without instability. The absence of nonlinear iterations significantly reduces computational overhead, making the method attractive for problems where Jacobian evaluation and linear solves are manageable.

The modular structure of the implementation allows for straightforward extension to higher-order Rosenbrock schemes, which involve multiple stages and more sophisticated coefficient structures. It also provides a foundation for incorporating advanced linear algebra techniques, such as matrix factorization reuse and preconditioning, which are essential for large-scale applications.

Overall, this program highlights the practical advantages of linearly implicit integration and demonstrates how theoretical concepts from Section 17.6.4 translate into efficient numerical algorithms for stiff differential equations.

## 17.6.5. Modern Developments in Stiff Integration

The numerical treatment of stiff systems continues to evolve, driven by the need to balance stability, accuracy, and computational efficiency across increasingly complex applications. Modern developments focus not only on improving integration formulas, but also on enhancing error control, adaptivity, and solver architecture.

One notable example is the Rosenbrock–Wanner family of methods. The scheme Rodas5P achieves fifth-order accuracy while incorporating embedded error estimation, A-stability, and L-stability, together with high-quality dense output. These properties make it particularly effective for stiff systems where both stability and interpolation accuracy are required. By combining linearly implicit structure with advanced error control, such methods provide a robust and efficient alternative to fully implicit schemes (Steinebach, 2023).

Recent studies have also emphasized the importance of stepsize control in stiff integration. While much attention is traditionally given to the underlying numerical method, the performance of the solver can be significantly influenced by the design of the controller. Optimized stepsize strategies in applications such as atmospheric chemistry have demonstrated substantial gains, reducing the number of function evaluations by up to 43% and improving overall runtime with minimal modifications to the solver implementation. This highlights the fact that adaptivity is not merely a secondary feature, but a central component of solver performance (Dreger et al., 2025).

For problems requiring very high accuracy, fully implicit Runge–Kutta methods remain indispensable. In particular, Radau IIA methods provide strong stability properties and high-order accuracy. Modern implementations of these methods incorporate full adaptivity, allowing both the stepsize and the order of the method to vary dynamically. This flexibility enables the solver to adjust to local problem characteristics, achieving state-of-the-art performance in regimes with tight error tolerances (Ekanathan, Smith and Rackauckas, 2024).

At a broader level, contemporary solver frameworks reflect the diversity of stiff problem structures encountered in practice. For example, the SUNDIALS ecosystem integrates a wide range of methods, including explicit, implicit, implicit–explicit (IMEX), and multirate schemes, as well as adaptive multistep methods such as backward differentiation formulas. This variety is essential because real-world systems may exhibit global stiffness, localized stiffness, or decomposable dynamics that benefit from specialized treatment of different components.

These developments underscore a central principle: there is no single optimal method for all stiff problems. Instead, effective numerical integration requires a flexible toolkit capable of adapting to the specific characteristics of the system. Modern research continues to refine both the mathematical foundations and the computational strategies of stiff solvers, ensuring that they remain capable of addressing the growing complexity of applications in science and engineering (Reynolds et al., 2023; Balos et al., 2025; Günther and Sandu, 2025).

## 17.6.6. Computational Complexity of Stiff Solvers

The computational cost of stiff integration methods is fundamentally different from that of nonstiff solvers. While explicit methods are dominated by function evaluations, stiff solvers are governed primarily by the cost of linear algebra operations, particularly the formation and solution of systems involving Jacobian-related matrices.

For a dense system of dimension $d$, the dominant operation is the factorization of a matrix of the form $W = I - \gamma h J$, where $J$ is the Jacobian. A standard LU factorization requires:

$$O(d^3) \quad \text{work}, \qquad O(d^2) \quad \text{storage} \tag{17.6.14}$$

reflecting the cubic growth in computational effort with system size. Once the factorization is available, solving a linear system with the same matrix involves forward and backward substitution, each requiring $O(d^2)$ operations.

In the context of a typical $s$-stage Rosenbrock method, this structure leads to a cost model in which one matrix factorization is performed per time step, followed by multiple linear solves. The total cost per step is therefore dominated by the factorization, with an additional contribution of order $O(s d^2)$ arising from the stage computations. Because the same matrix $W$ is reused across all stages, the method avoids repeated factorizations, significantly improving efficiency compared to fully implicit methods that require separate solves for each stage.

For large-scale problems, particularly those arising from spatial discretizations of partial differential equations, the Jacobian matrix is often sparse. In such cases, the computational complexity depends not only on the dimension $d$, but also on the sparsity pattern, the amount of fill-in generated during factorization, and the effectiveness of any preconditioning strategies employed. Sparse direct methods can reduce the cost relative to dense factorization, but their efficiency is highly problem-dependent.

In many practical applications, iterative solvers combined with appropriate preconditioners are used to handle large sparse systems. These approaches aim to approximate the solution of the linear systems at reduced cost, trading exact factorization for iterative convergence. The performance of such methods depends critically on the quality of the preconditioner, which must capture the essential structure of the Jacobian while remaining inexpensive to apply.

These considerations highlight that, for stiff solvers, linear algebra is the primary computational bottleneck. Efficient implementation therefore requires careful attention to matrix structure, reuse of factorizations, and the selection of appropriate solution strategies. Exploiting sparsity and tailoring linear algebra techniques to the problem at hand are essential for achieving scalability in large-scale simulations.

## 17.6.7. Applications and Practical Guidance for Stiff Systems

A representative and practically important setting for stiff integration arises in operator-split reactive flow and atmospheric chemistry. In such problems, a governing partial differential equation is decomposed into transport and reaction components. After spatial discretization, the transport terms are handled separately, while the reaction terms give rise to a large number of independent stiff ordinary differential equation systems that must be solved repeatedly at each spatial point.

This structure places stringent demands on numerical solvers. Because the reaction dynamics often involve widely separated time scales, explicit methods become impractical due to severe stability restrictions. Instead, stable implicit methods such as Rosenbrock schemes, backward differentiation formulas, and Radau-type implicit Runge–Kutta methods are essential. These methods allow large step sizes while maintaining stability, making them suitable for the repeated solution of stiff subsystems within a larger simulation framework.

In this context, improvements in stepsize control can have a substantial impact on overall performance. Since the stiff subsystems are solved many times across the spatial domain, even modest reductions in the number of function evaluations or linear solves per step can lead to significant savings in computational cost. Recent studies have demonstrated that optimized stepsize controllers can reduce the number of function evaluations and improve runtime efficiency without requiring major changes to the underlying solver (Balos et al., 2025; Dreger et al., 2025).

From a conceptual standpoint, an important lesson emerges. Stiffness should not be viewed as an intrinsic property of a differential equation alone, but rather as a relationship between the equation and the numerical method used to solve it. A system appears stiff when the stability constraints of the chosen method force the use of step sizes that are much smaller than those required for accuracy. In this sense, stiffness reflects a mismatch between the time-scale distribution of the system and the stability properties of the numerical scheme.

Understanding stiffness in this way provides a unifying perspective on the wide range of methods developed for stiff integration. Implicit Euler offers simplicity and strong damping properties. Rosenbrock methods provide a balance between stability and computational efficiency through linearization. Backward differentiation formulas exploit multistep structure for efficiency in large systems. Radau methods deliver high-order accuracy with strong stability guarantees. IMEX and multirate methods address problems with mixed or localized stiffness by treating different components with different strategies.

By recognizing stiffness as a methodological issue rather than a fixed classification, one can more effectively select and design numerical methods that align with the characteristics of the problem. This perspective helps organize the diverse landscape of stiff solvers and clarifies their respective roles in modern scientific computing.

# 17.7. Multistep, Multivalue, and Predictor-Corrector Methods

Multistep methods represent a distinct class of numerical integrators in which information from several previously computed solution values is reused to advance the solution. This reuse can significantly reduce the number of expensive function evaluations, particularly in large-scale simulations where the right-hand side is costly to compute and exhibits smooth temporal behavior. Unlike one-step methods, which rely solely on the current state, multistep methods build higher-order accuracy through historical data, making them especially effective in long-time integrations.

## 17.7.1. Conceptual Framework of Linear Multistep Methods

Consider the initial value problem:

$$y'(t) = f(t, y(t)), \qquad y(t_0) = y_0, \qquad y(t) \in \mathbb{R}^d \tag{17.7.1}$$

In a one-step method, such as a Runge–Kutta scheme, the approximation $y_{n+1}$ depends only on the current value $y_n$. In contrast, a multistep method incorporates multiple past values, effectively constructing the new approximation using a discrete representation of the solution history. This approach is particularly advantageous when the function $f(t,y)$ is expensive to evaluate, smooth in time, and repeatedly invoked within a larger simulation framework. In applications such as operator-split reacting flows and cosmological simulations, each spatial grid point may correspond to an independent ODE system, making the efficiency of the inner time integrator a dominant factor in overall performance (Balos et al., 2025).

The general form of an $s$-step linear multistep method is given by:

$$\sum_{j=0}^{s} \alpha_j y_{n+1-j}= h_n \sum_{j=0}^{s} \beta_j f_{n+1-j}, \qquad f_k = f(t_k, y_k) \tag{17.7.2}$$

where $y_n \approx y(t_n)$ and $h_n = t_{n+1} - t_n$. The coefficients $(\alpha_j, \beta_j)$ determine the specific method and its properties.

Different choices of these coefficients lead to well-known families of methods. Adams–Bashforth methods are explicit and use only previously computed function values. Adams–Moulton methods are implicit and include the function evaluation at the new time level. Backward differentiation formulas are also implicit and are particularly well suited for stiff systems due to their strong stability properties.

In modern implementations, these methods are employed in variable-step, variable-order form. Libraries such as CVODE and GNU GSL dynamically adjust both the step size and the order of the method to match the local behavior of the solution. For nonstiff problems, Adams–Moulton methods are typically used with orders up to 12, while for stiff systems, BDF methods are employed with orders up to 5, reflecting stability limitations at higher orders.

A useful way to visualize the operation of a multistep method is through the flow of data:

$$
\underbrace{\{y_n, y_{n-1}, \dots, y_{n-s+1}\}}_{\text{state history}}
\quad \text{and} \quad
\underbrace{\{f_n, f_{n-1}, \dots, f_{n-s+1}\}}_{\text{derivative history}}
\;\longrightarrow\;
y_{n+1}
\tag{17.7.3}
$$

This representation emphasizes that the next solution value is computed from a combination of past states and derivatives. Because the method depends on previously computed values, it is not self-starting. The initial history must therefore be generated using a one-step method, typically a Runge–Kutta scheme of comparable order. This startup phase ensures that sufficient data is available to begin the multistep recurrence.

The conceptual framework outlined here highlights both the strengths and the limitations of multistep methods. By leveraging historical information, they achieve high efficiency for smooth problems, but at the cost of increased complexity in initialization, adaptivity, and stability management.

## 17.7.2. Multivalue Methods and the Nordsieck Representation

An alternative and often more flexible viewpoint on time integration is provided by multivalue methods. Instead of storing multiple past solution values, as in classical multistep schemes, these methods maintain several pieces of information associated with the current time level. This shift in perspective leads to representations that are particularly well suited for adaptive stepsize and order control.

A standard formulation is the Nordsieck representation, in which the local solution is described by a scaled derivative vector, or jet,

$$
z_n =
\begin{bmatrix}
y_n \\
[2mm] h_n y_n' \\
[2mm] \dfrac{h_n^2}{2!} y_n'' \\
[2mm] \vdots \\
[2mm] \dfrac{h_n^q}{q!} y_n^{(q)}
\end{bmatrix}
\tag{17.7.4}
$$

Each component of this vector corresponds to a derivative of the solution at the current time, scaled by the appropriate power of the stepsize. This scaling ensures that all components have comparable magnitudes, which improves numerical stability and facilitates transformations when the stepsize changes.

The Nordsieck form offers several practical advantages. Because all relevant information is stored at a single time level, modifications to the stepsize can be handled by simple rescaling operations on the vector $z_n$, rather than reconstructing a history of past solution values. Similarly, changes in the order of the method can be implemented by adding or removing components from the vector, again through local transformations. Dense output, which provides continuous approximations between time steps, is also naturally supported, since the stored derivatives directly define a local polynomial approximation of the solution.

This representation is widely used in modern solver libraries, such as SUNDIALS, where efficient handling of adaptivity and interpolation is essential. By organizing the data in this way, the implementation becomes more modular and better aligned with the needs of variable-step, variable-order integration.

A broader unifying framework is provided by the general linear method formulation,

$$Y = h A F + U y^{[n-1]}, \qquad y^{[n]} = h B F + V y^{[n-1]} \tag{17.7.5}$$

In this formulation, $Y$ collects stage values, $F$ collects the corresponding stage derivatives, and $y^{[n]}$ represents the output quantities at the new time level. The matrices $A$, $B$, $U$, and $V$ define the method and determine how current and previous information is combined.

This framework encompasses both one-step and multistep approaches. Runge–Kutta methods arise when the stored information consists of a single solution value, while multivalue methods correspond to retaining multiple components, such as the entries of the Nordsieck vector. Multistep methods can also be interpreted within this setting, showing that these seemingly distinct approaches are part of a broader unified class of numerical integrators.

This unified perspective clarifies the relationships among different methods and provides a flexible foundation for the design of modern solvers. By choosing appropriate structures for the stored data and transformation matrices, one can tailor the method to balance accuracy, stability, and computational efficiency in a wide range of applications (Conte, Pagano and Paternoster, 2023).

## 17.7.3 Derivation of Adams Methods via Polynomial Interpolation

The Adams family of multistep methods is constructed by approximating the right-hand side ( f(t,y(t)) ) with an interpolating polynomial and then integrating this approximation over one time step. This approach provides a direct connection between interpolation theory and numerical integration, and explains both the efficiency and the structure of the resulting formulas.

To simplify the derivation, assume a constant stepsize $h$ and introduce the normalized variable:

$$\theta = \frac{t - t_n}{h} \tag{17.7.6}$$

so that the interval $[t_n, t_{n+1}]$ corresponds to $\theta \in [0,1]$, and previous time levels correspond to $\theta = -1, -2, \dots$.

The key idea is to approximate the function $f(t,y(t))$ by an interpolating polynomial constructed from previously computed values. For a third-order explicit method, one uses the values $f_n, f_{n-1}, f_{n-2}$, corresponding to $\theta = 0, -1, -2$. The quadratic interpolating polynomial $p_2(\theta)$ is given by:

$$p_2(\theta) = f_n \frac{(\theta+1)(\theta+2)}{2} - f_{n-1}\,\theta(\theta+2) + f_{n-2}\frac{\theta(\theta+1)}{2} \tag{17.7.7}$$

This polynomial reproduces the known values of $f$ at the selected nodes and provides an approximation to $f(t,y(t))$ over the entire interval. The next step is to integrate this interpolant over $\theta \in [0,1]$ to approximate the increment in the solution. Using the relation:

$$y_{n+1} = y_n + h \int_0^1 f(t_n + \theta h, y(t_n + \theta h))\, d\theta,$$

and replacing $f$ by $p_2(\theta)$, one obtains:

$$y_{n+1} = y_n + h \int_0^1 p_2(\theta) \, d\theta + \mathcal{O}(h^4) \tag{17.7.8}$$

Carrying out the integration of the polynomial term by term yields the third-order Adams–Bashforth formula,

$$y_{n+1} = y_n + \frac{h}{12}\left(23 f_n - 16 f_{n-1} + 5 f_{n-2}\right) + \mathcal{O}(h^4) \tag{17.7.9}$$

This method is explicit because it depends only on previously computed values of $f$. Its efficiency arises from the fact that it reuses derivative information from earlier steps, requiring only one new function evaluation per step after initialization.

A corresponding implicit method is obtained by including the value at the new time level in the interpolation. For the Adams–Moulton corrector, one interpolates $f$ at $\theta = 1, 0, -1$, leading to an approximation that incorporates $f_{n+1}$. Following the same integration procedure yields:

$$y_{n+1} = y_n + \frac{h}{12}\left(5 f_{n+1} + 8 f_n - f_{n-1}\right) + \mathcal{O}(h^4) \tag{17.7.10}$$

This formula is implicit because $f_{n+1} = f(t_{n+1}, y_{n+1})$ depends on the unknown solution at the new time level. In practice, this equation is solved iteratively, often using predictor–corrector techniques as described in Section 17.7.4.

The derivation highlights a fundamental principle: Adams methods arise from approximating the integral of $f$ using interpolating polynomials constructed from known data. Explicit Adams–Bashforth methods use only past values and are computationally efficient, while implicit Adams–Moulton methods include the future value to improve accuracy and stability. Together, they form a complementary pair that underpins many practical multistep integration strategies.

## 17.7.4. Predictor–Corrector Schemes for Multistep Integration

Predictor–corrector methods provide a practical and efficient way to combine the low cost of explicit multistep formulas with the improved accuracy and stability of implicit schemes. The central idea is to first generate a provisional approximation using an explicit method (the predictor), and then refine this approximation using an implicit formula (the corrector). This two-stage process allows one to approximate the solution at the new time level without solving a fully nonlinear system.

A representative example is obtained by combining an Adams–Bashforth predictor with an Adams–Moulton corrector. The predictor step is given by:

$$
\widehat{y}_{n+1} = y_n + \frac{h}{12}\left(23 f_n - 16 f_{n-1} + 5 f_{n-2}\right)
\tag{17.7.11}
$$

which uses previously computed derivative values to estimate the solution at $t_{n+1}$. The function is then evaluated at this predicted point,

$$
\widehat{f}_{n+1} = f(t_{n+1}, \widehat{y}_{n+1})
\tag{17.7.12}
$$

providing the information required for the corrector step. The corrected solution is computed using an implicit Adams–Moulton formula,

$$
y_{n+1} = y_n + \frac{h}{12}\left(5 \widehat{f}_{n+1} + 8 f_n - f_{n-1}\right)
\tag{17.7.13}
$$

This sequence can be interpreted as a single iteration toward solving the implicit equation associated with the corrector. In practice, additional correction sweeps may be applied to improve accuracy, although even a single correction often yields a significant improvement over the predictor alone.

The computational workflow is commonly summarized as:

$$\text{P} \;\rightarrow\; \text{E} \;\rightarrow\; \text{C} \;\rightarrow\; \text{E} \tag{17.7.14}$$

that is, predict the solution, evaluate the function at the predicted point, correct the solution, and optionally re-evaluate the function at the corrected value.

This approach offers a favorable balance between computational cost and accuracy. The predictor step avoids the need for solving nonlinear equations, while the corrector step incorporates implicit information that improves stability and accuracy. Because the corrector typically requires only a small number of iterations, the overall cost remains relatively low compared to fully implicit methods.

Predictor–corrector schemes are particularly effective when the solution is smooth and the derivative evaluations are moderately expensive. They also provide a flexible framework in which the number of correction iterations can be adjusted to meet accuracy requirements. As such, they represent an important class of methods bridging explicit and implicit multistep techniques.

### Rust Implementation

Following the development of predictor–corrector strategies in Section 17.7.4, Program 17.7.1 provides a practical implementation of the Adams–Bashforth–Moulton multistep method using a structured history of solution and derivative values. As discussed around Equations (17.7.11)–(17.7.13), this approach combines an explicit prediction step with an implicit correction step to achieve improved accuracy while avoiding the cost of fully nonlinear solves. In practical numerical computation, the efficiency of multistep methods depends critically on how past information is stored and reused. This program adopts a ring-buffer-based history structure, enabling constant-time updates and direct correspondence with the recurrence formulation (17.7.2). The implementation also incorporates a Runge–Kutta startup phase to generate the initial history required by the multistep scheme, ensuring consistency with the non-self-starting nature of these methods.

At the core of the implementation is the `AdamsBashforthMoulton` structure, which encapsulates the stepsize and a fixed-length history buffer storing recent solution values and their corresponding derivatives. Each entry in this buffer represents the quantities $y_n$ and $f_n = f(t_n, y_n)$, aligning directly with the data flow described in Equation (17.7.3). The buffer is implemented using a double-ended queue, allowing efficient insertion of new values and removal of outdated ones while maintaining the required ordering for the multistep recurrence.

The `initialize_with_rk4` function provides the startup phase required for multistep methods. Since the predictor–corrector formulation depends on multiple past values, the first few solution points are generated using a classical fourth-order Runge–Kutta method. This ensures that the history buffer is populated with sufficiently accurate initial values, allowing the multistep method to proceed without loss of order. This design reflects the theoretical requirement that multistep methods are not self-starting and must be initialized using a one-step scheme of comparable accuracy.

The `step` function implements the predictor–corrector mechanism described in Equations (17.7.11)–(17.7.13). The predictor stage computes an intermediate approximation $\widehat{y}_{n+1}$ using the explicit Adams–Bashforth formula, which reuses derivative information from previous steps. The function is then evaluated at this predicted point to obtain $\widehat{f}_{n+1}$, which is used in the corrector stage. The corrector applies the implicit Adams–Moulton formula, yielding a refined approximation of $y_{n+1}$. This sequence corresponds to a single correction iteration and provides a balance between computational efficiency and improved accuracy.

The auxiliary function `rk4_step` implements a standard fourth-order Runge–Kutta method used exclusively during initialization. Although independent of the multistep formulation, it plays a critical role in ensuring that the initial values supplied to the predictor–corrector scheme are consistent with the desired order of accuracy. The separation of this functionality into a dedicated function enhances modularity and clarity of the implementation.

The `main` function demonstrates the method on a representative linear test problem with known exact solution. It defines the model right-hand side, initializes the solver, and advances the solution over a fixed time interval. At each step, the numerical solution is compared with the exact solution, and the absolute error is reported. This output illustrates the convergence behavior of the method and confirms that the predictor–corrector scheme maintains accuracy consistent with its theoretical order.

```rust
// Program 17.7.1: Adams-Bashforth-Moulton Predictor-Corrector Method
//
// Problem statement:
// Solve the initial value problem
//
//     y'(t) = f(t, y),     y(t0) = y0,
//
// using a third-order Adams-Bashforth predictor and a third-order
// Adams-Moulton corrector. The method uses derivative history
// stored in a small ring buffer, matching the multistep structure
// described in Section 17.7.4.

use std::collections::VecDeque;

type OdeFunction = fn(f64, f64) -> f64;

#[derive(Clone, Debug)]
struct HistoryEntry {
    t: f64,
    y: f64,
    f: f64,
}

#[derive(Clone, Debug)]
struct AdamsBashforthMoulton {
    h: f64,
    history: VecDeque<HistoryEntry>,
}

impl AdamsBashforthMoulton {
    fn new(h: f64) -> Self {
        Self {
            h,
            history: VecDeque::with_capacity(3),
        }
    }

    fn push_history(&mut self, entry: HistoryEntry) {
        if self.history.len() == 3 {
            self.history.pop_back();
        }
        self.history.push_front(entry);
    }

    fn initialize_with_rk4(&mut self, f: OdeFunction, t0: f64, y0: f64) {
        let mut t = t0;
        let mut y = y0;

        self.push_history(HistoryEntry { t, y, f: f(t, y) });

        for _ in 0..2 {
            y = rk4_step(f, t, y, self.h);
            t += self.h;
            self.push_history(HistoryEntry { t, y, f: f(t, y) });
        }
    }

    fn step(&mut self, f: OdeFunction) -> HistoryEntry {
        assert!(
            self.history.len() == 3,
            "The third-order predictor requires three history values."
        );

        let h = self.h;

        let current = &self.history[0];
        let previous = &self.history[1];
        let older = &self.history[2];

        let t_next = current.t + h;

        let y_predict = current.y
            + h / 12.0 * (23.0 * current.f - 16.0 * previous.f + 5.0 * older.f);

        let f_predict = f(t_next, y_predict);

        let y_correct = current.y
            + h / 12.0 * (5.0 * f_predict + 8.0 * current.f - previous.f);

        let f_correct = f(t_next, y_correct);

        let next = HistoryEntry {
            t: t_next,
            y: y_correct,
            f: f_correct,
        };

        self.push_history(next.clone());
        next
    }
}

fn rk4_step(f: OdeFunction, t: f64, y: f64, h: f64) -> f64 {
    let k1 = f(t, y);
    let k2 = f(t + 0.5 * h, y + 0.5 * h * k1);
    let k3 = f(t + 0.5 * h, y + 0.5 * h * k2);
    let k4 = f(t + h, y + h * k3);

    y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
}

fn model_rhs(_t: f64, y: f64) -> f64 {
    -2.0 * y
}

fn exact_solution(t: f64) -> f64 {
    (-2.0 * t).exp()
}

fn main() {
    let t0 = 0.0;
    let y0 = 1.0;
    let t_final = 2.0;
    let h = 0.1;

    let mut solver = AdamsBashforthMoulton::new(h);
    solver.initialize_with_rk4(model_rhs, t0, y0);

    println!("Adams-Bashforth-Moulton Predictor-Corrector Method");
    println!("=================================================");
    println!();
    println!("Model problem:");
    println!("  y'(t) = -2y,    y(0) = 1");
    println!("Exact solution:");
    println!("  y(t) = exp(-2t)");
    println!();
    println!("Step size h = {:.6}", h);
    println!();
    println!(
        "{:>8} {:>18} {:>18} {:>18}",
        "t", "numerical y", "exact y", "abs error"
    );
    println!("{}", "-".repeat(68));

    let mut entries: Vec<HistoryEntry> = solver.history.iter().cloned().collect();
    entries.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());

    for entry in entries {
        let exact = exact_solution(entry.t);
        println!(
            "{:>8.4} {:>18.10} {:>18.10} {:>18.6e}",
            entry.t,
            entry.y,
            exact,
            (entry.y - exact).abs()
        );
    }

    let mut current_t = solver.history[0].t;

    while current_t + h <= t_final + 1.0e-12 {
        let entry = solver.step(model_rhs);
        current_t = entry.t;

        let exact = exact_solution(entry.t);
        println!(
            "{:>8.4} {:>18.10} {:>18.10} {:>18.6e}",
            entry.t,
            entry.y,
            exact,
            (entry.y - exact).abs()
        );
    }
}
```

Program 17.7.1 demonstrates the practical realization of predictor–corrector multistep methods through explicit management of solution history and staged evaluation of the governing equations. This implementation reflects the central idea discussed in Section 17.7.4: combining the efficiency of explicit multistep prediction with the improved accuracy of implicit correction.

The numerical results illustrate how the method maintains a controlled error profile while requiring only a single new function evaluation per step after initialization. This efficiency advantage is particularly significant in applications where the evaluation of $f(t,y)$ is computationally expensive. At the same time, the use of a correction step mitigates the accumulation of error typically associated with purely explicit schemes.

The structure of the code also highlights the importance of data organization in multistep methods. By representing the solution history explicitly and updating it through a ring buffer, the implementation closely mirrors the mathematical recurrence, ensuring both efficiency and transparency. This design can be extended naturally to higher-order methods, adaptive stepsize control, and systems of equations, providing a foundation for more advanced multistep solvers.

## 17.7.5. Stability, Consistency, and Root Conditions for Multistep Methods

The theoretical analysis of linear multistep methods is conveniently expressed in terms of two characteristic polynomials that encode the coefficients of the scheme. These are defined as:

$$
\rho(\zeta) = \sum_{j=0}^{s} \alpha_j \zeta^{s-j}, \qquad
\sigma(\zeta) = \sum_{j=0}^{s} \beta_j \zeta^{s-j}
\tag{17.7.15}
$$

These polynomials provide a compact representation of the recurrence (17.7.2) and form the basis for analyzing consistency, accuracy, and stability. In particular, they allow the behavior of the numerical method to be studied independently of any specific problem, by focusing on its algebraic structure.

Consistency is the first requirement for a valid numerical method. It ensures that the discrete scheme approximates the differential equation correctly as the stepsize tends to zero. In terms of the polynomials $\rho$ and $\sigma$, consistency requires:

$$\rho(1) = 0, \qquad \rho'(1) = \sigma(1) \tag{17.7.16}$$

The condition $\rho(1)=0$ guarantees that constant solutions are reproduced exactly, while the second condition ensures that the first derivative is correctly approximated. Together, these conditions ensure that the method aligns with the underlying differential equation in the limit of small step size.

The order of the method is determined by how well the numerical solution matches the Taylor expansion of the exact solution. This requirement can be expressed in terms of the generating functions as:

$$\rho(e^\xi)-\xi \sigma(e^\xi)=\mathcal{O}(\xi^{p+1}), \quad \xi \to 0 \tag{17.7.17}$$

This condition ensures that the truncation error is of order $p$, meaning that the difference between the numerical and exact solutions decreases at a rate proportional to $h^p$. As in Runge–Kutta methods, higher-order accuracy requires increasingly restrictive conditions on the coefficients.

In addition to consistency and accuracy, multistep methods must satisfy a stability condition known as zero-stability. This condition addresses the behavior of the homogeneous recurrence associated with the method and ensures that errors do not grow uncontrollably as the computation proceeds.

Zero-stability is characterized by the root condition: all roots $\zeta$ of the polynomial $\rho(\zeta)$ must satisfy:

$$|\zeta| \le 1, \quad \text{with simple roots on } |\zeta| = 1 \tag{17.7.18}$$

This requirement ensures that any parasitic modes introduced by the recurrence remain bounded. If a root lies outside the unit circle, errors associated with that mode will grow exponentially, rendering the method unstable. Similarly, if a root on the unit circle has multiplicity greater than one, small perturbations can lead to unbounded growth.

This root condition is a distinguishing feature of multistep methods. Unlike one-step methods such as Runge–Kutta schemes, which are self-starting and rely primarily on local stability properties, multistep methods involve a recurrence relation that must be stable in its own right. The root condition therefore provides an essential criterion for the viability of the method.

Taken together, these conditions establish the theoretical foundation for multistep integration. Consistency ensures correct approximation of the differential equation, order conditions determine accuracy, and the root condition guarantees stability of the recurrence. Only when all three are satisfied does the method provide a reliable and convergent numerical solution.

## 17.7.6. Complexity and Adaptive Control in Multistep Methods

The computational efficiency of multistep methods is one of their principal advantages, particularly after the initial startup phase has been completed. Once a sufficient history of past values has been generated, explicit multistep methods typically require only a single new evaluation of the function $f(t,y)$ per time step. This makes them significantly more efficient than one-step methods of comparable order, which often require multiple function evaluations per step. The associated memory requirement scales as $O(sd)$, where $s$ is the number of steps and $d$ is the dimension of the system, reflecting the need to store both state and derivative histories.

When implicit correctors are employed, additional computational cost is introduced. In predictor–corrector or fully implicit multistep methods, each step may require solving nonlinear equations, typically through Newton-type iterations. This leads to linear systems involving matrices of the form:

$$M \approx I - \gamma J, \qquad J = \frac{\partial f}{\partial y}, \qquad \gamma = h_n \beta_{n,0} \tag{17.7.19}$$

As in stiff solvers discussed earlier, the cost of forming and solving these systems can dominate the overall computation, especially for large-scale problems. Efficient handling of the Jacobian and associated linear algebra is therefore essential for maintaining performance.

Adaptive stepsize control plays a central role in modern multistep implementations. The goal is to adjust the stepsize dynamically so that the local truncation error remains within prescribed tolerances while minimizing computational effort. A common approach is to scale the estimated error componentwise using quantities of the form:

$$D_i = \varepsilon_{\text{abs}} + \varepsilon_{\text{rel}}\bigl(a_y |y_i| + a_{dydt} h |y_i'|\bigr) \tag{17.7.20}$$

which combine absolute and relative tolerances with measures of the solution magnitude and its rate of change. This scaling ensures that the error control adapts appropriately to different components of the solution, even when their magnitudes vary significantly.

Based on this scaling, the stepsize is updated according to a formula such as,

$$h_{\text{new}} = h_{\text{old}} \, S \left(\frac{E}{D}\right)^{-1/q} \tag{17.7.21}$$

where $E$ represents the estimated error, $D$ is the scaling factor, $q$ is the order of the method, and $S < 1$ is a safety factor introduced to maintain stability in the adaptation process. This formula reflects the asymptotic relationship between the error and the stepsize, ensuring that the new step is chosen to meet the desired tolerance.

In modern solver implementations, such as CVODE, error control is often performed using weighted root-mean-square norms. The weights are defined as:

$$W_i = \frac{1}{\text{rtol}|y_i| + \text{atol}_i} \tag{17.7.22}$$

so that the normalized error measure reflects both absolute and relative contributions across all components. This approach provides a robust and scalable mechanism for error control in high-dimensional systems.

A key design principle in multistep methods is that adaptivity is tightly coupled with the management of solution history. Changes in stepsize or method order require consistent updating of stored values, which can be complex in traditional formulations. This is one of the reasons why Nordsieck representations are widely used in practice. By storing a local polynomial representation of the solution, they allow stepsize and order changes to be handled through simple transformations, avoiding the need to reconstruct past states explicitly.

These considerations illustrate that the efficiency of multistep methods arises not only from their low per-step cost, but also from careful integration of adaptivity, error control, and data management.

## 17.7.7. Modern Developments and Applications of Multistep Methods

Multistep methods continue to play a central role in modern numerical analysis, not only in their classical forms but also through a variety of extensions that address emerging computational challenges. Their defining feature, the reuse of solution history, remains a powerful principle, and recent developments have expanded this idea to improve stability, parallel efficiency, and preservation of physical constraints.

One important direction is the development of explicit peer methods, which extend multistep concepts by evolving multiple solution values at the same time level. These methods retain the efficiency benefits of multistep schemes while improving stability properties and enabling parallel computation across stages. By distributing work across multiple “peers,” they provide a natural framework for modern high-performance architectures.

Another significant development is spectral deferred correction (SDC) methods. These methods begin with a low-order approximation and iteratively refine it through correction sweeps, effectively building a high-order solution from repeated applications of simpler schemes. The resulting process combines the advantages of multistep history usage with iterative improvement, allowing high-order accuracy to be achieved in a flexible and adaptive manner.

Structure-preserving multistep schemes have also received considerable attention. Methods such as modified Patankar schemes are designed to enforce positivity and conservation laws, which are critical in applications involving physical quantities like mass, energy, or probability. By embedding these constraints directly into the numerical method, such schemes ensure that the computed solution remains physically meaningful throughout the integration.

These developments are closely tied to important application domains. In operator-split partial differential equation simulations, large numbers of local ODE systems must be solved repeatedly and efficiently. Multistep methods are well suited to this setting because they minimize the number of expensive function evaluations while maintaining high accuracy. Similarly, in nonequilibrium quantum dynamics, where the cost of evaluating the governing equations can be extremely high, adaptive multistep methods can significantly reduce computational effort by exploiting smooth temporal behavior.

The underlying principle connecting these developments is the effective use of solution history. When the evolution of a system depends smoothly on its past behavior, it is natural for the numerical method to incorporate that history into its construction. Multistep methods achieve this in a controlled and adaptive manner, balancing efficiency with accuracy and stability.

This perspective explains both the enduring relevance of classical multistep schemes and their continued evolution in modern computational science.

## 17.7.8. Implementation Perspective for Multistep and Multivalue Methods

In practical implementations, especially in a systems-level language such as Rust, the defining feature of multistep methods, the use of history, should be reflected explicitly in the software design. Rather than treating past values as incidental data, they should be managed as structured objects with clear semantics and controlled access. This approach preserves the mathematical intent of the method while enabling efficient and maintainable code.

For Adams-type multistep methods, a natural representation is a ring buffer storing recent solution values ${y_n, y_{n-1}, \dots}$ and their corresponding derivatives. A ring buffer provides constant-time insertion and removal, avoids unnecessary data movement, and maintains a fixed memory footprint. This structure aligns directly with the recurrence (17.7.2), where the method advances by combining a fixed number of past values. By organizing the data in this way, the implementation mirrors the mathematical formulation and supports efficient evaluation of predictor and corrector formulas.

For variable-order and adaptive methods, the Nordsieck representation offers a more flexible alternative. By storing a scaled derivative vector as in (17.7.4), the solver can update the solution, change the stepsize, and adjust the order using local transformations of a single vector. This avoids the need to reconstruct or reinterpret a sequence of past values when the integration parameters change. In a Rust implementation, this can be realized through a structured vector type with well-defined operations for scaling, shifting, and truncating components, ensuring both clarity and performance.

Predictor–corrector methods introduce additional complexity, as they combine explicit and implicit components and may involve multiple passes over the same step. A robust implementation should therefore expose key control mechanisms explicitly. These include step rejection, which discards a trial step when the error exceeds tolerance; restart procedures, which rebuild the solution history after a rejected step or a significant change in stepsize; and order adaptation, which adjusts the number of stored components or steps to match the local smoothness of the solution. Making these operations explicit in the code helps maintain consistency between the numerical algorithm and its computational realization.

This design philosophy emphasizes a close alignment between mathematical structure and software architecture. By treating history as a first-class object, organizing data structures to reflect the underlying recurrences, and explicitly managing adaptivity and control flow, the implementation remains both efficient and transparent. In Rust, this approach also benefits from strong typing and ownership semantics, which help enforce correctness and prevent unintended data dependencies.

Overall, the effective implementation of multistep and multivalue methods depends not only on numerical formulas, but also on careful design of data structures and control logic. Aligning these elements with the mathematical formulation ensures that the solver remains both performant and maintainable in complex applications.

### Rust Implementation

Following the discussion in Sections 17.7.2 and 17.7.8 on multivalue formulations and implementation strategies, Program 17.7.2 provides a practical realization of the Nordsieck representation for multistep integration. As introduced in Equation (17.7.4), this formulation replaces explicit storage of past solution values with a scaled derivative vector that captures local solution behavior at a single time level. In numerical computation, such representations are essential for efficient handling of variable stepsizes and method orders, since they allow local transformations to replace costly reconstruction of solution history. This program demonstrates how the Nordsieck vector can be used to perform prediction, stepsize rescaling, and order truncation, illustrating the flexibility and numerical stability advantages emphasized in the preceding sections.

At the core of the implementation is the `NordsieckVector` structure, which stores the scaled derivative components defined in Equation (17.7.4). Each entry in the vector corresponds to a derivative of the solution at the current time, multiplied by the appropriate power of the stepsize and normalized by a factorial term. This representation ensures that all components are of comparable magnitude, which improves numerical stability and facilitates transformations when the stepsize is modified.

The `predict` function evaluates the local polynomial representation implied by the Nordsieck vector. Using the normalized variable introduced in Equation (17.7.6), the function computes the approximate solution at a future point by summing the contributions of each stored component scaled by powers of the parameter $\theta$. This operation corresponds to evaluating the truncated Taylor expansion encoded in the vector and provides a direct mechanism for generating dense output or advancing the solution.

The `rescale_stepsize` function implements the transformation required when the stepsize changes. As discussed in Section 17.7.2, modifying the stepsize does not require reconstructing past solution values; instead, the components of the Nordsieck vector are scaled by powers of the ratio between the new and old stepsizes. This operation preserves the consistency of the representation while adapting it to the new integration scale, enabling efficient stepsize control.

The `truncate_order` function adjusts the order of the method by removing higher-order components from the vector. This operation reflects the adaptive order control mechanisms discussed in Section 17.7.6, where the solver dynamically adjusts the number of retained derivatives based on the local smoothness of the solution. Truncating the vector reduces computational cost but also lowers accuracy, as fewer terms of the local polynomial are retained.

The helper functions `make_exp_nordsieck_vector` and `factorial` provide a concrete example of constructing a Nordsieck vector for a known analytical solution. In this case, the exponential function is used because all of its derivatives are equal, simplifying the construction of the scaled components and making it easy to verify correctness. The factorial function is used to compute the normalization factors in the definition of the vector.

The `main` function demonstrates the behavior of the Nordsieck representation through three stages. First, it constructs an initial vector and performs a prediction over one full step, illustrating the accuracy of the polynomial approximation. Next, it rescales the stepsize and repeats the prediction, showing how the representation adapts seamlessly to a new integration scale. Finally, it reduces the order of the method and evaluates the resulting prediction, highlighting the trade-off between computational cost and accuracy. These experiments collectively demonstrate the flexibility and effectiveness of the multivalue formulation.

```rust
// Program 17.7.2: Nordsieck Representation for Multivalue Step Management
//
// Problem statement:
// Demonstrate how a multivalue method stores local solution information
// in a Nordsieck vector
//
//     z_n = [y_n, h y'_n, h^2 y''_n / 2!, ..., h^q y_n^(q) / q!]^T,
//
// and how this representation supports prediction, stepsize rescaling,
// and order truncation in a transparent way.

#[derive(Clone, Debug)]
struct NordsieckVector {
    h: f64,
    components: Vec<f64>,
}

impl NordsieckVector {
    fn new(h: f64, components: Vec<f64>) -> Self {
        assert!(
            !components.is_empty(),
            "A Nordsieck vector must contain at least the solution component."
        );

        Self { h, components }
    }

    fn order(&self) -> usize {
        self.components.len() - 1
    }

    fn value(&self) -> f64 {
        self.components[0]
    }

    fn predict(&self, theta: f64) -> f64 {
        let mut value = 0.0;
        let mut power = 1.0;

        for component in &self.components {
            value += component * power;
            power *= theta;
        }

        value
    }

    fn rescale_stepsize(&mut self, new_h: f64) {
        let ratio = new_h / self.h;
        let mut factor = 1.0;

        for component in &mut self.components {
            *component *= factor;
            factor *= ratio;
        }

        self.h = new_h;
    }

    fn truncate_order(&mut self, new_order: usize) {
        assert!(
            new_order <= self.order(),
            "The requested order must not exceed the current order."
        );

        self.components.truncate(new_order + 1);
    }

    fn print(&self, label: &str) {
        println!("{}", label);
        println!("{}", "-".repeat(label.len()));
        println!("stepsize h = {:.6}", self.h);
        println!("order q    = {}", self.order());
        println!("stored y   = {:.10}", self.value());
    
        for (j, component) in self.components.iter().enumerate() {
            println!("z[{:>2}]     = {:>18.10}", j, component);
        }
    
        println!();
    }
}

fn exact_exp_solution(t: f64) -> f64 {
    t.exp()
}

fn make_exp_nordsieck_vector(t: f64, h: f64, order: usize) -> NordsieckVector {
    let y = exact_exp_solution(t);
    let mut components = Vec::with_capacity(order + 1);

    for j in 0..=order {
        let scaled_derivative = y * h.powi(j as i32) / factorial(j) as f64;
        components.push(scaled_derivative);
    }

    NordsieckVector::new(h, components)
}

fn factorial(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

fn main() {
    let t_n = 0.0;
    let h = 0.1;
    let order = 4;

    let mut z = make_exp_nordsieck_vector(t_n, h, order);

    println!("Nordsieck Representation for Multivalue Step Management");
    println!("=======================================================");
    println!();
    println!("Model function:");
    println!("  y(t) = exp(t)");
    println!();
    println!("Stored Nordsieck components:");
    println!("  z[j] = h^j y^(j)(t_n) / j!");
    println!();

    z.print("Initial Nordsieck Vector");

    let theta = 1.0;
    let predicted = z.predict(theta);
    let exact_next = exact_exp_solution(t_n + h);

    println!("Prediction over one full step");
    println!("-----------------------------");
    println!("theta                  = {:.6}", theta);
    println!("predicted y(t_n + h)   = {:.12}", predicted);
    println!("exact y(t_n + h)       = {:.12}", exact_next);
    println!("absolute error         = {:.6e}", (predicted - exact_next).abs());
    println!();

    let new_h = 0.05;
    z.rescale_stepsize(new_h);

    z.print("After Stepsize Rescaling");

    let predicted_half_step = z.predict(1.0);
    let exact_half_step = exact_exp_solution(t_n + new_h);

    println!("Prediction after rescaling");
    println!("--------------------------");
    println!("new stepsize h           = {:.6}", new_h);
    println!("predicted y(t_n + h)     = {:.12}", predicted_half_step);
    println!("exact y(t_n + h)         = {:.12}", exact_half_step);
    println!(
        "absolute error           = {:.6e}",
        (predicted_half_step - exact_half_step).abs()
    );
    println!();

    z.truncate_order(2);

    z.print("After Order Truncation");

    let predicted_low_order = z.predict(1.0);
    let exact_low_order = exact_exp_solution(t_n + new_h);

    println!("Prediction after order truncation");
    println!("---------------------------------");
    println!("retained order q          = {}", z.order());
    println!("predicted y(t_n + h)      = {:.12}", predicted_low_order);
    println!("exact y(t_n + h)          = {:.12}", exact_low_order);
    println!(
        "absolute error            = {:.6e}",
        (predicted_low_order - exact_low_order).abs()
    );
}
```

Program 17.7.2 demonstrates how the Nordsieck representation provides a powerful and flexible framework for multivalue time integration. By storing scaled derivatives at a single time level, the method avoids the complexity associated with managing a history of past solution values, while still retaining the information necessary to achieve high-order accuracy.

The numerical results illustrate several key properties. High-order representations yield highly accurate predictions, while stepsize rescaling can be performed efficiently through simple algebraic transformations. At the same time, reducing the order of the method leads to a noticeable increase in error, emphasizing the role of higher-order derivatives in capturing local solution behavior.

This implementation highlights the close relationship between mathematical formulation and software design emphasized in Section 17.7.8. By organizing the solution data as a structured vector and providing explicit operations for prediction, scaling, and truncation, the code mirrors the underlying numerical method and supports efficient adaptivity. This framework can be extended naturally to more advanced solvers, including variable-order multistep methods and general linear methods, forming a foundation for modern high-performance time integration algorithms.

# 17.8. Stochastic Simulation of Chemical Reaction Networks

In many applications, particularly in systems biology and chemical kinetics, deterministic models based on ordinary differential equations are insufficient to capture the true dynamics. When molecule counts are small, randomness plays a fundamental role, and the system must be modeled probabilistically. This leads to a description in terms of stochastic processes, where the evolution is governed by random reaction events rather than smooth trajectories.

### 17.8.1 Jump-Process Formulation of Chemical Reaction Systems

A deterministic ordinary differential equation model describes the evolution of concentrations or expected copy numbers. Such models are appropriate when species counts are large and fluctuations average out over time. However, in many systems of practical interest, including gene regulation, intracellular signaling, and surface chemistry, molecule counts may be small. In these regimes, stochastic fluctuations are not negligible but instead dominate the system behavior.

In such settings, the correct mathematical framework is a continuous-time Markov jump process. Rather than evolving continuously, the system undergoes discrete changes at random times, corresponding to individual reaction events. Recent work in single-cell biology emphasizes that deterministic ODEs capture only the mean behavior of the system, while stochastic models describe the full probability distribution over discrete molecule counts (Szavits-Nossan and Grima, 2024; Miles, 2025; Jia and Grima, 2024).

Consider a system with $N$ chemical species and $M$ reaction channels. Each reaction $R_j$ is represented as:

$$
R_j:\quad \nu_j^- \longrightarrow \nu_j^+, \qquad
\nu_j = \nu_j^+ - \nu_j^-
\tag{17.8.1}
$$

where $\nu_j^- , \nu_j^+ \in \mathbb{N}^N$ denote the input and output stoichiometric vectors, respectively. The net effect of the reaction is given by the vector $\nu_j$, which describes how the molecule counts change when the reaction occurs.

Collecting all such changes leads to the stoichiometric matrix:

$$
S =
\begin{bmatrix}
\,| & | &        & |\, \\
\nu_1 & \nu_2 & \cdots & \nu_M \\
\,| & | &        & |\, 
\end{bmatrix}
\in \mathbb{Z}^{N \times M}
\tag{17.8.2}
$$

This matrix encodes the structure of the reaction network and determines how the system evolves in response to reaction events.

Let $X(t) \in \mathbb{N}^N$ denote the random vector of molecule counts at time $t$. Each reaction channel $R_j$ is associated with a propensity function $a_j(x)$, which specifies the probability that the reaction occurs in an infinitesimal time interval. More precisely,

$$\Pr\{R_j \text{ fires in } [t, t+\Delta t) \mid X(t)=x\}= a_j(x)\,\Delta t + o(\Delta t) \tag{17.8.3}$$

The propensity function therefore determines the rate at which each reaction occurs, conditioned on the current state of the system. For mass-action kinetics in a well-mixed system, the propensity takes the form:

$$a_j(x) = c_j \prod_{i=1}^{N} \binom{x_i}{(\nu_j^-)_i} \tag{17.8.4}$$

where $c_j$ is a rate constant and the combinatorial factor accounts for the number of ways reactant molecules can be selected from the current population.

When a reaction $R_j$ occurs, the system state is updated according to:

$$X \leftarrow X + \nu_j \tag{17.8.5}$$

Thus, the evolution of the system consists of a sequence of random jumps, each corresponding to a reaction event that changes the molecule counts by a fixed amount. The timing and selection of these events are governed by the propensity functions, leading to a stochastic trajectory that reflects both the structure of the reaction network and the inherent randomness of molecular interactions.

This jump-process formulation provides the foundation for stochastic simulation algorithms, which generate sample paths of the system and allow the study of distributions, fluctuations, and rare events that cannot be captured by deterministic models.

## 17.8.2. Chemical Master Equation and Probability Evolution

The stochastic dynamics introduced in the jump-process formulation can be described at the level of probability distributions through the chemical master equation. Let $P(x,t) = \Pr\{X(t)=x\}$, denote the probability that the system is in state $x \in \mathbb{N}^N$ at time $t$. The time evolution of this distribution is governed by:

$$
\frac{d}{dt} P(x,t)
= \sum_{j=1}^{M} a_j(x-\nu_j)\, P(x-\nu_j,t)
- \sum_{j=1}^{M} a_j(x)\, P(x,t)
\tag{17.8.6}
$$

This equation expresses the conservation of probability through a balance of inflow and outflow terms. The first summation represents the flow of probability into the state $x$. Each term corresponds to a reaction $R_j$ that, when applied to a previous state $x - \nu_j$, produces the state $x$. The contribution of this pathway is weighted by both the probability of being in the precursor state and the propensity of the reaction.

The second summation represents the flow of probability out of the state $x$. For each reaction channel $R_j$, the system may leave state $x$ at a rate determined by the propensity $a_j(x)$, transferring probability mass to a new state $x + \nu_j$.

Taken together, these terms define a system of coupled linear differential equations over the countable state space. The chemical master equation is therefore an exact description of the stochastic dynamics, capturing the full time evolution of the probability distribution rather than a single trajectory.

Despite its exactness, direct numerical solution of the chemical master equation is rarely feasible in practice. The state space $\mathbb{N}^N$ is countably infinite, and even when truncated to a finite domain, the number of possible states grows combinatorially with the number of species and the range of molecule counts. For small systems, finite-state projection methods can be used to restrict the computation to a manageable subset of states while controlling the approximation error.

However, for larger networks, this approach quickly becomes impractical due to the rapid growth in dimensionality. In such cases, Monte Carlo simulation methods, which generate sample paths of the underlying jump process, provide a more scalable alternative. These methods approximate the probability distribution through repeated realizations of the stochastic dynamics, avoiding the need to solve the full system of equations directly (Jia and Grima, 2024).

## 17.8.3. Stochastic Simulation Algorithm and Exact Path Sampling

The stochastic simulation algorithm (SSA) provides a direct method for generating sample paths of the Markov jump process defined in Section 17.8.1. Its derivation follows from interpreting each reaction channel as an independent exponential clock, with rates determined by the corresponding propensity functions. At any given state, these clocks compete, and the next reaction event is identified as the first clock to “ring.”

Let the total propensity at state $x$ be defined as:

$$a_0(x) = \sum_{j=1}^{M} a_j(x) \tag{17.8.7}$$

This quantity represents the overall rate at which any reaction occurs. The probability that no reaction takes place over a time interval of length $\tau$, given the current state $X(t)=x$, is:

$$\Pr\{\text{no reaction in } [t,t+\tau)\mid X(t)=x\}= e^{-a_0(x)\tau} \tag{17.8.8}$$

This exponential form reflects the memoryless property of the underlying Poisson processes. Consequently, the waiting time until the next reaction is exponentially distributed, and can be sampled as:

$$\tau = -\frac{\ln u_1}{a_0(x)}, \qquad u_1 \sim U(0,1) \tag{17.8.9}$$

Once the time $\tau$ to the next event has been determined, the specific reaction channel must be selected. Conditioned on a reaction occurring, the probability that it is reaction $j$ is proportional to its propensity,

$$\Pr\{J=j \mid X(t)=x\}= \frac{a_j(x)}{a_0(x)} \tag{17.8.10}$$

This defines a discrete probability distribution over the reaction channels, from which the index $J$ is sampled. After selecting the reaction, the system state is updated according to the stoichiometric change associated with that reaction,

$$X(t+\tau) = X(t) + \nu_J \tag{17.8.11}$$

The process then repeats: the propensities are recomputed at the new state, a new waiting time is sampled, and another reaction is selected.

In summary, the SSA advances the system by repeatedly performing three steps: sampling the waiting time to the next reaction, selecting the reaction channel based on relative propensities, and updating the state accordingly. This procedure generates statistically exact sample paths of the underlying stochastic process for well-mixed systems, meaning that the trajectories produced are consistent with the chemical master equation without approximation.

The simplicity and exactness of the SSA make it a fundamental tool for simulating stochastic chemical kinetics, particularly in regimes where discrete fluctuations play a dominant role.

### Rust Implementation

Following the development of the stochastic simulation algorithm in Section 17.8.3, Program 17.8.1 provides a concrete implementation of the Gillespie direct method for simulating chemical reaction networks as continuous-time Markov jump processes. As described through Equations (17.8.7)–(17.8.11), the evolution of the system is governed by random reaction events, with both the timing and selection of reactions determined probabilistically by the propensity functions. In practical computation, this requires efficient evaluation of propensities, careful sampling of exponential waiting times, and consistent updates of the system state according to stoichiometric changes. This program realizes these steps in a structured manner, illustrating how stochastic trajectories can be generated that are statistically consistent with the underlying chemical master equation.

At the core of the implementation is the `Reaction` structure, which encapsulates the essential components of each reaction channel. These include the stoichiometric change vector, corresponding to the update rule in Equation (17.8.5), the rate constant, and a list of reactant indices used to compute the propensity function. The `propensity` method evaluates the rate $a_j(x)$ for a given state, following the general form introduced in Equation (17.8.4). The `apply` method performs the state update by adding the stoichiometric change vector to the current state, ensuring that molecule counts remain nonnegative.

The `ReactionNetwork` structure organizes the collection of reactions and provides methods for computing all propensities and selecting the next reaction event. The total propensity $a_0(x)$, defined in Equation (17.8.7), is computed as the sum of individual propensities. This quantity determines both the rate of the exponential waiting time and the normalization of the discrete distribution used to select the reaction channel. The `select_reaction` function implements this selection by accumulating propensities until a threshold is reached, corresponding to sampling from the distribution defined in Equation (17.8.10).

The function `simulate_ssa` implements the stochastic simulation algorithm itself. Starting from an initial state, it repeatedly performs the three fundamental steps: sampling the waiting time $\tau$ according to Equation (17.8.9), selecting a reaction channel based on relative propensities, and updating the system state using the stoichiometric rule of Equation (17.8.11). The simulation proceeds until the final time is reached or no further reactions are possible. Each event is recorded as a trajectory point, allowing the reconstruction of the full stochastic path.

The auxiliary structures `TrajectoryPoint` and the associated printing functions provide a clear representation of the simulated trajectory. Each point stores the time, the system state, and the reaction that occurred, making it possible to trace the evolution of the system step by step. This explicit representation highlights the discrete and stochastic nature of the dynamics, in contrast to the smooth trajectories produced by deterministic models.

The `main` function demonstrates the algorithm on a simple reaction network involving birth, conversion, and death processes. It initializes the system, executes the simulation, and prints the resulting trajectory along with summary statistics such as the number of events and the final state. This example illustrates how stochastic simulation captures both the randomness in event timing and the variability in system evolution, even for relatively simple reaction networks.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
// Program 17.8.1: Gillespie Stochastic Simulation Algorithm for a Chemical Reaction Network
//
// Problem statement:
// Simulate a continuous-time Markov jump process for a chemical reaction
// network using the direct stochastic simulation algorithm.
//
// The implementation follows the SSA structure described in Section 17.8.3:
// compute reaction propensities, sample the waiting time to the next event,
// select the reaction channel, and update the molecule counts according to
// the corresponding stoichiometric change.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug)]
struct Reaction {
    name: &'static str,
    stoichiometric_change: Vec<i32>,
    rate_constant: f64,
    reactants: Vec<usize>,
}

impl Reaction {
    fn propensity(&self, state: &[u32]) -> f64 {
        let mut value = self.rate_constant;

        for &species_index in &self.reactants {
            value *= state[species_index] as f64;
        }

        value
    }

    fn apply(&self, state: &mut [u32]) {
        for (x_i, &nu_i) in state.iter_mut().zip(self.stoichiometric_change.iter()) {
            let updated = *x_i as i32 + nu_i;

            assert!(
                updated >= 0,
                "Reaction {} produced a negative molecule count.",
                self.name
            );

            *x_i = updated as u32;
        }
    }
}

#[derive(Clone, Debug)]
struct ReactionNetwork {
    species_names: Vec<&'static str>,
    reactions: Vec<Reaction>,
}

impl ReactionNetwork {
    fn propensities(&self, state: &[u32]) -> Vec<f64> {
        self.reactions
            .iter()
            .map(|reaction| reaction.propensity(state))
            .collect()
    }

    fn total_propensity(propensities: &[f64]) -> f64 {
        propensities.iter().sum()
    }

    fn select_reaction(propensities: &[f64], total: f64, rng: &mut StdRng) -> usize {
        let threshold = rng.r#gen::<f64>() * total;
        let mut cumulative = 0.0;

        for (j, &a_j) in propensities.iter().enumerate() {
            cumulative += a_j;

            if cumulative >= threshold {
                return j;
            }
        }

        propensities.len() - 1
    }
}

#[derive(Clone, Debug)]
struct TrajectoryPoint {
    time: f64,
    state: Vec<u32>,
    reaction_name: &'static str,
}

fn simulate_ssa(
    network: &ReactionNetwork,
    initial_state: Vec<u32>,
    t_final: f64,
    seed: u64,
) -> Vec<TrajectoryPoint> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut time = 0.0;
    let mut state = initial_state;

    let mut trajectory = Vec::new();

    trajectory.push(TrajectoryPoint {
        time,
        state: state.clone(),
        reaction_name: "initial",
    });

    while time < t_final {
        let propensities = network.propensities(&state);
        let total = ReactionNetwork::total_propensity(&propensities);

        if total <= 0.0 {
            break;
        }

        let u1 = rng.r#gen::<f64>().max(f64::MIN_POSITIVE);
        let tau = -u1.ln() / total;

        if time + tau > t_final {
            break;
        }

        let reaction_index = ReactionNetwork::select_reaction(&propensities, total, &mut rng);
        let reaction = &network.reactions[reaction_index];

        time += tau;
        reaction.apply(&mut state);

        trajectory.push(TrajectoryPoint {
            time,
            state: state.clone(),
            reaction_name: reaction.name,
        });
    }

    trajectory
}

fn print_state_header(network: &ReactionNetwork) {
    print!("{:>10} {:>16}", "time", "reaction");

    for species in &network.species_names {
        print!(" {:>10}", species);
    }

    println!();
    println!("{}", "-".repeat(40 + 11 * network.species_names.len()));
}

fn print_trajectory(network: &ReactionNetwork, trajectory: &[TrajectoryPoint]) {
    print_state_header(network);

    for point in trajectory {
        print!("{:>10.5} {:>16}", point.time, point.reaction_name);

        for value in &point.state {
            print!(" {:>10}", value);
        }

        println!();
    }
}

fn main() {
    let network = ReactionNetwork {
        species_names: vec!["A", "B"],
        reactions: vec![
            Reaction {
                name: "birth_A",
                stoichiometric_change: vec![1, 0],
                rate_constant: 1.0,
                reactants: vec![],
            },
            Reaction {
                name: "conversion_A_to_B",
                stoichiometric_change: vec![-1, 1],
                rate_constant: 0.08,
                reactants: vec![0],
            },
            Reaction {
                name: "death_B",
                stoichiometric_change: vec![0, -1],
                rate_constant: 0.12,
                reactants: vec![1],
            },
        ],
    };

    let initial_state = vec![10, 0];
    let t_final = 20.0;
    let seed = 2026;

    let trajectory = simulate_ssa(&network, initial_state, t_final, seed);

    println!("Gillespie Stochastic Simulation Algorithm");
    println!("=========================================");
    println!();
    println!("Reaction network:");
    println!("  birth_A:           A increases by 1");
    println!("  conversion_A_to_B: A decreases by 1, B increases by 1");
    println!("  death_B:           B decreases by 1");
    println!();
    println!("Final simulation time = {:.4}", t_final);
    println!("Random seed           = {}", seed);
    println!("Number of events      = {}", trajectory.len() - 1);
    println!();

    print_trajectory(&network, &trajectory);

    if let Some(last) = trajectory.last() {
        println!();
        println!("Final state");
        println!("-----------");

        for (name, value) in network.species_names.iter().zip(last.state.iter()) {
            println!("{:>4} = {}", name, value);
        }

        println!("time = {:.6}", last.time);
    }
}
```

Program 17.8.1 demonstrates the practical implementation of the stochastic simulation algorithm for chemical reaction networks, translating the theoretical framework of Markov jump processes into a computational procedure. The algorithm generates statistically exact sample paths, reflecting the full stochastic dynamics described by the chemical master equation.

The results highlight several key features of stochastic simulation. The timing of events is inherently random, governed by exponential waiting times, and the sequence of reactions varies across realizations. This leads to trajectories that exhibit fluctuations and irregular behavior, in contrast to the smooth solutions of deterministic models. Such variability is essential for capturing phenomena that arise from intrinsic noise, particularly in systems with small molecule counts.

The structure of the implementation also emphasizes the importance of efficient data handling and sampling. By organizing reactions, propensities, and state updates in a modular way, the code closely mirrors the mathematical formulation and supports scalability to larger systems. This design can be extended naturally to more advanced methods, including accelerated algorithms and hybrid approaches discussed in later sections. Overall, the program provides a foundation for understanding and implementing stochastic simulation methods, illustrating how probabilistic models of chemical kinetics can be realized in practice and used to explore the dynamics of complex systems.

## 17.8.4. Connection to Deterministic Models and Mean-Field Limits

The stochastic formulation developed in the previous sections connects naturally to deterministic ordinary differential equation models through a large-copy-number approximation. When molecule counts are sufficiently large, random fluctuations become relatively small compared to the mean behavior, and the system can be approximated by a deterministic trajectory.

Starting from the jump-process description, one may consider the evolution of the expected value of the state. Taking expectations of the stochastic dynamics leads to:

$$\frac{d}{dt}\mathbb{E}[X(t)] = S\, \mathbb{E}[a(X(t))] \tag{17.8.12}$$

This equation expresses the rate of change of the expected molecule counts in terms of the stoichiometric matrix $S$ and the expected propensities. However, this system is not closed, since the expectation of a nonlinear function $a(X(t))$ generally depends on higher-order moments of the distribution.

A closure approximation is obtained when fluctuations are sufficiently small, allowing the approximation,

$$
\mathbb{E}[a(X)] \approx a(\mathbb{E}[X])
\tag{17.8.13}
$$

Under this assumption, the expectation operator can be moved inside the propensity function, yielding a closed system for the mean,

$$\frac{d}{dt} y = S \, a(y) \tag{17.8.14}$$

where $y(t) \approx \mathbb{E}[X(t)]$. This system corresponds precisely to the deterministic rate equations commonly used in chemical kinetics and reaction network modeling.

The validity of this approximation depends critically on the magnitude of stochastic fluctuations. When molecule counts are large and the system behaves smoothly, the approximation is often accurate, and the deterministic model provides a good description of the dynamics.

However, this approximation breaks down in regimes where fluctuations play a central role. In systems exhibiting bistability, bursty gene expression, or rare events, the distribution of $X(t)$ may be highly non-Gaussian, and the mean alone does not capture the essential behavior. In such cases, the approximation $\mathbb{E}[a(X)] \approx a(\mathbb{E}[X])$ fails, and the deterministic model may give misleading or incomplete predictions.

Thus, the deterministic rate equations can be understood as a mean-field limit of the stochastic system, valid under conditions of large copy numbers and small relative fluctuations. Outside this regime, the full stochastic formulation is necessary to accurately describe the system dynamics.

## 17.8.5. Accelerated and Hybrid Methods for Stochastic Simulation

While the stochastic simulation algorithm provides an exact realization of the Markov jump process, its computational cost can become prohibitive in systems where reactions occur frequently. Each reaction event requires recomputation of propensities and sampling operations, so the total cost grows with the number of events rather than the simulation time alone. This motivates the development of accelerated and hybrid approaches that retain accuracy while reducing computational effort.

A first level of acceleration is achieved through more efficient implementations of the SSA itself. In many reaction networks, only a subset of propensities changes after each reaction event. By exploiting dependency graphs, one can update only those propensities affected by the last reaction, avoiding unnecessary recomputation. This leads to significant savings in large systems with sparse interaction structure. Methods such as RNMC combine dependency tracking with parallel execution to further improve efficiency, reducing the overhead associated with repeated updates (Zichi et al., 2024).

Recent theoretical work has pushed this idea further by developing exact algorithms capable of simulating multiple reactions in sublinear time under certain assumptions. These methods exploit structural properties of the reaction network to reduce the complexity of event selection and state updates. Implementations in Rust emphasize both performance and safety, aligning with the need for reliable high-performance simulation in scientific computing (Petrack and Doty, 2025).

In addition to exact accelerations, approximate methods provide another avenue for reducing computational cost. A prominent example is tau-leaping, in which multiple reaction events are aggregated over a finite time interval $\tau$. Instead of simulating reactions one at a time, the number of firings of each reaction channel is sampled from a Poisson distribution,

$$
K_j \sim \text{Poisson}(a_j(x)\tau), \qquad
X(t+\tau) = x + \sum_{j=1}^{M} \nu_j K_j
\tag{17.8.15}
$$

This approach significantly reduces the number of steps required, especially in regimes where propensities vary slowly and many reactions occur over short time intervals. However, it introduces an approximation, since the propensities are assumed to remain approximately constant over the interval $\tau$.

To balance accuracy and efficiency, hybrid methods combine exact and approximate strategies. These methods dynamically switch between SSA and tau-leaping depending on the current regime of the system. For example, when molecule counts are low or fluctuations are critical, the exact SSA is used. When the system is in a regime with high activity and relatively smooth behavior, tau-leaping is employed to accelerate the simulation. This adaptive strategy allows the method to maintain accuracy where needed while achieving substantial computational savings overall (Trigo Trindade and Zygalakis, 2024).

These developments illustrate a central theme in stochastic simulation: efficiency depends on exploiting both the structure of the reaction network and the dynamical regime of the system. By combining exact algorithms, approximate methods, and adaptive switching strategies, modern approaches achieve scalable performance while preserving the essential stochastic behavior.

### Rust Implementation

Following the discussion in Section 17.8.5 on accelerated stochastic simulation methods, Program 17.8.2 provides a practical implementation of the tau-leaping approach for chemical reaction networks. As described in Equation (17.8.15), this method replaces the event-by-event evolution of the stochastic simulation algorithm with a coarse-grained update over a finite time interval, during which multiple reaction firings are sampled simultaneously. In numerical computation, this approximation significantly reduces the number of simulation steps when reaction activity is high, while retaining the essential stochastic behavior of the system. The program demonstrates how Poisson-distributed reaction counts can be used to advance the system state efficiently, illustrating the trade-off between computational cost and accuracy emphasized in the preceding discussion.

At the core of the implementation is the `Reaction` structure, which defines the stoichiometric change, rate constant, and reactant indices associated with each reaction channel. The `propensity` method evaluates the rate $a_j(x)$ for the current state, consistent with the formulation introduced in Equation (17.8.4). These propensities determine the expected number of reaction firings over a time interval of length $\tau$, forming the basis for the tau-leaping approximation.

The function `sample_poisson` generates random samples from a Poisson distribution with parameter $\lambda = a_j(x)\tau$. This corresponds directly to the stochastic model described in Equation (17.8.15), where each reaction channel fires a random number of times during the interval. The implementation uses a simple multiplicative algorithm, which is efficient for moderate values of $\lambda$ and sufficient for demonstration purposes.

The `apply_tau_leap` function performs a single tau-leap update. It computes the Poisson-distributed number of firings for each reaction channel, aggregates the resulting stoichiometric changes, and updates the system state accordingly. This approach contrasts with the stochastic simulation algorithm, where reactions are processed one at a time. Here, multiple reactions are applied simultaneously, leading to larger but less frequent state updates.

The `simulate_tau_leaping` function advances the system over the full simulation interval by repeatedly applying tau-leap steps. At each iteration, the time is incremented by a fixed value $\tau$, and the state is updated based on the sampled reaction counts. The trajectory is stored as a sequence of states and corresponding firing counts, providing insight into both the system evolution and the stochastic activity within each interval.

The `main` function demonstrates the method using the same reaction network as in Program 17.8.1, allowing direct comparison between exact and approximate simulation approaches. It initializes the system, executes the tau-leaping simulation, and prints the resulting trajectory. The output includes both species counts and the number of reaction firings in each interval, illustrating how the method aggregates stochastic events over time.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rand = "0.8"
```

```rust
// Program 17.8.2: Tau-Leaping Simulation for a Chemical Reaction Network
//
// Problem statement:
// Simulate a chemical reaction network using the tau-leaping approximation.
// Instead of firing one reaction at a time, the method samples the number
// of firings of each reaction channel over a fixed time interval tau.
//
// This program demonstrates Equation (17.8.15), where each reaction count
// is sampled from a Poisson distribution with mean a_j(x) tau.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug)]
struct Reaction {
    name: &'static str,
    stoichiometric_change: Vec<i32>,
    rate_constant: f64,
    reactants: Vec<usize>,
}

impl Reaction {
    fn propensity(&self, state: &[u32]) -> f64 {
        let mut value = self.rate_constant;

        for &species_index in &self.reactants {
            value *= state[species_index] as f64;
        }

        value
    }
}

#[derive(Clone, Debug)]
struct ReactionNetwork {
    species_names: Vec<&'static str>,
    reactions: Vec<Reaction>,
}

#[derive(Clone, Debug)]
struct TauLeapPoint {
    time: f64,
    state: Vec<u32>,
    firings: Vec<u32>,
}

fn sample_poisson(lambda: f64, rng: &mut StdRng) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }

    let limit = (-lambda).exp();
    let mut product = 1.0;
    let mut k = 0_u32;

    loop {
        k += 1;
        product *= rng.r#gen::<f64>();

        if product <= limit {
            return k - 1;
        }
    }
}

fn apply_tau_leap(
    network: &ReactionNetwork,
    state: &[u32],
    tau: f64,
    rng: &mut StdRng,
) -> (Vec<u32>, Vec<u32>) {
    let mut firings = Vec::with_capacity(network.reactions.len());

    for reaction in &network.reactions {
        let lambda = reaction.propensity(state) * tau;
        firings.push(sample_poisson(lambda, rng));
    }

    let mut net_change = vec![0_i32; state.len()];

    for (reaction, &k_j) in network.reactions.iter().zip(firings.iter()) {
        for (i, &nu_ij) in reaction.stoichiometric_change.iter().enumerate() {
            net_change[i] += nu_ij * k_j as i32;
        }
    }

    let mut next_state = Vec::with_capacity(state.len());

    for (&x_i, &delta_i) in state.iter().zip(net_change.iter()) {
        let updated = x_i as i32 + delta_i;

        next_state.push(updated.max(0) as u32);
    }

    (next_state, firings)
}

fn simulate_tau_leaping(
    network: &ReactionNetwork,
    initial_state: Vec<u32>,
    t_final: f64,
    tau: f64,
    seed: u64,
) -> Vec<TauLeapPoint> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut time = 0.0;
    let mut state = initial_state;

    let mut trajectory = Vec::new();

    trajectory.push(TauLeapPoint {
        time,
        state: state.clone(),
        firings: vec![0; network.reactions.len()],
    });

    while time + tau <= t_final + 1.0e-12 {
        let (next_state, firings) = apply_tau_leap(network, &state, tau, &mut rng);

        time += tau;
        state = next_state;

        trajectory.push(TauLeapPoint {
            time,
            state: state.clone(),
            firings,
        });
    }

    trajectory
}

fn print_header(network: &ReactionNetwork) {
    print!("{:>8}", "time");

    for species in &network.species_names {
        print!(" {:>8}", species);
    }

    for reaction in &network.reactions {
        print!(" {:>18}", reaction.name);
    }

    println!();
    println!("{}", "-".repeat(10 + 9 * network.species_names.len() + 19 * network.reactions.len()));
}

fn print_trajectory(network: &ReactionNetwork, trajectory: &[TauLeapPoint]) {
    print_header(network);

    for point in trajectory {
        print!("{:>8.3}", point.time);

        for value in &point.state {
            print!(" {:>8}", value);
        }

        for count in &point.firings {
            print!(" {:>18}", count);
        }

        println!();
    }
}

fn main() {
    let network = ReactionNetwork {
        species_names: vec!["A", "B"],
        reactions: vec![
            Reaction {
                name: "birth_A",
                stoichiometric_change: vec![1, 0],
                rate_constant: 1.0,
                reactants: vec![],
            },
            Reaction {
                name: "conversion_A_to_B",
                stoichiometric_change: vec![-1, 1],
                rate_constant: 0.08,
                reactants: vec![0],
            },
            Reaction {
                name: "death_B",
                stoichiometric_change: vec![0, -1],
                rate_constant: 0.12,
                reactants: vec![1],
            },
        ],
    };

    let initial_state = vec![10, 0];
    let t_final = 20.0;
    let tau = 0.5;
    let seed = 2026;

    let trajectory = simulate_tau_leaping(&network, initial_state, t_final, tau, seed);

    println!("Tau-Leaping Simulation for a Chemical Reaction Network");
    println!("=====================================================");
    println!();
    println!("Reaction network:");
    println!("  birth_A:           A increases by 1");
    println!("  conversion_A_to_B: A decreases by 1, B increases by 1");
    println!("  death_B:           B decreases by 1");
    println!();
    println!("Final simulation time = {:.4}", t_final);
    println!("Tau-leap step size    = {:.4}", tau);
    println!("Random seed           = {}", seed);
    println!("Number of tau steps   = {}", trajectory.len() - 1);
    println!();

    print_trajectory(&network, &trajectory);

    if let Some(last) = trajectory.last() {
        println!();
        println!("Final state");
        println!("-----------");

        for (name, value) in network.species_names.iter().zip(last.state.iter()) {
            println!("{:>4} = {}", name, value);
        }

        println!("time = {:.6}", last.time);
    }
}
```

Program 17.8.2 demonstrates how the tau-leaping method provides an efficient approximation to stochastic simulation by grouping multiple reaction events into discrete time intervals. This approach reduces computational cost significantly in systems with high reaction activity, where the event-by-event simulation of the stochastic simulation algorithm becomes expensive.

The numerical results highlight the key trade-off inherent in tau-leaping. By assuming that propensities remain approximately constant over each interval, the method sacrifices exactness in exchange for efficiency. This leads to trajectories that capture the overall stochastic behavior but differ from exact sample paths, particularly when the chosen stepsize is large or the system exhibits rapid changes.

The structure of the implementation emphasizes the importance of combining probabilistic sampling with efficient data handling. By computing reaction counts using Poisson distributions and applying aggregated updates, the method achieves a balance between accuracy and performance. This framework can be extended naturally to adaptive tau selection, hybrid methods that switch between exact and approximate regimes, and more advanced acceleration techniques discussed in Section 17.8.5.

## 17.8.6. Modern Developments in Stochastic Simulation

Recent developments in stochastic simulation extend the scope of classical algorithms beyond forward trajectory generation, introducing new capabilities for analysis, approximation, and optimization. These advances reflect the growing need to not only simulate stochastic systems, but also to extract quantitative information from them and to integrate them into larger computational pipelines.

One important direction is the construction of surrogate models that approximate the stochastic dynamics while remaining computationally tractable. The Holimap method, for example, builds effective linear surrogate models through moment matching. By approximating the evolution of probability distributions using reduced representations, this approach enables efficient computation of statistical quantities without explicitly simulating a large number of trajectories. This is particularly useful in systems where direct sampling would be computationally expensive, but where capturing the distributional behavior remains essential (Jia and Grima, 2024).

Another significant development is the emergence of differentiable stochastic simulation methods. In many applications, such as parameter estimation and system design, it is necessary to compute gradients of outputs with respect to model parameters. Classical stochastic simulation algorithms are not directly amenable to such gradient-based techniques, since they involve discrete random events and nondifferentiable operations.

To address this, methods such as the differentiable Gillespie algorithm modify the simulation procedure to enable gradient propagation. By carefully restructuring the sampling process and its dependence on parameters, these methods allow the use of gradient-based optimization techniques in stochastic settings. This capability is particularly valuable in the design of synthetic biological circuits and in data-driven modeling, where parameters must be tuned to match observed behavior (Rijal and Mehta, 2025).

These developments illustrate a broader shift in stochastic simulation. Traditional methods focused on generating sample paths, whereas modern approaches emphasize integration with statistical inference, optimization, and machine learning frameworks. By enabling efficient approximation of distributions and supporting gradient-based methods, these new techniques expand the applicability of stochastic models in both scientific and engineering contexts.

## 17.8.7. Applications and Implementation Perspective for Stochastic Simulation

The stochastic formulation of chemical reaction networks is not merely a theoretical refinement of deterministic models, but a necessity in a wide range of modern applications where fluctuations and discrete events fundamentally shape system behavior. At the same time, the practical implementation of stochastic simulation algorithms requires careful attention to data structures, sampling strategies, and computational efficiency, since these methods operate as probabilistic processes rather than traditional numerical solvers.

A major application area is single-cell gene expression. In such systems, molecule counts are often extremely small, and stochastic effects play a dominant role in determining system behavior. As a result, quantities of interest extend beyond mean trajectories to include variances, Fano factors, and full probability distributions. These statistical descriptors capture phenomena such as transcriptional bursting, noise-induced switching, and cell-to-cell variability, which cannot be represented within deterministic frameworks. Stochastic simulation therefore provides a direct means of studying the distributional structure of the system, aligning with experimental observations that emphasize variability rather than average behavior (Szavits-Nossan and Grima, 2024; Miles, 2025).

Another important application arises in heterogeneous catalysis and surface reaction networks. In these systems, reactions occur on discrete sites, and the state of the system is determined by the occupancy and configuration of those sites. Stochastic models describe not only reaction events, but also diffusion processes and interactions between neighboring sites. The resulting dynamics are inherently discrete and spatially structured, making stochastic simulation essential. Recent developments couple kinetic Monte Carlo methods with neural-network potentials, enabling realistic modeling of complex catalytic processes while capturing both stochastic effects and detailed energetic landscapes (Yokaichiya et al., 2024; Yang, Hellman and Grönbeck, 2025).

These applications illustrate that stochastic simulation is not simply an alternative to deterministic modeling, but a framework for capturing fundamentally different types of behavior. In regimes where randomness, discreteness, and rare events are central, the full probabilistic description provided by the jump-process formulation and its simulation algorithms becomes indispensable.

From an implementation perspective, these considerations translate directly into design requirements. In a Rust-based framework, the reaction network should be represented in a sparse form. Each reaction need only store the species it affects, together with the associated stoichiometric changes and propensity parameters. This avoids unnecessary storage and computation, particularly in large networks where each reaction involves only a small subset of species.

A key supporting structure is the dependency graph, which identifies which propensities must be updated after a given reaction event. Since most reactions affect only a limited number of propensities, this graph enables efficient updates by restricting recomputation to the relevant subset. This is essential for achieving scalable performance in large systems.

Sampling methods should be organized behind a common interface, reflecting the fact that multiple algorithms may be employed depending on the regime. Implementations may include the direct stochastic simulation algorithm, tree-based or indexed sampling methods for efficient reaction selection, and hybrid approaches such as tau-leaping. By abstracting these methods behind a unified interface, the implementation remains flexible and extensible, allowing different strategies to be employed without altering the overall structure of the code.

Efficient random-number generation is another critical component. Since stochastic simulation relies on repeated sampling, the quality and performance of the random-number generator directly affect both accuracy and efficiency. Reproducible seeded streams are particularly important, as they allow simulations to be repeated and verified, which is essential in scientific computing. Moreover, because stochastic simulation is inherently an ensemble computation, where many independent trajectories are generated, the ability to manage multiple independent random streams becomes a practical necessity.

These implementation considerations reflect a central conceptual insight. Stochastic simulation algorithms are not merely numerical methods for solving differential equations, but computational realizations of probabilistic models. Their efficiency depends not only on the mathematical formulation, but also on the organization of data, the design of sampling procedures, and the management of randomness. By aligning these elements with the underlying structure of the reaction network, one obtains implementations that are both faithful to the model and efficient in practice.

# 17.9. Conclusion

This chapter has developed a comprehensive framework for the numerical integration of ordinary differential equations, progressing from the initial value problem formulation and its integral representation through explicit Runge-Kutta methods, adaptive stepsize control, Richardson extrapolation and the Bulirsch-Stoer method, structure-preserving integration for second-order conservative systems, implicit and linearly implicit methods for stiff equations, multistep and predictor-corrector schemes, and stochastic simulation of chemical reaction networks. The central theme throughout is that the choice of numerical method must be guided by the mathematical structure of the problem, whether accuracy, stability, conservation, or stochastic fidelity is the dominant concern. Each section has linked the mathematical derivation of integration schemes to their algorithmic realization in Rust, emphasizing that reliable computation requires not only correct formulas but also careful attention to numerical stability, memory management, and modular software design.

## 17.9.1. Key Takeaways

- The initial value problem $\mathrm{d}y/\mathrm{d}t = f(t,y)$, $y(t_0) = y_0$, $y:[t_0,T] \to \mathbb{R}^m$ provides the canonical framework for numerical integration, with local Lipschitz continuity of $f$ ensuring existence, uniqueness, and controlled error propagation. Higher-order equations are reduced to first-order systems by introducing auxiliary variables as in the reformulation $\mathrm{d}/\mathrm{d}t [q, v]^\top = [v, g(t,q,v)]^\top$. The integral form $y(t_{n+1}) = y(t_n) + \int_{t_n}^{t_{n+1}} f(t,y(t))\,\mathrm{d}t$ provides the basis for all one-step methods, which approximate this integral through evaluations of $f$. The computational abstraction $y_{n+1} = \Psi_h(t_n, y_n)$ separates the state vector, the model, and the numerical method into independent components, enabling modular solver design.
- An explicit $s$-stage Runge-Kutta method computes stage derivatives $k_i = f(t_n + c_i h, y_n + h \sum_{j=1}^{i-1} a_{ij} k_j)$ and advances the solution as $y_{n+1} = y_n + h \sum_{i=1}^s b_i k_i$. The coefficients $a_{ij}$, $b_i$, $c_i$ are organized in a Butcher tableau with strictly lower triangular structure for explicit methods, satisfying the row-sum relation $c_i = \sum_{j<i} a_{ij}$. The order conditions $\sum b_i = 1$, $\sum b_i c_i = 1/2$, $\sum b_i c_i^2 = 1/3$, $\sum_{i,j} b_i a_{ij} c_j = 1/6$ ensure that the numerical update matches the Taylor expansion of the exact solution up to third order, with additional conditions such as $\sum b_i c_i^3 = 1/4$ and $\sum_{i,j,\ell} b_i a_{ij} a_{j\ell} c_\ell = 1/24$ required for fourth-order accuracy.
- The classical fourth-order Runge-Kutta method (RK4) uses four stages with weights $(1, 2, 2, 1)/6$ and achieves local truncation error $O(h^5)$ and global error $O(h^4)$. Applied to the linear test equation $y' = \lambda y$ with $z = h\lambda$, any Runge-Kutta method produces $y_{n+1} = R(z) y_n$ where $R(z) = 1 + z b^\top (I - zA)^{-1} \mathbf{1}$. For explicit $s$-stage methods, $R(z)$ is a polynomial of degree $s$; for RK4 specifically, $R_4(z) = 1 + z + z^2/2 + z^3/6 + z^4/24$, the fourth-order Taylor polynomial of $e^z$. Absolute stability requires $|R(z)| \leq 1$, defining a bounded region in the complex plane for explicit methods. The per-step cost is $s C_f(m) + O(sm)$ operations with $O(sm)$ storage.
- Embedded Runge-Kutta pairs compute two approximations $y_{n+1} = y_n + h \sum b_i k_i$ and $\hat{y}_{n+1} = y_n + h \sum \hat{b}_i k_i$ of different orders from the same stage derivatives, with the error estimate $e_{n+1} = y_{n+1} - \hat{y}_{n+1}$. The normalized error uses scale factors $\mathrm{sci}_i = \mathrm{atol}_i + \mathrm{rtol}_i \max(|y_{n,i}|, |y_{n+1,i}|)$ and the root-mean-square measure $\mathrm{err} = (m^{-1} \sum (e_{n+1,i}/\mathrm{sci}_i)^2)^{1/2}$. The I-type stepsize controller $h_{n+1} = \eta h_n \,\mathrm{err}_n^{-1/q}$ with safety factor $\eta < 1$ and bounds $h_{n+1} \in [f_{\min} h_n, f_{\max} h_n]$ adjusts the stepsize to maintain $\mathrm{err} \leq 1$. The PI-controller $h_{n+1} = \eta h_n \,\mathrm{err}_n^{-\alpha} \,\mathrm{err}_{n-1}^{\beta}$ incorporates memory of past errors to reduce oscillations in the stepsize sequence.
- Richardson extrapolation combines approximations at different resolutions to cancel leading error terms: if $A(h) = y(t+H) + c_1 h^p + O(h^{p+1})$, then $R(h) = (2^p A(h/2) - A(h))/(2^p - 1) = y(t+H) + O(h^{p+1})$. The modified midpoint method, with initialization $z_0 = y_n$, $z_1 = z_0 + h f(t_n, z_0)$, leapfrog recurrence $z_{m+1} = z_{m-1} + 2h f(t_n + mh, z_m)$, and endpoint correction $A_n(H) = \frac{1}{2}(z_n + z_{n-1} + h f(t_n + H, z_n))$, produces an error expansion containing only even powers of $h$: $A_n(H) = y(t_n + H) + \alpha_1 h^2 + \alpha_2 h^4 + \cdots$. The Bulirsch-Stoer method organizes these approximations into an extrapolation tableau using the Aitken-Neville recursion $T_{j,k} = T_{j,k-1} + (T_{j,k-1} - T_{j-1,k-1})/(x_{j-k+1}/x_j - 1)$ with $x_j = (H/n_j)^2$, achieving rapid order acceleration for smooth problems at cost $O(C_f \sum n_j)$ with $O(Kd)$ storage.
- Second-order conservative systems $q''(t) = a(t,q)$ are discretized by the Stormer scheme $q_{n+1} - 2q_n + q_{n-1} = h^2 a(t_n, q_n)$, derived from the centered approximation $q(t+h) - 2q(t) + q(t-h) = h^2 q''(t) + O(h^4)$. The numerically stable increment formulation initializes $\Delta_0 = h v_0 + (h^2/2) a(t_0, q_0)$ and recurs via $\Delta_n = \Delta_{n-1} + h^2 a(t_n, q_n)$, $q_{n+1} = q_n + \Delta_n$. For autonomous separable Hamiltonians $H(q,p) = \frac{1}{2} p^\top M^{-1} p + V(q)$, the velocity-Verlet method performs the Strang splitting $\Phi_h^{VV} = \Phi_{h/2}^V \circ \Phi_h^T \circ \Phi_{h/2}^V$ through half-step momentum $p_{n+1/2} = p_n - (h/2)\nabla V(q_n)$, full-step position $q_{n+1} = q_n + h M^{-1} p_{n+1/2}$, and half-step momentum $p_{n+1} = p_{n+1/2} - (h/2)\nabla V(q_{n+1})$. This composition is symplectic and preserves near-conservation of energy over long integrations.
- A system is stiff when stability rather than accuracy determines the allowable stepsize, characterized by eigenvalues with widely separated magnitudes in the linear system $y' = Ay$ with $A = S\Lambda S^{-1}$. The implicit (backward) Euler method has stability function $R(z) = 1/(1-z)$, which satisfies $|R(z)| \leq 1$ for all $\mathrm{Re}(z) \leq 0$ (A-stability) and $\lim_{z \to -\infty} R(z) = 0$ (L-stability). Each implicit step requires solving $G(y_{n+1}) = y_{n+1} - y_n - h f(t_{n+1}, y_{n+1}) = 0$ via Newton linearization $(I - hJ_n)\delta_n = hf(t_n, y_n)$, $y_{n+1} = y_n + \delta_n$, with $J_n = \partial f/\partial y$. Rosenbrock methods avoid nonlinear iteration by solving $(I - \gamma h J_n) k_i = h f(\cdots) + h J_n \sum_{j<i} \gamma_{ij} k_j$, reusing a single matrix factorization $W = I - \gamma h J$ across all stages at cost $O(d^3)$ for factorization and $O(d^2)$ storage.
- An $s$-step linear multistep method $\sum_{j=0}^s \alpha_j y_{n+1-j} = h_n \sum_{j=0}^s \beta_j f_{n+1-j}$ reuses past solution and derivative values to achieve high efficiency after initialization. The third-order Adams-Bashforth formula $y_{n+1} = y_n + (h/12)(23 f_n - 16 f_{n-1} + 5 f_{n-2})$ is explicit and derived from polynomial interpolation of $f$, while the third-order Adams-Moulton formula $y_{n+1} = y_n + (h/12)(5 f_{n+1} + 8 f_n - f_{n-1})$ is implicit. The predictor-corrector PECE cycle combines them: predict with Adams-Bashforth, evaluate $\hat{f}_{n+1} = f(t_{n+1}, \hat{y}_{n+1})$, correct with Adams-Moulton, and re-evaluate. After initialization, explicit multistep methods require only one new function evaluation per step with $O(sd)$ storage.
- The stability of linear multistep methods is governed by the characteristic polynomials $\rho(\zeta) = \sum \alpha_j \zeta^{s-j}$ and $\sigma(\zeta) = \sum \beta_j \zeta^{s-j}$. Consistency requires $\rho(1) = 0$ and $\rho'(1) = \sigma(1)$, order $p$ requires $\rho(e^\xi) - \xi \sigma(e^\xi) = O(\xi^{p+1})$, and zero-stability requires the root condition: all roots of $\rho(\zeta)$ satisfy $|\zeta| \leq 1$ with simple roots on $|\zeta| = 1$. The Nordsieck representation stores scaled derivatives $z_n = [y_n, h_n y_n', h_n^2 y_n''/2!, \ldots, h_n^q y_n^{(q)}/q!]^\top$ at a single time level, enabling stepsize changes through rescaling and order changes through truncation without reconstructing past values.

- Stochastic chemical reaction networks are modeled as continuous-time Markov jump processes with reactions $R_j: \nu_j^- \to \nu_j^+$ and stoichiometric matrix $S = [\nu_1 \mid \cdots \mid \nu_M] \in \mathbb{Z}^{N \times M}$. The propensity $a_j(x)$ gives the reaction rate with $\Pr\{R_j \text{ fires in } [t, t+\Delta t) \mid X(t) = x\} = a_j(x)\Delta t + o(\Delta t)$. The Gillespie SSA samples waiting time $\tau = -\ln(u_1)/a_0(x)$ where $a_0 = \sum a_j$, selects reaction $j$ with probability $a_j/a_0$, and updates $X \leftarrow X + \nu_j$. The chemical master equation $\mathrm{d}P(x,t)/\mathrm{d}t = \sum_j a_j(x - \nu_j) P(x-\nu_j,t) - \sum_j a_j(x) P(x,t)$ governs the full probability evolution. Tau-leaping approximates multiple firings as $K_j \sim \mathrm{Poisson}(a_j(x)\tau)$ with $X(t+\tau) = x + \sum_j \nu_j K_j$, trading exactness for efficiency. The deterministic rate equation $\mathrm{d}y/\mathrm{d}t = S a(y)$ emerges as a mean-field limit under the closure $E[a(X)] \approx a(E[X])$, valid when molecule counts are large.

## 17.9.2. Advice for Beginners

- Ordinary differential equations form one of the most important foundations of scientific computing because they describe how systems evolve over time. Before studying numerical methods, make sure you understand what an initial value problem represents physically. Whether modeling planetary motion, electrical circuits, population dynamics, chemical reactions, or machine learning systems, the central goal is always the same: determine how the state of a system changes as time progresses.
- Start with Section 17.1 and focus on the idea that differential equations are transformed into algebraic computations. The most important concept is not the formula itself but the realization that numerical integration approximates the continuous evolution of a system through a sequence of discrete time steps. Once this viewpoint is clear, the later algorithms become much easier to understand.
- Runge–Kutta methods in Sections 17.2 and 17.3 should be your first major focus. Learn the classical fourth-order Runge–Kutta method (RK4) thoroughly before studying more advanced techniques. RK4 is widely used, relatively easy to understand, and provides excellent accuracy for many problems. Pay particular attention to how multiple function evaluations are combined to produce a more accurate estimate than Euler's method. Understanding RK4 will make adaptive Runge–Kutta methods much easier to learn.
- When studying adaptive stepsize control, remember that numerical integration is always a balance between accuracy and computational cost. Smaller steps improve accuracy but increase computation time. Adaptive methods automatically adjust the step size to maintain a desired error level. This idea appears throughout modern scientific computing and is one of the most important practical concepts in numerical integration.
- The Bulirsch–Stoer method in Section 17.4 introduces extrapolation techniques. Although the algorithm may initially appear complicated, focus first on the central idea: compute several approximations with different step sizes and combine them to eliminate dominant error terms. Understanding this principle is more important than memorizing the extrapolation table.
- Section 17.5 introduces symplectic methods for conservative systems. Beginners often focus only on local accuracy, but this section teaches a different lesson: preserving the underlying physics can be more important than minimizing local truncation error. Velocity–Verlet and related symplectic methods maintain long-term energy behavior far better than many higher-order methods. This is a fundamental concept in computational physics and molecular dynamics.
- Stiff equations in Section 17.6 are often challenging for newcomers because explicit methods that work well elsewhere may completely fail. Before studying implicit solvers, understand the concept of stiffness and why stability can become more important than accuracy. Learn the Dahlquist test equation and stability regions carefully, as these ideas form the foundation of modern stiff integration algorithms.
- Multistep methods in Section 17.7 illustrate another important computational principle: reusing information from previous steps. Unlike Runge–Kutta methods, which compute everything from the current state, multistep methods leverage historical information to improve efficiency. Focus on understanding why this reuse reduces computational cost and how predictor–corrector schemes combine efficiency with reliability.
- Section 17.8 introduces stochastic simulation, which differs fundamentally from deterministic integration. Instead of computing a single trajectory, stochastic methods generate random realizations of a process. Beginners should remember that randomness is not numerical noise but an essential part of the model itself. Understanding this distinction is crucial when studying chemical reaction networks, biological systems, and many modern simulation methods.
- For Rust implementations, begin with simple examples such as the harmonic oscillator and exponential decay. Compare numerical solutions against known analytical solutions whenever possible. Plot trajectories, monitor errors, and experiment with different step sizes. These experiments often provide deeper intuition than mathematical derivations alone.
- Finally, do not judge a numerical method solely by its order of accuracy. In practice, robustness, stability, conservation properties, computational cost, and ease of implementation are often equally important. The best integrator is not necessarily the most accurate one, but the one that most effectively matches the mathematical structure of the problem being solved.

## 17.9.3. Further Learning with GenAI

To deepen your understanding of numerical integration of ordinary differential equations in Rust, consider using the following GenAI prompts:

 1. Write a Rust program that implements the computational abstraction for initial value problems by defining a state vector, a right-hand-side closure, and a one-step Euler update map $y_{n+1} = y_n + h f(t_n, y_n)$. Reformulate the second-order harmonic oscillator $q''(t) = -q(t)$ as a first-order system with state $y = [q, v]^\top$ and right-hand side $f(t, y) = [v, -q]^\top$. Integrate from $t_0 = 0$ with initial state $(1, 0)$ using step size $h = 0.05$ for 20 steps and print the computed trajectory. Compare the final values with the exact solution $q(t) = \cos(t)$, $v(t) = -\sin(t)$.
 2. Implement a Rust program that realizes a general explicit Runge-Kutta solver driven by a Butcher tableau data structure storing $a_{ij}$, $b_i$, and $c_i$. Define tableaux for the explicit midpoint method and the classical fourth-order Runge-Kutta method, and apply both to the harmonic oscillator with step size $h = 0.1$ for 10 steps. For each stage $i$, construct the intermediate state $y_n + h \sum_{j<i} a_{ij} k_j$, evaluate $f$ at $(t_n + c_i h)$, and form the update $y_{n+1} = y_n + h \sum b_i k_i$. Report the maximum absolute error at the final time for both methods and compare the number of stages.
 3. Build a Rust program that implements adaptive stepsize control using the Bogacki-Shampine RK3(2) embedded pair. Compute both the third-order and second-order approximations from the same four stage evaluations, estimate the local error as $e_{n+1} = y_{n+1} - \hat{y}_{n+1}$, normalize it using $\mathrm{sci}_i = \mathrm{atol} + \mathrm{rtol} \max(|y_{n,i}|, |y_{n+1,i}|)$ and the RMS norm, and apply an I-type controller $h_{n+1} = \eta h_n \,\mathrm{err}^{-1/q}$ with safety factor $\eta = 0.9$. Integrate the harmonic oscillator from $t = 0$ to $t = 10$ with tolerances $\mathrm{atol} = 10^{-8}$, $\mathrm{rtol} = 10^{-6}$. Report accepted steps, rejected steps, function evaluations, and the final error.
 4. Write a Rust program that implements the Bulirsch-Stoer method using a modified midpoint base integrator and Richardson extrapolation. For each macro-step, compute modified midpoint approximations with substep counts $n_j \in \{2, 4, 6, 8, 10, 12\}$, form the extrapolation tableau using the Aitken-Neville recursion $T_{j,k} = T_{j,k-1} + (T_{j,k-1} - T_{j-1,k-1})/(x_{j-k+1}/x_j - 1)$ with $x_j = (H/n_j)^2$, and accept the step when the error estimate $e_j = T_{j,j} - T_{j,j-1}$ satisfies a normalized tolerance. Integrate the harmonic oscillator from $t = 0$ to $t = 10$ with tolerances $\mathrm{atol} = 10^{-10}$, $\mathrm{rtol} = 10^{-8}$. Report macro-steps, function evaluations, and final error.
 5. Implement a Rust program that realizes the velocity-Verlet method for a separable Hamiltonian system with $H(q,p) = \frac{1}{2}p^2 + \frac{1}{2}q^2$. Perform the symmetric update: half-step momentum $p_{n+1/2} = p_n + (h/2)(-q_n)$, full-step position $q_{n+1} = q_n + h p_{n+1/2}$, half-step momentum $p_{n+1} = p_{n+1/2} + (h/2)(-q_{n+1})$. Integrate from $t = 0$ to $t = 100$ with $h = 0.05$ and initial state $(q_0, p_0) = (1, 0)$. Compute the Hamiltonian at each step and report the maximum energy deviation over the entire trajectory, demonstrating near-conservation of energy characteristic of symplectic integrators.
 6. Build a Rust program that implements the backward Euler method with Newton linearization for a stiff scalar equation $y'(t) = -1000(y(t) - \cos t) - \sin t$ with exact solution $y(t) = \cos t$. At each step, solve $G(y_{n+1}) = y_{n+1} - y_n - h f(t_{n+1}, y_{n+1}) = 0$ using Newton iteration with the Jacobian matrix $W = I - hJ_n$ where $J_n = -1000$. Use step size $h = 0.05$ over $[0, 1]$. Report the total number of Newton iterations, the final absolute error, and verify that the method remains stable despite $h |\lambda| = 50$ being far outside the explicit Euler stability region.
 7. Write a Rust program that implements a one-stage Rosenbrock method for the stiff autonomous equation $y'(t) = -1000(y(t) - 1)$ with exact solution $y(t) = 1 - e^{-1000t}$. Form the matrix $W = I - \gamma h J_n$ with $\gamma = 1$ and Jacobian $J = -1000$, solve the linear system $W k_1 = h f(t_n, y_n)$, and update $y_{n+1} = y_n + k_1$. Integrate over $[0, 1]$ with $h = 0.05$. Report the number of linear solves, the absence of nonlinear Newton iterations, and the final absolute error. Compare the computational structure with the backward Euler approach.
 8. Implement a Rust program that performs Adams-Bashforth-Moulton predictor-corrector integration for $y'(t) = -2y$ with $y(0) = 1$ and exact solution $y(t) = e^{-2t}$. Use a ring buffer storing three history entries initialized by RK4. At each subsequent step, predict with the third-order Adams-Bashforth formula $\hat{y}_{n+1} = y_n + (h/12)(23 f_n - 16 f_{n-1} + 5 f_{n-2})$, evaluate $\hat{f}_{n+1}$, and correct with Adams-Moulton $y_{n+1} = y_n + (h/12)(5 \hat{f}_{n+1} + 8 f_n - f_{n-1})$. Use $h = 0.1$ over $[0, 2]$ and report the absolute error at each step.
 9. Build a Rust program that demonstrates the Nordsieck representation for multivalue integration. Construct a Nordsieck vector $z_n = [y_n, h y_n', h^2 y_n''/2!, h^3 y_n'''/3!, h^4 y_n^{(4)}/4!]^\top$ for $y(t) = e^t$ at $t = 0$ with $h = 0.1$. Implement prediction by evaluating $\sum_j z_j \theta^j$ at $\theta = 1$, stepsize rescaling by multiplying each component by the appropriate power of the ratio $h_{\mathrm{new}}/h_{\mathrm{old}}$, and order truncation by discarding higher components. Report the prediction error before and after rescaling to $h = 0.05$, and after truncation to order 2.
10. Write a Rust program that implements both the Gillespie stochastic simulation algorithm and tau-leaping for a reaction network with birth ($\emptyset \to A$, rate 1.0), conversion ($A \to B$, rate 0.08 per $A$), and death ($B \to \emptyset$, rate 0.12 per $B$). For the SSA, sample waiting time $\tau = -\ln(u_1)/a_0$ and select reactions with probability $a_j/a_0$. For tau-leaping with $\tau = 0.5$, sample firings $K_j \sim \mathrm{Poisson}(a_j \tau)$. Start from $(A, B) = (10, 0)$ and simulate to $t = 20$. Report the number of events (SSA) versus number of tau-steps, and the final states from both methods.

By engaging with these prompts, you will gain a deeper understanding of how initial value problem formulation, Runge-Kutta methods, adaptive stepsize control, Richardson extrapolation, structure-preserving integration, stiff solvers, multistep methods, and stochastic simulation form an integrated toolkit for the numerical integration of ordinary differential equations, and how the Rust implementations ensure numerical stability, type safety, and reproducibility across all stages of the computation.

## 17.9.4. Homework Exercises

To reinforce your learning, complete the following exercises:

 1. Implement a Rust program that solves the harmonic oscillator $q''(t) = -q(t)$ reformulated as the first-order system $y' = [v, -q]^\top$ using the forward Euler method $y_{n+1} = y_n + h f(t_n, y_n)$. Use the initial condition $(q_0, v_0) = (1, 0)$, step size $h = 0.05$, and 20 steps from Program 17.1.1. Print the computed trajectory at each step and report the maximum absolute error in position at the final time $t = 1.0$ by comparing with $q(t) = \cos(t)$. Verify that the error is consistent with first-order accuracy.
 2. Implement a Rust program that solves the harmonic oscillator using both the explicit midpoint method and the classical fourth-order Runge-Kutta method, encoded as Butcher tableaux. For the midpoint method, use $a_{21} = 0.5$, $b = [0, 1]$, $c = [0, 0.5]$. For RK4, use $b = [1/6, 1/3, 1/3, 1/6]$, $c = [0, 0.5, 0.5, 1]$ with the standard $a_{ij}$ coefficients from Program 17.2.2. Use $h = 0.1$ for 10 steps with initial state $(1, 0)$. Report the maximum absolute error at the final time for both methods and verify that RK4 achieves significantly smaller error than the midpoint method.
 3. Implement a Rust program that performs adaptive Runge-Kutta integration of the harmonic oscillator using the Bogacki-Shampine RK3(2) embedded pair from Program 17.3.1. Use initial stepsize $h = 0.25$, minimum stepsize $10^{-8}$, maximum stepsize $0.5$, absolute tolerance $10^{-8}$, and relative tolerance $10^{-6}$. Integrate from $t = 0$ to $t = 10$. Report the number of accepted steps, rejected steps, total function evaluations, and the maximum absolute error at the final time. Verify that the error is consistent with the prescribed tolerances.
 4. Implement a Rust program that performs Bulirsch-Stoer integration of the harmonic oscillator using the modified midpoint method with substep sequence $\{2, 4, 6, 8, 10, 12\}$ and Aitken-Neville extrapolation, following Program 17.4.1. Use initial macro-step $H = 0.5$, tolerances $\mathrm{atol} = 10^{-10}$ and $\mathrm{rtol} = 10^{-8}$, and integrate from $t = 0$ to $t = 10$. Report the number of accepted macro-steps, rejected macro-steps, function evaluations, and the final absolute error. Compare the error with the adaptive RK3(2) result from the previous exercise.
 5. Implement a Rust program that integrates the harmonic oscillator using the velocity-Verlet method with $h = 0.05$ for 2000 steps (final time $t = 100$), using the parameters from Program 17.5.2. At each step, compute $p_{n+1/2} = p_n + (h/2)(-q_n)$, $q_{n+1} = q_n + h p_{n+1/2}$, $p_{n+1} = p_{n+1/2} + (h/2)(-q_{n+1})$. Compute the Hamiltonian $H = \frac{1}{2}p^2 + \frac{1}{2}q^2$ at every step and report the initial energy, final energy, final energy deviation, and maximum energy deviation over the entire trajectory. Verify that the energy deviation remains bounded and does not drift secularly.
 6. Implement a Rust program that solves the stiff equation $y'(t) = -1000(y(t) - \cos t) - \sin t$ with $y(0) = 1$ using the backward Euler method with Newton iteration, following Program 17.6.1. Use step size $h = 0.05$ over $[0, 1]$ with Newton tolerance $10^{-12}$ and maximum 20 Newton iterations per step. Report the total Newton iterations, the number of failed convergence steps, and the absolute error at the final time compared to $y(t) = \cos(t)$. Verify that the implicit method remains stable despite $h|\lambda| = 50$.
 7. Implement a Rust program that solves the stiff autonomous equation $y'(t) = -1000(y(t) - 1)$ with $y(0) = 0$ using the Rosenbrock-Euler method from Program 17.6.2. Form $W = I - \gamma h J$ with $\gamma = 1$, $J = -1000$, and $h = 0.05$ over $[0, 1]$. At each step solve $W k_1 = h f(t_n, y_n)$ and set $y_{n+1} = y_n + k_1$. Report the number of linear solves, confirm zero nonlinear iterations, and compute the absolute error against $y(t) = 1 - e^{-1000t}$ at the final time.
 8. Implement a Rust program that performs Adams-Bashforth-Moulton predictor-corrector integration for $y'(t) = -2y$ with $y(0) = 1$ using the third-order formulas from Program 17.7.1. Initialize a ring buffer with 3 entries generated by RK4 with $h = 0.1$. At each subsequent step, predict with $\hat{y}_{n+1} = y_n + (h/12)(23 f_n - 16 f_{n-1} + 5 f_{n-2})$, evaluate $\hat{f}_{n+1} = f(t_{n+1}, \hat{y}_{n+1})$, and correct with $y_{n+1} = y_n + (h/12)(5 \hat{f}_{n+1} + 8 f_n - f_{n-1})$. Integrate to $t = 2$ and report the absolute error $|y_n - e^{-2t_n}|$ at each printed step.
 9. Implement a Rust program that demonstrates the Nordsieck representation by constructing the vector $z_n = [y_n, h y_n', h^2 y_n''/2!, h^3 y_n'''/3!, h^4 y_n^{(4)}/4!]^\top$ for $y(t) = e^t$ at $t = 0$ with $h = 0.1$, following Program 17.7.2. Predict $y(t_n + h)$ by evaluating $\sum_j z_j \theta^j$ at $\theta = 1$ and report the error against $e^{0.1}$. Rescale to $h = 0.05$ by multiplying component $j$ by $(0.05/0.1)^j$ and predict again. Truncate to order 2 and predict once more. Report errors at all three stages.
10. Implement a Rust program that simulates a chemical reaction network with birth ($\emptyset \to A$, rate 1.0), conversion ($A \to B$, rate $0.08 \times A$), and death ($B \to \emptyset$, rate $0.12 \times B$) using the Gillespie SSA from Program 17.8.1 with seed 2026 and initial state $(A, B) = (10, 0)$ over $[0, 20]$. Then simulate the same network using tau-leaping from Program 17.8.2 with $\tau = 0.5$ and the same seed. Report the number of SSA events versus tau-leaping steps, the final states from both methods, and discuss how tau-leaping trades exactness for computational efficiency.

These exercises span the full range of numerical integration methods developed in this chapter, from Euler and Runge-Kutta schemes through adaptive stepsize control, Richardson extrapolation, structure-preserving symplectic methods, implicit and Rosenbrock methods for stiff systems, predictor-corrector multistep methods, Nordsieck multivalue representations, and stochastic simulation algorithms. By implementing them in Rust, you will gain direct experience with the numerical considerations, algorithmic design choices, and interpretive judgment that distinguish reliable integration from mechanical formula application.

# References

 1. Bachmann, B., Bonaventura, L., Casella, F., Fernández-García, S., Gómez-Mármol, M. and Hannebohm, P. (2025) ‘Self-Adjusting Multi-Rate Runge-Kutta Methods: Analysis and Efficient Implementation in an Open Source Framework’, *Journal of Scientific Computing*, 105, 30. doi:10.1007/s10915-025-03049-y.
 2. Balos, C.J., Day, M., Esclapez, L., Felden, A.M., Gardner, D.J., Hassanaly, M., Reynolds, D.R., Rood, J.S., Sexton, J.M., Wimer, N.T. and Woodward, C.S. (2025) ‘SUNDIALS time integrators for exascale applications with many independent systems of ordinary differential equations’, *The International Journal of High Performance Computing Applications*, 39(1), pp. 123–146. doi:10.1177/10943420241280060.
 3. Bayleyegn, T., Faragó, I. and Havasi, Á. (2024) ‘On the convergence of multiple Richardson extrapolation combined with explicit Runge–Kutta methods’, *Periodica Mathematica Hungarica*, 88, pp. 335–353. doi:10.1007/s10998-023-00557-y.
 4. Biswas, A., Ketcheson, D.I., Roberts, S., Seibold, B. and Shirokoff, D. (2025) ‘Explicit Runge–Kutta Methods that Alleviate Order Reduction’, *SIAM Journal on Numerical Analysis*, 63(4), pp. 1398–1426. doi:10.1137/23M1606812.
 5. Blanes, S., Casas, F. and Murua, A. (2024) ‘Splitting methods for differential equations’, *Acta Numerica*, 33, pp. 1–161. doi:10.1017/S0962492923000077.
 6. Caldana, M. and Hesthaven, J.S. (2024) *Neural ordinary differential equations for model order reduction of stiff systems*. MOX Report 53/2024. Politecnico di Milano.
 7. Caldas, F. and Soares, C. (2024) ‘Machine learning in orbit estimation: A survey’, *Acta Astronautica*, 220, pp. 97–107. doi:10.1016/j.actaastro.2024.03.072.
 8. Cano, B. and Moreta, M.J. (2024) ‘Solving reaction-diffusion problems with explicit Runge–Kutta exponential methods without order reduction’, *ESAIM: Mathematical Modelling and Numerical Analysis*, 58, pp. 1053–1085. doi:10.1051/m2an/2024011.
 9. Conte, D., Pagano, G. and Paternoster, B. (2023) ‘Time-accurate and highly-stable explicit peer methods for stiff differential problems’, *Communications in Nonlinear Science and Numerical Simulation*, 119, 107136. doi:10.1016/j.cnsns.2023.107136.
10. D’Afiero, F.M. (2026) ‘Embedded strong stability preserving Runge-Kutta methods with adaptive time stepping for shock-dominated flows’, *Computers & Fluids*, 305, 106916. doi:10.1016/j.compfluid.2025.106916.
11. Dravins, I., Koch, M., Griehl, V. and Kormann, K. (2025) ‘Performance evaluation of mixed-precision Runge–Kutta methods for the solution of partial differential equations’, *The International Journal of High Performance Computing Applications*. doi:10.1177/10943420251392963.
12. Dreger, R., Kirfel, T., Pozzer, A., Rosanka, S., Sander, R. and Taraborrelli, D. (2025) ‘Optimized step size control within the Rosenbrock solvers for stiff chemical ordinary differential equation systems in KPP version 2.2.3_rs4’, *Geoscientific Model Development*, 18, pp. 4273–4291. doi:10.5194/gmd-18-4273-2025.
13. Ekanathan, S., Smith, O. and Rackauckas, C. (2024) *A Fully Adaptive Radau Method for the Efficient Solution of Stiff Ordinary Differential Equations at Low Tolerances*. arXiv:2412.14362. doi:10.48550/arXiv.2412.14362.
14. Günther, M. and Sandu, A. (2025) *Multirate Methods for Ordinary Differential Equations*. arXiv:2505.20062. doi:10.48550/arXiv.2505.20062.
15. Hu, G.-D. and Wang, Z. (2024) ‘A modified Runge–Kutta method for increasing stability properties’, *Journal of Computational and Applied Mathematics*, 441, 115698. doi:10.1016/j.cam.2023.115698.
16. Jia, C. and Grima, R. (2024) ‘Holimap: an accurate and efficient method for solving stochastic gene network dynamics’, *Nature Communications*, 15, 6557. doi:10.1038/s41467-024-50716-z.
17. Miles, C.E. (2025) ‘Mechanistic inference of stochastic gene expression from structured single-cell data’, *Current Opinion in Systems Biology*, 42, 100555. doi:10.1016/j.coisb.2025.100555.
18. Mitsui, T. and Hu, G.-D. (2023) *Numerical Analysis of Ordinary and Delay Differential Equations*. Singapore: Springer. doi:10.1007/978-981-19-9263-6.
19. Murugesh, V., Priyadharshini, M., Sharma, Y.K., Lilhore, U.K., Alroobaea, R., Alsufyani, H., Baqasah, A.M. and Simaiya, S. (2025) ‘A novel hybrid framework for efficient higher order ODE solvers using neural networks and block methods’, *Scientific Reports*, 15, 8456. doi:10.1038/s41598-025-90556-5.
20. Oates, C.J., Karvonen, T., Teckentrup, A.L., Strocchi, M. and Niederer, S.A. (2025) ‘Probabilistic Richardson extrapolation’, *Journal of the Royal Statistical Society Series B*, 87(2), pp. 457–479. doi:10.1093/jrsssb/qkae098.
21. Pan, M., Zhang, J. and Zhang, S. (2025) ‘Three sixth-order explicit symplectic Runge–Kutta–Nyström methods with exact parameters’, *Results in Applied Mathematics*, 26, 100568. doi:10.1016/j.rinam.2025.100568.
22. Petrack, J. and Doty, D. (2025) ‘Exactly simulating stochastic chemical reaction networks in sub-constant time per reaction’, arXiv. doi:10.48550/arXiv.2508.04079.
23. Qin, X., Huang, Y., Miao, J. and Chen, T. (2024) ‘Strong Stability Preserving Two-Derivative Two-Step Runge-Kutta Methods’, *Mathematics*, 12(16), 2465. doi:10.3390/math12162465.
24. Ranocha, H. and Giesselmann, J. (2024) ‘Stability of step size control based on a posteriori error estimates’, *Computational Science and Engineering*, 1, 1. doi:10.1007/s44207-024-00001-0.
25. Reynolds, D.R., Gardner, D.J., Woodward, C.S. and Chinomona, R. (2023) ‘ARKODE: A flexible IVP solver infrastructure for one-step methods’, *ACM Transactions on Mathematical Software*, 49(2), pp. 1–26. doi:10.1145/3594632.
26. Saleh, M., Kovács, E. and Kallur, N. (2023) ‘Adaptive step size controllers based on Runge-Kutta and linear-neighbor methods for solving the non-stationary heat conduction equation’, *Networks and Heterogeneous Media*, 18(3), pp. 1059–1082. doi:10.3934/nhm.2023046.
27. Steinebach, G. (2023) ‘Construction of Rosenbrock–Wanner method Rodas5P and numerical benchmarks within the Julia Differential Equations package’, *BIT Numerical Mathematics*, 63, Article 27. doi:10.1007/s10543-023-00967-x.
28. Szavits-Nossan, J. and Grima, R. (2024) ‘Solving stochastic gene-expression models using queueing theory: A tutorial review’, *Biophysical Journal*, 123(9), pp. 1034–1057. doi:10.1016/j.bpj.2024.04.004.
29. Trigo Trindade, T. and Zygalakis, K.C. (2024) ‘A hybrid tau-leap for simulating chemical kinetics with applications to parameter estimation’, *Royal Society Open Science*, 11, 240157. doi:10.1098/rsos.240157.
30. Vermeire, B.C. (2023) ‘Embedded paired explicit Runge-Kutta schemes’, *Journal of Computational Physics*, 487, 112159. doi:10.1016/j.jcp.2023.112159.
31. Worsham, J.M. and Kalita, J.K. (2025) ‘A guide to neural ordinary differential equations: Machine learning for data-driven digital engineering’, *Digital Engineering*, 6, 100060. doi:10.1016/j.dte.2025.100060.
32. Yang, Y., Hellman, A. and Grönbeck, H. (2025) ‘Kinetic Monte Carlo-Based Reactor Model Including Catalyst Shape Changes’, *ACS Catalysis*, 15(13), pp. 11502–11511. doi:10.1021/acscatal.5c01592.
33. Ye, K., Cai, Z., Wang, M., Yang, K. and Liu, X. (2025) ‘An adaptive symplectic integrator for gravitational dynamics’, *Astronomy & Astrophysics*, 699, A170. doi:10.1051/0004-6361/202451822.
34. Yokaichiya, T., Ikeda, T., Muraoka, K. and Nakayama, A. (2024) ‘On-the-fly kinetic Monte Carlo simulations with neural network potentials for surface diffusion and reaction’, *Journal of Chemical Physics*, 160, 204108. doi:10.1063/5.0199240.
35. Zichi, L., Barter, D., Sivonxay, E., Spotte-Smith, E.W.C., Mohanakrishnan, R.S., Chan, E.M., Persson, K.A. and Blau, S.M. (2024) ‘RNMC: kinetic Monte Carlo implementations for complex reaction networks’, *Journal of Open Source Software*, 9(104), 7244. doi:10.21105/joss.07244.
