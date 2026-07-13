---
weight: 4500
title: "Chapter 20"
description: "Partial Differential Equations"
icon: "article"
date: "2026-07-06T00:00:00+07:00"
lastmod: "2026-07-06T00:00:00+07:00"
katex: true
draft: false
toc: true
---

{{% alert icon="💡" context="info" %}}
<strong>"<em>Partial differential equations are the most powerful way to express the laws of nature and the dynamics of the universe.\</em>" — *Clifford Truesdel*</strong>
{{% /alert %}}

{{% alert icon="📘" context="success" %}}
<p style="text-align: justify;"><em>Chapter 20 introduces numerical methods for partial differential equations (PDEs), which form the foundation of mathematical models in fluid dynamics, heat transfer, wave propagation, electromagnetics, and many other scientific and engineering applications. The chapter begins with the formulation of initial-value PDEs and the method-of-lines approach for converting spatially discretized PDEs into systems of ordinary differential equations. Numerical methods for flux-conservative and diffusive problems are then developed, emphasizing stability, conservation, and structure-preserving discretizations. The treatment is extended to multidimensional problems, including operator-splitting and alternating-direction techniques. The chapter subsequently examines efficient solvers for elliptic boundary value problems, including Fourier methods, cyclic reduction, relaxation schemes, and multigrid algorithms. Spectral methods based on Galerkin, tau, and collocation formulations are also introduced, highlighting their high-order accuracy and computational efficiency. Throughout the chapter, mathematical theory is integrated with practical Rust implementations, providing readers with the tools needed to develop accurate and efficient PDE solvers for modern scientific computing.</em></p>
{{% /alert %}}

# 20.1. Introduction

Partial differential equations arise when the unknown quantity depends on more than one independent variable, usually space and time. In this chapter the emphasis is on initial-value partial differential equations, where a state is prescribed at $t=0$ and the numerical task is to compute its evolution for $t>0$. Although PDEs are traditionally classified as hyperbolic, parabolic, or elliptic, a computationally more immediate distinction is whether the method advances a spatially distributed state forward in time or solves for a complete spatial field in one coupled operation. The present chapter focuses primarily on the first setting. This viewpoint is central in numerical models of flood propagation, electrochemical transport, heat transfer, and multiphysics systems where advective, diffusive, reactive, and source-driven effects occur simultaneously (Mama et al., 2025; Shen et al., 2024; Vijaywargiya and Fu, 2024).

## 20.1.1. Initial-Value PDEs and Canonical Evolution Forms

Let $\Omega\subset\mathbb{R}^d$ denote the spatial domain and let $t\in[0,T]$ denote time. The unknown may be a scalar field,

$$u:\Omega\times[0,T]\to\mathbb{R},$$

or a vector-valued field,

$$\mathbf{u}:\Omega\times[0,T]\to\mathbb{R}^m$$

Two canonical forms will recur throughout this chapter. The first is the conservative or balance-law form,

$$\partial_t \mathbf{u}+\nabla\cdot \mathbf{F}(\mathbf{u},x,t)=\mathbf{S}(\mathbf{u},x,t),\qquad\mathbf{u}(x,0)=\mathbf{u}_0(x)\tag{20.1.1}$$

Here $\mathbf{F}$ is the flux tensor or flux vector, depending on the spatial dimension and number of conserved quantities, and $\mathbf{S}$ represents source terms. Equation (20.1.1) states that the rate of change of the conserved state inside a region is determined by fluxes across the boundary of that region together with internal production, forcing, reaction, or loss mechanisms. This form is fundamental for conservation laws, shallow-water flow, gas dynamics, traffic flow, and many transport-dominated models.

The second canonical form is a diffusive or parabolic evolution equation,

\begin{equation}
\begin{aligned}
\partial_t u
&= \nabla \cdot \!\left(D(u,x,t)\nabla u\right)
+ R(u,x,t), \\
&u(x,0) = u_0(x)
\end{aligned}
\tag{20.1.2}
\end{equation}

where $D(u,x,t)$ is a diffusion coefficient, tensor, or mobility, and $R(u,x,t)$ denotes reaction or source terms. Equation (20.1.2) describes smoothing driven by spatial gradients. When $D$ is positive, gradients are dissipated over time, and short-wavelength features decay more rapidly than long-wavelength features. This form appears in heat conduction, species diffusion, electrochemical transport, porous-medium flow, and many gradient-flow models. Recent numerical work on porous-medium and electro-diffusion systems emphasizes that such discretizations should preserve positivity, mass, and energy dissipation whenever these properties are part of the continuous model (Vijaywargiya and Fu, 2024; Guo, Yin and Zhang, 2024).

The distinction between (20.1.1) and (20.1.2) is not merely formal. Conservative transport problems require careful representation of fluxes across cell interfaces, especially when discontinuities or sharp fronts occur. Diffusive problems require careful treatment of stiffness, since the smallest resolved spatial scales often impose the most severe time-step restrictions. In coupled advection-diffusion-reaction systems, both issues appear together, which motivates operator splitting, implicit-explicit methods, and exponential integrators that separate a stiff linear part from the remaining nonlinear or advective dynamics (Caliari et al., 2024; Caliari and Cassini, 2024).

## 20.1.2. Hyperbolic and Diffusive Behaviour in Fourier Space

A useful way to understand the numerical character of time-dependent PDEs is to examine the evolution of Fourier modes. Consider first the one-dimensional linear advection equation,

$$u_t+a u_x=0,\qquad a=\text{constant} \tag{20.1.3}$$

If the initial data contain a Fourier component $e^{ikx}$, then the exact solution contains:

$$e^{ikx}e^{-iakt}=e^{ik(x-at)} \tag{20.1.4}$$

The amplitude is preserved, while the phase is transported with speed $a$. Thus, linear advection is nondissipative. Its numerical approximation must therefore control phase error and avoid nonphysical growth without introducing excessive artificial diffusion. For nonlinear hyperbolic conservation laws, the situation is more delicate because waves may steepen into discontinuities. The numerical method must then approximate the physically relevant entropy solution, not merely an arbitrary weak solution. This is why modern high-order finite-volume, finite-difference, and discontinuous Galerkin schemes often include entropy-stable fluxes, limiters, WENO-type nonlinear reconstructions, or positivity-preserving corrections (Zhang et al., 2023; Liu, Lu and Shu, 2024; Liu et al., 2025; Charles and Ray, 2025).

By contrast, consider the one-dimensional linear diffusion equation,

$$u_t=D u_{xx},\qquad D>0\tag{20.1.5}$$

For the same Fourier mode $e^{ikx}$, the exact temporal factor is:

$$e^{-Dk^2t} \tag{20.1.6}$$

The amplitude decays exponentially, and the decay rate increases like $k^2$. High-frequency modes are therefore damped much faster than low-frequency modes. This is the fundamental smoothing mechanism of diffusion, but it is also the source of stiffness in explicit numerical methods. If the spatial mesh resolves wavelengths of order $\Delta x$, then the largest effective wave numbers scale like $1/\Delta x$, and the fastest diffusive decay rates scale like $D/\Delta x^2$. Consequently, explicit schemes for diffusion obey a parabolic time-step restriction of the form:

$$\Delta t=O(\Delta x^2) \tag{20.1.7}$$

which is substantially more restrictive than the hyperbolic Courant restriction,

$$\Delta t=O(\Delta x) \tag{20.1.8}$$

This contrast explains why explicit time stepping is often natural for nonstiff hyperbolic transport, while implicit, semi-implicit, exponential, or split methods become attractive for diffusion and advection-diffusion-reaction systems. Recent exponential Runge-Kutta and Lawson-type methods exploit this separation by treating a constant-coefficient stiff differential operator through matrix functions while retaining explicit or semi-explicit treatment of the remaining terms (Caliari et al., 2024; Caliari and Cassini, 2024).

## 20.1.3. Spatial Discretization and the Method-of-Lines Viewpoint

A central organizing principle in this chapter is the method-of-lines viewpoint. The continuous spatial field is first replaced by a finite set of grid values, nodal values, cell averages, modal coefficients, or finite-element degrees of freedom. These unknowns are collected into a vector

\begin{equation}
U(t) =
\begin{bmatrix}
U_1(t) \\
U_2(t) \\
\vdots \\
U_N(t)
\end{bmatrix}
\tag{20.1.9}
\end{equation}

After spatial discretization, the PDE is transformed into a large system of ordinary differential equations,

$$\frac{dU}{dt}=L(U,t) \tag{20.1.10}$$

where $L$ is the discrete spatial operator. The numerical problem is then divided into two conceptually separate tasks: first, construct a spatial discretization that respects the analytic structure of the PDE; second, choose a time integrator appropriate for the stability, accuracy, and stiffness properties of (20.1.10).

For a one-dimensional conservation law, a finite-volume discretization stores cell averages,

\begin{equation}
\bar{U}_j(t)
=
\frac{1}{\Delta x}
\int_{x_{j-1/2}}^{x_{j+1/2}} U(x,t)\,dx
\tag{20.1.11}
\end{equation}

The canonical semi-discrete conservative update has the form:

\begin{equation}
\frac{d\bar{U}_j}{dt}
=
-\frac{1}{\Delta x}
\left(
\widehat{F}_{j+1/2} - \widehat{F}_{j-1/2}
\right)
+ S_j
\tag{20.1.12}
\end{equation}

Here $\widehat{F}_{j+1/2}$ is a numerical flux at the interface between cells $j$ and $j+1$, and $S_j$ is a suitable cell-averaged or pointwise approximation of the source term. The significance of (20.1.12) is that neighbouring cells exchange equal and opposite fluxes. When the update is summed over all cells, the interior fluxes cancel telescopically, so the only change in the total discrete mass comes from boundary fluxes and sources. This property is essential in flood modeling, shallow-water flow, and other applications where conservation errors can accumulate into physically meaningless long-time behaviour (Shen et al., 2024; Del Grosso et al., 2024; Ersing, Goldberg and Winters, 2025).

For constant-coefficient diffusion on a uniform one-dimensional grid, the standard second-difference approximation gives a semi-discrete system of the form:

\begin{equation}
\frac{dU}{dt}
=
\frac{D}{\Delta x^2}
\begin{bmatrix}
-2 & 1 & 0 & \cdots & 0 \\
1 & -2 & 1 & \ddots & \vdots \\
0 & \ddots & \ddots & \ddots & 0 \\
\vdots & \ddots & 1 & -2 & 1 \\
0 & \cdots & 0 & 1 & -2
\end{bmatrix}
U
+ \text{boundary terms}
\tag{20.1.13}
\end{equation}

Equation (20.1.13) makes the stiffness of diffusion transparent. The discrete Laplacian has eigenvalues whose magnitudes grow like $\Delta x^{-2}$, so explicit time integrators must take small steps as the grid is refined. In one space dimension, implicit methods lead to tridiagonal linear systems, which can be solved in $O(N)$ operations. In higher dimensions, the corresponding sparse systems are larger and motivate Krylov methods, multigrid, alternating-direction splittings, tensor methods, or exponential integrators.

For variable diffusion coefficients, the conservative flux form is often preferable. Writing,

$$q=-D u_x,\qquad u_t=-q_x,\tag{20.1.14}$$

leads naturally to interface fluxes. A typical semi-discrete finite-volume approximation is:

\begin{equation}
\frac{d\bar{u}_j}{dt}
=
\frac{1}{\Delta x}
\left[
D_{j+1/2}\frac{u_{j+1}-u_j}{\Delta x}
-
D_{j-1/2}\frac{u_j-u_{j-1}}{\Delta x}
\right]
\tag{20.1.15}
\end{equation}

The placement of $D$ at interfaces is important. If $D$ has jumps, sharp layers, or material discontinuities, the interface value must represent the physically correct transmission of flux, not merely a smooth arithmetic average. This issue appears naturally in heterogeneous diffusion, porous media, electrochemical systems, and multiscale battery models where material properties may vary across particles, grains, cells, or cooling regions (Mama et al., 2025; Chen et al., 2024; Hasegawa et al., 2024; Kalungi and Menart, 2025).

### Rust Implementation

Following the discussion in Subsection 20.1.3 on spatial discretization and the method-of-lines formulation for time-dependent partial differential equations, Program 20.1.1 provides a practical implementation of a one-dimensional periodic advection-diffusion solver using conservative finite-volume transport and implicit diffusion updates. The program combines the conservative balance-law structure of equations (20.1.11)–(20.1.12) with the diffusive semi-discrete formulation of equations (20.1.13)–(20.1.15), illustrating how hyperbolic transport and parabolic smoothing can be integrated within a unified computational framework. The implementation advances the advection term explicitly through an upwind finite-volume flux while treating diffusion implicitly through a cyclic tridiagonal solve, thereby avoiding the severe parabolic timestep restriction associated with fully explicit diffusion schemes. This structure reflects the central method-of-lines viewpoint introduced in equation (20.1.10), where the spatially discretized PDE is transformed into a large system of ordinary differential equations whose components may be evolved using different numerical strategies according to their stability properties.

At the core of the implementation is the `Grid1D` structure, which encapsulates the spatial discretization of the one-dimensional computational domain. The structure stores the number of finite-volume cells, the spatial interval, and the mesh spacing $\Delta x$, while the `cell_center` function computes the geometric center of each control volume. This organization reflects the finite-volume interpretation of equation (20.1.11), where the numerical unknowns represent cell averages over discrete spatial regions rather than pointwise nodal values. The periodic indexing functions `periodic_left` and `periodic_right` implement the periodic topology of the domain by wrapping neighboring cell references across the boundaries, thereby ensuring consistency between the numerical fluxes and the assumed boundary conditions.

The function `conservative_advection_step` implements the conservative finite-volume transport update corresponding to equation (20.1.12). Numerical fluxes are first evaluated at cell interfaces using an upwind discretization determined by the sign of the advection velocity $a$. The conservative update then computes the difference between incoming and outgoing interface fluxes for each control volume. Because neighboring cells exchange equal and opposite fluxes, the resulting scheme preserves total discrete mass up to roundoff error. This telescoping flux structure is one of the defining advantages of finite-volume methods for conservation laws and is essential for obtaining physically meaningful long-time behavior in transport-dominated simulations.

The implicit diffusion update is implemented through the function `implicit_periodic_diffusion_step`, which discretizes the diffusive operator associated with equations (20.1.13)–(20.1.15). The resulting linear system possesses a cyclic tridiagonal structure due to the periodic boundary conditions. The function `cyclic_tridiagonal_solve` applies a Sherman–Morrison correction to reduce the cyclic system to two standard tridiagonal solves, each performed efficiently using the Thomas algorithm implemented in `thomas_solve`. This implicit treatment eliminates the severe explicit stability restriction associated with equation (20.1.7), allowing stable time integration even when the spatial grid is refined. The structure also demonstrates how implicit discretizations naturally lead to sparse linear algebra problems whose efficient solution becomes central in large-scale PDE simulation.

Several auxiliary diagnostic functions are included to monitor important physical and numerical properties of the evolving solution. The `total_mass` function computes the discrete integral of the solution over the domain and verifies conservation properties of the finite-volume discretization. The `l2_norm` function measures the decay of solution energy associated with diffusion, while `max_value` tracks the amplitude reduction caused by dissipative smoothing. Together, these diagnostics illustrate the distinct physical behavior discussed earlier in Subsection 20.1.2, where advection preserves amplitude while diffusion preferentially damps high-frequency modes according to equation (20.1.6).

The `main` function serves as a complete demonstration of the coupled advection-diffusion framework. It begins by constructing a periodic spatial grid and initializing a localized Gaussian pulse representing the initial condition. The timestep is then selected using a hyperbolic Courant restriction consistent with equation (20.1.8), while diffusion is treated implicitly to avoid the more restrictive parabolic stability condition of equation (20.1.7). During each timestep, the solution first undergoes an explicit conservative advection update and is subsequently advanced through an implicit diffusion solve. Periodic diagnostics report the evolution of the solution maximum, while final diagnostic quantities confirm mass conservation and diffusive smoothing. The resulting computation illustrates the essential method-of-lines philosophy: once the spatial structure of the PDE is discretized correctly, the evolution problem reduces to the stable and efficient numerical integration of a large coupled ODE system.

```rust
// Program 20.1.1. Periodic Finite-Volume Advection-Diffusion Solver Using the Method of Lines
//
// Problem statement:
// Solve the one-dimensional linear advection-diffusion equation
//
//     u_t + a u_x = D u_xx
//
// on a periodic domain. The advection term is advanced by a conservative
// finite-volume upwind flux, while the diffusion term is advanced implicitly
// using a periodic cyclic tridiagonal system. The program demonstrates the
// method-of-lines structure of Section 20.1.3 and combines equations
// (20.1.11)-(20.1.15) in one computational example.

#[derive(Clone)]
struct Grid1D {
    n: usize,
    xmin: f64,
    xmax: f64,
    dx: f64,
}

impl Grid1D {
    fn new(n: usize, xmin: f64, xmax: f64) -> Self {
        assert!(n >= 4, "Grid must contain at least four cells.");
        assert!(xmax > xmin, "Invalid domain.");

        let dx = (xmax - xmin) / n as f64;

        Self {
            n,
            xmin,
            xmax,
            dx,
        }
    }

    fn cell_center(&self, j: usize) -> f64 {
        self.xmin + (j as f64 + 0.5) * self.dx
    }
}

fn initial_condition(x: f64) -> f64 {
    let x0 = 0.35_f64;
    let width = 0.04_f64;

    (-((x - x0).powi(2)) / width).exp()
}

fn periodic_left(j: usize, n: usize) -> usize {
    if j == 0 {
        n - 1
    } else {
        j - 1
    }
}

fn periodic_right(j: usize, n: usize) -> usize {
    if j + 1 == n {
        0
    } else {
        j + 1
    }
}

fn total_mass(u: &[f64], dx: f64) -> f64 {
    u.iter().sum::<f64>() * dx
}

fn l2_norm(u: &[f64], dx: f64) -> f64 {
    (u.iter().map(|v| v * v).sum::<f64>() * dx).sqrt()
}

fn max_value(u: &[f64]) -> f64 {
    u.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn conservative_advection_step(
    u: &[f64],
    advection_speed: f64,
    dt: f64,
    dx: f64,
) -> Vec<f64> {
    let n = u.len();
    let mut flux = vec![0.0_f64; n];

    for j in 0..n {
        let right = periodic_right(j, n);

        flux[j] = if advection_speed >= 0.0 {
            advection_speed * u[j]
        } else {
            advection_speed * u[right]
        };
    }

    let mut next = vec![0.0_f64; n];

    for j in 0..n {
        let left_face = periodic_left(j, n);
        next[j] = u[j] - (dt / dx) * (flux[j] - flux[left_face]);
    }

    next
}

fn implicit_periodic_diffusion_step(
    u: &[f64],
    diffusion: f64,
    dt: f64,
    dx: f64,
) -> Vec<f64> {
    let n = u.len();
    let r = diffusion * dt / (dx * dx);

    let diag_value = 1.0 + 2.0 * r;
    let offdiag_value = -r;

    cyclic_tridiagonal_solve(
        offdiag_value,
        diag_value,
        offdiag_value,
        offdiag_value,
        offdiag_value,
        u,
        n,
    )
}

fn cyclic_tridiagonal_solve(
    lower_value: f64,
    diag_value: f64,
    upper_value: f64,
    corner_lower: f64,
    corner_upper: f64,
    rhs: &[f64],
    n: usize,
) -> Vec<f64> {
    assert_eq!(rhs.len(), n);
    assert!(n >= 4);

    let gamma = -diag_value;

    let mut modified_diag = vec![diag_value; n];
    modified_diag[0] = diag_value - gamma;
    modified_diag[n - 1] =
        diag_value - corner_lower * corner_upper / gamma;

    let lower = vec![lower_value; n - 1];
    let upper = vec![upper_value; n - 1];

    let x = thomas_solve(&lower, &modified_diag, &upper, rhs);

    let mut u_vec = vec![0.0_f64; n];
    u_vec[0] = gamma;
    u_vec[n - 1] = corner_lower;

    let z = thomas_solve(&lower, &modified_diag, &upper, &u_vec);

    let numerator = x[0] + corner_upper * x[n - 1] / gamma;
    let denominator = 1.0 + z[0] + corner_upper * z[n - 1] / gamma;

    let factor = numerator / denominator;

    x.iter()
        .zip(z.iter())
        .map(|(xi, zi)| xi - factor * zi)
        .collect()
}

fn thomas_solve(
    lower: &[f64],
    diag: &[f64],
    upper: &[f64],
    rhs: &[f64],
) -> Vec<f64> {
    let n = diag.len();

    assert_eq!(rhs.len(), n);
    assert_eq!(lower.len(), n - 1);
    assert_eq!(upper.len(), n - 1);

    let mut c_prime = vec![0.0_f64; n - 1];
    let mut d_prime = vec![0.0_f64; n];

    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let denom = diag[i] - lower[i - 1] * c_prime[i - 1];

        if i < n - 1 {
            c_prime[i] = upper[i] / denom;
        }

        d_prime[i] =
            (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom;
    }

    let mut solution = vec![0.0_f64; n];
    solution[n - 1] = d_prime[n - 1];

    for i in (0..n - 1).rev() {
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
    }

    solution
}

fn main() {
    let grid = Grid1D::new(120, 0.0, 1.0);

    let advection_speed: f64 = 1.0;
    let diffusion: f64 = 0.002;

    let final_time: f64 = 0.25;
    let cfl: f64 = 0.45;

    let dt_adv = cfl * grid.dx / advection_speed.abs();
    let number_of_steps = (final_time / dt_adv).ceil() as usize;
    let dt = final_time / number_of_steps as f64;

    let mut u: Vec<f64> = (0..grid.n)
        .map(|j| initial_condition(grid.cell_center(j)))
        .collect();

    let initial_mass = total_mass(&u, grid.dx);
    let initial_l2 = l2_norm(&u, grid.dx);

    println!("Periodic Finite-Volume Advection-Diffusion Solver");
    println!("=================================================");
    println!();

    println!("Grid Parameters");
    println!("---------------");
    println!("Number of cells           = {}", grid.n);
    println!("Domain                    = [{:.3}, {:.3}]", grid.xmin, grid.xmax);
    println!("Grid spacing dx           = {:.6}", grid.dx);
    println!();

    println!("Physical Parameters");
    println!("-------------------");
    println!("Advection speed a         = {:.6}", advection_speed);
    println!("Diffusion coefficient D   = {:.6}", diffusion);
    println!();

    println!("Time-Stepping Parameters");
    println!("------------------------");
    println!("Final time               = {:.6}", final_time);
    println!("Time step dt             = {:.6}", dt);
    println!("Number of steps          = {}", number_of_steps);
    println!();

    for step in 0..number_of_steps {
        let advected =
            conservative_advection_step(&u, advection_speed, dt, grid.dx);

        u = implicit_periodic_diffusion_step(
            &advected,
            diffusion,
            dt,
            grid.dx,
        );

        if step % 20 == 0 || step + 1 == number_of_steps {
            let time = (step + 1) as f64 * dt;

            println!(
                "Step {:>5} | time = {:>.6} | max(u) = {:>.8}",
                step + 1,
                time,
                max_value(&u),
            );
        }
    }

    let final_mass = total_mass(&u, grid.dx);
    let final_l2 = l2_norm(&u, grid.dx);

    println!();
    println!("Final Diagnostics");
    println!("-----------------");
    println!("Initial mass             = {:.12}", initial_mass);
    println!("Final mass               = {:.12}", final_mass);
    println!(
        "Absolute mass change     = {:.6e}",
        (final_mass - initial_mass).abs()
    );
    println!("Initial L2 norm          = {:.12}", initial_l2);
    println!("Final L2 norm            = {:.12}", final_l2);
    println!();

    println!("Sample Numerical Solution");
    println!("-------------------------");
    println!("{:>8} {:>14} {:>18}", "Cell", "x", "u(x,T)");

    for j in (0..grid.n).step_by(10) {
        println!(
            "{:>8} {:>14.6} {:>18.10}",
            j,
            grid.cell_center(j),
            u[j]
        );
    }
}
```

Program 20.1.1 demonstrates how conservative transport and diffusive smoothing can be integrated within a unified method-of-lines framework for time-dependent partial differential equations. The implementation reflects the central structural distinctions developed throughout Section 20.1.3: advection is handled through conservative finite-volume flux balances, while diffusion is treated through an implicit sparse linear solve. This separation allows each physical mechanism to be discretized in a manner consistent with its mathematical properties and stability requirements.

The conservative finite-volume update illustrates the importance of preserving discrete balance laws through interface flux cancellation. Because neighboring control volumes exchange equal and opposite fluxes, the resulting method preserves total mass to machine precision on the periodic domain. At the same time, the implicit diffusion step demonstrates how parabolic stiffness naturally leads to sparse linear algebra structures whose efficient solution becomes a major component of practical PDE simulation. The cyclic tridiagonal solver further highlights the close relationship between boundary conditions and algebraic structure in implicit discretizations.

The numerical results also reproduce the theoretical behavior discussed earlier in Subsection 20.1.2. The advective component transports the Gaussian pulse across the domain without altering the total conserved quantity, while the diffusive component gradually smooths the solution and reduces the $L^2$ norm through dissipation of high-frequency modes. The decrease in the maximum amplitude together with the preservation of total mass provides direct computational evidence of the distinct roles played by hyperbolic transport and parabolic diffusion.

The modular organization of the code makes the framework extensible to more advanced PDE solvers. Higher-order reconstructions, nonlinear limiters, Runge–Kutta time integrators, adaptive timestepping, operator splitting strategies, sparse Krylov solvers, and multidimensional discretizations can all be incorporated while preserving the same underlying method-of-lines structure. Consequently, this implementation serves as a foundation for more sophisticated conservative and multiphysics PDE solvers developed later in the chapter.

## 20.1.4. Boundary Conditions, Invariants, and Computational Structure

Boundary conditions play different roles depending on the type of PDE. For diffusive problems, one usually prescribes boundary data on the entire boundary $\partial\Omega$. Common examples are Dirichlet conditions,

$$u(x,t)=g_D(x,t),\qquad x\in\partial\Omega_D \tag{20.1.16}$$

and Neumann conditions,

$$D\nabla u(x,t)\cdot n=g_N(x,t),\qquad x\in\partial\Omega_N \tag{20.1.17}$$

where $n$ is the outward unit normal. Dirichlet data fix the value of the field, while Neumann data prescribe the normal diffusive flux. In heat transfer, these correspond respectively to prescribed temperature and prescribed heat flux. In electrochemical diffusion, analogous conditions prescribe concentration or flux at a reactive boundary.

For hyperbolic systems, boundary conditions are constrained by characteristic propagation. If the normal flux Jacobian has eigenvalues of both signs, only the incoming characteristic components should be prescribed at a boundary. Outgoing information is determined by the interior solution. Imposing too many boundary conditions can overdetermine the discrete problem and generate spurious reflections or instabilities. Thus, hyperbolic solvers are tightly coupled to wave direction, while diffusive solvers are tightly coupled to elliptic or parabolic linear algebra.

The relevant invariant also depends on the equation. For a conservative balance law without sources and without boundary flux, the integral of the conserved quantity should remain constant. In a finite-volume method this is enforced by the telescoping structure of (20.1.12). For diffusion, the relevant structure is often dissipation: gradients should decay, energy-like quantities should decrease, and positivity should be preserved when the continuous model preserves positivity. For Schrödinger-type equations, however, the second spatial derivative does not imply diffusion. The model:

$$i\psi_t=H\psi\tag{20.1.18}$$

with Hermitian spatial operator (H) has unitary time evolution rather than dissipative smoothing. If $H_h$ is a Hermitian discrete operator, the Crank-Nicolson update can be written as the Cayley transform:

\begin{equation}
\left(I + \frac{i\Delta t}{2}H_h\right)\psi^{n+1}
=
\left(I - \frac{i\Delta t}{2}H_h\right)\psi^n
\tag{20.1.19}
\end{equation}

This update preserves the discrete $\ell^2$ norm when $H_h$ is Hermitian. Thus, the same algebraic time-stepping framework may represent dissipation in a heat equation or norm preservation in a quantum evolution equation, depending on the operator and the physical invariant.

These distinctions have direct consequences for Rust implementations. A conservative hyperbolic solver should store cell averages in contiguous arrays, compute left and right reconstructed states at interfaces, evaluate a numerical flux, and update the state through a conservative difference of face fluxes. This structure maps naturally to a modular design in which the reconstruction, flux function, source term, limiter, and time integrator are separate components. Such separation is mathematically useful because it prevents the conservative update from being hidden inside a monolithic function.

A diffusive implicit solver has a different computational structure. It typically assembles or applies a banded, sparse, or structured matrix. If coefficients and boundary conditions are constant, the factorization or preconditioner can often be reused over many time steps. In one dimension this may reduce to a tridiagonal Thomas solve; in higher dimensions it may require sparse direct methods, Krylov iteration, multigrid, ADI splitting, or tensor-structured exponential methods. Mixed advection-diffusion-reaction problems combine these requirements, which is why recent methods often separate transport, diffusion, and reaction into components whose numerical treatment can be chosen according to stability and structure (Caliari et al., 2024; Caliari and Cassini, 2024; Guo, Yin and Zhang, 2024).

The guiding principle for the rest of this chapter is therefore structural consistency. Conservation laws should be discretized in a form that preserves balance across interfaces. Diffusion equations should be discretized in a form that respects smoothing, positivity, and dissipative behaviour where appropriate. Wave-like equations should preserve the correct phase, energy, or norm structure rather than being treated as if they were diffusive. Once the spatial structure is correctly represented, the PDE becomes a large ODE system, and the remaining numerical questions are those of stability, accuracy, stiffness, computational complexity, and implementation efficiency.

### Rust Implementation

Following the discussion in Subsection 20.1.4 on boundary conditions, invariants, and computational structure for time-dependent partial differential equations, Program 20.1.2 provides a practical implementation of Crank–Nicolson time evolution for a one-dimensional Schrödinger equation on a periodic domain. In contrast to diffusive evolution equations, where numerical methods are designed to reproduce smoothing and dissipation, the Schrödinger equation requires preservation of the underlying unitary structure associated with equation (20.1.18). The program implements the Cayley-transform formulation of the Crank–Nicolson method given in equation (20.1.19), which preserves the discrete $\ell^2$ norm whenever the discrete Hamiltonian operator is Hermitian. The implementation therefore illustrates an important principle emphasized throughout this section: numerical discretizations should preserve the physically relevant invariant of the continuous model rather than imposing artificial dissipative behavior inappropriate for wave-like evolution problems.

At the core of the implementation is the `Complex` structure, which provides a lightweight representation of complex-valued wave functions and supports the arithmetic operations required for Schrödinger evolution. The structure implements addition, subtraction, multiplication, division, and conjugate-related magnitude operations through Rust operator overloading, thereby allowing the numerical expressions appearing in the Crank–Nicolson update to closely resemble their mathematical form. The functions `abs` and `abs2` compute the modulus and squared modulus of a complex number and are used extensively in the invariant diagnostics that monitor preservation of the discrete $\ell^2$ norm throughout the simulation.

The spatial discretization is organized through the `Grid1D` structure, which stores the periodic computational domain, the number of grid points, and the mesh spacing $\Delta x$. The helper functions `periodic_left` and `periodic_right` enforce periodic boundary conditions by wrapping neighboring indices across the computational boundaries. This reflects the structural discussion surrounding equations (20.1.16)–(20.1.17), where the numerical treatment of a PDE depends fundamentally on the imposed boundary conditions. In the present case, periodic boundaries eliminate artificial reflections and preserve translational symmetry in the discrete Hamiltonian operator.

The function `initial_wave_packet` constructs a localized Gaussian wave packet modulated by a complex oscillatory phase. The Gaussian envelope provides spatial localization, while the oscillatory phase introduces a nonzero wave number corresponding to propagating quantum momentum. The resulting initial condition therefore contains both amplitude and phase structure, allowing the simulation to demonstrate dispersive wave propagation while preserving total probability norm. The function `normalize` subsequently rescales the initial data so that the discrete $\ell^2$ norm equals unity, thereby enforcing the probabilistic interpretation associated with Schrödinger evolution.

The discrete Hamiltonian operator is assembled by the function `build_hamiltonian`, which constructs a periodic second-difference approximation to the spatial Laplacian appearing in equation (20.1.18). The resulting matrix is symmetric and therefore Hermitian in the real-valued discrete setting. This property is essential because the norm-preserving behavior of the Crank–Nicolson update in equation (20.1.19) relies on the Hermitian structure of the discrete Hamiltonian. The matrix-vector multiplication required during time evolution is implemented through the function `mat_vec_real_complex`, which applies the real-valued Hamiltonian matrix to the complex-valued wave function.

The function `crank_nicolson_step` implements the invariant-preserving Cayley-transform update corresponding to equation (20.1.19). The method first computes the right-hand side by applying the Hamiltonian operator to the current wave function and then constructs the complex-valued linear system associated with the implicit Crank–Nicolson discretization. The resulting dense linear system is solved by the function `solve_dense_complex_system`, which performs Gaussian elimination with partial pivoting for complex-valued matrices. Although the implementation uses a dense solver for conceptual clarity, the structure naturally extends to sparse Hermitian solvers for large-scale simulations.

Several diagnostic functions are included to monitor the invariant structure of the numerical solution. The function `discrete_l2_norm` computes the discrete norm of the wave function and verifies preservation of the unitary structure throughout time evolution. The function `max_density` evaluates the maximum probability density $|\psi|^2$, which illustrates dispersive spreading of the wave packet without implying physical dissipation. Together, these diagnostics distinguish Schrödinger evolution from diffusive dynamics, where energy-like quantities decrease monotonically through smoothing.

The `main` function serves as a complete demonstration of invariant-preserving time integration for a wave-like PDE. It constructs the periodic spatial grid, assembles the Hermitian Hamiltonian operator, initializes and normalizes the wave packet, and advances the solution using repeated Crank–Nicolson updates. During the evolution, the program continuously monitors the discrete $\ell^2$ norm to verify unitary propagation. The final diagnostics demonstrate that the norm remains preserved up to roundoff error while the wave packet undergoes dispersive spreading. This behavior illustrates the central principle emphasized throughout Subsection 20.1.4: different PDEs possess different physically relevant invariants, and numerical methods should be designed to preserve the correct structural properties of the underlying continuous model.

```rust
// Program 20.1.2. Crank-Nicolson Evolution for a One-Dimensional Schrodinger Equation
//
// Problem statement:
// Solve the one-dimensional Schrodinger-type equation
//
//     i psi_t = H psi
//
// on a periodic domain using the Crank-Nicolson Cayley transform
//
//     (I + i dt H / 2) psi^{n+1} = (I - i dt H / 2) psi^n.
//
// The Hamiltonian H is discretized by a periodic second-difference operator.
// Because H is Hermitian, the update preserves the discrete l2 norm up to
// roundoff error.

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

    fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    fn i() -> Self {
        Self { re: 0.0, im: 1.0 }
    }

    fn abs2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    fn abs(self) -> f64 {
        self.abs2().sqrt()
    }
}

use std::ops::{Add, Div, Mul, Neg, Sub};

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;

        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

struct Grid1D {
    n: usize,
    xmin: f64,
    xmax: f64,
    dx: f64,
}

impl Grid1D {
    fn new(n: usize, xmin: f64, xmax: f64) -> Self {
        assert!(n >= 4, "Grid must contain at least four points.");
        assert!(xmax > xmin, "Invalid domain.");

        let dx = (xmax - xmin) / n as f64;

        Self {
            n,
            xmin,
            xmax,
            dx,
        }
    }

    fn point(&self, j: usize) -> f64 {
        self.xmin + j as f64 * self.dx
    }
}

fn periodic_left(j: usize, n: usize) -> usize {
    if j == 0 {
        n - 1
    } else {
        j - 1
    }
}

fn periodic_right(j: usize, n: usize) -> usize {
    if j + 1 == n {
        0
    } else {
        j + 1
    }
}

fn initial_wave_packet(x: f64) -> Complex {
    let x0 = 0.35_f64;
    let width = 0.01_f64;
    let wave_number = 40.0_f64;

    let envelope = (-((x - x0).powi(2)) / width).exp();
    let phase = wave_number * x;

    Complex::new(envelope * phase.cos(), envelope * phase.sin())
}

fn discrete_l2_norm(psi: &[Complex], dx: f64) -> f64 {
    (psi.iter().map(|z| z.abs2()).sum::<f64>() * dx).sqrt()
}

fn max_density(psi: &[Complex]) -> f64 {
    psi.iter()
        .map(|z| z.abs2())
        .fold(f64::NEG_INFINITY, f64::max)
}

fn normalize(psi: &mut [Complex], dx: f64) {
    let norm = discrete_l2_norm(psi, dx);

    for value in psi.iter_mut() {
        *value = *value * (1.0 / norm);
    }
}

fn build_hamiltonian(grid: &Grid1D) -> Vec<Vec<f64>> {
    let n = grid.n;
    let dx2 = grid.dx * grid.dx;

    let mut h = vec![vec![0.0_f64; n]; n];

    for j in 0..n {
        let left = periodic_left(j, n);
        let right = periodic_right(j, n);

        h[j][j] = 1.0 / dx2;
        h[j][left] = -0.5 / dx2;
        h[j][right] = -0.5 / dx2;
    }

    h
}

fn mat_vec_real_complex(matrix: &[Vec<f64>], x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    let mut y = vec![Complex::zero(); n];

    for i in 0..n {
        for j in 0..n {
            y[i] = y[i] + x[j] * matrix[i][j];
        }
    }

    y
}

fn crank_nicolson_step(
    psi: &[Complex],
    hamiltonian: &[Vec<f64>],
    dt: f64,
) -> Vec<Complex> {
    let n = psi.len();
    let imaginary_factor = Complex::i() * (0.5 * dt);

    let h_psi = mat_vec_real_complex(hamiltonian, psi);

    let mut rhs = vec![Complex::zero(); n];

    for j in 0..n {
        rhs[j] = psi[j] - imaginary_factor * h_psi[j];
    }

    let mut lhs = vec![vec![Complex::zero(); n]; n];

    for i in 0..n {
        for j in 0..n {
            lhs[i][j] = imaginary_factor * hamiltonian[i][j];

            if i == j {
                lhs[i][j] = lhs[i][j] + Complex::one();
            }
        }
    }

    solve_dense_complex_system(lhs, rhs)
}

fn solve_dense_complex_system(
    mut a: Vec<Vec<Complex>>,
    mut b: Vec<Complex>,
) -> Vec<Complex> {
    let n = b.len();

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_size = a[k][k].abs();

        for row in k + 1..n {
            let candidate = a[row][k].abs();

            if candidate > pivot_size {
                pivot_size = candidate;
                pivot_row = row;
            }
        }

        assert!(
            pivot_size > 1.0e-14,
            "Singular or nearly singular matrix detected."
        );

        if pivot_row != k {
            a.swap(k, pivot_row);
            b.swap(k, pivot_row);
        }

        let pivot = a[k][k];

        for j in k..n {
            a[k][j] = a[k][j] / pivot;
        }

        b[k] = b[k] / pivot;

        for row in 0..n {
            if row != k {
                let factor = a[row][k];

                for col in k..n {
                    a[row][col] = a[row][col] - factor * a[k][col];
                }

                b[row] = b[row] - factor * b[k];
            }
        }
    }

    b
}

fn main() {
    let grid = Grid1D::new(80, 0.0, 1.0);

    let final_time: f64 = 0.01;
    let dt: f64 = 0.0001;
    let number_of_steps = (final_time / dt).round() as usize;

    let hamiltonian = build_hamiltonian(&grid);

    let mut psi: Vec<Complex> = (0..grid.n)
        .map(|j| initial_wave_packet(grid.point(j)))
        .collect();

    normalize(&mut psi, grid.dx);

    let initial_norm = discrete_l2_norm(&psi, grid.dx);
    let initial_density = max_density(&psi);

    println!("Crank-Nicolson Schrodinger Evolution");
    println!("====================================");
    println!();

    println!("Grid Parameters");
    println!("---------------");
    println!("Number of grid points     = {}", grid.n);
    println!("Domain                    = [{:.3}, {:.3}]", grid.xmin, grid.xmax);
    println!("Grid spacing dx           = {:.6}", grid.dx);
    println!();

    println!("Time-Stepping Parameters");
    println!("------------------------");
    println!("Final time               = {:.6}", final_time);
    println!("Time step dt             = {:.6}", dt);
    println!("Number of steps          = {}", number_of_steps);
    println!();

    for step in 0..number_of_steps {
        psi = crank_nicolson_step(&psi, &hamiltonian, dt);

        if step % 20 == 0 || step + 1 == number_of_steps {
            let time = (step + 1) as f64 * dt;

            println!(
                "Step {:>5} | time = {:>.6} | l2 norm = {:>.12}",
                step + 1,
                time,
                discrete_l2_norm(&psi, grid.dx)
            );
        }
    }

    let final_norm = discrete_l2_norm(&psi, grid.dx);
    let final_density = max_density(&psi);

    println!();
    println!("Invariant Diagnostics");
    println!("---------------------");
    println!("Initial l2 norm          = {:.12}", initial_norm);
    println!("Final l2 norm            = {:.12}", final_norm);
    println!(
        "Absolute norm change     = {:.6e}",
        (final_norm - initial_norm).abs()
    );
    println!("Initial max |psi|^2      = {:.12}", initial_density);
    println!("Final max |psi|^2        = {:.12}", final_density);
    println!();

    println!("Sample Wave Function Values");
    println!("---------------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "Index", "x", "Re(psi)", "Im(psi)", "|psi|^2"
    );

    for j in (0..grid.n).step_by(8) {
        println!(
            "{:>8} {:>14.6} {:>18.10} {:>18.10} {:>18.10}",
            j,
            grid.point(j),
            psi[j].re,
            psi[j].im,
            psi[j].abs2()
        );
    }
}
```

Program 20.1.2 demonstrates how the same algebraic time-stepping framework may produce fundamentally different physical behavior depending on the structure of the spatial operator and the preserved invariant. Whereas diffusive equations generate dissipative smoothing and decay of energy-like quantities, the Schrödinger equation produces unitary wave evolution in which the total probability norm remains constant. The Crank–Nicolson Cayley transform successfully reproduces this invariant structure at the discrete level by preserving the Hermitian symmetry of the Hamiltonian operator.

The numerical results verify this behavior clearly. The discrete (\\ell^2) norm remains constant to machine precision throughout the computation, confirming the unitary nature of the update. At the same time, the maximum probability density decreases as the wave packet spreads across the periodic domain. This reduction in peak amplitude does not represent dissipation; rather, it reflects dispersive redistribution of probability density while preserving total norm. The distinction illustrates why wave-like equations should not be treated numerically as if they were diffusive systems.

The modular organization of the code also reflects the broader computational principles discussed in this section. The Hamiltonian assembly, boundary-condition treatment, invariant diagnostics, and time integrator are separated into independent components, making the framework extensible to more sophisticated quantum evolution models. Higher-order spatial discretizations, sparse Hermitian linear algebra, spectral methods, split-operator techniques, adaptive timestepping, and multidimensional quantum systems can all be incorporated within the same structural framework. Consequently, the program serves as a foundation for more advanced invariant-preserving PDE solvers later in the chapter.

# 20.2. Flux-Conservative Initial Value Problems

Flux-conservative initial-value problems describe the evolution of quantities whose change over any region is governed by what enters and leaves through the boundary of that region. The fundamental unknown may be a scalar, such as density or water height, or a vector of conserved variables, such as mass, momentum, and energy. The defining feature is not merely the presence of a spatial derivative, but the fact that the equation can be written as a divergence of a flux. This conservative structure is essential when the total amount of a physical quantity must be preserved up to boundary fluxes and sources. It is also the reason finite-volume methods occupy a central position in the numerical treatment of transport, shallow-water flow, compressible gas dynamics, and many other hyperbolic systems.

## 20.2.1. Integral Balance and Finite-Volume Formulation

Consider the one-dimensional scalar conservation law,

$$u_t+f(u)_x=0 \tag{20.2.1}$$

where $u(x,t)$ is the conserved scalar and $f(u)$ is its physical flux. Although (20.2.1) is written in differential form, the correct numerical starting point is the integral balance law. Let the computational mesh consist of cells:

\begin{equation}
\begin{aligned}
I_j &= [x_{j-1/2},x_{j+1/2}], \\
\Delta x &= x_{j+1/2}-x_{j-1/2}
\end{aligned}
\tag{20.2.2}
\end{equation}

Integrating (20.2.1) over $I_j$ gives:

\begin{equation}
\frac{d}{dt}
\int_{x_{j-1/2}}^{x_{j+1/2}} u(x,t)\,dx
+
\int_{x_{j-1/2}}^{x_{j+1/2}} f(u)_x\,dx
= 0
\tag{20.2.3}
\end{equation}

Using the fundamental theorem of calculus,

\begin{equation}
\frac{d}{dt}
\int_{x_{j-1/2}}^{x_{j+1/2}} u(x,t)\,dx
=
f\!\left(u(x_{j-1/2},t)\right)
-
f\!\left(u(x_{j+1/2},t)\right)
\tag{20.2.4}
\end{equation}

Define the cell average:

\begin{equation}
\bar{u}_j(t)
=
\frac{1}{\Delta x}
\int_{x_{j-1/2}}^{x_{j+1/2}} u(x,t)\,dx
\tag{20.2.5}
\end{equation}

Then the exact cell-average balance is:

\begin{equation}
\frac{d\bar{u}_j}{dt}
=
-\frac{1}{\Delta x}
\left(
f_{j+1/2} - f_{j-1/2}
\right)
\tag{20.2.6}
\end{equation}

where,

\begin{equation}
\begin{aligned}
f_{j+1/2} &= f\!\left(u(x_{j+1/2},t)\right),\\
f_{j-1/2} &= f\!\left(u(x_{j-1/2},t)\right)
\end{aligned}
\tag{20.2.7}
\end{equation}

Equation (20.2.6) is exact for sufficiently smooth solutions. In numerical computation, however, the interface values are not usually known exactly. Moreover, for nonlinear conservation laws, discontinuities may form even from smooth initial data, so pointwise values at interfaces may be ambiguous. The finite-volume method therefore replaces the physical interface fluxes by numerical fluxes,

\begin{equation}
\widehat{f}_{j+1/2}
=
\widehat{f}\!\left(\bar{u}_j,\bar{u}_{j+1}\right)
\tag{20.2.8}
\end{equation}

or, in higher-order methods, by fluxes depending on reconstructed left and right interface states. The first-order semi-discrete finite-volume method is:

\begin{equation}
\frac{d\bar{u}_j}{dt}
=
-\frac{1}{\Delta x}
\left(
\widehat{f}_{j+1/2}
-
\widehat{f}_{j-1/2}
\right)
\tag{20.2.9}
\end{equation}

A fully discrete explicit Euler step gives:

\begin{equation}
\bar{u}_j^{\,n+1}
=
\bar{u}_j^{\,n}
-
\frac{\Delta t}{\Delta x}
\left(
\widehat{f}_{j+1/2}^{\,n}
-
\widehat{f}_{j-1/2}^{\,n}
\right)
\tag{20.2.10}
\end{equation}

The numerical flux must satisfy the consistency condition,

\begin{equation}
\widehat{f}(u,u)=f(u)
\tag{20.2.11}
\end{equation}

which ensures that the finite-volume scheme reduces to the correct physical flux when the solution is locally constant. This condition is necessary for convergence to the correct conservation law, but it is not sufficient by itself. Stability, monotonicity, entropy consistency, and appropriate numerical dissipation also matter, especially near shocks or steep gradients.

The most important algebraic property of (20.2.10) is conservation. Multiplying by (\\Delta x) and summing over cells (j=1,\\dots,N) yields

\begin{equation}
\sum_{j=1}^{N}\bar{u}_j^{\,n+1}\Delta x
=
\sum_{j=1}^{N}\bar{u}_j^{\,n}\Delta x
-
\Delta t
\sum_{j=1}^{N}
\left(
\widehat{f}_{j+1/2}^{\,n}
-
\widehat{f}_{j-1/2}^{\,n}
\right)
\tag{20.2.12}
\end{equation}

The interior fluxes cancel telescopically, leaving,

\begin{equation}
\sum_{j=1}^{N}\bar{u}_j^{\,n+1}\Delta x
=
\sum_{j=1}^{N}\bar{u}_j^{\,n}\Delta x
-
\Delta t
\left(
\widehat{f}_{N+1/2}^{\,n}
-
\widehat{f}_{1/2}^{\,n}
\right)
\tag{20.2.13}
\end{equation}

Thus, in the absence of net boundary flux, the discrete total mass is preserved exactly. This is the defining strength of finite-volume methods. The method conserves because the flux leaving one cell is exactly the flux entering its neighbour. In physical applications, this local conservation is often more important than pointwise accuracy, since a scheme that fails to preserve the correct integral balance can produce globally misleading results over long time intervals.

### Rust Implementation

Following the discussion in Subsection 20.2.1 on integral balance laws and conservative finite-volume discretization, Program 20.2.1 provides a practical implementation of a first-order conservative finite-volume solver for a scalar conservation law on a periodic domain. The program demonstrates how the continuous balance relation expressed in equations (20.2.3)–(20.2.7) is converted into a discrete conservative update through numerical interface fluxes as introduced in equations (20.2.8)–(20.2.10). The implementation employs a Rusanov numerical flux together with explicit Euler time integration to evolve cell averages while preserving the fundamental conservative structure of the PDE. Most importantly, the program illustrates the telescoping flux cancellation described in equations (20.2.12)–(20.2.13), which guarantees exact conservation of the discrete total mass up to roundoff error. The resulting framework demonstrates why finite-volume methods are central in hyperbolic conservation laws and transport-dominated simulations where local and global conservation properties are physically essential.

At the core of the implementation is the `Grid1D` structure, which defines the one-dimensional finite-volume mesh used to discretize the computational domain. The structure stores the number of cells, domain boundaries, and uniform mesh spacing $\Delta x$, while the `cell_center` function computes the geometric center of each control volume. This organization reflects the finite-volume interpretation of equation (20.2.5), where the numerical unknowns represent cell averages rather than pointwise nodal values. The helper functions `periodic_left` and `periodic_right` implement periodic boundary conditions by wrapping neighboring cell indices across the domain boundaries, thereby ensuring consistent flux exchange between adjacent control volumes.

The physical conservation law is represented through the functions `physical_flux` and `wave_speed`. In the present example, the scalar conservation law is chosen to be the inviscid Burgers equation, whose nonlinear flux takes the form $f(u)=u^2/2$. The characteristic wave speed is therefore determined by the derivative of the flux function, which for Burgers equation reduces to $f'(u)=u$. The function `wave_speed` computes the magnitude of this characteristic speed and is later used in the construction of the numerical flux and the CFL timestep restriction.

The function `rusanov_flux` implements the local Lax–Friedrichs, or Rusanov, numerical flux corresponding to equation (20.2.19). The method combines a centered average of the physical fluxes with a dissipative correction proportional to the jump in the conserved variable across the interface. The dissipation coefficient is determined by the maximum local characteristic speed, consistent with the wave-speed estimate introduced in equation (20.2.20). This additional dissipation stabilizes the scheme and allows robust treatment of steep gradients and emerging discontinuities while maintaining the conservative finite-volume structure.

The conservative finite-volume update itself is implemented through the function `finite_volume_step`, which corresponds directly to the discrete update formulas in equations (20.2.9)–(20.2.10). The function first evaluates numerical fluxes at all cell interfaces and then updates the cell averages by subtracting the difference between outgoing and incoming interface fluxes. Because neighboring cells share the same interface flux with opposite signs, the update preserves discrete conservation through telescoping cancellation. This local flux balance is the defining structural property of conservative finite-volume methods and is the reason total mass remains conserved throughout the computation.

Several auxiliary diagnostic functions are included to monitor the physical and numerical properties of the evolving solution. The function `total_mass` computes the discrete integral of the conserved quantity over the domain and verifies the conservative property predicted by equations (20.2.12)–(20.2.13). The functions `min_value` and `max_value` monitor the range of the numerical solution, while `max_characteristic_speed` computes the maximum wave speed used to determine the timestep through the CFL stability condition. This reflects the geometric interpretation of equation (20.2.21), where the numerical domain of dependence must remain compatible with physical wave propagation.

The `main` function serves as a complete demonstration of conservative finite-volume evolution for a nonlinear hyperbolic PDE. The computation begins by constructing a periodic mesh and initializing a discontinuous step profile that evolves under Burgers dynamics. During each timestep, the CFL condition determines a stable timestep based on the maximum characteristic speed, after which the finite-volume update advances the cell averages using the Rusanov numerical flux. Diagnostic output monitors the evolution of the solution and confirms that the total discrete mass remains preserved to machine precision throughout the simulation. The resulting computation demonstrates the central principle of finite-volume methods: conservation is enforced directly through balanced interface flux exchange between neighboring control volumes.

```rust
// Program 20.2.1. Conservative Finite-Volume Solver for a Scalar Conservation Law
//
// Problem statement:
// Solve the one-dimensional scalar conservation law
//
//     u_t + f(u)_x = 0
//
// on a periodic domain using a first-order finite-volume method. The code
// implements the numerical flux formulation in equations (20.2.8)-(20.2.10)
// and verifies the discrete conservation property described in equations
// (20.2.12)-(20.2.13).

#[derive(Clone)]
struct Grid1D {
    n: usize,
    xmin: f64,
    xmax: f64,
    dx: f64,
}

impl Grid1D {
    fn new(n: usize, xmin: f64, xmax: f64) -> Self {
        assert!(n >= 4, "The grid must contain at least four cells.");
        assert!(xmax > xmin, "The domain length must be positive.");

        let dx = (xmax - xmin) / n as f64;

        Self { n, xmin, xmax, dx }
    }

    fn cell_center(&self, j: usize) -> f64 {
        self.xmin + (j as f64 + 0.5) * self.dx
    }
}

fn periodic_left(j: usize, n: usize) -> usize {
    if j == 0 { n - 1 } else { j - 1 }
}

fn periodic_right(j: usize, n: usize) -> usize {
    if j + 1 == n { 0 } else { j + 1 }
}

// Burgers flux:
//
//     f(u) = u^2 / 2
//
fn physical_flux(u: f64) -> f64 {
    0.5 * u * u
}

// For Burgers equation, the characteristic speed is:
//
//     f'(u) = u
//
fn wave_speed(u: f64) -> f64 {
    u.abs()
}

// Local Lax-Friedrichs, or Rusanov, numerical flux.
fn rusanov_flux(u_left: f64, u_right: f64) -> f64 {
    let f_left = physical_flux(u_left);
    let f_right = physical_flux(u_right);

    let alpha = wave_speed(u_left).max(wave_speed(u_right));

    0.5 * (f_left + f_right) - 0.5 * alpha * (u_right - u_left)
}

fn initial_condition(x: f64) -> f64 {
    if x < 0.5 {
        1.0
    } else {
        0.2
    }
}

fn total_mass(u: &[f64], dx: f64) -> f64 {
    u.iter().sum::<f64>() * dx
}

fn min_value(u: &[f64]) -> f64 {
    u.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_value(u: &[f64]) -> f64 {
    u.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn max_characteristic_speed(u: &[f64]) -> f64 {
    u.iter()
        .map(|value| wave_speed(*value))
        .fold(0.0_f64, f64::max)
}

fn finite_volume_step(u: &[f64], dt: f64, dx: f64) -> Vec<f64> {
    let n = u.len();

    // flux[j] represents the numerical flux at x_{j+1/2},
    // between cell j and cell j+1.
    let mut flux = vec![0.0_f64; n];

    for j in 0..n {
        let right = periodic_right(j, n);
        flux[j] = rusanov_flux(u[j], u[right]);
    }

    let mut next = vec![0.0_f64; n];

    for j in 0..n {
        let left_face = periodic_left(j, n);

        next[j] = u[j] - (dt / dx) * (flux[j] - flux[left_face]);
    }

    next
}

fn main() {
    let grid = Grid1D::new(200, 0.0, 1.0);

    let final_time: f64 = 0.20;
    let cfl: f64 = 0.80;

    let mut u: Vec<f64> = (0..grid.n)
        .map(|j| initial_condition(grid.cell_center(j)))
        .collect();

    let initial_mass = total_mass(&u, grid.dx);

    let mut time = 0.0_f64;
    let mut step = 0_usize;

    println!("Conservative Finite-Volume Solver for Burgers Equation");
    println!("======================================================");
    println!();

    println!("Grid Parameters");
    println!("---------------");
    println!("Number of cells           = {}", grid.n);
    println!("Domain                    = [{:.3}, {:.3}]", grid.xmin, grid.xmax);
    println!("Grid spacing dx           = {:.6}", grid.dx);
    println!();

    println!("Time Integration");
    println!("----------------");
    println!("Final time               = {:.6}", final_time);
    println!("CFL number               = {:.6}", cfl);
    println!();

    while time < final_time {
        let max_speed = max_characteristic_speed(&u);

        assert!(
            max_speed > 0.0,
            "The maximum characteristic speed must be positive."
        );

        let mut dt = cfl * grid.dx / max_speed;

        if time + dt > final_time {
            dt = final_time - time;
        }

        u = finite_volume_step(&u, dt, grid.dx);

        time += dt;
        step += 1;

        if step % 10 == 0 || time >= final_time {
            println!(
                "Step {:>5} | time = {:>.6} | min(u) = {:>.6} | max(u) = {:>.6}",
                step,
                time,
                min_value(&u),
                max_value(&u)
            );
        }
    }

    let final_mass = total_mass(&u, grid.dx);

    println!();
    println!("Conservation Diagnostics");
    println!("------------------------");
    println!("Initial mass             = {:.12}", initial_mass);
    println!("Final mass               = {:.12}", final_mass);
    println!(
        "Absolute mass change     = {:.6e}",
        (final_mass - initial_mass).abs()
    );
    println!();

    println!("Sample Cell Averages");
    println!("--------------------");
    println!("{:>8} {:>14} {:>18}", "Cell", "x", "u_bar");

    for j in (0..grid.n).step_by(20) {
        println!(
            "{:>8} {:>14.6} {:>18.10}",
            j,
            grid.cell_center(j),
            u[j]
        );
    }
}
```

Program 20.2.1 demonstrates the essential structure of conservative finite-volume discretization for scalar hyperbolic conservation laws. The implementation translates the integral balance formulation of equations (20.2.3)–(20.2.7) into a discrete numerical algorithm in which conservation is enforced directly through interface flux exchange. Unlike methods formulated purely through local differential approximations, the finite-volume update preserves the physically correct balance of conserved quantities across the computational mesh.

The numerical results clearly illustrate the conservative property described by equations (20.2.12)–(20.2.13). The total discrete mass remains constant to machine precision because the flux leaving one cell is exactly the flux entering the neighboring cell. This telescoping cancellation is independent of the complexity of the evolving solution and remains valid even when steep gradients or discontinuities develop. In practical applications such as shallow-water flow, compressible gas dynamics, and transport modeling, this local conservation property is often more important than high pointwise accuracy because global conservation errors can accumulate into physically meaningless long-time behavior.

The Rusanov numerical flux further illustrates the role of controlled numerical dissipation in hyperbolic computation. The dissipative correction stabilizes the scheme and enables robust shock capturing, although at the cost of some numerical smearing near discontinuities. This reflects a central trade-off in conservative numerical methods: stability and robustness often require carefully designed dissipation mechanisms that suppress nonphysical oscillations while preserving the correct integral structure of the PDE.

The modular organization of the code also mirrors the mathematical decomposition of modern conservative solvers. The physical flux, numerical flux, CFL timestep selection, boundary-condition treatment, and conservative update are implemented as distinct components that can be modified independently. This structure naturally supports extension to higher-order reconstruction methods, approximate Riemann solvers, strong-stability-preserving Runge–Kutta integrators, well-balanced source treatments, and multidimensional systems of conservation laws. Consequently, the program provides a foundational framework for more sophisticated conservative PDE solvers developed later in the chapter.

## 20.2.2. Systems, Hyperbolicity, and Numerical Fluxes

Many important conservation laws are systems rather than scalar equations. In one space dimension, a conservative system has the form:

\begin{equation}
\mathbf{u}_t+\mathbf{F}(\mathbf{u})_x=0,
\qquad
\mathbf{u}\in\mathbb{R}^m
\tag{20.2.14}
\end{equation}

where $\mathbf{u}$ is the vector of conserved variables and $\mathbf{F}(\mathbf{u})$ is the corresponding flux. The local propagation of waves is governed by the flux Jacobian,

\begin{equation}
A(\mathbf{u})
=
\frac{\partial \mathbf{F}}{\partial \mathbf{u}}
\tag{20.2.15}
\end{equation}

The system is hyperbolic if $A(\mathbf{u})$ has real eigenvalues and a complete set of eigenvectors for all physically admissible states. The eigenvalues of $A(\mathbf{u})$ are the characteristic speeds. They determine the directions and speeds at which information travels through the domain. This spectral structure is central to boundary conditions, numerical fluxes, stability restrictions, and the design of approximate Riemann solvers.

For a semi-discrete finite-volume method applied to (20.2.14), the update is:

\begin{equation}
\frac{d\bar{\mathbf{U}}_j}{dt}
=
-\frac{1}{\Delta x}
\left(
\widehat{\mathbf{F}}_{j+1/2}
-
\widehat{\mathbf{F}}_{j-1/2}
\right)
\tag{20.2.16}
\end{equation}

Here $\widehat{\mathbf{F}}_{j+1/2}$ is a numerical flux depending on the data from the left and right sides of the interface. In a first-order method,

\begin{equation}
\widehat{\mathbf{F}}_{j+1/2}
=
\widehat{\mathbf{F}}\!\left(\bar{\mathbf{U}}_j,\bar{\mathbf{U}}_{j+1}\right)
\tag{20.2.17}
\end{equation}

In a higher-order method, the numerical flux is evaluated using reconstructed interface states,

\begin{equation}
\widehat{\mathbf{F}}_{j+1/2}
=
\widehat{\mathbf{F}}\!\left(
\mathbf{U}_{j+1/2}^{-},
\mathbf{U}_{j+1/2}^{+}
\right)
\tag{20.2.18}
\end{equation}

where $\mathbf{U}_{j+1/2}^{-}$ *is the state reconstructed from the left cell and* $\mathbf{U}_{j+1/2}^{+}$ is the state reconstructed from the right cell.

A simple and robust approximate Riemann flux is the local Lax-Friedrichs, or Rusanov, flux,

\begin{equation}
\widehat{\mathbf{F}}^{\mathrm{Rus}}(\mathbf{U}_L,\mathbf{U}_R)
=
\frac{1}{2}
\left[
\mathbf{F}(\mathbf{U}_L)+\mathbf{F}(\mathbf{U}_R)
\right]
-
\frac{a_{j+1/2}}{2}
\left(
\mathbf{U}_R-\mathbf{U}_L
\right)
\tag{20.2.19}
\end{equation}

Here $a_{j+1/2}$ is an upper bound on the local wave speed, usually chosen so that,

\begin{equation}
a_{j+1/2}
\ge
\max\left\{
\rho\!\left(A(\mathbf{U}_L)\right),
\rho\!\left(A(\mathbf{U}_R)\right)
\right\}
\tag{20.2.20}
\end{equation}

where $\rho(\cdot)$ denotes the spectral radius. The first term in (20.2.19) is a centered flux average, while the second term supplies numerical dissipation proportional to the jump in the conserved state. This dissipation stabilizes the method and enables robust shock capturing, although it may smear discontinuities more than less dissipative approximate Riemann solvers.

For explicit time stepping, the time step must satisfy a Courant-Friedrichs-Lewy condition. A typical form is:

\begin{equation}
\Delta t
\le
\mathrm{CFL}\,
\frac{\Delta x}
{\displaystyle \max_j \rho\!\left(A(\bar{\mathbf{U}}_j)\right)}
\tag{20.2.21}
\end{equation}

The CFL condition expresses a geometric compatibility between physical wave propagation and the numerical domain of dependence. Information should not travel farther in one time step than the numerical stencil can represent. If this condition is violated, the discrete update cannot correctly transmit wave information and instability usually results.

When source terms are present,

\begin{equation}
\mathbf{u}_t+\mathbf{F}(\mathbf{u})_x
=
\mathbf{S}(\mathbf{u},x,t)
\tag{20.2.22}
\end{equation}

the semi-discrete conservative update becomes:

\begin{equation}
\frac{d\bar{\mathbf{U}}_j}{dt}
=
-\frac{1}{\Delta x}
\left(
\widehat{\mathbf{F}}_{j+1/2}
-
\widehat{\mathbf{F}}_{j-1/2}
\right)
+
\mathbf{S}_j
\tag{20.2.23}
\end{equation}

The source term must be discretized carefully. In balance laws such as the shallow-water equations over variable bathymetry, the flux gradient and source term may cancel exactly for important steady states. A method that does not preserve this balance can create artificial waves from a physically steady solution. This requirement motivates well-balanced schemes, positivity-preserving limiters, and entropy-stable flux-source discretizations in modern shallow-water computations (Abgrall and Liu, 2024; Del Grosso et al., 2024; Ersing, Goldberg and Winters, 2025).

## 20.2.3. Classical Linear Advection Schemes and Stability

Classical finite-difference schemes for the linear advection equation remain pedagogically important because they expose the trade-off among stability, numerical diffusion, and phase accuracy. Consider,

$$u_t+a u_x=0,\qquad a=\text{constant} \tag{20.2.24}$$

on a uniform grid $x_j=j\Delta x$, $t^n=n\Delta t$. Define the Courant number,

$$\nu=\frac{a\Delta t}{\Delta x} \tag{20.2.25}$$

The forward-time centered-space scheme is:

\begin{equation}
u_j^{n+1}
=
u_j^n
-
\frac{\nu}{2}
\left(
u_{j+1}^n-u_{j-1}^n
\right)
\tag{20.2.26}
\end{equation}

Although this approximation appears natural because it uses a centered difference for $u_x$, it is unstable for advection. To see this, insert the Fourier mode:

\begin{equation}
u_j^n
=
\xi^n e^{ij\theta},
\qquad
\theta=k\Delta x
\tag{20.2.27}
\end{equation}

The amplification factor is:

$$
\xi(\theta)

1-i\nu\sin\theta \tag{20.2.28}
$$

Therefore,

\begin{equation}
|\xi(\theta)|^2
=
1+\nu^2\sin^2\theta
\tag{20.2.29}
\end{equation}

For any nonzero mode with $\sin\theta\ne 0$, one has $|\xi(\theta)|>1$. Thus, the centered forward-time scheme is unconditionally unstable for linear advection. The failure is instructive: a nondissipative centered spatial derivative combined with forward Euler time stepping does not provide the correct stability mechanism for wave propagation.

The Lax-Friedrichs scheme replaces the value $u_j^n$ in the time update by a local average,

\begin{equation}
u_j^{n+1}
=
\frac{1}{2}
\left(
u_{j+1}^n+u_{j-1}^n
\right)
-
\frac{\nu}{2}
\left(
u_{j+1}^n-u_{j-1}^n
\right)
\tag{20.2.30}
\end{equation}

The corresponding amplification factor is:

$$\xi(\theta) = \cos\theta - i\nu\sin\theta \tag{20.2.31}$$

Hence,

$$|\xi(\theta)|^2 = \cos^2\theta+\nu^2\sin^2\theta \tag{20.2.32}$$

Stability requires,

$$|\nu|\le 1 \tag{20.2.33}$$

The price of stability is numerical viscosity. The local averaging in (20.2.30) damps short waves and stabilizes the update, but it also smears sharp fronts. A modified-equation calculation shows that Lax-Friedrichs behaves like an approximation to the advection equation with an added artificial diffusion term. This explains why the method is robust but diffusive.

For $a>0$, the first-order upwind scheme is:

$$u_j^{n+1} = (1-\nu)u_j^n+\nu u_{j-1}^n \tag{20.2.34}$$

Equivalently,

$$u_j^{n+1} = u_j^n-\nu\left(u_j^n-u_{j-1}^n\right) \tag{20.2.35}$$

The amplification factor is:

$$\xi(\theta) = 1-\nu\left(1-e^{-i\theta}\right) \tag{20.2.36}$$

For $0\le\nu\le 1$, the method is stable. Upwinding is only first-order accurate, but it respects the direction of information propagation. For transport-dominated problems, this physical alignment is often more important than formal symmetry. In particular, upwind discretizations are much more reliable near steep gradients than centered schemes because they introduce dissipation in the direction dictated by the characteristic speed.

Second-order accuracy can be obtained by the Lax-Wendroff scheme,

\begin{equation}
u_j^{n+1}
=
u_j^n
-
\frac{\nu}{2}
\left(
u_{j+1}^n-u_{j-1}^n
\right)
+
\frac{\nu^2}{2}
\left(
u_{j+1}^n-2u_j^n+u_{j-1}^n
\right)
\tag{20.2.37}
\end{equation}

The first correction term is centered advection, while the second term is a second-order time-correction derived from the PDE. Compared with Lax-Friedrichs, Lax-Wendroff is less diffusive and more accurate for smooth waves. However, its leading error is dispersive rather than strongly dissipative. Near shocks or discontinuities, this can generate nonphysical oscillations. This behaviour illustrates a recurring lesson in hyperbolic computation: higher formal order is not automatically better unless the method also controls oscillations and respects the entropy structure of the PDE.

Historically, leapfrog schemes were also important because of their low dissipation. However, their two-level-in-time structure introduces a parasitic computational mode. In modern software design, especially in modular method-of-lines frameworks, it is often more convenient to discretize space first and then apply a strong-stability-preserving Runge-Kutta method to the resulting ODE system. This approach separates the spatial and temporal components of the algorithm and makes the implementation easier to test and extend.

## 20.2.4. High-Resolution Conservative Solvers and Modern Structure Preservation

The method-of-lines formulation for a high-resolution conservative scheme may be written as:

\begin{equation}
\frac{d\bar{\mathbf{U}}_j}{dt}
=
-\frac{1}{\Delta x}
\left[
\widehat{\mathbf{F}}_{j+1/2}
\!\left(
\mathbf{U}_{j+1/2}^{-},
\mathbf{U}_{j+1/2}^{+}
\right)
-
\widehat{\mathbf{F}}_{j-1/2}
\!\left(
\mathbf{U}_{j-1/2}^{-},
\mathbf{U}_{j-1/2}^{+}
\right)
\right]
\tag{20.2.38}
\end{equation}

This formula separates the conservative solver into three mathematical operations. First, a reconstruction procedure constructs left and right states at each interface. Second, a numerical flux or approximate Riemann solver resolves the interface interaction. Third, a time integrator advances the resulting ODE system. Source terms, limiters, artificial viscosity, or positivity corrections may be added as separate components when required.

The reconstruction layer determines spatial accuracy. A first-order method takes the cell average itself as the interface state. A MUSCL-type method reconstructs a piecewise linear profile in each cell, usually with a slope limiter to prevent oscillations. WENO methods use several candidate stencils and nonlinear weights to obtain high-order accuracy in smooth regions while avoiding oscillatory interpolation near discontinuities. Discontinuous Galerkin methods represent the solution by local polynomials inside each cell and communicate between cells through numerical fluxes. In all cases, the conservative update is retained at the cell or element level.

The computational complexity remains favourable. On a one-dimensional grid with $N$ cells, a first-order explicit finite-volume step requires $O(N)$ work and $O(N)$ memory. MUSCL-type second-order reconstruction also remains $O(N)$. A WENO reconstruction using a $(2r-1)$-point stencil is still linear in $N$, but the constant factor is larger because each interface requires multiple candidate polynomials, smoothness indicators, and nonlinear weights. Discontinuous Galerkin methods with polynomial degree $p$ increase both arithmetic cost and storage per cell, but their data access is highly local. This locality is valuable for cache efficiency, SIMD execution, threading, and strongly typed Rust implementations.

Recent developments in conservative schemes emphasize structure preservation rather than accuracy alone. For hyperbolic systems, entropy stability is a central concern. A conservative scheme should not merely avoid numerical blow-up; it should approximate the entropy inequality that selects the physically relevant weak solution. Recent finite-difference schemes combine high-order accuracy with entropy-stable dissipation, while recent discontinuous Galerkin methods combine entropy stability with oscillation suppression, positivity preservation, and stiffness-aware time integration (Zhang et al., 2023; Liu, Lu and Shu, 2024; Liu et al., 2025). A particularly important modern direction is the use of machine learning only within constrained numerical structures. For example, learned WENO weights can be embedded inside an entropy-stable TeCNO framework so that learning improves adaptivity while the scheme still enforces mathematically necessary sign and entropy properties (Charles and Ray, 2025).

Balance laws introduce additional requirements. The two-dimensional shallow-water equations, for example, may be written as:

\begin{equation}
\partial_t
\begin{bmatrix}
h\\
hu\\
hv
\end{bmatrix}
+
\partial_x
\begin{bmatrix}
hu\\
hu^2+\dfrac{1}{2}gh^2\\
huv
\end{bmatrix}
+
\partial_y
\begin{bmatrix}
hv\\
huv\\
hv^2+\dfrac{1}{2}gh^2
\end{bmatrix}
=
\begin{bmatrix}
0\\
-gh\,b_x\\
-gh\,b_y
\end{bmatrix}
+
\mathbf{S}_f
\tag{20.2.39}
\end{equation}

Here $h$ is water depth, $u$ and $v$ are depth-averaged velocity components, $b(x,y)$ is bathymetry, $g$ is gravitational acceleration, and $\mathbf{S}_f$ represents frictional terms. The topographic source term is not a minor correction. It must balance the pressure-gradient part of the flux for steady states such as still water over uneven bathymetry. If the flux gradient and source term are discretized inconsistently, the numerical method manufactures false waves over topography.

For this reason, modern shallow-water solvers often require several properties simultaneously: conservation, well-balancedness, positivity preservation of water depth, entropy stability, and robust treatment of wetting and drying fronts. Recent schemes on unstructured and curvilinear meshes have been designed to preserve still-water or moving-water equilibria while maintaining positivity and entropy stability (Abgrall and Liu, 2024; Del Grosso et al., 2024; Ersing, Goldberg and Winters, 2025). These properties are not optional in flood inundation and dam-break prediction. High-resolution models may be coupled bidirectionally with coarser hydrologic models, and adaptive or multigrid strategies may be used so that fine spatial resolution is concentrated in flood-prone zones. In such applications, local conservation, accurate wave propagation, and positivity of water depth are essential for physically meaningful simulations (Shen et al., 2024; Del Grosso et al., 2024; Ersing, Goldberg and Winters, 2025).

For implementation in Rust, the mathematical decomposition in (20.2.38) suggests a clean software architecture. The conserved state, primitive variables, reconstruction operator, flux function, source discretization, limiter, and time integrator should be represented as distinct components. This design avoids mixing mathematically different operations inside a single opaque update. It also enables systematic benchmarking: one can replace the Rusanov flux by another approximate Riemann solver, replace a MUSCL reconstruction by a WENO reconstruction, or change the time integrator without rewriting the entire solver. Such modularity is not only a software convenience; it reflects the structure of the numerical method.

The main lesson is that conservative form is not optional when the modeled quantity is conserved. A discretization written only in primitive variables may be convenient for some local operations, but if it does not preserve the interface flux balance, it can accumulate global conservation errors that dominate the physical interpretation. Modern well-balanced methods may use primitive variables during reconstruction or source balancing, but the underlying update for cell averages remains conservative. This is the essential principle that connects the integral form of the PDE, the finite-volume update, and reliable long-time computation (Abgrall and Liu, 2024).

### Rust Implementation

Following the discussion in Subsections 20.2.2–20.2.4 on hyperbolic systems, approximate Riemann solvers, CFL stability restrictions, and high-resolution conservative discretization, Program 20.2.2 provides a practical implementation of a MUSCL-type finite-volume solver for the one-dimensional shallow-water equations. The program combines the conservative system formulation of equations (20.2.14)–(20.2.23) with the high-resolution method-of-lines structure described in equation (20.2.38). The implementation employs piecewise linear MUSCL reconstruction with a minmod slope limiter, a Rusanov approximate Riemann flux, and strong-stability-preserving Runge–Kutta time integration. This modular organization reflects the modern computational decomposition of conservative hyperbolic solvers into reconstruction, interface flux evaluation, and temporal integration components. The resulting framework demonstrates how conservative finite-volume methods preserve physically important quantities such as mass and momentum while simultaneously maintaining robustness near steep gradients and discontinuities.

At the core of the implementation is the `State` structure, which represents the vector of conserved variables for the shallow-water system. The conserved state consists of the water depth $h$ and the momentum $hu$, corresponding to the vector form introduced in equation (20.2.14). The structure also implements several important physical operations. The function `velocity` computes the primitive velocity variable from the conserved quantities, while the function `flux` evaluates the conservative flux vector associated with the shallow-water equations. The function `wave_speed` computes an upper bound on the local characteristic speed, corresponding to the spectral-radius estimate appearing in equation (20.2.20). This quantity combines the fluid velocity with the gravity-wave speed and is later used in the Rusanov numerical flux and CFL timestep restriction.

The spatial discretization is organized through the `Grid1D` structure, which stores the computational mesh and geometric spacing. The helper functions `periodic_left` and `periodic_right` implement periodic boundary conditions by wrapping neighboring cell indices across the domain boundaries. This ensures that numerical fluxes remain consistent at the periodic interfaces and preserves the conservative structure of the finite-volume update throughout the computation.

The reconstruction layer is implemented through the functions `minmod`, `limited_slope`, and `reconstruct_muscl`. The function `minmod` acts as a nonlinear slope limiter that suppresses oscillatory reconstructions near steep gradients by selecting the smallest admissible slope consistent with monotonicity. The function `limited_slope` applies this limiter componentwise to the conserved variables, while `reconstruct_muscl` constructs left and right interface states at each cell boundary according to the MUSCL methodology discussed after equation (20.2.38). This reconstruction procedure increases spatial accuracy beyond first order in smooth regions while preventing the nonphysical oscillations that commonly arise in high-order centered approximations near discontinuities.

The function `rusanov_flux` implements the local Lax–Friedrichs, or Rusanov, approximate Riemann solver introduced in equation (20.2.19). The numerical flux combines a centered average of the physical flux vectors with a dissipative correction proportional to the jump between the left and right interface states. The dissipation coefficient is chosen using the maximum local wave speed, consistent with equation (20.2.20). This additional dissipation stabilizes the numerical update and enables robust shock capturing, although it also introduces controlled smoothing near steep solution features.

The conservative spatial discretization itself is implemented through the function `finite_volume_rhs`, which corresponds directly to the semi-discrete method-of-lines formulation in equations (20.2.16) and (20.2.38). The function first reconstructs interface states, then evaluates numerical fluxes at each interface, and finally computes the conservative flux differences required for the finite-volume update. Because the same interface flux enters neighboring cells with opposite signs, the resulting discretization preserves conservation through telescoping cancellation of internal fluxes.

The time integration layer is implemented through the functions `forward_euler_step` and `ssp_rk2_step`. The forward Euler function performs a single explicit timestep using the finite-volume residual, while the SSP-RK2 routine constructs a second-order strong-stability-preserving Runge–Kutta update. This separation between spatial discretization and temporal integration reflects the method-of-lines philosophy discussed throughout Section 20.2, where the PDE is first converted into a system of ordinary differential equations and then advanced using a suitable time integrator. The CFL timestep restriction introduced in equation (20.2.21) is enforced dynamically using the maximum wave speed computed from the evolving solution.

Several auxiliary diagnostic functions are included to monitor the physical properties of the numerical solution. The functions `total_mass` and `total_momentum` compute the globally conserved quantities associated with the shallow-water system and verify preservation of the conservative structure. The functions `min_height` and `max_height` monitor positivity and wave evolution throughout the simulation. The positivity safeguard implemented through `enforce_positivity` prevents the water depth from becoming negative, which is an essential structural requirement in shallow-water computation.

The `main` function serves as a complete demonstration of a high-resolution conservative shallow-water solver. The program begins by constructing a periodic computational grid and initializing a discontinuous dam-break profile. During each timestep, the CFL condition determines a stable timestep using the maximum local characteristic speed, after which the SSP-RK2 integrator advances the solution using the MUSCL-Rusanov finite-volume formulation. Diagnostic output continuously monitors the evolution of the water depth and verifies that both total mass and total momentum remain conserved to machine precision. The resulting computation illustrates the modern structure-preserving philosophy emphasized throughout Subsection 20.2.4: conservative discretization, stable reconstruction, physically meaningful numerical dissipation, and modular algorithmic decomposition must all work together to produce reliable hyperbolic PDE solvers.

```rust
// Program 20.2.2. MUSCL-Rusanov Finite-Volume Solver for the Shallow-Water Equations
//
// Problem statement:
// Solve the one-dimensional shallow-water equations
//
//     h_t + (hu)_x = 0,
//     (hu)_t + (hu^2 + 1/2 g h^2)_x = 0,
//
// using a conservative finite-volume method with MUSCL reconstruction,
// a minmod slope limiter, the Rusanov numerical flux, and CFL-controlled
// explicit time stepping.
//
// The program demonstrates the modular conservative structure described
// in equations (20.2.16)-(20.2.21) and the high-resolution formulation
// in equation (20.2.38).

const G: f64 = 9.81;
const H_FLOOR: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug)]
struct State {
    h: f64,
    hu: f64,
}

impl State {
    fn new(h: f64, hu: f64) -> Self {
        Self { h, hu }
    }

    fn zero() -> Self {
        Self { h: 0.0, hu: 0.0 }
    }

    fn velocity(self) -> f64 {
        if self.h > H_FLOOR {
            self.hu / self.h
        } else {
            0.0
        }
    }

    fn flux(self) -> Self {
        let u = self.velocity();

        Self {
            h: self.hu,
            hu: self.hu * u + 0.5 * G * self.h * self.h,
        }
    }

    fn wave_speed(self) -> f64 {
        let u = self.velocity();
        let c = (G * self.h.max(H_FLOOR)).sqrt();

        u.abs() + c
    }
}

use std::ops::{Add, Mul, Sub};

impl Add for State {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            h: self.h + rhs.h,
            hu: self.hu + rhs.hu,
        }
    }
}

impl Sub for State {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            h: self.h - rhs.h,
            hu: self.hu - rhs.hu,
        }
    }
}

impl Mul<f64> for State {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self {
            h: self.h * rhs,
            hu: self.hu * rhs,
        }
    }
}

#[derive(Clone)]
struct Grid1D {
    n: usize,
    xmin: f64,
    xmax: f64,
    dx: f64,
}

impl Grid1D {
    fn new(n: usize, xmin: f64, xmax: f64) -> Self {
        assert!(n >= 4, "The grid must contain at least four cells.");
        assert!(xmax > xmin, "The domain length must be positive.");

        let dx = (xmax - xmin) / n as f64;

        Self { n, xmin, xmax, dx }
    }

    fn cell_center(&self, j: usize) -> f64 {
        self.xmin + (j as f64 + 0.5) * self.dx
    }
}

fn periodic_left(j: usize, n: usize) -> usize {
    if j == 0 { n - 1 } else { j - 1 }
}

fn periodic_right(j: usize, n: usize) -> usize {
    if j + 1 == n { 0 } else { j + 1 }
}

fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else if a.abs() < b.abs() {
        a
    } else {
        b
    }
}

fn limited_slope(left: State, center: State, right: State) -> State {
    let backward = center - left;
    let forward = right - center;

    State {
        h: minmod(backward.h, forward.h),
        hu: minmod(backward.hu, forward.hu),
    }
}

fn reconstruct_muscl(u: &[State]) -> Vec<(State, State)> {
    let n = u.len();
    let mut interface_states = vec![(State::zero(), State::zero()); n];

    let mut slopes = vec![State::zero(); n];

    for j in 0..n {
        let left = periodic_left(j, n);
        let right = periodic_right(j, n);

        slopes[j] = limited_slope(u[left], u[j], u[right]);
    }

    for j in 0..n {
        let right = periodic_right(j, n);

        let u_left = u[j] + slopes[j] * 0.5;
        let u_right = u[right] - slopes[right] * 0.5;

        interface_states[j] = (
            enforce_positivity(u_left),
            enforce_positivity(u_right),
        );
    }

    interface_states
}

fn enforce_positivity(state: State) -> State {
    if state.h >= H_FLOOR {
        state
    } else {
        State::new(H_FLOOR, 0.0)
    }
}

fn rusanov_flux(left: State, right: State) -> State {
    let f_left = left.flux();
    let f_right = right.flux();

    let alpha = left.wave_speed().max(right.wave_speed());

    (f_left + f_right) * 0.5 - (right - left) * (0.5 * alpha)
}

fn max_wave_speed(u: &[State]) -> f64 {
    u.iter()
        .map(|state| state.wave_speed())
        .fold(0.0_f64, f64::max)
}

fn finite_volume_rhs(u: &[State], dx: f64) -> Vec<State> {
    let n = u.len();

    let interface_states = reconstruct_muscl(u);
    let mut fluxes = vec![State::zero(); n];

    for j in 0..n {
        let (left_state, right_state) = interface_states[j];
        fluxes[j] = rusanov_flux(left_state, right_state);
    }

    let mut rhs = vec![State::zero(); n];

    for j in 0..n {
        let left_face = periodic_left(j, n);

        rhs[j] = (fluxes[left_face] - fluxes[j]) * (1.0 / dx);
    }

    rhs
}

fn forward_euler_step(u: &[State], dt: f64, dx: f64) -> Vec<State> {
    let rhs = finite_volume_rhs(u, dx);

    u.iter()
        .zip(rhs.iter())
        .map(|(state, rate)| enforce_positivity(*state + *rate * dt))
        .collect()
}

fn ssp_rk2_step(u: &[State], dt: f64, dx: f64) -> Vec<State> {
    let u_stage = forward_euler_step(u, dt, dx);
    let u_next_euler = forward_euler_step(&u_stage, dt, dx);

    u.iter()
        .zip(u_next_euler.iter())
        .map(|(old, updated)| enforce_positivity((*old * 0.5) + (*updated * 0.5)))
        .collect()
}

fn initial_condition(x: f64) -> State {
    let h = if x < 0.5 { 2.0 } else { 1.0 };
    let velocity = 0.0;

    State::new(h, h * velocity)
}

fn total_mass(u: &[State], dx: f64) -> f64 {
    u.iter().map(|state| state.h).sum::<f64>() * dx
}

fn total_momentum(u: &[State], dx: f64) -> f64 {
    u.iter().map(|state| state.hu).sum::<f64>() * dx
}

fn min_height(u: &[State]) -> f64 {
    u.iter().map(|state| state.h).fold(f64::INFINITY, f64::min)
}

fn max_height(u: &[State]) -> f64 {
    u.iter()
        .map(|state| state.h)
        .fold(f64::NEG_INFINITY, f64::max)
}

fn main() {
    let grid = Grid1D::new(200, 0.0, 1.0);

    let final_time: f64 = 0.05;
    let cfl: f64 = 0.45;

    let mut u: Vec<State> = (0..grid.n)
        .map(|j| initial_condition(grid.cell_center(j)))
        .collect();

    let initial_mass = total_mass(&u, grid.dx);
    let initial_momentum = total_momentum(&u, grid.dx);

    let mut time = 0.0_f64;
    let mut step = 0_usize;

    println!("MUSCL-Rusanov Solver for the Shallow-Water Equations");
    println!("====================================================");
    println!();

    println!("Grid Parameters");
    println!("---------------");
    println!("Number of cells           = {}", grid.n);
    println!("Domain                    = [{:.3}, {:.3}]", grid.xmin, grid.xmax);
    println!("Grid spacing dx           = {:.6}", grid.dx);
    println!();

    println!("Physical Parameters");
    println!("-------------------");
    println!("Gravity g                 = {:.6}", G);
    println!();

    println!("Time Integration");
    println!("----------------");
    println!("Final time               = {:.6}", final_time);
    println!("CFL number               = {:.6}", cfl);
    println!();

    while time < final_time {
        let speed = max_wave_speed(&u);

        assert!(
            speed > 0.0,
            "The maximum wave speed must be positive."
        );

        let mut dt = cfl * grid.dx / speed;

        if time + dt > final_time {
            dt = final_time - time;
        }

        u = ssp_rk2_step(&u, dt, grid.dx);

        time += dt;
        step += 1;

        if step % 10 == 0 || time >= final_time {
            println!(
                "Step {:>5} | time = {:>.6} | min(h) = {:>.6} | max(h) = {:>.6}",
                step,
                time,
                min_height(&u),
                max_height(&u)
            );
        }
    }

    let final_mass = total_mass(&u, grid.dx);
    let final_momentum = total_momentum(&u, grid.dx);

    println!();
    println!("Conservation Diagnostics");
    println!("------------------------");
    println!("Initial mass             = {:.12}", initial_mass);
    println!("Final mass               = {:.12}", final_mass);
    println!(
        "Absolute mass change     = {:.6e}",
        (final_mass - initial_mass).abs()
    );
    println!("Initial momentum         = {:.12}", initial_momentum);
    println!("Final momentum           = {:.12}", final_momentum);
    println!(
        "Absolute momentum change = {:.6e}",
        (final_momentum - initial_momentum).abs()
    );
    println!();

    println!("Sample Cell Averages");
    println!("--------------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "Cell", "x", "h", "hu", "u"
    );

    for j in (0..grid.n).step_by(20) {
        println!(
            "{:>8} {:>14.6} {:>18.10} {:>18.10} {:>18.10}",
            j,
            grid.cell_center(j),
            u[j].h,
            u[j].hu,
            u[j].velocity()
        );
    }
}
```

Program 20.2.2 demonstrates the essential computational structure of modern high-resolution conservative finite-volume methods for hyperbolic systems. The implementation combines MUSCL reconstruction, nonlinear limiting, approximate Riemann flux evaluation, and strong-stability-preserving time integration within a unified method-of-lines framework. This decomposition mirrors the mathematical structure of equation (20.2.38), where reconstruction, interface flux resolution, and time integration are treated as distinct but interacting numerical components.

The numerical results verify several important structural properties of conservative hyperbolic solvers. The discrete total mass and momentum remain conserved to machine precision because the finite-volume update preserves balanced interface flux exchange between neighboring cells. At the same time, the minmod limiter and Rusanov dissipation suppress spurious oscillations near the initial discontinuity while preserving positivity of the water depth. These features are essential in shallow-water computation because nonphysical negative depths or unstable oscillations can rapidly destroy the physical validity of a simulation.

The program also illustrates the trade-off between stability and numerical dissipation discussed throughout Sections 20.2.2 and 20.2.3. The Rusanov flux introduces sufficient dissipation to stabilize discontinuities and rapidly varying wave structures, although this comes at the cost of some smoothing of sharp fronts. The MUSCL reconstruction partially compensates for this effect by improving spatial resolution in smooth regions while retaining nonlinear limiting near steep gradients. This balance between accuracy, stability, and structure preservation is a central theme in modern hyperbolic computation.

The modular organization of the implementation naturally supports extension to more advanced conservative solvers. The reconstruction operator may be replaced by higher-order WENO procedures, the Rusanov flux may be replaced by more accurate approximate Riemann solvers, and the SSP-RK2 integrator may be extended to higher-order strong-stability-preserving Runge–Kutta methods. Additional components such as entropy-stable fluxes, well-balanced source discretizations, adaptive mesh refinement, wetting-and-drying algorithms, or multidimensional extensions can also be incorporated while preserving the same conservative method-of-lines architecture. Consequently, the program provides a foundation for more sophisticated structure-preserving shallow-water and hyperbolic PDE solvers later in the chapter.

# 20.3. Diffusive Initial Value Problems

Diffusive initial-value problems describe time-dependent processes in which gradients drive smoothing, relaxation, or redistribution. Unlike hyperbolic transport, where disturbances propagate with finite characteristic speeds, diffusion damps spatial variation continuously, with the shortest resolved wavelengths usually decaying fastest. This smoothing is physically stabilizing, but numerically it introduces stiffness. Explicit methods must take very small time steps on fine meshes, while implicit, semi-implicit, or exponential methods can step over the fastest diffusive scales when accuracy permits. Diffusive equations therefore form a natural bridge between spatial discretization, stability theory, sparse linear algebra, and structure-preserving time integration.

## 20.3.1. The Model Diffusion Equation and Explicit Stability

The model one-dimensional diffusion equation is:

\begin{equation}
u_t
=
\partial_x\!\left(D u_x\right),
\qquad
D\ge 0
\tag{20.3.1}
\end{equation}

When $D$ is constant, (20.3.1) reduces to the heat equation,

$$u_t=D u_{xx} \tag{20.3.2}$$

The qualitative effect of (20.3.2) is fundamentally different from the advection equation studied in Section 20.2. A Fourier mode $e^{ikx}$ evolves as $e^{-Dk^2t}e^{ikx}$, so high-frequency components decay faster than low-frequency components. This is the analytic origin of parabolic smoothing. It is also the origin of stiffness: after spatial discretization, the largest eigenvalues of the discrete diffusion operator scale like $\Delta x^{-2}$, so explicit time stepping becomes increasingly restrictive as the mesh is refined.

On a uniform grid $x_j=j\Delta x$, the standard forward-time centered-space discretization is:

\begin{equation}
\frac{u_j^{n+1}-u_j^n}{\Delta t}
=
D\,
\frac{
u_{j+1}^n-2u_j^n+u_{j-1}^n
}{\Delta x^2}
\tag{20.3.3}
\end{equation}

Equivalently,

$$u_j^{n+1} = u_j^n+\mu\left(u_{j+1}^n-2u_j^n+u_{j-1}^n\right),\qquad\mu=\frac{D\Delta t}{\Delta x^2} \tag{20.3.4}$$

To analyze stability, insert the Fourier mode,

$$u_j^n=\xi^n e^{ij\theta},\qquad\theta=k\Delta x \tag{20.3.5}$$

Substitution into (20.3.4) gives the amplification factor:

$$\xi(\theta) = 1-4\mu\sin^2\!\left(\frac{\theta}{2}\right) \tag{20.3.6}$$

Stability requires $|\xi(\theta)|\le 1$ for every Fourier mode. Since the maximum value of $\sin^2(\theta/2)$ is $1$, this gives:

$$0\le \mu\le \frac{1}{2} \tag{20.3.7}$$

Thus, the explicit time step must satisfy,

$$\Delta t\le \frac{\Delta x^2}{2D} \tag{20.3.8}$$

This is the parabolic time-step restriction. It is much more severe than the hyperbolic CFL restriction, because halving $\Delta x$ reduces the maximum stable explicit step by a factor of four. In one dimension this may still be acceptable for modest grids, but in high-resolution multidimensional simulations it can make purely explicit diffusion prohibitively expensive. This is why implicit and semi-implicit formulations are central to diffusion-dominated computation.

### Rust Implementation

Following the discussion in Subsection 20.3.1 on the diffusion equation, Fourier-mode damping, and the explicit stability restriction for parabolic problems, Program 20.3.1 provides a practical implementation of the forward-time centered-space (FTCS) method for the one-dimensional diffusion equation. The program directly implements the explicit discretization introduced in equations (20.3.3)–(20.3.4) and evaluates the stability parameter $\mu=D\Delta t/\Delta x^2$ governing the amplification factor in equation (20.3.6). Two computational cases are examined: one satisfying the stability condition of equation (20.3.7), and another deliberately violating it. By evolving a solution containing both smooth and near-grid-scale Fourier components, the program demonstrates the fundamentally different behavior of stable diffusive damping and unstable numerical growth. The implementation therefore illustrates the central numerical issue associated with explicit parabolic solvers: although diffusion physically smooths high-frequency modes, an unstable explicit discretization can instead amplify them catastrophically.

At the core of the implementation is the `Grid1D` structure, which defines the uniform spatial discretization used throughout the computation. The structure stores the number of grid points, the computational interval, and the mesh spacing $\Delta x$, while the `point` function computes the physical coordinate associated with each grid index. This organization reflects the uniform finite-difference mesh assumed in equations (20.3.3)–(20.3.4). The helper functions `periodic_left` and `periodic_right` implement periodic boundary conditions by wrapping neighboring indices across the domain boundaries, thereby ensuring that the discrete Laplacian remains translationally invariant and compatible with Fourier-mode analysis.

The function `initial_condition` constructs the initial solution profile used to investigate diffusive damping and numerical instability. The profile combines a smooth low-frequency sine wave with a small near-grid-scale perturbation. The low-frequency component represents slowly varying spatial structure, while the alternating perturbation excites the highest resolvable Fourier mode on the mesh. This high-frequency component is especially important because equation (20.3.6) predicts that the most restrictive amplification behavior occurs near the largest discrete wavenumbers. In the stable case, the perturbation is rapidly damped by diffusion, whereas in the unstable case it grows explosively because the explicit timestep violates the parabolic stability restriction.

The explicit FTCS update corresponding to equations (20.3.3)–(20.3.4) is implemented through the function `ftcs_diffusion_step`. For each grid point, the method applies the standard centered second-difference approximation to the Laplacian operator and advances the solution explicitly in time using the parameter $\mu=D\Delta t/\Delta x^2$. The resulting stencil couples each grid value only to its immediate neighbors, producing the classical three-point diffusion update. Because the method is explicit, no linear system solve is required, making the implementation computationally simple but conditionally stable.

Several auxiliary diagnostic functions are included to monitor important physical and numerical properties of the evolving solution. The function `discrete_mass` computes the discrete integral of the numerical solution over the periodic domain. Since the centered second-difference stencil is conservative in telescoping form, the total discrete mass remains nearly constant up to roundoff error. The function `l2_norm` measures the overall energy-like magnitude of the solution and is particularly useful for observing diffusive damping or numerical instability. The function `max_abs_value` tracks the maximum solution amplitude and provides a direct indicator of unstable mode growth in the supercritical timestep regime.

The function `run_case` serves as the primary driver for each numerical experiment. It computes the timestep associated with the requested stability parameter $\mu$, adjusts the timestep to match the prescribed final time, and reports the effective value of $\mu$ actually used in the computation. The function then repeatedly applies the FTCS update while monitoring diagnostic quantities including mass conservation, $L^2$-norm evolution, and maximum solution amplitude. By comparing the stable and unstable parameter regimes, the function demonstrates the sharp transition predicted by equations (20.3.6)–(20.3.8).

The `main` function coordinates the overall stability study by constructing the computational grid, specifying the diffusion coefficient and final integration time, and executing both stable and unstable FTCS simulations. The stable case uses a timestep satisfying $\mu\le 1/2$, while the unstable case deliberately violates this condition. The resulting output clearly illustrates the effect of the parabolic timestep restriction: in the stable regime, high-frequency components decay smoothly and the solution amplitude decreases over time, whereas in the unstable regime the near-grid-scale perturbation grows rapidly and eventually dominates the computation. This behavior provides direct computational confirmation of the Fourier stability analysis developed earlier in Subsection 20.3.1.

```rust
// Program 20.3.1. Explicit FTCS Diffusion Solver with Stability Monitoring
//
// Problem statement:
// Solve the one-dimensional constant-coefficient diffusion equation
//
//     u_t = D u_xx
//
// on a periodic domain using the forward-time centered-space method.
// The implementation follows equations (20.3.3)-(20.3.4), computes
// mu = D dt / dx^2, and demonstrates the stability restriction
// mu <= 1/2 from equations (20.3.7)-(20.3.8).

use std::f64::consts::PI;

#[derive(Clone)]
struct Grid1D {
    n: usize,
    xmin: f64,
    xmax: f64,
    dx: f64,
}

impl Grid1D {
    fn new(n: usize, xmin: f64, xmax: f64) -> Self {
        assert!(n >= 4, "The grid must contain at least four points.");
        assert!(xmax > xmin, "The domain length must be positive.");

        let dx = (xmax - xmin) / n as f64;

        Self { n, xmin, xmax, dx }
    }

    fn point(&self, j: usize) -> f64 {
        self.xmin + j as f64 * self.dx
    }
}

fn periodic_left(j: usize, n: usize) -> usize {
    if j == 0 { n - 1 } else { j - 1 }
}

fn periodic_right(j: usize, n: usize) -> usize {
    if j + 1 == n { 0 } else { j + 1 }
}

fn initial_condition(j: usize, x: f64) -> f64 {
    let low_frequency = (2.0 * PI * x).sin();

    // Near-grid-scale perturbation. This mode is deliberately small, but it
    // exposes the FTCS instability when mu > 1/2.
    let grid_scale_perturbation = if j % 2 == 0 { 0.02 } else { -0.02 };

    low_frequency + grid_scale_perturbation
}

fn ftcs_diffusion_step(u: &[f64], mu: f64) -> Vec<f64> {
    let n = u.len();
    let mut next = vec![0.0_f64; n];

    for j in 0..n {
        let left = periodic_left(j, n);
        let right = periodic_right(j, n);

        next[j] = u[j] + mu * (u[right] - 2.0 * u[j] + u[left]);
    }

    next
}

fn discrete_mass(u: &[f64], dx: f64) -> f64 {
    u.iter().sum::<f64>() * dx
}

fn l2_norm(u: &[f64], dx: f64) -> f64 {
    (u.iter().map(|value| value * value).sum::<f64>() * dx).sqrt()
}

fn max_abs_value(u: &[f64]) -> f64 {
    u.iter().map(|value| value.abs()).fold(0.0_f64, f64::max)
}

fn run_case(case_name: &str, grid: &Grid1D, diffusion: f64, requested_mu: f64, final_time: f64) {
    let nominal_dt = requested_mu * grid.dx * grid.dx / diffusion;
    let number_of_steps = (final_time / nominal_dt).ceil() as usize;
    let dt = final_time / number_of_steps as f64;
    let effective_mu = diffusion * dt / (grid.dx * grid.dx);

    let mut u: Vec<f64> = (0..grid.n)
        .map(|j| initial_condition(j, grid.point(j)))
        .collect();

    let initial_mass = discrete_mass(&u, grid.dx);
    let initial_l2 = l2_norm(&u, grid.dx);
    let initial_max = max_abs_value(&u);

    println!("{}", case_name);
    println!("{}", "-".repeat(case_name.len()));
    println!("Requested mu              = {:.6}", requested_mu);
    println!("Effective mu              = {:.6}", effective_mu);
    println!("Time step dt              = {:.8}", dt);
    println!("Number of steps           = {}", number_of_steps);

    if effective_mu <= 0.5 {
        println!("Stability status          = stable by mu <= 1/2");
    } else {
        println!("Stability status          = unstable by mu > 1/2");
    }

    for _ in 0..number_of_steps {
        u = ftcs_diffusion_step(&u, effective_mu);
    }

    let final_mass = discrete_mass(&u, grid.dx);
    let final_l2 = l2_norm(&u, grid.dx);
    let final_max = max_abs_value(&u);

    println!("Initial mass              = {:.12}", initial_mass);
    println!("Final mass                = {:.12}", final_mass);
    println!(
        "Absolute mass change      = {:.6e}",
        (final_mass - initial_mass).abs()
    );
    println!("Initial L2 norm           = {:.12}", initial_l2);
    println!("Final L2 norm             = {:.12}", final_l2);
    println!("Initial max |u|           = {:.12}", initial_max);
    println!("Final max |u|             = {:.12}", final_max);
    println!();
}

fn main() {
    let grid = Grid1D::new(128, 0.0, 1.0);

    let diffusion: f64 = 0.01;
    let final_time: f64 = 0.20;

    println!("Explicit FTCS Diffusion Solver with Stability Monitoring");
    println!("========================================================");
    println!();

    println!("Grid Parameters");
    println!("---------------");
    println!("Number of points          = {}", grid.n);
    println!("Domain                    = [{:.3}, {:.3}]", grid.xmin, grid.xmax);
    println!("Grid spacing dx           = {:.8}", grid.dx);
    println!();

    println!("Physical Parameters");
    println!("-------------------");
    println!("Diffusion coefficient D   = {:.6}", diffusion);
    println!("Final time                = {:.6}", final_time);
    println!();

    run_case("Stable FTCS Case", &grid, diffusion, 0.45, final_time);
    run_case("Unstable FTCS Case", &grid, diffusion, 0.75, final_time);

    println!("Interpretation");
    println!("--------------");
    println!("The stable case damps the high-frequency perturbation.");
    println!("The unstable case excites the near-grid-scale mode because mu > 1/2.");
}
```

Program 20.3.1 demonstrates the fundamental relationship between Fourier-mode stability analysis and explicit numerical discretization for parabolic partial differential equations. The FTCS method provides a simple and computationally inexpensive approximation to the diffusion equation, but its stability depends critically on the timestep parameter $\mu=D\Delta t/\Delta x^2$. The numerical experiments confirm the analytical prediction of equations (20.3.6)–(20.3.8): when $\mu\le 1/2$, the numerical solution behaves diffusively and high-frequency modes decay, whereas for $\mu>1/2$ the highest-frequency components become unstable and grow without bound.

The comparison between the stable and unstable simulations illustrates an important conceptual distinction between physical diffusion and numerical instability. Physically, diffusion smooths spatial gradients and dissipates high-frequency structure. Numerically, however, an explicit discretization that violates the stability condition reverses this behavior and amplifies precisely those high-frequency modes that should decay most rapidly. This instability is especially severe because the shortest resolved wavelengths are typically the first to become unstable, leading to rapid contamination of the entire numerical solution.

The computation also demonstrates the increasingly restrictive nature of explicit parabolic timestep constraints. Because the stability limit scales like $\Delta x^2$, refining the spatial mesh rapidly forces the timestep toward extremely small values. This quadratic scaling makes purely explicit diffusion solvers prohibitively expensive for high-resolution or multidimensional simulations and motivates the implicit, semi-implicit, and exponential integration techniques introduced later in Section 20.3.

The modular structure of the implementation allows the framework to be extended naturally to more advanced diffusive solvers. The FTCS update may be replaced by implicit (\\theta)-schemes, Crank–Nicolson discretizations, exponential integrators, or adaptive timestepping methods. Higher-dimensional Laplacians, variable diffusion coefficients, nonlinear diffusion operators, and sparse linear algebra solvers can also be incorporated while preserving the same underlying discretization framework. Consequently, Program 20.3.1 serves as a foundational example for the broader study of diffusion-dominated numerical computation developed throughout the remainder of the section.

## 20.3.2. The $\theta$-Scheme, Backward Euler, and Crank-Nicolson

A unified way to represent the main implicit and semi-implicit methods is the $\theta$-scheme. Let $L$ denote the standard second-difference matrix corresponding to the discrete Laplacian, so that:

\begin{equation}
(LU)_j
=
U_{j+1}-2U_j+U_{j-1}
\tag{20.3.9}
\end{equation}

The $\theta$-scheme for the constant-coefficient diffusion equation may be written as:

\begin{equation}
\frac{U^{n+1}-U^n}{\Delta t}
=
\frac{D}{\Delta x^2}
\left[
\theta L U^{n+1}
+
(1-\theta)L U^n
\right]
\tag{20.3.10}
\end{equation}

Using

$$\mu=\frac{D\Delta t}{\Delta x^2},\tag{20.3.11}$$

this becomes the linear system:

$$\left(I-\theta\mu L\right)U^{n+1} = \left(I+(1-\theta)\mu L\right)U^n \tag{20.3.12}$$

Several important schemes are obtained by choosing $\theta$. For $\theta=0$, (20.3.12) reduces to the explicit FTCS method in (20.3.4). For $\theta=1$, one obtains backward Euler,

$$\left(I-\mu L\right)U^{n+1}=U^n \tag{20.3.13}$$

In component form, for an interior grid point,

$$-\mu u_{j-1}^{n+1}+(1+2\mu)u_j^{n+1} - \mu u_{j+1}^{n+1} = u_j^n \tag{20.3.14}$$

Backward Euler is first-order accurate in time but unconditionally stable for the linear diffusion equation. Its numerical damping is strong, especially for stiff high-frequency modes. This makes it a robust choice when the initial data are rough, when grid-scale noise must be removed, or when the physical model itself is strongly dissipative. In one space dimension, the matrix in (20.3.14) is tridiagonal, so the linear system can be solved in $O(N)$ operations using the Thomas algorithm. If $D$, $\Delta t$, $\Delta x$, and the boundary conditions remain fixed, the same tridiagonal factorization can be reused at every time step.

For $\theta=\tfrac{1}{2}$, the $\theta$-scheme becomes the Crank-Nicolson method,

$$\left(I-\frac{\mu}{2}L\right)U^{n+1} = \left(I+\frac{\mu}{2}L\right)U^n \tag{20.3.15}$$

For the constant-coefficient diffusion equation, Crank-Nicolson is second-order accurate in time and second-order accurate in space when combined with the centered second-difference Laplacian. The Fourier amplification factor is:

\begin{equation}
\xi(\theta)
=
\frac{
1-2\mu\sin^2(\theta/2)
}{
1+2\mu\sin^2(\theta/2)
}
\tag{20.3.16}
\end{equation}

Therefore,

$$|\xi(\theta)|\le 1\qquad\text{for all}\qquad\mu\ge 0 \tag{20.3.17}$$

Crank-Nicolson is thus A-stable for diffusion. However, it is not $L$-stable. For very stiff modes, the amplification factor approaches $-1$, so the method does not strongly damp the highest-frequency components. This distinction is important in practice. Crank-Nicolson is highly accurate and efficient for smooth diffusive evolution, but backward Euler is often preferred as a startup step or fallback method when the initial condition is nonsmooth, when discontinuities are present in coefficients, or when aggressive damping of grid-scale oscillations is desired.

The computational trade-off is clear. An explicit diffusion step costs $O(N)$ work in one dimension, but stability enforces $\Delta t=O(\Delta x^2)$. Backward Euler and Crank-Nicolson also cost $O(N)$ per step in one dimension because the matrix is tridiagonal, but the severe explicit stability barrier is removed. The time step is then limited mainly by accuracy rather than blow-up. In two and three dimensions, the implicit systems become sparse rather than tridiagonal, so direct solvers, Krylov methods, multigrid, alternating-direction implicit splittings, or tensor-structured exponential integrators become more attractive.

### Rust Implementation

Following the discussion in Subsection 20.3.2 on the $\theta$-scheme, implicit diffusion discretization, and the contrasting damping behavior of backward Euler and Crank–Nicolson methods, Program 20.3.2 provides a practical implementation of implicit time integration for the one-dimensional diffusion equation using a unified (\\theta)-scheme framework. The program directly implements the matrix formulation introduced in equations (20.3.10)–(20.3.12) and demonstrates how different choices of $\theta$ recover backward Euler and Crank–Nicolson discretizations. Because the resulting one-dimensional implicit diffusion operator is tridiagonal, the program solves the linear systems efficiently using the Thomas algorithm. The implementation intentionally employs a timestep parameter far larger than the explicit FTCS stability limit, thereby illustrating the unconditional stability properties of implicit diffusion solvers. By comparing the damping behavior of backward Euler and Crank–Nicolson, the program also demonstrates the important distinction between A-stability and L-stability discussed later in the subsection.

At the core of the implementation is the `Grid1D` structure, which defines the spatial discretization used for the implicit diffusion problem. The structure stores the number of grid points, the number of interior unknowns, the computational interval, and the mesh spacing $\Delta x$. The `point` function computes the physical coordinate associated with each grid index and is used to initialize the numerical solution. Because homogeneous Dirichlet boundary conditions are imposed, only the interior grid values are treated as unknowns in the linear system, while the boundary values remain fixed throughout the simulation.

The program introduces the `TridiagonalMatrix` structure to represent the sparse linear systems generated by the implicit diffusion discretization. Since the discrete Laplacian operator in equation (20.3.9) couples only nearest-neighbor grid points, the resulting matrices contain nonzero entries only on the main diagonal and the first upper and lower diagonals. The function `build_theta_matrix` constructs the tridiagonal coefficient matrix corresponding to the operator $I-\theta\mu L$ appearing in equation (20.3.12). This sparse representation reflects one of the major computational advantages of one-dimensional implicit diffusion solvers: the linear systems can be solved in $O(N)$ operations rather than through dense matrix inversion.

The function `initial_condition` constructs the initial solution profile used to study diffusive smoothing and damping behavior. The profile combines a smooth low-frequency sine component with a smaller high-frequency oscillation. The high-frequency component is particularly important because diffusion preferentially damps the shortest resolved wavelengths, and the relative damping behavior of these stiff modes differs substantially between backward Euler and Crank–Nicolson discretizations. This allows the program to illustrate the stronger numerical dissipation of backward Euler and the weaker high-frequency damping associated with Crank–Nicolson.

The discrete Laplacian operator corresponding to equation (20.3.9) is implemented through the function `apply_laplacian_interior`. The function applies the centered second-difference stencil to the interior solution vector while incorporating homogeneous Dirichlet boundary conditions through fixed boundary values. The resulting operator approximates the continuous second derivative and forms the spatial component of the (\\theta)-scheme discretization.

The function `theta_rhs` constructs the right-hand side vector associated with equation (20.3.12). It evaluates the explicit contribution $(I+(1-\theta)\mu L)U^n$ by combining the current solution with the discrete Laplacian operator. The parameter (\\theta) therefore controls the balance between implicit and explicit contributions in the numerical update. When $\theta=1$, the scheme reduces to backward Euler as described in equation (20.3.13), while $\theta=\tfrac12$ produces the Crank–Nicolson method of equation (20.3.15).

The tridiagonal linear systems are solved through the function `thomas_solve`, which implements the Thomas algorithm for banded systems. The method performs forward elimination followed by backward substitution using only the three nonzero diagonals of the matrix. Because the diffusion operator remains constant throughout the computation, the same tridiagonal structure is reused at every timestep. This reflects the computational advantage emphasized in Subsection 20.3.2: implicit one-dimensional diffusion solvers retain linear complexity despite removing the severe explicit timestep restriction.

The function `theta_step` combines the right-hand side construction and tridiagonal solve into a complete implicit timestep update. The method first assembles the right-hand side vector and then solves the linear system corresponding to the chosen value of $\theta$. Several auxiliary diagnostic functions are included to monitor the evolving numerical solution. The functions `discrete_mass_interior`, `l2_norm_interior`, and `max_abs_value` track the overall amplitude and energy-like decay of the solution, while `roughness_measure` quantifies the amount of high-frequency structure remaining in the numerical solution. This roughness diagnostic is particularly useful for comparing the stronger damping behavior of backward Euler against the weaker damping associated with Crank–Nicolson.

The function `run_theta_case` serves as the primary driver for each implicit diffusion experiment. It constructs the $\theta$-scheme matrix, initializes the solution, advances the system through repeated implicit timesteps, and reports diagnostic quantities describing the smoothing behavior of the numerical method. The function intentionally uses a timestep corresponding to $\mu=5$, which is far larger than the explicit FTCS stability limit of equation (20.3.8). Despite this large timestep, both backward Euler and Crank–Nicolson remain stable, thereby illustrating the unconditional stability properties summarized in equation (20.3.17).

The `main` function coordinates the overall comparison between backward Euler and Crank–Nicolson discretizations. It constructs the spatial grid, specifies the diffusion coefficient and final integration time, and executes both implicit schemes using the same large timestep parameter. The resulting diagnostics clearly demonstrate the characteristic behavior of each method: backward Euler strongly damps high-frequency structure and rapidly reduces solution roughness, while Crank–Nicolson remains stable but preserves more oscillatory content because it is A-stable rather than L-stable. This computational comparison provides direct numerical evidence for the stability and damping properties developed analytically throughout Subsection 20.3.2.

```rust
// Program 20.3.2. Implicit Theta-Scheme Diffusion Solver with Thomas Algorithm
//
// Problem statement:
// Solve the one-dimensional constant-coefficient diffusion equation
//
//     u_t = D u_xx
//
// on a Dirichlet domain using the theta-scheme
//
//     (I - theta mu L) U^{n+1} = (I + (1 - theta) mu L) U^n.
//
// The program compares backward Euler (theta = 1) and Crank-Nicolson
// (theta = 1/2). It uses the tridiagonal structure of the one-dimensional
// implicit diffusion operator and solves each linear system by the Thomas
// algorithm.

use std::f64::consts::PI;

#[derive(Clone)]
struct Grid1D {
    n_total: usize,
    n_interior: usize,
    xmin: f64,
    xmax: f64,
    dx: f64,
}

impl Grid1D {
    fn new(n_total: usize, xmin: f64, xmax: f64) -> Self {
        assert!(
            n_total >= 5,
            "The grid must contain at least five points including boundaries."
        );
        assert!(xmax > xmin, "The domain length must be positive.");

        let dx = (xmax - xmin) / (n_total as f64 - 1.0);
        let n_interior = n_total - 2;

        Self {
            n_total,
            n_interior,
            xmin,
            xmax,
            dx,
        }
    }

    fn point(&self, i: usize) -> f64 {
        self.xmin + i as f64 * self.dx
    }
}

#[derive(Clone)]
struct TridiagonalMatrix {
    lower: Vec<f64>,
    diag: Vec<f64>,
    upper: Vec<f64>,
}

impl TridiagonalMatrix {
    fn new(n: usize, lower_value: f64, diag_value: f64, upper_value: f64) -> Self {
        assert!(n >= 1, "Matrix dimension must be positive.");

        let lower = if n > 1 {
            vec![lower_value; n - 1]
        } else {
            Vec::new()
        };

        let upper = if n > 1 {
            vec![upper_value; n - 1]
        } else {
            Vec::new()
        };

        let diag = vec![diag_value; n];

        Self { lower, diag, upper }
    }
}

fn initial_condition(x: f64) -> f64 {
    let smooth_part = (PI * x).sin();
    let high_frequency_part = 0.25 * (15.0 * PI * x).sin();

    smooth_part + high_frequency_part
}

fn apply_laplacian_interior(u: &[f64]) -> Vec<f64> {
    let n = u.len();
    let mut lu = vec![0.0_f64; n];

    for j in 0..n {
        let left = if j == 0 { 0.0 } else { u[j - 1] };
        let right = if j + 1 == n { 0.0 } else { u[j + 1] };

        lu[j] = right - 2.0 * u[j] + left;
    }

    lu
}

fn build_theta_matrix(n: usize, theta: f64, mu: f64) -> TridiagonalMatrix {
    let lower = -theta * mu;
    let diag = 1.0 + 2.0 * theta * mu;
    let upper = -theta * mu;

    TridiagonalMatrix::new(n, lower, diag, upper)
}

fn theta_rhs(u: &[f64], theta: f64, mu: f64) -> Vec<f64> {
    let lu = apply_laplacian_interior(u);
    let factor = (1.0 - theta) * mu;

    u.iter()
        .zip(lu.iter())
        .map(|(uj, luj)| uj + factor * luj)
        .collect()
}

fn thomas_solve(matrix: &TridiagonalMatrix, rhs: &[f64]) -> Vec<f64> {
    let n = matrix.diag.len();

    assert_eq!(rhs.len(), n);
    assert_eq!(matrix.lower.len(), n.saturating_sub(1));
    assert_eq!(matrix.upper.len(), n.saturating_sub(1));

    if n == 1 {
        return vec![rhs[0] / matrix.diag[0]];
    }

    let mut c_prime = vec![0.0_f64; n - 1];
    let mut d_prime = vec![0.0_f64; n];

    c_prime[0] = matrix.upper[0] / matrix.diag[0];
    d_prime[0] = rhs[0] / matrix.diag[0];

    for i in 1..n {
        let denominator = matrix.diag[i] - matrix.lower[i - 1] * c_prime[i - 1];

        if i < n - 1 {
            c_prime[i] = matrix.upper[i] / denominator;
        }

        d_prime[i] = (rhs[i] - matrix.lower[i - 1] * d_prime[i - 1]) / denominator;
    }

    let mut solution = vec![0.0_f64; n];
    solution[n - 1] = d_prime[n - 1];

    for i in (0..n - 1).rev() {
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
    }

    solution
}

fn theta_step(u: &[f64], matrix: &TridiagonalMatrix, theta: f64, mu: f64) -> Vec<f64> {
    let rhs = theta_rhs(u, theta, mu);
    thomas_solve(matrix, &rhs)
}

fn discrete_mass_interior(u: &[f64], dx: f64) -> f64 {
    u.iter().sum::<f64>() * dx
}

fn l2_norm_interior(u: &[f64], dx: f64) -> f64 {
    (u.iter().map(|value| value * value).sum::<f64>() * dx).sqrt()
}

fn max_abs_value(u: &[f64]) -> f64 {
    u.iter().map(|value| value.abs()).fold(0.0_f64, f64::max)
}

fn roughness_measure(u: &[f64]) -> f64 {
    let n = u.len();

    if n < 2 {
        return 0.0;
    }

    let mut sum = 0.0_f64;

    for j in 0..n - 1 {
        let diff = u[j + 1] - u[j];
        sum += diff * diff;
    }

    sum.sqrt()
}

fn run_theta_case(
    case_name: &str,
    theta: f64,
    grid: &Grid1D,
    diffusion: f64,
    final_time: f64,
    requested_mu: f64,
) {
    let dt_nominal = requested_mu * grid.dx * grid.dx / diffusion;
    let number_of_steps = (final_time / dt_nominal).ceil() as usize;
    let dt = final_time / number_of_steps as f64;
    let mu = diffusion * dt / (grid.dx * grid.dx);

    let matrix = build_theta_matrix(grid.n_interior, theta, mu);

    let mut u: Vec<f64> = (1..grid.n_total - 1)
        .map(|i| initial_condition(grid.point(i)))
        .collect();

    let initial_mass = discrete_mass_interior(&u, grid.dx);
    let initial_l2 = l2_norm_interior(&u, grid.dx);
    let initial_max = max_abs_value(&u);
    let initial_roughness = roughness_measure(&u);

    println!("{}", case_name);
    println!("{}", "-".repeat(case_name.len()));
    println!("theta                    = {:.6}", theta);
    println!("Requested mu             = {:.6}", requested_mu);
    println!("Effective mu             = {:.6}", mu);
    println!("Time step dt             = {:.8}", dt);
    println!("Number of steps          = {}", number_of_steps);

    for _ in 0..number_of_steps {
        u = theta_step(&u, &matrix, theta, mu);
    }

    let final_mass = discrete_mass_interior(&u, grid.dx);
    let final_l2 = l2_norm_interior(&u, grid.dx);
    let final_max = max_abs_value(&u);
    let final_roughness = roughness_measure(&u);

    println!("Initial interior mass    = {:.12}", initial_mass);
    println!("Final interior mass      = {:.12}", final_mass);
    println!(
        "Interior mass change     = {:.6e}",
        (final_mass - initial_mass).abs()
    );
    println!("Initial L2 norm          = {:.12}", initial_l2);
    println!("Final L2 norm            = {:.12}", final_l2);
    println!("Initial max |u|          = {:.12}", initial_max);
    println!("Final max |u|            = {:.12}", final_max);
    println!("Initial roughness        = {:.12}", initial_roughness);
    println!("Final roughness          = {:.12}", final_roughness);
    println!();
}

fn main() {
    let grid = Grid1D::new(101, 0.0, 1.0);

    let diffusion: f64 = 0.01;
    let final_time: f64 = 0.10;

    // This value is intentionally much larger than the explicit FTCS
    // stability limit mu <= 1/2, showing that implicit methods remove
    // the explicit parabolic stability barrier.
    let requested_mu: f64 = 5.0;

    println!("Implicit Theta-Scheme Diffusion Solver");
    println!("======================================");
    println!();

    println!("Grid Parameters");
    println!("---------------");
    println!("Total grid points        = {}", grid.n_total);
    println!("Interior unknowns        = {}", grid.n_interior);
    println!("Domain                   = [{:.3}, {:.3}]", grid.xmin, grid.xmax);
    println!("Grid spacing dx          = {:.8}", grid.dx);
    println!();

    println!("Physical Parameters");
    println!("-------------------");
    println!("Diffusion coefficient D  = {:.6}", diffusion);
    println!("Final time               = {:.6}", final_time);
    println!();

    run_theta_case(
        "Backward Euler Case",
        1.0,
        &grid,
        diffusion,
        final_time,
        requested_mu,
    );

    run_theta_case(
        "Crank-Nicolson Case",
        0.5,
        &grid,
        diffusion,
        final_time,
        requested_mu,
    );

    println!("Interpretation");
    println!("--------------");
    println!("Backward Euler is strongly damping and robust for stiff modes.");
    println!("Crank-Nicolson is A-stable and second-order accurate, but it damps");
    println!("the highest-frequency modes less aggressively than backward Euler.");
}
```

Program 20.3.2 demonstrates the central computational advantages of implicit diffusion discretizations for parabolic partial differential equations. By replacing the explicit FTCS update with the implicit $\theta$-scheme formulation of equations (20.3.10)–(20.3.12), the severe parabolic timestep restriction of equation (20.3.8) is removed. The numerical experiments show that both backward Euler and Crank–Nicolson remain stable even for timestep parameters far larger than the explicit stability limit, thereby illustrating the practical importance of implicit methods for stiff diffusive systems.

The comparison between backward Euler and Crank–Nicolson also highlights the important distinction between unconditional stability and high-frequency damping. Backward Euler is strongly dissipative and rapidly removes grid-scale oscillations and rough solution features, making it especially robust for nonsmooth initial data or strongly dissipative physical systems. Crank–Nicolson, while second-order accurate and A-stable, damps the highest-frequency modes less aggressively because its amplification factor approaches $-1$ for very stiff modes. Consequently, Crank–Nicolson often preserves oscillatory components that backward Euler suppresses more rapidly.

The implementation further illustrates the close relationship between implicit time integration and sparse linear algebra. In one dimension, the implicit diffusion operator produces tridiagonal matrices that can be solved efficiently in linear time using the Thomas algorithm. In higher dimensions, however, the same implicit philosophy leads naturally to sparse matrix methods, Krylov subspace iteration, multigrid techniques, alternating-direction implicit splittings, and tensor-structured exponential solvers. Thus, implicit diffusion discretization forms an important bridge between PDE stability theory and modern sparse numerical linear algebra.

The modular structure of the code allows the framework to be extended naturally to more advanced diffusion solvers. Variable diffusion coefficients, adaptive timestepping, nonlinear diffusion operators, semi-implicit splitting methods, higher-dimensional sparse discretizations, and matrix-free Krylov solvers can all be incorporated while preserving the same underlying (\\theta)-scheme structure. Consequently, Program 20.3.2 provides a foundational implementation for the broader study of implicit and stiff diffusion-dominated PDE computation developed later in the chapter.

## 20.3.3. Variable and Nonlinear Diffusion

Many physical diffusion problems do not have a constant coefficient. The coefficient may depend on space, temperature, concentration, material phase, or the unknown solution itself. For variable diffusion, it is preferable to discretize in conservative flux form. Define the diffusive flux:

$$q=-D u_x,\qquad u_t=-q_x \tag{20.3.18}$$

A finite-volume discretization over cell $j$ gives:

\begin{equation}
\frac{d\bar{u}_j}{dt}
=
\frac{1}{\Delta x}
\left[
D_{j+1/2}
\frac{u_{j+1}-u_j}{\Delta x}
-
D_{j-1/2}
\frac{u_j-u_{j-1}}{\Delta x}
\right]
\tag{20.3.19}
\end{equation}

The diffusion coefficient belongs naturally at cell interfaces, because the flux crosses interfaces. The choice of $D_{j+1/2}$ is not merely a numerical detail. If the diffusion coefficient has strong jumps, the interface value must represent the correct transmission of flux across material boundaries. In heterogeneous media, arithmetic averaging may not represent the correct physical interface law, and harmonic or problem-specific averaging may be more appropriate.

Nonlinear diffusion introduces additional structure. A standard example is the porous-medium equation,

$$u_t=\Delta(u^m),\qquad m>1 \tag{20.3.20}$$

Equivalently,

$$u_t = \nabla\cdot\!\left(m u^{m-1}\nabla u\right) \tag{20.3.21}$$

Here the effective diffusion coefficient is:

$$D(u)=m u^{m-1} \tag{20.3.22}$$

The coefficient degenerates when $u=0$, which changes both the analysis and the numerical behaviour. A good numerical method must now control more than stability and accuracy. It should preserve nonnegativity, conserve mass where appropriate, and respect the dissipative or energy-decaying structure of the equation. Recent finite-element methods for the porous-medium equation explicitly target positivity preservation and unconditional energy stability, because these properties are essential to the physical interpretation of the solution (Vijaywargiya and Fu, 2024).

A useful way to treat nonlinear diffusion in one dimension is to introduce the primitive diffusion potential,

$$z(u)=\int^u D(s)\,ds \tag{20.3.23}$$

Then,

$$z_x=D(u)u_x \tag{20.3.24}$$

and the nonlinear diffusion equation can be written as:

$$u_t=z_{xx} \tag{20.3.25}$$

A fully implicit method would evaluate $z(u^{n+1})$, leading to a nonlinear system. One simple linearized implicit approximation is obtained by expanding $z(u_j^{n+1})$ about $u_j^n$,

$$z(u_j^{n+1})\approx z(u_j^n)+D(u_j^n)\left(u_j^{n+1}-u_j^n\right) \tag{20.3.26}$$

This approximation converts the nonlinear implicit problem into a linear system at each time step, often with tridiagonal structure in one dimension. The resulting method is not as fully nonlinear as a Newton solve, but it captures the essential idea used in many production codes: treat the stiff diffusive coupling implicitly while avoiding the full cost of nonlinear iteration whenever the approximation is adequate.

The same structural concerns appear in electro-diffusion systems. Positivity of concentrations, discrete energy dissipation, and compatibility between fluxes and potentials are important when the PDE describes charged species, electric fields, or coupled transport. Recent implicit exponential time-differencing schemes for Maxwell-Ampère Nernst-Planck models are designed specifically to preserve positivity and discrete energy dissipation, illustrating how modern diffusion solvers increasingly combine stiffness handling with invariant preservation (Guo, Yin and Zhang, 2024).

### Rust Implementation

Following the discussion in Subsection 20.3.3 on variable and nonlinear diffusion, Program 20.3.3 provides a practical implementation of a conservative finite-volume solver for the porous-medium equation. Unlike constant-coefficient diffusion problems, the porous-medium equation introduces a nonlinear diffusion coefficient that depends directly on the evolving solution itself, as described by equations (20.3.20)–(20.3.22). This changes both the physical and numerical behavior of the problem because the effective diffusion strength degenerates when the solution approaches zero. The program therefore emphasizes several structural properties that are essential in nonlinear diffusion computation: conservative flux evaluation, positivity preservation, and monitoring of dissipative spreading behavior. The implementation uses a flux-form discretization based on equations (20.3.18)–(20.3.19), evaluates nonlinear interface fluxes through the diffusion potential introduced in equations (20.3.23)–(20.3.25), and advances the solution explicitly while enforcing nonnegativity at every timestep. The resulting framework demonstrates how modern nonlinear diffusion solvers combine stability considerations with physically meaningful invariant preservation.

At the core of the implementation is the `Grid1D` structure, which defines the one-dimensional finite-volume mesh used throughout the computation. The structure stores the number of computational cells, the domain boundaries, and the uniform mesh spacing $\Delta x$. The `cell_center` function computes the geometric center of each control volume and is used to initialize the numerical solution. The helper functions `periodic_left` and `periodic_right` implement periodic boundary conditions by wrapping neighboring cell indices across the domain boundaries, thereby ensuring that the numerical flux leaving one side of the domain reenters on the opposite side in a conservative manner.

The function `initial_condition` constructs the compactly supported initial profile used for the porous-medium simulation. The profile is localized within a finite region of the computational domain and vanishes identically outside that region. This choice is particularly appropriate for the porous-medium equation because nonlinear diffusion produces finite-speed spreading rather than the infinite-speed propagation associated with the classical linear heat equation. As the solution evolves, the support gradually expands while the maximum amplitude decreases, illustrating the characteristic nonlinear smoothing behavior of degenerate diffusion.

The nonlinear diffusion structure introduced in equations (20.3.20)–(20.3.22) is represented through the functions `diffusion_potential` and `nonlinear_diffusion_coefficient`. The function `diffusion_potential` evaluates the primitive diffusion potential $z(u)=u^m$, corresponding to equations (20.3.23)–(20.3.25), while `nonlinear_diffusion_coefficient` computes the effective nonlinear diffusion coefficient $D(u)=mu^{m-1}$. The coefficient vanishes when $u=0$, reflecting the degeneracy of the porous-medium equation near vacuum regions. This degeneracy is responsible for the finite propagation speed and compact support behavior observed in porous-medium flows.

The conservative finite-volume flux evaluation is implemented through the function `conservative_fluxes`. Following equations (20.3.18)–(20.3.19), the numerical method computes interface fluxes using differences of the diffusion potential across neighboring cells. The resulting discretization preserves the conservative divergence structure of the continuous PDE because the same interface flux enters adjacent cells with opposite signs. This conservative formulation is essential for maintaining correct mass balance during long-time diffusion evolution.

The explicit conservative timestep update is implemented through the function `explicit_conservative_step`. The function computes the divergence of the interface fluxes and updates the cell averages according to the conservative finite-volume balance law. After each update, a positivity safeguard clips negative solution values back to zero. Although this positivity correction is simple, it reflects an important structural principle emphasized throughout Subsection 20.3.3: physically meaningful nonlinear diffusion solvers should preserve nonnegativity whenever the continuous model preserves positivity.

Several auxiliary diagnostic functions are included to monitor the evolving physical structure of the numerical solution. The function `max_diffusion_coefficient` computes the largest effective nonlinear diffusion coefficient and is used to determine a stable explicit timestep. The function `total_mass` verifies conservative behavior by monitoring the discrete integral of the solution over the computational domain. The function `l2_norm` measures the dissipative decay of the solution magnitude, while `support_width` estimates the width of the region where the solution remains nonzero. This support diagnostic is especially important for porous-medium evolution because the finite propagation speed produces expanding compact support rather than instantaneous global spreading.

The `main` function serves as a complete demonstration of conservative nonlinear diffusion evolution. The program constructs the finite-volume mesh, initializes the compactly supported profile, computes stable timesteps based on the maximum nonlinear diffusion coefficient, and repeatedly advances the solution using the conservative explicit update. Throughout the computation, diagnostic output monitors positivity, amplitude decay, conservation of mass, and expansion of the support region. The numerical results clearly illustrate the characteristic behavior of the porous-medium equation: the solution spreads outward, its peak amplitude decreases, the support expands with finite speed, and the total mass remains conserved to machine precision. These properties provide direct computational confirmation of the structural principles discussed throughout Subsection 20.3.3.

```rust
// Program 20.3.3. Conservative Porous-Medium Diffusion Solver
//
// Problem statement:
// Solve the one-dimensional porous-medium equation
//
//     u_t = (u^m)_{xx},    m > 1,
//
// on a periodic domain using a conservative finite-volume flux formulation.
// The program follows the flux structure in equations (20.3.18)-(20.3.19)
// and uses the nonlinear diffusion relation in equations (20.3.20)-(20.3.22).
//
// The interface flux is computed from the diffusion potential z(u) = u^m,
// corresponding to equations (20.3.23)-(20.3.25). Positivity is enforced
// after each update, and total mass is monitored as a conservation diagnostic.

#[derive(Clone)]
struct Grid1D {
    n: usize,
    xmin: f64,
    xmax: f64,
    dx: f64,
}

impl Grid1D {
    fn new(n: usize, xmin: f64, xmax: f64) -> Self {
        assert!(n >= 4, "The grid must contain at least four cells.");
        assert!(xmax > xmin, "The domain length must be positive.");

        let dx = (xmax - xmin) / n as f64;

        Self { n, xmin, xmax, dx }
    }

    fn cell_center(&self, j: usize) -> f64 {
        self.xmin + (j as f64 + 0.5) * self.dx
    }
}

fn periodic_left(j: usize, n: usize) -> usize {
    if j == 0 { n - 1 } else { j - 1 }
}

fn periodic_right(j: usize, n: usize) -> usize {
    if j + 1 == n { 0 } else { j + 1 }
}

fn initial_condition(x: f64) -> f64 {
    let center = 0.5_f64;
    let radius = 0.18_f64;
    let distance = (x - center).abs();

    if distance < radius {
        1.0 - (distance / radius).powi(2)
    } else {
        0.0
    }
}

fn diffusion_potential(u: f64, exponent_m: f64) -> f64 {
    u.max(0.0).powf(exponent_m)
}

fn nonlinear_diffusion_coefficient(u: f64, exponent_m: f64) -> f64 {
    if u <= 0.0 {
        0.0
    } else {
        exponent_m * u.powf(exponent_m - 1.0)
    }
}

fn conservative_fluxes(u: &[f64], exponent_m: f64, dx: f64) -> Vec<f64> {
    let n = u.len();

    // flux[j] is the numerical flux crossing the interface x_{j+1/2}.
    // For u_t = (z(u))_{xx}, write q = -z_x and u_t = -q_x.
    let mut flux = vec![0.0_f64; n];

    for j in 0..n {
        let right = periodic_right(j, n);

        let z_left = diffusion_potential(u[j], exponent_m);
        let z_right = diffusion_potential(u[right], exponent_m);

        flux[j] = -(z_right - z_left) / dx;
    }

    flux
}

fn explicit_conservative_step(
    u: &[f64],
    exponent_m: f64,
    dt: f64,
    dx: f64,
) -> Vec<f64> {
    let n = u.len();
    let flux = conservative_fluxes(u, exponent_m, dx);

    let mut next = vec![0.0_f64; n];

    for j in 0..n {
        let left_face = periodic_left(j, n);

        // Conservative update:
        // u_j^{n+1} = u_j^n - dt/dx (q_{j+1/2} - q_{j-1/2})
        next[j] = u[j] - (dt / dx) * (flux[j] - flux[left_face]);

        // Positivity safeguard.
        if next[j] < 0.0 {
            next[j] = 0.0;
        }
    }

    next
}

fn max_diffusion_coefficient(u: &[f64], exponent_m: f64) -> f64 {
    u.iter()
        .map(|value| nonlinear_diffusion_coefficient(*value, exponent_m))
        .fold(0.0_f64, f64::max)
}

fn total_mass(u: &[f64], dx: f64) -> f64 {
    u.iter().sum::<f64>() * dx
}

fn l2_norm(u: &[f64], dx: f64) -> f64 {
    (u.iter().map(|value| value * value).sum::<f64>() * dx).sqrt()
}

fn min_value(u: &[f64]) -> f64 {
    u.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_value(u: &[f64]) -> f64 {
    u.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn support_width(u: &[f64], dx: f64, threshold: f64) -> f64 {
    let active_cells = u.iter().filter(|value| **value > threshold).count();

    active_cells as f64 * dx
}

fn main() {
    let grid = Grid1D::new(240, 0.0, 1.0);

    let exponent_m: f64 = 2.0;
    let final_time: f64 = 0.01;
    let cfl: f64 = 0.25;

    let mut u: Vec<f64> = (0..grid.n)
        .map(|j| initial_condition(grid.cell_center(j)))
        .collect();

    let initial_mass = total_mass(&u, grid.dx);
    let initial_l2 = l2_norm(&u, grid.dx);
    let initial_support = support_width(&u, grid.dx, 1.0e-6);

    let mut time = 0.0_f64;
    let mut step = 0_usize;

    println!("Conservative Porous-Medium Diffusion Solver");
    println!("===========================================");
    println!();

    println!("Grid Parameters");
    println!("---------------");
    println!("Number of cells           = {}", grid.n);
    println!("Domain                    = [{:.3}, {:.3}]", grid.xmin, grid.xmax);
    println!("Grid spacing dx           = {:.8}", grid.dx);
    println!();

    println!("Physical Parameters");
    println!("-------------------");
    println!("Porous-medium exponent m  = {:.6}", exponent_m);
    println!("Final time                = {:.6}", final_time);
    println!("CFL factor                = {:.6}", cfl);
    println!();

    while time < final_time {
        let max_d = max_diffusion_coefficient(&u, exponent_m).max(1.0e-14);
        let mut dt = cfl * grid.dx * grid.dx / max_d;

        if time + dt > final_time {
            dt = final_time - time;
        }

        u = explicit_conservative_step(&u, exponent_m, dt, grid.dx);

        time += dt;
        step += 1;

        if step % 50 == 0 || time >= final_time {
            println!(
                "Step {:>5} | time = {:>.6} | min(u) = {:>.6} | max(u) = {:>.6}",
                step,
                time,
                min_value(&u),
                max_value(&u)
            );
        }
    }

    let final_mass = total_mass(&u, grid.dx);
    let final_l2 = l2_norm(&u, grid.dx);
    let final_support = support_width(&u, grid.dx, 1.0e-6);

    println!();
    println!("Diagnostics");
    println!("-----------");
    println!("Initial mass             = {:.12}", initial_mass);
    println!("Final mass               = {:.12}", final_mass);
    println!(
        "Absolute mass change     = {:.6e}",
        (final_mass - initial_mass).abs()
    );
    println!("Initial L2 norm          = {:.12}", initial_l2);
    println!("Final L2 norm            = {:.12}", final_l2);
    println!("Initial support width    = {:.12}", initial_support);
    println!("Final support width      = {:.12}", final_support);
    println!();

    println!("Sample Cell Averages");
    println!("--------------------");
    println!("{:>8} {:>14} {:>18}", "Cell", "x", "u_bar");

    for j in (0..grid.n).step_by(20) {
        println!(
            "{:>8} {:>14.6} {:>18.10}",
            j,
            grid.cell_center(j),
            u[j]
        );
    }
}
```

Program 20.3.3 demonstrates the fundamental numerical challenges associated with nonlinear and degenerate diffusion equations. Unlike constant-coefficient linear diffusion, the porous-medium equation introduces a solution-dependent diffusion coefficient whose magnitude changes dynamically during the computation. This nonlinear coupling alters both the physical evolution and the stability properties of the numerical method, requiring careful treatment of conservation, positivity, and dissipative structure.

The numerical results clearly illustrate the characteristic behavior of porous-medium diffusion. The solution amplitude decreases steadily while the compact support expands outward over time, reflecting finite-speed nonlinear spreading rather than the instantaneous infinite-speed propagation associated with the classical heat equation. At the same time, the total mass remains conserved to machine precision because the finite-volume discretization preserves balanced interface flux exchange across the computational mesh.

The positivity safeguard also reflects an important principle in nonlinear diffusion computation. Because the diffusion coefficient degenerates when (u=0), negative numerical values can produce nonphysical behavior or undefined coefficients. Enforcing nonnegativity therefore helps preserve the physical interpretation of the solution and stabilizes the numerical evolution near vacuum regions. Modern production diffusion solvers often incorporate more sophisticated positivity-preserving or energy-stable discretizations, but the present implementation captures the essential structural ideas.

The modular structure of the code naturally supports extension to more advanced nonlinear diffusion models. Semi-implicit linearized methods based on equation (20.3.26), Newton-type nonlinear implicit solvers, adaptive timestepping, variable material coefficients, electro-diffusion systems, and multidimensional sparse discretizations can all be incorporated while preserving the same conservative flux-based framework. Consequently, Program 20.3.3 provides a foundational example for the broader study of nonlinear and structure-preserving diffusion computation developed later in the chapter.

## 20.3.4. Modern Diffusive Models and Structure-Preserving Time Integration

A strong modern theme in diffusive and advection-diffusion-reaction computation is the separation of stiff and nonstiff components. Suppose an evolution equation can be written schematically as:

$$U'(t)=AU(t)+G(U(t),t) \tag{20.3.27}$$

where $A$ represents a stiff diffusion or linear differential operator and $G$ contains advection, reaction, nonlinear forcing, or remaining variable-coefficient terms. Exponential and Lawson-type methods exploit the fact that the stiff linear part can be advanced through matrix functions involving $A$, while the remaining part is treated explicitly or semi-explicitly. Recent work shows that such methods can be accelerated by extracting a constant-coefficient differential operator for which the action of the matrix exponential or related functions is cheaper to compute. In important semilinear advection-diffusion-reaction settings, this can yield efficient and stable methods without treating the entire problem fully implicitly (Caliari et al., 2024).

Directional splitting is another important strategy. In multidimensional advection-diffusion-reaction systems, the operator may be decomposed according to coordinate directions or physical processes. A second-order directional split exponential integrator can reduce the cost of applying multidimensional stiff operators while preserving second-order temporal accuracy. Such methods are especially relevant in two- and three-dimensional systems where fully coupled implicit solves are expensive and where the diffusion operator has exploitable tensor or directional structure (Caliari and Cassini, 2024).

Battery modeling provides a representative applied setting in which diffusive PDEs occur at several coupled scales. In a spherical active-material particle, lithium concentration (c_s(r,t)) is often modeled by

\begin{equation}
\partial_t c_s
=
\frac{1}{r^2}
\partial_r\!\left(
r^2 D_s(c_s,T)\,\partial_r c_s
\right),
\qquad
0<r<R_p
\tag{20.3.28}
\end{equation}

The condition at $r=0$ is a symmetry condition, while the boundary condition at $r=R_p$ is a flux condition tied to the electrochemical reaction current. At the cell or pack scale, temperature may satisfy a heat equation of the form:

\begin{equation}
\rho c_p\,\partial_t T
=
\nabla\cdot(k\nabla T)
+
Q_{\mathrm{ohm}}
+
Q_{\mathrm{rxn}}
+
Q_{\mathrm{ent}}
\tag{20.3.29}
\end{equation}

Here $\rho$ is density, $c_p$ is heat capacity, $k$ is thermal conductivity, and the source terms represent ohmic, reaction, and entropic heat generation. Such models may also be coupled to coolant-flow PDEs or pack-level thermal transport equations. Recent battery modeling work emphasizes the need to balance electrochemical fidelity, thermal safety, and computational cost across particle, cell, and pack scales (Mama et al., 2025; Kalungi and Menart, 2025; Ferreira and Tang, 2025).

Diffusion coefficients in battery materials are not always spatially uniform or even well represented by a single effective value. Solid-phase diffusion mismatch can materially affect voltage prediction in high-power lithium titanate oxide cells, while grain-boundary diffusion in solid electrolytes can differ substantially from bulk diffusion. These observations motivate heterogeneous and multiscale diffusion models rather than oversimplified constant-coefficient approximations (Chen et al., 2024; Hasegawa et al., 2024). Numerically, such models require careful interface fluxes, stable implicit or semi-implicit time stepping, and efficient sparse linear algebra.

It is also useful to distinguish diffusion from other second-derivative evolution equations. The Schrödinger-type equation,

$$i\psi_t=H\psi\tag{20.3.30}$$

contains a spatial operator that may resemble a diffusion operator algebraically, but its physical invariant is different. If the discrete spatial operator $H_h$ is Hermitian, the correct evolution preserves the discrete $\ell^2$ norm rather than dissipating it. The Crank-Nicolson update becomes:

\begin{equation}
\left(
I+\frac{i\Delta t}{2}H_h
\right)\psi^{n+1}
=
\left(
I-\frac{i\Delta t}{2}H_h
\right)\psi^n
\tag{20.3.31}
\end{equation}

This is a Cayley transform and is unitary when $H_h$ is Hermitian. Thus, the same implicit midpoint algebra that gives a stable method for diffusion gives a norm-preserving method for Schrödinger evolution. The difference lies in the operator and the invariant. For diffusion, the numerical method should reproduce smoothing and dissipation. For Schrödinger-type evolution, it should preserve phase structure and norm. This distinction is essential in designing methods that respect the underlying PDE rather than only its superficial differential form.

For Rust implementation, diffusive problems suggest a different architecture from flux-conservative hyperbolic solvers. Explicit diffusion can be represented by stencil operations on contiguous arrays, but practical solvers often require implicit linear algebra. In one dimension this means tridiagonal storage and an $O(N)$ Thomas solver. In multiple dimensions it means sparse matrices, matrix-free stencil application, Krylov solvers, multigrid preconditioners, or separable tensor operators. When coefficients are constant, factorizations or preconditioners should be reused. When coefficients depend on the solution, the code should separate coefficient evaluation, flux assembly, boundary-condition enforcement, and linear or nonlinear solving. This separation mirrors the mathematical structure of (20.3.18) to (20.3.26) and makes it possible to test conservation, positivity, damping, and convergence independently.

# 20.4. Initial Value Problems in Multi-dimensions

The one-dimensional advection and diffusion problems studied in the preceding sections provide the essential building blocks for time-dependent partial differential equations. In two and three space dimensions, however, the problem is no longer merely transport along a line or smoothing along a single coordinate direction. The unknown is a scalar or vector field defined over a multidimensional spatial domain, and the numerical method must account for fluxes, gradients, source terms, and boundary interactions in more than one direction. This makes multidimensional initial value problems more than a direct repetition of the one-dimensional case. Directional coupling, anisotropy, stability restrictions, conservation, splitting error, and sparse linear algebra all become central parts of the numerical formulation.

Let,

$$\mathbf{x}\in\Omega\subset\mathbb{R}^d,\qquad d=2 \ \text{or}\ 3,$$

and let $u(\mathbf{x},t)$ denote the evolving unknown. A general scalar advection-diffusion-reaction model may be written as:

\begin{equation}
\begin{aligned}
\partial_t u+\nabla\cdot \mathbf{F}(u,\mathbf{x},t)
&=
\nabla\cdot\!\left(K(\mathbf{x},u)\nabla u\right)
+s(u,\mathbf{x},t),\\
d&=2\ \text{or}\ 3
\end{aligned}
\tag{20.4.1}
\end{equation}

Here $\mathbf{F}$ is the advective or conservative flux, $K$ is a diffusion tensor or matrix-valued coefficient, and $s$ is a source or reaction term. If $u$ is vector-valued, then $\mathbf{F}=(F^{(1)},\ldots,F^{(d)})$ contains one flux vector in each coordinate direction for each conserved component. Equation (20.4.1) covers many important applications, including compressible and incompressible flow, tracer transport, heat conduction, porous-media flow, plasma and magnetohydrodynamic models, and reaction-transport systems. In multidimensional conservation laws, the coupling between directions is not merely a matter of bookkeeping. Genuinely multidimensional wave propagation, physical constraint preservation, stationary-state preservation, and positivity can all be degraded if a method is assembled too naively from one-dimensional components (Ling and Tang, 2023; Barsukow et al., 2026).

## 20.4.1. Conservative Finite-Volume Balance in Several Space Dimensions

For conservative problems on a Cartesian grid, the natural finite-volume unknown is the cell average. In two dimensions, let,

$$
C_{ij}
=
[x_{i-1/2},x_{i+1/2}]
\times
[y_{j-1/2},y_{j+1/2}]
$$

be a rectangular control volume. The cell average is:

\begin{equation}
\bar{u}_{ij}(t)
=
\frac{1}{\Delta x\,\Delta y}
\int_{C_{ij}} u(x,y,t)\,dx\,dy
\tag{20.4.2}
\end{equation}

The same conservation principle used in one dimension now applies across all faces of the cell. In two dimensions, a cell exchanges flux through its left and right faces in the $x$-direction and through its lower and upper faces in the $y$-direction. If $F$ denotes the numerical flux in the $x$-direction and $G$ denotes the numerical flux in the $y$-direction, the semi-discrete finite-volume balance is:

\begin{equation}
\frac{d\bar{u}_{ij}}{dt}
=
-\frac{F_{i+1/2,j}-F_{i-1/2,j}}{\Delta x}
-\frac{G_{i,j+1/2}-G_{i,j-1/2}}{\Delta y}
+\bar{s}_{ij}
\tag{20.4.3}
\end{equation}

In three dimensions, an additional flux difference in the $z$-direction is added:

\begin{equation}
\begin{aligned}
\frac{d\bar{u}_{ijk}}{dt}
={}&
-\frac{F_{i+1/2,j,k}-F_{i-1/2,j,k}}{\Delta x}
-\frac{G_{i,j+1/2,k}-G_{i,j-1/2,k}}{\Delta y}\\
&-\frac{H_{i,j,k+1/2}-H_{i,j,k-1/2}}{\Delta z}
+\bar{s}_{ijk}
\end{aligned}
\tag{20.4.4}
\end{equation}

Equations (20.4.3) and (20.4.4) express the fundamental structural fact of multidimensional conservative computation: one should not discretize $\nabla\cdot\mathbf{F}$ as an abstract differential expression detached from the control volume. Instead, one balances numerical fluxes through faces. The flux leaving one control volume is the same flux entering the neighbouring control volume with opposite sign. Consequently, when the discrete equations are summed over all cells, all interior fluxes cancel and only boundary fluxes and sources remain.

This property is essential when the modeled quantities are mass, momentum, charge, water depth, energy, or other conserved physical fields. A method that is high-order pointwise but fails to preserve the correct integral balance may produce long-time errors that have no physical interpretation. For this reason, finite-volume methods remain the standard choice whenever local conservation is non-negotiable. Recent multidimensional finite-volume formulations increasingly construct stabilization and numerical fluxes in a direction-coupled way rather than by a purely dimension-by-dimension procedure, because stationary states and multidirectional waves may otherwise be represented incorrectly (Ling and Tang, 2023; Barsukow et al., 2026).

## 20.4.2. Matrix and Method-of-Lines Formulation

For linear advection-diffusion equations on rectangular grids, a matrix formulation gives a compact view of the multidimensional semi-discrete problem. Suppose $U(t)\in\mathbb{R}^{n_x\times n_y}$ stores nodal or cell-centred values on a two-dimensional grid. A typical semi-discrete advection-diffusion-reaction model may be written schematically as:

\begin{equation}
\frac{dU}{dt}
=
-a_xD_xU
-a_yUD_y^{\top}
+\kappa_xT_xU
+\kappa_yUT_y^{\top}
+S(U,t)
\tag{20.4.5}
\end{equation}

where $D_x$ and $D_y$ are first-derivative matrices in the $x$ and $y$ directions, while $T_x$ and $T_y$ are second-difference matrices. The terms $D_xU$ and $UD_y^{\top}$ represent directional advection, while $T_xU$ and $UT_y^{\top}$ represent directional diffusion. The source or reaction term is collected in $S(U,t)$.

After vectorization, the two-dimensional system becomes a large sparse ordinary differential equation,

\begin{equation}
\frac{d}{dt}\operatorname{vec}(U)
=
\mathcal{L}\operatorname{vec}(U)
+
\operatorname{vec}(S)
\tag{20.4.6}
\end{equation}

where $\mathcal{L}$ is assembled from Kronecker sums and products of the one-dimensional derivative matrices. This is the key algebraic transition in multidimensional method-of-lines computation: the PDE becomes a structured system of ODEs. Once this step is made, the numerical issues can be studied through the combined properties of the spatial operator $\mathcal{L}$, the time integrator, and the linear algebra required by implicit or semi-implicit methods.

The importance of the Kronecker structure is both theoretical and practical. It explains why multidimensional operators on rectangular grids can often be applied without forming the full matrix explicitly. For example, one can compute $T_xU$ and $UT_y^{\top}$ by applying one-dimensional stencils along coordinate lines. This preserves $O(N)$ storage and $O(N)$ work per explicit step, where,

$$N=n_xn_y$$

in two dimensions, or

$$N=n_xn_yn_z$$

in three dimensions. The same tensor-product structure also underlies alternating-direction methods, line relaxation, Fourier solvers, and many fast Poisson or diffusion solvers on rectangular domains.

The method-of-lines formulation also clarifies the role of modular implementation. A Rust implementation can represent the spatial grid, directional derivative operators, source terms, and time integrator as distinct components. For explicit finite-volume methods, one applies local stencil or face-flux operations. For implicit or split methods, one applies directional line solvers or structured sparse operators. The mathematical structure of (20.4.5) and (20.4.6) therefore translates directly into a software structure in which multidimensional operators are built from lower-dimensional pieces without losing the physical meaning of each contribution.

## 20.4.3. Stability Restrictions for Explicit Multidimensional Schemes

Explicit multidimensional schemes are attractive because each time step is local, inexpensive, and often easy to parallelize. For local stencil methods, the cost per time step is typically $O(N)$, and the memory requirement is also $O(N)$. The main limitation is stability.

Consider the linear multidimensional advection equation:

$$u_t+a_xu_x+a_yu_y+a_zu_z=0 \tag{20.4.7}$$

A standard multidimensional CFL condition has the form:

\begin{equation}
\Delta t
\le
C_{\mathrm{CFL}}
\left(
\frac{|a_x|}{\Delta x}
+
\frac{|a_y|}{\Delta y}
+
\frac{|a_z|}{\Delta z}
\right)^{-1}
\tag{20.4.8}
\end{equation}

In two dimensions, the $z$-term is omitted. This condition states that the combined effect of transport in all coordinate directions must remain within the numerical domain of dependence. It is not enough to check each direction in isolation if all directions are advanced together in one explicit step. The allowable time step decreases when the flow has significant velocity components in multiple directions or when the grid spacing becomes small in any coordinate direction.

For explicit diffusion with isotropic diffusivity $\kappa$,

$$u_t=\kappa\Delta u \tag{20.4.9}$$

the stability restriction is more severe,

\begin{equation}
\Delta t
\le
\frac{1}
{2\kappa\left(
\Delta x^{-2}
+
\Delta y^{-2}
+
\Delta z^{-2}
\right)}
\tag{20.4.10}
\end{equation}

Again, the $z$-term is omitted in two dimensions. The quadratic dependence on grid spacing is the multidimensional version of the parabolic restriction seen in one dimension. If a uniform grid has $\Delta x=\Delta y=\Delta z=h$, then in three dimensions (20.4.10) gives:

$$\Delta t\le \frac{h^2}{6\kappa} \tag{20.4.11}$$

Thus, explicit diffusion becomes extremely expensive on fine multidimensional grids. This explains why explicit methods are often suitable for nonstiff hyperbolic transport but much less attractive for strongly diffusive or mixed advection-diffusion problems.

Modern multidimensional methods improve accuracy and robustness through higher-order reconstructions, strong-stability-preserving Runge-Kutta time integration, and genuinely multidimensional fluxes. However, they do not remove the fundamental stability distinction between hyperbolic and parabolic dynamics. Hyperbolic explicit methods are constrained by wave speeds and cell sizes. Diffusive explicit methods are constrained by squared inverse cell sizes. In practical multidimensional codes, this difference often determines whether the time integrator can remain fully explicit or must be replaced by a split, implicit, semi-implicit, or exponential method (Ling and Tang, 2023; Tremblin et al., 2024; Barsukow et al., 2026).

## 20.4.4. Splitting, ADI, and Multidimensional Applications

A classical way to reduce multidimensional complexity is operator splitting. Suppose the spatial operator can be decomposed as:

$$L=L_x+L_y\tag{20.4.12}$$

in two dimensions, or

$$L=L_x+L_y+L_z \tag{20.4.13}$$

in three dimensions. Here $L_x$, $L_y$, and $L_z$ represent the directional or physical components of the full operator. Strang splitting advances the solution by composing substeps. In two dimensions, one common form is:

\begin{equation}
U^{n+1}
=
\mathcal{S}_x\!\left(\frac{\Delta t}{2}\right)
\mathcal{S}_y(\Delta t)
\mathcal{S}_x\!\left(\frac{\Delta t}{2}\right)
U^n
\tag{20.4.14}
\end{equation}

where $\mathcal{S}_x$ and $\mathcal{S}_y$ are numerical solution operators for the $x$ and $y$ directional subproblems. When the subsolvers are chosen consistently, Strang splitting is globally second-order accurate in time.

The attraction of splitting is that a difficult multidimensional evolution is replaced by a sequence of simpler one-dimensional or directional updates. This can reduce storage, simplify implementation, and allow specialized solvers for individual components. However, splitting is not exact unless the operators commute. In general,

$$L_xL_y\ne L_yL_x \tag{20.4.15}$$

The resulting commutator error becomes important when different physical processes interact strongly, when coefficients vary spatially, or when geometric symmetry must be preserved. Thus, splitting should be regarded as a controlled approximation rather than a purely mechanical simplification.

Recent work has shown that carefully designed split methods can sometimes be recast as standard flux-based finite-volume algorithms. This can reduce memory traffic, simplify extension to higher order, and make the method more compatible with conservative software structures. At the same time, robust multidimensional solvers for magnetohydrodynamics still use relaxation and splitting ideas because they remain computationally effective and physically stable when designed carefully (Bourgeois et al., 2024; Tremblin et al., 2024).

For multidimensional diffusion, an especially important implicit splitting is the alternating-direction implicit method. Consider the two-dimensional diffusion equation with a source term,

$$u_t=\kappa_xu_{xx}+\kappa_yu_{yy}+f \tag{20.4.16}$$

A Crank-Nicolson ADI step may be written in two stages:

\begin{equation}
\left(I-\mu_x T_x\right)U^\ast
=
\left(I+\mu_y T_y\right)U^n
+
\frac{\Delta t}{2}F^n
\tag{20.4.17}
\end{equation}

\begin{equation}
\left(I-\mu_y T_y\right)U^{n+1}
=
\left(I+\mu_x T_x\right)U^\ast
+
\frac{\Delta t}{2}F^{n+1}
\tag{20.4.18}
\end{equation}

where,

\begin{equation}
\begin{aligned}
\mu_x &= \frac{\kappa_x\Delta t}{2\Delta x^2},\\
\mu_y &= \frac{\kappa_y\Delta t}{2\Delta y^2}
\end{aligned}
\tag{20.4.19}
\end{equation}

Each half-step involves only a family of independent tridiagonal line solves. In the first stage, one solves along one coordinate direction; in the second stage, one solves along the other. Thus, on structured grids, the method is implicit but still requires only $O(N)$ work per time step. This is a key idea in multidimensional PDE computation: implicitness does not necessarily imply solving one large dense system. When the operator has separable directional structure, the implicit problem may decompose into many small one-dimensional solves.

ADI-type methods remain important because they address the parabolic time-step restriction without abandoning the efficiency of structured grids. Recent analyses continue to study their stability and convergence for multidimensional advection-diffusion and related PDEs in two and three dimensions (Albasatin et al., 2025; Zhang et al., 2025).

A representative application is contaminant transport in groundwater. If (c(\\mathbf{x},t)) denotes pollutant concentration, a standard advection-diffusion-reaction model is

\begin{equation}
c_t+\mathbf{v}\cdot\nabla c
=
\nabla\cdot(D\nabla c)
-\lambda c
+q
\tag{20.4.20}
\end{equation}

Here $\mathbf{v}$ is the Darcy velocity, $D$ is the hydrodynamic dispersion tensor, $\lambda$ is a decay rate, and $q$ represents injection, leakage, or other source terms. The problem is naturally multidimensional because geological media are heterogeneous and transport may be anisotropic. It is also an initial value problem because one studies the evolution of a contaminant plume after a spill, leak, or remediation action. Advection, diffusion, and reaction may have very different time scales, which makes operator splitting, semi-implicit methods, and line solvers attractive in practice. Recent groundwater simulations continue to use the advection-diffusion-reaction framework as the governing mathematical structure for contaminant migration (Owolabi and Alagoz, 2025).

The main lesson of multidimensional initial value problems is that spatial discretization, time integration, and linear algebra cannot be treated as independent afterthoughts. The method designer must decide whether conservation is required, whether directional splitting is acceptable, whether an unsplit multidimensional flux is needed, whether the problem is stiff enough to require implicit treatment, and whether anisotropy or diffusion suggests line solves. Dimension-by-dimension methods remain attractive because they are simple and cache-friendly. Unsplit multidimensional methods can be more faithful to the PDE when multidirectional wave structure or stationarity preservation matters. ADI-type ideas remain indispensable when diffusion would otherwise impose a prohibitively small explicit time step (Bourgeois et al., 2024; Albasatin et al., 2025; Barsukow et al., 2026).

# 20.5. Fourier and Cyclic Reduction Methods for Boundary Value Problems

Boundary value problems differ fundamentally from initial value problems because the unknown field is determined by constraints imposed throughout the spatial domain and on its boundary, rather than by marching forward from prescribed initial data. After discretization, the central question is therefore algebraic: what structure does the resulting linear system have, and how can that structure be exploited? On rectangular tensor-product domains, many important elliptic boundary value problems lead to sparse, separable systems. Fourier methods and cyclic reduction are powerful precisely because they use this separability rather than treating the discretized problem as an arbitrary sparse matrix.

The model problem is the two-dimensional Poisson equation:

\begin{equation}
-\Delta u = f
\qquad \text{in} \qquad
\Omega=(0,L_x)\times(0,L_y)
\tag{20.5.1}
\end{equation}

together with suitable Dirichlet, Neumann, or periodic boundary conditions. The Poisson equation appears as a model elliptic equation, as a pressure equation in incompressible flow, as a potential equation in electrostatics, and as a subproblem inside many projection, splitting, and implicit time-stepping methods. Because it is solved repeatedly in many time-dependent simulations, its algebraic solution cost can dominate the overall computation.

## 20.5.1. Tensor-Product Discretization and Block Structure

Consider a uniform tensor-product grid on $\Omega$. A standard second-order finite-difference approximation to (20.5.1) gives:

\begin{equation}
\frac{-u_{i-1,j}+2u_{i,j}-u_{i+1,j}}{\Delta x^2}
+
\frac{-u_{i,j-1}+2u_{i,j}-u_{i,j+1}}{\Delta y^2}
=
f_{i,j}
\tag{20.5.2}
\end{equation}

Let $U\in\mathbb{R}^{n_x\times n_y}$ store the unknown grid values, and let $T_x$ and $T_y$ denote the one-dimensional tridiagonal second-difference matrices in the $x$ and $y$-directions. Then the discrete Poisson equation can be written compactly as the Sylvester-type matrix equation:

\begin{equation}
\frac{1}{\Delta x^2}T_xU
+
\frac{1}{\Delta y^2}UT_y^{\top}
=
F
\tag{20.5.3}
\end{equation}

This expression is important because it retains the tensor-product structure of the grid. The two-dimensional operator is not an unstructured object; it is built from one-dimensional second-difference operators acting in separate directions.

After vectorizing $U$, the same system becomes:

\begin{equation}
A\,\operatorname{vec}(U)
=
\operatorname{vec}(F)
\tag{20.5.4}
\end{equation}

where,

\begin{equation}
A
=
\frac{1}{\Delta x^2} I_{n_y}\otimes T_x
+
\frac{1}{\Delta y^2} T_y\otimes I_{n_x}
\tag{20.5.5}
\end{equation}

Here $\otimes$ denotes the Kronecker product. Equation (20.5.5) is the algebraic form of separability: the two-dimensional operator is a Kronecker sum of one-dimensional operators. This structure is the reason that fast solvers are possible on rectangular domains.

If the unknowns are ordered by vertical grid lines or horizontal grid lines, the matrix $A$ has block tridiagonal form,

\begin{equation}
A=
\begin{bmatrix}
B & -\Delta y^{-2}I &        &        & 0\\
-\Delta y^{-2}I & B & -\Delta y^{-2}I &        &  \\
        & \ddots & \ddots & \ddots &  \\
        &        & -\Delta y^{-2}I & B & -\Delta y^{-2}I\\
0       &        &        & -\Delta y^{-2}I & B
\end{bmatrix}
\tag{20.5.6}
\end{equation}

with

$$B=\Delta x^{-2}T_x+2\Delta y^{-2}I \tag{20.5.7}$$

This block structure is sparse, regular, and highly exploitable. A general sparse direct factorization can solve the system, but it may introduce substantial fill-in and memory cost. Fourier methods and cyclic reduction avoid this by using the tensor-product and block-tridiagonal structure directly.

### Rust Implementation

Following the discussion in Subsection 20.5.1 on tensor-product discretization, Kronecker-sum structure, and sparse block-tridiagonal systems, Program 20.5.1 provides a practical implementation of sparse matrix assembly for the two-dimensional Poisson equation on a rectangular tensor-product grid. The program directly implements the finite-difference discretization introduced in equations (20.5.2)–(20.5.7) and demonstrates how the multidimensional elliptic operator can be constructed systematically from one-dimensional second-difference contributions in the coordinate directions. Rather than treating the Poisson operator as an arbitrary sparse matrix, the implementation preserves the tensor-product structure emphasized throughout the subsection. The resulting sparse matrix exhibits the characteristic five-point stencil and block-tridiagonal structure associated with separable elliptic operators on rectangular domains. The program also verifies the correctness of the assembled operator by applying it to a known analytic test function and comparing the discrete result against the corresponding continuous Poisson operator.

At the core of the implementation is the `Grid2D` structure, which defines the tensor-product spatial discretization used throughout the computation. The structure stores the number of interior grid points in the $x$ and $y$-directions, the domain dimensions, and the corresponding mesh spacings $\Delta x$ and $\Delta y$. The `index` function maps a two-dimensional grid coordinate $(i,j)$ into the corresponding vectorized global index associated with equation (20.5.4), while the `point` function computes the physical coordinates of an interior grid node. This indexing structure reflects the vectorization procedure used to convert the matrix equation (20.5.3) into the sparse linear system form of equation (20.5.5).

The sparse algebraic structure of the Poisson operator is represented through the `SparseMatrix` structure. Rather than storing a dense matrix, the implementation stores only the nonzero entries associated with the finite-difference stencil. Each row therefore contains only the neighboring couplings arising from the tensor-product second-difference operators. The functions `add_entry`, `matvec`, `nonzeros`, and `row_entries` provide the basic sparse linear algebra operations needed to construct and inspect the operator. This sparse representation reflects the block-tridiagonal structure described in equations (20.5.6)–(20.5.7), where each interior grid point couples only to its nearest neighbors in the coordinate directions.

The tensor-product Poisson operator itself is assembled through the function `assemble_poisson_matrix`. The function constructs the sparse finite-difference discretization corresponding to equation (20.5.2) by adding the diagonal and nearest-neighbor contributions associated with the discrete Laplacian. The diagonal coefficient combines the contributions from the $x$ and $y$-direction second-difference operators, while the off-diagonal entries represent the nearest-neighbor stencil couplings. The resulting sparse matrix is precisely the Kronecker-sum operator described by equation (20.5.5), although it is assembled directly through geometric neighbor relationships rather than through explicit Kronecker-product construction. This approach preserves the same algebraic structure while remaining computationally transparent.

The functions `test_function` and `analytic_negative_laplacian` provide an exact verification problem for the assembled operator. The chosen polynomial test function satisfies homogeneous Dirichlet boundary conditions, while its continuous negative Laplacian is known analytically. By evaluating the discrete operator on the sampled test function and comparing the result against the exact continuous forcing term, the implementation verifies that the assembled sparse matrix correctly represents the finite-difference Poisson operator.

The functions `build_sample_solution` and `build_exact_rhs` generate the vectorized numerical solution and forcing vectors associated with the verification problem. These vectors correspond directly to the vectorized representation appearing in equation (20.5.4). The function `max_abs_difference` then computes the maximum pointwise difference between the discrete matrix-vector product and the exact analytic forcing term, thereby providing a quantitative measure of discretization consistency and assembly correctness.

The function `print_selected_rows` provides an explicit illustration of the sparse block structure of the assembled operator. Selected matrix rows corresponding to corner, edge, and interior grid points are printed together with their nonzero stencil entries. This diagnostic clearly reveals the five-point coupling structure characteristic of the two-dimensional Poisson discretization and illustrates how the tensor-product finite-difference operator naturally produces sparse block-tridiagonal matrices.

The `main` function coordinates the complete sparse matrix assembly and verification process. It constructs the tensor-product grid, assembles the sparse Poisson operator, evaluates the matrix on the analytic test solution, and compares the result against the exact continuous forcing term. Diagnostic output reports the matrix dimension, total number of nonzero entries, average row sparsity, and representative stencil rows. The resulting verification error is observed to be near machine precision, confirming that the assembled sparse matrix correctly reproduces the finite-difference discretization of the Poisson equation. Altogether, the implementation provides a concrete demonstration of the tensor-product and block-structured algebraic framework developed throughout Subsection 20.5.1.

```rust
// Program 20.5.1. Tensor-Product Sparse Matrix Assembly for the Two-Dimensional Poisson Equation
//
// Problem statement:
// Assemble the finite-difference matrix for the two-dimensional Poisson equation
//
//     -Delta u = f
//
// on a rectangular domain with homogeneous Dirichlet boundary conditions.
// The implementation follows the tensor-product structure in equations
// (20.5.2)-(20.5.7), showing how the two-dimensional operator is built from
// one-dimensional second-difference matrices and appears as a sparse
// block-tridiagonal system.

#[derive(Clone)]
struct Grid2D {
    nx: usize,
    ny: usize,
    lx: f64,
    ly: f64,
    dx: f64,
    dy: f64,
}

impl Grid2D {
    fn new(nx: usize, ny: usize, lx: f64, ly: f64) -> Self {
        assert!(nx >= 2, "There must be at least two interior x-points.");
        assert!(ny >= 2, "There must be at least two interior y-points.");
        assert!(lx > 0.0, "The x-domain length must be positive.");
        assert!(ly > 0.0, "The y-domain length must be positive.");

        let dx = lx / (nx as f64 + 1.0);
        let dy = ly / (ny as f64 + 1.0);

        Self {
            nx,
            ny,
            lx,
            ly,
            dx,
            dy,
        }
    }

    fn index(&self, i: usize, j: usize) -> usize {
        assert!(i < self.nx);
        assert!(j < self.ny);

        j * self.nx + i
    }

    fn point(&self, i: usize, j: usize) -> (f64, f64) {
        assert!(i < self.nx);
        assert!(j < self.ny);

        let x = (i as f64 + 1.0) * self.dx;
        let y = (j as f64 + 1.0) * self.dy;

        (x, y)
    }

    fn number_of_unknowns(&self) -> usize {
        self.nx * self.ny
    }
}

#[derive(Clone)]
struct SparseMatrix {
    nrows: usize,
    ncols: usize,
    rows: Vec<Vec<(usize, f64)>>,
}

impl SparseMatrix {
    fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            rows: vec![Vec::new(); nrows],
        }
    }

    fn add_entry(&mut self, row: usize, col: usize, value: f64) {
        assert!(row < self.nrows);
        assert!(col < self.ncols);

        self.rows[row].push((col, value));
    }

    fn matvec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.ncols);

        let mut y = vec![0.0_f64; self.nrows];

        for row in 0..self.nrows {
            let mut sum = 0.0_f64;

            for &(col, value) in &self.rows[row] {
                sum += value * x[col];
            }

            y[row] = sum;
        }

        y
    }

    fn nonzeros(&self) -> usize {
        self.rows.iter().map(|row| row.len()).sum()
    }

    fn row_entries(&self, row: usize) -> &[(usize, f64)] {
        assert!(row < self.nrows);
        &self.rows[row]
    }
}

fn assemble_poisson_matrix(grid: &Grid2D) -> SparseMatrix {
    let n = grid.number_of_unknowns();

    let inv_dx2 = 1.0 / (grid.dx * grid.dx);
    let inv_dy2 = 1.0 / (grid.dy * grid.dy);

    let mut a = SparseMatrix::new(n, n);

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let row = grid.index(i, j);

            // Diagonal contribution from T_x / dx^2 and T_y / dy^2.
            a.add_entry(row, row, 2.0 * inv_dx2 + 2.0 * inv_dy2);

            // x-direction neighbours.
            if i > 0 {
                let col = grid.index(i - 1, j);
                a.add_entry(row, col, -inv_dx2);
            }

            if i + 1 < grid.nx {
                let col = grid.index(i + 1, j);
                a.add_entry(row, col, -inv_dx2);
            }

            // y-direction neighbours.
            if j > 0 {
                let col = grid.index(i, j - 1);
                a.add_entry(row, col, -inv_dy2);
            }

            if j + 1 < grid.ny {
                let col = grid.index(i, j + 1);
                a.add_entry(row, col, -inv_dy2);
            }
        }
    }

    a
}

fn test_function(x: f64, y: f64) -> f64 {
    x * (1.0 - x) * y * (1.0 - y)
}

fn analytic_negative_laplacian(x: f64, y: f64) -> f64 {
    2.0 * y * (1.0 - y) + 2.0 * x * (1.0 - x)
}

fn build_sample_solution(grid: &Grid2D) -> Vec<f64> {
    let mut u = vec![0.0_f64; grid.number_of_unknowns()];

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let (x, y) = grid.point(i, j);
            let idx = grid.index(i, j);

            u[idx] = test_function(x, y);
        }
    }

    u
}

fn build_exact_rhs(grid: &Grid2D) -> Vec<f64> {
    let mut f = vec![0.0_f64; grid.number_of_unknowns()];

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let (x, y) = grid.point(i, j);
            let idx = grid.index(i, j);

            f[idx] = analytic_negative_laplacian(x, y);
        }
    }

    f
}

fn max_abs_difference(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn print_selected_rows(matrix: &SparseMatrix, grid: &Grid2D) {
    let selected = [
        grid.index(0, 0),
        grid.index(grid.nx / 2, grid.ny / 2),
        grid.index(grid.nx - 1, grid.ny - 1),
    ];

    println!("Selected Sparse Matrix Rows");
    println!("---------------------------");

    for row in selected {
        print!("row {:>3}: ", row);

        for &(col, value) in matrix.row_entries(row) {
            print!("({:>3}, {:>10.4}) ", col, value);
        }

        println!();
    }

    println!();
}

fn main() {
    let grid = Grid2D::new(8, 6, 1.0, 1.0);

    let matrix = assemble_poisson_matrix(&grid);

    let u = build_sample_solution(&grid);
    let discrete_rhs = matrix.matvec(&u);
    let exact_rhs = build_exact_rhs(&grid);

    let max_error = max_abs_difference(&discrete_rhs, &exact_rhs);

    println!("Tensor-Product Sparse Poisson Matrix Assembly");
    println!("============================================");
    println!();

    println!("Grid Parameters");
    println!("---------------");
    println!("Interior x-points        = {}", grid.nx);
    println!("Interior y-points        = {}", grid.ny);
    println!("Domain length Lx         = {:.6}", grid.lx);
    println!("Domain length Ly         = {:.6}", grid.ly);
    println!("Grid spacing dx          = {:.6}", grid.dx);
    println!("Grid spacing dy          = {:.6}", grid.dy);
    println!("Total unknowns           = {}", grid.number_of_unknowns());
    println!();

    println!("Sparse Matrix Structure");
    println!("-----------------------");
    println!("Matrix dimension         = {} x {}", matrix.nrows, matrix.ncols);
    println!("Number of nonzeros       = {}", matrix.nonzeros());
    println!(
        "Average nonzeros per row = {:.3}",
        matrix.nonzeros() as f64 / matrix.nrows as f64
    );
    println!();

    print_selected_rows(&matrix, &grid);

    println!("Stencil Verification");
    println!("--------------------");
    println!(
        "The test function is u(x,y)=x(1-x)y(1-y), which satisfies homogeneous"
    );
    println!("Dirichlet boundary conditions.");
    println!(
        "Maximum |A u - f_exact| = {:.6e}",
        max_error
    );
}
```

Program 20.5.1 demonstrates how the tensor-product structure of multidimensional elliptic PDE discretizations naturally leads to sparse block-structured linear systems. Rather than treating the two-dimensional Poisson operator as a generic sparse matrix, the implementation explicitly exploits the separability of the coordinate directions described by equations (20.5.3)–(20.5.5). The resulting matrix exhibits the characteristic five-point stencil and block-tridiagonal structure that forms the foundation for many fast elliptic solvers.

The numerical verification confirms that the assembled sparse matrix correctly represents the finite-difference approximation of the continuous Poisson operator. The near-machine-precision agreement between the discrete matrix-vector product and the analytic forcing term demonstrates that the tensor-product stencil has been assembled consistently and that the vectorized sparse representation preserves the expected multidimensional coupling structure.

The sparse structure observed in the selected matrix rows also illustrates the major computational advantage of tensor-product discretization. Each interior grid point couples only to its immediate neighbors, producing a highly sparse operator with predictable structure and low memory cost. This regularity is precisely what enables efficient Fourier methods, cyclic reduction strategies, multigrid algorithms, and structured sparse direct solvers later in the chapter.

The modular organization of the implementation naturally supports extension to more advanced elliptic discretizations. Variable coefficients, anisotropic diffusion operators, higher-order finite differences, nonuniform meshes, and multidimensional sparse iterative solvers can all be incorporated while preserving the same tensor-product assembly philosophy. Consequently, Program 20.5.1 provides a foundational computational framework for the broader study of structured elliptic boundary value solvers developed throughout Section 20.5.

## 20.5.2. Fourier and Transform-Based Solvers

Fourier methods exploit the fact that uniform-grid difference operators with compatible boundary conditions are diagonalized by trigonometric transforms. For homogeneous Dirichlet boundary conditions in the $y$-direction, the matrix $T_y$ is diagonalized by the discrete sine transform,

$$T_y=S\Lambda S^{\top} \tag{20.5.8}$$

where,

\begin{equation}
S_{\ell q}
=
\sqrt{\frac{2}{n_y+1}}\,
\sin\!\left(\frac{\ell q\pi}{n_y+1}\right)
\tag{20.5.9}
\end{equation}

and

\begin{equation}
\Lambda_{qq}
=
2-2\cos\!\left(\frac{q\pi}{n_y+1}\right)
\tag{20.5.10}
\end{equation}

Substituting (20.5.8) into (20.5.3) and transforming in the $y$-direction decouples the two-dimensional problem into a family of independent one-dimensional problems. If,

$$\widehat{U}=US,\qquad\widehat{F}=FS \tag{20.5.11}$$

then each transformed mode satisfies:

\begin{equation}
\left(
\frac{1}{\Delta x^2}T_x
+
\frac{\lambda_q}{\Delta y^2}I
\right)
\widehat{u}^{(q)}
=
\widehat{f}^{(q)},
\qquad
q=1,\ldots,n_y
\tag{20.5.12}
\end{equation}

Each equation in (20.5.12) is tridiagonal. Thus the original two-dimensional problem has been reduced to $n_y$ independent one-dimensional tridiagonal solves, plus the cost of forward and inverse sine transforms. On an $n_x\times n_y$ grid, the total cost is typically,

$$O(N\log n_y)+O(N),\qquad N=n_xn_y \tag{20.5.13}$$

If diagonalization is carried out in both coordinate directions, the transformed system becomes diagonal mode by mode, leading to the familiar $O(N\log N)$ complexity associated with FFT, discrete sine transform, or discrete cosine transform solvers.

The choice of transform is determined by the boundary condition. Homogeneous Dirichlet boundaries correspond naturally to sine transforms. Periodic boundaries are diagonalized by complex Fourier transforms. Homogeneous Neumann boundaries lead to cosine transforms. When a zero eigenvalue is present, as in pure Neumann problems, the right-hand side must satisfy the usual solvability condition that its mean be zero. Otherwise the Poisson problem is inconsistent.

Boundary data must be handled carefully. The transform diagonalizes the homogeneous operator, not arbitrary boundary values. Therefore, nonhomogeneous boundary conditions are usually treated by a lifting decomposition,

$$u=v+w \tag{20.5.14}$$

The function $w$ is chosen to satisfy the nonhomogeneous boundary conditions, while $v$ satisfies homogeneous boundary conditions. The transformed solver is then applied to the homogeneous problem for $v$, with a modified right-hand side. On rectangular domains this lifting step is usually inexpensive, but it is conceptually essential. Without it, a separable PDE may appear nonseparable at the algebraic level because the boundary data have not been separated from the homogeneous operator.

Recent developments continue to extend transform-based Poisson solvers. Hybrid discrete-sine-transform accelerated finite-difference solvers have been developed for two- and three-dimensional Dirichlet problems, generalized eigendecomposition approaches support mixed boundary conditions, and spectral-ADI solvers achieve optimal complexity for smooth solutions on separable domains (Pei and Tong, 2025; Wu, 2025; Qin, 2025). These methods all exploit the same underlying fact: when the domain, coefficients, and boundary conditions preserve separability, the discrete operator can be diagonalized or nearly diagonalized by coordinate-wise transforms.

### Rust Implementation

Following the discussion in Subsection 20.5.2 on transform-based Poisson solvers, tensor-product separability, and diagonalization by discrete sine transforms, Program 20.5.2 provides a practical implementation of a two-dimensional Poisson solver using a discrete-sine-transform reduction in the $y$-direction combined with tridiagonal solves in the $x$-direction. The implementation directly follows the transform formulation introduced in equations (20.5.8)–(20.5.13), where the discrete sine transform diagonalizes the one-dimensional second-difference operator associated with homogeneous Dirichlet boundary conditions. Rather than solving the full two-dimensional sparse system simultaneously, the program transforms the problem into independent one-dimensional tridiagonal systems corresponding to individual Fourier modes. This reduction illustrates the fundamental computational advantage of separability: multidimensional elliptic problems on rectangular domains can often be decomposed into families of lower-dimensional problems whose algebraic structure is substantially simpler. The program also verifies the numerical solution against an exact analytic solution of the Poisson equation, thereby demonstrating both the correctness and efficiency of transform-based elliptic solvers.

At the core of the implementation is the `Grid2D` structure, which defines the tensor-product discretization used for the Poisson problem. The structure stores the number of interior grid points in the coordinate directions, the physical dimensions of the computational domain, and the corresponding mesh spacings $\Delta x$ and $\Delta y$. The `point` function computes the physical location of each interior node and is used to evaluate both the exact solution and the forcing term associated with the verification problem. This tensor-product grid structure reflects the separable discretization framework introduced in equations (20.5.2)–(20.5.5).

The multidimensional solution and right-hand side fields are represented through the `Matrix2D` structure. The structure stores the grid values in a contiguous one-dimensional vector while providing logical two-dimensional indexing through the `index`, `get`, and `set` functions. This layout preserves efficient memory access while maintaining a natural tensor-product interpretation of the numerical solution. The matrix structure is used throughout the transform operations, mode solves, and residual evaluation procedures.

The discrete sine transform introduced in equations (20.5.8)–(20.5.10) is implemented through the functions `sine_basis_entry`, `dst_y`, and `inverse_dst_y`. The function `sine_basis_entry` evaluates the orthonormal sine basis defined in equation (20.5.9), while `dst_y` applies the discrete sine transform in the $y$-direction. Because the sine basis is orthonormal, the inverse transform is identical to the forward transform, which is reflected in the implementation of `inverse_dst_y`. The transform diagonalizes the discrete second-difference operator associated with homogeneous Dirichlet boundary conditions, thereby converting the coupled two-dimensional Poisson problem into independent one-dimensional mode equations.

The eigenvalues of the transformed second-difference operator are computed through the function `ty_eigenvalue`, which directly implements the eigenvalue formula of equation (20.5.10). These eigenvalues determine the spectral shift appearing in each transformed tridiagonal system of equation (20.5.12). Each transformed Fourier mode therefore corresponds to an independent one-dimensional elliptic problem with its own shifted tridiagonal operator.

The tridiagonal linear systems arising from the transformed mode equations are solved using the function `thomas_solve`, which implements the Thomas algorithm for banded systems. The method performs forward elimination followed by backward substitution while exploiting the tridiagonal structure of the transformed operators. Because each transformed mode is independent, the original two-dimensional Poisson problem is reduced to a sequence of one-dimensional tridiagonal solves, precisely as described by equation (20.5.12).

The complete transform-based Poisson solver is implemented through the function `solve_poisson_dst_y`. The function first applies the discrete sine transform to the right-hand side, thereby transforming the problem into Fourier mode space. For each transformed mode $q$, the corresponding tridiagonal operator is assembled using the transformed eigenvalue $\lambda_q$, and the resulting one-dimensional system is solved using the Thomas algorithm. After all modes have been computed, the inverse discrete sine transform reconstructs the physical-space solution. This procedure directly realizes the separability framework developed throughout Subsection 20.5.2 and illustrates how the two-dimensional elliptic problem decomposes into independent lower-dimensional systems.

The functions `exact_solution` and `exact_rhs` define an analytic verification problem for the Poisson solver. The exact solution satisfies homogeneous Dirichlet boundary conditions naturally, making it compatible with the discrete sine transform framework. The function `build_rhs` samples the forcing term on the computational grid, while `max_abs_error` computes the maximum pointwise difference between the numerical and exact solutions. The function `discrete_residual_norm` evaluates the residual of the discrete finite-difference Poisson equation, thereby verifying that the transformed solver accurately solves the discrete elliptic system.

The `main` function coordinates the complete transform-based Poisson solve and verification procedure. It constructs the computational grid, assembles the forcing term, computes the transformed solution, and evaluates both the pointwise solution error and the discrete residual norm. Diagnostic output reports the transform direction, the number of independent mode solves, and representative numerical solution values. The small residual norm confirms that the transformed solver accurately solves the discrete finite-difference system, while the pointwise solution error reflects the difference between the continuous PDE and its discrete finite-difference approximation. Altogether, the implementation provides a concrete realization of the transform-diagonalization framework developed throughout Subsection 20.5.2.

```rust
// Program 20.5.2. Discrete-Sine-Transform Poisson Solver with Tridiagonal Mode Solves
//
// Problem statement:
// Solve the two-dimensional Poisson equation
//
//     -Delta u = f
//
// on a rectangular domain with homogeneous Dirichlet boundary conditions.
// The method applies the discrete sine transform in the y-direction,
// diagonalizes T_y, and reduces the two-dimensional problem to independent
// tridiagonal systems in the x-direction, as described in equations
// (20.5.8)-(20.5.13).

use std::f64::consts::PI;

#[derive(Clone)]
struct Grid2D {
    nx: usize,
    ny: usize,
    lx: f64,
    ly: f64,
    dx: f64,
    dy: f64,
}

impl Grid2D {
    fn new(nx: usize, ny: usize, lx: f64, ly: f64) -> Self {
        assert!(nx >= 2, "There must be at least two interior x-points.");
        assert!(ny >= 2, "There must be at least two interior y-points.");
        assert!(lx > 0.0, "The x-domain length must be positive.");
        assert!(ly > 0.0, "The y-domain length must be positive.");

        let dx = lx / (nx as f64 + 1.0);
        let dy = ly / (ny as f64 + 1.0);

        Self { nx, ny, lx, ly, dx, dy }
    }

    fn point(&self, i: usize, j: usize) -> (f64, f64) {
        let x = (i as f64 + 1.0) * self.dx;
        let y = (j as f64 + 1.0) * self.dy;
        (x, y)
    }
}

#[derive(Clone)]
struct Matrix2D {
    nx: usize,
    ny: usize,
    data: Vec<f64>,
}

impl Matrix2D {
    fn new(nx: usize, ny: usize) -> Self {
        Self {
            nx,
            ny,
            data: vec![0.0; nx * ny],
        }
    }

    fn index(&self, i: usize, j: usize) -> usize {
        assert!(i < self.nx);
        assert!(j < self.ny);
        j * self.nx + i
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        self.data[self.index(i, j)]
    }

    fn set(&mut self, i: usize, j: usize, value: f64) {
        let idx = self.index(i, j);
        self.data[idx] = value;
    }
}

#[derive(Clone)]
struct TridiagonalMatrix {
    lower: Vec<f64>,
    diag: Vec<f64>,
    upper: Vec<f64>,
}

impl TridiagonalMatrix {
    fn new(n: usize, lower_value: f64, diag_value: f64, upper_value: f64) -> Self {
        Self {
            lower: vec![lower_value; n - 1],
            diag: vec![diag_value; n],
            upper: vec![upper_value; n - 1],
        }
    }
}

fn sine_basis_entry(ell: usize, q: usize, n: usize) -> f64 {
    let scale = (2.0 / (n as f64 + 1.0)).sqrt();
    let angle = ((ell + 1) as f64 * (q + 1) as f64 * PI) / (n as f64 + 1.0);

    scale * angle.sin()
}

fn dst_y(input: &Matrix2D) -> Matrix2D {
    let mut output = Matrix2D::new(input.nx, input.ny);

    for i in 0..input.nx {
        for q in 0..input.ny {
            let mut sum = 0.0_f64;

            for ell in 0..input.ny {
                sum += input.get(i, ell) * sine_basis_entry(ell, q, input.ny);
            }

            output.set(i, q, sum);
        }
    }

    output
}

// Because the sine basis is orthonormal, the inverse transform is the same
// operation as the forward transform.
fn inverse_dst_y(input: &Matrix2D) -> Matrix2D {
    dst_y(input)
}

fn ty_eigenvalue(q: usize, ny: usize) -> f64 {
    2.0 - 2.0 * (((q + 1) as f64 * PI) / (ny as f64 + 1.0)).cos()
}

fn thomas_solve(matrix: &TridiagonalMatrix, rhs: &[f64]) -> Vec<f64> {
    let n = matrix.diag.len();

    assert_eq!(rhs.len(), n);
    assert_eq!(matrix.lower.len(), n - 1);
    assert_eq!(matrix.upper.len(), n - 1);

    let mut c_prime = vec![0.0_f64; n - 1];
    let mut d_prime = vec![0.0_f64; n];

    c_prime[0] = matrix.upper[0] / matrix.diag[0];
    d_prime[0] = rhs[0] / matrix.diag[0];

    for i in 1..n {
        let denom = matrix.diag[i] - matrix.lower[i - 1] * c_prime[i - 1];

        if i < n - 1 {
            c_prime[i] = matrix.upper[i] / denom;
        }

        d_prime[i] = (rhs[i] - matrix.lower[i - 1] * d_prime[i - 1]) / denom;
    }

    let mut solution = vec![0.0_f64; n];
    solution[n - 1] = d_prime[n - 1];

    for i in (0..n - 1).rev() {
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
    }

    solution
}

fn solve_poisson_dst_y(grid: &Grid2D, rhs: &Matrix2D) -> Matrix2D {
    let rhs_hat = dst_y(rhs);

    let inv_dx2 = 1.0 / (grid.dx * grid.dx);
    let inv_dy2 = 1.0 / (grid.dy * grid.dy);

    let mut solution_hat = Matrix2D::new(grid.nx, grid.ny);

    for q in 0..grid.ny {
        let lambda_q = ty_eigenvalue(q, grid.ny);

        let lower = -inv_dx2;
        let diag = 2.0 * inv_dx2 + lambda_q * inv_dy2;
        let upper = -inv_dx2;

        let matrix_q = TridiagonalMatrix::new(grid.nx, lower, diag, upper);

        let mut rhs_mode = vec![0.0_f64; grid.nx];

        for i in 0..grid.nx {
            rhs_mode[i] = rhs_hat.get(i, q);
        }

        let solution_mode = thomas_solve(&matrix_q, &rhs_mode);

        for i in 0..grid.nx {
            solution_hat.set(i, q, solution_mode[i]);
        }
    }

    inverse_dst_y(&solution_hat)
}

fn exact_solution(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (2.0 * PI * y).sin()
}

fn exact_rhs(x: f64, y: f64) -> f64 {
    5.0 * PI * PI * exact_solution(x, y)
}

fn build_rhs(grid: &Grid2D) -> Matrix2D {
    let mut rhs = Matrix2D::new(grid.nx, grid.ny);

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let (x, y) = grid.point(i, j);
            rhs.set(i, j, exact_rhs(x, y));
        }
    }

    rhs
}

fn max_abs_error(grid: &Grid2D, numerical: &Matrix2D) -> f64 {
    let mut max_error = 0.0_f64;

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let (x, y) = grid.point(i, j);
            let error = (numerical.get(i, j) - exact_solution(x, y)).abs();

            max_error = max_error.max(error);
        }
    }

    max_error
}

fn discrete_residual_norm(grid: &Grid2D, u: &Matrix2D, rhs: &Matrix2D) -> f64 {
    let inv_dx2 = 1.0 / (grid.dx * grid.dx);
    let inv_dy2 = 1.0 / (grid.dy * grid.dy);

    let mut sum = 0.0_f64;

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let center = u.get(i, j);

            let left = if i > 0 { u.get(i - 1, j) } else { 0.0 };
            let right = if i + 1 < grid.nx { u.get(i + 1, j) } else { 0.0 };
            let down = if j > 0 { u.get(i, j - 1) } else { 0.0 };
            let up = if j + 1 < grid.ny { u.get(i, j + 1) } else { 0.0 };

            let discrete_negative_laplacian =
                (2.0 * center - left - right) * inv_dx2
                    + (2.0 * center - down - up) * inv_dy2;

            let residual = discrete_negative_laplacian - rhs.get(i, j);
            sum += residual * residual;
        }
    }

    sum.sqrt()
}

fn main() {
    let grid = Grid2D::new(32, 32, 1.0, 1.0);

    let rhs = build_rhs(&grid);
    let numerical_solution = solve_poisson_dst_y(&grid, &rhs);

    let max_error = max_abs_error(&grid, &numerical_solution);
    let residual_norm = discrete_residual_norm(&grid, &numerical_solution, &rhs);

    println!("Discrete-Sine-Transform Poisson Solver");
    println!("=====================================");
    println!();

    println!("Grid Parameters");
    println!("---------------");
    println!("Interior x-points        = {}", grid.nx);
    println!("Interior y-points        = {}", grid.ny);
    println!("Domain length Lx         = {:.6}", grid.lx);
    println!("Domain length Ly         = {:.6}", grid.ly);
    println!("Grid spacing dx          = {:.8}", grid.dx);
    println!("Grid spacing dy          = {:.8}", grid.dy);
    println!();

    println!("Solver Structure");
    println!("----------------");
    println!("Transform direction      = y");
    println!("Boundary conditions       = homogeneous Dirichlet");
    println!("Mode solves              = {} independent tridiagonal systems", grid.ny);
    println!();

    println!("Verification");
    println!("------------");
    println!("Exact solution            = sin(pi x) sin(2 pi y)");
    println!("Maximum pointwise error   = {:.6e}", max_error);
    println!("Discrete residual norm    = {:.6e}", residual_norm);
    println!();

    println!("Sample Solution Values");
    println!("----------------------");
    println!("{:>8} {:>8} {:>14} {:>14} {:>18}", "i", "j", "x", "y", "u_num");

    for &(i, j) in &[(8, 8), (16, 16), (24, 24)] {
        let (x, y) = grid.point(i, j);
        println!(
            "{:>8} {:>8} {:>14.6} {:>14.6} {:>18.10}",
            i,
            j,
            x,
            y,
            numerical_solution.get(i, j)
        );
    }
}
```

Program 20.5.2 demonstrates how separability and transform diagonalization can dramatically simplify the solution of multidimensional elliptic boundary value problems. By applying the discrete sine transform in one coordinate direction, the coupled two-dimensional Poisson problem is reduced to a family of independent one-dimensional tridiagonal systems. This reduction exploits the tensor-product structure of the finite-difference operator and avoids the need to solve the full sparse multidimensional system simultaneously.

The numerical verification confirms that the transform-based solver accurately satisfies the discrete Poisson equation. The residual norm is near machine precision, demonstrating that the transformed tridiagonal systems have been solved correctly and that the inverse transform reconstructs the numerical solution consistently. The remaining pointwise error arises primarily from the finite-difference discretization itself rather than from the transform solver.

The implementation also illustrates the fundamental relationship between boundary conditions and transform choice. Because the problem uses homogeneous Dirichlet boundary conditions, the discrete sine transform diagonalizes the second-difference operator naturally. Different boundary conditions would lead to different transforms: periodic boundaries correspond to Fourier transforms, while homogeneous Neumann conditions lead naturally to cosine transforms. Thus, transform-based elliptic solvers are fundamentally tied to the compatibility between operator structure and boundary conditions.

The modular structure of the implementation naturally supports extension to more advanced transform-based methods. Fast Fourier transform libraries, multidimensional transform diagonalization, mixed boundary conditions, spectral-ADI methods, tensor-product preconditioners, and generalized eigendecomposition techniques can all be incorporated while preserving the same separability framework. Consequently, Program 20.5.2 provides a foundational computational example for the broader class of transform-accelerated elliptic solvers developed throughout Section 20.5.

## 20.5.3. Cyclic Reduction and Recursive Block Elimination

Cyclic reduction achieves fast solution of structured elliptic systems by a different mechanism. Instead of diagonalizing a coordinate direction, it recursively eliminates alternating block rows from a block tridiagonal system. Consider the block system:

$$E u_{j-1}+B u_j+E u_{j+1}=g_j,\qquad j=1,\ldots,m \tag{20.5.15}$$

where each $u_j$ is itself a vector of unknowns along a grid line. This form appears naturally when a two-dimensional Poisson problem is ordered line by line.

The cyclic reduction idea is to eliminate, for example, all odd-indexed unknowns and obtain a reduced system involving only even-indexed unknowns. Solving the odd equations for the odd unknowns and substituting into the neighbouring even equations yields a new block tridiagonal system of the same form,

\begin{equation}
\widetilde{E}u_{j-2}
+
\widetilde{B}u_j
+
\widetilde{E}u_{j+2}
=
\widetilde{g}_j
\tag{20.5.16}
\end{equation}

The reduced coefficients are:

\begin{equation}
\widetilde{E}
=
-E B^{-1}E
\tag{20.5.17}
\end{equation}

$$\widetilde{B} = B-2EB^{-1}E \tag{20.5.18}$$

and,

$$\widetilde{g}_j = g_j - EB^{-1}\left(g_{j-1}+g_{j+1}\right) \tag{20.5.19}$$

This reduction is repeated recursively until the remaining system is small enough to solve directly. A backward reconstruction then recovers the eliminated unknowns. The key point is that each stage halves the number of block rows. Therefore, the number of reduction levels is $O(\log m)$.

Cyclic reduction has several important computational features. First, its parallel depth is low because many eliminations at the same level are independent. Second, it is efficient when the action of $B^{-1}$ is cheap, for example when $B$ is tridiagonal, diagonalizable by a transform, or reused for many right-hand sides. Third, cyclic reduction does not require full periodicity or complete transform diagonalization. It is a direct recursive Schur-complement method for structured sparse systems.

Fourier methods and cyclic reduction are therefore complementary. Fourier methods rely on eigenvectors of translation-invariant difference operators and are extremely efficient on simple constant-coefficient rectangles with compatible boundary conditions. Cyclic reduction relies on block structure and recursive elimination. It is more structured than general sparse elimination but more flexible than a pure transform method. It is especially useful when only one coordinate direction is transform-compatible, when mixed boundary conditions are present, or when low synchronization depth is important on parallel hardware.

Modern high-performance implementations continue to make cyclic reduction relevant. Recent distributed-memory tridiagonal solver work emphasizes that many PDE discretizations repeatedly reduce to tridiagonal or block-tridiagonal solves, and that communication-avoiding data layout can be as important as the mathematical elimination process itself. Similarly, extreme-scale finite-difference solvers for wall-bounded turbulence combine pencil decompositions with reworked parallel cyclic reduction strategies to keep implicit wall-normal diffusion and Poisson or Helmholtz solves efficient on modern GPU supercomputers (Akkurt et al., 2025; Diez Sanhueza et al., 2025).

### Rust Implementation

Following the discussion in Subsection 20.5.3 on recursive block elimination and cyclic reduction methods for structured elliptic systems, Program 20.5.3 provides a practical implementation of cyclic reduction for a structured tridiagonal Poisson system. The implementation demonstrates the recursive elimination strategy described in equations (20.5.15)–(20.5.19), where alternating unknowns are eliminated systematically to produce progressively smaller reduced systems. Rather than diagonalizing the operator through Fourier modes as in transform-based solvers, cyclic reduction exploits the recursive algebraic structure of block-tridiagonal systems directly through Schur-complement elimination. In the present scalar implementation, the block matrices reduce to scalar tridiagonal coefficients, making the recursive elimination process transparent while preserving the essential structure of the full block cyclic reduction method used in multidimensional elliptic solvers. The program also verifies the numerical solution against an exact analytic solution of the Poisson equation, thereby illustrating both the algebraic accuracy and logarithmic reduction structure of cyclic reduction methods.

At the core of the implementation is the `TridiagonalSystem` structure, which stores the coefficients of the structured tridiagonal system arising from the finite-difference discretization of the one-dimensional Poisson equation. The vectors `lower`, `diag`, and `upper` represent the subdiagonal, diagonal, and superdiagonal coefficients respectively, while the vector `rhs` stores the forcing term. The `size` function returns the total number of unknowns in the system and is used throughout the recursive elimination process. Although the present implementation uses scalar coefficients, the structure directly mirrors the block form introduced in equation (20.5.15), where each scalar coefficient would be replaced by a block matrix in the multidimensional case.

The function `build_poisson_system` constructs the finite-difference discretization of the one-dimensional Poisson equation with homogeneous Dirichlet boundary conditions. The resulting tridiagonal system corresponds to the standard second-order discretization of the negative Laplacian operator. The function also constructs the exact analytic solution and the corresponding forcing term, thereby providing a verification problem for the cyclic reduction solver. The analytic solution $u(x)=\sin(\pi x)$ is particularly convenient because it satisfies homogeneous Dirichlet boundary conditions naturally and leads to a smooth forcing term with a known exact solution.

The recursive elimination process itself is implemented through the function `cyclic_reduction_solve`. The method follows the same structural principle described in equations (20.5.16)–(20.5.19): neighboring unknowns are eliminated recursively to produce reduced systems with increasingly larger stride spacing. At each reduction level, the algorithm constructs modified diagonal coefficients, off-diagonal couplings, and right-hand-side values corresponding to the Schur-complement elimination of alternating unknowns. The variables `alpha` and `beta` represent the elimination factors associated with neighboring couplings, while the updated coefficients correspond directly to the reduced operators described in equations (20.5.17)–(20.5.19). Each reduction stage doubles the elimination stride, thereby halving the effective number of coupled unknowns and producing the logarithmic reduction depth characteristic of cyclic reduction methods.

The recursive structure of the solver is particularly important computationally. Unlike standard sequential elimination methods, cyclic reduction performs many eliminations independently at each reduction level. This low synchronization depth is one of the major advantages of cyclic reduction on parallel architectures because multiple eliminations can proceed simultaneously. The printed diagnostic reporting the total number of cyclic reduction levels illustrates the logarithmic depth of the elimination process, which scales like $O(\log n)$ rather than linearly with the system size.

The function `residual_norm` evaluates the discrete residual associated with the numerical solution by directly applying the original tridiagonal operator to the computed solution vector. This diagnostic verifies that the cyclic reduction process accurately solves the discrete algebraic system. The function `max_abs_error` computes the maximum pointwise difference between the numerical and exact analytic solutions, thereby measuring the overall discretization accuracy of the finite-difference Poisson approximation.

The `main` function coordinates the complete cyclic reduction solve and verification procedure. It constructs the structured Poisson system, executes the recursive elimination solver, computes the discrete residual norm, and evaluates the maximum pointwise error relative to the exact analytic solution. Diagnostic output reports the number of cyclic reduction levels, the residual norm, and representative numerical solution values at selected grid points. The resulting small residual norm confirms that the recursive elimination procedure accurately solves the structured tridiagonal system, while the pointwise error reflects the underlying finite-difference discretization error of the Poisson approximation. Altogether, the implementation provides a direct computational realization of the recursive block-elimination framework developed throughout Subsection 20.5.3.

```rust
// Program 20.5.3. Cyclic Reduction Solver for a Structured Tridiagonal Poisson System
//
// Problem statement:
// Solve the one-dimensional Poisson system that arises as the line-solve
// building block inside block cyclic reduction:
//
//     a_i u_{i-1} + b_i u_i + c_i u_{i+1} = g_i.
//
// This program demonstrates the recursive elimination idea behind
// equations (20.5.15)-(20.5.19). In the scalar case, the block matrices
// E and B reduce to tridiagonal coefficients, and cyclic reduction becomes
// a structured elimination method with logarithmic reduction depth.

use std::f64::consts::PI;

#[derive(Clone)]
struct TridiagonalSystem {
    lower: Vec<f64>,
    diag: Vec<f64>,
    upper: Vec<f64>,
    rhs: Vec<f64>,
}

impl TridiagonalSystem {
    fn new(lower: Vec<f64>, diag: Vec<f64>, upper: Vec<f64>, rhs: Vec<f64>) -> Self {
        let n = diag.len();

        assert_eq!(lower.len(), n);
        assert_eq!(upper.len(), n);
        assert_eq!(rhs.len(), n);

        Self {
            lower,
            diag,
            upper,
            rhs,
        }
    }

    fn size(&self) -> usize {
        self.diag.len()
    }
}

fn build_poisson_system(n: usize) -> (TridiagonalSystem, Vec<f64>, f64) {
    assert!(n >= 3, "The system must contain at least three unknowns.");

    let dx = 1.0 / (n as f64 + 1.0);
    let inv_dx2 = 1.0 / (dx * dx);

    let mut lower = vec![0.0_f64; n];
    let mut diag = vec![0.0_f64; n];
    let mut upper = vec![0.0_f64; n];
    let mut rhs = vec![0.0_f64; n];
    let mut exact = vec![0.0_f64; n];

    for i in 0..n {
        let x = (i as f64 + 1.0) * dx;

        lower[i] = if i > 0 { -inv_dx2 } else { 0.0 };
        diag[i] = 2.0 * inv_dx2;
        upper[i] = if i + 1 < n { -inv_dx2 } else { 0.0 };

        exact[i] = (PI * x).sin();
        rhs[i] = PI * PI * exact[i];
    }

    (TridiagonalSystem::new(lower, diag, upper, rhs), exact, dx)
}

fn cyclic_reduction_solve(system: &TridiagonalSystem) -> Vec<f64> {
    let n = system.size();

    let mut lower = system.lower.clone();
    let mut diag = system.diag.clone();
    let mut upper = system.upper.clone();
    let mut rhs = system.rhs.clone();

    let mut stride = 1_usize;
    let mut levels = 0_usize;

    while stride < n {
        let old_lower = lower.clone();
        let old_diag = diag.clone();
        let old_upper = upper.clone();
        let old_rhs = rhs.clone();

        let mut new_lower = vec![0.0_f64; n];
        let mut new_diag = vec![0.0_f64; n];
        let mut new_upper = vec![0.0_f64; n];
        let mut new_rhs = vec![0.0_f64; n];

        for i in 0..n {
            let mut diagonal = old_diag[i];
            let mut right_hand_side = old_rhs[i];

            if i >= stride {
                let left = i - stride;
                let alpha = old_lower[i] / old_diag[left];

                diagonal -= alpha * old_upper[left];
                right_hand_side -= alpha * old_rhs[left];

                new_lower[i] = -alpha * old_lower[left];
            }

            if i + stride < n {
                let right = i + stride;
                let beta = old_upper[i] / old_diag[right];

                diagonal -= beta * old_lower[right];
                right_hand_side -= beta * old_rhs[right];

                new_upper[i] = -beta * old_upper[right];
            }

            new_diag[i] = diagonal;
            new_rhs[i] = right_hand_side;
        }

        lower = new_lower;
        diag = new_diag;
        upper = new_upper;
        rhs = new_rhs;

        stride *= 2;
        levels += 1;
    }

    println!("Cyclic reduction levels  = {}", levels);

    rhs.iter()
        .zip(diag.iter())
        .map(|(rhs_i, diag_i)| rhs_i / diag_i)
        .collect()
}

fn residual_norm(system: &TridiagonalSystem, x: &[f64]) -> f64 {
    let n = system.size();
    assert_eq!(x.len(), n);

    let mut sum = 0.0_f64;

    for i in 0..n {
        let mut ax = system.diag[i] * x[i];

        if i > 0 {
            ax += system.lower[i] * x[i - 1];
        }

        if i + 1 < n {
            ax += system.upper[i] * x[i + 1];
        }

        let r = ax - system.rhs[i];
        sum += r * r;
    }

    sum.sqrt()
}

fn max_abs_error(x: &[f64], exact: &[f64]) -> f64 {
    x.iter()
        .zip(exact.iter())
        .map(|(xi, ei)| (xi - ei).abs())
        .fold(0.0_f64, f64::max)
}

fn main() {
    let n = 63;

    let (system, exact, dx) = build_poisson_system(n);
    let numerical = cyclic_reduction_solve(&system);

    let residual = residual_norm(&system, &numerical);
    let error = max_abs_error(&numerical, &exact);

    println!("Cyclic Reduction Solver for a Structured Poisson System");
    println!("======================================================");
    println!();

    println!("Problem Setup");
    println!("-------------");
    println!("Interior unknowns        = {}", n);
    println!("Grid spacing dx          = {:.8}", dx);
    println!("Boundary conditions       = homogeneous Dirichlet");
    println!("Exact solution            = sin(pi x)");
    println!();

    println!("Verification");
    println!("------------");
    println!("Discrete residual norm    = {:.6e}", residual);
    println!("Maximum pointwise error   = {:.6e}", error);
    println!();

    println!("Sample Solution Values");
    println!("----------------------");
    println!("{:>8} {:>14} {:>18} {:>18}", "i", "x", "u_num", "u_exact");

    for i in (0..n).step_by(8) {
        let x = (i as f64 + 1.0) * dx;

        println!(
            "{:>8} {:>14.6} {:>18.10} {:>18.10}",
            i,
            x,
            numerical[i],
            exact[i]
        );
    }
}
```

Program 20.5.3 demonstrates how recursive elimination can exploit the structure of elliptic discretizations without relying on transform diagonalization. Instead of decomposing the operator into Fourier modes, cyclic reduction recursively eliminates alternating unknowns to generate smaller structured systems with the same algebraic form. This recursive Schur-complement strategy preserves sparsity and produces a logarithmic reduction depth, making cyclic reduction especially attractive for structured sparse systems arising from multidimensional PDE discretizations.

The numerical verification confirms that the recursive elimination procedure accurately solves the structured Poisson system. The residual norm is near machine precision, indicating that the cyclic reduction process preserves the algebraic consistency of the original tridiagonal system. The remaining pointwise solution error arises from the finite-difference discretization itself rather than from the recursive elimination solver.

The implementation also illustrates the complementary relationship between cyclic reduction and transform-based methods. Fourier methods exploit translation invariance and transform diagonalization, while cyclic reduction exploits recursive block structure directly through elimination. Consequently, cyclic reduction remains effective even when transform diagonalization is incomplete, when mixed boundary conditions are present, or when only one coordinate direction possesses a transform-compatible structure.

The modular structure of the implementation naturally supports extension to more advanced structured elliptic solvers. Full block cyclic reduction, multidimensional Poisson systems, parallel line solvers, communication-avoiding elimination schemes, GPU-oriented recursive solvers, and hybrid transform-cyclic-reduction methods can all be incorporated while preserving the same recursive elimination philosophy. Consequently, Program 20.5.3 provides a foundational computational example for the broader class of structured recursive elliptic solvers discussed throughout Section 20.5.

## 20.5.4. Pressure Poisson Solves and Practical Solver Choice

A major practical use case for Fourier and cyclic-reduction methods is the pressure solve in incompressible flow. In projection and PISO-type methods, one first computes an intermediate velocity $\mathbf{u}^\ast$. This intermediate velocity generally does not satisfy the incompressibility constraint. A pressure or pressure-increment equation is then solved so that the corrected velocity is divergence free. The pressure Poisson equation has the form:

\begin{equation}
\Delta p
=
\frac{\rho}{\Delta t}
\nabla\cdot\mathbf{u}^{\ast}
\tag{20.5.20}
\end{equation}

Once $p$ is computed, the velocity is corrected by subtracting the pressure-gradient contribution. Since this pressure equation is solved at every time step, its cost can dominate the entire incompressible-flow simulation. Therefore, the choice of Poisson solver is not a secondary implementation detail; it is often the central performance issue.

On channels, pipes, boxes, and other geometries with one or more homogeneous or periodic directions, transform-based solvers are especially attractive. One may apply Fourier, sine, or cosine transforms in the homogeneous directions and then solve tridiagonal systems in the remaining direction. In massively parallel implementations, these transform-based decompositions are often combined with tridiagonal or cyclic-reduction kernels. Recent incompressible-flow work demonstrates both sides of this approach: FFT-based Poisson solvers embedded in immersed-boundary PISO algorithms, and large-scale studies of how Poisson-solver workload grows with Reynolds number in turbulent simulations (Lian, Yao and Jiang, 2025; Trias, Alsalti-Baldellou and Oliva, 2026; Diez Sanhueza et al., 2025).

The comparison with general sparse factorization is instructive. Sparse LU factorization is flexible and applies to a wide range of geometries and coefficients, but it can be expensive in both memory and fill-in. Fourier methods are extremely fast, but they require separability, regular geometry, and boundary conditions compatible with a transform. Cyclic reduction occupies an intermediate position. It is not as general as sparse LU, but it is more flexible than a pure transform solve and has favourable parallel structure.

For textbook purposes, the main principle is that elliptic boundary value problems should be solved by exploiting the algebraic structure created by the PDE and the grid. On rectangular constant-coefficient domains, Fourier methods are difficult to beat. When one direction remains physical while another is diagonalized, cyclic reduction or tridiagonal line solvers become natural. When the coefficients, geometry, or boundary conditions become more complex, multigrid or sparse iterative methods may be more appropriate. The art of numerical PDE solution is not to choose the most general linear solver, but to choose the simplest solver that still captures enough of the problem structure to be efficient and reliable.

# 20.6. Relaxation Methods for Boundary Value Problems

Relaxation methods solve boundary value problems by starting from an approximate solution and repeatedly improving it until the residual becomes sufficiently small. They are among the oldest numerical methods for elliptic problems, but they remain important because they reveal the connection between iterative linear algebra, smoothing of discretization error, pseudo-time evolution, and multigrid acceleration. In modern large-scale solvers, classical relaxation methods such as Jacobi, Gauss-Seidel, and successive over-relaxation are rarely used as complete standalone solvers for difficult two- and three-dimensional problems. Their continuing importance is that they are effective smoothers: they rapidly damp oscillatory error components and therefore serve as essential building blocks inside multigrid and algebraic multigrid methods.

Consider a boundary value problem that has been discretized into the linear system:

$$A u=b \tag{20.6.1}$$

where $A\in\mathbb{R}^{N\times N}$, $u\in\mathbb{R}^N$, and $b\in\mathbb{R}^N$. A relaxation method begins with an initial guess $u^{(0)}$ and generates a sequence $u^{(0)},u^{(1)},u^{(2)},\ldots$ that ideally converges to the exact discrete solution $u=A^{-1}b$. The classical matrix splitting writes:

$$A=D-L-U \tag{20.6.2}$$

where $D$ is the diagonal of $A$, $L$ is the strictly lower-triangular part with sign convention chosen so that $A=D-L-U$, and $U$ is the corresponding strictly upper-triangular part.

## 20.6.1. Jacobi, Gauss-Seidel, and Successive Over-Relaxation

The weighted Jacobi method updates all components simultaneously by applying a diagonally scaled residual correction:

\begin{equation}
u^{(k+1)}
=
u^{(k)}
+
\omega D^{-1}\!\left(b-Au^{(k)}\right)
\tag{20.6.3}
\end{equation}

where $\omega$ is a relaxation parameter. For $\omega=1$, this is the standard Jacobi iteration. For $0<\omega<1$, the correction is under-relaxed, which may improve smoothing properties or stability in certain settings.

The Gauss-Seidel method uses newly computed values immediately. In matrix form it is:

\begin{equation}
(D-L)u^{(k+1)}
=
Uu^{(k)}+b
\tag{20.6.4}
\end{equation}

Because the lower-triangular part is treated implicitly, Gauss-Seidel usually converges faster than Jacobi on model elliptic problems. The price is that the update has sequential dependencies: component $i$ depends on the already updated components $1,\ldots,i-1$. This makes the lexicographic version less naturally parallel than Jacobi.

Successive over-relaxation, or SOR, introduces an extrapolation parameter $\omega$ into Gauss-Seidel:

\begin{equation}
(D-\omega L)u^{(k+1)}
=
\left(
(1-\omega)D+\omega U
\right)
u^{(k)}
+
\omega b
\tag{20.6.5}
\end{equation}

At the component level, SOR can be written as:

\begin{equation}
u_i^{(k+1)}
=
(1-\omega)u_i^{(k)}
+
\frac{\omega}{a_{ii}}
\left(
b_i
-
\sum_{j<i}a_{ij}u_j^{(k+1)}
-
\sum_{j>i}a_{ij}u_j^{(k)}
\right)
\tag{20.6.6}
\end{equation}

For $\omega=1$, SOR reduces to Gauss-Seidel. For carefully chosen $\omega>1$, the method may converge much faster on structured elliptic problems. However, the optimal value of $\omega$ depends on the matrix, the grid, the boundary conditions, and the ordering of the unknowns. This dependence limits SOR as a black-box solver, even though it remains highly instructive and useful in structured settings.

These formulas are not merely algebraic iterations. They can also be interpreted as pseudo-time evolution. If one introduces an artificial time variable $\tau$ and considers,

$$u_{\tau}=b-Au \tag{20.6.7}$$

then weighted Jacobi is a diagonally scaled forward Euler method applied to this artificial evolution equation. This interpretation explains both the usefulness and the weakness of relaxation. The iteration diffuses error away, but it does so gradually, especially for smooth low-frequency error components.

### Rust Implementation

Following the discussion in Subsection 20.6.1 on stationary relaxation schemes for elliptic boundary value problems, Program 20.6.1 provides a practical implementation of weighted Jacobi, Gauss-Seidel, and successive over-relaxation methods for the two-dimensional Poisson equation. The program directly implements the iterative update formulations introduced in equations (20.6.3)–(20.6.6) and compares their convergence behaviour on the same finite-difference discretization of the Poisson problem. Rather than solving the linear system (Au=b) through direct factorization, the implementation begins from an initial approximation and repeatedly applies residual-based corrections until the discrete residual becomes sufficiently small. This iterative viewpoint reflects the pseudo-time interpretation introduced in equation (20.6.7), where relaxation acts as an artificial diffusive evolution that progressively damps discretization error. The program also illustrates the central practical distinction between these methods: weighted Jacobi updates all unknowns simultaneously, Gauss-Seidel immediately reuses newly computed values, and SOR accelerates convergence through over-relaxed extrapolation. By comparing iteration counts, residual norms, and numerical errors, the implementation demonstrates why classical relaxation schemes remain important as smoothers and building blocks for more advanced multigrid solvers.

At the core of the implementation is the `Grid2D` structure, which defines the tensor-product finite-difference discretization of the computational domain. The structure stores the number of interior grid points in the coordinate directions together with the corresponding mesh spacings $\Delta x$ and $\Delta y$. The functions `point`, `index`, and `size` provide geometric indexing and vectorization utilities for mapping between two-dimensional grid coordinates and the underlying one-dimensional storage layout. This structured representation reflects the regular finite-difference discretization associated with the Poisson operator appearing in equation (20.6.1).

The functions `exact_solution` and `rhs_function` define the analytic verification problem used throughout the program. The chosen exact solution satisfies homogeneous Dirichlet boundary conditions naturally and produces a smooth forcing term for the Poisson equation. The function `build_rhs` evaluates the forcing term on the computational grid and constructs the right-hand-side vector $b$ appearing in equation (20.6.1). This verification framework allows the relaxation methods to be tested against a known analytic solution while simultaneously monitoring convergence of the discrete residual.

The discrete Poisson operator itself is implemented through the function `apply_poisson_operator`. This function applies the standard five-point finite-difference Laplacian to a grid function and therefore represents the matrix-vector product $Au$ appearing in equation (20.6.1). The implementation uses nearest-neighbour stencil couplings in the coordinate directions together with homogeneous Dirichlet boundary conditions enforced implicitly through zero boundary values outside the interior grid. The resulting operator corresponds to the sparse elliptic matrix arising from the finite-difference discretization of the two-dimensional Poisson equation.

The function `residual_norm` computes the Euclidean norm of the discrete residual vector $b-Au$. This quantity measures how accurately the current iterate satisfies the discrete linear system and serves as the stopping criterion for all relaxation methods in the program. The function `max_abs_error` computes the maximum pointwise difference between the numerical approximation and the exact analytic solution, thereby measuring the overall discretization accuracy of the finite-difference Poisson approximation independently of the iterative solver used.

The weighted Jacobi method introduced in equation (20.6.3) is implemented through the function `weighted_jacobi_step`. The method computes a diagonally scaled residual correction and updates all unknowns simultaneously using only values from the previous iteration. The relaxation parameter $\omega$ controls the size of the update, allowing either standard Jacobi iteration or under-relaxed weighted Jacobi smoothing. Because the method updates all unknowns independently, it is naturally parallelizable and serves as a useful smoother in multigrid algorithms, although its convergence is relatively slow when used as a standalone solver.

The Gauss-Seidel and SOR methods are implemented through the functions `gauss_seidel_step` and `sor_step`. The Gauss-Seidel method corresponds to the matrix formulation in equation (20.6.4) and updates each unknown sequentially using the most recently computed neighbouring values. This immediate reuse of updated values generally accelerates convergence relative to Jacobi iteration. The SOR method introduced in equations (20.6.5)–(20.6.6) further accelerates convergence by introducing the over-relaxation parameter $\omega$. The implementation computes the Gauss-Seidel update and then extrapolates the correction through weighted relaxation. For carefully chosen values of (\\omega>1), the method converges substantially faster on structured elliptic systems.

The functions `solve_weighted_jacobi`, `solve_gauss_seidel`, and `solve_sor` provide complete iterative solvers for the corresponding relaxation methods. Each solver repeatedly applies the appropriate relaxation update, computes the residual norm, and terminates once the prescribed tolerance is reached or the maximum number of iterations has been exceeded. These functions therefore provide a direct computational realization of the iterative relaxation framework discussed throughout Subsection 20.6.1.

The `main` function coordinates the complete comparison of the relaxation methods. It constructs the finite-difference Poisson problem, selects relaxation parameters and stopping tolerances, and executes the weighted Jacobi, Gauss-Seidel, and SOR solvers sequentially. Diagnostic output reports the number of iterations required for convergence together with the final residual norm and maximum pointwise solution error for each method. The resulting iteration counts illustrate the expected convergence hierarchy: SOR converges substantially faster than Gauss-Seidel, while Gauss-Seidel converges substantially faster than weighted Jacobi. Altogether, the implementation provides a practical demonstration of the relaxation framework developed throughout Subsection 20.6.1 and illustrates the connection between stationary iteration, residual smoothing, and elliptic PDE solution methods.

```rust
// Program 20.6.1. Jacobi, Gauss-Seidel, and SOR Relaxation for the 2D Poisson Equation
//
// Problem statement:
// Solve the two-dimensional Poisson boundary value problem
//
//     -Delta u = f
//
// on the unit square with homogeneous Dirichlet boundary conditions.
// The program compares weighted Jacobi, Gauss-Seidel, and successive
// over-relaxation as stationary iterations for the linear system A u = b.
// It follows the relaxation framework in equations (20.6.1)-(20.6.7).

use std::f64::consts::PI;

#[derive(Clone)]
struct Grid2D {
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
}

impl Grid2D {
    fn new(nx: usize, ny: usize) -> Self {
        assert!(nx >= 4, "At least four interior x-points are required.");
        assert!(ny >= 4, "At least four interior y-points are required.");

        let dx = 1.0 / (nx as f64 + 1.0);
        let dy = 1.0 / (ny as f64 + 1.0);

        Self { nx, ny, dx, dy }
    }

    fn point(&self, i: usize, j: usize) -> (f64, f64) {
        ((i as f64 + 1.0) * self.dx, (j as f64 + 1.0) * self.dy)
    }

    fn index(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    fn size(&self) -> usize {
        self.nx * self.ny
    }
}

fn exact_solution(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

fn rhs_function(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * exact_solution(x, y)
}

fn build_rhs(grid: &Grid2D) -> Vec<f64> {
    let mut rhs = vec![0.0_f64; grid.size()];

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let (x, y) = grid.point(i, j);
            rhs[grid.index(i, j)] = rhs_function(x, y);
        }
    }

    rhs
}

fn apply_poisson_operator(grid: &Grid2D, u: &[f64]) -> Vec<f64> {
    let mut au = vec![0.0_f64; grid.size()];

    let inv_dx2 = 1.0 / (grid.dx * grid.dx);
    let inv_dy2 = 1.0 / (grid.dy * grid.dy);

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let center = u[grid.index(i, j)];

            let left = if i > 0 {
                u[grid.index(i - 1, j)]
            } else {
                0.0
            };

            let right = if i + 1 < grid.nx {
                u[grid.index(i + 1, j)]
            } else {
                0.0
            };

            let down = if j > 0 {
                u[grid.index(i, j - 1)]
            } else {
                0.0
            };

            let up = if j + 1 < grid.ny {
                u[grid.index(i, j + 1)]
            } else {
                0.0
            };

            au[grid.index(i, j)] =
                (2.0 * center - left - right) * inv_dx2
                    + (2.0 * center - down - up) * inv_dy2;
        }
    }

    au
}

fn residual_norm(grid: &Grid2D, u: &[f64], rhs: &[f64]) -> f64 {
    let au = apply_poisson_operator(grid, u);

    au.iter()
        .zip(rhs.iter())
        .map(|(aui, bi)| (bi - aui).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn max_abs_error(grid: &Grid2D, u: &[f64]) -> f64 {
    let mut max_error = 0.0_f64;

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let (x, y) = grid.point(i, j);
            let error = (u[grid.index(i, j)] - exact_solution(x, y)).abs();
            max_error = max_error.max(error);
        }
    }

    max_error
}

fn weighted_jacobi_step(grid: &Grid2D, u: &[f64], rhs: &[f64], omega: f64) -> Vec<f64> {
    let mut next = u.to_vec();

    let inv_dx2 = 1.0 / (grid.dx * grid.dx);
    let inv_dy2 = 1.0 / (grid.dy * grid.dy);
    let diagonal = 2.0 * inv_dx2 + 2.0 * inv_dy2;

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let idx = grid.index(i, j);

            let left = if i > 0 {
                u[grid.index(i - 1, j)]
            } else {
                0.0
            };

            let right = if i + 1 < grid.nx {
                u[grid.index(i + 1, j)]
            } else {
                0.0
            };

            let down = if j > 0 {
                u[grid.index(i, j - 1)]
            } else {
                0.0
            };

            let up = if j + 1 < grid.ny {
                u[grid.index(i, j + 1)]
            } else {
                0.0
            };

            let sigma = (left + right) * inv_dx2 + (down + up) * inv_dy2;
            let jacobi_value = (rhs[idx] + sigma) / diagonal;

            next[idx] = (1.0 - omega) * u[idx] + omega * jacobi_value;
        }
    }

    next
}

fn gauss_seidel_step(grid: &Grid2D, u: &mut [f64], rhs: &[f64]) {
    sor_step(grid, u, rhs, 1.0);
}

fn sor_step(grid: &Grid2D, u: &mut [f64], rhs: &[f64], omega: f64) {
    let inv_dx2 = 1.0 / (grid.dx * grid.dx);
    let inv_dy2 = 1.0 / (grid.dy * grid.dy);
    let diagonal = 2.0 * inv_dx2 + 2.0 * inv_dy2;

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let idx = grid.index(i, j);

            let left = if i > 0 {
                u[grid.index(i - 1, j)]
            } else {
                0.0
            };

            let right = if i + 1 < grid.nx {
                u[grid.index(i + 1, j)]
            } else {
                0.0
            };

            let down = if j > 0 {
                u[grid.index(i, j - 1)]
            } else {
                0.0
            };

            let up = if j + 1 < grid.ny {
                u[grid.index(i, j + 1)]
            } else {
                0.0
            };

            let sigma = (left + right) * inv_dx2 + (down + up) * inv_dy2;
            let gauss_seidel_value = (rhs[idx] + sigma) / diagonal;

            u[idx] = (1.0 - omega) * u[idx] + omega * gauss_seidel_value;
        }
    }
}

fn solve_weighted_jacobi(
    grid: &Grid2D,
    rhs: &[f64],
    omega: f64,
    max_iterations: usize,
    tolerance: f64,
) -> (Vec<f64>, usize, f64) {
    let mut u = vec![0.0_f64; grid.size()];

    for iteration in 1..=max_iterations {
        u = weighted_jacobi_step(grid, &u, rhs, omega);

        let residual = residual_norm(grid, &u, rhs);

        if residual < tolerance {
            return (u, iteration, residual);
        }
    }

    let residual = residual_norm(grid, &u, rhs);
    (u, max_iterations, residual)
}

fn solve_gauss_seidel(
    grid: &Grid2D,
    rhs: &[f64],
    max_iterations: usize,
    tolerance: f64,
) -> (Vec<f64>, usize, f64) {
    let mut u = vec![0.0_f64; grid.size()];

    for iteration in 1..=max_iterations {
        gauss_seidel_step(grid, &mut u, rhs);

        let residual = residual_norm(grid, &u, rhs);

        if residual < tolerance {
            return (u, iteration, residual);
        }
    }

    let residual = residual_norm(grid, &u, rhs);
    (u, max_iterations, residual)
}

fn solve_sor(
    grid: &Grid2D,
    rhs: &[f64],
    omega: f64,
    max_iterations: usize,
    tolerance: f64,
) -> (Vec<f64>, usize, f64) {
    let mut u = vec![0.0_f64; grid.size()];

    for iteration in 1..=max_iterations {
        sor_step(grid, &mut u, rhs, omega);

        let residual = residual_norm(grid, &u, rhs);

        if residual < tolerance {
            return (u, iteration, residual);
        }
    }

    let residual = residual_norm(grid, &u, rhs);
    (u, max_iterations, residual)
}

fn main() {
    let grid = Grid2D::new(40, 40);
    let rhs = build_rhs(&grid);

    let max_iterations = 20_000;
    let tolerance = 1.0e-8;

    let jacobi_omega = 2.0 / 3.0;
    let sor_omega = 1.85;

    println!("Relaxation Methods for the 2D Poisson Equation");
    println!("==============================================");
    println!();

    println!("Grid Parameters");
    println!("---------------");
    println!("Interior x-points        = {}", grid.nx);
    println!("Interior y-points        = {}", grid.ny);
    println!("Grid spacing dx          = {:.8}", grid.dx);
    println!("Grid spacing dy          = {:.8}", grid.dy);
    println!("Unknowns                 = {}", grid.size());
    println!();

    println!("Stopping Criteria");
    println!("-----------------");
    println!("Maximum iterations       = {}", max_iterations);
    println!("Residual tolerance       = {:.2e}", tolerance);
    println!();

    let (u_jacobi, iter_jacobi, res_jacobi) =
        solve_weighted_jacobi(&grid, &rhs, jacobi_omega, max_iterations, tolerance);

    let (u_gs, iter_gs, res_gs) =
        solve_gauss_seidel(&grid, &rhs, max_iterations, tolerance);

    let (u_sor, iter_sor, res_sor) =
        solve_sor(&grid, &rhs, sor_omega, max_iterations, tolerance);

    println!("Solver Comparison");
    println!("-----------------");
    println!(
        "{:>22} {:>14} {:>18} {:>18}",
        "Method", "Iterations", "Residual", "Max Error"
    );

    println!(
        "{:>22} {:>14} {:>18.6e} {:>18.6e}",
        "Weighted Jacobi",
        iter_jacobi,
        res_jacobi,
        max_abs_error(&grid, &u_jacobi)
    );

    println!(
        "{:>22} {:>14} {:>18.6e} {:>18.6e}",
        "Gauss-Seidel",
        iter_gs,
        res_gs,
        max_abs_error(&grid, &u_gs)
    );

    println!(
        "{:>22} {:>14} {:>18.6e} {:>18.6e}",
        "SOR",
        iter_sor,
        res_sor,
        max_abs_error(&grid, &u_sor)
    );

    println!();
    println!("Relaxation Parameters");
    println!("---------------------");
    println!("Jacobi omega           = {:.6}", jacobi_omega);
    println!("SOR omega              = {:.6}", sor_omega);
}
```

Program 20.6.1 demonstrates how classical relaxation methods iteratively reduce the residual of elliptic boundary value problems through repeated local correction sweeps. Rather than computing the solution through direct matrix factorization, the methods progressively improve an approximate solution until the discrete residual becomes sufficiently small. This iterative viewpoint reflects the pseudo-time interpretation introduced in equation (20.6.7), where relaxation behaves like an artificial dissipative evolution that smooths discretization error.

The numerical results clearly illustrate the different convergence properties of the relaxation schemes. Weighted Jacobi converges relatively slowly because all unknowns are updated simultaneously using only information from the previous iteration. Gauss-Seidel converges more rapidly because newly computed values are immediately reused within the same sweep. SOR converges substantially faster still because the over-relaxation parameter extrapolates the correction and accelerates the damping of smooth error components. These differences explain why SOR historically served as an efficient standalone structured solver, while Jacobi and Gauss-Seidel remain especially important as smoothers inside multigrid algorithms.

The implementation also demonstrates the important distinction between residual reduction and discretization accuracy. All three methods converge to essentially the same discrete solution and therefore produce nearly identical maximum solution errors. The primary difference between the methods lies not in the final accuracy of the approximation, but in the computational work required to reach the prescribed residual tolerance.

The modular structure of the implementation naturally supports extension to more advanced iterative methods. Red-black ordering, line relaxation, block relaxation, incomplete factorization preconditioners, Krylov acceleration, multigrid V-cycles, and algebraic multigrid smoothers can all be incorporated while preserving the same residual-correction philosophy. Consequently, Program 20.6.1 provides a foundational computational example for the broader class of relaxation and multilevel elliptic solvers developed throughout Section 20.6.

## 20.6.2. Error Propagation and the Smoothing Property

Let $u$ denote the exact solution of (20.6.1), and define the error after $k$ iterations by:

$$e^{(k)}=u-u^{(k)}\tag{20.6.8}$$

For a stationary linear iteration, the error satisfies:

$$e^{(k+1)}=G e^{(k)} \tag{20.6.9}$$

where $G$ is the iteration matrix. Convergence requires:

$$\rho(G)<1,\tag{20.6.10}$$

where $\rho(G)$ is the spectral radius of $G$. This condition states that every error mode must eventually be damped.

For the five-point Laplacian on a uniform two-dimensional grid, the symbol of weighted Jacobi is:

\begin{equation}
\widehat{G}_J(\theta_x,\theta_y)
=
1-\omega
+
\frac{\omega}{2}
\left(
\cos\theta_x+\cos\theta_y
\right)
\tag{20.6.11}
\end{equation}

This expression reveals the central behaviour of relaxation methods. High-frequency error components, for which $\theta_x$ or $\theta_y$ is near $\pi$, are strongly damped. Smooth low-frequency error components, for which $\theta_x$ and $\theta_y$ are near zero, are damped very slowly. Thus, classical relaxation is an effective smoother but a poor complete solver for large elliptic systems.

This smoothing property explains why Jacobi, Gauss-Seidel, and SOR remain important. Their role in modern numerical PDEs is not primarily to solve $Au=b$ to high accuracy by themselves. Instead, they rapidly remove oscillatory error so that the remaining smooth error can be treated on coarser grids. This is the fundamental idea behind multigrid.

The classical comparison is therefore structural. Jacobi is simple, easy to analyze, trivially parallel, and convenient for vectorization, but it converges slowly. Gauss-Seidel generally damps error more effectively because it uses updated information immediately, but it introduces ordering dependencies. SOR can accelerate Gauss-Seidel, but it requires a suitable relaxation parameter and is not naturally parallel in its lexicographic form. These limitations motivate colored Gauss-Seidel, block relaxation, line relaxation, polynomial smoothers, and Chebyshev smoothers in modern implementations. Local Fourier analysis remains an important tool for studying and tuning such smoothers, especially for high-order finite element operators and $p$-multigrid methods (Thompson, Brown and He, 2023).

### Rust Implementation

Following the discussion in Subsection 20.6.2 on error propagation and the smoothing behaviour of stationary relaxation schemes, Program 20.6.2 provides a practical implementation of weighted Jacobi iteration for the two-dimensional Poisson equation. The program is designed specifically to illustrate the spectral behaviour described by equations (20.6.9)–(20.6.11), where the error evolves under repeated application of the iteration matrix. Rather than solving a boundary value problem completely, the implementation isolates individual Fourier error modes and studies how rapidly each mode is damped by weighted Jacobi relaxation. This directly exposes the smoothing property central to modern multigrid theory: oscillatory high-frequency errors decay rapidly, while smooth low-frequency errors persist for many iterations. The program therefore demonstrates why classical relaxation methods remain essential in modern PDE solvers, not primarily as standalone algorithms, but as highly effective smoothers inside multilevel and multigrid frameworks.

At the core of the implementation is the `Grid2D` structure, which defines the uniform tensor-product computational grid used for the finite-difference discretization of the Poisson operator. The structure stores the number of interior grid points in each coordinate direction and provides indexing utilities that map two-dimensional grid coordinates into the one-dimensional storage layout used by the vectors representing the discrete error field. This regular grid structure reflects the finite-difference discretization associated with the five-point Laplacian discussed throughout Subsection 20.6.2.

The function `sine_error_mode` constructs discrete sine modes corresponding to the eigenvectors of the Dirichlet five-point Laplacian. The mode indices $(p_x,p_y)$ determine the oscillatory frequency of the error in the coordinate directions, while the corresponding angles $\theta_x$ and $\theta_y$ define the Fourier frequencies appearing in equation (20.6.11). Because homogeneous Dirichlet boundary conditions are imposed, sine functions provide the natural eigenbasis of the discrete operator. This choice is essential because it allows the measured damping rates to agree directly with the theoretical predictions obtained from local Fourier analysis. The function therefore generates error fields that evolve independently under weighted Jacobi iteration according to the corresponding eigenvalue of the iteration matrix.

The weighted Jacobi update itself is implemented in the function `weighted_jacobi_error_step`. This function applies the iteration operator associated with equation (20.6.3) to the current error field. For each interior grid point, the new error value is computed using the average of neighbouring values together with the relaxation parameter (\\omega). The implementation corresponds directly to the weighted Jacobi error-propagation matrix appearing in equation (20.6.9). Because the update depends only on values from the previous iteration, all unknowns may be updated simultaneously, making the method naturally parallel and highly suitable for vectorized architectures and GPU execution.

The function `jacobi_symbol` evaluates the local Fourier symbol introduced in equation (20.6.11). This quantity predicts the damping factor associated with a particular Fourier error mode. Smooth modes with small $\theta_x$ and $\theta_y$ produce symbols close to one and therefore decay slowly. Highly oscillatory modes with frequencies near $\pi$ produce symbols of much smaller magnitude and therefore decay rapidly. The program uses this function to compare the theoretically predicted damping behaviour with the measured numerical reduction in the error norm after repeated relaxation sweeps.

The function `l2_norm` computes the Euclidean norm of the discrete error field and therefore measures the total magnitude of the error after each relaxation stage. The function `run_mode_case` coordinates the complete numerical experiment for a single Fourier mode. It constructs the initial error field, evaluates the corresponding Fourier symbol, repeatedly applies weighted Jacobi iteration, and computes both the measured and predicted damping factors. The program then reports the absolute difference between these quantities, demonstrating that the sine modes are indeed eigenmodes of the Dirichlet Laplacian and that the measured error decay agrees with local Fourier analysis to machine precision.

The `main` function orchestrates the complete smoothing experiment. It constructs the computational grid, selects the relaxation parameter and number of smoothing sweeps, and evaluates three representative error configurations: a smooth low-frequency mode, a mixed-frequency mode, and a highly oscillatory high-frequency mode. The resulting output clearly demonstrates the smoothing property described in Subsection 20.6.2. The low-frequency mode experiences very little damping, while the oscillatory mode decays by several orders of magnitude in only a few iterations. This directly illustrates why classical relaxation methods are highly effective smoothers but inefficient standalone solvers for large elliptic systems.

```rust
// Program 20.6.2. Weighted Jacobi Error Smoothing for the 2D Poisson Equation
//
// Problem statement:
// Demonstrate the smoothing property of weighted Jacobi relaxation for the
// five-point Laplacian on a two-dimensional uniform grid with homogeneous
// Dirichlet boundary conditions.
//
// The program initializes error fields as discrete sine modes, which are
// eigenmodes of the Dirichlet Laplacian. It then compares the measured
// damping factor with the local Fourier symbol in equation (20.6.11).

use std::f64::consts::PI;

#[derive(Clone)]
struct Grid2D {
    nx: usize,
    ny: usize,
}

impl Grid2D {
    fn new(nx: usize, ny: usize) -> Self {
        assert!(nx >= 4, "At least four x-points are required.");
        assert!(ny >= 4, "At least four y-points are required.");

        Self { nx, ny }
    }

    fn index(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    fn size(&self) -> usize {
        self.nx * self.ny
    }
}

fn sine_error_mode(grid: &Grid2D, px: usize, py: usize) -> Vec<f64> {
    let mut error = vec![0.0_f64; grid.size()];

    let theta_x = px as f64 * PI / (grid.nx as f64 + 1.0);
    let theta_y = py as f64 * PI / (grid.ny as f64 + 1.0);

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let value =
                ((i + 1) as f64 * theta_x).sin()
                    * ((j + 1) as f64 * theta_y).sin();

            error[grid.index(i, j)] = value;
        }
    }

    error
}

fn weighted_jacobi_error_step(grid: &Grid2D, error: &[f64], omega: f64) -> Vec<f64> {
    let mut next = vec![0.0_f64; grid.size()];

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let idx = grid.index(i, j);

            let left = if i > 0 {
                error[grid.index(i - 1, j)]
            } else {
                0.0
            };

            let right = if i + 1 < grid.nx {
                error[grid.index(i + 1, j)]
            } else {
                0.0
            };

            let down = if j > 0 {
                error[grid.index(i, j - 1)]
            } else {
                0.0
            };

            let up = if j + 1 < grid.ny {
                error[grid.index(i, j + 1)]
            } else {
                0.0
            };

            next[idx] =
                (1.0 - omega) * error[idx]
                    + (omega / 4.0) * (left + right + down + up);
        }
    }

    next
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn jacobi_symbol(theta_x: f64, theta_y: f64, omega: f64) -> f64 {
    1.0 - omega + 0.5 * omega * (theta_x.cos() + theta_y.cos())
}

fn run_mode_case(
    label: &str,
    grid: &Grid2D,
    px: usize,
    py: usize,
    omega: f64,
    sweeps: usize,
) {
    let theta_x = px as f64 * PI / (grid.nx as f64 + 1.0);
    let theta_y = py as f64 * PI / (grid.ny as f64 + 1.0);

    let mut error = sine_error_mode(grid, px, py);

    let initial_norm = l2_norm(&error);
    let symbol = jacobi_symbol(theta_x, theta_y, omega);

    for _ in 0..sweeps {
        error = weighted_jacobi_error_step(grid, &error, omega);
    }

    let final_norm = l2_norm(&error);
    let measured_factor = final_norm / initial_norm;
    let predicted_factor = symbol.abs().powi(sweeps as i32);

    println!("{}", label);
    println!("{}", "-".repeat(label.len()));
    println!("Mode indices              = ({}, {})", px, py);
    println!("theta_x                   = {:.6}", theta_x);
    println!("theta_y                   = {:.6}", theta_y);
    println!("Jacobi symbol             = {:.8}", symbol);
    println!("Sweeps                    = {}", sweeps);
    println!("Initial error norm        = {:.12}", initial_norm);
    println!("Final error norm          = {:.12}", final_norm);
    println!("Measured damping factor   = {:.8e}", measured_factor);
    println!("Predicted damping factor  = {:.8e}", predicted_factor);
    println!(
        "Absolute factor error     = {:.8e}",
        (measured_factor - predicted_factor).abs()
    );
    println!();
}

fn main() {
    let grid = Grid2D::new(64, 64);

    let omega: f64 = 2.0 / 3.0;
    let sweeps = 10;

    println!("Weighted Jacobi Error Smoothing");
    println!("==============================");
    println!();

    println!("Grid Parameters");
    println!("---------------");
    println!("Interior x-points        = {}", grid.nx);
    println!("Interior y-points        = {}", grid.ny);
    println!("Relaxation parameter      = {:.6}", omega);
    println!("Number of sweeps          = {}", sweeps);
    println!();

    run_mode_case(
        "Smooth Low-Frequency Error",
        &grid,
        1,
        1,
        omega,
        sweeps,
    );

    run_mode_case(
        "Mixed-Frequency Error",
        &grid,
        1,
        32,
        omega,
        sweeps,
    );

    run_mode_case(
        "Oscillatory High-Frequency Error",
        &grid,
        64,
        64,
        omega,
        sweeps,
    );

    println!("Interpretation");
    println!("--------------");
    println!("Using sine modes makes the initialized errors exact eigenmodes of the");
    println!("Dirichlet five-point Laplacian. Therefore, the measured damping factor");
    println!("matches the prediction from the weighted Jacobi symbol.");
}
```

Program 20.6.2 demonstrates the essential smoothing behaviour of weighted Jacobi relaxation through direct numerical experiments on individual Fourier error modes. By comparing measured damping factors against the predictions of local Fourier analysis, the implementation confirms that highly oscillatory error components are rapidly attenuated, while smooth low-frequency errors decay only slowly. This behaviour explains why stationary relaxation methods are fundamentally limited as complete standalone solvers for large elliptic systems.

The numerical experiments also illustrate the spectral interpretation of relaxation methods. Each Fourier mode behaves independently under the action of the iteration matrix, and its convergence rate is determined by the corresponding eigenvalue of the weighted Jacobi operator. The high-frequency modes associated with large values of $\theta_x$ and $\theta_y$ experience strong damping, while smooth modes associated with small frequencies remain nearly unchanged over many iterations. This separation of behaviour is precisely what makes relaxation suitable for multigrid acceleration.

The modular design of the implementation allows the same framework to be extended naturally to Gauss-Seidel, SOR, red-black relaxation, polynomial smoothers, Chebyshev smoothers, and block relaxation schemes. It also provides a computational foundation for local Fourier analysis of more advanced discretizations, including higher-order finite elements, discontinuous Galerkin methods, and p-multigrid smoothers. Consequently, Program 20.6.2 serves both as a practical demonstration of classical smoothing theory and as a foundation for modern multilevel iterative methods used in large-scale scientific computing.

## 20.6.3. Multigrid as the Natural Extension of Relaxation

Multigrid methods arise directly from the smoothing property of relaxation. A few relaxation sweeps efficiently damp high-frequency error, but leave smooth error largely unchanged. On a coarser grid, however, that same smooth error appears more oscillatory relative to the coarser mesh and can again be reduced efficiently. Multigrid exploits this observation by combining relaxation, residual restriction, coarse-grid correction, and interpolation.

Let $A_hu_h=f_h$ denote the fine-grid problem. After applying one or more smoothing sweeps, the residual is:

\begin{equation}
r_h
=
f_h-A_hu_h
\tag{20.6.12}
\end{equation}

The residual is restricted to a coarser grid:

\begin{equation}
r_{2h}
=
R_h^{2h}r_h
\tag{20.6.13}
\end{equation}

where $R_h^{2h}$ is the restriction operator. The coarse-grid error equation is:

\begin{equation}
A_{2h}e_{2h}
=
r_{2h}
\tag{20.6.14}
\end{equation}

After solving or approximately solving this coarse problem, the correction is interpolated back to the fine grid and added to the current approximation:

\begin{equation}
u_h
\leftarrow
u_h+P_{2h}^{h}e_{2h}
\tag{20.6.15}
\end{equation}

where $P_{2h}^{h}$ is the prolongation operator. A few post-smoothing sweeps are then applied to remove high-frequency error introduced by interpolation.

A multigrid V-cycle may therefore be summarized as:

\begin{equation}
\begin{aligned}
r_h &= f_h-A_hu_h,\\
r_{2h} &= R_h^{2h}r_h,\\
A_{2h}e_{2h} &= r_{2h},\\
u_h &\leftarrow u_h+P_{2h}^{h}e_{2h}
\end{aligned}
\tag{20.6.16}
\end{equation}

The process is repeated recursively over a hierarchy of grids,

\begin{equation}
h
\longrightarrow 2h
\longrightarrow 4h
\longrightarrow \cdots
\longrightarrow H
\longrightarrow \cdots
\longrightarrow 4h
\longrightarrow 2h
\longrightarrow h
\tag{20.6.17}
\end{equation}

Smoothing is applied on the way down and on the way back up. For structured elliptic problems, multigrid can achieve nearly $O(N)$ work and $O(N)$ storage, which is why it is often the asymptotic standard against which other elliptic solvers are compared.

The essential point is that multigrid is not separate from relaxation; it is the natural completion of relaxation. Classical relaxation damps oscillatory error but struggles with smooth error. Coarse-grid correction converts smooth error into a form that relaxation can reduce efficiently. This combination is what gives multigrid its efficiency. Recent pedagogical and performance-oriented work continues to emphasize both sides of this idea: clear implementation of multigrid cycles for diffusion problems and careful organization of smoothing kernels for modern many-core machines (Choi et al., 2025; Wichrowski et al., 2025).

### Rust Implementation

Following the discussion in Subsection 20.6.3 on multigrid methods as the natural extension of relaxation, Program 20.6.3 provides a practical implementation of a recursive multigrid V-cycle solver for the two-dimensional Poisson equation. The implementation combines weighted Jacobi smoothing, residual restriction, coarse-grid correction, and bilinear interpolation across a hierarchy of progressively coarser meshes, directly reflecting the multigrid framework introduced in equations (20.6.12)–(20.6.17). Rather than attempting to eliminate all error on a single grid, the program exploits the complementary strengths of relaxation and coarse-grid correction. Oscillatory error is rapidly damped by weighted Jacobi smoothing on the fine grid, while smooth error components are transferred to coarser grids where they appear more oscillatory and can again be reduced efficiently. The implementation therefore demonstrates the central multigrid principle that efficient elliptic solvers arise not from a single iteration strategy, but from the coordinated interaction of smoothing and coarse-grid correction across multiple spatial scales.

At the core of the implementation is the `GridLevel` structure, which represents a single level in the multigrid hierarchy. Each level stores the number of interior grid points, the grid spacing, the current approximate solution vector, and the right-hand-side vector corresponding to the discrete Poisson equation. The methods `index` and `point` provide indexing and geometric coordinate utilities that support tensor-product finite-difference discretization across all grid levels. This hierarchical grid representation mirrors the recursive mesh sequence described in equation (20.6.17), where the problem is repeatedly transferred between fine and coarse discretizations.

The functions `exact_solution` and `rhs_function` define the analytic verification problem used throughout the implementation. The chosen exact solution satisfies homogeneous Dirichlet boundary conditions naturally and generates a smooth forcing term for the Poisson equation. The function `initialize_rhs` evaluates this forcing term on the computational mesh and constructs the right-hand side $f_h$ appearing in equation (20.6.12). This analytic setup allows the convergence of the multigrid solver to be verified directly against a known exact solution.

The discrete elliptic operator is implemented through the function `apply_poisson`, which applies the standard five-point finite-difference Laplacian to a grid function. This function therefore represents the action of the discrete operator $A_h$ appearing throughout equations (20.6.12)–(20.6.16). The implementation uses nearest-neighbour stencil couplings together with homogeneous Dirichlet boundary conditions imposed implicitly through zero values outside the computational domain. The resulting sparse operator corresponds to the standard second-order finite-difference discretization of the two-dimensional Poisson equation.

The function `compute_residual` implements the residual equation introduced in equation (20.6.12). It evaluates the difference between the right-hand side and the action of the discrete operator on the current approximation, thereby producing the fine-grid residual vector $r_h=f_h-A_hu_h$. The functions `l2_norm` and `residual_norm` measure the Euclidean norm of this residual, providing a quantitative measure of how accurately the current iterate satisfies the discrete Poisson system. The function `max_abs_error` compares the numerical approximation against the exact analytic solution and therefore measures the overall discretization accuracy independently of the iterative residual reduction.

The weighted Jacobi smoother is implemented in the function `weighted_jacobi_smooth`. This function applies repeated relaxation sweeps corresponding to equation (20.6.3) using the weighted Jacobi iteration matrix. Each sweep computes a diagonally scaled residual correction while updating all unknowns simultaneously from the previous iterate. The relaxation parameter $\omega$ controls the damping strength of the smoother. As discussed in Subsection 20.6.2, weighted Jacobi efficiently removes oscillatory high-frequency error components, making it an effective multigrid smoother even though it converges slowly as a standalone solver.

The transfer operators between grid levels are implemented through the functions `restrict_full_weighting` and `prolong_bilinear`. The restriction function corresponds to equation (20.6.13) and transfers the fine-grid residual to the coarse mesh using full-weighting averaging. This operator preserves the large-scale structure of the residual while filtering high-frequency components that cannot be represented accurately on the coarser grid. The prolongation function implements the interpolation operator appearing in equation (20.6.15). Bilinear interpolation is used to transfer the coarse-grid correction back to the fine grid while maintaining smoothness and consistency between mesh levels. Together, these operators realize the coarse-grid correction framework central to multigrid methodology.

The recursive V-cycle itself is implemented through the function `v_cycle`. The algorithm begins by applying several pre-smoothing sweeps to damp high-frequency error on the current grid. The residual is then computed and restricted to the next coarser grid, where the coarse-grid error equation in equation (20.6.14) is solved recursively by another V-cycle. Once the coarse-grid correction has been computed, it is prolongated back to the fine grid and added to the current approximation according to equation (20.6.15). Finally, several post-smoothing sweeps are applied to remove oscillatory interpolation error introduced during prolongation. The recursion terminates on the coarsest grid, where repeated smoothing sweeps approximate a direct solve.

The `main` function coordinates the complete multigrid computation. It constructs the finest computational grid, initializes the Poisson right-hand side, selects smoothing parameters, and repeatedly applies multigrid V-cycles while monitoring convergence. The printed convergence history demonstrates the characteristic efficiency of multigrid methods: the residual norm decreases rapidly by several orders of magnitude within only a small number of V-cycles, while the solution error eventually levels off at the finite-difference discretization accuracy of the underlying mesh. Altogether, the implementation provides a complete practical realization of the multigrid framework developed throughout Subsection 20.6.3.

```rust
// Program 20.6.3. Multigrid V-Cycle Solver for the 2D Poisson Equation

use std::f64::consts::PI;

#[derive(Clone)]
struct GridLevel {
    n: usize,
    h: f64,
    u: Vec<f64>,
    f: Vec<f64>,
}

impl GridLevel {
    fn new(n: usize) -> Self {
        assert!(n >= 3 && n % 2 == 1);

        let h = 1.0 / (n as f64 + 1.0);

        Self {
            n,
            h,
            u: vec![0.0; n * n],
            f: vec![0.0; n * n],
        }
    }

    fn index(&self, i: usize, j: usize) -> usize {
        j * self.n + i
    }

    fn point(&self, i: usize, j: usize) -> (f64, f64) {
        ((i as f64 + 1.0) * self.h, (j as f64 + 1.0) * self.h)
    }
}

fn exact_solution(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

fn rhs_function(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * exact_solution(x, y)
}

fn initialize_rhs(level: &mut GridLevel) {
    for j in 0..level.n {
        for i in 0..level.n {
            let (x, y) = level.point(i, j);
            let idx = level.index(i, j);
            level.f[idx] = rhs_function(x, y);
        }
    }
}

fn apply_poisson(level: &GridLevel, u: &[f64]) -> Vec<f64> {
    let n = level.n;
    let inv_h2 = 1.0 / (level.h * level.h);
    let mut au = vec![0.0; n * n];

    for j in 0..n {
        for i in 0..n {
            let center = u[level.index(i, j)];

            let left = if i > 0 {
                u[level.index(i - 1, j)]
            } else {
                0.0
            };

            let right = if i + 1 < n {
                u[level.index(i + 1, j)]
            } else {
                0.0
            };

            let down = if j > 0 {
                u[level.index(i, j - 1)]
            } else {
                0.0
            };

            let up = if j + 1 < n {
                u[level.index(i, j + 1)]
            } else {
                0.0
            };

            au[level.index(i, j)] =
                (4.0 * center - left - right - down - up) * inv_h2;
        }
    }

    au
}

fn compute_residual(level: &GridLevel) -> Vec<f64> {
    let au = apply_poisson(level, &level.u);

    level
        .f
        .iter()
        .zip(au.iter())
        .map(|(fi, aui)| fi - aui)
        .collect()
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn residual_norm(level: &GridLevel) -> f64 {
    l2_norm(&compute_residual(level))
}

fn max_abs_error(level: &GridLevel) -> f64 {
    let mut max_error = 0.0;

    for j in 0..level.n {
        for i in 0..level.n {
            let (x, y) = level.point(i, j);
            let idx = level.index(i, j);

            max_error = f64::max(
                max_error,
                (level.u[idx] - exact_solution(x, y)).abs(),
            );
        }
    }

    max_error
}

fn weighted_jacobi_smooth(level: &mut GridLevel, omega: f64, sweeps: usize) {
    let n = level.n;
    let h2 = level.h * level.h;

    for _ in 0..sweeps {
        let old = level.u.clone();

        for j in 0..n {
            for i in 0..n {
                let left = if i > 0 {
                    old[level.index(i - 1, j)]
                } else {
                    0.0
                };

                let right = if i + 1 < n {
                    old[level.index(i + 1, j)]
                } else {
                    0.0
                };

                let down = if j > 0 {
                    old[level.index(i, j - 1)]
                } else {
                    0.0
                };

                let up = if j + 1 < n {
                    old[level.index(i, j + 1)]
                } else {
                    0.0
                };

                let idx = level.index(i, j);
                let jacobi_value =
                    0.25 * (h2 * level.f[idx] + left + right + down + up);

                level.u[idx] =
                    (1.0 - omega) * old[idx] + omega * jacobi_value;
            }
        }
    }
}

fn restrict_full_weighting(fine: &[f64], fine_n: usize) -> Vec<f64> {
    let coarse_n = (fine_n - 1) / 2;
    let mut coarse = vec![0.0; coarse_n * coarse_n];

    let fi = |i: usize, j: usize| -> usize { j * fine_n + i };
    let ci = |i: usize, j: usize| -> usize { j * coarse_n + i };

    for jc in 0..coarse_n {
        for ic in 0..coarse_n {
            let i = 2 * ic + 1;
            let j = 2 * jc + 1;

            coarse[ci(ic, jc)] =
                0.25 * fine[fi(i, j)]
                    + 0.125
                        * (fine[fi(i - 1, j)]
                            + fine[fi(i + 1, j)]
                            + fine[fi(i, j - 1)]
                            + fine[fi(i, j + 1)])
                    + 0.0625
                        * (fine[fi(i - 1, j - 1)]
                            + fine[fi(i + 1, j - 1)]
                            + fine[fi(i - 1, j + 1)]
                            + fine[fi(i + 1, j + 1)]);
        }
    }

    coarse
}

fn coarse_value(coarse: &[f64], coarse_n: usize, ic: isize, jc: isize) -> f64 {
    if ic < 0 || jc < 0 || ic >= coarse_n as isize || jc >= coarse_n as isize {
        0.0
    } else {
        coarse[jc as usize * coarse_n + ic as usize]
    }
}

fn prolong_bilinear(coarse: &[f64], coarse_n: usize) -> Vec<f64> {
    let fine_n = 2 * coarse_n + 1;
    let mut fine = vec![0.0; fine_n * fine_n];

    let fi = |i: usize, j: usize| -> usize { j * fine_n + i };

    for j in 0..fine_n {
        for i in 0..fine_n {
            let value = match (i % 2, j % 2) {
                (1, 1) => {
                    let ic = (i - 1) / 2;
                    let jc = (j - 1) / 2;
                    coarse_value(coarse, coarse_n, ic as isize, jc as isize)
                }

                (0, 1) => {
                    let ic_right = i / 2;
                    let ic_left = ic_right as isize - 1;
                    let jc = (j - 1) / 2;

                    0.5 * (coarse_value(coarse, coarse_n, ic_left, jc as isize)
                        + coarse_value(coarse, coarse_n, ic_right as isize, jc as isize))
                }

                (1, 0) => {
                    let ic = (i - 1) / 2;
                    let jc_up = j / 2;
                    let jc_down = jc_up as isize - 1;

                    0.5 * (coarse_value(coarse, coarse_n, ic as isize, jc_down)
                        + coarse_value(coarse, coarse_n, ic as isize, jc_up as isize))
                }

                (0, 0) => {
                    let ic_right = i / 2;
                    let ic_left = ic_right as isize - 1;
                    let jc_up = j / 2;
                    let jc_down = jc_up as isize - 1;

                    0.25 * (coarse_value(coarse, coarse_n, ic_left, jc_down)
                        + coarse_value(coarse, coarse_n, ic_right as isize, jc_down)
                        + coarse_value(coarse, coarse_n, ic_left, jc_up as isize)
                        + coarse_value(coarse, coarse_n, ic_right as isize, jc_up as isize))
                }

                _ => unreachable!(),
            };

            fine[fi(i, j)] = value;
        }
    }

    fine
}

fn v_cycle(level: &mut GridLevel, pre_sweeps: usize, post_sweeps: usize, omega: f64) {
    if level.n <= 3 {
        weighted_jacobi_smooth(level, omega, 200);
        return;
    }

    weighted_jacobi_smooth(level, omega, pre_sweeps);

    let residual = compute_residual(level);
    let coarse_rhs = restrict_full_weighting(&residual, level.n);

    let coarse_n = (level.n - 1) / 2;
    let mut coarse = GridLevel::new(coarse_n);
    coarse.f = coarse_rhs;

    v_cycle(&mut coarse, pre_sweeps, post_sweeps, omega);

    let correction = prolong_bilinear(&coarse.u, coarse.n);

    for i in 0..level.u.len() {
        level.u[i] += correction[i];
    }

    weighted_jacobi_smooth(level, omega, post_sweeps);
}

fn main() {
    let mut fine = GridLevel::new(63);
    initialize_rhs(&mut fine);

    let omega = 2.0 / 3.0;
    let pre_sweeps = 3;
    let post_sweeps = 3;
    let cycles = 12;

    println!("Multigrid V-Cycle Solver for the 2D Poisson Equation");
    println!("====================================================");
    println!();

    println!("Finest Grid");
    println!("-----------");
    println!("Interior points per direction = {}", fine.n);
    println!("Grid spacing h                = {:.8}", fine.h);
    println!("Unknowns                      = {}", fine.u.len());
    println!();

    println!("V-Cycle Parameters");
    println!("------------------");
    println!("Weighted Jacobi omega         = {:.6}", omega);
    println!("Pre-smoothing sweeps          = {}", pre_sweeps);
    println!("Post-smoothing sweeps         = {}", post_sweeps);
    println!("Number of V-cycles            = {}", cycles);
    println!();

    println!("Convergence History");
    println!("-------------------");
    println!("{:>8} {:>18} {:>18}", "Cycle", "Residual Norm", "Max Error");

    println!(
        "{:>8} {:>18.6e} {:>18.6e}",
        0,
        residual_norm(&fine),
        max_abs_error(&fine)
    );

    for cycle in 1..=cycles {
        v_cycle(&mut fine, pre_sweeps, post_sweeps, omega);

        println!(
            "{:>8} {:>18.6e} {:>18.6e}",
            cycle,
            residual_norm(&fine),
            max_abs_error(&fine)
        );
    }
}
```

Program 20.6.3 demonstrates how multigrid methods combine smoothing and coarse-grid correction to achieve highly efficient solution of elliptic boundary value problems. Classical relaxation methods alone damp oscillatory error efficiently but struggle to remove smooth low-frequency error. Multigrid resolves this difficulty by transferring smooth error to coarser meshes where it becomes relatively more oscillatory and therefore more amenable to relaxation. The resulting interaction between smoothing and coarse-grid correction produces convergence rates that are dramatically faster than those of standalone stationary iterations.

The convergence history produced by the program illustrates this behaviour clearly. Each V-cycle reduces the residual norm by a substantial factor, yielding rapid convergence toward the discrete solution in only a small number of cycles. Eventually, the residual approaches machine precision while the maximum solution error stabilizes near the finite-difference discretization error associated with the mesh resolution. This separation between iterative error and discretization error is characteristic of well-designed multigrid solvers.

The implementation also demonstrates the recursive structure of multigrid algorithms. Each grid level applies the same sequence of smoothing, residual computation, restriction, coarse-grid correction, prolongation, and post-smoothing operations. This recursive organization allows the algorithm to maintain nearly optimal computational complexity while exploiting the smoothing property of relaxation across multiple spatial scales.

The modular structure of the program naturally supports extension to more advanced multigrid strategies. Alternative smoothers such as Gauss-Seidel, red-black relaxation, line smoothers, polynomial smoothers, and Chebyshev smoothers may be incorporated directly into the same framework. Likewise, geometric multigrid may be generalized to algebraic multigrid, adaptive multigrid, nonlinear multigrid, and multilevel Krylov-preconditioned solvers for large-scale scientific computing applications. Consequently, Program 20.6.3 provides both a practical demonstration of classical geometric multigrid and a foundation for understanding modern multilevel elliptic solvers.

## 20.6.4. Modern Relaxation, Parallel Smoothers, and Algebraic Multigrid

On modern hardware, the effectiveness of a relaxation method depends not only on its spectral damping properties but also on memory movement, data locality, synchronization, and parallelism. A smoother that is mathematically strong but has poor memory behaviour may be slower in practice than a simpler method with better locality. This is why contemporary multigrid implementations often modify classical relaxation through coloring, blocking, line relaxation, patch smoothers, or polynomial smoothers.

Colored Gauss-Seidel updates independent groups of unknowns simultaneously. For example, on a structured two-dimensional grid, red-black ordering separates the grid into two sets so that all red points can be updated in parallel using black values, and then all black points can be updated using red values. This improves parallelism while retaining some of the convergence advantages of Gauss-Seidel. More general multi-color methods extend the same idea to complex stencils and unstructured grids. Such methods remain relevant in high-performance finite-volume and aerodynamic simulations, including modern GPU implementations (Yang and Yang, 2025).

Block and line relaxation are especially useful when the PDE contains anisotropy. If coupling is much stronger in one direction than another, pointwise relaxation may damp error poorly. A line smoother solves along the strongly coupled direction, treating each line problem implicitly. This is closely related to the ADI and directional solver ideas discussed earlier in the chapter. Patch-based smoothers generalize this idea by solving small local subproblems. Recent work on geometric multigrid for higher-order finite elements shows that organizing smoothers around localized residual computations can significantly affect performance on cache-based many-core architectures (Wichrowski et al., 2025).

Polynomial smoothers, such as Chebyshev smoothers, avoid some sequential dependencies of Gauss-Seidel while preserving strong damping over a chosen spectral range. They are often attractive on parallel architectures because they can be implemented using sparse matrix-vector products and vector operations. Local Fourier analysis and performance models are important tools for selecting such smoothers in high-order discretizations and (p)-multigrid settings (Thompson, Brown and He, 2023; Wichrowski et al., 2025).

Algebraic multigrid extends the multigrid idea beyond structured grids. Instead of relying on a geometric hierarchy, AMG constructs coarse variables, interpolation operators, and coarse matrices directly from the algebraic properties of $A$. Classical AMG uses heuristics based on strength of connection, compatible relaxation, and interpolation quality. Recent research increasingly treats relaxation not merely as a smoothing step but as a principle for constructing the multigrid hierarchy itself. A 2024 study used a neural network to tune a key AMG coarsening parameter and reported substantial reductions in computational time for heterogeneous elliptic problems, while recent relaxation-based AMG theory develops compatible relaxation and coarse-grid construction from the dynamics of error reduction (Caldana, Antonietti and Dede’, 2024; Moussa and Kahl, 2026).

For textbook purposes, the hierarchy of ideas is clear. Pure relaxation is simple, transparent, and valuable for understanding error damping, but it is usually too slow as a complete solver for large two- and three-dimensional elliptic systems. Multigrid uses relaxation as a smoother and adds coarse-grid correction to remove smooth error efficiently. Algebraic multigrid extends this strategy to difficult geometries and variable coefficients. Specialized direct solvers such as Fourier methods and cyclic reduction remain superior when the domain and coefficients are sufficiently structured. The practical goal is to choose the simplest solver that exploits the structure of the discretized PDE strongly enough to achieve reliable and efficient convergence.

# 20.7. Multigrid Methods for Boundary Value Problems

Multigrid methods are among the most important solver technologies for elliptic boundary value problems and related implicit PDE subproblems. Their purpose is not to introduce a new discretization, but to solve efficiently the sparse linear systems that arise after finite-difference, finite-volume, finite-element, or spectral-element discretization. The central observation is that different error components are best treated on different spatial scales. Classical relaxation methods such as Jacobi and Gauss-Seidel rapidly damp oscillatory error on a fine grid, but they reduce smooth error very slowly. Multigrid completes the relaxation idea by transferring the remaining smooth error to a coarser grid, where it becomes more oscillatory relative to the coarse mesh and can again be reduced efficiently. This multiscale mechanism is the foundation of both geometric multigrid and algebraic multigrid, including modern learned, matrix-free, and accelerator-oriented variants (Wang et al., 2023; Caldana, Antonietti and Dede’, 2024; Tsai, Beams and Anzt, 2023).

## 20.7.1. Elliptic Model Problem and the Need for Multiple Scales

Consider the model elliptic boundary value problem:

\begin{equation}
\begin{aligned}
-\nabla\cdot\!\left(\kappa(x)\nabla u\right)&=f
\qquad \text{in } \Omega,\\
u&=g
\qquad \text{on } \partial\Omega
\end{aligned}
\tag{20.7.1}
\end{equation}

where $\kappa(x)>0$. After discretization on a mesh with spacing $h$, one obtains a sparse linear system:

$$A_hu_h=f_h \tag{20.7.2}$$

For example, the standard five-point discretization of the Poisson equation on a rectangular grid leads to a block tridiagonal matrix of the form:

\begin{equation}
A_h
=
\frac{1}{h^2}
\begin{bmatrix}
T  & -I &        &        & 0\\
-I & T  & -I     &        &  \\
   & \ddots & \ddots & \ddots &  \\
   &        & -I & T  & -I\\
0  &        &    & -I & T
\end{bmatrix}
\tag{20.7.3}
\end{equation}

where,

\begin{equation}
T=
\begin{bmatrix}
4  & -1 &        &        & 0\\
-1 & 4  & -1     &        &  \\
   & \ddots & \ddots & \ddots &  \\
   &        & -1 & 4  & -1\\
0  &        &    & -1 & 4
\end{bmatrix}
\tag{20.7.4}
\end{equation}

The matrix in (20.7.3) is sparse and highly structured. The main numerical difficulty is not that the matrix is dense, but that one-level stationary iterations become ineffective as the mesh is refined. Jacobi and Gauss-Seidel may reduce high-frequency error quickly, but the smooth components of the error decay very slowly. As $h\to 0$, these smooth components increasingly dominate the convergence history.

Multigrid solves this problem by assigning different error scales to different meshes. A smoother damps oscillatory fine-grid error. The remaining slowly varying error is represented on a coarser grid, where it can be corrected more economically. A useful conceptual picture is:

\begin{equation}
\begin{aligned}
\text{fine-grid error}
&\longrightarrow \text{smoothing}
\longrightarrow \text{smooth residual error}\\
&\longrightarrow \text{restriction}
\longrightarrow \text{coarse-grid problem}
\longrightarrow \text{coarse correction}\\
&\longrightarrow \text{prolongation}
\longrightarrow \text{fine-grid update}\\&\tag{20.7.5}
\end{aligned}
\end{equation}

This description is not merely heuristic. Let $u_h^{(m)}$ denote the current approximation on the fine grid. The algebraic error and residual are:

$$e_h=u_h-u_h^{(m)},\qquad r_h=f_h-A_hu_h^{(m)} \tag{20.7.6}$$

Since the exact discrete solution satisfies $A_hu_h=f_h$, the error satisfies the residual equation:

$$A_he_h=r_h \tag{20.7.7}$$

The purpose of coarse-grid correction is to solve an approximation to (20.7.7) on a coarser space. If the smoother has already removed the oscillatory part of $e_h$, then the remaining smooth error can be represented accurately on a mesh with spacing $H\approx 2h$. This is the mathematical heart of multigrid: relaxation handles high-frequency error, while coarse-grid correction handles the low-frequency error that relaxation leaves behind.

## 20.7.2. Two-Grid Correction and Error Propagation

Let $V_h$ denote the fine-grid space and $V_H$ the coarse-grid space. Let,

$$R:V_h\to V_H\tag{20.7.8}$$

be the restriction operator, and let:

$$P:V_H\to V_h\tag{20.7.9}$$

be the prolongation operator. Starting from the residual equation (20.7.7), the smooth error is approximated by $Pe_H$, where $e_H$ solves the coarse-grid problem:

$$A_He_H=Rr_h \tag{20.7.10}$$

There are two common ways to choose the coarse-grid operator $A_H$. The first is the Galerkin coarse operator,

$$A_H=RA_hP \tag{20.7.11}$$

The second is rediscretization on the coarse mesh,

$$A_H\approx \mathcal{L}_H \tag{20.7.12}$$

where $\mathcal{L}_H$ denotes the same differential operator discretized directly on the coarse grid. The Galerkin choice is especially natural for finite-element methods and algebraic multigrid because it preserves variational and algebraic structure. Rediscretization is common in geometric multigrid when the mesh hierarchy and differential operator are explicitly available.

Once the coarse error approximation has been computed, the fine-grid solution is corrected by:

$$u_h^{(m+1)} = u_h^{(m)}+Pe_H \tag{20.7.13}$$

A complete two-grid method places smoothing steps before and after this correction. If $S_{\mathrm{pre}}$ is the pre-smoothing error-propagation operator, $S_{\mathrm{post}}$ is the post-smoothing error-propagation operator, and $\nu_1,\nu_2$ denote the numbers of pre- and post-smoothing steps, then the two-grid error-propagation operator is:

$$E_{\mathrm{TG}} = S_{\mathrm{post}}^{\nu_2}\left(I-PA_H^{-1}RA_h\right)S_{\mathrm{pre}}^{\nu_1} \tag{20.7.14}$$

This expression exposes the design problem in multigrid. The smoother must reduce error components that are poorly represented on the coarse grid, and the coarse space must approximate the remaining smooth error accurately enough for $Pe_H$ to be an effective correction. Thus, multigrid convergence depends on the interplay between a smoothing property and an approximation property. These two properties guide the choice of relaxation method, transfer operators, and coarse-grid operator.

Modern algebraic multigrid research continues to focus on precisely these ingredients. Learning-based AMG methods may attempt to improve prolongation construction or tune setup parameters, such as strength thresholds, that strongly influence convergence and cost. Such methods do not replace the multigrid principle; they automate or improve parts of the hierarchy construction while preserving the same two-grid and multilevel correction logic (Wang et al., 2023; Caldana, Antonietti and Dede’, 2024).

### Rust Implementation

Following the discussion in Subsections 20.7.1 and 20.7.2 on elliptic model problems, residual equations, and two-grid correction, Program 20.7.1 provides a practical implementation of a geometric two-grid multigrid solver for the two-dimensional Poisson equation. The implementation combines weighted Jacobi smoothing, residual restriction, coarse-grid correction, and bilinear prolongation within a unified multilevel framework that directly reflects equations (20.7.1)–(20.7.14). Rather than relying solely on repeated stationary iteration, the program demonstrates the central multigrid principle that different error components should be treated on different spatial scales. Oscillatory high-frequency error is reduced efficiently through weighted Jacobi smoothing on the fine grid, while the remaining smooth error is transferred to a coarser mesh where it becomes relatively more oscillatory and can be corrected more economically. The implementation therefore illustrates how coarse-grid correction complements relaxation and why multigrid methods overcome the slow convergence behaviour of one-level stationary iterations on refined meshes.

At the core of the implementation is the `GridLevel` structure, which represents a single discretization level in the geometric multigrid hierarchy. Each level stores the number of interior grid points, the mesh spacing, the current approximate solution vector, and the corresponding right-hand-side vector. The methods `index` and `point` provide indexing and geometric coordinate access for the tensor-product finite-difference mesh. This design reflects the structured sparse discretization introduced in equations (20.7.2)–(20.7.4), where the two-dimensional Poisson operator is represented through nearest-neighbour stencil couplings on a regular rectangular grid.

The functions `exact_solution` and `rhs_function` define the analytic verification problem used throughout the program. The exact solution satisfies homogeneous Dirichlet boundary conditions naturally and generates a smooth forcing term for the Poisson equation. The function `initialize_rhs` evaluates this forcing term on the computational mesh and constructs the discrete right-hand side $f_h$ appearing in equation (20.7.2). This analytic construction allows the convergence of the multigrid correction process to be measured directly against a known exact solution.

The function `apply_poisson` implements the standard five-point finite-difference discretization of the Poisson operator. This discrete operator corresponds directly to the sparse elliptic matrix structure described in equations (20.7.3) and (20.7.4). For each interior grid point, the function computes the action of the discrete Laplacian using neighbouring stencil values together with homogeneous Dirichlet boundary conditions imposed implicitly through zero exterior values. The resulting operator is sparse, structured, and symmetric positive definite, making it an ideal model problem for multigrid methods.

The residual equation introduced in equations (20.7.6) and (20.7.7) is implemented through the function `compute_residual`. This function evaluates the difference between the discrete right-hand side and the action of the discrete operator on the current approximation, thereby producing the residual vector $r_h=f_h-A_hu_h^{(m)}$. The functions `l2_norm` and `residual_norm` compute Euclidean norms of the residual vector and therefore measure the degree to which the current approximation satisfies the discrete elliptic system. The function `max_abs_error` compares the numerical approximation against the exact analytic solution and measures the overall discretization accuracy independently of the iterative convergence process.

The weighted Jacobi smoother is implemented through the function `weighted_jacobi_smooth`. This function applies repeated diagonally scaled residual corrections corresponding to the smoothing process discussed in Subsection 20.7.1. The relaxation parameter (\\omega) controls the strength of the correction, while all grid values are updated simultaneously from the previous iterate. As emphasized in equations (20.7.5)–(20.7.7), relaxation rapidly damps oscillatory fine-grid error but leaves smooth low-frequency error relatively unchanged. The smoother therefore serves as the first stage of the multigrid correction process rather than as a standalone solver.

The functions `restrict_full_weighting` and `prolong_bilinear` implement the transfer operators introduced in equations (20.7.8) and (20.7.9). The restriction operator transfers the fine-grid residual to the coarse grid using full-weighting averaging, thereby preserving smooth residual structure while filtering oscillatory components that cannot be represented accurately on the coarser mesh. The prolongation operator transfers the coarse-grid correction back to the fine grid using bilinear interpolation. Together, these operators realize the coarse-grid correction framework central to equations (20.7.10)–(20.7.13).

The function `two_grid_cycle` implements the complete two-grid correction algorithm. The procedure begins by applying several pre-smoothing sweeps on the fine grid to damp high-frequency error. The residual equation from equation (20.7.7) is then restricted to the coarse mesh, producing the coarse-grid system corresponding to equation (20.7.10). The coarse problem is approximated through additional smoothing sweeps on the coarse grid, after which the resulting correction is prolongated back to the fine grid and added to the current approximation according to equation (20.7.13). Finally, post-smoothing sweeps are applied to remove oscillatory interpolation error introduced during prolongation. This sequence realizes the two-grid error-propagation process summarized in equation (20.7.14).

The `main` function coordinates the overall multigrid experiment. It constructs the fine computational grid, initializes the elliptic right-hand side, selects smoothing and transfer parameters, and repeatedly applies two-grid correction cycles while monitoring convergence. The printed convergence history demonstrates the characteristic behaviour of multigrid correction methods: the residual norm and solution error decrease steadily over successive cycles, illustrating how smoothing and coarse-grid correction complement one another in reducing different scales of error. Altogether, the implementation provides a practical realization of the two-grid multigrid framework developed throughout Subsections 20.7.1 and 20.7.2.

```rust
// Program 20.7.1. Two-Grid Geometric Multigrid Correction for the 2D Poisson Equation
//
// Problem statement:
// Solve the two-dimensional Poisson boundary value problem
//
//     -Delta u = f
//
// on the unit square with homogeneous Dirichlet boundary conditions.
// The program combines the elliptic model problem and residual equation
// from equations (20.7.1)-(20.7.7) with the two-grid correction framework
// from equations (20.7.8)-(20.7.14).

use std::f64::consts::PI;

#[derive(Clone)]
struct GridLevel {
    n: usize,
    h: f64,
    u: Vec<f64>,
    f: Vec<f64>,
}

impl GridLevel {
    fn new(n: usize) -> Self {
        assert!(n >= 3 && n % 2 == 1);

        let h = 1.0 / (n as f64 + 1.0);

        Self {
            n,
            h,
            u: vec![0.0; n * n],
            f: vec![0.0; n * n],
        }
    }

    fn index(&self, i: usize, j: usize) -> usize {
        j * self.n + i
    }

    fn point(&self, i: usize, j: usize) -> (f64, f64) {
        ((i as f64 + 1.0) * self.h, (j as f64 + 1.0) * self.h)
    }
}

fn exact_solution(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

fn rhs_function(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * exact_solution(x, y)
}

fn initialize_rhs(level: &mut GridLevel) {
    for j in 0..level.n {
        for i in 0..level.n {
            let (x, y) = level.point(i, j);
            let idx = level.index(i, j);
            level.f[idx] = rhs_function(x, y);
        }
    }
}

fn apply_poisson(level: &GridLevel, u: &[f64]) -> Vec<f64> {
    let n = level.n;
    let inv_h2 = 1.0 / (level.h * level.h);
    let mut au = vec![0.0; n * n];

    for j in 0..n {
        for i in 0..n {
            let center = u[level.index(i, j)];

            let left = if i > 0 { u[level.index(i - 1, j)] } else { 0.0 };
            let right = if i + 1 < n { u[level.index(i + 1, j)] } else { 0.0 };
            let down = if j > 0 { u[level.index(i, j - 1)] } else { 0.0 };
            let up = if j + 1 < n { u[level.index(i, j + 1)] } else { 0.0 };

            au[level.index(i, j)] =
                (4.0 * center - left - right - down - up) * inv_h2;
        }
    }

    au
}

fn compute_residual(level: &GridLevel) -> Vec<f64> {
    let au = apply_poisson(level, &level.u);

    level
        .f
        .iter()
        .zip(au.iter())
        .map(|(fi, aui)| fi - aui)
        .collect()
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn residual_norm(level: &GridLevel) -> f64 {
    l2_norm(&compute_residual(level))
}

fn max_abs_error(level: &GridLevel) -> f64 {
    let mut max_error = 0.0;

    for j in 0..level.n {
        for i in 0..level.n {
            let (x, y) = level.point(i, j);
            let idx = level.index(i, j);
            max_error = f64::max(max_error, (level.u[idx] - exact_solution(x, y)).abs());
        }
    }

    max_error
}

fn weighted_jacobi_smooth(level: &mut GridLevel, omega: f64, sweeps: usize) {
    let n = level.n;
    let h2 = level.h * level.h;

    for _ in 0..sweeps {
        let old = level.u.clone();

        for j in 0..n {
            for i in 0..n {
                let left = if i > 0 { old[level.index(i - 1, j)] } else { 0.0 };
                let right = if i + 1 < n { old[level.index(i + 1, j)] } else { 0.0 };
                let down = if j > 0 { old[level.index(i, j - 1)] } else { 0.0 };
                let up = if j + 1 < n { old[level.index(i, j + 1)] } else { 0.0 };

                let idx = level.index(i, j);
                let jacobi_value = 0.25 * (h2 * level.f[idx] + left + right + down + up);

                level.u[idx] = (1.0 - omega) * old[idx] + omega * jacobi_value;
            }
        }
    }
}

fn restrict_full_weighting(fine: &[f64], fine_n: usize) -> Vec<f64> {
    let coarse_n = (fine_n - 1) / 2;
    let mut coarse = vec![0.0; coarse_n * coarse_n];

    let fi = |i: usize, j: usize| -> usize { j * fine_n + i };
    let ci = |i: usize, j: usize| -> usize { j * coarse_n + i };

    for jc in 0..coarse_n {
        for ic in 0..coarse_n {
            let i = 2 * ic + 1;
            let j = 2 * jc + 1;

            coarse[ci(ic, jc)] =
                0.25 * fine[fi(i, j)]
                    + 0.125
                        * (fine[fi(i - 1, j)]
                            + fine[fi(i + 1, j)]
                            + fine[fi(i, j - 1)]
                            + fine[fi(i, j + 1)])
                    + 0.0625
                        * (fine[fi(i - 1, j - 1)]
                            + fine[fi(i + 1, j - 1)]
                            + fine[fi(i - 1, j + 1)]
                            + fine[fi(i + 1, j + 1)]);
        }
    }

    coarse
}

fn coarse_value(coarse: &[f64], coarse_n: usize, ic: isize, jc: isize) -> f64 {
    if ic < 0 || jc < 0 || ic >= coarse_n as isize || jc >= coarse_n as isize {
        0.0
    } else {
        coarse[jc as usize * coarse_n + ic as usize]
    }
}

fn prolong_bilinear(coarse: &[f64], coarse_n: usize) -> Vec<f64> {
    let fine_n = 2 * coarse_n + 1;
    let mut fine = vec![0.0; fine_n * fine_n];

    let fi = |i: usize, j: usize| -> usize { j * fine_n + i };

    for j in 0..fine_n {
        for i in 0..fine_n {
            fine[fi(i, j)] = match (i % 2, j % 2) {
                (1, 1) => {
                    let ic = (i - 1) / 2;
                    let jc = (j - 1) / 2;
                    coarse_value(coarse, coarse_n, ic as isize, jc as isize)
                }
                (0, 1) => {
                    let ic_right = i / 2;
                    let ic_left = ic_right as isize - 1;
                    let jc = (j - 1) / 2;

                    0.5 * (coarse_value(coarse, coarse_n, ic_left, jc as isize)
                        + coarse_value(coarse, coarse_n, ic_right as isize, jc as isize))
                }
                (1, 0) => {
                    let ic = (i - 1) / 2;
                    let jc_up = j / 2;
                    let jc_down = jc_up as isize - 1;

                    0.5 * (coarse_value(coarse, coarse_n, ic as isize, jc_down)
                        + coarse_value(coarse, coarse_n, ic as isize, jc_up as isize))
                }
                (0, 0) => {
                    let ic_right = i / 2;
                    let ic_left = ic_right as isize - 1;
                    let jc_up = j / 2;
                    let jc_down = jc_up as isize - 1;

                    0.25 * (coarse_value(coarse, coarse_n, ic_left, jc_down)
                        + coarse_value(coarse, coarse_n, ic_right as isize, jc_down)
                        + coarse_value(coarse, coarse_n, ic_left, jc_up as isize)
                        + coarse_value(coarse, coarse_n, ic_right as isize, jc_up as isize))
                }
                _ => unreachable!(),
            };
        }
    }

    fine
}

fn two_grid_cycle(
    fine: &mut GridLevel,
    pre_sweeps: usize,
    post_sweeps: usize,
    coarse_sweeps: usize,
    omega: f64,
) {
    weighted_jacobi_smooth(fine, omega, pre_sweeps);

    let residual = compute_residual(fine);
    let coarse_rhs = restrict_full_weighting(&residual, fine.n);

    let coarse_n = (fine.n - 1) / 2;
    let mut coarse = GridLevel::new(coarse_n);
    coarse.f = coarse_rhs;

    weighted_jacobi_smooth(&mut coarse, omega, coarse_sweeps);

    let correction = prolong_bilinear(&coarse.u, coarse.n);

    for idx in 0..fine.u.len() {
        fine.u[idx] += correction[idx];
    }

    weighted_jacobi_smooth(fine, omega, post_sweeps);
}

fn main() {
    let mut fine = GridLevel::new(63);
    initialize_rhs(&mut fine);

    let omega = 2.0 / 3.0;
    let pre_sweeps = 3;
    let post_sweeps = 3;
    let coarse_sweeps = 200;
    let cycles = 10;

    println!("Two-Grid Geometric Multigrid Correction for the 2D Poisson Equation");
    println!("===================================================================");
    println!();

    println!("Fine Grid");
    println!("---------");
    println!("Interior points per direction = {}", fine.n);
    println!("Grid spacing h                = {:.8}", fine.h);
    println!("Unknowns                      = {}", fine.u.len());
    println!();

    println!("Two-Grid Parameters");
    println!("-------------------");
    println!("Weighted Jacobi omega         = {:.6}", omega);
    println!("Pre-smoothing sweeps          = {}", pre_sweeps);
    println!("Post-smoothing sweeps         = {}", post_sweeps);
    println!("Coarse-grid smoothing sweeps  = {}", coarse_sweeps);
    println!("Two-grid cycles               = {}", cycles);
    println!();

    println!("Convergence History");
    println!("-------------------");
    println!("{:>8} {:>18} {:>18}", "Cycle", "Residual Norm", "Max Error");

    println!(
        "{:>8} {:>18.6e} {:>18.6e}",
        0,
        residual_norm(&fine),
        max_abs_error(&fine)
    );

    for cycle in 1..=cycles {
        two_grid_cycle(
            &mut fine,
            pre_sweeps,
            post_sweeps,
            coarse_sweeps,
            omega,
        );

        println!(
            "{:>8} {:>18.6e} {:>18.6e}",
            cycle,
            residual_norm(&fine),
            max_abs_error(&fine)
        );
    }
}
```

Program 20.7.1 demonstrates how multigrid methods overcome the limitations of classical one-level relaxation schemes by combining smoothing with coarse-grid correction. Weighted Jacobi smoothing efficiently removes oscillatory fine-grid error, but smooth low-frequency error decays only slowly under relaxation alone. The two-grid correction framework addresses this weakness by transferring the smooth residual error to a coarser mesh where it becomes relatively more oscillatory and can therefore be reduced more effectively.

The convergence history produced by the program illustrates this multiscale behaviour clearly. Each two-grid cycle substantially reduces both the residual norm and the solution error, even though only a single coarse level is employed. Compared with standalone Jacobi iteration, the inclusion of coarse-grid correction significantly accelerates convergence by addressing the slowly decaying smooth error components that dominate one-level relaxation on refined meshes.

The implementation also highlights the structural importance of transfer operators and coarse-grid construction in multigrid design. The effectiveness of the method depends not only on the smoother itself, but also on the interaction between restriction, prolongation, and the approximation properties of the coarse space. This interplay between smoothing and approximation is the mathematical foundation underlying modern geometric and algebraic multigrid methods.

The modular organization of the code naturally supports extension to more advanced multilevel algorithms. Recursive V-cycles, W-cycles, full multigrid strategies, algebraic multigrid hierarchies, Krylov-preconditioned multigrid methods, and GPU-oriented matrix-free implementations may all be constructed by extending the same fundamental framework of smoothing, residual transfer, coarse correction, and interpolation. Consequently, Program 20.7.1 serves both as a concrete implementation of the classical two-grid method and as a conceptual foundation for modern large-scale multigrid solvers used throughout scientific computing.

## 20.7.3. Multilevel Cycles, Complexity, and Krylov Preconditioning

The multilevel method is obtained by applying the two-grid idea recursively. A V-cycle uses one recursive call to the next coarser level at each stage. A W-cycle revisits coarse levels more aggressively and is therefore more expensive but can be more robust. Full multigrid begins on the coarsest grid, solves there, and interpolates the approximation upward through progressively finer grids, applying correction cycles at each level.

Let $N_\ell$ denote the number of unknowns on level $\ell$, with $\ell=0$ representing the finest grid. Under standard geometric coarsening in $d$ dimensions,

$$N_\ell \approx \frac{N_0}{2^{d\ell}}\tag{20.7.15}$$

If the work on level $\ell$ is proportional to $N_\ell$, say $cN_\ell$, then the work of one V-cycle satisfies:

\begin{equation}
\sum_{\ell=0}^{L} cN_\ell
\approx
cN_0\sum_{\ell=0}^{L}2^{-d\ell}
=
O(N_0)
\tag{20.7.16}
\end{equation}

The memory footprint is also $O(N_0)$ when the hierarchy and a small number of work vectors are stored. This linear-complexity estimate is the reason multigrid is often described as an optimal solver for elliptic problems. The word optimal, however, must be interpreted carefully in modern computing. Actual performance depends not only on arithmetic complexity, but also on setup cost, data movement, communication, cache locality, memory bandwidth, precision choices, and parallel synchronization. Recent multigrid research therefore emphasizes matrix-free kernels, mixed precision, communication reduction, GPU-resident implementation, and multilevel methods designed for high-order discretizations (Tsai, Beams and Anzt, 2023; Richardson et al., 2025; Ohm, Harper and Jansson, 2026).

It is useful to compare multigrid with classical iterative and Krylov methods. Jacobi and Gauss-Seidel have low cost per iteration, but the number of iterations usually grows rapidly under mesh refinement. Krylov methods such as conjugate gradients and GMRES are more powerful, but without an effective preconditioner their convergence can still deteriorate as (h\\to 0). Multigrid is therefore frequently used not only as a standalone solver, but also as a preconditioner for Krylov methods. In this role, the multigrid cycle reduces error components that would otherwise slow the Krylov iteration. This combination is now standard in many large-scale PDE codes, including cardiac electrophysiology, astrophysical self-gravity, and high-order computational fluid dynamics (Tomida and Stone, 2023; Centofanti and Scacchi, 2024; Centofanti et al., 2025).

From an implementation perspective, the multigrid hierarchy should be represented explicitly. A Rust implementation can store each level as a data structure containing the local operator, smoother state, restriction and prolongation maps, coarse-grid operator, and reusable work buffers. The recursive V-cycle can then be implemented as a method on a hierarchy object. Matrix-free stencil application can be described through traits over slices, strided arrays, or element-local data. This mirrors the mathematical structure of (20.7.14): each level contributes a smoother, a residual computation, a transfer operation, and a correction.

## 20.7.4. Geometric Multigrid, Algebraic Multigrid, and Applications

There are two major branches of multigrid. Geometric multigrid assumes access to a mesh hierarchy and transfer operators derived from geometry. It is especially effective for structured grids, nested finite-element spaces, spectral-element discretizations, and matrix-free high-order methods. Algebraic multigrid constructs its hierarchy from the matrix itself. This makes AMG indispensable when the mesh is unstructured, when the geometry is complicated, or when the linear system comes from a complex multiphysics discretization for which explicit geometric coarsening is inconvenient.

Recent developments have strengthened both branches. Learning-based AMG methods target prolongation operators and setup-parameter tuning (Wang et al., 2023; Caldana, Antonietti and Dede’, 2024). GPU-oriented AMG uses mixed or three-precision arithmetic to reduce memory traffic while maintaining sufficient numerical robustness (Tsai, Beams and Anzt, 2023). Geometric and algebraic multigrid are also being pushed into matrix-free, high-order, and multi-GPU workflows, where the cost of assembling and storing sparse matrices may be prohibitive (Richardson et al., 2025; Ohm, Harper and Jansson, 2026). These trends reinforce the basic point that multigrid is not a single algorithm, but a hierarchy design principle.

A particularly important application is cardiac electrophysiology. The bidomain model describes the electrical activity of cardiac tissue through coupled diffusion-reaction equations for intra- and extracellular potentials. More detailed EMI models resolve cell-level structure more explicitly. A simplified bidomain form is:

\begin{equation}
\chi C_m\,\partial_t v
=
\nabla\cdot(\sigma_i\nabla v)
+
\nabla\cdot(\sigma_i\nabla u_e)
-
\chi I_{\mathrm{ion}}(v,w)
+
I_{\mathrm{app}}
\tag{20.7.17}
\end{equation}

together with,

\begin{equation}
-\nabla\cdot\!\left((\sigma_i+\sigma_e)\nabla u_e\right)
-
\nabla\cdot(\sigma_i\nabla v)
=
0
\tag{20.7.18}
\end{equation}

coupled to ordinary differential equations for ionic state variables $w$. After time discretization, these models require repeated solution of very large elliptic or block-elliptic systems. These linear systems are the computational bottleneck. Recent studies comparing AMG bidomain solvers on hybrid CPU-GPU architectures and developing parallel AMG solvers for cardiac EMI models show that multilevel preconditioning is an enabling technology for clinically and biologically relevant simulations at scale (Centofanti and Scacchi, 2024; Centofanti et al., 2025).

Another instructive application is self-gravity in astrophysical adaptive mesh refinement. The gravitational potential $\Phi$ satisfies the Poisson equation:

$$-\Delta \Phi=4\pi G\rho \tag{20.7.19}$$

which must be solved repeatedly on hierarchically refined meshes. In the Athena++ framework, multigrid solvers were integrated with adaptive mesh refinement and a task-based runtime. A conservative Laplacian formulation was used to avoid spurious forces at coarse-fine level boundaries. This example is important because it shows that multigrid is not only about asymptotic complexity on idealized domains. In production PDE codes, it is also about preserving physical fidelity across mesh hierarchies and refinement interfaces (Tomida and Stone, 2023).

The practical conclusion is that multigrid should be understood as a solver architecture rather than a single fixed method. The smoother, transfer operators, coarse-grid construction, and cycle type must be matched to the PDE, discretization, hardware, and desired accuracy. When this matching is successful, multigrid achieves nearly linear complexity and becomes one of the most effective tools for large elliptic and implicit PDE subproblems. When the ingredients are poorly matched, convergence may deteriorate or setup cost may dominate. The numerical art lies in exploiting the multiscale structure of the error while keeping memory movement, communication, and implementation complexity under control.

### Rust Implementation

Following the discussion in Section 20.7 on multigrid acceleration, coarse-grid correction, and Krylov-preconditioned elliptic solvers, Program 20.7.2 provides a practical implementation of a recursive multilevel V-cycle together with multigrid-preconditioned conjugate gradients for the two-dimensional Poisson equation. The implementation extends the earlier two-grid framework by constructing an entire hierarchy of successively coarser discretizations and recursively applying smoothing, restriction, coarse-grid correction, and prolongation across multiple levels. In numerical elliptic computation, classical relaxation alone becomes increasingly inefficient as the mesh is refined because smooth low-frequency error decays very slowly. This program demonstrates how multigrid overcomes that limitation by transferring smooth error to coarser meshes where it becomes relatively more oscillatory and therefore easier to reduce. The program also illustrates the modern role of multigrid as a preconditioner inside Krylov methods, where the V-cycle is used not as a standalone solver alone, but as a mechanism for rapidly improving the conditioning of the discrete elliptic operator. The framework therefore connects multilevel discretization, recursive error propagation, sparse elliptic structure, and iterative linear algebra into a unified computational strategy for large-scale boundary value problems.

At the core of the implementation is the `GridLevel` structure, which represents a single discretization level within the multigrid hierarchy. Each level stores the number of interior grid points, the mesh spacing, the approximate solution vector, and the corresponding right-hand side vector. The methods `index` and `point` provide structured access to tensor-product grid coordinates and flattened vector storage. This organization reflects the structured finite-difference discretization of the Poisson equation discussed throughout Section 20.7, where the elliptic operator is represented through nearest-neighbour stencil couplings on nested Cartesian meshes.

The `MultigridHierarchy` structure organizes the complete hierarchy of nested grids used by the recursive V-cycle. The constructor `new` recursively generates progressively coarser grids according to the mesh-sequencing process described in Equation (20.6.17). The hierarchy stores the relaxation parameter together with the numbers of pre-smoothing, post-smoothing, and coarse-grid smoothing sweeps. The methods `set_finest_rhs` and `set_finest_solution` initialize the finest-grid elliptic problem, while `total_unknowns_in_hierarchy` and `finest_unknowns` provide diagnostic information regarding multilevel storage complexity. These methods illustrate how multigrid methods require only modest additional storage beyond the finest-grid problem while dramatically improving convergence behaviour.

The functions `exact_solution` and `rhs_function` define the analytic verification problem used throughout the experiment. Rather than using a single eigenmode, the exact solution combines several Fourier components with different spatial frequencies. This construction is important because it generates a more representative elliptic problem containing both smooth and oscillatory structures. The associated right-hand side is obtained analytically from the continuous Poisson equation and is assembled on the computational mesh through the function `build_rhs`. This mixed-mode construction prevents conjugate gradients from converging artificially in one iteration and therefore produces a meaningful comparison between standard CG and multigrid-preconditioned CG.

The function `apply_poisson` implements the standard five-point finite-difference discretization of the two-dimensional Poisson operator. This discrete operator corresponds directly to the sparse elliptic matrix structure introduced earlier in Section 20.5. For each interior grid point, the routine evaluates the discrete Laplacian using neighbouring stencil values together with homogeneous Dirichlet boundary conditions imposed implicitly through zero exterior values. The resulting matrix is sparse, symmetric positive definite, and structured, making it particularly suitable for multigrid acceleration and Krylov iteration.

The residual equation introduced in Equation (20.6.12) is implemented through the function `compute_residual`, which evaluates the difference between the discrete right-hand side and the action of the discrete operator on the current approximation. The functions `dot`, `l2_norm`, and `axpy` provide the fundamental vector algebra required for both multigrid and conjugate-gradient iteration. The routine `max_abs_error` compares the numerical approximation against the analytic solution and measures the maximum pointwise discretization error independently of iterative convergence. Together, these routines provide the algebraic infrastructure required for monitoring residual reduction, error propagation, and Krylov orthogonalization.

The function `weighted_jacobi_smooth` implements the smoothing process central to multigrid theory. This applies repeated weighted Jacobi relaxation sweeps corresponding to Equation (20.6.3). As discussed in Section 20.6.2, weighted Jacobi rapidly damps oscillatory high-frequency error while leaving smooth low-frequency error relatively unchanged. The smoother therefore serves not as a complete solver, but as the first stage of the multilevel correction process. The relaxation parameter (\\omega) controls the strength of the smoothing correction and strongly influences convergence efficiency.

The functions `restrict_full_weighting` and `prolong_bilinear` implement the transfer operators introduced in Equations (20.6.13) and (20.6.15). The restriction operator transfers the fine-grid residual to the next coarser mesh using full-weighting averaging, thereby preserving smooth residual structure while filtering unresolved oscillatory components. The prolongation operator interpolates the coarse-grid correction back to the fine mesh using bilinear interpolation. These operators are essential because multigrid efficiency depends not only on smoothing, but also on the compatibility between coarse-grid representation and fine-grid error structure.

The recursive multilevel V-cycle is implemented through the method `v_cycle`. This routine realizes the multigrid correction process summarized in Equations (20.6.12)–(20.6.16). Each level applies several pre-smoothing sweeps, computes the fine-grid residual, restricts the residual to the next coarser mesh, recursively solves the coarse-grid error equation, prolongs the resulting correction back to the fine grid, and finally applies post-smoothing sweeps to remove interpolation-induced oscillatory error. At the coarsest level, the remaining elliptic system is approximately solved through repeated relaxation sweeps. The recursive structure therefore realizes the mesh hierarchy described conceptually in Equation (20.6.17).

The method `apply_v_cycle_preconditioner` illustrates the modern role of multigrid as a Krylov preconditioner. Instead of fully solving the elliptic system, the V-cycle approximately inverts the operator and therefore improves the conditioning of the linear system seen by conjugate gradients. This approximate inversion dramatically accelerates convergence for large structured elliptic problems because multigrid effectively reduces error components across all spatial frequencies.

The function `conjugate_gradient` implements the conjugate-gradient method for symmetric positive definite systems. The routine supports both unpreconditioned and multigrid-preconditioned variants. Residual vectors, search directions, and Krylov orthogonalization coefficients are updated iteratively according to the standard CG recurrence relations. When preconditioning is enabled, the residual vector is passed through a multigrid V-cycle before the Krylov update is formed. This combination illustrates the important modern principle that multigrid and Krylov methods are complementary rather than competing approaches: multigrid efficiently removes multiscale error, while Krylov iteration accelerates global convergence through subspace minimization.

The `main` function coordinates the entire multilevel experiment. It constructs the geometric hierarchy, initializes the elliptic right-hand side, performs standalone V-cycle iteration, and then compares plain conjugate gradients against multigrid-preconditioned conjugate gradients. The printed convergence histories demonstrate the rapid residual reduction achieved by recursive multigrid correction together with the acceleration obtained through preconditioning. Altogether, the implementation provides a complete realization of recursive geometric multigrid together with its modern role as a preconditioner for large sparse elliptic systems.

```rust
// Program 20.7.2: Recursive Multilevel V-Cycle and
// Multigrid-Preconditioned Conjugate Gradient Solver
// for the Two-Dimensional Poisson Equation
//
// Problem Statement
// -----------------
// This program implements a recursive geometric multigrid solver for the
// two-dimensional Poisson equation with homogeneous Dirichlet boundary
// conditions on the unit square:
//
//     -Δu = f
//
// using a standard five-point finite-difference discretization on a
// hierarchy of nested Cartesian grids.
//
// The implementation demonstrates the fundamental multigrid components:
//
// 1. Weighted Jacobi smoothing
// 2. Residual computation
// 3. Full-weighting restriction
// 4. Bilinear prolongation
// 5. Recursive V-cycle coarse-grid correction
// 6. Multigrid-preconditioned conjugate gradients (MGPCG)
//
// The program compares:
//
//     • Standalone multigrid V-cycle iteration
//     • Plain conjugate gradients
//     • Multigrid-preconditioned conjugate gradients
//
// Convergence is monitored through residual norms and maximum pointwise
// error relative to a manufactured analytical solution.
//
// The implementation illustrates how multigrid combines relaxation-based
// smoothing of oscillatory error with coarse-grid correction of smooth
// low-frequency error, achieving nearly optimal O(N) complexity for
// structured elliptic boundary value problems.
use std::f64::consts::PI;

#[derive(Clone)]
struct GridLevel {
    n: usize,
    h: f64,
    u: Vec<f64>,
    f: Vec<f64>,
}

impl GridLevel {
    fn new(n: usize) -> Self {
        assert!(n >= 3 && n % 2 == 1);

        let h = 1.0 / (n as f64 + 1.0);

        Self {
            n,
            h,
            u: vec![0.0; n * n],
            f: vec![0.0; n * n],
        }
    }

    fn index(&self, i: usize, j: usize) -> usize {
        j * self.n + i
    }

    fn point(&self, i: usize, j: usize) -> (f64, f64) {
        ((i as f64 + 1.0) * self.h, (j as f64 + 1.0) * self.h)
    }

    fn unknowns(&self) -> usize {
        self.n * self.n
    }
}

struct MultigridHierarchy {
    levels: Vec<GridLevel>,
    omega: f64,
    pre_sweeps: usize,
    post_sweeps: usize,
    coarse_sweeps: usize,
}

impl MultigridHierarchy {
    fn new(finest_n: usize, omega: f64, pre_sweeps: usize, post_sweeps: usize) -> Self {
        let mut levels = Vec::new();
        let mut n = finest_n;

        loop {
            levels.push(GridLevel::new(n));

            if n <= 3 {
                break;
            }

            n = (n - 1) / 2;
        }

        Self {
            levels,
            omega,
            pre_sweeps,
            post_sweeps,
            coarse_sweeps: 200,
        }
    }

    fn total_unknowns_in_hierarchy(&self) -> usize {
        self.levels.iter().map(|level| level.unknowns()).sum()
    }

    fn finest_unknowns(&self) -> usize {
        self.levels[0].unknowns()
    }

    fn set_finest_rhs(&mut self, rhs: &[f64]) {
        self.levels[0].f.copy_from_slice(rhs);
    }

    fn set_finest_solution(&mut self, u: &[f64]) {
        self.levels[0].u.copy_from_slice(u);
    }

    fn apply_v_cycle_as_solver(&mut self) {
        self.v_cycle(0);
    }

    fn apply_v_cycle_preconditioner(&mut self, residual: &[f64]) -> Vec<f64> {
        for level in self.levels.iter_mut() {
            level.u.fill(0.0);
            level.f.fill(0.0);
        }

        self.levels[0].f.copy_from_slice(residual);
        self.v_cycle(0);

        self.levels[0].u.clone()
    }

    fn v_cycle(&mut self, level_id: usize) {
        if self.levels[level_id].n <= 3 {
            weighted_jacobi_smooth(
                &mut self.levels[level_id],
                self.omega,
                self.coarse_sweeps,
            );
            return;
        }

        weighted_jacobi_smooth(
            &mut self.levels[level_id],
            self.omega,
            self.pre_sweeps,
        );

        let residual = compute_residual(&self.levels[level_id]);
        let coarse_rhs = restrict_full_weighting(&residual, self.levels[level_id].n);

        let next = level_id + 1;

        self.levels[next].u.fill(0.0);
        self.levels[next].f.copy_from_slice(&coarse_rhs);

        self.v_cycle(next);

        let correction = prolong_bilinear(&self.levels[next].u, self.levels[next].n);

        for i in 0..self.levels[level_id].u.len() {
            self.levels[level_id].u[i] += correction[i];
        }

        weighted_jacobi_smooth(
            &mut self.levels[level_id],
            self.omega,
            self.post_sweeps,
        );
    }
}

fn exact_solution(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
        + 0.25 * (3.0 * PI * x).sin() * (2.0 * PI * y).sin()
        + 0.10 * (7.0 * PI * x).sin() * (5.0 * PI * y).sin()
}

fn rhs_function(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * (PI * x).sin() * (PI * y).sin()
        + 0.25 * 13.0 * PI * PI * (3.0 * PI * x).sin() * (2.0 * PI * y).sin()
        + 0.10 * 74.0 * PI * PI * (7.0 * PI * x).sin() * (5.0 * PI * y).sin()
}

fn build_rhs(level: &GridLevel) -> Vec<f64> {
    let mut rhs = vec![0.0; level.unknowns()];

    for j in 0..level.n {
        for i in 0..level.n {
            let (x, y) = level.point(i, j);
            rhs[level.index(i, j)] = rhs_function(x, y);
        }
    }

    rhs
}

fn apply_poisson(level: &GridLevel, u: &[f64]) -> Vec<f64> {
    let n = level.n;
    let inv_h2 = 1.0 / (level.h * level.h);
    let mut au = vec![0.0; n * n];

    for j in 0..n {
        for i in 0..n {
            let center = u[level.index(i, j)];

            let left = if i > 0 { u[level.index(i - 1, j)] } else { 0.0 };
            let right = if i + 1 < n { u[level.index(i + 1, j)] } else { 0.0 };
            let down = if j > 0 { u[level.index(i, j - 1)] } else { 0.0 };
            let up = if j + 1 < n { u[level.index(i, j + 1)] } else { 0.0 };

            au[level.index(i, j)] =
                (4.0 * center - left - right - down - up) * inv_h2;
        }
    }

    au
}

fn compute_residual(level: &GridLevel) -> Vec<f64> {
    let au = apply_poisson(level, &level.u);

    level
        .f
        .iter()
        .zip(au.iter())
        .map(|(fi, aui)| fi - aui)
        .collect()
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn l2_norm(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for i in 0..x.len() {
        y[i] += alpha * x[i];
    }
}

fn max_abs_error(level: &GridLevel, u: &[f64]) -> f64 {
    let mut max_error = 0.0;

    for j in 0..level.n {
        for i in 0..level.n {
            let (x, y) = level.point(i, j);
            let idx = level.index(i, j);

            max_error = f64::max(max_error, (u[idx] - exact_solution(x, y)).abs());
        }
    }

    max_error
}

fn weighted_jacobi_smooth(level: &mut GridLevel, omega: f64, sweeps: usize) {
    let n = level.n;
    let h2 = level.h * level.h;

    for _ in 0..sweeps {
        let old = level.u.clone();

        for j in 0..n {
            for i in 0..n {
                let left = if i > 0 { old[level.index(i - 1, j)] } else { 0.0 };
                let right = if i + 1 < n { old[level.index(i + 1, j)] } else { 0.0 };
                let down = if j > 0 { old[level.index(i, j - 1)] } else { 0.0 };
                let up = if j + 1 < n { old[level.index(i, j + 1)] } else { 0.0 };

                let idx = level.index(i, j);
                let jacobi_value = 0.25 * (h2 * level.f[idx] + left + right + down + up);

                level.u[idx] = (1.0 - omega) * old[idx] + omega * jacobi_value;
            }
        }
    }
}

fn restrict_full_weighting(fine: &[f64], fine_n: usize) -> Vec<f64> {
    let coarse_n = (fine_n - 1) / 2;
    let mut coarse = vec![0.0; coarse_n * coarse_n];

    let fi = |i: usize, j: usize| -> usize { j * fine_n + i };
    let ci = |i: usize, j: usize| -> usize { j * coarse_n + i };

    for jc in 0..coarse_n {
        for ic in 0..coarse_n {
            let i = 2 * ic + 1;
            let j = 2 * jc + 1;

            coarse[ci(ic, jc)] =
                0.25 * fine[fi(i, j)]
                    + 0.125
                        * (fine[fi(i - 1, j)]
                            + fine[fi(i + 1, j)]
                            + fine[fi(i, j - 1)]
                            + fine[fi(i, j + 1)])
                    + 0.0625
                        * (fine[fi(i - 1, j - 1)]
                            + fine[fi(i + 1, j - 1)]
                            + fine[fi(i - 1, j + 1)]
                            + fine[fi(i + 1, j + 1)]);
        }
    }

    coarse
}

fn coarse_value(coarse: &[f64], coarse_n: usize, ic: isize, jc: isize) -> f64 {
    if ic < 0 || jc < 0 || ic >= coarse_n as isize || jc >= coarse_n as isize {
        0.0
    } else {
        coarse[jc as usize * coarse_n + ic as usize]
    }
}

fn prolong_bilinear(coarse: &[f64], coarse_n: usize) -> Vec<f64> {
    let fine_n = 2 * coarse_n + 1;
    let mut fine = vec![0.0; fine_n * fine_n];

    let fi = |i: usize, j: usize| -> usize { j * fine_n + i };

    for j in 0..fine_n {
        for i in 0..fine_n {
            fine[fi(i, j)] = match (i % 2, j % 2) {
                (1, 1) => {
                    let ic = (i - 1) / 2;
                    let jc = (j - 1) / 2;
                    coarse_value(coarse, coarse_n, ic as isize, jc as isize)
                }
                (0, 1) => {
                    let ic_right = i / 2;
                    let ic_left = ic_right as isize - 1;
                    let jc = (j - 1) / 2;

                    0.5 * (coarse_value(coarse, coarse_n, ic_left, jc as isize)
                        + coarse_value(coarse, coarse_n, ic_right as isize, jc as isize))
                }
                (1, 0) => {
                    let ic = (i - 1) / 2;
                    let jc_up = j / 2;
                    let jc_down = jc_up as isize - 1;

                    0.5 * (coarse_value(coarse, coarse_n, ic as isize, jc_down)
                        + coarse_value(coarse, coarse_n, ic as isize, jc_up as isize))
                }
                (0, 0) => {
                    let ic_right = i / 2;
                    let ic_left = ic_right as isize - 1;
                    let jc_up = j / 2;
                    let jc_down = jc_up as isize - 1;

                    0.25 * (coarse_value(coarse, coarse_n, ic_left, jc_down)
                        + coarse_value(coarse, coarse_n, ic_right as isize, jc_down)
                        + coarse_value(coarse, coarse_n, ic_left, jc_up as isize)
                        + coarse_value(coarse, coarse_n, ic_right as isize, jc_up as isize))
                }
                _ => unreachable!(),
            };
        }
    }

    fine
}

fn conjugate_gradient(
    hierarchy: &mut MultigridHierarchy,
    rhs: &[f64],
    max_iterations: usize,
    tolerance: f64,
    use_preconditioner: bool,
) -> (Vec<f64>, usize, f64) {
    let level = hierarchy.levels[0].clone();
    let n = rhs.len();

    let mut x = vec![0.0; n];
    let mut r = rhs.to_vec();

    let mut z = if use_preconditioner {
        hierarchy.apply_v_cycle_preconditioner(&r)
    } else {
        r.clone()
    };

    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);

    for iteration in 1..=max_iterations {
        let ap = apply_poisson(&level, &p);
        let alpha = rz_old / dot(&p, &ap);

        axpy(alpha, &p, &mut x);
        axpy(-alpha, &ap, &mut r);

        let residual = l2_norm(&r);

        if residual < tolerance {
            return (x, iteration, residual);
        }

        z = if use_preconditioner {
            hierarchy.apply_v_cycle_preconditioner(&r)
        } else {
            r.clone()
        };

        let rz_new = dot(&r, &z);
        let beta = rz_new / rz_old;

        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        rz_old = rz_new;
    }

    let residual = l2_norm(&r);
    (x, max_iterations, residual)
}

fn main() {
    let finest_n = 63;
    let omega = 2.0 / 3.0;
    let pre_sweeps = 3;
    let post_sweeps = 3;

    let mut hierarchy = MultigridHierarchy::new(finest_n, omega, pre_sweeps, post_sweeps);

    let rhs = build_rhs(&hierarchy.levels[0]);
    hierarchy.set_finest_rhs(&rhs);

    println!("Multilevel Multigrid V-Cycle and Krylov Preconditioning");
    println!("=======================================================");
    println!();

    println!("Hierarchy Information");
    println!("---------------------");
    println!("Finest grid n x n        = {} x {}", finest_n, finest_n);
    println!("Number of levels         = {}", hierarchy.levels.len());
    println!("Finest unknowns          = {}", hierarchy.finest_unknowns());
    println!(
        "Total hierarchy unknowns = {}",
        hierarchy.total_unknowns_in_hierarchy()
    );
    println!(
        "Hierarchy storage ratio  = {:.6}",
        hierarchy.total_unknowns_in_hierarchy() as f64
            / hierarchy.finest_unknowns() as f64
    );
    println!();

    println!("Standalone V-Cycle Solver");
    println!("-------------------------");
    println!("{:>8} {:>18} {:>18}", "Cycle", "Residual Norm", "Max Error");

    hierarchy.set_finest_solution(&vec![0.0; rhs.len()]);

    for cycle in 0..=10 {
        let residual = l2_norm(&compute_residual(&hierarchy.levels[0]));
        let error = max_abs_error(&hierarchy.levels[0], &hierarchy.levels[0].u);

        println!("{:>8} {:>18.6e} {:>18.6e}", cycle, residual, error);

        if cycle < 10 {
            hierarchy.apply_v_cycle_as_solver();
        }
    }

    println!();

    let cg_tolerance = 1.0e-8;
    let cg_max_iterations = 5000;

    let mut plain_hierarchy =
        MultigridHierarchy::new(finest_n, omega, pre_sweeps, post_sweeps);

    let (x_cg, iter_cg, res_cg) =
        conjugate_gradient(&mut plain_hierarchy, &rhs, cg_max_iterations, cg_tolerance, false);

    let mut pcg_hierarchy =
        MultigridHierarchy::new(finest_n, omega, pre_sweeps, post_sweeps);

    let (x_pcg, iter_pcg, res_pcg) =
        conjugate_gradient(&mut pcg_hierarchy, &rhs, cg_max_iterations, cg_tolerance, true);

    println!("Krylov Solver Comparison");
    println!("------------------------");
    println!(
        "{:>22} {:>14} {:>18} {:>18}",
        "Method", "Iterations", "Residual", "Max Error"
    );

    println!(
        "{:>22} {:>14} {:>18.6e} {:>18.6e}",
        "Plain CG",
        iter_cg,
        res_cg,
        max_abs_error(&plain_hierarchy.levels[0], &x_cg)
    );

    println!(
        "{:>22} {:>14} {:>18.6e} {:>18.6e}",
        "MG-Preconditioned CG",
        iter_pcg,
        res_pcg,
        max_abs_error(&pcg_hierarchy.levels[0], &x_pcg)
    );
}
```

Program 20.7.2 demonstrates the fundamental multilevel principle underlying modern elliptic solvers: relaxation efficiently damps oscillatory error, while coarse-grid correction efficiently removes smooth error that would otherwise decay extremely slowly under stationary iteration alone. By recursively combining these two mechanisms across a hierarchy of grids, the V-cycle achieves rapid convergence with computational complexity that scales nearly linearly with the number of unknowns.

The convergence history illustrates the characteristic behaviour of recursive multigrid methods. Residual norms decrease rapidly during successive V-cycles, while the solution error approaches the discretization limit imposed by the finite-difference approximation itself. Once the algebraic residual becomes sufficiently small, further iteration no longer reduces the discretization error significantly because the remaining error is dominated by the spatial approximation rather than incomplete iterative convergence.

The comparison between plain conjugate gradients and multigrid-preconditioned conjugate gradients illustrates the complementary relationship between multigrid and Krylov subspace methods. Conjugate gradients alone already converges efficiently for structured Poisson systems, but multigrid preconditioning further improves convergence by reducing the condition number seen by the Krylov iteration. In large-scale scientific computing, this combination of multilevel preconditioning and Krylov acceleration forms one of the most powerful and widely used approaches for solving sparse elliptic systems.

The modular structure of the implementation naturally supports extension to more advanced multigrid strategies. Recursive W-cycles, full multigrid initialization, algebraic multigrid hierarchies, matrix-free smoothers, block relaxation, domain decomposition, GPU-oriented multilevel kernels, and adaptive mesh refinement frameworks may all be constructed by extending the same hierarchy-based correction framework implemented here. Consequently, Program 20.7.2 serves both as a practical implementation of recursive geometric multigrid and as a foundation for modern large-scale elliptic solvers used throughout computational science and engineering.

# 20.8. Spectral Methods

Spectral methods form a high-order discretization framework for differential equations. In contrast to finite-difference and finite-volume methods, which approximate derivatives and fluxes through local stencils, spectral methods approximate the unknown solution by expansions in globally smooth basis functions or high-order polynomial spaces. Their central advantage is accuracy. When the exact solution is smooth, and especially when it is analytic, spectral approximations can converge extremely rapidly as the number of degrees of freedom increases. For suitable problems, this means that a highly accurate solution may be obtained with far fewer unknowns than would be required by low-order local methods.

The distinction between multigrid and spectral methods is useful. Multigrid is primarily a solver technology for sparse linear systems that arise after discretization. Spectral methods are primarily a discretization technology that replaces low-order local approximation by high-order or global approximation. In modern scientific computing, the two ideas are often combined: one may discretize a PDE with spectral, spectral-element, or high-order finite-element bases, and then solve the resulting algebraic systems using multigrid, algebraic multigrid, or multilevel preconditioners. Recent work demonstrates this complementarity in GPU-resident spectral-element solvers, high-order CFD workflows, cardiac simulations, adaptive mesh-refined gravity solvers, and matrix-free multilevel methods (Hafeez and Krawczuk, 2023; Tomida and Stone, 2023; Tsai, Beams and Anzt, 2023; Liu, Shen and Zhang, 2024; Richardson et al., 2025; Ohm, Harper and Jansson, 2026).

## 20.8.1. Spectral Approximation Spaces

On a periodic interval, the canonical spectral basis is the Fourier basis. A periodic function may be approximated by the truncated expansion:

$$u_N(x) = \sum_{|k|\le N/2}\widehat{u}_k e^{ikx} \tag{20.8.1}$$

The Fourier coefficients $\widehat{u}_k$ represent the amplitudes of the retained oscillatory modes. For smooth periodic functions, these coefficients often decay rapidly with $|k|$, and the truncation error becomes small with relatively few modes. This is why Fourier spectral and pseudospectral methods are natural for periodic wave propagation, turbulence in periodic boxes, acoustics, and many idealized fluid problems.

On nonperiodic intervals, polynomial bases are usually preferred. Two common choices are Chebyshev and Legendre expansions,

$$u_N(x) = \sum_{n=0}^{N}a_nT_n(x) \tag{20.8.2}$$

or

$$u_N(x) = \sum_{n=0}^{N}b_nP_n(x) \tag{20.8.3}$$

where $T_n$ denotes the Chebyshev polynomial of degree $n$, and $P_n$ denotes the Legendre polynomial of degree $n$. These bases are especially effective on bounded intervals because they provide high-order approximation while retaining useful orthogonality and quadrature properties.

The main payoff is rapid convergence. For sufficiently smooth solutions, the approximation error decreases faster than any fixed algebraic power of $N$ in many classical settings. For analytic solutions, one often observes geometric or exponential-type convergence. This is the principal reason spectral methods remain attractive for smooth elliptic problems, wave propagation, acoustics, incompressible flow, eigenvalue problems, and high-accuracy benchmark computations. Modern work on spectral finite elements, point-cloud spectral methods, collocation preconditioning, and harmonic-map-based spectral methods reinforces the same conclusion: when the solution is smooth and the geometry and coefficients can be handled cleanly, spectral approximation can deliver much more accuracy per degree of freedom than low-order discretizations (Hafeez and Krawczuk, 2023; Yan, Jiang and Harlim, 2023; Javeed et al., 2025; Shi et al., 2026).

This accuracy, however, is not automatic for every PDE. If the solution contains shocks, discontinuities, sharp internal layers, rough coefficients, or geometric singularities, the high-order convergence may degrade substantially. Spectral methods are therefore most effective when the smoothness assumptions behind the basis are consistent with the solution being approximated. When locality, discontinuity handling, or complex geometry becomes important, spectral-element, discontinuous Galerkin spectral-element, embedded-boundary, or mapped-domain methods are often used to retain much of the spectral advantage while reducing the limitations of purely global approximation.

## 20.8.2. Galerkin, Tau, and Collocation Formulations

There are three classical formulations of spectral methods: Galerkin, tau, and collocation. They differ mainly in how the residual of the differential equation is enforced and how boundary conditions are incorporated.

In a spectral Galerkin method, one chooses a finite-dimensional trial space $V_N$ and seeks $u_N\in V_N$ such that the residual is orthogonal to a set of test functions. For a differential operator $L$, the Galerkin condition is:

$$(Lu_N-f,\phi_j)=0,\qquad j=1,\ldots,N \tag{20.8.4}$$

where $(\cdot,\cdot)$ denotes an appropriate inner product and $\phi_j$ are basis or test functions. Galerkin methods are often natural for elliptic problems because they arise directly from weak formulations and can incorporate boundary conditions through the choice of trial space.

In a tau method, the residual orthogonality is imposed except for a finite number of correction terms that are used to enforce boundary conditions. This approach is historically important in spectral theory because it allows one to work with convenient polynomial expansions while modifying only a small number of equations to impose constraints.

In a collocation, or pseudospectral, method, one enforces the differential equation at carefully selected nodes. For Chebyshev methods on $[-1,1]$, a common choice is the Chebyshev-Gauss-Lobatto grid,

$$x_j=\cos\!\left(\frac{j\pi}{N}\right),\qquad j=0,\ldots,N \tag{20.8.5}$$

If $u_j=u(x_j)$, then differentiation is approximated by a differentiation matrix:

$$u'(x_i)\approx\sum_{j=0}^{N}D_{ij}u(x_j) \tag{20.8.6}$$

Higher derivatives are obtained either by constructing higher-order differentiation matrices or by applying $D$ repeatedly, subject to conditioning and stability considerations. Collocation methods are often straightforward to implement because the unknowns are nodal values and nonlinear terms can be evaluated directly in physical space.

Fourier spectral methods have an especially simple differentiation rule in coefficient space. If,

$$u(x)=\sum_k\widehat{u}_k e^{ikx}$$

then,

$$\widehat{\partial_x u}_k = ik\widehat{u}_k \tag{20.8.7}$$

and

$$\widehat{-\partial_{xx}u}_k = k^2\widehat{u}_k \tag{20.8.8}$$

Thus, differentiation becomes diagonal in Fourier coefficient space. This is one reason Fourier methods are so elegant: differential operators with constant coefficients become simple multipliers on spectral coefficients. In physical space one evaluates products or nonlinearities; in coefficient space one evaluates derivatives or solves diagonal linear systems. This physical-space and coefficient-space duality is the foundation of Fourier pseudospectral algorithms.

### Rust Implementation

Following the discussion in Section 20.8.2 on Galerkin, tau, and collocation formulations, Program 20.8.1 presents a practical implementation of Fourier pseudospectral differentiation on a periodic domain. The program demonstrates how the Fourier representation of a smooth periodic function converts differentiation from a local finite-difference operation into a diagonal operation in coefficient space. Rather than approximating derivatives through neighboring stencil values, the method transforms the sampled function into Fourier coefficients, applies the spectral differentiation multipliers associated with equations (20.8.7) and (20.8.8), and reconstructs the differentiated field through an inverse transform. This coefficient-space formulation illustrates one of the principal advantages of Fourier spectral methods: for smooth periodic problems, derivatives can be computed with extremely high accuracy while preserving the global structure of the solution. The implementation also emphasizes the distinction between physical-space and coefficient-space representations, which is fundamental to modern pseudospectral algorithms used in computational fluid dynamics, wave propagation, and high-order PDE solvers.

At the core of the implementation is the `Complex` structure, which provides a lightweight representation of complex arithmetic required for the discrete Fourier transform. The structure defines the real and imaginary components of a complex number and implements the algebraic operations needed for spectral coefficient manipulation, including addition, subtraction, multiplication, scalar multiplication, and division. Auxiliary methods such as `from_polar` and `abs` are used to construct complex exponential factors and evaluate coefficient magnitudes. Since Fourier spectral methods naturally operate in the complex plane through the basis functions $e^{ikx}$, this structure forms the mathematical foundation of the transform operations implemented throughout the program.

The function `wave_number` establishes the correspondence between discrete transform indices and signed Fourier wave numbers. In discrete periodic Fourier analysis, the transform indices above the Nyquist midpoint represent negative frequencies rather than additional positive modes. This function therefore converts the raw array index into the appropriate signed spectral mode $k$, ensuring that the differentiation multipliers in equations (20.8.7) and (20.8.8) are applied consistently across both positive and negative frequencies. Correct wave-number indexing is essential in Fourier pseudospectral methods because spectral differentiation depends directly on the algebraic properties of the Fourier basis.

The `forward_dft` function computes the discrete Fourier transform of the sampled nodal values. Given function values $u(x_j)$ on a periodic grid, the routine constructs the Fourier coefficients $\widehat{u}_k$ by summing complex exponential contributions from all grid points. Although the implementation uses a direct $O(N^2)$ transform rather than a fast Fourier transform, the explicit summation makes the underlying spectral representation transparent and pedagogically useful. The resulting coefficients correspond directly to the truncated Fourier expansion introduced earlier in Section 20.8.2. The companion function `inverse_dft` reconstructs physical-space nodal values from the spectral coefficients. Together, these two routines implement the duality between physical space and coefficient space that underlies Fourier pseudospectral algorithms.

The functions `spectral_first_derivative` and `spectral_negative_second_derivative` implement the spectral differentiation rules associated with equations (20.8.7) and (20.8.8). The first derivative is obtained by multiplying each Fourier coefficient by the factor $ik$, while the negative second derivative is obtained by multiplication with $k^2$. Because these operations are diagonal in coefficient space, differentiation becomes algebraically simple once the Fourier representation has been constructed. This diagonal structure is one of the defining computational advantages of Fourier spectral methods, especially for constant-coefficient differential operators on periodic domains.

The program also includes the functions `test_function`, `exact_first_derivative`, and `exact_negative_second_derivative`, which define the analytic test problem used for verification. The chosen function,

$$u(x)=\sin(x)+\frac{1}{2}\cos(3x),$$

contains only a small number of Fourier modes and is therefore exactly representable on the chosen periodic grid. Its analytic derivatives are known explicitly, making it possible to compare the pseudospectral approximations directly against exact values. The helper function `max_error` computes the maximum absolute difference between the numerical and exact solutions, providing a quantitative measure of spectral accuracy.

The `main` function serves to demonstrate the complete Fourier pseudospectral workflow on a periodic interval. It begins by constructing a uniform periodic grid on $[0,2\pi)$ and sampling the test function at the nodal points. The sampled values are transformed into Fourier coefficient space using `forward_dft`, after which the spectral differentiation operators are applied through `spectral_first_derivative` and `spectral_negative_second_derivative` functions. The differentiated coefficients are then transformed back into physical space through `inverse_dft`. Finally, the numerical derivatives are compared against the analytic derivatives, and the maximum errors are reported. The printed Fourier coefficients also reveal the modal structure of the solution, confirming that only the modes $k=\pm1$ and $k=\pm3$ are present. This output demonstrates the spectral sparsity and near machine-precision accuracy characteristic of Fourier pseudospectral methods for smooth periodic functions.

```rust
// Program 20.8.1: Fourier Pseudospectral Differentiation
//
// Problem statement:
// Approximate the first derivative u'(x) and the negative second derivative
// -u''(x) of a smooth periodic function on [0, 2*pi) using Fourier
// pseudospectral differentiation. The program demonstrates the coefficient-
// space rules
//
//     d/dx        -> i k,
//     -d^2/dx^2  -> k^2,
//
// which correspond to equations (20.8.7) and (20.8.8).

use std::f64::consts::PI;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

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

    fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
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

impl Mul<f64> for Complex {
    type Output = Complex;

    fn mul(self, rhs: f64) -> Complex {
        Complex::new(self.re * rhs, self.im * rhs)
    }
}

impl Div<f64> for Complex {
    type Output = Complex;

    fn div(self, rhs: f64) -> Complex {
        Complex::new(self.re / rhs, self.im / rhs)
    }
}

fn wave_number(index: usize, n: usize) -> i32 {
    if index <= n / 2 {
        index as i32
    } else {
        index as i32 - n as i32
    }
}

fn forward_dft(values: &[f64]) -> Vec<Complex> {
    let n = values.len();
    let mut coeffs = vec![Complex::zero(); n];

    for k in 0..n {
        let mut sum = Complex::zero();

        for (j, &value) in values.iter().enumerate() {
            let angle = -2.0 * PI * (k as f64) * (j as f64) / (n as f64);
            let phase = Complex::from_polar(1.0, angle);
            sum += phase * value;
        }

        coeffs[k] = sum / (n as f64);
    }

    coeffs
}

fn inverse_dft(coeffs: &[Complex]) -> Vec<f64> {
    let n = coeffs.len();
    let mut values = vec![0.0; n];

    for j in 0..n {
        let mut sum = Complex::zero();

        for (k, &coeff) in coeffs.iter().enumerate() {
            let angle = 2.0 * PI * (k as f64) * (j as f64) / (n as f64);
            let phase = Complex::from_polar(1.0, angle);
            sum += coeff * phase;
        }

        values[j] = sum.re;
    }

    values
}

fn spectral_first_derivative(coeffs: &[Complex]) -> Vec<Complex> {
    let n = coeffs.len();
    let mut differentiated = vec![Complex::zero(); n];

    for k_index in 0..n {
        let k = wave_number(k_index, n) as f64;

        // Multiplication by i k.
        let multiplier = Complex::new(0.0, k);
        differentiated[k_index] = multiplier * coeffs[k_index];
    }

    differentiated
}

fn spectral_negative_second_derivative(coeffs: &[Complex]) -> Vec<Complex> {
    let n = coeffs.len();
    let mut differentiated = vec![Complex::zero(); n];

    for k_index in 0..n {
        let k = wave_number(k_index, n) as f64;

        // Multiplication by k^2 for the operator -d^2/dx^2.
        differentiated[k_index] = coeffs[k_index] * (k * k);
    }

    differentiated
}

fn test_function(x: f64) -> f64 {
    x.sin() + 0.5 * (3.0 * x).cos()
}

fn exact_first_derivative(x: f64) -> f64 {
    x.cos() - 1.5 * (3.0 * x).sin()
}

fn exact_negative_second_derivative(x: f64) -> f64 {
    x.sin() + 4.5 * (3.0 * x).cos()
}

fn max_error(numerical: &[f64], exact: &[f64]) -> f64 {
    numerical
        .iter()
        .zip(exact.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn main() {
    let n = 32usize;
    let length = 2.0 * PI;

    let grid: Vec<f64> = (0..n)
        .map(|j| length * (j as f64) / (n as f64))
        .collect();

    let values: Vec<f64> = grid.iter().map(|&x| test_function(x)).collect();

    let coeffs = forward_dft(&values);

    let first_derivative_coeffs = spectral_first_derivative(&coeffs);
    let negative_second_derivative_coeffs = spectral_negative_second_derivative(&coeffs);

    let first_derivative_values = inverse_dft(&first_derivative_coeffs);
    let negative_second_derivative_values = inverse_dft(&negative_second_derivative_coeffs);

    let exact_first: Vec<f64> = grid.iter().map(|&x| exact_first_derivative(x)).collect();
    let exact_negative_second: Vec<f64> = grid
        .iter()
        .map(|&x| exact_negative_second_derivative(x))
        .collect();

    let error_first = max_error(&first_derivative_values, &exact_first);
    let error_negative_second =
        max_error(&negative_second_derivative_values, &exact_negative_second);

    println!("Fourier Pseudospectral Differentiation");
    println!("======================================");
    println!();
    println!("Grid points N                  = {}", n);
    println!("Domain                         = [0, 2*pi)");
    println!("Test function                  = sin(x) + 0.5 cos(3x)");
    println!();
    println!("Maximum Errors");
    println!("--------------");
    println!("max |u'_num - u'_exact|        = {:.6e}", error_first);
    println!(
        "max |-u''_num - (-u''_exact)|  = {:.6e}",
        error_negative_second
    );
    println!();
    println!("Selected Nodal Values");
    println!("---------------------");
    println!(
        "{:>5} {:>14} {:>18} {:>18} {:>18}",
        "j", "x_j", "u'(x_j)", "exact u'(x_j)", "abs error"
    );

    for j in 0..8 {
        let abs_err = (first_derivative_values[j] - exact_first[j]).abs();
        println!(
            "{:>5} {:>14.8} {:>18.10} {:>18.10} {:>18.6e}",
            j, grid[j], first_derivative_values[j], exact_first[j], abs_err
        );
    }

    println!();
    println!("Dominant Fourier Coefficients");
    println!("-----------------------------");
    println!("{:>8} {:>8} {:>18} {:>18}", "index", "k", "Re", "|coeff|");

    for k_index in 0..n {
        let k = wave_number(k_index, n);
        let magnitude = coeffs[k_index].abs();

        if magnitude > 1.0e-12 {
            println!(
                "{:>8} {:>8} {:>18.10} {:>18.10}",
                k_index, k, coeffs[k_index].re, magnitude
            );
        }
    }
}
```

Program 20.8.1 demonstrates the essential computational structure of a Fourier pseudospectral method by separating the algorithm into physical-space representation, spectral transformation, coefficient-space differentiation, and inverse reconstruction. This organization reflects the theoretical framework developed in Section 20.8.2, where differentiation operators become diagonal multipliers once the solution is expressed in the Fourier basis. The resulting numerical errors are close to machine precision, illustrating the rapid convergence and exceptional accuracy obtainable for smooth periodic functions.

The example also highlights one of the major conceptual differences between spectral methods and local finite-difference discretizations. In a finite-difference method, derivatives are approximated through local stencil operations involving neighboring nodes. In the Fourier pseudospectral method, by contrast, differentiation is performed globally through the spectral coefficients. The derivative approximation therefore depends on the global representation of the solution rather than only local nodal information. This global structure is responsible for the high-order accuracy and low numerical dispersion associated with spectral discretizations.

Although the implementation uses direct discrete Fourier transforms for clarity, practical large-scale solvers typically replace these routines with FFT-based algorithms to achieve $O(N\log N)$ complexity. The same spectral framework can then be extended to multidimensional turbulence simulation, incompressible flow solvers, wave propagation models, and other periodic PDE systems discussed later in Section 20.8.6. The modular organization of the code also provides a foundation for incorporating dealiasing, implicit spectral operators, matrix-free Krylov solvers, and GPU-accelerated FFT workflows in more advanced implementations.

## 20.8.3. A Model Spectral Boundary Value Problem

Consider the one-dimensional boundary value problem:

\begin{equation}
\begin{aligned}
-u''(x)+c(x)u(x)&=f(x),
\qquad x\in(-1,1),\\
u(-1)&=u(1)=0
\end{aligned}
\tag{20.8.9}
\end{equation}

A spectral Galerkin method chooses basis functions that already satisfy the homogeneous boundary conditions. For example, one may use modified Legendre or Chebyshev polynomial combinations $\psi_n$ such that,

$$\psi_n(-1)=\psi_n(1)=0 \tag{20.8.10}$$

The approximate solution is written as:

$$u_N(x) = \sum_{n=0}^{N-2}\alpha_n\psi_n(x) \tag{20.8.11}$$

Substitution into the weak form gives the linear system,

$$K\alpha=F \tag{20.8.12}$$

where,

\begin{equation}
K_{mn}
=
(\psi_n',\psi_m')
+
(c\psi_n,\psi_m)
\tag{20.8.13}
\end{equation}

and

$$F_m=(f,\psi_m) \tag{20.8.14}$$

The Galerkin formulation has several advantages. Boundary conditions are built into the basis, integration by parts gives a natural weak form, and the resulting matrices often have useful symmetry or conditioning properties. If $c(x)$ is smooth and the basis is chosen carefully, the method can achieve very high accuracy with a small number of coefficients.

A collocation method treats the same problem differently. One stores the nodal values $u(x_j)$ at collocation points and approximates the second derivative using a differentiation matrix $D^{(2)}$. The differential equation is enforced at the interior nodes, while the boundary equations enforce:

$$u(x_0)=0,\qquad u(x_N)=0 \tag{20.8.15}$$

The resulting algebraic system is direct and relatively easy to code. However, spectral collocation matrices can become poorly conditioned as $N$ increases, especially for higher derivatives. This does not negate their accuracy, but it means that the linear algebra must be treated carefully. Recent work on spectral-collocation preconditioners is important precisely because it addresses the gap between high-order approximation accuracy and the conditioning of the resulting algebraic systems (Javeed et al., 2025).

The Galerkin and collocation approaches therefore represent complementary design choices. Galerkin methods are often better conditioned and more natural for analysis. Collocation methods are often simpler to implement, especially for nonlinear problems, because nonlinearities can be evaluated pointwise at the nodes. A robust numerical code may use ideas from both: nodal evaluation for nonlinear terms, coefficient-space transforms for derivatives, and preconditioners designed to control the conditioning of the discrete operator.

### Rust Implementation

Following the discussion in Section 20.8.3 on spectral Galerkin and collocation discretizations for boundary value problems, Program 20.8.2 presents a practical implementation of a Chebyshev collocation solver for the model elliptic problem introduced in equation (20.8.9). The program demonstrates how a differential equation posed on the interval $(-1,1)$ can be converted into a dense algebraic system through spectral differentiation matrices evaluated at Chebyshev-Gauss-Lobatto nodes. Rather than constructing local finite-difference stencils, the method approximates derivatives globally through polynomial interpolation on the collocation grid defined by equation (20.8.5). The second derivative operator is represented through matrix multiplication, while the boundary conditions in equation (20.8.15) are imposed directly at the endpoint rows of the discrete system. This implementation illustrates one of the central ideas of spectral collocation methods: high-order global approximation can produce extremely accurate solutions with relatively few nodal degrees of freedom when the solution and coefficients are sufficiently smooth.

At the core of the implementation is the function `chebyshev_lobatto_nodes`, which generates the Chebyshev-Gauss-Lobatto collocation points introduced in equation (20.8.5). These nodes cluster near the interval endpoints, reducing the interpolation oscillations associated with equally spaced polynomial interpolation and improving numerical stability. The resulting distribution is fundamental in spectral collocation methods because it supports highly accurate polynomial approximation while maintaining stable quadrature and differentiation properties.

The function `chebyshev_differentiation_matrix` constructs the first-order Chebyshev differentiation matrix $D$ associated with the collocation grid. The off-diagonal entries are formed from the analytic differentiation formulas for Chebyshev interpolants, while the diagonal entries are computed so that each row satisfies the consistency condition required for differentiation of constant functions. Once constructed, the matrix provides a discrete approximation of the derivative operator appearing in equation (20.8.6). The second derivative operator is then obtained numerically by the matrix product $D^{(2)} = DD,$ implemented through the helper function `matmul`. This matrix-based differentiation framework is one of the defining features of spectral collocation methods: differentiation becomes a dense linear algebra operation acting on the vector of nodal values.

The functions `coefficient_c`, `exact_solution`, and `forcing_function` define the model problem used for verification. The coefficient function $c(x)$ corresponds to the reaction term in equation (20.8.9), while the exact solution $u(x)=(1-x^2)e^x$, is chosen so that the forcing term $f(x)$ can be constructed analytically. This approach allows the numerical solution produced by the spectral collocation method to be compared directly against an exact closed-form solution. The function `forcing_function` evaluates the right-hand side by substituting the analytic solution into the differential operator, thereby generating a self-consistent manufactured solution test problem. Such manufactured solutions are widely used in scientific computing to verify the correctness and convergence behavior of PDE solvers.

The function `solve_linear_system` performs the numerical solution of the dense algebraic system resulting from the collocation discretization. The implementation uses Gaussian elimination with partial pivoting to improve numerical stability during elimination. The resulting matrix system corresponds to the discrete version of equation (20.8.12), where the second-derivative operator and reaction term have been assembled into a collocation operator acting on the nodal unknowns. Although dense direct solvers are computationally expensive for very large systems, they are appropriate for moderate spectral discretizations and clearly illustrate the algebraic structure of the collocation formulation.

The helper function `max_error` computes the maximum nodal difference between the numerical and exact solutions. This provides a direct measure of the approximation accuracy obtained from the spectral discretization and demonstrates the rapid convergence associated with smooth problems. Since the chosen exact solution is analytic, the spectral approximation converges extremely rapidly, producing errors close to machine precision even with a relatively modest number of collocation points.

The `main` function demonstrates the complete workflow of a Chebyshev spectral collocation solver. It begins by constructing the Chebyshev-Gauss-Lobatto nodes and the associated differentiation matrix. The second derivative matrix is then formed through matrix multiplication, after which the discrete collocation system is assembled. Interior rows enforce the differential equation from equation (20.8.9), while the first and last rows impose the boundary conditions in equation (20.8.15). Once the algebraic system has been assembled, the program solves for the nodal values of the approximate solution and compares them against the exact analytic solution. The printed output reports the maximum nodal error together with representative nodal values across the interval, illustrating the high-order accuracy characteristic of spectral collocation methods.

```rust
// Program 20.8.2: Chebyshev Collocation Solver for a Spectral Boundary Value Problem
//
// Problem statement:
// Solve the model boundary value problem
//
//     -u''(x) + c(x)u(x) = f(x),    x in (-1, 1),
//      u(-1) = u(1) = 0,
//
// using Chebyshev-Gauss-Lobatto collocation points. The program constructs
// the Chebyshev differentiation matrix D, forms D^(2) = D D, enforces the
// differential equation at the interior nodes, and imposes the two boundary
// conditions directly at the endpoint rows.

use std::f64::consts::PI;

fn chebyshev_lobatto_nodes(n: usize) -> Vec<f64> {
    (0..=n)
        .map(|j| (PI * (j as f64) / (n as f64)).cos())
        .collect()
}

fn chebyshev_differentiation_matrix(x: &[f64]) -> Vec<Vec<f64>> {
    let n = x.len() - 1;
    let mut d = vec![vec![0.0; n + 1]; n + 1];

    let mut c = vec![1.0; n + 1];
    c[0] = 2.0;
    c[n] = 2.0;

    for i in 0..=n {
        for j in 0..=n {
            if i != j {
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                d[i][j] = sign * c[i] / (c[j] * (x[i] - x[j]));
            }
        }
    }

    for i in 0..=n {
        let mut row_sum = 0.0;
        for j in 0..=n {
            if i != j {
                row_sum += d[i][j];
            }
        }
        d[i][i] = -row_sum;
    }

    d
}

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];

    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    c
}

fn coefficient_c(x: f64) -> f64 {
    1.0 + x * x
}

fn exact_solution(x: f64) -> f64 {
    (1.0 - x * x) * x.exp()
}

fn forcing_function(x: f64) -> f64 {
    let u = exact_solution(x);

    // For u(x) = (1 - x^2)e^x,
    // u''(x) = -(x^2 + 4x + 1)e^x.
    let u_second = -(x * x + 4.0 * x + 1.0) * x.exp();

    -u_second + coefficient_c(x) * u
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_value = a[k][k].abs();

        for i in (k + 1)..n {
            if a[i][k].abs() > pivot_value {
                pivot_value = a[i][k].abs();
                pivot_row = i;
            }
        }

        if pivot_value < 1.0e-14 {
            panic!("Matrix is singular or nearly singular.");
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

    x
}

fn max_error(numerical: &[f64], exact: &[f64]) -> f64 {
    numerical
        .iter()
        .zip(exact.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn main() {
    let n = 24usize;

    let x = chebyshev_lobatto_nodes(n);
    let d = chebyshev_differentiation_matrix(&x);
    let d2 = matmul(&d, &d);

    let size = n + 1;
    let mut a = vec![vec![0.0; size]; size];
    let mut rhs = vec![0.0; size];

    for i in 0..=n {
        if i == 0 || i == n {
            // Boundary rows: u(x_0) = 0 and u(x_N) = 0.
            a[i][i] = 1.0;
            rhs[i] = 0.0;
        } else {
            // Interior rows: -u''(x_i) + c(x_i)u(x_i) = f(x_i).
            for j in 0..=n {
                a[i][j] = -d2[i][j];
            }

            a[i][i] += coefficient_c(x[i]);
            rhs[i] = forcing_function(x[i]);
        }
    }

    let numerical = solve_linear_system(a, rhs);
    let exact: Vec<f64> = x.iter().map(|&xi| exact_solution(xi)).collect();

    let error = max_error(&numerical, &exact);

    println!("Chebyshev Collocation Boundary Value Solver");
    println!("===========================================");
    println!();
    println!("Problem");
    println!("-------");
    println!("Solve -u''(x) + c(x)u(x) = f(x),  x in (-1,1)");
    println!("with u(-1) = u(1) = 0");
    println!();
    println!("Collocation Data");
    println!("----------------");
    println!("Polynomial degree N          = {}", n);
    println!("Number of nodes              = {}", size);
    println!("Coefficient c(x)             = 1 + x^2");
    println!("Exact solution               = (1 - x^2) exp(x)");
    println!();
    println!("Accuracy");
    println!("--------");
    println!("Maximum nodal error          = {:.6e}", error);
    println!();
    println!("Selected Nodal Values");
    println!("---------------------");
    println!(
        "{:>5} {:>14} {:>18} {:>18} {:>18}",
        "j", "x_j", "u_num(x_j)", "u_exact(x_j)", "abs error"
    );

    for j in 0..=n {
        if j % 3 == 0 || j == n {
            let abs_error = (numerical[j] - exact[j]).abs();

            println!(
                "{:>5} {:>14.8} {:>18.10} {:>18.10} {:>18.6e}",
                j, x[j], numerical[j], exact[j], abs_error
            );
        }
    }
}
```

Program 20.8.2 demonstrates how a one-dimensional elliptic boundary value problem can be transformed into a dense spectral collocation system through the use of Chebyshev differentiation matrices. This approach reflects the theoretical framework developed in Section 20.8.3, where derivatives are approximated globally through polynomial interpolation at carefully selected collocation points. The resulting numerical solution achieves near machine-precision accuracy, illustrating the effectiveness of spectral approximation for smooth problems.

The implementation also highlights one of the major distinctions between spectral collocation methods and local finite-difference discretizations. In finite-difference methods, derivatives are approximated through compact local stencils whose accuracy depends on nearby neighboring points. In the spectral collocation formulation, by contrast, each differentiation matrix row depends on all nodal values simultaneously, reflecting the global polynomial approximation underlying the method. This global coupling produces much higher accuracy for smooth solutions, although it also leads to dense operators and potential conditioning challenges as the polynomial degree increases.

The program further illustrates the complementary relationship between Galerkin and collocation formulations discussed in Section 20.8.3. The collocation method is straightforward to implement because the unknowns correspond directly to nodal values and the boundary conditions are imposed explicitly through matrix rows. At the same time, the dense differentiation operators reveal why conditioning and preconditioning become increasingly important in large-scale spectral solvers. In practical high-performance PDE codes, these collocation operators are often combined with matrix-free iterative methods, tensor-product decompositions, spectral-element localization, and multilevel preconditioners to retain spectral accuracy while improving computational scalability.

## 20.8.4. Fast Algorithms and Computational Complexity

The computational cost of a spectral method depends strongly on the basis, representation, geometry, and operator. If dense differentiation matrices are applied directly in one dimension, each derivative application costs $O(N^2)$. This cost may be acceptable for moderate $N$, but it becomes limiting for high-resolution multidimensional simulations. The main route to fast spectral algorithms is to exploit transforms and tensor-product structure.

For Fourier pseudospectral methods, the fast Fourier transform reduces the cost of transforming between physical and coefficient space. For $N$ total unknowns, multidimensional FFT-based differentiation typically costs $O(N\log N)$. This is why Fourier pseudospectral solvers remain highly competitive for periodic domains and direct numerical simulation of turbulence. Derivatives are computed accurately in coefficient space, while nonlinear terms are evaluated in physical space. The transform cost is higher than a low-order stencil, but the accuracy per degree of freedom can be substantially better for smooth solutions.

On Cartesian tensor-product domains, additional structure can be exploited. Spectral-element and spectral-Galerkin discretizations may use tensor contractions, separable bases, and fast transforms to reduce computational cost. Modern spectral-element Poisson solvers on rectangular domains use this structure to build very fast direct solvers. In $d$ dimensions, separability can lead to direct Poisson solution costs of the form

$$O\!\left(N^{(d+1)/d}\right) \tag{20.8.16}$$

while preserving high-order accuracy. Recent GPU implementations show that this is not only a theoretical advantage, but also a practical high-performance strategy for three-dimensional Poisson-type problems (Liu, Shen and Zhang, 2024; Roccon, 2024).

The comparison with finite differences and low-order finite elements must therefore be made carefully. A low-order method may have cheaper local operations, but it may require many more degrees of freedom to achieve the same accuracy for a smooth solution. A spectral method may have denser or more global operations, but may reach the desired accuracy at much smaller $N$. The best method depends on the regularity of the solution, the geometry of the domain, the coefficient structure, the boundary conditions, and the target accuracy.

Conditioning is a central computational issue. Classical spectral collocation can produce very accurate approximations, but the corresponding derivative matrices may be poorly conditioned. High-order differentiation amplifies this problem. Recent integration-based preconditioning for spectral collocation addresses this issue by approximating the inverse differentiation process and improving the behaviour of the linear system. This is an important lesson for numerical computing: high-order approximation and efficient linear solution are separate problems. A method may approximate the PDE very accurately and still require a carefully designed preconditioner to solve the resulting algebraic equations efficiently (Javeed et al., 2025).

## 20.8.5. Modern Extensions Beyond Classical Rectangular Domains

Classical spectral methods are often associated with periodic boxes, intervals, rectangles, and simple tensor-product geometries. Modern work has significantly broadened this scope. Spectral finite elements and discontinuous Galerkin spectral-element methods reintroduce locality while preserving high-order approximation. In these methods, the domain is divided into elements, and high-order polynomial approximations are used within each element. This makes it possible to treat complex geometries and local refinement more naturally than with a purely global polynomial expansion.

Entropy-stable discontinuous Galerkin spectral-element methods extend spectral ideas to nonlinear systems on curvilinear meshes. Such methods combine high-order polynomial approximation with numerical fluxes, metric terms, entropy stability, and conservation. This is important for hyperbolic and balance-law systems where high-order accuracy must be combined with nonlinear stability and physically admissible states (Ersing and Winters, 2024).

Embedded-boundary and harmonic-map approaches provide another route to nonrectangular domains. Rather than abandoning spectral approximation when the geometry becomes irregular, one may transform the domain or modify the operator so that spectral structure remains usable. Recent harmonic-map spectral methods transport general two-dimensional domains into settings compatible with efficient Fourier-Legendre spectral-Galerkin approximation. Embedded-boundary spectral methods incorporate boundary conditions directly into modified operators in Hilbert-space formulations (Guimarães and Piqueira, 2026; Shi et al., 2026).

Spectral methods have also been extended to data-defined and manifold settings. In point-cloud spectral methods, the basis functions may be constructed from approximate Laplace-Beltrami eigenfunctions inferred from data. This allows elliptic PDEs to be solved on unknown manifolds represented only by sampled points. Conceptually, this weakens the traditional limitation that spectral methods require simple analytic geometries. Instead, the spectral basis is adapted to the geometry through data-derived operators (Yan, Jiang and Harlim, 2023).

Another modern direction is the connection between spectral methods and scientific machine learning. Spectral operator learning uses orthogonal-function expansions as compact representations for parametric PDEs and operator maps. The purpose is not to replace spectral approximation, but to combine its compact coefficient representation with trainable mappings. This reflects a broader trend in which spectral coefficients are used not only for deterministic approximation, but also as compressed coordinates for reduced-order inference and data-driven operator approximation (Choi et al., 2024).

## 20.8.6. Practical Applications and Implementation Considerations

A major application of Fourier spectral methods is direct numerical simulation of incompressible flow and multiphase turbulence on periodic domains. A representative incompressible model is:

\begin{equation}
\partial_t\mathbf{u}
+
(\mathbf{u}\cdot\nabla)\mathbf{u}
=
-\nabla p
+
\nu\Delta\mathbf{u}
+
\mathbf{f},
\qquad
\nabla\cdot\mathbf{u}=0
\tag{20.8.17}
\end{equation}

In periodic geometry, Fourier pseudospectral differentiation is attractive because derivatives are represented exactly on the retained modes, numerical dispersion is low, and FFT-based implementation is natural. When coupled with phase-field equations, immersed-boundary forcing, or other interface descriptions, this framework supports high-accuracy simulations of drops, bubbles, fluid-structure interaction, and turbulence. Recent GPU-ready pseudospectral solvers for multiphase turbulence and immersed-boundary/Fourier pseudospectral methods for fluid-structure interaction show that classical spectral discretization remains central to state-of-the-art flow simulation on heterogeneous hardware (Roccon, 2024; Nascimento et al., 2024).

A second important use case is high-order Cartesian Poisson and elliptic solution. In pressure projection for incompressible Navier-Stokes equations, electrostatics, diffusion-dominated subproblems, and implicit time-stepping, Poisson-type equations must often be solved repeatedly. When the computational domain has tensor-product structure, spectral-element or spectral-Galerkin discretizations can exploit fast transforms and tensor contractions to produce highly efficient solvers. The recent GPU implementation of spectral-element methods for three-dimensional Poisson-type equations is pedagogically important because it turns the mathematical separability of the basis into a practical high-performance workflow (Liu, Shen and Zhang, 2024).

From a Rust implementation perspective, spectral methods suggest two broad software patterns. Fourier pseudospectral methods benefit from explicit separation between physical-space arrays and coefficient-space arrays. The code should make it clear when a field is represented by nodal values and when it is represented by Fourier coefficients. FFT workspaces, dealiasing rules, differentiation multipliers, and inverse transforms should be managed explicitly so that the mathematical representation is not obscured.

Polynomial spectral and spectral-element methods require a somewhat different structure. They benefit from precomputed basis values, quadrature weights, differentiation matrices, mass and stiffness operators, and tensor-product kernels. For Galerkin methods, one should separate basis construction, quadrature, assembly or matrix-free application, and boundary-condition enforcement. For collocation methods, one should separate node generation, differentiation matrices, physical coefficient evaluation, and preconditioning. In both cases, a clean distinction between approximation, operator application, boundary handling, and linear solution leads to safer and more extensible code.

The key takeaway is that spectral methods offer exceptional accuracy when the problem is smooth enough to justify them. Their strength is not simply that they are “high order,” but that the basis functions are chosen to represent the global or elementwise structure of the solution efficiently. Their weakness is that geometry, nonsmoothness, conditioning, and nonlinear stability can reduce or complicate this advantage. Modern spectral methods address these limitations by hybridization: spectral elements add locality, DG spectral elements add conservation and nonlinear stability, preconditioners address conditioning, mapped-domain methods handle geometry, and operator-learning variants use spectral coefficients as compact representations. Spectral methods are therefore not a replacement for finite differences, finite volumes, finite elements, or multigrid. They are a complementary high-accuracy framework whose effectiveness depends on matching the approximation space to the analytic and geometric structure of the PDE.

# 20.9. Conclusion

Throughout this chapter, we have developed the numerical foundations required for solving partial differential equations, one of the most important classes of mathematical models in scientific computing. We examined methods for hyperbolic, diffusive, and multidimensional initial value problems, together with numerical techniques for elliptic boundary value problems. The chapter introduced finite-volume formulations, method-of-lines discretizations, Fourier and cyclic-reduction solvers, relaxation methods, multigrid algorithms, and spectral techniques. These approaches represent a broad spectrum of numerical strategies, ranging from locally conservative low-order methods to highly accurate global approximations. By combining mathematical analysis with practical Rust implementations, this chapter has provided the tools needed to construct efficient and reliable PDE solvers for a wide range of scientific and engineering applications.

## 20.9.1. Key Takeaways

- Partial differential equations describe the evolution and equilibrium of physical systems involving transport, diffusion, waves, and field interactions. Their numerical solution requires careful consideration of both mathematical structure and computational efficiency.
- The method of lines provides a unifying framework by separating spatial discretization from time integration, transforming PDEs into systems of ordinary differential equations that can be solved using established numerical methods.
- Finite-volume methods preserve conservation laws at the discrete level and are particularly effective for hyperbolic problems involving shocks, discontinuities, and transport phenomena.
- Diffusive problems impose different numerical challenges, especially stability restrictions for explicit methods. Implicit schemes such as Backward Euler and Crank–Nicolson provide enhanced stability and are widely used in practice.
- Multidimensional PDEs introduce additional complexity through directional coupling and increased computational cost. Splitting methods and alternating-direction techniques provide practical strategies for reducing this complexity.
- Fourier-based methods and cyclic reduction exploit problem structure to achieve highly efficient solutions of certain classes of boundary value problems, particularly those defined on structured grids.
- Relaxation methods such as Jacobi, Gauss–Seidel, and Successive Over-Relaxation reduce errors iteratively and provide the foundation for more advanced multigrid algorithms.
- Multigrid methods address errors across multiple spatial scales and can achieve near-optimal computational complexity, making them among the most powerful solvers for large elliptic systems.
- Spectral methods achieve extremely high accuracy for smooth solutions by employing global approximation spaces, often requiring far fewer degrees of freedom than low-order discretizations.
- Rust provides an excellent platform for PDE solvers through its combination of performance, memory safety, concurrency support, and modern numerical computing libraries.

## 20.9.2. Advice for Beginners

- When beginning the study of numerical PDEs, focus first on understanding the physical meaning of hyperbolic, parabolic, and elliptic equations. Many numerical methods are designed specifically to preserve the mathematical and physical properties associated with these different classes of problems.
- Start with one-dimensional model equations such as the linear advection equation and the diffusion equation. These examples provide valuable insight into stability, consistency, convergence, and error propagation without introducing excessive complexity.
- Implement simple finite-difference or finite-volume schemes before moving to high-resolution methods. Understanding why basic schemes succeed or fail is essential for appreciating more advanced algorithms.
- Pay close attention to stability conditions, particularly CFL restrictions for explicit methods. Many numerical difficulties originate from violating these conditions rather than from errors in implementation.
- After mastering basic iterative solvers, explore multigrid methods and observe how they accelerate convergence by addressing errors at multiple scales. This provides important intuition about modern large-scale PDE solvers.
- Experiment with spectral methods and compare their accuracy with low-order discretizations. Such comparisons illustrate the trade-offs between local and global approximation strategies.
- For Rust implementations, libraries such as `ndarray`, `nalgebra`, `sprs`, and `rayon` can significantly simplify numerical development while maintaining performance and reliability.
- Most importantly, remember that successful PDE simulation requires a balance between mathematical understanding, algorithm selection, and careful implementation.

## 20.9.3. Further Learning with GenAI

To deepen your understanding of numerical methods for PDEs, consider exploring the following prompts:

 1. Explain the differences between hyperbolic, parabolic, and elliptic PDEs and discuss how these differences influence numerical method selection.
 2. Implement a method-of-lines solver in Rust for the advection equation and analyze the effects of different time-integration methods.
 3. Compare finite-difference, finite-volume, and finite-element viewpoints for solving conservation laws.
 4. Explain the CFL condition and visualize how violating it affects the stability of explicit numerical schemes.
 5. Implement a Crank–Nicolson solver for a diffusion equation and compare its performance with explicit and Backward Euler methods.
 6. Derive and implement a multigrid V-cycle in Rust and analyze its computational complexity.
 7. Compare Jacobi, Gauss–Seidel, Successive Over-Relaxation, and multigrid methods for solving the same elliptic boundary value problem.
 8. Implement a Fourier-based Poisson solver and compare its performance with iterative methods.
 9. Develop a spectral collocation solver for a smooth boundary value problem and compare its convergence with finite-difference methods.
10. Analyze the trade-offs between finite-volume, multigrid, and spectral methods for large-scale scientific computing applications.

By exploring these prompts, readers can gain deeper insight into both the theoretical and practical aspects of PDE computation.

## 20.9.4. Homework Exercises

To reinforce your understanding of the material covered in this chapter, complete the following exercises:

 1. Derive the method-of-lines formulation for a one-dimensional advection equation and analyze the resulting system of ordinary differential equations.
 2. Implement a conservative finite-volume solver for the linear advection equation and verify discrete conservation properties.
 3. Compare explicit and implicit methods for the diffusion equation and investigate their stability and accuracy as the time step varies.
 4. Extend a one-dimensional diffusion solver to two spatial dimensions and study the computational impact of the increased problem size.
 5. Implement an alternating-direction implicit method for a multidimensional diffusion problem and compare it with a fully implicit approach.
 6. Develop a Fourier-based Poisson solver and compare its computational cost with an iterative relaxation method.
 7. Implement Jacobi, Gauss–Seidel, and Successive Over-Relaxation methods and compare their convergence rates for the same boundary value problem.
 8. Construct a multigrid solver and investigate how the convergence rate changes as the grid resolution increases.
 9. Implement a spectral collocation method for a smooth boundary value problem and compare its accuracy with finite-difference discretizations.
10. Select a real-world PDE application such as heat conduction, fluid transport, wave propagation, or electrostatics. Formulate the governing equations, implement an appropriate numerical method, and evaluate its accuracy and computational performance.

Partial differential equations form the mathematical foundation of many scientific and engineering models. The numerical methods presented in this chapter provide the tools needed to simulate complex physical systems, analyze their behavior, and solve large-scale computational problems. As you continue your study of numerical computing, these techniques will serve as a cornerstone for advanced topics in computational fluid dynamics, electromagnetics, climate modeling, materials science, and many other disciplines. By combining mathematical insight with efficient Rust implementations, you will be well prepared to tackle increasingly sophisticated PDE-based applications.

# References

 1. Abgrall, R. and Liu, Y. (2024) ‘A New Approach for Designing Well-Balanced Schemes for the Shallow Water Equations: A Combination of Conservative and Primitive Formulations’, *SIAM Journal on Scientific Computing*, 46(6), pp. A3375–A3400. doi: 10.1137/23M1624610.
 2. Akkurt, S., Lemaire, S., Bartholomew, P. and Laizet, S. (2025) ‘A distributed-memory tridiagonal solver based on a specialised data structure optimised for CPU and GPU architectures’, *Computer Physics Communications*, 315, 109747. doi: 10.1016/j.cpc.2025.109747.
 3. Albasatin, R., Sunarsih and Hariyanto, S. (2025) ‘Stability analysis of the two-dimensional advection-diffusion equation for particle distribution by the Crank Nicolson-alternating direction implicit method’, *Communications in Mathematical Biology and Neuroscience*, 2025, Article 41. doi: 10.28919/cmbn/9120.
 4. Barsukow, W., Ciallella, M., Ricchiuto, M. and Torlo, D. (2026) ‘Genuinely multi-dimensional stationarity preserving Finite Volume formulation for nonlinear hyperbolic PDEs’, *Journal of Computational Physics*, 550, 114633. doi: 10.1016/j.jcp.2025.114633.
 5. Bourgeois, R., Tremblin, P., Kokh, S. and Padioleau, T. (2024) ‘Recasting an operator splitting solver into a standard finite volume flux-based algorithm. The case of a Lagrange-projection-type method for gas dynamics’, *Journal of Computational Physics*, 496, 112594. doi: 10.1016/j.jcp.2023.112594.
 6. Caldana, M., Antonietti, P.F. and Dede’, L. (2024) ‘A deep learning algorithm to accelerate algebraic multigrid methods in finite element solvers of 3D elliptic PDEs’, *Computers & Mathematics with Applications*, 167, pp. 217–231. doi: 10.1016/j.camwa.2024.05.013.
 7. Caliari, M. and Cassini, F. (2024) ‘A second order directional split exponential integrator for systems of advection–diffusion–reaction equations’, *Journal of Computational Physics*, 498, 112640. doi: 10.1016/j.jcp.2023.112640.
 8. Caliari, M., Cassini, F., Einkemmer, L. and Ostermann, A. (2024) ‘Accelerating Exponential Integrators to Efficiently Solve Semilinear Advection-Diffusion-Reaction Equations’, *SIAM Journal on Scientific Computing*, 46(2), pp. A906–A928. doi: 10.1137/23M1562056.
 9. Centofanti, E. and Scacchi, S. (2024) ‘A comparison of Algebraic Multigrid Bidomain solvers on hybrid CPU–GPU architectures’, *Computer Methods in Applied Mechanics and Engineering*, 423, 116875. doi: 10.1016/j.cma.2024.116875.
10. Centofanti, E., Huynh, N.M.M., Pavarino, L.F. and Scacchi, S. (2025) ‘Parallel Algebraic Multigrid Solvers for Composite Discontinuous Galerkin Discretization of the Cardiac EMI Model in Heterogeneous Media’, *Computer Methods in Applied Mechanics and Engineering*, 442, 118001. doi: 10.1016/j.cma.2025.118001.
11. Charles, P. and Ray, D. (2025) ‘Learning WENO for Entropy Stable Schemes to Solve Conservation Laws’, *SIAM Journal on Scientific Computing*, 47(6), pp. C1196–C1222. doi: 10.1137/24M1697566.
12. Chen, H., Zhang, W., Zhang, C., Sun, B., Yang, S. and Chen, D. (2024) ‘Diffusion-Equation-Based Electrical Modeling for High-Power Lithium Titanium Oxide Batteries’, *Batteries*, 10(7), 238. doi: 10.3390/batteries10070238.
13. Choi, J., Yun, T., Kim, N. and Hong, Y. (2024) ‘Spectral operator learning for parametric PDEs without data reliance’, *Computer Methods in Applied Mechanics and Engineering*, 420, 116678. doi: 10.1016/j.cma.2023.116678.
14. Choi, Y., Hwang, Y., Kwak, S., Ham, S., Jyoti, Kim, H. and Kim, J. (2025) ‘A cell structure implementation of the multigrid method for the two-dimensional diffusion equation’, *AIP Advances*, 15(1), 015019. doi: 10.1063/5.0247042.
15. Del Grosso, A., Castro, M.J., Chan, A., Gallice, G., Loubère, R. and Maire, P.-H. (2024) ‘A well-balanced, positive, entropy-stable, and multi-dimensional-aware finite volume scheme for 2D shallow-water equations with unstructured grids’, *Journal of Computational Physics*, 503, 112829. doi: 10.1016/j.jcp.2024.112829.
16. Diez Sanhueza, R., Peeters, J.W.R. and Costa, P. (2025) ‘A pencil-distributed finite-difference solver for extreme-scale calculations of turbulent wall flows at high Reynolds number’, *Computer Physics Communications*, 316, 109811. doi: 10.1016/j.cpc.2025.109811.
17. Ersing, P., Goldberg, S. and Winters, A.R. (2025) ‘Entropy stable hydrostatic reconstruction schemes for shallow water systems’, *Journal of Computational Physics*, 527, 113802. doi: 10.1016/j.jcp.2025.113802.
18. Ersing, P. and Winters, A.R. (2024) ‘An Entropy Stable Discontinuous Galerkin Method for the Two-Layer Shallow Water Equations on Curvilinear Meshes’, *Journal of Scientific Computing*, 98, 62. doi: 10.1007/s10915-024-02451-2.
19. Ferreira, P. and Tang, S.-X. (2025) ‘PDE Modeling of the Cooling Fluid Temperature in Battery Pack’, *IFAC-PapersOnLine*, 59(8), pp. 84–89. doi: 10.1016/j.ifacol.2025.08.071.
20. Guimarães, O. and Piqueira, J.R.C. (2026) ‘Embedded boundary conditions in spectral methods: A rectangular matrix approach’, *Mathematics and Computers in Simulation*, 241, pp. 225–237. doi: 10.1016/j.matcom.2025.10.001.
21. Guo, Y., Yin, Q. and Zhang, Z. (2024) ‘A Structure-preserving Implicit Exponential Time Differencing Scheme for Maxwell–Ampère Nernst–Planck Model’, *Journal of Scientific Computing*, 101, 51. doi: 10.1007/s10915-024-02669-0.
22. Hafeez, M.B. and Krawczuk, M. (2023) ‘A Review: Applications of the Spectral Finite Element Method’, *Archives of Computational Methods in Engineering*, 30, pp. 3453–3465. doi: 10.1007/s11831-023-09911-2.
23. Hasegawa, G., Kuwata, N., Ohnishi, T. and Takada, K. (2024) ‘Visualization and evaluation of lithium diffusion at grain boundaries in Li({}*{0.29})La({}*{0.57})TiO({}\_3) solid electrolytes using secondary ion mass spectrometry’, *Journal of Materials Chemistry A*, 12, pp. 731–738. doi: 10.1039/D3TA05012B.
24. Javeed, A., Kouri, D.P., Ridzal, D. and Steinman, J.D. (2025) ‘A Preconditioner for Spectral Collocation’, *SIAM Journal on Scientific Computing*, 47(5), pp. A2828–A2850. doi: 10.1137/24M1712539.
25. Kalungi, P. and Menart, J. (2025) ‘Electrochemical–Thermal Model of a Lithium-Ion Battery’, *Energies*, 18(7), 1764. doi: 10.3390/en18071764.
26. Lian, J., Yao, Q.H. and Jiang, Z. (2025) ‘An Overlapping IBM-PISO Algorithm with an FFT-Based Poisson Solver for Parallel Incompressible Flow Simulations’, *Fluids*, 10(7), 176. doi: 10.3390/fluids10070176.
27. Ling, D. and Tang, H. (2023) ‘Genuinely multidimensional physical-constraints-preserving finite volume schemes for the special relativistic hydrodynamics’, *Communications in Computational Physics*, 34(4), pp. 955–992. doi: 10.4208/cicp.OA-2023-0065.
28. Liu, X., Shen, J. and Zhang, X. (2024) ‘A Simple GPU Implementation of Spectral-Element Methods for Solving 3D Poisson Type Equations on Rectangular Domains and Its Applications’, *Communications in Computational Physics*, 36(5), pp. 1157–1185. doi: 10.4208/cicp.OA-2024-0072.
29. Liu, Y., Guo, W., Jiang, Y. and Zhang, M. (2025) ‘Non-oscillatory entropy stable DG schemes for hyperbolic conservation law’, *Journal of Computational Physics*, 531, 113926. doi: 10.1016/j.jcp.2025.113926.
30. Liu, Y., Lu, J. and Shu, C.-W. (2024) ‘An Entropy Stable Essentially Oscillation-Free Discontinuous Galerkin Method for Hyperbolic Conservation Laws’, *SIAM Journal on Scientific Computing*, 46(2), pp. A1132–A1159. doi: 10.1137/22M1524151.
31. Mama, M., Solai, E., Capurso, T., Danlos, A. and Khelladi, S. (2025) ‘Comprehensive review of multi-scale Lithium-ion batteries modeling: From electro-chemical dynamics up to heat transfer in battery thermal management system’, *Energy Conversion and Management*, 325, 119223. doi: 10.1016/j.enconman.2024.119223.
32. Moussa, R. and Kahl, K. (2026) ‘A Theory of Relaxation-Based Algebraic Multigrid’, *arXiv preprint* arXiv:2603.26513. doi: 10.48550/arXiv.2603.26513.
33. Nascimento, A.A., Mariano, F.P., da Silveira Neto, A. and Martínez Padilla, E.L. (2024) ‘Coupling of the immersed boundary and Fourier pseudo-spectral methods applied to solve fluid–structure interaction problems’, *Journal of the Brazilian Society of Mechanical Sciences and Engineering*, 46, 213. doi: 10.1007/s40430-024-04780-7.
34. Ohm, P., Harper, G. and Jansson, N. (2026) ‘A Matrix-Free Algebraic hp-Multigrid Method for Computational Fluid Dynamics Applications’, *Proceedings of the Supercomputing Asia and International Conference on High Performance Computing in Asia Pacific Region*, pp. 194–202. doi: 10.1145/3773656.3773686.
35. Owolabi, K.M. and Alagoz, S. (2025) ‘Advection–diffusion–reaction modeling of contaminant transport in groundwater: Analysis and simulation’, *Nonlinear Science*, 5, 100083. doi: 10.1016/j.nls.2025.100083.
36. Pei, J. and Tong, X. (2025) ‘A Hybrid DST-Accelerated Finite-Difference Solver for 2D and 3D Poisson Equations with Dirichlet Boundary Conditions’, *Mathematics*, 13(17), 2776. doi: 10.3390/math13172776.
37. Qin, O. (2025) ‘An Optimal Complexity Spectral Solver for the Poisson Equation’, *Journal of Scientific Computing*, 104, 102. doi: 10.1007/s10915-025-03011-y.
38. Richardson, C.N., Baratta, I.A., Dean, J.P., Jackson, W.A. and Wells, G.N. (2025) ‘An efficient multigrid solver for finite element methods on multi-GPU systems’, *Procedia Computer Science*, 267, pp. 82–91. doi: 10.1016/j.procs.2025.08.235.
39. Roccon, A. (2024) ‘A GPU-ready pseudo-spectral method for direct numerical simulations of multiphase turbulence’, *Procedia Computer Science*, 240, pp. 17–30. doi: 10.1016/j.procs.2024.07.005.
40. Shen, Y., Zhu, Z., Zhou, Q. and Jiang, C. (2024) ‘An improved dynamic bidirectional coupled hydrologic–hydrodynamic model for efficient flood inundation prediction’, *Natural Hazards and Earth System Sciences*, 24, pp. 2315–2330. doi: 10.5194/nhess-24-2315-2024.
41. Shi, S., Jiang, X., Zeng, F. and Zhang, H. (2026) ‘A Spectral Method with Harmonic Map for Elliptic PDEs on General Two-Dimensional Domains’, *Journal of Scientific Computing*, 107, 55. doi: 10.1007/s10915-026-03241-8.
42. Thompson, J.L., Brown, J. and He, Y. (2023) ‘Local Fourier Analysis of p-Multigrid for High-Order Finite Element Operators’, *SIAM Journal on Scientific Computing*, 45, pp. S351–S370. doi: 10.1137/21M1431199.
43. Tomida, K. and Stone, J.M. (2023) ‘The Athena++ Adaptive Mesh Refinement Framework: Multigrid Solvers for Self-gravity’, *Astrophysical Journal Supplement Series*, 266(1), Article 7. doi: 10.3847/1538-4365/acc2c0.
44. Tremblin, P., Bourgeois, R., Bulteau, S., Kokh, S., Padioleau, T., Delorme, M., Strugarek, A., González, M. and Brun, A.S. (2024) ‘A multi-dimensional, robust, and cell-centered finite-volume scheme for the ideal MHD equations’, *Journal of Computational Physics*, 519, 113455. doi: 10.1016/j.jcp.2024.113455.
45. Trias, F.X., Alsalti-Baldellou, À. and Oliva, A. (2026) ‘On the Reynolds-number scaling of Poisson solver complexity’, *Physics of Fluids*, 38(4), 045157. doi: 10.1063/5.0319857.
46. Tsai, Y.-H.M., Beams, N. and Anzt, H. (2023) ‘Three-precision algebraic multigrid on GPUs’, *Future Generation Computer Systems*, 149, pp. 280–293. doi: 10.1016/j.future.2023.07.024.
47. Vijaywargiya, A. and Fu, G. (2024) ‘Two Finite Element Approaches for the Porous Medium Equation That Are Positivity Preserving and Energy Stable’, *Journal of Scientific Computing*, 100, 86. doi: 10.1007/s10915-024-02642-x.
48. Wang, F., Gu, X., Sun, J. and Xu, Z. (2023) ‘Learning-based local weighted least squares for algebraic multigrid method’, *Journal of Computational Physics*, 493, 112437. doi: 10.1016/j.jcp.2023.112437.
49. Wichrowski, M., Munch, P., Kronbichler, M. and Kanschat, G. (2025) ‘Smoothers with Localized Residual Computations for Geometric Multigrid Methods for Higher-Order Finite Elements’, *SIAM Journal on Scientific Computing*, 47(3), pp. B645–B664. doi: 10.1137/23M1625962.
50. Wu, C.-H. (2025) ‘Enhanced Efficient 3D Poisson Solver Supporting Dirichlet, Neumann, and Periodic Boundary Conditions’, *Computation*, 13(4), 99. doi: 10.3390/computation13040099.
51. Yan, Q., Jiang, S.W. and Harlim, J. (2023) ‘Spectral methods for solving elliptic PDEs on unknown manifolds’, *Journal of Computational Physics*, 486, 112132. doi: 10.1016/j.jcp.2023.112132.
52. Yang, L. and Yang, J. (2025) ‘A multi-colored Gauss-Seidel solver for aerodynamic simulations of a transport aircraft model on graphics processing units’, *Advances in Aerodynamics*, 7, 8. doi: 10.1186/s42774-024-00200-5.
53. Zhang, Z., Zhou, X., Li, G., Qian, S. and Niu, Q. (2023) ‘A New Entropy Stable Finite Difference Scheme for Hyperbolic Systems of Conservation Laws’, *Mathematics*, 11(12), 2604. doi: 10.3390/math11122604.
54. Zhang, Z., Yang, X. and Wang, S. (2025) ‘The alternating direction implicit difference scheme and extrapolation method for a class of three dimensional hyperbolic equations with constant coefficients’, *Electronic Research Archive*, 33(5), pp. 3348–3377. doi: 10.3934/era.2025148.
