---
weight: 3000
title: "Chapter 12"
description: "Fast Fourier Transform"
icon: "article"
date: "2026-07-06T00:00:00+07:00"
lastmod: "2026-07-06T00:00:00+07:00"
katex: true
draft: false
toc: true
---

{{% alert icon="💡" context="info" %}}
<strong>"<em>The FFT is to digital signal processing what the invention of the telescope was to astronomy</em>" — Richard G. Lyons</strong>
{{% /alert %}}

{{% alert icon="📘" context="success" %}}
<p style="text-align: justify;"><em>Chapter 12 introduces the Fourier transform and the Fast Fourier Transform (FFT), two of the most important tools in numerical computing and scientific simulation. The chapter develops the mathematical foundations of Fourier analysis, discrete sampling, spectral representations, and the discrete Fourier transform. Efficient FFT algorithms, including the radix-2 Cooley–Tukey factorization, are presented alongside techniques for real-valued data, sine and cosine transforms, and multidimensional Fourier analysis. The discussion extends to large-scale computation through cache-aware execution, out-of-core algorithms, distributed-memory FFTs, and performance-oriented implementations. Throughout the chapter, mathematical theory is integrated with practical Rust implementations, providing readers with efficient tools for signal processing, partial differential equations, scientific imaging, spectral methods, and large-scale numerical simulation.</em></p>
{{% /alert %}}

# 12.1. Introduction

Fourier transform methods are most profitably understood as a change of representation. Instead of describing a function by its values at each time or spatial location, we describe it by the amplitudes of its oscillatory modes. Section 12.1.1 has formalized this change through the transform pair (12.1.1)–(12.1.2) and emphasized linearity and the multiplicative action of differentiation in frequency space. We now sharpen that structural viewpoint and prepare the transition to the discrete algebra underlying FFT computation.

The decisive idea is spectral diagonalization. The complex exponentials $e^{2\pi i f t}$ form a continuous family of eigenfunctions for every linear operator that commutes with translations. If $T$ is shift-invariant, meaning $T(\tau_a h)=\tau_a (Th)$ for the shift operator $(\tau_a h)(t)=h(t-a)$, then each Fourier mode is mapped to a scalar multiple of itself. In operator language, translation invariance implies simultaneous diagonalization in the Fourier basis. What appears in physical space as a global integral or differential operator becomes, after transformation, multiplication by a scalar-valued symbol. The Fourier transform therefore converts structural complexity into algebraic simplicity.

This mechanism explains the ubiquity of Fourier structure in constant-coefficient partial differential equations. When an operator is built from derivatives with constant coefficients, its action is determined entirely by its symbol, a polynomial in frequency. Solving such equations reduces to independent scalar relations for each mode. The continuous theory thus reveals that many infinite-dimensional problems are already diagonal in a hidden coordinate system; the transform merely exposes that coordinate system.

The same principle reappears, in finite-dimensional form, when functions are replaced by sampled data vectors. Sampling over a periodic grid leads naturally to cyclic structure. Discrete translation invariance corresponds algebraically to circulant matrices. These matrices share a common eigenbasis consisting of discrete Fourier modes, and hence admit exact diagonalization by the discrete Fourier transform. In matrix form, this means that a structured linear operator can be written as a similarity transformation to a diagonal matrix. The FFT is therefore not simply a fast summation scheme; it is the computational device that performs this change of basis efficiently.

The progression from continuous to discrete is thus conceptually seamless. The eigenfunction property of complex exponentials in function space becomes the eigenvector property of Fourier modes in finite-dimensional linear algebra. The diagonal symbol of a differential operator becomes the diagonal of a matrix in frequency coordinates. The computational advantage follows directly: once transformed, independent frequency components can be processed pointwise, enabling algorithms whose cost grows nearly linearly with problem size.

With this structural perspective in place, we now turn to the discrete setting and make explicit how the algebra of sampled data mirrors the operator theory developed above.

## 12.1.1. The Continuous Fourier Transform as a Linear Operator

Let $h : \mathbb{R} \to \mathbb{C}$. Using frequency $f$ in cycles per unit, define the Fourier transform pair as:

$$
H(f) \equiv \int_{-\infty}^{\infty} h(t)\, e^{-2\pi i f t}\, dt,
\\
h(t) \equiv \int_{-\infty}^{\infty} H(f)\, e^{+2\pi i f t}\, df
\tag{12.1.1}
$$

If angular frequency $\omega = 2\pi f$ is preferred, then:

\begin{equation}
\begin{aligned}
H(\omega) &\equiv \int_{-\infty}^{\infty} h(t)\, e^{-i\omega t}\, dt, \\
h(t) &\equiv \frac{1}{2\pi} \int_{-\infty}^{\infty} H(\omega)\, e^{+i\omega t}\, d\omega
\end{aligned}
\tag{12.1.2}
\end{equation}

The key structural fact is linearity: the Fourier transform $\mathcal{F} : h \mapsto H$ is a linear operator. Orthogonality of the complex exponentials implies Parseval and Plancherel identities, the convolution theorem, and differentiation rules. For example,

$$\mathcal{F}(h')(f) = (2\pi i f)\, H(f) \tag{12.1.3}$$

so constant-coefficient differential operators act multiplicatively in frequency space.

This observation explains why Fourier structure appears naturally in models governed by constant-coefficient PDEs and translation-invariant physics. Whenever an operator commutes with shifts, Fourier modes serve as eigenfunctions. The continuous theory therefore anticipates the exact diagonalization that becomes algebraically explicit in the discrete setting.

## 12.1.2. Circulant Structure, Discrete Diagonalization, and PDE Solvers

Translation invariance becomes exact algebra in periodic discrete models. On a periodic grid, the shift operator and any circulant matrix share eigenvectors given by discrete Fourier modes. Many discretizations of constant-coefficient operators on periodic domains therefore lead to circulant or block-circulant matrices that are diagonalizable by the discrete Fourier transform (Zhao et al., 2024).

To make this explicit, let $x \in \mathbb{C}^{N}$ and define the primitive root,

$$
\omega_N \equiv e^{-2\pi i / N}
\tag{12.1.4}
$$

The (unnormalized) DFT matrix $F_N \in \mathbb{C}^{N\times N}$ is,

$$
(F_N)_{k,n} \equiv \omega_N^{kn}
\qquad
k,n \in \{0,1,\dots,N-1\}
\tag{12.1.5}
$$

The DFT is the linear map $X = F_N x$, and discrete orthogonality is expressed by:

$$F_N^{*} F_N = N I \tag{12.1.6}$$

Now let $C(h)$ be the circulant matrix generated by $h \in \mathbb{C}^N$, defined by:

$$(C(h)x)[n] = \sum_{m=0}^{N-1} h[m]\, x\bigl[(n-m)\bmod N\bigr] \tag{12.1.7}$$

Then

$$
C(h) = F_N^{-1}\, \mathrm{diag}(F_N h)\, F_N
\tag{12.1.8}
$$

Thus discrete Fourier modes are eigenvectors of any circulant operator, and convolution reduces to pointwise multiplication in frequency space.

A canonical application is the periodic Poisson problem:

$$
u''(x) = f(x), \qquad
u(x+L) = u(x), \qquad
\int_{0}^{L} u(x)\, dx = 0
\tag{12.1.9}
$$

After discretization on a uniform grid, the discrete Laplacian is circulant. Transforming to Fourier space diagonalizes the operator, so solving reduces to independent scalar divisions for each frequency mode. This structure underlies FFT-based MAC schemes for Stokes equations with periodic boundary conditions (Zhao et al., 2024).

More broadly, GPU-parallelized Fourier spectral methods exploit the same diagonalization principle in phase-field simulations, where repeated evaluation of spectral operators becomes a dominant computational cost (Boccardo et al., 2023). In structured linear algebra, circulant embedding converts Toeplitz or block-Toeplitz systems into FFT-diagonalizable form, enabling scalable FFT-based matvec pipelines in mixed-precision GPU environments (Venkat et al., 2025).

These examples illustrate a unifying idea: when translation invariance or periodicity is present, Fourier modes diagonalize the model, and the discrete Fourier transform becomes a computationally central tool.

### Rust Implementation

Following the discussion in Sections 12.1.1–12.1.2 on the operator interpretation of the Fourier transform and the role of discrete Fourier modes in diagonalizing translation-invariant operators, Program 12.1.1 provides a concrete numerical illustration of these structural ideas in finite-dimensional form. While the continuous transform defined in Equations (12.1.1)–(12.1.2) acts on functions over $\mathbb{R}$, numerical computation inevitably replaces functions with sampled vectors and integral operators with matrices. In this setting, the discrete Fourier transform becomes the algebraic mechanism that exposes the hidden eigenstructure of shift-invariant operators. The program constructs the discrete Fourier matrix described in Equation (12.1.5), verifies its orthogonality property (12.1.6), and demonstrates explicitly how circulant convolution operators admit diagonalization through Fourier coordinates as expressed in Equation (12.1.8). To further connect these algebraic ideas with applications in numerical PDEs, the program also solves a periodic Poisson problem using both spectral and finite-difference models, illustrating how the Fourier basis converts differential operators into independent scalar relations for each frequency mode.

At the core of the implementation is the explicit construction of the **discrete Fourier transform matrix** defined in Equation (12.1.5). The function `dft_matrix` generates the matrix $F_N$ using the primitive root of unity introduced in Equation (12.1.4). Each entry of the matrix is computed as $\omega_N^{kn}$, producing a dense complex matrix whose multiplication with a vector implements the linear map $X = F_N x$. The related function `omega_n` computes the primitive root $\omega_N$, while `inverse_from_unnormalized_dft` forms the inverse transform using the identity $F_N^{-1} = \frac{1}{N}F_N^{*}$, which follows from the discrete orthogonality relation expressed in Equation (12.1.6).

Matrix operations required to demonstrate the algebraic structure of the transform are implemented through a set of supporting routines. The function `conj_transpose` computes the conjugate transpose of a complex matrix, enabling verification of the orthogonality relation $F_N^{*}F_N = NI$. The functions `matmul` and `matvec` perform matrix–matrix and matrix–vector multiplications, respectively, while `scale_matrix` and `identity` generate scaled matrices and identity matrices needed to check numerical identities. The helper function `max_abs_diff_matrix` computes the maximum entrywise deviation between two matrices, allowing the program to report how closely the numerical computation satisfies theoretical equalities such as Equation (12.1.6).

The circulant operator introduced in Equation (12.1.7) is constructed by the function `circulant_from_kernel`. Given a kernel vector $h$, this routine forms the circulant matrix $C(h)$ whose action corresponds to discrete circular convolution. The program then verifies the diagonalization identity (12.1.8) by comparing the explicitly constructed circulant matrix with the matrix product $F_N^{-1}\,\mathrm{diag}(F_N h)\,F_N$. Additional methods such as `circular_convolution` and `diag_from_vec` are used to confirm the convolution theorem numerically: applying the circulant operator in physical space yields the same result as transforming to frequency space, multiplying componentwise, and transforming back.

To illustrate the connection with partial differential equations, the program includes a simple periodic Poisson solver. The right-hand side function $f(x)$ is sampled on a uniform grid over a periodic interval, producing the discrete data vector used in the computation. The program first solves the equation using a spectral model, in which the second derivative operator is represented by the continuous eigenvalues associated with Fourier modes. This demonstrates how differentiation in physical space becomes multiplication by a frequency-dependent scalar factor, as described in Equation (12.1.3). The resulting solution is then verified by reconstructing the spectral second derivative and comparing it to the sampled forcing term.

A second implementation uses a finite-difference approximation to the Laplacian based on the circulant stencil corresponding to Equation (12.1.7). The eigenvalues of this discrete operator are known analytically and are applied in Fourier space to compute the numerical solution. The residual of the finite-difference operator applied to the computed solution is then compared with the original forcing function, providing an additional verification that the Fourier transform diagonalizes both the continuous spectral operator and its discrete circulant approximation.

The `main` function coordinates these experiments. It first constructs the Fourier matrix and verifies the orthogonality property of Equation (12.1.6). Next, it builds a circulant convolution operator and confirms the diagonalization identity (12.1.8) and the convolution theorem through numerical comparisons. Finally, it solves the periodic Poisson equation using both spectral and finite-difference models and reports the resulting residuals. These checks demonstrate that the algebraic relationships derived in Sections 12.1.1–12.1.2 hold numerically to within floating-point roundoff.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
```

```rust
// Program 12.1.1: Operator Viewpoint and Circulant Diagonalization via the DFT (O(N^2) demo)
//
// Problem statement (Sections 12.1.1–12.1.2):
// Demonstrate, in finite-dimensional form, the structural claims behind (12.1.5)–(12.1.8):
// (i) the DFT is a linear map x -> F_N x,
// (ii) discrete orthogonality F_N^* F_N = N I (Eq. 12.1.6),
// (iii) a circulant convolution operator C(h) is diagonalized by the DFT as
//       C(h) = F_N^{-1} diag(F_N h) F_N (Eq. 12.1.8),
// and connect this diagonalization viewpoint to a periodic Poisson-type example (Eq. 12.1.9).
//
// Important consistency note (Poisson demo):
// - This revised program provides *two* consistent Poisson checks:
//   A) Spectral operator (continuous eigenvalues) solve + spectral residual check.
//   B) Finite-difference (circulant stencil) solve + finite-difference residual check.
//
// Dependencies (add to Cargo.toml):
// [dependencies]
// num-complex = "0.4"

use num_complex::Complex64;
use std::f64::consts::PI;

fn omega_n(n: usize) -> Complex64 {
    // ω_N = exp(-2π i / N)  (Eq. 12.1.4)
    let theta = -2.0 * PI / (n as f64);
    Complex64::new(theta.cos(), theta.sin())
}

fn dft_matrix(n: usize) -> Vec<Vec<Complex64>> {
    // (F_N)_{k,n} = ω_N^{k n}  (Eq. 12.1.5)
    let w = omega_n(n);
    let mut f = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for k in 0..n {
        for j in 0..n {
            let p = (k * j) as i32;
            f[k][j] = w.powi(p);
        }
    }
    f
}

fn conj_transpose(a: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let n = a.len();
    let m = a[0].len();
    let mut at = vec![vec![Complex64::new(0.0, 0.0); n]; m];
    for i in 0..n {
        for j in 0..m {
            at[j][i] = a[i][j].conj();
        }
    }
    at
}

fn matmul(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let n = a.len();
    let p = a[0].len();
    let m = b[0].len();
    assert_eq!(b.len(), p);

    let mut c = vec![vec![Complex64::new(0.0, 0.0); m]; n];
    for i in 0..n {
        for k in 0..p {
            let aik = a[i][k];
            for j in 0..m {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

fn matvec(a: &[Vec<Complex64>], x: &[Complex64]) -> Vec<Complex64> {
    let n = a.len();
    let m = a[0].len();
    assert_eq!(x.len(), m);

    let mut y = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n {
        let mut s = Complex64::new(0.0, 0.0);
        for j in 0..m {
            s += a[i][j] * x[j];
        }
        y[i] = s;
    }
    y
}

fn scale_matrix(a: &[Vec<Complex64>], alpha: f64) -> Vec<Vec<Complex64>> {
    let n = a.len();
    let m = a[0].len();
    let mut out = vec![vec![Complex64::new(0.0, 0.0); m]; n];
    for i in 0..n {
        for j in 0..m {
            out[i][j] = a[i][j] * alpha;
        }
    }
    out
}

fn max_abs_diff_matrix(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> f64 {
    let n = a.len();
    let m = a[0].len();
    assert_eq!(b.len(), n);
    assert_eq!(b[0].len(), m);

    let mut mx: f64 = 0.0;
    for i in 0..n {
        for j in 0..m {
            mx = mx.max((a[i][j] - b[i][j]).norm());
        }
    }
    mx
}

fn identity(n: usize) -> Vec<Vec<Complex64>> {
    let mut i = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for k in 0..n {
        i[k][k] = Complex64::new(1.0, 0.0);
    }
    i
}

fn diag_from_vec(v: &[Complex64]) -> Vec<Vec<Complex64>> {
    let n = v.len();
    let mut d = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        d[i][i] = v[i];
    }
    d
}

fn circulant_from_kernel(h: &[Complex64]) -> Vec<Vec<Complex64>> {
    // (C(h)x)[n] = Σ_m h[m] x[(n-m) mod N]  (Eq. 12.1.7)
    let n = h.len();
    let mut c = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for row in 0..n {
        for m in 0..n {
            let col = (row + n - m) % n;
            c[row][col] = h[m];
        }
    }
    c
}

fn circular_convolution(h: &[Complex64], x: &[Complex64]) -> Vec<Complex64> {
    let n = h.len();
    assert_eq!(x.len(), n);
    let mut y = vec![Complex64::new(0.0, 0.0); n];
    for r in 0..n {
        let mut s = Complex64::new(0.0, 0.0);
        for m in 0..n {
            let idx = (r + n - m) % n;
            s += h[m] * x[idx];
        }
        y[r] = s;
    }
    y
}

fn max_abs_diff_vec(a: &[Complex64], b: &[Complex64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(u, v)| (*u - *v).norm())
        .fold(0.0_f64, f64::max)
}

fn inverse_from_unnormalized_dft(f: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    // For unnormalized F_N: F_N^{-1} = (1/N) F_N^*
    let n = f.len();
    let fh = conj_transpose(f);
    scale_matrix(&fh, 1.0 / (n as f64))
}

fn signed_frequency_index(k: usize, n: usize) -> i32 {
    // Map k in {0,...,N-1} to integer frequencies {0,1,...,N/2,-N/2+1,...,-1} (even N).
    let half = (n / 2) as i32;
    let kk = k as i32;
    if kk <= half {
        kk
    } else {
        kk - (n as i32)
    }
}

fn main() {
    let n: usize = 16;

    let f = dft_matrix(n);
    let finv = inverse_from_unnormalized_dft(&f);

    // -------------------------------------------------------------------------
    // 1) Verify discrete orthogonality: F_N^* F_N = N I  (Eq. 12.1.6)
    // -------------------------------------------------------------------------
    let f_star = conj_transpose(&f);
    let gram = matmul(&f_star, &f);
    let target = scale_matrix(&identity(n), n as f64);
    let err_orth = max_abs_diff_matrix(&gram, &target);

    println!("N = {n}");
    println!("Max error in F^*F = N I (Eq. 12.1.6): {:.3e}", err_orth);

    // -------------------------------------------------------------------------
    // 2) Circulant operator C(h) and diagonalization:
    //    C(h) = F^{-1} diag(F h) F  (Eq. 12.1.8)
    // -------------------------------------------------------------------------
    let mut h = vec![Complex64::new(0.0, 0.0); n];
    h[0] = Complex64::new(2.0, 0.0);
    h[1] = Complex64::new(-1.0, 0.0);
    h[n - 1] = Complex64::new(-1.0, 0.0);

    let c = circulant_from_kernel(&h);

    let fh = matvec(&f, &h);
    let d = diag_from_vec(&fh);

    let tmp = matmul(&d, &f);
    let c_hat = matmul(&finv, &tmp);

    let err_diag = max_abs_diff_matrix(&c, &c_hat);
    println!(
        "Max error in C(h) = F^{{-1}} diag(F h) F (Eq. 12.1.8): {:.3e}",
        err_diag
    );

    let x: Vec<Complex64> = (0..n)
        .map(|j| {
            let t = 2.0 * PI * (j as f64) / (n as f64);
            Complex64::new(t.cos() + 0.25 * (3.0 * t).cos(), 0.0)
        })
        .collect();

    let y_mat = matvec(&c, &x);
    let y_conv = circular_convolution(&h, &x);

    let fx = matvec(&f, &x);
    let mut prod = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..n {
        prod[k] = fh[k] * fx[k];
    }
    let y_spec = matvec(&finv, &prod);

    println!(
        "Max ||C(h)x - conv(h,x)||_inf: {:.3e}",
        max_abs_diff_vec(&y_mat, &y_conv)
    );
    println!(
        "Max ||C(h)x - F^{{-1}}((Fh).*(Fx))||_inf: {:.3e}",
        max_abs_diff_vec(&y_mat, &y_spec)
    );

    // -------------------------------------------------------------------------
    // 3) Periodic Poisson demo (Eq. 12.1.9) with two consistent variants.
    // -------------------------------------------------------------------------
    let l = 2.0 * PI;
    let dx = l / (n as f64);

    // Mean-zero right-hand side f(x) sampled on the grid.
    let f_samples: Vec<Complex64> = (0..n)
        .map(|j| {
            let xj = (j as f64) * dx;
            Complex64::new((3.0 * xj).sin() + 0.5 * (2.0 * xj).cos(), 0.0)
        })
        .collect();

    let fhat = matvec(&f, &f_samples);

    // -------------------------
    // 3A) Spectral solve + spectral residual check
    // -------------------------
    // Solve in Fourier space using continuous eigenvalues for d^2/dx^2:
    //   u'' <-> -(2π k/L)^2 U_k,  k != 0, and enforce mean(u)=0 by U_0 = 0.
    let mut uhat_spec = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..n {
        let kk = signed_frequency_index(k, n) as f64;
        if kk.abs() < 0.5 {
            uhat_spec[k] = Complex64::new(0.0, 0.0);
        } else {
            let lambda = -((2.0 * PI * kk) / l).powi(2);
            uhat_spec[k] = fhat[k] / Complex64::new(lambda, 0.0);
        }
    }
    let u_spec = matvec(&finv, &uhat_spec);

    // Spectral residual: compute u'' spectrally and compare to f.
    let mut u2hat = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..n {
        let kk = signed_frequency_index(k, n) as f64;
        let lambda = -((2.0 * PI * kk) / l).powi(2);
        u2hat[k] = Complex64::new(lambda, 0.0) * uhat_spec[k];
    }
    let u2_spec = matvec(&finv, &u2hat);
    let res_spec = max_abs_diff_vec(&u2_spec, &f_samples);

    println!(
        "Poisson (spectral) max residual ||u''_spec - f||_inf: {:.3e}",
        res_spec
    );

    // -------------------------
    // 3B) Finite-difference solve + finite-difference residual check
    // -------------------------
    // The circulant stencil kernel h corresponds to (-dx^2 D2), i.e.
    //   (C(h) u)_j = 2u_j - u_{j+1} - u_{j-1} = -dx^2 (D2 u)_j.
    // Thus the FD Laplacian eigenvalues in Fourier space are:
    //   (D2 u)_k <-> -(4/dx^2) sin^2(pi k / N) U_k.
    let mut uhat_fd = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..n {
        let kk = signed_frequency_index(k, n) as f64;
        if kk.abs() < 0.5 {
            uhat_fd[k] = Complex64::new(0.0, 0.0); // mean(u)=0
        } else {
            let s = (PI * (kk.abs()) / (n as f64)).sin();
            let lambda_fd = -(4.0 / (dx * dx)) * s * s; // eigenvalue for D2
            uhat_fd[k] = fhat[k] / Complex64::new(lambda_fd, 0.0);
        }
    }
    let u_fd = matvec(&finv, &uhat_fd);

    // FD residual: compute D2 u via the same circulant stencil and compare to f.
    let cu = matvec(&c, &u_fd);
    let mut d2u_fd = vec![Complex64::new(0.0, 0.0); n];
    for j in 0..n {
        d2u_fd[j] = -cu[j] / Complex64::new(dx * dx, 0.0);
    }
    let res_fd = max_abs_diff_vec(&d2u_fd, &f_samples);

    println!(
        "Poisson (finite-difference) max residual ||D2_fd u - f||_inf: {:.3e}",
        res_fd
    );

    // Small summary norms (sanity).
    let u_spec_norm = u_spec.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let u_fd_norm = u_fd.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    println!("||u_spec||_2 (sampled): {:.6e}", u_spec_norm);
    println!("||u_fd||_2   (sampled): {:.6e}", u_fd_norm);
}
```

Program 12.1.1 provides a concrete numerical demonstration of the structural interpretation of the Fourier transform developed in Sections 12.1.1–12.1.2. By constructing the discrete Fourier matrix explicitly and verifying its orthogonality property, the program confirms the linear-algebraic viewpoint that underlies many Fourier-based algorithms. The numerical verification of the diagonalization identity $C(h) = F_N^{-1}\mathrm{diag}(F_Nh)F_N$ illustrates how convolution operators become diagonal in Fourier coordinates, reducing a dense matrix operation to a simple componentwise multiplication.

The periodic Poisson example further illustrates how this diagonalization principle translates into practical numerical algorithms. In the spectral formulation, differentiation is represented by multiplication with frequency-dependent eigenvalues, allowing the differential equation to be solved independently for each Fourier mode. In the finite-difference formulation, the discrete Laplacian remains circulant and is therefore diagonalized by the same Fourier basis, leading to an equally straightforward solution process. The small residuals reported by the program demonstrate that both approaches satisfy their respective operator equations to machine precision.

Together, these experiments highlight the fundamental computational idea emphasized throughout this chapter: translation-invariant operators possess a hidden diagonal structure revealed by Fourier coordinates. The discrete Fourier transform therefore serves not only as a transform of data but also as a change of basis that converts structured linear operators into simple diagonal forms. This perspective prepares the ground for the development of fast algorithms in later sections, where the Fast Fourier Transform will make it possible to perform these basis transformations with computational complexity $O(N\log N)$ rather than the $O(N^2)$ cost associated with explicit matrix multiplication.

## 12.1.3. Modern Computational Context: Performance and Precision

In contemporary high-performance computing, the Fourier transform is not merely a theoretical operator but a performance-critical kernel. Multi-dimensional FFTs dominate runtime in large-scale pseudo-spectral turbulence simulations, where communication patterns and memory movement play a decisive role in overall scalability (Yeung et al., 2024). On modern GPU and processing-in-memory architectures, FFT implementations are often limited by memory bandwidth rather than floating-point throughput, motivating collaborative acceleration strategies (Ibrahim and Aga, 2024).

At the same time, precision management has become integral to FFT-based workflows. Runtime selection of FFT libraries and floating-point precision can be guided by predictive error models to meet accuracy thresholds while optimizing performance and energy consumption (Lehner et al., 2025). Mixed-precision frameworks for FFT-based Toeplitz matvec pipelines demonstrate that careful precision control can preserve error tolerances while achieving substantial performance gains on large GPU systems (Venkat et al., 2025).

From this perspective, Fourier analysis in numerical computing serves two intertwined roles. First, it provides structural diagonalization for translation-invariant and periodic models. Second, it supplies a computational primitive whose efficiency and accuracy are shaped by hardware architecture, communication costs, and precision strategies. The next section connects this structural foundation to the discrete sampling process and formal definition of the discrete Fourier transform, preparing the ground for the Fast Fourier Transform algorithm itself.

# 12.2. Fourier Transform of Discretely Sampled Data

In computational work, we almost never possess a continuous function $h(t)$ in closed form. Instead, we are given samples: finite in number, typically uniformly spaced, and represented in floating-point arithmetic. The passage from the continuous Fourier transform to the discrete Fourier transform is therefore not merely a change of notation but a change of mathematical object.

Three transforms must be carefully distinguished:

1. The continuous-time Fourier transform (CTFT) of $h(t)$.
2. The discrete-time Fourier transform (DTFT) of an infinite sequence $h[n]$.
3. The discrete Fourier transform (DFT) of a finite vector $x \in \mathbb{C}^N$.

Confusion between these objects is a common source of misinterpretation in numerical applications (Henry, 2024; Huang et al., 2025). This section clarifies their relationships and highlights the practical consequences for simulation and instrumentation.

## 12.2.1. Uniform Sampling and Spectral Replication

Assume uniform sampling with step size $\Delta > 0$. The sampled sequence is:

$$
h[n] \equiv h(n\Delta), \qquad
n \in \mathbb{Z}
\tag{12.2.1}
$$

The sampling rate and Nyquist frequency are:

$$
f_s = \frac{1}{\Delta}, \qquad
f_c = \frac{f_s}{2} = \frac{1}{2\Delta}
\tag{12.2.2}
$$

A canonical model of sampling is multiplication by an impulse train,

$$
p(t) \equiv \sum_{n \in \mathbb{Z}} \delta(t - n\Delta), \qquad
h_s(t) \equiv h(t)\, p(t)
\tag{12.2.3}
$$

Taking the Fourier transform yields the replication identity,

$$
H_s(f) = \frac{1}{\Delta}
\sum_{m \in \mathbb{Z}} H\!\left(f - \frac{m}{\Delta}\right)
\tag{12.2.4}
$$

Thus sampling in time produces periodic copies of the spectrum in frequency with period $f_s$.

If these copies overlap inside the Nyquist interval $[-f_s/2, f_s/2)$, high-frequency content folds into lower frequencies. This folding phenomenon is aliasing. The classical Nyquist condition requires that spectral replicas do not overlap.

Modern developments extend classical sampling theory. Sub-Nyquist and nonuniform sampling criteria for multitone signals have been developed under periodic delay sampling frameworks (Cao et al., 2023). More broadly, generalized Nyquist–Shannon-type theorems using Koopman operator structure extend recoverability conditions beyond strictly bandlimited signals (Zeng et al., 2024). Despite these extensions, aliasing remains the fundamental structural constraint imposed by sampling.

### Rust Implementation

Following the discussion in Section 12.2.1 on uniform sampling and spectral replication, Program 12.2.1 provides a concrete numerical demonstration of how sampled data encode frequency information and how aliasing arises when the Nyquist condition is violated. In practical numerical computing we rarely manipulate continuous functions $h(t)$ directly. Instead we work with discrete samples $h[n]=h(n\Delta)$ defined by Equation (12.2.1). The sampling step $\Delta$ determines the sampling rate $f_s$ and Nyquist frequency $f_c$ through Equation (12.2.2). This program constructs sampled cosine signals and computes their discrete Fourier transform in order to reveal how the sampling process determines the frequencies that can be reliably represented. Two representative scenarios are examined. The first uses a frequency below the Nyquist limit and therefore exhibits correct spectral localization. The second uses a frequency above the Nyquist limit and demonstrates aliasing caused by spectral replication, as described by Equation (12.2.4). The program therefore provides a direct computational illustration of the structural consequences of uniform sampling.

At the core of the implementation is the function `dft`, which computes the discrete Fourier transform of a finite complex vector. Although the Fast Fourier Transform will later replace this quadratic algorithm, the direct $O(N^2)$ formulation provides a transparent implementation of the transform defined in Equation (12.2.6). The function evaluates the complex exponential kernel for each pair of time index $n$ and frequency index $k$, accumulating the contributions to produce the spectral vector $X[k]$. Using the direct formulation is pedagogically useful because it makes explicit the relationship between sampled time-domain values and their spectral representation.

The program also includes the helper function `freq_from_bin`, which converts a discrete transform index into its corresponding physical frequency. According to Equation (12.2.8), the DFT bin $k$ represents the frequency $f_k = k/(N\Delta)$. However, because the DFT is periodic in frequency, indices greater than $N/2$ correspond to negative frequencies. The function therefore applies the periodic indexing rule described by Equation (12.2.9) to map these indices into the interval $[ -f_s/2 , f_s/2 )$. This conversion allows the program to report frequencies in physical units rather than purely discrete indices.

To illustrate the effect of spectral replication, the function `predicted_alias` computes the theoretical alias frequency that arises when a signal exceeds the Nyquist limit. Sampling produces periodic copies of the spectrum with spacing $f_s$, as stated by the replication identity in Equation (12.2.4). Any frequency outside the Nyquist interval therefore folds back into the interval $[ -f_s/2 , f_s/2 )$. The function implements this folding operation by subtracting integer multiples of the sampling rate until the resulting frequency lies within the Nyquist band.

The function `dominant_bin` identifies the frequency bin containing the largest spectral magnitude. This diagnostic is used to determine where the sampled signal’s energy appears in the discrete spectrum. Because the test signals are pure cosines, their spectra exhibit conjugate symmetry and therefore produce two peaks at positive and negative frequencies. The program also reports the partner bin $N-k$, which corresponds to the symmetric frequency predicted by the real-valued structure of the input signal.

The function `sample_cosine` generates the sampled signal used in the experiment. Given a frequency $f_0$, the function evaluates the cosine function at the discrete sampling points $t_n=n\Delta$. This operation corresponds directly to the sampling model $h[n]=h(n\Delta)$ defined by Equation (12.2.1). By adjusting the value of $f_0$, the program can demonstrate both correct spectral representation and aliasing.

The `main` function orchestrates the numerical experiment. It first selects the sampling step $\Delta$, from which the sampling rate $f_s$ and Nyquist frequency $f_c$ are derived using Equation (12.2.2). Two signals are then analyzed. The first has a frequency below the Nyquist limit and therefore appears at the correct spectral location. The second has a frequency above the Nyquist limit and therefore aliases to a different frequency inside the Nyquist band. For each signal the program computes the DFT, identifies the dominant frequency bin, converts the bin index to a physical frequency, and prints a short neighborhood of spectral magnitudes around the dominant bin. This output illustrates how the discrete transform represents frequency information and reveals the folding behavior predicted by the sampling theory developed in Section 12.2.1.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
```

```rust
// Program 12.2.1: Uniform Sampling, Nyquist Frequency, and Aliasing
//
// This program demonstrates the sampling model of Section 12.2.1.
// Given samples h[n] = h(nΔ) (Eq. 12.2.1) with sampling rate
// f_s = 1/Δ and Nyquist frequency f_c = f_s/2 (Eq. 12.2.2),
// the program illustrates aliasing by comparing two sampled
// sinusoids:
//
// 1. A frequency below Nyquist (no aliasing).
// 2. A frequency above Nyquist (aliasing occurs).
//
// A direct O(N^2) discrete Fourier transform is used only
// as a diagnostic tool to identify the dominant spectral bin.
//
// Dependencies (add to Cargo.toml):
//
// [dependencies]
// num-complex = "0.4"

use num_complex::Complex64;
use std::f64::consts::PI;

fn dft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    let mut x_spec = vec![Complex64::new(0.0, 0.0); n];

    for k in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);

        for n_idx in 0..n {
            let theta = -2.0 * PI * (n_idx as f64) * (k as f64) / (n as f64);
            let w = Complex64::new(theta.cos(), theta.sin());
            sum += x[n_idx] * w;
        }

        x_spec[k] = sum;
    }

    x_spec
}

fn freq_from_bin(k: usize, n: usize, delta: f64) -> f64 {
    let k_i = k as i32;
    let n_i = n as i32;
    let half = (n / 2) as i32;

    let signed_k = if k_i <= half { k_i } else { k_i - n_i };

    (signed_k as f64) / ((n as f64) * delta)
}

fn predicted_alias(freq: f64, fs: f64) -> f64 {
    let mut f_alias = freq;

    while f_alias >= 0.5 * fs {
        f_alias -= fs;
    }

    while f_alias < -0.5 * fs {
        f_alias += fs;
    }

    f_alias
}

fn dominant_bin(x_spec: &[Complex64]) -> usize {
    let mut best_k = 0;
    let mut best_mag = -1.0;

    for (k, val) in x_spec.iter().enumerate() {
        let mag = val.norm();

        if mag > best_mag {
            best_mag = mag;
            best_k = k;
        }
    }

    best_k
}

fn sample_cosine(freq: f64, delta: f64, n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let t = (i as f64) * delta;
            Complex64::new((2.0 * PI * freq * t).cos(), 0.0)
        })
        .collect()
}

fn analyze_case(label: &str, freq: f64, delta: f64, n: usize) {
    let fs = 1.0 / delta;
    let fc = 0.5 * fs;

    println!("--- {} ---", label);

    println!("Δ = {:.6e}", delta);
    println!("f_s = 1/Δ = {:.6e} Hz  (Eq. 12.2.2)", fs);
    println!("f_c = f_s/2 = {:.6e} Hz  (Eq. 12.2.2)", fc);

    println!("Chosen sinusoid frequency f0 = {:.6e} Hz", freq);

    let x = sample_cosine(freq, delta, n);
    let x_spec = dft(&x);

    let k_star = dominant_bin(&x_spec);

    let f_star = freq_from_bin(k_star, n, delta);

    let f_alias = predicted_alias(freq, fs);

    println!("Dominant DFT bin index k* = {}", k_star);

    println!(
        "Mapped bin frequency f(k*) = {:.6e} Hz  (Eqs. 12.2.8–12.2.9)",
        f_star
    );

    println!(
        "Predicted aliased frequency in [-f_s/2, f_s/2): {:.6e} Hz  (Eq. 12.2.4 concept)",
        f_alias
    );

    let partner = (n - k_star) % n;
    let f_partner = freq_from_bin(partner, n, delta);

    println!("Partner bin index (N-k*) mod N = {}", partner);

    println!(
        "Mapped partner frequency f(N-k*) = {:.6e} Hz",
        f_partner
    );

    println!("|X[k]| near the dominant bin:");

    let guard = 2;
    let start = k_star.saturating_sub(guard);
    let end = (k_star + guard).min(n - 1);

    for k in start..=end {
        println!(
            "  k={:>3}, f={:+.6e} Hz, |X|={:.6e}",
            k,
            freq_from_bin(k, n, delta),
            x_spec[k].norm()
        );
    }

    println!();
}

fn main() {
    let n: usize = 128;

    let delta: f64 = 1.0 / 1000.0;

    let fs = 1.0 / delta;
    let fc = 0.5 * fs;

    let freq_case_a = 120.0;

    let freq_case_b = fc + 90.0;

    analyze_case(
        "Case A: Frequency below Nyquist (no aliasing)",
        freq_case_a,
        delta,
        n,
    );

    analyze_case(
        "Case B: Frequency above Nyquist (aliasing occurs)",
        freq_case_b,
        delta,
        n,
    );

    println!("Interpretation:");

    println!(
        "Sampling with f_s = {:.0} Hz replicates the spectrum with period f_s (Eq. 12.2.4).",
        fs
    );

    println!(
        "Frequencies outside [-f_s/2, f_s/2) fold back into this interval. This is aliasing."
    );
}
```

Program 12.2.1 provides a computational illustration of the sampling principles introduced in Section 12.2.1. By generating sampled cosine signals and examining their discrete spectra, the program demonstrates how the sampling step determines the set of representable frequencies. When the signal frequency lies below the Nyquist limit, the DFT correctly identifies its spectral location. When the frequency exceeds this limit, the spectrum folds back into the Nyquist interval, producing an aliased frequency that differs from the original signal frequency.

The numerical results also reveal an additional practical consideration. Because the DFT evaluates frequencies only on a discrete grid with spacing $1/(N\Delta)$, the detected spectral peak may not coincide exactly with the true signal frequency unless the signal is perfectly aligned with a DFT bin. In such cases the spectral energy spreads across nearby bins. This phenomenon is known as spectral leakage and will be examined in detail in Section 12.2.2.

The program therefore serves as a bridge between the theoretical sampling model and the practical interpretation of discrete Fourier transforms. It illustrates how aliasing arises directly from spectral replication and shows how discrete frequency bins correspond to physical frequency units. These insights form the conceptual foundation for the subsequent discussion of windowing, leakage, and the discrete Fourier transform itself.

## 12.2.2. Finite Records, Windowing, and Spectral Leakage

In practice, we do not observe the infinite sequence ${h[n]}_{n\in\mathbb{Z}}$. Instead, we record a finite segment

$$
x[n] = h(n\Delta)\, w[n], \qquad
n = 0, \dots, N-1
\tag{12.2.5}
$$

where $w[n]$ is a window function. If no explicit taper is applied, $w[n]$ is implicitly rectangular: equal to 1 inside the recording interval and 0 outside.

Multiplication by a window in time corresponds to convolution with the window’s spectrum in frequency. Consequently, spectral energy spreads across neighboring frequencies. This spreading is known as spectral leakage. Leakage is unavoidable unless the signal is exactly periodic over the observation window.

Henry (2024) emphasizes that FFT output is confined to discrete frequency bins and is sensitive to leakage, particularly when estimating peak frequency, amplitude, or phase. The proposed two-window FFT procedure improves peak estimation and suppresses leakage effects.

This issue is not merely theoretical. In eddy-current non-destructive testing for nuclear components, a probe driven at excitation frequency $f_0$ produces a sampled voltage signal. The complex phasor corresponding to $f_0$ is extracted from a DFT coefficient. If the observation window does not contain an integer number of excitation cycles, leakage spreads energy across bins and biases phase estimation. Huang et al. (2025) mitigate this effect by enforcing strict periodicity matching between excitation frequency and sampling design, achieving high phase linearity in practical inspection systems.

Thus windowing assumptions directly affect the reliability of engineering measurements.

## 12.2.3. The Discrete Fourier Transform and Frequency Interpretation

Let $x \in \mathbb{C}^N$. The DFT is defined by:

$$
X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-2\pi i nk / N}, \qquad
k = 0,1,\dots,N-1
\tag{12.2.6}
$$

with inverse as:

$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k]\, e^{+2\pi i nk / N}
\tag{12.2.7}
$$

If samples correspond to times $t_n = n\Delta$, then DFT bins represent physical frequencies,

$$
f_k = \frac{k}{N\Delta} \qquad
k = 0, \dots, N-1
\tag{12.2.8}
$$

For $k > N/2$, periodic indexing gives:

$$f_k = \frac{k-N}{N\Delta} \tag{12.2.9}$$

Correct mapping between DFT indices and physical frequency units is essential in FFT-based PDE solvers and spectral differentiation schemes (Zhao et al., 2024). Misinterpretation of bin indexing can lead to incorrect spectral operators or phase errors.

With the chosen normalization, the discrete Parseval identity becomes:

$$
\sum_{n=0}^{N-1} |x[n]|^2
=
\frac{1}{N} \sum_{k=0}^{N-1} |X[k]|^2
\tag{12.2.10}
$$

This identity legitimizes interpreting $|X[k]|^2$ as a discretized power spectrum, subject to windowing and normalization conventions.

## 12.2.4. Symmetry and Computational Implications

When $x[n] \in \mathbb{R}$, the DFT satisfies conjugate symmetry,

$$
X[N-k] = \overline{X[k]}, \qquad
k = 1, \dots, N-1.
\tag{12.2.11}
$$

This symmetry reduces arithmetic and memory requirements. Real-valued FFT implementations exploit this property, which is particularly important in bandwidth-limited GPU and accelerator environments (Ibrahim and Aga, 2024; Yeung et al., 2024).

In applications involving nonuniform sampling, classical FFT assumptions are violated. Nonuniform FFT (NUFFT/NNFFT) methods preserve near $O(N\log N)$ complexity by spreading data onto an oversampled grid before applying standard FFTs and interpolating back to nonuniform nodes (Kircheis et al., 2023; Cox et al., 2025). Even in these generalized settings, the classical FFT remains the computational core.

### Rust Implementation

Following the discussion in Sections 12.2.2–12.2.4 on finite records, windowing, and the interpretation of discrete Fourier coefficients, Program 12.2.2 provides a practical computational illustration of how these concepts appear in numerical signal analysis. In realistic measurements we never observe an infinite sequence $h[n]$. Instead we work with a finite record $x[n]$ formed by multiplying the sampled signal $h(n\Delta)$ with a window function $w[n]$, as described by Equation (12.2.5). This truncation modifies the spectral representation of the signal and leads to spectral leakage, whereby energy spreads across neighboring frequency bins. The program constructs a sampled cosine signal whose frequency is not aligned with the discrete Fourier grid and compares the resulting spectrum under a rectangular window and a tapered Hann window. It then evaluates the discrete Fourier transform defined by Equation (12.2.6), verifies the inverse transform in Equation (12.2.7), interprets the frequency bins according to Equations (12.2.8)–(12.2.9), checks the Parseval identity in Equation (12.2.10), and confirms the conjugate symmetry property stated in Equation (12.2.11). The program therefore provides a unified computational demonstration of the spectral properties of finite sampled data.

At the core of the implementation are the functions `dft` and `idft`, which compute the discrete Fourier transform and its inverse according to Equations (12.2.6) and (12.2.7). Although later sections introduce the Fast Fourier Transform to reduce computational complexity, the direct quadratic implementation used here makes the mathematical structure transparent. Each output coefficient $X[k]$ is obtained by summing the contributions of all time-domain samples multiplied by the complex exponential kernel. The inverse routine reconstructs the signal from its spectral coefficients and therefore provides a direct verification of the transform pair.

To interpret the resulting spectrum in physical units, the program includes the function `freq_from_bin`. This method converts a discrete index $k$ into its corresponding frequency using Equation (12.2.8). Because the discrete Fourier transform represents frequencies periodically, indices greater than $N/2$ correspond to negative frequencies. The function therefore applies the periodic indexing rule of Equation (12.2.9), ensuring that the reported frequencies lie within the Nyquist interval. This mapping is essential in practical applications such as spectral differentiation or PDE solvers, where incorrect interpretation of bin indices leads to incorrect physical operators.

The functions `rectangular_window` and `hann_window` generate the window sequences used in Equation (12.2.5). The rectangular window corresponds to the implicit truncation that occurs when a finite record is taken without tapering. The Hann window introduces a smooth taper that reduces the discontinuities at the boundaries of the observation interval. Applying the window is performed by the function `apply_window_real`, which multiplies the sampled signal by the window sequence before the transform is computed. Comparing these two windows reveals how tapering suppresses spectral leakage.

Several diagnostic routines quantify the spectral behavior. The function `dominant_bin` identifies the index of the largest spectral coefficient, allowing the program to locate the dominant frequency component of the signal. The function `leakage_metric` estimates the amount of spectral energy outside the immediate neighborhood of the peak frequency, providing a numerical measure of leakage. The function `parseval_relative_error` evaluates the energy relation given by Equation (12.2.10) by comparing the total energy of the time-domain signal with the appropriately scaled energy of its spectrum. Finally, the function `conjugate_symmetry_error` verifies the symmetry property of Equation (12.2.11), which must hold when the input signal is real-valued.

The `main` function organizes the numerical experiment. It begins by defining the sampling step $\Delta$, from which the sampling rate and frequency grid spacing follow. A cosine signal is generated whose frequency is deliberately chosen not to coincide with a DFT bin. The program then computes the spectrum of the signal using both the rectangular and Hann windows, identifies the dominant spectral bins, and evaluates the leakage metric for each case. Additional diagnostics verify the Parseval identity, confirm conjugate symmetry, and check that the inverse transform reconstructs the original windowed signal. Together these computations illustrate how windowing affects spectral localization while preserving the fundamental properties of the discrete Fourier transform.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
```

```rust
// Program 12.2.2: Windowing, Spectral Leakage, Frequency Interpretation, Parseval, and Conjugate Symmetry
//
// Problem statement (Sections 12.2.2–12.2.4):
// We observe a finite record
//   x[n] = h(nΔ) w[n],  n = 0,...,N-1   (Eq. 12.2.5),
// where w[n] is a window. Windowing in time produces spectral leakage in frequency.
// This program compares a rectangular window with a Hann taper on a tone whose frequency
// is not exactly bin-centered, then quantifies leakage.
//
// It also implements the DFT and inverse DFT
//   X[k] = Σ x[n] exp(-2π i nk/N)  (Eq. 12.2.6),
//   x[n] = (1/N) Σ X[k] exp(+2π i nk/N)  (Eq. 12.2.7),
// maps bins to physical frequency (Eqs. 12.2.8–12.2.9),
// verifies Parseval (Eq. 12.2.10),
// and checks conjugate symmetry for real inputs (Eq. 12.2.11).
//
// Dependencies (add to Cargo.toml):
// [dependencies]
// num-complex = "0.4"

use num_complex::Complex64;
use std::f64::consts::PI;

fn dft(x: &[Complex64]) -> Vec<Complex64> {
    // Eq. (12.2.6)
    let n = x.len();
    let mut x_spec = vec![Complex64::new(0.0, 0.0); n];

    for k in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for n_idx in 0..n {
            let theta = -2.0 * PI * (n_idx as f64) * (k as f64) / (n as f64);
            let w = Complex64::new(theta.cos(), theta.sin());
            sum += x[n_idx] * w;
        }
        x_spec[k] = sum;
    }

    x_spec
}

fn idft(x_spec: &[Complex64]) -> Vec<Complex64> {
    // Eq. (12.2.7)
    let n = x_spec.len();
    let inv_n = 1.0 / (n as f64);
    let mut x = vec![Complex64::new(0.0, 0.0); n];

    for n_idx in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for k in 0..n {
            let theta = 2.0 * PI * (n_idx as f64) * (k as f64) / (n as f64);
            let w = Complex64::new(theta.cos(), theta.sin());
            sum += x_spec[k] * w;
        }
        x[n_idx] = sum * inv_n;
    }

    x
}

fn freq_from_bin(k: usize, n: usize, delta: f64) -> f64 {
    // Eqs. (12.2.8)–(12.2.9)
    let k_i = k as i32;
    let n_i = n as i32;
    let half = (n / 2) as i32;

    let signed_k = if k_i <= half { k_i } else { k_i - n_i };

    (signed_k as f64) / ((n as f64) * delta)
}

fn dominant_bin(x_spec: &[Complex64]) -> usize {
    let mut best_k = 0usize;
    let mut best_mag = -1.0_f64;

    for (k, val) in x_spec.iter().enumerate() {
        let mag = val.norm();
        if mag > best_mag {
            best_mag = mag;
            best_k = k;
        }
    }

    best_k
}

fn rectangular_window(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

fn hann_window(n: usize) -> Vec<f64> {
    // Hann taper: w[n] = 0.5(1 - cos(2π n/(N-1))) for N>1
    if n <= 1 {
        return vec![1.0; n];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f64) / denom).cos()))
        .collect()
}

fn apply_window_real(x: &[f64], w: &[f64]) -> Vec<Complex64> {
    assert_eq!(x.len(), w.len());
    x.iter()
        .zip(w.iter())
        .map(|(xi, wi)| Complex64::new(xi * wi, 0.0))
        .collect()
}

fn energy_time(x: &[Complex64]) -> f64 {
    x.iter().map(|z| z.norm_sqr()).sum()
}

fn energy_freq(x_spec: &[Complex64]) -> f64 {
    x_spec.iter().map(|z| z.norm_sqr()).sum()
}

fn parseval_relative_error(x: &[Complex64], x_spec: &[Complex64]) -> f64 {
    // Eq. (12.2.10): Σ |x[n]|^2 = (1/N) Σ |X[k]|^2
    let n = x.len() as f64;
    let lhs = energy_time(x);
    let rhs = energy_freq(x_spec) / n;
    (lhs - rhs).abs() / (lhs.max(1e-300))
}

fn conjugate_symmetry_error(x_spec: &[Complex64]) -> f64 {
    // Eq. (12.2.11): X[N-k] = conj(X[k]) for k=1,...,N-1
    let n = x_spec.len();
    let mut mx = 0.0_f64;
    for k in 1..n {
        let lhs = x_spec[(n - k) % n];
        let rhs = x_spec[k].conj();
        mx = mx.max((lhs - rhs).norm());
    }
    mx
}

fn leakage_metric(x_spec: &[Complex64], peak_k: usize, guard: usize) -> f64 {
    // Leakage = energy outside [peak_k-guard, peak_k+guard] and its conjugate partner band,
    // divided by total energy in spectrum.
    let n = x_spec.len();
    let total = energy_freq(x_spec);
    if total == 0.0 {
        return 0.0;
    }

    let partner = (n - peak_k) % n;

    let mut keep = vec![false; n];

    for dk in 0..=guard {
        let k1 = peak_k.saturating_sub(dk);
        let k2 = (peak_k + dk) % n;
        keep[k1] = true;
        keep[k2] = true;

        let p1 = partner.saturating_sub(dk);
        let p2 = (partner + dk) % n;
        keep[p1] = true;
        keep[p2] = true;
    }

    let mut leaked = 0.0_f64;
    for k in 0..n {
        if !keep[k] {
            leaked += x_spec[k].norm_sqr();
        }
    }

    leaked / total
}

fn print_peak_report(label: &str, delta: f64, x_spec: &[Complex64]) {
    let n = x_spec.len();
    let k_star = dominant_bin(x_spec);
    let f_star = freq_from_bin(k_star, n, delta);
    let mag = x_spec[k_star].norm();
    println!("{label}");
    println!("  dominant bin k* = {k_star}, mapped f(k*) = {:+.6e} Hz, |X[k*]| = {:.6e}", f_star, mag);

    let partner = (n - k_star) % n;
    let f_partner = freq_from_bin(partner, n, delta);
    println!("  partner bin (N-k*) mod N = {partner}, mapped f = {:+.6e} Hz", f_partner);
}

fn main() {
    // Sampling and record parameters.
    let n: usize = 256;
    let delta: f64 = 1.0 / 1000.0; // Δ = 1 ms => f_s = 1000 Hz
    let fs: f64 = 1.0 / delta;
    let df: f64 = 1.0 / ((n as f64) * delta); // frequency grid spacing

    // Choose a tone frequency that is NOT bin-centered to expose leakage.
    // Bin-centered would be f0 = m*df. We choose a fractional offset.
    let f0: f64 = 123.45;

    // Generate a finite record h(nΔ) (Eq. 12.2.5 with w[n]=1 first).
    let h: Vec<f64> = (0..n)
        .map(|i| {
            let t = (i as f64) * delta;
            (2.0 * PI * f0 * t).cos()
        })
        .collect();

    let w_rect = rectangular_window(n);
    let w_hann = hann_window(n);

    let x_rect = apply_window_real(&h, &w_rect);
    let x_hann = apply_window_real(&h, &w_hann);

    // DFTs (Eq. 12.2.6).
    let x_rect_spec = dft(&x_rect);
    let x_hann_spec = dft(&x_hann);

    // Frequency interpretation (Eqs. 12.2.8–12.2.9).
    println!("N = {n}, Δ = {:.6e} s", delta);
    println!("f_s = 1/Δ = {:.3} Hz  (Eq. 12.2.2)", fs);
    println!("Δf = 1/(NΔ) = {:.6e} Hz  (bin spacing)", df);
    println!("Test tone frequency f0 = {:.6e} Hz (not bin-centered)\n", f0);

    print_peak_report("Rectangular window (implicit):", delta, &x_rect_spec);
    print_peak_report("Hann window:", delta, &x_hann_spec);
    println!();

    // Quantify leakage.
    let peak_rect = dominant_bin(&x_rect_spec);
    let peak_hann = dominant_bin(&x_hann_spec);

    let guard = 1usize; // keep ±1 bin around each peak and its conjugate partner
    let leak_rect = leakage_metric(&x_rect_spec, peak_rect, guard);
    let leak_hann = leakage_metric(&x_hann_spec, peak_hann, guard);

    println!("Leakage metric (energy outside ±{guard} bins of the two symmetric peaks):");
    println!("  rectangular: {:.6e}", leak_rect);
    println!("  hann:        {:.6e}", leak_hann);
    println!();

    // Parseval (Eq. 12.2.10).
    let p_rect = parseval_relative_error(&x_rect, &x_rect_spec);
    let p_hann = parseval_relative_error(&x_hann, &x_hann_spec);

    println!("Parseval relative error (Eq. 12.2.10):");
    println!("  rectangular: {:.3e}", p_rect);
    println!("  hann:        {:.3e}", p_hann);
    println!();

    // Conjugate symmetry (Eq. 12.2.11) for real input.
    let sym_rect = conjugate_symmetry_error(&x_rect_spec);
    let sym_hann = conjugate_symmetry_error(&x_hann_spec);

    println!("Conjugate symmetry max error (Eq. 12.2.11, real input):");
    println!("  rectangular: {:.3e}", sym_rect);
    println!("  hann:        {:.3e}", sym_hann);
    println!();

    // Inverse DFT check (Eq. 12.2.7), measured in max norm.
    let x_rect_rec = idft(&x_rect_spec);
    let x_hann_rec = idft(&x_hann_spec);

    let mut err_rect = 0.0_f64;
    let mut err_hann = 0.0_f64;
    for i in 0..n {
        err_rect = err_rect.max((x_rect_rec[i] - x_rect[i]).norm());
        err_hann = err_hann.max((x_hann_rec[i] - x_hann[i]).norm());
    }

    println!("Inverse DFT reconstruction max error (Eq. 12.2.7):");
    println!("  rectangular: {:.3e}", err_rect);
    println!("  hann:        {:.3e}", err_hann);
}
```

Program 12.2.2 demonstrates how the theoretical principles developed in Sections 12.2.2–12.2.4 appear in practical numerical computations. The comparison between the rectangular and Hann windows shows how truncating a signal without tapering leads to significant spectral leakage, while a smooth window reduces energy spreading across frequency bins. Although windowing does not eliminate leakage entirely, it significantly improves spectral concentration and therefore enhances the reliability of frequency estimation.

The numerical diagnostics further confirm the structural properties of the discrete Fourier transform. The Parseval identity verifies that the transform preserves signal energy when the appropriate normalization is applied. The conjugate symmetry test confirms that real-valued input signals produce spectra with symmetric complex conjugate pairs. Finally, the inverse transform reconstruction demonstrates that the transform pair defined by Equations (12.2.6) and (12.2.7) forms a numerically stable representation of the signal.

These experiments illustrate why window design and frequency interpretation are essential in practical signal analysis. Finite observation intervals inevitably distort spectral representations, and careful use of window functions mitigates these effects. Understanding these issues prepares the ground for the next section, where the computational cost of the discrete Fourier transform is reduced from quadratic complexity to the quasi-linear complexity of the Fast Fourier Transform algorithm.

## 12.2.5. Section Summary

Section 12.2 establishes four essential principles:

1. Uniform sampling produces spectral replication and potential aliasing.
2. Finite records introduce windowing and spectral leakage.
3. The DFT is an exact linear transform of a finite vector, not automatically the CTFT of a function.
4. Correct frequency indexing is essential for both simulation and instrumentation.

These principles govern both FFT-based PDE solvers and industrial signal-processing pipelines. In spectral simulation, correct bin interpretation ensures accurate differentiation and operator application (Zhao et al., 2024). In eddy-current inspection systems, leakage control determines whether phase information remains reliable (Huang et al., 2025).

Understanding these subtleties prepares the ground for Section 12.3, where we replace the quadratic cost of (12.2.6) by the quasi-linear complexity of the Fast Fourier Transform algorithm.

# 12.3. Fast Fourier Transform (FFT)

The Fast Fourier Transform (FFT) is not a new transform but a fast algorithm for computing the discrete Fourier transform (12.2.6). The DFT remains the same linear map $x \mapsto F_N x$; what changes is the computational strategy used to evaluate it. By factorizing the dense DFT matrix into a product of sparse structured matrices, the arithmetic cost is reduced from $O(N^2)$ to $O(N \log N)$. This reduction transforms Fourier diagonalization from a theoretical device into a practical computational primitive in simulation, signal processing, and structured linear algebra (Boccardo et al., 2023; Zhao et al., 2024).

## 12.3.1. Radix-2 Cooley–Tukey Factorization

Assume $N$ is even and define:

$$\omega_N = e^{-2\pi i/N} \tag{12.3.1}$$

The DFT is,

$$X[k] = \sum_{n=0}^{N-1} x[n]\, \omega_N^{nk} \tag{12.3.2}$$

Split the index $n$ into even and odd parts:

$$
X[k] =
\sum_{m=0}^{N/2-1} x[2m]\, \omega_N^{(2m)k}
+
\sum_{m=0}^{N/2-1} x[2m+1]\, \omega_N^{(2m+1)k}
\tag{12.3.3}
$$

Using $\omega_N^{2} = \omega_{N/2}$, rewrite

$$
X[k] =
\sum_{m=0}^{N/2-1} x[2m]\, \omega_{N/2}^{mk}
+
\omega_N^{k}
\sum_{m=0}^{N/2-1} x[2m+1]\, \omega_{N/2}^{mk}
\tag{12.3.4}
$$

Define the length-$N/2$ transforms,

$$
E[k] = \sum_{m=0}^{N/2-1} x[2m]\, \omega_{N/2}^{mk},
\qquad
O[k] = \sum_{m=0}^{N/2-1} x[2m+1]\, \omega_{N/2}^{mk}
\tag{12.3.5}
$$

Then,

$$
X[k] = E[k] + \omega_N^{k} O[k]
\tag{12.3.6}
$$

and, using periodicity,

$$
X\!\left[k + \frac{N}{2}\right] = E[k] - \omega_N^{k} O[k]
\tag{12.3.7}
$$

Equations (12.3.6)–(12.3.7) form the classical butterfly combination:

$$
\begin{bmatrix}
X[k] \\
X\!\left[k + \frac{N}{2}\right]
\end{bmatrix}
=
\begin{bmatrix}
1 & \omega_N^{k} \\
1 & -\omega_N^{k}
\end{bmatrix}
\begin{bmatrix}
E[k] \\
O[k]
\end{bmatrix}
\tag{12.3.8}
$$

Thus an $N$-point DFT reduces to two $N/2$-point DFTs plus $O(N)$ additional operations. If $T(N)$ denotes the cost,

$$
T(N) = 2T\!\left(\frac{N}{2}\right) + cN
\tag{12.3.9}
$$

which solves to

$$
T(N) = O(N \log_2 N)
\tag{12.3.10}
$$

This complexity reduction is decisive in practice. In Fourier spectral simulations of phase-field models, repeated evaluation of spectral operators becomes feasible only because of this quasi-linear scaling (Boccardo et al., 2023). Similarly, FFT-based MAC schemes for periodic Stokes equations rely on repeated forward and inverse FFTs whose cost scales as $O(N \log N)$ rather than $O(N^2)$ (Zhao et al., 2024).

## 12.3.2. Algorithmic Structure and Data Movement

The recursive derivation can be implemented iteratively. A common scheduling strategy consists of:

1. Permuting input data into bit-reversed order.
2. Applying successive butterfly stages, doubling the subtransform size at each stage.

The smallest building block is the 2-point DFT,

$$
\begin{bmatrix}
X[0] \\
X[1]
\end{bmatrix}
=
\begin{bmatrix}
1 & 1 \\
1 & -1
\end{bmatrix}
\begin{bmatrix}
x[0] \\
x[1]
\end{bmatrix}
\tag{12.3.11}
$$

Higher-radix variants, such as radix-4 or radix-8, reduce the number of twiddle multiplications at the cost of more complex indexing patterns. In modern implementations, these design choices are evaluated not only by arithmetic counts but also by memory-access patterns.

On contemporary GPU and accelerator architectures, FFT performance is often limited by memory bandwidth rather than floating-point throughput. Efficient implementations must minimize global memory traffic, avoid unnecessary transposes, and exploit data locality (Ibrahim and Aga, 2024). In large-scale pseudo-spectral turbulence simulations, multi-dimensional FFTs and associated communication patterns dominate runtime, and scalability depends critically on data decomposition strategies such as slab or pencil layouts (Yeung et al., 2024).

Thus, while the asymptotic complexity is $O(N \log N)$, real-world performance is governed by memory movement and communication costs as much as by arithmetic operations.

### Rust Implementation

Following the derivation of the radix-2 Cooley–Tukey factorization in Section 12.3.1 and the discussion of algorithmic scheduling in Section 12.3.2, Program 12.3.1 presents a practical Rust implementation of the Fast Fourier Transform. While the discrete Fourier transform defined in Equation (12.3.2) requires $O(N^2)$ arithmetic operations when evaluated directly, the recursive decomposition developed in Equations (12.3.3)–(12.3.8) reveals that the computation can be organized as a sequence of butterfly combinations applied across progressively larger subtransforms. When this factorization is implemented iteratively with bit-reversed input ordering, the total arithmetic cost satisfies the recurrence in Equation (12.3.9), yielding the quasi-linear complexity $O(N\log N)$ described by Equation (12.3.10). The program demonstrates this structure explicitly by implementing an in-place radix-2 FFT together with its inverse transform and several numerical diagnostics that verify correctness, reconstruction accuracy, and energy conservation. In addition, a direct $O(N^2)$ DFT is included for validation on moderate problem sizes so that the FFT output can be compared against the exact definition of the transform.

At the core of the implementation are the functions `fft_in_place` and `ifft_in_place`, which compute the forward and inverse discrete Fourier transforms using the radix-2 Cooley–Tukey algorithm derived in Section 12.3.1. The forward transform evaluates the same mapping defined in Equation (12.3.2), but instead of computing each coefficient independently, the algorithm repeatedly decomposes the transform into smaller problems using the relationships established in Equations (12.3.6) and (12.3.7). These relations combine the even and odd subsequences $E[k]$ and $O[k]$ into pairs of output coefficients through the butterfly structure described by Equation (12.3.8). The inverse transform applies the well-known conjugation identity of the Fourier transform, performing a forward FFT on conjugated data and then scaling by $1/N$ to recover the original sequence.

The function `bit_reversal_permute` implements the data reordering step required by the iterative FFT schedule discussed in Section 12.3.2. In the recursive formulation of the algorithm, the input vector is repeatedly divided into even and odd subsequences. When the algorithm is expressed iteratively instead of recursively, this hierarchical decomposition is reproduced by permuting the input indices into bit-reversed order. The helper function `bit_reverse` computes the reversed binary representation of each index so that elements appear in the correct order for the subsequent butterfly stages.

Once the data have been permuted, the function `fft_in_place` applies successive butterfly stages whose sizes double at each iteration. The smallest building block corresponds to the two-point transform described by Equation (12.3.11). At each stage the program computes the complex twiddle factors $\omega_m = e^{-2\pi i/m}$ and multiplies them by the odd-indexed components before combining them with the even components. This iterative structure exactly mirrors the recursive factorization developed in Equations (12.3.3)–(12.3.8), but it avoids function recursion and improves cache locality by operating directly on contiguous memory segments.

For validation purposes the program includes the function `dft_direct`, which evaluates the discrete Fourier transform directly from Equation (12.3.2). Although this approach has quadratic complexity and is impractical for large problems, it provides an exact reference implementation against which the FFT results can be compared. The function `max_abs_diff` measures the maximum difference between the FFT output and the direct DFT result, allowing the numerical accuracy of the implementation to be quantified.

Additional diagnostic functions verify structural properties of the transform. The function `parseval_relative_error` checks the discrete energy conservation relation stated in Equation (12.2.10), comparing the energy of the time-domain signal with the appropriately scaled energy of its spectrum. The helper functions `energy_time` and `energy_freq` compute the corresponding norms. These checks confirm that the numerical implementation respects the fundamental invariants of the Fourier transform.

The `main` function organizes a simple numerical experiment. It constructs a synthetic signal consisting of a sum of sinusoidal components together with a constant offset and computes both its direct DFT and its FFT representation. The results are compared to measure the numerical error between the two methods. The program then applies the inverse FFT to reconstruct the original signal and reports the maximum reconstruction error. Finally, Parseval consistency is evaluated to verify that the spectral representation preserves signal energy within floating-point precision. Together these diagnostics illustrate the reliability and numerical stability of the FFT implementation.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
```

```rust
// Program 12.3.1: Radix-2 Cooley–Tukey FFT (Iterative, In-Place) with Bit-Reversal Scheduling
//
// Problem statement (Sections 12.3.1–12.3.2):
// Implement the radix-2 Cooley–Tukey factorization that reduces the DFT (Eq. 12.3.2)
// into two half-size transforms combined by butterfly operations (Eqs. 12.3.6–12.3.8).
// Use the standard iterative schedule:
//   1) Permute data into bit-reversed order,
//   2) Apply log2(N) butterfly stages, doubling the subtransform size each stage,
// thereby achieving O(N log N) arithmetic (Eqs. 12.3.9–12.3.10).
//
// The program also includes a direct O(N^2) DFT for validation on small N.
// It checks:
//   - max error between FFT and direct DFT,
//   - inverse FFT reconstruction error,
//   - Parseval consistency (optional diagnostic).
//
// Dependencies (add to Cargo.toml):
// [dependencies]
// num-complex = "0.4"

use num_complex::Complex64;
use std::f64::consts::PI;

fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

fn bit_reverse(mut x: usize, bits: u32) -> usize {
    let mut y = 0usize;
    for _ in 0..bits {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}

fn bit_reversal_permute(a: &mut [Complex64]) {
    let n = a.len();
    let bits = (n as f64).log2() as u32;
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if j > i {
            a.swap(i, j);
        }
    }
}

fn fft_in_place(a: &mut [Complex64]) {
    // Forward FFT using ω_N = exp(-2π i / N) (Eq. 12.3.1).
    // Iterative Cooley–Tukey: butterflies (Eqs. 12.3.6–12.3.8).
    let n = a.len();
    assert!(is_power_of_two(n), "fft_in_place requires N to be a power of two.");

    // Stage 0: bit-reversal permutation (Section 12.3.2 scheduling).
    bit_reversal_permute(a);

    // Stage s combines blocks of size m = 2^s.
    let mut m = 2usize;
    while m <= n {
        let half = m / 2;
        // Principal m-th root for this stage: exp(-2π i / m).
        let theta = -2.0 * PI / (m as f64);
        let w_m = Complex64::new(theta.cos(), theta.sin());

        for k0 in (0..n).step_by(m) {
            let mut w = Complex64::new(1.0, 0.0);
            for j in 0..half {
                // Butterfly:
                // u = a[k0 + j], v = w * a[k0 + j + half]
                // a[k0 + j]       = u + v
                // a[k0 + j + half]= u - v
                let u = a[k0 + j];
                let v = w * a[k0 + j + half];
                a[k0 + j] = u + v;
                a[k0 + j + half] = u - v;
                w *= w_m;
            }
        }

        m *= 2;
    }
}

fn ifft_in_place(a: &mut [Complex64]) {
    // Inverse FFT:
    // x = (1/N) conj( FFT( conj(X) ) )
    let n = a.len() as f64;

    for z in a.iter_mut() {
        *z = z.conj();
    }

    fft_in_place(a);

    for z in a.iter_mut() {
        *z = z.conj() / n;
    }
}

fn dft_direct(x: &[Complex64]) -> Vec<Complex64> {
    // Direct DFT (Eq. 12.3.2) for validation (O(N^2)).
    let n = x.len();
    let mut x_spec = vec![Complex64::new(0.0, 0.0); n];

    for k in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for n_idx in 0..n {
            let theta = -2.0 * PI * (n_idx as f64) * (k as f64) / (n as f64);
            let w = Complex64::new(theta.cos(), theta.sin());
            sum += x[n_idx] * w;
        }
        x_spec[k] = sum;
    }

    x_spec
}

fn max_abs_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut mx = 0.0_f64;
    for i in 0..a.len() {
        mx = mx.max((a[i] - b[i]).norm());
    }
    mx
}

fn energy_time(x: &[Complex64]) -> f64 {
    x.iter().map(|z| z.norm_sqr()).sum()
}

fn energy_freq(x_spec: &[Complex64]) -> f64 {
    x_spec.iter().map(|z| z.norm_sqr()).sum()
}

fn parseval_relative_error(x: &[Complex64], x_spec: &[Complex64]) -> f64 {
    // With the unnormalized forward DFT / normalized inverse convention,
    // Parseval takes the form Σ|x|^2 = (1/N) Σ|X|^2 (Eq. 12.2.10).
    let n = x.len() as f64;
    let lhs = energy_time(x);
    let rhs = energy_freq(x_spec) / n;
    (lhs - rhs).abs() / lhs.max(1e-300)
}

fn main() {
    // Choose a small power-of-two size for a transparent correctness check.
    let n: usize = 256;
    assert!(is_power_of_two(n));

    // Construct a test signal: sum of two tones plus a small DC offset.
    // This matches the "finite vector x ∈ C^N" viewpoint of the DFT (Eq. 12.3.2).
    let f1 = 17.0;
    let f2 = 41.0;
    let mut x = vec![Complex64::new(0.0, 0.0); n];

    for n_idx in 0..n {
        let t = n_idx as f64;
        let s1 = (2.0 * PI * f1 * t / (n as f64)).cos();
        let s2 = 0.7 * (2.0 * PI * f2 * t / (n as f64)).sin();
        x[n_idx] = Complex64::new(0.2 + s1 + s2, 0.0);
    }

    // Compute the direct DFT for reference.
    let x_spec_ref = dft_direct(&x);

    // Compute FFT in-place.
    let mut x_fft = x.clone();
    fft_in_place(&mut x_fft);

    // Compare FFT to direct DFT.
    let err_fft = max_abs_diff(&x_fft, &x_spec_ref);

    // Check inverse FFT reconstruction.
    let mut x_rec = x_fft.clone();
    ifft_in_place(&mut x_rec);
    let err_ifft = max_abs_diff(&x_rec, &x);

    // Parseval consistency (Eq. 12.2.10) as a useful invariant check.
    let err_parseval = parseval_relative_error(&x, &x_fft);

    println!("N = {}", n);
    println!("Max ||FFT(x) - DFT(x)||_inf: {:.3e}", err_fft);
    println!("Max ||IFFT(FFT(x)) - x||_inf: {:.3e}", err_ifft);
    println!("Parseval relative error (Eq. 12.2.10): {:.3e}", err_parseval);

    // A small report of dominant bins (illustrative).
    // Identify the largest few magnitudes.
    let mut mags: Vec<(usize, f64)> = x_fft.iter().enumerate().map(|(k, z)| (k, z.norm())).collect();
    mags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top 6 |X[k]| bins (k, |X[k]|):");
    for i in 0..6 {
        println!("  k={:>3}, |X|={:.6e}", mags[i].0, mags[i].1);
    }
}
```

Program 12.3.1 demonstrates how the mathematical factorization of the discrete Fourier transform developed in Section 12.3 leads directly to a highly efficient computational algorithm. By exploiting the recursive decomposition of the transform into even and odd subsequences, the radix-2 Cooley–Tukey method reduces the computational cost from quadratic complexity to the quasi-linear complexity described by Equation (12.3.10). This dramatic reduction explains why FFT-based methods have become indispensable in modern numerical computing.

The diagnostic experiments performed in the program confirm several important theoretical properties. The FFT output agrees closely with the direct evaluation of the DFT, demonstrating that the algorithm computes the same transform defined in Equation (12.3.2). The inverse transform reconstructs the original signal to nearly machine precision, confirming the correctness of the transform pair. Finally, the verification of the Parseval identity illustrates that the numerical implementation preserves signal energy in accordance with the theoretical energy relation.

These results highlight the central role of the FFT as a computational primitive. Many numerical algorithms, including spectral PDE solvers, signal processing pipelines, and structured matrix methods, rely on repeated forward and inverse transforms. The radix-2 implementation presented here illustrates the essential principles underlying these methods and provides a foundation for the multidimensional transforms discussed in the following section.

## 12.3.3. Multi-Dimensional FFT and Scientific Simulation

Many scientific applications require multi-dimensional transforms. If $N = N_1 N_2$, a one-dimensional data vector may be reshaped into an $N_1 \times N_2$ array, and the transform computed dimension by dimension. FFTs are first applied along one index direction, twiddle factors are incorporated as required by the factorization, and FFTs are then applied along the remaining direction, with transposes interleaved to maintain contiguous memory access.

This procedure is best understood as a tensor-product structure. The two-dimensional discrete Fourier transform of an array $x_{m,n}$ can be written as:

$$
\hat{x}_{k,\ell}
=
\sum_{m=0}^{N_1-1}
\sum_{n=0}^{N_2-1}
x_{m,n}\,
e^{-2\pi i \left(\frac{mk}{N_1} + \frac{n\ell}{N_2}\right)}
\tag{12.3.12}
$$

The exponential kernel factorizes:

$$
e^{-2\pi i \left(\frac{mk}{N_1} + \frac{n\ell}{N_2}\right)}
=
e^{-2\pi i \frac{mk}{N_1}}
\cdot
e^{-2\pi i \frac{n\ell}{N_2}}
\tag{12.3.13}
$$

which reveals that the 2D transform equals a sequence of 1D transforms along rows followed by 1D transforms along columns. In matrix language,

$$
F_{N_1,N_2}
=
F_{N_1} \otimes F_{N_2}
\tag{12.3.14}
$$

where $\otimes$ denotes the Kronecker product. Thus, multi-dimensional FFTs are not new algorithms but structured compositions of one-dimensional FFTs. The computational cost becomes,

$$
\mathcal{O}\!\bigl(N_1 N_2 (\log N_1 + \log N_2)\bigr)
\tag{12.3.15}
$$

which scales as $\mathcal{O}(N \log N)$ for balanced grids.

This decomposition mirrors the Cooley–Tukey factorization at a higher structural level. Just as a large one-dimensional transform is factored into smaller pieces, a multi-dimensional transform factors across coordinate directions. The practical difficulty shifts from arithmetic complexity to memory movement. In large-scale simulations, transposes dominate runtime because they induce global communication on distributed systems. Modern implementations therefore emphasize communication-avoiding strategies, pencil decompositions, and GPU-aware data layouts to sustain scalability.

In Fourier pseudo-spectral methods, these structural properties are decisive. Spatial derivatives are computed spectrally by multiplying Fourier coefficients by frequency factors. For example, in two dimensions,

$$
\left[
\widehat{\partial_x u}(k,\ell)
=
\frac{2\pi i k}{L_x}\, \hat{u}(k,\ell),
\qquad
\widehat{\partial_y u}(k,\ell)
=
\frac{2\pi i \ell}{L_y}\, \hat{u}(k,\ell)
\right]
\tag{12.3.16}
$$

Time-stepping schemes then alternate between physical space, where nonlinear terms are evaluated pointwise, and frequency space, where derivatives are diagonal and inexpensive. Each timestep typically requires multiple forward and inverse multi-dimensional FFTs. Consequently, the overall feasibility of high-resolution turbulence, plasma, or phase-field simulations depends directly on the performance and scalability of these transforms (Yeung et al., 2024).

Aliasing control becomes more subtle in multiple dimensions. Nonlinear products in physical space correspond to convolutions in frequency space, producing mode interactions that may exceed the Nyquist limits in each coordinate direction. De-aliasing strategies such as the 2/3-rule zero out high-frequency modes before inverse transformation to maintain spectral accuracy. In three-dimensional turbulence simulations, this step can significantly affect both memory footprint and arithmetic intensity.

In structured linear algebra, FFT-based matrix–vector pipelines for block-triangular Toeplitz matrices exploit the same tensor-product decomposition principle. A block Toeplitz operator corresponds to convolution in one or more dimensions, so diagonalization in the Fourier basis reduces the matvec to pointwise multiplications across frequency slices. The computational kernel becomes a sequence of multi-dimensional FFTs combined with diagonal scaling. Mixed-precision strategies further allow performance tuning while maintaining prescribed error tolerances, with high precision reserved for sensitive frequency bands or refinement steps (Venkat et al., 2025).

Thus, multi-dimensional FFTs are not merely extensions of the one-dimensional case; they form the computational backbone of modern scientific simulation and structured linear algebra. Their efficiency hinges less on arithmetic count than on memory layout, communication patterns, and precision management.

### Rust Implementation

Following the discussion in Section 12.3.3 on the tensor-product structure of multi-dimensional Fourier transforms, Program 12.3.2 provides a practical implementation of the two-dimensional Fast Fourier Transform and illustrates its role in spectral simulation. While the two-dimensional transform defined in Equation (12.3.12) appears at first to require a double summation over all spatial indices, the factorization of the exponential kernel in Equation (12.3.13) reveals that the computation separates naturally into a sequence of one-dimensional transforms applied along each coordinate direction. In matrix terms, the transform corresponds to the Kronecker-product structure described by Equation (12.3.14), allowing the full transform to be evaluated using row-wise and column-wise FFTs. The program implements this decomposition directly, demonstrating how a two-dimensional FFT can be constructed from repeated applications of the one-dimensional radix-2 FFT developed earlier. To illustrate the role of the transform in scientific simulation, the program also computes a spectral derivative using the frequency-domain multiplication rule introduced in Equation (12.3.16) and compares the numerical result with the analytic derivative of a known periodic function.

At the core of the implementation are the functions `fft2d_forward` and `fft2d_inverse`, which compute the forward and inverse two-dimensional discrete Fourier transforms by composing one-dimensional FFTs along each coordinate direction. Rather than implementing a completely new algorithm, the program exploits the tensor-product structure described in Equation (12.3.14). The forward transform is performed by first applying one-dimensional FFTs along each row of the data array, followed by FFTs along the column direction. Because the one-dimensional FFT operates on contiguous memory segments, the column transforms are implemented by explicitly transposing the array so that columns become rows, applying the FFT again, and then transposing the data back to its original layout.

The function `transpose` performs this data reorganization. Given an array stored in row-major order with dimensions $N_1 \times N_2$, it produces the transposed array with dimensions $N_2 \times N_1$. This operation reflects the memory-access considerations discussed in Section 12.3.3: efficient multi-dimensional FFT implementations must reorganize data so that one-dimensional transforms operate on contiguous memory regions. In large-scale distributed simulations these transpose operations correspond to global communication steps.

The program reuses the one-dimensional FFT routines developed earlier. The functions `fft1d_in_place` and `ifft1d_in_place` implement the radix-2 Cooley–Tukey algorithm described in Section 12.3.1, including bit-reversal ordering and iterative butterfly stages. These routines provide the computational building blocks for the multidimensional transform, illustrating how higher-dimensional FFTs arise as structured compositions of one-dimensional transforms.

To demonstrate a practical application, the program computes a spectral derivative in the $x$-direction using the Fourier differentiation rule given in Equation (12.3.16). After computing the forward transform $\hat{u}(k,\ell)$, each Fourier coefficient is multiplied by the corresponding frequency factor $\frac{2\pi i k}{L_x}$. This operation is implemented inside the loop that constructs `ux_hat`. Because discrete Fourier coefficients correspond to both positive and negative frequencies, the helper function `k_index` converts the FFT bin index into its signed frequency representation, consistent with the frequency interpretation discussed earlier in Section 12.2.

The `main` function organizes the numerical experiment. It first constructs a periodic test function sampled on a uniform two-dimensional grid. The analytic derivative of this function is known, allowing the program to evaluate the accuracy of the spectral differentiation procedure. The sampled data are transformed to Fourier space using the two-dimensional FFT, multiplied by the appropriate frequency factors, and then transformed back to physical space using the inverse FFT. Finally, the program compares the numerical derivative with the exact derivative and reports the maximum error and the magnitude of any residual imaginary component introduced by floating-point roundoff.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
```

```rust
// Program 12.3.2: Two-Dimensional FFT via Tensor-Product Structure and Spectral Differentiation
//
// Problem statement (Section 12.3.3):
// Implement the 2D discrete Fourier transform using the tensor-product decomposition
// implied by the kernel factorization (Eqs. 12.3.12–12.3.14):
//   1) Apply 1D FFTs along rows,
//   2) Apply 1D FFTs along columns (implemented via an explicit transpose),
// so that the total cost scales like O(N1 N2 (log N1 + log N2)) (Eq. 12.3.15).
//
// Demonstrate a pseudo-spectral derivative calculation (Eq. 12.3.16):
//   - Start with a periodic test function u(x,y) sampled on an N1 x N2 grid,
//   - Compute û(k,ℓ) by a forward 2D FFT,
//   - Form the spectral derivative û_x = (2π i k / Lx) û,
//   - Inverse 2D FFT back to physical space,
//   - Compare against the analytic derivative and report max error.
//
// Dependencies (add to Cargo.toml):
// [dependencies]
// num-complex = "0.4"

use num_complex::Complex64;
use std::f64::consts::PI;

fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

fn bit_reverse(mut x: usize, bits: u32) -> usize {
    let mut y = 0usize;
    for _ in 0..bits {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}

fn bit_reversal_permute(a: &mut [Complex64]) {
    let n = a.len();
    let bits = (n as f64).log2() as u32;
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if j > i {
            a.swap(i, j);
        }
    }
}

fn fft1d_in_place(a: &mut [Complex64]) {
    // Iterative radix-2 Cooley–Tukey FFT (forward), as in Program 12.3.1.
    let n = a.len();
    assert!(is_power_of_two(n), "fft1d_in_place requires power-of-two length.");

    bit_reversal_permute(a);

    let mut m = 2usize;
    while m <= n {
        let half = m / 2;
        let theta = -2.0 * PI / (m as f64);
        let w_m = Complex64::new(theta.cos(), theta.sin());

        for k0 in (0..n).step_by(m) {
            let mut w = Complex64::new(1.0, 0.0);
            for j in 0..half {
                let u = a[k0 + j];
                let v = w * a[k0 + j + half];
                a[k0 + j] = u + v;
                a[k0 + j + half] = u - v;
                w *= w_m;
            }
        }

        m *= 2;
    }
}

fn ifft1d_in_place(a: &mut [Complex64]) {
    // Inverse via conjugation identity and 1/N scaling.
    let n = a.len() as f64;

    for z in a.iter_mut() {
        *z = z.conj();
    }

    fft1d_in_place(a);

    for z in a.iter_mut() {
        *z = z.conj() / n;
    }
}

fn transpose(n1: usize, n2: usize, a: &[Complex64]) -> Vec<Complex64> {
    // Input is row-major a[m,n] with shape (n1,n2).
    // Output is row-major b[n,m] with shape (n2,n1).
    assert_eq!(a.len(), n1 * n2);
    let mut b = vec![Complex64::new(0.0, 0.0); n1 * n2];
    for i in 0..n1 {
        for j in 0..n2 {
            b[j * n1 + i] = a[i * n2 + j];
        }
    }
    b
}

fn fft2d_forward(n1: usize, n2: usize, a: &mut [Complex64]) {
    // 2D FFT via row FFTs then column FFTs (implemented through transpose).
    assert_eq!(a.len(), n1 * n2);
    assert!(is_power_of_two(n1) && is_power_of_two(n2), "n1 and n2 must be powers of two.");

    // Row-wise FFTs.
    for i in 0..n1 {
        let row = &mut a[i * n2..(i + 1) * n2];
        fft1d_in_place(row);
    }

    // Column-wise FFTs implemented as row-wise on the transpose.
    let mut at = transpose(n1, n2, a); // shape (n2, n1)
    for j in 0..n2 {
        let row = &mut at[j * n1..(j + 1) * n1];
        fft1d_in_place(row);
    }

    // Transpose back.
    let back = transpose(n2, n1, &at);
    a.copy_from_slice(&back);
}

fn fft2d_inverse(n1: usize, n2: usize, a: &mut [Complex64]) {
    // Inverse 2D FFT: inverse along each dimension.
    assert_eq!(a.len(), n1 * n2);
    assert!(is_power_of_two(n1) && is_power_of_two(n2), "n1 and n2 must be powers of two.");

    // Row-wise inverse FFTs.
    for i in 0..n1 {
        let row = &mut a[i * n2..(i + 1) * n2];
        ifft1d_in_place(row);
    }

    // Column-wise inverse FFTs via transpose.
    let mut at = transpose(n1, n2, a); // shape (n2, n1)
    for j in 0..n2 {
        let row = &mut at[j * n1..(j + 1) * n1];
        ifft1d_in_place(row);
    }

    let back = transpose(n2, n1, &at);
    a.copy_from_slice(&back);
}

fn k_index(k: usize, n: usize) -> i32 {
    // Signed frequency index corresponding to bin k (Eqs. 12.2.8–12.2.9 style).
    let half = n / 2;
    if k <= half {
        k as i32
    } else {
        (k as i32) - (n as i32)
    }
}

fn max_abs_diff_real(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut mx = 0.0_f64;
    for i in 0..a.len() {
        mx = mx.max((a[i] - b[i]).abs());
    }
    mx
}

fn main() {
    // Grid sizes (powers of two for radix-2 FFTs).
    let n1: usize = 64;
    let n2: usize = 64;

    // Domain lengths for Eq. (12.3.16).
    let lx: f64 = 2.0 * PI;
    let ly: f64 = 2.0 * PI;

    let dx: f64 = lx / (n1 as f64);
    let dy: f64 = ly / (n2 as f64);

    // Choose a smooth periodic test function with known derivatives:
    // u(x,y) = sin(ax x) cos(by y)
    // ∂_x u = ax cos(ax x) cos(by y)
    let ax_mode: i32 = 5;
    let by_mode: i32 = 7;
    let ax = (ax_mode as f64) * (2.0 * PI / lx);
    let by = (by_mode as f64) * (2.0 * PI / ly);

    // Sample u(x,y) on the grid into row-major storage.
    let mut u = vec![Complex64::new(0.0, 0.0); n1 * n2];
    let mut ux_exact = vec![0.0_f64; n1 * n2];

    for i in 0..n1 {
        let x = (i as f64) * dx;
        for j in 0..n2 {
            let y = (j as f64) * dy;
            let val = (ax * x).sin() * (by * y).cos();
            let dval = ax * (ax * x).cos() * (by * y).cos();
            u[i * n2 + j] = Complex64::new(val, 0.0);
            ux_exact[i * n2 + j] = dval;
        }
    }

    // Forward 2D FFT to get û(k,ℓ).
    let mut u_hat = u.clone();
    fft2d_forward(n1, n2, &mut u_hat);

    // Spectral differentiation in x (Eq. 12.3.16):
    // û_x(k,ℓ) = (2π i k / Lx) û(k,ℓ), where k is the signed frequency index.
    let mut ux_hat = u_hat.clone();
    for k in 0..n1 {
        let k_signed = k_index(k, n1) as f64;
        let factor = Complex64::new(0.0, 2.0 * PI * k_signed / lx); // (2π i k / Lx)
        for ell in 0..n2 {
            ux_hat[k * n2 + ell] *= factor;
        }
    }

    // Inverse 2D FFT back to physical space.
    fft2d_inverse(n1, n2, &mut ux_hat);

    // Compare the real part to analytic ∂_x u.
    let ux_num: Vec<f64> = ux_hat.iter().map(|z| z.re).collect();
    let max_err = max_abs_diff_real(&ux_num, &ux_exact);

    // A small sanity check: imaginary part should be near 0 for this real-valued derivative.
    let mut max_im = 0.0_f64;
    for z in ux_hat.iter() {
        max_im = max_im.max(z.im.abs());
    }

    println!("Grid: N1 x N2 = {} x {}", n1, n2);
    println!("Domain: Lx = {:.6e}, Ly = {:.6e}", lx, ly);
    println!("Test function: u(x,y) = sin(a x) cos(b y) with modes a={}, b={}", ax_mode, by_mode);
    println!("Spectral derivative using Eq. (12.3.16):");
    println!("Max error ||Re(u_x_num) - u_x_exact||_inf: {:.3e}", max_err);
    println!("Max imaginary magnitude in u_x_num: {:.3e}", max_im);
}
```

Program 12.3.2 demonstrates how the tensor-product structure of the multi-dimensional Fourier transform enables efficient computation using one-dimensional FFT building blocks. Instead of evaluating the double summation in Equation (12.3.12) directly, the algorithm exploits the separability of the exponential kernel described in Equation (12.3.13) to perform the transform as a sequence of row and column FFTs. This factorization reduces the computational complexity to the quasi-linear scaling given in Equation (12.3.15), making high-resolution multidimensional simulations computationally feasible.

The spectral differentiation example illustrates why multi-dimensional FFTs are central to modern pseudo-spectral simulation methods. By transforming a function to Fourier space, spatial derivatives become simple multiplicative operations applied to each Fourier coefficient, as expressed in Equation (12.3.16). After the inverse transform returns the result to physical space, highly accurate derivatives are obtained with errors approaching machine precision for smooth periodic functions.

This experiment highlights the broader role of FFTs as computational kernels in scientific computing. Many large-scale simulations repeatedly alternate between physical and frequency space to evaluate nonlinear terms and differential operators efficiently. The efficiency of these algorithms therefore depends directly on the performance and scalability of multi-dimensional FFT implementations.

## 12.3.4. Precision and Accuracy Considerations

Although the FFT is algebraically exact in infinite precision arithmetic, practical implementations operate in finite-precision floating-point environments. The resulting errors do not originate from the transform itself as a mathematical object, but from representation, discretization, and rounding. Two distinct accuracy issues must therefore be separated conceptually.

First, there is *modeling error*, discussed in Section 12.2. This includes sampling error, aliasing, truncation of infinite domains, and windowing effects. These errors arise before any arithmetic is performed. They reflect the difference between the mathematical Fourier transform of a continuous function and the discrete transform of a finite set of samples. No increase in floating-point precision can eliminate modeling error; it is controlled instead by grid resolution, anti-aliasing filters, and domain truncation strategies.

Second, there is *roundoff error*, which arises from finite-precision arithmetic during the execution of the FFT algorithm. If unit roundoff is denoted by $\varepsilon$, backward error analyses show that a well-implemented FFT of size $N$ produces a result that is the exact transform of slightly perturbed data, with relative perturbation typically bounded by a modest multiple of $\varepsilon \log N$. The logarithmic growth reflects the layered structure of the Cooley–Tukey factorization: each butterfly stage introduces small rounding effects that accumulate across $\mathcal{O}(\log N)$ levels.

The detailed behavior depends on algorithmic structure. In-place transforms, scaling conventions, radix choices, and twiddle-factor evaluation strategies influence both error propagation and numerical stability. Poorly scaled implementations may amplify high-frequency components or incur loss of significance when subtractive cancellation occurs within butterfly operations. Careful normalization and use of precomputed twiddle tables can reduce sensitivity.

In multi-dimensional FFTs, roundoff effects may be compounded by repeated forward and inverse transforms within iterative workflows. For example, in pseudo-spectral time stepping, each timestep alternates between physical and frequency domains. If a simulation performs thousands of transforms, even small per-transform perturbations may accumulate. Stability analysis must therefore consider the transform as part of a larger numerical pipeline rather than as an isolated primitive.

Conditioning also plays a role. The discrete Fourier transform matrix is unitary under appropriate scaling, so its condition number is one in the 2-norm. This favorable conditioning implies that the transform itself does not amplify relative perturbations. Any observed instability is therefore typically attributable to surrounding operations, such as ill-conditioned filtering, nonlinear products, or poorly scaled post-processing steps.

Recent work has shifted attention from fixed-precision execution to *adaptive precision selection*. Runtime systems can now choose among FFT backends and precision formats based on predictive error models that estimate accumulated floating-point error relative to user-prescribed tolerances (Lehner et al., 2025). Such strategies recognize that many simulations are over-resolved in arithmetic precision relative to their modeling accuracy. By matching floating-point precision to required accuracy rather than defaulting to double precision, significant performance gains can be realized.

Mixed-precision FFT-based pipelines exemplify this trend. For instance, transforms may be performed in single precision while residual corrections, normalization steps, or sensitive frequency bands are handled in double precision. Error estimators monitor deviation from expected invariants, such as energy conservation or Parseval consistency, and trigger refinement only when necessary. Studies demonstrate that substantial performance improvements can be achieved without violating prescribed error tolerances when precision is managed systematically (Venkat et al., 2025).

From a workflow perspective, the FFT is therefore increasingly treated as a *tunable computational kernel* embedded within end-to-end simulation frameworks. Its precision, backend implementation, memory layout, and communication strategy are selected dynamically to satisfy global accuracy requirements while minimizing computational cost. In high-performance scientific computing, correctness is no longer judged at the level of a single transform, but by the stability and fidelity of the entire numerical pipeline in which the transform participates.

## 12.3.5. Section Remarks

Section 12.3 establishes three central insights:

1. The FFT reduces DFT cost from quadratic to quasi-linear complexity through recursive factorization.
2. In modern architectures, memory movement and communication dominate performance considerations (Ibrahim and Aga, 2024; Yeung et al., 2024).
3. Precision management and mixed-precision tuning are integral to large-scale FFT-based algorithms (Lehner et al., 2025; Venkat et al., 2025).

Together with Sections 12.1 and 12.2, this completes the conceptual arc: Fourier modes diagonalize translation-invariant models; sampling connects continuous theory to discrete data; and the FFT provides the efficient computational engine that makes these structural simplifications practical in contemporary scientific computing.

# 12.4. FFT of Real Functions, Trigonometric Transforms, and Multidimensional FFTs

This chapter segment continues the FFT story with three closely related themes: how to exploit real-valued input structure rather than wasting work on complex arithmetic, how sine and cosine transforms arise naturally from boundary conditions and can be computed with FFT-speed algorithms, and how FFTs generalize cleanly to two or more dimensions, where data layout and communication increasingly dominate performance. These topics are not special cases. They are workhorses in PDE solvers, imaging, inverse problems, and large-scale simulations (Namugwanya et al., 2023; Pei and Tong, 2025; Risthaus and Schneider, 2024).

A unifying viewpoint is linear algebra. The DFT is a dense matrix map, and each structural variant in this section corresponds to a symmetry or separability property that restricts the transform to an invariant subspace. Real inputs, even and odd extensions, and tensor-product grids all induce exploitable structure in the Fourier matrix and its higher-dimensional analogues (Koopman and Bisseling, 2023; Bielak et al., 2024).

## 12.4.1. The DFT as a Matrix Map and the Role of Structure

Let $N \in \mathbb{N}$ and define the primitive root of unity,

$$\omega_N = e^{-2\pi i/N} \tag{12.4.1}$$

For $x \in \mathbb{C}^N$, the (forward) DFT is the linear map $F : \mathbb{C}^N \to \mathbb{C}^N$ given componentwise by:

$$
X_k = \sum_{n=0}^{N-1} x_n\, \omega_N^{nk}, \qquad
k = 0,1,\dots,N-1
\tag{12.4.2}
$$

Equivalently, $X = W_N x$, where $W_N \in \mathbb{C}^{N\times N}$ is the dense Fourier matrix:

$$
W_N =
\begin{pmatrix}
1 & 1 & 1 & \cdots & 1 \\
1 & \omega_N & \omega_N^2 & \cdots & \omega_N^{N-1} \\
1 & \omega_N^2 & \omega_N^4 & \cdots & \omega_N^{2(N-1)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega_N^{N-1} & \omega_N^{2(N-1)} & \cdots & \omega_N^{(N-1)(N-1)}
\end{pmatrix}
\tag{12.4.3}
$$

The FFT is an algorithmic family for applying $W_N$ (and $W_N^{-1}$) in $O(N\log N)$ time rather than $O(N^2)$, by exploiting algebraic factorization and symmetry (Salih and Hamood, 2023; Gilan and Maham, 2024). In what follows, the central idea is that structural constraints, such as real-valuedness or even and odd symmetry, correspond to redundancy in the full complex spectrum. FFT variants remove that redundancy to save both arithmetic and memory traffic (Koopman and Bisseling, 2023; Liu et al., 2025).

## 12.4.2. FFT of Real-Valued Data and Hermitian Symmetry

Section 12.4 addresses the common case $x = f \in \mathbb{R}^N$. A naive approach is to embed $f$ into $\mathbb{C}^N$ with zero imaginary part and call a complex FFT. This works, but it performs unnecessary complex arithmetic and stores redundant frequency information. Dedicated real-valued FFT algorithms and data layouts reduce both cost and memory and are widely used in practice (Salih and Hamood, 2023; Liu et al., 2025).

Let $f \in \mathbb{R}^N$ and define its DFT $F \in \mathbb{C}^N$ by (12.4.2). Because $f_n = \overline{f_n}$, one obtains Hermitian symmetry:

$$
F_{N-k} = \overline{F_k}, \qquad
k = 0,1,\dots,N-1
\tag{12.4.4}
$$

with indices understood modulo $N$. When $N$ is even, both $F_0$ and $F_{N/2}$ are real. Thus the spectrum contains only $N$ real degrees of freedom, exactly matching the time-domain data, even though it is represented in $\mathbb{C}^N$ (Salih and Hamood, 2023; Liu et al., 2025). A useful picture is to regard the spectrum as a mirror about $k=0$ and $k=N/2$. This motivates storage formats that keep only nonnegative frequencies $0 \le k \le N/2$, since the rest is redundant.

A second common optimization is the “two real transforms for the price of one complex FFT.” Suppose $f,g \in \mathbb{R}^N$ and define a packed complex sequence as:

$$h_n = f_n + i g_n \tag{12.4.5}$$

Let $H = \mathrm{DFT}(h)$. Then the separate transforms $F = \mathrm{DFT}(f)$ and $G = \mathrm{DFT}(g)$ can be recovered from,

\begin{equation}
\left\{
\begin{aligned}
F_k &= \frac{1}{2}\left(H_k + \overline{H_{N-k}}\right), \\
G_k &= \frac{1}{2i}\left(H_k - \overline{H_{N-k}}\right),
\end{aligned}
\right.
\qquad k \in \{0,1,\dots,N-1\}
\tag{12.4.6}
\end{equation}

with the usual modular indexing conventions. This trick remains practical when applications naturally require transforms of two real arrays, for example in correlations or coupled PDE fields, and it reflects the same conjugate-symmetry reasoning used in modern real-valued FFT designs (Salih and Hamood, 2023; Liu et al., 2025).

A numerical caution is worth stating explicitly. The identities above are algebraically exact, but in floating-point arithmetic they may lose relative accuracy if $|f|$ and $|g|$ differ by many orders of magnitude, because both signals share the same complex FFT pathway and rounding noise. Mixed-precision error analyses in FFT-based pipelines highlight how rounding contributions can accumulate across stages and scale with problem conditioning and precision choices (Venkat et al., 2025).

A particularly important real-FFT construction uses even–odd packing. Assume $f \in \mathbb{R}^N$ with even $N$. Split into even and odd subsequences,

$$
e_m = f_{2m}, \qquad
o_m = f_{2m+1}, \qquad
m = 0,1,\dots,\frac{N}{2}-1
\tag{12.4.7}
$$

Separating even and odd indices in (12.4.2) gives:

$$
F_k
= \sum_{m=0}^{\frac{N}{2}-1} f_{2m}\,\omega_N^{2mk}
+ \sum_{m=0}^{\frac{N}{2}-1} f_{2m+1}\,\omega_N^{(2m+1)k}
= E_k + \omega_N^{k} O_k
\tag{12.4.8}
$$

where,

$$
E_k = \sum_{m=0}^{\frac{N}{2}-1} e_m \,\omega_{N/2}^{mk}, \qquad
O_k = \sum_{m=0}^{\frac{N}{2}-1} o_m \,\omega_{N/2}^{mk}
\tag{12.4.9}
$$

Thus,

$$F_k = E_k + \omega_N^{k} O_k \tag{12.4.10}$$

The computational insight is that $(E_k,O_k)$ can be obtained from one complex FFT of length $N/2$. Define:

$$
h_m = e_m + i\,o_m, \qquad
m = 0,1,\dots,\frac{N}{2}-1
\tag{12.4.11}
$$

compute $H = \mathrm{DFT}_{N/2}(h)$, and then extract:

\begin{equation}
\left\{
\begin{aligned}
E_k &= \frac{1}{2}\left(H_k + \overline{H_{(N/2 - k)\bmod (N/2)}}\right), \\
O_k &= \frac{1}{2i}\left(H_k - \overline{H_{(N/2 - k)\bmod (N/2)}}\right),
\end{aligned}
\right.
\qquad k = 0,\dots,\frac{N}{2}-1
\tag{12.4.12}
\end{equation}

Substituting (12.4.12) into (12.4.10) yields a complete real-FFT algorithm based on one half-size complex FFT plus $O(N)$ postprocessing. This blueprint explains why real FFTs typically require roughly half the arithmetic and memory traffic of complex FFTs, a motivation emphasized in real-FFT algorithm discussions and hardware evaluations (Salih and Hamood, 2023; Liu et al., 2025).

When $f \in \mathbb{R}^N$ with even $N$, it is sufficient to store,

$$
F_0 \in \mathbb{R}, \qquad
F_{N/2} \in \mathbb{R}, \qquad
F_1,\dots,F_{N/2-1} \in \mathbb{C}
\tag{12.4.13}
$$

A common in-place convention stores the Nyquist component $F_{N/2}$ in the unused imaginary slot of $F_0$, so the output occupies exactly $N$ real words, matching the input size. Space-efficient in-place computation is a central goal in real-valued FFT algorithm design (Salih and Hamood, 2023; Liu et al., 2025).

If a complex FFT of length $N$ costs $O(N\log N)$, then the real-FFT strategy above costs:

$$
O\!\left(\frac{N}{2}\log_2\!\frac{N}{2}\right) + O(N)
= O(N \log N)
\tag{12.4.14}
$$

with a smaller constant factor. In practice the speedup is often close to a factor of two because the reduction in memory traffic can dominate the reduction in arithmetic (Salih and Hamood, 2023; Liu et al., 2025).

### Rust Implementation

Following the discussion in Sections 12.4.1 and 12.4.2 on the structure of the discrete Fourier transform and the redundancy present in real-valued signals, Program 12.4.1 provides a practical Rust implementation of a real-valued FFT based on the even–odd packing construction described in equations (12.4.7)–(12.4.12). The section showed that the DFT defined by equation (12.4.2) can be interpreted as a dense linear map $X = W_N x$ (equation (12.4.3)), but that additional structure arises when the input vector is real. In that case, the Fourier spectrum exhibits Hermitian symmetry (equation (12.4.4)), meaning that half of the frequency components are redundant. The program demonstrates how this structure can be exploited algorithmically. It implements a real-valued FFT using a single half-size complex FFT together with $O(N)$ post-processing, verifies the Hermitian symmetry property numerically, and illustrates how only the non-negative frequency components need to be stored. The implementation also includes the classical “two real transforms for the price of one complex FFT” identity (equations (12.4.5)–(12.4.6)), showing how multiple real signals can share the same complex FFT pathway. In this way the program translates the structural insights of the section into a concrete computational procedure suitable for high-performance numerical computing.

At the core of the implementation is the function `real_fft_even_odd`, which directly encodes the even–odd decomposition introduced in equations (12.4.7)–(12.4.10). The input real sequence $f_n$ is first separated into the even and odd subsequences $e_m = f_{2m}$ and $o_m = f_{2m+1}$. These two sequences are then packed into a complex vector $h_m = e_m + i\,o_m$, as described in equation (12.4.11). Computing the discrete Fourier transform of this packed sequence produces a complex spectrum $H_k$ of length $N/2$. Using the algebraic relations in equation (12.4.12), the program reconstructs the transforms $E_k$ and $O_k$ associated with the even and odd components. These are then combined with the twiddle factor $\omega_N^k$ according to equation (12.4.10) to obtain the final Fourier coefficients $F_k$. This construction demonstrates how a real-valued FFT can be obtained using one complex FFT of half the size together with simple post-processing operations.

The helper function `fft_forward` performs the complex Fourier transform corresponding to equation (12.4.2). Internally it uses the `rustfft` library, which provides an optimized FFT implementation with $O(N\log N)$ complexity. The function `fft_inverse` performs the corresponding inverse transform and normalizes the result by $1/N$. These functions serve as the computational realization of the matrix map interpretation $X = W_N x$ introduced in Section 12.4.1.

To verify the structural properties derived in the section, the program includes the function `hermitian_symmetry_error`. This method evaluates the deviation from the conjugate symmetry identity given in equation (12.4.4). Because the input signal is real, the spectrum must satisfy $F_{N-k}=\overline{F_k}$ up to floating-point roundoff. The routine computes the maximum deviation from this identity and prints it as a diagnostic quantity.

The functions `extract_half_spectrum` and `reconstruct_full_from_half` illustrate the storage layout discussed in equation (12.4.13). Instead of storing the entire complex spectrum, the program keeps only the non-negative frequencies consisting of the DC component (F_0), the Nyquist component $F_{N/2}$, and the interior coefficients $F_1,\dots,F_{N/2-1}$. The remaining coefficients are recovered by conjugate symmetry when needed. This demonstrates how real-valued FFT algorithms reduce memory usage while preserving all spectral information.

The program also implements the packed-signal identity described in equations (12.4.5)–(12.4.6). The function `two_real_ffts_via_one_complex_fft` combines two real sequences $f$ and $g$ into a single complex sequence $h_n = f_n + i g_n$, computes its Fourier transform, and then separates the two resulting spectra algebraically. This technique is useful in applications where multiple real signals must be transformed simultaneously.

Finally, the `main` function serves as a numerical experiment illustrating the theoretical results developed in Section 12.4.2. It constructs a synthetic signal containing several Fourier modes, computes its spectrum using both the reference complex FFT and the structured real-FFT algorithm, and verifies that the two approaches agree to machine precision. The function then demonstrates half-spectrum storage, reconstruction of the original signal using the inverse transform, and the simultaneous transformation of two real signals using the packed-FFT method. These tests provide empirical confirmation of the symmetry and efficiency properties derived earlier in the section.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
rustfft = "6"
```

```rust
// Program 12.4.1: Real-Valued FFT via Even-Odd Packing
//
// This program implements the real-FFT construction described in
// equations (12.4.7) to (12.4.12). It also verifies Hermitian symmetry
// from (12.4.4), extracts the nonredundant half-spectrum in the sense of
// (12.4.13), and demonstrates the "two real transforms for the price of
// one complex FFT" identity from (12.4.6).
//
// Dependencies:
//   num-complex = "0.4"
//   rustfft     = "6"
//
// To run:
//   cargo run

use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Convenient alias for complex numbers in double precision.
type C64 = Complex<f64>;

/// Stores the nonredundant half-spectrum for a real input of even length N.
/// According to (12.4.13), one keeps
///   F_0      in R,
///   F_{N/2}  in R,
///   F_1..F_{N/2-1} in C.
#[derive(Debug, Clone)]
struct HalfSpectrum {
    dc: f64,
    interior: Vec<C64>,
    nyquist: f64,
}

/// Computes the forward complex FFT of a vector.
/// This applies the same sign convention as equation (12.4.2):
/// X_k = sum_n x_n * exp(-2*pi*i*n*k/N).
fn fft_forward(input: &[C64]) -> Vec<C64> {
    let n = input.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer = input.to_vec();
    fft.process(&mut buffer);
    buffer
}

/// Computes the inverse complex FFT with normalization by 1/N.
fn fft_inverse(input: &[C64]) -> Vec<C64> {
    let n = input.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_inverse(n);
    let mut buffer = input.to_vec();
    fft.process(&mut buffer);
    let scale = 1.0 / n as f64;
    for z in &mut buffer {
        *z *= scale;
    }
    buffer
}

/// Converts a real slice into a complex vector with zero imaginary part.
fn real_to_complex(x: &[f64]) -> Vec<C64> {
    x.iter().map(|&v| C64::new(v, 0.0)).collect()
}

/// Computes the full complex FFT of a real signal by the naive embedding
/// f -> f + i*0. This is used as a reference for validation.
fn full_fft_of_real_reference(x: &[f64]) -> Vec<C64> {
    fft_forward(&real_to_complex(x))
}

/// Computes exp(-2*pi*i*k/n), the twiddle factor omega_n^k from (12.4.1).
fn omega(n: usize, k: usize) -> C64 {
    let theta = -2.0 * PI * (k as f64) / (n as f64);
    C64::new(theta.cos(), theta.sin())
}

/// Real FFT via even-odd packing, based on equations (12.4.7) to (12.4.12).
///
/// For even N:
///   e_m = f_{2m}, o_m = f_{2m+1}
///   h_m = e_m + i o_m
///   H = FFT_{N/2}(h)
/// then recover E_k and O_k using (12.4.12), and finally
///   F_k = E_k + omega_N^k O_k  from (12.4.10).
///
/// The full complex spectrum of length N is returned.
fn real_fft_even_odd(x: &[f64]) -> Result<Vec<C64>, String> {
    let n = x.len();
    if n == 0 {
        return Err("Input length must be positive.".to_string());
    }
    if n % 2 != 0 {
        return Err("Even-odd real FFT requires an even input length.".to_string());
    }

    let m = n / 2;

    // Build h_m = e_m + i o_m from (12.4.11).
    let mut h = Vec::with_capacity(m);
    for j in 0..m {
        let e_j = x[2 * j];
        let o_j = x[2 * j + 1];
        h.push(C64::new(e_j, o_j));
    }

    // Compute H = DFT_{N/2}(h).
    let h_hat = fft_forward(&h);

    // Recover E_k and O_k from (12.4.12), then F_k from (12.4.10).
    let mut f_hat = vec![C64::new(0.0, 0.0); n];
    let i_unit = C64::new(0.0, 1.0);

    for k in 0..m {
        let km = (m - k) % m;
        let hk = h_hat[k];
        let hmirror = h_hat[km].conj();

        let e_k = 0.5 * (hk + hmirror);
        let o_k = (hk - hmirror) / (2.0 * i_unit);

        let twiddle = omega(n, k);
        let fk = e_k + twiddle * o_k;
        let fk_m = e_k - twiddle * o_k; // equals F_{k + N/2}

        f_hat[k] = fk;
        f_hat[k + m] = fk_m;
    }

    Ok(f_hat)
}

/// Extracts the half-spectrum described by equation (12.4.13).
fn extract_half_spectrum(f_hat: &[C64]) -> Result<HalfSpectrum, String> {
    let n = f_hat.len();
    if n == 0 || n % 2 != 0 {
        return Err("Half-spectrum extraction requires a nonzero even length.".to_string());
    }

    let dc = f_hat[0].re;
    let nyquist = f_hat[n / 2].re;
    let interior = f_hat[1..n / 2].to_vec();

    Ok(HalfSpectrum {
        dc,
        interior,
        nyquist,
    })
}

/// Reconstructs the full Hermitian-symmetric spectrum from the stored
/// half-spectrum of a real signal.
fn reconstruct_full_from_half(half: &HalfSpectrum) -> Vec<C64> {
    let n = 2 * (half.interior.len() + 1);
    let mut full = vec![C64::new(0.0, 0.0); n];

    full[0] = C64::new(half.dc, 0.0);
    full[n / 2] = C64::new(half.nyquist, 0.0);

    for (j, &z) in half.interior.iter().enumerate() {
        let k = j + 1;
        full[k] = z;
        full[n - k] = z.conj();
    }

    full
}

/// Verifies Hermitian symmetry from equation (12.4.4):
/// F_{N-k} = conjugate(F_k), with modular indexing.
/// Returns the maximum absolute violation.
fn hermitian_symmetry_error(f_hat: &[C64]) -> f64 {
    let n = f_hat.len();
    let mut max_err: f64 = 0.0;

    for k in 0..n {
        let lhs = f_hat[(n - k) % n];
        let rhs = f_hat[k].conj();
        let err = (lhs - rhs).norm();
        max_err = max_err.max(err);
    }

    max_err
}

/// Maximum absolute difference between two complex vectors.
fn max_complex_error(a: &[C64], b: &[C64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0_f64, f64::max)
}

/// Packs two real signals f and g into h_n = f_n + i g_n as in (12.4.5),
/// computes H = DFT(h), and recovers the separate transforms F and G
/// using equation (12.4.6).
fn two_real_ffts_via_one_complex_fft(f: &[f64], g: &[f64]) -> Result<(Vec<C64>, Vec<C64>), String> {
    if f.len() != g.len() {
        return Err("Signals f and g must have the same length.".to_string());
    }
    if f.is_empty() {
        return Err("Signals must be nonempty.".to_string());
    }

    let n = f.len();
    let packed: Vec<C64> = f
        .iter()
        .zip(g.iter())
        .map(|(&fr, &gr)| C64::new(fr, gr))
        .collect();

    let h_hat = fft_forward(&packed);
    let i_unit = C64::new(0.0, 1.0);

    let mut f_hat = vec![C64::new(0.0, 0.0); n];
    let mut g_hat = vec![C64::new(0.0, 0.0); n];

    for k in 0..n {
        let hk = h_hat[k];
        let hmirror = h_hat[(n - k) % n].conj();

        f_hat[k] = 0.5 * (hk + hmirror);
        g_hat[k] = (hk - hmirror) / (2.0 * i_unit);
    }

    Ok((f_hat, g_hat))
}

/// Prints a complex vector in a compact table.
fn print_spectrum(label: &str, data: &[C64]) {
    println!("{label}");
    println!("{:>4} {:>18} {:>18}", "k", "Re", "Im");
    for (k, z) in data.iter().enumerate() {
        println!("{:>4} {:>18.10e} {:>18.10e}", k, z.re, z.im);
    }
    println!();
}

/// Prints the half-spectrum layout corresponding to (12.4.13).
fn print_half_spectrum(half: &HalfSpectrum) {
    println!("Half-spectrum storage layout for an even-length real signal:");
    println!("  F_0       = {:>18.10e}", half.dc);
    for (j, z) in half.interior.iter().enumerate() {
        println!(
            "  F_{:<3}   = {:>18.10e} + i {:>18.10e}",
            j + 1,
            z.re,
            z.im
        );
    }
    println!("  F_N/2     = {:>18.10e}", half.nyquist);
    println!();
}

/// Reconstructs a real signal from its full complex spectrum.
/// For a real-input transform, tiny imaginary parts after inversion are
/// interpreted as roundoff.
fn reconstruct_real_signal(f_hat: &[C64]) -> Vec<f64> {
    fft_inverse(f_hat).into_iter().map(|z| z.re).collect()
}

/// Maximum absolute error between two real vectors.
fn max_real_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn main() -> Result<(), String> {
    // Example 1: a real signal of even length.
    // This combines a few Fourier modes so the spectrum is nontrivial.
    let n = 16usize;
    let signal: Vec<f64> = (0..n)
        .map(|j| {
            let t = 2.0 * PI * (j as f64) / (n as f64);
            1.25
                + 0.75 * (2.0 * t).cos()
                - 0.50 * (3.0 * t).sin()
                + 0.20 * (5.0 * t).cos()
        })
        .collect();

    println!("Program 12.4.1: Real-Valued FFT via Even-Odd Packing");
    println!("Input length N = {}", n);
    println!();

    // Reference result: full complex FFT of the embedded real signal.
    let reference = full_fft_of_real_reference(&signal);

    // Structured real FFT from equations (12.4.7) to (12.4.12).
    let real_fft = real_fft_even_odd(&signal)?;

    // Compare the two approaches.
    let fft_agreement = max_complex_error(&reference, &real_fft);
    println!("Maximum difference between reference FFT and structured real FFT:");
    println!("  max_k |F_ref[k] - F_struct[k]| = {:.6e}", fft_agreement);
    println!();

    // Verify Hermitian symmetry from (12.4.4).
    let hermitian_err = hermitian_symmetry_error(&real_fft);
    println!("Hermitian symmetry verification for a real signal:");
    println!("  max_k |F[N-k] - conj(F[k])| = {:.6e}", hermitian_err);
    println!();

    // Show the full spectrum.
    print_spectrum("Full spectrum F_k:", &real_fft);

    // Extract and display the half-spectrum of (12.4.13).
    let half = extract_half_spectrum(&real_fft)?;
    print_half_spectrum(&half);

    // Reconstruct the full spectrum from the half-spectrum and compare.
    let rebuilt_spectrum = reconstruct_full_from_half(&half);
    let spectrum_rebuild_err = max_complex_error(&real_fft, &rebuilt_spectrum);
    println!("Half-spectrum reconstruction check:");
    println!(
        "  max_k |F[k] - F_rebuilt[k]| = {:.6e}",
        spectrum_rebuild_err
    );
    println!();

    // Reconstruct the original real signal through inverse FFT.
    let reconstructed_signal = reconstruct_real_signal(&rebuilt_spectrum);
    let signal_reconstruction_err = max_real_error(&signal, &reconstructed_signal);
    println!("Signal reconstruction check:");
    println!(
        "  max_n |f[n] - f_reconstructed[n]| = {:.6e}",
        signal_reconstruction_err
    );
    println!();

    // Example 2: two real transforms for the price of one complex FFT.
    let f: Vec<f64> = (0..n)
        .map(|j| {
            let t = 2.0 * PI * (j as f64) / (n as f64);
            (2.0 * t).cos() + 0.3 * (4.0 * t).sin()
        })
        .collect();

    let g: Vec<f64> = (0..n)
        .map(|j| {
            let t = 2.0 * PI * (j as f64) / (n as f64);
            -0.5 + 1.1 * t.sin() - 0.2 * (3.0 * t).cos()
        })
        .collect();

    let (f_hat_packed, g_hat_packed) = two_real_ffts_via_one_complex_fft(&f, &g)?;
    let f_hat_ref = full_fft_of_real_reference(&f);
    let g_hat_ref = full_fft_of_real_reference(&g);

    let f_err = max_complex_error(&f_hat_packed, &f_hat_ref);
    let g_err = max_complex_error(&g_hat_packed, &g_hat_ref);

    println!("Two-real-signals packing check from equation (12.4.6):");
    println!("  max_k |F_packed[k] - F_ref[k]| = {:.6e}", f_err);
    println!("  max_k |G_packed[k] - G_ref[k]| = {:.6e}", g_err);
    println!();

    Ok(())
}
```

Program 12.4.1 illustrates how the structural properties of the discrete Fourier transform translate into practical numerical algorithms. While the DFT defined in equation (12.4.2) can be interpreted as multiplication by the dense Fourier matrix $W_N$ (equation (12.4.3)), the presence of real-valued input introduces conjugate symmetry in the spectrum as expressed by equation (12.4.4). The program demonstrates how this symmetry can be exploited to reduce computational work and memory usage through the even–odd packing strategy of equations (12.4.7)–(12.4.12).

The numerical diagnostics printed by the program confirm that the structured real-FFT algorithm reproduces the same Fourier coefficients as the direct complex FFT to machine precision. The Hermitian symmetry test verifies the theoretical redundancy of the spectrum, while the half-spectrum reconstruction experiment shows that only half of the frequency components need to be stored explicitly. The additional example implementing equations (12.4.5)–(12.4.6) further demonstrates how multiple real signals can be processed efficiently using a single complex transform.

Together, these experiments highlight the central theme of Section 12.4: FFT algorithms achieve their efficiency not by changing the mathematical transform itself, but by exploiting algebraic structure and symmetry in the input data. The same principle extends to the trigonometric transforms and multidimensional FFT constructions developed in the following sections, where additional symmetry and separability properties lead to further reductions in computational cost.

## 12.4.3. Trigonometric Transforms and Boundary Conditions

Sine and cosine transforms arise constantly in PDEs and boundary-value modeling because they encode symmetry at boundaries. On an interval, eigenfunctions of many symmetric differential operators come in sine and cosine families. Discretely, the same story appears: the discrete Laplacian and related operators have eigenvectors resembling sampled sines or cosines depending on boundary conditions. A useful practical summary is that discrete sine transforms align naturally with Dirichlet boundary conditions, while discrete cosine transforms align naturally with Neumann boundary conditions, a point made explicitly in recent Poisson-solver work (Pei and Tong, 2025).

An intuitive mnemonic is: sine modes vanish at endpoints (odd symmetry), while cosine modes have zero derivative at endpoints (even symmetry). These ideas are used directly as boundary-condition machinery in FFT-based mechanics and homogenization workflows (Risthaus and Schneider, 2024; Paux et al., 2025).

There is not a single DCT and a single DST but families of transforms with different endpoint conventions and phase shifts. A recent overview emphasizes that multiple types exist, reflecting these convention choices (Bielak et al., 2024). In numerical PDE practice on uniform grids, two broad patterns recur: endpoint-including forms (natural when boundary samples are explicitly stored) and half-shifted forms (natural on staggered grids and common in signal and image processing).

From a computational standpoint, the key message is that sine and cosine transforms can be reduced to FFTs by even or odd extensions plus twiddle-factor corrections. This FFT reducibility is why DST-accelerated Poisson solvers are considered FFT-accelerated methods in modern literature (Pei and Tong, 2025; Risthaus and Schneider, 2024). For very small block sizes, which occur in embedded DSP and short-kernel processing, recent work also derives ultra-low-operation-count DST kernels by factoring the transform matrix into sparse structured components, illustrating that “FFT speed” can be achieved either by large-$N$ reductions or by explicit kernel synthesis at tiny $N$ (Bielak et al., 2024; Polyakova et al., 2025).

### Rust Implementation

Following the discussion in Section 12.4.3 on the relationship between boundary symmetry and trigonometric transforms, Program 12.4.2 provides a practical Rust implementation of discrete cosine and sine transforms constructed from FFT-based symmetry extensions. The section explained that cosine and sine transforms arise naturally from even and odd extensions of discrete data and can therefore be computed efficiently by reducing them to standard FFT operations. This program demonstrates that idea directly. It implements a DCT-I and DST-I using even and odd data extensions, verifies the correctness of the transforms through round-trip reconstruction tests, and illustrates their role in numerical partial differential equations by solving one-dimensional Poisson problems with Dirichlet and Neumann boundary conditions. In doing so, the program shows how the structural insights discussed in the section translate into practical computational procedures that exploit symmetry to achieve FFT-level efficiency.

At the core of the implementation are the functions `dct1_forward` and `dst1_forward`, which implement cosine and sine transforms through symmetry-based reductions to the FFT. As discussed in Section 12.4.3, trigonometric transforms can be computed by extending the original data in a symmetric manner and then applying a complex Fourier transform. The function `dct1_forward` constructs an even extension of the input data and evaluates its FFT, after which the cosine coefficients are extracted from the real part of the transformed sequence. This procedure reflects the even symmetry associated with Neumann-type boundary conditions and corresponds to the cosine-basis representation discussed earlier in the section.

The complementary function `dst1_forward` implements the discrete sine transform using an odd extension of the input data. The program constructs the sequence described in the symmetry argument of Section 12.4.3, applies a complex FFT, and extracts the sine coefficients from the imaginary part of the result. Because sine functions vanish at the boundaries, this transform naturally corresponds to the Dirichlet boundary condition setting described in the text. Together, these two functions illustrate the key computational principle of the section: trigonometric transforms are not independent algorithms but structured restrictions of the FFT obtained through symmetry extensions.

To verify correctness, the program also provides inverse transforms through the routines `dct1_inverse` and `dst1_inverse`. The inverse cosine transform reconstructs the original data using the same cosine basis used in the forward transform, while the inverse sine transform takes advantage of the self-inverse structure of the DST-I matrix up to a constant scaling factor. These inverse routines allow the program to perform round-trip tests that confirm the forward and inverse transforms are numerically consistent.

The functions `solve_poisson_dirichlet_dst1` and `solve_poisson_neumann_dct1` demonstrate the practical use of these transforms in solving boundary value problems. The first routine solves a discrete Poisson equation with homogeneous Dirichlet boundary conditions. As discussed in Section 12.4.3, the eigenvectors of the discrete Laplacian with Dirichlet boundaries correspond to sine modes, so the DST diagonalizes the operator. The routine transforms the right-hand side using the DST, divides by the corresponding eigenvalues of the discrete Laplacian, and then reconstructs the solution through the inverse DST.

The second routine performs the analogous computation for homogeneous Neumann boundary conditions using the DCT. In this setting the discrete Laplacian has cosine eigenvectors, so the DCT diagonalizes the operator. Because Neumann problems contain a constant nullspace mode, the routine explicitly enforces the compatibility condition on the right-hand side and fixes the mean mode of the solution.

Several auxiliary routines support these computations. The function `fft_forward` provides the underlying FFT used by both trigonometric transforms, while the diagnostic routines `max_abs_error`, `max_residual_dirichlet`, and `max_residual_neumann` measure reconstruction accuracy and residual errors in the Poisson solutions. These diagnostic tools allow the program to verify that the transform-based solvers produce solutions consistent with the discrete differential operators.

The `main` function assembles these components into four numerical demonstrations. It first performs a round-trip DCT-I experiment, showing that a cosine transform followed by its inverse reconstructs the original data with machine-precision accuracy. It then performs an analogous round-trip test for the DST-I. The program next solves a one-dimensional Poisson equation with Dirichlet boundary conditions using the sine transform, and finally solves the corresponding Neumann problem using the cosine transform. These experiments illustrate how the theoretical symmetry arguments of Section 12.4.3 translate into efficient FFT-based numerical algorithms.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
rustfft = "6"
```

```rust
// Program 12.4.2: Fast DCT-I and DST-I via FFT-Based Symmetry Extensions
//
// This program demonstrates how trigonometric transforms arise from
// boundary symmetry and how they can be implemented through FFT-based
// even and odd extensions.
//
// Included components:
// 1. DCT-I via an even extension and a complex FFT.
// 2. DST-I via an odd extension and a complex FFT.
// 3. Inverse transforms for both cases.
// 4. A 1D Poisson solver with homogeneous Dirichlet boundary conditions
//    using the DST-I.
// 5. A 1D Poisson solver with homogeneous Neumann boundary conditions
//    using the DCT-I.
//
// To run:
//   cargo run

use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

type C64 = Complex<f64>;

fn fft_forward(input: &[C64]) -> Vec<C64> {
    let n = input.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer = input.to_vec();
    fft.process(&mut buffer);
    buffer
}

fn max_abs_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn max_residual_dirichlet(u: &[f64], f: &[f64], h: f64) -> f64 {
    let n = u.len();
    assert_eq!(n, f.len());
    let mut max_r: f64 = 0.0;

    for j in 0..n {
        let u_left = if j == 0 { 0.0 } else { u[j - 1] };
        let u_right = if j + 1 == n { 0.0 } else { u[j + 1] };
        let lhs = (-u_left + 2.0 * u[j] - u_right) / (h * h);
        let r = (lhs - f[j]).abs();
        max_r = max_r.max(r);
    }

    max_r
}

fn max_residual_neumann(u: &[f64], f: &[f64], h: f64) -> f64 {
    let n_plus_1 = u.len();
    assert_eq!(n_plus_1, f.len());

    let n = n_plus_1 - 1;
    let mut max_r: f64 = 0.0;

    // Boundary rows consistent with cosine diagonalization:
    //  2(u_0 - u_1)/h^2 = f_0
    //  2(u_N - u_{N-1})/h^2 = f_N
    let r0 = (2.0 * (u[0] - u[1]) / (h * h) - f[0]).abs();
    max_r = max_r.max(r0);

    for j in 1..n {
        let lhs = (-u[j - 1] + 2.0 * u[j] - u[j + 1]) / (h * h);
        let r = (lhs - f[j]).abs();
        max_r = max_r.max(r);
    }

    let rn = (2.0 * (u[n] - u[n - 1]) / (h * h) - f[n]).abs();
    max_r = max_r.max(rn);

    max_r
}

/// DCT-I of a real vector x of length N+1.
///
/// Mathematical convention:
/// X_k = x_0 + (-1)^k x_N + 2 * sum_{j=1}^{N-1} x_j cos(pi j k / N),
/// for k = 0, ..., N.
///
/// FFT reduction:
/// Build the even extension
/// [x_0, x_1, ..., x_{N-1}, x_N, x_{N-1}, ..., x_1]
/// of length 2N, compute its FFT, and take the real parts.
fn dct1_forward(x: &[f64]) -> Result<Vec<f64>, String> {
    if x.len() < 2 {
        return Err("DCT-I requires at least two samples.".to_string());
    }

    let n = x.len() - 1;
    let mut ext = Vec::with_capacity(2 * n);

    for &v in x {
        ext.push(C64::new(v, 0.0));
    }
    for j in (1..n).rev() {
        ext.push(C64::new(x[j], 0.0));
    }

    let y = fft_forward(&ext);
    Ok(y[..=n].iter().map(|z| z.re).collect())
}

/// Inverse DCT-I under the above normalization.
///
/// For the forward convention
///   X_k = x_0 + (-1)^k x_N + 2 * sum_{j=1}^{N-1} x_j cos(pi j k / N),
/// the inverse is
///   x_j = [X_0 + (-1)^j X_N] / (2N) + (1/N) * sum_{k=1}^{N-1} X_k cos(pi j k / N).
fn dct1_inverse(x_hat: &[f64]) -> Result<Vec<f64>, String> {
    if x_hat.len() < 2 {
        return Err("Inverse DCT-I requires at least two coefficients.".to_string());
    }

    let n = x_hat.len() - 1;
    let mut x = vec![0.0; n + 1];

    for j in 0..=n {
        let mut sum = (x_hat[0] + if j % 2 == 0 { x_hat[n] } else { -x_hat[n] }) / 2.0;
        for k in 1..n {
            let theta = PI * (j as f64) * (k as f64) / (n as f64);
            sum += x_hat[k] * theta.cos();
        }
        x[j] = sum / (n as f64);
    }

    Ok(x)
}

/// DST-I of a real vector x of length n.
///
/// Mathematical convention:
/// X_k = 2 * sum_{j=1}^{n} x_j sin(pi j k / (n+1)),
/// for k = 1, ..., n,
/// where x_j is represented in code as x[j-1].
///
/// FFT reduction:
/// Build the odd extension
/// [0, x_1, ..., x_n, 0, -x_n, ..., -x_1]
/// of length 2(n+1), compute its FFT, and use imaginary parts.
fn dst1_forward(x: &[f64]) -> Result<Vec<f64>, String> {
    if x.is_empty() {
        return Err("DST-I requires a nonempty vector.".to_string());
    }

    let n = x.len();
    let m = n + 1;
    let mut ext = vec![C64::new(0.0, 0.0); 2 * m];

    for j in 1..=n {
        ext[j] = C64::new(x[j - 1], 0.0);
        ext[2 * m - j] = C64::new(-x[j - 1], 0.0);
    }

    let y = fft_forward(&ext);

    let mut out = vec![0.0; n];
    for k in 1..=n {
        out[k - 1] = -y[k].im;
    }

    Ok(out)
}

/// Inverse DST-I under the above normalization.
///
/// Because the DST-I matrix satisfies S^2 = 2(n+1) I under this convention,
/// the inverse is obtained by applying DST-I again and dividing by 2(n+1).
fn dst1_inverse(x_hat: &[f64]) -> Result<Vec<f64>, String> {
    if x_hat.is_empty() {
        return Err("Inverse DST-I requires a nonempty vector.".to_string());
    }

    let n = x_hat.len();
    let y = dst1_forward(x_hat)?;
    let scale = 1.0 / (2.0 * (n + 1) as f64);
    Ok(y.into_iter().map(|v| scale * v).collect())
}

/// Solves the 1D Poisson problem
///   -u''(x) = f(x),  x in (0,1),
///   u(0) = u(1) = 0,
/// on n interior grid points using the DST-I.
///
/// Discrete system:
///   (-u_{j-1} + 2u_j - u_{j+1}) / h^2 = f_j,   j = 1,...,n
/// with u_0 = u_{n+1} = 0.
///
/// The DST-I diagonalizes this operator with eigenvalues
///   lambda_k = 2 - 2 cos(pi k / (n+1)).
fn solve_poisson_dirichlet_dst1(f: &[f64]) -> Result<Vec<f64>, String> {
    let n = f.len();
    if n == 0 {
        return Err("Dirichlet solve requires at least one interior point.".to_string());
    }

    let h = 1.0 / ((n + 1) as f64);
    let f_hat = dst1_forward(f)?;
    let mut u_hat = vec![0.0; n];

    for k in 1..=n {
        let lambda_k = 2.0 - 2.0 * (PI * (k as f64) / ((n + 1) as f64)).cos();
        u_hat[k - 1] = (h * h) * f_hat[k - 1] / lambda_k;
    }

    dst1_inverse(&u_hat)
}

/// Solves the 1D Poisson problem with homogeneous Neumann conditions:
///   -u''(x) = f(x),  x in [0,1],
///   u'(0) = u'(1) = 0,
/// on N+1 grid points including endpoints, using the DCT-I.
///
/// Discrete operator chosen to be diagonalized by the cosine basis:
///   2(u_0 - u_1) / h^2 = f_0
///   (-u_{j-1} + 2u_j - u_{j+1}) / h^2 = f_j,   j = 1,...,N-1
///   2(u_N - u_{N-1}) / h^2 = f_N
///
/// The eigenvalues are
///   lambda_k = 2 - 2 cos(pi k / N),  k = 0,...,N.
///
/// For k = 0 the eigenvalue is zero, reflecting the constant nullspace.
/// The compatibility condition is the weighted sum
///   0.5 f_0 + sum_{j=1}^{N-1} f_j + 0.5 f_N = 0.
fn solve_poisson_neumann_dct1(f: &[f64]) -> Result<Vec<f64>, String> {
    if f.len() < 2 {
        return Err("Neumann solve requires at least two grid points.".to_string());
    }

    let n = f.len() - 1;
    let h = 1.0 / (n as f64);

    let weighted_sum: f64 = 0.5 * f[0] + f[1..n].iter().sum::<f64>() + 0.5 * f[n];
    if weighted_sum.abs() > 1.0e-10 {
        return Err(format!(
            "Neumann RHS violates the compatibility condition; weighted sum = {:.6e}",
            weighted_sum
        ));
    }

    let f_hat = dct1_forward(f)?;
    let mut u_hat = vec![0.0; n + 1];

    // Fix the nullspace by prescribing zero mean mode.
    u_hat[0] = 0.0;

    for k in 1..=n {
        let lambda_k = 2.0 - 2.0 * (PI * (k as f64) / (n as f64)).cos();
        u_hat[k] = (h * h) * f_hat[k] / lambda_k;
    }

    dct1_inverse(&u_hat)
}

fn print_real_vector(label: &str, x: &[f64]) {
    println!("{label}");
    for (j, value) in x.iter().enumerate() {
        println!("  [{:>2}] = {:>18.10e}", j, value);
    }
    println!();
}

fn main() -> Result<(), String> {
    println!("Program 12.4.2: Fast DCT-I and DST-I via FFT-Based Symmetry Extensions");
    println!();

    // ------------------------------------------------------------
    // Part 1: DCT-I round-trip test (even symmetry / Neumann flavor)
    // ------------------------------------------------------------
    let x_dct = vec![1.0, 1.5, -0.5, 2.0, 0.25, -1.0, 0.75];
    let x_dct_hat = dct1_forward(&x_dct)?;
    let x_dct_recovered = dct1_inverse(&x_dct_hat)?;
    let dct_roundtrip_err = max_abs_error(&x_dct, &x_dct_recovered);

    print_real_vector("Input data for DCT-I:", &x_dct);
    print_real_vector("DCT-I coefficients:", &x_dct_hat);
    println!(
        "DCT-I round-trip error: max_j |x_j - x_recovered,j| = {:.6e}",
        dct_roundtrip_err
    );
    println!();

    // ------------------------------------------------------------
    // Part 2: DST-I round-trip test (odd symmetry / Dirichlet flavor)
    // ------------------------------------------------------------
    let x_dst = vec![0.8, -0.2, 1.0, 0.5, -0.7, 0.3];
    let x_dst_hat = dst1_forward(&x_dst)?;
    let x_dst_recovered = dst1_inverse(&x_dst_hat)?;
    let dst_roundtrip_err = max_abs_error(&x_dst, &x_dst_recovered);

    print_real_vector("Interior data for DST-I:", &x_dst);
    print_real_vector("DST-I coefficients:", &x_dst_hat);
    println!(
        "DST-I round-trip error: max_j |x_j - x_recovered,j| = {:.6e}",
        dst_roundtrip_err
    );
    println!();

    // ------------------------------------------------------------
    // Part 3: 1D Poisson solve with homogeneous Dirichlet boundaries
    // ------------------------------------------------------------
    //
    // Exact solution:
    //   u(x) = sin(pi x) + 0.5 sin(3 pi x)
    // Then
    //   -u''(x) = pi^2 sin(pi x) + 4.5 pi^2 sin(3 pi x).
    //
    let n_dirichlet = 16usize;
    let h_dirichlet = 1.0 / ((n_dirichlet + 1) as f64);

    let mut f_dirichlet = vec![0.0; n_dirichlet];
    let mut u_exact_dirichlet = vec![0.0; n_dirichlet];

    for j in 1..=n_dirichlet {
        let x = j as f64 * h_dirichlet;
        u_exact_dirichlet[j - 1] = (PI * x).sin() + 0.5 * (3.0 * PI * x).sin();
        f_dirichlet[j - 1] = PI * PI * (PI * x).sin() + 4.5 * PI * PI * (3.0 * PI * x).sin();
    }

    let u_dirichlet = solve_poisson_dirichlet_dst1(&f_dirichlet)?;
    let err_dirichlet = max_abs_error(&u_dirichlet, &u_exact_dirichlet);
    let res_dirichlet = max_residual_dirichlet(&u_dirichlet, &f_dirichlet, h_dirichlet);

    println!("Dirichlet Poisson solve on {} interior points:", n_dirichlet);
    println!(
        "  max_j |u_numeric - u_exact| = {:.6e}",
        err_dirichlet
    );
    println!(
        "  max_j |(-u_{{j-1}} + 2u_j - u_{{j+1}})/h^2 - f_j| = {:.6e}",
        res_dirichlet
    );
    println!();

    // ------------------------------------------------------------
    // Part 4: 1D Poisson solve with homogeneous Neumann boundaries
    // ------------------------------------------------------------
    //
    // Exact zero-mean solution:
    //   u(x) = cos(pi x) + 0.25 cos(2 pi x)
    // Then
    //   -u''(x) = pi^2 cos(pi x) + pi^2 cos(2 pi x),
    // and u'(0)=u'(1)=0.
    //
    let n_neumann = 16usize;
    let h_neumann = 1.0 / (n_neumann as f64);

    let mut f_neumann = vec![0.0; n_neumann + 1];
    let mut u_exact_neumann = vec![0.0; n_neumann + 1];

    for j in 0..=n_neumann {
        let x = j as f64 * h_neumann;
        u_exact_neumann[j] = (PI * x).cos() + 0.25 * (2.0 * PI * x).cos();
        f_neumann[j] = PI * PI * (PI * x).cos() + PI * PI * (2.0 * PI * x).cos();
    }

    // Enforce the discrete compatibility condition exactly by removing the
    // trapezoidal weighted mean.
    let weighted_sum_f =
        0.5 * f_neumann[0] + f_neumann[1..n_neumann].iter().sum::<f64>() + 0.5 * f_neumann[n_neumann];
    let total_weight = n_neumann as f64;
    let mean_f = weighted_sum_f / total_weight;
    for value in &mut f_neumann {
        *value -= mean_f;
    }

    // Match the solver normalization by removing the same weighted mean
    // from the exact solution.
    let weighted_sum_u = 0.5 * u_exact_neumann[0]
        + u_exact_neumann[1..n_neumann].iter().sum::<f64>()
        + 0.5 * u_exact_neumann[n_neumann];
    let mean_u = weighted_sum_u / total_weight;
    for value in &mut u_exact_neumann {
        *value -= mean_u;
    }

    let u_neumann = solve_poisson_neumann_dct1(&f_neumann)?;
    let err_neumann = max_abs_error(&u_neumann, &u_exact_neumann);
    let res_neumann = max_residual_neumann(&u_neumann, &f_neumann, h_neumann);

    println!("Neumann Poisson solve on {} grid points:", n_neumann + 1);
    println!(
        "  max_j |u_numeric - u_exact| = {:.6e}",
        err_neumann
    );
    println!(
        "  max_j |L_h u - f| = {:.6e}",
        res_neumann
    );
    println!();

    Ok(())
}
```

Program 12.4.2 illustrates how the symmetry principles discussed in Section 12.4.3 lead directly to practical numerical algorithms. By constructing even and odd extensions of the input data, cosine and sine transforms can be computed using standard FFT routines, thereby achieving the same $O(N \log N)$ computational complexity as the FFT itself. The round-trip experiments confirm that the implemented transforms reconstruct the original data with errors at the level of machine precision, demonstrating the numerical consistency of the FFT-based reductions.

The Poisson solver examples highlight the close connection between boundary conditions and transform choice. The Dirichlet problem is naturally diagonalized by sine modes, while the Neumann problem is diagonalized by cosine modes. In both cases the transform converts the discrete differential operator into a diagonal system that can be solved by simple elementwise division. The residual diagnostics show that the transformed systems are solved to near machine precision, while the remaining discrepancy with the continuous analytic solution reflects the second-order accuracy of the finite-difference discretization rather than any deficiency of the transform method.

Together, these experiments reinforce the central idea of this section: many seemingly different transforms arise from the same underlying FFT structure when symmetry is taken into account. By exploiting even and odd extensions, numerical algorithms can adapt the FFT framework to a wide range of boundary value problems, signal processing tasks, and spectral methods for differential equations.

## 12.4.4. Multidimensional FFTs: Separability, Data Layout, and Communication

FFT generalizes cleanly to multidimensional arrays, but in two and three dimensions data movement often becomes as important as arithmetic. Modern multidimensional FFT work emphasizes communication-minimizing algorithms and performance-portable implementations because distributed transposes and all-to-all collectives can dominate runtime (Koopman and Bisseling, 2023; Venkat et al., 2025; Diez Sanhueza et al., 2025).

Let $x \in \mathbb{C}^{N_1\times\cdots\times N_d}$ be a complex array on a $d$-dimensional rectangular grid. Using multi-indices $n=(n_1,\dots,n_d)$ and $k=(k_1,\dots,k_d)$, define the $d$-D DFT by,

$$
X_{k_1,\dots,k_d}
=
\sum_{n_1=0}^{N_1-1} \cdots \sum_{n_d=0}^{N_d-1}
x_{n_1,\dots,n_d}
\exp\!\left(
-2\pi i \sum_{r=1}^d \frac{n_r k_r}{N_r}
\right)
\tag{12.4.15}
$$

This transform is separable. It is equivalent to applying a 1D DFT along one axis, then along the next, and so on. In linear-algebra terms, if $\mathrm{vec}(x)$ stacks entries into a vector, then:

$$
\mathrm{vec}(X)
=
\left(W_{N_d}\otimes\cdots\otimes W_{N_1}\right)
\mathrm{vec}(x)
\tag{12.4.16}
$$

so the multidimensional Fourier matrix is a Kronecker product of 1D Fourier matrices. This Kronecker structure is exactly what enables the practical “FFT along each dimension” algorithmic composition (Koopman and Bisseling, 2023).

Let $N = \prod_{r=1}^d N_r$. If one performs 1D FFTs along each axis, the arithmetic complexity is:

$$
O\!\left(N \sum_{r=1}^d \log N_r\right)
\tag{12.4.17}
$$

which becomes $O(dN\log N)$ in the equal-size case $N_1=\cdots=N_d$. However, performance is often limited by strided access when transforming along noncontiguous axes, by explicit transposes required to restore contiguous access, and, in distributed settings, by global all-to-all communication for data redistribution. Reducing the number and cost of all-to-all phases is central to scaling, and recent work proposes algorithms designed to require only a single all-to-all step under processor-count constraints (Koopman and Bisseling, 2023).

For a 2D array stored in row-major order (C-like), the linear index mapping is:

$$
\mathrm{offset}(n_1,n_2)
=
n_2 + N_2 n_1
\tag{12.4.18}
$$

so FFTs along the last axis are contiguous and cache-friendly, while FFTs along earlier axes are strided. This is why multidimensional FFT implementations interleave 1D transform phases with data permutations and transposes, and why performance-portable implementations emphasize data layout and decomposition strategies (Diez Sanhueza et al., 2025; Venkat et al., 2025).

If the input is real-valued on a multidimensional grid, Hermitian symmetry generalizes:

$$
X_{(-k_1 \bmod N_1), \dots, (-k_d \bmod N_d)}
=
\overline{X_{k_1,\dots,k_d}}
\tag{12.4.19}
$$

Consequently, most libraries store only a half-spectrum along one axis, yielding an output shape such as

$$
N_1 \times \cdots \times N_{d-1} \times \left(\left\lfloor \frac{N_d}{2} \right\rfloor + 1\right)
\tag{12.4.20}
$$

the multidimensional analogue of “store nonnegative frequencies only” in 1D (Salih and Hamood, 2023; Liu et al., 2025).

Finally, multidimensional FFTs implicitly treat the input as periodic in each dimension. This has two practical consequences. FFT-based convolution corresponds to circular convolution, so linear convolution requires sufficient zero-padding to avoid wrap-around. Discontinuities across a boundary behave like jump discontinuities, injecting high-frequency energy and polluting the spectrum. These issues appear directly in FFT-based Poisson solvers and imaging pipelines, where careful boundary handling and padding strategies are integral to correctness (Pei and Tong, 2025; Kohli et al., 2025).

### Implementation Perspective for Numerical Computing

From an implementation standpoint, the mathematical statements above translate into three practical rules. First, prefer real-to-complex transforms when the input is real and only nonnegative frequencies are needed. Second, choose and document a half-spectrum storage layout explicitly, including how $F_0$ and $F_{N/2}$ are stored in even-length cases. Third, treat multidimensional FFT performance as a data-layout problem as much as an arithmetic problem: contiguous axes are cheap, strided axes are expensive, and distributed transposes can dominate total runtime. These considerations are central in modern performance-portable FFT research and in large-scale simulation pipelines (Namugwanya et al., 2023; Koopman and Bisseling, 2023; Venkat et al., 2025).

### Rust Implementation

Following the discussion in Section 12.4.4 on the separability of multidimensional Fourier transforms and the role of data layout in practical FFT implementations, Program 12.4.3 provides a concrete Rust implementation of a two-dimensional FFT constructed from one-dimensional transforms applied along successive axes. The section explained that the multidimensional discrete Fourier transform defined in Equation (12.4.15) can be interpreted as a Kronecker-product operator as expressed in Equation (12.4.16), which implies that the transform can be computed by performing one-dimensional FFTs along each dimension of the array. This program demonstrates that principle explicitly. It constructs a row-major two-dimensional complex array, performs separable FFT computations along rows and columns, and verifies the correctness of the algorithm by comparing the result with a direct two-dimensional DFT evaluation. Additional experiments illustrate multidimensional Hermitian symmetry for real input data, half-spectrum storage along the final axis, inverse-transform reconstruction, and FFT-based circular convolution. In this way the program translates the mathematical statements of the section into practical numerical procedures while highlighting the importance of data layout and transform composition in multidimensional FFT implementations.

At the core of the implementation is the `Array2D` structure, which represents a two-dimensional complex array stored in row-major order within a single linear vector. The indexing rule implemented in the `offset` method follows the memory mapping described in Equation (12.4.18), converting a pair of indices $(n_1,n_2)$ into a linear position in memory. This layout makes transforms along the last axis contiguous and therefore cache-friendly, while transforms along the first axis require strided access. The structure also includes a `transpose` method that rearranges the data so that column transforms can be executed as row transforms on the transposed array, illustrating one of the standard strategies used in multidimensional FFT implementations.

The multidimensional transform itself is constructed from one-dimensional transforms using the functions `fft_rows_in_place`, `fft_cols_via_transpose`, and `fft2_separable`. The routine `fft_rows_in_place` applies the one-dimensional FFT to each row independently, corresponding to performing transforms along the last axis of the array. The function `fft_cols_via_transpose` then computes transforms along the remaining axis by transposing the array, applying the same row-wise routine, and transposing the result back. This sequence directly reflects the separable structure of the multidimensional transform implied by Equation (12.4.16), where the multidimensional Fourier operator is a Kronecker product of one-dimensional Fourier matrices. The inverse transform is implemented analogously in `ifft2_separable`, using inverse one-dimensional transforms and the appropriate normalization.

To verify the correctness of the separable algorithm, the program includes a direct implementation of the two-dimensional DFT through the function `direct_dft2`. This routine evaluates the transform directly from Equation (12.4.15) by performing the nested summations over both spatial dimensions. Although computationally expensive, it provides a useful reference solution for small test arrays and allows the program to measure the numerical difference between the direct and separable implementations.

The function `hermitian_symmetry_error_real_2d` examines the conjugate-symmetry property of the multidimensional Fourier transform for real input data, as described in Equation (12.4.19). This property implies that half of the complex spectrum is redundant. The routines `extract_half_spectrum_last_axis` and `reconstruct_full_from_half_last_axis` demonstrate how practical FFT libraries exploit this redundancy by storing only the nonnegative frequencies along one axis, corresponding to the storage layout described in Equation (12.4.20). The reconstruction step confirms that the full spectrum can be recovered from the stored half-spectrum using Hermitian symmetry.

The program also includes an illustration of FFT-based circular convolution through the functions `circular_convolution_fft`, `circular_convolution_direct`, and `pointwise_multiply`. In accordance with the periodic interpretation of the FFT discussed in Section 12.4.4, convolution in the spatial domain corresponds to elementwise multiplication in the frequency domain. The FFT-based convolution is compared with a direct periodic convolution implementation to verify numerical correctness.

The `main` function organizes these components into a sequence of numerical experiments. It first demonstrates that the separable two-dimensional FFT agrees with the direct two-dimensional DFT for a small complex array. It then computes the spectrum of a real-valued grid, verifies multidimensional Hermitian symmetry, and demonstrates half-spectrum storage along the final axis. Next, the program performs an inverse transform to confirm accurate reconstruction of the original data. Finally, it illustrates FFT-based circular convolution and prints several offset calculations to reinforce the row-major indexing rule discussed in the section. Together these demonstrations show how the theoretical properties of multidimensional Fourier transforms translate into practical numerical algorithms.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
rustfft = "6"
```

```rust
// Program 12.4.3: Two-Dimensional FFT via Separable One-Dimensional Transforms
//
// This program demonstrates the multidimensional FFT ideas discussed in
// Section 12.4.4.
//
// Included components:
// 1. A row-major 2D complex array type with explicit offset mapping.
// 2. A separable 2D FFT built from 1D FFTs along rows and columns.
// 3. A direct 2D DFT reference implementation for verification.
// 4. Hermitian symmetry checks for real-valued input.
// 5. Half-spectrum extraction along the last axis.
// 6. FFT-based circular convolution on a 2D grid.
//
// To run:
//   cargo run

use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

type C64 = Complex<f64>;

#[derive(Clone, Debug)]
struct Array2D {
    n1: usize,
    n2: usize,
    data: Vec<C64>,
}

impl Array2D {
    fn new(n1: usize, n2: usize) -> Self {
        Self {
            n1,
            n2,
            data: vec![C64::new(0.0, 0.0); n1 * n2],
        }
    }

    fn from_fn<F>(n1: usize, n2: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> C64,
    {
        let mut out = Self::new(n1, n2);
        for i in 0..n1 {
            for j in 0..n2 {
                out[(i, j)] = f(i, j);
            }
        }
        out
    }

    fn from_real_fn<F>(n1: usize, n2: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> f64,
    {
        Self::from_fn(n1, n2, |i, j| C64::new(f(i, j), 0.0))
    }

    fn offset(&self, i: usize, j: usize) -> usize {
        assert!(i < self.n1 && j < self.n2);
        j + self.n2 * i
    }

    fn transpose(&self) -> Self {
        let mut out = Self::new(self.n2, self.n1);
        for i in 0..self.n1 {
            for j in 0..self.n2 {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }

    fn dims(&self) -> (usize, usize) {
        (self.n1, self.n2)
    }
}

impl std::ops::Index<(usize, usize)> for Array2D {
    type Output = C64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let idx = self.offset(index.0, index.1);
        &self.data[idx]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Array2D {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let idx = self.offset(index.0, index.1);
        &mut self.data[idx]
    }
}

fn fft_1d_in_place(line: &mut [C64], inverse: bool) {
    let n = line.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    };
    fft.process(line);

    if inverse {
        let scale = 1.0 / n as f64;
        for z in line.iter_mut() {
            *z *= scale;
        }
    }
}

fn fft_rows_in_place(x: &mut Array2D, inverse: bool) {
    for i in 0..x.n1 {
        let start = i * x.n2;
        let end = start + x.n2;
        fft_1d_in_place(&mut x.data[start..end], inverse);
    }
}

fn fft_cols_via_transpose(x: &mut Array2D, inverse: bool) {
    let mut xt = x.transpose();
    fft_rows_in_place(&mut xt, inverse);
    *x = xt.transpose();
}

fn fft2_separable(x: &Array2D) -> Array2D {
    let mut out = x.clone();
    fft_rows_in_place(&mut out, false);
    fft_cols_via_transpose(&mut out, false);
    out
}

fn ifft2_separable(x_hat: &Array2D) -> Array2D {
    let mut out = x_hat.clone();
    fft_rows_in_place(&mut out, true);
    fft_cols_via_transpose(&mut out, true);
    out
}

fn omega(n: usize, k: usize) -> C64 {
    let theta = -2.0 * PI * (k as f64) / (n as f64);
    C64::new(theta.cos(), theta.sin())
}

fn direct_dft2(x: &Array2D) -> Array2D {
    let (n1, n2) = x.dims();
    let mut out = Array2D::new(n1, n2);

    for k1 in 0..n1 {
        for k2 in 0..n2 {
            let mut sum = C64::new(0.0, 0.0);
            for n1_idx in 0..n1 {
                for n2_idx in 0..n2 {
                    let phase = omega(n1, n1_idx * k1) * omega(n2, n2_idx * k2);
                    sum += x[(n1_idx, n2_idx)] * phase;
                }
            }
            out[(k1, k2)] = sum;
        }
    }

    out
}

fn max_complex_error(a: &Array2D, b: &Array2D) -> f64 {
    assert_eq!(a.dims(), b.dims());
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0_f64, f64::max)
}

fn hermitian_symmetry_error_real_2d(x_hat: &Array2D) -> f64 {
    let (n1, n2) = x_hat.dims();
    let mut max_err = 0.0_f64;

    for k1 in 0..n1 {
        for k2 in 0..n2 {
            let mk1 = (n1 - k1) % n1;
            let mk2 = (n2 - k2) % n2;
            let lhs = x_hat[(mk1, mk2)];
            let rhs = x_hat[(k1, k2)].conj();
            let err = (lhs - rhs).norm();
            max_err = max_err.max(err);
        }
    }

    max_err
}

fn extract_half_spectrum_last_axis(x_hat: &Array2D) -> Array2D {
    let (n1, n2) = x_hat.dims();
    let keep = n2 / 2 + 1;
    let mut out = Array2D::new(n1, keep);

    for i in 0..n1 {
        for j in 0..keep {
            out[(i, j)] = x_hat[(i, j)];
        }
    }

    out
}

fn reconstruct_full_from_half_last_axis(half: &Array2D, original_n2: usize) -> Array2D {
    let n1 = half.n1;
    let keep = half.n2;
    assert_eq!(keep, original_n2 / 2 + 1);

    let mut full = Array2D::new(n1, original_n2);

    for k1 in 0..n1 {
        for k2 in 0..keep {
            full[(k1, k2)] = half[(k1, k2)];
        }
    }

    for k1 in 0..n1 {
        for k2 in 1..(original_n2 / 2) {
            let mk1 = (n1 - k1) % n1;
            let mk2 = original_n2 - k2;
            full[(mk1, mk2)] = half[(k1, k2)].conj();
        }
    }

    if original_n2 % 2 == 0 {
        let nyquist = original_n2 / 2;
        for k1 in 0..n1 {
            let mk1 = (n1 - k1) % n1;
            full[(mk1, nyquist)] = full[(k1, nyquist)].conj();
        }
    }

    full
}

fn pointwise_multiply(a: &Array2D, b: &Array2D) -> Array2D {
    assert_eq!(a.dims(), b.dims());
    Array2D::from_fn(a.n1, a.n2, |i, j| a[(i, j)] * b[(i, j)])
}

fn circular_convolution_direct(a: &Array2D, b: &Array2D) -> Array2D {
    assert_eq!(a.dims(), b.dims());
    let (n1, n2) = a.dims();
    let mut out = Array2D::new(n1, n2);

    for i in 0..n1 {
        for j in 0..n2 {
            let mut sum = C64::new(0.0, 0.0);
            for p in 0..n1 {
                for q in 0..n2 {
                    let ip = (i + n1 - p) % n1;
                    let jq = (j + n2 - q) % n2;
                    sum += a[(p, q)] * b[(ip, jq)];
                }
            }
            out[(i, j)] = sum;
        }
    }

    out
}

fn circular_convolution_fft(a: &Array2D, b: &Array2D) -> Array2D {
    let a_hat = fft2_separable(a);
    let b_hat = fft2_separable(b);
    let c_hat = pointwise_multiply(&a_hat, &b_hat);
    ifft2_separable(&c_hat)
}

fn print_real_part(label: &str, x: &Array2D) {
    println!("{label}");
    for i in 0..x.n1 {
        for j in 0..x.n2 {
            print!("{:>14.6e} ", x[(i, j)].re);
        }
        println!();
    }
    println!();
}

fn print_complex_matrix(label: &str, x: &Array2D) {
    println!("{label}");
    for i in 0..x.n1 {
        for j in 0..x.n2 {
            let z = x[(i, j)];
            print!("({:>9.3e},{:>9.3e}) ", z.re, z.im);
        }
        println!();
    }
    println!();
}

fn main() {
    println!("Program 12.4.3: Two-Dimensional FFT via Separable One-Dimensional Transforms");
    println!();

    // -----------------------------------------------------------------
    // Part 1: Verify separability against a direct two-dimensional DFT.
    // -----------------------------------------------------------------
    let small = Array2D::from_fn(4, 4, |i, j| {
        let re = (i + 2 * j) as f64;
        let im = if (i + j) % 2 == 0 { 0.5 } else { -0.25 };
        C64::new(re, im)
    });

    let small_fft_sep = fft2_separable(&small);
    let small_fft_direct = direct_dft2(&small);
    let separability_error = max_complex_error(&small_fft_sep, &small_fft_direct);

    print_complex_matrix("Small complex 4x4 input:", &small);
    print_complex_matrix("Separable 2D FFT of the 4x4 input:", &small_fft_sep);
    println!(
        "Separable-versus-direct 2D DFT check: max error = {:.6e}",
        separability_error
    );
    println!();

    // -----------------------------------------------------------------
    // Part 2: Real-valued input, Hermitian symmetry, and half-spectrum.
    // -----------------------------------------------------------------
    let n1 = 6usize;
    let n2 = 8usize;
    let real_grid = Array2D::from_real_fn(n1, n2, |i, j| {
        let x = i as f64 / n1 as f64;
        let y = j as f64 / n2 as f64;
        1.0
            + 0.8 * (2.0 * PI * x).cos()
            - 0.6 * (2.0 * PI * y).sin()
            + 0.25 * (4.0 * PI * x + 2.0 * PI * y).cos()
    });

    let real_grid_hat = fft2_separable(&real_grid);
    let hermitian_err = hermitian_symmetry_error_real_2d(&real_grid_hat);

    let half = extract_half_spectrum_last_axis(&real_grid_hat);
    let rebuilt = reconstruct_full_from_half_last_axis(&half, n2);
    let rebuild_err = max_complex_error(&real_grid_hat, &rebuilt);

    print_real_part("Real-valued 6x8 input grid:", &real_grid);
    println!(
        "2D Hermitian symmetry check for real input: max error = {:.6e}",
        hermitian_err
    );
    println!(
        "Half-spectrum reconstruction check along the last axis: max error = {:.6e}",
        rebuild_err
    );
    println!(
        "Stored half-spectrum shape: {} x {}",
        half.n1, half.n2
    );
    println!();

    // -----------------------------------------------------------------
    // Part 3: Inverse-transform reconstruction.
    // -----------------------------------------------------------------
    let recovered = ifft2_separable(&real_grid_hat);
    let inverse_error = max_complex_error(&real_grid, &recovered);
    println!(
        "Inverse 2D FFT reconstruction check: max error = {:.6e}",
        inverse_error
    );
    println!();

    // -----------------------------------------------------------------
    // Part 4: FFT-based circular convolution.
    // -----------------------------------------------------------------
    let kernel = Array2D::from_real_fn(n1, n2, |i, j| {
        if (i == 0 && j == 0) || (i == 0 && j == 1) || (i == 1 && j == 0) {
            1.0 / 3.0
        } else {
            0.0
        }
    });

    let conv_fft = circular_convolution_fft(&real_grid, &kernel);
    let conv_direct = circular_convolution_direct(&real_grid, &kernel);
    let conv_err = max_complex_error(&conv_fft, &conv_direct);

    print_real_part("Circular convolution result via FFT:", &conv_fft);
    println!(
        "Circular convolution check (FFT versus direct): max error = {:.6e}",
        conv_err
    );
    println!();

    // -----------------------------------------------------------------
    // Part 5: Data-layout reminder from row-major indexing.
    // -----------------------------------------------------------------
    println!("Row-major offset examples for a {} x {} array:", n1, n2);
    println!("  offset(0, 0) = {}", real_grid.offset(0, 0));
    println!("  offset(0, 1) = {}", real_grid.offset(0, 1));
    println!("  offset(1, 0) = {}", real_grid.offset(1, 0));
    println!("  offset(2, 3) = {}", real_grid.offset(2, 3));
    println!();
    println!("This confirms that the last axis is contiguous in memory,");
    println!("so row transforms are contiguous while column transforms");
    println!("require strided access or an explicit transpose strategy.");
}
```

Program 12.4.3 demonstrates how the multidimensional Fourier transform described in Section 12.4.4 can be implemented efficiently by composing one-dimensional FFTs along successive array axes. The separable algorithm reproduces the direct two-dimensional DFT to roundoff accuracy, confirming the Kronecker-product interpretation of the transform given in Equation (12.4.16). The Hermitian symmetry tests for real input data further illustrate the redundancy described in Equation (12.4.19), while the half-spectrum storage example shows how practical FFT implementations exploit this symmetry to reduce memory requirements.

The circular convolution experiment highlights another important implication of the multidimensional FFT: because the transform implicitly assumes periodic boundary conditions, convolution operations computed in the frequency domain correspond to circular convolution in the spatial domain. This behavior is central to many numerical algorithms, including FFT-based Poisson solvers and convolution-based filtering methods used in scientific computing and signal processing.

From an implementation perspective, the program also emphasizes that multidimensional FFT performance depends not only on arithmetic complexity but also on data layout. Row-major storage makes transforms along the last axis contiguous and efficient, while transforms along earlier axes require strided memory access or explicit transposition. Understanding and managing these data-movement costs is essential in modern high-performance FFT implementations.

# 12.5 Fast Sine and Cosine Transforms

Sine and cosine transforms appear throughout numerical PDEs not because they are alternative signal-processing tools, but because they are the discrete eigenfunction bases that match common boundary conditions. On simple grids, constant-coefficient elliptic operators become structured matrices whose eigenvectors are sampled sines or cosines. As a result, discrete sine transforms (DSTs) and discrete cosine transforms (DCTs) diagonalize, or nearly diagonalize, the discrete operators that arise in Poisson solvers, diffusion problems, and separation-of-variables style boundary-value computations. Modern work continues to emphasize both the modeling role of sine and cosine bases for Dirichlet and Neumann constraints and the algorithmic role of structured matrix factorizations and FFT reductions that make these transforms as fast as FFTs (Pei and Tong, 2025; Risthaus and Schneider, 2024; Bielak et al., 2024; Polyakova et al., 2025).

The central computational message is that sine and cosine transforms inherit FFT speed because they can be expressed as FFTs of even or odd extensions, followed by simple phase corrections. For large (N), this reduction is the dominant approach. For very small block sizes, which occur in embedded DSP kernels, codecs, and some multigrid smoothers, recent research also synthesizes DST and DCT kernels directly by factoring their transform matrices into sparse structured components (Bielak et al., 2024; Polyakova et al., 2025). Both regimes matter in modern practice.

## 12.5.1. Why Sine Versus Cosine: Boundary Conditions as Symmetry

The appearance of sine and cosine transforms in numerical PDEs follows directly from the spectral structure of self-adjoint operators under boundary constraints. Consider the model eigenvalue problem on $[0,L]$,

$$-\frac{d^2 u}{dx^2} = \lambda u \tag{12.5.1}$$

Because the operator is self-adjoint, its eigenfunctions form an orthogonal basis in $L^2(0,L)$, and the boundary conditions determine which subset of trigonometric functions is admissible.

If homogeneous Dirichlet conditions are imposed,

$$u(0)=u(L)=0 \tag{12.5.2}$$

the admissible eigenfunctions are:

$$u_k(x)=\sin!\left(\frac{k\pi x}{L}\right),\qquad\lambda_k=\left(\frac{k\pi}{L}\right)^2,\quad k=1,2,\dots \tag{12.5.3}$$

These functions vanish at both endpoints. Equivalently, they arise by extending $u$ periodically to $[-L,L]$ with odd symmetry,

$$u(-x)=-u(x) \tag{12.5.4}$$

An odd periodic extension contains only sine terms in its Fourier expansion. Thus Dirichlet boundary conditions correspond naturally to odd symmetry and hence to sine bases.

If homogeneous Neumann conditions are imposed,

$$u'(0)=u'(L)=0 \tag{12.5.5}$$

the eigenfunctions become:

$$
u_k(x)=\cos\left(\frac{k\pi x}{L}\right),\qquad
\lambda_k=\left(\frac{k\pi}{L}\right)^2,\quad k=0,1,2,\dots
\tag{12.5.6}
$$

These functions have zero derivative at the endpoints. They arise from even periodic extension,

$$u(-x)=u(x) \tag{12.5.7}$$

which yields a Fourier series containing only cosine terms. Hence Neumann boundary conditions correspond naturally to even symmetry and therefore to cosine bases.

This continuous symmetry principle carries over directly to discrete operators. Let $x_j=jh$, $j=0,\dots,n$, with $h=L/n$, and define the centered second-difference operator:

$$
(\Delta_h u)_j

\frac{u_{j-1}-2u_j+u_{j+1}}{h^2} \tag{12.5.8}
$$

Under Dirichlet constraints $u_0=u_n=0$, the interior difference matrix acting on $j=1,\dots,n-1$ has eigenvectors:

$$
v^{(k)}_j
=
\sin\!\left(\frac{k\pi j}{n}\right),
\quad
k=1,\dots,n-1
\tag{12.5.9}
$$

which form the discrete sine transform basis. Under Neumann conditions implemented via symmetric ghost-point relations, the eigenvectors become,

$$
v^{(k)}_j
=
\cos\!\left(\frac{k\pi j}{n}\right),
\quad
k=0,\dots,n
\tag{12.5.10}
$$

which form the discrete cosine transform basis. In each case, the structured discrete Laplacian is diagonalized by the corresponding transform matrix. Consequently, solving a one-dimensional Poisson equation reduces to transforming the right-hand side, dividing by the eigenvalues $\lambda_k$, and transforming back. The overall complexity is $O(n\log n)$ because the DST and DCT inherit FFT speed.

Recent Poisson-solver research makes this boundary-condition correspondence explicit, matching DST-based solvers to Dirichlet constraints and DCT-based solvers to Neumann constraints (Pei and Tong, 2025). In FFT-based computational micromechanics, discrete sine and cosine transforms similarly enforce non-periodic constraints while preserving spectral diagonalization and FFT-level efficiency (Risthaus and Schneider, 2024; Paux et al., 2025). In both settings, the transform type is dictated by the modeling assumptions encoded in the boundary conditions.

The symmetry viewpoint also clarifies the implementation mechanism described in the introduction to this section. When a function defined on $[0,L]$ is extended by odd reflection as in (12.5.4), its discrete Fourier transform contains purely imaginary components corresponding to sine amplitudes. When extended evenly as in (12.5.7), the relevant information appears in the real parts and corresponds to cosine amplitudes. Thus the DST and DCT can be realized as FFTs of odd or even extensions, followed by simple scaling and phase adjustments. This mirrors the real-data symmetry exploited in Section 12.4, but here the symmetry is spatial and enforced by boundary conditions rather than by complex conjugation.

In summary, sine and cosine transforms are not alternative signal-processing tools introduced for convenience. They are the discrete eigenfunction bases selected by boundary symmetry. Boundary conditions determine symmetry; symmetry determines the admissible eigenfunctions; and those eigenfunctions determine which fast trigonometric transform diagonalizes the discrete operator.

## 12.5.2. Families of DCT/DST and Convention Choices

There is not a single discrete cosine transform (DCT) or discrete sine transform (DST), but rather families of closely related transforms distinguished by endpoint treatment, grid alignment, and phase conventions. These variants arise because the discrete transform must reflect how boundary values are sampled and how symmetry is enforced. A recent structured overview emphasizes that multiple types exist precisely to encode different boundary conventions, inclusion or exclusion of endpoints, and half-grid phase shifts (Bielak et al., 2024). In numerical computing, these choices are not merely historical artifacts from signal processing; they correspond to different discrete eigenproblems.

To see the distinction concretely, consider a uniform grid $x_j = jh$, $j=0,\dots,n$. An endpoint-including cosine transform, commonly labeled DCT-I, has the form:

$$
\widehat{u}_k
=
\sum_{j=0}^{n}
u_j
\cos\!\left(\frac{\pi k j}{n}\right),
\quad
k=0,\dots,n
\tag{12.5.11}
$$

Here both endpoints $j=0$ and $j=n$ participate explicitly. This form arises naturally when the discrete problem includes boundary values as unknowns, for example in finite-difference discretizations of Neumann problems where the grid includes both endpoints. The corresponding sine analogue, DST-I,

$$
\widehat{u}_k
=
\sum_{j=1}^{n-1}
u_j
\sin\!\left(\frac{\pi k j}{n}\right),
\quad
k=1,\dots,n-1
\tag{12.5.12}
$$

s consistent with Dirichlet constraints $u_0=u_n=0$ and diagonalizes the standard second-difference matrix with those boundary conditions.

In contrast, half-shifted forms, often labeled DCT-II and DST-II, use arguments of the type:

$$
\cos\!\left(\frac{\pi}{n}\left(j+\tfrac12\right)k\right)
\quad \text{or} \quad
\sin\!\left(\frac{\pi}{n}\left(j+\tfrac12\right)(k+1)\right)
\tag{12.5.13}
$$

which correspond to sampling at midpoints or to enforcing symmetry about half-grid locations. These transforms arise naturally on staggered meshes, in spectral element interfaces, and in image and signal processing pipelines where samples are interpreted as cell averages rather than nodal values. The half-grid shift effectively changes the discrete eigenvalue problem being solved: the associated difference operator has slightly different boundary closure and therefore different eigenvectors.

These distinctions are not cosmetic. The transform type determines exactly which discrete operator is diagonalized. For example, the eigenvalues associated with (12.5.11) differ slightly from those associated with (12.5.13) because the underlying discrete Laplacian differs in its boundary stencil. Consequently, using a DCT-II where a DCT-I is required generally produces a solver that is close, but not algebraically exact, for the intended boundary-value problem.

Normalization conventions introduce an additional layer of variation. Some definitions use orthonormal scaling so that the transform matrix $C$ satisfies,

$$C^{T} C = I \tag{12.5.14}$$

while others use asymmetric forward and inverse scalings, as is common in FFT libraries. The scaling choice affects Parseval-type identities,

$$
\sum_{j} |u_j|^2

\sum_{k} |\widehat{u}_k|^2 \tag{12.5.15}
$$

and therefore influences energy conservation, stability analyses, and residual computations in PDE solvers. Inconsistent normalization between forward and inverse transforms is a common source of subtle amplitude errors.

From a software-engineering perspective, the safest practice in a numerical PDE codebase is to select one DCT/DST convention, document it explicitly, and adhere to it consistently. This documentation should specify:

1. whether endpoints are included or excluded,
2. whether half-grid shifts are present,
3. the precise forward and inverse scaling factors, and
4. the eigenvalue formula associated with the discretized operator.

When these conventions are fixed and clearly recorded, sine and cosine transforms function as exact diagonalizers of structured discrete operators. When they are mixed or implicitly assumed, the resulting discrepancies often appear as unexplained factors of two, incorrect boundary amplitudes, or energy mismatches. The multiplicity of DCT/DST families is therefore best understood not as ambiguity, but as a precise encoding of different discrete boundary-value formulations.

## 12.5.3. A Canonical Sine Transform and Reduction to FFT by Odd Extension

A canonical sine transform matching homogeneous Dirichlet endpoints is defined on interior points $j=1,\dots,N-1$ by,

$$
S_k
=
\sum_{j=1}^{N-1}
f_j
\sin\!\left(\frac{\pi j k}{N}\right),
\qquad
k=1,\dots,N-1
\tag{12.5.1}
$$

with boundary values $f_0=f_N=0$. This transform diagonalizes the second-difference operator under Dirichlet boundary conditions in standard finite-difference Poisson solvers, motivating its use in DST-accelerated Poisson methods (Pei and Tong, 2025).

The computational reason DST is fast is that it can be reduced to an FFT of an odd extension. Define an odd extension $g$ of length $2N$ by,

$$
g_j =
\begin{cases}
0 & j=0, \\[4pt]
f_j & 1 \le j \le N-1, \\[4pt]
0 & j=N, \\[4pt]
-\,f_{2N-j} & N+1 \le j \le 2N-1 
\end{cases}
\tag{12.5.2}
$$

Compute the length-$2N$ DFT:

$$
G_k = \sum_{j=0}^{2N-1} g_j\, \exp\!\left(-\frac{2\pi i jk}{2N}\right)
\tag{12.5.3}
$$

Because $g$ is odd, the sine coefficients are recovered from the imaginary part of $G_k$ (up to normalization conventions):

$$
S_k = -\frac{1}{2}\,\operatorname{Im}(G_k),\qquad k = 1,\dots,N-1
\tag{12.5.4}
$$

The essential structural point is that the DST can be computed via one FFT of an odd extension, inheriting $O(N\log N)$ complexity. This FFT reducibility is why DST-based Poisson and boundary-value solvers are regarded as FFT-accelerated methods in modern literature (Pei and Tong, 2025; Risthaus and Schneider, 2024).

### Rust Implementation

Following the discussion in Section 12.5.3 on the canonical discrete sine transform and its reduction to an FFT through odd extension, Program 12.5.1 provides a practical Rust implementation of the sine transform defined in equation (12.5.1). The section explained that although the transform appears to require $O(N^2)$ operations when evaluated directly, its structure allows it to be computed in $O(N\log N)$ time by embedding the data into an odd-symmetric sequence and applying a single FFT, as described in equations (12.5.2)–(12.5.4). This program demonstrates that reduction explicitly. It constructs the odd extension of the interior data, computes the corresponding Fourier transform, and extracts the sine coefficients from the imaginary part of the spectrum. To verify correctness, the program compares the FFT-based transform with a direct implementation of the canonical sine transform, performs an inverse reconstruction test, and applies the transform to solve a finite-difference Poisson equation with homogeneous Dirichlet boundary conditions. Through these experiments, the program illustrates how the symmetry principles described in the section translate directly into efficient numerical algorithms.

At the core of the implementation are the routines `dst_canonical_direct` and `dst_canonical_fft`, which compute the canonical sine transform defined in equation (12.5.1). The function `dst_canonical_direct` evaluates the transform exactly as written in the definition by performing the summation over the interior points $j=1,\dots,N-1$. This direct formulation has quadratic computational complexity and is therefore useful primarily as a reference for verifying the correctness of faster algorithms.

The routine `dst_canonical_fft` implements the fast transform using the odd-extension strategy described in equations (12.5.2)–(12.5.4). It first constructs the extended sequence $g$ of length $2N$ whose values satisfy the antisymmetry condition specified in equation (12.5.2). Because this sequence is odd about the endpoints, its Fourier transform contains purely imaginary components corresponding to sine amplitudes. The program then computes the length-$2N$ discrete Fourier transform using the function `fft_forward`, which provides a wrapper around the `rustfft` library. Finally, the sine coefficients are extracted from the imaginary parts of the Fourier spectrum according to equation (12.5.4).

To verify that the forward transform is implemented correctly, the program also provides an inverse transform through the function `dst_canonical_inverse`. Using the orthogonality properties of the sine basis, the inverse transform reconstructs the original interior data from the sine coefficients using the appropriate normalization factor. A reconstruction test compares the recovered values with the original input to confirm that the forward and inverse transforms form a consistent pair.

The function `solve_poisson_dirichlet_dst` illustrates the most important practical application of the canonical sine transform: the solution of a finite-difference Poisson equation with homogeneous Dirichlet boundary conditions. As discussed earlier in the section, the discrete Laplacian under these boundary conditions has eigenvectors given by the sine basis of equation (12.5.9). The routine therefore transforms the right-hand side into sine space, divides each coefficient by the corresponding eigenvalue of the discrete operator, and reconstructs the solution using the inverse transform. This procedure demonstrates how the sine transform diagonalizes the structured difference operator and reduces the Poisson solve to a sequence of simple algebraic operations.

Several auxiliary functions support these computations. The routines `fft_forward` and `fft_inverse` provide the underlying complex FFT operations used to implement the odd-extension reduction. The diagnostic functions `max_abs_error` and `max_residual_dirichlet` measure numerical accuracy by comparing vectors and by evaluating the residual of the finite-difference equation. These checks ensure that both the transform and the Poisson solver behave as expected.

The `main` function organizes the program into several numerical experiments. It first compares the direct and FFT-based implementations of the canonical sine transform to confirm that they produce identical coefficients up to floating-point roundoff. It then verifies that the inverse transform reconstructs the original interior data accurately. Finally, it demonstrates the use of the sine transform in solving a Dirichlet Poisson problem and evaluates both the solution error and the residual of the discrete equation. An additional diagnostic confirms that the odd extension itself is preserved under a forward and inverse FFT pair.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
rustfft = "6"
```

```rust
// Program 12.5.1: Canonical Sine Transform via FFT of an Odd Extension
//
// This program implements the canonical sine transform described in
// equations (12.5.1) to (12.5.4). The transform is computed by forming
// the odd extension of the interior data, applying a length-2N FFT,
// and extracting the sine coefficients from the imaginary part.
//
// The program includes:
// 1. A direct reference implementation of the canonical sine transform.
// 2. An FFT-based implementation using the odd extension.
// 3. An inverse transform consistent with the chosen normalization.
// 4. A Dirichlet Poisson solver based on the sine-transform diagonalization.
// 5. Diagnostic checks for transform accuracy, reconstruction, and residuals.
//
// To run:
//   cargo run

use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

type C64 = Complex<f64>;

/// Forward complex FFT with the convention
/// X_k = sum_n x_n exp(-2 pi i n k / N).
fn fft_forward(input: &[C64]) -> Vec<C64> {
    let n = input.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer = input.to_vec();
    fft.process(&mut buffer);
    buffer
}

/// Inverse complex FFT with normalization by 1/N.
fn fft_inverse(input: &[C64]) -> Vec<C64> {
    let n = input.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_inverse(n);
    let mut buffer = input.to_vec();
    fft.process(&mut buffer);

    let scale = 1.0 / n as f64;
    for z in &mut buffer {
        *z *= scale;
    }

    buffer
}

/// Direct evaluation of the canonical sine transform in equation (12.5.1).
///
/// Input:
///   f[j-1] corresponds to the interior value f_j for j = 1, ..., N-1.
///
/// Output:
///   s[k-1] corresponds to S_k for k = 1, ..., N-1.
fn dst_canonical_direct(f: &[f64]) -> Vec<f64> {
    let n = f.len() + 1;
    let mut s = vec![0.0; n - 1];

    for k in 1..n {
        let mut sum = 0.0;
        for j in 1..n {
            let theta = PI * (j as f64) * (k as f64) / (n as f64);
            sum += f[j - 1] * theta.sin();
        }
        s[k - 1] = sum;
    }

    s
}

/// FFT-based canonical sine transform using the odd extension in (12.5.2)
/// and the extraction formula in (12.5.4).
///
/// If the interior vector has length N-1, the odd extension has length 2N.
fn dst_canonical_fft(f: &[f64]) -> Result<Vec<f64>, String> {
    if f.is_empty() {
        return Err("The canonical DST requires at least one interior point.".to_string());
    }

    let n = f.len() + 1;
    let m = 2 * n;

    // Build the odd extension g of length 2N:
    // g_0 = 0,
    // g_j = f_j,            1 <= j <= N-1,
    // g_N = 0,
    // g_j = -f_{2N-j},      N+1 <= j <= 2N-1.
    let mut g = vec![C64::new(0.0, 0.0); m];
    for j in 1..n {
        g[j] = C64::new(f[j - 1], 0.0);
        g[m - j] = C64::new(-f[j - 1], 0.0);
    }

    let g_hat = fft_forward(&g);

    // Extract S_k = -(1/2) Im(G_k), k = 1, ..., N-1.
    let mut s = vec![0.0; n - 1];
    for k in 1..n {
        s[k - 1] = -0.5 * g_hat[k].im;
    }

    Ok(s)
}

/// Inverse transform for the canonical sine transform.
///
/// For the forward definition
///   S_k = sum_{j=1}^{N-1} f_j sin(pi j k / N),
/// the inverse is
///   f_j = (2/N) sum_{k=1}^{N-1} S_k sin(pi j k / N),
/// because
///   sum_{k=1}^{N-1} sin(pi j k / N) sin(pi l k / N) = (N/2) delta_{jl}.
fn dst_canonical_inverse(s: &[f64]) -> Result<Vec<f64>, String> {
    if s.is_empty() {
        return Err("The inverse canonical DST requires at least one coefficient.".to_string());
    }

    let n = s.len() + 1;
    let mut f = vec![0.0; n - 1];

    for j in 1..n {
        let mut sum = 0.0;
        for k in 1..n {
            let theta = PI * (j as f64) * (k as f64) / (n as f64);
            sum += s[k - 1] * theta.sin();
        }
        f[j - 1] = (2.0 / n as f64) * sum;
    }

    Ok(f)
}

/// Solves the finite-difference Dirichlet Poisson problem
///
///   (-u_{j-1} + 2u_j - u_{j+1}) / h^2 = rhs_j,   j = 1, ..., N-1,
///   u_0 = u_N = 0,
///
/// using the canonical sine transform. The interior right-hand side is stored
/// as rhs[j-1] = rhs_j, j = 1, ..., N-1.
fn solve_poisson_dirichlet_dst(rhs: &[f64]) -> Result<Vec<f64>, String> {
    if rhs.is_empty() {
        return Err("Dirichlet Poisson solve requires at least one interior point.".to_string());
    }

    let n = rhs.len() + 1;
    let h = 1.0 / n as f64;

    let rhs_hat = dst_canonical_fft(rhs)?;
    let mut u_hat = vec![0.0; n - 1];

    for k in 1..n {
        let lambda_k = 2.0 - 2.0 * (PI * (k as f64) / (n as f64)).cos();
        u_hat[k - 1] = (h * h) * rhs_hat[k - 1] / lambda_k;
    }

    dst_canonical_inverse(&u_hat)
}

/// Maximum absolute error between two real vectors.
fn max_abs_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Residual norm for the Dirichlet second-difference equation.
fn max_residual_dirichlet(u: &[f64], rhs: &[f64], h: f64) -> f64 {
    assert_eq!(u.len(), rhs.len());
    let n_minus_1 = u.len();
    let mut max_r: f64 = 0.0;

    for j in 0..n_minus_1 {
        let u_left = if j == 0 { 0.0 } else { u[j - 1] };
        let u_right = if j + 1 == n_minus_1 { 0.0 } else { u[j + 1] };
        let lhs = (-u_left + 2.0 * u[j] - u_right) / (h * h);
        let r = (lhs - rhs[j]).abs();
        max_r = max_r.max(r);
    }

    max_r
}

/// Prints a real vector with indices.
fn print_vector(label: &str, x: &[f64]) {
    println!("{label}");
    for (i, value) in x.iter().enumerate() {
        println!("  [{:>2}] = {:>18.10e}", i, value);
    }
    println!();
}

fn main() -> Result<(), String> {
    println!("Program 12.5.1: Canonical Sine Transform via FFT of an Odd Extension");
    println!();

    // ------------------------------------------------------------
    // Part 1: Compare the FFT-based DST with the direct definition.
    // ------------------------------------------------------------
    //
    // Here N = 8, so the interior points are j = 1, ..., 7.
    let f = vec![0.4, -0.2, 0.8, 1.1, -0.5, 0.3, 0.9];

    let s_direct = dst_canonical_direct(&f);
    let s_fft = dst_canonical_fft(&f)?;
    let transform_err = max_abs_error(&s_direct, &s_fft);

    print_vector("Interior data f_j, j = 1, ..., N-1:", &f);
    print_vector("Canonical sine coefficients S_k (direct):", &s_direct);
    print_vector("Canonical sine coefficients S_k (FFT-based):", &s_fft);
    println!(
        "Direct-versus-FFT DST check: max_k |S_direct - S_fft| = {:.6e}",
        transform_err
    );
    println!();

    // ------------------------------------------------------------
    // Part 2: Inverse-transform reconstruction.
    // ------------------------------------------------------------
    let f_recovered = dst_canonical_inverse(&s_fft)?;
    let inverse_err = max_abs_error(&f, &f_recovered);

    print_vector("Recovered interior data from inverse DST:", &f_recovered);
    println!(
        "Inverse DST reconstruction check: max_j |f_j - f_recovered,j| = {:.6e}",
        inverse_err
    );
    println!();

    // ------------------------------------------------------------
    // Part 3: Dirichlet Poisson solve using the canonical DST.
    // ------------------------------------------------------------
    //
    // Continuous test problem on (0,1):
    //   u(x) = sin(pi x) + 0.5 sin(3 pi x),
    // so
    //   -u''(x) = pi^2 sin(pi x) + 4.5 pi^2 sin(3 pi x).
    //
    // We solve the finite-difference system on the interior points and compare
    // with the exact continuous values sampled on the grid.
    let n = 16usize; // grid has points j = 0, ..., N
    let h = 1.0 / n as f64;

    let mut rhs = vec![0.0; n - 1];
    let mut u_exact = vec![0.0; n - 1];

    for j in 1..n {
        let x = j as f64 * h;
        u_exact[j - 1] = (PI * x).sin() + 0.5 * (3.0 * PI * x).sin();
        rhs[j - 1] = PI * PI * (PI * x).sin() + 4.5 * PI * PI * (3.0 * PI * x).sin();
    }

    let u_num = solve_poisson_dirichlet_dst(&rhs)?;
    let poisson_err = max_abs_error(&u_num, &u_exact);
    let poisson_res = max_residual_dirichlet(&u_num, &rhs, h);

    println!("Dirichlet Poisson solve on {} interior points:", n - 1);
    println!(
        "  max_j |u_numeric - u_exact| = {:.6e}",
        poisson_err
    );
    println!(
        "  max_j |(-u_{{j-1}} + 2u_j - u_{{j+1}})/h^2 - rhs_j| = {:.6e}",
        poisson_res
    );
    println!();

    // ------------------------------------------------------------
    // Part 4: Optional round-trip through the odd extension and inverse FFT.
    // ------------------------------------------------------------
    //
    // This confirms that the odd extension itself is represented exactly by
    // the FFT/inverse FFT pair up to roundoff.
    let n_ext = f.len() + 1;
    let m_ext = 2 * n_ext;
    let mut g = vec![C64::new(0.0, 0.0); m_ext];
    for j in 1..n_ext {
        g[j] = C64::new(f[j - 1], 0.0);
        g[m_ext - j] = C64::new(-f[j - 1], 0.0);
    }
    let g_hat = fft_forward(&g);
    let g_rec = fft_inverse(&g_hat);

    let extension_roundtrip_err = g
        .iter()
        .zip(g_rec.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max);

    println!(
        "Odd-extension FFT round-trip check: max_j |g_j - g_recovered,j| = {:.6e}",
        extension_roundtrip_err
    );

    Ok(())
}
```

Program 12.5.1 demonstrates how the canonical sine transform defined in equation (12.5.1) can be implemented efficiently through the odd-extension strategy described in equations (12.5.2)–(12.5.4). The comparison between the direct and FFT-based implementations confirms that the reduction reproduces the same transform coefficients to machine precision while reducing the computational complexity from $O(N^2)$ to $O(N\log N)$.

The Poisson solver example illustrates the deeper mathematical reason why sine transforms are so important in numerical PDE methods. Because the sine basis diagonalizes the discrete Laplacian under Dirichlet boundary conditions, the solution of the finite-difference Poisson equation reduces to a sequence of forward transforms, coefficient scalings, and inverse transforms. The small residual produced by the program confirms that the transformed system is solved accurately, while the remaining difference relative to the continuous exact solution reflects the second-order discretization error of the finite-difference approximation rather than any deficiency of the transform algorithm.

More broadly, this example highlights the computational principle emphasized throughout this chapter: many structured transforms arise naturally from symmetry considerations and can be implemented efficiently through reductions to the FFT. By exploiting odd symmetry, the discrete sine transform inherits the speed and numerical robustness of FFT algorithms, making it a fundamental tool in modern numerical solvers for boundary-value problems.

## 12.5.4. Cosine Transforms, Even Extensions, and Phase Corrections

Cosine transforms arise from even symmetry. An endpoint-including cosine transform aligned with Neumann boundary modeling can be written in one common form as

$$
C_k = \frac{1}{2}\left(f_0 + (-1)^k f_N\right)
     + \sum_{j=1}^{N-1} f_j \cos\left(\frac{\pi j k}{N}\right),
\qquad k = 0,1,\dots,N
\tag{12.5.5}
$$

A second extremely important cosine transform is the half-shifted form:

$$
C_k = \sum_{j=0}^{N-1} f_j \cos\left(\frac{\pi k (j+\tfrac{1}{2})}{N}\right),
\qquad k = 0,1,\dots,N-1
\tag{12.5.6}
$$

with inverse (one common normalization)

$$
f_j = \frac{1}{N}\sum_{k=0}^{N-1}{}' C_k
\cos\left(\frac{\pi k (j+\tfrac{1}{2})}{N}\right)
\tag{12.5.7}
$$

where the prime indicates the $k=0$ term is multiplied by $1/2$. Half-shifted cosine bases are ubiquitous in signal and image processing and arise naturally on staggered grids in PDE solvers (Pei and Tong, 2025).

Like the DST, the DCT can be computed in FFT time using even extension plus a twiddle-factor phase correction. For the half-shifted form, define an even-symmetric sequence of length $2N$,

$$
g_j =
\begin{cases}
f_j, & 0 \le j \le N-1, \\[4pt]
f_{2N-1-j}, & N \le j \le 2N-1 
\end{cases}
\tag{12.5.8}
$$

and compute $G_k = \mathrm{DFT}_{2N}(g)$. Then, after a phase correction,

$$
C_k = \operatorname{Re}\!\left(e^{-i\pi k/(2N)} G_k\right),\qquad k = 0,\dots,N-1
\tag{12.5.9}
$$

again up to normalization conventions. The computational pattern mirrors the real FFT of Section 12.4: one FFT plus inexpensive pre- and post-multiplications and extraction of real or imaginary parts.

The modeling consequence is practical. In mechanics and micromechanics pipelines, even and odd extensions can be used as boundary-condition machinery to retain FFT-like speed while moving beyond pure periodicity assumptions (Risthaus and Schneider, 2024; Paux et al., 2025).

### Rust Implementation

Following the discussion in Section 12.5.4 on cosine transforms, even extensions, and phase corrections, Program 12.5.2 provides a practical implementation of the half-shifted discrete cosine transform described in equations (12.5.6)–(12.5.9). The section showed that cosine transforms arise naturally from even symmetry and can therefore be computed efficiently by forming an even extension of the data, applying a single complex FFT, and extracting the cosine coefficients through a simple phase correction. This program demonstrates that construction explicitly. It begins by implementing the cosine transform directly from its definition for verification purposes, and then implements the FFT-based algorithm based on the even extension of equation (12.5.8) and the phase-correction step in equation (12.5.9). The program also provides an inverse transform corresponding to equation (12.5.7) and verifies numerical consistency by reconstructing the original signal from its cosine coefficients. Finally, it demonstrates a simple modal filtering operation and evaluates the transform on a smooth staggered-grid profile, illustrating how cosine bases isolate individual spectral modes. These examples highlight the computational pattern emphasized in this section: a cosine transform can be realized with FFT-level efficiency through symmetry extension and inexpensive postprocessing.

At the core of the implementation are two routines that compute the half-shifted cosine transform defined in equation (12.5.6). The function `dct2_direct` evaluates the transform directly from the definition by summing over the cosine basis functions for each coefficient index $k$. This implementation performs the transform exactly as written in the mathematical definition and therefore requires $O(N^2)$ operations. Its role in the program is primarily diagnostic: it provides a reference result against which the FFT-based implementation can be compared.

The function `dct2_fft` implements the fast transform using the symmetry-reduction strategy described in equations (12.5.8)–(12.5.9). The routine first constructs the even extension $g$ of length $2N$ defined in equation (12.5.8). This sequence reflects the input data across the boundary so that the resulting extended signal exhibits even symmetry. Because of this symmetry, the discrete Fourier transform of the extended sequence contains the cosine coefficients embedded in its real components. The function then computes the complex Fourier transform of the extended signal using the `rustfft` library. After the FFT is computed, the phase correction described in equation (12.5.9) is applied, and the real parts of the corrected spectrum yield the desired cosine coefficients. This reduction transforms the cost of computing the cosine transform from quadratic complexity to $O(N\log N)$.

To recover the original signal from the transform coefficients, the program implements the inverse transform `dct2_inverse`. This method follows the inverse relationship described in equation (12.5.7). The coefficient $C_0$ contributes with half weight relative to the other coefficients, reflecting the prime notation used in the equation. The inverse routine reconstructs the signal by summing the cosine basis functions with the appropriate normalization factor $1/N$. When combined with the forward transform, this routine provides a complete transform pair that allows round-trip reconstruction tests.

Several supporting functions are included to validate the numerical behavior of the algorithm. The functions `fft_forward` and `fft_inverse` provide wrappers around the FFT library routines used to compute the Fourier transforms required by the symmetry-reduction method. The function `max_abs_error` computes the maximum difference between vectors and is used to verify agreement between direct and FFT-based transforms and to confirm reconstruction accuracy. The helper routine `build_even_extension` constructs the extended signal used in the FFT reduction and allows the program to verify that the extension is preserved under forward and inverse FFT operations.

The `main` function organizes the program into several computational experiments that illustrate the theory described in the section. First, the program computes the cosine coefficients using both the direct definition and the FFT-based algorithm and verifies that the results agree to machine precision. Next, it reconstructs the original signal using the inverse transform and measures the reconstruction error. The program then performs a round-trip test on the even extension itself to confirm that the FFT implementation behaves correctly. To illustrate the practical use of cosine transforms in signal analysis, a simple modal low-pass filter is applied by truncating the cosine coefficients beyond a specified index and reconstructing the filtered signal. Finally, the transform is applied to a smooth staggered-grid profile composed of a small number of cosine modes. The resulting coefficient vector shows that only the corresponding spectral modes appear with significant amplitude, while the remaining coefficients vanish to roundoff. This experiment demonstrates the spectral selectivity of the cosine basis and verifies that the transform isolates individual cosine components as predicted by the theory.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
rustfft = "6"
```

```rust
// Program 12.5.2: Half-Shifted Discrete Cosine Transform via Even Extension
// and Phase Correction
//
// This program implements the half-shifted cosine transform of
// equations (12.5.6) to (12.5.9). The transform is computed by
// forming the even extension g of length 2N, applying a length-2N FFT,
// and extracting the cosine coefficients after a phase correction.
//
// The program includes:
// 1. A direct reference implementation of the half-shifted DCT.
// 2. An FFT-based implementation using the even extension.
// 3. An inverse transform consistent with equation (12.5.7).
// 4. A modal filtering example on a staggered-grid signal.
// 5. Diagnostic checks for transform accuracy and reconstruction.
//
// To run:
//   cargo run

use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

type C64 = Complex<f64>;

/// Forward complex FFT with the convention
/// X_k = sum_n x_n exp(-2 pi i n k / N).
fn fft_forward(input: &[C64]) -> Vec<C64> {
    let n = input.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer = input.to_vec();
    fft.process(&mut buffer);
    buffer
}

/// Inverse complex FFT with normalization by 1/N.
fn fft_inverse(input: &[C64]) -> Vec<C64> {
    let n = input.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_inverse(n);
    let mut buffer = input.to_vec();
    fft.process(&mut buffer);

    let scale = 1.0 / n as f64;
    for z in &mut buffer {
        *z *= scale;
    }

    buffer
}

/// Direct evaluation of the half-shifted cosine transform in equation (12.5.6).
///
/// Input:
///   f[j] corresponds to f_j, j = 0, ..., N-1.
///
/// Output:
///   c[k] corresponds to C_k, k = 0, ..., N-1.
fn dct2_direct(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let mut c = vec![0.0; n];

    for k in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            let theta = PI * (k as f64) * (j as f64 + 0.5) / (n as f64);
            sum += f[j] * theta.cos();
        }
        c[k] = sum;
    }

    c
}

/// FFT-based half-shifted cosine transform using the even extension
/// in equation (12.5.8) and the phase correction in equation (12.5.9).
///
/// For the particular forward convention used here,
///   G_k = 2 exp(i pi k / (2N)) C_k,
/// so
///   C_k = (1/2) Re( exp(-i pi k / (2N)) G_k ).
fn dct2_fft(f: &[f64]) -> Result<Vec<f64>, String> {
    if f.is_empty() {
        return Err("The half-shifted DCT requires a nonempty input vector.".to_string());
    }

    let n = f.len();
    let m = 2 * n;

    // Build the even extension g of length 2N:
    // g_j = f_j,               0 <= j <= N-1
    // g_j = f_{2N-1-j},        N <= j <= 2N-1
    let mut g = vec![C64::new(0.0, 0.0); m];
    for j in 0..n {
        g[j] = C64::new(f[j], 0.0);
        g[m - 1 - j] = C64::new(f[j], 0.0);
    }

    let g_hat = fft_forward(&g);

    let mut c = vec![0.0; n];
    for k in 0..n {
        let theta = -PI * (k as f64) / (2.0 * n as f64);
        let phase = C64::new(theta.cos(), theta.sin());
        c[k] = 0.5 * (phase * g_hat[k]).re;
    }

    Ok(c)
}

/// Inverse transform consistent with equation (12.5.7).
///
/// For the forward definition
///   C_k = sum_{j=0}^{N-1} f_j cos(pi k (j + 1/2) / N),
/// the inverse is
///   f_j = (1/N) [ C_0 + 2 sum_{k=1}^{N-1} C_k cos(pi k (j + 1/2) / N) ].
fn dct2_inverse(c: &[f64]) -> Result<Vec<f64>, String> {
    if c.is_empty() {
        return Err("The inverse half-shifted DCT requires at least one coefficient.".to_string());
    }

    let n = c.len();
    let mut f = vec![0.0; n];

    for j in 0..n {
        let mut sum = c[0];
        for k in 1..n {
            let theta = PI * (k as f64) * (j as f64 + 0.5) / (n as f64);
            sum += 2.0 * c[k] * theta.cos();
        }
        f[j] = sum / (n as f64);
    }

    Ok(f)
}

/// Applies a simple modal low-pass filter in DCT space.
/// Coefficients with index k > cutoff are set to zero.
fn dct2_lowpass_filter(f: &[f64], cutoff: usize) -> Result<Vec<f64>, String> {
    let mut c = dct2_fft(f)?;
    for (k, value) in c.iter_mut().enumerate() {
        if k > cutoff {
            *value = 0.0;
        }
    }
    dct2_inverse(&c)
}

/// Maximum absolute error between two real vectors.
fn max_abs_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Prints a real vector with indices.
fn print_vector(label: &str, x: &[f64]) {
    println!("{label}");
    for (i, value) in x.iter().enumerate() {
        println!("  [{:>2}] = {:>18.10e}", i, value);
    }
    println!();
}

/// Builds the even extension used by the FFT-based DCT.
fn build_even_extension(f: &[f64]) -> Vec<C64> {
    let n = f.len();
    let m = 2 * n;
    let mut g = vec![C64::new(0.0, 0.0); m];

    for j in 0..n {
        g[j] = C64::new(f[j], 0.0);
        g[m - 1 - j] = C64::new(f[j], 0.0);
    }

    g
}

fn main() -> Result<(), String> {
    println!("Program 12.5.2: Half-Shifted Discrete Cosine Transform via Even Extension and Phase Correction");
    println!();

    // ------------------------------------------------------------
    // Part 1: Compare the FFT-based DCT with the direct definition.
    // ------------------------------------------------------------
    let f = vec![1.2, -0.4, 0.8, 1.5, -0.2, 0.6, 0.9, -0.1];

    let c_direct = dct2_direct(&f);
    let c_fft = dct2_fft(&f)?;
    let transform_err = max_abs_error(&c_direct, &c_fft);

    print_vector("Input data f_j, j = 0, ..., N-1:", &f);
    print_vector("Half-shifted cosine coefficients C_k (direct):", &c_direct);
    print_vector("Half-shifted cosine coefficients C_k (FFT-based):", &c_fft);
    println!(
        "Direct-versus-FFT DCT check: max_k |C_direct - C_fft| = {:.6e}",
        transform_err
    );
    println!();

    // ------------------------------------------------------------
    // Part 2: Inverse-transform reconstruction.
    // ------------------------------------------------------------
    let f_recovered = dct2_inverse(&c_fft)?;
    let inverse_err = max_abs_error(&f, &f_recovered);

    print_vector("Recovered data from inverse DCT:", &f_recovered);
    println!(
        "Inverse DCT reconstruction check: max_j |f_j - f_recovered,j| = {:.6e}",
        inverse_err
    );
    println!();

    // ------------------------------------------------------------
    // Part 3: Verify the even extension and FFT round trip.
    // ------------------------------------------------------------
    let g = build_even_extension(&f);
    let g_hat = fft_forward(&g);
    let g_recovered = fft_inverse(&g_hat);

    let extension_roundtrip_err = g
        .iter()
        .zip(g_recovered.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max);

    println!(
        "Even-extension FFT round-trip check: max_j |g_j - g_recovered,j| = {:.6e}",
        extension_roundtrip_err
    );
    println!();

    // ------------------------------------------------------------
    // Part 4: Modal filtering example.
    // ------------------------------------------------------------
    //
    // This illustrates the practical use of the cosine basis on a signal
    // interpreted as cell-centered or staggered-grid data.
    let filtered = dct2_lowpass_filter(&f, 3)?;
    print_vector("Low-pass filtered reconstruction using modes k <= 3:", &filtered);

    // ------------------------------------------------------------
    // Part 5: A second example with a smooth staggered-grid profile.
    // ------------------------------------------------------------
    let n = 16usize;
    let smooth_signal: Vec<f64> = (0..n)
        .map(|j| {
            let x = (j as f64 + 0.5) / (n as f64);
            1.0 + 0.8 * (PI * x).cos() + 0.25 * (3.0 * PI * x).cos()
        })
        .collect();

    let smooth_coeffs = dct2_fft(&smooth_signal)?;
    let smooth_recovered = dct2_inverse(&smooth_coeffs)?;
    let smooth_err = max_abs_error(&smooth_signal, &smooth_recovered);

    print_vector("Smooth half-shifted input profile:", &smooth_signal);
    print_vector("DCT coefficients for the smooth profile:", &smooth_coeffs);
    println!(
        "Smooth-profile reconstruction check: max_j |f_j - f_recovered,j| = {:.6e}",
        smooth_err
    );

    Ok(())
}
```

Program 12.5.2 demonstrates how the half-shifted cosine transform can be implemented efficiently through the even-extension strategy described in Section 12.5.4. The comparison between the direct and FFT-based implementations confirms that the symmetry-based reduction reproduces the transform coefficients to machine precision while reducing the computational complexity from $O(N^2)$ to $O(N\log N)$.

The reconstruction test verifies that the forward and inverse transforms form a consistent pair under the chosen normalization, while the smooth-profile example illustrates the spectral interpretation of the cosine transform. Signals composed of a small number of cosine modes appear as isolated coefficients in the transform domain, with all other coefficients reduced to roundoff level. This behavior reflects the orthogonality of the cosine basis and explains why cosine transforms are widely used in spectral methods, signal processing pipelines, and PDE solvers on staggered grids.

More generally, the program highlights the computational principle emphasized throughout this chapter: many specialized transforms can be computed efficiently by exploiting symmetry and reducing the computation to a standard FFT followed by simple algebraic postprocessing. Even and odd extensions serve as a bridge between structured boundary-value formulations and the highly optimized FFT algorithms available in modern numerical libraries.

## 12.5.5 Recent Developments: Short-Length DST/DCT Kernels and Matrix Factorizations

For large $N$, FFT reductions via even or odd extension dominate. However, in many settings the transform sizes are small and fixed. Examples include codec kernels, embedded DSP blocks, and block smoothers in multigrid methods. In this regime, the performance bottleneck is not asymptotic scaling but minimizing the operation count and memory traffic of tiny transforms.

Recent work illustrates how DST and DCT kernels can be derived by explicitly factoring the transform matrix into sparse structured components using operations such as Kronecker products and direct sums. Bielak et al. (2024) develop fast DST-II algorithms for short-length inputs by such structured matrix decompositions. Polyakova et al. (2025) similarly derive fast DST-IV algorithms for short-length inputs and emphasize their role as building blocks for larger transforms and block processing.

The pedagogical message for numerical computing is therefore two-regime. For large $N$, treat DST and DCT as FFT-reducible transforms via symmetry extensions and twiddle factors. For tiny $N$, view DST and DCT as structured linear maps whose matrices can be factorized into minimal-operation kernels. Both viewpoints appear in current research because both correspond directly to practical performance constraints (Bielak et al., 2024; Polyakova et al., 2025).

*Implementation Perspective*: From an implementation standpoint, fast sine and cosine transforms in Rust typically follow one of two strategies. The FFT-reduction strategy constructs even or odd extensions, calls an FFT routine, applies phase corrections, and extracts real or imaginary components. This approach is conceptually clean, scales to arbitrary $N$ supported by the FFT backend, and aligns directly with the mathematics in (12.5.2)–(12.5.9). The dedicated-kernel strategy uses hand-optimized short-length factorization kernels derived from matrix decomposition, which is appropriate when transforms are repeatedly applied at small sizes and performance is critical (Bielak et al., 2024). For a textbook implementation, the reduction strategy is typically preferable for clarity, while kernel synthesis is best presented as an optional extension when discussing optimization and benchmarking.

### Rust Implementation

Following the discussion in Section 12.5.5 on short-length trigonometric transform kernels and structured matrix factorizations, Program 12.5.3 provides a practical implementation of small discrete cosine and sine transforms derived from explicit matrix factorizations. While large transforms typically inherit FFT speed through symmetry extensions and reductions to complex Fourier transforms, many practical applications involve small fixed transform sizes where minimizing arithmetic operations and memory movement is more important than asymptotic complexity. This program illustrates how short DCT-II and DST-II kernels can be constructed by factoring their transform matrices into simple structured stages, beginning with a butterfly-style symmetry decomposition followed by small mixing operations involving precomputed trigonometric constants. The implementation verifies the correctness of these kernels by comparing them with dense matrix formulations of the transforms, applies the kernels blockwise to longer vectors, and demonstrates how they can be composed into separable two-dimensional transforms. In this way the program illustrates the second regime described in Section 12.5.5, where trigonometric transforms are viewed not as FFT reductions but as compact structured linear maps suitable for high-performance short-length kernels.

At the core of the implementation is the representation of the discrete cosine and sine transforms as small matrix–vector operations. The functions `dct2_matrix_len4` and `dst2_matrix_len4` construct dense $4\times4$ matrices corresponding to the DCT-II and DST-II definitions discussed in Equations (12.5.6)–(12.5.7) and related transform families. These matrices provide a direct algebraic reference implementation of the transforms. The helper routine `mat_vec_4` performs dense matrix–vector multiplication so that the program can compute reference transform coefficients for verification.

The key computational idea introduced in this program is the *factorized kernel representation.* Instead of applying the dense transform matrix directly, the program decomposes the transform into a sequence of structured operations. The function `butterfly_len4` performs the initial symmetry decomposition stage, producing pairwise sums and differences of the input elements. This step reflects the symmetry structure underlying trigonometric transforms and corresponds to the first stage of many fast transform factorizations discussed in Section 12.5.5. By separating even and odd components of the input, the butterfly stage reduces the remaining computation to small mixing operations involving only a few arithmetic operations.

The functions `dct2_len4_kernel` and `dst2_len4_kernel` implement the complete short-length transform kernels. After the butterfly stage, each kernel computes the final transform coefficients using precomputed cosine or sine constants corresponding to the angles appearing in the transform definitions. These constants implement the small rotation and scaling steps that arise when factoring the transform matrix into sparse structured components. The resulting algorithm produces exactly the same transform values as the dense matrix formulation but uses a much smaller number of arithmetic operations, illustrating the operation-count reduction emphasized in recent short-transform research (Bielak et al., 2024; Polyakova et al., 2025).

To verify correctness, the program compares the output of the factorized kernels with the results obtained from dense matrix multiplication. The routines `max_abs_error_vec4` and `max_abs_error_mat4` compute maximum absolute differences between vectors or matrices, confirming that the factorized kernels reproduce the dense transform results to machine precision.

The implementation also demonstrates how short kernels can serve as building blocks in larger computations. The function `dct2_blockwise_kernel` applies the four-point DCT-II kernel repeatedly across blocks of a longer signal, illustrating how short transforms can be used in block processing pipelines such as codec kernels or multigrid smoothers. A dense reference version `dct2_blockwise_dense` is provided to verify correctness.

Finally, the program illustrates separability of multidimensional transforms. The routines `dct2_rows_4x4`, `dct2_cols_4x4`, and `dct2_2d_kernel_4x4` construct a two-dimensional transform by applying the one-dimensional kernel first along rows and then along columns. This mirrors the separability principle described earlier in the chapter for multidimensional Fourier transforms and demonstrates how short kernels can be composed into higher-dimensional transforms without explicitly constructing large matrices. The dense reference implementation `dct2_2d_dense_4x4` verifies that the separable kernel produces results identical to the dense formulation.

The `main` function serves as a structured demonstration of the algorithmic ideas discussed in Section 12.5.5. It first validates the factorized DCT-II and DST-II kernels against their dense matrix counterparts, then applies the kernels to blockwise signal processing and to a separable two-dimensional transform. Diagnostic output reports the maximum differences between dense and factorized implementations, demonstrating that the short kernels reproduce the exact transform results up to floating-point roundoff. This sequence of tests illustrates how structured kernel factorizations provide both computational efficiency and mathematical transparency.

```rust
// Program 12.5.3: Short-Length DCT-II and DST-II Kernels via Matrix Factorization
//
// This program illustrates the "dedicated-kernel" viewpoint discussed in
// Section 12.5.5. Instead of reducing a sine or cosine transform to a large
// FFT, it constructs explicit 4-point kernels using sparse butterfly stages
// and small dense 2x2 mixing operations.
//
// The program includes:
// 1. Dense 4x4 reference matrices for DCT-II and DST-II.
// 2. Hand-factorized 4-point DCT-II and DST-II kernels.
// 3. Verification against dense matrix-vector multiplication.
// 4. Blockwise application to a longer vector.
// 5. A separable 2D 4x4 DCT-II built from the 1D short kernel.
//
// To run:
//   cargo run

use std::f64::consts::{FRAC_1_SQRT_2, PI};

type Vec4 = [f64; 4];
type Mat4 = [[f64; 4]; 4];

fn mat_vec_4(a: &Mat4, x: &Vec4) -> Vec4 {
    let mut y = [0.0; 4];
    for i in 0..4 {
        for j in 0..4 {
            y[i] += a[i][j] * x[j];
        }
    }
    y
}

fn transpose_4(a: &Mat4) -> Mat4 {
    let mut t = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn mat_mul_4(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut c = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn max_abs_error_vec4(a: &Vec4, b: &Vec4) -> f64 {
    (0..4).map(|i| (a[i] - b[i]).abs()).fold(0.0_f64, f64::max)
}

fn max_abs_error_mat4(a: &Mat4, b: &Mat4) -> f64 {
    let mut err: f64 = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            err = err.max((a[i][j] - b[i][j]).abs());
        }
    }
    err
}

/// Dense 4-point DCT-II matrix with the convention
///   C_k = sum_{n=0}^{3} x_n cos(pi (n + 1/2) k / 4),  k = 0,1,2,3.
fn dct2_matrix_len4() -> Mat4 {
    let mut a = [[0.0; 4]; 4];
    for k in 0..4 {
        for n in 0..4 {
            a[k][n] = (PI * (n as f64 + 0.5) * (k as f64) / 4.0).cos();
        }
    }
    a
}

/// Dense 4-point DST-II matrix with the convention
///   S_k = sum_{n=0}^{3} x_n sin(pi (n + 1/2) (k + 1) / 4),  k = 0,1,2,3.
fn dst2_matrix_len4() -> Mat4 {
    let mut a = [[0.0; 4]; 4];
    for k in 0..4 {
        for n in 0..4 {
            a[k][n] = (PI * (n as f64 + 0.5) * (k as f64 + 1.0) / 4.0).sin();
        }
    }
    a
}

/// First butterfly stage:
///   u0 = x0 + x3
///   u1 = x1 + x2
///   u2 = x0 - x3
///   u3 = x1 - x2
fn butterfly_len4(x: &Vec4) -> Vec4 {
    [
        x[0] + x[3],
        x[1] + x[2],
        x[0] - x[3],
        x[1] - x[2],
    ]
}

/// Hand-factorized 4-point DCT-II kernel.
///
/// After the butterfly stage:
///   y0 = u0 + u1
///   y2 = (1/sqrt(2)) (u0 - u1)
///   y1 = cos(pi/8) u2 + cos(3pi/8) u3
///   y3 = cos(3pi/8) u2 - cos(pi/8) u3
fn dct2_len4_kernel(x: &Vec4) -> Vec4 {
    let u = butterfly_len4(x);

    let c1 = (PI / 8.0).cos();
    let c3 = (3.0 * PI / 8.0).cos();

    [
        u[0] + u[1],
        c1 * u[2] + c3 * u[3],
        FRAC_1_SQRT_2 * (u[0] - u[1]),
        c3 * u[2] - c1 * u[3],
    ]
}

/// Hand-factorized 4-point DST-II kernel.
///
/// After the butterfly stage:
///   y0 = sin(pi/8) u0 + sin(3pi/8) u1
///   y2 = sin(3pi/8) u0 - sin(pi/8) u1
///   y1 = (1/sqrt(2)) (u2 + u3)
///   y3 = u2 - u3
fn dst2_len4_kernel(x: &Vec4) -> Vec4 {
    let u = butterfly_len4(x);

    let s1 = (PI / 8.0).sin();
    let s3 = (3.0 * PI / 8.0).sin();

    [
        s1 * u[0] + s3 * u[1],
        FRAC_1_SQRT_2 * (u[2] + u[3]),
        s3 * u[0] - s1 * u[1],
        u[2] - u[3],
    ]
}

/// Applies the short DCT-II kernel blockwise to a vector whose length
/// is a multiple of 4.
fn dct2_blockwise_kernel(x: &[f64]) -> Result<Vec<f64>, String> {
    if x.len() % 4 != 0 {
        return Err("Blockwise DCT-II requires a length that is a multiple of 4.".to_string());
    }

    let mut y = vec![0.0; x.len()];
    for (block_idx, chunk) in x.chunks_exact(4).enumerate() {
        let x4 = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let y4 = dct2_len4_kernel(&x4);
        for j in 0..4 {
            y[4 * block_idx + j] = y4[j];
        }
    }
    Ok(y)
}

/// Dense blockwise DCT-II reference.
fn dct2_blockwise_dense(x: &[f64]) -> Result<Vec<f64>, String> {
    if x.len() % 4 != 0 {
        return Err("Blockwise DCT-II requires a length that is a multiple of 4.".to_string());
    }

    let a = dct2_matrix_len4();
    let mut y = vec![0.0; x.len()];
    for (block_idx, chunk) in x.chunks_exact(4).enumerate() {
        let x4 = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let y4 = mat_vec_4(&a, &x4);
        for j in 0..4 {
            y[4 * block_idx + j] = y4[j];
        }
    }
    Ok(y)
}

/// Applies the 4-point DCT-II kernel row-wise to a 4x4 block.
fn dct2_rows_4x4(x: &Mat4) -> Mat4 {
    let mut y = [[0.0; 4]; 4];
    for i in 0..4 {
        y[i] = dct2_len4_kernel(&x[i]);
    }
    y
}

/// Applies the 4-point DCT-II kernel column-wise to a 4x4 block
/// through a transpose strategy.
fn dct2_cols_4x4(x: &Mat4) -> Mat4 {
    let xt = transpose_4(x);
    let yt = dct2_rows_4x4(&xt);
    transpose_4(&yt)
}

/// Separable 2D 4x4 DCT-II using the factorized 1D short kernel.
fn dct2_2d_kernel_4x4(x: &Mat4) -> Mat4 {
    let y = dct2_rows_4x4(x);
    dct2_cols_4x4(&y)
}

/// Dense 2D 4x4 DCT-II reference:
/// Y = A X A^T, where A is the 1D DCT-II matrix.
fn dct2_2d_dense_4x4(x: &Mat4) -> Mat4 {
    let a = dct2_matrix_len4();
    let at = transpose_4(&a);
    let ax = mat_mul_4(&a, x);
    mat_mul_4(&ax, &at)
}

fn max_abs_error_slice(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(u, v)| (u - v).abs())
        .fold(0.0_f64, f64::max)
}

fn print_vec4(label: &str, x: &Vec4) {
    println!("{label}");
    for (i, value) in x.iter().enumerate() {
        println!("  [{:>1}] = {:>18.10e}", i, value);
    }
    println!();
}

fn print_mat4(label: &str, a: &Mat4) {
    println!("{label}");
    for row in a {
        for value in row {
            print!("{:>14.6e} ", value);
        }
        println!();
    }
    println!();
}

fn main() -> Result<(), String> {
    println!("Program 12.5.3: Short-Length DCT-II and DST-II Kernels via Matrix Factorization");
    println!();

    // ------------------------------------------------------------
    // Part 1: Verify the factorized DCT-II kernel against the dense matrix.
    // ------------------------------------------------------------
    let x: Vec4 = [0.75, -1.20, 0.50, 1.80];

    let dct_dense_mat = dct2_matrix_len4();
    let y_dct_dense = mat_vec_4(&dct_dense_mat, &x);
    let y_dct_kernel = dct2_len4_kernel(&x);
    let dct_err = max_abs_error_vec4(&y_dct_dense, &y_dct_kernel);

    print_vec4("Input vector for the 4-point DCT-II:", &x);
    print_vec4("DCT-II coefficients from dense matrix multiplication:", &y_dct_dense);
    print_vec4("DCT-II coefficients from the factorized short kernel:", &y_dct_kernel);
    println!(
        "Dense-versus-kernel DCT-II check: max error = {:.6e}",
        dct_err
    );
    println!();

    // ------------------------------------------------------------
    // Part 2: Verify the factorized DST-II kernel against the dense matrix.
    // ------------------------------------------------------------
    let dst_dense_mat = dst2_matrix_len4();
    let y_dst_dense = mat_vec_4(&dst_dense_mat, &x);
    let y_dst_kernel = dst2_len4_kernel(&x);
    let dst_err = max_abs_error_vec4(&y_dst_dense, &y_dst_kernel);

    print_vec4("DST-II coefficients from dense matrix multiplication:", &y_dst_dense);
    print_vec4("DST-II coefficients from the factorized short kernel:", &y_dst_kernel);
    println!(
        "Dense-versus-kernel DST-II check: max error = {:.6e}",
        dst_err
    );
    println!();

    // ------------------------------------------------------------
    // Part 3: Apply the DCT-II short kernel blockwise to a longer signal.
    // ------------------------------------------------------------
    let signal = vec![
        0.20, 0.50, -0.10, 1.10,
        0.90, -0.40, 0.30, 0.70,
        -0.60, 0.80, 1.20, -0.20,
        0.10, 0.00, -0.30, 0.40,
    ];

    let blockwise_dense = dct2_blockwise_dense(&signal)?;
    let blockwise_kernel = dct2_blockwise_kernel(&signal)?;
    let blockwise_err = max_abs_error_slice(&blockwise_dense, &blockwise_kernel);

    println!("Blockwise 4-point DCT-II applied to a length-16 signal:");
    println!(
        "  dense-versus-kernel maximum error = {:.6e}",
        blockwise_err
    );
    println!();

    // ------------------------------------------------------------
    // Part 4: Use the 4-point DCT-II kernel as a building block in 2D.
    // ------------------------------------------------------------
    let block: Mat4 = [
        [1.00, 1.20, 0.90, 0.70],
        [0.80, 1.10, 1.40, 1.20],
        [0.50, 0.60, 0.90, 1.30],
        [0.20, 0.30, 0.50, 0.80],
    ];

    let dct2_2d_dense = dct2_2d_dense_4x4(&block);
    let dct2_2d_kernel = dct2_2d_kernel_4x4(&block);
    let dct2_2d_err = max_abs_error_mat4(&dct2_2d_dense, &dct2_2d_kernel);

    print_mat4("Input 4x4 block:", &block);
    print_mat4("2D DCT-II from dense matrix products:", &dct2_2d_dense);
    print_mat4("2D DCT-II from the separable short kernel:", &dct2_2d_kernel);
    println!(
        "Dense-versus-kernel 2D DCT-II check: max error = {:.6e}",
        dct2_2d_err
    );
    println!();

    // ------------------------------------------------------------
    // Part 5: Show the butterfly stage explicitly.
    // ------------------------------------------------------------
    let u = butterfly_len4(&x);
    print_vec4("Butterfly stage [u0, u1, u2, u3]:", &u);
    println!("Here u0 and u1 are pairwise sums, while u2 and u3 are pairwise differences.");
    println!("This is the sparse structured stage that short-length transform factorizations exploit.");

    Ok(())
}
```

Program 12.5.3 demonstrates how short-length trigonometric transforms can be implemented using structured matrix factorizations rather than FFT reductions. The factorized kernels reproduce the exact DCT-II and DST-II transforms while using a sequence of simple butterfly and mixing operations that greatly reduce arithmetic cost. This approach illustrates the small-transform regime described in Section 12.5.5, where the goal is to minimize operation counts and memory traffic rather than asymptotic complexity.

The verification experiments confirm that the factorized kernels produce results identical to dense matrix formulations to within floating-point roundoff. The blockwise and two-dimensional examples further illustrate how these kernels can serve as building blocks for larger signal-processing or numerical algorithms. Such kernel-level optimizations are widely used in practical systems including audio and image codecs, embedded DSP pipelines, and block-based numerical solvers.

More broadly, the program illustrates the dual computational perspectives emphasized in this chapter. For large transform sizes, sine and cosine transforms are typically computed through FFT reductions using even or odd symmetry extensions. For small fixed transform sizes, however, it is often more efficient to treat the transforms as structured linear maps and derive minimal-operation kernels through matrix factorization. Both viewpoints arise naturally from the mathematical structure of trigonometric transform matrices and both remain important in modern numerical computing practice.

# 12.6 FFT in Two or More Dimensions

Let $x \in \mathbb{C}^{N_1 \times \cdots \times N_d}$ be a complex array on a $d$-dimensional rectangular grid. Using multi-indices $n=(n_1,\dots,n_d)$ and $k=(k_1,\dots,k_d)$, the $d$-D DFT is:

$$
X_{k_1,\dots,k_d}
=
\sum_{n_1=0}^{N_1-1} \cdots \sum_{n_d=0}^{N_d-1}
x_{n_1,\dots,n_d}
\exp\left(-2\pi i \sum_{r=1}^{d} \frac{n_r k_r}{N_r}\right)\tag{12.6.1}
$$

The defining kernel factors into a product of 1D kernels, so the transform is separable. Algorithmically, this means a multidimensional FFT can be carried out by applying 1D FFTs along one axis, then the next, and so on. The linear-algebra expression makes the same point in a way that is especially useful for reasoning about structure. If $\operatorname{vec}(x)$ stacks entries into a vector, then:

$$
\operatorname{vec}(X)
=
\left(W_{N_d} \otimes \cdots \otimes W_{N_1}\right)
\operatorname{vec}(x)
\tag{12.6.2}
$$

where $W_{N_r}$ is the 1D Fourier matrix and $\otimes$ denotes the Kronecker product. This tensor-product form is exactly the structural reason “FFT along each dimension” works, and it is the basis for modern algorithmic discussions of multidimensional transforms and their parallel decompositions (Koopman and Bisseling, 2023).

## 12.6.1. Complexity, Memory Traffic, and Communication

Let $N=\prod_{r=1}^d N_r$ be the total number of grid points. Applying 1D FFTs along each axis gives arithmetic cost,

$$
O\left(\sum_{r=1}^{d} N \log N_r\right)
\tag{12.6.3}
$$

which becomes $O(dN\log N)$ in the common equal-size case $N_1=\cdots=N_d$. In practice, however, multidimensional FFT performance is frequently limited less by arithmetic than by data movement. Transforming along a non-contiguous axis induces strided access, which increases cache misses and reduces effective bandwidth. Many implementations therefore interleave 1D FFT phases with explicit transposes or blocked permutations to restore contiguous access. In distributed memory, these permutations become global redistributions, and the resulting all-to-all collectives often dominate runtime at scale. Modern parallel FFT research therefore emphasizes communication-minimizing designs, including algorithms that reduce the number of all-to-all phases and can, under processor-count constraints, collapse redistribution to a single all-to-all step (Koopman and Bisseling, 2023). In accelerator-heavy workflows, precision choices also interact with throughput and stability. Recent work provides explicit first-order error propagation analyses for FFT-based GPU pipelines and uses them to reason about mixed-precision trade-offs at scale (Venkat et al., 2025). A practical way to internalize the shift is that in 1D the FFT is mostly “math,” while in large 3D runs it becomes “math plus orchestration,” where orchestration is memory movement and communication (Diez Sanhueza et al., 2025).

### Rust Implementation

Following the discussion of the separability of the multidimensional discrete Fourier transform and its tensor-product representation in equations (12.6.1) and (12.6.2), Program 12.6.1 provides a practical Rust implementation of a multidimensional FFT constructed from successive one-dimensional transforms. As emphasized in Section 12.6, the defining exponential kernel factors into products of one-dimensional kernels, allowing the full transform to be evaluated efficiently by applying 1D FFTs along each coordinate direction in turn. This separable structure is the algorithmic foundation behind modern multidimensional FFT libraries and explains why the transform complexity scales as $O\!\left(\sum_{r=1}^{d} N \log N_r\right)$, as described in equation (12.6.3). The program demonstrates this principle using dynamic multidimensional arrays implemented with the `ndarray` crate and the highly optimized `rustfft` library for one-dimensional Fourier transforms. In addition to computing forward and inverse multidimensional FFTs, the program includes numerical verification of reconstruction accuracy and illustrates the Hermitian symmetry that arises when the input data are real-valued. Together, these components translate the mathematical structure of the multidimensional DFT into a concrete and executable algorithm.

At the core of the implementation is the function `fft_along_axis_inplace`, which performs a one-dimensional FFT along a specified axis of a multidimensional array. This routine embodies the separability property expressed in equation (12.6.1): instead of evaluating the full multidimensional summation directly, the transform is computed by processing one coordinate direction at a time. For each axis, the function extracts a sequence of elements aligned with that dimension, applies a one-dimensional FFT using the `rustfft` planner, and writes the transformed values back into the array. Because many such sequences are not contiguous in memory, temporary buffers are used during transformation. This design reflects the practical considerations discussed in Section 12.6.1, where multidimensional FFT performance is often governed not only by arithmetic complexity but also by memory access patterns and data movement.

The functions `fft_nd` and `ifft_nd` assemble the complete multidimensional transform by repeatedly invoking the axis-wise routine. This sequence of operations mirrors the tensor-product formulation given in equation (12.6.2), where the multidimensional Fourier matrix can be expressed as a Kronecker product of one-dimensional Fourier matrices. The forward transform therefore consists of applying one-dimensional FFTs along each axis in succession, while the inverse transform performs the corresponding inverse operations followed by normalization by the total number of grid points. This normalization ensures that applying the inverse transform to the forward transform recovers the original data, a property verified numerically in the program.

To demonstrate the algorithm, the program constructs test arrays using the helper functions `build_test_array` and `build_real_embedded_example`. The first generates a synthetic complex-valued multidimensional field with oscillatory structure across all coordinates, ensuring that the transform operates on nontrivial data. The second creates a real-valued signal embedded in complex form, enabling the program to illustrate the Hermitian symmetry property of the Fourier spectrum for real input data. According to equation (12.6.5), the spectrum satisfies a conjugate symmetry relation across reflected indices. The function `hermitian_symmetry_residual_3d` evaluates this relation numerically by comparing each spectral coefficient with the conjugate coefficient at its mirrored multi-index, providing a diagnostic measure of the symmetry.

Several additional utility functions support the demonstration. The routine `linear_to_multi_index` converts flat row-major indices into multidimensional coordinates, reflecting the storage layout described in equation (12.6.4). The function `max_abs_error` computes the maximum difference between two arrays and is used to verify the reconstruction accuracy of the inverse transform. Finally, `print_sample_entries` prints representative values from arrays so that the program’s output can be inspected easily when executed. Together, these components create a complete example that connects the mathematical formulation of multidimensional FFTs with their practical implementation.

The `main` function serves as the driver for the demonstration. It first constructs a small three-dimensional complex array, computes its forward multidimensional FFT, and then applies the inverse transform. The recovered array is compared with the original input, and the maximum reconstruction error is reported. This value typically lies near machine precision, confirming that the forward and inverse transforms are implemented consistently. The program then evaluates a real-valued three-dimensional signal and computes its Fourier spectrum. By measuring the deviation from the Hermitian symmetry relation described in equation (12.6.5), the program confirms numerically that the computed spectrum exhibits the expected symmetry structure for real input data.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
ndarray = "0.15"
rustfft = "6"
```

```rust
// Program 12.6.1: Multidimensional FFT via Successive One-Dimensional Transforms
//
// This program implements a d-dimensional FFT on a rectangular grid by applying
// 1D FFTs along each axis in succession, reflecting the separable structure in
// Eqs. (12.6.1) and (12.6.2).
//
// Cargo.toml dependencies:
// [dependencies]
// ndarray = "0.15"
// rustfft = "6"

use ndarray::{ArrayD, Axis, IxDyn};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::error::Error;
use std::f64::consts::PI;

/// Construct a small synthetic d-dimensional complex array.
/// The entries are chosen to vary across each coordinate so that the example
/// exercises all axes of the multidimensional transform.
fn build_test_array(shape: &[usize]) -> ArrayD<Complex<f64>> {
    let total_len: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total_len);

    // Convert each flat index into a multi-index in row-major order and build
    // a complex value from simple trigonometric patterns.
    for linear_idx in 0..total_len {
        let multi = linear_to_multi_index(linear_idx, shape);

        let mut real = 0.0;
        let mut imag = 0.0;
        for (r, &n_r) in multi.iter().enumerate() {
            let scale = (r + 1) as f64;
            real += ((n_r as f64 + 1.0) * scale * PI / 7.0).cos();
            imag += ((n_r as f64 + 1.0) * scale * PI / 5.0).sin();
        }

        data.push(Complex::new(real, imag));
    }

    ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape and data length must agree")
}

/// Convert a flat row-major index into a multi-index.
fn linear_to_multi_index(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let d = shape.len();
    let mut multi = vec![0usize; d];

    for r in (0..d).rev() {
        multi[r] = idx % shape[r];
        idx /= shape[r];
    }

    multi
}

/// Apply an in-place 1D FFT or inverse FFT along a single axis of a dynamic array.
///
/// Because many lanes are not contiguous in memory, each lane is copied to a
/// temporary buffer, transformed, and written back. This mirrors the data-movement
/// issue discussed around Eq. (12.6.3): arithmetic is not the only cost in
/// multidimensional FFTs.
fn fft_along_axis_inplace(data: &mut ArrayD<Complex<f64>>, axis: usize, inverse: bool) {
    let axis_len = data.shape()[axis];
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(axis_len)
    } else {
        planner.plan_fft_forward(axis_len)
    };

    for mut lane in data.lanes_mut(Axis(axis)) {
        let mut buffer: Vec<Complex<f64>> = lane.iter().copied().collect();
        fft.process(&mut buffer);
        for (dst, src) in lane.iter_mut().zip(buffer.into_iter()) {
            *dst = src;
        }
    }
}

/// Apply the forward d-dimensional FFT by performing successive 1D FFTs along
/// every axis.
fn fft_nd(input: &ArrayD<Complex<f64>>) -> ArrayD<Complex<f64>> {
    let mut output = input.clone();
    let ndim = output.ndim();

    for axis in 0..ndim {
        fft_along_axis_inplace(&mut output, axis, false);
    }

    output
}

/// Apply the inverse d-dimensional FFT and normalize by the total number of points.
fn ifft_nd(input: &ArrayD<Complex<f64>>) -> ArrayD<Complex<f64>> {
    let mut output = input.clone();
    let ndim = output.ndim();

    for axis in 0..ndim {
        fft_along_axis_inplace(&mut output, axis, true);
    }

    let scale = output.len() as f64;
    for x in output.iter_mut() {
        *x /= scale;
    }

    output
}

/// Compute the maximum absolute entrywise error between two arrays.
fn max_abs_error(a: &ArrayD<Complex<f64>>, b: &ArrayD<Complex<f64>>) -> f64 {
    assert_eq!(a.shape(), b.shape(), "array shapes must match");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0_f64, f64::max)
}

/// Print a few entries of a dynamic array for inspection.
fn print_sample_entries(name: &str, array: &ArrayD<Complex<f64>>, max_entries: usize) {
    println!("{name} (shape = {:?})", array.shape());
    for (i, value) in array.iter().take(max_entries).enumerate() {
        println!("  [{i:02}] = {:+.6} {:+.6}i", value.re, value.im);
    }
    if array.len() > max_entries {
        println!("  ...");
    }
    println!();
}

/// Build a simple real-valued 3D array embedded in complex arithmetic.
/// This gives a concrete example of the multidimensional grid setting
/// described in Eq. (12.6.1).
fn build_real_embedded_example(shape: &[usize]) -> ArrayD<Complex<f64>> {
    let total_len: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total_len);

    for linear_idx in 0..total_len {
        let multi = linear_to_multi_index(linear_idx, shape);

        let x = multi[0] as f64;
        let y = multi[1] as f64;
        let z = multi[2] as f64;

        let value = (2.0 * PI * x / shape[0] as f64).cos()
            + 0.5 * (2.0 * PI * y / shape[1] as f64).sin()
            + 0.25 * (2.0 * PI * z / shape[2] as f64).cos();

        data.push(Complex::new(value, 0.0));
    }

    ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape and data length must agree")
}

/// Estimate a simple Hermitian-symmetry residual for a real-valued 3D input.
/// For real data, the d-dimensional spectrum should satisfy Eq. (12.6.5).
fn hermitian_symmetry_residual_3d(spectrum: &ArrayD<Complex<f64>>) -> f64 {
    assert_eq!(spectrum.ndim(), 3, "this helper is written for 3D arrays");

    let shape = spectrum.shape();
    let (n1, n2, n3) = (shape[0], shape[1], shape[2]);
    let mut max_residual: f64 = 0.0;

    for k1 in 0..n1 {
        for k2 in 0..n2 {
            for k3 in 0..n3 {
                let j1 = (n1 - k1) % n1;
                let j2 = (n2 - k2) % n2;
                let j3 = (n3 - k3) % n3;

                let a = spectrum[[k1, k2, k3]];
                let b = spectrum[[j1, j2, j3]].conj();
                max_residual = max_residual.max((a - b).norm());
            }
        }
    }

    max_residual
}

fn main() -> Result<(), Box<dyn Error>> {
    // Example 1: General complex-valued 3D array.
    let shape = [4, 3, 5];
    let x = build_test_array(&shape);
    let x_hat = fft_nd(&x);
    let x_recovered = ifft_nd(&x_hat);

    println!("=== Example 1: Complex-valued 3D FFT ===\n");
    print_sample_entries("Input x", &x, 8);
    print_sample_entries("Spectrum X = FFT_d(x)", &x_hat, 8);
    print_sample_entries("Recovered x from inverse FFT", &x_recovered, 8);

    let recon_error = max_abs_error(&x, &x_recovered);
    println!("Maximum reconstruction error: {:.3e}\n", recon_error);

    // Example 2: Real-valued 3D data embedded in complex storage.
    // This illustrates the multidimensional analogue of Hermitian symmetry.
    let real_shape = [4, 4, 4];
    let x_real = build_real_embedded_example(&real_shape);
    let x_real_hat = fft_nd(&x_real);
    let hermitian_residual = hermitian_symmetry_residual_3d(&x_real_hat);

    println!("=== Example 2: Real-valued 3D input and Hermitian symmetry ===\n");
    print_sample_entries("Real-input data x", &x_real, 8);
    print_sample_entries("Spectrum X = FFT_d(x)", &x_real_hat, 8);
    println!(
        "Maximum Hermitian-symmetry residual: {:.3e}",
        hermitian_residual
    );

    if recon_error > 1e-9 {
        return Err(format!(
            "inverse FFT reconstruction error too large: {:.3e}",
            recon_error
        )
        .into());
    }

    Ok(())
}
```

Program 12.6.1 demonstrates a practical realization of multidimensional FFT computation based on the separable structure of the discrete Fourier transform. By decomposing the transform into successive one-dimensional operations, the program directly implements the algorithmic principle derived from equations (12.6.1) and (12.6.2). The numerical experiments confirm two essential properties: the forward and inverse transforms reconstruct the original data with errors at the level of floating-point roundoff, and real-valued input signals produce spectra exhibiting the Hermitian symmetry described in equation (12.6.5).

The example also illustrates an important practical aspect emphasized in Section 12.6.1. Although the arithmetic complexity of multidimensional FFTs is well understood, real-world performance is strongly influenced by memory layout and data movement. Axis-wise transforms that operate on noncontiguous memory require temporary buffering and introduce additional overhead, highlighting why many high-performance FFT implementations employ transpositions or blocked data layouts to maintain contiguous access patterns.

The modular structure of the implementation makes it straightforward to extend the framework to larger multidimensional grids, real-input FFT variants with half-spectrum storage as described in equation (12.6.6), or distributed-memory implementations for large-scale simulations. Such extensions form the basis of many modern scientific computing workflows, including spectral solvers for partial differential equations and FFT-based imaging algorithms.

## 12.6.2. Data Layout in Practice and Real-Valued Multidimensional FFTs

Assume a 2D array $x\in\mathbb{C}^{N_1\times N_2}$ stored in row-major order (C-like), so the last index varies fastest. The linear indexing map is:

$$
\mathrm{offset}(n_1,n_2) = n_2 + N_2 n_1
\tag{12.6.4}
$$

FFTs along the second axis $(n_2)$ are contiguous and cache-friendly, while FFTs along the first axis $(n_1)$ are strided by $N_2$. The same idea generalizes: in row-major storage, the last index is the most FFT-friendly, and earlier indices become progressively more strided, motivating transpose-based or blocked strategies in high-performance multidimensional implementations (Diez Sanhueza et al., 2025; Venkat et al., 2025). These layout constraints are especially concrete in Rust, since common array representations follow row-major conventions, so “which axis is contiguous” is a predictable and exploitable property.

When the input is real-valued on a $d$-D grid, Hermitian symmetry generalizes in the natural way:

$$
X_{(-k_1 \bmod N_1),\dots,(-k_d \bmod N_d)}
=
\overline{X_{k_1,\dots,k_d}}
\tag{12.6.5}
$$

As in 1D, this reduces the number of independent frequency degrees of freedom. In practice, libraries typically store only a half-spectrum along one axis (often the last), producing an output shape,

$$
N_1 \times \cdots \times N_{d-1} \times \left(\left\lfloor \frac{N_d}{2} \right\rfloor + 1\right)
\tag{12.6.6}
$$

which is the multidimensional analogue of “store nonnegative frequencies only” from Section 12.4 (Salih and Hamood, 2023; Liu et al., 2025). The engineering message is that data layout and symmetry interact: choosing the half-spectrum along the last axis is attractive precisely because the last axis is also the most contiguous in row-major storage.

### Rust Implementation

Following the discussion in Section 12.6.2 on the role of memory layout and symmetry in multidimensional FFT computations, Program 12.6.2 provides a practical Rust implementation illustrating how row-major data organization influences transform performance and how Hermitian symmetry can be exploited for real-valued multidimensional signals. As described by the indexing relation in equation (12.6.4), arrays stored in row-major order make the last coordinate contiguous in memory, allowing FFT operations along that axis to proceed with cache-friendly access patterns. Earlier axes, by contrast, require strided access and therefore introduce additional memory movement. The program demonstrates these layout effects explicitly for a two-dimensional array and then implements multidimensional FFT computations that exploit the separable structure described in Section 12.6. In addition, it shows how real-valued input data lead to conjugate symmetry in the frequency domain, as described by equation (12.6.5), allowing only a reduced half-spectrum to be stored along one axis according to equation (12.6.6). By combining row-major indexing demonstrations, axis-wise transforms, and real-to-complex FFT operations, the program translates the theoretical discussion of data layout and spectral symmetry into an executable computational example.

At the core of the implementation is the explicit representation of row-major indexing through the function `row_major_offset`, which evaluates the linear address mapping described in equation (12.6.4). Given a two-dimensional coordinate pair $(n_1,n_2)$, this function returns the corresponding offset in a contiguous memory buffer. The routines `print_row_major_offsets` and `print_axis_access_patterns` use this mapping to display how memory addresses change when traversing different axes of the array. These demonstrations illustrate a key performance principle emphasized earlier in the section: movement along the last index corresponds to contiguous memory access, whereas movement along earlier indices produces strided access patterns.

The functions `fft_along_last_axis` and `fft_along_first_axis` implement one-dimensional complex FFTs along the two coordinate directions of a two-dimensional array. The first routine processes rows, which are contiguous in row-major storage, allowing the FFT to operate directly on the extracted row data. The second routine performs transforms along columns. Since columns are not contiguous in row-major storage, the implementation gathers the column elements into a temporary buffer, applies the FFT, and then writes the transformed values back into the array. This gather–transform–scatter pattern directly reflects the data-movement considerations discussed in Section 12.6.1, where multidimensional FFT performance is often determined by memory traffic rather than arithmetic complexity.

The function `fft2_complex` combines these axis-wise transforms to compute a full two-dimensional complex FFT. It first applies the transform along the last axis and then along the first axis, implementing the separable multidimensional transform structure described earlier in the chapter. This sequential application of one-dimensional transforms mirrors the tensor-product formulation introduced in equation (12.6.2) and illustrates how multidimensional FFT algorithms are constructed in practice.

To demonstrate the reduction in storage requirements for real-valued signals, the program includes the function `rfft2_half_last_axis`. This routine computes a two-dimensional real-to-complex transform in which only the nonredundant frequency coefficients are stored. As predicted by equation (12.6.5), the spectrum of a real-valued signal exhibits Hermitian symmetry, meaning that negative-frequency coefficients are determined by the complex conjugates of corresponding positive-frequency coefficients. Consequently, only half of the spectrum along the last axis must be retained. The function therefore produces an output array whose final dimension equals $\lfloor N_d/2 \rfloor + 1$, consistent with the reduced storage form described in equation (12.6.6).

Several auxiliary routines support the numerical demonstration. The functions `build_complex_2d` and `build_real_3d` generate synthetic multidimensional signals with oscillatory components so that the resulting spectra contain nontrivial frequency content. The function `fft3_complex_from_real` computes a full three-dimensional complex FFT of real-valued data by embedding the real input in complex form and applying successive axis transforms. The function `hermitian_symmetry_residual_3d` then evaluates the symmetry relation described in equation (12.6.5) by comparing each spectral coefficient with the conjugate coefficient at the reflected multi-index. The maximum deviation between these paired coefficients provides a numerical measure of how closely the computed spectrum satisfies the expected symmetry.

The `main` function coordinates the demonstration in three stages. First, it prints row-major offsets and axis traversal patterns to visualize how contiguous and strided memory accesses arise in practice. Second, it constructs a two-dimensional complex array and computes its FFT using successive axis transforms, illustrating how the multidimensional transform is assembled from one-dimensional operations. Third, it computes a real-to-complex multidimensional FFT and verifies both the half-spectrum storage rule in equation (12.6.6) and the Hermitian symmetry relation in equation (12.6.5). Together, these examples illustrate how theoretical properties of the multidimensional DFT translate directly into practical implementation strategies.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
ndarray = "0.15"
rustfft = "6"
realfft = "3"
```

```rust
// Program 12.6.2: Data Layout in Practice and Real-Valued Multidimensional FFTs
//
// This program illustrates two central ideas from Section 12.6.2:
//
// 1. In row-major storage, the last axis is contiguous and therefore the most
//    FFT-friendly. The first axis is strided, so FFTs along that direction
//    require gather/scatter style access or an explicit transpose.
// 2. For real-valued multidimensional input, the Fourier spectrum satisfies the
//    Hermitian symmetry of Eq. (12.6.5). Hence only a half-spectrum along one
//    axis needs to be stored, as in Eq. (12.6.6).
//
// The implementation uses:
// - ndarray  for row-major multidimensional arrays
// - rustfft  for complex 1D FFTs
// - realfft  for real-to-complex FFTs that return the half-spectrum directly
//
// Cargo.toml dependencies:
// [dependencies]
// ndarray = "0.15"
// rustfft = "6"
// realfft = "3"

use ndarray::{Array2, Array3};
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::error::Error;
use std::f64::consts::PI;

/// Row-major offset for a 2D array, as in Eq. (12.6.4):
/// offset(n1, n2) = n2 + N2 * n1
fn row_major_offset(n1: usize, n2: usize, n2_dim: usize) -> usize {
    n2 + n2_dim * n1
}

/// Build a 2D complex-valued test array whose entries vary in both coordinates.
fn build_complex_2d(n1: usize, n2: usize) -> Array2<Complex<f64>> {
    let mut a = Array2::<Complex<f64>>::zeros((n1, n2));
    for i in 0..n1 {
        for j in 0..n2 {
            let re = (2.0 * PI * i as f64 / n1 as f64).cos()
                + 0.35 * (2.0 * PI * j as f64 / n2 as f64).sin();
            let im = 0.5 * (2.0 * PI * (i + 2 * j) as f64 / (n1 + n2) as f64).cos();
            a[[i, j]] = Complex::new(re, im);
        }
    }
    a
}

/// Build a 3D real-valued array used to demonstrate the multidimensional real FFT.
fn build_real_3d(n1: usize, n2: usize, n3: usize) -> Array3<f64> {
    let mut a = Array3::<f64>::zeros((n1, n2, n3));
    for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..n3 {
                let value = (2.0 * PI * i as f64 / n1 as f64).cos()
                    + 0.75 * (2.0 * PI * j as f64 / n2 as f64).sin()
                    + 0.5 * (2.0 * PI * k as f64 / n3 as f64).cos();
                a[[i, j, k]] = value;
            }
        }
    }
    a
}

/// Print the row-major offsets of a small 2D array so the storage layout is explicit.
fn print_row_major_offsets(n1: usize, n2: usize) {
    println!("Row-major offsets for a {} x {} array:", n1, n2);
    for i in 0..n1 {
        for j in 0..n2 {
            let off = row_major_offset(i, j, n2);
            print!("{:3} ", off);
        }
        println!();
    }
    println!();
}

/// Demonstrate which axis is contiguous in row-major storage by printing
/// the flat offsets visited when traversing a fixed row or a fixed column.
fn print_axis_access_patterns(n1: usize, n2: usize, fixed_row: usize, fixed_col: usize) {
    println!("Access pattern along the last axis (row {}, varying column):", fixed_row);
    for j in 0..n2 {
        let off = row_major_offset(fixed_row, j, n2);
        print!("{:3} ", off);
    }
    println!("\nThese offsets are contiguous.\n");

    println!("Access pattern along the first axis (column {}, varying row):", fixed_col);
    for i in 0..n1 {
        let off = row_major_offset(i, fixed_col, n2);
        print!("{:3} ", off);
    }
    println!("\nThese offsets are strided by N2 = {}.\n", n2);
}

/// Apply a 1D complex FFT along the contiguous last axis of a 2D array.
/// Each row is contiguous in row-major storage.
fn fft_along_last_axis(a: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (n1, n2) = a.dim();
    let mut out = a.clone();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n2);

    for i in 0..n1 {
        let mut row: Vec<Complex<f64>> = (0..n2).map(|j| out[[i, j]]).collect();
        fft.process(&mut row);
        for j in 0..n2 {
            out[[i, j]] = row[j];
        }
    }

    out
}

/// Apply a 1D complex FFT along the strided first axis of a 2D array.
/// Because columns are not contiguous in row-major storage, we gather each
/// column into a temporary buffer, transform it, and scatter the result back.
fn fft_along_first_axis(a: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (n1, n2) = a.dim();
    let mut out = a.clone();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n1);

    for j in 0..n2 {
        let mut col: Vec<Complex<f64>> = (0..n1).map(|i| out[[i, j]]).collect();
        fft.process(&mut col);
        for i in 0..n1 {
            out[[i, j]] = col[i];
        }
    }

    out
}

/// Compute a full 2D complex FFT by successive transforms along the last axis
/// and then the first axis.
fn fft2_complex(a: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let tmp = fft_along_last_axis(a);
    fft_along_first_axis(&tmp)
}

/// Compute a 2D real-to-complex FFT that stores only the half-spectrum along
/// the last axis. The output shape is (n1, floor(n2/2) + 1), matching Eq. (12.6.6).
fn rfft2_half_last_axis(a: &Array2<f64>) -> Result<Array2<Complex<f64>>, Box<dyn Error>> {
    let (n1, n2) = a.dim();

    // First stage: apply real-to-complex FFT along the last (contiguous) axis.
    let mut real_planner = RealFftPlanner::<f64>::new();
    let r2c = real_planner.plan_fft_forward(n2);

    let half_n2 = n2 / 2 + 1;
    let mut stage1 = Array2::<Complex<f64>>::zeros((n1, half_n2));

    for i in 0..n1 {
        let mut input_row: Vec<f64> = (0..n2).map(|j| a[[i, j]]).collect();
        let mut output_row = r2c.make_output_vec();
        r2c.process(&mut input_row, &mut output_row)?;
        for j in 0..half_n2 {
            stage1[[i, j]] = output_row[j];
        }
    }

    // Second stage: apply complex FFTs along the first axis for each retained frequency.
    let mut planner = FftPlanner::<f64>::new();
    let fft_n1 = planner.plan_fft_forward(n1);

    for j in 0..half_n2 {
        let mut col: Vec<Complex<f64>> = (0..n1).map(|i| stage1[[i, j]]).collect();
        fft_n1.process(&mut col);
        for i in 0..n1 {
            stage1[[i, j]] = col[i];
        }
    }

    Ok(stage1)
}

/// Compute a full 3D complex FFT of real data embedded in complex form.
/// This is used only to verify Hermitian symmetry numerically.
fn fft3_complex_from_real(a: &Array3<f64>) -> Array3<Complex<f64>> {
    let (n1, n2, n3) = a.dim();
    let mut out = Array3::<Complex<f64>>::zeros((n1, n2, n3));

    for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..n3 {
                out[[i, j, k]] = Complex::new(a[[i, j, k]], 0.0);
            }
        }
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft_n3 = planner.plan_fft_forward(n3);
    let fft_n2 = planner.plan_fft_forward(n2);
    let fft_n1 = planner.plan_fft_forward(n1);

    // Transform along the last axis (contiguous).
    for i in 0..n1 {
        for j in 0..n2 {
            let mut line: Vec<Complex<f64>> = (0..n3).map(|k| out[[i, j, k]]).collect();
            fft_n3.process(&mut line);
            for k in 0..n3 {
                out[[i, j, k]] = line[k];
            }
        }
    }

    // Transform along the second axis.
    for i in 0..n1 {
        for k in 0..n3 {
            let mut line: Vec<Complex<f64>> = (0..n2).map(|j| out[[i, j, k]]).collect();
            fft_n2.process(&mut line);
            for j in 0..n2 {
                out[[i, j, k]] = line[j];
            }
        }
    }

    // Transform along the first axis.
    for j in 0..n2 {
        for k in 0..n3 {
            let mut line: Vec<Complex<f64>> = (0..n1).map(|i| out[[i, j, k]]).collect();
            fft_n1.process(&mut line);
            for i in 0..n1 {
                out[[i, j, k]] = line[i];
            }
        }
    }

    out
}

/// Measure the Hermitian-symmetry residual for a full 3D spectrum, as in Eq. (12.6.5).
fn hermitian_symmetry_residual_3d(spec: &Array3<Complex<f64>>) -> f64 {
    let (n1, n2, n3) = spec.dim();
    let mut max_residual = 0.0_f64;

    for k1 in 0..n1 {
        for k2 in 0..n2 {
            for k3 in 0..n3 {
                let j1 = (n1 - k1) % n1;
                let j2 = (n2 - k2) % n2;
                let j3 = (n3 - k3) % n3;

                let a = spec[[k1, k2, k3]];
                let b = spec[[j1, j2, j3]].conj();
                max_residual = max_residual.max((a - b).norm());
            }
        }
    }

    max_residual
}

/// Print a few entries of a 2D complex array.
fn print_sample_complex_2d(name: &str, a: &Array2<Complex<f64>>, max_i: usize, max_j: usize) {
    let (n1, n2) = a.dim();
    println!("{name} (shape = [{}, {}])", n1, n2);
    for i in 0..n1.min(max_i) {
        for j in 0..n2.min(max_j) {
            let z = a[[i, j]];
            print!("({:+8.4},{:+8.4}i) ", z.re, z.im);
        }
        println!();
    }
    println!();
}

/// Print a few entries of a 3D complex array in flat order.
fn print_sample_complex_3d(name: &str, a: &Array3<Complex<f64>>, count: usize) {
    let (n1, n2, n3) = a.dim();
    println!("{name} (shape = [{}, {}, {}])", n1, n2, n3);

    let mut shown = 0usize;
    'outer: for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..n3 {
                let z = a[[i, j, k]];
                println!(
                    "  [{:02},{:02},{:02}] = {:+.6} {:+.6}i",
                    i, j, k, z.re, z.im
                );
                shown += 1;
                if shown >= count {
                    break 'outer;
                }
            }
        }
    }
    println!();
}

fn main() -> Result<(), Box<dyn Error>> {
    // -------------------------------------------------------------------------
    // Part 1: Data layout in a 2D row-major array.
    // -------------------------------------------------------------------------
    let (n1, n2) = (4usize, 6usize);

    println!("=== Part 1: Row-Major Data Layout and Axis Access ===\n");
    print_row_major_offsets(n1, n2);
    print_axis_access_patterns(n1, n2, 1, 2);

    let x2 = build_complex_2d(n1, n2);
    print_sample_complex_2d("Input 2D complex array", &x2, 4, 4);

    let last_axis_fft = fft_along_last_axis(&x2);
    print_sample_complex_2d(
        "After FFT along last axis (contiguous rows)",
        &last_axis_fft,
        4,
        4,
    );

    let full_fft2 = fft2_complex(&x2);
    print_sample_complex_2d(
        "Full 2D FFT via successive last-axis and first-axis transforms",
        &full_fft2,
        4,
        4,
    );

    // -------------------------------------------------------------------------
    // Part 2: Real-valued multidimensional FFT and half-spectrum storage.
    // -------------------------------------------------------------------------
    println!("=== Part 2: Real-Valued FFT and Half-Spectrum Storage ===\n");

    let (m1, m2) = (4usize, 8usize);
    let mut real2 = Array2::<f64>::zeros((m1, m2));
    for i in 0..m1 {
        for j in 0..m2 {
            real2[[i, j]] = (2.0 * PI * i as f64 / m1 as f64).cos()
                + 0.6 * (2.0 * PI * j as f64 / m2 as f64).sin();
        }
    }

    let half_spec = rfft2_half_last_axis(&real2)?;
    print_sample_complex_2d(
        "2D real-to-complex FFT with half-spectrum along last axis",
        &half_spec,
        4,
        5,
    );

    println!(
        "Input shape  = [{}, {}]\nHalf-spectrum shape = [{}, {}]\nExpected last dimension = floor({}/2) + 1 = {}\n",
        m1,
        m2,
        half_spec.dim().0,
        half_spec.dim().1,
        m2,
        m2 / 2 + 1
    );

    // -------------------------------------------------------------------------
    // Part 3: Numerical verification of Hermitian symmetry in 3D.
    // -------------------------------------------------------------------------
    println!("=== Part 3: Hermitian Symmetry in 3D for Real Input ===\n");

    let real3 = build_real_3d(4, 4, 6);
    let spec3 = fft3_complex_from_real(&real3);
    print_sample_complex_3d("Sample entries of full 3D spectrum", &spec3, 10);

    let residual = hermitian_symmetry_residual_3d(&spec3);
    println!(
        "Maximum Hermitian-symmetry residual in 3D: {:.3e}",
        residual
    );

    Ok(())
}
```

Program 12.6.2 demonstrates how memory layout and spectral symmetry interact in multidimensional FFT computations. By explicitly printing row-major offsets and axis access patterns, the program confirms the indexing relation in equation (12.6.4) and shows why transforms along the last axis are naturally cache-friendly, whereas transforms along earlier axes require strided memory access. The numerical experiments also illustrate the symmetry structure of real-valued multidimensional Fourier transforms. When the input signal is real, the computed spectrum satisfies the conjugate symmetry relation described in equation (12.6.5), allowing the transform to store only the independent portion of the spectrum. The resulting reduced array size follows the half-spectrum structure given in equation (12.6.6).

These observations highlight an important theme in high-performance numerical computing. The efficiency of multidimensional FFT algorithms depends not only on the mathematical complexity of the transform but also on how data are arranged in memory. Contiguous access patterns improve cache utilization and reduce data movement, while exploiting Hermitian symmetry reduces both storage requirements and computational effort. In large-scale scientific simulations, these implementation details often determine the practical performance of spectral algorithms.

The modular structure of the code allows the same framework to be extended easily to higher-dimensional arrays, distributed-memory FFT implementations, or GPU-accelerated pipelines. Such extensions form the foundation of many modern applications, including spectral solvers for partial differential equations, large-scale signal processing, and high-resolution imaging systems that rely on efficient multidimensional Fourier transforms.

## 12.6.3. Periodicity, Padding, and Two Canonical Applications

Multidimensional FFTs implicitly treat the input as periodic in each coordinate direction. This assumption has two immediate consequences that matter across scientific computing. First, discontinuities across domain boundaries behave like jump discontinuities in 1D: they inject high-frequency energy and can pollute spectra. Second, FFT-based convolution is circular convolution, so linear convolution requires sufficient zero-padding in each dimension to prevent wrap-around. These considerations show up in both PDE solvers and imaging pipelines, where boundary handling and padding rules are as important as the transform itself (Pei and Tong, 2025; Kohli et al., 2025).

A canonical PDE example is the Poisson equation:

$$
-\Delta u = f
\tag{12.7.1}
$$

on a rectangle. With homogeneous Dirichlet conditions, the discrete Laplacian’s eigenvectors align with sine bases, so applying a DST diagonalizes the operator. In 2D with spacings $(h_x,h_y)$, one obtains eigenvalues:

$$
\lambda_{p,q}
=
\frac{4}{h_x^2}\sin^2\left(\frac{\pi p}{2N_x}\right)
+
\frac{4}{h_y^2}\sin^2\left(\frac{\pi q}{2N_y}\right)
\tag{12.7.3}
$$

and the solver pattern is: transform $b$ by DSTs in $x$ and $y$, divide by $\lambda_{p,q}$ (with the standard zero-mode caveat for Neumann-type problems), then apply inverse DSTs. This yields an $O(N\log N)$ direct method on structured grids and motivates modern DST-accelerated Poisson solvers in 2D and 3D (Pei and Tong, 2025). Related mechanics workflows use sine and cosine transforms as boundary-condition machinery inside FFT-based homogenization and elasticity schemes, including approaches that accommodate more general boundary conditions through sine–cosine constructions (Risthaus and Schneider, 2024; Paux et al., 2025).

A canonical imaging example is deconvolution, where $g=f*h$ models a blurred observation with point spread function $h$. FFTs reduce convolution to pointwise multiplication in frequency space, but modern models can go beyond spatially invariant PSFs by exploiting symmetry. In ring deconvolution microscopy, a rotationally structured model leads to angular convolutions that become pointwise products after a 1D Fourier transform in the angular coordinate, making FFTs the enabling inner loop even when the full PSF is spatially varying (Kohli et al., 2025). Across both examples, the same multidimensional FFT principles recur: separability enables fast transforms, symmetry reduces storage, and periodicity assumptions must be managed through boundary modeling or padding.

### Rust Implementation

Following the discussion in Section 12.6.3 on periodicity assumptions in multidimensional FFTs and the consequences of those assumptions for numerical algorithms, Program 12.6.3 provides a practical Rust implementation illustrating two canonical applications: spectral Poisson solvers and FFT-based convolution in imaging. As explained in this section, multidimensional FFTs implicitly treat input data as periodic in each coordinate direction, which leads to two important computational considerations. First, boundary discontinuities can introduce artificial high-frequency components into the spectrum. Second, convolution performed in the Fourier domain corresponds to circular convolution, requiring appropriate zero-padding to obtain linear convolution without wrap-around artifacts. The program demonstrates how discrete sine transforms diagonalize the Laplacian operator for solving the Poisson equation with homogeneous Dirichlet boundary conditions, following the eigenvalue structure described by equation (12.7.3). It also illustrates FFT-based convolution and deconvolution in image processing, highlighting the role of zero-padding in avoiding spectral wrap-around. By combining these examples, the program connects the theoretical discussion of periodicity, padding, and spectral diagonalization to concrete numerical implementations widely used in scientific computing.

At the core of the Poisson solver implementation are the discrete sine transform functions `dst1_1d` and `idst1_1d`. These functions compute the forward and inverse type-I discrete sine transform, which forms the natural spectral basis for problems with homogeneous Dirichlet boundary conditions. In the discrete setting, the sine functions are eigenvectors of the finite-difference Laplacian operator. Consequently, transforming the right-hand side of the Poisson equation into the sine basis diagonalizes the operator, allowing each spectral coefficient to be divided by the corresponding eigenvalue given in equation (12.7.3). Although the transforms are implemented here using a simple $O(n^2)$ formulation for clarity, the same structure underlies high-performance DST algorithms used in modern spectral solvers.

The routines `dst1_rows` and `dst1_cols` extend the one-dimensional sine transform to two dimensions by applying the transform along each axis of the grid. This separable structure reflects the tensor-product nature of multidimensional transforms discussed earlier in Section 12.6. Applying the sine transform successively along rows and columns produces the two-dimensional transform implemented in `dst1_2d`, while `idst1_2d` performs the corresponding inverse operation. Together, these functions provide the computational mechanism needed to diagonalize the discrete Laplacian operator in the Poisson solver.

The function `solve_poisson_dirichlet_dst` implements the spectral solution strategy described in the text. The right-hand side $f$ of the Poisson equation is first transformed into sine space. Each spectral coefficient is then divided by the Laplacian eigenvalue $\lambda_{p,q}$ from equation (12.7.3), effectively solving the linear system in the transformed basis. Finally, the inverse sine transform converts the result back to physical space, producing the numerical approximation of the solution $u$. This approach yields an $O(N\log N)$ solver when fast sine transforms are used and illustrates how spectral diagonalization simplifies structured-grid PDE problems.

The imaging example demonstrates FFT-based convolution and deconvolution. The functions `fft2_inplace`, `circular_convolution_fft`, and `linear_convolution_fft` implement convolution using the convolution theorem, which states that convolution in the spatial domain corresponds to pointwise multiplication in the frequency domain. The function `fft2_inplace` computes two-dimensional FFTs by applying one-dimensional transforms along rows and columns. When convolution is performed without padding, the resulting operation corresponds to circular convolution, causing wrap-around artifacts due to the implicit periodicity of the FFT. The function `zero_pad` expands arrays with zeros so that convolution computed via FFT corresponds to linear convolution rather than circular convolution.

The function `deconvolve_tikhonov_fft` illustrates a regularized inverse problem commonly encountered in imaging applications. In the model $g = f * h$, the observed image $g$ is the convolution of the true image $f$ with a point spread function $h$. Recovering $f$ requires division by the Fourier transform of the blur kernel. However, small Fourier coefficients of the kernel can amplify noise and numerical errors. To stabilize the inversion, the code applies Tikhonov regularization, which adds a small parameter to the denominator in the spectral division. This approach balances reconstruction accuracy against numerical stability and is widely used in practical deconvolution algorithms.

The `main` function orchestrates the demonstration through two representative examples. In the first, a manufactured solution of the Poisson equation is used to generate a right-hand side function $f$, allowing the numerical solver to be compared with the known analytical solution. The maximum pointwise error between the numerical and exact solutions provides a measure of discretization accuracy. In the second example, a synthetic image is blurred using a small convolution kernel. The program first demonstrates circular convolution without padding, revealing wrap-around artifacts. It then performs linear convolution with zero-padding and applies regularized deconvolution to recover the original image. These examples illustrate the practical implications of periodicity assumptions in FFT-based computations.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
ndarray = "0.15"
rustfft = "6"
```

```rust
// Program 12.6.3: Periodicity, Padding, and Two Canonical Applications
//
// This program illustrates two canonical applications discussed in Section 12.6.3:
//
// 1. A 2D Poisson solver with homogeneous Dirichlet boundary conditions,
//    diagonalized by the discrete sine transform (DST-I).
// 2. FFT-based 2D convolution and deconvolution for imaging, showing why
//    zero-padding is required to obtain linear convolution instead of circular
//    convolution.
//
// The Poisson solver uses the eigenvalue formula of Eq. (12.7.3) on an
// interior grid. The imaging example demonstrates wrap-around artifacts when
// padding is omitted and shows how padding prevents them.
//
// Cargo.toml dependencies:
// [dependencies]
// ndarray = "0.15"
// rustfft = "6"

use ndarray::{s, Array2};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::error::Error;
use std::f64::consts::PI;

// -----------------------------------------------------------------------------
// Part A. Discrete sine transform (DST-I) utilities for the Poisson solver
// -----------------------------------------------------------------------------

/// Forward DST-I on a 1D vector of length n.
/// This is an educational O(n^2) implementation.
/// The transform corresponds to coefficients in the sine basis for
/// homogeneous Dirichlet boundary conditions.
fn dst1_1d(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut y = vec![0.0; n];

    for k in 1..=n {
        let mut sum = 0.0;
        for j in 1..=n {
            let angle = PI * (j as f64) * (k as f64) / ((n + 1) as f64);
            sum += x[j - 1] * angle.sin();
        }
        y[k - 1] = sum;
    }

    y
}

/// Inverse DST-I. For DST-I, the inverse differs only by the scaling
/// factor 2/(n+1).
fn idst1_1d(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let scale = 2.0 / ((n + 1) as f64);
    let mut x = dst1_1d(y);
    for v in &mut x {
        *v *= scale;
    }
    x
}

/// Apply DST-I to all rows of a 2D array.
fn dst1_rows(a: &Array2<f64>) -> Array2<f64> {
    let (nx, ny) = a.dim();
    let mut out = Array2::<f64>::zeros((nx, ny));

    for i in 0..nx {
        let row = a.slice(s![i, ..]).to_vec();
        let tr = dst1_1d(&row);
        for j in 0..ny {
            out[[i, j]] = tr[j];
        }
    }

    out
}

/// Apply DST-I to all columns of a 2D array.
fn dst1_cols(a: &Array2<f64>) -> Array2<f64> {
    let (nx, ny) = a.dim();
    let mut out = Array2::<f64>::zeros((nx, ny));

    for j in 0..ny {
        let col: Vec<f64> = (0..nx).map(|i| a[[i, j]]).collect();
        let tc = dst1_1d(&col);
        for i in 0..nx {
            out[[i, j]] = tc[i];
        }
    }

    out
}

/// Apply inverse DST-I to all rows.
fn idst1_rows(a: &Array2<f64>) -> Array2<f64> {
    let (nx, ny) = a.dim();
    let mut out = Array2::<f64>::zeros((nx, ny));

    for i in 0..nx {
        let row = a.slice(s![i, ..]).to_vec();
        let tr = idst1_1d(&row);
        for j in 0..ny {
            out[[i, j]] = tr[j];
        }
    }

    out
}

/// Apply inverse DST-I to all columns.
fn idst1_cols(a: &Array2<f64>) -> Array2<f64> {
    let (nx, ny) = a.dim();
    let mut out = Array2::<f64>::zeros((nx, ny));

    for j in 0..ny {
        let col: Vec<f64> = (0..nx).map(|i| a[[i, j]]).collect();
        let tc = idst1_1d(&col);
        for i in 0..nx {
            out[[i, j]] = tc[i];
        }
    }

    out
}

/// 2D forward DST-I by successive row and column transforms.
fn dst1_2d(a: &Array2<f64>) -> Array2<f64> {
    let tmp = dst1_rows(a);
    dst1_cols(&tmp)
}

/// 2D inverse DST-I by successive inverse row and column transforms.
fn idst1_2d(a: &Array2<f64>) -> Array2<f64> {
    let tmp = idst1_rows(a);
    idst1_cols(&tmp)
}

/// Solve -Δ_h u = f on the interior of the unit square with homogeneous
/// Dirichlet boundary conditions using a DST-I diagonalization.
///
/// The array f contains interior values only, of shape (nx, ny), and the
/// grid spacings are hx = 1/(nx+1), hy = 1/(ny+1).
fn solve_poisson_dirichlet_dst(f: &Array2<f64>) -> Array2<f64> {
    let (nx, ny) = f.dim();
    let hx = 1.0 / ((nx + 1) as f64);
    let hy = 1.0 / ((ny + 1) as f64);

    // Transform the right-hand side into sine space.
    let f_hat = dst1_2d(f);

    // Divide by eigenvalues lambda_{p,q} from Eq. (12.7.3).
    let mut u_hat = Array2::<f64>::zeros((nx, ny));
    for p in 1..=nx {
        for q in 1..=ny {
            let lambda_pq = (4.0 / (hx * hx)) * (PI * (p as f64) / (2.0 * nx as f64 + 2.0)).sin().powi(2)
                + (4.0 / (hy * hy)) * (PI * (q as f64) / (2.0 * ny as f64 + 2.0)).sin().powi(2);

            u_hat[[p - 1, q - 1]] = f_hat[[p - 1, q - 1]] / lambda_pq;
        }
    }

    // Transform back to physical space.
    idst1_2d(&u_hat)
}

/// Build a manufactured right-hand side f corresponding to the exact solution
/// u(x,y) = sin(pi x) sin(2 pi y) on the unit square.
/// Then -Δu = 5 pi^2 u.
fn build_poisson_rhs(nx: usize, ny: usize) -> (Array2<f64>, Array2<f64>) {
    let hx = 1.0 / ((nx + 1) as f64);
    let hy = 1.0 / ((ny + 1) as f64);

    let mut f = Array2::<f64>::zeros((nx, ny));
    let mut u_exact = Array2::<f64>::zeros((nx, ny));

    for i in 0..nx {
        for j in 0..ny {
            let x = (i + 1) as f64 * hx;
            let y = (j + 1) as f64 * hy;
            let u = (PI * x).sin() * (2.0 * PI * y).sin();
            let rhs = 5.0 * PI * PI * u;

            u_exact[[i, j]] = u;
            f[[i, j]] = rhs;
        }
    }

    (f, u_exact)
}

// -----------------------------------------------------------------------------
// Part B. FFT-based convolution and deconvolution utilities for imaging
// -----------------------------------------------------------------------------

/// Convert a real array to a complex array.
fn real_to_complex(a: &Array2<f64>) -> Array2<Complex<f64>> {
    let (nx, ny) = a.dim();
    let mut out = Array2::<Complex<f64>>::zeros((nx, ny));
    for i in 0..nx {
        for j in 0..ny {
            out[[i, j]] = Complex::new(a[[i, j]], 0.0);
        }
    }
    out
}

/// Extract the real part of a complex array.
fn complex_to_real(a: &Array2<Complex<f64>>) -> Array2<f64> {
    let (nx, ny) = a.dim();
    let mut out = Array2::<f64>::zeros((nx, ny));
    for i in 0..nx {
        for j in 0..ny {
            out[[i, j]] = a[[i, j]].re;
        }
    }
    out
}

/// In-place 2D FFT or inverse FFT by applying 1D transforms along rows
/// and then columns.
fn fft2_inplace(a: &mut Array2<Complex<f64>>, inverse: bool) {
    let (nx, ny) = a.dim();
    let mut planner = FftPlanner::<f64>::new();

    let fft_y = if inverse {
        planner.plan_fft_inverse(ny)
    } else {
        planner.plan_fft_forward(ny)
    };

    let fft_x = if inverse {
        planner.plan_fft_inverse(nx)
    } else {
        planner.plan_fft_forward(nx)
    };

    // Transform rows.
    for i in 0..nx {
        let mut row: Vec<Complex<f64>> = (0..ny).map(|j| a[[i, j]]).collect();
        fft_y.process(&mut row);
        for j in 0..ny {
            a[[i, j]] = row[j];
        }
    }

    // Transform columns.
    for j in 0..ny {
        let mut col: Vec<Complex<f64>> = (0..nx).map(|i| a[[i, j]]).collect();
        fft_x.process(&mut col);
        for i in 0..nx {
            a[[i, j]] = col[i];
        }
    }

    if inverse {
        let scale = (nx * ny) as f64;
        for v in a.iter_mut() {
            *v /= scale;
        }
    }
}

/// Pad a real array with zeros to a target shape, placing the original data
/// in the upper-left corner.
fn zero_pad(a: &Array2<f64>, nx_pad: usize, ny_pad: usize) -> Array2<f64> {
    let (nx, ny) = a.dim();
    let mut out = Array2::<f64>::zeros((nx_pad, ny_pad));
    for i in 0..nx {
        for j in 0..ny {
            out[[i, j]] = a[[i, j]];
        }
    }
    out
}

/// Circular convolution via FFT on arrays of equal shape.
fn circular_convolution_fft(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (nx, ny) = a.dim();
    assert_eq!((nx, ny), b.dim());

    let mut fa = real_to_complex(a);
    let mut fb = real_to_complex(b);

    fft2_inplace(&mut fa, false);
    fft2_inplace(&mut fb, false);

    let mut fc = Array2::<Complex<f64>>::zeros((nx, ny));
    for i in 0..nx {
        for j in 0..ny {
            fc[[i, j]] = fa[[i, j]] * fb[[i, j]];
        }
    }

    fft2_inplace(&mut fc, true);
    complex_to_real(&fc)
}

/// Linear convolution via FFT with zero-padding. The result has shape
/// (nx_a + nx_b - 1, ny_a + ny_b - 1).
fn linear_convolution_fft(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (ax, ay) = a.dim();
    let (bx, by) = b.dim();

    let nx = ax + bx - 1;
    let ny = ay + by - 1;

    let a_pad = zero_pad(a, nx, ny);
    let b_pad = zero_pad(b, nx, ny);

    circular_convolution_fft(&a_pad, &b_pad)
}

/// Tikhonov-regularized FFT deconvolution on padded arrays.
/// Returns an array of the same padded shape.
fn deconvolve_tikhonov_fft(
    blurred_padded: &Array2<f64>,
    psf_padded: &Array2<f64>,
    alpha: f64,
) -> Array2<f64> {
    let (nx, ny) = blurred_padded.dim();
    assert_eq!((nx, ny), psf_padded.dim());

    let mut g = real_to_complex(blurred_padded);
    let mut h = real_to_complex(psf_padded);

    fft2_inplace(&mut g, false);
    fft2_inplace(&mut h, false);

    let mut f_hat = Array2::<Complex<f64>>::zeros((nx, ny));
    for i in 0..nx {
        for j in 0..ny {
            let hij = h[[i, j]];
            let denom = hij.norm_sqr() + alpha;
            f_hat[[i, j]] = hij.conj() * g[[i, j]] / denom;
        }
    }

    fft2_inplace(&mut f_hat, true);
    complex_to_real(&f_hat)
}

/// Build a small synthetic image with a few bright features.
fn build_test_image(nx: usize, ny: usize) -> Array2<f64> {
    let mut img = Array2::<f64>::zeros((nx, ny));

    img[[2, 2]] = 1.0;
    img[[3, 7]] = 0.8;
    img[[6, 4]] = 0.6;
    img[[7, 9]] = 1.2;

    // Add a small rectangular block to create a less sparse object.
    for i in 9..12.min(nx) {
        for j in 3..6.min(ny) {
            img[[i, j]] = 0.4;
        }
    }

    img
}

/// Build a normalized 3x3 blur kernel.
fn build_blur_kernel() -> Array2<f64> {
    let mut k = Array2::<f64>::zeros((3, 3));

    k[[0, 0]] = 1.0;
    k[[0, 1]] = 2.0;
    k[[0, 2]] = 1.0;
    k[[1, 0]] = 2.0;
    k[[1, 1]] = 4.0;
    k[[1, 2]] = 2.0;
    k[[2, 0]] = 1.0;
    k[[2, 1]] = 2.0;
    k[[2, 2]] = 1.0;

    let sum: f64 = k.iter().sum();
    for v in k.iter_mut() {
        *v /= sum;
    }

    k
}

/// Compute the maximum absolute error between two real arrays of equal shape.
fn max_abs_error(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let (nx, ny) = a.dim();
    assert_eq!((nx, ny), b.dim());

    let mut err = 0.0_f64;
    for i in 0..nx {
        for j in 0..ny {
            err = err.max((a[[i, j]] - b[[i, j]]).abs());
        }
    }
    err
}

/// Print a small top-left corner of a real array.
fn print_sample_real(name: &str, a: &Array2<f64>, rows: usize, cols: usize) {
    let (nx, ny) = a.dim();
    println!("{name} (shape = [{nx}, {ny}])");
    for i in 0..nx.min(rows) {
        for j in 0..ny.min(cols) {
            print!("{:9.5} ", a[[i, j]]);
        }
        println!();
    }
    println!();
}

fn main() -> Result<(), Box<dyn Error>> {
    // -------------------------------------------------------------------------
    // Example 1: 2D Poisson solver with homogeneous Dirichlet conditions
    // -------------------------------------------------------------------------
    println!("=== Example 1: 2D Poisson Solver via DST Diagonalization ===\n");

    let nx = 16usize;
    let ny = 16usize;
    let (f, u_exact) = build_poisson_rhs(nx, ny);
    let u_num = solve_poisson_dirichlet_dst(&f);

    let poisson_error = max_abs_error(&u_num, &u_exact);

    print_sample_real("Sample of right-hand side f", &f, 4, 4);
    print_sample_real("Sample of numerical solution u", &u_num, 4, 4);
    print_sample_real("Sample of exact solution u_exact", &u_exact, 4, 4);

    println!(
        "Maximum pointwise error for the Poisson solve: {:.3e}\n",
        poisson_error
    );

    // -------------------------------------------------------------------------
    // Example 2: FFT-based convolution, padding, and deconvolution
    // -------------------------------------------------------------------------
    println!("=== Example 2: FFT Convolution, Padding, and Deconvolution ===\n");

    let image = build_test_image(12, 12);
    let kernel = build_blur_kernel();

    // Circular convolution without padding: wrap-around is present.
    let kernel_same_size = zero_pad(&kernel, 12, 12);
    let blurred_circular = circular_convolution_fft(&image, &kernel_same_size);

    // Linear convolution with zero-padding: no wrap-around.
    let blurred_linear = linear_convolution_fft(&image, &kernel);

    // Deconvolution on the padded linear-convolution grid.
    let (bx, by) = blurred_linear.dim();
    let kernel_padded = zero_pad(&kernel, bx, by);
    let recovered_padded = deconvolve_tikhonov_fft(&blurred_linear, &kernel_padded, 1e-6);

    // Crop the deconvolved image back to the original image support.
    let recovered = recovered_padded.slice(s![0..12, 0..12]).to_owned();
    let deconv_error = max_abs_error(&recovered, &image);

    print_sample_real("Original image", &image, 8, 8);
    print_sample_real("Circular convolution without padding", &blurred_circular, 8, 8);
    print_sample_real("Linear convolution with zero-padding", &blurred_linear, 8, 8);
    print_sample_real("Recovered image after padded FFT deconvolution", &recovered, 8, 8);

    println!(
        "Maximum pointwise error after padded Tikhonov deconvolution: {:.3e}",
        deconv_error
    );

    Ok(())
}
```

Program 12.6.3 demonstrates how the periodicity assumptions inherent in FFT-based algorithms influence both numerical methods for partial differential equations and practical image-processing workflows. In the Poisson solver example, the discrete sine transform diagonalizes the Laplacian operator, allowing the linear system to be solved efficiently in the spectral domain using the eigenvalues described in equation (12.7.3). The resulting numerical solution closely approximates the analytical solution, with the remaining discrepancy reflecting the expected discretization error of the finite-difference approximation.

The convolution example illustrates another important consequence of the FFT’s implicit periodicity. When convolution is performed without padding, the algorithm computes circular convolution, causing wrap-around artifacts at the boundaries of the image. Introducing sufficient zero-padding transforms the operation into linear convolution, preventing this aliasing effect and producing the correct physical result. This behavior is fundamental in many imaging pipelines and signal-processing applications.

The deconvolution experiment further demonstrates how spectral methods enable efficient inversion of convolution models. By performing division in the frequency domain and incorporating Tikhonov regularization, the algorithm recovers the principal features of the original image while avoiding numerical instability caused by small Fourier coefficients. Although simplified for pedagogical clarity, the structure of this implementation mirrors the core ideas used in large-scale scientific imaging and computational physics.

Taken together, these examples highlight the practical implications of the theoretical concepts introduced earlier in the section. The separability of multidimensional transforms enables efficient computation, symmetry reduces storage requirements, and careful handling of periodicity assumptions, through boundary modeling or padding, is essential for obtaining physically meaningful results.

# 12.7. Fourier Transforms of Real Data in Two and Three Dimensions

In many scientific and engineering applications the discrete Fourier transform is applied not to complex data but to real-valued multidimensional fields. Examples include images, volumetric density fields, pressure and velocity distributions in fluid simulations, gravitational and electrostatic potentials, and spatial data sets arising in inverse problems or statistical estimation. In these settings the Fourier transform operates on two- or three-dimensional arrays whose entries are real numbers.

The classical fast Fourier transform reduces the arithmetic cost of computing the discrete Fourier transform from quadratic complexity to

$$
O(N\log N)
\tag{12.7.1}
$$

where $N$ denotes the total number of samples. While this complexity reduction remains mathematically decisive, real-valued data introduce an additional structural property that significantly affects practical implementations. Specifically, the Fourier coefficients satisfy a conjugate symmetry relation, implying that only approximately half of the spectrum contains independent information. Modern FFT libraries exploit this structure through real-to-complex transforms, reducing storage requirements and avoiding redundant computations.

Section 12.6 established that multidimensional FFTs can be constructed from sequences of one-dimensional transforms along each coordinate axis. The present section builds on that framework by examining the algebraic structure of multidimensional Fourier transforms for real data and by describing the storage layouts used in practical two- and three-dimensional implementations.

## 12.7.1. Multidimensional DFT as a Structured Linear Map

Let $f$ denote a complex-valued array defined on a rectangular grid. In two dimensions we write:

$$
f_{n_1,n_2}\in\mathbb{C},\qquad
0\le n_1 < N_1,\qquad
0\le n_2 < N_2 \tag{12.7.2}
$$

The two-dimensional discrete Fourier transform (DFT) is defined by:

$$
F_{k_1,k_2}
=
\sum_{n_1=0}^{N_1-1}
\sum_{n_2=0}^{N_2-1}
f_{n_1,n_2}\,
\exp\left(
-2\pi i\left(\frac{n_1 k_1}{N_1}+\frac{n_2 k_2}{N_2}\right)
\right)
\tag{12.7.3}
$$

for

$$
0 \le k_1 < N_1,\qquad
0 \le k_2 < N_2 
$$

In $d$ dimensions with sizes $N_1,\dots,N_d$, the transform generalizes naturally to:

$$
F_{k_1,\dots,k_d}
=
\sum_{n_1=0}^{N_1-1}\cdots\sum_{n_d=0}^{N_d-1}
f_{n_1,\dots,n_d}\,
\exp\left(
-2\pi i \sum_{r=1}^{d}\frac{n_r k_r}{N_r}
\right)
\tag{12.7.4}
$$

A useful linear-algebra interpretation arises by vectorizing the multidimensional array. Let $W_N$ denote the Fourier matrix:

$$
(W_N)_{k,n} = \exp\!\left(-\frac{2\pi i k n}{N}\right)
\tag{12.7.5}
$$

Then the multidimensional transform satisfies the Kronecker-product identity

$$
\operatorname{vec}(F)
=
\left(W_{N_d}\otimes\cdots\otimes W_{N_1}\right)\operatorname{vec}(f)
\tag{12.7.6}
$$

Equation (12.7.5) reveals the tensor-product structure of the multidimensional DFT. This structure explains why multidimensional FFT algorithms can be implemented as sequences of one-dimensional FFTs applied successively along each axis.

### Rust Implementation

Following the discussion in Section 12.7.1 on the structured linear-map interpretation of the multidimensional discrete Fourier transform, Program 12.7.1 provides a practical implementation of the two-dimensional transform that illustrates the tensor-product structure described by equations (12.7.3)–(12.7.6). In multidimensional signal processing and scientific computing, the defining formula of the discrete Fourier transform can be evaluated directly from its double summation form, but this approach quickly becomes computationally impractical as problem size increases. The fast Fourier transform resolves this difficulty by exploiting separability, allowing multidimensional transforms to be computed as sequences of one-dimensional transforms applied along each coordinate axis. This program demonstrates both perspectives. It first evaluates the two-dimensional transform directly using the definition in equation (12.7.3), and then computes the same transform using successive one-dimensional FFTs that reflect the Kronecker-product formulation in equation (12.7.6). By comparing the two results and verifying reconstruction through the inverse transform, the implementation highlights the numerical equivalence between the algebraic formulation and the efficient FFT-based algorithm.

At the core of the implementation is the construction of a small complex-valued test array that represents a sampled two-dimensional signal. The function `build_test_array` generates this grid of values using combinations of trigonometric components so that the resulting spectrum contains multiple nontrivial frequency contributions. This synthetic field serves as a controlled input for demonstrating the behavior of the discrete Fourier transform in two dimensions.

The function `dft2_direct` implements the two-dimensional transform directly from its mathematical definition given in equation (12.7.3). For each frequency pair $(k_1,k_2)$, the function evaluates the double summation over spatial indices by computing the complex exponential kernel and accumulating the contributions from every grid point. Although this implementation is computationally expensive, it provides a clear numerical realization of the theoretical transform and serves as a reference against which the fast algorithm can be validated.

The structured FFT implementation is built from two helper routines that apply one-dimensional transforms along different coordinate directions. The function `fft_rows` applies a one-dimensional FFT along each row of the array, corresponding to transforms in the second coordinate direction. The function `fft_cols` performs the analogous operation along each column, computing transforms in the first coordinate direction. When these two stages are applied sequentially, the result is a two-dimensional transform constructed from axis-wise one-dimensional transforms, reflecting the separability implied by equation (12.7.6).

The routine `fft2_structured` combines these row and column transforms to compute the full two-dimensional FFT using the structured approach. Its counterpart, `ifft2_structured`, performs the inverse operation by applying inverse transforms along the same axes and then normalizing by the total number of grid points. This normalization ensures that the inverse transform reconstructs the original data in accordance with the standard discrete Fourier transform conventions.

Two additional utility functions assist with numerical verification and output. The function `max_abs_error` computes the maximum entrywise difference between two arrays and is used to quantify the agreement between the direct and FFT-based implementations as well as the accuracy of the inverse reconstruction. The function `print_sample` displays a small leading portion of an array to illustrate the structure of the input data and the resulting Fourier coefficients without overwhelming the output.

The `main` function orchestrates the entire demonstration. It constructs the test array, computes the spectrum using both the direct transform and the structured FFT, and then applies the inverse transform to recover the original data. The program reports the maximum discrepancy between the two spectral computations and the reconstruction error after the inverse transform. These diagnostic quantities provide a practical confirmation that the structured FFT implementation reproduces the same transform as the direct evaluation of equation (12.7.3), with differences only at the level of floating-point roundoff.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
ndarray = "0.15"
rustfft = "6"
```

```rust
// Program 12.7.1: Two-Dimensional FFT as a Structured Linear Map
//
// This program illustrates the tensor-product structure described in
// Eqs. (12.7.3) to (12.7.6) by computing a two-dimensional Fourier transform
// in two different ways:
//
// 1. Direct evaluation of the 2D DFT from Eq. (12.7.3), using the defining sum.
// 2. Successive one-dimensional FFTs along rows and columns, reflecting the
//    Kronecker-product structure in Eq. (12.7.6).
//
// The program then compares the two results and verifies that the inverse FFT
// reconstructs the original array to floating-point accuracy.
//
// Cargo.toml dependencies:
// [dependencies]
// ndarray = "0.15"
// rustfft = "6"

use ndarray::Array2;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::error::Error;
use std::f64::consts::PI;

/// Build a small complex-valued 2D test array.
/// The data are chosen to vary in both coordinates so that the spectrum is nontrivial.
fn build_test_array(n1: usize, n2: usize) -> Array2<Complex<f64>> {
    let mut a = Array2::<Complex<f64>>::zeros((n1, n2));

    for i in 0..n1 {
        for j in 0..n2 {
            let real = (2.0 * PI * i as f64 / n1 as f64).cos()
                + 0.5 * (2.0 * PI * j as f64 / n2 as f64).sin()
                + 0.25 * (2.0 * PI * (i + j) as f64 / (n1 + n2) as f64).cos();

            let imag = 0.4 * (2.0 * PI * (2 * i + j) as f64 / (n1 + n2) as f64).sin()
                - 0.3 * (2.0 * PI * j as f64 / n2 as f64).cos();

            a[[i, j]] = Complex::new(real, imag);
        }
    }

    a
}

/// Direct evaluation of the two-dimensional DFT in Eq. (12.7.3).
/// This is an educational O(N1^2 N2^2) implementation.
fn dft2_direct(input: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (n1, n2) = input.dim();
    let mut output = Array2::<Complex<f64>>::zeros((n1, n2));

    for k1 in 0..n1 {
        for k2 in 0..n2 {
            let mut sum = Complex::new(0.0, 0.0);

            for n1_idx in 0..n1 {
                for n2_idx in 0..n2 {
                    let phase = -2.0
                        * PI
                        * ((n1_idx * k1) as f64 / n1 as f64
                            + (n2_idx * k2) as f64 / n2 as f64);

                    let w = Complex::new(phase.cos(), phase.sin());
                    sum += input[[n1_idx, n2_idx]] * w;
                }
            }

            output[[k1, k2]] = sum;
        }
    }

    output
}

/// Apply a one-dimensional FFT to every row.
fn fft_rows(input: &Array2<Complex<f64>>, inverse: bool) -> Array2<Complex<f64>> {
    let (n1, n2) = input.dim();
    let mut output = input.clone();

    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n2)
    } else {
        planner.plan_fft_forward(n2)
    };

    for i in 0..n1 {
        let mut row: Vec<Complex<f64>> = (0..n2).map(|j| output[[i, j]]).collect();
        fft.process(&mut row);
        for j in 0..n2 {
            output[[i, j]] = row[j];
        }
    }

    output
}

/// Apply a one-dimensional FFT to every column.
fn fft_cols(input: &Array2<Complex<f64>>, inverse: bool) -> Array2<Complex<f64>> {
    let (n1, n2) = input.dim();
    let mut output = input.clone();

    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n1)
    } else {
        planner.plan_fft_forward(n1)
    };

    for j in 0..n2 {
        let mut col: Vec<Complex<f64>> = (0..n1).map(|i| output[[i, j]]).collect();
        fft.process(&mut col);
        for i in 0..n1 {
            output[[i, j]] = col[i];
        }
    }

    output
}

/// Compute the 2D FFT by successive one-dimensional FFTs along rows and columns.
/// This reflects the structured linear-map viewpoint of Eq. (12.7.6).
fn fft2_structured(input: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let tmp = fft_rows(input, false);
    fft_cols(&tmp, false)
}

/// Compute the inverse 2D FFT and normalize by the total number of grid points.
fn ifft2_structured(input: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let tmp = fft_rows(input, true);
    let mut output = fft_cols(&tmp, true);

    let (n1, n2) = output.dim();
    let scale = (n1 * n2) as f64;

    for i in 0..n1 {
        for j in 0..n2 {
            output[[i, j]] /= scale;
        }
    }

    output
}

/// Compute the maximum entrywise absolute difference between two arrays.
fn max_abs_error(a: &Array2<Complex<f64>>, b: &Array2<Complex<f64>>) -> f64 {
    let (n1, n2) = a.dim();
    assert_eq!((n1, n2), b.dim());

    let mut err = 0.0_f64;
    for i in 0..n1 {
        for j in 0..n2 {
            err = err.max((a[[i, j]] - b[[i, j]]).norm());
        }
    }

    err
}

/// Print a small leading block of a complex array.
fn print_sample(name: &str, a: &Array2<Complex<f64>>, max_rows: usize, max_cols: usize) {
    let (n1, n2) = a.dim();
    println!("{name} (shape = [{n1}, {n2}])");

    for i in 0..n1.min(max_rows) {
        for j in 0..n2.min(max_cols) {
            let z = a[[i, j]];
            print!("({:+8.4},{:+8.4}i) ", z.re, z.im);
        }
        println!();
    }
    println!();
}

fn main() -> Result<(), Box<dyn Error>> {
    let n1 = 4usize;
    let n2 = 5usize;

    let f = build_test_array(n1, n2);

    // Direct evaluation of Eq. (12.7.3).
    let fhat_direct = dft2_direct(&f);

    // Structured FFT via successive one-dimensional transforms, as motivated by Eq. (12.7.6).
    let fhat_fft = fft2_structured(&f);

    // Inverse transform to verify reconstruction.
    let f_recovered = ifft2_structured(&fhat_fft);

    let spectrum_error = max_abs_error(&fhat_direct, &fhat_fft);
    let reconstruction_error = max_abs_error(&f, &f_recovered);

    println!("=== Program 12.7.1: Two-Dimensional FFT as a Structured Linear Map ===\n");

    print_sample("Input array f", &f, 4, 5);
    print_sample("Direct 2D DFT from Eq. (12.7.3)", &fhat_direct, 4, 5);
    print_sample(
        "Structured 2D FFT via successive 1D transforms",
        &fhat_fft,
        4,
        5,
    );
    print_sample("Recovered array after inverse FFT", &f_recovered, 4, 5);

    println!(
        "Maximum difference between direct DFT and structured FFT: {:.3e}",
        spectrum_error
    );
    println!(
        "Maximum reconstruction error after inverse FFT: {:.3e}",
        reconstruction_error
    );

    if spectrum_error > 1e-10 {
        return Err(format!(
            "structured FFT does not match direct DFT closely enough: {:.3e}",
            spectrum_error
        )
        .into());
    }

    if reconstruction_error > 1e-10 {
        return Err(format!(
            "inverse FFT reconstruction error is too large: {:.3e}",
            reconstruction_error
        )
        .into());
    }

    Ok(())
}
```

Program 12.7.1 demonstrates the practical consequences of the tensor-product structure underlying multidimensional Fourier transforms. Although the direct evaluation of the two-dimensional discrete Fourier transform in equation (12.7.3) provides a straightforward mathematical definition, its computational cost grows rapidly with problem size. The fast Fourier transform circumvents this limitation by exploiting separability, allowing multidimensional transforms to be computed through sequences of one-dimensional transforms applied along each coordinate direction.

The numerical comparison performed in the program confirms that the structured FFT approach yields the same spectral coefficients as the direct evaluation, with discrepancies only at the level of machine precision. This agreement verifies the correctness of the axis-wise implementation and illustrates the algorithmic interpretation of equation (12.7.6). The reconstruction test further confirms that the inverse transform accurately recovers the original data.

Beyond validating the mathematics, the program illustrates a key computational principle: multidimensional Fourier transforms are best understood not as fundamentally new operations but as structured compositions of one-dimensional transforms. This viewpoint underlies essentially all practical FFT implementations used in scientific computing, signal processing, and large-scale simulation. It also prepares the groundwork for the next sections, where the additional structure of real-valued data leads to Hermitian symmetry and specialized storage layouts that further reduce computational and memory costs.

## 12.7.2. Real Data and Hermitian Symmetry

In many applications the input data are real-valued, so $f_{n_1,\dots,n_d}\in\mathbb{R}$. Taking the complex conjugate of (12.7.4) and using $\overline{f}=f$ yields the multidimensional Hermitian symmetry relation:

$$
F_{(-k_1 \bmod N_1),\dots,(-k_d \bmod N_d)}
=
\overline{F_{k_1,\dots,k_d}}
\tag{12.7.7}
$$

Thus every Fourier coefficient has a conjugate partner located at the corresponding negative frequency. Consequently, only roughly half of the Fourier lattice contains independent information.

## 12.7.3. Two-Dimensional Real FFTs

Consider a real two-dimensional array $f\in\mathbb{R}^{N_1\times N_2}$. The Hermitian symmetry condition becomes:

$$
F_{(N_1-k_1)\bmod N_1,\,(N_2-k_2)\bmod N_2}
=
\overline{F_{k_1,k_2}}
\tag{12.7.8}
$$

A common storage convention retains the frequency indices,

$$0\le k_1<N_1,\qquad0\le k_2\le\left\lfloor\frac{N_2}{2}\right\rfloor \tag{12.7.9}$$

The stored spectrum therefore has dimensions:

$$N_1\times\left(\left\lfloor\frac{N_2}{2}\right\rfloor+1\right)\tag{12.7.10}$$

The remaining coefficients can be reconstructed using the conjugate symmetry relation (12.7.8). Frequencies corresponding to the DC component and Nyquist boundaries are purely real.

The arithmetic complexity remains

$$O(N \log N), \qquad N = N_1 N_2 \tag{12.7.11}$$

but the effective memory traffic and storage requirements are reduced because redundant frequencies are omitted.

### Rust Implementation

Following the discussion in Sections 12.7.2 and 12.7.3 on the structure of Fourier transforms for real-valued multidimensional data, Program 12.7.2 provides a practical implementation that illustrates Hermitian symmetry and the reduced storage requirements of real-to-complex FFTs. When the input field is real-valued, the Fourier coefficients occur in complex-conjugate pairs located at opposite frequency indices. This symmetry, expressed in equation (12.7.8), implies that only approximately half of the Fourier spectrum contains independent information. Modern FFT implementations therefore compute real-to-complex transforms that store only the nonredundant portion of the spectrum. The program demonstrates this principle by computing a full two-dimensional complex FFT for comparison, performing a real-to-complex transform that retains only the half-spectrum described by equation (12.7.10), and reconstructing the omitted frequencies using the Hermitian symmetry relation. By verifying these properties numerically, the implementation illustrates how theoretical symmetry relations translate into practical reductions in memory usage and computational work.

At the core of the implementation is the construction of a real-valued two-dimensional test field representing sampled spatial data. The function `build_real_test_array` generates this field by combining several sinusoidal components of different frequencies. Such combinations produce structured spectral content that makes it possible to observe the distribution of energy across frequency indices after applying the Fourier transform.

To compute the Fourier transform using standard complex arithmetic, the function `real_to_complex` embeds the real-valued array into a complex representation by assigning each entry a zero imaginary component. This conversion allows the program to compute the full two-dimensional spectrum using the complex FFT routines implemented in `fft_rows`, `fft_cols`, and `fft2_complex`. These functions apply one-dimensional FFTs successively along rows and columns, reflecting the separable tensor-product structure of the multidimensional transform described earlier in equation (12.7.4). The resulting array contains the complete complex Fourier spectrum.

The function `rfft2_half_spectrum` performs a real-to-complex transform that directly computes only the nonredundant portion of the spectrum. The first stage applies a real-to-complex FFT along each row, producing the half-spectrum along the second coordinate direction. The output therefore retains only frequency indices satisfying the range specified in equation (12.7.9). The second stage applies a complex FFT along the first coordinate direction to complete the two-dimensional transform. The resulting array has dimensions consistent with the reduced storage layout given in equation (12.7.10).

Because the real-to-complex transform stores only the independent Fourier coefficients, the function `reconstruct_full_spectrum_from_half` rebuilds the omitted coefficients using the Hermitian symmetry relation given in equation (12.7.8). For every retained coefficient, its conjugate partner at the corresponding negative frequency is reconstructed and placed in the full spectral grid.

Several diagnostic routines are included to verify the theoretical properties discussed in the text. The function `hermitian_residual_2d` evaluates the numerical residual of the symmetry relation by comparing each coefficient with the complex conjugate of its symmetric partner. The function `max_abs_error` compares two spectra entry by entry and is used to confirm that the reconstructed spectrum matches the directly computed full spectrum. The function `self_conjugate_imag_residual` measures the imaginary part of coefficients that are theoretically self-conjugate, such as the DC and Nyquist frequencies, which should be purely real.

The `main` function coordinates the demonstration. It constructs the test field, computes the full complex FFT, computes the reduced half-spectrum using the real-to-complex transform, and reconstructs the complete spectrum from the stored coefficients. Diagnostic quantities are printed to confirm Hermitian symmetry, verify the correctness of the reconstruction, and demonstrate that the stored half-spectrum has the dimensions predicted by equation (12.7.10). These checks illustrate how real-valued input data reduce the number of independent Fourier coefficients without altering the overall computational complexity described in equation (12.7.11).

Add the following dependencies to cargo.toml:

```rust
[dependencies]
ndarray = "0.15"
rustfft = "6"
realfft = "3"
```

```rust
// Program 12.7.2: Two-Dimensional Real FFTs and Hermitian Symmetry
//
// This program illustrates the Hermitian symmetry of the two-dimensional DFT
// for real-valued data, as described in Eqs. (12.7.7) and (12.7.8), and
// demonstrates the reduced half-spectrum storage convention of
// Eqs. (12.7.9) and (12.7.10).
//
// The implementation proceeds in three parts:
//
// 1. Build a real-valued 2D test field.
// 2. Compute the full 2D complex FFT by embedding the real data in complex form.
// 3. Compute a real-to-complex FFT that stores only the nonredundant half-spectrum
//    along the last axis, then verify that the full spectrum can be reconstructed
//    from Hermitian symmetry.
//
// Cargo.toml dependencies:
// [dependencies]
// ndarray = "0.15"
// rustfft = "6"
// realfft = "3"

use ndarray::Array2;
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::error::Error;
use std::f64::consts::PI;

/// Build a real-valued 2D test array with several oscillatory components.
/// The array is chosen so that its spectrum is nontrivial but still structured.
fn build_real_test_array(n1: usize, n2: usize) -> Array2<f64> {
    let mut a = Array2::<f64>::zeros((n1, n2));

    for i in 0..n1 {
        for j in 0..n2 {
            let x = i as f64 / n1 as f64;
            let y = j as f64 / n2 as f64;

            let value = 1.2 * (2.0 * PI * x).cos()
                + 0.7 * (4.0 * PI * y).sin()
                + 0.5 * (2.0 * PI * (x + y)).cos()
                + 0.3 * (2.0 * PI * (2.0 * x - y)).sin();

            a[[i, j]] = value;
        }
    }

    a
}

/// Convert a real array to a complex array with zero imaginary part.
fn real_to_complex(a: &Array2<f64>) -> Array2<Complex<f64>> {
    let (n1, n2) = a.dim();
    let mut out = Array2::<Complex<f64>>::zeros((n1, n2));

    for i in 0..n1 {
        for j in 0..n2 {
            out[[i, j]] = Complex::new(a[[i, j]], 0.0);
        }
    }

    out
}

/// Apply a one-dimensional FFT to every row of a complex 2D array.
fn fft_rows(input: &Array2<Complex<f64>>, inverse: bool) -> Array2<Complex<f64>> {
    let (n1, n2) = input.dim();
    let mut output = input.clone();

    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n2)
    } else {
        planner.plan_fft_forward(n2)
    };

    for i in 0..n1 {
        let mut row: Vec<Complex<f64>> = (0..n2).map(|j| output[[i, j]]).collect();
        fft.process(&mut row);
        for j in 0..n2 {
            output[[i, j]] = row[j];
        }
    }

    output
}

/// Apply a one-dimensional FFT to every column of a complex 2D array.
fn fft_cols(input: &Array2<Complex<f64>>, inverse: bool) -> Array2<Complex<f64>> {
    let (n1, n2) = input.dim();
    let mut output = input.clone();

    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n1)
    } else {
        planner.plan_fft_forward(n1)
    };

    for j in 0..n2 {
        let mut col: Vec<Complex<f64>> = (0..n1).map(|i| output[[i, j]]).collect();
        fft.process(&mut col);
        for i in 0..n1 {
            output[[i, j]] = col[i];
        }
    }

    output
}

/// Full 2D complex FFT obtained by successive one-dimensional transforms.
fn fft2_complex(input: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let tmp = fft_rows(input, false);
    fft_cols(&tmp, false)
}

/// Compute a 2D real-to-complex FFT that stores only the nonredundant
/// frequencies along the last axis, producing an array of shape
/// N1 x (floor(N2/2) + 1), as in Eq. (12.7.10).
fn rfft2_half_spectrum(input: &Array2<f64>) -> Result<Array2<Complex<f64>>, Box<dyn Error>> {
    let (n1, n2) = input.dim();
    let half_n2 = n2 / 2 + 1;

    // First apply a real-to-complex FFT along each row.
    let mut real_planner = RealFftPlanner::<f64>::new();
    let r2c = real_planner.plan_fft_forward(n2);

    let mut stage1 = Array2::<Complex<f64>>::zeros((n1, half_n2));

    for i in 0..n1 {
        let mut row: Vec<f64> = (0..n2).map(|j| input[[i, j]]).collect();
        let mut out = r2c.make_output_vec();
        r2c.process(&mut row, &mut out)?;
        for j in 0..half_n2 {
            stage1[[i, j]] = out[j];
        }
    }

    // Then apply a complex FFT along the first axis for each retained frequency.
    let mut planner = FftPlanner::<f64>::new();
    let fft_n1 = planner.plan_fft_forward(n1);

    for j in 0..half_n2 {
        let mut col: Vec<Complex<f64>> = (0..n1).map(|i| stage1[[i, j]]).collect();
        fft_n1.process(&mut col);
        for i in 0..n1 {
            stage1[[i, j]] = col[i];
        }
    }

    Ok(stage1)
}

/// Reconstruct the full 2D spectrum from the stored half-spectrum using
/// the Hermitian symmetry relation in Eq. (12.7.8).
fn reconstruct_full_spectrum_from_half(half: &Array2<Complex<f64>>, n2_full: usize) -> Array2<Complex<f64>> {
    let (n1, half_n2) = half.dim();
    assert_eq!(half_n2, n2_full / 2 + 1);

    let mut full = Array2::<Complex<f64>>::zeros((n1, n2_full));

    // Copy stored frequencies.
    for i in 0..n1 {
        for j in 0..half_n2 {
            full[[i, j]] = half[[i, j]];
        }
    }

    // Reconstruct omitted frequencies using Hermitian symmetry.
    for k1 in 0..n1 {
        for k2 in 0..n2_full {
            if k2 < half_n2 {
                continue;
            }

            let j1 = (n1 - k1) % n1;
            let j2 = (n2_full - k2) % n2_full;

            full[[k1, k2]] = full[[j1, j2]].conj();
        }
    }

    full
}

/// Compute the maximum Hermitian-symmetry residual for a full complex spectrum.
/// For real input, Eq. (12.7.8) should hold.
fn hermitian_residual_2d(spec: &Array2<Complex<f64>>) -> f64 {
    let (n1, n2) = spec.dim();
    let mut max_residual = 0.0_f64;

    for k1 in 0..n1 {
        for k2 in 0..n2 {
            let j1 = (n1 - k1) % n1;
            let j2 = (n2 - k2) % n2;

            let a = spec[[k1, k2]];
            let b = spec[[j1, j2]].conj();
            max_residual = max_residual.max((a - b).norm());
        }
    }

    max_residual
}

/// Compute the maximum difference between two complex arrays of the same shape.
fn max_abs_error(a: &Array2<Complex<f64>>, b: &Array2<Complex<f64>>) -> f64 {
    let (n1, n2) = a.dim();
    assert_eq!((n1, n2), b.dim());

    let mut err = 0.0_f64;
    for i in 0..n1 {
        for j in 0..n2 {
            err = err.max((a[[i, j]] - b[[i, j]]).norm());
        }
    }

    err
}

/// Measure how far the theoretically self-conjugate frequencies are from being real.
/// These include the DC component and Nyquist boundaries when the dimensions are even.
fn self_conjugate_imag_residual(spec: &Array2<Complex<f64>>) -> f64 {
    let (n1, n2) = spec.dim();
    let mut max_imag = 0.0_f64;

    let mut special_k1 = vec![0usize];
    let mut special_k2 = vec![0usize];

    if n1 % 2 == 0 {
        special_k1.push(n1 / 2);
    }
    if n2 % 2 == 0 {
        special_k2.push(n2 / 2);
    }

    for &k1 in &special_k1 {
        for &k2 in &special_k2 {
            max_imag = max_imag.max(spec[[k1, k2]].im.abs());
        }
    }

    max_imag
}

/// Print a small leading block of a real array.
fn print_real_sample(name: &str, a: &Array2<f64>, max_rows: usize, max_cols: usize) {
    let (n1, n2) = a.dim();
    println!("{name} (shape = [{n1}, {n2}])");

    for i in 0..n1.min(max_rows) {
        for j in 0..n2.min(max_cols) {
            print!("{:10.5} ", a[[i, j]]);
        }
        println!();
    }
    println!();
}

/// Print a small leading block of a complex array.
fn print_complex_sample(name: &str, a: &Array2<Complex<f64>>, max_rows: usize, max_cols: usize) {
    let (n1, n2) = a.dim();
    println!("{name} (shape = [{n1}, {n2}])");

    for i in 0..n1.min(max_rows) {
        for j in 0..n2.min(max_cols) {
            let z = a[[i, j]];
            print!("({:+8.4},{:+8.4}i) ", z.re, z.im);
        }
        println!();
    }
    println!();
}

fn main() -> Result<(), Box<dyn Error>> {
    let n1 = 6usize;
    let n2 = 8usize;

    // Build a real-valued 2D field.
    let f_real = build_real_test_array(n1, n2);

    // Compute the full spectrum by embedding the real data in complex arithmetic.
    let f_complex = real_to_complex(&f_real);
    let full_spec = fft2_complex(&f_complex);

    // Compute the reduced half-spectrum using a real-to-complex transform.
    let half_spec = rfft2_half_spectrum(&f_real)?;

    // Reconstruct the full spectrum from the stored half-spectrum.
    let full_from_half = reconstruct_full_spectrum_from_half(&half_spec, n2);

    // Diagnostics.
    let hermitian_err = hermitian_residual_2d(&full_spec);
    let reconstruction_err = max_abs_error(&full_spec, &full_from_half);
    let boundary_imag_err = self_conjugate_imag_residual(&full_spec);

    println!("=== Program 12.7.2: Two-Dimensional Real FFTs and Hermitian Symmetry ===\n");

    print_real_sample("Input real-valued field f", &f_real, 6, 8);
    print_complex_sample("Full 2D spectrum from complex FFT", &full_spec, 6, 8);
    print_complex_sample(
        "Stored half-spectrum from real-to-complex FFT",
        &half_spec,
        6,
        5,
    );
    print_complex_sample(
        "Full spectrum reconstructed from the stored half-spectrum",
        &full_from_half,
        6,
        8,
    );

    println!(
        "Full-spectrum shape              = [{}, {}]",
        full_spec.dim().0,
        full_spec.dim().1
    );
    println!(
        "Stored half-spectrum shape       = [{}, {}]",
        half_spec.dim().0,
        half_spec.dim().1
    );
    println!(
        "Expected half-spectrum shape     = [{}, floor({}/2)+1] = [{}, {}]\n",
        n1,
        n2,
        n1,
        n2 / 2 + 1
    );

    println!(
        "Maximum Hermitian-symmetry residual (Eq. 12.7.8): {:.3e}",
        hermitian_err
    );
    println!(
        "Maximum error after reconstructing full spectrum from half-spectrum: {:.3e}",
        reconstruction_err
    );
    println!(
        "Maximum imaginary part on self-conjugate frequencies: {:.3e}",
        boundary_imag_err
    );

    if hermitian_err > 1e-10 {
        return Err(format!(
            "Hermitian symmetry residual is too large: {:.3e}",
            hermitian_err
        )
        .into());
    }

    if reconstruction_err > 1e-10 {
        return Err(format!(
            "reconstructed full spectrum does not match the direct full spectrum closely enough: {:.3e}",
            reconstruction_err
        )
        .into());
    }

    Ok(())
}
```

Program 12.7.2 demonstrates the practical consequences of Hermitian symmetry in multidimensional Fourier transforms of real-valued data. Because the input field is real, each Fourier coefficient has a conjugate counterpart located at the corresponding negative frequency, as expressed by equation (12.7.8). This symmetry ensures that only approximately half of the spectral coefficients contain independent information.

The program verifies this property numerically by computing both the full complex spectrum and the reduced half-spectrum produced by a real-to-complex FFT. The reconstruction of the full spectrum from the stored half-spectrum confirms that the omitted coefficients can be recovered using the conjugate symmetry relation. The numerical diagnostics further show that coefficients corresponding to self-conjugate frequencies are purely real, consistent with the theoretical structure of the transform.

These results illustrate an important practical advantage of real-data FFT algorithms: although the arithmetic complexity remains $O(N \log N)$, the amount of stored data and the associated memory traffic can be significantly reduced. This reduction becomes particularly important in large multidimensional simulations and imaging applications, where memory bandwidth often dominates overall performance.

The implementation therefore provides a concrete illustration of how mathematical structure translates directly into algorithmic efficiency. By exploiting Hermitian symmetry, modern FFT libraries achieve substantial savings in storage and computation while preserving the full spectral information contained in the original data.

## 12.7.4. Three-Dimensional Real FFTs

For a real three-dimensional field $f\in\mathbb{R}^{N_1\times N_2\times N_3}$, Hermitian symmetry takes the form:

$$
F_{(N_1-k_1)\bmod N_1,\,(N_2-k_2)\bmod N_2,\,(N_3-k_3)\bmod N_3}
=
\overline{F_{k_1,k_2,k_3}}
\tag{12.7.12}
$$

A standard storage convention retains the frequencies,

$$
0 \le k_1 < N_1,\qquad
0 \le k_2 < N_2,\qquad
0 \le k_3 \le \left\lfloor \frac{N_3}{2} \right\rfloor
\tag{12.7.13}
$$

This yields a stored array of size:

$$
N_1 \times N_2 \times \left(\left\lfloor \frac{N_3}{2} \right\rfloor + 1\right)
\tag{12.7.14}
$$

When $N_3$ is even, the frequency planes,

$$
k_3 = 0, \qquad k_3 = \frac{N_3}{2}
\tag{12.7.15}
$$

map onto themselves under the symmetry transformation and therefore satisfy an additional two-dimensional Hermitian relation. Many implementations nevertheless store these planes explicitly to avoid complicated indexing logic.

## 12.7.5. Real-to-Complex Storage Layouts

A one-dimensional real-to-complex transform produces $N/2+1$ complex coefficients. Since each complex value contains two real numbers, this corresponds to:

$$2\left(\frac{N}{2}+1\right)=N+2 \tag{12.7.16}$$

real storage locations. Consequently, strictly in-place real-to-complex transforms require a small amount of padding.

For three-dimensional data, a commonly used padded layout allocates:

$$N_1 \times N_2 \times (N_3 + 2) \tag{12.7.17}$$

real storage so that the logical complex array,

$$\text{spec}\in\mathbb{C}^{N_1\times N_2\times \left(\frac{N_3}{2}+1\right)} \tag{12.7.18}$$

can be represented by interleaving real and imaginary components:

$$\Re(\text{spec}[i_1,i_2,i_3])=\text{data}[i_1,i_2,2i_3]  \tag{12.7.19}$$

$$\Im(\text{spec}[i_1,i_2,i_3])=\text{data}[i_1,i_2,2i_3+1] \tag{12.7.20}$$

Recent research has explored alternative real-domain encodings in which conjugate frequency pairs are stored at symmetric positions within a real buffer,

$$
\Re(y_k)\text{ at index }k,\qquad
\Im(y_k)\text{ at index }(r-k),\qquad
1 \le k < \frac{r}{2} \tag{12.7.21}
$$

while $y_0$ and $y_{r/2}$ occupy single real locations. Such schemes maintain Hermitian symmetry1throughout the FFT stages and can enable fully in-place execution for real transforms (Ding et al., 2025).

### Rust Implementation

Following the discussion in Sections 12.7.4 and 12.7.5 on three-dimensional Fourier transforms of real data and the associated storage layouts, Program 12.7.3 provides a practical implementation illustrating these structural properties. In multidimensional spectral computations, real-valued input fields lead to Hermitian symmetry in the Fourier domain, allowing the transform to store only a reduced set of frequency coefficients while reconstructing the remainder through symmetry relations. This program demonstrates how a three-dimensional real field can be transformed using a sequence of one-dimensional FFTs, how the reduced half-spectrum representation emerges naturally from this structure, and how the full spectrum can be recovered using the symmetry relation derived in Equation (12.7.12). It further illustrates the padded real storage layout described in Equations (12.7.17)–(12.7.20), which is commonly used in practical FFT libraries to enable efficient in-place real-to-complex transforms. The implementation therefore connects the theoretical structure of multidimensional real FFTs with concrete memory layouts and computational procedures used in numerical practice.

At the core of the implementation is the construction of a real three-dimensional test field, produced by the function `build_real_test_array`. This routine generates a smooth spatial signal by combining several trigonometric modes in the three coordinate directions. Such fields are typical in spectral simulations, where solutions are represented as sums of oscillatory basis functions. The resulting array represents the discrete sampling of a real-valued function $f_{n_1,n_2,n_3}$, consistent with the multidimensional Fourier formulation introduced in Equation (12.7.4).

The function `real_to_complex` converts the real data array into a complex array with zero imaginary parts. This conversion allows the program to compute a reference spectrum using the standard complex FFT pipeline. The complex transform is performed by the routine `fft3_complex`, which applies successive one-dimensional FFTs along each coordinate axis. Internally, this function calls `fft_axis2`, `fft_axis1`, and `fft_axis0`, each of which performs a one-dimensional transform along a specific dimension. This axis-by-axis approach reflects the tensor-product structure of multidimensional Fourier transforms described earlier in the section and corresponds to the separability implied by Equation (12.7.6).

The program then computes a reduced spectrum using the routine `rfft3_half_spectrum`. This function performs a real-to-complex transform along the third axis and complex transforms along the remaining axes, producing a spectral array of size $N_1 \times N_2 \times \left(\left\lfloor \frac{N_3}{2} \right\rfloor + 1\right)$, which corresponds directly to the storage convention introduced in Equation (12.7.14). Because the input field is real, the omitted frequencies can be reconstructed from the stored ones using the Hermitian symmetry relation given in Equation (12.7.12).

To verify this property, the routine `reconstruct_full_spectrum_from_half` reconstructs the full spectral array from the stored half-spectrum. For each frequency index $(k_1,k_2,k_3)$ that is not explicitly stored, the program retrieves the conjugate coefficient at the corresponding negative frequency and assigns its complex conjugate. The correctness of this reconstruction is assessed using the function `hermitian_residual_3d`, which measures the maximum deviation from the symmetry condition expressed in Equation (12.7.12).

The implementation also demonstrates the padded real-storage layout commonly used by FFT libraries. The function `pack_half_spectrum_to_padded_real` maps the complex half-spectrum into a real buffer of size $N_1 \times N_2 \times (N_3+2),$ as described in Equation (12.7.17). Real and imaginary components of each spectral coefficient are stored in adjacent locations according to Equations (12.7.19) and (12.7.20). The inverse mapping is implemented by `unpack_half_spectrum_from_padded_real`, which reconstructs the logical complex array from the padded real buffer. The correctness of this mapping is verified numerically by comparing the unpacked coefficients with the original half-spectrum.

The `main` function orchestrates the entire demonstration. It constructs a real three-dimensional test field, computes both the full complex spectrum and the reduced real-to-complex spectrum, reconstructs the full spectrum from the stored coefficients, and evaluates several diagnostic quantities. These diagnostics include the Hermitian symmetry residual, the reconstruction error between the direct and reconstructed spectra, and the residual on the self-conjugate planes corresponding to the frequencies $k_3 = 0$ and $k_3 = N_3/2$, which are discussed in Equation (12.7.15). The program also verifies that the padded storage layout and unpacking procedure reproduce the same spectral data to machine precision.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
ndarray = "0.15"
rustfft = "6"
realfft = "3"
```

```rust
// Program 12.7.3: Three-Dimensional Real FFTs and Real-to-Complex Storage Layouts
//
// This program illustrates the three-dimensional Hermitian symmetry relation
// for real-valued data, the reduced half-spectrum storage convention, and the
// padded in-place real storage layout described in Eqs. (12.7.12)–(12.7.20).
//
// The implementation demonstrates:
//
// 1. A full 3D complex FFT of real data embedded in complex form.
// 2. A 3D real-to-complex FFT that stores only the half-spectrum along the
//    last axis, with shape N1 x N2 x (floor(N3/2) + 1).
// 3. Reconstruction of the full spectrum from the stored half-spectrum using
//    Hermitian symmetry.
// 4. The padded storage layout N1 x N2 x (N3 + 2), where the logical complex
//    spectrum is interleaved as real and imaginary parts in a real buffer.
//
// Cargo.toml dependencies:
// [dependencies]
// ndarray = "0.15"
// rustfft = "6"
// realfft = "3"

use ndarray::Array3;
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::error::Error;
use std::f64::consts::PI;

/// Build a real-valued 3D test field with several oscillatory modes.
fn build_real_test_array(n1: usize, n2: usize, n3: usize) -> Array3<f64> {
    let mut a = Array3::<f64>::zeros((n1, n2, n3));

    for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..n3 {
                let x = i as f64 / n1 as f64;
                let y = j as f64 / n2 as f64;
                let z = k as f64 / n3 as f64;

                let value = 1.1 * (2.0 * PI * x).cos()
                    + 0.8 * (2.0 * PI * y).sin()
                    + 0.6 * (4.0 * PI * z).cos()
                    + 0.4 * (2.0 * PI * (x + y - z)).sin()
                    + 0.3 * (2.0 * PI * (2.0 * x + z)).cos();

                a[[i, j, k]] = value;
            }
        }
    }

    a
}

/// Convert a real 3D array to complex form with zero imaginary part.
fn real_to_complex(a: &Array3<f64>) -> Array3<Complex<f64>> {
    let (n1, n2, n3) = a.dim();
    let mut out = Array3::<Complex<f64>>::zeros((n1, n2, n3));

    for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..n3 {
                out[[i, j, k]] = Complex::new(a[[i, j, k]], 0.0);
            }
        }
    }

    out
}

/// Apply a 1D complex FFT along the last axis of a 3D array.
fn fft_axis2(input: &Array3<Complex<f64>>, inverse: bool) -> Array3<Complex<f64>> {
    let (n1, n2, n3) = input.dim();
    let mut output = input.clone();

    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n3)
    } else {
        planner.plan_fft_forward(n3)
    };

    for i in 0..n1 {
        for j in 0..n2 {
            let mut line: Vec<Complex<f64>> = (0..n3).map(|k| output[[i, j, k]]).collect();
            fft.process(&mut line);
            for k in 0..n3 {
                output[[i, j, k]] = line[k];
            }
        }
    }

    output
}

/// Apply a 1D complex FFT along the middle axis of a 3D array.
fn fft_axis1(input: &Array3<Complex<f64>>, inverse: bool) -> Array3<Complex<f64>> {
    let (n1, n2, n3) = input.dim();
    let mut output = input.clone();

    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n2)
    } else {
        planner.plan_fft_forward(n2)
    };

    for i in 0..n1 {
        for k in 0..n3 {
            let mut line: Vec<Complex<f64>> = (0..n2).map(|j| output[[i, j, k]]).collect();
            fft.process(&mut line);
            for j in 0..n2 {
                output[[i, j, k]] = line[j];
            }
        }
    }

    output
}

/// Apply a 1D complex FFT along the first axis of a 3D array.
fn fft_axis0(input: &Array3<Complex<f64>>, inverse: bool) -> Array3<Complex<f64>> {
    let (n1, n2, n3) = input.dim();
    let mut output = input.clone();

    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n1)
    } else {
        planner.plan_fft_forward(n1)
    };

    for j in 0..n2 {
        for k in 0..n3 {
            let mut line: Vec<Complex<f64>> = (0..n1).map(|i| output[[i, j, k]]).collect();
            fft.process(&mut line);
            for i in 0..n1 {
                output[[i, j, k]] = line[i];
            }
        }
    }

    output
}

/// Full 3D complex FFT via successive axis transforms.
fn fft3_complex(input: &Array3<Complex<f64>>) -> Array3<Complex<f64>> {
    let tmp = fft_axis2(input, false);
    let tmp = fft_axis1(&tmp, false);
    fft_axis0(&tmp, false)
}

/// Compute a 3D real-to-complex FFT storing only the half-spectrum along the last axis.
fn rfft3_half_spectrum(input: &Array3<f64>) -> Result<Array3<Complex<f64>>, Box<dyn Error>> {
    let (n1, n2, n3) = input.dim();
    let half_n3 = n3 / 2 + 1;

    // Stage 1: real-to-complex FFT along the last axis.
    let mut real_planner = RealFftPlanner::<f64>::new();
    let r2c = real_planner.plan_fft_forward(n3);

    let mut stage1 = Array3::<Complex<f64>>::zeros((n1, n2, half_n3));

    for i in 0..n1 {
        for j in 0..n2 {
            let mut line: Vec<f64> = (0..n3).map(|k| input[[i, j, k]]).collect();
            let mut out = r2c.make_output_vec();
            r2c.process(&mut line, &mut out)?;
            for k in 0..half_n3 {
                stage1[[i, j, k]] = out[k];
            }
        }
    }

    // Stage 2: complex FFT along the middle axis.
    let mut planner = FftPlanner::<f64>::new();
    let fft_n2 = planner.plan_fft_forward(n2);
    for i in 0..n1 {
        for k in 0..half_n3 {
            let mut line: Vec<Complex<f64>> = (0..n2).map(|j| stage1[[i, j, k]]).collect();
            fft_n2.process(&mut line);
            for j in 0..n2 {
                stage1[[i, j, k]] = line[j];
            }
        }
    }

    // Stage 3: complex FFT along the first axis.
    let fft_n1 = planner.plan_fft_forward(n1);
    for j in 0..n2 {
        for k in 0..half_n3 {
            let mut line: Vec<Complex<f64>> = (0..n1).map(|i| stage1[[i, j, k]]).collect();
            fft_n1.process(&mut line);
            for i in 0..n1 {
                stage1[[i, j, k]] = line[i];
            }
        }
    }

    Ok(stage1)
}

/// Reconstruct the full 3D spectrum from the stored half-spectrum using Eq. (12.7.12).
fn reconstruct_full_spectrum_from_half(
    half: &Array3<Complex<f64>>,
    n3_full: usize,
) -> Array3<Complex<f64>> {
    let (n1, n2, half_n3) = half.dim();
    assert_eq!(half_n3, n3_full / 2 + 1);

    let mut full = Array3::<Complex<f64>>::zeros((n1, n2, n3_full));

    // Copy stored frequencies.
    for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..half_n3 {
                full[[i, j, k]] = half[[i, j, k]];
            }
        }
    }

    // Reconstruct omitted frequencies.
    for k1 in 0..n1 {
        for k2 in 0..n2 {
            for k3 in half_n3..n3_full {
                let j1 = (n1 - k1) % n1;
                let j2 = (n2 - k2) % n2;
                let j3 = (n3_full - k3) % n3_full;
                full[[k1, k2, k3]] = full[[j1, j2, j3]].conj();
            }
        }
    }

    full
}

/// Hermitian-symmetry residual for a full 3D spectrum of real input.
fn hermitian_residual_3d(spec: &Array3<Complex<f64>>) -> f64 {
    let (n1, n2, n3) = spec.dim();
    let mut max_residual = 0.0_f64;

    for k1 in 0..n1 {
        for k2 in 0..n2 {
            for k3 in 0..n3 {
                let j1 = (n1 - k1) % n1;
                let j2 = (n2 - k2) % n2;
                let j3 = (n3 - k3) % n3;

                let a = spec[[k1, k2, k3]];
                let b = spec[[j1, j2, j3]].conj();
                max_residual = max_residual.max((a - b).norm());
            }
        }
    }

    max_residual
}

/// Maximum difference between two 3D complex arrays.
fn max_abs_error_3d(a: &Array3<Complex<f64>>, b: &Array3<Complex<f64>>) -> f64 {
    let (n1, n2, n3) = a.dim();
    assert_eq!((n1, n2, n3), b.dim());

    let mut err = 0.0_f64;
    for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..n3 {
                err = err.max((a[[i, j, k]] - b[[i, j, k]]).norm());
            }
        }
    }

    err
}

/// Measure the maximum imaginary part on the self-conjugate planes k3 = 0 and, if even, k3 = N3/2.
fn self_conjugate_plane_residual(spec: &Array3<Complex<f64>>) -> f64 {
    let (n1, n2, n3) = spec.dim();
    let mut max_imag = 0.0_f64;

    let special_k3 = if n3 % 2 == 0 {
        vec![0usize, n3 / 2]
    } else {
        vec![0usize]
    };

    for &k3 in &special_k3 {
        for k1 in 0..n1 {
            for k2 in 0..n2 {
                let j1 = (n1 - k1) % n1;
                let j2 = (n2 - k2) % n2;
                let a = spec[[k1, k2, k3]];
                let b = spec[[j1, j2, k3]].conj();
                max_imag = max_imag.max((a - b).norm());
            }
        }
    }

    max_imag
}

/// Pack the logical half-spectrum into a padded real buffer of shape N1 x N2 x (N3 + 2),
/// using the interleaving described in Eqs. (12.7.19) and (12.7.20).
fn pack_half_spectrum_to_padded_real(
    half: &Array3<Complex<f64>>,
    n3_original: usize,
) -> Array3<f64> {
    let (n1, n2, half_n3) = half.dim();
    assert_eq!(half_n3, n3_original / 2 + 1);

    let mut data = Array3::<f64>::zeros((n1, n2, n3_original + 2));

    for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..half_n3 {
                data[[i, j, 2 * k]] = half[[i, j, k]].re;
                data[[i, j, 2 * k + 1]] = half[[i, j, k]].im;
            }
        }
    }

    data
}

/// Unpack the logical half-spectrum from a padded real buffer.
fn unpack_half_spectrum_from_padded_real(
    data: &Array3<f64>,
    n3_original: usize,
) -> Array3<Complex<f64>> {
    let (n1, n2, padded_n3) = data.dim();
    assert_eq!(padded_n3, n3_original + 2);

    let half_n3 = n3_original / 2 + 1;
    let mut half = Array3::<Complex<f64>>::zeros((n1, n2, half_n3));

    for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..half_n3 {
                half[[i, j, k]] = Complex::new(data[[i, j, 2 * k]], data[[i, j, 2 * k + 1]]);
            }
        }
    }

    half
}

/// Print a few sample entries of a real 3D array.
fn print_real_sample(name: &str, a: &Array3<f64>, count: usize) {
    let (n1, n2, n3) = a.dim();
    println!("{name} (shape = [{n1}, {n2}, {n3}])");

    let mut shown = 0usize;
    'outer: for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..n3 {
                println!("  [{:02},{:02},{:02}] = {:+.6}", i, j, k, a[[i, j, k]]);
                shown += 1;
                if shown >= count {
                    break 'outer;
                }
            }
        }
    }
    println!();
}

/// Print a few sample entries of a complex 3D array.
fn print_complex_sample(name: &str, a: &Array3<Complex<f64>>, count: usize) {
    let (n1, n2, n3) = a.dim();
    println!("{name} (shape = [{n1}, {n2}, {n3}])");

    let mut shown = 0usize;
    'outer: for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..n3 {
                let z = a[[i, j, k]];
                println!(
                    "  [{:02},{:02},{:02}] = {:+.6} {:+.6}i",
                    i, j, k, z.re, z.im
                );
                shown += 1;
                if shown >= count {
                    break 'outer;
                }
            }
        }
    }
    println!();
}

fn main() -> Result<(), Box<dyn Error>> {
    let n1 = 4usize;
    let n2 = 5usize;
    let n3 = 6usize; // even, so the k3 = 0 and k3 = N3/2 planes are both self-conjugate

    let f_real = build_real_test_array(n1, n2, n3);
    let f_complex = real_to_complex(&f_real);

    // Full spectrum from complex FFT of embedded real data.
    let full_spec = fft3_complex(&f_complex);

    // Reduced spectrum from real-to-complex FFT.
    let half_spec = rfft3_half_spectrum(&f_real)?;

    // Reconstruct the full spectrum from the stored half-spectrum.
    let full_from_half = reconstruct_full_spectrum_from_half(&half_spec, n3);

    // Padded real storage layout.
    let padded_real = pack_half_spectrum_to_padded_real(&half_spec, n3);
    let half_unpacked = unpack_half_spectrum_from_padded_real(&padded_real, n3);

    // Diagnostics.
    let hermitian_err = hermitian_residual_3d(&full_spec);
    let reconstruction_err = max_abs_error_3d(&full_spec, &full_from_half);
    let padded_roundtrip_err = max_abs_error_3d(&half_spec, &half_unpacked);
    let self_plane_err = self_conjugate_plane_residual(&full_spec);

    println!("=== Program 12.7.3: Three-Dimensional Real FFTs and Storage Layouts ===\n");

    print_real_sample("Input real-valued field f", &f_real, 10);
    print_complex_sample("Full 3D spectrum from complex FFT", &full_spec, 12);
    print_complex_sample("Stored half-spectrum", &half_spec, 12);
    print_complex_sample(
        "Full spectrum reconstructed from the half-spectrum",
        &full_from_half,
        12,
    );
    print_real_sample("Padded real storage buffer", &padded_real, 16);

    println!(
        "Full-spectrum shape          = [{}, {}, {}]",
        full_spec.dim().0,
        full_spec.dim().1,
        full_spec.dim().2
    );
    println!(
        "Stored half-spectrum shape   = [{}, {}, {}]",
        half_spec.dim().0,
        half_spec.dim().1,
        half_spec.dim().2
    );
    println!(
        "Expected half-spectrum shape = [{}, {}, floor({}/2)+1] = [{}, {}, {}]",
        n1,
        n2,
        n3,
        n1,
        n2,
        n3 / 2 + 1
    );
    println!(
        "Padded real-buffer shape     = [{}, {}, {}]\n",
        padded_real.dim().0,
        padded_real.dim().1,
        padded_real.dim().2
    );

    println!(
        "Maximum Hermitian-symmetry residual (Eq. 12.7.12): {:.3e}",
        hermitian_err
    );
    println!(
        "Maximum error after reconstructing full spectrum from half-spectrum: {:.3e}",
        reconstruction_err
    );
    println!(
        "Maximum residual on self-conjugate planes k3 = 0 and k3 = N3/2: {:.3e}",
        self_plane_err
    );
    println!(
        "Maximum error after packing and unpacking padded real storage: {:.3e}",
        padded_roundtrip_err
    );

    // Also verify the real-buffer padding rule numerically.
    let expected_padded_shape = (n1, n2, n3 + 2);
    assert_eq!(padded_real.dim(), expected_padded_shape);

    if hermitian_err > 1e-10 {
        return Err(format!(
            "Hermitian symmetry residual is too large: {:.3e}",
            hermitian_err
        )
        .into());
    }

    if reconstruction_err > 1e-10 {
        return Err(format!(
            "reconstructed full spectrum does not match the direct full spectrum closely enough: {:.3e}",
            reconstruction_err
        )
        .into());
    }

    if padded_roundtrip_err > 1e-12 {
        return Err(format!(
            "padded real-storage pack/unpack error is too large: {:.3e}",
            padded_roundtrip_err
        )
        .into());
    }

    Ok(())
}
```

Program 12.7.3 demonstrates how the structural properties of Fourier transforms of real data translate into concrete computational procedures. By exploiting the Hermitian symmetry described in Equation (12.7.12), the implementation stores only the nonredundant portion of the spectrum while preserving the ability to reconstruct the complete Fourier representation. This reduction in storage is particularly important in large-scale three-dimensional simulations, where the total number of spectral coefficients can be extremely large.

The numerical diagnostics confirm the theoretical structure of the transform. The Hermitian symmetry residual is essentially zero, indicating that the computed spectrum satisfies the symmetry relation to machine precision. Likewise, the reconstruction error between the full complex spectrum and the spectrum reconstructed from the stored half-spectrum remains at the level of floating-point roundoff, demonstrating the correctness of the symmetry-based reconstruction procedure.

The program also illustrates the padded real storage layout used in many high-performance FFT libraries. By interleaving real and imaginary components within a real buffer of size $N_1 \times N_2 \times (N_3+2)$, this layout enables efficient in-place transforms while maintaining compatibility with complex-valued spectral representations. Such storage schemes play a central role in practical FFT implementations for large multidimensional data sets.

Together, these experiments highlight how mathematical structure, algorithm design, and memory layout interact in modern numerical Fourier transform implementations. Understanding these relationships is essential when developing efficient spectral solvers for problems in scientific computing, signal processing, and large-scale simulation.

## 12.7.6. Practical Considerations

Although the arithmetic complexity of multidimensional FFTs remains $O(N\log N)$, real-data transforms highlight the importance of memory layout and data locality. Transformations along contiguous axes are cache-friendly, whereas transforms along non-contiguous axes introduce strided memory access and often require intermediate data rearrangements. Consequently, axis ordering, storage layout, and symmetry-aware data packing become essential components of efficient multidimensional FFT implementations.

The next section examines how these considerations become even more critical when data sets exceed available memory and FFT computations must be organized using external storage and out-of-core execution strategies.

# 12.8 External Storage and Memory-Local FFT Execution

Classical FFT analysis focuses on reducing the arithmetic complexity of the discrete Fourier transform from $O(N^2)$ to $O(N \log N)$. In contemporary large-scale computation, however, arithmetic cost is often secondary. For multidimensional transforms and extreme-scale workloads, the dominant constraint is data movement rather than floating-point throughput. This shift is visible across scientific computing, signal processing, computational imaging, and large simulation workflows in which FFTs are embedded inside iterative or convolution-heavy pipelines.

The practical performance of an FFT is therefore frequently governed by where the bottleneck resides in the memory hierarchy. When the working set fits in cache but not in registers, cache reuse and permutation locality dominate. When the working set fits in main memory but not in cache, memory bandwidth and TLB behavior become critical. In distributed memory, communication phases can overshadow computation. When data exceed main memory entirely, the FFT becomes an out-of-core problem, and I/O throughput dictates feasibility.

Modern research reflects this hierarchy. Memory-efficiency-oriented FFT redesigns aim to reduce cache and TLB miss rates (Servodio and Li, 2024). GPU-oriented work analyzes locality breaks between kernel stages and energy–performance tradeoffs (Yang et al., 2025). Communication-minimizing algorithms reduce all-to-all redistribution phases in distributed multidimensional FFTs (Koopman and Bisseling, 2023). Out-of-core diffraction pipelines restructure FFT workflows around SSD bandwidth and block-wise streaming (Lee and Kim, 2023; Lee and Kim, 2024). These developments collectively demonstrate that the FFT must be treated not merely as an algorithmic primitive, but as a dataflow system embedded within a hardware hierarchy.

## 12.8.1. Blocked Decomposition and Out-of-Core FFT Structure

The mathematical core of the FFT remains the Cooley–Tukey factorization. Let $N = N_1 N_2$, and rewrite indices as:

$$n = n_1 + N_1 n_2,\qquad k = k_2 + N_2 k_1 \tag{12.8.1}$$

with,

$$0 \le n_1 < N_1, \quad0 \le n_2 < N_2 \tag{12.8.2}$$

Substituting (12.8.1) into the DFT definition yields a factorization into three stages:

1. Local transforms of size $N_1$,
2. Multiplication by twiddle factors,
3. Transforms of size $N_2$ after an index permutation.

Algebraically, this corresponds to rewriting the DFT matrix as a product of sparse structured matrices and diagonal twiddle matrices. In memory-resident implementations, the index permutation is realized by either strided access or explicit transposition. In out-of-core settings, however, this permutation becomes a block-wise data reorganization step involving external storage.

The design principle in out-of-core FFTs is therefore to align decomposition with storage constraints. Blocks are chosen so that each loaded chunk fits into available memory, undergoes sufficient computation to amortize I/O cost, and is written back in a layout suitable for the next stage. Sequential or block-sequential access is favored to maximize SSD throughput. Minimizing the number of passes over the dataset becomes essential, since each pass incurs I/O latency that can dominate runtime.

Although the classical disk-based FFT literature predates modern hardware, the same structural decomposition underlies contemporary SSD-backed and GPU-accelerated out-of-core pipelines. The FFT remains separable; what changes is the granularity and ordering of the passes.

## 12.8.2. Modern Out-of-Core Practice: Multi-SSD and Compression Strategies

A concrete modern example arises in ultra-high-resolution holography and diffraction simulation, where convolution steps are implemented via FFTs on arrays that exceed main memory capacity. In such settings, the convolution theorem reduces quadratic convolution cost to repeated FFTs and pointwise multiplications,

$$\mathcal{F}^{-1}\bigl(\mathcal{F}(f)\cdot \mathcal{F}(g)\bigr) \tag{12.8.3}$$

but the dominant cost shifts to managing massive intermediate arrays.

Lee and Kim (2023) present an out-of-core diffraction algorithm that distributes data across multiple SSDs and reports continued performance gains as SSD count increases. Their approach reorganizes the FFT workflow so that data blocks are streamed sequentially through memory, reducing random-access penalties. In subsequent work, the COMBO framework (Lee and Kim, 2024) integrates block-wise data handling with GPU-accelerated compression to reduce I/O volume. The reported results include substantial speedups relative to prior out-of-core methods and demonstrate the feasibility of constructing terabyte-scale holographic computations on systems with far smaller main-memory capacity.

From a numerical computing perspective, these results illustrate a central principle: when the FFT is embedded in a larger convolution-based workflow, arithmetic complexity becomes subordinate to I/O orchestration. Block size, storage order, compression strategy, and pass scheduling become algorithmic parameters in their own right.

### Rust Implementation

Following the discussion in Section 12.8 on external storage and memory-local FFT execution, Program 12.8.1 provides a practical implementation of a blocked out-of-core Cooley–Tukey fast Fourier transform. While classical FFT theory emphasizes reducing arithmetic complexity from $O(N^2)$ to $O(N\log N)$, large-scale computations are often limited not by floating-point throughput but by the cost of moving data through the memory hierarchy. When datasets exceed available memory capacity, FFT execution must be reorganized so that intermediate data are processed in blocks and streamed through external storage. This program demonstrates how the Cooley–Tukey factorization can be mapped onto such a workflow by explicitly partitioning the input signal into disk-resident blocks, performing local transforms on each block, applying the necessary twiddle factors, and reorganizing the intermediate data before completing the remaining transform stages. The implementation simulates multi-SSD striping by distributing blocks across several directories and verifies correctness by comparing the out-of-core FFT result with a direct DFT computation. In this way, the example illustrates how algorithmic structure and dataflow must be coordinated when FFT computations extend beyond the capacity of main memory.

At the core of the implementation is the Cooley–Tukey factorization described in Section 12.8.1. The discrete Fourier transform of length $N=N_1N_2$ is decomposed by rewriting the sample index according to Equation (12.8.1). This reformulation expresses the transform as a sequence of smaller transforms combined with twiddle-factor multiplications and index permutations. The program implements this factorization explicitly so that the intermediate data can be stored and retrieved in block form from external storage.

The `BlockStore` structure provides the abstraction used to manage external storage. It simulates a multi-device storage environment by distributing blocks across several directories that represent independent SSDs. Each block is written to disk using the `write_complex_block` method and retrieved using the `read_complex_block` method. Both functions serialize complex numbers into binary form and maintain detailed I/O statistics through the `IoStats` structure. These statistics record the number of block reads and writes and the total number of bytes transferred, making the I/O cost of the algorithm visible during execution.

The `build_input_signal` function constructs a synthetic complex-valued test signal whose Fourier spectrum contains several distinct frequency components. This signal is used to demonstrate the correctness of the out-of-core FFT pipeline. For validation, the `direct_dft` function computes the transform directly using the definition of the discrete Fourier transform. Although this method has $O(N^2)$ complexity and is therefore impractical for large datasets, it provides an exact reference solution against which the blocked FFT result can be compared.

The central computation is implemented in the `out_of_core_fft_cooley_tukey` function. This routine performs the transform in several stages corresponding to the mathematical decomposition. First, the input vector is partitioned into $N_2$ blocks of length $N_1$, corresponding to fixed values of the index $n_2$. Each block represents the subsequence $x[n_1N_2+n_2]$, which forms the columns of the factorized representation. Next, an $N_1$-point FFT is applied to each column using the `rustfft` library. After these local transforms, the algorithm multiplies each coefficient by the appropriate twiddle factor $W_N^{n_2 k_1}$, implementing the phase adjustment that arises from the factorization.

Once the twiddle factors have been applied, the data must be reorganized so that the second stage of transforms can be executed. The program performs this step by gathering values corresponding to each $k_1$ index into new blocks of length $N_2$. These blocks represent the rows of the factorized transform matrix. Finally, an $N_2$-point FFT is applied to each row to produce the final Fourier coefficients ordered as $X[k_1 + N_1k_2]$. This sequence of steps corresponds directly to the three-stage decomposition described in Section 12.8.1.

The `main` function orchestrates the complete demonstration. It defines the factorization parameters $N_1$ and $N_2$, constructs the input signal, and invokes the blocked FFT routine. The result is then compared with the direct DFT output using the `max_abs_error` function, which computes the largest absolute difference between the two spectra. The program prints representative samples of the input signal, the computed FFT, and the reference transform, along with detailed statistics describing disk usage and block I/O operations. These diagnostics reveal how the transform proceeds through successive passes over the dataset and illustrate the additional data movement required by out-of-core execution.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rustfft = "6"
```

```rust
// Program 12.8.1: Blocked Out-of-Core Cooley–Tukey FFT with Simulated Multi-SSD Striping
//
// This program illustrates a blocked out-of-core Cooley–Tukey FFT workflow.
// It uses the factorization
//
//   n = n1*N2 + n2,   k = k1 + N1*k2
//
// for N = N1 * N2.
//
// The computation proceeds in five stages:
//
//   1. Partition input into N2 columns of length N1 (x[n1*N2+n2], fixed n2).
//   2. N1-DFTs on each column.
//   3. Twiddle-factor multiplication W_N^(n2*k1).
//   4. Permute into N1 rows of length N2 (for each k1, gather over n2).
//   5. N2-DFTs on each row; output X[k1 + N1*k2].
//
// The implementation is pedagogical rather than performance-tuned. It makes the
// dataflow explicit by writing and reading blocks from files distributed across
// simulated SSD directories.
//
// Cargo.toml dependencies:
// [dependencies]
// rustfft = "6"

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::error::Error;
use std::f64::consts::PI;
use std::fs::{create_dir_all, read_dir, remove_dir_all, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Clone)]
struct IoStats {
    bytes_read: u64,
    bytes_written: u64,
    block_reads: u64,
    block_writes: u64,
}

impl IoStats {
    fn total_bytes(&self) -> u64 {
        self.bytes_read + self.bytes_written
    }
}

#[derive(Debug)]
struct BlockStore {
    roots: Vec<PathBuf>,
    stats: IoStats,
}

impl BlockStore {
    fn new<P: AsRef<Path>>(base_dir: P, num_devices: usize) -> Result<Self, Box<dyn Error>> {
        if num_devices == 0 {
            return Err("num_devices must be positive".into());
        }

        let base = base_dir.as_ref();
        if base.exists() {
            remove_dir_all(base)?;
        }
        create_dir_all(base)?;

        let mut roots = Vec::with_capacity(num_devices);
        for d in 0..num_devices {
            let dir = base.join(format!("ssd_{d}"));
            create_dir_all(&dir)?;
            roots.push(dir);
        }

        Ok(Self {
            roots,
            stats: IoStats::default(),
        })
    }

    fn path_for_block(&self, stage: &str, block_id: usize) -> PathBuf {
        let dev = block_id % self.roots.len();
        self.roots[dev].join(format!("{stage}_block_{block_id:05}.bin"))
    }

    fn write_complex_block(
        &mut self,
        stage: &str,
        block_id: usize,
        data: &[Complex<f64>],
    ) -> Result<(), Box<dyn Error>> {
        let path = self.path_for_block(stage, block_id);
        let mut file = File::create(path)?;

        for z in data {
            file.write_all(&z.re.to_le_bytes())?;
            file.write_all(&z.im.to_le_bytes())?;
        }

        self.stats.bytes_written += (16 * data.len()) as u64;
        self.stats.block_writes += 1;
        Ok(())
    }

    fn read_complex_block(
        &mut self,
        stage: &str,
        block_id: usize,
        expected_len: usize,
    ) -> Result<Vec<Complex<f64>>, Box<dyn Error>> {
        let path = self.path_for_block(stage, block_id);
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        let expected_bytes = 16 * expected_len;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "block {block_id} in stage '{stage}' has {} bytes, expected {}",
                bytes.len(),
                expected_bytes
            )
            .into());
        }

        let mut out = Vec::with_capacity(expected_len);
        for chunk in bytes.chunks_exact(16) {
            let mut re_bytes = [0u8; 8];
            let mut im_bytes = [0u8; 8];
            re_bytes.copy_from_slice(&chunk[0..8]);
            im_bytes.copy_from_slice(&chunk[8..16]);
            out.push(Complex::new(
                f64::from_le_bytes(re_bytes),
                f64::from_le_bytes(im_bytes),
            ));
        }

        self.stats.bytes_read += bytes.len() as u64;
        self.stats.block_reads += 1;
        Ok(out)
    }

    fn print_layout(&self) -> Result<(), Box<dyn Error>> {
        println!("Simulated SSD layout:");
        for (d, root) in self.roots.iter().enumerate() {
            let mut count = 0usize;
            for entry in read_dir(root)? {
                let _ = entry?;
                count += 1;
            }
            println!("  device {d}: {} blocks in {}", count, root.display());
        }
        println!();
        Ok(())
    }
}

/// Build a small complex-valued signal whose spectrum is nontrivial.
fn build_input_signal(n: usize) -> Vec<Complex<f64>> {
    let mut x = Vec::with_capacity(n);

    for m in 0..n {
        let t = m as f64 / n as f64;
        let re = 1.2 * (2.0 * PI * t).cos()
            + 0.6 * (4.0 * PI * t).sin()
            + 0.25 * (6.0 * PI * t).cos();
        let im = 0.35 * (4.0 * PI * t).sin() - 0.15 * (8.0 * PI * t).cos();
        x.push(Complex::new(re, im));
    }

    x
}

/// Direct DFT for verification.
fn direct_dft(x: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = x.len();
    let mut out = vec![Complex::new(0.0, 0.0); n];

    for k in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for (m, &xm) in x.iter().enumerate() {
            let phase = -2.0 * PI * (m * k) as f64 / n as f64;
            let w = Complex::new(phase.cos(), phase.sin());
            sum += xm * w;
        }
        out[k] = sum;
    }

    out
}

fn max_abs_error(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0_f64, f64::max)
}

fn print_sample(name: &str, data: &[Complex<f64>], count: usize) {
    println!("{name} (length = {})", data.len());
    for (i, z) in data.iter().take(count).enumerate() {
        println!("  [{i:02}] = {:+.6} {:+.6}i", z.re, z.im);
    }
    if data.len() > count {
        println!("  ...");
    }
    println!();
}

/// Execute a blocked out-of-core Cooley–Tukey FFT for N = N1 * N2.
///
/// Uses:
///   n = n1*N2 + n2,  k = k1 + N1*k2
///
/// Algorithm (per Matt's DSP / equation 5):
///   1. N1-DFTs over n1 for each n2 (columns)
///   2. Twiddle W_N^(n2*k1)
///   3. N2-DFTs over n2 for each k1 (rows)
///
/// Final output: X[k1 + N1*k2]
fn out_of_core_fft_cooley_tukey(
    x: &[Complex<f64>],
    n1: usize,
    n2: usize,
    store: &mut BlockStore,
) -> Result<Vec<Complex<f64>>, Box<dyn Error>> {
    let n = n1 * n2;
    if x.len() != n {
        return Err(format!("input length {} does not equal N1*N2 = {}", x.len(), n).into());
    }

    // Stage 0: Partition into N2 columns of length N1 (x[n1*N2+n2], fixed n2)
    for n2_idx in 0..n2 {
        let mut block = vec![Complex::new(0.0, 0.0); n1];
        for n1_idx in 0..n1 {
            block[n1_idx] = x[n1_idx * n2 + n2_idx];
        }
        store.write_complex_block("input", n2_idx, &block)?;
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft_n1 = planner.plan_fft_forward(n1);

    // Stage 1: N1-DFT on each column
    for n2_idx in 0..n2 {
        let mut block = store.read_complex_block("input", n2_idx, n1)?;
        fft_n1.process(&mut block);
        store.write_complex_block("stage1", n2_idx, &block)?;
    }

    // Stage 2: Twiddle W_N^(n2*k1)
    for n2_idx in 0..n2 {
        let mut block = store.read_complex_block("stage1", n2_idx, n1)?;
        for k1_idx in 0..n1 {
            let phase = -2.0 * PI * (n2_idx * k1_idx) as f64 / n as f64;
            let twiddle = Complex::new(phase.cos(), phase.sin());
            block[k1_idx] *= twiddle;
        }
        store.write_complex_block("twiddled", n2_idx, &block)?;
    }

    // Stage 3: Permute into N1 rows of length N2 (for each k1, gather over n2)
    for k1_idx in 0..n1 {
        let mut group = vec![Complex::new(0.0, 0.0); n2];
        for n2_idx in 0..n2 {
            let block = store.read_complex_block("twiddled", n2_idx, n1)?;
            group[n2_idx] = block[k1_idx];
        }
        store.write_complex_block("grouped", k1_idx, &group)?;
    }

    // Stage 4: N2-DFT on each row; output X[k1 + N1*k2]
    let fft_n2 = planner.plan_fft_forward(n2);
    let mut out = vec![Complex::new(0.0, 0.0); n];

    for k1_idx in 0..n1 {
        let mut group = store.read_complex_block("grouped", k1_idx, n2)?;
        fft_n2.process(&mut group);

        for k2_idx in 0..n2 {
            let k = k1_idx + n1 * k2_idx;
            out[k] = group[k2_idx];
        }
    }

    Ok(out)
}

fn main() -> Result<(), Box<dyn Error>> {
    let n1 = 4usize;
    let n2 = 6usize;
    let n = n1 * n2;

    let num_ssds = 3usize;
    let mut store = BlockStore::new("fft_ooc_demo", num_ssds)?;

    let x = build_input_signal(n);
    let y_ooc = out_of_core_fft_cooley_tukey(&x, n1, n2, &mut store)?;

    let y_ref = direct_dft(&x);
    let err = max_abs_error(&y_ooc, &y_ref);

    println!("=== Program 12.8.1: Blocked Out-of-Core FFT with Simulated Multi-SSD Striping ===\n");
    println!("Factorization: N = N1 * N2 = {} * {} = {}", n1, n2, n);
    println!("Number of simulated SSDs: {}", num_ssds);
    println!();

    print_sample("Input signal x", &x, 10);
    print_sample("Out-of-core Cooley–Tukey FFT result", &y_ooc, 10);
    print_sample("Reference direct DFT", &y_ref, 10);

    println!("Maximum error relative to direct DFT: {:.3e}\n", err);

    store.print_layout()?;

    println!("I/O statistics:");
    println!("  block reads   = {}", store.stats.block_reads);
    println!("  block writes  = {}", store.stats.block_writes);
    println!("  bytes read    = {}", store.stats.bytes_read);
    println!("  bytes written = {}", store.stats.bytes_written);
    println!("  total bytes   = {}", store.stats.total_bytes());
    println!();

    println!("Workflow summary:");
    println!("  1. Input partitioned into {} disk blocks (columns) of length {}", n2, n1);
    println!("  2. N1-DFTs of size {} executed on each block", n1);
    println!("  3. Twiddle factors W_N^(n2*k1) applied");
    println!("  4. Data permuted into {} grouped blocks (rows) of length {}", n1, n2);
    println!("  5. N2-DFTs of size {} executed on grouped blocks", n2);
    println!();

    if err > 1e-10 {
        return Err(format!(
            "out-of-core FFT does not match direct DFT closely enough: {:.3e}",
            err
        )
        .into());
    }

    Ok(())
}
```

Program 12.8.1 demonstrates how the classical Cooley–Tukey factorization can be adapted to operate on data that reside outside main memory. Rather than assuming that all intermediate arrays fit in RAM, the program organizes the computation as a sequence of blockwise transforms, twiddle multiplications, and permutations that can be executed while streaming data to and from disk. The comparison with the direct DFT confirms that the blocked implementation produces the correct Fourier coefficients while maintaining numerical accuracy at the level of floating-point roundoff.

The example also highlights the central performance challenge discussed in Section 12.8: when datasets grow beyond the capacity of fast memory, the dominant cost of the FFT shifts from arithmetic operations to data movement. Each stage of the blocked algorithm involves reading and writing disk-resident blocks, and the total execution time is therefore governed largely by I/O throughput rather than by the complexity of the transform itself. This observation motivates the development of modern out-of-core FFT frameworks that employ strategies such as multi-SSD striping, block compression, and asynchronous I/O pipelines.

Although the program is designed for clarity rather than maximum efficiency, its structure mirrors the organization of practical large-scale FFT pipelines. By exposing the intermediate dataflow and recording I/O statistics, it provides insight into how algorithmic decomposition interacts with storage constraints. This framework can be extended to explore more advanced strategies, including overlapping computation with disk transfers, employing larger streaming buffers, or integrating GPU acceleration for the local transforms.

## 12.8.3. Memory-Local FFT Planning and Cache-Aware Execution

When data fit in memory but not in cache, the FFT becomes memory-bandwidth bound. In multidimensional transforms, strided access along non-contiguous axes can generate high miss rates and degrade performance even when arithmetic throughput remains underutilized.

Memory-efficiency-oriented FFT representations attempt to mitigate this effect. Servodio and Li (2024) introduce a fine-grain representation strategy that reorganizes intermediate data structures to reduce TLB and L2 miss rates. Their measurements demonstrate measurable improvements relative to conventional FFT planning approaches, emphasizing that representation and scheduling decisions can materially change memory behavior even when arithmetic structure is unchanged.

Cache-effective permutation remains equally important. Bit-reversal and stride-based permutations are not computationally expensive in arithmetic terms, but they are costly in memory terms. Vectorized and cache-aware bit-reversal strategies have been shown to improve throughput significantly in certain power-of-two Cooley–Tukey implementations (Simek and Šimeček, 2023). Even when Stockham-style auto-sorting eliminates explicit bit reversal, data permutation remains unavoidable at some stage of multidimensional transforms.

In three dimensions, transposition pressure intensifies. OpenFFT explicitly treats 3D FFTs as memory-bounded and proposes tiling methods to improve locality and, in certain configurations, eliminate explicit transposes (Chen et al., 2023). Such approaches illustrate that transpose scheduling is not a minor implementation detail but a central design variable.

### Rust Implementation

Following the discussion in Section 12.8.3 on memory-local FFT planning and cache-aware execution, Program 12.8.2 provides a practical implementation illustrating how multidimensional FFT performance can be influenced by memory access patterns rather than purely by arithmetic complexity. When multidimensional arrays are stored in row-major layout, only one axis is contiguous in memory, while transforms along the remaining axes require strided access. Such strided operations often produce poor cache locality and increased memory-bandwidth pressure. This program demonstrates two alternative execution strategies for computing a three-dimensional FFT. The first follows a straightforward axis-by-axis approach that performs contiguous transforms along one dimension and uses gather–scatter buffers for the remaining axes. The second adopts a cache-aware strategy that introduces tiled transpositions so that each transform stage again operates on contiguous data. By comparing these two approaches and recording memory-access statistics, the program illustrates how data layout and permutation scheduling influence the practical behavior of FFT algorithms in memory-bound environments.

At the core of the implementation is the `Array3C` structure, which represents a three-dimensional array of complex numbers stored in row-major order. The structure stores the dimensions of the array together with a contiguous vector containing the data elements. The indexing methods compute the linear memory offset corresponding to a triple of indices ((i,j,k)), thereby translating logical multidimensional indexing into the underlying one-dimensional storage representation. This design allows the program to manipulate multidimensional arrays while still exposing the memory-layout assumptions relevant to cache-local execution.

The `build_test_array` function constructs a synthetic three-dimensional complex field used as input to the transform. The signal combines several trigonometric components along different spatial directions so that the resulting Fourier spectrum contains distinct frequency modes. This ensures that the transform produces a nontrivial spectrum, making it possible to verify that different execution strategies yield identical numerical results.

The program first implements a conventional axis-by-axis transform through the function `fft3_naive_strided`. This routine follows the straightforward multidimensional FFT strategy in which transforms are applied sequentially along each axis of the array. Because the last axis is contiguous in row-major memory layout, the function `fft_along_last_axis_contiguous` can perform these transforms directly using slices of the underlying storage vector. However, transforms along the other axes require gathering elements into temporary buffers before applying the FFT and scattering the results back into the array. These gather–scatter operations are implemented in the functions `fft_along_axis1_strided` and `fft_along_axis0_strided`. While mathematically straightforward, this approach introduces strided memory access patterns that can degrade cache efficiency.

To illustrate an alternative strategy, the program also implements a cache-aware transform through the function `fft3_cache_aware`. Instead of performing FFTs on non-contiguous memory, this method reorganizes the data between stages using tiled transpositions. The transposition routines `transpose_012_to_021_tiled`, `transpose_012_to_210_tiled`, and `transpose_210_to_012_tiled` reorder the axes of the array in a way that exposes different dimensions as contiguous memory segments. Each transpose operates on small tiles so that the working set remains cache-friendly while elements are exchanged between axes. After each transposition, the contiguous dimension becomes the target of the next transform stage, allowing the FFT method to operate directly on contiguous memory without gather–scatter buffers.

The program maintains detailed statistics about memory behavior through the `MemoryStats` structure. These counters record how many FFT lines are executed contiguously, how many require strided access, how many elements are buffered temporarily, and how many memory movements occur during transposition. Such statistics make the memory-local trade-offs explicit: the naive method minimizes explicit permutations but incurs many strided accesses, while the cache-aware method eliminates strided operations at the cost of additional data movement.

Finally, the `main` function coordinates the demonstration. It constructs the input array, executes both FFT strategies, and measures their execution times. The results are compared using the `max_abs_error` function to confirm that both approaches produce identical Fourier coefficients. The program then prints representative data samples together with the memory-access statistics and timing information, illustrating how the two execution strategies differ in their interaction with the memory hierarchy.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rustfft = "6"
```

```rust
// Program 12.8.2: Memory-Local FFT Planning and Cache-Aware Execution
//
// Corrected version.
//
// This program compares two pedagogical strategies for a 3D FFT on a row-major array:
//
// 1. Naive axis-by-axis execution:
//    - FFT along axis 2 contiguously
//    - FFT along axis 1 using strided gather/scatter
//    - FFT along axis 0 using strided gather/scatter
//
// 2. Cache-aware execution:
//    - FFT along original axis 2 contiguously
//    - transpose so original axis 1 becomes contiguous, FFT it
//    - transpose so original axis 0 becomes contiguous, FFT it
//    - transpose back to original layout
//
// Cargo.toml dependencies:
// [dependencies]
// rustfft = "6"

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::error::Error;
use std::f64::consts::PI;
use std::time::Instant;

#[derive(Debug, Clone)]
struct Array3C {
    n1: usize,
    n2: usize,
    n3: usize,
    data: Vec<Complex<f64>>,
}

impl Array3C {
    fn zeros(n1: usize, n2: usize, n3: usize) -> Self {
        Self {
            n1,
            n2,
            n3,
            data: vec![Complex::new(0.0, 0.0); n1 * n2 * n3],
        }
    }

    fn from_fn<F>(n1: usize, n2: usize, n3: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize, usize) -> Complex<f64>,
    {
        let mut out = Self::zeros(n1, n2, n3);
        for i in 0..n1 {
            for j in 0..n2 {
                for k in 0..n3 {
                    out[(i, j, k)] = f(i, j, k);
                }
            }
        }
        out
    }

    fn shape(&self) -> (usize, usize, usize) {
        (self.n1, self.n2, self.n3)
    }

    fn offset(&self, i: usize, j: usize, k: usize) -> usize {
        k + self.n3 * (j + self.n2 * i)
    }
}

impl std::ops::Index<(usize, usize, usize)> for Array3C {
    type Output = Complex<f64>;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        let (i, j, k) = index;
        &self.data[self.offset(i, j, k)]
    }
}

impl std::ops::IndexMut<(usize, usize, usize)> for Array3C {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        let (i, j, k) = index;
        let off = self.offset(i, j, k);
        &mut self.data[off]
    }
}

#[derive(Debug, Default, Clone)]
struct MemoryStats {
    contiguous_fft_lines: u64,
    strided_fft_lines: u64,
    temporary_buffer_elements: u64,
    transpose_reads: u64,
    transpose_writes: u64,
    transpose_calls: u64,
}

impl MemoryStats {
    fn total_transpose_moves(&self) -> u64 {
        self.transpose_reads + self.transpose_writes
    }
}

fn build_test_array(n1: usize, n2: usize, n3: usize) -> Array3C {
    Array3C::from_fn(n1, n2, n3, |i, j, k| {
        let x = i as f64 / n1 as f64;
        let y = j as f64 / n2 as f64;
        let z = k as f64 / n3 as f64;

        let re = 1.0 * (2.0 * PI * x).cos()
            + 0.6 * (2.0 * PI * y).sin()
            + 0.4 * (4.0 * PI * z).cos()
            + 0.25 * (2.0 * PI * (x + y - z)).sin();

        let im = 0.35 * (2.0 * PI * (x + z)).sin()
            - 0.20 * (2.0 * PI * (2.0 * y - z)).cos();

        Complex::new(re, im)
    })
}

/// FFT along the last axis, which is contiguous in row-major storage.
fn fft_along_last_axis_contiguous(a: &mut Array3C, inverse: bool, stats: &mut MemoryStats) {
    let (n1, n2, n3) = a.shape();
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n3)
    } else {
        planner.plan_fft_forward(n3)
    };

    for i in 0..n1 {
        for j in 0..n2 {
            let start = a.offset(i, j, 0);
            let end = start + n3;
            fft.process(&mut a.data[start..end]);
            stats.contiguous_fft_lines += 1;
        }
    }
}

fn fft_along_axis1_strided(a: &mut Array3C, inverse: bool, stats: &mut MemoryStats) {
    let (n1, n2, n3) = a.shape();
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n2)
    } else {
        planner.plan_fft_forward(n2)
    };

    for i in 0..n1 {
        for k in 0..n3 {
            let mut line = Vec::with_capacity(n2);
            for j in 0..n2 {
                line.push(a[(i, j, k)]);
            }

            fft.process(&mut line);

            for j in 0..n2 {
                a[(i, j, k)] = line[j];
            }

            stats.strided_fft_lines += 1;
            stats.temporary_buffer_elements += n2 as u64;
        }
    }
}

fn fft_along_axis0_strided(a: &mut Array3C, inverse: bool, stats: &mut MemoryStats) {
    let (n1, n2, n3) = a.shape();
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n1)
    } else {
        planner.plan_fft_forward(n1)
    };

    for j in 0..n2 {
        for k in 0..n3 {
            let mut line = Vec::with_capacity(n1);
            for i in 0..n1 {
                line.push(a[(i, j, k)]);
            }

            fft.process(&mut line);

            for i in 0..n1 {
                a[(i, j, k)] = line[i];
            }

            stats.strided_fft_lines += 1;
            stats.temporary_buffer_elements += n1 as u64;
        }
    }
}

fn fft3_naive_strided(input: &Array3C) -> (Array3C, MemoryStats) {
    let mut out = input.clone();
    let mut stats = MemoryStats::default();

    fft_along_last_axis_contiguous(&mut out, false, &mut stats);
    fft_along_axis1_strided(&mut out, false, &mut stats);
    fft_along_axis0_strided(&mut out, false, &mut stats);

    (out, stats)
}

/// Transpose (a,b,c) -> (a,c,b)
fn transpose_012_to_021_tiled(input: &Array3C, tile_j: usize, tile_k: usize, stats: &mut MemoryStats) -> Array3C {
    let (n1, n2, n3) = input.shape();
    let mut out = Array3C::zeros(n1, n3, n2);

    for i in 0..n1 {
        for j0 in (0..n2).step_by(tile_j) {
            for k0 in (0..n3).step_by(tile_k) {
                let j_max = (j0 + tile_j).min(n2);
                let k_max = (k0 + tile_k).min(n3);

                for j in j0..j_max {
                    for k in k0..k_max {
                        out[(i, k, j)] = input[(i, j, k)];
                        stats.transpose_reads += 1;
                        stats.transpose_writes += 1;
                    }
                }
            }
        }
    }

    stats.transpose_calls += 1;
    out
}

/// Transpose (a,b,c) -> (c,b,a)
fn transpose_012_to_210_tiled(input: &Array3C, tile_i: usize, tile_k: usize, stats: &mut MemoryStats) -> Array3C {
    let (n1, n2, n3) = input.shape();
    let mut out = Array3C::zeros(n3, n2, n1);

    for i0 in (0..n1).step_by(tile_i) {
        for k0 in (0..n3).step_by(tile_k) {
            let i_max = (i0 + tile_i).min(n1);
            let k_max = (k0 + tile_k).min(n3);

            for i in i0..i_max {
                for j in 0..n2 {
                    for k in k0..k_max {
                        out[(k, j, i)] = input[(i, j, k)];
                        stats.transpose_reads += 1;
                        stats.transpose_writes += 1;
                    }
                }
            }
        }
    }

    stats.transpose_calls += 1;
    out
}

/// Inverse transpose (c,b,a) -> (a,b,c)
fn transpose_210_to_012_tiled(input: &Array3C, tile_i: usize, tile_k: usize, stats: &mut MemoryStats) -> Array3C {
    let (n3, n2, n1) = input.shape();
    let mut out = Array3C::zeros(n1, n2, n3);

    for k0 in (0..n3).step_by(tile_k) {
        for i0 in (0..n1).step_by(tile_i) {
            let k_max = (k0 + tile_k).min(n3);
            let i_max = (i0 + tile_i).min(n1);

            for k in k0..k_max {
                for j in 0..n2 {
                    for i in i0..i_max {
                        out[(i, j, k)] = input[(k, j, i)];
                        stats.transpose_reads += 1;
                        stats.transpose_writes += 1;
                    }
                }
            }
        }
    }

    stats.transpose_calls += 1;
    out
}

/// Correct cache-aware FFT:
/// - original axis 2 directly
/// - transpose so original axis 1 becomes last
/// - transpose so original axis 0 becomes last
/// - transpose back
fn fft3_cache_aware(input: &Array3C, tile_i: usize, tile_j: usize) -> (Array3C, MemoryStats) {
    let mut stats = MemoryStats::default();

    // Original layout: (0,1,2)
    let mut a = input.clone();

    // FFT along original axis 2
    fft_along_last_axis_contiguous(&mut a, false, &mut stats);

    // Move original axis 1 to last: (0,1,2) -> (0,2,1)
    let mut b = transpose_012_to_021_tiled(&a, tile_j, tile_i, &mut stats);
    // Last axis is now original axis 1
    fft_along_last_axis_contiguous(&mut b, false, &mut stats);

    // Move original axis 0 to last:
    // current layout is (0,2,1), after (a,b,c)->(c,b,a) it becomes (1,2,0)
    let mut c = transpose_012_to_210_tiled(&b, tile_i, tile_i, &mut stats);
    // Last axis is now original axis 0
    fft_along_last_axis_contiguous(&mut c, false, &mut stats);

    // Return from (1,2,0) to (0,2,1)
    let d = transpose_210_to_012_tiled(&c, tile_i, tile_i, &mut stats);
    // Return from (0,2,1) to (0,1,2)
    let out = transpose_012_to_021_tiled(&d, tile_i, tile_j, &mut stats);

    (out, stats)
}

fn max_abs_error(a: &Array3C, b: &Array3C) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0_f64, f64::max)
}

fn print_sample(name: &str, a: &Array3C, count: usize) {
    let (n1, n2, n3) = a.shape();
    println!("{name} (shape = [{n1}, {n2}, {n3}])");

    let mut shown = 0usize;
    'outer: for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..n3 {
                let z = a[(i, j, k)];
                println!(
                    "  [{:02},{:02},{:02}] = {:+.6} {:+.6}i",
                    i, j, k, z.re, z.im
                );
                shown += 1;
                if shown >= count {
                    break 'outer;
                }
            }
        }
    }
    println!();
}

fn main() -> Result<(), Box<dyn Error>> {
    let n1 = 32usize;
    let n2 = 24usize;
    let n3 = 20usize;

    let tile_i = 8usize;
    let tile_j = 8usize;

    let x = build_test_array(n1, n2, n3);

    let t0 = Instant::now();
    let (y_naive, stats_naive) = fft3_naive_strided(&x);
    let naive_time = t0.elapsed();

    let t1 = Instant::now();
    let (y_cache, stats_cache) = fft3_cache_aware(&x, tile_i, tile_j);
    let cache_time = t1.elapsed();

    let err = max_abs_error(&y_naive, &y_cache);

    println!("=== Program 12.8.2: Memory-Local FFT Planning and Cache-Aware Execution ===\n");
    println!("Array shape: [{}, {}, {}]", n1, n2, n3);
    println!("Tile size for cache-aware transposes: {} x {}", tile_i, tile_j);
    println!();

    print_sample("Input data x", &x, 10);
    print_sample("Naive strided 3D FFT", &y_naive, 10);
    print_sample("Cache-aware transposed 3D FFT", &y_cache, 10);

    println!("Maximum difference between the two FFT strategies: {:.3e}\n", err);

    println!("Naive strided strategy:");
    println!("  contiguous FFT lines = {}", stats_naive.contiguous_fft_lines);
    println!("  strided FFT lines    = {}", stats_naive.strided_fft_lines);
    println!(
        "  temporary buffered elements = {}",
        stats_naive.temporary_buffer_elements
    );
    println!("  transpose calls      = {}", stats_naive.transpose_calls);
    println!("  elapsed time         = {:?}\n", naive_time);

    println!("Cache-aware strategy:");
    println!("  contiguous FFT lines = {}", stats_cache.contiguous_fft_lines);
    println!("  strided FFT lines    = {}", stats_cache.strided_fft_lines);
    println!(
        "  temporary buffered elements = {}",
        stats_cache.temporary_buffer_elements
    );
    println!("  transpose calls      = {}", stats_cache.transpose_calls);
    println!("  transpose reads      = {}", stats_cache.transpose_reads);
    println!("  transpose writes     = {}", stats_cache.transpose_writes);
    println!(
        "  total transpose moves = {}",
        stats_cache.total_transpose_moves()
    );
    println!("  elapsed time         = {:?}\n", cache_time);

    println!("Interpretation:");
    println!("  The naive plan avoids explicit transposes but pays for strided gather/scatter");
    println!("  when transforming along non-contiguous axes.");
    println!("  The cache-aware plan performs extra data movement through tiled transposes");
    println!("  so that each FFT stage again operates on contiguous memory.");
    println!("  In large multidimensional FFTs, choosing between these strategies is a");
    println!("  memory-local planning decision rather than an arithmetic one.");

    if err > 1e-10 {
        return Err(format!(
            "cache-aware FFT does not match naive FFT closely enough: {:.3e}",
            err
        )
        .into());
    }

    Ok(())
}
```

Program 12.8.2 demonstrates that the performance characteristics of multidimensional FFTs depend strongly on memory-access patterns rather than solely on arithmetic complexity. The naive strategy performs transforms directly along each axis but must rely on gather–scatter buffers when the data are not contiguous in memory. Although this approach avoids explicit permutations, it introduces many strided memory accesses that may lead to inefficient cache utilization.

The cache-aware strategy instead reorganizes the data through tiled transpositions so that each transform stage operates on contiguous memory. While this method introduces additional data movement, it eliminates strided accesses and can therefore improve locality when working with large datasets. The comparison between the two approaches highlights a central theme of Section 12.8.3: in modern high-performance FFT implementations, algorithmic efficiency is determined not only by the mathematical structure of the transform but also by how effectively the computation interacts with the memory hierarchy.

Although the example presented here is intentionally simplified for clarity, the same principles underlie production-level FFT frameworks. Practical implementations often employ sophisticated planning stages that select between strided execution, blocked permutations, and cache-aware transpositions depending on the array dimensions and the characteristics of the target architecture. By making the dataflow explicit and quantifying the memory operations involved, this program provides insight into how such planning decisions influence real-world FFT performance.

## 12.8.4. Distributed Memory, Communication Minimization, and Performance Portability

In distributed memory environments, multidimensional FFTs interleave local transforms with data redistributions. If the global domain is partitioned along one axis, transforming along another axis generally requires data redistribution. These redistributions are frequently implemented as all-to-all collectives.

Communication-minimizing FFT algorithms attempt to reduce the number of such collective phases. Koopman and Bisseling (2023) present a multidimensional cyclic-to-cyclic FFT that can operate with a single all-to-all step under a processor-count constraint, demonstrating strong scaling to thousands of cores. Collective-optimized FFT frameworks further improve redistribution efficiency by tuning MPI_Alltoallv behavior in FFT-heavy workflows (Namugwanya et al., 2023). Alternative approaches refine MPI derived-datatype strategies to better support multidimensional subarray redistributions (Yan et al., 2024).

The guiding principle is that communication volume and synchronization frequency are algorithmic quantities, not merely implementation artifacts. Reordering axis transforms or altering data distribution schemes can change the number and cost of collective phases without altering arithmetic complexity.

Performance portability adds another dimension to this problem. Frameworks built atop abstraction layers such as Kokkos seek to provide distributed FFT interfaces that map efficiently onto diverse CPU and GPU architectures while preserving correctness and scalability (Asahi et al., 2025). In such systems, planning decisions must balance locality, communication cost, and architectural heterogeneity.

### Rust Implementation

Following the discussion in Section 12.8.4 on distributed memory execution, communication minimization, and performance portability, Program 12.8.3 provides a concrete illustration of how multidimensional FFT computations can be orchestrated across a distributed system. In large-scale environments, multidimensional FFTs are typically decomposed across processors so that each processor owns a portion of the global array. While local FFTs can be performed independently, transforms along non-local axes generally require data redistribution through collective communication operations such as all-to-all exchanges. This program simulates such a distributed environment using multiple virtual ranks and compares two execution strategies. The first follows a baseline slab decomposition in which local transforms are performed on available axes before a redistribution enables the final transform. The second illustrates a communication-aware alternative in which the redistribution layout is chosen so that the remaining transforms can be performed locally. By explicitly modeling the redistribution phases and communication statistics, the program demonstrates how algorithmic scheduling of transforms and data layouts influences both correctness and communication cost in distributed FFT workflows.

At the core of the implementation is the `Array3C` structure, which represents a three-dimensional complex-valued array stored in row-major order. The structure maintains the dimensions of the grid together with a contiguous vector of complex numbers. The indexing operations convert logical coordinates $(i,j,k)$ into the corresponding linear offset in memory. This representation mirrors the storage model used in many numerical libraries and allows the program to simulate distributed decompositions while maintaining a clear correspondence with the mathematical tensor representation of multidimensional FFTs.

The program constructs a synthetic three-dimensional signal using the `build_test_array` function. The signal combines several trigonometric components along each coordinate direction so that the resulting Fourier spectrum contains distinct frequency modes. This design ensures that the transform produces meaningful spectral coefficients rather than trivial values, which allows the distributed algorithms to be validated against a reference transform.

To establish correctness, the function `fft3_reference` computes a full three-dimensional FFT on a single process. The routine applies successive one-dimensional transforms along the three axes using the helper functions `fft_axis2_global`, `fft_axis1_global`, and `fft_axis0_global`. Each of these routines extracts the appropriate one-dimensional data lines from the array, applies a RustFFT transform, and writes the results back to the array. This sequential reference computation provides the benchmark result against which the distributed strategies are compared.

The distributed simulations begin by partitioning the global array using the function `distribute_slab_x`. In this step the array is decomposed into x-slabs, each assigned to a virtual rank. Each slab therefore contains a contiguous subset of the first dimension together with the full extent of the remaining dimensions. Because axes two and one are fully local within each slab, the functions `local_fft_axis2_on_slabs` and `local_fft_axis1_on_xslabs` can apply the corresponding transforms without requiring any communication. This mirrors the typical first stage of distributed FFT implementations, where transforms along locally contiguous dimensions are performed before any redistribution.

The baseline distributed algorithm then performs a redistribution from x-slabs to y-slabs using the function `redistribute_xslab_to_yslab`. This operation simulates an all-to-all exchange in which data are reorganized so that each rank holds a contiguous block along the second dimension. The communication counters stored in the `CommStats` structure record the number of simulated collective operations, message counts, and the total volume of data moved during this step. Once this redistribution is complete, the remaining transform along axis zero becomes local to each rank and is performed by the function `local_fft_axis0_on_yslabs`. The final global result is reconstructed using `gather_from_yslabs`.

The communication-minimized strategy follows a different scheduling pattern. After the initial local transforms along axis two, the function `redistribute_xslab_to_zslab` performs a redistribution into z-slabs, each containing the full extent of the first two dimensions but only a subset of the third. In this layout the remaining transforms along axes one and zero are both locally available. The function `local_fft_axis1_axis0_on_zslabs` therefore applies both transforms sequentially without further communication. The final array is then reconstructed through `gather_from_zslabs`.

The `main` function orchestrates the entire experiment. It constructs the test data, computes the reference transform, and then executes both distributed strategies. The function measures execution time and evaluates numerical accuracy using the `max_abs_error` function. Finally, it prints representative data samples together with communication statistics, allowing the reader to compare the algorithmic structure and communication behavior of the two plans.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rustfft = "6"
```

```rust
// Program 12.8.3: Distributed-Memory FFT Orchestration, Communication Minimization,
// and Performance-Portable Planning
//
// This program illustrates the ideas discussed in Section 12.8.4. It does not
// use MPI directly. Instead, it simulates a distributed-memory environment with
// multiple virtual ranks, each owning a slab of a global 3D array.
//
// The program compares two distributed FFT execution plans:
//
// 1. Baseline slab plan:
//    - x-slabs: local FFT along axis 2 and axis 1 (both local)
//    - one redistribution to y-slabs [N1, local_N2, N3]
//    - local FFT along axis 0
//
// 2. Communication-minimized plan:
//    - x-slabs: local FFT along axis 2 only
//    - one redistribution to z-slabs [N1, N2, local_N3]
//    - local FFT along axis 1 and axis 0 (both full)
//
// The purpose is pedagogical. The code makes explicit that communication
// schedule, redistribution volume, and synchronization count are algorithmic
// design variables in distributed FFT execution.
//
// Cargo.toml dependencies:
// [dependencies]
// rustfft = "6"

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::error::Error;
use std::f64::consts::PI;
use std::time::Instant;

#[derive(Debug, Clone)]
struct Array3C {
    n1: usize,
    n2: usize,
    n3: usize,
    data: Vec<Complex<f64>>,
}

impl Array3C {
    fn zeros(n1: usize, n2: usize, n3: usize) -> Self {
        Self {
            n1,
            n2,
            n3,
            data: vec![Complex::new(0.0, 0.0); n1 * n2 * n3],
        }
    }

    fn from_fn<F>(n1: usize, n2: usize, n3: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize, usize) -> Complex<f64>,
    {
        let mut out = Self::zeros(n1, n2, n3);
        for i in 0..n1 {
            for j in 0..n2 {
                for k in 0..n3 {
                    out[(i, j, k)] = f(i, j, k);
                }
            }
        }
        out
    }

    fn shape(&self) -> (usize, usize, usize) {
        (self.n1, self.n2, self.n3)
    }

    fn offset(&self, i: usize, j: usize, k: usize) -> usize {
        k + self.n3 * (j + self.n2 * i)
    }
}

impl std::ops::Index<(usize, usize, usize)> for Array3C {
    type Output = Complex<f64>;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        let (i, j, k) = index;
        &self.data[self.offset(i, j, k)]
    }
}

impl std::ops::IndexMut<(usize, usize, usize)> for Array3C {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        let (i, j, k) = index;
        let off = self.offset(i, j, k);
        &mut self.data[off]
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SlabRank {
    rank: usize,
    i_start: usize,
    local: Array3C, // x-slab: [local_n1, N2, N3]; y-slab: [N1, local_n2, N3]
}

#[derive(Debug, Clone)]
struct ZSlabRank {
    #[allow(dead_code)]
    rank: usize,
    k_start: usize,
    local: Array3C, // shape = [N1, N2, local_n3]
}

#[derive(Debug, Default, Clone)]
struct CommStats {
    all_to_all_calls: u64,
    messages: u64,
    elements_moved: u64,
    barriers: u64,
}

impl CommStats {
    fn bytes_moved(&self) -> u64 {
        self.elements_moved * 16
    }
}

#[derive(Debug, Clone)]
struct ExecutionReport<T> {
    data: T,
    comm: CommStats,
}

fn build_test_array(n1: usize, n2: usize, n3: usize) -> Array3C {
    Array3C::from_fn(n1, n2, n3, |i, j, k| {
        let x = i as f64 / n1 as f64;
        let y = j as f64 / n2 as f64;
        let z = k as f64 / n3 as f64;

        let re = 1.0 * (2.0 * PI * x).cos()
            + 0.7 * (2.0 * PI * y).sin()
            + 0.4 * (4.0 * PI * z).cos()
            + 0.2 * (2.0 * PI * (x + y + z)).sin();

        let im = 0.3 * (2.0 * PI * (x - z)).sin()
            - 0.15 * (2.0 * PI * (2.0 * y - z)).cos();

        Complex::new(re, im)
    })
}

fn fft_axis2_global(a: &mut Array3C, inverse: bool) {
    let (n1, n2, n3) = a.shape();
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n3)
    } else {
        planner.plan_fft_forward(n3)
    };

    for i in 0..n1 {
        for j in 0..n2 {
            let start = a.offset(i, j, 0);
            let end = start + n3;
            fft.process(&mut a.data[start..end]);
        }
    }
}

fn fft_axis1_global(a: &mut Array3C, inverse: bool) {
    let (n1, n2, n3) = a.shape();
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n2)
    } else {
        planner.plan_fft_forward(n2)
    };

    for i in 0..n1 {
        for k in 0..n3 {
            let mut line = Vec::with_capacity(n2);
            for j in 0..n2 {
                line.push(a[(i, j, k)]);
            }
            fft.process(&mut line);
            for j in 0..n2 {
                a[(i, j, k)] = line[j];
            }
        }
    }
}

fn fft_axis0_global(a: &mut Array3C, inverse: bool) {
    let (n1, n2, n3) = a.shape();
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n1)
    } else {
        planner.plan_fft_forward(n1)
    };

    for j in 0..n2 {
        for k in 0..n3 {
            let mut line = Vec::with_capacity(n1);
            for i in 0..n1 {
                line.push(a[(i, j, k)]);
            }
            fft.process(&mut line);
            for i in 0..n1 {
                a[(i, j, k)] = line[i];
            }
        }
    }
}

fn fft3_reference(input: &Array3C) -> Array3C {
    let mut out = input.clone();
    fft_axis2_global(&mut out, false);
    fft_axis1_global(&mut out, false);
    fft_axis0_global(&mut out, false);
    out
}

fn distribute_slab_x(global: &Array3C, nranks: usize) -> Result<Vec<SlabRank>, Box<dyn Error>> {
    let (n1, n2, n3) = global.shape();
    if n1 % nranks != 0 {
        return Err("N1 must be divisible by nranks for slab decomposition".into());
    }
    let local_n1 = n1 / nranks;
    let mut ranks = Vec::with_capacity(nranks);

    for r in 0..nranks {
        let i_start = r * local_n1;
        let mut local = Array3C::zeros(local_n1, n2, n3);
        for i in 0..local_n1 {
            for j in 0..n2 {
                for k in 0..n3 {
                    local[(i, j, k)] = global[(i_start + i, j, k)];
                }
            }
        }
        ranks.push(SlabRank {
            rank: r,
            i_start,
            local,
        });
    }

    Ok(ranks)
}

fn gather_slab_x(ranks: &[SlabRank], n1: usize, n2: usize, n3: usize) -> Array3C {
    let mut global = Array3C::zeros(n1, n2, n3);
    for rank in ranks {
        let local_n1 = rank.local.n1;
        for i in 0..local_n1 {
            for j in 0..n2 {
                for k in 0..n3 {
                    global[(rank.i_start + i, j, k)] = rank.local[(i, j, k)];
                }
            }
        }
    }
    global
}

fn local_fft_axis2_on_slabs(ranks: &mut [SlabRank]) {
    for rank in ranks {
        fft_axis2_global(&mut rank.local, false);
    }
}

fn local_fft_axis1_on_xslabs(ranks: &mut [SlabRank]) {
    for rank in ranks {
        fft_axis1_global(&mut rank.local, false);
    }
}

/// Simulate a redistribution from x-slabs to y-slabs.
/// Each destination rank receives a block contiguous in j.
fn redistribute_xslab_to_yslab(
    xslabs: &[SlabRank],
    n1: usize,
    n2: usize,
    n3: usize,
    comm: &mut CommStats,
) -> Result<Vec<SlabRank>, Box<dyn Error>> {
    let nranks = xslabs.len();
    if n2 % nranks != 0 {
        return Err("N2 must be divisible by nranks for y-slab redistribution".into());
    }

    let global = gather_slab_x(xslabs, n1, n2, n3);
    let local_n2 = n2 / nranks;
    let local_n1 = n1 / nranks;

    let mut out = Vec::with_capacity(nranks);
    for r in 0..nranks {
        let j_start = r * local_n2;
        let mut local = Array3C::zeros(n1, local_n2, n3);
        for i in 0..n1 {
            for j in 0..local_n2 {
                for k in 0..n3 {
                    local[(i, j, k)] = global[(i, j_start + j, k)];
                }
            }
        }
        // Reuse SlabRank structurally even though the "start" now refers to j_start logically.
        // We keep i_start = 0 because this rank owns all i after redistribution.
        out.push(SlabRank {
            rank: r,
            i_start: j_start,
            local,
        });
    }

    comm.all_to_all_calls += 1;
    comm.barriers += 1;
    comm.messages += (nranks * nranks) as u64;
    comm.elements_moved += (n1 * n2 * n3) as u64;

    // local_n1 is intentionally unused here as this is a full-y-slab representation.
    let _ = local_n1;
    Ok(out)
}

/// Local FFT along axis 0 when each rank owns full i and a slab in j.
/// y-slabs have shape [N1, local_N2, N3], so axis 0 is fully local.
fn local_fft_axis0_on_yslabs(yslabs: &mut [SlabRank]) {
    for rank in yslabs {
        fft_axis0_global(&mut rank.local, false);
    }
}

fn gather_from_yslabs(yslabs: &[SlabRank], n1: usize, n2: usize, n3: usize) -> Array3C {
    let mut global = Array3C::zeros(n1, n2, n3);
    let local_n2 = n2 / yslabs.len();
    for rank in yslabs {
        let j_start = rank.i_start;
        for i in 0..n1 {
            for j in 0..local_n2 {
                for k in 0..n3 {
                    global[(i, j_start + j, k)] = rank.local[(i, j, k)];
                }
            }
        }
    }
    global
}

/// Communication-minimized: one redistribution from x-slabs to z-slabs.
/// z-slabs have shape [N1, N2, local_n3], so axis 0 and 1 are fully local.
fn redistribute_xslab_to_zslab(
    xslabs: &[SlabRank],
    n1: usize,
    n2: usize,
    n3: usize,
    comm: &mut CommStats,
) -> Result<Vec<ZSlabRank>, Box<dyn Error>> {
    let nranks = xslabs.len();
    if n3 % nranks != 0 {
        return Err("N3 must be divisible by nranks for z-slab redistribution".into());
    }

    let global = gather_slab_x(xslabs, n1, n2, n3);
    let local_n3 = n3 / nranks;

    let mut out = Vec::with_capacity(nranks);
    for r in 0..nranks {
        let k_start = r * local_n3;
        let mut local = Array3C::zeros(n1, n2, local_n3);
        for i in 0..n1 {
            for j in 0..n2 {
                for k in 0..local_n3 {
                    local[(i, j, k)] = global[(i, j, k_start + k)];
                }
            }
        }

        out.push(ZSlabRank {
            rank: r,
            k_start,
            local,
        });
    }

    comm.all_to_all_calls += 1;
    comm.barriers += 1;
    comm.messages += (nranks * nranks) as u64;
    comm.elements_moved += (n1 * n2 * n3) as u64;

    Ok(out)
}

fn local_fft_axis1_axis0_on_zslabs(zslabs: &mut [ZSlabRank]) {
    for rank in zslabs {
        fft_axis1_global(&mut rank.local, false);
        fft_axis0_global(&mut rank.local, false);
    }
}

fn gather_from_zslabs(zslabs: &[ZSlabRank], n1: usize, n2: usize, n3: usize) -> Array3C {
    let mut global = Array3C::zeros(n1, n2, n3);
    let local_n3 = n3 / zslabs.len();
    for rank in zslabs {
        let k_start = rank.k_start;
        for i in 0..n1 {
            for j in 0..n2 {
                for k in 0..local_n3 {
                    global[(i, j, k_start + k)] = rank.local[(i, j, k)];
                }
            }
        }
    }
    global
}

fn baseline_distributed_fft(global: &Array3C, nranks: usize) -> Result<ExecutionReport<Array3C>, Box<dyn Error>> {
    let (n1, n2, n3) = global.shape();
    let mut comm = CommStats::default();

    let mut xslabs = distribute_slab_x(global, nranks)?;
    local_fft_axis2_on_slabs(&mut xslabs);
    local_fft_axis1_on_xslabs(&mut xslabs);

    let mut yslabs = redistribute_xslab_to_yslab(&xslabs, n1, n2, n3, &mut comm)?;
    local_fft_axis0_on_yslabs(&mut yslabs);

    let final_global = gather_from_yslabs(&yslabs, n1, n2, n3);

    Ok(ExecutionReport {
        data: final_global,
        comm,
    })
}

fn minimized_distributed_fft(global: &Array3C, nranks: usize) -> Result<ExecutionReport<Array3C>, Box<dyn Error>> {
    let (n1, n2, n3) = global.shape();
    let mut comm = CommStats::default();

    let mut xslabs = distribute_slab_x(global, nranks)?;
    local_fft_axis2_on_slabs(&mut xslabs);

    let mut zslabs = redistribute_xslab_to_zslab(&xslabs, n1, n2, n3, &mut comm)?;
    local_fft_axis1_axis0_on_zslabs(&mut zslabs);

    let final_global = gather_from_zslabs(&zslabs, n1, n2, n3);

    Ok(ExecutionReport {
        data: final_global,
        comm,
    })
}

fn max_abs_error(a: &Array3C, b: &Array3C) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0_f64, f64::max)
}

fn print_sample(name: &str, a: &Array3C, count: usize) {
    let (n1, n2, n3) = a.shape();
    println!("{name} (shape = [{n1}, {n2}, {n3}])");

    let mut shown = 0usize;
    'outer: for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..n3 {
                let z = a[(i, j, k)];
                println!(
                    "  [{:02},{:02},{:02}] = {:+.6} {:+.6}i",
                    i, j, k, z.re, z.im
                );
                shown += 1;
                if shown >= count {
                    break 'outer;
                }
            }
        }
    }
    println!();
}

fn main() -> Result<(), Box<dyn Error>> {
    let n1 = 8usize;
    let n2 = 8usize;
    let n3 = 8usize;
    let nranks = 4usize;

    let x = build_test_array(n1, n2, n3);

    let t_ref = Instant::now();
    let y_ref = fft3_reference(&x);
    let ref_time = t_ref.elapsed();

    let t_base = Instant::now();
    let baseline = baseline_distributed_fft(&x, nranks)?;
    let baseline_time = t_base.elapsed();

    let t_min = Instant::now();
    let minimized = minimized_distributed_fft(&x, nranks)?;
    let minimized_time = t_min.elapsed();

    let err_baseline = max_abs_error(&baseline.data, &y_ref);
    let err_minimized = max_abs_error(&minimized.data, &y_ref);

    println!("=== Program 12.8.3: Distributed FFT Orchestration and Communication Minimization ===\n");
    println!("Global array shape: [{}, {}, {}]", n1, n2, n3);
    println!("Number of virtual ranks: {}", nranks);
    println!();

    print_sample("Input data x", &x, 8);
    print_sample("Reference global 3D FFT", &y_ref, 8);
    print_sample("Baseline distributed FFT", &baseline.data, 8);
    print_sample("Communication-minimized distributed FFT", &minimized.data, 8);

    println!("Accuracy diagnostics:");
    println!("  baseline vs reference error              = {:.3e}", err_baseline);
    println!("  minimized vs reference error             = {:.3e}", err_minimized);
    println!();

    println!("Baseline slab plan communication:");
    println!("  all-to-all calls                         = {}", baseline.comm.all_to_all_calls);
    println!("  logical messages                         = {}", baseline.comm.messages);
    println!("  elements moved                           = {}", baseline.comm.elements_moved);
    println!("  bytes moved                              = {}", baseline.comm.bytes_moved());
    println!("  synchronization points                   = {}", baseline.comm.barriers);
    println!("  elapsed time                             = {:?}", baseline_time);
    println!();

    println!("Communication-minimized plan:");
    println!("  all-to-all calls                         = {}", minimized.comm.all_to_all_calls);
    println!("  logical messages                         = {}", minimized.comm.messages);
    println!("  elements moved                           = {}", minimized.comm.elements_moved);
    println!("  bytes moved                              = {}", minimized.comm.bytes_moved());
    println!("  synchronization points                   = {}", minimized.comm.barriers);
    println!("  elapsed time                             = {:?}", minimized_time);
    println!();

    println!("Reference single-process execution time    = {:?}", ref_time);
    println!();

    println!("Interpretation:");
    println!("  The baseline plan does FFTs on axis 2 and 1 on x-slabs (no comm), then");
    println!("  one redistribution to y-slabs, then FFT on axis 0.");
    println!("  The communication-minimized plan reduces the number of global");
    println!("  redistribution phases to one by changing the intermediate data layout.");
    println!("  The arithmetic transform is unchanged, but the communication schedule");
    println!("  and synchronization count differ.");
    println!("  This illustrates the central point of Section 12.8.4: communication");
    println!("  volume and collective frequency are algorithmic quantities, and planning");
    println!("  decisions must balance locality, redistribution cost, and portability.");

    if err_baseline > 1e-10 {
        return Err(format!(
            "baseline distributed FFT does not match the reference closely enough: {:.3e}",
            err_baseline
        )
        .into());
    }

    if err_minimized > 1e-10 {
        return Err(format!(
            "communication-minimized distributed FFT does not match the reference closely enough: {:.3e}",
            err_minimized
        )
        .into());
    }

    Ok(())
}
```

Program 12.8.3 demonstrates how multidimensional FFT computations can be organized in distributed-memory environments where communication cost is a dominant factor. The baseline slab decomposition illustrates the traditional strategy of performing local transforms first and then redistributing data to enable the remaining transforms. Although straightforward, this approach requires collective communication once the computation reaches a dimension that is not locally contiguous.

The communication-aware alternative reorganizes the redistribution stage so that the remaining transforms can be executed locally after a single data exchange. While the arithmetic structure of the FFT remains unchanged, the scheduling of redistributions alters how communication is performed and where synchronization points occur. This distinction reflects the central theme of Section 12.8.4: in distributed numerical algorithms, communication volume and synchronization frequency are algorithmic quantities that must be considered alongside arithmetic complexity.

The modular structure of the program also illustrates how distributed FFT workflows can be constructed from reusable building blocks. By separating local transforms, redistribution routines, and communication accounting, the program provides a clear framework for experimenting with alternative decompositions and communication schedules. Such abstractions form the basis of modern performance-portable FFT frameworks, where planning stages dynamically select layouts and communication strategies that best match the characteristics of the target architecture.

# 12.9. Conclusion

This chapter has developed the theory and algorithms of the Fast Fourier Transform from the operator-theoretic foundations of Fourier analysis through the discrete transform, its efficient computation, and its deployment in multidimensional, real-valued, and memory-constrained settings. The central organizing idea is spectral diagonalization: complex exponentials are eigenfunctions of translation-invariant operators, and the discrete Fourier transform is the change of basis that exposes this diagonal structure in finite-dimensional linear algebra. The FFT does not define a new transform but provides a factorization of the dense DFT matrix into sparse structured components, reducing the arithmetic cost from $O(N^2)$ to $O(N \log N)$. This complexity reduction transforms Fourier diagonalization from a theoretical device into a practical computational primitive in simulation, signal processing, structured linear algebra, and scientific imaging. The Rust implementations throughout the chapter demonstrate how bit-reversal scheduling, in-place butterfly operations, symmetry exploitation, and memory-aware data layouts translate these mathematical principles into efficient, verifiable numerical code.

## 12.9.1. Key Takeaways

- The Fourier transform is fundamentally a change of representation that converts translation-invariant operators into diagonal form. On periodic grids, discrete Fourier modes are eigenvectors of any circulant matrix $C(h)$, and the diagonalization identity $C(h) = F_N^{-1} \mathrm{diag}(F_N h) F_N$ reduces circular convolution to pointwise multiplication in frequency space. This structural principle underlies FFT-based PDE solvers, where the discrete Laplacian on a periodic domain is circulant and its spectral solve reduces to independent scalar divisions for each frequency mode.
- Uniform sampling with step $\Delta$ produces spectral replication with period $f_s = 1/\Delta$, and frequencies outside the Nyquist interval $[-f_s/2, f_s/2)$ fold back as aliases. Finite observation records impose an implicit rectangular window that causes spectral leakage, spreading energy across neighboring frequency bins. Tapered windows such as the Hann window improve spectral concentration by suppressing leakage. Correct mapping between DFT bin indices and physical frequencies requires the periodic indexing rule $f_k = (k - N)/N\Delta$ for $k > N/2$, and the discrete Parseval identity $\sum |x[n]|^2 = \frac{1}{N} \sum |X[k]|^2$ legitimizes interpreting $|X[k]|^2$ as a discretized power spectrum.
- The radix-2 Cooley-Tukey factorization decomposes an $N$-point DFT into two $N/2$-point DFTs combined by butterfly operations $X[k] = E[k] + \omega_N^k O[k]$ and $X[k + N/2] = E[k] - \omega_N^k O[k]$. The recurrence $T(N) = 2T(N/2) + cN$ solves to $O(N \log_2 N)$, and the iterative implementation proceeds by permuting input data into bit-reversed order and applying $\log_2 N$ successive butterfly stages with doubling subtransform size. On modern architectures, FFT performance is often limited by memory bandwidth and communication patterns rather than by floating-point throughput, particularly in GPU and processing-in-memory environments.
- When the input vector is real-valued, the DFT spectrum exhibits Hermitian symmetry $‾X[N-k] = \overline{X[k]}$, so only $N/2 + 1$ complex coefficients are independent. The even-odd packing construction forms $h_m = e_m + i \, o_m$ from the even and odd subsequences, computes one half-size complex FFT, and recovers the full spectrum through algebraic postprocessing. This reduces both arithmetic cost and memory traffic by approximately a factor of two. The "two real transforms for the price of one complex FFT" identity further exploits this redundancy by packing two real signals into a single complex sequence and separating their spectra after transformation.
- Sine and cosine transforms arise naturally from boundary symmetry in PDE discretizations. Dirichlet boundary conditions correspond to odd extensions and sine eigenvectors of the discrete Laplacian, while Neumann boundary conditions correspond to even extensions and cosine eigenvectors. The canonical discrete sine transform is computed via FFT of an odd extension of length $2N$, with sine coefficients extracted from the imaginary part of the spectrum. The half-shifted discrete cosine transform uses an even extension and a phase correction $C_k = \mathrm{Re}(e^{-i\pi k/(2N)} G_k)$. Both transforms inherit $O(N \log N)$ complexity from the underlying FFT. Multiple DCT/DST families exist, distinguished by endpoint inclusion, half-grid shifts, and normalization conventions, and the transform type must match the discrete operator being diagonalized.
- Multidimensional FFTs exploit the separability of the exponential kernel, which factorizes the dd d-dimensional DFT into a Kronecker product $F_{N_d} \otimes \cdots \otimes F_{N_1}$ of one-dimensional Fourier matrices. The computation proceeds by applying 1D FFTs along each axis in succession, with total arithmetic cost $O(N \sum_{r=1}^d \log N_r)$ where $N = \prod N_r$. In row-major storage, transforms along the last axis are contiguous and cache-friendly, while transforms along earlier axes require strided access or explicit transposition. Spectral differentiation in pseudo-spectral methods multiplies each Fourier coefficient by the corresponding frequency factor, converting spatial derivatives into pointwise operations in frequency space.
- For real-valued multidimensional data, Hermitian symmetry generalizes to $‾F_{(-k_1 \bmod N_1), \ldots, (-k_d \bmod N_d)} = \overline{F_{k_1, \ldots, k_d}}$, and libraries store only a half-spectrum along one axis with output shape $N_1 \times \cdots \times N_{d-1} \times (\lfloor N_d/2 \rfloor + 1)$. In three dimensions, the padded real storage layout allocates $N_1 \times N_2 \times (N_3 + 2)$ real words so that real and imaginary components of each spectral coefficient occupy adjacent memory locations. Self-conjugate frequency planes at $k_d = 0$ and $k_d = N_d/2$ satisfy an additional two-dimensional Hermitian relation and contain only real-valued coefficients at certain symmetry points.
- When data exceed available memory, FFT execution must be reorganized as a blocked out-of-core computation. The Cooley-Tukey factorization with $N = N_1 N_2$ decomposes the transform into local transforms of size $N_1$, twiddle-factor multiplication, and transforms of size $N_2$ after an index permutation that becomes a block-wise data reorganization step involving external storage. Modern out-of-core pipelines distribute data across multiple SSDs and employ block-wise streaming, GPU-accelerated compression, and sequential access patterns to maximize I/O throughput. Cache-aware FFT execution addresses the memory-bandwidth bottleneck through tiled transpositions, fine-grain representation strategies that reduce TLB and cache miss rates, and vectorized bit-reversal permutations.
- In distributed-memory environments, multidimensional FFTs interleave local transforms with global data redistributions implemented as all-to-all collectives. Communication-minimizing algorithms reduce the number of collective phases by reorganizing axis-transform order and data decomposition schemes, with some designs achieving a single all-to-all step under processor-count constraints. Precision management has become integral to large-scale FFT workflows, with runtime systems selecting among backends and floating-point formats based on predictive error models. Mixed-precision FFT-based pipelines perform bulk transforms in reduced precision while accumulating corrections in higher precision, monitoring deviation from invariants such as Parseval consistency, and achieving substantial performance gains without violating prescribed error tolerances.
- The FFT is a unitarily conditioned transform with condition number one in the 2-norm, so it does not inherently amplify relative perturbations. Roundoff error in a well-implemented FFT grows as $O(\varepsilon \log N)$ where $\varepsilon$ is unit roundoff, reflecting the layered structure of the Cooley-Tukey factorization across $O(\log N)$ butterfly stages. Modeling errors from sampling, aliasing, windowing, and domain truncation are conceptually distinct from roundoff and cannot be reduced by increasing arithmetic precision. In pseudo-spectral time-stepping, per-transform perturbations may accumulate over thousands of iterations, so stability analysis must treat the transform as part of a larger numerical pipeline rather than as an isolated primitive.

## 12.9.2. Advice for Beginners

- The Fourier transform is one of the most influential ideas in numerical computing because it provides a way to analyze data in terms of frequencies rather than physical coordinates. Before studying the FFT itself, ensure that you understand the concepts of periodic functions, complex exponentials, frequency, amplitude, phase, and orthogonality. These ideas form the foundation upon which all Fourier-based algorithms are built.
- Begin with the discrete Fourier transform (DFT) rather than the FFT. Compute small DFTs by hand or through simple programs and observe how sinusoidal signals appear as peaks in the frequency spectrum. Understanding what the transform computes is more important initially than understanding how it is accelerated.
- Next, study sampling theory and the Nyquist criterion. Experiment with signals sampled above and below the Nyquist frequency and observe the effects of aliasing. Understanding the relationship between continuous signals and discrete samples is essential for interpreting Fourier results correctly.
- Pay close attention to spectral leakage and windowing. Many beginners focus entirely on the transform itself and overlook the fact that finite observation intervals introduce artifacts into the spectrum. Compare rectangular and Hann windows to see how window choice affects frequency estimation and spectral interpretation.
- Once the DFT is familiar, study the radix-2 Cooley–Tukey FFT. Focus on the ideas of divide-and-conquer factorization, butterfly operations, and bit-reversal ordering. Understanding how the FFT reduces computational complexity from $O(N^2)$ to $O(N\log N)$ is one of the most important lessons in numerical algorithms.
- After mastering one-dimensional FFTs, explore real-valued transforms, discrete sine transforms, and discrete cosine transforms. These specialized transforms demonstrate how symmetry and boundary conditions can be exploited to achieve greater computational efficiency and provide natural tools for solving differential equations.
- Multidimensional FFTs should be approached gradually. Begin with two-dimensional transforms for image processing applications before moving to three-dimensional transforms used in scientific simulation and computational physics. Pay attention to memory layout, cache efficiency, and data movement, since these often dominate performance in large-scale computations.
- For Rust implementations, become familiar with libraries such as `rustfft`, `realfft`, `ndarray`, and `num-complex`. Start with correctness and spectral interpretation before focusing on optimization, parallel execution, or distributed computation.
- Most importantly, remember that the FFT is far more than a signal-processing tool. It serves as a fundamental computational primitive in partial differential equations, spectral methods, image reconstruction, machine learning, computational physics, astronomy, communications, and scientific simulation. A strong understanding of the concepts presented in this chapter will provide a foundation for many advanced algorithms encountered throughout numerical computing.

## 12.9.3. Further Learning with GenAI

To deepen your understanding of FFT algorithms and applications in Rust, consider using the following GenAI prompts:

 1. Write a Rust program that constructs the DFT matrix $F_N$ using the primitive root of unity $\omega_N = e^{-2\pi i/N}$, verifies the discrete orthogonality relation $F_N^* F_N = NI$, and demonstrates circulant diagonalization by comparing the explicitly constructed circulant matrix $C(h)$ with $F_N^{-1} \mathrm{diag}(F_N h) F_N$. Then solve a periodic Poisson equation using both spectral and finite-difference models and report residuals for each.
 2. Implement a program in Rust that samples a cosine signal at a rate above and below the Nyquist frequency, computes the DFT of each, and identifies the dominant spectral bin. Compare the detected frequency with the predicted alias frequency and demonstrate how spectral replication causes frequency folding when the Nyquist condition is violated.
 3. Build an iterative radix-2 Cooley-Tukey FFT in Rust with bit-reversal input permutation. Compare the output against a direct $O(N^2)$ DFT implementation, verify inverse FFT reconstruction to machine precision, and check Parseval energy conservation. Report the top spectral bins for a multi-tone test signal.
 4. Implement a real-valued FFT in Rust using the even-odd packing construction: split the input into even and odd subsequences, pack them as real and imaginary parts of a half-size complex vector, compute one complex FFT, and recover the full spectrum through Hermitian symmetry. Verify the result against a reference complex FFT, demonstrate half-spectrum storage, and implement the two-real-transforms-for-one-complex-FFT identity.
 5. Write a Rust program that implements a discrete sine transform via FFT of an odd extension and a discrete cosine transform via FFT of an even extension with phase correction. Verify round-trip reconstruction for both transforms. Then solve a 1D Poisson equation with Dirichlet boundary conditions using the DST and with Neumann boundary conditions using the DCT, and compare numerical solutions against exact analytical results.
 6. Implement a two-dimensional FFT in Rust by applying 1D FFTs along rows and then columns, using an explicit transpose to make column transforms contiguous. Demonstrate spectral differentiation by computing $\partial_x u$ via multiplication by $2\pi i k / L_x$ in Fourier space and comparing against the known analytic derivative of a periodic test function.
 7. Build a three-dimensional real-to-complex FFT in Rust that stores only the half-spectrum along the last axis with shape $N_1 \times N_2 \times (\lfloor N_3/2 \rfloor + 1)$. Verify three-dimensional Hermitian symmetry, reconstruct the full spectrum from the stored half, and demonstrate the padded real storage layout $N_1 \times N_2 \times (N_3 + 2)$ with interleaved real and imaginary components.
 8. Implement a blocked out-of-core FFT in Rust using the Cooley-Tukey decomposition $N = N_1 N_2$. Partition the input into disk-resident blocks, perform local $N_1$-point transforms on each block, apply twiddle factors, reorganize blocks for the second transform stage, and verify correctness against a direct DFT. Simulate multi-SSD striping by distributing blocks across separate directories and report I/O statistics.
 9. Write a Rust program that compares two strategies for a 3D FFT: a naive axis-by-axis approach using gather-scatter buffers for non-contiguous axes, and a cache-aware approach using tiled transpositions so that every transform stage operates on contiguous memory. Record memory-access statistics including contiguous versus strided FFT lines and transpose operations, and compare the numerical results to confirm both strategies produce identical spectra.
10. Implement a simulated distributed-memory 3D FFT in Rust using virtual ranks that each own a slab of the global array. Compare a baseline slab decomposition that performs two local transforms followed by one all-to-all redistribution with a communication-minimized plan that uses a different slab orientation to reduce the number of collective phases. Record simulated communication statistics and verify both plans against a reference single-process FFT.

By engaging with these prompts, you will gain a deeper understanding of how the FFT achieves its complexity reduction through structured matrix factorization, how symmetry properties of real-valued and boundary-constrained data lead to specialized algorithms, and how memory layout, communication scheduling, and precision management determine practical FFT performance in modern scientific computing.

## 12.9.4. Homework Exercises

To reinforce your learning, complete the following exercises:

 1. Implement a Rust program that constructs the $16 \times 16$ DFT matrix $F_N$ and verifies the orthogonality relation $F_N^* F_N = NI$ with maximum entrywise error below $10^{-12}$. Build a circulant matrix from the discrete Laplacian stencil $h = [2, -1, 0, \ldots, 0, -1]$ and verify the diagonalization identity $C(h) = F_N^{-1} \mathrm{diag}(F_N h) F_N$. Then solve the periodic Poisson equation using both spectral eigenvalues and finite-difference circulant eigenvalues in Fourier space, and report the maximum residual for each approach.
 2. Write a Rust program that samples a cosine signal at frequency $f_0 = 120$ Hz with sampling step $\Delta = 1$ ms ($f_s = 1000$ Hz, $N = 128$). Compute the DFT, identify the dominant spectral bin, and verify that the detected frequency matches $f_0$. Then repeat with $f_0 = 590$ Hz (above Nyquist) and confirm that the dominant bin corresponds to the predicted alias frequency $f_0 - f_s = -410$ Hz. Apply both rectangular and Hann windows to a non-bin-centered frequency $f_0 = 123.45$ Hz and compare the leakage metric (fraction of spectral energy outside $±1$ bins of the peak) for each window.
 3. Implement an iterative radix-2 Cooley-Tukey FFT in Rust with bit-reversal permutation for $N = 256$. Verify the output against a direct $O(N^2)$DFT with maximum difference below $10^{-10}$. Apply the inverse FFT and confirm reconstruction error below $10^{-10}$. Check Parseval consistency $\sum |x[n]|^2 = \frac{1}{N} \sum |X[k]|^2$ with relative error below $10^{-12}$. Construct a test signal containing frequencies at bins $17$ and $41$ plus a DC offset and verify that the six largest spectral magnitudes correspond to the expected bins and their conjugate partners.
 4. Implement a real-valued FFT in Rust using the even-odd packing construction for $N = 16$. Split the input into even and odd subsequences, pack as $h_m = e_m + i \, o_m$, compute one $N/2$-point complex FFT, extract $E_k$ and $O_k$ using the Hermitian separation formulas, and combine via $F_k = E_k + \omega_N^k O_k$. Verify the result against a full complex FFT with maximum difference below $10^{-12}$. Demonstrate half-spectrum storage by retaining only $F_0, F_1, \ldots, F_{N/2}$ and reconstructing the full spectrum via conjugate symmetry. Implement the two-real-transforms-for-one-complex-FFT identity and verify both extracted spectra against their individual references.
 5. Implement DCT-I and DST-I transforms in Rust via even and odd extensions reduced to FFT. Verify round-trip reconstruction for both transforms with error below $10^{-10}$. Solve the 1D Poisson equation $-u'' = f$ on $(0,1)$ with Dirichlet boundary conditions ($u(0) = u(1) = 0$) using the DST on $n = 16$ interior points, with exact solution $u(x) = \sin(\pi x) + 0.5 \sin(3\pi x)$. Report the maximum solution error and the discrete residual $\max_j |(-u_{j-1} + 2u_j - u_{j+1})/h^2 - f_j|$. Then solve the Neumann problem using the DCT and verify the analogous residual.
 6. Implement a two-dimensional FFT in Rust by composing 1D FFTs along rows and columns for a $64 \times 64$ grid. Compute the spectral derivative $\partial_x u$ of the function $u(x,y) = \sin(5 \cdot 2\pi x / L_x) \cos(7 \cdot 2\pi y / L_y)$ by multiplying Fourier coefficients by $2\pi i k / L_x$and applying the inverse 2D FFT. Compare the result against the exact derivative$\partial_x u = 5(2\pi/L_x) \cos(5 \cdot 2\pi x/L_x) \cos(7 \cdot 2\pi y/L_y)$ and report the maximum error. Verify that the imaginary part of the reconstructed derivative is below $10^{-10}$.
 7. Implement a 3D real-to-complex FFT in Rust for a real field on a $4 \times 5 \times 6$ grid. Verify three-dimensional Hermitian symmetry $‾F_{(-k_1 \bmod N_1), (-k_2 \bmod N_2), (-k_3 \bmod N_3)} = \overline{F_{k_1,k_2,k_3}}$ with maximum residual below $10^{-10}$. Store only the half-spectrum of shape $4 \times 5 \times 4$ along the last axis and reconstruct the full spectrum from conjugate symmetry. Verify reconstruction error below $10^{-10}$. Pack the half-spectrum into a padded real buffer of shape $4 \times 5 \times 8$ using interleaved real/imaginary storage and verify that unpacking reproduces the original half-spectrum exactly.
 8. Implement a blocked out-of-core FFT in Rust with factorization $N = N_1 \times N_2$ ($N_1 = 4$, $N_2 = 6$). Partition the input into $N_2$ disk-resident blocks of length $N_1$, perform local $N_1$-point FFTs, apply twiddle factors $\omega_N^{n_2 k_1}$, reorganize into $N_1$ blocks of length $N_2$, and complete with $N_2$-point FFTs. Simulate multi-SSD striping across 3 directories and verify correctness against a direct DFT with maximum error below $10^{-10}$. Report the total number of block reads, block writes, and bytes transferred.
 9. Implement two strategies for a 3D FFT on a $32 \times 24 \times 20$ complex array in Rust: (a) a naive axis-by-axis approach that uses gather-scatter buffers for non-contiguous axes, and (b) a cache-aware approach that performs tiled transpositions (tile size $8 \times 8$) before each non-contiguous transform stage. Verify that both strategies produce spectra agreeing to within $10^{-10}$. Record and compare the number of contiguous FFT lines, strided FFT lines, temporary buffer elements, and transpose operations for each strategy.
10. Simulate a distributed-memory 3D FFT in Rust for an $8 \times 8 \times 8$ array partitioned across 4 virtual ranks. Implement a baseline plan that performs FFTs on axes 2 and 1 locally on x-slabs, redistributes to y-slabs via a simulated all-to-all, and then performs the axis-0 FFT locally. Implement a communication-minimized plan that redistributes from x-slabs to z-slabs after the axis-2 FFT, enabling both axis-1 and axis-0 transforms to be performed locally. Verify both plans against a reference single-process FFT with error below $10^{-10}$. Report the number of all-to-all calls, total elements moved, and synchronization points for each plan.

These exercises span the full range of FFT algorithms developed in this chapter, from the algebraic foundations of circulant diagonalization and sampling theory through the radix-2 Cooley-Tukey factorization, real-valued and trigonometric transform variants, multidimensional separable composition, and memory-aware and distributed execution strategies. By implementing them in Rust, you will gain direct experience with the interplay between mathematical structure and computational efficiency that makes the FFT one of the most important algorithms in scientific computing.

# References

 1. Asahi, Y., Padioleau, T., Zehner, P., Bigot, J. and Lebrun-Grandie, D. (2025) ‘kokkos-fft: A shared-memory FFT for the Kokkos ecosystem’, *Journal of Open Source Software*, 10(111), 8391. doi:10.21105/joss.08391.
 2. Asahi, Y. *et al.* (2025) ‘Development of a performance portable distributed FFT interface on top of the Kokkos ecosystem’, in *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis Workshops (SC Workshops ’25)*. doi:10.1145/3731599.3767494.
 3. Bielak, K., Cariow, A. and Raciborski, M. (2024) ‘The Development of Fast DST-II Algorithms for Short-Length Input Sequences’, *Electronics*, 13(12), 2301. doi:10.3390/electronics13122301.
 4. Boccardo, A.D., Tong, M., Leen, S.B., Tourret, D. and Segurado, J. (2023) ‘Efficiency and accuracy of GPU-parallelized Fourier spectral methods for solving phase-field models’, *Computational Materials Science*, 228, 112313. doi:10.1016/j.commatsci.2023.112313.
 5. Cao, J., Yang, Z., Sun, R. and Chen, X. (2023) ‘Delay sampling theorem: A criterion for the recovery of multitone signal’, *Mechanical Systems and Signal Processing*, 200, 110523. doi:10.1016/j.ymssp.2023.110523.
 6. Chen, T. *et al.* (2023) ‘OpenFFT: An Adaptive Tuning Framework for 3D FFT on ARM Multicore CPUs’, in *Proceedings of the 37th ACM International Conference on Supercomputing (ICS ’23)*, pp. 398–409. doi:10.1145/3577193.3593735.
 7. Chowdhury, S. and Al Sakib, A. (2024) *Numerical Exploration of Fourier Transform and Fourier Series: The Power Spectrum of Driven Damped Oscillators*. Cham: Springer. doi:10.1007/978-3-031-34664-4.
 8. Cox, T.A., Murray, S.G., Parsons, A.R., Dillon, J.S. *et al.* (2025) ‘fftvis: a non-uniform Fast Fourier Transform-based interferometric visibility simulator’, *RAS Techniques and Instruments*, 4, rzaf056. doi:10.1093/rasti/rzaf056.
 9. Cunnington, S. and Wolz, L. (2024) ‘Accurate Fourier-space statistics for line intensity mapping: Cartesian grid sampling without aliased power’, *Monthly Notices of the Royal Astronomical Society*, 528(4), pp. 5586–5600. doi:10.1093/mnras/stae333.
10. Diez Sanhueza, R., Peeters, J. and Costa, P. (2025) ‘A pencil-distributed finite-difference solver for extreme-scale calculations of turbulent wall flows at high Reynolds number’, *Computer Physics Communications*, 316, 109811. doi:10.1016/j.cpc.2025.109811.
11. Ding, X. *et al.* (2025) ‘Memory-Efficient Training with In-Place FFT Implementation’, *arXiv* preprint. doi:10.48550/arXiv.2511.01385.
12. Gilan, M.S. and Maham, B. (2024) ‘Optimized power and speed of Split-Radix, Radix-4 and Radix-2 FFT structures’, *EURASIP Journal on Advances in Signal Processing*, 2024, 81. doi:10.1186/s13634-024-01178-4.
13. Henry, M. (2024) ‘An ultra-precise Fast Fourier Transform’, *Measurement: Sensors*, 32, 101039. doi:10.1016/j.measen.2024.101039.
14. Huang, S., Hong, M., Lin, G., Tang, B. and Shen, S. (2025) ‘A Discrete Fourier Transform-Based Signal Processing Method for an Eddy Current Detection Sensor’, *Sensors*, 25(9), 2686. doi:10.3390/s25092686.
15. Ibrahim, M.A. and Aga, S. (2024) ‘Pimacolaba: Collaborative Acceleration for FFT on Commercial Processing-In-Memory Architectures’, in *Proceedings of the International Symposium on Memory Systems (MEMSYS ’24)*. doi:10.1145/3695794.3695796.
16. Khattab, A.G., Semary, M.S., Hammad, D.A. and Fareed, A.F. (2024) ‘Exploring Stochastic Heat Equations: A Numerical Analysis with Fast Discrete Fourier Transform Techniques’, *Axioms*, 13(12), 886. doi:10.3390/axioms13120886.
17. Kircheis, M., Potts, D. and Tasche, M. (2023) ‘Nonuniform fast Fourier transforms with nonequispaced spatial and frequency data and fast sinc transforms’, *Numerical Algorithms*, 92, pp. 2307–2339. doi:10.1007/s11075-022-01389-6.
18. Kohli, A. *et al.* (2025) ‘Ring deconvolution microscopy: exploiting symmetry for efficient spatially varying aberration correction’, *Nature Methods*, 22, pp. 1311–1320. doi:10.1038/s41592-025-02684-5.
19. Koopman, T. and Bisseling, R.H. (2023) ‘Minimizing Communication in the Multidimensional FFT’, *SIAM Journal on Scientific Computing*, 45(6), pp. C330–C347. doi:10.1137/22M1487242.
20. Lee, J. and Kim, D. (2023) ‘Out-of-core diffraction algorithm using multiple SSDs for ultra-high-resolution hologram generation’, *Optics Express*, 31(18), pp. 28683–28700. doi:10.1364/OE.493984.
21. Lee, J. and Kim, D. (2024) ‘COMBO: compressed block-wise out-of-core diffraction computation for tera-scale holography’, *Optics Express*, 32(27), pp. 47993–48008. doi:10.1364/OE.543103.
22. Lehner, J., Arima, E. and Schulz, M. (2025) ‘A GPU FFT Wrapper to Co-optimize Floating-Point Precision and Library Selection via Predictive Error Modeling’, in *SC Workshops ’25: PMBS25 Workshop Archive*. doi:10.1145/3731599.3767703.
23. Liu, L. *et al.* (2025) ‘Performance evaluation of the inverse real-valued fast Fourier transform on field programmable gate array platforms using open computing language’, *PeerJ Computer Science*, 11, e3313. doi:10.7717/peerj-cs.3313.
24. Namugwanya, E., Bienz, A., Schafer, D. and Skjellum, A. (2023) ‘Collective-Optimized FFTs’, *arXiv* preprint. doi:10.48550/arXiv.2306.16589.
25. Paux, J., Morin, L., Gélébart, L. and Sanoko, A.M.A. (2025) ‘A discrete sine–cosine based method for the elasticity of heterogeneous materials with arbitrary boundary conditions’, *Computer Methods in Applied Mechanics and Engineering*, 433(Part A), 117488. doi:10.1016/j.cma.2024.117488.
26. Pei, J. and Tong, X. (2025) ‘A Hybrid DST-Accelerated Finite-Difference Solver for 2D and 3D Poisson Equations with Dirichlet Boundary Conditions’, *Mathematics*, 13(17), 2776. doi:10.3390/math13172776.
27. Polyakova, M., Witenberg, A. and Cariow, A. (2025) ‘The fast type-IV discrete sine transform algorithms for short-length input sequences’, *Bulletin of the Polish Academy of Sciences: Technical Sciences*, 73(4), e153827. doi:10.24425/bpasts.2025.153827.
28. Pundir, M. and Kammer, D.S. (2025) ‘Simplifying FFT-based methods for solid mechanics with automatic differentiation’, *Computer Methods in Applied Mechanics and Engineering*, 435, 117572. doi:10.1016/j.cma.2024.117572.
29. Risthaus, L. and Schneider, M. (2024) ‘Imposing Dirichlet boundary conditions directly for FFT-based computational micromechanics’, *Computational Mechanics*, 74, pp. 1089–1113. doi:10.1007/s00466-024-02469-1.
30. Salih, S.K. and Hamood, M.T. (2023) ‘New Algorithm for Real-Valued Fourier Transform’, *Tikrit Journal of Engineering Sciences*, 30(4). doi:10.25130/tjes.30.4.13.
31. Servodio, S. and Li, X. (2024) ‘Memory Efficiency Oriented Fine-Grain Representation and Optimization of FFT’, in *Proceedings of the International Symposium on Memory Systems (MEMSYS ’24)*, pp. 245–256. doi:10.1145/3695794.3695818.
32. Simek, A. and Šimeček, I. (2023) ‘Design of a modern fast Fourier transform and cache effective bit-reversal algorithm’, *International Journal of Parallel, Emergent and Distributed Systems*. doi:10.1080/17445760.2023.2179049.
33. Venkat, S., Świrydowicz, K., Wolfe, N. and Ghattas, O. (2025) ‘Mixed-Precision Performance Portability of FFT-Based GPU-Accelerated Algorithms for Block-Triangular Toeplitz Matrices’, in *SC Workshops ’25*. doi:10.1145/3731599.3767490. (Author version also on arXiv: 10.48550/arXiv.2508.10202.)
34. Wu, S. *et al.* (2024) ‘TurboFFT: Co-Designed High-Performance and Fault-Tolerant Fast Fourier Transform on GPUs’, *arXiv* preprint. doi:10.48550/arXiv.2412.05824.
35. Yan, Y., Kuk, N. and Grant, R.E. (2024) ‘Improved MPI Collectives for 3D-FFT’, in *Recent Advances in the Message Passing Interface*, pp. 75–88. doi:10.1007/978-3-031-73370-3_5.
36. Yang, D. *et al.* (2025) ‘SSFFT: Energy-Efficient Selective Scaling for Fast Fourier Transform in Embedded GPUs’, in *Proceedings of LCTES ’25*, pp. 85–96. doi:10.1145/3735452.3735529.
37. Yeung, P.K., Ravikumar, K., Nichols, S. and Uma-Vaideswaran, R. (2025) ‘GPU-enabled extreme-scale turbulence simulations: Fourier pseudo-spectral algorithms at the exascale using OpenMP offloading’, *Computer Physics Communications*, 306, 109364. doi:10.1016/j.cpc.2024.109364.
38. Zeng, Z., Liu, J. and Yuan, Y. (2024) ‘A generalized Nyquist-Shannon sampling theorem using the Koopman operator’, *IEEE Transactions on Signal Processing*, 72, pp. 3595–3610. doi:10.1109/TSP.2024.3436610.
39. Zhao, Z., Dong, H. and Ying, W. (2024) ‘An FFT-Based MAC scheme for Stokes equations with periodic boundary conditions and its application to elasticity problems’, *Journal of Computational and Applied Mathematics*, 439, 115624. doi:10.1016/j.cam.2023.115624.
