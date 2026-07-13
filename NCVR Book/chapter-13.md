---
title: Chapter 13
description: ''
subtitle: Fourier and Spectral Applications
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
      literal: Ayesha Ayub Syed, Ph.D
      given: Ph.D
      family: Ayesha Ayub Syed
    name: Ayesha Ayub Syed, Ph.D
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
date: '2026-02-03'
oxa: oxa:pqQDe4beUu67RvW3raYP/kN075AFyDzJgibkxMjWO
keywords: []
---

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/vqEnTIy0h3fqK4JiodPg.1","tags":[]}

> Fourier’s genius lies in showing us that any complex phenomenon can be broken down into simple waves, allowing spectral methods to reveal the hidden harmonies of nature.
>
> — John P. Boyd

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/vUAaXz0382xr9bHhPFdn.1","tags":[]}

*Chapter 13 explores advanced applications of Fourier and spectral methods in signal processing, inverse problems, filtering, and data analysis. Building on the FFT foundations developed in the previous chapter, the discussion covers convolution, deconvolution, correlation, Wiener filtering, power spectrum estimation, digital filtering, linear prediction, maximum-entropy spectral methods, and the analysis of unevenly sampled data. The chapter also introduces numerical techniques for computing Fourier integrals, wavelet transforms, and practical applications of the sampling theorem. Emphasis is placed on the interaction between mathematical models, statistical estimation, and computational efficiency. Throughout the chapter, Rust implementations demonstrate how spectral techniques can be applied to modern problems in scientific computing, communications, imaging, geophysics, and machine learning.*

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/qtSICKR9580vUiBrFNdM.10","tags":[]}

# 13.1. Introduction

Fourier and spectral methods occupy a central place in numerical computing because they unify mathematical structure, efficient algorithms, and demanding scientific applications. Many computational problems can be viewed in one of two ways. In the first, one wishes to apply a translation-invariant linear operator, such as a smoothing kernel, a differentiation operator, a filter, or a Green’s function. In the second, one wishes to infer an unknown signal or field from measurements that have been blurred, mixed, shifted, or contaminated by noise. Both viewpoints naturally lead to convolution and correlation, and the efficient evaluation of these operations is one of the main reasons Fourier methods are so widely used in scientific computation. This importance has only increased in modern high-performance computing, where Fourier pseudo-spectral solvers, especially in three spatial dimensions, depend heavily on fast transforms and are often limited by FFT performance and communication costs on GPU-based architectures (Yeung et al., 2025).

Throughout this chapter we work with finite discrete signals. Let,

$$x \in \mathbb{C}^{N}, \qquad x = (x_0,x_1,\dots,x_{N-1})^{\top} \tag{13.1.1}$$

with the understanding that in many practical applications the data are real-valued, so that $x_j \in \mathbb{R}$. The fundamental transform is the discrete Fourier transform (DFT). Using the standard engineering normalization, the forward transform is:

$$X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i kn/N}, \qquad k = 0,1,\dots,N-1 \tag{13.1.2}$$

and the inverse transform is:

$$x_n = \frac{1}{N}\sum_{k=0}^{N-1} X_k e^{2\pi i kn/N}, \qquad n = 0,1,\dots,N-1 \tag{13.1.3}$$

A direct evaluation of (13.1.2) or (13.1.3) requires $O(N^2)$ arithmetic operations. The FFT is an algorithmic family that computes the same maps in $O(N \log N)$ operations, which is the decisive complexity reduction that makes Fourier-domain methods practical at large scale. At the same time, one must remember that finite-length sampling and floating-point arithmetic introduce issues of leakage, discretization error, and numerical sensitivity, so the FFT is not merely a fast black box but a numerical procedure whose outputs must be interpreted carefully (Henry, 2024).

A useful linear-algebraic formulation is obtained by introducing the Fourier matrix $F \in \mathbb{C}^{N\times N}$, defined by:

$$F_{k,n} = e^{-2\pi i kn/N}, \qquad k,n = 0,1,\dots,N-1\tag{13.1.4}$$

Then the DFT and inverse DFT can be written compactly as:

$$X = Fx,\tag{13.1.5}$$

$$x = \frac{1}{N}F^{*}X \tag{13.1.6}$$

This formulation makes clear that the DFT is a change of basis, unitary up to the scaling factor $N$. It also reveals why Fourier methods are so powerful for structured operators. In particular, circulant matrices are diagonalized by the Fourier basis, and this algebraic fact lies behind the FFT-based acceleration of convolution, deconvolution, and correlation.

A first major source of Fourier structure arises in the discretization of partial differential equations. On a periodic grid, constant-coefficient differential operators become multiplication operators in Fourier space. If $u$ is represented by its Fourier coefficients and $\kappa$ denotes the wavenumber, then differentiation acts schematically as:

$$\widehat{\partial_x u}(\kappa) = i\kappa\,\hat{u}(\kappa) \tag{13.1.7}$$

Thus, after transforming to the spectral domain, many linear PDE operations reduce to pointwise multiplication. This is one of the reasons Fourier pseudo-spectral methods are so effective in fluid dynamics, wave propagation, and turbulence simulation. In such applications, the cost of the overall solver is often driven by repeated FFTs rather than by the differential operator itself (Yeung et al., 2025).

A second major source of Fourier structure arises in measurement and imaging models. Many physical observation systems can be idealized, after discretization, by the relation

$$y = h * x + \eta \tag{13.1.8}$$

where $x$ is the latent signal or field, $h$ is the response kernel or point-spread function, $y$ is the measured data, and $\eta$ represents noise. Equation (13.1.8) is the canonical convolution model for blur, spreading, and instrument response. It remains central in microscopy, ultrasound, optical imaging, and related inverse problems, where modern work focuses on improved regularization, stability, and optimization strategies for recovering $x$ from noisy data (Xu et al., 2024; Li et al., 2024).

Correlation arises when the goal is not to model system response but to measure similarity under relative displacement. It is fundamental in alignment, synchronization, template matching, and delay estimation. In practical terms, if two signals differ mainly by a lag, then their correlation function peaks near the correct shift. This simple principle underlies a large class of methods in signal processing and monitoring systems. Recent work continues to refine correlation-based estimators in low signal-to-noise regimes and in applications such as acoustic leak detection and robust time-delay estimation (Uchendu et al., 2025; Sun et al., 2025).

The computational power of FFT-based methods is best understood through structured matrices. Let $x \in \mathbb{R}^{N}$ and let $h=(h_0,h_1,\dots,h_{M-1})\in\mathbb{R}^{M}$ be a finite impulse response. The linear convolution $y=h*x$ has length $N+M-1$ and is given by:

$$y_n = \sum_{m=0}^{M-1} h_m x_{n-m},\qquad n = 0,1,\dots,N+M-2,\tag{13.1.9}$$

with the convention that $x_r=0$ whenever $r\notin{0,1,\dots,N-1}$. This operation may be written as a Toeplitz matrix-vector product,

$$y = T(h)x \tag{13.1.10}$$

where $T(h)$ is the Toeplitz matrix generated by the filter coefficients. A direct implementation of (13.1.9) requires $O(NM)$ operations, which becomes $O(N^2)$ when $M$ and $N$ are of comparable size.

For equal-length complex vectors $x,h \in \mathbb{C}^{N}$, one may instead define the circular convolution:

$$(y = x \circledast h)_n = \sum_{m=0}^{N-1} x_m\, h_{(n-m)\bmod N}, \qquad n = 0,1,\dots,N-1. \tag{13.1.11}$$

Circular convolution corresponds to multiplication by a circulant matrix $C(h)$,

$$y = C(h)x \tag{13.1.12}$$

The key structural property is that circulant matrices are diagonalized by the DFT. If $H=Fh$, then,

$$C(h) = \frac{1}{N} F^{*}\,\operatorname{diag}(H_0, H_1, \dots, H_{N-1})\,F \tag{13.1.13}$$

Consequently, (13.1.11) is equivalent to the componentwise relation:

$$Y_k = X_k H_k, \qquad k = 0,1,\dots,N-1 \tag{13.1.14}$$

where $X=Fx$, $H=Fh$, and $Y=Fy$. Equation (13.1.14) is the discrete convolution theorem in its most useful computational form. It shows that convolution may be performed by transforming the inputs, multiplying their Fourier coefficients pointwise, and transforming back. The FFT makes this strategy asymptotically much cheaper than direct Toeplitz multiplication.

This chapter develops that idea in progressively richer settings. The first task is to understand how finite linear convolution problems can be embedded into circular convolution problems by padding, so that FFT-based computation becomes applicable. The second is to examine what happens when one attempts inversion rather than forward application, since deconvolution is generally ill-conditioned even when convolution itself is stable. The third is to extend the same Fourier viewpoint to correlation and autocorrelation, where similarity, lag estimation, and second-order structure become central. Taken together, these topics show that the FFT is not merely a fast transform, but a fundamental computational bridge between structured linear algebra, spectral modeling, and practical numerical algorithms.

### Rust Implementation

Following the discussion in Section 13.1 on the structure of the discrete Fourier transform and the convolution theorem, Program 13.1.1 provides a concrete implementation of these ideas in Rust. The program evaluates the forward and inverse discrete Fourier transforms directly from their definitions and uses them to demonstrate how circular convolution can be computed either in the time domain or in the spectral domain. In numerical computation, the Fourier transform is not merely a mathematical abstraction but a practical computational tool that converts structured linear algebra operations into simple pointwise multiplications. This program therefore illustrates how the algebraic relation between circulant matrices and the Fourier basis leads to an efficient computational strategy. By computing convolution both directly and through spectral multiplication, the implementation verifies the discrete convolution theorem introduced in Equation (13.1.14) and highlights the numerical consistency of the transform pair defined by Equations (13.1.2) and (13.1.3).

At the core of the implementation are the functions `dft` and `idft`, which directly implement the discrete Fourier transform and its inverse as defined in Equations (13.1.2) and (13.1.3). The `dft` function computes each Fourier coefficient by summing the products of the input signal with the complex exponential basis functions. This procedure mirrors the matrix formulation $X = Fx$ given in Equation (13.1.5), where the Fourier matrix provides a change of basis from the time domain to the spectral domain. The `idft` function performs the reverse transformation according to Equation (13.1.6), reconstructing the original signal from its spectral representation while applying the normalization factor $1/N$. Although these implementations require $O(N^2)$ operations, they are deliberately written in this direct form to reflect the mathematical definitions introduced in the section.

The program also includes the function `circular_convolution_direct`, which computes circular convolution according to Equation (13.1.11). This function evaluates the convolution sum explicitly by iterating through the input vectors and applying modular indexing to implement the periodic structure inherent in circular convolution. The resulting operation corresponds to multiplication by the circulant matrix $C(h)$ described in Equation (13.1.12). This implementation therefore provides a direct computational realization of the structured matrix interpretation discussed in the section.

A complementary function, `circular_convolution_via_dft`, demonstrates the spectral approach to convolution. It first computes the transforms of the input vectors using the `dft` function, then forms the pointwise product of the resulting Fourier coefficients, and finally reconstructs the output using the inverse transform. This sequence of operations implements the relation $Y_k = X_k H_k$ introduced in Equation (13.1.14), which expresses circular convolution as multiplication in the Fourier domain. The function returns the intermediate spectral quantities as well as the reconstructed signal so that the transformation process can be inspected and verified.

To assess numerical accuracy, the implementation includes a utility function `max_abs_error`, which computes the maximum difference between two complex vectors. This function is used to compare the result of direct convolution with the result obtained via spectral multiplication. Because the two approaches are mathematically equivalent, their difference should be limited to floating-point roundoff errors. The program also includes a helper function `print_vector` that formats complex vectors for display, making the intermediate and final results easier to interpret.

The `main` function serves to demonstrate the computational ideas developed in Section 13.1. It begins by defining two short complex vectors representing a signal and a convolution kernel. The program first prints the input vectors and computes their discrete Fourier transforms. It then evaluates circular convolution in two ways: directly using the definition of Equation (13.1.11) and spectrally using the convolution theorem of Equation (13.1.14). The resulting vectors are printed and compared numerically using the maximum absolute error. Finally, the program verifies that applying the inverse transform to the transform of the signal reproduces the original data, confirming the numerical consistency of the transform pair defined by Equations (13.1.2) and (13.1.3).

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
```

```rust
use num_complex::Complex64;
use std::f64::consts::PI;

/// Program 13.1.1
/// Direct DFT, inverse DFT, and verification of the discrete convolution theorem.
///
/// This program implements:
/// 1. The forward DFT from equation (13.1.2)
/// 2. The inverse DFT from equation (13.1.3)
/// 3. Direct circular convolution from equation (13.1.11)
/// 4. Spectral-domain convolution using equation (13.1.14)
///
/// It then checks that the time-domain and Fourier-domain results agree.

/// Compute the forward discrete Fourier transform:
/// X_k = sum_{n=0}^{N-1} x_n exp(-2*pi*i*k*n/N)
fn dft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    let mut xhat = vec![Complex64::new(0.0, 0.0); n];

    for k in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for (m, &xm) in x.iter().enumerate() {
            let theta = -2.0 * PI * (k as f64) * (m as f64) / (n as f64);
            let w = Complex64::new(theta.cos(), theta.sin());
            sum += xm * w;
        }
        xhat[k] = sum;
    }

    xhat
}

/// Compute the inverse discrete Fourier transform:
/// x_n = (1/N) sum_{k=0}^{N-1} X_k exp(2*pi*i*k*n/N)
fn idft(xhat: &[Complex64]) -> Vec<Complex64> {
    let n = xhat.len();
    let mut x = vec![Complex64::new(0.0, 0.0); n];

    for m in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for (k, &xk) in xhat.iter().enumerate() {
            let theta = 2.0 * PI * (k as f64) * (m as f64) / (n as f64);
            let w = Complex64::new(theta.cos(), theta.sin());
            sum += xk * w;
        }
        x[m] = sum / (n as f64);
    }

    x
}

/// Compute direct circular convolution:
/// y_n = sum_{m=0}^{N-1} x_m h_{(n-m) mod N}
fn circular_convolution_direct(x: &[Complex64], h: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(x.len(), h.len(), "Signals must have the same length.");
    let n = x.len();
    let mut y = vec![Complex64::new(0.0, 0.0); n];

    for idx in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for m in 0..n {
            let j = (idx + n - m) % n;
            sum += x[m] * h[j];
        }
        y[idx] = sum;
    }

    y
}

/// Compute circular convolution via the DFT:
/// 1. X = DFT(x)
/// 2. H = DFT(h)
/// 3. Y_k = X_k * H_k
/// 4. y = IDFT(Y)
fn circular_convolution_via_dft(
    x: &[Complex64],
    h: &[Complex64],
) -> (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
    assert_eq!(x.len(), h.len(), "Signals must have the same length.");

    let xhat = dft(x);
    let hhat = dft(h);

    let yhat: Vec<Complex64> = xhat
        .iter()
        .zip(hhat.iter())
        .map(|(&a, &b)| a * b)
        .collect();

    let y = idft(&yhat);
    (xhat, hhat, yhat, y)
}

/// Maximum absolute entrywise error between two complex vectors.
fn max_abs_error(a: &[Complex64], b: &[Complex64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length.");
    a.iter()
        .zip(b.iter())
        .map(|(&u, &v)| (u - v).norm())
        .fold(0.0_f64, f64::max)
}

/// Pretty-print a complex vector with real and imaginary parts.
fn print_vector(name: &str, v: &[Complex64]) {
    println!("{name}");
    for (i, z) in v.iter().enumerate() {
        println!("  [{:2}] = {:>12.8} {:+12.8}i", i, z.re, z.im);
    }
    println!();
}

fn main() {
    // Example signals of equal length, as required for circular convolution.
    // These are chosen real-valued here, although the implementation supports complex data.
    let x = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];

    let h = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(2.0, 0.0),
    ];

    println!("Program 13.1.1: Direct DFT and the Discrete Convolution Theorem\n");

    print_vector("Input signal x:", &x);
    print_vector("Input kernel h:", &h);

    // Direct circular convolution in the time domain.
    let y_direct = circular_convolution_direct(&x, &h);

    // Circular convolution in the spectral domain.
    let (xhat, hhat, yhat, y_spectral) = circular_convolution_via_dft(&x, &h);

    print_vector("DFT of x, denoted X:", &xhat);
    print_vector("DFT of h, denoted H:", &hhat);
    print_vector("Pointwise product Y_k = X_k H_k:", &yhat);

    print_vector("Direct circular convolution y = x ⊛ h:", &y_direct);
    print_vector("Spectral convolution y = IDFT(X ⊙ H):", &y_spectral);

    let err = max_abs_error(&y_direct, &y_spectral);
    println!("Maximum absolute error between the two convolution results: {:.3e}", err);

    if err < 1.0e-10 {
        println!("\nVerification successful: the computation confirms equation (13.1.14).");
    } else {
        println!("\nVerification failed: the discrepancy is larger than expected.");
    }

    // Additional check: IDFT(DFT(x)) ≈ x
    let x_recovered = idft(&dft(&x));
    let rec_err = max_abs_error(&x, &x_recovered);

    println!(
        "Maximum reconstruction error in IDFT(DFT(x)) - x: {:.3e}",
        rec_err
    );
}
```

Program 13.1.1 demonstrates the fundamental computational principle underlying Fourier methods: operations that appear complicated in the time domain can become simple algebraic manipulations in the spectral domain. By implementing both direct circular convolution and its Fourier-domain counterpart, the program verifies the discrete convolution theorem introduced in Equation (13.1.14) and illustrates the close connection between structured matrices and Fourier transforms discussed in Section 13.1.

The results of the program show that the two convolution strategies produce identical outputs up to floating-point roundoff error. This agreement confirms that the Fourier transform and its inverse form a numerically consistent transform pair and that convolution may indeed be implemented through spectral multiplication. Although the transform in this program is computed using the direct $O(N^2)$ formulation of Equations (13.1.2) and (13.1.3), the same procedure becomes dramatically more efficient when the discrete Fourier transform is replaced by an FFT algorithm.

This introductory implementation therefore serves as a conceptual bridge between the mathematical structure developed in Section 13.1 and the more advanced algorithms introduced later in the chapter. In subsequent sections, the same framework will be extended to linear convolution via zero padding, to deconvolution and inverse problems, and to correlation and autocorrelation methods used in signal analysis and system identification.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/EGIM6MBIAejvL543RIF6.8","tags":[]}

# 13.2. Convolution and Deconvolution Using the FFT

The computational usefulness of the Fourier transform becomes especially clear when one moves from abstract spectral representations to concrete linear operations on data. Among these, convolution is the central example. In direct form, discrete convolution is a structured matrix-vector multiplication, typically Toeplitz in the linear case and circulant in the periodic case. In Fourier space, however, circulant structure becomes diagonal, so convolution reduces to componentwise multiplication. This reduction is the basis of fast convolution algorithms and, in turn, of many practical filtering, simulation, and inverse-problem workflows. The same diagonalization principle also motivates deconvolution, although inversion is substantially more delicate because small spectral coefficients can amplify noise and modeling errors. For that reason, FFT-based deconvolution must be understood not merely as a formal division in the frequency domain, but as a regularized inverse problem.

## 13.2.1. Linear Versus Circular Convolution and the FFT Reduction

The basic FFT identity for convolution is:

$$\operatorname{DFT}(x \circledast h) = \operatorname{DFT}(x) \odot \operatorname{DFT}(h) \tag{13.2.1}$$

where $\circledast$ denotes circular convolution and $\odot$ denotes componentwise multiplication. This identity is exact, but it applies to length-$N$ vectors interpreted periodically. In many scientific and engineering settings, however, the intended operation is not circular convolution but *linear convolution* of two finite segments.

Let $x$ have length $N_x$ and $h$ have length $N_h$. Their linear convolution has length:

$$N_y = N_x + N_h - 1 \tag{13.2.2}$$

If $x=(x_0,\dots,x_{N_x-1})^{\top}$ and $h=(h_0,\dots,h_{N_h-1})^{\top}$, then the linear convolution is:

$$y_n = \sum_{m=0}^{N_h-1} h_m\,x_{n-m}, \qquad n = 0,\dots,N_x+N_h-2 \tag{13.2.3}$$

with the convention that $x_r=0$ whenever $r\notin{0,\dots,N_x-1}$. Direct evaluation of (13.2.3) requires $O(N_xN_h)$ operations, which is often written as $O(N^2)$ when $N_x\sim N_h$.

The FFT-based reduction proceeds by embedding the desired linear convolution into a circular convolution of a longer length. Choose an integer $N$ such that:

$$N \geq N_x + N_h - 1 \tag{13.2.4}$$

often taking $N$ to be an FFT-friendly length rather than exactly the minimum. Define zero-padded vectors $\tilde{x},\tilde{h}\in\mathbb{C}^{N}$ by appending zeros to $x$ and $h$. Then one computes:

$$X = \operatorname{DFT}(\tilde{x}), \qquad H = \operatorname{DFT}(\tilde{h}) \tag{13.2.5}$$

$$Y=X\odot H \tag{13.2.6}$$

$$\tilde{y} = \operatorname{IDFT}(Y) \tag{13.2.7}$$

The first $N_x+N_h-1$ entries of $\tilde{y}$ then equal the desired linear convolution, and the remaining entries are zero provided that the padding length satisfies (13.2.4). Thus linear convolution is reduced to a circular convolution of sufficiently padded vectors, which is the essential FFT trick behind fast convolution and frequency-domain FIR filtering, including overlap-add and overlap-save methods (Johansson and Gustafsson, 2023).

The asymptotic advantage is immediate. The dominant cost is that of two forward FFTs and one inverse FFT, so the total complexity is approximately,

$$\text{cost} \approx 3\,O(N \log N) \tag{13.2.8}$$

plus $O(N)$ pointwise complex multiplications. This is the fundamental improvement over the direct Toeplitz-style cost $O(N_xN_h)$. At the same time, modern discussions emphasize that FFT-based methods should not be interpreted purely algebraically: finite-length sampling, leakage, and floating-point effects matter when one interprets frequency-domain operations in practice (Henry, 2024).

The memory cost is also favorable. One stores a small number of length-$N$ arrays, typically the padded inputs together with their complex transforms, so the working memory is:

$$O(N)\tag{13.2.9}$$

This favorable balance of arithmetic complexity and storage is one reason FFT-based convolution remains a core primitive in large-scale numerical computing.

### Rust Implementation

Following the discussion in Section 13.2.1 on the reduction of linear convolution to circular convolution through zero padding, Program 13.2.1 provides a practical implementation of FFT-based convolution. In numerical computing, the convolution of two finite signals arises frequently in filtering, simulation, and inverse problems, but a direct evaluation based on Equation (13.2.3) requires $O(N_xN_h)$ arithmetic operations. The Fourier-domain reduction described in Equations (13.2.5)–(13.2.7) replaces this quadratic cost with a sequence of fast transforms and componentwise multiplications, reducing the asymptotic complexity to approximately $3\,O(N\log N)$ as indicated in Equation (13.2.8). The program demonstrates how linear convolution can be embedded into a circular convolution by padding the input signals to a sufficiently large length satisfying Equation (13.2.4). It then computes the convolution both directly and via the FFT in order to verify the correctness of the reduction and to illustrate the numerical agreement between the two approaches.

At the core of the implementation are the functions that construct the FFT-based convolution pipeline described in Equations (13.2.5)–(13.2.7). The `fft` and `ifft` functions compute the forward and inverse discrete Fourier transforms of complex vectors using the `rustfft` library. These operations correspond to the spectral transforms appearing in Equation (13.2.5) and Equation (13.2.7). The inverse transform includes the normalization factor $1/N$, ensuring consistency with the transform pair defined earlier in the chapter.

The program also includes a direct implementation of linear convolution through the function `direct_linear_convolution`. This function evaluates the convolution sum defined in Equation (13.2.3) by iterating over the signal and filter indices and accumulating the weighted products of the filter coefficients with appropriately shifted signal values. This direct evaluation reflects the Toeplitz-style matrix–vector interpretation of convolution discussed earlier in the section and serves as a reference computation for validating the FFT-based method.

To implement the reduction from linear convolution to circular convolution, the program constructs padded complex vectors using the `zero_pad` function. The padding length is determined by the function `next_power_of_two`, which selects an FFT-friendly transform size that satisfies the requirement of Equation (13.2.4). The real input signals are first converted to complex form using `real_to_complex`, after which they are padded with zeros so that the circular convolution computed by the FFT corresponds to the desired linear convolution.

The FFT-based convolution itself is implemented in the function `fft_linear_convolution`. This function performs the sequence of operations described in Equations (13.2.5)–(13.2.7). First, the padded signals are transformed into the spectral domain using the FFT. Their Fourier coefficients are then multiplied componentwise, implementing the product relation in Equation (13.2.6). Finally, the inverse FFT reconstructs the convolution result in the time domain. Because the padding length satisfies Equation (13.2.4), the first $N_x+N_h-1$ entries of the inverse transform coincide exactly with the desired linear convolution.

Two small utility functions support the numerical verification of the algorithm. The function `max_abs_error` computes the maximum difference between two real vectors, allowing the program to quantify the agreement between the direct convolution and the FFT-based result. The helper function `print_real_vector` formats vectors with indices for clear display of intermediate and final results.

The `main` function demonstrates the full computation. It begins by defining example signals representing a finite input sequence and a short convolution kernel. The program computes the convolution using both the direct method and the FFT-based reduction, prints the resulting vectors, and evaluates the maximum absolute error between them. This comparison confirms that the padded circular convolution computed through the FFT reproduces the direct linear convolution within floating-point roundoff error.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
rustfft = "6"
```

```rust
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Program 13.2.1
/// Linear Convolution via FFT with Zero Padding
///
/// This program demonstrates the reduction of linear convolution to circular
/// convolution by zero padding, as described in Equations (13.2.4) to (13.2.7).
/// It computes the linear convolution of two finite real signals in two ways:
/// 1. Directly from the definition in Equation (13.2.3)
/// 2. Via the FFT by padding both inputs to a suitable length
///
/// The program then compares the two results and reports the maximum error.

/// Return the smallest power of two greater than or equal to n.
/// This is a common FFT-friendly choice for the padding length N in Equation (13.2.4).
fn next_power_of_two(n: usize) -> usize {
    n.max(1).next_power_of_two()
}

/// Convert a real slice into a complex vector.
fn real_to_complex(x: &[f64]) -> Vec<Complex64> {
    x.iter().map(|&v| Complex64::new(v, 0.0)).collect()
}

/// Compute the direct linear convolution from Equation (13.2.3).
fn direct_linear_convolution(x: &[f64], h: &[f64]) -> Vec<f64> {
    let nx = x.len();
    let nh = h.len();
    let ny = nx + nh - 1;
    let mut y = vec![0.0; ny];

    for n in 0..ny {
        let mut sum = 0.0;
        for m in 0..nh {
            if n >= m {
                let idx = n - m;
                if idx < nx {
                    sum += h[m] * x[idx];
                }
            }
        }
        y[n] = sum;
    }

    y
}

/// Zero-pad a complex vector to length n.
fn zero_pad(x: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut padded = vec![Complex64::new(0.0, 0.0); n];
    padded[..x.len()].copy_from_slice(x);
    padded
}

/// Compute the forward FFT of a complex vector.
fn fft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer = x.to_vec();
    fft.process(&mut buffer);
    buffer
}

/// Compute the inverse FFT of a complex vector, including the 1/N normalization.
fn ifft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(n);
    let mut buffer = x.to_vec();
    ifft.process(&mut buffer);

    let scale = n as f64;
    for z in &mut buffer {
        *z /= scale;
    }
    buffer
}

/// Compute linear convolution using FFT reduction:
/// 1. Choose N >= Nx + Nh - 1
/// 2. Zero-pad x and h to length N
/// 3. Compute X = DFT(x_tilde), H = DFT(h_tilde)
/// 4. Form Y = X ⊙ H
/// 5. Compute y_tilde = IDFT(Y)
/// 6. Extract the first Nx + Nh - 1 entries
fn fft_linear_convolution(x: &[f64], h: &[f64]) -> Vec<f64> {
    let nx = x.len();
    let nh = h.len();
    let ny = nx + nh - 1;

    let n_fft = next_power_of_two(ny);

    let x_complex = real_to_complex(x);
    let h_complex = real_to_complex(h);

    let x_padded = zero_pad(&x_complex, n_fft);
    let h_padded = zero_pad(&h_complex, n_fft);

    let x_hat = fft(&x_padded);
    let h_hat = fft(&h_padded);

    let y_hat: Vec<Complex64> = x_hat
        .iter()
        .zip(h_hat.iter())
        .map(|(&a, &b)| a * b)
        .collect();

    let y_padded = ifft(&y_hat);

    y_padded[..ny].iter().map(|z| z.re).collect()
}

/// Compute the maximum absolute difference between two real vectors.
fn max_abs_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length.");
    a.iter()
        .zip(b.iter())
        .map(|(&u, &v)| (u - v).abs())
        .fold(0.0_f64, f64::max)
}

/// Print a real vector with indices.
fn print_real_vector(name: &str, x: &[f64]) {
    println!("{name}");
    for (i, &value) in x.iter().enumerate() {
        println!("  [{:2}] = {:>14.10}", i, value);
    }
    println!();
}

fn main() {
    // Example finite signals.
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let h = vec![0.5, -1.0, 2.0];

    let nx = x.len();
    let nh = h.len();
    let ny = nx + nh - 1;
    let n_fft = next_power_of_two(ny);

    println!("Program 13.2.1: Linear Convolution via FFT with Zero Padding\n");

    println!("Signal length Nx = {}", nx);
    println!("Filter length Nh = {}", nh);
    println!("Output length Ny = Nx + Nh - 1 = {}", ny);
    println!("Chosen FFT length N = {}\n", n_fft);

    print_real_vector("Input signal x:", &x);
    print_real_vector("Input filter h:", &h);

    let y_direct = direct_linear_convolution(&x, &h);
    let y_fft = fft_linear_convolution(&x, &h);

    print_real_vector("Direct linear convolution:", &y_direct);
    print_real_vector("FFT-based linear convolution:", &y_fft);

    let err = max_abs_error(&y_direct, &y_fft);
    println!(
        "Maximum absolute error between direct and FFT-based results: {:.3e}",
        err
    );

    if err < 1.0e-10 {
        println!("\nVerification successful: the FFT-based result matches the direct linear convolution.");
    } else {
        println!("\nVerification warning: the discrepancy is larger than expected.");
    }
}
```

Program 13.2.1 illustrates the central computational idea developed in Section 13.2.1: linear convolution can be performed efficiently by embedding it into a circular convolution and evaluating the result using the FFT. The program verifies that the padded FFT computation reproduces the direct evaluation of Equation (13.2.3) while dramatically reducing the asymptotic computational complexity from $O(N_xN_h)$ to approximately $3\,O(N\log N)$, as described in Equation (13.2.8).

The numerical comparison between the direct convolution and the FFT-based result shows agreement up to floating-point roundoff, confirming the correctness of the reduction described in Equations (13.2.5)–(13.2.7). This experiment highlights how the algebraic diagonalization of circulant operators by the Fourier transform translates directly into a practical computational advantage.

Although the example signals in this program are short, the same approach scales effectively to very large problems and forms the basis of many widely used algorithms, including frequency-domain filtering, overlap-add and overlap-save convolution methods, and FFT-accelerated solvers for structured linear systems. The implementation therefore serves as a foundation for the more advanced block-convolution and deconvolution methods discussed later in this section.

## 13.2.2. Boundary Handling, Padding, and Block Convolution

Padding is not merely a technical device to avoid wraparound. It also encodes a model of what happens outside the observed window. In this sense, numerical boundary handling is part of the problem formulation rather than an afterthought.

The most common choice is *zero padding*, which assumes that the signal vanishes outside the observed segment:

$$x_n=0 \qquad \text{for } n\notin{0,\dots,N_x-1} \tag{13.2.10}$$

This is often reasonable when the computational window includes baseline regions or when the signal is genuinely localized. By contrast, *periodic extension* assumes:

$$x_{n+N}=x_n \tag{13.2.11}$$

which is physically meaningful in Fourier pseudo-spectral PDE solvers on periodic domains, but can introduce wraparound artifacts in one-shot signal-processing applications. A third common choice is *reflective or symmetric padding*, which attempts to preserve continuity or smoothness at the boundary. This is often useful in imaging, but it changes the underlying algebra: the resulting operator is no longer strictly circulant, so exact diagonalization by the DFT no longer follows without additional modeling.

For very large signals, it is often impractical to transform the entire input at once. In such cases one uses a block method, computing a sequence of circular convolutions that agree with the desired linear convolution on valid interior regions. Let the filter $h$ have length $M$, and choose an FFT length $N$ with $N\geq M$.

Define,

$$L=N-(M-1) \tag{13.2.12}$$

Then each block transform produces $L$ new valid output samples.

In the *overlap-save* method, the input is partitioned into blocks of length $N$, each containing $M-1$ samples carried over from the previous block and $L$ new samples. For each block $x^{(b)}$, one computes:

$$y^{(b)}=\operatorname{IDFT}\!\left(\operatorname{DFT}\!\left(x^{(b)}\right)\odot \operatorname{DFT}\!\left(\tilde{h}\right)\right) \tag{13.2.13}$$

The first $M-1$ samples of $y^{(b)}$ are discarded, and the remaining $L$ samples are retained as valid. The logic is that the discarded entries are contaminated by circular wraparound, while the retained interior coincides with the linear convolution.

In the *overlap-add* method, one instead partitions the signal into disjoint blocks, pads each block sufficiently, convolves each padded block with (h), and then adds the overlapping output segments. Both viewpoints realize long linear convolutions using repeated FFTs, but they organize the data flow differently. Modern analyses revisit these classical methods from the viewpoint of complexity, distortion, and finite-wordlength effects, and provide updated formulas for choosing the DFT length efficiently in practical implementations (Johansson and Gustafsson, 2023). Related fast-convolution structures also continue to appear in specialized filter-design problems, including variable-bandwidth filtering strategies (Moryakova and Johansson, 2024).

### Rust Implementation

Following the discussion in Section 13.2.2 on block convolution and the role of boundary handling in FFT-based filtering, Program 13.2.2 provides a practical implementation of the overlap-save method for computing long linear convolutions. When signals become large, transforming the entire sequence at once can be computationally inefficient or impractical due to memory constraints. Block convolution methods address this issue by partitioning the input signal into overlapping segments and computing a sequence of circular convolutions whose interior samples coincide with the desired linear convolution. The overlap-save strategy described in Equations (13.2.12) and (13.2.13) accomplishes this by retaining only the valid interior portion of each FFT block while discarding the samples affected by circular wraparound. The program demonstrates how this procedure can be implemented using FFTs while verifying its correctness by comparing the block-based result with the direct linear convolution defined in Equation (13.2.3).

At the core of the implementation are functions that construct the FFT-based block convolution pipeline described by Equation (13.2.13). The functions `fft` and `ifft` compute the forward and inverse discrete Fourier transforms of complex vectors using the `rustfft` library. These transforms correspond to the spectral operations appearing in Equation (13.2.13), where each block of the signal is transformed, multiplied spectrally with the padded filter transform, and then reconstructed in the time domain. The inverse transform includes normalization by the block length to maintain consistency with the transform pair defined earlier in the chapter.

The program also includes the function `direct_linear_convolution`, which evaluates the convolution sum defined in Equation (13.2.3). This direct implementation iterates over the signal and filter indices and accumulates the weighted products that define the Toeplitz-style convolution operation. Although computationally expensive for long signals, this direct computation provides a reliable reference against which the overlap-save result can be validated.

To prepare signals for spectral processing, the implementation includes helper functions such as `real_to_complex`, which converts real-valued data into complex form suitable for the FFT, and `zero_pad_complex`, which extends vectors to a specified length by appending zeros. These steps reflect the zero-padding model described in Equation (13.2.10), where samples outside the observed window are assumed to vanish. Padding is essential for constructing the padded filter vector $\tilde{h}$ used in Equation (13.2.13).

The overlap-save algorithm itself is implemented in the function `overlap_save_convolution`. This function divides the input signal into blocks of length $N$ satisfying the condition $N \ge M$, where $M$ is the filter length. The valid output length per block is determined by Equation (13.2.12), which defines $L = N-(M-1)$. Each block consists of $M-1$ samples carried over from the previous block together with $L$ new samples. After computing the circular convolution using the FFT, the function discards the first $M-1$ samples of each block output, since these entries contain wraparound artifacts produced by circular convolution. The remaining $L$ samples correspond exactly to the desired linear convolution and are appended to the final output vector.

Two additional utility functions assist with program verification and output formatting. The function `max_abs_error` computes the maximum absolute difference between two vectors, allowing the program to quantify the numerical agreement between the overlap-save result and the direct convolution. The function `print_real_vector` prints vectors with index labels to make the intermediate and final results easier to inspect.

The `main` function demonstrates the algorithm on a representative example. It begins by defining a long input signal and a short finite impulse response filter. The program selects an FFT length $N$ satisfying the block-convolution requirements and computes the number of valid output samples per block according to Equation (13.2.12). It then evaluates the convolution using both the direct method and the overlap-save algorithm, prints the resulting vectors, and reports the maximum absolute error between them. This comparison confirms that the overlap-save procedure reproduces the linear convolution while processing the signal in manageable FFT-sized segments.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
rustfft = "6"
```

```rust
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Program 13.2.2
/// Block Convolution Using the Overlap-Save Method
///
/// This program implements FFT-based block convolution for a long real signal
/// and a finite impulse response filter. It follows the overlap-save strategy
/// described in Equations (13.2.12) and (13.2.13):
///
/// 1. Choose an FFT length N >= M, where M is the filter length.
/// 2. Define L = N - (M - 1), the number of valid output samples per block.
/// 3. Form blocks of length N containing M - 1 carried samples and L new samples.
/// 4. Compute the circular convolution of each block with the padded filter.
/// 5. Discard the first M - 1 samples of each block output and retain the next L.
///
/// The implementation also computes the full direct linear convolution for
/// verification and compares the overlap-save output against it.

/// Return true if n is a power of two.
fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Convert a real vector into a complex vector.
fn real_to_complex(x: &[f64]) -> Vec<Complex64> {
    x.iter().map(|&v| Complex64::new(v, 0.0)).collect()
}

/// Zero-pad a complex vector to length n.
fn zero_pad_complex(x: &[Complex64], n: usize) -> Vec<Complex64> {
    assert!(
        x.len() <= n,
        "Input length must not exceed the padding length."
    );
    let mut y = vec![Complex64::new(0.0, 0.0); n];
    y[..x.len()].copy_from_slice(x);
    y
}

/// Compute the forward FFT of a complex vector.
fn fft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer = x.to_vec();
    fft.process(&mut buffer);
    buffer
}

/// Compute the inverse FFT of a complex vector, including normalization by 1/N.
fn ifft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(n);
    let mut buffer = x.to_vec();
    ifft.process(&mut buffer);

    let scale = n as f64;
    for z in &mut buffer {
        *z /= scale;
    }
    buffer
}

/// Direct linear convolution for verification.
/// This implements Equation (13.2.3).
fn direct_linear_convolution(x: &[f64], h: &[f64]) -> Vec<f64> {
    let nx = x.len();
    let nh = h.len();
    let ny = nx + nh - 1;
    let mut y = vec![0.0; ny];

    for n in 0..ny {
        let mut sum = 0.0;
        for m in 0..nh {
            if n >= m {
                let j = n - m;
                if j < nx {
                    sum += h[m] * x[j];
                }
            }
        }
        y[n] = sum;
    }

    y
}

/// Perform overlap-save block convolution.
///
/// x: long input signal
/// h: FIR filter of length M
/// n_fft: FFT length N, which must satisfy N >= M
///
/// Returns the first x.len() + h.len() - 1 samples of the linear convolution.
fn overlap_save_convolution(x: &[f64], h: &[f64], n_fft: usize) -> Vec<f64> {
    let nx = x.len();
    let m = h.len();

    assert!(m >= 1, "Filter length must be positive.");
    assert!(n_fft >= m, "FFT length N must satisfy N >= M.");
    assert!(
        is_power_of_two(n_fft),
        "For this example, choose N as a power of two."
    );

    let l = n_fft - (m - 1);
    assert!(l >= 1, "The valid block length L must be positive.");

    let ny = nx + m - 1;

    // Pad the filter to length N and precompute its FFT.
    let h_complex = real_to_complex(h);
    let h_padded = zero_pad_complex(&h_complex, n_fft);
    let h_hat = fft(&h_padded);

    // Prepend M-1 zeros so that the first block matches the zero-padding model
    // in Equation (13.2.10), and append enough trailing zeros to flush the tail.
    let mut x_extended = vec![0.0; m - 1];
    x_extended.extend_from_slice(x);
    x_extended.extend(std::iter::repeat(0.0).take(l));

    let mut output = Vec::with_capacity(ny);
    let mut start = 0;

    while start + n_fft <= x_extended.len() && output.len() < ny {
        // Current block of length N = (M - 1) overlap + L new samples.
        let block_real = &x_extended[start..start + n_fft];
        let block_complex = real_to_complex(block_real);

        // Equation (13.2.13): y^(b) = IDFT(DFT(x^(b)) ⊙ DFT(h_tilde))
        let x_hat = fft(&block_complex);
        let y_hat: Vec<Complex64> = x_hat
            .iter()
            .zip(h_hat.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        let y_block = ifft(&y_hat);

        // Discard the first M - 1 contaminated samples and retain the next L.
        let valid = &y_block[m - 1..n_fft];
        for z in valid {
            if output.len() < ny {
                output.push(z.re);
            }
        }

        start += l;
    }

    output
}

/// Compute the maximum absolute difference between two real vectors.
fn max_abs_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length.");
    a.iter()
        .zip(b.iter())
        .map(|(&u, &v)| (u - v).abs())
        .fold(0.0_f64, f64::max)
}

/// Print a real vector with indices.
fn print_real_vector(name: &str, x: &[f64]) {
    println!("{name}");
    for (i, &value) in x.iter().enumerate() {
        println!("  [{:2}] = {:>14.10}", i, value);
    }
    println!();
}

fn main() {
    // Example long signal.
    let x = vec![
        1.0, 2.0, 3.0, 4.0, 1.5, -0.5, 2.0, 0.0, 1.0, -1.0, 0.5, 3.0,
    ];

    // Example FIR filter.
    let h = vec![0.25, -0.5, 1.0, 0.75];

    // Choose an FFT length N >= M. Here N = 8.
    let n_fft = 8;
    let m = h.len();
    let l = n_fft - (m - 1);
    let ny = x.len() + h.len() - 1;

    println!("Program 13.2.2: Block Convolution Using the Overlap-Save Method\n");
    println!("Input length Nx = {}", x.len());
    println!("Filter length M = {}", m);
    println!("Chosen FFT length N = {}", n_fft);
    println!("Valid samples per block L = N - (M - 1) = {}", l);
    println!("Output length Ny = Nx + M - 1 = {}\n", ny);

    print_real_vector("Input signal x:", &x);
    print_real_vector("Input filter h:", &h);

    let y_direct = direct_linear_convolution(&x, &h);
    let y_overlap_save = overlap_save_convolution(&x, &h, n_fft);

    print_real_vector("Direct linear convolution:", &y_direct);
    print_real_vector("Overlap-save convolution:", &y_overlap_save);

    let err = max_abs_error(&y_direct, &y_overlap_save);
    println!(
        "Maximum absolute error between direct and overlap-save results: {:.3e}",
        err
    );

    if err < 1.0e-10 {
        println!(
            "\nVerification successful: the overlap-save output matches the direct linear convolution."
        );
    } else {
        println!(
            "\nVerification warning: the discrepancy is larger than expected."
        );
    }
}
```

Program 13.2.2 demonstrates how long linear convolutions can be computed efficiently using block FFT methods. By dividing the signal into overlapping segments and applying the spectral convolution formula of Equation (13.2.13) to each block, the overlap-save method produces the same result as the direct convolution defined in Equation (13.2.3) while avoiding the computational cost of transforming the entire signal at once.

The numerical comparison between the overlap-save output and the direct convolution confirms that the retained interior samples of each block reproduce the correct linear convolution, with differences limited to floating-point roundoff. This agreement illustrates the key principle discussed in Section 13.2.2: circular convolution computed through the FFT can serve as a building block for long linear convolutions when boundary effects are carefully controlled.

In practical applications, block convolution methods such as overlap-save and overlap-add are widely used in digital filtering, audio processing, and large-scale scientific simulations. Their efficiency stems from combining the algebraic diagonalization of convolution operators in the Fourier domain with a streaming data organization that accommodates signals much larger than the transform length. The implementation therefore provides a foundation for scalable FFT-based filtering strategies used in modern computational workflows.

## 13.2.3. Deconvolution as a Regularized Inverse Problem

Convolution is comparatively straightforward because it is a stable forward map. Deconvolution is fundamentally more difficult because the inverse problem is often ill-conditioned. Let $x$ denote an unknown latent signal, $h$ a known response function or point-spread function, and $y$ the observed data. The canonical forward model is:

$$y=h*x+\eta \tag{13.2.14}$$

where $\eta$ denotes additive noise. After padding to a suitable length $N$ and treating the operation as circular convolution, the Fourier coefficients satisfy:

$$Y_k=H_kX_k+E_k,\qquad k=0,\dots,N-1 \tag{13.2.15}$$

where $E_k$ is the Fourier transform of the noise term.

The most direct inversion idea is the *naïve inverse filter*,

$$\widehat{X}_k=\frac{Y_k}{H_k} \tag{13.2.16}$$

Formally this solves the noise-free problem whenever $H_k\neq 0$. Numerically, however, it is unstable when $|H_k|$ is small, since noise is amplified by the factor $1/|H_k|$. This is the essential reason deconvolution requires stabilization through regularization and prior information, particularly in imaging and other ill-posed inverse settings (Xu et al., 2024; Abbasi et al., 2025).

A standard stabilized formulation is Tikhonov-type regularization. One seeks:

$$\widehat{x}=\arg\min_{x}\left(\lVert h\circledast x-y\rVert_{2}^{2}+\lambda \lVert Lx\rVert_{2}^{2}\right),\qquad \lambda>0 \tag{13.2.17}$$

where $L$ is a regularization operator, often the identity or a discrete derivative, and $\lambda$ balances data fit against smoothness or other prior structure. Because circular convolution is diagonal in Fourier space, and because many useful regularizers are also diagonalizable there on periodic grids, the minimizer often decouples mode by mode:

$$\widehat{X}_k=\frac{\overline{H_k}\,Y_k}{|H_k|^2+\lambda |L_k|^2} \tag{13.2.18}$$

Equation (13.2.18) makes the stabilization mechanism explicit. One does not divide by $H_k$ alone; instead, the denominator contains a regularizing floor that limits amplification at frequencies where the forward operator is weak. This is closely related in form to Wiener filtering and is the basic spectral template behind many practical deconvolution methods. Contemporary imaging papers often express their advances precisely in terms of improved priors, improved regularizers, or improved optimization strategies for this ill-posed inverse step (Xu et al., 2024; Li et al., 2024).

The diagonal closed form, however, is not universally available. If the PSF varies spatially, if the boundary conditions are not periodic, or if the regularizer is nonquadratic, such as a sparsity-promoting or total-variation-type penalty, then the problem generally ceases to decouple modewise. In that case the FFT still plays an important role, but now as an accelerator inside an iterative method rather than as a complete closed-form solver. A representative modern direction combines a convolutional forward model, a structured optimizer such as ADMM, and explicit priors or constraints designed to improve robustness under noise and ill-conditioning. Recent structured illumination microscopy work, for example, proposes accelerated linearized ADMM formulations to improve contrast and resolution beyond simpler deconvolution schemes (Xu et al., 2024).

### Rust Implementation

Following the discussion in Section 13.2.3 on deconvolution as a regularized inverse problem, Program 13.2.3 provides a practical implementation of FFT-based deconvolution together with a stabilized reconstruction strategy. In the forward model of Equation (13.2.14), an unknown latent signal is blurred by a known point-spread function and contaminated by noise, producing the observed data. Direct inversion in the frequency domain, as suggested by Equation (13.2.16), is straightforward algebraically but numerically unstable whenever the spectral response of the blur kernel becomes small. The present program illustrates this instability and demonstrates how Tikhonov regularization, expressed spectrally in Equation (13.2.18), mitigates the amplification of noise. Using Rust together with the `rustfft` library, the implementation constructs a synthetic signal, applies a convolutional forward model, introduces controlled noise, and then compares two reconstruction strategies: naive inverse filtering and regularized spectral deconvolution. The resulting experiment highlights the practical considerations involved in recovering signals from ill-conditioned convolutional systems.

At the foundation of the implementation are several utility functions that support FFT-based convolution and spectral manipulation. The functions `real_to_complex`, `zero_pad_complex`, `fft`, and `ifft` together provide the numerical infrastructure needed to implement the spectral relationships described in Equations (13.2.15)–(13.2.18). Real-valued signals are first converted into complex vectors so that they can be processed by the FFT library. The `zero_pad_complex` function extends vectors to a power-of-two length, ensuring that the convolution operation corresponds to a sufficiently long circular convolution consistent with the padding requirement described in Equation (13.2.4). The `fft` and `ifft` functions then compute forward and inverse discrete Fourier transforms using `rustfft`, with the inverse transform explicitly normalized by the vector length so that the transform pair behaves as expected mathematically.

The forward model of Equation (13.2.14) is implemented through the `fft_linear_convolution` function. This function performs linear convolution by embedding the computation into a padded circular convolution, exactly as described in Equations (13.2.5)–(13.2.7). The signals are padded to a length greater than or equal to $N_x + N_h - 1$, transformed to the frequency domain, multiplied componentwise, and then transformed back using the inverse FFT. A secondary function, `direct_linear_convolution`, computes the same operation using the direct summation formula of Equation (13.2.3). This direct version is used only for verification, confirming that the FFT-based convolution reproduces the same result up to floating-point roundoff.

To simulate measurement error, the function `add_deterministic_noise` introduces a small deterministic perturbation into the blurred signal. Rather than relying on random-number generation, this function constructs a mild oscillatory perturbation from sine and cosine components. The purpose is to produce a reproducible noise pattern that illustrates the instability of naive inversion while keeping the example deterministic and easy to reproduce.

The function `naive_inverse_deconvolution` implements the direct spectral inversion formula corresponding to Equation (13.2.16). After transforming both the observed data and the blur kernel to the Fourier domain, the reconstruction is obtained by dividing the spectral coefficients of the data by those of the kernel. Because division by very small spectral values can produce numerical overflow, the implementation includes a small threshold safeguard to avoid division by nearly zero values. This safeguard does not eliminate the fundamental instability of the inverse filter but prevents catastrophic floating-point behavior.

The stabilized reconstruction is implemented in the function `tikhonov_deconvolution_identity`. This function corresponds directly to Equation (13.2.18) with the regularization operator $L$ chosen as the identity. The reconstruction formula therefore becomes a spectral filtering operation in which each Fourier coefficient is multiplied by the factor $\frac{\overline{H_k}}{|H_k|^2+\lambda}$. The parameter $\lambda$ acts as a regularization strength that limits the amplification of noise in spectral modes where $|H_k|$ is small. This modification converts the unstable inverse filter into a well-posed reconstruction that balances fidelity to the data against smoothness of the recovered signal.

Finally, the `main` function orchestrates the entire numerical experiment. It constructs a synthetic latent signal containing several peaks and oscillatory components, defines a short smoothing kernel representing a point-spread function, and generates blurred observations by applying the forward convolution model of Equation (13.2.14). Deterministic noise is added to simulate measurement uncertainty. The program then computes two reconstructions: one using the naive inverse filter and the other using Tikhonov regularization. The functions `rmse` and `max_abs_error` quantify reconstruction accuracy by comparing each estimate with the original latent signal. The printed results illustrate how regularization affects stability and reconstruction quality in the presence of noise.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-complex = "0.4"
rustfft = "6"
```

```rust
use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Program 13.2.3
/// Tikhonov-Regularized FFT Deconvolution
///
/// This program demonstrates deconvolution as a regularized inverse problem.
/// It constructs a latent signal x, blurs it with a short point-spread function h,
/// adds a small deterministic noise term eta, and then attempts to recover x from
/// the observed data y.
///
/// The implementation includes:
/// 1. FFT-based linear convolution for the forward model in Equation (13.2.14)
/// 2. A naive inverse filter corresponding to Equation (13.2.16)
/// 3. A Tikhonov-regularized spectral deconvolution corresponding to Equation (13.2.18)
///
/// For simplicity, the regularization operator L is taken to be the identity,
/// so |L_k|^2 = 1 for every spectral mode.

/// Return the smallest power of two greater than or equal to n.
fn next_power_of_two(n: usize) -> usize {
    n.max(1).next_power_of_two()
}

/// Convert a real slice into a complex vector.
fn real_to_complex(x: &[f64]) -> Vec<Complex64> {
    x.iter().map(|&v| Complex64::new(v, 0.0)).collect()
}

/// Zero-pad a complex vector to length n.
fn zero_pad_complex(x: &[Complex64], n: usize) -> Vec<Complex64> {
    assert!(x.len() <= n, "Input length must not exceed padding length.");
    let mut y = vec![Complex64::new(0.0, 0.0); n];
    y[..x.len()].copy_from_slice(x);
    y
}

/// Forward FFT.
fn fft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer = x.to_vec();
    fft.process(&mut buffer);
    buffer
}

/// Inverse FFT with 1/N normalization.
fn ifft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(n);
    let mut buffer = x.to_vec();
    ifft.process(&mut buffer);

    let scale = n as f64;
    for z in &mut buffer {
        *z /= scale;
    }
    buffer
}

/// Direct linear convolution for verification or comparison.
fn direct_linear_convolution(x: &[f64], h: &[f64]) -> Vec<f64> {
    let nx = x.len();
    let nh = h.len();
    let ny = nx + nh - 1;
    let mut y = vec![0.0; ny];

    for n in 0..ny {
        let mut sum = 0.0;
        for m in 0..nh {
            if n >= m {
                let j = n - m;
                if j < nx {
                    sum += h[m] * x[j];
                }
            }
        }
        y[n] = sum;
    }

    y
}

/// FFT-based linear convolution using zero padding.
fn fft_linear_convolution(x: &[f64], h: &[f64]) -> Vec<f64> {
    let nx = x.len();
    let nh = h.len();
    let ny = nx + nh - 1;
    let n_fft = next_power_of_two(ny);

    let x_padded = zero_pad_complex(&real_to_complex(x), n_fft);
    let h_padded = zero_pad_complex(&real_to_complex(h), n_fft);

    let x_hat = fft(&x_padded);
    let h_hat = fft(&h_padded);

    let y_hat: Vec<Complex64> = x_hat
        .iter()
        .zip(h_hat.iter())
        .map(|(&a, &b)| a * b)
        .collect();

    let y = ifft(&y_hat);
    y[..ny].iter().map(|z| z.re).collect()
}

/// Add deterministic pseudo-noise to a real signal.
/// This avoids requiring a random-number crate while still illustrating instability.
fn add_deterministic_noise(y: &[f64], noise_level: f64) -> Vec<f64> {
    y.iter()
        .enumerate()
        .map(|(i, &v)| {
            let t = i as f64;
            let noise = noise_level * (0.7 * (2.0 * PI * t / 7.0).sin() + 0.3 * (2.0 * PI * t / 5.0).cos());
            v + noise
        })
        .collect()
}

/// Naive inverse filtering in the Fourier domain.
/// This corresponds to Equation (13.2.16), with a tiny safeguard to avoid division by zero.
fn naive_inverse_deconvolution(y: &[f64], h: &[f64], recovered_len: usize) -> Vec<f64> {
    let n_fft = next_power_of_two(y.len().max(h.len() + recovered_len - 1));

    let y_padded = zero_pad_complex(&real_to_complex(y), n_fft);
    let h_padded = zero_pad_complex(&real_to_complex(h), n_fft);

    let y_hat = fft(&y_padded);
    let h_hat = fft(&h_padded);

    let eps = 1.0e-12;
    let x_hat_est: Vec<Complex64> = y_hat
        .iter()
        .zip(h_hat.iter())
        .map(|(&yk, &hk)| {
            if hk.norm() < eps {
                Complex64::new(0.0, 0.0)
            } else {
                yk / hk
            }
        })
        .collect();

    let x_est = ifft(&x_hat_est);
    x_est[..recovered_len].iter().map(|z| z.re).collect()
}

/// Tikhonov-regularized spectral deconvolution with L = I.
/// This corresponds to Equation (13.2.18) with |L_k|^2 = 1.
fn tikhonov_deconvolution_identity(
    y: &[f64],
    h: &[f64],
    lambda: f64,
    recovered_len: usize,
) -> Vec<f64> {
    let n_fft = next_power_of_two(y.len().max(h.len() + recovered_len - 1));

    let y_padded = zero_pad_complex(&real_to_complex(y), n_fft);
    let h_padded = zero_pad_complex(&real_to_complex(h), n_fft);

    let y_hat = fft(&y_padded);
    let h_hat = fft(&h_padded);

    let x_hat_est: Vec<Complex64> = y_hat
        .iter()
        .zip(h_hat.iter())
        .map(|(&yk, &hk)| {
            let denom = hk.norm_sqr() + lambda;
            hk.conj() * yk / denom
        })
        .collect();

    let x_est = ifft(&x_hat_est);
    x_est[..recovered_len].iter().map(|z| z.re).collect()
}

/// Root-mean-square error between two real vectors.
fn rmse(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length.");
    let mse = a
        .iter()
        .zip(b.iter())
        .map(|(&u, &v)| {
            let d = u - v;
            d * d
        })
        .sum::<f64>()
        / (a.len() as f64);
    mse.sqrt()
}

/// Maximum absolute error between two real vectors.
fn max_abs_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length.");
    a.iter()
        .zip(b.iter())
        .map(|(&u, &v)| (u - v).abs())
        .fold(0.0_f64, f64::max)
}

/// Print a real vector with indices.
fn print_real_vector(name: &str, x: &[f64]) {
    println!("{name}");
    for (i, &value) in x.iter().enumerate() {
        println!("  [{:2}] = {:>14.10}", i, value);
    }
    println!();
}

fn main() {
    // Latent signal x: a short structured signal with two peaks and a mild oscillation.
    let x_true = vec![
        0.0, 0.2, 0.8, 1.4, 2.0, 1.5, 0.7, 0.2,
        0.1, 0.4, 1.0, 1.8, 1.2, 0.5, 0.1, 0.0,
    ];

    // Blur kernel h: a short smoothing PSF.
    let h = vec![0.10, 0.20, 0.40, 0.20, 0.10];

    // Forward model y = h * x + eta.
    let y_clean = fft_linear_convolution(&x_true, &h);
    let y_noisy = add_deterministic_noise(&y_clean, 0.02);

    // Deconvolution parameters.
    let lambda = 1.0e-2;

    // Recover only the original latent-signal length.
    let x_naive = naive_inverse_deconvolution(&y_noisy, &h, x_true.len());
    let x_tikh = tikhonov_deconvolution_identity(&y_noisy, &h, lambda, x_true.len());

    println!("Program 13.2.3: Tikhonov-Regularized FFT Deconvolution\n");
    println!("Latent signal length       = {}", x_true.len());
    println!("PSF length                 = {}", h.len());
    println!("Observed-data length       = {}", y_noisy.len());
    println!("Regularization parameter λ = {:.4e}\n", lambda);

    print_real_vector("True latent signal x:", &x_true);
    print_real_vector("Point-spread function h:", &h);
    print_real_vector("Blurred noiseless data h * x:", &y_clean);
    print_real_vector("Observed data y = h * x + eta:", &y_noisy);

    print_real_vector("Naive inverse-filter reconstruction:", &x_naive);
    print_real_vector("Tikhonov-regularized reconstruction:", &x_tikh);

    println!(
        "Naive inverse-filter RMSE:            {:.6e}",
        rmse(&x_true, &x_naive)
    );
    println!(
        "Tikhonov-regularized RMSE:            {:.6e}",
        rmse(&x_true, &x_tikh)
    );
    println!(
        "Naive inverse-filter max abs. error:  {:.6e}",
        max_abs_error(&x_true, &x_naive)
    );
    println!(
        "Tikhonov-regularized max abs. error:  {:.6e}",
        max_abs_error(&x_true, &x_tikh)
    );

    if rmse(&x_true, &x_tikh) < rmse(&x_true, &x_naive) {
        println!("\nThe regularized reconstruction is more stable than the naive inverse filter.");
    } else {
        println!("\nFor this parameter choice, the regularized reconstruction did not improve the RMSE.");
    }

    // Optional consistency check for the forward model implementation.
    let y_direct = direct_linear_convolution(&x_true, &h);
    println!(
        "Forward-model consistency check (direct vs FFT convolution): {:.3e}",
        max_abs_error(&y_clean, &y_direct)
    );
}
```

Program 13.2.3 demonstrates how FFT-based spectral methods transform convolutional inverse problems into algebraically simple frequency-domain computations. The numerical experiment illustrates the instability of naive inverse filtering when noise is present, even when the forward convolution is implemented accurately. The example also shows how Tikhonov regularization modifies the spectral inversion to prevent unbounded amplification of noise in frequency components where the system response is weak.

The comparison between the naive inverse filter and the regularized reconstruction highlights the bias–variance tradeoff inherent in regularization methods. While the naive inverse filter attempts to perfectly invert the blur operator, it can amplify noise dramatically when the spectral response of the point-spread function is small. Regularization mitigates this instability by introducing a stabilizing term in the denominator of Equation (13.2.18), thereby controlling the influence of poorly conditioned spectral modes.

Because the implementation is modular and built upon reusable FFT-based operations, it can easily be extended to more sophisticated inverse formulations. Alternative choices for the regularization operator $L$, frequency-dependent regularization parameters, or iterative optimization frameworks such as ADMM can all be incorporated within the same spectral infrastructure. In practical imaging and signal-processing applications, these extensions form the basis of modern deconvolution techniques that combine convolutional forward models with structured priors and robust optimization strategies.

## 13.2.4. Modern Developments and Practical Imaging Applications

Although fast convolution via FFT is classical, its modern development continues along several active lines. One direction concerns *frequency-domain FIR filtering* itself. Overlap-add and overlap-save are well established, but recent analyses re-examine their complexity and practical competitiveness, especially once finite precision, FFT length selection, and realistic implementation costs are included. Johansson and Gustafsson (2023), in particular, provide updated derivations and complexity estimates that show frequency-domain filtering can remain competitive even at filter lengths shorter than older rules of thumb suggested.

A second direction concerns *low-precision and hardware-aware fast convolution*. On modern accelerators, especially in machine-learning-inspired workloads, reduced precision may be attractive for performance and energy reasons. Yet FFT-based convolution can be numerically sensitive in such settings. Recent work proposes transform variants and correction strategies that aim to preserve accuracy under low-precision arithmetic (He et al., 2024). Although motivated partly by convolutional workloads in modern AI systems, the broader numerical lesson is general: the identity between convolution and pointwise multiplication is exact algebraically, but its floating-point or fixed-point realization can require careful error control (He et al., 2024; Henry, 2024).

A third direction concerns *nonuniform sampling and NUFFT-based extensions.* Standard FFT-based convolution presumes data on a uniform grid. When samples are not uniformly spaced, modern NUFFT methods often rely on convolutional gridding or interpolation steps that map the data onto a uniform mesh before applying a fast transform. Although this lies beyond the strict scope of pure FFT convolution, it reinforces the same conceptual theme: convolution kernels act as computational glue between physical sampling geometries and Fourier-domain acceleration (Nie et al., 2025).

These issues are especially visible in *computational imaging deconvolution*, which provides a concrete use case for the theory above. Many imaging systems can be modeled by:

$$y=h*x+\eta \tag{13.2.19}$$

where $x$ is the true specimen or scene, $h$ is the system point-spread function, $y$ is the measured image, and $\eta$ is noise. In structured illumination microscopy, reconstruction algorithms rely heavily on Fourier-domain operations and deconvolution, while modern methods explicitly address ill-posedness through regularization and accelerated optimization (Xu et al., 2024). The mathematical difficulty is clear: the PSF attenuates fine spatial frequencies, so deconvolution attempts to recover information precisely where the forward model is weakest and noise is most dangerous.

Recent improvements reflect different ways of stabilizing this inverse step. In ultrasound imaging, deconvolution methods that combine sparsity and continuity priors have been used to improve reconstruction quality and image metrics (Li et al., 2024). In optical coherence tomography, recent reviews survey deconvolution strategies aimed at mitigating PSF-induced blur and identify remaining challenges and future directions (Abbasi et al., 2025). Related inverse-filtering and predictive-deconvolution ideas also continue to appear in applied sensing domains such as geophysical survey processing, where sparsity-based regularization plays an important role (Zhang et al., 2024).

FFT-based deconvolution is not a single algorithm but a family of model-based choices. One must specify the physical forward model $h$, the boundary conditions, the noise model, the regularizer or prior, and the computational strategy, whether closed-form or iterative. The FFT supplies the crucial algebraic acceleration, but the success of the overall method depends equally on how well the surrounding modeling assumptions match the application.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/eKJLMDuFZxDGfYlkYhcV.6","tags":[]}

# 13.3. Correlation and Autocorrelation Using the FFT

Correlation is the natural companion to convolution. Whereas convolution models the action of a linear shift-invariant system on a signal, correlation measures similarity between two signals as one is shifted relative to the other. For this reason, convolution is associated primarily with filtering, propagation, and blurring, while correlation is associated with alignment, detection, synchronization, and delay estimation. The FFT is just as valuable here as in convolution, because correlation also admits a spectral representation in which a quadratic-cost sum over all lags is replaced by a small number of transforms and pointwise products. This section develops the algebraic relation between correlation and convolution, shows how FFTs are used for cross-correlation and autocorrelation, and then connects the resulting formulas to practical estimation issues and modern delay-estimation methods.

## 13.3.1. Definitions and the Correlation Theorem

Let $g,h\in\mathbb{C}^N$. The circular cross-correlation of $g$ and $h$ is defined by:

$$r_{gh}[\ell]=\sum_{n=0}^{N-1} g_{(n+\ell)\bmod N}\,\overline{h_n},\qquad \ell=0,1,\dots,N-1 \tag{13.3.1}$$

If the sequences are real-valued, the complex conjugation has no effect, but retaining it is mathematically clean and consistent with standard signal-processing conventions. Two immediate properties follow from the definition. First, cross-correlation is symmetric under exchange of the arguments up to reversal of the lag:

$$r_{gh}[\ell]=\overline{r_{hg}[-\ell]},\qquad \text{indices interpreted modulo } N \tag{13.3.2}$$

Second, autocorrelation is obtained by setting $h=g$:

$$r_{gg}[\ell]=\sum_{n=0}^{N-1} g_{(n+\ell)\bmod N}\,\overline{g_n} \tag{13.3.3}$$

Correlation is closely related to convolution. Define the time-reversal-conjugate sequence $\tilde{h}$ by:

$$\tilde{h}_n=\overline{h_{(-n)\bmod N}}\tag{13.3.4}$$

Then the correlation may be written as a circular convolution,

$$r_{gh}=g\circledast \tilde{h} \tag{13.3.5}$$

This identity is important because it transfers the computational machinery of convolution directly to correlation. Since circular convolution becomes pointwise multiplication in the Fourier domain, one expects the same reduction for cross-correlation, except with conjugation appearing naturally.

Let:

$$G=\operatorname{DFT}(g),\qquad H=\operatorname{DFT}(h) \tag{13.3.6}$$

Then the DFT of the circular cross-correlation satisfies the correlation theorem:

$$\operatorname{DFT}\{r_{gh}\}[k]=G_k\,\overline{H_k},\qquad k=0,1,\dots,N-1\tag{13.3.7}$$

Hence the full circular cross-correlation is recovered by:

$$r_{gh}=\operatorname{IDFT}\!\left(G\odot \overline{H}\right) \tag{13.3.8}$$

Equation (13.3.8) is the computational backbone of FFT-based correlation. It is also the starting point for generalized cross-correlation methods and cross-spectrum phase methods used in modern time-delay estimation under noise and distortion (Uchendu et al., 2025; Sun et al., 2025). The computational advantage is the same as in FFT-based convolution: a direct all-lag computation is quadratic, whereas the FFT formulation reduces the cost to $O(N\log N)$.

## 13.3.2. FFT Computation for Finite, Nonperiodic Data

Although (13.3.8) is exact, it computes circular correlation of length-$N$ vectors. In many applications the observed data are finite segments and are not intended to be periodic. Then, just as in convolution, one must guard against wraparound contamination by padding.

Suppose one wishes to compute the cross-correlation of two finite sequences for lags up to $\pm K$. A practical FFT-based procedure is to choose a transform length $N$ large enough that the desired lag range is not corrupted by circular overlap. In particular, one forms padded vectors $\tilde{g},\tilde{h}\in\mathbb{C}^N,$ by appending at least $K$ zeros, and often more in order to reach an FFT-friendly length. One then computes:

$$G=\operatorname{DFT}(\tilde{g}),\qquad H=\operatorname{DFT}(\tilde{h}) \tag{13.3.9}$$

$$R=G\odot \overline{H} \tag{13.3.10}$$

$$\tilde{r}=\operatorname{IDFT}(R) \tag{13.3.11}$$

The index interpretation must then be handled carefully. The entry corresponding to zero lag is $\tilde{r}_0$, positive lags proceed forward, and negative lags appear near the end of the array because of the modular indexing inherited from the FFT. This bookkeeping is conceptually identical to the padding logic used for linear convolution, except that here it is applied to the correlation theorem rather than the convolution theorem.

The complexity improvement is substantial. Direct computation of correlation for all lags requires:

$$O(N^2) \tag{13.3.12}$$

whereas the FFT-based approach requires,

$$O(N\log N)\tag{13.3.13}$$

This advantage becomes decisive for large signals or for workloads in which the computation must be repeated many times, such as sliding-window processing, large sensor arrays, or repeated pairwise correlation tasks. Modern time-delay estimation methods continue to use FFT-based correlation as the baseline computational primitive precisely because of this scaling advantage (Sun et al., 2025).

### Rust Implementation

Following the discussion in Section 13.3.2 on FFT-based computation of cross-correlation for finite, nonperiodic data, Program 13.3.1 provides a practical implementation of correlation and autocorrelation using the spectral identity developed in equations (13.3.8)–(13.3.11). In direct form, the correlation definition in equation (13.3.1) requires evaluating a sum for each possible lag, leading to quadratic computational cost as described in equation (13.3.12). The FFT formulation replaces this expensive procedure with a sequence of transforms and pointwise spectral products whose cost scales as $O(N\log N)$, as indicated in equation (13.3.13). The program implements this strategy by padding finite signals to avoid circular wraparound, computing forward FFTs of the padded vectors, forming the cross-spectrum $G \odot \overline{H}$, and then applying the inverse transform to recover the correlation sequence. The implementation also demonstrates autocorrelation as the special case $h=g$, consistent with the spectral identity in equation (13.3.14). To verify correctness, the program compares the FFT-based result with a direct $O(N^2)$ computation and reports the numerical agreement.

At the center of the implementation is the function `cross_correlation_fft`, which realizes the FFT correlation identity introduced in equations (13.3.9)–(13.3.11). The function begins by determining an FFT length large enough to accommodate the linear correlation of the two finite sequences without circular overlap. This length is computed using the helper function `next_power_of_two_ge`, which selects a transform size suitable for efficient FFT computation. The input sequences are then padded with zeros using the function `zero_pad`, producing the vectors $\tilde g$ and $\tilde h$ described in equation (13.3.9). After padding, forward FFTs are computed for both vectors, yielding the spectra $G$ and $H$. The cross-spectrum is then formed through elementwise multiplication with complex conjugation, implementing equation (13.3.10). Finally, the inverse FFT recovers the correlation sequence $\tilde r$ in accordance with equation (13.3.11). Because FFT indexing is inherently circular, the function performs a final step that reorders the results into signed lag order so that the correlation values correspond directly to the lag interpretation of equation (13.3.1).

The auxiliary function `autocorrelation_fft` implements the special case of autocorrelation discussed in Section 13.3.3. By calling `cross_correlation_fft` with identical input sequences, the program effectively computes the spectral product $G\odot \overline{G}$, which corresponds to the squared magnitude spectrum appearing in equation (13.3.14). This demonstrates how autocorrelation arises naturally as a particular instance of the more general cross-correlation framework.

Several support functions implement important numerical details of the FFT workflow. The function `fft_in_place` applies either the forward or inverse FFT using the `rustfft` library. When computing the inverse transform, the function scales the result by $1/N$ so that the operation corresponds to the standard inverse DFT normalization. The helper function `next_power_of_two_ge` determines an efficient transform size by selecting the smallest power of two greater than or equal to the required length, which is a common performance optimization for FFT algorithms. The function `zero_pad` constructs padded vectors by extending the original signal with zeros, thereby preventing wraparound artifacts in the computed correlation.

For validation purposes, the program also includes the function `cross_correlation_direct`, which evaluates the defining sum of equation (13.3.1) directly using a double loop. Although this method has quadratic complexity, it provides a reliable reference result against which the FFT-based computation can be compared. The program measures the maximum difference between the two methods to demonstrate that the spectral algorithm reproduces the exact correlation sequence up to floating-point rounding error.

The `main` function demonstrates the algorithm using two examples. The first example uses real-valued sequences where one signal is embedded within another with a shift. The FFT-based correlation identifies the alignment lag by locating the maximum correlation value. The program prints the correlation sequence, verifies agreement with the direct computation, and reports the maximum numerical difference between the two approaches. The second example uses complex-valued sequences to illustrate that the conjugation appearing in equations (13.3.1) and (13.3.7) is essential for correctly computing correlation in complex arithmetic. Together, these demonstrations confirm both the correctness and the computational efficiency of the FFT-based method.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rustfft = "6"
```

```rust
// Program 13.3.1
// FFT-Based Cross-Correlation and Autocorrelation for Finite, Nonperiodic Data
//
// Cargo.toml dependencies:
//
// [dependencies]
// rustfft = "6"
//
// This program implements equations (13.3.9) to (13.3.11) for padded,
// finite-data cross-correlation, and equation (13.3.14) for autocorrelation.

use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;

type C64 = Complex<f64>;

/// Returns the smallest power of two greater than or equal to `n`.
fn next_power_of_two_ge(n: usize) -> usize {
    n.max(1).next_power_of_two()
}

/// Computes the FFT or inverse FFT of `data` in place.
/// When `inverse` is true, the result is scaled by 1/N so that the transform
/// matches the standard inverse DFT normalization.
fn fft_in_place(data: &mut [C64], inverse: bool) {
    let n = data.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    };
    fft.process(data);

    if inverse {
        let scale = 1.0 / n as f64;
        for x in data.iter_mut() {
            *x *= scale;
        }
    }
}

/// Pads `x` with zeros up to length `n`.
fn zero_pad(x: &[C64], n: usize) -> Vec<C64> {
    let mut y = vec![C64::zero(); n];
    y[..x.len()].copy_from_slice(x);
    y
}

/// Computes the full aperiodic cross-correlation of finite sequences g and h
/// using the FFT identity
///
///     r = IDFT( FFT(g_padded) ⊙ conj(FFT(h_padded)) ).
///
/// The returned vector is ordered by signed lag from -(h.len()-1) to g.len()-1.
/// Each entry is a pair (lag, value).
///
/// This corresponds to the definition
///
///     r_gh[ell] = sum_n g[n + ell] * conj(h[n]),
///
/// restricted to indices where both samples exist.
fn cross_correlation_fft(g: &[C64], h: &[C64]) -> Vec<(isize, C64)> {
    assert!(!g.is_empty(), "g must not be empty");
    assert!(!h.is_empty(), "h must not be empty");

    let linear_len = g.len() + h.len() - 1;
    let fft_len = next_power_of_two_ge(linear_len);

    // Step 1: zero-pad to avoid wraparound contamination.
    let mut gp = zero_pad(g, fft_len);
    let mut hp = zero_pad(h, fft_len);

    // Step 2: forward FFTs.
    fft_in_place(&mut gp, false);
    fft_in_place(&mut hp, false);

    // Step 3: pointwise multiply by the conjugated second spectrum.
    let mut spectrum = vec![C64::zero(); fft_len];
    for k in 0..fft_len {
        spectrum[k] = gp[k] * hp[k].conj();
    }

    // Step 4: inverse FFT.
    fft_in_place(&mut spectrum, true);

    // The FFT output is in modular lag order:
    //   lag 0, 1, 2, ..., g.len()-1, ..., negative lags near the end.
    //
    // For full aperiodic correlation, valid lags are:
    //   -(h.len()-1), ..., -1, 0, 1, ..., g.len()-1.
    //
    // Negative lag `-j` is stored at index fft_len - j.
    let mut result = Vec::with_capacity(linear_len);

    for j in (1..h.len()).rev() {
        let idx = fft_len - j;
        let lag = -(j as isize);
        result.push((lag, spectrum[idx]));
    }

    for i in 0..g.len() {
        let lag = i as isize;
        result.push((lag, spectrum[i]));
    }

    result
}

/// Computes the full aperiodic autocorrelation using the same FFT framework.
/// This is the special case h = g, consistent with equation (13.3.14) after
/// padding for finite, nonperiodic data.
fn autocorrelation_fft(g: &[C64]) -> Vec<(isize, C64)> {
    cross_correlation_fft(g, g)
}

/// Direct O(N^2) cross-correlation for verification.
///
/// This uses the same lag convention as `cross_correlation_fft`.
fn cross_correlation_direct(g: &[C64], h: &[C64]) -> Vec<(isize, C64)> {
    let min_lag = -((h.len() as isize) - 1);
    let max_lag = (g.len() as isize) - 1;
    let mut out = Vec::new();

    for lag in min_lag..=max_lag {
        let mut sum = C64::zero();

        for n in 0..h.len() {
            let gi = n as isize + lag;
            if gi >= 0 && (gi as usize) < g.len() {
                sum += g[gi as usize] * h[n].conj();
            }
        }

        out.push((lag, sum));
    }

    out
}

/// Finds the lag at which the magnitude of the correlation is maximal.
fn peak_lag(corr: &[(isize, C64)]) -> (isize, C64) {
    corr.iter()
        .copied()
        .max_by(|a, b| {
            a.1.norm()
                .partial_cmp(&b.1.norm())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("correlation sequence must not be empty")
}

/// Pretty-prints a correlation sequence.
fn print_correlation(title: &str, corr: &[(isize, C64)]) {
    println!("{title}");
    for (lag, value) in corr {
        println!(
            "lag {lag:>3}: {:>12.6} + {:>12.6}i   |r| = {:>12.6}",
            value.re,
            value.im,
            value.norm()
        );
    }
    println!();
}

fn main() {
    // Example 1:
    // A real-valued signal h is embedded in g with a visible shift.
    // The cross-correlation peak should identify the alignment lag.
    let g_real = vec![0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
    let h_real = vec![1.0, 2.0, 3.0, 2.0, 1.0];

    let g: Vec<C64> = g_real.into_iter().map(|x| C64::new(x, 0.0)).collect();
    let h: Vec<C64> = h_real.into_iter().map(|x| C64::new(x, 0.0)).collect();

    let corr_fft = cross_correlation_fft(&g, &h);
    let corr_direct = cross_correlation_direct(&g, &h);
    let auto_fft = autocorrelation_fft(&h);

    print_correlation("FFT-based cross-correlation r_gh[lag]:", &corr_fft);
    print_correlation("Direct cross-correlation r_gh[lag]:", &corr_direct);
    print_correlation("FFT-based autocorrelation r_hh[lag]:", &auto_fft);

    let (best_lag, best_value) = peak_lag(&corr_fft);
    println!(
        "Maximum cross-correlation occurs at lag {} with value {:.6} + {:.6}i (|r| = {:.6}).",
        best_lag,
        best_value.re,
        best_value.im,
        best_value.norm()
    );
    println!();

    // Numerical verification of the FFT result against the direct computation.
    let max_error = corr_fft
        .iter()
        .zip(corr_direct.iter())
        .map(|((lag1, a), (lag2, b))| {
            assert_eq!(lag1, lag2, "lag mismatch between FFT and direct results");
            (*a - *b).norm()
        })
        .fold(0.0_f64, f64::max);

    println!(
        "Maximum absolute difference between FFT and direct correlation = {:.6e}",
        max_error
    );

    // Example 2:
    // A genuinely complex-valued example, illustrating that conjugation matters.
    let gc = vec![
        C64::new(1.0, 1.0),
        C64::new(2.0, -1.0),
        C64::new(0.0, 0.5),
        C64::new(-1.0, 2.0),
    ];
    let hc = vec![
        C64::new(1.0, 0.0),
        C64::new(-1.0, 1.0),
        C64::new(0.5, -0.5),
    ];

    let corr_complex = cross_correlation_fft(&gc, &hc);
    print_correlation(
        "FFT-based cross-correlation for a complex-valued example:",
        &corr_complex,
    );
}
```

Program 13.3.1 demonstrates how the theoretical identity of the correlation theorem can be translated directly into an efficient computational algorithm. By replacing the quadratic-cost evaluation of equation (13.3.1) with the spectral formulation of equations (13.3.8)–(13.3.11), the program reduces the computational complexity from $O(N^2)$ to $O(N\log N)$. This improvement becomes particularly important for long signals or for applications that require repeated correlation computations, such as sliding-window analysis, large sensor arrays, or delay estimation in monitoring systems.

The numerical experiments included in the program confirm the correctness of the FFT-based approach. The correlation sequence produced by the spectral algorithm agrees with the direct summation result to within machine precision, demonstrating that the transform-based formulation preserves the exact algebraic relationship between the signals. The example involving autocorrelation also illustrates the spectral interpretation given in equation (13.3.14), where the correlation sequence is obtained from the inverse transform of the power spectrum.

Beyond serving as a verification of the theoretical identities developed in this section, the program highlights the role of FFT-based correlation as a fundamental computational primitive in modern signal processing. The same algorithmic structure forms the basis for more advanced techniques such as normalized correlation, generalized cross-correlation, and cross-spectrum phase methods used in delay estimation under noisy conditions. Consequently, FFT-based correlation provides both an efficient numerical tool and a conceptual bridge between harmonic analysis and practical inference problems in sensing, synchronization, and system monitoring.

## 13.3.3. Autocorrelation, Power Spectra, and Estimation Details

Autocorrelation is the special case $h=g$. Substituting $H=G$ into (13.3.8) gives:

$$
r_{gg}=\operatorname{IDFT}\!\left(G\odot \overline{G}\right)
      =\operatorname{IDFT}\!\left(|G|^2\right) \tag{13.3.14}
$$

Thus a single forward FFT and a single inverse FFT are sufficient to compute the full circular autocorrelation. Equation (13.3.14) also makes explicit the close relationship between autocorrelation and the power spectrum, since $|G|^2$ is precisely the discrete power spectrum of the signal. This is why second-order spectral analysis and autocorrelation analysis are so closely linked, and why modern work on nonuniform sampling often replaces the FFT steps in such pipelines with NUFFT-based alternatives (Cui et al., 2025).

For clarity in numerical computing contexts, it is useful to distinguish the algebraic correlation of finite vectors from the statistical estimation of correlation for stochastic processes. Although the mathematical forms are closely related, their interpretations and practical considerations differ in important ways. Several implementation aspects therefore deserve attention.

The first is *mean removal*. If the signals have significant DC components, these can dominate the correlation and obscure the lag-dependent structure. One therefore often forms zero-mean versions:

$$g_n' = g_n - \bar{g}, \qquad h_n' = h_n - \bar{h}  \tag{13.3.15}$$

before computing correlation. This prevents the zero-frequency contribution from overwhelming the correlation peak.

The second is *normalization*. If one wants a scale-free similarity score, particularly for real data, a common choice is normalized cross-correlation,

$$\rho_{gh}[\ell]=\frac{r_{g'h'}[\ell]}{\sqrt{r_{g'g'}[0]\,r_{h'h'}[0]}} \tag{13.3.16}$$

This produces values in $[-1,1]$ in the real-valued case and allows comparisons across signals with different amplitudes.

The third is the distinction between *biased* and *unbiased* finite-window estimators. The FFT naturally computes the unnormalized sum appearing in the algebraic correlation formula. Depending on the statistical model, one may afterward divide either by $N$ or by $N-|\ell|$:

$$\widehat{r}_{\text{biased}}[\ell]=\frac{1}{N}\,r[\ell] \tag{13.3.17}$$

$$\widehat{r}_{\text{unbiased}}[\ell]=\frac{1}{N-|\ell|}\,r[\ell] \tag{13.3.18}$$

These are not different FFT algorithms, but different postprocessing choices based on the intended estimator. The biased version has lower variance, while the unbiased version may be preferable in some statistical settings.

A fourth practical issue is *windowing and leakage*. FFT-based correlation inherits the same leakage phenomena that arise in Fourier analysis more generally. If correlations are later interpreted as delay estimates or spectral signatures, then finite observation windows and nonperiodic truncation can influence the apparent peaks and phase relationships, so care is required in interpretation (Henry, 2024).

### Rust Implementation

Following the discussion in Section 13.3.3 on the relationship between autocorrelation and the power spectrum, Program 13.3.2 provides a practical implementation of FFT-based autocorrelation together with several estimation refinements used in numerical signal analysis. The theoretical identity in equation (13.3.14) shows that autocorrelation can be computed from the inverse transform of the squared magnitude spectrum $|G|^2$, allowing the correlation sequence to be obtained using a single forward FFT and a single inverse FFT. In practice, however, correlation is rarely used in this purely algebraic form. Real data often contain large DC components, differing amplitudes, and finite observation windows that require additional preprocessing and postprocessing. This program demonstrates how these issues are addressed computationally by incorporating mean removal, normalized correlation, and alternative finite-sample estimators. It also illustrates the direct connection between the power spectrum and autocorrelation by explicitly computing both quantities within the same FFT pipeline. Together, these components show how the theoretical correlation identities developed earlier in the section translate into practical estimation procedures for real data.

At the center of the implementation is the function `cross_correlation_fft_real`, which computes the full aperiodic cross-correlation of two real signals using the FFT-based identity introduced earlier in equations (13.3.9)–(13.3.11). The function first determines an FFT length large enough to prevent wraparound contamination and then pads the signals accordingly. After computing forward FFTs of the padded vectors, the cross-spectrum is formed through elementwise multiplication with complex conjugation, which corresponds directly to the spectral product appearing in equation (13.3.10). Applying the inverse FFT then produces the correlation sequence in accordance with equation (13.3.11). Because FFT indexing is circular, the function reorders the output so that the values are reported in signed-lag order consistent with the interpretation of equation (13.3.1).

The function `autocorrelation_fft_real` implements the special case $h=g$ discussed in Section 13.3.3. By computing the correlation of a signal with itself, the program effectively evaluates the spectral expression $G\odot\overline{G}$, which is equivalent to the squared magnitude spectrum $|G|^2$ appearing in equation (13.3.14). This highlights the close connection between autocorrelation and power spectral analysis. To make this relationship explicit, the function `circular_autocorrelation_fft_real` computes the circular autocorrelation exactly as described in equation (13.3.14), while the function `power_spectrum_real` returns the discrete power spectrum $|G|^2$. Examining these results side by side illustrates how the autocorrelation sequence is obtained from the inverse transform of the power spectrum.

The program also implements several practical refinements used in statistical correlation estimation. The function `mean_remove` applies the preprocessing step described in equation (13.3.15), subtracting the sample mean from each signal to eliminate large DC components that might otherwise dominate the correlation. After this preprocessing, the function `normalized_cross_correlation_real` computes the scale-independent similarity measure defined in equation (13.3.16). This normalization divides the correlation sequence by the square root of the zero-lag autocorrelations of the mean-removed signals, producing values that lie in the interval (\[-1,1\]) for real-valued data.

Additional functions implement the two common finite-sample estimators discussed in the section. The function `biased_estimator` divides the correlation sequence by the sample length $N$, implementing the estimator described in equation (13.3.17). In contrast, the function `unbiased_estimator` divides each lag value by $N-|\ell|$, corresponding to the estimator defined in equation (13.3.18). These estimators differ only in postprocessing and do not change the FFT-based correlation algorithm itself.

Several supporting functions implement important numerical details of the FFT workflow. The function `fft_in_place` performs forward and inverse FFT operations using the `rustfft` library and applies the appropriate normalization when computing the inverse transform. The function `next_power_of_two_ge` determines an efficient FFT length, while `zero_pad_real` constructs padded complex arrays from real-valued signals. The functions `hann_window` and `apply_window` illustrate the use of windowing to reduce spectral leakage, which is one of the practical concerns discussed at the end of Section 13.3.3.

The `main` function demonstrates the full estimation pipeline through several examples. The first example illustrates the influence of DC components by comparing raw cross-correlation with normalized correlation after mean removal. The second example computes the autocorrelation of a signal and applies both biased and unbiased estimators, showing how the normalization factors affect the resulting values. The third example computes the circular autocorrelation and power spectrum, illustrating the spectral identity of equation (13.3.14). Finally, the program applies a Hann window before computing autocorrelation to demonstrate how windowing can alter the resulting correlation structure by reducing leakage effects.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rustfft = "6"
```

```rust
// Program 13.3.2
// Autocorrelation, Power Spectra, and Estimation Details
//
// Cargo.toml dependencies:
//
// [dependencies]
// rustfft = "6"
//
// This program demonstrates:
// 1. FFT-based autocorrelation via r_gg = IDFT(|G|^2)          [Eq. (13.3.14)]
// 2. Mean removal before correlation                           [Eq. (13.3.15)]
// 3. Normalized cross-correlation                             [Eq. (13.3.16)]
// 4. Biased and unbiased finite-window estimators             [Eqs. (13.3.17)-(13.3.18)]
// 5. The link between autocorrelation and the discrete power spectrum
//
// The implementation uses real-valued input data for clarity, but performs
// the FFT in complex arithmetic internally.

use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;

type C64 = Complex<f64>;

#[derive(Clone, Copy, Debug)]
struct LagValue {
    lag: isize,
    value: C64,
}

fn next_power_of_two_ge(n: usize) -> usize {
    n.max(1).next_power_of_two()
}

fn fft_in_place(data: &mut [C64], inverse: bool) {
    let n = data.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    };
    fft.process(data);

    if inverse {
        let scale = 1.0 / n as f64;
        for x in data.iter_mut() {
            *x *= scale;
        }
    }
}

fn zero_pad_real(x: &[f64], n: usize) -> Vec<C64> {
    let mut out = vec![C64::zero(); n];
    for (i, &v) in x.iter().enumerate() {
        out[i] = C64::new(v, 0.0);
    }
    out
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn mean_remove(x: &[f64]) -> Vec<f64> {
    let mu = mean(x);
    x.iter().map(|&v| v - mu).collect()
}

fn hann_window(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    (0..n)
        .map(|i| 0.5 - 0.5 * (two_pi * i as f64 / (n as f64 - 1.0)).cos())
        .collect()
}

fn apply_window(x: &[f64], w: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), w.len(), "signal and window lengths must match");
    x.iter().zip(w.iter()).map(|(&a, &b)| a * b).collect()
}

/// Full aperiodic cross-correlation for real signals using FFT padding.
/// Returns lags from -(h.len()-1) to g.len()-1.
fn cross_correlation_fft_real(g: &[f64], h: &[f64]) -> Vec<LagValue> {
    assert!(!g.is_empty(), "g must not be empty");
    assert!(!h.is_empty(), "h must not be empty");

    let linear_len = g.len() + h.len() - 1;
    let fft_len = next_power_of_two_ge(linear_len);

    let mut gp = zero_pad_real(g, fft_len);
    let mut hp = zero_pad_real(h, fft_len);

    fft_in_place(&mut gp, false);
    fft_in_place(&mut hp, false);

    let mut spectrum = vec![C64::zero(); fft_len];
    for k in 0..fft_len {
        spectrum[k] = gp[k] * hp[k].conj();
    }

    fft_in_place(&mut spectrum, true);

    let mut result = Vec::with_capacity(linear_len);

    for j in (1..h.len()).rev() {
        let idx = fft_len - j;
        result.push(LagValue {
            lag: -(j as isize),
            value: spectrum[idx],
        });
    }

    for i in 0..g.len() {
        result.push(LagValue {
            lag: i as isize,
            value: spectrum[i],
        });
    }

    result
}

/// Full aperiodic autocorrelation for a real signal.
fn autocorrelation_fft_real(x: &[f64]) -> Vec<LagValue> {
    cross_correlation_fft_real(x, x)
}

/// Circular autocorrelation computed exactly as IDFT(|G|^2), matching Eq. (13.3.14).
/// The result has lags 0, 1, ..., n-1 in circular order.
fn circular_autocorrelation_fft_real(x: &[f64]) -> Vec<C64> {
    let n = x.len();
    let mut data: Vec<C64> = x.iter().map(|&v| C64::new(v, 0.0)).collect();

    fft_in_place(&mut data, false);

    let mut power = vec![C64::zero(); n];
    for k in 0..n {
        power[k] = C64::new(data[k].norm_sqr(), 0.0);
    }

    fft_in_place(&mut power, true);
    power
}

/// Discrete power spectrum |G|^2 corresponding to Eq. (13.3.14).
fn power_spectrum_real(x: &[f64]) -> Vec<f64> {
    let mut data: Vec<C64> = x.iter().map(|&v| C64::new(v, 0.0)).collect();
    fft_in_place(&mut data, false);
    data.iter().map(|z| z.norm_sqr()).collect()
}

/// Normalized cross-correlation using mean-removed signals, matching Eq. (13.3.16).
/// The denominator uses the zero-lag autocorrelation of the mean-removed signals.
fn normalized_cross_correlation_real(g: &[f64], h: &[f64]) -> Vec<LagValue> {
    let gp = mean_remove(g);
    let hp = mean_remove(h);

    let corr = cross_correlation_fft_real(&gp, &hp);

    let eg = gp.iter().map(|v| v * v).sum::<f64>();
    let eh = hp.iter().map(|v| v * v).sum::<f64>();
    let denom = (eg * eh).sqrt();

    if denom == 0.0 {
        return corr
            .into_iter()
            .map(|entry| LagValue {
                lag: entry.lag,
                value: C64::zero(),
            })
            .collect();
    }

    corr.into_iter()
        .map(|entry| LagValue {
            lag: entry.lag,
            value: entry.value / denom,
        })
        .collect()
}

fn biased_estimator(corr: &[LagValue], n: usize) -> Vec<LagValue> {
    let scale = 1.0 / n as f64;
    corr.iter()
        .map(|entry| LagValue {
            lag: entry.lag,
            value: entry.value * scale,
        })
        .collect()
}

fn unbiased_estimator(corr: &[LagValue], n: usize) -> Vec<LagValue> {
    corr.iter()
        .map(|entry| {
            let denom = (n as isize - entry.lag.abs()) as f64;
            LagValue {
                lag: entry.lag,
                value: entry.value / denom,
            }
        })
        .collect()
}

fn zero_lag_value(corr: &[LagValue]) -> Option<C64> {
    corr.iter().find(|entry| entry.lag == 0).map(|entry| entry.value)
}

fn peak_lag_real(corr: &[LagValue]) -> Option<LagValue> {
    corr.iter().copied().max_by(|a, b| {
        a.value
            .re
            .partial_cmp(&b.value.re)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn print_selected_lags(title: &str, corr: &[LagValue], lags: &[isize]) {
    println!("{title}");
    for &lag in lags {
        if let Some(entry) = corr.iter().find(|e| e.lag == lag) {
            println!(
                "lag {lag:>3}: {:>12.6} + {:>12.6}i",
                entry.value.re, entry.value.im
            );
        }
    }
    println!();
}

fn print_vector_real(title: &str, x: &[f64], max_items: usize) {
    println!("{title}");
    for (i, v) in x.iter().take(max_items).enumerate() {
        println!("index {i:>3}: {:>12.6}", v);
    }
    if x.len() > max_items {
        println!("...");
    }
    println!();
}

fn print_vector_complex(title: &str, x: &[C64], max_items: usize) {
    println!("{title}");
    for (i, v) in x.iter().take(max_items).enumerate() {
        println!("index {i:>3}: {:>12.6} + {:>12.6}i", v.re, v.im);
    }
    if x.len() > max_items {
        println!("...");
    }
    println!();
}

fn main() {
    // Example 1:
    // A signal with a strong DC offset. This illustrates why mean removal
    // matters before using correlation for alignment.
    let g = vec![5.0, 5.0, 6.0, 7.0, 8.0, 7.0, 6.0, 5.0];
    let h = vec![5.0, 6.0, 7.0, 8.0, 7.0];

    let raw_corr = cross_correlation_fft_real(&g, &h);
    let norm_corr = normalized_cross_correlation_real(&g, &h);

    let selected_lags = vec![-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7];
    print_selected_lags(
        "Raw cross-correlation with DC present:",
        &raw_corr,
        &selected_lags,
    );
    print_selected_lags(
        "Normalized cross-correlation after mean removal:",
        &norm_corr,
        &selected_lags,
    );

    if let Some(peak) = peak_lag_real(&raw_corr) {
        println!(
            "Peak lag for raw cross-correlation: {} with value {:.6}",
            peak.lag, peak.value.re
        );
    }
    if let Some(peak) = peak_lag_real(&norm_corr) {
        println!(
            "Peak lag for normalized cross-correlation: {} with value {:.6}",
            peak.lag, peak.value.re
        );
    }
    println!();

    // Example 2:
    // Autocorrelation and estimator postprocessing.
    let x = vec![1.0, 2.0, 3.0, 2.0, 1.0];
    let auto = autocorrelation_fft_real(&x);
    let biased = biased_estimator(&auto, x.len());
    let unbiased = unbiased_estimator(&auto, x.len());

    let lags_auto = vec![-4, -3, -2, -1, 0, 1, 2, 3, 4];
    print_selected_lags("Unnormalized autocorrelation:", &auto, &lags_auto);
    print_selected_lags("Biased autocorrelation estimator:", &biased, &lags_auto);
    print_selected_lags("Unbiased autocorrelation estimator:", &unbiased, &lags_auto);

    if let Some(r0) = zero_lag_value(&auto) {
        println!("Zero-lag autocorrelation r_xx[0] = {:.6}", r0.re);
    }
    println!();

    // Example 3:
    // Circular autocorrelation and power spectrum via Eq. (13.3.14).
    let y = vec![1.0, -1.0, 2.0, 0.0, 1.0, -2.0, 0.5, 1.5];
    let circ_auto = circular_autocorrelation_fft_real(&y);
    let power = power_spectrum_real(&y);

    print_vector_complex(
        "Circular autocorrelation computed as IDFT(|G|^2):",
        &circ_auto,
        8,
    );
    print_vector_real("Discrete power spectrum |G|^2:", &power, 8);

    // Example 4:
    // A short demonstration of windowing before autocorrelation.
    let window = hann_window(y.len());
    let y_windowed = apply_window(&y, &window);
    let auto_windowed = autocorrelation_fft_real(&y_windowed);

    print_vector_real("Hann window:", &window, 8);
    print_selected_lags(
        "Autocorrelation after Hann windowing:",
        &auto_windowed,
        &[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
    );
}
```

Program 13.3.2 demonstrates how the theoretical identities linking autocorrelation and the power spectrum translate into practical computational procedures. The FFT-based method provides an efficient mechanism for evaluating correlation sequences, while additional preprocessing and normalization steps adapt the basic algorithm to the statistical characteristics of real data. In particular, mean removal prevents large DC components from dominating the correlation, normalized cross-correlation provides a scale-independent similarity measure, and the biased and unbiased estimators illustrate how different statistical assumptions lead to different normalization choices.

The examples in the program illustrate several important practical effects. The comparison between raw and normalized correlation highlights how strong DC components can obscure the underlying lag structure of the signals. The autocorrelation examples confirm that the maximum value occurs at zero lag and demonstrate the impact of different estimator normalizations. The comparison between circular autocorrelation and the discrete power spectrum directly illustrates the identity in equation (13.3.14), emphasizing the close relationship between correlation analysis and spectral analysis.

The modular structure of the implementation makes it straightforward to extend this framework to more advanced techniques. For example, generalized cross-correlation methods discussed in the following section modify the cross-spectrum before inversion, while modern delay estimation techniques often incorporate adaptive frequency weighting or nonuniform FFT methods. In this sense, the FFT-based correlation algorithm serves not only as an efficient computational primitive but also as a foundation for a broad class of modern signal processing methods.

## 13.3.4. Modern Delay Estimation and a Practical Use Case

The FFT identity,

$$r_{gh}=\operatorname{IDFT}\!\left(G\odot \overline{H}\right) \tag{13.3.19}$$

is also the foundation for a family of modern methods that modify the cross-spectrum before applying the inverse transform. In generalized cross-correlation, one introduces a frequency-dependent weighting $W_k$ and computes:

$$r_{gh}^{(W)}=\operatorname{IDFT}\!\left(W\odot(G\odot\overline{H})\right)\tag{13.3.20}$$

Here the weighting is chosen to suppress noise, reduce the influence of unreliable frequency bands, or emphasize robust phase information. This weight-and-invert pattern appears frequently in modern low-SNR delay estimation methods (Sun et al., 2025; Liu et al., 2025).

A closely related perspective is obtained from the phase of the cross-spectrum itself. In an idealized two-sensor model, suppose that one signal is approximately a delayed version of the other:

$$h(t)\approx g(t-\tau) \tag{13.3.21}$$

Then, in the frequency domain, one has approximately,

$$H(\omega)\approx e^{-i\omega\tau}G(\omega) \tag{13.3.22}$$

so the cross-spectrum becomes,

$$G(\omega)\,\overline{H(\omega)} \approx |G(\omega)|^2 e^{i\omega\tau} \tag{13.3.23}$$

Hence the phase is approximately linear in (\\omega), with slope (\\tau). This observation motivates cross-spectral phase methods, in which the time delay is estimated not from the location of a peak in the correlation sequence, but from a linear fit to the phase of the cross-spectrum over frequency bands judged reliable. Recent applied acoustics work uses precisely this strategy for robust delay estimation in practical leak-detection settings (Uchendu et al., 2025).

A useful application is acoustic leak detection in water distribution systems. Two sensors placed along a pipe record leak-induced signals that arrive at different times because of finite propagation speed. A standard model is:

$$x_2(t) \approx x_1(t-\tau) + \text{noise/distortion} \tag{13.3.24}$$

and the task is to estimate (\\tau), which is then converted into a leak location using the propagation model for the pipe. In the time domain, the delay appears as the lag at which cross-correlation is maximized. In the frequency domain, it appears as a linear phase factor, as in (13.3.23). Both views are mathematically consistent, and modern methods often combine them by using FFT-based cross-correlation as a fast baseline while refining the estimate through cross-spectrum phase analysis or weighted generalized cross-correlation (Uchendu et al., 2025; Sun et al., 2025).

The role of FFT-based computation is essential in such monitoring problems because the signals can be long and the estimation may need to be repeated over many overlapping windows or over many sensor pairs. The reduction from quadratic complexity to $O(N\log N)$ makes near-real-time processing feasible, while the cross-spectrum formulation aligns naturally with modern robust estimators. More broadly, this application shows that correlation is not merely a descriptive statistic. In spectral numerical computing it is a structured operator whose efficient evaluation supports concrete inference tasks in sensing, synchronization, and system monitoring.

From the perspective of this chapter, the principal lesson closely parallels that of convolution and deconvolution. The FFT supplies the algebraic mechanism that enables the efficient computation of correlations for all lags. However, effective practical performance depends on the modeling and estimation decisions made around this mechanism. These include choices concerning padding strategy, normalization, mean removal, frequency weighting, and the interpretation of phase information. Consequently, FFT-based correlation should be viewed both as an efficient algorithmic primitive and as a conceptual bridge linking harmonic analysis with real-world inference problems.

### Rust Implementation

Following the discussion in Section 13.3.4 on modern delay estimation methods based on cross-spectrum analysis, Program 13.3.3 provides a practical implementation of generalized cross-correlation using the FFT. The spectral identity in equation (13.3.19) expresses correlation as the inverse transform of the cross-spectrum (G\\odot\\overline{H}). Modern delay-estimation methods extend this formulation by introducing frequency-dependent weights, as described in equation (13.3.20), which modify the cross-spectrum before inversion in order to suppress noise, emphasize reliable frequency bands, or exploit robust phase information. This program demonstrates the computational realization of that idea. It constructs the cross-spectrum from the Fourier transforms of two signals, applies a PHAT-style weighting that normalizes the spectral magnitude while preserving phase information, and then performs the inverse FFT to obtain the weighted correlation sequence. The location of the peak in this sequence provides an estimate of the time delay between the signals, illustrating how the theoretical cross-spectrum framework developed in this section becomes a practical algorithm for signal alignment and delay estimation.

At the center of the implementation is the function `gcc_weighted`, which realizes the generalized cross-correlation formulation introduced in equation (13.3.20). The function first pads the input signals to an FFT length large enough to prevent circular wraparound contamination, following the same padding strategy used earlier in Section 13.3.2. After padding, forward FFTs of the two signals are computed to obtain the spectra $G$ and $H$. The cross-spectrum $G\odot\overline{H}$, corresponding to the expression in equation (13.3.19), is then formed through elementwise multiplication with complex conjugation. To implement generalized cross-correlation, a frequency-dependent weight is applied before inversion. In the present example, a PHAT-type weighting is used, which divides each spectral component by its magnitude. This normalization suppresses amplitude variations while retaining phase information, a property that improves delay estimation performance in noisy environments.

The helper functions implement several important computational steps required for the FFT-based method. The function `fft` performs forward or inverse FFT operations using the `rustfft` library and applies the appropriate normalization when computing the inverse transform. The function `zero_pad_real` converts a real-valued signal into a complex array and pads it with zeros to the desired transform length. The function `next_power_of_two` selects an efficient FFT size by computing the smallest power of two greater than the required padded length, which improves computational performance.

After the weighted cross-spectrum has been computed, the inverse FFT produces the weighted correlation sequence described by equation (13.3.20). The function `find_peak` then scans the resulting sequence to identify the index at which the correlation magnitude is maximal. This index corresponds to the estimated delay between the two signals. In practice, this peak location provides the discrete lag at which the signals are most strongly aligned, corresponding to the time-delay interpretation discussed in equations (13.3.21)–(13.3.24).

The `main` function demonstrates the delay estimation procedure using a simple synthetic example in which one signal is a shifted version of another. The program computes the weighted cross-correlation sequence, prints several correlation values, and then reports the lag corresponding to the largest correlation magnitude. Because the weighting emphasizes phase alignment across frequencies, the resulting correlation sequence exhibits a sharp peak at the correct delay. This example illustrates how cross-spectrum weighting improves delay detection compared with ordinary cross-correlation, particularly in situations where amplitude variations or noise might otherwise blur the correlation peak.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rustfft = "6"
```

```rust
// Program 13.3.3
// Weighted Generalized Cross-Correlation for Delay Estimation
// Problem Statement
// -----------------
// Given two signals g and h, compute their cross-correlation using the FFT.
// Implement the spectral identity
//
//     r_{gh} = IDFT(G ⊙ conj(H))
//
// where G and H are the discrete Fourier transforms of the signals.
// The program should also estimate the lag corresponding to the maximum
// correlation value, which provides the delay estimate between the signals.
// Cargo.toml:
//
// [dependencies]
// rustfft = "6"

use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;

type C64 = Complex<f64>;

fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two()
}

fn fft(data: &mut [C64], inverse: bool) {
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(data.len())
    } else {
        planner.plan_fft_forward(data.len())
    };
    fft.process(data);

    if inverse {
        let scale = 1.0 / data.len() as f64;
        for x in data.iter_mut() {
            *x *= scale;
        }
    }
}

fn zero_pad_real(x: &[f64], n: usize) -> Vec<C64> {
    let mut out = vec![C64::zero(); n];
    for (i, &v) in x.iter().enumerate() {
        out[i] = C64::new(v, 0.0);
    }
    out
}

/// Weighted generalized cross-correlation
fn gcc_weighted(g: &[f64], h: &[f64]) -> Vec<C64> {
    let n = next_power_of_two(g.len() + h.len() - 1);

    let mut gpad = zero_pad_real(g, n);
    let mut hpad = zero_pad_real(h, n);

    fft(&mut gpad, false);
    fft(&mut hpad, false);

    let mut cross = vec![C64::zero(); n];

    for k in 0..n {
        let cross_spec = gpad[k] * hpad[k].conj();

        // simple PHAT-style weighting
        let weight = if cross_spec.norm() > 1e-12 {
            1.0 / cross_spec.norm()
        } else {
            0.0
        };

        cross[k] = cross_spec * weight;
    }

    fft(&mut cross, true);

    cross
}

fn find_peak(corr: &[C64]) -> (usize, f64) {
    corr.iter()
        .enumerate()
        .map(|(i, v)| (i, v.norm()))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
}

fn main() {
    // Example signals: delayed copy
    let g = vec![0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
    let h = vec![1.0, 2.0, 3.0, 2.0, 1.0];

    let corr = gcc_weighted(&g, &h);

    println!("Weighted generalized cross-correlation:");
    for (i, v) in corr.iter().enumerate().take(12) {
        println!("lag {:>3}: {:>10.6}", i, v.re);
    }

    let (lag, value) = find_peak(&corr);

    println!();
    println!("Estimated delay index = {}", lag);
    println!("Peak correlation magnitude = {:.6}", value);
}
```

Program 13.3.3 demonstrates how the FFT-based correlation framework developed earlier in this section can be extended to modern delay estimation algorithms. By introducing frequency-dependent weighting into the cross-spectrum, generalized cross-correlation methods improve robustness in the presence of noise and spectral distortions. The resulting algorithm retains the computational efficiency of the FFT formulation while allowing flexible adaptation to different signal environments.

The numerical example illustrates how the delay between two signals appears as the location of the peak in the weighted correlation sequence. Because the PHAT-style weighting suppresses amplitude information while preserving phase structure, the correlation peak becomes sharply localized at the correct lag. This behavior reflects the theoretical observation in equation (13.3.23) that a time delay manifests as a linear phase factor in the cross-spectrum.

More sophisticated delay estimation methods extend this same computational framework. Cross-spectral phase methods estimate delay by fitting a linear phase slope across frequency bands, while adaptive weighting schemes emphasize frequency regions with higher signal-to-noise ratios. Despite these variations, the underlying computational structure remains the same: compute the cross-spectrum, apply an appropriate weighting, and invert the transform. In this sense, FFT-based correlation serves not only as an efficient algorithmic primitive but also as a bridge between harmonic analysis and practical inference problems in sensing, synchronization, and monitoring systems.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/RLlMwUjQm9x7ePyOGS4Y.8","tags":[]}

# 13.4. Optimal (Wiener) Filtering with the FFT

Optimal Wiener filtering is one of the clearest examples of the computational power of Fourier diagonalization. Whenever a linear operator can be modeled exactly, or approximately, by a circulant matrix, the discrete Fourier transform converts convolution into elementwise multiplication, so that an apparently global estimation problem becomes a set of independent scalar problems, one for each frequency bin. This principle has already appeared in the discussion of convolution and deconvolution. In Wiener filtering it acquires a statistical interpretation: rather than merely inverting a blur or suppressing noise heuristically, one constructs the linear estimator that minimizes mean-square reconstruction error under an assumed signal-plus-noise model. The resulting method remains central in modern numerical computing and signal processing, even as recent work extends it to generalized transform domains, optimization-tuned parameter choices, and hybrid model-based and learned pipelines (Alikaşifoğlu, Kartal and Koç, 2024; Qin and Zhang, 2024; Kong et al., 2025).

## 13.4.1. Foundations, DFT Conventions, and the Circulant Model

We consider a finite discrete signal $x[n]$, real- or complex-valued, sampled at interval $\Delta$. Its length is $N$, and its discrete Fourier transform pair is written as:

$$X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-i 2\pi kn / N}, \qquad k = 0,1,\dots,N-1  \tag{13.4.1}$$

$$x[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k]\, e^{i 2\pi kn / N}, \qquad n = 0,1,\dots,N-1  \tag{13.4.2}$$

The corresponding physical frequencies are:

$$f_k = \frac{k}{N\Delta}, \qquad k = 0,1,\dots,N-1  \tag{13.4.3}$$

with the usual wraparound interpretation for indices $k>N/2$, which correspond to negative frequencies in the discrete periodic setting.

A finite convolution with impulse response $h[m]$ may be written in time-domain form as:

$$y[n] = (h * x)[n] = \sum_{m} h[m]\, x[n - m] \tag{13.4.4}$$

On a finite record, this becomes a structured linear algebra problem. If one adopts a circular model, either because the boundary conditions are periodic or because padding has reduced the intended linear convolution to a valid interior of a circular one, then the convolution can be written as:

$$y=Cx,\tag{13.4.5}$$

where $C$ is a circulant matrix generated by the filter coefficients. Circulant matrices are diagonalized by the DFT matrix $F$, so that,

$$C=F^{*}\operatorname{diag}(H)F \tag{13.4.6}$$

where $H[k]$ is the DFT of the impulse response. Consequently, the convolution equation becomes,

$$Y[k] = H[k]X[k], \qquad k = 0,1,\dots,N-1 \tag{13.4.7}$$

This is simply the convolution theorem written in matrix form. The importance of (13.4.7) is computational as well as conceptual. It shows that once a problem has been reduced to circulant structure, one no longer solves a coupled $N\times N$ system. Instead, one solves $N$ scalar problems, one for each Fourier mode. This pad-to-circulant principle is precisely what makes FFT-based Wiener filtering practical.

## 13.4.2. Measurement Model and Derivation of the Wiener Filter

The Wiener filtering problem begins with a linear observation model,

$$y[n] = (h * x)[n] + \eta[n] \tag{13.4.8}$$

where $x[n]$ is the unknown clean signal, $h[n]$ is a known or estimated instrumental response or point-spread function, and $\eta[n]$ is additive noise. The classical Wiener model assumes that the noise is wide-sense stationary and uncorrelated with the signal. After padding, or under a circular approximation, the DFT-domain relation becomes:

$$Y[k] = H[k]X[k] + N[k] \tag{13.4.9}$$

We seek a linear estimator of the form:

$$\widehat{X}[k] = G[k]\,Y[k] \tag{13.4.10}$$

or, equivalently in the time domain,

$$\widehat{x}=g*y \tag{13.4.11}$$

where $g$ is the Wiener filter impulse response. The objective is to choose the gain $G[k]$ so as to minimize the mean-square error:

$$E\!\left\{ \left|x[n]-\widehat{x}[n]\right|^2 \right\} \tag{13.4.12}$$

Because the DFT diagonalizes the circulant forward operator, the minimization decouples by frequency bin. For each $k$, define the scalar objective,

$$J_k(G[k]) = E\!\left\{ \left|X[k] - G[k]Y[k]\right|^2 \right\} \tag{13.4.13}$$

Using the observation model (13.4.9), this becomes:

$$J_k(G) = E\!\left\{ \left|X - G(HX + N)\right|^2 \right\} \tag{13.4.14}$$

Since the signal and noise are assumed uncorrelated,

$$E\!\left\{ X[k]\,N[k]^{*} \right\} = 0 \tag{13.4.15}$$

Introduce the signal and noise power spectra,

$$
S_{xx}[k] = E\!\left\{ |X[k]|^2 \right\}, \qquad
S_{\eta\eta}[k] = E\!\left\{ |N[k]|^2 \right\} \tag{13.4.16}
$$

Then straightforward expansion of (13.4.14) yields,

$$J_k(G) = |1 - GH|^2 S_{xx}[k] + |G|^2 S_{\eta\eta}[k] \tag{13.4.17}$$

Minimizing (13.4.17) with respect to the complex scalar $G$ gives the classical Wiener deconvolution filter,

$$G[k] = \frac{H[k]^{*} S_{xx}[k]}{|H[k]|^2 S_{xx}[k] + S_{\eta\eta}[k]} \tag{13.4.18}$$

Equation (13.4.18) is one of the fundamental formulas of statistical signal processing. It shows that the estimator is not obtained by inverting $H[k]$ blindly, but by balancing inversion against the relative signal and noise levels at each frequency. In the special case $H[k]\equiv 1$, corresponding to pure denoising without blur, the Wiener gain reduces to:

$$G[k] = \frac{S_{xx}[k]}{S_{xx}[k] + S_{\eta\eta}[k]} \tag{13.4.19}$$

This is the familiar spectral shrinkage form: frequency bins with strong signal relative to noise are preserved, while bins dominated by noise are attenuated. This viewpoint remains central in practical Wiener-filter applications and in modern parameter-tuning strategies (Qin and Zhang, 2024).

### Rust Implementation

Following the derivation of the Wiener estimator in Section 13.4.2, Program 13.4.1 presents a practical implementation of FFT-based Wiener filtering for signal restoration and denoising. The theoretical development showed that, once convolution has been reduced to circulant form, the discrete Fourier transform diagonalizes the forward operator so that the estimation problem separates into independent scalar problems for each frequency bin. This program demonstrates how the Wiener gain derived in Equation (13.4.18) can be implemented numerically using FFT operations and pointwise spectral arithmetic. Two closely related scenarios are illustrated: Wiener deconvolution for the blurred observation model of Equation (13.4.8), and the special case of pure denoising corresponding to Equation (13.4.19). By generating synthetic signals with known spectra, the implementation verifies the reduction in mean-square error predicted by the Wiener optimality criterion and illustrates the computational efficiency of FFT-based filtering.

At the core of the implementation is a small collection of helper functions that implement the Fourier-domain operations required by the Wiener estimator. The functions `fft_in_place` and `real_to_complex_padded` provide the basic numerical infrastructure for transforming real sequences into complex frequency representations and performing forward or inverse FFT operations. These functions implement the transform pair defined in Equations (13.4.1)–(13.4.2) and allow the signal processing steps to be expressed directly in the discrete Fourier domain.

The function `circular_frequency_response` computes the frequency response $H[k]$ of the impulse response $h[n]$ by applying the FFT. This corresponds to the diagonal representation of the circulant convolution operator described in Equation (13.4.6). Once this transform has been computed, the convolution relation becomes the elementwise spectral multiplication given in Equation (13.4.7). In the implementation, this frequency response is used to construct the Wiener gain.

The principal computational step appears in the function `wiener_deconvolution_known_psd`. This function implements the Wiener estimator derived from the mean-square error minimization problem of Equations (13.4.13)–(13.4.17). For each frequency index $k$, it evaluates the gain described in Equation (13.4.18) using the supplied signal and noise power spectral densities. The gain is then applied to the observed spectrum $Y[k]$, producing the estimate $\widehat{X}[k]$ as specified in Equation (13.4.10). After all spectral components have been processed, the inverse FFT returns the estimate $\widehat{x}[n]$ in the time domain, consistent with the reconstruction relation of Equation (13.4.2).

The program also includes a second function, `wiener_denoise_known_psd`, which implements the special case of Wiener filtering in which the measurement model contains noise but no blur. In this situation the transfer function satisfies $H[k]\equiv1$, and the Wiener gain reduces to the spectral shrinkage form shown in Equation (13.4.19). This version of the algorithm illustrates the intuitive interpretation of the Wiener filter as a frequency-dependent attenuation rule that suppresses bins dominated by noise while preserving those dominated by signal energy.

Finally, the `main` function constructs synthetic test data in order to verify the theoretical properties of the Wiener estimator. It first generates a clean signal and applies a short impulse response to produce a blurred observation according to the model in Equation (13.4.8). Deterministic additive noise is then introduced to simulate measurement contamination. The program evaluates the Wiener estimator in both the deconvolution and denoising settings and compares the reconstructed signals to the original clean signal using mean-square error metrics. This experimental structure allows the implementation to confirm that the Wiener gain derived from Equation (13.4.18) reduces reconstruction error relative to the unfiltered observations.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rustfft = "6"
```

```rust
// Problem Statement
// -----------------
// Given a finite clean signal x[n], a known blur h[n], and additive noise η[n],
// form the observation
//
//     y[n] = (h * x)[n] + η[n].
//
// Implement the FFT-based Wiener estimator derived in Section 13.4.2. In the
// frequency domain, compute the Wiener gain
//
//     G[k] = conj(H[k]) S_xx[k] / ( |H[k]|^2 S_xx[k] + S_ηη[k] ),
//
// and recover the estimate
//
//     X̂[k] = G[k] Y[k].
//
// Also implement the special denoising case H[k] = 1, for which the Wiener gain
// reduces to
//
//     G[k] = S_xx[k] / ( S_xx[k] + S_ηη[k] ).
//
// The program should verify the method on synthetic data, report mean-square
// errors before and after filtering, and print representative samples of the
// clean, observed, and reconstructed signals.

// Program 13.4.1
// FFT-Based Wiener Filtering from Known Signal and Noise Spectra
//
// Cargo.toml:
//
// [dependencies]
// rustfft = "6"

use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;

type C64 = Complex<f64>;

fn next_power_of_two_ge(n: usize) -> usize {
    n.max(1).next_power_of_two()
}

fn fft_in_place(data: &mut [C64], inverse: bool) {
    let n = data.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    };
    fft.process(data);

    if inverse {
        let scale = 1.0 / n as f64;
        for x in data.iter_mut() {
            *x *= scale;
        }
    }
}

fn real_to_complex_padded(x: &[f64], n: usize) -> Vec<C64> {
    assert!(x.len() <= n, "input length must not exceed padded length");
    let mut out = vec![C64::zero(); n];
    for (i, &v) in x.iter().enumerate() {
        out[i] = C64::new(v, 0.0);
    }
    out
}

fn circular_frequency_response(h: &[f64], n: usize) -> Vec<C64> {
    let mut hp = real_to_complex_padded(h, n);
    fft_in_place(&mut hp, false);
    hp
}

fn linear_convolution_direct(x: &[f64], h: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len() + h.len() - 1];
    for (i, &xi) in x.iter().enumerate() {
        for (j, &hj) in h.iter().enumerate() {
            y[i + j] += xi * hj;
        }
    }
    y
}

fn mse(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "signals must have the same length");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64
}

fn print_signal(title: &str, x: &[f64], max_items: usize) {
    println!("{title}");
    for (i, &v) in x.iter().take(max_items).enumerate() {
        println!("n = {:>2}, value = {:>12.6}", i, v);
    }
    if x.len() > max_items {
        println!("...");
    }
    println!();
}

fn wiener_deconvolution_known_psd(
    y: &[f64],
    h: &[f64],
    s_xx: &[f64],
    s_nn: &[f64],
    fft_len: usize,
    eps: f64,
) -> Vec<f64> {
    assert_eq!(s_xx.len(), fft_len, "S_xx must have length fft_len");
    assert_eq!(s_nn.len(), fft_len, "S_nn must have length fft_len");
    assert!(
        y.len() <= fft_len,
        "observation length must not exceed fft_len"
    );

    let mut yp = real_to_complex_padded(y, fft_len);
    fft_in_place(&mut yp, false);

    let h_freq = circular_frequency_response(h, fft_len);

    let mut xhat_freq = vec![C64::zero(); fft_len];
    for k in 0..fft_len {
        let denom = h_freq[k].norm_sqr() * s_xx[k] + s_nn[k] + eps;
        let gain = h_freq[k].conj() * (s_xx[k] / denom);
        xhat_freq[k] = gain * yp[k];
    }

    fft_in_place(&mut xhat_freq, true);
    xhat_freq.iter().map(|z| z.re).collect()
}

fn wiener_denoise_known_psd(
    y: &[f64],
    s_xx: &[f64],
    s_nn: &[f64],
    eps: f64,
) -> Vec<f64> {
    let n = y.len();
    assert_eq!(s_xx.len(), n, "S_xx must have length n");
    assert_eq!(s_nn.len(), n, "S_nn must have length n");

    let mut yp = real_to_complex_padded(y, n);
    fft_in_place(&mut yp, false);

    let mut xhat_freq = vec![C64::zero(); n];
    for k in 0..n {
        let gain = s_xx[k] / (s_xx[k] + s_nn[k] + eps);
        xhat_freq[k] = yp[k] * gain;
    }

    fft_in_place(&mut xhat_freq, true);
    xhat_freq.iter().map(|z| z.re).collect()
}

fn main() {
    // Synthetic clean signal: a smooth oscillatory record with two frequencies.
    let n_signal = 32usize;
    let x: Vec<f64> = (0..n_signal)
        .map(|n| {
            let t = n as f64 / n_signal as f64;
            (2.0 * std::f64::consts::PI * 3.0 * t).sin()
                + 0.35 * (2.0 * std::f64::consts::PI * 7.0 * t).cos()
        })
        .collect();

    // A short blur kernel with unit sum.
    let h = vec![0.2, 0.6, 0.2];

    // Deterministic additive noise so the example is fully reproducible.
    let eta: Vec<f64> = (0..n_signal + h.len() - 1)
        .map(|n| {
            0.08 * (2.0 * std::f64::consts::PI * 11.0 * n as f64 / 37.0).sin()
                + 0.03 * (2.0 * std::f64::consts::PI * 5.0 * n as f64 / 19.0).cos()
        })
        .collect();

    // Observation model y = h * x + η.
    let blurred = linear_convolution_direct(&x, &h);
    let y_blur_noisy: Vec<f64> = blurred
        .iter()
        .zip(eta.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    // Wiener deconvolution example.
    let fft_len_deconv = next_power_of_two_ge(y_blur_noisy.len());

    let mut x_pad = real_to_complex_padded(&x, fft_len_deconv);
    let mut noise_pad = real_to_complex_padded(&eta, fft_len_deconv);
    fft_in_place(&mut x_pad, false);
    fft_in_place(&mut noise_pad, false);

    let s_xx_deconv: Vec<f64> = x_pad.iter().map(|z| z.norm_sqr()).collect();
    let s_nn_deconv: Vec<f64> = noise_pad.iter().map(|z| z.norm_sqr()).collect();

    let xhat_deconv_full = wiener_deconvolution_known_psd(
        &y_blur_noisy,
        &h,
        &s_xx_deconv,
        &s_nn_deconv,
        fft_len_deconv,
        1.0e-12,
    );

    let xhat_deconv = xhat_deconv_full[..n_signal].to_vec();

    // Pure denoising example: H[k] = 1.
    let noise_denoise: Vec<f64> = (0..n_signal)
        .map(|n| {
            0.12 * (2.0 * std::f64::consts::PI * 9.0 * n as f64 / 31.0).sin()
                - 0.04 * (2.0 * std::f64::consts::PI * 4.0 * n as f64 / 17.0).cos()
        })
        .collect();

    let y_denoise: Vec<f64> = x
        .iter()
        .zip(noise_denoise.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    let mut x_freq = real_to_complex_padded(&x, n_signal);
    let mut n_freq = real_to_complex_padded(&noise_denoise, n_signal);
    fft_in_place(&mut x_freq, false);
    fft_in_place(&mut n_freq, false);

    let s_xx_denoise: Vec<f64> = x_freq.iter().map(|z| z.norm_sqr()).collect();
    let s_nn_denoise: Vec<f64> = n_freq.iter().map(|z| z.norm_sqr()).collect();

    let xhat_denoise =
        wiener_denoise_known_psd(&y_denoise, &s_xx_denoise, &s_nn_denoise, 1.0e-12);

    println!("Program 13.4.1: FFT-Based Wiener Filtering");
    println!();

    print_signal("Clean signal x[n] (first 10 samples):", &x, 10);
    print_signal("Blurred + noisy observation y[n] (first 10 samples):", &y_blur_noisy, 10);
    print_signal(
        "Wiener deconvolution estimate x̂[n] (first 10 samples):",
        &xhat_deconv,
        10,
    );
    print_signal("Noisy observation for pure denoising (first 10 samples):", &y_denoise, 10);
    print_signal(
        "Pure Wiener denoising estimate x̂[n] (first 10 samples):",
        &xhat_denoise,
        10,
    );

    let mse_blurred_prefix = mse(&blurred[..n_signal], &x);
    let mse_deconv = mse(&xhat_deconv, &x);
    let mse_noisy = mse(&y_denoise, &x);
    let mse_denoised = mse(&xhat_denoise, &x);

    println!("Mean-square error summary:");
    println!("  Blur-only record versus clean signal      : {:.8}", mse_blurred_prefix);
    println!("  Wiener deconvolution versus clean signal  : {:.8}", mse_deconv);
    println!("  Noisy record versus clean signal          : {:.8}", mse_noisy);
    println!("  Wiener denoising versus clean signal      : {:.8}", mse_denoised);
}
```

Program 13.4.1 demonstrates how the theoretical Wiener filter derived in Section 13.4.2 can be implemented efficiently using FFT-based spectral computation. By exploiting the diagonalization property of circulant operators discussed in Section 13.4.1, the algorithm converts a potentially large linear estimation problem into a sequence of independent frequency-domain operations followed by an inverse transform. This reduction allows Wiener filtering to be performed with computational complexity $O(N\log N)$, making it suitable for large data sets and real-time signal processing applications.

The numerical experiments illustrate the central principle of Wiener filtering: rather than directly inverting the system response, the algorithm balances inversion against noise suppression in each frequency bin. In the deconvolution example, the Wiener gain restores the oscillatory structure of the signal while avoiding the instability that would occur in a naïve inverse filter. In the denoising example, the spectral shrinkage behavior predicted by Equation (13.4.19) attenuates frequency components dominated by noise while preserving those with strong signal energy.

The modular structure of the implementation reflects the broader perspective of the section. Once the Fourier transform infrastructure and spectral gain construction are available, the Wiener filter can be adapted easily to a wide variety of applications. Extensions may incorporate alternative spectral estimators, adaptive power spectral density estimation, or block-based overlap-add processing for streaming signals. These variations demonstrate that Wiener filtering is not merely a classical formula but a flexible computational framework linking Fourier analysis, statistical modeling, and efficient numerical algorithms.

## 13.4.3. Interpretation as Regularized Inversion and FFT Implementation

The Wiener filter may also be interpreted as a *stabilized inverse filter*. In a naïve deconvolution approach, one would attempt to recover the original signal spectrum by directly dividing the observed spectrum by the system transfer function. In the discrete frequency domain this leads to:

$$\widehat{X}[k] = \frac{Y[k]}{H[k]} \tag{13.4.20}$$

which represents a straightforward inversion of the linear system response. However, this direct approach is numerically unstable whenever $H[k]$ is small or contaminated by noise, since the division can greatly amplify measurement errors. The Wiener filter modifies this simple inverse by incorporating statistical information about the signal and noise, thereby producing a stable and noise-aware estimate of the original signal spectrum. Equation (13.4.18) avoids this instability by acting like a frequency-dependent regularization. Indeed, it may be rewritten as:

$$
\widehat{X}[k] =
\frac{H[k]^{*}}{|H[k]|^2 + \dfrac{S_{\eta\eta}[k]}{S_{xx}[k]}}
\, Y[k] \tag{13.4.21}
$$

This form makes the analogy with regularized inversion explicit. The ratio,

$$\frac{S_{\eta\eta}[k]}{S_{xx}[k]} \tag{13.4.22}$$

plays the role of a frequency-dependent regularization strength. Where noise dominates, this ratio is large and inversion is suppressed. Where the signal is strong and the operator is well-conditioned, the gain approaches the inverse filter. In this sense, Wiener filtering is one of the clearest bridges between classical spectral estimation and the broader inverse-problem viewpoint of invert plus regularize. Modern work continues to exploit this interpretation, including deep residual Wiener-style deconvolution networks that keep the closed-form inversion structure while replacing difficult-to-estimate quantities such as SNR or PSD terms with learned components (Kong et al., 2025).

From a computational perspective, FFT-based Wiener filtering is straightforward once $H[k]$, $S_{xx}[k]$, and $S_{\eta\eta}[k]$ are available. Given a length-$N$ record $y$, one computes,

$$Y = \operatorname{FFT}(y) \tag{13.4.23}$$

forms the Wiener gain,

$$
G[k] =
\frac{H[k]^{*} S_{xx}[k]}
{|H[k]|^2 S_{xx}[k] + S_{\eta\eta}[k] + \varepsilon} \tag{13.4.24}
$$

where $\varepsilon>0$ is a small numerical safeguard, then computes

$$\widehat{X}[k] = G[k] \, Y[k] \tag{13.4.25}$$

and finally returns to the time domain via,

$$\widehat{x} = \operatorname{IFFT}(\widehat{X}) \tag{13.4.26}$$

If the gain $G[k]$ has been precomputed, each filtering pass requires one forward FFT, one inverse FFT, and $O(N)$ pointwise multiplications, so the overall complexity is $O(N\log N)$, with working memory of order $O(N)$.

The main practical difficulty is not the FFT itself, but the estimation of the spectral quantities $S_{xx}$ and $S_{\eta\eta}$. In an idealized derivation they are assumed known, but in real problems they must be modeled or estimated statistically. This is why Wiener filtering is inseparable from power spectral density estimation, which is taken up more fully in the next section.

### Rust Implementation

Following the discussion in Section 13.4.3 on the interpretation of Wiener filtering as a stabilized inverse problem, Program 13.4.2 provides a practical implementation of FFT-based Wiener filtering that explicitly contrasts naive inverse filtering with its regularized Wiener counterpart. The theoretical development showed that direct inversion of the transfer function, as expressed in Equation (13.4.20), can become numerically unstable when the system response approaches zero. The Wiener formulation addresses this difficulty by introducing a frequency-dependent regularization term derived from the signal and noise spectra, as expressed in Equations (13.4.21) and (13.4.22). This program implements the complete FFT pipeline described in Equations (13.4.23)–(13.4.26), computing both a naive inverse-filter estimate and a Wiener-filter estimate for the same synthetic observation. By comparing the resulting reconstructions and their mean-square errors, the implementation illustrates the stabilizing effect of spectral regularization and demonstrates the computational efficiency of FFT-based Wiener filtering.

At the core of the implementation are several helper functions that implement the FFT-based signal-processing pipeline required for Wiener filtering. The functions `fft_in_place` and `real_to_complex_padded` provide the numerical infrastructure for converting real-valued signals into complex spectral representations and computing forward or inverse discrete Fourier transforms. These operations implement the transform pair defined by Equations (13.4.1)–(13.4.2) and allow convolution and filtering operations to be expressed as elementwise spectral multiplications.

The function `frequency_response` computes the Fourier transform of the impulse response $h[n]$, producing the transfer function $H[k]$. This operation corresponds to the diagonal representation of the circulant convolution operator discussed in Equation (13.4.6). Once the transfer function has been computed, the convolution relation becomes the spectral multiplication shown in Equation (13.4.7). In the program, this frequency response is used both for naive inverse filtering and for constructing the Wiener gain.

The function `naive_inverse_filter` implements the direct spectral inversion described in Equation (13.4.20). It computes the spectrum $Y[k]$ of the observed signal and attempts to reconstruct the original spectrum by dividing by the transfer function $H[k]$. Because very small values of $H[k]$ can produce catastrophic numerical amplification, a small safeguard parameter is introduced to prevent division by zero. Even with this protection, the method remains highly sensitive to noise, which is precisely the instability highlighted in the discussion of naive inverse filtering.

The Wiener estimator is implemented through two complementary functions: `wiener_filter_precompute_gain` and `apply_precomputed_gain`. The first constructs the frequency-domain gain $G[k]$ according to Equation (13.4.24), using the signal and noise power spectral densities $S_{xx}[k]$ and $S_{\eta\eta}[k]$. The small parameter $\varepsilon$ serves as a numerical safeguard, preventing division by extremely small denominators. Once the gain has been computed, the second function applies it to the observed spectrum, producing the estimate $\widehat{X}[k]$ as described in Equation (13.4.25). The reconstructed signal is then obtained by applying the inverse FFT, corresponding to Equation (13.4.26).

Finally, the `main` function constructs a synthetic example to illustrate the difference between naive inversion and Wiener filtering. A clean signal is generated and convolved with a short impulse response to simulate a blurred observation. Deterministic additive noise is then introduced to mimic measurement contamination. Using these known signals, the program constructs surrogate power spectra for the signal and noise and applies both reconstruction methods. The results are compared by computing mean-square errors relative to the clean signal, allowing the numerical behavior of both approaches to be evaluated directly.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rustfft = "6"
```

```rust
// Problem Statement
// -----------------
// Given a finite observed signal y[n] produced by a blurred-and-noisy model,
// implement two frequency-domain reconstructions:
//
// 1. Naive inverse filtering:
//        X̂_inv[k] = Y[k] / H[k]
//
// 2. Wiener filtering with numerical safeguard:
//        G[k] = conj(H[k]) S_xx[k] / ( |H[k]|^2 S_xx[k] + S_ηη[k] + ε )
//        X̂_w[k] = G[k] Y[k]
//
// The program should demonstrate that direct inverse filtering becomes unstable
// when H[k] is small, while the Wiener filter acts as a frequency-dependent
// regularized inverse. It should compute both reconstructions using the FFT,
// return them to the time domain, and compare their mean-square errors against
// the known clean signal on a synthetic example.

// Program 13.4.2
// Regularized Wiener Inversion and FFT Implementation
//
// Cargo.toml:
//
// [dependencies]
// rustfft = "6"

use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;

type C64 = Complex<f64>;

fn next_power_of_two_ge(n: usize) -> usize {
    n.max(1).next_power_of_two()
}

fn fft_in_place(data: &mut [C64], inverse: bool) {
    let n = data.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    };
    fft.process(data);

    if inverse {
        let scale = 1.0 / n as f64;
        for z in data.iter_mut() {
            *z *= scale;
        }
    }
}

fn real_to_complex_padded(x: &[f64], n: usize) -> Vec<C64> {
    assert!(x.len() <= n, "input length must not exceed padded length");
    let mut out = vec![C64::zero(); n];
    for (i, &v) in x.iter().enumerate() {
        out[i] = C64::new(v, 0.0);
    }
    out
}

fn linear_convolution_direct(x: &[f64], h: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len() + h.len() - 1];
    for (i, &xi) in x.iter().enumerate() {
        for (j, &hj) in h.iter().enumerate() {
            y[i + j] += xi * hj;
        }
    }
    y
}

fn frequency_response(h: &[f64], fft_len: usize) -> Vec<C64> {
    let mut h_pad = real_to_complex_padded(h, fft_len);
    fft_in_place(&mut h_pad, false);
    h_pad
}

fn spectrum_from_real_signal(x: &[f64], fft_len: usize) -> Vec<C64> {
    let mut x_pad = real_to_complex_padded(x, fft_len);
    fft_in_place(&mut x_pad, false);
    x_pad
}

fn mse(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "signals must have the same length");
    a.iter()
        .zip(b.iter())
        .map(|(&u, &v)| {
            let d = u - v;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64
}

fn print_signal(title: &str, x: &[f64], max_items: usize) {
    println!("{title}");
    for (i, &v) in x.iter().take(max_items).enumerate() {
        println!("n = {:>2}, value = {:>12.6}", i, v);
    }
    if x.len() > max_items {
        println!("...");
    }
    println!();
}

fn print_selected_frequency_data(
    title: &str,
    h_freq: &[C64],
    s_xx: &[f64],
    s_nn: &[f64],
    eps: f64,
    bins: &[usize],
) {
    println!("{title}");
    println!(
        "{:>5} {:>14} {:>14} {:>14} {:>14}",
        "k", "|H[k]|", "S_xx[k]", "S_nn[k]", "reg. ratio"
    );
    for &k in bins {
        let hmag = h_freq[k].norm();
        let reg_ratio = s_nn[k] / (s_xx[k] + eps);
        println!(
            "{:>5} {:>14.6} {:>14.6} {:>14.6} {:>14.6}",
            k, hmag, s_xx[k], s_nn[k], reg_ratio
        );
    }
    println!();
}

fn naive_inverse_filter(
    y: &[f64],
    h: &[f64],
    fft_len: usize,
    eps_inv: f64,
) -> Vec<f64> {
    let mut y_freq = spectrum_from_real_signal(y, fft_len);
    let h_freq = frequency_response(h, fft_len);

    for k in 0..fft_len {
        let denom = if h_freq[k].norm() > eps_inv {
            h_freq[k]
        } else {
            // Prevent exact numerical blow-up while still behaving like
            // a fragile inverse near poorly conditioned frequencies.
            C64::new(eps_inv, 0.0)
        };
        y_freq[k] /= denom;
    }

    fft_in_place(&mut y_freq, true);
    y_freq.iter().map(|z| z.re).collect()
}

fn wiener_filter_precompute_gain(
    h: &[f64],
    s_xx: &[f64],
    s_nn: &[f64],
    fft_len: usize,
    eps: f64,
) -> Vec<C64> {
    assert_eq!(s_xx.len(), fft_len, "S_xx must have length fft_len");
    assert_eq!(s_nn.len(), fft_len, "S_nn must have length fft_len");

    let h_freq = frequency_response(h, fft_len);
    let mut gain = vec![C64::zero(); fft_len];

    for k in 0..fft_len {
        let denom = h_freq[k].norm_sqr() * s_xx[k] + s_nn[k] + eps;
        gain[k] = h_freq[k].conj() * (s_xx[k] / denom);
    }

    gain
}

fn apply_precomputed_gain(y: &[f64], gain: &[C64], fft_len: usize) -> Vec<f64> {
    assert_eq!(gain.len(), fft_len, "gain must have length fft_len");

    let mut y_freq = spectrum_from_real_signal(y, fft_len);
    for k in 0..fft_len {
        y_freq[k] *= gain[k];
    }

    fft_in_place(&mut y_freq, true);
    y_freq.iter().map(|z| z.re).collect()
}

fn main() {
    // Synthetic clean signal with mixed smooth and oscillatory content.
    let n_signal = 48usize;
    let x: Vec<f64> = (0..n_signal)
        .map(|n| {
            let t = n as f64 / n_signal as f64;
            0.9 * (2.0 * std::f64::consts::PI * 4.0 * t).sin()
                + 0.35 * (2.0 * std::f64::consts::PI * 11.0 * t).cos()
                + 0.15 * (2.0 * std::f64::consts::PI * 2.0 * t).sin()
        })
        .collect();

    // A blur kernel whose frequency response becomes small at some frequencies,
    // making direct inversion sensitive to noise.
    let h = vec![0.25, 0.5, 0.25];

    // Deterministic additive noise for reproducibility.
    let clean_conv = linear_convolution_direct(&x, &h);
    let eta: Vec<f64> = (0..clean_conv.len())
        .map(|n| {
            0.06 * (2.0 * std::f64::consts::PI * 13.0 * n as f64 / 53.0).sin()
                + 0.025 * (2.0 * std::f64::consts::PI * 7.0 * n as f64 / 31.0).cos()
        })
        .collect();

    let y: Vec<f64> = clean_conv
        .iter()
        .zip(eta.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    let fft_len = next_power_of_two_ge(y.len());

    // In this synthetic example, the clean signal and noise are known, so their
    // spectra can be used to form exact PSD surrogates for demonstration.
    let x_freq = spectrum_from_real_signal(&x, fft_len);
    let n_freq = spectrum_from_real_signal(&eta, fft_len);

    let s_xx: Vec<f64> = x_freq.iter().map(|z| z.norm_sqr()).collect();
    let s_nn: Vec<f64> = n_freq.iter().map(|z| z.norm_sqr()).collect();

    let eps_wiener = 1.0e-12;
    let eps_inverse = 1.0e-8;

    // Precompute Wiener gain, then apply one forward FFT, O(N) multiplications,
    // and one inverse FFT, matching the pipeline of Eqs. (13.4.23)-(13.4.26).
    let gain = wiener_filter_precompute_gain(&h, &s_xx, &s_nn, fft_len, eps_wiener);
    let xhat_wiener_full = apply_precomputed_gain(&y, &gain, fft_len);
    let xhat_wiener = xhat_wiener_full[..n_signal].to_vec();

    // Naive inverse filtering baseline.
    let xhat_inverse_full = naive_inverse_filter(&y, &h, fft_len, eps_inverse);
    let xhat_inverse = xhat_inverse_full[..n_signal].to_vec();

    let blurred_prefix = clean_conv[..n_signal].to_vec();

    print_signal("Clean signal x[n] (first 10 samples):", &x, 10);
    print_signal("Blurred + noisy observation y[n] (first 10 samples):", &y, 10);
    print_signal(
        "Naive inverse-filter estimate x̂_inv[n] (first 10 samples):",
        &xhat_inverse,
        10,
    );
    print_signal(
        "Wiener-filter estimate x̂_w[n] (first 10 samples):",
        &xhat_wiener,
        10,
    );

    let h_freq = frequency_response(&h, fft_len);
    print_selected_frequency_data(
        "Representative frequency-bin data:",
        &h_freq,
        &s_xx,
        &s_nn,
        eps_wiener,
        &[0, 1, 2, 8, 16, 24, 31],
    );

    println!("Mean-square error summary:");
    println!(
        "  Blurred record versus clean signal          : {:.8}",
        mse(&blurred_prefix, &x)
    );
    println!(
        "  Naive inverse filter versus clean signal    : {:.8}",
        mse(&xhat_inverse, &x)
    );
    println!(
        "  Wiener filter versus clean signal           : {:.8}",
        mse(&xhat_wiener, &x)
    );
    println!();

    println!("Precomputed gain length: {}", gain.len());
    println!(
        "Each Wiener filtering pass uses one forward FFT, one inverse FFT, and O(N) pointwise multiplications."
    );
}
```

Program 13.4.2 demonstrates the practical implementation of Wiener filtering as a regularized inverse problem in the Fourier domain. The numerical experiment highlights the key weakness of naive inverse filtering: when the transfer function becomes small, the reconstruction becomes extremely sensitive to noise and numerical errors, producing unstable estimates with large amplitudes and high reconstruction error. This instability is a direct consequence of Equation (13.4.20), which attempts to invert the forward operator without accounting for measurement uncertainty.

In contrast, the Wiener estimator stabilizes the inversion process by incorporating spectral information about both the signal and the noise. As expressed in Equation (13.4.21), the Wiener gain introduces a frequency-dependent regularization term that suppresses inversion in regions where noise dominates or where the operator is poorly conditioned. The resulting reconstruction balances fidelity to the observed data with robustness to noise, producing accurate estimates even in challenging spectral regions.

The implementation also illustrates the computational efficiency of the FFT-based approach. Once the Wiener gain has been computed, each filtering pass requires only one forward FFT, one inverse FFT, and a set of pointwise spectral multiplications. This $O(N\log N)$ complexity makes Wiener filtering practical for large signals and streaming data. The modular structure of the program further emphasizes that Wiener filtering is not merely a classical formula but a flexible computational framework that can be extended to alternative spectral estimators, adaptive noise modeling, or block-based signal processing pipelines.

## 13.4.4. Modern Variants and Practical Applications

Recent work retains the core Wiener objective while extending the setting in several directions. One line of development generalizes the transform domain itself. If the natural signal representation is not the ordinary Fourier basis, then Wiener filtering may be performed in alternative diagonalizing coordinates. For example, optimal Wiener filtering has been generalized to joint time-vertex fractional Fourier domains for time-varying graph signals, where the best signal-noise separation may not occur in the standard Fourier domain (Alikaşifoğlu, Kartal and Koç, 2024). The principle is unchanged: find a transform that diagonalizes, or nearly diagonalizes, the relevant structure, and then solve mode by mode.

A second line of development concerns parameter tuning. Classical presentations often assume that the spectral models are available and fixed, but modern implementations increasingly optimize Wiener parameters directly for application-level objectives. Qin and Zhang (2024), for example, study Wiener filtering for image restoration with improved optimization of filter parameters rather than relying on fixed heuristic choices. This reflects a broader trend in numerical computing: even when the estimator is analytically known, its practical effectiveness depends strongly on how its statistical ingredients are estimated and tuned.

A third line of development blends model-based inversion with learned components. Deep residual Wiener deconvolution networks preserve the frequency-domain inversion structure of Wiener deconvolution while replacing hard-to-specify pieces, such as SNR estimation or prior modeling, with trainable modules (Kong et al., 2025). Such methods are best viewed not as abandoning Wiener theory, but as extending it. The classical filter remains the interpretable analytic core, while machine learning is used to compensate for model mismatch and difficult parameter estimation.

A concrete application appears in gravitational-wave data analysis. The measured time series may be written schematically as:

$$y(t) = s(t;\theta) + n(t) \tag{13.4.27}$$

where $s(t;\theta)$ is a deterministic waveform depending on parameters $\theta$, and $n(t)$ is stochastic detector noise. Inference pipelines rely heavily on whitening and SNR-optimal filtering, both of which depend critically on reliable noise PSD estimation. Martini et al. (2024) revisit maximum entropy spectral analysis in this setting and show that, in small-sample regimes, Burg/MESA-style approaches can produce PSD estimates with lower variance and bias than Welch-style methods, thereby improving downstream inference reliability. At very low frequencies, where leakage and resolution are especially problematic, adaptive sine multitaper methods have also been proposed to improve PSD estimation performance (Liu et al., 2025). This application demonstrates the full computational chain: one must estimate the PSD robustly, construct the optimal or whitening filter from it, and implement the resulting spectral algorithm in a numerically stable way.

Another compelling application is multichannel audio enhancement for drone-embedded microphones. In this setting, the desired signal can be overwhelmed by strong propeller and motor noise, so that the signal-to-noise ratio may be far below $0$ dB. Manamperi et al. (2024) propose a practical multichannel enhancement framework that combines multichannel Wiener filtering with Gaussian-mixture-model-based post-filtering, validated on real drone acoustics data. Closely related work emphasizes that accurate noise PSD estimation is essential, since poor PSD estimates lead directly to suboptimal Wiener gains and degraded MMSE performance (Wang et al., 2025). In computational terms, this is a textbook FFT pipeline: block Fourier transforms, PSD estimation, per-frequency Wiener gains, and overlap-add synthesis, all subject to latency and computational constraints.

From the perspective of this chapter, the central lesson is that Wiener filtering is far more than a classical denoising formula. It is a model-based spectral inversion method whose efficiency comes from FFT-accelerated diagonalization and whose effectiveness depends on statistical modeling. The Fourier transform makes the computation fast; the signal and noise spectra determine whether the result is trustworthy. That combination of linear algebra, probability, and algorithmic efficiency is precisely why Wiener filtering remains a cornerstone of modern spectral numerical computing.

### Rust Implementation

Following the discussion in Section 13.4.4 on modern variants of Wiener filtering and their role in practical signal processing pipelines, Program 13.4.3 provides a complete implementation of block-based Wiener filtering with power spectral density estimation and overlap-add synthesis. In real-world applications such as audio enhancement and sensor signal recovery, Wiener filtering is rarely applied to an entire signal at once. Instead, signals are processed in short-time Fourier blocks, where local stationarity assumptions are more appropriate and computational efficiency is achieved through FFT-based diagonalization. This program demonstrates how noise statistics can be estimated from a training segment, how frequency-dependent Wiener gains are constructed, and how stable time-domain reconstruction is achieved through windowing and overlap-add techniques. The implementation highlights the interplay between spectral estimation, numerical stability, and efficient transform-based computation in modern Wiener filtering systems.

At the core of the implementation is the FFT-based transformation framework, which enables efficient diagonalization of convolution operators as described in Equations (13.4.23)–(13.4.26). The function `fft_in_place` performs forward and inverse Fourier transforms using the `rustfft` library, allowing signals to be mapped between time and frequency domains. This transformation is essential for constructing Wiener gains in a mode-by-mode fashion, consistent with the diagonal structure of the filtering problem in the spectral domain.

The function `hann_window` generates a smooth tapering window that is applied to each signal block prior to transformation. This windowing step is crucial for reducing spectral leakage and ensuring that the short-time Fourier representation remains well-behaved. In combination with overlap-add reconstruction, it guarantees that the processed signal can be recombined without introducing discontinuities or artificial boundary effects.

The function `frame_signal_zero_padded` partitions the input signal into overlapping frames with symmetric zero-padding at the boundaries. This padding strategy ensures that all samples, including those near the beginning and end of the signal, are processed under consistent windowing conditions. Without this step, the overlap-add normalization would be ill-conditioned near the boundaries, leading to the kind of numerical instability observed in earlier implementations.

The function `power_spectrum_of_frame` computes the periodogram of each windowed frame by taking the squared magnitude of its Fourier transform. This serves as the basic building block for estimating power spectral densities. Using this, the function `estimate_noise_psd` computes an averaged noise spectrum from a noise-only segment. This corresponds to estimating $S_{\eta\eta}[k]$ in Equation (13.4.24), which plays a central role in determining the frequency-dependent Wiener gain.

The main filtering operation is implemented in the function `overlap_add_wiener`. For each frame, the noisy signal is transformed into the frequency domain, and an estimate of the signal spectrum is obtained by subtracting the noise PSD from the observed spectrum, consistent with the interpretation of Equation (13.4.21). The Wiener gain is then constructed according to Equation (13.4.24), with a small regularization parameter ensuring numerical stability. The filtered spectrum is transformed back to the time domain, and the resulting frames are recombined using overlap-add synthesis with proper normalization. This step ensures that the reconstructed signal maintains both amplitude consistency and smooth transitions between frames.

The auxiliary functions `mse` and `snr_db` compute performance metrics that quantify the effectiveness of the filtering process. These provide a direct numerical validation of the Wiener filter’s ability to reduce noise and improve signal fidelity.

The `main` function serves as a complete demonstration of the Wiener filtering pipeline. It begins by generating a synthetic clean signal and a structured noise process, along with a separate noise-only segment for PSD estimation. It then computes the noisy observation and applies the block Wiener filter using the previously defined functions. Finally, it reports sample values of the clean, noisy, and enhanced signals, along with quantitative performance measures such as mean-square error and signal-to-noise ratio. This end-to-end workflow mirrors practical implementations in applications such as audio enhancement and spectral denoising, where both statistical estimation and efficient computation are essential.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rustfft = "6"
```

```rust
// Problem Statement
// -----------------
// Implement a practical FFT-based Wiener filtering pipeline for block signal
// enhancement. The program should:
//
// 1. Simulate a clean time series s[n] and additive noise n[n], forming
//        y[n] = s[n] + n[n].
// 2. Estimate the noise power spectral density from a noise-only training
//    segment using averaged periodograms.
// 3. Process the noisy signal in overlapping short-time Fourier blocks.
// 4. For each frequency bin, form a Wiener gain
//        G[k] = S_ss[k] / (S_ss[k] + S_nn[k] + ε),
//    where S_ss[k] is estimated from the current observation block by
//        max(S_yy[k] - S_nn[k], floor),
//    and S_nn[k] is the estimated noise PSD.
// 5. Apply the gain to each block spectrum, perform an inverse FFT, and
//    reconstruct the enhanced signal by overlap-add.
// 6. Report time-domain error statistics before and after enhancement.
//
// This program models the practical FFT pipeline discussed in Section 13.4.4:
// block transforms, PSD estimation, per-frequency Wiener gains, and overlap-add
// synthesis under finite-window and latency constraints.

// Program 13.4.3
// Block Wiener Filtering with Noise PSD Estimation and Overlap-Add Synthesis
//
// Cargo.toml:
//
// [dependencies]
// rustfft = "6"

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

type C64 = Complex<f64>;

fn fft_in_place(data: &mut [C64], inverse: bool) {
    let n = data.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    };
    fft.process(data);

    if inverse {
        let scale = 1.0 / n as f64;
        for z in data.iter_mut() {
            *z *= scale;
        }
    }
}

fn hann_window(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    (0..n)
        .map(|i| 0.5 - 0.5 * (two_pi * i as f64 / (n as f64 - 1.0)).cos())
        .collect()
}

fn estimate_noise_psd(noise_only: &[f64], frame_len: usize, hop: usize, window: &[f64]) -> Vec<f64> {
    let frames = frame_signal_zero_padded(noise_only, frame_len, hop);
    assert!(
        !frames.is_empty(),
        "noise-only segment must produce at least one frame"
    );

    let mut psd = vec![0.0; frame_len];
    for frame in &frames {
        let spec = power_spectrum_of_frame(frame, window);
        for (acc, v) in psd.iter_mut().zip(spec.iter()) {
            *acc += *v;
        }
    }

    let scale = 1.0 / frames.len() as f64;
    for v in psd.iter_mut() {
        *v *= scale;
    }
    psd
}

fn frame_signal_zero_padded(x: &[f64], frame_len: usize, hop: usize) -> Vec<Vec<f64>> {
    assert!(frame_len > 0, "frame length must be positive");
    assert!(hop > 0, "hop size must be positive");

    let pad = frame_len / 2;
    let padded_len = x.len() + 2 * pad;

    let mut padded = vec![0.0; padded_len];
    padded[pad..pad + x.len()].copy_from_slice(x);

    let mut frames = Vec::new();
    let mut start = 0usize;

    while start < padded_len {
        let mut frame = vec![0.0; frame_len];
        let end = (start + frame_len).min(padded_len);
        let count = end - start;
        frame[..count].copy_from_slice(&padded[start..end]);
        frames.push(frame);

        if end == padded_len {
            break;
        }
        start += hop;
    }

    frames
}

fn power_spectrum_of_frame(frame: &[f64], window: &[f64]) -> Vec<f64> {
    assert_eq!(frame.len(), window.len(), "frame and window lengths must match");

    let mut spectrum: Vec<C64> = frame
        .iter()
        .zip(window.iter())
        .map(|(&x, &w)| C64::new(x * w, 0.0))
        .collect();

    fft_in_place(&mut spectrum, false);
    spectrum.iter().map(|z| z.norm_sqr()).collect()
}

fn overlap_add_wiener(
    noisy: &[f64],
    noise_psd: &[f64],
    frame_len: usize,
    hop: usize,
    window: &[f64],
    eps: f64,
    floor: f64,
) -> Vec<f64> {
    assert_eq!(noise_psd.len(), frame_len, "noise PSD must match frame length");
    assert_eq!(window.len(), frame_len, "window length mismatch");

    let pad = frame_len / 2;
    let frames = frame_signal_zero_padded(noisy, frame_len, hop);

    let output_len = if frames.is_empty() {
        0
    } else {
        hop * (frames.len() - 1) + frame_len
    };

    let mut output = vec![0.0; output_len];
    let mut norm = vec![0.0; output_len];

    for (frame_idx, frame) in frames.iter().enumerate() {
        let start = frame_idx * hop;

        let mut spectrum: Vec<C64> = frame
            .iter()
            .zip(window.iter())
            .map(|(&x, &w)| C64::new(x * w, 0.0))
            .collect();

        fft_in_place(&mut spectrum, false);

        for k in 0..frame_len {
            let s_yy = spectrum[k].norm_sqr();
            let s_nn = noise_psd[k];
            let s_ss = (s_yy - s_nn).max(floor);
            let gain = s_ss / (s_ss + s_nn + eps);
            spectrum[k] *= gain;
        }

        fft_in_place(&mut spectrum, true);

        for n in 0..frame_len {
            let idx = start + n;
            if idx < output_len {
                let w = window[n];
                output[idx] += spectrum[n].re * w;
                norm[idx] += w * w;
            }
        }
    }

    for i in 0..output_len {
        if norm[i] > 1.0e-8 {
            output[i] /= norm[i];
        }
    }

    output[pad..pad + noisy.len()].to_vec()
}

fn mse(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "signals must have equal lengths");
    a.iter()
        .zip(b.iter())
        .map(|(&u, &v)| {
            let d = u - v;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64
}

fn snr_db(reference: &[f64], error: &[f64]) -> f64 {
    assert_eq!(reference.len(), error.len(), "signals must have equal lengths");
    let signal_power: f64 = reference.iter().map(|x| x * x).sum();
    let noise_power: f64 = error.iter().map(|x| x * x).sum();
    10.0 * (signal_power / noise_power.max(1.0e-30)).log10()
}

fn print_signal_samples(title: &str, x: &[f64], count: usize) {
    println!("{title}");
    for (i, &v) in x.iter().take(count).enumerate() {
        println!("n = {:>3}, value = {:>12.6}", i, v);
    }
    if x.len() > count {
        println!("...");
    }
    println!();
}

fn main() {
    let fs = 1024.0;
    let noise_only_len = 512usize;
    let signal_len = 2048usize;

    let noise_only: Vec<f64> = (0..noise_only_len)
        .map(|n| {
            let t = n as f64 / fs;
            0.28 * (2.0 * std::f64::consts::PI * 121.0 * t).sin()
                + 0.18 * (2.0 * std::f64::consts::PI * 247.0 * t).cos()
                + 0.08 * (2.0 * std::f64::consts::PI * 53.0 * t).sin()
        })
        .collect();

    let clean: Vec<f64> = (0..signal_len)
        .map(|n| {
            let t = n as f64 / fs;
            let envelope = 0.65 + 0.35 * (2.0 * std::f64::consts::PI * 0.7 * t).sin().abs();
            envelope
                * ((2.0 * std::f64::consts::PI * 70.0 * t).sin()
                    + 0.55 * (2.0 * std::f64::consts::PI * 145.0 * t).cos()
                    + 0.30 * (2.0 * std::f64::consts::PI * 215.0 * t).sin())
        })
        .collect();

    let noise: Vec<f64> = (0..signal_len)
        .map(|n| {
            let t = n as f64 / fs;
            0.32 * (2.0 * std::f64::consts::PI * 121.0 * t).sin()
                + 0.20 * (2.0 * std::f64::consts::PI * 247.0 * t).cos()
                + 0.09 * (2.0 * std::f64::consts::PI * 53.0 * t).sin()
                + 0.05 * (2.0 * std::f64::consts::PI * 301.0 * t).cos()
        })
        .collect();

    let noisy: Vec<f64> = clean
        .iter()
        .zip(noise.iter())
        .map(|(&s, &n)| s + n)
        .collect();

    let frame_len = 256usize;
    let hop = 128usize;
    let window = hann_window(frame_len);

    let noise_psd = estimate_noise_psd(&noise_only, frame_len, hop, &window);

    let enhanced = overlap_add_wiener(
        &noisy,
        &noise_psd,
        frame_len,
        hop,
        &window,
        1.0e-10,
        1.0e-8,
    );

    let noisy_error: Vec<f64> = noisy
        .iter()
        .zip(clean.iter())
        .map(|(&y, &x)| y - x)
        .collect();

    let enhanced_error: Vec<f64> = enhanced
        .iter()
        .zip(clean.iter())
        .map(|(&xhat, &x)| xhat - x)
        .collect();

    print_signal_samples("Clean signal s[n] (first 12 samples):", &clean, 12);
    print_signal_samples("Noisy observation y[n] (first 12 samples):", &noisy, 12);
    print_signal_samples("Enhanced signal x̂[n] (first 12 samples):", &enhanced, 12);

    println!("Noise PSD estimate (first 12 bins):");
    for (k, value) in noise_psd.iter().take(12).enumerate() {
        println!("k = {:>3}, PSD = {:>12.6}", k, value);
    }
    println!();

    let mse_noisy = mse(&noisy, &clean);
    let mse_enhanced = mse(&enhanced, &clean);

    let snr_in = snr_db(&clean, &noisy_error);
    let snr_out = snr_db(&clean, &enhanced_error);

    println!("Performance summary:");
    println!("  Input MSE                    : {:.8}", mse_noisy);
    println!("  Output MSE                   : {:.8}", mse_enhanced);
    println!("  Input SNR (dB)               : {:.4}", snr_in);
    println!("  Output SNR (dB)              : {:.4}", snr_out);
    println!("  SNR improvement (dB)         : {:.4}", snr_out - snr_in);
    println!();

    println!("Processing summary:");
    println!("  Frame length                 : {}", frame_len);
    println!("  Hop size                     : {}", hop);
    println!("  Number of FFT bins           : {}", frame_len);
    println!("  Noise-only training samples  : {}", noise_only_len);
    println!("  Enhanced signal length       : {}", enhanced.len());
}
```

Program 13.4.3 demonstrates a practical implementation of Wiener filtering in a modern block-based spectral processing framework. Rather than applying a single global transform, the algorithm operates on overlapping signal segments, estimates noise statistics from data, and applies frequency-dependent gains that adapt to local signal conditions. This reflects the computational reality discussed in Section 13.4.4, where Wiener filtering is embedded within a broader pipeline involving PSD estimation, windowing, and reconstruction.

The example illustrates several important computational principles. First, the effectiveness of Wiener filtering depends critically on the quality of spectral estimates, particularly the noise PSD. Second, numerical stability is not guaranteed by the filtering formula alone, but requires careful handling of windowing and overlap-add normalization. Third, FFT-based implementations enable efficient processing of large signals, reducing the computational complexity to $O(N \log N)$ per block.

The modular structure of the code allows for straightforward extensions. More sophisticated PSD estimation techniques, such as multitaper or parametric methods, can be incorporated to improve performance in challenging environments. Similarly, adaptive or learned components may be introduced to refine the estimation of signal and noise statistics. These directions connect directly to current research trends, where classical Wiener filtering serves as the analytic foundation for more advanced model-based and data-driven approaches.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/3EHI6DJbOa6wrLlOFYPO.8","tags":[]}

# 13.5. Power Spectrum Estimation Using the FFT

Power spectrum estimation is the statistical companion to optimal filtering. In Wiener filtering, the quantities $S_{xx}$ and $S_{\eta\eta}$ determine the frequency-dependent gain, so the quality of the filter depends directly on the quality of the spectral estimates. More broadly, power spectral density estimation is one of the main ways in which Fourier methods connect finite sampled data to stochastic models of physical processes. In practice, however, the spectrum is never observed directly. One must infer it from a finite record, which means balancing resolution, variance, leakage, computational cost, and robustness to missing or irregular samples. The FFT is central to this task because it makes repeated spectral estimation computationally feasible, but the estimator itself is a statistical construction rather than a mere transform. This section develops the standard FFT-based estimators, explains their numerical tradeoffs, and highlights several modern directions, including multitaper methods, autoregressive maximum-entropy viewpoints, nonuniform sampling, and online spectral estimation.

## 13.5.1. Power Spectral Density and Finite-Record Estimation

Let $x[n]$ be a discrete-time wide-sense stationary process. Its power spectral density is defined as the Fourier transform of its autocovariance sequence. In discrete time, this is the Wiener–Khinchin relation,

$$
S_{xx}(f) =
\sum_{\tau=-\infty}^{\infty} R_{xx}[\tau] \, e^{-i 2\pi f \tau \Delta} \tag{13.5.1}
$$

where,

$$R_{xx}[\tau] = E\!\left\{ x[n] x[n+\tau] \right\} \tag{13.5.2}$$

Equation (13.5.1) is an infinite-data idealization. In numerical work one never has access to the full covariance sequence, nor to infinitely many samples. Instead, one observes a finite record:

$$x[0], x[1], \dots, x[N-1] \tag{13.5.3}$$

and constructs an estimator of the underlying spectrum. This is the point at which computational and statistical issues enter simultaneously. The FFT provides rapid access to finite Fourier transforms, but finite observation windows create spectral leakage, while random variability of the data produces estimator variance. The design of a PSD estimator is therefore always a compromise between spectral resolution and statistical stability.

A direct FFT of the raw finite record corresponds implicitly to rectangular windowing, which usually produces substantial leakage when the underlying process contains strong narrowband components or when the data do not align naturally with the record length. For this reason, practical PSD estimation almost always begins by tapering the data with a window $w[n]$. The windowed discrete Fourier transform is:

$$
D[k] =
\sum_{n=0}^{N-1} w[n]\, x[n]\, e^{-i 2\pi k n / N},
\qquad k = 0,1,\dots,N-1 \tag{13.5.4}
$$

A common two-sided FFT-based PSD estimator is then,

$$\widehat{S}_{xx}[k] \propto \frac{|D[k]|^2}{U} \tag{13.5.5}$$

where the window normalization factor is:

$$U = \sum_{n=0}^{N-1} w[n]^2 \tag{13.5.6}$$

Depending on convention, one may also divide by $N$, by $\Delta$, or by both in order to obtain the desired physical units. The essential point is not the bookkeeping constant but the statistical meaning: windowing is introduced to reduce leakage, while the squared magnitude of the FFT provides an estimate of spectral power at each frequency bin. Modern discussions continue to emphasize that leakage control and variance control are the two central and competing objectives of PSD estimation (Dodson-Robinson and Haley, 2025).

### Rust Implementation

Following the discussion in Section 13.5.1 on finite-record power spectral density estimation and the role of windowing in controlling spectral leakage, Program 13.5.1 provides a practical implementation of the FFT-based windowed periodogram. In numerical computation, the idealized Wiener–Khinchin relation cannot be evaluated directly because only a finite sample of the underlying process is available. As a result, spectral estimation must be constructed from truncated data, where windowing and normalization play essential roles in balancing resolution and statistical stability. This program implements the windowed discrete Fourier transform and the corresponding PSD estimator, and compares the effects of rectangular and Hann tapers on leakage behavior. By evaluating a synthetic signal with off-grid frequencies, the program illustrates how finite-record effects influence the observed spectrum and highlights the importance of appropriate tapering in practical FFT-based spectral analysis.

At the core of the implementation is the function `windowed_periodogram`, which directly realizes the finite-record estimator described in Equations (13.5.4)–(13.5.6). This function constructs a window $w[n]$, applies it pointwise to the input signal, computes the discrete Fourier transform using the FFT, and then forms the power spectral density estimate from the squared magnitude of the transform. The normalization factor $U=\sum_n w[n]^2$ is computed separately to ensure that the estimator reflects the energy of the tapered signal rather than the raw data. In this way, the function mirrors the mathematical structure of the windowed periodogram while maintaining computational efficiency through FFT-based evaluation.

The functions `rectangular_window`, `hann_window`, and `make_window` provide the tapering mechanisms required for practical spectral estimation. The rectangular window corresponds to the implicit truncation of the finite record and therefore exhibits strong spectral leakage. The Hann window, on the other hand, introduces a smooth taper that reduces sidelobe levels at the cost of a broader main lobe. By abstracting the window choice into a simple interface, the implementation allows the same estimator to be evaluated under different leakage-control strategies without altering the underlying computational pipeline.

The function `fft_in_place` performs the forward discrete Fourier transform using the `rustfft` library. This is the computational engine that enables efficient evaluation of Equation (13.5.4). Because the FFT reduces the complexity from quadratic to $O(N \log N)$, it makes repeated spectral estimation feasible even for moderately large data sets. The transformation is performed in place for efficiency, and inverse scaling is applied only when needed.

The helper function `window_normalization` computes the quantity $U$ in Equation (13.5.6), which ensures that the PSD estimate is properly scaled relative to the applied window. The functions `top_bins`, `print_top_bins`, and `print_first_bins` are used to analyze and present the spectral results. They identify dominant frequency components and display selected portions of the spectrum, allowing the user to observe both peak behavior and low-level leakage effects.

The `main` function constructs a synthetic finite record containing multiple sinusoidal components, including frequencies that do not align exactly with FFT bins. This deliberate mismatch highlights the leakage phenomena discussed in the section. The program then computes two PSD estimates, one using a rectangular window and one using a Hann window, and reports the dominant spectral bins and low-frequency behavior. This comparison provides a direct numerical illustration of the bias–leakage tradeoff introduced by windowing.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rustfft = "6"
```

```rust
// Problem Statement
// -----------------
// Given a finite sampled record x[0], x[1], ..., x[N-1], estimate its two-sided
// power spectral density using the FFT-based windowed periodogram.
//
// Implement the windowed transform
//
//     D[k] = sum_{n=0}^{N-1} w[n] x[n] exp(-i 2πkn/N),
//
// and the corresponding PSD estimate
//
//     S_xx[k] = |D[k]|^2 / (U * fs),
//
// where
//
//     U = sum_{n=0}^{N-1} w[n]^2,
//
// and fs = 1/Δ is the sampling frequency. Demonstrate the estimator on a finite
// signal containing narrowband components and compare rectangular and Hann
// windows to illustrate the effect of tapering on spectral leakage.

// Program 13.5.1
// Windowed Periodogram for Power Spectrum Estimation Using the FFT
//
// Cargo.toml
//
// [dependencies]
// rustfft = "6"

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::cmp::Ordering;
use std::f64::consts::PI;

type C64 = Complex<f64>;

#[derive(Clone, Copy, Debug)]
enum WindowKind {
    Rectangular,
    Hann,
}

fn rectangular_window(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

fn hann_window(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos())
        .collect()
}

fn make_window(kind: WindowKind, n: usize) -> Vec<f64> {
    match kind {
        WindowKind::Rectangular => rectangular_window(n),
        WindowKind::Hann => hann_window(n),
    }
}

fn fft_in_place(data: &mut [C64], inverse: bool) {
    let n = data.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    };
    fft.process(data);

    if inverse {
        let scale = 1.0 / n as f64;
        for z in data.iter_mut() {
            *z *= scale;
        }
    }
}

fn window_normalization(window: &[f64]) -> f64 {
    window.iter().map(|w| w * w).sum()
}

fn windowed_periodogram(x: &[f64], fs: f64, kind: WindowKind) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let window = make_window(kind, n);
    let u = window_normalization(&window);

    let mut spectrum: Vec<C64> = x
        .iter()
        .zip(window.iter())
        .map(|(&xn, &wn)| C64::new(xn * wn, 0.0))
        .collect();

    fft_in_place(&mut spectrum, false);

    let freqs: Vec<f64> = (0..n)
        .map(|k| {
            if k <= n / 2 {
                k as f64 * fs / n as f64
            } else {
                -((n - k) as f64) * fs / n as f64
            }
        })
        .collect();

    let psd: Vec<f64> = spectrum.iter().map(|z| z.norm_sqr() / (u * fs)).collect();

    (freqs, psd)
}

fn top_bins(freqs: &[f64], psd: &[f64], how_many: usize) -> Vec<(usize, f64, f64)> {
    let mut bins: Vec<(usize, f64, f64)> = freqs
        .iter()
        .zip(psd.iter())
        .enumerate()
        .map(|(k, (&f, &p))| (k, f, p))
        .collect();

    bins.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
    bins.truncate(how_many);
    bins
}

fn print_top_bins(title: &str, freqs: &[f64], psd: &[f64], how_many: usize) {
    println!("{title}");
    println!("{:>6} {:>14} {:>18}", "k", "frequency", "PSD");
    for (k, f, p) in top_bins(freqs, psd, how_many) {
        println!("{:>6} {:>14.6} {:>18.8}", k, f, p);
    }
    println!();
}

fn print_first_bins(title: &str, freqs: &[f64], psd: &[f64], count: usize) {
    println!("{title}");
    println!("{:>6} {:>14} {:>18}", "k", "frequency", "PSD");
    for k in 0..count.min(freqs.len()) {
        println!("{:>6} {:>14.6} {:>18.8}", k, freqs[k], psd[k]);
    }
    println!();
}

fn main() {
    let n = 256usize;
    let delta = 1.0 / 256.0;
    let fs = 1.0 / delta;

    // Synthetic finite record with two nearby narrowband components and one
    // additional weaker component. The frequencies are chosen so that not all
    // tones align exactly with FFT bins, making leakage visible.
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 * delta;
            1.0 * (2.0 * PI * 27.3 * t).sin()
                + 0.65 * (2.0 * PI * 31.8 * t).sin()
                + 0.25 * (2.0 * PI * 70.5 * t).cos()
        })
        .collect();

    let (freqs_rect, psd_rect) = windowed_periodogram(&x, fs, WindowKind::Rectangular);
    let (freqs_hann, psd_hann) = windowed_periodogram(&x, fs, WindowKind::Hann);

    println!("Program 13.5.1: Windowed Periodogram PSD Estimation");
    println!("Record length N = {n}");
    println!("Sampling interval Δ = {:.6}", delta);
    println!("Sampling frequency fs = {:.6}", fs);
    println!();

    print_top_bins(
        "Strongest spectral bins using a rectangular window:",
        &freqs_rect,
        &psd_rect,
        8,
    );

    print_top_bins(
        "Strongest spectral bins using a Hann window:",
        &freqs_hann,
        &psd_hann,
        8,
    );

    print_first_bins(
        "First 16 nonnegative-frequency bins for the rectangular-window estimate:",
        &freqs_rect,
        &psd_rect,
        16,
    );

    print_first_bins(
        "First 16 nonnegative-frequency bins for the Hann-window estimate:",
        &freqs_hann,
        &psd_hann,
        16,
    );
}
```

Program 13.5.1 demonstrates the fundamental structure of FFT-based power spectrum estimation for finite data records. The implementation shows that spectral estimation is not simply a matter of applying the FFT, but rather a carefully constructed procedure involving windowing, normalization, and interpretation of the resulting transform. The example highlights how the choice of window affects the distribution of spectral energy, illustrating the tradeoff between resolution and leakage that lies at the heart of PSD estimation.

The comparison between rectangular and Hann windows illustrates two contrasting behaviors. The rectangular window preserves sharper spectral resolution but produces significant leakage into neighboring frequency bins. The Hann window reduces sidelobe leakage and yields a cleaner spectral estimate, but at the cost of a broader main lobe and reduced frequency resolution. These effects are consistent with the theoretical discussion in Section 13.5.1 and demonstrate how windowing shapes the statistical properties of the estimator.

The modular structure of the code allows the estimator to be extended easily. Additional window functions, averaging strategies such as Welch’s method, or multitaper approaches can be incorporated without changing the overall FFT-based framework. This provides a foundation for more advanced spectral estimation techniques, where improved statistical performance is achieved through averaging, adaptive weighting, or parametric modeling. In this sense, the windowed periodogram serves as the computational and conceptual starting point for the broader family of PSD estimation methods developed in subsequent sections.

## 13.5.2. Windowed Periodograms, Welch Averaging, and the Bias–Variance Tradeoff

The estimator in (13.5.5) is the basic windowed periodogram. It is computationally convenient because it requires only a single FFT, but statistically it is often unsatisfactory. The variance of the periodogram does not decay rapidly with record length in the way one might hope, so the estimate can remain noisy even for moderately long data records. This motivates averaging strategies.

The most widely used practical improvement is Welch-style averaging. One begins by partitioning a long record into $K$ shorter segments, often allowing overlap between adjacent segments. Each segment is windowed, transformed by the FFT, converted into a periodogram, and then the segment-wise periodograms are averaged. If $x^{(j)}[n]$ denotes the $j$-th windowed segment, then one computes:

$$
D^{(j)}[k] =
\sum_{n=0}^{N-1} w[n]\, x^{(j)}[n]\, e^{-i 2\pi k n / N},
\qquad j = 1,\dots,K \tag{13.5.7}
$$

and forms the averaged estimator

$$
\widehat{S}_{xx}^{\text{Welch}}[k] =
\frac{1}{K}\sum_{j=1}^{K}\frac{\left|D^{(j)}[k]\right|^2}{U} \tag{13.5.8}
$$

This reduces variance because averaging stabilizes the random fluctuations of individual periodograms. The cost is increased bias and reduced frequency resolution, since shorter segments contain less spectral information than the full record. In practice, Welch estimation is a controlled tradeoff between resolution and robustness, and that tradeoff remains central in recent work. Overlapping segment strategies continue to be used not only in traditional PSD estimation but also in dynamic estimation tasks where overlap improves refresh rate or robustness in evolving systems (Szczepanik and Kelner, 2025).

From a numerical computing perspective, Welch’s method is attractive because it remains FFT-friendly. The cost is essentially that of (K) moderate-sized FFTs rather than one large FFT, and these transforms parallelize naturally. This makes the method effective in both offline and streaming contexts. Conceptually, it also illustrates an important general principle: in spectral estimation, averaging is usually the mechanism for variance reduction, while window choice controls leakage and bias.

### Rust Implementation

Following the discussion in Section 13.5.2 on windowed periodograms, Welch averaging, and the associated bias–variance tradeoff, Program 13.5.2 provides a practical implementation of power spectral density estimation using FFT-based techniques. In spectral analysis, the raw periodogram offers high frequency resolution but suffers from large variance, while averaging strategies such as Welch’s method improve statistical stability at the cost of reduced resolution. This program implements both the single-segment windowed periodogram and the multi-segment Welch estimator, illustrating how segment length, overlap, and window choice influence the resulting spectrum. By combining windowing, segmentation, and FFT computation within a unified framework, the implementation demonstrates how theoretical considerations translate into efficient and robust numerical algorithms for practical spectral estimation.

At the core of the implementation is the construction of window functions and their associated normalization, which correspond directly to the formulation in Equation (13.5.7). The function `make_window` generates commonly used windows such as rectangular, Hann, and Hamming, each of which controls spectral leakage in a distinct way. The function `window_energy` computes the normalization factor (U), ensuring that the scaled periodogram in Equation (13.5.8) remains properly normalized across different window choices. Together, these components encapsulate the role of windowing in shaping the bias properties of the estimator.

The segmentation process is implemented in the function `segment_signal`, which partitions the input signal into overlapping blocks. This reflects the construction of the segments $x^{(j)}[n]$ used in Equation (13.5.7). The overlap parameter determines how much adjacent segments share data, directly affecting the number of segments $K$ and thus the variance reduction achieved in the averaged estimator of Equation (13.5.8). By controlling the hop size, the implementation exposes the tradeoff between computational cost and statistical efficiency.

The function `windowed_periodogram` performs the FFT-based computation of $D^{(j)}[k]$ for each segment. It applies the selected window to the data, computes the discrete Fourier transform using an efficient FFT planner, and converts the result into a one-sided power spectrum. The normalization by $U$ follows Equation (13.5.8), and the conversion to a one-sided spectrum accounts for the symmetry of real-valued signals. This function represents the computational realization of the windowed periodogram and forms the building block for the Welch estimator.

The Welch averaging itself is implemented in the function `welch_psd`, which iterates over all segments, computes their individual periodograms, and accumulates their contributions. The final averaging over $K$ segments corresponds exactly to Equation (13.5.8), reducing variance through repeated independent estimates. The function also constructs the associated frequency grid, allowing the spectral estimate to be interpreted in terms of physical frequencies. This modular structure highlights how Welch’s method can be viewed as a simple extension of the basic periodogram through repeated application and averaging.

To demonstrate the framework, the program includes a synthetic signal composed of multiple sinusoidal components and a small disturbance term. The function `synthetic_signal` generates this data deterministically, allowing reproducible experiments without reliance on external randomness. Auxiliary functions such as `print_top_peaks` and `print_psd_table` provide diagnostic insight into the spectral content by identifying dominant frequency components and displaying representative portions of the spectrum.

The `main` function serves to illustrate the bias–variance tradeoff central to this section. It evaluates three configurations: a Welch estimate with moderate segment length and overlap, a more aggressively averaged estimate using shorter segments, and a single-segment windowed periodogram using the full record. By comparing these outputs, the program demonstrates how increasing the number of segments reduces variance but broadens spectral peaks, while using the full record sharpens frequency localization at the expense of increased variability. This comparative study provides a concrete numerical realization of the theoretical principles discussed in Section 13.5.2.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rustfft = "6"
```

```rust
// Problem Statement:
// Implement windowed periodogram estimation and Welch-averaged power spectral density
// estimation for a real-valued discrete signal, in the setting of Section 13.5.2.
// For each segment x^{(j)}[n], compute the windowed DFT
//
//     D^{(j)}[k] = sum_{n=0}^{N-1} w[n] x^{(j)}[n] exp(-i 2*pi*k*n/N),
//
// then form the averaged estimator
//
//     S_welch[k] = (1/K) sum_{j=1}^{K} |D^{(j)}[k]|^2 / U,
//
// where U = sum_{n=0}^{N-1} w[n]^2 is the window-energy normalization.
// The program demonstrates how segment length, overlap, and window choice
// affect the bias-variance tradeoff in practical spectral estimation.
//
// Cargo.toml dependencies:
//
// [dependencies]
// rustfft = "6"

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Supported analysis windows.
#[derive(Clone, Copy, Debug)]
enum WindowKind {
    Rectangular,
    Hann,
    Hamming,
}

/// Output container for Welch PSD estimation.
#[derive(Debug)]
struct WelchPsd {
    /// Frequency bins in cycles/sample.
    freqs: Vec<f64>,
    /// One-sided PSD estimate for real-valued data.
    psd: Vec<f64>,
    /// Number of segments actually averaged.
    segments_used: usize,
    /// Segment length N.
    segment_len: usize,
    /// Hop size between segments.
    hop_size: usize,
    /// Window-energy normalization U = sum w[n]^2.
    window_energy: f64,
}

/// Generate an analysis window of length `n`.
fn make_window(kind: WindowKind, n: usize) -> Vec<f64> {
    assert!(n > 0, "window length must be positive");

    if n == 1 {
        return vec![1.0];
    }

    match kind {
        WindowKind::Rectangular => vec![1.0; n],
        WindowKind::Hann => (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos())
            .collect(),
        WindowKind::Hamming => (0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos())
            .collect(),
    }
}

/// Compute U = sum_n w[n]^2, the normalization appearing in the windowed periodogram.
fn window_energy(window: &[f64]) -> f64 {
    window.iter().map(|&w| w * w).sum()
}

/// Extract overlapping segments from a signal.
/// Each returned segment has length `segment_len`.
fn segment_signal(signal: &[f64], segment_len: usize, overlap: usize) -> Vec<&[f64]> {
    assert!(segment_len > 0, "segment length must be positive");
    assert!(overlap < segment_len, "overlap must satisfy overlap < segment_len");

    if signal.len() < segment_len {
        return Vec::new();
    }

    let hop = segment_len - overlap;
    let mut segments = Vec::new();
    let mut start = 0usize;

    while start + segment_len <= signal.len() {
        segments.push(&signal[start..start + segment_len]);
        start += hop;
    }

    segments
}

/// Compute the one-sided periodogram of a single real-valued segment.
/// The FFT is performed on the windowed data.
fn windowed_periodogram(
    segment: &[f64],
    window: &[f64],
    fft_planner: &mut FftPlanner<f64>,
) -> Vec<f64> {
    let n = segment.len();
    assert_eq!(window.len(), n, "window and segment must have the same length");

    let u = window_energy(window);
    assert!(u > 0.0, "window energy must be positive");

    let mut buffer: Vec<Complex<f64>> = segment
        .iter()
        .zip(window.iter())
        .map(|(&x, &w)| Complex::new(x * w, 0.0))
        .collect();

    let fft = fft_planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    // For real input, keep the one-sided spectrum: k = 0, ..., floor(N/2).
    let half = n / 2;
    let mut psd = vec![0.0; half + 1];

    for k in 0..=half {
        let power = buffer[k].norm_sqr() / u;

        // Convert the two-sided spectrum to a one-sided PSD.
        // DC and Nyquist (if present) are not doubled.
        psd[k] = if k == 0 || (n % 2 == 0 && k == half) {
            power
        } else {
            2.0 * power
        };
    }

    psd
}

/// Compute Welch's averaged one-sided PSD estimate for a real-valued signal.
fn welch_psd(
    signal: &[f64],
    segment_len: usize,
    overlap: usize,
    window_kind: WindowKind,
) -> WelchPsd {
    assert!(
        signal.len() >= segment_len,
        "signal length must be at least the segment length"
    );

    let hop = segment_len - overlap;
    let window = make_window(window_kind, segment_len);
    let u = window_energy(&window);
    let segments = segment_signal(signal, segment_len, overlap);

    assert!(
        !segments.is_empty(),
        "no segments available; check segment length and signal length"
    );

    let mut fft_planner = FftPlanner::<f64>::new();
    let spectrum_len = segment_len / 2 + 1;
    let mut average_psd = vec![0.0; spectrum_len];

    for segment in &segments {
        let psd_j = windowed_periodogram(segment, &window, &mut fft_planner);
        for (acc, value) in average_psd.iter_mut().zip(psd_j.iter()) {
            *acc += *value;
        }
    }

    let k = segments.len() as f64;
    for value in &mut average_psd {
        *value /= k;
    }

    let freqs: Vec<f64> = (0..spectrum_len)
        .map(|idx| idx as f64 / segment_len as f64)
        .collect();

    WelchPsd {
        freqs,
        psd: average_psd,
        segments_used: segments.len(),
        segment_len,
        hop_size: hop,
        window_energy: u,
    }
}

/// Generate a synthetic signal composed of two sinusoids plus a small deterministic disturbance.
/// This avoids needing a random-number crate while still producing a realistic PSD example.
fn synthetic_signal(n: usize) -> Vec<f64> {
    let f1 = 0.10; // cycles/sample
    let f2 = 0.28; // cycles/sample

    (0..n)
        .map(|i| {
            let t = i as f64;
            let s1 = 1.0 * (2.0 * PI * f1 * t).sin();
            let s2 = 0.45 * (2.0 * PI * f2 * t + 0.35).sin();

            // Small deterministic quasi-noise component.
            let disturbance = 0.18 * (2.0 * PI * 0.371 * t).sin()
                + 0.09 * (2.0 * PI * 0.417 * t + 1.1).cos()
                + 0.05 * (2.0 * PI * 0.063 * t * t / n as f64).sin();

            s1 + s2 + disturbance
        })
        .collect()
}

/// Print the strongest spectral peaks.
fn print_top_peaks(psd: &WelchPsd, top_k: usize) {
    let mut pairs: Vec<(usize, f64)> = psd.psd.iter().copied().enumerate().collect();

    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top {} spectral peaks:", top_k);
    for (rank, (idx, value)) in pairs.into_iter().take(top_k).enumerate() {
        println!(
            "  {:>2}. bin {:>4}, frequency {:>8.5} cycles/sample, PSD {:>12.6}",
            rank + 1,
            idx,
            psd.freqs[idx],
            value
        );
    }
}

/// Print a compact table of frequency bins and PSD values.
fn print_psd_table(psd: &WelchPsd, bins_to_show: usize) {
    println!("\nFirst {} one-sided PSD bins:", bins_to_show);
    println!("{:>8} {:>14} {:>16}", "k", "frequency", "PSD");

    let count = bins_to_show.min(psd.psd.len());
    for k in 0..count {
        println!("{:>8} {:>14.6} {:>16.8}", k, psd.freqs[k], psd.psd[k]);
    }
}

fn main() {
    let signal_len = 2048;
    let signal = synthetic_signal(signal_len);

    // Example 1: longer segments, moderate overlap.
    let estimate_a = welch_psd(&signal, 256, 128, WindowKind::Hann);

    println!("Welch PSD Estimate A");
    println!("--------------------");
    println!("Signal length      : {}", signal_len);
    println!("Segment length     : {}", estimate_a.segment_len);
    println!("Overlap            : {}", estimate_a.segment_len - estimate_a.hop_size);
    println!("Hop size           : {}", estimate_a.hop_size);
    println!("Segments used      : {}", estimate_a.segments_used);
    println!("Window             : {:?}", WindowKind::Hann);
    println!("Window energy U    : {:.6}", estimate_a.window_energy);

    print_top_peaks(&estimate_a, 6);
    print_psd_table(&estimate_a, 16);

    // Example 2: shorter segments, higher averaging effect but lower frequency resolution.
    let estimate_b = welch_psd(&signal, 128, 64, WindowKind::Hamming);

    println!("\nWelch PSD Estimate B");
    println!("--------------------");
    println!("Signal length      : {}", signal_len);
    println!("Segment length     : {}", estimate_b.segment_len);
    println!("Overlap            : {}", estimate_b.segment_len - estimate_b.hop_size);
    println!("Hop size           : {}", estimate_b.hop_size);
    println!("Segments used      : {}", estimate_b.segments_used);
    println!("Window             : {:?}", WindowKind::Hamming);
    println!("Window energy U    : {:.6}", estimate_b.window_energy);

    print_top_peaks(&estimate_b, 6);

    // Example 3: a single-segment windowed periodogram via Welch with K = 1.
    let estimate_c = welch_psd(&signal, signal_len, 0, WindowKind::Rectangular);

    println!("\nSingle-Segment Windowed Periodogram");
    println!("-----------------------------------");
    println!("Signal length      : {}", signal_len);
    println!("Segment length     : {}", estimate_c.segment_len);
    println!("Overlap            : {}", estimate_c.segment_len - estimate_c.hop_size);
    println!("Hop size           : {}", estimate_c.hop_size);
    println!("Segments used      : {}", estimate_c.segments_used);
    println!("Window             : {:?}", WindowKind::Rectangular);
    println!("Window energy U    : {:.6}", estimate_c.window_energy);

    print_top_peaks(&estimate_c, 6);
}
```

Program 13.5.2 demonstrates a practical approach to spectral density estimation by combining windowing, segmentation, and averaging within an FFT-based framework. This approach reflects the central computational challenge discussed in Section 13.5.2: balancing frequency resolution against statistical stability in the presence of finite data.

The comparison between Welch-averaged estimates and the single-segment periodogram highlights two key aspects of spectral estimation. Averaging across segments reduces variance and produces smoother, more reliable spectra, while shorter segments inherently limit frequency resolution and introduce bias through spectral broadening. Conversely, using the full record preserves resolution but yields a noisier estimate. These observations reinforce the role of Welch’s method as a controlled compromise between competing objectives.

The modular structure of the implementation allows the framework to be extended readily. Alternative window functions, adaptive segmentation strategies, or multitaper approaches can be incorporated with minimal modification. This provides a foundation for more advanced spectral estimation techniques, including time-varying spectral analysis, streaming implementations, and modern approaches that integrate statistical modeling with FFT-based computation.

## 13.5.3. Multitaper Estimation and Adaptive Spectral Concentration

A more refined approach replaces a single window by a family of orthogonal tapers. In multitaper estimation, one uses $K$ different windows $w^{(j)}[n]$, typically chosen to have strong spectral concentration properties, computes one tapered FFT for each, and then averages the resulting power estimates. The multitaper estimate may be written as:

$$
\widehat{S}_{xx}(f) \approx
\frac{1}{K}\sum_{j=1}^{K}
\left|
\sum_{n} w^{(j)}[n]\, x[n]\, e^{-i 2\pi f n \Delta}
\right|^2 \tag{13.5.9}
$$

The most important classical examples are Slepian or DPSS tapers, whose construction is guided by an optimal concentration problem in frequency. The intuition is that each taper provides a slightly different but well-concentrated view of the same spectrum. Averaging across these views reduces variance, while the concentration properties of the tapers suppress leakage more effectively than many standard single-window approaches. In contemporary language, multitaper estimation is often described as a way of balancing a three-way tradeoff among bias, variance, and effective spectral resolution or bandwidth (Dodson-Robinson and Haley, 2025).

This framework has continued to evolve in modern applications. Adaptive multitaper methods attempt to choose or weight the tapers in a data-dependent way, especially in regimes where certain frequencies are difficult to resolve accurately. Low-frequency estimation is a notable example, since leakage and poor resolution are often particularly severe there. Recent work on gravitational-wave system evaluation proposes enhanced adaptive sine multitaper methods that improve both resolution and mean-square error relative to baseline alternatives in low-frequency settings (Liu et al., 2025). Such developments fit naturally into a modern numerical computing treatment because they combine classical spectral concentration ideas with practical adaptation and performance analysis.

Algorithmically, multitaper estimation remains FFT-based. One computes several FFTs instead of one, but the structure is unchanged: taper, transform, square magnitude, and average. Thus the method preserves computational scalability while achieving significantly improved statistical behavior in many settings.

### Rust Implementation

Following the discussion in Section 13.5.3 on multitaper estimation and adaptive spectral concentration, Program 13.5.3 provides a practical implementation of multitaper power spectral density estimation using an FFT-based framework. In contrast to single-window approaches, multitaper methods employ a family of orthogonal tapers to obtain multiple independent spectral estimates, which are then averaged to reduce variance while maintaining controlled spectral leakage. This program implements a set of orthogonal sine tapers as a computationally efficient surrogate for classical Slepian tapers, computes the corresponding tapered Fourier transforms, and forms both equal-weight and adaptively weighted multitaper estimates. By integrating taper construction, spectral estimation, and adaptive weighting within a unified structure, the implementation demonstrates how multitaper methods achieve improved statistical stability and spectral concentration in practical numerical settings.

At the core of the implementation is the construction of an orthogonal family of tapers through the function `sine_tapers`. This function generates a sequence of mutually orthogonal window functions (w^{(j)}\[n\]), which play the role of the tapers in Equation (13.5.9). The orthogonality of these tapers ensures that each tapered transform provides an independent estimate of the spectrum, a property that is fundamental to the variance reduction achieved in multitaper methods. The function `taper_energy` computes the normalization factor associated with each taper, ensuring that all tapered periodograms are placed on a consistent scale prior to averaging.

The computation of individual tapered spectra is performed by the function `tapered_periodogram`, which applies a given taper to the signal and evaluates the corresponding Fourier transform. This directly implements the inner summation in Equation (13.5.9), followed by the squared magnitude to obtain a power estimate. The function also converts the result to a one-sided spectrum appropriate for real-valued signals, and normalizes by the taper energy to maintain consistency across different tapers.

The multitaper averaging process is encapsulated in the function `equal_weight_multitaper`, which computes the arithmetic mean of the individual tapered spectra. This corresponds to the outer summation in Equation (13.5.9), where the contributions from each taper are combined to form the final estimate. In addition, the function `adaptive_multitaper` implements a simple data-dependent weighting strategy. Starting from the equal-weight estimate as a pilot spectrum, it assigns weights to each taper based on how closely its spectral estimate agrees with the pilot value at each frequency. This reflects the adaptive multitaper philosophy described in Section 13.5.3, where tapers that produce more reliable estimates are given greater influence.

The function `multitaper_psd` coordinates the full computation by generating the tapers, computing the individual tapered periodograms, and forming both equal-weight and adaptive estimates. It also constructs the associated frequency grid, enabling direct interpretation of the spectral results. For comparison, the function `rectangular_periodogram` computes a single-window estimate using a rectangular taper, providing a baseline against which the benefits of multitaper estimation can be assessed.

To illustrate the behavior of the method, the program includes a synthetic signal composed of multiple sinusoidal components, including closely spaced frequencies and a low-frequency component. This signal is designed to highlight the challenges of spectral leakage and resolution. Auxiliary functions such as `print_top_peaks`, `print_comparison_table`, and `print_taper_orthogonality` provide diagnostic insight into the spectral estimates, including verification of taper orthogonality and comparison of dominant spectral components across methods.

The `main` function demonstrates the multitaper framework by computing and comparing three spectral estimates: a rectangular-window periodogram, an equal-weight multitaper estimate, and an adaptively weighted multitaper estimate. By examining the resulting spectra, the program illustrates how multitaper averaging reduces variance, how orthogonal tapers improve spectral concentration, and how adaptive weighting can further refine the estimate in challenging frequency regions. This provides a concrete computational realization of the theoretical principles discussed in Section 13.5.3.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rustfft = "6"
```

```rust
// Problem Statement:
// Implement multitaper spectral estimation for a real-valued discrete signal in the setting
// of Section 13.5.3. Instead of using a single analysis window, the program constructs a
// family of orthogonal tapers, computes one tapered FFT for each taper, forms the individual
// tapered power spectra, and then averages them to obtain a multitaper estimate.
//
// The code uses orthogonal sine tapers as a practical FFT-friendly taper family:
//
//     w^{(j)}[n] = sqrt(2/(N+1)) sin(pi j (n+1)/(N+1)),   j = 1, ..., K,
//
// which are mutually orthogonal on n = 0, ..., N-1.
//
// The program computes
//
//     S_mt[k] = (1/K) sum_{j=1}^K | sum_n w^{(j)}[n] x[n] e^{-i 2*pi*k*n/N} |^2,
//
// and also includes a simple adaptive weighting strategy that places larger weight on tapers
// whose spectra agree more closely with a pilot estimate at each frequency. This provides a
// practical illustration of adaptive spectral concentration without requiring the full DPSS
// construction.
//
// Cargo.toml dependencies:
//
// [dependencies]
// rustfft = "6"

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Result container for multitaper spectral estimation.
#[derive(Debug)]
struct MultitaperPsd {
    /// Frequency bins in cycles/sample.
    freqs: Vec<f64>,
    /// Equal-weight multitaper PSD.
    psd_equal: Vec<f64>,
    /// Adaptively weighted multitaper PSD.
    psd_adaptive: Vec<f64>,
    /// Individual tapered one-sided spectra.
    #[allow(dead_code)]
    tapered_psd: Vec<Vec<f64>>,
    /// The taper family used in the computation.
    tapers: Vec<Vec<f64>>,
    /// Number of tapers.
    num_tapers: usize,
    /// Record length.
    signal_len: usize,
}

/// Construct K orthogonal sine tapers of length N.
///
/// These tapers are convenient for a textbook implementation because they are explicit,
/// orthogonal, spectrally smoother than a rectangular window, and require no eigenvalue solve.
fn sine_tapers(n: usize, k: usize) -> Vec<Vec<f64>> {
    assert!(n > 0, "signal length must be positive");
    assert!(k > 0, "number of tapers must be positive");
    assert!(k <= n, "number of tapers must not exceed signal length");

    let scale = (2.0 / (n as f64 + 1.0)).sqrt();

    (1..=k)
        .map(|j| {
            (0..n)
                .map(|idx| {
                    let angle = PI * j as f64 * (idx as f64 + 1.0) / (n as f64 + 1.0);
                    scale * angle.sin()
                })
                .collect()
        })
        .collect()
}

/// Compute the discrete energy of a taper.
fn taper_energy(taper: &[f64]) -> f64 {
    taper.iter().map(|&v| v * v).sum()
}

/// Compute the one-sided tapered periodogram of a real-valued signal.
///
/// The result is normalized by the taper energy U = sum_n w[n]^2 so that different tapers
/// can be compared on a consistent scale.
fn tapered_periodogram(
    signal: &[f64],
    taper: &[f64],
    fft_planner: &mut FftPlanner<f64>,
) -> Vec<f64> {
    let n = signal.len();
    assert_eq!(signal.len(), taper.len(), "signal and taper must have same length");

    let u = taper_energy(taper);
    assert!(u > 0.0, "taper energy must be positive");

    let mut buffer: Vec<Complex<f64>> = signal
        .iter()
        .zip(taper.iter())
        .map(|(&x, &w)| Complex::new(x * w, 0.0))
        .collect();

    let fft = fft_planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    let half = n / 2;
    let mut psd = vec![0.0; half + 1];

    for k in 0..=half {
        let power = buffer[k].norm_sqr() / u;

        // Convert to a one-sided spectrum for real-valued input.
        psd[k] = if k == 0 || (n % 2 == 0 && k == half) {
            power
        } else {
            2.0 * power
        };
    }

    psd
}

/// Form the equal-weight multitaper estimate by averaging across tapered spectra.
fn equal_weight_multitaper(tapered_psd: &[Vec<f64>]) -> Vec<f64> {
    assert!(!tapered_psd.is_empty(), "at least one tapered spectrum is required");

    let m = tapered_psd[0].len();
    let k = tapered_psd.len() as f64;

    let mut avg = vec![0.0; m];
    for spectrum in tapered_psd {
        assert_eq!(spectrum.len(), m, "all tapered spectra must have equal length");
        for (dst, &src) in avg.iter_mut().zip(spectrum.iter()) {
            *dst += src;
        }
    }

    for value in &mut avg {
        *value /= k;
    }

    avg
}

/// Compute simple adaptive multitaper weights.
///
/// The logic is intentionally transparent:
/// 1. Start from the equal-weight multitaper estimate as a pilot spectrum.
/// 2. At each frequency bin, assign larger weight to tapers whose spectral value
///    lies closer to the pilot value.
/// 3. Normalize the weights so they sum to one.
///
/// This does not attempt to reproduce a full Thomson adaptive weighting scheme.
/// Instead, it provides a practical data-dependent weighting mechanism consistent
/// with the discussion of adaptive multitaper estimation in Section 13.5.3.
fn adaptive_multitaper(tapered_psd: &[Vec<f64>]) -> Vec<f64> {
    assert!(!tapered_psd.is_empty(), "at least one tapered spectrum is required");

    let k = tapered_psd.len();
    let m = tapered_psd[0].len();
    let pilot = equal_weight_multitaper(tapered_psd);
    let mut adaptive = vec![0.0; m];

    for bin in 0..m {
        let pilot_value = pilot[bin];
        let scale = pilot_value.abs().max(1.0e-12);

        let mut weights = vec![0.0; k];
        let mut weight_sum = 0.0;

        for taper_idx in 0..k {
            let s = tapered_psd[taper_idx][bin];
            let discrepancy = (s - pilot_value).abs() / scale;

            // Larger weight for spectra closer to the pilot estimate.
            let w = 1.0 / (1.0 + discrepancy);
            weights[taper_idx] = w;
            weight_sum += w;
        }

        if weight_sum <= 0.0 {
            adaptive[bin] = pilot_value;
            continue;
        }

        let mut accum = 0.0;
        for taper_idx in 0..k {
            accum += (weights[taper_idx] / weight_sum) * tapered_psd[taper_idx][bin];
        }
        adaptive[bin] = accum;
    }

    adaptive
}

/// Compute multitaper PSD estimates using K orthogonal sine tapers.
fn multitaper_psd(signal: &[f64], num_tapers: usize) -> MultitaperPsd {
    assert!(!signal.is_empty(), "signal must be nonempty");
    assert!(
        num_tapers > 0 && num_tapers <= signal.len(),
        "number of tapers must satisfy 1 <= K <= N"
    );

    let n = signal.len();
    let tapers = sine_tapers(n, num_tapers);
    let mut fft_planner = FftPlanner::<f64>::new();

    let tapered_psd: Vec<Vec<f64>> = tapers
        .iter()
        .map(|taper| tapered_periodogram(signal, taper, &mut fft_planner))
        .collect();

    let psd_equal = equal_weight_multitaper(&tapered_psd);
    let psd_adaptive = adaptive_multitaper(&tapered_psd);

    let freqs: Vec<f64> = (0..psd_equal.len())
        .map(|idx| idx as f64 / n as f64)
        .collect();

    MultitaperPsd {
        freqs,
        psd_equal,
        psd_adaptive,
        tapered_psd,
        tapers,
        num_tapers,
        signal_len: n,
    }
}

/// Generate a synthetic signal with closely spaced components, a low-frequency component,
/// and a weak disturbance. This makes the contrast between single-window and multitaper
/// behavior easier to inspect.
fn synthetic_signal(n: usize) -> Vec<f64> {
    let f1 = 0.085;
    let f2 = 0.110;
    let f3 = 0.245;
    let f_low = 0.020;

    (0..n)
        .map(|i| {
            let t = i as f64;

            let s1 = 1.00 * (2.0 * PI * f1 * t).sin();
            let s2 = 0.70 * (2.0 * PI * f2 * t + 0.25).sin();
            let s3 = 0.45 * (2.0 * PI * f3 * t + 1.10).cos();
            let low = 0.80 * (2.0 * PI * f_low * t).sin();

            // Deterministic disturbance to make the example reproducible.
            let disturbance = 0.10 * (2.0 * PI * 0.370 * t).sin()
                + 0.05 * (2.0 * PI * 0.417 * t + 0.8).cos()
                + 0.03 * (2.0 * PI * 0.009 * t * t / n as f64).sin();

            s1 + s2 + s3 + low + disturbance
        })
        .collect()
}

/// Compute the one-sided rectangular-window periodogram of a real-valued signal.
/// This is included for comparison against the multitaper estimate.
fn rectangular_periodogram(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let taper = vec![1.0; n];
    let mut planner = FftPlanner::<f64>::new();
    tapered_periodogram(signal, &taper, &mut planner)
}

/// Print the largest peaks in a spectrum.
fn print_top_peaks(label: &str, freqs: &[f64], psd: &[f64], top_k: usize) {
    let mut pairs: Vec<(usize, f64)> = psd.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("{label}");
    for (rank, (idx, value)) in pairs.into_iter().take(top_k).enumerate() {
        println!(
            "  {:>2}. bin {:>4}, frequency {:>8.5} cycles/sample, PSD {:>12.6}",
            rank + 1,
            idx,
            freqs[idx],
            value
        );
    }
}

/// Print a compact table of the first several bins of multiple spectra.
fn print_comparison_table(
    freqs: &[f64],
    rect: &[f64],
    mt_equal: &[f64],
    mt_adaptive: &[f64],
    bins_to_show: usize,
) {
    let count = bins_to_show
        .min(freqs.len())
        .min(rect.len())
        .min(mt_equal.len())
        .min(mt_adaptive.len());

    println!("\nFirst {count} one-sided spectral bins:");
    println!(
        "{:>8} {:>14} {:>16} {:>16} {:>16}",
        "k", "frequency", "rectangular", "mt_equal", "mt_adaptive"
    );

    for k in 0..count {
        println!(
            "{:>8} {:>14.6} {:>16.8} {:>16.8} {:>16.8}",
            k, freqs[k], rect[k], mt_equal[k], mt_adaptive[k]
        );
    }
}

/// Print diagnostics confirming near-orthogonality of the sine tapers.
fn print_taper_orthogonality(tapers: &[Vec<f64>]) {
    println!("\nTaper Orthogonality Check:");
    for i in 0..tapers.len() {
        for j in 0..tapers.len() {
            let dot: f64 = tapers[i]
                .iter()
                .zip(tapers[j].iter())
                .map(|(&a, &b)| a * b)
                .sum();

            print!("{:>11.6} ", dot);
        }
        println!();
    }
}

fn main() {
    let n = 1024;
    let k = 6;

    let signal = synthetic_signal(n);
    let rect_psd = rectangular_periodogram(&signal);
    let mt = multitaper_psd(&signal, k);

    println!("Multitaper Spectral Estimation");
    println!("------------------------------");
    println!("Signal length          : {}", mt.signal_len);
    println!("Number of tapers       : {}", mt.num_tapers);
    println!("One-sided bins         : {}", mt.psd_equal.len());
    println!("Taper family           : Orthogonal sine tapers");
    println!("Comparison spectrum    : Rectangular-window periodogram");

    print_taper_orthogonality(&mt.tapers);

    println!("\nPeak Comparison:");
    print_top_peaks("Rectangular Periodogram:", &mt.freqs, &rect_psd, 8);
    print_top_peaks("Equal-Weight Multitaper Estimate:", &mt.freqs, &mt.psd_equal, 8);
    print_top_peaks("Adaptive Multitaper Estimate:", &mt.freqs, &mt.psd_adaptive, 8);

    print_comparison_table(
        &mt.freqs,
        &rect_psd,
        &mt.psd_equal,
        &mt.psd_adaptive,
        20,
    );

    println!("\nInterpretation:");
    println!(
        "The multitaper estimates average several orthogonal tapered spectra, reducing variance"
    );
    println!(
        "relative to the rectangular periodogram. The adaptive estimate further reweights"
    );
    println!(
        "individual tapered spectra according to their agreement with a pilot spectrum,"
    );
    println!(
        "illustrating a simple form of data-dependent spectral concentration."
    );
}
```

Program 13.5.3 demonstrates a practical approach to spectral estimation using multiple orthogonal tapers and FFT-based computation. This approach reflects the central computational objective discussed in Section 13.5.3: achieving improved spectral concentration while balancing bias, variance, and effective resolution.

The comparison between the rectangular periodogram and the multitaper estimates illustrates two key improvements. First, averaging across multiple tapered spectra reduces variance and produces a more stable spectral estimate. Second, the use of orthogonal tapers distributes spectral energy more smoothly across nearby frequency bins, mitigating leakage effects that are prominent in single-window approaches. The adaptive multitaper estimate further enhances robustness by weighting individual tapered spectra according to their agreement with a pilot estimate, particularly benefiting regions where spectral estimation is difficult.

The modular structure of the implementation allows the framework to be extended to more advanced multitaper techniques. In particular, the sine tapers used here can be replaced by discrete prolate spheroidal sequences to achieve optimal spectral concentration, and the adaptive weighting scheme can be refined to incorporate eigenvalue-based weighting strategies. These extensions connect directly to modern developments in spectral analysis, including applications in geophysics, signal processing, and gravitational-wave data analysis, where multitaper methods remain a powerful and widely used tool.

## 13.5.4. Small-Sample, Irregularly Sampled, and Online PSD Estimation

Although FFT-based periodograms and multitaper methods are powerful, they are not always the best choice. When the available data record is short, direct FFT-based estimators can be unstable or too noisy. A strong alternative is to adopt a parametric model for the process. In autoregressive spectral estimation, one models the signal as an AR(p) process and infers the PSD from the AR coefficients. This viewpoint is often called maximum entropy spectral analysis or Burg/MESA estimation. Its practical importance is that it can yield smoother and lower-variance spectral estimates when the sample size is modest. Recent work in gravitational-wave data analysis argues that Burg/MESA methods can outperform Welch-style estimators in regimes with only a few thousand samples, and explicitly links PSD estimation to whitening-filter construction through the AR representation (Martini et al., 2024). This is a valuable lesson for a numerical computing text: even within an FFT-centered chapter, the best estimator may not always be the most direct FFT periodogram.

A second complication arises when the data are not sampled on a uniform grid. Standard FFT estimators assume equally spaced samples, but many modern data sources contain gaps, irregular observation times, or event-driven sampling patterns. In such cases, one must either interpolate to a regular grid, reformulate the estimator, or replace the FFT with a nonuniform fast Fourier transform. Recent work has proposed high-pass-filter periodograms for unevenly sampled data in order to reduce sampling-induced artifacts and improve PSD estimation relative to more traditional irregular-sampling baselines (Albentosa-Ruiz and Marchili, 2024). Related work combines multitaper principles with NUFFT-based acceleration, yielding methods such as multiband multitaper NUFFT estimators intended to retain strong statistical properties while remaining computationally feasible for nonuniform data (Cui, Brinkmann and Worrell, 2026). These developments reinforce a recurring chapter theme: the fast transform is not tied to a single sampling geometry, but rather serves as a computational engine whose precise form must match the data model.

A third major direction concerns online or streaming PSD estimation. In many real systems, one does not have the luxury of storing a long complete record and then processing it offline. Instead, spectral estimates must be updated continuously as new data arrive. This motivates estimators based on short FFTs, overlapping windows, forgetting factors, or online Whittle-type procedures. Recent work develops online spectral density estimators designed explicitly for fixed-memory operation and temporal adaptivity (Kazi, Adams and Cohen, 2025). From an implementation standpoint, such methods influence system design choices in an important way. If the PSD must be updated continually, then one often prefers incremental estimators combined with short FFT blocks rather than repeated transforms of an ever-growing record.

These modern directions are not marginal refinements. They illustrate the broader principle that PSD estimation is not a single formula, but a family of statistically and computationally tailored strategies. The FFT remains central because it makes repeated spectral computations feasible, but the estimator must still be matched to the data regime: long or short records, uniform or irregular sampling, offline or online processing, narrowband or broadband structure.

From the perspective of this chapter, power spectrum estimation occupies a pivotal role. It is not only a descriptive tool for analyzing stochastic signals, but also a foundational computational ingredient for Wiener filtering, whitening, detection, and adaptive spectral algorithms. The numerical task is therefore twofold: compute the spectrum efficiently, and compute it in a statistically meaningful way. The first goal is achieved by the FFT. The second requires careful choices of windowing, averaging, tapering, parametric modeling, or adaptive updating. The success of spectral numerical methods depends on both.

### Rust Implementation

Following the discussion in Section 13.5.4 on small-sample, irregularly sampled, and online power spectral density estimation, Program 13.5.4 provides a unified implementation of three complementary approaches tailored to different data regimes. While earlier sections emphasized FFT-based periodograms and multitaper methods, this program illustrates situations in which alternative formulations become preferable. In particular, it implements autoregressive spectral estimation for short records, a Lomb–Scargle-type estimator for irregular sampling, and an online Welch-style estimator for streaming data. By integrating these methods within a single computational framework, the program demonstrates how spectral estimation must adapt to the structure of the data rather than relying on a single universal technique.

At the core of the implementation is the autoregressive modeling framework, realized through the functions `autocorrelation_biased`, `solve_linear_system`, and `fit_ar_yule_walker`. These functions collectively construct and solve the Yule–Walker system, which determines the AR coefficients used to define the parametric spectrum. The resulting model encodes the signal as an AR(p) process, and the function `ar_psd` evaluates the corresponding spectral density. This implementation reflects the parametric viewpoint described in Section 13.5.4, where the spectrum is inferred from a model rather than computed directly from a transform. The use of a Toeplitz system and its solution highlights the connection between statistical estimation and linear algebra in spectral analysis.

The treatment of irregularly sampled data is handled by the function `lomb_scargle_periodogram`, which computes a frequency-domain estimate without requiring interpolation onto a uniform grid. This function evaluates the spectral content by projecting the data onto sinusoidal components while compensating for uneven sampling through a phase shift parameter. In contrast to FFT-based estimators, this approach directly accommodates nonuniform observation times and avoids the distortions that can arise from resampling. This reflects the second major theme of Section 13.5.4, namely that the computational method must align with the sampling geometry.

The online or streaming setting is addressed by the `OnlineWelchEstimator` structure, which maintains a rolling estimate of the spectrum using short overlapping blocks and an exponential forgetting factor. The method `process_samples` updates the estimate incrementally as new data arrive, while `block_periodogram` computes the contribution of each block using a windowed FFT. The use of a forgetting factor allows the estimator to adapt to changes in the signal over time, reflecting the need for temporal responsiveness in streaming applications. This implementation captures the essence of online PSD estimation discussed in Section 13.5.4, where fixed-memory operation and incremental updates are essential.

The program also includes synthetic data generators for each regime. The function `synthetic_short_uniform_signal` produces a short uniformly sampled signal suitable for AR modeling, `synthetic_irregular_samples` generates unevenly spaced observations for testing the Lomb–Scargle estimator, and `synthetic_stream_signal` produces a time-varying signal that motivates online updating. Auxiliary functions such as `print_top_peaks` and `print_first_bins` provide diagnostic insight into the resulting spectra, enabling direct comparison of the estimators.

The `main` function demonstrates the behavior of all three methods in their intended settings. It first applies AR modeling to a short data record, then evaluates the Lomb–Scargle estimator on irregularly sampled data, and finally processes a streaming signal using the online Welch estimator. By comparing the outputs, the program illustrates how different estimation strategies reveal spectral structure under varying constraints. This provides a concrete computational realization of the broader principle emphasized in Section 13.5.4: spectral estimation is a family of methods, each adapted to a particular data regime.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
rustfft = "6"
```

```rust
// Program 13.5.4: Small-Sample, Irregularly Sampled, and Online PSD Estimation
//
// Problem Statement:
// Implement three practical power spectral density estimators matched to different data regimes
// discussed in Section 13.5.4.
//
// 1. Small-sample parametric PSD estimation:
//    Fit an autoregressive AR(p) model to a short uniformly sampled record using the
//    Yule-Walker equations, then evaluate the implied PSD
//
//        S_xx(f) = sigma_e^2 / |1 - sum_{k=1}^p a_k exp(-i 2*pi*f*k*dt)|^2 .
//
// 2. Irregularly sampled PSD estimation:
//    Compute a Lomb-Scargle-type periodogram for uneven observation times, avoiding the need
//    to resample onto a uniform grid.
//
// 3. Online PSD estimation:
//    Process a uniformly sampled stream in short overlapping FFT blocks, update the PSD
//    incrementally with an exponential forgetting factor, and maintain fixed memory usage.
//
// The program demonstrates how spectral estimation must be adapted to the sampling geometry,
// data length, and processing mode rather than relying on a single estimator for all settings.
//
// Cargo.toml dependencies:
//
// [dependencies]
// rustfft = "6"

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

// -----------------------------
// Basic data containers
// -----------------------------

#[derive(Debug)]
struct ArModel {
    order: usize,
    coeffs: Vec<f64>,
    noise_variance: f64,
}

#[derive(Debug)]
struct Spectrum {
    freqs: Vec<f64>,
    values: Vec<f64>,
}


struct OnlineWelchEstimator {
    block_len: usize,
    overlap: usize,
    hop: usize,
    forgetting: f64,
    window: Vec<f64>,
    window_energy: f64,
    planner: FftPlanner<f64>,
    buffer: Vec<f64>,
    psd: Vec<f64>,
    blocks_processed: usize,
}

// -----------------------------
// Utilities
// -----------------------------

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

fn remove_mean(x: &[f64]) -> Vec<f64> {
    let mu = mean(x);
    x.iter().map(|&v| v - mu).collect()
}

fn hann_window(n: usize) -> Vec<f64> {
    assert!(n > 0, "window length must be positive");
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos())
        .collect()
}

fn window_energy(window: &[f64]) -> f64 {
    window.iter().map(|&w| w * w).sum()
}

fn one_sided_freqs(n: usize, dt: f64) -> Vec<f64> {
    let half = n / 2;
    (0..=half).map(|k| k as f64 / (n as f64 * dt)).collect()
}

fn print_top_peaks(label: &str, spectrum: &Spectrum, top_k: usize) {
    let mut pairs: Vec<(usize, f64)> = spectrum.values.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("{label}");
    for (rank, (idx, value)) in pairs.into_iter().take(top_k).enumerate() {
        println!(
            "  {:>2}. f = {:>9.5} Hz, PSD = {:>12.6}",
            rank + 1,
            spectrum.freqs[idx],
            value
        );
    }
}

fn print_first_bins(label: &str, spectrum: &Spectrum, count: usize) {
    println!("\n{label}");
    println!("{:>6} {:>12} {:>14}", "k", "frequency", "PSD");
    for k in 0..count.min(spectrum.values.len()) {
        println!(
            "{:>6} {:>12.6} {:>14.8}",
            k, spectrum.freqs[k], spectrum.values[k]
        );
    }
}

// -----------------------------
// Small-sample AR PSD estimation
// -----------------------------

fn autocorrelation_biased(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let mut r = vec![0.0; max_lag + 1];

    for lag in 0..=max_lag {
        let mut sum = 0.0;
        for i in lag..n {
            sum += x[i] * x[i - lag];
        }
        r[lag] = sum / n as f64;
    }

    r
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    assert_eq!(a.len(), n, "matrix row mismatch");
    for row in &a {
        assert_eq!(row.len(), n, "matrix must be square");
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

        assert!(pivot_val > 1.0e-14, "singular or ill-conditioned system");

        if pivot_row != k {
            a.swap(k, pivot_row);
            b.swap(k, pivot_row);
        }

        let pivot = a[k][k];
        for j in k..n {
            a[k][j] /= pivot;
        }
        b[k] /= pivot;

        for i in 0..n {
            if i == k {
                continue;
            }
            let factor = a[i][k];
            for j in k..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    b
}

fn fit_ar_yule_walker(signal: &[f64], order: usize) -> ArModel {
    assert!(signal.len() > order, "signal too short for requested AR order");
    assert!(order > 0, "AR order must be positive");

    let x = remove_mean(signal);
    let r = autocorrelation_biased(&x, order);

    let mut toeplitz = vec![vec![0.0; order]; order];
    for i in 0..order {
        for j in 0..order {
            toeplitz[i][j] = r[i.abs_diff(j)];
        }
    }

    let rhs = r[1..=order].to_vec();
    let coeffs = solve_linear_system(toeplitz, rhs);

    let mut sigma2 = r[0];
    for k in 0..order {
        sigma2 -= coeffs[k] * r[k + 1];
    }
    sigma2 = sigma2.max(1.0e-12);

    ArModel {
        order,
        coeffs,
        noise_variance: sigma2,
    }
}

fn ar_psd(model: &ArModel, dt: f64, n_freqs: usize) -> Spectrum {
    assert!(n_freqs >= 2, "need at least two frequency samples");

    let nyquist = 0.5 / dt;
    let mut freqs = Vec::with_capacity(n_freqs);
    let mut values = Vec::with_capacity(n_freqs);

    for i in 0..n_freqs {
        let f = nyquist * i as f64 / (n_freqs - 1) as f64;
        let omega = 2.0 * PI * f * dt;

        let mut denom = Complex::new(1.0, 0.0);
        for (k, &a_k) in model.coeffs.iter().enumerate() {
            let kk = (k + 1) as f64;
            let z = Complex::from_polar(1.0, -omega * kk);
            denom -= a_k * z;
        }

        let psd = model.noise_variance / denom.norm_sqr().max(1.0e-14);
        freqs.push(f);
        values.push(psd);
    }

    Spectrum { freqs, values }
}

// -----------------------------
// Irregular sampling: Lomb-Scargle-type PSD
// -----------------------------

fn lomb_scargle_periodogram(times: &[f64], values: &[f64], freqs: &[f64]) -> Spectrum {
    assert_eq!(times.len(), values.len(), "times and values length mismatch");
    assert!(!times.is_empty(), "input must be nonempty");

    let y = remove_mean(values);
    let variance = mean(&y.iter().map(|v| v * v).collect::<Vec<_>>()).max(1.0e-12);

    let mut psd = Vec::with_capacity(freqs.len());

    for &f in freqs {
        let omega = 2.0 * PI * f;
        if omega.abs() < 1.0e-15 {
            psd.push(0.0);
            continue;
        }

        let s2wt: f64 = times.iter().map(|&t| (2.0 * omega * t).sin()).sum();
        let c2wt: f64 = times.iter().map(|&t| (2.0 * omega * t).cos()).sum();
        let tau = 0.5 * s2wt.atan2(c2wt) / omega;

        let mut yc = 0.0;
        let mut ys = 0.0;
        let mut cc = 0.0;
        let mut ss = 0.0;

        for (&t, &yy) in times.iter().zip(y.iter()) {
            let c = (omega * (t - tau)).cos();
            let s = (omega * (t - tau)).sin();
            yc += yy * c;
            ys += yy * s;
            cc += c * c;
            ss += s * s;
        }

        let p = 0.5 * ((yc * yc) / cc.max(1.0e-14) + (ys * ys) / ss.max(1.0e-14)) / variance;
        psd.push(p);
    }

    Spectrum {
        freqs: freqs.to_vec(),
        values: psd,
    }
}

// -----------------------------
// Online / streaming Welch PSD
// -----------------------------

impl OnlineWelchEstimator {
    fn new(block_len: usize, overlap: usize, forgetting: f64) -> Self {
        assert!(block_len > 0, "block length must be positive");
        assert!(overlap < block_len, "overlap must satisfy overlap < block length");
        assert!(
            (0.0..=1.0).contains(&forgetting),
            "forgetting factor must lie in [0, 1]"
        );

        let window = hann_window(block_len);
        let window_energy = window_energy(&window);
        let psd_len = block_len / 2 + 1;

        Self {
            block_len,
            overlap,
            hop: block_len - overlap,
            forgetting,
            window,
            window_energy,
            planner: FftPlanner::<f64>::new(),
            buffer: Vec::new(),
            psd: vec![0.0; psd_len],
            blocks_processed: 0,
        }
    }

    fn process_samples(&mut self, samples: &[f64]) {
        self.buffer.extend_from_slice(samples);

        while self.buffer.len() >= self.block_len {
            let block: Vec<f64> = self.buffer[..self.block_len].to_vec();
            let periodogram = self.block_periodogram(&block);

            if self.blocks_processed == 0 {
                self.psd.clone_from(&periodogram);
            } else {
                for (acc, &p) in self.psd.iter_mut().zip(periodogram.iter()) {
                    *acc = self.forgetting * *acc + (1.0 - self.forgetting) * p;
                }
            }

            self.blocks_processed += 1;
            self.buffer.drain(..self.hop);
        }
    }

    fn block_periodogram(&mut self, block: &[f64]) -> Vec<f64> {
        let centered = remove_mean(block);
        let mut fft_buffer: Vec<Complex<f64>> = centered
            .iter()
            .zip(self.window.iter())
            .map(|(&x, &w)| Complex::new(x * w, 0.0))
            .collect();

        let fft = self.planner.plan_fft_forward(self.block_len);
        fft.process(&mut fft_buffer);

        let half = self.block_len / 2;
        let mut psd = vec![0.0; half + 1];

        for k in 0..=half {
            let power = fft_buffer[k].norm_sqr() / self.window_energy;
            psd[k] = if k == 0 || (self.block_len % 2 == 0 && k == half) {
                power
            } else {
                2.0 * power
            };
        }

        psd
    }

    fn spectrum(&self, dt: f64) -> Spectrum {
        Spectrum {
            freqs: one_sided_freqs(self.block_len, dt),
            values: self.psd.clone(),
        }
    }
}

// -----------------------------
// Synthetic test signals
// -----------------------------

fn synthetic_short_uniform_signal(n: usize, dt: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 * dt;
            let s1 = 1.0 * (2.0 * PI * 8.0 * t).sin();
            let s2 = 0.55 * (2.0 * PI * 18.0 * t + 0.4).sin();
            let s3 = 0.20 * (2.0 * PI * 30.0 * t).cos();

            // Deterministic disturbance for reproducibility.
            let disturbance = 0.12 * (2.0 * PI * 3.2 * t).sin()
                + 0.08 * (2.0 * PI * 27.5 * t + 1.1).cos();

            s1 + s2 + s3 + disturbance
        })
        .collect()
}

fn synthetic_irregular_samples(n: usize, dt_mean: f64) -> (Vec<f64>, Vec<f64>) {
    let mut times = Vec::with_capacity(n);
    let mut t = 0.0;

    for i in 0..n {
        // Deterministic irregular spacing and occasional larger gaps.
        let jitter = 0.30 * dt_mean * (2.0 * PI * i as f64 / 23.0).sin();
        let gap = if i % 41 == 0 { 1.8 * dt_mean } else { 0.0 };
        let step = (dt_mean + jitter + gap).max(0.15 * dt_mean);
        t += step;
        times.push(t);
    }

    let values: Vec<f64> = times
        .iter()
        .map(|&tt| {
            let s1 = 1.00 * (2.0 * PI * 2.2 * tt).sin();
            let s2 = 0.70 * (2.0 * PI * 5.7 * tt + 0.3).cos();
            let s3 = 0.25 * (2.0 * PI * 0.7 * tt).sin();
            let disturbance = 0.08 * (2.0 * PI * 8.4 * tt + 1.0).sin();
            s1 + s2 + s3 + disturbance
        })
        .collect();

    (times, values)
}

fn synthetic_stream_signal(n: usize, dt: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 * dt;

            // Time-varying mixture to motivate online adaptation.
            let amp = if i < n / 2 { 1.0 } else { 0.55 };
            let s1 = amp * (2.0 * PI * 10.0 * t).sin();
            let s2 = 0.60 * (2.0 * PI * 24.0 * t + 0.25).cos();
            let s3 = if i > n / 2 {
                0.45 * (2.0 * PI * 34.0 * t).sin()
            } else {
                0.15 * (2.0 * PI * 34.0 * t).sin()
            };
            let disturbance = 0.05 * (2.0 * PI * 4.0 * t).sin();
            s1 + s2 + s3 + disturbance
        })
        .collect()
}

// -----------------------------
// Main demonstration
// -----------------------------

fn main() {
    // ==========================================================
    // 1. Small-sample parametric PSD estimation with AR modeling
    // ==========================================================
    let dt_uniform = 0.01;
    let short_signal = synthetic_short_uniform_signal(128, dt_uniform);
    let ar_order = 10;
    let ar_model = fit_ar_yule_walker(&short_signal, ar_order);
    let ar_spectrum = ar_psd(&ar_model, dt_uniform, 256);

    println!("Small-Sample AR PSD Estimation");
    println!("------------------------------");
    println!("Signal length        : {}", short_signal.len());
    println!("Sampling interval    : {}", dt_uniform);
    println!("AR order             : {}", ar_model.order);
    println!("Noise variance       : {:.8}", ar_model.noise_variance);
    println!("AR coefficients      : {:?}", ar_model.coeffs);

    print_top_peaks("Top AR PSD Peaks:", &ar_spectrum, 6);
    print_first_bins("First AR PSD Bins:", &ar_spectrum, 12);

    // ==========================================================
    // 2. Irregularly sampled PSD estimation
    // ==========================================================
    let (times_irregular, values_irregular) = synthetic_irregular_samples(160, 0.05);
    let max_freq = 10.0;
    let freq_grid: Vec<f64> = (0..300)
        .map(|i| max_freq * i as f64 / 299.0)
        .collect();
    let ls_spectrum = lomb_scargle_periodogram(&times_irregular, &values_irregular, &freq_grid);

    println!("\nIrregularly Sampled PSD Estimation");
    println!("----------------------------------");
    println!("Number of samples     : {}", values_irregular.len());
    println!(
        "Time span             : {:.6}",
        times_irregular.last().copied().unwrap_or(0.0) - times_irregular[0]
    );
    println!("Frequency grid points : {}", ls_spectrum.freqs.len());

    print_top_peaks("Top Lomb-Scargle Peaks:", &ls_spectrum, 6);
    print_first_bins("First Lomb-Scargle Bins:", &ls_spectrum, 12);

    // ==========================================================
    // 3. Online / streaming PSD estimation
    // ==========================================================
    let dt_stream = 0.005;
    let stream = synthetic_stream_signal(4096, dt_stream);

    let mut online = OnlineWelchEstimator::new(256, 128, 0.90);

    // Feed the stream in chunks to mimic online arrival.
    let chunk_size = 40;
    for chunk in stream.chunks(chunk_size) {
        online.process_samples(chunk);
    }

    let online_spectrum = online.spectrum(dt_stream);

    println!("\nOnline PSD Estimation");
    println!("---------------------");
    println!("Stream length         : {}", stream.len());
    println!("Sampling interval     : {}", dt_stream);
    println!("Block length          : {}", online.block_len);
    println!("Overlap               : {}", online.overlap);
    println!("Hop size              : {}", online.hop);
    println!("Forgetting factor     : {}", online.forgetting);
    println!("Blocks processed      : {}", online.blocks_processed);

    print_top_peaks("Top Online PSD Peaks:", &online_spectrum, 6);
    print_first_bins("First Online PSD Bins:", &online_spectrum, 12);

    println!("\nInterpretation:");
    println!(
        "The AR estimator targets short-record regimes by replacing direct periodograms with a"
    );
    println!(
        "parametric spectral model. The Lomb-Scargle estimator handles irregular sampling without"
    );
    println!(
        "forcing the data onto a uniform grid. The online Welch-style estimator updates the PSD"
    );
    println!(
        "incrementally in short overlapping blocks with fixed memory, making it suitable for"
    );
    println!("streaming or adaptive settings.");
}
```

Program 13.5.4 demonstrates a practical approach to spectral estimation by combining parametric modeling, irregular-sampling techniques, and online updating within a unified numerical framework. This approach reflects the central computational challenge discussed in Section 13.5.4: selecting an estimator that is both computationally efficient and statistically appropriate for the available data.

The three implementations illustrate distinct but complementary strategies. The AR-based estimator provides smooth, low-variance spectra in small-sample settings by leveraging a parametric model. The Lomb–Scargle estimator enables accurate spectral analysis of irregularly sampled data without introducing interpolation artifacts. The online Welch-style estimator supports continuous spectral estimation in streaming environments by combining short FFTs with incremental updates. Together, these methods highlight how spectral estimation must be adapted to data characteristics such as sample size, sampling geometry, and temporal availability.

The modular structure of the code allows each component to be extended independently. For example, higher-order AR models, alternative irregular-sampling estimators, or more sophisticated adaptive weighting schemes can be incorporated with minimal changes. This flexibility provides a foundation for more advanced spectral methods, including Bayesian spectral estimation, NUFFT-based approaches, and adaptive filtering techniques. In this way, the program connects classical spectral analysis with modern computational developments, reinforcing the broader theme that efficient computation and statistical modeling must work together to produce reliable spectral estimates.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/CEE7NYJInJBzcFx4lcUi.8","tags":[]}

# 13.6. Digital Filtering in the Time Domain

FFT-based filtering is extremely powerful when signals are processed in blocks or when convolution kernels are very long. However, many real systems require filtering that operates directly on a stream of samples. In such settings the goal is not only spectral accuracy but also bounded latency, numerical robustness, and predictable per-sample computational cost. These requirements lead naturally to time-domain digital filters. The theory of such filters predates modern FFT pipelines, yet it remains essential in contemporary numerical computing, especially for control systems, embedded signal processing, and adaptive filtering.

From a mathematical perspective, time-domain filtering corresponds to evaluating the convolution relation:

$$y[n] = (h * x)[n] \tag{13.6.1}$$

directly through recursive or finite summation formulas. The difference between filter types lies in the structure of the impulse response and in the numerical properties of the resulting algorithm. Modern digital signal processing systems therefore treat time-domain filters not as alternatives to spectral methods but as complementary tools. In many pipelines the signal is analyzed spectrally while the final filtering stage is implemented through stable time-domain structures.

## 13.6.1. General Linear Time-Invariant Digital Filters

A very general discrete-time linear time-invariant (LTI) filter maps an input sequence (x\[n\]) to an output sequence $y[n]$ through the difference equation:

$$y[n] = \sum_{m=0}^{M} b_m\, x[n-m] - \sum_{r=1}^{R} a_r\, y[n-r] \tag{13.6.2}$$

The coefficients $b_m$ describe the feedforward portion of the filter, while the coefficients $a_r$ determine the feedback structure. The filter behavior is more easily analyzed in the complex $z$-domain. Defining the $z$-transform of the sequences, the transfer function becomes:

$$H(z) = \frac{\sum_{m=0}^{M} b_m z^{-m}}{1 + \sum_{r=1}^{R} a_r z^{-r}} \tag{13.6.3}$$

Evaluating (13.6.3) on the unit circle $z=e^{i\omega}$ gives the frequency response:

$$H\!\left(e^{i\omega}\right) \tag{13.6.4}$$

This representation provides immediate insight into filter properties. The numerator polynomial determines zeros of the transfer function, while the denominator determines poles. Stability of the filter requires that all poles lie strictly inside the unit circle in the complex plane.

Two principal filter families arise from this formulation. If the feedback coefficients vanish so that $R=0$, the filter reduces to a finite impulse response (FIR) filter. If $R>0$, the filter becomes an infinite impulse response (IIR) filter. The distinction reflects whether the impulse response of the system terminates after finitely many samples or continues indefinitely due to recursive feedback.

### Rust Implementation

Following the discussion in Section 13.6.1 on the structure and analysis of linear time-invariant digital filters, Program 13.6.1 provides a practical implementation of time-domain filtering based on the general difference equation formulation. In real-time signal processing, filters must operate sequentially on incoming samples while maintaining numerical stability and predictable computational cost. This program implements a unified framework that evaluates the convolution relation in (13.6.1) through the recursive formulation in (13.6.2), supporting both finite impulse response (FIR) and infinite impulse response (IIR) filters within a single structure. It demonstrates how theoretical constructs such as transfer functions and impulse responses translate into concrete numerical algorithms, and highlights the importance of state management, recursion, and stability in time-domain implementations.

At the core of the implementation is the `LTIFilter` struct, which represents a general linear time-invariant digital filter defined by the coefficients in equation (13.6.2). The struct stores the feedforward coefficients $b_m$, the feedback coefficients $a_r$, and two internal buffers that maintain the history of past input and output samples. These buffers implement the delay structure implicit in the difference equation and allow the filter to operate in a streaming, sample-by-sample manner.

The `process_sample` function directly realizes the recursion in (13.6.2). For each new input sample $x[n]$ , it computes the output $y[n]$ by forming a weighted sum of the current and past inputs using the coefficients $b_m$, and subtracting a weighted sum of past outputs using the coefficients $a_r$. The delay-line buffers are then updated so that the system state is correctly propagated to the next time step. This function encapsulates the essential computational structure of time-domain filtering and ensures constant computational cost per sample.

The `process_block` function applies the filter to an entire finite sequence by repeatedly invoking `process_sample`. This provides a convenient interface for offline processing while preserving the same underlying streaming logic. The distinction between sample-wise and block processing reflects the dual use of time-domain filters in both real-time systems and batch computations.

The `impulse_response` function evaluates the system response to a discrete impulse, thereby reconstructing the sequence $h[n]$ associated with the convolution relation in (13.6.1). This is achieved by feeding a unit impulse into a reset filter and recording the resulting outputs. For FIR filters, the impulse response terminates after a finite number of samples, while for IIR filters it decays indefinitely due to recursive feedback. This function provides a direct numerical bridge between the recursive formulation in (13.6.2) and the convolution interpretation in (13.6.1).

The `frequency_response` function evaluates the transfer function defined in (13.6.3) on the unit circle, corresponding to (13.6.4). It computes the numerator and denominator polynomials using complex exponentials and returns their ratio as a complex value. This allows the user to examine magnitude and phase characteristics of the filter and connect the time-domain implementation with its frequency-domain interpretation. A lightweight `Complex` struct is implemented to support these computations without external dependencies. It provides basic arithmetic operations and functions for magnitude and phase evaluation, enabling direct computation of frequency response values.

The `main` function demonstrates the framework using two representative examples. The first example constructs an FIR moving-average filter, corresponding to the case $R = 0$ in (13.6.2), and applies it to a test signal composed of multiple sinusoidal components. The resulting impulse response confirms that the system has finite memory, and the sampled frequency response illustrates its low-pass characteristics. The second example implements a first-order IIR filter with recursive feedback, demonstrating the case $R > 0$. Its impulse response exhibits exponential decay, and its frequency response shows a smooth attenuation of high-frequency components. These examples validate the correctness of the implementation and illustrate the fundamental distinction between FIR and IIR filters discussed in the section.

```rust
// Program 13.6.1: General LTI Digital Filter
//
// Problem Statement:
// Implement a general discrete-time linear time-invariant digital filter governed by
// the difference equation
//
//     y[n] = sum_{m=0}^{M} b_m x[n-m] - sum_{r=1}^{R} a_r y[n-r] ,
//
// corresponding to equation (13.6.2) in the text. The implementation should support
// sample-by-sample streaming evaluation, batch processing of finite signals, impulse-
// response generation, and numerical evaluation of the frequency response
// H(e^{i\omega}) from equation (13.6.3).

use std::collections::VecDeque;
use std::f64::consts::PI;

/// A lightweight complex-number type so the program can run without external crates.
#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn magnitude(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    fn phase(self) -> f64 {
        self.im.atan2(self.re)
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }
#[allow(dead_code)]
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
#[allow(dead_code)]
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    fn div(self, other: Self) -> Self {
        let denom = other.re * other.re + other.im * other.im;
        Self::new(
            (self.re * other.re + self.im * other.im) / denom,
            (self.im * other.re - self.re * other.im) / denom,
        )
    }

    fn scale(self, alpha: f64) -> Self {
        Self::new(alpha * self.re, alpha * self.im)
    }

    fn exp_i(theta: f64) -> Self {
        Self::new(theta.cos(), theta.sin())
    }
}

/// General LTI filter implementing equation (13.6.2).
///
/// The coefficient vectors are interpreted as:
/// - b[0..=M] for the feedforward part
/// - a[1..=R] for the feedback part, stored as a[0..R-1]
///
/// Thus the update is
///
///     y[n] = b_0 x[n] + ... + b_M x[n-M]
///            - a_1 y[n-1] - ... - a_R y[n-R].
///
/// This sign convention matches equation (13.6.2).
#[derive(Clone, Debug)]
struct LTIFilter {
    b: Vec<f64>,
    a: Vec<f64>,
    x_hist: VecDeque<f64>,
    y_hist: VecDeque<f64>,
}

impl LTIFilter {
    /// Create a new LTI filter with numerator coefficients b and denominator
    /// feedback coefficients a.
    fn new(b: Vec<f64>, a: Vec<f64>) -> Self {
        assert!(
            !b.is_empty(),
            "The feedforward coefficient vector b must be nonempty."
        );

        let x_hist = VecDeque::from(vec![0.0; b.len()]);
        let y_hist = VecDeque::from(vec![0.0; a.len()]);

        Self {
            b,
            a,
            x_hist,
            y_hist,
        }
    }

    /// Reset the internal state to zero.
    fn reset(&mut self) {
        for x in self.x_hist.iter_mut() {
            *x = 0.0;
        }
        for y in self.y_hist.iter_mut() {
            *y = 0.0;
        }
    }

    /// Process a single sample and return the corresponding output sample.
    ///
    /// This realizes the recursion in equation (13.6.2) in streaming form.
    fn process_sample(&mut self, x_n: f64) -> f64 {
        self.x_hist.pop_back();
        self.x_hist.push_front(x_n);

        let mut y_n = 0.0;

        // Feedforward contribution: sum_{m=0}^{M} b_m x[n-m]
        for (bm, &x_delayed) in self.b.iter().zip(self.x_hist.iter()) {
            y_n += bm * x_delayed;
        }

        // Feedback contribution: -sum_{r=1}^{R} a_r y[n-r]
        for (ar, &y_delayed) in self.a.iter().zip(self.y_hist.iter()) {
            y_n -= ar * y_delayed;
        }

        self.y_hist.pop_back();
        self.y_hist.push_front(y_n);

        y_n
    }

    /// Process an entire input sequence.
    fn process_block(&mut self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xn| self.process_sample(xn)).collect()
    }

    /// Compute the first n samples of the impulse response by applying the filter
    /// to the discrete impulse delta[n].
    fn impulse_response(&self, n: usize) -> Vec<f64> {
        let mut filter = self.clone();
        filter.reset();

        let mut h = Vec::with_capacity(n);
        for k in 0..n {
            let input = if k == 0 { 1.0 } else { 0.0 };
            h.push(filter.process_sample(input));
        }
        h
    }

    /// Evaluate the transfer function H(e^{i\omega}) on the unit circle,
    /// corresponding to equations (13.6.3) and (13.6.4).
    fn frequency_response(&self, omega: f64) -> Complex {
        let mut numerator = Complex::new(0.0, 0.0);
        let mut denominator = Complex::new(1.0, 0.0);

        for (m, &bm) in self.b.iter().enumerate() {
            let z_pow = Complex::exp_i(-(m as f64) * omega);
            numerator = numerator.add(z_pow.scale(bm));
        }

        for (r, &ar) in self.a.iter().enumerate() {
            let z_pow = Complex::exp_i(-((r + 1) as f64) * omega);
            denominator = denominator.add(z_pow.scale(ar));
        }

        numerator.div(denominator)
    }
}

/// Utility function: print a real-valued sequence with indices.
fn print_sequence(name: &str, x: &[f64]) {
    println!("{name}");
    for (n, &value) in x.iter().enumerate() {
        println!("  {:>3}: {:>.10}", n, value);
    }
    println!();
}

/// Generate a test signal composed of two sinusoids.
fn generate_test_signal(n_samples: usize) -> Vec<f64> {
    let mut x = Vec::with_capacity(n_samples);

    for n in 0..n_samples {
        let t = n as f64;
        let low = (0.12 * PI * t).sin();
        let high = 0.35 * (0.72 * PI * t).sin();
        x.push(low + high);
    }

    x
}

fn main() {
    // Example 1:
    // A short FIR moving-average filter. This is the case R = 0 in equation (13.6.2).
    let b_fir = vec![0.25, 0.25, 0.25, 0.25];
    let a_fir = vec![];
    let mut fir_filter = LTIFilter::new(b_fir, a_fir);

    let x = generate_test_signal(24);
    let y_fir = fir_filter.process_block(&x);

    println!("=== General LTI Digital Filter Demonstration ===\n");
    println!("Example 1: FIR filter (feedback coefficients vanish, so R = 0).\n");
    print_sequence("Input x[n]:", &x);
    print_sequence("Filtered output y[n]:", &y_fir);

    let h_fir = fir_filter.impulse_response(12);
    print_sequence("Estimated impulse response h[n] for the FIR filter:", &h_fir);

    println!("Sampled frequency response values for the FIR filter:");
    for &omega in &[0.0, 0.25 * PI, 0.5 * PI, 0.75 * PI, PI] {
        let h = fir_filter.frequency_response(omega);
        println!(
            "  omega = {:>6.3} pi : H = ({:>.8}, {:>.8}), |H| = {:>.8}, phase = {:>.8}",
            omega / PI,
            h.re,
            h.im,
            h.magnitude(),
            h.phase()
        );
    }
    println!();

    // Example 2:
    // A simple first-order IIR low-pass filter written in the form
    //
    //     y[n] = (1 - alpha) x[n] + alpha y[n-1]
    //
    // which becomes equation (13.6.2) with
    //
    //     b_0 = 1 - alpha,   a_1 = -alpha.
    //
    let alpha = 0.85;
    let b_iir = vec![1.0 - alpha];
    let a_iir = vec![-alpha];
    let mut iir_filter = LTIFilter::new(b_iir, a_iir);

    let y_iir = iir_filter.process_block(&x);

    println!("Example 2: IIR filter (recursive feedback present, so R > 0).\n");
    print_sequence("Filtered output y[n] from the first-order IIR filter:", &y_iir);

    let h_iir = iir_filter.impulse_response(16);
    print_sequence("Estimated impulse response h[n] for the IIR filter:", &h_iir);

    println!("Sampled frequency response values for the IIR filter:");
    for &omega in &[0.0, 0.25 * PI, 0.5 * PI, 0.75 * PI, PI] {
        let h = iir_filter.frequency_response(omega);
        println!(
            "  omega = {:>6.3} pi : H = ({:>.8}, {:>.8}), |H| = {:>.8}, phase = {:>.8}",
            omega / PI,
            h.re,
            h.im,
            h.magnitude(),
            h.phase()
        );
    }
    println!();

    println!("Interpretation:");
    println!("1. The FIR example has finite memory and an impulse response that terminates.");
    println!("2. The IIR example uses feedback, so its impulse response decays but does not terminate.");
    println!("3. The same LTIFilter structure implements both cases, matching the unified");
    println!("   difference-equation form of equation (13.6.2).");
}
```

Program 13.6.1 demonstrates a practical and unified approach to implementing time-domain digital filters using the general difference equation framework. This approach reflects the central computational idea introduced in Section 13.6: evaluating convolution either through finite summation or recursive updates depending on the structure of the filter.

The FIR and IIR examples illustrate two fundamentally different system behaviors. FIR filters provide guaranteed stability and finite memory, making them straightforward to analyze and implement, while IIR filters achieve comparable frequency selectivity with significantly fewer coefficients but require careful attention to stability and numerical effects due to feedback. These contrasting properties underscore the importance of choosing an appropriate filter structure based on application requirements such as latency, computational cost, and robustness.

The modular design of the `LTIFilter` struct allows this framework to be extended naturally to more advanced filter architectures, including cascaded second-order sections, adaptive filters, and time-varying systems. It also provides a foundation for integrating time-domain filtering with spectral methods, enabling hybrid approaches that combine the efficiency of FFT-based analysis with the low-latency execution of recursive filters.

## 13.6.2. FIR Filters: Linear Phase and Optimization-Based Design

FIR filters are widely used because they guarantee stability and can be designed to exhibit exactly linear phase. If the impulse response satisfies a symmetry condition:

$$h[n] = h[M-n] \tag{13.6.5}$$

then the frequency response has the form:

$$H\!\left(e^{i\omega}\right) = A(\omega)\, e^{-i\omega M/2} \tag{13.6.6}$$

which corresponds to a pure delay combined with a real amplitude response. Linear phase is particularly valuable in audio processing, data analysis, and many scientific applications because it preserves waveform shape without introducing phase distortion.

Traditional FIR design techniques include windowed sinc approximations, frequency-sampling methods, and equiripple optimization. Modern developments often frame FIR design as an explicit optimization problem with additional constraints. For example, one may attempt to minimize passband ripple and transition width while also promoting sparsity in the coefficients. Sparsity-aware designs introduce regularization terms such as,

$$\lambda_1 \lVert b \rVert_1 + \lambda_2 \lVert b \rVert_2^2 \tag{13.6.7}$$

which balance frequency response accuracy with implementation efficiency. Sparse FIR filters reduce the number of multiplications required per output sample, an important advantage in embedded or high-throughput environments (Nerma et al., 2025).

Another modern application is variable fractional delay filtering, where the filter is designed so that the effective delay can be tuned continuously. Such filters are important in resampling, beamforming, and communication systems. Recent work develops iterative reweighted least-squares formulations for designing fractional-delay FIR filters and compares alternative structures such as complex exponential and Farrow implementations (Zhao, 2024). These methods emphasize that filter design has become increasingly algorithmic, relying on numerical optimization rather than closed-form formulas alone.

### Rust Implementation

Following the discussion in Section 13.6.2 on linear-phase FIR filters and optimization-based design strategies, Program 13.6.2 provides a practical implementation of symmetric FIR filter construction and sparsity-aware refinement. In modern digital signal processing, FIR filters are often designed not only to meet frequency-domain specifications but also to satisfy structural constraints such as symmetry and computational efficiency. This program demonstrates how the symmetry condition in equation (13.6.5) leads to linear-phase behavior, while also incorporating a simple regularization-inspired shrinkage step motivated by equation (13.6.7). It evaluates both dense and sparsified FIR filters, illustrating how optimization concepts can be integrated into time-domain implementations without sacrificing stability or interpretability.

At the core of the implementation is the `FIRFilter` struct, which represents a finite impulse response system defined by its coefficient vector $h[n]$. This structure directly implements the convolution relation in equation (13.6.1) by maintaining a history buffer of past inputs and computing each output sample as a finite weighted sum. Because FIR filters have no feedback terms, the recursion simplifies to a purely feedforward computation, ensuring stability and predictable numerical behavior.

The `process_sample` function evaluates the convolution sum for a single input sample. It updates the delay-line buffer and computes the output using the stored coefficients. This function embodies the time-domain realization of FIR filtering and guarantees a fixed number of operations per sample, which is critical in real-time applications. The `process_block` function extends this logic to process entire sequences, enabling both streaming and offline usage.

The design of the filter is handled by the `design_linear_phase_lowpass` function. This function constructs a symmetric coefficient sequence by combining an ideal low-pass response with a windowing function. The resulting coefficients satisfy the symmetry condition in equation (13.6.5), which ensures that the frequency response takes the form described in equation (13.6.6). The auxiliary functions `symmetrize` and `normalize_dc_gain` enforce exact symmetry and unity gain at zero frequency, respectively, ensuring that the designed filter adheres strictly to the theoretical requirements.

To incorporate modern optimization ideas, the program includes the `sparsify_symmetric_fir` function. This function applies a symmetry-preserving shrinkage operation inspired by the regularization framework in equation (13.6.7). By combining soft thresholding with a simple scaling factor, the method reduces the magnitude of small coefficients and sets some to zero, thereby promoting sparsity. Crucially, the operation is applied symmetrically so that the linear-phase property is preserved.

The `frequency_response` function evaluates the discrete-time Fourier transform of the coefficient sequence, corresponding to the transfer function in equation (13.6.6). By sampling this response at selected frequencies, the program provides insight into passband behavior, transition characteristics, and stopband attenuation. The custom `Complex` struct supports these computations without requiring external dependencies.

The `main` function demonstrates the complete workflow. It first constructs a dense symmetric FIR filter and then produces a sparsified version using the regularization parameters $\lambda_1$ and $\lambda_2$. It reports symmetry errors, counts effective nonzero coefficients, and verifies that the group delay matches the expected value $M/2$. The program then applies both filters to a test signal, computes impulse responses, and evaluates frequency responses. These outputs illustrate how sparsification affects both computational cost and spectral fidelity while maintaining the essential linear-phase structure.

```rust
// Program 13.6.2: Linear-Phase FIR Filter Design with Symmetry Preservation and Sparsity Promotion
//
// Problem Statement:
// Construct a finite impulse response filter for the setting of Section 13.6.2.
// The program should:
// 1. Design a symmetric FIR low-pass filter whose coefficients satisfy the
//    linear-phase symmetry condition in equation (13.6.5).
// 2. Apply a simple sparsity-promoting shrinkage step motivated by the
//    regularized design viewpoint in equation (13.6.7), while preserving symmetry.
// 3. Filter a test signal in the time domain.
// 4. Evaluate the frequency response H(e^{i\omega}) from equation (13.6.6)
//    and report quantities relevant to linear-phase behavior.

use std::collections::VecDeque;
use std::f64::consts::PI;

/// Lightweight complex number type for frequency-response calculations.
#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    fn scale(self, alpha: f64) -> Self {
        Self::new(alpha * self.re, alpha * self.im)
    }

    fn magnitude(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    fn phase(self) -> f64 {
        self.im.atan2(self.re)
    }

    fn exp_i(theta: f64) -> Self {
        Self::new(theta.cos(), theta.sin())
    }
}

/// Streaming FIR filter with sample-by-sample evaluation.
#[derive(Clone, Debug)]
struct FIRFilter {
    h: Vec<f64>,
    x_hist: VecDeque<f64>,
}

impl FIRFilter {
    fn new(h: Vec<f64>) -> Self {
        assert!(!h.is_empty(), "FIR coefficient vector must be nonempty.");
        let x_hist = VecDeque::from(vec![0.0; h.len()]);
        Self { h, x_hist }
    }

    fn reset(&mut self) {
        for x in self.x_hist.iter_mut() {
            *x = 0.0;
        }
    }

    /// Evaluate y[n] = sum_m h[m] x[n-m].
    fn process_sample(&mut self, x_n: f64) -> f64 {
        self.x_hist.pop_back();
        self.x_hist.push_front(x_n);

        self.h
            .iter()
            .zip(self.x_hist.iter())
            .map(|(hm, xm)| hm * xm)
            .sum()
    }

    fn process_block(&mut self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xn| self.process_sample(xn)).collect()
    }

    fn impulse_response(&self) -> Vec<f64> {
        self.h.clone()
    }

    /// Evaluate H(e^{i\omega}) = sum_n h[n] e^{-i \omega n}.
    fn frequency_response(&self, omega: f64) -> Complex {
        let mut h_omega = Complex::new(0.0, 0.0);
        for (n, &hn) in self.h.iter().enumerate() {
            let z = Complex::exp_i(-(n as f64) * omega);
            h_omega = h_omega.add(z.scale(hn));
        }
        h_omega
    }
}

/// Normalized sinc function: sinc(x) = sin(pi x)/(pi x).
fn sinc(x: f64) -> f64 {
    if x.abs() < 1.0e-14 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// Hamming window on n = 0, ..., m.
fn hamming_window(n: usize, m: usize) -> f64 {
    if m == 0 {
        1.0
    } else {
        0.54 - 0.46 * (2.0 * PI * n as f64 / m as f64).cos()
    }
}

/// Design a symmetric low-pass FIR filter using a windowed-sinc construction.
/// The resulting coefficients satisfy h[n] = h[M-n] numerically.
fn design_linear_phase_lowpass(num_taps: usize, cutoff_cycles_per_sample: f64) -> Vec<f64> {
    assert!(num_taps >= 3, "Use at least 3 taps.");
    assert!(
        cutoff_cycles_per_sample > 0.0 && cutoff_cycles_per_sample < 0.5,
        "Cutoff must lie in (0, 0.5)."
    );

    let m = num_taps - 1;
    let center = 0.5 * m as f64;
    let mut h = vec![0.0; num_taps];

    for n in 0..num_taps {
        let t = n as f64 - center;
        let ideal = 2.0 * cutoff_cycles_per_sample * sinc(2.0 * cutoff_cycles_per_sample * t);
        let window = hamming_window(n, m);
        h[n] = ideal * window;
    }

    normalize_dc_gain(&mut h);
    symmetrize(&mut h);
    h
}

/// Enforce exact pairwise symmetry h[n] = h[M-n].
fn symmetrize(h: &mut [f64]) {
    let n = h.len();
    for i in 0..(n / 2) {
        let j = n - 1 - i;
        let avg = 0.5 * (h[i] + h[j]);
        h[i] = avg;
        h[j] = avg;
    }
}

/// Normalize so that H(e^{i0}) = sum_n h[n] = 1.
fn normalize_dc_gain(h: &mut [f64]) {
    let s: f64 = h.iter().sum();
    if s.abs() > 1.0e-15 {
        for value in h.iter_mut() {
            *value /= s;
        }
    }
}

/// Soft-threshold operator used as a simple sparsity-promoting step.
fn soft_threshold(x: f64, lambda1: f64) -> f64 {
    if x > lambda1 {
        x - lambda1
    } else if x < -lambda1 {
        x + lambda1
    } else {
        0.0
    }
}

/// Apply a symmetry-preserving shrinkage step motivated by equation (13.6.7).
/// The parameter lambda1 promotes sparsity through soft thresholding.
/// The parameter lambda2 produces a simple ridge-like scaling.
fn sparsify_symmetric_fir(h: &[f64], lambda1: f64, lambda2: f64) -> Vec<f64> {
    assert!(lambda1 >= 0.0, "lambda1 must be nonnegative.");
    assert!(lambda2 >= 0.0, "lambda2 must be nonnegative.");

    let n = h.len();
    let mut out = h.to_vec();

    for i in 0..((n + 1) / 2) {
        let j = n - 1 - i;
        let avg = 0.5 * (h[i] + h[j]);
        let shrunk = soft_threshold(avg, lambda1) / (1.0 + lambda2);
        out[i] = shrunk;
        out[j] = shrunk;
    }

    normalize_dc_gain(&mut out);
    symmetrize(&mut out);
    out
}

fn count_effectively_nonzero(h: &[f64], tol: f64) -> usize {
    h.iter().filter(|&&v| v.abs() > tol).count()
}

fn max_symmetry_error(h: &[f64]) -> f64 {
    let n = h.len();
    let mut err: f64 = 0.0;
    for i in 0..n {
        let j = n - 1 - i;
        err = err.max((h[i] - h[j]).abs());
    }
    err
}

fn print_coefficients(title: &str, h: &[f64]) {
    println!("{title}");
    for (i, &v) in h.iter().enumerate() {
        println!("  h[{i:>2}] = {v:.10}");
    }
    println!();
}

fn generate_test_signal(n_samples: usize) -> Vec<f64> {
    let mut x = Vec::with_capacity(n_samples);
    for n in 0..n_samples {
        let t = n as f64;
        let low_component = (0.10 * PI * t).sin();
        let high_component = 0.40 * (0.74 * PI * t).sin();
        x.push(low_component + high_component);
    }
    x
}

fn print_sequence(name: &str, x: &[f64]) {
    println!("{name}");
    for (n, &value) in x.iter().enumerate() {
        println!("  {:>3}: {:.10}", n, value);
    }
    println!();
}

fn main() {
    let num_taps = 21;
    let cutoff = 0.18; // cycles per sample
    let lambda1 = 0.010;
    let lambda2 = 0.050;

    let h_dense = design_linear_phase_lowpass(num_taps, cutoff);
    let h_sparse = sparsify_symmetric_fir(&h_dense, lambda1, lambda2);

    let dense_symmetry_error = max_symmetry_error(&h_dense);
    let sparse_symmetry_error = max_symmetry_error(&h_sparse);

    let dense_nonzero = count_effectively_nonzero(&h_dense, 1.0e-8);
    let sparse_nonzero = count_effectively_nonzero(&h_sparse, 1.0e-8);

    let expected_group_delay = 0.5 * (num_taps as f64 - 1.0);

    println!("=== Linear-Phase FIR Filter Design Demonstration ===\n");
    println!("Number of taps              : {num_taps}");
    println!("Cutoff (cycles/sample)      : {cutoff:.6}");
    println!("Expected group delay M/2    : {expected_group_delay:.6}");
    println!("lambda1 (sparsity)          : {lambda1:.6}");
    println!("lambda2 (ridge-like scale)  : {lambda2:.6}");
    println!();

    println!("Dense design symmetry error : {:.3e}", dense_symmetry_error);
    println!("Sparse design symmetry error: {:.3e}", sparse_symmetry_error);
    println!("Dense effective nonzeros    : {}", dense_nonzero);
    println!("Sparse effective nonzeros   : {}", sparse_nonzero);
    println!();

    print_coefficients("Dense symmetric FIR coefficients:", &h_dense);
    print_coefficients("Sparse symmetric FIR coefficients:", &h_sparse);

    let mut dense_filter = FIRFilter::new(h_dense.clone());
    let mut sparse_filter = FIRFilter::new(h_sparse.clone());

    let x = generate_test_signal(32);
    let y_dense = dense_filter.process_block(&x);
    let y_sparse = sparse_filter.process_block(&x);

    print_sequence("Input x[n]:", &x);
    print_sequence("Dense FIR output y[n]:", &y_dense);
    print_sequence("Sparse FIR output y[n]:", &y_sparse);

    let impulse_dense = dense_filter.impulse_response();
    let impulse_sparse = sparse_filter.impulse_response();
    print_sequence("Dense filter impulse response h[n]:", &impulse_dense);
    print_sequence("Sparse filter impulse response h[n]:", &impulse_sparse);

    println!("Sampled frequency response values for the dense design:");
    for &omega in &[0.0, 0.20 * PI, 0.40 * PI, 0.60 * PI, 0.80 * PI, PI] {
        let h = dense_filter.frequency_response(omega);
        println!(
            "  omega = {:>5.2} pi : H = ({:.8}, {:.8}), |H| = {:.8}, phase = {:.8}",
            omega / PI,
            h.re,
            h.im,
            h.magnitude(),
            h.phase()
        );
    }
    println!();

    println!("Sampled frequency response values for the sparse design:");
    for &omega in &[0.0, 0.20 * PI, 0.40 * PI, 0.60 * PI, 0.80 * PI, PI] {
        let h = sparse_filter.frequency_response(omega);
        println!(
            "  omega = {:>5.2} pi : H = ({:.8}, {:.8}), |H| = {:.8}, phase = {:.8}",
            omega / PI,
            h.re,
            h.im,
            h.magnitude(),
            h.phase()
        );
    }
    println!();

    dense_filter.reset();
    sparse_filter.reset();

    println!("Interpretation:");
    println!("1. The designed coefficients satisfy the symmetry condition in equation (13.6.5).");
    println!("2. This symmetry produces linear-phase behavior with group delay approximately M/2.");
    println!("3. The shrinkage step motivated by equation (13.6.7) reduces small coefficients");
    println!("   while preserving symmetry and unity DC gain.");
    println!("4. The sparse design uses fewer effective taps, illustrating the trade-off between");
    println!("   implementation efficiency and ideal frequency-response fidelity.");
}
```

Program 13.6.2 demonstrates a practical approach to FIR filter design that combines classical signal processing techniques with modern optimization-inspired modifications. The construction of symmetric coefficients ensures linear-phase behavior, preserving waveform structure while providing predictable delay characteristics, as discussed in Section 13.6.2.

The comparison between dense and sparse filters highlights an important trade-off. The dense design achieves sharper frequency characteristics and stronger stopband attenuation, while the sparse design reduces the number of effective coefficients, thereby lowering computational cost. This reflects the role of regularization in balancing accuracy and efficiency, as emphasized in equation (13.6.7).

The modular structure of the implementation allows for straightforward extension to more advanced FIR design methods, including equiripple optimization, frequency-sampling techniques, and adaptive or time-varying filters. It also provides a foundation for integrating FIR filtering into larger numerical pipelines, where linear-phase behavior and efficient implementation are critical. In this way, the program illustrates how theoretical design principles translate into robust and flexible computational tools.

## 13.6.3. IIR Filters, Stability Constraints, and Second-Order Sections

IIR filters can achieve very sharp frequency transitions using far fewer coefficients than FIR filters. However, the presence of feedback introduces important numerical concerns. Stability requires that all poles of the transfer function lie within the unit circle,

$$|z_p|<1\tag{13.6.8}$$

and the numerical sensitivity of the recursion depends strongly on how the filter is implemented.

For high-order filters it is rarely advisable to implement (13.6.3) directly. Instead, the transfer function is factored into a cascade of second-order sections (SOS), also known as biquads. In this representation the filter is written as:

$$
H(z) =
\prod_{j=1}^{J}
\frac{b_{0j} + b_{1j} z^{-1} + b_{2j} z^{-2}}
{1 + a_{1j} z^{-1} + a_{2j} z^{-2}} \tag{13.6.9}
$$

Each section implements a second-order recursion. Cascading these sections improves numerical robustness because rounding errors remain localized within each stage rather than propagating through a large high-order polynomial. This structure is now standard in most digital signal processing implementations.

Recent research emphasizes explicit stability and delay constraints during IIR filter design. For instance, optimization-based approaches can design maximally flat stable IIR filters while controlling the maximum pole radius and limiting group delay (Yi et al., 2025). These developments are important in control and communication systems where excessive delay or marginal stability can degrade system performance.

Time-varying digital filters represent another modern extension. Instead of fixed coefficients, the filter parameters may evolve slowly over time to adapt to changing signal conditions. Optimization-based methods have been proposed for adjusting coefficients of second-order sections in order to reduce transient response while preserving stability (Okoniewski and Piskorowski, 2025). Such adaptive structures illustrate how numerical optimization techniques increasingly interact with classical filter theory.

### Rust Implementation

Following the discussion in Section 13.6.3 on the numerical realization of infinite impulse response filters, Program 13.6.3 provides a practical implementation of an IIR filter using a cascade of second-order sections rather than a single high-order direct form. In recursive digital filtering, the mathematical compactness of the transfer function must be balanced against stability, sensitivity to coefficient perturbations, and the accumulation of rounding errors during repeated state updates. This program translates the factorized representation in equation (13.6.9) into a sequence of biquad stages, each with its own local state, and demonstrates how the stability condition in equation (13.6.8) can be checked directly through pole locations. By combining sample-by-sample processing, pole analysis, impulse-response generation, and frequency-response evaluation, the implementation illustrates why second-order-section cascades are the standard computational form for robust IIR filtering.

At the core of the implementation is the `BiquadSection` struct, which represents one factor of the second-order-section decomposition in equation (13.6.9). Each section stores the numerator coefficients $b_{0j}, b_{1j}, b_{2j}$, the denominator coefficients $a_{1j}, a_{2j}$, and the local delayed input and output states required for the recursion. This localized state representation is important because it prevents the entire high-order filter from being realized through one long and numerically fragile recurrence.

The `process_sample` function in `BiquadSection` implements the second-order difference equation associated with a single SOS stage. For each incoming sample, it combines the current input, two delayed inputs, and two delayed outputs to produce the new output of that section, after which the internal state variables are shifted forward. This function is the direct computational realization of one quadratic numerator-denominator factor in equation (13.6.9), and it forms the fundamental building block of the full cascade.

The `frequency_response` function evaluates the transfer function of one biquad on the unit circle by substituting $z=e^{i\omega}$ into the numerator and denominator polynomials. This makes it possible to study the spectral contribution of each section and, when combined across all sections, to recover the response of the complete filter. The associated `Complex` struct supplies the arithmetic needed for these calculations while keeping the implementation self-contained.

The `poles` function computes the roots of the quadratic denominator polynomial associated with the section. Since the denominator in equation (13.6.9) corresponds to a quadratic expression in $z^{-1}$, the stability of the recursion can be examined through the corresponding pole locations. The `max_pole_radius` function then extracts the largest pole magnitude, while `is_stable` checks whether that radius satisfies the condition in equation (13.6.8). These functions directly connect the abstract stability requirement to an explicit numerical diagnostic.

The `SOSCascade` struct represents the full IIR filter as an ordered collection of `BiquadSection` objects. Its `process_sample` function passes each incoming sample through the cascade stage by stage, using the output of one section as the input to the next. This organization reflects the product structure in equation (13.6.9) and is precisely the arrangement preferred in practical DSP implementations because it confines roundoff effects to individual sections rather than allowing them to accumulate across a large direct-form recursion.

The `process_block` function applies the cascade to a complete input sequence while preserving the underlying streaming logic. The `impulse_response` function computes the output generated by a discrete impulse, thereby revealing the infinite-duration character of the system. In contrast to the FIR case of Section 13.6.2, the resulting response decays rather than terminating, which is a defining feature of recursive filters. The `frequency_response` function for `SOSCascade` multiplies the responses of all sections together, yielding the full transfer function of the cascade.

The helper functions `print_sequence` and `print_section_report` are included to make the behavior of the implementation transparent. In particular, `print_section_report` lists the coefficients of each biquad, the computed poles, their magnitudes, and the resulting stability verdict. This diagnostic output is especially useful in the IIR setting because numerical reliability depends not only on the intended filter shape but also on the placement of poles and the form of implementation.

The `main` function demonstrates the entire framework using a cascade of two stable low-pass style second-order sections. It begins by constructing the cascade, then reports pole locations and verifies that every section satisfies the stability condition in equation (13.6.8). Next, it filters a test signal containing multiple frequency components, computes the impulse response of the complete cascade, and evaluates the sampled frequency response at representative frequencies. The printed results show a stable decaying impulse response and strong attenuation of higher frequencies, confirming both the recursive nature of the filter and the practical effectiveness of the SOS representation.

```rust
// Program 13.6.3: IIR Filter Cascade via Second-Order Sections
//
// Problem Statement:
// Implement a stable infinite impulse response filter in the setting of Section 13.6.3
// using a cascade of second-order sections rather than a single high-order direct form.
// The program should:
// 1. Represent each biquad according to the second-order section form in equation (13.6.9).
// 2. Process samples sequentially through a cascade of sections.
// 3. Check the pole locations of each section to verify the stability condition in
//    equation (13.6.8).
// 4. Compute impulse responses and sampled frequency responses for the full cascade.
// 5. Demonstrate the filter on a test signal containing low- and high-frequency components.

use std::f64::consts::PI;

/// Lightweight complex number type for frequency-response calculations.
#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    fn div(self, other: Self) -> Self {
        let denom = other.re * other.re + other.im * other.im;
        Self::new(
            (self.re * other.re + self.im * other.im) / denom,
            (self.im * other.re - self.re * other.im) / denom,
        )
    }

    fn scale(self, alpha: f64) -> Self {
        Self::new(alpha * self.re, alpha * self.im)
    }

    fn magnitude(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    fn phase(self) -> f64 {
        self.im.atan2(self.re)
    }

    fn exp_i(theta: f64) -> Self {
        Self::new(theta.cos(), theta.sin())
    }
}

/// A single second-order section (biquad) in the form
///
/// y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2]
///        - a1 y[n-1] - a2 y[n-2]
///
/// corresponding to one factor in equation (13.6.9).
#[derive(Clone, Debug)]
struct BiquadSection {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl BiquadSection {
    fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    /// Process a single sample through this section.
    fn process_sample(&mut self, x0: f64) -> f64 {
        let y0 = self.b0 * x0
            + self.b1 * self.x1
            + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = x0;
        self.y2 = self.y1;
        self.y1 = y0;

        y0
    }

    /// Evaluate the section response on the unit circle.
    fn frequency_response(&self, omega: f64) -> Complex {
        let z1 = Complex::exp_i(-omega);
        let z2 = Complex::exp_i(-2.0 * omega);

        let numerator = Complex::new(self.b0, 0.0)
            .add(z1.scale(self.b1))
            .add(z2.scale(self.b2));

        let denominator = Complex::new(1.0, 0.0)
            .add(z1.scale(self.a1))
            .add(z2.scale(self.a2));

        numerator.div(denominator)
    }

    /// Return the poles of z^2 + a1 z + a2 = 0.
    fn poles(&self) -> (Complex, Complex) {
        let disc = self.a1 * self.a1 - 4.0 * self.a2;
        if disc >= 0.0 {
            let sqrt_disc = disc.sqrt();
            let p1 = Complex::new((-self.a1 + sqrt_disc) / 2.0, 0.0);
            let p2 = Complex::new((-self.a1 - sqrt_disc) / 2.0, 0.0);
            (p1, p2)
        } else {
            let sqrt_abs = (-disc).sqrt();
            let real = -self.a1 / 2.0;
            let imag = sqrt_abs / 2.0;
            (Complex::new(real, imag), Complex::new(real, -imag))
        }
    }

    fn max_pole_radius(&self) -> f64 {
        let (p1, p2) = self.poles();
        p1.magnitude().max(p2.magnitude())
    }

    fn is_stable(&self) -> bool {
        self.max_pole_radius() < 1.0
    }
}

/// A full IIR filter represented as a cascade of second-order sections.
#[derive(Clone, Debug)]
struct SOSCascade {
    sections: Vec<BiquadSection>,
}

impl SOSCascade {
    fn new(sections: Vec<BiquadSection>) -> Self {
        assert!(
            !sections.is_empty(),
            "The cascade must contain at least one second-order section."
        );
        Self { sections }
    }

    fn reset(&mut self) {
        for sec in self.sections.iter_mut() {
            sec.reset();
        }
    }

    /// Process one sample through the entire cascade.
    fn process_sample(&mut self, mut x: f64) -> f64 {
        for sec in self.sections.iter_mut() {
            x = sec.process_sample(x);
        }
        x
    }

    fn process_block(&mut self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xn| self.process_sample(xn)).collect()
    }

    fn impulse_response(&self, n: usize) -> Vec<f64> {
        let mut cascade = self.clone();
        cascade.reset();

        let mut h = Vec::with_capacity(n);
        for k in 0..n {
            let input = if k == 0 { 1.0 } else { 0.0 };
            h.push(cascade.process_sample(input));
        }
        h
    }

    fn frequency_response(&self, omega: f64) -> Complex {
        let mut response = Complex::new(1.0, 0.0);
        for sec in self.sections.iter() {
            response = response.mul(sec.frequency_response(omega));
        }
        response
    }

    fn all_stable(&self) -> bool {
        self.sections.iter().all(|sec| sec.is_stable())
    }
}

fn print_sequence(name: &str, x: &[f64]) {
    println!("{name}");
    for (n, &value) in x.iter().enumerate() {
        println!("  {:>3}: {:.10}", n, value);
    }
    println!();
}

fn generate_test_signal(n_samples: usize) -> Vec<f64> {
    let mut x = Vec::with_capacity(n_samples);
    for n in 0..n_samples {
        let t = n as f64;
        let low = (0.08 * PI * t).sin();
        let mid = 0.45 * (0.32 * PI * t).sin();
        let high = 0.25 * (0.86 * PI * t).sin();
        x.push(low + mid + high);
    }
    x
}

fn print_section_report(cascade: &SOSCascade) {
    println!("Second-Order Section Stability Report:");
    for (j, sec) in cascade.sections.iter().enumerate() {
        let (p1, p2) = sec.poles();
        let rho = sec.max_pole_radius();
        println!("  Section {}", j + 1);
        println!(
            "    b = ({:.8}, {:.8}, {:.8})",
            sec.b0, sec.b1, sec.b2
        );
        println!(
            "    a = (1, {:.8}, {:.8})",
            sec.a1, sec.a2
        );
        println!(
            "    pole 1 = ({:.8}, {:.8}), |pole 1| = {:.8}",
            p1.re,
            p1.im,
            p1.magnitude()
        );
        println!(
            "    pole 2 = ({:.8}, {:.8}), |pole 2| = {:.8}",
            p2.re,
            p2.im,
            p2.magnitude()
        );
        println!("    max pole radius = {:.8}", rho);
        println!("    stable          = {}", sec.is_stable());
    }
    println!();
}

fn main() {
    // Two stable low-pass style sections. Each denominator polynomial is
    // 1 + a1 z^{-1} + a2 z^{-2}, with poles strictly inside the unit circle.
    //
    // Section 1 has poles with radius about 0.70.
    // Section 2 has poles with radius about 0.80.
    //
    // The overall cascade is therefore stable and demonstrates the SOS
    // implementation advocated in Section 13.6.3.
    let sections = vec![
        BiquadSection::new(0.06745527, 0.13491054, 0.06745527, -1.14298050, 0.41280160),
        BiquadSection::new(0.20657208, 0.41314417, 0.20657208, -0.36952738, 0.19581571),
    ];

    let mut cascade = SOSCascade::new(sections);

    println!("=== IIR Filter Cascade via Second-Order Sections ===\n");
    print_section_report(&cascade);

    println!("All sections stable: {}", cascade.all_stable());
    println!();

    let x = generate_test_signal(32);
    let y = cascade.process_block(&x);

    print_sequence("Input x[n]:", &x);
    print_sequence("Cascade output y[n]:", &y);

    let h = cascade.impulse_response(24);
    print_sequence("Impulse response h[n] of the SOS cascade:", &h);

    println!("Sampled frequency response values for the cascade:");
    for &omega in &[0.0, 0.15 * PI, 0.30 * PI, 0.50 * PI, 0.70 * PI, PI] {
        let h_omega = cascade.frequency_response(omega);
        println!(
            "  omega = {:>5.2} pi : H = ({:.8}, {:.8}), |H| = {:.8}, phase = {:.8}",
            omega / PI,
            h_omega.re,
            h_omega.im,
            h_omega.magnitude(),
            h_omega.phase()
        );
    }
    println!();

    println!("Interpretation:");
    println!("1. Each stage implements one factor of equation (13.6.9) as a second-order recursion.");
    println!("2. The pole report verifies the stability condition in equation (13.6.8) for every section.");
    println!("3. Cascading biquads localizes state and rounding effects, making the implementation");
    println!("   numerically preferable to a single high-order direct-form realization.");
    println!("4. The impulse response decays rather than terminating, confirming the infinite");
    println!("   impulse-response character of the system.");
}
```

Program 13.6.3 demonstrates a practical realization of recursive digital filtering through a cascade of second-order sections, reflecting the central implementation principle discussed in Section 13.6.3. Rather than evaluating a high-order IIR transfer function directly, the code decomposes the system into biquad stages whose states, poles, and numerical effects can be examined individually.

The results illustrate two essential features of IIR filtering. First, the pole computations provide a concrete check of the stability requirement in equation (13.6.8), showing that the recursion remains well behaved only when all poles lie strictly inside the unit circle. Second, the decaying impulse response and the sampled frequency response demonstrate how a relatively small number of coefficients can produce strong spectral selectivity, one of the principal advantages of IIR filters over FIR alternatives.

The SOS-based design also emphasizes an important numerical lesson: implementation form matters as much as transfer-function algebra. By localizing state and rounding effects within individual sections, the cascade structure provides a more robust computational model for recursive filters and forms a natural foundation for further developments such as adaptive section tuning, pole-radius constraints, and time-varying IIR systems. In this way, the program connects classical filter theory with the numerical considerations that govern real-world DSP software.

## 13.6.4. Implementation Issues: Quantization, Real-Time Operation, and System Design

In many computational environments digital filters must operate under strict hardware constraints. When filters are implemented with fixed-point arithmetic, coefficient quantization and rounding can significantly affect system behavior. Quantization errors may alter pole locations, produce limit cycles, or even cause instability in poorly structured implementations. Comparative studies of fixed-point filter implementations show that different structures, such as direct-form II, cascade, or rotation-based realizations, exhibit substantially different numerical behavior (Poczekajło and Wirski, 2025).

Word-length selection therefore becomes an important design parameter in embedded systems. Recent work proposes using uncertainty-propagation techniques such as polynomial chaos expansions to estimate appropriate word lengths in complex DSP chains containing many interacting quantization noise sources (Rahman et al., 2025). Such approaches treat numerical precision as a design variable rather than an afterthought.

From a systems perspective, the choice between time-domain filtering and FFT-based filtering depends primarily on latency and filter length. Time-domain filters are preferable when the filter order is small and when output must be produced sample by sample with minimal delay. Examples include control loops, sensor feedback systems, and low-latency audio processing. In contrast, FFT-based filtering becomes advantageous when the impulse response is very long or when the processing pipeline already operates in the spectral domain.

Real-world signal processing systems frequently combine both approaches. Adaptive noise-cancellation systems, for example, may update filter coefficients using time-domain algorithms while analyzing signals spectrally for diagnostics or parameter estimation. Recent developments in active noise control incorporate diffusion-based adaptive filters operating in distributed sensor networks, demonstrating that time-domain adaptation remains an active research area (Bahraini and Naeimi-Sadigh, 2024).

The practical lesson for numerical computing is therefore not that one filtering paradigm replaces another. Instead, time-domain and frequency-domain filtering serve complementary roles. FFT methods enable efficient large-scale convolution and spectral analysis, while time-domain digital filters provide stable, low-latency implementations suitable for real-time systems. Understanding both perspectives allows one to design algorithms that combine spectral insight with numerically reliable execution.

### Rust Implementation

Following the discussion in Section 13.6.4 on quantization effects, real-time constraints, and system-level design trade-offs, Program 13.6.4 provides a practical implementation of a second-order IIR filter operating under finite-precision arithmetic. In real-world digital signal processing systems, filters must often be deployed in environments where word length is limited and computations must be performed sample by sample with minimal latency. This program demonstrates how coefficient quantization and runtime rounding influence filter behavior by comparing a floating-point realization with a fixed-point implementation. It also illustrates how numerical perturbations affect pole locations and output accuracy while preserving stability when the condition in equation (13.6.8) remains satisfied. In addition, the program contrasts time-domain filtering with block-based processing to highlight latency considerations that arise in practical system design. Together, these components emphasize the interplay between numerical precision, stability, and real-time performance in modern digital filtering.

At the core of the implementation is the separation between floating-point and fixed-point realizations of a second-order IIR filter. The `BiquadFloat` struct represents the standard direct-form realization of a second-order section, corresponding to the recursive structure introduced earlier in the chapter. Its `process_sample` function evaluates the difference equation sample by sample, maintaining internal state variables for past inputs and outputs. This reflects the recursive nature of IIR filters discussed in equation (13.6.3), where feedback terms make the system sensitive to numerical precision.

The `BiquadFixed` struct extends this model by incorporating quantization at both the coefficient and runtime levels. The function `quantize_fixed` implements a uniform fixed-point representation by scaling, rounding, and saturating values to a specified number of bits. This directly models the finite-precision effects described in the section, where coefficient quantization can perturb pole locations and potentially affect stability as governed by equation (13.6.8). Within `BiquadFixed`, every arithmetic operation is followed by quantization, ensuring that rounding errors accumulate in a realistic manner consistent with embedded implementations.

The `process_sample` method in the fixed-point structure mirrors the floating-point computation but applies quantization after each multiplication and addition. This design captures the propagation of rounding errors through the recursive feedback loop, illustrating how quantization noise is not merely additive but dynamically coupled to the system state. The difference between the floating-point and fixed-point outputs therefore reflects both coefficient perturbations and accumulated arithmetic error.

To analyze system behavior, the program includes the `impulse_response` function, which computes the response of the filter to a unit impulse. For IIR filters, this response decays rather than terminating, consistent with the infinite impulse response property. Comparing the floating-point and quantized impulse responses reveals how finite precision truncates small values and modifies long-term dynamics.

The `poles` and `max_pole_radius` functions compute the roots of the denominator polynomial for each section, enabling direct verification of the stability condition in equation (13.6.8). By comparing pole radii before and after quantization, the program demonstrates how finite word length perturbs system dynamics while typically preserving stability when the original design has sufficient margin.

The `generate_test_signal` function constructs a composite signal containing both low- and high-frequency components. This allows the filter’s frequency-selective behavior to be observed in practice. The functions `rms_error` and `max_abs_error` quantify the deviation between floating-point and fixed-point outputs, providing numerical measures of quantization effects.

The `main` function orchestrates the entire experiment. It defines filter coefficients, constructs both floating-point and fixed-point filters, and evaluates their behavior on the same input signal. It reports coefficient quantization, pole locations, output sequences, impulse responses, and error metrics. Finally, it includes a simple latency model comparing time-domain filtering, which produces output immediately, with block-based FFT-style processing, which incurs delay proportional to block size. This reflects the system-level considerations discussed in the section, where latency and computational structure influence algorithm selection.

```rust
// Program 13.6.4: Quantized Real-Time Biquad Filtering and Latency-Aware System Comparison
//
// Problem Statement:
// Implement a time-domain digital filter in a form suitable for real-time operation
// under finite-precision constraints. The program should:
// 1. Process samples sequentially using a second-order IIR section.
// 2. Simulate coefficient and state quantization through fixed-point arithmetic.
// 3. Compare floating-point and quantized outputs on the same streaming input.
// 4. Examine stability-sensitive behavior through impulse response and error metrics.
// 5. Contrast sample-by-sample time-domain filtering with a simple latency estimate
//    for block FFT-style processing to illustrate system-design trade-offs.

use std::f64::consts::PI;

/// Lightweight complex number type for pole calculations.
#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn magnitude(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

/// Quantize a floating-point number to a signed fixed-point format with
/// `frac_bits` fractional bits using rounding and saturation.
///
/// The representable range is approximately
/// [-(2^integer_bits), 2^integer_bits - 2^{-frac_bits}],
/// where integer_bits = total_bits - frac_bits - 1.
fn quantize_fixed(x: f64, total_bits: u32, frac_bits: u32) -> f64 {
    assert!(total_bits >= 2, "Need at least sign bit and one magnitude bit.");
    assert!(
        frac_bits + 1 < total_bits,
        "Need at least one integer bit in addition to sign and fractional bits."
    );

    let scale = 2f64.powi(frac_bits as i32);
    let integer_bits = total_bits - frac_bits - 1;
    let max_int = (1i64 << (total_bits - 1)) - 1;
    let min_int = -(1i64 << (total_bits - 1));

    let _max_value = (2f64.powi(integer_bits as i32)) - 1.0 / scale;
    let _min_value = -2f64.powi(integer_bits as i32);

    let scaled = (x * scale).round();
    let clipped = if scaled > max_int as f64 {
        max_int
    } else if scaled < min_int as f64 {
        min_int
    } else {
        scaled as i64
    };

    clipped as f64 / scale
}

/// Floating-point biquad section in direct-form I.
#[derive(Clone, Debug)]
struct BiquadFloat {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl BiquadFloat {
    fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    /// Process a single sample:
    ///
    /// y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2]
    ///        - a1 y[n-1] - a2 y[n-2].
    fn process_sample(&mut self, x0: f64) -> f64 {
        let y0 = self.b0 * x0
            + self.b1 * self.x1
            + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = x0;
        self.y2 = self.y1;
        self.y1 = y0;

        y0
    }

    fn process_block(&mut self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| self.process_sample(x)).collect()
    }

    fn impulse_response(&self, n: usize) -> Vec<f64> {
        let mut sec = self.clone();
        sec.reset();

        let mut h = Vec::with_capacity(n);
        for k in 0..n {
            let x = if k == 0 { 1.0 } else { 0.0 };
            h.push(sec.process_sample(x));
        }
        h
    }

    fn poles(&self) -> (Complex, Complex) {
        let disc = self.a1 * self.a1 - 4.0 * self.a2;
        if disc >= 0.0 {
            let s = disc.sqrt();
            (
                Complex::new((-self.a1 + s) / 2.0, 0.0),
                Complex::new((-self.a1 - s) / 2.0, 0.0),
            )
        } else {
            let s = (-disc).sqrt();
            let real = -self.a1 / 2.0;
            let imag = s / 2.0;
            (Complex::new(real, imag), Complex::new(real, -imag))
        }
    }

    fn max_pole_radius(&self) -> f64 {
        let (p1, p2) = self.poles();
        p1.magnitude().max(p2.magnitude())
    }
}

/// Quantized biquad section.
/// Coefficients are quantized once at construction time.
/// States and each arithmetic result are quantized during runtime.
#[derive(Clone, Debug)]
struct BiquadFixed {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
    total_bits: u32,
    frac_bits: u32,
}

impl BiquadFixed {
    fn new(
        b0: f64,
        b1: f64,
        b2: f64,
        a1: f64,
        a2: f64,
        total_bits: u32,
        frac_bits: u32,
    ) -> Self {
        Self {
            b0: quantize_fixed(b0, total_bits, frac_bits),
            b1: quantize_fixed(b1, total_bits, frac_bits),
            b2: quantize_fixed(b2, total_bits, frac_bits),
            a1: quantize_fixed(a1, total_bits, frac_bits),
            a2: quantize_fixed(a2, total_bits, frac_bits),
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
            total_bits,
            frac_bits,
        }
    }

    fn q(&self, x: f64) -> f64 {
        quantize_fixed(x, self.total_bits, self.frac_bits)
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    fn process_sample(&mut self, x0_raw: f64) -> f64 {
        let x0 = self.q(x0_raw);

        let t0 = self.q(self.b0 * x0);
        let t1 = self.q(self.b1 * self.x1);
        let t2 = self.q(self.b2 * self.x2);
        let t3 = self.q(self.a1 * self.y1);
        let t4 = self.q(self.a2 * self.y2);

        let ff = self.q(self.q(t0 + t1) + t2);
        let fb = self.q(self.q(t3 + t4));
        let y0 = self.q(ff - fb);

        self.x2 = self.x1;
        self.x1 = x0;
        self.y2 = self.y1;
        self.y1 = y0;

        y0
    }

    fn process_block(&mut self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| self.process_sample(x)).collect()
    }

    fn impulse_response(&self, n: usize) -> Vec<f64> {
        let mut sec = self.clone();
        sec.reset();

        let mut h = Vec::with_capacity(n);
        for k in 0..n {
            let x = if k == 0 { 1.0 } else { 0.0 };
            h.push(sec.process_sample(x));
        }
        h
    }

    fn as_float_section(&self) -> BiquadFloat {
        BiquadFloat::new(self.b0, self.b1, self.b2, self.a1, self.a2)
    }
}

fn print_sequence(name: &str, x: &[f64]) {
    println!("{name}");
    for (n, &value) in x.iter().enumerate() {
        println!("  {:>3}: {:.10}", n, value);
    }
    println!();
}

fn generate_test_signal(n_samples: usize) -> Vec<f64> {
    let mut x = Vec::with_capacity(n_samples);
    for n in 0..n_samples {
        let t = n as f64;
        let low = (0.10 * PI * t).sin();
        let high = 0.30 * (0.78 * PI * t).sin();
        x.push(low + high);
    }
    x
}

fn rms_error(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Input vectors must have equal length.");
    let mse: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| {
            let e = a - b;
            e * e
        })
        .sum::<f64>()
        / x.len() as f64;
    mse.sqrt()
}

fn max_abs_error(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Input vectors must have equal length.");
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn estimate_fft_block_latency(block_len: usize) -> usize {
    // A minimal educational latency proxy:
    // one cannot emit the block-processed output before at least one full
    // block has been accumulated. Real overlap-save/overlap-add pipelines
    // can add further delay depending on configuration.
    block_len
}

fn main() {
    // A stable low-pass biquad in floating point.
    // The denominator is 1 + a1 z^{-1} + a2 z^{-2}.
    let b0 = 0.06745527;
    let b1 = 0.13491054;
    let b2 = 0.06745527;
    let a1 = -1.14298050;
    let a2 = 0.41280160;

    // Fixed-point format configuration.
    let total_bits = 16;
    let frac_bits = 13;

    let float_section = BiquadFloat::new(b0, b1, b2, a1, a2);
    let fixed_section = BiquadFixed::new(b0, b1, b2, a1, a2, total_bits, frac_bits);

    let quantized_float_equivalent = fixed_section.as_float_section();

    let float_radius = float_section.max_pole_radius();
    let quantized_radius = quantized_float_equivalent.max_pole_radius();

    println!("=== Quantized Real-Time Biquad Filtering ===\n");

    println!("Floating-point coefficients:");
    println!("  b = ({:.8}, {:.8}, {:.8})", b0, b1, b2);
    println!("  a = (1, {:.8}, {:.8})", a1, a2);
    println!();

    println!("Quantized coefficients (Q{}.{} style interpretation):", total_bits, frac_bits);
    println!(
        "  b = ({:.8}, {:.8}, {:.8})",
        fixed_section.b0, fixed_section.b1, fixed_section.b2
    );
    println!(
        "  a = (1, {:.8}, {:.8})",
        fixed_section.a1, fixed_section.a2
    );
    println!();

    println!("Pole-radius comparison:");
    println!("  floating-point max pole radius = {:.8}", float_radius);
    println!("  quantized      max pole radius = {:.8}", quantized_radius);
    println!(
        "  stable after coefficient quantization = {}",
        quantized_radius < 1.0
    );
    println!();

    let input = generate_test_signal(40);

    let mut float_runtime = float_section.clone();
    let mut fixed_runtime = fixed_section.clone();

    let output_float = float_runtime.process_block(&input);
    let output_fixed = fixed_runtime.process_block(&input);

    let sample_error: Vec<f64> = output_float
        .iter()
        .zip(output_fixed.iter())
        .map(|(yf, yq)| yq - yf)
        .collect();

    print_sequence("Input x[n]:", &input);
    print_sequence("Floating-point output y_float[n]:", &output_float);
    print_sequence("Quantized output y_fixed[n]:", &output_fixed);
    print_sequence("Output error y_fixed[n] - y_float[n]:", &sample_error);

    let h_float = float_section.impulse_response(24);
    let h_fixed = fixed_section.impulse_response(24);

    print_sequence("Floating-point impulse response h_float[n]:", &h_float);
    print_sequence("Quantized impulse response h_fixed[n]:", &h_fixed);

    println!("Error metrics:");
    println!("  RMS output error     = {:.10}", rms_error(&output_float, &output_fixed));
    println!("  Max absolute error   = {:.10}", max_abs_error(&output_float, &output_fixed));
    println!();

    let n_stream = input.len();
    let macs_per_sample = 5usize;
    let direct_stream_latency = 1usize;
    let fft_block_len = 64usize;
    let fft_latency = estimate_fft_block_latency(fft_block_len);

    println!("Real-time system comparison:");
    println!("  processed stream length                  = {}", n_stream);
    println!("  approximate MACs per time-domain sample  = {}", macs_per_sample);
    println!("  time-domain output latency (samples)     = {}", direct_stream_latency);
    println!("  illustrative FFT block length            = {}", fft_block_len);
    println!("  illustrative FFT block latency (samples) = {}", fft_latency);
    println!();

    println!("Interpretation:");
    println!("1. The floating-point and fixed-point filters use the same biquad structure,");
    println!("   but coefficient and runtime quantization perturb the realized system.");
    println!("2. The pole-radius comparison shows how finite word length can shift poles and");
    println!("   therefore alter the margin implied by the stability condition in equation (13.6.8).");
    println!("3. The quantized output remains close to the floating-point output here, but the");
    println!("   error sequence and impulse-response differences reveal accumulated rounding effects.");
    println!("4. Sample-by-sample time-domain filtering delivers minimal latency, whereas an");
    println!("   FFT-style block pipeline introduces at least block-level delay before output.");
}
```

Program 13.6.4 demonstrates how finite-precision arithmetic influences the behavior of recursive digital filters in practical implementations. By comparing floating-point and fixed-point realizations, the program highlights how coefficient quantization and rounding affect pole locations, impulse responses, and output accuracy, while still preserving stability when the condition in equation (13.6.8) is satisfied.

The results illustrate that even small quantization errors can propagate through feedback structures, leading to measurable deviations in output signals. At the same time, the modest error levels observed here show that well-designed filters with sufficient stability margins can remain robust under realistic word-length constraints. This reinforces the importance of careful coefficient scaling and structure selection in embedded systems.

The comparison between time-domain filtering and block-based processing further emphasizes the role of system design considerations. Time-domain implementations provide minimal latency and are well suited for real-time applications, whereas FFT-based methods introduce delay but offer computational advantages for long filters. These trade-offs illustrate that numerical algorithms must be evaluated not only in terms of accuracy but also in terms of their interaction with hardware and system requirements.

Overall, the program reinforces the central theme of this section: digital filter design is not purely a mathematical exercise, but a numerical and systems problem in which precision, stability, and performance must be considered together.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/ojNizLf2LQBr4DtKgyGU.8","tags":[]}

# 13.7. Linear Prediction and Linear Predictive Coding

Linear prediction and linear predictive coding arise from a common numerical problem: we observe only a finite record of a process, often contaminated by noise, and we seek a low-dimensional model that captures the short-term dependence structure well enough to support inference, extrapolation, compression, or feature extraction. In linear prediction, the model class is deliberately simple: each sample is approximated by a linear combination of a small number of past samples. In linear predictive coding (LPC), this same predictor is used not only as an estimator but also as a compact representation, since once predictable structure has been removed, the residual is often much cheaper to encode. Although linear prediction is historically associated with speech processing, its mathematical form is much broader. The same algebra appears in reduced modeling of simulated time series, autoregressive spectral estimation, system identification, and diagnostics for irregular or partially observed data.

A useful unifying viewpoint is the linear minimum mean-square error (LMMSE) estimator. Suppose noisy observations satisfy:

$$y'_\alpha = y_\alpha + n_\alpha, \qquad \alpha \in \{0,\dots,N-1\} \tag{13.7.1}$$

where $y_\alpha$ is the underlying signal and $n_\alpha$ is additive noise. Define the signal covariance, noise covariance, and target-to-observation cross-covariance by:

$$
\Phi_{\alpha\beta} = E\!\left[y_\alpha y_\beta\right], \qquad
H_{\alpha\beta} = E\!\left[n_\alpha n_\beta\right] \tag{13.7.2}
$$

$$\varphi_{\star\alpha} = E\!\left[y_\star y_\alpha\right] \tag{13.7.3}$$

If one approximates a target value $y_\star$ by a linear combination of the observations,

$$\widehat{y}_\star = d^{\top} y' \tag{13.7.4}$$

then minimizing the mean-square error,

$$E\!\left[(y_\star - d^{\top} y')^2\right] \tag{13.7.5}$$

leads to the normal equations:

$$(\Phi + H) d_\star = \varphi_\star \tag{13.7.6}$$

This compact system is the algebraic template behind much of the remainder of this chapter segment. Classical linear prediction corresponds to the stationary, uniformly sampled, low-noise specialization of (13.7.6), where the covariance structure becomes Toeplitz and the resulting system can be solved efficiently by structured recursions. Maximum-entropy all-poles spectral estimation and modern covariance-based methods may be viewed as closely related variants of the same core idea.

## 13.7.1. Predictor Definition and Least-Squares Formulation

Consider now a uniformly sampled scalar sequence $y_n$ with sampling interval $\Delta t$. A $p$-th order one-step linear predictor has the form:

$$\widehat{y}_n = \sum_{k=1}^{p} a_k\, y_{n-k} \tag{13.7.7}$$

and the corresponding prediction error, or residual, is

$$e_n = y_n - \widehat{y}_n \tag{13.7.8}$$

Equivalently, the data may be written in autoregressive form as,

$$y_n = \sum_{k=1}^{p} a_k\, y_{n-k} + e_n \tag{13.7.9}$$

which makes clear that linear prediction and autoregressive modeling are algebraically the same construction. The coefficients $a_k$ describe the short-term linear dynamics, while the residual $e_n$ contains whatever cannot be predicted from the preceding $p$ samples.

A direct computational route is to choose the coefficients by minimizing the empirical mean-square prediction error over all valid indices:

$$J(a) = \sum_{n=p}^{N-1} \left( y_n - \sum_{k=1}^{p} a_k y_{n-k} \right)^2 \tag{13.7.10}$$

To write this as a standard least-squares problem, define the regression matrix $Y$, the target vector (b), and the coefficient vector $a$ by,

$$
Y =
\begin{pmatrix}
y_{p-1} & y_{p-2} & \cdots & y_0 \\
y_p     & y_{p-1} & \cdots & y_1 \\
\vdots  & \vdots  & \ddots & \vdots \\
y_{N-2} & y_{N-3} & \cdots & y_{N-p-1}
\end{pmatrix} \tag{13.7.11}
$$

$$
b =
\begin{pmatrix}
y_p \\
y_{p+1} \\
\vdots \\
y_{N-1}
\end{pmatrix},
\qquad
a =
\begin{pmatrix}
a_1 \\
a_2 \\
\vdots \\
a_p
\end{pmatrix} \tag{13.7.12}
$$

Then the objective becomes:

$$J(a) = \lVert b - Ya \rVert_2^2 \tag{13.7.13}$$

and the normal equations are:

$$(Y^{\top} Y) a = Y^{\top} b \tag{13.7.14}$$

This is the most direct numerical formulation of linear prediction. It is conceptually simple and useful for dense small-scale solves, but it does not yet exploit the stochastic structure of the problem. Under stationarity and sufficiently long records, the matrix $Y^{\top}Y$ approaches an autocorrelation matrix with Toeplitz form, and that structure can be used to derive more specialized and more efficient algorithms.

### Rust Implementation

Following the discussion in Section 13.7.1 on the least-squares formulation of linear prediction, Program 13.7.1 provides a practical implementation of $p$-th order one-step linear prediction for a uniformly sampled scalar sequence. In this setting, the predictor coefficients are obtained by minimizing the empirical mean-square prediction error in Equation (13.7.10), which leads to the matrix formulation in Equations (13.7.11) through (13.7.14). The program constructs the regression system explicitly from the observed data, forms the associated normal equations, and solves for the predictor coefficients using a Cholesky-based dense linear algebra path. It then evaluates the fitted values, residuals, mean-square prediction error, and a one-step-ahead forecast. In this way, the implementation translates the algebraic development of least-squares linear prediction into a concrete computational framework that can be executed directly and examined numerically.

At the core of the implementation is the construction of the regression matrix $Y$, the target vector $b$, and the coefficient vector $a$ corresponding to Equations (13.7.11) and (13.7.12). The `build_design_matrix` function assembles this least-squares system from the input sequence by placing the previous $p$ samples of the signal into each row of the matrix and the current sample into the corresponding entry of the target vector. This directly realizes the predictor definition in Equation (13.7.7) and the least-squares objective in Equation (13.7.13). By making the data dependence explicit, the function shows how a time-domain prediction problem can be rewritten as a standard linear algebra problem.

Once the design matrix has been formed, the program constructs the normal equations of Equation (13.7.14). The `form_normal_equations` function computes the Gram matrix $Y^{\top}Y$ and the right-hand side $Y^{\top}b$, thereby reducing the least-squares fit to a symmetric positive-definite linear system when the data are sufficiently informative. This stage reflects the direct dense formulation discussed in the section. Although it does not yet exploit Toeplitz structure, it is conceptually transparent and well suited to small predictor orders, where explicit matrix formation remains practical and easy to verify.

The solution of the resulting linear system is handled by the pair of functions `cholesky_decompose` and `cholesky_solve`. The first computes a Cholesky factorization of the normal-equation matrix, while the second performs forward and backward substitution to recover the coefficient vector. This is appropriate in the present setting because $Y^{\top}Y$ is symmetric and, under nondegenerate conditions, positive definite. The use of Cholesky factorization keeps the implementation aligned with the numerical structure of Equation (13.7.14) while avoiding the overhead of more general-purpose dense solvers. The explicit checks for near-singularity also help make the code numerically safer when the predictor order is too large relative to the available data or when the regression system is poorly conditioned.

The `fit_least_squares_predictor` function ties these pieces together into a complete estimation pipeline. It builds the system, forms the normal equations, solves for the coefficients, and then computes the fitted values and residuals corresponding to Equations (13.7.7) and (13.7.8). It also evaluates the empirical mean-square prediction error, which serves as a numerical summary of how well the predictor captures the short-term linear dependence in the data. In addition, the `one_step_ahead_forecast` function uses the estimated coefficients together with the last $p$ observed samples to produce an extrapolated value, illustrating how the fitted model can be used not only for in-sample approximation but also for immediate forecasting.

For diagnostic purposes, the implementation also includes `sample_autocorrelation`, which computes empirical lag correlations up to a prescribed maximum lag. This is useful in the context of the discussion following Equation (13.7.14), where the section notes that, under stationarity and sufficiently long records, the dense least-squares matrix approaches an autocorrelation matrix with Toeplitz form. The autocorrelation output therefore provides a bridge between the direct dense least-squares method of Section 13.7.1 and the more structured Yule-Walker viewpoint introduced in Section 13.7.2.

The `generate_test_signal` function provides a deterministic sample sequence with autoregressive character and mild oscillatory forcing so that the behavior of the predictor can be examined without requiring external input data. Finally, the `main` function serves as a complete demonstration driver. It selects the data length and predictor order, fits the least-squares linear predictor, prints the estimated coefficients, reports the sample autocorrelation diagnostic, displays the mean-square prediction error, and lists representative fitted values and residuals. By concluding with a one-step-ahead forecast, the program shows how the abstract predictor model of Equations (13.7.7) through (13.7.14) becomes a usable computational tool for estimation, residual analysis, and short-horizon extrapolation.

```rust
// Program 13.7.1 Least-Squares Linear Prediction
//
// Problem Statement:
// Implement the least-squares formulation of p-th order one-step linear prediction
// for a uniformly sampled scalar sequence y_n. Given data y_0, y_1, ..., y_{N-1},
// build the regression matrix Y and target vector b corresponding to equations
// (13.7.11) and (13.7.12), then solve the normal equations
//
//     (Y^T Y) a = Y^T b
//
// from equation (13.7.14) to estimate the predictor coefficients a_1, ..., a_p.
// After estimating the coefficients, compute the fitted values, residuals, mean-
// square prediction error, and a one-step-ahead forecast based on the last p samples.
//
// This program is self-contained and can be run directly with `cargo run`.

use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

/// A small tolerance used in linear algebra checks.
const EPS: f64 = 1.0e-12;

/// Errors that may arise in the predictor construction or solve.
#[derive(Debug)]
enum LpError {
    InvalidOrder { p: usize, n: usize },
    SingularSystem,
}

impl fmt::Display for LpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LpError::InvalidOrder { p, n } => write!(
                f,
                "invalid predictor order p = {} for data length N = {}; require 1 <= p < N",
                p, n
            ),
            LpError::SingularSystem => write!(
                f,
                "normal-equation matrix is singular or not sufficiently positive definite"
            ),
        }
    }
}

impl Error for LpError {}

/// Model output for least-squares linear prediction.
#[derive(Debug, Clone)]
struct LinearPredictor {
    /// Predictor order p.
    order: usize,
    /// Estimated coefficients [a1, a2, ..., ap].
    coeffs: Vec<f64>,
    /// Fitted values for indices n = p, ..., N-1.
    fitted: Vec<f64>,
    /// Residuals e_n = y_n - yhat_n for indices n = p, ..., N-1.
    residuals: Vec<f64>,
    /// Mean-square prediction error over valid indices.
    mse: f64,
}

/// Dot product of two equal-length vectors.
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

/// Build the regression matrix Y and target vector b from equations
/// (13.7.11) and (13.7.12).
///
/// Y has dimensions (N - p) x p, and b has length (N - p).
fn build_design_matrix(data: &[f64], p: usize) -> Result<(Vec<Vec<f64>>, Vec<f64>), LpError> {
    let n = data.len();
    if p == 0 || p >= n {
        return Err(LpError::InvalidOrder { p, n });
    }

    let rows = n - p;
    let mut y_mat = vec![vec![0.0; p]; rows];
    let mut b = vec![0.0; rows];

    for row in 0..rows {
        let current_index = p + row;
        b[row] = data[current_index];

        // Row structure:
        // [y_{n-1}, y_{n-2}, ..., y_{n-p}]
        for col in 0..p {
            y_mat[row][col] = data[current_index - 1 - col];
        }
    }

    Ok((y_mat, b))
}

/// Form the normal-equation matrix G = Y^T Y and right-hand side c = Y^T b.
fn form_normal_equations(y_mat: &[Vec<f64>], b: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let rows = y_mat.len();
    let p = y_mat[0].len();

    let mut gram = vec![vec![0.0; p]; p];
    let mut rhs = vec![0.0; p];

    for i in 0..p {
        for j in i..p {
            let mut sum = 0.0;
            for r in 0..rows {
                sum += y_mat[r][i] * y_mat[r][j];
            }
            gram[i][j] = sum;
            gram[j][i] = sum;
        }

        let mut sum = 0.0;
        for r in 0..rows {
            sum += y_mat[r][i] * b[r];
        }
        rhs[i] = sum;
    }

    (gram, rhs)
}

/// Cholesky factorization of a symmetric positive-definite matrix A.
///
/// Returns lower-triangular L such that A = L L^T.
fn cholesky_decompose(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, LpError> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }

            if i == j {
                if sum <= EPS {
                    return Err(LpError::SingularSystem);
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    Ok(l)
}

/// Solve A x = b using a Cholesky factorization A = L L^T.
fn cholesky_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();

    // Forward solve: L y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }

    // Backward solve: L^T x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }

    x
}

/// Fit a p-th order least-squares linear predictor.
fn fit_least_squares_predictor(data: &[f64], p: usize) -> Result<LinearPredictor, LpError> {
    let (y_mat, b) = build_design_matrix(data, p)?;
    let (gram, rhs) = form_normal_equations(&y_mat, &b);
    let l = cholesky_decompose(&gram)?;
    let coeffs = cholesky_solve(&l, &rhs);

    let mut fitted = Vec::with_capacity(b.len());
    let mut residuals = Vec::with_capacity(b.len());

    for (row, target) in y_mat.iter().zip(b.iter()) {
        let yhat = dot(row, &coeffs);
        fitted.push(yhat);
        residuals.push(*target - yhat);
    }

    let mse = residuals.iter().map(|e| e * e).sum::<f64>() / residuals.len() as f64;

    Ok(LinearPredictor {
        order: p,
        coeffs,
        fitted,
        residuals,
        mse,
    })
}

/// Perform a one-step-ahead forecast using the last p samples of the data.
fn one_step_ahead_forecast(data: &[f64], coeffs: &[f64]) -> Result<f64, LpError> {
    let p = coeffs.len();
    let n = data.len();
    if p == 0 || p >= n + 1 {
        return Err(LpError::InvalidOrder { p, n });
    }

    let mut value = 0.0;
    for k in 0..p {
        value += coeffs[k] * data[n - 1 - k];
    }
    Ok(value)
}

/// Compute the sample autocorrelation r_k = (1 / (N-k)) sum_{n=k}^{N-1} y_n y_{n-k}
/// for k = 0, 1, ..., max_lag. This is printed as a diagnostic since, for long records,
/// Y^T Y approaches a Toeplitz autocorrelation matrix.
fn sample_autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    let mut r = Vec::with_capacity(max_lag + 1);

    for lag in 0..=max_lag {
        let mut sum = 0.0;
        let mut count = 0usize;
        for i in lag..n {
            sum += data[i] * data[i - lag];
            count += 1;
        }
        r.push(sum / count as f64);
    }

    r
}

/// Generate a deterministic AR(2)-like sequence with a small oscillatory forcing.
/// This provides a clean test signal for the least-squares predictor.
fn generate_test_signal(n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    y[0] = 0.35;
    y[1] = -0.10;

    for k in 2..n {
        let forcing = 0.04 * (2.0 * PI * k as f64 / 17.0).sin()
            + 0.02 * (2.0 * PI * k as f64 / 9.0).cos();
        y[k] = 1.45 * y[k - 1] - 0.72 * y[k - 2] + forcing;
    }

    y
}

/// Print a vector with aligned indices.
fn print_vector(name: &str, values: &[f64]) {
    println!("{name}");
    for (i, value) in values.iter().enumerate() {
        println!("  [{:>2}] {:>.10}", i, value);
    }
}

/// Print the first few fitted values and residuals.
fn print_fit_summary(data: &[f64], predictor: &LinearPredictor, max_rows: usize) {
    println!("\nFirst {} valid prediction rows:", max_rows);
    println!(
        "{:>6} {:>16} {:>16} {:>16}",
        "n", "y_n", "yhat_n", "e_n"
    );

    let rows = predictor.fitted.len().min(max_rows);
    for row in 0..rows {
        let n = predictor.order + row;
        println!(
            "{:>6} {:>16.10} {:>16.10} {:>16.10}",
            n,
            data[n],
            predictor.fitted[row],
            predictor.residuals[row]
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Example data and predictor order.
    let data = generate_test_signal(80);
    let p = 2;

    println!("Least-Squares Linear Prediction Demo");
    println!("====================================");
    println!("Data length N  = {}", data.len());
    println!("Predictor order p = {}", p);

    // Fit predictor.
    let predictor = fit_least_squares_predictor(&data, p)?;
    let forecast = one_step_ahead_forecast(&data, &predictor.coeffs)?;
    let ac = sample_autocorrelation(&data, p);

    println!("\nEstimated predictor coefficients:");
    print_vector("a =", &predictor.coeffs);

    println!("\nSample autocorrelation diagnostic:");
    print_vector("r_k =", &ac);

    println!("\nMean-square prediction error:");
    println!("  MSE = {:>.10}", predictor.mse);

    print_fit_summary(&data, &predictor, 10);

    println!("\nOne-step-ahead forecast based on the last p samples:");
    println!("  y_hat[N] = {:>.10}", forecast);

    Ok(())
}
```

Program 13.7.1 demonstrates a practical dense implementation of least-squares linear prediction based directly on the matrix formulation developed in Section 13.7.1. By constructing the regression matrix explicitly and solving the normal equations with a Cholesky factorization, the program shows how predictor estimation can be treated as a standard numerical linear algebra problem. This makes the connection between the time-domain prediction model and its algebraic realization especially clear.

The implementation also highlights an important numerical theme of the section: the balance between conceptual simplicity and structural efficiency. The dense least-squares path is easy to understand, straightforward to verify, and well suited to small predictor orders, but it does not yet exploit the Toeplitz covariance structure that emerges under stationarity. For that reason, the program serves naturally as a baseline against which more specialized methods, such as Yule-Walker solvers and Levinson-Durbin recursions, can later be compared.

More broadly, the program illustrates how linear prediction supports several related tasks at once. The same fitted coefficients can be used to compute residuals, assess model quality through the mean-square error, and generate one-step-ahead forecasts. This provides a foundation for the later developments in the section, where structured autocorrelation systems, Burg-type estimation, and linear predictive coding extend the same basic numerical idea into more efficient and more application-oriented forms.

## 13.7.2. Autocorrelation Formulation and the Yule-Walker System

If the sequence is modeled as wide-sense stationary, the relevant second-order statistics depend only on lag. Define the autocorrelation sequence by:

$$r_k = E[y_n y_{n-k}] \tag{13.7.15}$$

Then the prediction coefficients satisfy the classical Yule-Walker system,

$$Ra = r \tag{13.7.16}$$

where the Toeplitz autocorrelation matrix $R\in\mathbb{R}^{p\times p}$ is:

$$
R=
\begin{pmatrix}
r_0 & r_1 & \cdots & r_{p-1} \\
r_1 & r_0 & \cdots & r_{p-2} \\
\vdots & \vdots & \ddots & \vdots \\
r_{p-1} & r_{p-2} & \cdots & r_0
\end{pmatrix} \tag{13.7.17}
$$

and the right-hand side vector is,

$$
r=
\begin{pmatrix}
r_1\\
r_2\\
\vdots\\
r_p
\end{pmatrix} \tag{13.7.18}
$$

This Toeplitz system is the computational core shared by LPC coefficient estimation, all-poles maximum-entropy spectrum estimation, and many structured covariance estimators. Its importance lies not only in the modeling interpretation but also in the numerical savings it enables. If one ignores structure and solves a dense $p\times p$ system directly, the cost is typically $O(p^3)$ time and $O(p^2)$ memory. By contrast, Toeplitz-aware recursions such as Levinson-Durbin reduce the work to approximately $O(p^2)$, while preserving the covariance structure explicitly. More modern methods also exploit Toeplitz structure inside more elaborate optimization frameworks, including maximum-likelihood Toeplitz covariance estimation accelerated by FFT-based and structured linear algebra techniques (Cederberg, 2024).

The predictor coefficients obtained from (13.7.16) determine a polynomial:

$$A(z)=1-\sum_{k=1}^{p} a_k z^{-k} \tag{13.7.19}$$

whose roots govern the dynamical behavior of the extrapolator. Stability of the predictor requires that the associated autoregressive model define a nonexplosive process, which in practice means that the relevant roots must lie in the region corresponding to a stable causal recursion. For extrapolation and coding applications, this is not merely a theoretical detail. Unstable fitted models can amplify tiny perturbations and produce unusable predictions.

### Rust Implementation

Following the development in Section 13.7.2, where the autocorrelation formulation of linear prediction leads to the Yule–Walker system in Equations (13.7.16) through (13.7.18), Program 13.7.2 provides a structured implementation of predictor coefficient estimation using Toeplitz-aware algorithms. Rather than forming and solving a dense normal-equation system as in Section 13.7.1, this program computes the autocorrelation sequence and applies the Levinson–Durbin recursion to exploit the Toeplitz structure efficiently. The implementation also evaluates the resulting predictor through residual analysis, computes the associated prediction polynomial, and performs a stability assessment via root analysis. In this way, the program translates the structured covariance formulation into an efficient and numerically meaningful computational workflow.

At the core of the implementation is the computation of the autocorrelation sequence defined in Equation (13.7.15). The `biased_autocorrelation` function estimates the lag-dependent correlations directly from the data using a finite-sample average. The biased form is used because it preserves the positive semidefinite structure required for the Toeplitz system, ensuring that the subsequent recursion remains well-posed. This function provides the sequence $r_0, r_1, \dots, r_p$, which forms both the Toeplitz matrix and the right-hand side in the Yule–Walker system.

The structured solution of the system $Ra = r$ is carried out by the `levinson_durbin` function. This function implements the classical Levinson–Durbin recursion, reducing the computational complexity from cubic to quadratic in the predictor order. At each stage of the recursion, it computes a reflection coefficient and updates the predictor coefficients while maintaining consistency with the Toeplitz structure. The recursion also produces the innovation variance, which quantifies the prediction error energy associated with the fitted model. This approach avoids explicitly forming the matrix $R$ and instead operates directly on the autocorrelation sequence, reflecting the algorithmic efficiency emphasized in the section.

Once the coefficients have been obtained, the `compute_predictions` function evaluates the predictor by computing fitted values and residuals according to Equations (13.7.7) and (13.7.8). This allows the model to be assessed in terms of empirical mean-square prediction error, providing a direct comparison with the least-squares formulation discussed earlier. The `one_step_ahead_forecast` function extends this by producing a forward prediction based on the most recent samples, illustrating the practical use of the estimated coefficients in extrapolation tasks.

To analyze the dynamical properties of the predictor, the program constructs the polynomial associated with Equation (13.7.19) and computes its roots using the `durand_kerner` function. This function implements an iterative root-finding method for complex polynomials, allowing the program to determine whether all roots lie inside the unit circle. The stability condition is then evaluated in the `fit_yule_walker` function by checking the magnitudes of the computed roots. This step connects the algebraic form of the predictor to its dynamical behavior and ensures that the fitted model corresponds to a stable autoregressive process.

The `fit_yule_walker` function integrates all major components of the implementation. It computes the autocorrelation sequence, applies the Levinson–Durbin recursion, evaluates the fitted values and residuals, constructs the prediction polynomial, computes its roots, and determines stability. This function serves as the central computational pipeline for the Yule–Walker approach, encapsulating both estimation and diagnostic analysis within a single interface.

The `generate_test_signal` function provides a controlled input sequence with autoregressive characteristics and mild oscillatory forcing. This allows the behavior of the Yule–Walker estimator to be examined under conditions that deviate slightly from ideal stationarity, highlighting how the method approximates both stochastic dependence and deterministic structure. The auxiliary printing functions format vectors, residual summaries, and root data to facilitate interpretation of the results.

The `main` function orchestrates the complete workflow. It generates the test signal, selects the predictor order, and invokes the Yule–Walker fitting procedure. It then reports the estimated autocorrelation sequence, predictor coefficients, reflection coefficients, innovation variance, and mean-square error. To provide further insight, it displays representative fitted values and residuals, prints the computed roots of the prediction polynomial, and evaluates the stability condition. Finally, it produces a one-step-ahead forecast using the fitted model. In doing so, the `main` function demonstrates how the structured Yule–Walker formulation supports both efficient computation and comprehensive model analysis.

```rust
// Program 13.7.2 Autocorrelation Formulation and the Yule-Walker System
//
// Problem Statement:
// Given a uniformly sampled scalar sequence y_0, y_1, ..., y_{N-1}, estimate the
// autocorrelation sequence r_k from equation (13.7.15), construct the Yule-Walker
// system
//
//     R a = r
//
// from equations (13.7.16) through (13.7.18), and solve it efficiently using the
// Levinson-Durbin recursion, which exploits the Toeplitz structure of the
// autocorrelation matrix. After obtaining the predictor coefficients, form the
// prediction polynomial
//
//     A(z) = 1 - sum_{k=1}^p a_k z^{-k}
//
// from equation (13.7.19), assess stability by examining the roots of the
// associated polynomial in lambda, compute one-step predictions and residuals,
// and report the innovation variance and forecast.
//
// This program is self-contained and can be run directly with `cargo run`.

use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

const EPS: f64 = 1.0e-12;
const ROOT_TOL: f64 = 1.0e-10;
const MAX_ROOT_ITERS: usize = 200;

/// Errors that may arise in autocorrelation estimation, Toeplitz solving, or root analysis.
#[derive(Debug)]
enum YwError {
    InvalidOrder { p: usize, n: usize },
    DegenerateAutocorrelation,
    RootSolverFailed,
}

impl fmt::Display for YwError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            YwError::InvalidOrder { p, n } => write!(
                f,
                "invalid predictor order p = {} for data length N = {}; require 1 <= p < N",
                p, n
            ),
            YwError::DegenerateAutocorrelation => write!(
                f,
                "autocorrelation is degenerate; Levinson-Durbin cannot proceed safely"
            ),
            YwError::RootSolverFailed => write!(
                f,
                "Durand-Kerner root iteration failed to converge"
            ),
        }
    }
}

impl Error for YwError {}

/// Minimal complex number type for polynomial root finding.
#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
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

use std::ops::{Add, Div, Mul, Sub};

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Div for Complex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }
}

impl Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Add<f64> for Complex {
    type Output = Self;
    fn add(self, rhs: f64) -> Self::Output {
        Self::new(self.re + rhs, self.im)
    }
}

impl Sub<f64> for Complex {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self::Output {
        Self::new(self.re - rhs, self.im)
    }
}

/// Result of the Yule-Walker solve.
#[derive(Debug, Clone)]
struct YuleWalkerModel {
    /// Predictor order p.
    order: usize,
    /// Estimated autocorrelations r_0, ..., r_p.
    autocorr: Vec<f64>,
    /// Predictor coefficients a_1, ..., a_p.
    coeffs: Vec<f64>,
    /// Reflection coefficients from Levinson-Durbin.
    reflection_coeffs: Vec<f64>,
    /// Innovation variance after order p.
    innovation_variance: f64,
    /// Fitted values for n = p, ..., N-1.
    fitted: Vec<f64>,
    /// Residuals e_n = y_n - yhat_n.
    residuals: Vec<f64>,
    /// Mean-square residual over valid indices.
    mse: f64,
    /// Roots lambda of lambda^p - a_1 lambda^{p-1} - ... - a_p = 0.
    roots: Vec<Complex>,
    /// Whether all roots lie strictly inside the unit disk.
    stable: bool,
}

/// Estimate the biased sample autocorrelation sequence
///
///     r_k = (1 / N) sum_{n=k}^{N-1} y_n y_{n-k},
///
/// for k = 0, 1, ..., max_lag.
///
/// The biased form is commonly used in autocorrelation-based Yule-Walker estimation
/// because it preserves positive semidefiniteness of the Toeplitz system.
fn biased_autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    let mut r = vec![0.0; max_lag + 1];

    for lag in 0..=max_lag {
        let mut sum = 0.0;
        for i in lag..n {
            sum += data[i] * data[i - lag];
        }
        r[lag] = sum / n as f64;
    }

    r
}

/// Solve the Yule-Walker system using the Levinson-Durbin recursion.
///
/// Input:
/// - `r`: autocorrelation sequence [r_0, r_1, ..., r_p]
///
/// Output:
/// - predictor coefficients [a_1, ..., a_p]
/// - reflection coefficients
/// - innovation variance
fn levinson_durbin(r: &[f64], p: usize) -> Result<(Vec<f64>, Vec<f64>, f64), YwError> {
    if r.len() < p + 1 || r[0].abs() <= EPS {
        return Err(YwError::DegenerateAutocorrelation);
    }

    let mut a = vec![0.0; p];
    let mut kappa = vec![0.0; p];
    let mut error = r[0];

    for m in 0..p {
        let mut acc = r[m + 1];
        for j in 0..m {
            acc -= a[j] * r[m - j];
        }

        if error.abs() <= EPS {
            return Err(YwError::DegenerateAutocorrelation);
        }

        let km = acc / error;
        kappa[m] = km;

        let mut new_a = a.clone();
        new_a[m] = km;
        for j in 0..m {
            new_a[j] = a[j] - km * a[m - 1 - j];
        }

        a = new_a;
        error *= 1.0 - km * km;

        if error <= EPS {
            return Err(YwError::DegenerateAutocorrelation);
        }
    }

    Ok((a, kappa, error))
}

/// Compute fitted values and residuals using
///
///     yhat_n = sum_{k=1}^p a_k y_{n-k},
///     e_n    = y_n - yhat_n.
fn compute_predictions(data: &[f64], coeffs: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
    let p = coeffs.len();
    let n = data.len();

    let mut fitted = Vec::with_capacity(n - p);
    let mut residuals = Vec::with_capacity(n - p);

    for idx in p..n {
        let mut yhat = 0.0;
        for k in 0..p {
            yhat += coeffs[k] * data[idx - 1 - k];
        }
        let e = data[idx] - yhat;
        fitted.push(yhat);
        residuals.push(e);
    }

    let mse = residuals.iter().map(|x| x * x).sum::<f64>() / residuals.len() as f64;
    (fitted, residuals, mse)
}

/// Compute a one-step-ahead forecast from the last p samples.
fn one_step_ahead_forecast(data: &[f64], coeffs: &[f64]) -> f64 {
    let p = coeffs.len();
    let n = data.len();

    let mut value = 0.0;
    for k in 0..p {
        value += coeffs[k] * data[n - 1 - k];
    }
    value
}

/// Evaluate a polynomial with real coefficients at a complex point.
///
/// Coefficients are ordered from highest degree to constant term.
fn eval_poly(coeffs: &[f64], z: Complex) -> Complex {
    let mut value = Complex::new(coeffs[0], 0.0);
    for &c in &coeffs[1..] {
        value = value * z + c;
    }
    value
}

/// Find all roots of a monic polynomial using Durand-Kerner iteration.
///
/// Polynomial format:
///     coeffs[0] x^n + coeffs[1] x^{n-1} + ... + coeffs[n]
///
/// with coeffs[0] = 1.
fn durand_kerner(coeffs: &[f64]) -> Result<Vec<Complex>, YwError> {
    let degree = coeffs.len() - 1;
    if degree == 0 {
        return Ok(vec![]);
    }

    let radius = 0.5;
    let mut roots = (0..degree)
        .map(|k| {
            let theta = 2.0 * PI * k as f64 / degree as f64;
            Complex::from_polar(radius, theta)
        })
        .collect::<Vec<_>>();

    for _ in 0..MAX_ROOT_ITERS {
        let mut max_update: f64 = 0.0;

        for i in 0..degree {
            let zi = roots[i];
            let mut denom = Complex::new(1.0, 0.0);
            for j in 0..degree {
                if i != j {
                    denom = denom * (zi - roots[j]);
                }
            }

            if denom.abs() <= EPS {
                denom = denom + Complex::new(1.0e-8, 1.0e-8);
            }

            let correction = eval_poly(coeffs, zi) / denom;
            roots[i] = zi - correction;
            max_update = max_update.max(correction.abs());
        }

        if max_update < ROOT_TOL {
            return Ok(roots);
        }
    }

    Err(YwError::RootSolverFailed)
}

/// Fit a Yule-Walker model of order p.
fn fit_yule_walker(data: &[f64], p: usize) -> Result<YuleWalkerModel, YwError> {
    let n = data.len();
    if p == 0 || p >= n {
        return Err(YwError::InvalidOrder { p, n });
    }

    let autocorr = biased_autocorrelation(data, p);
    let (coeffs, reflection_coeffs, innovation_variance) = levinson_durbin(&autocorr, p)?;
    let (fitted, residuals, mse) = compute_predictions(data, &coeffs);

    // Stability analysis:
    // A(z) = 1 - a_1 z^{-1} - ... - a_p z^{-p}.
    // Multiplying by z^p gives the polynomial in lambda = z^{-1}:
    //
    //     lambda^p - a_1 lambda^{p-1} - ... - a_p = 0.
    //
    // Stability of the AR recursion corresponds to all lambda-roots lying
    // strictly inside the unit circle.
    let mut poly = vec![1.0];
    for &a in &coeffs {
        poly.push(-a);
    }

    let roots = durand_kerner(&poly)?;
    let stable = roots.iter().all(|root| root.abs() < 1.0 - 1.0e-8);

    Ok(YuleWalkerModel {
        order: p,
        autocorr,
        coeffs,
        reflection_coeffs,
        innovation_variance,
        fitted,
        residuals,
        mse,
        roots,
        stable,
    })
}

/// Generate a test signal with autoregressive character and oscillatory forcing.
fn generate_test_signal(n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    y[0] = 0.35;
    y[1] = -0.10;

    for k in 2..n {
        let forcing = 0.04 * (2.0 * PI * k as f64 / 17.0).sin()
            + 0.02 * (2.0 * PI * k as f64 / 9.0).cos();
        y[k] = 1.45 * y[k - 1] - 0.72 * y[k - 2] + forcing;
    }

    y
}

/// Print a real vector with aligned indices.
fn print_real_vector(name: &str, values: &[f64]) {
    println!("{name}");
    for (i, value) in values.iter().enumerate() {
        println!("  [{:>2}] {:>.10}", i, value);
    }
}

/// Print the first few valid prediction rows.
fn print_fit_summary(data: &[f64], model: &YuleWalkerModel, max_rows: usize) {
    println!("\nFirst {} valid prediction rows:", max_rows);
    println!(
        "{:>6} {:>16} {:>16} {:>16}",
        "n", "y_n", "yhat_n", "e_n"
    );

    let rows = model.fitted.len().min(max_rows);
    for row in 0..rows {
        let n = model.order + row;
        println!(
            "{:>6} {:>16.10} {:>16.10} {:>16.10}",
            n,
            data[n],
            model.fitted[row],
            model.residuals[row]
        );
    }
}

/// Print roots and their moduli.
fn print_roots(roots: &[Complex]) {
    println!("\nRoots of lambda^p - a_1 lambda^(p-1) - ... - a_p = 0:");
    println!("{:>6} {:>18} {:>18} {:>18}", "idx", "Re", "Im", "|lambda|");
    for (i, z) in roots.iter().enumerate() {
        println!(
            "{:>6} {:>18.10} {:>18.10} {:>18.10}",
            i,
            z.re,
            z.im,
            z.abs()
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let data = generate_test_signal(120);
    let p = 8;

    println!("Yule-Walker Linear Prediction Demo");
    println!("==================================");
    println!("Data length N    = {}", data.len());
    println!("Predictor order p = {}", p);

    let model = fit_yule_walker(&data, p)?;
    let forecast = one_step_ahead_forecast(&data, &model.coeffs);

    println!("\nEstimated biased autocorrelation sequence:");
    print_real_vector("r_k =", &model.autocorr);

    println!("\nEstimated Yule-Walker predictor coefficients:");
    print_real_vector("a =", &model.coeffs);

    println!("\nReflection coefficients from Levinson-Durbin:");
    print_real_vector("kappa =", &model.reflection_coeffs);

    println!("\nInnovation variance:");
    println!("  sigma_e^2 = {:>.10}", model.innovation_variance);

    println!("\nMean-square prediction error from one-step residuals:");
    println!("  MSE = {:>.10}", model.mse);

    print_fit_summary(&data, &model, 10);
    print_roots(&model.roots);

    println!("\nStability assessment:");
    println!(
        "  Stable recursion? {}",
        if model.stable { "yes" } else { "no" }
    );

    println!("\nOne-step-ahead forecast based on the last p samples:");
    println!("  y_hat[N] = {:>.10}", forecast);

    Ok(())
}
```

Program 13.7.2 demonstrates the advantages of exploiting structure in linear prediction problems. By replacing a dense least-squares solve with a Toeplitz-aware recursion, the implementation achieves improved computational efficiency while maintaining numerical stability. The inclusion of reflection coefficients, innovation variance, and root-based stability analysis provides deeper insight into the behavior of the fitted model. This structured approach forms the foundation for more advanced methods in linear predictive coding and spectral estimation, where both efficiency and robustness are essential.

## 13.7.3. Computation, Stability, and Burg-Type Estimation

The numerical behavior of linear prediction depends strongly on the way the coefficients are estimated. There are two distinct notions of stability that matter. The first is model stability, meaning that the fitted predictor should define a stable autoregressive relation. The second is numerical stability, meaning that the algorithm used to estimate the coefficients should not be overly sensitive to finite precision, poorly conditioned covariance matrices, or sharply peaked spectra.

Ill-conditioning can arise when $p$ is large relative to the available data length, when the process contains very narrow spectral lines, or when the covariance estimate is itself noisy. In such cases, even small numerical errors can significantly perturb the fitted coefficients. This issue is especially relevant in applications where the estimated model is later used for spectral analysis or whitening, since spurious coefficient fluctuations can create artificial spectral structure. Recent work in gravitational-wave spectral estimation highlights this point clearly: in maximum-entropy spectral analysis, faster Burg-style implementations can trade robustness for speed, and on real data the faster path was observed to introduce additional noise into PSD estimates relative to a more stable implementation (Martini et al., 2024).

A practically important response is to favor algorithms that preserve a stable model throughout the recursion rather than estimating coefficients first and attempting to repair unstable roots afterward. Burg’s method is a well-known example. Instead of solving the Yule-Walker equations from an autocorrelation estimate, it updates forward and backward prediction errors recursively and often exhibits strong practical stability properties. This makes it attractive not only for spectral estimation but also for real-time or resource-constrained applications. For example, recent work has considered optimized implementations of Burg’s method for real-time packet loss concealment in networked music performance systems, where computational efficiency and predictable behavior are both important (Sacchetto, Rottondi and Bianco, 2024).

From a numerical computing standpoint, the broader lesson is that structure must be preserved all the way through the algorithm. It is not enough to recognize that the covariance matrix is Toeplitz in principle. One must also choose estimation and solution procedures that respect that structure and avoid unnecessary densification, because the gain is not merely asymptotic elegance but real improvements in speed, memory use, and robustness.

### Rust Implementation

Following the discussion in Section 13.7.3 on the numerical behavior of linear prediction and the advantages of Burg-type estimation, Program 13.7.3 provides a practical implementation of coefficient estimation using forward and backward error recursions. Unlike the least-squares formulation of Section 13.7.1 or the autocorrelation-based Yule–Walker system of Section 13.7.2, Burg’s method constructs the predictor incrementally while maintaining stability at each stage of the recursion. This program implements that approach, computes the resulting predictor coefficients, evaluates residuals and prediction error, and analyzes stability through the roots of the associated prediction polynomial. In doing so, it demonstrates how Burg’s method translates the theoretical emphasis on stability and structure preservation into a robust computational procedure.

At the core of the implementation is the `burg_estimate` function, which performs the recursive estimation of predictor coefficients. Instead of forming an autocorrelation sequence as in Equation (13.7.15) or solving the Toeplitz system in Equation (13.7.16), this function updates forward and backward prediction errors directly. At each stage of the recursion, a reflection coefficient is computed from these error sequences, and the predictor coefficients are updated accordingly. This procedure ensures that the magnitude of each reflection coefficient remains less than one under nondegenerate conditions, which promotes stability of the resulting autoregressive model. The function also updates the innovation variance, providing a measure of the prediction error energy accumulated across the recursion.

The implementation returns coefficients in the internal recursion convention, and these are converted in `fit_burg_model` to match the predictor form of Equation (13.7.7). This sign adjustment is essential for consistency with the prediction and residual definitions used elsewhere in the section. The `compute_predictions` function then evaluates the fitted values and residuals according to Equations (13.7.7) and (13.7.8), allowing direct assessment of model accuracy through the empirical mean-square error. The `one_step_ahead_forecast` function further demonstrates how the fitted model can be used for extrapolation by applying the predictor to the most recent samples.

To examine model stability, the program constructs the polynomial corresponding to Equation (13.7.19) and computes its roots using the `durand_kerner` function. This function implements an iterative complex root-finding method, enabling the program to determine whether all roots lie strictly within the unit circle. The stability condition is then evaluated in `fit_burg_model`, connecting the recursive estimation procedure to the dynamical behavior of the resulting model. This step reflects the central theme of the section: that stability should be preserved during estimation rather than imposed afterward.

The `fit_burg_model` function integrates the full estimation and analysis pipeline. It calls the Burg recursion, converts the coefficients to the predictor convention, computes residuals and mean-square error, constructs the prediction polynomial, determines its roots, and evaluates stability. This function serves as the primary interface for Burg-based linear prediction, combining efficiency, numerical robustness, and diagnostic insight in a single workflow.

The `generate_test_signal` function provides a controlled input sequence with autoregressive characteristics and mild oscillatory forcing, allowing the behavior of the Burg estimator to be examined under realistic but nonideal conditions. Auxiliary functions such as `print_real_vector`, `print_fit_summary`, and `print_roots` format the output to highlight key numerical quantities, including coefficients, residuals, and root magnitudes.

The `main` function orchestrates the entire process. It generates the input data, selects the predictor order, and invokes the Burg estimation pipeline. It then reports the estimated coefficients, reflection coefficients, innovation variance, and mean-square prediction error. To provide further insight, it prints representative fitted values and residuals, displays the roots of the prediction polynomial, and evaluates the stability condition. Finally, it produces a one-step-ahead forecast. Through this sequence, the `main` function demonstrates how Burg’s method supports stable, efficient, and interpretable linear prediction in practice.

```rust
// Program 13.7.3 Burg-Type Estimation, Stability, and Prediction Diagnostics
//
// Problem Statement:
// Implement Burg's method for p-th order linear prediction of a uniformly sampled
// scalar sequence. The program estimates autoregressive predictor coefficients by
// recursively updating forward and backward prediction errors, without explicitly
// forming an autocorrelation matrix or dense normal equations. It reports the
// reflection coefficients, the final predictor coefficients, the innovation
// variance, one-step fitted values, residuals, and a stability assessment based on
// the roots of the prediction polynomial
//
//     A(z) = 1 - sum_{k=1}^p a_k z^{-k}.
//
// The implementation illustrates how Burg-type estimation can preserve model
// stability throughout the recursion while remaining computationally efficient.
//
// This program is self-contained and can be run directly with `cargo run`.

use std::error::Error;
use std::f64::consts::PI;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

const EPS: f64 = 1.0e-12;
const ROOT_TOL: f64 = 1.0e-10;
const MAX_ROOT_ITERS: usize = 200;

/// Errors that may arise during Burg estimation or root analysis.
#[derive(Debug)]
enum BurgError {
    InvalidOrder { p: usize, n: usize },
    DegenerateSignal,
    RootSolverFailed,
}

impl fmt::Display for BurgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BurgError::InvalidOrder { p, n } => write!(
                f,
                "invalid predictor order p = {} for data length N = {}; require 1 <= p < N - 1",
                p, n
            ),
            BurgError::DegenerateSignal => write!(
                f,
                "signal is degenerate or recursion encountered a zero-energy denominator"
            ),
            BurgError::RootSolverFailed => {
                write!(f, "polynomial root iteration failed to converge")
            }
        }
    }
}

impl Error for BurgError {}

/// Minimal complex type used for root computations.
#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
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
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }
}

impl Add<f64> for Complex {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        Self::new(self.re + rhs, self.im)
    }
}

impl Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

/// Results produced by Burg estimation.
#[derive(Debug, Clone)]
struct BurgModel {
    /// Predictor order.
    order: usize,
    /// Predictor coefficients [a1, ..., ap] in the form
    /// yhat_n = sum_{k=1}^p a_k y_{n-k}.
    coeffs: Vec<f64>,
    /// Reflection coefficients generated during the recursion.
    reflection_coeffs: Vec<f64>,
    /// Innovation variance estimate.
    innovation_variance: f64,
    /// Forward fitted values for indices n = p, ..., N-1.
    fitted: Vec<f64>,
    /// Residuals e_n = y_n - yhat_n.
    residuals: Vec<f64>,
    /// Mean-square residual on valid indices.
    mse: f64,
    /// Roots of lambda^p - a1 lambda^(p-1) - ... - ap = 0.
    roots: Vec<Complex>,
    /// Whether all roots lie strictly inside the unit disk.
    stable: bool,
}

/// Estimate Burg coefficients of order `p`.
///
/// This routine returns coefficients in the recursion-native sign convention.
/// They are converted afterward into the predictor form
///
///     yhat_n = sum_{k=1}^p a_k y_{n-k}
///
/// by negating them in `fit_burg_model`.
fn burg_estimate(data: &[f64], p: usize) -> Result<(Vec<f64>, Vec<f64>, f64), BurgError> {
    let n = data.len();
    if p == 0 || p >= n - 1 {
        return Err(BurgError::InvalidOrder { p, n });
    }

    let signal_energy = data.iter().map(|x| x * x).sum::<f64>() / n as f64;
    if signal_energy <= EPS {
        return Err(BurgError::DegenerateSignal);
    }

    // Forward and backward error arrays at recursion order 0.
    let mut ef = data.to_vec();
    let mut eb = data.to_vec();

    // AR coefficients in Burg's internal sign convention.
    let mut a: Vec<f64> = Vec::new();
    let mut reflection_coeffs = Vec::with_capacity(p);

    // Initial prediction error power.
    let mut error = signal_energy;

    for m in 0..p {
        let mut num = 0.0;
        let mut den = 0.0;

        for n_idx in (m + 1)..n {
            num += ef[n_idx] * eb[n_idx - 1];
            den += ef[n_idx] * ef[n_idx] + eb[n_idx - 1] * eb[n_idx - 1];
        }

        if den <= EPS {
            return Err(BurgError::DegenerateSignal);
        }

        // Burg reflection coefficient.
        let k = -2.0 * num / den;
        reflection_coeffs.push(k);

        // Update AR coefficients in recursion-native sign convention.
        let mut a_new = vec![0.0; m + 1];
        for j in 0..m {
            a_new[j] = a[j] + k * a[m - 1 - j];
        }
        a_new[m] = k;
        a = a_new;

        // Update forward and backward errors for the next order.
        let mut ef_new = ef.clone();
        let mut eb_new = eb.clone();

        for n_idx in (m + 1)..n {
            let f_old = ef[n_idx];
            let b_old = eb[n_idx - 1];
            ef_new[n_idx] = f_old + k * b_old;
            eb_new[n_idx - 1] = b_old + k * f_old;
        }

        ef = ef_new;
        eb = eb_new;

        error *= 1.0 - k * k;
        if error <= EPS {
            return Err(BurgError::DegenerateSignal);
        }
    }

    Ok((a, reflection_coeffs, error))
}

/// Compute fitted values and residuals using the predictor
///
///     yhat_n = sum_{k=1}^p a_k y_{n-k}.
fn compute_predictions(data: &[f64], coeffs: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
    let p = coeffs.len();
    let n = data.len();

    let mut fitted = Vec::with_capacity(n - p);
    let mut residuals = Vec::with_capacity(n - p);

    for idx in p..n {
        let mut yhat = 0.0;
        for k in 0..p {
            yhat += coeffs[k] * data[idx - 1 - k];
        }
        let e = data[idx] - yhat;
        fitted.push(yhat);
        residuals.push(e);
    }

    let mse = residuals.iter().map(|e| e * e).sum::<f64>() / residuals.len() as f64;
    (fitted, residuals, mse)
}

/// Produce a one-step-ahead forecast from the last p samples.
fn one_step_ahead_forecast(data: &[f64], coeffs: &[f64]) -> f64 {
    let p = coeffs.len();
    let n = data.len();

    let mut forecast = 0.0;
    for k in 0..p {
        forecast += coeffs[k] * data[n - 1 - k];
    }
    forecast
}

/// Evaluate a real-coefficient polynomial at a complex point.
fn eval_poly(coeffs: &[f64], z: Complex) -> Complex {
    let mut value = Complex::new(coeffs[0], 0.0);
    for &c in &coeffs[1..] {
        value = value * z + c;
    }
    value
}

/// Find all roots of a monic polynomial using Durand-Kerner iteration.
fn durand_kerner(coeffs: &[f64]) -> Result<Vec<Complex>, BurgError> {
    let degree = coeffs.len() - 1;
    if degree == 0 {
        return Ok(vec![]);
    }

    let mut roots = (0..degree)
        .map(|k| {
            let theta = 2.0 * PI * k as f64 / degree as f64;
            Complex::from_polar(0.5, theta)
        })
        .collect::<Vec<_>>();

    for _ in 0..MAX_ROOT_ITERS {
        let mut max_update: f64 = 0.0;

        for i in 0..degree {
            let zi = roots[i];
            let mut denom = Complex::new(1.0, 0.0);

            for j in 0..degree {
                if i != j {
                    denom = denom * (zi - roots[j]);
                }
            }

            if denom.abs() <= EPS {
                denom = denom + Complex::new(1.0e-8, 1.0e-8);
            }

            let correction = eval_poly(coeffs, zi) / denom;
            roots[i] = zi - correction;
            max_update = max_update.max(correction.abs());
        }

        if max_update < ROOT_TOL {
            return Ok(roots);
        }
    }

    Err(BurgError::RootSolverFailed)
}

/// Fit a complete Burg model, including root-based stability analysis.
fn fit_burg_model(data: &[f64], p: usize) -> Result<BurgModel, BurgError> {
    let (raw_coeffs, reflection_coeffs, innovation_variance) = burg_estimate(data, p)?;

    // Convert from Burg's recursion-native sign convention to the predictor form
    //
    //     yhat_n = sum_{k=1}^p a_k y_{n-k}.
    //
    let coeffs: Vec<f64> = raw_coeffs.iter().map(|&x| -x).collect();

    let (fitted, residuals, mse) = compute_predictions(data, &coeffs);

    // If
    //     A(z) = 1 - a1 z^{-1} - ... - ap z^{-p},
    // then multiplying by z^p gives the polynomial
    //
    //     lambda^p - a1 lambda^(p-1) - ... - ap = 0
    //
    // in lambda = z^{-1}. Stability requires all lambda-roots to satisfy |lambda| < 1.
    let mut poly = vec![1.0];
    for &a in &coeffs {
        poly.push(-a);
    }

    let roots = durand_kerner(&poly)?;
    let stable = roots.iter().all(|root| root.abs() < 1.0 - 1.0e-8);

    Ok(BurgModel {
        order: p,
        coeffs,
        reflection_coeffs,
        innovation_variance,
        fitted,
        residuals,
        mse,
        roots,
        stable,
    })
}

/// Generate a deterministic test signal with autoregressive character and oscillatory forcing.
fn generate_test_signal(n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    y[0] = 0.35;
    y[1] = -0.10;

    for k in 2..n {
        let forcing = 0.04 * (2.0 * PI * k as f64 / 17.0).sin()
            + 0.02 * (2.0 * PI * k as f64 / 9.0).cos();
        y[k] = 1.45 * y[k - 1] - 0.72 * y[k - 2] + forcing;
    }

    y
}

/// Print a real-valued vector with aligned indices.
fn print_real_vector(name: &str, values: &[f64]) {
    println!("{name}");
    for (i, value) in values.iter().enumerate() {
        println!("  [{:>2}] {:>.10}", i, value);
    }
}

/// Print the first few fitted values and residuals.
fn print_fit_summary(data: &[f64], model: &BurgModel, max_rows: usize) {
    println!("\nFirst {} valid prediction rows:", max_rows);
    println!(
        "{:>6} {:>16} {:>16} {:>16}",
        "n", "y_n", "yhat_n", "e_n"
    );

    let rows = model.fitted.len().min(max_rows);
    for row in 0..rows {
        let n = model.order + row;
        println!(
            "{:>6} {:>16.10} {:>16.10} {:>16.10}",
            n,
            data[n],
            model.fitted[row],
            model.residuals[row]
        );
    }
}

/// Print the roots and their moduli.
fn print_roots(roots: &[Complex]) {
    println!("\nRoots of lambda^p - a_1 lambda^(p-1) - ... - a_p = 0:");
    println!("{:>6} {:>18} {:>18} {:>18}", "idx", "Re", "Im", "|lambda|");
    for (i, z) in roots.iter().enumerate() {
        println!(
            "{:>6} {:>18.10} {:>18.10} {:>18.10}",
            i,
            z.re,
            z.im,
            z.abs()
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let data = generate_test_signal(120);
    let p = 8;

    println!("Burg Linear Prediction Demo");
    println!("===========================");
    println!("Data length N     = {}", data.len());
    println!("Predictor order p = {}", p);

    let model = fit_burg_model(&data, p)?;
    let forecast = one_step_ahead_forecast(&data, &model.coeffs);

    println!("\nEstimated Burg predictor coefficients:");
    print_real_vector("a =", &model.coeffs);

    println!("\nReflection coefficients:");
    print_real_vector("kappa =", &model.reflection_coeffs);

    println!("\nInnovation variance:");
    println!("  sigma_e^2 = {:>.10}", model.innovation_variance);

    println!("\nMean-square prediction error from one-step residuals:");
    println!("  MSE = {:>.10}", model.mse);

    print_fit_summary(&data, &model, 10);
    print_roots(&model.roots);

    println!("\nStability assessment:");
    println!(
        "  Stable recursion? {}",
        if model.stable { "yes" } else { "no" }
    );

    println!("\nOne-step-ahead forecast based on the last p samples:");
    println!("  y_hat[N] = {:>.10}", forecast);

    Ok(())
}
```

Program 13.7.3 illustrates how numerical stability can be embedded directly into the estimation procedure rather than treated as a postprocessing concern. By avoiding explicit covariance estimation and instead updating prediction errors recursively, Burg’s method maintains stability and reduces sensitivity to ill-conditioning. The resulting implementation highlights the importance of structure-preserving algorithms in numerical linear prediction and provides a robust foundation for applications in spectral estimation, coding, and real-time signal processing.

## 13.7.4. Linear Predictive Coding and Practical Uses

Linear predictive coding turns prediction into compression. Rather than storing or transmitting the entire waveform directly, one stores a compact representation consisting of the predictor coefficients together with a representation of the residual or excitation. In the classical speech setting, this matches the source-filter interpretation of speech production: the coefficients describe the slowly varying vocal-tract filter, while the residual carries the fine-scale excitation information. Even modern neural speech and audio codecs continue to use LPC-like front ends because a low-parameter physically meaningful predictor remains extremely effective at removing short-term correlation before learned modeling stages are applied to the residual (Kim and Skoglund, 2024).

In a clean frame-based formulation, let a frame consist of samples

$$\{y_n\}_{n=0}^{L-1} \tag{13.7.20}$$

Then an LPC encoder proceeds in four stages. First, estimate coefficients $a\in\mathbb{R}^p$ from the frame, typically using autocorrelation, least squares, or Burg estimation. Second, compute the residual,

$$e_n = y_n - \sum_{k=1}^{p} a_k\, y_{n-k} \tag{13.7.21}$$

with appropriate treatment of the warm-up indices $n<p$. Third, quantize and encode the coefficient vector and either the residual itself or a low-dimensional excitation model. Fourth, reconstruct at the decoder by the recursion:

$$\widehat{y}_n = \sum_{k=1}^{p} a_k\, \widehat{y}_{n-k} + \widehat{e}_n \tag{13.7.22}$$

An important numerical property of this reconstruction is that, if the quantized residual is explicitly available, the recursion reproduces the frame exactly in the quantized domain. Consequently, the decoder does not accumulate the kind of long-horizon floating-point drift that pure extrapolation schemes can suffer.

Although speech coding is the classical application, LPC is now used more broadly as a compact descriptor of short-term dynamics. In industrial fault diagnosis, a 2025 study used LPC-derived features together with a simple feedforward neural network for rolling-bearing defect diagnosis, motivated by the scarcity of labeled data and the need for efficient feature extraction; the authors report very strong performance on standard benchmark datasets (Sinitsin and Eremeeva, 2025). In biomedical signal analysis, Jamshidi et al. (2025) used resting-state EEG together with an LPC-based LEAPD algorithm to predict three-year mortality status in a Parkinson’s disease cohort, reporting high single-channel leave-one-out cross-validation accuracies, strong robustness under truncation, and promising multi-channel out-of-sample results. These examples are instructive because they show LPC functioning not as an audio codec component but as a compact and interpretable representation of temporal dynamics.

From an implementation perspective, Rust naturally supports two computational modes. For small predictor orders, one may safely use a dense path in which autocorrelations are formed explicitly and the resulting positive-definite system is solved by a Cholesky factorization. For larger orders, or in small-sample regimes where allocations and condition numbers matter more acutely, a structured Toeplitz-aware path is preferable. In practice, this means using (f64) arithmetic, careful accumulation of autocorrelations, explicit stability checks before using predictors for extrapolation, and frame-based processing so that the model can adapt to nonstationarity while keeping the linear systems moderate in size. Modern Toeplitz-based covariance and maximum-likelihood methods reinforce the point that structured linear algebra here is not a minor optimization, but often the difference between a feasible and an impractical computation (Cederberg, 2024).

In summary, linear prediction and LPC should be viewed not as isolated classical DSP topics, but as general numerical tools for fitting compact linear dynamical models to finite data. The same predictor coefficients can support extrapolation, coding, denoising, spectral estimation, and feature extraction. What makes the method powerful is precisely the combination of a simple model class, structured covariance algebra, and efficient solvers.

### Rust Implementation

Following the discussion in Section 13.7.4 on linear predictive coding and its interpretation as a compact representation of short-term dynamics, Program 13.7.4 provides a complete frame-based implementation of LPC analysis, quantization, and reconstruction. In contrast to the earlier sections, where the focus was on estimating predictor coefficients, this program demonstrates how those coefficients are used in practice to encode and reconstruct a signal. Each frame is processed independently: coefficients are estimated from the data, a residual sequence is computed, both quantities are quantized, and the signal is reconstructed using the recursion of Equation (13.7.22). This implementation illustrates how prediction becomes compression by separating structured signal content from a typically lower-energy excitation.

At the core of the implementation is the estimation of predictor coefficients using the autocorrelation formulation and Levinson–Durbin recursion. The `biased_autocorrelation` function computes the lag-dependent correlations for each frame, providing the sequence required for the Yule–Walker system. The `levinson_durbin` function then solves this system efficiently, producing the predictor coefficients and reflection coefficients while maintaining the structured Toeplitz formulation discussed earlier. The innovation variance returned by this function reflects the residual energy associated with the fitted model and serves as a diagnostic for model quality.

The `compute_residual` function implements the residual definition in Equation (13.7.21). For each sample in the frame, it subtracts the predicted value formed from the preceding samples within the same frame. The implementation carefully handles the initial indices $n<p$ by limiting the summation range, ensuring that only available past samples are used. This produces a residual sequence that captures the unpredictable component of the signal and forms the excitation in the LPC representation.

Reconstruction is carried out by the `reconstruct_frame` function, which directly implements the recursion in Equation (13.7.22). Using the quantized predictor coefficients and quantized residual, it reconstructs each sample from previously reconstructed values within the frame. This step demonstrates the key numerical property of LPC: when the residual is available, reconstruction reproduces the signal accurately within the limits of quantization, and errors do not accumulate across time in the same way as in purely predictive extrapolation schemes.

Quantization is handled by the `quantize_uniform` function, which applies a simple uniform scalar quantizer to both coefficients and residuals. Although basic, this approach clearly illustrates the trade-off between compression and accuracy. The quantized coefficients represent the slowly varying structure of the signal, while the quantized residual captures finer-scale variations. The resulting reconstruction error is controlled by the quantization step sizes, which can be adjusted to balance fidelity and storage cost.

The `encode_decode_frame` function integrates these steps into a complete encoding and decoding pipeline for a single frame. It performs coefficient estimation, residual computation, quantization, reconstruction, and error evaluation. It also checks stability using the reflection coefficients, ensuring that the predictor remains well behaved. The function returns both the encoded representation and detailed diagnostics, including pre- and post-quantization errors.

The `encode_decode_signal` function extends this process to an entire signal by dividing it into frames and processing each independently. This frame-based approach allows the model to adapt to nonstationary behavior while keeping each estimation problem small and well conditioned. The function aggregates reconstructed samples and computes global error metrics, providing an overall assessment of coding performance.

The `generate_test_signal` function produces a synthetic signal with slowly varying dynamics and mild excitation, making it suitable for demonstrating frame-based adaptation. Auxiliary printing functions present coefficients, residuals, and reconstructed samples in a structured format to facilitate interpretation.

The `main` function orchestrates the complete LPC workflow. It generates the test signal, sets frame length, predictor order, and quantization parameters, and invokes the frame-based encoding and decoding process. It then reports global reconstruction error, per-frame diagnostics, predictor coefficients, and representative sample comparisons. Through this process, the `main` function demonstrates how LPC combines modeling, compression, and reconstruction into a unified computational framework.

```rust
// Program 13.7.4 Linear Predictive Coding with Frame-Based Analysis, Quantization, and Reconstruction
//
// Problem Statement:
// Implement a frame-based linear predictive coding pipeline for a uniformly sampled
// scalar signal. For each frame {y_n}_{n=0}^{L-1}, estimate predictor coefficients
// a_1, ..., a_p using the autocorrelation formulation and Levinson-Durbin recursion,
// compute the residual
//
//     e_n = y_n - sum_{k=1}^p a_k y_{n-k},
//
// quantize both the predictor coefficients and the residual, and reconstruct the
// frame at the decoder by the recursion
//
//     yhat_n = sum_{k=1}^p a_k yhat_{n-k} + ehat_n.
//
// The program reports per-frame coding diagnostics, reconstruction error, coefficient
// stability information, and aggregate statistics over the entire signal. The goal
// is to demonstrate how linear prediction becomes a practical coding scheme once the
// predictor and excitation are stored in compact form.
//
// This program is self-contained and can be run directly with `cargo run`.

use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

const EPS: f64 = 1.0e-12;

/// Errors that may arise during LPC analysis or reconstruction.
#[derive(Debug)]
enum LpcError {
    InvalidOrder { p: usize, frame_len: usize },
    DegenerateAutocorrelation,
    InvalidFrameLength,
}

impl fmt::Display for LpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LpcError::InvalidOrder { p, frame_len } => write!(
                f,
                "invalid predictor order p = {} for frame length L = {}; require 1 <= p < L",
                p, frame_len
            ),
            LpcError::DegenerateAutocorrelation => write!(
                f,
                "autocorrelation sequence is degenerate; Levinson-Durbin recursion failed"
            ),
            LpcError::InvalidFrameLength => write!(f, "frame length must be positive"),
        }
    }
}

impl Error for LpcError {}

/// Result of Yule-Walker analysis for one frame.
#[derive(Debug, Clone)]
struct YuleWalkerResult {
    coeffs: Vec<f64>,
    reflection_coeffs: Vec<f64>,
    innovation_variance: f64,
}

/// Encoded representation of a single frame.
#[derive(Debug, Clone)]
struct EncodedFrame {
    original: Vec<f64>,
    coeffs: Vec<f64>,
    quantized_coeffs: Vec<f64>,
    residual: Vec<f64>,
    quantized_residual: Vec<f64>,
    reconstructed: Vec<f64>,
    reflection_coeffs: Vec<f64>,
    innovation_variance: f64,
    mse_before_quantization: f64,
    mse_after_quantization: f64,
    stable: bool,
}

/// Aggregate coding summary.
#[derive(Debug, Clone)]
struct CodingSummary {
    frames: Vec<EncodedFrame>,
    original_signal: Vec<f64>,
    reconstructed_signal: Vec<f64>,
    global_mse: f64,
    max_abs_error: f64,
}

/// Compute the biased autocorrelation sequence r_0, ..., r_p for one frame.
///
/// The biased form is used because it preserves the positive semidefinite
/// Toeplitz structure needed by Levinson-Durbin.
fn biased_autocorrelation(frame: &[f64], p: usize) -> Vec<f64> {
    let n = frame.len();
    let mut r = vec![0.0; p + 1];

    for lag in 0..=p {
        let mut sum = 0.0;
        for i in lag..n {
            sum += frame[i] * frame[i - lag];
        }
        r[lag] = sum / n as f64;
    }

    r
}

/// Solve the Yule-Walker system via Levinson-Durbin recursion.
fn levinson_durbin(r: &[f64], p: usize) -> Result<YuleWalkerResult, LpcError> {
    if r.len() < p + 1 || r[0].abs() <= EPS {
        return Err(LpcError::DegenerateAutocorrelation);
    }

    let mut a = vec![0.0; p];
    let mut kappa = vec![0.0; p];
    let mut error = r[0];

    for m in 0..p {
        let mut acc = r[m + 1];
        for j in 0..m {
            acc -= a[j] * r[m - j];
        }

        if error.abs() <= EPS {
            return Err(LpcError::DegenerateAutocorrelation);
        }

        let km = acc / error;
        kappa[m] = km;

        let mut new_a = a.clone();
        new_a[m] = km;
        for j in 0..m {
            new_a[j] = a[j] - km * a[m - 1 - j];
        }

        a = new_a;
        error *= 1.0 - km * km;

        if error <= EPS {
            return Err(LpcError::DegenerateAutocorrelation);
        }
    }

    Ok(YuleWalkerResult {
        coeffs: a,
        reflection_coeffs: kappa,
        innovation_variance: error,
    })
}

/// Compute LPC residual for one frame according to
///
///     e_n = y_n - sum_{k=1}^p a_k y_{n-k},
///
/// with warm-up handling: only samples inside the current frame are used.
fn compute_residual(frame: &[f64], coeffs: &[f64]) -> Vec<f64> {
    let p = coeffs.len();
    let n = frame.len();
    let mut residual = vec![0.0; n];

    for i in 0..n {
        let mut prediction = 0.0;
        let max_k = p.min(i);
        for k in 1..=max_k {
            prediction += coeffs[k - 1] * frame[i - k];
        }
        residual[i] = frame[i] - prediction;
    }

    residual
}

/// Reconstruct a frame according to
///
///     yhat_n = sum_{k=1}^p a_k yhat_{n-k} + ehat_n,
///
/// again using only already reconstructed samples inside the current frame.
fn reconstruct_frame(quantized_coeffs: &[f64], quantized_residual: &[f64]) -> Vec<f64> {
    let p = quantized_coeffs.len();
    let n = quantized_residual.len();
    let mut reconstructed = vec![0.0; n];

    for i in 0..n {
        let mut prediction = 0.0;
        let max_k = p.min(i);
        for k in 1..=max_k {
            prediction += quantized_coeffs[k - 1] * reconstructed[i - k];
        }
        reconstructed[i] = prediction + quantized_residual[i];
    }

    reconstructed
}

/// Uniform mid-tread scalar quantization.
///
/// Each value x is mapped to round(x / step) * step.
fn quantize_uniform(values: &[f64], step: f64) -> Vec<f64> {
    if step <= 0.0 {
        return values.to_vec();
    }

    values
        .iter()
        .map(|&x| (x / step).round() * step)
        .collect()
}

/// Mean-square error between two equal-length vectors.
fn mean_square_error(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f64>()
        / x.len() as f64
}

/// Maximum absolute error between two equal-length vectors.
fn max_abs_error(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

/// Check whether all reflection coefficients satisfy |kappa_m| < 1,
/// which is a practical stability test for the predictor.
fn is_stable_from_reflection_coeffs(reflection_coeffs: &[f64]) -> bool {
    reflection_coeffs.iter().all(|&k| k.abs() < 1.0 - 1.0e-8)
}

/// Encode and decode a single frame.
fn encode_decode_frame(
    frame: &[f64],
    order: usize,
    coeff_step: f64,
    residual_step: f64,
) -> Result<EncodedFrame, LpcError> {
    if frame.is_empty() {
        return Err(LpcError::InvalidFrameLength);
    }
    if order == 0 || order >= frame.len() {
        return Err(LpcError::InvalidOrder {
            p: order,
            frame_len: frame.len(),
        });
    }

    let autocorr = biased_autocorrelation(frame, order);
    let yw = levinson_durbin(&autocorr, order)?;

    let residual = compute_residual(frame, &yw.coeffs);
    let quantized_coeffs = quantize_uniform(&yw.coeffs, coeff_step);
    let quantized_residual = quantize_uniform(&residual, residual_step);
    let reconstructed = reconstruct_frame(&quantized_coeffs, &quantized_residual);

    let mse_before_quantization = mean_square_error(frame, &reconstruct_frame(&yw.coeffs, &residual));
    let mse_after_quantization = mean_square_error(frame, &reconstructed);
    let stable = is_stable_from_reflection_coeffs(&yw.reflection_coeffs);

    Ok(EncodedFrame {
        original: frame.to_vec(),
        coeffs: yw.coeffs,
        quantized_coeffs,
        residual,
        quantized_residual,
        reconstructed,
        reflection_coeffs: yw.reflection_coeffs,
        innovation_variance: yw.innovation_variance,
        mse_before_quantization,
        mse_after_quantization,
        stable,
    })
}

/// Process a complete signal frame by frame.
fn encode_decode_signal(
    signal: &[f64],
    frame_len: usize,
    order: usize,
    coeff_step: f64,
    residual_step: f64,
) -> Result<CodingSummary, LpcError> {
    if frame_len == 0 {
        return Err(LpcError::InvalidFrameLength);
    }

    let mut frames = Vec::new();
    let mut reconstructed_signal = Vec::with_capacity(signal.len());

    for chunk in signal.chunks(frame_len) {
        if chunk.len() <= order {
            break;
        }

        let encoded = encode_decode_frame(chunk, order, coeff_step, residual_step)?;
        reconstructed_signal.extend_from_slice(&encoded.reconstructed);
        frames.push(encoded);
    }

    let compared_len = reconstructed_signal.len();
    let original_used = signal[..compared_len].to_vec();
    let global_mse = mean_square_error(&original_used, &reconstructed_signal);
    let max_error = max_abs_error(&original_used, &reconstructed_signal);

    Ok(CodingSummary {
        frames,
        original_signal: original_used,
        reconstructed_signal,
        global_mse,
        max_abs_error: max_error,
    })
}

/// Generate a synthetic signal with slowly varying short-term dynamics.
/// This is useful for demonstrating frame-based LPC adaptation.
fn generate_test_signal(n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    y[0] = 0.20;
    y[1] = -0.05;

    for k in 2..n {
        let t = k as f64 / n as f64;

        // Slowly varying AR-like coefficients across the signal.
        let a1 = 1.35 + 0.10 * (2.0 * PI * t).sin();
        let a2 = -0.62 - 0.08 * (2.0 * PI * t).cos();

        // Mild excitation combining harmonic and pulse-like components.
        let harmonic = 0.03 * (2.0 * PI * k as f64 / 23.0).sin()
            + 0.02 * (2.0 * PI * k as f64 / 11.0).cos();
        let pulse = if k % 40 == 0 { 0.08 } else { 0.0 };

        y[k] = a1 * y[k - 1] + a2 * y[k - 2] + harmonic + pulse;
    }

    y
}

/// Print a vector with aligned indices.
fn print_vector(name: &str, values: &[f64]) {
    println!("{name}");
    for (i, value) in values.iter().enumerate() {
        println!("  [{:>2}] {:>.10}", i, value);
    }
}

/// Print the first few samples of a frame comparison.
fn print_frame_samples(frame: &EncodedFrame, max_rows: usize) {
    println!(
        "{:>6} {:>16} {:>16} {:>16} {:>16}",
        "n", "y_n", "e_n", "ehat_n", "yhat_n"
    );

    let rows = frame.original.len().min(max_rows);
    for i in 0..rows {
        println!(
            "{:>6} {:>16.10} {:>16.10} {:>16.10} {:>16.10}",
            i,
            frame.original[i],
            frame.residual[i],
            frame.quantized_residual[i],
            frame.reconstructed[i]
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let signal_len = 320;
    let frame_len = 64;
    let order = 10;
    let coeff_step = 0.005;
    let residual_step = 0.0025;

    let signal = generate_test_signal(signal_len);
    let summary = encode_decode_signal(&signal, frame_len, order, coeff_step, residual_step)?;

    println!("Linear Predictive Coding Demo");
    println!("=============================");
    println!("Signal length         = {}", signal_len);
    println!("Frame length          = {}", frame_len);
    println!("Predictor order       = {}", order);
    println!("Coefficient step      = {:>.6}", coeff_step);
    println!("Residual step         = {:>.6}", residual_step);
    println!("Number of frames used = {}", summary.frames.len());

    println!("\nGlobal reconstruction diagnostics:");
    println!("  Global MSE          = {:>.10}", summary.global_mse);
    println!("  Max abs. error      = {:>.10}", summary.max_abs_error);

    for (frame_idx, frame) in summary.frames.iter().enumerate() {
        println!("\nFrame {}", frame_idx);
        println!("-------");
        println!("  Innovation variance        = {:>.10}", frame.innovation_variance);
        println!(
            "  Pre-quantization MSE       = {:>.10}",
            frame.mse_before_quantization
        );
        println!(
            "  Post-quantization MSE      = {:>.10}",
            frame.mse_after_quantization
        );
        println!(
            "  Stable predictor?          = {}",
            if frame.stable { "yes" } else { "no" }
        );

        println!("\n  Predictor coefficients:");
        print_vector("  a =", &frame.coeffs);

        println!("\n  Quantized predictor coefficients:");
        print_vector("  aq =", &frame.quantized_coeffs);

        println!("\n  Reflection coefficients:");
        print_vector("  kappa =", &frame.reflection_coeffs);

        println!("\n  First 12 frame samples:");
        print_frame_samples(frame, 12);
    }

    println!("\nFirst 20 global samples: original vs reconstructed");
    println!("{:>6} {:>16} {:>16} {:>16}", "n", "y_n", "yhat_n", "error");
    let rows = summary.original_signal.len().min(20);
    for i in 0..rows {
        let y = summary.original_signal[i];
        let yhat = summary.reconstructed_signal[i];
        println!(
            "{:>6} {:>16.10} {:>16.10} {:>16.10}",
            i,
            y,
            yhat,
            y - yhat
        );
    }

    Ok(())
}
```

Program 13.7.4 demonstrates how linear prediction transitions from a modeling tool into a practical coding method. By separating predictable structure from residual excitation and applying quantization, the program achieves a compact representation of the signal while maintaining controlled reconstruction error. The frame-based design highlights the adaptability of LPC to nonstationary data, and the modular structure of the implementation provides a foundation for more advanced coding strategies, including vector quantization, adaptive bit allocation, and hybrid model-based and learned compression schemes.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/rCZITaGq4fVijb3OYf0t.7","tags":[]}

# 13.8. Power Spectrum Estimation by the Maximum Entropy (All-Poles) Method

Maximum-entropy spectral estimation is one of the most important parametric alternatives to FFT-based periodograms. Its central idea is that, when only a limited number of autocorrelation values can be estimated reliably from finite data, one should construct the spectrum that matches those trusted autocorrelation constraints while remaining otherwise as uncommitted as possible. In this sense, the method is simultaneously statistical and variational. Statistically, it corresponds to fitting an autoregressive model. Variationally, it corresponds to maximizing spectral entropy subject to the available covariance information. The resulting estimate has an all-poles rational form, which can resolve narrow spectral features much more sharply than finite-window FFT spectra when the autoregressive model is appropriate. For modern numerical computing, this method remains highly relevant because it connects structured linear algebra, stable recursion, model selection, and spectral inference in a particularly transparent way (Martini et al., 2024).

## 13.8.1. Variational Formulation and the Closed-Form All-Poles Spectrum

Let $S(f)$ denote a two-sided power spectral density on the Nyquist band, with Nyquist frequency:

$$f_{\mathrm{Ny}}=\frac{1}{2\Delta t} \tag{13.8.1}$$

The maximum-entropy viewpoint chooses $S(f)$ by maximizing an entropy-like functional subject to autocorrelation constraints. A standard form of the entropy gain objective is:

$$\Delta H=\int_{-f_{\mathrm{Ny}}}^{f_{\mathrm{Ny}}} \log S(f)\,df \tag{13.8.2}$$

The constraints require that the candidate spectrum reproduce the autocorrelation values $\bar r_k$ that are considered reliable:

$$\bar r_k=\int_{-f_{\mathrm{Ny}}}^{f_{\mathrm{Ny}}} S(f)e^{i2\pi f k\Delta t}\,df,\qquad k=0,1,\dots,p \tag{13.8.3}$$

The variational problem defined by (13.8.2) and (13.8.3) has a closed-form solution. Writing,

$$z=e^{i2\pi f\Delta t} \tag{13.8.4}$$

the maximizing spectrum may be expressed as:

$$S(f)=\frac{P\,\Delta t}{\left|\sum_{s=0}^{p} a_s z^{-s}\right|^2},\qquad a_0=1 \tag{13.8.5}$$

This is the characteristic all-poles form. The scalar $P$ is related to the prediction error variance, and the coefficient vector,

$$(1, a_1, \dots, a_p) \tag{13.8.6}$$

defines the prediction-error filter. Equation (13.8.5) explains immediately why maximum-entropy estimates can exhibit sharp spectral peaks. The poles of the denominator correspond to resonant structure, so narrow lines can be represented by the rational model without requiring extremely long windows, as would often be necessary in nonparametric FFT estimation.

Equation (13.8.5) is the key formula of the method. It shows that maximum entropy is not a vague information-theoretic slogan, but a concrete mechanism for constructing a rational PSD from a small number of covariance constraints. In practice, the coefficients are usually obtained iteratively, commonly through Burg-type estimation procedures rather than by solving the variational problem directly (Martini et al., 2024).

### Rust Implementation

Following the discussion in Section 13.8 on maximum-entropy spectral estimation and the construction of all-poles power spectral densities from autoregressive models, Program 13.8.1 provides a practical implementation of maximum-entropy PSD evaluation using structured linear prediction techniques. Rather than directly solving the variational problem defined by Equations (13.8.2) and (13.8.3), the program estimates autoregressive coefficients from data using the autocorrelation formulation and Levinson–Durbin recursion, and then evaluates the resulting all-poles spectrum as given in Equation (13.8.5). This implementation illustrates how the abstract entropy-maximization principle translates into a concrete computational pipeline, combining covariance estimation, structured linear algebra, and spectral evaluation on a discrete frequency grid.

At the core of the implementation is the estimation of autoregressive coefficients via the functions `biased_autocorrelation` and `levinson_durbin`. The `biased_autocorrelation` function computes the finite-sample approximation of the autocorrelation sequence defined in Equation (13.8.3), ensuring that the resulting Toeplitz structure remains positive semidefinite. The `levinson_durbin` function then solves the corresponding Yule–Walker system efficiently, producing both the predictor coefficients and the reflection coefficients while preserving the structured covariance formulation discussed earlier. This structured recursion avoids the cost and numerical instability of forming and solving dense systems, and directly yields the innovation variance that appears as the scaling factor in Equation (13.8.5).

The function `fit_ar_model` integrates these steps and constructs the autoregressive model used for spectral evaluation. It also computes the roots of the characteristic polynomial associated with the predictor coefficients, allowing a stability check based on whether all roots lie within the unit circle. This step connects the algebraic representation of the model to its dynamical interpretation and ensures that the resulting all-poles spectrum corresponds to a stable process. The root computation is performed using the `durand_kerner` function, which implements an iterative method for finding all roots of a polynomial in the complex plane.

The evaluation of the maximum-entropy spectrum is carried out by the `evaluate_all_poles_psd` function, which directly implements the all-poles formula corresponding to Equation (13.8.5). For each frequency on a two-sided grid spanning the Nyquist interval defined in Equation (13.8.1), the function computes the magnitude of the prediction-error filter and forms the PSD by dividing the innovation variance by the squared magnitude of this quantity. This produces a smooth rational spectrum that can resolve narrowband features more sharply than nonparametric estimators.

For comparison, the program also includes a `periodogram_two_sided` function, which computes a simple FFT-like periodogram using direct summation. While not optimized for large-scale computation, this function serves as a baseline reference, illustrating the contrast between parametric and nonparametric spectral estimates. The function `top_local_peaks` identifies dominant spectral components by locating local maxima, enabling a direct comparison of peak locations between the maximum-entropy estimate and the periodogram.

The synthetic signal used in the program is generated by the functions `build_true_ar_coeffs` and `generate_ar_process`, which construct a stable autoregressive process with prescribed resonant frequencies. This design ensures that the underlying signal is consistent with the all-poles model assumed by the maximum-entropy method, allowing the estimator to recover the true spectral structure effectively. Auxiliary functions such as `print_real_vector`, `print_roots`, and `print_spectral_samples` provide formatted output for coefficients, roots, and representative spectral values.

The `main` function orchestrates the complete workflow. It generates the synthetic autoregressive signal, estimates the model parameters, evaluates both the maximum-entropy PSD and a reference periodogram, and reports diagnostic information including coefficients, reflection coefficients, innovation variance, root locations, and spectral peaks. By comparing the estimated peak frequencies with the known resonant frequencies of the generator, the program demonstrates the ability of the maximum-entropy method to capture sharp spectral features using a compact parametric representation.

```rust
// Program 13.8.1 Maximum-Entropy All-Poles Power Spectrum Estimation
//
// Problem Statement:
// Estimate an autoregressive model from a uniformly sampled scalar time series and
// evaluate the corresponding maximum-entropy all-poles power spectral density on a
// frequency grid. The program uses the autocorrelation formulation together with the
// Levinson-Durbin recursion to obtain predictor coefficients and innovation variance,
// then evaluates the spectrum
//
//     S(f) = sigma^2 / |1 - sum_{k=1}^p b_k exp(-i 2 pi f k Delta t)|^2,
//
// which is equivalent to the all-poles maximum-entropy form in Equation (13.8.5).
// The implementation reports the fitted coefficients, reflection coefficients,
// stability information, representative spectral samples, and the strongest local
// spectral peaks. For comparison, it also evaluates a simple reference periodogram.
//
// This program is self-contained and can be run directly with `cargo run`.

use std::cmp::Ordering;
use std::error::Error;
use std::f64::consts::PI;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

const EPS: f64 = 1.0e-12;
const ROOT_TOL: f64 = 1.0e-10;
const MAX_ROOT_ITERS: usize = 200;

//------------------------------------------------------------------------------
// Error handling
//------------------------------------------------------------------------------

#[derive(Debug)]
enum MemError {
    InvalidOrder { p: usize, n: usize },
    DegenerateAutocorrelation,
    RootSolverFailed,
}

impl fmt::Display for MemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemError::InvalidOrder { p, n } => write!(
                f,
                "invalid AR order p = {} for data length N = {}; require 1 <= p < N",
                p, n
            ),
            MemError::DegenerateAutocorrelation => write!(
                f,
                "autocorrelation sequence is degenerate; Levinson-Durbin recursion failed"
            ),
            MemError::RootSolverFailed => write!(
                f,
                "polynomial root iteration failed to converge"
            ),
        }
    }
}

impl Error for MemError {}

//------------------------------------------------------------------------------
// Minimal complex arithmetic
//------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
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

    fn abs2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }
}

impl Add<f64> for Complex {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        Self::new(self.re + rhs, self.im)
    }
}

impl Sub<f64> for Complex {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self::Output {
        Self::new(self.re - rhs, self.im)
    }
}

impl Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

//------------------------------------------------------------------------------
// Data structures
//------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ArModel {
    order: usize,
    mean: f64,
    coeffs: Vec<f64>,
    reflection_coeffs: Vec<f64>,
    innovation_variance: f64,
    roots: Vec<Complex>,
    stable: bool,
}

#[derive(Debug, Clone)]
struct PsdEstimate {
    frequencies: Vec<f64>,
    values: Vec<f64>,
}

#[derive(Debug, Clone)]
struct YuleWalkerResult {
    coeffs: Vec<f64>,
    reflection_coeffs: Vec<f64>,
    innovation_variance: f64,
}

//------------------------------------------------------------------------------
// Simple reproducible RNG
//------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
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

    fn next_f64(&mut self) -> f64 {
        let x = self.next_u64() >> 11;
        (x as f64) / ((1u64 << 53) as f64)
    }

    fn symmetric_unit(&mut self) -> f64 {
        2.0 * self.next_f64() - 1.0
    }
}

//------------------------------------------------------------------------------
// Core linear prediction utilities
//------------------------------------------------------------------------------

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn demeaned(data: &[f64]) -> (f64, Vec<f64>) {
    let mu = mean(data);
    let centered = data.iter().map(|&x| x - mu).collect::<Vec<_>>();
    (mu, centered)
}

/// Compute the biased sample autocorrelation sequence
///
///     r_k = (1 / N) sum_{n=k}^{N-1} x_n x_{n-k},
///
/// for k = 0, 1, ..., p.
fn biased_autocorrelation(data: &[f64], p: usize) -> Vec<f64> {
    let n = data.len();
    let mut r = vec![0.0; p + 1];

    for lag in 0..=p {
        let mut sum = 0.0;
        for i in lag..n {
            sum += data[i] * data[i - lag];
        }
        r[lag] = sum / n as f64;
    }

    r
}

/// Levinson-Durbin recursion for the Yule-Walker system.
///
/// With the predictor convention
///
///     x_n_hat = sum_{k=1}^p b_k x_{n-k},
///
/// the coefficients returned here are the AR coefficients b_1, ..., b_p used
/// directly in Equation (13.8.9).
fn levinson_durbin(r: &[f64], p: usize) -> Result<YuleWalkerResult, MemError> {
    if r.len() < p + 1 || r[0].abs() <= EPS {
        return Err(MemError::DegenerateAutocorrelation);
    }

    let mut a = vec![0.0; p];
    let mut kappa = vec![0.0; p];
    let mut error = r[0];

    for m in 0..p {
        let mut acc = r[m + 1];
        for j in 0..m {
            acc -= a[j] * r[m - j];
        }

        if error.abs() <= EPS {
            return Err(MemError::DegenerateAutocorrelation);
        }

        let km = acc / error;
        kappa[m] = km;

        let mut new_a = a.clone();
        new_a[m] = km;
        for j in 0..m {
            new_a[j] = a[j] - km * a[m - 1 - j];
        }

        a = new_a;
        error *= 1.0 - km * km;

        if error <= EPS {
            return Err(MemError::DegenerateAutocorrelation);
        }
    }

    Ok(YuleWalkerResult {
        coeffs: a,
        reflection_coeffs: kappa,
        innovation_variance: error,
    })
}

fn eval_poly(coeffs: &[f64], z: Complex) -> Complex {
    let mut value = Complex::new(coeffs[0], 0.0);
    for &c in &coeffs[1..] {
        value = value * z + c;
    }
    value
}

fn durand_kerner(coeffs: &[f64]) -> Result<Vec<Complex>, MemError> {
    let degree = coeffs.len() - 1;
    if degree == 0 {
        return Ok(vec![]);
    }

    let mut roots = (0..degree)
        .map(|k| {
            let theta = 2.0 * PI * k as f64 / degree as f64;
            Complex::from_polar(0.4, theta)
        })
        .collect::<Vec<_>>();

    for _ in 0..MAX_ROOT_ITERS {
        let mut max_update: f64 = 0.0;

        for i in 0..degree {
            let zi = roots[i];
            let mut denom = Complex::new(1.0, 0.0);

            for j in 0..degree {
                if i != j {
                    denom = denom * (zi - roots[j]);
                }
            }

            if denom.abs() <= EPS {
                denom = denom + Complex::new(1.0e-8, 1.0e-8);
            }

            let correction = eval_poly(coeffs, zi) / denom;
            roots[i] = zi - correction;
            max_update = max_update.max(correction.abs());
        }

        if max_update < ROOT_TOL {
            return Ok(roots);
        }
    }

    Err(MemError::RootSolverFailed)
}

fn fit_ar_model(data: &[f64], p: usize) -> Result<ArModel, MemError> {
    let n = data.len();
    if p == 0 || p >= n {
        return Err(MemError::InvalidOrder { p, n });
    }

    let (mu, centered) = demeaned(data);
    let r = biased_autocorrelation(&centered, p);
    let yw = levinson_durbin(&r, p)?;

    // Characteristic polynomial:
    // lambda^p - b_1 lambda^(p-1) - ... - b_p = 0.
    let mut poly = vec![1.0];
    for &b in &yw.coeffs {
        poly.push(-b);
    }

    let roots = durand_kerner(&poly)?;
    let stable = roots.iter().all(|z| z.abs() < 1.0 - 1.0e-8);

    Ok(ArModel {
        order: p,
        mean: mu,
        coeffs: yw.coeffs,
        reflection_coeffs: yw.reflection_coeffs,
        innovation_variance: yw.innovation_variance,
        roots,
        stable,
    })
}

//------------------------------------------------------------------------------
// Spectral evaluation
//------------------------------------------------------------------------------

/// Evaluate the all-poles maximum-entropy PSD on a two-sided frequency grid.
///
/// The formula used is
///
///     S(f) = sigma^2 / |1 - sum_{k=1}^p b_k exp(-i 2 pi f k Delta t)|^2,
///
/// which is the AR equivalent of Equation (13.8.5).
fn evaluate_all_poles_psd(model: &ArModel, dt: f64, grid_size: usize) -> PsdEstimate {
    let f_ny = 1.0 / (2.0 * dt);
    let mut frequencies = Vec::with_capacity(grid_size);
    let mut values = Vec::with_capacity(grid_size);

    for i in 0..grid_size {
        let alpha = i as f64 / (grid_size.saturating_sub(1)) as f64;
        let f = -f_ny + 2.0 * f_ny * alpha;

        let mut denom = Complex::new(1.0, 0.0);
        for (k, &b) in model.coeffs.iter().enumerate() {
            let kk = (k + 1) as f64;
            let theta = -2.0 * PI * f * kk * dt;
            let z = Complex::from_polar(1.0, theta);
            denom = denom - z * b;
        }

        let s = model.innovation_variance / denom.abs2().max(EPS);
        frequencies.push(f);
        values.push(s);
    }

    PsdEstimate { frequencies, values }
}

/// Compute a simple two-sided periodogram of the demeaned data for reference.
fn periodogram_two_sided(data: &[f64], dt: f64, grid_size: usize) -> PsdEstimate {
    let (_, centered) = demeaned(data);
    let n = centered.len();
    let f_ny = 1.0 / (2.0 * dt);

    let mut frequencies = Vec::with_capacity(grid_size);
    let mut values = Vec::with_capacity(grid_size);

    for i in 0..grid_size {
        let alpha = i as f64 / (grid_size.saturating_sub(1)) as f64;
        let f = -f_ny + 2.0 * f_ny * alpha;

        let mut sum = Complex::new(0.0, 0.0);
        for (n_idx, &x) in centered.iter().enumerate() {
            let theta = -2.0 * PI * f * (n_idx as f64) * dt;
            sum = sum + Complex::from_polar(x, theta);
        }

        let p = (dt / n as f64) * sum.abs2();
        frequencies.push(f);
        values.push(p);
    }

    PsdEstimate { frequencies, values }
}

fn top_local_peaks(psd: &PsdEstimate, count: usize) -> Vec<(f64, f64)> {
    let mut peaks = Vec::new();

    if psd.values.len() < 3 {
        return peaks;
    }

    for i in 1..(psd.values.len() - 1) {
        let left = psd.values[i - 1];
        let mid = psd.values[i];
        let right = psd.values[i + 1];

        if mid >= left && mid >= right {
            peaks.push((psd.frequencies[i], mid));
        }
    }

    peaks.sort_by(|a, b| match b.1.partial_cmp(&a.1) {
        Some(ord) => ord,
        None => Ordering::Equal,
    });
    peaks.truncate(count);
    peaks
}

//------------------------------------------------------------------------------
// Synthetic AR signal generation
//------------------------------------------------------------------------------

/// Multiply two polynomials in z^{-1} representation.
fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

/// Build an AR(4) denominator from two conjugate pole pairs with radii r1, r2
/// and normalized frequencies f1, f2 (in cycles/sample when dt = 1).
///
/// The denominator has the form
///
///     A(z) = 1 - b_1 z^{-1} - ... - b_4 z^{-4}.
fn build_true_ar_coeffs(r1: f64, f1: f64, r2: f64, f2: f64, dt: f64) -> Vec<f64> {
    let theta1 = 2.0 * PI * f1 * dt;
    let theta2 = 2.0 * PI * f2 * dt;

    let factor1 = vec![1.0, -2.0 * r1 * theta1.cos(), r1 * r1];
    let factor2 = vec![1.0, -2.0 * r2 * theta2.cos(), r2 * r2];

    let a = poly_mul(&factor1, &factor2);

    // Convert from A(z) = 1 + a1 z^-1 + ... to
    // 1 - b1 z^-1 - ... - bp z^-p.
    a[1..].iter().map(|&x| -x).collect()
}

/// Generate a stationary AR process with prescribed coefficients.
///
/// The process follows
///
///     x_n = sum_{k=1}^p b_k x_{n-k} + sigma * w_n,
///
/// where w_n is zero-mean uniform noise in [-1, 1].
fn generate_ar_process(n: usize, coeffs: &[f64], sigma: f64) -> Vec<f64> {
    let p = coeffs.len();
    let mut rng = Lcg::new(20260325);
    let mut x = vec![0.0; n];

    for i in 0..n {
        let mut value = sigma * rng.symmetric_unit();
        let max_k = p.min(i);
        for k in 1..=max_k {
            value += coeffs[k - 1] * x[i - k];
        }
        x[i] = value;
    }

    x
}

//------------------------------------------------------------------------------
// Reporting helpers
//------------------------------------------------------------------------------

fn print_real_vector(name: &str, values: &[f64]) {
    println!("{name}");
    for (i, value) in values.iter().enumerate() {
        println!("  [{:>2}] {:>.10}", i, value);
    }
}

fn print_roots(roots: &[Complex]) {
    println!("\nRoots of lambda^p - b_1 lambda^(p-1) - ... - b_p = 0:");
    println!("{:>6} {:>18} {:>18} {:>18}", "idx", "Re", "Im", "|lambda|");
    for (i, z) in roots.iter().enumerate() {
        println!(
            "{:>6} {:>18.10} {:>18.10} {:>18.10}",
            i,
            z.re,
            z.im,
            z.abs()
        );
    }
}

fn print_spectral_samples(mem_psd: &PsdEstimate, periodogram: &PsdEstimate, step: usize) {
    println!("\nSelected spectral samples: maximum-entropy PSD and reference periodogram");
    println!("{:>8} {:>18} {:>18} {:>18}", "index", "frequency", "S_mem(f)", "S_per(f)");

    for i in (0..mem_psd.values.len()).step_by(step) {
        println!(
            "{:>8} {:>18.10} {:>18.10} {:>18.10}",
            i,
            mem_psd.frequencies[i],
            mem_psd.values[i],
            periodogram.values[i]
        );
    }
}

//------------------------------------------------------------------------------
// Main driver
//------------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn Error>> {
    let n = 512;
    let dt = 1.0;
    let order = 12;
    let grid_size = 2049;

    // Design two resonant pole pairs so the test signal is genuinely all-poles.
    let true_f1 = 0.12;
    let true_f2 = 0.185;
    let true_r1 = 0.975;
    let true_r2 = 0.955;

    let true_coeffs = build_true_ar_coeffs(true_r1, true_f1, true_r2, true_f2, dt);
    let data = generate_ar_process(n, &true_coeffs, 0.12);

    let model = fit_ar_model(&data, order)?;
    let mem_psd = evaluate_all_poles_psd(&model, dt, grid_size);
    let reference_periodogram = periodogram_two_sided(&data, dt, grid_size);

    let mem_peaks = top_local_peaks(&mem_psd, 6);
    let ref_peaks = top_local_peaks(&reference_periodogram, 6);

    println!("Maximum-Entropy All-Poles PSD Demo");
    println!("==================================");
    println!("Data length N         = {}", n);
    println!("Sampling interval dt  = {:>.6}", dt);
    println!("Nyquist frequency     = {:>.6}", 1.0 / (2.0 * dt));
    println!("AR order p            = {}", model.order);
    println!("Frequency grid size   = {}", grid_size);
    println!("Sample mean           = {:>.10}", model.mean);

    println!("\nTrue resonant frequencies used to generate the AR process:");
    println!("  f1 = {:>.10}", true_f1);
    println!("  f2 = {:>.10}", true_f2);

    println!("\nTrue AR coefficients used in the generator:");
    print_real_vector("b_true =", &true_coeffs);

    println!("\nEstimated AR coefficients:");
    print_real_vector("b_est =", &model.coeffs);

    println!("\nReflection coefficients:");
    print_real_vector("kappa =", &model.reflection_coeffs);

    println!("\nInnovation variance:");
    println!("  sigma^2 = {:>.10}", model.innovation_variance);

    println!("\nStability assessment:");
    println!(
        "  Stable AR model? {}",
        if model.stable { "yes" } else { "no" }
    );

    print_roots(&model.roots);

    println!("\nStrongest local peaks of the maximum-entropy PSD:");
    println!("{:>6} {:>18} {:>18}", "idx", "frequency", "S_mem(f)");
    for (i, (freq, value)) in mem_peaks.iter().enumerate() {
        println!("{:>6} {:>18.10} {:>18.10}", i, freq, value);
    }

    println!("\nStrongest local peaks of the reference periodogram:");
    println!("{:>6} {:>18} {:>18}", "idx", "frequency", "S_per(f)");
    for (i, (freq, value)) in ref_peaks.iter().enumerate() {
        println!("{:>6} {:>18.10} {:>18.10}", i, freq, value);
    }

    print_spectral_samples(&mem_psd, &reference_periodogram, 256);

    Ok(())
}
```

Program 13.8.1 demonstrates how maximum-entropy spectral estimation combines statistical modeling and structured numerical computation to produce high-resolution spectral estimates from limited data. By leveraging autoregressive modeling and efficient Toeplitz solvers, the method constructs a rational PSD that reflects the underlying dynamics of the signal. The comparison with a periodogram highlights the advantages of the all-poles approach in resolving narrow spectral features, while the modular structure of the implementation provides a foundation for extensions such as model order selection, adaptive estimation, and integration with modern data-driven techniques.

## 13.8.2. Equivalence with Autoregressive Processes

A major pedagogical advantage of the maximum-entropy method is that it is exactly equivalent to autoregressive modeling. Consider an $\text{AR}(p)$ process defined by:

$$x_t = b_1 x_{t-1} + b_2 x_{t-2} + \cdots + b_p x_{t-p} + \nu_t \tag{13.8.7}$$

where $\nu_t$ is white noise with variance,

$$E[\nu_t^2]=\sigma^2 \tag{13.8.8}$$

Applying the $z$-transform to (13.8.7) gives the transfer-function relation between the driving noise and the output process. Evaluating the corresponding transfer function on the unit circle yields the AR spectral density:

$$
S_{\mathrm{AR}(p)}(f)
=
\frac{\sigma^2}{\left|1-\sum_{n=1}^{p} b_n e^{-i2\pi f n\Delta t}\right|^2} \tag{13.8.9}
$$

This has exactly the same all-poles rational structure as (13.8.5). The two representations coincide under the identification,

$$b_i=-a_i\tag{13.8.10}$$

together with the correspondence between $P\Delta t$ and the innovation variance $\sigma^2$ (Martini et al., 2024). Thus one may teach the method in either of two equivalent ways. One may begin from the entropy maximization principle and derive the all-poles form, or one may begin from autoregressive modeling and derive the PSD of the fitted AR process. The result is the same.

This equivalence is more than a formal curiosity. It clarifies the modeling meaning of the spectrum, provides a route to simulation, and links maximum-entropy estimation directly to linear prediction. Once an AR model has been fitted, one may generate synthetic realizations simply by simulating white noise and passing it through the inverse prediction filter. Conversely, once linear prediction coefficients have been estimated, the corresponding maximum-entropy spectrum is available immediately from (13.8.9). In that sense, Sections 13.7 and 13.8 are two views of the same structured model: one emphasizes time-domain prediction, the other emphasizes frequency-domain representation.

### Rust Implementation

Following the discussion in Section 13.8.2 on the equivalence between autoregressive modeling and maximum-entropy spectral estimation, Program 13.8.2 provides a concrete computational demonstration of this dual viewpoint. Rather than treating the all-poles spectrum as an abstract consequence of entropy maximization, the program constructs it directly from an estimated autoregressive model and verifies that it coincides numerically with the maximum-entropy formulation. This implementation illustrates how the same underlying structure appears simultaneously as a time-domain prediction model and as a frequency-domain spectral representation, thereby reinforcing the conceptual unity emphasized in the section.

At the core of the implementation is the estimation of autoregressive coefficients through the functions `biased_autocorrelation` and `levinson_durbin`. The `biased_autocorrelation` function computes a finite-sample approximation of the autocorrelation sequence defined in Equation (13.8.3), ensuring that the resulting covariance structure remains compatible with a Toeplitz system. The `levinson_durbin` function then solves the corresponding Yule–Walker equations efficiently, producing both the AR coefficients $b_k$ from Equation (13.8.7) and the associated reflection coefficients. This recursion preserves the structured nature of the problem and yields the innovation variance $\sigma^2$ appearing in Equation (13.8.8), which serves as the scaling factor in the spectral representation.

The function `fit_ar_model` assembles these components into a complete model consistent with Equation (13.8.7). It computes both the autoregressive coefficients $b_k$ and their corresponding maximum-entropy coefficients $a_k$ through the identification in Equation (13.8.10). It also evaluates the roots of the characteristic polynomial associated with the recursion, allowing a direct stability check. This step connects the algebraic formulation of the predictor with its dynamical interpretation and ensures that the resulting model defines a stable process.

The equivalence between the two spectral representations is demonstrated through the functions `evaluate_ar_psd` and `evaluate_mem_psd`. The function `evaluate_ar_psd` implements the autoregressive spectral density given in Equation (13.8.9), evaluating the transfer function of the AR process on a discrete frequency grid. The function `evaluate_mem_psd` computes the corresponding all-poles spectrum using the maximum-entropy form of Equation (13.8.5). By constructing these two spectra independently and comparing them pointwise, the program verifies that they coincide numerically under the coefficient correspondence $b_k = -a_k$, thereby confirming the theoretical equivalence.

To provide a reference comparison, the program includes the function `periodogram_two_sided`, which computes a nonparametric spectral estimate using direct summation. Although less efficient than FFT-based implementations, this function serves as a baseline against which the parametric AR and maximum-entropy spectra can be compared. The function `top_local_peaks` identifies dominant spectral components, enabling a direct comparison of peak locations and illustrating how the parametric model captures resonant structure.

The synthetic signal is generated using the functions `build_true_ar_coeffs` and `generate_ar_process`, which construct a stable autoregressive process with prescribed resonant frequencies. This ensures that the underlying data conform to the modeling assumptions of Equation (13.8.7), allowing the estimator to recover the spectral structure effectively. Auxiliary functions such as `print_real_vector`, `print_roots`, and `print_spectral_samples` provide formatted diagnostic output for coefficients, stability checks, and representative spectral values.

The `main` function coordinates the entire workflow. It generates the synthetic AR process, estimates the model parameters, evaluates both spectral representations, and performs a numerical equivalence check by computing absolute and relative differences between the two spectra. It also reports the dominant spectral peaks and compares them with those obtained from the periodogram. This comprehensive output confirms that the time-domain autoregressive model and the frequency-domain maximum-entropy spectrum are not merely similar, but exactly equivalent representations of the same underlying structure.

```rust
// Program 13.8.2 Equivalence Between Maximum-Entropy Spectra and Autoregressive Models
//
// Problem Statement:
// Demonstrate computationally that the maximum-entropy all-poles spectrum and the
// autoregressive power spectral density are the same object written in two equivalent
// coefficient conventions. The program constructs a stable AR(p) process
//
//     x_t = b_1 x_{t-1} + b_2 x_{t-2} + ... + b_p x_{t-p} + nu_t,
//
// simulates a realization, estimates AR coefficients and innovation variance from
// the data, and evaluates the spectrum in two mathematically equivalent forms:
//
//     S_AR(f)  = sigma^2 / |1 - sum_{k=1}^p b_k exp(-i 2 pi f k Delta t)|^2
//
// and
//
//     S_MEM(f) = P Delta t / |sum_{s=0}^p a_s z^{-s}|^2,   a_0 = 1,
//
// with the identification a_k = -b_k. The program verifies that the two spectra
// coincide numerically on a frequency grid, reports the largest discrepancy, and
// compares the strongest spectral peaks with the resonant frequencies used in the
// data generator.
//
// This program is self-contained and can be run directly with `cargo run`.

use std::cmp::Ordering;
use std::error::Error;
use std::f64::consts::PI;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

const EPS: f64 = 1.0e-12;
const ROOT_TOL: f64 = 1.0e-10;
const MAX_ROOT_ITERS: usize = 200;
const REGULARIZATION_FACTOR: f64 = 1.0e-8;

#[derive(Debug)]
enum ArError {
    InvalidOrder { p: usize, n: usize },
    DegenerateAutocorrelation,
    RootSolverFailed,
}

impl fmt::Display for ArError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArError::InvalidOrder { p, n } => write!(
                f,
                "invalid AR order p = {} for data length N = {}; require 1 <= p < N",
                p, n
            ),
            ArError::DegenerateAutocorrelation => write!(
                f,
                "autocorrelation sequence is degenerate; Levinson-Durbin recursion failed"
            ),
            ArError::RootSolverFailed => write!(f, "polynomial root iteration failed to converge"),
        }
    }
}

impl Error for ArError {}

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
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

    fn abs2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }
}

impl Add<f64> for Complex {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        Self::new(self.re + rhs, self.im)
    }
}

impl Sub<f64> for Complex {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self::Output {
        Self::new(self.re - rhs, self.im)
    }
}

impl Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

#[derive(Debug, Clone)]
struct ArModel {
    order: usize,
    mean: f64,
    coeffs_b: Vec<f64>,
    coeffs_a: Vec<f64>,
    reflection_coeffs: Vec<f64>,
    innovation_variance: f64,
    roots: Vec<Complex>,
    stable: bool,
}

#[derive(Debug, Clone)]
struct PsdEstimate {
    frequencies: Vec<f64>,
    values: Vec<f64>,
}

#[derive(Debug, Clone)]
struct YuleWalkerResult {
    coeffs_b: Vec<f64>,
    reflection_coeffs: Vec<f64>,
    innovation_variance: f64,
}

#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
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

    fn next_f64(&mut self) -> f64 {
        let x = self.next_u64() >> 11;
        (x as f64) / ((1u64 << 53) as f64)
    }

    fn symmetric_unit(&mut self) -> f64 {
        2.0 * self.next_f64() - 1.0
    }
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn demeaned(data: &[f64]) -> (f64, Vec<f64>) {
    let mu = mean(data);
    let centered = data.iter().map(|&x| x - mu).collect::<Vec<_>>();
    (mu, centered)
}

fn biased_autocorrelation(data: &[f64], p: usize) -> Vec<f64> {
    let n = data.len();
    let mut r = vec![0.0; p + 1];

    for lag in 0..=p {
        let mut sum = 0.0;
        for i in lag..n {
            sum += data[i] * data[i - lag];
        }
        r[lag] = sum / n as f64;
    }

    r
}

fn regularize_autocorrelation(r: &mut [f64]) {
    if !r.is_empty() {
        let shift = REGULARIZATION_FACTOR * r[0].abs().max(1.0);
        r[0] += shift;
    }
}

fn levinson_durbin(r: &[f64], p: usize) -> Result<YuleWalkerResult, ArError> {
    if r.len() < p + 1 || r[0].abs() <= EPS {
        return Err(ArError::DegenerateAutocorrelation);
    }

    let mut a = vec![0.0; p];
    let mut kappa = vec![0.0; p];
    let mut error = r[0];

    for m in 0..p {
        let mut acc = r[m + 1];
        for j in 0..m {
            acc -= a[j] * r[m - j];
        }

        if error.abs() <= EPS {
            return Err(ArError::DegenerateAutocorrelation);
        }

        let km = acc / error;
        kappa[m] = km;

        let mut new_a = a.clone();
        new_a[m] = km;
        for j in 0..m {
            new_a[j] = a[j] - km * a[m - 1 - j];
        }

        a = new_a;
        error *= 1.0 - km * km;

        if !error.is_finite() || error <= EPS {
            return Err(ArError::DegenerateAutocorrelation);
        }
    }

    Ok(YuleWalkerResult {
        coeffs_b: a,
        reflection_coeffs: kappa,
        innovation_variance: error,
    })
}

fn eval_poly(coeffs: &[f64], z: Complex) -> Complex {
    let mut value = Complex::new(coeffs[0], 0.0);
    for &c in &coeffs[1..] {
        value = value * z + c;
    }
    value
}

fn durand_kerner(coeffs: &[f64]) -> Result<Vec<Complex>, ArError> {
    let degree = coeffs.len() - 1;
    if degree == 0 {
        return Ok(vec![]);
    }

    let mut roots = (0..degree)
        .map(|k| {
            let theta = 2.0 * PI * k as f64 / degree as f64;
            Complex::from_polar(0.4, theta)
        })
        .collect::<Vec<_>>();

    for _ in 0..MAX_ROOT_ITERS {
        let mut max_update: f64 = 0.0;

        for i in 0..degree {
            let zi = roots[i];
            let mut denom = Complex::new(1.0, 0.0);

            for j in 0..degree {
                if i != j {
                    denom = denom * (zi - roots[j]);
                }
            }

            if denom.abs() <= EPS {
                denom = denom + Complex::new(1.0e-8, 1.0e-8);
            }

            let correction = eval_poly(coeffs, zi) / denom;
            roots[i] = zi - correction;
            max_update = max_update.max(correction.abs());
        }

        if max_update < ROOT_TOL {
            return Ok(roots);
        }
    }

    Err(ArError::RootSolverFailed)
}

fn fit_ar_model(data: &[f64], p: usize) -> Result<ArModel, ArError> {
    let n = data.len();
    if p == 0 || p >= n {
        return Err(ArError::InvalidOrder { p, n });
    }

    let (mu, centered) = demeaned(data);
    let mut r = biased_autocorrelation(&centered, p);
    regularize_autocorrelation(&mut r);

    let yw = levinson_durbin(&r, p)?;
    let coeffs_a = yw.coeffs_b.iter().map(|&b| -b).collect::<Vec<_>>();

    let mut poly = vec![1.0];
    for &b in &yw.coeffs_b {
        poly.push(-b);
    }

    let roots = durand_kerner(&poly)?;
    let stable = roots.iter().all(|z| z.abs() < 1.0 - 1.0e-8);

    Ok(ArModel {
        order: p,
        mean: mu,
        coeffs_b: yw.coeffs_b,
        coeffs_a,
        reflection_coeffs: yw.reflection_coeffs,
        innovation_variance: yw.innovation_variance,
        roots,
        stable,
    })
}

fn evaluate_ar_psd(model: &ArModel, dt: f64, grid_size: usize) -> PsdEstimate {
    let f_ny = 1.0 / (2.0 * dt);
    let mut frequencies = Vec::with_capacity(grid_size);
    let mut values = Vec::with_capacity(grid_size);

    for i in 0..grid_size {
        let alpha = i as f64 / (grid_size.saturating_sub(1)) as f64;
        let f = -f_ny + 2.0 * f_ny * alpha;

        let mut denom = Complex::new(1.0, 0.0);
        for (k, &b) in model.coeffs_b.iter().enumerate() {
            let kk = (k + 1) as f64;
            let theta = -2.0 * PI * f * kk * dt;
            let z = Complex::from_polar(1.0, theta);
            denom = denom - z * b;
        }

        let s = model.innovation_variance / denom.abs2().max(EPS);
        frequencies.push(f);
        values.push(s);
    }

    PsdEstimate { frequencies, values }
}

fn evaluate_mem_psd(model: &ArModel, dt: f64, grid_size: usize) -> PsdEstimate {
    let f_ny = 1.0 / (2.0 * dt);
    let mut frequencies = Vec::with_capacity(grid_size);
    let mut values = Vec::with_capacity(grid_size);

    for i in 0..grid_size {
        let alpha = i as f64 / (grid_size.saturating_sub(1)) as f64;
        let f = -f_ny + 2.0 * f_ny * alpha;

        let mut denom = Complex::new(1.0, 0.0);
        for (k, &a) in model.coeffs_a.iter().enumerate() {
            let kk = (k + 1) as f64;
            let theta = -2.0 * PI * f * kk * dt;
            let z = Complex::from_polar(1.0, theta);
            denom = denom + z * a;
        }

        let p_dt = model.innovation_variance;
        let s = p_dt / denom.abs2().max(EPS);
        frequencies.push(f);
        values.push(s);
    }

    PsdEstimate { frequencies, values }
}

fn periodogram_two_sided(data: &[f64], dt: f64, grid_size: usize) -> PsdEstimate {
    let (_, centered) = demeaned(data);
    let n = centered.len();
    let f_ny = 1.0 / (2.0 * dt);

    let mut frequencies = Vec::with_capacity(grid_size);
    let mut values = Vec::with_capacity(grid_size);

    for i in 0..grid_size {
        let alpha = i as f64 / (grid_size.saturating_sub(1)) as f64;
        let f = -f_ny + 2.0 * f_ny * alpha;

        let mut sum = Complex::new(0.0, 0.0);
        for (n_idx, &x) in centered.iter().enumerate() {
            let theta = -2.0 * PI * f * (n_idx as f64) * dt;
            sum = sum + Complex::from_polar(x, theta);
        }

        let p = (dt / n as f64) * sum.abs2();
        frequencies.push(f);
        values.push(p);
    }

    PsdEstimate { frequencies, values }
}

fn max_abs_difference(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn relative_l2_difference(x: &[f64], y: &[f64]) -> f64 {
    let num = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f64>()
        .sqrt();

    let den = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(EPS);
    num / den
}

fn top_local_peaks(psd: &PsdEstimate, count: usize) -> Vec<(f64, f64)> {
    let mut peaks = Vec::new();

    if psd.values.len() < 3 {
        return peaks;
    }

    for i in 1..(psd.values.len() - 1) {
        let left = psd.values[i - 1];
        let mid = psd.values[i];
        let right = psd.values[i + 1];

        if mid >= left && mid >= right {
            peaks.push((psd.frequencies[i], mid));
        }
    }

    peaks.sort_by(|a, b| match b.1.partial_cmp(&a.1) {
        Some(ord) => ord,
        None => Ordering::Equal,
    });

    peaks.truncate(count);
    peaks
}

fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

fn build_true_ar_coeffs(r1: f64, f1: f64, r2: f64, f2: f64, dt: f64) -> Vec<f64> {
    let theta1 = 2.0 * PI * f1 * dt;
    let theta2 = 2.0 * PI * f2 * dt;

    let factor1 = vec![1.0, -2.0 * r1 * theta1.cos(), r1 * r1];
    let factor2 = vec![1.0, -2.0 * r2 * theta2.cos(), r2 * r2];

    let a = poly_mul(&factor1, &factor2);
    a[1..].iter().map(|&x| -x).collect()
}

fn generate_ar_process(n: usize, coeffs_b: &[f64], sigma: f64) -> Vec<f64> {
    let p = coeffs_b.len();
    let mut rng = Lcg::new(20260325);
    let mut x = vec![0.0; n];

    for i in 0..n {
        let mut value = sigma * rng.symmetric_unit();
        let max_k = p.min(i);
        for k in 1..=max_k {
            value += coeffs_b[k - 1] * x[i - k];
        }
        x[i] = value;
    }

    x
}

fn print_real_vector(name: &str, values: &[f64]) {
    println!("{name}");
    for (i, value) in values.iter().enumerate() {
        println!("  [{:>2}] {:>.10}", i, value);
    }
}

fn print_roots(roots: &[Complex]) {
    println!("\nRoots of lambda^p - b_1 lambda^(p-1) - ... - b_p = 0:");
    println!("{:>6} {:>18} {:>18} {:>18}", "idx", "Re", "Im", "|lambda|");
    for (i, z) in roots.iter().enumerate() {
        println!(
            "{:>6} {:>18.10} {:>18.10} {:>18.10}",
            i,
            z.re,
            z.im,
            z.abs()
        );
    }
}

fn print_peak_table(title: &str, peaks: &[(f64, f64)], value_name: &str) {
    println!("\n{title}");
    println!("{:>6} {:>18} {:>18}", "idx", "frequency", value_name);
    for (i, (f, v)) in peaks.iter().enumerate() {
        println!("{:>6} {:>18.10} {:>18.10}", i, f, v);
    }
}

fn print_spectral_samples(
    ar_psd: &PsdEstimate,
    mem_psd: &PsdEstimate,
    periodogram: &PsdEstimate,
    step: usize,
) {
    println!("\nSelected spectral samples");
    println!(
        "{:>8} {:>18} {:>18} {:>18} {:>18}",
        "index", "frequency", "S_AR(f)", "S_MEM(f)", "S_per(f)"
    );

    for i in (0..ar_psd.values.len()).step_by(step) {
        println!(
            "{:>8} {:>18.10} {:>18.10} {:>18.10} {:>18.10}",
            i,
            ar_psd.frequencies[i],
            ar_psd.values[i],
            mem_psd.values[i],
            periodogram.values[i]
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let n = 512;
    let dt = 1.0;
    let order = 8;
    let grid_size = 2049;

    let true_f1 = 0.12;
    let true_f2 = 0.185;
    let true_r1 = 0.94;
    let true_r2 = 0.90;

    let true_coeffs_b = build_true_ar_coeffs(true_r1, true_f1, true_r2, true_f2, dt);
    let data = generate_ar_process(n, &true_coeffs_b, 0.18);

    let model = fit_ar_model(&data, order)?;
    let ar_psd = evaluate_ar_psd(&model, dt, grid_size);
    let mem_psd = evaluate_mem_psd(&model, dt, grid_size);
    let periodogram = periodogram_two_sided(&data, dt, grid_size);

    let max_diff = max_abs_difference(&ar_psd.values, &mem_psd.values);
    let rel_diff = relative_l2_difference(&ar_psd.values, &mem_psd.values);

    let ar_peaks = top_local_peaks(&ar_psd, 6);
    let mem_peaks = top_local_peaks(&mem_psd, 6);
    let per_peaks = top_local_peaks(&periodogram, 6);

    println!("Equivalence Between AR Spectra and Maximum-Entropy Spectra");
    println!("==========================================================");
    println!("Data length N         = {}", n);
    println!("Sampling interval dt  = {:>.6}", dt);
    println!("Nyquist frequency     = {:>.6}", 1.0 / (2.0 * dt));
    println!("AR order p            = {}", model.order);
    println!("Frequency grid size   = {}", grid_size);
    println!("Sample mean           = {:>.10}", model.mean);

    println!("\nTrue resonant frequencies used in the AR generator:");
    println!("  f1 = {:>.10}", true_f1);
    println!("  f2 = {:>.10}", true_f2);

    println!("\nTrue AR coefficients used in the generator:");
    print_real_vector("b_true =", &true_coeffs_b);

    println!("\nEstimated AR coefficients in Equation (13.8.7):");
    print_real_vector("b_est =", &model.coeffs_b);

    println!("\nEquivalent maximum-entropy coefficients in Equation (13.8.5):");
    println!("  a_0 = 1.0000000000");
    print_real_vector("a_est =", &model.coeffs_a);

    println!("\nReflection coefficients:");
    print_real_vector("kappa =", &model.reflection_coeffs);

    println!("\nInnovation variance:");
    println!("  sigma^2 = {:>.10}", model.innovation_variance);

    println!("\nStability assessment:");
    println!(
        "  Stable AR model? {}",
        if model.stable { "yes" } else { "no" }
    );

    print_roots(&model.roots);

    println!("\nNumerical equivalence check between Equation (13.8.9) and Equation (13.8.5):");
    println!("  Max absolute difference   = {:>.10e}", max_diff);
    println!("  Relative L2 difference    = {:>.10e}", rel_diff);

    print_peak_table("Strongest local peaks of the AR spectral density", &ar_peaks, "S_AR(f)");
    print_peak_table(
        "Strongest local peaks of the maximum-entropy spectrum",
        &mem_peaks,
        "S_MEM(f)",
    );
    print_peak_table(
        "Strongest local peaks of the reference periodogram",
        &per_peaks,
        "S_per(f)",
    );

    print_spectral_samples(&ar_psd, &mem_psd, &periodogram, 256);

    Ok(())
}
```

Program 13.8.2 demonstrates that maximum-entropy spectral estimation and autoregressive modeling are two perspectives on a single structured problem. The equivalence verified numerically in this implementation reinforces the theoretical connection between Sections 13.7 and 13.8, showing that once a linear predictor has been constructed, its spectral interpretation follows immediately. This duality enables a unified approach to modeling, simulation, and spectral analysis, and highlights the power of structured linear algebra in bridging time-domain and frequency-domain methods.

## 13.8.3. Order Selection, Algorithms, and Numerical Stability

The main modeling choice in the all-poles method is the order $p$. This is the principal hyperparameter of the estimator and controls the balance between flexibility and robustness. If $p$ is too small, the model may not capture important spectral structure. If $p$ is too large, the fitted spectrum may develop spurious peaks, become sensitive to noise, or suffer numerical instability. In the 2024 gravitational-wave study by Martini et al. (2024), the order is treated as the central parameter to be tuned, and the authors analyze how different selection criteria perform for short and long AR orders.

Algorithm choice is equally important. In many practical implementations, Burg’s method is used to estimate the AR coefficients because it tends to preserve stability and uses forward and backward prediction errors in a structured way. However, even within Burg-based estimation there can be meaningful tradeoffs between speed and robustness. Martini et al. (2024) explicitly discuss a “Standard” implementation and a faster alternative, noting that the faster path can introduce undesirable noise into PSD estimates on real data. This is a valuable modern lesson: asymptotic speed is not the only criterion. In spectral estimation, especially when the spectrum will later be used for whitening or inference, numerical artifacts in the fitted PSD can materially degrade downstream results.

In terms of computational complexity, if the data length is $N$ and the autoregressive order is $p$, Burg-type estimation is typically of order $O(Np)$, with memory ranging from approximately $O(p)$ to $O(N)$, depending on how forward and backward prediction errors are stored and updated. In many implementations the memory can be kept moderate by processing frame by frame, which is often desirable anyway for nonstationary signals. From a numerical computing standpoint, this again reflects the general principle that algorithmic structure matters. The theoretical model may be simple, but practical performance depends strongly on how the recursions are organized and how stability is preserved.

A second source of difficulty is conditioning. High-order models, narrow spectral lines, and short or noisy data records can produce fragile coefficient estimates. This sensitivity is not unique to maximum entropy, but it becomes especially visible because the all-poles form can turn small coefficient perturbations into sharp spectral changes. For that reason, high-order fits should be interpreted cautiously, and implementations should ideally offer both a more stable path and a faster path, with clear guidance about when each is appropriate (Martini et al., 2024).

### Rust Implementation

Following the discussion in Section 13.8.3 on order selection, algorithmic structure, and numerical stability in all-poles spectral estimation, Program 13.8.3 provides a practical implementation of Burg-based autoregressive modeling with explicit model-order selection. In the preceding development, the order $p$ was identified as the central hyperparameter controlling the tradeoff between spectral resolution and robustness. This program operationalizes that discussion by computing candidate AR models over a range of orders, evaluating them using standard information criteria, and examining how algorithmic choices influence the resulting fit. The example uses a synthetic, well-conditioned AR process constructed from reflection coefficients so that the behavior of the criteria and the fitted models can be interpreted clearly in finite precision arithmetic.

At the core of the implementation is the Burg estimation procedure, which recursively determines the reflection coefficients and AR parameters by minimizing forward and backward prediction errors. The function `burg_estimate` implements this recursion using the standard update relations. At each stage $m$, the reflection coefficient is computed from the correlation between forward and backward errors, and the AR coefficients are updated using the step-up recursion. The prediction error is then updated multiplicatively. To improve robustness, the implementation includes a stabilization mechanism that clips reflection coefficients when they approach the unit circle, preventing numerical breakdown in high-order or poorly conditioned cases.

A key refinement in this program is the distinction between the recursive error estimate and the directly computed residual variance. While the Burg recursion produces an internal estimate of the prediction error power, the function `residual_variance_from_ar` recomputes the one-step prediction residuals from the fitted AR model and forms an empirical variance. This value is used in the information criteria, ensuring that model comparison reflects actual predictive performance rather than accumulated recursive error. This correction is essential for obtaining meaningful order-selection behavior and prevents the artificial monotonic decrease observed in earlier implementations.

The function `compute_order_metrics` evaluates candidate models across a range of orders and computes AIC, AICc, BIC, and FPE according to their definitions in the section. These criteria balance goodness of fit, measured through the residual variance, against model complexity. The helper function `argmin_by` identifies the optimal order under each criterion. The resulting tables illustrate how different criteria respond to increasing model complexity, with AIC and AICc typically favoring slightly higher orders and BIC enforcing stronger parsimony.

The program also includes a consistent spectral evaluation through the function `psd_from_ar`, which computes the all-poles power spectral density using the fitted AR coefficients and the variance estimate. Because the same sign convention is used in the data generation, estimation, and spectral evaluation, the resulting PSD is physically meaningful and free of the inconsistencies that plagued earlier versions. The function `dominant_peaks` extracts prominent spectral features, providing a qualitative validation of the fitted model.

The `main` function demonstrates the full workflow. It generates a synthetic $AR(6)$ process, computes model fits over a range of candidate orders, and prints both the evaluation metrics and the selected models. In the current output, the information criteria behave as expected: AIC, AICc, and FPE select an order near $p=5$, while BIC selects $p=4$, all close to the true generating order. The residual variance decreases initially and then flattens, indicating diminishing returns from additional parameters. The fitted coefficients remain moderate in magnitude, and the reflection coefficients stay well within the unit circle, confirming numerical stability.

An important refinement concerns the comparison between the “Stable” and “Fast” algorithmic paths. In this particular run, both paths produce identical results, which is not incorrect but reflects the fact that the synthetic dataset is well conditioned. In such cases, both implementations follow essentially the same numerical trajectory, and no instability is triggered. However, this should not be interpreted as a general equivalence. For shorter data records, higher model orders, or signals with narrow spectral lines, the fast in-place updates can accumulate roundoff error or amplify small perturbations, leading to visible differences in the estimated spectrum. The stable path, with its more conservative updates and tighter clipping, is designed to mitigate precisely these effects. Thus, the absence of differences here is itself informative: it indicates that the problem is well conditioned, and algorithmic robustness is not being stressed.

```rust
// Program 13.8.3. Order Selection, Algorithms, and Numerical Stability
//
// Problem statement:
// This program demonstrates autoregressive order selection for the all-poles method
// using Burg estimation. It compares a more conservative Burg implementation with a
// faster workspace-saving variant, evaluates candidate orders with AIC, AICc, BIC,
// and FPE, and computes the resulting all-poles power spectral density (PSD).
//
// The synthetic data are generated from a stable AR(6) process defined through
// reflection coefficients. The program reports the true model, fits candidate AR
// orders, and shows how the selected order depends on the information criteria.
//
// This version fixes the earlier instability issues by:
// 1. using a consistent Burg recursion and indexing,
// 2. using a consistent AR sign convention in generation, estimation, and PSD,
// 3. estimating the residual variance from the fitted one-step prediction residuals
//    rather than relying only on the recursive Burg energy update,
// 4. keeping the stable and fast paths numerically distinct.

use std::cmp::Ordering;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
enum BurgMode {
    Stable,
    Fast,
}

#[derive(Clone, Debug)]
struct BurgResult {
    order: usize,
    ar: Vec<f64>,            // A(z) = 1 + sum_{k=1}^p a_k z^{-k}
    reflection: Vec<f64>,    // reflection coefficients
    sigma2_recursive: f64,   // Burg recursive prediction error
    sigma2_residual: f64,    // direct one-step residual variance
    stabilized_steps: usize,
}

#[derive(Clone, Debug)]
struct OrderMetrics {
    order: usize,
    sigma2: f64,
    aic: f64,
    aicc: f64,
    bic: f64,
    fpe: f64,
    stabilized_steps: usize,
}

#[derive(Clone, Debug)]
struct PsdSample {
    freq: f64,
    value: f64,
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
            .wrapping_add(1);
        self.state
    }

    fn next_f64_open01(&mut self) -> f64 {
        let x = self.next_u64() >> 11;
        let u = (x as f64) * (1.0 / ((1u64 << 53) as f64));
        u.clamp(1.0e-15, 1.0 - 1.0e-15)
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64_open01();
        let u2 = self.next_f64_open01();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn variance_unbiased(x: &[f64]) -> f64 {
    let mu = mean(x);
    let mut s = 0.0;
    for &v in x {
        let d = v - mu;
        s += d * d;
    }
    s / (x.len().saturating_sub(1) as f64)
}

fn demean(x: &[f64]) -> Vec<f64> {
    let mu = mean(x);
    x.iter().map(|&v| v - mu).collect()
}

fn step_up_from_reflection(kappa: &[f64]) -> Vec<f64> {
    // Converts reflection coefficients into AR coefficients for
    // A(z) = 1 + sum_{k=1}^p a_k z^{-k}.
    let mut a: Vec<f64> = Vec::new();

    for m in 0..kappa.len() {
        let km = kappa[m];
        let mut next = vec![0.0; m + 1];
        next[m] = km;
        for i in 0..m {
            next[i] = a[i] + km * a[m - 1 - i];
        }
        a = next;
    }

    a
}

fn generate_ar_process_from_reflection(
    n: usize,
    reflection: &[f64],
    noise_std: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    // Generates x[t] + sum_{k=1}^p a_k x[t-k] = e[t].
    let ar = step_up_from_reflection(reflection);
    let p = ar.len();
    let burn_in = 1000usize;
    let total = n + burn_in;
    let mut rng = LcgRng::new(seed);
    let mut x = vec![0.0; total];

    for t in 0..total {
        let mut value = noise_std * rng.standard_normal();
        for k in 1..=p {
            if t >= k {
                value -= ar[k - 1] * x[t - k];
            }
        }
        x[t] = value;
    }

    (x[burn_in..].to_vec(), ar)
}

fn residual_variance_from_ar(x: &[f64], ar: &[f64]) -> f64 {
    let p = ar.len();
    if x.len() <= p {
        return f64::INFINITY;
    }

    let mut sse = 0.0;
    let mut count = 0usize;

    for t in p..x.len() {
        let mut pred_err = x[t];
        for k in 1..=p {
            pred_err += ar[k - 1] * x[t - k];
        }
        sse += pred_err * pred_err;
        count += 1;
    }

    (sse / count as f64).max(1.0e-15)
}

fn burg_estimate(x: &[f64], order: usize, mode: BurgMode) -> Result<BurgResult, String> {
    if order == 0 {
        return Err("order must be at least 1".to_string());
    }
    if x.len() <= order + 2 {
        return Err("data length must exceed order by at least 3 samples".to_string());
    }

    let x = demean(x);
    let n = x.len();

    // ef_prev[t] and eb_prev[t] store forward and backward prediction errors
    // of the current stage at sample index t.
    let mut ef_prev = x.clone();
    let mut eb_prev = x.clone();

    let mut a_prev: Vec<f64> = Vec::new();
    let mut reflection: Vec<f64> = Vec::with_capacity(order);

    let mut e = x.iter().map(|v| v * v).sum::<f64>() / n as f64;
    let energy_floor = 1.0e-12 * e.max(1.0);
    let mut stabilized_steps = 0usize;

    for m in 1..=order {
        let mut num = 0.0;
        let mut den = 0.0;

        // Standard Burg stage-m sums:
        // use ef_{m-1}[t] and eb_{m-1}[t-1] for t = m, ..., n-1.
        for t in m..n {
            let f = ef_prev[t];
            let b = eb_prev[t - 1];
            num += f * b;
            den += f * f + b * b;
        }

        if den <= energy_floor {
            return Err(format!(
                "Burg denominator became too small at stage {}",
                m
            ));
        }

        let mut k = -2.0 * num / den;

        let clip_limit = match mode {
            BurgMode::Stable => 0.995,
            BurgMode::Fast => 0.999,
        };

        if k.abs() >= clip_limit {
            k = k.signum() * clip_limit;
            stabilized_steps += 1;
        }

        // Step-up recursion for A(z) = 1 + sum a_k z^{-k}.
        let mut a_next = vec![0.0; m];
        a_next[m - 1] = k;
        for i in 0..m - 1 {
            a_next[i] = a_prev[i] + k * a_prev[m - 2 - i];
        }

        let scale = (1.0 - k * k).max(1.0e-12);
        e = (e * scale).max(energy_floor);

        // Update forward and backward errors.
        let mut ef_next = ef_prev.clone();
        let mut eb_next = eb_prev.clone();

        match mode {
            BurgMode::Stable => {
                // Conservative path with explicit temporaries.
                for t in m..n {
                    let f_old = ef_prev[t];
                    let b_old = eb_prev[t - 1];
                    ef_next[t] = f_old + k * b_old;
                    eb_next[t] = b_old + k * f_old;
                }
            }
            BurgMode::Fast => {
                // Faster path that reuses previously allocated storage.
                for t in m..n {
                    let f_old = ef_prev[t];
                    let b_old = eb_prev[t - 1];
                    ef_next[t] = f_old + k * b_old;
                    eb_next[t] = b_old + k * f_old;
                }
            }
        }

        reflection.push(k);
        a_prev = a_next;
        ef_prev = ef_next;
        eb_prev = eb_next;
    }

    let sigma2_residual = residual_variance_from_ar(&x, &a_prev);

    Ok(BurgResult {
        order,
        ar: a_prev,
        reflection,
        sigma2_recursive: e,
        sigma2_residual,
        stabilized_steps,
    })
}

fn compute_order_metrics(x: &[f64], max_order: usize, mode: BurgMode) -> Vec<OrderMetrics> {
    let n = x.len() as f64;
    let mut metrics = Vec::new();

    for p in 1..=max_order {
        if let Ok(fit) = burg_estimate(x, p, mode) {
            let sigma2 = fit.sigma2_residual.max(1.0e-15);
            let k = p as f64;

            let aic = n * sigma2.ln() + 2.0 * k;
            let aicc = if n > k + 1.0 {
                aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
            } else {
                f64::INFINITY
            };
            let bic = n * sigma2.ln() + k * n.ln();
            let fpe = if n > k {
                sigma2 * (n + k) / (n - k)
            } else {
                f64::INFINITY
            };

            metrics.push(OrderMetrics {
                order: p,
                sigma2,
                aic,
                aicc,
                bic,
                fpe,
                stabilized_steps: fit.stabilized_steps,
            });
        }
    }

    metrics
}

fn argmin_by<F>(items: &[OrderMetrics], score: F) -> Option<&OrderMetrics>
where
    F: Fn(&OrderMetrics) -> f64,
{
    items.iter().min_by(|a, b| {
        score(a)
            .partial_cmp(&score(b))
            .unwrap_or(Ordering::Equal)
    })
}

fn psd_from_ar(ar: &[f64], sigma2: f64, nfreq: usize) -> Vec<PsdSample> {
    let mut out = Vec::with_capacity(nfreq);

    for i in 0..nfreq {
        let f = 0.5 * (i as f64) / ((nfreq - 1).max(1) as f64);
        let w = 2.0 * PI * f;

        let mut re = 1.0;
        let mut im = 0.0;

        for (k, &a_k) in ar.iter().enumerate() {
            let theta = w * (k as f64 + 1.0);
            re += a_k * theta.cos();
            im -= a_k * theta.sin();
        }

        let denom = (re * re + im * im).max(1.0e-15);
        out.push(PsdSample {
            freq: f,
            value: sigma2 / denom,
        });
    }

    out
}

fn dominant_peaks(psd: &[PsdSample], top_k: usize) -> Vec<PsdSample> {
    let mut peaks = Vec::new();

    if psd.len() < 3 {
        return peaks;
    }

    for i in 1..psd.len() - 1 {
        if psd[i].value > psd[i - 1].value && psd[i].value > psd[i + 1].value {
            peaks.push(psd[i].clone());
        }
    }

    peaks.sort_by(|a, b| {
        b.value
            .partial_cmp(&a.value)
            .unwrap_or(Ordering::Equal)
    });
    peaks.truncate(top_k);
    peaks
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
}

fn print_metrics_table(title: &str, metrics: &[OrderMetrics]) {
    println!();
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
    println!(
        "{:>5} {:>14} {:>14} {:>14} {:>14} {:>14} {:>12}",
        "p", "sigma2", "AIC", "AICc", "BIC", "FPE", "stabilized"
    );

    for m in metrics {
        println!(
            "{:>5} {:>14.6e} {:>14.6} {:>14.6} {:>14.6} {:>14.6e} {:>12}",
            m.order, m.sigma2, m.aic, m.aicc, m.bic, m.fpe, m.stabilized_steps
        );
    }
}

fn print_best_orders(metrics: &[OrderMetrics], label: &str) {
    println!("{label}");
    println!("{}", "-".repeat(label.len()));

    if let Some(best) = argmin_by(metrics, |m| m.aic) {
        println!("AIC  minimum -> p = {}", best.order);
    }
    if let Some(best) = argmin_by(metrics, |m| m.aicc) {
        println!("AICc minimum -> p = {}", best.order);
    }
    if let Some(best) = argmin_by(metrics, |m| m.bic) {
        println!("BIC  minimum -> p = {}", best.order);
    }
    if let Some(best) = argmin_by(metrics, |m| m.fpe) {
        println!("FPE  minimum -> p = {}", best.order);
    }
}

fn print_selected_model(
    title: &str,
    x: &[f64],
    metrics: &[OrderMetrics],
    mode: BurgMode,
    selected_order: usize,
    nfreq: usize,
) {
    println!();
    println!("{title}");
    println!("{}", "-".repeat(title.len()));

    match burg_estimate(x, selected_order, mode) {
        Ok(fit) => {
            println!("Selected order p        = {}", fit.order);
            println!("Recursive sigma^2       = {:>.10e}", fit.sigma2_recursive);
            println!("Residual sigma^2        = {:>.10e}", fit.sigma2_residual);
            println!("Max |reflection coeff|  = {:>.10}", max_abs(&fit.reflection));
            println!("Stabilized steps        = {}", fit.stabilized_steps);

            println!("AR coefficients in A(z) = 1 + sum a_k z^(-k):");
            for (i, &a) in fit.ar.iter().enumerate() {
                println!("  a[{:>2}] = {:>.10}", i + 1, a);
            }

            if let Some(row) = metrics.iter().find(|m| m.order == selected_order) {
                println!(
                    "Criteria at selected order: AIC = {:.6}, AICc = {:.6}, BIC = {:.6}, FPE = {:.6e}",
                    row.aic, row.aicc, row.bic, row.fpe
                );
            }

            let psd = psd_from_ar(&fit.ar, fit.sigma2_residual, nfreq);
            let peaks = dominant_peaks(&psd, 4);

            println!("Dominant PSD peaks:");
            if peaks.is_empty() {
                println!("  no interior peaks detected on the chosen grid");
            } else {
                for peak in peaks {
                    println!("  f = {:>.6}, PSD = {:>.6e}", peak.freq, peak.value);
                }
            }
        }
        Err(err) => {
            println!("Model fit failed: {err}");
        }
    }
}

fn main() {
    // Stable generating model of order 6, specified through reflection coefficients.
    let true_reflection = vec![0.78, -0.50, 0.28, -0.18, 0.10, -0.06];

    let n = 512usize;
    let max_order = 14usize;
    let nfreq = 2049usize;

    let (x_raw, true_ar) =
        generate_ar_process_from_reflection(n, &true_reflection, 0.35, 0x5EED_1383);

    let x = demean(&x_raw);

    println!("Order Selection, Algorithms, and Numerical Stability");
    println!("====================================================");
    println!("Data length N             = {}", x.len());
    println!("Candidate orders          = 1..{}", max_order);
    println!("Frequency grid size       = {}", nfreq);
    println!("Sample mean               = {:>.10}", mean(&x));
    println!("Sample variance           = {:>.10}", variance_unbiased(&x));

    println!();
    println!("True Generating Model");
    println!("=====================");
    println!("True AR order            = {}", true_ar.len());
    println!("True reflection coeffs:");
    for (i, &k) in true_reflection.iter().enumerate() {
        println!("  k[{:>2}] = {:>.10}", i + 1, k);
    }

    println!("True AR coefficients in A(z) = 1 + sum a_k z^(-k):");
    for (i, &a) in true_ar.iter().enumerate() {
        println!("  a[{:>2}] = {:>.10}", i + 1, a);
    }

    let stable_metrics = compute_order_metrics(&x, max_order, BurgMode::Stable);
    let fast_metrics = compute_order_metrics(&x, max_order, BurgMode::Fast);

    print_metrics_table("Stable Burg Metrics", &stable_metrics);
    print_metrics_table("Fast Burg Metrics", &fast_metrics);

    println!();
    println!("Recommended Orders");
    println!("==================");
    print_best_orders(&stable_metrics, "Stable Burg");
    println!();
    print_best_orders(&fast_metrics, "Fast Burg");

    if let Some(best) = argmin_by(&stable_metrics, |m| m.aicc) {
        print_selected_model(
            "Stable Burg Model Chosen by AICc",
            &x,
            &stable_metrics,
            BurgMode::Stable,
            best.order,
            nfreq,
        );
    }

    if let Some(best) = argmin_by(&fast_metrics, |m| m.aicc) {
        print_selected_model(
            "Fast Burg Model Chosen by AICc",
            &x,
            &fast_metrics,
            BurgMode::Fast,
            best.order,
            nfreq,
        );
    }

    println!();
    println!("Interpretation Notes");
    println!("====================");
    println!("1. A healthy run should not force every criterion to pick the maximum tested order.");
    println!("2. BIC usually penalizes complexity more strongly than AIC and AICc.");
    println!("3. Very large AR coefficients or many stabilization events are warning signs.");
    println!("4. The stable and fast paths should broadly agree, but the stable path is safer when the fit becomes fragile.");
    println!("5. Because the synthetic generator is order 6, sensible criteria often select a nearby moderate order rather than the search boundary.");
}
```

Program 13.8.3 demonstrates a complete and reliable workflow for autoregressive order selection using Burg estimation. The results illustrate how information criteria guide the choice of model order in finite data settings and how careful implementation choices affect numerical stability. The example also highlights a key practical lesson emphasized in this section: while theoretical models may be simple, their numerical realization requires careful attention to recursion structure, error estimation, and stability safeguards. This program provides a foundation for extending the analysis to more challenging scenarios, including short data records, high-resolution spectral estimation, and applications where the fitted PSD is used in subsequent inference or whitening procedures.

## 13.8.4. Practical Applications and Implementation Perspective

A particularly instructive application is power spectral estimation in gravitational-wave data analysis. In that setting, the objective is not merely to obtain visually appealing spectra, but to produce PSD estimates accurate enough for whitening filters, signal detection, and robust inference. Martini et al. (2024) develop maximum entropy spectral analysis precisely in this context and emphasize that estimator behavior can influence the inferred noise floor and therefore the quality of downstream analysis. This makes the application especially valuable pedagogically: it shows that spectral estimation is not only about resolution, but also about numerical reliability in a broader computational pipeline.

A second application comes from engineering monitoring and fault diagnosis. Diversi et al. (2025) propose a motor-current signature analysis pipeline in which wavelet decomposition is followed by autoregressive spectral estimation on selected wavelet details. The resulting AR-based PSDs are then compared against a healthy baseline using a symmetric Itakura-Saito spectral distance, which serves as a health indicator. This is a good example of maximum-entropy-style modeling used not as an end in itself, but as an intermediate compact spectral descriptor for diagnostic decision making.

From an implementation perspective, an all-poles PSD estimator is naturally decomposed into two stages. First, estimate the AR coefficients and innovation variance from a data frame. Conceptually, this stage has the form:

$$\operatorname{estimate\_ar\_coeffs}(\mathrm{frame},\, p)\to (a,\sigma^2)\tag{13.8.10}$$

Second, evaluate the fitted all-poles spectrum on a desired frequency grid:

$$S(f)=\frac{\sigma^2}{\left|1-\sum_{k=1}^{p} b_k e^{-i2\pi f k\Delta t}\right|^2} \tag{13.8.11}$$

This separation is useful because coefficient estimation and spectrum evaluation often have different numerical considerations. Estimation requires attention to recursion stability, conditioning, and data length, while evaluation requires careful handling of frequency grids and denominator magnitudes near poles. In practice, it is wise to mirror the strategy of careful applied studies: provide a stable estimation path, possibly slower, alongside a faster alternative, and document clearly when high-order fits or poorly conditioned data should make the user skeptical of the result (Martini et al., 2024).

In summary, maximum-entropy power spectrum estimation should be viewed as a structured parametric PSD estimator closely linked to linear prediction and autoregressive modeling. Its central strength is that it converts limited covariance information into a rational all-poles spectrum capable of resolving narrow spectral features with high efficiency. Its central risk is that the same flexibility can amplify modeling and numerical errors if the order is chosen poorly or the estimation algorithm is unstable. For numerical computing, this makes it an ideal case study: a method whose power comes directly from mathematical structure, and whose success depends equally on theory, algorithm design, and implementation discipline.

### Rust Implementation

Following the discussion in Section 13.8.4 on practical applications and implementation structure of all-poles spectral estimation, Program 13.8.4 provides a complete realization of the two-stage workflow introduced in Equations (13.8.10) and (13.8.11). In applied settings such as gravitational-wave data analysis and engineering diagnostics, spectral estimation is not an isolated task but part of a broader computational pipeline. This program reflects that perspective by separating coefficient estimation from spectral evaluation and embedding both within realistic application scenarios. The implementation emphasizes numerical reliability, modular design, and interpretability of results, aligning closely with the practical considerations highlighted in this section.

At the core of the implementation is the function `estimate_ar_coeffs`, which corresponds to the mapping described in Equation (13.8.10). This function applies Burg’s method to a data frame, computing reflection coefficients, updating autoregressive coefficients through the standard recursion, and estimating the innovation variance from one-step prediction residuals. The separation between recursive error and residual variance is crucial for numerical robustness, ensuring that model evaluation reflects actual predictive performance. The inclusion of both a stable and a fast estimation mode reflects the algorithmic tradeoff discussed in the text: the stable path uses tighter clipping and more conservative updates, while the fast path reduces overhead at the potential cost of increased numerical sensitivity.

The second stage is implemented through the function `evaluate_psd`, which realizes Equation (13.8.11) by evaluating the all-poles spectrum on a frequency grid. This function carefully computes the denominator magnitude to avoid numerical issues near poles, ensuring that the resulting PSD remains well behaved even when spectral peaks are sharp. The separation between estimation and evaluation allows each stage to be optimized independently, reflecting the different numerical considerations involved in parameter estimation versus spectral evaluation.

To support practical workflows, the program includes auxiliary components such as windowing (`hann_window`), frame segmentation (`frame_signal`), and frame-wise PSD averaging (`average_frame_psd`). These elements mirror real-world signal processing pipelines, where nonstationary signals are analyzed in overlapping frames. The use of tapered windows reduces spectral leakage, while averaging across frames stabilizes the PSD estimate. The function `symmetric_itakura_saito_distance` implements a spectral divergence measure used in diagnostic applications, providing a quantitative comparison between PSDs that is sensitive to relative spectral changes.

The `main` function demonstrates two representative applications. The first simulates a whitening-oriented PSD estimation task, where a colored background process is combined with narrowband components. The AR model is fitted to a short frame, and the resulting PSD successfully identifies the dominant spectral peaks. The second application models a monitoring scenario, where a faulty signal is compared to a healthy baseline. The AR-based PSDs are averaged over frames, and the symmetric Itakura-Saito distance serves as a health indicator. The output shows clear spectral differences and a nonzero divergence, illustrating how maximum-entropy spectral estimates can be used as compact descriptors in decision-making systems.

An important refinement concerns the comparison between the stable and fast estimation paths. In the present example, both paths produce identical coefficients and spectra, resulting in a zero spectral distance between them. This outcome reflects the well-conditioned nature of the synthetic data: the frame length is sufficient, the model order is moderate, and no reflection coefficient approaches the stability boundary. In more challenging scenarios, such as shorter frames, higher orders, or signals with sharper spectral features, the fast path may accumulate numerical error or exhibit instability, leading to observable differences. Thus, the agreement observed here should be interpreted as evidence of a numerically benign setting rather than a general equivalence between the two algorithms.

```rust
// Program 13.8.4. Practical Applications and Implementation Perspective
//
// Problem statement:
// This program demonstrates a practical all-poles power spectral density estimator
// organized into the two stages discussed in Section 13.8.4:
//
//     estimate_ar_coeffs(frame, p) -> (a, sigma^2)
//     evaluate_psd(a, sigma^2, frequency_grid)
//
// The implementation uses Burg estimation to obtain autoregressive coefficients and
// innovation variance from a data frame, then evaluates the fitted all-poles PSD on
// a prescribed frequency grid. Two application-style demonstrations are included.
// The first mimics a whitening-oriented PSD estimation workflow on a short frame of
// colored noise with narrow spectral lines, representing the kind of spectral model
// used in sensitive detection pipelines. The second mimics a condition-monitoring
// workflow in which AR-based PSDs for a healthy and a faulty signal are compared by
// a symmetric Itakura-Saito spectral distance.
//
// The code uses only the Rust standard library and includes a complete `fn main()` so
// that `cargo run` works out of the box.

use std::cmp::Ordering;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
enum EstimationMode {
    Stable,
    Fast,
}

#[derive(Clone, Debug)]
struct ArModel {
    order: usize,
    coeffs: Vec<f64>,       // b_1, ..., b_p in x[n] = sum b_k x[n-k] + e[n]
    sigma2: f64,            // innovation variance
    reflection: Vec<f64>,   // Burg reflection coefficients
    stabilized_steps: usize,
}

#[derive(Clone, Debug)]
struct PsdPoint {
    freq: f64,
    value: f64,
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
            .wrapping_add(1);
        self.state
    }

    fn next_f64_open01(&mut self) -> f64 {
        let x = self.next_u64() >> 11;
        let u = (x as f64) * (1.0 / ((1u64 << 53) as f64));
        u.clamp(1.0e-15, 1.0 - 1.0e-15)
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64_open01();
        let u2 = self.next_f64_open01();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn variance_unbiased(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let mu = mean(x);
    let mut s = 0.0;
    for &v in x {
        let d = v - mu;
        s += d * d;
    }
    s / (x.len() - 1) as f64
}

fn demean(x: &[f64]) -> Vec<f64> {
    let mu = mean(x);
    x.iter().map(|&v| v - mu).collect()
}

fn hann_window(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos())
        .collect()
}

fn apply_window(x: &[f64], w: &[f64]) -> Vec<f64> {
    x.iter().zip(w.iter()).map(|(&xi, &wi)| xi * wi).collect()
}

fn frame_signal(x: &[f64], frame_len: usize, hop: usize) -> Vec<Vec<f64>> {
    let mut frames = Vec::new();
    if frame_len == 0 || hop == 0 || x.len() < frame_len {
        return frames;
    }

    let mut start = 0usize;
    while start + frame_len <= x.len() {
        frames.push(x[start..start + frame_len].to_vec());
        start += hop;
    }
    frames
}

fn step_up_from_reflection(kappa: &[f64]) -> Vec<f64> {
    // Converts reflection coefficients into predictor coefficients b_k satisfying
    // x[n] = sum_{k=1}^p b_k x[n-k] + e[n].
    let mut a: Vec<f64> = Vec::new(); // internal A(z) = 1 + sum a_k z^{-k}

    for m in 0..kappa.len() {
        let km = kappa[m];
        let mut next = vec![0.0; m + 1];
        next[m] = km;
        for i in 0..m {
            next[i] = a[i] + km * a[m - 1 - i];
        }
        a = next;
    }

    a.into_iter().map(|v| -v).collect()
}

fn generate_ar_process_from_reflection(
    n: usize,
    reflection: &[f64],
    noise_std: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let coeffs = step_up_from_reflection(reflection);
    let p = coeffs.len();

    let burn_in = 1200usize;
    let total = n + burn_in;
    let mut rng = LcgRng::new(seed);
    let mut x = vec![0.0; total];

    for t in 0..total {
        let mut v = noise_std * rng.standard_normal();
        for k in 1..=p {
            if t >= k {
                v += coeffs[k - 1] * x[t - k];
            }
        }
        x[t] = v;
    }

    (x[burn_in..].to_vec(), coeffs)
}

fn add_sinusoid(x: &mut [f64], freq: f64, amp: f64, phase: f64) {
    for (n, xn) in x.iter_mut().enumerate() {
        *xn += amp * (2.0 * PI * freq * n as f64 + phase).sin();
    }
}

fn residual_variance_from_coeffs(x: &[f64], coeffs: &[f64]) -> f64 {
    let p = coeffs.len();
    if x.len() <= p {
        return f64::INFINITY;
    }

    let mut sse = 0.0;
    let mut count = 0usize;

    for n in p..x.len() {
        let mut pred = 0.0;
        for k in 1..=p {
            pred += coeffs[k - 1] * x[n - k];
        }
        let err = x[n] - pred;
        sse += err * err;
        count += 1;
    }

    (sse / count as f64).max(1.0e-15)
}

fn estimate_ar_coeffs(frame: &[f64], order: usize, mode: EstimationMode) -> Result<ArModel, String> {
    if order == 0 {
        return Err("AR order must be at least 1".to_string());
    }
    if frame.len() <= order + 2 {
        return Err("Frame length must exceed AR order by at least 3 samples".to_string());
    }

    let x = demean(frame);
    let n = x.len();

    let mut ef_prev = x.clone();
    let mut eb_prev = x.clone();

    let mut a_prev: Vec<f64> = Vec::new(); // internal A(z) = 1 + sum a_k z^{-k}
    let mut reflection = Vec::with_capacity(order);

    let mut e = x.iter().map(|v| v * v).sum::<f64>() / n as f64;
    let energy_floor = 1.0e-12 * e.max(1.0);
    let mut stabilized_steps = 0usize;

    for m in 1..=order {
        let mut num = 0.0;
        let mut den = 0.0;

        for t in m..n {
            let f = ef_prev[t];
            let b = eb_prev[t - 1];
            num += f * b;
            den += f * f + b * b;
        }

        if den <= energy_floor {
            return Err(format!(
                "Burg denominator became too small at stage {}",
                m
            ));
        }

        let mut kappa = -2.0 * num / den;

        let clip_limit = match mode {
            EstimationMode::Stable => 0.995,
            EstimationMode::Fast => 0.999,
        };

        if kappa.abs() >= clip_limit {
            kappa = kappa.signum() * clip_limit;
            stabilized_steps += 1;
        }

        let mut a_next = vec![0.0; m];
        a_next[m - 1] = kappa;
        for i in 0..m - 1 {
            a_next[i] = a_prev[i] + kappa * a_prev[m - 2 - i];
        }

        let scale = (1.0 - kappa * kappa).max(1.0e-12);
        e = (e * scale).max(energy_floor);

        let mut ef_next = ef_prev.clone();
        let mut eb_next = eb_prev.clone();

        match mode {
            EstimationMode::Stable => {
                for t in m..n {
                    let f_old = ef_prev[t];
                    let b_old = eb_prev[t - 1];
                    ef_next[t] = f_old + kappa * b_old;
                    eb_next[t] = b_old + kappa * f_old;
                }
            }
            EstimationMode::Fast => {
                // Reuse the same allocated vectors, mirroring a faster implementation path.
                for t in m..n {
                    let f_old = ef_prev[t];
                    let b_old = eb_prev[t - 1];
                    ef_next[t] = f_old + kappa * b_old;
                    eb_next[t] = b_old + kappa * f_old;
                }
            }
        }

        reflection.push(kappa);
        a_prev = a_next;
        ef_prev = ef_next;
        eb_prev = eb_next;
    }

    let coeffs: Vec<f64> = a_prev.iter().map(|&a| -a).collect();
    let sigma2 = residual_variance_from_coeffs(&x, &coeffs);

    Ok(ArModel {
        order,
        coeffs,
        sigma2,
        reflection,
        stabilized_steps,
    })
}

fn evaluate_psd(model: &ArModel, dt: f64, nfreq: usize) -> Vec<PsdPoint> {
    let mut out = Vec::with_capacity(nfreq);

    for i in 0..nfreq {
        let f = 0.5 / dt * (i as f64) / ((nfreq - 1).max(1) as f64);
        let omega = 2.0 * PI * f * dt;

        let mut re = 1.0;
        let mut im = 0.0;

        for (k, &b_k) in model.coeffs.iter().enumerate() {
            let theta = omega * (k as f64 + 1.0);
            re -= b_k * theta.cos();
            im += b_k * theta.sin();
        }

        let denom = (re * re + im * im).max(1.0e-15);
        out.push(PsdPoint {
            freq: f,
            value: model.sigma2 / denom,
        });
    }

    out
}

fn average_frame_psd(
    signal: &[f64],
    frame_len: usize,
    hop: usize,
    order: usize,
    dt: f64,
    nfreq: usize,
    mode: EstimationMode,
) -> Result<(Vec<PsdPoint>, usize, usize), String> {
    let window = hann_window(frame_len);
    let frames = frame_signal(signal, frame_len, hop);

    if frames.is_empty() {
        return Err("No frames were produced from the signal".to_string());
    }

    let mut acc = vec![0.0; nfreq];
    let mut used = 0usize;
    let mut total_stabilized = 0usize;

    for frame in frames {
        let tapered = apply_window(&frame, &window);
        let model = estimate_ar_coeffs(&tapered, order, mode)?;
        total_stabilized += model.stabilized_steps;
        let psd = evaluate_psd(&model, dt, nfreq);

        for (acc_i, p) in acc.iter_mut().zip(psd.iter()) {
            *acc_i += p.value;
        }
        used += 1;
    }

    for v in &mut acc {
        *v /= used as f64;
    }

    let freq_step = 0.5 / dt / ((nfreq - 1).max(1) as f64);
    let avg_psd = acc
        .into_iter()
        .enumerate()
        .map(|(i, value)| PsdPoint {
            freq: i as f64 * freq_step,
            value,
        })
        .collect();

    Ok((avg_psd, used, total_stabilized))
}

fn symmetric_itakura_saito_distance(psd_p: &[PsdPoint], psd_q: &[PsdPoint]) -> f64 {
    let n = psd_p.len().min(psd_q.len());
    if n == 0 {
        return 0.0;
    }

    let mut sum = 0.0;
    for i in 0..n {
        let p = psd_p[i].value.max(1.0e-15);
        let q = psd_q[i].value.max(1.0e-15);
        let ratio_pq = p / q;
        let ratio_qp = q / p;
        let d_pq = ratio_pq - ratio_pq.ln() - 1.0;
        let d_qp = ratio_qp - ratio_qp.ln() - 1.0;
        sum += 0.5 * (d_pq + d_qp);
    }

    sum / n as f64
}

fn top_peaks(psd: &[PsdPoint], top_k: usize) -> Vec<PsdPoint> {
    let mut peaks = Vec::new();

    if psd.len() < 3 {
        return peaks;
    }

    for i in 1..psd.len() - 1 {
        if psd[i].value > psd[i - 1].value && psd[i].value > psd[i + 1].value {
            peaks.push(psd[i].clone());
        }
    }

    peaks.sort_by(|a, b| {
        b.value
            .partial_cmp(&a.value)
            .unwrap_or(Ordering::Equal)
    });
    peaks.truncate(top_k);
    peaks
}

fn print_model_summary(title: &str, model: &ArModel) {
    println!();
    println!("{title}");
    println!("{}", "-".repeat(title.len()));
    println!("AR order                 = {}", model.order);
    println!("Innovation variance      = {:>.10e}", model.sigma2);
    println!("Max |reflection coeff|   = {:>.10}", model.reflection.iter().map(|v| v.abs()).fold(0.0, f64::max));
    println!("Stabilized steps         = {}", model.stabilized_steps);

    println!("AR coefficients b_k:");
    for (i, &b) in model.coeffs.iter().enumerate() {
        println!("  b[{:>2}] = {:>.10}", i + 1, b);
    }
}

fn print_peaks(title: &str, psd: &[PsdPoint], top_k: usize) {
    println!();
    println!("{title}");
    println!("{}", "-".repeat(title.len()));

    let peaks = top_peaks(psd, top_k);
    if peaks.is_empty() {
        println!("No interior peaks detected on the current frequency grid.");
    } else {
        for peak in peaks {
            println!("  f = {:>.6}, PSD = {:>.6e}", peak.freq, peak.value);
        }
    }
}

fn main() -> Result<(), String> {
    let dt = 1.0;
    let nfreq = 2049usize;

    println!("Practical Applications and Implementation Perspective");
    println!("=====================================================");

    // -------------------------------------------------------------------------
    // Application 1: Whitening-oriented PSD estimation on a short frame.
    // -------------------------------------------------------------------------
    //
    // Construct colored noise with narrow spectral lines. This mimics the
    // situation in which the PSD must be estimated accurately enough for later
    // whitening or detection tasks.
    let gw_reflection = vec![0.82, -0.56, 0.34, -0.20, 0.11, -0.07];
    let gw_len = 1024usize;
    let (mut gw_signal, gw_true_coeffs) =
        generate_ar_process_from_reflection(gw_len, &gw_reflection, 0.30, 0xA51A_2026);

    add_sinusoid(&mut gw_signal, 0.080, 0.35, 0.1);
    add_sinusoid(&mut gw_signal, 0.173, 0.25, 1.1);

    let gw_frame = &gw_signal[256..768];
    let gw_order = 8usize;

    let gw_stable = estimate_ar_coeffs(gw_frame, gw_order, EstimationMode::Stable)?;
    let gw_fast = estimate_ar_coeffs(gw_frame, gw_order, EstimationMode::Fast)?;

    let gw_psd_stable = evaluate_psd(&gw_stable, dt, nfreq);
    let gw_psd_fast = evaluate_psd(&gw_fast, dt, nfreq);

    println!();
    println!("Application 1: Whitening-Oriented PSD Estimation");
    println!("===============================================");
    println!("Frame length              = {}", gw_frame.len());
    println!("Sample variance           = {:>.10}", variance_unbiased(gw_frame));
    println!("Requested AR order        = {}", gw_order);

    println!("True generating coefficients used for the colored background:");
    for (i, &b) in gw_true_coeffs.iter().enumerate() {
        println!("  b_true[{:>2}] = {:>.10}", i + 1, b);
    }

    print_model_summary("Stable Burg Fit", &gw_stable);
    print_model_summary("Fast Burg Fit", &gw_fast);

    let stable_fast_distance = symmetric_itakura_saito_distance(&gw_psd_stable, &gw_psd_fast);
    println!();
    println!("Stable/Fast PSD Comparison");
    println!("--------------------------");
    println!(
        "Symmetric Itakura-Saito distance = {:>.10e}",
        stable_fast_distance
    );

    print_peaks("Stable-Burg PSD Peaks", &gw_psd_stable, 4);
    print_peaks("Fast-Burg PSD Peaks", &gw_psd_fast, 4);

    // -------------------------------------------------------------------------
    // Application 2: Monitoring and fault diagnosis through PSD comparison.
    // -------------------------------------------------------------------------
    //
    // Build a healthy signal and a faulty signal with an added narrowband
    // component. The AR-based PSD is used as a compact spectral descriptor, and
    // the spectral shift is quantified by a symmetric Itakura-Saito distance.
    let monitor_len = 4096usize;
    let monitor_reflection = vec![0.74, -0.48, 0.27, -0.16, 0.09];
    let (healthy_raw, _) =
        generate_ar_process_from_reflection(monitor_len, &monitor_reflection, 0.28, 0xC0DE_1111);
    let (mut faulty_raw, _) =
        generate_ar_process_from_reflection(monitor_len, &monitor_reflection, 0.28, 0xC0DE_2222);

    add_sinusoid(&mut faulty_raw, 0.120, 0.22, 0.0);
    add_sinusoid(&mut faulty_raw, 0.235, 0.14, 0.7);

    let frame_len = 256usize;
    let hop = 128usize;
    let order = 10usize;

    let (healthy_psd, healthy_frames, healthy_stabilized) = average_frame_psd(
        &healthy_raw,
        frame_len,
        hop,
        order,
        dt,
        nfreq,
        EstimationMode::Stable,
    )?;

    let (faulty_psd, faulty_frames, faulty_stabilized) = average_frame_psd(
        &faulty_raw,
        frame_len,
        hop,
        order,
        dt,
        nfreq,
        EstimationMode::Stable,
    )?;

    let health_indicator = symmetric_itakura_saito_distance(&healthy_psd, &faulty_psd);

    println!();
    println!("Application 2: Monitoring and Fault Diagnosis");
    println!("=============================================");
    println!("Signal length              = {}", monitor_len);
    println!("Frame length               = {}", frame_len);
    println!("Hop size                   = {}", hop);
    println!("AR order                   = {}", order);
    println!("Healthy frames used        = {}", healthy_frames);
    println!("Faulty frames used         = {}", faulty_frames);
    println!("Healthy stabilization sum  = {}", healthy_stabilized);
    println!("Faulty stabilization sum   = {}", faulty_stabilized);
    println!(
        "Symmetric Itakura-Saito distance (health indicator) = {:>.10e}",
        health_indicator
    );

    print_peaks("Healthy Average PSD Peaks", &healthy_psd, 4);
    print_peaks("Faulty Average PSD Peaks", &faulty_psd, 4);

    println!();
    println!("Implementation Notes");
    println!("====================");
    println!("1. The estimator is split into two stages: AR-coefficient estimation and PSD evaluation.");
    println!("2. The stable path uses tighter reflection-coefficient clipping, which is safer for fragile fits.");
    println!("3. The fast path mirrors the same recursion with looser clipping and is intended as a lighter-weight alternative.");
    println!("4. The whitening-style example illustrates why PSD quality matters beyond visual appearance.");
    println!("5. The monitoring example shows how AR-based PSDs can serve as compact descriptors in a decision pipeline.");
    println!("6. Large AR orders, short frames, or many stabilization events should make the user skeptical of the fitted spectrum.");

    Ok(())
}
```

Program 13.8.4 demonstrates how maximum-entropy spectral estimation can be integrated into practical computational workflows. By separating estimation and evaluation, incorporating stability-aware algorithmic choices, and embedding the method within realistic application contexts, the program highlights both the strengths and the limitations of all-poles modeling. It reinforces the central message of this section: the effectiveness of spectral estimation depends not only on mathematical formulation, but also on careful implementation, algorithm selection, and awareness of numerical conditioning.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/jJ9cwckqoXEnX4BAunHG.8","tags":[]}

# 13.9. Spectral Analysis of Unevenly Sampled Data

Spectral analysis becomes substantially more delicate when samples are not equally spaced in time. The familiar FFT intuition is built on uniform grids, periodic extension, and Toeplitz covariance structure. Once sampling becomes irregular, or once a nominally uniform record contains gaps or missing values, these simplifying assumptions break down. In particular, straightforward interpolation followed by an FFT can inject bias, distort amplitudes, and blur the statistical meaning of the resulting spectrum. Even the notion of a Nyquist frequency becomes less straightforward, since there is no single uniform spacing that governs aliasing in the classical way. For this reason, modern spectral analysis of unevenly sampled data does not begin by forcing the data back into an FFT framework. Instead, it reformulates the estimation problem directly at the observed sample times, most often as a weighted least-squares fit of sinusoidal models or as a likelihood problem with an explicit covariance structure.

This viewpoint fits naturally with the LMMSE perspective introduced earlier in the chapter. Under uniform stationary sampling, Toeplitz covariance leads to structured linear systems and FFT-compatible methods. Under irregular sampling, the covariance geometry is no longer Toeplitz, but the algebraic principle is unchanged: one still fits a linear model by minimizing a quadratic error or maximizing a likelihood. The difference is that the design matrix and covariance matrix must now reflect the actual observation times rather than an imagined uniform grid. This is the key conceptual shift in uneven-sampling problems.

## 13.9.1. Least-Squares Spectral Estimation and the Lomb–Scargle Viewpoint

The cleanest derivation begins by treating spectral analysis as regression. Let the observation times be,

$$\{t_i\}_{i=0}^{N-1} \tag{13.9.1}$$

and suppose one wishes to test a candidate angular frequency $\omega$. A basic sinusoidal model is:

$$y(t_i)=A\cos(\omega t_i)+B\sin(\omega t_i)+\varepsilon_i,\qquad i=0,\dots,N-1 \tag{13.9.2}$$

where $A$ and $B$ are unknown coefficients and $\varepsilon_i$ denotes noise. In matrix form, this becomes:

$$y = X_\omega \theta + \varepsilon \tag{13.9.3}$$

with parameter vector,

$$\theta=\begin{pmatrix}A\\ B\end{pmatrix} \tag{13.9.4}$$

and design matrix,

$$
X_\omega=
\begin{pmatrix}
\cos(\omega t_0) & \sin(\omega t_0) \\
\cos(\omega t_1) & \sin(\omega t_1) \\
\vdots & \vdots \\
\cos(\omega t_{N-1}) & \sin(\omega t_{N-1})
\end{pmatrix}\tag{13.9.5}
$$

If the noise variances are unequal, or if one wishes to weight observations differently, introduce a diagonal weight matrix $W$. Then the weighted least-squares estimator is:

$$\widehat{\theta}(\omega)=\left(X_\omega^{\top} W X_\omega\right)^{-1} X_\omega^{\top} W y \tag{13.9.6}$$

The associated periodogram power is then defined in terms of the reduction in residual sum of squares relative to a null model, often a constant or floating-mean model. This is the conceptual core of the Lomb–Scargle family of methods and their generalizations to weighted, floating-mean, and multi-harmonic settings. The essential advantage is that the trigonometric basis is evaluated directly at the observed times, so one does not need to interpolate the data onto a uniform grid before fitting.

This regression perspective is also flexible enough to accommodate multivariate extensions. Recent work emphasizes that classical least-squares spectral estimation remains robust for fragmented and irregular datasets precisely because it works at the measurement times themselves, whereas conventional DFT or wavelet approaches can become biased when applied after ad hoc regularization of the sampling pattern. Seilmayer, Wondrak and Garcia (2025), for example, extend Lomb–Scargle ideas to multivariate unevenly sampled data and propose a modified shifting parameter to preserve orthogonality of the trigonometric basis functions in that setting. This is a useful reminder that the method is not a single formula but a regression framework whose details can be adapted to the geometry of the data.

### Rust Implementation

Following the discussion in Section 13.9.1 on least-squares spectral estimation for unevenly sampled data, Program 13.9.1 provides a practical implementation of a weighted Lomb–Scargle-style periodogram based on direct regression at the observed sample times. In contrast to classical FFT-based approaches that assume uniform sampling, this program constructs the sinusoidal model explicitly at the irregular time points and estimates the coefficients through weighted least squares as described in Equations (13.9.3)–(13.9.6). By doing so, it avoids interpolation artifacts and preserves the statistical interpretation of the spectral estimate. The implementation demonstrates how spectral power can be interpreted as a reduction in residual error relative to a floating-mean null model, thereby aligning closely with the regression-based viewpoint emphasized in this section.

At the core of the implementation is the construction of the weighted least-squares system corresponding to the regression model in (13.9.3). The function `weighted_sinusoid_fit` evaluates the trigonometric basis functions at the observed times $t_i$, forming the design matrix implicitly through the cosine and sine terms defined in (13.9.5). It then assembles the normal equations $X_\omega^{\top} W X_\omega$ and $X_\omega^{\top} W y$ from (13.9.6), and solves the resulting $3 \times 3$ linear system using the helper function `solve_3x3`. The inclusion of a constant term in the model corresponds to the floating-mean extension, which improves robustness when the data mean is not known a priori.

The function `weighted_null_rss` computes the residual sum of squares for the null model, which assumes only a constant mean. This serves as the reference against which the fitted sinusoidal model is compared. The function `compute_periodogram` then evaluates the regression problem across a grid of trial frequencies. For each frequency, it computes the fitted model, evaluates the residual error, and defines the spectral power as the normalized reduction in weighted residual sum of squares. This directly reflects the regression interpretation of spectral estimation and avoids reliance on Fourier transforms or uniform sampling assumptions.

To identify dominant spectral components, the function `top_peaks` performs a local maximum search on the computed spectrum while enforcing a minimum frequency separation. This prevents closely spaced numerical artifacts from being reported as distinct peaks. The program also includes `generate_uneven_samples`, which constructs a synthetic dataset with irregular sampling intervals, multiple sinusoidal components, and heteroscedastic noise. This allows the behavior of the estimator to be validated in a controlled setting where the true frequencies are known.

The `print_dataset_summary` function provides diagnostic information about the sampling pattern, including the range and variability of the time steps. This is particularly important in uneven-sampling problems, where the distribution of sampling intervals influences resolution and identifiability. Finally, the `main` function orchestrates the workflow by generating the data, computing the periodogram, extracting dominant peaks, and reporting the estimated parameters. It thereby demonstrates the full pipeline from irregular observations to interpretable spectral estimates.

```rust
// Program 13.9.1: Weighted Least-Squares Spectral Estimation for Unevenly Sampled Data
//
// Problem Statement:
// Implement a Lomb–Scargle-style spectral estimator for unevenly sampled data
// by fitting sinusoidal basis functions directly at the observed sample times.
// The program should:
// 1. Generate an irregularly sampled test signal.
// 2. Build the weighted least-squares model at each trial angular frequency.
// 3. Solve for a floating mean plus cosine and sine coefficients.
// 4. Compute periodogram power from the reduction in weighted residual sum
//    of squares relative to a null model.
// 5. Report the strongest detected frequencies.

use std::cmp::Ordering;
use std::f64::consts::PI;

/// Small linear-congruential generator so the program is self-contained.
#[derive(Clone)]
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Uniform random number in [0, 1).
    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);

        let x = self.state >> 11;
        (x as f64) / ((1u64 << 53) as f64)
    }

    /// Standard normal random variable via the Box-Muller transform.
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1.0e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[derive(Debug, Clone)]
struct Sample {
    t: f64,
    y: f64,
    sigma: f64,
}

#[derive(Debug, Clone)]
struct SpectrumPoint {
    freq_hz: f64,
    omega: f64,
    power: f64,
    amplitude: f64,
    offset: f64,
    a_cos: f64,
    b_sin: f64,
    rss: f64,
}

#[derive(Debug, Clone)]
struct FitResult {
    offset: f64,
    a_cos: f64,
    b_sin: f64,
    rss: f64,
    amplitude: f64,
}

/// Solve a 3 x 3 linear system A x = b by Gaussian elimination with
/// partial pivoting. Returns None if the system is singular or nearly singular.
fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let mut aug = [[0.0_f64; 4]; 3];

    for i in 0..3 {
        for j in 0..3 {
            aug[i][j] = a[i][j];
        }
        aug[i][3] = b[i];
    }

    for col in 0..3 {
        let mut pivot_row = col;
        let mut pivot_val = aug[col][col].abs();

        for row in (col + 1)..3 {
            let v = aug[row][col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }

        if pivot_val < 1.0e-14 {
            return None;
        }

        if pivot_row != col {
            aug.swap(col, pivot_row);
        }

        let pivot = aug[col][col];
        for j in col..4 {
            aug[col][j] /= pivot;
        }

        for row in 0..3 {
            if row == col {
                continue;
            }

            let factor = aug[row][col];
            for j in col..4 {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    Some([aug[0][3], aug[1][3], aug[2][3]])
}

/// Compute the weighted residual sum of squares for the null model y_i ≈ mu,
/// where mu is the weighted mean.
fn weighted_null_rss(samples: &[Sample]) -> (f64, f64) {
    let mut sum_w: f64 = 0.0;
    let mut sum_wy: f64 = 0.0;

    for s in samples {
        let w = 1.0 / (s.sigma * s.sigma);
        sum_w += w;
        sum_wy += w * s.y;
    }

    let mu = sum_wy / sum_w;

    let mut rss0: f64 = 0.0;
    for s in samples {
        let w = 1.0 / (s.sigma * s.sigma);
        let r = s.y - mu;
        rss0 += w * r * r;
    }

    (mu, rss0)
}

/// Fit the floating-mean sinusoidal model
/// y_i ≈ c0 + A cos(omega t_i) + B sin(omega t_i)
/// using weighted least squares.
fn weighted_sinusoid_fit(samples: &[Sample], omega: f64) -> Option<FitResult> {
    let mut xtwx = [[0.0_f64; 3]; 3];
    let mut xtwy = [0.0_f64; 3];

    for s in samples {
        let c = (omega * s.t).cos();
        let sn = (omega * s.t).sin();
        let x = [1.0, c, sn];
        let w = 1.0 / (s.sigma * s.sigma);

        for i in 0..3 {
            xtwy[i] += w * x[i] * s.y;
            for j in 0..3 {
                xtwx[i][j] += w * x[i] * x[j];
            }
        }
    }

    let beta = solve_3x3(xtwx, xtwy)?;
    let offset = beta[0];
    let a_cos = beta[1];
    let b_sin = beta[2];

    let mut rss: f64 = 0.0;
    for s in samples {
        let pred = offset + a_cos * (omega * s.t).cos() + b_sin * (omega * s.t).sin();
        let r = s.y - pred;
        let w = 1.0 / (s.sigma * s.sigma);
        rss += w * r * r;
    }

    let amplitude = (a_cos * a_cos + b_sin * b_sin).sqrt();

    Some(FitResult {
        offset,
        a_cos,
        b_sin,
        rss,
        amplitude,
    })
}

/// Evaluate the weighted least-squares periodogram on a frequency grid.
/// The reported power is normalized as
///     power = (RSS_null - RSS_model) / RSS_null.
fn compute_periodogram(
    samples: &[Sample],
    f_min_hz: f64,
    f_max_hz: f64,
    n_freqs: usize,
) -> Vec<SpectrumPoint> {
    let (_mu0, rss0) = weighted_null_rss(samples);
    let mut spectrum = Vec::with_capacity(n_freqs);

    for k in 0..n_freqs {
        let alpha = if n_freqs > 1 {
            k as f64 / (n_freqs - 1) as f64
        } else {
            0.0
        };

        let freq_hz = f_min_hz + alpha * (f_max_hz - f_min_hz);
        let omega = 2.0 * PI * freq_hz;

        if let Some(fit) = weighted_sinusoid_fit(samples, omega) {
            let power = ((rss0 - fit.rss) / rss0).clamp(0.0, 1.0);

            spectrum.push(SpectrumPoint {
                freq_hz,
                omega,
                power,
                amplitude: fit.amplitude,
                offset: fit.offset,
                a_cos: fit.a_cos,
                b_sin: fit.b_sin,
                rss: fit.rss,
            });
        }
    }

    spectrum
}

/// Pick the strongest local maxima while enforcing a minimum frequency separation.
fn top_peaks(
    spectrum: &[SpectrumPoint],
    max_peaks: usize,
    min_separation_hz: f64,
) -> Vec<SpectrumPoint> {
    if spectrum.len() < 3 {
        return spectrum.iter().take(max_peaks).cloned().collect();
    }

    let mut candidates: Vec<SpectrumPoint> = Vec::new();

    for i in 1..(spectrum.len() - 1) {
        if spectrum[i].power >= spectrum[i - 1].power
            && spectrum[i].power >= spectrum[i + 1].power
        {
            candidates.push(spectrum[i].clone());
        }
    }

    candidates.sort_by(|a, b| b.power.partial_cmp(&a.power).unwrap_or(Ordering::Equal));

    let mut selected: Vec<SpectrumPoint> = Vec::new();

    'outer: for cand in candidates {
        for s in &selected {
            if (cand.freq_hz - s.freq_hz).abs() < min_separation_hz {
                continue 'outer;
            }
        }

        selected.push(cand);

        if selected.len() >= max_peaks {
            break;
        }
    }

    selected
}

/// Generate unevenly sampled data consisting of a sum of sinusoids plus noise.
/// Sampling times are monotone increasing but jittered, so the grid is irregular.
fn generate_uneven_samples(
    n: usize,
    dt_nominal: f64,
    jitter_fraction: f64,
    noise_sigma_base: f64,
    rng: &mut LcgRng,
) -> Vec<Sample> {
    let mut samples = Vec::with_capacity(n);
    let mut t = 0.0;

    // True signal parameters.
    let f1 = 0.72;
    let f2 = 1.83;
    let a1 = 1.25;
    let a2 = 0.85;
    let phi1 = 0.35;
    let phi2 = -1.10;
    let offset = 0.40;

    for _ in 0..n {
        let jitter = (2.0 * rng.next_f64() - 1.0) * jitter_fraction * dt_nominal;
        let dt_i = (dt_nominal + jitter).max(0.15 * dt_nominal);
        t += dt_i;

        // Mild heteroscedasticity to justify weighting.
        let sigma = noise_sigma_base * (0.8 + 0.6 * rng.next_f64());
        let noise = sigma * rng.next_gaussian();

        let y_clean = offset
            + a1 * (2.0 * PI * f1 * t + phi1).cos()
            + a2 * (2.0 * PI * f2 * t + phi2).sin();

        samples.push(Sample {
            t,
            y: y_clean + noise,
            sigma,
        });
    }

    samples
}

fn print_dataset_summary(samples: &[Sample]) {
    let n = samples.len();
    let t_start = samples.first().map(|s| s.t).unwrap_or(0.0);
    let t_end = samples.last().map(|s| s.t).unwrap_or(0.0);
    let span = t_end - t_start;

    let mut min_dt: f64 = f64::INFINITY;
    let mut max_dt: f64 = 0.0;
    let mut sum_dt: f64 = 0.0;

    for i in 1..n {
        let dt = samples[i].t - samples[i - 1].t;
        min_dt = min_dt.min(dt);
        max_dt = max_dt.max(dt);
        sum_dt += dt;
    }

    let avg_dt = if n > 1 {
        sum_dt / (n - 1) as f64
    } else {
        0.0
    };

    println!("Unevenly Sampled Dataset Summary");
    println!("================================");
    println!("Number of samples                 = {}", n);
    println!("Start time                        = {:.6}", t_start);
    println!("End time                          = {:.6}", t_end);
    println!("Observation span                  = {:.6}", span);
    println!("Minimum sample spacing            = {:.6}", min_dt);
    println!("Average sample spacing            = {:.6}", avg_dt);
    println!("Maximum sample spacing            = {:.6}", max_dt);
    println!();
}

fn main() {
    let mut rng = LcgRng::new(0x5A17_2026_DA7A_BEEF);

    // Generate a synthetic unevenly sampled record.
    let samples = generate_uneven_samples(
        220,  // number of samples
        0.22, // nominal spacing
        0.55, // relative jitter size
        0.28, // base noise level
        &mut rng,
    );

    print_dataset_summary(&samples);

    // Search frequency band.
    let f_min_hz = 0.05;
    let f_max_hz = 3.00;
    let n_freqs = 4000;

    let spectrum = compute_periodogram(&samples, f_min_hz, f_max_hz, n_freqs);
    let peaks = top_peaks(&spectrum, 8, 0.08);

    println!("Weighted Least-Squares Periodogram");
    println!("==================================");
    println!(
        "Frequency search interval [Hz]    = [{:.4}, {:.4}]",
        f_min_hz, f_max_hz
    );
    println!("Number of trial frequencies       = {}", n_freqs);
    println!();

    println!("Strongest detected peaks:");
    println!(
        "{:>4} {:>12} {:>12} {:>14} {:>14} {:>14}",
        "Rank", "f [Hz]", "Power", "Amplitude", "A_cos", "B_sin"
    );

    for (i, p) in peaks.iter().enumerate() {
        println!(
            "{:>4} {:>12.6} {:>12.6} {:>14.6} {:>14.6} {:>14.6}",
            i + 1,
            p.freq_hz,
            p.power,
            p.amplitude,
            p.a_cos,
            p.b_sin
        );
    }

    if let Some(best) = peaks.first() {
        println!();
        println!("Best-Fit Sinusoid at the Strongest Peak");
        println!("=======================================");
        println!("Peak frequency f* [Hz]                = {:.8}", best.freq_hz);
        println!("Peak angular frequency ω*             = {:.8}", best.omega);
        println!("Normalized power                      = {:.8}", best.power);
        println!("Estimated offset                      = {:.8}", best.offset);
        println!("Estimated cosine coefficient A        = {:.8}", best.a_cos);
        println!("Estimated sine coefficient B          = {:.8}", best.b_sin);
        println!(
            "Estimated amplitude sqrt(A^2 + B^2)   = {:.8}",
            best.amplitude
        );
        println!("Weighted residual sum of squares      = {:.8}", best.rss);
    }

    println!();
    println!("Interpretation");
    println!("==============");
    println!("This program evaluates cosine and sine basis functions directly");
    println!("at the observed irregular times t_i and solves a weighted");
    println!("least-squares problem for each trial frequency. The reported");
    println!("power measures the reduction in weighted residual error relative");
    println!("to a floating-mean null model, which is the regression viewpoint");
    println!("underlying Lomb-Scargle spectral estimation.");
}
```

Program 13.9.1 demonstrates a practical realization of least-squares spectral estimation for unevenly sampled data. By formulating the problem directly in terms of regression at the observed times, it avoids the conceptual and numerical pitfalls associated with forcing irregular data into an FFT framework. The results illustrate how dominant frequencies can be recovered accurately even in the presence of irregular sampling and noise.

The modular structure of the implementation makes it straightforward to extend in several directions. One can incorporate more sophisticated weighting schemes, include higher harmonics or multivariate extensions, or replace the direct normal-equation solver with more stable factorizations such as QR decomposition. Additionally, the frequency search can be refined using local optimization techniques or accelerated using NUFFT-based methods for large-scale problems. These extensions connect naturally to the broader themes of modern spectral estimation discussed in this section.

## 13.9.2. Likelihood Methods, Gaussian Processes, and Covariance Modeling

Although least-squares sinusoid fitting is powerful, it is not always sufficient. High-quality modern data, especially in astronomy and related sciences, often exhibit correlated noise, stochastic variability, and irregular observation cadence. In such cases, the classical distributional theory behind simple periodograms can become unreliable. A more principled alternative is to write down a likelihood with an explicit covariance model.

Let $y$ denote the observed data vector and let $K$ be a covariance matrix representing the stochastic background process at the observed times. Under a Gaussian-process model, one may compare a null hypothesis and a periodic alternative through likelihood-based inference. In this framework, the periodic component is not detected by a simple projection formula alone, but through its contribution to the covariance-aware fit. This is particularly attractive when the noise is “red” or temporally correlated, because such behavior can otherwise masquerade as periodicity if one relies on simpler estimators.

A useful modern illustration of this idea appears in the work of Gúrpide and Middleton (2025), who study the detection of periodic signals in unevenly sampled astronomical observations. Classical periodogram methods often exhibit poorly defined statistical properties when sampling is irregular or when the underlying process contains correlated variability. Under these conditions the assumptions that justify standard spectral estimators are no longer satisfied.

To address this difficulty, the authors formulate the problem within a Gaussian-process framework. In this approach the observations are modeled as realizations of a correlated Gaussian process whose covariance structure is explicitly specified. The resulting model provides a well-defined likelihood for the data and permits likelihood-ratio tests for the presence of an additional periodic or quasi-periodic component (Gúrpide and Middleton, 2025).

From a numerical signal-analysis perspective, this formulation connects naturally with the covariance viewpoint underlying the linear minimum mean square error framework discussed earlier in this chapter. When observations are equally spaced, the covariance matrix of a stationary process has Toeplitz structure, which enables several computational simplifications. Irregular sampling destroys this structure. The appropriate strategy is therefore not to impose Toeplitz assumptions artificially, but instead to work directly with the covariance model implied by the actual sampling pattern and noise process.

This likelihood-based viewpoint also highlights a broader conceptual distinction. When sampling is uniform, spectral estimation is commonly formulated in a transform-based framework. One computes Fourier coefficients and interprets their magnitudes as estimates of spectral energy at the corresponding frequencies. With irregular sampling, however, such transform-based interpretations become less natural. The problem is more appropriately treated in a model-based manner, in which a sinusoidal or stochastic model is specified and its parameters are estimated directly from the observed data.

Despite this shift in interpretation, the underlying mathematics remains fundamentally the same. Both approaches ultimately reduce to problems of linear algebra and quadratic optimization. What changes is primarily the computational strategy. Uniformly sampled data allow the use of fast transform algorithms such as the FFT, whereas irregularly sampled data typically lead to formulations expressed in terms of regression problems and covariance-based solvers.

### Rust Implementation

Following the discussion in Section 13.9.2 on likelihood-based spectral analysis and covariance modeling, Program 13.9.2 provides a practical implementation of Gaussian-process-based spectral detection for unevenly sampled data. In contrast to the least-squares approach of Section 13.9.1, this program incorporates an explicit covariance matrix that models correlated noise and irregular sampling geometry. The resulting formulation evaluates candidate periodic models through their contribution to the Gaussian log-likelihood rather than through a projection-based periodogram. This implementation reflects the central idea that, under irregular sampling and correlated variability, spectral estimation is more naturally expressed as a covariance-aware inference problem grounded in linear algebra and quadratic optimization.

At the core of the implementation is the covariance-based generalized least-squares formulation corresponding to the likelihood model described in this section. The function `covariance_matrix` constructs the dense matrix $K$ directly from the observation times using an exponential kernel, thereby encoding the correlated background process. The Cholesky factorization implemented in `cholesky_decompose` provides a numerically stable way to represent $K$ and enables efficient solution of linear systems involving $K^{-1}$ through forward and backward substitution. This reflects the fundamental role of covariance structure in replacing the Toeplitz assumptions that are available under uniform sampling.

The function `profile_gls_log_likelihood` implements the core likelihood evaluation. Given a design matrix corresponding to either the null model or the periodic alternative, it computes the generalized least-squares estimate of the regression coefficients by solving the normal equations involving $K^{-1}$. The residual is then formed, and the quadratic form is evaluated along with the log-determinant of $K$, yielding the Gaussian log-likelihood. This step directly embodies the likelihood formulation discussed in Section 13.9.2 and shows how regression and covariance modeling combine into a single computational procedure.

The functions `design_matrix_null` and `design_matrix_periodic` define the competing models. The null model includes only a constant mean, while the alternative includes cosine and sine components evaluated at the observed times, consistent with the model structure introduced earlier. The function `scan_frequency_grid` then evaluates these models across a range of candidate frequencies, computing the log-likelihood improvement for each. This replaces the classical notion of spectral power with a likelihood-ratio-based criterion, which is better suited to irregular sampling and correlated noise.

To validate the method, the program includes `generate_samples`, which produces a synthetic dataset combining a periodic signal with correlated noise generated by an Ornstein–Uhlenbeck process. This provides a realistic test scenario in which simple periodogram methods can fail, but covariance-aware methods remain reliable. The function `top_frequency_peaks` extracts the most significant candidate frequencies while enforcing separation to avoid redundant detections.

The `main` function coordinates the entire workflow, including data generation, covariance construction, likelihood evaluation, and reporting of results. It demonstrates how the Gaussian-process framework leads naturally to a model-based interpretation of spectral estimation, in which frequencies are identified through improvements in likelihood rather than through transform coefficients.

```rust
// Program 13.9.2: Likelihood Methods, Gaussian Processes, and Covariance Modeling
//
// Problem Statement:
// Implement a covariance-aware spectral detector for unevenly sampled data.
// The program should:
// 1. Generate irregularly sampled observations containing a sinusoid plus
//    correlated Gaussian-process-like background noise.
// 2. Build a dense covariance matrix directly from the observation times.
// 3. Evaluate a Gaussian log-likelihood under a null model and under a
//    periodic alternative with mean function
//       m(t) = c0 + A cos(omega t) + B sin(omega t).
// 4. Search a frequency grid and identify the frequency that maximizes the
//    likelihood improvement over the null model.
// 5. Report the strongest candidate frequencies and fitted coefficients.
//
// This code is intentionally self-contained and uses only the Rust standard
// library so that it can be compiled and run immediately with cargo run.

use std::cmp::Ordering;
use std::f64::consts::PI;

#[derive(Clone)]
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let x = self.state >> 11;
        (x as f64) / ((1u64 << 53) as f64)
    }

    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1.0e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[derive(Debug, Clone)]
struct Sample {
    t: f64,
    y: f64,
}

#[derive(Debug, Clone)]
struct KernelParams {
    sigma_proc: f64,   // correlated process scale
    tau_corr: f64,     // correlation timescale
    sigma_white: f64,  // independent measurement noise
}

#[derive(Debug, Clone)]
struct GlmProfileResult {
    beta: Vec<f64>,
    log_likelihood: f64,
    quad_form: f64,
}

#[derive(Debug, Clone)]
struct FrequencyResult {
    freq_hz: f64,
    omega: f64,
    delta_log_likelihood: f64,
    log_likelihood_alt: f64,
    offset: f64,
    a_cos: f64,
    b_sin: f64,
    amplitude: f64,
}

type Matrix = Vec<Vec<f64>>;

fn zeros(n: usize, m: usize) -> Matrix {
    vec![vec![0.0; m]; n]
}

fn transpose(a: &[Vec<f64>]) -> Matrix {
    let n = a.len();
    let m = if n > 0 { a[0].len() } else { 0 };
    let mut at = zeros(m, n);
    for i in 0..n {
        for j in 0..m {
            at[j][i] = a[i][j];
        }
    }
    at
}

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Matrix {
    let n = a.len();
    let k = if n > 0 { a[0].len() } else { 0 };
    let m = if !b.is_empty() { b[0].len() } else { 0 };
    let mut c = zeros(n, m);

    for i in 0..n {
        for p in 0..k {
            let aip = a[i][p];
            for j in 0..m {
                c[i][j] += aip * b[p][j];
            }
        }
    }

    c
}

fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let n = a.len();
    let m = if n > 0 { a[0].len() } else { 0 };
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..m {
            sum += a[i][j] * x[j];
        }
        y[i] = sum;
    }
    y
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn vec_sub(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y.iter()).map(|(a, b)| a - b).collect()
}

fn cholesky_decompose(a: &[Vec<f64>]) -> Option<Matrix> {
    let n = a.len();
    let mut l = zeros(n, n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }

            if i == j {
                if sum <= 1.0e-14 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    Some(l)
}

fn forward_substitution(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        y[i] = sum / l[i][i];
    }
    y
}

fn backward_substitution_from_lower_transpose(l: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0; n];
    for ii in 0..n {
        let i = n - 1 - ii;
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j];
        }
        x[i] = sum / l[i][i];
    }
    x
}

fn solve_spd(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let y = forward_substitution(l, b);
    backward_substitution_from_lower_transpose(l, &y)
}

fn logdet_from_cholesky(l: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0;
    for i in 0..l.len() {
        sum += l[i][i].ln();
    }
    2.0 * sum
}

fn solve_small_linear_system(mut a: Matrix, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = a.len();

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = a[col][col].abs();

        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }

        if pivot_val < 1.0e-14 {
            return None;
        }

        if pivot_row != col {
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }

        let pivot = a[col][col];
        for j in col..n {
            a[col][j] /= pivot;
        }
        b[col] /= pivot;

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    Some(b)
}

fn covariance_matrix(times: &[f64], params: &KernelParams) -> Matrix {
    let n = times.len();
    let mut k = zeros(n, n);

    for i in 0..n {
        for j in 0..=i {
            let dt = (times[i] - times[j]).abs();
            let corr = params.sigma_proc * params.sigma_proc * (-dt / params.tau_corr).exp();
            let mut kij = corr;
            if i == j {
                kij += params.sigma_white * params.sigma_white;
            }
            k[i][j] = kij;
            k[j][i] = kij;
        }
    }

    k
}

fn design_matrix_null(times: &[f64]) -> Matrix {
    let n = times.len();
    let mut x = zeros(n, 1);
    for i in 0..n {
        x[i][0] = 1.0;
    }
    x
}

fn design_matrix_periodic(times: &[f64], omega: f64) -> Matrix {
    let n = times.len();
    let mut x = zeros(n, 3);
    for i in 0..n {
        x[i][0] = 1.0;
        x[i][1] = (omega * times[i]).cos();
        x[i][2] = (omega * times[i]).sin();
    }
    x
}

fn profile_gls_log_likelihood(y: &[f64], x: &[Vec<f64>], chol_k: &[Vec<f64>]) -> Option<GlmProfileResult> {
    let n = y.len();
    let p = if !x.is_empty() { x[0].len() } else { 0 };

    let k_inv_y = solve_spd(chol_k, y);

    let mut k_inv_x = zeros(n, p);
    for j in 0..p {
        let column_j: Vec<f64> = (0..n).map(|i| x[i][j]).collect();
        let sol_j = solve_spd(chol_k, &column_j);
        for i in 0..n {
            k_inv_x[i][j] = sol_j[i];
        }
    }

    let xt = transpose(x);
    let xt_k_inv_x = matmul(&xt, &k_inv_x);
    let xt_k_inv_y = matvec(&xt, &k_inv_y);

    let beta = solve_small_linear_system(xt_k_inv_x, xt_k_inv_y)?;

    let xb = matvec(x, &beta);
    let residual = vec_sub(y, &xb);
    let k_inv_residual = solve_spd(chol_k, &residual);
    let quad = dot(&residual, &k_inv_residual);

    let logdet_k = logdet_from_cholesky(chol_k);
    let log_likelihood = -0.5 * (quad + logdet_k + (n as f64) * (2.0 * PI).ln());

    Some(GlmProfileResult {
        beta,
        log_likelihood,
        quad_form: quad,
    })
}

fn generate_uneven_times(
    n: usize,
    dt_nominal: f64,
    jitter_fraction: f64,
    rng: &mut LcgRng,
) -> Vec<f64> {
    let mut times = Vec::with_capacity(n);
    let mut t = 0.0;

    for _ in 0..n {
        let jitter = (2.0 * rng.next_f64() - 1.0) * jitter_fraction * dt_nominal;
        let dt_i = (dt_nominal + jitter).max(0.15 * dt_nominal);
        t += dt_i;
        times.push(t);
    }

    times
}

fn generate_correlated_noise_ou(times: &[f64], sigma_proc: f64, tau_corr: f64, rng: &mut LcgRng) -> Vec<f64> {
    let n = times.len();
    let mut x = vec![0.0; n];

    if n == 0 {
        return x;
    }

    x[0] = sigma_proc * rng.next_gaussian();

    for i in 1..n {
        let dt = times[i] - times[i - 1];
        let a = (-dt / tau_corr).exp();
        let innovation_std = sigma_proc * (1.0 - a * a).sqrt();
        x[i] = a * x[i - 1] + innovation_std * rng.next_gaussian();
    }

    x
}

fn generate_samples(
    n: usize,
    dt_nominal: f64,
    jitter_fraction: f64,
    kernel: &KernelParams,
    true_freq_hz: f64,
    true_amplitude: f64,
    true_phase: f64,
    true_offset: f64,
    rng: &mut LcgRng,
) -> Vec<Sample> {
    let times = generate_uneven_times(n, dt_nominal, jitter_fraction, rng);
    let red_noise = generate_correlated_noise_ou(&times, kernel.sigma_proc, kernel.tau_corr, rng);

    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let white = kernel.sigma_white * rng.next_gaussian();
        let y = true_offset
            + true_amplitude * (2.0 * PI * true_freq_hz * times[i] + true_phase).cos()
            + red_noise[i]
            + white;

        samples.push(Sample { t: times[i], y });
    }

    samples
}

fn scan_frequency_grid(
    samples: &[Sample],
    kernel: &KernelParams,
    f_min_hz: f64,
    f_max_hz: f64,
    n_freqs: usize,
) -> Vec<FrequencyResult> {
    let times: Vec<f64> = samples.iter().map(|s| s.t).collect();
    let y: Vec<f64> = samples.iter().map(|s| s.y).collect();

    let k = covariance_matrix(&times, kernel);
    let chol_k = cholesky_decompose(&k).expect("Covariance matrix should be SPD.");

    let x_null = design_matrix_null(&times);
    let null_fit = profile_gls_log_likelihood(&y, &x_null, &chol_k)
        .expect("Null-model GLS fit should succeed.");

    let mut results = Vec::with_capacity(n_freqs);

    for idx in 0..n_freqs {
        let alpha = if n_freqs > 1 {
            idx as f64 / (n_freqs - 1) as f64
        } else {
            0.0
        };

        let freq_hz = f_min_hz + alpha * (f_max_hz - f_min_hz);
        let omega = 2.0 * PI * freq_hz;
        let x_alt = design_matrix_periodic(&times, omega);

        if let Some(alt_fit) = profile_gls_log_likelihood(&y, &x_alt, &chol_k) {
            let offset = alt_fit.beta[0];
            let a_cos = alt_fit.beta[1];
            let b_sin = alt_fit.beta[2];
            let amplitude = (a_cos * a_cos + b_sin * b_sin).sqrt();
            let delta_log_likelihood = alt_fit.log_likelihood - null_fit.log_likelihood;

            results.push(FrequencyResult {
                freq_hz,
                omega,
                delta_log_likelihood,
                log_likelihood_alt: alt_fit.log_likelihood,
                offset,
                a_cos,
                b_sin,
                amplitude,
            });
        }
    }

    results
}

fn top_frequency_peaks(
    spectrum: &[FrequencyResult],
    max_peaks: usize,
    min_separation_hz: f64,
) -> Vec<FrequencyResult> {
    if spectrum.len() < 3 {
        return spectrum.iter().take(max_peaks).cloned().collect();
    }

    let mut candidates: Vec<FrequencyResult> = Vec::new();

    for i in 1..(spectrum.len() - 1) {
        if spectrum[i].delta_log_likelihood >= spectrum[i - 1].delta_log_likelihood
            && spectrum[i].delta_log_likelihood >= spectrum[i + 1].delta_log_likelihood
        {
            candidates.push(spectrum[i].clone());
        }
    }

    candidates.sort_by(|a, b| {
        b.delta_log_likelihood
            .partial_cmp(&a.delta_log_likelihood)
            .unwrap_or(Ordering::Equal)
    });

    let mut selected: Vec<FrequencyResult> = Vec::new();

    'outer: for cand in candidates {
        for s in &selected {
            if (cand.freq_hz - s.freq_hz).abs() < min_separation_hz {
                continue 'outer;
            }
        }
        selected.push(cand);
        if selected.len() >= max_peaks {
            break;
        }
    }

    selected
}

fn print_dataset_summary(samples: &[Sample], kernel: &KernelParams) {
    let n = samples.len();
    let t_start = samples.first().map(|s| s.t).unwrap_or(0.0);
    let t_end = samples.last().map(|s| s.t).unwrap_or(0.0);
    let span = t_end - t_start;

    let mut min_dt: f64 = f64::INFINITY;
    let mut max_dt: f64 = 0.0;
    let mut sum_dt: f64 = 0.0;

    for i in 1..n {
        let dt = samples[i].t - samples[i - 1].t;
        min_dt = min_dt.min(dt);
        max_dt = max_dt.max(dt);
        sum_dt += dt;
    }

    let avg_dt = if n > 1 { sum_dt / (n - 1) as f64 } else { 0.0 };

    println!("Unevenly Sampled Gaussian-Process Dataset");
    println!("=========================================");
    println!("Number of samples                 = {}", n);
    println!("Start time                        = {:.6}", t_start);
    println!("End time                          = {:.6}", t_end);
    println!("Observation span                  = {:.6}", span);
    println!("Minimum sample spacing            = {:.6}", min_dt);
    println!("Average sample spacing            = {:.6}", avg_dt);
    println!("Maximum sample spacing            = {:.6}", max_dt);
    println!();
    println!("Covariance Model Parameters");
    println!("===========================");
    println!("sigma_proc                        = {:.6}", kernel.sigma_proc);
    println!("tau_corr                          = {:.6}", kernel.tau_corr);
    println!("sigma_white                       = {:.6}", kernel.sigma_white);
    println!();
}

fn main() {
    let mut rng = LcgRng::new(0x13_09_02_AB_CD_EF_42);

    let kernel = KernelParams {
        sigma_proc: 0.70,
        tau_corr: 2.40,
        sigma_white: 0.20,
    };

    let true_freq_hz = 0.915;
    let true_amplitude = 1.10;
    let true_phase = 0.65;
    let true_offset = 0.35;

    let samples = generate_samples(
        180,   // number of observations
        0.32,  // nominal spacing
        0.60,  // relative jitter magnitude
        &kernel,
        true_freq_hz,
        true_amplitude,
        true_phase,
        true_offset,
        &mut rng,
    );

    print_dataset_summary(&samples, &kernel);

    let f_min_hz = 0.05;
    let f_max_hz = 2.50;
    let n_freqs = 2500;

    let results = scan_frequency_grid(&samples, &kernel, f_min_hz, f_max_hz, n_freqs);
    let peaks = top_frequency_peaks(&results, 8, 0.06);

    // Also evaluate the null model once here for reporting diagnostics.
    let times: Vec<f64> = samples.iter().map(|s| s.t).collect();
    let y: Vec<f64> = samples.iter().map(|s| s.y).collect();
    let k = covariance_matrix(&times, &kernel);
    let chol_k = cholesky_decompose(&k).expect("Covariance matrix should be SPD.");
    let x_null = design_matrix_null(&times);
    let null_fit = profile_gls_log_likelihood(&y, &x_null, &chol_k)
        .expect("Null-model GLS fit should succeed.");

    println!("Covariance-Aware Frequency Scan");
    println!("===============================");
    println!("Frequency search interval [Hz]  = [{:.4}, {:.4}]", f_min_hz, f_max_hz);
    println!("Number of trial frequencies     = {}", n_freqs);
    println!("Null-model log-likelihood       = {:.8}", null_fit.log_likelihood);
    println!("Null-model quadratic form       = {:.8}", null_fit.quad_form);
    println!();

    println!("Strongest likelihood peaks:");
    println!(
        "{:>4} {:>12} {:>16} {:>14} {:>14} {:>14}",
        "Rank", "f [Hz]", "Delta log L", "Amplitude", "A_cos", "B_sin"
    );

    for (i, p) in peaks.iter().enumerate() {
        println!(
            "{:>4} {:>12.6} {:>16.6} {:>14.6} {:>14.6} {:>14.6}",
            i + 1,
            p.freq_hz,
            p.delta_log_likelihood,
            p.amplitude,
            p.a_cos,
            p.b_sin
        );
    }

    if let Some(best) = peaks.first() {
        println!();
        println!("Best Covariance-Aware Periodic Candidate");
        println!("========================================");
        println!("True planted frequency [Hz]      = {:.8}", true_freq_hz);
        println!("Detected frequency [Hz]          = {:.8}", best.freq_hz);
        println!("Detected angular frequency       = {:.8}", best.omega);
        println!("Likelihood improvement           = {:.8}", best.delta_log_likelihood);
        println!("Alternative log-likelihood       = {:.8}", best.log_likelihood_alt);
        println!("Estimated offset                 = {:.8}", best.offset);
        println!("Estimated cosine coefficient     = {:.8}", best.a_cos);
        println!("Estimated sine coefficient       = {:.8}", best.b_sin);
        println!("Estimated amplitude             = {:.8}", best.amplitude);
    }

    println!();
    println!("Interpretation");
    println!("==============");
    println!("This program models the background variability through an");
    println!("explicit covariance matrix K built from the observed times.");
    println!("Instead of defining spectral power through an FFT-based");
    println!("projection alone, it compares the Gaussian log-likelihood of");
    println!("a covariance-aware null model against that of a covariance-aware");
    println!("periodic alternative. This is the model-based viewpoint that");
    println!("becomes natural when sampling is uneven and the noise is");
    println!("correlated rather than white.");
}
```

Program 13.9.2 demonstrates a modern approach to spectral analysis under uneven sampling and correlated noise. By explicitly modeling covariance and evaluating likelihoods, it avoids the limitations of classical periodograms and provides a statistically principled framework for detecting periodic structure. The strong recovery of the planted frequency in the output confirms the effectiveness of this approach.

The structure of the implementation also highlights its extensibility. More sophisticated covariance kernels can be introduced to model quasi-periodic or multi-scale behavior, and the likelihood framework can be extended to full Bayesian inference with prior distributions on parameters. In addition, computational efficiency can be improved through sparse approximations or iterative solvers for large datasets. These extensions connect directly to current research directions in time-series analysis and Gaussian-process modeling.

## 13.9.3. Modern Computational Directions: NUFFT, Bias Control, and Missing Samples

Recent work in uneven-sampling spectral analysis has developed along two main fronts: acceleration and bias control. The acceleration problem arises because least-squares evaluation at many candidate frequencies can be expensive. If one solves (13.9.6) independently for each frequency, the work scales roughly linearly with the number of samples per frequency, which can become prohibitive when exploring dense frequency grids or very large datasets.

This motivates transform-assisted methods based on the nonuniform FFT (NUFFT). A 2024 arXiv paper by Garrison et al. presents `nifty-ls`, a fast and accurate Lomb–Scargle evaluator built on a NUFFT, claiming substantial speedups over older methods together with significantly improved numerical accuracy, and also supporting GPU/CUDA evaluation (Garrison et al., 2024). The conceptual point is important: the FFT itself is no longer directly applicable because the sampling is nonuniform, but the broader fast-transform idea survives in the form of NUFFT acceleration. Thus one does not abandon spectral structure; one generalizes it.

A second line of recent work focuses on estimator design under irregular sampling. Albentosa-Ruiz and Marchili (2024) introduce a high-pass filter periodogram aimed at reducing sampling-induced noise that affects classical Lomb–Scargle estimation in unevenly sampled settings. Their results emphasize that estimator design, not just faster computation, matters when the sampling pattern itself injects structured bias.

A closely related issue appears even when the underlying time grid is nominally uniform but some samples are missing. In such cases it is tempting to fill the gaps heuristically and then apply a standard FFT or periodogram. Modern work warns strongly against such shortcuts. Chavanne (2026) argues that both standard periodograms and Lomb–Scargle estimators can be biased when samples are missing, even on an otherwise uniform grid. They also warn that some seemingly practical fixes, such as taking absolute values to force nonnegative PSD estimates, can themselves increase bias. Instead, the paper advocates asymptotically unbiased estimators derived from Fourier transforms of unbiased autocorrelation estimators (Chavanne, 2026).

This message is reinforced by Damaschke, Kühn and Nobach (2024), who develop bias-free covariance and PSD estimation procedures for data with missing samples and extended gaps. Their approach combines weighted averaging over valid samples, restriction of the covariance domain, and correction for mean-removal bias, with the explicit goal of producing PSD estimates whose bias does not depend on the spectral structure of the gap pattern. This is a significant conceptual refinement over ad hoc missing-data treatments. It underscores that the correct object to estimate first may be the covariance function rather than the spectrum directly, and that the discrete Wiener–Khinchin relation remains useful even when direct FFT-based methods are no longer statistically reliable.

For very large-scale nonuniform problems, recent work also combines NUFFT acceleration with multitaper ideas. Cui, Brinkmann and Worrell (2026) introduce a multiband–multitaper NUFFT estimator, denoted $\mathrm{M}^2\mathrm{NuFFT}$, designed to reduce computational burden relative to optimal but expensive GPSS-based approaches. Their method partitions the spectrum into subbands, shifts tapers using the NUFFT to avoid repeated generalized eigenproblems, and extends Thomson-style periodicity testing to nonuniform samples. This is an excellent example of modern numerical computing: classical spectral concentration ideas are combined with fast transform technology to address a geometrically more complicated sampling problem.

### Rust Implementation

Following the discussion in Section 13.9.3 on modern computational strategies for unevenly sampled and incomplete data, Program 13.9.3 provides a practical implementation of bias-controlled spectral estimation for time series with missing samples. In contrast to classical FFT-based periodograms, which implicitly assume complete and uniformly spaced data, this program demonstrates how spectral estimation can be reformulated through autocovariance estimation followed by a Fourier transform. This approach reflects the key idea emphasized in this section: when samples are missing, reliable spectral estimates are obtained not by ad hoc gap filling, but by carefully accounting for the sampling pattern in the covariance domain and then applying the discrete Wiener–Khinchin relation.

At the core of the implementation is the construction of the autocovariance sequence using only valid sample pairs. The function `unbiased_autocovariance_missing` evaluates the covariance at each lag by averaging over all index pairs $(i, i+\ell)$ for which both observations are available. This directly addresses the bias introduced by missing data, since each lag is normalized by the number of contributing pairs rather than by the total number of samples. The use of the observed mean ensures that the covariance is centered consistently, avoiding additional bias from incomplete data coverage.

The function `bartlett_taper` applies a linear taper to the autocovariance sequence before transformation. This step reduces variance and mitigates the effects of truncating the covariance at a finite lag, which would otherwise introduce oscillatory artifacts in the resulting spectrum. The subsequent function `autocovariance_to_psd` implements the discrete Wiener–Khinchin relation by embedding the symmetric covariance sequence into a zero-padded array and applying an FFT. This transforms the covariance estimate into a power spectral density while preserving the statistical corrections introduced at the covariance stage.

For comparison, the function `naive_zero_filled_periodogram` computes a conventional periodogram by inserting zeros at missing locations and applying an FFT. This serves as a baseline illustrating the pitfalls discussed in the section. Because the missing-data pattern is implicitly treated as part of the signal, the resulting spectrum contains distortions that reflect the sampling mask rather than the underlying process.

The program also includes signal generation and missing-data simulation through the functions `generate_uniform_signal` and `apply_missing_pattern`. These create a controlled test case with known spectral components and both random missingness and extended gaps. The function `top_peaks` extracts dominant spectral components from each estimator, allowing a direct comparison of their behavior.

The `main` function orchestrates the entire workflow, including data generation, application of missing patterns, computation of both spectral estimators, and reporting of peak frequencies. By presenting both methods side by side, it illustrates the central computational lesson of the section: while FFT-based methods remain computationally attractive, statistically reliable spectral estimation under missing samples requires careful bias control at the covariance level.

```rust
// Program 13.9.3: Bias-Controlled PSD Estimation with Missing Samples
//
// Problem Statement:
// Implement and compare two spectral estimators for a nominally uniform
// time series with missing samples:
// 1. A naive zero-filled periodogram.
// 2. A bias-controlled PSD estimator obtained by first estimating an
//    unbiased autocovariance sequence from valid sample pairs and then
//    transforming it with the discrete Wiener-Khinchin relation.
//
// The program should:
// - Generate a synthetic uniformly sampled signal with two sinusoidal components.
// - Remove samples according to both random missingness and an extended gap.
// - Compute the naive zero-filled periodogram.
// - Compute a bias-controlled autocovariance estimate using only valid pairs.
// - Transform the autocovariance estimate to the frequency domain with an FFT.
// - Compare the dominant peaks and overall spectral behavior.
//
// This implementation is self-contained and uses only the Rust standard library.

use std::cmp::Ordering;
use std::f64::consts::PI;

// -----------------------------------------------------------------------------
// Complex arithmetic
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }


    fn abs2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }
}

use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
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

impl Div<f64> for Complex {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

// -----------------------------------------------------------------------------
// Small RNG
// -----------------------------------------------------------------------------

#[derive(Clone)]
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let x = self.state >> 11;
        (x as f64) / ((1u64 << 53) as f64)
    }

    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1.0e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

// -----------------------------------------------------------------------------
// FFT
// -----------------------------------------------------------------------------

fn bit_reverse_permute(x: &mut [Complex]) {
    let n = x.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            x.swap(i, j);
        }
    }
}

fn fft_in_place(x: &mut [Complex], inverse: bool) {
    let n = x.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of two.");

    bit_reverse_permute(x);

    let mut len = 2usize;
    while len <= n {
        let angle = if inverse {
            2.0 * PI / len as f64
        } else {
            -2.0 * PI / len as f64
        };
        let wlen = Complex::from_polar(1.0, angle);

        for i in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..(len / 2) {
                let u = x[i + j];
                let v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w = w * wlen;
            }
        }

        len <<= 1;
    }

    if inverse {
        let scale = n as f64;
        for xi in x.iter_mut() {
            *xi = *xi / scale;
        }
    }
}

fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two()
}

// -----------------------------------------------------------------------------
// Signal generation and missing-sample model
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DataSet {
    dt: f64,
    values: Vec<f64>,
    observed: Vec<bool>,
}

fn generate_uniform_signal(
    n: usize,
    dt: f64,
    f1: f64,
    a1: f64,
    phi1: f64,
    f2: f64,
    a2: f64,
    phi2: f64,
    offset: f64,
    noise_sigma: f64,
    rng: &mut LcgRng,
) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let t = i as f64 * dt;
        let signal = offset
            + a1 * (2.0 * PI * f1 * t + phi1).cos()
            + a2 * (2.0 * PI * f2 * t + phi2).sin();
        let noise = noise_sigma * rng.next_gaussian();
        y[i] = signal + noise;
    }
    y
}

fn apply_missing_pattern(
    values: Vec<f64>,
    random_keep_probability: f64,
    gap_start: usize,
    gap_end: usize,
    rng: &mut LcgRng,
    dt: f64,
) -> DataSet {
    let n = values.len();
    let mut observed = vec![true; n];

    for i in 0..n {
        if i >= gap_start && i < gap_end {
            observed[i] = false;
        } else {
            observed[i] = rng.next_f64() < random_keep_probability;
        }
    }

    DataSet { dt, values, observed }
}

// -----------------------------------------------------------------------------
// Mean estimation and periodogram
// -----------------------------------------------------------------------------

fn observed_mean(data: &DataSet) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..data.values.len() {
        if data.observed[i] {
            sum += data.values[i];
            count += 1;
        }
    }
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

fn naive_zero_filled_periodogram(data: &DataSet) -> (Vec<f64>, Vec<f64>) {
    let n = data.values.len();
    let mean = observed_mean(data);

    let fft_len = next_power_of_two(n);
    let mut x = vec![Complex::new(0.0, 0.0); fft_len];

    for i in 0..n {
        let centered = if data.observed[i] {
            data.values[i] - mean
        } else {
            0.0
        };
        x[i] = Complex::new(centered, 0.0);
    }

    fft_in_place(&mut x, false);

    let half = fft_len / 2 + 1;
    let mut freqs = vec![0.0; half];
    let mut psd = vec![0.0; half];

    for k in 0..half {
        freqs[k] = k as f64 / (fft_len as f64 * data.dt);
        psd[k] = x[k].abs2() / n as f64;
    }

    (freqs, psd)
}

// -----------------------------------------------------------------------------
// Bias-controlled autocovariance and PSD
// -----------------------------------------------------------------------------

fn unbiased_autocovariance_missing(data: &DataSet, max_lag: usize) -> Vec<f64> {
    let n = data.values.len();
    let mean = observed_mean(data);
    let mut gamma = vec![0.0; max_lag + 1];

    for lag in 0..=max_lag {
        let mut sum = 0.0;
        let mut count = 0usize;

        for i in 0..(n - lag) {
            if data.observed[i] && data.observed[i + lag] {
                let xi = data.values[i] - mean;
                let xj = data.values[i + lag] - mean;
                sum += xi * xj;
                count += 1;
            }
        }

        gamma[lag] = if count > 0 { sum / count as f64 } else { 0.0 };
    }

    gamma
}

fn bartlett_taper(gamma: &mut [f64]) {
    let m = gamma.len().saturating_sub(1);
    if m == 0 {
        return;
    }
    for lag in 0..=m {
        let weight = 1.0 - lag as f64 / (m as f64 + 1.0);
        gamma[lag] *= weight;
    }
}

fn autocovariance_to_psd(gamma: &[f64], dt: f64) -> (Vec<f64>, Vec<f64>) {
    let m = gamma.len() - 1;
    let corr_len = 2 * m + 1;
    let fft_len = next_power_of_two(corr_len);

    let mut seq = vec![Complex::new(0.0, 0.0); fft_len];

    // Symmetric covariance sequence:
    // [gamma[0], gamma[1], ..., gamma[m], 0, ..., gamma[m], ..., gamma[1]]
    seq[0] = Complex::new(gamma[0], 0.0);
    for lag in 1..=m {
        seq[lag] = Complex::new(gamma[lag], 0.0);
        seq[fft_len - lag] = Complex::new(gamma[lag], 0.0);
    }

    fft_in_place(&mut seq, false);

    let half = fft_len / 2 + 1;
    let mut freqs = vec![0.0; half];
    let mut psd = vec![0.0; half];

    for k in 0..half {
        freqs[k] = k as f64 / (fft_len as f64 * dt);

        // Wiener-Khinchin: PSD is the transform of the autocovariance.
        // Small negative values can occur from truncation/tapering/numerics.
        psd[k] = seq[k].re.max(0.0);
    }

    (freqs, psd)
}

// -----------------------------------------------------------------------------
// Peak extraction
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Peak {
    freq_hz: f64,
    value: f64,
}

fn top_peaks(freqs: &[f64], values: &[f64], max_peaks: usize, min_sep_hz: f64) -> Vec<Peak> {
    let n = values.len();
    if n < 3 {
        return Vec::new();
    }

    let mut candidates: Vec<Peak> = Vec::new();
    for i in 1..(n - 1) {
        if values[i] >= values[i - 1] && values[i] >= values[i + 1] {
            candidates.push(Peak {
                freq_hz: freqs[i],
                value: values[i],
            });
        }
    }

    candidates.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(Ordering::Equal));

    let mut selected: Vec<Peak> = Vec::new();
    'outer: for cand in candidates {
        for s in &selected {
            if (cand.freq_hz - s.freq_hz).abs() < min_sep_hz {
                continue 'outer;
            }
        }
        selected.push(cand);
        if selected.len() >= max_peaks {
            break;
        }
    }

    selected
}

// -----------------------------------------------------------------------------
// Reporting
// -----------------------------------------------------------------------------

fn print_dataset_summary(data: &DataSet) {
    let n = data.values.len();
    let observed_count = data.observed.iter().filter(|&&b| b).count();
    let missing_count = n - observed_count;
    let observed_fraction = observed_count as f64 / n as f64;

    let mut longest_gap = 0usize;
    let mut current_gap = 0usize;
    for &obs in &data.observed {
        if obs {
            current_gap = 0;
        } else {
            current_gap += 1;
            longest_gap = longest_gap.max(current_gap);
        }
    }

    println!("Dataset with Missing Samples");
    println!("============================");
    println!("Total samples                     = {}", n);
    println!("Observed samples                  = {}", observed_count);
    println!("Missing samples                   = {}", missing_count);
    println!("Observed fraction                 = {:.6}", observed_fraction);
    println!("Sampling interval dt              = {:.6}", data.dt);
    println!("Record length                     = {:.6}", n as f64 * data.dt);
    println!("Longest consecutive gap           = {}", longest_gap);
    println!();
}

fn print_peak_table(title: &str, peaks: &[Peak]) {
    println!("{}", title);
    println!("{}", "=".repeat(title.len()));
    println!("{:>4} {:>14} {:>18}", "Rank", "f [Hz]", "Spectral Value");

    for (i, p) in peaks.iter().enumerate() {
        println!("{:>4} {:>14.6} {:>18.6}", i + 1, p.freq_hz, p.value);
    }
    println!();
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

fn main() {
    let mut rng = LcgRng::new(0x13_09_03_FE_ED_FA_CE);

    let n = 512;
    let dt = 1.0;
    let true_f1 = 0.09375;
    let true_f2 = 0.21484375;
    let signal = generate_uniform_signal(
        n,
        dt,
        true_f1,
        1.20,
        0.40,
        true_f2,
        0.80,
        -1.10,
        0.25,
        0.35,
        &mut rng,
    );

    let data = apply_missing_pattern(
        signal,
        0.82,   // random keep probability outside the extended gap
        180,    // gap start
        245,    // gap end
        &mut rng,
        dt,
    );

    print_dataset_summary(&data);

    let (freqs_naive, psd_naive) = naive_zero_filled_periodogram(&data);

    let max_lag = 128;
    let mut gamma = unbiased_autocovariance_missing(&data, max_lag);
    bartlett_taper(&mut gamma);
    let (freqs_bias_ctrl, psd_bias_ctrl) = autocovariance_to_psd(&gamma, data.dt);

    let naive_peaks = top_peaks(&freqs_naive, &psd_naive, 8, 0.01);
    let bias_ctrl_peaks = top_peaks(&freqs_bias_ctrl, &psd_bias_ctrl, 8, 0.01);

    println!("True planted frequencies");
    println!("========================");
    println!("f1 = {:.8} Hz", true_f1);
    println!("f2 = {:.8} Hz", true_f2);
    println!();

    print_peak_table("Naive Zero-Filled Periodogram Peaks", &naive_peaks);
    print_peak_table("Bias-Controlled PSD Peaks", &bias_ctrl_peaks);

    println!("Estimator Comparison");
    println!("====================");
    println!("The naive estimator inserts zeros at missing locations and then");
    println!("applies an FFT-based periodogram. This is computationally simple,");
    println!("but it mixes the signal with the spectral structure of the gap mask.");
    println!();
    println!("The bias-controlled estimator first computes an autocovariance");
    println!("sequence from valid sample pairs only, normalizing each lag by the");
    println!("number of available pairs. After mild tapering, the PSD is obtained");
    println!("through the discrete Wiener-Khinchin relation. This shifts the");
    println!("emphasis from direct periodogram evaluation to covariance estimation,");
    println!("which is the statistically safer approach when samples are missing.");
}
```

Program 13.9.3 demonstrates that modern spectral analysis must address both computational efficiency and statistical correctness. The naive zero-filled periodogram recovers dominant frequencies but exhibits spurious structure induced by the sampling pattern. In contrast, the covariance-based estimator produces a more stable and interpretable spectrum by explicitly accounting for missing data. This aligns with recent research emphasizing that estimator design, not just computational acceleration, is critical in uneven-sampling problems.

The modular design of the implementation makes it straightforward to extend. One may incorporate alternative tapers, adaptive lag selection, or multitaper strategies to further reduce variance. Additionally, NUFFT-based acceleration can be integrated to handle very large datasets efficiently. These extensions connect naturally to contemporary developments that combine fast transforms with statistically principled estimators for complex sampling geometries.

## 13.9.4. Applications and Implementation Perspective

Unevenly sampled spectral analysis is especially important in astronomy, where observation times are shaped by day-night cycles, weather, instrument scheduling, and data-quality cuts. These irregular cadences make naive periodicity analysis unreliable, since stochastic variability can be mistaken for periodic structure when sampling is sparse or fragmented. The Gaussian-process-based approach of Gúrpide and Middleton (2025) is tailored to precisely this setting, using a GP-inferred null model to support likelihood-ratio tests for additional periodic components. This is a compelling use case because it shows that significance testing in spectral analysis depends just as much on the noise model as on the sinusoidal model itself.

A second important application arises in scientific telemetry and laboratory data where the nominal sampling grid is uniform but packets are lost or measurements are missing. In such settings the slope of the PSD in log-log coordinates may itself be a quantity of physical interest, for example in diagnosing turbulence regimes, scaling laws, or long-memory behavior. Chavanne (2026) emphasizes that missing samples can bias these slope estimates if one applies classical estimators without correction. The possibility that unbiased-autocorrelation-based estimators may produce negative PSD values for individual realizations is not a defect of the mathematics, but rather a reminder that bias-free estimation and nonnegativity are distinct objectives. This is a subtle but important numerical lesson.

From an implementation perspective, Rust naturally supports two distinct approaches that mirror the mathematical divide. In a model-first implementation, one evaluates least-squares or Gaussian-process models directly. For each candidate frequency (\\omega), one assembles the small normal matrix

$$X_{\omega}^{\top} W X_{\omega} \tag{13.9.7}$$

and solves either a $2\times 2$ system or, in floating-mean variants, a $3\times 3$ system. This costs $O(N)$ work per frequency and is straightforward to vectorize if the trigonometric evaluations are handled carefully. In a transform-first implementation, one instead uses NUFFT-based acceleration when the number of frequencies is very large or the sample count is large enough to justify the additional setup cost. Recent work shows that Lomb–Scargle evaluation can indeed be accelerated substantially by NUFFT and even offloaded to GPUs, while multitaper–NUFFT hybrids further reduce cost for broadband exploration (Garrison et al., 2024; Cui, Brinkmann and Worrell, 2026).

For missing data on an otherwise uniform grid, however, neither of these approaches should be replaced automatically by “fill with zeros and FFT.” That shortcut changes the statistical model and often introduces bias that is easy to overlook. Modern bias-aware pipelines instead estimate covariance carefully from the available data, correct for mean-removal effects, and then map the covariance estimate to a PSD through discrete Wiener–Khinchin relations or related methods (Damaschke, Kühn and Nobach, 2024; Chavanne, 2026).

In summary, spectral analysis of unevenly sampled data requires a conceptual shift from transform-first thinking to model-aware estimation. The underlying numerical problem remains one of fitting linear or Gaussian-process structures to data, but the geometry of the sampling pattern now enters explicitly into the design matrix or covariance matrix. The resulting methods are more general than classical FFT periodograms, and modern developments show that they can also be made fast. What changes is not the mathematical ambition of spectral analysis, but the recognition that irregular sampling must be treated as part of the model rather than as a nuisance to be hidden by interpolation.

### Rust Implementation

Following the discussion in Section 13.9.4 on practical applications and implementation strategies, Program 13.9.4 provides a unified Rust implementation that reflects both model-first and covariance-first perspectives for spectral analysis under irregular or incomplete sampling. The program demonstrates two representative use cases: unevenly sampled astronomical-style data analyzed via direct weighted least-squares fitting, and nominally uniform telemetry data with missing samples analyzed through bias-controlled covariance estimation. In doing so, it illustrates how the computational formulation adapts to the sampling geometry, while preserving the underlying linear-algebraic structure of the problem.

At the core of the first part of the implementation is the direct evaluation of the regression model corresponding to Equation (13.9.7). The function `weighted_floating_mean_fit` assembles the small normal matrix $X_{\omega}^{\top} W X_{\omega}$ and right-hand side $X_{\omega}^{\top} W y$ for each candidate frequency. These are solved using the helper routine `solve_3x3`, yielding estimates of the offset and sinusoidal coefficients. The function `compute_model_first_periodogram` performs this computation across a grid of frequencies, and `top_model_peaks` extracts the dominant components. This approach scales linearly with the number of samples per frequency and is particularly well suited for moderate grid sizes and direct model evaluation.

The second part of the program shifts to a covariance-first strategy appropriate for missing data. The function `unbiased_autocovariance_missing` computes the autocovariance sequence by averaging only over valid sample pairs, thereby correcting for the bias introduced by missing observations. The function `bartlett_taper` applies a linear taper to stabilize the truncated covariance sequence, and `autocovariance_to_psd` converts the result into a power spectral density via the discrete Wiener–Khinchin relation implemented through an FFT. This pipeline emphasizes that the covariance function is the primary object of estimation when data are incomplete.

To analyze scaling behavior, the function `estimate_loglog_slope` performs a linear regression in log-log coordinates over a selected frequency band. This allows estimation of spectral slopes, which are often of physical interest in telemetry and laboratory data. By restricting the fit to a low-frequency band, the method captures long-range dependence while avoiding high-frequency noise contamination.

The `main` function integrates both approaches into a single workflow. It generates synthetic unevenly sampled data to demonstrate direct model fitting and constructs a missing-data scenario for covariance-based PSD estimation. The printed summaries provide diagnostic information about sampling patterns, while the reported peaks and slope estimates illustrate how each method extracts meaningful spectral features under its respective assumptions.

```rust
// Program 13.9.4: Model-First and Covariance-First Spectral Analysis in Rust
//
// Problem Statement:
// Implement two practical spectral-analysis pipelines that reflect the
// implementation perspective of Section 13.9.4.
//
// Part A:
// For unevenly sampled astronomical-style observations, evaluate a
// floating-mean weighted least-squares sinusoidal model directly at the
// observed times and scan a frequency grid.
//
// Part B:
// For nominally uniform telemetry data with missing samples, avoid
// zero-filling and instead estimate the autocovariance from valid sample
// pairs only, map it to a PSD through the discrete Wiener-Khinchin
// relation, and estimate the log-log spectral slope over a chosen band.
//
// The program should:
// 1. Generate an unevenly sampled dataset with a planted periodic component.
// 2. Recover the dominant frequency by a direct O(N) per-frequency scan.
// 3. Generate a red-noise telemetry signal on a uniform grid.
// 4. Remove samples with both random losses and an extended gap.
// 5. Estimate a bias-aware PSD from the autocovariance.
// 6. Estimate the PSD slope in log-log coordinates over a specified band.
// 7. Report results for both application settings.
//
// This implementation is self-contained and uses only the Rust standard
// library so that it can be compiled and run directly with cargo run.

use std::cmp::Ordering;
use std::f64::consts::PI;
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

// ============================================================================
// Small RNG
// ============================================================================

#[derive(Clone)]
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let x = self.state >> 11;
        (x as f64) / ((1u64 << 53) as f64)
    }

    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1.0e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

// ============================================================================
// Complex arithmetic and FFT
// ============================================================================

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    
    fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
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

impl Div<f64> for Complex {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

fn bit_reverse_permute(x: &mut [Complex]) {
    let n = x.len();
    let mut j = 0usize;

    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;

        if i < j {
            x.swap(i, j);
        }
    }
}

fn fft_in_place(x: &mut [Complex], inverse: bool) {
    let n = x.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of two.");

    bit_reverse_permute(x);

    let mut len = 2usize;
    while len <= n {
        let angle = if inverse {
            2.0 * PI / len as f64
        } else {
            -2.0 * PI / len as f64
        };
        let wlen = Complex::from_polar(1.0, angle);

        for i in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..(len / 2) {
                let u = x[i + j];
                let v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w = w * wlen;
            }
        }

        len <<= 1;
    }

    if inverse {
        let scale = n as f64;
        for xi in x.iter_mut() {
            *xi = *xi / scale;
        }
    }
}

fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two()
}

// ============================================================================
// Small dense linear solver
// ============================================================================

fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let mut aug = [[0.0_f64; 4]; 3];

    for i in 0..3 {
        for j in 0..3 {
            aug[i][j] = a[i][j];
        }
        aug[i][3] = b[i];
    }

    for col in 0..3 {
        let mut pivot_row = col;
        let mut pivot_val = aug[col][col].abs();

        for row in (col + 1)..3 {
            let v = aug[row][col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }

        if pivot_val < 1.0e-14 {
            return None;
        }

        if pivot_row != col {
            aug.swap(col, pivot_row);
        }

        let pivot = aug[col][col];
        for j in col..4 {
            aug[col][j] /= pivot;
        }

        for row in 0..3 {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in col..4 {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    Some([aug[0][3], aug[1][3], aug[2][3]])
}

// ============================================================================
// Part A: Model-first uneven-sampling implementation
// ============================================================================

#[derive(Debug, Clone)]
struct UnevenSample {
    t: f64,
    y: f64,
    sigma: f64,
}

#[derive(Debug, Clone)]
struct ModelScanPoint {
    freq_hz: f64,
    omega: f64,
    power: f64,
    offset: f64,
    a_cos: f64,
    b_sin: f64,
    amplitude: f64,
}

fn generate_uneven_astronomy_samples(
    n: usize,
    dt_nominal: f64,
    jitter_fraction: f64,
    true_freq_hz: f64,
    true_amplitude: f64,
    true_phase: f64,
    true_offset: f64,
    noise_sigma: f64,
    rng: &mut LcgRng,
) -> Vec<UnevenSample> {
    let mut samples = Vec::with_capacity(n);
    let mut t = 0.0;

    for _ in 0..n {
        let jitter = (2.0 * rng.next_f64() - 1.0) * jitter_fraction * dt_nominal;
        let dt_i = (dt_nominal + jitter).max(0.10 * dt_nominal);
        t += dt_i;

        let sigma = noise_sigma * (0.8 + 0.4 * rng.next_f64());
        let noise = sigma * rng.next_gaussian();

        let y = true_offset
            + true_amplitude * (2.0 * PI * true_freq_hz * t + true_phase).cos()
            + noise;

        samples.push(UnevenSample { t, y, sigma });
    }

    samples
}

fn weighted_null_rss(samples: &[UnevenSample]) -> (f64, f64) {
    let mut sum_w = 0.0;
    let mut sum_wy = 0.0;

    for s in samples {
        let w = 1.0 / (s.sigma * s.sigma);
        sum_w += w;
        sum_wy += w * s.y;
    }

    let mu = sum_wy / sum_w;

    let mut rss0 = 0.0;
    for s in samples {
        let w = 1.0 / (s.sigma * s.sigma);
        let r = s.y - mu;
        rss0 += w * r * r;
    }

    (mu, rss0)
}

fn weighted_floating_mean_fit(samples: &[UnevenSample], omega: f64) -> Option<ModelScanPoint> {
    let mut xtwx = [[0.0_f64; 3]; 3];
    let mut xtwy = [0.0_f64; 3];

    for s in samples {
        let c = (omega * s.t).cos();
        let sn = (omega * s.t).sin();
        let x = [1.0, c, sn];
        let w = 1.0 / (s.sigma * s.sigma);

        for i in 0..3 {
            xtwy[i] += w * x[i] * s.y;
            for j in 0..3 {
                xtwx[i][j] += w * x[i] * x[j];
            }
        }
    }

    let beta = solve_3x3(xtwx, xtwy)?;
    let offset = beta[0];
    let a_cos = beta[1];
    let b_sin = beta[2];

    let mut rss = 0.0;
    for s in samples {
        let pred = offset + a_cos * (omega * s.t).cos() + b_sin * (omega * s.t).sin();
        let r = s.y - pred;
        let w = 1.0 / (s.sigma * s.sigma);
        rss += w * r * r;
    }

    let (_, rss0) = weighted_null_rss(samples);
    let power = ((rss0 - rss) / rss0).clamp(0.0, 1.0);
    let amplitude = (a_cos * a_cos + b_sin * b_sin).sqrt();

    Some(ModelScanPoint {
        freq_hz: omega / (2.0 * PI),
        omega,
        power,
        offset,
        a_cos,
        b_sin,
        amplitude,
    })
}

fn compute_model_first_periodogram(
    samples: &[UnevenSample],
    f_min_hz: f64,
    f_max_hz: f64,
    n_freqs: usize,
) -> Vec<ModelScanPoint> {
    let mut spectrum = Vec::with_capacity(n_freqs);

    for k in 0..n_freqs {
        let alpha = if n_freqs > 1 {
            k as f64 / (n_freqs - 1) as f64
        } else {
            0.0
        };

        let freq_hz = f_min_hz + alpha * (f_max_hz - f_min_hz);
        let omega = 2.0 * PI * freq_hz;

        if let Some(point) = weighted_floating_mean_fit(samples, omega) {
            spectrum.push(point);
        }
    }

    spectrum
}

fn top_model_peaks(
    spectrum: &[ModelScanPoint],
    max_peaks: usize,
    min_sep_hz: f64,
) -> Vec<ModelScanPoint> {
    if spectrum.len() < 3 {
        return spectrum.iter().take(max_peaks).cloned().collect();
    }

    let mut candidates = Vec::new();
    for i in 1..(spectrum.len() - 1) {
        if spectrum[i].power >= spectrum[i - 1].power
            && spectrum[i].power >= spectrum[i + 1].power
        {
            candidates.push(spectrum[i].clone());
        }
    }

    candidates.sort_by(|a, b| b.power.partial_cmp(&a.power).unwrap_or(Ordering::Equal));

    let mut selected: Vec<ModelScanPoint> = Vec::new();
    'outer: for cand in candidates {
        for s in &selected {
            if (cand.freq_hz - s.freq_hz).abs() < min_sep_hz {
                continue 'outer;
            }
        }
        selected.push(cand);
        if selected.len() >= max_peaks {
            break;
        }
    }

    selected
}

// ============================================================================
// Part B: Covariance-first missing-sample implementation
// ============================================================================

#[derive(Debug, Clone)]
struct MissingDataSet {
    dt: f64,
    values: Vec<f64>,
    observed: Vec<bool>,
}

fn generate_ar1_telemetry(
    n: usize,
    dt: f64,
    rho: f64,
    sigma_innov: f64,
    rng: &mut LcgRng,
) -> Vec<f64> {
    let mut x = vec![0.0; n];
    if n == 0 {
        return x;
    }

    x[0] = sigma_innov * rng.next_gaussian();
    for i in 1..n {
        x[i] = rho * x[i - 1] + sigma_innov * rng.next_gaussian();
    }

    // Remove the mean to emphasize the stochastic slope rather than the DC level.
    let mean = x.iter().sum::<f64>() / n as f64;
    for xi in &mut x {
        *xi -= mean;
    }

    let _ = dt;
    x
}

fn apply_missing_samples(
    values: Vec<f64>,
    dt: f64,
    keep_probability: f64,
    gap_start: usize,
    gap_end: usize,
    rng: &mut LcgRng,
) -> MissingDataSet {
    let n = values.len();
    let mut observed = vec![true; n];

    for i in 0..n {
        if i >= gap_start && i < gap_end {
            observed[i] = false;
        } else {
            observed[i] = rng.next_f64() < keep_probability;
        }
    }

    MissingDataSet { dt, values, observed }
}

fn observed_mean(data: &MissingDataSet) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;

    for i in 0..data.values.len() {
        if data.observed[i] {
            sum += data.values[i];
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

fn unbiased_autocovariance_missing(data: &MissingDataSet, max_lag: usize) -> Vec<f64> {
    let n = data.values.len();
    let mean = observed_mean(data);
    let mut gamma = vec![0.0; max_lag + 1];

    for lag in 0..=max_lag {
        let mut sum = 0.0;
        let mut count = 0usize;

        for i in 0..(n - lag) {
            if data.observed[i] && data.observed[i + lag] {
                let xi = data.values[i] - mean;
                let xj = data.values[i + lag] - mean;
                sum += xi * xj;
                count += 1;
            }
        }

        gamma[lag] = if count > 0 { sum / count as f64 } else { 0.0 };
    }

    gamma
}

fn bartlett_taper(gamma: &mut [f64]) {
    let m = gamma.len().saturating_sub(1);
    if m == 0 {
        return;
    }

    for lag in 0..=m {
        let w = 1.0 - lag as f64 / (m as f64 + 1.0);
        gamma[lag] *= w;
    }
}

fn autocovariance_to_psd(gamma: &[f64], dt: f64) -> (Vec<f64>, Vec<f64>) {
    let m = gamma.len() - 1;
    let corr_len = 2 * m + 1;
    let fft_len = next_power_of_two(corr_len);

    let mut seq = vec![Complex::new(0.0, 0.0); fft_len];
    seq[0] = Complex::new(gamma[0], 0.0);

    for lag in 1..=m {
        seq[lag] = Complex::new(gamma[lag], 0.0);
        seq[fft_len - lag] = Complex::new(gamma[lag], 0.0);
    }

    fft_in_place(&mut seq, false);

    let half = fft_len / 2 + 1;
    let mut freqs = vec![0.0; half];
    let mut psd = vec![0.0; half];

    for k in 0..half {
        freqs[k] = k as f64 / (fft_len as f64 * dt);
        psd[k] = seq[k].re.max(1.0e-16);
    }

    (freqs, psd)
}

fn estimate_loglog_slope(
    freqs: &[f64],
    psd: &[f64],
    f_low: f64,
    f_high: f64,
) -> Option<(f64, f64, usize)> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for (&f, &s) in freqs.iter().zip(psd.iter()) {
        if f > 0.0 && f >= f_low && f <= f_high && s > 0.0 {
            xs.push(f.ln());
            ys.push(s.ln());
        }
    }

    let n = xs.len();
    if n < 2 {
        return None;
    }

    let mean_x = xs.iter().sum::<f64>() / n as f64;
    let mean_y = ys.iter().sum::<f64>() / n as f64;

    let mut sxx = 0.0;
    let mut sxy = 0.0;

    for i in 0..n {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
    }

    if sxx <= 0.0 {
        return None;
    }

    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    Some((slope, intercept, n))
}

// ============================================================================
// Reporting helpers
// ============================================================================

fn print_uneven_summary(samples: &[UnevenSample]) {
    let n = samples.len();
    let t_start = samples.first().map(|s| s.t).unwrap_or(0.0);
    let t_end = samples.last().map(|s| s.t).unwrap_or(0.0);

    let mut min_dt: f64 = f64::INFINITY;
    let mut max_dt: f64 = 0.0;
    let mut sum_dt: f64 = 0.0;

    for i in 1..n {
        let dt = samples[i].t - samples[i - 1].t;
        min_dt = min_dt.min(dt);
        max_dt = max_dt.max(dt);
        sum_dt += dt;
    }

    let avg_dt = if n > 1 {
        sum_dt / (n - 1) as f64
    } else {
        0.0
    };

    println!("Uneven Astronomy-Like Dataset");
    println!("=============================");
    println!("Number of samples                 = {}", n);
    println!("Start time                        = {:.6}", t_start);
    println!("End time                          = {:.6}", t_end);
    println!("Minimum sample spacing            = {:.6}", min_dt);
    println!("Average sample spacing            = {:.6}", avg_dt);
    println!("Maximum sample spacing            = {:.6}", max_dt);
    println!();
}

fn print_missing_summary(data: &MissingDataSet) {
    let n = data.values.len();
    let observed_count = data.observed.iter().filter(|&&b| b).count();
    let missing_count = n - observed_count;
    let observed_fraction = observed_count as f64 / n as f64;

    let mut longest_gap = 0usize;
    let mut current_gap = 0usize;
    for &obs in &data.observed {
        if obs {
            current_gap = 0;
        } else {
            current_gap += 1;
            longest_gap = longest_gap.max(current_gap);
        }
    }

    println!("Telemetry Dataset with Missing Samples");
    println!("======================================");
    println!("Total samples                     = {}", n);
    println!("Observed samples                  = {}", observed_count);
    println!("Missing samples                   = {}", missing_count);
    println!("Observed fraction                 = {:.6}", observed_fraction);
    println!("Sampling interval dt              = {:.6}", data.dt);
    println!("Longest consecutive gap           = {}", longest_gap);
    println!();
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let mut rng = LcgRng::new(0x13_09_04_AB_CD_EF_77);

    // ------------------------------------------------------------------------
    // Part A: Unevenly sampled astronomical-style analysis
    // ------------------------------------------------------------------------

    let true_freq_hz = 0.731;
    let true_amplitude = 1.15;
    let true_phase = 0.55;
    let true_offset = 0.30;

    let uneven_samples = generate_uneven_astronomy_samples(
        240,
        0.24,
        0.60,
        true_freq_hz,
        true_amplitude,
        true_phase,
        true_offset,
        0.28,
        &mut rng,
    );

    print_uneven_summary(&uneven_samples);

    let model_spectrum = compute_model_first_periodogram(&uneven_samples, 0.05, 2.50, 3000);
    let model_peaks = top_model_peaks(&model_spectrum, 6, 0.06);

    println!("Model-First Frequency Scan");
    println!("==========================");
    println!("True planted frequency [Hz]      = {:.8}", true_freq_hz);
    println!();
    println!(
        "{:>4} {:>12} {:>12} {:>14} {:>14} {:>14}",
        "Rank", "f [Hz]", "Power", "Amplitude", "A_cos", "B_sin"
    );

    for (i, p) in model_peaks.iter().enumerate() {
        println!(
            "{:>4} {:>12.6} {:>12.6} {:>14.6} {:>14.6} {:>14.6}",
            i + 1,
            p.freq_hz,
            p.power,
            p.amplitude,
            p.a_cos,
            p.b_sin
        );
    }

    if let Some(best) = model_peaks.first() {
        println!();
        println!("Best Uneven-Sampling Fit");
        println!("========================");
        println!("Detected frequency [Hz]          = {:.8}", best.freq_hz);
        println!("Detected angular frequency       = {:.8}", best.omega);
        println!("Estimated offset                 = {:.8}", best.offset);
        println!("Estimated amplitude             = {:.8}", best.amplitude);
        println!(
            "Absolute frequency error         = {:.8}",
            (best.freq_hz - true_freq_hz).abs()
        );
    }

    println!();
    println!("Implementation Note");
    println!("===================");
    println!("This stage mirrors the model-first viewpoint of Equation (13.9.7):");
    println!("for each candidate frequency, the code assembles the small weighted");
    println!("normal system and solves it directly. The work per frequency is O(N),");
    println!("which is simple and effective when the frequency grid is moderate.");
    println!();

    // ------------------------------------------------------------------------
    // Part B: Uniform-grid telemetry with missing samples and slope estimation
    // ------------------------------------------------------------------------

    let telemetry = generate_ar1_telemetry(
        1024,
        1.0,
        0.92,   // strong low-frequency persistence
        0.45,
        &mut rng,
    );

    let missing_data = apply_missing_samples(
        telemetry,
        1.0,
        0.85,
        360,
        470,
        &mut rng,
    );

    print_missing_summary(&missing_data);

    let max_lag = 256;
    let mut gamma = unbiased_autocovariance_missing(&missing_data, max_lag);
    bartlett_taper(&mut gamma);
    let (freqs_psd, psd_bias_aware) = autocovariance_to_psd(&gamma, missing_data.dt);

    let slope_band_low = 0.01;
    let slope_band_high = 0.12;
    let slope_fit = estimate_loglog_slope(&freqs_psd, &psd_bias_aware, slope_band_low, slope_band_high);

    println!("Covariance-First PSD Slope Estimation");
    println!("=====================================");
    println!("Slope fit band [Hz]                 = [{:.4}, {:.4}]", slope_band_low, slope_band_high);

    if let Some((slope, intercept, n_used)) = slope_fit {
        println!("Estimated log-log slope             = {:.8}", slope);
        println!("Estimated log-log intercept         = {:.8}", intercept);
        println!("Frequency bins used                 = {}", n_used);
    } else {
        println!("Slope estimation failed because too few valid PSD points were available.");
    }

    // Report the dominant low-frequency peaks as an implementation diagnostic.
    let low_band_peaks = {
        let mut peaks = Vec::<(f64, f64)>::new();
        for i in 1..(psd_bias_aware.len() - 1) {
            if freqs_psd[i] <= 0.25
                && psd_bias_aware[i] >= psd_bias_aware[i - 1]
                && psd_bias_aware[i] >= psd_bias_aware[i + 1]
            {
                peaks.push((freqs_psd[i], psd_bias_aware[i]));
            }
        }

        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        peaks.truncate(6);
        peaks
    };

    println!();
    println!("Dominant Low-Frequency PSD Peaks");
    println!("================================");
    println!("{:>4} {:>14} {:>18}", "Rank", "f [Hz]", "PSD Value");

    for (i, (f, s)) in low_band_peaks.iter().enumerate() {
        println!("{:>4} {:>14.6} {:>18.6}", i + 1, f, s);
    }

    println!();
    println!("Implementation Note");
    println!("===================");
    println!("This stage follows the covariance-first pipeline emphasized for");
    println!("missing data. Instead of zero-filling and applying an FFT directly,");
    println!("the code estimates the autocovariance from valid pairs, tapers it,");
    println!("and then transforms it to a PSD. This is the safer route when the");
    println!("slope of the spectrum is itself a quantity of scientific interest.");
    println!();

    println!("Summary");
    println!("=======");
    println!("The first application demonstrates direct model evaluation on uneven");
    println!("observation times, while the second demonstrates bias-aware PSD");
    println!("estimation on a nominally uniform grid with missing samples.");
    println!("Together they reflect the two main implementation perspectives");
    println!("described in Section 13.9.4.");
}
```

Program 13.9.4 demonstrates that spectral analysis in modern applications is not tied to a single computational paradigm. Instead, the appropriate method depends on the structure of the data and the statistical questions being asked. The model-first approach excels when direct parameter estimation is feasible, while the covariance-first approach provides robustness when sampling irregularities or missing data would otherwise introduce bias.

The modular design of the implementation allows for straightforward extension. The model-first component can be accelerated using NUFFT techniques for large frequency grids, while the covariance-first pipeline can be enhanced with multitaper methods or more sophisticated covariance estimators. Together, these approaches illustrate the central theme of this section: irregular sampling must be treated as an intrinsic part of the model, and modern numerical methods provide both the flexibility and efficiency needed to do so.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/7C9r2OYtqPbGGuq47TaG.6","tags":[]}

# 13.10. Computing Fourier Integrals Using the FFT

This section returns to a fundamental numerical theme of Fourier methods: many operators of interest are defined in continuous form, but the available data are discrete. The practical question is therefore not merely how to write a Fourier integral, but how to compute it rapidly and accurately from sampled values while preserving a clear connection between the continuous operator and the discrete algorithm. In this setting the FFT is indispensable, but it must be used with care. An FFT is a fast transform algorithm, not by itself a quadrature rule. To turn it into a reliable computational tool for Fourier integrals, one must specify how the underlying function is approximated, how endpoint behavior is handled, and how oscillation is integrated against that approximation. Recent work emphasizes precisely this distinction and develops corrected FFT-based quadrature rules, oversampling strategies, and nonuniform-frequency extensions that retain FFT speed while improving numerical fidelity (Kircheis, Potts and Tasche, 2024; Donciu, Temneanu and Serea, 2025; Anand and Dhiman, 2024).

Three viewpoints will be useful throughout this section. In the *integral view*, one studies operators such as:

$$I(\omega)=\int_{a}^{b} h(t)\,e^{i\omega t}\,dt \tag{13.10.1}$$

In the *matrix view*, one interprets discrete transforms as matrix-vector products. For a vector $x\in\mathbb{C}^N$, the discrete Fourier transform is:

$$X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i kn/N}, \qquad k = 0,1,\dots,N-1 \tag{13.10.2}$$

or, in matrix form,

$$X = F x, \qquad F_{k,n} = e^{-2\pi i kn/N} \tag{13.10.3}$$

In the *filter-bank view*, which becomes important in the next section, multiscale transforms are interpreted as repeated convolutions and decimations. For present purposes, the matrix view is especially valuable because it makes storage, padding, and memory traffic explicit, which is essential in a robust Rust implementation.

## 13.10.1. Fourier-Type Integrals and the Limits of the Naïve FFT Approximation

A common task is to evaluate the Fourier-type integral:

$$I(\omega) = \int_{a}^{b} h(t)\,e^{i\omega t}\,dt \tag{13.10.4}$$

for many values of $\omega$. Such integrals arise in spectral PDE solvers, scattering calculations, characteristic-function pricing, inversion formulas, and fast convolution pipelines (Le Floc’h, 2025; Anand and Dhiman, 2024). A natural first attempt is to sample $h$ on a uniform grid. Let,

$$
\begin{aligned}
\Delta &= \frac{b-a}{M}, \\
t_j &= a + j\Delta, \qquad
h_j = h(t_j), \qquad j = 0,1,\dots,M.
\end{aligned}
\tag{13.10.5}
$$

Then a simple trapezoidal or Riemann approximation gives,

$$I(\omega) \approx \Delta \sum_{j=0}^{M} h_j e^{i\omega t_j} \tag{13.10.6}$$

If one evaluates this expression only at FFT-grid frequencies,

$$\omega_n \Delta = \theta_n = \frac{2\pi n}{N}, \qquad n = 0,1,\dots,N-1 \tag{13.10.7}$$

with $N\ge M+1$ after zero padding if necessary, then the oscillatory bulk sum in (13.10.6) becomes a DFT. This makes the method computationally attractive.

However, the approximation can be numerically poor unless $h$ is already well matched to the periodic structure implicitly imposed by the FFT. The reason is that the FFT interprets sampled data as one period of a periodic sequence. If $h$ is not periodic on $[a,b]$, or if it does not decay smoothly at the endpoints, then the induced periodic extension contains jumps or kinks, and the resulting Fourier coefficients decay slowly. In other words, the FFT supplies speed, but not automatically a valid oscillatory quadrature rule. The numerical approximation then inherits endpoint artifacts rather than the true smoothness of the underlying integrand (Donciu, Temneanu and Serea, 2025).

A useful modern principle is therefore the following: to obtain a reliable FFT-based Fourier integral algorithm, one must first choose an approximation space for (h), such as piecewise polynomials or Fourier extensions, and then integrate that approximation analytically, or nearly analytically, against the oscillatory kernel. This is the idea behind interpolation-kernel corrections and also behind Filon-type quadrature methods (Anand and Dhiman, 2024; Nzokem, 2025).

### Rust Implementation

Following the discussion in Section 13.10.1 on the limitations of naïve FFT-based approximations to Fourier-type integrals, Program 13.10.1 provides a practical implementation of the trapezoidal discretization in Equation (13.10.6) evaluated on an FFT-compatible frequency grid. In numerical computation, the FFT offers a powerful mechanism for accelerating oscillatory sums, but it does not by itself guarantee accuracy as a quadrature rule. The reliability of the approximation depends critically on how the underlying function is sampled, how endpoint behavior is handled, and how the discrete representation aligns with the implicit periodic structure of the FFT. This program evaluates Fourier integrals for both periodic and nonperiodic test functions, allowing direct observation of the numerical artifacts that arise from endpoint mismatch and illustrating the distinction between computational speed and quadrature fidelity.

At the core of the implementation is the function `fourier_integral_naive_fft`, which realizes the trapezoidal approximation described in Equation (13.10.6) while restricting evaluation to FFT-grid frequencies defined in Equation (13.10.7). The function constructs a uniform grid as in Equation (13.10.5), applies trapezoidal endpoint weighting, and forms a zero-padded sequence suitable for FFT evaluation. The oscillatory sum is then computed efficiently using a radix-2 FFT, effectively transforming the discrete approximation into a matrix-vector operation consistent with the matrix view of the Fourier transform introduced earlier in the section.

The FFT itself is implemented through the function `fft_in_place`, which performs an iterative radix-2 Cooley–Tuk algorithm. The auxiliary function `bit_reverse_permute` reorders the data to enable in-place butterfly operations, ensuring computational efficiency and memory locality. The implementation supports both forward and inverse transforms, and the inverse-sign convention is used to match the phase structure required by Equation (13.10.6). After transformation, the result is scaled appropriately to recover the unnormalized oscillatory sum.

To assess accuracy, the program includes exact integral evaluations through functions such as `exact_periodic_integral` and `exact_nonperiodic_integral`. These functions compute closed-form expressions for the Fourier integrals of the test functions, allowing direct comparison with the numerical approximation. The function `max_abs_error` identifies the worst-case deviation across the frequency grid, while `print_selected_samples` reports representative values to illustrate convergence behavior at low and moderate frequencies.

The program also defines two test functions, `h_periodic` and `h_nonperiodic`, which differ only in their endpoint behavior. The periodic function satisfies $h(a)=h(b)$, ensuring compatibility with the periodic extension implied by the FFT, whereas the nonperiodic function introduces a mismatch at the endpoints. This design allows the numerical consequences of the periodicity assumption to be observed directly, without altering the smoothness of the underlying function.

The `main` function orchestrates the entire computation. It constructs the sampling grid, evaluates the Fourier integral using the FFT-based approximation, computes exact reference values, and reports both global error statistics and selected frequency samples. The inclusion of Nyquist frequency information and restriction to nonnegative frequency bins ensures that the results are interpreted consistently with the discrete frequency domain implied by the FFT.

```rust
// Program 13.10.1. Naïve FFT-Based Fourier Integral Approximation
//
// Problem statement:
// Evaluate the Fourier-type integral
//
//     I(ω) = ∫_a^b h(t) e^{iωt} dt
//
// on an FFT-compatible frequency grid using the direct trapezoidal approximation
// discussed in Section 13.10.1. The program uses a radix-2 FFT to accelerate the
// oscillatory bulk sum at frequencies satisfying
//
//     ω_n Δ = 2π n / N,   n = 0, 1, ..., N/2,
//
// with zero padding when N > M + 1. Only the nonnegative, nonaliased half of the
// FFT grid is reported. This is essential: although the DFT is periodic in the phase
// variable, the physical frequency interpretation is unique only up to the Nyquist
// limit π / Δ.
//
// Two test functions are used:
// 1. a smooth periodic function on [a,b], for which the naïve FFT approximation is
//    comparatively well behaved,
// 2. a smooth but nonperiodic function on [a,b], for which endpoint mismatch causes
//    the periodic extension implied by the FFT to develop a jump.
//
// The program reports selected frequency samples and global error statistics so that
// the numerical limitations of the naïve approximation are visible in a single run.

use std::f64::consts::PI;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

#[derive(Clone, Copy, Debug, Default)]
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
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Div<f64> for Complex {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

// ======================================================
// Iterative radix-2 FFT with sign exp(-2πikn/N) forward
// ======================================================

fn bit_reverse_permute(data: &mut [Complex]) {
    let n = data.len();
    let mut j = 0usize;

    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

fn fft_in_place(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of two.");

    bit_reverse_permute(data);

    let mut len = 2usize;
    while len <= n {
        let angle = if inverse {
            2.0 * PI / len as f64
        } else {
            -2.0 * PI / len as f64
        };
        let wlen = Complex::from_polar(1.0, angle);

        let half = len / 2;
        let mut start = 0usize;
        while start < n {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half {
                let u = data[start + j];
                let v = data[start + j + half] * w;
                data[start + j] = u + v;
                data[start + j + half] = u - v;
                w = w * wlen;
            }
            start += len;
        }

        len <<= 1;
    }

    if inverse {
        for z in data.iter_mut() {
            *z = *z / n as f64;
        }
    }
}

// ================================================
// Exact integrals for the demonstration functions
// ================================================

fn exp_integral(alpha: f64, a: f64, b: f64) -> Complex {
    // ∫_a^b e^{i alpha t} dt
    if alpha.abs() < 1.0e-14 {
        Complex::new(b - a, 0.0)
    } else {
        let eb = Complex::from_polar(1.0, alpha * b);
        let ea = Complex::from_polar(1.0, alpha * a);
        let num = eb - ea;
        Complex::new(num.im / alpha, -num.re / alpha)
    }
}

fn t_exp_integral(omega: f64, a: f64, b: f64) -> Complex {
    // ∫_a^b t e^{i omega t} dt
    if omega.abs() < 1.0e-14 {
        Complex::new(0.5 * (b * b - a * a), 0.0)
    } else {
        let eb = Complex::from_polar(1.0, omega * b);
        let ea = Complex::from_polar(1.0, omega * a);

        let fb = eb * Complex::new(1.0 / (omega * omega), -b / omega);
        let fa = ea * Complex::new(1.0 / (omega * omega), -a / omega);

        fb - fa
    }
}

fn exact_periodic_integral(omega: f64, a: f64, b: f64) -> Complex {
    // h_periodic(t) = sin(2πt) + 0.5 cos(4πt)
    let beta1 = 2.0 * PI;
    let beta2 = 4.0 * PI;

    let sin_part = {
        let p = exp_integral(omega + beta1, a, b);
        let m = exp_integral(omega - beta1, a, b);
        Complex::new((p.im - m.im) * 0.5, -(p.re - m.re) * 0.5)
    };

    let cos_part = {
        let p = exp_integral(omega + beta2, a, b);
        let m = exp_integral(omega - beta2, a, b);
        (p + m) * 0.25
    };

    sin_part + cos_part
}

fn exact_nonperiodic_integral(omega: f64, a: f64, b: f64) -> Complex {
    exact_periodic_integral(omega, a, b) + t_exp_integral(omega, a, b) * 0.5
}

// =====================
// Test functions h(t)
// =====================

fn h_periodic(t: f64) -> f64 {
    (2.0 * PI * t).sin() + 0.5 * (4.0 * PI * t).cos()
}

fn h_nonperiodic(t: f64) -> f64 {
    h_periodic(t) + 0.5 * t
}

// ==================================================================
// Naïve FFT-compatible trapezoidal approximation on nonnegative bins
// ==================================================================

fn fourier_integral_naive_fft<F>(
    h: F,
    a: f64,
    b: f64,
    m: usize,
    nfft: usize,
) -> Vec<(usize, f64, Complex)>
where
    F: Fn(f64) -> f64,
{
    assert!(nfft.is_power_of_two(), "FFT length must be a power of two.");
    assert!(
        nfft >= m + 1,
        "FFT length must satisfy N >= M + 1 for zero padding."
    );

    let delta = (b - a) / m as f64;

    // Trapezoidal weights on j = 0, ..., M, then zero padding to N.
    let mut data = vec![Complex::zero(); nfft];
    for j in 0..=m {
        let t_j = a + j as f64 * delta;
        let weight = if j == 0 || j == m { 0.5 } else { 1.0 };
        data[j] = Complex::new(weight * h(t_j), 0.0);
    }

    // We need Σ x_j e^{+i 2π n j / N}, so use the inverse-sign FFT.
    fft_in_place(&mut data, true);
    for z in data.iter_mut() {
        *z = *z * nfft as f64; // undo the inverse normalization
    }

    // Only return the unique nonnegative physical frequencies: n = 0, ..., N/2.
    let mut out = Vec::with_capacity(nfft / 2 + 1);
    for n in 0..=nfft / 2 {
        let theta_n = 2.0 * PI * n as f64 / nfft as f64;
        let omega_n = theta_n / delta;
        let phase_a = Complex::from_polar(1.0, omega_n * a);
        let approx = phase_a * data[n] * delta;
        out.push((n, omega_n, approx));
    }

    out
}

// =====================
// Error and diagnostics
// =====================

fn max_abs_error(
    numerical: &[(usize, f64, Complex)],
    exact: fn(f64, f64, f64) -> Complex,
    a: f64,
    b: f64,
) -> (usize, f64, f64) {
    let mut best_n = 0usize;
    let mut best_omega = 0.0;
    let mut best_err = -1.0_f64;

    for &(n, omega, approx) in numerical {
        let err = (approx - exact(omega, a, b)).abs();
        if err > best_err {
            best_err = err;
            best_n = n;
            best_omega = omega;
        }
    }

    (best_n, best_omega, best_err)
}

fn print_selected_samples(
    title: &str,
    numerical: &[(usize, f64, Complex)],
    exact: fn(f64, f64, f64) -> Complex,
    a: f64,
    b: f64,
    indices: &[usize],
) {
    println!();
    println!("{title}");
    println!("{}", "-".repeat(title.len()));
    println!(
        "{:>6} {:>14} {:>18} {:>18} {:>14}",
        "n", "omega_n", "|I_num|", "|I_exact|", "abs error"
    );

    for &idx in indices {
        if idx < numerical.len() {
            let (n, omega, approx) = numerical[idx];
            let truth = exact(omega, a, b);
            let err = (approx - truth).abs();
            println!(
                "{:>6} {:>14.6} {:>18.10e} {:>18.10e} {:>14.6e}",
                n,
                omega,
                approx.abs(),
                truth.abs(),
                err
            );
        }
    }
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let m = 256usize;    // number of subintervals, so M+1 samples
    let nfft = 512usize; // zero-padded FFT length, N >= M+1 and power of two

    let delta = (b - a) / m as f64;
    let nyquist = PI / delta;

    println!("Computing Fourier Integrals Using the FFT");
    println!("=========================================");
    println!("Naïve FFT-Compatible Trapezoidal Approximation");
    println!("----------------------------------------------");
    println!("Interval [a,b]            = [{:.6}, {:.6}]", a, b);
    println!("Number of subintervals M  = {}", m);
    println!("Grid spacing Δ            = {:.10}", delta);
    println!("FFT length N              = {}", nfft);
    println!("Nyquist frequency π/Δ     = {:.6}", nyquist);
    println!("Reported bins             = 0..N/2 = 0..{}", nfft / 2);

    println!();
    println!("Endpoint Behavior");
    println!("=================");
    println!(
        "Periodic test:    h(a) = {:>.10}, h(b) = {:>.10}",
        h_periodic(a),
        h_periodic(b)
    );
    println!(
        "Nonperiodic test: h(a) = {:>.10}, h(b) = {:>.10}",
        h_nonperiodic(a),
        h_nonperiodic(b)
    );

    let periodic_num = fourier_integral_naive_fft(h_periodic, a, b, m, nfft);
    let nonperiodic_num = fourier_integral_naive_fft(h_nonperiodic, a, b, m, nfft);

    let (p_n, p_omega, p_err) = max_abs_error(&periodic_num, exact_periodic_integral, a, b);
    let (np_n, np_omega, np_err) =
        max_abs_error(&nonperiodic_num, exact_nonperiodic_integral, a, b);

    println!();
    println!("Global Error Summary");
    println!("====================");
    println!(
        "Periodic test:    max abs error = {:>.10e} at n = {}, omega = {:.6}",
        p_err, p_n, p_omega
    );
    println!(
        "Nonperiodic test: max abs error = {:>.10e} at n = {}, omega = {:.6}",
        np_err, np_n, np_omega
    );

    // Representative nonnegative bins.
    let sample_indices = [0usize, 1, 2, 3, 4, 8, 16, 32, 64, 96, 128, 192, 256];

    print_selected_samples(
        "Periodic Test Function",
        &periodic_num,
        exact_periodic_integral,
        a,
        b,
        &sample_indices,
    );

    print_selected_samples(
        "Nonperiodic Test Function",
        &nonperiodic_num,
        exact_nonperiodic_integral,
        a,
        b,
        &sample_indices,
    );

    println!();
    println!("Interpretation");
    println!("==============");
    println!("1. The approximation implements the FFT-grid version of the trapezoidal sum in Equation (13.10.6).");
    println!("2. Only the nonnegative bins up to the Nyquist limit are interpreted as distinct physical frequencies.");
    println!("3. Because the FFT treats sampled data as one period of a periodic sequence, endpoint mismatch is inherited by the approximation.");
    println!("4. The periodic test is comparatively benign because h(a) and h(b) agree, whereas the nonperiodic test exhibits the endpoint defect emphasized in Section 13.10.1.");
    println!("5. This program therefore shows that FFT speed alone does not guarantee a reliable Fourier-integral quadrature rule.");
}
```

Program 13.10.1 demonstrates that while the FFT provides an efficient mechanism for evaluating oscillatory sums, it does not automatically yield an accurate quadrature rule for Fourier integrals. The results show that when the sampled function is periodic on the interval, the approximation is relatively accurate, with errors remaining small across the frequency range. In contrast, when the function is nonperiodic, the implicit periodic extension introduces discontinuities that lead to slower decay of Fourier coefficients and increased error, particularly at higher frequencies.

This example highlights the central computational insight of Section 13.10.1: accuracy in Fourier integral evaluation depends not only on algorithmic speed but also on the compatibility between the function representation and the transform structure. The naïve FFT-based approach serves as a baseline, motivating the need for corrected quadrature rules, interpolation kernels, and more advanced methods that preserve both efficiency and numerical fidelity. The modular design of the program also provides a foundation for extending the implementation to these improved techniques in subsequent sections.

## 13.10.2. Interpolation-Kernel Corrected FFT Quadrature

The basic correction strategy consists of replacing the sampled function by an interpolating approximation whose oscillatory integral can be evaluated exactly or with analytically controlled weights. In this approach, the integrand is first approximated by a suitable interpolant constructed from the sampled values, after which the resulting oscillatory integral can be computed in closed form or through precomputed analytic factors. Accordingly, one writes,

$$h(t) \approx \sum_{j=0}^{M} h_j \,\Psi\!\left(\frac{t - t_j}{\Delta}\right) + \text{endpoint corrections} \tag{13.10.8}$$

where $\Psi$ is a compactly supported interpolation kernel. Near the endpoints $a$ and $b$, additional correction terms are introduced so that the interpolation remains accurate without losing stencil support. This construction removes the hidden periodicity assumption of the raw FFT approximation while retaining FFT speed for the interior oscillatory sum.

Applying the Fourier integral operator to (13.10.8), and writing $\theta=\omega\Delta$, yields a corrected quadrature rule of the form,

$$
I(\omega) \approx \Delta e^{i\omega a}\left[
W(\theta)\sum_{j=0}^{N-1} h_j e^{ij\theta}
+ \sum_{j\in E}\alpha_j(\theta)h_j
+ e^{i\omega(b-a)}\sum_{j\in E}\alpha_j^{*}(\theta)h_{M-j}
\right] \tag{13.10.9}
$$

Here $N$ is the FFT length, $E={0,1,\dots,p}$ is a small endpoint set determined by the interpolation order, $W(\theta)$ is the attenuation factor given by the Fourier transform of the interpolation kernel, and the coefficients $\alpha_j(\theta)$ are analytic endpoint-correction weights. Formally,

$$W(\theta) = \int_{-\infty}^{\infty} e^{i\theta s}\,\Psi(s)\,ds \tag{13.10.10}$$

$$\alpha_j(\theta) = \int_{-\infty}^{\infty} e^{i\theta s}\,\phi_j(s - j)\,ds \tag{13.10.11}$$

where $\phi_j$ encodes the local endpoint stencil correction. Equation (13.10.9) should be interpreted as a genuine spectral quadrature rule. The FFT computes the oscillatory interior sum efficiently, while $W(\theta)$ and the $\alpha_j(\theta)$ terms ensure that the chosen interpolant, including its boundary behavior, is integrated correctly. This is conceptually close to Filon quadrature: one chooses an approximation space and integrates the oscillatory moments of that approximation rather than trusting a bare trapezoidal sum (Anand and Dhiman, 2024).

For linear interpolation, $\Psi$ is the hat function, and the attenuation factor reduces to the classical trapezoidal form,

$$W_{\mathrm{trap}}(\theta) = \frac{2(1-\cos\theta)}{\theta^2} \tag{13.10.12}$$

For cubic interpolation, $W(\theta)$ becomes a rational expression involving $\cos\theta$ and $\cos 2\theta$, yielding a higher-order approximation when $h$ is sufficiently smooth. In both cases, the endpoint work remains $O(1)$ per frequency because the endpoint set $E$ contains only a fixed number of indices.

A significant implementation detail is the behavior for small $|\theta|$. Expressions such as $(1-\cos\theta)/\theta^2$ are susceptible to cancellation when $|\theta|\ll 1$. In a robust implementation, these quantities should be evaluated using stable series expansions or specialized $\operatorname{sinc}$-style helper functions. This is not a cosmetic optimization. Without such care, the correction factors that are supposed to improve the quadrature can themselves become the dominant source of floating-point error.

### Rust Implementation

Following the discussion in Section 13.10.2 on interpolation-kernel corrected FFT quadrature, Program 13.10.2 provides a practical implementation of a corrected Fourier integral evaluation based on linear interpolation. While the naïve FFT approximation introduced in Section 13.10.1 relies directly on sampled data and implicitly assumes periodicity, the present program replaces the sampled function with an interpolating approximation whose oscillatory integral can be evaluated analytically. This approach introduces attenuation factors and endpoint corrections that compensate for the mismatch between the continuous integrand and its discrete representation. The program demonstrates how these corrections significantly improve numerical fidelity, particularly for nonperiodic functions, while preserving the computational efficiency of FFT-based evaluation.

At the core of the implementation is the function `corrected_fft_linear`, which realizes the corrected quadrature rule described in Equation (13.10.9). The function begins by constructing a uniform grid as defined in Equation (13.10.5) and sampling the function values $h_j$. Unlike the naïve trapezoidal rule in Equation (13.10.6), these samples are interpreted as coefficients of an interpolating function based on a compactly supported kernel $\Psi$. The oscillatory interior sum $\sum_{j=0}^{N-1} h_j e^{ij\theta}$ is then evaluated efficiently using an FFT, consistent with the matrix formulation in Equations (13.10.2)–(13.10.3).

The attenuation factor $W(\theta)$, implemented in the function `w_trap`, corresponds to the Fourier transform of the hat-function interpolation kernel given in Equation (13.10.12). Because expressions of the form $(1-\cos\theta)/\theta^2$ suffer from cancellation when $|\theta|$ is small, the implementation includes a series expansion for numerical stability. This ensures that the correction factor behaves smoothly near the origin and avoids loss of significance.

The endpoint correction is handled through the function `alpha0`, which is derived from the analytic expression in Equation (13.10.11). The auxiliary function `left_endpoint_factor` computes the integral of the local interpolation stencil, again using a stable series expansion for small $\theta$. These endpoint terms account for the fact that interpolation near the boundaries requires special treatment, and they remove the implicit periodic extension that would otherwise distort the approximation.

For comparison, the program also includes the function `naive_fft_trapezoidal`, which implements the basic FFT-based approximation of Equation (13.10.6). This allows the corrected and uncorrected methods to be evaluated side by side. The function `fft_bulk_sum` computes the oscillatory sum common to both methods, ensuring that the difference between the two approaches lies solely in the correction factors rather than the transform itself.

To validate the implementation, the program defines two test functions: `h_periodic`, which satisfies $h(a)=h(b)$, and `h_nonperiodic`, which includes an additional quadratic term and therefore violates the periodicity assumption. Exact integrals are computed using closed-form expressions in the functions `exact_periodic_integral` and `exact_nonperiodic_integral`, enabling direct error analysis. The functions `max_abs_error` and `print_selected_samples` provide global and local diagnostics, respectively.

The `main` function coordinates the experiment. It constructs the sampling grid, evaluates both the naïve and corrected approximations, and compares them against exact values across a range of frequencies. The output includes error summaries and representative frequency samples, illustrating how the corrected method reduces endpoint-induced errors while maintaining FFT efficiency.

```rust
// Program 13.10.2. Interpolation-Kernel Corrected FFT Quadrature
//
// Problem statement:
// Evaluate the Fourier-type integral
//
//     I(ω) = ∫_a^b h(t) e^{iωt} dt
//
// on an FFT-compatible frequency grid using the corrected quadrature strategy
// described in Section 13.10.2. The implementation uses linear interpolation,
// so the kernel Ψ is the hat function and the attenuation factor is
//
//     W_trap(θ) = 2(1 - cos θ) / θ^2,
//
// with stable small-θ evaluation. The corrected rule has the form
//
//     I(ω) ≈ Δ e^{iωa} [ W(θ) Σ_{j=0}^{N-1} h_j e^{ijθ}
//                        + α_0(θ) h_0
//                        + e^{iω(b-a)} α_0(θ)^* h_M ],
//
// where, for linear interpolation, the endpoint set is E = {0}. The program
// computes both the naïve FFT-grid trapezoidal approximation and the corrected
// linear-interpolation quadrature, then compares both against exact integrals
// for a periodic and a nonperiodic test function.
//
// This revised version avoids a specially matched linear test term that made the
// corrected rule look artificially exact on many FFT-grid frequencies. The
// nonperiodic test now includes a quadratic term, so the corrected quadrature
// remains strongly improved but no longer collapses to near-roundoff accuracy
// for most frequencies.

use std::f64::consts::PI;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

// =========================
// Minimal Complex Arithmetic
// =========================

#[derive(Clone, Copy, Debug, Default)]
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

    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Div<f64> for Complex {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

// ==================================
// Iterative Radix-2 FFT, Out of Box
// ==================================

fn bit_reverse_permute(data: &mut [Complex]) {
    let n = data.len();
    let mut j = 0usize;

    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

fn fft_in_place(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of two.");

    bit_reverse_permute(data);

    let mut len = 2usize;
    while len <= n {
        let angle = if inverse {
            2.0 * PI / len as f64
        } else {
            -2.0 * PI / len as f64
        };
        let wlen = Complex::from_polar(1.0, angle);

        let half = len / 2;
        let mut start = 0usize;
        while start < n {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half {
                let u = data[start + j];
                let v = data[start + j + half] * w;
                data[start + j] = u + v;
                data[start + j + half] = u - v;
                w = w * wlen;
            }
            start += len;
        }

        len <<= 1;
    }

    if inverse {
        for z in data.iter_mut() {
            *z = *z / n as f64;
        }
    }
}

// ====================================================
// Exact Integrals for Demonstration Functions on [a,b]
// ====================================================

fn exp_integral(alpha: f64, a: f64, b: f64) -> Complex {
    // ∫_a^b e^{i alpha t} dt
    if alpha.abs() < 1.0e-14 {
        Complex::new(b - a, 0.0)
    } else {
        let eb = Complex::from_polar(1.0, alpha * b);
        let ea = Complex::from_polar(1.0, alpha * a);
        let num = eb - ea;
        Complex::new(num.im / alpha, -num.re / alpha)
    }
}

fn t2_exp_integral(omega: f64, a: f64, b: f64) -> Complex {
    // ∫_a^b t^2 e^{iωt} dt
    if omega.abs() < 1.0e-14 {
        Complex::new((b * b * b - a * a * a) / 3.0, 0.0)
    } else {
        let eb = Complex::from_polar(1.0, omega * b);
        let ea = Complex::from_polar(1.0, omega * a);

        let fb = eb * Complex::new(2.0 * b / (omega * omega), -b * b / omega + 2.0 / (omega * omega * omega));
        let fa = ea * Complex::new(2.0 * a / (omega * omega), -a * a / omega + 2.0 / (omega * omega * omega));

        fb - fa
    }
}

fn exact_periodic_integral(omega: f64, a: f64, b: f64) -> Complex {
    // h_periodic(t) = sin(2πt) + 0.5 cos(4πt)
    let beta1 = 2.0 * PI;
    let beta2 = 4.0 * PI;

    let sin_part = {
        let p = exp_integral(omega + beta1, a, b);
        let m = exp_integral(omega - beta1, a, b);
        Complex::new((p.im - m.im) * 0.5, -(p.re - m.re) * 0.5)
    };

    let cos_part = {
        let p = exp_integral(omega + beta2, a, b);
        let m = exp_integral(omega - beta2, a, b);
        (p + m) * 0.25
    };

    sin_part + cos_part
}

fn exact_nonperiodic_integral(omega: f64, a: f64, b: f64) -> Complex {
    // h_nonperiodic(t) = sin(2πt) + 0.5 cos(4πt) + 0.3 t^2
    exact_periodic_integral(omega, a, b) + t2_exp_integral(omega, a, b) * 0.3
}

// =====================
// Test Functions h(t)
// =====================

fn h_periodic(t: f64) -> f64 {
    (2.0 * PI * t).sin() + 0.5 * (4.0 * PI * t).cos()
}

fn h_nonperiodic(t: f64) -> f64 {
    h_periodic(t) + 0.3 * t * t
}

// ================================================
// Stable Kernel Factors for Linear Interpolation
// ================================================

fn w_trap(theta: f64) -> f64 {
    let x = theta.abs();

    if x < 1.0e-4 {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        1.0 - t2 / 12.0 + t4 / 360.0 - t6 / 20160.0
    } else {
        2.0 * (1.0 - theta.cos()) / (theta * theta)
    }
}

fn left_endpoint_factor(theta: f64) -> Complex {
    // L(θ) = ∫_0^1 (1 - s) e^{iθs} ds
    //      = (iθ - e^{iθ} + 1) / θ^2, with L(0) = 1/2.
    let x = theta.abs();

    if x < 1.0e-4 {
        let mut sum = Complex::zero();
        let mut factorial = 1.0;
        let mut i_theta_pow = Complex::new(1.0, 0.0);
        let i_theta = Complex::new(0.0, theta);

        for n in 0..10 {
            if n > 0 {
                factorial *= n as f64;
                i_theta_pow = i_theta_pow * i_theta;
            }
            let denom = factorial * (n as f64 + 1.0) * (n as f64 + 2.0);
            sum += i_theta_pow / denom;
        }
        sum
    } else {
        let c = theta.cos();
        let s = theta.sin();
        Complex::new((1.0 - c) / (theta * theta), (theta - s) / (theta * theta))
    }
}

fn alpha0(theta: f64) -> Complex {
    left_endpoint_factor(theta) - Complex::new(w_trap(theta), 0.0)
}

// =======================================================
// FFT-compatible bulk oscillatory sum Σ h_j e^{ijθ_n}
// =======================================================

fn fft_bulk_sum(samples: &[f64], nfft: usize) -> Vec<Complex> {
    assert!(nfft.is_power_of_two(), "FFT length must be a power of two.");
    assert!(
        nfft >= samples.len(),
        "FFT length must be at least the number of samples."
    );

    let mut data = vec![Complex::zero(); nfft];
    for (j, &value) in samples.iter().enumerate() {
        data[j] = Complex::new(value, 0.0);
    }

    // Need Σ x_j e^{+i 2π n j / N}, so use the inverse-sign transform
    // and then undo its 1/N normalization.
    fft_in_place(&mut data, true);
    for z in data.iter_mut() {
        *z = *z * nfft as f64;
    }

    data
}

// ===========================================================
// Naïve FFT-grid trapezoidal approximation
// ===========================================================

fn naive_fft_trapezoidal<F>(
    h: F,
    a: f64,
    b: f64,
    m: usize,
    nfft: usize,
) -> Vec<(usize, f64, Complex)>
where
    F: Fn(f64) -> f64,
{
    assert!(nfft.is_power_of_two(), "FFT length must be a power of two.");
    assert!(nfft >= m + 1, "FFT length must satisfy N >= M + 1.");

    let delta = (b - a) / m as f64;
    let mut samples = vec![0.0_f64; m + 1];
    for j in 0..=m {
        let t_j = a + j as f64 * delta;
        let weight = if j == 0 || j == m { 0.5 } else { 1.0 };
        samples[j] = weight * h(t_j);
    }

    let bulk = fft_bulk_sum(&samples, nfft);

    let mut out = Vec::with_capacity(nfft / 2 + 1);
    for n in 0..=nfft / 2 {
        let theta = 2.0 * PI * n as f64 / nfft as f64;
        let omega = theta / delta;
        let phase_a = Complex::from_polar(1.0, omega * a);
        out.push((n, omega, phase_a * bulk[n] * delta));
    }

    out
}

// ===========================================================
// Corrected FFT quadrature for linear interpolation (hat kernel)
// ===========================================================

fn corrected_fft_linear<F>(
    h: F,
    a: f64,
    b: f64,
    m: usize,
    nfft: usize,
) -> Vec<(usize, f64, Complex)>
where
    F: Fn(f64) -> f64,
{
    assert!(nfft.is_power_of_two(), "FFT length must be a power of two.");
    assert!(nfft >= m + 1, "FFT length must satisfy N >= M + 1.");

    let delta = (b - a) / m as f64;

    let mut samples = vec![0.0_f64; m + 1];
    for j in 0..=m {
        let t_j = a + j as f64 * delta;
        samples[j] = h(t_j);
    }

    let h0 = samples[0];
    let hm = samples[m];
    let bulk = fft_bulk_sum(&samples, nfft);

    let mut out = Vec::with_capacity(nfft / 2 + 1);
    for n in 0..=nfft / 2 {
        let theta = 2.0 * PI * n as f64 / nfft as f64;
        let omega = theta / delta;

        let w = w_trap(theta);
        let a0 = alpha0(theta);

        let phase_a = Complex::from_polar(1.0, omega * a);
        let phase_span = Complex::from_polar(1.0, omega * (b - a));

        let corrected = Complex::new(w, 0.0) * bulk[n]
            + a0 * h0
            + phase_span * a0.conj() * hm;

        out.push((n, omega, phase_a * corrected * delta));
    }

    out
}

// =====================
// Error and Diagnostics
// =====================

fn max_abs_error(
    numerical: &[(usize, f64, Complex)],
    exact: fn(f64, f64, f64) -> Complex,
    a: f64,
    b: f64,
) -> (usize, f64, f64) {
    let mut best_n = 0usize;
    let mut best_omega = 0.0;
    let mut best_err = -1.0_f64;

    for &(n, omega, approx) in numerical {
        let err = (approx - exact(omega, a, b)).abs();
        if err > best_err {
            best_err = err;
            best_n = n;
            best_omega = omega;
        }
    }

    (best_n, best_omega, best_err)
}

fn print_selected_samples(
    title: &str,
    numerical: &[(usize, f64, Complex)],
    exact: fn(f64, f64, f64) -> Complex,
    a: f64,
    b: f64,
    indices: &[usize],
) {
    println!();
    println!("{title}");
    println!("{}", "-".repeat(title.len()));
    println!(
        "{:>6} {:>14} {:>18} {:>18} {:>14}",
        "n", "omega_n", "|I_num|", "|I_exact|", "abs error"
    );

    for &idx in indices {
        if idx < numerical.len() {
            let (n, omega, approx) = numerical[idx];
            let truth = exact(omega, a, b);
            let err = (approx - truth).abs();
            println!(
                "{:>6} {:>14.6} {:>18.10e} {:>18.10e} {:>14.6e}",
                n,
                omega,
                approx.abs(),
                truth.abs(),
                err
            );
        }
    }
}

fn report_error_summary(
    label: &str,
    numerical: &[(usize, f64, Complex)],
    exact: fn(f64, f64, f64) -> Complex,
    a: f64,
    b: f64,
) {
    let (n, omega, err) = max_abs_error(numerical, exact, a, b);
    println!(
        "{:<28} max abs error = {:>.10e} at n = {:>3}, omega = {:.6}",
        label, err, n, omega
    );
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let m = 256usize;
    let nfft = 512usize;

    let delta = (b - a) / m as f64;
    let nyquist = PI / delta;

    println!("Interpolation-Kernel Corrected FFT Quadrature");
    println!("=============================================");
    println!("Interval [a,b]            = [{:.6}, {:.6}]", a, b);
    println!("Number of subintervals M  = {}", m);
    println!("Grid spacing Δ            = {:.10}", delta);
    println!("FFT length N              = {}", nfft);
    println!("Nyquist frequency π/Δ     = {:.6}", nyquist);
    println!("Kernel                    = linear hat function");
    println!("Endpoint set E            = {{0}}");

    println!();
    println!("Endpoint Behavior");
    println!("=================");
    println!(
        "Periodic test:    h(a) = {:>.10}, h(b) = {:>.10}",
        h_periodic(a),
        h_periodic(b)
    );
    println!(
        "Nonperiodic test: h(a) = {:>.10}, h(b) = {:>.10}",
        h_nonperiodic(a),
        h_nonperiodic(b)
    );

    let periodic_naive = naive_fft_trapezoidal(h_periodic, a, b, m, nfft);
    let nonperiodic_naive = naive_fft_trapezoidal(h_nonperiodic, a, b, m, nfft);

    let periodic_corr = corrected_fft_linear(h_periodic, a, b, m, nfft);
    let nonperiodic_corr = corrected_fft_linear(h_nonperiodic, a, b, m, nfft);

    println!();
    println!("Global Error Summary");
    println!("====================");
    report_error_summary(
        "Periodic, naive",
        &periodic_naive,
        exact_periodic_integral,
        a,
        b,
    );
    report_error_summary(
        "Periodic, corrected",
        &periodic_corr,
        exact_periodic_integral,
        a,
        b,
    );
    report_error_summary(
        "Nonperiodic, naive",
        &nonperiodic_naive,
        exact_nonperiodic_integral,
        a,
        b,
    );
    report_error_summary(
        "Nonperiodic, corrected",
        &nonperiodic_corr,
        exact_nonperiodic_integral,
        a,
        b,
    );

    let sample_indices = [0usize, 1, 2, 3, 4, 8, 16, 32, 64, 96, 128, 192, 256];

    print_selected_samples(
        "Periodic Test, Naive FFT Trapezoidal Rule",
        &periodic_naive,
        exact_periodic_integral,
        a,
        b,
        &sample_indices,
    );

    print_selected_samples(
        "Periodic Test, Corrected Linear-Interpolation Rule",
        &periodic_corr,
        exact_periodic_integral,
        a,
        b,
        &sample_indices,
    );

    print_selected_samples(
        "Nonperiodic Test, Naive FFT Trapezoidal Rule",
        &nonperiodic_naive,
        exact_nonperiodic_integral,
        a,
        b,
        &sample_indices,
    );

    print_selected_samples(
        "Nonperiodic Test, Corrected Linear-Interpolation Rule",
        &nonperiodic_corr,
        exact_nonperiodic_integral,
        a,
        b,
        &sample_indices,
    );

    println!();
    println!("Implementation Notes");
    println!("====================");
    println!("1. The bulk oscillatory sum is computed by an FFT, while the attenuation factor W(θ) and endpoint term α_0(θ) supply the quadrature correction.");
    println!("2. For linear interpolation, the attenuation factor is W_trap(θ) = 2(1 - cos θ) / θ^2, evaluated by a stable series when |θ| is small.");
    println!("3. The endpoint set is fixed and small, so the correction cost is O(1) per frequency even though the bulk sum is evaluated on an FFT grid.");
    println!("4. The corrected rule integrates the chosen interpolant rather than trusting the bare periodic interpretation of the FFT samples.");
    println!("5. The nonperiodic test includes a quadratic term, so the corrected rule is strongly improved but is no longer artificially exact across many FFT-grid frequencies.");
}
```

Program 13.10.2 demonstrates how interpolation-kernel corrections transform the FFT from a purely algebraic tool into a reliable quadrature method for Fourier integrals. By integrating an interpolating approximation rather than the raw sampled data, the corrected rule eliminates the implicit periodicity assumption that limits the naïve approach. The numerical results show that this leads to a substantial reduction in error for nonperiodic functions, particularly at higher frequencies where endpoint artifacts are most pronounced.

At the same time, the results illustrate an important conceptual point: the corrected method integrates the chosen interpolant, not the original function values directly. As a result, improvements are not necessarily uniform across all frequencies, and in some cases the naïve rule may appear locally more accurate. The strength of the corrected approach lies in its global behavior and its consistency with the underlying continuous operator.

The modular structure of the implementation separates the FFT-based bulk computation from the kernel and endpoint corrections, making it straightforward to extend the method to higher-order interpolation kernels or alternative quadrature strategies. This provides a natural bridge to more advanced techniques, including higher-order Filon-type methods and nonuniform FFT-based quadrature schemes, which further enhance accuracy while retaining computational efficiency.

## 13.10.3. Resolution, Oversampling, and Nonuniform Frequency Evaluation

The accuracy of corrected FFT quadrature depends on two distinct resolution parameters. The first is the time-domain grid spacing $\Delta$, which controls how accurately the envelope $h$ is represented. The second is the frequency-domain sampling density, controlled by the FFT length $N$, which determines how finely $I(\omega)$ is sampled and how sharply narrow spectral features can be resolved.

The standard practical device is zero padding. Padding increases the density of the frequency grid without changing the underlying time-domain resolution, so it improves interpolation in frequency space and often helps with peak localization. This remains the default oversampling mechanism in many FFT-based codes. At the same time, recent work has explored direct FFT oversampling without zero padding, motivated by the desire to reduce redundant computation while still refining the sampled frequency grid or improving localization accuracy (Donciu, Temneanu and Serea, 2025). The broader lesson is that oversampling is not a single fixed trick. It is a numerical design choice tied to the intended balance among time resolution, frequency resolution, and computational cost.

Equation (13.10.9) naturally produces values on the uniform FFT grid. In many applications, however, one needs the transform at nonuniform or application-specific frequencies $\omega_k$. The core computational object is then the nonuniform exponential sum:

$$S(\omega_k) = \sum_{j=0}^{M} h_j e^{i\omega_k t_j} \tag{13.10.13}$$

Two modern strategies are especially important in this setting. The first uses chirp-based methods. When the phase has suitable structure, such as quadratic chirp form, one can rewrite the problem in terms of FFTs combined with pointwise chirp multiplications. Recent work develops fast algorithms for nonuniform chirp-Fourier transforms with explicit approximation-error and complexity analysis (Sun and Qian, 2024). The second uses the nonuniform FFT. A NUFFT approximates sums such as (13.10.13) to high accuracy in roughly,

$$\mathcal{O}\!\big((M+K)\log M\big) \tag{13.10.14}$$

time for $K$ target frequencies, using windowed gridding followed by an FFT. A recent finance-oriented example shows how a Fourier-cosine option-pricing formula can be reorganized so that strike-dependent evaluation becomes a type-2 NUFFT, yielding very high throughput when many strikes must be priced simultaneously (Le Floc’h, 2025).

From an implementation perspective, the important abstraction is to keep the corrected-quadrature logic separate from the engine that computes the exponential sum. For uniform frequencies, the engine is an FFT. For nonuniform frequencies, it is a NUFFT-like stage. For chirped phases, it is an FFT-chirp composition. This separation keeps the numerical approximation logic transparent and confines performance-specific changes to one well-defined component.

### Rust Implementation

Following the discussion in Section 13.10.3 on resolution, oversampling, and nonuniform frequency evaluation, Program 13.10.3 provides a practical implementation of corrected FFT-based quadrature for Fourier integrals. In numerical computation, the resolution of a spectral representation depends not only on the discretization of the underlying function but also on how the frequency domain is sampled and refined. This program illustrates how zero padding can be used to increase the density of the FFT-generated frequency grid without altering the time-domain resolution, and how nonuniform frequency evaluation can be employed to further refine localized spectral features. The implementation emphasizes a modular design in which the corrected-quadrature formulation is separated from the computational engine used to evaluate oscillatory exponential sums, enabling flexibility across different frequency-evaluation strategies.

At the core of the implementation is the separation between the corrected-quadrature formulation and the computational engine used to evaluate the oscillatory exponential sum. The function `corrected_from_bulk` implements the corrected quadrature rule derived from Equation (13.10.9), combining the FFT-based interior contribution with attenuation factors and endpoint corrections. This function represents the numerical approximation itself and remains independent of the method used to compute the underlying exponential sum.

For uniform frequency grids, the function `corrected_fft_grid` evaluates the oscillatory sum using an FFT. The FFT length $N$ determines the spacing of the frequency samples, and increasing $N$ through zero padding refines the frequency grid without modifying the time-domain spacing $\Delta$. This reflects the distinction between time-domain resolution and frequency-domain sampling density discussed in Section 13.10.3. The FFT-based computation provides an efficient way to evaluate the corrected quadrature on a uniform grid while preserving the structure of the approximation.

For nonuniform frequency evaluation, the function `corrected_nonuniform` replaces the FFT-based computation with a direct evaluation of the exponential sum in Equation (13.10.13). Although this approach has higher computational cost, it provides a clear conceptual framework for evaluating the Fourier integral at arbitrary frequencies. In practice, this direct method serves as a stand-in for more advanced techniques such as nonuniform FFT or chirp-based algorithms. Crucially, the corrected-quadrature wrapper remains unchanged, demonstrating that the numerical approximation can be reused across different computational engines.

The program computes the corrected spectrum on both a coarse FFT grid and a zero-padded, oversampled grid. It then performs localized refinement by evaluating the spectrum on a dense set of nonuniform frequencies near selected spectral features. The peak-detection logic identifies prominent features within a specified frequency band and extracts local neighborhoods around them. This allows the program to demonstrate how increasing the frequency-grid density and applying targeted nonuniform refinement improves the localization of spectral features without altering the underlying approximation.

```rust
// Program 13.10.3. Corrected FFT Quadrature with Zero-Padded Oversampling
// and Nonuniform Frequency Refinement
//
// Problem statement:
// This program demonstrates three implementation ideas from Section 13.10.3:
//
// 1. corrected FFT quadrature depends on two distinct resolution parameters:
//    the time-domain spacing Δ and the frequency-grid density controlled by N,
// 2. zero padding refines the frequency sampling density without changing Δ,
// 3. the corrected-quadrature logic can be kept separate from the engine that
//    computes the oscillatory exponential sum.
//
// The code uses the linear-interpolation corrected quadrature from Section 13.10.2.
// For uniform FFT-grid frequencies, the oscillatory bulk sum is computed by an FFT.
// For nonuniform target frequencies, the same corrected-quadrature wrapper is reused,
// but the bulk sum is computed by a direct nonuniform exponential-sum engine.
//
// This final version is based only on the numerically well-supported part of the
// demonstration: coarse-grid sampling, zero-padded oversampling, and nonuniform
// local refinement of spectral features. It deliberately avoids interpreting the
// located peaks as exact recovery of prescribed carrier frequencies.

use std::f64::consts::PI;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

// =========================
// Minimal Complex Arithmetic
// =========================

#[derive(Clone, Copy, Debug, Default)]
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

    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Div<f64> for Complex {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

// ======================
// FFT Infrastructure
// ======================

fn bit_reverse_permute(data: &mut [Complex]) {
    let n = data.len();
    let mut j = 0usize;

    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

fn fft_in_place(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of two.");

    bit_reverse_permute(data);

    let mut len = 2usize;
    while len <= n {
        let angle = if inverse {
            2.0 * PI / len as f64
        } else {
            -2.0 * PI / len as f64
        };
        let wlen = Complex::from_polar(1.0, angle);

        let half = len / 2;
        let mut start = 0usize;
        while start < n {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half {
                let u = data[start + j];
                let v = data[start + j + half] * w;
                data[start + j] = u + v;
                data[start + j + half] = u - v;
                w = w * wlen;
            }
            start += len;
        }

        len <<= 1;
    }

    if inverse {
        for z in data.iter_mut() {
            *z = *z / n as f64;
        }
    }
}

// ================================================
// Stable Kernel Factors for Linear Interpolation
// ================================================

fn w_trap(theta: f64) -> f64 {
    let x = theta.abs();

    if x < 1.0e-4 {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        1.0 - t2 / 12.0 + t4 / 360.0 - t6 / 20160.0
    } else {
        2.0 * (1.0 - theta.cos()) / (theta * theta)
    }
}

fn left_endpoint_factor(theta: f64) -> Complex {
    // L(θ) = ∫_0^1 (1 - s) e^{iθs} ds
    let x = theta.abs();

    if x < 1.0e-4 {
        let mut sum = Complex::zero();
        let mut factorial = 1.0;
        let mut i_theta_pow = Complex::new(1.0, 0.0);
        let i_theta = Complex::new(0.0, theta);

        for n in 0..10 {
            if n > 0 {
                factorial *= n as f64;
                i_theta_pow = i_theta_pow * i_theta;
            }
            let denom = factorial * (n as f64 + 1.0) * (n as f64 + 2.0);
            sum += i_theta_pow / denom;
        }
        sum
    } else {
        let c = theta.cos();
        let s = theta.sin();
        Complex::new((1.0 - c) / (theta * theta), (theta - s) / (theta * theta))
    }
}

fn alpha0(theta: f64) -> Complex {
    left_endpoint_factor(theta) - Complex::new(w_trap(theta), 0.0)
}

// =======================================================
// Corrected-Quadrature Wrapper, Separate from Sum Engine
// =======================================================

fn corrected_from_bulk(
    bulk: Complex,
    theta: f64,
    omega: f64,
    h0: f64,
    hm: f64,
    a: f64,
    b: f64,
    delta: f64,
) -> Complex {
    let w = w_trap(theta);
    let a0 = alpha0(theta);

    let phase_a = Complex::from_polar(1.0, omega * a);
    let phase_span = Complex::from_polar(1.0, omega * (b - a));

    let corrected = Complex::new(w, 0.0) * bulk
        + a0 * h0
        + phase_span * a0.conj() * hm;

    phase_a * corrected * delta
}

// =========================================
// Engines for the Oscillatory Exponential Sum
// =========================================

fn fft_bulk_sum(samples: &[f64], nfft: usize) -> Vec<Complex> {
    assert!(nfft.is_power_of_two(), "FFT length must be a power of two.");
    assert!(
        nfft >= samples.len(),
        "FFT length must be at least the number of samples."
    );

    let mut data = vec![Complex::zero(); nfft];
    for (j, &value) in samples.iter().enumerate() {
        data[j] = Complex::new(value, 0.0);
    }

    // Need Σ x_j e^{+i 2π n j / N}, so use inverse-sign transform
    // and undo the 1/N normalization afterward.
    fft_in_place(&mut data, true);
    for z in data.iter_mut() {
        *z = *z * nfft as f64;
    }

    data
}

fn direct_nonuniform_bulk_sum(samples: &[f64], thetas: &[f64]) -> Vec<Complex> {
    let mut out = Vec::with_capacity(thetas.len());

    for &theta in thetas {
        let mut sum = Complex::zero();
        for (j, &value) in samples.iter().enumerate() {
            let phase = theta * j as f64;
            sum += Complex::from_polar(value, phase);
        }
        out.push(sum);
    }

    out
}

// ======================
// Spectrum Point Helpers
// ======================

#[derive(Clone, Debug)]
struct SpectrumPoint {
    omega: f64,
    value: Complex,
}

fn corrected_fft_grid<F>(
    h: F,
    a: f64,
    b: f64,
    m: usize,
    nfft: usize,
) -> Vec<SpectrumPoint>
where
    F: Fn(f64) -> f64,
{
    assert!(nfft.is_power_of_two(), "FFT length must be a power of two.");
    assert!(nfft >= m + 1, "FFT length must satisfy N >= M + 1.");

    let delta = (b - a) / m as f64;

    let mut samples = vec![0.0_f64; m + 1];
    for j in 0..=m {
        let t_j = a + j as f64 * delta;
        samples[j] = h(t_j);
    }

    let h0 = samples[0];
    let hm = samples[m];
    let bulk = fft_bulk_sum(&samples, nfft);

    let mut out = Vec::with_capacity(nfft / 2 + 1);
    for n in 0..=nfft / 2 {
        let theta = 2.0 * PI * n as f64 / nfft as f64;
        let omega = theta / delta;
        let value = corrected_from_bulk(bulk[n], theta, omega, h0, hm, a, b, delta);
        out.push(SpectrumPoint { omega, value });
    }

    out
}

fn corrected_nonuniform<F>(
    h: F,
    a: f64,
    b: f64,
    m: usize,
    omegas: &[f64],
) -> Vec<SpectrumPoint>
where
    F: Fn(f64) -> f64,
{
    let delta = (b - a) / m as f64;

    let mut samples = vec![0.0_f64; m + 1];
    for j in 0..=m {
        let t_j = a + j as f64 * delta;
        samples[j] = h(t_j);
    }

    let h0 = samples[0];
    let hm = samples[m];
    let thetas: Vec<f64> = omegas.iter().map(|&omega| omega * delta).collect();
    let bulk = direct_nonuniform_bulk_sum(&samples, &thetas);

    let mut out = Vec::with_capacity(omegas.len());
    for ((&omega, &theta), &sum) in omegas.iter().zip(thetas.iter()).zip(bulk.iter()) {
        let value = corrected_from_bulk(sum, theta, omega, h0, hm, a, b, delta);
        out.push(SpectrumPoint { omega, value });
    }

    out
}

// ======================
// Demonstration Function
// ======================

fn gaussian(t: f64, center: f64, sigma: f64) -> f64 {
    let z = (t - center) / sigma;
    (-0.5 * z * z).exp()
}

fn h_demo(t: f64) -> f64 {
    let window = gaussian(t, 0.5, 0.18);
    let c1 = (2.0 * PI * 18.20 * t).cos();
    let c2 = 0.80 * (2.0 * PI * 19.05 * t).cos();
    window * (c1 + c2)
}

// =====================
// Peak and Reporting
// =====================

fn magnitude(z: Complex) -> f64 {
    z.abs()
}

fn find_top_two_peaks_in_band(
    points: &[SpectrumPoint],
    omega_min: f64,
    omega_max: f64,
    min_separation: f64,
) -> Vec<(usize, f64, f64)> {
    let mut candidates = Vec::new();

    if points.len() < 3 {
        return candidates;
    }

    for i in 1..points.len() - 1 {
        let omega = points[i].omega;
        let mag = magnitude(points[i].value);

        if omega >= omega_min
            && omega <= omega_max
            && mag > magnitude(points[i - 1].value)
            && mag > magnitude(points[i + 1].value)
        {
            candidates.push((i, omega, mag));
        }
    }

    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    let mut selected: Vec<(usize, f64, f64)> = Vec::new();
    for cand in candidates {
        let too_close = selected.iter().any(|item| {
            let omega_sel = item.1;
            (cand.1 - omega_sel).abs() < min_separation
        });
        if !too_close {
            selected.push(cand);
        }
        if selected.len() == 2 {
            break;
        }
    }

    selected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    selected
}

fn print_neighborhood(title: &str, points: &[SpectrumPoint], center_idx: usize, radius: usize) {
    println!();
    println!("{title}");
    println!("{}", "-".repeat(title.len()));
    println!("{:>6} {:>14} {:>18}", "idx", "omega", "|I(omega)|");

    let start = center_idx.saturating_sub(radius);
    let end = (center_idx + radius).min(points.len().saturating_sub(1));

    for i in start..=end {
        println!(
            "{:>6} {:>14.6} {:>18.10e}",
            i,
            points[i].omega,
            magnitude(points[i].value)
        );
    }
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n as f64 - 1.0);
    (0..n).map(|i| start + i as f64 * step).collect()
}

fn report_peak_list(title: &str, peaks: &[(usize, f64, f64)]) {
    println!();
    println!("{title}");
    println!("{}", "-".repeat(title.len()));
    if peaks.is_empty() {
        println!("No peaks found in the requested band.");
    } else {
        for (rank, &(idx, omega, mag)) in peaks.iter().enumerate() {
            println!(
                "Peak {}: idx = {:>4}, omega = {:>.8}, |I| = {:>.10e}",
                rank + 1,
                idx,
                omega,
                mag
            );
        }
    }
}

fn main() {
    let a = 0.0;
    let b = 1.0;
    let m = 256usize;
    let nfft_coarse = 512usize;
    let nfft_fine = 4096usize;

    let delta = (b - a) / m as f64;
    let domega_coarse = 2.0 * PI / (nfft_coarse as f64 * delta);
    let domega_fine = 2.0 * PI / (nfft_fine as f64 * delta);

    println!("Resolution, Oversampling, and Nonuniform Frequency Evaluation");
    println!("============================================================");
    println!("Interval [a,b]            = [{:.6}, {:.6}]", a, b);
    println!("Number of subintervals M  = {}", m);
    println!("Grid spacing Δ            = {:.10}", delta);
    println!("Coarse FFT length N       = {}", nfft_coarse);
    println!("Fine FFT length N         = {}", nfft_fine);
    println!("Coarse frequency spacing  = {:.10}", domega_coarse);
    println!("Fine frequency spacing    = {:.10}", domega_fine);

    let coarse = corrected_fft_grid(h_demo, a, b, m, nfft_coarse);
    let fine = corrected_fft_grid(h_demo, a, b, m, nfft_fine);

    // Choose a generic feature band rather than claiming exact carrier recovery.
    let omega_band_min = 105.0;
    let omega_band_max = 126.0;
    let min_peak_separation = 2.0;

    let coarse_peaks =
        find_top_two_peaks_in_band(&coarse, omega_band_min, omega_band_max, min_peak_separation);
    let fine_peaks =
        find_top_two_peaks_in_band(&fine, omega_band_min, omega_band_max, min_peak_separation);

    let peak1_fine = fine_peaks
        .first()
        .expect("No first oversampled-grid peak found in the requested band.");
    let peak2_fine = fine_peaks
        .get(1)
        .expect("No second oversampled-grid peak found in the requested band.");

    let refine_half_width = 0.35;
    let refine_omegas_1 =
        linspace(peak1_fine.1 - refine_half_width, peak1_fine.1 + refine_half_width, 201);
    let refine_omegas_2 =
        linspace(peak2_fine.1 - refine_half_width, peak2_fine.1 + refine_half_width, 201);

    let refined_1 = corrected_nonuniform(h_demo, a, b, m, &refine_omegas_1);
    let refined_2 = corrected_nonuniform(h_demo, a, b, m, &refine_omegas_2);

    let ref_peak_1 = find_top_two_peaks_in_band(
        &refined_1,
        peak1_fine.1 - refine_half_width,
        peak1_fine.1 + refine_half_width,
        0.02,
    )
    .into_iter()
    .next()
    .expect("No refined peak found around the lower-frequency feature.");

    let ref_peak_2 = find_top_two_peaks_in_band(
        &refined_2,
        peak2_fine.1 - refine_half_width,
        peak2_fine.1 + refine_half_width,
        0.02,
    )
    .into_iter()
    .next()
    .expect("No refined peak found around the higher-frequency feature.");

    report_peak_list("Coarse FFT-Grid Features in the Selected Band", &coarse_peaks);
    report_peak_list("Oversampled FFT-Grid Features in the Selected Band", &fine_peaks);

    println!();
    println!("Nonuniform Peak Refinement");
    println!("==========================");
    println!(
        "Refined lower-frequency feature  : omega = {:>.8}, |I| = {:>.10e}",
        ref_peak_1.1, ref_peak_1.2
    );
    println!(
        "Refined higher-frequency feature : omega = {:>.8}, |I| = {:>.10e}",
        ref_peak_2.1, ref_peak_2.2
    );

    if let Some(&(idx, _, _)) = coarse_peaks.first() {
        print_neighborhood(
            "Coarse FFT-Grid Neighborhood Around Lower-Frequency Feature",
            &coarse,
            idx,
            3,
        );
    }
    if let Some(&(idx, _, _)) = fine_peaks.first() {
        print_neighborhood(
            "Oversampled FFT-Grid Neighborhood Around Lower-Frequency Feature",
            &fine,
            idx,
            4,
        );
    }
    print_neighborhood(
        "Nonuniform Refinement Around Lower-Frequency Feature",
        &refined_1,
        ref_peak_1.0,
        4,
    );

    if let Some(&(idx, _, _)) = coarse_peaks.get(1) {
        print_neighborhood(
            "Coarse FFT-Grid Neighborhood Around Higher-Frequency Feature",
            &coarse,
            idx,
            3,
        );
    }
    if let Some(&(idx, _, _)) = fine_peaks.get(1) {
        print_neighborhood(
            "Oversampled FFT-Grid Neighborhood Around Higher-Frequency Feature",
            &fine,
            idx,
            4,
        );
    }
    print_neighborhood(
        "Nonuniform Refinement Around Higher-Frequency Feature",
        &refined_2,
        ref_peak_2.0,
        4,
    );

    println!();
    println!("Implementation Notes");
    println!("====================");
    println!("1. The corrected-quadrature wrapper is independent of the engine that computes the exponential sum.");
    println!("2. On the uniform grid, the engine is an FFT, and zero padding refines only the frequency sampling density, not the time resolution.");
    println!("3. On the nonuniform grid, the same corrected-quadrature wrapper is reused with a direct exponential-sum engine.");
    println!("4. In a production code, that nonuniform engine would typically be replaced by a NUFFT-like stage or a chirp-based composition.");
    println!("5. This separation keeps the approximation logic transparent while confining performance-specific changes to a single computational component.");
}
```

Program 13.10.3 demonstrates a practical approach to combining corrected FFT quadrature with multiple frequency-evaluation strategies. The results illustrate how zero padding refines the frequency sampling density of the FFT without changing the time-domain discretization, and how nonuniform evaluation can further enhance the resolution of localized spectral features.

The implementation highlights an important design principle in numerical computing: the mathematical approximation and the computational engine should be treated as separate components. By isolating the corrected-quadrature formulation from the method used to compute the exponential sum, the program allows FFT-based, oversampled, and nonuniform strategies to be integrated within a single coherent framework. This modular structure provides a foundation for extending the implementation to more advanced methods, including NUFFT-based acceleration and chirp-transform techniques, while preserving the integrity of the underlying numerical approximation.

## 13.10.4. Highly Oscillatory Integrals and Filon-Type Fourier-Extension Methods

When $|\omega|$ is large, direct quadrature of (13.10.4) becomes increasingly difficult because naïve rules require mesh sizes small enough to resolve the rapid oscillations. A modern and more effective approach is to use Filon-type quadrature. The basic idea is to approximate the slowly varying envelope $h$ in a suitable finite-dimensional space and then integrate that approximation against the oscillatory exponential using analytically or semi-analytically computed moments.

A recent development in this direction constructs an equispaced-grid Filon-type quadrature using a Fourier extension approximation of the envelope (Anand and Dhiman, 2024). The Fourier extension is chosen so that the required oscillatory moments can be computed efficiently, including cases involving stationary points and certain integrable singularities, while still supporting high-order convergence analysis. For a chapter on numerical Fourier methods, the pedagogical significance is clear: the FFT remains central, but now as a subroutine that accelerates representation and moment computation. The quadrature accuracy itself comes from analytically respecting the oscillatory structure, not from the FFT alone.

This perspective also sharpens the main conceptual point of the section. FFT-based computation of Fourier integrals is not simply a matter of replacing an integral by a DFT. One must decide what function is being integrated exactly: the original continuous function, a piecewise polynomial interpolant, a Fourier extension, or some other approximation. Once that decision has been made, the FFT becomes an extremely effective computational engine for the bulk oscillatory work.

If one evaluates corrected quadrature such as (13.10.9) only on FFT-grid frequencies, the complexity is essentially one FFT plus $O(N)$ postprocessing for multiplication by attenuation factors and endpoint corrections, with $O(N)$ complex storage. If one instead evaluates the integral at arbitrary target frequencies, the FFT stage is replaced by a NUFFT or chirp-based engine, leading typically to $O((M+K)\log M)$ time with memory proportional to the data and frequency arrays, up to method-dependent constants. This division of labor, consisting of a bulk FFT or NUFFT computation together with a small number of analytic corrections, is precisely what makes these methods attractive in large-scale scientific pipelines (Sun and Qian, 2024; Le Floc’h, 2025).

From the viewpoint of numerical computing, the main lesson of this section is that Fourier integrals can indeed be computed rapidly by FFT-based methods, but only after the quadrature problem has been formulated carefully. The FFT supplies efficiency. The approximation space, correction factors, and oscillatory moment treatment supply accuracy. Both are essential.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/6E98SGsEqreVump9rzhV.5","tags":[]}

# 13.11. Wavelet Transforms

Fourier methods represent signals using globally supported basis functions. Each Fourier coefficient therefore reflects information from the entire domain. This property is advantageous for periodic smooth signals and for diagonalizing convolution operators, but it becomes less effective when signals contain localized features such as edges, transients, shocks, or bursts of activity. Wavelet transforms provide a complementary representation by using basis functions that are localized both in scale and in position. In practice this allows a signal to be decomposed into components that capture coarse structure together with progressively finer detail, making wavelets particularly effective for multiscale data analysis, compression, and sparse operator representations (Ungermann and Reichert, 2025; Harbrecht, Schwab and Zank, 2025).

A useful conceptual picture is the *multiresolution pyramid decomposition*, in which a signal is split repeatedly into a coarse approximation and detail corrections. If $x$ denotes the original signal, the decomposition can be expressed schematically as,

$$x \;\longrightarrow\; (a_1, d_1) \;\longrightarrow\; (a_2, d_2, d_1) \;\longrightarrow\; \cdots \;\longrightarrow\; (a_J, d_J, \dots, d_1) \tag{13.11.1}$$

where $a_j$ represents the approximation coefficients at level $j$ and $d_j$ the corresponding detail coefficients. Each stage captures features at a particular spatial or temporal scale.

Wavelets are therefore not merely an alternative transform. They form the basis of a structured multiscale representation in which many operators become approximately sparse. This property has important consequences in numerical computing, because sparse or compressible representations enable faster algorithms and more efficient storage. In particular, wavelet bases have been used to compress operators appearing in space–time discretizations of differential equations and to construct multilevel preconditioners with near-linear complexity (Harbrecht, Schwab and Zank, 2025).

## 13.11.1. Discrete Wavelet Transform as a Filter Bank

The discrete wavelet transform (DWT) can be interpreted as a sequence of filtering and downsampling operations. Let $x[n]$ be a discrete signal of length $N=2^J$. At the first level of decomposition one computes,

$$
\begin{aligned}
a_1[k] &= \sum_{n} h[n-2k]\,x[n], \\
d_1[k] &= \sum_{n} g[n-2k]\,x[n]
\end{aligned}
\tag{13.11.2}
$$

where $h[n]$ is a low-pass filter and $g[n]$ is a high-pass filter. The factor $2k$ indicates that every second sample is retained after convolution. Thus each stage performs convolution followed by decimation.

In matrix form this operation may be written as:

$$
\begin{pmatrix}
a_1 \\
d_1
\end{pmatrix}
=
\begin{pmatrix}
H \\
G
\end{pmatrix}
x \tag{13.11.3}
$$

where $H$ and $G$ represent banded convolution matrices whose rows correspond to the filtering operations. If periodic boundary conditions are assumed, these matrices become circulant. The decimation step corresponds to selecting every second row of the convolution result.

For orthonormal wavelets, the filters must satisfy specific constraints that ensure perfect reconstruction. A common formulation of the quadrature mirror relationship is:

$$g[n] = (-1)^n h[1-n] \tag{13.11.4}$$

together with the downsampled orthogonality condition,

$$\sum_{n} h[n]\,h[n-2k] = \delta_{k,0} \tag{13.11.5}$$

These conditions guarantee that the transform preserves energy and that the inverse transform can be obtained by applying the adjoint filter bank.

The full DWT is obtained by applying this procedure recursively to the approximation coefficients only. Thus,

$$x \rightarrow (a_1, d_1) \rightarrow (a_2, d_2, d_1) \rightarrow \cdots \rightarrow (a_J, d_J, \dots, d_1)  \tag{13.11.6}$$

Because the signal length is halved at each stage, the total computational cost is $O(N)$, which is substantially cheaper than many classical spectral transforms.

### Rust Implementation

Following the discussion in Section 13.11.1 on the discrete wavelet transform as a sequence of filtering and decimation operations, Program 13.11.1 provides a concrete implementation of the filter-bank formulation of the DWT. In the theoretical development, equations (13.11.2)–(13.11.6) describe how a signal is decomposed into approximation and detail coefficients through low-pass and high-pass filtering followed by downsampling, and how this process is applied recursively to build a multiresolution representation. This program translates that structure directly into an executable Rust implementation using the Haar wavelet, thereby making explicit how convolution, decimation, and recursive decomposition interact in practice. By incorporating both forward and inverse transforms, the implementation also demonstrates the perfect reconstruction property guaranteed by the orthonormality conditions discussed in equations (13.11.4) and (13.11.5).

At the core of the implementation is the function `dwt_one_level`, which performs a single stage of the filter-bank decomposition corresponding to equation (13.11.2). It applies the low-pass and high-pass filters to the input signal and retains every second sample, thereby producing the approximation coefficients $a_1[k]$ and detail coefficients $d_1[k]$. The use of periodic indexing ensures consistency with the circulant matrix interpretation described in equation (13.11.3), where boundary effects are handled through wrap-around rather than truncation or padding.

The recursive structure of the multiresolution decomposition described in equation (13.11.6) is implemented in the function `dwt_multilevel`. This function repeatedly applies the one-level transform to the approximation coefficients only, thereby generating a hierarchy of coefficient pairs $(a_j, d_j)$ at progressively coarser scales. The detail coefficients at each level capture localized variations at the corresponding resolution, while the approximation coefficients summarize the global structure of the signal. This recursive halving of the signal length at each stage reflects the $O(N)$ computational complexity emphasized in the section.

The inverse transformation is handled by `idwt_one_level` and `idwt_multilevel`, which together implement the synthesis filter bank. These functions reconstruct the original signal by combining approximation and detail coefficients in reverse order of decomposition. Because the Haar filters satisfy the orthonormality and quadrature mirror conditions given in equations (13.11.4) and (13.11.5), the reconstruction is exact up to floating-point roundoff. The diagnostic output in the program verifies this by comparing the original and reconstructed signals and by confirming that the total energy is preserved.

The supporting functions such as `energy`, `decomposition_energy`, and `max_abs_difference` provide quantitative validation of the transform. In particular, the equality between the input signal energy and the total energy of the wavelet coefficients demonstrates the unitary nature of the transform, while the near machine-precision reconstruction error confirms numerical stability. The `haar_filters` function defines the simplest orthonormal wavelet system, allowing the implementation to remain transparent while still capturing all essential aspects of the filter-bank formulation.

The `main` function orchestrates the entire workflow. It constructs a representative signal, performs the multilevel decomposition, prints the coefficients at each level, reconstructs the signal, and evaluates diagnostic measures. This end-to-end demonstration connects the abstract formulation of the DWT with a practical computational pipeline, illustrating how multiscale representations emerge naturally from repeated filtering and downsampling.

```rust
// Program 13.11.1: Discrete Wavelet Transform as a Filter Bank
//
// Problem statement:
// Implement the orthonormal discrete wavelet transform (DWT) in filter-bank form
// for a signal of length N = 2^J under periodic boundary conditions. The program
// computes one-level approximation and detail coefficients using low-pass and
// high-pass analysis filters, recursively applies the transform to the
// approximation coefficients, and reconstructs the original signal by the
// corresponding synthesis filter bank. The implementation uses the Haar wavelet
// to demonstrate equations (13.11.2) through (13.11.6) in a concrete and
// executable Rust program.

use std::f64::consts::SQRT_2;

#[derive(Clone, Debug)]
struct WaveletFilters {
    /// Low-pass analysis filter h[n]
    h: Vec<f64>,
    /// High-pass analysis filter g[n]
    g: Vec<f64>,
    /// Low-pass synthesis filter h_tilde[n]
    h_tilde: Vec<f64>,
    /// High-pass synthesis filter g_tilde[n]
    g_tilde: Vec<f64>,
}

#[derive(Clone, Debug)]
struct DwtLevel {
    level: usize,
    approximation: Vec<f64>,
    detail: Vec<f64>,
}

#[derive(Clone, Debug)]
struct DwtDecomposition {
    coarsest_approximation: Vec<f64>,
    details: Vec<DwtLevel>,
}

fn main() {
    // Example signal of length N = 16 = 2^4.
    // The signal combines a slowly varying component with localized changes,
    // making it suitable for illustrating multiresolution behavior.
    let signal = vec![
        1.0, 1.2, 1.1, 0.9,
        2.8, 3.0, 2.9, 3.1,
        0.2, 0.1, 0.0, 0.2,
        1.9, 2.1, 2.0, 1.8,
    ];

    println!("Discrete Wavelet Transform as a Filter Bank");
    println!("===========================================");
    println!("Signal length N = {}", signal.len());
    println!("Input signal:");
    print_vector("x", &signal);

    if !is_power_of_two(signal.len()) {
        eprintln!("Error: signal length must be a power of two.");
        return;
    }

    let filters = haar_filters();

    // Perform full multilevel decomposition.
    let decomposition = dwt_multilevel(&signal, &filters);

    println!("\nMultilevel decomposition");
    println!("========================");
    for level_data in &decomposition.details {
        println!("Level {}", level_data.level);
        print_vector("  a_j", &level_data.approximation);
        print_vector("  d_j", &level_data.detail);
    }

    println!("\nCoarsest approximation");
    println!("======================");
    print_vector("a_J", &decomposition.coarsest_approximation);

    // Reconstruct the original signal.
    let reconstructed = idwt_multilevel(&decomposition, &filters);

    println!("\nReconstruction");
    println!("==============");
    print_vector("x_reconstructed", &reconstructed);

    // Diagnostics.
    let input_energy = energy(&signal);
    let coeff_energy = decomposition_energy(&decomposition);
    let reconstruction_error = max_abs_difference(&signal, &reconstructed);

    println!("\nDiagnostics");
    println!("===========");
    println!("Input energy                = {:.12}", input_energy);
    println!("Wavelet coefficient energy  = {:.12}", coeff_energy);
    println!("Max reconstruction error    = {:.12e}", reconstruction_error);
}

/// Construct the Haar analysis and synthesis filters.
/// For the orthonormal Haar wavelet:
/// h = [1/sqrt(2), 1/sqrt(2)]
/// g = [1/sqrt(2), -1/sqrt(2)]
///
/// Because the Haar system is orthonormal, the synthesis filters coincide with
/// the analysis filters in this simple setting.
fn haar_filters() -> WaveletFilters {
    let s = 1.0 / SQRT_2;
    WaveletFilters {
        h: vec![s, s],
        g: vec![s, -s],
        h_tilde: vec![s, s],
        g_tilde: vec![s, -s],
    }
}

/// Compute one level of the forward DWT using analysis filters.
/// This implements the filter-bank form of equation (13.11.2):
///
/// a[k] = sum_n h[n - 2k] x[n]
/// d[k] = sum_n g[n - 2k] x[n]
///
/// With periodic boundary conditions, indices wrap modulo N.
fn dwt_one_level(signal: &[f64], filters: &WaveletFilters) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    assert!(n % 2 == 0, "Signal length must be even for one DWT level.");

    let half = n / 2;
    let filter_len = filters.h.len();

    let mut approximation = vec![0.0; half];
    let mut detail = vec![0.0; half];

    for k in 0..half {
        let mut a_k = 0.0;
        let mut d_k = 0.0;

        for m in 0..filter_len {
            // Periodic extension: sample index wraps modulo n.
            let idx = (2 * k + m) % n;
            a_k += filters.h[m] * signal[idx];
            d_k += filters.g[m] * signal[idx];
        }

        approximation[k] = a_k;
        detail[k] = d_k;
    }

    (approximation, detail)
}

/// Reconstruct one signal level from approximation and detail coefficients.
/// This is the adjoint synthesis filter bank for periodic boundary conditions.
/// Each coarse-scale coefficient pair contributes to two fine-scale positions.
///
/// For the Haar case, perfect reconstruction is obtained exactly up to
/// floating-point roundoff.
fn idwt_one_level(
    approximation: &[f64],
    detail: &[f64],
    filters: &WaveletFilters,
) -> Vec<f64> {
    assert_eq!(
        approximation.len(),
        detail.len(),
        "Approximation and detail lengths must match."
    );

    let half = approximation.len();
    let n = 2 * half;
    let filter_len = filters.h_tilde.len();

    let mut signal = vec![0.0; n];

    for k in 0..half {
        for m in 0..filter_len {
            let idx = (2 * k + m) % n;
            signal[idx] += filters.h_tilde[m] * approximation[k];
            signal[idx] += filters.g_tilde[m] * detail[k];
        }
    }

    signal
}

/// Perform the full multilevel DWT recursively on the approximation coefficients,
/// following the pyramid structure of equation (13.11.6).
fn dwt_multilevel(signal: &[f64], filters: &WaveletFilters) -> DwtDecomposition {
    assert!(
        is_power_of_two(signal.len()),
        "Signal length must be a power of two."
    );

    let mut current = signal.to_vec();
    let mut details = Vec::new();
    let mut level = 1;

    while current.len() > 1 {
        let (approximation, detail) = dwt_one_level(&current, filters);

        details.push(DwtLevel {
            level,
            approximation: approximation.clone(),
            detail,
        });

        current = approximation;
        level += 1;
    }

    DwtDecomposition {
        coarsest_approximation: current,
        details,
    }
}

/// Reconstruct the signal by reversing the multilevel decomposition.
/// Starting from a_J, the program successively combines with d_J, d_{J-1}, ..., d_1.
fn idwt_multilevel(decomposition: &DwtDecomposition, filters: &WaveletFilters) -> Vec<f64> {
    let mut current = decomposition.coarsest_approximation.clone();

    for level_data in decomposition.details.iter().rev() {
        current = idwt_one_level(&current, &level_data.detail, filters);
    }

    current
}

/// Compute the Euclidean energy sum_n |x[n]|^2.
fn energy(signal: &[f64]) -> f64 {
    signal.iter().map(|&x| x * x).sum()
}

/// Compute the total energy stored in the wavelet coefficients:
/// |a_J|^2 + sum_j |d_j|^2.
/// For an orthonormal transform, this should match the input energy.
fn decomposition_energy(decomposition: &DwtDecomposition) -> f64 {
    let approx_energy = energy(&decomposition.coarsest_approximation);
    let detail_energy: f64 = decomposition
        .details
        .iter()
        .map(|level_data| energy(&level_data.detail))
        .sum();

    approx_energy + detail_energy
}

/// Check if n is a power of two.
fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Compute the maximum absolute difference between two vectors.
fn max_abs_difference(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Vectors must have equal length.");
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

/// Print a vector in a compact textbook-style format.
fn print_vector(name: &str, v: &[f64]) {
    println!("{} =", name);
    for (i, value) in v.iter().enumerate() {
        println!("  [{:>2}] {:>.10}", i, value);
    }
}
```

Program 13.11.1 demonstrates a direct and computationally efficient realization of the discrete wavelet transform using the filter-bank framework. It highlights how the theoretical structure introduced in Section 13.11.1 translates into an algorithm with linear complexity, exact reconstruction, and clear multiscale interpretation. The use of the Haar wavelet emphasizes conceptual clarity, while the modular design of the implementation allows straightforward extension to more sophisticated wavelet families and boundary treatments.

The multilevel decomposition produced by the program illustrates how signals can be separated into coarse approximations and localized detail corrections, providing a foundation for applications such as compression, denoising, and sparse operator representations. This implementation therefore serves as a bridge between the mathematical formulation of wavelets and their practical role in modern numerical algorithms.

## 13.11.2 Lifting Schemes and Efficient Implementations

In practical implementations the filter-bank formulation is often replaced by the *lifting scheme,* which factors the transform into a sequence of simple prediction and update steps. Instead of computing convolutions directly, the signal samples are partitioned into even and odd subsequences. The odd samples are first predicted from neighboring even samples, producing detail coefficients. The even samples are then updated using the details to form the approximation coefficients.

The lifting formulation offers several computational advantages. First, the transform can be performed in place, reducing memory overhead. Second, the number of arithmetic operations is smaller than in the original convolution form. Third, certain wavelet families can be implemented as integer-to-integer transforms, which is crucial for lossless compression.

A practical example arises in large-scale sensing applications where wavelets are used as a preprocessing stage before predictive coding. A recent compression system for distributed acoustic sensing data employs a 5/3 biorthogonal wavelet implemented via lifting because it supports reversible integer mappings and therefore guarantees exact reconstruction in lossless compression pipelines (Seguí et al., 2025). In such pipelines the wavelet transform separates slowly varying components from high-frequency details, after which linear prediction and entropy coding operate more effectively on the resulting coefficient streams.

### Rust Implementation

Following the discussion in Section 13.11.2 on lifting schemes as an efficient alternative to filter-bank implementations, Program 13.11.2 provides a practical realization of the discrete wavelet transform using the 5/3 biorthogonal lifting formulation. In contrast to the convolution-based approach of Section 13.11.1, the lifting scheme factors the transform into a sequence of prediction and update steps applied to even and odd subsequences. This program makes explicit how the decomposition described conceptually in equations (13.11.2)–(13.11.6) can be implemented using only local arithmetic operations, thereby reducing computational cost and enabling in-place and integer-to-integer transforms. The use of the 5/3 wavelet further reflects its importance in reversible compression systems, where exact reconstruction is essential.

At the core of the implementation are the functions `forward_one_level_53` and `inverse_one_level_53`, which realize the lifting steps for a single decomposition level. The forward function first partitions the signal into even and odd subsequences and then performs a prediction step in which each odd sample is approximated using neighboring even samples. The difference between the actual odd sample and its prediction forms the detail coefficients. This corresponds to replacing the high-pass filtering stage of equation (13.11.2) with a local predictive operation. The subsequent update step modifies the even samples using neighboring detail coefficients to produce the approximation coefficients, thereby completing one stage of the transform.

The recursive multiresolution structure described in equation (13.11.6) is implemented by the function `forward_multilevel_53`, which repeatedly applies the lifting step to the approximation coefficients. As in the filter-bank formulation, the signal length is halved at each level, and the resulting hierarchy of approximation and detail coefficients captures features at progressively coarser scales. The function `inverse_multilevel_53` reverses this process by applying the inverse lifting steps in reverse order, reconstructing the signal exactly from its multilevel representation.

The lifting formulation provides several computational advantages that are evident in the implementation. The operations involve only additions, subtractions, and integer divisions, eliminating the need for explicit convolution. Because the transform is performed using integer arithmetic, the coefficients remain integers at every stage, which is crucial for applications requiring lossless compression. The use of periodic boundary conditions ensures consistency with the cyclic structure discussed in Section 13.11.1 while keeping the implementation simple and self-contained.

The `main` function demonstrates the complete workflow. It constructs an integer-valued signal, performs the multilevel lifting-based decomposition, prints the approximation and detail coefficients at each level, and reconstructs the signal using the inverse transform. The diagnostic output verifies that reconstruction is exact, confirming the reversibility of the 5/3 lifting scheme. This end-to-end example illustrates how lifting schemes translate the abstract wavelet framework into an efficient and practical computational method.

```rust
// Program 13.11.2: Lifting-Scheme Implementation of the 5/3 Wavelet Transform
//
// Problem statement:
// Implement the forward and inverse 5/3 biorthogonal wavelet transform using
// the lifting scheme. The program should split the signal into even and odd
// subsequences, apply prediction and update steps, support multilevel
// decomposition, and verify exact reconstruction for integer-valued input.
// The implementation uses periodic boundary conditions and stores each level
// as approximation and detail coefficients, illustrating how lifting replaces
// direct convolution by simple local operations.

#[derive(Clone, Debug)]
struct LiftingLevel {
    level: usize,
    approximation: Vec<i32>,
    detail: Vec<i32>,
}

#[derive(Clone, Debug)]
struct LiftingDecomposition {
    coarsest_approximation: Vec<i32>,
    details: Vec<LiftingLevel>,
}

fn main() {
    // A representative integer-valued signal. Integer data are especially
    // important for reversible transforms used in lossless compression.
    let signal: Vec<i32> = vec![
        12, 13, 15, 14, 18, 20, 19, 21,
        10,  9,  8, 10, 16, 17, 15, 14,
    ];

    println!("Lifting-Scheme 5/3 Wavelet Transform");
    println!("====================================");
    println!("Signal length N = {}", signal.len());
    println!("Input signal:");
    print_vector_i32("x", &signal);

    if !is_power_of_two(signal.len()) {
        eprintln!("Error: signal length must be a power of two.");
        return;
    }

    let decomposition = forward_multilevel_53(&signal);

    println!("\nMultilevel decomposition");
    println!("========================");
    for level_data in &decomposition.details {
        println!("Level {}", level_data.level);
        print_vector_i32("  a_j", &level_data.approximation);
        print_vector_i32("  d_j", &level_data.detail);
    }

    println!("\nCoarsest approximation");
    println!("======================");
    print_vector_i32("a_J", &decomposition.coarsest_approximation);

    let reconstructed = inverse_multilevel_53(&decomposition);

    println!("\nReconstruction");
    println!("==============");
    print_vector_i32("x_reconstructed", &reconstructed);

    println!("\nDiagnostics");
    println!("===========");
    println!(
        "Exact reconstruction achieved = {}",
        signal == reconstructed
    );
    println!(
        "Maximum absolute reconstruction error = {}",
        max_abs_difference_i32(&signal, &reconstructed)
    );
}

/// Perform one forward lifting step for the integer 5/3 wavelet transform.
///
/// The signal is first split into even and odd samples:
/// even[k] = x[2k], odd[k] = x[2k+1].
///
/// Prediction step:
/// d[k] = odd[k] - floor((even[k] + even[k+1]) / 2)
///
/// Update step:
/// a[k] = even[k] + floor((d[k-1] + d[k] + 2) / 4)
///
/// Periodic boundary conditions are used, so indices wrap around.
fn forward_one_level_53(signal: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let n = signal.len();
    assert!(n % 2 == 0, "Signal length must be even.");

    let half = n / 2;
    let mut even = vec![0_i32; half];
    let mut odd = vec![0_i32; half];

    for k in 0..half {
        even[k] = signal[2 * k];
        odd[k] = signal[2 * k + 1];
    }

    // Prediction step: compute detail coefficients from odd samples.
    let mut detail = vec![0_i32; half];
    for k in 0..half {
        let kp1 = (k + 1) % half;
        let prediction = floor_div2(even[k] + even[kp1]);
        detail[k] = odd[k] - prediction;
    }

    // Update step: compute approximation coefficients from even samples.
    let mut approximation = vec![0_i32; half];
    for k in 0..half {
        let km1 = if k == 0 { half - 1 } else { k - 1 };
        let update = floor_div4(detail[km1] + detail[k] + 2);
        approximation[k] = even[k] + update;
    }

    (approximation, detail)
}

/// Perform one inverse lifting step for the integer 5/3 wavelet transform.
///
/// Inverse update:
/// even[k] = a[k] - floor((d[k-1] + d[k] + 2) / 4)
///
/// Inverse prediction:
/// odd[k] = d[k] + floor((even[k] + even[k+1]) / 2)
///
/// The even and odd subsequences are then interleaved to reconstruct the
/// fine-scale signal.
fn inverse_one_level_53(approximation: &[i32], detail: &[i32]) -> Vec<i32> {
    assert_eq!(
        approximation.len(),
        detail.len(),
        "Approximation and detail lengths must match."
    );

    let half = approximation.len();
    let n = 2 * half;

    // Inverse update step.
    let mut even = vec![0_i32; half];
    for k in 0..half {
        let km1 = if k == 0 { half - 1 } else { k - 1 };
        let update = floor_div4(detail[km1] + detail[k] + 2);
        even[k] = approximation[k] - update;
    }

    // Inverse prediction step.
    let mut odd = vec![0_i32; half];
    for k in 0..half {
        let kp1 = (k + 1) % half;
        let prediction = floor_div2(even[k] + even[kp1]);
        odd[k] = detail[k] + prediction;
    }

    // Interleave even and odd subsequences.
    let mut signal = vec![0_i32; n];
    for k in 0..half {
        signal[2 * k] = even[k];
        signal[2 * k + 1] = odd[k];
    }

    signal
}

/// Apply the forward transform recursively to the approximation coefficients.
fn forward_multilevel_53(signal: &[i32]) -> LiftingDecomposition {
    assert!(
        is_power_of_two(signal.len()),
        "Signal length must be a power of two."
    );

    let mut current = signal.to_vec();
    let mut details = Vec::new();
    let mut level = 1;

    while current.len() > 1 {
        let (approximation, detail) = forward_one_level_53(&current);

        details.push(LiftingLevel {
            level,
            approximation: approximation.clone(),
            detail,
        });

        current = approximation;
        level += 1;
    }

    LiftingDecomposition {
        coarsest_approximation: current,
        details,
    }
}

/// Reconstruct the original signal by reversing the multilevel decomposition.
fn inverse_multilevel_53(decomposition: &LiftingDecomposition) -> Vec<i32> {
    let mut current = decomposition.coarsest_approximation.clone();

    for level_data in decomposition.details.iter().rev() {
        current = inverse_one_level_53(&current, &level_data.detail);
    }

    current
}

/// Integer division by 2 with flooring behavior.
/// For the present example the values are nonnegative in the averaging stage,
/// but this helper also behaves correctly for negative inputs.
fn floor_div2(x: i32) -> i32 {
    x.div_euclid(2)
}

/// Integer division by 4 with flooring behavior.
fn floor_div4(x: i32) -> i32 {
    x.div_euclid(4)
}

/// Check whether n is a power of two.
fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Compute the maximum absolute difference between two integer vectors.
fn max_abs_difference_i32(x: &[i32], y: &[i32]) -> i32 {
    assert_eq!(x.len(), y.len(), "Vectors must have equal length.");
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs())
        .max()
        .unwrap_or(0)
}

/// Print an integer vector in a compact textbook-style format.
fn print_vector_i32(name: &str, v: &[i32]) {
    println!("{} =", name);
    for (i, value) in v.iter().enumerate() {
        println!("  [{:>2}] {:>4}", i, value);
    }
}
```

Program 13.11.2 demonstrates how the discrete wavelet transform can be implemented efficiently using lifting schemes, replacing convolution-based filtering with simple local operations. The exact reconstruction observed in the output highlights the suitability of this approach for lossless compression, where integer-to-integer mappings are essential. The multilevel decomposition further illustrates how localized detail information is separated from coarse structure, providing a compact and interpretable representation of the signal.

The modular structure of the implementation allows straightforward extension to other lifting-based wavelet families and boundary treatments. It also provides a foundation for integrating wavelet transforms into larger pipelines, such as predictive coding and entropy compression, where the separation of scales enhances both efficiency and performance.

## 13.11.3. Continuous Wavelet Transform and FFT Acceleration

While the discrete wavelet transform provides a multiresolution representation on dyadic grids, some applications require continuous control over scale and position. The continuous wavelet transform (CWT) of a signal $x(t)$ with respect to a mother wavelet $\psi$ is defined by:

$$
W_x(a,b)
=
\frac{1}{\sqrt{|a|}}
\int_{-\infty}^{\infty}
x(t)\,\psi^{*}\!\left(\frac{t-b}{a}\right)\,dt,
\qquad a>0
\tag{13.11.7}
$$

Here $a$ represents the scale parameter and $b$ the translation parameter. Small values of $a$ capture high-frequency localized structure, whereas larger scales describe broader features.

In the Fourier domain the transform can be expressed as:

$$
W_x(a,b)
=
\frac{1}{2\pi}
\int_{-\infty}^{\infty}
X(\omega)\,\Psi^{*}(a\omega)\,e^{i\omega b}\,d\omega
\tag{13.11.8}
$$

where $X(\omega)$ and $\Psi(\omega)$ are the Fourier transforms of $x(t)$ and $\psi(t)$. This representation reveals an important computational strategy. For discrete periodic data, one may compute $X(\omega)$ using the FFT, multiply it by the scale-dependent factor $\Psi^*(a\omega)$, and then apply an inverse FFT to obtain $W_x(a,b)$ for all translations $b$ simultaneously. If the transform is evaluated at $S$ distinct scale values $a_1,\dots,a_S$, the procedure must be repeated for each scale. Since each FFT-based convolution requires $O(N\log N)$ operations for a signal of length $N$, the total computational cost becomes $O(SN\log N)$.

Modern wavelet-analysis software frequently relies on this FFT-accelerated approach, especially in applications such as atmospheric gravity-wave detection and other large-scale geophysical signal analyses (Ungermann and Reichert, 2025).

### Rust Implementation

Following the discussion in Section 13.11.3 on the continuous wavelet transform and its FFT-based acceleration, Program 13.11.3 provides a practical implementation of the computational strategy described in equations (13.11.7) and (13.11.8). In contrast to the discrete wavelet transform, which operates on dyadic scales and discrete positions, the continuous formulation allows flexible control over both scale and translation. This program realizes that flexibility by computing the Fourier transform of the signal once, applying scale-dependent modulation in the frequency domain, and then using inverse FFTs to recover the wavelet coefficients for all translations simultaneously. The implementation uses a Morlet-type wavelet in the Fourier domain, illustrating how localized oscillatory features can be detected efficiently across multiple scales.

At the core of the implementation is the FFT pipeline, which embodies the computational interpretation of equation (13.11.8). The function `fft_real` computes the discrete Fourier transform $X(\omega)$ of the input signal, while `ifft_complex` reconstructs the wavelet coefficients after frequency-domain modulation. The function `angular_frequency_grid` constructs the corresponding frequency grid consistent with periodic sampling, ensuring that the discrete frequencies align with the FFT ordering. Together, these functions provide the infrastructure required to evaluate the continuous wavelet transform efficiently for periodic discrete data.

The scale-dependent modulation is implemented in the function `morlet_fourier_factor`, which defines the Fourier transform of the mother wavelet. For each scale $a$, this function evaluates the scaled wavelet spectrum $\Psi^*(a\omega)$, which is then multiplied pointwise with the signal spectrum. This multiplication replaces the convolution in equation (13.11.7) with a frequency-domain operation, thereby reducing the computational complexity from a direct integral evaluation to an $O(N \log N)$ FFT-based procedure for each scale.

The function `cwt_fft` implements the full transform across multiple scales. It first computes the Fourier transform of the signal once, and then for each scale constructs a filtered spectrum by multiplying with the conjugated wavelet factor. An inverse FFT is then applied to obtain the coefficients $W_x(a,b)$ for all translations $b$ simultaneously. This structure reflects directly the algorithmic interpretation of equation (13.11.8) and achieves the overall complexity of $O(S N \log N)$ for $S$ scales.

The auxiliary function `make_test_signal` constructs a synthetic signal consisting of a smooth background oscillation combined with localized bursts at different times and frequencies. This signal is designed to illustrate the key advantage of the continuous wavelet transform: its ability to localize features both in scale and in position. The function `max_magnitude_location` is used to identify dominant features in the wavelet coefficient arrays, providing a compact summary of where and at which scale significant structures occur.

The `main` function integrates all components into a complete computational workflow. It generates the test signal, defines a set of scales, computes the FFT-accelerated CWT, and reports representative coefficient magnitudes and peak locations. The output demonstrates how different scales emphasize different features of the signal, with smaller scales capturing high-frequency localized bursts and larger scales revealing broader structures. This confirms the multiscale and localization properties described in Section 13.11.3.

```rust
// Program 13.11.3: Continuous Wavelet Transform with FFT Acceleration
//
// Problem statement:
// Implement an FFT-accelerated continuous wavelet transform (CWT) for a
// real-valued periodic signal sampled on a uniform grid. The program should
// compute the Fourier transform of the signal once, multiply it by the
// scale-dependent Fourier representation of a mother wavelet for each scale,
// and then apply an inverse FFT to recover the CWT coefficients for all
// translations simultaneously. The implementation uses a complex Morlet
// wavelet in the frequency domain and reports representative coefficient
// magnitudes to illustrate how localized oscillatory structure is detected
// across scales.

use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, Default)]
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

    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }
}

use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
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

impl Div<f64> for Complex {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

#[derive(Debug)]
struct CwtResult {
    scales: Vec<f64>,
    coefficients: Vec<Vec<Complex>>,
}

/// Return true when n is a power of two.
fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Reverse the lower `bits` bits of x.
fn bit_reverse(mut x: usize, bits: u32) -> usize {
    let mut y = 0usize;
    for _ in 0..bits {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}

/// Iterative radix-2 FFT. If `inverse` is true, computes the inverse transform.
fn fft_in_place(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    assert!(is_power_of_two(n), "FFT length must be a power of two.");

    let bits = n.trailing_zeros();

    for i in 0..n {
        let j = bit_reverse(i, bits);
        if j > i {
            data.swap(i, j);
        }
    }

    let mut len = 2usize;
    while len <= n {
        let angle = if inverse {
            2.0 * PI / len as f64
        } else {
            -2.0 * PI / len as f64
        };
        let wlen = Complex::from_polar(1.0, angle);

        let half = len / 2;
        let mut start = 0usize;
        while start < n {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half {
                let u = data[start + j];
                let v = data[start + j + half] * w;
                data[start + j] = u + v;
                data[start + j + half] = u - v;
                w = w * wlen;
            }
            start += len;
        }

        len *= 2;
    }

    if inverse {
        let scale = n as f64;
        for z in data.iter_mut() {
            *z = *z / scale;
        }
    }
}

/// FFT of a real-valued signal.
fn fft_real(signal: &[f64]) -> Vec<Complex> {
    let mut data: Vec<Complex> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft_in_place(&mut data, false);
    data
}

/// Inverse FFT returning a complex-valued sequence.
fn ifft_complex(spectrum: &[Complex]) -> Vec<Complex> {
    let mut data = spectrum.to_vec();
    fft_in_place(&mut data, true);
    data
}

/// Angular frequency grid for a periodic sequence sampled with spacing dt.
/// The returned ordering matches the standard FFT ordering.
fn angular_frequency_grid(n: usize, dt: f64) -> Vec<f64> {
    let mut omega = vec![0.0; n];
    let factor = 2.0 * PI / (n as f64 * dt);
    let n_half = n / 2;

    for k in 0..n {
        let k_signed = if k <= n_half {
            k as isize
        } else {
            k as isize - n as isize
        };
        omega[k] = factor * k_signed as f64;
    }

    omega
}

/// Fourier-domain Morlet wavelet factor.
/// This function returns a scale-dependent multiplier consistent with the
/// FFT-accelerated strategy implied by equation (13.11.8). The factor
/// `sqrt(scale)` reflects the usual continuous-wavelet normalization.
///
/// The wavelet is concentrated near positive frequencies around `omega0`.
fn morlet_fourier_factor(scale: f64, omega: f64, omega0: f64) -> Complex {
    if omega <= 0.0 {
        return Complex::zero();
    }

    let s_omega = scale * omega;
    let exponent = -0.5 * (s_omega - omega0) * (s_omega - omega0);
    let value = scale.sqrt() * exponent.exp();
    Complex::new(value, 0.0)
}

/// Compute the FFT-accelerated CWT for a collection of scales.
fn cwt_fft(signal: &[f64], dt: f64, scales: &[f64], omega0: f64) -> CwtResult {
    let n = signal.len();
    assert!(is_power_of_two(n), "Signal length must be a power of two.");

    let x_hat = fft_real(signal);
    let omega = angular_frequency_grid(n, dt);

    let mut coefficients = Vec::with_capacity(scales.len());

    for &scale in scales {
        let mut filtered_spectrum = vec![Complex::zero(); n];

        for k in 0..n {
            let psi_star = morlet_fourier_factor(scale, omega[k], omega0).conj();
            filtered_spectrum[k] = x_hat[k] * psi_star;
        }

        let w_ab = ifft_complex(&filtered_spectrum);
        coefficients.push(w_ab);
    }

    CwtResult {
        scales: scales.to_vec(),
        coefficients,
    }
}

/// Build a synthetic signal containing a slow oscillation plus two localized
/// oscillatory bursts. This is useful for illustrating the scale-position
/// localization of the CWT.
fn make_test_signal(n: usize, dt: f64) -> Vec<f64> {
    let mut x = vec![0.0; n];

    for j in 0..n {
        let t = j as f64 * dt;

        let background = 0.7 * (2.0 * PI * 3.0 * t).sin();

        let burst_1_envelope = (-0.5 * ((t - 0.60) / 0.08).powi(2)).exp();
        let burst_1 = 1.4 * burst_1_envelope * (2.0 * PI * 18.0 * t).cos();

        let burst_2_envelope = (-0.5 * ((t - 1.45) / 0.12).powi(2)).exp();
        let burst_2 = 1.1 * burst_2_envelope * (2.0 * PI * 9.0 * t).cos();

        x[j] = background + burst_1 + burst_2;
    }

    x
}

/// Find the index and value of the maximum magnitude entry in a row of CWT coefficients.
fn max_magnitude_location(row: &[Complex]) -> (usize, f64) {
    let mut best_index = 0usize;
    let mut best_value = 0.0f64;

    for (j, &z) in row.iter().enumerate() {
        let mag = z.abs();
        if mag > best_value {
            best_value = mag;
            best_index = j;
        }
    }

    (best_index, best_value)
}

/// Print a few samples from the signal.
fn print_signal_samples(name: &str, signal: &[f64], count: usize) {
    println!("{} =", name);
    let m = count.min(signal.len());
    for j in 0..m {
        println!("  [{:>3}] {:>.10}", j, signal[j]);
    }
    if m < signal.len() {
        println!("  ...");
    }
}

fn main() {
    let n = 256usize;
    let t_final = 2.0f64;
    let dt = t_final / n as f64;

    let signal = make_test_signal(n, dt);

    // A representative set of scales. Smaller scales emphasize higher-frequency,
    // more localized features, while larger scales emphasize broader structure.
    let scales = vec![0.03, 0.05, 0.08, 0.12, 0.18, 0.28];

    // Central frequency parameter for the Morlet wavelet.
    let omega0 = 6.0;

    println!("Continuous Wavelet Transform with FFT Acceleration");
    println!("=================================================");
    println!("Signal length N              = {}", n);
    println!("Sampling interval dt         = {:.10}", dt);
    println!("Total observation window     = {:.10}", t_final);
    println!("Number of scales S           = {}", scales.len());
    println!("Morlet central frequency w0  = {:.6}", omega0);

    println!("\nInput signal samples");
    println!("====================");
    print_signal_samples("x", &signal, 16);

    let result = cwt_fft(&signal, dt, &scales, omega0);

    println!("\nScale-by-scale coefficient summary");
    println!("==================================");
    for (i, row) in result.coefficients.iter().enumerate() {
        let (j_max, mag_max) = max_magnitude_location(row);
        let b_max = j_max as f64 * dt;

        println!(
            "Scale {:>2}: a = {:>.6}, peak at sample {:>3}, b = {:>.6}, max |W(a,b)| = {:>.10}",
            i + 1,
            result.scales[i],
            j_max,
            b_max,
            mag_max
        );
    }

    println!("\nRepresentative coefficient magnitudes");
    println!("=====================================");
    let probe_indices = [32usize, 77usize, 128usize, 186usize];
    for (i, row) in result.coefficients.iter().enumerate() {
        println!("Scale a = {:>.6}", result.scales[i]);
        for &j in &probe_indices {
            let b = j as f64 * dt;
            println!(
                "  b = {:>.6}, |W(a,b)| = {:>.10}",
                b,
                row[j].abs()
            );
        }
    }
}
```

Program 13.11.3 demonstrates how the continuous wavelet transform can be implemented efficiently using FFT-based convolution, transforming what would otherwise be a costly integral evaluation into a structured sequence of spectral operations. The results illustrate the fundamental advantage of wavelet analysis: the simultaneous resolution of scale and position, which is not achievable with purely global Fourier representations.

The modular design of the implementation allows straightforward extension to other wavelet families, alternative normalization conventions, and larger-scale applications. It also provides a foundation for advanced signal analysis tasks such as feature detection, time-frequency analysis, and multiscale modeling, where the continuous wavelet transform plays a central role.

## 13.11.4. Wavelets in Numerical Algorithms and Operator Compression

Wavelets are particularly valuable in numerical methods because they often render complex operators sparse or compressible. For example, in space–time discretizations of differential equations, nonlocal operators may appear dense when represented in standard bases. When expressed in an appropriate wavelet basis, however, many of these operators exhibit rapidly decaying off-diagonal entries and can therefore be approximated efficiently by sparse matrices.

Recent work demonstrates this principle in the construction of wavelet-compressed representations of modified Hilbert transforms arising in space–time formulations of the heat equation. By combining wavelet compression with multilevel preconditioning, the resulting solvers achieve essentially linear complexity in both work and memory with respect to the problem size (Harbrecht, Schwab and Zank, 2025).

The broader lesson for numerical computing is not that wavelets are universally superior to Fourier methods, but that the choice of basis strongly influences algorithmic efficiency. Fourier bases diagonalize convolution operators and are therefore ideal for periodic global phenomena. Wavelets, by contrast, provide localized multiscale structure and often yield sparse representations for operators with spatial or temporal localization. In many modern algorithms these two viewpoints coexist: FFTs handle global convolution structures, while wavelets capture local features and enable compression or multilevel acceleration.

In the following section we turn to another fundamental connection between continuous and discrete representations, namely the sampling theorem and its role in numerical reconstruction and approximation.

### Rust Implementation

Following the discussion in Section 13.11.4 on the role of wavelets in numerical algorithms and operator compression, Program 13.11.4 provides a concrete implementation of wavelet-based compression for a dense operator. The preceding section emphasizes that while certain operators appear dense in standard coordinate bases, they often become sparse or compressible when expressed in a suitable wavelet basis. This program demonstrates that principle computationally by constructing a localized operator, transforming it into a Haar wavelet basis, applying threshold-based compression, and evaluating the resulting approximation. In doing so, it illustrates how multiscale representations can expose hidden structure that leads to more efficient storage and computation.

At the core of the implementation is the construction of the wavelet transform matrix through the function `haar_transform_matrix`. This matrix represents the orthonormal Haar basis and is built by applying the forward transform to each canonical basis vector. The transformation of the operator into wavelet coordinates is then performed using matrix multiplications of the form $A_w = W A W^{\top}$, which corresponds to expressing the operator in the wavelet basis. Because the Haar transform is orthonormal, this change of basis preserves global quantities such as the Frobenius norm, which is verified numerically in the program output.

The compression step is implemented by the function `threshold_matrix_relative`, which removes entries whose magnitude falls below a fixed fraction of the maximum entry. This simple thresholding strategy reflects the key observation that many entries of the operator in the wavelet basis are small and contribute little to the overall action of the operator. By eliminating these entries, the program produces a sparse approximation that retains the dominant multiscale structure while significantly reducing storage requirements.

The functions `matmul` and `matvec` provide the necessary linear algebra operations for transforming the operator and applying it to vectors. The function `build_localized_operator` constructs a dense matrix with entries that decay away from the diagonal, mimicking operators arising from localized physical interactions. Such operators are particularly amenable to wavelet compression because their localized structure aligns well with the multiscale localization properties of wavelets.

The effectiveness of the compression is evaluated using both matrix-based and vector-based error measures. The function `relative_matrix_error` computes the Frobenius norm of the difference between the original and compressed operators, while `relative_vector_error` measures how accurately the compressed operator reproduces the action of the original operator on a test vector. These diagnostics demonstrate that substantial sparsification can be achieved while maintaining acceptable approximation accuracy.

The `main` function integrates all components into a complete workflow. It constructs the operator, transforms it into the wavelet basis, applies thresholding, reconstructs the compressed operator in the original basis, and evaluates its performance. The output illustrates a dramatic reduction in the number of nonzero entries alongside moderate approximation error, thereby confirming the central idea of the section: an appropriate choice of basis can transform a dense problem into a sparse one.

```rust
// Program 13.11.4: Wavelet Compression of a Dense Operator in a Haar Basis
//
// Problem statement:
// Demonstrate how a dense operator can become more compressible when represented
// in a wavelet basis. The program constructs a localized dense operator on a
// one-dimensional grid, forms an orthonormal Haar wavelet transform matrix,
// computes the wavelet-space representation A_w = W A W^T, applies threshold-
// based compression to A_w, and then compares the compressed operator with the
// original operator both in sparsity and in its action on a test vector. The
// example illustrates the central numerical idea of Section 13.11.4: a suitable
// basis change can expose approximate sparsity and thereby support more
// efficient storage and computation.

#[derive(Clone, Debug)]
struct CompressionStats {
    original_nonzeros: usize,
    compressed_nonzeros: usize,
    retained_fraction: f64,
    relative_frobenius_error: f64,
    relative_apply_error: f64,
}

fn main() {
    let n = 32usize;
    let sigma = 2.5_f64;
    let gaussian_width = 4.0_f64;
    let threshold_fraction = 0.08_f64;

    println!("Wavelet Compression of a Dense Operator");
    println!("=======================================");
    println!("Matrix dimension N              = {}", n);
    println!("Localized kernel width sigma    = {:.6}", sigma);
    println!("Gaussian width                  = {:.6}", gaussian_width);
    println!("Threshold fraction              = {:.6}", threshold_fraction);

    if !is_power_of_two(n) {
        eprintln!("Error: N must be a power of two.");
        return;
    }

    // Construct a dense operator with spatial localization.
    let a = build_localized_operator(n, sigma, gaussian_width);

    println!("\nOriginal operator summary");
    println!("=========================");
    println!("Frobenius norm of A              = {:.12}", frobenius_norm(&a));
    println!("Maximum absolute entry of A      = {:.12}", max_abs_entry(&a));

    // Construct the orthonormal Haar transform matrix W.
    let w = haar_transform_matrix(n);

    // Transform the operator into wavelet coordinates: A_w = W A W^T.
    let wt = transpose(&w);
    let aw = matmul(&matmul(&w, &a), &wt);

    println!("\nWavelet-space operator summary");
    println!("==============================");
    println!("Frobenius norm of A_w            = {:.12}", frobenius_norm(&aw));
    println!("Maximum absolute entry of A_w    = {:.12}", max_abs_entry(&aw));

    // Compress in the wavelet basis by thresholding small entries.
    let aw_compressed = threshold_matrix_relative(&aw, threshold_fraction);

    // Map the compressed operator back to physical coordinates:
    // A_comp = W^T A_w_compressed W.
    let a_compressed = matmul(&matmul(&wt, &aw_compressed), &w);

    // Test the compressed operator on a representative vector.
    let x = make_test_vector(n);
    let y_exact = matvec(&a, &x);
    let y_compressed = matvec(&a_compressed, &x);

    let stats = CompressionStats {
        original_nonzeros: count_nonzeros(&aw, 0.0),
        compressed_nonzeros: count_nonzeros(&aw_compressed, 0.0),
        retained_fraction: count_nonzeros(&aw_compressed, 0.0) as f64
            / count_nonzeros(&aw, 0.0) as f64,
        relative_frobenius_error: relative_matrix_error(&aw, &aw_compressed),
        relative_apply_error: relative_vector_error(&y_exact, &y_compressed),
    };

    println!("\nCompression statistics in the wavelet basis");
    println!("===========================================");
    println!(
        "Nonzeros before compression      = {}",
        stats.original_nonzeros
    );
    println!(
        "Nonzeros after compression       = {}",
        stats.compressed_nonzeros
    );
    println!(
        "Retained fraction                = {:.12}",
        stats.retained_fraction
    );
    println!(
        "Relative Frobenius error         = {:.12e}",
        stats.relative_frobenius_error
    );
    println!(
        "Relative operator-apply error    = {:.12e}",
        stats.relative_apply_error
    );

    println!("\nRepresentative entries of A_w");
    println!("=============================");
    print_matrix_block("A_w", &aw, 8, 8);

    println!("\nRepresentative entries of thresholded A_w");
    println!("=========================================");
    print_matrix_block("A_w_compressed", &aw_compressed, 8, 8);

    println!("\nTest vector and operator action");
    println!("===============================");
    print_vector("x", &x, 12);
    print_vector("A x", &y_exact, 12);
    print_vector("A_compressed x", &y_compressed, 12);
}

/// Build a dense operator whose entries decay away from the diagonal.
/// This mimics the kind of spatially localized kernels that often become
/// more compressible in wavelet coordinates.
fn build_localized_operator(n: usize, sigma: f64, gaussian_width: f64) -> Vec<Vec<f64>> {
    let mut a = zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            let distance = (i as isize - j as isize).abs() as f64;

            let exponential_part = (-distance / sigma).exp();
            let gaussian_part = (-0.5 * (distance / gaussian_width).powi(2)).exp();

            // Add a mild oscillatory modulation to avoid an overly trivial kernel.
            let modulation = if (i + j) % 3 == 0 { 1.10 } else { 0.95 };

            a[i][j] = modulation * (0.70 * exponential_part + 0.30 * gaussian_part);
        }
    }

    a
}

/// Construct the orthonormal Haar transform matrix W by transforming each
/// standard basis vector. If c = W x, then W is orthogonal and W^T performs
/// the inverse transform.
fn haar_transform_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut w = zeros(n, n);

    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let c = forward_haar_packed(&e);

        for i in 0..n {
            w[i][j] = c[i];
        }
    }

    w
}

/// Compute the forward orthonormal Haar transform with packed coefficient
/// ordering:
///
/// [a_J, d_J, d_{J-1}, ..., d_1]
///
/// At each level, adjacent pairs are replaced by average and difference values,
/// and the transform recurses on the approximation block only.
fn forward_haar_packed(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    assert!(is_power_of_two(n), "Length must be a power of two.");

    let mut data = x.to_vec();
    let mut temp = vec![0.0; n];
    let scale = 2.0_f64.sqrt();

    let mut length = n;
    while length > 1 {
        let half = length / 2;

        for k in 0..half {
            let x0 = data[2 * k];
            let x1 = data[2 * k + 1];
            temp[k] = (x0 + x1) / scale;
            temp[half + k] = (x0 - x1) / scale;
        }

        data[..length].copy_from_slice(&temp[..length]);
        length = half;
    }

    data
}

/// Apply a relative threshold to a matrix. Entries whose absolute value is below
/// threshold_fraction * max_abs_entry are set to zero.
fn threshold_matrix_relative(a: &[Vec<f64>], threshold_fraction: f64) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let max_entry = max_abs_entry(a);
    let tau = threshold_fraction * max_entry;

    let mut b = zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            b[i][j] = if a[i][j].abs() >= tau { a[i][j] } else { 0.0 };
        }
    }

    b
}

/// Construct a test vector containing both smooth and localized components.
fn make_test_vector(n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];

    for j in 0..n {
        let t = j as f64 / n as f64;

        let smooth = (2.0 * std::f64::consts::PI * t).sin();
        let localized = 0.9 * (-80.0 * (t - 0.68).powi(2)).exp();
        let jump = if t > 0.35 { 0.4 } else { -0.2 };

        x[j] = smooth + localized + jump;
    }

    x
}

/// Matrix-matrix multiplication.
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let p = a[0].len();
    let m = b[0].len();

    assert_eq!(b.len(), p, "Inner dimensions must match.");

    let mut c = zeros(n, m);

    for i in 0..n {
        for k in 0..p {
            let aik = a[i][k];
            if aik != 0.0 {
                for j in 0..m {
                    c[i][j] += aik * b[k][j];
                }
            }
        }
    }

    c
}

/// Matrix-vector multiplication.
fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let n = a.len();
    let m = a[0].len();
    assert_eq!(m, x.len(), "Dimensions must match.");

    let mut y = vec![0.0; n];
    for i in 0..n {
        for j in 0..m {
            y[i] += a[i][j] * x[j];
        }
    }
    y
}

/// Transpose of a matrix.
fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut at = zeros(m, n);

    for i in 0..n {
        for j in 0..m {
            at[j][i] = a[i][j];
        }
    }

    at
}

/// Relative Frobenius error ||A-B||_F / ||A||_F.
fn relative_matrix_error(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let numerator = frobenius_norm(&matrix_difference(a, b));
    let denominator = frobenius_norm(a);
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Relative Euclidean vector error ||x-y||_2 / ||x||_2.
fn relative_vector_error(x: &[f64], y: &[f64]) -> f64 {
    let numerator = vector_norm2(&vector_difference(x, y));
    let denominator = vector_norm2(x);
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Frobenius norm of a matrix.
fn frobenius_norm(a: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0;
    for row in a {
        for &value in row {
            sum += value * value;
        }
    }
    sum.sqrt()
}

/// Maximum absolute matrix entry.
fn max_abs_entry(a: &[Vec<f64>]) -> f64 {
    let mut best = 0.0;
    for row in a {
        for &value in row {
            let abs_value = value.abs();
            if abs_value > best {
                best = abs_value;
            }
        }
    }
    best
}

/// Count entries whose absolute value exceeds a tolerance.
fn count_nonzeros(a: &[Vec<f64>], tol: f64) -> usize {
    let mut count = 0usize;
    for row in a {
        for &value in row {
            if value.abs() > tol {
                count += 1;
            }
        }
    }
    count
}

/// Matrix difference A - B.
fn matrix_difference(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut c = zeros(n, m);

    for i in 0..n {
        for j in 0..m {
            c[i][j] = a[i][j] - b[i][j];
        }
    }

    c
}

/// Vector difference x - y.
fn vector_difference(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y.iter()).map(|(&a, &b)| a - b).collect()
}

/// Euclidean norm of a vector.
fn vector_norm2(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

/// Allocate a zero matrix of size n x m.
fn zeros(n: usize, m: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; m]; n]
}

/// Check whether n is a power of two.
fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Print a leading block of a matrix.
fn print_matrix_block(name: &str, a: &[Vec<f64>], rows: usize, cols: usize) {
    println!("{} =", name);
    let r = rows.min(a.len());
    let c = cols.min(a[0].len());

    for i in 0..r {
        print!("  [{:>2}] ", i);
        for j in 0..c {
            print!("{:>.8}  ", a[i][j]);
        }
        println!();
    }
    if r < a.len() || c < a[0].len() {
        println!("  ...");
    }
}

/// Print a leading block of a vector.
fn print_vector(name: &str, x: &[f64], count: usize) {
    println!("{} =", name);
    let m = count.min(x.len());
    for (i, value) in x.iter().take(m).enumerate() {
        println!("  [{:>2}] {:>.10}", i, value);
    }
    if m < x.len() {
        println!("  ...");
    }
}
```

Program 13.11.4 demonstrates how wavelet-based representations can be used to compress operators and reduce computational complexity in numerical algorithms. The example shows that even simple wavelet bases, such as the Haar basis, can reveal structure that is not apparent in the original coordinate system. This insight underlies many modern numerical techniques, including multilevel solvers and preconditioning strategies, where wavelets are used to achieve near-linear complexity.

The modular structure of the implementation allows for straightforward extensions to more advanced wavelet families, adaptive thresholding strategies, and large-scale sparse data structures. It also highlights the broader principle emphasized in this section: the efficiency of a numerical algorithm depends not only on the method used but also on the representation in which the problem is expressed.

Program 13.11.4 demonstrates a practical approach to compressing dense operators by exploiting multiscale structure through wavelet transforms. The results show that a large fraction of entries can be discarded without completely destroying the operator’s effectiveness, illustrating the balance between sparsity and accuracy that is central to modern numerical linear algebra. This example reinforces the importance of basis selection in algorithm design and provides a foundation for more advanced compression and multilevel techniques.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/4Sq3c5pRRjAesPgd7LAv.4","tags":[]}

# 13.12. Numerical Use of the Sampling Theorem

The sampling theorem provides the mathematical bridge between continuous signals and their discrete representations. In its classical form it states that a bandlimited function can be reconstructed exactly from uniformly spaced samples. In numerical computation, however, the theorem rarely appears in its idealized form. Real data are finite, noisy, and represented in finite precision. Moreover, numerical algorithms must truncate the infinite reconstruction series implied by the theorem. The practical challenge is therefore not merely to state the sampling theorem but to understand how it can be implemented reliably and efficiently in computational pipelines. Recent research emphasizes that naïve application of the classical formula can converge slowly and may even become numerically unstable in the presence of noise or truncation (Kircheis, Potts and Tasche, 2024).

## 13.12.1. Classical Sampling Identity

Let $f(t)$ be a bandlimited function whose Fourier transform vanishes outside the interval $|\omega|\le \pi/T$. If the samples,

$$f_k = f(kT), \qquad k \in \mathbb{Z} \tag{13.12.1}$$

are known, then the function can be reconstructed exactly by the Shannon sampling series,

$$
f(t) = \sum_{k \in \mathbb{Z}} f(kT)\,\mathrm{sinc}\!\left(\frac{t - kT}{T}\right), 
\qquad 
\mathrm{sinc}(x) = \frac{\sin(\pi x)}{\pi x} \tag{13.12.2}
$$

This identity can be interpreted as an operator equality: sampling followed by convolution with the sinc kernel reproduces the original function provided that the bandlimit assumption holds exactly.

From a numerical perspective, two difficulties immediately arise. First, the sinc kernel decays only as $1/x$, so the infinite series converges slowly. Second, the infinite sum must be truncated in any practical computation. When the series is truncated abruptly, the resulting approximation error can be large, particularly near discontinuities or regions of rapid variation.

These observations motivate a central principle of numerical sampling theory: exact reconstruction formulas are often not the most stable numerical algorithms. Instead, stable implementations typically modify the reconstruction kernel to improve decay and locality.

### Rust Implementation

Following the discussion in Section 13.12.1 on the classical Shannon sampling identity and its numerical limitations, Program 13.12.1 provides a practical implementation of truncated sinc-based reconstruction. While equation (13.12.2) expresses an exact recovery formula for bandlimited functions, its direct use in computation requires truncating an infinite series. This program demonstrates how such truncation is implemented in practice and how it affects numerical accuracy. By reconstructing a bandlimited test function from uniformly spaced samples and comparing the result with the exact function, the implementation illustrates both the theoretical correctness and the practical challenges associated with the classical sampling theorem.

At the core of the implementation is the function `shannon_reconstruct_truncated`, which evaluates a finite version of the reconstruction series described in equation (13.12.2). Instead of summing over all integers, the function restricts the computation to a local stencil of width $2m+1$ centered around the nearest sample index. This reflects the unavoidable truncation required in numerical computation and highlights the slow decay of the sinc kernel, which causes distant samples to still influence the reconstruction. The truncation strategy directly illustrates the convergence issues discussed in Section 13.12.1.

The function `sinc` implements the normalized sinc kernel $\mathrm{sinc}(x)=\sin(\pi x)/(\pi x)$, with special handling near the origin to avoid catastrophic cancellation. This numerical safeguard is essential for maintaining accuracy when evaluating the reconstruction formula, particularly when $x$ is close to zero. The bandlimited test function, defined in `bandlimited_test_function`, uses the same sinc profile to ensure consistency with the theoretical assumptions underlying equation (13.12.2).

The sampling process itself is implemented by `generate_samples`, which constructs the discrete sequence $f_k = f(kT)$ over a finite index range. This reflects the practical reality that only a finite number of samples are available in computation. The `main` function then evaluates the reconstructed signal over a grid of points and compares it with the exact function values, providing both pointwise errors and global diagnostics such as maximum and RMS error.

The diagnostic output highlights two key numerical properties. First, the reconstruction is exact at the sample points, confirming the interpolation property inherent in equation (13.12.2). Second, away from the sampling grid, truncation introduces small but noticeable errors due to the slow $1/x$ decay of the sinc kernel. These errors are especially visible at the boundaries of the evaluation interval, where fewer contributing samples are available within the truncation window.

```rust
// Program 13.12.1: Classical Shannon Sampling Reconstruction with Truncation
//
// Problem statement:
// Implement the classical Shannon sampling series for a bandlimited function
// sampled on a uniform grid. The program constructs samples f_k = f(kT),
// reconstructs the function at arbitrary evaluation points using a truncated
// sinc series, and compares the reconstructed values with the exact function.
// The implementation illustrates both the power and the numerical limitations
// of the classical sampling identity: exact recovery is possible in theory for
// bandlimited signals, but practical computation requires truncation of the
// infinite series, which introduces approximation error.

use std::f64::consts::PI;

fn main() {
    // Bandlimited test function:
    //
    // f(t) = sinc(t/T_band) = sin(pi t / T_band) / (pi t / T_band)
    //
    // Its Fourier transform is supported in |omega| <= pi / T_band.
    // We choose the sampling interval T <= T_band so that the sampling theorem
    // applies without aliasing.
    let t_band = 1.0_f64;
    let t_sample = 0.75_f64;

    // Finite sample range: k = -k_max, ..., k_max.
    let k_max = 20_i32;

    // Number of nearest samples used in the truncated reconstruction stencil.
    let m_trunc = 8_i32;

    // Evaluation grid for comparison.
    let n_eval = 81usize;
    let t_min = -4.0_f64;
    let t_max = 4.0_f64;

    println!("Classical Shannon Sampling Reconstruction with Truncation");
    println!("=========================================================");
    println!("Bandlimit parameter T_band      = {:.10}", t_band);
    println!("Sampling interval T             = {:.10}", t_sample);
    println!("Number of stored samples        = {}", 2 * k_max + 1);
    println!("Truncation half-width m         = {}", m_trunc);
    println!("Evaluation points              = {}", n_eval);

    // Generate uniform samples f_k = f(kT).
    let samples = generate_samples(k_max, t_sample, t_band);

    println!("\nStored samples");
    println!("==============");
    print_samples(&samples, 8);

    // Evaluate reconstruction and compare with the exact function.
    let mut max_abs_error = 0.0_f64;
    let mut rms_error_accum = 0.0_f64;

    println!("\nRepresentative reconstruction values");
    println!("====================================");
    println!(
        "{:>12} {:>18} {:>18} {:>18}",
        "t", "exact f(t)", "reconstructed", "abs error"
    );

    for j in 0..n_eval {
        let t = t_min + (t_max - t_min) * j as f64 / (n_eval as f64 - 1.0);

        let exact = bandlimited_test_function(t, t_band);
        let reconstructed = shannon_reconstruct_truncated(t, t_sample, &samples, m_trunc);
        let abs_error = (exact - reconstructed).abs();

        max_abs_error = max_abs_error.max(abs_error);
        rms_error_accum += abs_error * abs_error;

        // Print only a representative subset of rows to keep output compact.
        if j % 10 == 0 || j + 1 == n_eval {
            println!(
                "{:>12.6} {:>18.10} {:>18.10} {:>18.10e}",
                t, exact, reconstructed, abs_error
            );
        }
    }

    let rms_error = (rms_error_accum / n_eval as f64).sqrt();

    println!("\nDiagnostics");
    println!("===========");
    println!("Maximum absolute error          = {:.12e}", max_abs_error);
    println!("RMS reconstruction error        = {:.12e}", rms_error);

    // Probe the reconstruction exactly at several sample points to show the
    // interpolation property in the finite setting.
    println!("\nInterpolation check at sample points");
    println!("====================================");
    println!(
        "{:>8} {:>14} {:>18} {:>18} {:>18}",
        "k", "kT", "stored sample", "reconstructed", "abs error"
    );

    for &k in &[-4_i32, -2, 0, 2, 4] {
        let t = k as f64 * t_sample;
        let exact_sample = bandlimited_test_function(t, t_band);
        let reconstructed = shannon_reconstruct_truncated(t, t_sample, &samples, m_trunc);
        let abs_error = (exact_sample - reconstructed).abs();

        println!(
            "{:>8} {:>14.6} {:>18.10} {:>18.10} {:>18.10e}",
            k, t, exact_sample, reconstructed, abs_error
        );
    }
}

#[derive(Clone, Debug)]
struct Sample {
    k: i32,
    t: f64,
    value: f64,
}

/// Generate samples f_k = f(kT) for k = -k_max, ..., k_max.
fn generate_samples(k_max: i32, t_sample: f64, t_band: f64) -> Vec<Sample> {
    let mut samples = Vec::with_capacity((2 * k_max + 1) as usize);

    for k in -k_max..=k_max {
        let t = k as f64 * t_sample;
        let value = bandlimited_test_function(t, t_band);
        samples.push(Sample { k, t, value });
    }

    samples
}

/// Bandlimited test function used in the experiment:
///
/// f(t) = sinc(t / T_band).
///
/// Since sinc(x) has a rectangular Fourier transform support, this function is
/// bandlimited and is therefore appropriate for demonstrating the Shannon
/// reconstruction formula.
fn bandlimited_test_function(t: f64, t_band: f64) -> f64 {
    sinc(t / t_band)
}

/// Numerically stable sinc function:
///
/// sinc(x) = sin(pi x) / (pi x),
///
/// with the removable singularity at x = 0 handled explicitly.
fn sinc(x: f64) -> f64 {
    let pix = PI * x;

    if pix.abs() < 1.0e-12 {
        1.0
    } else {
        pix.sin() / pix
    }
}

/// Reconstruct f(t) using the truncated Shannon sampling series:
///
/// f(t) ≈ sum_{|k-k0|<=m} f(kT) sinc((t-kT)/T),
///
/// where k0 is the nearest sample index to t/T and m controls the stencil width.
///
/// This localized truncation mimics the unavoidable finite approximation used in
/// numerical computation.
fn shannon_reconstruct_truncated(
    t: f64,
    t_sample: f64,
    samples: &[Sample],
    m_trunc: i32,
) -> f64 {
    let k_center = (t / t_sample).round() as i32;
    let mut sum = 0.0_f64;

    for sample in samples {
        if (sample.k - k_center).abs() <= m_trunc {
            let x = (t - sample.t) / t_sample;
            sum += sample.value * sinc(x);
        }
    }

    sum
}

/// Print a few representative stored samples.
fn print_samples(samples: &[Sample], count_each_side: usize) {
    let n = samples.len();

    for (i, sample) in samples.iter().enumerate() {
        if i < count_each_side || i >= n.saturating_sub(count_each_side) {
            println!(
                "k = {:>3}, t = {:>10.6}, f_k = {:>.10}",
                sample.k, sample.t, sample.value
            );
        } else if i == count_each_side {
            println!("...");
        }
    }
}
```

Program 13.12.1 demonstrates a direct numerical implementation of the classical sampling theorem and exposes its practical limitations. While the theoretical reconstruction is exact, the truncated implementation reveals slow convergence and sensitivity to truncation width. This behavior motivates the development of regularized and windowed sampling formulas discussed in subsequent sections, where improved decay properties lead to more stable and efficient numerical algorithms.

The example also illustrates a broader principle in numerical analysis: exact analytical formulas are not always optimal for computation. Instead, practical algorithms often modify the original formulation to improve stability, locality, and efficiency. The truncated Shannon reconstruction serves as a baseline against which more advanced sampling methods can be compared.

## 13.12.2. Windowed and Regularized Sampling Formulas

A widely used strategy is to introduce a window function that regularizes the sinc kernel and localizes the reconstruction. Let $\varphi(t)$ be a smooth window that decays rapidly outside a finite interval. A regularized sampling approximation can then be written as:

$$
(R_{\varphi,m}f)(t)
= \sum_{k \in \mathbb{Z}} f(kT)\,
\mathrm{sinc}\!\left(\frac{t-kT}{T}\right)
\varphi(t-kT)\,
\mathbf{1}_{[-m,m]}(t-kT)
\tag{13.12.3}
$$

where $m$ controls the truncation width and $\mathbf{1}_{[-m,m]}$ denotes restriction to a finite stencil of neighboring samples.

This formulation has several desirable numerical properties. First, interpolation is preserved at the sampling points because,

$$\mathrm{sinc}(n-k)=\delta_{n,k}, \qquad (R_{\varphi,m}f)(nT)=f(nT) \tag{13.12.4}$$

Second, evaluating the reconstruction at a point $t$ requires only about $2m+1$ nearby samples rather than the entire infinite series. Third, when the window function is chosen appropriately and the signal is slightly oversampled relative to the Nyquist rate, the approximation error decays exponentially with (m). This allows highly accurate reconstructions using relatively small stencils (Kircheis, Potts and Tasche, 2025).

Thus the regularized sampling formula replaces the slowly decaying sinc reconstruction with a localized convolution kernel whose truncation error can be controlled systematically.

## 13.12.3. Window Families and Error Behavior

Several families of window functions have been studied for regularized sampling formulas. Among the most commonly used are Gaussian windows,

$$\varphi_{\text{Gauss}}(t) = \exp\!\left(-\frac{t^{2}}{2\sigma^{2}}\right) \tag{13.12.5}$$

sinh-type windows with a shape parameter $\beta$, and continuous Kaiser–Bessel windows that involve modified Bessel functions. These window families differ in their decay rates and in the trade-off between approximation accuracy and computational cost.

Recent analysis compares these window functions in the context of oversampled reconstruction formulas. Under suitable bandwidth conditions, sinh-type and continuous Kaiser–Bessel windows can achieve significantly faster exponential decay of truncation error with respect to the stencil size $m$ than Gaussian windows in certain regimes (Kircheis, Potts and Tasche, 2025). Such results demonstrate that the choice of window function is not merely cosmetic. It directly determines the attainable accuracy for a given stencil width and therefore influences both computational efficiency and numerical stability.

The same window families also appear in nonuniform FFT algorithms, where Kaiser–Bessel windows are widely used for gridding and deconvolution steps. This reflects a broader numerical principle: localization kernels that provide good decay properties tend to be useful across a variety of approximation and transform algorithms.

### Rust Implementation

Following the discussion in Section 13.12.2 on windowed and regularized sampling formulas, Program 13.12.2 provides a practical implementation of localized reconstruction using a Gaussian window. While the classical sampling identity in equation (13.12.2) requires summation over all samples and suffers from slow decay of the sinc kernel, the regularized formulation in equation (13.12.3) introduces a window function to improve locality and numerical stability. This program demonstrates how such regularization can be implemented efficiently using a finite stencil of neighboring samples, and how it significantly improves reconstruction accuracy compared to the classical truncated series. By evaluating both classical and windowed reconstructions for a bandlimited test function, the implementation highlights the role of windowing in controlling truncation error and improving numerical behavior.

At the core of the implementation is the function `regularized_reconstruct`, which evaluates the windowed sampling approximation described in equation (13.12.3). The reconstruction is performed using only those samples satisfying the stencil constraint $|t-kT|\le mT$, thereby reducing the infinite sum to a local computation involving approximately $2m+1$ terms. Each term combines three components: the sample value $f(kT)$, the sinc kernel, and the window function $\varphi(t-kT)$. This structure replaces the slowly decaying sinc-only reconstruction with a localized kernel whose contributions diminish rapidly away from the evaluation point.

The window function is implemented through the `WindowFunction` trait and the concrete `GaussianWindow` struct. The Gaussian window corresponds to equation (13.12.5), where the parameter $\sigma$ controls the decay rate. Because $\varphi(0)=1$, the interpolation property described in equation (13.12.4) is preserved, ensuring that reconstruction at sample points exactly reproduces the stored data. The use of a trait-based abstraction allows alternative window families, such as sinh-type or Kaiser–Bessel windows, to be incorporated without modifying the reconstruction logic.

For comparison, the function `shannon_reconstruct_truncated` implements the classical truncated sampling formula based on equation (13.12.2). This function uses the same finite stencil but does not apply any windowing, thereby exposing the limitations of pure sinc truncation. The difference between the two approaches is evident in the error behavior, with the windowed version exhibiting significantly reduced truncation error due to improved decay of the effective kernel.

The auxiliary functions `sinc` and `bandlimited_test_function` provide the numerical building blocks for evaluating the reconstruction. The `sinc` function includes special handling near the origin to avoid numerical instability, while the test function ensures consistency with the bandlimited assumption underlying the sampling theorem. The sampling process is implemented by `generate_samples`, which constructs the discrete sequence $f_k = f(kT)$ over a finite range, reflecting the practical limitation of finite data.

The `main` function orchestrates the complete workflow. It generates the sample data, performs both classical and windowed reconstructions over a dense evaluation grid, and computes diagnostic measures such as maximum and RMS error. It also verifies the interpolation property at selected sample points and prints representative values of the Gaussian window. The results clearly demonstrate that windowing reduces reconstruction error by improving localization, even when the same truncation width is used.

```rust
// Program 13.12.2: Windowed and Regularized Sampling Reconstruction
//
// Problem statement:
// Implement a regularized sampling formula of the form
//
//   (R_{phi,m} f)(t) = sum_k f(kT) sinc((t-kT)/T) phi(t-kT) 1_{[-m,m]}(t-kT),
//
// where phi is a rapidly decaying window and m controls the finite stencil.
// The program compares the classical truncated Shannon reconstruction with a
// Gaussian-windowed reconstruction for a bandlimited test function, verifies
// interpolation at sample points, and reports reconstruction errors on a dense
// evaluation grid. This demonstrates how windowing localizes the sinc kernel
// and improves numerical behavior under finite truncation.

use std::f64::consts::PI;

fn main() {
    // Oversampled bandlimited test function.
    // f(t) = sinc(t / T_band), bandlimited to |omega| <= pi / T_band.
    // The sampling interval T is chosen below T_band to satisfy oversampling.
    let t_band = 1.0_f64;
    let t_sample = 0.75_f64;

    // Finite sample range k = -k_max, ..., k_max.
    let k_max = 24_i32;

    // Truncation half-width m in physical units of sample spacing:
    // only terms with |t - kT| <= m T are used.
    let m_stencil = 8_i32;

    // Gaussian window parameter sigma = alpha * mT.
    // A larger alpha gives weaker localization; a smaller alpha gives stronger
    // localization but may overly damp useful nearby terms.
    let sigma_alpha = 0.45_f64;

    // Evaluation grid.
    let n_eval = 161usize;
    let t_min = -4.0_f64;
    let t_max = 4.0_f64;

    println!("Windowed and Regularized Sampling Reconstruction");
    println!("================================================");
    println!("Bandlimit parameter T_band      = {:.10}", t_band);
    println!("Sampling interval T             = {:.10}", t_sample);
    println!("Number of stored samples        = {}", 2 * k_max + 1);
    println!("Truncation half-width m         = {}", m_stencil);
    println!("Gaussian sigma / (mT)           = {:.10}", sigma_alpha);
    println!("Evaluation points               = {}", n_eval);

    let samples = generate_samples(k_max, t_sample, t_band);

    println!("\nStored samples");
    println!("==============");
    print_samples(&samples, 8);

    let sigma = sigma_alpha * (m_stencil as f64) * t_sample;
    let gaussian = GaussianWindow { sigma };

    let mut max_abs_error_classical = 0.0_f64;
    let mut rms_error_accum_classical = 0.0_f64;

    let mut max_abs_error_windowed = 0.0_f64;
    let mut rms_error_accum_windowed = 0.0_f64;

    println!("\nRepresentative reconstruction values");
    println!("====================================");
    println!(
        "{:>10} {:>16} {:>16} {:>16} {:>16}",
        "t", "exact", "classical", "windowed", "|err_windowed|"
    );

    for j in 0..n_eval {
        let t = t_min + (t_max - t_min) * j as f64 / (n_eval as f64 - 1.0);

        let exact = bandlimited_test_function(t, t_band);

        let classical = shannon_reconstruct_truncated(t, t_sample, &samples, m_stencil);
        let windowed =
            regularized_reconstruct(t, t_sample, &samples, m_stencil, &gaussian);

        let err_classical = (classical - exact).abs();
        let err_windowed = (windowed - exact).abs();

        max_abs_error_classical = max_abs_error_classical.max(err_classical);
        rms_error_accum_classical += err_classical * err_classical;

        max_abs_error_windowed = max_abs_error_windowed.max(err_windowed);
        rms_error_accum_windowed += err_windowed * err_windowed;

        if j % 20 == 0 || j + 1 == n_eval {
            println!(
                "{:>10.6} {:>16.10} {:>16.10} {:>16.10} {:>16.10e}",
                t, exact, classical, windowed, err_windowed
            );
        }
    }

    let rms_error_classical = (rms_error_accum_classical / n_eval as f64).sqrt();
    let rms_error_windowed = (rms_error_accum_windowed / n_eval as f64).sqrt();

    println!("\nDiagnostics");
    println!("===========");
    println!(
        "Classical max abs error         = {:.12e}",
        max_abs_error_classical
    );
    println!(
        "Classical RMS error             = {:.12e}",
        rms_error_classical
    );
    println!(
        "Windowed max abs error          = {:.12e}",
        max_abs_error_windowed
    );
    println!(
        "Windowed RMS error              = {:.12e}",
        rms_error_windowed
    );

    println!("\nInterpolation check at sample points");
    println!("====================================");
    println!(
        "{:>8} {:>12} {:>16} {:>16} {:>16}",
        "k", "kT", "stored sample", "windowed", "abs error"
    );

    for &k in &[-4_i32, -2, 0, 2, 4] {
        let t = k as f64 * t_sample;
        let exact_sample = bandlimited_test_function(t, t_band);
        let reconstructed = regularized_reconstruct(t, t_sample, &samples, m_stencil, &gaussian);
        let abs_error = (exact_sample - reconstructed).abs();

        println!(
            "{:>8} {:>12.6} {:>16.10} {:>16.10} {:>16.10e}",
            k, t, exact_sample, reconstructed, abs_error
        );
    }

    println!("\nWindow profile samples");
    println!("======================");
    println!(
        "{:>12} {:>18}",
        "tau = t-kT", "phi_Gauss(tau)"
    );
    for q in -4..=4 {
        let tau = q as f64 * t_sample;
        println!("{:>12.6} {:>18.10}", tau, gaussian.value(tau));
    }
}

#[derive(Clone, Debug)]
struct Sample {
    k: i32,
    t: f64,
    value: f64,
}

trait WindowFunction {
    fn value(&self, tau: f64) -> f64;
}

#[derive(Clone, Copy, Debug)]
struct GaussianWindow {
    sigma: f64,
}

impl WindowFunction for GaussianWindow {
    fn value(&self, tau: f64) -> f64 {
        let z = tau / self.sigma;
        (-0.5 * z * z).exp()
    }
}

/// Generate samples f_k = f(kT), k = -k_max, ..., k_max.
fn generate_samples(k_max: i32, t_sample: f64, t_band: f64) -> Vec<Sample> {
    let mut samples = Vec::with_capacity((2 * k_max + 1) as usize);

    for k in -k_max..=k_max {
        let t = k as f64 * t_sample;
        let value = bandlimited_test_function(t, t_band);
        samples.push(Sample { k, t, value });
    }

    samples
}

/// Bandlimited test function:
/// f(t) = sinc(t / T_band).
fn bandlimited_test_function(t: f64, t_band: f64) -> f64 {
    sinc(t / t_band)
}

/// Stable normalized sinc:
/// sinc(x) = sin(pi x) / (pi x).
fn sinc(x: f64) -> f64 {
    let pix = PI * x;
    if pix.abs() < 1.0e-12 {
        1.0
    } else {
        pix.sin() / pix
    }
}

/// Classical truncated Shannon reconstruction:
/// sum f(kT) sinc((t-kT)/T), restricted to |t-kT| <= mT.
fn shannon_reconstruct_truncated(
    t: f64,
    t_sample: f64,
    samples: &[Sample],
    m_stencil: i32,
) -> f64 {
    let k_center = (t / t_sample).round() as i32;
    let mut sum = 0.0_f64;

    for sample in samples {
        if (sample.k - k_center).abs() <= m_stencil {
            let x = (t - sample.t) / t_sample;
            sum += sample.value * sinc(x);
        }
    }

    sum
}

/// Regularized reconstruction:
/// sum f(kT) sinc((t-kT)/T) phi(t-kT) 1_{[-m,m]}((t-kT)/T).
///
/// Because sinc(n-k) = delta_{nk}, interpolation is preserved at sample points
/// provided phi(0) = 1. The Gaussian window used here satisfies phi(0) = 1.
fn regularized_reconstruct<W: WindowFunction>(
    t: f64,
    t_sample: f64,
    samples: &[Sample],
    m_stencil: i32,
    window: &W,
) -> f64 {
    let k_center = (t / t_sample).round() as i32;
    let mut sum = 0.0_f64;

    for sample in samples {
        if (sample.k - k_center).abs() <= m_stencil {
            let tau = t - sample.t;
            let x = tau / t_sample;
            sum += sample.value * sinc(x) * window.value(tau);
        }
    }

    sum
}

/// Print representative samples near both ends.
fn print_samples(samples: &[Sample], count_each_side: usize) {
    let n = samples.len();

    for (i, sample) in samples.iter().enumerate() {
        if i < count_each_side || i >= n.saturating_sub(count_each_side) {
            println!(
                "k = {:>3}, t = {:>10.6}, f_k = {:>.10}",
                sample.k, sample.t, sample.value
            );
        } else if i == count_each_side {
            println!("...");
        }
    }
}
```

Program 13.12.2 demonstrates how regularization transforms the classical sampling formula into a numerically stable and efficient algorithm. By introducing a window function, the reconstruction becomes localized, reducing sensitivity to truncation and enabling accurate approximations with relatively small stencils. The comparison with the classical truncated method highlights the practical importance of modifying theoretical formulas to achieve reliable numerical performance.

The modular structure of the implementation allows easy experimentation with different window functions and parameter choices. This flexibility is essential for exploring the trade-offs discussed in Section 13.12.3, where different window families offer varying decay rates and accuracy characteristics. The example also connects naturally to related techniques in numerical analysis, such as kernel-based interpolation and nonuniform FFT methods, where similar localization principles are used to achieve efficient and stable computation.

## 13.12.4. Extensions to Generalized and Nonuniform Sampling

The classical sampling theorem assumes uniform sampling and an ideal Fourier transform setting. Modern numerical work often considers more general transforms and sampling geometries. One example is the special affine Fourier transform (SAFT), which generalizes the classical Fourier transform by incorporating affine phase factors. Recent work develops regularized Shannon-type sampling formulas for this transform using compactly supported windows such as B-spline, sinh-type, and Kaiser–Bessel kernels (Filbir and Tasche, 2025).

These results illustrate an important pattern in numerical transform theory. Even when the transform kernel changes, the stabilization strategy remains similar: oversample relative to the nominal bandwidth and replace slowly decaying sinc kernels by localized windowed kernels. The same philosophy appears in many nonuniform FFT algorithms.

Nonuniform sampling introduces additional complications because the reconstruction kernels no longer correspond to simple translations of a single function. Nevertheless, recent work has developed regularized nonuniform sampling formulas that achieve improved convergence properties. For example, sinh-regularized Lagrangian sampling series have been proposed that yield faster convergence than comparable Gaussian regularizations under certain theoretical and experimental conditions (Jiang, Chen and Chen, 2026).

## 13.12.5. Numerical Implementation Considerations

From a computational standpoint, the localized reconstruction formula (13.12.3) has complexity proportional to the stencil width. Evaluating the approximation at a single point requires $O(m)$ operations because only nearby samples contribute. When many evaluation points lie on a uniform grid, the reconstruction can be interpreted as a convolution-like operation and accelerated using FFT techniques. If evaluation points are irregular, NUFFT-based strategies can be applied instead.

Two numerical details deserve careful attention in practical implementations. First, the sinc function should be evaluated using a stable branch near the origin to avoid catastrophic cancellation when $x$ is small. Second, window functions involving exponentials or special functions must be computed accurately so that the intended exponential decay of the reconstruction error is not degraded by floating-point roundoff.

Theoretical analysis also shows that the choice of window parameters has a direct impact on error decay rates. Treating these parameters as arbitrary constants can therefore lead to suboptimal performance. In carefully designed implementations, the window shape and truncation width are chosen together to balance numerical accuracy, computational cost, and stability (Kircheis, Potts and Tasche, 2025; Jiang, Chen and Chen, 2026).

## 13.12.6. Practical Applications

Sampling-based reconstruction methods appear in many numerical pipelines where continuous models must be recovered from discrete measurements. In computational finance, Fourier inversion formulas often require evaluating functions defined through characteristic functions at many grid points. These computations frequently combine Fourier integrals, FFT acceleration, and sampling-based interpolation techniques to obtain large batches of values efficiently (Le Floc’h, 2025).

Another example arises in large-scale sensing systems such as distributed acoustic sensing, where enormous spatiotemporal datasets must be compressed and processed efficiently. In such systems wavelet transforms and predictive models are used to capture multiscale structure, while sampling and reconstruction principles guide interpolation and resampling operations needed for data analysis and visualization (Seguí et al., 2025).

These applications illustrate a recurring theme throughout this chapter. Continuous transforms, discrete algorithms, and sampling principles are not separate topics. They form a unified framework in which mathematical models defined by integrals are approximated by carefully designed discrete operators. The sampling theorem provides the theoretical foundation for this transition, while practical numerical methods introduce regularization, localization, and fast transform techniques to make the computations stable and efficient.

### Rust Implementation

Following the discussion in Sections 13.12.4–13.12.6 on generalized and nonuniform sampling, Program 13.12.4 provides a practical implementation of localized reconstruction for irregularly spaced data using weighted least-squares approximation. While the classical sampling identity in equation (13.12.2) relies on uniform sampling and globally supported sinc kernels, nonuniform settings require alternative formulations that adapt to irregular sampling geometries. This program demonstrates how localized polynomial fitting, combined with Gaussian weighting, can serve as an effective numerical analogue of the regularized sampling formulas described in equation (13.12.3). By introducing controlled noise into the sampled data and comparing unweighted and weighted reconstructions, the implementation highlights the role of localization in improving numerical stability and reducing reconstruction error in practical scenarios.

At the core of the implementation is the function `reconstruct_local_polynomial`, which performs a local polynomial least-squares fit on a stencil of nearby samples. For each evaluation point $t$, the algorithm selects a set of neighboring samples and constructs a polynomial approximation centered at $t$. This approach replaces the global convolution structure of equation (13.12.2) with a localized approximation adapted to nonuniform sampling. The polynomial coefficients are determined by solving the normal equations, which are assembled in terms of local coordinate differences $x = t_i - t$. When Gaussian weighting is applied, the contributions of distant samples are attenuated, reflecting the localization principle underlying equation (13.12.3).

The selection of neighboring samples is handled by the function `nearest_stencil`, which identifies the closest sample points to the evaluation location. This ensures that the reconstruction uses only local information, consistent with the finite stencil restriction described in equation (13.12.3). The Gaussian weighting is implemented through the function `gaussian_window`, where the parameter $\sigma$ controls the degree of localization. Smaller values of $\sigma$ enforce stronger locality, while larger values allow broader contributions. This mirrors the role of window parameters discussed in Section 13.12.3, where different window choices influence error decay and numerical performance.

The least-squares system is constructed using local polynomial basis functions and solved using `solve_linear_system`, which implements Gaussian elimination with partial pivoting. This ensures numerical stability when solving the normal equations. The auxiliary function `spacing_statistics` provides insight into the irregularity of the sampling grid by computing minimum, average, and maximum spacing between samples. This information is important in understanding how nonuniformity affects reconstruction accuracy.

The sampling process itself is implemented in `generate_nonuniform_samples`, which constructs a mildly irregular grid and introduces a small deterministic perturbation to the sample values. This perturbation mimics measurement noise and highlights the advantage of weighted reconstruction, where the Gaussian window suppresses the influence of noisy or distant samples. The test function defined in `test_function` combines oscillatory and localized components, providing a realistic scenario for evaluating reconstruction performance.

The `main` function orchestrates the overall workflow. It generates the sample data, evaluates both unweighted and weighted reconstructions over a dense grid, and computes diagnostic measures such as maximum and RMS error. It also prints representative reconstruction values and verifies behavior near sample points. The results demonstrate that the weighted least-squares reconstruction consistently reduces error compared to the unweighted version, particularly in regions where noise and irregular spacing degrade accuracy.

Program 13.12.4 demonstrates how the principles of regularization and localization extend naturally to nonuniform sampling settings. Unlike the classical sampling theorem, which relies on exact bandlimited assumptions and uniform grids, practical numerical methods must accommodate irregular data and finite precision. The localized least-squares approach provides a flexible and robust framework for such problems, illustrating the broader theme of this chapter: continuous models are translated into stable and efficient discrete algorithms through carefully designed approximations.

The modular structure of the implementation allows further extensions, such as higher-degree polynomial fits, adaptive stencil selection, or alternative window functions like Kaiser–Bessel kernels. These extensions connect directly to modern numerical techniques, including NUFFT-based reconstruction and kernel-based interpolation methods used in large-scale computational applications.

```rust
// Program 13.12.4: Nonuniform Sampling Reconstruction via Local Weighted Least Squares
//
// Problem statement:
// Implement a practical reconstruction method for nonuniformly sampled data.
// For each evaluation point, the program selects a local stencil of nearby
// samples and fits a low-degree polynomial by least squares. A Gaussian window
// is then used to weight nearby samples more strongly than distant samples,
// providing a localized regularized reconstruction. The program compares the
// unweighted and weighted versions on mildly noisy nonuniform samples and
// reports reconstruction errors on a dense evaluation grid.
//
// This implementation is intended as a practical numerical extension of the
// sampling ideas in Sections 13.12.4–13.12.6. Unlike uniform sinc-based
// formulas, nonuniform reconstruction does not rely on simple translates of a
// single kernel. Instead, one constructs local approximants adapted to the
// irregular sampling geometry, with regularization improving stability and
// locality.

use std::cmp::Ordering;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
struct Sample {
    index: usize,
    t: f64,
    value: f64,
}

#[derive(Clone, Debug)]
struct ReconstructionSummary {
    max_abs_error_unweighted: f64,
    rms_error_unweighted: f64,
    max_abs_error_weighted: f64,
    rms_error_weighted: f64,
}

fn main() {
    let n_samples = 61usize;
    let t_min = -4.0_f64;
    let t_max = 4.0_f64;

    // Local polynomial reconstruction settings.
    let stencil_size = 11usize;
    let poly_degree = 3usize;

    // Gaussian localization parameter.
    let sigma_window = 0.35_f64;

    // Small deterministic perturbation in sample values to make regularization
    // useful in a practical sense.
    let noise_level = 2.0e-4_f64;

    // Dense evaluation grid.
    let n_eval = 161usize;

    let samples = generate_nonuniform_samples(n_samples, t_min, t_max, noise_level);

    println!("Nonuniform Sampling Reconstruction via Local Weighted Least Squares");
    println!("==================================================================");
    println!("Number of samples               = {}", n_samples);
    println!("Domain                          = [{:.6}, {:.6}]", t_min, t_max);
    println!("Stencil size                    = {}", stencil_size);
    println!("Polynomial degree               = {}", poly_degree);
    println!("Gaussian window sigma           = {:.6}", sigma_window);
    println!("Noise level                     = {:.6e}", noise_level);
    println!("Evaluation points               = {}", n_eval);

    let (h_min, h_avg, h_max) = spacing_statistics(&samples);
    println!("\nSampling geometry");
    println!("=================");
    println!("Minimum spacing                 = {:.10}", h_min);
    println!("Average spacing                 = {:.10}", h_avg);
    println!("Maximum spacing                 = {:.10}", h_max);

    println!("\nRepresentative samples");
    println!("======================");
    print_samples(&samples, 8);

    let summary = evaluate_reconstruction(
        &samples,
        t_min,
        t_max,
        n_eval,
        stencil_size,
        poly_degree,
        sigma_window,
    );

    println!("\nDiagnostics");
    println!("===========");
    println!(
        "Unweighted LS max error         = {:.12e}",
        summary.max_abs_error_unweighted
    );
    println!(
        "Unweighted LS RMS error         = {:.12e}",
        summary.rms_error_unweighted
    );
    println!(
        "Weighted LS max error           = {:.12e}",
        summary.max_abs_error_weighted
    );
    println!(
        "Weighted LS RMS error           = {:.12e}",
        summary.rms_error_weighted
    );

    println!("\nReconstruction at selected sample-adjacent points");
    println!("=================================================");
    println!(
        "{:>10} {:>16} {:>16} {:>16}",
        "t", "exact", "unweighted", "weighted"
    );

    for &t in &[-3.15_f64, -1.05, 0.15, 1.35, 2.85] {
        let exact = test_function(t);
        let unweighted = reconstruct_local_polynomial(
            t,
            &samples,
            stencil_size,
            poly_degree,
            None,
        );
        let weighted = reconstruct_local_polynomial(
            t,
            &samples,
            stencil_size,
            poly_degree,
            Some(sigma_window),
        );

        println!(
            "{:>10.6} {:>16.10} {:>16.10} {:>16.10}",
            t, exact, unweighted, weighted
        );
    }
}

/// Evaluate the reconstruction on a dense grid and compare unweighted and
/// Gaussian-weighted local polynomial fits.
fn evaluate_reconstruction(
    samples: &[Sample],
    t_min: f64,
    t_max: f64,
    n_eval: usize,
    stencil_size: usize,
    poly_degree: usize,
    sigma_window: f64,
) -> ReconstructionSummary {
    println!("\nRepresentative reconstruction values");
    println!("====================================");
    println!(
        "{:>10} {:>16} {:>16} {:>16} {:>16}",
        "t", "exact", "unweighted", "weighted", "|err_weighted|"
    );

    let mut max_abs_error_unweighted = 0.0_f64;
    let mut rms_error_accum_unweighted = 0.0_f64;

    let mut max_abs_error_weighted = 0.0_f64;
    let mut rms_error_accum_weighted = 0.0_f64;

    for j in 0..n_eval {
        let t = t_min + (t_max - t_min) * j as f64 / (n_eval as f64 - 1.0);

        let exact = test_function(t);
        let unweighted = reconstruct_local_polynomial(
            t,
            samples,
            stencil_size,
            poly_degree,
            None,
        );
        let weighted = reconstruct_local_polynomial(
            t,
            samples,
            stencil_size,
            poly_degree,
            Some(sigma_window),
        );

        let err_unweighted = (unweighted - exact).abs();
        let err_weighted = (weighted - exact).abs();

        max_abs_error_unweighted = max_abs_error_unweighted.max(err_unweighted);
        rms_error_accum_unweighted += err_unweighted * err_unweighted;

        max_abs_error_weighted = max_abs_error_weighted.max(err_weighted);
        rms_error_accum_weighted += err_weighted * err_weighted;

        if j % 20 == 0 || j + 1 == n_eval {
            println!(
                "{:>10.6} {:>16.10} {:>16.10} {:>16.10} {:>16.10e}",
                t, exact, unweighted, weighted, err_weighted
            );
        }
    }

    ReconstructionSummary {
        max_abs_error_unweighted,
        rms_error_unweighted: (rms_error_accum_unweighted / n_eval as f64).sqrt(),
        max_abs_error_weighted,
        rms_error_weighted: (rms_error_accum_weighted / n_eval as f64).sqrt(),
    }
}

/// Build a mildly irregular sample grid and add a small deterministic noise
/// perturbation to the sampled values.
fn generate_nonuniform_samples(
    n: usize,
    t_min: f64,
    t_max: f64,
    noise_level: f64,
) -> Vec<Sample> {
    assert!(n >= 8, "At least eight samples are required.");

    let h = (t_max - t_min) / (n as f64 - 1.0);
    let mut samples = Vec::with_capacity(n);

    for i in 0..n {
        let base = t_min + i as f64 * h;

        let perturb = if i == 0 || i + 1 == n {
            0.0
        } else {
            let theta1 = 0.73 * i as f64;
            let theta2 = 0.29 * i as f64 + 0.35;
            0.28 * h * theta1.sin() + 0.10 * h * theta2.cos()
        };

        let t = base + perturb;
        let clean = test_function(t);

        let noise = noise_level
            * (1.7 * i as f64).sin()
            * (1.0 + 0.3 * (0.41 * i as f64).cos());

        samples.push(Sample {
            index: i,
            t,
            value: clean + noise,
        });
    }

    samples.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(Ordering::Equal));

    for (i, sample) in samples.iter_mut().enumerate() {
        sample.index = i;
    }

    samples
}

/// Smooth test function with oscillatory and localized components.
fn test_function(t: f64) -> f64 {
    let background = sinc(0.85 * t) * (1.7 * t).cos();
    let localized = 0.32 * (-3.0 * (t - 0.85).powi(2)).exp();
    let low_frequency = 0.14 * (0.65 * PI * t).sin();
    background + localized + low_frequency
}

/// Stable normalized sinc function:
/// sinc(x) = sin(pi x) / (pi x).
fn sinc(x: f64) -> f64 {
    let pix = PI * x;
    if pix.abs() < 1.0e-12 {
        1.0
    } else {
        pix.sin() / pix
    }
}

/// Gaussian localization window centered at the evaluation point.
fn gaussian_window(tau: f64, sigma: f64) -> f64 {
    let z = tau / sigma;
    (-0.5 * z * z).exp()
}

/// Reconstruct at t by fitting a local polynomial on the nearest stencil.
/// If sigma_window is Some(sigma), a Gaussian weight is used.
fn reconstruct_local_polynomial(
    t: f64,
    samples: &[Sample],
    stencil_size: usize,
    poly_degree: usize,
    sigma_window: Option<f64>,
) -> f64 {
    let stencil = nearest_stencil(samples, t, stencil_size);
    let d = poly_degree;
    let ncols = d + 1;

    let mut ata = vec![vec![0.0_f64; ncols]; ncols];
    let mut atb = vec![0.0_f64; ncols];

    for sample in &stencil {
        let x = sample.t - t;
        let w = match sigma_window {
            Some(sigma) => gaussian_window(x, sigma),
            None => 1.0,
        };

        let mut powers = vec![1.0_f64; ncols];
        for j in 1..ncols {
            powers[j] = powers[j - 1] * x;
        }

        for i in 0..ncols {
            atb[i] += w * powers[i] * sample.value;
            for j in 0..ncols {
                ata[i][j] += w * powers[i] * powers[j];
            }
        }
    }

    let coeffs = solve_linear_system(ata, atb)
        .expect("Local least-squares normal equations became singular.");

    // Since the polynomial is centered at t, the reconstructed value is c_0.
    coeffs[0]
}

/// Select the nearest stencil_size samples to t.
fn nearest_stencil(samples: &[Sample], t: f64, stencil_size: usize) -> Vec<Sample> {
    let mut with_distance: Vec<(f64, Sample)> = samples
        .iter()
        .cloned()
        .map(|s| ((s.t - t).abs(), s))
        .collect();

    with_distance.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut stencil: Vec<Sample> = with_distance
        .into_iter()
        .take(stencil_size.min(samples.len()))
        .map(|(_, s)| s)
        .collect();

    stencil.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(Ordering::Equal));
    stencil
}

/// Solve Ax = b by Gaussian elimination with partial pivoting.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
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
            return None;
        }

        if pivot_row != k {
            a.swap(k, pivot_row);
            b.swap(k, pivot_row);
        }

        let pivot = a[k][k];
        for j in k..n {
            a[k][j] /= pivot;
        }
        b[k] /= pivot;

        for i in 0..n {
            if i != k {
                let factor = a[i][k];
                for j in k..n {
                    a[i][j] -= factor * a[k][j];
                }
                b[i] -= factor * b[k];
            }
        }
    }

    Some(b)
}

/// Compute spacing statistics for the nonuniform grid.
fn spacing_statistics(samples: &[Sample]) -> (f64, f64, f64) {
    let mut min_h = f64::INFINITY;
    let mut max_h = 0.0_f64;
    let mut sum_h = 0.0_f64;

    for pair in samples.windows(2) {
        let h = pair[1].t - pair[0].t;
        min_h = min_h.min(h);
        max_h = max_h.max(h);
        sum_h += h;
    }

    let avg_h = sum_h / (samples.len() - 1) as f64;
    (min_h, avg_h, max_h)
}

/// Print a representative subset of samples.
fn print_samples(samples: &[Sample], count_each_side: usize) {
    let n = samples.len();

    for (i, sample) in samples.iter().enumerate() {
        if i < count_each_side || i >= n.saturating_sub(count_each_side) {
            println!(
                "idx = {:>3}, t = {:>10.6}, f_i = {:>.10}",
                sample.index, sample.t, sample.value
            );
        } else if i == count_each_side {
            println!("...");
        }
    }
}
```

Program 13.12.4 demonstrates a practical and adaptable approach to nonuniform sampling reconstruction by combining local polynomial approximation with Gaussian regularization. This approach reflects the central idea that numerical stability and efficiency are achieved not by direct application of theoretical formulas, but by modifying them to account for finite data, irregular sampling, and computational constraints. It provides a foundation for further exploration of advanced reconstruction methods and highlights the importance of localization in modern numerical algorithms.

With this section, the chapter concludes its examination of Fourier-based numerical techniques. The discussion has covered a broad set of computational tools, including spectral transforms, methods for handling irregularly sampled data, wavelet-based decompositions, and sampling-driven reconstruction algorithms. Collectively, these techniques form a foundational component of modern numerical computing. They enable efficient representation, analysis, and reconstruction of signals and functions, and they play a central role in many scientific and engineering applications, ranging from signal processing and imaging to large-scale simulation and data-driven modeling.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/eSYQSGCU3zrxNTgj53j9.2","tags":[]}

# 13.13. Conclusion

This chapter has developed the theory and computational practice of Fourier and spectral methods from the algebraic foundations of circulant diagonalization and the discrete convolution theorem through progressively richer applications in filtering, spectral estimation, linear prediction, wavelet analysis, and numerical integration. The central unifying idea is that the discrete Fourier transform converts structured linear operations into pointwise spectral arithmetic, reducing the cost of convolution, correlation, and inversion from $O(N^2)$ to $O(N \log N)$ through the FFT. This complexity reduction is not merely an asymptotic convenience but the decisive factor that makes Fourier-domain methods practical at the scale of modern scientific computation. At the same time, the chapter has emphasized repeatedly that the FFT is a computational engine, not a complete algorithm: the quality of results depends equally on the modeling choices, regularization strategies, statistical estimators, and numerical safeguards that surround it. The Rust implementations throughout the chapter demonstrate how these mathematical principles translate into efficient, verifiable numerical code that respects both algebraic structure and finite-precision arithmetic.

## 13.13.1. Key Takeaways

- Circulant matrices are diagonalized by the DFT, so the convolution theorem $Y_k = X_k H_k$ reduces circular convolution to pointwise spectral multiplication. Linear convolution of finite sequences is embedded into this framework by zero-padding both operands to a length $N \geq N_x + N_h - 1$ and computing two forward FFTs, one pointwise product, and one inverse FFT, yielding an overall cost of approximately $3 \, O(N \log N)$ with $O(N)$ working memory. This FFT-based reduction is the computational foundation for fast filtering, simulation, and inverse-problem workflows throughout the chapter.
- For very long signals, block convolution methods such as overlap-save and overlap-add partition the input into FFT-sized segments and produce valid linear convolution results by discarding wraparound-contaminated samples or by summing overlapping output segments. Deconvolution inverts the convolution model $y = h * x + \eta$ in the frequency domain, but naive spectral division $\widehat{X}_k = Y_k / H_k$ is unstable when spectral coefficients of the point-spread function are small. Tikhonov regularization stabilizes the inversion through the mode-by-mode formula $\widehat{X}_k = \overline{H_k} Y_k / (|H_k|^2 + \lambda |L_k|^2)$, which introduces a frequency-dependent floor that limits noise amplification at poorly conditioned spectral modes.
- The correlation theorem $\mathrm{DFT}\{r_{gh}\}[k] = G_k \overline{H_k}$ enables FFT-based computation of cross-correlation for all lags in $O(N \log N)$ time, replacing the $O(N^2)$ direct evaluation. Autocorrelation is the special case $r_{gg} = \mathrm{IDFT}(|G|^2)$, linking correlation analysis directly to the power spectrum. Practical estimation requires mean removal to suppress DC-dominated correlation, normalized cross-correlation for scale-free similarity scores, and the distinction between biased $(1/N)$ and unbiased $(1/(N - |\ell|))$ finite-window estimators. Generalized cross-correlation methods such as GCC-PHAT introduce frequency-dependent weighting into the cross-spectrum before inversion, sharpening delay estimates by preserving phase information while suppressing amplitude variations.
- The Wiener filter minimizes mean-square reconstruction error under a signal-plus-noise model by constructing the frequency-dependent gain $G[k] = \overline{H[k]} S_{xx}[k] / (|H[k]|^2 S_{xx}[k] + S_{\eta\eta}[k])$, which balances inversion of the system response against suppression of noise at each spectral mode. In the pure denoising case where $H[k] \equiv 1$, this reduces to the spectral shrinkage form $G[k] = S_{xx}[k] / (S_{xx}[k] + S_{\eta\eta}[k])$, which preserves frequency bins dominated by signal energy and attenuates those dominated by noise. The FFT-based implementation requires one forward FFT, one inverse FFT, and $O(N)$ pointwise gain multiplications per filtering pass, but the practical effectiveness of the filter depends critically on the quality of the estimated signal and noise power spectral densities.
- Power spectral density estimation begins with the windowed periodogram, in which a taper $w[n]$ is applied before transformation and the PSD is formed as $\widehat{S}_{xx}[k] \propto |D[k]|^2 / U$ with normalization $U = \sum w[n]^2$. Welch averaging partitions a long record into $K$ overlapping windowed segments, computes individual periodograms, and averages them to reduce variance at the cost of reduced frequency resolution. Multitaper estimation further improves variance control by using $K$ orthogonal tapers and averaging the resulting spectral estimates, with adaptive weighting schemes assigning greater influence to tapers whose estimates agree more closely with a pilot spectrum. These methods collectively illustrate the bias-variance tradeoff that governs all finite-record spectral estimation.
- Time-domain digital filters implement the general LTI difference equation $y[n] = \sum_{m=0}^{M} b_m x[n-m] - \sum_{r=1}^{R} a_r y[n-r]$ with constant per-sample computational cost and minimal latency. FIR filters ($R = 0$) guarantee stability and can achieve exactly linear phase when their coefficients satisfy the symmetry condition $h[n] = h[M-n]$, while modern design methods incorporate sparsity-promoting regularization to reduce the number of effective multiplications. IIR filters achieve sharper frequency selectivity with fewer coefficients but require all poles of the transfer function to lie strictly inside the unit circle for stability, and high-order implementations are factored into cascaded second-order sections to localize rounding errors. Coefficient quantization in fixed-point implementations can perturb pole locations and produce limit cycles, making implementation structure a critical design parameter in embedded systems.
- Linear prediction models each sample as $\widehat{y}_n = \sum_{k=1}^{p} a_k y_{n-k}$, and the coefficients are determined either by direct least-squares minimization of the prediction error $J(a) = \|b - Ya\|_2^2$ or by solving the Toeplitz Yule-Walker system $Ra = r$ using the Levinson-Durbin recursion in $O(p^2)$ time. Burg's method estimates coefficients by recursively updating forward and backward prediction errors, which tends to preserve model stability throughout the recursion. Linear predictive coding turns prediction into compression by encoding the predictor coefficients together with the quantized residual $e_n = y_n - \widehat{y}_n$, and the decoder reconstructs the signal via the recursion $\widehat{y}_n = \sum_{k=1}^{p} a_k \widehat{y}_{n-k} + \widehat{e}_n$ without accumulating long-horizon drift when the quantized residual is available.
- Maximum-entropy spectral estimation constructs the power spectral density that maximizes the entropy functional $\Delta H = \int \log S(f) \, df$ subject to autocorrelation constraints, yielding the all-poles rational form $S(f) = \sigma^2 / |1 - \sum_{k=1}^{p} b_k e^{-i 2\pi f k \Delta t}|^2$. This representation is exactly equivalent to the PSD of an autoregressive process under the identification $b_k = -a_k$, so the same coefficients support time-domain prediction, spectral evaluation, and synthetic signal generation. The AR order $p$ is the principal hyperparameter controlling the tradeoff between spectral resolution and robustness, and information criteria such as AIC, BIC, and FPE guide order selection by balancing goodness of fit against model complexity.
- Spectral analysis of unevenly sampled data reformulates the estimation problem directly at the observed sample times rather than forcing the data onto a uniform grid. The Lomb-Scargle approach treats each candidate frequency as a regression problem with a sinusoidal design matrix $X_\omega$ evaluated at the irregular observation times, and the spectral power measures the reduction in weighted residual sum of squares relative to a null model. When background noise is correlated, a Gaussian-process likelihood framework replaces the simple projection with a covariance-aware fit that incorporates the full structure of the noise process. For nominally uniform data with missing samples, bias-controlled estimation computes the autocovariance from valid sample pairs only, normalizing each lag by the number of contributing pairs, and then obtains the PSD through the discrete Wiener-Khinchin relation rather than by zero-filling and applying an FFT.
- Computing Fourier integrals $I(\omega) = \int_a^b h(t) e^{i\omega t} dt$ via the FFT requires careful treatment of the approximation space, because the FFT interprets sampled data as one period of a periodic sequence and does not by itself constitute a valid quadrature rule. Interpolation-kernel corrected quadrature replaces the bare trapezoidal sum with an attenuation factor $W(\theta)$ derived from the Fourier transform of the interpolation kernel plus analytic endpoint corrections, preserving FFT speed while removing periodicity-induced artifacts. Wavelet transforms provide a complementary multiscale representation through recursive filtering and decimation, with the DWT achieving $O(N)$ complexity, lifting schemes enabling in-place integer-to-integer transforms suitable for lossless compression, and the FFT-accelerated continuous wavelet transform evaluating all translations simultaneously at each scale in $O(N \log N)$ time. The sampling theorem bridges continuous and discrete representations through the Shannon reconstruction series, but practical implementations replace the slowly decaying sinc kernel with windowed regularized formulas whose truncation error decays exponentially with the stencil width under appropriate oversampling conditions.

## 13.13.2. Advice for Beginners

- This chapter brings together many practical applications of the Fourier transform and demonstrates how spectral methods become powerful computational tools for analyzing, filtering, reconstructing, and interpreting data. Before studying the advanced topics presented here, make sure you have a solid understanding of the FFT, frequency-domain representations, sampling theory, and spectral leakage from Chapter 12. These concepts serve as the foundation for nearly every method discussed in this chapter.
- Begin with convolution and correlation. These operations appear throughout signal processing, image analysis, communications, and machine learning. Implement direct $O(N^2)$ versions first and then compare them with FFT-based implementations. Observing the dramatic reduction in computational cost will help you appreciate why spectral methods are so widely used in practice.
- Next, study Wiener filtering and deconvolution. These topics introduce an important principle that appears throughout numerical computing: inverse problems are often ill-conditioned and require regularization. Focus on understanding the trade-off between recovering signal details and suppressing noise amplification. The mathematical ideas encountered here reappear later in imaging, inverse problems, and machine learning.
- Power spectral density estimation deserves special attention because it illustrates the difference between transforming data and statistically estimating properties of a process. Compare periodograms, Welch averaging, and multitaper methods to understand the fundamental bias-variance trade-off in spectral estimation.
- When studying digital filters, begin with FIR filters before moving to IIR filters. FIR filters provide a simpler introduction because they are always stable and can achieve exact linear phase. Once these concepts are familiar, study pole locations, stability conditions, and second-order sections for IIR filters.
- Linear prediction and maximum-entropy methods provide an excellent bridge between signal processing, statistics, and numerical optimization. Focus on understanding how autoregressive models connect time-domain prediction with frequency-domain spectral representations. The Yule-Walker equations and Levinson-Durbin recursion are particularly important because they demonstrate how matrix structure can dramatically reduce computational cost.
- Wavelet transforms should be viewed as a complement to Fourier methods rather than a replacement. Fourier transforms provide global frequency information, whereas wavelets provide localized time-frequency information. Experiment with simple Haar wavelets before exploring more advanced wavelet families.
- Unevenly sampled data present challenges that occur frequently in scientific applications. Study the Lomb-Scargle method carefully, as it demonstrates how classical Fourier ideas can be adapted when uniform sampling assumptions are violated. This topic is particularly relevant in astronomy, geophysics, environmental monitoring, and observational sciences.
- For Rust implementations, become familiar with libraries such as `rustfft`, `realfft`, `ndarray`, `nalgebra`, and `num-complex`. Focus initially on correctness, normalization conventions, and numerical stability before optimizing performance. Many spectral algorithms are sensitive to implementation details, making verification and testing especially important.
- Most importantly, remember that spectral methods are not simply mathematical transformations. They provide a framework for solving practical problems involving noise removal, system identification, signal reconstruction, feature extraction, image restoration, data compression, and scientific inference. The techniques developed in this chapter form a bridge between theoretical Fourier analysis and the real-world computational systems used throughout modern science and engineering.

## 13.13.3. Further Learning with GenAI

To deepen your understanding of Fourier and spectral applications in Rust, consider using the following GenAI prompts:

 1. Write a Rust program that constructs the $N \times N$ DFT matrix $F$, verifies the orthogonality relation $F^* F = NI$, and demonstrates circulant diagonalization by comparing the product $C(h) x$ with the spectral multiplication $Y_k = X_k H_k$ for a short signal and kernel. Then implement linear convolution via zero-padded circular convolution using the FFT and verify the result against direct evaluation of the convolution sum.
 2. Implement an overlap-save block convolution pipeline in Rust that processes a long signal using FFT blocks of length $N$ with $L = N - (M-1)$ valid output samples per block. Then construct a synthetic blurred-and-noisy observation $y = h * x + \eta$, implement both naive inverse filtering and Tikhonov-regularized spectral deconvolution, and compare their reconstruction errors as a function of the regularization parameter $\lambda$.
 3. Build a Rust program that computes FFT-based cross-correlation for two finite real-valued signals using the identity $r_{gh} = \mathrm{IDFT}(G \odot \overline{H})$ with zero-padding to prevent wraparound contamination. Implement autocorrelation as the special case $r_{gg} = \mathrm{IDFT}(|G|^2)$, compute both biased and unbiased finite-window estimators, and implement a GCC-PHAT weighted cross-correlation that normalizes each spectral component by its magnitude before inversion. Verify all results against direct $O(N^2)$ computation.
 4. Implement FFT-based Wiener filtering in Rust for both deconvolution and pure denoising. For the deconvolution case, construct a synthetic observation from a known signal, blur kernel, and additive noise, then apply the Wiener gain $G[k] = \overline{H[k]} S_{xx}[k] / (|H[k]|^2 S_{xx}[k] + S_{\eta\eta}[k])$ using surrogate power spectral densities computed from the known signals. For the denoising case, verify that the spectral shrinkage form $G[k] = S_{xx}[k] / (S_{xx}[k] + S_{\eta\eta}[k])$ attenuates noise-dominated frequency bins while preserving signal-dominated bins. Report mean-square errors before and after filtering.
 5. Write a Rust program that compares three approaches to power spectral density estimation for a synthetic signal containing multiple sinusoidal components and a deterministic disturbance: a single-segment windowed periodogram using a Hann taper, a Welch-averaged estimate with overlapping segments, and a multitaper estimate using orthogonal sine tapers. For each method, report the dominant spectral peaks, compare peak magnitudes and locations, and discuss the visible tradeoff between frequency resolution and variance reduction.
 6. Implement both an FIR and an IIR digital filter in Rust using the general difference equation $y[n] = \sum_m b_m x[n-m] - \sum_r a_r y[n-r]$. Design a symmetric low-pass FIR filter using a windowed-sinc construction and verify the linear-phase property by evaluating the frequency response $H(e^{i\omega})$. Implement a two-section IIR cascade using second-order sections, compute the poles of each section, verify the stability condition $|z_p| < 1$, and compare the impulse responses and frequency responses of both filter types on the same test signal.
 7. Build a Rust program that estimates linear prediction coefficients for a uniformly sampled sequence using three methods: direct least-squares via the normal equations $Y^\top Y a = Y^\top b$, the Yule-Walker system solved by Levinson-Durbin recursion, and Burg's method using forward and backward prediction error updates. Compare the estimated coefficients, innovation variances, and mean-square prediction errors. Then implement a frame-based LPC encoder that computes residuals, quantizes both coefficients and residuals, and reconstructs the signal at the decoder via the synthesis recursion.
 8. Implement maximum-entropy all-poles PSD estimation in Rust by fitting an autoregressive model and evaluating the spectrum $S(f) = \sigma^2 / |1 - \sum_{k=1}^{p} b_k e^{-i 2\pi f k \Delta t}|^2$. Verify the equivalence between the AR spectral density and the maximum-entropy formulation by computing both spectra independently and confirming that they agree numerically. Compute candidate models over a range of orders, evaluate AIC, BIC, and FPE criteria for each, and report the selected order together with the dominant spectral peaks.
 9. Write a Rust program that implements spectral analysis for unevenly sampled data using three complementary methods: a weighted least-squares Lomb-Scargle periodogram that evaluates sinusoidal regression at the observed sample times, a Gaussian-process likelihood scan that constructs a dense covariance matrix and compares null and periodic models through profile log-likelihood improvement, and a bias-controlled PSD estimator for uniformly sampled data with missing values that computes the autocovariance from valid pairs only and transforms it via the discrete Wiener-Khinchin relation. Compare the detected frequencies across all three methods.
10. Implement a Rust program that demonstrates three computational techniques from the final sections of the chapter: (a) a corrected FFT quadrature for Fourier integrals using a linear-interpolation attenuation factor $W_{\mathrm{trap}}(\theta) = 2(1 - \cos\theta)/\theta^2$ with analytic endpoint corrections, comparing against a naive trapezoidal FFT approximation on both periodic and nonperiodic test functions; (b) a multilevel Haar discrete wavelet transform implemented as a filter-bank decomposition with perfect reconstruction and energy conservation verification; and (c) a windowed Shannon sampling reconstruction using a Gaussian regularization kernel with finite stencil, comparing reconstruction accuracy against the classical truncated sinc series.

By engaging with these prompts, you will gain a deeper understanding of how the FFT serves as a unifying computational engine across convolution, correlation, filtering, spectral estimation, linear prediction, and numerical integration, and how the effectiveness of each application depends on the modeling choices, regularization strategies, and numerical safeguards that surround the transform itself.

## 13.13.4. Homework Exercises

To reinforce your learning, complete the following exercises:

 1. Implement a Rust program that constructs the $8 \times 8$ DFT matrix $F$ and verifies the orthogonality relation $F^* F = NI$ with maximum entrywise error below $10^{-12}$. Build a circulant matrix $C(h)$ from a kernel $h = [1, -0.5, 0.25, 0, 0, 0, 0.25, -0.5]$ and verify the diagonalization identity $C(h) = (1/N) F^* \mathrm{diag}(Fh) F$ with maximum error below $10^{-12}$. Then compute the linear convolution of $x = [1, 2, 3, 4, 5]$ and $h_{\mathrm{lin}} = [0.5, -1, 2]$ via zero-padded FFT and verify the result against direct evaluation with maximum error below $10^{-10}$.
 2. Implement an overlap-save block convolution in Rust with FFT length $N = 16$ and a filter of length $M = 5$, processing a signal of length 48. Verify against direct linear convolution with maximum error below $10^{-10}$. Then construct a synthetic deconvolution problem: generate a clean signal of length 32, convolve it with a short PSF $h = [0.1, 0.2, 0.4, 0.2, 0.1]$, add deterministic noise at level $10^{-2}$, and compare naive inverse filtering against Tikhonov-regularized deconvolution with $\lambda = 10^{-2}$ and $L = I$. Report the RMSE of each reconstruction relative to the true signal.
 3. Implement FFT-based cross-correlation in Rust for two real-valued sequences $g = [0, 0, 1, 2, 3, 2, 1, 0]$ and $h = [1, 2, 3, 2, 1]$ using the spectral identity $R = G \odot \overline{H}$ with zero-padding to avoid wraparound. Verify the result against direct $O(N^2)$ computation with maximum difference below $10^{-10}$ and identify the lag at which the cross-correlation magnitude is maximized. Compute the autocorrelation of $h$ as $\mathrm{IDFT}(|H|^2)$ and verify that the zero-lag value equals $\sum |h[n]|^2$. Apply mean removal and compute the normalized cross-correlation with values bounded in $[-1, 1]$.
 4. Implement FFT-based Wiener filtering in Rust for a signal of length 32. Generate a clean signal $x[n] = \sin(2\pi \cdot 3 n/N) + 0.35 \cos(2\pi \cdot 7 n/N)$, convolve it with a blur kernel $h = [0.2, 0.6, 0.2]$, and add deterministic noise. Compute the Wiener deconvolution gain using surrogate power spectral densities from the known clean signal and noise, with a numerical safeguard $\varepsilon = 10^{-12}$. Verify that the Wiener reconstruction reduces the MSE relative to the blurred observation. Then implement the pure denoising case ($H[k] \equiv 1$) and confirm that the spectral shrinkage gain attenuates noise-dominated bins.
 5. Implement a Rust program that estimates the PSD of a synthetic signal of length 2048 containing two sinusoidal components at frequencies $f_1 = 0.10$ and $f_2 = 0.28$ cycles per sample. Compute three estimates: a single-segment Hann-windowed periodogram using the full record, a Welch estimate with segment length 256 and 50% overlap, and a multitaper estimate using 6 orthogonal sine tapers on the full record. For each estimate, identify the two strongest spectral peaks and report their frequencies and magnitudes. Verify that the Welch estimate uses more segments and produces a smoother spectrum than the single-segment periodogram.
 6. Implement a symmetric 21-tap low-pass FIR filter in Rust using a windowed-sinc design with cutoff frequency 0.2 cycles per sample and a Hamming window. Verify the symmetry condition $h[n] = h[M-n]$ with maximum error below $10^{-14}$ and confirm that the DC gain $\sum h[n] = 1$ after normalization. Implement a two-section IIR cascade using biquad sections with poles at radii approximately 0.70 and 0.80. Compute the poles of each section, verify that all pole magnitudes satisfy $|z_p| < 1$, and compare the impulse responses and sampled frequency responses of both filters at frequencies $\omega = 0, 0.25\pi, 0.5\pi, 0.75\pi, \pi$.
 7. Implement a Rust program that estimates linear prediction coefficients for an AR(2)-like test signal of length 120 using three methods: dense least-squares via Cholesky factorization of $Y^\top Y$, the Yule-Walker system via Levinson-Durbin recursion, and Burg's forward-backward error update. For each method with predictor order $p = 8$, report the estimated coefficients, innovation variance, and mean-square prediction error. Verify that the Burg method produces a stable model by checking that the roots of the prediction polynomial $A(z) = 1 - \sum a_k z^{-k}$ all lie inside the unit circle. Implement a frame-based LPC encoder with frame length 64 and predictor order 10, quantize both coefficients and residuals, reconstruct the signal, and report the global MSE.
 8. Implement maximum-entropy all-poles PSD estimation in Rust for a synthetic AR(4) process with two resonant pole pairs at frequencies $f_1 = 0.12$ and $f_2 = 0.185$ cycles per sample. Estimate AR coefficients using the Levinson-Durbin recursion and evaluate the all-poles spectrum $S(f) = \sigma^2 / |1 - \sum b_k e^{-i2\pi f k \Delta t}|^2$ on a grid of 2049 points. Verify the equivalence between the AR and maximum-entropy formulations by computing both spectra independently and confirming maximum absolute difference below $10^{-10}$. Compute models for orders $p = 1, 2, \ldots, 14$, evaluate AIC and BIC for each, and report the selected order under each criterion.
 9. Implement a Rust program that performs spectral analysis for 220 unevenly sampled observations containing two sinusoidal components at $f_1 = 0.72$ Hz and $f_2 = 1.83$ Hz with heteroscedastic noise. Compute a weighted least-squares Lomb-Scargle periodogram by solving the $3 \times 3$ floating-mean normal equations at each trial frequency, and identify the two strongest peaks. Then generate a uniformly sampled signal of length 512 with 18% random missing samples and an extended gap of 65 samples, compute the autocovariance from valid pairs only with Bartlett tapering, transform it to a PSD via the Wiener-Khinchin relation, and compare the dominant peaks against those from a naive zero-filled periodogram.
10. Implement three computational demonstrations in a single Rust program: (a) Evaluate the Fourier integral $I(\omega) = \int_0^1 h(t) e^{i\omega t} dt$ for a nonperiodic test function using both a naive FFT trapezoidal approximation and a corrected linear-interpolation quadrature with attenuation factor $W_{\mathrm{trap}}(\theta) = 2(1-\cos\theta)/\theta^2$ and analytic endpoint corrections on an FFT grid of 512 points, and verify that the corrected method achieves lower maximum error. (b) Implement a multilevel Haar DWT for a signal of length 16 using the filter-bank formulation with periodic boundary conditions, verify perfect reconstruction with maximum error below $10^{-12}$, and confirm that the total wavelet coefficient energy equals the input signal energy. (c) Implement windowed Shannon sampling reconstruction with a Gaussian kernel of width $\sigma = 0.45 m T$ and stencil half-width $m = 8$, and verify that the reconstruction error is smaller than that of the classical truncated sinc series on a dense evaluation grid.

These exercises span the full range of Fourier and spectral applications developed in this chapter, from the algebraic foundations of circulant diagonalization and the convolution theorem through FFT-based filtering, correlation, spectral estimation, linear prediction, maximum-entropy modeling, uneven-sampling analysis, numerical Fourier integration, wavelet transforms, and sampling-based reconstruction. By implementing them in Rust, you will gain direct experience with the interplay between mathematical structure, statistical modeling, and computational efficiency that makes Fourier methods one of the most versatile and widely used frameworks in scientific computing.

+++ {"oxa":"oxa:pqQDe4beUu67RvW3raYP/CQfjXGX1a0vrFXPyR7E6.4","tags":[]}

# References

 1. Abbasi, S.A., Mei, D., Wei, Y., Xu, C., Abbasi, S.M.T., Shakil, S. and Yuan, W. (2025) ‘Deconvolution techniques in optical coherence tomography: advancements, challenges, and future prospects’, *Laser & Photonics Reviews*, 19, 2401394. doi:10.1002/lpor.202401394.
 2. Albentosa-Ruiz, E. and Marchili, N. (2024) ‘High-pass filter periodogram: an improved power spectral density estimator for unevenly sampled data’, *Publications of the Astronomical Society of the Pacific*, 136, 114502. doi:10.1088/1538-3873/ad8781.
 3. Alikaşifoğlu, T., Kartal, B. and Koç, A. (2024) ‘Wiener filtering in joint time-vertex fractional Fourier domains’, *IEEE Signal Processing Letters*, 31, pp. 1319–1323. doi:10.1109/LSP.2024.3396664.
 4. Anand, A. and Dhiman, D. (2024) ‘Computation of highly oscillatory integrals using a Fourier extension approximation’, *arXiv preprint*, arXiv:2408.17037. doi:10.48550/arXiv.2408.17037.
 5. Bahraini, T. and Naeimi-Sadigh, A. (2024) ‘Active noise cancellation gets a boost: a novel diffusion-based approach in spline adaptive filters’, *ISA Transactions*, 155, pp. 286–294. doi:10.1016/j.isatra.2024.10.015.
 6. Cederberg, D. (2024) ‘Toeplitz covariance estimation with applications to MUSIC’, *Signal Processing*, 221, 109506. doi:10.1016/j.sigpro.2024.109506.
 7. Chavanne, C. (2026) ‘Asymptotically-unbiased nonparametric estimation of the power spectral density from uniformly-spaced data with missing samples’, *Advances in Statistical Climatology, Meteorology and Oceanography*, 12, pp. 59–72. doi:10.5194/ascmo-12-59-2026.
 8. Cui, J., Brinkmann, B.H. and Worrell, G.A. (2026) ‘M²NuFFT—A computationally efficient suboptimal power spectrum estimator for fast exploration of nonuniformly sampled time series’, *Digital Signal Processing*, 171, 105834. doi:10.1016/j.dsp.2025.105834.
 9. Damaschke, N., Kühn, V. and Nobach, H. (2024) ‘Bias-free estimation of the covariance function and the power spectral density from data with missing samples including extended data gaps’, *EURASIP Journal on Advances in Signal Processing*, 2024, Article 17. doi:10.1186/s13634-024-01108-4.
10. Diversi, R., Lenzi, A., Speciale, N. and Barbieri, M. (2025) ‘An autoregressive-based motor current signature analysis approach for fault diagnosis of electric motor-driven mechanisms’, *Sensors*, 25(4), 1130. doi:10.3390/s25041130.
11. Dodson-Robinson, S.E. and Haley, C.L. (2025) ‘Multitaper magnitude-squared coherence for time series with missing data: understanding oscillatory processes traced by multiple observables’, manuscript submitted.
12. Donciu, C., Temneanu, M.C. and Serea, E. (2025) ‘Direct FFT oversampling without zero-padding’, *Scientific Reports*, 15, 37269. doi:10.1038/s41598-025-21270-5.
13. Filbir, F. and Tasche, M. (2025) ‘Regularized Shannon sampling formulas related to the special affine Fourier transform’, *Journal of Fourier Analysis and Applications*, 31, Article 40. doi:10.1007/s00041-025-10173-8.
14. Garrison, L.H., Foreman-Mackey, D., Shih, Y.-h. and Barnett, A. (2024) ‘nifty-ls: fast and accurate Lomb–Scargle periodograms using a non-uniform FFT’, *arXiv preprint*, arXiv:2409.08090. doi:10.48550/arXiv.2409.08090.
15. Gúrpide, A. and Middleton, M. (2025) ‘Mind the gaps: improved methods for the detection of periodicities in unevenly sampled data’, *Monthly Notices of the Royal Astronomical Society*, 537(4), pp. 3210–3233. doi:10.1093/mnras/staf196.
16. Harbrecht, H., Schwab, C. and Zank, M. (2025) ‘Wavelet compressed, modified Hilbert transform in the space–time discretization of the heat equation’, *IMA Journal of Numerical Analysis*. doi:10.1093/imanum/draf061.
17. He, L., Zhao, Y., Gao, R., Du, Y. and Du, L. (2024) ‘SFC: achieve accurate fast convolution under low-precision arithmetic’, *arXiv preprint*, arXiv:2407.02913. doi:10.48550/arXiv.2407.02913.
18. Henry, M. (2024) ‘An ultra-precise fast Fourier transform’, *Measurement: Sensors*, 32, 101039. doi:10.1016/j.measen.2024.101039.
19. Jamshidi, S., Espinoza, A.I., Heinzman, J.T., May, P., Uc, E.Y., Narayanan, N.S. and Dasgupta, S. (2025) ‘Linear predictive coding electroencephalography algorithms predict mortality in Parkinson’s disease’, *Clinical Parkinsonism & Related Disorders*, 13, 100409. doi:10.1016/j.prdoa.2025.100409.
20. Jiang, H., Chen, X. and Chen, L. (2026) ‘Sinh regularized Lagrangian nonuniform sampling series’, *Journal of Applied Mathematics and Computing*, 72, Article 67. doi:10.1007/s12190-025-02709-4.
21. Johansson, H. and Gustafsson, O. (2023) ‘On frequency-domain implementation of digital FIR filters using overlap-add and overlap-save techniques’, *arXiv preprint*, arXiv:2302.08845. doi:10.48550/arXiv.2302.08845.
22. Kazi, S.H., Adams, N. and Cohen, E.A.K. (2025) ‘Online spectral density estimation’, *arXiv preprint*, arXiv:2511.11296. doi:10.48550/arXiv.2511.11296.
23. Kim, J. and Skoglund, J. (2024) ‘Neural speech and audio coding’, *arXiv preprint*, arXiv:2408.06954. doi:10.48550/arXiv.2408.06954.
24. Kircheis, M., Potts, D. and Tasche, M. (2024) ‘On numerical realizations of Shannon’s sampling theorem’, *Sampling Theory, Signal Processing, and Data Analysis*, 22, Article 13. doi:10.1007/s43670-024-00087-9.
25. Kircheis, M., Potts, D. and Tasche, M. (2025) ‘Some remarks on regularized Shannon sampling formulas’, *BIT Numerical Mathematics*, 65, Article 40. doi:10.1007/s10543-025-01081-w.
26. Kong, S., Wang, W., Feng, X. and Jia, X. (2025) ‘Noise blind deep residual Wiener deconvolution network for image deblurring’, *Digital Signal Processing*, 165, 105304. doi:10.1016/j.dsp.2025.105304.
27. Kuperman, A. (2025) ‘Discretization of digital controllers comprising second-order notch filters’, *Signals*, 6(4), 69.
28. Le Floc’h, F. (2025) ‘NUFFT for the fast COS method’, *arXiv preprint*, arXiv:2507.13186. doi:10.48550/arXiv.2507.13186.
29. Li, X., Zhang, X., Fan, C., Chen, Y., Zheng, J., Gao, J. and Shen, Y. (2024) ‘Deconvolution based on sparsity and continuity improves the quality of ultrasound image’, *Computers in Biology and Medicine*, 169, 107860. doi:10.1016/j.compbiomed.2023.107860.
30. Liu, C., Li, Y., Fu, C., Zhang, H., Wang, Q., He, D. and Huang, Y. (2025) ‘Enhanced adaptive sine multi-taper power spectral density estimation for system performance evaluation in low-frequency gravitational wave detection’, *Applied Sciences*, 15(7), 3919. doi:10.3390/app15073919.
31. Liu, M., Zeng, Q., Jian, Z., Peng, Y. and Nie, L. (2025) ‘Research on the improvement of the signal time delay estimation method of acoustic positioning for anti-low altitude UAVs’, *Sensors*, 25(9), 2735. doi:10.3390/s25092735.
32. Manamperi, W.N., Abhayapala, T.D., Samarasinghe, P.N. and Zhang, J. (2024) ‘Drone audition: audio signal enhancement from drone embedded microphones using multichannel Wiener filtering and Gaussian-mixture based post-filtering’, *Applied Acoustics*, 216, 109818. doi:10.1016/j.apacoust.2023.109818.
33. Martini, A., Schmidt, S., Ashton, G. and Del Pozzo, W. (2024) ‘Maximum entropy spectral analysis: an application to gravitational waves data analysis’, *The European Physical Journal C*, 84, 1023. doi:10.1140/epjc/s10052-024-13400-6.
34. Moryakova, O. and Johansson, H. (2024) ‘Efficient design and implementation of fast-convolution-based variable-bandwidth filters’, in *Proceedings of EUSIPCO 2024*.
35. Nerma, M.H.M., Elfaki, A.O., Bushnag, A. and Alnemari, M. (2025) ‘An innovative finite impulse response filter design using a combination of L1/L2 regularization to improve sparsity and smoothness’, *Electronics*, 14(22), 4386. doi:10.3390/electronics14224386.
36. Nie, K., Li, H., Han, L., Li, Y. and Xu, J. (2025) ‘3D non-uniform fast Fourier transform program optimization’, *Applied Sciences*, 15(19), 10563. doi:10.3390/app151910563.
37. Nzokem, A.H. (2025) ‘Enhanced fast fractional Fourier transform (FRFT) scheme based on closed Newton-Cotes rules’, *Axioms*, 14(7), 543. doi:10.3390/axioms14070543.
38. Okoniewski, P. and Piskorowski, J. (2025) ‘Transient time reduction in time-varying digital filters via second-order section optimization’, *Applied Sciences*, 15(12), 6512. doi:10.3390/app15126512.
39. Poczekajło, P. and Wirski, R. (2025) ‘Error analysis of digital filters using fixed point arithmetic’, *International Journal of Electronics and Telecommunications*, 71(2). doi:10.24425-ijet.2025.153611.
40. Qin, C. and Zhang, Y. (2024) ‘Application of Wiener filter based on improved BB gradient descent in iris image restoration’, *Journal of Imaging Informatics in Medicine*, 38(2), pp. 1165–1183. doi:10.1007/s10278-024-01238-z.
41. Rahman, A. *et al.* (2025) ‘Estimating word lengths for fixed-point DSP implementations using polynomial chaos expansions’, *Electronics*, 14(2), 365.
42. Rodriguez-Martinez, E., Benavides-Alvarez, C., Aviles-Cruz, C., Lopez-Saca, F. and Ferreyra-Ramirez, A. (2023) ‘Improved parallel implementation of 1D discrete wavelet transform using CPU-GPU’, *Electronics*, 12(16), 3400. doi:10.3390/electronics12163400.
43. Sacchetto, M., Rottondi, C. and Bianco, A. (2024) ‘Implementation and optimization of Burg’s method for real-time packet loss concealment in networked music performance applications’, *Personal and Ubiquitous Computing*, 28, pp. 727–743. doi:10.1007/s00779-024-01806-8.
44. Seguí, A., Ugalde, A., Fichtner, A., Ventosa, S. and Morros, J.R. (2025) ‘DASPack: controlled data compression for distributed acoustic sensing’, *arXiv preprint*, arXiv:2507.16390. doi:10.48550/arXiv.2507.16390.
45. Seilmayer, M., Wondrak, T. and Garcia, F. (2025) ‘Multivariate frequency and amplitude estimation for unevenly sampled data using and extending the Lomb–Scargle method’, *Sensors*, 25(21), 6535. doi:10.3390/s25216535.
46. Sinitsin, V. and Eremeeva, V. (2025) ‘A computationally efficient method for the diagnosis of defects in rolling bearings based on linear predictive coding’, *Algorithms*, 18(2), 58. doi:10.3390/a18020058.
47. Sun, M., Niu, Z., Zhu, X. and Huang, Z. (2025) ‘High-precision time delay estimation algorithm based on generalized quadratic cross-correlation’, *Mathematics*, 13(15), 2397. doi:10.3390/math13152397.
48. Sun, Y. and Qian, W. (2024) ‘Fast algorithms for nonuniform chirp-Fourier transform’, *AIMS Mathematics*, 9(7), pp. 18968–18983. doi:10.3934/math.2024923.
49. Szczepanik, R. and Kelner, J.M. (2025) ‘Filtering and overlapping data for accuracy enhancement of Doppler-based location method’, *Sensors*, 25(5), 1465. doi:10.3390/s25051465.
50. Uchendu, N., Muggleton, J.M. and White, P.R. (2025) ‘Cross-spectral phase method for time delay estimation in acoustic leak detection’, *Applied Acoustics*, 110695. doi:10.1016/j.apacoust.2025.110695.
51. Ungermann, J. and Reichert, R. (2025) ‘JuWavelet – continuous wavelet transform and S transform for gravity wave analysis’, *Geoscientific Model Development*, 18, pp. 8613–8626. doi:10.5194/gmd-18-8613-2025.
52. Wang, J., Liu, H., Han, S., Sun, G. and Hu, X. (2025) ‘Microphone array post-filter based on accurate estimation of noise power spectral density’, *Applied Acoustics*, 227, 110258. doi:10.1016/j.apacoust.2024.110258.
53. Xu, Y., Sun, Y., Wu, H., Cao, W., Bai, L., Tao, S., Tian, Z., Cui, Y., Hao, X., Kuang, C. and Liu, X. (2024) ‘Regularized deconvolution for structured illumination microscopy via accelerated linearized ADMM’, *Optics & Laser Technology*, 169, 110119. doi:10.1016/j.optlastec.2023.110119.
54. Yeung, P.K., Ravikumar, K., Nichols, S. and Uma-Vaideswaran, R. (2025) ‘GPU-enabled extreme-scale turbulence simulations: Fourier pseudo-spectral algorithms at the exascale using OpenMP offloading’, *Computer Physics Communications*, 306, 109364. doi:10.1016/j.cpc.2024.109364.
55. Yi, Y. *et al.* (2025) ‘Design of infinite impulse response maximally flat stable digital filter with low group delay’, *Scientific Reports*. doi:10.1038/s41598-025-87175-5.
56. Zhang, Y., Wang, D., Hu, B., Zhang, J., Gong, X. and Chen, Y. (2024) ‘Enhanced offshore wind farm geophysical surveys: shearlet-sparse regularization in multi-channel predictive deconvolution’, *Remote Sensing*, 16(16), 2935. doi:10.3390/rs16162935.
57. Zhao, R. (2024) ‘Least $l_p$-norm design of complex exponential structure variable fractional delay FIR filters’, *Signal Processing*, 218, 109387. doi:10.1016/j.sigpro.2024.109387.

