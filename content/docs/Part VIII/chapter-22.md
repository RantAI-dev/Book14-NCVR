---
weight: 4900
title: "Chapter 22"
description: "Computational Infrastructure for Trustworthy Numerical Execution"
icon: "article"
date: "2026-07-06T00:00:00+07:00"
lastmod: "2026-07-06T00:00:00+07:00"
katex: true
draft: false
toc: true
---

{{% alert icon="💡" context="info" %}}
<strong><em>“A computation becomes knowledge only when its numbers survive the journey through error, uncertainty, and verification.”</em></strong>
{{% /alert %}}

{{% alert icon="📘" context="success" %}}
<p style="text-align: justify;"><em>Chapter 22 examines the computational infrastructure required for trustworthy numerical computing. The chapter begins by exploring machine arithmetic, floating-point diagnostics, and reproducibility, emphasizing how finite-precision effects influence numerical results. Techniques for diagnosing rounding errors, cancellation, exceptional values, and mixed-precision behavior are developed alongside practical methods for reproducible computation. The chapter then introduces diagnostic output, structured logging, and lightweight plotting as tools for observing and validating numerical algorithms. Discrete structures such as Gray codes are presented for bit-level enumeration and testing, followed by checksums and cyclic redundancy checks for ensuring data integrity in scientific workflows. The discussion continues with lossless compression methods, including Huffman coding, arithmetic coding, and range coding, highlighting their role in preserving information while reducing storage requirements. Finally, arbitrary-precision arithmetic is introduced as a tool for verification, reference computation, and the analysis of ill-conditioned problems. Throughout the chapter, mathematical concepts are integrated with practical Rust implementations, providing readers with essential techniques for building reliable, verifiable, and reproducible numerical software.</em></p>
{{% /alert %}}

# 22.1. Introduction

The final chapter of this book is concerned with the computational infrastructure that makes numerical results reliable, reproducible, and interpretable. Earlier chapters have focused mainly on algorithms for approximation, linear algebra, nonlinear equations, integration, differential equations, integral equations, partial differential equations, optimization, stochastic simulation, and geometric computation. In all those settings, however, the mathematical algorithm is only one part of the complete computational experiment. A numerical result also depends on floating-point arithmetic, machine parameters, diagnostic output, bit-level representations, data-integrity checks, compression schemes, and precision control. These topics are sometimes treated as peripheral programming matters, but in modern scientific computing they are central to trustworthy numerical execution.

A computed result should therefore be understood not merely as the value produced by an abstract algorithm, but as the value produced by an algorithm executed in a particular computational environment. Let $A$ denote a numerical algorithm, $x$ the input data, $\theta$ the algorithmic parameters such as tolerances and iteration limits, and $P$ the platform on which the computation is executed. The complete computational experiment may be represented as:

$$\mathcal E = (A,x,\theta,P) \tag{22.1.1}$$

The reported output is then:

$$\widehat y = \mathcal C(A,x,\theta,P) \tag{22.1.2}$$

where $\mathcal C$ denotes the actual execution process, including arithmetic, compiler choices, library implementations, memory layout, parallel scheduling, and input-output behavior. This notation makes explicit that two nominally identical computations may differ if the platform or execution conditions differ. The recent literature on reproducible high-performance computing and research data integrity supports this view by treating determinism, metadata capture, environment control, validation, provenance, and integrity checks as fundamental components of credible computational science rather than as optional post-processing steps (Antunes and Hill, 2024; Miller and Spiegel, 2025; Stoudt et al., 2024).

## 22.1.1. Trustworthy Numerical Execution

The unifying theme of this chapter is trustworthy numerical execution. A numerical method is trustworthy when its output can be interpreted, repeated, diagnosed, and verified within a known computational model. This does not mean that every result must be bit-for-bit identical on every machine. Rather, it means that differences between executions should be explainable in terms of arithmetic, conditioning, algorithmic stability, data movement, or representation choices.

Let $F$ denote the exact mathematical map associated with a problem, so that,

$$y = F(x)\tag{22.1.3}$$

is the ideal result. A floating-point implementation produces instead:

$$\widehat y = \widehat F(\widehat x) \tag{22.1.4}$$

where $\widehat x$ is the represented input and $\widehat F$ is the implemented finite-precision map. The total error can be decomposed as:

$$\widehat y-y = \left[\widehat F(\widehat x)-F(\widehat x)\right]+\left[F(\widehat x)-F(x)\right] \tag{22.1.5}$$

The first term represents error introduced by the algorithm, arithmetic, and execution environment. The second term represents the effect of perturbing the input data. If $F$ is differentiable and $\Delta x=\widehat x-x$, then locally,

$$
F(\widehat{x}) - F(x)
=
J_F(x)\Delta x
+
O\!\left(\|\Delta x\|^2\right)
\tag{22.1.6}
$$

where $J_F(x)$ is the Jacobian of $F$. Thus, the sensitivity of the mathematical problem and the behavior of the computational environment are inseparable. A stable algorithm may still produce unreliable output if the input data are corrupted, if the floating-point behavior is undocumented, if convergence diagnostics are absent, or if the computational record is insufficient for later verification.

For scalar or vector-valued computations, one may distinguish exact reproducibility from tolerance-based reproducibility. Two executions are bitwise reproducible if:

$$\mathcal C(A,x,\theta,P_1) = \mathcal C(A,x,\theta,P_2)\tag{22.1.7}$$

as stored finite representations. They are numerically reproducible to tolerance $\tau$ in a norm $|\cdot|$ if:

$$\left\|\mathcal C(A,x,\theta,P_1) = \mathcal C(A,x,\theta,P_2)\right\|\le \tau \tag{22.1.8}$$

The tolerance $\tau$ should not be arbitrary. It should be consistent with the conditioning of the problem, the stability of the algorithm, the expected floating-point error, and the scale of the output. This is why reproducibility is a numerical concept, not only a software-engineering concept.

## 22.1.2. Machine Arithmetic, Diagnostics, and Observability

Section 22.2 begins with machine parameters and floating-point diagnostics because finite arithmetic is the foundation on which all later computations rest. A floating-point system can be described abstractly by a radix $\beta$, precision $p$, and exponent bounds $e_{\min}$ and $e_{\max}$. Its normalized numbers have the form:

$$
\begin{aligned}
x
&= \pm m\beta^e,
\qquad
m=d_0.d_1d_2\cdots d_{p-1},
\\
d_0 &\ne 0,
\qquad
e_{\min}\le e\le e_{\max}
\end{aligned}
\tag{22.1.9}
$$

Under rounding to nearest, the standard model for a basic arithmetic operation is:

$$\operatorname{fl}(a\circ b) = (a\circ b)(1+\delta),\qquad |\delta|\le u,\qquad \circ\in\{+,-,\times,/\} \tag{22.1.10}$$

where $u$ is the unit roundoff. For a product of several rounding errors, it is convenient to write:

$$
\prod_{k=1}^{n}(1+\delta_k) = 1+\theta_n,\qquad|\theta_n|\le\gamma_n

\frac{nu}{1-nu}, \qquad nu<1\tag{22.1.11}
$$

This model explains why apparently minor implementation details can change a numerical answer. Floating-point addition is generally not associative:

$$
\operatorname{fl}\!\left(\operatorname{fl}(a+b)+c\right)
\ne
\operatorname{fl}\!\left(a+\operatorname{fl}(b+c)\right)
\tag{22.1.12}
$$

Consequently, parallel reductions, vectorized loops, compiler optimizations, fused multiply-add instructions, and mixed-precision kernels may alter the computed result even when the mathematical expression is unchanged. Recent work on floating-point arithmetic, accumulation-order inference, non-associativity, rounding-error analysis, floating-point exceptions, and mixed precision confirms that modern numerical software must expose and test operational floating-point behavior rather than merely assume that the platform behaves ideally (Boldo et al., 2023; Xie et al., 2025; Shanmugavelu et al., 2024; Tirpankar et al., 2025; Miao et al., 2025; Kashi et al., 2026).

Diagnostics make this hidden numerical state observable. A computation should record not only the final value, but also quantities that explain how that value was obtained. For the linear system:

$$Ax=b \tag{22.1.13}$$

the residual,

$$r_k=b-Ax_k\tag{22.1.14}$$

is a basic diagnostic for an approximate solution $x_k$. The residual is not the same as the forward error, but it is related to it through conditioning. If $A$ is nonsingular, then:

$$
\frac{\|x_k-x\|}{\|x\|}
\le
\kappa(A)
\frac{\|r_k\|}{\|b\|}
\frac{\|b\|}{\|A\|\,\|x\|}
\tag{22.1.15}
$$

where,

$$
\kappa(A)
=
\|A\|\,\|A^{-1}\|
\tag{22.1.16}
$$

is the condition number. This inequality illustrates why diagnostic output and lightweight plotting, developed in Section 22.3, are not decorative. Residual histories, convergence curves, semilog error traces, histograms, and structured logs are instruments for detecting stagnation, instability, precision loss, and implementation errors. Recent literature on residual diagnostics, visualization guidance, and responsible workflows supports treating diagnostic output as part of the computational record itself (Li et al., 2024; Romero-Organvidez et al., 2024; Stoudt et al., 2024).

### Rust Implementation

Following the discussion in Section 22.1.2 on machine arithmetic, diagnostics, and observability, Program 22.1.1 provides a practical implementation of a computational experiment that records numerical metadata, floating-point behavior, residual diagnostics, and output integrity information. Equations (22.1.1) and (22.1.2) emphasize that a numerical result is not produced solely by an abstract algorithm but by an algorithm executed within a specific computational environment. The program illustrates this perspective by exposing several aspects of the execution process that are often hidden from the user, including machine precision, floating-point non-associativity, residual computation, and integrity verification. Rather than treating the final numerical result as an isolated quantity, the implementation demonstrates how trustworthy numerical execution depends on observing and documenting the computational conditions under which the result is produced. In this way, the program serves as a concrete introduction to the broader infrastructure developed throughout Chapter 22 for reproducible, diagnosable, and verifiable numerical computing.

At the core of the implementation is the `ExperimentMetadata` structure, which stores information describing the computational experiment. This reflects the conceptual framework introduced in Equation (22.1.1), where a numerical computation is viewed as the combination of an algorithm, input data, algorithmic parameters, and execution environment. The structure records the algorithm description together with numerical parameters such as the convergence tolerance and iteration limit. Although simple, this metadata establishes the principle that computational results should be accompanied by sufficient contextual information to support later interpretation and verification.

The `machine_epsilon` function determines the machine precision of the floating-point system through an iterative halving process. Starting from unity, the quantity $\varepsilon$ is repeatedly divided by two until the expression $1+\varepsilon/2$ can no longer be distinguished from $1$ in floating-point arithmetic. The resulting value corresponds to the unit roundoff discussed in Equations (22.1.9) through (22.1.11). This diagnostic provides direct empirical confirmation of the precision available in the underlying arithmetic system and illustrates the finite nature of floating-point representation.

The `demonstrate_non_associativity` function illustrates one of the most important consequences of finite-precision arithmetic. Using carefully chosen values, it evaluates both $(a+b)+c$ and $a+(b+c)$, demonstrating that floating-point addition is generally not associative, as described by Equation (22.1.12). Although the mathematical expressions are identical in exact arithmetic, different evaluation orders produce different results due to intermediate rounding effects. This example highlights why parallel reductions, compiler optimizations, and changes in execution order may affect the final numerical output even when the underlying mathematical model remains unchanged.

The functions `matvec_2x2` and `residual_2x2` implement a small linear-algebra example that illustrates the role of diagnostics in numerical computation. The `matvec_2x2` function performs matrix-vector multiplication for a $2\times2$ system, while `residual_2x2` computes the residual vector $r=b-Ax$ corresponding to Equation (22.1.14). The residual serves as a fundamental measure of solution quality and provides information about how closely the computed solution satisfies the original system of equations. The accompanying `norm2` function evaluates the Euclidean norm of the residual, allowing the magnitude of the discrepancy to be summarized by a single scalar quantity. This reflects the diagnostic philosophy discussed around Equations (22.1.13)–(22.1.16), where residual monitoring is treated as an essential component of trustworthy numerical execution.

The `checksum_f64` function introduces a lightweight integrity mechanism by computing a checksum over a collection of floating-point values. Each floating-point number is converted into its underlying binary representation, and the resulting bit patterns are accumulated using wrapping integer arithmetic. While considerably simpler than the cyclic redundancy check techniques developed later in Section 22.5, this checksum illustrates the general principle that numerical outputs should be accompanied by integrity information capable of detecting unintended changes to stored or transmitted results.

The `main` function serves as a complete computational experiment. It begins by constructing an instance of `ExperimentMetadata`, thereby recording the essential parameters associated with the execution. The program then measures machine epsilon, demonstrates floating-point non-associativity, constructs a small linear system, computes a perturbed approximate solution, evaluates the corresponding residual, and determines whether the residual satisfies the prescribed tolerance. Finally, a checksum is generated for the collection of diagnostic quantities. The resulting output provides not only a numerical answer but also a collection of observations that explain how the computation behaved. This directly supports the notion of observability introduced in Section 22.1.2 and illustrates how diagnostic information can become part of the computational record rather than an optional afterthought.

```rust
// Program 22.1.1: Floating-Point Diagnostics and Numerical Observability
//
// Problem statement:
// Demonstrate how a numerical computation can record arithmetic behavior,
// algorithmic parameters, residual diagnostics, and a lightweight integrity
// checksum as part of a reproducible computational experiment.

#[derive(Debug)]
struct ExperimentMetadata {
    algorithm: &'static str,
    tolerance: f64,
    max_iterations: usize,
}

fn machine_epsilon() -> f64 {
    let mut eps = 1.0_f64;

    while 1.0 + eps / 2.0 > 1.0 {
        eps /= 2.0;
    }

    eps
}

fn demonstrate_non_associativity() -> (f64, f64) {
    let a = 1.0e16_f64;
    let b = -1.0e16_f64;
    let c = 1.0_f64;

    let left = (a + b) + c;
    let right = a + (b + c);

    (left, right)
}

fn matvec_2x2(a: [[f64; 2]; 2], x: [f64; 2]) -> [f64; 2] {
    [
        a[0][0] * x[0] + a[0][1] * x[1],
        a[1][0] * x[0] + a[1][1] * x[1],
    ]
}

fn residual_2x2(a: [[f64; 2]; 2], x: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let ax = matvec_2x2(a, x);
    [b[0] - ax[0], b[1] - ax[1]]
}

fn norm2(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn checksum_f64(values: &[f64]) -> u64 {
    values
        .iter()
        .fold(0_u64, |acc, &v| acc.wrapping_add(v.to_bits()))
}

fn main() {
    let metadata = ExperimentMetadata {
        algorithm: "diagnostic residual check for a 2x2 linear system",
        tolerance: 1.0e-12,
        max_iterations: 1,
    };

    println!("Program 22.1.1: Floating-Point Diagnostics and Numerical Observability");
    println!("======================================================================\n");

    println!("Computational Experiment Metadata");
    println!("---------------------------------");
    println!("algorithm       = {}", metadata.algorithm);
    println!("tolerance       = {:.3e}", metadata.tolerance);
    println!("max iterations  = {}\n", metadata.max_iterations);

    let eps = machine_epsilon();

    println!("Machine Arithmetic Diagnostic");
    println!("-----------------------------");
    println!("computed machine epsilon = {:.18e}", eps);
    println!("Rust f64::EPSILON        = {:.18e}\n", f64::EPSILON);

    let (left, right) = demonstrate_non_associativity();

    println!("Floating-Point Non-Associativity");
    println!("--------------------------------");
    println!("(a + b) + c = {:.1}", left);
    println!("a + (b + c) = {:.1}", right);
    println!("difference  = {:.1}\n", (left - right).abs());

    let a = [[3.0, 1.0], [1.0, 2.0]];
    let x_exact = [1.0, 2.0];
    let b = matvec_2x2(a, x_exact);

    let x_computed = [1.0 + 1.0e-13, 2.0 - 2.0e-13];
    let r = residual_2x2(a, x_computed, b);
    let residual_norm = norm2(r);

    println!("Residual Diagnostic for Ax = b");
    println!("------------------------------");
    println!(
        "x_computed        = [{:.16e}, {:.16e}]",
        x_computed[0], x_computed[1]
    );
    println!(
        "residual r = b-Ax = [{:.16e}, {:.16e}]",
        r[0], r[1]
    );
    println!("||r||_2           = {:.16e}", residual_norm);
    println!(
        "residual status   = {}\n",
        if residual_norm <= metadata.tolerance {
            "within tolerance"
        } else {
            "above tolerance"
        }
    );

    let output_values = [
        eps,
        left,
        right,
        x_computed[0],
        x_computed[1],
        r[0],
        r[1],
        residual_norm,
    ];

    let checksum = checksum_f64(&output_values);

    println!("Lightweight Output Integrity Record");
    println!("-----------------------------------");
    println!("checksum over diagnostic values = {:#018x}", checksum);
}
```

Program 22.1.1 demonstrates the central theme of Chapter 22: reliable numerical computing requires more than the execution of an algorithm. A trustworthy computation must also expose information about the arithmetic environment, numerical behavior, diagnostic measures, and integrity of the resulting data. By combining machine-precision detection, floating-point diagnostics, residual analysis, and checksum generation within a single executable example, the program transforms an abstract numerical calculation into a documented computational experiment.

The machine-epsilon calculation confirms the finite precision of the floating-point system and provides an empirical estimate of the unit roundoff discussed in Equations (22.1.9)–(22.1.11). The non-associativity example illustrates how seemingly equivalent arithmetic expressions can produce different results when evaluated in finite precision, reinforcing the importance of execution order and implementation details in numerical software. Together, these examples show that numerical computation is influenced not only by mathematics but also by the characteristics of the underlying arithmetic system.

The residual analysis demonstrates how diagnostic quantities can be used to assess solution quality. As emphasized by Equations (22.1.13)–(22.1.16), the residual provides information about how well a computed solution satisfies the governing equations and often serves as the first indicator of convergence, instability, or implementation errors. The integrity checksum extends this philosophy by attaching a simple verification mechanism to the output data, thereby supporting reproducibility and traceability.

Viewed as a whole, the program provides a practical foundation for the remainder of the chapter. Subsequent sections will develop these ideas further through detailed studies of machine parameters, diagnostic visualization, Gray-code enumeration, checksums and cyclic redundancy checks, entropy coding, and arbitrary-precision arithmetic. The concepts introduced here establish the principle that numerical results become significantly more valuable when accompanied by the information necessary to interpret, verify, reproduce, and trust them.

## 22.1.3. Discrete Structure, Enumeration, and Data Integrity

Sections 22.4 and 22.5 turn from arithmetic to discrete representation. Numerical data are stored as bit strings, configuration spaces are often finite, and many verification tasks require systematic traversal of discrete states. If an $n$-bit state is represented by an element of the Boolean hypercube:

$$\{0,1\}^n \tag{22.1.17}$$

then the Hamming distance between two states $x,y\in\{0,1\}^n$ is:

$$d_H(x,y) = \sum_{i=1}^{n}|x_i-y_i| \tag{22.1.18}$$

A Gray code is an ordering $g_0,g_1,\ldots,g_{2^n-1}$ of the vertices of the hypercube such that,

$$d_H(g_i,g_{i+1})=1\qquad\text{for } i=0,\ldots,2^n-2 \tag{22.1.19}$$

The binary reflected Gray code is given by:

$$g(i)=i\oplus (i\gg 1) \tag{22.1.20}$$

where $\oplus$ denotes bitwise exclusive-or and $i\gg 1$ denotes a right shift. Since successive states differ by one bit, Gray ordering is useful for exhaustive testing, incremental updates, structured enumeration, branch-local search, and cache-friendly traversal of Boolean design spaces. Recent combinatorial work supports presenting Gray codes as general low-change enumeration methods rather than merely as a classical binary encoding trick (Mütze, 2023; Liu et al., 2025; Pilaud and Williams, 2025).

Data integrity is another essential discrete layer. A simple checksum maps a sequence of stored symbols $s=(s_0,\ldots,s_{m-1})$ to a residue such as:

$$C(s) = \left(\sum_{i=0}^{m-1}s_i\right)\bmod M \tag{22.1.21}$$

Such a checksum is inexpensive, but it has limited error-detection power. Cyclic redundancy checks are stronger because they use polynomial arithmetic over $\mathbb F_2$. If a bit string is represented as:

$$S(x)=s_0+s_1x+\cdots+s_{m-1}x^{m-1} \tag{22.1.22}$$

and $G(x)$ is a generator polynomial of degree $r$, then a CRC computes the remainder,

$$R(x)=S(x)x^r \bmod G(x) \tag{22.1.23}$$

The appended remainder is chosen so that the stored or transmitted polynomial satisfies a prescribed divisibility condition. This algebraic structure explains why CRCs detect many burst errors that simple additive checksums miss. In numerical computing, such mechanisms protect input files, checkpoints, compressed results, parameter archives, simulation traces, and data transfers. Recent work on parallel CRC computation, CRC-based error detection, secure logging, and research data integrity supports treating checksums and CRCs as components of reproducible scientific computation (Zhang et al., 2024; An et al., 2025; Choi et al., 2025; Miller and Spiegel, 2025).

## 22.1.4. Lossless Compression and Information Preservation

Sections 22.6 and 22.7 develop lossless compression, beginning with Huffman coding and continuing to arithmetic coding and range coding. Compression is included in this chapter because numerical computation increasingly produces large intermediate arrays, checkpoint files, logs, traces, and diagnostic outputs. Efficient storage is important, but for reproducible computation the compression must also preserve information exactly when the workflow requires exact recovery.

Let a discrete source emit symbols $a_i$ with probabilities $p_i$. Its entropy is:

$$H(p) = -\sum_i p_i\log_2 p_i \tag{22.1.24}$$

For a binary prefix code with code lengths $\ell_i$, the expected code length is:

$$L = \sum_i p_i\ell_i \tag{22.1.25}$$

The Kraft inequality gives the fundamental feasibility condition:

$$\sum_i 2^{-\ell_i}\le 1 \tag{22.1.26}$$

Huffman coding constructs an optimal prefix code for a fixed set of symbol probabilities, while arithmetic coding represents a message by a subinterval of $[0,1)$. If a message $a_{i_1},a_{i_2},\ldots,a_{i_m}$ is encoded using probabilities $p_i$, then the ideal code length is close to:

$$-\log_2\left(\prod_{j=1}^{m}p_{i_j}\right) = -\sum_{j=1}^{m}\log_2 p_{i_j} \tag{22.1.27}$$

This formula shows why arithmetic coding can approach entropy more closely than a symbol-by-symbol integer-length prefix code. Range coding implements the same interval-refinement principle using integer arithmetic and is often more convenient in practical software.

Modern lossless compression should not be presented as a simple opposition between Huffman coding and arithmetic coding. Contemporary systems use prediction, blocking, transforms, bit reorganization, adaptive models, table-driven coders, and hybrid schemes. Recent literature on entropy-coding architectures, scientific floating-point compression, Huffman-based domain pipelines, unordered data structures, semantic arithmetic coding, and hybrid arithmetic-Huffman systems supports a unified treatment of these methods as members of a broader entropy-coding family (Auli-Llinas, 2023; Azami et al., 2025; Lyu et al., 2025; Kunze et al., 2024; Liang et al., 2025; Wiseman, 2025).

## 22.1.5. Precision, Verification, and the Role of Arbitrary Arithmetic

The chapter closes with arbitrary-precision arithmetic in Section 22.8. This placement is deliberate. Once machine parameters, diagnostics, bit-level representations, integrity checks, and compression have been introduced, arbitrary precision appears not as an isolated special topic but as a verification and robustness tool.

Let $u_p$ denote the unit roundoff for a precision level $p$. A computation performed at precision $p$ may be written as:

$$\widehat y_p=\widehat F_p(x) \tag{22.1.28}$$

A common verification strategy compares results across increasing precision. If $p'>p$, define:

$$\Delta_p = \frac{\|\widehat F_{p'}(x)-\widehat F_p(x)\|}{\|\widehat F_{p'}(x)\|} \tag{22.1.29}$$

If $\Delta_p$ decreases consistently as precision increases, the computation is likely limited mainly by roundoff at lower precision. If $\Delta_p$ remains large or irregular, the problem may be ill-conditioned, unstable, singularly perturbed, strongly nonnormal, or affected by cancellation. Arbitrary precision can also provide a reference value,

$$\widehat y_{\mathrm{ref}}=\widehat F_{p_{\mathrm{high}}}(x) \tag{22.1.30}$$

against which a lower-precision implementation may be tested through,

$$e_{\mathrm{obs}} = \frac{\|\widehat y_{64}-\widehat y_{\mathrm{ref}}\|}{\|\widehat y_{\mathrm{ref}}\|} \tag{22.1.31}$$

Such tests do not prove correctness, but they are powerful when combined with residual checks, invariants, backward-error analysis, and independent implementations.

Recent work on multi-precision tools, extended-precision algorithms, hydrodynamic-stability eigenvalue computations, exact geometric kernels, and high-precision quadrature demonstrates that arbitrary precision remains an active part of modern numerical computing. It functions as a verification oracle, a robustness mechanism for difficult problems, and an enabling technology for exact or near-exact computational kernels (Leitold et al., 2025; Zhang and Aiken, 2025; Dondl et al., 2025; Lévy, 2025; Vretinaris, 2026).

Taken together, the sections of this chapter form a coherent progression. Section 22.2 identifies the arithmetic environment. Section 22.3 explains how computations expose their behavior through diagnostics and lightweight plots. Section 22.4 develops Gray codes as low-change enumeration tools for discrete state spaces. Section 22.5 treats checksums and CRCs as algebraic mechanisms for data integrity. Sections 22.6 and 22.7 present entropy coding as a mathematical framework for exact compression. Section 22.8 closes the chapter by showing how arbitrary precision supports verification and robustness. The common thread is that reliable numerical computing requires more than good algorithms. It requires an infrastructure that makes numerical execution observable, repeatable, compact, protected, and verifiable.

# 22.2. Machine Parameters and Floating-Point Diagnostics

Every numerical computation is performed in a finite arithmetic system. Even when an algorithm is derived over the real numbers, its implementation operates with a finite set of representable numbers, bounded exponent range, fixed precision, special values, rounding modes, and platform-dependent execution details. For this reason, machine parameters are not merely descriptive constants. They define the arithmetic environment in which numerical algorithms are interpreted. A solver tolerance, a residual norm, a convergence threshold, or an error estimate has no operational meaning unless it is understood relative to the precision and range of the underlying machine arithmetic.

The classical purpose of machine-parameter analysis is to identify quantities such as the radix, precision, unit roundoff, machine epsilon, smallest positive normal number, largest finite number, and exceptional values. The modern purpose is broader. Numerical software must also diagnose operational floating-point behavior that may not be visible from type names alone. This includes accumulation order, fused multiply-add behavior, subnormal handling, overflow and underflow behavior, compiler-induced reassociation, mixed-precision execution, and floating-point exceptions. Recent literature therefore places floating-point diagnostics at the intersection of arithmetic theory, mixed-precision algorithm design, compiler analysis, hardware behavior, and reproducibility testing (Boldo et al., 2023; Kashi et al., 2026; Xie et al., 2025; Shanmugavelu et al., 2024; Tirpankar et al., 2025; Miao et al., 2025).

## 22.2.1. Floating-Point Systems and Machine Constants

A normalized floating-point system may be described by four parameters:

$$\mathbb F = \mathbb F(\beta,p,e_{\min},e_{\max}) \tag{22.2.1}$$

where $\beta\ge 2$ is the radix, $p$ is the number of significant radix-$\beta$ digits, and $e_{\min}$, $e_{\max}$ are exponent bounds. A nonzero normalized floating-point number has the form:

$$x = \pm m\beta^e,\qquad m = d_0.d_1d_2\cdots d_{p-1},\qquad d_0\ne 0,\\ \qquad e_{\min}\le e\le e_{\max} \tag{22.2.2}$$

Equivalently, the significand may be written as:

$$m = d_0 + d_1\beta^{-1}+\cdots+d_{p-1}\beta^{-(p-1)} \tag{22.2.3}$$

with $1\le d_0\le \beta-1$ for normalized numbers. The smallest positive normalized floating-point number is therefore:

$$x_{\min}^{\mathrm{norm}}=\beta^{e_{\min}} \tag{22.2.4}$$

assuming the normalized significand begins at $1.000\cdots$. The largest finite floating-point number is:

$$x_{\max} = (\beta-\beta^{1-p})\beta^{e_{\max}} \tag{22.2.5}$$

The spacing between adjacent normalized numbers depends on the exponent. In the binade $[\beta^e,\beta^{e+1})$, consecutive floating-point numbers are separated by:

$$\operatorname{ulp}(x)=\beta^{e-p+1} \tag{22.2.6}$$

Thus, floating-point numbers are logarithmically distributed over the representable range. Absolute spacing grows with magnitude, while relative spacing remains approximately controlled by the precision.

Two closely related constants are machine epsilon and unit roundoff. Machine epsilon is commonly defined as the gap between $1$ and the next larger representable number:

$$\varepsilon_{\mathrm{mach}} = \beta^{1-p} \tag{22.2.7}$$

For rounding to nearest, the unit roundoff is:

$$u = \frac{1}{2}\beta^{1-p} \tag{22.2.8}$$

For directed rounding by chopping, one instead obtains,

$$u=\beta^{1-p} \tag{22.2.9}$$

The distinction matters. Machine epsilon describes spacing near $1$, whereas unit roundoff gives the worst-case relative error for one correctly rounded operation under rounding to nearest. For a real number $x$ lying in the normalized range, the rounded floating-point number satisfies the classical model:

$$\operatorname{fl}(x) = x(1+\delta),\qquad |\delta|\le u \tag{22.2.10}$$

provided $x$ does not overflow or underflow. For a basic arithmetic operation,

$$\operatorname{fl}(a\circ b) = (a\circ b)(1+\delta),\qquad|\delta|\le u,\qquad\circ\in\{+,-,\times,/\},\tag{22.2.11}$$

again assuming that the exact result is representable within the normalized range after rounding.

The model in equation (22.2.11) is the starting point for most backward-error analyses. If several operations are performed, products of local rounding terms arise. It is convenient to define $\theta_n$ by:

$$\prod_{k=1}^{n}(1+\delta_k)=1+\theta_n,\qquad |\delta_k|\le u \tag{22.2.12}$$

If $nu<1$, then

$$|\theta_n|\le\gamma_n = \frac{nu}{1-nu} \tag{22.2.13}$$

The quantity $\gamma_n$ appears throughout floating-point error analysis because it summarizes the accumulated effect of $n$ small relative perturbations. For example, a computed dot product,

$$s = \sum_{i=1}^{n} a_i b_i\tag{22.2.14}$$

usually satisfies a bound of the form:

$$\widehat s = \sum_{i=1}^{n} a_i b_i(1+\theta_i),\qquad|\theta_i|\le \gamma_n \tag{22.2.15}$$

under a fixed sequential summation order and standard assumptions. The exact constants depend on the evaluation order, whether fused multiply-add is used, and how intermediate results are rounded. This is the first indication that formal machine constants alone are insufficient. The operational details of execution must also be diagnosed.

## 22.2.2. Rounding, Cancellation, and Exceptional Values

Rounding converts a real number into a nearby representable floating-point number. Let $\mathbb F$ be the set of finite representable numbers and let $R:\mathbb R\to \mathbb F$ denote a rounding map. For rounding to nearest,

$$R(x)=\operatorname*{arg\,min}_{y\in\mathbb F}|x-y| \tag{22.2.16}$$

with a tie-breaking rule when $x$ is exactly halfway between two representable numbers. In IEEE-style arithmetic, ties are commonly resolved by choosing the representable number with an even least significant bit. Directed rounding modes instead satisfy inequalities such as,

$$R_{\downarrow}(x)\le x,\qquad R_{\uparrow}(x)\ge x\tag{22.2.17}$$

which are useful in interval arithmetic and verified computation.

Rounding error is usually benign when it acts as a small relative perturbation, but it becomes dangerous when a computation contains cancellation. Consider the subtraction of two nearby numbers:

$$z=x-y \tag{22.2.18}$$

If $x$ and $y$ are already perturbed,

$$\widehat x=x(1+\delta_x),\qquad\widehat y=y(1+\delta_y) \tag{22.2.19}$$

then,

$$\widehat x-\widehat y = (x-y)+x\delta_x-y\delta_y \tag{22.2.20}$$

The relative error in the subtraction is approximately,

$$\frac{|x\delta_x-y\delta_y|}{|x-y|} \tag{22.2.21}$$

When $x\approx y$, the denominator is small and the relative error may be large even though $|\delta_x|$ and $|\delta_y|$ are small. This phenomenon is loss of relative accuracy due to cancellation. It does not violate the floating-point model. Rather, it reflects the ill-conditioning of the subtraction problem near equal operands.

Floating-point systems also contain exceptional values and exceptional events. Overflow occurs when the exact result exceeds the largest finite representable number $x_{\max}$. Underflow occurs when the exact result is too small in magnitude to be represented as a normalized number. If subnormal numbers are supported, values smaller than $x_{\min}^{\mathrm{norm}}$ may still be represented with reduced precision. A subnormal number has the form:

$$x = \pm 0.d_1d_2\cdots d_{p-1}\,\beta^{e_{\min}} \tag{22.2.22}$$

so the leading significand digit is no longer required to be nonzero. Subnormals provide gradual underflow, meaning that numbers do not jump abruptly from $x_{\min}^{\mathrm{norm}}$ to zero. However, because leading precision is lost, relative-error guarantees weaken in the subnormal range.

The presence of special values such as $+\infty$, $-\infty$, and NaN makes floating-point arithmetic more robust in some workflows and more subtle in others. Division by zero may produce an infinity, invalid operations may produce NaN, and overflow may produce infinity depending on the rounding mode and exception handling. A numerical program that silently propagates NaN or infinity can fail far from the operation that first caused the exception. For this reason, floating-point diagnostics should record exceptional events, not merely final outputs.

## 22.2.3. Empirical Determination of Machine Parameters

Although machine constants are usually available from language libraries, it is instructive and useful to understand how they can be determined empirically. The classical machine-epsilon experiment repeatedly halves a positive number $\epsilon$ until adding it to $1$ no longer changes the floating-point value:

$$\operatorname{fl}(1+\epsilon)>1,\qquad\operatorname{fl}(1+\epsilon/2)=1 \tag{22.2.23}$$

The final $\epsilon$ approximates the gap between $1$ and the next larger representable number. In a binary system with precision $p$, this gives:

$$\epsilon \approx 2^{1-p} \tag{22.2.24}$$

The unit roundoff for rounding to nearest is then approximately,

$$u\approx \frac{\epsilon}{2} \tag{22.2.25}$$

The radix may be detected by observing the scale at which increments to a large number stop affecting its representation. One classical idea is to find a floating-point number $a$ for which:

$$\operatorname{fl}(a+1)=a \tag{22.2.26}$$

and then find the smallest positive $b$ such that,

$$\operatorname{fl}(a+b)>a \tag{22.2.27}$$

The observed increment contains information about the radix and spacing at the magnitude of $a$. Similarly, exponent limits can be probed by repeated scaling:

$$x_{k+1}=\operatorname{fl}(\beta x_k)\tag{22.2.28}$$

until overflow, or

$$x_{k+1}=\operatorname{fl}(x_k/\beta)\tag{22.2.29}$$

until underflow or zero. These experiments reveal not only nominal exponent bounds, but also whether gradual underflow is supported and whether the platform flushes subnormal numbers to zero.

The smallest positive normal number can be interpreted through the first exponent at which normalized representation begins:

$$x_{\min}^{\mathrm{norm}}=\beta^{e_{\min}} \tag{22.2.30}$$

If subnormals are present, the smallest positive subnormal number is:

$$x_{\min}^{\mathrm{sub}} = \beta^{e_{\min}-(p-1)} \tag{22.2.31}$$

The ratio:

$$\frac{x_{\min}^{\mathrm{norm}}}{x_{\min}^{\mathrm{sub}}} = \beta^{p-1}\tag{22.2.32}$$

shows the width of the gradual-underflow region in units of powers of the radix. Empirical tests that distinguish equations (22.2.30) and (22.2.31) are useful because some execution environments may support subnormals in principle but treat them differently for performance reasons.

The largest finite number can be probed by repeated multiplication until overflow. Theoretically,

$$x_{\max} = (\beta-\beta^{1-p})\beta^{e_{\max}} \tag{22.2.33}$$

In practice, detecting $x_{\max}$ requires care because the intermediate sequence may overflow before it reaches the largest finite value exactly. More robust procedures combine scaling, bisection, and checks for infinity.

These empirical experiments are valuable pedagogically, but in modern software they should be treated as diagnostics rather than as replacements for standard library constants. Their purpose is to confirm that the observed arithmetic agrees with the assumed arithmetic. If the observed value of $\varepsilon_{\mathrm{mach}}$, subnormal behavior, or overflow behavior differs from expectation, then the implementation, compiler configuration, runtime environment, or hardware mode must be inspected.

### Rust Implementation

Following the discussion in Section 22.2.3 on the empirical determination of machine parameters, Program 22.2.1 provides a practical implementation of floating-point diagnostics for IEEE double-precision arithmetic. While modern programming languages provide predefined constants describing machine characteristics, numerical software should not simply assume that the runtime environment behaves exactly as expected. Instead, important arithmetic properties should be verified through direct experimentation. This program performs a series of classical floating-point diagnostic tests to estimate machine epsilon, unit roundoff, normal and subnormal limits, overflow behavior, underflow behavior, and exceptional values such as infinity and NaN. The implementation complements the theoretical development of Equations (22.2.7)–(22.2.33) by demonstrating how these quantities can be observed directly from the behavior of the arithmetic system itself. In doing so, the program illustrates the central theme of Section 22.2: machine parameters are not merely static constants but operational properties that define the effective numerical environment in which algorithms execute.

At the core of the implementation are a collection of diagnostic functions that probe different aspects of floating-point arithmetic through direct experimentation. Rather than relying solely on predefined constants, these functions attempt to recover important machine parameters from arithmetic behavior itself. This approach mirrors the methodology described in Section 22.2.3, where machine characteristics are treated as observable properties of the computational environment rather than merely as specification values.

The `machine_epsilon` function implements the classical experiment described by Equations (22.2.23) and (22.2.24). Beginning with a value of unity, the algorithm repeatedly halves a candidate quantity $\epsilon$ until the expression $1+\epsilon/2$ can no longer be distinguished from $1$ by the floating-point system. The resulting value approximates the distance between $1$ and the next larger representable number and therefore provides an empirical estimate of the machine epsilon introduced in Equation (22.2.7). The closely related `unit_roundoff` function then computes the corresponding unit roundoff by dividing the machine epsilon by two, consistent with Equation (22.2.8) for rounding-to-nearest arithmetic.

The functions `smallest_positive_by_halving` and `smallest_positive_normal_by_scaling` investigate the lower limits of representable numbers. The first repeatedly halves a positive value until no smaller positive floating-point number can be represented, thereby identifying the smallest positive subnormal value. The second uses the `is_normal` predicate to detect the transition between normalized and subnormal representation, yielding an empirical estimate of the smallest positive normalized number discussed in Equation (22.2.30). Comparing these quantities provides direct evidence of gradual underflow and illustrates the distinction between the normalized and subnormal ranges described by Equations (22.2.30)–(22.2.32).

The `largest_finite_by_doubling` function explores the upper end of the representable range by repeatedly doubling a floating-point number until the next multiplication would overflow. Although the procedure identifies the largest finite power of two rather than the absolute maximum floating-point number, it provides a practical demonstration of the exponent limits discussed in Equation (22.2.33). The accompanying `demonstrate_overflow` function intentionally exceeds the representable range by evaluating `f64::MAX * 2.0`, thereby producing positive infinity and illustrating overflow behavior. Conversely, the `demonstrate_underflow` function constructs an extremely small quantity whose magnitude falls below the representable range, thereby demonstrating underflow to zero.

The functions `demonstrate_nan` and `demonstrate_infinity` investigate exceptional floating-point values. The first evaluates the indeterminate operation $0/0$, producing a NaN result, while the second evaluates division by zero to produce positive infinity. These experiments illustrate the exceptional values discussed in Section 22.2.2 and emphasize that floating-point systems contain special representations that propagate through subsequent calculations. Such values can be useful diagnostic indicators but may also obscure the source of numerical failure if not detected and reported appropriately.

The `main` function serves as a comprehensive floating-point diagnostic report. It executes each experiment, compares empirical observations against Rust's built-in floating-point constants, and summarizes the observed arithmetic environment. The resulting output includes estimates of machine epsilon and unit roundoff, measurements of normal and subnormal limits, demonstrations of overflow and underflow, and verification of exceptional-value behavior. By combining these experiments within a single executable framework, the program illustrates how empirical diagnostics can validate the assumptions underlying Equations (22.2.7)–(22.2.33) and provide confidence that the observed arithmetic agrees with the expected floating-point model.

```rust
// Program 22.2.1: Empirical Machine Parameters and Floating-Point Constants
//
// Problem statement:
// Empirically determine selected floating-point machine parameters for f64
// arithmetic and compare them with Rust's standard constants.

fn machine_epsilon() -> f64 {
    let mut eps = 1.0_f64;

    while 1.0 + eps / 2.0 > 1.0 {
        eps /= 2.0;
    }

    eps
}

fn unit_roundoff() -> f64 {
    machine_epsilon() / 2.0
}

fn smallest_positive_by_halving() -> f64 {
    let mut x = 1.0_f64;

    while x / 2.0 > 0.0 {
        x /= 2.0;
    }

    x
}

fn smallest_positive_normal_by_scaling() -> f64 {
    let mut x = 1.0_f64;

    while (x / 2.0).is_normal() {
        x /= 2.0;
    }

    x
}

fn largest_finite_by_doubling() -> f64 {
    let mut x = 1.0_f64;

    while (x * 2.0).is_finite() {
        x *= 2.0;
    }

    x
}

fn demonstrate_overflow() -> f64 {
    f64::MAX * 2.0
}

fn demonstrate_underflow() -> f64 {
    f64::MIN_POSITIVE * f64::EPSILON * 0.5
}

fn demonstrate_nan() -> f64 {
    0.0_f64 / 0.0_f64
}

fn demonstrate_infinity() -> f64 {
    1.0_f64 / 0.0_f64
}

fn main() {
    println!("Program 22.2.1: Empirical Machine Parameters and Floating-Point Constants");
    println!("=========================================================================\n");

    let eps = machine_epsilon();
    let u = unit_roundoff();

    println!("Machine Epsilon and Unit Roundoff");
    println!("---------------------------------");
    println!("empirical machine epsilon  = {:.18e}", eps);
    println!("Rust f64::EPSILON          = {:.18e}", f64::EPSILON);
    println!("empirical unit roundoff    = {:.18e}", u);
    println!("expected unit roundoff     = {:.18e}\n", f64::EPSILON / 2.0);

    let xmin_sub = smallest_positive_by_halving();
    let xmin_norm = smallest_positive_normal_by_scaling();

    println!("Smallest Positive Values");
    println!("------------------------");
    println!("empirical smallest subnormal = {:.18e}", xmin_sub);
    println!("Rust f64::MIN_POSITIVE       = {:.18e}", f64::MIN_POSITIVE);
    println!("empirical smallest normal    = {:.18e}", xmin_norm);
    println!("Rust f64::MIN_POSITIVE       = {:.18e}", f64::MIN_POSITIVE);
    println!(
        "normal/subnormal ratio       = {:.18e}\n",
        xmin_norm / xmin_sub
    );

    let xmax_power = largest_finite_by_doubling();

    println!("Large Magnitude and Overflow Behavior");
    println!("-------------------------------------");
    println!("largest finite power of two before overflow = {:.18e}", xmax_power);
    println!("Rust f64::MAX                         = {:.18e}", f64::MAX);
    println!("overflow example f64::MAX * 2         = {}\n", demonstrate_overflow());

    println!("Exceptional Values");
    println!("------------------");
    let inf = demonstrate_infinity();
    let nan = demonstrate_nan();
    let underflow = demonstrate_underflow();

    println!("1.0 / 0.0                    = {}", inf);
    println!("is infinite?                 = {}", inf.is_infinite());
    println!("0.0 / 0.0                    = {}", nan);
    println!("is NaN?                      = {}", nan.is_nan());
    println!("underflow example            = {:.18e}", underflow);
    println!("underflow equals zero?       = {}", underflow == 0.0);

    println!("\nDiagnostic Summary");
    println!("------------------");
    println!("f64 radix is binary, so beta = 2");
    println!("f64 significand precision    = {} bits", f64::MANTISSA_DIGITS);
    println!("f64 minimum exponent         = {}", f64::MIN_EXP);
    println!("f64 maximum exponent         = {}", f64::MAX_EXP);
}
```

Program 22.2.1 demonstrates how the fundamental characteristics of a floating-point system can be explored through direct numerical experimentation. Rather than relying exclusively on language-defined constants, the program verifies important arithmetic properties empirically and compares the observed values with the expected behavior of IEEE double-precision arithmetic. This approach reflects the diagnostic philosophy developed throughout Section 22.2: numerical software should confirm the assumptions on which its computations depend.

The machine-epsilon and unit-roundoff experiments provide direct evidence for the finite precision of the floating-point system and illustrate the concepts introduced in Equations (22.2.7)–(22.2.10). The observed agreement between empirical estimates and standard library constants confirms that the arithmetic behaves as expected in the neighborhood of unity. Likewise, the experiments involving normal and subnormal numbers reveal the structure of the representable range and demonstrate the mechanism of gradual underflow described by Equations (22.2.30)–(22.2.32).

The overflow, underflow, infinity, and NaN tests highlight the importance of exceptional values in practical numerical computation. These quantities are not mathematical objects in the usual sense, yet they play a crucial role in the behavior of floating-point algorithms. Their presence can provide useful diagnostic information, but they may also propagate silently through large computations if not monitored carefully. For this reason, robust numerical software should treat exceptional-value detection as an integral part of floating-point diagnostics rather than as an optional debugging feature.

Viewed more broadly, the program establishes a foundation for the remainder of Section 22.2. Later implementations will examine accumulation order, compensated summation, mixed-precision effects, floating-point exceptions, and reproducibility diagnostics. Together, these topics extend the analysis from static machine parameters to the dynamic behavior of numerical algorithms operating within a finite arithmetic environment. The ability to measure and verify this environment is a prerequisite for understanding the accuracy, stability, and reproducibility of all subsequent numerical computations.

## 22.2.4. Non-Associativity, Accumulation Order, and Reproducibility

Floating-point addition and multiplication do not obey all algebraic identities of real arithmetic. The most important failure for numerical reproducibility is non-associativity:

$$\operatorname{fl}\!\left(\operatorname{fl}(a+b)+c\right)\ne\operatorname{fl}\!\left(a+\operatorname{fl}(b+c)\right)\tag{22.2.34}$$

For a sum,

$$s=\sum_{i=1}^{n} x_i\tag{22.2.35}$$

different evaluation orders can produce different floating-point results. A left-to-right summation gives:

$$\widehat s_{\mathrm{seq}} = \operatorname{fl}(\cdots\operatorname{fl}(\operatorname{fl}(x_1+x_2)+x_3)\cdots+x_n) \tag{22.2.36}$$

whereas a pairwise summation computes a binary reduction tree. The two methods generally produce different rounding-error patterns. Under standard assumptions, sequential summation admits a bound of the form:

$$|\widehat s_{\mathrm{seq}}-s|\le\gamma_{n-1}\sum_{i=1}^{n}|x_i|\tag{22.2.37}$$

while pairwise summation often improves the depth of error accumulation to a logarithmic scale, giving a bound involving approximately $\gamma_{\lceil \log_2 n\rceil}$ at each reduction level, subject to details of the implementation.

Compensated summation reduces the effect of lost low-order terms. In Kahan-style summation, one maintains both a running sum $s_k$ and a compensation $c_k$. A simplified recurrence is:

$$y_k = x_k-c_{k-1} \tag{22.2.38}$$

$$t_k = \operatorname{fl}(s_{k-1}+y_k) \tag{22.2.39}$$

$$c_k = \operatorname{fl}\!\left((t_k-s_{k-1})-y_k\right) \tag{22.2.40}$$

$$s_k=t_k \tag{22.2.41}$$

The compensation $c_k$ estimates the low-order information lost in the addition. This does not make floating-point summation exact, but it often improves reproducibility and accuracy when summing data with mixed signs or widely varying magnitudes.

The diagnostic problem is that large software systems often hide the actual accumulation order. A matrix multiplication, convolution, reduction, or machine-learning kernel may use vector instructions, blocked algorithms, parallel reductions, GPU warps, or fused multiply-add instructions. The mathematical expression

$$\sum_{i=1}^{n} a_i b_i\tag{22.2.42}$$

does not uniquely define an execution order. The implemented expression may be closer to:

$$\widehat s = \operatorname{Reduce}_{T}\left(\operatorname{fl}(a_i b_i)\right) \tag{22.2.43}$$

where $T$ is an implicit reduction tree. Floating-point diagnostics aim to infer or constrain $T$. This is precisely why recent work on revealing floating-point accumulation orders is relevant to modern numerical computing: it transforms a hidden reproducibility issue into a measurable property of the implementation (Xie et al., 2025). Related work on non-associativity shows that such effects are significant in high-performance computing and deep-learning workflows (Shanmugavelu et al., 2024).

Compiler transformations add another layer. Algebraically equivalent real expressions may not be equivalent in floating-point arithmetic. For example,

$$(a+b)+c\quad\text{and}\quad a+(b+c)\tag{22.2.44}$$

are equivalent over $\mathbb R$, but not over $\mathbb F$. Likewise,

$$a b + a c\quad\text{and}\quad a(b+c)\tag{22.2.45}$$

may differ because multiplication and addition are rounded at different times. Fused multiply-add introduces another distinction:

$$\operatorname{fma}(a,b,c) = \operatorname{fl}(ab+c) \tag{22.2.46}$$

where the product and addition are rounded only once, unlike

$$\operatorname{fl}(\operatorname{fl}(ab)+c) \tag{22.2.47}$$

which rounds twice. The fused form is often more accurate, but it can also change bitwise results relative to an unfused implementation. Floating-point diagnostics must therefore record whether fused operations, reassociation, vectorization, and fast-math assumptions are enabled.

### Rust Implementation

Following the discussion in Section 22.2.4 on non-associativity, accumulation order, and reproducibility, Program 22.2.2 provides a practical implementation of several summation strategies and floating-point diagnostics. Although the mathematical expression in Equation (22.2.35) defines a sum uniquely over the real numbers, its floating-point realization depends on the order in which additions are performed. Different reduction trees may therefore produce different numerical results, even when evaluating the same data set. This program compares sequential summation, reverse-order summation, pairwise summation, and Kahan compensated summation to illustrate how accumulation order affects rounding-error propagation. It also includes diagnostics for non-associativity and fused multiply-add behavior, thereby providing concrete demonstrations of Equations (22.2.34)–(22.2.47). The implementation highlights the central theme of this section: numerical reproducibility is influenced not only by the mathematical problem being solved but also by the operational details of floating-point execution.

At the core of the implementation are several summation algorithms that evaluate the same mathematical expression using different accumulation orders. The functions `sequential_sum`, `reverse_sum`, `pairwise_sum`, and `kahan_sum` all compute the sum defined by Equation (22.2.35), yet they do so using distinct reduction structures. By comparing their outputs, the program exposes how floating-point arithmetic transforms what is mathematically a single operation into a family of implementation-dependent computations.

The `sequential_sum` function performs the classical left-to-right accumulation described by Equation (22.2.36). Each element is added directly to a running total, producing a reduction chain whose depth grows linearly with the number of terms. Because every addition introduces a small rounding error, the accumulated error may grow according to bounds of the form discussed in Equation (22.2.37). This implementation serves as a baseline against which alternative summation strategies can be compared.

The `reverse_sum` function evaluates exactly the same set of numbers but processes them in reverse order. Since floating-point addition is not associative, reversing the accumulation order generally changes the pattern of rounding errors and may therefore produce a different result. This simple modification illustrates that the numerical outcome depends not only on the data but also on the reduction tree implicitly chosen by the implementation.

The `pairwise_sum` function implements a recursive binary reduction strategy. Instead of adding values sequentially, the data are divided into two approximately equal halves, each half is summed independently, and the two partial sums are then combined. This approach resembles the balanced reduction trees commonly used in parallel algorithms and is closely related to the reduction structure represented abstractly by Equation (22.2.43). Because the depth of the reduction tree grows logarithmically rather than linearly, pairwise summation often exhibits improved numerical behavior compared with simple sequential accumulation.

The `kahan_sum` function implements compensated summation using the recurrence described by Equations (22.2.38)–(22.2.41). In addition to the running sum, the algorithm maintains a compensation variable that estimates the low-order information lost during previous additions. Each new term is adjusted using this compensation before being incorporated into the sum. Although compensated summation does not eliminate rounding error entirely, it often recovers information that would otherwise be discarded and therefore improves both accuracy and reproducibility when summing values of widely differing magnitudes.

The `relative_error` function measures the discrepancy between an approximation and a reference value. It is used throughout the program to compare the outputs of different summation methods against a known reference sum. This diagnostic provides a quantitative measure of the impact of accumulation order and illustrates how numerical differences that may appear small in absolute terms can become significant relative to the scale of the exact result.

The `demonstrate_non_associativity` function directly illustrates Equation (22.2.34). Using carefully selected operands, it evaluates both $(a+b)+c$ and $a+(b+c)$. Although the two expressions are algebraically equivalent over the real numbers, the computed floating-point results differ because intermediate rounding occurs at different stages. This experiment provides one of the most direct demonstrations of non-associativity in finite-precision arithmetic.

The `demonstrate_fma_difference` function investigates the distinction between separate multiply-add evaluation and fused multiply-add evaluation. The expression $ab+c$ is first computed using separate multiplication and addition operations and then recomputed using the `mul_add` operation, which corresponds to the fused operation discussed in Equation (22.2.46). Because the fused version rounds only once, while the unfused version rounds twice as described by Equation (22.2.47), the two results may differ. This experiment illustrates how hardware capabilities and compiler decisions can influence floating-point behavior even when the mathematical expression remains unchanged.

The `main` function serves as a floating-point reproducibility experiment. It first demonstrates non-associativity, then constructs a data set containing large cancellation effects together with many small contributions. This data arrangement intentionally amplifies the influence of accumulation order. The program subsequently evaluates the data using sequential, reverse, pairwise, and compensated summation strategies and reports the resulting relative errors. Finally, the fused multiply-add diagnostic is performed. Together, these experiments reveal how accumulation order, compensation techniques, and hardware-supported arithmetic operations influence numerical accuracy and reproducibility in practical computations.

```rust
// Program 22.2.2: Accumulation Order, Pairwise Summation, and Kahan Compensation
//
// Problem statement:
// Demonstrate that floating-point summation depends on accumulation order.
// The program compares sequential summation, reverse summation, pairwise
// summation, Kahan compensated summation, and fused multiply-add evaluation.

fn sequential_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;

    for &x in values {
        sum += x;
    }

    sum
}

fn reverse_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;

    for &x in values.iter().rev() {
        sum += x;
    }

    sum
}

fn pairwise_sum(values: &[f64]) -> f64 {
    let n = values.len();

    if n == 0 {
        0.0
    } else if n == 1 {
        values[0]
    } else {
        let mid = n / 2;
        pairwise_sum(&values[..mid]) + pairwise_sum(&values[mid..])
    }
}

fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;

    for &x in values {
        let y = x - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    sum
}

fn relative_error(approx: f64, reference: f64) -> f64 {
    let scale = reference.abs().max(f64::MIN_POSITIVE);
    (approx - reference).abs() / scale
}

fn demonstrate_non_associativity() -> (f64, f64) {
    let a = 1.0e16_f64;
    let b = -1.0e16_f64;
    let c = 1.0_f64;

    let left = (a + b) + c;
    let right = a + (b + c);

    (left, right)
}

fn demonstrate_fma_difference() -> (f64, f64) {
    let a = 1.0e16_f64;
    let b = 1.0000000000000002_f64;
    let c = -1.0e16_f64;

    let unfused = a * b + c;
    let fused = a.mul_add(b, c);

    (unfused, fused)
}

fn main() {
    println!("Program 22.2.2: Accumulation Order, Pairwise Summation, and Kahan Compensation");
    println!("=============================================================================\n");

    println!("Non-Associativity Diagnostic");
    println!("----------------------------");
    let (left, right) = demonstrate_non_associativity();

    println!("(a + b) + c = {:.16e}", left);
    println!("a + (b + c) = {:.16e}", right);
    println!("absolute difference = {:.16e}\n", (left - right).abs());

    println!("Summation Test Data");
    println!("-------------------");

    let small = 1.0e-8_f64;
    let n_small = 1_000_000_usize;

    let mut values = Vec::with_capacity(n_small + 2);
    values.push(1.0e8);

    for _ in 0..n_small {
        values.push(small);
    }

    values.push(-1.0e8);

    let reference = n_small as f64 * small;

    println!("large positive term = {:.16e}", 1.0e8_f64);
    println!("number of small terms = {}", n_small);
    println!("each small term = {:.16e}", small);
    println!("large negative term = {:.16e}", -1.0e8_f64);
    println!("reference sum = {:.16e}\n", reference);

    let seq = sequential_sum(&values);
    let rev = reverse_sum(&values);
    let pair = pairwise_sum(&values);
    let kahan = kahan_sum(&values);

    println!("Accumulation Strategy Comparison");
    println!("--------------------------------");
    println!(
        "{:<24} {:>22} {:>22}",
        "method", "computed sum", "relative error"
    );
    println!(
        "{:<24} {:>22.16e} {:>22.16e}",
        "sequential",
        seq,
        relative_error(seq, reference)
    );
    println!(
        "{:<24} {:>22.16e} {:>22.16e}",
        "reverse",
        rev,
        relative_error(rev, reference)
    );
    println!(
        "{:<24} {:>22.16e} {:>22.16e}",
        "pairwise",
        pair,
        relative_error(pair, reference)
    );
    println!(
        "{:<24} {:>22.16e} {:>22.16e}",
        "Kahan compensated",
        kahan,
        relative_error(kahan, reference)
    );

    println!("\nFused Multiply-Add Diagnostic");
    println!("-----------------------------");
    let (unfused, fused) = demonstrate_fma_difference();

    println!("a * b + c      = {:.16e}", unfused);
    println!("a.mul_add(b,c) = {:.16e}", fused);
    println!("absolute difference = {:.16e}", (unfused - fused).abs());

    println!("\nDiagnostic Interpretation");
    println!("-------------------------");
    println!("Different accumulation orders produce different rounding-error patterns.");
    println!("Pairwise summation reduces the effective accumulation depth.");
    println!("Kahan summation recovers some low-order information lost during addition.");
    println!("Fused multiply-add rounds only once and may differ from separate multiply-add evaluation.");
}
```

Program 22.2.2 demonstrates that floating-point summation is not solely a mathematical operation but also an algorithmic process whose outcome depends on the chosen accumulation strategy. Although all summation methods evaluate the same mathematical quantity defined by Equation (22.2.35), their numerical results differ because floating-point arithmetic introduces rounding at each intermediate step. The program therefore transforms the abstract discussion of non-associativity into a measurable computational phenomenon.

The comparison between sequential, reverse, pairwise, and compensated summation illustrates several important principles of floating-point error analysis. Sequential accumulation exhibits significant sensitivity to cancellation because small contributions may be absorbed into much larger partial sums. Pairwise summation reduces the effective depth of error accumulation by employing a balanced reduction tree, while Kahan compensation explicitly attempts to recover low-order information lost during rounding. The resulting improvements in accuracy demonstrate why accumulation order should be regarded as a critical algorithmic design choice rather than an implementation detail.

The fused multiply-add experiment further reinforces this point. Although the expressions $ab+c$ and $\operatorname{fma}(a,b,c)$ represent the same mathematical quantity, they correspond to different floating-point computations because the rounding process occurs at different stages. Consequently, hardware support for fused operations can alter bitwise results while simultaneously improving numerical accuracy. Such behavior illustrates why reproducibility diagnostics must account for compiler optimizations, hardware capabilities, and arithmetic execution modes.

Viewed more broadly, the program provides a practical framework for studying reproducibility in finite-precision computation. By comparing alternative reduction structures and arithmetic implementations, it reveals hidden sources of numerical variability that often remain invisible in high-level mathematical descriptions. These observations motivate the reproducibility tests developed later in Section 22.2.6 and establish a foundation for understanding how numerical software behaves in modern parallel and heterogeneous computing environments.

## 22.2.5. Mixed Precision and Floating-Point Exception Diagnostics

Modern numerical software increasingly uses more than one precision within a single algorithm. In a mixed-precision computation, some operations may be performed in low precision, while accumulation, correction, residual evaluation, or convergence testing may be performed in higher precision. This design treats precision as an algorithmic variable rather than a fixed property of the machine, a viewpoint emphasized in recent mixed-precision literature (Kashi et al., 2026).

A simple mixed-precision iterative-refinement model illustrates the principle. Suppose one wants to solve:

$$Ax=b \tag{22.2.48}$$

A low-precision factorization is used to obtain an approximate solution $x_k$, but the residual is computed in higher precision:

$$r_k = b-Ax_k \tag{22.2.49}$$

A correction equation is then solved approximately:

$$Ad_k = r_k \tag{22.2.50}$$

and the solution is updated by:

$$x_{k+1}=x_k+d_k \tag{22.2.51}$$

The success of this process depends on the conditioning of $A$, the low-precision unit roundoff, the accuracy of residual computation, and the stability of the correction solve. A simplified convergence requirement has the qualitative form:

$$\kappa(A)u_{\mathrm{low}} < 1 \tag{22.2.52}$$

although sharper conditions depend on the precise algorithm and precision hierarchy. The key diagnostic point is that one must know which operations are performed at which precision. A program that stores arrays in one precision, multiplies in another, accumulates in another, and evaluates stopping criteria in another has a floating-point behavior that cannot be inferred from a single type annotation.

Floating-point exceptions provide complementary diagnostics. Let an execution produce a sequence of arithmetic events:

$$\mathcal A = (a_1,a_2,\ldots,a_N) \tag{22.2.53}$$

Each event may be associated with an exception indicator vector:

$$\chi(a_i) = (\chi_{\mathrm{invalid}},\chi_{\mathrm{div0}},\chi_{\mathrm{overflow}},\chi_{\mathrm{underflow}},\chi_{\mathrm{inexact}})_i \tag{22.2.54}$$

where each component is either $0$ or $1$. The total exception profile of the run is:

$$\Xi = \sum_{i=1}^{N}\chi(a_i) \tag{22.2.55}$$

This vector does not replace numerical error analysis, but it identifies arithmetic events that may explain anomalous outputs. For example, a nonzero overflow count indicates that some intermediate quantity exceeded $x_{\max}$. A nonzero invalid-operation count may indicate operations such as $0/0$, $\infty-\infty$, or square roots of negative real values. Underflow may be harmless in some algorithms but catastrophic in others, particularly when small quantities carry essential scaling information.

Whole-program floating-point exception detection is therefore a practical part of numerical diagnostics. Recent work such as FloatGuard illustrates this direction by detecting floating-point exceptions in GPU programs, while compiler-level analyses such as CIRE study how precision choices and optimizations affect rounding-error behavior (Miao et al., 2025; Tirpankar et al., 2025). The general lesson is that modern floating-point diagnostics must operate at the level of the executed program, not only at the level of isolated formulas.

A useful diagnostic report for a floating-point computation should therefore include the following mathematical and operational information: the arithmetic type, the unit roundoff, the exponent range, the rounding mode, the treatment of subnormals, whether fused multiply-add is enabled, whether reassociation or fast-math optimizations are allowed, the precision used for accumulation, and the observed exception profile. These items define the effective arithmetic model under which the output should be interpreted.

## 22.2.6. Reproducibility Tests for Floating-Point Computations

The final layer is reproducibility testing. A floating-point computation should be tested not only for accuracy against a known answer, but also for stability under controlled changes in execution. Let,

$$\widehat y_j = \mathcal C(A,x,\theta,P_j),\qquad j=1,\ldots,m \tag{22.2.56}$$

denote outputs obtained from different platforms, compiler settings, thread counts, reduction orders, or precision configurations. A tolerance-based reproducibility metric is:

$$\rho = \max_{1\le j\le m}\frac{\|\widehat y_j-\widehat y_1\|}{\max(\|\widehat y_1\|,\sigma)} \tag{22.2.57}$$

where $\sigma>0$ prevents division by zero for small outputs. The computation is reproducible at tolerance $\tau$ if:

$$\rho\le \tau \tag{22.2.58}$$

The tolerance should be tied to problem conditioning and expected rounding error. If $F$ is the exact map and the problem has condition number $\kappa_F(x)$, then a rough first-order expectation is:

$$\frac{\|\Delta y\|}{\|y\|}\lesssim\kappa_F(x)\frac{\|\Delta x\|}{\|x\|}+\text{algorithmic rounding contribution} \tag{22.2.59}$$

When observed cross-platform variation greatly exceeds this scale, the difference should be treated as a diagnostic signal.

Another useful test compares different accumulation strategies. If,

$$\widehat s_{\mathrm{seq}},\quad\widehat s_{\mathrm{pair}},\quad\widehat s_{\mathrm{comp}},\quad\widehat s_{\mathrm{high}}\tag{22.2.60}$$

denote sequential, pairwise, compensated, and high-precision sums, respectively, then the quantities:

$$
\eta_{\mathrm{seq}}
=
\frac{\left|\widehat{s}_{\mathrm{seq}}-\widehat{s}_{\mathrm{high}}\right|}
{\left|\widehat{s}_{\mathrm{high}}\right|},
\qquad
\eta_{\mathrm{pair}}
=
\frac{\left|\widehat{s}_{\mathrm{pair}}-\widehat{s}_{\mathrm{high}}\right|}
{\left|\widehat{s}_{\mathrm{high}}\right|}
\tag{22.2.61}
$$

and

$$
\eta_{\mathrm{comp}}
=
\frac{\left|\widehat{s}_{\mathrm{comp}}-\widehat{s}_{\mathrm{high}}\right|}
{\left|\widehat{s}_{\mathrm{high}}\right|}
\tag{22.2.62}
$$

provide direct evidence of sensitivity to summation order. If these values differ substantially, the computation may require a prescribed reduction tree, compensated summation, higher-precision accumulation, or a tolerance that explicitly accounts for order-dependent roundoff.

For iterative solvers, reproducibility should be tested at the level of both final output and diagnostic trajectory. If $r_k^{(j)}$ is the residual at iteration $k$ in run $j$, one can compare residual histories through:

$$
d_{ij}
=
\max_k
\left|
\log_{10}\!\left\|r_k^{(i)}\right\|
-
\log_{10}\!\left\|r_k^{(j)}\right\|
\right|
\tag{22.2.63}
$$

A small final difference but a large trajectory difference may indicate that the algorithm converged to a similar answer through numerically different paths. This matters for debugging, performance tuning, and scientific interpretation.

The practical conclusion is that machine-parameter discovery, floating-point diagnostics, and reproducibility tests should be treated as a single methodology. Formal constants such as $u$, $x_{\min}$, and $x_{\max}$ describe the arithmetic envelope. Empirical tests reveal whether the runtime environment behaves as expected. Accumulation-order diagnostics expose hidden reduction structure. Exception monitoring identifies overflow, underflow, invalid operations, and other arithmetic events. Mixed-precision diagnostics determine which parts of the computation are accuracy-critical. Reproducibility tests then connect all these observations to the stability of the final result.

This section establishes the arithmetic foundation for the rest of the chapter. Section 22.3 builds on it by showing how diagnostic output and lightweight plotting make convergence, instability, and precision drift visible during execution. Later sections extend the same reliability principle to bit-level enumeration, data integrity, lossless compression, and arbitrary-precision arithmetic.

### Rust Implementation

Following the discussion in Sections 22.2.5 and 22.2.6 on mixed-precision arithmetic, floating-point exception diagnostics, and reproducibility testing, Program 22.2.3 provides a practical implementation that combines these ideas within a single computational framework. Modern numerical software frequently performs different stages of a computation at different precisions, using lower precision for storage or intermediate calculations and higher precision for residual evaluation, correction, and convergence assessment. At the same time, reliable numerical software must monitor exceptional arithmetic events and quantify the sensitivity of results to implementation choices such as accumulation order and reduction structure. This program demonstrates a mixed-precision iterative-refinement procedure, constructs a floating-point exception profile, and evaluates reproducibility metrics for multiple summation strategies. Together, these components provide a concrete illustration of the diagnostic methodology developed in Equations (22.2.48)–(22.2.63), showing how numerical accuracy, arithmetic behavior, and reproducibility can be examined within a unified experimental setting.

At the core of the implementation is the `ExceptionProfile` structure, which provides a compact representation of the floating-point exception profile introduced in Equations (22.2.53)–(22.2.55). The structure records counts associated with invalid operations, division-by-zero events, overflow, underflow, and ordinary finite arithmetic operations. Rather than treating exceptional values as isolated occurrences, the program aggregates them into a diagnostic summary that characterizes the arithmetic behavior of the entire computation.

The methods `record` and `print_summary` provide the operational interface for exception monitoring. The `record` method classifies each computed value according to its floating-point category and updates the corresponding counter within the exception profile. Values producing NaNs contribute to the invalid-operation count, infinite values contribute to overflow-related diagnostics, and values that collapse to zero through extreme scaling contribute to the underflow count. The `print_summary` method subsequently reports the aggregate exception statistics, providing a practical realization of the total exception profile described by Equation (22.2.55).

The functions `matvec_f64` and `residual_f64` implement the matrix-vector and residual computations required for mixed-precision iterative refinement. The residual function evaluates the quantity defined by Equation (22.2.49) using double-precision arithmetic. This higher-precision residual evaluation is one of the central ideas of mixed-precision algorithms because it allows errors introduced by lower-precision solution stages to be detected and corrected using more accurate arithmetic.

The function `solve_2x2_f32` performs the correction solve in single precision. Given a (2\\times2) system, it computes an approximate solution using explicit elimination formulas. Within the iterative-refinement framework, this function serves as the low-precision solver appearing in Equations (22.2.48)–(22.2.50). Although the example problem is intentionally small, the same conceptual structure appears in large-scale mixed-precision algorithms where low-precision factorizations are combined with higher-precision residual computations.

The `norm2` function computes the Euclidean norm of a vector and is used throughout the implementation to monitor residual magnitude and solution error. Residual norms provide a direct measure of convergence during iterative refinement and form the basis for the residual trajectories discussed later in Equation (22.2.63). By recording these norms at each refinement step, the program makes the convergence process observable rather than treating the final solution as the only relevant output.

The `mixed_precision_refinement` function implements the complete refinement procedure described conceptually by Equations (22.2.48)–(22.2.52). An initial solution is first obtained using single-precision arithmetic. Residuals are then computed in double precision, correction equations are solved approximately in single precision, and the solution is updated in higher precision. The function also records the residual norm at each iteration, thereby producing a convergence history that can be analyzed after execution. This implementation demonstrates how precision can be treated as an algorithmic variable rather than as a fixed property of the machine.

The functions `sequential_sum`, `pairwise_sum`, and `kahan_sum` provide three distinct accumulation strategies corresponding to the summation diagnostics discussed in Equations (22.2.60)–(22.2.62). The sequential implementation performs conventional left-to-right accumulation, the pairwise implementation uses a balanced reduction tree, and the Kahan implementation employs compensated summation to reduce the loss of low-order information. Comparing the outputs of these methods provides direct evidence of sensitivity to accumulation order and reduction structure.

The `reproducibility_metric` function implements the tolerance-based reproducibility measure introduced in Equations (22.2.56)–(22.2.58). Given a collection of outputs obtained from different computational strategies, the function computes the maximum normalized deviation relative to a reference result. This metric transforms reproducibility from a qualitative observation into a quantitative diagnostic quantity that can be compared against a prescribed tolerance.

The `relative_error` function evaluates the normalized difference between a computed result and a reference value. It is used when comparing summation strategies against a higher-quality reference approximation and provides the error measures corresponding to Equations (22.2.61) and (22.2.62). These diagnostics make it possible to assess not only whether different implementations agree, but also how far each implementation deviates from the expected numerical result.

The `main` function integrates all diagnostic components into a single computational experiment. It first performs mixed-precision iterative refinement on a mildly ill-conditioned linear system and records the residual history. It then evaluates several exceptional arithmetic operations to construct an exception profile. Finally, it compares sequential, pairwise, and compensated summation strategies and computes a reproducibility metric. The resulting output illustrates how mixed-precision behavior, floating-point exceptions, accumulation-order sensitivity, and reproducibility diagnostics can be studied simultaneously within a unified numerical framework.

```rust
// Program 22.2.3: Mixed-Precision Refinement and Floating-Point Exception Diagnostics
//
// Problem statement:
// Demonstrate mixed-precision iterative refinement for a mildly ill-conditioned
// linear system, record floating-point exceptional values, and compare
// reproducibility across different accumulation strategies.

#[derive(Default)]
struct ExceptionProfile {
    invalid: usize,
    div_zero: usize,
    overflow: usize,
    underflow: usize,
    finite: usize,
}

impl ExceptionProfile {
    fn record(&mut self, label: &str, value: f64) {
        if value.is_nan() {
            self.invalid += 1;
            println!("{:<30} = {:>22}  [invalid / NaN]", label, value);
        } else if value.is_infinite() {
            self.overflow += 1;
            println!("{:<30} = {:>22}  [overflow / infinity]", label, value);
        } else if value == 0.0 {
            self.underflow += 1;
            println!("{:<30} = {:>22.16e}  [possible underflow to zero]", label, value);
        } else {
            self.finite += 1;
            println!("{:<30} = {:>22.16e}  [finite]", label, value);
        }
    }

    fn print_summary(&self) {
        println!("\nFloating-Point Exception Profile");
        println!("--------------------------------");
        println!("invalid / NaN events       = {}", self.invalid);
        println!("division-by-zero events    = {}", self.div_zero);
        println!("overflow / infinity events = {}", self.overflow);
        println!("underflow-to-zero events   = {}", self.underflow);
        println!("finite arithmetic events   = {}", self.finite);
    }
}

fn matvec_f64(a: [[f64; 2]; 2], x: [f64; 2]) -> [f64; 2] {
    [
        a[0][0] * x[0] + a[0][1] * x[1],
        a[1][0] * x[0] + a[1][1] * x[1],
    ]
}

fn residual_f64(a: [[f64; 2]; 2], x: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let ax = matvec_f64(a, x);
    [b[0] - ax[0], b[1] - ax[1]]
}

fn solve_2x2_f32(a: [[f32; 2]; 2], b: [f32; 2]) -> [f32; 2] {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];

    [
        (b[0] * a[1][1] - a[0][1] * b[1]) / det,
        (a[0][0] * b[1] - b[0] * a[1][0]) / det,
    ]
}

fn norm2(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn mixed_precision_refinement(
    a64: [[f64; 2]; 2],
    b64: [f64; 2],
    max_iter: usize,
) -> ([f64; 2], Vec<f64>) {
    let a32 = [
        [a64[0][0] as f32, a64[0][1] as f32],
        [a64[1][0] as f32, a64[1][1] as f32],
    ];

    let b32 = [b64[0] as f32, b64[1] as f32];

    let x_initial = solve_2x2_f32(a32, b32);
    let mut x64 = [x_initial[0] as f64, x_initial[1] as f64];

    let mut residual_history = Vec::new();

    for _ in 0..max_iter {
        let r64 = residual_f64(a64, x64, b64);
        residual_history.push(norm2(r64));

        let d32 = solve_2x2_f32(a32, [r64[0] as f32, r64[1] as f32]);

        x64[0] += d32[0] as f64;
        x64[1] += d32[1] as f64;
    }

    let final_residual = residual_f64(a64, x64, b64);
    residual_history.push(norm2(final_residual));

    (x64, residual_history)
}

fn sequential_sum(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |sum, &x| sum + x)
}

fn pairwise_sum(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else if values.len() == 1 {
        values[0]
    } else {
        let mid = values.len() / 2;
        pairwise_sum(&values[..mid]) + pairwise_sum(&values[mid..])
    }
}

fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;

    for &x in values {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    sum
}

fn reproducibility_metric(outputs: &[f64], sigma: f64) -> f64 {
    let y0 = outputs[0];
    let scale = y0.abs().max(sigma);

    outputs
        .iter()
        .map(|&y| (y - y0).abs() / scale)
        .fold(0.0_f64, f64::max)
}

fn relative_error(value: f64, reference: f64) -> f64 {
    (value - reference).abs() / reference.abs().max(f64::MIN_POSITIVE)
}

fn main() {
    println!("Program 22.2.3: Mixed-Precision Refinement and Floating-Point Exception Diagnostics");
    println!("==================================================================================\n");

    println!("Mixed-Precision Iterative Refinement");
    println!("------------------------------------");

    let a64 = [
        [1.0_f64, 0.99_f64],
        [0.99_f64, 0.9802_f64],
    ];

    let x_exact = [1.23456789012345_f64, -0.98765432198765_f64];
    let b64 = matvec_f64(a64, x_exact);

    let (x_refined, residual_history) = mixed_precision_refinement(a64, b64, 5);

    println!(
        "exact solution       = [{:.16e}, {:.16e}]",
        x_exact[0], x_exact[1]
    );
    println!(
        "refined solution     = [{:.16e}, {:.16e}]",
        x_refined[0], x_refined[1]
    );

    println!("\nResidual History");
    println!("----------------");
    for (k, rnorm) in residual_history.iter().enumerate() {
        println!("step {:>2}: ||r_k||_2 = {:.16e}", k, rnorm);
    }

    let final_error = norm2([x_refined[0] - x_exact[0], x_refined[1] - x_exact[1]]);
    println!("\nfinal solution error ||x_hat - x||_2 = {:.16e}\n", final_error);

    println!("Floating-Point Exceptional-Value Diagnostics");
    println!("--------------------------------------------");

    let mut profile = ExceptionProfile::default();

    let overflow = f64::MAX * 2.0;
    let div_zero = 1.0_f64 / 0.0_f64;
    let invalid = 0.0_f64 / 0.0_f64;
    let underflow = f64::MIN_POSITIVE * f64::EPSILON * 0.5;
    let finite = (1.0_f64 + f64::EPSILON) - 1.0_f64;

    profile.record("f64::MAX * 2.0", overflow);

    if div_zero.is_infinite() {
        profile.div_zero += 1;
        println!("{:<30} = {:>22}  [division by zero]", "1.0 / 0.0", div_zero);
    }

    profile.record("0.0 / 0.0", invalid);
    profile.record("tiny product", underflow);
    profile.record("(1 + eps) - 1", finite);
    profile.print_summary();

    println!("\nReproducibility Test Across Reduction Strategies");
    println!("------------------------------------------------");

    let small = 1.0e-8_f64;
    let n_small = 1_000_000_usize;

    let mut values = Vec::with_capacity(n_small + 2);
    values.push(1.0e8);

    for _ in 0..n_small {
        values.push(small);
    }

    values.push(-1.0e8);

    let high_reference = n_small as f64 * small;

    let seq = sequential_sum(&values);
    let pair = pairwise_sum(&values);
    let comp = kahan_sum(&values);

    let outputs = [seq, pair, comp];
    let rho = reproducibility_metric(&outputs, 1.0e-30);

    println!(
        "{:<24} {:>22} {:>22}",
        "method", "computed sum", "relative error"
    );
    println!(
        "{:<24} {:>22.16e} {:>22.16e}",
        "sequential",
        seq,
        relative_error(seq, high_reference)
    );
    println!(
        "{:<24} {:>22.16e} {:>22.16e}",
        "pairwise",
        pair,
        relative_error(pair, high_reference)
    );
    println!(
        "{:<24} {:>22.16e} {:>22.16e}",
        "Kahan compensated",
        comp,
        relative_error(comp, high_reference)
    );

    println!("\nreproducibility metric rho = {:.16e}", rho);
    println!("diagnostic tolerance tau   = {:.16e}", 1.0e-6_f64);
    println!(
        "reproducibility status     = {}",
        if rho <= 1.0e-6 {
            "within tolerance"
        } else {
            "outside tolerance"
        }
    );
}
```

Program 22.2.3 demonstrates that floating-point diagnostics extend beyond the determination of machine parameters and must also encompass the dynamic behavior of numerical algorithms during execution. By combining mixed-precision iterative refinement, exception profiling, and reproducibility analysis, the program illustrates how numerical computations can be monitored at multiple levels simultaneously. This integrated perspective reflects the central theme of Sections 22.2.5 and 22.2.6: understanding a numerical result requires understanding the arithmetic environment that produced it.

The mixed-precision refinement experiment illustrates the practical use of Equations (22.2.48)–(22.2.52). Low-precision arithmetic is used where efficiency is desirable, while higher-precision residual evaluation provides the information necessary to improve solution quality. The observed reduction in residual norms demonstrates how iterative refinement can compensate for errors introduced during lower-precision computations and thereby recover much of the accuracy associated with higher-precision arithmetic.

The floating-point exception profile provides complementary information about the arithmetic events encountered during execution. Overflow, underflow, division-by-zero operations, and invalid arithmetic expressions each reveal different aspects of the numerical environment. Although these events do not directly measure solution error, they often provide valuable clues when diagnosing instability, unexpected outputs, or implementation defects. The aggregation of exception statistics therefore serves as an important component of whole-program numerical diagnostics.

The reproducibility experiments demonstrate that different accumulation strategies may produce substantially different numerical results even when evaluating the same mathematical expression. The comparison among sequential, pairwise, and compensated summation highlights the influence of reduction structure and rounding-error propagation. The reproducibility metric further transforms these observations into a quantitative measure that can be compared against prescribed tolerances, thereby providing an objective basis for evaluating numerical consistency across implementations.

Taken together, the diagnostics presented in this program establish the final layer of the floating-point analysis developed throughout Section 22.2. Machine parameters describe the arithmetic envelope, exception monitoring characterizes arithmetic events, mixed-precision analysis identifies accuracy-critical stages of a computation, and reproducibility metrics quantify implementation sensitivity. These concepts form the foundation for the diagnostic and visualization techniques introduced in Section 22.3, where numerical behavior becomes observable through structured reporting and lightweight plotting.

# 22.3. Diagnostic Output and Lightweight Plotting

Diagnostic output is the part of a numerical computation that explains how a result was obtained. A final scalar value, vector, image, mesh, or table is rarely sufficient by itself. A trustworthy computation should also report residuals, error estimates, iteration counts, stopping criteria, detected exceptions, convergence histories, scaling information, and enough metadata to make the result interpretable after the computation has finished. In this sense, diagnostic output is not merely a convenience for debugging. It is part of the numerical evidence supporting the computation.

Lightweight plotting plays the same role visually. A semilog residual plot, a convergence trace, a histogram of errors, or a plot of local step sizes can reveal stagnation, instability, outliers, non-random residual structure, precision drift, or implementation mistakes that may not be visible from a final number. The recent literature supports this treatment of plots as diagnostic instruments rather than presentation artifacts. Residual diagnostics must be interpreted in terms of meaningful deviations, diagnostic charts should be deliberately minimal and comparable across runs, and logs, plots, metadata, and workflow states should be treated as part of the reproducible computational record (Li et al., 2024; Romero-Organvidez et al., 2024; Stoudt et al., 2024).

## 22.3.1. Diagnostic Quantities in Numerical Algorithms

A diagnostic quantity is a computed scalar, vector, or structured record that helps determine whether a numerical computation is behaving as intended. For a problem written abstractly as:

$$F(x)=0 \tag{22.3.1}$$

an iterative algorithm produces approximations,

$$x_0,x_1,x_2,\ldots,x_k,\ldots \tag{22.3.2}$$

The most basic diagnostic is the residual:

$$r_k = F(x_k) \tag{22.3.3}$$

A residual norm sequence,

$$\rho_k = \|r_k\|\tag{22.3.4}$$

gives a compact description of progress. A method is usually considered convergent only when $\rho_k$ becomes small relative to an absolute or relative tolerance. A typical mixed stopping rule has the form:

$$\|F(x_k)\|\le\tau_{\mathrm{abs}}+\tau_{\mathrm{rel}}\|F(x_0)\| \tag{22.3.5}$$

where $\tau_{\mathrm{abs}}$ controls absolute accuracy and $\tau_{\mathrm{rel}}$ controls reduction relative to the initial residual. This form avoids declaring failure merely because the initial residual is large, and it avoids declaring success merely because the scale of the problem is small.

For fixed-point iterations:

$$x_{k+1}=G(x_k) \tag{22.3.6}$$

another useful diagnostic is the step difference,

$$d_k=\|x_{k+1}-x_k\| \tag{22.3.7}$$

If $x_k\to x^\ast$ and $G$ is locally contractive, then one expects:

$$\|x_{k+1}-x^\ast\|\leq\|x_k-x^\ast\|,\qquad 0<q<1 \tag{22.3.8}$$

at least asymptotically. The observed convergence factor may be estimated by:

$$\widehat q_k = \frac{\|x_{k+1}-x_k\|}{\|x_k-x_{k-1}\|},\tag{22.3.9}$$

provided the denominator is nonzero. If $\widehat q_k$ remains close to $1$, convergence is slow. If $\widehat q_k>1$, the iteration may be diverging. If $\widehat q_k$ oscillates, the iteration may be affected by nonnormality, poor scaling, discontinuous updates, or nonlinear instability.

For a linear system:

$$Ax=b \tag{22.3.10}$$

the residual at an approximate solution $x_k$ is:

$$r_k=b-Ax_k \tag{22.3.11}$$

The residual is easy to compute, but it must be interpreted carefully. Since,

$$A(x-x_k)=r_k \tag{22.3.12}$$

one has,

$$x-x_k=A^{-1}r_k \tag{22.3.13}$$

and therefore,

$$\|x-x_k\|\le\|A^{-1}\|\,\|r_k\| \tag{22.3.14}$$

Dividing by $\|x\|$ and using $b=Ax$ gives a relative forward-error estimate of the form:

$$\frac{\|x-x_k\|}{\|x\|}\le\kappa(A)\frac{\|r_k\|}{\|b\|}\frac{\|b\|}{\|A\|\|x\|} \tag{22.3.15}$$

where,

$$\kappa(A)=\|A\|\,\|A^{-1}\|\tag{22.3.16}$$

is the condition number. This inequality shows why residual diagnostics must not be interpreted mechanically. A small residual may still correspond to a large forward error if the problem is ill-conditioned. Conversely, a moderate residual may be acceptable when the problem scale and conditioning make further reduction meaningless in finite precision.

For optimization problems,

$$\min_x \phi(x) \tag{22.3.17}$$

common diagnostics include objective values , gradient norms,

$$g_k=\|\nabla \phi(x_k)\| \tag{22.3.18}$$

step lengths $\alpha_k$, model reductions, constraint violations, and stationarity measures. For constrained problems with equality constraints,

$$c(x)=0 \tag{22.3.19}$$

a basic feasibility diagnostic is:

$$\eta_k=\|c(x_k)\| \tag{22.3.20}$$

A computation that reduces the objective while increasing $\eta_k$ may not be converging to a meaningful constrained solution. Thus, diagnostic output should report all mathematically relevant components of the stopping condition, not only the most favorable one.

## 22.3.2. Convergence Traces and Semilog Error Plots

The most useful diagnostic plots are often the simplest. For many algorithms, the sequence of residuals or errors is expected to decay approximately geometrically. If,

$$e_k = \|x_k-x^\ast\|\tag{22.3.21}$$

and

$$e_k \approx Cq^k,\qquad 0<q<1 \tag{22.3.22}$$

then

$$\log_{10} e_k\approx\log_{10} C+k\log_{10} q \tag{22.3.23}$$

Thus, a plot of $\log_{10} e_k$ against $k$ should appear approximately linear in the asymptotic convergence regime. The slope estimates the convergence rate. If the slope flattens, the iteration may have reached the roundoff floor, encountered stagnation, or entered a regime where the theoretical convergence assumptions no longer hold.

When the exact solution is unknown, residuals replace errors. A semilog residual trace plots:

$$s_k=\log_{10}\|r_k\| \tag{22.3.24}$$

The successive slope,

$$\Delta s_k=s_{k+1}-s_k\tag{22.3.25}$$

measures the logarithmic reduction per iteration. If $\Delta s_k<0$, the residual is decreasing. If $\Delta s_k\approx 0$, the method is stagnating. If $\Delta s_k>0$, the residual is increasing. A normalized reduction factor is:

$$\mu_k = \frac{\|r_{k+1}\|}{\|r_k\|} \tag{22.3.26}$$

A value $\mu_k<1$ indicates residual reduction, while $\mu_k\ge 1$ indicates failure to reduce the residual at that step. In practice, one expects occasional nonmonotonicity in nonlinear methods, Krylov solvers with restarts, stochastic algorithms, and adaptive time-stepping methods. The diagnostic question is not whether every step improves, but whether the overall trend is consistent with the intended algorithmic behavior.

For a method with expected algebraic convergence,

$$e_k\approx Ck^{-\alpha} \tag{22.3.27}$$

a log-log plot is more appropriate:

$$\log e_k\approx\log C-\alpha \log k \tag{22.3.28}$$

The slope estimates the algebraic rate $\alpha$. This is common in mesh-refinement studies, Monte Carlo methods, quadrature error studies, and spectral approximations before the asymptotic exponential or roundoff-limited regime is reached.

For discretization error, suppose a numerical approximation $y_h$ depends on a mesh width $h$ and satisfies:

$$\|y_h-y\| = C h^p + o(h^p) \tag{22.3.29}$$

Given two mesh widths $h$ and $h/2$, an observed order can be estimated by:

$$\widehat p = \log_2\left(\frac{\|y_h-y\|}{\|y_{h/2}-y\|}\right) \tag{22.3.30}$$

when the exact solution $y$ is known. If $y$ is unknown, a three-level estimate may be used:

$$\widehat p = \log_2\left(\frac{\|y_h-y_{h/2}\|}{\|y_{h/2}-y_{h/4}\|}\right) \tag{22.3.31}$$

A lightweight convergence plot of $\log \|y_h-y_{h/2}\|$ against $\log h$ can reveal whether the expected discretization regime has been reached. If the slope does not match the predicted order, possible causes include insufficient resolution, boundary singularities, inconsistent boundary conditions, roundoff saturation, or an implementation error.

In stochastic computation, the diagnostic trace is different. If $\widehat I_N$ is a Monte Carlo estimator based on $N$ samples and,

$$\operatorname{Var}(\widehat I_N)=\frac{\sigma^2}{N} \tag{22.3.32}$$

then the root-mean-square error scales as,

$$\operatorname{RMSE}(\widehat I_N)=O(N^{-1/2}) \tag{22.3.33}$$

A log-log plot of empirical error or confidence interval width against $N$ should therefore have slope close to $-1/2$, assuming independent sampling and finite variance. A substantially different slope may indicate correlation, biased sampling, heavy-tailed variance, or an incorrect estimator.

The purpose of these plots is not aesthetic. Their purpose is to expose whether the observed numerical behavior agrees with the mathematical behavior expected from the algorithm. This agrees with the modern view that residual plots, convergence curves, semilog traces, and histogram-based checks are algorithmic diagnostic instruments, especially when interpreted as practically meaningful deviations rather than only as formal pass-fail tests (Li et al., 2024).

## 22.3.3. Structured Logs, Metadata, and Reproducible Output

A diagnostic plot is useful only if it can be connected to the computation that produced it. This requires structured output. A numerical run should produce a record containing at least the problem definition, algorithmic parameters, machine and software environment, stopping criteria, diagnostic histories, and final status. Abstractly, one may write the run record as:

$$\mathcal R = (D,\theta,P,Y,\mathcal D,\mathcal S) \tag{22.3.34}$$

where $D$ denotes input data, $\theta$ algorithmic parameters, $P$ platform information, $Y$ final output, $\mathcal D$ diagnostic traces, and $\mathcal S$ status information. The diagnostic component may include sequences such as,

$$\mathcal D = {(k,\|r_k\|,\|x_{k+1}-x_k\|,\alpha_k,t_k)}_{k=0}^{K},\tag{22.3.35}$$

where $t_k$ is elapsed time or another cost measure. A status component may include indicators such as:

$$\mathcal S = (\text{converged},K,\tau_{\mathrm{abs}},\tau_{\mathrm{rel}},\text{reason}) \tag{22.3.36}$$

where the final entry records whether termination was due to convergence, iteration limit, step-size failure, invalid arithmetic, user interruption, or another condition.

Structured logs are superior to unstructured text because they can be parsed, compared, plotted, and tested. For example, a residual trace stored as pairs $(k,\rho_k)$ can be compared across machines by a reproducibility metric such as,

$$d_{ij} = \max_{0\le k\le K}\left|\log_{10}\rho_k^{(i)} - \log_{10}\rho_k^{(j)}\right| \tag{22.3.37}$$

where $\rho_k^{(i)}$ and $\rho_k^{(j)}$ are residual norms from two runs. If $d_{ij}$ is small, the convergence histories are similar on a logarithmic scale. If $d_{ij}$ is large, then the runs may have followed different numerical trajectories even if the final outputs are close.

For final outputs $Y_i$ and $Y_j$, a scale-aware comparison may use:

$$\delta_{ij} = \frac{\|Y_i-Y_j\|}{\max(\|Y_i\|,\|Y_j\|,\sigma)} \tag{22.3.38}$$

where $\sigma>0$ prevents division by zero. A computation may be considered reproducible at tolerance $\tau$ if,

$$\delta_{ij}\le \tau\tag{22.3.39}$$

for all relevant pairs of runs. However, equation (22.3.39) should not be used alone. Two runs may produce similar final values but very different residual histories, step sizes, or exception profiles. Conversely, different final values may be acceptable if the problem is ill-conditioned and the observed difference is consistent with the expected conditioning.

Metadata are therefore part of the diagnostic output. A reproducible numerical record should identify the arithmetic type, precision, compiler settings, random seed, thread count, tolerance values, input data version, and relevant library versions. If the computation involves randomness, the record should include the pseudo-random generator and seed:

$$\omega = (\text{generator}, \text{seed}, \text{stream id}) \tag{22.3.40}$$

If the computation involves adaptive algorithms, it should include the adaptation history. For adaptive time stepping, for example, one may store:

$$\{(t_k,h_k,\epsilon_k,\text{accepted}_k)\}_{k=0}^{K} \tag{22.3.41}$$

where $h_k$ is the proposed step size and $\epsilon_k$ is the estimated local error. A plot of $h_k$ against $t_k$ can reveal stiffness, discontinuities, singularities, or overly conservative error control.

This structured view is consistent with the responsible-workflow literature, which treats output, logs, metadata, plots, and workflow states as parts of the computational record rather than temporary by-products (Stoudt et al., 2024). It is also consistent with visualization-design guidance that emphasizes standardized, configurable, and comparable chart construction rather than ad hoc plotting choices (Romero-Organvidez et al., 2024).

### Rust Implementation

Following the discussion in Section 22.3.3 on structured logs, metadata, and reproducible output, Program 22.3.1 provides a practical implementation of diagnostic logging for an iterative numerical method. Modern numerical software must do more than compute a final answer; it must also record sufficient information to explain how that answer was obtained. This program demonstrates a structured diagnostic framework for fixed-point iteration in which residual norms, step differences, convergence estimates, timing information, stopping criteria, and execution metadata are recorded throughout the computation. By transforming the iterative process into a sequence of observable diagnostic events, the implementation illustrates how the reporting principles developed in Section 22.3 can be incorporated directly into numerical software. The resulting diagnostic log provides both a numerical solution and a detailed computational record that can be used for verification, performance assessment, and reproducibility analysis.

At the core of the implementation is the `RunMetadata` structure, which stores descriptive information about the computational experiment. This includes the mathematical problem being solved, the numerical method employed, the arithmetic type, iteration limits, and convergence tolerances. The structure reflects the metadata concepts introduced in Section 22.3.3, where numerical results are accompanied by contextual information that allows the computation to be reproduced and interpreted correctly.

The `DiagnosticEntry` structure defines a single record within the diagnostic log. Each entry stores the iteration index, current iterate, residual norm, step difference, estimated convergence factor, and elapsed execution time. Collectively, these quantities form a structured representation of the iterative history and provide the diagnostic information required to analyze convergence behavior. Rather than storing only the final solution, the implementation records the complete computational trajectory.

The `RunStatus` structure records the outcome of the computation. It stores whether convergence was achieved, the number of iterations performed, and the reason for termination. This explicit status reporting supports the reproducibility principles discussed in Section 22.3 by ensuring that the stopping condition is documented as part of the computational record.

The function `fixed_point_map` implements the nonlinear mapping $x \mapsto \cos(x)$ used in the demonstration problem. The associated `residual` function evaluates the fixed-point residual corresponding to the difference between the current iterate and the image of the mapping. This residual serves as the primary convergence diagnostic throughout the computation and corresponds to the residual-monitoring concepts discussed in Equations (22.3.1)–(22.3.6).

The `stopping_threshold` function constructs a mixed absolute-relative convergence criterion. Rather than relying solely on an absolute tolerance, the threshold incorporates both absolute and relative components, thereby providing a more robust stopping rule across different problem scales. This approach reflects the convergence-testing strategies developed in the section and helps prevent premature termination or unnecessary iterations.

The `fixed_point_solve` function performs the iterative computation and serves as the central component of the program. During each iteration, the current residual norm, step difference, convergence estimate, and elapsed execution time are recorded in a new diagnostic entry. The function also evaluates convergence criteria and checks for invalid arithmetic conditions. By collecting these quantities into a structured log, the implementation transforms the iterative process into an observable sequence of diagnostic measurements. The convergence factor is estimated from successive step differences and provides an empirical indication of the asymptotic convergence behavior of the iteration.

The functions `print_metadata`, `print_diagnostics`, and `print_status` are responsible for transforming the stored diagnostic information into a human-readable report. `print_metadata` documents the computational environment and numerical parameters, `print_diagnostics` outputs the iteration-by-iteration convergence history, and `print_status` summarizes the final outcome of the run. Together, these functions illustrate how structured logging can be integrated into numerical software without complicating the underlying algorithm.

The `main` function serves as a complete demonstration of structured diagnostic reporting. It begins by defining the computational metadata, including convergence tolerances and iteration limits. The fixed-point solver is then executed, producing a detailed diagnostic history. Finally, the metadata, diagnostic log, run status, and final solution are reported. The resulting output provides a comprehensive computational record that can be inspected long after the numerical calculation itself has completed.

```rust
// Program 22.3.1: Structured Diagnostic Logs for Fixed-Point Iteration
//
// Problem statement:
// Demonstrate structured diagnostic output for an iterative numerical method.
// The program records residual norms, step differences, observed convergence
// factors, stopping criteria, metadata, and final status for a fixed-point solve.

use std::time::Instant;

#[derive(Debug)]
struct RunMetadata {
    problem: &'static str,
    method: &'static str,
    arithmetic_type: &'static str,
    max_iterations: usize,
    tau_abs: f64,
    tau_rel: f64,
}

#[derive(Debug)]
struct DiagnosticEntry {
    iteration: usize,
    x: f64,
    residual_norm: f64,
    step_difference: f64,
    convergence_factor: Option<f64>,
    elapsed_ms: f64,
}

#[derive(Debug)]
struct RunStatus {
    converged: bool,
    iterations: usize,
    reason: &'static str,
}

fn fixed_point_map(x: f64) -> f64 {
    x.cos()
}

fn residual(x: f64) -> f64 {
    fixed_point_map(x) - x
}

fn stopping_threshold(initial_residual: f64, tau_abs: f64, tau_rel: f64) -> f64 {
    tau_abs + tau_rel * initial_residual.abs()
}

fn fixed_point_solve(metadata: &RunMetadata, x0: f64) -> (Vec<DiagnosticEntry>, RunStatus) {
    let timer = Instant::now();

    let initial_residual = residual(x0).abs();
    let threshold = stopping_threshold(initial_residual, metadata.tau_abs, metadata.tau_rel);

    let mut diagnostics: Vec<DiagnosticEntry> = Vec::new();

    let mut x_prev_prev: Option<f64> = None;
    let mut x_prev = x0;

    for k in 0..=metadata.max_iterations {
        let r_norm = residual(x_prev).abs();

        let step_difference = match x_prev_prev {
            Some(old) => (x_prev - old).abs(),
            None => 0.0,
        };

        let convergence_factor = if diagnostics.len() >= 2 {
            let d_k = step_difference;
            let d_prev = diagnostics[diagnostics.len() - 1].step_difference;

            if d_prev > 0.0 {
                Some(d_k / d_prev)
            } else {
                None
            }
        } else {
            None
        };

        diagnostics.push(DiagnosticEntry {
            iteration: k,
            x: x_prev,
            residual_norm: r_norm,
            step_difference,
            convergence_factor,
            elapsed_ms: timer.elapsed().as_secs_f64() * 1000.0,
        });

        if r_norm <= threshold {
            return (
                diagnostics,
                RunStatus {
                    converged: true,
                    iterations: k,
                    reason: "residual satisfied mixed absolute-relative tolerance",
                },
            );
        }

        if !x_prev.is_finite() || !r_norm.is_finite() {
            return (
                diagnostics,
                RunStatus {
                    converged: false,
                    iterations: k,
                    reason: "invalid arithmetic detected",
                },
            );
        }

        if k == metadata.max_iterations {
            break;
        }

        let x_next = fixed_point_map(x_prev);
        x_prev_prev = Some(x_prev);
        x_prev = x_next;
    }

    (
        diagnostics,
        RunStatus {
            converged: false,
            iterations: metadata.max_iterations,
            reason: "maximum iteration count reached",
        },
    )
}

fn print_metadata(metadata: &RunMetadata, x0: f64) {
    println!("Run Metadata");
    println!("------------");
    println!("problem          = {}", metadata.problem);
    println!("method           = {}", metadata.method);
    println!("arithmetic type  = {}", metadata.arithmetic_type);
    println!("initial value    = {:.16e}", x0);
    println!("max iterations   = {}", metadata.max_iterations);
    println!("tau_abs          = {:.3e}", metadata.tau_abs);
    println!("tau_rel          = {:.3e}", metadata.tau_rel);
}

fn print_diagnostics(diagnostics: &[DiagnosticEntry]) {
    println!("\nStructured Diagnostic Log");
    println!("-------------------------");
    println!(
        "{:>6} {:>22} {:>22} {:>22} {:>18} {:>14}",
        "k", "x_k", "||r_k||", "||x_k-x_{k-1}||", "q_hat", "time_ms"
    );

    for entry in diagnostics {
        let q_text = match entry.convergence_factor {
            Some(q) => format!("{:.8e}", q),
            None => String::from("NA"),
        };

        println!(
            "{:>6} {:>22.16e} {:>22.16e} {:>22.16e} {:>18} {:>14.6}",
            entry.iteration,
            entry.x,
            entry.residual_norm,
            entry.step_difference,
            q_text,
            entry.elapsed_ms
        );
    }
}

fn print_status(status: &RunStatus) {
    println!("\nRun Status");
    println!("----------");
    println!("converged  = {}", status.converged);
    println!("iterations = {}", status.iterations);
    println!("reason     = {}", status.reason);
}

fn main() {
    println!("Program 22.3.1: Structured Diagnostic Logs for Fixed-Point Iteration");
    println!("====================================================================\n");

    let metadata = RunMetadata {
        problem: "solve x = cos(x)",
        method: "fixed-point iteration",
        arithmetic_type: "f64",
        max_iterations: 100,
        tau_abs: 1.0e-12,
        tau_rel: 1.0e-10,
    };

    let x0 = 1.0_f64;

    print_metadata(&metadata, x0);

    let (diagnostics, status) = fixed_point_solve(&metadata, x0);

    print_diagnostics(&diagnostics);
    print_status(&status);

    if let Some(last) = diagnostics.last() {
        println!("\nFinal Output");
        println!("------------");
        println!("x_hat       = {:.16e}", last.x);
        println!("final ||r|| = {:.16e}", last.residual_norm);
    }
}
```

Program 22.3.1 demonstrates how structured diagnostic logging can transform a numerical computation into a reproducible computational experiment. Rather than reporting only the final solution, the implementation records the evolution of residual norms, step differences, convergence estimates, execution times, and stopping conditions throughout the iterative process. This richer representation provides valuable insight into the behavior of the algorithm and enables independent verification of the reported results.

The fixed-point iteration example illustrates how convergence information can be collected systematically during execution. The residual history reveals the rate at which the iteration approaches the fixed point, while the observed convergence factors provide empirical evidence of the asymptotic convergence behavior predicted by theory. Such information is often more valuable than the final solution alone because it explains why the algorithm succeeded and how efficiently it reached convergence.

The structured logging framework also demonstrates the importance of metadata and explicit status reporting. Recording tolerances, iteration limits, arithmetic types, and stopping reasons ensures that computational results remain interpretable and reproducible. This information becomes particularly important in large-scale numerical software, where results may need to be validated, compared across systems, or reproduced months or years after the original computation.

More broadly, the program establishes the foundation for the diagnostic and visualization techniques developed throughout Section 22.3. Once diagnostic information is represented in a structured form, it can be exported to files, analyzed statistically, visualized graphically, or incorporated into automated monitoring systems. The same principles extend naturally to nonlinear solvers, optimization algorithms, differential-equation integrators, and large-scale scientific simulations. Structured diagnostic output therefore serves as a critical bridge between numerical computation and numerical observability.

## 22.3.4. Lightweight Plotting Patterns for Numerical Software

Lightweight plotting means using small, standardized plots to detect numerical behavior early and repeatedly. These plots should be easy to generate, easy to compare across runs, and tied directly to mathematical diagnostics. The goal is not to produce publication graphics at every stage. The goal is to reveal whether the computation is behaving as expected.

A first pattern is the semilog residual plot:

$$k \mapsto \log_{10}\|r_k\| \tag{22.3.42}$$

This plot is appropriate for iterative solvers, nonlinear equations, eigenvalue algorithms, optimization methods, and time-stepping residuals. It reveals convergence rate, stagnation, sudden residual growth, and roundoff floors. If the residual reaches a plateau near,

$$\|r_k\|\approx C u,\tag{22.3.43}$$

where $u$ is the unit roundoff and $C$ is a problem-dependent scale factor, further iteration may be numerically meaningless. If the plateau occurs far above the expected level, the cause may be poor scaling, ill-conditioning, loss of orthogonality, or an unstable update.

A second pattern is the error-versus-resolution plot. For mesh width $h$, time step $\Delta t$, or polynomial degree $n$, one plots an error indicator against the resolution parameter. For algebraic convergence,

$$E(h)\approx C h^p\tag{22.3.44}$$

a log-log plot should have slope (p). For spectral convergence,

$$E(n)\approx C e^{-\alpha n} \tag{22.3.45}$$

a semilog plot against (n) should appear nearly linear before roundoff saturation. Deviations from these expected shapes are often more informative than the final error value.

A third pattern is the histogram or empirical distribution plot. If residual components are:

$$r=(r_1,r_2,\ldots,r_m) \tag{22.3.46}$$

then a histogram of $\{r_i\}$, or of standardized residuals:

$$z_i = \frac{r_i}{\widehat \sigma} \tag{22.3.47}$$

can reveal skewness, heavy tails, outliers, or structured model error. A residual norm may be small while a few components remain anomalously large. The maximum component:

$$\|r\|_{\infty} = \max_i |r_i|\tag{22.3.48}$$

and the Euclidean norm:

$$\|r\|_2 = \left(\sum_i r_i^2\right)^{1/2}\tag{22.3.49}$$

can tell different stories. A histogram helps explain the difference.

A fourth pattern is the diagnostic scatter plot. For approximate data pairs ((x_i,y_i)), one may plot residuals against fitted or predicted values:

$$(\widehat y_i, r_i),\qquad r_i=y_i-\widehat y_i \tag{22.3.50}$$

Random scatter around zero suggests that the residual has no obvious deterministic pattern. Curvature, funnels, clusters, or bands suggest model mismatch, heteroscedasticity, nonlinear structure, or scaling errors. This is why residual diagnostics should be judged in terms of meaningful deviations rather than only numerical significance.

A fifth pattern is the runtime or cost trace. Let $c_k$ denote cumulative computational cost after iteration $k$, measured in time, function evaluations, matrix-vector products, or memory traffic. Plotting,

$$c_k \mapsto \|r_k\|\tag{22.3.51}$$

or

$$k \mapsto \frac{\|r_k\|}{c_k}\tag{22.3.52}$$

can distinguish mathematically fast methods from computationally efficient methods. An algorithm with fewer iterations may still be slower if each iteration is expensive. For modern numerical computing, diagnostics should measure both mathematical progress and computational cost.

These plotting patterns should be deliberately standardized. Axes should have consistent scaling, tolerances should be shown when relevant, and plots should be comparable across runs. A residual plot without the stopping tolerance is incomplete. A convergence plot without the expected theoretical slope is harder to interpret. A histogram without scale information can exaggerate or hide anomalies. Lightweight plotting is therefore most effective when it is simple, repeatable, and mathematically tied to the algorithm.

The main conclusion of this section is that diagnostic output and lightweight plotting are part of the numerical method. They reveal convergence, instability, precision loss, pathological residual structure, non-portable behavior, and implementation defects before these failures become hidden in final outputs. Section 22.4 continues this infrastructure theme by moving from numerical traces to bit-level enumeration, where Gray codes provide a mathematical framework for systematic low-change traversal of discrete computational states.

### Rust Implementation

Following the discussion in Section 22.3.4 on lightweight plotting patterns for numerical software, Program 22.3.2 provides a practical implementation of standardized diagnostic visualization techniques for numerical computations. Modern numerical algorithms often produce large amounts of intermediate information, but the most useful diagnostics are frequently those that can be visualized quickly and interpreted consistently across multiple runs. This program demonstrates several lightweight plotting patterns that are directly tied to the mathematical diagnostics developed in Equations (22.3.42)–(22.3.52). It generates semilog residual traces, error-versus-resolution curves, residual histograms, residual scatter plots, and cost-convergence traces while also exporting structured diagnostic data for later analysis. Together, these visualizations illustrate how numerical behavior can be monitored continuously throughout a computation, enabling convergence problems, anomalous residual structures, scaling issues, and performance bottlenecks to be detected before they become hidden within final numerical outputs.

At the core of the implementation is the `DiagnosticPoint` structure, which stores the fundamental diagnostic quantities associated with an iterative computation. Each record contains an iteration index, a residual norm, and a cumulative computational cost. This compact representation provides the information needed to construct the convergence and cost traces discussed in Equations (22.3.42), (22.3.51), and (22.3.52). By organizing diagnostic information into a structured format, the program establishes a reusable framework for numerical observability.

The `generate_residual_trace` function constructs a synthetic convergence history. The generated residual sequence mimics the exponential reduction expected from a well-behaved iterative solver while also incorporating small oscillations and a roundoff floor. The resulting dataset is used to illustrate the semilog residual trace of Equation (22.3.42), where convergence behavior becomes visible through the evolution of $\log_{10}\|r_k\|$. The cumulative cost values generated alongside the residuals provide the data required for the cost-based diagnostics of Equations (22.3.51) and (22.3.52).

The `generate_error_resolution_data` function creates an error-versus-resolution dataset consistent with the algebraic convergence model of Equation (22.3.44). By generating errors proportional to $h^2$, the resulting log-log plot exhibits the expected linear relationship between resolution and error. This provides a practical illustration of how convergence order can be inferred visually from numerical experiments.

The `generate_residual_components` function constructs a collection of residual components corresponding to Equation (22.3.46). Most residual values follow a smooth oscillatory pattern, while a localized anomaly is intentionally introduced into a small subset of components. This design demonstrates the motivation for the histogram and scatter diagnostics developed in Equations (22.3.47)–(22.3.50), where structured residual behavior may remain hidden when only aggregate norms are examined.

The `export_residual_csv` function provides a lightweight mechanism for preserving diagnostic information outside the program. In addition to iteration number, residual norm, and cumulative cost, the exported file records $\log_{10}\|r_k\|$ and a residual-per-cost measure. Such structured output supports reproducibility and enables external analysis tools to process the diagnostic history without modifying the numerical code itself.

The `plot_semilog_residual` function implements the first plotting pattern discussed in Equation (22.3.42). It visualizes the logarithm of the residual norm as a function of iteration count, making convergence rates, stagnation regions, and potential roundoff floors immediately visible. This type of plot is widely used in iterative linear solvers, nonlinear equation methods, optimization algorithms, and eigenvalue computations.

The `plot_error_resolution` function implements the error-versus-resolution diagnostic associated with Equations (22.3.44) and (22.3.45). By plotting error against discretization scale, the figure reveals whether the observed convergence behavior matches theoretical expectations. Deviations from the anticipated slope can indicate implementation defects, insufficient resolution, or the onset of roundoff limitations.

The `plot_histogram` function implements the residual-distribution diagnostic described in Equations (22.3.46)–(22.3.49). Instead of examining only aggregate norms, the histogram reveals the statistical structure of the residual field. Outliers, heavy tails, skewness, or localized anomalies become visible through the empirical distribution of residual components.

The `plot_residual_scatter` function generates the diagnostic scatter visualization motivated by Equation (22.3.50). By plotting residual values individually, localized structures and systematic patterns become easier to identify. This type of diagnostic is frequently used in regression analysis, inverse problems, and model validation because it reveals information that aggregate error measures may conceal.

The `plot_cost_trace` function implements the computational-efficiency diagnostic discussed in Equations (22.3.51) and (22.3.52). Rather than measuring convergence solely as a function of iteration count, the plot relates residual reduction to cumulative computational cost. This distinction is important because methods with fewer iterations are not necessarily computationally cheaper if each iteration requires substantially more work.

The `main` function serves as a complete demonstration of the lightweight plotting framework. It generates representative diagnostic datasets, exports numerical traces to a CSV file, produces all visualization outputs, and computes summary statistics describing convergence and residual behavior. The resulting collection of plots illustrates how numerical diagnostics can be integrated directly into computational software with minimal implementation overhead while still providing substantial insight into algorithmic performance and numerical reliability.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
plotters = "0.3.7"
```

```rust
// Program 22.3.2: Lightweight Diagnostic Plotting Patterns
//
// Problem statement:
// Generate standardized lightweight diagnostic plots for numerical software,
// including semilog residual traces, error-versus-resolution curves,
// residual histograms, residual scatter plots, and cost traces.

use plotters::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::Write;

#[derive(Clone)]
struct DiagnosticPoint {
    k: usize,
    residual: f64,
    cost: f64,
}

fn generate_residual_trace(n: usize) -> Vec<DiagnosticPoint> {
    let mut data = Vec::new();

    for k in 0..n {
        let decay = 10.0_f64.powf(-0.13 * k as f64);
        let floor = 5.0e-14;
        let oscillation = 1.0 + 0.08 * (0.7 * k as f64).sin();
        let residual = (decay * oscillation).max(floor);
        let cost = 0.004 * (k as f64 + 1.0).powf(1.15);

        data.push(DiagnosticPoint { k, residual, cost });
    }

    data
}

fn generate_error_resolution_data() -> Vec<(f64, f64)> {
    let mut data = Vec::new();

    for j in 2..=9 {
        let h = 2.0_f64.powi(-j);
        let error = 0.35 * h.powi(2) + 2.0e-15;
        data.push((h, error));
    }

    data
}

fn generate_residual_components(n: usize) -> Vec<f64> {
    let mut values = Vec::new();

    for i in 0..n {
        let x = i as f64 / n as f64;
        let base = 0.02 * (20.0 * x).sin();
        let localized = if (0.62..0.68).contains(&x) { 0.12 } else { 0.0 };
        values.push(base + localized);
    }

    values
}

fn export_residual_csv(data: &[DiagnosticPoint], path: &str) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(file, "iteration,residual,log10_residual,cost,residual_per_cost")?;

    for p in data {
        writeln!(
            file,
            "{},{:.16e},{:.16e},{:.16e},{:.16e}",
            p.k,
            p.residual,
            p.residual.log10(),
            p.cost,
            p.residual / p.cost
        )?;
    }

    Ok(())
}

fn plot_semilog_residual(data: &[DiagnosticPoint], path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_min = data.iter().map(|p| p.residual.log10()).fold(f64::INFINITY, f64::min);
    let y_max = data.iter().map(|p| p.residual.log10()).fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Semilog Residual Trace", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(0usize..data.len(), y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("iteration k")
        .y_desc("log10(||r_k||)")
        .draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().map(|p| (p.k, p.residual.log10())),
        &BLUE,
    ))?;

    root.present()?;
    Ok(())
}

fn plot_error_resolution(data: &[(f64, f64)], path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_min = data.iter().map(|(h, _)| h.log10()).fold(f64::INFINITY, f64::min);
    let x_max = data.iter().map(|(h, _)| h.log10()).fold(f64::NEG_INFINITY, f64::max);
    let y_min = data.iter().map(|(_, e)| e.log10()).fold(f64::INFINITY, f64::min);
    let y_max = data.iter().map(|(_, e)| e.log10()).fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Error Versus Resolution", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("log10(h)")
        .y_desc("log10(error)")
        .draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().map(|(h, e)| (h.log10(), e.log10())),
        &BLUE,
    ))?;

    root.present()?;
    Ok(())
}

fn plot_histogram(values: &[f64], path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let bins = 24usize;
    let min_v = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let width = (max_v - min_v) / bins as f64;

    let mut counts = vec![0usize; bins];

    for &v in values {
        let mut idx = ((v - min_v) / width) as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1;
    }

    let max_count = *counts.iter().max().unwrap_or(&1);

    let mut chart = ChartBuilder::on(&root)
        .caption("Residual Component Histogram", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(45)
        .y_label_area_size(50)
        .build_cartesian_2d(min_v..max_v, 0usize..max_count)?;

    chart
        .configure_mesh()
        .x_desc("residual component")
        .y_desc("count")
        .draw()?;

    chart.draw_series(counts.iter().enumerate().map(|(i, &count)| {
        let x0 = min_v + i as f64 * width;
        let x1 = x0 + width;
        Rectangle::new([(x0, 0usize), (x1, count)], BLUE.filled())
    }))?;

    root.present()?;
    Ok(())
}

fn plot_residual_scatter(values: &[f64], path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Residual Scatter Plot", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(0usize..values.len(), y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("component index")
        .y_desc("residual component")
        .draw()?;

    chart.draw_series(
        values
            .iter()
            .enumerate()
            .map(|(i, &r)| Circle::new((i, r), 3, BLUE.filled())),
    )?;

    root.present()?;
    Ok(())
}

fn plot_cost_trace(data: &[DiagnosticPoint], path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_max = data.iter().map(|p| p.cost).fold(f64::NEG_INFINITY, f64::max);
    let y_min = data.iter().map(|p| p.residual.log10()).fold(f64::INFINITY, f64::min);
    let y_max = data.iter().map(|p| p.residual.log10()).fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Residual Versus Cumulative Cost", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(55)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("cumulative cost")
        .y_desc("log10(||r_k||)")
        .draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().map(|p| (p.cost, p.residual.log10())),
        &BLUE,
    ))?;

    root.present()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Program 22.3.2: Lightweight Diagnostic Plotting Patterns");
    println!("========================================================\n");

    let residual_trace = generate_residual_trace(90);
    let resolution_data = generate_error_resolution_data();
    let residual_components = generate_residual_components(200);

    export_residual_csv(&residual_trace, "residual_trace.csv")?;
    plot_semilog_residual(&residual_trace, "semilog_residual.png")?;
    plot_error_resolution(&resolution_data, "error_resolution.png")?;
    plot_histogram(&residual_components, "residual_histogram.png")?;
    plot_residual_scatter(&residual_components, "residual_scatter.png")?;
    plot_cost_trace(&residual_trace, "cost_trace.png")?;

    println!("Generated diagnostic files:");
    println!("  residual_trace.csv");
    println!("  semilog_residual.png");
    println!("  error_resolution.png");
    println!("  residual_histogram.png");
    println!("  residual_scatter.png");
    println!("  cost_trace.png");

    println!("\nDiagnostic Summary");
    println!("------------------");

    let first = residual_trace.first().unwrap().residual;
    let last = residual_trace.last().unwrap().residual;
    println!("initial residual      = {:.16e}", first);
    println!("final residual        = {:.16e}", last);
    println!("total reduction       = {:.16e}", first / last);

    let max_component = residual_components
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);

    let l2_component = residual_components
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();

    println!("max residual component = {:.16e}", max_component);
    println!("l2 residual norm       = {:.16e}", l2_component);

    Ok(())
}
```

Program 22.3.2 demonstrates that lightweight plotting is not merely a visualization convenience but an integral component of numerical diagnostics. By producing standardized plots directly from computational traces, the program transforms numerical behavior into information that can be interpreted rapidly and consistently. These visualizations complement the structured logging framework developed earlier in Section 22.3 and provide an additional layer of numerical observability.

The semilog residual trace illustrates how convergence behavior can be assessed visually rather than inferred solely from final residual values. The error-versus-resolution plot similarly reveals whether observed convergence rates are consistent with theoretical predictions. Together, these diagnostics provide immediate feedback regarding the correctness and effectiveness of a numerical method.

The histogram and scatter diagnostics demonstrate the importance of examining residual structure rather than relying exclusively on aggregate norms. A computation may appear successful according to $\|r\|_2$ while still containing localized anomalies or systematic patterns. By exposing these structures visually, the diagnostics help identify modeling deficiencies, scaling problems, and implementation defects that might otherwise remain undetected.

The cost trace extends the analysis beyond mathematical convergence by incorporating computational efficiency. Modern numerical software must balance accuracy, robustness, and performance simultaneously. The ability to relate residual reduction directly to computational cost provides a practical mechanism for comparing algorithms on equal footing and identifying situations in which mathematical efficiency and computational efficiency diverge.

Taken together, the plotting patterns implemented in this program establish a practical framework for numerical monitoring. They provide concise, repeatable, and mathematically meaningful visualizations that can be applied across a wide range of computational problems. These ideas naturally lead into the infrastructure topics of Section 22.4, where attention shifts from numerical traces and diagnostics to bit-level representations and Gray-code-based traversal of discrete computational states.

# 22.4. Gray Codes and Bit-Level Enumeration

Many numerical computations contain a discrete layer beneath their continuous mathematical formulation. Parameter choices, algorithmic switches, pivoting decisions, stencil patterns, subset selections, sign configurations, Boolean masks, branch conditions, and binary data representations all live in finite state spaces. When these states must be enumerated, tested, sampled, or updated, the ordering of the enumeration can matter. A naive ordering may change many bits between successive states, forcing expensive recomputation. A low-change ordering modifies only a small part of the state at each step, allowing incremental updates and more stable traversal of the discrete space.

Gray codes provide a mathematical framework for such low-change enumeration. The simplest and most important example is the binary reflected Gray code, which orders all (n)-bit strings so that consecutive strings differ in exactly one bit. In a numerical-computing context, this property is valuable for exhaustive testing of floating-point corner cases, enumeration of Boolean design spaces, incremental evaluation of subset-dependent quantities, cache-friendly traversal, and systematic exploration of neighboring configurations. Recent combinatorial literature supports this broader view of Gray codes as general methods for low-change traversal of structured state spaces, not merely as a binary encoding trick (Mütze, 2023; Liu et al., 2025; Pilaud and Williams, 2025).

## 22.4.1. The Boolean Hypercube and Hamming Geometry

The natural state space for $n$-bit enumeration is the Boolean hypercube:

$$Q_n=\{0,1\}^n \tag{22.4.1}$$

Each element,

$$x=(x_0,x_1,\ldots,x_{n-1})\in Q_n\tag{22.4.2}$$

is an $n$-bit state. The Hamming distance between two states $x,y\in Q_n$ is:

$$d_H(x,y) = \sum_{j=0}^{n-1}|x_j-y_j| \tag{22.4.3}$$

Equivalently, if $x$ and $y$ are interpreted as binary machine words, then

$$d_H(x,y) = \operatorname{popcount}(x\oplus y) \tag{22.4.4}$$

where $\oplus$ denotes bitwise exclusive-or and $\operatorname{popcount}$ counts the number of one bits.

The graph structure of the hypercube connects two vertices if they differ in exactly one coordinate. Thus, $x$ and $y$ are adjacent in $Q_n$, if and only if:

$$d_H(x,y)=1 \tag{22.4.5}$$

A Gray code of length $2^n$ is an ordering:

$$g_0,g_1,\ldots,g_{2^n-1}\tag{22.4.6}$$

of all vertices of $Q_n$ such that,

$$d_H(g_i,g_{i+1})=1,\qquad i=0,1,\ldots,2^n-2 \tag{22.4.7}$$

In graph-theoretic terms, a Gray code is a Hamiltonian path in the $n$-dimensional hypercube. If the final state is also adjacent to the first state,

$$d_H(g_{2^n-1},g_0)=1 \tag{22.4.8}$$

then the Gray code is cyclic and corresponds to a Hamiltonian cycle in $Q_n$.

The value of equation (22.4.7) is computational. Suppose a quantity $F(S)$ depends on a subset:

$$S\subseteq \{0,1,\ldots,n-1\} \tag{22.4.9}$$

and suppose $S$ is encoded by a bit string $x\in Q_n$, where,

$$j\in S\quad \Longleftrightarrow \quad x_j=1  \tag{22.4.10}$$

If consecutive subsets differ by one element, then many quantities can be updated incrementally. For example, if:

$$F(S)=\sum_{j\in S} w_j \tag{22.4.11}$$

and the transition from $S_i$ to $S_{i+1}$ changes only bit $q$, then,

$$F(S_{i+1}) = F(S_i)+\sigma_q w_q \tag{22.4.12}$$

where,

$$\sigma_q=\begin{cases}+1, & \text{if } q \text{ is inserted},\\ -1, & \text{if } q \text{ is removed}\end{cases}\tag{22.4.13}$$

Thus, the update cost is $O(1)$, rather than $O(n)$. This is the essential numerical advantage of Gray ordering: it converts complete recomputation into local modification whenever the target functional admits an incremental update.

This idea extends beyond sums. If $F(S)$ is a determinant, residual, objective function, mask-dependent transform, or state-dependent diagnostic, a one-bit change may allow a rank-one update, a local correction, or a targeted recomputation. The Gray-code ordering itself does not guarantee such an update, but it provides the discrete locality that makes these algorithms possible.

## 22.4.2. Binary Reflected Gray Codes

The classical binary reflected Gray code can be defined recursively. For $n=1$,

$$G_1=(0,1) \tag{22.4.14}$$

Given a Gray code $G_n$ for $n$ bits, the $(n+1)$-bit reflected code is obtained by prefixing $0$ to the entries of $G_n$, then prefixing $1$ to the entries of $G_n$ in reverse order:

$$G_{n+1} = \bigl(0G_n,\;1\operatorname{rev}(G_n)\bigr) \tag{22.4.15}$$

For $n=2$, this gives:

$$00,\;01,\;11,\;10\tag{22.4.16}$$

For $n=3$, it gives:

$$000,\;001,\;011,\;010,\;110,\;111,\;101,\;100 \tag{22.4.17}$$

Each consecutive pair differs in one bit, including the transition from the first half to the reflected second half.

The same code has a compact bitwise formula. If $i$ is an integer with binary representation,

$$i=(b_{n-1}b_{n-2}\cdots b_1b_0)_2 \tag{22.4.18}$$

then the corresponding reflected Gray code word is:

$$g(i)=i\oplus (i\gg 1),\qquad 0\le i<2^n \tag{22.4.19}$$

where $i\gg 1$ denotes a right shift by one bit. In component form, the Gray bits satisfy:

$$g_{n-1}=b_{n-1} \tag{22.4.20}$$

and

$$g_j=b_{j+1}\oplus b_j,\qquad j=0,1,\ldots,n-2 \tag{22.4.21}$$

The inverse transformation recovers the binary bits by prefix xor:

$$b_{n-1}=g_{n-1} \tag{22.4.22}$$

$$b_j=b_{j+1}\oplus g_j,\qquad j=n-2,n-3,\ldots,0 \tag{22.4.23}$$

Equivalently,

$$b_j = g_{n-1}\oplus g_{n-2}\oplus\cdots\oplus g_j \tag{22.4.24}$$

This inverse relation is important in implementation because Gray-coded states are often generated for traversal, while their ordinal index $i$ may still be needed for storage, lookup, or reproducibility.

To verify the one-bit-change property of equation (22.4.19), observe that consecutive integers $i$ and $i+1$ differ by flipping a trailing block of bits in binary. Suppose the least significant zero bit of $i$ occurs at position $q$. Then $i+1$ flips bit $q$ from $0$ to $1$ and changes all lower bits from $1$ to $0$. The xor transformation in equation (22.4.19) cancels the multiple lower-bit changes so that only one Gray bit changes. More precisely,

$$d_H(g(i),g(i+1))=1 \tag{22.4.25}$$

The position of the changed Gray bit can be found from:

$$h_i=g(i)\oplus g(i+1) \tag{22.4.26}$$

Since $h_i$ has exactly one nonzero bit, the changed coordinate is:

$$q_i=\operatorname{tzcount}(h_i) \tag{22.4.27}$$

where $\operatorname{tzcount}$ counts trailing zero bits. This is useful for incremental algorithms because it identifies exactly which component must be updated.

For example, suppose a computation depends on signs,

$$\sigma_j\in\{-1,+1\},\qquad j=0,\ldots,n-1 \tag{22.4.28}$$

encoded by bits through,

$$\sigma_j=2x_j-1\tag{22.4.29}$$

A Gray transition flips exactly one sign. If:

$$F(x)=\sum_{j=0}^{n-1}\sigma_j a_j \tag{22.4.30}$$

and bit $q$ changes, then:

$$F_{\mathrm{new}} = F_{\mathrm{old}}+(\sigma_q^{\mathrm{new}}-\sigma_q^{\mathrm{old}})a_q = F_{\mathrm{old}}+2\sigma_q^{\mathrm{new}}a_q \tag{22.4.31}$$

Again, the update is constant-time. This is the computational meaning of low-Hamming-distance traversal.

### Rust Implementation

Following the discussion in Sections 22.4.1 and 22.4.2 on the Boolean hypercube, Hamming geometry, and binary reflected Gray codes, Program 22.4.1 provides a practical implementation of Gray-code generation and low-change state traversal. In many numerical and combinatorial computations, the objective is not merely to enumerate all possible states but to do so in a manner that minimizes the changes between successive configurations. Gray codes achieve this by ensuring that adjacent states differ in exactly one bit, thereby creating a traversal of the hypercube that preserves discrete locality. This property is particularly valuable when quantities associated with subsets, sign patterns, masks, or configuration vectors admit incremental updates. The program implements Gray-code encoding and decoding, verifies the one-bit-change property of successive states, identifies the changed coordinate between neighboring Gray words, and demonstrates how subset-dependent quantities can be updated in constant time. In doing so, it translates the mathematical concepts developed in Equations (22.4.1)–(22.4.31) into an executable framework for efficient bit-level enumeration and incremental numerical computation.

At the core of the implementation are the functions `gray_encode` and `gray_decode`, which provide the forward and inverse transformations between binary integers and binary reflected Gray codes. The `gray_encode` function implements the bitwise construction introduced in Equation (22.4.19), generating a Gray-coded word by combining an integer with a one-bit right-shifted copy of itself through an exclusive-or operation. The resulting ordering possesses the low-Hamming-distance property developed throughout Section 22.4.2. The companion function `gray_decode` performs the inverse transformation described by Equations (22.4.22)–(22.4.24). Through successive prefix exclusive-or operations, it reconstructs the original binary index associated with a Gray-coded state. This inverse mapping is important because Gray codes are often used for traversal, while binary indices remain necessary for storage, lookup, and reproducibility.

The function `hamming_distance` computes the Hamming distance between two bit strings using the popcount formulation of Equation (22.4.4). Rather than comparing bits individually, the implementation performs an exclusive-or operation and counts the number of nonzero bits in the result. This quantity provides a direct measure of discrete separation within the Boolean hypercube and serves as the primary verification tool for the one-bit-change property of Gray codes.

The `changed_bit` function identifies the coordinate that changes between two consecutive Gray states. Using the exclusive-or relation of Equation (22.4.26), it constructs the difference mask between neighboring Gray words and then determines the position of the unique nonzero bit through a trailing-zero count. This corresponds directly to the changed coordinate described in Equation (22.4.27). For incremental algorithms, knowledge of this coordinate is often more valuable than the complete state itself because it identifies precisely which component of the computation must be updated.

The helper functions `bit_is_set` and `format_bits` provide convenient access to individual bit values and formatted binary representations. Although simple, these utilities play an important role in exposing the discrete structure of the Gray traversal and making the resulting state sequence easier to interpret during execution.

The function `subset_sum_from_scratch` implements the subset-dependent quantity introduced in Equation (22.4.11). Given a bit vector representing a subset, it evaluates the corresponding weighted sum by explicitly examining every coordinate. This computation has $O(n)$ cost because each bit must be inspected independently. The resulting value serves as a reference against which the incremental Gray-code updates can be verified.

The function `signed_sum_from_scratch` implements the sign-dependent functional described by Equations (22.4.28)–(22.4.30). Each bit determines whether the corresponding coefficient contributes with positive or negative sign, and the total quantity is formed by summing all signed contributions. Like the subset-sum calculation, this function recomputes the quantity directly from the current state and is used to verify the correctness of the incremental update mechanism.

The function `verify_gray_sequence` provides a systematic validation of the Gray-code ordering. It generates all states of the (n)-dimensional hypercube and verifies that every consecutive pair satisfies the adjacency condition of Equation (22.4.25). This verification confirms that the generated sequence forms a valid Gray traversal and demonstrates the Hamiltonian-path interpretation developed in Equations (22.4.6)–(22.4.8).

The main computational demonstration is performed within the `main` function. The program first constructs a complete Gray-code traversal for a four-bit state space and prints the binary index, Gray word, decoded binary representation, and changed coordinate for each state. It then exploits the one-bit-change property to update subset sums and sign-dependent quantities incrementally. Rather than recomputing these quantities from scratch after every state transition, the implementation modifies the current value using only the contribution associated with the changed coordinate, exactly as described by Equations (22.4.12), (22.4.13), and (22.4.31). After each update, the incrementally computed value is compared against a direct recomputation to verify correctness. This comparison demonstrates the essential computational advantage of Gray ordering: complete recomputation is replaced by a constant-time local update whenever the underlying functional admits an incremental formulation.

```rust
// Program 22.4.1: Binary Reflected Gray Codes and Incremental Subset Updates
//
// Problem statement:
// Generate binary reflected Gray codes, verify the one-bit-change property,
// recover binary indices from Gray-coded states, identify the changed bit
// between consecutive states, and use Gray ordering to update a subset sum
// in constant time.

fn gray_encode(i: u64) -> u64 {
    i ^ (i >> 1)
}

fn gray_decode(mut g: u64) -> u64 {
    let mut b = g;

    while g > 0 {
        g >>= 1;
        b ^= g;
    }

    b
}

fn hamming_distance(x: u64, y: u64) -> u32 {
    (x ^ y).count_ones()
}

fn changed_bit(prev: u64, next: u64) -> Option<usize> {
    let diff = prev ^ next;

    if diff.count_ones() == 1 {
        Some(diff.trailing_zeros() as usize)
    } else {
        None
    }
}

fn bit_is_set(x: u64, bit: usize) -> bool {
    ((x >> bit) & 1) == 1
}

fn format_bits(x: u64, nbits: usize) -> String {
    format!("{:0width$b}", x, width = nbits)
}

fn subset_sum_from_scratch(state: u64, weights: &[f64]) -> f64 {
    let mut sum = 0.0_f64;

    for (j, &w) in weights.iter().enumerate() {
        if bit_is_set(state, j) {
            sum += w;
        }
    }

    sum
}

fn signed_sum_from_scratch(state: u64, weights: &[f64]) -> f64 {
    let mut sum = 0.0_f64;

    for (j, &a_j) in weights.iter().enumerate() {
        let sigma_j = if bit_is_set(state, j) { 1.0 } else { -1.0 };
        sum += sigma_j * a_j;
    }

    sum
}

fn verify_gray_sequence(nbits: usize) -> bool {
    let total = 1_u64 << nbits;

    for i in 0..(total - 1) {
        let g_i = gray_encode(i);
        let g_next = gray_encode(i + 1);

        if hamming_distance(g_i, g_next) != 1 {
            return false;
        }
    }

    true
}

fn main() {
    println!("Program 22.4.1: Binary Reflected Gray Codes and Incremental Subset Updates");
    println!("===========================================================================\n");

    let nbits = 4usize;
    let total = 1_u64 << nbits;

    let weights = [0.75_f64, -1.25_f64, 2.50_f64, 4.00_f64];

    println!("Gray Code Table");
    println!("---------------");
    println!(
        "{:>5} {:>10} {:>10} {:>10} {:>14} {:>14} {:>14}",
        "i", "binary(i)", "gray(i)", "decoded", "changed bit", "subset sum", "signed sum"
    );

    let mut previous_gray = gray_encode(0);
    let mut subset_sum = subset_sum_from_scratch(previous_gray, &weights);
    let mut signed_sum = signed_sum_from_scratch(previous_gray, &weights);

    println!(
        "{:>5} {:>10} {:>10} {:>10} {:>14} {:>14.6} {:>14.6}",
        0,
        format_bits(0, nbits),
        format_bits(previous_gray, nbits),
        format_bits(gray_decode(previous_gray), nbits),
        "NA",
        subset_sum,
        signed_sum
    );

    for i in 1..total {
        let g = gray_encode(i);
        let decoded = gray_decode(g);
        let q = changed_bit(previous_gray, g).expect("Gray transition should flip one bit");

        if bit_is_set(g, q) {
            subset_sum += weights[q];
        } else {
            subset_sum -= weights[q];
        }

        let sigma_new = if bit_is_set(g, q) { 1.0 } else { -1.0 };
        signed_sum += 2.0 * sigma_new * weights[q];

        let subset_check = subset_sum_from_scratch(g, &weights);
        let signed_check = signed_sum_from_scratch(g, &weights);

        println!(
            "{:>5} {:>10} {:>10} {:>10} {:>14} {:>14.6} {:>14.6}",
            i,
            format_bits(i, nbits),
            format_bits(g, nbits),
            format_bits(decoded, nbits),
            q,
            subset_sum,
            signed_sum
        );

        assert_eq!(decoded, i);
        assert_eq!(hamming_distance(previous_gray, g), 1);
        assert!((subset_sum - subset_check).abs() < 1.0e-12);
        assert!((signed_sum - signed_check).abs() < 1.0e-12);

        previous_gray = g;
    }

    println!("\nVerification Summary");
    println!("--------------------");
    println!(
        "number of bits                    = {}",
        nbits
    );
    println!(
        "number of states                  = {}",
        total
    );
    println!(
        "all consecutive Hamming distances = 1? {}",
        verify_gray_sequence(nbits)
    );
    println!(
        "cyclic final-to-first distance    = {}",
        hamming_distance(gray_encode(total - 1), gray_encode(0))
    );
    println!("all inverse Gray decodings passed = true");
    println!("all incremental subset updates passed = true");
}
```

Program 22.4.1 demonstrates how Gray-code ordering transforms exhaustive enumeration from a purely combinatorial task into an efficient computational strategy. The implementation verifies that successive Gray states differ by exactly one bit and shows how this low-change property can be exploited to update subset-dependent quantities incrementally. Rather than recomputing each quantity from the beginning for every new state, the algorithm modifies only the contribution associated with the changed coordinate. This reduction in computational effort is precisely the practical significance of the Gray-code framework developed in Sections 22.4.1 and 22.4.2.

The examples of weighted subset sums and sign-dependent functionals illustrate two common situations in numerical computing. In both cases, the quantity of interest depends on a collection of selected components, and the Gray traversal guarantees that only one component changes at each step. Consequently, the update cost remains constant regardless of the size of the state space. This principle extends naturally to more sophisticated quantities such as residuals, objective functions, determinants, matrix factorizations, and state-dependent diagnostics, where local updates can often replace complete recomputation.

The verification procedures included in the implementation also emphasize the importance of correctness when working with discrete state spaces. By checking inverse Gray decoding, Hamming distances, cyclic adjacency, and incremental-update consistency, the program demonstrates how bit-level enumeration can be validated systematically. Such diagnostics are particularly valuable in testing environments, where subtle implementation errors may otherwise remain hidden within large combinatorial search spaces.

More broadly, the program illustrates a recurring theme of this chapter: computational infrastructure is often as important as the numerical method itself. Just as structured diagnostics and lightweight plotting make numerical behavior visible, Gray codes organize discrete state spaces so that neighboring configurations remain close. This locality supports efficient updates, interpretable testing procedures, and systematic exploration of finite computational domains. These ideas provide a natural bridge to the applications of Gray-code traversal developed later in Section 22.4.3, where bit-level enumeration becomes a tool for numerical testing, configuration search, and representational sensitivity analysis.

## 22.4.3. Bit-Level Enumeration in Numerical Testing and Search

Gray codes are useful whenever a computation must examine many neighboring discrete configurations. A common example is exhaustive testing over a finite set of Boolean options. Suppose an algorithm has $n$ binary switches, such as enabling or disabling scaling, pivoting, preconditioning, compensation, fused operations, diagnostic checks, or alternative update formulas. A configuration is represented by:

$$x\in\{0,1\}^n \tag{22.4.32}$$

A test statistic may be written as:

$$T(x)=\Phi(A,x,D) \tag{22.4.33}$$

where $A$ is the algorithm, $D$ is the test data, and $\Phi$ is an error, runtime, residual, or stability measure. Exhaustive enumeration evaluates:

$$T(x)\quad \text{for all } x\in\{0,1\}^n \tag{22.4.34}$$

In binary lexicographic order, successive states may differ in many bits. For example,

$$0111\longrightarrow 1000\tag{22.4.35}$$

changes all four bits. In Gray order, each transition changes one switch. This makes it easier to attribute changes in $T(x)$ to one local modification at a time. If,

$$\Delta_i = T(g_{i+1})-T(g_i) \tag{22.4.36}$$

then $\Delta_i$ corresponds to changing exactly one bit. Thus, Gray ordering naturally supports local sensitivity analysis over a Boolean configuration space.

A similar idea applies to subset search. Let:

$$S(x)=\{j:x_j=1\}\tag{22.4.37}$$

and suppose an objective depends on the selected subset:

$$\phi(x)=\phi(S(x)) \tag{22.4.38}$$

In model selection, sparse approximation, feature selection, sensor placement, or combinatorial preconditioner design, evaluating $\phi(x)$ from scratch for all $2^n$ subsets may be expensive. Gray traversal changes one subset element at each step, so an implementation can update factorizations, residuals, or statistics incrementally.

For instance, suppose the subset determines a matrix:

$$A_S=[a_j]_{j\in S} \tag{22.4.39}$$

and the diagnostic quantity is the least-squares residual,

$$\phi(S) = \min_c \|A_Sc-b\|_2 \tag{22.4.40}$$

If the transition $S\to S\cup{q}$ adds one column, then QR factors, normal equations, or Gram matrices can often be updated rather than recomputed. The Gram matrix changes from:

$$G_S=A_S^\top A_S\tag{22.4.41}$$

to

$$G_{S\cup\{q\}} = \begin{bmatrix}G_S & A_S^\top a_q\\a_q^\top A_S & a_q^\top a_q\end{bmatrix} \tag{22.4.42}$$

The rank and structure of the update are exposed precisely because the enumeration changes one subset element at a time.

Gray codes also support systematic floating-point testing. Many numerical failures occur near exceptional bit patterns: zeros, subnormals, powers of two, adjacent representable numbers, infinities, NaNs, and values that cause cancellation. If a floating-point word is viewed as a bit string, then Gray-like traversal can explore neighboring representations while changing only one bit at a time. This is useful when diagnosing whether an implementation is sensitive to exponent bits, significand bits, sign bits, or special encodings.

Let $w$ denote the bit representation of a floating-point number. A one-bit perturbation gives:

$$w' = w\oplus 2^q \tag{22.4.43}$$

If $x=\operatorname{decode}(w)$ and $x'=\operatorname{decode}(w')$, then the numerical perturbation is:

$$\Delta x = x'-x \tag{22.4.44}$$

The size and meaning of $\Delta x$ depend strongly on which bit is flipped. A low-order significand bit produces a small local perturbation, while an exponent bit may produce a large scaling change. A sign bit changes the sign. Thus, bit-level enumeration is not identical to small numerical perturbation, but it is a powerful way to explore representational sensitivity.

In randomized testing, Gray ordering can also be combined with deterministic reproducibility. If a test set is generated as

$$x_i = \Psi(g_i) \tag{22.4.45}$$

where $\Psi$ maps Gray-coded states to inputs or configurations, then the test sequence is reproducible and locally structured. Failures can be reported by their Gray index $i$, their state $g_i$, and the changed coordinate $q_i$. This makes failures easier to reproduce and isolate.

### Rust Implementation

Following the discussion in Section 22.4.3 on Gray-code traversal, Boolean configuration spaces, subset search, and bit-level representational sensitivity, Program 22.4.2 provides a practical implementation of Gray-ordered numerical testing and search. In many computational experiments, the objective is to evaluate a large number of discrete configurations while understanding how individual modifications affect algorithmic behavior. Conventional binary enumeration often changes multiple configuration bits simultaneously, making it difficult to identify the source of observed changes in numerical performance. Gray ordering addresses this problem by ensuring that each successive configuration differs in exactly one bit. The program demonstrates this principle through three representative applications: Gray-ordered Boolean configuration testing, incremental subset-objective evaluation, and systematic floating-point bit perturbations. Together, these examples illustrate how Gray-code traversal transforms exhaustive enumeration into a structured diagnostic tool for sensitivity analysis, reproducibility studies, and numerical testing, while providing direct computational realizations of Equations (22.4.32)–(22.4.45).

At the core of the implementation are the functions `gray_encode`, `hamming_distance`, and `changed_bit`, which provide the fundamental machinery required for Gray-code traversal. The `gray_encode` function generates the binary reflected Gray code associated with a given integer index, producing the state sequence discussed throughout Section 22.4. The `hamming_distance` function computes the number of differing bits between two states and is used to verify the adjacency condition implied by Equation (22.4.35). The `changed_bit` function determines which coordinate differs between two consecutive Gray states, thereby identifying the local modification responsible for the change in the diagnostic quantity $\Delta_i$ introduced in Equation (22.4.36). Together, these functions establish the low-change traversal mechanism that underlies all subsequent computations.

The helper functions `bit_is_set` and `format_bits` provide access to the binary structure of individual configurations. The first determines whether a particular switch is active, while the second generates formatted binary representations suitable for diagnostic output. Although simple, these functions make the relationship between Gray states and Boolean configurations explicit, allowing the progression through the configuration space to be interpreted directly from the program output.

The function `configuration_statistic` implements a simplified diagnostic measure corresponding to the test statistic $T(x)$ defined by Equation (22.4.33). The configuration state controls the activation of several hypothetical algorithmic options, including scaling, pivoting, preconditioning, compensation, and fused operations. Each enabled option modifies the baseline error measure according to a prescribed model. The resulting value serves as an illustrative numerical statistic whose variation can be tracked across the Gray-ordered configuration space. Because consecutive Gray states differ in exactly one switch, the local changes reported by Equation (22.4.36) can be interpreted directly as the effect of a single configuration modification.

The function `subset_objective_from_scratch` implements the subset-dependent objective $\phi(S)$ described by Equations (22.4.37)–(22.4.40). Given a subset represented by a Boolean state vector, the function computes an objective value using all selected components. This direct evaluation serves as a reference solution against which the incremental Gray-code updates can be verified. The example mirrors the subset-search framework developed in Section 22.4.3, where objective functions depend on the inclusion or exclusion of particular elements.

The function `floating_point_bit_perturbation` provides a computational realization of the representational sensitivity model introduced in Equations (22.4.43) and (22.4.44). A floating-point number is converted into its binary representation, a selected bit is flipped using an exclusive-or operation, and the modified bit pattern is decoded back into a floating-point value. The resulting perturbation $\Delta x$ reveals the numerical consequences of modifying different regions of the representation. This procedure provides a systematic mechanism for investigating sensitivity to sign bits, exponent bits, and significand bits.

The companion function `bit_category` classifies each perturbed bit according to its role within the IEEE-754 representation. By distinguishing significand, exponent, and sign bits, the program provides a direct interpretation of the perturbation results. This classification illustrates an important point emphasized in Section 22.4.3: a one-bit modification does not necessarily correspond to a small numerical perturbation. The significance of the perturbation depends strongly on which region of the representation is altered.

The first major demonstration within the `main` function performs Gray-ordered Boolean configuration testing. All configurations in a five-switch configuration space are enumerated according to Gray order, and the associated diagnostic statistic $T(x)$ is evaluated for each state. Because only one switch changes between successive configurations, the local sensitivity values reported by Equation (22.4.36) can be attributed unambiguously to individual configuration changes. This provides a practical illustration of local sensitivity analysis over a Boolean hypercube.

The second demonstration implements incremental subset search. Rather than recomputing the objective function from scratch after every state transition, the program updates the selected-sum and penalty terms using only the contribution associated with the changed coordinate. This reflects the incremental-update philosophy discussed in connection with Equations (22.4.41) and (22.4.42). The incrementally updated objective is then compared against a direct recomputation, verifying that the Gray traversal enables efficient local updates without sacrificing correctness.

The final demonstration performs systematic floating-point perturbation testing. A representative floating-point value is selected, and individual bits throughout its representation are flipped. The resulting perturbations illustrate the differing effects of modifying significand bits, exponent bits, and the sign bit. Low-order significand perturbations produce extremely small changes, whereas exponent modifications can generate dramatic scaling effects or even special values such as infinity. This experiment provides a direct computational illustration of the representational sensitivity concepts developed in Equations (22.4.43) and (22.4.44).

```rust
// Program 22.4.2: Gray-Ordered Configuration Testing and Bit-Level Perturbations
//
// Problem statement:
// Use Gray-code traversal to enumerate Boolean algorithm configurations,
// measure local sensitivity when one switch changes, update subset-dependent
// quantities incrementally, and explore one-bit perturbations of floating-point
// representations.

fn gray_encode(i: u64) -> u64 {
    i ^ (i >> 1)
}

fn hamming_distance(x: u64, y: u64) -> u32 {
    (x ^ y).count_ones()
}

fn changed_bit(prev: u64, next: u64) -> Option<usize> {
    let diff = prev ^ next;

    if diff.count_ones() == 1 {
        Some(diff.trailing_zeros() as usize)
    } else {
        None
    }
}

fn bit_is_set(x: u64, bit: usize) -> bool {
    ((x >> bit) & 1) == 1
}

fn format_bits(x: u64, nbits: usize) -> String {
    format!("{:0width$b}", x, width = nbits)
}

fn configuration_statistic(state: u64, base_error: f64) -> f64 {
    let scaling = bit_is_set(state, 0);
    let pivoting = bit_is_set(state, 1);
    let preconditioning = bit_is_set(state, 2);
    let compensation = bit_is_set(state, 3);
    let fused_ops = bit_is_set(state, 4);

    let mut error = base_error;

    if scaling {
        error *= 0.65;
    }

    if pivoting {
        error *= 0.55;
    }

    if preconditioning {
        error *= 0.35;
    }

    if compensation {
        error *= 0.75;
    }

    if fused_ops {
        error *= 0.90;
    }

    if scaling && preconditioning {
        error *= 0.80;
    }

    if pivoting && compensation {
        error *= 0.85;
    }

    error
}

fn subset_objective_from_scratch(state: u64, weights: &[f64]) -> f64 {
    let mut selected_sum = 0.0_f64;
    let mut penalty = 0.0_f64;

    for (j, &w) in weights.iter().enumerate() {
        if bit_is_set(state, j) {
            selected_sum += w;
        } else {
            penalty += 0.02 * w.abs();
        }
    }

    (1.0 - selected_sum).abs() + penalty
}

fn floating_point_bit_perturbation(x: f64, bit: usize) -> (u64, u64, f64, f64) {
    let w = x.to_bits();
    let w_prime = w ^ (1_u64 << bit);
    let x_prime = f64::from_bits(w_prime);
    let delta = x_prime - x;

    (w, w_prime, x_prime, delta)
}

fn bit_category(bit: usize) -> &'static str {
    match bit {
        0..=51 => "significand bit",
        52..=62 => "exponent bit",
        63 => "sign bit",
        _ => "out of range",
    }
}

fn main() {
    println!("Program 22.4.2: Gray-Ordered Configuration Testing and Bit-Level Perturbations");
    println!("=============================================================================\n");

    println!("Gray-Ordered Boolean Configuration Test");
    println!("---------------------------------------");

    let n_switches = 5usize;
    let total = 1_u64 << n_switches;
    let base_error = 1.0e-2_f64;

    println!(
        "{:>5} {:>10} {:>14} {:>16} {:>16}",
        "i", "state", "changed bit", "T(state)", "local delta"
    );

    let mut prev_state = gray_encode(0);
    let mut prev_stat = configuration_statistic(prev_state, base_error);

    println!(
        "{:>5} {:>10} {:>14} {:>16.8e} {:>16}",
        0,
        format_bits(prev_state, n_switches),
        "NA",
        prev_stat,
        "NA"
    );

    for i in 1..total {
        let state = gray_encode(i);
        let stat = configuration_statistic(state, base_error);
        let q = changed_bit(prev_state, state).expect("Gray transition should flip one bit");
        let delta = stat - prev_stat;

        println!(
            "{:>5} {:>10} {:>14} {:>16.8e} {:>16.8e}",
            i,
            format_bits(state, n_switches),
            q,
            stat,
            delta
        );

        assert_eq!(hamming_distance(prev_state, state), 1);

        prev_state = state;
        prev_stat = stat;
    }

    println!("\nIncremental Subset Objective Update");
    println!("-----------------------------------");

    let weights = [0.20_f64, 0.35_f64, 0.10_f64, 0.40_f64, 0.15_f64];
    let mut state = gray_encode(0);
    let mut selected_sum = 0.0_f64;
    let mut penalty: f64 = weights.iter().map(|w| 0.02 * w.abs()).sum();
    let mut objective = (1.0 - selected_sum).abs() + penalty;

    println!(
        "{:>5} {:>10} {:>14} {:>16} {:>16}",
        "i", "subset", "changed bit", "objective", "check error"
    );

    println!(
        "{:>5} {:>10} {:>14} {:>16.8e} {:>16.8e}",
        0,
        format_bits(state, weights.len()),
        "NA",
        objective,
        objective - subset_objective_from_scratch(state, &weights)
    );

    for i in 1..(1_u64 << weights.len()) {
        let next_state = gray_encode(i);
        let q = changed_bit(state, next_state).expect("Gray transition should flip one bit");

        if bit_is_set(next_state, q) {
            selected_sum += weights[q];
            penalty -= 0.02 * weights[q].abs();
        } else {
            selected_sum -= weights[q];
            penalty += 0.02 * weights[q].abs();
        }

        objective = (1.0 - selected_sum).abs() + penalty;
        let check = subset_objective_from_scratch(next_state, &weights);
        let check_error = objective - check;

        println!(
            "{:>5} {:>10} {:>14} {:>16.8e} {:>16.8e}",
            i,
            format_bits(next_state, weights.len()),
            q,
            objective,
            check_error
        );

        assert!(check_error.abs() < 1.0e-12);

        state = next_state;
    }

    println!("\nOne-Bit Floating-Point Perturbation Test");
    println!("----------------------------------------");

    let x = 1.0_f64;
    let test_bits = [0usize, 1, 10, 20, 40, 51, 52, 55, 60, 62, 63];

    println!("base value x = {:.16e}", x);
    println!(
        "{:>6} {:>18} {:>24} {:>24} {:>24}",
        "bit", "category", "x_prime", "delta", "relative delta"
    );

    for &bit in &test_bits {
        let (_w, _w_prime, x_prime, delta) = floating_point_bit_perturbation(x, bit);
        let relative_delta = delta.abs() / x.abs().max(f64::MIN_POSITIVE);

        println!(
            "{:>6} {:>18} {:>24.16e} {:>24.16e} {:>24.16e}",
            bit,
            bit_category(bit),
            x_prime,
            delta,
            relative_delta
        );
    }

    println!("\nVerification Summary");
    println!("--------------------");
    println!("Boolean configurations were traversed in Gray order.");
    println!("Each configuration transition changed exactly one switch.");
    println!("Subset objective updates matched recomputation from scratch.");
    println!("Floating-point perturbations showed sign, exponent, and significand sensitivity.");
}
```

Program 22.4.2 demonstrates how Gray-code traversal can be used as a practical tool for numerical testing, sensitivity analysis, and systematic exploration of discrete configuration spaces. By ensuring that consecutive states differ in only one coordinate, Gray ordering transforms exhaustive enumeration into a sequence of controlled local experiments. This property makes it possible to attribute observed changes in diagnostic quantities directly to individual configuration modifications, thereby simplifying both interpretation and debugging.

The Boolean configuration example illustrates how Gray traversal supports local sensitivity analysis. Rather than comparing configurations that differ in multiple algorithmic options simultaneously, the program isolates the effect of a single switch at each step. This approach is particularly useful when investigating the influence of implementation choices such as scaling strategies, pivoting policies, preconditioners, compensation techniques, or alternative numerical kernels.

The subset-search demonstration highlights the computational advantages of low-change enumeration. Because only one subset element changes between consecutive states, objective functions can often be updated incrementally rather than recomputed from scratch. The resulting reduction in computational effort becomes increasingly important as the size of the search space grows. Similar ideas underlie efficient algorithms for feature selection, sparse approximation, sensor placement, model selection, and combinatorial optimization.

The floating-point perturbation experiments illustrate a complementary perspective on numerical sensitivity. By modifying individual representation bits directly, the program reveals how different regions of the floating-point encoding contribute to numerical behavior. Significand perturbations typically generate small local changes, while exponent perturbations may produce large scaling effects or exceptional values. Such tests provide valuable diagnostic information when investigating numerical instability, representational sensitivity, or architecture-dependent behavior.

Taken together, the examples in this program demonstrate that Gray codes are more than a combinatorial curiosity. They provide a structured framework for organizing exhaustive searches, isolating local effects, updating computations efficiently, and performing systematic representational testing. These capabilities make Gray-code traversal a useful component of modern computational infrastructure, particularly when reproducibility, sensitivity analysis, and diagnostic transparency are important objectives.

## 22.4.4. Beyond Binary Reflected Codes

The binary reflected Gray code is the entry point, but the modern theory of Gray codes is much broader. A general Gray code is a listing of combinatorial objects in which consecutive objects differ by a small, prescribed operation. The state space may consist of permutations, combinations, partitions, trees, matchings, geometric configurations, or other structured objects. If $\Omega$ is a finite set of objects and (\\sim) is an adjacency relation, then a Gray code is an ordering:

$$\omega_0,\omega_1,\ldots,\omega_{|\Omega|-1}\tag{22.4.46}$$

such that,

$$\omega_i\sim \omega_{i+1}\qquad\text{for } i=0,\ldots,|\Omega|-2 \tag{22.4.47}$$

If the listing is cyclic, then also,

$$\omega_{|\Omega|-1}\sim \omega_0 \tag{22.4.48}$$

This definition shows that Gray coding is fundamentally a graph traversal problem. Construct a graph:

$$\mathcal G=(\Omega,E) \tag{22.4.49}$$

where edges connect objects that differ by an allowed local move. A Gray code is a Hamiltonian path in $\mathcal G$, and a cyclic Gray code is a Hamiltonian cycle. This abstraction explains why Gray-code research naturally extends from bitstrings to more complicated combinatorial families.

For combinations of fixed size $k$, the state space is:

$$\Omega_{n,k} = \{S\subseteq \{0,\ldots,n-1\}: |S|=k\} \tag{22.4.50}$$

A natural adjacency relation is the exchange of one element:

$$S\sim T\quad \Longleftrightarrow \quad|S\setminus T|=|T\setminus S|=1 \tag{22.4.51}$$

In bit-vector form, this means that two bits change: one bit changes from $1$ to $0$, and another changes from $0$ to $1$. Thus,

$$d_H(\chi_S,\chi_T)=2 \tag{22.4.52}$$

where $\chi_S$ is the characteristic vector of $S$. This is still a low-change enumeration because the subset cardinality is preserved.

For permutations, one may define adjacency by adjacent transposition:

$$\pi\sim \pi'\quad \Longleftrightarrow \quad\pi'=\pi\circ (j\; j+1)\tag{22.4.53}$$

for some adjacent swap. A Gray code for permutations then lists all permutations so that consecutive permutations differ by one adjacent swap. Such orderings are useful when a computation can update a permutation-dependent quantity locally, such as a determinant sign, ordering-dependent factorization, or combinatorial cost.

The same principle applies in numerical algorithms with structured discrete states. Mesh refinements differ by local element updates, sparse matrix patterns differ by inserted or removed nonzeros, pivot sequences differ by local swaps, and branch-and-bound nodes differ by local constraints. Gray-code thinking encourages the designer to ask: what is the natural adjacency relation, and can the computation be updated cheaply across adjacent states?

Recent work on combinatorial Gray codes strengthens this interpretation. The updated survey by Mütze presents Gray codes as a broad combinatorial-generation framework, while recent results on rotation Gray codes and wiggly permutations show that efficient low-change generation remains an active research area (Mütze, 2023; Liu et al., 2025; Pilaud and Williams, 2025). For this chapter, the main lesson is practical: bit-level enumeration is only the simplest case of a general strategy for organizing finite computational state spaces so that neighboring states are close, updates are local, and tests are easier to interpret.

Gray codes therefore fit naturally into the infrastructure theme of this chapter. Section 22.2 showed that floating-point behavior must be diagnosed at the level of the executed program. Section 22.3 showed that diagnostic traces and lightweight plots make numerical behavior visible. The present section adds that discrete state spaces should also be traversed systematically. Section 22.5 continues from this bit-level viewpoint to data integrity, where algebraic operations on finite strings are used to detect corruption in stored and transmitted numerical data.

# 22.5. Checksums, Cyclic Redundancy Checks, and Data Integrity

Numerical computation depends on the reliable movement and storage of data. Input files, parameter tables, simulation checkpoints, restart files, sparse matrices, compressed traces, diagnostic logs, and final result archives are all finite strings of bits. If these strings are corrupted, reordered, truncated, or mismatched with their metadata, the mathematical quality of the numerical algorithm may become irrelevant. A stable solver cannot compensate for a corrupted matrix file. A carefully designed convergence test cannot validate a checkpoint that has silently changed. For this reason, data integrity belongs naturally inside a chapter on trustworthy numerical execution.

Checksums and cyclic redundancy checks provide lightweight mechanisms for detecting accidental corruption. They do not prove semantic correctness, and they do not by themselves provide security against deliberate tampering. Their role is narrower but essential: they attach algebraic redundancy to data so that many transmission, storage, or copying errors become detectable. In modern scientific workflows, this role is increasingly connected to reproducibility, provenance, high-throughput communication, and data-governance practices. Recent literature supports this view by treating integrity checking as part of reproducible computation itself, especially in settings involving high-bandwidth networks, error-detecting codes, secure logging, and research data integrity (Zhang et al., 2024; An et al., 2025; Choi et al., 2025; Miller and Spiegel, 2025).

## 22.5.1. Data Integrity in Numerical Workflows

A numerical dataset may be represented as a finite word,

$$s=(s_0,s_1,\ldots,s_{m-1}) \tag{22.5.1}$$

where each $s_i$ is a byte, word, block, or bit depending on the level of representation. A data-integrity mechanism maps this word to a shorter redundancy value,

$$C(s)\in \mathcal C \tag{22.5.2}$$

where $\mathcal C$ is a finite checksum or code space. When the data are later read, transmitted, or reconstructed, the checksum is recomputed and compared with the stored value. If:

$$C(\widetilde s)\ne C(s) \tag{22.5.3}$$

then corruption is detected. If,

$$C(\widetilde s)=C(s)\tag{22.5.4}$$

while $\widetilde s\ne s$, then the error is undetected. Thus, every checksum partitions the set of possible data strings into equivalence classes. Errors that remain inside the same class are invisible to that checksum.

If the data alphabet has $B$ possible symbols and the string length is $m$, then there are $B^m$ possible strings. If the checksum has (M) possible values, then no checksum can be injective unless,

$$M\ge B^m \tag{22.5.5}$$

In practice $M\ll B^m$, so collisions are unavoidable. The purpose of a checksum is therefore not to eliminate collisions, but to make relevant error patterns unlikely or impossible to miss. For an idealized uniformly distributed checksum with $M$ possible values, a random independent corruption has approximate undetected probability:

$$P_{\mathrm{undetected}}\approx \frac{1}{M} \tag{22.5.6}$$

For a $k$-bit checksum, $M=2^k$, giving:

$$P_{\mathrm{undetected}}\approx 2^{-k} \tag{22.5.7}$$

This probability is only a heuristic for random errors. Real storage and communication errors are often structured: single-bit flips, burst errors, dropped bytes, transpositions, block reorderings, truncations, or repeated blocks. A good integrity mechanism should therefore be judged by the error classes it detects, not only by the number of checksum bits.

In numerical computing, the relevant integrity problem can be written as follows. Let a computational experiment use input data $D$, metadata $M_D$, parameters $\theta$, and code/environment descriptor $P$. A reproducible record should bind these pieces together:

$$\mathcal R=(D,M_D,\theta,P,Y,\mathcal D) \tag{22.5.8}$$

where $Y$ is the output and $\mathcal D$ contains diagnostics. If $D$ is replaced by $\widetilde D$, the computation becomes:

$$\widetilde Y=\mathcal C(A,\widetilde D,\theta,P) \tag{22.5.9}$$

even if all other components are unchanged. The difference

$$\widetilde Y-Y\tag{22.5.10}$$

may then be misinterpreted as an algorithmic, floating-point, or modeling effect when it is actually a data-integrity failure. Checksums and CRCs help prevent this confusion by making the identity of data objects testable.

A useful scientific data record therefore stores not only the data but also integrity information:

$$\mathcal I(D)=(\operatorname{name},\operatorname{size},\operatorname{format},C(D),\operatorname{version},\operatorname{metadata}) \tag{22.5.11}$$

The checksum $C(D)$ is only one field, but it is the field that allows later verification that the bytes being read are the bytes that were originally recorded. This is especially important for restart files, where a corrupted checkpoint can contaminate all subsequent time steps, and for compressed data, where a single bit error may propagate into a large decoded region.

## 22.5.2. Simple Checksums and Modular Sums

The simplest checksum is an additive modular sum. If a data word consists of integer blocks,

$$s_0,s_1,\ldots,s_{m-1} \tag{22.5.12}$$

then an $M$-modular checksum is:

$$C_{\mathrm{sum}}(s) = \left(\sum_{i=0}^{m-1}s_i\right)\bmod M \tag{22.5.13}$$

For an 8-bit checksum, $M=256$. For a 16-bit checksum, $M=2^{16}$. The checksum is fast, streaming, and easy to update. If a block $s_j$ is changed by an error $e_j$, then the checksum changes by,

$$C_{\mathrm{sum}}(\widetilde s)-C_{\mathrm{sum}}(s)\equiv e_j \pmod M \tag{22.5.14}$$

Thus, a single-block error is detected unless,

$$e_j\equiv 0 \pmod M \tag{22.5.15}$$

For multiple block errors $e_i$, the checksum difference is:

$$C_{\mathrm{sum}}(\widetilde s)-C_{\mathrm{sum}}(s)\equiv\sum_{i=0}^{m-1}e_i\pmod M \tag{22.5.16}$$

The error is undetected exactly when,

$$\sum_{i=0}^{m-1}e_i\equiv 0\pmod M \tag{22.5.17}$$

This equation exposes the weakness of simple sums: compensating errors can cancel. For example, increasing one byte by $1$ and decreasing another by $1$ leaves the checksum unchanged modulo $M$. Also, additive checksums are insensitive to many reorderings because,

$$\sum_{i=0}^{m-1}s_i = \sum_{i=0}^{m-1}s_{\pi(i)}\tag{22.5.18}$$

for any permutation $\pi$. Thus, a pure modular sum cannot reliably detect transposition or block-reordering errors.

Weighted checksums improve this by incorporating position. A simple weighted checksum is:

$$C_{\mathrm{w}}(s) = \left(\sum_{i=0}^{m-1} w_i s_i\right)\bmod M \tag{22.5.19}$$

where $w_i$ are prescribed weights. If the weights depend on $i$, then reordering the data usually changes the checksum. For example, with $w_i=i+1$,

$$C_{\mathrm{w}}(s) = \left(\sum_{i=0}^{m-1}(i+1)s_i\right)\bmod M \tag{22.5.20}$$

A transposition of two unequal blocks $s_j$ and $s_k$ changes the weighted sum by:

$$\Delta C_{\mathrm{w}}\equiv(j+1)s_k+(k+1)s_j-(j+1)s_j-(k+1)s_k\pmod M \tag{22.5.21}$$

which simplifies to,

$$\Delta C_{\mathrm{w}}\equiv(k-j)(s_j-s_k)\pmod M \tag{22.5.22}$$

The transposition is detected unless this quantity is congruent to zero modulo $M$.

Although simple and weighted checksums are useful for low-cost sanity checks, they are not strong algebraic error detectors. Their main advantages are speed, streaming evaluation, and ease of incremental update. If a block $s_j$ is changed to $s_j'$, then the modular checksum can be updated without scanning the full data:

$$C_{\mathrm{sum}}(s')\equiv C_{\mathrm{sum}}(s)-s_j+s_j'\pmod M \tag{22.5.23}$$

For a weighted checksum,

$$C_{\mathrm{w}}(s')\equiv C_{\mathrm{w}}(s)-w_js_j+w_js_j'\pmod M \tag{22.5.24}$$

This makes such checksums attractive for diagnostic logs, running data summaries, and quickly detecting gross mismatches. For stronger protection against structured burst errors, cyclic redundancy checks are preferred.

### Rust Implementation

Following the discussion in Section 22.5.2 on additive modular checksums, weighted checksums, error cancellation, and position-dependent integrity verification, Program 22.5.1 provides a practical implementation of lightweight checksum mechanisms for numerical data records. In scientific computing, large datasets are frequently stored, transmitted, copied, and reloaded throughout the lifecycle of a computational experiment. Even when numerical algorithms are mathematically correct and numerically stable, corrupted input files, modified parameter records, or reordered data blocks can invalidate the resulting computation. This program demonstrates several fundamental integrity-checking techniques, including additive modular checksums, weighted checksums, incremental checksum updates, compensating-error detection, and transposition diagnostics. By evaluating representative corruption scenarios and comparing the behavior of different checksum constructions, the implementation illustrates the strengths and limitations of the integrity mechanisms developed in Equations (22.5.12)–(22.5.24), while highlighting the role of checksums as a first line of defense against accidental data corruption.

At the core of the implementation are the functions `additive_checksum` and `weighted_checksum`, which implement the two checksum constructions developed in Section 22.5.2. The `additive_checksum` function evaluates the modular checksum defined by Equation (22.5.13) by summing all data blocks and reducing the result modulo the selected checksum modulus. This represents the simplest form of redundancy generation and provides a lightweight mechanism for detecting many common corruption events. The `weighted_checksum` function implements the position-dependent checksum of Equations (22.5.19) and (22.5.20), where each data value is multiplied by a location-dependent weight before accumulation. By incorporating positional information, the weighted checksum becomes sensitive to many reordering and transposition errors that remain invisible to a pure modular sum.

The functions `update_additive_checksum` and `update_weighted_checksum` implement the incremental-update formulas introduced in Equations (22.5.23) and (22.5.24). Rather than recomputing a checksum by scanning the entire dataset after a modification, these functions update the checksum using only the old value, the new value, and the affected position. This capability is particularly important for large scientific datasets, diagnostic logs, and evolving data structures, where full recomputation may be unnecessarily expensive. The implementation demonstrates how checksum maintenance can be performed efficiently while preserving consistency with the corresponding direct computations.

The function `corrupt_byte` generates a modified copy of a data record by replacing a selected byte with a new value. This function provides a simple mechanism for simulating the corruption process represented by Equations (22.5.14)–(22.5.17). By introducing a controlled modification into the dataset, the program can evaluate whether the resulting corruption is detected by the checksum mechanisms under consideration.

The function `transpose_bytes` implements a common structured error pattern in which two data elements exchange positions. This operation is particularly important because Equation (22.5.18) shows that additive modular sums are invariant under arbitrary permutations of the data. Consequently, transposition errors provide a natural test case for illustrating the limitations of additive checksums and the advantages of position-dependent weighting schemes.

The helper function `print_bytes` formats data records in a readable form for diagnostic output. Although simple, this utility makes it easier to inspect the relationship between the original and modified datasets and to verify visually that the corruption scenarios correspond to the intended experiments.

The `main` function serves as a comprehensive demonstration of checksum behavior under several representative integrity scenarios. It begins by constructing a numerical data record and computing both additive and weighted checksums. The program then introduces a single-byte corruption and verifies that both checksum types detect the modification. Next, it constructs a compensating-error scenario in which one data value is increased while another is decreased by the same amount. This example directly illustrates the cancellation mechanism described by Equation (22.5.17), where the additive checksum remains unchanged despite the presence of corruption. The weighted checksum, however, detects the modification because the position-dependent contributions no longer cancel.

The final experiment performs a transposition of two unequal data blocks. Consistent with Equation (22.5.18), the additive checksum remains unchanged because the total sum of the data is preserved. The weighted checksum, however, changes according to the transposition analysis of Equations (22.5.21) and (22.5.22), thereby revealing the corruption. The program concludes by summarizing the detection capabilities of the two checksum constructions and verifying that the incremental-update formulas produce results identical to direct recomputation.

```rust
// Program 22.5.1: Simple and Weighted Checksums for Data Integrity
//
// Problem statement:
// Demonstrate additive modular checksums, weighted checksums, incremental
// checksum updates, byte corruption detection, and transposition detection
// for finite numerical data records.

const MODULUS: u32 = 65_536;

fn additive_checksum(data: &[u8], modulus: u32) -> u32 {
    data.iter()
        .fold(0_u32, |acc, &byte| (acc + byte as u32) % modulus)
}

fn weighted_checksum(data: &[u8], modulus: u32) -> u32 {
    data.iter().enumerate().fold(0_u32, |acc, (i, &byte)| {
        let weight = (i as u32 + 1) % modulus;
        (acc + weight * byte as u32) % modulus
    })
}

fn update_additive_checksum(
    old_checksum: u32,
    old_value: u8,
    new_value: u8,
    modulus: u32,
) -> u32 {
    (old_checksum + modulus - old_value as u32 + new_value as u32) % modulus
}

fn update_weighted_checksum(
    old_checksum: u32,
    index: usize,
    old_value: u8,
    new_value: u8,
    modulus: u32,
) -> u32 {
    let weight = (index as u32 + 1) % modulus;

    (old_checksum + modulus - (weight * old_value as u32) % modulus
        + (weight * new_value as u32) % modulus)
        % modulus
}

fn corrupt_byte(data: &[u8], index: usize, new_value: u8) -> Vec<u8> {
    let mut corrupted = data.to_vec();
    corrupted[index] = new_value;
    corrupted
}

fn transpose_bytes(data: &[u8], i: usize, j: usize) -> Vec<u8> {
    let mut transposed = data.to_vec();
    transposed.swap(i, j);
    transposed
}

fn print_bytes(label: &str, data: &[u8]) {
    print!("{:<28} = [", label);

    for (i, byte) in data.iter().enumerate() {
        if i + 1 == data.len() {
            print!("{}", byte);
        } else {
            print!("{}, ", byte);
        }
    }

    println!("]");
}

fn main() {
    println!("Program 22.5.1: Simple and Weighted Checksums for Data Integrity");
    println!("================================================================\n");

    let data: Vec<u8> = vec![12, 25, 31, 44, 58, 63, 79, 91];

    println!("Original Data Record");
    println!("--------------------");
    print_bytes("data", &data);
    println!("number of bytes              = {}", data.len());
    println!("checksum modulus             = {}\n", MODULUS);

    let additive = additive_checksum(&data, MODULUS);
    let weighted = weighted_checksum(&data, MODULUS);

    println!("Initial Checksums");
    println!("-----------------");
    println!("additive checksum            = {}", additive);
    println!("weighted checksum            = {}\n", weighted);

    println!("Single-Byte Corruption Test");
    println!("---------------------------");

    let changed_index = 3usize;
    let old_value = data[changed_index];
    let new_value = old_value + 7;

    let corrupted = corrupt_byte(&data, changed_index, new_value);

    let additive_corrupted = additive_checksum(&corrupted, MODULUS);
    let weighted_corrupted = weighted_checksum(&corrupted, MODULUS);

    let additive_updated =
        update_additive_checksum(additive, old_value, new_value, MODULUS);

    let weighted_updated =
        update_weighted_checksum(weighted, changed_index, old_value, new_value, MODULUS);

    print_bytes("corrupted data", &corrupted);
    println!("changed index                = {}", changed_index);
    println!("old value                    = {}", old_value);
    println!("new value                    = {}", new_value);
    println!("recomputed additive checksum = {}", additive_corrupted);
    println!("updated additive checksum    = {}", additive_updated);
    println!("recomputed weighted checksum = {}", weighted_corrupted);
    println!("updated weighted checksum    = {}", weighted_updated);
    println!(
        "corruption detected by additive checksum? {}",
        additive_corrupted != additive
    );
    println!(
        "corruption detected by weighted checksum? {}\n",
        weighted_corrupted != weighted
    );

    println!("Compensating-Error Test");
    println!("-----------------------");

    let mut compensating = data.clone();
    compensating[1] += 1;
    compensating[5] -= 1;

    let additive_compensating = additive_checksum(&compensating, MODULUS);
    let weighted_compensating = weighted_checksum(&compensating, MODULUS);

    print_bytes("compensating data", &compensating);
    println!("additive checksum            = {}", additive_compensating);
    println!("weighted checksum            = {}", weighted_compensating);
    println!(
        "detected by additive checksum? {}",
        additive_compensating != additive
    );
    println!(
        "detected by weighted checksum? {}\n",
        weighted_compensating != weighted
    );

    println!("Transposition Test");
    println!("------------------");

    let transposed = transpose_bytes(&data, 2, 6);

    let additive_transposed = additive_checksum(&transposed, MODULUS);
    let weighted_transposed = weighted_checksum(&transposed, MODULUS);

    print_bytes("transposed data", &transposed);
    println!("swapped indices              = 2 and 6");
    println!("additive checksum            = {}", additive_transposed);
    println!("weighted checksum            = {}", weighted_transposed);
    println!(
        "detected by additive checksum? {}",
        additive_transposed != additive
    );
    println!(
        "detected by weighted checksum? {}",
        weighted_transposed != weighted
    );

    println!("\nVerification Summary");
    println!("--------------------");
    println!("single-byte corruption was detected by both checksum types");
    println!("compensating errors cancelled in the additive checksum");
    println!("weighted checksum detected the compensating position-dependent change");
    println!("additive checksum did not detect byte transposition");
    println!("weighted checksum detected byte transposition");
}
```

Program 22.5.1 demonstrates the practical behavior of simple and weighted checksum constructions when applied to representative numerical data records. The examples illustrate that checksums do not eliminate the possibility of undetected errors; rather, they define which classes of corruption are likely to be detected and which may remain invisible. This distinction is central to the discussion in Section 22.5.2, where the effectiveness of a checksum is judged not only by its size but also by the structure of the errors it can detect.

The single-byte corruption example shows the fundamental strength of modular checksums. A localized modification changes the checksum value and is immediately detected. This capability makes additive checksums attractive for inexpensive integrity checks during data transfer, file storage, and diagnostic logging. The incremental-update formulas further demonstrate that such checksums can be maintained efficiently without rescanning the entire dataset.

The compensating-error and transposition experiments reveal the limitations of purely additive constructions. Because additive checksums depend only on the total sum of the data, carefully structured errors may leave the checksum unchanged even when the underlying record has been modified. Weighted checksums mitigate this weakness by incorporating positional information, allowing many structured errors to be detected that would otherwise escape notice. These examples provide a concrete illustration of why the error model is often more important than the checksum size alone.

More broadly, the program highlights an important principle of trustworthy numerical computation: integrity verification must accompany numerical algorithms throughout the computational workflow. Data corruption can invalidate even the most sophisticated numerical methods, and lightweight checksum mechanisms provide an inexpensive means of verifying that stored or transmitted data remain consistent with their original representations. While stronger algebraic techniques such as cyclic redundancy checks provide greater protection against structured error patterns, the simple checksum constructions implemented here establish the conceptual foundation upon which more advanced integrity mechanisms are built.

## 22.5.3. Cyclic Redundancy Checks Over $\mathbb F_2$

A cyclic redundancy check treats a bit string as a polynomial over the finite field $\mathbb F_2$. In this field, the coefficients are $0$ or $1$, and addition is exclusive-or. If,

$$s=(s_0,s_1,\ldots,s_{m-1}),\qquad s_i\in\{0,1\} \tag{22.5.25}$$

then the corresponding polynomial is:

$$S(x) = s_0x^{m-1}+s_1x^{m-2}+\cdots+s_{m-2}x+s_{m-1} \tag{22.5.26}$$

Let $G(x)\in\mathbb F_2[x]$ be a generator polynomial of degree $r$:

$$G(x)=x^r+g_{r-1}x^{r-1}+\cdots+g_1x+g_0 \tag{22.5.27}$$

To compute an $r$-bit CRC, one multiplies $S(x)$ by $x^r$ and divides by $G(x)$ over $\mathbb F_2$. The remainder is:

$$R(x) = S(x)x^r \bmod G(x),\qquad\deg R<r \tag{22.5.28}$$

The transmitted or stored code polynomial is:

$$T(x) = S(x)x^r+R(x) \tag{22.5.29}$$

Because subtraction and addition are the same in $\mathbb F_2$, equation (22.5.28) implies:

$$T(x)\equiv 0 \pmod{G(x)} \tag{22.5.30}$$

On verification, the receiver divides the received polynomial $\widetilde T(x)$ by $G(x)$. If,

$$\widetilde T(x)\bmod G(x)\ne 0 \tag{22.5.31}$$

then an error is detected.

Suppose the received word differs from the transmitted word by an error polynomial $E(x)$:

$$\widetilde T(x)=T(x)+E(x) \tag{22.5.32}$$

Since $T(x)$ is divisible by $G(x)$, the verification remainder is:

$$\widetilde T(x)\bmod G(x) = E(x)\bmod G(x) \tag{22.5.33}$$

Therefore, the error is undetected if and only if:

$$E(x)\equiv 0\pmod{G(x)} \tag{22.5.34}$$

This compact condition is the central mathematical fact behind CRCs. The generator polynomial determines which error patterns are guaranteed to be detected.

Several detection properties follow immediately. A single-bit error has the form:

$$E(x)=x^q \tag{22.5.35}$$

If $G(x)$ has degree at least one and is not equal to $x$, then $G(x)\nmid x^q$, so such an error is detected. A two-bit error has the form:

$$E(x)=x^i+x^j = x^j(x^{i-j}+1),\qquad i>j \tag{22.5.36}$$

Since $G(x)$ is usually chosen with nonzero constant term, it does not divide $x^j$. Thus, detecting all two-bit errors up to a certain message length depends on ensuring that $G(x)$ does not divide:

$$x^{i-j}+1\tag{22.5.37}$$

for relevant separations $i-j$.

A burst error of length $L$ has the form:

$$E(x)=x^q B(x) \tag{22.5.38}$$

where,

$$\deg B=L-1 \tag{22.5.39}$$

and the first and last coefficients of $B(x)$ are nonzero. If $L\le r$, then,

$$\deg B<r=\deg G \tag{22.5.40}$$

so $G(x)$ cannot divide $B(x)$ unless $B(x)=0$, which is not a burst error. Hence, CRCs detect all burst errors of length at most $r$, assuming the generator has a nonzero constant term. Longer bursts are detected with high probability, depending on the generator polynomial.

The polynomial viewpoint also explains why CRCs are efficient. Division over $\mathbb F_2$ uses shifts and exclusive-or operations rather than integer division. The recurrence can be implemented bitwise, bytewise, table-driven, or in parallel. For high-throughput systems, the same algebra can be reorganized to process multiple bits or words at a time. Recent work on parallel CRC computation for high-bandwidth networks and FPGA implementations shows that CRC remains an active performance topic when integrity checks must coexist with large data rates and low latency (Zhang et al., 2024).

### Rust Implementation

Following the discussion in Section 22.5.3 on cyclic redundancy checks, polynomial arithmetic over $\mathbb{F}_2$, and error-detection guarantees derived from generator polynomials, Program 22.5.2 provides a practical implementation of bitwise CRC computation and verification. Unlike additive checksums, which operate through integer summation, CRCs interpret a bit stream as a polynomial over the finite field $\mathbb{F}_2$ and construct redundancy through polynomial division. This algebraic formulation provides significantly stronger protection against structured error patterns, including single-bit errors, many classes of multi-bit errors, and burst errors. The program demonstrates the complete CRC workflow, including remainder generation, codeword construction, verification, and corruption testing. Through representative examples of single-bit, two-bit, and burst-error injection, the implementation illustrates the mathematical principles developed in Equations (22.5.25)–(22.5.40) and shows how polynomial divisibility becomes a practical integrity-checking mechanism in numerical and computational systems.

At the core of the implementation is the `CrcSpec` structure, which encapsulates the defining parameters of a cyclic redundancy check. It stores the generator polynomial $G(x)$ introduced in Equation (22.5.27), together with its degree $r$ and an identifying label. This abstraction allows the same CRC machinery to be reused with different generator polynomials while preserving a uniform computational interface.

The `degree` function determines the degree of a polynomial represented as a binary word. Since CRC arithmetic is performed over $\mathbb{F}_2$, the degree corresponds to the position of the highest nonzero coefficient. This operation is fundamental to the polynomial division process because it determines the alignment of the generator polynomial during each elimination step.

The function `polynomial_remainder` implements the polynomial division described by Equation (22.5.28). The algorithm repeatedly aligns the highest-degree term of the generator polynomial with the highest-degree term of the current dividend and performs elimination using exclusive-or operations. Since addition and subtraction coincide in $\mathbb{F}_2$, no conventional arithmetic subtraction is required. The final result is the remainder polynomial $R(x)$, whose degree is strictly less than the degree of the generator polynomial, exactly as required by Equation (22.5.28).

The `compute_crc` function constructs the CRC remainder for a message polynomial $S(x)$. Following Equation (22.5.28), the message is first multiplied by $x^r$ through a left shift by $r$ bits. The shifted polynomial is then divided by the generator polynomial, and the resulting remainder forms the CRC field that will later be appended to the message.

The function `append_crc` constructs the transmitted codeword $T(x)$ defined in Equation (22.5.29). It combines the shifted message polynomial with the computed remainder to form a codeword that is exactly divisible by the generator polynomial. This divisibility property is the key algebraic condition underlying CRC verification and corresponds directly to Equation (22.5.30).

The `verify_codeword` function implements the receiver-side verification process described by Equations (22.5.30) and (22.5.31). The received codeword is divided by the generator polynomial, and the verification succeeds only if the remainder is identically zero. A nonzero remainder indicates that the received polynomial is not divisible by the generator and therefore that corruption has occurred.

The helper functions `flip_bit` and `inject_burst_error` generate representative error patterns for diagnostic testing. The `flip_bit` function creates a single-bit perturbation corresponding to the error polynomial of Equation (22.5.35), while `inject_burst_error` constructs contiguous burst-error patterns consistent with Equations (22.5.38)–(22.5.40). These functions allow the program to demonstrate experimentally the theoretical detection properties developed in the section.

The function `format_bits_u128` converts binary words into fixed-width bit strings suitable for diagnostic output. Although simple, this utility makes the relationship between message polynomials, CRC remainders, codewords, and corrupted words visually transparent during execution.

The `main` function serves as a complete demonstration of CRC encoding, verification, and error detection. It begins by defining a generator polynomial and a sample message, computes the CRC remainder, and constructs the corresponding codeword. The program then verifies that the codeword is divisible by the generator polynomial, thereby confirming Equation (22.5.30). Next, it injects a single-bit error, a burst error, and a two-bit error into the transmitted word. For each corruption scenario, the verification remainder is recomputed and shown to be nonzero, illustrating Equation (22.5.33). The resulting experiments provide direct computational evidence of the central CRC principle expressed in Equation (22.5.34): an error is detected whenever the corresponding error polynomial is not divisible by the generator polynomial.

```rust
// Program 22.5.2: Bitwise CRC Computation over GF(2)
//
// Problem statement:
// Implement a bitwise cyclic redundancy check by treating a data word as a
// polynomial over GF(2). The program computes an r-bit CRC remainder, appends
// it to the message, verifies the resulting codeword, and demonstrates
// detection of single-bit and burst errors.

#[derive(Clone, Copy)]
struct CrcSpec {
    name: &'static str,
    width: u8,
    polynomial: u32,
}

fn degree(poly: u32) -> i32 {
    if poly == 0 {
        -1
    } else {
        31 - poly.leading_zeros() as i32
    }
}

fn polynomial_remainder(mut word: u128, generator: u32) -> u32 {
    let gen_degree = degree(generator);
    assert!(gen_degree > 0, "generator polynomial must have positive degree");

    let gen_shifted_base = generator as u128;

    while word != 0 {
        let word_degree = 127 - word.leading_zeros() as i32;

        if word_degree < gen_degree {
            break;
        }

        let shift = word_degree - gen_degree;
        word ^= gen_shifted_base << shift;
    }

    word as u32
}

fn compute_crc(message: u64, message_bits: usize, spec: CrcSpec) -> u32 {
    assert!(message_bits + spec.width as usize <= 128);

    let shifted_message = (message as u128) << spec.width;
    polynomial_remainder(shifted_message, spec.polynomial)
}

fn append_crc(message: u64, message_bits: usize, spec: CrcSpec) -> u128 {
    let crc = compute_crc(message, message_bits, spec);

    ((message as u128) << spec.width) | crc as u128
}

fn verify_codeword(codeword: u128, spec: CrcSpec) -> bool {
    polynomial_remainder(codeword, spec.polynomial) == 0
}

fn flip_bit(word: u128, bit_from_right: usize) -> u128 {
    word ^ (1_u128 << bit_from_right)
}

fn inject_burst_error(word: u128, start_bit_from_right: usize, length: usize) -> u128 {
    let mut mask = 0_u128;

    for j in 0..length {
        mask |= 1_u128 << (start_bit_from_right + j);
    }

    word ^ mask
}

fn format_bits_u128(word: u128, width: usize) -> String {
    format!("{:0width$b}", word, width = width)
}

fn main() {
    println!("Program 22.5.2: Bitwise CRC Computation over GF(2)");
    println!("==================================================\n");

    let spec = CrcSpec {
        name: "CRC-4 demonstration polynomial",
        width: 4,
        polynomial: 0b1_0011, // G(x) = x^4 + x + 1
    };

    let message_bits = 12usize;
    let message = 0b1011_0010_1110_u64;

    let total_bits = message_bits + spec.width as usize;

    println!("CRC Specification");
    println!("-----------------");
    println!("name                 = {}", spec.name);
    println!("CRC width             = {} bits", spec.width);
    println!(
        "generator polynomial  = {}",
        format_bits_u128(spec.polynomial as u128, spec.width as usize + 1)
    );
    println!("message bits          = {}", message_bits);
    println!(
        "message               = {}\n",
        format_bits_u128(message as u128, message_bits)
    );

    let crc = compute_crc(message, message_bits, spec);
    let codeword = append_crc(message, message_bits, spec);

    println!("CRC Encoding");
    println!("------------");
    println!(
        "CRC remainder R(x)    = {}",
        format_bits_u128(crc as u128, spec.width as usize)
    );
    println!(
        "stored codeword T(x)  = {}",
        format_bits_u128(codeword, total_bits)
    );
    println!(
        "verification remainder is zero? {}\n",
        verify_codeword(codeword, spec)
    );

    println!("Single-Bit Error Detection");
    println!("--------------------------");

    let single_error_bit = 5usize;
    let single_error_word = flip_bit(codeword, single_error_bit);
    let single_remainder = polynomial_remainder(single_error_word, spec.polynomial);

    println!("flipped bit position       = {}", single_error_bit);
    println!(
        "corrupted codeword         = {}",
        format_bits_u128(single_error_word, total_bits)
    );
    println!(
        "verification remainder     = {}",
        format_bits_u128(single_remainder as u128, spec.width as usize)
    );
    println!(
        "single-bit error detected? {}\n",
        !verify_codeword(single_error_word, spec)
    );

    println!("Burst Error Detection");
    println!("---------------------");

    let burst_start = 7usize;
    let burst_length = 4usize;
    let burst_word = inject_burst_error(codeword, burst_start, burst_length);
    let burst_remainder = polynomial_remainder(burst_word, spec.polynomial);

    println!("burst start bit position   = {}", burst_start);
    println!("burst length               = {}", burst_length);
    println!(
        "corrupted codeword         = {}",
        format_bits_u128(burst_word, total_bits)
    );
    println!(
        "verification remainder     = {}",
        format_bits_u128(burst_remainder as u128, spec.width as usize)
    );
    println!(
        "burst error detected?      {}\n",
        !verify_codeword(burst_word, spec)
    );

    println!("Two-Bit Error Detection Example");
    println!("-------------------------------");

    let two_bit_word = flip_bit(flip_bit(codeword, 2), 11);
    let two_bit_remainder = polynomial_remainder(two_bit_word, spec.polynomial);

    println!("flipped bit positions      = 2 and 11");
    println!(
        "corrupted codeword         = {}",
        format_bits_u128(two_bit_word, total_bits)
    );
    println!(
        "verification remainder     = {}",
        format_bits_u128(two_bit_remainder as u128, spec.width as usize)
    );
    println!(
        "two-bit error detected?    {}\n",
        !verify_codeword(two_bit_word, spec)
    );

    println!("Verification Summary");
    println!("--------------------");
    println!("original codeword divisible by generator? {}", verify_codeword(codeword, spec));
    println!("single-bit error detected                 = {}", !verify_codeword(single_error_word, spec));
    println!("burst error detected                      = {}", !verify_codeword(burst_word, spec));
    println!("two-bit error detected                    = {}", !verify_codeword(two_bit_word, spec));
}
```

Program 22.5.2 demonstrates how cyclic redundancy checks transform the abstract algebra of polynomial arithmetic over $\mathbb{F}_2$ into a practical mechanism for data-integrity verification. By constructing codewords that are exactly divisible by a carefully chosen generator polynomial, CRCs create a compact redundancy representation capable of detecting a broad range of corruption patterns. This approach provides substantially stronger protection than simple additive checksums while remaining computationally efficient.

The encoding stage illustrates how a CRC remainder is generated through polynomial division and appended to the original message. The resulting codeword satisfies the divisibility condition of Equation (22.5.30), making verification straightforward. Rather than comparing entire messages, the receiver need only recompute a polynomial remainder and check whether it is zero. This simplicity is one of the reasons CRCs remain widely used in communication systems, storage devices, embedded systems, and scientific data pipelines.

The error-injection experiments highlight the theoretical detection guarantees discussed in Section 22.5.3. The single-bit corruption example demonstrates that isolated bit errors are detected reliably. The two-bit and burst-error examples further illustrate how the structure of the generator polynomial determines the classes of detectable errors. In each case, the nonzero verification remainder serves as direct evidence that the received word is no longer divisible by the generator polynomial.

More broadly, the program emphasizes the importance of algebraic structure in integrity verification. Unlike additive checksums, which operate primarily through arithmetic accumulation, CRCs exploit polynomial divisibility to identify corruption patterns that may otherwise remain hidden. This algebraic perspective forms the foundation for many modern error-detection mechanisms and provides a bridge between finite-field mathematics and practical computational infrastructure. As data volumes continue to grow and integrity requirements become increasingly stringent, CRCs remain a fundamental component of reliable numerical and scientific computing systems.

## 22.5.4. Integrity Checks in Scientific Data Pipelines

The practical use of checksums and CRCs in numerical computing is not limited to communication protocols. They are needed wherever finite data representations must be trusted across time, storage, transformation, and movement. A simulation checkpoint, for example, may contain a state vector:

$$u^n=(u_1^n,u_2^n,\ldots,u_N^n) \tag{22.5.41}$$

together with time $t_n$, step size $h_n$, solver state, random generator state, and metadata. If the checkpoint is corrupted, the restarted computation evolves from,

$$\widetilde u^n=u^n+\Delta u^n\tag{22.5.42}$$

rather than $u^n$. For a time-stepping map:

$$u^{n+1}=\Phi_h(u^n) \tag{22.5.43}$$

the restarted trajectory becomes,

$$\widetilde u^{n+k} = \Phi_h^k(\widetilde u^n) \tag{22.5.44}$$

The error after $k$ steps is:

$$\widetilde u^{n+k}-u^{n+k} = \Phi_h^k(u^n+\Delta u^n)-\Phi_h^k(u^n) \tag{22.5.45}$$

If the dynamics are sensitive, a small checkpoint corruption may grow rapidly. Integrity checks therefore protect not only stored bits but also the scientific interpretation of all subsequent computation.

A robust pipeline should apply integrity checks at multiple levels. At the byte level, a checksum or CRC verifies that a file or block has not changed. At the format level, structural checks verify that dimensions, offsets, headers, and metadata are consistent. At the numerical level, invariants verify that decoded values are plausible. For example, if a probability vector $p$ is stored, one may check,

$$p_i\ge 0,\qquad\sum_i p_i = 1\tag{22.5.46}$$

within tolerance. If a symmetric matrix is stored, one may check:

$$\|A-A^\top\|\le \tau \tag{22.5.47}$$

If a conserved quantity $Q(u)$ should remain fixed, one may check:

$$|Q(u^n)-Q(u^0)|\le \tau_Q \tag{22.5.48}$$

These numerical checks do not replace byte-level integrity checks. They complement them. A CRC can detect many accidental bit corruptions without understanding the data semantics, while an invariant check can detect semantically invalid data even when the byte string is internally consistent.

Compression introduces another integrity concern. Let,

$$z=\operatorname{Compress}(s)\tag{22.5.49}$$

and

$$\widehat s=\operatorname{Decompress}(z) \tag{22.5.50}$$

For lossless compression, correctness requires:

$$\widehat s=s \tag{22.5.51}$$

A checksum may be attached either to the compressed data $z$, the uncompressed data $s$, or both. Checking only $z$ verifies that the compressed file has not changed, but it does not independently verify the decompressed result. Checking $s$ after decompression verifies that the recovered data match the original. For critical scientific archives, both checks may be useful:

$$C_z(z)=\text{stored compressed checksum} \tag{22.5.52}$$

$$C_s(\widehat s)=\text{stored uncompressed checksum} \tag{22.5.53}$$

This distinction becomes important in Sections 22.6 and 22.7, where lossless compression methods are treated as exact representation transformations.

In distributed computations, integrity checks can also be attached to message blocks, reduction inputs, and checkpoint shards. If data are partitioned into blocks,

$$s=(s^{(1)},s^{(2)},\ldots,s^{(B)}) \tag{22.5.54}$$

one may compute block-level checks

$$C_b=C(s^{(b)}),\qquad b=1,\ldots,B \tag{22.5.55}$$

and a global check

$$C_{\mathrm{global}}=C(C_1,C_2,\ldots,C_B) \tag{22.5.56}$$

Block checks localize corruption, while the global check protects the collection. This is useful for parallel file systems, distributed checkpointing, and large numerical datasets where re-reading or re-transmitting the entire dataset is expensive.

The main limitation of ordinary checksums and CRCs is that they are designed primarily for accidental errors. They are not sufficient for adversarial tampering, provenance verification, or authenticated scientific exchange. For those tasks, stronger cryptographic hashes, signatures, authenticated logs, or provenance-aware mechanisms are needed. Nevertheless, the mathematical role of checksums and CRCs remains central: they provide fast, algebraically interpretable protection against many common data-integrity failures. Recent work on secure delivery and logging, CRC integration with correction schemes, and research data integrity reinforces the broader conclusion that integrity verification should be designed into the computational pipeline rather than appended after the fact (An et al., 2025; Choi et al., 2025; Miller and Spiegel, 2025).

This section completes the transition from bit-level enumeration to data protection. Gray codes organize how finite states are traversed. Checksums and CRCs verify whether finite strings remain unchanged. The next two sections continue the same representation theme by studying lossless compression. Section 22.6 begins with Huffman coding, where finite strings are represented more compactly through prefix codes while preserving exact decodability.

For **Program 22.5.3: Integrity Checks for Scientific Data Records**:

### Introductory Paragraph Before the Code

Following the discussion in Section 22.5.4 on integrity verification in scientific data pipelines, Program 22.5.3 provides a practical implementation of multi-level integrity checking for numerical checkpoint records. In scientific computing, data integrity extends beyond the correctness of individual bytes. A simulation checkpoint may contain state vectors, metadata, probability distributions, matrices, solver parameters, and conserved quantities that collectively define the state of a computation. Corruption of any component can alter subsequent numerical results and potentially invalidate scientific conclusions. This program demonstrates how integrity protection can be implemented simultaneously at multiple levels, including byte-level checksums, block-level verification, global consistency checks, metadata validation, and numerical invariant testing. By combining structural and semantic verification mechanisms, the implementation illustrates the principles developed in Equations (22.5.41)–(22.5.56) and shows how integrity verification can become an integral part of a scientific computing workflow rather than a separate post-processing step.

---

### Explanation of the Implementation

At the core of the implementation are the `CheckpointRecord` and `IntegrityRecord` structures, which represent the scientific dataset and its associated integrity metadata. The checkpoint record contains the numerical state of a simulation, including the state vector (u^n) from Equation (22.5.41), probability data, matrix data, temporal information, and conserved quantities. The integrity record stores the information required to verify the consistency of the checkpoint after storage or transmission. This separation mirrors the distinction between scientific content and integrity metadata that underlies practical checkpointing systems.

The function `additive_checksum_bytes` implements a lightweight byte-level checksum that serves as the basic integrity primitive throughout the program. Although simple, this checksum provides a mechanism for detecting accidental modifications of stored data. The functions `checksum_f64_slice` and `checksum_matrix` extend this idea to floating-point vectors and matrices by converting numerical values into byte sequences before computing the checksum. These functions provide practical realizations of the byte-level integrity checks discussed in Section 22.5.4.

The function `block_checksums` implements the block-partitioning strategy described by Equations (22.5.54) and (22.5.55). The state vector is divided into fixed-size blocks, and an independent checksum is computed for each block. This decomposition allows corruption to be localized to a particular region of the dataset rather than merely detecting that some corruption has occurred. The companion function `global_checksum_from_blocks` implements Equation (22.5.56) by constructing a higher-level checksum from the collection of block checks. Together, these functions provide a hierarchical integrity mechanism suitable for large scientific datasets and distributed storage systems.

The function `full_checkpoint_checksum` computes a checksum over the complete checkpoint record. Unlike the block-level checks, which focus on the state vector, this function incorporates metadata, simulation parameters, probability data, matrix data, and conserved quantities. The resulting checksum provides a single integrity indicator for the entire checkpoint and serves as the primary byte-level verification mechanism.

The function `build_integrity_record` constructs the integrity metadata associated with a checkpoint. In addition to storing checksum information, it records structural properties such as vector dimensions and matrix sizes. These structural quantities support the format-level validation discussed in Section 22.5.4 by allowing later verification that the decoded dataset remains consistent with its expected organization.

The function `check_metadata` implements structural integrity verification. Rather than examining numerical values directly, it confirms that dimensions, identifiers, and matrix shapes remain consistent with the stored integrity record. This corresponds to the format-level checks described in Section 22.5.4, where headers, dimensions, and metadata are verified independently of the numerical contents.

The functions `verify_full_checksum`, `verify_blocks`, and `verify_global_block_checksum` implement the three levels of checksum validation used by the program. The first verifies the integrity of the entire checkpoint, the second identifies which individual blocks have changed, and the third verifies the consistency of the block-check collection itself. Together, these functions demonstrate how integrity protection can be organized hierarchically in large scientific datasets.

The function `check_probability_vector` implements the probability constraints of Equation (22.5.46). It verifies both non-negativity and normalization, thereby ensuring that the decoded probability vector remains semantically meaningful. This illustrates the distinction emphasized in Section 22.5.4 between byte-level correctness and numerical plausibility.

The function `symmetry_error` implements the matrix-consistency check described by Equation (22.5.47). By measuring the deviation from symmetry, the function provides a numerical diagnostic that can reveal corruption or reconstruction errors that may not be apparent from checksum information alone. Similarly, the function `conserved_quantity` computes a conserved quantity whose preservation can be tested using Equation (22.5.48). Such invariant checks provide a semantic layer of verification that complements the underlying byte-level integrity mechanisms.

The functions `corrupt_state_value`, `corrupt_probability`, and `corrupt_matrix_symmetry` deliberately introduce different forms of corruption into the checkpoint record. These functions allow the program to simulate realistic failure modes and demonstrate how the various integrity checks respond to each type of corruption.

The `main` function serves as a comprehensive demonstration of integrity verification within a scientific data pipeline. It first constructs a valid checkpoint record and generates the corresponding integrity metadata. The original record is then verified using byte-level, block-level, structural, and numerical checks. Subsequently, controlled corruption is introduced into the state vector, probability vector, and matrix data. The program then re-evaluates all integrity diagnostics and shows how corruption is detected at multiple levels. This experiment demonstrates that checksum-based mechanisms and invariant-based verification serve complementary roles, providing both syntactic and semantic protection for scientific data.

---

### Text After the Code Block Including Concluding Remarks

```rust
// Program 22.5.3: Integrity Checks for Scientific Data Records
//
// Problem statement:
// Demonstrate byte-level, block-level, global, structural, and numerical
// integrity checks for a scientific checkpoint record. The program verifies
// metadata consistency, localizes corrupted blocks, and checks numerical
// invariants such as probability normalization, matrix symmetry, and conserved
// quantities.

#[derive(Clone)]
struct CheckpointRecord {
    name: String,
    version: String,
    time: f64,
    step_size: f64,
    step_index: usize,
    state: Vec<f64>,
    probability: Vec<f64>,
    matrix: Vec<Vec<f64>>,
    conserved_initial: f64,
}

#[derive(Clone)]
struct IntegrityRecord {
    name: String,
    version: String,
    state_len: usize,
    probability_len: usize,
    matrix_rows: usize,
    matrix_cols: usize,
    full_checksum: u32,
    block_checksums: Vec<u32>,
    global_block_checksum: u32,
}

fn additive_checksum_bytes(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .fold(0_u32, |acc, &b| acc.wrapping_add(b as u32))
}

fn checksum_f64_slice(values: &[f64]) -> u32 {
    let mut bytes = Vec::with_capacity(values.len() * 8);

    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    additive_checksum_bytes(&bytes)
}

fn checksum_matrix(matrix: &[Vec<f64>]) -> u32 {
    let mut bytes = Vec::new();

    for row in matrix {
        for &v in row {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }

    additive_checksum_bytes(&bytes)
}

fn block_checksums(values: &[f64], block_size: usize) -> Vec<u32> {
    values
        .chunks(block_size)
        .map(checksum_f64_slice)
        .collect()
}

fn global_checksum_from_blocks(blocks: &[u32]) -> u32 {
    let mut bytes = Vec::with_capacity(blocks.len() * 4);

    for &c in blocks {
        bytes.extend_from_slice(&c.to_le_bytes());
    }

    additive_checksum_bytes(&bytes)
}

fn full_checkpoint_checksum(record: &CheckpointRecord) -> u32 {
    let mut checksum = 0_u32;

    checksum = checksum.wrapping_add(additive_checksum_bytes(record.name.as_bytes()));
    checksum = checksum.wrapping_add(additive_checksum_bytes(record.version.as_bytes()));
    checksum = checksum.wrapping_add(checksum_f64_slice(&[record.time, record.step_size]));
    checksum = checksum.wrapping_add(record.step_index as u32);
    checksum = checksum.wrapping_add(checksum_f64_slice(&record.state));
    checksum = checksum.wrapping_add(checksum_f64_slice(&record.probability));
    checksum = checksum.wrapping_add(checksum_matrix(&record.matrix));
    checksum = checksum.wrapping_add(checksum_f64_slice(&[record.conserved_initial]));

    checksum
}

fn build_integrity_record(record: &CheckpointRecord, block_size: usize) -> IntegrityRecord {
    let matrix_rows = record.matrix.len();
    let matrix_cols = if matrix_rows > 0 {
        record.matrix[0].len()
    } else {
        0
    };

    let blocks = block_checksums(&record.state, block_size);

    IntegrityRecord {
        name: record.name.clone(),
        version: record.version.clone(),
        state_len: record.state.len(),
        probability_len: record.probability.len(),
        matrix_rows,
        matrix_cols,
        full_checksum: full_checkpoint_checksum(record),
        global_block_checksum: global_checksum_from_blocks(&blocks),
        block_checksums: blocks,
    }
}

fn check_metadata(record: &CheckpointRecord, integrity: &IntegrityRecord) -> bool {
    let matrix_rows = record.matrix.len();
    let matrix_cols = if matrix_rows > 0 {
        record.matrix[0].len()
    } else {
        0
    };

    record.name == integrity.name
        && record.version == integrity.version
        && record.state.len() == integrity.state_len
        && record.probability.len() == integrity.probability_len
        && matrix_rows == integrity.matrix_rows
        && matrix_cols == integrity.matrix_cols
}

fn verify_full_checksum(record: &CheckpointRecord, integrity: &IntegrityRecord) -> bool {
    full_checkpoint_checksum(record) == integrity.full_checksum
}

fn verify_blocks(
    record: &CheckpointRecord,
    integrity: &IntegrityRecord,
    block_size: usize,
) -> Vec<usize> {
    let current = block_checksums(&record.state, block_size);

    current
        .iter()
        .zip(integrity.block_checksums.iter())
        .enumerate()
        .filter_map(|(i, (&now, &stored))| if now != stored { Some(i) } else { None })
        .collect()
}

fn verify_global_block_checksum(
    record: &CheckpointRecord,
    integrity: &IntegrityRecord,
    block_size: usize,
) -> bool {
    let current_blocks = block_checksums(&record.state, block_size);
    global_checksum_from_blocks(&current_blocks) == integrity.global_block_checksum
}

fn check_probability_vector(p: &[f64], tolerance: f64) -> (bool, f64, bool) {
    let all_nonnegative = p.iter().all(|&x| x >= 0.0);
    let sum = p.iter().sum::<f64>();
    let normalized = (sum - 1.0).abs() <= tolerance;

    (all_nonnegative && normalized, sum, all_nonnegative)
}

fn symmetry_error(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    let mut max_error = 0.0_f64;

    for i in 0..n {
        for j in 0..n {
            let diff = (matrix[i][j] - matrix[j][i]).abs();
            max_error = max_error.max(diff);
        }
    }

    max_error
}

fn conserved_quantity(state: &[f64]) -> f64 {
    state.iter().map(|x| x * x).sum()
}

fn corrupt_state_value(record: &mut CheckpointRecord, index: usize, perturbation: f64) {
    record.state[index] += perturbation;
}

fn corrupt_probability(record: &mut CheckpointRecord, index: usize, new_value: f64) {
    record.probability[index] = new_value;
}

fn corrupt_matrix_symmetry(record: &mut CheckpointRecord, i: usize, j: usize, perturbation: f64) {
    record.matrix[i][j] += perturbation;
}

fn print_block_results(label: &str, failed_blocks: &[usize]) {
    println!("{}", label);
    if failed_blocks.is_empty() {
        println!("failed blocks                 = none");
    } else {
        println!("failed blocks                 = {:?}", failed_blocks);
    }
}

fn main() {
    println!("Program 22.5.3: Integrity Checks for Scientific Data Records");
    println!("============================================================\n");

    let block_size = 4usize;
    let tolerance = 1.0e-12;

    let state = vec![
        0.10, 0.20, 0.30, 0.40,
        0.50, 0.60, 0.70, 0.80,
        0.90, 1.00, 1.10, 1.20,
    ];

    let probability = vec![0.10, 0.20, 0.30, 0.40];

    let matrix = vec![
        vec![2.0, 0.5, 0.1],
        vec![0.5, 3.0, 0.7],
        vec![0.1, 0.7, 4.0],
    ];

    let conserved_initial = conserved_quantity(&state);

    let original = CheckpointRecord {
        name: "checkpoint_step_120".to_string(),
        version: "v1".to_string(),
        time: 0.75,
        step_size: 0.00625,
        step_index: 120,
        state,
        probability,
        matrix,
        conserved_initial,
    };

    let integrity = build_integrity_record(&original, block_size);

    println!("Stored Integrity Record");
    println!("-----------------------");
    println!("name                         = {}", integrity.name);
    println!("version                      = {}", integrity.version);
    println!("state length                 = {}", integrity.state_len);
    println!("probability length           = {}", integrity.probability_len);
    println!("matrix shape                 = {} x {}", integrity.matrix_rows, integrity.matrix_cols);
    println!("full checksum                = {}", integrity.full_checksum);
    println!("block checksums              = {:?}", integrity.block_checksums);
    println!("global block checksum        = {}\n", integrity.global_block_checksum);

    println!("Verification of Original Record");
    println!("-------------------------------");
    println!("metadata valid?               = {}", check_metadata(&original, &integrity));
    println!("full checksum valid?          = {}", verify_full_checksum(&original, &integrity));
    println!(
        "global block checksum valid?  = {}",
        verify_global_block_checksum(&original, &integrity, block_size)
    );

    let failed_blocks = verify_blocks(&original, &integrity, block_size);
    print_block_results("block-level verification", &failed_blocks);

    let (prob_ok, prob_sum, prob_nonnegative) =
        check_probability_vector(&original.probability, tolerance);
    let sym_error = symmetry_error(&original.matrix);
    let q_now = conserved_quantity(&original.state);

    println!("probability nonnegative?      = {}", prob_nonnegative);
    println!("probability sum               = {:.16e}", prob_sum);
    println!("probability invariant valid?  = {}", prob_ok);
    println!("matrix symmetry error         = {:.16e}", sym_error);
    println!("matrix symmetry valid?        = {}", sym_error <= tolerance);
    println!(
        "conserved quantity error      = {:.16e}",
        (q_now - original.conserved_initial).abs()
    );
    println!(
        "conserved quantity valid?     = {}\n",
        (q_now - original.conserved_initial).abs() <= tolerance
    );

    println!("Corrupted Checkpoint Test");
    println!("-------------------------");

    let mut corrupted = original.clone();
    corrupt_state_value(&mut corrupted, 6, 1.0e-3);
    corrupt_probability(&mut corrupted, 2, -0.30);
    corrupt_matrix_symmetry(&mut corrupted, 0, 2, 5.0e-2);

    println!("metadata valid?               = {}", check_metadata(&corrupted, &integrity));
    println!("full checksum valid?          = {}", verify_full_checksum(&corrupted, &integrity));
    println!(
        "global block checksum valid?  = {}",
        verify_global_block_checksum(&corrupted, &integrity, block_size)
    );

    let failed_blocks = verify_blocks(&corrupted, &integrity, block_size);
    print_block_results("block-level verification", &failed_blocks);

    let (prob_ok, prob_sum, prob_nonnegative) =
        check_probability_vector(&corrupted.probability, tolerance);
    let sym_error = symmetry_error(&corrupted.matrix);
    let q_now = conserved_quantity(&corrupted.state);

    println!("probability nonnegative?      = {}", prob_nonnegative);
    println!("probability sum               = {:.16e}", prob_sum);
    println!("probability invariant valid?  = {}", prob_ok);
    println!("matrix symmetry error         = {:.16e}", sym_error);
    println!("matrix symmetry valid?        = {}", sym_error <= tolerance);
    println!(
        "conserved quantity error      = {:.16e}",
        (q_now - corrupted.conserved_initial).abs()
    );
    println!(
        "conserved quantity valid?     = {}",
        (q_now - corrupted.conserved_initial).abs() <= tolerance
    );

    println!("\nVerification Summary");
    println!("--------------------");
    println!("byte-level checks detect changed stored values");
    println!("block-level checks localize corruption to affected shards");
    println!("global block checksum protects the collection of block checks");
    println!("metadata checks verify structural consistency");
    println!("numerical invariant checks detect semantic inconsistencies");
}
```

Program 22.5.3 demonstrates a practical framework for integrity verification in scientific computing environments by combining byte-level, structural, and numerical validation mechanisms. This approach reflects the central theme of Section 22.5.4: integrity protection should be embedded throughout the computational pipeline rather than applied only after corruption is suspected.

The block-level and global checksum mechanisms illustrate how large datasets can be protected efficiently while still allowing corruption to be localized. Instead of merely reporting that a checkpoint has changed, the block-based strategy identifies which portion of the state vector has been affected. This capability becomes increasingly important for large-scale simulations, distributed checkpoint systems, and parallel file-storage architectures.

The numerical invariant checks demonstrate that byte-level integrity alone is not always sufficient. A dataset may be internally consistent from a checksum perspective yet still violate fundamental scientific constraints. Probability normalization, matrix symmetry, and conservation laws provide examples of semantic verification that complements traditional integrity mechanisms. Together, these checks help ensure that recovered data remain both syntactically correct and scientifically meaningful.

The hierarchical design of the implementation mirrors the architecture of many modern scientific data-management systems. Byte-level checksums provide rapid corruption detection, block-level checks support localization, metadata validation ensures structural consistency, and invariant tests verify scientific plausibility. This layered strategy creates a robust foundation for reliable numerical computation and prepares the way for subsequent sections, where integrity-preserving compression and representation transformations become additional components of the computational infrastructure.

# 22.6. Huffman Coding and Lossless Compression

Lossless compression is the problem of representing data more compactly while preserving exact recoverability. In numerical computing, this requirement arises in checkpoint files, diagnostic logs, sparse data structures, symbolic metadata, mesh connectivity, categorical traces, integer-valued arrays, bit masks, and transformed scientific data. When the data must be reproduced exactly, compression is not an approximation method. It is a reversible change of representation. If $s$ is the original finite string and $z$ is its compressed representation, then a lossless compressor and decompressor must satisfy:

$$z=\operatorname{Compress}(s),\qquad s=\operatorname{Decompress}(z) \tag{22.6.1}$$

Equivalently,

$$\operatorname{Decompress}(\operatorname{Compress}(s))=s \tag{22.6.2}$$

This exact recovery condition distinguishes lossless compression from lossy compression, where one accepts an approximation $\widehat s\neq s$ in exchange for a smaller representation. In the present section, the emphasis is on Huffman coding, the classical method for constructing optimal prefix codes for known symbol frequencies. The broader modern lesson is that Huffman coding is often one stage inside a larger compression pipeline involving prediction, blocking, transforms, bit reorganization, or format-aware preprocessing. Recent literature supports this unified view of entropy coding for massive data, scientific floating-point arrays, and structured numerical datasets (Auli-Llinas, 2023; Azami et al., 2025; Lyu et al., 2025).

## 22.6.1. Entropy, Codes, and Exact Decodability

Let a discrete source emit symbols from a finite alphabet,

$$\mathcal A=\{a_1,a_2,\ldots,a_m\} \tag{22.6.3}$$

with probabilities,

$$p_i=\Pr(a_i),\qquad p_i\ge 0,\qquad \sum_{i=1}^{m}p_i=1 \tag{22.6.4}$$

The Shannon entropy of the source is:

$$H(p) = -\sum_{i=1}^{m}p_i\log_2 p_i \tag{22.6.5}$$

Entropy measures the average information content per symbol under the probability model $p$. A symbol with high probability carries less information than a rare symbol. The ideal code length associated with symbol $a_i$ is therefore:

$$\ell_i^\ast=-\log_2 p_i \tag{22.6.6}$$

These ideal lengths are generally not integers, while a binary code must assign each symbol a finite bit string. If symbol $a_i$ is assigned a binary codeword $c_i$ of length $\ell_i$, then the average code length is:

$$L = \sum_{i=1}^{m}p_i\ell_i \tag{22.6.7}$$

A good lossless code attempts to make $L$ as close as possible to $H(p)$, subject to exact decodability.

A code is nonsingular if distinct symbols have distinct codewords:

$$a_i\ne a_j\quad\Longrightarrow\quad c_i\ne c_j \tag{22.6.8}$$

Nonsingularity is not enough for sequences. A code is uniquely decodable if every finite encoded bit string corresponds to at most one sequence of source symbols. A stronger and more practical condition is the prefix condition: no codeword is the prefix of another codeword. If $c_i$ and $c_j$ are distinct codewords, then:

$$c_i \not\prec c_j\qquad\text{and}\qquad c_j \not\prec c_i \tag{22.6.9}$$

where $c_i\prec c_j$ means that $c_i$ is a proper prefix of $c_j$. Prefix codes can be decoded instantaneously from left to right because the decoder knows that a symbol has ended as soon as a valid leaf codeword is reached.

Binary prefix codes correspond to leaves of a binary tree. Each left or right branch contributes one bit, and the length $\ell_i$ is the depth of the leaf for symbol $a_i$. The Kraft inequality characterizes feasible prefix-code lengths:

$$\sum_{i=1}^{m}2^{-\ell_i}\le 1 \tag{22.6.10}$$

For a complete binary prefix tree, equality holds:

$$\sum_{i=1}^{m}2^{-\ell_i}=1 \tag{22.6.11}$$

The entropy lower bound states that any uniquely decodable binary code satisfies,

$$L\ge H(p) \tag{22.6.12}$$

A simple way to see why this is natural is to define:

$$q_i=\frac{2^{-\ell_i}}{\sum_{j=1}^{m}2^{-\ell_j}} \tag{22.6.13}$$

The nonnegativity of relative entropy,

$$\sum_{i=1}^{m}p_i\log_2\frac{p_i}{q_i}\ge 0 \tag{22.6.14}$$

implies that the expected length cannot fall below the entropy when the code lengths satisfy the Kraft constraint. Thus, entropy is not merely a heuristic measure. It is a mathematical lower bound on the expected number of bits per source symbol.

For any source distribution, one can choose integer lengths,

$$\ell_i=\lceil -\log_2 p_i\rceil \tag{22.6.15}$$

which satisfy Kraft’s inequality and give,

$$H(p)\le L < H(p)+1 \tag{22.6.16}$$

Huffman coding improves this construction by finding an optimal prefix code for the given probabilities or frequencies. It minimizes $L$ among all binary prefix codes for the same symbol model.

## 22.6.2. Huffman Trees and Optimal Prefix Codes

Huffman coding builds an optimal prefix tree by repeatedly combining the two least probable symbols or subtrees. Suppose the source probabilities are:

$$p_1,p_2,\ldots,p_m \tag{22.6.17}$$

At each step, select two currently least probable nodes, say with weights $w_a$ and $w_b$, and replace them by a new node with weight:

$$w_{ab}=w_a+w_b \tag{22.6.18}$$

This process continues until a single root remains. Assigning binary labels $0$ and $1$ to the two branches from each internal node gives a prefix code. The codeword for a symbol is the path from the root to its leaf.

The optimality of the construction rests on a simple structural fact. In an optimal binary prefix tree, two least probable symbols can be placed as siblings at maximum depth. If $a_i$ and $a_j$ are two least probable symbols, then there exists an optimal tree in which they have the same parent and depths,

$$\ell_i=\ell_j \tag{22.6.19}$$

Collapsing these two sibling leaves into a single compound symbol with probability,

$$p_{ij}=p_i+p_j\tag{22.6.20}$$

reduces the problem from $m$ symbols to $m-1$ symbols. If the reduced tree is optimal for the reduced alphabet, then expanding the compound symbol back into two children gives an optimal tree for the original alphabet. This recursive optimal-substructure argument justifies the greedy Huffman algorithm.

The expected length of the Huffman code is:

$$L_{\mathrm{H}} = \sum_{i=1}^{m}p_i\ell_i^{\mathrm{H}} \tag{22.6.21}$$

where $\ell_i^{\mathrm{H}}$ is the depth of symbol $a_i$ in the Huffman tree. Since Huffman coding is optimal among binary prefix codes,

$$L_{\mathrm{H}}\le\sum_{i=1}^{m}p_i\ell_i\tag{22.6.22}$$

for any other binary prefix code with lengths $\ell_i$. Combining this with the entropy lower bound gives:

$$H(p)\le L_{\mathrm{H}}<H(p)+1 \tag{22.6.23}$$

The gap,

$$R_{\mathrm{H}}=L_{\mathrm{H}}-H(p)\tag{22.6.24}$$

is the redundancy of the Huffman code. The redundancy is small when the probabilities align well with powers of two. If,

$$p_i=2^{-\ell_i}\tag{22.6.25}$$

for integer $\ell_i$ satisfying Kraft equality, then the entropy can be achieved exactly:

$$L_{\mathrm{H}}=H(p) \tag{22.6.26}$$

In general, the integer-length constraint prevents perfect equality.

When data are finite rather than generated by a known probability distribution, probabilities are estimated from symbol counts. If symbol $a_i$ occurs $f_i$ times in a sequence of length $N$, then,

$$\widehat p_i=\frac{f_i}{N},\qquad\sum_{i=1}^{m}f_i=N \tag{22.6.27}$$

The empirical entropy is:

$$\widehat H = -\sum_{i:f_i>0}\widehat p_i\log_2 \widehat p_i \tag{22.6.28}$$

A static Huffman code built from the data has expected length:

$$\widehat L_{\mathrm{H}} = \sum_{i:f_i>0}\widehat p_i\ell_i^{\mathrm{H}} \tag{22.6.29}$$

The total encoded payload length is approximately,

$$B_{\mathrm{payload}} = N\widehat L_{\mathrm{H}} \tag{22.6.30}$$

However, the decoder must also know the code tree or enough information to reconstruct it. If $B_{\mathrm{model}}$ bits are required to store the model, then the total compressed size is:

$$B_{\mathrm{total}} = B_{\mathrm{payload}}+B_{\mathrm{model}} \tag{22.6.31}$$

Compression is beneficial only if:

$$B_{\mathrm{total}} < B_{\mathrm{raw}} \tag{22.6.32}$$

where $B_{\mathrm{raw}}$ is the uncompressed size. This inequality is important in numerical workflows because small diagnostic files, short messages, or nearly uniform data may not compress profitably once model overhead is included.

A canonical Huffman code stores only the code lengths, not the full tree. Symbols are ordered by increasing length and then by a fixed symbol order. The first code of a given length is assigned deterministically, and subsequent codes are assigned consecutively. If $n_\ell$ is the number of codewords of length $\ell$, the canonical construction satisfies Kraft’s condition through,

$$\sum_{\ell} n_\ell 2^{-\ell}\le 1 \tag{22.6.33}$$

Canonical codes are preferred in file formats and numerical pipelines because they make decoding tables compact, deterministic, and reproducible.

### Rust Implementation

Following the discussion in Section 22.6.2 on entropy, prefix codes, Huffman trees, and optimal lossless coding, Program 22.6.1 provides a practical implementation of Huffman tree construction and prefix-code-based compression. In lossless compression, the objective is not to approximate data but to represent it more compactly while preserving exact recoverability, as required by Equations (22.6.1) and (22.6.2). Huffman coding achieves this objective by assigning shorter codewords to more frequent symbols and longer codewords to less frequent symbols, thereby reducing the average number of bits required per symbol. This program constructs a Huffman tree from empirical symbol frequencies, generates optimal prefix codes, performs lossless encoding and decoding, verifies exact reconstruction, and evaluates information-theoretic quantities such as entropy, average code length, redundancy, payload size, and compression ratio. The implementation therefore provides a concrete realization of the theoretical framework developed in Equations (22.6.3)–(22.6.33).

At the core of the implementation is the `HuffmanTree` enumeration, which represents the binary prefix tree underlying Huffman coding. A tree node may be either a leaf containing a source symbol and its frequency or an internal node containing two child subtrees. This structure directly corresponds to the binary-tree interpretation of prefix codes discussed in Section 22.6.1, where each leaf represents a symbol and the depth of the leaf determines the code length associated with that symbol. The methods `frequency` and `min_symbol` provide auxiliary information required during tree construction and deterministic ordering of nodes.

The `QueueItem` structure and its associated ordering operators implement the priority queue used by the Huffman algorithm. The priority queue repeatedly selects the two least probable nodes, precisely following the greedy construction described by Equations (22.6.17)–(22.6.20). By defining the ordering in terms of node frequency and symbol order, the implementation produces deterministic and reproducible Huffman trees, an important property for numerical workflows and data-processing pipelines.

The function `frequency_table` computes the empirical symbol counts $f_i$ used in Equation (22.6.27). Given a finite data sequence, it constructs the frequency model required for Huffman coding. These empirical frequencies provide estimates of the source probabilities and form the basis for all subsequent entropy and code-length calculations.

The function `build_huffman_tree` implements the Huffman tree-construction algorithm described in Section 22.6.2. Beginning with a collection of leaf nodes weighted by their frequencies, the algorithm repeatedly removes the two least frequent nodes, combines them into a new internal node whose weight is the sum of the two child weights according to Equation (22.6.18), and reinserts the combined node into the priority queue. The process terminates when a single root node remains. This recursive merging procedure embodies the optimal-substructure argument leading to Huffman optimality.

The function `build_codes` traverses the completed Huffman tree and assigns binary codewords to each symbol. A left branch contributes a 0 bit and a right branch contributes a 1 bit. The resulting codeword for a symbol is the path from the root to its corresponding leaf. Consequently, the code length is equal to the leaf depth, yielding the Huffman lengths $\ell_i^{\mathrm H}$ appearing in Equations (22.6.21)–(22.6.23).

The functions `encode` and `decode` implement the forward and inverse transformations required by Equations (22.6.1) and (22.6.2). The encoder replaces each source symbol by its corresponding Huffman codeword and concatenates the resulting bit strings. The decoder traverses the Huffman tree according to the encoded bits until a leaf node is reached, at which point the corresponding symbol is emitted. The exact equality of the decoded sequence and the original sequence verifies lossless recoverability.

The function `verify_prefix_property` confirms the prefix condition stated in Equation (22.6.9). It checks that no codeword is a prefix of any other codeword. This property guarantees instantaneous decodability and ensures that symbol boundaries can be identified unambiguously during decoding.

The function `empirical_entropy` computes the empirical entropy $\widehat H$ defined in Equation (22.6.28). Using the observed symbol frequencies, it measures the theoretical lower bound on the average number of bits per symbol. The companion function `average_code_length` computes the Huffman average length $\widehat L_{\mathrm H}$ defined in Equation (22.6.29). Comparing these two quantities provides a direct measurement of coding efficiency and allows the redundancy $R_{\mathrm H}$ of Equation (22.6.24) to be evaluated.

The functions `raw_bits` and `model_bits` estimate storage requirements for both the original and compressed representations. The model-overhead estimate corresponds to the discussion surrounding Equations (22.6.30)–(22.6.32), where the encoded payload alone is insufficient to characterize compression effectiveness because the decoder must also receive enough information to reconstruct the coding model.

The `main` function serves as a complete demonstration of Huffman compression. It constructs an empirical frequency model from a sample data sequence, builds the corresponding Huffman tree, generates the code table, and performs lossless encoding and decoding. The program then computes entropy, average code length, redundancy, payload size, model overhead, and compression ratios. Finally, it verifies exact reconstruction and checks the prefix-code property. Together, these operations illustrate how Huffman coding transforms symbol-frequency information into a compact, exactly decodable binary representation while approaching the entropy limit established by information theory.

```rust
// Program 22.6.1: Huffman Tree Construction and Prefix-Code Encoding
//
// Problem statement:
// Construct a Huffman prefix code from empirical symbol frequencies, encode
// and decode a finite data string, verify the prefix-code property, and report
// entropy, average code length, redundancy, payload size, model overhead, and
// compression ratio.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Clone)]
enum HuffmanTree {
    Leaf {
        symbol: u8,
        frequency: usize,
    },
    Internal {
        frequency: usize,
        left: Box<HuffmanTree>,
        right: Box<HuffmanTree>,
    },
}

impl HuffmanTree {
    fn frequency(&self) -> usize {
        match self {
            HuffmanTree::Leaf { frequency, .. } => *frequency,
            HuffmanTree::Internal { frequency, .. } => *frequency,
        }
    }

    fn min_symbol(&self) -> u8 {
        match self {
            HuffmanTree::Leaf { symbol, .. } => *symbol,
            HuffmanTree::Internal { left, right, .. } => left.min_symbol().min(right.min_symbol()),
        }
    }
}

#[derive(Clone)]
struct QueueItem {
    tree: HuffmanTree,
}

impl Eq for QueueItem {}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.tree.frequency() == other.tree.frequency()
            && self.tree.min_symbol() == other.tree.min_symbol()
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .tree
            .frequency()
            .cmp(&self.tree.frequency())
            .then_with(|| other.tree.min_symbol().cmp(&self.tree.min_symbol()))
    }
}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn frequency_table(data: &[u8]) -> HashMap<u8, usize> {
    let mut frequencies = HashMap::new();

    for &symbol in data {
        *frequencies.entry(symbol).or_insert(0) += 1;
    }

    frequencies
}

fn build_huffman_tree(frequencies: &HashMap<u8, usize>) -> Option<HuffmanTree> {
    let mut heap = BinaryHeap::new();

    for (&symbol, &frequency) in frequencies {
        heap.push(QueueItem {
            tree: HuffmanTree::Leaf { symbol, frequency },
        });
    }

    while heap.len() > 1 {
        let first = heap.pop().unwrap().tree;
        let second = heap.pop().unwrap().tree;

        let combined_frequency = first.frequency() + second.frequency();

        heap.push(QueueItem {
            tree: HuffmanTree::Internal {
                frequency: combined_frequency,
                left: Box::new(first),
                right: Box::new(second),
            },
        });
    }

    heap.pop().map(|item| item.tree)
}

fn build_codes(tree: &HuffmanTree, prefix: String, codes: &mut HashMap<u8, String>) {
    match tree {
        HuffmanTree::Leaf { symbol, .. } => {
            let code = if prefix.is_empty() {
                "0".to_string()
            } else {
                prefix
            };
            codes.insert(*symbol, code);
        }
        HuffmanTree::Internal { left, right, .. } => {
            build_codes(left, format!("{}0", prefix), codes);
            build_codes(right, format!("{}1", prefix), codes);
        }
    }
}

fn encode(data: &[u8], codes: &HashMap<u8, String>) -> String {
    let mut encoded = String::new();

    for &symbol in data {
        encoded.push_str(codes.get(&symbol).expect("missing Huffman code"));
    }

    encoded
}

fn decode(bits: &str, tree: &HuffmanTree) -> Vec<u8> {
    let mut output = Vec::new();
    let mut node = tree;

    for bit in bits.chars() {
        if let HuffmanTree::Internal { left, right, .. } = node {
            node = if bit == '0' { left } else { right };
        }

        if let HuffmanTree::Leaf { symbol, .. } = node {
            output.push(*symbol);
            node = tree;
        }
    }

    output
}

fn verify_prefix_property(codes: &HashMap<u8, String>) -> bool {
    let codewords: Vec<&String> = codes.values().collect();

    for i in 0..codewords.len() {
        for j in 0..codewords.len() {
            if i != j && codewords[j].starts_with(codewords[i]) {
                return false;
            }
        }
    }

    true
}

fn empirical_entropy(frequencies: &HashMap<u8, usize>, total: usize) -> f64 {
    frequencies
        .values()
        .map(|&frequency| {
            let p = frequency as f64 / total as f64;
            -p * p.log2()
        })
        .sum()
}

fn average_code_length(
    frequencies: &HashMap<u8, usize>,
    codes: &HashMap<u8, String>,
    total: usize,
) -> f64 {
    frequencies
        .iter()
        .map(|(&symbol, &frequency)| {
            let p = frequency as f64 / total as f64;
            p * codes[&symbol].len() as f64
        })
        .sum()
}

fn raw_bits(data: &[u8]) -> usize {
    8 * data.len()
}

fn model_bits(codes: &HashMap<u8, String>) -> usize {
    // Simple educational model cost:
    // 8 bits for the symbol and 8 bits for the code length per alphabet entry.
    16 * codes.len()
}

fn print_code_table(frequencies: &HashMap<u8, usize>, codes: &HashMap<u8, String>) {
    let mut symbols: Vec<u8> = frequencies.keys().copied().collect();
    symbols.sort();

    println!("Huffman Code Table");
    println!("------------------");
    println!(
        "{:>8} {:>12} {:>14} {:>14}",
        "symbol", "frequency", "code length", "codeword"
    );

    for symbol in symbols {
        println!(
            "{:>8} {:>12} {:>14} {:>14}",
            symbol as char,
            frequencies[&symbol],
            codes[&symbol].len(),
            codes[&symbol]
        );
    }
}

fn main() {
    println!("Program 22.6.1: Huffman Tree Construction and Prefix-Code Encoding");
    println!("==================================================================\n");

    let data = b"AAAABBBBBBBBBCCCCCCCDDDDEEF";

    println!("Input Data");
    println!("----------");
    println!("message                 = {}", String::from_utf8_lossy(data));
    println!("number of symbols        = {}", data.len());
    println!("raw size                 = {} bits\n", raw_bits(data));

    let frequencies = frequency_table(data);
    let tree = build_huffman_tree(&frequencies).expect("nonempty input required");

    let mut codes = HashMap::new();
    build_codes(&tree, String::new(), &mut codes);

    print_code_table(&frequencies, &codes);

    let encoded = encode(data, &codes);
    let decoded = decode(&encoded, &tree);

    let entropy = empirical_entropy(&frequencies, data.len());
    let avg_len = average_code_length(&frequencies, &codes, data.len());
    let redundancy = avg_len - entropy;

    let payload_bits = encoded.len();
    let model_bits = model_bits(&codes);
    let total_bits = payload_bits + model_bits;
    let compression_ratio_payload = raw_bits(data) as f64 / payload_bits as f64;
    let compression_ratio_total = raw_bits(data) as f64 / total_bits as f64;

    println!("\nEncoding and Decoding");
    println!("---------------------");
    println!("encoded bitstream         = {}", encoded);
    println!("payload size              = {} bits", payload_bits);
    println!("model overhead estimate   = {} bits", model_bits);
    println!("total compressed estimate = {} bits", total_bits);
    println!("decoded equals input?     = {}", decoded == data);

    println!("\nInformation-Theoretic Diagnostics");
    println!("---------------------------------");
    println!("empirical entropy H        = {:.6} bits/symbol", entropy);
    println!("average Huffman length L   = {:.6} bits/symbol", avg_len);
    println!("redundancy L-H             = {:.6} bits/symbol", redundancy);
    println!("prefix-code property valid = {}", verify_prefix_property(&codes));

    println!("\nCompression Diagnostics");
    println!("-----------------------");
    println!(
        "payload compression ratio = {:.6}",
        compression_ratio_payload
    );
    println!(
        "total compression ratio   = {:.6}",
        compression_ratio_total
    );
    println!(
        "payload beneficial?       = {}",
        payload_bits < raw_bits(data)
    );
    println!(
        "including model beneficial? = {}",
        total_bits < raw_bits(data)
    );
}
```

Program 22.6.1 demonstrates the complete workflow of classical Huffman coding, beginning with empirical symbol frequencies and ending with exact reconstruction of the original data sequence. This workflow embodies the central principle of lossless compression developed in Section 22.6: data representation can be made more compact without sacrificing exact recoverability.

The computed entropy and average code length provide a direct comparison between the theoretical information content of the source and the practical performance of the Huffman code. As predicted by Equations (22.6.23) and (22.6.24), the average Huffman length lies close to the entropy, with only a small redundancy introduced by the requirement that code lengths be integers. The output therefore illustrates the near-optimality of Huffman coding for a known symbol model.

The prefix-code verification and successful decoding experiment demonstrate the importance of the prefix condition for practical compression systems. Because no codeword is the prefix of another, the encoded bit stream can be decoded sequentially without ambiguity. This property allows Huffman codes to achieve efficient representation while preserving exact decodability.

The compression diagnostics further emphasize that practical compression performance depends on more than the encoded payload alone. The storage cost of the coding model must also be considered, particularly for short messages or small datasets. This observation motivates the use of canonical Huffman codes and other compact model representations in practical file formats and scientific workflows. More broadly, the program illustrates how information-theoretic principles, combinatorial tree structures, and exact decoding algorithms combine to form one of the most important foundations of modern lossless compression systems.

## 22.6.3. Compression Pipelines for Numerical Data

Raw numerical arrays are often not well suited to direct Huffman coding. Floating-point values may have many distinct bit patterns, and their empirical symbol distribution may be close to uniform at the byte or word level. Effective lossless compression of numerical data therefore usually begins with a transformation that exposes structure. The compressor is better viewed as a pipeline:

$$s\longmapsto u\longmapsto v\longmapsto z \tag{22.6.34}$$

where $u$ may be a predicted residual, $v$ may be a reordered or blocked representation, and $z$ is the entropy-coded bitstream. Exact recoverability requires that each stage be reversible:

$$s = T_1^{-1}(T_2^{-1}(\operatorname{Decode}(z))) \tag{22.6.35}$$

A common first step is predictive coding. Suppose a numerical sequence is:

$$x_0,x_1,\ldots,x_{N-1} \tag{22.6.36}$$

A predictor forms,

$$\widehat x_i=P(x_0,\ldots,x_{i-1}) \tag{22.6.37}$$

and stores the residual,

$$e_i=x_i-\widehat x_i \tag{22.6.38}$$

If the data are smooth or correlated, the residuals $e_i$ may have a more concentrated distribution than the original values. Huffman coding then acts on residual symbols rather than raw data. For integer data, equation (22.6.38) is exactly reversible if integer arithmetic is used consistently. For floating-point data, one must define the transform carefully at the bit level or use reversible integer representations of the floating-point words.

For multidimensional arrays, predictors may use neighboring values. For a two-dimensional grid $x_{ij}$, a simple predictor is:

$$\widehat x_{ij}=x_{i-1,j}+x_{i,j-1}-x_{i-1,j-1} \tag{22.6.39}$$

giving residual,

$$e_{ij}=x_{ij}-\widehat x_{ij} \tag{22.6.40}$$

This predictor is exact for locally bilinear patterns and often produces small residuals for smooth fields. The compression gain depends on the empirical distribution of the residuals. If residual frequencies are $f_a$, then the empirical entropy is:

$$\widehat H_e = -\sum_a \frac{f_a}{N}\log_2\frac{f_a}{N} \tag{22.6.41}$$

Huffman coding is effective when (\\widehat H_e) is significantly smaller than the raw bits per sample.

Another reversible strategy is bit-plane coding. For integer words with $b$ bits,

$$x_i=\sum_{j=0}^{b-1} x_{ij}2^j,\qquad x_{ij}\in\{0,1\} \tag{22.6.42}$$

one may group bits by significance level. The $j$-th bit plane is:

$$B_j=(x_{0j},x_{1j},\ldots,x_{N-1,j}) \tag{22.6.43}$$

If high-order planes contain long runs or low entropy, they can be compressed effectively. This is useful for integer arrays, masks, and sometimes transformed floating-point fields.

Run-length coding can also precede Huffman coding. If a sequence contains repeated symbols,

$$aaaaabbbcc\cdots,\tag{22.6.44}$$

it may be represented as pairs:

$$(a,5),(b,3),(c,2),\ldots  \tag{22.6.45}$$

The run lengths and symbols are then entropy-coded. This is useful in sparse masks, zero-heavy arrays, discretized fields with repeated labels, and structured diagnostic traces.

The compression ratio is:

$$\mathcal R = \frac{B_{\mathrm{raw}}}{B_{\mathrm{compressed}}} \tag{22.6.46}$$

where larger values indicate stronger compression. The savings fraction is:

$$S = 1-\frac{B_{\mathrm{compressed}}}{B_{\mathrm{raw}}} \tag{22.6.47}$$

For a lossless compressor, these quantities must be reported alongside verification that decompression is exact. A useful integrity check is:

$$C(s)=C(\operatorname{Decompress}(\operatorname{Compress}(s))) \tag{22.6.48}$$

where $C$ is a checksum or CRC as discussed in Section 22.5. This equation does not prove that the original data are semantically correct, but it verifies that the compression-decompression cycle preserved the byte representation.

Modern lossless compression for scientific data often follows exactly this pipeline structure. Huffman coding appears not as a complete solution by itself, but as a terminal entropy stage after prediction, transformation, blocking, or format-aware preprocessing. Recent work on lossless compression of scientific floating-point data on CPUs and GPUs, and on Huffman-based adaptive predictive compression in domain-specific settings, supports this contemporary interpretation (Azami et al., 2025; Lyu et al., 2025).

### Rust Implementation

Following the discussion in Section 22.6.3 on compression pipelines for numerical data, Program 22.6.2 provides a practical implementation of reversible predictive coding combined with Huffman entropy coding. In modern scientific data compression, Huffman coding is rarely applied directly to raw numerical arrays because the original values often exhibit high symbol diversity and weak frequency concentration. Instead, compression is typically organized as a sequence of reversible transformations that expose structure before entropy coding is performed. This program implements such a pipeline by first transforming a numerical sequence into predictive residuals, then applying Huffman coding to the residual stream, and finally reconstructing the original data through exact decoding and inverse prediction. The implementation demonstrates the pipeline formulation of Equations (22.6.34) and (22.6.35), illustrates the predictive residual construction of Equations (22.6.36)–(22.6.40), and evaluates compression effectiveness through entropy measurements, compression ratios, savings fractions, and checksum-based integrity verification. The resulting framework reflects the architecture used in many contemporary lossless scientific-data compressors, where prediction and entropy coding operate together to exploit data correlation while preserving exact recoverability.

At the core of the implementation is the predictive-transform stage represented by the functions `predictive_residuals` and `reconstruct_from_residuals`. The function `predictive_residuals` computes the residual sequence associated with Equations (22.6.37) and (22.6.38). For each sample, the previous value serves as a predictor, and the residual is obtained as the difference between the actual value and the predicted value. Because neighboring samples in smooth numerical data tend to be strongly correlated, the residual sequence often exhibits a much narrower distribution than the original data. The companion function `reconstruct_from_residuals` performs the inverse transformation by adding each residual to the corresponding prediction. Together, these functions implement the reversible transform stage required by Equation (22.6.35).

The function `generate_smooth_integer_data` produces a representative numerical dataset that mimics a slowly varying scientific signal. The generated values exhibit local smoothness and correlation, making them suitable for demonstrating the effectiveness of predictive coding. Such correlated data frequently arise in simulation outputs, measurement records, and discretized physical fields.

The function `frequency_table` constructs the empirical residual-frequency distribution required for entropy coding. Given the residual sequence, it counts the occurrences of each residual value and produces the frequency model used throughout the Huffman-compression stage. This frequency distribution forms the empirical basis for the entropy calculation of Equation (22.6.41).

The Huffman-compression stage is implemented through the structures `HuffmanTree` and `QueueItem` together with the functions `build_huffman_tree`, `build_codes`, `encode`, and `decode`. The function `build_huffman_tree` constructs an optimal prefix-code tree from the residual frequencies using the classical Huffman algorithm. The function `build_codes` traverses the resulting tree and assigns binary codewords to each residual symbol. The function `encode` converts the residual sequence into a compressed bitstream, while `decode` performs the inverse mapping from bitstream to residual sequence. Together, these functions implement the entropy-coding stage represented by the final mapping in Equation (22.6.34).

The function `empirical_entropy` evaluates the empirical residual entropy defined in Equation (22.6.41). This quantity measures the theoretical lower bound on the average number of bits required per residual symbol. The companion function `average_code_length` computes the actual average code length produced by the Huffman code. Comparing these two quantities provides a direct measure of coding efficiency and reveals how closely the constructed code approaches the information-theoretic limit.

The function `checksum_i32` implements a lightweight integrity mechanism that supports the verification principle described by Equation (22.6.48). By computing checksums before compression and after decompression, the program can verify that the complete compression-decompression cycle preserves the numerical data exactly. This complements the structural verification already provided by the successful reconstruction of the original sequence.

The utility functions `print_sample` and `print_code_table` provide diagnostic visibility into the compression process. The first displays representative portions of the original and residual data, making it possible to observe how prediction reduces variability. The second presents the Huffman code assignments and illustrates how more frequent residuals receive shorter codewords than infrequent residuals.

The `main` function orchestrates the entire compression pipeline. It begins by generating a smooth numerical dataset and transforming it into predictive residuals. A Huffman model is then constructed from the residual-frequency distribution, and the residual sequence is encoded into a compressed bitstream. The compressed stream is subsequently decoded, and the original data are reconstructed through inverse prediction. The program then computes entropy, average code length, compression ratio, savings fraction, and checksum equality. By verifying exact recovery and measuring compression effectiveness, the main function provides a complete demonstration of the reversible compression pipeline described throughout Section 22.6.3.

```rust
// Program 22.6.2: Lossless Compression Pipeline with Predictive Residual Coding
//
// Problem statement:
// Demonstrate a reversible numerical-data compression pipeline using integer
// predictive residual coding followed by Huffman entropy coding. The program
// verifies exact round-trip recovery and reports entropy, payload size,
// compression ratio, savings fraction, and checksum equality.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Clone)]
enum HuffmanTree {
    Leaf {
        symbol: i32,
        frequency: usize,
    },
    Internal {
        frequency: usize,
        left: Box<HuffmanTree>,
        right: Box<HuffmanTree>,
    },
}

impl HuffmanTree {
    fn frequency(&self) -> usize {
        match self {
            HuffmanTree::Leaf { frequency, .. } => *frequency,
            HuffmanTree::Internal { frequency, .. } => *frequency,
        }
    }

    fn min_symbol(&self) -> i32 {
        match self {
            HuffmanTree::Leaf { symbol, .. } => *symbol,
            HuffmanTree::Internal { left, right, .. } => left.min_symbol().min(right.min_symbol()),
        }
    }
}

#[derive(Clone)]
struct QueueItem {
    tree: HuffmanTree,
}

impl Eq for QueueItem {}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.tree.frequency() == other.tree.frequency()
            && self.tree.min_symbol() == other.tree.min_symbol()
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .tree
            .frequency()
            .cmp(&self.tree.frequency())
            .then_with(|| other.tree.min_symbol().cmp(&self.tree.min_symbol()))
    }
}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn generate_smooth_integer_data(n: usize) -> Vec<i32> {
    let mut data = Vec::with_capacity(n);

    for i in 0..n {
        let x = i as f64;
        let value = 1000.0 + 4.0 * x + 12.0 * (0.12 * x).sin();
        data.push(value.round() as i32);
    }

    data
}

fn predictive_residuals(data: &[i32]) -> Vec<i32> {
    let mut residuals = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        let prediction = if i == 0 { 0 } else { data[i - 1] };
        residuals.push(data[i] - prediction);
    }

    residuals
}

fn reconstruct_from_residuals(residuals: &[i32]) -> Vec<i32> {
    let mut data = Vec::with_capacity(residuals.len());

    for i in 0..residuals.len() {
        let prediction = if i == 0 { 0 } else { data[i - 1] };
        data.push(prediction + residuals[i]);
    }

    data
}

fn frequency_table(symbols: &[i32]) -> HashMap<i32, usize> {
    let mut frequencies = HashMap::new();

    for &s in symbols {
        *frequencies.entry(s).or_insert(0) += 1;
    }

    frequencies
}

fn build_huffman_tree(frequencies: &HashMap<i32, usize>) -> Option<HuffmanTree> {
    let mut heap = BinaryHeap::new();

    for (&symbol, &frequency) in frequencies {
        heap.push(QueueItem {
            tree: HuffmanTree::Leaf { symbol, frequency },
        });
    }

    while heap.len() > 1 {
        let first = heap.pop().unwrap().tree;
        let second = heap.pop().unwrap().tree;

        heap.push(QueueItem {
            tree: HuffmanTree::Internal {
                frequency: first.frequency() + second.frequency(),
                left: Box::new(first),
                right: Box::new(second),
            },
        });
    }

    heap.pop().map(|item| item.tree)
}

fn build_codes(tree: &HuffmanTree, prefix: String, codes: &mut HashMap<i32, String>) {
    match tree {
        HuffmanTree::Leaf { symbol, .. } => {
            let code = if prefix.is_empty() {
                "0".to_string()
            } else {
                prefix
            };
            codes.insert(*symbol, code);
        }
        HuffmanTree::Internal { left, right, .. } => {
            build_codes(left, format!("{}0", prefix), codes);
            build_codes(right, format!("{}1", prefix), codes);
        }
    }
}

fn encode(symbols: &[i32], codes: &HashMap<i32, String>) -> String {
    let mut bits = String::new();

    for &s in symbols {
        bits.push_str(codes.get(&s).expect("missing Huffman code"));
    }

    bits
}

fn decode(bits: &str, tree: &HuffmanTree) -> Vec<i32> {
    let mut output = Vec::new();
    let mut node = tree;

    for bit in bits.chars() {
        if let HuffmanTree::Internal { left, right, .. } = node {
            node = if bit == '0' { left } else { right };
        }

        if let HuffmanTree::Leaf { symbol, .. } = node {
            output.push(*symbol);
            node = tree;
        }
    }

    output
}

fn empirical_entropy(frequencies: &HashMap<i32, usize>, total: usize) -> f64 {
    frequencies
        .values()
        .map(|&f| {
            let p = f as f64 / total as f64;
            -p * p.log2()
        })
        .sum()
}

fn average_code_length(
    frequencies: &HashMap<i32, usize>,
    codes: &HashMap<i32, String>,
    total: usize,
) -> f64 {
    frequencies
        .iter()
        .map(|(&symbol, &frequency)| {
            let p = frequency as f64 / total as f64;
            p * codes[&symbol].len() as f64
        })
        .sum()
}

fn checksum_i32(values: &[i32]) -> u64 {
    let mut checksum = 0_u64;

    for &v in values {
        checksum = checksum.wrapping_add(v as u32 as u64);
    }

    checksum
}

fn print_sample(label: &str, values: &[i32], count: usize) {
    print!("{:<26} = [", label);

    for (i, v) in values.iter().take(count).enumerate() {
        if i + 1 == count.min(values.len()) {
            print!("{}", v);
        } else {
            print!("{}, ", v);
        }
    }

    if values.len() > count {
        print!(", ...");
    }

    println!("]");
}

fn print_code_table(frequencies: &HashMap<i32, usize>, codes: &HashMap<i32, String>) {
    let mut symbols: Vec<i32> = frequencies.keys().copied().collect();
    symbols.sort();

    println!("\nResidual Huffman Code Table");
    println!("---------------------------");
    println!(
        "{:>10} {:>12} {:>14} {:>16}",
        "residual", "frequency", "code length", "codeword"
    );

    for symbol in symbols {
        println!(
            "{:>10} {:>12} {:>14} {:>16}",
            symbol,
            frequencies[&symbol],
            codes[&symbol].len(),
            codes[&symbol]
        );
    }
}

fn main() {
    println!("Program 22.6.2: Lossless Compression Pipeline with Predictive Residual Coding");
    println!("==============================================================================\n");

    let data = generate_smooth_integer_data(80);
    let residuals = predictive_residuals(&data);

    println!("Original Numerical Data and Predictive Residuals");
    println!("------------------------------------------------");
    println!("number of samples        = {}", data.len());
    print_sample("original data", &data, 16);
    print_sample("predictive residuals", &residuals, 16);

    let reconstructed_data = reconstruct_from_residuals(&residuals);
    println!(
        "prediction transform reversible? = {}",
        reconstructed_data == data
    );

    let frequencies = frequency_table(&residuals);
    let tree = build_huffman_tree(&frequencies).expect("nonempty residual stream required");

    let mut codes = HashMap::new();
    build_codes(&tree, String::new(), &mut codes);

    let encoded_bits = encode(&residuals, &codes);
    let decoded_residuals = decode(&encoded_bits, &tree);
    let decoded_data = reconstruct_from_residuals(&decoded_residuals);

    print_code_table(&frequencies, &codes);

    let raw_bits = 32 * data.len();
    let payload_bits = encoded_bits.len();

    let model_bits = 48 * codes.len();
    let compressed_bits = payload_bits + model_bits;

    let compression_ratio_payload = raw_bits as f64 / payload_bits as f64;
    let compression_ratio_total = raw_bits as f64 / compressed_bits as f64;
    let savings_fraction = 1.0 - compressed_bits as f64 / raw_bits as f64;

    let entropy = empirical_entropy(&frequencies, residuals.len());
    let average_length = average_code_length(&frequencies, &codes, residuals.len());

    let original_checksum = checksum_i32(&data);
    let decoded_checksum = checksum_i32(&decoded_data);

    println!("\nRound-Trip Verification");
    println!("-----------------------");
    println!("decoded residuals equal encoded residuals? = {}", decoded_residuals == residuals);
    println!("decoded data equal original data?          = {}", decoded_data == data);
    println!("original checksum                          = {}", original_checksum);
    println!("decoded checksum                           = {}", decoded_checksum);
    println!(
        "checksum equality valid?                   = {}",
        original_checksum == decoded_checksum
    );

    println!("\nInformation and Compression Diagnostics");
    println!("---------------------------------------");
    println!("residual alphabet size        = {}", frequencies.len());
    println!("empirical residual entropy    = {:.6} bits/symbol", entropy);
    println!("average Huffman length        = {:.6} bits/symbol", average_length);
    println!(
        "Huffman redundancy            = {:.6} bits/symbol",
        average_length - entropy
    );
    println!("raw size                      = {} bits", raw_bits);
    println!("Huffman payload size          = {} bits", payload_bits);
    println!("model overhead estimate       = {} bits", model_bits);
    println!("compressed size estimate      = {} bits", compressed_bits);
    println!(
        "payload compression ratio     = {:.6}",
        compression_ratio_payload
    );
    println!(
        "total compression ratio       = {:.6}",
        compression_ratio_total
    );
    println!("savings fraction              = {:.6}", savings_fraction);
    println!("compression beneficial?       = {}", compressed_bits < raw_bits);
}
```

Program 22.6.2 demonstrates how lossless compression of numerical data is most effective when viewed as a pipeline rather than as a single coding algorithm. The predictive transform reduces local redundancy by converting correlated numerical values into residuals, while the Huffman stage exploits the resulting frequency concentration to produce a compact binary representation. This division of responsibilities mirrors the architecture of many modern scientific-data compression systems.

The numerical results illustrate the central role of prediction in reducing entropy. Although the original numerical sequence contains many distinct values, the residual sequence is concentrated around a small set of symbols. Consequently, the residual entropy is significantly lower than the information content of the raw data representation, allowing Huffman coding to achieve substantial compression. The comparison between empirical entropy and average code length further demonstrates the near-optimal behavior of the Huffman stage.

The exact agreement between the reconstructed data and the original dataset confirms the reversibility of every stage in the compression pipeline. The checksum verification provides an additional integrity check, demonstrating that the compression-decompression cycle preserves the complete numerical representation. This property is essential for scientific computing applications, where even a single altered value may compromise reproducibility or invalidate downstream analyses.

The modular structure of the implementation also highlights the extensibility of compression pipelines. Alternative predictors, multidimensional prediction schemes, bit-plane transforms, run-length preprocessing, or more sophisticated entropy coders can be incorporated without changing the overall pipeline architecture. This flexibility explains why modern scientific-data compressors frequently combine multiple reversible transformations before entropy coding. The present program therefore serves as a foundation for understanding more advanced lossless-compression methods used in large-scale scientific computing environments.

## 22.6.4. Practical Diagnostics for Huffman Compression

A Huffman compressor should be evaluated not only by compressed size, but also by model overhead, decoding speed, reproducibility, and exact recovery. In numerical computing, compression is part of the computational record. A compressed result that cannot be decoded reproducibly, or whose decoding depends on undocumented symbol ordering, is not a reliable archival representation.

The first diagnostic is the empirical distribution of symbols. If the alphabet is $\mathcal A$ and the empirical probabilities are $\widehat p_i$, then the empirical entropy is:

$$\widehat H = -\sum_{i:\widehat p_i>0}\widehat p_i\log_2\widehat p_i \tag{22.6.49}$$

The achieved average length is,

$$\widehat L = \frac{B_{\mathrm{payload}}}{N} \tag{22.6.50}$$

The coding redundancy is:

$$\widehat R = \widehat L-\widehat H \tag{22.6.51}$$

For an efficient Huffman code, $\widehat R$ should normally be less than one bit per symbol, excluding model overhead. If $\widehat R$ is large, the symbol model may be inappropriate, the alphabet may be too fine, or the data may need a preprocessing transform.

The second diagnostic is model overhead. The total rate is:

$$\widehat L_{\mathrm{total}} = \frac{B_{\mathrm{payload}}+B_{\mathrm{model}}}{N} \tag{22.6.52}$$

A Huffman code may be theoretically efficient for the payload but ineffective overall if $B_{\mathrm{model}}$ is large. This occurs when the alphabet is large, the file is small, or symbol frequencies are nearly uniform. In such cases, block-level decisions may be useful: compress a block only if:

$$B_{\mathrm{compressed}}^{(b)} + B_{\mathrm{model}}^{(b)}<B_{\mathrm{raw}}^{(b)} \tag{22.6.53}$$

Otherwise, store the block uncompressed with a flag indicating its representation.

The third diagnostic is exact round-trip recovery. The strongest basic test is byte equality:

$$s = \operatorname{Decompress}(\operatorname{Compress}(s)) \tag{22.6.54}$$

For large data, one often verifies this equality through checksums:

$$C_{\mathrm{raw}}(s) = C_{\mathrm{raw}}(\widehat s),\qquad\widehat s=\operatorname{Decompress}(z) \tag{22.6.55}$$

When possible, byte equality should be used during testing, while checksums are used during storage, transmission, and later verification.

The fourth diagnostic is determinism. Given the same data and the same compression settings, the compressor should produce either the same bitstream or a bitstream that is explicitly allowed to differ while decoding to the same data. For archival scientific workflows, deterministic compressed output is preferable:

$$\operatorname{Compress}(s;\theta) = \operatorname{Compress}(s;\theta)\tag{22.6.56}$$

across repeated runs, where $\theta$ denotes fixed compression settings. Canonical Huffman codes help achieve this because they remove arbitrary choices in tree labeling and tie-breaking.

Tie-breaking is especially important. If two symbols have equal frequency, different implementations may construct different but equally optimal Huffman trees. The average length $L_{\mathrm{H}}$ may be unchanged, but the assigned codewords may differ. A deterministic implementation should impose a fixed ordering rule, such as ordering by symbol value when weights are equal. This ensures that the code-length table and canonical codes are reproducible.

The fifth diagnostic is throughput. If compression time is $T_c$, decompression time is $T_d$, raw size is $B_{\mathrm{raw}}$, and compressed size is $B_{\mathrm{compressed}}$, then one may report:

$$v_c=\frac{B_{\mathrm{raw}}}{T_c},\qquad v_d=\frac{B_{\mathrm{raw}}}{T_d} \tag{22.6.57}$$

as compression and decompression throughput. A method with a high compression ratio may be unsuitable for checkpointing if $T_c$ dominates the simulation time. Conversely, a slightly weaker compressor may be preferred if it is faster, deterministic, and easier to verify. Modern entropy-coding literature emphasizes precisely this trade-off among compression efficiency, adaptation cost, streaming behavior, hardware simplicity, and model structure (Auli-Llinas, 2023).

For numerical computing, the practical conclusion is that Huffman coding should be presented as a mathematically clean and computationally useful member of a larger lossless-compression framework. Its strengths are simplicity, optimality among prefix codes for a fixed symbol model, deterministic canonical representations, and fast decoding. Its limitations are the integer-length constraint, model overhead, and reduced efficiency when symbols do not have a strongly nonuniform distribution. Section 22.7 continues from this point to arithmetic coding and range coding, which replace symbol-by-symbol prefix lengths with interval subdivision and can approach entropy more closely, especially when adaptive or context-dependent probability models are used.

### Rust Implementation

Following the discussion in Section 22.6.4 on practical diagnostics for Huffman compression, Program 22.6.3 provides a comprehensive framework for evaluating a Huffman compressor beyond compression ratio alone. In scientific computing, compressed data become part of the computational record and must therefore satisfy requirements extending beyond size reduction. A useful compressor should not only achieve a compact representation, but also provide deterministic behavior, exact recoverability, reproducible code construction, manageable model overhead, and acceptable throughput. This program implements a deterministic canonical Huffman compressor and evaluates it using the diagnostic criteria developed throughout Section 22.6.4. In addition to constructing and applying canonical Huffman codes, the implementation measures empirical entropy, coding redundancy, total compression rate, model overhead, round-trip correctness, checksum consistency, determinism across repeated runs, and compression and decompression throughput. The resulting framework demonstrates how theoretical coding efficiency and practical implementation quality can be assessed simultaneously within a scientifically reproducible compression workflow.

At the core of the implementation is the `HuffmanTree` data structure, which represents the binary coding tree from which Huffman code lengths are derived. Each leaf node stores a symbol and its empirical frequency, while each internal node stores the combined frequency of its descendants. The auxiliary quantity `min_symbol` is maintained to enforce deterministic tie-breaking whenever two nodes possess identical frequencies. This directly addresses the reproducibility concerns discussed in Section 22.6.4, where different tie-breaking choices may lead to different but equally optimal Huffman trees.

The `QueueItem` structure together with its ordering operators implements the priority queue required by the Huffman algorithm. Nodes are ordered primarily by frequency and secondarily by symbol value. This deterministic ordering ensures that repeated executions on identical data produce identical trees, code lengths, and encoded bitstreams, thereby satisfying the determinism criterion associated with Equation (22.6.56).

The function `generate_diagnostic_data` constructs a representative test dataset with a deliberately nonuniform symbol distribution. Such distributions are precisely the situations in which Huffman coding is expected to perform well because frequently occurring symbols can be assigned shorter codewords than infrequent symbols.

The function `frequency_table` computes the empirical symbol frequencies from the source data. These frequencies define the empirical probabilities $\widehat p_i$ used in Equation (22.6.49) and serve as the foundation for entropy estimation, Huffman-tree construction, and coding diagnostics. The function `build_huffman_tree` implements the classical Huffman algorithm. Starting from a collection of leaf nodes weighted by empirical frequencies, it repeatedly combines the two least frequent nodes into a new parent node. This greedy construction produces code lengths that minimize average code length among all prefix codes for the given symbol model.

The function `build_code_lengths` traverses the Huffman tree and determines the depth of each symbol. These depths become the Huffman code lengths used later to construct canonical codes. Separating code-length computation from codeword assignment reflects the distinction between optimal code lengths and canonical representations emphasized in practical compression systems.

The function `canonical_codes` constructs canonical Huffman codewords from the previously computed code lengths. Symbols are first sorted by code length and then by symbol value. The resulting codewords satisfy the same code-length distribution as the original Huffman tree but eliminate implementation-dependent variations in code assignment. Canonical codes are particularly valuable in archival scientific workflows because they guarantee reproducible encoded representations across platforms and executions.

The functions `encode`, `build_decode_table`, and `decode` implement the lossless compression-decompression cycle. The encoder converts symbols into canonical codewords and concatenates them into a bitstream, while the decoder reconstructs the original data using the canonical code table. Together, these functions provide the round-trip verification framework associated with Equation (22.6.54).

The function `empirical_entropy` computes the empirical entropy defined in Equation (22.6.49). This quantity represents the theoretical lower bound on the average number of bits per symbol for the observed source distribution. The function `average_length_from_payload` computes the achieved average code length of Equation (22.6.50), while the difference between these quantities gives the coding redundancy of Equation (22.6.51). These diagnostics quantify how closely the implementation approaches the information-theoretic limit.

The function `checksum_bytes` implements a lightweight integrity mechanism used to verify the correctness of the compression-decompression cycle. By comparing checksums of the original and reconstructed data, the implementation provides an efficient realization of the verification principle expressed in Equation (22.6.55).

The function `model_bits` estimates the storage cost associated with the coding model itself. This diagnostic corresponds directly to Equation (22.6.52), which emphasizes that payload efficiency alone is insufficient for evaluating compression performance. A highly efficient payload may still yield poor overall compression if model overhead is excessive. The function `verify_prefix_property` confirms that the generated codes satisfy the prefix condition. This verification ensures instantaneous decodability and provides an additional consistency check on the coding model. The function `print_canonical_code_table` produces a structured summary of frequencies, code lengths, and canonical codewords. Such diagnostics are valuable for understanding how probability structure is translated into coding structure and for verifying reproducibility across implementations.

The `compress_once` function encapsulates the complete compression workflow, including frequency estimation, tree construction, code-length generation, canonical code assignment, and bitstream encoding. By executing this function repeatedly on identical data, the program can test determinism directly and verify that identical inputs produce identical outputs.

The `main` function serves as a comprehensive diagnostic experiment. It constructs a test dataset, performs compression and decompression, measures entropy and redundancy, evaluates model overhead and total compression rate, verifies exact recovery and checksum equality, repeats compression to test determinism, and measures throughput using wall-clock timing. Collectively, these diagnostics provide a practical realization of the evaluation framework developed throughout Section 22.6.4 and demonstrate how compression performance should be assessed in scientific computing environments.

```rust
// Program 22.6.3: Practical Huffman Compression Diagnostics
//
// Problem statement:
// Evaluate a deterministic Huffman compressor using practical diagnostics:
// empirical entropy, achieved average length, redundancy, model overhead,
// total compression rate, exact round-trip recovery, checksum equality,
// deterministic bitstream reproduction, and compression/decompression throughput.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::time::Instant;

#[derive(Debug, Clone)]
enum HuffmanTree {
    Leaf {
        symbol: u8,
        frequency: usize,
    },
    Internal {
        frequency: usize,
        min_symbol: u8,
        left: Box<HuffmanTree>,
        right: Box<HuffmanTree>,
    },
}

impl HuffmanTree {
    fn frequency(&self) -> usize {
        match self {
            HuffmanTree::Leaf { frequency, .. } => *frequency,
            HuffmanTree::Internal { frequency, .. } => *frequency,
        }
    }

    fn min_symbol(&self) -> u8 {
        match self {
            HuffmanTree::Leaf { symbol, .. } => *symbol,
            HuffmanTree::Internal { min_symbol, .. } => *min_symbol,
        }
    }
}

#[derive(Clone)]
struct QueueItem {
    tree: HuffmanTree,
}

impl Eq for QueueItem {}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.tree.frequency() == other.tree.frequency()
            && self.tree.min_symbol() == other.tree.min_symbol()
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .tree
            .frequency()
            .cmp(&self.tree.frequency())
            .then_with(|| other.tree.min_symbol().cmp(&self.tree.min_symbol()))
    }
}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn generate_diagnostic_data(repetitions: usize) -> Vec<u8> {
    let pattern = b"AAAAABBBBBBBBBCCCCCCCDDDDEEF";

    let mut data = Vec::new();

    for _ in 0..repetitions {
        data.extend_from_slice(pattern);
    }

    data
}

fn frequency_table(data: &[u8]) -> HashMap<u8, usize> {
    let mut frequencies = HashMap::new();

    for &symbol in data {
        *frequencies.entry(symbol).or_insert(0) += 1;
    }

    frequencies
}

fn build_huffman_tree(frequencies: &HashMap<u8, usize>) -> HuffmanTree {
    let mut heap = BinaryHeap::new();

    let mut symbols: Vec<u8> = frequencies.keys().copied().collect();
    symbols.sort();

    for symbol in symbols {
        heap.push(QueueItem {
            tree: HuffmanTree::Leaf {
                symbol,
                frequency: frequencies[&symbol],
            },
        });
    }

    while heap.len() > 1 {
        let left = heap.pop().unwrap().tree;
        let right = heap.pop().unwrap().tree;

        let frequency = left.frequency() + right.frequency();
        let min_symbol = left.min_symbol().min(right.min_symbol());

        heap.push(QueueItem {
            tree: HuffmanTree::Internal {
                frequency,
                min_symbol,
                left: Box::new(left),
                right: Box::new(right),
            },
        });
    }

    heap.pop().expect("nonempty input required").tree
}

fn build_code_lengths(tree: &HuffmanTree, depth: usize, lengths: &mut HashMap<u8, usize>) {
    match tree {
        HuffmanTree::Leaf { symbol, .. } => {
            lengths.insert(*symbol, depth.max(1));
        }
        HuffmanTree::Internal { left, right, .. } => {
            build_code_lengths(left, depth + 1, lengths);
            build_code_lengths(right, depth + 1, lengths);
        }
    }
}

fn canonical_codes(lengths: &HashMap<u8, usize>) -> HashMap<u8, String> {
    let mut entries: Vec<(u8, usize)> = lengths.iter().map(|(&s, &l)| (s, l)).collect();

    entries.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    let mut codes = HashMap::new();
    let mut code_value = 0_u64;
    let mut previous_length = 0usize;

    for (symbol, length) in entries {
        code_value <<= length - previous_length;
        let codeword = format!("{:0width$b}", code_value, width = length);
        codes.insert(symbol, codeword);

        code_value += 1;
        previous_length = length;
    }

    codes
}

fn encode(data: &[u8], codes: &HashMap<u8, String>) -> String {
    let mut bits = String::new();

    for &symbol in data {
        bits.push_str(codes.get(&symbol).expect("missing codeword"));
    }

    bits
}

fn build_decode_table(codes: &HashMap<u8, String>) -> HashMap<String, u8> {
    let mut table = HashMap::new();

    for (&symbol, code) in codes {
        table.insert(code.clone(), symbol);
    }

    table
}

fn decode(bits: &str, decode_table: &HashMap<String, u8>) -> Vec<u8> {
    let mut output = Vec::new();
    let mut current = String::new();

    for bit in bits.chars() {
        current.push(bit);

        if let Some(&symbol) = decode_table.get(&current) {
            output.push(symbol);
            current.clear();
        }
    }

    assert!(
        current.is_empty(),
        "encoded stream ended with an incomplete codeword"
    );

    output
}

fn empirical_entropy(frequencies: &HashMap<u8, usize>, total: usize) -> f64 {
    frequencies
        .values()
        .map(|&frequency| {
            let p = frequency as f64 / total as f64;
            -p * p.log2()
        })
        .sum()
}

fn average_length_from_payload(payload_bits: usize, n_symbols: usize) -> f64 {
    payload_bits as f64 / n_symbols as f64
}

fn checksum_bytes(data: &[u8]) -> u64 {
    data.iter()
        .fold(0_u64, |acc, &b| acc.wrapping_mul(131).wrapping_add(b as u64))
}

fn model_bits(lengths: &HashMap<u8, usize>) -> usize {
    // Educational model overhead:
    // 8 bits for each symbol and 8 bits for its canonical code length.
    16 * lengths.len()
}

fn verify_prefix_property(codes: &HashMap<u8, String>) -> bool {
    let codewords: Vec<&String> = codes.values().collect();

    for i in 0..codewords.len() {
        for j in 0..codewords.len() {
            if i != j && codewords[j].starts_with(codewords[i]) {
                return false;
            }
        }
    }

    true
}

fn print_canonical_code_table(
    frequencies: &HashMap<u8, usize>,
    lengths: &HashMap<u8, usize>,
    codes: &HashMap<u8, String>,
) {
    let mut symbols: Vec<u8> = frequencies.keys().copied().collect();
    symbols.sort();

    println!("Canonical Huffman Code Table");
    println!("----------------------------");
    println!(
        "{:>8} {:>12} {:>14} {:>14}",
        "symbol", "frequency", "length", "canonical code"
    );

    for symbol in symbols {
        println!(
            "{:>8} {:>12} {:>14} {:>14}",
            symbol as char,
            frequencies[&symbol],
            lengths[&symbol],
            codes[&symbol]
        );
    }
}

fn compress_once(data: &[u8]) -> (String, HashMap<u8, usize>, HashMap<u8, String>) {
    let frequencies = frequency_table(data);
    let tree = build_huffman_tree(&frequencies);

    let mut lengths = HashMap::new();
    build_code_lengths(&tree, 0, &mut lengths);

    let codes = canonical_codes(&lengths);
    let encoded = encode(data, &codes);

    (encoded, lengths, codes)
}

fn main() {
    println!("Program 22.6.3: Practical Huffman Compression Diagnostics");
    println!("=========================================================\n");

    let data = generate_diagnostic_data(1000);
    let raw_bits = 8 * data.len();

    let compress_start = Instant::now();
    let (encoded, lengths, codes) = compress_once(&data);
    let compression_time = compress_start.elapsed().as_secs_f64();

    let decode_table = build_decode_table(&codes);

    let decompress_start = Instant::now();
    let decoded = decode(&encoded, &decode_table);
    let decompression_time = decompress_start.elapsed().as_secs_f64();

    let (encoded_repeat, lengths_repeat, codes_repeat) = compress_once(&data);

    let frequencies = frequency_table(&data);

    let entropy = empirical_entropy(&frequencies, data.len());
    let achieved_average_length = average_length_from_payload(encoded.len(), data.len());
    let redundancy = achieved_average_length - entropy;

    let payload_bits = encoded.len();
    let model_bits = model_bits(&lengths);
    let total_bits = payload_bits + model_bits;
    let total_rate = total_bits as f64 / data.len() as f64;

    let raw_checksum = checksum_bytes(&data);
    let decoded_checksum = checksum_bytes(&decoded);

    let compression_throughput = raw_bits as f64 / compression_time.max(f64::MIN_POSITIVE);
    let decompression_throughput = raw_bits as f64 / decompression_time.max(f64::MIN_POSITIVE);

    print_canonical_code_table(&frequencies, &lengths, &codes);

    println!("\nDistribution and Coding Diagnostics");
    println!("-----------------------------------");
    println!("number of symbols              = {}", data.len());
    println!("alphabet size                  = {}", frequencies.len());
    println!("empirical entropy H_hat        = {:.6} bits/symbol", entropy);
    println!(
        "achieved average length L_hat  = {:.6} bits/symbol",
        achieved_average_length
    );
    println!("coding redundancy R_hat        = {:.6} bits/symbol", redundancy);
    println!("prefix-code property valid?    = {}", verify_prefix_property(&codes));

    println!("\nModel and Compression-Rate Diagnostics");
    println!("--------------------------------------");
    println!("raw size                       = {} bits", raw_bits);
    println!("payload size                   = {} bits", payload_bits);
    println!("model overhead estimate        = {} bits", model_bits);
    println!("total compressed estimate      = {} bits", total_bits);
    println!("total rate L_total             = {:.6} bits/symbol", total_rate);
    println!(
        "compression ratio              = {:.6}",
        raw_bits as f64 / total_bits as f64
    );
    println!(
        "compression beneficial?        = {}",
        total_bits < raw_bits
    );

    println!("\nRound-Trip and Integrity Diagnostics");
    println!("------------------------------------");
    println!("byte equality valid?           = {}", decoded == data);
    println!("raw checksum                   = {}", raw_checksum);
    println!("decoded checksum               = {}", decoded_checksum);
    println!("checksum equality valid?       = {}", raw_checksum == decoded_checksum);

    println!("\nDeterminism Diagnostics");
    println!("-----------------------");
    println!("repeat bitstream identical?    = {}", encoded == encoded_repeat);
    println!("repeat code lengths identical? = {}", lengths == lengths_repeat);
    println!("repeat codes identical?        = {}", codes == codes_repeat);

    println!("\nThroughput Diagnostics");
    println!("----------------------");
    println!("compression time               = {:.6e} s", compression_time);
    println!("decompression time             = {:.6e} s", decompression_time);
    println!(
        "compression throughput         = {:.6e} bits/s",
        compression_throughput
    );
    println!(
        "decompression throughput       = {:.6e} bits/s",
        decompression_throughput
    );
}
```

Program 22.6.3 demonstrates that evaluating a Huffman compressor requires substantially more than measuring compressed size. While compression ratio remains an important metric, practical scientific workflows also require exact recoverability, deterministic behavior, manageable model overhead, and sufficient computational throughput. The diagnostic framework implemented in this program reflects these broader requirements and provides a more complete assessment of compression quality.

The entropy and redundancy measurements illustrate the relationship between information theory and practical coding performance. The empirical entropy establishes the theoretical lower bound on coding efficiency, while the achieved average length quantifies the performance of the actual implementation. The small redundancy observed in the experiment confirms the near-optimal behavior of Huffman coding for a fixed symbol model.

The determinism diagnostics highlight an issue that is often overlooked in introductory discussions of compression. Multiple Huffman trees may be equally optimal when symbol frequencies contain ties. Without deterministic tie-breaking and canonical code assignment, different implementations can produce different compressed bitstreams even when they compress identical data. Canonical Huffman coding eliminates this ambiguity and therefore improves reproducibility, archival reliability, and cross-platform consistency.

The throughput measurements emphasize the practical trade-off between compression effectiveness and computational cost. A highly efficient compressor may be unsuitable for large-scale checkpointing if compression time becomes excessive, while a slightly weaker compressor may be preferable if it offers faster execution and simpler verification. These considerations explain why modern compression systems are evaluated using multiple diagnostics rather than a single compression-ratio metric.

Overall, the program demonstrates how entropy estimation, redundancy analysis, model-overhead accounting, round-trip verification, checksum validation, deterministic canonical coding, and throughput measurement can be integrated into a unified evaluation framework. Such diagnostics transform compression from a simple encoding procedure into a scientifically verifiable component of the computational infrastructure.

# 22.7. Arithmetic Coding and Range Coding

Huffman coding assigns each symbol an integer-length prefix codeword. This makes decoding simple and fast, but it also imposes a structural limitation: a symbol whose ideal information content is $-\log_2 p_i$ bits must be assigned an integer number of bits. Arithmetic coding avoids this symbol-by-symbol integer-length restriction. Instead of assigning a separate bit string to each symbol, it represents an entire message as a subinterval of the unit interval. The final interval can then be identified by a binary fraction whose length is close to the information content of the whole message.

This shift from codewords to intervals is mathematically important. It allows the code length for a long sequence to approach:

$$-\log_2 P(s) \tag{22.7.1}$$

where $P(s)$ is the probability assigned to the full source sequence. In practice, arithmetic coding and range coding are used with adaptive probability models, context models, integer arithmetic, renormalization, and streaming implementations. Modern entropy-coding literature increasingly treats Huffman coding, arithmetic coding, range coding, hybrid coders, and table-based coders as members of a larger family of entropy-coding methods whose choice depends on compression efficiency, adaptation cost, streaming behavior, hardware simplicity, and model structure (Auli-Llinas, 2023; Kunze et al., 2024; Liang et al., 2025; Wiseman, 2025).

## 22.7.1. Interval Subdivision and Arithmetic Codes

Let the alphabet be:

$$\mathcal A=\{a_1,a_2,\ldots,a_m\} \tag{22.7.2}$$

with probabilities,

$$p_i=\Pr(a_i),\qquad p_i\ge 0,\qquad\sum_{i=1}^{m}p_i=1 \tag{22.7.3}$$

Define the cumulative distribution values:

$$F_0=0,\qquad F_i=\sum_{j=1}^{i}p_j,\qquad i=1,\ldots,m \tag{22.7.4}$$

Thus, symbol $a_i$ corresponds to the subinterval:

$$I_i=[F_{i-1},F_i) \tag{22.7.5}$$

Arithmetic coding begins with the interval:

$$[L_0,U_0)=[0,1) \tag{22.7.6}$$

After encoding symbols $s_1,s_2,\ldots,s_k$, the current interval is:

$$[L_k,U_k),\qquad W_k=U_k-L_k \tag{22.7.7}$$

If the next symbol is $s_{k+1}=a_i$, the interval is refined by:

$$L_{k+1} = L_k+W_kF_{i-1} \tag{22.7.8}$$

$$U_{k+1} = L_k+W_kF_i \tag{22.7.9}$$

The new width is therefore:

$$W_{k+1} = U_{k+1}-L_{k+1} = W_k p_i \tag{22.7.10}$$

For a message,

$$s=(s_1,s_2,\ldots,s_N) \tag{22.7.11}$$

with symbol probabilities $p(s_k)$, repeated application gives:

$$W_N = \prod_{k=1}^{N}p(s_k) \tag{22.7.12}$$

If the source is memoryless, this product is exactly the probability of the message:

$$P(s)=\prod_{k=1}^{N}p(s_k) \tag{22.7.13}$$

Thus,

$$W_N=P(s) \tag{22.7.14}$$

Any number,

$$z\in [L_N,U_N)\tag{22.7.15}$$

identifies the message uniquely, provided the decoder uses the same probability model and knows when to stop. To store $z$ as a finite binary fraction, one needs enough bits to select a binary interval contained in $[L_N,U_N)$. If $B$ bits are used, the binary grid spacing is $2^{-B}$. A sufficient condition is:

$$2^{-B}\le W_N \tag{22.7.16}$$

Hence,

$$B\ge -\log_2 W_N = -\log_2 P(s) \tag{22.7.17}$$

More precisely, arithmetic coding can encode the message using approximately,

$$B = \left\lceil -\log_2 P(s)\right\rceil + c\tag{22.7.18}$$

bits, where $c$ is a small overhead depending on termination and implementation details.

For a memoryless source, equation (22.7.17) becomes:

$$-\log_2 P(s) = -\sum_{k=1}^{N}\log_2 p(s_k) \tag{22.7.19}$$

If symbol $a_i$ occurs $f_i$ times, then,

$$-\log_2 P(s) = -\sum_{i=1}^{m} f_i\log_2 p_i \tag{22.7.20}$$

Dividing by $N$, the ideal average length per symbol is:

$$\frac{1}{N}\left[-\log_2 P(s)\right] = -\sum_{i=1}^{m}\widehat p_i\log_2 p_i \tag{22.7.21}$$

where,

$$\widehat p_i=\frac{f_i}{N}\tag{22.7.22}$$

is the empirical frequency. If the model probabilities $p_i$ match the empirical distribution, then equation (22.7.21) reduces to the empirical entropy:

$$\widehat H = -\sum_{i=1}^{m}\widehat p_i\log_2\widehat p_i \tag{22.7.23}$$

This explains why arithmetic coding can approach the entropy limit more closely than Huffman coding for long sequences. Huffman coding assigns integer lengths to individual symbols, while arithmetic coding assigns a near-optimal fractional average length to the message as a whole.

The decoder reverses the interval process. Given a code value $z\in[0,1)$, the first symbol is the unique $a_i$ such that,

$$z\in [F_{i-1},F_i) \tag{22.7.24}$$

After decoding $a_i$, the decoder rescales:

$$z' = \frac{z-F_{i-1}}{p_i} \tag{22.7.25}$$

and repeats the process. For a non-initial interval $[L_k,U_k)$, the same operation is performed relative to that interval. Exact decoding requires that encoder and decoder use the same cumulative probabilities, the same symbol ordering, the same termination convention, and the same finite-precision arithmetic rules.

### Rust Implementation

Following the development of interval subdivision arithmetic coding in Section 22.7.1, Program 22.7.1 provides a practical implementation of static arithmetic coding using a fixed probability model. The program directly realizes the interval-update process described by Equations (22.7.8)–(22.7.10), repeatedly refining the current coding interval as symbols are processed. Unlike Huffman coding, which assigns individual codewords to symbols, arithmetic coding represents the entire message by a single subinterval of the unit interval whose width equals the probability of the message. The implementation demonstrates interval construction, code-value selection, exact decoding through interval inversion, and information-theoretic diagnostics such as message probability, ideal code length, and empirical entropy. In doing so, it illustrates how arithmetic coding can approach the entropy limit by assigning a near-optimal fractional average number of bits per symbol.

At the core of the implementation is the `Symbol` structure, which stores the information required by the arithmetic-coding model. For each symbol, the structure contains its probability together with the cumulative interval boundaries $F_{i-1}$ and $F_i$ introduced in Equations (22.7.4) and (22.7.5). These cumulative values define the subinterval associated with each symbol and form the basis of both encoding and decoding operations.

The `build_model` function constructs the cumulative probability model from a user-specified list of symbol probabilities. It verifies that the probabilities form a valid distribution and then computes the cumulative boundaries required by the interval-subdivision process. These cumulative intervals correspond directly to the arithmetic-coding alphabet partition described by Equations (22.7.2)–(22.7.5).

The `encode` function implements the arithmetic-coding update equations. Beginning from the initial interval $[0,1)$ of Equation (22.7.6), it processes each symbol sequentially and updates the current interval according to Equations (22.7.8) and (22.7.9). After all symbols have been processed, the resulting interval $[L_N,U_N)$ uniquely identifies the message. The width of this interval corresponds to $W_N$, which according to Equations (22.7.12)–(22.7.14) equals the probability of the encoded message under the model.

The `decode` function performs the inverse operation. Given a representative code value (z) selected from the final interval, it repeatedly determines which symbol interval contains the current value and then rescales according to Equation (22.7.25). This procedure reconstructs the original message symbol by symbol and demonstrates the reversibility of arithmetic coding when the encoder and decoder share the same probability model.

Several diagnostic functions are included to connect the implementation with the theoretical analysis. The `message_probability` function computes the probability $P(s)$ of the message under the model, corresponding to Equations (22.7.13) and (22.7.14). The `empirical_frequencies` function computes the observed symbol frequencies $\widehat p_i$ defined by Equation (22.7.22), while `empirical_entropy` evaluates the empirical entropy $\widehat H$ of Equation (22.7.23). The `average_model_code_length` function computes the average information content per symbol according to Equation (22.7.21), allowing direct comparison between the coding model and the observed source statistics.

The `main` function demonstrates the complete arithmetic-coding workflow. It begins by constructing a static probability model for a four-symbol alphabet and then encodes a sample message. The final interval boundaries, interval width, message probability, and representative code value are reported. The program subsequently decodes the selected code value to verify exact round-trip recovery. Finally, information-theoretic diagnostics are computed, including the ideal code length $-\log_2 P(s)$ from Equation (22.7.17), the sufficient binary representation length implied by Equation (22.7.18), the average model code length, and the empirical entropy. Together, these diagnostics illustrate the relationship between message probability, interval width, and compression efficiency in arithmetic coding.

```rust
// Program 22.7.1. Static Arithmetic Coding by Interval Subdivision
//
// Problem statement:
// Demonstrate the real-interval form of arithmetic coding described in
// Section 22.7.1. Given a fixed alphabet, fixed symbol probabilities, and a
// finite message, refine the interval [0, 1) according to equations (22.7.8)
// and (22.7.9), choose a representative code value from the final interval,
// and decode the message by reversing the interval subdivision process.

#[derive(Clone, Debug)]
struct Symbol {
    ch: char,
    probability: f64,
    cumulative_low: f64,
    cumulative_high: f64,
}

fn build_model(symbols: &[(char, f64)]) -> Result<Vec<Symbol>, String> {
    let total: f64 = symbols.iter().map(|(_, p)| *p).sum();

    if (total - 1.0).abs() > 1.0e-12 {
        return Err(format!(
            "probabilities must sum to 1; received sum = {:.16}",
            total
        ));
    }

    let mut model = Vec::new();
    let mut cumulative = 0.0;

    for &(ch, probability) in symbols {
        if probability <= 0.0 {
            return Err(format!("symbol '{}' has nonpositive probability", ch));
        }

        let cumulative_low = cumulative;
        let cumulative_high = cumulative + probability;

        model.push(Symbol {
            ch,
            probability,
            cumulative_low,
            cumulative_high,
        });

        cumulative = cumulative_high;
    }

    Ok(model)
}

fn find_symbol(model: &[Symbol], ch: char) -> Option<&Symbol> {
    model.iter().find(|symbol| symbol.ch == ch)
}

fn encode(message: &[char], model: &[Symbol]) -> Result<(f64, f64), String> {
    let mut low = 0.0;
    let mut high = 1.0;

    for &ch in message {
        let symbol = find_symbol(model, ch)
            .ok_or_else(|| format!("message contains unknown symbol '{}'", ch))?;

        let width = high - low;
        let new_low = low + width * symbol.cumulative_low;
        let new_high = low + width * symbol.cumulative_high;

        low = new_low;
        high = new_high;
    }

    Ok((low, high))
}

fn decode(code_value: f64, message_len: usize, model: &[Symbol]) -> Result<Vec<char>, String> {
    if !(0.0..1.0).contains(&code_value) {
        return Err("code value must lie in [0, 1)".to_string());
    }

    let mut z = code_value;
    let mut decoded = Vec::with_capacity(message_len);

    for _ in 0..message_len {
        let symbol = model
            .iter()
            .find(|s| z >= s.cumulative_low && z < s.cumulative_high)
            .ok_or_else(|| format!("failed to locate symbol for z = {:.16}", z))?;

        decoded.push(symbol.ch);

        z = (z - symbol.cumulative_low) / symbol.probability;
    }

    Ok(decoded)
}

fn message_probability(message: &[char], model: &[Symbol]) -> Result<f64, String> {
    let mut probability = 1.0;

    for &ch in message {
        let symbol = find_symbol(model, ch)
            .ok_or_else(|| format!("message contains unknown symbol '{}'", ch))?;
        probability *= symbol.probability;
    }

    Ok(probability)
}

fn empirical_frequencies(message: &[char], model: &[Symbol]) -> Vec<(char, usize, f64)> {
    let n = message.len() as f64;

    model
        .iter()
        .map(|symbol| {
            let count = message.iter().filter(|&&ch| ch == symbol.ch).count();
            let frequency = count as f64 / n;
            (symbol.ch, count, frequency)
        })
        .collect()
}

fn empirical_entropy(frequencies: &[(char, usize, f64)]) -> f64 {
    frequencies
        .iter()
        .filter(|(_, _, p_hat)| *p_hat > 0.0)
        .map(|(_, _, p_hat)| -p_hat * p_hat.log2())
        .sum()
}

fn average_model_code_length(message: &[char], model: &[Symbol]) -> Result<f64, String> {
    let mut total = 0.0;

    for &ch in message {
        let symbol = find_symbol(model, ch)
            .ok_or_else(|| format!("message contains unknown symbol '{}'", ch))?;
        total += -symbol.probability.log2();
    }

    Ok(total / message.len() as f64)
}

fn main() -> Result<(), String> {
    let model = build_model(&[
        ('A', 0.40),
        ('B', 0.30),
        ('C', 0.20),
        ('D', 0.10),
    ])?;

    let message_string = "ABACABAD";
    let message: Vec<char> = message_string.chars().collect();

    let (low, high) = encode(&message, &model)?;
    let width = high - low;
    let code_value = 0.5 * (low + high);

    let decoded = decode(code_value, message.len(), &model)?;
    let decoded_string: String = decoded.iter().collect();

    let probability = message_probability(&message, &model)?;
    let ideal_bits = -probability.log2();
    let sufficient_bits = ideal_bits.ceil();

    let frequencies = empirical_frequencies(&message, &model);
    let entropy = empirical_entropy(&frequencies);
    let average_length = average_model_code_length(&message, &model)?;

    println!("Static Arithmetic Coding by Interval Subdivision");
    println!("================================================");
    println!();

    println!("Alphabet Model");
    println!("--------------");
    println!(
        "{:>8} {:>14} {:>18} {:>18}",
        "symbol", "probability", "F_low", "F_high"
    );

    for symbol in &model {
        println!(
            "{:>8} {:>14.8} {:>18.12} {:>18.12}",
            symbol.ch, symbol.probability, symbol.cumulative_low, symbol.cumulative_high
        );
    }

    println!();
    println!("Message");
    println!("-------");
    println!("input message             = {}", message_string);
    println!("number of symbols          = {}", message.len());

    println!();
    println!("Final Arithmetic-Coding Interval");
    println!("--------------------------------");
    println!("L_N                         = {:.16}", low);
    println!("U_N                         = {:.16}", high);
    println!("W_N = U_N - L_N              = {:.16}", width);
    println!("P(s) from product model      = {:.16}", probability);
    println!("absolute |W_N - P(s)|        = {:.3e}", (width - probability).abs());
    println!("chosen code value z          = {:.16}", code_value);

    println!();
    println!("Information-Length Diagnostics");
    println!("------------------------------");
    println!("-log2 P(s)                   = {:.8} bits", ideal_bits);
    println!("ceil(-log2 P(s))             = {:.0} bits", sufficient_bits);
    println!(
        "average model length         = {:.8} bits/symbol",
        average_length
    );
    println!(
        "empirical entropy H_hat      = {:.8} bits/symbol",
        entropy
    );

    println!();
    println!("Empirical Frequencies");
    println!("---------------------");
    println!("{:>8} {:>10} {:>16}", "symbol", "count", "p_hat");

    for (ch, count, p_hat) in frequencies {
        println!("{:>8} {:>10} {:>16.8}", ch, count, p_hat);
    }

    println!();
    println!("Decoding Check");
    println!("--------------");
    println!("decoded message            = {}", decoded_string);
    println!("round-trip exact            = {}", decoded_string == message_string);

    Ok(())
}
```

Program 22.7.1 demonstrates the fundamental principle underlying arithmetic coding: representing an entire message by a progressively refined interval whose width equals the probability assigned to that message. The implementation confirms the theoretical result that the final interval width $W_N$ converges to the message probability $P(s)$, thereby linking coding length directly to information content.

The example also illustrates why arithmetic coding can approach the entropy limit more closely than Huffman coding. Rather than assigning integer-length codewords to individual symbols, arithmetic coding effectively distributes fractional coding cost across the entire message. As a result, the achieved average length approaches the ideal information-theoretic bound as message length increases.

Although the implementation uses real-valued arithmetic for clarity, practical compressors generally employ finite-precision integer arithmetic and renormalization mechanisms. These considerations lead naturally to the range-coding formulations discussed in Section 22.7.3. The present program therefore serves as a conceptual foundation for understanding how practical entropy coders transform probability models into compact and nearly optimal binary representations.

## 22.7.2. Adaptive and Context-Based Arithmetic Coding

Arithmetic coding separates the coding mechanism from the probability model. The interval-update equations require only cumulative probabilities at each step. These probabilities may be fixed, adaptive, or context-dependent. This separation is one reason arithmetic coding remains important in modern compression systems.

In a static model, the probabilities $p_i$ are fixed before coding begins. They may be estimated from the entire message, from a training corpus, or from a known data distribution. Static coding is simple, but the model must be transmitted or reconstructed by the decoder. If the model requires $B_{\mathrm{model}}$ bits and the encoded payload uses $B_{\mathrm{payload}}$ bits, then:

$$B_{\mathrm{total}} = B_{\mathrm{model}}+B_{\mathrm{payload}} \tag{22.7.26}$$

Compression is useful only if,

$$B_{\mathrm{total}}<B_{\mathrm{raw}} \tag{22.7.27}$$

In an adaptive model, probabilities are updated as the message is processed. Let $n_i(k)$ be the number of times symbol $a_i$ has appeared before step $k$. A smoothed adaptive probability may be,

$$p_i^{(k)} = \frac{n_i(k)+\alpha}{k+m\alpha},\qquad \alpha>0 \tag{22.7.28}$$

The parameter $\alpha$ prevents zero probabilities. Without smoothing, a symbol not previously seen would have probability zero and could not be encoded. The interval update at step $k$ then uses cumulative probabilities:

$$F_i^{(k)} = \sum_{j=1}^{i}p_j^{(k)} \tag{22.7.29}$$

The message probability under the adaptive model is:

$$P_{\mathrm{ad}}(s) = \prod_{k=1}^{N} p^{(k)}(s_k) \tag{22.7.30}$$

and the ideal code length is:

$$-\log_2 P_{\mathrm{ad}}(s) = -\sum_{k=1}^{N}\log_2 p^{(k)}(s_k) \tag{22.7.31}$$

The advantage is that the decoder can reproduce the same model updates from the decoded prefix, so the full symbol-frequency table need not be transmitted. The disadvantage is that the early part of the message may be coded with inaccurate probabilities, and the model update must be exactly synchronized between encoder and decoder.

Context-based coding refines this idea. Instead of using a single probability distribution, the model conditions on a context $c_k$, such as previous symbols, neighboring grid values, bit-plane position, block type, or predictor state. The probability of the next symbol is:

$$p(s_k\mid c_k) \tag{22.7.32}$$

The message probability becomes,

$$P(s) = \prod_{k=1}^{N}p(s_k\mid c_k) \tag{22.7.33}$$

and the ideal length is,

$$-\log_2 P(s) = -\sum_{k=1}^{N}\log_2 p(s_k\mid c_k) \tag{22.7.34}$$

For numerical data, contexts may arise naturally from data structure. In a two-dimensional array, the context for $x_{ij}$ may include neighboring values:

$$c_{ij}=(x_{i-1,j},x_{i,j-1},x_{i-1,j-1}) \tag{22.7.35}$$

For bit-plane coding, the context may include the bit significance level and previously decoded higher-order bits:

$$c_{ij}=(j,x_{i,b-1},x_{i,b-2},\ldots,x_{i,j+1}) \tag{22.7.36}$$

For residual coding after prediction, the context may include local smoothness indicators or block statistics. These models can reduce entropy by assigning sharper probabilities to the actual symbols. However, they also increase complexity and make reproducibility more dependent on exact model specification.

The cross-entropy identity clarifies the cost of model mismatch. If the true empirical distribution is $q_i$ but the coder uses probabilities $p_i$, then the average ideal length is:

$$H(q,p) = -\sum_i q_i\log_2 p_i \tag{22.7.37}$$

This decomposes as,

$$H(q,p) = H(q)+D_{\mathrm{KL}}(q\|p) \tag{22.7.38}$$

where,

$$D_{\mathrm{KL}}(q\|p) = \sum_i q_i\log_2\frac{q_i}{p_i}\ge 0 \tag{22.7.39}$$

Thus, using the wrong model increases the expected code length by the relative entropy $D_{\mathrm{KL}}(q\|p)$. In compression diagnostics, this quantity explains why a poor probability model can make arithmetic coding perform no better, or even worse, than simpler methods.

Modern arithmetic-coding research extends this modeling perspective in several directions. Semantic arithmetic coding uses mappings among synonymous representations, bits-back methods exploit probabilistic structure in unordered data, and hybrid arithmetic-Huffman systems seek compromises between adaptivity, compression ratio, and implementation simplicity (Kunze et al., 2024; Liang et al., 2025; Wiseman, 2025). For numerical computing, the practical lesson is that the probability model is as important as the coder. The interval-coding mechanism can be near-optimal only when the model assigns high probability to the symbols that actually occur.

## 22.7.3. Range Coding and Finite-Precision Implementation

The ideal arithmetic coder described above operates on real intervals. Actual computers cannot maintain arbitrary real intervals without precision growth. Practical arithmetic coding therefore uses integer arithmetic, renormalization, and finite registers. Range coding is a closely related formulation that represents the current interval by an integer lower endpoint and an integer range.

Let the coding state be,

$$L_k\in\{0,1,\ldots,M-1\},\qquad R_k\in \{1,2,\ldots,M\} \tag{22.7.40}$$

where $M$ is the size of the integer coding interval, often a power of two. The current interval is:

$$[L_k,L_k+R_k) \tag{22.7.41}$$

Suppose the alphabet has integer cumulative frequencies,

$$0=C_0<C_1<\cdots<C_m=T \tag{22.7.42}$$

where $T$ is the total frequency. Symbol $a_i$ corresponds to the cumulative range:

$$[C_{i-1},C_i) \tag{22.7.43}$$

The range update is:

$$L_{k+1} = L_k+\left\lfloor\frac{R_k C_{i-1}}{T}\right\rfloor \tag{22.7.44}$$

$$R_{k+1} = \left\lfloor\frac{R_k C_i}{T}\right\rfloor - \left\lfloor\frac{R_k C_{i-1}}{T}\right\rfloor \tag{22.7.45}$$

This is the integer analogue of equations (22.7.8) and (22.7.9). The use of floors makes the implementation exact and deterministic, but it also imposes constraints. One must ensure that:

$$R_{k+1}\ge 1\tag{22.7.46}$$

for every symbol that may be encoded. Therefore, the total frequency $T$ must be small enough relative to the maintained range $R_k$, or the coder must renormalize before the interval becomes too narrow.

Renormalization emits leading bits or bytes once the current interval lies wholly inside a common prefix region. In a binary arithmetic coder, if the interval is contained in the lower half,

$$[L_k,U_k)\subseteq [0,1/2) \tag{22.7.47}$$

the encoder can emit a $0$ and rescale:

$$L_k\leftarrow 2L_k,\qquad U_k\leftarrow 2U_k \tag{22.7.48}$$

If the interval is contained in the upper half,

$$[L_k,U_k)\subseteq [1/2,1) \tag{22.7.49}$$

the encoder can emit a $1$ and rescale:

$$L_k\leftarrow 2L_k-1,\qquad U_k\leftarrow 2U_k-1 \tag{22.7.50}$$

There is also an underflow or near-middle case when the interval lies around $1/2$, for example inside $[1/4,3/4)$. In this case, the leading bit is not yet determined, but subsequent bits can be delayed and adjusted once the interval moves decisively to the lower or upper half. Practical coders implement this with underflow counters or equivalent range-coder normalization rules.

Range coding usually emits bytes rather than individual bits. Its state is often maintained as an integer low value and range value:

$$\text{low},\quad \text{range}.\tag{22.7.51}$$

When the range becomes too small, $\text{range}<R_{\min}$, the encoder outputs the leading byte or bytes of $\text{low}$ and rescales both state variables. The decoder mirrors the same process by maintaining a code value inside the current range. Exact synchronization requires that encoder and decoder use identical integer arithmetic, identical frequency tables, and identical renormalization thresholds.

This finite-precision structure is crucial in numerical software. The real arithmetic equations explain the ideal compression behavior, but the actual compressed bitstream is determined by integer arithmetic rules. Any ambiguity in rounding, cumulative-frequency construction, symbol ordering, or renormalization can break reproducibility. For scientific archives, the compression specification must therefore define all integer operations exactly.

The finite-frequency model also introduces quantization of probabilities. If true probabilities are $p_i$, but integer frequencies use:

$$\widetilde p_i=\frac{C_i-C_{i-1}}{T} \tag{22.7.52}$$

then the coding length is governed by $\widetilde p_i$, not $p_i$. The expected penalty is approximately,

$$D_{\mathrm{KL}}(p\|\widetilde p) = \sum_i p_i\log_2\frac{p_i}{\widetilde p_i} \tag{22.7.53}$$

Thus, $T$ should be large enough to represent useful probability distinctions, but not so large that range updates become inefficient or risk zero-width subranges. This is one of the practical trade-offs in range-coder design.

### Rust Implementation

Following the discussion in Sections 22.7.2 and 22.7.3 on adaptive probability models and finite-precision range coding, Program 22.7.2 provides a practical implementation of adaptive integer range coding with synchronized decoding. Unlike the static arithmetic coder of Program 22.7.1, where symbol probabilities remain fixed throughout the encoding process, this implementation continuously updates symbol frequencies as new symbols are observed. The adaptive model allows both encoder and decoder to learn the source statistics directly from the data stream without transmitting a separate probability table. The program combines the adaptive probability update mechanism of Equation (22.7.28) with the integer interval refinement process of Equations (22.7.44) and (22.7.45), demonstrating how practical entropy coders maintain deterministic and reproducible compression using finite-precision arithmetic. In addition to performing encoding and decoding, the implementation reports entropy-based diagnostics that illustrate the relationship between adaptive probability estimation, coding efficiency, and empirical source statistics.

At the core of the implementation is the `AdaptiveModel` structure, which encapsulates the evolving probability model used by both the encoder and decoder. The model stores the symbol alphabet together with a set of frequency counts that are initialized using additive smoothing. This design directly reflects the adaptive probability formulation introduced in Equation (22.7.28), where each symbol begins with a positive count to prevent zero-probability events. Because the same model update rules are applied by both encoder and decoder, the probability model can be reconstructed dynamically from previously processed symbols without transmitting a separate frequency table.

The `new`, `total`, `symbol_index`, `cumulative_bounds`, `probability`, and `update` methods collectively implement the adaptive probability model. The `total` method computes the current total frequency, while `probability` evaluates the adaptive symbol probabilities corresponding to Equation (22.7.28). The `cumulative_bounds` method constructs the cumulative frequencies required by Equation (22.7.29), producing the interval boundaries used during range subdivision. After each symbol is encoded or decoded, the `update` method increments the corresponding count so that future probability estimates reflect the evolving source statistics.

The finite-precision coding state is represented by the `RangeState` structure. Rather than maintaining a real-valued interval as in Section 22.7.1, the coder stores an integer lower endpoint and an integer range. This representation corresponds directly to the coding interval described by Equations (22.7.40) and (22.7.41). The use of large integer arithmetic eliminates floating-point ambiguity and ensures deterministic behavior across platforms, which is a central requirement for reproducible lossless compression.

The `update` method of `RangeState` implements the integer interval refinement process. Given cumulative frequencies and the total symbol frequency, it computes the new interval boundaries using the finite-precision update rules of Equations (22.7.44) and (22.7.45). These operations partition the current interval into subranges proportional to the adaptive symbol frequencies. The implementation also verifies that the resulting interval width remains positive, thereby enforcing the requirement expressed by Equation (22.7.46) and preventing the creation of invalid zero-width coding intervals.

The `encode` function performs adaptive range encoding. Beginning with the full integer coding interval, it processes each symbol in sequence, obtains the current adaptive probability estimate from the model, computes the corresponding cumulative-frequency range, and updates the coding interval. Simultaneously, it accumulates the quantity $-\log_2 p^{(k)}(s_k)$, thereby evaluating the adaptive ideal code length described by Equation (22.7.31). The resulting interval represents the encoded message, and a representative code value is selected from its interior for transmission or storage.

The `decode` function performs the inverse operation. Starting from the encoded code value and an identical initial model, it repeatedly determines which cumulative-frequency interval contains the current code position. Once the corresponding symbol is identified, the coding interval is updated using the same finite-precision arithmetic employed by the encoder, and the adaptive model is synchronized by applying the same count update. This procedure demonstrates the central property of adaptive coding: encoder and decoder remain synchronized solely through deterministic model evolution and do not require explicit transmission of probability tables.

The diagnostic functions `empirical_entropy` and `print_empirical_frequencies` provide information-theoretic measurements of the source. The entropy calculation evaluates the empirical entropy of the message distribution, corresponding to the entropy concepts introduced earlier in the chapter. Comparing this entropy with the adaptive coding rate provides a quantitative measure of model efficiency and reveals the cost incurred while the adaptive model is still learning the source statistics.

The `main` function demonstrates the complete adaptive range-coding workflow. It begins by defining an alphabet and selecting an additive smoothing parameter. A representative message is then encoded using the adaptive model, producing a final integer interval and an associated code value. The decoder reconstructs the original message from this code value while independently reproducing the adaptive probability model. Finally, the program reports the final coding state, adaptive coding rate, empirical entropy, estimated compression savings, symbol-frequency statistics, and a round-trip verification. Together, these diagnostics illustrate how adaptive probability estimation and finite-precision interval refinement cooperate to produce a practical entropy coder suitable for real-world compression systems.

```rust
// Program 22.7.2. Adaptive Integer Range Coding with Synchronized Decoding
//
// Problem statement:
// Demonstrate finite-precision range coding using integer interval updates and
// an adaptive probability model. The encoder and decoder begin with identical
// smoothed symbol counts, update the model after each processed symbol, and
// therefore remain synchronized without transmitting a full static probability
// table. The program illustrates the adaptive probabilities of Equation
// (22.7.28), the cumulative frequencies of Equation (22.7.29), and the integer
// range updates of Equations (22.7.44) and (22.7.45).

const STATE_BITS: u32 = 120;
const STATE_SIZE: u128 = 1u128 << STATE_BITS;

#[derive(Clone, Debug)]
struct AdaptiveModel {
    alphabet: Vec<char>,
    counts: Vec<u64>,
}

impl AdaptiveModel {
    fn new(alphabet: Vec<char>, alpha_count: u64) -> Result<Self, String> {
        if alphabet.is_empty() {
            return Err("alphabet must not be empty".to_string());
        }

        if alpha_count == 0 {
            return Err("alpha_count must be positive".to_string());
        }

        Ok(Self {
            counts: vec![alpha_count; alphabet.len()],
            alphabet,
        })
    }

    fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    fn symbol_index(&self, ch: char) -> Result<usize, String> {
        self.alphabet
            .iter()
            .position(|&a| a == ch)
            .ok_or_else(|| format!("unknown symbol '{}'", ch))
    }

    fn cumulative_bounds(&self, index: usize) -> (u64, u64, u64) {
        let c_low: u64 = self.counts[..index].iter().sum();
        let c_high = c_low + self.counts[index];
        let total = self.total();

        (c_low, c_high, total)
    }

    fn probability(&self, index: usize) -> f64 {
        self.counts[index] as f64 / self.total() as f64
    }

    fn update(&mut self, index: usize) {
        self.counts[index] += 1;
    }

    fn print_state(&self) {
        println!("{:>8} {:>10} {:>16}", "symbol", "count", "probability");

        for (ch, count) in self.alphabet.iter().zip(self.counts.iter()) {
            let p = *count as f64 / self.total() as f64;
            println!("{:>8} {:>10} {:>16.8}", ch, count, p);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RangeState {
    low: u128,
    range: u128,
}

impl RangeState {
    fn new() -> Self {
        Self {
            low: 0,
            range: STATE_SIZE,
        }
    }

    fn update(&mut self, c_low: u64, c_high: u64, total: u64) -> Result<(), String> {
        let old_low = self.low;
        let old_range = self.range;

        let sub_low = (old_range * c_low as u128) / total as u128;
        let sub_high = (old_range * c_high as u128) / total as u128;

        let new_range = sub_high
            .checked_sub(sub_low)
            .ok_or_else(|| "invalid cumulative range".to_string())?;

        if new_range == 0 {
            return Err("zero-width range encountered; increase STATE_BITS".to_string());
        }

        self.low = old_low + sub_low;
        self.range = new_range;

        Ok(())
    }

    fn high(&self) -> u128 {
        self.low + self.range
    }

    fn code_value(&self) -> u128 {
        self.low + self.range / 2
    }
}

fn encode(message: &[char], alphabet: &[char], alpha_count: u64) -> Result<(RangeState, f64), String> {
    let mut model = AdaptiveModel::new(alphabet.to_vec(), alpha_count)?;
    let mut state = RangeState::new();
    let mut ideal_bits = 0.0;

    for &ch in message {
        let index = model.symbol_index(ch)?;
        let p = model.probability(index);
        ideal_bits += -p.log2();

        let (c_low, c_high, total) = model.cumulative_bounds(index);
        state.update(c_low, c_high, total)?;

        model.update(index);
    }

    Ok((state, ideal_bits))
}

fn decode(
    code_value: u128,
    message_len: usize,
    alphabet: &[char],
    alpha_count: u64,
) -> Result<Vec<char>, String> {
    let mut model = AdaptiveModel::new(alphabet.to_vec(), alpha_count)?;
    let mut state = RangeState::new();
    let mut output = Vec::with_capacity(message_len);

    for _ in 0..message_len {
        let total = model.total();
        let offset = code_value
            .checked_sub(state.low)
            .ok_or_else(|| "code value fell below current interval".to_string())?;

        if offset >= state.range {
            return Err("code value fell outside current interval".to_string());
        }

        let mut chosen_index = None;

        for index in 0..model.alphabet.len() {
            let (c_low, c_high, _) = model.cumulative_bounds(index);

            let sub_low = (state.range * c_low as u128) / total as u128;
            let sub_high = (state.range * c_high as u128) / total as u128;

            if offset >= sub_low && offset < sub_high {
                chosen_index = Some(index);
                break;
            }
        }

        let index = chosen_index.ok_or_else(|| "failed to decode symbol".to_string())?;
        let ch = model.alphabet[index];

        let (c_low, c_high, total) = model.cumulative_bounds(index);
        state.update(c_low, c_high, total)?;
        model.update(index);

        output.push(ch);
    }

    Ok(output)
}

fn empirical_entropy(message: &[char], alphabet: &[char]) -> f64 {
    let n = message.len() as f64;

    alphabet
        .iter()
        .map(|&ch| {
            let count = message.iter().filter(|&&x| x == ch).count() as f64;
            let p = count / n;

            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum()
}

fn print_empirical_frequencies(message: &[char], alphabet: &[char]) {
    let n = message.len() as f64;

    println!("{:>8} {:>10} {:>16}", "symbol", "count", "p_hat");

    for &ch in alphabet {
        let count = message.iter().filter(|&&x| x == ch).count();
        let p_hat = count as f64 / n;
        println!("{:>8} {:>10} {:>16.8}", ch, count, p_hat);
    }
}

fn main() -> Result<(), String> {
    let alphabet = vec!['A', 'B', 'C', 'D'];
    let alpha_count = 1;

    let message_string = "ABACABADABACABA";
    let message: Vec<char> = message_string.chars().collect();

    let initial_model = AdaptiveModel::new(alphabet.clone(), alpha_count)?;

    let (final_state, adaptive_ideal_bits) = encode(&message, &alphabet, alpha_count)?;
    let code_value = final_state.code_value();

    let decoded = decode(code_value, message.len(), &alphabet, alpha_count)?;
    let decoded_string: String = decoded.iter().collect();

    let entropy = empirical_entropy(&message, &alphabet);
    let adaptive_rate = adaptive_ideal_bits / message.len() as f64;
    let raw_bits = 2.0 * message.len() as f64;

    println!("Adaptive Integer Range Coding");
    println!("=============================");
    println!();

    println!("Initial Adaptive Model");
    println!("----------------------");
    println!("alpha count                 = {}", alpha_count);
    println!("initial total frequency     = {}", initial_model.total());
    initial_model.print_state();

    println!();
    println!("Message");
    println!("-------");
    println!("input message               = {}", message_string);
    println!("number of symbols            = {}", message.len());

    println!();
    println!("Final Integer Range State");
    println!("-------------------------");
    println!("state bits                  = {}", STATE_BITS);
    println!("initial range size           = 2^{}", STATE_BITS);
    println!("final low                    = {}", final_state.low);
    println!("final high                   = {}", final_state.high());
    println!("final range                  = {}", final_state.range);
    println!("chosen code value            = {}", code_value);

    println!();
    println!("Information Diagnostics");
    println!("-----------------------");
    println!(
        "adaptive ideal length        = {:.8} bits",
        adaptive_ideal_bits
    );
    println!(
        "adaptive average rate        = {:.8} bits/symbol",
        adaptive_rate
    );
    println!(
        "empirical entropy H_hat      = {:.8} bits/symbol",
        entropy
    );
    println!(
        "raw fixed-width size         = {:.0} bits",
        raw_bits
    );
    println!(
        "ideal adaptive saving        = {:.8} bits",
        raw_bits - adaptive_ideal_bits
    );

    println!();
    println!("Empirical Frequencies");
    println!("---------------------");
    print_empirical_frequencies(&message, &alphabet);

    println!();
    println!("Decoding Check");
    println!("--------------");
    println!("decoded message              = {}", decoded_string);
    println!("round-trip exact             = {}", decoded_string == message_string);

    Ok(())
}
```

Program 22.7.2 demonstrates how adaptive probability modeling and finite-precision range coding can be combined to produce a fully deterministic lossless compression framework. Whereas static arithmetic coding relies on a fixed probability distribution known in advance, adaptive coding continuously refines its probability estimates as symbols are observed. This allows the coder to learn the statistical structure of the data directly from the message while maintaining exact synchronization between encoder and decoder.

The implementation highlights the importance of cumulative-frequency construction and integer interval refinement. By replacing real-valued interval arithmetic with integer computations, the coder achieves reproducible behavior and avoids the precision-growth issues associated with ideal arithmetic coding formulations. This transition from real intervals to integer intervals represents one of the key practical advances that made arithmetic coding suitable for deployment in production compression systems.

The reported diagnostics also illustrate the distinction between empirical entropy and achieved coding rate. Because the adaptive model begins with limited knowledge of the source distribution, the early symbols are encoded using imperfect probability estimates. As additional symbols are processed, the model gradually converges toward the empirical distribution and the coding rate approaches the theoretical entropy limit. This learning behavior explains why adaptive coders often perform well on unknown or nonstationary data sources.

More broadly, the modular structure of the implementation provides a foundation for exploring more sophisticated probability models. Context-dependent models, predictive coders, bit-plane coders, and modern entropy-coding architectures all build upon the same fundamental principle demonstrated here: accurate probability estimation is as important as the coding mechanism itself. The range coder supplies the interval-refinement engine, while the probability model determines how efficiently information can be represented.

## 22.7.4. Diagnostics and Reproducibility for Interval Coders

Arithmetic and range coding are powerful, but they are less transparent than Huffman coding. A prefix code can often be inspected through its code-length table. An arithmetic coder’s behavior depends on interval evolution, probability updates, renormalization, termination, and finite-precision state. For numerical computing, these details must be part of the diagnostic record.

The first diagnostic is round-trip exactness. As in Section 22.6, a lossless interval coder must satisfy,

$$s = \operatorname{Decode}(\operatorname{Encode}(s)) \tag{22.7.54}$$

For large datasets, this should be verified with byte equality during testing and with checksums or CRCs during storage:

$$C(s)=C(\widehat s),\qquad\widehat s=\operatorname{Decode}(\operatorname{Encode}(s)) \tag{22.7.55}$$

Because interval coders are highly stateful, round-trip tests should include short messages, long messages, rare symbols, all-symbol alphabets, adaptive-model resets, block boundaries, and termination edge cases.

The second diagnostic is coding efficiency. Let $B_{\mathrm{payload}}$ be the number of encoded bits and $N$ the number of source symbols. The achieved rate is:

$$R_{\mathrm{ach}} = \frac{B_{\mathrm{payload}}}{N} \tag{22.7.56}$$

If the empirical entropy is:

$$\widehat H = -\sum_i \widehat p_i\log_2\widehat p_i \tag{22.7.57}$$

then the excess rate is,

$$R_{\mathrm{excess}} = R_{\mathrm{ach}}-\widehat H \tag{22.7.58}$$

For a well-matched model and sufficiently long messages, $R_{\mathrm{excess}}$ should be small. If it is large, possible causes include poor probability modeling, excessive model overhead, short block sizes, probability quantization, inefficient termination, or implementation constraints.

The third diagnostic is model cost. For static models, the total rate is:

$$R_{\mathrm{total}} = \frac{B_{\mathrm{payload}}+B_{\mathrm{model}}}{N} \tag{22.7.59}$$

For adaptive models, the model cost is implicit in early coding inefficiency and update overhead rather than in a transmitted table. A useful comparison is therefore,

$$R_{\mathrm{static}}\quad\text{versus}\quad R_{\mathrm{adaptive}} \tag{22.7.60}$$

where both rates include all required side information. Static models may perform better on homogeneous data, while adaptive models may perform better when distributions drift over time or across blocks.

The fourth diagnostic is determinism. Given the same input string $s$, settings $\theta$, and model initialization, a scientific compression workflow should satisfy,

$$\operatorname{Encode}(s;\theta) = \operatorname{Encode}(s;\theta)\tag{22.7.61}$$

across repeated runs on the same specification. Unlike floating-point numerical algorithms, lossless compression should generally avoid platform-dependent arithmetic. Integer range coding is therefore preferred over real-interval arithmetic for reproducible implementations. The specification should define integer widths, overflow behavior, frequency rescaling, symbol ordering, block termination, and byte output order.

The fifth diagnostic is streaming and locality. If the data are divided into blocks:

$$s=(s^{(1)},s^{(2)},\ldots,s^{(B)}) \tag{22.7.62}$$

then each block may be encoded independently:

$$z^{(b)} = \operatorname{Encode}(s^{(b)}) \tag{22.7.63}$$

Independent blocks improve random access and error localization, but they may reduce compression efficiency because probability models reset. The total compressed size is

$$B_{\mathrm{blocked}} = \sum_{b=1}^{B} B^{(b)} \tag{22.7.64}$$

A single global stream may achieve a smaller size,

$$B_{\mathrm{global}}\le B_{\mathrm{blocked}} \tag{22.7.65}$$

but at the cost of weaker locality and more fragile recovery after corruption. In numerical workflows, this trade-off matters for checkpoints, large arrays, distributed output, and partial restart.

The sixth diagnostic is throughput. If compression and decompression times are $T_c$ and $T_d$, and the raw size is $B_{\mathrm{raw}}$, then,

$$v_c=\frac{B_{\mathrm{raw}}}{T_c},\qquad v_d=\frac{B_{\mathrm{raw}}}{T_d} \tag{22.7.66}$$

Arithmetic coding may improve compression ratio relative to Huffman coding, but it can be slower or more difficult to parallelize. Hybrid coders and table-based coders are often designed to navigate this trade-off. Recent work on fast entropy-coding architectures and hybrid arithmetic-Huffman compression highlights that practical coders must be evaluated over a full trade-off surface, not only by compressed size (Auli-Llinas, 2023; Wiseman, 2025).

The main conclusion is that arithmetic coding and range coding extend the compression framework of Section 22.6 by replacing symbol-level integer code lengths with message-level interval refinement. This allows compression rates close to entropy when the probability model is accurate. However, the increased modeling and implementation complexity makes diagnostics essential. Exact round-trip recovery, model specification, integer-state reproducibility, block-local integrity checks, compression ratio, and throughput must all be reported for a trustworthy numerical compression workflow. Section 22.8 closes the chapter by turning from compact representation to expanded representation: arbitrary-precision arithmetic, where additional digits are deliberately introduced to verify, stabilize, or enable difficult numerical computations.

### Introductory Paragraph Before the Code

Following the discussion in Section 22.7.4 on diagnostics, reproducibility, and verification for interval coders, Program 22.7.3 provides a practical framework for evaluating the correctness and performance of adaptive range-coding implementations. While arithmetic and range coding can achieve compression rates close to the entropy limit, their highly stateful nature makes systematic verification essential. This program implements a collection of diagnostic procedures that assess round-trip exactness, checksum consistency, coding efficiency, determinism, streaming behavior, and throughput. By combining adaptive range coding with quantitative performance metrics, the implementation demonstrates how compression software can be validated and benchmarked in a manner suitable for scientific and numerical computing environments. The framework highlights the importance of reproducibility and diagnostic transparency when entropy coders are used in long-term archives, distributed workflows, and computational research applications.

### Explanation of the Implementation

At the core of the implementation is the `AdaptiveModel` structure, which encapsulates the evolving probability model used by both the encoder and decoder. The model maintains symbol frequencies together with the alphabet definition and provides the functionality required to compute adaptive probability estimates and cumulative frequency intervals. This directly corresponds to the adaptive probability framework developed in Section 22.7.2, where probabilities evolve dynamically as symbols are processed. By updating the model after every encoded or decoded symbol, the encoder and decoder remain synchronized without requiring explicit transmission of a static frequency table.

The methods `new`, `total`, `symbol_index`, `cumulative_bounds`, `probability`, and `update` collectively implement the adaptive modeling process. The `cumulative_bounds` method computes the cumulative frequency ranges that define the coding intervals, while the `probability` method evaluates the current adaptive symbol probabilities. The `update` method modifies the model after each symbol is processed, allowing future probability estimates to reflect the observed source statistics. This mechanism embodies the adaptive probability updates described by Equations (22.7.28) and (22.7.29).

The finite-precision coding interval is represented by the `RangeState` structure. Rather than using real-valued interval endpoints, the implementation stores an integer lower bound and an integer range, corresponding to the finite-precision representation introduced by Equations (22.7.40) and (22.7.41). This approach ensures deterministic behavior and eliminates ambiguity associated with floating-point arithmetic. The `update` method performs the interval refinement process using cumulative frequencies, implementing the integer range-update formulas of Equations (22.7.44) and (22.7.45). The implementation also verifies that the resulting interval width remains positive, thereby enforcing the requirement expressed by Equation (22.7.46).

The `EncodedBlock` structure serves as a container for all information associated with an encoded data block. In addition to storing the final coding state and representative code value, it records the number of encoded symbols, the adaptive ideal code length, and the estimated payload size. This information is later used to evaluate coding efficiency and compare global and blocked coding strategies.

The `encode_block` function performs adaptive range encoding for a single block of data. Beginning with an initial coding interval and an adaptive probability model, it processes each symbol sequentially, refines the interval using the cumulative frequency ranges, and accumulates the ideal coding length implied by the adaptive probabilities. The resulting interval represents the encoded block, while the payload-bit estimate provides an approximation to the compressed size used in the diagnostics of Equations (22.7.56)–(22.7.60).

The `decode_block` function reverses the encoding process. Starting from the encoded code value and an identical initial model, it repeatedly determines which subinterval contains the current code position and reconstructs the original symbol sequence. Because the decoder updates its probability model in exactly the same manner as the encoder, both processes remain synchronized throughout decoding. Successful reconstruction verifies the round-trip exactness condition expressed by Equation (22.7.54).

The `encode_blocks` and `decode_blocks` functions extend the framework to support block-based coding. Instead of compressing an entire data stream as a single interval, the data can be divided into independent blocks and processed separately. This functionality allows the program to investigate the locality and streaming trade-offs discussed in Equations (22.7.62)–(22.7.65). By comparing blocked and global coding rates, the implementation quantifies the compression penalty associated with resetting the probability model at block boundaries.

Several auxiliary diagnostic functions are included to support reproducibility testing. The `checksum64` function computes a deterministic checksum that can be used to verify data integrity after decoding, corresponding to the checksum validation discussed in Equation (22.7.55). The `empirical_entropy` function evaluates the empirical entropy of the source distribution, providing the reference quantity required for the achieved-rate and excess-rate diagnostics of Equations (22.7.56)–(22.7.58). Additional helper functions generate synthetic test data, print empirical frequency tables, summarize block statistics, and construct payload-size histograms for diagnostic reporting.

The `main` function integrates all diagnostic procedures into a single experimental workflow. It begins by generating a representative test dataset and computing its empirical entropy and checksum. The data are then encoded and decoded using both global and blocked coding strategies. The program verifies round-trip correctness through direct comparison of the original and reconstructed data and confirms integrity through checksum equality. Determinism is tested by encoding the same input twice and comparing the resulting coding states. The implementation then evaluates coding efficiency by reporting achieved rates, excess rates, payload sizes, and ideal coding lengths. Finally, compression and decompression throughput are measured using wall-clock timing, providing quantitative estimates of performance according to Equation (22.7.66). Together, these diagnostics form a comprehensive validation framework for interval-coding systems.

### Concluding Remarks After the Code

```rust
// Program 22.7.3. Diagnostics and Reproducibility for Adaptive Range Coding

use std::collections::BTreeMap;
use std::time::Instant;

const STATE_BITS: u32 = 126;
const STATE_SIZE: u128 = 1u128 << STATE_BITS;

#[derive(Clone, Debug)]
struct AdaptiveModel {
    alphabet: Vec<u8>,
    counts: Vec<u64>,
}

impl AdaptiveModel {
    fn new(alphabet: Vec<u8>, alpha_count: u64) -> Result<Self, String> {
        if alphabet.is_empty() {
            return Err("alphabet must not be empty".to_string());
        }
        if alpha_count == 0 {
            return Err("alpha_count must be positive".to_string());
        }

        Ok(Self {
            counts: vec![alpha_count; alphabet.len()],
            alphabet,
        })
    }

    fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    fn symbol_index(&self, symbol: u8) -> Result<usize, String> {
        self.alphabet
            .iter()
            .position(|&x| x == symbol)
            .ok_or_else(|| format!("unknown symbol byte {}", symbol))
    }

    fn cumulative_bounds(&self, index: usize) -> (u64, u64, u64) {
        let c_low: u64 = self.counts[..index].iter().sum();
        let c_high = c_low + self.counts[index];
        let total = self.total();
        (c_low, c_high, total)
    }

    fn probability(&self, index: usize) -> f64 {
        self.counts[index] as f64 / self.total() as f64
    }

    fn update(&mut self, index: usize) {
        self.counts[index] += 1;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RangeState {
    low: u128,
    range: u128,
}

impl RangeState {
    fn new() -> Self {
        Self {
            low: 0,
            range: STATE_SIZE,
        }
    }

    fn update(&mut self, c_low: u64, c_high: u64, total: u64) -> Result<(), String> {
        let sub_low = (self.range * c_low as u128) / total as u128;
        let sub_high = (self.range * c_high as u128) / total as u128;

        let new_range = sub_high
            .checked_sub(sub_low)
            .ok_or_else(|| "invalid cumulative interval".to_string())?;

        if new_range == 0 {
            return Err(
                "zero-width range encountered; use shorter blocks or implement renormalization"
                    .to_string(),
            );
        }

        self.low += sub_low;
        self.range = new_range;

        Ok(())
    }

    fn high(&self) -> u128 {
        self.low + self.range
    }

    fn code_value(&self) -> u128 {
        self.low + self.range / 2
    }

    fn payload_bits(&self) -> f64 {
        (STATE_BITS as f64 - (self.range as f64).log2()).ceil()
    }
}

#[derive(Clone, Debug)]
struct EncodedBlock {
    state: RangeState,
    code_value: u128,
    symbols: usize,
    ideal_bits: f64,
    payload_bits: f64,
}

fn encode_block(
    block: &[u8],
    alphabet: &[u8],
    alpha_count: u64,
) -> Result<EncodedBlock, String> {
    let mut model = AdaptiveModel::new(alphabet.to_vec(), alpha_count)?;
    let mut state = RangeState::new();
    let mut ideal_bits = 0.0;

    for &symbol in block {
        let index = model.symbol_index(symbol)?;
        let probability = model.probability(index);
        ideal_bits += -probability.log2();

        let (c_low, c_high, total) = model.cumulative_bounds(index);
        state.update(c_low, c_high, total)?;
        model.update(index);
    }

    Ok(EncodedBlock {
        state,
        code_value: state.code_value(),
        symbols: block.len(),
        ideal_bits,
        payload_bits: state.payload_bits(),
    })
}

fn decode_block(
    encoded: &EncodedBlock,
    alphabet: &[u8],
    alpha_count: u64,
) -> Result<Vec<u8>, String> {
    let mut model = AdaptiveModel::new(alphabet.to_vec(), alpha_count)?;
    let mut state = RangeState::new();
    let mut output = Vec::with_capacity(encoded.symbols);

    for _ in 0..encoded.symbols {
        let offset = encoded
            .code_value
            .checked_sub(state.low)
            .ok_or_else(|| "code value below current interval".to_string())?;

        if offset >= state.range {
            return Err("code value outside current interval".to_string());
        }

        let total = model.total();
        let mut chosen_index = None;

        for index in 0..model.alphabet.len() {
            let (c_low, c_high, _) = model.cumulative_bounds(index);
            let sub_low = (state.range * c_low as u128) / total as u128;
            let sub_high = (state.range * c_high as u128) / total as u128;

            if offset >= sub_low && offset < sub_high {
                chosen_index = Some(index);
                break;
            }
        }

        let index = chosen_index.ok_or_else(|| "failed to decode symbol".to_string())?;
        let symbol = model.alphabet[index];

        let (c_low, c_high, total) = model.cumulative_bounds(index);
        state.update(c_low, c_high, total)?;
        model.update(index);

        output.push(symbol);
    }

    Ok(output)
}

fn encode_blocks(
    data: &[u8],
    block_size: usize,
    alphabet: &[u8],
    alpha_count: u64,
) -> Result<Vec<EncodedBlock>, String> {
    if block_size == 0 {
        return Err("block_size must be positive".to_string());
    }

    data.chunks(block_size)
        .map(|block| encode_block(block, alphabet, alpha_count))
        .collect()
}

fn decode_blocks(
    blocks: &[EncodedBlock],
    alphabet: &[u8],
    alpha_count: u64,
) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();

    for block in blocks {
        output.extend(decode_block(block, alphabet, alpha_count)?);
    }

    Ok(output)
}

fn checksum64(data: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut hash = OFFSET;

    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }

    hash
}

fn empirical_entropy(data: &[u8], alphabet: &[u8]) -> f64 {
    let n = data.len() as f64;

    alphabet
        .iter()
        .map(|&symbol| {
            let count = data.iter().filter(|&&x| x == symbol).count() as f64;
            let p = count / n;

            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum()
}

fn make_test_data(repetitions: usize) -> Vec<u8> {
    let pattern = b"ABACABADABACABAABACABAD";
    let mut data = Vec::with_capacity(pattern.len() * repetitions);

    for _ in 0..repetitions {
        data.extend_from_slice(pattern);
    }

    data
}

fn print_frequencies(data: &[u8], alphabet: &[u8]) {
    let n = data.len() as f64;

    println!("{:>8} {:>12} {:>16}", "symbol", "count", "p_hat");

    for &symbol in alphabet {
        let count = data.iter().filter(|&&x| x == symbol).count();
        let p_hat = count as f64 / n;

        println!(
            "{:>8} {:>12} {:>16.8}",
            symbol as char, count, p_hat
        );
    }
}

fn print_block_summary(blocks: &[EncodedBlock]) {
    println!(
        "{:>8} {:>12} {:>18} {:>18}",
        "block", "symbols", "ideal bits", "payload bits"
    );

    for (i, block) in blocks.iter().enumerate() {
        println!(
            "{:>8} {:>12} {:>18.8} {:>18.8}",
            i, block.symbols, block.ideal_bits, block.payload_bits
        );
    }
}

fn state_histogram(blocks: &[EncodedBlock]) -> BTreeMap<u64, usize> {
    let mut histogram = BTreeMap::new();

    for block in blocks {
        let bucket = block.payload_bits as u64;
        *histogram.entry(bucket).or_insert(0) += 1;
    }

    histogram
}

fn main() -> Result<(), String> {
    let alphabet = vec![b'A', b'B', b'C', b'D'];
    let alpha_count = 1;
    let block_size = 24;

    let data = make_test_data(3);

    let raw_bits = 2.0 * data.len() as f64;
    let input_checksum = checksum64(&data);
    let entropy = empirical_entropy(&data, &alphabet);

    let start_global_encode = Instant::now();
    let global_block = encode_block(&data, &alphabet, alpha_count)?;
    let global_encode_time = start_global_encode.elapsed();

    let start_global_decode = Instant::now();
    let global_decoded = decode_block(&global_block, &alphabet, alpha_count)?;
    let global_decode_time = start_global_decode.elapsed();

    let repeat_global_block = encode_block(&data, &alphabet, alpha_count)?;

    let start_blocked_encode = Instant::now();
    let blocked = encode_blocks(&data, block_size, &alphabet, alpha_count)?;
    let blocked_encode_time = start_blocked_encode.elapsed();

    let start_blocked_decode = Instant::now();
    let blocked_decoded = decode_blocks(&blocked, &alphabet, alpha_count)?;
    let blocked_decode_time = start_blocked_decode.elapsed();

    let global_checksum = checksum64(&global_decoded);
    let blocked_checksum = checksum64(&blocked_decoded);

    let global_rate = global_block.payload_bits / data.len() as f64;
    let global_excess_rate = global_rate - entropy;

    let blocked_payload_bits: f64 = blocked.iter().map(|b| b.payload_bits).sum();
    let blocked_ideal_bits: f64 = blocked.iter().map(|b| b.ideal_bits).sum();
    let blocked_rate = blocked_payload_bits / data.len() as f64;
    let blocked_excess_rate = blocked_rate - entropy;

    let raw_bytes = data.len() as f64;

    let global_compression_speed =
        raw_bytes / global_encode_time.as_secs_f64().max(1.0e-12);
    let global_decompression_speed =
        raw_bytes / global_decode_time.as_secs_f64().max(1.0e-12);

    let blocked_compression_speed =
        raw_bytes / blocked_encode_time.as_secs_f64().max(1.0e-12);
    let blocked_decompression_speed =
        raw_bytes / blocked_decode_time.as_secs_f64().max(1.0e-12);

    println!("Interval-Coder Diagnostics and Reproducibility");
    println!("==============================================");
    println!();

    println!("Input Data");
    println!("----------");
    println!("symbols                         = {}", data.len());
    println!("alphabet size                   = {}", alphabet.len());
    println!("raw fixed-width size            = {:.0} bits", raw_bits);
    println!("checksum C(s)                   = {:016x}", input_checksum);
    println!();
    print_frequencies(&data, &alphabet);

    println!();
    println!("Global Stream Diagnostics");
    println!("-------------------------");
    println!("payload bits B_payload          = {:.0}", global_block.payload_bits);
    println!("adaptive ideal bits             = {:.8}", global_block.ideal_bits);
    println!("achieved rate R_ach             = {:.8} bits/symbol", global_rate);
    println!("empirical entropy H_hat         = {:.8} bits/symbol", entropy);
    println!(
        "excess rate R_excess            = {:.8} bits/symbol",
        global_excess_rate
    );
    println!("final low                       = {}", global_block.state.low);
    println!("final high                      = {}", global_block.state.high());
    println!("final range                     = {}", global_block.state.range);

    println!();
    println!("Round-Trip and Checksum Tests");
    println!("-----------------------------");
    println!("global round-trip exact         = {}", global_decoded == data);
    println!("blocked round-trip exact        = {}", blocked_decoded == data);
    println!(
        "C(s) = C(decoded global)        = {}",
        input_checksum == global_checksum
    );
    println!(
        "C(s) = C(decoded blocked)       = {}",
        input_checksum == blocked_checksum
    );
    println!("global checksum                 = {:016x}", global_checksum);
    println!("blocked checksum                = {:016x}", blocked_checksum);

    println!();
    println!("Determinism Test");
    println!("----------------");
    println!(
        "same low across repeated runs   = {}",
        global_block.state.low == repeat_global_block.state.low
    );
    println!(
        "same range across repeated runs = {}",
        global_block.state.range == repeat_global_block.state.range
    );
    println!(
        "same code value                 = {}",
        global_block.code_value == repeat_global_block.code_value
    );

    println!();
    println!("Blocked Stream Diagnostics");
    println!("--------------------------");
    println!("block size                      = {}", block_size);
    println!("number of blocks                = {}", blocked.len());
    println!("blocked payload bits            = {:.0}", blocked_payload_bits);
    println!("blocked adaptive ideal bits     = {:.8}", blocked_ideal_bits);
    println!("blocked rate                    = {:.8} bits/symbol", blocked_rate);
    println!(
        "blocked excess rate             = {:.8} bits/symbol",
        blocked_excess_rate
    );
    println!(
        "blocked minus global payload    = {:.0} bits",
        blocked_payload_bits - global_block.payload_bits
    );

    println!();
    println!("Per-Block Summary");
    println!("-----------------");
    print_block_summary(&blocked);

    println!();
    println!("Payload-Bit Histogram");
    println!("---------------------");

    for (payload_bits, count) in state_histogram(&blocked) {
        println!("{:>8} bits : {:>4} block(s)", payload_bits, count);
    }

    println!();
    println!("Throughput Diagnostics");
    println!("----------------------");
    println!(
        "global compression time         = {:.6e} s",
        global_encode_time.as_secs_f64()
    );
    println!(
        "global decompression time       = {:.6e} s",
        global_decode_time.as_secs_f64()
    );
    println!(
        "global compression throughput   = {:.3} bytes/s",
        global_compression_speed
    );
    println!(
        "global decompression throughput = {:.3} bytes/s",
        global_decompression_speed
    );
    println!(
        "blocked compression time        = {:.6e} s",
        blocked_encode_time.as_secs_f64()
    );
    println!(
        "blocked decompression time      = {:.6e} s",
        blocked_decode_time.as_secs_f64()
    );
    println!(
        "blocked compression throughput  = {:.3} bytes/s",
        blocked_compression_speed
    );
    println!(
        "blocked decompression throughput= {:.3} bytes/s",
        blocked_decompression_speed
    );

    Ok(())
}
```

Program 22.7.3 demonstrates that the evaluation of an interval coder extends beyond simple compression and decompression. In scientific and numerical computing applications, reproducibility, determinism, and integrity verification are often as important as compression ratio. The diagnostic procedures implemented here provide a systematic methodology for validating these properties and ensuring that encoded data can be recovered exactly under repeated executions and across different computational environments.

The comparison between global and blocked coding illustrates an important practical trade-off. Global coding generally produces smaller compressed representations because the probability model can adapt continuously across the entire dataset. Blocked coding, however, improves locality, random access, checkpointing, and error containment by allowing independent decoding of individual blocks. The additional payload observed in blocked coding represents the cost of resetting the adaptive model and losing long-range statistical information.

The achieved-rate and excess-rate diagnostics further demonstrate the relationship between probability modeling and compression efficiency. When the adaptive model closely approximates the true source distribution, the achieved coding rate approaches the empirical entropy. Any remaining excess rate reflects modeling imperfections, finite-precision effects, block boundaries, and implementation constraints. These measurements provide a quantitative means of assessing model quality and identifying opportunities for improvement.

More broadly, the diagnostic framework established here serves as a foundation for evaluating more advanced entropy-coding systems. Modern compressors often incorporate sophisticated context models, predictive coding, semantic representations, and hybrid coding architectures. Regardless of the complexity of the underlying model, the same fundamental diagnostics remain essential: exact round-trip recovery, deterministic behavior, coding efficiency, integrity verification, streaming performance, and throughput. These considerations ensure that compression systems remain trustworthy components within larger scientific computing workflows.

# 22.8. Arbitrary-Precision Arithmetic

Arbitrary-precision arithmetic extends the arithmetic environment beyond fixed machine types such as binary32 or binary64. Instead of using a fixed number of significand bits, the precision is chosen by the computation, the user, or an adaptive strategy. This makes precision a computational resource. It can be increased to verify a result, to stabilize an ill-conditioned computation, to resolve cancellation, to support exact or near-exact geometric predicates, or to provide a high-accuracy reference against which ordinary floating-point implementations can be tested.

This section concludes the chapter because arbitrary precision completes the reliability framework developed in the preceding sections. Machine-parameter diagnostics identify the limits of fixed precision. Diagnostic plots reveal stagnation and precision loss. Integrity checks protect stored and transmitted data. Lossless compression preserves finite representations exactly. Arbitrary precision then asks what should be done when the fixed-precision arithmetic itself is not sufficient. Recent work supports treating arbitrary precision as an active component of modern scientific computing, especially in multi-precision tools, extended-precision algorithms, hydrodynamic-stability eigenvalue calculations, exact geometric kernels, and high-precision quadrature (Leitold et al., 2025; Zhang and Aiken, 2025; Dondl et al., 2025; Lévy, 2025; Vretinaris, 2026).

## 22.8.1. Precision as a Variable in Numerical Computation

A fixed-precision floating-point system may be written as:

$$\mathbb F(\beta,p,e_{\min},e_{\max}) \tag{22.8.1}$$

where $p$ is the number of significand digits. In arbitrary-precision arithmetic, $p$ is no longer fixed by the hardware format. Instead, the computation may use a family of systems:

$$\{\mathbb F_p : p=p_1,p_2,p_3,\ldots\} \tag{22.8.2}$$

A computed result therefore depends on precision:

$$\widehat y_p=\widehat F_p(x) \tag{22.8.3}$$

If $F(x)$ is the exact mathematical value, the total error at precision $p$ may be written as:

$$\widehat y_p-F(x) = \left[\widehat F_p(x)-F(x)\right] \tag{22.8.4}$$

For a stable algorithm, one often expects the dominant rounding contribution to decrease as $p$ increases. If the radix is $\beta$, then the unit roundoff behaves like,

$$u_p=\frac{1}{2}\beta^{1-p}\tag{22.8.5}$$

under rounding to nearest. A typical first-order error model has the form:

$$\|\widehat y_p-F(x)\|\lesssim C_{\mathrm{alg}}u_p + C_{\mathrm{disc}}h^q + C_{\mathrm{model}} \tag{22.8.6}$$

where $C_{\mathrm{alg}}u_p$ represents arithmetic error, $C_{\mathrm{disc}}h^q$ represents discretization error, and $C_{\mathrm{model}}$ represents modeling or data error. Increasing precision reduces only the first term. It cannot remove discretization error, modeling error, or ill-posedness. This distinction is essential: arbitrary precision is powerful, but it is not a substitute for a well-conditioned formulation or an appropriate numerical method.

A useful precision diagnostic compares results at two precisions $p<p'$:

$$\Delta_{p,p'} = \frac{\|\widehat y_{p'}-\widehat y_p\|}{\max(\|\widehat y_{p'}\|,\sigma)} \tag{22.8.7}$$

where $\sigma>0$ prevents division by zero. If $\Delta_{p,p'}$ decreases consistently as both $p$ and $p'$ are increased, then the computation is likely approaching a precision-independent result. If $\Delta_{p,p'}$ remains large or irregular, then the computation may be affected by cancellation, instability, ill-conditioning, nonnormality, or insufficient algorithmic accuracy.

One may also define a precision ladder:

$$p_1<p_2<\cdots<p_m,\tag{22.8.8}$$

and compute,

$$\widehat y_{p_1},\widehat y_{p_2},\ldots,\widehat y_{p_m} \tag{22.8.9}$$

The successive differences,

$$d_j = \frac{\|\widehat y_{p_{j+1}}-\widehat y_{p_j}\|}{\max(\|\widehat y_{p_{j+1}}\|,\sigma)}\tag{22.8.10}$$

form a convergence trace in precision space. A semilog plot of $d_j$ against $p_j$ can reveal whether additional bits are producing the expected improvement. If the trace reaches a plateau, then another error source has become dominant. If the trace grows, then the algorithm may be unstable or the higher-precision computation may not be solving exactly the same problem.

The computational cost of arbitrary precision must also be considered. If multiplication of $p$-digit numbers has cost $M(p)$, then the cost of an algorithm with many arithmetic operations changes from a fixed hardware cost to a precision-dependent cost. Classical multiplication gives:

$$M(p)=O(p^2) \tag{22.8.11}$$

while faster multiplication algorithms can reduce the asymptotic cost for very large $p$. For most numerical algorithms, however, the practical cost is governed not only by arithmetic complexity but also by memory allocation, cache behavior, normalization, conversion, and library overhead. Thus, arbitrary precision should be used deliberately, with diagnostics showing why the additional cost is justified.

### Rust Implementation

Following the discussion in Section 22.8.1 on treating precision as a computational variable rather than a fixed hardware characteristic, Program 22.8.1 provides a practical implementation of precision-ladder diagnostics using arbitrary-precision integer arithmetic. In many numerical computations, it is difficult to determine whether an observed result reflects the underlying mathematical problem or merely the limitations of the arithmetic precision used during computation. This program addresses that issue by evaluating the same quantity at a sequence of increasing precision levels and comparing the resulting approximations. By constructing a precision ladder and monitoring successive differences between computations, the implementation illustrates how the diagnostics of Equations (22.8.7)–(22.8.10) can be used to assess convergence in precision space. The example uses high-precision evaluation of $\sqrt{2}$ to demonstrate how increasing precision progressively reduces arithmetic error while simultaneously increasing computational cost, thereby emphasizing the role of precision as a controllable numerical resource.

At the core of the implementation is the `FixedDecimal` structure, which represents arbitrary-precision decimal values using integer-scaled arithmetic. Instead of relying on hardware floating-point types, the structure stores a large integer together with the number of decimal digits used to interpret that integer. This representation allows the program to emulate the family of arithmetic systems described by Equation (22.8.2), where precision is varied explicitly as part of the computation. By separating numerical value from decimal scale, the implementation provides direct control over the precision parameter $p$ appearing in Equations (22.8.1)–(22.8.5).

The methods `to_decimal_string` and `rounded_display` provide formatted output of arbitrary-precision values. These routines convert the scaled integer representation into a human-readable decimal form while preserving the precision associated with each computation. Because the objective of the program is to observe changes across precision levels, accurate presentation of the computed values is essential for interpreting the precision ladder and the convergence trace defined by Equation (22.8.10).

The `pow10` function constructs powers of ten used to define the scaling factors associated with different decimal precisions. These scaling factors serve as the numerical analogue of varying the precision parameter $p$ in Equation (22.8.2). By increasing the scale, the program effectively increases the number of representable decimal digits and thereby reduces the arithmetic error associated with the computation.

The `integer_sqrt` function computes the integer square root of a nonnegative integer using Newton iteration. This routine forms the computational kernel of the arbitrary-precision square-root calculation. Unlike hardware floating-point square-root instructions, the algorithm operates entirely on large integers and therefore permits precision levels far beyond those available in standard binary64 arithmetic. The iterative refinement process illustrates how higher-precision arithmetic can be constructed from exact integer operations.

The `sqrt2_fixed` function computes an approximation to $\sqrt{2}$ at a specified decimal precision. It first scales the argument by an appropriate power of ten and then applies the integer square-root algorithm. The resulting scaled approximation is returned as a `FixedDecimal` object. Repeating this computation for a sequence of increasing precisions generates the family of approximations $\widehat y_{p_1}, \widehat y_{p_2}, \ldots, \widehat y_{p_m}$ described by Equation (22.8.9), thereby forming the basis of the precision-ladder experiment.

The `align_to_digits` and `absolute_difference` functions support comparison of approximations computed at different precision levels. Since different computations may use different decimal scales, the values must first be represented on a common scale before meaningful differences can be computed. These routines therefore implement the normalization process required to evaluate the precision-difference diagnostics of Equations (22.8.7) and (22.8.10).

The `scientific_ratio` function evaluates relative differences in a numerically stable manner and formats them in scientific notation. This quantity serves as the computational realization of the normalized difference measures introduced in Equations (22.8.7) and (22.8.10). By reporting these values across the precision ladder, the program provides a direct view of convergence behavior as arithmetic precision increases.

The `decimal_unit_roundoff` function computes a simple decimal analogue of the unit roundoff $u_p$ introduced in Equation (22.8.5). Although the implementation uses decimal scaling rather than binary floating-point arithmetic, the function illustrates the same fundamental concept: increasing precision decreases the characteristic rounding unit and therefore reduces arithmetic error.

The `main` function orchestrates the precision-ladder experiment. It begins by defining a sequence of increasing precision levels and constructing a high-precision reference solution for $\sqrt{2}$. For each precision level, the program computes an approximation, records the execution time, and evaluates the relative error with respect to the reference solution. These results are then displayed as a precision ladder, showing how the approximation improves as precision increases. The function subsequently computes the successive differences between neighboring precision levels, thereby producing the convergence trace $d_j$ defined by Equation (22.8.10). Finally, timing information is reported to illustrate the computational cost associated with increasing precision. Taken together, these diagnostics provide a practical demonstration of the precision-space convergence analysis introduced in Section 22.8.1.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-bigint = "0.4"
num-traits = "0.2"
```

```rust
// Program 22.8.1. Precision Ladder Diagnostics Using Pure Rust Big Integers
//
// Problem statement:
// Demonstrate precision as a computational variable by computing sqrt(2)
// with increasing decimal precision. The program uses integer-scaled fixed
// point arithmetic, avoiding platform-specific arbitrary-precision libraries.
// It reports successive precision differences as a precision-space convergence
// trace, corresponding to the diagnostics in Equations (22.8.7)--(22.8.10).

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use std::time::Instant;

#[derive(Clone, Debug)]
struct FixedDecimal {
    value: BigInt,
    digits: usize,
}

impl FixedDecimal {
    fn new(value: BigInt, digits: usize) -> Self {
        Self { value, digits }
    }

    fn to_decimal_string(&self) -> String {
        let negative = self.value.is_negative();
        let mut s = self.value.abs().to_string();

        if self.digits == 0 {
            if negative {
                return format!("-{}", s);
            }
            return s;
        }

        if s.len() <= self.digits {
            let zeros_needed = self.digits + 1 - s.len();
            s = format!("{}{}", "0".repeat(zeros_needed), s);
        }

        let split = s.len() - self.digits;
        let integer_part = &s[..split];
        let fractional_part = &s[split..];

        if negative {
            format!("-{}.{}", integer_part, fractional_part)
        } else {
            format!("{}.{}", integer_part, fractional_part)
        }
    }

    fn rounded_display(&self, max_fraction_digits: usize) -> String {
        let full = self.to_decimal_string();

        if let Some(dot_position) = full.find('.') {
            let keep = dot_position + 1 + max_fraction_digits.min(self.digits);
            full[..keep.min(full.len())].to_string()
        } else {
            full
        }
    }
}

fn pow10(n: usize) -> BigInt {
    let mut result = BigInt::one();

    for _ in 0..n {
        result *= 10;
    }

    result
}

fn integer_sqrt(n: &BigInt) -> BigInt {
    if *n < BigInt::zero() {
        panic!("integer square root is undefined for negative integers");
    }

    if *n < BigInt::from(2) {
        return n.clone();
    }

    let two = BigInt::from(2);
    let mut x = n.clone();
    let mut y = (&x + n / &x) / &two;

    while y < x {
        x = y.clone();
        y = (&x + n / &x) / &two;
    }

    x
}

fn sqrt2_fixed(decimal_digits: usize) -> FixedDecimal {
    let scale = pow10(decimal_digits);
    let scaled_argument = BigInt::from(2) * &scale * &scale;
    let root = integer_sqrt(&scaled_argument);

    FixedDecimal::new(root, decimal_digits)
}

fn align_to_digits(x: &FixedDecimal, target_digits: usize) -> BigInt {
    if target_digits >= x.digits {
        &x.value * pow10(target_digits - x.digits)
    } else {
        &x.value / pow10(x.digits - target_digits)
    }
}

fn absolute_difference(a: &FixedDecimal, b: &FixedDecimal) -> BigInt {
    let target_digits = a.digits.max(b.digits);
    let a_scaled = align_to_digits(a, target_digits);
    let b_scaled = align_to_digits(b, target_digits);

    (a_scaled - b_scaled).abs()
}

fn scientific_ratio(numerator: &BigInt, denominator: &BigInt) -> String {
    if numerator.is_zero() {
        return "0.000000e+0".to_string();
    }

    let n_string = numerator.abs().to_string();
    let d_string = denominator.abs().to_string();

    let n_len = n_string.len() as i32;
    let d_len = d_string.len() as i32;

    let n_prefix_len = n_string.len().min(16);
    let d_prefix_len = d_string.len().min(16);

    let n_prefix: f64 = n_string[..n_prefix_len].parse::<f64>().unwrap();
    let d_prefix: f64 = d_string[..d_prefix_len].parse::<f64>().unwrap();

    let n_mantissa = n_prefix / 10_f64.powi(n_prefix_len as i32 - 1);
    let d_mantissa = d_prefix / 10_f64.powi(d_prefix_len as i32 - 1);

    let mut mantissa = n_mantissa / d_mantissa;
    let mut exponent = n_len - d_len;

    while mantissa < 1.0 {
        mantissa *= 10.0;
        exponent -= 1;
    }

    while mantissa >= 10.0 {
        mantissa /= 10.0;
        exponent += 1;
    }

    format!("{:.6}e{:+}", mantissa, exponent)
}

fn decimal_unit_roundoff(decimal_digits: usize) -> String {
    format!("5.0e-{}", decimal_digits)
}

fn main() {
    let precisions = vec![8, 16, 24, 32, 40, 48, 56, 64];
    let reference_digits = 90;

    let reference = sqrt2_fixed(reference_digits);
    let mut results = Vec::new();

    println!("Precision Ladder Diagnostics for Arbitrary-Precision Arithmetic");
    println!("===============================================================");
    println!();

    println!("Problem");
    println!("-------");
    println!("quantity evaluated              = sqrt(2)");
    println!("arithmetic model                = integer-scaled decimal fixed point");
    println!("reference decimal digits         = {}", reference_digits);
    println!(
        "reference value                  = {}",
        reference.rounded_display(70)
    );
    println!();

    println!("Precision Ladder");
    println!("----------------");
    println!(
        "{:>12} {:>14} {:>34} {:>22} {:>16}",
        "digits", "u_p", "computed value", "rel. error vs ref", "time(s)"
    );

    for &digits in &precisions {
        let start = Instant::now();
        let value = sqrt2_fixed(digits);
        let elapsed = start.elapsed().as_secs_f64();

        let diff = absolute_difference(&reference, &value);
        let denominator = reference.value.abs();
        let rel_error = scientific_ratio(&diff, &denominator);

        println!(
            "{:>12} {:>14} {:>34} {:>22} {:>16.6e}",
            digits,
            decimal_unit_roundoff(digits),
            value.rounded_display(28),
            rel_error,
            elapsed
        );

        results.push(value);
    }

    println!();
    println!("Successive Precision Differences");
    println!("--------------------------------");
    println!(
        "{:>12} {:>12} {:>24}",
        "p_low", "p_high", "Delta_{p,p'}"
    );

    for window in results.windows(2) {
        let low = &window[0];
        let high = &window[1];

        let diff = absolute_difference(high, low);
        let denominator = high.value.abs();
        let delta = scientific_ratio(&diff, &denominator);

        println!(
            "{:>12} {:>12} {:>24}",
            low.digits, high.digits, delta
        );
    }

    println!();
    println!("Interpretation");
    println!("--------------");
    println!("Increasing the decimal precision produces a sequence of approximations.");
    println!("The relative error against the high-precision reference decreases with precision.");
    println!("The successive differences form the precision-space convergence trace d_j.");
    println!("This demonstrates precision as a computational resource rather than a fixed hardware property.");
}
```

Program 22.8.1 demonstrates the central principle of arbitrary-precision arithmetic: precision should be viewed as a computational parameter rather than a fixed property of the hardware. By evaluating the same mathematical quantity across a sequence of increasing precision levels, the program provides direct evidence of how arithmetic error decreases as additional digits become available. This behavior corresponds to the precision-dependent error model discussed in Equation (22.8.6), where the arithmetic contribution to the total error decreases as the effective unit roundoff becomes smaller.

The precision ladder and successive-difference diagnostics provide a practical mechanism for assessing convergence in precision space. When the differences between successive precision levels decrease systematically, the computation is approaching a precision-independent result. Such behavior increases confidence that the observed approximation reflects the underlying mathematical problem rather than artifacts of finite arithmetic. Conversely, if the differences stagnate or grow, the diagnostics may indicate instability, cancellation, ill-conditioning, or the presence of other dominant error sources.

The timing measurements reported by the program also emphasize an important practical consideration. Although increased precision generally improves numerical accuracy, it does so at a computational cost. Arbitrary-precision arithmetic therefore requires balancing accuracy requirements against execution time, memory usage, and algorithmic complexity. The precision ladder provides a quantitative framework for making such decisions by revealing how much accuracy is gained from each additional increase in precision.

More broadly, the methodology demonstrated here forms the foundation for many modern multi-precision algorithms. High-precision reference computations, adaptive precision control, verified numerical algorithms, exact geometric predicates, and reliability-oriented scientific computing all rely on the same fundamental idea: increasing precision when diagnostics indicate that fixed-precision arithmetic is no longer sufficient. The techniques developed in this program therefore serve as a bridge between conventional floating-point computation and the more advanced arbitrary-precision methods explored in the remainder of the section.

## 22.8.2. Verification by High-Precision Reference Computation

One of the most important uses of arbitrary precision is verification. A high-precision computation can serve as a reference against which a lower-precision implementation is tested. Let $\widehat y_{64}$ denote a binary64 result and let,

$$\widehat y_{\mathrm{ref}}=\widehat y_{p_{\mathrm{high}}}\tag{22.8.12}$$

denote a high-precision result computed with $p_{\mathrm{high}}$ sufficiently large. The observed relative error is:

$$e_{\mathrm{obs}} = \frac{\|\widehat y_{64}-\widehat y_{\mathrm{ref}}\|}{\max(\|\widehat y_{\mathrm{ref}}\|,\sigma)} \tag{22.8.13}$$

This diagnostic is empirical, not absolute. It assumes that the reference computation is substantially more accurate than the tested computation. Therefore, the reference precision itself should be checked by increasing it further and verifying that,

$$\frac{\|\widehat y_{p_{\mathrm{high}}'}-\widehat y_{p_{\mathrm{high}}}\|}{\max(\|\widehat y_{p_{\mathrm{high}}'}\|,\sigma)}\ll e_{\mathrm{obs}},\qquad p_{\mathrm{high}}'>p_{\mathrm{high}}.\tag{22.8.14}$$

If equation (22.8.14) fails, the reference value is not yet reliable.

For root-finding problems, arbitrary precision can be used to verify residuals and sensitivity. Suppose $f(x^\ast)=0$ and a fixed-precision computation returns $\widehat x$. The residual is:

$$r=f(\widehat x)\tag{22.8.15}$$

If $f'(\widehat x)\ne 0$, a first-order estimate of the forward error is:

$$\widehat x-x^\ast\approx-\frac{f(\widehat x)}{f'(\widehat x)} \tag{22.8.16}$$

Computing both $f(\widehat x)$ and $f'(\widehat x)$ in high precision avoids mistaking residual cancellation for accuracy. This is especially important when $f(\widehat x)$ is the difference of nearly equal quantities.

For linear systems $Ax=b$, a high-precision residual,

$$r_{\mathrm{hp}}=b-A\widehat x\tag{22.8.17}$$

can reveal whether a low-precision solution actually satisfies the equations. If $r_{\mathrm{hp}}$ is computed in the same low precision as $\widehat x$, cancellation in (A\\widehat x) may hide the true residual. The backward error,

$$\eta = \frac{\|r_{\mathrm{hp}}\|}{\|A\|\|\widehat x\|+\|b\|}\tag{22.8.18}$$

is then a scale-aware measure of how much the data would need to be perturbed for $\widehat x$ to be an exact solution. A small $\eta$ gives evidence of backward stability, while a large $\eta$ signals algorithmic or arithmetic failure.

For eigenvalue problems,

$$Ax=\lambda x \tag{22.8.19}$$

the high-precision residual,

$$r_{\mathrm{hp}}=A\widehat x-\widehat\lambda \widehat x\tag{20.8.20}$$

provides a similar diagnostic. When matrices are nonnormal or eigenvalues are clustered, small perturbations can cause large changes in eigenvectors or eigenvalues. Arbitrary precision helps separate actual spectral sensitivity from errors caused by inadequate arithmetic. This is one reason high-precision computation remains important in difficult stability and eigenvalue calculations (Dondl et al., 2025).

A high-precision reference can also validate implementations of special functions, quadrature rules, interpolation formulas, and recurrence relations. Suppose an algorithm computes

$$I=\int_a^b f(x)\,dx \tag{22.8.21}$$

A lower-precision quadrature result $I_{64}$ may be checked against a high-precision result $I_{\mathrm{hp}}$:

$$e_I = \frac{|I_{64}-I_{\mathrm{hp}}|}{\max(|I_{\mathrm{hp}}|,\sigma)} \tag{22.8.22}$$

If $I_{\mathrm{hp}}$ is obtained by a different algorithm as well as higher precision, the comparison is stronger because it tests both arithmetic and method-specific implementation assumptions. Recent high-precision quadrature software illustrates this role of arbitrary precision as a dependability mechanism for difficult integration problems (Vretinaris, 2026).

### Rust Implementation

Following the discussion in Section 22.8.2 on verification by high-precision reference computation, Program 22.8.2 provides a practical framework for validating numerical results against independently computed high-precision references. In scientific computing, it is often difficult to determine whether an apparently accurate result is genuinely correct or merely the consequence of coincidental cancellation, insufficient testing, or hidden arithmetic limitations. High-precision reference computations provide an effective mechanism for addressing this problem by establishing a numerical benchmark against which lower-precision implementations can be evaluated. This program demonstrates the verification process using the computation of $\sqrt{2}$, comparing a binary64 approximation against high-precision fixed-point references while simultaneously validating the reliability of the reference itself through additional precision refinement. The resulting framework illustrates the verification methodology embodied in Equations (22.8.12)–(22.8.14) and demonstrates how arbitrary precision can be used as a practical dependability tool in numerical software development.

At the core of the implementation is the `FixedDecimal` structure, which provides an arbitrary-precision decimal representation using a large integer together with an explicit decimal scale. This design allows computations to be performed independently of hardware floating-point formats and provides direct control over the number of decimal digits used in the reference calculations. The structure serves as the computational foundation for constructing the high-precision reference values required by Equation (22.8.12).

The methods `to_decimal_string` and `rounded_display` convert arbitrary-precision values into human-readable decimal representations. Since one objective of the verification process is to compare the binary64 result with increasingly accurate reference values, these routines provide a convenient mechanism for inspecting the numerical agreement between different computations. The ability to display many digits also makes it possible to observe the stability of the reference computation when the precision is increased further, as required by Equation (22.8.14).

The `pow10` function constructs powers of ten that define the scaling factors associated with the fixed-point representation. These scaling factors allow decimal arithmetic to be emulated exactly using integer operations. As the number of decimal digits increases, the scaling factor grows correspondingly, enabling the computation of increasingly accurate reference values.

The `integer_sqrt` function implements Newton iteration for computing the integer square root of a nonnegative integer. This routine forms the computational kernel of the high-precision reference calculation. Unlike hardware square-root instructions operating on binary64 values, the algorithm works entirely with large integers and therefore supports precision levels far beyond standard floating-point arithmetic. By computing square roots in this manner, the program obtains reference values that are substantially more accurate than the binary64 approximation under investigation.

The `sqrt2_fixed` function constructs a fixed-point approximation to $\sqrt{2}$ at a specified decimal precision. It first scales the argument appropriately, computes the integer square root, and then returns the result as a `FixedDecimal` object. Evaluating this function at different precision levels produces the sequence of reference computations required to test the stability criterion of Equation (22.8.14). The resulting values play the role of $\widehat y_{p_{\mathrm{high}}}$ and $\widehat y_{p_{\mathrm{high}}'}$ in the theoretical development.

The functions `align_to_digits` and `absolute_difference` support comparison of quantities computed at different precisions. Because each fixed-point number may use a different decimal scale, meaningful comparison requires that all values first be represented using a common scale. These routines perform that normalization and subsequently compute absolute differences between approximations. Such differences form the basis for evaluating the observed relative error of Equation (22.8.13) and the reference-stability diagnostic of Equation (22.8.14).

The `f64_to_fixed_nonnegative` function converts a binary64 value into the fixed-point representation used throughout the program. This conversion allows the binary64 approximation and the arbitrary-precision reference values to be compared within a common arithmetic framework. As a result, differences between the computations can be measured directly without introducing additional conversion-related ambiguities.

The `scientific_ratio` function evaluates normalized error measures and formats them in scientific notation. This routine provides the computational realization of the relative-error diagnostics developed in Equations (22.8.13) and (22.8.14). By expressing both the observed binary64 error and the reference-stability error in normalized form, the implementation enables meaningful comparison across different scales and magnitudes.

The `reference_is_much_smaller` function evaluates whether the stability error associated with the reference computation is significantly smaller than the observed error of the tested implementation. This directly corresponds to the requirement expressed by Equation (22.8.14). If the higher-precision refinement changes the reference value by an amount that is negligible relative to the observed binary64 error, then the reference may reasonably be treated as reliable for verification purposes.

The `main` function integrates all stages of the verification workflow. It begins by computing a binary64 approximation of $\sqrt{2}$, then constructs two arbitrary-precision references at different decimal precisions. The first reference serves as the benchmark against which the binary64 result is tested, while the second reference is used to verify that the benchmark itself has converged. The program subsequently computes the observed relative error, evaluates the reference-stability diagnostic, and reports whether the reference satisfies the reliability criterion implied by Equation (22.8.14). Together, these computations demonstrate how arbitrary precision can be used to validate lower-precision implementations and establish confidence in numerical results.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-bigint = "0.4"
num-traits = "0.2"
```

```rust
// Program 22.8.2. High-Precision Reference Verification with Pure Rust Big Integers

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

#[derive(Clone, Debug)]
struct FixedDecimal {
    value: BigInt,
    digits: usize,
}

impl FixedDecimal {
    fn new(value: BigInt, digits: usize) -> Self {
        Self { value, digits }
    }

    fn to_decimal_string(&self) -> String {
        let negative = self.value.is_negative();
        let mut s = self.value.abs().to_string();

        if s.len() <= self.digits {
            let zeros_needed = self.digits + 1 - s.len();
            s = format!("{}{}", "0".repeat(zeros_needed), s);
        }

        let split = s.len() - self.digits;
        let integer_part = &s[..split];
        let fractional_part = &s[split..];

        if negative {
            format!("-{}.{}", integer_part, fractional_part)
        } else {
            format!("{}.{}", integer_part, fractional_part)
        }
    }

    fn rounded_display(&self, max_fraction_digits: usize) -> String {
        let full = self.to_decimal_string();

        if let Some(dot_position) = full.find('.') {
            let keep = dot_position + 1 + max_fraction_digits.min(self.digits);
            full[..keep.min(full.len())].to_string()
        } else {
            full
        }
    }
}

fn pow10(n: usize) -> BigInt {
    let mut result = BigInt::one();

    for _ in 0..n {
        result *= 10;
    }

    result
}

fn integer_sqrt(n: &BigInt) -> BigInt {
    if *n < BigInt::zero() {
        panic!("integer square root is undefined for negative integers");
    }

    if *n < BigInt::from(2) {
        return n.clone();
    }

    let two = BigInt::from(2);
    let mut x = n.clone();
    let mut y = (&x + n / &x) / &two;

    while y < x {
        x = y.clone();
        y = (&x + n / &x) / &two;
    }

    x
}

fn sqrt2_fixed(decimal_digits: usize) -> FixedDecimal {
    let scale = pow10(decimal_digits);
    let scaled_argument = BigInt::from(2) * &scale * &scale;
    let root = integer_sqrt(&scaled_argument);

    FixedDecimal::new(root, decimal_digits)
}

fn align_to_digits(x: &FixedDecimal, target_digits: usize) -> BigInt {
    if target_digits >= x.digits {
        &x.value * pow10(target_digits - x.digits)
    } else {
        &x.value / pow10(x.digits - target_digits)
    }
}

fn absolute_difference(a: &FixedDecimal, b: &FixedDecimal) -> BigInt {
    let target_digits = a.digits.max(b.digits);
    let a_scaled = align_to_digits(a, target_digits);
    let b_scaled = align_to_digits(b, target_digits);

    (a_scaled - b_scaled).abs()
}

fn f64_to_fixed_nonnegative(x: f64, digits: usize) -> FixedDecimal {
    assert!(x >= 0.0);

    let formatted = format!("{:.*}", digits, x);
    let mut parts = formatted.split('.');

    let integer_part = parts.next().unwrap_or("0");
    let fractional_part = parts.next().unwrap_or("");

    let mut fraction = fractional_part.to_string();

    if fraction.len() < digits {
        fraction.push_str(&"0".repeat(digits - fraction.len()));
    }

    if fraction.len() > digits {
        fraction.truncate(digits);
    }

    let combined = format!("{}{}", integer_part, fraction);
    let value = BigInt::parse_bytes(combined.as_bytes(), 10)
        .expect("failed to parse fixed decimal value");

    FixedDecimal::new(value, digits)
}

fn scientific_ratio(numerator: &BigInt, denominator: &BigInt) -> String {
    if numerator.is_zero() {
        return "0.000000e+0".to_string();
    }

    let n_string = numerator.abs().to_string();
    let d_string = denominator.abs().to_string();

    let n_prefix_len = n_string.len().min(16);
    let d_prefix_len = d_string.len().min(16);

    let n_prefix: f64 = n_string[..n_prefix_len].parse::<f64>().unwrap();
    let d_prefix: f64 = d_string[..d_prefix_len].parse::<f64>().unwrap();

    let mut mantissa =
        (n_prefix / 10_f64.powi(n_prefix_len as i32 - 1))
            / (d_prefix / 10_f64.powi(d_prefix_len as i32 - 1));

    let mut exponent = n_string.len() as i32 - d_string.len() as i32;

    while mantissa < 1.0 {
        mantissa *= 10.0;
        exponent -= 1;
    }

    while mantissa >= 10.0 {
        mantissa /= 10.0;
        exponent += 1;
    }

    format!("{:.6}e{:+}", mantissa, exponent)
}

fn reference_is_much_smaller(reference_check: &BigInt, observed_error: &BigInt) -> bool {
    reference_check * BigInt::from(1000) < *observed_error
}

fn main() {
    let reference_digits = 80;
    let higher_reference_digits = 110;

    let y64 = 2.0_f64.sqrt();

    let y64_fixed = f64_to_fixed_nonnegative(y64, reference_digits);
    let y_ref = sqrt2_fixed(reference_digits);
    let y_ref_high = sqrt2_fixed(higher_reference_digits);

    let observed_error_numerator = absolute_difference(&y64_fixed, &y_ref);
    let reference_check_numerator = absolute_difference(&y_ref, &y_ref_high);

    let observed_error = scientific_ratio(
        &observed_error_numerator,
        &y_ref.value.abs(),
    );

    let reference_check = scientific_ratio(
        &reference_check_numerator,
        &y_ref_high.value.abs(),
    );

    let reference_verified =
        reference_is_much_smaller(&reference_check_numerator, &observed_error_numerator);

    println!("High-Precision Reference Verification");
    println!("=====================================");
    println!();

    println!("Problem");
    println!("-------");
    println!("quantity tested                  = sqrt(2)");
    println!("binary64 computation y_64         = {:.17e}", y64);
    println!(
        "reference precision               = {} decimal digits",
        reference_digits
    );
    println!(
        "higher reference precision        = {} decimal digits",
        higher_reference_digits
    );

    println!();
    println!("Reference Values");
    println!("----------------");
    println!(
        "y_64 as fixed decimal              = {}",
        y64_fixed.rounded_display(70)
    );
    println!(
        "y_ref                              = {}",
        y_ref.rounded_display(70)
    );
    println!(
        "y_ref_high                         = {}",
        y_ref_high.rounded_display(70)
    );

    println!();
    println!("Verification Diagnostics");
    println!("------------------------");
    println!("observed relative error e_obs      = {}", observed_error);
    println!("reference stability check          = {}", reference_check);
    println!(
        "reference check << e_obs           = {}",
        reference_verified
    );

    println!();
    println!("Interpretation");
    println!("--------------");
    println!("The binary64 result is compared against a high-precision reference.");
    println!("The reference is recomputed at a still higher precision to test stability.");
    println!("A much smaller reference check supports the reliability of the reference value.");
    println!("This implements the verification logic of Equations (22.8.12)--(22.8.14).");
}
```

Program 22.8.2 demonstrates one of the most important practical applications of arbitrary-precision arithmetic: numerical verification. Rather than relying solely on theoretical error bounds or assumptions regarding algorithmic behavior, the program evaluates a low-precision result against an independently computed high-precision reference. This approach provides direct empirical evidence regarding the accuracy of the implementation and reflects the verification methodology introduced in Equations (22.8.12)–(22.8.14).

The example also illustrates an important principle of numerical validation: a reference value must itself be verified. Simply computing a quantity at a higher precision does not automatically guarantee correctness. By recomputing the reference at an even higher precision and comparing the two results, the program establishes that the reference error is substantially smaller than the observed binary64 error. This additional verification step greatly increases confidence in the validity of the comparison.

The observed relative error provides a quantitative assessment of the accuracy of the binary64 computation, while the reference-stability diagnostic measures the trustworthiness of the benchmark itself. When the stability error is orders of magnitude smaller than the observed error, the reference value can be regarded as sufficiently accurate for verification purposes. This distinction between the accuracy of the tested computation and the reliability of the reference is fundamental to dependable numerical analysis.

More broadly, the verification framework demonstrated here extends naturally to a wide range of computational problems. Root-finding algorithms, linear systems, eigenvalue computations, quadrature methods, interpolation procedures, recurrence relations, and special-function evaluations can all be validated using the same general strategy. In each case, arbitrary-precision arithmetic serves as a mechanism for separating genuine algorithmic behavior from artifacts introduced by finite-precision computation. Consequently, high-precision reference computation remains one of the most powerful tools available for the verification and certification of scientific software.

## 22.8.3. Cancellation, Ill-Conditioning, and Adaptive Precision

Arbitrary precision is especially valuable when ordinary precision is destroyed by cancellation. Consider,

$$z=x-y,\qquad x\approx y \tag{22.8.23}$$

If $x$ and $y$ are known with relative errors of order $u$, then the relative error in $z$ can be approximately,

$$\frac{|x|\cdot u+|y|\cdot u}{|x-y|} \tag{22.8.24}$$

The amplification factor,

$$\chi_{\mathrm{cancel}} = \frac{|x|+|y|}{|x-y|}\tag{22.8.25}$$

measures the severity of cancellation. When $\chi_{\mathrm{cancel}}$ is large, many leading digits are lost. If the desired relative accuracy is $\tau$, a rough precision requirement is:

$$u_p \chi_{\mathrm{cancel}} \lesssim \tau \tag{22.8.26}$$

Using equation (22.8.5), this becomes:

$$\frac{1}{2}\beta^{1-p}\chi_{\mathrm{cancel}}\lesssim \tau \tag{22.8.27}$$

Solving approximately for $p$,

$$p\gtrsim1+\log_\beta\left(\frac{\chi_{\mathrm{cancel}}}{2\tau}\right) \tag{22.8.28}$$

This formula shows how precision can be selected from a local numerical diagnostic rather than chosen arbitrarily.

Ill-conditioned functions require a similar analysis. The relative condition number of a scalar function $f$ at $x$ is,

$$\kappa_f(x) = \left|\frac{x f'(x)}{f(x)}\right| \tag{22.8.29}$$

when $x\ne 0$ and $f(x)\ne 0$. A relative perturbation $\Delta x/x$ produces, to first order,

$$\frac{\Delta f}{f}\approx\kappa_f(x)\frac{\Delta x}{x} \tag{22.8.30}$$

If input or rounding errors are of size $u_p$, then a rough expected relative output error is:

$$\frac{|\Delta f|}{|f|}\approx\kappa_f(x)u_p\tag{22.8.31}$$

To achieve tolerance $\tau$, one needs,

$$\kappa_f(x)u_p\lesssim \tau \tag{22.8.32}$$

Thus,

$$p\gtrsim1+\log_\beta\left(\frac{\kappa_f(x)}{2\tau}\right)\tag{22.8.33}$$

under rounding to nearest. This expression is only a first-order guide, but it captures the main principle: required precision grows logarithmically with the condition number and inversely with the requested tolerance.

For vector-valued problems,

$$y=F(x) \tag{22.8.34}$$

a normwise condition estimate is:

$$\kappa_F(x) = \frac{\|J_F(x)\|\|x\|}{\|F(x)\|} \tag{22.8.35}$$

where $J_F(x)$ is the Jacobian. The expected relative perturbation satisfies,

$$\frac{\|\Delta y\|}{\|y\|}\lesssim\kappa_F(x)\frac{\|\Delta x\|}{\|x\|} \tag{22.8.36}$$

If the perturbation is dominated by arithmetic error, an adaptive-precision strategy can increase $p$ until the estimated error is below tolerance.

A generic adaptive-precision loop may be described mathematically as follows. Begin with precision $p_0$, compute $\widehat y_{p_0}$, then increase precision according to:

$$p_{j+1}=p_j+\Delta p_j \tag{22.8.37}$$

Stop when,

$$\frac{\|\widehat y_{p_{j+1}}-\widehat y_{p_j}\|}{\max(\|\widehat y_{p_{j+1}}\|,\sigma)}\le \tau \tag{22.8.38}$$

This criterion is simple, but it must be used carefully. The difference between two successive precisions can be small for the wrong reason, for example because both computations stagnate at the same inaccurate value. A stronger test also checks a residual, invariant, or backward error:

$$\eta(\widehat y_{p_{j+1}})\le \tau_{\eta} \tag{22.8.39}$$

Thus, a robust adaptive-precision stopping rule combines agreement across precisions with a problem-specific correctness diagnostic.

Mixed-precision algorithms can be interpreted as a structured version of adaptive precision. Some operations are cheap in low precision, while sensitive parts are promoted to higher precision. If a computation is decomposed as:

$$\widehat y = G_{\mathrm{high}}\bigl(G_{\mathrm{low}}(x)\bigr) \tag{22.8.40}$$

then the design question is which components require high precision. For example, residual evaluation, correction steps, orthogonalization, and final validation are often more precision-sensitive than preliminary approximations. Recent work on mixed-precision numerics and extended-precision algorithms supports this viewpoint, treating precision allocation as an algorithmic design problem rather than a fixed hardware choice (Kashi et al., 2026; Zhang and Aiken, 2025).

### Rust Implementation

Following the discussion in Section 22.8.3 on cancellation, ill-conditioning, and adaptive precision, Program 22.8.3 provides a practical implementation of precision selection driven by numerical diagnostics rather than fixed arithmetic assumptions. In finite-precision computation, subtraction of nearly equal quantities can amplify rounding errors dramatically, causing the computed result to lose many significant digits even when the original quantities are represented accurately. This phenomenon, known as catastrophic cancellation, is one of the principal motivations for arbitrary-precision arithmetic. The program demonstrates how a cancellation diagnostic can be used to estimate the precision required for a target accuracy and how an adaptive-precision strategy can subsequently verify that the selected precision is sufficient. By combining precision-ladder comparisons with a residual-style correctness check, the implementation illustrates the practical interpretation of Equations (22.8.23)–(22.8.39) and shows how precision can be treated as an algorithmic resource rather than a fixed hardware constraint.

At the core of the implementation is the `FixedDecimal` structure, which provides a fixed-point arbitrary-precision representation based on large integers and explicit decimal scaling. Instead of relying on hardware floating-point formats, each value is stored as an integer together with the number of decimal digits used for scaling. This design makes it possible to emulate a family of arithmetic systems with different precisions and thereby investigate the precision-dependent behavior described by Equations (22.8.26)–(22.8.38).

The methods `to_decimal_string` and `rounded_display` are responsible for converting arbitrary-precision values into readable decimal form. These routines allow the program to display intermediate and final results at different precision levels while preserving the underlying numerical accuracy. Since the objective of the program is to compare computations across a precision ladder, clear presentation of the computed values is essential for interpreting the adaptive-precision process.

The `pow10` function generates powers of ten used to define decimal scaling factors. These scaling factors determine the effective arithmetic precision used by each computation. By increasing the number of decimal digits, the program emulates the precision increments described by Equation (22.8.37), allowing the adaptive algorithm to move progressively toward a more accurate result.

The functions `align_to_digits` and `quantize_to_digits` provide the mechanisms for converting quantities between different precision levels. The first aligns two arbitrary-precision values to a common scale so that meaningful arithmetic operations can be performed, while the second deliberately reduces a high-precision value to a specified working precision. This controlled reduction of precision allows the program to simulate the loss of significant digits that occurs when cancellation is encountered in finite arithmetic.

The `subtract` function performs arbitrary-precision subtraction after aligning both operands to a common scale. This operation represents the computation of $z=x-y$ discussed in Equation (22.8.23). Because the chosen example involves nearly equal quantities, the subtraction is highly sensitive to the number of digits retained in the operands, making it an ideal demonstration of cancellation effects.

The `absolute_difference` function computes differences between approximations generated at different precision levels. These differences form the basis of the agreement test appearing in Equation (22.8.38). By monitoring how successive approximations change as precision increases, the program determines whether the computation is converging toward a precision-independent result.

The `scientific_ratio` function evaluates normalized error measures and presents them in scientific notation. This routine is used both for reporting agreement between successive precision levels and for evaluating the residual-style diagnostic associated with Equation (22.8.39). Expressing these quantities in normalized form allows the program to compare errors across different scales and precision levels.

The `ratio_is_below_tolerance` function implements the adaptive stopping criterion. Given a numerator, denominator, and tolerance exponent, the function determines whether the normalized error satisfies the prescribed accuracy requirement. This corresponds directly to the adaptive-precision convergence criterion of Equation (22.8.38), which requires successive computations to agree within a specified tolerance before the precision is accepted.

The functions `cancellation_amplification_as_f64` and `suggested_decimal_precision` implement the cancellation analysis developed in Equations (22.8.25)–(22.8.28). The first estimates the cancellation amplification factor $\chi_{\mathrm{cancel}}$, while the second converts this estimate into a recommended decimal precision for a specified tolerance. These routines demonstrate how local conditioning information can be used to guide precision selection before the adaptive loop begins.

The `main` function coordinates the entire adaptive-precision experiment. It first constructs a cancellation-sensitive subtraction problem in which $x$ and $y$ differ only in the fortieth decimal place. The cancellation amplification factor is then estimated and used to predict the required precision. The program subsequently evaluates the subtraction using a sequence of increasing precisions, producing a precision ladder analogous to that described by Equations (22.8.37) and (22.8.38). At each level, both an agreement diagnostic and a residual-style correctness diagnostic are evaluated. Only when both tests are satisfied is the precision accepted. This dual verification process illustrates the principle that convergence between successive precisions alone is not sufficient; a problem-specific correctness measure must also be satisfied to ensure that the result is genuinely accurate.

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-bigint = "0.4"
num-traits = "0.2"
```

```rust
// Program 22.8.3. Adaptive Precision for Cancellation-Sensitive Subtraction
//
// Problem statement:
// Demonstrate how cancellation diagnostics can guide precision selection.
// The program considers z = x - y with x approximately equal to y, estimates
// the cancellation amplification factor, predicts a required decimal precision,
// and then increases precision adaptively until both successive agreement and
// a residual-style diagnostic are below the requested tolerance.

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

#[derive(Clone, Debug)]
struct FixedDecimal {
    value: BigInt,
    digits: usize,
}

impl FixedDecimal {
    fn new(value: BigInt, digits: usize) -> Self {
        Self { value, digits }
    }

    
    fn to_decimal_string(&self) -> String {
        let negative = self.value.is_negative();
        let mut s = self.value.abs().to_string();

        if self.digits == 0 {
            return if negative { format!("-{}", s) } else { s };
        }

        if s.len() <= self.digits {
            let zeros_needed = self.digits + 1 - s.len();
            s = format!("{}{}", "0".repeat(zeros_needed), s);
        }

        let split = s.len() - self.digits;
        let integer_part = &s[..split];
        let fractional_part = &s[split..];

        if negative {
            format!("-{}.{}", integer_part, fractional_part)
        } else {
            format!("{}.{}", integer_part, fractional_part)
        }
    }

    fn rounded_display(&self, max_fraction_digits: usize) -> String {
        let full = self.to_decimal_string();

        if let Some(dot_position) = full.find('.') {
            let keep = dot_position + 1 + max_fraction_digits.min(self.digits);
            full[..keep.min(full.len())].to_string()
        } else {
            full
        }
    }
}

fn pow10(n: usize) -> BigInt {
    let mut result = BigInt::one();

    for _ in 0..n {
        result *= 10;
    }

    result
}

fn align_to_digits(x: &FixedDecimal, target_digits: usize) -> BigInt {
    if target_digits >= x.digits {
        &x.value * pow10(target_digits - x.digits)
    } else {
        &x.value / pow10(x.digits - target_digits)
    }
}

fn subtract(a: &FixedDecimal, b: &FixedDecimal, output_digits: usize) -> FixedDecimal {
    let a_scaled = align_to_digits(a, output_digits);
    let b_scaled = align_to_digits(b, output_digits);

    FixedDecimal::new(a_scaled - b_scaled, output_digits)
}

fn quantize_to_digits(x: &FixedDecimal, work_digits: usize) -> FixedDecimal {
    if work_digits >= x.digits {
        FixedDecimal::new(
            &x.value * pow10(work_digits - x.digits),
            work_digits,
        )
    } else {
        FixedDecimal::new(
            &x.value / pow10(x.digits - work_digits),
            work_digits,
        )
    }
}

fn absolute_difference(a: &FixedDecimal, b: &FixedDecimal) -> BigInt {
    let target_digits = a.digits.max(b.digits);
    let a_scaled = align_to_digits(a, target_digits);
    let b_scaled = align_to_digits(b, target_digits);

    (a_scaled - b_scaled).abs()
}

fn scientific_ratio(numerator: &BigInt, denominator: &BigInt) -> String {
    if numerator.is_zero() {
        return "0.000000e+0".to_string();
    }

    let n_string = numerator.abs().to_string();
    let d_string = denominator.abs().to_string();

    let n_prefix_len = n_string.len().min(16);
    let d_prefix_len = d_string.len().min(16);

    let n_prefix: f64 = n_string[..n_prefix_len].parse::<f64>().unwrap();
    let d_prefix: f64 = d_string[..d_prefix_len].parse::<f64>().unwrap();

    let mut mantissa =
        (n_prefix / 10_f64.powi(n_prefix_len as i32 - 1))
            / (d_prefix / 10_f64.powi(d_prefix_len as i32 - 1));

    let mut exponent = n_string.len() as i32 - d_string.len() as i32;

    while mantissa < 1.0 {
        mantissa *= 10.0;
        exponent -= 1;
    }

    while mantissa >= 10.0 {
        mantissa /= 10.0;
        exponent += 1;
    }

    format!("{:.6}e{:+}", mantissa, exponent)
}

fn ratio_is_below_tolerance(numerator: &BigInt, denominator: &BigInt, tol_exponent: usize) -> bool {
    // Tests numerator / denominator <= 10^(-tol_exponent)
    numerator * pow10(tol_exponent) <= *denominator
}

fn cancellation_amplification_as_f64(gap_digits: usize) -> f64 {
    // For x = 1 + 10^(-gap_digits) and y = 1,
    // chi_cancel is approximately 2 / 10^(-gap_digits).
    2.0 * 10_f64.powi(gap_digits as i32)
}

fn suggested_decimal_precision(chi_cancel: f64, tolerance: f64) -> usize {
    // Decimal analogue of p >= 1 + log_beta(chi / (2 tau)).
    let estimate = 1.0 + (chi_cancel / (2.0 * tolerance)).log10();
    estimate.ceil() as usize
}

fn main() {
    let exact_digits = 90;
    let gap_digits = 40;
    let tolerance_exponent = 12;
    let tolerance = 10_f64.powi(-(tolerance_exponent as i32));

    let exact_scale = pow10(exact_digits);
    let gap_units = pow10(exact_digits - gap_digits);

    let y_exact = FixedDecimal::new(exact_scale.clone(), exact_digits);
    let x_exact = FixedDecimal::new(&exact_scale + &gap_units, exact_digits);
    let z_exact = subtract(&x_exact, &y_exact, exact_digits);

    let chi_cancel = cancellation_amplification_as_f64(gap_digits);
    let suggested_p = suggested_decimal_precision(chi_cancel, tolerance);

    let precision_ladder = vec![12, 20, 28, 36, 40, 48, 56, 64];

    let mut previous_z: Option<FixedDecimal> = None;
    let mut selected: Option<(usize, FixedDecimal, String, String)> = None;

    println!("Adaptive Precision for Cancellation-Sensitive Subtraction");
    println!("=========================================================");
    println!();

    println!("Problem");
    println!("-------");
    println!("operation                       = z = x - y");
    println!("x                               = {}", x_exact.rounded_display(50));
    println!("y                               = {}", y_exact.rounded_display(50));
    println!("exact z                         = {}", z_exact.rounded_display(50));
    println!("gap size                        = 10^(-{})", gap_digits);
    println!("target tolerance                = 1.0e-{}", tolerance_exponent);
    println!();

    println!("Cancellation Diagnostic");
    println!("-----------------------");
    println!("chi_cancel                     ≈ {:.6e}", chi_cancel);
    println!(
        "suggested decimal precision    ≈ {} digits",
        suggested_p
    );
    println!();

    println!("Adaptive Precision Ladder");
    println!("-------------------------");
    println!(
        "{:>10} {:>28} {:>22} {:>22} {:>12}",
        "digits", "computed z", "agreement", "residual eta", "accepted"
    );

    for digits in precision_ladder {
        let x_work = quantize_to_digits(&x_exact, digits);
        let y_work = quantize_to_digits(&y_exact, digits);
        let z_work = subtract(&x_work, &y_work, digits);

        let agreement_numerator = if let Some(prev) = &previous_z {
            absolute_difference(&z_work, prev)
        } else {
            BigInt::from(1)
        };

        let z_work_denominator = align_to_digits(&z_work, z_work.digits).abs();

        let agreement_denominator = if z_work_denominator.is_zero() {
            BigInt::one()
        } else {
            z_work_denominator
        };

        let residual_numerator = absolute_difference(&z_work, &z_exact);
        let residual_denominator = z_exact.value.abs().max(BigInt::one());

        let agreement = if previous_z.is_some() {
            scientific_ratio(&agreement_numerator, &agreement_denominator)
        } else {
            "not available".to_string()
        };

        let residual_eta = scientific_ratio(&residual_numerator, &residual_denominator);

        let agreement_ok = previous_z.is_some()
            && ratio_is_below_tolerance(
                &agreement_numerator,
                &agreement_denominator,
                tolerance_exponent,
            );

        let residual_ok = ratio_is_below_tolerance(
            &residual_numerator,
            &residual_denominator,
            tolerance_exponent,
        );

        let accepted = agreement_ok && residual_ok;

        println!(
            "{:>10} {:>28} {:>22} {:>22} {:>12}",
            digits,
            z_work.rounded_display(18),
            agreement,
            residual_eta,
            accepted
        );

        if accepted && selected.is_none() {
            selected = Some((digits, z_work.clone(), agreement, residual_eta));
        }

        previous_z = Some(z_work);
    }

    println!();
    println!("Selected Precision");
    println!("------------------");

    if let Some((digits, z_value, agreement, residual_eta)) = selected {
        println!("selected decimal digits         = {}", digits);
        println!("accepted z                      = {}", z_value.rounded_display(50));
        println!("successive agreement            = {}", agreement);
        println!("residual eta                    = {}", residual_eta);
    } else {
        println!("no precision level satisfied both stopping tests");
    }

    println!();
    println!("Interpretation");
    println!("--------------");
    println!("Low precision rounds x and y to the same value, so the computed difference vanishes.");
    println!("Agreement alone can be misleading because two low-precision results may agree at zero.");
    println!("The residual-style diagnostic prevents premature acceptance.");
    println!("The selected precision satisfies both successive agreement and correctness checks.");
}
```

Program 22.8.3 demonstrates how cancellation can destroy the effectiveness of ordinary finite-precision arithmetic even when the individual operands are represented accurately. In the example considered here, the difference $x-y$ is many orders of magnitude smaller than either operand. As a result, low-precision computations lose the information contained in the subtraction and incorrectly produce a zero result. This behavior illustrates the amplification mechanism quantified by the cancellation factor $\chi_{\mathrm{cancel}}$ in Equation (22.8.25).

The adaptive-precision framework provides a practical solution to this problem. Rather than selecting a precision arbitrarily, the program first estimates the severity of cancellation and then uses a precision ladder to determine whether the computed result has stabilized. This reflects the theoretical development of Equations (22.8.26)–(22.8.38), where precision is increased until the estimated arithmetic error falls below a prescribed tolerance.

An important feature of the implementation is the use of two independent acceptance criteria. Agreement between successive precision levels alone can be misleading because multiple inaccurate computations may converge to the same incorrect value. The residual-style diagnostic therefore serves as an additional correctness check, ensuring that the accepted result is consistent with the underlying mathematical problem. This approach reflects the stronger stopping rule advocated in Equation (22.8.39).

More broadly, the program illustrates the fundamental philosophy of modern adaptive and mixed-precision computation. Precision is not treated as a fixed property of the machine but as a resource that can be allocated dynamically according to the numerical sensitivity of the problem. The same principles extend naturally to ill-conditioned functions, nonlinear systems, optimization problems, eigenvalue computations, and large-scale scientific simulations. Consequently, adaptive precision provides a powerful mechanism for balancing accuracy, reliability, and computational cost in advanced numerical algorithms.

## 22.8.4. Exact Kernels, Robust Predicates, and Chapter Summary

Some computations require not merely more precision, but correct decisions. This occurs especially in geometry, combinatorics, mesh generation, computational topology, and branch-based algorithms. A geometric predicate may determine the sign of a determinant. For example, the orientation of three points in the plane,

$$p_1=(x_1,y_1),\qquad p_2=(x_2,y_2),\qquad p_3=(x_3,y_3) \tag{22.8.41}$$

is determined by:

$$\Delta = \det\begin{bmatrix}x_2-x_1 & y_2-y_1\\x_3-x_1 & y_3-y_1\end{bmatrix}\tag{22.8.42}$$

If $\Delta>0,$ the points are oriented counterclockwise. If $\Delta<0$, they are oriented clockwise. If $\Delta=0$, they are collinear. Near collinearity, the computed determinant may have the wrong sign in fixed precision. A wrong sign is not a small numerical error in the usual sense. It can send a geometric algorithm down the wrong branch and produce a topologically invalid result.

The same issue appears in the in-circle test, mesh intersection tests, constructive solid geometry, and exact combinatorial decisions. A robust predicate must return the correct sign even when the determinant is small. One strategy is adaptive precision: compute a fast approximate value first, estimate an error bound, and increase precision only if the sign is uncertain. Abstractly, if an approximate predicate value is $\widehat \Delta$ and an error bound is $E$, then the sign is certified when $|\widehat \Delta|>E$. If $|\widehat \Delta|\le E,$ the computation must be refined. This is an example of precision being used selectively to guarantee a discrete decision. Recent work on exact predicates, exact constructions, and mesh constructive solid geometry shows that such exact or near-exact kernels remain essential in modern geometric computation (Lévy, 2025).

Arbitrary precision also connects naturally to interval and ball arithmetic. Instead of computing a single approximation, one computes an enclosure

$$F(x)\in [\underline y,\overline y] \tag{22.8.43}$$

or, in multidimensional form,

$$F(x)\in B(c,r)=\{y:\|y-c\|\le r\} \tag{22.8.44}$$

If the radius $r$ is small enough, the result is certified to the requested accuracy. For a scalar result, a sufficient condition for absolute accuracy $\tau$ is:

$$\overline y-\underline y\le 2\tau \tag{22.8.45}$$

For a sign decision, certification is possible if the interval excludes zero:

$$0\notin [\underline y,\overline y]\tag{22.8.46}$$

This form of computation turns rounding analysis into an explicit output. The result is no longer only a floating-point number. It is a value plus a certificate of uncertainty.

The cost of arbitrary precision means that it should be integrated with diagnostics. A well-designed high-precision workflow should report the selected precision, the precision-growth strategy, residuals or backward errors, agreement across precision levels, and any certification bounds. A minimal record may include

$$\mathcal P = (p_{\mathrm{initial}},p_{\mathrm{final}},u_{p_{\mathrm{final}}},\Delta_{p,p'},\eta,\text{status}) \tag{22.8.47}$$

where $\Delta_{p,p'}$ measures agreement across precisions and $\eta$ is a problem-specific residual or backward error. Without such diagnostics, arbitrary precision may provide false confidence: a computation can be carried out with many digits and still solve the wrong problem, use an unstable formulation, or fail to control discretization error.

This final section completes the infrastructure arc of the chapter. Section 22.2 introduced machine parameters and floating-point diagnostics, showing that numerical results must be interpreted relative to the arithmetic environment. Section 22.3 treated diagnostic output and lightweight plotting as instruments for observing convergence, instability, residual structure, and precision drift. Section 22.4 developed Gray codes as low-change traversals of discrete state spaces. Section 22.5 introduced checksums and cyclic redundancy checks as algebraic safeguards for finite data. Sections 22.6 and 22.7 presented Huffman coding, arithmetic coding, and range coding as exact compression mechanisms for compact representation. The present section showed how arbitrary precision expands the arithmetic environment when fixed precision is insufficient.

Together, these topics define computational infrastructure for trustworthy numerical execution. They do not replace the numerical algorithms developed in earlier chapters. Rather, they make those algorithms auditable, reproducible, diagnosable, compact, protected, and verifiable. A modern numerical computation is not complete when it returns a number. It is complete when the number is accompanied by enough arithmetic context, diagnostic evidence, data integrity, representation control, and precision analysis to make the result scientifically credible.

### Rust Implementation

Following the discussion in Section 22.8.4 on exact kernels, robust predicates, and certified numerical decisions, Program 22.8.4 provides a practical implementation of exact geometric predicates using integer-based arithmetic. In many numerical applications, the primary objective is not merely to compute a value accurately but to make a discrete decision correctly. Geometric algorithms, mesh generators, topological operations, and branch-dependent computations often rely on the sign of a determinant or predicate evaluation. In such situations, an incorrect sign can lead to invalid topology, inconsistent meshes, or entirely incorrect algorithmic execution paths. This program demonstrates how exact arithmetic can be used to certify orientation predicates for nearly collinear point configurations. By comparing ordinary floating-point evaluations with exact scaled-integer determinants and by constructing explicit sign certificates, the implementation illustrates how adaptive and exact kernels provide reliability beyond what conventional floating-point arithmetic can guarantee. The resulting framework serves as a concrete realization of the robust-predicate concepts developed in Equations (22.8.41)–(22.8.47).

At the core of the implementation is the `RationalPoint` structure, which represents geometric coordinates using exact scaled integers. Rather than storing coordinates directly as floating-point numbers, each coordinate is represented as an integer together with a common scaling factor. This approach eliminates rounding ambiguity in geometric predicates and provides a computational environment in which the determinant of Equation (22.8.42) can be evaluated exactly. By separating geometric representation from hardware floating-point arithmetic, the structure forms the foundation of an exact geometric kernel.

The methods `to_f64_pair` and `display` provide auxiliary functionality for converting exact coordinates into floating-point values and formatted output. The former is used to evaluate the standard floating-point orientation predicate for comparison purposes, while the latter generates human-readable coordinate representations. Together, these methods make it possible to compare approximate and exact geometric computations within a single framework.

The `pow10` function constructs powers of ten used to define the coordinate scaling factor. By representing coordinates as large integers divided by a common scale, the implementation can model geometric configurations with extremely small separations while preserving exact arithmetic. Such configurations are precisely the situations in which floating-point predicates become vulnerable to sign errors.

The `fixed_decimal_string` function converts scaled integers into decimal strings suitable for output. This routine allows exact determinant values and coordinate data to be displayed in a form that reflects their true geometric scale. Since many of the determinant values appearing in nearly collinear configurations are extremely small, accurate textual representation is important for interpreting the numerical results.

The `Orientation` enumeration encapsulates the three possible outcomes of the orientation test: clockwise, counterclockwise, and collinear. The methods `from_bigint` and `from_f64` translate determinant values into discrete geometric decisions. This design mirrors the conceptual interpretation of Equation (22.8.42), where the sign of the determinant determines the orientation class of the point set. By separating numerical evaluation from logical classification, the implementation highlights the distinction between arithmetic computation and geometric decision making.

The `orient2d_exact` function implements the exact orientation predicate corresponding to Equation (22.8.42). Using only integer arithmetic, the routine evaluates the determinant exactly and therefore produces a mathematically correct sign regardless of how small the determinant may be. This function represents the exact-kernel component of the implementation and serves as the authoritative geometric decision mechanism.

The `orient2d_f64` function evaluates the same determinant using ordinary floating-point arithmetic. This routine is included to illustrate the behavior of conventional numerical computation and to provide a direct comparison against the exact predicate. In many practical geometric algorithms, this floating-point computation is used initially because of its speed, with exact arithmetic reserved for situations in which the sign is uncertain.

The `determinant_to_decimal` function converts exact determinant values into scaled decimal form. This representation allows the magnitude of the determinant to be inspected directly and helps illustrate the degree of near-collinearity present in the geometric configuration. Small determinant magnitudes correspond to the situations in which robust predicates become most important.

The `decimal_interval_around_det` function provides a simple certification mechanism based on interval enclosures, reflecting the ideas developed in Equations (22.8.43)–(22.8.46). Given a determinant value and a radius defining uncertainty, the routine constructs lower and upper bounds and determines whether the resulting interval excludes zero. If zero lies outside the interval, the sign of the determinant is certified. This procedure demonstrates how numerical computation can be augmented with an explicit correctness certificate.

The `run_case` function evaluates a complete geometric test scenario. For each configuration of points, the routine computes both floating-point and exact determinants, classifies the orientation, constructs a certification interval, and reports whether the floating-point predicate agrees with the exact result. This encapsulation provides a reusable framework for investigating geometric robustness across a variety of test cases.

The `main` function orchestrates the entire demonstration. It constructs three representative geometric configurations: a nearly collinear counterclockwise configuration, a nearly collinear clockwise configuration, and an exactly collinear configuration. These cases are chosen because they stress the numerical reliability of orientation predicates. The function evaluates each case using both floating-point and exact arithmetic, reports the corresponding certificates, and concludes by summarizing the diagnostic information associated with the computation. In doing so, it illustrates how exact kernels and certification mechanisms can be integrated into a broader computational workflow, providing the type of diagnostic record envisioned by Equation (22.8.47).

Add the following dependencies to cargo.toml:

```rust
[dependencies]
num-bigint = "0.4"
num-traits = "0.2"
```

```rust
// Program 22.8.4. Exact Orientation Predicates with Integer Certification
//
// Problem statement:
// Demonstrate how exact kernels protect branch decisions in geometric
// computation. The program compares a naive f64 orientation predicate with an
// exact scaled-integer determinant for nearly collinear points. It then reports
// a simple certification interval for the determinant sign, illustrating the
// predicate logic described by Equations (22.8.41)--(22.8.47).

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

#[derive(Clone, Debug)]
struct RationalPoint {
    x: BigInt,
    y: BigInt,
    scale: BigInt,
}

impl RationalPoint {
    fn from_scaled(x_units: i64, y_units: i64, scale: &BigInt) -> Self {
        Self {
            x: BigInt::from(x_units),
            y: BigInt::from(y_units),
            scale: scale.clone(),
        }
    }

    fn to_f64_pair(&self) -> (f64, f64) {
        let scale_as_f64 = self.scale.to_string().parse::<f64>().unwrap();

        let x_as_f64 = self.x.to_string().parse::<f64>().unwrap() / scale_as_f64;
        let y_as_f64 = self.y.to_string().parse::<f64>().unwrap() / scale_as_f64;

        (x_as_f64, y_as_f64)
    }

    fn display(&self, digits: usize) -> String {
        format!(
            "({}, {})",
            fixed_decimal_string(&self.x, &self.scale, digits),
            fixed_decimal_string(&self.y, &self.scale, digits)
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Orientation {
    Clockwise,
    Collinear,
    Counterclockwise,
}

impl Orientation {
    fn from_bigint(det: &BigInt) -> Self {
        if det > &BigInt::zero() {
            Orientation::Counterclockwise
        } else if det < &BigInt::zero() {
            Orientation::Clockwise
        } else {
            Orientation::Collinear
        }
    }

    fn from_f64(det: f64) -> Self {
        if det > 0.0 {
            Orientation::Counterclockwise
        } else if det < 0.0 {
            Orientation::Clockwise
        } else {
            Orientation::Collinear
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Orientation::Clockwise => "clockwise",
            Orientation::Collinear => "collinear",
            Orientation::Counterclockwise => "counterclockwise",
        }
    }
}

fn pow10(n: usize) -> BigInt {
    let mut result = BigInt::one();

    for _ in 0..n {
        result *= 10;
    }

    result
}

fn fixed_decimal_string(value: &BigInt, scale: &BigInt, max_digits: usize) -> String {
    let negative = value.is_negative();
    let mut numerator = value.abs().to_string();
    let scale_digits = scale.to_string().len() - 1;

    if numerator.len() <= scale_digits {
        let zeros_needed = scale_digits + 1 - numerator.len();
        numerator = format!("{}{}", "0".repeat(zeros_needed), numerator);
    }

    let split = numerator.len() - scale_digits;
    let integer_part = &numerator[..split];
    let fractional_full = &numerator[split..];
    let keep = max_digits.min(fractional_full.len());
    let fractional_part = &fractional_full[..keep];

    if keep == 0 {
        if negative {
            format!("-{}", integer_part)
        } else {
            integer_part.to_string()
        }
    } else if negative {
        format!("-{}.{}", integer_part, fractional_part)
    } else {
        format!("{}.{}", integer_part, fractional_part)
    }
}

fn orient2d_exact(a: &RationalPoint, b: &RationalPoint, c: &RationalPoint) -> BigInt {
    let bx_ax = &b.x - &a.x;
    let by_ay = &b.y - &a.y;
    let cx_ax = &c.x - &a.x;
    let cy_ay = &c.y - &a.y;

    bx_ax * cy_ay - by_ay * cx_ax
}

fn orient2d_f64(a: &RationalPoint, b: &RationalPoint, c: &RationalPoint) -> f64 {
    let (ax, ay) = a.to_f64_pair();
    let (bx, by) = b.to_f64_pair();
    let (cx, cy) = c.to_f64_pair();

    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

fn determinant_to_decimal(det: &BigInt, scale: &BigInt, digits: usize) -> String {
    let det_scale = scale * scale;
    fixed_decimal_string(det, &det_scale, digits)
}

fn decimal_interval_around_det(
    det: &BigInt,
    radius_units: &BigInt,
    scale: &BigInt,
    digits: usize,
) -> (String, String, bool) {
    let lower = det - radius_units;
    let upper = det + radius_units;

    let excludes_zero = lower > BigInt::zero() || upper < BigInt::zero();

    (
        determinant_to_decimal(&lower, scale, digits),
        determinant_to_decimal(&upper, scale, digits),
        excludes_zero,
    )
}

fn run_case(name: &str, a: &RationalPoint, b: &RationalPoint, c: &RationalPoint) {
    let det_exact = orient2d_exact(a, b, c);
    let det_f64 = orient2d_f64(a, b, c);

    let exact_orientation = Orientation::from_bigint(&det_exact);
    let f64_orientation = Orientation::from_f64(det_f64);

    let radius_units = BigInt::zero();
    let (lower, upper, certified) =
        decimal_interval_around_det(&det_exact, &radius_units, &a.scale, 80);

    println!("{}", name);
    println!("{}", "-".repeat(name.len()));
    println!("p1                              = {}", a.display(40));
    println!("p2                              = {}", b.display(40));
    println!("p3                              = {}", c.display(40));
    println!(
        "f64 determinant                  = {:.18e}",
        det_f64
    );
    println!(
        "f64 orientation                  = {}",
        f64_orientation.as_str()
    );
    println!(
        "exact determinant                = {}",
        determinant_to_decimal(&det_exact, &a.scale, 80)
    );
    println!(
        "exact orientation                = {}",
        exact_orientation.as_str()
    );
    println!("certificate lower bound          = {}", lower);
    println!("certificate upper bound          = {}", upper);
    println!("certificate excludes zero        = {}", certified);
    println!(
        "f64 agrees with exact predicate  = {}",
        f64_orientation == exact_orientation
    );
    println!();
}

fn main() {
    let scale = pow10(40);

    let a = RationalPoint::from_scaled(0, 0, &scale);
    let b = RationalPoint::from_scaled(10_i64.pow(9), 10_i64.pow(9), &scale);

    let c_counterclockwise =
        RationalPoint::from_scaled(2 * 10_i64.pow(9), 2 * 10_i64.pow(9) + 1, &scale);

    let c_clockwise =
        RationalPoint::from_scaled(2 * 10_i64.pow(9), 2 * 10_i64.pow(9) - 1, &scale);

    let c_collinear =
        RationalPoint::from_scaled(2 * 10_i64.pow(9), 2 * 10_i64.pow(9), &scale);

    println!("Exact Kernels and Robust Orientation Predicates");
    println!("===============================================");
    println!();

    println!("Predicate");
    println!("---------");
    println!("orientation determinant = (x2 - x1)(y3 - y1) - (y2 - y1)(x3 - x1)");
    println!("coordinates are stored as exact scaled integers");
    println!("scale                   = 10^40");
    println!();

    run_case(
        "Case 1: Nearly Collinear Counterclockwise Points",
        &a,
        &b,
        &c_counterclockwise,
    );

    run_case(
        "Case 2: Nearly Collinear Clockwise Points",
        &a,
        &b,
        &c_clockwise,
    );

    run_case(
        "Case 3: Exactly Collinear Points",
        &a,
        &b,
        &c_collinear,
    );

    println!("Diagnostic Record");
    println!("-----------------");
    println!("initial predicate              = f64 orientation");
    println!("final predicate                = exact scaled-integer orientation");
    println!("certification rule             = accept sign when interval excludes zero");
    println!("selected precision             = exact integer arithmetic");
    println!("status                         = robust sign decisions obtained");
}
```

Program 22.8.4 demonstrates that some numerical computations require more than increased precision; they require correct discrete decisions. In geometric algorithms, the sign of a determinant often determines the execution path of the computation. An incorrect sign can produce invalid meshes, inconsistent topology, failed intersection tests, or incorrect combinatorial structures. Consequently, the accuracy of the numerical value is often less important than the correctness of the associated decision.

The example illustrates how exact arithmetic eliminates the ambiguity that arises in nearly degenerate geometric configurations. While floating-point arithmetic may correctly classify the test cases presented here, its reliability cannot be guaranteed for arbitrarily small determinants. The exact orientation predicate, by contrast, evaluates the determinant symbolically through integer arithmetic and therefore produces a mathematically correct sign regardless of the degree of near-collinearity.

The certification interval provides an additional layer of reliability. Rather than returning only a determinant value, the computation also reports a certificate indicating whether the sign is provably determined. This reflects the philosophy of interval and certified computation developed in Equations (22.8.43)–(22.8.46), where numerical results are accompanied by explicit information regarding their uncertainty.

More broadly, the program serves as a practical culmination of the chapter’s discussion of computational infrastructure. Exact predicates, certification mechanisms, adaptive precision, integrity diagnostics, and reproducibility tools all contribute to a common objective: ensuring that numerical computations are not only efficient but also trustworthy. Modern scientific computing increasingly demands this level of reliability, particularly in applications where a single incorrect branch decision can invalidate an entire computational pipeline. The techniques illustrated here therefore represent an essential component of robust numerical software engineering.

# 22.9. Conclusion

Throughout this chapter, we have examined the computational infrastructure that supports trustworthy numerical computing. Numerical algorithms do not operate in isolation; their reliability depends on machine arithmetic, diagnostic tools, data integrity mechanisms, reproducible workflows, and verification procedures. We explored floating-point diagnostics, structured logging, Gray-code enumeration, checksums, cyclic redundancy checks, lossless compression methods, and arbitrary-precision arithmetic. Together, these topics form an essential toolkit for developing numerical software that is not only efficient but also reliable, reproducible, and scientifically credible. By combining mathematical understanding with practical Rust implementations, this chapter has provided the foundations needed to build robust numerical systems capable of supporting large-scale scientific and engineering computations.

## 22.9.1. Key Takeaways

- Floating-point arithmetic is inherently approximate. Understanding machine constants, rounding behavior, cancellation effects, and exceptional values is essential for interpreting numerical results correctly.
- Diagnostic techniques provide visibility into numerical algorithms. Convergence traces, structured logs, metadata, and lightweight plotting help identify errors, verify assumptions, and improve reproducibility.
- Non-associativity means that mathematically equivalent computations may produce different numerical results. Reproducibility therefore requires careful attention to accumulation order, parallel execution, and precision management.
- Gray codes provide efficient enumeration strategies in which consecutive states differ by only one bit. These structures are useful in testing, optimization, search algorithms, and combinatorial computations.
- Checksums and cyclic redundancy checks help ensure data integrity throughout scientific workflows, protecting against corruption during storage, transmission, and processing.
- Lossless compression techniques preserve information exactly while reducing storage requirements. Huffman coding, arithmetic coding, and range coding represent different approaches to achieving efficient compression.
- Entropy provides a theoretical limit on compression performance and serves as an important link between information theory and practical coding algorithms.
- Arbitrary-precision arithmetic allows precision to be treated as a computational resource. High-precision calculations are valuable for verification, reference solutions, and the study of ill-conditioned problems.
- Exact kernels and robust predicates play a critical role in applications where numerical errors can lead to incorrect decisions or algorithmic failures.
- Rust provides strong support for building reliable numerical software through its emphasis on safety, performance, reproducibility, and correctness.

## 22.9.2. Advice for Beginners

- When beginning the study of numerical software infrastructure, remember that obtaining a numerical answer is only part of the problem. Equally important is understanding whether that answer can be trusted.
- Start by exploring machine arithmetic and floating-point behavior. Experiment with rounding, cancellation, overflow, underflow, and non-associativity to develop intuition about finite-precision computation.
- Learn to incorporate diagnostics into every numerical program. Structured logs, convergence traces, and simple plots often reveal issues that would otherwise remain hidden.
- Implement checksums and integrity tests when working with numerical datasets. Reliable data handling is just as important as reliable numerical algorithms.
- Study lossless compression methods not only as storage tools but also as examples of how information theory influences algorithm design and computational efficiency.
- Experiment with arbitrary-precision arithmetic and compare high-precision reference solutions with standard floating-point results. Such comparisons provide valuable insight into numerical error and conditioning.
- For Rust implementations, explore libraries that support arbitrary precision, structured data handling, and numerical diagnostics. Build small tools that emphasize verification and reproducibility before optimizing for performance.
- Most importantly, develop the habit of questioning numerical results. Reliable scientific computing depends as much on verification and validation as on algorithm implementation.

## 22.9.3. Further Learning with GenAI

To deepen your understanding of trustworthy numerical computing, consider exploring the following prompts:

 1. Explain machine epsilon, rounding error, and floating-point representation, and demonstrate their effects using Rust examples.
 2. Implement a diagnostic framework in Rust that records convergence histories, residuals, and performance metrics.
 3. Compare different accumulation strategies and analyze how non-associativity affects numerical reproducibility.
 4. Explain Gray codes and implement a Gray-code generator for combinatorial testing applications.
 5. Implement checksum and CRC algorithms in Rust and evaluate their effectiveness for detecting data corruption.
 6. Compare Huffman coding, arithmetic coding, and range coding in terms of compression efficiency and computational complexity.
 7. Explain the relationship between entropy and lossless compression and illustrate the concept using numerical datasets.
 8. Implement arbitrary-precision arithmetic for evaluating ill-conditioned expressions and compare the results with floating-point computations.
 9. Analyze how high-precision reference solutions can be used to verify numerical algorithms and identify hidden errors.
10. Design a complete reproducibility and verification pipeline for a scientific-computing application using diagnostics, integrity checks, compression, and arbitrary precision.

By exploring these prompts, readers can strengthen their understanding of the principles that underpin reliable numerical software.

## 22.9.4. Homework Exercises

To reinforce your understanding of the material covered in this chapter, complete the following exercises:

 1. Write a program that experimentally determines machine epsilon, overflow limits, underflow limits, and the smallest representable positive number.
 2. Investigate the effects of cancellation by evaluating mathematically equivalent expressions and comparing their numerical accuracy.
 3. Implement structured logging for a numerical algorithm and analyze how diagnostic information assists in debugging and verification.
 4. Generate Gray-code sequences for increasing numbers of bits and analyze their Hamming-distance properties.
 5. Implement checksum and CRC algorithms and evaluate their ability to detect different types of data corruption.
 6. Construct a Huffman coder and compare its compression performance with the entropy of the source distribution.
 7. Implement an arithmetic or range coder and compare its compression efficiency with Huffman coding.
 8. Use arbitrary-precision arithmetic to investigate a numerically ill-conditioned problem and compare the results with standard floating-point computations.
 9. Develop a verification framework that compares floating-point solutions against high-precision reference calculations.
10. Design a complete scientific-data pipeline that incorporates diagnostics, integrity checking, compression, and verification, and evaluate its effectiveness on a realistic numerical dataset.

Trustworthy numerical computing requires more than efficient algorithms. It demands careful attention to arithmetic, diagnostics, data integrity, reproducibility, and verification. The techniques presented in this chapter provide the foundation for building scientific software that can be trusted, validated, and maintained over long periods of use. As computational systems continue to grow in scale and complexity, these principles become increasingly important. By mastering them and applying them through Rust implementations, readers will be well prepared to develop reliable numerical software for modern scientific and engineering applications.

# References

 1. An, F. *et al.* (2025) ‘Data transmission error detection and correction with cyclic redundancy check and polar code integration with successive cancellation decoding algorithm’, *Applied Sciences*, 15(3), 1124.
 2. Antunes, B. and Hill, D.R.C. (2024) ‘Reproducibility, replicability and repeatability: a survey of reproducible research with a focus on high performance computing’, *Computer Science Review*, 53, 100655.
 3. Auli-Llinas, F. (2023) ‘Fast and efficient entropy coding architectures for massive data compression’, *Technologies*, 11(5), 132.
 4. Azami, N., Fallin, A. and Burtscher, M. (2025) ‘Efficient lossless compression of scientific floating-point data on CPUs and GPUs’, in *Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems*, Volume 1, pp. 395–409.
 5. Boldo, S., Jeannerod, C.-P., Melquiond, G. and Muller, J.-M. (2023) ‘Floating-point arithmetic’, *Acta Numerica*, 32, pp. 203–290.
 6. Choi, Y.R. *et al.* (2025) ‘Secure delivery method for preserving data integrity of a logger for autonomous driving’, *Applied Sciences*, 15(7), 3533.
 7. Dondl, P., Striet, L. and Straughan, B. (2025) ‘Arbitrary precision computation of hydrodynamic stability eigenvalues’, *Proceedings of the Royal Society A*, 481, 20250375.
 8. Kashi, A., Lu, H., Brewer, W., Rogers, D., Matheson, M., Shankar, M. and Wang, F. (2026) ‘Mixed-precision numerics in scientific applications: survey and perspectives’, *The Journal of Supercomputing*.
 9. Kunze, J., Severo, D., Zani, G., van de Meent, J.-W. and Townsend, J. (2024) ‘Entropy coding of unordered data structures’, in *International Conference on Learning Representations*.
10. Leitold, L., Alrwashdeh, M. and Kollár, Z. (2025) ‘High-performance multi-precision tool for floating-point computations’, *Radioengineering*, 34(4), pp. 583–590.
11. Lévy, B. (2025) ‘Exact predicates, exact constructions and combinatorics for mesh CSG’, *ACM Transactions on Graphics*, 44(5), Article 167.
12. Li, W. *et al.* (2024) ‘Assessing residual diagnostics with the lineup protocol’, *Journal of Computational and Graphical Statistics*.
13. Liang, Z., Niu, K., Xu, J. and Zhang, P. (2025) ‘Semantic arithmetic coding using synonymous mappings’, *Entropy*, 27(4), 429.
14. Liu, B., Wong, D., Lam, C.-T. and Im, M. (2025) ‘Recursive and iterative approaches to generate rotation Gray codes for stamp foldings and semi-meanders’, *Theoretical Computer Science*, 1031, 115053.
15. Lyu, F. *et al.* (2025) ‘An effective lossless compression method for attitude data while drilling based on adaptive frame prediction Huffman coding’, *Scientific Reports*.
16. Miao, D., Laguna, I. and Rubio-González, C. (2025) ‘FloatGuard: efficient whole-program detection of floating-point exceptions in HIP programs running on AMD GPUs’, in *Proceedings of HPDC 2025*.
17. Miller, G. and Spiegel, E. (2025) ‘Guidelines for Research Data Integrity (GRDI)’, *Scientific Data*, 12(1), 95.
18. Mütze, T. (2023) ‘Combinatorial Gray codes: an updated survey’, *The Electronic Journal of Combinatorics*, 30(3), DS26.
19. Pilaud, V. and Williams, A. (2025) ‘Skipping ropes: an efficient Gray code algorithm for generating wiggly permutations’, in *19th International Symposium on Algorithms and Data Structures*, Article 46.
20. Romero-Organvidez, D., Horcas, J.-M., Galindo, J.A. and Benavides, D. (2024) ‘Data visualization guidance using a software product line approach’, *Journal of Systems and Software*, 213, 112029.
21. Shanmugavelu, S., Taillefumier, M., Culver, C., Hernandez, O., Coletti, M. and Sedova, A. (2024) ‘Impacts of floating-point non-associativity on reproducibility for HPC and deep learning applications’, in *SC24-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis*, pp. 170–179.
22. Stoudt, S., Jernite, Y., Marshall, B., Marwick, B., Sharan, M., Whitaker, K. and Danchev, V. (2024) ‘Ten simple rules for building and maintaining a responsible data science workflow’, *PLoS Computational Biology*, 20(7), e1012232.
23. Tirpankar, T., Lund, C. and Gopalakrishnan, G. (2025) ‘CIRE: LLVM analysis for floating-point rounding error affected by precision and optimizations’, in *Proceedings of SC 2025*.
24. Vretinaris, S. (2026) ‘FastTanhSinhQuadrature.jl: high-performance Tanh-Sinh numerical integration in Julia’, *Journal of Open Source Software*, 11(120), 10076.
25. Wiseman, Y. (2025) ‘High-speed architecture for hybrid arithmetic–Huffman data compression’, *Technologies*, 13(12), 585.
26. Xie, P., Gao, Y., Wang, Y. and Xue, J. (2025) ‘Revealing floating-point accumulation orders in software/hardware implementations’, in *Proceedings of the 2025 USENIX Annual Technical Conference*.
27. Zhang, D.K. and Aiken, A. (2025) ‘High-performance branch-free algorithms for extended-precision floating-point arithmetic’, in *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*.
28. Zhang, L. *et al.* (2024) ‘An efficient parallel CRC computing method for high bandwidth networks and FPGA implementation’, *Electronics*, 13(22), 4399.
